"""Local-only browser interface for station setup, datum registration, and path export.

`scanplan web` serves `index.html`, `app.css` and `app.js` from this package and
answers their JSON requests with the planner's own API; nothing is uploaded
anywhere. The page is packaged with the service (unlike a repo-root asset), so an
installed `scanplan web` finds it.
"""

from __future__ import annotations

import base64
import csv
import html
import json
import re
import shutil
import socket
import sqlite3
import threading
import time
import traceback
import uuid
import zlib
from dataclasses import replace
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import numpy as np
import trimesh

from scanplan.config import load_config, override
from scanplan.kinematics import KinematicGate
from scanplan.pipeline import plan, write_outputs
from scanplan.registration import RigidRegistration, fit_points
from scanplan.report import (
    Viewpoint,
    write_report,
    write_samples_csv,
    write_viewpoints_csv,
    write_viewpoints_json,
)
from scanplan.roboguide.adapt import export as export_ls
from scanplan.roboguide.wpr import normalise_prog_name
from scanplan.scene import StationDescription, load_station
from scanview import load_target_samples, render_run, save_target_samples

from .cad_import import mesh_from_bytes, to_stl_bytes

_WEB_ROOT = Path(__file__).resolve().parent
_MAX_STL_BYTES = 120 * 1024 * 1024
_MAX_REQUEST_BYTES = 380 * 1024 * 1024
_SAFE_RUN_ID = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_SAFE_ARTIFACT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,96}$")

# --- progress board --------------------------------------------------------
#
# A planning run is one synchronous request, so the page cannot be told anything until
# the whole thing finishes. The job therefore publishes (stage, percent) into this
# in-process board and the page polls `GET /api/progress`. It is deliberately tiny and
# lossy: entries expire, and the page falls back to elapsed time if they stop coming.

_PROGRESS_TTL_S = 900.0
_PROGRESS: dict[str, dict] = {}
_PROGRESS_LOCK = threading.Lock()


def _publish_progress(token: str | None, stage: str, fraction: float,
                      detail: str = "") -> None:
    """Record one stage update for the page to poll; a no-op without a token."""
    if not token:
        return
    pct = round(100.0 * min(max(float(fraction), 0.0), 1.0), 1)
    now = time.time()
    with _PROGRESS_LOCK:
        for key, entry in list(_PROGRESS.items()):
            if now - entry["updated"] > _PROGRESS_TTL_S:
                _PROGRESS.pop(key, None)
        previous = _PROGRESS.get(token)
        started = previous["started"] if previous else now
        if previous is not None:
            # Never regress: a stage that reports late must not rewind the bar.
            pct = max(pct, previous["pct"])
        _PROGRESS[token] = {
            "stage": stage,
            "pct": pct,
            "detail": detail,
            "started": started,
            "updated": now,
        }


def _progress_snapshot(token: str) -> dict | None:
    now = time.time()
    with _PROGRESS_LOCK:
        entry = _PROGRESS.get(token)
        if entry is None:
            return None
        snapshot = dict(entry)
    snapshot.pop("started", None)
    snapshot["age_s"] = round(now - snapshot["updated"], 2)
    snapshot["elapsed_s"] = round(now - entry["started"], 1)
    snapshot.pop("updated", None)
    return snapshot


def _progress_token(data: dict) -> str | None:
    """The client-chosen id this job publishes under, or None when it did not ask."""
    token = data.get("progress_id")
    if isinstance(token, str) and _SAFE_RUN_ID.fullmatch(token):
        return token
    return None


def _progress_writer(token: str | None):
    def write(stage: str, fraction: float, detail: str = "") -> None:
        _publish_progress(token, stage, fraction, detail)

    return write


def _vector(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain exactly three finite numbers")
    return result


def _pose_matrix(pose: dict, name: str) -> np.ndarray:
    from scanplan.roboguide.wpr import wpr_to_rot

    position = _vector(pose.get("xyz"), f"{name} reset XYZ")
    wpr = _vector(pose.get("wpr"), f"{name} reset WPR")
    matrix = np.eye(4)
    matrix[:3, :3] = wpr_to_rot(*wpr)
    matrix[:3, 3] = position
    return matrix


def _registration_from_payload(data: dict) -> tuple[RigidRegistration, list[np.ndarray]]:
    cad = np.asarray(data.get("cad_points_vehicle_mm"), dtype=float)
    measured = np.asarray(data.get("measured_points_mm"), dtype=float)
    if cad.ndim != 2 or cad.shape != measured.shape or cad.shape[1:] != (3,) \
            or cad.shape[0] not in (3, 4):
        raise ValueError(
            "pick three CAD points (or four) and enter the same number of matching "
            "measured XYZ points"
        )
    registration = fit_points(
        cad,
        measured,
        target_frame=str(data.get("measured_frame", "robot_uframe_1")),
        uframe=int(data.get("uframe", 1)),
    )
    reset = data.get("reset_poses", {})
    if not isinstance(reset, dict):
        raise ValueError("reset_poses must be an object when provided")
    identity_pose = {"xyz": [0.0, 0.0, 0.0], "wpr": [0.0, 0.0, 0.0]}
    matrices = [
        _pose_matrix(reset.get(key, identity_pose), key)
        for key in ("base", "support", "part")
    ]
    return registration, matrices


def _decode_stl(data: dict, name: str) -> trimesh.Trimesh:
    return _load_stl(_decode_stl_bytes(data, name), name)


def _load_stl(raw: bytes, name: str) -> trimesh.Trimesh:
    loaded = trimesh.load(file_obj=__import__("io").BytesIO(raw), file_type="stl")
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise ValueError(f"{name} STL contains no triangles")
    if not np.isfinite(loaded.vertices).all():
        raise ValueError(f"{name} STL contains non-finite coordinates")
    return loaded


def _decode_stl_bytes(data: dict, name: str) -> bytes:
    encoded = data.get("stl_base64")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError(f"select an STL file for {name}")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except ValueError as error:
        raise ValueError(f"{name} STL data is not valid base64") from error
    if len(raw) > _MAX_STL_BYTES:
        raise ValueError(f"{name} STL exceeds the {_MAX_STL_BYTES // (1024 * 1024)} MB limit")
    return raw


def _sync_input_part(raw: bytes, input_dir: Path) -> Path:
    part_path = input_dir / "part.stl"
    input_dir.mkdir(parents=True, exist_ok=True)
    staged_path = input_dir / f".part-{uuid.uuid4().hex}.tmp"
    try:
        staged_path.write_bytes(raw)
        staged_path.replace(part_path)
    finally:
        staged_path.unlink(missing_ok=True)
    return part_path


def _reference_base(part: trimesh.Trimesh, fixture: trimesh.Trimesh,
                    scan_axis) -> trimesh.Trimesh:
    """A flat reference plate placed BEHIND the main scan face.

    The planner traces the base as an obstacle, so it cannot simply be omitted. Where the
    plate goes is not cosmetic either, and the way it used to be built got it wrong on a
    real part: it was laid under the models in the VEHICLE frame, which assumes +Z is "down"
    there. A vertically clamped panel is exactly the case where that is false - on the
    2026-10-29 station the main scan face axis is +Y, so the plate was built below in Z,
    ended up in FRONT of the panel once the registration was applied, and the coverage
    report then described the plate's own shadow.

    The plate is therefore built in the frame of the SCAN AXIS: square to it, spanning the
    part's silhouette, with its mounting surface just behind the lowest point of part and
    fixture along that axis. Whatever "down" means in the customer's CAD, the plate can then
    never stand between the head and the face being scanned.
    """
    axis = np.asarray(scan_axis, dtype=float).reshape(3)
    length = float(np.linalg.norm(axis))
    if not np.isfinite(length) or length <= 1e-9:
        raise ValueError("the main scan-face axis has no direction")
    axis = axis / length
    helper = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    first = np.cross(helper, axis)
    first = first / float(np.linalg.norm(first))
    second = np.cross(axis, first)

    points = np.vstack((np.asarray(part.vertices, dtype=float),
                        np.asarray(fixture.vertices, dtype=float)))
    along = points @ axis
    low = float(along.min())
    span = max(float(along.max()) - low, 1.0)
    across_first = points @ first
    across_second = points @ second
    extent_first = float(np.ptp(across_first)) + 120.0
    extent_second = float(np.ptp(across_second)) + 120.0
    thickness = max(20.0, span * 0.04)

    plate = trimesh.creation.box(extents=[extent_first, extent_second, thickness])
    frame = np.eye(4)
    frame[:3, :3] = np.stack([first, second, axis], axis=1)
    plate.apply_transform(frame)
    plate.apply_translation(
        first * float((across_first.min() + across_first.max()) * 0.5)
        + second * float((across_second.min() + across_second.max()) * 0.5)
        + axis * (low - 2.0 - thickness * 0.5)
    )
    return plate


def _default_base(part: trimesh.Trimesh, fixture: trimesh.Trimesh, scan_axis,
                  input_dir: Path | None = None) -> tuple[trimesh.Trimesh, str]:
    """The base to use when none was uploaded, and where it came from.

    `input/base.stl` first: it is the station's own base, so tracing it is more honest than
    tracing a plate - the clamp, the part's shadow and the head's standoff all depend on
    what is really there. It is used exactly as it is, in the uploaded (vehicle) frame,
    because a real base's own coordinates are the assembly's coordinates.

    A reference plate behind the scan face is the fallback for a run that has no base file
    at all - a fresh clone, or a station whose base was never exported. Returning the source
    string keeps "which base was traced" answerable from the report instead of guessable
    from the viewer.
    """
    folder = Path.cwd() / "input" if input_dir is None else Path(input_dir)
    candidate = folder / "base.stl"
    if candidate.exists():
        try:
            base = _load_stl(candidate.read_bytes(), "input/base.stl")
        except ValueError:
            # A truncated or empty file must not stop a run: the plate is still a valid
            # (if conservative) obstacle, and the report says which one was used.
            base = None
        if base is not None and _behind_the_scan_face(base, part, fixture, scan_axis):
            return base, "input/base.stl"
        if base is not None:
            # A base that sits entirely IN FRONT of the main scan face cannot be this
            # station's base - it was exported in some other frame - and tracing it would
            # put a wall between the head and the part. That is the exact failure the
            # reference plate exists to avoid, so the plate wins here.
            return (_reference_base(part, fixture, scan_axis),
                    "reference plate behind the scan face "
                    "(input/base.stl is in front of the scan face)")
    return _reference_base(part, fixture, scan_axis), "reference plate behind the scan face"


def _behind_the_scan_face(base: trimesh.Trimesh, part: trimesh.Trimesh,
                          fixture: trimesh.Trimesh, scan_axis) -> bool:
    """Whether `base` is on the far side of the main scan face, or at least not in front.

    The head works in the +axis half-space, so a body whose every point is beyond the
    assembly along +axis is somewhere else entirely and would shade the whole face. A real
    base sits under or behind the fixture, i.e. its projection overlaps the assembly's.
    """
    axis = np.asarray(scan_axis, dtype=float).reshape(3)
    length = float(np.linalg.norm(axis))
    if not np.isfinite(length) or length <= 1e-9:
        return True
    axis = axis / length
    assembly = np.vstack((np.asarray(part.vertices, dtype=float),
                          np.asarray(fixture.vertices, dtype=float)))
    return float((np.asarray(base.vertices, dtype=float) @ axis).min()) <= \
        float((assembly @ axis).max()) + 1.0


def _synthetic_base(part: trimesh.Trimesh, fixture: trimesh.Trimesh) -> trimesh.Trimesh:
    """The reference plate with the default (vehicle +Z) scan axis - kept for callers.

    `_reference_base` is the real implementation; this name stays because a plate "under the
    models" is what a caller with no scan axis in hand means.
    """
    return _reference_base(part, fixture, np.array([0.0, 0.0, 1.0]))


def _vehicle_point(matrix: np.ndarray, point) -> list[float]:
    """Map a point from the registered measured frame back to the uploaded STL frame."""
    inverse = np.linalg.inv(matrix)
    moved = inverse[:3, :3] @ np.asarray(point, dtype=float) + inverse[:3, 3]
    return [round(float(value), 4) for value in moved]


def _vehicle_axis(matrix: np.ndarray, axis) -> list[float]:
    inverse = np.linalg.inv(matrix)
    moved = inverse[:3, :3] @ np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(moved))
    if norm < 1e-12:
        return [0.0, 0.0, 0.0]
    return [round(float(value / norm), 9) for value in moved]


def _write_planning_report(path: Path, *, run_id: str, prog_name: str,
                           report: dict, registration: RigidRegistration,
                           face_axis: np.ndarray, downloads: list[str]) -> Path:
    """A human-readable planning report next to the machine-readable JSON."""
    pair_rows = []
    for index in range(len(registration.cad_points)):
        cad = ", ".join(f"{value:.3f}" for value in registration.cad_points[index])
        measured = ", ".join(
            f"{value:.3f}" for value in registration.measured_points[index]
        )
        error = float(np.linalg.norm(registration.residuals_mm[index]))
        pair_rows.append(
            f"<tr><td>P{index + 1}</td><td>{cad}</td>"
            f"<td>{measured}</td><td>{error:.3f}</td></tr>"
        )
    pairs = "".join(pair_rows)
    uncovered = report.get("uncovered_by_reason", {})
    reason_text = "、".join(f"{key} {value}" for key, value in uncovered.items()) or "无"
    file_rows = "".join(f"<li>{html.escape(name)}</li>" for name in downloads)
    face_text = ", ".join(f"{value:.4f}" for value in face_axis)
    coverage = float(report.get("coverage_pct", 0))
    target_area = float(report.get("target_face_mm2", 0)) / 1e6
    body = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>扫描规划报告 · {html.escape(prog_name)}</title>
<style>
body{{
  margin:0;background:#f3f6fa;color:#1c2c40;
  font:14px/1.7 "Segoe UI","Microsoft YaHei",sans-serif;
}}
main{{max-width:980px;margin:32px auto;padding:0 22px 48px}}
header{{
  padding:24px 28px;border-radius:10px;
  background:linear-gradient(120deg,#10243d,#1d4a74);color:#fff;
}}
header h1{{margin:0 0 6px;font-size:24px}}
header p{{margin:0;color:#b9d2e8}}
section{{
  margin-top:16px;padding:20px 22px;border:1px solid #e5eaf0;
  border-radius:8px;background:#fff;
}}
h2{{margin:0 0 14px;font-size:17px}}
.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}}
.metric{{padding:14px;border:1px solid #e9edf2;border-radius:6px;background:#f8fafc}}
.metric span{{display:block;color:#78879a;font-size:12px}}
.metric b{{display:block;margin-top:7px;color:#2f6896;font:600 24px Consolas,monospace}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{padding:7px 8px;border-bottom:1px solid #edf0f4;text-align:left}}
th{{color:#78879a;font-weight:600}}
.warn{{
  padding:11px 13px;border:1px solid #eee6d8;border-radius:5px;
  background:#fcf8f1;color:#8d7957;
}}
code{{color:#315c82}}
</style></head><body><main>
<header><h1>扫描路径规划报告</h1>
<p>运行编号 {html.escape(run_id)} · 程序 {html.escape(prog_name)}</p></header>
<section><h2>主要结果</h2><div class="metrics">
<div class="metric"><span>目标面覆盖率</span><b>{coverage:.2f}%</b></div>
<div class="metric"><span>扫描视点</span><b>{int(report.get('viewpoints', 0))}</b></div>
<div class="metric"><span>目标采样点</span><b>{int(report.get('target_points', 0))}</b></div>
<div class="metric"><span>漏扫点</span><b>{int(report.get('uncovered_points', 0))}</b></div>
</div>
<p>目标面：<b>{html.escape(str(report.get('face', '—')))}</b> ·
面积 {target_area:.4f} m² · 漏扫原因：{html.escape(reason_text)}</p></section>
<section><h2>三点/四点配准</h2>
<p>现场坐标系：<b>{html.escape(registration.frame_label)}</b> ·
RMS {registration.rms_error_mm:.3f} mm ·
最大误差 {registration.max_error_mm:.3f} mm</p>
<table><thead><tr><th>点</th><th>CAD / 整车坐标 (mm)</th>
<th>现场测量坐标 (mm)</th><th>误差 (mm)</th></tr></thead>
<tbody>{pairs}</tbody></table>
<p>扫描面外法向（整车坐标）：<code>{face_text}</code></p></section>
<section><h2>交付文件</h2><ul>{file_rows}</ul></section>
<section><h2>使用限制</h2>
<div class="warn">本报告只描述几何可达性，不包含机器人逆解、关节限位、
机械臂碰撞和跟踪器被机械臂遮挡的判断。导入 ROBOGUIDE 后必须复核 UFRAME、
UTOOL、可达性与碰撞。</div></section>
</main></body></html>"""
    path.write_text(body, encoding="utf-8")
    return path


def _available_run_id(requested: str, root: Path) -> str:
    """Keep a requested run name when free, otherwise append a timestamp.

    A rerun with the same run name used to fail with "folder already exists",
    which surfaced as a planning error even though the inputs were fine.  The
    run name is a label, not a uniqueness constraint: reuse it when the folder is
    empty, otherwise make a new folder that still starts with the user's name.
    """
    folder = root / requested
    if not folder.exists() or not any(folder.iterdir()):
        return requested
    stem = requested[:40].rstrip("_-") or "run"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = f"{stem}_{stamp}"
    index = 2
    while (root / candidate).exists() and any((root / candidate).iterdir()):
        candidate = f"{stem}_{stamp}_{index}"
        index += 1
    return candidate


def _download_entries(prog_name: str) -> list[dict[str, str]]:
    return [
        {"name": f"{prog_name}.LS", "label": "FANUC 控制程序 (.LS)"},
        {"name": f"{prog_name}_points.csv", "label": "程序点位 CSV"},
        {"name": f"{prog_name}_frame.json", "label": "帧契约 JSON"},
        {"name": "planning_report.html", "label": "扫描规划报告 (HTML)"},
        {"name": "coverage_report.json", "label": "覆盖报告 JSON"},
        {"name": "registration_report.json", "label": "配准报告 JSON"},
        {"name": "summary.txt", "label": "规划摘要 TXT"},
        {"name": "viewpoints.csv", "label": "视点数据 CSV"},
        {"name": "unreachable_points.csv", "label": "漏扫点 CSV"},
        {"name": "exempt_points.csv", "label": "夹具豁免点 CSV"},
    ]


def _public_config() -> dict:
    """What the page is allowed to know about the shipped configuration, as defaults.

    Only settings a page control can express, and the mode in the page's own vocabulary
    (`stroke` / `pose`) rather than the config's (`path` / `lean`). Everything here is a
    default: the page can still ask for something else, and the run reports what it used.
    """
    config = load_config()
    return {
        "delivery_mode": "pose" if config.target.delivery_mode == "lean" else "stroke",
        "coverage_target_pct": config.target.coverage_target_pct,
        # The second acceptance figure: a page that shows the coverage target must show
        # this one too, or "97 %" reads as if the walls were included.
        "feature_coverage_pct": config.target.feature_coverage_pct,
        "surface_bias": float(config.target.surface_bias),
        "max_points": config.target.max_points,
        "pool_mode": config.stroke.pool_mode,
        "robot_enabled": bool(config.robot.enabled),
        "motion": config.motion.type,
        "termination": config.motion.termination,
        "speed_mm_s": float(config.motion.speed_mm_s),
        "joint_speed": config.motion.joint_speed,
        "config_source": config.source,
        "parameter_sources": dict(config.parameter_sources),
    }


def _delivery_config(data: dict):
    """The configuration for one job: the shipped file plus the page's delivery mode.

    The two modes differ in the PLANNING unit, not only in the export: "pose" is the v1
    deliverable of isolated stops, so it switches the stroke stage off. Re-sorting the
    export order would not make a stroke plan pose-only.
    """
    config = load_config(data.get("config_path"))
    mode = str(data.get("delivery_mode") or "stroke").strip().lower()
    if mode not in ("stroke", "pose"):
        raise ValueError("delivery mode must be 'stroke' or 'pose'")
    if mode == "pose":
        return override(config, stroke={"enabled": False})
    return config


def _export_options(config, viewpoints, data: dict) -> dict:
    """Motion, termination and ordering for one deliverable.

    A stroke plan's order IS the sweep: re-sorting it would split each stroke into two
    unrelated moves. Its motion and termination come from the configuration for the same
    reason - a sweep flown as J/FINE is not the sweep the coverage figure was computed for.
    The pose-only deliverable keeps the permissive J move a reachability review is read with.
    """
    if any(point.stroke_id is not None for point in viewpoints):
        return {
            "motion": config.motion.type,
            "speed": f"{config.motion.speed_mm_s:g} mm/sec",
            "termination": config.motion.termination,
            "order": False,
            # The J moves between sweeps are the ones that turn the wrist; their speed is
            # its own setting (`motion.joint_speed`, 5 % by default).
            "joint_speed": config.motion.joint_speed,
        }
    return {
        "motion": str(data.get("motion", "J")),
        "speed": str(data.get("speed", "100%")),
        "termination": "FINE",
        "order": True,
    }


def solve_job(data: dict, progress=None) -> dict:
    """Register uploaded meshes, plan the selected side, and export FANUC files.

    `progress(stage, fraction, detail)` is optional; the page polls a snapshot of what it
    last reported while this call is still running.
    """
    started = time.perf_counter()
    if not isinstance(data, dict):
        raise ValueError("request body must be a JSON object")
    announce = progress or (lambda stage, fraction, detail="": None)
    registration, resets = _registration_from_payload(data)
    tolerance = float(data.get("max_registration_error_mm", 3.0))
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("registration error limit must be a positive finite number")
    if registration.max_error_mm > tolerance:
        raise ValueError(
            f"three-point registration max error is {registration.max_error_mm:.3f} mm; "
            f"the allowed limit is {tolerance:.3f} mm"
        )

    models = data.get("models", {})
    if not isinstance(models, dict):
        raise ValueError("models must be an object")
    announce("registration", 0.04, "reading the station meshes")
    support = _decode_stl(models.get("support", {}), "support/fixture")
    part_raw = _decode_stl_bytes(models.get("part", {}), "part")
    part = _load_stl(part_raw, "part")
    base_payload = models.get("base") or {}
    base_supplied = isinstance(base_payload, dict) and bool(base_payload.get("stl_base64"))
    # The scan axis is read before the base because the fallback base is PLACED relative to
    # it: a reference plate has to sit behind the face being scanned, whichever vehicle axis
    # that face happens to look along (see `_reference_base`).
    face_axis_vehicle = _vector(data.get("scan_face_axis"), "main scan-face axis")
    if not np.isclose(np.linalg.norm(face_axis_vehicle), 1.0, atol=1e-6):
        raise ValueError("main scan-face axis must be one of the six unit vehicle axes")
    if base_supplied:
        base, base_source = _decode_stl(base_payload, "base"), "uploaded"
    else:
        base, base_source = _default_base(part, support, face_axis_vehicle)
    base_reset, support_reset, part_reset = resets

    try:
        prog_name = normalise_prog_name(data.get("prog_name") or "PART01")
    except ValueError as error:
        raise ValueError(str(error)) from error

    axis_options = np.vstack((np.eye(3), -np.eye(3)))
    if not np.any(np.all(np.isclose(axis_options, face_axis_vehicle, atol=1e-9), axis=1)):
        raise ValueError("main scan-face axis must be exactly +X/-X/+Y/-Y/+Z/-Z")

    combined_part = np.eye(4)
    combined_part[:3, :3] = registration.rotation
    combined_part[:3, 3] = registration.translation
    combined_part = combined_part @ part_reset
    combined_support = np.eye(4)
    combined_support[:3, :3] = registration.rotation
    combined_support[:3, 3] = registration.translation
    combined_support = combined_support @ support_reset
    combined_base = np.eye(4)
    combined_base[:3, :3] = registration.rotation
    combined_base[:3, 3] = registration.translation
    combined_base = combined_base @ base_reset
    scan_axis_target = registration.rotation @ face_axis_vehicle

    requested_run_id = str(data.get("run_id") or (
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    ))
    if not _SAFE_RUN_ID.fullmatch(requested_run_id):
        raise ValueError("run id may contain only letters, digits, underscores, and hyphens")
    runs_root = (Path.cwd() / "output" / "web_runs").resolve()
    runs_root.mkdir(parents=True, exist_ok=True)
    run_id = _available_run_id(requested_run_id, runs_root)
    output = (runs_root / run_id).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _sync_input_part(part_raw, Path.cwd() / "input")

    base_path = output / "base.stl"
    base_vehicle_path = output / "base_vehicle.stl"
    support_path = output / "support.stl"
    part_path = output / "part.stl"
    # Keep the base in uploaded (vehicle) coordinates as well: the planner traces the
    # transformed copy, but a re-localisation registers again and must not transform it
    # twice. Older runs can still be inverted back through their registration report.
    base.export(base_vehicle_path)
    base.apply_transform(combined_base)
    base.export(base_path)
    support.export(support_path)
    part_path.write_bytes(part_raw)
    station_path = output / "station.json"
    station_path.write_text(json.dumps({
        "part": "part.stl",
        "fixture": "support.stl",
        "base": "base.stl",
        "transform_part": combined_part.tolist(),
        "transform_fixture": combined_support.tolist(),
        "note": (
            "Generated by scanplan web UI. STL reset poses are in vehicle coordinates; "
            "the three datum pairs register vehicle coordinates into the selected "
            "measured robot frame."
        ),
    }, indent=2) + "\n", encoding="utf-8")

    frame_num = registration.uframe
    if frame_num == 0:
        output_uframe = 0
    else:
        output_uframe = frame_num

    try:
        config = _delivery_config(data)
        announce("registration", 0.08, "station written")
        description = StationDescription(
            part=part_path,
            fixture=support_path,
            base=base_path,
            transform_part=tuple(tuple(float(v) for v in row) for row in combined_part),
            transform_fixture=tuple(
                tuple(float(v) for v in row) for row in combined_support
            ),
            source=station_path,
        )
        station = load_station(description, cloud_seed=config.seed)
        announce("segmentation", 0.10, "station loaded")
        axis_in_measured = registration.rotation @ face_axis_vehicle
        # The blank split is NOT built here. Handing `plan()` a precomputed one looked
        # harmless - same mesh, same parameters - but the split seeds the sign of the local
        # sheet normal from a single through-thickness axis, so a different axis reassigns
        # whole regions between "upper" and "lower". On station 20261028_193014 the exact
        # measured axis gave 53.8% coverage while the axis `plan()` derives itself gave
        # 99.0% on the SAME station file: the run was planning a face made of the wrong
        # regions, and the delivery stage then dropped eight strokes it could not reach.
        # One source of truth for the split, so the page and the CLI cannot disagree.
        #
        # Which SIDE is planned is decided by the axis the operator picked, not by the
        # split's "upper"/"lower" labels: those come from an arbitrary sign (see
        # `pipeline.facing_face`). Hard-coding "upper" here planned a 113 cm2 spray of
        # back-facing facets on station 20261029_225454 while the 1.15 m2 face the
        # operator had clicked - the one drawn in the "main scan face" colour - was never
        # sampled.
        result = plan(station, config, face_axis=axis_in_measured,
                      progress=lambda stage, fraction, detail="": announce(
                          stage, 0.10 + 0.80 * fraction, detail))
        announce("report", 0.92, "planning finished")
        result.report["installation_registration"] = {
            "measured_frame": registration.frame_label,
            "uframe": registration.uframe,
            "rms_error_mm": registration.rms_error_mm,
            "max_error_mm": registration.max_error_mm,
            "cad_face_axis_vehicle": face_axis_vehicle.tolist(),
            "scan_face_axis_measured_frame": axis_in_measured.tolist(),
            "target_face": str(result.report.get("face", "")),
        }
        write_outputs(result, output)
        announce("export", 0.93, "coverage and viewer written")
        options = _export_options(config, result.viewpoints, data)
        stroke_plan = not options["order"]
        contract = export_ls(
            result.viewpoints,
            output,
            prog_name=prog_name,
            uframe=output_uframe,
            utool=int(data.get("utool", 1)),
            robot=config.robot,
            **options,
        )
        announce("export", 0.98, "program exported")
        contract["frame"]["axes"] = (
            f"Robot UFRAME {output_uframe} axes; CAD viewpoints were registered "
            "from the vehicle reset frame using three field datum points."
        )
        contract["frame"]["note"] = (
            "See registration_report.json for reset poses, point pairs, rigid transform "
            "and residuals. Verify UFRAME and TCP in ROBOGUIDE before motion."
        )
        contract["uframe_utool"]["uframe_note"] = (
            f"Teach/verify UFRAME {output_uframe} matches the field coordinate frame "
            "used for the three measured datum points."
        )
        contract["placement"]["registration"] = {
            "method": "three corresponding non-collinear datum points; rigid transform",
            "measured_frame": registration.frame_label,
            "rms_error_mm": registration.rms_error_mm,
            "max_error_mm": registration.max_error_mm,
            "report": "registration_report.json",
        }
        (output / f"{prog_name}_frame.json").write_text(
            json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        report = {
            "measured_frame": registration.frame_label,
            "uframe": registration.uframe,
            "rotation_vehicle_to_measured": registration.rotation.tolist(),
            "translation_vehicle_origin_in_measured_mm": registration.translation.tolist(),
            "rms_error_mm": registration.rms_error_mm,
            "max_error_mm": registration.max_error_mm,
            "point_pairs": [
                {
                    "vehicle_mm": cad.tolist(),
                    "measured_mm": measured.tolist(),
                    "residual_mm": residual.tolist(),
                    "error_mm": float(np.linalg.norm(residual)),
                }
                for cad, measured, residual in zip(
                    registration.cad_points,
                    registration.measured_points,
                    registration.residuals_mm,
                    strict=True,
                )
            ],
            "reset_poses_vehicle": data.get("reset_poses", {
                name: {"xyz": [0.0, 0.0, 0.0], "wpr": [0.0, 0.0, 0.0]}
                for name in ("base", "support", "part")
            }),
            "part_scan_face_axis_vehicle": face_axis_vehicle.tolist(),
            "scan_face_axis_measured_frame": scan_axis_target.tolist(),
            "max_registration_error_mm": tolerance,
            # Which base the plan traced: an uploaded one, the station's own `input/base.stl`,
            # or the reference plate placed behind the scan face. It is a planning input, so
            # the report has to say which one it was.
            "base_source": base_source,
            "outputs": [entry["name"] for entry in _download_entries(prog_name)],
            "warning": (
                "This is geometric planning only; verify robot reachability, collision, "
                "UFRAME, and the scanner TCP in ROBOGUIDE before running."
            ),
        }
        (output / "registration_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        _write_planning_report(
            output / "planning_report.html",
            run_id=run_id,
            prog_name=prog_name,
            report=result.report,
            registration=registration,
            face_axis=face_axis_vehicle,
            downloads=[entry["name"] for entry in _download_entries(prog_name)],
        )
    except Exception:
        (output / "job_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise

    delivered = []
    for viewpoint in result.viewpoints:
        record = viewpoint.as_dict()
        record["position"] = _vehicle_point(combined_part, viewpoint.position)
        record["axis"] = _vehicle_axis(combined_part, viewpoint.axis)
        delivered.append(record)
    delivered.sort(key=lambda record: record["order"])

    return {
        "run_id": run_id,
        "output_dir": str(output),
        "program": str(output / f"{prog_name}.LS"),
        "program_name": prog_name,
        "delivery_mode": "stroke" if stroke_plan else "pose",
        "point_count": len(result.viewpoints),
        "coverage_pct": float(result.report["coverage_pct"]),
        "target_points": int(result.report.get("target_points", 0)),
        "uncovered_points": int(result.report.get("uncovered_points", 0)),
        "target_face_mm2": float(result.report.get("target_face_mm2", 0.0)),
        "face": str(result.report.get("face", "")),
        "uncovered_by_reason": result.report.get("uncovered_by_reason", {}),
        "scan_face_axis_vehicle": face_axis_vehicle.tolist(),
        "scan_face_axis_measured_frame": scan_axis_target.tolist(),
        "measured_frame": registration.frame_label,
        "registration_rms_error_mm": registration.rms_error_mm,
        "registration_max_error_mm": registration.max_error_mm,
        "datum_count": int(len(registration.cad_points)),
        # The datum inputs are part of the result so the page can show them after a run
        # and re-localise from them without asking the operator to retype the pairs.
        "cad_points_vehicle_mm": [
            [float(value) for value in point] for point in registration.cad_points
        ],
        "measured_points_mm": [
            [float(value) for value in point] for point in registration.measured_points
        ],
        "point_errors_mm": [
            float(value) for value in np.linalg.norm(registration.residuals_mm, axis=1)
        ],
        "uframe": int(registration.uframe),
        "error_limit_mm": float(tolerance),
        "utool": int(data.get("utool", 1)),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "base_supplied": base_supplied,
        "base_source": base_source,
        "viewpoints": delivered,
        "viewer_url": f"/api/viewer?run_id={run_id}",
        "downloads": _download_entries(prog_name),
    }


def _relocated_run_id(run_id: str) -> str:
    suffix = "_reloc"
    stem = run_id[: 64 - len(suffix)].rstrip("_-") or "run"
    return stem + suffix


def _frame_delta(registration: RigidRegistration, old_rotation: np.ndarray,
                 old_translation: np.ndarray) -> np.ndarray:
    """The rigid motion that carries a pose from the old measured frame to the new one.

    The path is a rigid-body manoeuvre relative to the part: a point the planner put at
    `T_old p` in the old measured frame sits at `T_new p` once the installation is
    corrected, so the whole path moves by `T_new · T_old⁻¹` and nothing has to be
    re-planned.
    """
    old = np.eye(4)
    old[:3, :3] = old_rotation
    old[:3, 3] = old_translation
    new = np.eye(4)
    new[:3, :3] = registration.rotation
    new[:3, 3] = registration.translation
    return new @ np.linalg.inv(old)


def _moved_point(delta: np.ndarray, point) -> list[float]:
    moved = delta[:3, :3] @ np.asarray(point, dtype=float) + delta[:3, 3]
    return [round(float(value), 4) for value in moved]


def _moved_axis(delta: np.ndarray, axis) -> list[float]:
    moved = delta[:3, :3] @ np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(moved))
    if norm < 1e-12:
        return [0.0, 0.0, 0.0]
    return [round(float(value / norm), 9) for value in moved]


def _viewpoint_from_record(record: dict) -> Viewpoint:
    """Rebuild a delivered viewpoint so the standard writers can re-emit it.

    The joint evidence (`config`, `branch`, `joints_deg`, `j5_margin_deg`) is
    deliberately dropped: those numbers are the verdict of a reachability test in the
    OLD frame, and a relocated program must not ship a stale verdict as if it were
    re-checked. Re-plan with `robot.enabled` if that evidence is needed again.
    """
    return Viewpoint(
        id=int(record.get("id", 0)),
        position=np.asarray(record.get("position", [0.0, 0.0, 0.0]), dtype=float),
        axis=np.asarray(record.get("axis", [0.0, 0.0, 1.0]), dtype=float),
        roll_deg=float(record.get("roll_deg", 0.0)),
        n_points=int(record.get("n_points", 0)),
        area_mm2=float(record.get("area_mm2", 0.0)),
        mean_incidence_deg=float(record.get("mean_incidence_deg", 0.0)),
        max_incidence_deg=float(record.get("max_incidence_deg", 0.0)),
        tracker_ok=bool(record.get("tracker_ok", True)),
        source=str(record.get("source", "")),
        order=int(record.get("order", record.get("id", 0))),
        stroke_id=None if record.get("stroke_id") is None else int(record["stroke_id"]),
        role=str(record.get("role", "coverage")),
    )


def _moved_viewpoints(records, delta: np.ndarray) -> list[Viewpoint]:
    moved: list[Viewpoint] = []
    for record in records:
        viewpoint = _viewpoint_from_record(record)
        moved.append(replace(
            viewpoint,
            position=np.asarray(_moved_point(delta, viewpoint.position), dtype=float),
            axis=np.asarray(_moved_axis(delta, viewpoint.axis), dtype=float),
        ))
    return moved


def _moved_strokes(strokes: dict, delta: np.ndarray) -> dict:
    """The authoritative move list, transformed. Lengths and ids are frame-free."""
    moved = json.loads(json.dumps(strokes))
    for point in moved.get("points", []):
        point["position_mm"] = _moved_point(delta, point["position_mm"])
        point["axis"] = _moved_axis(delta, point["axis"])
    for stroke in moved.get("strokes", []):
        stroke["start_mm"] = _moved_point(delta, stroke["start_mm"])
        if stroke.get("end_mm") is not None:
            stroke["end_mm"] = _moved_point(delta, stroke["end_mm"])
        stroke["axis"] = _moved_axis(delta, stroke["axis"])
    return moved


def _move_samples_csv(source: Path, target: Path, delta: np.ndarray) -> bool:
    """Re-express a point-cloud CSV (points and normals) in the new frame."""
    if not source.is_file():
        return False
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))[1:]
    if not rows:
        return False
    points = np.asarray([[float(row[0]), float(row[1]), float(row[2])] for row in rows])
    normals = np.asarray([[float(row[3]), float(row[4]), float(row[5])] for row in rows])
    reasons = [row[6] if len(row) > 6 else "" for row in rows]
    moved_points = points @ delta[:3, :3].T + delta[:3, 3]
    moved_normals = normals @ delta[:3, :3].T
    write_samples_csv(target, moved_points, moved_normals, reasons)
    return True


def _relocation_export_options(run_dir: Path, prog_name: str) -> dict:
    """Motion and termination of the ORIGINAL program, so the move is the same move.

    Reading them from the old frame contract rather than from today's configuration is
    what keeps a relocation from silently changing the speed the coverage was computed
    for. Whether the order may be re-sorted is decided by the viewpoints themselves (a
    stroke plan carries `stroke_id`, and re-sorting would split sweeps).
    """
    old_path = run_dir / f"{prog_name}_frame.json"
    program: dict = {}
    if old_path.is_file():
        try:
            loaded = json.loads(old_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = None
        if isinstance(loaded, dict) and isinstance(loaded.get("program"), dict):
            program = loaded["program"]
    return {
        "motion": str(program.get("motion") or "J"),
        "speed": str(program.get("speed") or "100%"),
        "termination": str(program.get("termination") or "FINE"),
        # The speed of the NON-scanning joint moves of the original program. Older runs
        # carry none, in which case the exporter's own default applies.
        "joint_speed": program.get("joint_speed"),
    }


def _run_program_name(run_dir: Path) -> str:
    for path in sorted(run_dir.glob("*_frame.json")):
        try:
            return normalise_prog_name(path.name[: -len("_frame.json")])
        except ValueError:
            continue
    return ""


def _relocation_config(run_dir: Path):
    """The configuration the ORIGINAL plan ran with, for the post-shift reach audit.

    Read from the old coverage report because that is the file that names it; fall back to
    the shipped configuration when it is missing (an old run, or one planned with a config
    that has since moved). The robot section is what matters here.
    """
    source = None
    report_path = run_dir / "coverage_report.json"
    if report_path.is_file():
        try:
            loaded = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = None
        if isinstance(loaded, dict):
            source = (loaded.get("config") or {}).get("source")
    try:
        return load_config(source) if source else load_config()
    except (OSError, ValueError):
        return load_config()


def _reach_audit(moved, config) -> dict:
    """Ask the arm about every moved point, in the CORRECTED frame.

    A re-localisation is not a re-plan, but it is a rigid MOVE, and the envelope does not
    move with it. On 2026-10-30 a path was relocated three times (50 -> 100 -> 350 mm) and
    the exported program stopped at MOTN-018 on its first J move: the shift had walked the
    poses out of reach and no stage had asked the robot, because the relocated viewpoints
    carry no joint evidence and the exporter only audits Cartesian poses under UFRAME 0.

    The gate used here is the planner's own, on the same numbers the planner checked, so a
    verdict means exactly what it meant during planning - unlike a blind IK in a user frame
    nobody calibrated.
    """
    gate = KinematicGate(config.robot)
    if not gate.active:
        return {"enabled": False, "checked": 0, "feasible": 0, "failures": []}
    failures = []
    for viewpoint in moved:
        verdict = gate.check(viewpoint.position, viewpoint.axis, viewpoint.roll_deg)
        if verdict.ok:
            continue
        failures.append({
            "order": int(viewpoint.order),
            "position": [round(float(value), 2) for value in viewpoint.position],
            "reason": verdict.reason,
        })
    return {"enabled": True, "checked": len(moved),
            "feasible": len(moved) - len(failures),
            "envelope_source": gate.robot.limits_source if gate.robot else None,
            "failures": failures}


def _reach_refusal_message(reach: dict, *, relocalisation_shift: float) -> str:
    """Why the `.LS` is not written, in words the operator can act on."""
    count = len(reach["failures"])
    orders = ", ".join(str(item["order"]) for item in reach["failures"][:8])
    more = "..." if count > 8 else ""
    return (
        f"重新定位后 {count} / {reach['checked']} 个点位超出机器人包络"
        f"（本步位移 {relocalisation_shift:.0f} mm，程序序号 {orders}{more}），"
        "本次没有写出 .LS：这么大的刚体位移已不是基准修正，请用修正后的现场坐标重新规划路径。"
    )


def relocate_job(data: dict, progress=None) -> dict:
    """Move a completed run's path into a corrected field frame - without re-planning.

    Correcting a datum point does not change which poses the workshop wants to fly; it
    only moves them. The path is rigid relative to the part, so the whole delivered set
    - viewpoints, strokes, `.LS`, reports and the 3D view - is transformed by
    `T_new · T_old⁻¹`. That takes seconds instead of minutes and, unlike a re-plan,
    cannot produce a different path from one datum edit.

    What a transform cannot answer is stated in the report instead of being guessed:
    coverage is unchanged (part and path move together), but tracker visibility, robot
    reachability and joint limits were judged in the old frame and are NOT re-tested.
    """
    if not isinstance(data, dict):
        raise ValueError("request body must be a JSON object")
    run_id = str(data.get("run_id") or "")
    if not _SAFE_RUN_ID.fullmatch(run_id):
        raise ValueError("invalid run id")
    run_dir = _run_directory(run_id)
    announce = progress or (lambda stage, fraction, detail="": None)
    started = time.perf_counter()
    announce("registration", 0.04, "reading the previous run")

    report_path = run_dir / "registration_report.json"
    viewpoints_path = run_dir / "viewpoints.json"
    if not report_path.is_file() or not viewpoints_path.is_file():
        raise ValueError(
            "this run has no viewpoints/registration report; run a full plan instead"
        )
    old_report = json.loads(report_path.read_text(encoding="utf-8"))
    old_rotation = np.asarray(old_report["rotation_vehicle_to_measured"], dtype=float)
    old_translation = np.asarray(
        old_report["translation_vehicle_origin_in_measured_mm"], dtype=float
    )
    records = json.loads(viewpoints_path.read_text(encoding="utf-8"))
    if not records:
        raise ValueError("this run's viewpoint file is empty")

    previous = _history_run_payload(run_id) or {}
    datums = _run_datums(run_id, previous)
    cad_points = data.get("cad_points_vehicle_mm") or datums["cad_points_vehicle_mm"]
    if not cad_points:
        raise ValueError("this run has no recorded datum points; plan it again from the model")
    cad = np.asarray(cad_points, dtype=float)
    measured = np.asarray(data.get("measured_points_mm"), dtype=float)
    if cad.ndim != 2 or cad.shape[1:] != (3,) or measured.shape != cad.shape:
        raise ValueError(
            f"this run registered {len(np.atleast_2d(cad))} datum points; enter the same "
            "number of matching field XYZ coordinates"
        )
    if not np.isfinite(cad).all() or not np.isfinite(measured).all():
        raise ValueError("datum coordinates must be finite numbers")

    uframe = data.get("uframe")
    if uframe is None:
        uframe = datums["uframe"] if datums["uframe"] is not None else 1
    uframe = int(uframe)
    tolerance = data.get("max_registration_error_mm")
    if tolerance is None:
        tolerance = datums["max_registration_error_mm"]
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("registration error limit must be a positive finite number")

    registration = fit_points(
        cad, measured,
        target_frame="robot_base" if uframe == 0 else "robot_uframe_1",
        uframe=uframe,
    )
    if registration.max_error_mm > tolerance:
        raise ValueError(
            f"corrected datum registration max error is {registration.max_error_mm:.3f} mm; "
            f"the allowed limit is {tolerance:.3f} mm"
        )
    announce("registration", 0.18, "corrected datum points fitted")
    delta = _frame_delta(registration, old_rotation, old_translation)

    face_axis_vehicle = np.asarray(
        data.get("scan_face_axis") or datums["scan_face_axis_vehicle"]
        or old_report.get("part_scan_face_axis_vehicle") or [0.0, 0.0, 1.0],
        dtype=float,
    )
    scan_axis_target = registration.rotation @ face_axis_vehicle
    prog_name = normalise_prog_name(
        data.get("prog_name") or previous.get("program_name")
        or _run_program_name(run_dir) or "PART01"
    )
    utool = int(previous.get("utool", 1))
    requested = _relocated_run_id(run_id)
    run_id_new = _available_run_id(requested, _runs_root())
    output = _runs_root() / run_id_new
    output.mkdir(parents=True, exist_ok=False)

    try:
        announce("export", 0.28, "moving the planned viewpoints")
        moved = _moved_viewpoints(records, delta)
        strokes_path = run_dir / "strokes.json"
        strokes = (json.loads(strokes_path.read_text(encoding="utf-8"))
                   if strokes_path.is_file() else None)

        resets = old_report.get("reset_poses_vehicle") or {}
        identity = {"xyz": [0.0, 0.0, 0.0], "wpr": [0.0, 0.0, 0.0]}
        part_reset = _pose_matrix(resets.get("part", identity), "part")
        support_reset = _pose_matrix(resets.get("support", identity), "support")
        base_reset = _pose_matrix(resets.get("base", identity), "base")
        placement = np.eye(4)
        placement[:3, :3] = registration.rotation
        placement[:3, 3] = registration.translation
        combined_part = placement @ part_reset
        combined_support = placement @ support_reset
        combined_base = placement @ base_reset

        # --- meshes: same uploaded bodies, base re-installed in the new frame ---------
        part_path = run_dir / "part.stl"
        support_path = run_dir / "support.stl"
        if part_path.is_file() and support_path.is_file():
            (output / "part.stl").write_bytes(part_path.read_bytes())
            (output / "support.stl").write_bytes(support_path.read_bytes())
        base_vehicle = _vehicle_frame_base(run_dir)
        if base_vehicle:
            (output / "base_vehicle.stl").write_bytes(base_vehicle)
            base_mesh = _load_stl(base_vehicle, "base")
            base_mesh.apply_transform(combined_base)
            base_mesh.export(output / "base.stl")

        station_path = output / "station.json"
        station_path.write_text(json.dumps({
            "part": "part.stl",
            "fixture": "support.stl",
            "base": "base.stl",
            "transform_part": combined_part.tolist(),
            "transform_fixture": combined_support.tolist(),
            "note": (
                "Generated by scanplan web UI from a re-localisation: the planned path "
                "was transformed into the corrected field frame, not re-planned."
            ),
        }, indent=2) + "\n", encoding="utf-8")

        # --- the delivered path -------------------------------------------------------
        write_viewpoints_json(output / "viewpoints.json", moved)
        write_viewpoints_csv(output / "viewpoints.csv", moved)
        announce("export", 0.38, "viewpoints written")
        if strokes is not None:
            (output / "strokes.json").write_text(
                json.dumps(_moved_strokes(strokes, delta), indent=2, ensure_ascii=False)
                + "\n", encoding="utf-8")
        for name in ("exempt_points.csv", "unreachable_points.csv"):
            _move_samples_csv(run_dir / name, output / name, delta)
        samples = run_dir / "target_samples.npz"
        if samples.is_file():
            # The sidecar carries the sampled surface the 3D page draws, so it moves with
            # the part. Copying it would leave a point cloud stranded in the old frame -
            # the one kind of misalignment a rigid relocation must not produce.
            sample_points, sampled_covered, sampled_mask = load_target_samples(samples)
            save_target_samples(
                output / "target_samples.npz",
                points=sample_points @ delta[:3, :3].T + delta[:3, 3],
                covered=sampled_covered,
                mask=sampled_mask,
            )

        announce("export", 0.5, "exporting the FANUC program")
        options = _relocation_export_options(run_dir, prog_name)
        # Ask the arm before writing anything: a shift large enough to leave the envelope is
        # exactly what a relocation cannot detect on its own, and the .LS is the one artifact
        # that must never be written when the answer is no.
        reach = _reach_audit(moved, _relocation_config(run_dir))
        announce("export", 0.55,
                 f"reachability {reach['feasible']} / {reach['checked']} points")
        export_error = None
        contract: dict = {}
        if reach["failures"]:
            export_error = _reach_refusal_message(reach, relocalisation_shift=float(
                np.linalg.norm(delta[:3, 3])))
        else:
            contract = export_ls(moved, output, prog_name=prog_name, uframe=uframe,
                                 utool=utool,
                                 order=not any(vp.stroke_id is not None for vp in moved),
                                 **options)
        contract.setdefault("frame", {})["axes"] = (
            f"Robot UFRAME {uframe} axes; the delivered path was rigidly moved from the "
            "original installation by the corrected datum registration."
        )
        contract.setdefault("frame", {})["note"] = (
            "See registration_report.json for the datum pairs, the new transform and the "
            "old transform it replaced. The path itself was NOT re-planned."
        )
        contract.setdefault("uframe_utool", {})["uframe_note"] = (
            f"Teach/verify UFRAME {uframe} matches the field coordinate frame used for the "
            "corrected measured datum points."
        )
        contract.setdefault("placement", {})["registration"] = {
            "method": "corrected datum registration; path transformed, not re-planned",
            "measured_frame": registration.frame_label,
            "rms_error_mm": registration.rms_error_mm,
            "max_error_mm": registration.max_error_mm,
            "relocated_from": run_id,
            "report": "registration_report.json",
        }
        if export_error is not None:
            contract["program_written"] = False
            contract["refusal"] = export_error
        (output / f"{prog_name}_frame.json").write_text(
            json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if export_error is None:
            announce("export", 0.62, "program exported")
        else:
            announce("export", 0.62, "program NOT written: points outside the arm envelope")

        # --- reports -----------------------------------------------------------------
        relocalisation = {
            "from_run": run_id,
            "method": "rigid transform of the delivered path: delta = T_new . T_old^-1",
            "delta_rotation": np.round(delta[:3, :3], 9).tolist(),
            "delta_translation_mm": np.round(delta[:3, 3], 4).tolist(),
            "shift_mm": round(float(np.linalg.norm(delta[:3, 3])), 3),
            "replanned": False,
            "unchanged": [
                "coverage: the part and the path move together, so the reachable set and "
                "the covered area are the same as the original run",
                "viewpoint count, stroke pairing and program order",
                "incidence angles and standoff distances",
            ],
            "not_recomputed": [
                "tracker_ok per viewpoint: the tracker does not move with the part, so "
                "visibility was NOT re-tested",
                "collision and cell occlusion: never modelled, in the old frame or the new one",
                "the joint evidence itself: CONFIG and joints are dropped, not carried over, "
                "so the reachability audit starts from the Cartesian pose",
            ],
            # What the shift DID cost, measured rather than assumed: the envelope does not
            # move with the part, so every point is asked again in the corrected frame.
            "robot_reachability_after_shift": reach,
            "when_to_replan": (
                "re-plan when the correction is large compared with the scanner standoff, "
                "when the tracker is marginal, or when robot.enabled is on and the shift "
                "could move a pose towards a joint limit"
            ),
        }
        registration_report = {
            "measured_frame": registration.frame_label,
            "uframe": int(registration.uframe),
            "rotation_vehicle_to_measured": registration.rotation.tolist(),
            "translation_vehicle_origin_in_measured_mm": registration.translation.tolist(),
            "rms_error_mm": registration.rms_error_mm,
            "max_error_mm": registration.max_error_mm,
            "max_registration_error_mm": tolerance,
            "point_pairs": [
                {
                    "vehicle_mm": cad_point.tolist(),
                    "measured_mm": measured_point.tolist(),
                    "residual_mm": residual.tolist(),
                    "error_mm": float(np.linalg.norm(residual)),
                }
                for cad_point, measured_point, residual in zip(
                    registration.cad_points,
                    registration.measured_points,
                    registration.residuals_mm,
                    strict=True,
                )
            ],
            "reset_poses_vehicle": resets or {
                name: identity for name in ("base", "support", "part")
            },
            "part_scan_face_axis_vehicle": face_axis_vehicle.tolist(),
            "scan_face_axis_measured_frame": scan_axis_target.tolist(),
            "relocalisation": relocalisation,
            "outputs": [entry["name"] for entry in _download_entries(prog_name)],
            "warning": (
                "The path was moved by a rigid transform, not re-planned. Verify UFRAME, "
                "UTOOL, reachability and tracker sight lines in ROBOGUIDE before motion."
            ),
        }
        (output / "registration_report.json").write_text(
            json.dumps(registration_report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")

        coverage_report: dict = {}
        coverage_path = run_dir / "coverage_report.json"
        if coverage_path.is_file():
            try:
                loaded = json.loads(coverage_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                loaded = None
            if isinstance(loaded, dict):
                coverage_report = loaded
        coverage_report["installation_registration"] = {
            "measured_frame": registration.frame_label,
            "uframe": int(registration.uframe),
            "rms_error_mm": registration.rms_error_mm,
            "max_error_mm": registration.max_error_mm,
            "cad_face_axis_vehicle": face_axis_vehicle.tolist(),
            "scan_face_axis_measured_frame": scan_axis_target.tolist(),
        }
        coverage_report["viewpoint_records"] = [vp.as_dict() for vp in moved]
        coverage_report["relocalisation"] = relocalisation
        for region in coverage_report.get("leak_regions") or []:
            center = region.get("center")
            if isinstance(center, list) and len(center) == 3:
                region["center"] = _moved_point(delta, center)
        if coverage_report:
            write_report(output, coverage_report)

        summary_path = run_dir / "summary.txt"
        if summary_path.is_file():
            if reach["failures"]:
                reach_line = (
                    f"robot reachability after the shift: {reach['feasible']} / "
                    f"{reach['checked']} points inside the arm's envelope - the .LS was "
                    "NOT written, re-plan instead\n"
                )
            elif reach["enabled"]:
                reach_line = (
                    f"robot reachability after the shift: {reach['checked']} / "
                    f"{reach['checked']} points inside the arm's envelope\n"
                )
            else:
                reach_line = "robot reachability after the shift: NOT checked (gate off)\n"
            note = (
                f"\nre-localised from {run_id}: the planned path was transformed by the "
                f"corrected datum registration (shift {relocalisation['shift_mm']:.3f} mm), "
                "not re-planned.\n"
                "  coverage, incidence and order are the original run's figures.\n"
                "  tracker visibility was NOT re-tested.\n"
                f"  {reach_line}"
            )
            (output / "summary.txt").write_text(
                summary_path.read_text(encoding="utf-8") + note, encoding="utf-8")

        _write_planning_report(
            output / "planning_report.html",
            run_id=run_id_new,
            prog_name=prog_name,
            report=coverage_report or {"coverage_pct": 0.0, "target_face_mm2": 0.0},
            registration=registration,
            face_axis=face_axis_vehicle,
            downloads=[entry["name"] for entry in _download_entries(prog_name)],
        )
        announce("report", 0.78, "reports written")

        viewer_note = ""
        try:
            announce("report", 0.84, "re-rendering the 3D view")
            render_run(output / "viewpoints.json", output,
                       station=output / "station.json")
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            viewer_note = f"；3D 视图未能重新生成：{error}"
            print(f"[scanplan-web] 3D view not re-rendered: {error}", flush=True)
        # Last, so the operator still gets the reports and the 3D view that show WHY: a
        # program the arm cannot fly is not a delivery.
        if export_error is not None:
            raise ValueError(export_error)
    except Exception:
        (output / "job_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    announce("export", 0.97, "re-localisation written" + viewer_note)

    delivered = []
    for viewpoint in moved:
        record = viewpoint.as_dict()
        record["position"] = _vehicle_point(combined_part, viewpoint.position)
        record["axis"] = _vehicle_axis(combined_part, viewpoint.axis)
        delivered.append(record)
    delivered.sort(key=lambda record: record["order"])

    result = dict(previous)
    result.update({
        "relocated_from": run_id,
        "relocalised": True,
        "run_id": run_id_new,
        "output_dir": str(output),
        "program": str(output / f"{prog_name}.LS"),
        "program_name": prog_name,
        "delivery_mode": "stroke" if strokes is not None else "pose",
        "point_count": len(moved),
        "scan_face_axis_vehicle": face_axis_vehicle.tolist(),
        "scan_face_axis_measured_frame": scan_axis_target.tolist(),
        "measured_frame": registration.frame_label,
        "registration_rms_error_mm": registration.rms_error_mm,
        "registration_max_error_mm": registration.max_error_mm,
        "datum_count": int(len(registration.cad_points)),
        "cad_points_vehicle_mm": [
            [float(value) for value in point] for point in registration.cad_points
        ],
        "measured_points_mm": [
            [float(value) for value in point] for point in registration.measured_points
        ],
        "point_errors_mm": [
            float(value) for value in np.linalg.norm(registration.residuals_mm, axis=1)
        ],
        "uframe": int(registration.uframe),
        "error_limit_mm": tolerance,
        "shift_mm": relocalisation["shift_mm"],
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "viewpoints": delivered,
        "viewer_url": f"/api/viewer?run_id={run_id_new}",
        "downloads": _download_entries(prog_name),
    })
    # Numbers a relocated run inherits are the original plan's; fill them in when the
    # history payload predates the run (or the record was deleted from the database).
    result.setdefault("coverage_pct", float(coverage_report.get("coverage_pct", 0.0)))
    result.setdefault("target_points", int(coverage_report.get("target_points", 0)))
    result.setdefault("uncovered_points", int(coverage_report.get("uncovered_points", 0)))
    result.setdefault("target_face_mm2", float(coverage_report.get("target_face_mm2", 0.0)))
    result.setdefault("face", str(coverage_report.get("face", "")))
    result.setdefault("uncovered_by_reason", coverage_report.get("uncovered_by_reason", {}))
    result.setdefault("base_supplied", bool(base_vehicle))
    return result


# --- history (SQLite) ------------------------------------------------------
#
# Every completed run is recorded next to its output directory, so the browser
# can list, reopen and re-download earlier work after a restart. The payload is
# the same JSON that /api/solve returned, zlib-compressed to keep viewpoint
# tables small. History is advisory: a write failure must never fail a job.

_HISTORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    prog_name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    coverage_pct REAL,
    point_count INTEGER,
    target_points INTEGER,
    uncovered_points INTEGER,
    target_face_mm2 REAL,
    face TEXT,
    error TEXT,
    payload BLOB
);
CREATE INDEX IF NOT EXISTS runs_created_idx ON runs (created_at DESC);
"""


def _history_connection() -> sqlite3.Connection:
    root = Path.cwd() / "output" / "web_runs"
    root.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(root / "history.sqlite3", timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.executescript(_HISTORY_SCHEMA)
    return connection


def _init_history() -> None:
    connection = _history_connection()
    try:
        connection.commit()
    finally:
        connection.close()


def _record_history_run(result: dict, display_name: str = "") -> None:
    try:
        packed = zlib.compress(
            json.dumps(result, ensure_ascii=False).encode("utf-8"), 6
        )
        name = display_name.strip() or f"{result['program_name']} · {result['run_id']}"
        connection = _history_connection()
        try:
            connection.execute(
                "INSERT OR REPLACE INTO runs (run_id, display_name, prog_name,"
                " created_at, status, coverage_pct, point_count, target_points,"
                " uncovered_points, target_face_mm2, face, error, payload)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)",
                (
                    result["run_id"],
                    name,
                    result["program_name"],
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "completed",
                    float(result["coverage_pct"]),
                    int(result["point_count"]),
                    int(result["target_points"]),
                    int(result["uncovered_points"]),
                    float(result.get("target_face_mm2", 0.0)),
                    str(result.get("face", "")),
                    packed,
                ),
            )
            connection.commit()
        finally:
            connection.close()
    except (sqlite3.Error, OSError) as error:
        print(f"[scanplan-web] history write skipped: {error}", flush=True)


def _history_runs(limit: int = 200) -> list[dict]:
    connection = _history_connection()
    try:
        rows = connection.execute(
            "SELECT run_id, display_name, prog_name, created_at, status, coverage_pct,"
            " point_count, target_points, uncovered_points, target_face_mm2, face, error"
            " FROM runs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        connection.close()
    return [dict(row) for row in rows]


def _history_run_payload(run_id: str) -> dict | None:
    connection = _history_connection()
    try:
        row = connection.execute(
            "SELECT payload FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
    finally:
        connection.close()
    if row is None or row["payload"] is None:
        return None
    return json.loads(zlib.decompress(row["payload"]).decode("utf-8"))


def _delete_history_run(run_id: str) -> bool:
    connection = _history_connection()
    try:
        deleted = connection.execute(
            "DELETE FROM runs WHERE run_id = ?", (run_id,)
        ).rowcount > 0
        connection.commit()
    finally:
        connection.close()
    runs_root = (Path.cwd() / "output" / "web_runs").resolve()
    run_dir = (runs_root / run_id).resolve()
    if deleted and run_dir.parent == runs_root and run_dir.is_dir():
        shutil.rmtree(run_dir, ignore_errors=True)
    return deleted


def _runs_root() -> Path:
    return (Path.cwd() / "output" / "web_runs").resolve()


def _run_directory(run_id: str) -> Path:
    root = _runs_root()
    run_dir = (root / run_id).resolve()
    if run_dir.parent != root or not run_dir.is_dir():
        raise ValueError("run not found")
    return run_dir


def _vehicle_frame_base(run_dir: Path) -> bytes | None:
    """The base mesh in uploaded (vehicle) coordinates, for a re-localisation.

    Runs store `base_vehicle.stl` next to the transformed copy.  Runs recorded before
    that file existed hold only the installed copy, so it is inverted back through the
    registration recorded in the same directory.
    """
    raw = run_dir / "base_vehicle.stl"
    if raw.is_file():
        return raw.read_bytes()
    transformed = run_dir / "base.stl"
    report_path = run_dir / "registration_report.json"
    if not transformed.is_file() or not report_path.is_file():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        combined = np.eye(4)
        combined[:3, :3] = np.asarray(report["rotation_vehicle_to_measured"], dtype=float)
        combined[:3, 3] = np.asarray(
            report["translation_vehicle_origin_in_measured_mm"], dtype=float
        )
        reset = (report.get("reset_poses_vehicle") or {}).get("base") or {}
        if reset:
            combined = combined @ _pose_matrix(reset, "base")
        mesh = _load_stl(transformed.read_bytes(), "base")
        mesh.apply_transform(np.linalg.inv(combined))
        return mesh.export(file_type="stl")
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _run_datums(run_id: str, stored: dict | None = None) -> dict:
    """The datum inputs a completed run used, for the page's re-localise editor.

    Newer runs record them in the stored result; older ones only keep the pairs in
    `registration_report.json`, which is read as the fallback.
    """
    stored = stored or {}
    report: dict = {}
    run_dir = _runs_root() / run_id
    report_path = run_dir / "registration_report.json"
    if report_path.is_file():
        try:
            loaded = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = None
        if isinstance(loaded, dict):
            report = loaded
    pairs = [pair for pair in report.get("point_pairs") or [] if isinstance(pair, dict)]

    cad = stored.get("cad_points_vehicle_mm") or [
        pair.get("vehicle_mm") for pair in pairs
    ]
    measured = stored.get("measured_points_mm") or [
        pair.get("measured_mm") for pair in pairs
    ]
    errors = stored.get("point_errors_mm")
    if not errors:
        errors = [
            float(np.linalg.norm(np.asarray(pair["residual_mm"], dtype=float)))
            for pair in pairs
            if pair.get("residual_mm") is not None
        ]
    limit = stored.get("error_limit_mm")
    if limit is None:
        limit = report.get("max_registration_error_mm")
    if limit is None:
        observed = report.get("max_error_mm")
        limit = max(3.0, float(observed)) if isinstance(observed, (int, float)) else 3.0
    uframe = stored.get("uframe")
    if uframe is None:
        uframe = report.get("uframe")
    return {
        "cad_points_vehicle_mm": cad,
        "measured_points_mm": measured,
        "point_errors_mm": errors,
        "uframe": uframe,
        "max_registration_error_mm": float(limit),
        "scan_face_axis_vehicle": (
            stored.get("scan_face_axis_vehicle")
            or report.get("part_scan_face_axis_vehicle")
        ),
        # The stroke stage writes `strokes.json` only when sweeps are the deliverable,
        # so a run recorded before the result kept its mode still reports it correctly.
        "delivery_mode": (
            stored.get("delivery_mode")
            or ("stroke" if (run_dir / "strokes.json").is_file() else "pose")
        ),
        "prog_name": stored.get("program_name") or "PART01",
    }


def _augment_datums(record: dict, run_id: str) -> dict:
    """Fill the datum inputs the page edits when a stored result predates them."""
    datums = _run_datums(run_id, record)
    merged = dict(record)
    for key in ("cad_points_vehicle_mm", "measured_points_mm", "point_errors_mm"):
        if not merged.get(key):
            merged[key] = datums[key] or []
    if merged.get("uframe") is None:
        merged["uframe"] = datums["uframe"]
    if merged.get("error_limit_mm") is None:
        merged["error_limit_mm"] = datums["max_registration_error_mm"]
    if not merged.get("scan_face_axis_vehicle"):
        merged["scan_face_axis_vehicle"] = datums["scan_face_axis_vehicle"] or []
    if not merged.get("delivery_mode"):
        merged["delivery_mode"] = datums["delivery_mode"]
    if not merged.get("program_name"):
        merged["program_name"] = datums["prog_name"]
    return merged


class _Handler(BaseHTTPRequestHandler):
    server_version = "ScanplanWeb/1.0"

    def do_GET(self) -> None:
        route = urlsplit(self.path)
        if route.path in ("/", "/index.html"):
            self._send_bytes((_WEB_ROOT / "index.html").read_bytes(),
                             "text/html; charset=utf-8")
            return
        if route.path == "/app.css":
            css = _WEB_ROOT / "app.css"
            if not css.is_file():
                self.send_error(404)
                return
            self._send_bytes(css.read_bytes(), "text/css; charset=utf-8")
            return
        if route.path == "/app.js":
            script = _WEB_ROOT / "app.js"
            if not script.is_file():
                self.send_error(404)
                return
            self._send_bytes(script.read_bytes(), "text/javascript; charset=utf-8")
            return
        if route.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        if route.path == "/api/config":
            # The page's defaults come from the configuration the server will actually use,
            # not from a `selected` attribute in the HTML: two sources for "which deliverable"
            # is how the screen says sweep and the run plans poses.
            self._json_response(200, _public_config())
            return
        if route.path == "/api/progress":
            query = parse_qs(route.query)
            token = query.get("id", [""])[0]
            if not _SAFE_RUN_ID.fullmatch(token):
                self.send_error(400, "invalid progress id")
                return
            self._json_response(200, {"ok": True,
                                     "progress": _progress_snapshot(token)})
            return
        if route.path == "/api/history":
            self._json_response(200, {"ok": True, "runs": _history_runs()})
            return
        if route.path == "/api/history/detail":
            query = parse_qs(route.query)
            run_id = query.get("run_id", [""])[0]
            if not _SAFE_RUN_ID.fullmatch(run_id):
                self.send_error(400, "invalid run id")
                return
            record = _history_run_payload(run_id)
            if record is None:
                self._json_response(404, {"ok": False, "error": "run not found"})
                return
            self._json_response(200, {"ok": True, "result": _augment_datums(record, run_id)})
            return
        if route.path == "/api/history/model":
            query = parse_qs(route.query)
            run_id = query.get("run_id", [""])[0]
            name = query.get("name", [""])[0]
            if not _SAFE_RUN_ID.fullmatch(run_id) or name not in (
                "part", "support", "base", "base_vehicle"
            ):
                self.send_error(400, "invalid run or model name")
                return
            runs_root = (Path.cwd() / "output" / "web_runs").resolve()
            model = (runs_root / run_id / f"{name}.stl").resolve()
            if model.parent.parent != runs_root or not model.is_file():
                self.send_error(404)
                return
            self._send_bytes(model.read_bytes(), "model/stl")
            return
        if route.path == "/api/viewer":
            query = parse_qs(route.query)
            run_id = query.get("run_id", [""])[0]
            if not _SAFE_RUN_ID.fullmatch(run_id):
                self.send_error(400, "invalid run id")
                return
            path = Path.cwd() / "output" / "web_runs" / run_id / "viewpoints_3d.html"
            if not path.is_file():
                self.send_error(404)
                return
            self._send_bytes(path.read_bytes(), "text/html; charset=utf-8")
            return
        if route.path == "/api/file":
            query = parse_qs(route.query)
            run_id = query.get("run_id", [""])[0]
            name = query.get("name", [""])[0]
            if not _SAFE_RUN_ID.fullmatch(run_id) or not _SAFE_ARTIFACT.fullmatch(name):
                self.send_error(400, "invalid output file")
                return
            run_dir = (Path.cwd() / "output" / "web_runs" / run_id).resolve()
            path = (run_dir / name).resolve()
            if path.parent != run_dir or not path.is_file():
                self.send_error(404)
                return
            content = path.read_bytes()
            content_type = (
                "application/octet-stream" if name.upper().endswith(".LS")
                else "application/json; charset=utf-8" if name.endswith(".json")
                else "text/html; charset=utf-8" if name.endswith(".html")
                else "text/csv; charset=utf-8" if name.endswith(".csv")
                else "text/plain; charset=utf-8"
            )
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Content-Disposition", f'attachment; filename="{name}"')
            self.end_headers()
            self.wfile.write(content)
            return
        self.send_error(404)

    def _send_bytes(self, content: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self) -> None:
        if self.path not in (
            "/api/register", "/api/solve", "/api/convert", "/api/history/delete",
            "/api/relocate",
        ):
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > _MAX_REQUEST_BYTES:
                raise ValueError(
                    f"request size must be between 1 and {_MAX_REQUEST_BYTES} bytes"
                )
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("request body must be a JSON object")
            if self.path == "/api/convert":
                filename = str(data.get("filename") or "")
                encoded = data.get("data_base64")
                if not isinstance(encoded, str) or not encoded:
                    raise ValueError("缺少模型文件数据")
                try:
                    raw = base64.b64decode(encoded, validate=True)
                except ValueError as error:
                    raise ValueError("模型文件数据不是有效的 base64") from error
                if len(raw) > _MAX_STL_BYTES:
                    raise ValueError(
                        f"模型文件超过 {_MAX_STL_BYTES // (1024 * 1024)} MB 限制"
                    )
                try:
                    mesh = mesh_from_bytes(raw, filename)
                except ValueError:
                    raise
                except Exception as error:
                    raise ValueError(f"模型转换失败：{error}") from error
                stl = to_stl_bytes(mesh)
                if len(stl) > _MAX_STL_BYTES:
                    raise ValueError(
                        f"转换后的网格超过 {_MAX_STL_BYTES // (1024 * 1024)} MB，"
                        "请先在 CAD 中简化模型"
                    )
                self._json_response(200, {
                    "ok": True,
                    "stl_base64": base64.b64encode(stl).decode("ascii"),
                    "faces": int(len(mesh.faces)),
                    "vertices": int(len(mesh.vertices)),
                })
                return
            if self.path == "/api/history/delete":
                run_id = str(data.get("run_id") or "")
                if not _SAFE_RUN_ID.fullmatch(run_id):
                    raise ValueError("invalid run id")
                self._json_response(
                    200, {"ok": True, "deleted": _delete_history_run(run_id)}
                )
                return
            if self.path == "/api/register":
                self._json_response(
                    200, {"ok": True, "result": _preview_registration(data)}
                )
                return
            if self.path == "/api/relocate":
                token = _progress_token(data)
                try:
                    result = relocate_job(data, progress=_progress_writer(token))
                except Exception:
                    _publish_progress(token, "failed", 100.0, "re-localisation failed")
                    raise
                _publish_progress(token, "done", 1.0, "re-localisation finished")
                _record_history_run(result, str(data.get("run_name") or ""))
                self._json_response(200, {"ok": True, "result": result})
                return
            token = _progress_token(data)
            try:
                result = solve_job(data, progress=_progress_writer(token))
            except Exception:
                _publish_progress(token, "failed", 100.0, "planning failed")
                raise
            _publish_progress(token, "done", 1.0, "planning finished")
            _record_history_run(result, str(data.get("run_name") or ""))
            self._json_response(200, {"ok": True, "result": result})
        except (ValueError, json.JSONDecodeError, OSError) as error:
            self._json_response(400, {"ok": False, "error": str(error)})
        except Exception as error:
            traceback.print_exc()
            self._json_response(500, {"ok": False, "error": f"planning failed: {error}"})

    def _json_response(self, status: int, body: dict) -> None:
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args) -> None:
        print(f"[scanplan-web] {self.address_string()} {format % args}")


def _preview_registration(data: dict) -> dict:
    registration, _ = _registration_from_payload(data)
    return {
        "measured_frame": registration.frame_label,
        "uframe": registration.uframe,
        "rms_error_mm": registration.rms_error_mm,
        "max_error_mm": registration.max_error_mm,
        "point_errors_mm": np.linalg.norm(registration.residuals_mm, axis=1).tolist(),
        "rotation_vehicle_to_measured": registration.rotation.tolist(),
        "translation_vehicle_origin_in_measured_mm": registration.translation.tolist(),
    }


def _local_addresses() -> list[str]:
    """Best-effort LAN IPv4 addresses, for the startup banner."""
    addresses: list[str] = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            address = str(info[4][0])
            if address not in addresses and not address.startswith("127."):
                addresses.append(address)
    except OSError:
        pass
    if not addresses:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                probe.connect(("8.8.8.8", 80))
                addresses.append(str(probe.getsockname()[0]))
        except OSError:
            pass
    return addresses


def _bind_urls(host: str, port: int) -> list[str]:
    if host in ("", "0.0.0.0", "::"):
        urls = [f"http://127.0.0.1:{port}"]
        urls.extend(f"http://{address}:{port}" for address in _local_addresses())
        return urls
    if host in ("127.0.0.1", "localhost", "::1"):
        return [f"http://127.0.0.1:{port}"]
    return [f"http://{host}:{port}"]


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    """Serve the UI. Bind 0.0.0.0 to let other machines on the LAN open it."""
    _init_history()
    exposed = host not in ("127.0.0.1", "localhost", "::1")
    if exposed:
        print(
            "warning: the UI is reachable from other machines. Anyone who can reach "
            "this port can submit jobs and download previous outputs, so use it only "
            "on a trusted network.",
            flush=True,
        )
    server = ThreadingHTTPServer((host, port), _Handler)
    for url in _bind_urls(host, port):
        print(f"scanplan web UI: {url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down scanplan web UI", flush=True)
    finally:
        server.server_close()
