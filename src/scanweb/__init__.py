"""Local-only browser interface for station setup, datum registration, and path export.

`scanplan web` serves `index.html` from this package and answers its JSON requests
with the planner's own API; nothing is uploaded anywhere. The page is packaged with
the service (unlike a repo-root asset), so an installed `scanplan web` finds it.
"""

from __future__ import annotations

import base64
import html
import json
import re
import traceback
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import numpy as np
import trimesh

from scanplan.config import load_config, override
from scanplan.pipeline import plan, write_outputs
from scanplan.registration import RigidRegistration, fit_points
from scanplan.roboguide.adapt import export as export_ls
from scanplan.roboguide.wpr import normalise_prog_name
from scanplan.scene import StationDescription, load_station
from scanplan.sheet_split import split_sheet

_WEB_ROOT = Path(__file__).resolve().parent
_STATIC_ROOT = Path(__file__).resolve().parents[1] / "scanplan" / "static"
_MAX_STL_BYTES = 120 * 1024 * 1024
_MAX_REQUEST_BYTES = 380 * 1024 * 1024
_SAFE_RUN_ID = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_SAFE_ARTIFACT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,96}$")


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


def _synthetic_base(part: trimesh.Trimesh, fixture: trimesh.Trimesh) -> trimesh.Trimesh:
    """A flat reference plate under the uploaded models when no base is supplied.

    The planner traces the base as an obstacle, so it cannot simply be omitted.
    A plate well below the part and fixture keeps the base from creating false
    occlusion while still giving the station a closed, printable reference body.
    """
    bounds = np.vstack((np.asarray(part.bounds, dtype=float),
                        np.asarray(fixture.bounds, dtype=float)))
    low = bounds.min(axis=0)
    high = bounds.max(axis=0)
    span = np.maximum(high - low, 1.0)
    thickness = max(20.0, float(span[2]) * 0.04)
    plate = trimesh.creation.box(extents=[
        float(span[0]) + 120.0,
        float(span[1]) + 120.0,
        thickness,
    ])
    plate.apply_translation([
        float((low[0] + high[0]) * 0.5),
        float((low[1] + high[1]) * 0.5),
        float(low[2] - thickness * 0.5 - 2.0),
    ])
    return plate


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
    program = report.get("program", {}) or {}
    levels = program.get("connector_levels", {}) or {}
    joint_travels = int(levels.get("J2-retract-travel", 0))
    sweep = (f"{program.get('motion', '—')} {program.get('speed_mm_s', '—')} mm/sec "
             f"{program.get('termination', '')}".strip())
    if joint_travels:
        motion_text = (
            f"扫描段 {sweep}；行与行之间 {joint_travels} 处用「短 L 抬刀 + J 关节运动到预备点 "
            f"+ 短 L 切入」(J 100% CNT100)"
        )
    elif program:
        motion_text = f"扫描段 {sweep}；行与行之间全部为直线（未使用关节运动）"
    else:
        motion_text = "本次结果为纯位姿交付，点与点之间的运动不做承诺"
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
面积 {target_area:.4f} m² · 漏扫原因：{html.escape(reason_text)}</p>
<p>程序运动：<b>{html.escape(motion_text)}</b></p></section>
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


def _runs_root() -> Path:
    """Where every web run lives. One definition, so listing and writing cannot drift."""
    return (Path.cwd() / "output" / "web_runs").resolve()


def _run_meta_path(folder: Path) -> Path:
    return folder / "run_meta.json"


def _read_run_meta(folder: Path) -> dict:
    """The summary written when a run finished, or an empty dict for older runs."""
    path = _run_meta_path(folder)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _run_summary(folder: Path) -> dict:
    """One run folder as the history panel needs it, read back from its own files.

    A run made before `run_meta.json` existed still has to be listable - the folder holds
    the report, the program and the registration file - so every field falls back to
    reading those instead of returning nothing.
    """
    meta = _read_run_meta(folder)
    report_path = folder / "coverage_report.json"
    report: dict = {}
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            report = {}
    programs = sorted(path.stem for path in folder.glob("*.LS"))
    program = str(meta.get("program_name") or (programs[0] if programs else ""))
    registration: dict = {}
    registration_path = folder / "registration_report.json"
    if registration_path.is_file():
        try:
            registration = json.loads(registration_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            registration = {}
    name = program or folder.name
    created = meta.get("created_at")
    if not created:
        created = datetime.fromtimestamp(folder.stat().st_mtime).isoformat(timespec="seconds")
    files = _download_entries(program) if program else [
        {"name": "planning_report.html", "label": "扫描规划报告 (HTML)"},
        {"name": "coverage_report.json", "label": "覆盖报告 JSON"},
    ]
    available = [entry for entry in files if (folder / entry["name"]).is_file()]
    return {
        "run_id": folder.name,
        "program_name": name,
        "created_at": created,
        "coverage_pct": float(meta.get("coverage_pct", report.get("coverage_pct") or 0.0)),
        "point_count": int(meta.get("point_count", report.get("viewpoints") or 0)),
        "target_points": int(meta.get("target_points", report.get("target_points") or 0)),
        "uncovered_points": int(meta.get("uncovered_points",
                                         report.get("uncovered_points") or 0)),
        "face": str(meta.get("face") or report.get("face") or ""),
        "delivery_mode": str(meta.get("delivery_mode") or (
            "stroke" if report.get("stroke_plan", {}).get("enabled") else "pose")),
        "bound": (meta.get("bound")
                  or (report.get("program") or {}).get("bound")),
        "max_points": (meta.get("max_points")
                       if "max_points" in meta
                       else (report.get("program") or {}).get("max_points")),
        "surface_bias": meta.get("surface_bias",
                                 (report.get("derived_parameters") or {}).get("surface_bias")),
        "datum_count": int(meta.get("datum_count", len(registration.get("point_pairs") or []))),
        "measured_frame": str(meta.get("measured_frame")
                              or registration.get("measured_frame") or ""),
        "has_viewer": (folder / "viewpoints_3d.html").is_file(),
        "viewer_url": f"/api/viewer?run_id={folder.name}",
        "failed": (folder / "job_error.txt").is_file(),
        "output_dir": str(folder),
        "downloads": available,
    }


def list_runs(limit: int = 40) -> list[dict]:
    """Every finished run under `output/web_runs`, newest first."""
    root = _runs_root()
    if not root.is_dir():
        return []
    folders = [item for item in root.iterdir() if item.is_dir()]
    folders.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return [_run_summary(folder) for folder in folders[:max(1, int(limit))]]


def _write_run_meta(output: Path, payload: dict) -> Path:
    """Record what this run was, so the history panel does not have to guess later."""
    path = _run_meta_path(output)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return path


def config_defaults() -> dict:
    """The shipped settings the page's controls start from.

    Served rather than hard-coded in the page: the page used to carry its own copy of
    these numbers, so a change to `configs/` silently left the form showing the old
    defaults (merge plan P2-4).
    """
    config = load_config()
    return {
        "max_points": config.target.max_points,
        "coverage_target_pct": config.target.coverage_target_pct,
        "surface_bias": config.target.surface_bias,
        "delivery_mode": "stroke" if config.stroke.enabled else "pose",
        "max_wrist_step_deg": config.stroke.max_wrist_step_deg,
        "roll_mode": config.stroke.roll_mode,
        "connector_level": config.stroke.connector_level,
        "cost_model": config.stroke.cost_model,
        "source": config.source,
        "derived": dict(getattr(config, "derived", {}) or {}),
    }


def _delivery_config(data: dict):
    """The configuration for one job: the shipped file plus the page's delivery mode.

    The two modes differ in the PLANNING unit, not only in the export: "pose" is the v1
    deliverable of isolated stops, so it switches the stroke stage off. Re-sorting the
    export order would not make a stroke plan pose-only.

    A point cap from the page goes through the same `override` path as a config file, so
    it is validated in one place and its derived keys (the tail-repair share) are
    re-derived rather than left at the old profile.
    """
    config = load_config(data.get("config_path"))
    mode = str(data.get("delivery_mode") or "stroke").strip().lower()
    if mode not in ("stroke", "pose"):
        raise ValueError("delivery mode must be 'stroke' or 'pose'")
    if mode == "pose":
        config = override(config, stroke={"enabled": False})
    cap = data.get("max_points")
    if cap is not None and str(cap).strip() != "":
        try:
            numeric = int(cap)
        except (TypeError, ValueError) as error:
            raise ValueError("max_points must be a whole number of program points, "
                             "or left empty for no cap") from error
        config = override(config, target={"max_points": numeric})
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
        }
    return {
        "motion": str(data.get("motion", "J")),
        "speed": str(data.get("speed", "100%")),
        "termination": "FINE",
        "order": True,
    }


def solve_job(data: dict) -> dict:
    """Register uploaded meshes, plan the selected side, and export FANUC files."""
    if not isinstance(data, dict):
        raise ValueError("request body must be a JSON object")
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
    support = _decode_stl(models.get("support", {}), "support/fixture")
    part_raw = _decode_stl_bytes(models.get("part", {}), "part")
    part = _load_stl(part_raw, "part")
    base_payload = models.get("base") or {}
    base_supplied = isinstance(base_payload, dict) and bool(base_payload.get("stl_base64"))
    base = (_decode_stl(base_payload, "base") if base_supplied
            else _synthetic_base(part, support))
    base_reset, support_reset, part_reset = resets

    try:
        prog_name = normalise_prog_name(data.get("prog_name") or "PART01")
    except ValueError as error:
        raise ValueError(str(error)) from error

    face_axis_vehicle = _vector(data.get("scan_face_axis"), "main scan-face axis")
    if not np.isclose(np.linalg.norm(face_axis_vehicle), 1.0, atol=1e-6):
        raise ValueError("main scan-face axis must be one of the six unit vehicle axes")
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
    support_path = output / "support.stl"
    part_path = output / "part.stl"
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
        axis_in_measured = registration.rotation @ face_axis_vehicle
        split = split_sheet(
            station.part,
            band_deg=config.target.band_deg,
            knn=config.target.knn,
            seed=config.seed,
            through_axis=axis_in_measured,
        )
        result = plan(station, config, face="upper", split=split)
        result.report["installation_registration"] = {
            "measured_frame": registration.frame_label,
            "uframe": registration.uframe,
            "rms_error_mm": registration.rms_error_mm,
            "max_error_mm": registration.max_error_mm,
            "cad_face_axis_vehicle": face_axis_vehicle.tolist(),
            "scan_face_axis_measured_frame": axis_in_measured.tolist(),
        }
        write_outputs(result, output)
        options = _export_options(config, result.viewpoints, data)
        stroke_plan = not options["order"]
        contract = export_ls(
            result.viewpoints,
            output,
            prog_name=prog_name,
            uframe=output_uframe,
            utool=int(data.get("utool", 1)),
            **options,
        )
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

    program_block = result.report.get("program") or {}
    _write_run_meta(output, {
        "run_id": run_id,
        "program_name": prog_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "coverage_pct": float(result.report["coverage_pct"]),
        "point_count": len(result.viewpoints),
        "target_points": int(result.report.get("target_points", 0)),
        "uncovered_points": int(result.report.get("uncovered_points", 0)),
        "face": str(result.report.get("face", "")),
        "delivery_mode": "stroke" if stroke_plan else "pose",
        "bound": program_block.get("bound"),
        "max_points": program_block.get("max_points"),
        "surface_bias": (result.report.get("derived_parameters") or {}).get("surface_bias"),
        "datum_count": int(len(registration.cad_points)),
        "measured_frame": registration.frame_label,
        "target_face_mm2": float(result.report.get("target_face_mm2", 0.0)),
        "uncovered_by_reason": result.report.get("uncovered_by_reason", {}),
        "downloads": _download_entries(prog_name),
    })

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
        "base_supplied": base_supplied,
        "max_points": program_block.get("max_points"),
        "bound": program_block.get("bound"),
        "viewpoints": delivered,
        "viewer_url": f"/api/viewer?run_id={run_id}",
        "downloads": _download_entries(prog_name),
    }


class _Handler(BaseHTTPRequestHandler):
    server_version = "ScanplanWeb/1.0"

    def do_GET(self) -> None:
        route = urlsplit(self.path)
        if route.path in ("/", "/index.html"):
            self._send_bytes((_WEB_ROOT / "index.html").read_bytes(),
                             "text/html; charset=utf-8")
            return
        if route.path == "/app.css":
            css = _STATIC_ROOT / "app.css"
            if not css.is_file():
                self.send_error(404)
                return
            self._send_bytes(css.read_bytes(), "text/css; charset=utf-8")
            return
        if route.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        if route.path == "/api/config":
            self._json_response(200, {"ok": True, "result": config_defaults()})
            return
        if route.path == "/api/runs":
            query = parse_qs(route.query)
            try:
                limit = int(query.get("limit", ["40"])[0])
            except ValueError:
                self.send_error(400, "invalid limit")
                return
            self._json_response(200, {"ok": True, "result": {"runs": list_runs(limit)}})
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
        if self.path not in ("/api/register", "/api/solve"):
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > _MAX_REQUEST_BYTES:
                raise ValueError(
                    f"request size must be between 1 and {_MAX_REQUEST_BYTES} bytes"
                )
            data = json.loads(self.rfile.read(length))
            result = _preview_registration(data) if self.path == "/api/register" \
                else solve_job(data)
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


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    """Serve the UI only on loopback; STL files never leave the local machine."""
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("the browser UI only binds to localhost")
    server = ThreadingHTTPServer((host, port), _Handler)
    print(f"scanplan web UI: http://127.0.0.1:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down scanplan web UI", flush=True)
    finally:
        server.server_close()
