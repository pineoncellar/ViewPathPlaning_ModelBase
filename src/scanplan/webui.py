"""Local-only browser interface for station setup, datum registration, and path export."""

from __future__ import annotations

import base64
import json
import re
import traceback
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from .config import load_config
from .pipeline import plan, write_outputs
from .registration import RigidRegistration, fit_three_points
from .roboguide.adapt import export as export_ls
from .scene import StationDescription, load_station
from .sheet_split import split_sheet

_WEB_ROOT = Path(__file__).resolve().parents[2] / "web"
_MAX_STL_BYTES = 120 * 1024 * 1024
_MAX_REQUEST_BYTES = 380 * 1024 * 1024
_SAFE_RUN_ID = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def _vector(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain exactly three finite numbers")
    return result


def _pose_matrix(pose: dict, name: str) -> np.ndarray:
    from .roboguide.wpr import wpr_to_rot

    position = _vector(pose.get("xyz"), f"{name} reset XYZ")
    wpr = _vector(pose.get("wpr"), f"{name} reset WPR")
    matrix = np.eye(4)
    matrix[:3, :3] = wpr_to_rot(*wpr)
    matrix[:3, 3] = position
    return matrix


def _registration_from_payload(data: dict) -> tuple[RigidRegistration, list[np.ndarray]]:
    cad = np.asarray(data.get("cad_points_vehicle_mm"), dtype=float)
    measured = np.asarray(data.get("measured_points_mm"), dtype=float)
    if cad.shape != (3, 3) or measured.shape != (3, 3):
        raise ValueError("pick three CAD points and enter three matching measured XYZ points")
    registration = fit_three_points(
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

    base = _decode_stl(data.get("models", {}).get("base", {}), "base")
    support = _decode_stl(data.get("models", {}).get("support", {}), "support/fixture")
    part_raw = _decode_stl_bytes(data.get("models", {}).get("part", {}), "part")
    _load_stl(part_raw, "part")
    base_reset, support_reset, part_reset = resets

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

    run_id = str(data.get("run_id") or (
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    ))
    if not _SAFE_RUN_ID.fullmatch(run_id):
        raise ValueError("run id may contain only letters, digits, underscores, and hyphens")
    output = (Path.cwd() / "output" / "web_runs" / run_id).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output folder already exists and is not empty: {output}")
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
        config = load_config(data.get("config_path"))
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
        contract = export_ls(
            result.viewpoints,
            output,
            prog_name="PART01",
            uframe=output_uframe,
            utool=int(data.get("utool", 1)),
            motion=str(data.get("motion", "J")),
            speed=str(data.get("speed", "100%")),
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
        (output / "PART01_frame.json").write_text(
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
            "outputs": ["PART01.LS", "PART01_points.csv", "viewpoints.json"],
            "warning": (
                "This is geometric planning only; verify robot reachability, collision, "
                "UFRAME, and the scanner TCP in ROBOGUIDE before running."
            ),
        }
        (output / "registration_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except Exception:
        (output / "job_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise

    return {
        "run_id": run_id,
        "output_dir": str(output),
        "program": str(output / "PART01.LS"),
        "point_count": len(result.viewpoints),
        "coverage_pct": result.report["coverage_pct"],
        "scan_face_axis_measured_frame": scan_axis_target.tolist(),
        "registration_rms_error_mm": registration.rms_error_mm,
        "registration_max_error_mm": registration.max_error_mm,
    }


class _Handler(BaseHTTPRequestHandler):
    server_version = "ScanplanWeb/1.0"

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            page = (_WEB_ROOT / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)
            return
        if self.path.startswith("/api/file?"):
            from urllib.parse import parse_qs, urlsplit

            query = parse_qs(urlsplit(self.path).query)
            run_id = query.get("run_id", [""])[0]
            name = query.get("name", [""])[0]
            allowed = {
                "PART01.LS",
                "PART01_points.csv",
                "PART01_frame.json",
                "registration_report.json",
                "coverage_report.json",
                "summary.txt",
                "viewpoints.json",
                "viewpoints.csv",
            }
            if not _SAFE_RUN_ID.fullmatch(run_id) or name not in allowed:
                self.send_error(400, "invalid output file")
                return
            path = Path.cwd() / "output" / "web_runs" / run_id / name
            if not path.is_file():
                self.send_error(404)
                return
            content = path.read_bytes()
            content_type = "application/octet-stream" if name.endswith(".LS") \
                else "application/json; charset=utf-8" if name.endswith(".json") \
                else "text/plain; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Content-Disposition", f'attachment; filename="{name}"')
            self.end_headers()
            self.wfile.write(content)
            return
        self.send_error(404)

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
