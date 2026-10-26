"""Local web-interface payload validation and frame math."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

import scanweb
from scanweb import (
    _available_run_id,
    _decode_stl,
    _delete_history_run,
    _history_run_payload,
    _history_runs,
    _init_history,
    _preview_registration,
    _record_history_run,
    _registration_from_payload,
    _sync_input_part,
    _synthetic_base,
)


def valid_payload():
    vehicle = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 80.0, 0.0]])
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = np.array([500.0, -250.0, 900.0])
    measured = vehicle @ rotation.T + translation
    return {
        "cad_points_vehicle_mm": vehicle.tolist(),
        "measured_points_mm": measured.tolist(),
        "measured_frame": "robot_uframe_1",
        "uframe": 1,
        "reset_poses": {
            name: {"xyz": [0, 0, 0], "wpr": [0, 0, 0]}
            for name in ("base", "support", "part")
        },
    }


def test_the_page_defaults_come_from_the_shipped_configuration():
    """`GET /api/config` is the single source for "which deliverable is the default"."""
    config = scanweb._public_config()
    assert config["delivery_mode"] == "stroke"
    assert config["surface_bias"] == pytest.approx(0.5)
    assert config["robot_enabled"] is False
    assert config["motion"] in ("J", "L")
    assert set(config["parameter_sources"]) == {
        "stroke.min_gain_mm2_per_step", "stroke.min_stroke_mm", "stroke.seed_dirs",
        "stroke.order_axis_weight_mm_per_deg", "stroke.parallel_weight",
        "stroke.repair_budget_share"}


def test_preview_registration_reports_exact_measured_transform():
    result = _preview_registration(valid_payload())

    assert result["measured_frame"] == "robot UFRAME 1"
    assert result["rms_error_mm"] == pytest.approx(0.0, abs=1e-10)
    assert result["max_error_mm"] == pytest.approx(0.0, abs=1e-10)
    assert result["translation_vehicle_origin_in_measured_mm"] == pytest.approx(
        [500, -250, 900]
    )


def test_registration_payload_requires_three_matching_points():
    payload = valid_payload()
    payload["cad_points_vehicle_mm"] = payload["cad_points_vehicle_mm"][:2]

    with pytest.raises(ValueError, match="pick three CAD points"):
        _registration_from_payload(payload)


def test_four_point_registration_fits_the_same_rigid_transform():
    payload = valid_payload()
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = np.array([500.0, -250.0, 900.0])
    vehicle = np.array([
        [0.0, 0.0, 0.0], [100.0, 0.0, 0.0],
        [0.0, 80.0, 0.0], [40.0, 30.0, 25.0],
    ])
    payload["cad_points_vehicle_mm"] = vehicle.tolist()
    payload["measured_points_mm"] = (vehicle @ rotation.T + translation).tolist()

    result = _preview_registration(payload)

    assert len(result["point_errors_mm"]) == 4
    assert result["rms_error_mm"] == pytest.approx(0.0, abs=1e-10)
    assert result["translation_vehicle_origin_in_measured_mm"] == pytest.approx(
        [500, -250, 900]
    )


def test_synthetic_base_sits_below_the_part_and_fixture():
    part = trimesh.creation.box(extents=[100.0, 80.0, 2.0])
    part.apply_translation([0.0, 0.0, 50.0])
    fixture = trimesh.creation.box(extents=[140.0, 120.0, 40.0])
    fixture.apply_translation([0.0, 0.0, 20.0])

    base = _synthetic_base(part, fixture)

    assert base.bounds[1][2] < fixture.bounds[0][2]
    assert base.bounds[1][2] < part.bounds[0][2]
    assert base.extents[0] >= fixture.extents[0]
    assert base.extents[1] >= fixture.extents[1]


def test_reset_poses_are_parsed_as_vehicle_frame_rigid_transforms():
    payload = valid_payload()
    payload["reset_poses"]["part"] = {
        "xyz": [10.0, 20.0, 30.0],
        "wpr": [0.0, 0.0, 90.0],
    }

    _, transforms = _registration_from_payload(payload)

    assert transforms[2][:3, 3] == pytest.approx([10, 20, 30])
    assert np.allclose(transforms[2][:3, :3], [[0, -1, 0], [1, 0, 0], [0, 0, 1]])


def test_reset_poses_default_to_stl_vehicle_coordinates():
    payload = valid_payload()
    payload.pop("reset_poses")

    _, transforms = _registration_from_payload(payload)

    assert len(transforms) == 3
    assert all(np.array_equal(transform, np.eye(4)) for transform in transforms)


def test_stl_upload_is_decoded_as_a_triangular_mesh():
    mesh = trimesh.creation.box(extents=[20.0, 30.0, 40.0])
    encoded = base64.b64encode(trimesh.exchange.stl.export_stl(mesh)).decode("ascii")

    loaded = _decode_stl({"stl_base64": encoded}, "part")

    assert len(loaded.faces) == len(mesh.faces)
    assert loaded.extents == pytest.approx(mesh.extents)


def test_stl_upload_rejects_bad_base64():
    with pytest.raises(ValueError, match="base64"):
        _decode_stl({"stl_base64": "!invalid!"}, "part")


def test_uploaded_part_replaces_canonical_input_part(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "part.stl").write_bytes(b"previous part")
    uploaded_stl = b"uploaded part"

    part_path = _sync_input_part(uploaded_stl, input_dir)

    assert part_path == input_dir / "part.stl"
    assert part_path.read_bytes() == uploaded_stl
    assert list(input_dir.iterdir()) == [part_path]


def test_the_page_offers_both_delivery_modes_with_sweeps_as_the_default():
    page = (Path(scanweb.__file__).parent / "index.html").read_text(encoding="utf-8")

    assert 'id="scanMode"' in page
    assert 'value="stroke" selected' in page
    assert 'value="pose"' in page


def test_delivery_mode_reaches_the_planner_not_only_the_exporter():
    from scanweb import _delivery_config

    assert _delivery_config({}).stroke.enabled is True, "sweeps are the shipped default"
    assert _delivery_config({"delivery_mode": "stroke"}).stroke.enabled is True
    assert _delivery_config({"delivery_mode": "pose"}).stroke.enabled is False
    with pytest.raises(ValueError, match="delivery mode"):
        _delivery_config({"delivery_mode": "fast"})


def test_a_stroke_plan_is_exported_as_delivered_and_flown_as_configured():
    from types import SimpleNamespace

    from scanplan.config import load_config
    from scanweb import _export_options

    cfg = load_config()
    stroke = [SimpleNamespace(stroke_id=3), SimpleNamespace(stroke_id=3)]
    pose = [SimpleNamespace(stroke_id=None)]

    options = _export_options(cfg, stroke, {})
    assert options["order"] is False, "re-sorting would split each sweep in two"
    assert options["motion"] == cfg.motion.type
    assert options["termination"] == cfg.motion.termination
    assert options["speed"].endswith("mm/sec")

    legacy = _export_options(cfg, pose, {})
    assert legacy["order"] is True
    assert legacy["termination"] == "FINE"


def test_repeated_run_name_gets_a_unique_folder_instead_of_failing(tmp_path):
    assert _available_run_id("AB1", tmp_path) == "AB1"

    (tmp_path / "AB1").mkdir()
    assert _available_run_id("AB1", tmp_path) == "AB1"

    (tmp_path / "AB1" / "summary.txt").write_text("existing", encoding="utf-8")
    first = _available_run_id("AB1", tmp_path)
    assert first.startswith("AB1_")
    assert first != "AB1"
    assert len(first) <= 64
    assert not (tmp_path / first).exists()

    (tmp_path / first).mkdir()
    (tmp_path / first / "summary.txt").write_text("existing", encoding="utf-8")
    second = _available_run_id("AB1", tmp_path)
    assert second != first
    assert second.startswith("AB1_")


def history_result(run_dir):
    return {
        "run_id": "run_001",
        "output_dir": str(run_dir),
        "program_name": "PART01",
        "point_count": 5,
        "coverage_pct": 91.5,
        "target_points": 100,
        "uncovered_points": 8,
        "target_face_mm2": 1000.0,
        "face": "upper",
        "viewer_url": "/api/viewer?run_id=run_001",
        "downloads": [{"name": "PART01.LS", "label": "FANUC LS 程序"}],
        "viewpoints": [{"order": 1, "position": [1.0, 2.0, 3.0]}],
    }


def test_history_round_trip_keeps_the_full_payload(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _init_history()
    result = history_result(tmp_path / "output" / "web_runs" / "run_001")

    _record_history_run(result, "  侧围 01  ")

    runs = _history_runs()
    assert len(runs) == 1
    assert runs[0]["run_id"] == "run_001"
    assert runs[0]["display_name"] == "侧围 01"
    assert runs[0]["prog_name"] == "PART01"
    assert runs[0]["status"] == "completed"
    assert runs[0]["coverage_pct"] == pytest.approx(91.5)
    assert _history_run_payload("run_001") == result


def test_history_delete_removes_the_record_and_run_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _init_history()
    run_dir = tmp_path / "output" / "web_runs" / "run_001"
    run_dir.mkdir(parents=True)
    (run_dir / "PART01.LS").write_text("program", encoding="utf-8")
    _record_history_run(history_result(run_dir), "侧围 01")

    assert _delete_history_run("run_001") is True
    assert _history_runs() == []
    assert _history_run_payload("run_001") is None
    assert not run_dir.exists()


def test_history_is_empty_on_a_fresh_database(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _init_history()

    assert _history_runs() == []
    assert _history_run_payload("missing") is None
    assert _delete_history_run("missing") is False


# --- re-localisation after a plan -----------------------------------------
#
# The page lets an operator correct field datum coordinates once a run exists.  The
# server side of that is `relocate_job`: it reads the run's own meshes back and replans
# instead of asking for an upload, and it must not double-apply the installation
# transform to the base.

DATUM_PAIRS = [
    {"vehicle_mm": [0.0, 0.0, 0.0], "measured_mm": [1.0, 2.0, 3.0],
     "residual_mm": [0.0, 0.0, 0.25]},
    {"vehicle_mm": [100.0, 0.0, 0.0], "measured_mm": [101.0, 2.0, 3.0],
     "residual_mm": [0.0, 0.0, 0.1]},
    {"vehicle_mm": [0.0, 80.0, 0.0], "measured_mm": [1.0, 82.0, 3.0],
     "residual_mm": [0.0, 0.0, 0.05]},
]


def make_run_folder(root: Path, run_id: str, *, with_meshes: bool = True) -> Path:
    run_dir = root / "output" / "web_runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "registration_report.json").write_text(json.dumps({
        "measured_frame": "robot UFRAME 1",
        "uframe": 1,
        "max_registration_error_mm": 2.0,
        "part_scan_face_axis_vehicle": [0.0, 0.0, 1.0],
        "point_pairs": DATUM_PAIRS,
    }), encoding="utf-8")
    if with_meshes:
        raw = trimesh.creation.box(extents=[10.0, 10.0, 10.0]).export(file_type="stl")
        for name in ("part", "support", "base_vehicle"):
            (run_dir / f"{name}.stl").write_bytes(raw)
    return run_dir


def test_the_page_offers_re_localisation_after_a_plan():
    root = Path(scanweb.__file__).parent
    page = (root / "index.html").read_text(encoding="utf-8")
    script = (root / "app.js").read_text(encoding="utf-8")

    assert 'id="relocateButton"' in page
    assert 'id="relocateRows"' in page
    assert 'id="relocateLimit"' in page
    assert "/api/relocate" in script
    assert "syncSetupFromResult" in script
    assert "不重新规划" in page


def test_vehicle_frame_base_inverts_the_installation_transform(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    vehicle = trimesh.creation.box(extents=[20.0, 30.0, 10.0])
    vehicle.apply_translation([100.0, 50.0, 5.0])
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = np.array([500.0, -250.0, 900.0])
    installed = vehicle.copy()
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    installed.apply_transform(transform)
    (run_dir / "base.stl").write_bytes(installed.export(file_type="stl"))
    (run_dir / "registration_report.json").write_text(json.dumps({
        "rotation_vehicle_to_measured": rotation.tolist(),
        "translation_vehicle_origin_in_measured_mm": translation.tolist(),
        "reset_poses_vehicle": {
            "base": {"xyz": [0.0, 0.0, 0.0], "wpr": [0.0, 0.0, 0.0]}
        },
    }), encoding="utf-8")

    raw = scanweb._vehicle_frame_base(run_dir)

    assert raw is not None
    recovered = trimesh.load(io.BytesIO(raw), file_type="stl")
    assert recovered.bounds == pytest.approx(vehicle.bounds, abs=1e-3)


def test_run_datums_fall_back_to_the_registration_report(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make_run_folder(tmp_path, "run_009", with_meshes=False)

    datums = scanweb._run_datums("run_009", {})

    assert datums["cad_points_vehicle_mm"] == [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0],
                                               [0.0, 80.0, 0.0]]
    assert datums["measured_points_mm"] == [[1.0, 2.0, 3.0], [101.0, 2.0, 3.0],
                                            [1.0, 82.0, 3.0]]
    assert datums["point_errors_mm"] == pytest.approx([0.25, 0.1, 0.05])
    assert datums["max_registration_error_mm"] == pytest.approx(2.0)
    assert datums["uframe"] == 1
    assert datums["scan_face_axis_vehicle"] == [0.0, 0.0, 1.0]


def test_a_stored_result_without_datums_still_gets_them_for_editing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make_run_folder(tmp_path, "run_009", with_meshes=False)

    merged = scanweb._augment_datums({"run_id": "run_009", "program_name": "PART01"},
                                     "run_009")

    assert len(merged["cad_points_vehicle_mm"]) == 3
    assert merged["measured_points_mm"][1] == [101.0, 2.0, 3.0]
    assert merged["error_limit_mm"] == pytest.approx(2.0)
    assert merged["program_name"] == "PART01"


def make_relocatable_run(root: Path, run_id: str, *, rotation=None, translation=None,
                         viewpoints=None) -> Path:
    """A finished run folder with everything a relocation reads, and nothing else."""
    run_dir = root / "output" / "web_runs" / run_id
    run_dir.mkdir(parents=True)
    rotation = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
    translation = (np.zeros(3) if translation is None
                   else np.asarray(translation, dtype=float))
    raw = trimesh.creation.box(extents=[10.0, 10.0, 10.0]).export(file_type="stl")
    for name in ("part", "support", "base_vehicle"):
        (run_dir / f"{name}.stl").write_bytes(raw)
    (run_dir / "viewpoints.json").write_text(json.dumps(viewpoints or [
        {"id": 0, "order": 0, "stroke_id": None, "role": "coverage",
         "position": [10.0, 20.0, 30.0], "axis": [0.0, 0.0, 1.0], "roll_deg": 15.0,
         "n_points": 100, "area_mm2": 50.0, "mean_incidence_deg": 3.0,
         "max_incidence_deg": 8.0, "tracker_ok": True, "source": "greedy",
         "config": "N U T, 0, 0, 0", "branch": "NUT",
         "joints_deg": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "j5_margin_deg": 20.0},
        {"id": 1, "order": 1, "stroke_id": 0, "role": "coverage",
         "position": [40.0, 20.0, 30.0], "axis": [0.0, 0.0, 1.0], "roll_deg": 15.0,
         "n_points": 90, "area_mm2": 45.0, "mean_incidence_deg": 3.0,
         "max_incidence_deg": 8.0, "tracker_ok": True, "source": "stroke"},
    ]), encoding="utf-8")
    (run_dir / "registration_report.json").write_text(json.dumps({
        "measured_frame": "robot UFRAME 1",
        "uframe": 1,
        "rotation_vehicle_to_measured": rotation.tolist(),
        "translation_vehicle_origin_in_measured_mm": translation.tolist(),
        "max_registration_error_mm": 2.0,
        "part_scan_face_axis_vehicle": [0.0, 0.0, 1.0],
        "reset_poses_vehicle": {
            name: {"xyz": [0.0, 0.0, 0.0], "wpr": [0.0, 0.0, 0.0]}
            for name in ("base", "support", "part")
        },
        "point_pairs": DATUM_PAIRS,
    }), encoding="utf-8")
    return run_dir


def read_moved_viewpoints(result: dict) -> list[dict]:
    path = Path(result["output_dir"]) / "viewpoints.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_relocation_moves_the_planned_path_without_replanning(tmp_path, monkeypatch):
    """A datum correction must translate the delivered path, not plan a new one."""
    monkeypatch.chdir(tmp_path)
    make_relocatable_run(tmp_path, "run_020")

    def explode(data, progress=None):
        raise AssertionError("a relocation must not re-plan")

    monkeypatch.setattr(scanweb, "solve_job", explode)

    result = scanweb.relocate_job({
        "run_id": "run_020",
        "measured_points_mm": [[10.0, 0.0, 0.0], [110.0, 0.0, 0.0], [10.0, 80.0, 0.0]],
        "prog_name": "PART03",
    })

    assert result["relocalised"] is True
    assert result["relocated_from"] == "run_020"
    assert result["run_id"] == "run_020_reloc"
    assert result["point_count"] == 2
    assert result["shift_mm"] == pytest.approx(10.0)

    moved = read_moved_viewpoints(result)
    assert [record["position"] for record in moved] == [[20.0, 20.0, 30.0],
                                                      [50.0, 20.0, 30.0]]
    assert [record["order"] for record in moved] == [0, 1]
    assert [record["roll_deg"] for record in moved] == [15.0, 15.0]
    assert moved[0]["axis"] == [0.0, 0.0, 1.0]
    # Joint evidence was judged in the old frame; it must not survive the move.
    assert moved[0]["config"] is None
    assert moved[0]["branch"] is None
    assert moved[0]["joints_deg"] is None
    assert (Path(result["output_dir"]) / "PART03.LS").is_file()
    report = json.loads((Path(result["output_dir"]) / "registration_report.json")
                        .read_text(encoding="utf-8"))
    assert report["relocalisation"]["replanned"] is False
    assert report["relocalisation"]["from_run"] == "run_020"


def test_relocation_rotates_the_path_with_the_new_frame(tmp_path, monkeypatch):
    """A rotated installation must rotate the poses, not just their positions."""
    monkeypatch.chdir(tmp_path)
    make_relocatable_run(tmp_path, "run_021")
    # new registration = 90 deg about Z; measured points are the CAD points rotated.
    measured = [[0.0, 0.0, 0.0], [0.0, 100.0, 0.0], [-80.0, 0.0, 0.0]]

    result = scanweb.relocate_job({
        "run_id": "run_021",
        "measured_points_mm": measured,
        "prog_name": "PART04",
    })

    moved = read_moved_viewpoints(result)
    assert moved[0]["position"] == pytest.approx([-20.0, 10.0, 30.0], abs=1e-4)
    assert moved[1]["position"] == pytest.approx([-20.0, 40.0, 30.0], abs=1e-4)
    # Axis +Z is invariant under a rotation about Z, and roll is a rotation about the axis.
    assert moved[0]["axis"] == pytest.approx([0.0, 0.0, 1.0], abs=1e-9)
    assert moved[0]["roll_deg"] == pytest.approx(15.0)
    assert result["measured_frame"] == "robot UFRAME 1"


def test_relocation_refuses_a_correction_beyond_the_error_limit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make_relocatable_run(tmp_path, "run_022")

    with pytest.raises(ValueError, match="max error"):
        scanweb.relocate_job({
            "run_id": "run_022",
            # One datum measured 20 mm too far along an edge: a rigid fit cannot absorb it.
            "measured_points_mm": [[10.0, 0.0, 0.0], [110.0, 0.0, 0.0], [10.0, 100.0, 0.0]],
            "max_registration_error_mm": 0.5,
        })


def test_relocate_rejects_an_unknown_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="run not found"):
        scanweb.relocate_job({"run_id": "missing", "measured_points_mm": []})


def test_relocate_requires_one_field_point_per_datum(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make_relocatable_run(tmp_path, "run_011")

    with pytest.raises(ValueError, match="same"):
        scanweb.relocate_job({
            "run_id": "run_011",
            "measured_points_mm": [[0.0, 0.0, 0.0]],
        })


# --- progress board --------------------------------------------------------
#
# The page is told where a run is by polling this board.  It is deliberately small, so
# the two properties that matter are pinned here: percentages never go backwards, and a
# token that stops publishing is cleaned up instead of growing the dict forever.


def test_progress_snapshot_never_rewinds():
    scanweb._PROGRESS.clear()
    scanweb._publish_progress("tok_progress", "occlusion", 0.42, "120/300 poses")
    scanweb._publish_progress("tok_progress", "occlusion", 0.10, "late report")

    snapshot = scanweb._progress_snapshot("tok_progress")

    assert snapshot["stage"] == "occlusion"
    assert snapshot["pct"] == pytest.approx(42.0)
    assert snapshot["detail"] == "late report"
    assert scanweb._progress_snapshot("never_published") is None
    assert scanweb._progress_snapshot("bad id") is None


def test_progress_entries_expire():
    scanweb._PROGRESS.clear()
    scanweb._publish_progress("stale_run", "export", 0.9)
    with scanweb._PROGRESS_LOCK:
        scanweb._PROGRESS["stale_run"]["updated"] -= scanweb._PROGRESS_TTL_S + 1

    scanweb._publish_progress("fresh_run", "export", 0.5)

    assert scanweb._progress_snapshot("stale_run") is None
    assert scanweb._progress_snapshot("fresh_run") is not None


def test_progress_ids_must_be_plain_identifiers():
    assert scanweb._progress_token({"progress_id": "abc-123_XY"}) == "abc-123_XY"
    assert scanweb._progress_token({"progress_id": "abc 123"}) is None
    assert scanweb._progress_token({"progress_id": "../escape"}) is None
    assert scanweb._progress_token({}) is None


def test_the_page_shows_server_stages_instead_of_a_time_curve():
    root = Path(scanweb.__file__).parent
    page = (root / "index.html").read_text(encoding="utf-8")
    script = (root / "app.js").read_text(encoding="utf-8")

    assert 'data-stage="strokes"' in page
    assert 'data-stage="report"' in page
    assert "/api/progress" in script
    assert "applyServer" in script
    # The old estimator saturated at 99% after a minute and looked stuck; it must not
    # come back as the primary source.
    assert "Math.exp(-seconds / 26)" not in script
