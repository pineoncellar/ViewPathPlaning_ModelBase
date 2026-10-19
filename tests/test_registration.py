"""Three-point registration of CAD paths to a robot-base frame."""

from __future__ import annotations

import json

import numpy as np
import pytest
import trimesh

import scanplan.cli as cli
from scanplan.cli import main
from scanplan.datum_picker import snap_to_reference
from scanplan.registration import fit_three_points, load_registration, parse_measured_points


@pytest.fixture
def viewpoints_json(tmp_path):
    path = tmp_path / "viewpoints.json"
    path.write_text(json.dumps([
        {"id": 0, "position": [10.0, 20.0, 300.0], "axis": [0.0, 0.0, -1.0],
         "roll_deg": 0.0},
        {"id": 1, "position": [40.0, 20.0, 300.0], "axis": [0.1, 0.0, -0.995],
         "roll_deg": 90.0},
    ]), encoding="utf-8")
    return path


def test_three_point_fit_recovers_a_rigid_transform():
    cad = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 80.0, 0.0]])
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = np.array([500.0, -250.0, 900.0])
    measured = cad @ rotation.T + translation

    result = fit_three_points(cad, measured)

    assert result.rotation == pytest.approx(rotation)
    assert result.translation == pytest.approx(translation)
    assert result.rms_error_mm == pytest.approx(0.0, abs=1e-10)
    assert result.max_error_mm == pytest.approx(0.0, abs=1e-10)


def test_three_point_fit_rejects_collinear_datums():
    collinear = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match="collinear"):
        fit_three_points(collinear, collinear)


def test_measured_points_parser_requires_three_non_collinear_xyz_values():
    points = parse_measured_points("0,0,0;100,0,0;0,80,0")
    assert np.array_equal(points, [[0, 0, 0], [100, 0, 0], [0, 80, 0]])
    with pytest.raises(ValueError, match="exactly three"):
        parse_measured_points("0,0,0;1,1,1")
    with pytest.raises(ValueError, match="collinear"):
        parse_measured_points("0,0,0;10,0,0;20,0,0")


def test_picker_snaps_click_near_reference_to_exact_cad_point():
    references = np.array([
        [100.0, 15.0, 200.0],
        [500.0, 25.0, 600.0],
    ])

    snapped = snap_to_reference([512.0, 610.0], references, radius_mm=30.0)

    assert snapped == pytest.approx(references[1])
    assert snap_to_reference([550.0, 600.0], references, radius_mm=30.0) is None


def test_registration_file_requires_exactly_three_pairs(tmp_path):
    path = tmp_path / "registration.json"
    path.write_text(json.dumps({
        "coordinate_system": "robot_base_mm",
        "point_pairs": [{"cad_mm": [0, 0, 0], "robot_base_mm": [1, 2, 3]}],
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly three"):
        load_registration(path)


def test_registration_reads_existing_uframe_point_array_format(tmp_path):
    points = [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 80.0, 0.0]]
    path = tmp_path / "registration.json"
    path.write_text(json.dumps({
        "measured_coordinate_frame": "robot UFRAME 1",
        "units": "mm",
        "cad_points_mm": points,
        "measured_points_mm": points,
    }), encoding="utf-8")

    result = load_registration(path)

    assert result.target_frame == "robot_uframe_1"
    assert result.uframe == 1
    assert result.max_error_mm == pytest.approx(0.0, abs=1e-10)


def test_registration_file_reads_uframe_pairs_schema(tmp_path):
    points = [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 80.0, 0.0]]
    path = tmp_path / "registration.json"
    path.write_text(json.dumps({
        "coordinate_system": "robot_uframe_1_mm",
        "point_pairs": [
            {"cad_mm": point, "robot_uframe_1_mm": point} for point in points
        ],
    }), encoding="utf-8")

    result = load_registration(path)

    assert result.uframe == 1
    assert result.max_error_mm == pytest.approx(0.0, abs=1e-10)


def test_ls_registers_viewpoints_to_robot_base(tmp_path, viewpoints_json, capsys):
    cad = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 80.0, 0.0]])
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = np.array([500.0, -250.0, 900.0])
    measured = cad @ rotation.T + translation
    registration_path = tmp_path / "registration.json"
    registration_path.write_text(json.dumps({
        "coordinate_system": "robot_base_mm",
        "point_pairs": [
            {"cad_mm": source.tolist(), "robot_base_mm": target.tolist()}
            for source, target in zip(cad, measured, strict=True)
        ],
    }), encoding="utf-8")
    output = tmp_path / "registered"

    assert main([
        "ls", "--viewpoints", str(viewpoints_json), "--prog-name", "REGISTERED",
        "--registration", str(registration_path), "--out", str(output),
    ]) == 0

    program = (output / "REGISTERED.LS").read_text(encoding="ascii")
    assert "UFRAME_NUM=0 ;" in program
    points = (output / "REGISTERED_points.csv").read_text(encoding="utf-8")
    # CAD [10,20,300] maps to robot-base [480, -240, 1200].
    assert "480.0,-240.0,1200.0" in points
    report = json.loads((output / "registration_report.json").read_text(encoding="utf-8"))
    assert report["rms_error_mm"] == pytest.approx(0.0, abs=1e-10)
    assert report["point_pairs"][0]["robot_base_mm"] == pytest.approx(measured[0])
    frame = json.loads((output / "REGISTERED_frame.json").read_text(encoding="utf-8"))
    assert frame["uframe_utool"]["uframe"] == 0
    assert frame["placement"]["registration"]["report"] == "registration_report.json"
    assert "max error 0.000 mm" in capsys.readouterr().out


def test_pick_datums_writes_paired_registration_without_opening_viewer(
        tmp_path, monkeypatch, capsys):
    part = trimesh.creation.box(extents=[100.0, 2.0, 80.0])
    part.export(tmp_path / "part.stl")
    station = tmp_path / "station.json"
    identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    station.write_text(json.dumps({
        "part": "part.stl", "fixture": "part.stl", "base": "part.stl",
        "transform_part": identity,
    }), encoding="utf-8")
    cad = np.array([[-30.0, -1.0, -20.0], [30.0, -1.0, -20.0], [-30.0, -1.0, 20.0]])
    measured = cad + np.array([200.0, 100.0, 300.0])
    monkeypatch.setattr(
        cli, "pick_cad_datums", lambda _part, reference_points=None: cad
    )
    output = tmp_path / "registration.json"

    assert main([
        "pick-datums", "--station", str(station),
        "--measured-points", ";".join(",".join(map(str, point)) for point in measured),
        "--coordinate-system", "robot_uframe_1_mm", "--out", str(output),
    ]) == 0

    body = json.loads(output.read_text(encoding="utf-8"))
    assert body["coordinate_system"] == "robot_uframe_1_mm"
    assert body["point_pairs"][0]["cad_mm"] == pytest.approx(cad[0])
    assert body["point_pairs"][0]["robot_uframe_1_mm"] == pytest.approx(measured[0])
    assert "maximum 0.000 mm" in capsys.readouterr().out


def test_ls_registration_preserves_the_measured_uframe(tmp_path, viewpoints_json):
    cad = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 80.0, 0.0]])
    measured = cad + np.array([100.0, 200.0, 300.0])
    registration_path = tmp_path / "registration.json"
    registration_path.write_text(json.dumps({
        "measured_coordinate_frame": "robot UFRAME 1",
        "units": "mm",
        "cad_points_mm": cad.tolist(),
        "measured_points_mm": measured.tolist(),
    }), encoding="utf-8")
    output = tmp_path / "uframe1"

    assert main([
        "ls", "--viewpoints", str(viewpoints_json), "--prog-name", "UFRAME1",
        "--registration", str(registration_path), "--out", str(output),
    ]) == 0

    program = (output / "UFRAME1.LS").read_text(encoding="ascii")
    assert "UFRAME_NUM=1 ;" in program
    contract = json.loads((output / "UFRAME1_frame.json").read_text(encoding="utf-8"))
    assert contract["uframe_utool"]["uframe"] == 1
    report = json.loads((output / "registration_report.json").read_text(encoding="utf-8"))
    assert "robot UFRAME 1 coordinates" in report["target_frame"]
    assert report["point_pairs"][0]["robot_uframe_1_mm"] == pytest.approx(measured[0])
    assert "robot base" not in report["warning"]
    assert "UFRAME 1 axes" in contract["frame"]["axes"]
    assert "NOT the CAD origin" in contract["uframe_utool"]["uframe_note"]


def test_ls_refuses_registration_errors_above_limit(
        tmp_path, viewpoints_json, capsys):
    registration_path = tmp_path / "registration.json"
    registration_path.write_text(json.dumps({
        "coordinate_system": "robot_base_mm",
        "point_pairs": [
            {"cad_mm": [0, 0, 0], "robot_base_mm": [0, 0, 0]},
            {"cad_mm": [100, 0, 0], "robot_base_mm": [100, 0, 0]},
            {"cad_mm": [0, 100, 0], "robot_base_mm": [0, 100, 5]},
        ],
    }), encoding="utf-8")
    output = tmp_path / "rejected"

    with pytest.raises(SystemExit) as caught:
        main([
            "ls", "--viewpoints", str(viewpoints_json), "--prog-name", "REJECTED",
            "--registration", str(registration_path),
            "--max-registration-error-mm", "0.01", "--out", str(output),
        ])

    assert caught.value.code == 2
    assert "exceeding" in capsys.readouterr().err
    assert not output.exists()
