"""Local web-interface payload validation and frame math."""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import pytest
import trimesh

import scanweb
from scanweb import (
    _STATIC_ROOT,
    _available_run_id,
    _decode_stl,
    _preview_registration,
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


def test_web_page_stylesheet_ships_with_the_package():
    page = (Path(scanweb.__file__).parent / "index.html").read_text(encoding="utf-8")
    stylesheet = _STATIC_ROOT / "app.css"

    assert 'href="/app.css"' in page
    assert stylesheet.is_file(), "index.html links /app.css but the file is missing"
    assert ".app-shell" in stylesheet.read_text(encoding="utf-8")


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
