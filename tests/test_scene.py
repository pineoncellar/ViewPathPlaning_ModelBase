"""Loading the station and proving the three models really share one frame.

To assert: the rigid transform is validated and applied to the part and the
fixture but never to the base; relative model paths resolve against the station
file rather than the working directory; the alignment self-checks report the
distances a human would measure and reach a verdict; and a station file with a
typo in it is refused, because a silently ignored key is a silently wrong frame.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import write_station_file

from scanplan.scene import load_station, load_station_description, read_transform


def test_transform_must_be_rigid():
    with pytest.raises(ValueError, match="not a rigid transform"):
        read_transform([[2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def test_transform_must_end_in_the_identity_row():
    with pytest.raises(ValueError, match="last row"):
        read_transform([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]])


def test_identity_leaves_the_part_where_it_is(station, folded_plate):
    assert np.allclose(station.part.bounds, folded_plate.bounds, atol=1e-6)


def test_relative_model_paths_resolve_against_the_station_file(
        tmp_path, folded_plate, fixture_box, base_box):
    path = write_station_file(tmp_path, folded_plate, fixture_box, base_box)
    assert path.parent == tmp_path
    description = load_station_description(path)
    assert description.part == tmp_path / "part.stl"


def test_a_part_transform_moves_part_and_fixture_but_never_the_base(
        tmp_path, folded_plate, fixture_box, base_box):
    shifted = [[1, 0, 0, 10.0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    path = write_station_file(tmp_path, folded_plate, fixture_box, base_box,
                              transform_part=shifted, transform_fixture=shifted)
    loaded = load_station(load_station_description(path), cloud_seed=1)
    assert pytest.approx(float(loaded.part.bounds[0][0]), abs=1e-6) \
        == folded_plate.bounds[0][0] + 10.0
    assert pytest.approx(float(loaded.fixture.bounds[0][0]), abs=1e-6) \
        == fixture_box.bounds[0][0] + 10.0
    assert np.allclose(loaded.base.bounds, base_box.bounds, atol=1e-6)


def test_alignment_reports_the_distances_a_human_would_measure(station):
    checks = station.check_alignment(seed=5)
    # the plate is propped 5 mm above the clamp and 60 mm above the base
    assert 3.0 < checks["min_distance_part_to_fixture_mm"] < 7.0
    assert 55.0 < checks["min_distance_part_to_base_mm"] < 65.0
    assert checks["min_distance_fixture_to_base_mm"] < 1.0
    assert checks["fixture_footprint_fits_on_base"] is True
    assert checks["passed"] is True and checks["warnings"] == []


def test_alignment_records_which_axis_the_part_is_flat_in(station):
    checks = station.check_alignment(seed=5)
    assert checks["part_flat_in_z"] is True
    assert checks["part_dominant_axis"] in {"x", "y", "z"}
    assert 0.0 <= checks["part_top_face_share_pct"] <= 100.0


def test_alignment_flags_interpenetration(station_interpenetrating):
    checks = station_interpenetrating.check_alignment(seed=5)
    assert checks["passed"] is False
    assert checks["min_distance_part_to_fixture_mm"] < 1.0
    assert any("interfer" in warning.lower() for warning in checks["warnings"])


def test_alignment_flags_a_fixture_that_dips_below_the_mounting_plane(
        station_fixture_sunk_into_base):
    checks = station_fixture_sunk_into_base.check_alignment(seed=5)
    assert checks["passed"] is False
    assert checks["fixture_points_below_mounting_plane"] > 0
    assert checks["min_distance_fixture_BODY_to_base_mm"] < 5.0
    assert any("mounting plane" in warning for warning in checks["warnings"])


def test_clearance_finds_the_nearest_obstacle(station):
    # 2.5 mm above the clamp and 2.5 mm under the plate; the sampled clouds mean the
    # reported figure is a little larger than the exact 2.5 mm
    gap = float(station.clearance_mm(np.array([[0.0, 0.0, -2.5]]))[0])
    far = float(station.clearance_mm(np.array([[0.0, 0.0, 1000.0]]))[0])
    assert 1.0 < gap < 5.0
    assert far > 100.0


def test_nearest_obstacle_point_is_where_that_distance_was_measured(station):
    probe = np.array([[0.0, 0.0, -2.5]])
    nearest = station.nearest_obstacle_point(probe)
    assert np.allclose(np.linalg.norm(nearest - probe, axis=1),
                       station.clearance_mm(probe), atol=1e-9)


def test_a_typo_in_the_station_file_is_rejected(tmp_path, folded_plate, fixture_box, base_box):
    path = write_station_file(tmp_path, folded_plate, fixture_box, base_box)
    body = json.loads(path.read_text(encoding="utf-8"))
    body["part_tranform"] = body.pop("transform_part")
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown key"):
        load_station_description(path)


def test_a_missing_model_is_reported_with_its_path(
        tmp_path, folded_plate, fixture_box, base_box):
    path = write_station_file(tmp_path, folded_plate, fixture_box, base_box)
    (tmp_path / "fixture.stl").unlink()
    description = load_station_description(path)
    with pytest.raises(FileNotFoundError, match="fixture"):
        load_station(description, cloud_seed=1)


def test_a_station_file_with_a_utf8_bom_is_accepted(tmp_path, folded_plate, fixture_box,
                                                    base_box):
    path = write_station_file(tmp_path, folded_plate, fixture_box, base_box)
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    assert load_station_description(path).part == tmp_path / "part.stl"
