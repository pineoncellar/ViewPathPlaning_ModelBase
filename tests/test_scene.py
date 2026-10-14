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
from conftest import box, write_station_file

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
    assert checks["part_points_inside_fixture"] > 0
    assert checks["deepest_part_point_inside_fixture_mm"] > 0.2
    assert any("INSIDE the fixture" in warning for warning in checks["warnings"])


def test_a_hairline_gap_is_reported_but_not_called_interference(
        tmp_path, folded_plate, base_box):
    # a stamped part resting on its supports has a gap of a few hundredths of a
    # millimetre: real, and nothing a point cloud can call interference. The strip's
    # underside sits at z = -0.35, so this clamp's top face leaves 0.05 mm.
    from scanplan.scene import load_station, load_station_description

    hairline = box(-60.0, 60.0, -60.0, 60.0, -60.0, -0.40)
    path = write_station_file(tmp_path, folded_plate, hairline, base_box)
    station_with_gap = load_station(load_station_description(path), cloud_seed=3)
    checks = station_with_gap.check_alignment(seed=5)
    assert 0.0 < checks["min_distance_part_to_fixture_mm"] < 0.5
    assert checks["part_points_inside_fixture"] == 0
    assert checks["passed"] is True


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


def test_a_mounting_plane_from_the_station_file_is_used(tmp_path, folded_plate, fixture_box,
                                                        base_box):
    # the base's own highest point is not the mounting plane on a real cell: the fixture
    # can stand on a shelf well below it, which is why this is station data
    path = write_station_file(tmp_path, folded_plate, fixture_box, base_box,
                              extra={"mounting_plane_z_mm": -60.0})
    station_from_file = load_station(load_station_description(path), cloud_seed=3)
    checks = station_from_file.check_alignment(seed=5)
    assert checks["mounting_plane_from_station_file"] is True
    assert checks["mounting_plane_z_mm"] == pytest.approx(-60.0)


def test_without_a_mounting_plane_the_base_top_is_used_and_said_so(tmp_path, folded_plate,
                                                                  fixture_box, base_box):
    path = write_station_file(tmp_path, folded_plate, fixture_box, base_box)
    station_default = load_station(load_station_description(path), cloud_seed=3)
    checks = station_default.check_alignment(seed=5)
    assert checks["mounting_plane_from_station_file"] is False


def test_a_part_straddling_the_fixture_top_face_is_reported_as_buried(tmp_path):
    # A fixture this size is what exposes the sampling mismatch: the station's own cloud
    # and the check's cloud are drawn at different sizes, so a check that indexes one
    # cloud's normals with the other's rows reads unrelated normals and reports noise.
    big_fixture = box(-200.0, 200.0, -200.0, 200.0, -100.0, 0.0)     # 0.48 m2
    straddling = box(-100.0, 100.0, -100.0, 100.0, -2.5, 2.5)        # 2.5 mm inside
    base = box(-400.0, 400.0, -400.0, 400.0, -160.0, -100.0)
    path = write_station_file(tmp_path, straddling, big_fixture, base)
    station_buried = load_station(load_station_description(path), cloud_seed=3)
    checks = station_buried.check_alignment(seed=5)
    assert checks["part_points_inside_fixture"] > 0
    assert checks["deepest_part_point_inside_fixture_mm"] > 1.0
    assert checks["passed"] is False


def test_a_hairline_gap_on_a_large_fixture_is_not_called_interference(tmp_path):
    big_fixture = box(-200.0, 200.0, -200.0, 200.0, -100.0, 0.0)
    hovering = box(-100.0, 100.0, -100.0, 100.0, 0.3, 5.3)           # 0.3 mm clear
    base = box(-400.0, 400.0, -400.0, 400.0, -160.0, -100.0)
    path = write_station_file(tmp_path, hovering, big_fixture, base)
    station_hovering = load_station(load_station_description(path), cloud_seed=3)
    checks = station_hovering.check_alignment(seed=5)
    assert checks["part_points_inside_fixture"] == 0
    assert checks["passed"] is True


def test_the_interference_check_does_not_change_with_the_sampling_seed(tmp_path):
    # the verdict has to be a property of the geometry, not of the draw
    big_fixture = box(-200.0, 200.0, -200.0, 200.0, -100.0, 0.0)
    straddling = box(-100.0, 100.0, -100.0, 100.0, -2.5, 2.5)
    base = box(-400.0, 400.0, -400.0, 400.0, -160.0, -100.0)
    path = write_station_file(tmp_path, straddling, big_fixture, base)
    station_seeded = load_station(load_station_description(path), cloud_seed=11)
    verdicts = {station_seeded.check_alignment(seed=seed)["part_points_inside_fixture"] > 0
                for seed in (1, 5, 17)}
    assert verdicts == {True}


def test_the_interpenetration_answer_does_not_move_with_the_sampling_seed(
        station_interpenetrating):
    # Same geometry, same answer. The coordinate is the whole point of the warning: it is
    # what a human goes and looks at on the real cell. Measured before this was fixed, the
    # real station's answer moved with the seed (2 points at 5.31 mm, then none at all),
    # because the clash region is a fraction of a square millimetre and a random cloud is
    # about 7 mm apart there.
    answers = {(tuple(checks["deepest_part_point_inside_fixture_at"]),
                checks["part_points_inside_fixture"],
                checks["deepest_part_point_inside_fixture_mm"])
               for checks in (station_interpenetrating.check_alignment(seed=seed)
                              for seed in (5, 11, 17))}
    assert len(answers) == 1


def test_the_interpenetration_probe_examines_the_part_mesh_itself(
        station_interpenetrating):
    # A random cloud alone leaves holes wider than the clash regions it is meant to find:
    # on the real station it reported 2 points at 5.31 mm for one seed and nothing at all
    # for another. The probe therefore walks the mesh's own points as well.
    checks = station_interpenetrating.check_alignment(seed=5)
    vertices = np.unique(np.round(np.asarray(station_interpenetrating.part.vertices), 3),
                         axis=0)
    assert checks["interpenetration_probe_points"]["part"] >= len(vertices)


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
