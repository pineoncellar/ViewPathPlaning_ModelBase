"""Tracker visibility, and where the tracker should be mounted.

To assert: a pose behind the part is not visible; visibility is decided on the SCANNER
and not on each target point (that mix-up once produced a reading of 87% against a true
99.6%); the half angle and the distance limits bite inclusively; looking along the
part's thin axis keeps more poses than looking square-on to its large faces; the sweep
covers the documented grid; and a non-omnidirectional marker geometry is refused rather
than guessed.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from conftest import box

from scanplan.config import load_config
from scanplan.raycast import RayCaster
from scanplan.tracker import (
    Tracker,
    part_centroid,
    sweep_placements,
    visible_mask,
)


@pytest.fixture(scope="session")
def tracker_params():
    return load_config().tracker


def make_tracker(position, aim, params=None, name="tracker"):
    return Tracker(position=np.asarray(position, dtype=float),
                   aim=np.asarray(aim, dtype=float),
                   params=params if params is not None else load_config().tracker,
                   name=name)


def test_a_pose_behind_the_part_is_not_visible(flat_plate):
    tracker = make_tracker([0.0, 0.0, -1000.0], [0.0, 0.0, 0.0])
    hidden = np.array([[0.0, 0.0, 500.0]])          # the plate sits between the two
    assert not visible_mask(tracker, hidden, {"part": RayCaster(flat_plate)})[0]


def test_a_pose_in_the_open_is_visible(flat_plate):
    tracker = make_tracker([0.0, 0.0, 3000.0], [0.0, 0.0, 0.0])
    clear = np.array([[0.0, 0.0, 400.0]])
    assert visible_mask(tracker, clear, {"part": RayCaster(flat_plate)})[0]


def test_visibility_is_decided_on_the_scanner_and_not_on_each_target_point(flat_plate):
    # a column stands over one corner of the plate: it hides that corner, but the sight
    # line to a scanner hovering above the middle is clear
    column = box(60.0, 100.0, 60.0, 100.0, 10.0, 2000.0)
    obstacles = {"part": RayCaster(flat_plate), "column": RayCaster(column)}
    tracker = make_tracker([0.0, 0.0, 3000.0], [0.0, 0.0, 0.0])
    scanner_clear = np.array([[0.0, 0.0, 400.0]])
    scanner_behind_the_column = np.array([[90.0, 90.0, 200.0]])
    assert visible_mask(tracker, scanner_clear, obstacles)[0]
    assert not visible_mask(tracker, scanner_behind_the_column, obstacles)[0]


def test_outside_the_half_angle_is_not_visible():
    narrow = replace(load_config().tracker, half_angle_deg=5.0)
    tracker = make_tracker([0.0, 0.0, 3000.0], [0.0, 0.0, 0.0], narrow)
    assert visible_mask(tracker, np.array([[0.0, 0.0, 400.0]]), {})[0]
    assert not visible_mask(tracker, np.array([[2000.0, 0.0, 400.0]]), {})[0]


def test_distance_limits_are_inclusive():
    params = replace(load_config().tracker, distance_mm=(300.0, 6000.0))
    tracker = make_tracker([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], params)
    assert visible_mask(tracker, np.array([[0.0, 0.0, 300.0]]), {})[0]
    assert visible_mask(tracker, np.array([[0.0, 0.0, 6000.0]]), {})[0]
    assert not visible_mask(tracker, np.array([[0.0, 0.0, 299.999]]), {})[0]
    assert not visible_mask(tracker, np.array([[0.0, 0.0, 6000.001]]), {})[0]


def test_looking_along_the_thin_axis_keeps_more_poses_than_looking_square_on(station):
    positions = np.array([[x, y, z] for x in np.linspace(-200.0, 200.0, 9)
                          for y in (-100.0, 0.0, 100.0)
                          for z in (-300.0, 300.0)])
    obstacles = {"part": RayCaster(station.part),
                 "fixture": RayCaster(station.fixture),
                 "base": RayCaster(station.base)}
    centroid = part_centroid(station.part)
    along = make_tracker([5000.0, 0.0, 100.0], centroid, name="along the thin axis")
    square_on = make_tracker([0.0, 0.0, 4500.0], centroid, name="square on to the faces")
    along_visible = int(visible_mask(along, positions, obstacles).sum())
    square_visible = int(visible_mask(square_on, positions, obstacles).sum())
    # the part's own body is what hides the scanner from a tracker mounted square-on
    assert along_visible > square_visible
    assert along_visible >= 0.9 * len(positions)


def test_the_sweep_uses_the_documented_grid(flat_plate, tracker_params):
    placements = sweep_placements(flat_plate, tracker_params)
    assert len(placements) == 192          # 16 azimuths x 3 distances x 4 heights
    positions = {tuple(np.round(placement.position, 3)) for placement in placements}
    assert len(positions) == 192
    for placement in placements:
        assert np.linalg.norm(placement.axis) == pytest.approx(1.0)
        assert np.linalg.norm(placement.position[:2]) > 0.0


def test_the_sweep_spans_both_sides_and_the_documented_heights(flat_plate, tracker_params):
    placements = sweep_placements(flat_plate, tracker_params)
    heights = sorted({float(placement.position[2]) for placement in placements})
    assert heights == [100.0, 400.0, 800.0, 1200.0]
    centroid = part_centroid(flat_plate)
    directions = set()
    for placement in placements:
        horizontal = (placement.position - centroid)[:2]
        unit = horizontal / np.linalg.norm(horizontal)
        directions.add((round(float(unit[0]), 3), round(float(unit[1]), 3)))
    assert len(directions) == 16
    # the sweep has to span the whole circle, not just one side of the part
    assert any(unit[0] < 0 for unit in directions) and any(unit[0] > 0 for unit in directions)
    assert any(unit[1] < 0 for unit in directions) and any(unit[1] > 0 for unit in directions)


def test_the_sweep_axes_point_at_the_part(flat_plate, tracker_params):
    centroid = part_centroid(flat_plate)
    for placement in sweep_placements(flat_plate, tracker_params)[:6]:
        toward = centroid - placement.position
        toward = toward / np.linalg.norm(toward)
        assert float(np.dot(toward, placement.axis)) > 0.99


def test_non_omnidirectional_markers_are_refused_rather_than_guessed():
    params = replace(load_config().tracker, markers_omnidirectional=False)
    tracker = make_tracker([0.0, 0.0, 3000.0], [0.0, 0.0, 0.0], params)
    with pytest.raises(NotImplementedError, match="marker"):
        visible_mask(tracker, np.array([[0.0, 0.0, 400.0]]), {})


def test_non_omnidirectional_markers_are_refused_by_the_sweep_too(flat_plate):
    params = replace(load_config().tracker, markers_omnidirectional=False)
    with pytest.raises(NotImplementedError, match="marker"):
        sweep_placements(flat_plate, params)


def test_the_sweep_skips_a_tracker_position_that_is_inside_the_part(flat_plate, tracker_params):
    placements = sweep_placements(flat_plate, tracker_params)
    inside = [placement for placement in placements
              if np.linalg.norm(placement.position - part_centroid(flat_plate)) < 100.0]
    assert inside == []

