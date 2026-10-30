"""Generating candidate scanner poses, coarsely and then finely.

To assert: every aim point gets the full distance/tilt/roll ladder; the roll is not
duplicated at zero tilt because it would repeat the same optical axis; de-duplication
collapses repeats; the pool respects its cap and the cap is reproducible; generation
is reproducible from its seed; and the refinement pass aims at the leaked region
instead of resampling the whole face.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from scanplan.candidates import _cap, aim_indices, dedupe, generate, ladder, refine
from scanplan.config import load_config
from scanplan.scanner_model import ScannerPose
from scanplan.target_surface import sample_face


@pytest.fixture(scope="session")
def params_placeholder():
    return load_config().scanner


@pytest.fixture(scope="session")
def params_tight(params_placeholder):
    """A pool that hits its cap on a small plate: a fine aim grid and a low limit."""
    return replace(params_placeholder, aim_grid_mm=10.0, max_candidates=50)


def grid_samples(mesh, grid_mm):
    """Area-weighted samples of the whole mesh on a grid, as the target stage produces."""
    samples = sample_face(mesh, np.ones(len(mesh.faces), dtype=bool),
                          grid_mm=grid_mm, seed=5)
    return samples.points, samples.normals


def expected_ladder_length(params):
    """Poses one aim point yields: every distance, azimuth, tilt and roll.

    The zero tilt repeats across azimuths - there is only one head-on axis - so it counts
    once: `1 + azimuths * tilted` axes per distance.
    """
    azimuths = max(1, int(params.tilt_azimuths))
    tilted = [value for value in params.tilts_deg if value != 0.0]
    axes = (1 if len(tilted) < len(params.tilts_deg) else 0) + azimuths * len(tilted)
    return len(params.distances_mm) * axes * len(params.rolls_deg)


def test_every_aim_point_gets_the_full_distance_tilt_roll_ladder(
        flat_plate, params_placeholder):
    points, normals = grid_samples(flat_plate, 100.0)
    aims = aim_indices(points, params_placeholder.aim_grid_mm)
    poses = generate(points, normals, params_placeholder, seed=0)
    assert len(aims) > 1
    assert len(poses) == len(aims) * expected_ladder_length(params_placeholder)
    assert {round(pose.roll_deg, 1) for pose in poses} == set(params_placeholder.rolls_deg)


def test_every_distance_in_the_ladder_is_used(flat_plate, params_placeholder):
    points, normals = grid_samples(flat_plate, 200.0)
    poses = generate(points, normals, params_placeholder, seed=0)
    aim = points[aim_indices(points, params_placeholder.aim_grid_mm)[0]]
    offsets = sorted({round(float(np.linalg.norm(pose.position - aim)), 1) for pose in poses})
    for distance in params_placeholder.distances_mm:
        assert any(pytest.approx(distance, abs=0.2) == value for value in offsets), distance


def test_the_zero_tilt_pose_gets_both_rolls(params_placeholder):
    # The optical axis repeats, the FOOTPRINT does not: the field of view is a rectangle,
    # so a quarter turn about the axis covers a different set of samples. Design 4.2 lists
    # two rolls for exactly this reason, and zero tilt is the incidence every selection
    # prefers.
    point = np.array([[10.0, 10.0, 0.7]])
    normal = np.array([[0.0, 0.0, 1.0]])
    poses = generate(point, normal, params_placeholder, seed=0)
    head_on = [pose for pose in poses if np.allclose(pose.axis, [0.0, 0.0, -1.0])]
    assert {pose.roll_deg for pose in head_on} == set(params_placeholder.rolls_deg)
    assert len(head_on) == (len(params_placeholder.distances_mm)
                            * len(params_placeholder.rolls_deg))


def test_the_two_rolls_cover_different_samples_on_a_wide_target(params_placeholder):
    from scanplan.scanner_model import ScannerModel

    model = ScannerModel(params_placeholder)
    wide = np.array([[x, y, 0.0] for x in range(-560, 561, 20)
                     for y in range(-460, 461, 20)], dtype=float)
    normals = np.tile([0.0, 0.0, 1.0], (len(wide), 1))
    head_on = [pose for pose in ladder(np.zeros(3), np.array([0.0, 0.0, 1.0]),
                                       params_placeholder)
               if np.allclose(pose.axis, [0.0, 0.0, -1.0])
               and pose.position[2] == params_placeholder.standoff_mm]
    assert len(head_on) == len(params_placeholder.rolls_deg)
    coverage = [set(np.flatnonzero(model.geometry_mask(pose, wide, normals)).tolist())
                for pose in head_on]
    assert coverage[0] != coverage[1]
    assert coverage[0] - coverage[1] and coverage[1] - coverage[0]


def test_the_tilt_fan_sweeps_the_full_circle(params_placeholder):
    """A one-plane ladder approaches each point from two directions; the fan from all of them.

    This is what recovered the coverage: 91 % of the samples the joint gate had left
    uncovered on 2026-10-28 had a feasible pose somewhere round the normal, so the pool -
    not the arm - was the limit.
    """
    poses = ladder(np.zeros(3), np.array([0.0, 0.0, 1.0]), params_placeholder)
    tilted = [pose.axis for pose in poses if pose.axis[2] > -0.99]
    # rounded before the modulo: a -1e-17 radian component would otherwise land on 360.0
    azimuths = sorted({round(float(np.degrees(np.arctan2(axis[1], axis[0]))), 1) % 360.0
                       for axis in tilted})
    assert len(azimuths) == 2 * max(1, int(params_placeholder.tilt_azimuths))
    assert azimuths[0] == pytest.approx(0.0, abs=1e-6)
    assert max(abs(float(np.degrees(np.arccos(abs(axis[2])))) - 30.0)
               for axis in tilted) < 1e-6


def test_a_joint_filter_is_applied_before_the_cap(flat_plate, params_placeholder):
    """Filtering afterwards would spend the budget on poses the arm can never hold."""
    points, normals = grid_samples(flat_plate, 100.0)
    posed = replace(params_placeholder, aim_grid_mm=10.0)
    poses = generate(points, normals, posed, seed=0, max_candidates=50,
                     keep=lambda pose: pose.roll_deg == 0.0)
    assert poses
    assert {pose.roll_deg for pose in poses} == {0.0}


def test_the_cap_keeps_every_aim_point_represented():
    """A uniform random cut leaves whole regions with no candidate at all."""
    groups = [[ScannerPose(np.array([float(index), 0.0, 0.0]),
                           np.array([0.0, 0.0, -1.0]), float(step)) for step in range(10)]
              for index in range(3)]
    picked = _cap(groups, 6)
    assert len(picked) == 6
    assert [sum(1 for pose in picked if pose.position[0] == index)
            for index in range(3)] == [2, 2, 2]


def test_the_tilt_ladder_leans_both_ways_about_the_normal(params_placeholder):
    point = np.array([10.0, 10.0, 0.7])
    normal = np.array([0.0, 0.0, 1.0])
    poses = ladder(point, normal, params_placeholder)
    angles = [float(np.degrees(np.arccos(np.clip(np.dot(pose.axis, -normal), -1.0, 1.0))))
              for pose in poses]
    assert min(angles) == pytest.approx(0.0, abs=1e-6)
    assert max(angles) == pytest.approx(30.0, abs=1e-6)
    # both signs have to be present, or the pool only ever leans the same way
    assert sum(1 for angle in angles if angle > 29.0) >= 2


def test_dedupe_collapses_repeated_poses():
    pose = ScannerPose(np.array([0.0, 0.0, 300.0]), np.array([0.0, 0.0, -1.0]), 0.0)
    near_duplicate = ScannerPose(np.array([0.2, 0.3, 299.6]), np.array([0.0, 0.0, -1.0]), 0.0)
    assert len(dedupe([pose, pose, near_duplicate])) == 1


def test_the_cap_is_respected_and_reproducible(flat_plate, params_tight):
    points, normals = grid_samples(flat_plate, 20.0)
    first = generate(points, normals, params_tight, seed=1, max_candidates=50)
    second = generate(points, normals, params_tight, seed=1, max_candidates=50)
    assert len(first) <= 50
    assert [tuple(pose.position) for pose in first] == [tuple(pose.position) for pose in second]


def test_generation_is_deterministic_for_a_seed(flat_plate, params_placeholder):
    points, normals = grid_samples(flat_plate, 50.0)
    first = generate(points, normals, params_placeholder, seed=9)
    second = generate(points, normals, params_placeholder, seed=9)
    assert [tuple(pose.position) for pose in first] == [tuple(pose.position) for pose in second]
    assert [tuple(pose.axis) for pose in first] == [tuple(pose.axis) for pose in second]


def test_a_default_pool_is_capped_by_the_configuration(flat_plate, params_tight):
    points, normals = grid_samples(flat_plate, 20.0)
    poses = generate(points, normals, params_tight, seed=3)
    assert len(poses) == params_tight.max_candidates


def test_refine_aims_at_the_leak_and_stays_unique(params_placeholder):
    leak_points = np.array([[10.0, 10.0, 0.7], [180.0, 180.0, 0.7]])
    leak_normals = np.tile([0.0, 0.0, 1.0], (2, 1))
    poses = refine(leak_points, leak_normals, params_placeholder, seed=0)
    assert len(poses) > 0
    assert len({pose.key() for pose in poses}) == len(poses)
    near_first = [pose for pose in poses
                  if np.linalg.norm(pose.position - leak_points[0]) < 500.0]
    assert near_first, "the refinement has to aim at the leak, not at the part in general"


def test_refine_dedupes_leak_points_that_share_a_cell(params_placeholder):
    # a couple of millimetres apart, well inside the refinement grid: one ladder, not two
    leak_points = np.array([[10.0, 10.0, 0.7], [11.0, 12.0, 0.7]])
    leak_normals = np.tile([0.0, 0.0, 1.0], (2, 1))
    poses = refine(leak_points, leak_normals, params_placeholder, seed=0)
    assert len(poses) == expected_ladder_length(params_placeholder)


def test_refine_on_nothing_is_empty(params_placeholder):
    empty = np.zeros((0, 3))
    assert refine(empty, empty, params_placeholder, seed=0) == []


def test_a_large_refinement_is_capped_reproducibly(flat_plate, params_placeholder):
    points, normals = grid_samples(flat_plate, 5.0)
    first = refine(points, normals, params_placeholder, seed=2, max_candidates=40)
    second = refine(points, normals, params_placeholder, seed=2, max_candidates=40)
    assert len(first) == 40
    assert [tuple(pose.position) for pose in first] == [tuple(pose.position) for pose in second]
