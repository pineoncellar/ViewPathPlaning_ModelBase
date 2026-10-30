"""The six acceptance conditions of a single pose.

To assert: the footprint shrinks towards the standoff and is capped above it; the
range limits are inclusive; incidence is computed elementwise (the matrix form
materialises gigabytes on a real part) and behaves as the tilt ladder expects; a
target point hidden behind the part is not covered, while a hit inside the
occlusion tolerance is; and a pose whose body would not fit is refused.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from scanplan.config import load_config
from scanplan.raycast import RayCaster
from scanplan.scanner_model import ScannerModel, ScannerPose, measure_mask


@pytest.fixture(scope="session")
def model_placeholder():
    """The shipped placeholder scanner, the one every coverage figure refers to."""
    return ScannerModel(load_config().scanner)


@pytest.fixture(scope="session")
def model_wide_fov():
    """A scanner so generous that only the condition under test can fail a point."""
    params = replace(load_config().scanner, range_mm=(1.0, 1.0e6),
                     fov_at_standoff_mm=(1.0e6, 1.0e6), fov_cap_mm=(1.0e6, 1.0e6),
                     max_incidence_deg=89.9)
    return ScannerModel(params)


def pose_aiming_at_origin(distance_mm, *, axis=(0.0, 0.0, -1.0), roll_deg=0.0):
    """A pose `distance_mm` above the origin, looking down at a point with normal +Z."""
    return ScannerPose(position=np.array([0.0, 0.0, distance_mm]),
                       axis=np.array(axis, dtype=float), roll_deg=roll_deg)


def test_field_of_view_shrinks_towards_the_standoff_and_is_capped_above_it(model_placeholder):
    near_width, near_height = model_placeholder.footprint_mm(150.0)
    stand_width, stand_height = model_placeholder.footprint_mm(300.0)
    far_width, far_height = model_placeholder.footprint_mm(900.0)
    assert pytest.approx(float(near_width), rel=1e-9) == 0.5 * float(stand_width)
    assert pytest.approx(float(near_height), rel=1e-9) == 0.5 * float(stand_height)
    assert pytest.approx(float(far_width)) == 200.0
    assert pytest.approx(float(far_height)) == 300.0


def test_the_footprint_is_elementwise_so_a_vector_of_distances_works(model_placeholder):
    widths, heights = model_placeholder.footprint_mm(np.array([150.0, 300.0, 900.0]))
    assert widths.shape == (3,) and heights.shape == (3,)
    assert pytest.approx(float(widths[0]), rel=1e-9) == 100.0


def test_range_limits_are_inclusive(model_placeholder):
    point = np.array([[0.0, 0.0, 0.0]])
    normal = np.array([[0.0, 0.0, 1.0]])

    def covers(distance):
        return model_placeholder.geometry_mask(pose_aiming_at_origin(distance), point, normal)

    assert covers(120.0).all() and covers(480.0).all()
    assert not covers(119.999).any()
    assert not covers(480.001).any()


def test_a_point_outside_the_footprint_is_rejected(model_placeholder):
    normal = np.array([[0.0, 0.0, 1.0]])
    inside = np.array([[90.0, 0.0, 0.0]])          # 90 < the narrower 100 mm half-width
    outside = np.array([[250.0, 0.0, 0.0]])        # beyond both half-widths (100 and 150)
    pose = pose_aiming_at_origin(300.0)
    assert model_placeholder.geometry_mask(pose, inside, normal).all()
    assert not model_placeholder.geometry_mask(pose, outside, normal).any()


def test_a_point_behind_the_scanner_is_rejected(model_placeholder):
    behind = np.array([[0.0, 0.0, 600.0]])
    normal = np.array([[0.0, 0.0, -1.0]])
    pose = pose_aiming_at_origin(300.0)
    assert not model_placeholder.geometry_mask(pose, behind, normal).any()


def test_incidence_boundary_is_inclusive_and_symmetric(model_placeholder):
    normal = np.array([[0.0, 0.0, 1.0]])
    head_on = np.array([[0.0, 0.0, -1.0]])
    bound = float(np.cos(np.deg2rad(45.0)))

    def cosine(view):
        return float(model_placeholder.incidence_cos(view, normal)[0])

    assert pytest.approx(cosine(head_on)) == 1.0
    tilted = np.array([[np.sin(np.deg2rad(45.0)), 0.0, -np.cos(np.deg2rad(45.0))]])
    assert pytest.approx(cosine(tilted), abs=1e-9) == bound
    # flipping direction and normal together leaves the cosine alone; flipping the
    # direction alone negates it - the symmetry the tilt ladder relies on
    assert pytest.approx(cosine(-tilted), abs=1e-9) == -bound
    assert pytest.approx(cosine(tilted), abs=1e-9) == bound
    both_flipped = float(model_placeholder.incidence_cos(-tilted, -normal)[0])
    assert pytest.approx(both_flipped, abs=1e-9) == bound


def test_beyond_the_incidence_limit_is_rejected(model_placeholder):
    steep = np.array([[np.sin(np.deg2rad(60.0)), 0.0, -np.cos(np.deg2rad(60.0))]])
    normal = np.array([[0.0, 0.0, 1.0]])
    pose = ScannerPose(position=np.array([0.0, 0.0, 300.0]), axis=steep[0], roll_deg=0.0)
    assert not model_placeholder.geometry_mask(pose, np.array([[0.0, 0.0, 0.0]]), normal).any()


def test_incidence_never_builds_a_dense_matrix(model_placeholder):
    # through the matrix form this would be 20000 x 20000, i.e. 3 GB and an apparent hang
    view = np.tile([0.0, 0.0, -1.0], (20_000, 1))
    normal = np.tile([0.0, 0.0, 1.0], (20_000, 1))
    assert model_placeholder.incidence_cos(view, normal).shape == (20_000,)


def test_point_behind_the_part_is_not_covered(model_wide_fov, flat_plate):
    obstacles = {"part": RayCaster(flat_plate)}
    pose = ScannerPose(position=np.array([0.0, 0.0, -200.0]),
                       axis=np.array([0.0, 0.0, 1.0]), roll_deg=0.0)
    covered = measure_mask(model_wide_fov, pose, np.array([[0.0, 0.0, 5.0]]),
                           np.array([[0.0, 0.0, -1.0]]), obstacles, occlusion_tol_mm=2.0)
    assert not covered[0]


def test_a_hit_within_the_occlusion_tolerance_does_not_block(model_wide_fov, flat_plate):
    # the ray reaches the plate's top face 0.7 mm before the target point itself, which is
    # inside the tolerance - the plate's own surface must not shadow the point it carries
    obstacles = {"part": RayCaster(flat_plate)}
    pose = ScannerPose(position=np.array([10.0, 10.0, 8.0]),
                       axis=np.array([0.0, 0.0, -1.0]), roll_deg=0.0)
    covered = measure_mask(model_wide_fov, pose, np.array([[10.0, 10.0, 0.0]]),
                           np.array([[0.0, 0.0, 1.0]]), obstacles, occlusion_tol_mm=2.0)
    assert covered[0]


def test_measure_mask_requires_every_condition(model_wide_fov, flat_plate):
    obstacles = {"part": RayCaster(flat_plate)}
    point = np.array([[10.0, 10.0, 0.7]])
    normal = np.array([[0.0, 0.0, 1.0]])
    visible = ScannerPose(position=np.array([10.0, 10.0, 300.0]),
                          axis=np.array([0.0, 0.0, -1.0]), roll_deg=0.0)
    facing_away = ScannerPose(position=np.array([10.0, 10.0, 300.0]),
                              axis=np.array([0.0, 0.0, 1.0]), roll_deg=0.0)
    assert measure_mask(model_wide_fov, visible, point, normal, obstacles,
                        occlusion_tol_mm=2.0).all()
    assert not measure_mask(model_wide_fov, facing_away, point, normal, obstacles,
                            occlusion_tol_mm=2.0).any()


def test_a_pose_whose_body_would_not_fit_is_rejected(model_placeholder):
    position = np.array([0.0, 0.0, 10.0])
    assert model_placeholder.body_fits(position, clearance_mm=100.0) is True
    assert model_placeholder.body_fits(position, clearance_mm=45.001) is True
    assert model_placeholder.body_fits(position, clearance_mm=45.0) is False
    assert model_placeholder.body_fits(position, clearance_mm=10.0) is False


def test_the_body_check_adds_the_safety_clearance_on_top_of_the_radius():
    params = replace(load_config().scanner, body_sphere_r_mm=45.0, body_clearance_mm=10.0)
    model = ScannerModel(params)
    position = np.array([0.0, 0.0, 10.0])
    assert model.body_fits(position, clearance_mm=55.001) is True
    assert model.body_fits(position, clearance_mm=55.0) is False


def test_the_pose_frame_is_the_scanner_frame(model_placeholder):
    pose = ScannerPose(position=np.array([0.0, 0.0, 300.0]),
                       axis=np.array([0.0, 0.0, -1.0]), roll_deg=0.0)
    assert np.allclose(pose.frame[:, 2], [0.0, 0.0, -1.0])
    assert np.allclose(pose.frame.T @ pose.frame, np.eye(3), atol=1e-12)

