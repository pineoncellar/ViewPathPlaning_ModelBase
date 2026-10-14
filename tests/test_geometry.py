"""The scanner body frame, which every other module depends on.

To assert: the frame is a proper rotation (orthonormal, determinant +1); local
+Z equals the optical axis; roll leaves +Z alone but rotates the footprint axes;
axis-aligned directions do not degenerate; a zero axis is rejected.
"""

from __future__ import annotations

import numpy as np
import pytest

from scanplan.geometry import (
    grid_dedupe,
    sample_surface_seeded,
    scanner_frame,
    tangents,
    tilt_axis,
    unit,
)


def test_frame_is_a_proper_rotation_whose_plus_z_is_the_axis():
    for axis in ([0, 0, -1], [1, 0, 0], [0.3, -0.4, 0.866]):
        frame = scanner_frame(axis, roll_deg=37.0)
        assert np.allclose(frame.T @ frame, np.eye(3), atol=1e-12)
        assert np.linalg.det(frame) > 0
        assert np.allclose(frame[:, 2], unit(axis), atol=1e-12)


def test_roll_leaves_the_axis_alone_and_rotates_the_footprint_axes():
    a = scanner_frame([0, 0, 1], 0.0)
    b = scanner_frame([0, 0, 1], 90.0)
    assert np.allclose(a[:, 2], b[:, 2], atol=1e-12)
    assert np.allclose(b[:, 0], a[:, 1], atol=1e-12)
    assert np.allclose(b[:, 1], -a[:, 0], atol=1e-12)


def test_frame_does_not_degenerate_on_any_axis_aligned_direction():
    for axis in ([0, 0, 1], [0, 0, -1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]):
        frame = scanner_frame(axis)
        assert np.isfinite(frame).all()
        assert np.allclose(frame.T @ frame, np.eye(3), atol=1e-12)


def test_zero_axis_is_rejected():
    with pytest.raises(ValueError):
        scanner_frame([0.0, 0.0, 0.0])


def test_tilt_axis_keeps_the_requested_angle():
    t1, t2 = tangents([0, 0, 1])
    axis = tilt_axis([0, 0, -1], t1, 30.0)
    assert pytest.approx(float(np.dot(axis, [0, 0, -1])), abs=1e-12) == np.cos(np.deg2rad(30.0))
    assert pytest.approx(float(np.dot(axis, t2)), abs=1e-12) == 0.0


def test_grid_dedupe_keeps_one_sample_per_cell_and_stays_aligned():
    points = np.array([[0.1, 0.1, 0.0], [0.2, 0.2, 0.0], [5.1, 5.1, 0.0]])
    tag = np.array([7, 8, 9])
    kept_points, kept_tag = grid_dedupe(points, tag, grid_mm=1.0)
    assert len(kept_points) == len(kept_tag) == 2
    assert np.allclose(kept_points[0], points[0]) and kept_tag[0] == 7


def test_sampling_is_reproducible_for_the_same_seed(flat_plate):
    a, na = sample_surface_seeded(flat_plate, 500, seed=3)
    b, nb = sample_surface_seeded(flat_plate, 500, seed=3)
    assert np.array_equal(a, b) and np.array_equal(na, nb)
