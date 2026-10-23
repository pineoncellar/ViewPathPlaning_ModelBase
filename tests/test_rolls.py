"""Roll decided from the axis alone.

Why this is a module and a test of its own: the roll was previously whatever the frame
builder's helper axis produced, and `scanner_frame` switches helpers when the optical axis
comes within about 25 degrees of vertical. Two nearly identical sweep directions could then
differ by 90 degrees of roll - and with a 600 x 500 footprint that changes which samples the
sweep can reach. Deciding it from the axis alone makes it a pure function.
"""

from __future__ import annotations

import numpy as np
import pytest

from scanplan.geometry import scanner_frame, unit
from scanplan.rolls import REFERENCE, roll_consistent, roll_for_axis


def test_the_roll_points_the_local_x_at_the_station_x():
    for axis in ([0.0, 0.0, -1.0], [0.3, 0.0, -0.95], [0.0, 0.4, -0.92],
                 [0.6, 0.5, -0.62]):
        optical = unit(axis)
        roll = roll_for_axis(optical)
        frame = scanner_frame(optical, roll)
        target = unit(REFERENCE - optical * float(np.dot(REFERENCE, optical)))
        assert float(np.dot(frame[:, 0], target)) == pytest.approx(1.0, abs=1e-9)
        assert roll_consistent(optical, roll)


def test_the_roll_is_a_pure_function_of_the_axis():
    axis = unit([0.2, 0.1, -0.97])
    assert roll_for_axis(axis) == roll_for_axis(axis.copy())


def test_an_axis_along_the_reference_has_no_roll():
    """No in-plane direction to align to: zero, not a number made of rounding noise."""
    assert roll_for_axis([1.0, 0.0, 0.0]) == 0.0
    assert roll_for_axis([-1.0, 0.0, 0.0]) == 0.0


def test_neighbouring_axes_get_neighbouring_frames():
    """The whole point: no 90 degree jump in the FRAME where the helper axis switches.

    The roll NUMBER may jump there - `scanner_frame` picks a different helper on either
    side of the threshold and `roll_for_axis` compensates - which is why this asserts on
    the frame, the thing that decides which samples the footprint reaches.
    """
    first = unit([0.0, 0.40, -0.9165])
    second = unit([0.0, 0.45, -0.8930])
    assert abs(first[2]) > 0.9 > abs(second[2])          # opposite sides of the switch
    one = scanner_frame(first, roll_for_axis(first))
    two = scanner_frame(second, roll_for_axis(second))
    assert float(np.dot(one[:, 0], two[:, 0])) > 0.99


def test_the_frame_is_the_reference_projection_for_every_axis():
    """The composition is helper-free in effect, which is what makes it usable in growth."""
    for axis in ([0.0, 0.0, -1.0], [0.0, 0.5, -0.866], [0.0, 0.95, -0.31],
                 [0.7, 0.1, -0.7]):
        optical = unit(axis)
        frame = scanner_frame(optical, roll_for_axis(optical))
        expected = unit(REFERENCE - optical * float(np.dot(REFERENCE, optical)))
        assert np.allclose(frame[:, 0], expected, atol=1e-9)
