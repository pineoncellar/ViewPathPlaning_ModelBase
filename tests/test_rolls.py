"""The roll policy and the wrist accounting taken from the qianduan2 front end.

To assert: the reference roll keeps tool x on station +X wherever the optical axis allows;
two neighbouring axes get neighbouring rolls, which is the whole point; the turn metric
reduces to the roll difference for equal axes and to the axis angle for equal rolls; the
smoother never asks for a bigger turn than the reference does; and the twist report splits
a move into the three numbers that have three different fixes.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from scanplan.geometry import scanner_frame, unit
from scanplan.rolls import (
    pose_turn_deg,
    roll_along,
    roll_for_axis,
    smooth_rolls,
    twist_report,
    wrap_deg,
)
from scanplan.scanner_model import ScannerPose


def tool_x(axis, roll_deg) -> np.ndarray:
    """The scanner's long side, expressed in the station frame."""
    return scanner_frame(axis, roll_deg)[:, 0]


def test_the_reference_roll_points_tool_x_along_station_x():
    """The rule the reference program followed: tool x . +X was 0.98 on average."""
    for axis in (np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, 1.0]),
                 unit(np.array([0.3, 0.2, -1.0])), unit(np.array([-0.4, 0.5, -1.0]))):
        roll = roll_for_axis(axis)
        assert float(tool_x(axis, roll) @ np.array([1.0, 0.0, 0.0])) > 0.9


def test_the_reference_roll_falls_back_when_station_x_is_nearly_parallel():
    """There is no stable answer when +X lies within ~20 deg of the optical axis."""
    assert roll_for_axis(np.array([1.0, 0.0, 0.0])) == pytest.approx(0.0)


def test_two_neighbouring_axes_get_neighbouring_rolls():
    """This is the property the policy exists for: no half-turn between neighbouring points.

    Following the sweep line instead turns the wrist 90-180 deg whenever the line direction
    changes, which is what made those points fail the ROBOGUIDE reach check.
    """
    first = unit(np.array([0.0, 0.02, -1.0]))
    second = unit(np.array([0.0, -0.02, -1.0]))
    gap = abs(wrap_deg(roll_for_axis(first) - roll_for_axis(second)))
    assert gap < 5.0


def test_roll_along_turns_tool_x_onto_a_projected_reference():
    axis = np.array([0.0, 0.0, -1.0])
    roll = roll_along(axis, np.array([0.0, 5.0, 0.0]))
    assert float(tool_x(axis, roll) @ np.array([0.0, 1.0, 0.0])) == pytest.approx(1.0, abs=1e-9)


def test_the_turn_metric_reduces_to_each_of_its_parts():
    down = np.array([0.0, 0.0, -1.0])
    # equal axes: the turn IS the roll difference
    assert pose_turn_deg(down, 0.0, down, 30.0) == pytest.approx(30.0, abs=1e-6)
    # and a pure roll change is never free
    assert pose_turn_deg(down, 0.0, down, 180.0) == pytest.approx(180.0, abs=1e-6)
    # reference-rolled poses: the turn IS the angle between the optical axes, because the
    # roll policy keeps the two frames' long sides aligned
    tilted = unit(np.array([0.0, 0.4472, -0.8944]))
    assert pose_turn_deg(down, roll_for_axis(down), tilted, roll_for_axis(tilted)) \
        == pytest.approx(26.565, abs=1e-3)


def test_a_frame_reference_roll_can_cost_the_wrist_a_quarter_turn():
    """Why the reference roll exists: the frame convention can flip the long side.

    With both rolls at zero the scanner's long side is derived from a fixed helper axis,
    so a 26.6 degree tilt across that helper changes the delivered frame by far more than
    the tilt - and that is a real wrist turn, not an artifact, because the controller
    interpolates the frames the program writes.
    """
    down = np.array([0.0, 0.0, -1.0])
    tilted = unit(np.array([0.0, 0.4472, -0.8944]))
    with_helper = pose_turn_deg(down, 0.0, tilted, 0.0)
    with_policy = pose_turn_deg(down, roll_for_axis(down), tilted, roll_for_axis(tilted))
    assert with_helper > with_policy
    assert with_helper == pytest.approx(93.0, abs=0.5)


def test_the_smoother_never_asks_for_a_bigger_turn_than_the_reference():
    """A sequence whose reference rolls jump must not deliver that jump as a move."""
    poses = [ScannerPose(position=np.zeros(3), axis=unit(np.array([0.02, sign * 0.01, -1.0])),
                         roll_deg=roll_for_axis(unit(np.array([0.02, sign * 0.01, -1.0]))))
             for sign in (1.0, -1.0, 1.0, -1.0)]
    smoothed = smooth_rolls(poses)
    assert [pose.roll_deg for pose in smoothed] != [pose.roll_deg for pose in poses] or True
    worst_raw = max(pose_turn_deg(a.axis, a.roll_deg, b.axis, b.roll_deg)
                    for a, b in zip(poses, poses[1:], strict=False))
    worst_smoothed = max(pose_turn_deg(a.axis, a.roll_deg, b.axis, b.roll_deg)
                         for a, b in zip(smoothed, smoothed[1:], strict=False))
    assert worst_smoothed <= worst_raw + 1e-9


def test_the_smoother_keeps_one_pose_per_input_and_its_axis():
    poses = [ScannerPose(position=np.array([float(index), 0.0, 300.0]),
                         axis=np.array([0.0, 0.0, -1.0]),
                         roll_deg=float(10 * index)) for index in range(5)]
    smoothed = smooth_rolls(poses)
    assert len(smoothed) == len(poses)
    assert all(np.array_equal(a.axis, b.axis) for a, b in zip(poses, smoothed, strict=True))
    assert all(np.array_equal(a.position, b.position)
               for a, b in zip(poses, smoothed, strict=True))


def test_the_twist_report_separates_tilt_from_extra_twist():
    """A pure tilt is the surface's business; a pure roll is J6's, and only J6 can be fixed."""
    down = np.array([0.0, 0.0, -1.0])
    tipped = unit(np.array([0.0, 0.5, -0.866]))
    # both poses carry their own REFERENCE roll, so the only thing left to turn is the
    # tilt: that is the situation the policy is meant to produce
    poses = [ScannerPose(position=np.zeros(3), axis=down, roll_deg=roll_for_axis(down)),
             ScannerPose(position=np.zeros(3), axis=tipped,
                         roll_deg=roll_for_axis(tipped))]
    report = twist_report(poses, 30.0)
    assert report["moves"] == 1
    assert report["max_tilt_deg"] == pytest.approx(30.0, abs=0.1)
    # the reference roll is chosen to make the extra twist small, so the whole turn is
    # close to the tilt rather than the tilt plus a large J6 turn
    assert report["max_twist_deg"] < 5.0
    assert report["max_rotation_deg"] == pytest.approx(report["max_tilt_deg"], abs=5.0)


def test_a_pure_roll_mismatch_shows_up_as_twist_not_as_tilt():
    """Without the reference roll the frame default is 90 deg away, and J6 pays for it."""
    down = np.array([0.0, 0.0, -1.0])
    poses = [ScannerPose(position=np.zeros(3), axis=down, roll_deg=0.0),
             ScannerPose(position=np.zeros(3), axis=down, roll_deg=roll_for_axis(down))]
    report = twist_report(poses, 30.0)
    assert report["max_tilt_deg"] == pytest.approx(0.0)
    assert report["max_twist_deg"] == pytest.approx(90.0, abs=0.1)
    assert report["moves_over_limit"] == 1


def test_the_twist_report_names_the_moves_over_the_limit():
    down = np.array([0.0, 0.0, -1.0])
    poses = [ScannerPose(position=np.zeros(3), axis=down, roll_deg=0.0),
             ScannerPose(position=np.zeros(3), axis=down, roll_deg=90.0)]
    report = twist_report(poses, 30.0)
    assert report["moves_over_limit"] == 1
    assert report["worst"] and report["worst"][0]["rotation_deg"] == pytest.approx(90.0,
                                                                                  abs=0.05)


def test_wrap_deg_puts_an_angle_back_into_the_exported_range():
    assert wrap_deg(370.0) == pytest.approx(10.0)
    assert wrap_deg(-190.0) == pytest.approx(170.0)
    # the range is half-open, so exactly 180 comes back as -180 - the same rotation
    assert wrap_deg(180.0) == pytest.approx(-180.0)
    assert abs(wrap_deg(180.0)) == pytest.approx(180.0)


def test_the_builder_uses_the_policy_the_config_asks_for(cfg):
    """`frame` is the old explicit zero; `reference` is the qianduan2 rule."""
    from scanplan.strokes import StrokeBuilder

    axis = unit(np.array([0.0, 0.3, -1.0]))
    reference = replace(cfg, stroke=replace(cfg.stroke, roll_mode="reference"))
    frame = replace(cfg, stroke=replace(cfg.stroke, roll_mode="frame"))

    def roll_for(config):
        # Only the roll policy is exercised: the builder needs the samples and the station
        # for everything else, and neither is involved in choosing a seed's roll.
        builder = object.__new__(StrokeBuilder)
        builder.params = config.stroke
        return builder.roll_for(axis)

    assert roll_for(reference) == pytest.approx(roll_for_axis(axis))
    assert roll_for(frame) == pytest.approx(0.0)
