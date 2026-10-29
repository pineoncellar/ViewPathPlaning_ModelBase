"""Kinematics: inverse/forward round trip, branches, limits.

Every pose here is SYNTHETIC - random joint angles put through the forward model. The
measured station data used to fit the model is deliberately not in this file: it describes
the customer's cell.

The round trip is the test that matters. A sign error in one wrist axis produces joint
values that look entirely plausible and are wrong, and only putting them back through
`forward` makes that visible.
"""

from __future__ import annotations

import numpy as np
import pytest

from scanplan.config import RobotParams
from scanplan.kinematics import Robot, rotation_angle_deg, turn_digits

SPREAD = np.array([[-120.0, 120.0], [-50.0, 30.0], [-80.0, 80.0],
                   [-180.0, 180.0], [-100.0, 100.0], [-180.0, 180.0]])


@pytest.fixture(scope="module")
def robot() -> Robot:
    params = RobotParams.disabled()
    return Robot(RobotParams(**{**params.__dict__, "enabled": True}))


def random_joints(rng, count: int) -> np.ndarray:
    low, high = SPREAD[:, 0], SPREAD[:, 1]
    return rng.uniform(low, high, size=(count, 6))


def joint_gap(first, second) -> float:
    """Largest per-axis difference, taking the 360 degree wrap into account."""
    return max(abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)
               for a, b in zip(first, second, strict=True))


def test_forward_reproduces_a_pose_its_own_inverse_computed(robot):
    rng = np.random.default_rng(11)
    for joints in random_joints(rng, 40):
        position, rotation = robot.forward(joints)
        solutions = robot.solutions(position, rotation)
        assert solutions, f"no solution for joints {joints}"
        assert min(joint_gap(item.joints_deg, joints) for item in solutions) < 1e-3


def test_wrist_centre_matches_the_tool_offset(robot):
    """The flange origin is exactly |tool| from the TCP, in every orientation."""
    rng = np.random.default_rng(12)
    for joints in random_joints(rng, 20):
        position, _rotation = robot.forward(joints)
        wrist = robot.wrist_centre(joints)
        assert np.linalg.norm(position - wrist) == pytest.approx(
            float(np.linalg.norm(robot.tool_vector)), abs=1e-9)


def test_a_generic_pose_has_four_arm_branches(robot):
    """Two shoulders times two elbows, each with two wrist signs: up to eight solutions."""
    rng = np.random.default_rng(13)
    position, rotation = robot.forward(random_joints(rng, 1)[0])
    solutions = robot.solutions(position, rotation)
    shoulders = {(item.shoulder, item.elbow) for item in solutions}
    assert len(solutions) >= 4
    assert len(shoulders) >= 2


def test_wrist_flip_changes_only_the_sign_of_j5(robot):
    """Why changing the T letter in a CONFIG string never fixes an over-travel."""
    rng = np.random.default_rng(14)
    position, rotation = robot.forward(random_joints(rng, 1)[0])
    solutions = robot.solutions(position, rotation)
    grouped: dict[tuple[str, str], list[float]] = {}
    for item in solutions:
        grouped.setdefault((item.shoulder, item.elbow), []).append(item.joints_deg[4])
    for values in grouped.values():
        if len(values) == 2:
            assert abs(abs(values[0]) - abs(values[1])) < 1e-6


def test_solutions_are_sorted_by_j5_room(robot):
    rng = np.random.default_rng(15)
    position, rotation = robot.forward(random_joints(rng, 1)[0])
    margins = [item.j5_margin_deg for item in robot.solutions(position, rotation)]
    assert margins == sorted(margins, reverse=True)


def test_config_string_reads_the_joint_values(robot):
    """The letters come from the solution, not from a constant.

    The T/F letter is `T` for NEGATIVE J5: on the 2026-10-23 bank all eight poses sat at
    J5 between -9 and -124 and the pendant called every one of them `T`.
    """
    positive = robot.config_string((0, 0, 0, 0, 40, 0), "F", "D")
    negative = robot.config_string((0, 0, 0, 0, -40, 0), "F", "D")
    assert positive.startswith("F D F")
    assert negative.startswith("F D T")
    assert positive.endswith(", 0, 0, 0")


def test_the_wrist_letter_can_be_pinned_for_a_controller_that_refuses_one(robot):
    """This cell's ROBOGUIDE refuses `N U F` outright (ASBN-023), so the letter is pinned.

    The measured rule (`T` for J5 < 0) stays available as `auto`; pinning costs nothing
    geometrically because the two wrist branches are the same TCP pose.
    """
    auto = RobotParams(**{**robot.params.__dict__, "config_wrist_letter": "auto"})
    assert Robot(auto).config_string((0, 0, 0, 0, 40.0, 0), "N", "U").startswith("N U F")
    assert Robot(auto).config_string((0, 0, 0, 0, -40.0, 0), "N", "U").startswith("N U T")
    pinned = RobotParams(**{**robot.params.__dict__, "config_wrist_letter": "T"})
    assert Robot(pinned).config_string((0, 0, 0, 0, 40.0, 0), "N", "U").startswith("N U T")
    assert Robot(pinned).config_string((0, 0, 0, 0, -40.0, 0), "N", "U").startswith("N U T")


def test_the_turn_digits_stay_zero_inside_the_axis_travel(robot):
    """A negative turn digit is not a FANUC CONFIG: ROBOGUIDE refuses to load the program.

    The old rule was `floor(j / 360)`, so J1 = -102 deg was written as `-1` and the 2026-10-27
    delivery stopped at import with ASBN-009 "Configuration error" - the whole file, not the
    one point. A turn means a revolution past the axis's own range, not a negative angle.
    """
    assert robot.config_string((-102.0, 0, 0, 0, 30.0, 0), "N", "U").endswith(", 0, 0, 0")
    assert robot.config_string((0, 0, 0, 226.0, 30.0, -300.0), "N", "U").endswith(
        ", 0, 0, 0")
    # ... and a genuinely wound axis is flagged, not silently mis-written
    assert turn_digits(-370.0) == -1
    assert turn_digits(370.0) == 1


def test_wraps_accepts_the_equivalent_reading(robot):
    """370 degrees and 10 degrees are the same physical J6, and both readings are legal.

    The usable envelope is the datasheet's +-360 for J4/J6 (the reserve is applied as a
    margin, not by shrinking the envelope), so 370 is offered as both -350 and 10.
    """
    assert robot.wraps(5, 370.0) == pytest.approx([-350.0, 10.0])
    assert robot.wraps(3, 370.0) == pytest.approx([-350.0, 10.0])


def test_margins_prefer_the_roomiest_equivalent_reading(robot):
    """J4 is +-360, so the 200 reading has 160 deg of room and the -160 one has 200."""
    roomy = robot.margins((0, 0, 0, 200.0, 30.0, 0))[3]
    assert roomy == pytest.approx(200.0)


def test_a_value_outside_every_reading_has_no_margin(robot):
    """J2 has no full turn, so 200 degrees is simply out of range."""
    assert robot.margins((0, 200.0, 0, 0, 30.0, 0))[1] == -np.inf


def test_j5_margin_prefers_a_mid_band_angle_over_zero(robot):
    """J5 = 0 is the wrist singularity, so the middle of the mechanical range is worst."""
    assert robot.j5_margin((0, 0, 0, 0, 0.0, 0)) < 0.0
    assert robot.j5_margin((0, 0, 0, 0, 64.0, 0)) > robot.j5_margin((0, 0, 0, 0, 100.0, 0))
    assert robot.j5_margin((0, 0, 0, 0, 100.0, 0)) > 0.0


def test_the_usable_envelope_is_the_datasheet_and_the_reserve_is_a_margin(robot):
    """No percentage any more: the envelope is the limits, the reserve is `joint_margin_deg`.

    90 % of the range is a different reserve on every axis (18.5 deg on J1, 6.8 on J2, 36 on
    J4) and nothing about the machine justifies that; a flat 5-10 deg is what the site asked
    for on 2026-10-28.
    """
    assert robot.params.limits_usage_frac == pytest.approx(1.0)
    assert robot.params.joint_margin_deg == pytest.approx(5.0)
    assert robot.limits[0] == pytest.approx((-185.0, 185.0))
    assert robot.limits[1] == pytest.approx((-60.0, 46.0))
    assert robot.limits[2] == pytest.approx((-122.0, 190.0))
    assert robot.limits[3] == pytest.approx((-360.0, 360.0))
    assert robot.limits[4] == pytest.approx((-125.0, 125.0))
    assert robot.limits[5] == pytest.approx((-360.0, 360.0))
    # the reserve is what `feasible` enforces inside that envelope
    assert robot.feasible((0.0, 40.0, 0.0, 0.0, 64.0, 0.0))          # 6 deg from +46
    assert not robot.feasible((0.0, 43.0, 0.0, 0.0, 64.0, 0.0))      # 3 deg from +46
    assert not robot.feasible((0.0, 0.0, 0.0, 0.0, 121.0, 0.0))      # 4 deg from +125


def test_feasible_rejects_a_point_sitting_on_the_usable_limit(robot):
    on_limit = (0.0, robot.limits[1][0], 0.0, 0.0, 64.0, 0.0)
    assert robot.margins(on_limit)[1] == pytest.approx(0.0)
    assert not robot.feasible(on_limit)


def test_feasible_accepts_a_comfortable_pose(robot):
    assert robot.feasible((0.0, 0.0, 0.0, 0.0, 64.0, 0.0))


def test_unreachable_point_has_no_solutions(robot):
    assert robot.solutions(np.array([9000.0, 0.0, 0.0]), np.eye(3)) == []


def test_limits_source_says_which_envelope_is_in_force(robot):
    """The report has to be able to say "soft" - pretending the datasheet is in force is
    exactly how a simulation passes and the machine alarms - and it has to name the share,
    because 90 % of the travel is not the same claim as the whole envelope."""
    params = RobotParams.disabled()
    # the controller's J2 floor (-60) is narrower than the datasheet's (-90), so the
    # intersection is in force even with no percentage shrink
    assert robot.limits_source.startswith("soft")
    assert "5 deg reserved" in robot.limits_source
    same = RobotParams(**{**params.__dict__, "enabled": True,
                          "soft_limits_deg": params.mechanical_limits_deg})
    assert Robot(same).limits_source.startswith("mechanical")
    shared = RobotParams(**{**params.__dict__, "enabled": True, "limits_usage_frac": 0.9})
    assert "central 90%" in Robot(shared).limits_source
    assert Robot(shared).limits[4] == pytest.approx((-112.5, 112.5))
    no_margin = RobotParams(**{**params.__dict__, "enabled": True, "joint_margin_deg": 0.0})
    assert Robot(no_margin).limits_source == "soft (intersected with mechanical)"


def test_rotation_angle_is_zero_for_identical_frames(robot):
    rotation = np.eye(3)
    assert rotation_angle_deg(rotation, rotation) == pytest.approx(0.0)
