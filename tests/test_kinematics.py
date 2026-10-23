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
from scanplan.kinematics import Robot, rotation_angle_deg

SPREAD = np.array([[-120.0, 120.0], [-50.0, 100.0], [-80.0, 80.0],
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
    """The letters come from the solution, not from a constant."""
    positive = robot.config_string((0, 0, 0, 0, 40, 0), "F", "D")
    negative = robot.config_string((0, 0, 0, 0, -40, 0), "F", "D")
    assert positive.startswith("F D T")
    assert negative.startswith("F D F")
    assert positive.endswith(", 0, 0, 0")


def test_turn_counts_are_reported_for_the_wrist_axes(robot):
    assert robot.config_string((0, 0, 0, 370, 30, -370), "N", "U").endswith(", 0, 1, -2")


def test_wraps_accepts_the_equivalent_reading(robot):
    """370 degrees and 10 degrees are the same physical J6, so both must be offered."""
    assert robot.wraps(5, 370.0) == pytest.approx([-350.0, 10.0])
    # J4 spans two full turns, so 370 is already legal there as written.
    assert 370.0 in robot.wraps(3, 370.0)


def test_margins_prefer_the_roomiest_equivalent_reading(robot):
    """J4 spans +-720, so the reading nearest the middle of that range wins."""
    roomy = robot.margins((0, 0, 0, 200.0, 30.0, 0))[3]
    assert roomy == pytest.approx(560.0)


def test_a_value_outside_every_reading_has_no_margin(robot):
    """J2 has no full turn, so 200 degrees is simply out of range."""
    assert robot.margins((0, 200.0, 0, 0, 30.0, 0))[1] == -np.inf


def test_j5_margin_prefers_a_mid_band_angle_over_zero(robot):
    """J5 = 0 is the wrist singularity, so the middle of the mechanical range is worst."""
    assert robot.j5_margin((0, 0, 0, 0, 0.0, 0)) < 0.0
    assert robot.j5_margin((0, 0, 0, 0, 64.0, 0)) > robot.j5_margin((0, 0, 0, 0, 110.0, 0))
    assert robot.j5_margin((0, 0, 0, 0, 110.0, 0)) > 0.0


def test_feasible_rejects_a_point_sitting_on_the_soft_limit(robot):
    on_limit = (0.0, -60.0, 0.0, 0.0, 64.0, 0.0)
    assert robot.margins(on_limit)[1] == pytest.approx(0.0)
    assert not robot.feasible(on_limit)


def test_feasible_accepts_a_comfortable_pose(robot):
    assert robot.feasible((0.0, 0.0, 0.0, 0.0, 64.0, 0.0))


def test_unreachable_point_has_no_solutions(robot):
    assert robot.solutions(np.array([9000.0, 0.0, 0.0]), np.eye(3)) == []


def test_limits_source_says_which_envelope_is_in_force(robot):
    """The report has to be able to say "soft" - pretending the datasheet is in force is
    exactly how a simulation passes and the machine alarms."""
    assert robot.limits_source.startswith("soft")
    params = RobotParams.disabled()
    same = RobotParams(**{**params.__dict__, "enabled": True, "soft_limits_deg":
                          params.mechanical_limits_deg})
    assert Robot(same).limits_source == "mechanical"


def test_rotation_angle_is_zero_for_identical_frames(robot):
    rotation = np.eye(3)
    assert rotation_angle_deg(rotation, rotation) == pytest.approx(0.0)
