"""The delivery check: the joint side of a program, checked whether or not the planner looked.

The point of this file is the seam, not the geometry - the geometry is `test_kinematics`.
What it pins is that the check runs on the FILE (so it cannot be switched off with the planning
sieve), that "not verified" never reads as "passed", and that a failure names the point and the
axis rather than the transition as a whole.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from scanplan.config import RobotParams
from scanplan.geometry import scanner_frame
from scanplan.kinematics import Robot
from scanplan.roboguide.validate import check_points


def measured_robot() -> Robot:
    """The measured machine with the gate ON - what the delivery check always builds."""
    return Robot(replace(RobotParams.disabled(), enabled=True))


def looking_down(position=(2000.0, 0.0, 1000.0)) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(position, dtype=float)
    return position, scanner_frame(np.array([0.0, 0.0, -1.0]), 0.0)


def point(position, rotation, *, joints=None, motion=None) -> dict:
    return {"xyz_mm": np.asarray(position, dtype=float),
            "rotation": np.asarray(rotation, dtype=float),
            "joints_deg": joints, "motion": motion}


def first_solution(position, rotation) -> tuple[float, ...]:
    found = measured_robot().solutions(position, rotation)
    assert found, "the fixture pose must be reachable for this test to mean anything"
    return found[0].joints_deg


def test_a_joint_point_outside_the_usable_band_is_a_failure():
    """J5 = -122 is inside the controller's -125 and still not a pose to deliver."""
    position, rotation = looking_down()
    joints = list(first_solution(position, rotation))
    joints[4] = -122.0
    report = check_points([point(position, rotation, joints=tuple(joints))])
    assert report["ok"] is False
    assert report["failures"][0]["reason"] == "joint_limit"
    assert report["failures"][0]["axis"] == 4
    assert report["failures"][0]["point"] == 1


def test_a_joint_point_that_takes_j5_through_zero_is_a_failure():
    """Two legal ends, one illegal middle - the case a joint move is supposed to avoid."""
    robot = measured_robot()
    right = (10.0, 20.0, 30.0, 40.0, 20.0, 20.0)
    left = (10.0, 20.0, 30.0, 40.0, -20.0, 20.0)
    assert robot.feasible(right) and robot.feasible(left)
    first = point(*robot.forward(right), joints=right, motion="J")
    second = point(*robot.forward(left), joints=left, motion="J")
    report = check_points([first, second])
    assert [item["reason"] for item in report["failures"]] == ["j5_crossover"]
    assert report["failures"][0]["point"] == 2


def test_a_joint_point_reached_by_L_is_walked_instead_of_trusted():
    """A scan point has joints AND is flown as a straight line; the line is what can fail.

    Having a joint solution recorded for a Cartesian point says nothing about the L move that
    reaches it, and the L move is where the over-travel alarm came from. So the motion decides
    which check runs, not the presence of joint values.
    """
    position, rotation = looking_down()
    start = point(position, rotation, joints=first_solution(position, rotation), motion="J")
    far = np.array([2000.0, 0.0, 3400.0])
    end = point(far, rotation, joints=first_solution(position, rotation), motion="L")
    report = check_points([start, end], default_motion="L")
    assert report["ok"] is False
    assert report["failures"][0]["point"] == 2


def test_an_L_move_that_runs_out_of_branch_is_a_failure_and_says_how_far():
    """The `MOTN-017` shape: both ends look fine, the line between them does not."""
    position, rotation = looking_down()
    start = point(position, rotation, joints=first_solution(position, rotation))
    far = point(np.array([2000.0, 0.0, 3400.0]), rotation, motion="L")
    report = check_points([start, far], default_motion="L")
    assert report["ok"] is False
    failure = report["failures"][0]
    assert failure["point"] == 2
    assert failure["reason"] in {"unreachable", "branch_lost", "joint_limit"}
    assert failure["at_mm"] is not None


def test_a_program_without_any_joint_values_is_still_checked():
    """The whole reason the check does not reuse the gate: no joints recorded is normal."""
    position, rotation = looking_down()
    report = check_points([point(position, rotation)], default_motion="L")
    assert report["ok"] is True
    assert report["poses_solved_here"] == 1


def test_an_unreachable_first_point_is_reported_rather_than_assumed():
    position, rotation = looking_down((9000.0, 0.0, 1000.0))
    report = check_points([point(position, rotation)])
    assert report["ok"] is False
    assert report["failures"][0]["reason"] == "no_ik_solution"


def test_the_check_does_not_depend_on_robot_enabled():
    """`robot.enabled` describes the planning sieve, never whether the file can be flown."""
    position, rotation = looking_down()
    joints = list(first_solution(position, rotation))
    joints[4] = -122.0
    off = check_points([point(position, rotation, joints=tuple(joints))],
                       params=replace(RobotParams.disabled(), enabled=False))
    assert off["enabled"] is True
    assert off["ok"] is False
