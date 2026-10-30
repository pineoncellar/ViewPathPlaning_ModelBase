"""Two delivery filters that decide whether a planned point can actually be commanded.

Both were added after a real program (2026-10-29, PART010) stopped on the pendant with
TPIF-012 / MOTN-018 on a point the planner had accepted:

* the CONFIG wrist letter was PINNED to `T` while 13 of the 27 points had been solved on
  the `F` branch (J5 positive). The controller reads the letters, takes the other branch,
  and where that branch is outside the travel it reports the point as unreachable;
* one point sat at an elbow bend of 8.4 deg - the two links almost collinear - which is the
  posture the fitted geometry describes worst and the controller refuses to solve.
"""

from __future__ import annotations

import numpy as np
import pytest

from scanplan.candidates import ladder
from scanplan.config import RobotParams
from scanplan.kinematics import KinematicGate, Robot
from scanplan.rolls import roll_for_axis


def gate(**changes) -> KinematicGate:
    params = RobotParams.disabled()
    return KinematicGate(RobotParams(**{**params.__dict__, "enabled": True, **changes}))


def pose_with_bend(arm: Robot, target_deg: float):
    """A reachable joint set whose elbow bend is as close to `target_deg` as we can get."""
    best = None
    for j2 in np.arange(-50.0, 40.0, 2.0):
        for j3 in np.arange(-100.0, 100.0, 2.0):
            joints = (0.0, float(j2), float(j3), 0.0, -60.0, 0.0)
            bend = arm.elbow_bend_deg(joints)
            if best is None or abs(bend - target_deg) < abs(best[0] - target_deg):
                best = (bend, joints)
    return best


def test_a_pinned_wrist_letter_is_a_constraint_not_a_label():
    """`N U T` means the negative-J5 branch. Writing it over an F solution is a lie."""
    pose = (np.array([2000.0, 0.0, 1000.0]), np.array([0.0, 0.0, -1.0]), 0.0)

    def branches(**changes):
        arm = gate(**changes)
        found = arm.solutions(*pose)
        return [solution.joints_deg[4] for solution in found]

    negative = branches(config_wrist_letter="T")
    positive = branches(config_wrist_letter="F")
    both = branches(config_wrist_letter="auto")
    assert negative and all(value < 0.0 for value in negative)
    assert positive and all(value > 0.0 for value in positive)
    assert any(value < 0.0 for value in both) and any(value > 0.0 for value in both)


def test_the_pinned_branch_still_reproduces_the_pose():
    """Filtering a branch must not change which TCP pose the gate accepts."""
    arm = gate(config_wrist_letter="T")
    position = np.array([2000.0, 0.0, 1000.0])
    axis = np.array([0.0, 0.0, -1.0])
    verdict = arm.check(position, axis, 0.0)
    assert verdict.ok
    assert verdict.solution.joints_deg[4] < 0.0
    assert arm.robot.reproduces(verdict.solution.joints_deg, position,
                                verdict.solution and _frame(axis, 0.0))


def _frame(axis, roll_deg):
    from scanplan.geometry import scanner_frame

    return scanner_frame(np.asarray(axis, dtype=float), float(roll_deg))


def test_a_collinear_elbow_solution_is_filtered_out():
    """`bend -> 0` is the posture the controller answers with MOTN-018.

    The rule is per SOLUTION, not per pose: one TCP can be reached with a normal bend from
    the other shoulder side, and that branch is perfectly commandable. What must never reach
    the controller is the joint set whose links are collinear.
    """
    loose = Robot(RobotParams(**{**RobotParams.disabled().__dict__, "enabled": True,
                                 "elbow_margin_deg": 0.0}))
    picky = Robot(RobotParams(**{**RobotParams.disabled().__dict__, "enabled": True,
                                 "elbow_margin_deg": 10.0}))
    bend, joints = pose_with_bend(loose, 5.0)
    assert bend < 10.0
    position, rotation = loose.forward(joints)

    offered = [picky.elbow_bend_deg(item.joints_deg)
               for item in picky.solutions(position, rotation)]
    unfiltered = [loose.elbow_bend_deg(item.joints_deg)
                  for item in loose.solutions(position, rotation)]
    assert unfiltered, "the unfiltered solver must still find this pose"
    # every branch of THIS pose has collinear links (one side folded, the other straight),
    # so the filtered solver finds nothing at all - which is the point of the rule.
    assert all(value < 10.0 or value > 170.0 for value in unfiltered)
    assert offered == []


def test_a_normal_posture_is_untouched_by_the_margin():
    picky = gate(elbow_margin_deg=10.0)
    arm = picky.robot
    bend, joints = pose_with_bend(arm, 90.0)
    assert 80.0 < bend < 100.0
    position, rotation = arm.forward(joints)
    assert picky.check(position, rotation[:, 2], 0.0).ok


def test_the_elbow_margin_also_reaches_the_export_audit():
    """An old viewpoint file must be refused before a program is written."""
    from scanplan.config import load_config

    cfg = load_config()
    arm = Robot(cfg.robot)
    _, joints = pose_with_bend(arm, 5.0)
    assert not arm.feasible(joints)


def test_a_locked_roll_follows_the_axis_in_the_candidate_ladder(clamped_plate, cfg):
    """Roll 0 next to a roll-90 sweep is a 90 deg J6 turn in the middle of the program."""
    normal = np.array([0.0, 0.0, 1.0])
    point = np.array([0.0, 0.0, 0.0])
    loose = ladder(point, normal, cfg.scanner, lock_roll=False)
    locked = ladder(point, normal, cfg.scanner, lock_roll=True)
    assert {round(pose.roll_deg, 6) for pose in loose} >= {0.0, 90.0}
    for pose in locked:
        assert pose.roll_deg == pytest.approx(roll_for_axis(pose.axis), abs=1e-9)
    assert len(locked) < len(loose)
