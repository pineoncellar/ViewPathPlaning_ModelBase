"""Two delivery filters that the 2026-10-29 field run proved are not optional.

ROBOGUIDE stopped PART010 with `TPIF-012 / MOTN-018 position not reachable`. Two defects in
our own output produced that class of stop, and both are cheap to check here:

* a pose delivered at a COLLINEAR elbow (`bend = 8.4 deg` on that program) - the closed form
  still returns numbers, the controller refuses to solve it;
* a solution written with `N U T` whose J5 is POSITIVE, i.e. the other wrist branch. The
  letter is a constraint, not a label: the controller then asks for a branch we never checked.
"""

from __future__ import annotations

import numpy as np

from scanplan.config import RobotParams
from scanplan.kinematics import Robot
from scanplan.roboguide.adapt import joint_audit


def robot(**changes) -> Robot:
    params = RobotParams.disabled()
    fields = {**params.__dict__, "enabled": True, **changes}
    return Robot(RobotParams(**fields))


def a_collinear_pose():
    """Joints of a real posture with the links almost in line (bend ~ 177 deg)."""
    joints = (-7.90, -27.83, -66.70, 142.54, -14.01, -126.83)
    arm = robot()
    position, rotation = arm.forward(joints)
    return arm, joints, position, rotation


def test_a_nearly_straight_arm_has_no_deliverable_solution_by_default():
    arm, joints, position, rotation = a_collinear_pose()
    assert arm.elbow_bend_deg(joints) > 170.0
    # `disabled()` carries no delivery filters by design; the shipped configuration does.
    delivered = robot(elbow_margin_deg=10.0)
    assert delivered.solutions(position, rotation) == []
    assert delivered.feasible(joints) is False


def test_the_same_pose_is_solved_when_the_elbow_margin_is_switched_off():
    """The margin is a DELIVERY rule, not a limit of the solver: geometry still solves it."""
    _, joints, position, rotation = a_collinear_pose()
    plain = robot(elbow_margin_deg=0.0)
    assert plain.solutions(position, rotation)
    assert plain.feasible(joints) is True


def test_a_pinned_wrist_letter_only_returns_solutions_of_that_branch():
    """`N U T` means J5 < 0. Writing it over a positive-J5 solution is a different pose."""
    arm = robot()
    joints = (15.0, 26.0, 0.6, -133.3, -108.1, -85.7)
    position, rotation = arm.forward(joints)

    for letter, sign in (("T", -1.0), ("F", 1.0)):
        pinned = robot(config_wrist_letter=letter)
        found = [item for item in pinned.solutions(position, rotation) if item.ok]
        assert found, f"the {letter} branch must exist for this pose"
        assert all(float(item.joints_deg[4]) * sign > 0.0 for item in found)
        assert all(item.config.split(",")[0].strip().endswith(letter) for item in found)

    every = robot(config_wrist_letter="auto")
    signs = {float(item.joints_deg[4]) > 0.0 for item in every.solutions(position, rotation)
             if item.ok}
    assert signs == {True, False}, "an unpinned gate reports both branches"


def test_the_export_audit_refuses_a_collinear_elbow():
    """The last line of defence: a point the controller would refuse never gets written."""
    params = RobotParams.disabled()
    strict = RobotParams(**{**params.__dict__, "enabled": True, "elbow_margin_deg": 10.0})
    arm, joints, position, rotation = a_collinear_pose()
    point = {"vp_key": 1, "xyz_mm": position, "rotation": rotation,
             "joints_deg": joints, "config": "N U T, 0, 0, 0"}
    audit = joint_audit([point], strict, check_pose=False)
    assert audit["enabled"] is True
    assert [item["point"] for item in audit["illegal"]] == [1]
    # ... and the same point passes when the margin is off, which is what the old
    # behaviour was: the audit only checked the six travel limits.
    lenient = joint_audit([point], params, check_pose=False)
    assert lenient["enabled"] is False or lenient["illegal"] == []
    assert np.all(np.isfinite(np.asarray(joints, dtype=float)))
