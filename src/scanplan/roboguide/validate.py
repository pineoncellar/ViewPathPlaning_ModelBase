"""The joint side of a delivered program, checked whether or not the planner looked.

Why this exists separately from `scanplan.kinematics.KinematicGate`: the gate is a sieve over
candidate poses, and the site can switch it off to make a run faster or to reproduce an old
number. That switch says something about the SEARCH, never about whether the arm can fly the
file. A delivery check that read the same flag would mean one unchecked checkbox ships an
unverified program - which is exactly what happened: `X10290` was written with a constant
`N U T` on every point, imported cleanly, and died at J2 in the middle of its last segment.

Two further rules, both learned the hard way:

* the `Robot` here is a FRESH one. Sharing the planner's per-pose cache would let an answer
  computed under another configuration validate this file.
* "unverified" is not "passed". A segment this cannot check is reported as a failure with the
  reason `unverified`, not skipped.

What it deliberately does NOT do: collision. There is no station here, and pretending the TCP
is a point would be worse than saying so. The plan-side connector check does that, with the
station in hand. This one is the joint side only: limits, the J5 band, the wrist singularity,
and the branch the arm has to hold.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..config import RobotParams
from ..kinematics import Robot

# The planner's own fine step, reused rather than exposed: a delivery check that resolves the
# line more coarsely than the planning check would disagree with it for no reason.
STEP_MM = 10.0
# How far the joints the walk arrives at may sit from the joints the plan recorded before the
# two are called different branches. The same 1 degree `Robot.branch_of` uses to identify one.
BRANCH_TOLERANCE_DEG = 1.0


def _joint_gap_deg(first, second) -> float:
    """Largest per-axis difference, circularly, the way the controller measures a step."""
    delta = np.asarray(second, dtype=float) - np.asarray(first, dtype=float)
    return float(np.max(np.abs((delta + 180.0) % 360.0 - 180.0)))


def check_points(points: list[dict], *, default_motion: str = "L",
                 params: RobotParams | None = None) -> dict:
    """Walk the point table the way the controller will, and report what it cannot fly.

    The decision is per MOTION, not per "does this point have joint values". A scan point is
    written as a Cartesian position and flown as an `L`, so it has to be walked along the line
    holding the branch the arm is on - that is where `MOTN-017` came from, and having joints
    recorded for it does not change what the controller does. When it does have joints, the
    walk's arrival is additionally compared against them, because a plan whose record and whose
    motion disagree is not verifiable.

    A `J` point is a joint-representation point: the arm goes exactly there, so both ends are
    checked for margin and for the J5 band, and there is no middle to fall out of. A `J` point
    without joint values lets the controller solve it, so only the pose can be checked - and
    that is stated (`poses_solved_here`) rather than assumed away.

    The first point is the program's start, reached from wherever the arm already is; only its
    own pose is checked, because the pose it starts from is not in the plan.
    """
    base = params or RobotParams.disabled()
    # 状态：交付校验永不关闭，所以这里单独把臂打开，不借用规划器的开关或缓存
    robot = Robot(replace(base, enabled=True))
    failures: list[dict] = []
    held: np.ndarray | None = None
    segments = 0
    solved = 0

    def fail(index: int, reason: str, **extra) -> None:
        failures.append({"point": int(index) + 1, "reason": reason, **extra})

    for index, point in enumerate(points):
        position = np.asarray(point["xyz_mm"], dtype=float).reshape(3)
        rotation = np.asarray(point["rotation"], dtype=float).reshape(3, 3)
        motion = str(point.get("motion") or default_motion).upper()
        raw = point.get("joints_deg")
        target = None if raw is None else np.asarray(raw, dtype=float).reshape(6)

        if held is None:
            if target is not None:
                if not robot.feasible(target):
                    fail(index, "joint_limit", axis=int(np.argmin(robot.margins(target))))
                held = target
            else:
                found = [item for item in robot.solutions(position, rotation) if item.ok]
                if not found:
                    fail(index, "no_ik_solution")
                else:
                    solved += 1
                    held = np.asarray(found[0].joints_deg, dtype=float)
            continue

        segments += 1
        if motion == "J":
            if target is None:
                found = [item for item in robot.solutions(position, rotation) if item.ok]
                if not found:
                    fail(index, "no_ik_solution")
                else:
                    solved += 1
                    held = np.asarray(found[0].joints_deg, dtype=float)
                continue
            if not robot.feasible(target):
                # out-of-band target: the move cannot be flown, then carry on from it so one
                # bad point does not blame every point after it
                fail(index, "joint_limit", axis=int(np.argmin(robot.margins(target))))
                held = target
                continue
            if float(held[4]) * float(target[4]) < 0.0:
                fail(index, "j5_crossover")
            held = target
            continue

        verdict = robot.follow(held, position, rotation, step_mm=STEP_MM)
        if verdict.ok:
            arrived = np.asarray(verdict.joints_deg, dtype=float)
            if target is not None:
                gap = _joint_gap_deg(arrived, target)
                if gap > BRANCH_TOLERANCE_DEG:
                    fail(index, "branch_mismatch", at_deg=round(gap, 2))
            held = arrived
            continue
        fail(index, verdict.reason or "unverified",
             axis=None if verdict.limiting_axis is None else int(verdict.limiting_axis),
             at_mm=None if verdict.failure_mm is None else round(float(verdict.failure_mm), 1))
        # continue from where the pose can be held, so the report names the first failure of
        # each run of failures rather than every point downstream of the first one
        found = [item for item in robot.solutions(position, rotation) if item.ok]
        if found:
            held = np.asarray(found[0].joints_deg, dtype=float)

    return {
        "enabled": True,
        "ok": not failures,
        "points": len(points),
        "segments_checked": segments,
        "poses_solved_here": solved,
        "failures": failures,
        "limits_source": robot.limits_source,
        "note": (
            "joint side only: limits, the usable band, the J5 band, the wrist singularity and "
            "the held branch. Collision is checked by the planner's connector, with the "
            "station; robot-arm occlusion of the tracker is not modelled at all. This runs "
            "whether or not robot.enabled is set, because that switch describes the planning "
            "sieve, not whether the arm can fly the file"
        ),
    }
