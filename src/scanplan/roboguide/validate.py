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
from ..kinematics import Robot, rotation_angle_deg

# The planner's own fine step, reused rather than exposed: a delivery check that resolves the
# line more coarsely than the planning check would disagree with it for no reason.
STEP_MM = 10.0
# How far the joints the walk arrives at may sit from the joints the plan recorded before the
# two are called different branches. The same 1 degree `Robot.branch_of` uses to identify one.
BRANCH_TOLERANCE_DEG = 1.0
# How far a joint-representation point's recorded joints may be from its recorded pose. A file
# whose `P[n]` says "J1..J6 = ..." is flown by those numbers, so a mismatch means the point
# that was planned and the point that will be reached are two different points.
POSE_TOL_MM = 1.0
POSE_TOL_DEG = 0.2
# How far a straight move may arrive from the rail reading its block claims, in mm. The same
# order as the pose tolerance: this is rounding, not preference.
TRACK_TOL_MM = 1.0


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
    held_track = 0.0
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
        # The seventh axis. Absent means a floor-mounted arm, which is 0.0 and commutes with
        # every check below unchanged.
        rail_raw = point.get("track_mm")
        rail = 0.0 if rail_raw is None else float(rail_raw)
        if target is not None and robot.feasible(target, rail):
            # A record has to agree with itself before anything is walked: the joints AND the
            # rail reading must reproduce the XYZ the block claims. With a rail this is not
            # redundant bookkeeping - the rail makes the arm redundant, so a wrong `E1` still
            # solves, it just lands somewhere else. Gated on feasibility so a pose that is
            # simply out of band is reported as `joint_limit` (the reason someone can act on)
            # rather than as a record that disagrees with itself.
            reached, frame = robot.forward(target, rail)
            gap_mm = float(np.linalg.norm(
                np.asarray(reached, dtype=float).reshape(3) - position))
            gap_deg = rotation_angle_deg(frame, rotation)
            if gap_mm > POSE_TOL_MM or gap_deg > POSE_TOL_DEG:
                fail(index,
                     "track_mismatch" if robot.track_enabled else "joint_pose_mismatch",
                     detail=f"joints + E1 = {rail:.1f} mm land "
                            f"{gap_mm:.2f} mm / {gap_deg:.2f} deg from the block's XYZ"
                            if robot.track_enabled else
                            f"{gap_mm:.2f} mm / {gap_deg:.2f} deg from its own XYZ")

        if held is None:
            if target is not None:
                if not robot.feasible(target, rail):
                    fail(index, "joint_limit", axis=int(np.argmin(robot.margins(target))))
                held = target
                held_track = rail
            else:
                found = [item for item in robot.solutions(position, rotation,
                                                          track_mm=rail) if item.ok]
                if not found:
                    fail(index, "no_ik_solution")
                else:
                    solved += 1
                    held = np.asarray(found[0].joints_deg, dtype=float)
                    held_track = rail
            continue

        segments += 1
        if motion == "J":
            if target is None:
                found = [item for item in robot.solutions(position, rotation,
                                                          track_mm=rail) if item.ok]
                if not found:
                    fail(index, "no_ik_solution")
                else:
                    solved += 1
                    held = np.asarray(found[0].joints_deg, dtype=float)
                    held_track = rail
                continue
            if not robot.feasible(target, rail):
                # out-of-band target: the move cannot be flown, then carry on from it so one
                # bad point does not blame every point after it
                fail(index, "joint_limit", axis=int(np.argmin(robot.margins(target))))
                held = target
                held_track = rail
                continue
            # A joint-representation point is flown by the numbers in it; the record's own
            # consistency was checked above.
            if float(held[4]) * float(target[4]) < 0.0:
                fail(index, "j5_crossover")
            held = target
            held_track = rail
            continue

        verdict = robot.follow(held, position, rotation, step_mm=STEP_MM,
                               track_start=held_track, track_end=rail)
        if verdict.ok:
            arrived = np.asarray(verdict.joints_deg, dtype=float)
            # The record's own consistency (joints + E1 against XYZ) was checked above.
            if target is not None:
                gap = _joint_gap_deg(arrived, target)
                if gap > BRANCH_TOLERANCE_DEG:
                    fail(index, "branch_mismatch", at_deg=round(gap, 2))
            held = arrived
            held_track = rail
            continue
        fail(index, verdict.reason or "unverified",
             axis=None if verdict.limiting_axis is None else int(verdict.limiting_axis),
             at_mm=None if verdict.failure_mm is None else round(float(verdict.failure_mm), 1))
        # continue from where the pose can be held, so the report names the first failure of
        # each run of failures rather than every point downstream of the first one
        found = [item for item in robot.solutions(position, rotation, track_mm=rail) if item.ok]
        if found:
            held = np.asarray(found[0].joints_deg, dtype=float)
            held_track = rail

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


def check_config_hints(points: list[dict]) -> dict:
    """The CONFIG column, checked against the joints it claims to describe.

    Two things make a CONFIG unreadable or wrong, and both are cheap to detect here:

    * a NEGATIVE turn digit. FANUC has no such thing, and ROBOGUIDE refuses the whole
      program at that column with ASBN-009 rather than skipping the point. A delivery whose
      J1 was -102 deg carried `N U F, -1, -1, -1` and never loaded.
    * a wrist letter that disagrees with the sign of J5. `T` is the negative branch; writing
      it over a positive-J5 solution asks the controller for the other wrist branch, which
      is the same pose turned 180 degrees (MOTN-063 / MOTN-018 on the cell).

    Returns the same shape as `check_points` so the two can be merged into one refusal.
    """
    failures: list[dict] = []
    for index, point in enumerate(points):
        config = point.get("config")
        joints = point.get("joints_deg")
        if not config or joints is None:
            continue
        head, _, tail = str(config).partition(",")
        words = head.split()
        wrist = words[-1].upper() if words else ""
        j5 = float(np.asarray(joints, dtype=float).reshape(6)[4])
        if wrist == "T" and j5 > 0.0:
            failures.append({"point": index + 1, "reason": "config_wrist_mismatch",
                             "detail": f"written T over J5={j5:.2f}"})
        elif wrist == "F" and j5 < 0.0:
            failures.append({"point": index + 1, "reason": "config_wrist_mismatch",
                             "detail": f"written F over J5={j5:.2f}"})
        for digit in tail.split(","):
            text = digit.strip()
            if not text:
                continue
            try:
                value = int(text)
            except ValueError:
                failures.append({"point": index + 1, "reason": "config_unreadable",
                                 "detail": f"turn digit {text!r}"})
                continue
            if value < 0:
                failures.append({"point": index + 1, "reason": "config_negative_turn",
                                 "detail": f"turn digit {value}"})
    return {
        "enabled": True,
        "ok": not failures,
        "points": len(points),
        "failures": failures,
        "note": ("the CONFIG column is a promise to the controller: negative turn digits are "
                 "refused at load (ASBN-009) and a wrist letter over the other branch asks for "
                 "a different pose (MOTN-063)"),
    }

