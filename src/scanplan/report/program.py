"""The program block: what the workshop is handed, in motion terms."""

from __future__ import annotations

import numpy as np

from ..config import PlannerConfig
from ..path import PathPlan


def program_block(config: PlannerConfig, viewpoints, path: PathPlan | None) -> dict:
    """What the workshop is handed: the motion, the corners, and what the path costs."""
    target = config.target.coverage_target_pct
    block = {
        "motion": config.motion.type,
        "termination": config.motion.termination,
        "speed_mm_s": config.motion.speed_mm_s,
        "speed_note": ("set on the floor and deliberately not optimised here: scan speed "
                       "is a site setting, not a planning variable"),
        "delivery_mode": config.target.delivery_mode,
        "coverage_target_pct": target,
        "coverage_target_note": (
            "the acceptance figure in force for this run; null means the selection ran "
            "until no candidate added new coverage, which is the v1 behaviour"),
        "points": len(viewpoints),
        "coverage_points": sum(1 for vp in viewpoints if vp.role == "coverage"),
        "transition_points": sum(1 for vp in viewpoints if vp.role != "coverage"),
        "turns": len(viewpoints),
    }
    if path is None:
        block["ordering"] = ("planner order of single poses; nothing constrains it, so an "
                             "exporter may re-sort")
        return block
    block.update({
        "ordering": path.ordering,
        "path_length_mm": round(float(path.path_length_mm), 1),
        "interrupts": {"translation": int(path.interrupts_a),
                       "reorient": int(path.interrupts_b)},
        "interrupts_note": ("every one of these is the scanner idling while the head moves; "
                            "the site has accepted them"),
        "min_clearance_mm": (None if path.min_clearance_mm is None
                             else round(float(path.min_clearance_mm), 1)),
        "constraint_violations": list(path.violations),
        # Which route every transition took. Without this, "the connector got cheaper" is
        # a claim rather than a measurement.
        "connector_levels": dict(sorted(path.connector_levels.items())),
        "cost": _cost(config, path),
    })
    return block


def _cost(config: PlannerConfig, path: PathPlan) -> dict:
    """What the program costs, split into the half that converts and the half that does not.

    The linear half converts to seconds because the floor speed is a setting. The joint half
    does not: turning degrees into seconds needs a joint speed, and no joint speed has been
    measured on this cell. Reporting only the convertible half would hide the cost of the
    repositions, and inventing a joint speed would be worse than leaving the number as degrees
    - so the degrees are reported as degrees and the note says why.
    """
    points = list(getattr(path, "points", []) or [])
    solutions = list(getattr(getattr(path, "branches", None), "solutions", []) or [])
    linear_mm = 0.0
    joint_deg = 0.0
    worst_deg = 0.0
    linear_segments = 0
    joint_segments = 0
    for index in range(1, len(points)):
        step = float(np.linalg.norm(np.asarray(points[index].position, dtype=float)
                                    - np.asarray(points[index - 1].position, dtype=float)))
        motion = str(getattr(points[index], "motion", None) or config.motion.type).upper()
        if motion == "L":
            linear_mm += step
            linear_segments += 1
            continue
        joint_segments += 1
        first = solutions[index - 1] if index - 1 < len(solutions) else None
        second = solutions[index] if index < len(solutions) else None
        if first is None or second is None:
            continue
        delta = np.abs(np.asarray(second.joints_deg, dtype=float)
                       - np.asarray(first.joints_deg, dtype=float))
        joint_deg += float(delta.max())
        worst_deg = max(worst_deg, float(delta.max()))
    speed = float(config.motion.speed_mm_s)
    return {
        "linear_segments": linear_segments,
        "linear_mm": round(linear_mm, 1),
        "linear_seconds_at_floor_speed": (round(linear_mm / speed, 1) if speed > 0.0
                                          else None),
        "joint_segments": joint_segments,
        "joint_move_deg_total": round(joint_deg, 1),
        "joint_move_deg_worst": round(worst_deg, 1),
        "note": ("the linear half is seconds because the floor speed is a setting; the joint "
                 "half stays in degrees because no joint speed has been measured on this cell. "
                 "Read the degrees as the SHAPE of the transfer cost - a program with five "
                 "corners and a program with fifty differ mainly here"),
    }
