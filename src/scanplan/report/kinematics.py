"""The kinematics, budget and surface-bias sections of the report.

Three blocks that all answer the same kind of question - "why does the plan look like
this?" - and that were impossible to answer before:

* `kinematics` says whether the joint gate ran, which envelope it used, where the delivered
  poses sit relative to the limits, and how far the worst joint step is. Without it a plan
  that silently dropped a third of its candidates to the joint limits is indistinguishable
  from one that simply could not reach them.
* `budget` says which of the three end states the run reached - met the coverage target,
  ran into the point ceiling, or ran out of candidates - and what each extra tolerance
  bought, so "four times the points for half a percent" is a number.
* `surface_profile` says where each derived growth/ordering threshold came from, because a
  knob that silently changes six other numbers has to be auditable.
"""

from __future__ import annotations

import numpy as np

from ..config import SURFACE_ANCHORS, PlannerConfig


def kinematics_block(config: PlannerConfig, viewpoints, path, kinematics=None) -> dict:
    """Joint-side evidence: limits, J5 distribution, branches, worst step."""
    robot = getattr(kinematics, "robot", None)
    assignment = getattr(path, "branches", None) if path is not None else None
    solutions = [item for item in getattr(assignment, "solutions", []) if item is not None]
    joints = np.array([item.joints_deg for item in solutions], dtype=float).reshape(-1, 6)
    histogram: dict[str, int] = {}
    if len(joints):
        magnitudes = np.abs(joints[:, 4])
        edges = (0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 180.0)
        for low, high in zip(edges[:-1], edges[1:], strict=True):
            count = int(((magnitudes >= low) & (magnitudes < high)).sum())
            if count:
                histogram[f"{low:.0f}-{high:.0f}"] = count
    block = {
        "enabled": bool(getattr(kinematics, "active", False)),
        "model": config.robot.model,
        "limits_source": (robot.limits_source if robot is not None else None),
        "limits_usage_frac": config.robot.limits_usage_frac,
        "datasheet_limits_deg": [list(pair) for pair in config.robot.mechanical_limits_deg],
        "usable_limits_deg": ([list(pair) for pair in robot.limits]
                             if robot is not None else None),
        "deliverable_limits_deg": (
            [[low + config.robot.joint_margin_deg, high - config.robot.joint_margin_deg]
             for low, high in robot.limits] if robot is not None else None),
        "wrist_identified": bool(config.robot.wrist_identified),
        "allowed_configs": list(config.robot.allowed_configs),
        "joint_margin_deg": config.robot.joint_margin_deg,
        "j5_min_abs_deg": config.robot.j5_min_abs_deg,
        "max_step_deg_limit": config.robot.max_step_deg,
        "limits_note": (
            "usable_limits_deg is the datasheet envelope intersected with the controller's "
            "set after shrinking it to limits_usage_frac; deliverable_limits_deg is that "
            "envelope with joint_margin_deg reserved at each end, which is what every "
            "accepted pose satisfies"),
        "j5_histogram_deg": histogram,
        "branch_histogram": dict(getattr(assignment, "histogram", {})),
        "min_margin_deg": getattr(assignment, "min_margin_deg", None),
        "max_step_deg": getattr(assignment, "max_step_deg", None),
        "max_step_at": getattr(assignment, "max_step_at", None),
        "max_j6_step_deg": getattr(assignment, "max_j6_step_deg", None),
        "max_j6_step_deg_limit": config.stroke.max_j6_step_deg,
        "max_j6_step_at": getattr(assignment, "max_j6_step_at", None),
        "j6_steps_over_limit": len(getattr(assignment, "j6_over", [])),
        "j6_step_weight": config.stroke.j6_step_weight,
        "j6_note": (
            "a sweep is written as one L move between two points of the same constant pose, "
            "so J6 does not turn while the head scans; max_j6_step_deg is the largest J6 turn "
            "asked of a single re-positioning move, and j6_steps_over_limit counts the moves "
            "above stroke.max_j6_step_deg (a review trigger, not a drop rule)"),
        "motion_check": {
            "step_mm": config.robot.motion_check_step_mm,
            "step_deg": config.robot.motion_check_step_deg,
            "moves_checked": (kinematics.summary().get("moves_checked")
                              if kinematics is not None else None),
            "samples": (kinematics.summary().get("move_samples")
                        if kinematics is not None else None),
            "note": (
                "every move is sampled between its corners and the IK is followed from the "
                "branch the previous point ended on: a legal corner pair can still leave the "
                "travel in the middle, and the controller flies the segment, not the corners"),
        },
        "points_without_solution": len(getattr(assignment, "failed", [])),
        "steps_over_limit": len(getattr(assignment, "broken", [])),
        "cache": (kinematics.summary() if kinematics is not None else None),
        "config_note": (
            "the CONFIG letters are derived from the chosen joint solution. The T/F letter was "
            "verified on the 2026-10-23 bank of eight poses (T for negative J5, 8 of 8); the "
            "elbow letter saw only U samples and the turn digits match nothing the pendant "
            "showed, so both stay hints"),
    }
    if not config.robot.wrist_identified:
        block["joint_values_withheld"] = (
            "wrist_identified is false, so absolute joint values are not reported; the gate "
            "may only rank branches against each other")
    if viewpoints:
        records = [viewpoint.as_dict() for viewpoint in viewpoints]
        block["points_with_config"] = sum(1 for item in records if item.get("config"))
    return block


def budget_block(config: PlannerConfig, path) -> dict:
    """Which end state the run reached, and what each tolerance bought."""
    history = getattr(path, "budget_history", []) if path is not None else []
    verdict = getattr(path, "verdict", "") if path is not None else ""
    delivered = getattr(path, "turns", None) if path is not None else None
    dropped = len(getattr(path, "dropped", [])) if path is not None else 0
    return {
        "max_points": config.target.max_points,
        "delivered_points": delivered,
        "verdict": verdict or ("unbudgeted" if config.target.max_points is None
                               else "unknown"),
        "dropped_by_hard_gate": dropped,
        "trials": [{"lambda": float(lam), "points": int(points), "coverage_pct": float(pct)}
                   for lam, points, pct in history],
        "note": ("verdict is one of target_bound (reached the coverage target first), "
                 "budget_bound (ran into max_points first) or exhausted (the pool had "
                 "nothing more to offer); a fourth answer does not exist"),
    }


def surface_profile_block(config: PlannerConfig) -> dict:
    """The bias in force, the values it derived, and where each one came from."""
    keys = tuple(SURFACE_ANCHORS[0][1])
    derived = {f"stroke.{key}": getattr(config.stroke, key) for key in keys}
    anchors = {f"{anchor:g}": {key: float(values[key]) for key in keys}
               for anchor, values in SURFACE_ANCHORS}
    return {
        "surface_bias": float(config.target.surface_bias),
        "effective": derived,
        "sources": dict(config.parameter_sources),
        "anchors": anchors,
        "note": ("a value written in the override file wins over the profile, and the "
                 "source column says so; 0.5 must reproduce the pre-2026-10-22 values "
                 "exactly, which is what the anchor table is pinned to"),
    }
