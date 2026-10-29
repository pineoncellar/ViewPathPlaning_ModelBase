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
        "usable_limits_deg": ([list(pair) for pair in robot.usable]
                              if robot is not None else None),
        "wrist_identified": bool(config.robot.wrist_identified),
        "joint_margin_deg": config.robot.joint_margin_deg,
        "joint_span_fraction": config.robot.joint_span_fraction,
        "wrist_flip_policy": config.robot.wrist_flip_policy,
        "j5_min_abs_deg": config.robot.j5_min_abs_deg,
        "max_step_deg_limit": config.robot.max_step_deg,
        "j5_histogram_deg": histogram,
        "branch_histogram": dict(getattr(assignment, "histogram", {})),
        "min_margin_deg": getattr(assignment, "min_margin_deg", None),
        "max_step_deg": getattr(assignment, "max_step_deg", None),
        "max_step_at": getattr(assignment, "max_step_at", None),
        "points_without_solution": len(getattr(assignment, "failed", [])),
        "steps_over_limit": len(getattr(assignment, "broken", [])),
        # Whether the joint state was decided while the path was built, and what the
        # connector had to give up on. Both exist because "the plan got worse" needs a
        # cause, and the old single sentence covered four of them.
        "joint_planned": bool(getattr(path, "joint_planned", False)) if path else False,
        "joint_connector": bool(config.stroke.joint_connector),
        "connector_failures": dict(sorted(getattr(path, "connector_failures", {}).items()))
        if path is not None else {},
        "cache": (kinematics.summary() if kinematics is not None else None),
        "config_note": (
            "the CONFIG letters are derived from the chosen joint solution. The T/F letter was "
            "verified on the 2026-10-23 bank of eight poses (T for negative J5, 8 of 8), and "
            "the turn digits now follow the pendant's convention (a reading inside [-180, 180) "
            "is turn 0, which reproduces the bank's 0, 0, 0); the elbow letter saw only U "
            "samples, so it stays a hint"),
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


def failure_attribution_block(config: PlannerConfig, path, uncovered_by_reason) -> dict:
    """Why things were lost, in three separate tables because they are three separate layers.

    The old report had one sentence - "no route into it keeps clearance and rate" - covering
    clearance, turn rate, joint reach and "nothing else left". That sentence cannot be acted
    on: fixing it might mean the fixture, the connector, or the limit policy. These counters
    can.

    * `sample_reasons` is about SAMPLES: why a piece of the part is not covered. Geometric.
    * `item_drop_reasons` is about ITEMS: why a whole sweep or pose left the program.
    * `connector_primitive_refusals` is about ATTEMPTS: how often each connector primitive was
      tried and refused. A refusal is not a loss unless the item was then dropped - the joint
      move is refused on almost every transition that ends up using the retract fallback.
    """
    sample = dict(uncovered_by_reason or {})
    drops = dict(getattr(path, "drop_reasons", {}) or {})
    refusals = dict(getattr(path, "connector_failures", {}) or {})
    return {
        "sample_reasons": sample,
        "item_drop_reasons": drops,
        "connector_primitive_refusals": refusals,
        "layers_note": {
            "sample_reasons": "why a SAMPLE is uncovered; geometric, and the only one of the "
                              "three that moves the coverage figure",
            "item_drop_reasons": "why an ITEM (a sweep or a pose) left the program",
            "connector_primitive_refusals": "how often each connector primitive was tried and "
                                            "refused; a refusal is not a loss unless the item "
                                            "was then dropped",
        },
        "joint_gate_enabled": bool(config.robot.enabled),
        "note": ("with the joint gate off, `item_drop_reasons` only ever says the hard gate "
                 "found no connector - there is no arm to refuse anything"),
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
