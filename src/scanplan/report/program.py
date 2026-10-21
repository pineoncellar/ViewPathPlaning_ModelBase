"""The program block: what the workshop is handed, in motion terms."""

from __future__ import annotations

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
    })
    return block
