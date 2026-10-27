"""The program block: what the workshop is handed, in motion terms."""

from __future__ import annotations

from ..config import PlannerConfig
from ..path import PathPlan
from ..selection import Selection


def program_block(config: PlannerConfig, viewpoints, path: PathPlan | None,
                  selection: Selection | None = None) -> dict:
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
        "max_points": config.target.max_points,
        "max_points_note": (
            "the program point cap in force, or null when the budget layer was skipped "
            "entirely - in which case the selection and ordering are exactly the ones "
            "without a budget, by construction"),
        "points": len(viewpoints),
        "coverage_points": sum(1 for vp in viewpoints if vp.role == "coverage"),
        "transition_points": sum(1 for vp in viewpoints if vp.role != "coverage"),
        "turns": len(viewpoints),
    }
    if selection is not None:
        # Which of the three limits stopped the run. Reported for every run, because the
        # three ask for three different next steps: lower the acceptance figure, raise the
        # cap, or change the scan direction.
        block["bound"] = selection.bound or None
        block["bound_note"] = ("`target_bound` met the coverage target first, "
                               "`budget_bound` hit the point cap first, `exhausted` means "
                               "the candidate pool had nothing more to offer")
        block["budget"] = dict(selection.budget) if selection.budget else None
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
        # How far the wrist has to turn between consecutive delivered points, split into
        # tilt (the surface sets it), extra twist (the roll policy sets it) and the whole
        # turn. Three numbers, because they have three different fixes.
        "wrist": dict(path.wrist) if path.wrist else None,
        "wrist_note": ("`rotation_deg` is what the wrist must deliver between two corners; "
                       "`twist_deg` is the part of it that is a pure J6 turn about the "
                       "optical axis and that the seed roll policy exists to keep small; "
                       "`moves_over_limit` counts the moves past stroke.max_wrist_step_deg "
                       "and they are listed in `worst`, because a program that hides them "
                       "gets rejected in ROBOGUIDE instead"),
        # What the delivered sequence looks like in joint space. Empty when no arm was
        # configured, which is the honest record of "nobody checked".
        "joints": dict(path.joints) if path.joints else None,
        "joints_note": ("every delivered point and every straight move between two of them "
                        "was solved against the arm's derated window; "
                        "`out_of_window_moves` counts the ones that had to be delivered "
                        "anyway (they are also named in `constraint_violations`), and "
                        "`worst_utilization` is the largest share of an allowed half-travel "
                        "any accepted configuration used - 1.0 would be the derated bound"),
    })
    return block
