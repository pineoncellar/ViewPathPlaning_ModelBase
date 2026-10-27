"""The vocabulary of the report: what was targeted, what the numbers are, what is a guess.

This is the module to edit when the report's *wording* changes; the coverage figures
themselves live in :mod:`scanplan.report.coverage`.
"""

from __future__ import annotations

from ..config import PlannerConfig
from ..coverage import CoverageResult
from ..sheet_split import SheetSplit
from ..target_surface import TargetSet, contact_summary
from ..tracker import Tracker

EXEMPT_REASON = "fixture_contact"

# The note for a run where no arm was configured, or where the base pose is still unknown
# (`robot.enabled` off): the pre-arm wording, kept because it is still exactly true there.
SCOPE_NOTE = (
    "Coverage is GEOMETRIC reachability only. Robot reachability, joint limits and "
    "robot collision are NOT modelled here: they are decided later in ROBOGUIDE, so "
    "this viewpoint set deliberately contains poses ROBOGUIDE may reject. That is "
    "expected, not a defect - import the .LS, try another CONFIG before calling a "
    "pose unreachable, and feed the rejected poses back."
)

# The note for a run WITH an arm: joint limits are now part of the coverage figure, so
# saying they are not would be false. Collision still is not.
SCOPE_NOTE_ARM = (
    "Coverage is GEOMETRIC reachability filtered by the arm. Joint limits ARE enforced: "
    "every delivered pose - and every straight move between two of them - is solved "
    "against the robot's kinematic chain with each axis derated by `robot."
    "joint_limit_margin`, and anything that needs more travel than that is dropped before "
    "the selection. Robot COLLISION is still NOT modelled, and neither is the arm's "
    "occlusion of the tracker: import the .LS, confirm the clearances in ROBOGUIDE, and "
    "treat a pose it rejects as a question for the base pose or the fixture, not as a "
    "planning result."
)


def scope_note(config: PlannerConfig) -> str:
    """What this run's coverage figure does and does not include."""
    return SCOPE_NOTE_ARM if config.robot.enabled else SCOPE_NOTE

TRACKER_CAVEAT = (
    "The tracker test covers part, fixture and base only. Occlusion by the ROBOT ARM "
    "carrying the scanner is NOT modelled, because there is no robot model at this "
    "stage. Mount the tracker along the part's thin axis (along the part, not square "
    "on to its large faces) and keep it clear of the arm's normal working sector."
)

PLACEHOLDERS_NOTE = (
    "`placeholders` below lists the scanner and tracker values as delivered, and "
    "`parameter_provenance` says which of them are datasheet facts, which are "
    "derived from the datasheet, and which are still unmeasured. Only the unmeasured "
    "ones can move a coverage figure without changing the part: max_incidence_deg is "
    "the most sensitive of them and fov_anchor_distance_mm is the one that changes the "
    "footprint everywhere at once. Do not read a coverage figure as a prediction until "
    "those are measured."
)


def definition_block(*, config: PlannerConfig, target: TargetSet, split: SheetSplit,
                     coverage: CoverageResult) -> dict:
    """What the target was, before any coverage number is read."""
    return {
        "definition": {
            "name": "topological: one face of the original flat blank",
            "target": f"the {target.face} face of the blank, minus fixture contact",
            "split": "mirror partner per face -> mid-surface PCA -> local sheet normal m; "
                     "|n.m| ~ 1 = a face of the blank, ~ 0 = sheared cross-section; the "
                     "sign of n.m after orienting m says which face",
            "upper_area_mm2": round(split.areas["upper"], 1),
            "lower_area_mm2": round(split.areas["lower"], 1),
            "blanked_edge_area_mm2": round(split.areas["band"], 1),
            "material_thickness_mm": round(split.thickness_mm, 3),
            "orientation_confidence": round(split.oriented_fraction, 4),
            "not_target": ["the other face of the blank",
                           "blanked edge / sheared cross-section", EXEMPT_REASON],
            "contact_rule": (f"distance < {config.contact_distance_mm:.1f} mm AND on the "
                             "outward side of the sheet"),
            "sampling": f"{config.target.sample_grid_mm:g} mm grid, one sample per cell, "
                        "equal area share",
            "through_axis": [round(float(v), 6) for v in split.through_axis],
            "face_naming_note": (
                "`upper` and `lower` are GEOMETRIC names: they are the two sides of the "
                "blank as the split's orientation seed happens to order them, and that seed "
                "is not anchored to the station file. Swapping the seed swaps the names "
                "while leaving both areas, the thickness and the orientation fraction "
                "identical - so check `through_axis` above (which way the split calls "
                "'up') instead of trusting the word, and prefer face='auto', which compares "
                "both sides."
            ),
        },
        "face": target.face,
        "target_face_mm2": round(target.face_area_mm2, 1),
        "target_points": int(coverage.n_targets),
        "point_area_mm2": round(
            float(target.samples.total_area_mm2 / max(coverage.n_targets, 1)), 3),
        "exempt": {**contact_summary(target), "reason": EXEMPT_REASON},
    }


def placeholders_block(config: PlannerConfig, tracker: Tracker | None) -> dict:
    """The scanner and tracker values as delivered, measured or not."""
    return {
        "max_incidence_deg": config.scanner.max_incidence_deg,
        "range_mm": list(config.scanner.range_mm),
        "fov_at_standoff_mm": list(config.scanner.fov_at_standoff_mm),
        "fov_cap_mm": list(config.scanner.fov_cap_mm),
        "standoff_mm": config.scanner.standoff_mm,
        "body_sphere_r_mm": config.scanner.body_sphere_r_mm,
        "body_clearance_mm": config.scanner.body_clearance_mm,
        "tracker_half_angle_deg": config.tracker.half_angle_deg,
        "tracker_distance_mm": list(config.tracker.distance_mm),
        "tracker_model_position": ([round(float(v), 1) for v in tracker.position]
                                   if tracker is not None else None),
    }


def provenance(config: PlannerConfig) -> dict:
    """Which numbers are measured, which are derived, and which are still guesses.

    The report used to call every value a placeholder, which is as misleading as trusting
    them all: a reader cannot tell which number would move the coverage figure.
    """
    scanner = config.scanner
    return {
        "from_datasheet": {
            "standoff_mm": scanner.standoff_mm,
            "depth_of_field_mm": scanner.depth_of_field_mm,
            "fov_cap_mm": list(scanner.fov_cap_mm),
            "tracker_distance_mm": list(config.tracker.distance_mm),
        },
        "derived_from_datasheet": {
            "range_mm": list(scanner.range_mm),
            "tracker_half_angle_deg": config.tracker.half_angle_deg,
        },
        "unmeasured": {
            "max_incidence_deg": scanner.max_incidence_deg,
            "body_sphere_r_mm": scanner.body_sphere_r_mm,
            "body_clearance_mm": scanner.body_clearance_mm,
            "markers_omnidirectional": config.tracker.markers_omnidirectional,
            "fov_anchor_distance_mm": scanner.fov_anchor_distance_mm,
        },
        "design_choice": {
            "distances_mm": list(scanner.distances_mm),
            "tilts_deg": list(scanner.tilts_deg),
            "rolls_deg": list(scanner.rolls_deg),
            "max_stroke_mm": config.stroke.max_stroke_mm,
            "coverage_step_mm": config.stroke.coverage_step_mm,
            "check_step_mm": config.stroke.check_step_mm,
        },
        "note": ("the unmeasured block is the one to look at before believing a coverage "
                 "figure; max_incidence_deg is the most sensitive of them and "
                 "fov_anchor_distance_mm the one that moves the viewpoint count by ~20%"),
    }


def derived_block(config: PlannerConfig) -> dict:
    """The `surface_bias` profile: what it produced, and which layer wrote each key.

    One semantic input ("does this part have to be scanned completely, or does a regular
    program matter more?") drives six keys, and the report has to say so per key - both
    the value and whether it came from the base file, from the profile or from an explicit
    override. Without that, "why is min_gain 200?" is unanswerable from the report, which
    is how a derived parameter turns into a mystery number (merge plan 6.6).
    """
    derived = dict(getattr(config, "derived", {}) or {})
    keys = ("min_gain_mm2_per_step", "min_stroke_mm", "seed_dirs",
            "order_axis_weight_mm_per_deg", "parallel_weight", "repair_budget_share")
    return {
        "surface_bias": config.target.surface_bias,
        "surface_bias_source": getattr(config, "surface_bias_source", "base"),
        "derived_keys": {key: {"value": getattr(config.stroke, key),
                               "source": derived.get(key, "profile")} for key in keys},
        "note": ("0.0 is coverage first (grow further, accept more broken bands), 0.5 is "
                 "the shipped behaviour - at which every key below equals its factory "
                 "value - and 1.0 is regular programs first (stop earlier, fewer corners). "
                 "Writing a key explicitly in an override file always beats the profile, "
                 "which is why the source is recorded per key"),
        "excluded_on_purpose": {
            "scanner.max_incidence_deg": config.scanner.max_incidence_deg,
            "scanner.range_mm": list(config.scanner.range_mm),
            "reason": ("sensor physics, not taste: bending either of these to buy coverage "
                       "would fake a capability and make the report look better while the "
                       "cell cannot scan it"),
        },
    }
