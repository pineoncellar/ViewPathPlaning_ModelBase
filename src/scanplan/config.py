"""Planner configuration: scanner limits, sampling, tracker.

Holds the scanner envelope in exactly one place, so replacing a measured value
with the real one stays a one-file change. Which values are datasheet facts,
which are derived from them, and which are still unmeasured is recorded in
`configs/scanner_trackscan49.json` and in the design document's section 1.4 —
`max_incidence_deg` is the most sensitive of the unmeasured ones.

Loads overrides from a JSON file (see `configs/`).

An override file may name only the keys it wants to change: it is merged into
the shipped defaults. Unknown sections and unknown keys are refused, because a
mistyped parameter would otherwise leave a placeholder silently in force.
"""

from __future__ import annotations

import json
import re
import types
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

DEFAULT_CONFIG_NAME = "scanner_trackscan49.json"
FACES = ("upper", "lower", "auto")
# How the delivered program is assembled. `path` checks the motion between every pair of
# delivered points; `lean` delivers the poses alone, exactly like v1, and promises nothing
# about what the head does between them.
DELIVERY_MODES = ("path", "lean")
# What the selection optimises against. Millimetres is the upstream design's choice and is
# kept so the two can be compared on the same part; corners is what the workshop counts.
COST_MODELS = ("corners", "millimetres")


@dataclass(frozen=True)
class ScannerParams:
    """The scanner's geometric and quality envelope.

    Provenance of each value is tracked in the design document
    (docs/superpowers/specs/2026-10-19-continuous-scan-stroke-planning-design.md, section 1.4):
    standoff, depth of field and the maximum scan area come from the TrackScan-Sharp 49
    datasheet; the range and the tracker half-angle are derived from it; the incidence limit
    and the body envelope are still unmeasured.
    """

    standoff_mm: float
    depth_of_field_mm: float
    fov_anchor_distance_mm: float
    range_mm: tuple[float, float]
    fov_at_standoff_mm: tuple[float, float]
    fov_cap_mm: tuple[float, float]
    max_incidence_deg: float
    body_sphere_r_mm: float
    body_clearance_mm: float
    distances_mm: tuple[float, ...]
    tilts_deg: tuple[float, ...]
    rolls_deg: tuple[float, ...]
    aim_grid_mm: float
    tilt_azimuths: int
    max_candidates: int
    refine_rounds: int
    refine_aim_grid_mm: float
    # The hard-feature pass. A wall, a bend and the rim beside a clamp are not scanned by
    # the head-on ladder at all: its footprint lies parallel to the wall and the incidence
    # limit bites within a few millimetres. What reaches them is a head tilted almost to
    # that limit from the free side, so this family is generated on its own, aimed only at
    # the leftovers, and never competes with the head-on ladder for area it already covers.
    feature_tilts_deg: tuple[float, ...]
    feature_tilt_azimuths: int
    feature_aim_grid_mm: float
    feature_rounds: int


@dataclass(frozen=True)
class TargetParams:
    """Sampling of the target face, the fixture-contact exemption, and the acceptance.

    `coverage_target_pct` is the acceptance criterion, and it is a site setting rather
    than a planner constant: what it buys is "how much area the workshop is willing to
    patch by hand". Left at None the selection runs until no candidate adds new coverage,
    which reproduces the validated v1 count.

    `delivery_mode` chooses between the two deliverables. `path` orders the chosen sweeps
    and inserts a checked connector between them; `lean` hands over the poses alone and
    makes no claim about the motion between them - see the config comment.

    `surface_bias` is the site's taste on a surface where a constant-pose sweep cannot stay
    valid for long: 0.0 keeps growing into the low-yield tail, 0.5 is the shipped behaviour,
    1.0 stops at the first sign of diminishing returns. It is one knob rather than six
    because the workshop can answer "sweep this part fully, or keep the program tidy" and
    cannot answer "what should min_gain_mm2_per_step be". The derived numbers live in
    `SURFACE_ANCHORS` and are written back into the report.

    `max_points` is the hard program-point ceiling. `None` skips the whole budget layer, so
    the selection path is byte-for-byte today's.
    """

    sample_grid_mm: float
    contact_mm: float
    contact_margin_mm: float
    face: str
    band_deg: float
    knn: int
    coverage_target_pct: float | None
    delivery_mode: str
    surface_bias: float
    max_points: int | None
    # The second acceptance figure, and it exists because the area target cannot express
    # it: a wall is a fraction of a percent of a 1 m2 panel, so a 97 % area plan can leave
    # every side wall unscanned and still meet its number. `feature_coverage_pct` is the
    # share of the hard features - walls, bends and the rim beside a clamp - that must be
    # covered for the run to count as done. null switches the criterion off.
    feature_coverage_pct: float | None
    # What counts as a hard feature: a surface whose normal leans further than this off the
    # face's dominant normal, or a sample closer than `feature_clearance_mm` to the fixture
    # or base (the region a clamp shades).
    feature_wall_deg: float
    feature_clearance_mm: float


@dataclass(frozen=True)
class TrackerParams:
    """The tracker's envelope. Its real pose is unknown until the site provides it."""

    half_angle_deg: float
    distance_mm: tuple[float, float]
    aim_at: str
    markers_omnidirectional: bool
    model_robot_occlusion: bool


@dataclass(frozen=True)
class StrokeParams:
    """A straight scan segment with a constant pose, and the steps the two passes use.

    `coverage_step_mm` and `check_step_mm` are deliberately different: a coverage
    evaluation costs one ray per sample, a legality check one KD-tree lookup per
    position, so only the cheap one may be fine.

    `enabled=False` reproduces the v1 result exactly, which is what makes every
    later change provable; `max_stroke_mm=0` is the same thing said another way.

    Three of these belong to the point-count work (design 2026-10-20, sections 3-M2,
    3-M3 and 3-M5):

    * `cost_model` is what the selection divides by. `millimetres` is the upstream
      design's "travel plus one stop", and it makes a stroke cost 8.9 times a pose, so
      the greedy pass takes poses first and strokes only for the tail. `corners` counts
      what the workshop counts: one stop for a pose, two for a stroke, plus the
      transition an extra `transition_corners_estimate` each.
    * `connector_level` is the most compressed connector the planner may use: 2 is the
      route through a retract AND a re-orientation point, 1 spreads the rotation over the
      flight and drops the second point, 0 also allows a straight move between two
      identical poses. 0 stays off until ROBOGUIDE has confirmed it.
    * `order_axis_weight_mm_per_deg` prices a pose change in the ordering. Distance alone
      ignores the axis, so two adjacent strokes 60 degrees apart in pose look free to
      travel between and cost a large re-orientation instead.

    The last three belong to the merged coverage/continuity design (2026-10-22):

    * `lock_roll` fixes the roll at seed time from the axis alone (`rolls.roll_for_axis`)
      instead of leaving every seed at 0 and letting the exporter derive a frame. With a
      600 x 500 footprint the roll decides which samples a constant-pose sweep can reach,
      so it has to be decided once, before the coverage is measured - not smoothed
      afterwards. `False` reproduces the pre-2026-10-22 sweep exactly.
    * `scan_gate` judges a growth step as a SCAN rather than as a move: at every check step
      there must still be a target sample inside the footprint, the range and the incidence
      limit. Without it a stroke keeps growing for up to one coverage step past the point
      where the head can see anything, and the report still shows the longer sweep as
      covered. `False` keeps the legacy behaviour (the gap is at most one coverage step,
      which is why the regression profile can afford to turn it off).
    * `hard_gate` makes an unconnectable item an ERROR rather than a note. With it on, a
      stroke whose connector cannot keep positive clearance is removed and the selection is
      re-run; with it off the path is delivered with a violation line in the report, which
      is what "legality is a hard constraint" cannot mean. Off only in the regression
      profile, which exists to reproduce that old behaviour.
    * `parallel_weight` is the tie-break multiplier for the long parallel bands that form
      the skeleton. It is 1.0 at the default bias and must never be applied to the repair
      pass: the last percent of coverage comes from bands in other directions.
    * `repair_budget_share` is the fraction of `target.max_points` the tail repair may
      spend. Only has an effect when a budget is set.
    * `pool_mode` selects the candidate pool. `full` is the aim-grid ladder plus the rail
      and side-scan families; `grid` is the historical aim-grid ladder alone, which is what
      the regression profile needs to reproduce; `rails` lays only the rail lines, the
      "ten minute draft" the design asks for - a smaller pool of the same algorithm, not a
      second pipeline.
    * `side_scan_tilt_deg` adds a family of seeds whose optical axis is tilted off the
      surface normal by this angle, for the walls and deep pockets a head-on sweep cannot
      reach. 0 disables the family.
    * `require_sweeps` keeps the SWEEP plan even when the pose-only reference scores better
      on corners and travel. A scanning cell is bought to sweep: the pose plan is the
      fallback for a pool whose sweeps cannot meet the target, not an equally good
      deliverable that happens to stop the arm thirty times. On the 2026-10-28 station the
      two tie on coverage (99.02% vs 99.04%) and the pose plan wins on corners, so without
      this flag "stroke mode" silently delivers poses.
    """

    enabled: bool
    max_stroke_mm: float
    min_stroke_mm: float
    extend_step_mm: float
    min_gain_mm2_per_step: float
    seed_dirs: int
    dedupe_mm: float
    stop_cost_mm: float
    coverage_step_mm: float
    check_step_mm: float
    retract_clearance_mm: float
    max_rot_deg_per_mm: float
    max_prune_rounds: int
    cost_model: str
    transition_corners_estimate: float
    connector_level: int
    order_axis_weight_mm_per_deg: float
    lock_roll: bool
    scan_gate: bool
    hard_gate: bool
    parallel_weight: float
    repair_budget_share: float
    pool_mode: str
    side_scan_tilt_deg: float
    require_sweeps: bool
    # J6 is the axis that turns the head about its own optical axis. It is constant inside
    # a sweep by construction (one pose), and between sweeps the branch search is what
    # decides how much of the turn to spend there. `j6_step_weight` is what one degree of
    # J6 costs next to a degree of the worst other axis, and `max_j6_step_deg` is the turn
    # a repositioning move should stay under - a review trigger like `robot.max_step_deg`,
    # weighted hard in the search (so it is avoided whenever a branch exists) and reported
    # when it cannot be.
    j6_step_weight: float
    max_j6_step_deg: float


@dataclass(frozen=True)
class MotionParams:
    """How the delivered program moves between the planned points.

    Speed is a floor setting, not a planning variable; it is carried here only so the
    program has one source for it instead of a flag and a config that can disagree.

    `joint_speed` is the speed of the moves that do NOT scan. In a stroke plan those are
    joint (`J`) moves, and they are the ones that turn the wrist between sweeps - a 90 deg
    J6 turn at 50 % is a violent move that no scan needs, so it has its own setting. A
    percentage is required: FANUC refuses a mm/sec rate on a joint instruction.
    """

    type: str
    termination: str
    speed_mm_s: float
    joint_speed: str = "5%"


def usable_limits(mechanical_deg, soft_deg, usage_frac: float = 1.0):
    """The envelope the planner may actually command, per axis.

    Two steps, in this order, and the order matters: shrink the DATASHEET envelope to its
    central `usage_frac`, then intersect with what the controller enforces. Shrinking after
    the intersection double-counts - the controller's own soft limit is already a
    configured safety, and taking another 5 % off it is how J2 ended up commanding -83 deg
    on a machine whose controller stops at -60 (MOTN-017, G:1 A:2).

    `usage_frac` is 1.0 in the shipped configuration: the site's reserve is a flat
    `joint_margin_deg` (5-10 deg) at each end, not a share of the range, because a share is
    a different number of degrees on every axis (18.5 on J1, 6.8 on J2, 36 on J4). The
    fraction is still supported - measured about the CENTRE of the datasheet range, so an
    asymmetric axis stays honest - for a site that wants a proportional band.

    The result is a hard envelope, not a report: every pose the gate accepts is inside it,
    and `Robot.feasible` then requires `joint_margin_deg` of room at each end of it.
    """
    fraction = float(usage_frac)
    usable = []
    for (low, high), (soft_low, soft_high) in zip(mechanical_deg, soft_deg, strict=True):
        low, high = float(low), float(high)
        centre = 0.5 * (low + high)
        half = 0.5 * (high - low) * fraction
        usable.append((max(centre - half, float(soft_low)),
                       min(centre + half, float(soft_high))))
    return tuple(usable)


@dataclass(frozen=True)
class RobotParams:
    """The arm, its two limit sets, and the tool - everything the kinematics needs.

    Not standard DH and deliberately not written that way: on this controller J3 reports
    the absolute pitch of the forearm, so `j3_zero_deg`/`j3_sign` mean what the machine
    means, and the wrist is described by its measured axis order and fixed tilts rather
    than by a textbook ball-wrist assumption. The numbers below were solved from eight
    measured poses (position residual <= 0.015 mm, attitude <= 0.001 deg, leave-one-out
    at the same level) and then checked against four drawing dimensions, so they are
    measurements, not datasheet guesses. The arm numbers came back unchanged when the
    pose bank was replaced - only the tool did, which is what a re-taught UTOOL looks
    like. `docs/robot-calibration.md` is the procedure.

    `mechanical_limits_deg` is the FANUC datasheet envelope for the R-2000iC/125L
    (floor standard version, reach 2655) supplied for this cell: J1 +-185, J2 -90..+46,
    J3 -122..+190, J4 +-360, J5 +-125, J6 +-360. `model` is a report label only - the
    kinematics comes from the fitted geometry above, so renaming the arm moves no number.
    `soft_limits_deg` is what the controller actually enforces; it is kept separate so a
    narrower pendant setting can be written in without touching the datasheet, and the
    planner intersects the two.

    `limits_usage_frac` is a share of the datasheet envelope (1.0 = the whole of it), and
    `joint_margin_deg` is the reserve kept at each end on top of that. The site asked for
    the second alone on 2026-10-28: a percentage of the range is a different reserve on
    every axis (18.5 deg on J1, 6.8 on J2, 36 on J4) and nothing about the machine justifies
    that; 5-10 deg is a number a workshop can reason about. The gate judges against the
    limits MINUS that margin, so "5 deg reserved" is what the delivered path satisfies.

    `enabled=False` skips the whole gate, which is what keeps the pre-2026-10-22 result
    reproducible bit for bit.
    """

    enabled: bool
    model: str

    base_offset_mm: float
    j2_height_mm: float
    upper_arm_mm: float
    forearm_mm: float
    forearm_offset_mm: float

    j1_sign: int
    j1_zero_deg: float
    j2_sign: int
    j2_zero_deg: float
    j3_sign: int
    j3_zero_deg: float
    j4_sign: int
    j4_zero_deg: float
    j5_sign: int
    j5_zero_deg: float
    j6_sign: int
    j6_zero_deg: float
    wrist_tilt_deg: tuple[float, float]

    tool_xyz_mm: tuple[float, float, float]
    tool_ypr_deg: tuple[float, float, float]

    mechanical_limits_deg: tuple[tuple[float, float], ...]
    soft_limits_deg: tuple[tuple[float, float], ...]
    limits_usage_frac: float

    joint_margin_deg: float
    j5_min_abs_deg: float
    max_step_deg: float
    wrist_identified: bool
    # How finely an interpolated MOVE is sampled when it is checked against the envelope.
    # `max_step_deg` judges the corners; a move between two legal corners can still leave
    # the envelope in between (a wrist that unwinds, a reach that straightens), and the
    # controller interpolates the whole segment whatever the corners look like. Sampled at
    # both steps - the Cartesian length and the rotation angle - and the tighter of the two
    # counts, because an in-place 180 deg turn covers no millimetres at all.
    motion_check_step_mm: float = 50.0
    motion_check_step_deg: float = 10.0
    # How far the elbow may be from fully straight or fully folded. `bend -> 0` (and
    # `bend -> 180`) is where the two links are collinear: the closed form still returns a
    # number, but it is the posture our fitted geometry describes worst, and the controller
    # refuses it with MOTN-018 rather than solving it. The 2026-10-29 program carried a
    # point at `bend = 8.4 deg` and ROBOGUIDE stopped there.
    elbow_margin_deg: float = 10.0
    # The third letter of every CONFIG string: "auto" follows the measured rule (`T` for
    # J5 < 0, 8 pendant-labelled poses on 2026-10-23), "T"/"F" pins it. This cell needs "T":
    # its ROBOGUIDE refuses an `N U F, ...` block at import with ASBN-023 "Configuration
    # error" - the whole file, not the point. Pinning is safe because the two wrist branches
    # are the SAME TCP pose (J5 negated, J4/J6 turned 180), so the scanner sees no difference;
    # what it buys is a program that loads.
    config_wrist_letter: str = "auto"

    # The CONFIG families the cell is allowed to use, as letter triples (shoulder, elbow,
    # wrist). Empty means "any family the solver finds". This is not a preference, it is a
    # correctness boundary: on 2026-10-28 a delivered point whose only in-envelope solution
    # was `F D T` reached the controller as J2 = 143.06 deg and stopped the program with
    # MOTN-017 (G:1, A:2,3). The closed form's front/back family numbers J2 around -40 for
    # that posture while the controller reports ~143 for it, so the family cannot be
    # validated against the limits this config carries - and it is not needed either: every
    # point of the program that ran on the cell (COV90.LS) is `N U T`.
    allowed_configs: tuple[str, ...] = ()

    @classmethod
    def disabled(cls) -> RobotParams:
        """The measured machine with the gate off - the default when no `robot` block."""
        return cls(
            enabled=False, model="FANUC R-2000iC/125L",
            base_offset_mm=312.0, j2_height_mm=0.0, upper_arm_mm=1075.0,
            forearm_mm=1730.0, forearm_offset_mm=225.0,
            j1_sign=1, j1_zero_deg=0.0, j2_sign=1, j2_zero_deg=0.0,
            j3_sign=-1, j3_zero_deg=0.0, j4_sign=-1, j4_zero_deg=180.0,
            j5_sign=1, j5_zero_deg=0.0, j6_sign=1, j6_zero_deg=180.0,
            wrist_tilt_deg=(-90.0, -90.0),
            tool_xyz_mm=(-248.001, -26.674, 584.158),
            tool_ypr_deg=(179.279, -45.934, 172.758),
            mechanical_limits_deg=((-185.0, 185.0), (-90.0, 46.0), (-122.0, 190.0),
                                   (-360.0, 360.0), (-125.0, 125.0), (-360.0, 360.0)),
            soft_limits_deg=((-185.0, 185.0), (-60.0, 46.0), (-122.0, 190.0),
                             (-360.0, 360.0), (-125.0, 125.0), (-360.0, 360.0)),
            limits_usage_frac=1.0,
            joint_margin_deg=5.0, j5_min_abs_deg=8.0, max_step_deg=40.0,
            # A gate that is off applies no delivery filters, and this one is a filter
            # (`Robot.solutions` uses it to throw away collinear-elbow postures). The shipped
            # configuration sets 10 deg; leaving it at 0 here keeps `Robot` a pure geometry
            # model for callers that asked for no gating.
            elbow_margin_deg=0.0,
            wrist_identified=True,
            config_wrist_letter="auto",
            allowed_configs=(),
        )


# The single place the surface-bias table lives. Three anchors, linear in between; the
# 0.5 column is deliberately equal to the values the shipped configuration carried before
# 2026-10-22, so `surface_profile(0.5)` is the regression test for the whole table.
SURFACE_ANCHORS: tuple[tuple[float, dict[str, float]], ...] = (
    (0.0, {"min_gain_mm2_per_step": 50.0, "min_stroke_mm": 60.0, "seed_dirs": 4.0,
           "order_axis_weight_mm_per_deg": 1.0, "parallel_weight": 1.0,
           "repair_budget_share": 0.35}),
    (0.5, {"min_gain_mm2_per_step": 200.0, "min_stroke_mm": 150.0, "seed_dirs": 4.0,
           "order_axis_weight_mm_per_deg": 2.0, "parallel_weight": 1.0,
           "repair_budget_share": 0.20}),
    (1.0, {"min_gain_mm2_per_step": 600.0, "min_stroke_mm": 300.0, "seed_dirs": 2.0,
           "order_axis_weight_mm_per_deg": 3.5, "parallel_weight": 1.3,
           "repair_budget_share": 0.05}),
)

# Deliberately NOT in the table: `scanner.max_incidence_deg` and `scanner.range_mm`. Making
# the scanner see further is not a taste setting, and tuning it to buy coverage would make
# the report look better without the head being able to scan any of it.


def surface_profile(bias: float) -> dict[str, dict[str, float]]:
    """The `stroke` keys that follow from one `target.surface_bias`, as a config patch.

    Pure function of the bias by design: it is merged *under* the override file, so an
    explicitly written key always wins and nothing has to track which keys were typed by
    hand. `seed_dirs` is interpolated and rounded because a direction family cannot be
    fractional.
    """
    value = float(bias)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"'surface_bias' must be in [0, 1], not {bias!r}: an out-of-range "
                         "bias is refused rather than clamped, so a typo cannot silently "
                         "become 'maximum tidiness'")
    low, high = SURFACE_ANCHORS[0], SURFACE_ANCHORS[-1]
    for anchor, values in SURFACE_ANCHORS:
        if abs(value - anchor) < 1e-12:
            # Exactly on an anchor: return the table's own numbers rather than a
            # interpolation that lands one ulp away. The test that pins the three columns
            # compares against the table, so it must not have to guess at 0.049999...
            return {"stroke": {key: float(item) for key, item in values.items()}}
    for (start, first), (end, second) in zip(SURFACE_ANCHORS, SURFACE_ANCHORS[1:],
                                             strict=True):
        if start <= value <= end:
            low, high = (start, first), (end, second)
            break
    span = high[0] - low[0]
    share = 0.0 if span <= 0.0 else (value - low[0]) / span
    derived: dict[str, float] = {}
    for key, start_value in low[1].items():
        end_value = high[1][key]
        mixed = start_value + (end_value - start_value) * share
        derived[key] = float(round(mixed)) if key == "seed_dirs" else float(mixed)
    return {"stroke": derived}


@dataclass(frozen=True)
class PlannerConfig:
    scanner: ScannerParams
    target: TargetParams
    tracker: TrackerParams
    stroke: StrokeParams
    motion: MotionParams
    occlusion_tol_mm: float
    seed: int
    robot: RobotParams = field(default_factory=RobotParams.disabled)
    source: str = ""
    override: str | None = None
    # Where each surface-derived `stroke` value came from: `base`, `profile(bias=x)` or
    # `override`. Written into the report so "why is min_stroke_mm 300" is answerable.
    parameter_sources: dict[str, str] = field(default_factory=dict)

    @property
    def contact_distance_mm(self) -> float:
        """Distance below which a sample counts as clamped, margin included."""
        return self.target.contact_mm + self.target.contact_margin_mm


def default_config_path() -> Path:
    """The shipped configuration, resolved from this file rather than the CWD."""
    return Path(__file__).resolve().parents[2] / "configs" / DEFAULT_CONFIG_NAME


def load_config(path: str | Path | None = None) -> PlannerConfig:
    """Load the shipped configuration, optionally overridden by `path`.

    Three layers, lowest first: the shipped file, the patch derived from
    `target.surface_bias`, and the override file. Putting the derived patch *above* the
    shipped file is the point - the shipped file writes all of these keys, so a profile
    merged underneath it could never take effect. Putting it *below* the override file is
    the other half: an explicitly written key always wins, and nothing has to remember
    which keys were typed by hand.

    The bias itself is read from the override file first, so `{"target": {"surface_bias":
    1.0}}` alone is enough to ask for the tidy profile.
    """
    base_path = default_config_path()
    base = _read_json(base_path)
    override_data: dict[str, Any] = {} if path is None else _read_json(Path(path))
    bias = _surface_bias(base, override_data)
    profile = surface_profile(bias)
    data = _deep_merge(_deep_merge(base, profile), override_data)
    return _build(data, str(base_path), None if path is None else str(path),
                  parameter_sources=_profile_sources(base, override_data, bias, profile))


def _surface_bias(base: dict[str, Any], override_data: dict[str, Any]) -> float:
    """The bias in force: the override's value if written, else the shipped one, else 0.5."""
    for source in (override_data, base):
        target = source.get("target")
        if isinstance(target, dict) and "surface_bias" in target:
            return float(target["surface_bias"])
    return 0.5


def _profile_sources(base: dict[str, Any], override_data: dict[str, Any], bias: float,
                     profile: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Attribute every derived `stroke` key: `override`, `base`, or `profile(bias=x)`."""
    written = override_data.get("stroke")
    written = written if isinstance(written, dict) else {}
    shipped = base.get("stroke")
    shipped = shipped if isinstance(shipped, dict) else {}
    sources: dict[str, str] = {}
    for section, values in profile.items():
        for key in values:
            if key in written:
                sources[f"{section}.{key}"] = "override"
            elif abs(bias - 0.5) < 1e-12 and key in shipped:
                sources[f"{section}.{key}"] = "base"
            else:
                sources[f"{section}.{key}"] = f"profile(bias={bias:g})"
    return sources


def override(config: PlannerConfig, **sections) -> PlannerConfig:
    """A copy of `config` with some section fields replaced, then re-validated.

    The command line is a second writer of the same knobs as the JSON file, so it has to go
    through the same validation; otherwise the two can disagree about what is legal and the
    illegal value wins only when it arrives by flag.
    """
    replacement = config
    for name, changes in sections.items():
        section = getattr(config, name)
        replacement = replace(replacement, **{name: replace(section, **changes)})
    _validate(replacement)
    return replacement


def _read_json(path: Path) -> dict[str, Any]:
    return read_json(path)


def read_json(path: str | Path) -> Any:
    """Read a JSON file, tolerating the UTF-8 BOM that Windows tools write.

    PowerShell's `Set-Content -Encoding utf8` and several Windows editors emit a BOM, and
    a hand-written config or station file that refuses to load because of one is a trap
    rather than strictness.
    """
    with open(path, encoding="utf-8-sig") as handle:
        return json.load(handle)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _build(data: dict[str, Any], source: str, override: str | None,
           parameter_sources: dict[str, str] | None = None) -> PlannerConfig:
    _reject_unknown(data)
    scanner = _section(data, "scanner", ScannerParams)
    target = _section(data, "target", TargetParams)
    tracker = _section(data, "tracker", TrackerParams)
    stroke = _section(data, "stroke", StrokeParams)
    motion = _section(data, "motion", MotionParams)
    robot = (_section(data, "robot", RobotParams) if "robot" in data
             else RobotParams.disabled())
    config = PlannerConfig(
        scanner=scanner,
        target=target,
        tracker=tracker,
        stroke=stroke,
        motion=motion,
        occlusion_tol_mm=float(data["occlusion_tol_mm"]),
        seed=int(data["seed"]),
        robot=robot,
        source=source,
        override=override,
        parameter_sources=dict(parameter_sources or {}),
    )
    _validate(config)
    return config


def _reject_unknown(data: dict[str, Any]) -> None:
    known = {"scanner", "target", "tracker", "stroke", "motion", "robot",
             "occlusion_tol_mm", "seed"}
    unknown = sorted(k for k in data if k not in known and not k.startswith("_"))
    if unknown:
        raise ValueError(f"unknown config section(s): {unknown}")

    for section, cls in (
        ("scanner", ScannerParams),
        ("target", TargetParams),
        ("tracker", TrackerParams),
        ("stroke", StrokeParams),
        ("motion", MotionParams),
        ("robot", RobotParams),
    ):
        body = data.get(section, {})
        allowed = {f.name for f in fields(cls)}
        bad = sorted(k for k in body if k not in allowed and not k.startswith("_"))
        if bad:
            raise ValueError(f"unknown key(s) in section '{section}': {bad}")


def _section(data: dict[str, Any], name: str, cls: type) -> Any:
    body = {k: v for k, v in data.get(name, {}).items() if not k.startswith("_")}
    hints = get_type_hints(cls)
    missing = sorted(f.name for f in fields(cls) if f.name not in body)
    if missing:
        raise ValueError(f"section '{name}' is missing {missing}")
    converted = {key: _coerce(hints[key], value) for key, value in body.items()}
    return cls(**converted)


def _coerce(annotation: Any, value: Any) -> Any:
    """Turn JSON lists into the tuples the frozen dataclasses declare, at any depth.

    The recursion is not cosmetic: `robot.mechanical_limits_deg` is a list of six pairs,
    and a one-level conversion would leave each pair as a JSON list. That reads fine in
    `limits[i][0]` but silently breaks anything that relies on the declared type - a
    frozen dataclass that quietly holds mutable lists is a trap left for the next edit.
    """
    if get_origin(annotation) is tuple:
        members = get_args(annotation)
        if not members:
            return tuple(value)
        if members[-1] is Ellipsis:
            return tuple(_coerce(members[0], item) for item in value)
        if len(value) != len(members):
            raise ValueError(f"expected {len(members)} values, got {value!r}")
        return tuple(_coerce(member, item) for member, item in zip(members, value, strict=True))
    return _scalar(annotation, value)


def _scalar(annotation: Any, value: Any) -> Any:
    optional = _optional_member(annotation)
    if value is None:
        if optional is None and annotation is not type(None):
            raise ValueError("this value cannot be null")
        return None
    if optional is not None:
        return _scalar(optional, value)
    if isinstance(annotation, type):
        try:
            return annotation(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"cannot read {value!r} as {annotation.__name__}") from error
    return value


def _optional_member(annotation: Any):
    """The single non-null member of `X | None`, or None when this is not such a union.

    Needed because `null` is a meaningful setting for the acceptance threshold: it is the
    switch back to "run until the coverage stops growing".
    """
    if isinstance(annotation, types.UnionType) or get_origin(annotation) is Union:
        members = [member for member in get_args(annotation) if member is not type(None)]
        if len(members) == 1:
            return members[0]
    return None


MOTION_TYPES = ("J", "L")
TERMINATIONS = ("FINE", "CNT0", "CNT100")
POOL_MODES = ("full", "grid", "rails")
# The third letter of a FANUC CONFIG string: "auto" derives it from the sign of J5 (the
# measured rule), "T"/"F" pins it for a controller that refuses one of them.
WRIST_LETTERS = ("AUTO", "T", "F")
# The three letters of a CONFIG family: shoulder side, elbow side, wrist flip.
CONFIG_SHOULDERS = ("F", "N")
CONFIG_ELBOWS = ("U", "D")
CONFIG_WRISTS = ("T", "F")


def _validate(config: PlannerConfig) -> None:
    scanner = config.scanner
    target = config.target
    tracker = config.tracker

    for name, limits in (
        ("range_mm", scanner.range_mm),
        ("distance_mm", tracker.distance_mm),
    ):
        if limits[0] > limits[1]:
            raise ValueError(f"'{name}' has its limits the wrong way round: {limits}")
    if not 0.0 < scanner.max_incidence_deg <= 90.0:
        raise ValueError(
            "'max_incidence_deg' must be in (0, 90] - it is an angle to the surface normal"
        )
    if scanner.max_candidates <= 0:
        raise ValueError("'max_candidates' must be positive")
    if scanner.tilt_azimuths < 1:
        raise ValueError("'tilt_azimuths' must be at least 1: it is how many directions "
                         "around the surface normal the tilt fan sweeps, so 0 would leave "
                         "the ladder unable to tilt at all")
    if scanner.refine_rounds < 0:
        raise ValueError("'refine_rounds' cannot be negative")
    if scanner.feature_rounds < 0:
        raise ValueError("'feature_rounds' cannot be negative")
    if scanner.feature_tilt_azimuths < 1:
        raise ValueError("'feature_tilt_azimuths' must be at least 1")
    if scanner.feature_aim_grid_mm <= 0.0:
        raise ValueError("'feature_aim_grid_mm' must be positive")
    for tilt in scanner.feature_tilts_deg:
        if not 0.0 <= tilt < scanner.max_incidence_deg:
            raise ValueError(
                f"'feature_tilts_deg' entry {tilt:g} is outside [0, "
                f"{scanner.max_incidence_deg:g}): a tilt at or past the incidence limit "
                "cannot see the sample it is aimed at, so it would only cost rays")
    if scanner.standoff_mm <= 0.0 or target.sample_grid_mm <= 0.0:
        raise ValueError("'standoff_mm' and 'sample_grid_mm' must be positive")
    if scanner.fov_anchor_distance_mm <= 0.0:
        raise ValueError("'fov_anchor_distance_mm' must be positive")
    if scanner.depth_of_field_mm <= 0.0:
        raise ValueError("'depth_of_field_mm' must be positive")
    if target.face not in FACES:
        raise ValueError(f"'face' must be one of {FACES}, not {target.face!r}")
    if target.coverage_target_pct is not None and not 0.0 < target.coverage_target_pct <= 100.0:
        raise ValueError("'coverage_target_pct' must be in (0, 100], or null to run until "
                         "the coverage stops growing - a value outside that range is "
                         "refused rather than silently clamped")
    if target.delivery_mode not in DELIVERY_MODES:
        raise ValueError(f"'delivery_mode' must be one of {DELIVERY_MODES}, "
                         f"not {target.delivery_mode!r}")
    if config.tracker.model_robot_occlusion:
        # There is no robot model at this stage, so this flag cannot be honoured.
        raise ValueError("'model_robot_occlusion' is not implemented; leave it false")
    _validate_strokes(config.stroke)
    _validate_surface(target)
    _validate_robot(config.robot)
    if config.motion.type not in MOTION_TYPES:
        raise ValueError(f"'motion.type' must be one of {MOTION_TYPES}, "
                         f"not {config.motion.type!r}")
    if config.motion.termination not in TERMINATIONS:
        raise ValueError(f"'motion.termination' must be one of {TERMINATIONS}, "
                         f"not {config.motion.termination!r}")
    if config.motion.speed_mm_s <= 0.0:
        raise ValueError("'motion.speed_mm_s' must be positive")
    joint_speed = str(config.motion.joint_speed).strip()
    if not re.fullmatch(r"\d+(?:\.\d+)?%", joint_speed) \
            or not 0.0 < float(joint_speed[:-1]) <= 100.0:
        raise ValueError(f"'motion.joint_speed' must be a percentage in (0, 100] such as "
                         f"'5%', not {config.motion.joint_speed!r}: a joint instruction "
                         "cannot take a mm/sec rate")


def _validate_strokes(stroke: StrokeParams) -> None:
    if stroke.max_stroke_mm < 0.0:
        raise ValueError("'max_stroke_mm' cannot be negative")
    if stroke.min_stroke_mm < 0.0:
        raise ValueError("'min_stroke_mm' cannot be negative")
    if stroke.min_stroke_mm > stroke.max_stroke_mm and stroke.max_stroke_mm > 0.0:
        raise ValueError("'min_stroke_mm' cannot exceed 'max_stroke_mm'")
    for name in ("coverage_step_mm", "check_step_mm", "extend_step_mm"):
        if getattr(stroke, name) <= 0.0:
            raise ValueError(f"'{name}' must be positive")
    if stroke.check_step_mm > stroke.coverage_step_mm:
        raise ValueError("'check_step_mm' must not exceed 'coverage_step_mm'")
    if not 0 <= stroke.seed_dirs <= 4:
        raise ValueError("'seed_dirs' must be in [0, 4]")
    if stroke.max_prune_rounds < 0:
        raise ValueError("'max_prune_rounds' cannot be negative")
    if stroke.max_rot_deg_per_mm <= 0.0:
        raise ValueError("'max_rot_deg_per_mm' must be positive")
    if stroke.cost_model not in COST_MODELS:
        raise ValueError(f"'cost_model' must be one of {COST_MODELS}, "
                         f"not {stroke.cost_model!r}")
    if stroke.transition_corners_estimate < 0.0:
        raise ValueError("'transition_corners_estimate' cannot be negative")
    if not 0 <= stroke.connector_level <= 2:
        raise ValueError("'connector_level' must be in [0, 2]: 2 is the retract-and-"
                         "reorient route, 1 drops the reorient point, 0 also allows a "
                         "straight move between two identical poses")
    if stroke.order_axis_weight_mm_per_deg < 0.0:
        raise ValueError("'order_axis_weight_mm_per_deg' cannot be negative")
    if stroke.parallel_weight <= 0.0:
        raise ValueError("'parallel_weight' must be positive: it multiplies the gain of "
                         "the parallel bands, so zero would delete the skeleton")
    if not 0.0 <= stroke.repair_budget_share <= 1.0:
        raise ValueError("'repair_budget_share' must be in [0, 1]")
    if stroke.pool_mode not in POOL_MODES:
        raise ValueError(f"'pool_mode' must be one of {POOL_MODES}, "
                         f"not {stroke.pool_mode!r}")
    if not 0.0 <= stroke.side_scan_tilt_deg < 90.0:
        raise ValueError("'side_scan_tilt_deg' must be in [0, 90): it is the angle between "
                         "the optical axis and the surface normal")
    if stroke.j6_step_weight < 0.0:
        raise ValueError("'j6_step_weight' cannot be negative")
    if not 0.0 < stroke.max_j6_step_deg <= 360.0:
        raise ValueError("'max_j6_step_deg' must be in (0, 360]: it is the J6 turn a "
                         "repositioning move should stay under")


def _validate_surface(target: TargetParams) -> None:
    """The two site-facing knobs introduced with the merged coverage design."""
    if not 0.0 <= target.surface_bias <= 1.0:
        raise ValueError(f"'surface_bias' must be in [0, 1], not {target.surface_bias!r} - "
                         "out of range is refused rather than clamped, because a clamped "
                         "bias looks like a deliberate choice")
    if target.max_points is not None and target.max_points < 2:
        raise ValueError("'max_points' must be at least 2, or null to skip the budget "
                         "layer - one point cannot describe a path")
    if target.feature_coverage_pct is not None \
            and not 0.0 < target.feature_coverage_pct <= 100.0:
        raise ValueError("'feature_coverage_pct' must be in (0, 100], or null to drop the "
                         "hard-feature criterion")
    if not 0.0 < target.feature_wall_deg <= 90.0:
        raise ValueError("'feature_wall_deg' must be in (0, 90]: it is the angle a surface "
                         "may lean off the face's dominant normal and still count as flat")
    if target.feature_clearance_mm < 0.0:
        raise ValueError("'feature_clearance_mm' cannot be negative")


def _validate_robot(robot: RobotParams) -> None:
    if not robot.enabled:
        return
    for name in ("upper_arm_mm", "forearm_mm", "forearm_offset_mm"):
        if getattr(robot, name) <= 0.0:
            raise ValueError(f"'robot.{name}' must be positive")
    if robot.forearm_mm <= robot.forearm_offset_mm:
        raise ValueError("'robot.forearm_mm' must exceed 'forearm_offset_mm': the wrist "
                         "centre sits off the forearm axis, not behind it")
    for name in ("j1_sign", "j2_sign", "j3_sign", "j4_sign", "j5_sign", "j6_sign"):
        if getattr(robot, name) not in (-1, 1):
            raise ValueError(f"'robot.{name}' must be -1 or +1")
    for name in ("mechanical_limits_deg", "soft_limits_deg"):
        limits = getattr(robot, name)
        if len(limits) != 6:
            raise ValueError(f"'robot.{name}' needs six [low, high] pairs")
        for low, high in limits:
            if low >= high:
                raise ValueError(f"'robot.{name}' has an empty range [{low}, {high}]")
    if len(robot.wrist_tilt_deg) != 2:
        raise ValueError("'robot.wrist_tilt_deg' needs two angles")
    if len(robot.tool_xyz_mm) != 3 or len(robot.tool_ypr_deg) != 3:
        raise ValueError("'robot.tool_xyz_mm' and 'robot.tool_ypr_deg' need three values")
    if len(robot.mechanical_limits_deg) == 6 and len(robot.soft_limits_deg) == 6:
        for axis, ((low, high), (soft_low, soft_high)) in enumerate(
                zip(robot.mechanical_limits_deg, robot.soft_limits_deg, strict=True), start=1):
            if soft_low < low or soft_high > high:
                raise ValueError(
                    f"'robot.soft_limits_deg' axis J{axis} [{soft_low}, {soft_high}] is "
                    f"wider than the mechanical envelope [{low}, {high}]: the controller "
                    "cannot be more permissive than the arm")
    if robot.joint_margin_deg < 0.0:
        raise ValueError("'robot.joint_margin_deg' cannot be negative")
    if robot.motion_check_step_mm <= 0.0:
        raise ValueError("'robot.motion_check_step_mm' must be positive")
    if robot.motion_check_step_deg <= 0.0:
        raise ValueError("'robot.motion_check_step_deg' must be positive")
    if not 0.0 <= robot.elbow_margin_deg < 90.0:
        raise ValueError("'robot.elbow_margin_deg' must be in [0, 90): it is the room kept "
                         "at each end of the elbow's own range")
    if robot.max_step_deg <= 0.0:
        raise ValueError("'robot.max_step_deg' must be positive")
    if str(robot.config_wrist_letter).upper() not in WRIST_LETTERS:
        raise ValueError(f"'robot.config_wrist_letter' must be one of {WRIST_LETTERS}, "
                         f"not {robot.config_wrist_letter!r}")
    for family in robot.allowed_configs:
        letters = str(family).upper().split()
        if (len(letters) != 3 or letters[0] not in CONFIG_SHOULDERS
                or letters[1] not in CONFIG_ELBOWS or letters[2] not in CONFIG_WRISTS):
            raise ValueError(
                f"'robot.allowed_configs' entry {family!r} is not a CONFIG family: write "
                "three letters, e.g. 'N U T' (shoulder F/N, elbow U/D, wrist T/F)")
    if not 0.0 < robot.limits_usage_frac <= 1.0:
        raise ValueError("'robot.limits_usage_frac' must be in (0, 1]: it is the share of "
                         "each axis range, measured from the range centre, that the "
                         "planner may command")
    # The J5 band is judged inside the ENVELOPE THE GATE USES, not the datasheet: a share
    # of 0.9 already shaves 12.5 deg off each end of +-125, so validating against the raw
    # limits would accept a j5_min_abs_deg that the shrunk band cannot hold.
    j5_low, j5_high = usable_limits(robot.mechanical_limits_deg, robot.soft_limits_deg,
                                    robot.limits_usage_frac)[4]
    if robot.j5_min_abs_deg >= (j5_high - j5_low) / 2.0:
        raise ValueError("'robot.j5_min_abs_deg' leaves no usable J5 band inside the usable "
                         f"limits [{j5_low:g}, {j5_high:g}] at "
                         f"limits_usage_frac={robot.limits_usage_frac:g}")
    if robot.j5_min_abs_deg <= 0.0:
        raise ValueError("'robot.j5_min_abs_deg' must be positive: J5 = 0 is the wrist "
                         "singularity, not a safe centre")
