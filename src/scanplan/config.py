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
import types
from dataclasses import MISSING, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

DEFAULT_CONFIG_NAME = "scanner_trackscan49.json"
FACES = ("upper", "lower", "auto")
# Slot A (stage 2): how candidates are rolled. Declared here rather than in `policies` so
# the config validator can name the alternatives without importing the stage's code.
ROLL_MODES = ("locked", "discrete")
# Slot B (stage 3): how the head travels between two sweeps. Two levels only - a single
# joint move, then the lift ladder.
CONNECTOR_MODES = ("direct-then-lift", "straight-then-lateral")
# The route style is a NAME for a set of settings, not a setting of its own: it exists so a
# page control can switch the whole chain at once, and `route_preset` is the only thing that
# reads it. `direct` is the identity - it leaves every other key alone, which is what keeps
# the shipped behaviour exactly what it was before the style existed.
ROUTE_STYLES = ("direct", "tidy")
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
    max_candidates: int
    refine_rounds: int
    refine_aim_grid_mm: float
    # Slot A (stage 2): which rolls a candidate is generated at.  `locked` pins one
    # roll per optical axis and re-tries others only when the joint gate refuses the
    # pose; `discrete` uses every entry in `rolls_deg`.  Locked is the default because
    # the discrete convention measured four J6 steps of 74-95 degrees on the reference
    # station for no scan benefit - but it is a trade-off, not a law, which is why it
    # is a key rather than a constant.
    roll_mode: str = "locked"


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
    * `order_j6_weight_mm_per_deg` prices the WRIST TWIST in the same cost. It is the only
      lever that moves J6 on a station where every pose has one solution, because J6 is
      decided by which pair of strokes the ordering puts next to each other, not by the
      branch search - see `docs/validation.md` for how to measure it.

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
    * `joint_connector` makes the connector a single joint move from one sweep's end to the
      next sweep's start. It exists because the retract corners were the problem: in the
      99% run, 7 of 50 delivered points had no joint solution at all and every one of them
      was a retract or reorient corner, while none of the 27 scanning points did. A joint
      move interpolates all six axes linearly, so two legal endpoints cannot produce an
      illegal middle, and it needs no corner in between.
    * `joint_step_deg` is how finely that joint move is sampled when its swept TCP path is
      checked for clearance. Degrees rather than millimetres because the TCP of a
      wrist-heavy move can travel hundreds of millimetres per degree of J6.
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
    # Primary connector primitive: one joint move from a sweep's end to the next sweep's
    # start, no retract corner, no CONFIG, no middle to exceed. Off (or with the kinematic
    # gate off) falls back to the retract routes, which is also the pre-2026-10-29 behaviour.
    joint_connector: bool = True
    # Joint resolution used to sample the swept TCP path of a joint move for clearance.
    # In degrees, not millimetres, because a wrist-heavy move can swing hundreds of
    # millimetres of TCP per degree of J6.
    joint_step_deg: float = 5.0
    # Slot B (stage 3): which connector chain runs between two sweeps. `direct-then-lift`
    # is the shipped implementation - one joint move, then the lift ladder. The level of
    # that ladder is still `connector_level`; this key names the policy, not its depth.
    connector: str = "direct-then-lift"
    # Which named set of connector and cost settings is in force. A LABEL, expanded by
    # `route_preset`: `direct` changes nothing, `tidy` swaps in `straight-then-lateral` and
    # the two sequence prices that make the arm hold one CONFIG family. Written into the
    # report so "why is this program shaped differently" is answerable two weeks later.
    route_style: str = "direct"
    # What one SWEEP costs the selection, in corners, next to a single pose's 1.0. Only read
    # when `cost_model` is `corners`.
    #
    # This is the one price in the planner that is a preference rather than a count, and it
    # is the only lever that decides how much of the part is swept instead of stopped on.
    # The arithmetic it overrides: a stop covers one footprint, and a band pays a SECOND stop
    # for whatever the extra length buys - so at the true count of 2.0 a band shorter than
    # about twice a footprint loses to the two poses that would cover the same area. That is
    # correct accounting, and on a curved panel it means the tail arrives as stops rather
    # than as bands.
    #
    #   2.0  the count itself: every corner a band really needs. The pre-2026-11-01 value.
    #   1.5  a band is charged half a corner for its far end.
    #   1.0  a band is charged like a pose, i.e. the sweep is preferred wherever it fits.
    #
    # It never changes what is deliverable - feasibility is decided by the connector and the
    # gate - only which of two answers that cover the same area is chosen, so it is reported
    # beside the run it shaped.
    stroke_corner_cost: float = 2.0
    # How many azimuths the TILTED seed family is laid on, and how far apart those seeds are.
    #
    # The head-on ladder is laid on one aim grid and grown along two tangent lines. A wall
    # or a clamped rim is not merely under-served by it, it is unreachable by it: the
    # footprint lies parallel to the wall, so the wall's own surface is at 90 degrees of
    # incidence a few millimetres off the bend. What does reach it is the same ladder tilted
    # far off the normal from the FREE side - and which side is free is not known in advance,
    # so the tilt has to be swept round the normal.
    #
    #   side_scan_azimuths  1 = one tangent direction, both signs: the pre-2026-11-01 family,
    #                       and exactly the two axes this code used before the key existed.
    #                       6 = twelve axes, 30 degrees apart, sweeping the whole circle.
    #   side_scan_spacing_mm 0.0 = `aim_grid_mm * 2`, the pre-2026-11-01 rule. A wall is a few
    #                       millimetres wide and `aim_grid_mm` is 100, so 0.0 seeds the tilted
    #                       family once every 200 mm and misses most of it.
    #
    # Every extra azimuth is a grown stroke and a set of rays, which is why 1 and 0.0 ship.
    side_scan_azimuths: int = 1
    side_scan_spacing_mm: float = 0.0
    # Only seed the tilted family on samples whose normal is at least this many degrees off
    # the face's dominant normal: side walls, bends, and the rim a clamp shades. 0.0 seeds it
    # everywhere, which is the pre-2026-11-01 behaviour - and on a 1 m2 panel that is a few
    # thousand seeds, each grown into two bands, which turned a 4-minute run into something
    # that had not finished in twenty. A wall is a small fraction of the area and all of the
    # reason the family exists, so the filter is what makes the dense version affordable.
    side_scan_wall_deg: float = 0.0
    # A hard ceiling on how many SEEDS the tilted family may emit in total (aim points x
    # tangent directions x signs). 0 means no ceiling, which is the pre-2026-11-01 behaviour
    # and is what made the dense version unusable: on a 1 m2 panel the family grew from 52
    # seeds to a few thousand and a 4-minute run had not finished in twenty. The ceiling is
    # spent by thinning the aim points evenly, so a capped family still covers the whole
    # wall instead of the first corner of it.
    side_scan_max_seeds: int = 0
    # Which named set of candidate, selection and gate settings is in force. A LABEL, expanded
    # by `plan_preset` - see PLAN_PRESETS. `standard` is the shipped behaviour and expands to
    # nothing at all.
    plan_mode: str = "standard"
    # What one degree of J6 costs the ordering, in millimetres of travel, on top of the pose
    # change `order_axis_weight_mm_per_deg` prices. A separate term because it is a separate
    # cost: two poses a few degrees apart can still need 180 deg of J6, and J6 is the axis
    # that slides the scanner sideways along the surface - the one direction the 45 mm body
    # sphere cannot see. Charging only the pose change leaves that out, and the ordering then
    # zig-zags across the part and spends the wrist on it.
    #
    # Measured on the 2026-11-01 station (30 delivered poses, uniform 180-flip parity): the
    # run's own order cost 1056 deg of J6 with a worst step of 175; travel only 1353; this
    # term at 10 gives 544 with a worst step of 63 for 8 % more travel, and at 20 gives 409
    # for 16 % more. The knee is 10-20 because below it the wrist still pays for ordering
    # errors and above it the search is choosing J6 over the part it has to reach.
    #
    # It is the ONLY lever that moves J6 on a station where every pose has one joint
    # solution: the branch search then has nothing to choose, and `robot.j6_weight` is inert.
    # 0.0 is "not priced", i.e. the behaviour before 2026-11-01.
    order_j6_weight_mm_per_deg: float = 0.0


@dataclass(frozen=True)
class MotionParams:
    """How the delivered program moves between the planned points.

    Speed is a floor setting, not a planning variable; it is carried here only so the
    program has one source for it instead of a flag and a config that can disagree.

    `joint_speed_ladder` is the same idea for the JOINT moves, and it is a ladder rather than
    one number because a joint move is the one place where the danger scales with the size of
    the move: a few degrees of J6 is nothing, a hundred and eighty swings the scanner through a
    400 mm arc about the flange. Each entry is `[biggest single-axis change in degrees, speed
    in percent]`, read top to bottom, and the last entry's degree value MUST be `null` - that
    is the catch-all bucket, so a move always has a speed. `[[null, 100.0]]` is the pre-2026-10-30
    behaviour (every joint move at the controller default) and is what a config without this key
    gets, so an old file exports the same program it always did.

    The scan speed is NOT in here on purpose: a linear sweep is flown at `speed_mm_s`, and the
    coverage figure was computed for that speed. The ladder only ever reaches `J` instructions.
    """

    type: str
    termination: str
    speed_mm_s: float
    joint_speed_ladder: tuple[tuple[float | None, float], ...] = ((None, 100.0),)

    def joint_speed_pct(self, step_deg: float | None) -> float:
        """The percentage to command for a joint move whose biggest axis change is `step_deg`.

        `None` means "the size is not known" - the first point of a program has no predecessor
        to measure against - and it takes the LAST bucket, which is the slowest one by
        construction. Guessing fast for an unmeasured move is the one direction that cannot be
        undone once the program is on the controller.
        """
        if step_deg is None:
            return float(self.joint_speed_ladder[-1][1])
        for limit, speed in self.joint_speed_ladder:
            if limit is None or float(step_deg) <= float(limit):
                return float(speed)
        return float(self.joint_speed_ladder[-1][1])


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

    `soft_limits_deg` is what the controller actually enforces, and it is NOT the datasheet
    envelope. Since 2026-10-29 the six rows are READ OFF the pendant
    (`MENU > SYSTEM > Axis Limit`), not inferred:

    | axis | lower   | upper  |
    |------|---------|--------|
    | J1   | -185.00 | 185.00 |
    | J2   |  -60.00 |  76.00 |
    | J3   | -120.71 | 180.00 |
    | J4   | -360.00 | 360.00 |
    | J5   | -125.00 | 125.00 |
    | J6   | -360.00 | 360.00 |

    Two of these ended a long argument. J2's ceiling is +76, not the +136 the datasheet
    suggested, and J3's floor is -120.71, not -185. The old -185 floor is what let the
    planner emit `F`-shoulder solutions with J3 near -175: those poses were written to a
    joint-representation program, the controller re-expressed J3 as -175 + 360 = 185, found
    that above its +180 ceiling as well, and alarmed `MOTN-017 (G:1, A:2,3)`. The whole `F`
    branch was unreachable and the model was calling it reachable.

    `joint_span_fraction` is the "do not ride the limits" policy: each axis is only used
    inside a fraction of its span, measured about the span's CENTRE. At 0.98 with the numbers
    above the usable envelope is J1 +-181.3, J2 [-58.64, 74.64], J3 [-117.70, 176.99],
    J4 +-352.8, J5 +-122.5, J6 +-352.8; `joint_margin_deg` is applied inside that again,
    leaving J1 +-176.3, J2 [-53.64, 69.64], J3 [-112.70, 171.99], J4 +-347.8, J5 +-117.5,
    J6 +-347.8. It exists at all because J5 = -122 alarmed in ROBOGUIDE although -125 is the
    documented limit, and a pose that sits three degrees from a limit is not one the machine
    will hold - that pose is still refused here, and the refusal comes from the 5 degree
    margin, not from the span. The tests are what keep this honest - see
    `test_kinematics.py`.

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

    joint_margin_deg: float
    j5_min_abs_deg: float
    max_step_deg: float
    wrist_identified: bool
    # Degrees of J6 change above which a segment is LISTED in the report. Reported, never
    # enforced: refusing such a move means dropping the sweep behind it, and the coverage
    # that sweep carried. The lever arm below is what makes the number mean something.
    j6_report_deg: float = 90.0
    joint_span_fraction: float = 0.98
    # Which wrist branch a delivered solution may use: `T` (J5 negative), `F` (J5 positive)
    # or `any`. `T` is the site default because the program writes Cartesian points with the
    # constant CONFIG `N U T` - see `Robot.solutions` for the two measurements behind it.
    wrist_flip_policy: str = "T"
    # How many times its own size a J6 change counts for when the planner chooses BETWEEN
    # otherwise comparable solutions. 1.0 is "count it like any other axis", i.e. the
    # behaviour before 2026-10-30. J6 turns the flange about the tool axis, and the scanner
    # sits 249 mm off that axis, so one degree of J6 slides it 4.35 mm sideways - along the
    # part rather than towards it, which is the one direction the 45 mm body sphere cannot
    # see anything in. This is a preference, never a limit: it cannot make an infeasible
    # solution feasible, and it never drops a sweep. 1.5 is the merged default: it still
    # cannot invent an alternative, but when one exists it now prefers the arm to turn.
    j6_weight: float = 1.5
    # A J6 step above this costs an extra penalty in the sequencing (`path._J6_OVER_CAP_COST`),
    # which is still a cost rather than a refusal. None turns the cap off.
    j6_max_step_deg: float | None = None
    # A SECOND charge on J6, added on top of the all-axis step. `j6_weight` above only bites
    # when J6 is the largest axis of the move, so a 20 deg twist next to a 100 deg arm swing
    # is free under it; this one pays per degree of J6 whatever the rest of the move does,
    # which is what "keep the total J6 travel small" needs. The two charge different things
    # - one the worst step, one the running sum - and a site may want either, both or
    # neither. 0.0 is the behaviour before 2026-11-01, so the default changes nothing.
    j6_travel_weight: float = 0.0
    # What a CONFIG family change costs the sequencing, WHEN one is allowed at all. As of
    # 2026-11-02 it is not: a straight Cartesian move cannot change family mid-line, the
    # connector proves those lines wherever it can, and the delivery audit refuses a program
    # whose recorded joints are on a different family than the line arrives at
    # (`branch_mismatch` - it killed a whole 99% run at export). Both sequencing passes now
    # treat the change as impossible and only fall back to this price when no single family
    # carries the chain, so a point whose only solution sits in another family keeps it. Zero
    # prices the fallback as free, which is the behaviour every program before 2026-11-01 was
    # written with; `route_style = "tidy"` expands it to 240.
    config_change_cost_deg: float = 0.0
    # REPORTED, not enforced: a delivered point closer than this to its usable band is named
    # in the report. 0 margin is the `banded` mode's own choice, and it is how a plan looks
    # healthy here and comes back `MOTN-018 position not reachable` from ROBOGUIDE: measured
    # 2026-11-02, one point sat 1.25 deg under the J2 ceiling and the controller wanted a
    # hair more than the model allowed.
    min_margin_warn_deg: float = 2.0
    # What a wrist flip (J5 changing sign) costs the sequencing. A joint move interpolates
    # J5 linearly, so a sign change sends the wrist through the singularity where the arm
    # whips round. 180 is what every item-level sequence used before it was configurable.
    wrist_flip_cost_deg: float = 180.0

    # --- the seventh axis: a linear rail under the arm ---------------------
    # The arm rides a track along +X of its own base frame. `t` is millimetres from the
    # rail's zero, and it is an axis like any other: it has a usable band, it is walked
    # along with the arm on a straight move, it is priced in the sequencing, and it is
    # written into the program as `E1`. Disabled by default so every stored run and every
    # test keeps the six-axis behaviour it was measured with; the shipped configuration
    # turns it on for this cell.
    track_enabled: bool = False
    # Only `+X` is implemented. The rail is a translation along one base-frame axis and the
    # pose table, the CONFIG letters and the frame contract all assume the arm's own axes;
    # a rail at an angle would need a mounting transform nothing here measures.
    track_axis: str = "+X"
    track_length_mm: float = 8000.0
    # The controller's travel, if it is shorter than the rail. None means "the whole rail".
    track_limits_mm: tuple[float, float] | None = None
    # Kept clear of the ends for the same reason `joint_margin_deg` exists: a carriage on
    # its end stop is not a carriage the machine holds.
    track_margin_mm: float = 100.0
    # Candidate offsets tried per pose, spaced this far apart around the geometric best.
    track_grid_mm: float = 250.0
    track_candidates: int = 5
    # The horizontal distance from the base to the wrist centre the arm likes to work at.
    # The rail is placed to bring the pose to roughly this radius, then a few neighbours are
    # tried - that is what keeps the rail from parking the arm at full stretch.
    track_nominal_reach_mm: float = 1900.0
    # What one millimetre of rail travel costs the sequencing, in degrees: the same scale as
    # a joint step, so "slide 2 m" has to earn its place against "turn the wrist 30 deg".
    track_cost_deg_per_mm: float = 0.05
    # REPORTED, not enforced - the same contract as `max_step_deg`.
    track_max_step_mm: float = 1500.0
    # Write the rail value into the `.LS` as `E1`. Kept switchable because where that column
    # sits depends on how the controller's group is configured, and the post-processor here
    # assumes the extended-axis form.
    track_emit_e1: bool = True
    # How the seventh axis is spelled in a JOINT-representation block. A joint block counts
    # axes by NUMBER inside the group, so the extended axis is `J7` there, while a Cartesian
    # block names it `E1`. `J7` is the default because a joint block carrying six `J` values
    # and a stray `E1` is refused at load with `ASBN-009 Mismatch number of joint` - measured
    # on a real ROBOGUIDE import 2026-11-02. Set this to `E1` if a controller insists on the
    # Cartesian spelling there as well.
    track_joint_label: str = "J7"
    # Whether a joint-representation point may carry the rail at all. `false` - the default
    # WITH a rail - writes those points as Cartesian blocks instead, which is the shape a
    # working export from the cell has. Measured 2026-11-02: a real ROBOGUIDE import refused
    # BOTH spellings inside a joint block (`J7=` and `E1=`) at the same column with `ASBN-009
    # Mismatch number of joint`, while the same value in a Cartesian block is accepted. The
    # cost is that those poses are solved by the controller again instead of being pinned.
    track_joint_blocks: bool = False

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
            mechanical_limits_deg=((-185.0, 185.0), (-136.0, 136.0), (-301.0, 301.0),
                                   (-720.0, 720.0), (-250.0, 250.0), (-720.0, 720.0)),
            soft_limits_deg=((-185.0, 185.0), (-60.0, 76.0), (-120.71, 180.0),
                             (-360.0, 360.0), (-125.0, 125.0), (-360.0, 360.0)),
            joint_margin_deg=5.0, j5_min_abs_deg=8.0, max_step_deg=40.0,
            wrist_identified=True, joint_span_fraction=0.98, wrist_flip_policy="T",
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


# The sampling grid a fast run is allowed to use, and the coarsest value it must not
# exceed. 10 mm is what the 2026-10-28 A/B measured 7.7x end to end on the real part
# (3 mm: 1724 s, 10 mm: 223 s, `face: auto` both); 8 mm is the floor, so a "fast" run
# that is still at 3 mm is refused instead of being trusted for being labelled fast.
FAST_SAMPLE_GRID_MM = 10.0
FAST_MIN_SAMPLE_GRID_MM = 8.0


@dataclass(frozen=True)
class FastParams:
    """The two things a fast run changes that no ordinary key can express.

    Everything else a fast run does - the sampling grid, which face - is written into
    the ordinary keys by :func:`fast_preset`, so the report shows the value actually in
    force instead of a flag nobody can audit. What is left here is the state itself and
    the one semantic switch: whether lines of sight are modelled at all.

    `layers` is the record of what the preset changed, carried so the report can say
    "this run was fast, and here is what that meant" without re-deriving it from the
    numbers - which would be circular, because the numbers are the thing not to trust.

    The default is off, and off must be bit-for-bit the pre-2026-10-28 behaviour.
    """

    enabled: bool
    no_occlusion: bool
    layers: tuple[str, ...] = ()

    @classmethod
    def disabled(cls) -> FastParams:
        return cls(enabled=False, no_occlusion=False, layers=())


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
    fast: FastParams = field(default_factory=FastParams.disabled)
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


def fast_preset(config: PlannerConfig, *, grid: float = FAST_SAMPLE_GRID_MM,
                no_occlusion: bool = False,
                single_face: bool = True) -> PlannerConfig:
    """The fast run: coarse sampling grid, one face, and the state that says so.

    `single_face=False` is for a caller that already plans one face on its own terms (the
    web UI always asks for the upper face), so the preset must not claim a speedup it did
    not cause - the report's `layers` says exactly what was changed.

    Deliberately NOT in this preset:

    * `stroke.check_step_mm` / `extend_step_mm`. They look like an easy 1.5-2.5x, but
      `check_step_mm` is also the sampling step `RobotArm.along` uses for a straight move
      in the joint screen, so coarsening it here would make the very check a fast run
      exists to feed coarser than the motion it is checking.
    * `stroke.enabled = False`. The J5 over-travel in `output/run_j5_real_upper.log`
      happens along the straight sweeps, so a path without sweeps cannot reproduce it.

    The grid and the face are written into `target`, not carried as flags, so the report
    shows what actually ran.
    """
    layers = ["sample_grid"]
    target: dict[str, object] = {"sample_grid_mm": float(grid)}
    if single_face:
        layers.append("single_face")
        target["face"] = "upper"
    if no_occlusion:
        layers.append("no_occlusion")
    return override(
        config,
        target=target,
        fast={"enabled": True, "no_occlusion": bool(no_occlusion),
              "layers": tuple(layers)},
    )


def _read_json(path: Path) -> dict[str, Any]:
    return read_json(path)


#: What each route style expands to. `direct` is spelled out rather than left empty so that
#: asking for it explicitly restores the shipped chain even if the file it is applied to
#: carries a `tidy` value; it deliberately does NOT name `robot.j6_weight`, which is the
#: site's own cost preference rather than part of the route's shape.
#:
#: `tidy` keeps `joint_connector` ON even though the chain it belongs to is the fallback:
#: the joint move is what stops "make it straight" from meaning "lose the sweep", and a style
#: whose fallback could not fly would trade coverage for appearance.
ROUTE_PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "direct": {
        "stroke": {"route_style": "direct", "connector": "direct-then-lift",
                   "joint_connector": True},
        "robot": {"config_change_cost_deg": 0.0, "wrist_flip_cost_deg": 180.0},
    },
    "tidy": {
        "stroke": {"route_style": "tidy", "connector": "straight-then-lateral",
                   "joint_connector": True, "order_j6_weight_mm_per_deg": 10.0},
        "robot": {"config_change_cost_deg": 240.0, "wrist_flip_cost_deg": 1000.0,
                  "j6_weight": 0.5},
    },
}


def _route_values(config: PlannerConfig, style: str) -> PlannerConfig:
    """`config` with one style's settings applied, unvalidated."""
    replacement = config
    for name, changes in ROUTE_PRESETS[style].items():
        section = getattr(replacement, name)
        replacement = replace(replacement, **{name: replace(section, **changes)})
    return replacement


#: Which plan mode the page and the command line offer.
PLAN_MODES: tuple[str, ...] = ("standard", "banded")

#: The joint envelope every run has to fit in, and a CEILING rather than a default: the usable
#: band of each axis is its controller span times `MAX_JOINT_SPAN_FRACTION`, taken about the
#: span's centre, minus `MIN_JOINT_MARGIN_DEG` at each end. Measured twice:
#:
#:  * J5 = -122 alarmed in ROBOGUIDE while -125 is the documented limit, which is where the
#:    margin came from;
#:  * a `banded` run at full span and zero margin delivered a point at J2 = 74.75 of 76 and the
#:    controller answered `MOTN-018 position not reachable` - our model said legal, theirs
#:    wanted a hair more room (2026-11-02).
#:
#: So a configuration may be STRICTER than this and never looser; `_validate_robot` refuses the
#: rest, and no plan mode may widen it. Widening it is not a knob, it is how a program that
#: plans cleanly here fails on the cell.
MAX_JOINT_SPAN_FRACTION = 0.98
MIN_JOINT_MARGIN_DEG = 5.0

#: What each plan mode expands to. `standard` is spelled out as the values the shipped
#: configuration already carries, so asking for it is the identity on any file - that is what
#: makes "no mode chosen" reproduce the old runs exactly.
#:
#: `banded` used to move THREE things at once, and the third was the joint envelope: it stopped
#: shrinking each axis to 98% and reserving 5 more degrees, because a tilted band leaves a
#: shrunken envelope halfway along and `gate_ok` drops the WHOLE band (measured 2026-11-01: 43
#: grown candidates with the shrink, 216 without). That widening is GONE by policy (2026-11-02):
#: a banded run at 0 % margin put one delivered point 1.25 deg under the J2 ceiling and
#: ROBOGUIDE answered `MOTN-018 position not reachable`. The envelope is now a ceiling no mode
#: may raise - see `MAX_JOINT_SPAN_FRACTION` / `MIN_JOINT_MARGIN_DEG` - so `banded` keeps its
#: dense tilted seed family and its corner price and nothing else.
PLAN_PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "standard": {
        "stroke": {"plan_mode": "standard", "side_scan_azimuths": 1,
                   "side_scan_spacing_mm": 0.0, "side_scan_wall_deg": 0.0,
                   "side_scan_max_seeds": 0, "stroke_corner_cost": 2.0},
        "robot": {"joint_span_fraction": MAX_JOINT_SPAN_FRACTION,
                  "joint_margin_deg": MIN_JOINT_MARGIN_DEG},
    },
    "banded": {
        "stroke": {"plan_mode": "banded", "side_scan_azimuths": 3,
                   "side_scan_spacing_mm": 60.0, "side_scan_wall_deg": 45.0,
                   "side_scan_max_seeds": 96, "stroke_corner_cost": 1.0},
        # Same envelope as `standard`, spelled out rather than left empty so the mode still
        # describes the whole state it asks for - and so a later mode cannot drift from it.
        "robot": {"joint_span_fraction": MAX_JOINT_SPAN_FRACTION,
                  "joint_margin_deg": MIN_JOINT_MARGIN_DEG},
    },
}


def _preset_values(config: PlannerConfig, table: dict[str, dict[str, Any]]) -> PlannerConfig:
    """`config` with one preset's settings applied, unvalidated."""
    replacement = config
    for name, changes in table.items():
        section = getattr(replacement, name)
        replacement = replace(replacement, **{name: replace(section, **changes)})
    return replacement


def plan_preset(config: PlannerConfig, mode: str | None = None) -> PlannerConfig:
    """Expand `stroke.plan_mode` into the candidate, gate and price settings it names.

    Same shape as `route_preset`, and for the same reason: one expansion shared by the page
    and the command line cannot drift. `mode=None` applies the label the configuration
    already carries.

    It is deliberately a SEPARATE axis from `route_style`. A plan mode decides what gets
    selected (which candidate families exist, how big the envelope is, what a band costs);
    a route style decides how the selected items are connected. Both can be asked for at
    once, and neither implies the other.
    """
    chosen = str(config.stroke.plan_mode if mode is None else mode)
    chosen = chosen.strip().lower()
    if chosen not in PLAN_MODES:
        raise ValueError(f"'plan_mode' must be one of {PLAN_MODES}, not {mode!r}")
    replacement = _preset_values(config, PLAN_PRESETS[chosen])
    _validate(replacement)
    return replacement


def route_preset(config: PlannerConfig, style: str | None = None) -> PlannerConfig:
    """Expand `stroke.route_style` into the settings it names.

    One expansion for the page and the command line, so the two cannot drift apart - the
    same reason `fast_preset` exists. `style=None` means "the style this configuration
    already carries", which is what a caller that only wants the file's own label applied
    passes.

    `direct` restores the shipped connector and the two sequence prices to the values every
    program before this key existed was written with, and leaves everything else -
    `j6_weight` above all - exactly as the file set it. On the shipped configuration it is
    therefore the identity, which is what makes "no style chosen" reproduce the old runs
    byte for byte.

    The style owns the keys it names. A file that writes `route_style: tidy` AND
    `j6_weight: 1.0` gets 0.5 at the top level, because the preset is the single writer of
    that set; the report then shows the value that actually ran.
    """
    chosen = str(config.stroke.route_style if style is None else style)
    chosen = chosen.strip().lower()
    if chosen not in ROUTE_STYLES:
        raise ValueError(f"'route_style' must be one of {ROUTE_STYLES}, not {style!r}")
    replacement = _route_values(config, chosen)
    _validate(replacement)
    return replacement


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
    fast = (_section(data, "fast", FastParams) if "fast" in data
            else FastParams.disabled())
    config = PlannerConfig(
        scanner=scanner,
        target=target,
        tracker=tracker,
        stroke=stroke,
        motion=motion,
        occlusion_tol_mm=float(data["occlusion_tol_mm"]),
        seed=int(data["seed"]),
        robot=robot,
        fast=fast,
        source=source,
        override=override,
        parameter_sources=dict(parameter_sources or {}),
    )
    # A file may carry `route_style: tidy` and nothing else; the label is expanded here, so
    # no consumer can read a style that the rest of the configuration does not implement.
    if config.stroke.route_style not in ROUTE_STYLES:
        raise ValueError(f"'stroke.route_style' must be one of {ROUTE_STYLES}, not "
                         f"{config.stroke.route_style!r}")
    if config.stroke.route_style != "direct":
        config = _route_values(config, config.stroke.route_style)
    if config.stroke.plan_mode not in PLAN_MODES:
        raise ValueError(f"'stroke.plan_mode' must be one of {PLAN_MODES}, not "
                         f"{config.stroke.plan_mode!r}")
    if config.stroke.plan_mode != "standard":
        config = _preset_values(config, PLAN_PRESETS[config.stroke.plan_mode])
    _validate(config)
    return config


def _reject_unknown(data: dict[str, Any]) -> None:
    known = {"scanner", "target", "tracker", "stroke", "motion", "robot", "fast",
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
        ("fast", FastParams),
    ):
        body = data.get(section, {})
        allowed = {f.name for f in fields(cls)}
        bad = sorted(k for k in body if k not in allowed and not k.startswith("_"))
        if bad:
            raise ValueError(f"unknown key(s) in section '{section}': {bad}")


def _section(data: dict[str, Any], name: str, cls: type) -> Any:
    body = {k: v for k, v in data.get(name, {}).items() if not k.startswith("_")}
    hints = get_type_hints(cls)
    # A field with a default is optional in the file. That is what lets a new key land
    # without turning every existing config into an error, which matters here because the
    # default is how "off" is spelled: `motion.joint_speed_ladder` absent means "one speed
    # for every joint move", i.e. the program this repository exported before 2026-10-30.
    missing = sorted(f.name for f in fields(cls)
                     if f.name not in body
                     and f.default is MISSING and f.default_factory is MISSING)
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
WRIST_FLIP_POLICIES = ("T", "F", "any")
TRACK_AXES = ("+X", "-X")
#: How a joint-representation block names the seventh axis. `J7` is the axis number in the
#: group; `E1` is the Cartesian spelling, kept as an escape hatch.
TRACK_JOINT_LABELS = ("J7", "E1")


def _track_limits(robot) -> tuple[float, float]:
    """The rail's controller travel: what the file says, else the whole rail."""
    limits = getattr(robot, "track_limits_mm", None)
    if limits is None:
        return (0.0, float(getattr(robot, "track_length_mm", 0.0)))
    low, high = (float(value) for value in limits)
    return (low, high)


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
    if scanner.roll_mode not in ROLL_MODES:
        raise ValueError(f"'scanner.roll_mode' must be one of {ROLL_MODES}, not "
                         f"{scanner.roll_mode!r}")
    if scanner.max_candidates <= 0:
        raise ValueError("'max_candidates' must be positive")
    if scanner.refine_rounds < 0:
        raise ValueError("'refine_rounds' cannot be negative")
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
    _validate_fast(config)
    if config.motion.type not in MOTION_TYPES:
        raise ValueError(f"'motion.type' must be one of {MOTION_TYPES}, "
                         f"not {config.motion.type!r}")
    if config.motion.termination not in TERMINATIONS:
        raise ValueError(f"'motion.termination' must be one of {TERMINATIONS}, "
                         f"not {config.motion.termination!r}")
    if config.motion.speed_mm_s <= 0.0:
        raise ValueError("'motion.speed_mm_s' must be positive")
    _validate_joint_speed_ladder(config.motion.joint_speed_ladder)


def _validate_joint_speed_ladder(ladder) -> None:
    """The buckets must ascend and the last one must be the catch-all.

    Refused rather than repaired: a ladder with two equal thresholds has an unreachable
    bucket, and one that does not end in `null` leaves large moves with no speed at all -
    both are typos that would otherwise show up as a program with the wrong speed on the
    floor, which is exactly the kind of thing nobody notices until an alarm.
    """
    if not ladder:
        raise ValueError("'motion.joint_speed_ladder' cannot be empty: the largest joint "
                         "moves would have no speed")
    last = len(ladder) - 1
    previous = None
    for index, entry in enumerate(ladder):
        if len(entry) != 2:
            raise ValueError(f"'motion.joint_speed_ladder' entry {entry!r} must be "
                             "[degrees, percent]")
        limit, speed = entry
        if not 0.0 < float(speed) <= 100.0:
            raise ValueError(f"'motion.joint_speed_ladder' speed {speed!r} must be in "
                             "(0, 100] percent")
        if index == last:
            if limit is not None:
                raise ValueError(
                    "'motion.joint_speed_ladder' must end with a null limit: it is the "
                    "catch-all bucket, and without it a move larger than the last threshold "
                    "has no speed")
            continue
        if limit is None:
            raise ValueError(f"'motion.joint_speed_ladder' entry {index + 1} is null but is "
                             "not the last: everything after it would be unreachable")
        if float(limit) <= 0.0:
            raise ValueError("'motion.joint_speed_ladder' limits must be positive")
        if previous is not None and float(limit) <= previous:
            raise ValueError("'motion.joint_speed_ladder' limits must strictly increase")
        previous = float(limit)


def _validate_fast(config: PlannerConfig) -> None:
    fast = config.fast
    if fast.no_occlusion and not fast.enabled:
        # Turning sight lines off on its own would quietly change what "coverage" means
        # in a run that does not announce itself as fast.
        raise ValueError("'fast.no_occlusion' needs 'fast.enabled': switching the "
                         "occlusion test off is only honest inside a fast run")
    if fast.enabled and config.target.sample_grid_mm < FAST_MIN_SAMPLE_GRID_MM:
        raise ValueError(
            f"a fast run needs target.sample_grid_mm >= {FAST_MIN_SAMPLE_GRID_MM:g} mm, "
            f"not {config.target.sample_grid_mm:g}: at the shipped grid it would cost the "
            "same time as an accurate run and only be slower to distrust")


def _validate_strokes(stroke: StrokeParams) -> None:
    if stroke.connector not in CONNECTOR_MODES:
        raise ValueError(f"'stroke.connector' must be one of {CONNECTOR_MODES}, not "
                         f"{stroke.connector!r}")
    if stroke.route_style not in ROUTE_STYLES:
        raise ValueError(f"'stroke.route_style' must be one of {ROUTE_STYLES}, not "
                         f"{stroke.route_style!r}")
    if stroke.stroke_corner_cost <= 0.0:
        raise ValueError("'stroke.stroke_corner_cost' must be positive: it is what one "
                         "sweep costs the selection next to a pose's 1.0, and 2.0 is the "
                         "count of corners a band really needs")
    if stroke.cost_model == "corners" and stroke.stroke_corner_cost < 1.0:
        raise ValueError("'stroke.stroke_corner_cost' below 1.0 would price a sweep as "
                         "cheaper than the single stop it still has to make; 1.0 is the "
                         "floor, i.e. 'a band costs like a pose'")
    if stroke.side_scan_azimuths < 1:
        raise ValueError("'stroke.side_scan_azimuths' must be at least 1: it is how many "
                         "tangent directions the tilted seed family is laid on, and 1 is "
                         "the pre-2026-11-01 rule (one tangent, both signs)")
    if stroke.side_scan_spacing_mm < 0.0:
        raise ValueError("'stroke.side_scan_spacing_mm' cannot be negative: 0.0 means "
                         "'twice aim_grid_mm', which is the pre-2026-11-01 rule")
    if not 0.0 <= stroke.side_scan_wall_deg < 90.0:
        raise ValueError("'stroke.side_scan_wall_deg' must be in [0, 90): it is how far a "
                         "sample's normal has to lean off the face's dominant normal before "
                         "the tilted family is seeded there, and 0.0 means 'everywhere'")
    if stroke.side_scan_max_seeds < 0:
        raise ValueError("'stroke.side_scan_max_seeds' cannot be negative: 0 means no "
                         "ceiling, which is the pre-2026-11-01 behaviour")
    if stroke.plan_mode not in PLAN_MODES:
        raise ValueError(f"'stroke.plan_mode' must be one of {PLAN_MODES}, not "
                         f"{stroke.plan_mode!r}")
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
    if stroke.joint_step_deg <= 0.0:
        raise ValueError("'joint_step_deg' must be positive")
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
    if stroke.order_j6_weight_mm_per_deg < 0.0:
        raise ValueError("'order_j6_weight_mm_per_deg' cannot be negative: it is a price per "
                         "degree of wrist twist, and 0.0 is 'not priced'")
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


def _validate_surface(target: TargetParams) -> None:
    """The two site-facing knobs introduced with the merged coverage design."""
    if not 0.0 <= target.surface_bias <= 1.0:
        raise ValueError(f"'surface_bias' must be in [0, 1], not {target.surface_bias!r} - "
                         "out of range is refused rather than clamped, because a clamped "
                         "bias looks like a deliberate choice")
    if target.max_points is not None and target.max_points < 2:
        raise ValueError("'max_points' must be at least 2, or null to skip the budget "
                         "layer - one point cannot describe a path")


def _validate_robot(robot: RobotParams) -> None:
    # The two sequence prices and the J6 travel charge are meaningful whether or not the
    # gate is on: a pose-only delivery still chooses a branch for every point, and a negative
    # price would reward the change it exists to avoid.
    for name in ("config_change_cost_deg", "wrist_flip_cost_deg", "j6_travel_weight"):
        if getattr(robot, name) < 0.0:
            raise ValueError(f"'robot.{name}' cannot be negative: it is a cost added to the "
                             "alternative that carries the change, so a negative value "
                             "would reward the change it exists to avoid")
    _validate_track(robot)
    # The envelope ceiling is checked BEFORE the `enabled` early return, because it is a fact
    # about the machine rather than a setting of the planning sieve: a file that carries a
    # wider band would otherwise only be refused on the day someone turned the gate on.
    if robot.joint_margin_deg < MIN_JOINT_MARGIN_DEG:
        raise ValueError(
            f"'robot.joint_margin_deg' must be at least {MIN_JOINT_MARGIN_DEG:g} deg, not "
            f"{robot.joint_margin_deg:g}: the usable band is the "
            f"{MAX_JOINT_SPAN_FRACTION:.0%} span with this margin inside it, and a point that "
            "sits closer than this to a limit is one ROBOGUIDE calls unreachable over a hair "
            "of model difference (measured 2026-11-02: J2 = 74.75 of 76 answered MOTN-018). "
            "Tighter is fine; looser is refused")
    if not 0.0 < robot.joint_span_fraction <= MAX_JOINT_SPAN_FRACTION:
        raise ValueError(
            f"'robot.joint_span_fraction' must be in (0, {MAX_JOINT_SPAN_FRACTION:g}]: the "
            "usable band is the axis span times this fraction, about the span's centre, and "
            f"{MAX_JOINT_SPAN_FRACTION:g} is the ceiling. A wider envelope is how a plan looks "
            "healthy here and comes back 'position not reachable' from the controller; "
            "tighter is fine, looser is refused")
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
    if robot.max_step_deg <= 0.0:
        raise ValueError("'robot.max_step_deg' must be positive")
    if robot.min_margin_warn_deg < 0.0:
        raise ValueError("'robot.min_margin_warn_deg' cannot be negative")
    if robot.j6_weight <= 0.0:
        raise ValueError("'robot.j6_weight' must be positive: it multiplies the cost of a "
                         "J6 change when two solutions are compared, and 1.0 is 'no "
                         "preference'")
    if robot.wrist_flip_policy not in WRIST_FLIP_POLICIES:
        raise ValueError(f"'robot.wrist_flip_policy' must be one of "
                         f"{WRIST_FLIP_POLICIES}, not {robot.wrist_flip_policy!r}: it picks "
                         "which wrist branch the delivered program may use")
    j5_low, j5_high = robot.soft_limits_deg[4]
    if robot.j5_min_abs_deg >= (j5_high - j5_low) / 2.0:
        raise ValueError("'robot.j5_min_abs_deg' leaves no usable J5 band inside the soft "
                         f"limits [{j5_low}, {j5_high}]")
    if robot.j5_min_abs_deg <= 0.0:
        raise ValueError("'robot.j5_min_abs_deg' must be positive: J5 = 0 is the wrist "
                         "singularity, not a safe centre")


def _validate_track(robot: RobotParams) -> None:
    """The seventh axis, checked whether or not the gate is on.

    Separate from `_validate_robot`'s early return on `enabled=false` on purpose: the shipped
    configuration ships the gate off, and a rail whose limits fall outside its own length is
    a typo that would otherwise only surface on the day someone turned the gate on.
    """
    if not robot.track_enabled:
        return
    if robot.track_axis not in TRACK_AXES:
        raise ValueError(f"'robot.track_axis' must be one of {TRACK_AXES}, not "
                         f"{robot.track_axis!r}: the rail is a translation along one "
                         "base-frame axis and nothing here measures a mounting angle")
    if robot.track_length_mm <= 0.0:
        raise ValueError("'robot.track_length_mm' must be positive")
    low, high = _track_limits(robot)
    if not 0.0 <= low < high <= robot.track_length_mm:
        raise ValueError(
            f"'robot.track_limits_mm' [{low}, {high}] must sit inside the rail "
            f"[0, {robot.track_length_mm:g}] and be non-empty")
    if robot.track_margin_mm < 0.0:
        raise ValueError("'robot.track_margin_mm' cannot be negative")
    if high - low <= 2.0 * robot.track_margin_mm:
        raise ValueError("'robot.track_margin_mm' leaves no usable travel on the rail")
    if robot.track_grid_mm <= 0.0:
        raise ValueError("'robot.track_grid_mm' must be positive")
    if int(robot.track_candidates) < 1:
        raise ValueError("'robot.track_candidates' must be at least 1")
    if robot.track_nominal_reach_mm <= 0.0:
        raise ValueError("'robot.track_nominal_reach_mm' must be positive")
    if robot.track_cost_deg_per_mm < 0.0:
        raise ValueError("'robot.track_cost_deg_per_mm' cannot be negative")
    if str(robot.track_joint_label).upper() not in TRACK_JOINT_LABELS:
        raise ValueError(f"'robot.track_joint_label' must be one of {TRACK_JOINT_LABELS}, "
                         f"not {robot.track_joint_label!r}: it is how the seventh axis is "
                         "spelled inside a joint-representation block")


