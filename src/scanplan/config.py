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
import math
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
# How a seed's roll is chosen. `reference` follows the qianduan2 rule (tool x on station
# +X); `frame` is the old explicit zero, kept for the legacy preset.
ROLL_MODES = ("reference", "frame")


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
    """

    sample_grid_mm: float
    contact_mm: float
    contact_margin_mm: float
    face: str
    band_deg: float
    knn: int
    coverage_target_pct: float | None
    delivery_mode: str
    # The one curvature-preference input (merge plan section 6). 0.0 = coverage first,
    # 0.5 = the shipped behaviour, 1.0 = regular programs first. Everything it implies is
    # derived into the `stroke` section by `surface_profile`, and written back per key.
    surface_bias: float
    # The program point cap. null means no budget layer at all, which is the shipped
    # behaviour and keeps the selection path identical to the one before the budget existed.
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
    * `connector_level` caps the most elaborate connector the planner may use: 2 allows
      the joint route (a short straight lift, a joint move to the next sweep's pre-point,
      then a short straight approach), 1 forbids joint travel and keeps every move
      straight, 0 also allows the corner-less straight move. It is a cap and not a
      preference: a move between two identical poses shorter than `joint_travel_min_mm`
      stays straight at every level, and a longer or rotated move is the joint route
      wherever joint travel is allowed. 0 stays off until ROBOGUIDE has confirmed it.
    * `order_axis_weight_mm_per_deg` prices a pose change in the ordering. Distance alone
      ignores the axis, so two adjacent strokes 60 degrees apart in pose look free to
      travel between and cost a large re-orientation instead.
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
    # The "short" threshold of the connector decision: a move between two IDENTICAL poses
    # shorter than this is flown straight, without inserting a joint move into it. Longer
    # than this - or any pose change at all - and the transition is the joint route, whose
    # long move is a joint move, so no straight segment ever has to turn the wrist while
    # travelling. See `scanplan.path._connector`.
    joint_travel_min_mm: float
    max_prune_rounds: int
    cost_model: str
    transition_corners_estimate: float
    connector_level: int
    order_axis_weight_mm_per_deg: float
    # -- roll and wrist (the qianduan2 half of the merge) ------------------------
    # `reference` keeps tool x on station +X, which is what made the reference program's
    # wrist turns small; `frame` is the old explicit zero, kept so an old preset can still
    # reproduce itself. The roll turns the field-of-view rectangle, so it is decided when
    # the seed is built and never re-picked for a coverage point.
    roll_mode: str
    # The per-move wrist turn the program aims for. Exceeding it is reported, never hidden.
    max_wrist_step_deg: float
    # Degrees of deviation from the reference posture the smoother accepts per degree of
    # rotation it saves. Only transition points are ever smoothed.
    roll_reference_bias: float
    # -- derived by `surface_profile` -------------------------------------------
    # Tie-break multiplier for long parallel bands in the selection; 1.0 is neutral.
    parallel_weight: float
    # Share of a point budget the tail repair may spend; only read when `max_points` is set.
    repair_budget_share: float


@dataclass(frozen=True)
class MotionParams:
    """How the delivered program moves between the planned points.

    Speed is a floor setting, not a planning variable; it is carried here only so the
    program has one source for it instead of a flag and a config that can disagree.
    """

    type: str
    termination: str
    speed_mm_s: float


@dataclass(frozen=True)
class RobotParams:
    """The arm the scanner rides on, and the joint window the planner may use.

    `joint_limits_deg` is written in the ROBOT'S WORKSPACE TABLE convention - the numbers a
    workshop reads on the controller and in the datasheet - not in the CAD model's own joint
    zero. `scanplan.robot` owns the conversion between the two, because for this arm they
    are the same physical stops under a different zero (J2 is -90..+46 in the table against
    -60..+76 in the CAD chain, both 136 deg wide).

    `joint_limit_margin` is the derating applied to every stop: 0.9 means the outermost 10%
    of each axis' travel is off limits, so the arm never works against a mechanical stop and
    the controller's own settling has somewhere to go.

    The rest are the three things about the mounting that the planner cannot derive:

    * `base_position_mm` / `base_rpy_deg` place the robot in the station CAD frame, in the
      same way the station file places the part. Get this wrong and every verdict is wrong
      while still looking plausible, which is why the report records both;
    * `tool_length_mm` is the flange-to-TCP distance along the optical axis: 0.0 means the
      TCP is the mounting flange, which is the honest "not measured yet" value and
      understates the reach the scanner ball needs;
    * `tool_mount_deg` is how the bracket is clocked about that axis. It shifts J6 by a
      constant and nothing else, so a wrong value costs J6 accuracy and not the plan.

    `enabled=False` is the escape hatch for a station where the base pose is not known yet:
    the run then behaves exactly as it did before the joint check existed.
    """

    enabled: bool
    model: str
    joint_limits_deg: tuple[tuple[float, float], ...]
    joint_limit_margin: float
    base_position_mm: tuple[float, float, float]
    base_rpy_deg: tuple[float, float, float]
    tool_length_mm: float
    tool_mount_deg: float


@dataclass(frozen=True)
class PlannerConfig:
    scanner: ScannerParams
    target: TargetParams
    tracker: TrackerParams
    stroke: StrokeParams
    motion: MotionParams
    robot: RobotParams
    occlusion_tol_mm: float
    seed: int
    source: str = ""
    override: str | None = None
    # Which layer each `surface_bias`-derived key came from: `base`, `profile` or
    # `override`. Written into the report so "why is this number 200?" has an answer.
    derived: dict = field(default_factory=dict)

    @property
    def surface_bias_source(self) -> str:
        """Where `target.surface_bias` itself was set: `base` or `override`."""
        return str(self.derived.get("surface_bias", "base"))

    @property
    def contact_distance_mm(self) -> float:
        """Distance below which a sample counts as clamped, margin included."""
        return self.target.contact_mm + self.target.contact_margin_mm


def default_config_path() -> Path:
    """The shipped configuration, resolved from this file rather than the CWD."""
    return Path(__file__).resolve().parents[2] / "configs" / DEFAULT_CONFIG_NAME


# The surface profile (merge plan section 6.2). One semantic input - "does this part have
# to be scanned completely, or does a regular program matter more?" - and the `stroke` keys
# it implies. Three anchors, interpolated linearly, and the middle anchor IS the shipped
# configuration: at bias 0.5 every pre-existing key keeps its factory value and the two
# new ones are neutral, which is what makes every old config file behave as before.
#
# `scanner.max_incidence_deg` and `scanner.range_mm` are deliberately NOT in here. They
# are sensor physics, not taste; bending them to buy coverage would fake a capability and
# make the report look better while the cell cannot scan it.
SURFACE_PROFILE_ANCHORS: dict[str, tuple[float, float, float]] = {
    # growth stop threshold: low keeps growing, high gives up earlier
    "min_gain_mm2_per_step": (50.0, 200.0, 600.0),
    # dropped fragments: high means fewer corners and fewer short bands
    "min_stroke_mm": (60.0, 150.0, 300.0),
    # direction family: 2 = the in-plane long axis only, so the program stays parallel
    "seed_dirs": (4.0, 4.0, 2.0),
    # what a pose change costs the ordering: high means fewer re-orientations
    "order_axis_weight_mm_per_deg": (1.0, 2.0, 3.5),
    # neutral tie-break for long parallel bands in the selection
    "parallel_weight": (1.00, 1.00, 1.30),
    # share of a point budget the tail repair may spend
    "repair_budget_share": (0.35, 0.20, 0.05),
}
# Keys the profile derives, in report order.
SURFACE_PROFILE_KEYS = tuple(SURFACE_PROFILE_ANCHORS)


def surface_profile(bias: float) -> dict[str, float]:
    """The `stroke` keys a `surface_bias` implies, at the three anchors and in between.

    Pure, so the anchors are unit-testable, and monotone in `bias`: a higher bias never
    grows a band further, never keeps a shorter fragment, never adds a direction family,
    never makes a pose change cheaper and never hands the repair pass more budget.
    """
    value = float(bias)
    if not 0.0 <= value <= 1.0:
        raise ValueError("surface_bias must be in [0.0, 1.0]")
    profile: dict[str, float] = {}
    for key, (low, middle, high) in SURFACE_PROFILE_ANCHORS.items():
        if value <= 0.5:
            share = value / 0.5
            blended = low + (middle - low) * share
        else:
            share = (value - 0.5) / 0.5
            blended = middle + (high - middle) * share
        if key == "seed_dirs":
            profile[key] = int(round(blended))
        else:
            profile[key] = round(float(blended), 6)
    return profile


def _surface_bias(base: dict[str, Any], override: dict[str, Any]) -> tuple[float, str]:
    """The effective `surface_bias` and which layer set it. An override wins outright."""
    if "surface_bias" in override.get("target", {}):
        return float(override["target"]["surface_bias"]), "override"
    return float(base.get("target", {}).get("surface_bias", 0.5)), "base"


def _profile_sources(base: dict[str, Any], override: dict[str, Any], bias: float,
                     ) -> dict[str, str]:
    """Which layer each derived key came from, for the report.

    An explicit value always wins over the profile, so a key written in the override file
    is reported as `override`; a key the base file already sets to the profile's value is
    reported as `base` (at bias 0.5 that is every pre-existing key).
    """
    profile = surface_profile(bias)
    base_stroke = base.get("stroke", {})
    override_stroke = override.get("stroke", {})
    sources: dict[str, str] = {"surface_bias": _surface_bias(base, override)[1]}
    for key in SURFACE_PROFILE_KEYS:
        if key in override_stroke:
            sources[key] = "override"
        elif key in base_stroke and float(base_stroke[key]) == float(profile[key]):
            sources[key] = "base"
        else:
            sources[key] = "profile"
    return sources


def load_config(path: str | Path | None = None) -> PlannerConfig:
    """Load the shipped configuration, optionally overridden by `path`.

    Three layers, in this order: the shipped file, the `surface_bias` profile, and the
    override file - so an explicitly written key always beats a derived one, and nobody
    has to track which of the derived keys a given preset was written against.
    """
    base_path = default_config_path()
    base = _read_json(base_path)
    override_name: str | None = None
    explicit: dict[str, Any] = {}
    if path is not None:
        override_path = Path(path)
        explicit = _read_json(override_path)
        override_name = str(override_path)
    bias, _source = _surface_bias(base, explicit)
    data = _deep_merge(_deep_merge(base, {"stroke": surface_profile(bias)}), explicit)
    return _build(data, str(base_path), override_name,
                  derived=_profile_sources(base, explicit, bias))


def override(config: PlannerConfig, **sections) -> PlannerConfig:
    """A copy of `config` with some section fields replaced, then re-validated.

    The command line is a second writer of the same knobs as the JSON file, so it has to go
    through the same validation; otherwise the two can disagree about what is legal and the
    illegal value wins only when it arrives by flag.

    Setting `target.surface_bias` here re-derives the profile, exactly as loading a file
    with that key does - a flag that silently left the derived keys at the old profile would
    be a knob that does not turn anything.
    """
    if "surface_bias" in sections.get("target", {}):
        bias = float(sections["target"]["surface_bias"])
        profile = surface_profile(bias)   # raises on an out-of-range bias
        stroke = dict(sections.get("stroke", {}))
        derived = dict(config.derived)
        for key, value in profile.items():
            if key in stroke:
                derived[key] = "override"
            else:
                stroke[key] = value
                derived[key] = f"profile(bias={bias:g})"
        derived["surface_bias"] = "override"
        sections = {**sections, "stroke": stroke}
        config = replace(config, derived=derived)

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
           derived: dict[str, str] | None = None) -> PlannerConfig:
    _reject_unknown(data)
    scanner = _section(data, "scanner", ScannerParams)
    target = _section(data, "target", TargetParams)
    tracker = _section(data, "tracker", TrackerParams)
    stroke = _section(data, "stroke", StrokeParams)
    motion = _section(data, "motion", MotionParams)
    robot = _section(data, "robot", RobotParams)
    config = PlannerConfig(
        scanner=scanner,
        target=target,
        tracker=tracker,
        stroke=stroke,
        motion=motion,
        robot=robot,
        occlusion_tol_mm=float(data["occlusion_tol_mm"]),
        seed=int(data["seed"]),
        source=source,
        override=override,
        derived=dict(derived or {}),
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
    """Turn JSON lists into the tuples the frozen dataclasses declare, nesting included."""
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
    if not 0.0 <= float(target.surface_bias) <= 1.0:
        raise ValueError("'surface_bias' must be in [0.0, 1.0] - 0 is coverage first, "
                         "1 is regular programs first. A value outside that range is "
                         "refused rather than silently clamped, like coverage_target_pct")
    if target.max_points is not None and int(target.max_points) < 2:
        raise ValueError("'max_points' must be at least 2 (a program needs two corners to "
                         "be a program), or null to skip the budget layer entirely")
    if config.tracker.model_robot_occlusion:
        # There is no robot model at this stage, so this flag cannot be honoured.
        raise ValueError("'model_robot_occlusion' is not implemented; leave it false")
    _validate_strokes(config.stroke)
    if config.motion.type not in MOTION_TYPES:
        raise ValueError(f"'motion.type' must be one of {MOTION_TYPES}, "
                         f"not {config.motion.type!r}")
    if config.motion.termination not in TERMINATIONS:
        raise ValueError(f"'motion.termination' must be one of {TERMINATIONS}, "
                         f"not {config.motion.termination!r}")
    if config.motion.speed_mm_s <= 0.0:
        raise ValueError("'motion.speed_mm_s' must be positive")
    _validate_robot(config.robot)


def _validate_robot(robot: RobotParams) -> None:
    """Refuse a joint window or a base pose that could not describe a robot.

    The model name is checked against the kinematic table in `scanplan.robot` before
    anything else: a typo there would otherwise fall through to "every pose unreachable",
    which reads as a planning result rather than as a configuration mistake. The import is
    local because that module needs this one for its own parameter type.
    """
    from .robot import MODELS  # local: that module imports this one for its parameter type

    if not robot.enabled:
        return
    if robot.model not in MODELS:
        raise ValueError(f"'robot.model' must be one of {sorted(MODELS)}, not "
                         f"{robot.model!r} - the kinematic chain and the workspace table "
                         "are both per-model facts, so a new one has to be added to "
                         "scanplan.robot rather than guessed here")
    if len(robot.joint_limits_deg) != 6:
        raise ValueError("'robot.joint_limits_deg' needs one [min, max] pair per axis, "
                         f"six in all, not {len(robot.joint_limits_deg)}")
    for axis, (low, high) in enumerate(robot.joint_limits_deg, start=1):
        if not math.isfinite(low) or not math.isfinite(high) or low >= high:
            raise ValueError(f"'robot.joint_limits_deg' J{axis} needs a min below its max, "
                             f"not {[low, high]}")
    if not 0.0 < robot.joint_limit_margin <= 1.0:
        raise ValueError("'robot.joint_limit_margin' must be in (0, 1]: a share of the "
                         "joint travel, where 0.9 leaves the outermost 10% unused. Above 1 "
                         "the check would allow the mechanical stops themselves")
    for name, values in (("base_position_mm", robot.base_position_mm),
                         ("base_rpy_deg", robot.base_rpy_deg)):
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"'robot.{name}' must be three finite numbers, not {values}")
    if robot.tool_length_mm < 0.0:
        raise ValueError("'robot.tool_length_mm' cannot be negative: it is the distance "
                         "from the mounting flange to the TCP along the optical axis")


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
    if stroke.joint_travel_min_mm <= 0.0:
        raise ValueError("'joint_travel_min_mm' must be positive: it is the travel below "
                         "which a move between two identical poses stays straight")
    if stroke.cost_model not in COST_MODELS:
        raise ValueError(f"'cost_model' must be one of {COST_MODELS}, "
                         f"not {stroke.cost_model!r}")
    if stroke.transition_corners_estimate < 0.0:
        raise ValueError("'transition_corners_estimate' cannot be negative")
    if not 0 <= stroke.connector_level <= 2:
        raise ValueError("'connector_level' must be in [0, 2]: 2 allows the joint travel "
                         "route between sweeps, 1 forbids it and keeps every move "
                         "straight, 0 also allows a corner-less straight move between two "
                         "identical poses")
    if stroke.order_axis_weight_mm_per_deg < 0.0:
        raise ValueError("'order_axis_weight_mm_per_deg' cannot be negative")
    if stroke.roll_mode not in ROLL_MODES:
        raise ValueError(f"'roll_mode' must be one of {ROLL_MODES}, not {stroke.roll_mode!r}")
    if not 0.0 < stroke.max_wrist_step_deg <= 180.0:
        raise ValueError("'max_wrist_step_deg' must be in (0, 180]")
    if stroke.roll_reference_bias < 0.0:
        raise ValueError("'roll_reference_bias' cannot be negative")
    if stroke.parallel_weight <= 0.0:
        raise ValueError("'parallel_weight' must be positive: it multiplies a score, "
                         "and a zero or negative multiplier would invert the ranking")
    if not 0.0 <= stroke.repair_budget_share <= 1.0:
        raise ValueError("'repair_budget_share' must be in [0.0, 1.0]")
