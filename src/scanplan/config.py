"""Planner configuration: scanner limits, sampling, tracker.

Holds the PLACEHOLDER scanner values in exactly one place, so replacing them
with measured ones stays a one-file change. `max_incidence_deg` is the most
sensitive number in the planner and is not yet measured.

Loads overrides from a JSON file (see `configs/`).

An override file may name only the keys it wants to change: it is merged into
the shipped defaults. Unknown sections and unknown keys are refused, because a
mistyped parameter would otherwise leave a placeholder silently in force.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

DEFAULT_CONFIG_NAME = "scanner_trackscan49.json"
FACES = ("upper", "lower", "auto")


@dataclass(frozen=True)
class ScannerParams:
    """The scanner's geometry and quality envelope. All values are placeholders."""

    standoff_mm: float
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
    """Sampling of the target face and the fixture-contact exemption."""

    sample_grid_mm: float
    contact_mm: float
    contact_margin_mm: float
    face: str
    band_deg: float
    knn: int


@dataclass(frozen=True)
class TrackerParams:
    """The tracker's envelope. Its real pose is unknown until the site provides it."""

    half_angle_deg: float
    distance_mm: tuple[float, float]
    aim_at: str
    markers_omnidirectional: bool
    model_robot_occlusion: bool


@dataclass(frozen=True)
class PlannerConfig:
    scanner: ScannerParams
    target: TargetParams
    tracker: TrackerParams
    occlusion_tol_mm: float
    seed: int
    source: str = ""
    override: str | None = None

    @property
    def contact_distance_mm(self) -> float:
        """Distance below which a sample counts as clamped, margin included."""
        return self.target.contact_mm + self.target.contact_margin_mm


def default_config_path() -> Path:
    """The shipped configuration, resolved from this file rather than the CWD."""
    return Path(__file__).resolve().parents[2] / "configs" / DEFAULT_CONFIG_NAME


def load_config(path: str | Path | None = None) -> PlannerConfig:
    """Load the shipped configuration, optionally overridden by `path`."""
    base_path = default_config_path()
    data = _read_json(base_path)
    override_name: str | None = None
    if path is not None:
        override_path = Path(path)
        data = _deep_merge(data, _read_json(override_path))
        override_name = str(override_path)
    return _build(data, str(base_path), override_name)


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


def _build(data: dict[str, Any], source: str, override: str | None) -> PlannerConfig:
    _reject_unknown(data)
    scanner = _section(data, "scanner", ScannerParams)
    target = _section(data, "target", TargetParams)
    tracker = _section(data, "tracker", TrackerParams)
    config = PlannerConfig(
        scanner=scanner,
        target=target,
        tracker=tracker,
        occlusion_tol_mm=float(data["occlusion_tol_mm"]),
        seed=int(data["seed"]),
        source=source,
        override=override,
    )
    _validate(config)
    return config


def _reject_unknown(data: dict[str, Any]) -> None:
    known = {"scanner", "target", "tracker", "occlusion_tol_mm", "seed"}
    unknown = sorted(k for k in data if k not in known and not k.startswith("_"))
    if unknown:
        raise ValueError(f"unknown config section(s): {unknown}")

    for section, cls in (
        ("scanner", ScannerParams),
        ("target", TargetParams),
        ("tracker", TrackerParams),
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
    """Turn JSON lists into the tuples the frozen dataclasses declare."""
    if get_origin(annotation) is tuple:
        members = get_args(annotation)
        if not members:
            return tuple(value)
        if members[-1] is Ellipsis:
            return tuple(_scalar(members[0], item) for item in value)
        if len(value) != len(members):
            raise ValueError(f"expected {len(members)} values, got {value!r}")
        return tuple(_scalar(member, item) for member, item in zip(members, value, strict=True))
    return _scalar(annotation, value)


def _scalar(annotation: Any, value: Any) -> Any:
    if isinstance(annotation, type):
        try:
            return annotation(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"cannot read {value!r} as {annotation.__name__}") from error
    return value


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
    if target.face not in FACES:
        raise ValueError(f"'face' must be one of {FACES}, not {target.face!r}")
    if config.tracker.model_robot_occlusion:
        # There is no robot model at this stage, so this flag cannot be honoured.
        raise ValueError("'model_robot_occlusion' is not implemented; leave it false")
