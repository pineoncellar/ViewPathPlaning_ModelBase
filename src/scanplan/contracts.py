"""The data contracts the five planning stages exchange.

The planner runs as one direction of data flow:

    stage 1  geometry      meshes + normals, repaired     -> Station, SheetSplit, TargetSet
    stage 2  candidates    poses the arm can hold         -> list[ScannerPose]
    stage 3  strokes       ordered items + connectors     -> PathPoint sequence
    stage 4  branches      joints, letters, readings      -> PathPoint + Solution
    stage 5  delivery      verified program + report      -> .LS, coverage_report.json

Three rules keep that direction one-way, and this module is where they are written down:

1. **A parameter crosses a stage boundary only as a frozen value.**  `RobotEnvelope` is
   built once, before stage 2, and is the only thing stages 2-4 read about the arm's
   envelope.  No stage re-derives the usable band from a config file: two derivations of
   the same rule drift, and a narrow envelope decided in stage 2 cannot be widened again
   by a different policy in stage 4.

2. **A stage's own trade-off never leaves the stage.**  Which roll convention to use,
   which route to connect two sweeps with, which wrist branch to allow - each is a policy
   inside one stage (see :mod:`scanplan.policies`), chosen from parameters, never from
   mutable state that a later stage could observe.

3. **What a stage hands on is data, not a verdict.**  `ConnectorResult` carries the
   corners AND the joint states that were verified for them, so stage 4 does not have to
   re-guess what stage 3 proved; `GateVerdict` carries the reason next to the boolean, so
   the report can group failures without re-running the IK.

`DeliveredPoint` is the one contract that is a shape rather than a class: the planner
builds `PathPoint`, the report builds `Viewpoint`, and both must carry the same fields
for the exporter to treat them alike.  It is described here once, checked by
:func:`is_delivered_point`, and deliberately not forced on either class - converting
between them would copy arrays on every run for no gain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

__all__ = [
    "AXES",
    "ConnectorResult",
    "DeliveredPoint",
    "GateVerdict",
    "RobotEnvelope",
    "is_delivered_point",
]

#: Joint names, in controller order (J1..J6).
AXES = ("J1", "J2", "J3", "J4", "J5", "J6")


# --------------------------------------------------------------------------- envelope


@dataclass(frozen=True)
class RobotEnvelope:
    """The arm's usable envelope, decided once per run.

    `limits` is what the CONTROLLER accepts; `usable` is what the planner is willing to
    command: the same band pulled in to `span_fraction` of each axis's range about its
    centre, then shrunk again by `margin_deg`.  Both are here because they answer
    different questions - "is this number even legal" versus "is this a number we are
    willing to sit on" - and the report has to be able to say which one rejected a pose.

    Frozen on purpose: the plan calls this the single owner of the envelope decision.
    Stage 2 filters candidates with it, stage 4 picks branches inside it, and stage 5
    audits the file against it.  Three readers, one value.
    """

    enabled: bool
    model: str
    limits_source: str
    limits: tuple[tuple[float, float], ...]
    usable: tuple[tuple[float, float], ...]
    margin_deg: float
    span_fraction: float
    j5_min_abs_deg: float
    max_step_deg: float
    wrist_policy: str
    j6_weight: float
    j6_max_step_deg: float | None

    @classmethod
    def from_gate(cls, gate) -> RobotEnvelope:
        """Snapshot a :class:`~scanplan.kinematics.KinematicGate`.

        The gate already resolved the two limit sets and the tool, so this is a copy, not
        a second derivation.  A disabled gate still yields an envelope (`enabled=False`,
        `limits_source="disabled"`) so downstream code never has to branch on None.
        """
        params = gate.params
        robot = getattr(gate, "robot", None)
        if robot is None:
            raw = tuple(tuple(float(value) for value in pair)
                        for pair in params.soft_limits_deg)
            return cls(
                enabled=False, model=params.model, limits_source="disabled",
                limits=raw, usable=raw,
                margin_deg=float(params.joint_margin_deg),
                span_fraction=float(getattr(params, "joint_span_fraction", 1.0)),
                j5_min_abs_deg=float(params.j5_min_abs_deg),
                max_step_deg=float(params.max_step_deg),
                wrist_policy=str(getattr(params, "wrist_flip_policy", "any")),
                j6_weight=float(getattr(params, "j6_weight", 1.0)),
                j6_max_step_deg=getattr(params, "j6_max_step_deg", None),
            )
        return cls(
            enabled=True, model=params.model, limits_source=str(robot.limits_source),
            limits=tuple(tuple(float(value) for value in pair) for pair in robot.limits),
            usable=tuple(tuple(float(value) for value in pair) for pair in robot.usable),
            margin_deg=float(params.joint_margin_deg),
            span_fraction=float(getattr(params, "joint_span_fraction", 1.0)),
            j5_min_abs_deg=float(params.j5_min_abs_deg),
            max_step_deg=float(params.max_step_deg),
            wrist_policy=str(getattr(params, "wrist_flip_policy", "any")),
            j6_weight=float(getattr(params, "j6_weight", 1.0)),
            j6_max_step_deg=getattr(params, "j6_max_step_deg", None),
        )

    def band(self, axis: int) -> tuple[float, float]:
        """The usable band of one axis, as (low, high) in degrees."""
        low, high = self.usable[int(axis)]
        return float(low), float(high)

    def describe(self) -> dict:
        """The report's `kinematics.envelope` block."""
        return {
            "enabled": self.enabled,
            "model": self.model,
            "limits_source": self.limits_source,
            "limits_deg": [[round(low, 3), round(high, 3)] for low, high in self.limits],
            "usable_limits_deg": [[round(low, 3), round(high, 3)]
                                  for low, high in self.usable],
            "margin_deg": self.margin_deg,
            "span_fraction": self.span_fraction,
            "wrist_policy": self.wrist_policy,
            "j6_weight": self.j6_weight,
            "j6_max_step_deg": self.j6_max_step_deg,
            "note": (
                "one envelope per run: stage 2 filters candidates with it, stage 4 picks "
                "branches inside it, stage 5 audits the file against it"
            ),
        }


# --------------------------------------------------------------------------- verdict


@dataclass(frozen=True)
class GateVerdict:
    """What the kinematic gate decided about one pose, and why.

    `reason` is what the report groups by - "the plan lost these points to the joint
    limits" and "the plan lost them because the arm cannot reach that far" are different
    problems with different fixes, and a single boolean cannot say which happened.

    `solution` is the accepted solution when `ok`; typed loosely because the concrete
    class lives in :mod:`scanplan.kinematics`, which imports this module, not the other
    way round.
    """

    ok: bool
    reason: str
    solution: Any = None


# --------------------------------------------------------------------------- connector


@dataclass
class ConnectorResult:
    """What a connector policy produced between two sweeps.

    The point of the contract is that stage 4 does not have to re-derive anything: a
    route is only returned together with the joint states that were verified for it.  A
    policy that inserts corners but cannot name their joints returns `joints=[]`, which
    stage 4 treats as "unverified" - never as "fine".

    `level` is the route's name in the report (`J-direct`, `lift-1`, ...).  `failures`
    counts why the stages above it were refused, so the report can say "the joint move
    collided 6 times" instead of only "lift used".
    """

    level: str
    corners: list = field(default_factory=list)
    joints: list[np.ndarray] = field(default_factory=list)
    margin_mm: float = float("inf")
    #: Rotation per millimetre of flight, when the route could not spread the turn enough.
    #: Kept for the report; the hard gate does not refuse on it (see `path._connector`).
    rate: float = 0.0
    failures: dict[str, int] = field(default_factory=dict)

    @property
    def direct(self) -> bool:
        """True when the connector inserted no corner at all."""
        return not self.corners

    @property
    def verified(self) -> bool:
        """True when every corner carries a joint state that was checked."""
        return len(self.joints) == len(self.corners)


# ---------------------------------------------------------------------- delivered point


class DeliveredPoint(Protocol):
    """The shape everything after stage 3 must carry.

    Not a base class: `PathPoint` (planner) and `Viewpoint` (report) both satisfy it
    structurally, and converting between them on every run would only copy arrays.
    `is_delivered_point` is the runtime check the tests and the exporter use.
    """

    position: np.ndarray
    axis: np.ndarray
    roll_deg: float
    role: str
    motion: str | None
    representation: str | None


#: The attributes a delivered point must expose; `joints_deg` may be absent when the
#: gate is off, and `config` is a hint rather than a requirement.
_REQUIRED = ("position", "axis", "roll_deg", "role")
_OPTIONAL = ("motion", "representation")


def is_delivered_point(candidate: object) -> bool:
    """Whether `candidate` satisfies the delivered-point contract.

    Data attributes, not methods: `@runtime_checkable` protocols only validate method
    presence on the Python versions this project supports, and a silent `isinstance`
    answer is worse than this explicit check.
    """
    return all(hasattr(candidate, name) for name in _REQUIRED + _OPTIONAL)
