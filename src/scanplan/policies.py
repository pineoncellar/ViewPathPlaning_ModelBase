"""The trade-offs each stage is allowed to make, and the one implementation that ships.

The plan this repository follows puts every undecided engineering trade-off behind an
explicit slot, one slot per stage, and keeps each slot's implementation private to its
stage.  A slot has exactly three things:

* a **Protocol** in this module: the shape every implementation has to satisfy,
* a **factory** that resolves a configuration value to one implementation,
* one **shipped implementation**, chosen because it is the chain that has been closed on
  the machine.  A second implementation is added when a second site needs it, not before:
  the alternative is a plugin surface around code nobody has run.

Slots implemented here:

    A  roll strategy      stage 2  which rolls a candidate is generated at
    B  connector          stage 3  how the head gets from one sweep to the next

Slot C (envelope and wrist branch) is deliberately NOT a slot: it is a parameter,
`RobotEnvelope`.  An envelope that could be changed by a stage would break the one rule
that keeps the pipeline one-way - see `scanplan.contracts`.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .config import CONNECTOR_MODES, ROLL_MODES
from .contracts import ConnectorResult
from .geometry import tangents, tilt_axis
from .rolls import roll_for_axis
from .scanner_model import ScannerPose

__all__ = [
    "ConnectorPolicy",
    "DirectThenLift",
    "DiscreteRolls",
    "LockedRolls",
    "RollPolicy",
    "connector_policy",
    "roll_policy",
    "ROLL_MODES",
    "CONNECTOR_MODES",
]

#: Rolls tried about the SAME optical axis when the joint gate refuses the pose's own roll.
#: A spin about the optical axis does not change what the scanner looks at, but it does move
#: the FLANGE, so it decides whether the arm can hold the pose at all.  On the reference
#: station a locked roll was infeasible for 1.4 % of the target face and 21 of 30 sampled
#: leftovers were reachable at another roll about the same axis.
ROLL_RETRY_DEG = (45.0, 315.0, 90.0, 270.0, 135.0, 225.0, 180.0)


class RollPolicy(Protocol):
    """Slot A: the rolls a candidate is generated at, for one aim point."""

    name: str

    def rolls(self, axis: np.ndarray, params) -> list[float]:
        """The roll values (degrees) to emit for `axis`."""

    def retry_rolls(self) -> tuple[float, ...]:
        """Extra rolls to try when the gate refuses the pose at its own roll."""


class DiscreteRolls:
    """Every roll in `scanner.rolls_deg`: the larger search space, more J6 to pay for.

    Listed first in the baseline for a reason - the field of view is a 600 x 500 mm
    rectangle, so a quarter turn about the optical axis covers a different set of samples.
    Making this the default costs a J6 turn between neighbours that share an axis: measured
    on the reference station as four J6 steps of 74-95 degrees.
    """

    name = "discrete"

    def rolls(self, axis, params) -> list[float]:
        return [float(value) for value in params.rolls_deg]

    def retry_rolls(self) -> tuple[float, ...]:
        return ()


class LockedRolls:
    """One roll per optical axis, decided by `rolls.roll_for_axis` (the shipped default).

    J6 stays still because two poses that share an axis share a roll, so a program made of
    them has no wrist spin for free.  The price is that the roll is no longer chosen to
    place the footprint, and that a locked roll is occasionally infeasible for the arm;
    `retry_rolls` is what buys those poses back, tried in order of the smallest turn so a
    retried pose stays close to its neighbours.
    """

    name = "locked"

    def rolls(self, axis, params) -> list[float]:
        return [roll_for_axis(axis)]

    def retry_rolls(self) -> tuple[float, ...]:
        return ROLL_RETRY_DEG


def roll_policy(mode: str | None) -> RollPolicy:
    """Resolve `scanner.roll_mode` to its implementation."""
    chosen = str(mode or "locked").strip().lower()
    if chosen not in ROLL_MODES:
        raise ValueError(f"scanner.roll_mode must be one of {ROLL_MODES}, not {mode!r}")
    return LockedRolls() if chosen == "locked" else DiscreteRolls()


def ladder(point, normal, params, *, policy: RollPolicy | None = None
           ) -> list[ScannerPose]:
    """The poses aimed at one surface point: every distance, tilt and policy roll.

    Replaces the roll loop of the baseline: the tilt still decides the optical axis, and
    the policy decides which rolls are emitted for it.
    """
    policy = policy or roll_policy(getattr(params, "roll_mode", "locked"))
    point = np.asarray(point, dtype=float).reshape(3)
    normal = np.asarray(normal, dtype=float).reshape(3)
    normal = normal / np.linalg.norm(normal)
    first_tangent, second_tangent = tangents(normal)
    inward = -normal

    poses: list[ScannerPose] = []
    for distance in params.distances_mm:
        position = point + normal * distance
        for tilt_deg in params.tilts_deg:
            if tilt_deg == 0.0:
                axis = inward
            elif tilt_deg > 0.0:
                axis = tilt_axis(inward, first_tangent, tilt_deg)
            else:
                axis = tilt_axis(inward, second_tangent, -tilt_deg)
            for roll_deg in policy.rolls(axis, params):
                poses.append(ScannerPose(position=position.copy(), axis=axis.copy(),
                                         roll_deg=float(roll_deg)))
    return poses


def gate_poses(poses, keep, *, policy: RollPolicy) -> list[ScannerPose]:
    """Apply `keep` to `poses`, re-trying a refused pose at the policy's retry rolls.

    Only the roll changes, so a retried pose keeps its sight line: what buys it is the
    wrist position, not the aim.  A pose the gate accepts first time is passed through
    untouched, which is what keeps a plan whose locked roll already works unchanged.
    """
    if keep is None:
        return list(poses)
    kept: list[ScannerPose] = []
    for pose in poses:
        if keep(pose):
            kept.append(pose)
            continue
        for turn in policy.retry_rolls():
            retried = ScannerPose(position=pose.position, axis=pose.axis,
                                  roll_deg=(pose.roll_deg + turn) % 360.0)
            if keep(retried):
                kept.append(retried)
                break
    return kept


class ConnectorPolicy(Protocol):
    """Slot B: how the head travels between two delivered items."""

    name: str

    def connect(self, previous, entry, ctx) -> ConnectorResult | None:
        """A verified route, or None when this policy has none."""


def connector_policy(mode: str | None) -> ConnectorPolicy:
    """Resolve `stroke.connector` to its implementation.

    The mechanics live in :mod:`scanplan.path` because the chain needs the clearance sweep,
    the kinematic walk and the path record together; this factory is the slot's name, so a
    second connector is added by registering it here rather than by editing `path`.

    Two implementations, and they differ in what the FIRST move is:

    * `direct-then-lift` - one joint move, then the lift ladder. Every extra fallback is a
      route nobody has flown: the measurement that removed the old three-level chain was that
      its retract corners, not the wrist, held all seven unsolvable points of the 99 % run.
    * `straight-then-lateral` - a straight Cartesian line, a sideways waiting point when the
      line is blocked, and this chain as the fallback. The line is what the program looks like
      it does, so the delivered path reads as one continuous sweep instead of an arc between
      sweeps; the fallback is what keeps the style from losing the coverage it was chosen for.
    """
    chosen = str(mode or "direct-then-lift").strip().lower()
    if chosen not in CONNECTOR_MODES:
        raise ValueError(f"stroke.connector must be one of {CONNECTOR_MODES}, "
                         f"not {mode!r}")
    from .path import ConnectorChain, StraightThenLateral  # local import: path imports us

    return StraightThenLateral() if chosen == "straight-then-lateral" else ConnectorChain()

