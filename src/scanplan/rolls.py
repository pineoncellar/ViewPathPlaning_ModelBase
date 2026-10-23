"""Roll decided from the optical axis alone, at the moment the seed is made.

Why this exists at all. A stroke carries a CONSTANT pose, and the footprint is a
rectangle - 600 x 500 mm on this scanner, not a disc - so the roll decides which
samples a sweep can reach. Leaving every seed at roll 0 and letting the exporter
derive a frame means the roll is effectively chosen by whichever of +Z or +X the
frame builder happens to use as a helper, and `scanner_frame` switches helpers when
the optical axis gets within about 25 degrees of vertical. Two neighbouring sweep
directions then differ by ~90 degrees of roll for no geometric reason.

The alternative that looks tempting - grow first, then smooth the roll and re-measure
the coverage - is wrong here. A roll change of 25 degrees moves samples out of the
non-square footprint, so re-measuring after smoothing can drop coverage the plan had
already spent points on, and the second measurement has to be the one believed.
Deciding once, before the coverage is measured, keeps "one evaluation, one number".

The rule is deliberately boring: put the scanner's local x as close to a fixed
station direction as the axis allows. For a mostly horizontal optical axis that x is
already near the horizontal plane, so this only removes the arbitrary 90 degree jumps
the helper switch introduces. It is a function of the axis alone, which is what makes
it safe to apply inside growth.

One consequence worth knowing before reading a report: the roll NUMBER can still jump
by ~90 degrees when the axis crosses the helper threshold, even though the FRAME does
not. `scanner_frame` picks a different helper on either side and this module compensates,
so the composed frame is the same continuous X-projection throughout. Only the number is
discontinuous, and only the number is stored on the stroke.
"""

from __future__ import annotations

import numpy as np

from .geometry import scanner_frame, unit

# The station direction the footprint's long side should follow. +X is the workshop's
# "along the bench", and it is also the direction a hand-driven sweep naturally takes.
REFERENCE = np.array([1.0, 0.0, 0.0])
# Below this the reference is too close to the optical axis to define a roll, so the
# base frame is used unchanged (which is roll 0 by definition).
_DEGENERATE = 1e-6


def roll_for_axis(axis, reference=REFERENCE) -> float:
    """The roll, in degrees, that turns the scanner's local x towards `reference`.

    Returns 0 when the axis is parallel to the reference: there is no in-plane
    direction to align to, and pretending otherwise would produce a roll that depends on
    floating-point noise.
    """
    optical = unit(np.asarray(axis, dtype=float).reshape(3))
    target = np.asarray(reference, dtype=float).reshape(3)
    target = target - optical * float(np.dot(target, optical))
    length = float(np.linalg.norm(target))
    if length <= _DEGENERATE:
        return 0.0
    target = target / length
    base = scanner_frame(optical, 0.0)
    x0, y0 = base[:, 0], base[:, 1]
    return float(np.degrees(np.arctan2(float(np.dot(target, y0)),
                                       float(np.dot(target, x0)))))


def roll_consistent(axis, roll_deg: float, reference=REFERENCE,
                    tol_deg: float = 1.0) -> bool:
    """Whether `roll_deg` is the roll this axis would be given by `roll_for_axis`.

    Used as a cheap self-check in the tests and in the report: a stroke whose roll does not
    match its axis is a stroke the exporter will place differently from what was measured,
    which is the failure mode this module exists to remove.
    """
    expected = roll_for_axis(axis, reference)
    difference = (float(roll_deg) - expected + 180.0) % 360.0 - 180.0
    return abs(difference) <= tol_deg
