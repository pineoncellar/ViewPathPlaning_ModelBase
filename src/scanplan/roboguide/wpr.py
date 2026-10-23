"""FANUC orientation: 3x3 rotation matrix <-> (W, P, R) Euler angles.

Vendored from `reference/PHASE_6_Export_Roboguide.py` when implemented - port it
rather than rewriting, because two things here are easy to get wrong and
expensive to debug.

The convention is FANUC's fixed-axis XYZ Euler angles, R = Rz(R) * Ry(P) * Rx(W).
This is NOT intrinsic ZYX; mixing the two yields a program that loads happily
and puts the tool in the wrong attitude.

The pitch singularity (|P| = 90 deg) needs its own branch: when the cosine of
the pitch vanishes only the combination of W and R is determined, so W is fixed
at zero and R carries the in-plane rotation. Without that branch the angles
jitter.

Number formatting is part of the contract too: the controller writes fixed three
decimals with no leading zero (`.094`, not `0.094`).
"""

from __future__ import annotations

import math
import re

import numpy as np

# FANUC registers an ASCII program under a plain program name, and the ASCII file
# has to carry that same name. A name with spaces (or a timestamped file name) is
# rejected with "Invalid name in /PROG section".
PROG_NAME_PATTERN = re.compile(r"[A-Z][A-Z0-9_]{0,15}")


def normalise_prog_name(name: str) -> str:
    """Validate a FANUC program name, which is also the .LS file name."""
    candidate = str(name).strip().upper()
    if not PROG_NAME_PATTERN.fullmatch(candidate):
        raise ValueError(
            f"invalid FANUC program name {name!r}: the program name is also the .LS "
            "file name, so it must start with a letter, contain only A-Z, 0-9 and "
            "underscore, and be at most 16 characters with no spaces"
        )
    return candidate


def fanuc_num(value: float, width: int = 10) -> str:
    """Format like the controller does: fixed 3 decimals, no leading zero.

    e.g. 1349.372 -> "  1349.372",  0.094 -> "      .094",  -0.0 -> "       .000"
    """
    if abs(value) < 5e-4:
        text = "0.000"
    else:
        text = f"{value:.3f}"
    if text.startswith("0."):
        text = text[1:]
    elif text.startswith("-0."):
        text = "-" + text[2:]
    return text.rjust(width)


def rot_to_fanuc_wpr(rotation: np.ndarray) -> tuple[float, float, float]:
    """3x3 rotation matrix -> FANUC (W, P, R) in degrees.

    FANUC uses fixed-axis XYZ Euler angles, i.e. R = Rz(R) * Ry(P) * Rx(W).
    """
    rot = np.asarray(rotation, dtype=float).reshape(3, 3)
    sin_p = float(np.clip(-rot[2, 0], -1.0, 1.0))
    pitch = math.asin(sin_p)
    if abs(math.cos(pitch)) > 1e-9:
        w = math.atan2(rot[2, 1], rot[2, 2])
        r = math.atan2(rot[1, 0], rot[0, 0])
    else:
        # gimbal lock: only W - R (pitch = +90) resp. W + R (pitch = -90) is
        # determined, so W is fixed at zero and R carries the in-plane rotation
        w = 0.0
        if sin_p > 0:
            r = -math.atan2(rot[0, 1], rot[0, 2])
        else:
            r = math.atan2(-rot[0, 1], -rot[0, 2])
    return tuple(math.degrees(angle) for angle in (w, pitch, r))


def wpr_to_rot(w: float, p: float, r: float) -> np.ndarray:
    """Inverse of rot_to_fanuc_wpr, used for the round-trip self check."""
    w, p, r = (math.radians(v) for v in (w, p, r))
    cw, sw = math.cos(w), math.sin(w)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cw, -sw], [0.0, sw, cw]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def _nearest_turn(value: float, reference: float) -> float:
    """`value` shifted by whole turns to sit closest to `reference`."""
    return value + 360.0 * round((reference - value) / 360.0)


def unwrap_wpr(points) -> list[dict]:
    """Make W and R continuous along the delivered order.

    The angles from `rot_to_fanuc_wpr` are in (-180, 180], so two consecutive poses whose
    wrist is near +-180 come back as +179 and -179 - a 358 degree change in the FILE for a
    2 degree change in the world. The controller interpolates the numbers it is given, so
    that difference is a real wrist spin even though the pose is continuous. Only W and R
    can be unwrapped (P is the pitch, confined to +-90); the rotation they describe is
    unchanged, which is why the round trip still holds.

    Order matters, so this runs after any re-placement and before the file is rendered.
    """
    previous = None
    for point in points:
        w, p, r = (float(value) for value in point["wpr_deg"])
        if previous is not None:
            w = _nearest_turn(w, previous[0])
            r = _nearest_turn(r, previous[2])
        point["wpr_deg"] = (w, p, r)
        point["rotation"] = wpr_to_rot(w, p, r)
        previous = (w, p, r)
    return points
