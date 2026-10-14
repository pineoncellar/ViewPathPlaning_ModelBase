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

SKELETON ONLY - no implementation yet.
"""
