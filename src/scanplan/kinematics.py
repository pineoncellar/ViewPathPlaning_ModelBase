"""Forward and inverse kinematics for the measured FANUC R-2000iC/125L on this cell.

Why a hand-written model instead of a DH table or a numerical solver. Three facts about
this machine are not the textbook ones, and each was measured rather than assumed:

* **J3 is the absolute pitch of the forearm**, not its angle relative to the upper arm.
  Fitting both hypotheses to the measured poses gives 0.004 mm for the absolute reading
  and 519.7 mm for the relative one - five orders of magnitude apart, so there is nothing
  to interpret.
* **The wrist centre sits 225 mm off the forearm axis.** The forearm is described by a
  length (`forearm_mm`) and a perpendicular offset (`forearm_offset_mm`), which is exactly
  the 7.411 deg tilt the attitude fit needed and missed on its first two attempts.
* **The tool is long and folded back** (|v| = 635.2 mm, frame W 179.3 / P -45.9 / R 172.8).
  That vector moves the wrist centre, hence shifts |J5|, which is the joint that actually
  over-travels. The 2026-10-23 pose bank moved this and nothing else - the arm numbers came
  back identical, which is what makes a tool change separable from a model error.

So the model is written in the machine's own quantities (`j3_sign`, `j3_zero_deg`, the
measured tool pose) and not translated into DH terms that would hide all three.

Solvability. Because J3 is absolute, the inverse splits into two independent halves: the
arm (J1..J3) has a closed-form two-link solution for each of four branch combinations, and
the wrist (J4..J6) has a closed form once the arm is fixed. That matters more than speed:
an analytic solve always returns every branch, whereas a damped least squares solve needs
several starts and returns whichever one it happened to converge to - which is precisely
the failure this module exists to remove.

Verification is not optional. Every candidate is put back through :meth:`Robot.forward`
and dropped when the pose does not reproduce. A sign convention that is wrong in one axis
produces solutions that look plausible and are not; the round trip is what makes the
difference visible in a test instead of on the cell.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import RobotParams, usable_limits

# A solution is kept only when forward() reproduces the request this closely. The model
# itself reproduces measured poses to 0.02 mm, so anything worse is a wrong root.
_VERIFY_POSITION_MM = 0.5
_VERIFY_ANGLE_DEG = 0.05
# Below this |sin J5| the wrist is singular: J4 and J6 are collinear and only their sum is
# determined.
_SINGULAR_SIN = 1e-6
# How far outside a limit a computed angle may be and still count as "on it". The measured
# input poses are rounded to 0.001 mm and the model reproduces them to 0.02 mm, so a pose
# captured exactly on a clamp comes back a few thousandths of a degree outside it. That is
# input rounding, not an unreachable point - and the distinction matters, because "dropped
# by the limit" and "dropped as unreachable" are different verdicts in the report. Ten
# thousandths of a degree is two orders of magnitude below `joint_margin_deg`, so this can
# never make an infeasible pose look feasible; it only decides which branch gets reported.
_WRAP_TOL_DEG = 0.01
_EPS = 1e-9


def _rx(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rz(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ypr_rotation(ypr) -> np.ndarray:
    """FANUC fixed-axis XYZ: R = Rz(R) * Ry(P) * Rx(W)."""
    w, p, r = (np.deg2rad(float(value)) for value in ypr)
    return _rz(r) @ _ry(p) @ _rx(w)


def rotation_angle_deg(first, second) -> float:
    """The angle of the rotation taking `first` to `second`, in degrees."""
    relative = np.asarray(first, dtype=float).T @ np.asarray(second, dtype=float)
    return float(np.degrees(np.arccos(np.clip((float(np.trace(relative)) - 1.0) / 2.0,
                                             -1.0, 1.0))))


def _unit(vector) -> np.ndarray:
    array = np.asarray(vector, dtype=float).reshape(3)
    length = float(np.linalg.norm(array))
    if not np.isfinite(length) or length <= _EPS:
        raise ValueError("direction has no length")
    return array / length


def slerp_axis(first, second, fraction: float) -> np.ndarray:
    """The optical axis `fraction` of the way from `first` towards `second`.

    A great-circle interpolation, which is what a controller holds when it turns a tool
    from one attitude to another along a straight line: the axis sweeps the shortest arc.
    The antiparallel case has no shortest arc - every perpendicular is equal - so a fixed
    perpendicular is chosen and the axis walks through it, and the caller is guaranteed a
    well-defined 1-degree-per-degree turn instead of a silent zero-length step.
    """
    start, end = _unit(first), _unit(second)
    cosine = float(np.clip(np.dot(start, end), -1.0, 1.0))
    angle = float(np.arccos(cosine))
    share = float(np.clip(fraction, 0.0, 1.0))
    if angle < 1e-9:
        return start
    if np.pi - angle < 1e-9:
        helper = np.array([1.0, 0.0, 0.0]) if abs(start[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        perpendicular = _unit(np.cross(start, helper))
        return _unit(np.cos(np.pi * share) * start + np.sin(np.pi * share) * perpendicular)
    sine = float(np.sin(angle))
    return _unit((np.sin((1.0 - share) * angle) * start + np.sin(share * angle) * end)
                 / sine)


def interpolated_pose(first, second, fraction: float):
    """One point along an `L` move between two scanner poses.

    Position is linear (that is what `L` means), the axis follows :func:`slerp_axis`, and
    the roll is interpolated as an angle. `first` and `second` may be anything carrying
    `position`, `axis` and `roll_deg` - a `ScannerPose`, a `PathPoint` or a viewpoint - so
    the same walk serves the planner and the report.
    """
    share = float(np.clip(fraction, 0.0, 1.0))
    start = np.asarray(first.position, dtype=float).reshape(3)
    end = np.asarray(second.position, dtype=float).reshape(3)
    roll_start = float(first.roll_deg)
    roll_end = float(second.roll_deg)
    # shortest way round: a 350 degree roll difference is a 10 degree turn
    delta_roll = (roll_end - roll_start + 180.0) % 360.0 - 180.0
    return (start + (end - start) * share,
            slerp_axis(first.axis, second.axis, share),
            roll_start + delta_roll * share)


def pose_walk(first, second, *, step_mm: float, step_deg: float):
    """The intermediate poses of an `L` move, destination included, origin excluded.

    Sampled on BOTH steps and the tighter one wins: an in-place re-orientation covers no
    millimetres at all, so a length-only rule would sample it once - at the far end - and
    report a move legal that spent its middle outside the envelope. The number of samples
    is capped by whichever step is coarser, so a 900 mm flight costs the same whether it
    turns or not.
    """
    if step_mm <= 0.0 or step_deg <= 0.0:
        raise ValueError("step_mm and step_deg must be positive")
    length = float(np.linalg.norm(np.asarray(second.position, dtype=float)
                                  - np.asarray(first.position, dtype=float)))
    swing = float(np.degrees(np.arccos(np.clip(
        float(np.dot(_unit(first.axis), _unit(second.axis))), -1.0, 1.0))))
    roll = abs((float(second.roll_deg) - float(first.roll_deg) + 180.0) % 360.0 - 180.0)
    count = max(1, int(np.ceil(length / step_mm)), int(np.ceil((swing + roll) / step_deg)))
    for index in range(1, count + 1):
        fraction = index / count
        yield (fraction, *interpolated_pose(first, second, fraction))


@dataclass(frozen=True)
class MoveVerdict:
    """What checking one whole MOVE said, as opposed to checking its two corners.

    `reason` is `ok`, `off` (gate disabled), `unreachable` (no solution at some point along
    the way) or `joint_limit` (a solution exists but sits outside the deliverable envelope).
    `at_fraction` says where along the move that happened, which is what turns "this move
    is refused" into something a person can look at.
    """

    ok: bool
    reason: str
    at_fraction: float | None = None
    joints_deg: tuple[float, ...] | None = None


def turn_digits(value_deg: float) -> int:
    """Whole revolutions past the axis's own travel; 0 for anything inside +-360.

    This was `floor(j / 360)`, which wrote `-1` for every negative joint - and a negative
    turn digit is not a legal FANUC CONFIG. ROBOGUIDE stops at that column on import with
    ASBN-009 `on line ..., column ... Configuration error` and refuses the WHOLE program,
    even when every joint is well inside its limits: on 2026-10-27 a delivery whose J1 was
    -102 deg carried `N U F, -1, -1, -1` and never loaded.

    A turn is a revolution BEYOND the axis's own range (J1 +-185, J4/J6 +-360 here), not
    the sign of the value. Every usable joint value lives inside that range, so the digit
    is 0; a value outside it is one (or more) turns and is flagged as negative here so the
    exporter can refuse rather than write a program whose CONFIG the controller cannot read.
    """
    if abs(float(value_deg)) <= 360.0:
        return 0
    return int(np.sign(value_deg) * np.floor(abs(float(value_deg)) / 360.0 + 0.5))


@dataclass(frozen=True)
class Solution:
    """One complete joint solution for a Cartesian pose, with the numbers that judge it.

    `config` is a HINT, not a verdict: the letters were read off the pendant on one bank of
    eight poses (2026-10-23), where all eight said `T` while J5 was negative and the first
    letter matched the shoulder on seven of the eight. What that bank CANNOT settle is the
    elbow letter (no `D` sample) and the turn digits. So the caller must not treat a
    mismatch as an error. `singular` says the wrist was at J4/J6 collinearity, where the
    two angles are only known as a sum.
    """

    joints_deg: tuple[float, ...]
    shoulder: str
    elbow: str
    wrist_flip: str
    config: str
    j5_margin_deg: float
    min_margin_deg: float
    ok: bool
    singular: bool = False

    @property
    def j5_abs_deg(self) -> float:
        return abs(float(self.joints_deg[4]))


class Robot:
    """The arm as configured: geometry, two limit sets, and the tool pose."""

    def __init__(self, params: RobotParams):
        self.params = params
        # Not the datasheet envelope and not the controller's set either: the intersection
        # shrunk to the central `limits_usage_frac` of every axis range. Every `wraps`,
        # `margins`, `j5_margin` and `feasible` answer below is therefore relative to the
        # envelope the site actually allows the planner to use, not to the machine's stops.
        self.limits = usable_limits(params.mechanical_limits_deg, params.soft_limits_deg,
                                    params.limits_usage_frac)
        self.tool_vector = np.asarray(params.tool_xyz_mm, dtype=float).reshape(3)
        self.tool_rotation = _ypr_rotation(params.tool_ypr_deg)
        # Effective forearm: the offset turns the forearm into a slightly longer link that
        # starts at an angle delta off the nominal one.
        self.forearm_effective_mm = float(np.hypot(params.forearm_mm,
                                                   params.forearm_offset_mm))
        self.forearm_tilt_rad = float(np.arctan2(params.forearm_offset_mm,
                                                 params.forearm_mm))
        self.limits_source = self._limits_source()

    def _limits_source(self) -> str:
        """Which envelope is in force, in words a report can print.

        The share belongs in the string: "the datasheet is in force" would be a false
        claim once 10 % of the travel is withheld at each end, and the report is read by
        people deciding whether a delivered path may be commanded.
        """
        sharp = any(soft != mech for soft, mech in zip(self.params.soft_limits_deg,
                                                       self.params.mechanical_limits_deg,
                                                       strict=True))
        base = "soft (intersected with mechanical)" if sharp else "mechanical"
        fraction = float(self.params.limits_usage_frac)
        margin = float(self.params.joint_margin_deg)
        notes: list[str] = []
        if fraction < 1.0:
            notes.append(f"central {fraction * 100:.0f}% of each axis range")
        if margin > 0.0:
            notes.append(f"{margin:g} deg reserved at each end")
        return f"{base}, {', '.join(notes)}" if notes else base

    # -- forward -------------------------------------------------------------

    def forward(self, joints) -> tuple[np.ndarray, np.ndarray]:
        """TCP position and TCP rotation for `joints` (degrees, controller convention).

        The returned rotation is the SCANNER frame: the flange rotation followed by the
        measured tool rotation, so its third column is the optical axis - the same
        convention `ScannerPose.axis` uses.
        """
        angles = self._forward_angles(joints)
        a1x, a2, b = angles["a1x"], angles["a2"], angles["b"]
        flange = self._flange_rotation(angles)
        params = self.params
        radius = (params.base_offset_mm
                  + params.upper_arm_mm * np.sin(a2)
                  + params.forearm_mm * np.sin(b)
                  - params.forearm_offset_mm * np.cos(b))
        height = (params.j2_height_mm
                  + params.upper_arm_mm * np.cos(a2)
                  + params.forearm_mm * np.cos(b)
                  + params.forearm_offset_mm * np.sin(b))
        wrist = np.array([radius * np.cos(a1x), radius * np.sin(a1x), height])
        frame = flange @ self.tool_rotation
        # The tool vector is expressed in the SCANNER frame, not the flange frame: the
        # measured six-pose fit is reproduced only when the tool rotation is applied before
        # the offset. Applying it the other way round shifts every point by |Rt*v - v|,
        # which is ~550 mm - a constant that hides in a residual report and looks like a
        # geometry error.
        return wrist + frame @ self.tool_vector, frame

    def wrist_centre(self, joints) -> np.ndarray:
        """The flange origin - the point the arm solution is written about."""
        angles = self._forward_angles(joints)
        a1x, a2, b = angles["a1x"], angles["a2"], angles["b"]
        params = self.params
        radius = (params.base_offset_mm
                  + params.upper_arm_mm * np.sin(a2)
                  + params.forearm_mm * np.sin(b)
                  - params.forearm_offset_mm * np.cos(b))
        height = (params.j2_height_mm
                  + params.upper_arm_mm * np.cos(a2)
                  + params.forearm_mm * np.cos(b)
                  + params.forearm_offset_mm * np.sin(b))
        return np.array([radius * np.cos(a1x), radius * np.sin(a1x), height])

    def _forward_angles(self, joints) -> dict[str, float]:
        params = self.params
        values = np.asarray(joints, dtype=float).reshape(6)
        return {
            "a1x": np.deg2rad(params.j1_zero_deg + params.j1_sign * values[0]),
            "a2": np.deg2rad(params.j2_zero_deg + params.j2_sign * values[1]),
            "b": np.deg2rad(90.0 + params.j3_sign * (values[2] - params.j3_zero_deg)),
            "z4": np.deg2rad(params.j4_zero_deg + params.j4_sign * values[3]),
            "z5": np.deg2rad(params.j5_zero_deg + params.j5_sign * values[4]),
            "z6": np.deg2rad(params.j6_zero_deg + params.j6_sign * values[5]),
        }

    def _flange_rotation(self, angles: dict[str, float]) -> np.ndarray:
        first, second = self.params.wrist_tilt_deg
        return (_rz(angles["a1x"]) @ _ry(angles["b"])
                @ _rz(angles["z4"]) @ _rx(np.deg2rad(first))
                @ _rz(angles["z5"]) @ _rx(np.deg2rad(second))
                @ _rz(angles["z6"]))

    # -- limits --------------------------------------------------------------

    def wraps(self, axis: int, value_deg: float) -> list[float]:
        """Every equivalent reading of `value_deg` inside axis `axis`'s limits.

        Only J1, J4 and J6 span a full turn, and for those the same physical pose has
        several readings. Ignoring that would reject a perfectly good solution whose
        computed angle happens to be 370 degrees.
        """
        low, high = self.limits[axis]
        found = [value_deg + 360.0 * turn for turn in (-2, -1, 0, 1, 2)]
        return [value for value in found
                if low - _WRAP_TOL_DEG <= value <= high + _WRAP_TOL_DEG]

    def margins(self, joints) -> np.ndarray:
        """Degrees from each axis to its nearest limit, taking the roomiest reading.

        A reading outside the limits yields -inf, not a negative distance: "0.1 deg past
        the limit" and "0.1 deg inside" must not compare as neighbours.
        """
        values = np.asarray(joints, dtype=float).reshape(6)
        out = np.full(6, -np.inf)
        for axis in range(6):
            low, high = self.limits[axis]
            best = -np.inf
            for candidate in self.wraps(axis, float(values[axis])):
                best = max(best, min(candidate - low, high - candidate))
            out[axis] = best
        return out

    def j5_margin(self, joints) -> float:
        """Room left on J5 inside the usable band, in degrees; negative means unusable.

        The band is not `[-limit, +limit]`: `|J5| -> 0` is the wrist singularity, so the
        centre of the mechanical range is the worst place to be. With +-125 limits the
        optimum sits near |J5| = 64, not 0, and this is the number that says so.
        """
        value = float(np.asarray(joints, dtype=float).reshape(6)[4])
        low, high = self.limits[4]
        margin = float(self.params.joint_margin_deg)
        reachable = (high if value >= 0.0 else -low) - margin
        return float(min(abs(value) - self.params.j5_min_abs_deg, reachable - abs(value)))

    def elbow_bend_deg(self, joints) -> float:
        """The angle the two links make, in degrees: 0 is folded flat, 180 is straight.

        Recomputed from the joint values rather than carried along, because the value that
        matters for a delivered point is the one the CONTROLLER will be at - not the one the
        solver happened to work in. `bend -> 0` and `bend -> 180` are the two collinear
        postures where the closed form is most sensitive to a millimetre of tool error and
        where a real controller reports MOTN-018 instead of solving.
        """
        values = np.asarray(joints, dtype=float).reshape(6)
        angles = self._forward_angles(values)
        a2, b = angles["a2"], angles["b"]
        params = self.params
        radius = (params.base_offset_mm + params.upper_arm_mm * np.sin(a2)
                  + params.forearm_mm * np.sin(b) - params.forearm_offset_mm * np.cos(b))
        height = (params.j2_height_mm + params.upper_arm_mm * np.cos(a2)
                  + params.forearm_mm * np.cos(b) + params.forearm_offset_mm * np.sin(b))
        links = (params.upper_arm_mm, self.forearm_effective_mm)
        dr = radius - params.base_offset_mm
        cosine = ((dr * dr + height * height - links[0] ** 2 - links[1] ** 2)
                  / (2.0 * links[0] * links[1]))
        return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

    def feasible(self, joints, margin_deg: float | None = None) -> bool:
        """Whether this joint set is inside the limits with margin and away from J5 = 0.

        Also refuses a collinear elbow (`elbow_margin_deg`): the export audit goes through
        here, so a point an older plan accepted at `bend = 8 deg` is caught before a program
        is written rather than on the pendant.
        """
        required = (self.params.joint_margin_deg if margin_deg is None else float(margin_deg))
        elbow = float(getattr(self.params, "elbow_margin_deg", 0.0))
        if elbow > 0.0:
            bend = self.elbow_bend_deg(joints)
            if bend < elbow or bend > 180.0 - elbow:
                return False
        return bool(self.margins(joints).min() >= required and self.j5_margin(joints) >= 0.0)

    # -- inverse -------------------------------------------------------------

    def solutions(self, position, rotation, *, verify: bool = True) -> list[Solution]:
        """Every joint solution for a TCP pose, best J5 margin first.

        At most eight: two shoulder sides times two elbow configurations times two wrist
        flips. `wrist_flip` changes only the SIGN of J5, which is why changing the T letter
        in a CONFIG string never fixes an over-travel - the magnitude is set by the arm.
        """
        point = np.asarray(position, dtype=float).reshape(3)
        target = np.asarray(rotation, dtype=float).reshape(3, 3)
        flange = target @ self.tool_rotation.T
        wrist = point - target @ self.tool_vector
        found: list[Solution] = []
        allowed = {str(family).upper().strip().replace("  ", " ")
                   for family in getattr(self.params, "allowed_configs", ()) or ()}
        # A pinned wrist letter is a CONSTRAINT on which branch is acceptable, not just a
        # label to print. Writing `N U T` over a solution whose J5 is positive asks the
        # controller for the other wrist branch - J4 and J6 turn 180 deg, and where that
        # branch is outside the travel the controller stops with MOTN-018 "position not
        # reachable". The 2026-10-29 PART010 program had 13 of its 27 points like that.
        pinned = str(getattr(self.params, "config_wrist_letter", "auto")).upper()
        elbow_margin = float(getattr(self.params, "elbow_margin_deg", 0.0))
        for shoulder, elbow, arm in self._arm_solutions(wrist):
            a1x, a2, b = arm["a1x"], arm["a2"], arm["b"]
            for flip, wrist_angles, singular in self._wrist_solutions(a1x, b, flange):
                if pinned in ("T", "F"):
                    # `T` is the negative-J5 branch by construction (see `_wrist_solutions`).
                    if (pinned == "T") != (flip == "T"):
                        continue
                joints = self._controller_angles(a1x, a2, b, *wrist_angles)
                placed = self._place(joints)
                if placed is None:
                    continue
                if verify and not self._reproduces(placed, point, target):
                    continue
                if elbow_margin > 0.0:
                    bend = self.elbow_bend_deg(placed)
                    if bend < elbow_margin or bend > 180.0 - elbow_margin:
                        # Collinear links: the posture the fitted geometry describes worst,
                        # and the one the controller refuses rather than solving.
                        continue
                config = self.config_string(placed, shoulder, elbow)
                if allowed and config.split(",")[0].strip().upper() not in allowed:
                    # A family this cell has not proven. The front/back posture numbers J2
                    # around -40 here and around +143 on the pendant, so the limits this
                    # config carries cannot judge it - and a point that slips through stops
                    # the program at MOTN-017 instead of being planned around.
                    continue
                margins = self.margins(placed)
                found.append(Solution(
                    joints_deg=tuple(float(value) for value in placed),
                    shoulder=shoulder, elbow=elbow, wrist_flip=flip,
                    config=config,
                    j5_margin_deg=self.j5_margin(placed),
                    min_margin_deg=float(margins.min()),
                    ok=bool(margins.min() >= self.params.joint_margin_deg
                            and self.j5_margin(placed) >= 0.0),
                    singular=singular,
                ))
        found.sort(key=lambda item: item.j5_margin_deg, reverse=True)
        return found

    def _controller_angles(self, a1x: float, a2: float, b: float,
                           z4: float, z5: float, z6: float) -> tuple[float, ...]:
        """Model angles back to the numbers the controller reports for J1..J6."""
        params = self.params
        return (
            (float(np.rad2deg(a1x)) - params.j1_zero_deg) / params.j1_sign,
            (float(np.rad2deg(a2)) - params.j2_zero_deg) / params.j2_sign,
            params.j3_zero_deg + (float(np.rad2deg(b)) - 90.0) / params.j3_sign,
            (float(np.rad2deg(z4)) - params.j4_zero_deg) / params.j4_sign,
            (float(np.rad2deg(z5)) - params.j5_zero_deg) / params.j5_sign,
            (float(np.rad2deg(z6)) - params.j6_zero_deg) / params.j6_sign,
        )

    def _place(self, joints) -> tuple[float, ...] | None:
        """Snap each axis to the roomiest equivalent reading, or None if none fits."""
        values = np.asarray(joints, dtype=float).reshape(6)
        placed = []
        for axis in range(6):
            candidates = self.wraps(axis, float(values[axis]))
            if not candidates:
                return None
            low, high = self.limits[axis]
            placed.append(max(candidates, key=lambda v: min(v - low, high - v)))
        return tuple(float(value) for value in placed)

    def _arm_solutions(self, wrist_centre_mm):
        """The four arm branches, analytic, no iteration and no seed."""
        params = self.params
        x, y, z = (float(value) for value in np.asarray(wrist_centre_mm, dtype=float))
        rho = float(np.hypot(x, y))
        bearing = float(np.arctan2(y, x))
        links = (params.upper_arm_mm, self.forearm_effective_mm)
        for shoulder, a1x, reach in (("N", bearing, rho),
                                     ("F", bearing + np.pi, -rho)):
            dr = reach - params.base_offset_mm
            dz = z - params.j2_height_mm
            cosine = ((dr * dr + dz * dz - links[0] ** 2 - links[1] ** 2)
                      / (2.0 * links[0] * links[1]))
            if not -1.0 - 1e-9 <= cosine <= 1.0 + 1e-9:
                continue
            bend = float(np.arccos(np.clip(cosine, -1.0, 1.0)))
            for elbow, psi in (("U", bend), ("D", -bend)):
                a2 = (float(np.arctan2(dr, dz))
                      - float(np.arctan2(links[1] * np.sin(psi),
                                         links[0] + links[1] * np.cos(psi))))
                forearm_pitch = a2 + psi + self.forearm_tilt_rad
                yield shoulder, elbow, {"a1x": a1x, "a2": a2, "b": forearm_pitch}

    def _wrist_solutions(self, a1x: float, b: float, flange: np.ndarray):
        """The two wrist branches for a fixed arm, as (T/F, (z4, z5, z6), singular).

        `T` is the negative-J5 branch, matching what the pendant calls `T` - see
        `config_string` for the measurement that settled it.
        """
        tilt = float(self.params.wrist_tilt_deg[0])
        if abs(abs(tilt) - 90.0) > 1e-6 or abs(self.params.wrist_tilt_deg[1] - tilt) > 1e-6:
            raise ValueError(
                "the analytic wrist solve assumes both fixed tilts are +-90 deg and equal; "
                f"this configuration has {self.params.wrist_tilt_deg!r}. The design keeps a "
                "numerical fallback for other geometries, but it is not implemented - a "
                "silently wrong wrist would produce joint values that look plausible")
        inner = (_rz(a1x) @ _ry(b)).T @ flange
        cosine = float(np.clip(-inner[2, 2], -1.0, 1.0))
        magnitude = float(np.arccos(cosine))
        for flip, z5 in (("F", magnitude), ("T", -magnitude)):
            sine = float(np.sin(z5))
            if abs(sine) < _SINGULAR_SIN:
                # J4 and J6 are collinear: only z4 + z6 is defined. Put the whole turn on
                # J6 and say so, rather than inventing a split.
                angle = (float(np.arctan2(inner[1, 0], inner[0, 0])) if cosine > 0.0
                         else float(np.arctan2(-inner[0, 1], -inner[0, 0])))
                yield flip, (0.0, z5, -angle), True
                continue
            # Derived for a tilt of -90; a +90 tilt mirrors the off-diagonal signs.
            negative = (sine > 0.0) == (tilt < 0.0)
            if negative:
                z4 = float(np.arctan2(-inner[1, 2], -inner[0, 2]))
                z6 = float(np.arctan2(inner[2, 1], -inner[2, 0]))
            else:
                z4 = float(np.arctan2(inner[1, 2], inner[0, 2]))
                z6 = float(np.arctan2(-inner[2, 1], inner[2, 0]))
            yield flip, (z4, z5, z6), False

    def _reproduces(self, joints, position, rotation) -> bool:
        """Forward-check a candidate solution against the pose it came from."""
        point, frame = self.forward(joints)
        return (float(np.linalg.norm(point - np.asarray(position, dtype=float).reshape(3)))
                <= _VERIFY_POSITION_MM
                and rotation_angle_deg(frame, rotation) <= _VERIFY_ANGLE_DEG)

    def config_string(self, joints, shoulder: str, elbow: str) -> str:
        """The `.LS` CONFIG hint for one point, derived from the joint values themselves.

        Derived rather than guessed: the letters are read off the solution that was chosen,
        so a point whose best branch is `F D` is written as `F D`, not as the constant
        `N U T` that locked every point onto one branch in the first place. Note that the
        T letter only reflects the sign of J5; it cannot move |J5|.

        The T/F letter is `T` for NEGATIVE J5. That is the one letter the 2026-10-23 bank
        settles: eight poses with J5 between -9 and -124 were all labelled `T` on the
        pendant, so the previous rule (`T` for positive J5) got every one of them wrong.
        A wrong letter is not cosmetic - it tells the controller to take the other wrist
        branch, which is the same pose turned 180 degrees.
        """
        values = np.asarray(joints, dtype=float).reshape(6)
        turns = [turn_digits(float(values[index])) for index in (0, 3, 5)]
        pinned = str(getattr(self.params, "config_wrist_letter", "auto")).upper()
        if pinned in ("T", "F"):
            # Pinned for the cell: its controller refuses the other letter outright, and
            # the two wrist branches describe the same TCP pose, so nothing is given up.
            flip = pinned
        else:
            flip = "T" if values[4] < 0.0 else "F"
        return f"{shoulder} {elbow} {flip}, {turns[0]}, {turns[1]}, {turns[2]}"

    def reproduces(self, joints, position, rotation) -> bool:
        """Public forward-check, for callers that solved without verifying every root."""
        return self._reproduces(joints, position, rotation)


@dataclass(frozen=True)
class Verdict:
    """What the kinematic gate decided about one pose, and why.

    `reason` is what the report groups by - "the plan lost these points to the joint
    limits" and "the plan lost them because the arm cannot reach that far" are different
    problems with different fixes, and a single boolean cannot say which happened.
    """

    ok: bool
    reason: str
    solution: Solution | None = None


class KinematicGate:
    """Joint-limit feasibility for scanner poses, memoised.

    Memoised because the same pose is asked about several times: once per candidate, then
    again per check step along every stroke that crosses it, then again when the delivered
    path is assigned its branches. The IK is cheap, but not so cheap that repeating it
    hundreds of thousands of times inside a lambda budget loop is free.

    `enabled=false` makes `check` answer "fine" without touching any maths, which is how the
    whole feature stays invisible until the site has confirmed its limits.
    """

    def __init__(self, params: RobotParams):
        self.params = params
        self.robot = Robot(params) if params.enabled else None
        self._cache: dict[tuple, Verdict] = {}
        self._expand: dict[tuple, list[Solution]] = {}
        self._moves: dict[tuple, MoveVerdict] = {}
        self.queries = 0
        # Samples taken along interpolated moves, so "the move was checked" is a number in
        # the report rather than a claim about code that ran.
        self.move_samples = 0

    @property
    def active(self) -> bool:
        return self.robot is not None

    def check(self, position, axis, roll_deg: float) -> Verdict:
        """Whether the head may be at this pose at all."""
        if self.robot is None:
            return Verdict(True, "off")
        key = self._key(position, axis, roll_deg)
        found = self._cache.get(key)
        if found is not None:
            return found
        self.queries += 1
        found = self._solve(np.asarray(position, dtype=float),
                            np.asarray(axis, dtype=float), float(roll_deg))
        self._cache[key] = found
        return found

    def _solve(self, position, axis, roll_deg: float) -> Verdict:
        from .geometry import scanner_frame

        rotation = scanner_frame(axis, roll_deg)
        # Solve without verifying every root, then confirm only the winner: the analytic
        # model is exact for this geometry and the round-trip test pins it, so paying for
        # eight forward checks per pose buys nothing.
        candidates = self.robot.solutions(position, rotation, verify=False)
        for item in candidates:
            if item.ok and self.robot.reproduces(item.joints_deg, position, rotation):
                return Verdict(True, "ok", item)
        if not candidates:
            return Verdict(False, "unreachable", None)
        return Verdict(False, "joint_limit", candidates[0])

    def summary(self) -> dict:
        """Cache statistics, so "the gate was on" is a measurement rather than a claim."""
        return {"poses_asked": int(self.queries), "distinct_poses": len(self._cache),
                "moves_checked": len(self._moves),
                "move_samples": int(self.move_samples),
                "robot": self.params.model, "limits_source":
                (self.robot.limits_source if self.robot is not None else "gate disabled")}

    def solutions(self, position, axis, roll_deg: float) -> list[Solution]:
        """Every feasible solution at this pose, best J5 room first (cached).

        The gate itself only needs to know whether one exists; branch continuity needs the
        whole set, because choosing the locally best branch at every point is exactly the
        mistake that makes adjacent points jump across a branch.
        """
        if self.robot is None:
            return []
        cached = self._expand.get(self._key(position, axis, roll_deg))
        if cached is not None:
            return cached
        from .geometry import scanner_frame

        rotation = scanner_frame(np.asarray(axis, dtype=float), float(roll_deg))
        found = [item for item in self.robot.solutions(position, rotation)
                 if item.ok]
        self._expand[self._key(position, axis, roll_deg)] = found
        return found

    def follow(self, reference_joints, position, axis, roll_deg: float) -> Solution | None:
        """The feasible solution at this pose that stays closest to `reference_joints`.

        This is what a controller does: it does not pick a branch per point, it stays on the
        one it is already on. Judging each sample of a move on its own ("some solution
        exists here") would answer a different question from the one that matters, because
        the branch that exists halfway along may be one the arm has to jump to - which is a
        move the controller refuses.
        """
        if self.robot is None:
            return None
        from .geometry import scanner_frame

        rotation = scanner_frame(np.asarray(axis, dtype=float), float(roll_deg))
        options = [item for item in self.robot.solutions(position, rotation, verify=False)
                   if item.ok]
        options.sort(key=lambda item: joint_distance(item.joints_deg, reference_joints))
        for item in options:
            if self.robot.reproduces(item.joints_deg, position, rotation):
                return item
        return None

    def move(self, first, second, *, motion: str = "L") -> MoveVerdict:
        """Whether the WHOLE move between two poses stays inside the deliverable envelope.

        Two different questions, and the configured motion decides which one applies:

        * `J` - the controller interpolates the joint values. The deliverable envelope is a
          box, so a straight line between two inside points is inside everywhere: the two
          corners settle it, and sampling would only re-prove convexity. (J6 gets no such
          guarantee in the other direction: it is the axis a joint move spins, which is
          why `stroke.max_j6_step_deg` exists separately.)
        * `L` - the controller interpolates the TCP along a straight line and turns the tool
          along the shortest arc, so the joints in between are the IK of poses that were
          never checked. A reach that has to straighten, or a wrist that has to unwind,
          leaves the envelope halfway along while both ends sit comfortably inside it. Those
          are the moves this samples.

        The branch is followed from the start of the move (see :meth:`follow`), not chosen
        afresh at every sample.
        """
        if self.robot is None:
            return MoveVerdict(True, "off")
        motion = str(motion).upper()
        key = (self._key(first.position, first.axis, first.roll_deg),
               self._key(second.position, second.axis, second.roll_deg), motion,
               float(self.params.motion_check_step_mm),
               float(self.params.motion_check_step_deg))
        cached = self._moves.get(key)
        if cached is not None:
            return cached
        start = self.check(first.position, first.axis, first.roll_deg)
        if not start.ok:
            # Reported by the caller as an unreachable CORNER, which is the more specific
            # answer; the move itself cannot be judged from a start that does not exist.
            verdict = MoveVerdict(False, f"start_{start.reason}", 0.0)
            self._moves[key] = verdict
            return verdict
        if motion == "J":
            # A joint move interpolates each axis linearly, and a box is convex, so the two
            # corners settle the limits. What they do NOT settle is the wrist: J5 crossing
            # zero is the singularity, and a linear interpolation between +30 and -30 goes
            # straight through it. `feasible` keeps |J5| >= j5_min_abs at every point, so
            # the only way across is a sign change - refused here.
            end = self.check(second.position, second.axis, second.roll_deg)
            start_j5 = float(start.solution.joints_deg[4]) if start.solution else 0.0
            end_j5 = float(end.solution.joints_deg[4]) if end.solution else 0.0
            if start_j5 * end_j5 < 0.0:
                verdict = MoveVerdict(False, "wrist_flip", None,
                                      tuple(start.solution.joints_deg))
                self._moves[key] = verdict
                return verdict
            verdict = MoveVerdict(True, "ok", None,
                                  (start.solution.joints_deg if start.solution else None))
            self._moves[key] = verdict
            return verdict
        reference = start.solution.joints_deg if start.solution is not None else None
        if reference is None:
            verdict = MoveVerdict(False, "joint_limit", 0.0)
            self._moves[key] = verdict
            return verdict
        verdict = MoveVerdict(True, "ok", None, tuple(reference))
        for fraction, position, axis, roll in pose_walk(
                first, second, step_mm=float(self.params.motion_check_step_mm),
                step_deg=float(self.params.motion_check_step_deg)):
            self.move_samples += 1
            found = self.follow(reference, position, axis, roll)
            if found is None:
                verdict = MoveVerdict(False, "unreachable", float(fraction), tuple(reference))
                break
            reference = found.joints_deg
            verdict = MoveVerdict(True, "ok", float(fraction), tuple(reference))
        self._moves[key] = verdict
        return verdict

    @staticmethod
    def _key(position, axis, roll_deg: float) -> tuple:
        return (round(float(position[0]), 1), round(float(position[1]), 1),
                round(float(position[2]), 1), round(float(axis[0]), 4),
                round(float(axis[1]), 4), round(float(axis[2]), 4),
                round(float(roll_deg), 2))


def joint_distance(first, second) -> float:
    """The worst single-axis joint change between two joint vectors, the short way round."""
    delta = (np.asarray(second, dtype=float) - np.asarray(first, dtype=float))
    return float(np.max(np.abs((delta + 180.0) % 360.0 - 180.0)))


def joint_step(first, second, axis: int) -> float:
    """One axis' joint change between two joint vectors, degrees, shortest way round."""
    delta = float(np.asarray(second, dtype=float)[axis] - np.asarray(first, dtype=float)[axis])
    return abs((delta + 180.0) % 360.0 - 180.0)
