"""Where the arm can put the scanner, and how far from its stops it has to stay.

The planner emits Cartesian poses and leaves the joints to the controller. This module
puts part of that back into the loop: every pose and every straight move a plan delivers
is solved with the robot's own geometry, and anything that would need a joint past
`robot.joint_limit_margin` - 0.9 in the shipped configuration, i.e. the outermost 10% of
every axis' travel is off limits - is rejected BEFORE the selection. The coverage figure
then describes a set the arm can fly, instead of one ROBOGUIDE may hand back.

What is modelled
----------------
* forward and inverse kinematics of the R-2000iC arm;
* the joint limits of the robot's WORKSPACE TABLE, derated by `joint_limit_margin`;
* the robot base pose in the station frame (`robot.base_position_mm`, `base_rpy_deg`);
* the flange-to-TCP distance along the optical axis (`robot.tool_length_mm`) and the
  bracket's rotation about that axis (`robot.tool_mount_deg`).

What is NOT modelled, and why that is not simply a gap to plug later
-------------------------------------------------------------------
* **Collision.** A pose this module accepts can still run the arm into the fixture. That
  is `stroke`/`station` clearance work, and it stays ROBOGUIDE's to confirm;
* **the interior of a joint travel move.** Both of its ends are checked, and because the
  legal joint window is a box - a convex set - a joint-space interpolation between two
  legal ends stays legal. The same argument does NOT hold for a straight (`L`) move, which
  interpolates in Cartesian space, so those are sampled at `stroke.check_step_mm`;
* **singular regions and joint rates**, and the arm's own occlusion of the tracker.

Where the numbers come from
---------------------------
The geometry is the CAD-derived chain of `ros-industrial/fanuc`,
`fanuc_r2000ic_support/urdf/r2000ic210f_macro.xacro` (Apache-2.0; FANUC's 165F and 210F
share the arm, which is why that file's meshes are named `r2000ic165f`). Cross-check,
because a kinematic model that is subtly wrong looks exactly like a working one:
`shoulder_offset_mm + upper_arm_mm + hypot(elbow_offset_mm, forearm_mm)` is 2687 mm against
the datasheet's 2655 mm reach, the 32 mm difference being the datasheet measuring to the
flange. `tests/test_robot.py` asserts that, and the forward/inverse round trip.

Two conventions meet here and must not be mixed
-----------------------------------------------
* the MODEL convention of that CAD chain - its zero pose and its sign per joint;
* the WORKSPACE TABLE convention of the robot datasheet, which is what a workshop reads and
  what `robot.joint_limits_deg` is written in.

For this arm they are the same range under a different zero: J2 is -90..+46 in the table
and -60..+76 in the model, both 136 deg wide, and J3 is -122..+190 against -132..+180, both
312 deg. So `RobotSpec.joint_datum_deg` converts one to the other exactly, at both ends and
in between, and either could be used for the verdict. The check runs in the MODEL's
convention, because that is where the kinematics live, and reports `utilization` - the
share of the allowed half-travel a joint uses - which is invariant under that conversion.
A mirrored datum would fit the same two tables, and would relabel the joints without moving
a single accepted pose: the verdict does not depend on which of the two it is.

The wrist is not spherical
--------------------------
J4 and J5 intersect, but J6 sits `wrist_offset_mm` (215 mm) away along the tool axis, so the
usual "place the wrist centre, then solve a spherical wrist" shortcut needs the offset put
back in. It has a closed form: with `D = wrist_offset_mm + tool_length_mm`,

    wrist_centre = TCP - D * (optical axis)

because the tool's optical axis IS the J6 rotation axis and the TCP sits on it. That identity
is what `solve` starts from, and `tests/test_robot.py` asserts it against the forward chain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as _Rotation

from .config import RobotParams

JOINT_COUNT = 6
# Where the geometry above comes from, quoted in the report so a number can be traced.
SOURCE = ("ros-industrial/fanuc fanuc_r2000ic_support/urdf/r2000ic210f_macro.xacro "
          "(Apache-2.0); joint limits from the R-2000iC floor-model workspace table")
# How far INSIDE the window a solution must be. The rule is a planning margin, not a
# measurement, so a solution exactly on the derated bound is accepted.
_INSIDE_TOL = 1e-9
# Below this the J5 axis is (nearly) aligned with the tool axis: J4 and J6 then carry only
# their sum, and the split between them is free. Picking it here keeps it continuous.
_SINGULAR_SIN = 1e-9


def _xyz(x: float, y: float, z: float) -> np.ndarray:
    return np.array([x, y, z], dtype=float)


def _pairs(values) -> list[list[float]]:
    """A `(6, 2)` window as the rounded nested lists the report writes out."""
    return [[round(float(low), 3), round(float(high), 3)] for low, high in values]


def _rot_x(angle: float) -> np.ndarray:
    cos, sin = float(np.cos(angle)), float(np.sin(angle))
    return np.array([[1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos]])


def _rot_y(angle: float) -> np.ndarray:
    cos, sin = float(np.cos(angle)), float(np.sin(angle))
    return np.array([[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]])


def _rot_z(angle: float) -> np.ndarray:
    cos, sin = float(np.cos(angle)), float(np.sin(angle))
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


def _rotation_from_rpy(rpy_deg) -> np.ndarray:
    """A frame from yaw-pitch-roll degrees about the STATION axes, applied X then Y then Z."""
    roll, pitch, yaw = (float(np.deg2rad(value)) for value in rpy_deg)
    return _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)


@dataclass(frozen=True)
class RobotSpec:
    """One robot's kinematic chain, in the CAD model's own convention.

    The chain, from the mounting plane upwards, where every offset is a pure translation
    applied before that joint's rotation (the URDF's own structure):

        base --d1 up--> J1 (about Z) --a1 out--> J2 (about Y) --a2 up--> J3 (about -Y)
             --elbow_offset up--> J4 (about X) --forearm out--> J5 (about -Y)
             --wrist_offset out--> J6 (about X)

    `cad_limits_deg` and `table_limits_deg` are the same physical stops written in the two
    conventions above; their widths must agree, which is what makes the datum a pure
    offset and `joint_datum_deg` exact.
    """

    name: str
    shoulder_height_mm: float
    shoulder_offset_mm: float
    upper_arm_mm: float
    elbow_offset_mm: float
    forearm_mm: float
    wrist_offset_mm: float
    reach_mm: float
    cad_limits_deg: tuple
    table_limits_deg: tuple
    source: str = SOURCE

    @property
    def joint_datum_deg(self) -> tuple[float, ...]:
        """The CAD model's joint zero minus the workspace table's, per joint."""
        return tuple(
            float(cad_lo - table_lo)
            for (cad_lo, _cad_hi), (table_lo, _table_hi)
            in zip(self.cad_limits_deg, self.table_limits_deg, strict=True)
        )

    @property
    def arm_reach_mm(self) -> float:
        """How far the wrist centre can get from the J1 axis, from this chain alone.

        `shoulder_offset + upper_arm + hypot(elbow_offset, forearm)`: the last two are one
        rigid vector of constant length that the elbow can line up with the upper arm. That
        is the number to compare with `reach_mm`.
        """
        return (float(self.shoulder_offset_mm) + float(self.upper_arm_mm)
                + float(np.hypot(self.elbow_offset_mm, self.forearm_mm)))


# Floor-model R-2000iC/165F and /210F: the same arm (FANUC's own CAD files say so - the
# 210F model carries the 165F meshes) and, per the workspace table, the same stops. Only
# the payload differs, and payload is not a kinematic quantity.
_R2000IC_FLOOR = {
    "shoulder_height_mm": 670.0,
    "shoulder_offset_mm": 312.0,
    "upper_arm_mm": 1075.0,
    "elbow_offset_mm": 225.0,
    "forearm_mm": 1280.0,
    "wrist_offset_mm": 215.0,
    "cad_limits_deg": ((-185.0, 185.0), (-60.0, 76.0), (-132.0, 180.0),
                       (-360.0, 360.0), (-125.0, 125.0), (-360.0, 360.0)),
    "table_limits_deg": ((-185.0, 185.0), (-90.0, 46.0), (-122.0, 190.0),
                         (-360.0, 360.0), (-125.0, 125.0), (-360.0, 360.0)),
}
MODELS: dict[str, RobotSpec] = {
    "R-2000iC/210F": RobotSpec(name="R-2000iC/210F", reach_mm=2655.0, **_R2000IC_FLOOR),
    "R-2000iC/165F": RobotSpec(name="R-2000iC/165F", reach_mm=2655.0, **_R2000IC_FLOOR),
}


@dataclass(frozen=True)
class Reach:
    """One configuration that puts the TCP on the asked-for pose.

    `table_deg` is the same configuration in the workspace table's convention, which is the
    one a fitter reads off the controller. `utilization` is the largest share of a joint's
    allowed half-travel the configuration uses: 1.0 is the derated bound, and anything
    below it is inside.
    """

    joints_deg: np.ndarray
    table_deg: np.ndarray
    utilization: float
    worst_joint: int

    def describe(self) -> dict:
        """The record the report and the diagnostics print."""
        return {
            "table_deg": [round(float(value), 3) for value in self.table_deg],
            "utilization": round(float(self.utilization), 4),
            "worst_joint": f"J{int(self.worst_joint) + 1}",
        }


@dataclass(frozen=True)
class MoveCheck:
    """The verdict on one straight move, sampled along its length.

    `failed_at_mm` is where along the move the first pose without a legal configuration sat,
    which is what a reviewer needs to see whether it just clipped a limit or never had a
    chance. It is None when the move is fine.
    """

    ok: bool
    joints_deg: np.ndarray | None = None
    worst_utilization: float = 0.0
    worst_joint: int = 0
    failed_at_mm: float | None = None
    samples: int = 0


class RobotArm:
    """The robot, its derated joint windows, and the two questions the planner asks.

    Stateless between calls: a plan can ask about thousands of poses without any state
    machine, and the same pose always gets the same answer.
    """

    def __init__(self, params: RobotParams):
        try:
            self.spec = MODELS[str(params.model)]
        except KeyError as error:
            raise ValueError(
                f"unknown robot model {params.model!r}: the kinematic table has "
                f"{sorted(MODELS)}. Add the CAD chain of the arm AND its workspace table "
                "before using a model that is not listed, because both are needed and "
                "neither can be guessed from the other"
            ) from error
        self.params = params
        self.base_position_mm = np.asarray(params.base_position_mm, dtype=float).reshape(3)
        self.base_rotation = _rotation_from_rpy(params.base_rpy_deg)
        # The flange -> tool frame rotation. Its third column is the flange's X axis, i.e.
        # the J6 rotation axis, which is the physical claim the whole model rests on: the
        # scanner's optical axis IS the axis J6 turns about, so the roll the planner picks
        # is a rotation of J6 and nothing else.
        self._mount = _rot_y(np.pi / 2.0) @ _rot_z(np.deg2rad(float(params.tool_mount_deg)))

        datum = np.asarray(self.spec.joint_datum_deg, dtype=float)
        # The window is derated in the WORKSPACE TABLE's convention, which is where the
        # 90% rule is written and where a fitter checks it, and only then converted into the
        # one the kinematics use. Derating the CAD-convention stops instead would pull J2's
        # two ends in by different amounts than the table says (-54/68.4 against -51/71.4),
        # which is a different robot, not a different rounding.
        margin = float(params.joint_limit_margin)
        self.table_limits_deg = np.asarray(params.joint_limits_deg, dtype=float).reshape(
            JOINT_COUNT, 2)
        self.allowed_table_deg = self.table_limits_deg * margin
        self.limits_deg = self.table_limits_deg + datum[:, None]
        self.allowed_deg = self.allowed_table_deg + datum[:, None]
        self._centre_deg = self.allowed_deg.mean(axis=1)
        self._half_deg = np.maximum(
            (self.allowed_deg[:, 1] - self.allowed_deg[:, 0]) / 2.0, 1e-9)
        self.wrist_to_tcp_mm = (float(self.spec.wrist_offset_mm)
                                + float(params.tool_length_mm))

    # -- conversions ---------------------------------------------------------

    def table_deg(self, joints_deg) -> np.ndarray:
        """A model-space configuration in the workspace table's convention."""
        joints = np.asarray(joints_deg, dtype=float).reshape(JOINT_COUNT)
        return joints - np.asarray(self.spec.joint_datum_deg, dtype=float)

    def utilization(self, joints_deg) -> np.ndarray:
        """Per joint, the share of its allowed half-travel that the value uses."""
        joints = np.asarray(joints_deg, dtype=float).reshape(JOINT_COUNT)
        return np.abs(joints - self._centre_deg) / self._half_deg

    # -- kinematics ----------------------------------------------------------

    def forward(self, joints_deg) -> tuple[np.ndarray, np.ndarray]:
        """`(frame, position)` of the TCP for a configuration, in the station frame.

        The frame's third column is the optical axis, exactly as
        :func:`scanplan.geometry.scanner_frame` builds it, so the two are interchangeable
        wherever a pose is passed around.
        """
        spec = self.spec
        q = np.deg2rad(np.asarray(joints_deg, dtype=float).reshape(JOINT_COUNT))
        rotation = _rot_z(q[0])
        offset = _xyz(0.0, 0.0, spec.shoulder_height_mm)
        offset = offset + rotation @ _xyz(spec.shoulder_offset_mm, 0.0, 0.0)
        rotation = rotation @ _rot_y(q[1])
        offset = offset + rotation @ _xyz(0.0, 0.0, spec.upper_arm_mm)
        rotation = rotation @ _rot_y(-q[2])
        offset = offset + rotation @ _xyz(0.0, 0.0, spec.elbow_offset_mm)
        rotation = rotation @ _rot_x(-q[3])
        offset = offset + rotation @ _xyz(spec.forearm_mm, 0.0, 0.0)
        rotation = rotation @ _rot_y(-q[4])
        offset = offset + rotation @ _xyz(spec.wrist_offset_mm, 0.0, 0.0)
        rotation = rotation @ _rot_x(-q[5])
        frame = self.base_rotation @ rotation @ self._mount
        position = self.base_position_mm + self.base_rotation @ offset
        return frame, position

    def solve(self, frame, position, reference=None) -> Reach | None:
        """The configuration closest to `reference` that is inside the derated window.

        None means no configuration of this arm reaches the pose with every joint inside
        its window - which is the question the planner asks. `reference` is the previous
        configuration on the path: giving it keeps the arm on one branch instead of letting
        each pose pick a different elbow-up/wrist-flip solution, which is what a controller
        does and what makes a sequence of accepted poses a sequence the arm can fly.
        """
        reach = None
        best_rank: tuple[float, float] | None = None
        for raw in self._configurations(frame, position, reference):
            joints = self._representative(raw, reference)
            usage = self.utilization(joints)
            if float(usage.max()) > 1.0 + _INSIDE_TOL:
                continue
            rank = (_reference_distance(joints, reference), float(usage.max()))
            if best_rank is None or rank < best_rank:
                best_rank = rank
                reach = _reach_of(self, joints)
        return reach

    def along(self, start_frame, end_frame, start, end, *, step_mm: float,
              reference=None) -> MoveCheck:
        """Sample one straight move: does every pose on it have a legal configuration?

        Positions are interpolated linearly and frames by the shortest rotation between
        them, which is what an `L` move does. Both ends are included, and each sample is
        solved with the previous sample's configuration as the reference, so a move that
        leaves the branch it started on is rejected rather than silently flipped.
        """
        start_position = np.asarray(start, dtype=float).reshape(3)
        end_position = np.asarray(end, dtype=float).reshape(3)
        length = float(np.linalg.norm(end_position - start_position))
        count = max(1, int(np.ceil(length / max(float(step_mm), 1e-9))))
        fractions = np.linspace(0.0, 1.0, count + 1)
        positions = (start_position[None, :] * (1.0 - fractions)[:, None]
                     + end_position[None, :] * fractions[:, None])
        frames = _interpolate_frames(start_frame, end_frame, fractions)
        joints = None if reference is None else np.asarray(reference, dtype=float)
        worst, worst_joint = 0.0, 0
        for index, (frame, position) in enumerate(zip(frames, positions, strict=True)):
            reach = self.solve(frame, position, joints)
            if reach is None:
                return MoveCheck(ok=False, joints_deg=joints, worst_utilization=worst,
                                 worst_joint=worst_joint, samples=index + 1,
                                 failed_at_mm=round(float(fractions[index]) * length, 3))
            joints = reach.joints_deg
            if reach.utilization > worst:
                worst, worst_joint = float(reach.utilization), int(reach.worst_joint)
        return MoveCheck(ok=True, joints_deg=joints, worst_utilization=worst,
                         worst_joint=worst_joint, samples=len(fractions))

    # -- the solver itself ---------------------------------------------------

    def _configurations(self, frame, position, reference):
        """Every configuration of the arm on this pose: 8, or 4 at a wrist singularity."""
        spec = self.spec
        frame = np.asarray(frame, dtype=float).reshape(3, 3)
        position = np.asarray(position, dtype=float).reshape(3)
        # The wrist centre, not the TCP: J6's offset from the J4/J5 intersection lives on
        # the optical axis (see the module docstring), so this is exact, not an estimate.
        wrist = position - self.wrist_to_tcp_mm * frame[:, 2]
        local = self.base_rotation.T @ (wrist - self.base_position_mm)

        shoulder_height = float(spec.shoulder_height_mm)
        shoulder_offset = float(spec.shoulder_offset_mm)
        upper_arm = float(spec.upper_arm_mm)
        elbow_offset = float(spec.elbow_offset_mm)
        forearm = float(spec.forearm_mm)
        rho = float(np.hypot(elbow_offset, forearm))
        beta = float(np.arctan2(elbow_offset, forearm))

        link6 = frame @ self._mount.T
        solutions: list[np.ndarray] = []
        planar = local - _xyz(0.0, 0.0, shoulder_height)
        base_angle = float(np.arctan2(planar[1], planar[0]))
        # Two shoulder solutions: the arm reaching over the base, and reaching back past it.
        for q1 in (base_angle, base_angle + np.pi):
            cos, sin = float(np.cos(q1)), float(np.sin(q1))
            reach_x = cos * planar[0] + sin * planar[1]
            reach_z = float(planar[2])
            x_offset = reach_x - shoulder_offset
            span = x_offset * x_offset + reach_z * reach_z
            sin_elbow = (span - rho * rho - upper_arm * upper_arm) / (2.0 * upper_arm * rho)
            if not -1.0 <= sin_elbow <= 1.0:
                continue  # outside the arm's own envelope, whatever the joint limits say
            base_elbow = float(np.arcsin(float(np.clip(sin_elbow, -1.0, 1.0))))
            # Elbow down and elbow up: the two intersections of the upper arm with the
            # constant-length elbow-to-wrist vector, which is the same pair twice because
            # asin and pi - asin describe the two sides of the shoulder.
            for elbow in (base_elbow, np.pi - base_elbow):
                q3 = elbow - beta
                height = rho * float(np.cos(elbow))
                vertical = upper_arm + rho * float(np.sin(elbow))
                q2 = float(np.arctan2(x_offset, reach_z)) - float(np.arctan2(height, vertical))
                arm_rotation = _rot_z(q1) @ _rot_y(q2) @ _rot_y(-q3)
                posture = arm_rotation.T @ (self.base_rotation.T @ link6)
                last = None if reference is None else float(np.deg2rad(
                    np.asarray(reference, dtype=float)[5]))
                # Every angle below is radians, which is what the trigonometry works in;
                # the one conversion to degrees happens here, where the configuration
                # leaves the solver, because everything outside it - limits, margin,
                # report - is in degrees.
                for q4, q5, q6 in _wrist(posture, last):
                    solutions.append(np.rad2deg(
                        np.array([q1, q2, q3, q4, q5, q6], dtype=float)))
        return solutions

    def _representative(self, joints_deg, reference=None) -> np.ndarray:
        """Per joint, the 360 deg turn nearest the reference, or the one inside the window.

        A revolute joint reaches the same posture one full turn away, and a travel of less
        than 360 deg still admits one of those turns: J3's window is -122..+190, so a
        solution the closed form writes as +251.7 deg is the -108.3 deg the axis can
        actually reach. Picking the representative is part of solving, not cosmetics - it is
        the difference between "unreachable" and a legal configuration. Rejecting a joint
        that is outside every representative is left to the utilization test.
        """
        joints = np.asarray(joints_deg, dtype=float).reshape(JOINT_COUNT).copy()
        previous = (None if reference is None
                    else np.asarray(reference, dtype=float).reshape(JOINT_COUNT))
        for index in range(JOINT_COUNT):
            raw = float(joints[index])
            low, high = (float(value) for value in self.allowed_deg[index])
            centre = (low + high) / 2.0
            best_key, best_value = None, raw
            for turn in (-1.0, 0.0, 1.0):
                value = raw + 360.0 * turn
                excess = max(low - value, value - high, 0.0)
                away = abs(value - previous[index]) if previous is not None else 0.0
                key = (excess, away, abs(value - centre))
                if best_key is None or key < best_key:
                    best_key, best_value = key, value
            joints[index] = best_value
        return joints

    def describe(self) -> dict:
        """The static facts about this model, for the report.

        The windows are printed in the WORKSPACE TABLE convention, because that is the one
        a fitter can compare with the controller, and the CAD-convention numbers are given
        beside them because that is where the verdict is computed. `joint_datum_deg` is the
        conversion, so the two can always be checked against each other.
        """
        return {
            "model": self.spec.name,
            "geometry_source": self.spec.source,
            "datasheet_reach_mm": float(self.spec.reach_mm),
            "chain_reach_mm": round(float(self.spec.arm_reach_mm), 1),
            "base_position_mm": [round(float(value), 3) for value in self.base_position_mm],
            "base_rpy_deg": [round(float(value), 3) for value in self.params.base_rpy_deg],
            "joint_limit_margin": float(self.params.joint_limit_margin),
            "joint_limits_deg": _pairs(self.table_limits_deg),
            "allowed_deg": _pairs(self.allowed_table_deg),
            "joint_datum_deg": [round(float(value), 3) for value in self.spec.joint_datum_deg],
            "same_window_in_the_cad_chain_deg": _pairs(self.allowed_deg),
            "flange_to_tcp_mm": round(float(self.params.tool_length_mm), 3),
            "wrist_to_tcp_mm": round(float(self.wrist_to_tcp_mm), 3),
            "tool_mount_deg": float(self.params.tool_mount_deg),
            "not_checked": [
                "collision between the arm and the station",
                "the interior of a joint travel move: both of its ends are checked, and a "
                "joint-space interpolation between two legal ends stays legal because the "
                "allowed window is a box",
                "singular regions, joint rates, and the arm's occlusion of the tracker",
            ],
        }


def arm_for(params: RobotParams) -> RobotArm | None:
    """The arm a configuration asks for, or None when the joint check is switched off."""
    return RobotArm(params) if params.enabled else None


def _reach_of(arm: RobotArm, joints: np.ndarray) -> Reach:
    usage = arm.utilization(joints)
    return Reach(joints_deg=joints, table_deg=arm.table_deg(joints),
                 utilization=float(usage.max()), worst_joint=int(usage.argmax()))


def _reference_distance(joints: np.ndarray, reference) -> float:
    """How far a configuration is from the one before it, in degrees, per joint at worst.

    Both sides have already been taken to the 360 deg turn nearest the other, so this is a
    plain difference and not a shortest-path-per-joint computation.
    """
    if reference is None:
        return 0.0
    reference = np.asarray(reference, dtype=float).reshape(JOINT_COUNT)
    return float(np.abs(np.asarray(joints, dtype=float) - reference).max())


def _wrist(posture: np.ndarray, reference_q6_rad):
    """The `(q4, q5, q6)` triples, in radians, of `posture = Rx(-q4) Ry(-q5) Rx(-q6)`.

    Two of them, except when q5 is at zero: there the J4 and J6 axes are parallel and only
    their sum (or difference) is observable from the orientation. The free share is then
    handed to J6, taking the reference's value when there is one, which keeps a path that
    passes through the singular pose continuous instead of flipping the wrist.
    """
    sin_b = float(np.hypot(posture[0, 1], posture[0, 2]))
    if sin_b > _SINGULAR_SIN:
        triples = []
        for sign in (1.0, -1.0):
            signed = sign * sin_b
            q5 = -float(np.arctan2(signed, float(posture[0, 0])))
            q4 = -float(np.arctan2(float(posture[1, 0]) / signed,
                                   -float(posture[2, 0]) / signed))
            q6 = -float(np.arctan2(float(posture[0, 1]) / signed,
                                   float(posture[0, 2]) / signed))
            triples.append((q4, q5, q6))
        return triples
    total = float(np.arctan2(-float(posture[1, 2]), float(posture[1, 1])))
    q6 = 0.0 if reference_q6_rad is None else float(reference_q6_rad)
    return [(-total - q6, 0.0, q6)]


def _interpolate_frames(start_frame, end_frame, fractions) -> np.ndarray:
    """Frames between two poses, by the shortest rotation, for a straight move."""
    start = np.asarray(start_frame, dtype=float).reshape(3, 3)
    end = np.asarray(end_frame, dtype=float).reshape(3, 3)
    if np.allclose(start, end, atol=1e-12):
        return np.repeat(start[None, :, :], len(fractions), axis=0)
    rotation = _Rotation.from_matrix(np.stack([start, end]))
    turned = rotation[0].inv() * rotation[1]
    angles = turned.as_rotvec()
    if float(np.linalg.norm(angles)) > np.pi:
        # `as_rotvec` already takes the short way for < 180 deg; past that the short way is
        # the other way round, and the interpolation must not take the long one.
        angles = angles - 2.0 * np.pi * angles / float(np.linalg.norm(angles))
    steps = _Rotation.from_rotvec(np.asarray(fractions)[:, None] * angles[None, :])
    return (rotation[0] * steps).as_matrix()
