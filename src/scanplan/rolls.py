"""Roll about the optical axis: which way the scanner's long side points.

`scanner_frame` derives x and y from a fixed reference axis, so the same optical axis
always yields the same frame - but two neighbouring poses whose normals sit either side
of that reference get frames up to a half turn apart. The controller then has to turn its
last joint (J6, which carries the scanner ball) by that whole amount in a single move,
and those are exactly the points that failed the ROBOGUIDE reach check on the reference
cell.

The rule here is the one measured on the `qianduan2` front-end planner: keep the
scanner's tool `+X` as close to the station's `+X` as the optical axis allows. The
reference program did that (tool x . +X was 0.98 on average and never below 0.73, the
exported R stayed within a few degrees of zero), and its whole point is stability: two
neighbouring axes that differ by a degree get rolls that differ by about a degree, so
no move asks for a large wrist turn. Following the sweep line instead turns the wrist by
90-180 deg whenever the line direction changes.

Three pieces, and each has exactly one job:

* :func:`roll_for_axis` - the reference roll of one optical axis. Picked at SEED time,
  because the roll turns the rectangle of the field of view and therefore changes what a
  stroke covers; evaluating coverage with one roll and delivering another would break the
  single-coverage-caliber invariant (merge plan section 3.2).
* :func:`smooth_rolls` - re-pick the roll of a sequence so every move turns the tool as
  little as possible. Only for sequences that carry no coverage promise, i.e. transitions
  and the pose-only delivery.
* :func:`twist_report` and :func:`pose_turn_deg` - the wrist numbers per move, so "the
  wrist does not take a big turn" is a measurement rather than a claim.

`pose_turn_deg` is the single definition of "how far did the head have to turn": the
ordering prices it, the connector spreads it, and the report counts it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .geometry import scanner_frame, unit

# The reference program keeps the scanner's long side (tool +X) on station +X.
TOOL_X_REFERENCE = np.array([1.0, 0.0, 0.0])
# Below this projected length, station +X lies within about 20 deg of the optical axis and
# "keep tool x on +X" has no stable answer; the roll then stays at the frame default.
TOOL_X_MIN_PROJECTION = 0.35
# Degrees of deviation from the reference posture the smoother accepts per degree of
# rotation it saves. Small on purpose: the reference posture is the one the program was
# verified with, so the smoother returns to it wherever that costs nothing.
DEFAULT_REFERENCE_BIAS = 0.05
# The per-move wrist rotation a path aims for. The optical axis follows the surface, so a
# steep feature can exceed it on its own; the report names those moves instead of the
# export pretending they are not there.
DEFAULT_MAX_WRIST_STEP_DEG = 30.0

_ROLL_OFFSETS_DEG = np.arange(-180.0, 180.0, 1.0)
_EPS = 1e-9


def wrap_deg(value: float) -> float:
    """`value` wrapped into [-180, 180).

    Stated as a half-open range because that is what the arithmetic gives: a value of
    exactly 180 maps to -180. The two are the same rotation, and every caller takes the
    magnitude, so the choice only has to be documented rather than special-cased.
    """
    return float((float(value) + 180.0) % 360.0) - 180.0


def roll_along(axis, tangent) -> float:
    """The roll that turns the scanner frame's x axis onto `tangent`.

    `tangent` is projected onto the plane perpendicular to the optical axis first, so the
    result is the roll whose tool x points as far along `tangent` as the axis allows.
    """
    frame = scanner_frame(axis, 0.0)
    direction = np.asarray(tangent, dtype=float).reshape(3)
    axis_vector = np.asarray(axis, dtype=float).reshape(3)
    direction = direction - float(direction @ axis_vector) * axis_vector
    length = float(np.linalg.norm(direction))
    if length <= _EPS:
        return 0.0
    direction = direction / length
    return float(np.degrees(np.arctan2(frame[:, 1] @ direction, frame[:, 0] @ direction)))


def roll_for_axis(axis) -> float:
    """The roll that keeps tool x as close to station +X as the optical axis allows."""
    axis_vector = unit(np.asarray(axis, dtype=float).reshape(3))
    projected = TOOL_X_REFERENCE - float(TOOL_X_REFERENCE @ axis_vector) * axis_vector
    if float(np.linalg.norm(projected)) < TOOL_X_MIN_PROJECTION:
        return 0.0
    return roll_along(axis_vector, projected)


def pose_turn_deg(axis_a, roll_a: float, axis_b, roll_b: float) -> float:
    """The rotation between two scanner poses, as one angle.

    Measured on the two frames the program actually delivers, because that is what the
    controller interpolates and therefore what the wrist has to turn. Measuring it as
    "swing to the new axis plus the roll difference" used to agree with this whenever both
    rolls came from the same fixed reference, and stops agreeing as soon as a roll follows
    its own axis: part of that difference is produced by the swing itself, and counting it
    again reported a 94 degree turn for a 30 degree tilt.

    A large value here is a real wrist turn, not an artifact of a reference axis: two
    nearly identical optical axes whose rolls are taken from a fixed helper really do
    export frames 90 degrees apart, which is why the seed roll policy follows station +X
    instead (see :func:`roll_for_axis`) and the report names the moves that still exceed
    the limit.
    """
    before = tool_frame(axis_a, roll_a)
    after = tool_frame(axis_b, roll_b)
    relative = before.T @ after
    cosine = float(np.clip(0.5 * (float(np.trace(relative)) - 1.0), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def tool_frame(axis, roll_deg: float) -> np.ndarray:
    """The scanner body frame of one pose; `frame[:, 2]` is the optical axis."""
    return scanner_frame(axis, float(roll_deg))


def _rotation_angles(previous_frame: np.ndarray, axis, rolls: np.ndarray,
                     ) -> tuple[np.ndarray, np.ndarray]:
    """The tool-frame rotation from `previous_frame` for each candidate roll."""
    base = scanner_frame(axis, 0.0)
    angles = np.deg2rad(np.asarray(rolls, dtype=float))
    cos_a, sin_a = np.cos(angles)[:, None], np.sin(angles)[:, None]
    x = cos_a * base[:, 0] + sin_a * base[:, 1]
    y = -sin_a * base[:, 0] + cos_a * base[:, 1]
    frames = np.stack((x, y, np.broadcast_to(base[:, 2], x.shape)), axis=2)
    relative = np.einsum("ij,njk->nik", np.asarray(previous_frame, dtype=float).T, frames)
    trace = np.trace(relative, axis1=1, axis2=2)
    return np.degrees(np.arccos(np.clip(0.5 * (trace - 1.0), -1.0, 1.0))), frames


def roll_variants(reference: float) -> np.ndarray:
    """The rolls one point may take when its reference roll asks for too big a turn.

    The reference itself, the two quarter turns, and every degree in between - the simple
    ladder, because what has to be ruled out is a jump of tens of degrees, not a fraction
    of one.
    """
    return float(reference) + _ROLL_OFFSETS_DEG


def smooth_rolls(poses: Sequence, bias: float = DEFAULT_REFERENCE_BIAS) -> list:
    """Re-pick every roll so each move turns the tool frame as little as possible.

    Each point takes the roll that minimises `rotation from the previous pose + bias *
    distance from the reference`, so a move never turns the tool more than it has to and
    the scanner still returns to the reference posture wherever that costs nothing.

    The input is anything exposing `axis` and `roll_deg` (a :class:`ScannerPose` or a
    :class:`~scanplan.path.PathPoint`); the output has the same type, with `roll_deg`
    replaced where it changed. Only call this on points that carry no coverage promise:
    a coverage point's roll is part of the coverage figure.
    """
    from dataclasses import replace

    result: list = []
    previous_frame: np.ndarray | None = None
    for pose in poses:
        reference = float(pose.roll_deg)
        axis = np.asarray(pose.axis, dtype=float).reshape(3)
        if previous_frame is None:
            result.append(pose)
            previous_frame = scanner_frame(axis, reference)
            continue
        candidates = roll_variants(reference)
        angles, frames = _rotation_angles(previous_frame, axis, candidates)
        deviation = np.abs((candidates - reference + 180.0) % 360.0 - 180.0)
        best = int(np.argmin(angles + float(bias) * deviation))
        chosen = float(candidates[best])
        result.append(replace(pose, roll_deg=chosen) if chosen != reference else pose)
        previous_frame = frames[best]
    return result


def twist_report(poses: Sequence, limit_deg: float = DEFAULT_MAX_WRIST_STEP_DEG,
                 *, worst_count: int = 8) -> dict:
    """How far the tool has to turn between consecutive points, and why.

    Three numbers per move, because they ask for different fixes:

    * `tilt_deg` - the angle between two neighbouring optical axes. The surface sets it;
      a coarser or finer sample step changes it, the export cannot.
    * `twist_deg` - the extra rotation about the optical axis beyond the one the tilt
      already carries. This is the wrist's last joint (J6, which carries the scanner
      ball), and the reference roll keeps it near zero.
    * `rotation_deg` - the whole tool-frame rotation, i.e. what the wrist has to deliver.

    Reported in three parts because "the wrist moves too much" has three different fixes:
    resample the surface, re-pick the roll, or re-order the path.
    """
    items = list(poses)
    if len(items) < 2:
        return {"moves": 0, "max_tilt_deg": 0.0, "max_twist_deg": 0.0,
                "max_rotation_deg": 0.0, "moves_over_limit": 0,
                "limit_deg": float(limit_deg), "total_tilt_deg": 0.0,
                "total_twist_deg": 0.0, "total_rotation_deg": 0.0, "worst": []}
    tilts: list[float] = []
    twists: list[float] = []
    rotations: list[float] = []
    for first, second in zip(items, items[1:], strict=False):
        axis_a = unit(np.asarray(first.axis, dtype=float))
        axis_b = unit(np.asarray(second.axis, dtype=float))
        tilts.append(float(np.degrees(np.arccos(
            float(np.clip(np.dot(axis_a, axis_b), -1.0, 1.0))))))
        frame_a = scanner_frame(axis_a, float(first.roll_deg))
        twists.append(abs(wrap_deg(float(second.roll_deg) - roll_along(axis_b, frame_a[:, 0]))))
        rotations.append(pose_turn_deg(axis_a, float(first.roll_deg),
                                       axis_b, float(second.roll_deg)))
    tilt = np.asarray(tilts)
    twist = np.asarray(twists)
    rotation = np.asarray(rotations)
    limit = float(limit_deg)
    worst = sorted(range(len(rotation)), key=lambda index: -rotation[index])[:int(worst_count)]
    return {
        "moves": int(len(rotation)),
        "max_tilt_deg": round(float(tilt.max()), 1),
        "max_twist_deg": round(float(twist.max()), 1),
        "max_rotation_deg": round(float(rotation.max()), 1),
        "moves_over_limit": int(np.count_nonzero(rotation > limit)),
        "limit_deg": limit,
        "total_tilt_deg": round(float(tilt.sum()), 1),
        "total_twist_deg": round(float(twist.sum()), 1),
        "total_rotation_deg": round(float(rotation.sum()), 1),
        "worst": [{"move": int(index), "rotation_deg": round(float(rotation[index]), 1),
                   "tilt_deg": round(float(tilt[index]), 1),
                   "twist_deg": round(float(twist[index]), 1)} for index in worst
                  if rotation[index] > limit],
    }
