"""MDH-based reachability screening for Cartesian scan paths.

Geometry
--------
The table below is the arm's own MDH sheet.  It is a standard (distal) DH chain,

    A_i = Rz(theta_i) * Tz(d_i) * Tx(a_i) * Rx(alpha_i),
    theta_i = q_i + theta0_i,

so ``d``/``a``/``alpha`` are the link's own offset, length and twist, and
``theta0`` is the encoder zero written in the DH convention.  ``q`` is the
controller joint command in degrees, which is also the quantity the joint
limits are written against.

    axis   theta0   d       a      alpha
    J1        0    450       0      -90
    J2      -90      0     800        0
    J3        0      0     900      -90
    J4        0   1700       0       90
    J5        0      0       0      -90
    J6        0    240       0        0

Reachability rule
-----------------
A point is reachable only when BOTH hold:

* the inverse kinematics converges to the requested TCP pose, and
* every joint of that solution is inside its (optionally derated) window.

A path is checked point by point and, because a straight Cartesian move can
leave the joint box between its ends, by sampling every straight segment as
well.  :meth:`MDHReachability.filter_path` then drops points that cannot be
reached and re-plans the surviving points, either as straight segments or - if
the caller allows it - as joint-space moves.

The shipped joint limits are the R-2000iC workspace table used by this project.
They are a placeholder for the arm in the MDH sheet: pass the real datasheet
limits to :class:`MDHReachability` when they differ.

This is a kinematic screen, not a collision or controller-timing check.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import scanner_frame

JOINT_COUNT = 6
_EPS = 1e-9


@dataclass(frozen=True)
class MDHLink:
    """One row of the MDH table, angles in degrees and lengths in millimetres."""

    theta0_deg: float
    d_mm: float
    a_mm: float
    alpha_deg: float


MDH_TABLE: tuple[MDHLink, ...] = (
    MDHLink(theta0_deg=0.0, d_mm=450.0, a_mm=0.0, alpha_deg=-90.0),
    MDHLink(theta0_deg=-90.0, d_mm=0.0, a_mm=800.0, alpha_deg=0.0),
    MDHLink(theta0_deg=0.0, d_mm=0.0, a_mm=900.0, alpha_deg=-90.0),
    MDHLink(theta0_deg=0.0, d_mm=1700.0, a_mm=0.0, alpha_deg=90.0),
    MDHLink(theta0_deg=0.0, d_mm=0.0, a_mm=0.0, alpha_deg=-90.0),
    MDHLink(theta0_deg=0.0, d_mm=240.0, a_mm=0.0, alpha_deg=0.0),
)

# Controller-convention limits from the shipped R-2000iC configuration.  Replace
# them with the datasheet values when this module is used for another arm.
DEFAULT_JOINT_LIMITS_DEG: tuple[tuple[float, float], ...] = (
    (-185.0, 185.0),
    (-90.0, 46.0),
    (-122.0, 190.0),
    (-360.0, 360.0),
    (-125.0, 125.0),
    (-360.0, 360.0),
)


def _dh(theta_rad: float, d_mm: float, a_mm: float,
        alpha_rad: float) -> np.ndarray:
    """Standard DH link transform ``Rz(theta) Tz(d) Tx(a) Rx(alpha)``."""
    ct, st = float(np.cos(theta_rad)), float(np.sin(theta_rad))
    ca, sa = float(np.cos(alpha_rad)), float(np.sin(alpha_rad))
    return np.array([
        [ct, -st * ca, st * sa, a_mm * ct],
        [st, ct * ca, -ct * sa, a_mm * st],
        [0.0, sa, ca, d_mm],
        [0.0, 0.0, 0.0, 1.0],
    ])


def _orthonormal(matrix: np.ndarray) -> np.ndarray:
    """The nearest proper rotation, so a noisy input frame cannot skew the IK error."""
    u, _singular, vt = np.linalg.svd(np.asarray(matrix, dtype=float))
    result = u @ vt
    if float(np.linalg.det(result)) < 0.0:
        u[:, -1] *= -1.0
        result = u @ vt
    return result


def _pose(position, rotation) -> np.ndarray:
    frame = np.eye(4)
    frame[:3, :3] = _orthonormal(np.asarray(rotation, dtype=float).reshape(3, 3))
    frame[:3, 3] = np.asarray(position, dtype=float).reshape(3)
    return frame


def pose_from_view(position, axis, roll_deg: float = 0.0) -> np.ndarray:
    """A TCP frame from the planner's position/axis/roll convention."""
    return _pose(position, scanner_frame(axis, float(roll_deg)))


def _as_pose(value) -> np.ndarray:
    """Accept a 4x4 frame, a ``ScannerPose`` or a ``PathPoint``."""
    if hasattr(value, "position") and hasattr(value, "axis"):
        return pose_from_view(value.position, value.axis,
                              float(getattr(value, "roll_deg", 0.0)))
    array = np.asarray(value, dtype=float)
    if array.shape != (4, 4):
        raise ValueError("a pose must be a 4x4 homogeneous matrix, a ScannerPose "
                         "or a PathPoint")
    if not np.isfinite(array).all():
        raise ValueError("pose contains a non-finite value")
    if not np.allclose(array[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError("the last row of a homogeneous pose must be [0, 0, 0, 1]")
    return _pose(array[:3, 3], array[:3, :3])


def _halton(index: int, base: int) -> float:
    result, fraction = 0.0, 1.0
    while index > 0:
        fraction /= base
        result += fraction * (index % base)
        index //= base
    return result


def _interpolate_poses(start: np.ndarray, end: np.ndarray, step_mm: float,
                       step_deg: float, max_samples: int) -> list[np.ndarray]:
    """Straight Cartesian samples from ``start`` to ``end``, end included."""
    if step_mm <= 0.0 or step_deg <= 0.0 or max_samples < 1:
        raise ValueError("step_mm and step_deg must be positive and max_samples >= 1")
    delta = end[:3, 3] - start[:3, 3]
    distance = float(np.linalg.norm(delta))
    rel = Rotation.from_matrix(start[:3, :3].T @ end[:3, :3])
    rotvec = rel.as_rotvec()
    angle = float(np.degrees(np.linalg.norm(rotvec)))
    count = max(1, int(np.ceil(max(distance / step_mm, angle / step_deg))))
    count = min(count, int(max_samples))
    frames: list[np.ndarray] = []
    for index in range(1, count + 1):
        if index == count:
            frames.append(end.copy())
            continue
        fraction = index / count
        frame = np.eye(4)
        frame[:3, :3] = (start[:3, :3]
                         @ Rotation.from_rotvec(fraction * rotvec).as_matrix())
        frame[:3, 3] = start[:3, 3] + fraction * delta
        frames.append(frame)
    return frames


@dataclass(frozen=True)
class IKSolution:
    """One numerical IK result, converged or not, with the pose error it reached."""

    q_deg: np.ndarray
    position_error_mm: float
    rotation_error_deg: float
    converged: bool
    inside_limits: bool
    iterations: int
    weighted_error: float


@dataclass(frozen=True)
class PointReachability:
    """The verdict for one TCP pose."""

    index: int
    pose: np.ndarray
    reachable: bool
    joints_deg: np.ndarray | None
    position_error_mm: float
    rotation_error_deg: float
    utilization: float
    worst_joint: int
    reason: str


@dataclass(frozen=True)
class SegmentReachability:
    """The verdict for the move between two poses, including its interior."""

    start_index: int
    end_index: int
    ok: bool
    mode: str
    samples: int
    worst_utilization: float
    worst_joint: int
    reason: str = ""


@dataclass(frozen=True)
class PathWaypoint:
    """A pose on the re-planned path, with the configuration that reaches it."""

    pose: np.ndarray
    joints_deg: np.ndarray
    source_index: int | None
    mode: str
    utilization: float
    worst_joint: int


@dataclass(frozen=True)
class PathReachability:
    """Every point and every segment of a path, without changing it."""

    points: tuple[PointReachability, ...]
    segments: tuple[SegmentReachability, ...]

    @property
    def reachable(self) -> bool:
        return all(point.reachable for point in self.points) and all(
            segment.ok for segment in self.segments)

    @property
    def failed_indices(self) -> tuple[int, ...]:
        return tuple(point.index for point in self.points if not point.reachable)

    @property
    def failed_segments(self) -> tuple[SegmentReachability, ...]:
        return tuple(segment for segment in self.segments if not segment.ok)


@dataclass(frozen=True)
class ReplanResult:
    """A path rebuilt from the reachable points of an earlier plan."""

    waypoints: tuple[PathWaypoint, ...]
    kept_indices: tuple[int, ...]
    dropped_indices: tuple[int, ...]
    segments: tuple[SegmentReachability, ...]
    notes: tuple[str, ...]

    @property
    def fully_reachable(self) -> bool:
        """Whether every move in the rebuilt path was verified as reachable."""
        return bool(self.waypoints) and all(segment.ok for segment in self.segments)

    @property
    def all_cartesian(self) -> bool:
        return all(segment.mode == "cartesian" for segment in self.segments)


@dataclass(frozen=True)
class _SegmentWalk:
    ok: bool
    mode: str
    samples: int
    worst_utilization: float
    worst_joint: int
    reason: str
    final_q: np.ndarray | None
    waypoints: tuple[PathWaypoint, ...]


class MDHReachability:
    """Forward/inverse kinematics plus a joint-window test for the MDH arm.

    ``joint_limits_deg`` and ``joint_limit_margin`` are applied to the controller
    command ``q``.  A margin below 1 derates both ends of every window; 1.0 uses
    the table's stops exactly.
    """

    def __init__(
        self,
        links: Sequence[MDHLink] = MDH_TABLE,
        joint_limits_deg: Sequence[Sequence[float]] = DEFAULT_JOINT_LIMITS_DEG,
        joint_limit_margin: float = 1.0,
        base_transform=None,
        tool_transform=None,
        position_tol_mm: float = 0.5,
        rotation_tol_deg: float = 0.1,
        orientation_weight_mm: float = 200.0,
        max_iterations: int = 120,
        restarts: int = 20,
        max_step_deg: float = 20.0,
    ) -> None:
        if len(links) != JOINT_COUNT:
            raise ValueError(f"the MDH table must have {JOINT_COUNT} links")
        if not 0.0 < float(joint_limit_margin) <= 1.0:
            raise ValueError("joint_limit_margin must be in (0, 1]")
        limits = np.asarray(joint_limits_deg, dtype=float).reshape(JOINT_COUNT, 2)
        if not np.isfinite(limits).all() or np.any(limits[:, 0] >= limits[:, 1]):
            raise ValueError("each joint limit must be a finite [low, high] pair")
        if position_tol_mm <= 0.0 or rotation_tol_deg <= 0.0:
            raise ValueError("IK tolerances must be positive")
        if orientation_weight_mm <= 0.0 or max_step_deg <= 0.0:
            raise ValueError("orientation_weight_mm and max_step_deg must be positive")
        if max_iterations < 1 or restarts < 0:
            raise ValueError("max_iterations must be >= 1 and restarts >= 0")

        self.links = tuple(links)
        self.limits_deg = limits
        self.joint_limit_margin = float(joint_limit_margin)
        span = limits[:, 1] - limits[:, 0]
        # A margin of 0.9 leaves the outer 5% of the travel unused at each stop.
        self.allowed_deg = np.column_stack((
            limits[:, 0] + 0.5 * (1.0 - self.joint_limit_margin) * span,
            limits[:, 1] - 0.5 * (1.0 - self.joint_limit_margin) * span,
        ))
        self.base_transform = (np.eye(4) if base_transform is None
                               else _as_pose(base_transform))
        self.tool_transform = (np.eye(4) if tool_transform is None
                               else _as_pose(tool_transform))
        self.tool_inverse = np.linalg.inv(self.tool_transform)
        self.position_tol_mm = float(position_tol_mm)
        self.rotation_tol_deg = float(rotation_tol_deg)
        self.rotation_tol_rad = float(np.deg2rad(rotation_tol_deg))
        self.orientation_weight_mm = float(orientation_weight_mm)
        self.max_iterations = int(max_iterations)
        self.restarts = int(restarts)
        self.max_step_deg = float(max_step_deg)
        self.theta0_deg = np.array([link.theta0_deg for link in self.links],
                                   dtype=float)
        self.d_mm = np.array([link.d_mm for link in self.links], dtype=float)
        self.a_mm = np.array([link.a_mm for link in self.links], dtype=float)
        self.alpha_rad = np.deg2rad([link.alpha_deg for link in self.links])

    # ---------------------------------------------------------------- geometry
    def forward(self, joints_deg) -> np.ndarray:
        """TCP pose in the station frame for a vector of six joint commands."""
        q = np.asarray(joints_deg, dtype=float).reshape(JOINT_COUNT)
        if not np.isfinite(q).all():
            raise ValueError("joint values must be finite")
        theta = np.deg2rad(q + self.theta0_deg)
        transform = self.base_transform.copy()
        for index in range(JOINT_COUNT):
            transform = transform @ _dh(theta[index], self.d_mm[index],
                                        self.a_mm[index], self.alpha_rad[index])
        return transform @ self.tool_transform

    def _jacobian(self, joints_deg) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                             np.ndarray]:
        """Geometric Jacobian at the TCP, plus its position and rotation."""
        q = np.asarray(joints_deg, dtype=float).reshape(JOINT_COUNT)
        theta = np.deg2rad(q + self.theta0_deg)
        origins = [self.base_transform[:3, 3].copy()]
        axes = [self.base_transform[:3, 2].copy()]
        transform = self.base_transform.copy()
        for index in range(JOINT_COUNT):
            transform = transform @ _dh(theta[index], self.d_mm[index],
                                        self.a_mm[index], self.alpha_rad[index])
            origins.append(transform[:3, 3].copy())
            axes.append(transform[:3, 2].copy())
        flange = transform
        tcp = flange @ self.tool_transform
        position = tcp[:3, 3]
        jacobian_v = np.empty((3, JOINT_COUNT))
        jacobian_w = np.empty((3, JOINT_COUNT))
        for index in range(JOINT_COUNT):
            jacobian_v[:, index] = np.cross(axes[index], position - origins[index])
            jacobian_w[:, index] = axes[index]
        return jacobian_v, jacobian_w, position, tcp[:3, :3]

    # ------------------------------------------------------------- joint window
    def within_limits(self, joints_deg) -> bool:
        q = np.asarray(joints_deg, dtype=float).reshape(JOINT_COUNT)
        low = self.allowed_deg[:, 0] - 1e-7
        high = self.allowed_deg[:, 1] + 1e-7
        return bool(np.all(q >= low) and np.all(q <= high))

    def utilization(self, joints_deg) -> np.ndarray:
        """Per joint, the share of the allowed half-travel the value uses."""
        q = np.asarray(joints_deg, dtype=float).reshape(JOINT_COUNT)
        centre = 0.5 * (self.allowed_deg[:, 0] + self.allowed_deg[:, 1])
        half = 0.5 * (self.allowed_deg[:, 1] - self.allowed_deg[:, 0])
        return np.abs(q - centre) / np.maximum(half, _EPS)

    # ---------------------------------------------------------------------- IK
    def _error(self, target: np.ndarray, joints_deg) -> tuple[np.ndarray, float,
                                                              float, np.ndarray]:
        tcp = self.forward(joints_deg)
        position_error = target[:3, 3] - tcp[:3, 3]
        rotation_error = Rotation.from_matrix(
            _orthonormal(target[:3, :3] @ tcp[:3, :3].T)).as_rotvec()
        weighted = np.concatenate((position_error,
                                   self.orientation_weight_mm * rotation_error))
        return (weighted, float(np.linalg.norm(position_error)),
                float(np.degrees(np.linalg.norm(rotation_error))), rotation_error)

    def _solve_from(self, target: np.ndarray, seed) -> IKSolution:
        q = np.asarray(seed, dtype=float).reshape(JOINT_COUNT).copy()
        if not np.isfinite(q).all():
            q = np.zeros(JOINT_COUNT)
        q = np.clip(q, self.allowed_deg[:, 0], self.allowed_deg[:, 1])
        best_q = q.copy()
        best_error = np.inf
        best_position = np.inf
        best_rotation = np.inf
        damping = 10.0
        for iteration in range(1, self.max_iterations + 1):
            weighted, position_error, rotation_error, _rotvec = self._error(target, q)
            error_norm = float(np.linalg.norm(weighted))
            if error_norm < best_error:
                best_error, best_q = error_norm, q.copy()
                best_position, best_rotation = position_error, rotation_error
            if (position_error <= self.position_tol_mm
                    and rotation_error <= self.rotation_tol_deg):
                return IKSolution(q_deg=q.copy(), position_error_mm=position_error,
                                  rotation_error_deg=rotation_error, converged=True,
                                  inside_limits=self.within_limits(q),
                                  iterations=iteration, weighted_error=error_norm)
            jacobian_v, jacobian_w, _position, _rotation = self._jacobian(q)
            jacobian = np.vstack((jacobian_v,
                                  self.orientation_weight_mm * jacobian_w))
            normal = jacobian.T @ jacobian + damping * damping * np.eye(JOINT_COUNT)
            try:
                delta = np.linalg.solve(normal, jacobian.T @ weighted)
            except np.linalg.LinAlgError:
                break
            delta_deg = np.rad2deg(delta)
            largest = float(np.max(np.abs(delta_deg)))
            scale = min(1.0, self.max_step_deg / max(largest, _EPS))
            improved = False
            for factor in (1.0, 0.5, 0.25, 0.1):
                candidate = q + factor * scale * delta_deg
                candidate_error = float(np.linalg.norm(self._error(target, candidate)[0]))
                if candidate_error < error_norm:
                    q = candidate
                    damping = max(1.0, damping * 0.7)
                    improved = True
                    break
            if not improved:
                damping = min(500.0, damping * 2.0)
                if damping >= 500.0:
                    break
        weighted, position_error, rotation_error, _rotvec = self._error(target, best_q)
        return IKSolution(q_deg=best_q, position_error_mm=best_position,
                          rotation_error_deg=best_rotation, converged=False,
                          inside_limits=self.within_limits(best_q),
                          iterations=self.max_iterations,
                          weighted_error=float(np.linalg.norm(weighted)))

    def _restart_seeds(self, count: int) -> list[np.ndarray]:
        middle = 0.5 * (self.allowed_deg[:, 0] + self.allowed_deg[:, 1])
        span = self.allowed_deg[:, 1] - self.allowed_deg[:, 0]
        bases = (2, 3, 5, 7, 11, 13)
        seeds = [middle]
        for index in range(1, count + 1):
            seeds.append(self.allowed_deg[:, 0]
                         + np.array([_halton(index, base) for base in bases]) * span)
        return seeds

    def solve(self, target, seed=None, restarts: int | None = None) -> IKSolution:
        """Find a configuration for ``target``; prefers one inside the joint window."""
        target_pose = _as_pose(target) @ self.tool_inverse
        seeds: list[np.ndarray] = []
        if seed is not None:
            seeds.append(np.asarray(seed, dtype=float).reshape(JOINT_COUNT))
        seeds.extend(self._restart_seeds(self.restarts if restarts is None
                                         else max(0, int(restarts))))
        best_outside: IKSolution | None = None
        best_failed: IKSolution | None = None
        for candidate in seeds:
            solution = self._solve_from(target_pose, candidate)
            if solution.converged and solution.inside_limits:
                return solution
            if solution.converged and best_outside is None:
                best_outside = solution
            if (best_failed is None
                    or solution.weighted_error < best_failed.weighted_error):
                best_failed = solution
        if best_outside is not None:
            return best_outside
        # No converged seed: report the closest attempt for diagnostics.
        return best_failed if best_failed is not None else self._solve_from(
            target_pose, np.zeros(JOINT_COUNT))

    def check_pose(self, pose, *, index: int = 0, seed=None) -> PointReachability:
        """Verify one TCP pose: IK exists AND every joint is inside the window."""
        target = _as_pose(pose)
        solution = self.solve(target, seed=seed)
        if solution is None or not solution.converged:
            return PointReachability(
                index=index, pose=target, reachable=False, joints_deg=None,
                position_error_mm=0.0 if solution is None
                else solution.position_error_mm,
                rotation_error_deg=0.0 if solution is None
                else solution.rotation_error_deg,
                utilization=0.0, worst_joint=0,
                reason="no IK solution for the requested TCP pose")
        usage = self.utilization(solution.q_deg)
        if not solution.inside_limits:
            return PointReachability(
                index=index, pose=target, reachable=False,
                joints_deg=solution.q_deg, position_error_mm=solution.position_error_mm,
                rotation_error_deg=solution.rotation_error_deg,
                utilization=float(usage.max()), worst_joint=int(usage.argmax()),
                reason="IK solution lies outside the joint limits")
        return PointReachability(
            index=index, pose=target, reachable=True, joints_deg=solution.q_deg,
            position_error_mm=solution.position_error_mm,
            rotation_error_deg=solution.rotation_error_deg,
            utilization=float(usage.max()), worst_joint=int(usage.argmax()),
            reason="ok")

    # -------------------------------------------------------------------- path
    def check_path(self, points: Iterable, *, step_mm: float = 25.0,
                   step_deg: float = 5.0, max_samples: int = 400) -> PathReachability:
        """Check every pose and every straight move between consecutive poses."""
        poses = [_as_pose(point) for point in points]
        checked: list[PointReachability] = []
        previous_q = None
        for index, pose in enumerate(poses):
            result = self.check_pose(pose, index=index, seed=previous_q)
            checked.append(result)
            previous_q = result.joints_deg if result.reachable else None
        segments: list[SegmentReachability] = []
        for index in range(len(poses) - 1):
            start, end = checked[index], checked[index + 1]
            if not start.reachable or not end.reachable or start.joints_deg is None:
                segments.append(SegmentReachability(
                    start_index=index, end_index=index + 1, ok=False, mode="blocked",
                    samples=0, worst_utilization=0.0, worst_joint=0,
                    reason="an endpoint is not reachable"))
                continue
            walk = self._walk_segment(
                poses[index], poses[index + 1], start.joints_deg, index + 1,
                step_mm=step_mm, step_deg=step_deg, max_samples=max_samples,
                collect=False)
            segments.append(SegmentReachability(
                start_index=index, end_index=index + 1, ok=walk.ok,
                mode=walk.mode, samples=walk.samples,
                worst_utilization=walk.worst_utilization,
                worst_joint=walk.worst_joint, reason=walk.reason))
        return PathReachability(points=tuple(checked), segments=tuple(segments))

    def _walk_segment(self, start_pose: np.ndarray, end_pose: np.ndarray,
                      start_q, end_index: int, *, step_mm: float, step_deg: float,
                      max_samples: int, collect: bool = True) -> _SegmentWalk:
        frames = _interpolate_poses(start_pose, end_pose, step_mm, step_deg,
                                    max_samples)
        q = np.asarray(start_q, dtype=float).reshape(JOINT_COUNT).copy()
        usage = self.utilization(q)
        worst, worst_joint = float(usage.max()), int(usage.argmax())
        waypoints: list[PathWaypoint] = []
        for index, frame in enumerate(frames):
            solution = self.solve(frame, seed=q)
            if (solution is None or not solution.converged
                    or not solution.inside_limits):
                reason = ("no IK solution" if solution is None or not solution.converged
                          else "a joint leaves its limit")
                return _SegmentWalk(False, "cartesian", len(frames), worst, worst_joint,
                                    f"{reason} at sample {index + 1}/{len(frames)}",
                                    None, ())
            q = solution.q_deg
            usage = self.utilization(q)
            if float(usage.max()) > worst:
                worst, worst_joint = float(usage.max()), int(usage.argmax())
            if collect:
                waypoints.append(PathWaypoint(
                    pose=frame, joints_deg=q.copy(),
                    source_index=end_index if index == len(frames) - 1 else None,
                    mode="cartesian", utilization=float(usage.max()),
                    worst_joint=int(usage.argmax())))
        return _SegmentWalk(True, "cartesian", len(frames), worst, worst_joint, "",
                            q.copy(), tuple(waypoints))

    def _joint_waypoints(self, start_q, end_q, end_pose: np.ndarray, end_index: int,
                         *, step_deg: float, max_samples: int
                         ) -> tuple[tuple[PathWaypoint, ...], float, int]:
        start = np.asarray(start_q, dtype=float).reshape(JOINT_COUNT)
        end = np.asarray(end_q, dtype=float).reshape(JOINT_COUNT)
        count = max(1, int(np.ceil(float(np.max(np.abs(end - start))) / step_deg)))
        count = min(count, int(max_samples))
        waypoints: list[PathWaypoint] = []
        worst, worst_joint = 0.0, 0
        for index in range(1, count + 1):
            if index == count:
                q, pose = end.copy(), end_pose.copy()
            else:
                q = start + (end - start) * (index / count)
                pose = self.forward(q)
            usage = self.utilization(q)
            if float(usage.max()) >= worst:
                worst, worst_joint = float(usage.max()), int(usage.argmax())
            waypoints.append(PathWaypoint(
                pose=pose, joints_deg=q.copy(),
                source_index=end_index if index == count else None, mode="joint",
                utilization=float(usage.max()), worst_joint=int(usage.argmax())))
        return tuple(waypoints), worst, worst_joint

    def filter_path(self, points: Sequence, *, allow_joint_moves: bool = False,
                    reorder: bool = False, step_mm: float = 25.0,
                    step_deg: float = 5.0, max_samples: int = 400) -> ReplanResult:
        """Drop unreachable points and rebuild a path through the survivors.

        A point is dropped when its own IK fails or leaves the joint window.  If
        the straight move that would connect two surviving points also fails, the
        later point is dropped and the search continues with the following one.
        With ``allow_joint_moves`` the move is re-planned in joint space instead;
        a joint-space interpolation between two legal configurations stays legal
        because the window is a box.  ``reorder`` re-sequences the survivors by
        nearest TCP distance before the moves are checked.
        """
        poses = [_as_pose(point) for point in points]
        checked = self.check_path(poses, step_mm=step_mm, step_deg=step_deg,
                                  max_samples=max_samples)
        survivors = [point.index for point in checked.points if point.reachable]
        dropped = [point.index for point in checked.points if not point.reachable]
        notes: list[str] = []
        for point in checked.points:
            if not point.reachable:
                notes.append(f"point {point.index} removed: {point.reason}")
        if not survivors:
            return ReplanResult((), (), tuple(dropped), (), tuple(notes))
        if reorder and len(survivors) > 2:
            survivors = self._nearest_order(poses, survivors)

        first = checked.points[survivors[0]]
        current_pose = poses[survivors[0]]
        current_q = first.joints_deg
        assert current_q is not None
        usage = self.utilization(current_q)
        waypoints: list[PathWaypoint] = [PathWaypoint(
            pose=current_pose, joints_deg=current_q.copy(),
            source_index=survivors[0], mode="point",
            utilization=float(usage.max()), worst_joint=int(usage.argmax()))]
        kept = [survivors[0]]
        segments: list[SegmentReachability] = []
        pending = list(survivors[1:])
        while pending:
            next_index = pending.pop(0)
            next_pose = poses[next_index]
            walk = self._walk_segment(
                current_pose, next_pose, current_q, next_index,
                step_mm=step_mm, step_deg=step_deg, max_samples=max_samples)
            if walk.ok:
                waypoints.extend(walk.waypoints)
                current_pose, current_q = next_pose, walk.final_q
                kept.append(next_index)
                segments.append(SegmentReachability(
                    start_index=kept[-2], end_index=next_index, ok=True,
                    mode="cartesian", samples=walk.samples,
                    worst_utilization=walk.worst_utilization,
                    worst_joint=walk.worst_joint))
                continue
            if allow_joint_moves:
                end_q = checked.points[next_index].joints_deg
                if end_q is not None:
                    joint_points, joint_worst, joint_worst_joint = self._joint_waypoints(
                        current_q, end_q, next_pose, next_index, step_deg=step_deg,
                        max_samples=max_samples)
                    waypoints.extend(joint_points)
                    segments.append(SegmentReachability(
                        start_index=kept[-1], end_index=next_index, ok=True,
                        mode="joint", samples=len(joint_points),
                        worst_utilization=joint_worst,
                        worst_joint=joint_worst_joint,
                        reason="straight segment infeasible; joint move substituted"))
                    notes.append(
                        f"segment {kept[-1]}->{next_index} re-planned as a joint move "
                        f"({walk.reason})")
                    current_pose, current_q = next_pose, end_q
                    kept.append(next_index)
                    continue
            dropped.append(next_index)
            notes.append(
                f"point {next_index} removed: straight move from {kept[-1]} "
                f"is not reachable ({walk.reason})")
        return ReplanResult(tuple(waypoints), tuple(kept), tuple(sorted(dropped)),
                           tuple(segments), tuple(notes))

    @staticmethod
    def _nearest_order(poses: Sequence[np.ndarray], indices: Sequence[int]) -> list[int]:
        remaining = list(indices)
        order = [remaining.pop(0)]
        while remaining:
            last = poses[order[-1]][:3, 3]
            next_index = min(
                remaining, key=lambda item: float(np.linalg.norm(
                    poses[item][:3, 3] - last)))
            order.append(next_index)
            remaining.remove(next_index)
        return order


def _self_check() -> int:
    """Run a forward/inverse round trip and one filtered path from the command line."""
    robot = MDHReachability()
    q1 = np.array([10.0, -20.0, 30.0, 40.0, 20.0, 60.0])
    q2 = np.array([12.0, -18.0, 32.0, 42.0, 22.0, 62.0])
    start, end = robot.forward(q1), robot.forward(q2)
    unreachable = start.copy()
    unreachable[0, 3] += 10_000.0
    first = robot.check_pose(start)
    far = robot.check_pose(unreachable)
    print(f"FK/IK round trip: reachable={first.reachable}, "
          f"joints={np.round(first.joints_deg, 3) if first.joints_deg is not None else None}")
    print(f"far pose: reachable={far.reachable} ({far.reason})")
    result = robot.filter_path([start, unreachable, end])
    print(f"filtered path: kept={result.kept_indices}, dropped={result.dropped_indices}, "
          f"waypoints={len(result.waypoints)}")
    for note in result.notes:
        print(f"  {note}")
    ok = (first.reachable and not far.reachable and result.dropped_indices == (1,)
          and len(result.kept_indices) == 2)
    print("SELF-CHECK", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_self_check())
