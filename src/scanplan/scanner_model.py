"""When can one scanner pose validly measure one surface point?

A point counts as covered by a pose only when ALL hold:
  1. inside the field of view (rectangular footprint, scaled with distance);
  2. distance within the measuring range;
  3. incidence angle within the limit;
  4. line of sight clear of part, fixture and base;
  5. the scanner body fits, including the safety clearance;
  6. the tracker can see the scanner.

The body test uses a convex envelope on purpose: it can only ever call a fitting
pose unfitting, never the reverse, so the error direction is "report more
misses", which is the safe one.

On point 3: `max_incidence_deg` is the most sensitive parameter of the planner
and is a placeholder. Never hard-code it here.

Conditions 5 and 6 stay out of `measure_mask` on purpose: `body_fits` takes an
already-computed clearance, so the model never needs a station and the body test
stays a one-line comparison, while the tracker lives with the placement sweep
that decides where the tracker even is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ScannerParams
from .geometry import local_coords, scanner_frame
from .raycast import nearest_across

_EPS = 1e-9


@dataclass(frozen=True)
class ScannerPose:
    """Where the scanner sits, where it looks, and how its rectangle is turned.

    `axis` points from the scanner towards the part and is the scanner's local +Z,
    which is the frame contract the exported pose and the ROBOGUIDE UTOOL share.
    """

    position: np.ndarray
    axis: np.ndarray
    roll_deg: float = 0.0

    @property
    def frame(self) -> np.ndarray:
        """The scanner body frame in the station frame; `frame[:, 2]` is the axis."""
        return scanner_frame(self.axis, self.roll_deg)

    def key(self, decimals: int = 0) -> tuple:
        """A rounded identity, for de-duplicating generated poses."""
        return (tuple(np.round(np.asarray(self.position, dtype=float), decimals))
                + tuple(np.round(np.asarray(self.axis, dtype=float), 2))
                + (round(float(self.roll_deg), 0),))


class ScannerModel:
    """The scanner's geometric and quality envelope, from the configuration."""

    def __init__(self, params: ScannerParams):
        self.params = params
        self.cos_incidence_limit = float(np.cos(np.deg2rad(params.max_incidence_deg)))
        self.min_distance_mm, self.max_distance_mm = params.range_mm
        self.body_limit_mm = params.body_sphere_r_mm + params.body_clearance_mm

    def footprint_mm(self, distance_mm):
        """The measuring footprint at the given distance, capped at the datasheet value.

        Elementwise, so a vector of distances is fine. The cap is the footprint at the
        standoff by default, which means it stops growing there and shrinks below it.
        """
        distance = np.asarray(distance_mm, dtype=float)
        scale = distance / self.params.standoff_mm
        width = np.minimum(self.params.fov_at_standoff_mm[0] * scale,
                           self.params.fov_cap_mm[0])
        height = np.minimum(self.params.fov_at_standoff_mm[1] * scale,
                            self.params.fov_cap_mm[1])
        return width, height

    def incidence_cos(self, view_dirs, normals) -> np.ndarray:
        """Cosine of the angle between each sight line and the surface normal.

        Elementwise on purpose: `(views @ normals.T).diagonal()` is algebraically the
        same and builds a candidates x targets dense matrix, which on a real part is
        gigabytes and looks like a hang.
        """
        views = np.asarray(view_dirs, dtype=float).reshape(-1, 3)
        normal_array = np.asarray(normals, dtype=float).reshape(-1, 3)
        if len(views) != len(normal_array):
            raise ValueError("view directions and normals must be the same length")
        return -(views * normal_array).sum(axis=1)

    def geometry_mask(self, pose: ScannerPose, points, normals) -> np.ndarray:
        """Conditions 1-3: inside the footprint, within range, within the incidence limit."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        normals = np.asarray(normals, dtype=float).reshape(-1, 3)
        if len(points) == 0:
            return np.zeros(0, dtype=bool)

        offset = points - np.asarray(pose.position, dtype=float)
        distance = np.linalg.norm(offset, axis=1)
        safe_distance = np.maximum(distance, _EPS)
        view = offset / safe_distance[:, None]

        inside_range = (distance >= self.min_distance_mm) & (distance <= self.max_distance_mm)
        cosine = self.incidence_cos(view, normals)
        within_incidence = cosine >= self.cos_incidence_limit

        local = local_coords(offset, pose.frame)
        ahead = local[:, 2] > 0.0
        depth = np.where(ahead, local[:, 2], 1.0)
        width, height = self.footprint_mm(depth)
        inside_footprint = ((np.abs(local[:, 0]) <= width / 2.0)
                            & (np.abs(local[:, 1]) <= height / 2.0))
        return inside_range & within_incidence & ahead & inside_footprint

    def body_fits(self, position_mm, clearance_mm) -> bool:
        """Condition 5: is there room for the scanner body at this position?"""
        return bool(clearance_mm > self.body_limit_mm)


def measure_mask(model: ScannerModel, pose: ScannerPose, points, normals, obstacles, *,
                 occlusion_tol_mm: float) -> np.ndarray:
    """Conditions 1-4 for one pose: geometry, then a clear line of sight.

    `obstacles` maps a name to a :class:`~scanplan.raycast.RayCaster`; the part, the
    fixture and the base all go in, because any of them can block the sight line.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    normals = np.asarray(normals, dtype=float).reshape(-1, 3)
    candidates = model.geometry_mask(pose, points, normals)
    index = np.flatnonzero(candidates)
    if len(index) == 0:
        return candidates

    position = np.asarray(pose.position, dtype=float)
    offset = points[index] - position
    distance = np.linalg.norm(offset, axis=1)
    directions = offset / np.maximum(distance, _EPS)[:, None]
    origins = np.repeat(position[None, :], len(index), axis=0)

    nearest, _, _ = nearest_across(obstacles, origins, directions)
    blocked = nearest < distance - occlusion_tol_mm
    candidates[index[blocked]] = False
    return candidates
