"""Can the tracker see the scanner from a given pose?

The tracker is fixed at the station, so a pose is usable only if the tracker has
a clear line to the SCANNER. This constrains the POSE, not the target point.

Do not confuse the two ways of scoring it. Asking, per target point, "is there a
tracker-visible pose aimed at THIS point" is a systematically pessimistic proxy
and understates coverage badly (observed: a proxy reading of 87% against a true
99.6%). The right question is whether some set of tracker-visible poses covers
the point - and one pose covers its whole footprint, so the tracker only needs
to see the scanner.

The real tracker pose is unknown until the site provides it, so sweep plausible
mounting positions and report the sensitivity rather than inventing one pose and
quoting a single number. That sweep showed azimuth dominating (mount it looking
ALONG the part's thin axis, not square-on to the large faces), height and
distance barely mattering, and the mechanism being occlusion of the scanner by
the part rather than the tracker's field of view.

Deliberate omission: robot-arm occlusion is not modelled - there is no robot
model at this stage. Say so in the report instead of implying completeness,
because on the real cell the arm holding the scanner can block the tracker.

A pose is visible when it is inside the tracker's cone, inside its range, and the
line from the tracker to the SCANNER is unobstructed. The position of the scanner
is a single point; the target points never enter this test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import TrackerParams
from .geometry import sample_surface_seeded
from .raycast import nearest_across
from .scene import StationDescription

AZIMUTHS_DEG = (0.0, 30.0, 60.0, 75.0, 90.0, 105.0, 120.0, 150.0, 180.0,
                -150.0, -120.0, -105.0, -90.0, -75.0, -60.0, -30.0)
DISTANCES_MM = (2500.0, 3500.0, 4500.0)
HEIGHTS_MM = (100.0, 400.0, 800.0, 1200.0)
_CENTROID_SAMPLES = 50_000


@dataclass(frozen=True)
class Tracker:
    """A tracker at a station, looking at `aim`."""

    position: np.ndarray
    aim: np.ndarray
    params: TrackerParams
    name: str = "tracker"

    @property
    def axis(self) -> np.ndarray:
        """The tracker's optical axis, pointing from the tracker towards `aim`."""
        direction = np.asarray(self.aim, dtype=float) - np.asarray(self.position, dtype=float)
        length = float(np.linalg.norm(direction))
        if length <= 0.0:
            raise ValueError("a tracker cannot be aimed at its own position")
        return direction / length

    def describe(self) -> dict:
        """The placement as a report entry, so a number can be traced to a mounting."""
        return {
            "name": self.name,
            "position_mm": [round(float(v), 1) for v in self.position],
            "aim_mm": [round(float(v), 1) for v in self.aim],
            "half_angle_deg": self.params.half_angle_deg,
            "distance_mm": list(self.params.distance_mm),
            "markers_omnidirectional": self.params.markers_omnidirectional,
            "robot_arm_occlusion_modelled": self.params.model_robot_occlusion,
        }


def part_centroid(part) -> np.ndarray:
    """An area-weighted surface centroid, which is what a tracker is aimed at."""
    points, _ = sample_surface_seeded(part, _CENTROID_SAMPLES, seed=11)
    return points.mean(axis=0)


def tracker_from_description(description: StationDescription, params: TrackerParams, *,
                             part=None, name="tracker") -> Tracker | None:
    """The tracker the station file names, or None when it does not name one.

    None is not a fallback: it means the real placement is unknown, which is the state
    the design is written for, and the caller is expected to sweep instead of inventing
    a position.
    """
    if description.tracker_position_mm is None:
        return None
    aim = description.tracker_aim_mm
    if aim is None:
        if part is None:
            raise ValueError(
                "the station file names no tracker aim point and no part was given to "
                "derive one from"
            )
        aim = tuple(float(v) for v in part_centroid(part))
    return Tracker(position=np.array(description.tracker_position_mm, dtype=float),
                   aim=np.array(aim, dtype=float), params=params, name=name)


def visible_mask(tracker: Tracker, positions, obstacles, *,
                 occlusion_tol_mm: float = 2.0) -> np.ndarray:
    """Which scanner positions the tracker can see.

    Occlusion is tested from the tracker to each scanner position, never to the target
    points: one pose covers its whole footprint, so the tracker only has to see the head.
    """
    if not tracker.params.markers_omnidirectional:
        raise NotImplementedError(
            "the marker visibility geometry of the scanner is not measured yet (design "
            "6.7.5 R11), so a directional marker constraint cannot be modelled; leave "
            "'markers_omnidirectional' true or supply the marker geometry first"
        )
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    if len(positions) == 0:
        return np.zeros(0, dtype=bool)

    offset = positions - np.asarray(tracker.position, dtype=float)
    distance = np.linalg.norm(offset, axis=1)
    safe = np.maximum(distance, 1e-9)
    cosine = (offset / safe[:, None]) @ tracker.axis
    low, high = tracker.params.distance_mm
    inside = ((cosine >= float(np.cos(np.deg2rad(tracker.params.half_angle_deg))))
              & (distance >= low) & (distance <= high))
    if not inside.any() or not obstacles:
        return inside

    index = np.flatnonzero(inside)
    origins = np.repeat(np.asarray(tracker.position, dtype=float)[None, :], len(index), axis=0)
    directions = offset[index] / safe[index][:, None]
    nearest, _, _ = nearest_across(obstacles, origins, directions)
    inside[index] = nearest >= distance[index] - occlusion_tol_mm
    return inside


def sweep_placements(part, params: TrackerParams, *, azimuths_deg=AZIMUTHS_DEG,
                     distances_mm=DISTANCES_MM, heights_mm=HEIGHTS_MM) -> list[Tracker]:
    """Hypothetical tracker placements, so sensitivity can be reported instead of a guess.

    The real position is unknown until the site provides it, and the sweep is what turns
    "where does the tracker go" into a layout requirement rather than a design change.
    """
    if not params.markers_omnidirectional:
        raise NotImplementedError(
            "the marker visibility geometry of the scanner is not measured yet (design "
            "6.7.5 R11), so the sweep cannot rank placements; supply it first"
        )
    centroid = part_centroid(part)
    placements: list[Tracker] = []
    for azimuth in azimuths_deg:
        angle = np.deg2rad(azimuth)
        direction = np.array([np.sin(angle), np.cos(angle), 0.0])
        for distance in distances_mm:
            for height in heights_mm:
                position = centroid + direction * distance
                position = np.array([position[0], position[1], float(height)])
                placements.append(Tracker(
                    position=position, aim=centroid, params=params,
                    name=f"az {azimuth:g} deg, {distance:g} mm, z {height:g} mm",
                ))
    return placements


def rank_placements(tracker: Tracker, model_positions, obstacles) -> dict:
    """How much of a candidate pool one placement keeps, for ranking the sweep."""
    positions = np.asarray(model_positions, dtype=float)
    visible = visible_mask(tracker, positions, obstacles)
    return {"tracker": tracker.describe(), "poses": int(len(positions)),
            "visible": int(visible.sum()),
            "visible_pct": round(100.0 * float(visible.mean()) if len(positions) else 0.0, 3)}
