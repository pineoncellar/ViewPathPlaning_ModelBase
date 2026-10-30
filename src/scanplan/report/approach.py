"""Where the delivered poses sit relative to the part and the base, as a REPORT ONLY.

Two things the workshop keeps asking about, and that this planner deliberately does not act
on:

* **"is the arm being asked to come at the part from underneath?"** The candidate generator
  places a pose at `aim point + normal x distance`, so a downward-facing sample gets a pose
  below the part - and on a real cell the base is wider than the part, so that pose is not
  reachable even when the joint math says it is. Measured on the 2026-10-30 station: 4 of 64
  delivered poses looked upwards by 30 degrees, one sat below the part's underside, and the
  whole target's downward-facing area was 0.43%.

* **"how much room is there between the base and the part?"** It changes with every part and
  every fixture, so a threshold in the config would be a number nobody can maintain. It is
  reported so the next decision can be made with it in hand.

Nothing here filters, reorders or re-prices anything. `report_only` says so in the block
itself, because a number this close to a constraint will otherwise be read as one.
"""

from __future__ import annotations

import numpy as np

from ..config import PlannerConfig
from ..scene import Station
from ..target_surface import TargetSet

# A sample counts as "facing down" past these, and they describe the part rather than the
# planner: 0.5 is 60 degrees off horizontal, so it is genuinely an underside.
_DOWN_COS = -0.5
_STEEP_COS = 0.0


def approach_block(*, config: PlannerConfig, station: Station, target: TargetSet,
                   viewpoints, path=None) -> dict:
    """Pose heights and orientations, and how much of the target faces downwards."""
    block = {
        "report_only": True,
        "note": ("these numbers describe what the plan asks for, they do not screen anything. "
                 "The base-to-part gap changes with every part and fixture, so no threshold "
                 "is derived from it here"),
        **_geometry(station),
    }
    block["poses"] = _poses(station, viewpoints, path)
    block["target"] = _target(target)
    return block


def _geometry(station: Station) -> dict:
    base_bounds = np.asarray(station.base.bounds, dtype=float)
    part_bounds = np.asarray(station.part.bounds, dtype=float)
    base_top = float(base_bounds[1][2])
    part_bottom = float(part_bounds[0][2])
    return {
        "base_top_z_mm": round(base_top, 1),
        "part_bottom_z_mm": round(part_bottom, 1),
        "part_top_z_mm": round(float(part_bounds[1][2]), 1),
        # negative means the part hangs below the top of the base, which is what a part
        # clamped in a fixture usually does
        "clearance_part_to_base_mm": round(part_bottom - base_top, 1),
        "base_footprint_mm": [round(float(value), 1) for value in
                              (base_bounds[1][:2] - base_bounds[0][:2])],
        "part_footprint_mm": [round(float(value), 1) for value in
                              (part_bounds[1][:2] - part_bounds[0][:2])],
    }


def _poses(station: Station, viewpoints, path) -> dict:
    """Height and tilt of every delivered pose, and the bedroom the head has near the base."""
    positions, axes = _delivered(viewpoints, path)
    if len(positions) == 0:
        return {"available": False, "reason": "nothing was delivered"}
    part_bottom = float(np.asarray(station.part.bounds, dtype=float)[0][2])
    base_top = float(np.asarray(station.base.bounds, dtype=float)[1][2])
    upward = axes[:, 2] if len(axes) else np.zeros(0)
    clearance = station.clearance_mm(positions) if len(positions) else np.zeros(0)
    return {
        "available": True,
        "total": int(len(positions)),
        "below_part_bottom": int((positions[:, 2] < part_bottom).sum()),
        "below_base_top": int((positions[:, 2] < base_top).sum()),
        # how far the optical axis points up, in degrees; 0 is horizontal, 90 is straight up
        "looking_up_over_0_deg": int((upward > 0.0).sum()),
        "looking_up_over_20_deg": int((upward > np.sin(np.deg2rad(20.0))).sum()),
        "max_look_up_deg": round(float(max(0.0, np.degrees(
            np.arcsin(np.clip(upward.max(), -1.0, 1.0))))) if len(upward) else 0.0, 1),
        "min_scanner_z_mm": round(float(positions[:, 2].min()), 1),
        "min_body_clearance_mm": round(float(clearance.min()), 1),
        "note": ("looking_up counts the OPTICAL axis, so a side wall scanned almost "
                 "horizontally is not a violation; below_* count the scanner origin"),
    }


def _target(target: TargetSet) -> dict:
    """How much of the goal surface faces downwards - the area those poses exist for."""
    normals = np.asarray(target.samples.normals, dtype=float).reshape(-1, 3)
    areas = np.asarray(target.samples.area_mm2, dtype=float).reshape(-1)
    total = float(areas.sum())
    if total <= 0.0 or len(normals) == 0:
        return {"available": False, "reason": "the target has no area"}
    down = normals[:, 2] < _DOWN_COS
    steep = (normals[:, 2] >= _DOWN_COS) & (normals[:, 2] < _STEEP_COS)
    return {
        "available": True,
        "samples_facing_down_pct": round(100.0 * float(down.sum()) / len(normals), 3),
        "samples_steep_down_pct": round(100.0 * float(steep.sum()) / len(normals), 3),
        "area_facing_down_pct": round(100.0 * float(areas[down].sum()) / total, 3),
        "area_steep_down_pct": round(100.0 * float(areas[steep].sum()) / total, 3),
        "note": ("facing down is a normal more than 60 degrees below horizontal; steep down is "
                 "between horizontal and that. This is the area a 'never look up' rule would "
                 "have to give up"),
    }


def _delivered(viewpoints, path):
    """Positions and axes of the delivered poses, from the path when there is one.

    The path is preferred because it is what the program actually visits; the viewpoint list
    is the fallback for a single-pose deliverable.
    """
    points = getattr(path, "points", None) if path is not None else None
    if points:
        positions = np.array([np.asarray(point.position, dtype=float) for point in points])
        axes = np.array([np.asarray(point.axis, dtype=float) for point in points])
    else:
        positions = np.array([np.asarray(point.position, dtype=float)
                              for point in viewpoints], dtype=float)
        axes = np.array([np.asarray(_axis_of(point), dtype=float)
                         for point in viewpoints], dtype=float)
    if len(positions) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    lengths = np.linalg.norm(axes, axis=1, keepdims=True)
    axes = axes / np.maximum(lengths, 1e-12)
    return positions, axes


def _axis_of(point):
    axis = getattr(point, "axis", None)
    if axis is None and isinstance(point, dict):
        axis = point.get("axis")
    return axis if axis is not None else (0.0, 0.0, -1.0)
