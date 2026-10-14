"""Generating candidate scanner poses, coarsely and then finely.

For each aim point take several working distances, tilt the optical axis a few
degrees off the surface normal, and try a couple of rolls about it - the roll
matters because the field of view is a rectangle, not a disc.


Candidates are generated without regard to occlusion or reachability, then
filtered. Filtering first and planning second is the right order: a viewpoint
discovered to be unusable after selection leaves a hole that cannot be patched
without planning again.

Density is what actually buys coverage. A pool of one pose per target point caps
coverage far below what the geometry allows, because one pose covers its whole
footprint. Measured on a real part, per-point reachability was about 96% while
the pool as a whole reached about 99.6% - so the gap between achieved coverage
and the pool's ceiling is a candidate-generation question, not a set-cover one.
Refine towards the still-uncovered region before concluding anything is
unreachable, and seed that refinement too.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .config import ScannerParams
from .geometry import grid_dedupe, tangents, tilt_axis
from .scanner_model import ScannerPose


def aim_indices(points, grid_mm: float) -> np.ndarray:
    """The first sample of every `grid_mm` cell: the points a ladder is aimed at.

    Aiming at cells rather than at every sample is what keeps the pool from growing
    with the sampling density, because one pose already covers its whole footprint.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(points) == 0:
        return np.zeros(0, dtype=np.int64)
    indices = np.arange(len(points))
    _, kept = grid_dedupe(points, indices, grid_mm=grid_mm)
    return np.sort(np.asarray(kept, dtype=np.int64))


def ladder(point, normal, params: ScannerParams) -> list[ScannerPose]:
    """The poses aimed at one surface point: every distance, tilt and roll.

    The roll is skipped where the tilt is zero, because rolling about the optical axis
    only turns the rectangular footprint - at zero tilt the two rolls differ by a
    quarter turn of the same pose's footprint, which the selection stage cannot use
    and which would double the pool for nothing. Away from zero tilt the tilt itself
    is what turns the footprint, so both rolls are kept.
    """
    point = np.asarray(point, dtype=float).reshape(3)
    normal = np.asarray(normal, dtype=float).reshape(3)
    normal = normal / np.linalg.norm(normal)
    first_tangent, second_tangent = tangents(normal)
    inward = -normal

    poses: list[ScannerPose] = []
    for distance in params.distances_mm:
        position = point + normal * distance
        for tilt_deg in params.tilts_deg:
            if tilt_deg == 0.0:
                axis = inward
                rolls = params.rolls_deg[:1]
            elif tilt_deg > 0.0:
                axis = tilt_axis(inward, first_tangent, tilt_deg)
                rolls = params.rolls_deg
            else:
                axis = tilt_axis(inward, second_tangent, -tilt_deg)
                rolls = params.rolls_deg
            for roll_deg in rolls:
                poses.append(ScannerPose(position=position.copy(), axis=axis.copy(),
                                         roll_deg=float(roll_deg)))
    return poses


def dedupe(poses) -> list[ScannerPose]:
    """Drop poses that repeat to within a millimetre and a hundredth of an axis."""
    seen: set[tuple] = set()
    unique: list[ScannerPose] = []
    for pose in poses:
        key = pose.key()
        if key in seen:
            continue
        seen.add(key)
        unique.append(pose)
    return unique


def generate(points, normals, params: ScannerParams, *, seed: int,
             aim_grid_mm: float | None = None, max_candidates: int | None = None,
             log: Callable[[str], None] | None = None) -> list[ScannerPose]:
    """Build the candidate pool: a ladder on every aim cell of the target."""
    say = log or (lambda _message: None)
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    normals = np.asarray(normals, dtype=float).reshape(-1, 3)
    if len(points) == 0:
        return []
    if len(points) != len(normals):
        raise ValueError("points and normals must be the same length")

    grid = float(params.aim_grid_mm if aim_grid_mm is None else aim_grid_mm)
    aims = aim_indices(points, grid)
    pool: list[ScannerPose] = []
    for index in aims:
        pool.extend(ladder(points[index], normals[index], params))
    pool = dedupe(pool)

    limit = int(params.max_candidates if max_candidates is None else max_candidates)
    if limit <= 0:
        raise ValueError("max_candidates must be positive")
    if len(pool) > limit:
        chosen = np.random.default_rng(seed).choice(len(pool), limit, replace=False)
        pool = [pool[int(i)] for i in np.sort(chosen)]
    say(f"  {len(aims)} aim points on a {grid:g} mm grid -> {len(pool)} candidates")
    return pool


def refine(leak_points, leak_normals, params: ScannerParams, *, seed: int,
           max_candidates: int | None = None,
           log: Callable[[str], None] | None = None) -> list[ScannerPose]:
    """Aim a new ladder at what is still uncovered, on a finer grid than the face.

    The leak region is small by definition, and the measured viewpoint count is limited
    by candidate generation rather than by the selection stage, so this is where a
    better answer comes from. The refinement pool still honours the same cap as the
    first pass - a safety net rather than a limit, since a leak on a 25 mm aim grid
    stays well below it - and the `seed` is what keeps a capped refinement reproducible.
    """
    say = log or (lambda _message: None)
    leak_points = np.asarray(leak_points, dtype=float).reshape(-1, 3)
    leak_normals = np.asarray(leak_normals, dtype=float).reshape(-1, 3)
    if len(leak_points) == 0:
        return []
    if len(leak_points) != len(leak_normals):
        raise ValueError("leak points and normals must be the same length")

    aims = aim_indices(leak_points, params.refine_aim_grid_mm)
    pool: list[ScannerPose] = []
    for index in aims:
        pool.extend(ladder(leak_points[index], leak_normals[index], params))
    pool = dedupe(pool)
    limit = int(params.max_candidates if max_candidates is None else max_candidates)
    if len(pool) > limit:
        chosen = np.random.default_rng(seed).choice(len(pool), limit, replace=False)
        pool = [pool[int(i)] for i in np.sort(chosen)]
    say(f"  refinement: {len(aims)} leak aim points on a "
        f"{params.refine_aim_grid_mm:g} mm grid -> {len(pool)} candidates")
    return pool
