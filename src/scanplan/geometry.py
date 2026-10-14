"""Pure geometry: the scanner body frame and its tangent helpers.

`scanner_frame` is the single definition of the scanner frame. The view-field
tests, the exported pose and the ROBOGUIDE UTOOL all depend on it, so it must
stay the only place a frame is built.

Watch out for the degenerate case where a reference axis is parallel to the
optical axis: a naive cross product then vanishes.

This module also owns the deterministic sampling primitives. `trimesh` 3.12.7
draws from the GLOBAL numpy random state, so an unseeded sample changes the
answer between runs; every draw therefore goes through `sample_surface_seeded`.
"""

from __future__ import annotations

import numpy as np
import trimesh

_HELPER_Z = np.array([0.0, 0.0, 1.0])
_HELPER_X = np.array([1.0, 0.0, 0.0])
# A reference axis closer than this to the optical axis would make the cross
# product ill-conditioned, so the other helper is used instead.
_PARALLEL_LIMIT = 0.9


def unit(v) -> np.ndarray:
    """`v` scaled to unit length, refusing anything without one."""
    vector = np.asarray(v, dtype=float)
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("direction has no length")
    return vector / length


def scanner_frame(axis, roll_deg: float = 0.0) -> np.ndarray:
    """The scanner body frame: 3x3 with `+Z` along `axis`, x/y across the footprint.

    Columns are the local axes expressed in the station frame, so a world vector
    becomes local with `local_coords`. `roll_deg` turns x and y about the optical
    axis, which matters because the footprint is a rectangle, not a disc.
    """
    z = unit(axis)
    helper = _HELPER_Z if abs(z[2]) < _PARALLEL_LIMIT else _HELPER_X
    x = unit(np.cross(helper, z))
    y = np.cross(z, x)
    if roll_deg:
        angle = np.deg2rad(roll_deg)
        cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
        x, y = cos_a * x + sin_a * y, -sin_a * x + cos_a * y
    return np.stack([x, y, z], axis=1)


def tangents(n) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors spanning the plane perpendicular to `n`."""
    normal = unit(n)
    helper = _HELPER_Z if abs(normal[2]) < _PARALLEL_LIMIT else _HELPER_X
    t1 = unit(np.cross(normal, helper))
    return t1, np.cross(normal, t1)


def tilt_axis(base_axis, tangent, deg: float) -> np.ndarray:
    """`base_axis` rotated `deg` towards `tangent`, as a unit vector."""
    angle = np.deg2rad(deg)
    tilted = unit(base_axis) * float(np.cos(angle)) + unit(tangent) * float(np.sin(angle))
    return unit(tilted)


def local_coords(vectors, frame) -> np.ndarray:
    """Express station-frame `vectors` in `frame` (whose columns are its axes)."""
    return np.asarray(vectors, dtype=float) @ np.asarray(frame, dtype=float)


def grid_cells(points, grid_mm: float) -> np.ndarray:
    """The integer cell each point falls in, for a grid of `grid_mm`."""
    if grid_mm <= 0.0:
        raise ValueError("grid_mm must be positive")
    return np.floor(np.asarray(points, dtype=float) / grid_mm).astype(np.int64)


def grid_dedupe(points, *arrays, grid_mm: float) -> tuple[np.ndarray, ...]:
    """Keep the first sample of every grid cell, and every array row-aligned with it.

    Returned in input order, so a run is reproducible from the seed alone.
    """
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return (points,) + tuple(np.asarray(a) for a in arrays)
    _, first = np.unique(grid_cells(points, grid_mm), axis=0, return_index=True)
    keep = np.sort(first)
    return tuple(np.asarray(array)[keep] for array in (points, *arrays))


def sample_surface_seeded(mesh, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Area-uniform surface samples and the normal of the face each one landed on.

    trimesh samples from the process-wide numpy RNG, so the seed is applied here
    rather than to a private generator; that is also why this must not be called
    from concurrent threads.
    """
    count = int(n)
    if count <= 0:
        raise ValueError("n must be positive")
    np.random.seed(int(seed) & 0xFFFFFFFF)
    points, face_ids = trimesh.sample.sample_surface(mesh, count)
    normals = np.asarray(mesh.face_normals)[face_ids]
    return np.asarray(points, dtype=float), np.asarray(normals, dtype=float)
