"""Ray casting: one interface over the accelerated and the pure-Python intersector.

Every module that has to know what a line of sight meets - the sheet split, the
coverage calculation, the tracker - goes through here, so the decision between
Embree and trimesh's own intersector is made exactly once. Embree is roughly
three orders of magnitude faster, but it is an optional extra: when it is missing
the planner still runs, only slowly.

Two things earn this module its own file rather than a helper:

* trimesh returns `(index_tri, index_ray, locations)` from `intersects_id`, NOT
  `(locations, ray, tri)`. The wrong order unpacks without any error and yields
  nonsense distances, so the unpacking happens in one place and one test pins it.
* `nearest_across` reports the NAME of the mesh that the ray met first, because
  an uncovered target point has to be attributed to the part, the fixture or the
  base - a bare distance cannot say which.

Only the nearest hit per ray is ever asked for: `intersects_any` is several times
slower than `intersects_id(..., multiple_hits=False)`, and the caller needs the
location anyway.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping

import numpy as np

DEFAULT_CHUNK = 250_000
# The pure-Python backend materialises every candidate (ray, triangle) pair of a chunk
# before it narrows them down, and a ray that grazes the surface can have hundreds of
# candidates: 110k raster rays against a 110k-triangle shell asked for a 2.5 GB array.
# The accelerated backend has no such step, so its chunk stays large.
FALLBACK_CHUNK = 2_000


def _batch_failure(error: BaseException) -> bool:
    """Whether an intersector error is about the batch, not about the mesh or the code.

    `rtree` refuses a bounding box whose mins exceed its maxes - which a pathological ray
    can produce - and the pure-Python backend allocates candidates for the whole chunk.
    Both shrink with the chunk, so they are retried rather than raised. Matched by name so
    `rtree` does not become an import this module needs.
    """
    return isinstance(error, MemoryError) or type(error).__name__ == "RTreeError"


def _fallback_intersector_class():
    """trimesh's own intersector: correct, and about 1600x slower."""
    from trimesh.ray import ray_triangle

    return ray_triangle.RayMeshIntersector


def _rebuild_ray_bridge():
    """Re-import trimesh's Embree bridge after purging a cached failed import.

    `import trimesh` imports `trimesh.ray`, which tries to load the Embree bridge and,
    when that fails, stores an ExceptionModule under the name - so registering
    `pyembree` afterwards has no effect and everything silently runs on the pure-Python
    intersector. Dropping the cached failure and importing again recovers the fast
    path. Returns None if the bridge really cannot be imported.
    """
    import importlib

    import trimesh.ray

    sys.modules.pop("trimesh.ray.ray_pyembree", None)
    if hasattr(trimesh.ray, "ray_pyembree"):
        delattr(trimesh.ray, "ray_pyembree")
    importlib.invalidate_caches()
    try:
        return importlib.import_module("trimesh.ray.ray_pyembree")
    except (ImportError, OSError, AttributeError):
        return None


def _intersector_class():
    """The fastest available intersector, or trimesh's own if the extra is absent."""
    try:
        import embreex
        import embreex.mesh_construction
    except (ImportError, OSError, AttributeError):
        # A missing rtree or an unloadable native library lands here too; the planner
        # is allowed to be slow, not broken.
        return _fallback_intersector_class(), False

    # trimesh's bridge looks for a module named `pyembree`; registering embreex under
    # that name before trimesh.ray is imported is what makes it pick the fast path.
    sys.modules["pyembree"] = embreex
    sys.modules["pyembree.mesh_construction"] = embreex.mesh_construction

    import trimesh.ray

    bridge = None
    if getattr(trimesh.ray, "has_embree", False):
        bridge = trimesh.ray.ray_pyembree
    else:
        bridge = _rebuild_ray_bridge()
    if bridge is None:
        return _fallback_intersector_class(), False
    return bridge.RayMeshIntersector, True


_INTERSECTOR_CLASS, EMBREE_AVAILABLE = _intersector_class()


def backend_name() -> str:
    """`"embree"` or `"trimesh-python"` - recorded in the report so runs are comparable."""
    return "embree" if EMBREE_AVAILABLE else "trimesh-python"


def _as_rays(origins, directions) -> tuple[np.ndarray, np.ndarray]:
    origins = np.asarray(origins, dtype=float).reshape(-1, 3)
    directions = np.asarray(directions, dtype=float).reshape(-1, 3)
    if len(origins) != len(directions):
        raise ValueError("origins and directions must have the same length")
    return origins, directions


class RayCaster:
    """Nearest-hit queries against one mesh, in chunks so memory stays bounded."""

    def __init__(self, mesh, chunk: int | None = None):
        self.mesh = mesh
        self.chunk = max(1, int(chunk if chunk is not None else (
            DEFAULT_CHUNK if EMBREE_AVAILABLE else FALLBACK_CHUNK)))
        self._intersector = _INTERSECTOR_CLASS(mesh)

    def nearest(self, origins, directions) -> tuple[np.ndarray, np.ndarray]:
        """Distance and triangle index of the first hit, `inf`/`-1` where a ray misses.

        A ray with a non-finite origin or direction, or a direction of zero length, is a
        miss rather than an error: the backends disagree about those (embree ignores them,
        the pure-Python one hands NaN bounds to rtree and dies), and a degenerate triangle
        normal on a real STL is not a reason to lose a whole run. Such rays never reach the
        intersector, so the chunk offsets become the row numbers of the rays that were cast.

        A batch the backend cannot box or hold is retried at half the size, and a single ray
        that still fails is a miss; that keeps one grazing ray on a dense shell from taking
        the whole batch with it.
        """
        origins, directions = _as_rays(origins, directions)
        count = len(origins)
        distance = np.full(count, np.inf)
        triangle = np.full(count, -1, dtype=np.int64)
        if count == 0:
            return distance, triangle

        usable = np.flatnonzero(
            np.isfinite(origins).all(axis=1)
            & np.isfinite(directions).all(axis=1)
            & (np.linalg.norm(directions, axis=1) > 0.0))
        chunk = self.chunk
        start = 0
        while start < len(usable):
            rows = usable[start:start + chunk]
            try:
                index_tri, index_ray, locations = self._intersector.intersects_id(
                    origins[rows], directions[rows],
                    multiple_hits=False, return_locations=True,
                )
            except Exception as error:
                if not _batch_failure(error):
                    raise
                if len(rows) == 1:
                    start += 1
                    continue
                chunk = max(1, len(rows) // 2)
                continue
            start += len(rows)
            if len(index_ray) == 0:
                continue
            hits = np.asarray(locations, dtype=float).reshape(-1, 3)
            hit_rows = rows[np.asarray(index_ray, dtype=np.int64)]
            distance[hit_rows] = np.linalg.norm(hits - origins[hit_rows], axis=1)
            triangle[hit_rows] = np.asarray(index_tri, dtype=np.int64)
        return distance, triangle


def nearest_across(casters: Mapping[str, RayCaster], origins, directions,
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The nearest hit over several meshes: `(distance, mesh_name, triangle_index)`.

    Ties go to the first mesh in iteration order, so callers should pass a stable
    mapping when the attribution has to be reproducible.
    """
    origins, directions = _as_rays(origins, directions)
    count = len(origins)
    distance = np.full(count, np.inf)
    names = np.full(count, "", dtype=object)
    triangle = np.full(count, -1, dtype=np.int64)
    for name, caster in casters.items():
        candidate_distance, candidate_triangle = caster.nearest(origins, directions)
        closer = candidate_distance < distance
        distance[closer] = candidate_distance[closer]
        names[closer] = name
        triangle[closer] = candidate_triangle[closer]
    return distance, names, triangle
