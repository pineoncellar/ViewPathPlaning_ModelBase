"""Ray casting: one interface over the accelerated and the pure-Python intersector.

Every module that has to know what a line of sight meets - the sheet split, the
coverage calculation, the tracker - goes through here, so the decision between
Embree and trimesh's own intersector is made exactly once. Embree is roughly
three orders of magnitude faster, but only Embree is optional: `rtree` is a hard
requirement, because trimesh 3.12.7's own intersector builds a triangle bounds
tree as well. Without `rtree` neither path runs, so a missing one is reported
here rather than several frames down inside trimesh.

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

import importlib
import sys
from collections.abc import Mapping

import numpy as np

DEFAULT_CHUNK = 250_000


def _fallback_intersector_class():
    """trimesh's own intersector: correct, and about 1600x slower.

    It is not dependency-free: `RayMeshIntersector.intersects_id` passes
    `mesh.triangles_tree`, and building that imports `rtree`. So rtree is checked
    here and `None` is returned when it is absent, which `RayCaster` turns into a
    message naming the missing package instead of a traceback inside trimesh.
    """
    try:
        importlib.import_module("rtree")
    except (ImportError, OSError):
        return None

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
        # No Embree: fall back to trimesh's own intersector, which is slow but
        # correct. A missing rtree lands here too, and comes back as None.
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

    def __init__(self, mesh, chunk: int = DEFAULT_CHUNK):
        if _INTERSECTOR_CLASS is None:
            raise RuntimeError(
                "ray casting needs `rtree`, which is missing from this environment; "
                "without it even trimesh's own intersector cannot build its triangle "
                "tree. Install it with `uv sync --extra embree` (that also adds the "
                "~1600x faster Embree backend) or `uv add rtree`."
            )
        self.mesh = mesh
        self.chunk = max(1, int(chunk))
        self._intersector = _INTERSECTOR_CLASS(mesh)

    def nearest(self, origins, directions) -> tuple[np.ndarray, np.ndarray]:
        """Distance and triangle index of the first hit, `inf`/`-1` where a ray misses."""
        origins, directions = _as_rays(origins, directions)
        count = len(origins)
        distance = np.full(count, np.inf)
        triangle = np.full(count, -1, dtype=np.int64)
        if count == 0:
            return distance, triangle

        for start in range(0, count, self.chunk):
            stop = min(start + self.chunk, count)
            index_tri, index_ray, locations = self._intersector.intersects_id(
                origins[start:stop], directions[start:stop],
                multiple_hits=False, return_locations=True,
            )
            if len(index_ray) == 0:
                continue
            hits = np.asarray(locations, dtype=float).reshape(-1, 3)
            rows = start + np.asarray(index_ray, dtype=np.int64)
            distance[rows] = np.linalg.norm(hits - origins[rows], axis=1)
            triangle[rows] = np.asarray(index_tri, dtype=np.int64)
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
