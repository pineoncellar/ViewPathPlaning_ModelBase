"""The coverage figures: what was covered, what was not, and why.

Everything a reviewer checks after reading the target definition is assembled here, so
a new metric lands in this file rather than in the report's top-level dictionary.
"""

from __future__ import annotations

import numpy as np

from ..config import PlannerConfig
from ..coverage import CoverageResult, attribute_uncovered
from ..scanner_model import ScannerModel
from ..scene import Station
from ..selection import Selection
from ..target_surface import TargetSet

_INCIDENCE_BINS = 10
_CLUSTER_LIMIT = 12
# Rounded optical axes that count as "the same angle of view", for the diversity histogram.
_AXIS_DECIMALS = 2


def cluster_points(points, area_mm2, *, cell_mm: float = 30.0,
                   limit: int | None = _CLUSTER_LIMIT) -> list[dict]:
    """Group nearby samples into blocks, largest first.

    The point of clustering is that a reviewer can act on "one and a half dozen small
    features" but not on a list of hundreds of stray samples.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    areas = np.asarray(area_mm2, dtype=float).reshape(-1)
    if len(points) == 0:
        return []
    cells = np.floor(points / cell_mm).astype(np.int64)
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    regions: list[dict] = []
    for group in np.unique(inverse):
        selected = inverse == group
        block = points[selected]
        regions.append({
            "points": int(selected.sum()),
            "area_mm2": round(float(areas[selected].sum()), 1),
            "center": [round(float(v), 1) for v in block.mean(axis=0)],
            "size_mm": [round(float(v), 1) for v in (block.max(axis=0) - block.min(axis=0))],
        })
    regions.sort(key=lambda region: -region["area_mm2"])
    return regions[:limit] if limit is not None else regions


def coverage_block(*, config: PlannerConfig, model: ScannerModel, station: Station,
                   target: TargetSet, coverage: CoverageResult, selection: Selection,
                   say) -> dict:
    """Every number between "how much" and "what is left"; keys stay in report order."""
    uncovered = ~selection.covered
    leak_points = np.asarray(target.samples.points)[uncovered]
    leak_normals = np.asarray(target.samples.normals)[uncovered]
    leak_areas = np.asarray(target.samples.area_mm2)[uncovered]
    say(f"  attributing {len(leak_points)} uncovered samples")
    leak_reasons = attribute_uncovered(leak_points, leak_normals, coverage.poses,
                                       station, model, config, obstacles=station.obstacles(),
                                       tracker_ok=coverage.tracker_ok)

    counts: dict[str, int] = {}
    for reason in leak_reasons:
        counts[str(reason)] = counts.get(str(reason), 0) + 1
    leak_reason_list = [str(reason) for reason in leak_reasons]

    angles = np.concatenate([np.degrees(np.arccos(np.clip(row, -1.0, 1.0)))
                             for row in coverage.quality]) if coverage.quality else np.zeros(0)
    limit = float(config.scanner.max_incidence_deg)
    histogram, edges = np.histogram(angles, bins=_INCIDENCE_BINS, range=(0.0, limit))

    return {
        "coverage_pct": round(selection.coverage_pct, 3),
        "pool_reach_pct": round(coverage.pool_reach_pct, 3),
        "coverage_area_mm2": round(selection.coverage_area_mm2, 1),
        "uncovered_points": int(uncovered.sum()),
        "uncovered_area_mm2": round(float(leak_areas.sum()), 1),
        "uncovered_by_reason": counts,
        "leak_reasons": leak_reason_list,
        "leak_regions": cluster_points(leak_points, leak_areas),
        "incidence_histogram": {
            "counts": [int(v) for v in histogram],
            "bin_edges_deg": [round(float(v), 3) for v in edges],
            "limit_deg": limit,
            "population": ("every (pose, sample) pair in the candidate POOL, not only the "
                           "pairs the delivered viewpoints cover - so it describes what the "
                           "pool could reach, which is the ceiling the selection works from"),
            "empty_pool": bool(len(angles) == 0),
        },
        "greedy_curve": [[count, round(pct, 3)] for count, pct in selection.curve],
    }


def angle_diversity_block(coverage: CoverageResult, chosen, path) -> dict:
    """How many distinct optical axes see each covered sample.

    Reported, never constrained: the workshop's extra passes are hand tremor and
    experience, and the design explicitly does not ask the planner to imitate them.
    """
    if path is None:
        return {"available": False,
                "reason": "the delivered set is single poses, so every sample is seen from "
                          "as many axes as there are poses aimed at it"}
    if len(chosen) == 0:
        return {"available": False, "reason": "nothing was selected"}
    axes = np.array([np.asarray(coverage.poses[int(index)].axis, dtype=float)
                     for index in chosen])
    lengths = np.linalg.norm(axes, axis=1, keepdims=True)
    keys = np.round(axes / np.maximum(lengths, 1e-12), _AXIS_DECIMALS)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    looks = np.zeros(coverage.n_targets, dtype=np.int32)
    for group in range(len(unique)):
        seen = np.zeros(coverage.n_targets, dtype=bool)
        for row in np.flatnonzero(np.asarray(inverse).ravel() == group):
            seen[coverage.sets[int(chosen[int(row)])]] = True
        looks += seen
    covered = looks > 0
    histogram = np.bincount(looks[covered], minlength=2)[1:]
    return {
        "meaning": "distinct optical axes that observe each covered sample",
        "distinct_axes": int(len(unique)),
        "looks": list(range(1, len(histogram) + 1)),
        "samples": [int(value) for value in histogram],
        "mean_looks": round(float(looks[covered].mean()), 2) if covered.any() else 0.0,
        "note": "reported only; more looks is not a requirement",
    }
