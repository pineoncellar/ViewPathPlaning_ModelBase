"""Build the candidate x target coverage sets.

For each candidate, decide which target samples it can validly measure (see
:mod:`scanplan.scanner_model`) and record them, with a quality term (incidence
angle) so that among equals the more head-on pose can win - a grazing pose is
geometrically covered but scans badly.

Implementation notes that cost real time to learn:

* use the vectorised ray intersector and ask for the FIRST hit only; the generic
  "does this ray hit anything" query is several times slower than the one
  returning the hit index and location;
* never build the incidence cosine as a full matrix product of candidates x
  target points - at realistic sizes that materialises gigabytes and the process
  appears to hang;
* do not assume the intersector's return order; a swapped tuple unpacks without
  error and yields nonsense;
* seed the RNG, or the numbers wobble between runs for no visible reason.

Order per candidate, and it is not arbitrary: the body-clearance test comes first
because it is a KD-tree lookup that costs nothing next to a ray, and a pose that
cannot physically be there must not spend rays. The tracker gate comes next, for
the same reason: it filters POSES, not target points, so it belongs here rather
than as a later pass over the result.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix

from .config import PlannerConfig
from .raycast import nearest_across
from .scanner_model import ScannerModel, ScannerPose, measure_mask
from .scene import Station
from .target_surface import SampleSet

REASONS = ("never_in_view", "blocked_by_part", "blocked_by_fixture", "blocked_by_base",
           "blocked_by_tracker", "only_poses_that_collide", "candidate_found")
_COUNTER_KEYS = ("rejected_clearance", "rejected_tracker", "rejected_no_points", "rays")


@dataclass
class CoverageResult:
    """Which target samples each candidate can measure, and what was thrown out.

    `sets[i]` holds ascending target indices covered by `poses[i]`, with `quality[i]`
    the matching incidence cosines, so that of two poses covering the same sample the
    more head-on one can win.
    """

    poses: list[ScannerPose]
    sets: list[np.ndarray]
    quality: list[np.ndarray]
    tracker_ok: np.ndarray
    counters: dict[str, int]
    n_targets: int
    area_mm2: np.ndarray
    target_area_mm2: float
    _matrix: csr_matrix | None = field(default=None, repr=False, compare=False)

    def matrix(self) -> csr_matrix:
        """The pose x sample coverage matrix, built once and kept as sparse bools."""
        if self._matrix is None:
            rows: list[int] = []
            columns: list[int] = []
            for index, row in enumerate(self.sets):
                rows.extend([index] * len(row))
                columns.extend(int(column) for column in row)
            data = np.ones(len(rows), dtype=bool)
            self._matrix = csr_matrix((data, (rows, columns)),
                                      shape=(len(self.poses), self.n_targets), dtype=bool)
        return self._matrix

    def pool_reach(self) -> np.ndarray:
        """Which samples any candidate in the pool can measure at all."""
        reached = np.zeros(self.n_targets, dtype=bool)
        for row in self.sets:
            reached[row] = True
        return reached

    @property
    def pool_reach_pct(self) -> float:
        """Area share of the target the pool can reach - the ceiling for selection."""
        if self.target_area_mm2 <= 0.0:
            return 0.0
        return float(100.0 * self.area_mm2[self.pool_reach()].sum() / self.target_area_mm2)


def evaluate(poses: Sequence[ScannerPose], samples: SampleSet, station: Station,
             model: ScannerModel, cfg: PlannerConfig, *,
             visible: Callable[[np.ndarray], np.ndarray] | None = None,
             obstacles: dict | None = None,
             log: Callable[[str], None] | None = None) -> CoverageResult:
    """Evaluate every pose against the target samples, returning sparse coverage sets.

    `visible(positions) -> bool mask` is the optional tracker gate: candidates it
    rejects keep an empty coverage set, so the tracker constraint flows into the
    selection problem instead of being applied to the result afterwards.
    """
    say = log or (lambda _message: None)
    points = np.asarray(samples.points, dtype=float).reshape(-1, 3)
    normals = np.asarray(samples.normals, dtype=float).reshape(-1, 3)
    areas = np.asarray(samples.area_mm2, dtype=float)
    n_targets = len(points)
    if len(areas) != n_targets:
        raise ValueError("every sample needs an area")
    casters = station.obstacles() if obstacles is None else obstacles

    positions = (np.array([pose.position for pose in poses], dtype=float)
                 if poses else np.zeros((0, 3)))
    tracker_ok = (np.asarray(visible(positions), dtype=bool) if visible is not None
                  else np.ones(len(poses), dtype=bool))
    counters = dict.fromkeys(_COUNTER_KEYS, 0)

    sets: list[np.ndarray] = []
    quality: list[np.ndarray] = []
    for index, pose in enumerate(poses):
        rejected = None
        if not model.body_fits(pose.position, station.clearance_mm(pose.position)[0]):
            counters["rejected_clearance"] += 1
            rejected = "clearance"
        elif not tracker_ok[index]:
            counters["rejected_tracker"] += 1
            rejected = "tracker"
        if rejected is not None:
            sets.append(np.zeros(0, dtype=np.int64))
            quality.append(np.zeros(0))
            continue

        mask = measure_mask(model, pose, points, normals, casters,
                            occlusion_tol_mm=cfg.occlusion_tol_mm)
        chosen = np.flatnonzero(mask)
        counters["rays"] += len(chosen)
        if len(chosen) == 0:
            counters["rejected_no_points"] += 1
            sets.append(chosen)
            quality.append(np.zeros(0))
            continue

        offset = points[chosen] - np.asarray(pose.position, dtype=float)
        distance = np.linalg.norm(offset, axis=1)
        cosines = model.incidence_cos(offset / np.maximum(distance, 1e-9)[:, None],
                                     normals[chosen])
        sets.append(chosen)
        quality.append(cosines)

    say(f"  evaluated {len(poses)} candidates: {counters}")
    return CoverageResult(poses=list(poses), sets=sets, quality=quality,
                          tracker_ok=tracker_ok, counters=counters, n_targets=n_targets,
                          area_mm2=areas,
                          target_area_mm2=float(samples.total_area_mm2))


def merge(first: CoverageResult, second: CoverageResult) -> CoverageResult:
    """Append the poses of `second` to `first`, for the refinement pass."""
    if first.n_targets != second.n_targets:
        raise ValueError("cannot merge coverage over different target sets")
    counters = {key: first.counters.get(key, 0) + second.counters.get(key, 0)
                for key in _COUNTER_KEYS}
    return CoverageResult(
        poses=first.poses + second.poses,
        sets=first.sets + second.sets,
        quality=first.quality + second.quality,
        tracker_ok=np.concatenate([first.tracker_ok, second.tracker_ok]),
        counters=counters, n_targets=first.n_targets, area_mm2=first.area_mm2,
        target_area_mm2=first.target_area_mm2,
    )


def attribute_uncovered(leak_points, leak_normals, poses: Sequence[ScannerPose],
                        station: Station, model: ScannerModel, cfg: PlannerConfig,
                        *, obstacles: dict | None = None,
                        tracker_ok=None) -> np.ndarray:
    """Say why each uncovered sample is uncovered, one reason per sample.

    `only_poses_that_collide` is kept separate from `blocked_by_*`: it means the part is
    reachable in principle but the scanner body has nowhere to stand, which is a layout
    question, not an occlusion one.

    `tracker_ok` is the per-pose tracker verdict from :func:`evaluate`. Passing it matters:
    a pose the tracker cannot see covers nothing, so without it a sample that only
    tracker-rejected poses could reach would be reported as `candidate_found`, which names
    the wrong cause and hides a mounting problem behind an algorithm one. Poses the tracker
    CAN see are answered first, and the ray is traced even for the others, so a genuine
    occlusion still outranks the tracker as the more specific explanation.
    """
    points = np.asarray(leak_points, dtype=float).reshape(-1, 3)
    normals = np.asarray(leak_normals, dtype=float).reshape(-1, 3)
    count = len(points)
    if len(normals) != count:
        raise ValueError("leak points and normals must be the same length")
    reasons = np.full(count, "never_in_view", dtype=object)
    if count == 0:
        return reasons

    casters = station.obstacles() if obstacles is None else obstacles
    resolved = np.zeros(count, dtype=bool)
    only_colliding = np.zeros(count, dtype=bool)
    pending = np.arange(count)
    base_positions = np.array([pose.position for pose in poses], dtype=float) if poses \
        else np.zeros((0, 3))
    clearances = (station.clearance_mm(base_positions) if len(base_positions)
                  else np.zeros(0))
    if tracker_ok is None:
        visible = np.ones(len(poses), dtype=bool)
    else:
        visible = np.asarray(tracker_ok, dtype=bool)
        if len(visible) != len(poses):
            raise ValueError("one tracker verdict per pose is required")
    # answer from the poses the tracker can actually see first, so their verdicts win
    order = np.argsort(~visible, kind="stable")

    for index in order:
        pose = poses[index]
        if resolved.all():
            break
        candidates = pending[~resolved[pending]]
        if len(candidates) == 0:
            break
        geometry = model.geometry_mask(pose, points[candidates], normals[candidates])
        index_set = candidates[geometry]
        if len(index_set) == 0:
            continue
        if not model.body_fits(pose.position, clearances[index]):
            only_colliding[index_set] = True
            continue

        offset = points[index_set] - np.asarray(pose.position, dtype=float)
        distance = np.linalg.norm(offset, axis=1)
        directions = offset / np.maximum(distance, 1e-9)[:, None]
        origins = np.repeat(np.asarray(pose.position, dtype=float)[None, :],
                            len(index_set), axis=0)
        nearest, names, _ = nearest_across(casters, origins, directions)
        blocked = nearest < distance - cfg.occlusion_tol_mm
        for local, target in enumerate(index_set):
            if blocked[local]:
                reasons[target] = f"blocked_by_{names[local]}"
            else:
                reasons[target] = "candidate_found" if visible[index] else "blocked_by_tracker"
            resolved[target] = True

    reasons[~resolved & only_colliding] = "only_poses_that_collide"
    return reasons
