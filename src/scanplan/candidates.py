"""Generating candidate scanner poses, coarsely and then finely.

For each aim point take several working distances, tilt the optical axis a few
degrees off the surface normal, and emit the rolls the roll policy asks for - the
roll matters because the field of view is a rectangle, not a disc.

Candidates are generated without regard to occlusion or reachability, then filtered.
Filtering first and planning second is the right order: a viewpoint discovered to be
unusable after selection leaves a hole that cannot be patched without planning again.

The joint gate is applied HERE, before the pool is capped, and that ordering is the
whole point of this stage's filter.  The cap is far below the natural pool (thousands
of aims times dozens of poses), so filtering after the cap spends the budget on poses
the arm can never hold: on the 2026-10-28 delivery 786 of 4000 candidates were rejected
by the gate after the fact, i.e. a fifth of the ray budget described undeliverable poses.
Filtering here also turns "how much can this cell reach" into a question about poses the
arm can hold, which is the question the selection stage is actually asking.

Density is what buys coverage. A pool of one pose per target point caps coverage far
below what the geometry allows, because one pose covers its whole footprint. Measured on
a real part, per-point reachability was about 96% while the pool as a whole reached about
99.6% - so the gap between achieved coverage and the pool's ceiling is a
candidate-generation question, not a set-cover one. Refine towards the still-uncovered
region before concluding anything is unreachable, and seed that refinement too.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .config import ScannerParams
from .geometry import grid_dedupe
from .policies import RollPolicy, gate_poses, roll_policy
from .policies import ladder as policy_ladder
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


def ladder(point, normal, params: ScannerParams, *,
           policy: RollPolicy | None = None) -> list[ScannerPose]:
    """The poses aimed at one surface point, at the rolls `policy` allows.

    Kept as a named function because it is the unit the tests probe; the roll decision
    itself belongs to the policy (slot A).
    """
    return policy_ladder(point, normal, params, policy=policy)


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


def cap(groups: list[list[ScannerPose]], limit: int) -> list[ScannerPose]:
    """Cut the pool to `limit` poses WITHOUT starving any aim point.

    A uniform random cut is what the pool used to do, and it is the wrong shape of cut: the
    cap is far below the natural pool, so a random sample leaves whole regions of the part
    with no candidate at all - and a region with no candidate can never be covered, however
    good the selection is. Taking one pose per aim in turn keeps every aim point
    represented, then spends the rest on the aims that still have poses left.

    Deterministic by construction: the old random cut made two runs of the same seed agree
    only by luck of iteration order, and a plan that cannot be re-created is a plan that
    cannot be reviewed.
    """
    total = sum(len(group) for group in groups)
    if limit <= 0 or total <= limit:
        return [pose for group in groups for pose in group]
    picked: list[ScannerPose] = []
    cursor = [0] * len(groups)
    while len(picked) < limit:
        moved = False
        for index, group in enumerate(groups):
            if len(picked) >= limit:
                break
            if cursor[index] < len(group):
                picked.append(group[cursor[index]])
                cursor[index] += 1
                moved = True
        if not moved:
            break
    return picked


def _limited(params: ScannerParams, max_candidates: int | None) -> int:
    limit = int(params.max_candidates if max_candidates is None else max_candidates)
    if limit <= 0:
        raise ValueError("max_candidates must be positive")
    return limit


def generate(points, normals, params: ScannerParams, *, seed: int,
             aim_grid_mm: float | None = None, max_candidates: int | None = None,
             keep: Callable[[ScannerPose], bool] | None = None,
             policy: RollPolicy | None = None,
             progress: Callable[[int, int], None] | None = None,
             log: Callable[[str], None] | None = None) -> list[ScannerPose]:
    """Build the candidate pool: a ladder on every aim cell of the target.

    `keep` is the joint gate, applied before the cap - see this module's docstring for
    why that order is the stage's contract rather than an optimisation.
    """
    say = log or (lambda _message: None)
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    normals = np.asarray(normals, dtype=float).reshape(-1, 3)
    if len(points) == 0:
        return []
    if len(points) != len(normals):
        raise ValueError("points and normals must be the same length")

    policy = policy or roll_policy(getattr(params, "roll_mode", "locked"))
    grid = float(params.aim_grid_mm if aim_grid_mm is None else aim_grid_mm)
    aims = aim_indices(points, grid)
    limit = _limited(params, max_candidates)

    groups: list[list[ScannerPose]] = []
    for done, index in enumerate(aims):
        poses = dedupe(ladder(points[index], normals[index], params, policy=policy))
        groups.append(gate_poses(poses, keep, policy=policy))
        if progress is not None:
            progress(done + 1, len(aims))
    pool = cap(groups, limit)
    say(f"  {len(aims)} aim points on a {grid:g} mm grid, rolls {policy.name}"
        f" -> {len(pool)} candidates")
    return pool


def refine(leak_points, leak_normals, params: ScannerParams, *, seed: int,
           max_candidates: int | None = None,
           keep: Callable[[ScannerPose], bool] | None = None,
           policy: RollPolicy | None = None,
           progress: Callable[[int, int], None] | None = None,
           log: Callable[[str], None] | None = None) -> list[ScannerPose]:
    """Aim a new ladder at what is still uncovered, on a finer grid than the face.

    The leak region is small by definition, and the measured viewpoint count is limited
    by candidate generation rather than by the selection stage, so this is where a
    better answer comes from. The refinement pool still honours the same cap as the
    first pass - a safety net rather than a limit, since a leak on a 25 mm aim grid
    stays well below it.
    """
    say = log or (lambda _message: None)
    leak_points = np.asarray(leak_points, dtype=float).reshape(-1, 3)
    leak_normals = np.asarray(leak_normals, dtype=float).reshape(-1, 3)
    if len(leak_points) == 0:
        return []
    if len(leak_points) != len(leak_normals):
        raise ValueError("leak points and normals must be the same length")

    policy = policy or roll_policy(getattr(params, "roll_mode", "locked"))
    aims = aim_indices(leak_points, params.refine_aim_grid_mm)
    limit = _limited(params, max_candidates)
    groups: list[list[ScannerPose]] = []
    for done, index in enumerate(aims):
        poses = dedupe(ladder(leak_points[index], leak_normals[index], params,
                              policy=policy))
        groups.append(gate_poses(poses, keep, policy=policy))
        if progress is not None:
            progress(done + 1, len(aims))
    pool = cap(groups, limit)
    say(f"  refinement: {len(aims)} leak aim points on a "
        f"{params.refine_aim_grid_mm:g} mm grid -> {len(pool)} candidates")
    return pool
