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
from .rolls import roll_for_axis
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


def ladder(point, normal, params: ScannerParams, *, lock_roll: bool = False
           ) -> list[ScannerPose]:
    """The poses aimed at one surface point: every distance, tilt, azimuth and roll.

    Every tilt gets every roll, including zero tilt. The roll does NOT change the optical
    axis, but the field of view is a rectangle (600 x 500 mm at the standoff), so a quarter
    turn about that axis covers a different set of samples - and zero tilt is the pose with
    the best incidence, i.e. exactly the one a selection prefers. Design 4.2 lists two
    rolls for this reason.

    `tilt_azimuths` sweeps the tilt around the surface normal: a tilt only about one tangent
    (+30 deg) and its opposite (-30 deg) describe a single plane, so a sample can only be
    approached from two of the (infinitely many) directions around it. Measured on the
    2026-10-28 delivery, 91 % of the uncovered samples had a feasible pose somewhere in a
    four-azimuth fan - the arm simply could not hold the poses the one-plane ladder offered.
    One azimuth reproduces the old behaviour exactly.

    The sweep spans 180 degrees and emits BOTH tilt signs about each azimuth: the signed
    pair then names the two opposite axes for that azimuth, so `azimuths` rotations cover the
    full circle twice over - no direction is missed and none is emitted twice. The head-on
    pose does not depend on the azimuth at all, so it is emitted once.
    """
    point = np.asarray(point, dtype=float).reshape(3)
    normal = np.asarray(normal, dtype=float).reshape(3)
    normal = normal / np.linalg.norm(normal)
    first_tangent, second_tangent = tangents(normal)
    inward = -normal
    azimuths = max(1, int(params.tilt_azimuths))

    poses: list[ScannerPose] = []
    for distance in params.distances_mm:
        position = point + normal * distance
        for index in range(azimuths):
            turn = np.pi * index / azimuths
            tangent = np.cos(turn) * first_tangent + np.sin(turn) * second_tangent
            for tilt_deg in params.tilts_deg:
                if tilt_deg == 0.0:
                    if index > 0:
                        continue
                    axis = inward
                else:
                    axis = tilt_axis(inward, tangent, float(tilt_deg))
                if lock_roll:
                    # One roll convention for the whole program: the roll the strokes are
                    # grown with (`rolls.roll_for_axis`). Mixing a pose at roll 0 with a
                    # sweep at roll 90 puts a 90 deg J6 turn in the middle of the program
                    # for nothing - measured on 2026-10-29 as four J6 steps of 74-95 deg.
                    poses.append(ScannerPose(position=position.copy(), axis=axis.copy(),
                                             roll_deg=roll_for_axis(axis)))
                    continue
                for roll_deg in params.rolls_deg:
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


# Rolls tried about the SAME optical axis when the joint gate refuses the pose's own roll.
# A spin about the optical axis does not change what the scanner looks at, but it does move
# the FLANGE: the fitted tool is 635 mm long, so the roll sweeps the wrist round a 0.6 m
# circle and decides whether the arm can hold the pose at all. `stroke.lock_roll` pins one
# roll to keep J6 still, which is right for the many poses where that roll is feasible - but
# on station 20261030_105857 it was infeasible for 1.4 % of the target face, and those
# points came back as "never aimed at" when the sight line was fine all along. 21 of 30
# sampled leftovers were reachable at another roll about the same axis.
_ROLL_RETRY_DEG = (45.0, 315.0, 90.0, 270.0, 135.0, 225.0, 180.0)


def gate_with_roll_retry(poses, keep) -> list[ScannerPose]:
    """Apply `keep` (the joint gate), re-trying a refused pose at other rolls.

    Only the roll changes, so the coverage geometry of a retried pose is the same sight
    line; what buys the pose is the wrist position, not the aim. The smallest turn that
    works is taken first so a retried pose stays as close as possible to the roll its
    neighbours use (a roll difference between two delivered points is a J6 turn). A pose
    the gate accepts first time is passed through untouched, so a plan whose locked roll
    already works is bit-for-bit the plan it was before.
    """
    if keep is None:
        return list(poses)
    kept: list[ScannerPose] = []
    for pose in poses:
        if keep(pose):
            kept.append(pose)
            continue
        for turn in _ROLL_RETRY_DEG:
            retried = ScannerPose(position=pose.position, axis=pose.axis,
                                  roll_deg=(pose.roll_deg + turn) % 360.0)
            if keep(retried):
                kept.append(retried)
                break
    return kept


def _cap(groups: list[list[ScannerPose]], limit: int) -> list[ScannerPose]:
    """Cut the pool to `limit` poses WITHOUT starving any aim point.

    A uniform random cut is what the pool used to do, and it is the wrong shape of cut: the
    cap is far below the natural pool (thousands of aims x dozens of poses), so a random
    sample leaves whole regions of the part with no candidate at all - and a region with no
    candidate can never be covered, however good the selection is. Taking one pose per aim
    in turn keeps every aim point represented, then spends the rest on the aims that still
    have poses left.

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


def generate(points, normals, params: ScannerParams, *, seed: int,
             aim_grid_mm: float | None = None, max_candidates: int | None = None,
             keep: Callable[[ScannerPose], bool] | None = None,
             lock_roll: bool = False,
             progress: Callable[[int, int], None] | None = None,
             log: Callable[[str], None] | None = None) -> list[ScannerPose]:
    """Build the candidate pool: a ladder on every aim cell of the target.

    `keep` is the joint gate, applied HERE, before the cap. Order matters: the cap is far
    below the natural pool, so filtering afterwards would spend the budget on poses the arm
    can never hold. On the 2026-10-28 delivery 786 of 4000 candidates were rejected by the
    gate after the fact - i.e. a fifth of the ray budget was spent describing poses that
    could not be delivered.
    """
    say = log or (lambda _message: None)
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    normals = np.asarray(normals, dtype=float).reshape(-1, 3)
    if len(points) == 0:
        return []
    if len(points) != len(normals):
        raise ValueError("points and normals must be the same length")

    grid = float(params.aim_grid_mm if aim_grid_mm is None else aim_grid_mm)
    aims = aim_indices(points, grid)
    groups: list[list[ScannerPose]] = []
    for done, index in enumerate(aims):
        ladder_poses = dedupe(ladder(points[index], normals[index], params,
                                     lock_roll=lock_roll))
        groups.append(gate_with_roll_retry(ladder_poses, keep))
        if progress is not None:
            progress(done + 1, len(aims))

    limit = int(params.max_candidates if max_candidates is None else max_candidates)
    if limit <= 0:
        raise ValueError("max_candidates must be positive")
    pool = _cap(groups, limit)
    say(f"  {len(aims)} aim points on a {grid:g} mm grid x "
        f"{max(1, int(params.tilt_azimuths))} azimuth(s) -> {len(pool)} candidates")
    return pool


def refine(leak_points, leak_normals, params: ScannerParams, *, seed: int,
           max_candidates: int | None = None,
           keep: Callable[[ScannerPose], bool] | None = None,
           lock_roll: bool = False,
           progress: Callable[[int, int], None] | None = None,
           log: Callable[[str], None] | None = None) -> list[ScannerPose]:
    """Aim a new ladder at what is still uncovered, on a finer grid than the face.

    The leak region is small by definition, and the measured viewpoint count is limited
    by candidate generation rather than by the selection stage, so this is where a
    better answer comes from. The refinement pool honours the same cap as the first pass
    and the same rules: `keep` (the joint gate) filters BEFORE the cap, and the cap takes
    one pose per aim in turn so a crowded corner cannot starve the rest of the leak.

    `seed` is accepted for call-site symmetry and no longer changes the answer: the cap is
    a deterministic round-robin, so two runs of the same inputs agree by construction
    rather than by luck of iteration order.
    """
    say = log or (lambda _message: None)
    leak_points = np.asarray(leak_points, dtype=float).reshape(-1, 3)
    leak_normals = np.asarray(leak_normals, dtype=float).reshape(-1, 3)
    if len(leak_points) == 0:
        return []
    if len(leak_points) != len(leak_normals):
        raise ValueError("leak points and normals must be the same length")

    aims = aim_indices(leak_points, params.refine_aim_grid_mm)
    groups: list[list[ScannerPose]] = []
    for done, index in enumerate(aims):
        ladder_poses = dedupe(ladder(leak_points[index], leak_normals[index], params,
                                     lock_roll=lock_roll))
        groups.append(gate_with_roll_retry(ladder_poses, keep))
        if progress is not None:
            progress(done + 1, len(aims))
    limit = int(params.max_candidates if max_candidates is None else max_candidates)
    pool = _cap(groups, limit)
    say(f"  refinement: {len(aims)} leak aim points on a "
        f"{params.refine_aim_grid_mm:g} mm grid -> {len(pool)} candidates")
    return pool


def feature_ladder(point, normal, params: ScannerParams, *,
                   lock_roll: bool = False) -> list[ScannerPose]:
    """The poses aimed at ONE hard feature: tilted almost to the incidence limit.

    A wall or a clamped rim is not merely under-served by the head-on ladder, it is
    unreachable by it: the footprint of a head-on pose lies parallel to the wall, so its own
    surface is at 90 degrees of incidence a few millimetres off the bend, and the rim beside
    a clamp is shaded for the same geometric reason. What does reach them is the same ladder
    tilted a long way off the normal from the free side, and that side is not known in
    advance - the clamp is on one side and open air on the other - so the tilt is swept all
    the way round in `feature_tilt_azimuths` steps, both signs.

    The roll is fixed at 0. A tilted pose is already a compromise on incidence, and its
    second roll (90 degrees) would add a 90 degree J6 turn for area the head-on sweep covers
    anyway - the one axis the page asked to keep still.

    The tilt magnitudes come from `feature_tilts_deg`, which the configuration validates to
    stay BELOW `max_incidence_deg`: a tilt at the limit sees the sample it was aimed at at
    zero cosine, i.e. it cannot see it at all.
    """
    point = np.asarray(point, dtype=float).reshape(3)
    normal = np.asarray(normal, dtype=float).reshape(3)
    normal = normal / np.linalg.norm(normal)
    first_tangent, second_tangent = tangents(normal)
    inward = -normal
    azimuths = max(1, int(params.feature_tilt_azimuths))

    poses: list[ScannerPose] = []
    for distance in params.distances_mm:
        position = point + normal * distance
        # head-on first: a leak aim is not always a wall, sometimes the coarse aim grid
        # simply skipped the cell this sample sits in.
        poses.append(ScannerPose(position=position.copy(), axis=inward.copy(),
                                 roll_deg=_feature_roll(inward, lock_roll)))
        for index in range(azimuths):
            turn = np.pi * index / azimuths
            tangent = np.cos(turn) * first_tangent + np.sin(turn) * second_tangent
            for tilt_deg in params.feature_tilts_deg:
                for sign in (1.0, -1.0):
                    axis = tilt_axis(inward, tangent, sign * float(tilt_deg))
                    poses.append(ScannerPose(position=position.copy(), axis=axis.copy(),
                                             roll_deg=_feature_roll(axis, lock_roll)))
    return poses


def _feature_roll(axis, lock_roll: bool) -> float:
    """The roll a hard-feature pose is written with: 0, or the axis's own roll.

    `lock_roll` follows the same convention as every other family, so a tilted sweep does
    not arrive 90 deg twisted relative to its neighbours - a twist is a J6 turn, and the
    page asked for those to stay small.
    """
    return float(roll_for_axis(axis)) if lock_roll else 0.0


def feature_refine(leak_points, leak_normals, params: ScannerParams, *, seed: int,
                   max_candidates: int | None = None,
                   keep: Callable[[ScannerPose], bool] | None = None,
                   lock_roll: bool = False,
                   progress: Callable[[int, int], None] | None = None,
                   log: Callable[[str], None] | None = None) -> list[ScannerPose]:
    """A tilted pool aimed at the hard leftovers, capped the same way as every other pool.

    `keep` (the joint gate) filters BEFORE the cap and the cap takes one pose per aim in
    turn, so a wall whose best tilt the arm cannot hold does not starve its neighbours of
    candidates. The aim grid is `feature_aim_grid_mm`, finer than the face grid because a
    wall is a few millimetres wide.
    """
    say = log or (lambda _message: None)
    leak_points = np.asarray(leak_points, dtype=float).reshape(-1, 3)
    leak_normals = np.asarray(leak_normals, dtype=float).reshape(-1, 3)
    if len(leak_points) == 0:
        return []
    if len(leak_points) != len(leak_normals):
        raise ValueError("leak points and normals must be the same length")

    aims = aim_indices(leak_points, params.feature_aim_grid_mm)
    groups: list[list[ScannerPose]] = []
    for done, index in enumerate(aims):
        poses = dedupe(feature_ladder(leak_points[index], leak_normals[index], params,
                                      lock_roll=lock_roll))
        groups.append(gate_with_roll_retry(poses, keep))
        if progress is not None:
            progress(done + 1, len(aims))
    limit = int(params.max_candidates if max_candidates is None else max_candidates)
    pool = _cap(groups, limit)
    say(f"  hard features: {len(aims)} aim points on a "
        f"{params.feature_aim_grid_mm:g} mm grid x "
        f"{max(1, int(params.feature_tilt_azimuths))} tilt azimuth(s) x "
        f"{len(params.feature_tilts_deg)} tilt(s) -> {len(pool)} candidates")
    return pool
