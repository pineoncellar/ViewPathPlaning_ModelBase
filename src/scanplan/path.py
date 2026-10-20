"""Turn a chosen set of strokes into one ordered, connected TCP path.

A stroke is an unordered, undirected segment; a program is a sequence. Closing that
gap is the whole job of this module:

* **order** - a greedy nearest-neighbour pass over the strokes, choosing which end to
  enter by, then single-stroke relocation passes. Not an optimum, and not claimed to
  be: what the workshop feels is the number of corners, not the last millimetre of
  travel.
* **connect** - between two strokes the head has to get from one to the other without
  dragging the scanner through the part. Two cases, and they are genuinely different:
  same pose (a translation: retract along -axis, traverse, dive back in) and different
  pose (add a re-orientation while retracted).
* **account** - turn count, travel length, measurement interrupts, and the smallest
  clearance margin anywhere on the path.

Ordering lives here and is written into the delivered file. It must not be redone at
export time: re-sorting a stroke plan silently splits strokes into two unrelated
moves, and the sweep the coverage figure was computed for never happens. See
`scanplan.roboguide.adapt.export`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .config import PlannerConfig
from .geometry import unit
from .scanner_model import ScannerModel, ScannerPose
from .scene import Station
from .strokes import Stroke

# Two poses count as the same when the axis and the roll agree this closely, which is
# the point where a translation-only connector stays valid.
_SAME_POSE_DEG = 0.5
# Multiples of `retract_clearance_mm` the connector tries before giving up. A fixed lift
# assumes the space behind the head is empty, and on a real cell it is not: the base sits
# behind the upper face and the fixture behind the lower one.
_LIFT_FACTORS = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0)

ROLE_COVERAGE = "coverage"
ROLE_RETRACT = "retract"
ROLE_TRAVERSE = "traverse"
ROLE_REORIENT = "reorient"


@dataclass(frozen=True)
class PathPoint:
    """One TCP target in program order.

    `coverage_index` is the row of the combined coverage object that this point's sweep
    is described by, or None for a point on a transition. It is what lets the report and
    the 3D view say "this corner covers these samples" without re-deriving anything.
    """

    position: np.ndarray
    axis: np.ndarray
    roll_deg: float
    role: str
    source: str
    stroke_id: int | None = None
    coverage_index: int | None = None


@dataclass
class PathPlan:
    """The ordered path plus everything the report and the acceptance criteria need."""

    points: list[PathPoint]
    interrupts_a: int = 0
    interrupts_b: int = 0
    path_length_mm: float = 0.0
    min_clearance_mm: float | None = None
    violations: list[str] = field(default_factory=list)
    ordering: str = ""

    @property
    def turns(self) -> int:
        """How many corners the head has to stop at."""
        return len(self.points)

    @property
    def coverage_points(self) -> int:
        """How many of those corners are part of a scanning sweep."""
        return sum(1 for point in self.points if point.role == ROLE_COVERAGE)

    @property
    def transition_points(self) -> int:
        """How many are there only to get the head from one sweep to the next."""
        return len(self.points) - self.coverage_points


def _same_pose(first: PathPoint, second: PathPoint) -> bool:
    cosine = float(np.clip(np.dot(unit(first.axis), unit(second.axis)), -1.0, 1.0))
    angle = float(np.degrees(np.arccos(cosine)))
    roll = abs((float(first.roll_deg) - float(second.roll_deg) + 180.0) % 360.0 - 180.0)
    return angle <= _SAME_POSE_DEG and roll <= _SAME_POSE_DEG


def _rotation_deg(first: PathPoint, second: PathPoint) -> float:
    cosine = float(np.clip(np.dot(unit(first.axis), unit(second.axis)), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def order_strokes(strokes: Sequence[Stroke], start_near, *,
                  max_passes: int = 2) -> list[tuple[int, bool]]:
    """Order the strokes and choose an end for each: `(index, reversed)` in path order.

    Greedy nearest neighbour, then single-stroke relocation. The relocation pass is
    what a 2-opt cannot be here: reversing a run of strokes also reverses each of them,
    which changes every connector inside the run, so the usual "only two edges change"
    argument does not hold. Moving one stroke keeps the rest of the sequence intact and
    the improvement can be priced exactly.
    """
    count = len(strokes)
    if count == 0:
        return []
    if count == 1:
        return [(0, False)]
    starts = np.array([np.asarray(stroke.start, dtype=float) for stroke in strokes])
    ends = np.array([np.asarray(stroke.end, dtype=float) for stroke in strokes])
    anchor = np.asarray(start_near, dtype=float).reshape(3)

    remaining = list(range(count))
    order: list[tuple[int, bool]] = []
    cursor = anchor
    while remaining:
        best = None
        for index in remaining:
            for flip in (False, True):
                entry = ends[index] if flip else starts[index]
                cost = float(np.linalg.norm(entry - cursor))
                if best is None or cost < best[0]:
                    best = (cost, index, flip)
        _, index, flip = best
        order.append((index, flip))
        cursor = starts[index] if flip else ends[index]
        remaining.remove(index)

    order = _relocate(order, starts, ends, anchor, max_passes)
    return order


def _path_endpoints(order, starts, ends, anchor):
    """The entry and exit position of every stroke in the given order."""
    entry = np.array([ends[i] if flip else starts[i] for i, flip in order])
    exit_ = np.array([starts[i] if flip else ends[i] for i, flip in order])
    return entry, exit_


def _connector_length(order, starts, ends, anchor) -> float:
    entry, exit_ = _path_endpoints(order, starts, ends, anchor)
    if len(order) == 0:
        return 0.0
    total = float(np.linalg.norm(entry[0] - anchor))
    if len(order) > 1:
        total += float(np.linalg.norm(entry[1:] - exit_[:-1], axis=1).sum())
    return total


def _relocate(order, starts, ends, anchor, max_passes: int) -> list[tuple[int, bool]]:
    """Move one stroke at a time to a better slot, either way round, while it helps."""
    if len(order) < 3 or max_passes <= 0:
        return order
    best = list(order)
    best_cost = _connector_length(best, starts, ends, anchor)
    for _ in range(max_passes):
        improved = False
        for source in range(len(best)):
            without = best[:source] + best[source + 1:]
            item = best[source]
            for target in range(len(without) + 1):
                for flip in (item[1], not item[1]):
                    candidate = without[:target] + [(item[0], flip)] + without[target:]
                    cost = _connector_length(candidate, starts, ends, anchor)
                    if cost < best_cost - 1e-9:
                        best, best_cost, improved = candidate, cost, True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return best


def build_path(strokes: Sequence[Stroke], *, cfg: PlannerConfig, station: Station,
               model: ScannerModel, coverage_index=None, sources=None,
               start_near=None,
               visible: Callable[[np.ndarray], np.ndarray] | None = None,
               log: Callable[[str], None] | None = None) -> PathPlan:
    """Order `strokes`, insert the transitions, and account for the result.

    `coverage_index(i)` gives the coverage row that stroke `i` is described by, so each
    delivered point can point at the samples its sweep covers. `sources(i)` says whether
    the stroke came from the greedy pass or from repair.
    """
    say = log or (lambda _message: None)
    if not strokes:
        return PathPlan(points=[], ordering="empty")
    if start_near is None:
        start_near = np.asarray(strokes[0].start, dtype=float)
    order = order_strokes(strokes, start_near)
    points: list[PathPoint] = []
    plan = PathPlan(points=points, ordering=(
        "nearest neighbour + single-stroke relocation over stroke endpoints"))

    previous: PathPoint | None = None
    for index, flip in order:
        stroke = strokes[index]
        start = np.asarray(stroke.end if flip else stroke.start, dtype=float)
        end = np.asarray(stroke.start if flip else stroke.end, dtype=float)
        row = None if coverage_index is None else coverage_index(index)
        source = "greedy" if sources is None else str(sources(index))
        axis = np.asarray(stroke.axis, dtype=float)
        entry = PathPoint(position=start, axis=axis, roll_deg=float(stroke.roll_deg),
                          role=ROLE_COVERAGE, source=source, stroke_id=int(index),
                          coverage_index=row)
        if previous is not None:
            points.extend(_connector(previous, entry, cfg, plan, station, model,
                                     visible=visible))
        points.append(entry)
        points.append(PathPoint(position=end, axis=axis, roll_deg=float(stroke.roll_deg),
                                role=ROLE_COVERAGE, source=source,
                                stroke_id=int(index), coverage_index=row))
        previous = points[-1]

    plan.path_length_mm = _travel(points)
    say(f"  path: {len(strokes)} strokes -> {plan.turns} corners "
        f"({plan.transition_points} transitions), {plan.path_length_mm:.0f} mm, "
        f"{plan.interrupts_a} translation interrupts, {plan.interrupts_b} re-orientations")
    return plan


def _travel(points: Sequence[PathPoint]) -> float:
    """Total TCP travel along the delivered sequence."""
    if len(points) < 2:
        return 0.0
    positions = np.array([np.asarray(point.position, dtype=float) for point in points])
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())


def _segment_margin(first: PathPoint, second: PathPoint, cfg: PlannerConfig,
                    station: Station, model: ScannerModel, *, visible=None
                    ) -> tuple[float, bool]:
    """Smallest body clearance along one short move, and whether the tracker held.

    Sampled at the cheap step: a transition is not a scan, so what matters is that the
    head fits everywhere on it, not what it can see.
    """
    start = np.asarray(first.position, dtype=float)
    end = np.asarray(second.position, dtype=float)
    length = float(np.linalg.norm(end - start))
    count = max(1, int(np.ceil(length / cfg.stroke.check_step_mm)))
    fractions = np.linspace(0.0, 1.0, count + 1)[:, None]
    positions = start[None, :] * (1.0 - fractions) + end[None, :] * fractions
    margin = float(station.clearance_mm(positions).min() - model.body_limit_mm)
    seen = True if visible is None else bool(np.all(visible(positions)))
    return margin, seen


def _connector(previous: PathPoint, entry: PathPoint, cfg: PlannerConfig, plan: PathPlan,
               station: Station, model: ScannerModel, *,
               visible=None) -> list[PathPoint]:
    """The points that take the head from `previous` to `entry` without cutting a corner.

    Both cases retract along -axis first, because the part is in front of the head: a
    straight line between two poses would sweep the head through whatever is between
    them, and the scanner body is behind its reference point.

    The retract distance is not taken on trust. A fixed 50 mm assumes the space behind the
    head is empty, and on this cell it is not - behind the upper face is the base, and
    behind the lower face is the floor side of the fixture - so the lift is raised until
    the three moves are clear of both AND the rotation between the two poses is spread over
    enough travel to stay inside the rate limit. If even the largest lift cannot do it, the
    largest is used and the reason is recorded rather than hidden.
    """
    same = _same_pose(previous, entry)
    rotation = _rotation_deg(previous, entry)
    best: tuple[float, list[PathPoint], float] | None = None
    for factor in _LIFT_FACTORS:
        lift = float(cfg.stroke.retract_clearance_mm) * factor
        retract = PathPoint(
            position=np.asarray(previous.position, dtype=float)
            - np.asarray(previous.axis, dtype=float) * lift,
            axis=np.asarray(previous.axis, dtype=float), roll_deg=float(previous.roll_deg),
            role=ROLE_RETRACT, source=str(previous.source), stroke_id=previous.stroke_id)
        through = PathPoint(
            position=np.asarray(entry.position, dtype=float)
            - np.asarray(entry.axis, dtype=float) * lift,
            axis=np.asarray(previous.axis, dtype=float),
            roll_deg=float(previous.roll_deg),
            role=ROLE_TRAVERSE if same else ROLE_REORIENT,
            source=str(entry.source), stroke_id=entry.stroke_id)
        reach = float(np.linalg.norm(np.asarray(through.position, dtype=float)
                                     - np.asarray(retract.position, dtype=float)))
        rate = rotation / reach if reach > 0.0 else float("inf")
        margin = np.inf
        seen = True
        for first, second in ((previous, retract), (retract, through), (through, entry)):
            found, tracker_ok = _segment_margin(first, second, cfg, station, model,
                                                visible=visible)
            margin = min(margin, found)
            seen = seen and tracker_ok
        rate_ok = same or rate <= cfg.stroke.max_rot_deg_per_mm
        tried = (margin, [retract, through], rate if not rate_ok else 0.0)
        # keep the roomiest attempt seen so far: a bigger lift is not monotonically safer,
        # because retracting away from the part can walk towards the base behind it
        if best is None or margin > best[0]:
            best = tried
        if margin > 0.0 and rate_ok and seen:
            best = tried
            break

    margin, points, rate = best
    plan.min_clearance_mm = (margin if plan.min_clearance_mm is None
                             else min(plan.min_clearance_mm, margin))
    if same:
        plan.interrupts_a += 1
    else:
        plan.interrupts_b += 1
        if rate:
            plan.violations.append(
                f"re-orientation of {rotation:.0f} deg cannot be spread under "
                f"{cfg.stroke.max_rot_deg_per_mm:g} deg/mm within "
                f"{_LIFT_FACTORS[-1] * cfg.stroke.retract_clearance_mm:g} mm of lift")
    if margin <= 0.0:
        plan.violations.append(
            f"the connector into stroke {points[1].stroke_id} keeps {margin:.1f} mm of body "
            f"clearance at every lift tried, up to "
            f"{_LIFT_FACTORS[-1] * cfg.stroke.retract_clearance_mm:g} mm")
    return points


def transition_poses(plan: PathPlan, cfg: PlannerConfig) -> list[ScannerPose]:
    """Poses along the transition segments, for the "what did the path sweep" figure.

    The pose during a move is interpolated by the controller and is not modelled here;
    the destination pose is used, which can only overstate the incidence angle - the
    safe direction for a number that is reported and never used for acceptance.
    """
    poses: list[ScannerPose] = []
    step = float(cfg.stroke.coverage_step_mm)
    for first, second in zip(plan.points, plan.points[1:], strict=False):
        if first.role == ROLE_COVERAGE and second.role == ROLE_COVERAGE:
            continue
        start = np.asarray(first.position, dtype=float)
        end = np.asarray(second.position, dtype=float)
        length = float(np.linalg.norm(end - start))
        count = max(1, int(np.ceil(length / step)))
        fractions = np.linspace(0.0, 1.0, count + 1)[1:, None]
        positions = start[None, :] * (1.0 - fractions) + end[None, :] * fractions
        axis = np.asarray(second.axis, dtype=float)
        for position in positions:
            poses.append(ScannerPose(position=position, axis=axis,
                                     roll_deg=float(second.roll_deg)))
    return poses


def prune(strokes: Sequence[Stroke], sets: Sequence[np.ndarray], chosen: Sequence[int],
          area_mm2, target_area_mm2: float, *, target_pct: float,
          max_rounds: int) -> list[int]:
    """Drop strokes the target does not need, cheapest contribution first.

    `target_pct` is the requirement (99%), not the achieved peak: pruning against the
    peak would be a no-op by construction.

    Contributions are priced from a per-sample "how many kept strokes cover me" count, so
    a stroke's own share is the area where that count is exactly one. Removing it is then
    one decrement rather than a fresh union over the whole set, which is what keeps the
    loop cheap enough to run in rounds. No rays are cast here, so only the ordering has to
    be rebuilt afterwards - and that is cheap too.
    """
    if not chosen or target_area_mm2 <= 0.0:
        return list(chosen)
    area = np.asarray(area_mm2, dtype=float)
    target = float(target_pct) * target_area_mm2 / 100.0
    kept = list(chosen)
    counts = np.zeros(int(len(area)), dtype=np.int32)
    for index in kept:
        counts[sets[index]] += 1
    total = float(area[counts > 0].sum())
    for _ in range(max(0, int(max_rounds))):
        if total <= target or not kept:
            break
        best: tuple[float, int] | None = None
        for index in kept:
            row = sets[index]
            lost = float(area[row[counts[row] == 1]].sum())
            if total - lost < target:
                continue
            if best is None or lost < best[0]:
                best = (lost, index)
        if best is None:
            break
        lost, index = best
        counts[sets[index]] -= 1
        total -= lost
        kept.remove(index)
    return kept
