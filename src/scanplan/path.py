"""Turn a chosen set of strokes into one ordered, connected TCP path.

A stroke is an unordered, undirected segment; a program is a sequence. Closing that
gap is the whole job of this module:

* **order** - a greedy nearest-neighbour pass over the items, choosing which end to
  enter by, then single-stroke relocation passes. Not an optimum, and not claimed to
  be: what the workshop feels is the number of corners, not the last millimetre of
  travel. Distance alone is not the whole cost either - a short move between two
  unrelated poses costs a wrist turn, so the pose change is priced in as well.
* **connect** - between two items the head has to get from one to the other without
  dragging the scanner through the part. Three routes, cheapest first: straight there
  (only between two identical poses), one retract with the rotation spread over the
  flight, or retract plus a re-orientation point.
* **account** - turn count, travel length, measurement interrupts, which route each
  transition took, and the smallest clearance margin anywhere on the path.

An item is a stroke or a single pose: a pose is exactly the stroke of length zero, and it
is delivered as one corner. Keeping that identity is what makes the reported coverage and
the program's actual sweep the same number - the measured run promised 99.741% while its
delivered strokes swept 91.98%, because the selected poses were counted and then dropped.

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
# Shorter than this and the item is a pose, not a sweep: delivering both of its ends would
# put two identical corners in the program.
_ZERO_MM = 1e-9
# Multiples of `retract_clearance_mm` the connector tries before giving up. A fixed lift
# assumes the space behind the head is empty, and on a real cell it is not: the base sits
# behind the upper face and the fixture behind the lower one.
_LIFT_FACTORS = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0)

ROLE_COVERAGE = "coverage"
ROLE_RETRACT = "retract"
ROLE_TRAVERSE = "traverse"
ROLE_REORIENT = "reorient"

# The three connector routes, cheapest first. Reported per transition so "the connector
# got cheaper" is a measurement rather than a claim.
LEVEL_STRAIGHT = "L0-straight"
LEVEL_ONE_CORNER = "L1-retract"
LEVEL_TWO_CORNERS = "L2-retract-through"


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
    # How many of the delivered items were real sweeps rather than single poses. A
    # delivered pose is a zero-length stroke, so counting distinct `stroke_id`s would
    # report poses as strokes.
    coverage_strokes: int = 0
    # Which connector route each transition actually took. Reported because "the
    # connector got cheaper" is otherwise an unfalsifiable claim.
    connector_levels: dict[str, int] = field(default_factory=dict)

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
    """How far the head has to turn between two corners, roll included.

    Roll belongs in here. `_same_pose` already treats a roll-only change as a different
    pose, so leaving it out of the angle made a pure roll cost nothing to order and pass
    the rate check while still being counted as a re-orientation interrupt.
    """
    return pose_gap_deg(first.axis, first.roll_deg, second.axis, second.roll_deg)


def pose_gap_deg(axis_a, roll_a: float, axis_b, roll_b: float) -> float:
    """The angle of the rotation between two scanner poses, as one number.

    Swing from A's axis to B's, then the roll difference as a twist about the new axis.
    This reduces exactly to the roll difference when the axes agree and to the axis angle
    when the rolls agree, which is what makes it a superset of the old axis-only measure.

    Deliberately NOT built from `scanner_frame`. That frame picks its reference axis from
    whichever of +Z and +X is not too parallel to the optical axis, so two nearly identical
    axes on either side of that switch produce frames that differ by about 90 degrees. The
    metric must not depend on an arbitrary reference, and this one does not.
    """
    first = unit(np.asarray(axis_a, dtype=float).reshape(3))
    second = unit(np.asarray(axis_b, dtype=float).reshape(3))
    cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
    swing = float(np.arccos(cosine))
    twist = np.deg2rad(float(roll_b) - float(roll_a))

    normal = np.cross(first, second)
    length = float(np.linalg.norm(normal))
    if length <= 1e-9:
        # Parallel or antiparallel: the swing axis is arbitrary, and for the antiparallel
        # case any perpendicular will do because every such swing is 180 degrees.
        helper = np.array([1.0, 0.0, 0.0]) if abs(first[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        normal = np.cross(first, helper)
        length = float(np.linalg.norm(normal))
    total = _rotation(second, twist) @ _rotation(normal / length, swing)
    return float(np.degrees(np.arccos(np.clip((float(np.trace(total)) - 1.0) / 2.0,
                                              -1.0, 1.0))))


def _rotation(axis, angle: float) -> np.ndarray:
    """Rodrigues' rotation matrix about a unit `axis` by `angle` radians."""
    unit_axis = unit(np.asarray(axis, dtype=float).reshape(3))
    skew = np.array([[0.0, -unit_axis[2], unit_axis[1]],
                     [unit_axis[2], 0.0, -unit_axis[0]],
                     [-unit_axis[1], unit_axis[0], 0.0]])
    return (np.eye(3) + np.sin(angle) * skew
            + (1.0 - np.cos(angle)) * (skew @ skew))


def order_strokes(strokes: Sequence[Stroke], start_near, *,
                  max_passes: int = 2,
                  axis_weight: float = 0.0) -> list[tuple[int, bool]]:
    """Order the strokes and choose an end for each: `(index, reversed)` in path order.

    Greedy nearest neighbour, then single-stroke relocation. The relocation pass is
    what a 2-opt cannot be here: reversing a run of strokes also reverses each of them,
    which changes every connector inside the run, so the usual "only two edges change"
    argument does not hold. Moving one stroke keeps the rest of the sequence intact and
    the improvement can be priced exactly.

    `axis_weight` (mm per degree) prices a pose change into the same cost as travel.
    Distance alone rates a 60 degree re-orientation at whatever the two endpoints happen
    to be apart, which is nothing - so the ordering happily pairs strokes whose poses have
    no relation and pays for it in wrist turns.
    """
    count = len(strokes)
    if count == 0:
        return []
    if count == 1:
        return [(0, False)]
    starts = np.array([np.asarray(stroke.start, dtype=float) for stroke in strokes])
    ends = np.array([np.asarray(stroke.end, dtype=float) for stroke in strokes])
    anchor = np.asarray(start_near, dtype=float).reshape(3)
    gaps = _pose_gaps(strokes) if axis_weight > 0.0 else None

    remaining = list(range(count))
    order: list[tuple[int, bool]] = []
    cursor = anchor
    previous = -1
    while remaining:
        best = None
        for index in remaining:
            for flip in (False, True):
                entry = ends[index] if flip else starts[index]
                cost = _edge(entry, cursor, gaps, previous, index, axis_weight)
                if best is None or cost < best[0]:
                    best = (cost, index, flip)
        _, index, flip = best
        order.append((index, flip))
        cursor = starts[index] if flip else ends[index]
        previous = index
        remaining.remove(index)

    order = _relocate(order, starts, ends, anchor, max_passes, gaps, axis_weight)
    return order


def _pose_gaps(strokes: Sequence[Stroke]) -> np.ndarray:
    """The pose change between every ordered pair of items, in degrees."""
    count = len(strokes)
    gaps = np.zeros((count, count), dtype=float)
    for row in range(count):
        for column in range(count):
            gaps[row, column] = pose_gap_deg(strokes[row].axis, strokes[row].roll_deg,
                                             strokes[column].axis, strokes[column].roll_deg)
    return gaps


def _edge(entry, cursor, gaps, previous: int, index: int, axis_weight: float) -> float:
    """Travel to reach `entry` plus what the pose change costs, in mm."""
    cost = float(np.linalg.norm(np.asarray(entry, dtype=float) - cursor))
    if gaps is None or previous < 0:
        return cost
    return cost + axis_weight * float(gaps[previous, index])


def _path_endpoints(order, starts, ends, anchor):
    """The entry and exit position of every stroke in the given order."""
    entry = np.array([ends[i] if flip else starts[i] for i, flip in order])
    exit_ = np.array([starts[i] if flip else ends[i] for i, flip in order])
    return entry, exit_


def _connector_length(order, starts, ends, anchor, gaps=None, axis_weight: float = 0.0
                      ) -> float:
    """Travel plus pose change along a trial order, in mm - the ordering's objective."""
    entry, exit_ = _path_endpoints(order, starts, ends, anchor)
    if len(order) == 0:
        return 0.0
    total = float(np.linalg.norm(entry[0] - anchor))
    if len(order) > 1:
        total += float(np.linalg.norm(entry[1:] - exit_[:-1], axis=1).sum())
        if gaps is not None and axis_weight > 0.0:
            for left, right in zip(order, order[1:], strict=False):
                total += axis_weight * float(gaps[left[0], right[0]])
    return total


def _relocate(order, starts, ends, anchor, max_passes: int, gaps=None,
              axis_weight: float = 0.0) -> list[tuple[int, bool]]:
    """Move one stroke at a time to a better slot, either way round, while it helps."""
    if len(order) < 3 or max_passes <= 0:
        return order
    best = list(order)
    best_cost = _connector_length(best, starts, ends, anchor, gaps, axis_weight)
    for _ in range(max_passes):
        improved = False
        for source in range(len(best)):
            without = best[:source] + best[source + 1:]
            item = best[source]
            for target in range(len(without) + 1):
                for flip in (item[1], not item[1]):
                    candidate = without[:target] + [(item[0], flip)] + without[target:]
                    cost = _connector_length(candidate, starts, ends, anchor, gaps,
                                             axis_weight)
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

    `coverage_index(i)` gives the coverage row that item `i` is described by, so each
    delivered point can point at the samples its sweep covers. `sources(i)` says whether
    the item came from the greedy pass or from repair.

    The items are not all sweeps: a selected pose arrives here as a zero-length stroke, and
    it is delivered as ONE corner because that is what it is. Leaving it out - or delivering
    both of its identical ends - is how the reported coverage and the program came apart in
    the first place.
    """
    say = log or (lambda _message: None)
    if not strokes:
        return PathPlan(points=[], ordering="empty")
    if start_near is None:
        start_near = np.asarray(strokes[0].start, dtype=float)
    order = order_strokes(strokes, start_near,
                          axis_weight=float(cfg.stroke.order_axis_weight_mm_per_deg))
    points: list[PathPoint] = []
    plan = PathPlan(points=points, ordering=(
        "nearest neighbour + single-stroke relocation over stroke endpoints, priced by "
        "travel plus pose change"))
    plan.coverage_strokes = sum(1 for stroke in strokes if stroke.length_mm > _ZERO_MM)

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
        # A zero-length stroke IS a pose: appending its "end" as well would deliver two
        # identical corners, which the point count and the .LS program would both carry.
        if float(np.linalg.norm(end - start)) > _ZERO_MM:
            points.append(PathPoint(position=end, axis=axis,
                                    roll_deg=float(stroke.roll_deg),
                                    role=ROLE_COVERAGE, source=source,
                                    stroke_id=int(index), coverage_index=row))
        previous = points[-1]

    plan.path_length_mm = _travel(points)
    say(f"  path: {len(strokes)} items ({plan.coverage_strokes} strokes) -> "
        f"{plan.turns} corners ({plan.transition_points} transitions), "
        f"{plan.path_length_mm:.0f} mm, {plan.interrupts_a} translation interrupts, "
        f"{plan.interrupts_b} re-orientations")
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

    Three routes, cheapest first, and `stroke.connector_level` says how far down the
    planner may go:

    * **L0** - straight there, no corner at all. Only for two identical poses, and only
      when that line keeps its clearance. Off by default: the "body" in this model is a
      sphere, which does not rotate with the head, so the model cannot see the part of a
      real handset that swings round behind the reference point.
    * **L1** - retract along -axis, then fly to the next entry point, spreading the
      re-orientation over the flight instead of over a short dive. This is safer than the
      two-corner route, not more dangerous: L2 puts the whole turn on the segment closest
      to the part.
    * **L2** - retract, traverse to a point behind the next entry, dive in. Two corners.

    Both of the retracting routes have to agree with the clearance check, because a fixed
    lift assumes the space behind the head is empty and on this cell it is not: behind the
    upper face is the base, behind the lower face is the floor side of the fixture. So the
    lift is raised until the moves are clear AND the rotation is spread over enough travel
    to stay inside the rate limit.
    """
    same = _same_pose(previous, entry)
    rotation = _rotation_deg(previous, entry)
    limit = int(cfg.stroke.connector_level)

    if limit <= 0 and same:
        margin, seen = _segment_margin(previous, entry, cfg, station, model,
                                       visible=visible)
        if margin > 0.0 and seen:
            _record(plan, same, margin, rotation, 0.0, LEVEL_STRAIGHT, entry, cfg)
            return []

    if limit <= 1:
        found = _one_corner(previous, entry, cfg, station, model, rotation, visible=visible)
        if found is not None:
            points, margin = found
            _record(plan, same, margin, rotation, 0.0, LEVEL_ONE_CORNER, entry, cfg)
            return points

    points, margin, rate = _two_corners(previous, entry, cfg, station, model, rotation,
                                        same, visible=visible)
    _record(plan, same, margin, rotation, rate, LEVEL_TWO_CORNERS, entry, cfg)
    return points


def _count_level(plan: PathPlan, level: str) -> None:
    plan.connector_levels[level] = plan.connector_levels.get(level, 0) + 1


def _record(plan: PathPlan, same: bool, margin: float, rotation: float, rate: float,
            level: str, entry: PathPoint, cfg: PlannerConfig) -> None:
    """Book-keep one transition: its level, the clearance it kept, and its interrupts."""
    _count_level(plan, level)
    plan.min_clearance_mm = (margin if plan.min_clearance_mm is None
                             else min(plan.min_clearance_mm, margin))
    if same:
        plan.interrupts_a += 1
    else:
        plan.interrupts_b += 1
        if rate:
            plan.violations.append(
                f"re-orientation of {rotation:.0f} deg cannot be spread under "
                f"{cfg.stroke.max_rot_deg_per_mm:g} deg/mm on the {level} route, within "
                f"{_LIFT_FACTORS[-1] * cfg.stroke.retract_clearance_mm:g} mm of lift")
    if margin <= 0.0:
        plan.violations.append(
            f"the connector into stroke {entry.stroke_id} keeps {margin:.1f} mm of body "
            f"clearance on every route tried, up to "
            f"{_LIFT_FACTORS[-1] * cfg.stroke.retract_clearance_mm:g} mm of lift")


def _retract_point(previous: PathPoint, lift: float) -> PathPoint:
    """A point `lift` millimetres behind the head, keeping its pose."""
    return PathPoint(
        position=np.asarray(previous.position, dtype=float)
        - np.asarray(previous.axis, dtype=float) * lift,
        axis=np.asarray(previous.axis, dtype=float), roll_deg=float(previous.roll_deg),
        role=ROLE_RETRACT, source=str(previous.source), stroke_id=previous.stroke_id)


def _one_corner(previous: PathPoint, entry: PathPoint, cfg: PlannerConfig,
                station: Station, model: ScannerModel, rotation: float, *,
                visible=None) -> tuple[list[PathPoint], float] | None:
    """L1: retract, then fly straight to the next entry. None when no lift works."""
    for factor in _LIFT_FACTORS:
        retract = _retract_point(previous, float(cfg.stroke.retract_clearance_mm) * factor)
        reach = float(np.linalg.norm(np.asarray(entry.position, dtype=float)
                                     - np.asarray(retract.position, dtype=float)))
        if reach <= 0.0 or rotation / reach > cfg.stroke.max_rot_deg_per_mm:
            continue
        margin = np.inf
        seen = True
        for first, second in ((previous, retract), (retract, entry)):
            found, tracker_ok = _segment_margin(first, second, cfg, station, model,
                                                visible=visible)
            margin = min(margin, found)
            seen = seen and tracker_ok
        if margin > 0.0 and seen:
            return [retract], float(margin)
    return None


def _two_corners(previous: PathPoint, entry: PathPoint, cfg: PlannerConfig,
                 station: Station, model: ScannerModel, rotation: float, same: bool, *,
                 visible=None) -> tuple[list[PathPoint], float, float]:
    """L2, the original route: retract, traverse behind the next entry, dive in."""
    best: tuple[float, list[PathPoint], float] | None = None
    for factor in _LIFT_FACTORS:
        lift = float(cfg.stroke.retract_clearance_mm) * factor
        retract = _retract_point(previous, lift)
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
        points = [retract, through]
        if margin > 0.0 and rate_ok and seen:
            return points, float(margin), 0.0
        # keep the roomiest attempt seen so far: a bigger lift is not monotonically safer,
        # because retracting away from the part can walk towards the base behind it
        if best is None or margin > best[0]:
            best = (float(margin), points, float(rate if not rate_ok else 0.0))
    margin, points, rate = best
    return points, margin, rate


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
