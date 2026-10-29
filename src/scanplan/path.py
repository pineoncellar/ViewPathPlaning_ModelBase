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
from dataclasses import dataclass, field, replace

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
# An axis whose usable range is at least this wide can reach the same pose with more than
# one number, which is what makes the written joint values ambiguous.
_FULL_TURN_DEG = 360.0
# Multiples of `retract_clearance_mm` the connector tries before giving up. A fixed lift
# assumes the space behind the head is empty, and on a real cell it is not: the base sits
# behind the upper face and the fixture behind the lower one.
_LIFT_FACTORS = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
# What a change of the CONFIG letters costs the branch search, in degrees of joint travel.
# Sized above the swings it is meant to avoid (158-168 deg on the measured program) but not
# so high that it forces a family which then needs a bigger detour elsewhere.
_CONFIG_CHANGE_COST_DEG = 240.0

# J6 gets a second charge on top of the all-axis one. J6 is the axis that turns the head about
# the optical axis; while the head scans it is the axis that must stay still, and every degree
# it turns is a different pixel-to-part mapping on the fly. The DP already prices the worst
# axis of each step; this makes it prefer the branch that spends the turn on the arm instead
# when both options cost the same all-axis step.
_J6_COST_WEIGHT = 0.5

ROLE_COVERAGE = "coverage"
ROLE_RETRACT = "retract"
ROLE_TRAVERSE = "traverse"
ROLE_REORIENT = "reorient"

# The connector routes, cheapest first. Reported per transition so "the connector got
# cheaper" is a measurement rather than a claim.
LEVEL_STRAIGHT = "L0-straight"
LEVEL_ONE_CORNER = "L1-retract"
LEVEL_TWO_CORNERS = "L2-retract-through"
LEVEL_SAFE_PLANE = "L3-safe-plane"

# Two successive sweeps whose optical axes differ by more than this belong to different
# scan directions, and the transfer between them has to leave through the common plane
# rather than slide from one to the other. Sixty degrees is chosen so that a merely tilted
# neighbour (the usual case on a curved panel) still uses the cheap local route; what this
# is meant to catch is a genuinely different side of the part.
_GROUP_DEG = 60.0
# Sweeps whose outward directions agree to within this belong to one direction group.
_GROUP_MERGE_DEG = 30.0


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
    # Items the hard gate refused to deliver. They are removed from the path, and the
    # caller re-runs the selection without them; keeping the list here is what lets the
    # report say "these sweeps were dropped and why" instead of quietly shortening the plan.
    dropped: list[int] = field(default_factory=list)
    # The largest joint change between two consecutive delivered points, when a kinematic
    # gate supplied the joint values. Reported, and used by the acceptance criteria.
    max_step_deg: float | None = None
    # The branch assignment (`path.assign_branches`) that produced the per-point CONFIG
    # strings, kept here so the viewpoint records and the report read one source.
    branches: object | None = None
    # (lambda, delivered points, coverage) for every budget trial, so "four times the
    # points bought this much" is a number in the report rather than an impression.
    budget_history: list[tuple[float, int, float]] = field(default_factory=list)
    verdict: str = ""

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
                  axis_weight: float = 0.0,
                  two_opt_limit: int = 40) -> list[tuple[int, bool]]:
    """Order the strokes and choose an end for each: `(index, reversed)` in path order.

    Greedy nearest neighbour, then single-stroke relocation, then 2-opt.

    The relocation pass is what a 2-opt cannot be *alone*: reversing a run of strokes also
    reverses each of them, which changes every connector inside the run, so the textbook
    "only two edges change" argument needs the reversal to be reflected in each item's
    direction flag - which is exactly what `_two_opt` does. Moving one stroke keeps the
    rest of the sequence intact and its improvement can be priced exactly; reversing a run
    undoes a crossing that relocation cannot.

    Both passes are capped: the number of candidate items grows with the pool, and an
    ordering pass that takes minutes would be paid again on every budget trial. Past the
    cap the result is the relocation answer, which is the design's stated fallback.

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
    return _two_opt(order, starts, ends, anchor, gaps, axis_weight,
                    limit=int(two_opt_limit))


def _two_opt(order, starts, ends, anchor, gaps, axis_weight: float, *, limit: int = 40,
             max_passes: int = 8) -> list[tuple[int, bool]]:
    """Undo crossings by reversing a run of strokes, flipping each one as it turns round.

    A reversed run is still the same physical plan: every stroke inside it is traversed the
    other way (free - a stroke is undirected), so the connectors inside the run are the same
    two endpoints in the other order and only the two edges at the cut change. That is what
    makes the usual 2-opt delta argument valid here, provided the flip flags are toggled with
    the order rather than left alone.
    """
    n = len(order)
    if n < 4 or n > limit:
        return order
    plan = list(order)
    for _ in range(max_passes):
        entry = np.array([ends[i] if flip else starts[i] for i, flip in plan])
        leave = np.array([starts[i] if flip else ends[i] for i, flip in plan])
        index = [i for i, _ in plan]

        def edge(from_point, to_point, from_index: int, to_index: int) -> float:
            cost = float(np.linalg.norm(np.asarray(to_point, dtype=float)
                                        - np.asarray(from_point, dtype=float)))
            if gaps is not None and axis_weight > 0.0:
                cost += axis_weight * float(gaps[from_index, to_index])
            return cost

        improved = False
        for left in range(n - 1):
            for right in range(left + 1, n):
                before = 0.0
                after = 0.0
                if left == 0:
                    before += float(np.linalg.norm(np.asarray(entry[0], dtype=float) - anchor))
                    after += float(np.linalg.norm(np.asarray(leave[right], dtype=float)
                                                  - anchor))
                else:
                    before += edge(leave[left - 1], entry[left], index[left - 1], index[left])
                    after += edge(leave[left - 1], leave[right], index[left - 1],
                                  index[right])
                if right + 1 < n:
                    before += edge(leave[right], entry[right + 1], index[right],
                                   index[right + 1])
                    after += edge(entry[left], entry[right + 1], index[left],
                                  index[right + 1])
                if after < before - 1e-9:
                    plan[left:right + 1] = [(item, not flip)
                                            for item, flip in reversed(plan[left:right + 1])]
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return plan


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
               strict: bool | None = None,
               kin=None,
               log: Callable[[str], None] | None = None) -> PathPlan:
    """Order `strokes`, insert the transitions, and account for the result.

    `coverage_index(i)` gives the coverage row that item `i` is described by, so each
    delivered point can point at the samples its sweep covers. `sources(i)` says whether
    the item came from the greedy pass or from repair.

    The items are not all sweeps: a selected pose arrives here as a zero-length stroke, and
    it is delivered as ONE corner because that is what it is. Leaving it out - or delivering
    both of its identical ends - is how the reported coverage and the program came apart in
    the first place.

    `strict` decides what an unconnectable item means. With it on (the configured
    `stroke.hard_gate`), no legal route means the item is removed and listed in
    `plan.dropped`, so the caller can re-run the selection without it; the alternative is
    recording a violation and delivering it anyway, which makes "legality first" a slogan.
    """
    say = log or (lambda _message: None)
    if not strokes:
        return PathPlan(points=[], ordering="empty")
    hard = bool(cfg.stroke.hard_gate) if strict is None else bool(strict)
    if start_near is None:
        start_near = np.asarray(strokes[0].start, dtype=float)
    order = order_strokes(strokes, start_near,
                          axis_weight=float(cfg.stroke.order_axis_weight_mm_per_deg))
    normal, plane = _safe_plane(strokes, cfg)
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
            routed = _connector(previous, entry, cfg, plan, station, model,
                                visible=visible, strict=hard, normal=normal, plane=plane,
                                kin=kin)
            if routed is None:
                plan.dropped.append(int(index))
                plan.violations.append(
                    f"stroke {index} dropped: no route into it keeps clearance and rate "
                    f"({rotation_text(previous, entry)})")
                continue
            points.extend(routed)
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
        f"{plan.interrupts_b} re-orientations"
        + (f", {len(plan.dropped)} dropped by the hard gate" if plan.dropped else ""))
    return plan


def rotation_text(previous: PathPoint, entry: PathPoint) -> str:
    """A short phrase for a violation line: how far the head has to turn."""
    return f"{_rotation_deg(previous, entry):.0f} deg"


def _safe_plane(strokes: Sequence[Stroke], cfg: PlannerConfig):
    """One plane every waiting point can sit on, and its outward normal.

    A retract that only knows the head's own axis puts the waiting point 50 mm behind that
    axis, which is the right place for one direction and the wrong place for the next: on a
    cell scanned from two sides, the two planes face opposite ways and the straight line
    between two waiting points cuts through the part. A single plane with one normal fixes
    that by construction, because both ends of the traverse satisfy `n . p >= plane` and the
    whole chord between them does too (the dot product is linear).

    The normal is the outward direction of the DOMINANT group, not the mean of all of them.
    A mean would cancel to zero on a two-sided part - which is exactly the case the plane
    exists for - and then every transfer would quietly fall back to the local route. Groups
    whose own outward is nearly perpendicular to the dominant one (cosine below 0.35) keep
    the local route by design; a plane they do not face is not a place to wait.
    """
    if not strokes:
        return None, None
    limit = float(np.cos(np.deg2rad(_GROUP_MERGE_DEG)))
    groups: list[list] = []
    for stroke in strokes:
        outward = -unit(stroke.axis)
        weight = max(float(stroke.length_mm), 1.0)
        for group in groups:
            if float(np.dot(unit(group[0]), outward)) >= limit:
                group[0] = np.asarray(group[0], dtype=float) + outward * weight
                group[1] = float(group[1]) + weight
                break
        else:
            groups.append([outward * weight, weight])
    normal = unit(max(groups, key=lambda group: float(group[1]))[0])
    lifts = max(float(cfg.stroke.retract_clearance_mm), float(cfg.scanner.standoff_mm))
    reach = max(max(float(np.dot(normal, np.asarray(stroke.start, dtype=float))),
                    float(np.dot(normal, np.asarray(stroke.end, dtype=float))))
                for stroke in strokes)
    return normal, reach + lifts


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
               visible=None, strict: bool = False, normal=None, plane=None, kin=None
               ) -> list[PathPoint] | None:
    """The points that take the head from `previous` to `entry` without cutting a corner.

    Routes, cheapest first, and `stroke.connector_level` says how far down the planner may
    go:

    * **L3** - leave through the common safe plane and come back in on the other side. Used
      only when the two sweeps look in different directions, because that is the case the
      per-axis retract gets wrong.
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

    Returns None when no route qualifies. Two reasons are judged differently: a route that
    fails CLEARANCE is refused only when `strict` is set - that is the hard gate, and its
    caller turns a refusal into "this item is not delivered" rather than a line in a report.
    A route whose transition corner the arm cannot hold falls back to a direct move between
    the two poses (recorded), because both ends are legal by definition and the alternative
    is losing the area they cover - unless the straight line itself is inside the geometry,
    which under `strict` drops the item like any other clearance refusal.
    """
    same = _same_pose(previous, entry)
    rotation = _rotation_deg(previous, entry)
    limit = int(cfg.stroke.connector_level)

    if normal is not None and plane is not None and _crosses_direction(previous, entry):
        found = _cross_group(previous, entry, cfg, station, model, rotation,
                             normal=normal, plane=plane, visible=visible)
        if found is not None and _route_allowed(found[0], kin):
            route, margin = found
            _record(plan, same, margin, rotation, 0.0, LEVEL_SAFE_PLANE, entry, cfg)
            return route

    if limit <= 0 and same:
        margin, seen = _segment_margin(previous, entry, cfg, station, model,
                                       visible=visible)
        if margin > 0.0 and seen:
            _record(plan, same, margin, rotation, 0.0, LEVEL_STRAIGHT, entry, cfg)
            return []

    if limit <= 1:
        found = _one_corner(previous, entry, cfg, station, model, rotation,
                            visible=visible, kin=kin, normal=normal)
        if found is not None and _route_allowed(found[0], kin):
            points, margin = found
            _record(plan, same, margin, rotation, 0.0, LEVEL_ONE_CORNER, entry, cfg)
            return points

    points, margin, rate = _two_corners(previous, entry, cfg, station, model, rotation,
                                        same, visible=visible, kin=kin, normal=normal)
    # Two different things can make a route unusable, and they are judged differently:
    #
    # * JOINT feasibility is never negotiable. A transition corner the arm cannot hold is
    #   not a judgement call - the controller will not execute it, and `assign_branches`
    #   could only report the point as unsolved and make the .LS export refuse. So an
    #   unreachable route is refused here whatever `stroke.hard_gate` says.
    # * CLEARANCE is the configurable one. `hard_gate` on means "drop the sweep rather than
    #   fly it close to the fixture"; off means "deliver it and list every route that failed
    #   clearance", because a transit the planner thinks is 15 mm short is something
    #   ROBOGUIDE can actually answer while a 27-point coverage hole cannot.
    #
    # `rate` is never a refusal either: how much rotation a millimetre of flight can absorb
    # rests on an uncalibrated constant, and dropping sweeps over it would make the plan look
    # legal by making it worse. Rate failures are recorded and counted instead.
    reachable = _route_allowed(points, kin)
    legal = margin > 0.0 and reachable
    if not legal and normal is not None and plane is not None:
        # Last resort: leave through the common plane. The plane route used to be reserved
        # for pairs that face opposite ways, but a pocket wall blocks the LOCAL routes
        # whatever the re-orientation is - "no route into it keeps clearance (1 deg)" is
        # exactly that case - and the plane is the one surface that gets both waiting points
        # out of the pocket. Only tried when everything cheaper has failed, so it cannot
        # cost a corner on a route that already worked.
        found = _cross_group(previous, entry, cfg, station, model, rotation,
                             normal=normal, plane=plane, visible=visible)
        if found is not None and _route_allowed(found[0], kin):
            route, plane_margin = found
            _record(plan, same, plane_margin, rotation, 0.0, LEVEL_SAFE_PLANE, entry, cfg)
            return route
    _record(plan, same, margin, rotation, rate, LEVEL_TWO_CORNERS, entry, cfg)
    if not reachable:
        # Every corner of every route is outside the arm's envelope, but BOTH ENDS are
        # points the gate already accepted - they are in the program either way. A direct
        # move between two holdable poses is what a programmer writes by hand, and the
        # controller interpolates it in joint space, so it is executable. Refusing it drops
        # the sweep or pose and loses the area it covered: on the 2026-10-28 station that
        # was 15 of 17 poses and 4 of 7 sweeps, i.e. 95.9% selected becoming 51.4%
        # delivered. It is recorded on its own line, never silent.
        direct, _seen = _segment_margin(previous, entry, cfg, station, model,
                                        visible=visible)
        plan.violations.append(
            f"the connector into stroke {entry.stroke_id} goes straight: no retract corner "
            f"the arm can hold was found on any route tried (direct clearance "
            f"{direct:.1f} mm)")
        _record(plan, same, direct, rotation, 0.0, LEVEL_STRAIGHT, entry, cfg)
        if strict and direct <= 0.0:
            # ...but "a move a programmer would write by hand" means a move that MISSES the
            # part. Two poses 800 mm apart on a curved surface are joined by a line that
            # dips through it, and the model says so here: on the 2026-10-29 station this
            # branch was the only thing between a 97.4% plan and a transit 28 mm INSIDE the
            # blank. Losing the sweep is a coverage hole the report can name; flying that
            # line is a crash, so under the hard gate the item goes instead.
            plan.violations.append(
                f"the straight line into stroke {entry.stroke_id} is refused: it passes "
                f"{abs(direct):.1f} mm inside the geometry, so the sweep is dropped rather "
                "than flown")
            return None
        return []
    if strict and margin <= 0.0:
        return None
    return points


def _route_allowed(points, kin) -> bool:
    """Whether the arm can hold every transition corner a route would insert.

    Only the corners are tested, not the segments between them: a `J` move between two
    reachable poses is not a pose problem, and pretending to check an interpolated joint
    path that this library does not model would be worse than saying so. What this catches
    is the real failure - a retract point behind the head that is inside the robot's own
    base, where no branch exists.
    """
    if kin is None or not kin.active:
        return True
    for point in points:
        if not kin.check(point.position, point.axis, point.roll_deg).ok:
            return False
    return True


def _crosses_direction(previous: PathPoint, entry: PathPoint) -> bool:
    """Whether two corners belong to different scan directions."""
    cosine = float(np.clip(np.dot(unit(previous.axis), unit(entry.axis)), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine))) > _GROUP_DEG


def _cross_group(previous: PathPoint, entry: PathPoint, cfg: PlannerConfig,
                 station: Station, model: ScannerModel, rotation: float, *,
                 normal, plane: float, visible=None
                 ) -> tuple[list[PathPoint], float] | None:
    """L3: retract to the common plane, traverse on it, dive back in.

    The report's rule, and the reason it is a separate route rather than a bigger lift: two
    waiting points that both satisfy `n . p >= plane` are joined by a segment that satisfies
    it everywhere, so a transfer between two scan directions stays outside whatever the
    plane was built to clear. Retracting along each axis separately has no such property -
    on a two-sided part the two retract points face opposite ways and the line between them
    runs through the part.

    The `n . outward < 0.35` case falls back to the local routes on purpose: when the sweep
    faces sideways relative to the plane, the plane is not a good place to wait.
    """
    cosine = min(float(np.dot(unit(-previous.axis), normal)),
                 float(np.dot(unit(-entry.axis), normal)))
    if cosine < 0.35:
        return None
    base = float(cfg.stroke.retract_clearance_mm)
    lift_out = max(base, plane - float(np.dot(normal, previous.position)))
    lift_in = max(base, plane - float(np.dot(normal, entry.position)))
    wait_out = PathPoint(
        position=np.asarray(previous.position, dtype=float) + normal * lift_out,
        axis=previous.axis, roll_deg=float(previous.roll_deg), role=ROLE_RETRACT,
        source=str(previous.source), stroke_id=previous.stroke_id)
    wait_in = PathPoint(
        position=np.asarray(entry.position, dtype=float) + normal * lift_in,
        axis=entry.axis, roll_deg=float(entry.roll_deg), role=ROLE_TRAVERSE,
        source=str(entry.source), stroke_id=entry.stroke_id)
    reach = float(np.linalg.norm(wait_in.position - wait_out.position))
    if reach <= 0.0 or rotation / reach > cfg.stroke.max_rot_deg_per_mm:
        return None
    margin = np.inf
    seen = True
    for first, second in ((previous, wait_out), (wait_out, wait_in), (wait_in, entry)):
        found, tracker_ok = _segment_margin(first, second, cfg, station, model,
                                            visible=visible)
        margin = min(margin, found)
        seen = seen and tracker_ok
    if margin <= 0.0 or not seen:
        return None
    return [wait_out, wait_in], float(margin)


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


def _retract_point(previous: PathPoint, lift: float, direction=None) -> PathPoint:
    """A waiting point `lift` millimetres away from the head, keeping its pose.

    `direction` defaults to -axis, i.e. straight back along the optical axis, which is the
    obvious guess and the one that keeps the retract and the dive collinear. It is only a
    guess: behind a horizontal-looking head is the part's own side wall, and a corner there
    can be blocked by the fixture or - more often on this cell - sit outside the arm's
    envelope while the pose itself is fine. The caller therefore also offers the outward
    direction of the safe plane, which is up and away from everything the plane was built
    to clear.
    """
    if direction is None:
        direction = -np.asarray(previous.axis, dtype=float)
    return PathPoint(
        position=np.asarray(previous.position, dtype=float)
        + np.asarray(direction, dtype=float) * lift,
        axis=np.asarray(previous.axis, dtype=float), roll_deg=float(previous.roll_deg),
        role=ROLE_RETRACT, source=str(previous.source), stroke_id=previous.stroke_id)


def _retract_directions(previous: PathPoint, normal) -> list[np.ndarray]:
    """The directions to try for a waiting point, best guess first, without duplicates."""
    back = -unit(np.asarray(previous.axis, dtype=float))
    out = [back]
    if normal is not None:
        plane_out = unit(np.asarray(normal, dtype=float))
        # within about 25 degrees of "behind the head" it is the same move, so trying it
        # again would only double the ray casts
        if float(np.dot(back, plane_out)) < float(np.cos(np.deg2rad(25.0))):
            out.append(plane_out)
    return out


def _one_corner(previous: PathPoint, entry: PathPoint, cfg: PlannerConfig,
                station: Station, model: ScannerModel, rotation: float, *,
                visible=None, kin=None, normal=None) -> tuple[list[PathPoint], float] | None:
    """L1: retract, then fly straight to the next entry. None when no lift works."""
    for direction in _retract_directions(previous, normal):
        found = _one_corner_along(previous, entry, cfg, station, model, rotation, direction,
                                  visible=visible, kin=kin)
        if found is not None:
            return found
    return None


def _one_corner_along(previous: PathPoint, entry: PathPoint, cfg: PlannerConfig,
                      station: Station, model: ScannerModel, rotation: float, direction,
                      *, visible=None, kin=None
                      ) -> tuple[list[PathPoint], float] | None:
    """L1 along one particular retract direction, over the ladder of lifts."""
    for factor in _LIFT_FACTORS:
        retract = _retract_point(previous, float(cfg.stroke.retract_clearance_mm) * factor,
                                 direction)
        reach = float(np.linalg.norm(np.asarray(entry.position, dtype=float)
                                     - np.asarray(retract.position, dtype=float)))
        if reach <= 0.0 or rotation / reach > cfg.stroke.max_rot_deg_per_mm:
            continue
        if not _route_allowed([retract], kin):
            # A smaller lift is worth trying: the retract point only has to be reachable,
            # and 50 mm is far more likely to be than 400 mm. Testing the gate INSIDE the
            # lift search is what keeps the hard gate from rejecting a sweep whose only
            # problem was that the first lift it tried was too greedy.
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
                 visible=None, kin=None, normal=None
                 ) -> tuple[list[PathPoint], float, float]:
    """L2, the original route: retract, traverse behind the next entry, dive in."""
    best: tuple[list[PathPoint], float, float] | None = None
    for direction in _retract_directions(previous, normal):
        points, margin, rate = _two_corners_along(previous, entry, cfg, station, model,
                                                 rotation, same, direction,
                                                 visible=visible, kin=kin)
        if margin > 0.0 and rate == 0.0:
            return points, margin, rate
        if best is None or margin > best[1]:
            best = (points, margin, rate)
    assert best is not None
    return best


def _two_corners_along(previous: PathPoint, entry: PathPoint, cfg: PlannerConfig,
                       station: Station, model: ScannerModel, rotation: float, same: bool,
                       direction, *, visible=None, kin=None
                       ) -> tuple[list[PathPoint], float, float]:
    """L2 along one particular retract direction, over the ladder of lifts."""
    best: tuple[float, list[PathPoint], float] | None = None
    for factor in _LIFT_FACTORS:
        lift = float(cfg.stroke.retract_clearance_mm) * factor
        retract = _retract_point(previous, lift, direction)
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
        reachable = _route_allowed([retract, through], kin)
        points = [retract, through]
        if margin > 0.0 and rate_ok and seen and reachable:
            return points, float(margin), 0.0
        # keep the roomiest attempt seen so far: a bigger lift is not monotonically safer,
        # because retracting away from the part can walk towards the base behind it
        if best is None or margin > best[0]:
            best = (float(margin), points, float(rate if not rate_ok else 0.0))
    margin, points, rate = best
    return points, margin, rate


@dataclass
class BranchAssignment:
    """One joint solution per delivered point, chosen as a SEQUENCE rather than one by one.

    `failed` lists the point indices the assignment could not place at all (no feasible
    solution), and `broken` the indices where the best available sequence still exceeds
    `max_step_deg`. They are separate because they mean different things: "the arm cannot be
    there" is a reachability problem, "the arm cannot get there from the previous point in
    one move" is an ordering problem.
    """

    solutions: list = field(default_factory=list)
    failed: list[int] = field(default_factory=list)
    broken: list[int] = field(default_factory=list)
    max_step_deg: float = 0.0
    max_step_at: int | None = None
    max_j6_step_deg: float = 0.0
    min_margin_deg: float | None = None
    histogram: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failed and not self.broken


def assign_branches(points: Sequence[PathPoint], gate, max_step_deg: float
                    ) -> BranchAssignment:
    """Pick a joint branch for every delivered point so the sequence jumps as little as it can.

    Picking the locally best branch at every point is the mistake this exists to prevent:
    each point may be fine on its own while the pair is not, because a `J` move interpolates
    in joint space. The search is a small dynamic programme - at most eight branches per
    point, a few hundred points - minimising the total joint travel subject to a hard limit
    on each step.

    Points without any feasible solution are reported, never silently filled in with
    something plausible.

    Changing the CONFIG letters mid-program costs extra: the letters are the arm's posture,
    not the tool's pose, so the controller gets to the next one by swinging the whole arm
    round (J1 167 deg on the 2026-10-28 program, J3 159 deg on the way back in). Nothing in
    the delivery geometry models that swing, so it is paid for here - as a preference, not a
    ban: when the only solution for a point sits in another family, the path still takes it.
    """
    if gate is None or not gate.active:
        return BranchAssignment()
    candidates = [gate.solutions(point.position, point.axis, point.roll_deg)
                  for point in points]
    if not points:
        return BranchAssignment()
    failed = [index for index, options in enumerate(candidates) if not options]
    # `total[i][j]` is the cheapest joint travel that ends on option j of point i;
    # `parent[i][j]` remembers which option of point i-1 got there.
    total = [[0.0] * len(options) for options in candidates]
    parent: list[list[int]] = [[0] * len(options) for options in candidates]
    for index in range(1, len(candidates)):
        for j, option in enumerate(candidates[index]):
            best = (float("inf"), 0)
            for k, previous in enumerate(candidates[index - 1]):
                cost = (total[index - 1][k] + _step_between(previous, option)
                        + _J6_COST_WEIGHT * _axis_step_between(previous, option, 5)
                        + _config_penalty(previous, option))
                if cost < best[0]:
                    best = (cost, k)
            total[index][j] = best[0]
            parent[index][j] = best[1]
    choice = int(np.argmin(total[-1])) if candidates[-1] else 0
    picked: list = [None] * len(points)
    for index in range(len(candidates) - 1, -1, -1):
        if not candidates[index]:
            continue
        picked[index] = candidates[index][choice]
        if index > 0:
            choice = parent[index][choice]

    picked = _unwrap_along(picked, getattr(getattr(gate, "robot", None), "limits", None))
    steps = [_step_between(picked[index - 1], picked[index])
             for index in range(1, len(picked))]
    # J6 on its own: the head turns about the optical axis with J6 alone, so a large J6 step
    # is the wrist spinning in place. That is harmless between sweeps (the move scans
    # nothing) and impossible during one (a sweep is a constant pose), so it is reported
    # separately from the all-axis worst step rather than folded into it.
    j6_steps = [_axis_step_between(picked[index - 1], picked[index], 5)
                for index in range(1, len(picked))]
    broken = [index + 1 for index, step in enumerate(steps) if step > max_step_deg]
    worst = max(steps) if steps else 0.0
    margins = [item.min_margin_deg for item in picked if item is not None]
    histogram: dict[str, int] = {}
    for item in picked:
        if item is not None:
            key = item.config.split(",")[0].strip()
            histogram[key] = histogram.get(key, 0) + 1
    return BranchAssignment(
        solutions=picked, failed=failed, broken=broken, max_step_deg=float(worst),
        max_step_at=(int(np.argmax(steps)) + 1 if steps else None),
        max_j6_step_deg=(float(max(j6_steps)) if j6_steps else 0.0),
        min_margin_deg=(float(min(margins)) if margins else None),
        histogram=histogram)


def _unwrap_along(solutions: Sequence, limits) -> list:
    """Rewrite the turn-free axes so the program turns the short way, not the long one.

    An axis with a full 720 degree range has more than one number for the same pose, and the
    closed form is free to hand back whichever one it likes: two consecutive points can read
    -179.8 and +177.5, which are 2.6 degrees apart as a pose and 357 degrees apart as a J
    move. The controller interpolates the numbers it is given, so a turn like that is a real
    full revolution of the wrist in the middle of the path, not a display quirk.

    Each point is therefore shifted by whole turns to the representative nearest the point
    before it. Two conditions keep the written program equivalent: the shifted value must
    stay inside the usable envelope, and it must stay on the same side of one full turn, so
    the CONFIG turn digits - which this module does not recompute - stay correct.

    Only an axis with at least two full turns of travel is treated as one that can wrap.
    A 370 degree range is a wide joint, not a turn-enabled one: J1 on this machine stops at
    plus or minus 185 and cannot be told to come the other way round.
    """
    if limits is None or not solutions:
        return list(solutions)
    spans = [(float(upper) - float(lower)) for lower, upper in limits]
    out: list = [solutions[0]]
    for index in range(1, len(solutions)):
        first, second = out[index - 1], solutions[index]
        if first is None or second is None:
            out.append(second)
            continue
        joints = [float(value) for value in second.joints_deg]
        moved = False
        for axis, span in enumerate(spans):
            if span < 2.0 * _FULL_TURN_DEG - 1e-6:
                continue
            lower, upper = (float(limits[axis][0]), float(limits[axis][1]))
            shift = round((float(first.joints_deg[axis]) - joints[axis]) / _FULL_TURN_DEG)
            if not shift:
                continue
            shifted = joints[axis] + shift * _FULL_TURN_DEG
            if not lower <= shifted <= upper:
                continue
            if abs(shifted) > _FULL_TURN_DEG or abs(joints[axis]) > _FULL_TURN_DEG:
                continue
            joints[axis] = shifted
            moved = True
        out.append(replace(second, joints_deg=tuple(joints)) if moved else second)
    return out


def _step_between(first, second) -> float:
    """Largest single-axis joint change between two solutions, in degrees.

    Circular: a reading of +179 and one of -181 are two degrees apart, not 360. The
    controller takes the shortest path per axis, so the sequence has to be judged the same
    way - otherwise a wrist sitting near the wrap is reported as a 358 degree jump and the
    continuity constraint looks violated when it is not.
    """
    if first is None or second is None:
        return 0.0
    delta = (np.asarray(second.joints_deg, dtype=float)
             - np.asarray(first.joints_deg, dtype=float))
    return float(np.max(np.abs((delta + 180.0) % 360.0 - 180.0)))


def _axis_step_between(first, second, axis: int) -> float:
    """One axis' joint change between two solutions, in degrees, shortest way round."""
    if first is None or second is None:
        return 0.0
    delta = float(second.joints_deg[axis]) - float(first.joints_deg[axis])
    return abs((delta + 180.0) % 360.0 - 180.0)


def _config_penalty(first, second) -> float:
    """The extra cost of changing the arm's CONFIG letters between two points.

    The letters say which posture the arm is in - front or back, elbow up or down, wrist
    flipped or not - and two postures for the same tool pose are not two nearby joint
    vectors: the controller swings the whole arm round to get from one to the other. On the
    2026-10-28 program that swing was J1 167.6 deg out and J3 158.8 deg in, in the middle of
    a scan. It is invisible to every clearance check in this module (they only see the tool
    path), so it is charged here instead, where the sequence is chosen.

    A charge, not a prohibition: a point whose only solution is in another family keeps it.
    """
    if first is None or second is None:
        return 0.0
    one = str(getattr(first, "config", "")).split(",")[0].strip()
    other = str(getattr(second, "config", "")).split(",")[0].strip()
    return 0.0 if one == other else _CONFIG_CHANGE_COST_DEG


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
