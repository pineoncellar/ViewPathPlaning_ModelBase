"""Turn a chosen set of strokes into one ordered, connected TCP path.

A stroke is an unordered, undirected segment; a program is a sequence. Closing that
gap is the whole job of this module:

* **order** - a greedy nearest-neighbour pass over the items, choosing which end to
  enter by, then single-stroke relocation passes. Not an optimum, and not claimed to
  be: what the workshop feels is the number of corners, not the last millimetre of
  travel. Distance alone is not the whole cost either - a short move between two
  unrelated poses costs a wrist turn, so the pose change is priced in as well.
* **connect** - between two items the head has to get from one to the other without
  dragging the scanner through the part. A transition is one of four shapes, and the
  shape decides which move in it is the long one:

  - **joint route** (the default once a move is long or the pose changes) - a short
    straight lift, a joint move to the next sweep's pre-point, then a short straight
    approach along the next optical axis. The rotation is carried by the joint move, so
    no straight segment ever has to turn the wrist quickly, which is what a long `L`
    across a big pose change does to a real machine.
  - **one corner** - retract, then fly straight in. Straight all the way, and only for a
    move short enough that the rotation spread over it stays inside the rate limit.
  - **two corners** - retract, traverse behind the next entry, dive in. Straight all the
    way; the fallback for a site that forbids joint travel moves.
  - **straight** - no corner at all, for a pose that does not change and a clear line.

  The choice is the geometry's, not a flag: a short move between identical poses stays
  straight (`joint_travel_min_mm` is the "short" threshold), everything else is a joint
  travel. Reachability aside, that is the rule the workshop stated: never cross a big
  pose change with one long straight move, and never insert a joint move into a move that
  did not need one.
* **account** - turn count, travel length, measurement interrupts, which route each
  transition took, the smallest clearance margin anywhere on the path, and - when an arm
  is configured - whether every delivered point and every STRAIGHT move between two of
  them has a configuration inside the arm's derated joint window. A joint move needs only
  its two ends checked, because a joint-space interpolation between two legal
  configurations stays legal; a straight move does not, so it is sampled.

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
from .geometry import scanner_frame, unit
from .robot import RobotArm
from .rolls import pose_turn_deg, smooth_rolls, twist_report
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
# The pre-point of a joint travel: the head is brought to it with a joint move, carrying
# the NEXT sweep's pose, so the rotation happens on that move instead of on the short
# straight approach that follows it. The export reads this role to decide which moves are
# joint moves, so the name is part of the contract between the two modules.
ROLE_JOINT = "joint"

# The connector routes, cheapest first. Reported per transition so "the connector got
# cheaper" is a measurement rather than a claim.
LEVEL_STRAIGHT = "L0-straight"
LEVEL_ONE_CORNER = "L1-retract"
LEVEL_TWO_CORNERS = "L2-retract-through"
# The joint route. Prefixed `J` because the travel move is a joint move, unlike the three
# above, which are straight all the way.
LEVEL_JOINT = "J2-retract-travel"


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
    # How far the wrist has to turn between consecutive delivered points, in three parts
    # (tilt, extra twist, whole turn) plus the moves over the configured limit. Filled by
    # `build_path`; empty for a path nobody built.
    wrist: dict = field(default_factory=dict)
    # What the joint check saw on the delivered path: how many moves were walked, how many
    # had no configuration inside the derated window anywhere along them, the worst
    # utilization of an allowed half-travel any accepted configuration reached, and the
    # items (`unreachable_items`, indices into the delivered strokes) whose own pose or
    # move could not be solved. The caller drops those and builds again - see
    # `unreachable_items` and `pipeline.delivery`. Empty when no arm was configured, which
    # is what "the joint window was not checked" looks like.
    joints: dict = field(default_factory=dict)

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

    Defined once, in :mod:`scanplan.rolls`, because the ordering prices the turn, the
    connector spreads it and the report counts it - three callers that must not be able
    to disagree about how far the head turned.
    """
    return pose_turn_deg(axis_a, roll_a, axis_b, roll_b)


def _frame(point: PathPoint) -> np.ndarray:
    """The TCP frame of a delivered point, from the one definition of that frame."""
    return scanner_frame(point.axis, float(point.roll_deg))


def _is_joint_move(point: PathPoint) -> bool:
    """Whether the move INTO this point is a joint move.

    The same rule the `.LS` export applies (`roboguide/ls.py: REPOSITION_ROLES`): the
    pre-point of a joint route is the point a joint move lands on, and every other move in
    the program is straight.
    """
    return str(point.role).strip().lower() == ROLE_JOINT


def _walk_joints(arm: RobotArm, cfg: PlannerConfig, joints, previous: PathPoint,
                 route: Sequence[PathPoint]):
    """Walk one candidate route through the arm's joint window.

    `joints` is the configuration at `previous` and `route` is what comes after it, last
    point included. Returns `(ok, joints_at_the_end, worst_utilization, worst_joint)`.

    Straight moves are sampled with `RobotArm.along`, because a straight line in Cartesian
    space can leave the legal joint box in between its ends. A joint move is checked at its
    end only: the window is a box, so a joint-space interpolation between two legal
    configurations stays legal on the way.
    """
    worst, worst_joint = 0.0, 0
    for point in route:
        if _is_joint_move(point):
            reach = arm.solve(_frame(point), point.position, joints)
            if reach is None:
                return False, None, worst, worst_joint
            joints = reach.joints_deg
            found, found_joint = reach.utilization, reach.worst_joint
        else:
            check = arm.along(_frame(previous), _frame(point), previous.position,
                              point.position, step_mm=cfg.stroke.check_step_mm,
                              reference=joints)
            if not check.ok:
                return False, None, worst, worst_joint
            joints = check.joints_deg
            found, found_joint = check.worst_utilization, check.worst_joint
        if found > worst:
            worst, worst_joint = float(found), int(found_joint)
        previous = point
    return True, joints, worst, worst_joint


def _settle(arm: RobotArm, plan: PathPlan, joints, point: PathPoint, what: str):
    """The configuration of a delivered point nobody moves into, or None when there is none.

    Only the first point of a path needs this: every other one is reached by a move, and
    that move is what carries the reference configuration.
    """
    reach = arm.solve(_frame(point), point.position, joints)
    _note_joints(plan, reach is not None, 0.0 if reach is None else reach.utilization,
                 0 if reach is None else reach.worst_joint, what, point.stroke_id)
    return None if reach is None else reach.joints_deg


def _note_joints(plan: PathPlan, ok: bool, worst: float, worst_joint: int,
                 what: str, stroke_id: int | None = None) -> None:
    """Book-keep one walked move, and name it when it left the window.

    A failed move also names the item it belongs to, so a caller can drop that item and
    build the path again instead of delivering a point no configuration reaches.
    """
    stats = plan.joints
    stats["checked"] = True
    stats["checked_moves"] = int(stats.get("checked_moves", 0)) + 1
    if not ok:
        stats["out_of_window_moves"] = int(stats.get("out_of_window_moves", 0)) + 1
        plan.violations.append(f"{what} has no configuration inside the arm's joint window")
        if stroke_id is not None:
            blocked = stats.setdefault("unreachable_items", [])
            if int(stroke_id) not in blocked:
                blocked.append(int(stroke_id))
        return
    if worst >= float(stats.get("worst_utilization", 0.0)):
        stats["worst_utilization"] = round(float(worst), 4)
        stats["worst_joint"] = f"J{int(worst_joint) + 1}"


def unreachable_items(plan: PathPlan) -> tuple[int, ...]:
    """Indices of the delivered items whose joint check failed, in plan order.

    The indices address the stroke sequence `build_path` was given, so a caller holding
    that sequence drops exactly these and rebuilds. An empty tuple means every delivered
    point and every straight move between them has a configuration inside the window.
    """
    return tuple(int(value) for value in plan.joints.get("unreachable_items", ()))


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
               robot: RobotArm | None = None,
               log: Callable[[str], None] | None = None) -> PathPlan:
    """Order `strokes`, insert the transitions, and account for the result.

    `coverage_index(i)` gives the coverage row that item `i` is described by, so each
    delivered point can point at the samples its sweep covers. `sources(i)` says whether
    the item came from the greedy pass or from repair.

    The items are not all sweeps: a selected pose arrives here as a zero-length stroke, and
    it is delivered as ONE corner because that is what it is. Leaving it out - or delivering
    both of its identical ends - is how the reported coverage and the program came apart in
    the first place.

    `robot` is the optional joint window. With one, every connector shape is tried against
    it as well as against clearance, so a lift that keeps the head clear but puts a joint
    past its derated stop is passed over for the next one - which is why the joint route
    matters: its middle move is a joint move, and those need only their two ends checked.
    Every accepted point and every straight move is then walked once more, and a move that
    still cannot be made is named in `plan.violations` instead of being delivered silently;
    the item it belongs to is listed in ``plan.joints["unreachable_items"]`` so a caller
    can drop it and build again (`unreachable_items` reads that list back).
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
    joints = None
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
            connector, joints = _connector(previous, entry, cfg, plan, station, model,
                                           visible=visible, arm=robot, joints=joints)
            points.extend(connector)
        points.append(entry)
        if robot is not None:
            joints = _settle(robot, plan, joints, entry, f"stroke {index}'s start")
        # A zero-length stroke IS a pose: appending its "end" as well would deliver two
        # identical corners, which the point count and the .LS program would both carry.
        if float(np.linalg.norm(end - start)) > _ZERO_MM:
            last = PathPoint(position=end, axis=axis,
                             roll_deg=float(stroke.roll_deg),
                             role=ROLE_COVERAGE, source=source,
                             stroke_id=int(index), coverage_index=row)
            points.append(last)
            if robot is not None:
                # The sweep itself is one straight move, so it is walked for the same
                # reason the connectors are: its ends being legal does not make its
                # middle legal.
                ok, joints, worst, worst_joint = _walk_joints(robot, cfg, joints, entry,
                                                              [last])
                _note_joints(plan, ok, worst, worst_joint, f"stroke {index}'s sweep",
                             index)
        previous = points[-1]

    plan.path_length_mm = _travel(points)
    plan.wrist = twist_report(points, float(cfg.stroke.max_wrist_step_deg))
    if plan.wrist["moves_over_limit"]:
        plan.violations.append(
            f"{plan.wrist['moves_over_limit']} move(s) ask the wrist for more than "
            f"{float(cfg.stroke.max_wrist_step_deg):g} deg at once (worst "
            f"{plan.wrist['max_rotation_deg']:g} deg); the checks that see this first are "
            "listed in `wrist`")
    say(f"  path: {len(strokes)} items ({plan.coverage_strokes} strokes) -> "
        f"{plan.turns} corners ({plan.transition_points} transitions), "
        f"{plan.path_length_mm:.0f} mm, {plan.interrupts_a} translation interrupts, "
        f"{plan.interrupts_b} re-orientations")
    if plan.joints.get("checked"):
        blocked = unreachable_items(plan)
        say(f"  joint window: {plan.joints['checked_moves']} move(s) walked, "
            f"{plan.joints.get('out_of_window_moves', 0)} without a legal configuration, "
            f"worst utilization {plan.joints.get('worst_utilization', 0.0):.3f}"
            + (f" on {plan.joints['worst_joint']}" if plan.joints.get("worst_joint") else "")
            + (f"; unreachable items {list(blocked)}" if blocked else ""))
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
               visible=None, arm: RobotArm | None = None,
               joints=None) -> tuple[list[PathPoint], np.ndarray | None]:
    """The points that take the head from `previous` to `entry` without cutting a corner.

    Four shapes, and which one is used is decided by the geometry, then capped by
    `stroke.connector_level`:

    * **joint** (`ROLE_JOINT`) - a short straight lift, a joint move to the pre-point that
      carries the next sweep's pose, then a short straight approach. The default: it is
      the only shape in which the pose change rides on a move that does not have to be
      straight, so a large pose change or a long travel never turns into a long straight
      move with a wrist turn inside it.
    * **L1** - retract along -axis, then fly straight in. Straight all the way, and only
      for a move short enough that the rotation spread over it stays inside the rate
      limit.
    * **L2** - retract, traverse behind the next entry, dive in. Straight all the way; the
      fallback for a site that forbids joint travel moves.
    * **L0** - straight there, no corner at all. Only for two identical poses, and only
      when that line keeps its clearance. Off by default: the "body" in this model is a
      sphere, which does not rotate with the head, so the model cannot see the part of a
      real handset that swings round behind the reference point.

    `connector_level` caps the most elaborate shape allowed: 2 allows the joint route,
    1 forbids it (straight-only site), 0 also allows the corner-less straight move. It
    does not force the joint route: a move between two identical poses shorter than
    `joint_travel_min_mm` stays straight at every level, because inserting a joint move
    into a move that did not need one costs a corner and buys nothing.

    Every retracting shape has to agree with the clearance check, because a fixed lift
    assumes the space behind the head is empty and on this cell it is not: behind the
    upper face is the base, behind the lower face is the floor side of the fixture. So the
    lift is raised until the moves are clear AND - for the straight shapes - the rotation
    is spread over enough travel to stay inside the rate limit.

    `arm`/`joints` add the fourth test: a shape whose straight moves leave the arm's
    derated joint window is passed over for the next lift or the next shape. The joint
    route wins ties here more often than it does on clearance alone, because its middle
    move is the one that is not sampled - a joint move only needs legal ends.

    Returns the points and the arm configuration at `entry`, which is what the next
    transition continues from.
    """
    same = _same_pose(previous, entry)
    rotation = _rotation_deg(previous, entry)
    limit = int(cfg.stroke.connector_level)
    reach = float(np.linalg.norm(np.asarray(entry.position, dtype=float)
                                 - np.asarray(previous.position, dtype=float)))
    # a short move between two identical poses is the one case that is flown straight at
    # every level: a joint move there would add a corner and change nothing else
    short_same_pose = same and reach < float(cfg.stroke.joint_travel_min_mm)

    def fits(points: Sequence[PathPoint]) -> bool:
        """Whether this candidate also has a configuration inside the joint window."""
        if arm is None:
            return True
        ok, _end, _worst, _worst_joint = _walk_joints(arm, cfg, joints, previous,
                                                      [*points, entry])
        return ok

    points: list[PathPoint] | None = None
    if limit <= 0 and same:
        margin, seen = _segment_margin(previous, entry, cfg, station, model,
                                       visible=visible)
        if margin > 0.0 and seen and fits([]):
            _record(plan, same, margin, rotation, 0.0, LEVEL_STRAIGHT, entry, cfg)
            points = []

    if points is None and (short_same_pose or limit <= 1):
        found = _one_corner(previous, entry, cfg, station, model, rotation, visible=visible,
                            fits=fits)
        if found is not None:
            points, margin = found
            _record(plan, same, margin, rotation, 0.0, LEVEL_ONE_CORNER, entry, cfg)
        # nothing to do here when the compressed straight shape does not fit anywhere
        # within the lift range: a corner more is better than a connector that cuts the
        # part, so the transition falls through to the shapes below

    if points is None and limit >= 2:
        points, margin = _joint_route(previous, entry, cfg, station, model,
                                      visible=visible, fits=fits)
        _record(plan, same, margin, rotation, 0.0, LEVEL_JOINT, entry, cfg)

    if points is None:
        # a straight-only site: the last straight shape, which never fails outright - it
        # returns its roomiest attempt and records a violation if that attempt is not clear
        points, margin, rate = _two_corners(previous, entry, cfg, station, model, rotation,
                                            same, visible=visible, fits=fits)
        _record(plan, same, margin, rotation, rate, LEVEL_TWO_CORNERS, entry, cfg)

    if arm is None:
        return points, None
    ok, end, worst, worst_joint = _walk_joints(arm, cfg, joints, previous, [*points, entry])
    _note_joints(plan, ok, worst, worst_joint, f"the connector into stroke {entry.stroke_id}",
                 entry.stroke_id)
    return points, end


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
                visible=None, fits=None) -> tuple[list[PathPoint], float] | None:
    """L1: retract, then fly straight to the next entry. None when no lift works.

    Straight all the way, so the pose change has to fit on the flight: the lift is raised
    until the rotation spread over that segment is inside `max_rot_deg_per_mm`. None means
    every lift in the range failed that check, the clearance one, or `fits` - the optional
    joint-window test, which is given the candidate's intermediate points.
    """
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
        if margin > 0.0 and seen and (fits is None or fits([retract])):
            return [retract], float(margin)
    return None


def _joint_route(previous: PathPoint, entry: PathPoint, cfg: PlannerConfig,
                 station: Station, model: ScannerModel, *,
                 visible=None, fits=None) -> tuple[list[PathPoint], float]:
    """The L/J/L route: lift straight, joint-move to the pre-point, approach straight.

    Three moves, and each one exists for a different reason:

    * `previous -> retract` - a short straight lift keeping the pose, so the head leaves
      the surface without a swing. A lift that changed the pose would sweep the scanner
      across the part it is standing on.
    * `retract -> pre-point` - the joint move. It covers the distance AND the pose change
      between the two sweeps, because a joint move interpolates in joint space: the wrist
      turns while the arm travels, and the rotation does not have to be spread over TCP
      travel the way a straight move needs.
    * `pre-point -> entry` - a short straight approach along the next optical axis. The
      pose is already the sweep's pose, so this segment only closes the standoff.

    The pre-point is the whole point of the route: it carries the next sweep's pose, so
    the pose is settled BEFORE the head is close to the part. The straight-with-reorient
    route does the opposite - it arrives at the next pose while diving - which is what
    turns a large pose change into a wrist alarm.

    Clearance is measured on the three straight segments as a proxy. The joint move's own
    TCP path is the controller's and is not modelled; the lift is raised until the proxy
    fits, and the export contract records that the joint move is unverified.
    """
    entry_axis = np.asarray(entry.axis, dtype=float)
    best: tuple[float, list[PathPoint]] | None = None
    for factor in _LIFT_FACTORS:
        lift = float(cfg.stroke.retract_clearance_mm) * factor
        retract = _retract_point(previous, lift)
        pre_point = PathPoint(
            position=np.asarray(entry.position, dtype=float) - entry_axis * lift,
            axis=entry_axis, roll_deg=float(entry.roll_deg), role=ROLE_JOINT,
            source=str(entry.source), stroke_id=entry.stroke_id)
        margin = np.inf
        seen = True
        for first, second in ((previous, retract), (retract, pre_point), (pre_point, entry)):
            found, tracker_ok = _segment_margin(first, second, cfg, station, model,
                                                visible=visible)
            margin = min(margin, found)
            seen = seen and tracker_ok
        if margin > 0.0 and seen and (fits is None or fits([retract, pre_point])):
            return [retract, pre_point], float(margin)
        # keep the roomiest attempt seen so far, for the same reason the two-corner route
        # does: a bigger lift is not monotonically safer
        if best is None or margin > best[0]:
            best = (float(margin), [retract, pre_point])
    margin, points = best
    return points, margin


def _two_corners(previous: PathPoint, entry: PathPoint, cfg: PlannerConfig,
                 station: Station, model: ScannerModel, rotation: float, same: bool, *,
                 visible=None, fits=None) -> tuple[list[PathPoint], float, float]:
    """The straight two-corner route: retract, traverse behind the next entry, dive in.

    Straight all the way through, and the fallback when a joint travel move is not
    allowed: the rotation is spread over both the flight and the dive, which is the most
    a straight-only program can do about a large pose change.
    """
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
        if margin > 0.0 and rate_ok and seen and (fits is None or fits(points)):
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
