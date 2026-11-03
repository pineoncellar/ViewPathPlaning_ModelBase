"""Turn a chosen set of strokes into one ordered, connected TCP path.

A stroke is an unordered, undirected segment; a program is a sequence. Closing that
gap is the whole job of this module:

* **order** - a greedy nearest-neighbour pass over the items, choosing which end to
  enter by, then single-stroke relocation passes. Not an optimum, and not claimed to
  be: what the workshop feels is the number of corners, not the last millimetre of
  travel. Distance alone is not the whole cost either - a short move between two
  unrelated poses costs a wrist turn, so the pose change is priced in as well.
* **connect** - between two items the head has to get from one to the other without
  dragging the scanner through the part. Two levels, in order: one joint move (no
  corner at all), then a lift ladder with one or two corners. A straight segment
  survives only between two identical poses, where it inserts nothing.
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
from .contracts import ConnectorResult
from .geometry import tangents, unit
from .kinematics import turn_digits
from .policies import connector_policy
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
_LIFT_FACTORS = (0.35, 0.6, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)

ROLE_COVERAGE = "coverage"
ROLE_RETRACT = "retract"
ROLE_TRAVERSE = "traverse"
ROLE_REORIENT = "reorient"

# The connector routes, cheapest first. Reported per transition so "the connector got
# cheaper" is a measurement rather than a claim.
LEVEL_JOINT = "J-direct"
LEVEL_STRAIGHT = "L0-straight"
LEVEL_ONE_CORNER = "L1-retract"
LEVEL_TWO_CORNERS = "L2-retract-through"

# A joint move crosses the wrist singularity when it takes J5 through zero, and the arm
# whips round there. The DP will do it rather than lose a sweep, but it should lose ties
# long before it does that. The price is `robot.wrist_flip_cost_deg`, defaulted there to the
# 180 this constant used to hold.
#
# How many sideways directions the `straight-then-lateral` chain tries before it gives up on
# a waiting point: evenly spread in the plane PERPENDICULAR to the optical axis, because a
# retract along the axis walks into whatever is behind the head.
_LATERAL_DIRECTIONS = 8

# What it costs the branch search to exceed `robot.j6_max_step_deg` on one move.  Two orders
# of magnitude above any real alternative, so the search only accepts an over-cap J6 move
# when no branch avoids it - and the report then names the segment instead of hiding it.
_J6_OVER_CAP_COST = 10_000.0

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
    # How the TCP arrives here, and how the P[n] block is written. None means "whatever the
    # program default is", which is what keeps a gate-off run byte-for-byte the old file.
    # A joint-representation point takes the controller's choice of solution away, and that
    # is the only way a planned branch survives to the machine.
    motion: str | None = None
    representation: str | None = None
    # The seventh axis: where the carriage sits when the TCP is here, in millimetres of
    # rail. Zero on a floor-mounted arm, and the reason the programme can carry an `E1`
    # column without every stored run changing.
    track_mm: float = 0.0


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
    # True when the joint state of every delivered point was decided HERE, by the sequencing
    # pass, rather than by a post-hoc `assign_branches` over the finished point list. The
    # difference matters: a point list already containing retract corners cannot be re-branched
    # without invalidating the reason those corners were inserted.
    joint_planned: bool = False
    # True when the seventh axis was part of this run. Zero rails and no rail are the same
    # number, so the flag is what keeps `E1` out of a floor-mounted program.
    track_enabled: bool = False
    # Transitions the joint move could not take, by reason, so "the connector got worse" has
    # a cause instead of a sentence that covers four of them.
    connector_failures: dict[str, int] = field(default_factory=dict)
    # Items that never made it into the program, by reason. Kept apart from `dropped` (which
    # says WHICH) so the report can say WHY without re-parsing a sentence: "the arm cannot hold
    # that pose" and "there is no legal way to get there" are different problems with different
    # fixes, and the old single violation line could not tell them apart.
    drop_reasons: dict[str, int] = field(default_factory=dict)
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
                  j6_weight: float = 0.0,
                  j6=None,
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

    `j6_weight` (mm per degree) prices the WRIST TWIST on top of that, using `j6`: the J6
    reading each item is entered with. It is a separate term because it is a separate cost -
    two poses a few degrees apart can need a hundred and eighty degrees of J6, and it is the
    axis that slides the scanner sideways where the body sphere cannot see. Measured on the
    2026-11-01 station: travel-only ordering 1353 deg of J6, the shipped order 1056, and
    `j6_weight = 10` 544 for 8 % more travel.
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
    twist = None if j6 is None or j6_weight <= 0.0 else np.asarray(j6, dtype=float)

    remaining = list(range(count))
    order: list[tuple[int, bool]] = []
    cursor = anchor
    previous = -1
    while remaining:
        best = None
        for index in remaining:
            for flip in (False, True):
                entry = ends[index] if flip else starts[index]
                cost = _edge(entry, cursor, gaps, previous, index, axis_weight,
                             twist, j6_weight)
                if best is None or cost < best[0]:
                    best = (cost, index, flip)
        _, index, flip = best
        order.append((index, flip))
        cursor = starts[index] if flip else ends[index]
        previous = index
        remaining.remove(index)

    order = _relocate(order, starts, ends, anchor, max_passes, gaps, axis_weight,
                      twist, j6_weight)
    return _two_opt(order, starts, ends, anchor, gaps, axis_weight,
                    limit=int(two_opt_limit), j6=twist, j6_weight=j6_weight)


def _twist_deg(j6, first: int, second: int) -> float:
    """|delta J6| between two items, circularly - the wrist turn the ordering pays for."""
    if j6 is None:
        return 0.0
    delta = float(j6[second]) - float(j6[first])
    return abs((delta + 180.0) % 360.0 - 180.0)


def _item_j6_readings(strokes: Sequence[Stroke], kin, weight: float):
    """The J6 reading every item would be entered with, and the weight to price it at.

    Returns `(None, 0.0)` - i.e. "do not use the term" - when the weight is off, the gate is
    off, or ANY item has no solution. A partial table would price half the program and order
    the other half on travel alone, and the two halves would then disagree about what
    "near" means; no term at all is the honest alternative.

    One reading per item, taken at the stroke's start pose. The two ends of a sweep share an
    axis and a roll, so they differ only in where the wrist ends up along it, which is small
    next to the between-item twist this term exists to avoid.
    """
    if weight <= 0.0 or kin is None or not getattr(kin, "active", False):
        return None, 0.0
    readings: list[float] = []
    for stroke in strokes:
        found = kin.solutions(stroke.start, stroke.axis, float(stroke.roll_deg))
        if not found:
            return None, 0.0
        readings.append(float(np.asarray(found[0].joints_deg, dtype=float)[5]))
    return np.asarray(readings, dtype=float), float(weight)


def _two_opt(order, starts, ends, anchor, gaps, axis_weight: float, *, limit: int = 40,
             max_passes: int = 8, j6=None, j6_weight: float = 0.0
             ) -> list[tuple[int, bool]]:
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
            if j6 is not None and j6_weight > 0.0:
                cost += j6_weight * _twist_deg(j6, from_index, to_index)
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


def _edge(entry, cursor, gaps, previous: int, index: int, axis_weight: float,
          j6=None, j6_weight: float = 0.0) -> float:
    """Travel to reach `entry` plus what the pose change and the wrist twist cost, in mm."""
    cost = float(np.linalg.norm(np.asarray(entry, dtype=float) - cursor))
    if previous < 0:
        return cost
    if gaps is not None and axis_weight > 0.0:
        cost += axis_weight * float(gaps[previous, index])
    if j6 is not None and j6_weight > 0.0:
        cost += j6_weight * _twist_deg(j6, previous, index)
    return cost


def _path_endpoints(order, starts, ends, anchor):
    """The entry and exit position of every stroke in the given order."""
    entry = np.array([ends[i] if flip else starts[i] for i, flip in order])
    exit_ = np.array([starts[i] if flip else ends[i] for i, flip in order])
    return entry, exit_


def _connector_length(order, starts, ends, anchor, gaps=None, axis_weight: float = 0.0,
                      j6=None, j6_weight: float = 0.0) -> float:
    """Travel plus pose change plus wrist twist along a trial order, in mm."""
    entry, exit_ = _path_endpoints(order, starts, ends, anchor)
    if len(order) == 0:
        return 0.0
    total = float(np.linalg.norm(entry[0] - anchor))
    if len(order) > 1:
        total += float(np.linalg.norm(entry[1:] - exit_[:-1], axis=1).sum())
        if gaps is not None and axis_weight > 0.0:
            for left, right in zip(order, order[1:], strict=False):
                total += axis_weight * float(gaps[left[0], right[0]])
        if j6 is not None and j6_weight > 0.0:
            for left, right in zip(order, order[1:], strict=False):
                total += j6_weight * _twist_deg(j6, left[0], right[0])
    return total


def _relocate(order, starts, ends, anchor, max_passes: int, gaps=None,
              axis_weight: float = 0.0, j6=None, j6_weight: float = 0.0
              ) -> list[tuple[int, bool]]:
    """Move one stroke at a time to a better slot, either way round, while it helps."""
    if len(order) < 3 or max_passes <= 0:
        return order
    best = list(order)
    best_cost = _connector_length(best, starts, ends, anchor, gaps, axis_weight,
                                  j6, j6_weight)
    for _ in range(max_passes):
        improved = False
        for source in range(len(best)):
            without = best[:source] + best[source + 1:]
            item = best[source]
            for target in range(len(without) + 1):
                for flip in (item[1], not item[1]):
                    candidate = without[:target] + [(item[0], flip)] + without[target:]
                    cost = _connector_length(candidate, starts, ends, anchor, gaps,
                                             axis_weight, j6, j6_weight)
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


def _program_targets(order, strokes: Sequence[Stroke], coverage_index, sources):
    """The poses the program must visit, in program order, with the item each belongs to.

    A zero-length stroke is a pose and yields ONE target; a real sweep yields two. That
    identity is what keeps the reported coverage and the delivered sweep the same number, and
    it has to be settled BEFORE the joint search, because a pose and a sweep differ in what
    the arm is asked to do between their ends (nothing, versus a straight line).
    """
    items = []
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
        leave = None
        if float(np.linalg.norm(end - start)) > _ZERO_MM:
            leave = PathPoint(position=end, axis=axis, roll_deg=float(stroke.roll_deg),
                              role=ROLE_COVERAGE, source=source, stroke_id=int(index),
                              coverage_index=row)
        items.append((int(index), entry, leave))
    return items


@dataclass
class _ItemJoints:
    """One item's joint state: the solution it is entered with, left with, or why neither."""

    start: object | None = None
    exit_deg: np.ndarray | None = None
    exit_margin_deg: float | None = None
    exit_track_mm: float = 0.0
    reason: str = ""


def _item_options(entry: PathPoint, leave: PathPoint | None, kin, cfg: PlannerConfig
                  ) -> tuple[list[tuple], str]:
    """Every (start solution, exit joints, worst margin) triple this item can be delivered with.

    The sweep is verified here and not later, and that is the point of the whole pass. For a
    joint move, "both ends inside the limits" is enough; for a scan it is not, because the
    scan is flown as a straight Cartesian line and the arm's branch has to carry it all the
    way. `Robot.follow` is what walks that line holding the branch it started on - the check
    whose absence let `X10290` be written, imported and then die at J2 in the middle of
    P[12]->P[13].

    An item with no triple at all cannot be delivered at any branch, and the caller drops it
    rather than filling in something plausible.
    """
    found = kin.solutions(entry.position, entry.axis, entry.roll_deg)
    if not found:
        return [], "no_ik_solution"
    if leave is None:
        return [(item, np.asarray(item.joints_deg, dtype=float),
                 float(item.min_margin_deg), float(item.track_mm)) for item in found], ""
    options = []
    reason = ""
    for item in found:
        # The carriage HOLDS through a sweep. A tilted band is one straight line at a fixed
        # attitude; sliding the rail while the head scans adds nothing the arm cannot do and
        # spends travel the programme has to pay for. So the walk is asked to end where it
        # started and the exit rail reading is the entry one.
        verdict = kin.follow(item.joints_deg, leave.position, leave.axis, leave.roll_deg,
                             step_mm=float(cfg.stroke.check_step_mm),
                             track_start=float(item.track_mm),
                             track_end=float(item.track_mm))
        if verdict.ok:
            options.append((item, np.asarray(verdict.joints_deg, dtype=float),
                            float(verdict.min_margin_deg), float(item.track_mm)))
        elif not reason:
            reason = f"stroke_{verdict.reason}"
    return options, ("" if options else reason or "stroke_unknown")


def _sequence_joints(items, kin, cfg: PlannerConfig, plan: PathPlan) -> list[_ItemJoints]:
    """Pick one joint state per item, as a SEQUENCE rather than item by item.

    Choosing the roomiest solution at every point is the mistake this exists to prevent:
    each point can be fine alone while the pair is not, because what connects them is a joint
    move whose cost is the joint travel. So this is the same small dynamic programme
    `assign_branches` runs over delivered points, moved one level up - over ITEMS, so that a
    sweep's exit joints are derived from its start solution instead of being chosen
    separately and then found to be inconsistent.

    A wrist flip (J5 changing sign) is allowed but priced at `robot.wrist_flip_cost_deg`,
    because the alternative may be losing the sweep and the controller does perform the move.
    J6 changes are priced at `robot.j6_weight` times their size, for the reason
    `_joint_step_deg` gives.

    A CONFIG family change is NOT allowed here, and that is not a preference: a straight
    Cartesian move cannot change family mid-line (the controller answers `MOTN-063`), the
    connector proves those lines wherever it can, and the delivery audit refuses to write a
    program whose recorded joints are on another family than the line arrives at. Pricing the
    change - which is what `robot.config_change_cost_deg` used to do - only made it expensive,
    so the sequencing could still buy it, and the whole run died at export on
    `branch_mismatch`.
    `robot.config_change_cost_deg` is now the price of the FALLBACK: when no single family can
    carry the chain, this pass relaxes to pricing (a joint move may legally change family) and
    records a violation saying so.
    """
    states = [_ItemJoints() for _ in items]
    weight = float(cfg.robot.j6_weight)
    cap = getattr(cfg.robot, "j6_max_step_deg", None)
    travel = float(getattr(cfg.robot, "j6_travel_weight", 0.0))
    flip_cost = float(getattr(cfg.robot, "wrist_flip_cost_deg", 0.0))
    family_cost = float(getattr(cfg.robot, "config_change_cost_deg", 0.0))
    track_price = float(getattr(cfg.robot, "track_cost_deg_per_mm", 0.0))
    options: list[list[tuple]] = []
    for position, (_index, entry, leave) in enumerate(items):
        found, reason = _item_options(entry, leave, kin, cfg)
        options.append(found)
        if not found:
            states[position].reason = reason
    keep = [position for position, found in enumerate(options) if found]
    if not keep:
        return states

    def solve(forbid_family: bool) -> tuple[list[list[float]], list[list[int]]]:
        """The dynamic programme, with a family change either impossible or just priced."""
        total: list[list[float]] = [[0.0] * len(options[keep[0]])]
        parent: list[list[int]] = [[0] * len(options[keep[0]])]
        for depth in range(1, len(keep)):
            previous = options[keep[depth - 1]]
            same_sweep = (
                items[keep[depth]][1].stroke_id is not None
                and items[keep[depth]][1].stroke_id == items[keep[depth - 1]][1].stroke_id)
            row_total: list[float] = []
            row_parent: list[int] = []
            for solution, _exit, _margin, _exit_track in options[keep[depth]]:
                best_cost, best_source = float("inf"), 0
                family = _config_family(solution)
                for source, (_prev_solution, prev_exit, _prev_margin,
                             prev_exit_track) in enumerate(previous):
                    if forbid_family and _config_family(_prev_solution) != family:
                        # Not a cheap move - an impossible one for the L that may fly it.
                        continue
                    cost = total[depth - 1][source] + _step_cost(prev_exit,
                                                                solution.joints_deg,
                                                                weight, cap, travel)
                    cost += _track_cost(prev_exit_track, float(solution.track_mm), track_price)
                    if float(np.asarray(prev_exit)[4]) * float(solution.joints_deg[4]) < 0.0:
                        cost += flip_cost
                    cost += _config_penalty(_prev_solution, solution, family_cost,
                                            doubled=same_sweep)
                    if cost < best_cost:
                        best_cost, best_source = cost, source
                row_total.append(best_cost)
                row_parent.append(best_source)
            total.append(row_total)
            parent.append(row_parent)
        return total, parent

    total, parent = solve(True)
    if not np.isfinite(np.asarray(total[-1], dtype=float)).any():
        total, parent = solve(False)
        plan.violations.append(
            "no single CONFIG family carries this program: the sequencing fell back to "
            "pricing family changes at robot.config_change_cost_deg, and any straight move "
            "that arrives on a family other than the one recorded will be refused at export")
    pick = int(np.argmin(total[-1]))
    for depth in range(len(keep) - 1, -1, -1):
        position = keep[depth]
        solution, exit_deg, exit_margin, exit_track = options[position][pick]
        states[position].start = solution
        states[position].exit_deg = exit_deg
        states[position].exit_margin_deg = exit_margin
        states[position].exit_track_mm = float(exit_track)
        if depth:
            pick = parent[depth][pick]
    return states


def _track_cost(first_mm, second_mm, price_deg_per_mm: float) -> float:
    """What sliding the carriage between two points costs the sequencing, in degrees.

    A rail is an axis: moving it is a move, and a programme that slides two metres to save
    five degrees of wrist has usually chosen wrong. Pricing it on the same scale as a joint
    step is what lets the search compare the two - `track_cost_deg_per_mm` is the exchange
    rate, and 0 makes the rail free, which is what a floor-mounted arm is.
    """
    if not price_deg_per_mm:
        return 0.0
    return abs(float(second_mm) - float(first_mm)) * float(price_deg_per_mm)


def _branch_record(points: Sequence[PathPoint], solutions: Sequence,
                   limit_deg: float) -> BranchAssignment:
    """The per-point branch record the report and the exporters read.

    Built from what was actually delivered rather than recomputed: a second search over the
    finished point list could choose a different branch for a corner, and then the CONFIG
    string would describe a move nobody verified.

    `broken` counts the steps over `limit_deg`, which is REPORTED and not enforced. Leaving the
    count at zero would make the report state "worst step 171.3, limit 40, over-limit steps 0",
    and a block that contradicts itself is worse than no block.
    """
    steps = [_joint_step_deg(solutions[index - 1].joints_deg, solutions[index].joints_deg)
             if solutions[index - 1] is not None and solutions[index] is not None else 0.0
             for index in range(1, len(points))]
    histogram: dict[str, int] = {}
    for item in solutions:
        if item is not None:
            key = item.config.split(",")[0].strip()
            histogram[key] = histogram.get(key, 0) + 1
    margins = [float(item.min_margin_deg) for item in solutions if item is not None]
    return BranchAssignment(
        solutions=list(solutions),
        failed=[index for index, item in enumerate(solutions) if item is None],
        broken=[index + 1 for index, step in enumerate(steps) if step > limit_deg],
        max_step_deg=float(max(steps)) if steps else 0.0,
        max_step_at=(int(np.argmax(steps)) + 1 if steps else None),
        min_margin_deg=(float(min(margins)) if margins else None),
        histogram=histogram)


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

    Three passes, in this order, because each one constrains the next:

    * the poses the program must visit, in program order (no geometry invented yet);
    * the joint state of every one of them, chosen as a sequence, with every sweep verified
      along its own straight line;
    * the move between them, cheapest primitive first, verified hop by hop.

    Doing joints in the middle is what makes the connector checkable. The old order - build
    the corners first, solve the joints afterwards - cannot check a segment, because by then
    the corners are fixed and the joint state that would have to carry the arm through them
    is a property of a point list that was already committed to.

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
    twist, twist_weight = _item_j6_readings(
        strokes, kin, float(getattr(cfg.stroke, "order_j6_weight_mm_per_deg", 0.0)))
    order = order_strokes(strokes, start_near,
                          axis_weight=float(cfg.stroke.order_axis_weight_mm_per_deg),
                          j6_weight=twist_weight, j6=twist)
    points: list[PathPoint] = []
    solutions: list = []
    plan = PathPlan(points=points, ordering=(
        "nearest neighbour + single-stroke relocation over stroke endpoints, priced by "
        "travel plus pose change"
        + (" plus wrist twist" if twist is not None else "")))
    plan.coverage_strokes = sum(1 for stroke in strokes if stroke.length_mm > _ZERO_MM)

    # 状态：定序位姿，不带任何连接几何
    targets = _program_targets(order, strokes, coverage_index, sources)
    joints = None
    if kin is not None and kin.active:
        joints = _sequence_joints(targets, kin, cfg, plan)
        plan.joint_planned = True
        plan.track_enabled = bool(getattr(kin.robot, "track_enabled", False))

    previous: PathPoint | None = None
    previous_joints = None
    previous_track = 0.0
    for position, (index, entry, leave) in enumerate(targets):
        state = None if joints is None else joints[position]
        if state is not None and state.start is None:
            plan.dropped.append(index)
            _note_drop(plan, state.reason or "no_joint_solution")
            plan.violations.append(
                f"stroke {index} dropped: {state.reason or 'no joint solution'}")
            continue
        start_joints = None if state is None else state.start.joints_deg
        start_track = 0.0 if state is None else float(getattr(state.start, "track_mm", 0.0))
        entry = PathPoint(
            position=entry.position, axis=entry.axis, roll_deg=entry.roll_deg,
            role=entry.role, source=entry.source, stroke_id=entry.stroke_id,
            coverage_index=entry.coverage_index,
            motion=("J" if state is not None else None),
            representation=("joint" if state is not None else None),
            track_mm=start_track)
        if previous is not None:
            routed = _connector(previous, entry, cfg, plan, station, model,
                                visible=visible, strict=hard,
                                kin=kin, start_joints=previous_joints,
                                end_joints=start_joints,
                                start_track=previous_track, end_track=start_track)
            if routed is None:
                plan.dropped.append(index)
                _note_drop(plan, "connector_no_route")
                plan.violations.append(
                    f"stroke {index} dropped: no legal connector "
                    f"({rotation_text(previous, entry)}, reasons "
                    f"{dict(plan.connector_failures)})")
                continue
            if state is not None and routed.level == LEVEL_STRAIGHT:
                # The connector proved a straight Cartesian line from the previous corner to
                # this one, so this is the instruction that flies it. Writing `J` here -
                # which is what every gate-on program did before 2026-11-01 - threw that
                # proof away: a `J` move interpolates joints, so the TCP arced between the
                # two sweeps and the wrist unwound whatever J6 the IK happened to pick.
                entry = _arrival(entry, routed.level)
            for corner, corner_solution in zip(routed.points,
                                               routed.solutions or [None] * len(
                                                   routed.points), strict=True):
                # A corner inserted by the connector is a pose like any other, so it carries
                # the rail reading its own solution was solved at. Without this the default
                # 0.0 would command the carriage back to the rail's zero on every detour -
                # outside the usable band, and a move nobody planned.
                corner_track = (float(getattr(corner_solution, "track_mm", 0.0))
                                if corner_solution is not None
                                else float(previous_track))
                points.append(PathPoint(
                    position=corner.position, axis=corner.axis,
                    roll_deg=corner.roll_deg, role=corner.role, source=corner.source,
                    stroke_id=corner.stroke_id,
                    motion=("J" if corner_solution is not None else None),
                    representation=("joint" if corner_solution is not None else None),
                    track_mm=corner_track))
                solutions.append(corner_solution)
        points.append(entry)
        solutions.append(None if state is None else state.start)
        # A zero-length stroke IS a pose: appending its "end" as well would deliver two
        # identical corners, which the point count and the .LS program would both carry.
        if leave is not None:
            points.append(PathPoint(
                position=leave.position, axis=leave.axis, roll_deg=leave.roll_deg,
                role=leave.role, source=leave.source, stroke_id=leave.stroke_id,
                coverage_index=leave.coverage_index,
                motion=("L" if state is not None else None),
                representation=("cart" if state is not None else None),
                track_mm=0.0 if state is None else float(state.exit_track_mm)))
            solutions.append(None if state is None else _exit_solution(state, kin.robot))
        previous = points[-1]
        previous_joints = None if state is None else state.exit_deg
        previous_track = 0.0 if state is None else float(state.exit_track_mm)

    if plan.joint_planned:
        _unwrap_readings(solutions, kin.robot)
        # Stage 4's last resort when the chain still commands an over-cap J6 step: re-try the
        # point's roll (a spin about the same axis keeps the sight line), and when even that
        # cannot help, leave the move and name it - the speed ladder already flies anything
        # over 60 deg at the slowest bucket, so "slow down" is not a promise, it is the default.
        eased = ease_over_cap_j6(
            points, solutions, kin, getattr(cfg.robot, "j6_max_step_deg", None),
            verify=lambda a, b: _joint_move(a, b, cfg, station, model,
                                            visible=visible, kin=kin)[0])
        plan.violations.extend(eased)
        # AGAIN, because `ease_over_cap_j6` replaces a point with a FRESHLY solved solution and
        # a fresh solve is placed at the roomiest equivalent reading, not at the continuous
        # one. Without this second pass the easing silently undid the unwrap: measured on the
        # 2026-11-02 rail run, the chain was unwrapped to J6 = +191.5 and the easing put back
        # -168.5, so P[14] -> P[15] commanded 299 degrees where 61 were meant. Every physical
        # pose is unchanged by this pass; only the numbers the program carries are.
        _unwrap_readings(solutions, kin.robot)
        plan.branches = _branch_record(points, solutions, float(cfg.robot.max_step_deg))
        plan.max_step_deg = plan.branches.max_step_deg
        warn = float(getattr(cfg.robot, "min_margin_warn_deg", 2.0))
        tight = [(index + 1, float(item.min_margin_deg))
                 for index, item in enumerate(solutions)
                 if item is not None and float(item.min_margin_deg) < warn]
        if tight:
            worst = min(tight, key=lambda pair: pair[1])
            plan.violations.append(
                f"{len(tight)} delivered point(s) sit within {warn:g} deg of the usable band "
                f"(worst P[{worst[0]}] at {worst[1]:.2f} deg). A controller with even a "
                "slightly different model can call those unreachable (MOTN-018), which is "
                "what 'rides the limits' means in practice; --joint-margin adds a cushion")
        if plan.branches.broken:
            plan.violations.append(
                f"{len(plan.branches.broken)} joint step(s) exceed max_step_deg "
                f"({plan.max_step_deg:.1f} > {cfg.robot.max_step_deg:g} deg); these are "
                "reported, not refused - import the program and watch those moves")
        positive = sum(1 for item in solutions
                       if item is not None and float(item.joints_deg[4]) > 0.0)
        if positive:
            # The file asks for one wrist branch and it can only ask for `N U T`. A plan on the
            # other one is not a plan the controller will fly, and the failure is silent until
            # the first L: MOTN-063, position configuration mismatch. Refusing to write it
            # quietly is the whole reason `robot.wrist_flip_policy` defaults to T.
            plan.violations.append(
                f"{positive} of {len(points)} delivered point(s) sit on the J5-positive wrist "
                "branch. The program writes Cartesian points with the constant CONFIG "
                "'N U T', and an L move cannot change the wrist branch, so those points will "
                "stop with MOTN-063 (and the letter F is refused at load with ASBN-023). Set "
                "robot.wrist_flip_policy to 'T'.")

    plan.path_length_mm = _travel(points)
    say(f"  path: {len(strokes)} items ({plan.coverage_strokes} strokes) -> "
        f"{plan.turns} corners ({plan.transition_points} transitions), "
        f"{plan.path_length_mm:.0f} mm, {plan.interrupts_a} translation interrupts, "
        f"{plan.interrupts_b} re-orientations"
        + (f", {len(plan.dropped)} dropped by the hard gate" if plan.dropped else ""))
    return plan


def _arrival(point: PathPoint, level: str) -> PathPoint:
    """The corner written as the instruction the connector actually proved.

    `LEVEL_STRAIGHT` is the connector saying "a straight Cartesian line from here to there is
    clear, within the turn rate, and keeps the branch" - so the program flies `L`. Every other
    level inserts corners or falls back to the joint move, and the arrival stays `J`.

    A point with no motion (`motion is None`) is a gate-off run, and it keeps whatever the
    program default is: that is the byte-for-byte regression profile and this must not touch
    it.
    """
    if point.motion is None or level != LEVEL_STRAIGHT:
        return point
    return replace(point, motion="L", representation="cart")


def _exit_solution(state: _ItemJoints, robot):
    """The far end of a sweep, as the walk left it.

    Not a `Solution`: nothing was solved at this pose, the walk derived it. The shoulder and
    elbow letters are carried over because the CONFIG string has to be re-derivable after the
    sequence is unwrapped to the nearest readings - the letters never change, only the turn
    digits.
    """
    if state.exit_deg is None or state.start is None:
        return None
    exit_deg = np.asarray(state.exit_deg, dtype=float)
    start = state.start
    margin = (float(state.exit_margin_deg) if state.exit_margin_deg is not None
              else float(start.min_margin_deg))
    return _BranchOnly(
        joints_deg=exit_deg, shoulder=str(start.shoulder), elbow=str(start.elbow),
        config=robot.config_string(exit_deg, start.shoulder, start.elbow),
        j5_margin_deg=float(start.j5_margin_deg),
        min_margin_deg=min(margin, float(start.min_margin_deg)),
        track_mm=float(state.exit_track_mm))


@dataclass(frozen=True)
class _BranchOnly:
    """A sweep's far end: the joints the verified walk arrived at, and its CONFIG hint."""

    joints_deg: np.ndarray
    shoulder: str
    elbow: str
    config: str
    j5_margin_deg: float
    min_margin_deg: float
    track_mm: float = 0.0




def _unwrap_readings(solutions: list, robot) -> None:
    """Rewrite the delivered joint readings so no axis travels the long way round.

    In place, over the finished sequence. The physical pose is untouched - only the number the
    controller is told changes - so nothing that was verified has to be verified again.

    Chain-wide, not point by point. The previous version was a single forward pass: each point
    took the reading nearest the one before it, so a point that had already been written a turn
    away dragged everything after it along. Measured on the 64-point program of 2026-10-30, one
    point carried J6 = -340.09 where +19.91 was meant, and the forward pass never recovered:
    the segment out of it commanded 332 degrees for a 28 degree move, and the file's total J6
    travel was 2439 degrees against a physical lower bound of 2135.

    What replaces it is a bottleneck shortest path over the whole chain: each point offers every
    equivalent reading of each axis that is still inside the usable band, and the choice
    minimises the LARGEST single-axis change anywhere on the path, with total travel breaking
    ties. Largest rather than total on purpose - "no segment may jump" is the requirement, and
    minimising the sum will happily move a jump from one segment to a better-looking one
    elsewhere. Both criteria are computed over the delivered order, so what is optimised is what
    the program commands. A few dozen states per point and a few hundred points, so this is exact
    and costs nothing next to the inverse kinematics that already ran.

    Every candidate must also be expressible as a CONFIG turn digit (non-negative), because
    that is the only column a Cartesian block has to name the revolution with. `+196.7` and
    `-163.3` are the same wrist pose; the first needs digit 1 and the second digit 0, and a
    chain that chooses the two sides of that seam at neighbouring points commands a 360 deg
    turn that nobody planned. This pass is idempotent and is run AFTER the J6 easing as well,
    for the same reason: a freshly solved point is placed roomiest, not continuously.
    """
    live = [index for index, item in enumerate(solutions) if item is not None]
    if len(live) < 2:
        return
    options: list[list[np.ndarray]] = []
    for index in live:
        values = np.asarray(solutions[index].joints_deg, dtype=float)
        per_axis = [robot.wraps(axis, float(values[axis])) or [float(values[axis])]
                    for axis in range(6)]
        # Feasibility is judged at the point's OWN rail reading. Asking about `track_mm=0` - the
        # default - makes every candidate infeasible on a rail that is in use, because 0 is the
        # carriage's end stop, so the whole pass silently did nothing on a rail run: this is why
        # the 2026-11-02 delivery kept a 307.9 deg J6 step between two readings that are 52 deg
        # apart.
        rail = float(getattr(solutions[index], "track_mm", 0.0) or 0.0)
        feasible = [np.array(combo, dtype=float) for combo in _readings_product(per_axis)
                    if robot.feasible(combo, rail)]
        # What the FILE can say comes first. A Cartesian block names the wrist revolution
        # through the CONFIG turn digits, and a negative digit is refused at load (ASBN-009,
        # measured 2026-10-27), so a reading of -200 can only be delivered as its equivalent
        # +160. Offering the inexpressible ones would win the bottleneck objective with a
        # number the exporter then has to refuse or silently rewrite. Readings with a
        # non-negative turn digit only, with the full set as the fallback so a chain that
        # cannot be expressed at all still gets its bottleneck minimised and reported.
        expressible = [combo for combo in feasible
                       if all(turn_digits(float(value)) >= 0 for value in combo)]
        options.append(expressible or feasible or [values])

    infinity = (float("inf"), float("inf"))
    total = [[infinity] * len(option) for option in options]
    parent = [[0] * len(option) for option in options]
    total[0] = [(0.0, 0.0)] * len(options[0])
    for depth in range(1, len(options)):
        for pick, candidate in enumerate(options[depth]):
            for source, previous in enumerate(options[depth - 1]):
                step = np.abs(candidate - previous)
                cost = (max(total[depth - 1][source][0], float(step.max())),
                        total[depth - 1][source][1] + float(step.sum()))
                if cost < total[depth][pick]:
                    total[depth][pick] = cost
                    parent[depth][pick] = source

    pick = min(range(len(options[-1])), key=lambda slot: total[-1][slot])
    chosen: list[np.ndarray] = [options[0][0]] * len(options)
    for depth in range(len(options) - 1, -1, -1):
        chosen[depth] = options[depth][pick]
        if depth:
            pick = parent[depth][pick]

    for slot, index in enumerate(live):
        values = chosen[slot]
        item = solutions[index]
        if float(np.abs(values - np.asarray(item.joints_deg, dtype=float)).max()) <= 1e-9:
            continue
        solutions[index] = replace(
            item, joints_deg=tuple(float(value) for value in values),
            config=robot.config_string(values, item.shoulder, item.elbow))


def ease_over_cap_j6(points: Sequence[PathPoint], solutions: Sequence, kin,
                     cap: float | None, *,
                     verify: Callable[[np.ndarray, np.ndarray], bool] | None = None) -> list[str]:
    """Re-try the roll of a point whose J6 step still exceeds `robot.j6_max_step_deg`.

    This is the fallback the merge agreed on, and it is deliberately narrow.  A roll about
    the optical axis does not change what the scanner looks at, but it does move the FLANGE,
    so for a transition point it is free to re-choose - that is what the candidate stage's
    retry list already does for the pool.  A COVERAGE point is left alone: its roll decides
    which samples its rectangular footprint reaches, so re-rolling it would move the very
    coverage figure the plan was accepted on.

    A candidate roll is only accepted when it does not cost the arm anything: no J5 sign
    flip, and no axis outside J6 travelling further than the move it replaces. Without that
    guard the pass would happily trade a 200 deg wrist twist for a 250 deg J1 swing and call
    it an improvement, because J6 is the one axis a roll touches directly.

    It is also accepted only when the worse of the two moves through the point - the one
    arriving and the one leaving - gets strictly better. A point in the middle of a chain is
    the endpoint of both, so judging one of them is how a "fix" turns into a 360 deg seam one
    point later; see the note in the body.

    `verify(a, b)` is the caller's joint-move check, run on BOTH sides of the point that gets
    re-rolled. It has to be both: the corner is an endpoint of the move arriving and of the
    move leaving, so verifying one leaves the other checked against joints that no longer
    exist. Without it the pass would be choosing a pose and asserting the arm can reach it.

    Returns the violations to record.  An over-cap move that survives is not an error: the
    speed ladder already flies anything over 60 deg at the slowest bucket, so the move is
    slow rather than illegal - but it is named, because "the program turns the wrist 200 deg
    while the report says 40" is the failure this project has already paid for once.
    """
    if cap is None or kin is None or not getattr(kin, "active", False):
        return []
    from .policies import ROLL_RETRY_DEG

    limit = float(cap)
    notes: list[str] = []
    stuck: list[int] = []
    for index in range(1, len(solutions)):
        first, second = solutions[index - 1], solutions[index]
        if first is None or second is None:
            continue
        if _j6_step_deg(first, second) <= limit:
            continue
        point = points[index]
        if point.role != ROLE_COVERAGE:
            arm_budget = _other_axes_deg(first, second)
            after = solutions[index + 1] if index + 1 < len(solutions) else None
            # A replacement is scored on the WORSE of the two moves that touch this point, not
            # just the one that was over the cap. Changing the reading of a point in the middle
            # of a chain moves both sides of it, and a pass that only looks at the arrival can
            # hold that one down while it sends the departure through a 360 deg seam - which is
            # not a hypothetical: on the 2026-11-02 rail run this function took J6 from +191.5
            # to -168.5, kept the 61 deg wrist turn it was aiming at, and the program commanded
            # 299 degrees for it. `_unwrap_readings` now runs again afterwards, but a pass that
            # cannot make the chain worse should not depend on the repair.
            budget = max(_j6_step_deg(first, second),
                         0.0 if after is None else _j6_step_deg(second, after))
            # A re-roll is a different ROLL of the same pose, not a different rail reading.
            # `kin.solutions` re-solves with the carriage free, so its answer can sit at another
            # offset - and the delivered `E1` comes from the route point, which this pass does
            # not change. Taking such an option would write joints the program's own rail
            # reading does not reproduce, which the delivery audit then reports as
            # `track_mismatch` and refuses.
            rail = float(getattr(point, "track_mm", 0.0) or 0.0)
            best = None
            for turn in ROLL_RETRY_DEG:
                roll = (float(point.roll_deg) + turn) % 360.0
                for option in kin.solutions(point.position, point.axis, roll):
                    if not option.ok:
                        continue
                    if abs(float(getattr(option, "track_mm", 0.0) or 0.0) - rail) > 1.0:
                        continue
                    score = max(_j6_step_deg(first, option),
                                0.0 if after is None else _j6_step_deg(option, after))
                    # Any strict improvement of the worse side is worth taking - a 200 deg
                    # move cut to 100 is still over the cap and still better - but an option
                    # under the cap always wins.
                    if score >= budget - 1e-9:
                        continue
                    if _j5_crossover(first, option):
                        continue
                    if _other_axes_deg(first, option) > arm_budget + 1e-9:
                        continue
                    if verify is not None and not verify(first.joints_deg, option.joints_deg):
                        continue
                    if (after is not None and verify is not None
                            and not verify(option.joints_deg, after.joints_deg)):
                        continue
                    key = (score > limit, score)
                    if best is None or key < best[0]:
                        best = (key, score, option)
            if best is not None:
                solutions[index] = best[2]
                notes.append(
                    f"point {index + 1}: roll re-tried to hold the J6 step under "
                    f"{limit:g} deg ({best[1]:.1f} deg)")
                continue
        stuck.append(index + 1)
    if stuck:
        notes.append(
            f"{len(stuck)} move(s) still turn J6 more than {limit:g} deg in one instruction "
            f"({stuck[:6]}); they are delivered because the speed ladder already flies them "
            "at the slowest bucket, and dropping the sweep would cost its coverage")
    return notes


def _readings_product(per_axis: list[list[float]]):
    """Every combination of the per-axis equivalent readings, as tuples."""
    combos: list[tuple[float, ...]] = [()]
    for options in per_axis:
        combos = [prefix + (float(value),) for prefix in combos for value in options]
    return combos




def rotation_text(previous: PathPoint, entry: PathPoint) -> str:
    """A short phrase for a violation line: how far the head has to turn."""
    return f"{_rotation_deg(previous, entry):.0f} deg"




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
    return _positions_margin(positions, station, model, visible=visible)


def _positions_margin(positions, station: Station, model: ScannerModel, *, visible=None
                      ) -> tuple[float, bool]:
    """Smallest body clearance over a set of TCP positions, and whether the tracker held.

    Separate from `_segment_margin` because a joint move does not sweep a straight line: the
    positions have to come from forward kinematics along the joint path, and then the same
    clearance test applies. An empty set is "nothing to check" and not a pass - callers that
    could not walk the path must say so themselves rather than read infinity as clearance.
    """
    if len(positions) == 0:
        return float("inf"), True
    stacked = np.asarray(positions, dtype=float)
    # `body_clearance_mm`, not `clearance_mm`: the head must fit past the PART as well. A
    # connector is a straight (or joint-interpolated) line between two legal poses and it can
    # cross the sheet in between - the fixture/base gate could not see that, so a straight
    # `L` move was written through the part on the 2026-11-03 rail run.
    margin = float(station.body_clearance_mm(stacked, limit_mm=model.body_limit_mm).min()
                   - model.body_limit_mm)
    seen = True if visible is None else bool(np.all(visible(stacked)))
    return margin, seen


def _cartesian_leg(previous: PathPoint, entry: PathPoint, ctx) -> tuple[bool, float, str]:
    """Whether the move into `entry` can be flown as a STRAIGHT CARTESIAN line, and why not.

    This is what makes `LEVEL_STRAIGHT` mean something. The connector used to test clearance
    alone, and the exporter then wrote the arrival as a `J` anyway, so the verification was
    thrown away and the arm interpolated joints through a path nobody had checked. Now the
    level decides the instruction, so the level has to be earned:

    `straight_blocked` - the head must fit everywhere along the line and the tracker must
      keep seeing it.
    `straight_rate` - an `L` move turns the head AS IT GOES, so the pose change has to be
      spread over the distance; `stroke.max_rot_deg_per_mm` is a measured lower bound.
    `straight_*` - a straight line can run out of branch: the controller solves the pose at
      every intermediate point and alarms when the branch it is on disappears.
      `KinematicGate.follow` is that test, and it is the same one the stroke walk uses.
    """
    cfg = ctx.cfg
    margin, seen = _segment_margin(previous, entry, cfg, ctx.station, ctx.model,
                                   visible=ctx.visible)
    if not (margin > 0.0 and seen):
        return False, margin, "straight_blocked"
    length = float(np.linalg.norm(np.asarray(entry.position, dtype=float)
                                  - np.asarray(previous.position, dtype=float)))
    rotation = _rotation_deg(previous, entry)
    if length <= 1e-6:
        # A "straight line" of zero length is a wrist turn in place, and it has no rate to
        # be within. Only an unchanged pose may be flown this way.
        if rotation > _SAME_POSE_DEG:
            return False, margin, "straight_stationary"
    elif rotation / length > float(cfg.stroke.max_rot_deg_per_mm):
        return False, margin, "straight_rate"
    if ctx.kin is not None and getattr(ctx.kin, "active", False) and ctx.start_joints is not None:
        verdict = ctx.kin.follow(ctx.start_joints, entry.position, entry.axis,
                                 entry.roll_deg, step_mm=float(cfg.stroke.check_step_mm),
                                 track_start=float(ctx.start_track),
                                 track_end=float(ctx.end_track))
        if not verdict.ok:
            return False, margin, f"straight_{verdict.reason}"
    return True, margin, ""


@dataclass
class Route:
    """What the connector decided: the corners it inserts, the route it took, and joints.

    `solutions` is one feasible joint solution per inserted corner, in the same order, or
    empty when the kinematic gate is off. It is carried here rather than recomputed at export
    time because the corner that was verified is the corner that must be written - solving it
    again later can land on another branch and quietly deliver a different move.
    """

    points: list[PathPoint] = field(default_factory=list)
    level: str = ""
    solutions: list = field(default_factory=list)


def _note_failure(plan: PathPlan, reason: str) -> None:
    """Count why a connector had to give up on one primitive, by name.

    The old code collapsed "clearance", "rate", "joint" and "nothing else left" into one
    sentence, which made "why did this sweep get dropped" unanswerable. Four counters cost
    nothing and answer it.
    """
    if not reason:
        return
    plan.connector_failures[reason] = plan.connector_failures.get(reason, 0) + 1


def _note_drop(plan: PathPlan, reason: str) -> None:
    """Count why an item never reached the program, by name."""
    key = reason or "unknown"
    plan.drop_reasons[key] = plan.drop_reasons.get(key, 0) + 1


def _config_family(item) -> str:
    """The posture letter of a joint solution: `N`/`F` for the shoulder, `U`/`D`, `T`/`B`.

    Only the first letter: it is the one the sequence changes by swinging the whole arm round,
    and the one the eight-pose bank could actually name.
    """
    return str(getattr(item, "config", "") or "").split(",")[0].strip()


def _config_penalty(first, second, cost_deg: float, *, doubled: bool = False) -> float:
    """What changing the CONFIG family between two points costs the sequencing.

    A charge, not a prohibition: a point whose only solution sits in another family keeps it.
    Zero `cost_deg` means the change is priced as free, which is the behaviour before this
    became configurable - the swing it hides is real either way, so the value is a site
    decision rather than a correction nobody can turn off.
    """
    if not cost_deg or first is None or second is None:
        return 0.0
    if _config_family(first) == _config_family(second):
        return 0.0
    return float(cost_deg) * (2.0 if doubled else 1.0)


def _joint_step_deg(first, second, j6_weight: float = 1.0) -> float:
    """Largest single-axis change between two joint sets, circularly.

    Circular because the controller takes the shortest path per axis: +179 and -181 are two
    degrees apart, not 360, and a straight subtraction would report a crossing that is not
    there.

    `j6_weight` > 1 prices a J6 change as if it were bigger than it is. J6 turns the flange
    about the tool axis, and the scanner sits about 250 mm off that axis, so one degree of J6
    slides the head 4.35 mm ALONG an arc that is roughly parallel to the surface being
    scanned - a direction the 45 mm body sphere cannot see anything in. Weighting it makes the
    two sequencing passes prefer a solution that leaves J6 alone whenever the alternatives are
    otherwise comparable. It never makes an infeasible move feasible: feasibility is decided
    elsewhere, by the limits and by the connector.
    """
    delta = (np.asarray(second, dtype=float) - np.asarray(first, dtype=float))
    steps = np.abs((delta + 180.0) % 360.0 - 180.0)
    if j6_weight != 1.0:
        steps = steps.copy()
        steps[5] *= float(j6_weight)
    return float(np.max(steps))


def _j6_step_deg(first, second) -> float:
    """|delta J6| between two joint sets, circularly: the turn about the flange axis.

    Accepts a Solution or a plain joint sequence, because the sequencing passes hold the
    former and the connector's verified chain holds the latter.
    """
    first = getattr(first, "joints_deg", first)
    second = getattr(second, "joints_deg", second)
    delta = float(np.asarray(second, dtype=float)[5]) - float(np.asarray(first, dtype=float)[5])
    return abs((delta + 180.0) % 360.0 - 180.0)


def _other_axes_deg(first, second) -> float:
    """Largest change outside J6 between two joint sets: what the ARM has to do.

    The counterpart of `_j6_step_deg`. A re-roll can always be made to look good on J6 alone,
    because J6 turns the flange about the optical axis and a roll IS a turn about that axis -
    so "hold J6 still" is only worth anything if the rest of the move does not grow to pay
    for it.
    """
    first = getattr(first, "joints_deg", first)
    second = getattr(second, "joints_deg", second)
    delta = np.asarray(second, dtype=float) - np.asarray(first, dtype=float)
    return float(np.max(np.abs((delta + 180.0) % 360.0 - 180.0)[:5]))


def _j5_crossover(first, second) -> bool:
    """Whether a joint move between the two passes through the wrist singularity."""
    first = np.asarray(getattr(first, "joints_deg", first), dtype=float).reshape(6)
    second = np.asarray(getattr(second, "joints_deg", second), dtype=float).reshape(6)
    return float(first[4]) * float(second[4]) < 0.0


def _step_cost(first, second, j6_weight: float = 1.0,
               j6_cap: float | None = None, j6_travel: float = 0.0) -> float:
    """What one segment costs the sequencing: the weighted step, plus its two J6 charges.

    The cap is a price, not a limit.  Refusing an over-cap move means dropping the sweep
    behind it and the coverage that sweep carried, so the delivery stage has cheaper answers:
    re-try the point at another roll about the same axis, or slow the move down.  A penalty
    an order of magnitude above any real alternative makes the search use one only when
    nothing else works - which is exactly the case the report then names.

    `j6_travel` is the second J6 charge and it is ADDITIVE. `j6_weight` only changes the
    answer when J6 is the worst axis of the move, so a small twist alongside a big arm swing
    costs nothing under it; `j6_travel` pays per degree regardless, which is the only way the
    search sees a sequence of small twists as the budget it is.
    """
    if first is None or second is None:
        return 0.0
    if hasattr(first, "joints_deg") or hasattr(second, "joints_deg"):
        cost = _step_between(first, second, j6_weight)
    else:
        cost = _joint_step_deg(first, second, j6_weight)
    j6 = _j6_step_deg(first, second)
    if j6_travel:
        cost += float(j6_travel) * j6
    if j6_cap is not None and j6 > float(j6_cap):
        cost += _J6_OVER_CAP_COST
    return cost


def _joint_move(first, second, cfg: PlannerConfig, station: Station, model: ScannerModel,
                *, visible=None, kin=None) -> tuple[bool, float, str]:
    """Whether a single joint move from `first` to `second` is deliverable.

    This is the primary connector primitive and the reason the plan got simpler. A joint move
    interpolates every axis linearly, so the joint-space path is a straight segment inside a
    box whose two ends are known to be legal - the limits cannot be crossed in the middle. It
    needs no intermediate point, so it needs no retract corner, so it needs no CONFIG, so the
    seven unsolvable points that the retract corners contributed simply do not exist.

    Three things are still checked:
    `j5_crossover` - J5 changing sign means the wrist goes through the singularity, where the
      arm whips round; allowed by the controller, not something to do without a reason.
    `collision` - the TCP does NOT travel a straight line when the joints do, so the swept
      path has to be sampled through forward kinematics.
    `tracker_blocked` - a transition still has to look like one, because the measurement must
      not be interrupted by the arm covering the tracker.
    """
    from_deg = np.asarray(first, dtype=float).reshape(6)
    to_deg = np.asarray(second, dtype=float).reshape(6)
    if _j5_crossover(from_deg, to_deg):
        return False, float("inf"), "j5_crossover"
    positions = kin.sweep_points(from_deg, to_deg,
                                 step_deg=float(cfg.stroke.joint_step_deg))
    if positions is None:
        return False, float("inf"), "unverified"
    margin, seen = _positions_margin(positions, station, model, visible=visible)
    if margin <= 0.0:
        return False, margin, "collision"
    if not seen:
        return False, margin, "tracker_blocked"
    return True, margin, ""


def _chain_route(first_joints, corners, entry_joints, cfg: PlannerConfig, station: Station,
                 model: ScannerModel, *, visible=None, kin=None):
    """Walk a fallback route hop by hop, or give up.

    Greedy on purpose: this is the route the joint move already refused, so spending a
    dynamic programme on it would cost more than the sweep it saves. What is not greedy is
    the verification - every hop is checked, including the last one into the entry, because
    the whole reason a segment check exists is a move that looked fine at both ends.

    Returns (solutions, ok). `solutions` is what the exporter must write, so the joints that
    were verified are the joints that get delivered.
    """
    if kin is None or not kin.active:
        return [None] * len(corners), True
    if first_joints is None:
        return [], False
    held = np.asarray(first_joints, dtype=float).reshape(6)
    chosen: list = []
    for corner in corners:
        options = kin.solutions(corner.position, corner.axis, corner.roll_deg)
        if not options:
            return [], False
        ordered = sorted(options, key=lambda item: _joint_step_deg(
            held, item.joints_deg, float(cfg.robot.j6_weight)))
        landed = None
        for item in ordered:
            ok, _margin, _reason = _joint_move(held, item.joints_deg, cfg, station, model,
                                               visible=visible, kin=kin)
            if ok:
                landed = item
                break
        if landed is None:
            return [], False
        chosen.append(landed)
        held = np.asarray(landed.joints_deg, dtype=float).reshape(6)
    if entry_joints is not None:
        ok, _margin, _reason = _joint_move(held, entry_joints, cfg, station, model,
                                           visible=visible, kin=kin)
        if not ok:
            return [], False
    return chosen, True


@dataclass
class ConnectorContext:
    """Everything slot B is allowed to read, passed explicitly instead of reached for.

    The chain used to be a function with eleven keyword arguments that also mutated the
    `plan` record; a policy gets one argument and returns one value.  `plan` is still in
    here because the failure counters are part of the report - but the chain only writes
    diagnostics into it, never decisions.
    """

    cfg: PlannerConfig
    plan: PathPlan
    station: Station
    model: ScannerModel
    visible: Any = None
    strict: bool = False
    kin: Any = None
    start_joints: Any = None
    end_joints: Any = None
    # The seventh axis at each end of the move. Equal values mean the carriage holds, which
    # is what a floor-mounted arm always has.
    start_track: float = 0.0
    end_track: float = 0.0


class ConnectorChain:
    """Slot B's shipped implementation: one joint move, then the lift ladder.

    Two levels, in this order, and no third:

    * **J-direct** - one joint move, no corner at all.  A joint move interpolates every
      axis linearly, so the joint-space path is a straight segment inside a box whose two
      ends are legal: the limits cannot be crossed in the middle, and no retract corner
      (and no CONFIG) is needed.  It is the primary route whenever the joint state of both
      ends is known.
    * **lift ladder** - leave along the surface normal in escalating steps, traverse, come
      back in.  One corner (`L1`) when the far end can be dived into directly, two (`L2`)
      when it cannot.  The lift is raised until the moves are clear AND the rotation is
      spread over enough travel to stay inside the rate limit, because a fixed lift
      assumes the space behind the head is empty and on this cell it is not.

    A straight segment (`L0`) survives only as the degenerate case where the two poses are
    identical: it inserts nothing, so it is not a level of its own.

    The old `L3` safe-plane route is gone.  It existed for transitions where the two
    sweeps look in different directions, but its corners are inserted points with their
    own inverse kinematics, and the measurement that motivated the rewrite was that
    inserted points - not the wrist - held every unsolvable point of the 99 % run.
    """

    name = "direct-then-lift"

    def connect(self, previous: PathPoint, entry: PathPoint,
                ctx: ConnectorContext) -> ConnectorResult | None:
        found = self.joint_leg(previous, entry, ctx)
        if found is not None:
            return found
        return self.fallback_legs(previous, entry, ctx)

    def joint_leg(self, previous: PathPoint, entry: PathPoint,
                  ctx: ConnectorContext) -> ConnectorResult | None:
        """One joint move between the two items' joint states, when both are known.

        A joint move cannot fail in the middle - that is the point of it - so this is the
        cheapest primitive and the one a chain should try before inserting anything.
        """
        cfg = ctx.cfg
        if not (bool(cfg.stroke.joint_connector) and ctx.kin is not None and ctx.kin.active
                and ctx.start_joints is not None and ctx.end_joints is not None):
            return None
        ok, margin, reason = _joint_move(ctx.start_joints, ctx.end_joints, cfg,
                                         ctx.station, ctx.model,
                                         visible=ctx.visible, kin=ctx.kin)
        if ok:
            return ConnectorResult(level=LEVEL_JOINT, margin_mm=margin)
        _note_failure(ctx.plan, reason)
        return None

    def fallback_legs(self, previous: PathPoint, entry: PathPoint,
                      ctx: ConnectorContext) -> ConnectorResult | None:
        """The retract routes: a straight move for identical poses, then the lift ladder."""
        cfg = ctx.cfg
        rotation = _rotation_deg(previous, entry)
        limit = int(cfg.stroke.connector_level)

        # A pair of identical poses still gets the old straight move when the level allows
        # it; changing that would move the gate-off program, which is the regression profile.
        if limit <= 0 and _same_pose(previous, entry):
            ok, margin, _reason = _cartesian_leg(previous, entry, ctx)
            if ok:
                return ConnectorResult(level=LEVEL_STRAIGHT, margin_mm=margin)

        # The lift ladder.  `connector_level` is how far down it may go.
        if limit <= 0:
            return None
        if limit <= 1:
            found = _one_corner(previous, entry, cfg, ctx.station, ctx.model, rotation,
                                visible=ctx.visible, kin=ctx.kin)
            if found is not None and _route_allowed(found[0], ctx.kin):
                corners, margin = found
                solutions, ok = _chain_route(ctx.start_joints, corners, ctx.end_joints,
                                             cfg, ctx.station, ctx.model,
                                             visible=ctx.visible, kin=ctx.kin)
                if ok:
                    return ConnectorResult(level=LEVEL_ONE_CORNER, corners=list(corners),
                                           joints=list(solutions), margin_mm=margin)
            _note_failure(ctx.plan, "retract")
            return None

        same = _same_pose(previous, entry)
        corners, margin, rate = _two_corners(previous, entry, cfg, ctx.station, ctx.model,
                                             rotation, same, visible=ctx.visible,
                                             kin=ctx.kin)
        solutions, chain_ok = _chain_route(ctx.start_joints, corners, ctx.end_joints, cfg,
                                           ctx.station, ctx.model, visible=ctx.visible,
                                           kin=ctx.kin)
        # The hard gate refuses on CLEARANCE and on joint feasibility. It does not refuse on
        # `rate`: how much rotation one millimetre of flight can absorb is a modelling
        # assumption whose constant (`max_rot_deg_per_mm`) was only ever bounded from below
        # on this cell, and turning an unmeasured heuristic into a dropped sweep makes the
        # plan look legal by making it worse. Rate failures are still recorded as violations
        # and counted in the report, so they stay visible rather than accepted.
        legal = margin > 0.0 and _route_allowed(corners, ctx.kin) and chain_ok
        if ctx.strict and not legal:
            _note_failure(ctx.plan, "retract_through" if margin > 0.0 else "collision")
            return None
        return ConnectorResult(level=LEVEL_TWO_CORNERS, corners=list(corners),
                               joints=list(solutions), margin_mm=margin, rate=rate)


class StraightThenLateral(ConnectorChain):
    """Slot B's second implementation: fly straight, sidestep only when the line is blocked.

    The order is the whole difference from `direct-then-lift`, and the reason it is a
    separate class rather than a flag is that the two disagree about what the FIRST move
    should be - a joint move interpolates in joint space, so the TCP arcs between two sweeps
    even when nothing is in the way, while a straight Cartesian line is what the program
    looks like it does.

    * **straight** - no corner at all, whatever `connector_level` says, as long as the line
      keeps clearance and the tracker. One instruction instead of three, and the wrist turn
      rides along on it rather than needing a corner to start turning at.
    * **lateral** - one waiting point SIDEWAYS of the part, over a ring of directions
      perpendicular to the optical axis and the lift ladder's distances. Sideways rather
      than backwards: retracting along the axis walks into whatever is behind the head,
      which on this cell is the base behind the upper face and the fixture behind the lower.
    * **the parent's chain** - joint move, then the lift ladder. This is the part that keeps
      the style from LOSING coverage: a straight line through the part is worse than a
      detour, and losing the sweep is worse than both.

    A straight line is only honest while it is clear, so it is sampled like any other move.
    What this deliberately does not do is fly a blocked line and note it (that is what the
    upstream lateral route does): an item is worth more than a transit through the part.
    """

    name = "straight-then-lateral"

    def connect(self, previous: PathPoint, entry: PathPoint,
                ctx: ConnectorContext) -> ConnectorResult | None:
        ok, margin, reason = _cartesian_leg(previous, entry, ctx)
        if ok:
            return ConnectorResult(level=LEVEL_STRAIGHT, margin_mm=margin)
        _note_failure(ctx.plan, reason)

        if int(ctx.cfg.stroke.connector_level) >= 1:
            found = _lateral_corner(previous, entry, ctx)
            if found is not None:
                return found
        return super().connect(previous, entry, ctx)


def _connector(previous: PathPoint, entry: PathPoint, cfg: PlannerConfig, plan: PathPlan,
               station: Station, model: ScannerModel, *,
               visible=None, strict: bool = False, kin=None,
               start_joints=None, end_joints=None,
               start_track: float = 0.0, end_track: float = 0.0) -> Route | None:
    """Run slot B's chain and record what it decided.

    The mechanics live in :class:`ConnectorChain`; this function is the adapter that keeps
    the old call site (and the report's counters) unchanged.  A second connector would be
    registered with `policies.connector_policy` and arrive here without touching `build_path`.
    """
    found = connector_policy(cfg.stroke.connector).connect(
        previous, entry,
        ConnectorContext(cfg=cfg, plan=plan, station=station, model=model,
                         visible=visible, strict=strict, kin=kin,
                         start_joints=start_joints, end_joints=end_joints,
                         start_track=float(start_track), end_track=float(end_track)))
    if found is None:
        return None
    same = _same_pose(previous, entry)
    rotation = _rotation_deg(previous, entry)
    _record(plan, same, found.margin_mm, rotation, found.rate, found.level, entry, cfg)
    return Route(found.corners, found.level, found.joints)


def _route_allowed(points, kin) -> bool:
    """Whether the arm can hold every transition corner a route would insert.

    Only that a solution EXISTS at each corner: this is the lift search's own filter, and
    asking it for a branch would be meaningless because a joint move may legitimately switch
    one. The hop-by-hop check, including the move into the entry, is `_chain_route`.
    """
    if kin is None or not kin.active:
        return True
    for point in points:
        if not kin.check(point.position, point.axis, point.roll_deg).ok:
            return False
    return True







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
    """A point `lift` millimetres from the head along `direction`, keeping its pose.

    `direction` defaults to straight back along the optical axis, which is what every route
    before 2026-11-01 used and what the lift ladder still means. The lateral route passes a
    direction in the plane perpendicular to the axis instead, so the waiting point leaves
    the part's volume without walking towards the base behind the head.
    """
    step = (-np.asarray(previous.axis, dtype=float) if direction is None
            else np.asarray(direction, dtype=float).reshape(3))
    return PathPoint(
        position=np.asarray(previous.position, dtype=float) + step * lift,
        axis=np.asarray(previous.axis, dtype=float), roll_deg=float(previous.roll_deg),
        role=ROLE_RETRACT, source=str(previous.source), stroke_id=previous.stroke_id)


def _one_corner(previous: PathPoint, entry: PathPoint, cfg: PlannerConfig,
                station: Station, model: ScannerModel, rotation: float, *,
                visible=None, kin=None, direction=None
                ) -> tuple[list[PathPoint], float] | None:
    """L1: retract, then fly straight to the next entry. None when no lift works."""
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


def _lateral_directions(previous: PathPoint,
                        count: int = _LATERAL_DIRECTIONS) -> list[np.ndarray]:
    """Unit directions in the plane PERPENDICULAR to the head's optical axis, evenly spread.

    Perpendicular on purpose: the retract this backs up along the axis walks into whatever
    is behind the head, which on this cell is the base behind the upper face and the fixture
    behind the lower one. A sidestep leaves the part's own volume immediately, whatever the
    head happens to be pointing at.
    """
    axis = unit(np.asarray(previous.axis, dtype=float))
    first, second = tangents(axis)
    angles = np.linspace(0.0, 2.0 * np.pi * (1.0 - 1.0 / count), count)
    return [unit(np.cos(angle) * first + np.sin(angle) * second) for angle in angles]


def _lateral_corner(previous: PathPoint, entry: PathPoint,
                    ctx: ConnectorContext) -> ConnectorResult | None:
    """One sideways waiting point, over the ring of directions and the lift ladder.

    The corner is verified like any other: `_one_corner` already refuses a direction whose
    waiting point keeps no clearance, and `_chain_route` is what turns the point into joint
    states the exporter can write. That second check is not optional - inserted points are
    exactly what the 99 % run could not solve, so a lateral corner that no branch can hold
    has to be refused here rather than delivered and found out at import.
    """
    rotation = _rotation_deg(previous, entry)
    for direction in _lateral_directions(previous):
        found = _one_corner(previous, entry, ctx.cfg, ctx.station, ctx.model, rotation,
                            visible=ctx.visible, kin=ctx.kin, direction=direction)
        if found is None:
            continue
        corners, margin = found
        solutions, ok = _chain_route(ctx.start_joints, corners, ctx.end_joints, ctx.cfg,
                                     ctx.station, ctx.model, visible=ctx.visible,
                                     kin=ctx.kin)
        if not ok:
            _note_failure(ctx.plan, "lateral_joint")
            continue
        return ConnectorResult(level=LEVEL_ONE_CORNER, corners=list(corners),
                               joints=list(solutions), margin_mm=margin)
    _note_failure(ctx.plan, "lateral")
    return None


def _two_corners(previous: PathPoint, entry: PathPoint, cfg: PlannerConfig,
                 station: Station, model: ScannerModel, rotation: float, same: bool, *,
                 visible=None, kin=None) -> tuple[list[PathPoint], float, float]:
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

    `fallback` says the no-family-change rule could not be met and the search relaxed to
    pricing changes: the plan may then contain a straight move whose recorded family is a
    different one, which the delivery audit refuses.
    """

    solutions: list = field(default_factory=list)
    failed: list[int] = field(default_factory=list)
    broken: list[int] = field(default_factory=list)
    max_step_deg: float = 0.0
    max_step_at: int | None = None
    min_margin_deg: float | None = None
    histogram: dict[str, int] = field(default_factory=dict)
    fallback: bool = False

    @property
    def ok(self) -> bool:
        return not self.failed and not self.broken


def assign_branches(points: Sequence[PathPoint], gate, max_step_deg: float,
                    j6_weight: float = 1.0,
                    j6_cap: float | None = None,
                    config_change_cost_deg: float = 0.0,
                    j6_travel: float = 0.0,
                    track_price: float = 0.0) -> BranchAssignment:
    """Pick a joint branch for every delivered point so the sequence jumps as little as it can.

    Picking the locally best branch at every point is the mistake this exists to prevent:
    each point may be fine on its own while the pair is not, because a `J` move interpolates
    in joint space. The search is a small dynamic programme - at most eight branches per
    point, a few hundred points - minimising the total joint travel subject to a hard limit
    on each step.

    `j6_weight` prices a J6 change as if it were that many times bigger, for the reason
    `_joint_step_deg` gives; it decides between candidates, never whether a point is feasible.
    The steps it REPORTS are always the real degrees, because that is what the program does.

    A CONFIG family change is treated as IMPOSSIBLE between two delivered points, for the
    same reason `_sequence_joints` does: the moves that get proven as straight Cartesian lines
    cannot change family, and the delivery audit refuses a program whose record disagrees with
    the line it flies (`branch_mismatch`). `config_change_cost_deg` is the price of the
    fallback, used only when no single family carries the whole chain; `fallback` in the result
    says that happened.

    `j6_travel` adds a charge per degree of J6 on top of the all-axis step, which is what
    makes a run of moderate twists visible to the search; it defaults to zero for the same
    reason.

    Points without any feasible solution are reported, never silently filled in with
    something plausible.
    """
    if gate is None or not gate.active:
        return BranchAssignment()
    candidates = [gate.solutions(point.position, point.axis, point.roll_deg)
                  for point in points]
    if not points:
        return BranchAssignment()
    failed = [index for index, options in enumerate(candidates) if not options]

    def solve(forbid_family: bool) -> tuple[list[list[float]], list[list[int]]]:
        # `total[i][j]` is the cheapest joint travel that ends on option j of point i;
        # `parent[i][j]` remembers which option of point i-1 got there.
        total = [[0.0] * len(options) for options in candidates]
        parent: list[list[int]] = [[0] * len(options) for options in candidates]
        for index in range(1, len(candidates)):
            for j, option in enumerate(candidates[index]):
                family = _config_family(option)
                best = (float("inf"), 0)
                for k, previous in enumerate(candidates[index - 1]):
                    if forbid_family and _config_family(previous) != family:
                        continue
                    cost = total[index - 1][k] + _step_cost(previous, option,
                                                            j6_weight, j6_cap, j6_travel)
                    cost += _track_cost(getattr(previous, "track_mm", 0.0),
                                        getattr(option, "track_mm", 0.0), track_price)
                    cost += _config_penalty(previous, option, config_change_cost_deg)
                    if cost < best[0]:
                        best = (cost, k)
                total[index][j] = best[0]
                parent[index][j] = best[1]
        return total, parent

    total, parent = solve(True)
    fallback = False
    if not np.isfinite(np.asarray(total[-1], dtype=float)).any():
        total, parent = solve(False)
        fallback = True
    choice = int(np.argmin(total[-1])) if candidates[-1] else 0
    picked: list = [None] * len(points)
    for index in range(len(candidates) - 1, -1, -1):
        if not candidates[index]:
            continue
        picked[index] = candidates[index][choice]
        if index > 0:
            choice = parent[index][choice]

    steps = [_step_between(picked[index - 1], picked[index])
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
        min_margin_deg=(float(min(margins)) if margins else None),
        histogram=histogram, fallback=fallback)


def _step_between(first, second, j6_weight: float = 1.0) -> float:
    """Largest single-axis joint change between two solutions, in degrees.

    Circular: a reading of +179 and one of -181 are two degrees apart, not 360. The
    controller takes the shortest path per axis, so the sequence has to be judged the same
    way - otherwise a wrist sitting near the wrap is reported as a 358 degree jump and the
    continuity constraint looks violated when it is not.

    `j6_weight` is a cost multiplier for the ordering pass only; callers that REPORT a step
    leave it at 1.0 so the number they print is the number the machine will move.
    """
    if first is None or second is None:
        return 0.0
    delta = (np.asarray(second.joints_deg, dtype=float)
             - np.asarray(first.joints_deg, dtype=float))
    steps = np.abs((delta + 180.0) % 360.0 - 180.0)
    if j6_weight != 1.0:
        steps = steps.copy()
        steps[5] *= float(j6_weight)
    return float(np.max(steps))


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

