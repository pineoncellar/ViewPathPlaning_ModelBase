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
# Multiples of `retract_clearance_mm` the connector tries before giving up. A fixed lift
# assumes the space behind the head is empty, and on a real cell it is not: the base sits
# behind the upper face and the fixture behind the lower one.
_LIFT_FACTORS = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0)

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
LEVEL_SAFE_PLANE = "L3-safe-plane"

# A joint move crosses the wrist singularity when it takes J5 through zero, and the arm
# whips round there. The DP will do it rather than lose a sweep, but it should lose ties
# long before it does that.
_J5_FLIP_PENALTY_DEG = 180.0

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
                 float(item.min_margin_deg)) for item in found], ""
    options = []
    reason = ""
    for item in found:
        verdict = kin.follow(item.joints_deg, leave.position, leave.axis, leave.roll_deg,
                             step_mm=float(cfg.stroke.check_step_mm))
        if verdict.ok:
            options.append((item, np.asarray(verdict.joints_deg, dtype=float),
                            float(verdict.min_margin_deg)))
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

    A wrist flip (J5 changing sign) is allowed but priced at `_J5_FLIP_PENALTY_DEG`, because
    the alternative may be losing the sweep and the controller does perform the move. J6
    changes are priced at `robot.j6_weight` times their size, for the reason
    `_joint_step_deg` gives.
    """
    states = [_ItemJoints() for _ in items]
    weight = float(cfg.robot.j6_weight)
    options: list[list[tuple]] = []
    for position, (_index, entry, leave) in enumerate(items):
        found, reason = _item_options(entry, leave, kin, cfg)
        options.append(found)
        if not found:
            states[position].reason = reason
    keep = [position for position, found in enumerate(options) if found]
    if not keep:
        return states
    total: list[list[float]] = [[0.0] * len(options[keep[0]])]
    parent: list[list[int]] = [[0] * len(options[keep[0]])]
    for depth in range(1, len(keep)):
        previous = options[keep[depth - 1]]
        row_total: list[float] = []
        row_parent: list[int] = []
        for solution, _exit, _margin in options[keep[depth]]:
            best_cost, best_source = float("inf"), 0
            for source, (_prev_solution, prev_exit, _prev_margin) in enumerate(previous):
                cost = total[depth - 1][source] + _joint_step_deg(prev_exit,
                                                                  solution.joints_deg,
                                                                  weight)
                if float(np.asarray(prev_exit)[4]) * float(solution.joints_deg[4]) < 0.0:
                    cost += _J5_FLIP_PENALTY_DEG
                if cost < best_cost:
                    best_cost, best_source = cost, source
            row_total.append(best_cost)
            row_parent.append(best_source)
        total.append(row_total)
        parent.append(row_parent)
    pick = int(np.argmin(total[-1]))
    for depth in range(len(keep) - 1, -1, -1):
        position = keep[depth]
        solution, exit_deg, exit_margin = options[position][pick]
        states[position].start = solution
        states[position].exit_deg = exit_deg
        states[position].exit_margin_deg = exit_margin
        if depth:
            pick = parent[depth][pick]
    return states


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
    order = order_strokes(strokes, start_near,
                          axis_weight=float(cfg.stroke.order_axis_weight_mm_per_deg))
    normal, plane = _safe_plane(strokes, cfg)
    points: list[PathPoint] = []
    solutions: list = []
    plan = PathPlan(points=points, ordering=(
        "nearest neighbour + single-stroke relocation over stroke endpoints, priced by "
        "travel plus pose change"))
    plan.coverage_strokes = sum(1 for stroke in strokes if stroke.length_mm > _ZERO_MM)

    # 状态：定序位姿，不带任何连接几何
    targets = _program_targets(order, strokes, coverage_index, sources)
    joints = None
    if kin is not None and kin.active:
        joints = _sequence_joints(targets, kin, cfg, plan)
        plan.joint_planned = True

    previous: PathPoint | None = None
    previous_joints = None
    for position, (index, entry, leave) in enumerate(targets):
        state = None if joints is None else joints[position]
        if state is not None and state.start is None:
            plan.dropped.append(index)
            _note_drop(plan, state.reason or "no_joint_solution")
            plan.violations.append(
                f"stroke {index} dropped: {state.reason or 'no joint solution'}")
            continue
        start_joints = None if state is None else state.start.joints_deg
        entry = PathPoint(
            position=entry.position, axis=entry.axis, roll_deg=entry.roll_deg,
            role=entry.role, source=entry.source, stroke_id=entry.stroke_id,
            coverage_index=entry.coverage_index,
            motion=("J" if state is not None else None),
            representation=("joint" if state is not None else None))
        if previous is not None:
            routed = _connector(previous, entry, cfg, plan, station, model,
                                visible=visible, strict=hard, normal=normal, plane=plane,
                                kin=kin, start_joints=previous_joints,
                                end_joints=start_joints)
            if routed is None:
                plan.dropped.append(index)
                _note_drop(plan, "connector_no_route")
                plan.violations.append(
                    f"stroke {index} dropped: no legal connector "
                    f"({rotation_text(previous, entry)}, reasons "
                    f"{dict(plan.connector_failures)})")
                continue
            for corner, corner_solution in zip(routed.points,
                                               routed.solutions or [None] * len(
                                                   routed.points), strict=True):
                points.append(PathPoint(
                    position=corner.position, axis=corner.axis,
                    roll_deg=corner.roll_deg, role=corner.role, source=corner.source,
                    stroke_id=corner.stroke_id,
                    motion=("J" if corner_solution is not None else None),
                    representation=("joint" if corner_solution is not None else None)))
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
                representation=("cart" if state is not None else None)))
            solutions.append(None if state is None else _exit_solution(state, kin.robot))
        previous = points[-1]
        previous_joints = None if state is None else state.exit_deg

    if plan.joint_planned:
        _unwrap_readings(solutions, kin.robot)
        plan.branches = _branch_record(points, solutions, float(cfg.robot.max_step_deg))
        plan.max_step_deg = plan.branches.max_step_deg
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
        min_margin_deg=min(margin, float(start.min_margin_deg)))


@dataclass(frozen=True)
class _BranchOnly:
    """A sweep's far end: the joints the verified walk arrived at, and its CONFIG hint."""

    joints_deg: np.ndarray
    shoulder: str
    elbow: str
    config: str
    j5_margin_deg: float
    min_margin_deg: float


def _nearest_readings(values, previous, robot) -> np.ndarray:
    """Every axis moved to the equivalent reading closest to `previous`, if still usable.

    `Robot._place` picks the roomiest reading for a pose on its own, and two neighbouring poses
    can come back with readings almost a full turn apart. Measured on the delivered 59-point
    plan before this pass existed: 18 of 58 segments had a literal J6 change over 180 degrees,
    the worst 352.6 degrees for a 17 degree change of the pose. A joint move interpolates the
    NUMBER it is given, not the shortest angle, so that reading buys a wrist spinning almost
    all the way round for nothing.
    """
    moved = np.asarray(values, dtype=float).copy()
    for axis in range(6):
        options = robot.wraps(axis, float(moved[axis]))
        if options:
            moved[axis] = min(options, key=lambda value: abs(value - float(previous[axis])))
    # the alternative readings stay inside the controller limits by construction, but not
    # necessarily inside the do-not-ride-the-limit band, so the policy gets the last word
    return moved if robot.feasible(moved) else np.asarray(values, dtype=float)


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
    """
    live = [index for index, item in enumerate(solutions) if item is not None]
    if len(live) < 2:
        return
    options: list[list[np.ndarray]] = []
    for index in live:
        values = np.asarray(solutions[index].joints_deg, dtype=float)
        per_axis = [robot.wraps(axis, float(values[axis])) or [float(values[axis])]
                    for axis in range(6)]
        feasible = [np.array(combo, dtype=float) for combo in _readings_product(per_axis)
                    if robot.feasible(combo)]
        options.append(feasible or [values])

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


def _readings_product(per_axis: list[list[float]]):
    """Every combination of the per-axis equivalent readings, as tuples."""
    combos: list[tuple[float, ...]] = [()]
    for options in per_axis:
        combos = [prefix + (float(value),) for prefix in combos for value in options]
    return combos




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
    margin = float(station.clearance_mm(stacked).min() - model.body_limit_mm)
    seen = True if visible is None else bool(np.all(visible(stacked)))
    return margin, seen


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
    if float(from_deg[4]) * float(to_deg[4]) < 0.0:
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


def _connector(previous: PathPoint, entry: PathPoint, cfg: PlannerConfig, plan: PathPlan,
               station: Station, model: ScannerModel, *,
               visible=None, strict: bool = False, normal=None, plane=None, kin=None,
               start_joints=None, end_joints=None) -> Route | None:
    """The corners that take the head from `previous` to `entry`, and which route won.

    Routes, cheapest first, and `stroke.connector_level` says how far down the planner may
    go:

    * **J-direct** - one joint move, no corner at all. The primary route whenever the joint
      state of both ends is known, which is whenever the kinematic gate is on. See
      `_joint_move` for why this cannot fail in the middle.
    * **L3** - leave through the common safe plane and come back in on the other side. Used
      only when the two sweeps look in different directions, because that is the case the
      per-axis retract gets wrong. Now a fallback: it exists for the transitions where the
      joint move collides.
    * **L0** - straight there, no corner at all. Only for two identical poses, and only
      when that line keeps its clearance. Off by default: the "body" in this model is a
      sphere, which does not rotate with the head, so the model cannot see the part of a
      real handset that swings round behind the reference point.
    * **L1** - retract along -axis, then fly to the next entry point.
    * **L2** - retract, traverse to a point behind the next entry, dive in. Two corners.

    The retracting routes have to agree with the clearance check, because a fixed lift
    assumes the space behind the head is empty and on this cell it is not: behind the upper
    face is the base, behind the lower face is the floor side of the fixture. So the lift is
    raised until the moves are clear AND the rotation is spread over enough travel to stay
    inside the rate limit.

    Returns None when no route qualifies and `strict` is set - that is the hard gate, and the
    caller turns it into "this item is not delivered" rather than a line in a report.
    """
    same = _same_pose(previous, entry)
    rotation = _rotation_deg(previous, entry)
    limit = int(cfg.stroke.connector_level)

    # 状态：先试关节直连，它既不需要中间点，也不存在中段超限
    if (bool(cfg.stroke.joint_connector) and kin is not None and kin.active
            and start_joints is not None and end_joints is not None):
        ok, margin, reason = _joint_move(start_joints, end_joints, cfg, station, model,
                                         visible=visible, kin=kin)
        if ok:
            _record(plan, same, margin, rotation, 0.0, LEVEL_JOINT, entry, cfg)
            return Route([], LEVEL_JOINT)
        _note_failure(plan, reason)

    if normal is not None and _crosses_direction(previous, entry):
        found = _cross_group(previous, entry, cfg, station, model, rotation,
                             normal=normal, plane=plane, visible=visible)
        if found is not None and _route_allowed(found[0], kin):
            corners, margin = found
            solutions, ok = _chain_route(start_joints, corners, end_joints, cfg, station,
                                         model, visible=visible, kin=kin)
            if ok:
                _record(plan, same, margin, rotation, 0.0, LEVEL_SAFE_PLANE, entry, cfg)
                return Route(corners, LEVEL_SAFE_PLANE, solutions)
        _note_failure(plan, "safe_plane")

    if limit <= 0 and same:
        margin, seen = _segment_margin(previous, entry, cfg, station, model,
                                       visible=visible)
        if margin > 0.0 and seen:
            _record(plan, same, margin, rotation, 0.0, LEVEL_STRAIGHT, entry, cfg)
            return Route([], LEVEL_STRAIGHT)

    if limit <= 1:
        found = _one_corner(previous, entry, cfg, station, model, rotation,
                            visible=visible, kin=kin)
        if found is not None and _route_allowed(found[0], kin):
            corners, margin = found
            solutions, ok = _chain_route(start_joints, corners, end_joints, cfg, station,
                                         model, visible=visible, kin=kin)
            if ok:
                _record(plan, same, margin, rotation, 0.0, LEVEL_ONE_CORNER, entry, cfg)
                return Route(corners, LEVEL_ONE_CORNER, solutions)
        _note_failure(plan, "retract")

    corners, margin, rate = _two_corners(previous, entry, cfg, station, model, rotation,
                                         same, visible=visible, kin=kin)
    solutions, chain_ok = _chain_route(start_joints, corners, end_joints, cfg, station,
                                       model, visible=visible, kin=kin)
    # The hard gate refuses on CLEARANCE and on joint feasibility. It does not refuse on
    # `rate`: how much rotation one millimetre of flight can absorb is a modelling
    # assumption whose constant (`max_rot_deg_per_mm`) was only ever bounded from below on
    # this cell, and turning an unmeasured heuristic into a dropped sweep makes the plan look
    # legal by making it worse. Rate failures are still recorded as violations and counted
    # in the report, so they stay visible rather than accepted.
    legal = margin > 0.0 and _route_allowed(corners, kin) and chain_ok
    _record(plan, same, margin, rotation, rate, LEVEL_TWO_CORNERS, entry, cfg)
    if strict and not legal:
        _note_failure(plan, "retract_through" if margin > 0.0 else "collision")
        return None
    return Route(corners, LEVEL_TWO_CORNERS, solutions)


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


def _retract_point(previous: PathPoint, lift: float) -> PathPoint:
    """A point `lift` millimetres behind the head, keeping its pose."""
    return PathPoint(
        position=np.asarray(previous.position, dtype=float)
        - np.asarray(previous.axis, dtype=float) * lift,
        axis=np.asarray(previous.axis, dtype=float), roll_deg=float(previous.roll_deg),
        role=ROLE_RETRACT, source=str(previous.source), stroke_id=previous.stroke_id)


def _one_corner(previous: PathPoint, entry: PathPoint, cfg: PlannerConfig,
                station: Station, model: ScannerModel, rotation: float, *,
                visible=None, kin=None) -> tuple[list[PathPoint], float] | None:
    """L1: retract, then fly straight to the next entry. None when no lift works."""
    for factor in _LIFT_FACTORS:
        retract = _retract_point(previous, float(cfg.stroke.retract_clearance_mm) * factor)
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
    """

    solutions: list = field(default_factory=list)
    failed: list[int] = field(default_factory=list)
    broken: list[int] = field(default_factory=list)
    max_step_deg: float = 0.0
    max_step_at: int | None = None
    min_margin_deg: float | None = None
    histogram: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failed and not self.broken


def assign_branches(points: Sequence[PathPoint], gate, max_step_deg: float,
                    j6_weight: float = 1.0) -> BranchAssignment:
    """Pick a joint branch for every delivered point so the sequence jumps as little as it can.

    Picking the locally best branch at every point is the mistake this exists to prevent:
    each point may be fine on its own while the pair is not, because a `J` move interpolates
    in joint space. The search is a small dynamic programme - at most eight branches per
    point, a few hundred points - minimising the total joint travel subject to a hard limit
    on each step.

    `j6_weight` prices a J6 change as if it were that many times bigger, for the reason
    `_joint_step_deg` gives; it decides between candidates, never whether a point is feasible.
    The steps it REPORTS are always the real degrees, because that is what the program does.

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
    # `total[i][j]` is the cheapest joint travel that ends on option j of point i;
    # `parent[i][j]` remembers which option of point i-1 got there.
    total = [[0.0] * len(options) for options in candidates]
    parent: list[list[int]] = [[0] * len(options) for options in candidates]
    for index in range(1, len(candidates)):
        for j, option in enumerate(candidates[index]):
            best = (float("inf"), 0)
            for k, previous in enumerate(candidates[index - 1]):
                cost = total[index - 1][k] + _step_between(previous, option, j6_weight)
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
        histogram=histogram)


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
