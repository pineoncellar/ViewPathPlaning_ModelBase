"""The joint-limit gate: where it rejects, what it reports, and how branches are assigned.

The robot geometry itself is tested in `test_kinematics` with synthetic joint angles. This
file tests the SEAM: that a pose the arm cannot hold never reaches the coverage matrix, that
a stroke which leaves the envelope stops growing, that the uncovered samples are attributed
to the joint limits rather than to a layout problem, and that the delivered path gets one
consistent branch per point.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from scanplan.config import RobotParams
from scanplan.coverage import attribute_uncovered, evaluate
from scanplan.kinematics import KinematicGate
from scanplan.path import (
    LEVEL_JOINT,
    ROLE_COVERAGE,
    PathPoint,
    _BranchOnly,
    _joint_move,
    _nearest_readings,
    _unwrap_readings,
    assign_branches,
    build_path,
)
from scanplan.scanner_model import ScannerModel, ScannerPose
from scanplan.strokes import Stroke
from scanplan.target_surface import SampleSet


def samples(size_mm=100.0, step_mm=10.0, z_mm=700.0, x_mm=2000.0) -> SampleSet:
    """A flat grid of target samples where the arm can actually hold the scanner.

    Deliberately not at the origin: the test station puts the part where a real cell would
    not - right on top of the robot base - and a pose 300 mm above it is inside the arm.
    The gate is right to reject that, so the fixtures used here sit in the workspace.
    """
    coordinates = np.arange(-size_mm / 2.0, size_mm / 2.0 + step_mm / 2.0, step_mm)
    points = np.array([[x_mm + x, y, z_mm] for x in coordinates for y in coordinates])
    normals = np.tile([0.0, 0.0, 1.0], (len(points), 1))
    return SampleSet(points=points, normals=normals,
                     area_mm2=np.full(len(points), 1.0),
                     total_area_mm2=float(len(points)))


def overhead(position=(2000.0, 0.0, 1000.0)) -> ScannerPose:
    """A pose looking straight down at the sample grid."""
    return ScannerPose(position=np.asarray(position, dtype=float),
                       axis=np.array([0.0, 0.0, -1.0]), roll_deg=0.0)


def gate(**changes) -> KinematicGate:
    """A gate over the measured machine, with the fields a test wants to break."""
    params = RobotParams.disabled()
    return KinematicGate(RobotParams(**{**params.__dict__, "enabled": True, **changes}))


def test_an_unreachable_robot_rejects_every_pose(clamped_plate, cfg):
    """The gate sits in the rejection chain, so a pose it refuses covers nothing."""
    model = ScannerModel(cfg.scanner)
    tiny = gate(upper_arm_mm=1.0, forearm_mm=2.0, forearm_offset_mm=1.0)
    poses = [overhead()]
    result = evaluate(poses, samples(), clamped_plate.station, model, cfg, gate=tiny)
    assert result.counters["rejected_kinematics"] == 1
    assert len(result.sets[0]) == 0


def test_a_disabled_gate_changes_nothing(clamped_plate, cfg):
    """`robot.enabled=false` must reproduce today's result exactly."""
    model = ScannerModel(cfg.scanner)
    poses = [overhead()]
    plain = evaluate(poses, samples(), clamped_plate.station, model, cfg)
    off = evaluate(poses, samples(), clamped_plate.station, model, cfg,
                   gate=KinematicGate(replace(cfg.robot, enabled=False)))
    assert plain.counters["rejected_kinematics"] == 0
    assert off.counters["rejected_kinematics"] == 0
    assert np.array_equal(plain.sets[0], off.sets[0])


def test_a_reachable_pose_still_covers(clamped_plate, cfg):
    """The gate must not reject a pose the arm can actually hold."""
    model = ScannerModel(cfg.scanner)
    poses = [overhead()]
    result = evaluate(poses, samples(), clamped_plate.station, model, cfg, gate=gate())
    assert result.counters["rejected_kinematics"] == 0
    assert np.count_nonzero(result.sets[0]) > 0


def test_uncovered_samples_from_a_limited_pose_are_attributed_to_the_joint_limit(
        clamped_plate, cfg):
    """R5: "the arm cannot hold that pose" must not be reported as "the layout is wrong"."""
    model = ScannerModel(cfg.scanner)
    tiny = gate(upper_arm_mm=1.0, forearm_mm=2.0, forearm_offset_mm=1.0)
    probes = np.array([[2000.0, 0.0, 700.0]])
    normals = np.array([[0.0, 0.0, 1.0]])
    poses = [overhead()]
    reasons = attribute_uncovered(probes, normals, poses, clamped_plate.station, model, cfg,
                                  gate=tiny)
    assert list(reasons) == ["unreachable_joint_limit"]


def test_the_gate_is_memoised(clamped_plate, cfg):
    """The same pose is asked about many times; the answer must be computed once."""
    robot = gate()
    pose = overhead()
    first = robot.check(pose.position, pose.axis, pose.roll_deg)
    second = robot.check(pose.position, pose.axis, pose.roll_deg)
    assert first is second
    assert robot.summary()["poses_asked"] == 1


def test_branches_are_assigned_as_a_sequence(clamped_plate, cfg):
    """Every delivered point gets one solution, and the sequence reports its worst step."""
    robot = gate()
    points = []
    for offset in (-60.0, -20.0, 20.0, 60.0):
        points.append(PathPoint(
            position=np.array([2000.0 + offset, 0.0, 1000.0]),
            axis=np.array([0.0, 0.0, -1.0]), roll_deg=0.0,
            role="coverage", source="greedy", stroke_id=0))
    assignment = assign_branches(points, robot, float(cfg.robot.max_step_deg))
    assert assignment.failed == []
    assert all(item is not None for item in assignment.solutions)
    assert assignment.histogram
    assert 0.0 <= assignment.max_step_deg < 180.0


def test_branch_assignment_without_a_gate_is_empty():
    """No gate, no joints - and no invented ones."""
    assignment = assign_branches([], None, 40.0)
    assert assignment.solutions == []
    assert assignment.ok


def band(x0: float, x1: float, y: float) -> Stroke:
    """One sweep in the arm's workspace, straight down, at the working standoff."""
    return Stroke(start=np.array([x0, y, 1000.0]), end=np.array([x1, y, 1000.0]),
                  axis=np.array([0.0, 0.0, -1.0]))


def two_bands() -> list[Stroke]:
    """Two same-pose bands 80 mm apart - the cheapest possible transition."""
    return [band(1940.0, 1990.0, 0.0), band(1940.0, 1990.0, 80.0)]


def test_the_connector_is_one_joint_move_and_inserts_no_corner(clamped_plate, cfg):
    """The primary primitive: sweep end -> next sweep start in a single J, no retract point.

    This is the whole point of the connector rewrite. The retract corners are what carried all
    seven of the unsolvable points in the 99% run, and a joint move needs none of them: it
    interpolates all six axes linearly, so two legal ends cannot produce an illegal middle, and
    there is nothing to insert between them.
    """
    model = ScannerModel(cfg.scanner)
    path = build_path(two_bands(), cfg=cfg, station=clamped_plate.station, model=model,
                      kin=gate(), start_near=np.zeros(3))
    assert path.connector_levels == {LEVEL_JOINT: 1}
    # two sweeps of two ends each, and NOTHING in between
    assert [point.role for point in path.points] == [ROLE_COVERAGE] * 4
    assert path.dropped == []
    assert path.violations == []


def test_a_joint_connector_delivers_its_own_joint_solution_per_point(clamped_plate, cfg):
    """Every delivered point carries the joints it was verified with, sweep ends included.

    The far end of a sweep is not solved independently: it is whatever the verified walk
    arrived at. Solving it fresh would let the record describe a branch the arm does not
    actually take.
    """
    model = ScannerModel(cfg.scanner)
    path = build_path(two_bands(), cfg=cfg, station=clamped_plate.station, model=model,
                      kin=gate(), start_near=np.zeros(3))
    assert path.joint_planned
    assert path.branches is not None
    assert path.branches.failed == []
    assert all(item is not None for item in path.branches.solutions)
    # a sweep is one straight Cartesian line; the move to the next band is a joint move
    assert [point.motion for point in path.points] == ["J", "L", "J", "L"]
    # a joint-representation point is the only form in which the planned branch arrives
    assert [point.representation for point in path.points] == [
        "joint", "cart", "joint", "cart"]
    assert path.max_step_deg is not None and path.max_step_deg < 180.0


def test_without_a_gate_the_path_is_still_the_old_one(clamped_plate, cfg):
    """`robot.enabled=false` must keep every per-point decision unset, so the file is unchanged.

    None rather than a copied default: the exporter reads None as "use the program default",
    which is what makes turning the gate off a no-op instead of a different program.
    """
    model = ScannerModel(cfg.scanner)
    path = build_path(two_bands(), cfg=cfg, station=clamped_plate.station, model=model,
                      start_near=np.zeros(3))
    assert not path.joint_planned
    assert path.branches is None
    assert all(point.motion is None and point.representation is None
               for point in path.points)
    assert LEVEL_JOINT not in path.connector_levels


def test_a_joint_move_never_takes_j5_through_zero(clamped_plate, cfg):
    """A J5 sign change IS the wrist singularity, so it is refused rather than priced.

    The controller will perform the move - the arm whips round - but a plan that does it
    without a reason is a plan asking for a surprise. The dynamic programme prices a flip at
    a large penalty instead, so it only happens when the alternative is losing a sweep.
    """
    model = ScannerModel(cfg.scanner)
    ok, _margin, reason = _joint_move(
        (0.0, 0.0, 0.0, 0.0, 20.0, 0.0), (0.0, 0.0, 0.0, 0.0, -20.0, 0.0),
        cfg, clamped_plate.station, model, kin=gate())
    assert not ok and reason == "j5_crossover"


def test_nearest_readings_undoes_an_avoidable_full_turn():
    """A joint move interpolates the NUMBER it is given, so +340 where -20 was meant is a spin.

    `Robot._place` picks the roomiest reading for a pose on its own, which is right for one
    point and wrong for a sequence: two neighbouring poses can come back almost a full turn
    apart. Measured on the real 59-point plan, 18 of 58 segments had a literal J6 change over
    180 degrees, the worst 352.6 degrees for a 17 degree change of the pose.
    """
    robot = gate().robot
    previous = np.array([0.0, -20.0, -10.0, 10.0, -60.0, 20.0])
    values = np.array([0.0, -20.0, -10.0, 10.0, -60.0, 340.0])
    moved = _nearest_readings(values, previous, robot)
    assert abs(moved[5] - (-20.0)) < 1e-9
    assert np.array_equal(moved[:5], values[:5])


def test_a_whole_chain_is_unwrapped_not_just_the_next_point():
    """A reading a turn away at the START of the chain is unreachable for a forward pass.

    The single-pass version had no predecessor to compare the first point against, so a first
    point written -340 where +19.91 was meant stayed wrong and every later point inherited the
    offset. This is the shape measured on the delivered 64-point program: J6 -340.09 -> -8.07
    commanded 332 degrees for a 28 degree move. Looking at the chain as a whole fixes it,
    because -8.07 next door says which side of the turn the first point belongs on.
    """
    robot = gate().robot
    chain = [_BranchOnly(joints_deg=(0.0, 0.0, 0.0, 0.0, -60.0, j6),
                         shoulder="N", elbow="U", config="N U T, 0, 0, 0",
                         j5_margin_deg=30.0, min_margin_deg=30.0)
             for j6 in (-340.09, -8.07, 0.0)]
    _unwrap_readings(chain, robot)
    assert chain[0].joints_deg[5] == pytest.approx(19.91, abs=1e-6)
    assert chain[1].joints_deg[5] == pytest.approx(-8.07, abs=1e-6)


def test_no_segment_of_the_unwrapped_chain_jumps_more_than_the_bottleneck():
    """The objective is the worst segment, not the total, so one bad step cannot hide in it."""
    robot = gate().robot
    readings = [10.0, -170.0, 20.0, 190.0]
    chain = [_BranchOnly(joints_deg=(0.0, 0.0, 0.0, 0.0, -60.0, j6),
                         shoulder="N", elbow="U", config="N U T, 0, 0, 0",
                         j5_margin_deg=30.0, min_margin_deg=30.0)
             for j6 in readings]
    _unwrap_readings(chain, robot)
    joined = np.array([item.joints_deg for item in chain], dtype=float)
    steps = np.abs(np.diff(joined, axis=0))
    assert steps.max() <= 190.0
    for item in chain:
        assert robot.feasible(item.joints_deg)


def _stub_solution(joints, config="N U T, 0, 0, 0"):
    return SimpleNamespace(joints_deg=tuple(float(v) for v in joints), config=config,
                           min_margin_deg=30.0, j5_margin_deg=30.0)


class _StubGate:
    """A gate that hands back prepared solutions, so the DP can be tested on its own."""

    active = True

    def __init__(self, table):
        self.table = table

    def solutions(self, position, axis, roll_deg):
        return self.table[tuple(np.round(np.asarray(position, dtype=float), 6))]


def test_the_j6_weight_changes_the_choice_but_never_the_reported_degrees():
    """The weight is a preference between options; the program still moves what it says.

    Two branches can both be feasible while one turns J6 far more than the other. Without a
    weight the sequencing pass takes whichever has the smaller WORST axis change, which may be
    the one that spins the wrist; with a weight it takes the smaller J6 when that is the axis
    that costs most. The step the report prints must stay the real degrees either way.
    """
    near = np.array([0.0, 0.0, 0.0])
    far = np.array([100.0, 0.0, 0.0])
    table = {
        tuple(np.round(near, 6)): [_stub_solution((0.0, 0.0, 0.0, 0.0, -60.0, 0.0))],
        tuple(np.round(far, 6)): [
            _stub_solution((40.0, 0.0, 0.0, 0.0, -60.0, 0.0)),     # big J1, no J6
            _stub_solution((5.0, 0.0, 0.0, 0.0, -60.0, 20.0)),     # small J1, 20 deg of J6
        ],
    }
    points = [PathPoint(position=near, axis=np.array([0.0, 0.0, -1.0]), roll_deg=0.0,
                        role=ROLE_COVERAGE, source="greedy"),
              PathPoint(position=far, axis=np.array([0.0, 0.0, -1.0]), roll_deg=0.0,
                        role=ROLE_COVERAGE, source="greedy")]
    gate = _StubGate(table)

    plain = assign_branches(points, gate, 400.0)
    assert plain.solutions[1].joints_deg[5] == 20.0
    assert plain.max_step_deg == 20.0

    weighted = assign_branches(points, gate, 400.0, j6_weight=3.0)
    assert weighted.solutions[1].joints_deg[5] == 0.0
    # the weight is 3 and the worst change is 40 degrees of J1: reported as 40, not 120
    assert weighted.max_step_deg == 40.0
    assert weighted.failed == [] and weighted.broken == []


def test_every_delivered_point_is_on_the_branch_the_program_can_ask_for(clamped_plate, cfg):
    """The file's CONFIG hint is the constant `N U T`, so every point must be on that branch.

    The pendant calls the negative-J5 branch `T`, and an L move cannot change the wrist
    branch, so a point planned on the positive branch is a point the controller refuses with
    `MOTN-063`. Writing the positive branch as a per-point CONFIG instead does not help: the
    letter `F` is refused at load with `ASBN-023`. Both were measured on 2026-10-29, and this
    test is what keeps the planner from producing either.
    """
    model = ScannerModel(cfg.scanner)
    path = build_path(two_bands(), cfg=cfg, station=clamped_plate.station, model=model,
                      kin=gate(), start_near=np.zeros(3))
    assert path.joint_planned
    for solution in path.branches.solutions:
        assert float(solution.joints_deg[4]) < 0.0
        assert solution.config.split(",")[0].strip().endswith("T")
    assert path.branches.histogram == {"N U T": 4}
    assert path.violations == []


def test_over_limit_steps_are_counted_even_though_they_are_not_refused(clamped_plate, cfg):
    """`max_step_deg` is a review trigger, and a trigger that never fires is not a trigger.

    The count and the worst value have to agree with each other: a report saying "worst step
    171.3, limit 40, over-limit steps 0" is the kind of self-contradiction that makes the whole
    block unreadable.
    """
    tiny = replace(cfg, robot=replace(cfg.robot, max_step_deg=1.0))
    model = ScannerModel(cfg.scanner)
    path = build_path(two_bands(), cfg=tiny, station=clamped_plate.station, model=model,
                      kin=gate(), start_near=np.zeros(3))
    assert len(path.points) == 4                      # nothing was refused for being far
    assert len(path.branches.broken) == len(path.points) - 1
    assert any("exceed max_step_deg" in line for line in path.violations)


def test_a_sweep_the_arm_cannot_walk_is_dropped_with_its_reason(clamped_plate, cfg):
    """A start pose that solves but a line the branch cannot carry is not delivered.

    The old order could not say this: it wrote the point, and the controller found out. Here
    the item leaves the plan and the reason names the failing check.
    """
    model = ScannerModel(cfg.scanner)
    # 600 mm of standoff travel: the start is inside the envelope, the far end is not.
    far = Stroke(start=np.array([2000.0, 0.0, 1000.0]),
                 end=np.array([2000.0, 0.0, 3400.0]),
                 axis=np.array([0.0, 0.0, -1.0]))
    path = build_path([far], cfg=cfg, station=clamped_plate.station, model=model,
                      kin=gate(), start_near=np.zeros(3))
    assert path.dropped == [0]
    assert path.points == []
    assert path.violations and "dropped" in path.violations[0]
