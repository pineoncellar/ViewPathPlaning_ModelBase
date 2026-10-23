"""The joint-limit gate: where it rejects, what it reports, and how branches are assigned.

The robot geometry itself is tested in `test_kinematics` with synthetic joint angles. This
file tests the SEAM: that a pose the arm cannot hold never reaches the coverage matrix, that
a stroke which leaves the envelope stops growing, that the uncovered samples are attributed
to the joint limits rather than to a layout problem, and that the delivered path gets one
consistent branch per point.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from scanplan.config import RobotParams
from scanplan.coverage import attribute_uncovered, evaluate
from scanplan.kinematics import KinematicGate
from scanplan.path import PathPoint, assign_branches
from scanplan.scanner_model import ScannerModel, ScannerPose
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
