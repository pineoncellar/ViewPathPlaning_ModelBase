"""The arm, its workspace-table joint limits, and the 90% window the planner may use.

To assert: the shipped configuration derates every documented stop of the R-2000iC
workspace table to 90% (all six axes, both ends); the CAD chain and the table describe
the same physical travel, which is what makes the datum conversion exact; a pose exactly
on the derated bound is accepted and one that is inside the documented limit but past 90%
is refused, which is what makes the window a planning margin rather than a copy of the
datasheet; a straight move that leaves the window between two legal ends is caught by
sampling it; the chain reproduces the datasheet reach it cross-checks; forward and inverse
kinematics round trip; the report states both windows with the degrees each end gives up;
and the arm is absent - plus a bad window refused - when the check is switched off or
configured impossibly.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from scanplan.config import load_config, override
from scanplan.coverage import evaluate
from scanplan.report.robot import robot_block
from scanplan.robot import MODELS, RobotArm, arm_for
from scanplan.scanner_model import ScannerModel, ScannerPose
from scanplan.target_surface import SampleSet

# The workspace table of the R-2000iC/210F - the numbers a workshop reads off the
# controller, pasted as published so a typo in the config cannot hide behind the model.
DOCUMENTED_DEG = ((-185.0, 185.0), (-90.0, 46.0), (-122.0, 190.0),
                  (-360.0, 360.0), (-125.0, 125.0), (-360.0, 360.0))
MARGIN = 0.9


@pytest.fixture(scope="module")
def arm() -> RobotArm:
    return RobotArm(load_config().robot)


def test_the_window_is_ninety_percent_of_the_documented_travel(arm):
    documented = np.array(DOCUMENTED_DEG)
    assert np.allclose(arm.table_limits_deg, documented)
    assert float(arm.params.joint_limit_margin) == MARGIN
    # Both ends of every axis pull in together: the outermost 10% of the travel is what
    # the margin gives up, not 10% of the distance from the datum to one stop.
    assert np.allclose(arm.allowed_table_deg, documented * MARGIN)
    assert np.allclose(arm.allowed_table_deg,
                       [[-166.5, 166.5], [-81.0, 41.4], [-109.8, 171.0],
                        [-324.0, 324.0], [-112.5, 112.5], [-324.0, 324.0]])


def test_the_chain_and_the_table_describe_the_same_travel():
    """The two conventions differ by a zero, so every axis has to be the same width."""
    for spec in MODELS.values():
        for axis, ((cad_low, cad_high), (table_low, table_high)) in enumerate(
                zip(spec.cad_limits_deg, spec.table_limits_deg, strict=True), start=1):
            assert cad_high - cad_low == pytest.approx(table_high - table_low), f"J{axis}"


def test_the_chain_reproduces_the_datasheet_reach():
    arm = RobotArm(load_config().robot)
    assert arm.spec.reach_mm == pytest.approx(2655.0)
    # Shoulder offset + upper arm + the rigid elbow vector: 32 mm past the datasheet,
    # which measures to the flange and this chain to the wrist centre.
    assert arm.spec.arm_reach_mm == pytest.approx(2686.6, abs=0.1)
    assert 0.0 < arm.spec.arm_reach_mm - arm.spec.reach_mm < 40.0


def test_a_pose_on_the_derated_bound_is_legal_and_one_past_it_is_not(arm):
    """J1 at 166.5 deg is 90% of 185; at 170 deg it is still inside the datasheet."""
    inside = np.array([166.5, 20.0, 60.0, 30.0, 30.0, 40.0])
    frame, position = arm.forward(inside)
    reach = arm.solve(frame, position)
    assert reach is not None
    assert reach.utilization == pytest.approx(1.0, abs=1e-6)
    assert reach.table_deg[0] == pytest.approx(166.5, abs=1e-6)

    for j1 in (170.0, 175.0, -170.0):
        beyond = np.array([j1, 20.0, 60.0, 30.0, 30.0, 40.0])
        assert arm.utilization(beyond).max() > 1.0, "the case must be past the window"
        assert arm.solve(*arm.forward(beyond)) is None


def test_a_straight_move_that_leaves_the_window_between_its_ends_is_caught(arm, cfg):
    """A chord between two legal poses can ask for an illegal configuration in between."""
    posture = [20.0, 60.0, 30.0, 30.0, 40.0]
    start = np.array([150.0] + posture)
    end = np.array([-150.0] + posture)
    start_frame, start_position = arm.forward(start)
    end_frame, end_position = arm.forward(end)
    step = float(cfg.stroke.check_step_mm)

    chord = arm.along(start_frame, end_frame, start_position, end_position,
                      step_mm=step, reference=start)
    assert not chord.ok
    assert chord.failed_at_mm is not None and chord.failed_at_mm < 1599.0
    assert chord.samples > 1

    # The same two ends, each solved on its own, are legal - so it is the interior that
    # fails, which is exactly why the move is sampled instead of only its ends checked.
    assert arm.solve(start_frame, start_position) is not None
    assert arm.solve(end_frame, end_position) is not None

    still = arm.along(start_frame, start_frame, start_position, start_position,
                      step_mm=step, reference=start)
    assert still.ok


def test_forward_and_inverse_agree_on_a_generic_configuration(arm):
    """The round trip has to close, or a "reachable" verdict means nothing."""
    for posture in ([10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                    [-60.0, -20.0, 90.0, -120.0, 40.0, 200.0],
                    [100.0, 10.0, 150.0, 10.0, 60.0, -90.0]):
        wanted_frame, wanted_position = arm.forward(posture)
        reach = arm.solve(wanted_frame, wanted_position)
        assert reach is not None
        frame, position = arm.forward(reach.joints_deg)
        assert position == pytest.approx(wanted_position, abs=1e-6)
        assert frame == pytest.approx(wanted_frame, abs=1e-6)
        assert reach.utilization <= 1.0 + 1e-9


def test_the_report_states_both_windows_and_what_each_end_gives_up(arm, cfg):
    block = robot_block(arm, cfg)
    assert block["checked"] is True
    assert block["model"] == "R-2000iC/210F"
    assert block["joint_limit_margin"] == MARGIN
    assert "90%" in block["margin_note"]
    window = {entry["joint"]: entry for entry in block["window"]}
    assert window["J1"]["documented_deg"] == [-185.0, 185.0]
    assert window["J1"]["allowed_deg"] == [-166.5, 166.5]
    assert window["J1"]["given_up_deg"] == [18.5, 18.5]
    # every axis gives up 10% of its own travel, at both ends
    for axis, (low, high) in enumerate(DOCUMENTED_DEG, start=1):
        entry = window[f"J{axis}"]
        assert entry["given_up_deg"] == pytest.approx([abs(low) * 0.1, abs(high) * 0.1],
                                                      abs=1e-3)
        assert min(entry["given_up_deg"]) >= 0.0, "a degrees-given-up figure cannot be signed"


def test_the_arm_is_absent_when_the_check_is_switched_off(cfg):
    off = replace(cfg, robot=replace(cfg.robot, enabled=False))
    assert arm_for(off.robot) is None
    block = robot_block(None, off)
    assert block["checked"] is False
    assert "ROBOGUIDE" in block["reason"]


def test_a_margin_outside_the_unit_interval_is_refused(cfg):
    for margin in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="joint_limit_margin"):
            override(cfg, robot={"joint_limit_margin": margin})


def test_a_model_without_a_kinematic_chain_is_refused(cfg):
    unknown = replace(cfg.robot, model="M-10iD/12")
    with pytest.raises(ValueError, match="unknown robot model"):
        RobotArm(unknown)


def test_a_pose_the_arm_cannot_reach_is_dropped_before_a_single_ray_is_cast(
        clamped_plate, cfg):
    """The joint window gates the candidate pool the way clearance does.

    A pose with no legal configuration keeps an empty coverage set, so it cannot be
    selected and the coverage figure never counts a pose the cell cannot fly - which is
    the difference between "the planner found a viewpoint" and "the arm can fly it".
    """
    arm = arm_for(cfg.robot)
    assert arm is not None
    model = ScannerModel(cfg.scanner)
    grid = np.arange(-20.0, 20.5, 10.0)
    points = np.array([[x, y, 0.7] for x in grid for y in grid])
    samples = SampleSet(points=points, normals=np.tile([0.0, 0.0, 1.0], (len(points), 1)),
                        area_mm2=np.ones(len(points)), total_area_mm2=float(len(points)))
    # Six metres from a robot with a 2655 mm reach: no configuration gets there, so the
    # verdict cannot depend on which branch the solver happens to try.
    far_away = ScannerPose(position=np.array([0.0, 0.0, 6000.0]),
                           axis=np.array([0.0, 0.0, -1.0]))
    assert arm.solve(far_away.frame, far_away.position) is None

    gated = evaluate([far_away], samples, clamped_plate.station, model, cfg, robot=arm)
    assert gated.counters["rejected_joint_limit"] == 1
    assert gated.counters["rays"] == 0
    assert len(gated.sets[0]) == 0 and len(gated.quality[0]) == 0

    ungated = evaluate([far_away], samples, clamped_plate.station, model, cfg)
    assert ungated.counters["rejected_joint_limit"] == 0
    assert len(ungated.sets[0]) == 0


