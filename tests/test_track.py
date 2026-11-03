"""The seventh axis: a linear rail under the arm, along the base frame's +X.

What is being pinned here, in order:

* the rail is an AXIS, so a pose it makes reachable is reachable - and the same pose is not
  reachable on the floor-mounted arm;
* the pose a solution describes is the pose `forward()` reproduces once the carriage is
  added back in, which is the only way the six joints and the rail reading stay one answer;
* a straight move carries the carriage with it, which is what an extended axis does;
* the programme writes it (`E1`), the record carries it (`track_mm`), and the delivery audit
  refuses a record whose joints plus `E1` do not land on the point's own XYZ.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from scanplan.config import load_config, override
from scanplan.kinematics import Robot
from scanplan.path import PathPoint
from scanplan.report.kinematics import kinematics_block
from scanplan.roboguide.adapt import export, viewpoints_to_points
from scanplan.roboguide.ls import build_ls
from scanplan.roboguide.validate import check_points
from scanplan.viewpoints import Viewpoint

# An arm pose that is comfortably inside the limits on this cell.
JOINTS = (0.0, 10.0, -20.0, 0.0, -60.0, 0.0)


def _config(**robot_changes):
    return override(load_config(), robot=robot_changes)


def _robot(**robot_changes) -> Robot:
    params = replace(load_config().robot, enabled=True, **robot_changes)
    return Robot(params)


def _axis_roll(frame: np.ndarray) -> tuple[np.ndarray, float]:
    """The (axis, roll) that `scanner_frame` rebuilds `frame` from.

    A viewpoint record stores the pose the way the planner does - optical axis plus a roll
    about it - so a hand-made record has to be written back in that form, or the exporter
    rebuilds a different frame and the audit (rightly) refuses it.
    """
    from scanplan.geometry import unit

    axis = np.asarray(frame[:, 2], dtype=float)
    helper = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x0 = unit(np.cross(helper, axis))
    y0 = np.cross(axis, x0)
    first = np.asarray(frame[:, 0], dtype=float)
    roll = float(np.degrees(np.arctan2(float(first @ y0), float(first @ x0))))
    return axis, roll


def _record(robot: Robot, joints=JOINTS, rail: float | None = 5000.0,
            motion: str = "L") -> dict:
    """One viewpoint record, in the shape the exporter reads."""
    value = 0.0 if rail is None else float(rail)
    position, frame = robot.forward(joints, value)
    axis, roll = _axis_roll(frame)
    branch = robot.branch_of(joints, value)
    return {
        "position": position,
        "axis": axis,
        "roll_deg": roll,
        "joints_deg": joints,
        "config": robot.config_string(joints, branch[0], branch[1]),
        "track_mm": rail,
        "motion": motion,
        "representation": "cart" if motion == "L" else "joint",
    }


def _points(records):
    return viewpoints_to_points(records, per_point_config=True)


# ------------------------------------------------------------------ the axis itself


def test_the_shipped_configuration_carries_an_eight_metre_rail():
    """The cell's rail is part of the shipped numbers, not a local experiment."""
    config = load_config()
    assert config.robot.track_enabled is True
    assert config.robot.track_axis == "+X"
    assert config.robot.track_length_mm == pytest.approx(8000.0)
    assert _robot().track_usable == pytest.approx((100.0, 7900.0))


def test_the_rail_moves_the_base_and_nothing_else():
    """`forward(j, t)` is `forward(j, 0)` translated along +X by exactly `t`."""
    robot = _robot()
    here, frame_here = robot.forward(JOINTS, 0.0)
    there, frame_there = robot.forward(JOINTS, 2500.0)
    assert np.allclose(there - here, [2500.0, 0.0, 0.0])
    assert np.allclose(frame_here, frame_there)


def test_a_pose_only_the_rail_can_reach_is_reachable_with_it():
    """The point of the feature: the same pose, two envelopes, two answers."""
    far = np.array([9500.0, 0.0, 1400.0])
    rotation = _robot().forward(JOINTS, 7000.0)[1]

    assert any(item.ok for item in _robot().solutions(far, rotation))
    assert not any(item.ok for item in _robot(track_enabled=False).solutions(far, rotation))


def test_every_solution_carries_the_rail_reading_it_was_solved_at():
    """A joint set without its rail value is half an answer, so the pair travels together."""
    robot = _robot()
    position, rotation = robot.forward(JOINTS, 5000.0)
    found = [item for item in robot.solutions(position, rotation) if item.ok]
    assert found, "the pose must be reachable"
    for item in found:
        reached, frame = robot.forward(item.joints_deg, item.track_mm)
        assert np.allclose(reached, position, atol=0.5)
        assert np.allclose(frame, rotation, atol=1e-3)
    assert len({round(item.track_mm, 3) for item in found}) > 1, (
        "the rail makes the arm redundant: several carriage positions reach one pose")


def test_a_straight_move_carries_the_carriage_with_it():
    robot = _robot()
    target = robot.forward((5.0, 10.0, -20.0, 0.0, -60.0, 0.0), 5400.0)
    moving = robot.follow(JOINTS, *target, step_mm=20.0,
                          track_start=5000.0, track_end=5400.0)
    assert moving.ok, moving.reason
    assert moving.track_mm == pytest.approx(5400.0)


def test_the_rail_keeps_its_margin_like_any_other_axis():
    """A carriage on its end stop is not a carriage the machine holds."""
    robot = _robot()
    position, rotation = robot.forward(JOINTS, 0.0)
    assert not [item for item in robot.solutions(position, rotation, track_mm=10.0)
                if item.ok], "10 mm of travel is inside the forbidden margin"
    assert not robot.feasible(JOINTS, 10.0)


def test_a_rail_that_does_not_fit_the_cell_is_refused():
    with pytest.raises(ValueError, match="track_limits_mm"):
        _config(track_limits_mm=(0.0, 9000.0))
    with pytest.raises(ValueError, match="track_axis"):
        _config(track_axis="+Y")


# ------------------------------------------------------------------ the programme


def test_the_programme_writes_the_rail_as_the_extended_axis_column():
    robot = _robot()
    points = _points([_record(robot, rail=5000.0)])
    text = build_ls(points, prog_name="RAIL01", uframe=1, utool=1, motion="L")
    assert "E1=" in text and "5000.000" in text

    without = build_ls(points, prog_name="RAIL01", uframe=1, utool=1, motion="L",
                       emit_e1=False)
    assert "E1" not in without


def test_export_writes_the_rail_into_the_file(tmp_path):
    robot = _robot()
    export([_record(robot, rail=5000.0), _record(robot, rail=5400.0)],
           tmp_path, prog_name="RAIL02", motion="L", termination="CNT0",
           robot_params=robot.params)
    text = (tmp_path / "RAIL02.LS").read_text(encoding="utf-8")
    assert "E1=  5000.000  mm" in text
    assert "E1=  5400.000  mm" in text


def test_an_e1_that_does_not_belong_to_the_joints_is_refused():
    """The rail is redundant, so a wrong `E1` still solves - it just lands elsewhere."""
    robot = _robot()
    points = _points([_record(robot, rail=5000.0)])
    points[0]["track_mm"] = 3000.0  # the joints belong to 5000 mm of travel
    audit = check_points(points, default_motion="L", params=robot.params)
    assert not audit["ok"]
    assert [item["reason"] for item in audit["failures"]] == ["track_mismatch"]


def test_a_consistent_rail_record_passes_the_audit():
    robot = _robot()
    audit = check_points(_points([_record(robot, rail=5000.0)]), default_motion="L",
                         params=robot.params)
    assert audit["ok"], audit["failures"]


def test_a_joint_block_names_the_rail_by_its_axis_number():
    """`J7`, not `E1`: a joint block counts axes by number inside the group.

    A real ROBOGUIDE import of a rail plan stopped on exactly this point with `ASBN-009
    Mismatch number of joint` - six `J` values plus an `E1` line is not seven joints to the
    loader - while an exported program from the same cell carries `E1` in its CARTESIAN
    blocks. Both spellings exist; they are not interchangeable.
    """
    robot = _robot()
    points = _points([_record(robot, rail=5000.0, motion="J")])
    points[0]["motion"] = "J"
    points[0]["representation"] = "joint"
    text = build_ls(points, prog_name="RAIL04", uframe=1, utool=1, motion="J")
    assert "J7=" in text and "5000.000" in text
    assert "E1" not in text

    # the escape hatch, for a controller that wants the Cartesian spelling everywhere
    other = build_ls(points, prog_name="RAIL04", uframe=1, utool=1, motion="J",
                     joint_rail_label="E1")
    assert "E1=" in other and "J7" not in other


def test_the_cartesian_block_matches_the_shape_a_working_cell_exports():
    """Byte shape taken from a `.LS` this cell produced: comma after `R`, then `E1=  n  mm`."""
    robot = _robot()
    points = _points([_record(robot, rail=301.184)])
    text = build_ls(points, prog_name="RAIL05", uframe=1, utool=1, motion="L")
    lines = text.splitlines()
    index = next(number for number, line in enumerate(lines) if "R =" in line)
    assert lines[index].endswith("deg,")
    assert lines[index + 1] == "\tE1=   301.184  mm"


def test_a_rail_run_exports_cartesian_blocks_only(tmp_path):
    """Measured on the cell: a joint block carrying the seventh axis does not load.

    Both spellings (`J7=` and `E1=`) were refused at the same column with `ASBN-009 Mismatch
    number of joint`, while the same value in a Cartesian block is accepted - and that is the
    shape the cell's own exports have. So a rail run writes Cartesian blocks, and the cost is
    recorded rather than hidden: those poses are solved by the controller again.
    """
    robot = _robot()
    record = _record(robot, rail=5000.0, motion="J")
    export([record], tmp_path, prog_name="RAIL06", motion="J", termination="FINE",
           robot_params=robot.params)
    text = (tmp_path / "RAIL06.LS").read_text(encoding="utf-8")
    assert "J1=" not in text and "J7" not in text
    assert "E1=  5000.000  mm" in text


def test_a_floor_mounted_run_still_writes_the_joint_block(tmp_path):
    """Without a rail nothing changes: the pinned pose is still a joint block."""
    floor = _robot(track_enabled=False)
    export([_record(floor, rail=None, motion="J")], tmp_path, prog_name="FLAT01",
           motion="J", termination="FINE", robot_params=floor.params)
    text = (tmp_path / "FLAT01.LS").read_text(encoding="utf-8")
    assert "J1=" in text
    assert "E1" not in text


def test_export_can_leave_the_rail_out_and_says_so(tmp_path):
    """A six-axis controller refuses a point that declares seven axes.

    ROBOGUIDE stops on the first `E1` line with `ASBN-009 Mismatch number of joint`, which
    reads like a broken file rather than a cell with no rail. Exporting without it is a
    supported answer, as long as the contract says the plan and the program now disagree.
    """
    robot = _robot()
    export([_record(robot, rail=5000.0)], tmp_path, prog_name="RAIL03", motion="L",
           termination="CNT0", robot_params=robot.params, emit_e1=False)
    assert "E1" not in (tmp_path / "RAIL03.LS").read_text(encoding="utf-8")
    contract = json.loads((tmp_path / "RAIL03_frame.json").read_text(encoding="utf-8"))
    axis = contract["program"]["extended_axis"]
    assert axis["written"] is False
    assert axis["plan_carries_rail_mm"] is True
    assert axis["used_span_mm"] == [5000.0, 5000.0]
    assert "ASBN-009" in axis["requires"]
    assert "robot.track_enabled = false" in axis["warning"]


def test_a_floor_mounted_programme_carries_no_e1_at_all():
    """The regression profile: no rail, no column, byte for byte the old file."""
    floor = _robot(track_enabled=False)
    points = _points([_record(floor, rail=None)])
    text = build_ls(points, prog_name="NORAIL", uframe=1, utool=1, motion="L")
    assert "E1" not in text


# ------------------------------------------------------------------ the report


def test_the_report_measures_what_the_rail_did():
    robot = _robot()
    records = []
    for order, rail in enumerate((5000.0, 5400.0, 5300.0)):
        position, frame = robot.forward(JOINTS, rail)
        records.append(Viewpoint(
            id=order, position=position, axis=frame[:, 2], roll_deg=0.0,
            n_points=1, area_mm2=1.0, mean_incidence_deg=0.0, max_incidence_deg=0.0,
            tracker_ok=True, source="greedy", order=order,
            joints_deg=JOINTS, j5_margin_deg=30.0, track_mm=rail))

    track = kinematics_block(load_config(), records, path=None)["track"]
    assert track["enabled"] is True
    assert track["length_mm"] == pytest.approx(8000.0)
    assert track["travel_mm"] == pytest.approx(500.0)
    assert track["max_step_mm"] == pytest.approx(400.0)
    assert track["used_span_mm"] == pytest.approx([5000.0, 5400.0])


def test_a_run_without_a_rail_reports_no_track_block():
    assert kinematics_block(_config(track_enabled=False), [], path=None)["track"] is None


def test_a_path_point_defaults_to_no_rail():
    """Every record written before the rail existed compares unchanged."""
    point = PathPoint(position=np.zeros(3), axis=np.array([0.0, 0.0, -1.0]), roll_deg=0.0,
                      role="coverage", source="greedy")
    assert point.track_mm == 0.0
