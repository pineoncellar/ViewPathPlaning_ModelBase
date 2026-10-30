"""End-to-end check of the `.LS` export on a hand-made viewpoint set.

To assert: the three files appear; W/P/R reproduces the frame it came from
including the gimbal-lock case; local +Z of every exported pose is the optical
axis; positions come out in the CAD frame unchanged, and moved/rotated when a
placement is given; bad program names and an empty set are refused; and the
contract file records the two things the export cannot model.

Every viewpoint below is hand-made and deliberately plain: the real station is
customer CAD and can never be committed.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from scanplan.config import RobotParams
from scanplan.geometry import scanner_frame
from scanplan.roboguide import adapt, ls, wpr

PROG = "SCAN_A"


def viewpoints() -> list[dict]:
    """Four synthetic viewpoints: straight down, and two tilted ones, various rolls."""
    return [
        {"id": 1, "position": (0.0, 0.0, 300.0), "axis": (0.0, 0.0, -1.0), "roll_deg": 0.0},
        {"id": 2, "position": (120.0, 0.0, 300.0), "axis": (0.0, 0.0, -1.0), "roll_deg": 90.0},
        {"id": 3, "position": (0.0, 120.0, 300.0), "axis": (0.0, 0.3, -0.95), "roll_deg": 37.0},
        {
            "id": 4,
            "position": (-120.0, -120.0, 250.0),
            "axis": (0.6, 0.0, -0.8),
            "roll_deg": 210.0,
        },
    ]


def test_an_unreachable_point_is_refused_instead_of_written(tmp_path):
    """The joint wall. A program the arm cannot execute must not be produced at all.

    This is the failure the site hit: a Cartesian program was written, ROBOGUIDE answered
    MOTN-018 / TPIF-012, and nothing in the pipeline had objected because the joint gate
    was off. The export now refuses on its own, before a single byte is written.
    """
    robot = replace(RobotParams.disabled(), enabled=True)
    with pytest.raises(ValueError, match="joint envelope"):
        adapt.export(viewpoints(), tmp_path / "refused", prog_name=PROG,
                     place="0,0,0,0,0,0", robot=robot)
    assert not (tmp_path / "refused" / f"{PROG}.LS").exists()


def test_a_point_with_infeasible_joints_is_refused(tmp_path):
    """Stored joint values are checked even when the numbers live in a taught UFRAME."""
    robot = replace(RobotParams.disabled(), enabled=True)
    point = {"id": 1, "position": (2000.0, 0.0, 1000.0), "axis": (0.0, 0.0, -1.0),
             "roll_deg": 0.0,
             # J2 = 44 deg is inside the datasheet's +46 but only 2 deg from it, and the
             # gate reserves `joint_margin_deg` (5) at each end
             "joints_deg": (0.0, 44.0, 0.0, 0.0, 64.0, 0.0)}
    with pytest.raises(ValueError, match="joint envelope"):
        adapt.export([point], tmp_path, prog_name=PROG, robot=robot)


def test_the_joint_audit_records_the_envelope_it_used(tmp_path):
    """Evidence, not an opinion: the contract says which envelope was enforced."""
    robot = replace(RobotParams.disabled(), enabled=True)
    reachable = {"id": 1, "position": (2000.0, 0.0, 1000.0),
                 "axis": (0.0, 0.0, -1.0), "roll_deg": 0.0}
    contract = adapt.export([reachable], tmp_path, prog_name=PROG,
                            place="0,0,0,0,0,0", robot=robot)
    audit = contract["joint_audit"]
    assert audit["enabled"] is True
    assert audit["illegal"] == []
    assert audit["checked"] == 1 and audit["from_pose"] == 1
    assert audit["envelope_deg"][4] == pytest.approx([-125.0, 125.0])
    assert "joint limits WERE checked" in contract["not_modelled"][0]


def test_without_a_robot_block_the_export_says_nothing_was_verified(tmp_path):
    contract = adapt.export(viewpoints(), tmp_path, prog_name=PROG)
    assert contract["joint_audit"] == {
        "enabled": False, "checked": 0, "from_joints": 0, "from_pose": 0,
        "illegal": [], "unverified": []}


def test_a_negative_turn_digit_is_refused_before_a_byte_is_written(tmp_path):
    """`N U F, -1, -1, -1` is what stopped the 2026-10-27 import (ASBN-009).

    The field goes into every `P[n]` block, so a malformed one costs the whole file, and
    ROBOGUIDE reports it as a load error with a column number and no hint about where the
    string came from. Refusing it here names the string instead.
    """
    with pytest.raises(ValueError, match="CONFIG"):
        ls.check_config("N U F, -1, 0, 0")
    with pytest.raises(ValueError, match="CONFIG"):
        ls.check_config("N U T")
    assert ls.check_config("n u t,0,0,0") == "N U T, 0, 0, 0"
    points = viewpoints()
    points[0]["config"] = "N U F, -1, 0, 0"
    with pytest.raises(ValueError, match="CONFIG"):
        adapt.export(points, tmp_path / "bad", prog_name=PROG)
    assert not (tmp_path / "bad" / f"{PROG}.LS").exists()


def test_the_three_files_appear(tmp_path):
    contract = adapt.export(viewpoints(), tmp_path, prog_name=" scan_a ")
    assert (tmp_path / f"{PROG}.LS").exists()
    assert (tmp_path / f"{PROG}_points.csv").exists()
    frame_path = tmp_path / f"{PROG}_frame.json"
    assert frame_path.exists()
    assert json.loads(frame_path.read_text(encoding="utf-8"))["point_count"] == 4
    assert contract["files"]["program"] == f"{PROG}.LS"


def test_the_program_name_matches_the_file_name(tmp_path):
    adapt.export(viewpoints(), tmp_path, prog_name=PROG)
    text = (tmp_path / f"{PROG}.LS").read_text(encoding="ascii")
    assert text.splitlines()[0].split() == ["/PROG", PROG]


def test_the_uframe_default_follows_the_placement(tmp_path):
    plain = adapt.export(viewpoints(), tmp_path / "cad", prog_name=PROG)
    assert plain["uframe_utool"]["uframe"] == 1
    assert plain["placement"]["applied"] is None

    placed = adapt.export(
        viewpoints(), tmp_path / "world", prog_name=PROG, place="1000,2000,3000,0,0,90"
    )
    assert placed["uframe_utool"]["uframe"] == 0
    assert placed["placement"]["applied"] == "1000,2000,3000,0,0,90"
    text = (tmp_path / "world" / f"{PROG}.LS").read_text(encoding="ascii")
    assert "UFRAME_NUM=0 ;" in text


def test_a_speed_without_a_known_unit_is_refused():
    """A typo in the speed field would otherwise be written into the program."""
    with pytest.raises(ValueError, match="speed"):
        ls.check_speed("fast")
    with pytest.raises(ValueError, match="positive"):
        ls.check_speed("0%")
    with pytest.raises(ValueError, match="100 percent"):
        ls.check_speed("150%")
    with pytest.raises(ValueError, match="joint move"):
        ls.check_speed("250mm/sec", motion="J")
    assert ls.check_speed("250mm/sec", motion="L") == "250mm/sec"
    assert ls.check_speed("100%") == "100%"


def test_each_point_carries_its_own_config_string(tmp_path):
    """One CONFIG for the whole program is how every point got pinned to one branch."""
    points = viewpoints()
    for index, point in enumerate(points):
        point["config"] = f"F D T, {index}, 0, 0"
    adapt.export(points, tmp_path, prog_name=PROG, config="N U T, 0, 0, 0")
    text = (tmp_path / f"{PROG}.LS").read_text(encoding="ascii")
    for index in range(len(points)):
        assert f"CONFIG : 'F D T, {index}, 0, 0'" in text
    assert "CONFIG : 'N U T, 0, 0, 0'" not in text


def test_a_point_without_a_config_falls_back_to_the_program_default(tmp_path):
    points = viewpoints()
    points[0]["config"] = "F D F, 0, 0, 0"
    adapt.export(points, tmp_path, prog_name=PROG, config="N U T, 0, 0, 0")
    text = (tmp_path / f"{PROG}.LS").read_text(encoding="ascii")
    assert "CONFIG : 'F D F, 0, 0, 0'" in text
    assert text.count("CONFIG : 'N U T, 0, 0, 0'") == len(points) - 1


def test_joint_blocks_are_written_when_asked_for(tmp_path):
    """The most direct fix for the branch problem: the controller has nothing to solve."""
    points = viewpoints()
    for point in points:
        point["joints_deg"] = (10.0, 20.0, 30.0, 40.0, 50.0, 60.0)
    adapt.export(points, tmp_path, prog_name=PROG, joint_blocks=True, motion="L")
    text = (tmp_path / f"{PROG}.LS").read_text(encoding="ascii")
    assert text.count("J1=") == len(points)
    assert "X =" not in text
    # a joint-representation point cannot be reached with an L move
    assert text.count("J P[") == len(points)
    assert "L P[" not in text


def test_joint_blocks_are_ignored_without_joint_values(tmp_path):
    adapt.export(viewpoints(), tmp_path, prog_name=PROG, joint_blocks=True)
    text = (tmp_path / f"{PROG}.LS").read_text(encoding="ascii")
    assert "J1=" not in text
    assert text.count("X =") == len(viewpoints())


def test_a_joint_block_never_inherits_a_rate_speed(tmp_path):
    """`J P[n] 100mm/sec` is not a legal instruction: a joint move takes a percentage.

    The rate belongs to the L instruction a sweep is written with, so writing the same
    point as joint angles has to fall back to a percentage rather than pass the rate on.
    """
    points = viewpoints()
    for point in points:
        point["joints_deg"] = (10.0, 20.0, 30.0, 40.0, 50.0, 60.0)
    adapt.export(points, tmp_path / "joints", prog_name=PROG, joint_blocks=True,
                 motion="L", speed="100mm/sec")
    text = (tmp_path / "joints" / f"{PROG}.LS").read_text(encoding="ascii")
    assert "mm/sec" not in text
    assert text.count("J P[") == len(points)


def test_a_stroke_plan_scans_with_l_moves_and_repositions_with_j(tmp_path):
    """The head scans while it travels, so the move ALONG a sweep must be interpolated (L).

    A joint move's TCP path is not knowable, so writing one down a stroke would mean the
    coverage figure no longer describes what the scanner sees. The move into the second
    corner of a stroke IS the sweep; the move into the first corner and anything reaching a
    retract point scan nothing, and there a joint move is the honest way to turn the wrist.
    """
    points = viewpoints()
    points[0].update(stroke_id=1, role="coverage")
    points[1].update(stroke_id=1, role="coverage")
    points[2].update(stroke_id=2, role="coverage")
    points[3].update(stroke_id=None, role="retract")
    adapt.export(points, tmp_path, prog_name=PROG, order=False, motion="L",
                 speed="100mm/sec", termination="CNT0")
    text = (tmp_path / f"{PROG}.LS").read_text(encoding="ascii")
    moves = re.findall(r"^\s*\d+:([LJ]) P\[(\d+)\]", text, re.MULTILINE)
    assert moves == [("J", "1"), ("L", "2"), ("J", "3"), ("J", "4")]
    assert "L P[2] 100mm/sec CNT0" in text
    assert "J P[1] 50% FINE" in text


def test_a_pose_only_plan_keeps_the_configured_motion(tmp_path):
    """No stroke_id means no sweep, so there is nothing to interpolate: J stays J."""
    adapt.export(viewpoints(), tmp_path, prog_name=PROG)
    text = (tmp_path / f"{PROG}.LS").read_text(encoding="ascii")
    assert "L P[" not in text
    assert text.count("J P[") == len(viewpoints())


def test_wpr_is_unwrapped_along_the_delivered_order():
    """Two consecutive poses must not be written 358 degrees apart."""
    points = [
        {"id": 1, "position": (0.0, 0.0, 300.0), "axis": (0.0, 0.0, -1.0), "roll_deg": 179.0},
        {"id": 2, "position": (1.0, 0.0, 300.0), "axis": (0.0, 0.0, -1.0), "roll_deg": 181.0},
    ]
    converted = adapt.viewpoints_to_points(points)
    first, second = (np.asarray(item["wpr_deg"], dtype=float) for item in converted)
    assert abs(second[2] - first[2]) < 10.0
    # the rotation is unchanged by the whole turns that were added
    for item, source in zip(converted, points, strict=True):
        assert np.allclose(wpr.wpr_to_rot(*item["wpr_deg"]),
                           scanner_frame(source["axis"], source["roll_deg"]), atol=1e-9)


def test_the_exported_frame_is_the_single_scanner_frame_definition():
    points = adapt.viewpoints_to_points(viewpoints())
    for point, vp in zip(points, viewpoints(), strict=True):
        expected = scanner_frame(vp["axis"], vp["roll_deg"])
        assert np.allclose(point["rotation"], expected, atol=1e-12)


def test_local_plus_z_of_every_exported_pose_is_the_optical_axis():
    points = adapt.viewpoints_to_points(viewpoints())
    assert len(points) == len(viewpoints())
    for point, vp in zip(points, viewpoints(), strict=True):
        axis = np.asarray(vp["axis"], dtype=float)
        axis = axis / np.linalg.norm(axis)
        rotation = point["rotation"]
        assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        assert np.linalg.det(rotation) > 0
        assert np.allclose(rotation[:, 2], axis, atol=1e-12)
        # the angles written into the program carry the same local +Z
        assert np.allclose(wpr.wpr_to_rot(*point["wpr_deg"])[:, 2], axis, atol=1e-9)


def test_the_angles_round_trip_for_random_orientations():
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(500):
        rotation = wpr.wpr_to_rot(*rng.uniform(-180.0, 180.0, 3))
        back = wpr.wpr_to_rot(*wpr.rot_to_fanuc_wpr(rotation))
        worst = max(worst, float(np.abs(rotation - back).max()))
    assert worst < 1e-9


@pytest.mark.parametrize("pitch", [90.0, -90.0])
def test_a_pose_at_the_gimbal_lock_still_round_trips(pitch):
    rotation = wpr.wpr_to_rot(0.0, pitch, 37.5)
    w, p, r = wpr.rot_to_fanuc_wpr(rotation)
    assert p == pytest.approx(pitch)
    assert w == 0.0  # only the combination of W and R is determined
    assert np.abs(rotation - wpr.wpr_to_rot(w, p, r)).max() < 1e-9


def test_fanuc_numbers_have_no_leading_zero():
    assert wpr.fanuc_num(0.094) == "      .094"
    assert wpr.fanuc_num(-86.371) == "   -86.371"
    assert wpr.fanuc_num(-0.0) == "      .000"
    assert wpr.fanuc_num(1349.372) == "  1349.372"


def test_positions_are_unchanged_in_the_cad_frame():
    points = adapt.viewpoints_to_points(viewpoints())
    for point, vp in zip(points, viewpoints(), strict=True):
        assert np.allclose(point["xyz_mm"], vp["position"], atol=1e-12)


def test_a_placement_moves_and_rotates_the_points():
    plain = adapt.viewpoints_to_points(viewpoints())
    rotation = wpr.wpr_to_rot(0.0, 0.0, 90.0)
    translation = np.array([1000.0, 2000.0, 3000.0])
    placed = adapt.apply_placement(plain, rotation, translation)
    assert len(placed) == len(plain)
    for point, original in zip(placed, plain, strict=True):
        moved = rotation @ original["xyz_mm"] + translation
        assert np.allclose(point["xyz_mm"], moved, atol=1e-9)
        assert np.allclose(point["rotation"], rotation @ original["rotation"], atol=1e-12)
        assert np.allclose(wpr.wpr_to_rot(*point["wpr_deg"]), point["rotation"], atol=1e-9)
    assert not np.allclose(
        np.stack([pt["xyz_mm"] for pt in placed]), np.stack([pt["xyz_mm"] for pt in plain])
    )


def test_viewpoints_are_accepted_as_dicts_or_as_objects():
    as_objects = [
        SimpleNamespace(
            id=vp["id"], position=vp["position"], axis=vp["axis"], roll_deg=vp["roll_deg"]
        )
        for vp in viewpoints()
    ]
    from_dicts = adapt.viewpoints_to_points(viewpoints())
    from_objects = adapt.viewpoints_to_points(as_objects)
    assert [pt["vp_key"] for pt in from_objects] == [pt["vp_key"] for pt in from_dicts]
    for a, b in zip(from_dicts, from_objects, strict=True):
        assert np.allclose(a["rotation"], b["rotation"], atol=1e-12)


def test_ordering_starts_at_the_point_nearest_the_start():
    points = adapt.viewpoints_to_points(viewpoints())
    ordered = adapt.order_points(points, np.array([120.0, 0.0, 300.0]))
    assert ordered[0]["vp_key"] == 2
    assert sorted(pt["vp_key"] for pt in ordered) == [1, 2, 3, 4]


@pytest.mark.parametrize("bad", ["SCAN ONE", "1SCAN", "SCAN-1", "TOOLONGPROGRAMNAME", ""])
def test_a_bad_program_name_is_refused(bad, tmp_path):
    with pytest.raises(ValueError, match="program name"):
        wpr.normalise_prog_name(bad)
    with pytest.raises(ValueError, match="program name"):
        adapt.export(viewpoints(), tmp_path, prog_name=bad)
    assert list(tmp_path.iterdir()) == []


def test_an_empty_viewpoint_set_is_refused(tmp_path):
    with pytest.raises(ValueError, match="viewpoint"):
        adapt.viewpoints_to_points([])
    with pytest.raises(ValueError, match="viewpoint"):
        adapt.export([], tmp_path, prog_name=PROG)
    assert list(tmp_path.iterdir()) == []


def test_the_contract_records_what_the_export_cannot_model(tmp_path):
    contract = adapt.export(viewpoints(), tmp_path, prog_name=PROG)
    text = json.dumps(contract).lower()
    assert "robot reachability" in text
    assert "not modelled" in text
    assert "occlusion of the tracker" in text
    assert "robot arm" in text
    assert contract["not_modelled"]
    assert all(isinstance(line, str) for line in contract["not_modelled"])


def test_the_contract_carries_the_frame_the_tcp_contract_and_the_measured_error(tmp_path):
    contract = adapt.export(
        viewpoints(), tmp_path, prog_name=PROG, utool=7, motion="L", speed="250mm/sec"
    )
    assert contract["frame"]["unit"] == "mm"
    assert contract["frame"]["origin"] and contract["frame"]["axes"]
    assert "Rz(R) * Ry(P) * Rx(W)" in contract["wpr_convention"]["definition"]
    assert contract["wpr_convention"]["max_roundtrip_error"] < 1e-9
    assert contract["uframe_utool"]["uframe"] == 1
    assert contract["uframe_utool"]["utool"] == 7
    assert "+Z" in contract["tcp_contract"]["local_z"]
    assert "UTOOL" in contract["tcp_contract"]["roboguide_action"]
    assert contract["program"]["motion"] == "L"
    assert contract["program"]["config_string"] == "N U T, 0, 0, 0"
    assert contract["point_count"] == 4
    assert len(contract["extents_mm"]["min"]) == 3


def test_the_attribute_header_is_self_consistent():
    points = adapt.viewpoints_to_points(viewpoints())
    text = ls.build_ls(points, prog_name=PROG, uframe=1, utool=2)
    lines = text.splitlines()
    assert lines[0].split() == ["/PROG", PROG]
    assert "UFRAME_NUM=1 ;" in text and "UTOOL_NUM=2 ;" in text
    mn, pos = lines.index("/MN"), lines.index("/POS")
    body = "\n".join(lines[mn + 1 : pos])
    blocks = text.split("\n/POS\n", 1)[1].rsplit("\n/END", 1)[0]
    line_count = int(re.search(r"LINE_COUNT\s*=\s*(\d+)", text).group(1))
    prog_size = int(re.search(r"PROG_SIZE\s*=\s*(\d+)", text).group(1))
    memory_size = int(re.search(r"MEMORY_SIZE\s*=\s*(\d+)", text).group(1))
    assert line_count == len(lines[mn + 1 : pos]) == 2 + len(points)
    # the header has to describe the file it is written into: PROG_SIZE covers the
    # instructions plus the position blocks (joined by a newline, one per gap)
    assert blocks.count("};") == len(points)
    assert prog_size == len(body) + len(blocks) - (len(points) - 1)
    assert memory_size == prog_size + 316
    # J + FINE is the default, and every point is referenced by index
    assert text.count("J P[") == len(points)
    for index in range(1, len(points) + 1):
        assert f"P[{index}]{{" in text


def test_the_points_csv_is_the_flat_table_the_field_asked_for(tmp_path):
    adapt.export(viewpoints(), tmp_path, prog_name=PROG)
    lines = (tmp_path / f"{PROG}_points.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0].split(",") == ["order", "vp_key", "x", "y", "z", "w", "p", "r"]
    assert len(lines) == 1 + len(viewpoints())
    rows = [line.split(",") for line in lines[1:]]
    assert [row[0] for row in rows] == ["1", "2", "3", "4"]
    assert sorted(row[1] for row in rows) == ["1", "2", "3", "4"]
    assert all(len(row) == 8 for row in rows)
    assert "-0.0" not in "\n".join(lines)  # a table for eyeballing has no negative zero
