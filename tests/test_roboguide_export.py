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
from types import SimpleNamespace

import numpy as np
import pytest

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


def test_each_point_carries_its_own_config_string_when_asked(tmp_path):
    """One CONFIG for the whole program is how every point got pinned to one branch."""
    points = viewpoints()
    for index, point in enumerate(points):
        point["config"] = f"F D T, {index}, 0, 0"
    adapt.export(points, tmp_path, prog_name=PROG, config="N U T, 0, 0, 0",
                 per_point_config=True)
    text = (tmp_path / f"{PROG}.LS").read_text(encoding="ascii")
    for index in range(len(points)):
        assert f"CONFIG : 'F D T, {index}, 0, 0'" in text
    assert "CONFIG : 'N U T, 0, 0, 0'" not in text


def test_per_point_letters_are_off_unless_asked_for(tmp_path):
    """They were refused on load, so the default must ignore them.

    A program whose first point said `N U F, 0, 0, 0` came back from ROBOGUIDE as
    `ASBN-023 Configuration error`, column exactly on the CONFIG field, while the same
    file with the constant `N U T, 0, 0, 0` on every point loads and runs. Only the T/F
    half of the letters was ever measured, so a computed letter is a guess that can make
    the whole program unimportable - hence opt-in.
    """
    points = viewpoints()
    for index, point in enumerate(points):
        point["config"] = f"F D T, {index}, 0, 0"
    adapt.export(points, tmp_path, prog_name=PROG, config="N U T, 0, 0, 0")
    text = (tmp_path / f"{PROG}.LS").read_text(encoding="ascii")
    assert text.count("CONFIG : 'N U T, 0, 0, 0'") == len(points)
    assert "F D T," not in text


def test_a_point_without_a_config_falls_back_to_the_program_default(tmp_path):
    points = viewpoints()
    points[0]["config"] = "F D F, 0, 0, 0"
    adapt.export(points, tmp_path, prog_name=PROG, config="N U T, 0, 0, 0",
                 per_point_config=True)
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
    # and the J instruction takes a percentage, not the linear speed the L move wanted:
    # `J P[1] 100 mm/sec FINE` is refused by the controller at import (ASBN-009)
    assert "J P[1] 100% FINE" in text
    assert "mm/sec" not in text


def test_a_joint_speed_must_be_a_percentage(tmp_path):
    """A rate in the joint branch is what the controller rejects on load, so refuse it here."""
    points = viewpoints()
    for point in points:
        point["joints_deg"] = (10.0, 20.0, 30.0, 40.0, 50.0, 60.0)
    with pytest.raises(ValueError, match="percentage"):
        adapt.export(points, tmp_path, prog_name=PROG, joint_blocks=True, motion="L",
                     speed="100 mm/sec", joint_speed="100 mm/sec")


def test_joint_blocks_are_ignored_without_joint_values(tmp_path):
    adapt.export(viewpoints(), tmp_path, prog_name=PROG, joint_blocks=True)
    text = (tmp_path / f"{PROG}.LS").read_text(encoding="ascii")
    assert "J1=" not in text
    assert text.count("X =") == len(viewpoints())


JOINTS = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)


def test_a_mixed_program_writes_scans_as_cartesian_and_transitions_as_joints(tmp_path):
    """The shape the connector actually produces, and the speed bug it used to hit.

    A joint-representation point removes the controller's choice of solution, so it is the
    only form in which a planned branch reaches the machine. A scan has to stay Cartesian,
    because the sweep the coverage figure was computed from is the straight line between two
    of them. A program of exactly this shape - joint point, then Cartesian points reached by
    L - was imported and run on the controller.

    The speed field is the other half: the site's linear speed is `100 mm/sec`, and a joint
    point validated against it raises, for a speed that point was never going to use.
    """
    points = viewpoints()
    points[0].update({"joints_deg": JOINTS, "motion": "J", "representation": "joint"})
    points[1].update({"motion": "L", "representation": "cart"})
    adapt.export(points, tmp_path, prog_name=PROG, order=False, motion="L",
                 speed="100 mm/sec", termination="CNT0")
    text = (tmp_path / f"{PROG}.LS").read_text(encoding="ascii")
    assert text.count("J1=") == 1
    assert text.count("X =") == len(points) - 1
    assert "J P[1] 100% FINE" in text
    assert "L P[2] 100 mm/sec CNT0" in text


def test_per_point_representation_beats_the_program_wide_flag(tmp_path):
    """`joint_blocks` is a default, not an override: a stroke plan needs both kinds at once."""
    points = viewpoints()
    points[0].update({"joints_deg": JOINTS, "motion": "J", "representation": "joint"})
    adapt.export(points, tmp_path, prog_name=PROG, order=False, motion="L",
                 speed="100 mm/sec", joint_blocks=False)
    text = (tmp_path / f"{PROG}.LS").read_text(encoding="ascii")
    assert text.count("J1=") == 1
    assert text.count("X =") == len(points) - 1


def test_the_point_table_carries_motion_and_representation():
    """The exporter is the only consumer, so both keys must survive the conversion."""
    points = viewpoints()
    points[0].update({"joints_deg": JOINTS, "motion": "J", "representation": "joint"})
    table = adapt.viewpoints_to_points(points)
    assert table[0]["motion"] == "J" and table[0]["representation"] == "joint"
    assert table[1]["motion"] is None and table[1]["representation"] is None


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
