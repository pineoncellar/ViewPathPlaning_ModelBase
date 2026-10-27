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

_POSITION_BLOCK = re.compile(r"P\[(\d+)\]\{(.*?)\};", re.S)


def exported_positions(text: str) -> dict[int, tuple[float, ...]]:
    """The `/POS` section as `{P index: (X, Y, Z)}`, so a test can compare numbers."""
    positions: dict[int, tuple[float, ...]] = {}
    for block in _POSITION_BLOCK.finditer(text):
        numbers = dict(re.findall(r"([XYZ])\s*=\s*(-?[\d.]+)\s*mm", block.group(2)))
        positions[int(block.group(1))] = (float(numbers["X"]), float(numbers["Y"]),
                                          float(numbers["Z"]))
    return positions


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


def test_a_position_is_never_skipped_or_repeated(tmp_path):
    """`P[1] .. P[n]`, one per delivered point, in the delivered order.

    This is the numbering the 3D view has to show. A gap here would make every later
    point one number lower in ROBOGUIDE than in the picture, and a repeat would make two
    program points indistinguishable.
    """
    adapt.export(viewpoints(), tmp_path, prog_name=PROG)
    text = (tmp_path / f"{PROG}.LS").read_text(encoding="ascii")
    positions = exported_positions(text)
    assert sorted(positions) == list(range(1, len(viewpoints()) + 1))
    for index, vp in enumerate(viewpoints(), start=1):
        assert positions[index] == pytest.approx(vp["position"], abs=0.001)
    moves = re.findall(r"^\s*\d+:\s*[JL] P\[(\d+)\]", text, re.M)
    assert [int(move) for move in moves] == list(range(1, len(viewpoints()) + 1))


def test_the_3d_view_numbers_points_the_way_the_program_does(tmp_path):
    """The picture's `P[n]` and the controller's `P[n]` must be the same point.

    The page used to print the 0-based index, which made every point look one lower than
    the same point in ROBOGUIDE - the difference an operator notices when they check a
    pose in the controller.
    """
    from scanview.payload import build_payload

    adapt.export(viewpoints(), tmp_path, prog_name=PROG)
    text = (tmp_path / f"{PROG}.LS").read_text(encoding="ascii")
    positions = exported_positions(text)
    payload = build_payload(part=_unit_cube(), viewpoints=viewpoints())
    records = payload["viewpoints"]
    assert [record["p_index"] for record in records] == list(range(1, len(records) + 1))
    for record in records:
        assert positions[record["p_index"]] == pytest.approx(record["position"], abs=0.001)


def _unit_cube():
    import trimesh
    return trimesh.creation.box(extents=(10.0, 10.0, 10.0))


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


def test_a_joint_review_ignores_the_travel_settings():
    """A J program is a reachability review: every point is a joint move, at FINE.

    The travel settings describe the joint travel of a straight program. There are no
    sweeps to travel between in a review, so they must not reach the program at all.
    """
    points = adapt.viewpoints_to_points(viewpoints())
    assert len(points) == 4
    for point, role in zip(points, ["coverage", "coverage", "joint", "joint"],
                           strict=True):
        point["role"] = role
    text = ls.build_ls(points, prog_name=PROG, uframe=1, utool=2, motion="J", speed="50%")
    assert text.count("J P[") == len(points)
    assert "50% FINE" in text
    assert "CNT100" not in text


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
