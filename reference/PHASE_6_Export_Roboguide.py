"""PHASE 6 - export planned scan viewpoints as Cartesian FANUC positions.

This is an *output adapter* only. It does not touch PHASE 1..5 and it does not
solve any inverse kinematics: it strips the (UR5e specific) joint angles and
emits the viewpoints as Cartesian scanner-TCP poses, leaving reachability,
joint limits and collision checks to the FANUC controller / ROBOGUIDE.

Input : a PHASE 2 (or PHASE 3/4/5) output file, i.e. a JSON that carries a
        viewpoint set such as "VPs_greedy".
Output: <out>/PH6_export <ts>.LS     FANUC ASCII program (P[] + motion)
        <out>/PH6_frame  <ts>.json   frame contract + self-check numbers
        <out>/PH6_points <ts>.csv    flat table for eyeballing / Excel

Frame contract (see PH6_frame*.json for the emitted copy):
  origin = the object voxel-bounding-box minimum corner, plus --origin-mm
  axes   = the algorithm world axes (right handed, Z up)
  unit   = mm
  so:  p_mm = (voxel_index - pad_voxels) * gridsize_mm + origin_mm
  Pass the CAD coordinates of that corner (the part's bounding-box minimum) via
  --origin-mm to get true CAD coordinates out of the exporter.

Alternatively pass --place "x,y,z,w,p,r", the pose of that CAD frame inside the
ROBOGUIDE world. The whole program is then expressed in world coordinates and
runs with UFRAME 0, so no UFRAME has to be taught by hand.

Pose contract:
  the exported pose is the algorithm VP pose, i.e. a point s_size_length
  (150 mm for the modelled sensor) behind the sensor front face, on the optical
  axis. Local +Z is the optical axis, pointing from the sensor towards the part.
  The ROBOGUIDE UTOOL must be defined to match that point.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
from datetime import datetime

import numpy as np

OUT_DIR_DEFAULT = "00_Output Collection"
DEFAULT_VPS_KEY = "VPs_greedy"
PROG_NAME_DEFAULT = "PH6SCAN"
DEFAULT_CONFIG = "N U T, 0, 0, 0"

# FANUC registers an ASCII program under a name that must be a plain program
# name, and the ASCII file has to carry that same name. A name with spaces (or a
# timestamped file name) is rejected with "Invalid name in /PROG section".
PROG_NAME_PATTERN = re.compile(r"[A-Z][A-Z0-9_]{0,15}")


def normalise_prog_name(name: str) -> str:
    """Validate a FANUC program name, which is also the .LS file name."""
    candidate = name.strip().upper()
    if not PROG_NAME_PATTERN.fullmatch(candidate):
        raise ValueError(
            f"invalid FANUC program name {name!r}: the name is also the .LS file "
            "name, so it must start with a letter and contain only A-Z, 0-9 and "
            "underscore (at most 16 characters, no spaces)"
        )
    return candidate


# ----------------------------------------------------------------------------
# small numeric helpers
# ----------------------------------------------------------------------------
def fanuc_num(value: float, width: int = 10) -> str:
    """Format like the controller does: fixed 3 decimals, no leading zero.

    e.g. 1349.372 -> "  1349.372",  0.094 -> "      .094",  -0.0 -> "       .000"
    """
    if abs(value) < 5e-4:
        text = "0.000"
    else:
        text = f"{value:.3f}"
    if text.startswith("0."):
        text = text[1:]
    elif text.startswith("-0."):
        text = "-" + text[2:]
    return text.rjust(width)


def rot_to_fanuc_wpr(rotation: np.ndarray) -> tuple[float, float, float]:
    """3x3 rotation matrix -> FANUC (W, P, R) in degrees.

    FANUC uses fixed-axis XYZ Euler angles, i.e. R = Rz(R) * Ry(P) * Rx(W).
    """
    rot = np.asarray(rotation, dtype=float).reshape(3, 3)
    sin_p = float(np.clip(-rot[2, 0], -1.0, 1.0))
    pitch = math.asin(sin_p)
    if abs(math.cos(pitch)) > 1e-9:
        w = math.atan2(rot[2, 1], rot[2, 2])
        r = math.atan2(rot[1, 0], rot[0, 0])
    else:
        # gimbal lock: only W-R (pitch = +90) resp. W+R (pitch = -90) is defined
        w = 0.0
        if sin_p > 0:
            r = -math.atan2(rot[0, 1], rot[0, 2])
        else:
            r = math.atan2(-rot[0, 1], -rot[0, 2])
    return tuple(math.degrees(angle) for angle in (w, pitch, r))


def wpr_to_rot(w: float, p: float, r: float) -> np.ndarray:
    """Inverse of rot_to_fanuc_wpr, used for the round-trip self check."""
    w, p, r = (math.radians(v) for v in (w, p, r))
    cw, sw = math.cos(w), math.sin(w)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)
    rx = np.array([[1, 0, 0], [0, cw, -sw], [0, sw, cw]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return rz @ ry @ rx


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return math.degrees(math.acos(dot))


# ----------------------------------------------------------------------------
# input handling
# ----------------------------------------------------------------------------
def find_latest_output(out_dir: str, phase: str = "PH2") -> str:
    pattern = os.path.join(out_dir, f"{phase} output *.json")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"no '{phase} output *.json' found in '{out_dir}' - run PHASE {phase[2]} first"
        )
    return candidates[-1]


def read_voxel_geometry(data: dict) -> dict:
    """Derive the voxel->mm mapping from the file itself (no magic numbers)."""
    gridsize = float(data["statistics"]["gridsize (mm)"])
    extension = float(data["statistics"]["sampleSpace_extension factor"])

    expanded = data["obj"]["voxel_ocupancy_matrix_extended"]
    expanded_shape = np.array(
        [len(expanded), len(expanded[0]), len(expanded[0][0])], dtype=int
    )
    bbox_shape = np.array(data["obj"]["voxel bounding box"], dtype=int)

    pad_from_shapes = (expanded_shape - bbox_shape)
    if np.any(pad_from_shapes % 2 != 0):
        raise ValueError("object grid is not centred inside the padded grid")
    pad_from_shapes = pad_from_shapes // 2

    sensor = data["sensor"]
    pad_formula = int(
        round(
            (float(sensor["s_maxRange (mm)"]) + float(sensor["s_size_length (mm)"]))
            * extension
            / gridsize
        )
    )
    if not np.all(pad_from_shapes == pad_formula):
        raise ValueError(
            f"padding mismatch: from grid shapes {pad_from_shapes.tolist()} "
            f"vs from sensor parameters {pad_formula}"
        )

    return {
        "gridsize_mm": gridsize,
        "sample_space_extension": extension,
        "pad_voxels": pad_from_shapes,
        "pad_mm": pad_from_shapes * gridsize,
        "expanded_grid_shape": expanded_shape.tolist(),
        "obj_voxel_bbox_shape": bbox_shape.tolist(),
        "sensor": sensor,
    }


def collect_points(
    data: dict, vps_key: str, geom: dict, origin_mm: np.ndarray
) -> list[dict]:
    if vps_key not in data:
        available = [k for k in data if k.startswith("VPs_")]
        raise KeyError(f"'{vps_key}' not in file; available viewpoint sets: {available}")

    gridsize = geom["gridsize_mm"]
    pad = geom["pad_voxels"]

    points = []
    for key in sorted(data[vps_key], key=lambda k: int(k.split("_")[-1])):
        vp = data[vps_key][key]
        position = (np.asarray(vp["coord"], dtype=float) - pad) * gridsize + origin_mm
        rotation = np.asarray(vp["RotationMatrix"], dtype=float)
        w, p, r = rot_to_fanuc_wpr(rotation)
        points.append(
            {
                "vp_key": key,
                "xyz_mm": position,
                "rotation": rotation,
                "wpr_deg": (w, p, r),
            }
        )

    if not points:
        raise ValueError(f"'{vps_key}' is empty")
    return points


def apply_placement(
    points: list[dict], rotation: np.ndarray, translation: np.ndarray
) -> list[dict]:
    """Express the (already CAD frame) points in another frame."""
    placed = []
    for point in points:
        rotation_new = rotation @ point["rotation"]
        placed.append(
            {
                "vp_key": point["vp_key"],
                "xyz_mm": rotation @ point["xyz_mm"] + translation,
                "rotation": rotation_new,
                "wpr_deg": rot_to_fanuc_wpr(rotation_new),
            }
        )
    return placed


def parse_pose(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse 'x,y,z,w,p,r' (W/P/R in degrees, FANUC convention) into a 4x4 pose."""
    try:
        values = [float(v) for v in text.split(",")]
    except ValueError as error:
        raise ValueError("--place expects six comma separated numbers") from error
    if len(values) != 6:
        raise ValueError("--place expects x,y,z,w,p,r")
    x, y, z, w, p, r = values
    return wpr_to_rot(w, p, r), np.array([x, y, z], dtype=float)


def object_centroid_mm(data: dict, geom: dict, origin_mm: np.ndarray) -> np.ndarray:
    coords = np.asarray(data["obj"]["voxel coordinates"], dtype=float)
    return (coords.mean(axis=0) - geom["pad_voxels"]) * geom["gridsize_mm"] + origin_mm


# ----------------------------------------------------------------------------
# ordering (PHASE 5's joint-space order is void: joints were stripped)
# ----------------------------------------------------------------------------
def order_points(
    points: list[dict],
    start_near: np.ndarray,
    rot_weight: float = 0.0,
    two_opt_limit: int = 2000,
    two_opt_passes: int = 40,
) -> list[dict]:
    """Greedy nearest neighbour, optionally improved by 2-opt.

    Cost = Euclidean travel of the TCP in mm, plus rot_weight * the angle (deg)
    between the two optical axes. rot_weight defaults to 0 (position only).
    """
    n = len(points)
    pos = np.stack([pt["xyz_mm"] for pt in points])
    axes = np.stack([pt["rotation"][:, 2] for pt in points])

    def axis_penalty(i: int, j: int) -> float:
        if rot_weight == 0.0:
            return 0.0
        return rot_weight * angle_between(axes[i], axes[j])

    start = int(np.argmin(np.linalg.norm(pos - start_near, axis=1)))
    remaining = np.ones(n, dtype=bool)
    remaining[start] = False
    order = [start]

    for _ in range(n - 1):
        last = order[-1]
        costs = np.linalg.norm(pos - pos[last], axis=1)
        if rot_weight:
            dots = np.clip(axes @ axes[last], -1.0, 1.0)
            costs = costs + rot_weight * np.degrees(np.arccos(dots))
        costs[~remaining] = np.inf
        nxt = int(np.argmin(costs))
        remaining[nxt] = False
        order.append(nxt)

    if n <= two_opt_limit and n > 3:
        order = _two_opt(order, pos, axes, rot_weight, two_opt_passes)

    return [points[i] for i in order]


def _two_opt(
    order: list[int],
    pos: np.ndarray,
    axes: np.ndarray,
    rot_weight: float,
    max_passes: int,
) -> list[int]:
    dots = np.clip(axes @ axes.T, -1.0, 1.0)
    cost_matrix = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    if rot_weight:
        cost_matrix = cost_matrix + rot_weight * np.degrees(np.arccos(dots))

    best = list(order)
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                a, b = best[i - 1], best[i]
                c, d = best[j], best[j + 1]
                delta = (
                    cost_matrix[a, c] + cost_matrix[b, d]
                    - cost_matrix[a, b] - cost_matrix[c, d]
                )
                if delta < -1e-9:
                    best[i : j + 1] = reversed(best[i : j + 1])
                    improved = True
    return best


# ----------------------------------------------------------------------------
# FANUC .LS writer
# ----------------------------------------------------------------------------
def _pos_block(index: int, kind: str, uframe: int, utool: int, config: str, payload) -> str:
    if kind == "joint":
        joints = payload
        return (
            f"P[{index}]{{\n"
            f"   GP1:\n"
            f"\tUF : {uframe}, UT : {utool}, \n"
            f"\tJ1= {fanuc_num(joints[0])} deg,\tJ2= {fanuc_num(joints[1])} deg,"
            f"\tJ3= {fanuc_num(joints[2])} deg,\n"
            f"\tJ4= {fanuc_num(joints[3])} deg,\tJ5= {fanuc_num(joints[4])} deg,"
            f"\tJ6= {fanuc_num(joints[5])} deg\n"
            f"}};"
        )
    x, y, z, w, p, r = payload
    return (
        f"P[{index}]{{\n"
        f"   GP1:\n"
        f"\tUF : {uframe}, UT : {utool},\t\tCONFIG : '{config}',\n"
        f"\tX = {fanuc_num(x)}  mm,\tY = {fanuc_num(y)}  mm,\tZ = {fanuc_num(z)}  mm,\n"
        f"\tW = {fanuc_num(w)} deg,\tP = {fanuc_num(p)} deg,\tR = {fanuc_num(r)} deg\n"
        f"}};"
    )


def build_ls(
    points: list[dict],
    uframe: int,
    utool: int,
    motion: str,
    speed: str,
    j_speed: str,
    config: str,
    constant: str,
    prog_name: str,
    comment: str,
    home_joints=None,
    timestamp: datetime | None = None,
) -> str:
    stamp = (timestamp or datetime.now()).strftime("%y-%m-%d  TIME %H:%M:%S")

    def command(index: int, is_joint: bool = False) -> str:
        # a joint representation point can only be reached with a joint move,
        # and joint moves take a percentage while linear moves take a speed
        if motion.upper() == "L" and not is_joint:
            return f"L P[{index}] {speed} FINE"
        return f"J P[{index}] {j_speed} FINE"

    mn_lines: list[str] = []
    pos_blocks: list[str] = []
    next_index = 1

    home_index = None
    if home_joints:
        home_index = next_index
        next_index += 1
        pos_blocks.append(
            _pos_block(home_index, "joint", uframe, utool, config, home_joints)
        )

    point_indices = []
    for pt in points:
        point_indices.append(next_index)
        pos_blocks.append(
            _pos_block(
                next_index,
                "cart",
                uframe,
                utool,
                config,
                (*pt["xyz_mm"].tolist(), *pt["wpr_deg"]),
            )
        )
        next_index += 1

    mn_lines.append(f"{len(mn_lines) + 1:>4}:  UFRAME_NUM={uframe} ;")
    mn_lines.append(f"{len(mn_lines) + 1:>4}:  UTOOL_NUM={utool} ;")
    if constant:
        mn_lines.append(f"{len(mn_lines) + 1:>4}:  {constant} ;")

    moves: list[tuple[int, bool]] = []
    if home_index:
        moves.append((home_index, True))
    moves.extend((index, False) for index in point_indices)
    if home_index:
        moves.append((home_index, True))

    for index, is_joint in moves:
        mn_lines.append(f"{len(mn_lines) + 1:>4}:{command(index, is_joint)}    ;")
    # real FANUC programs simply end after the last instruction; there is no
    # END statement in the instruction list

    body = "\n".join(mn_lines)
    prog_size = len(body) + sum(len(b) for b in pos_blocks)

    return (
        f"/PROG  {prog_name}\n"
        f"/ATTR\n"
        f"OWNER\t\t= MNEDITOR;\n"
        f'COMMENT\t\t= "{comment}";\n'
        f"PROG_SIZE\t= {prog_size};\n"
        f"CREATE\t\t= DATE {stamp};\n"
        f"MODIFIED\t= DATE {stamp};\n"
        f"FILE_NAME\t= ;\n"
        f"VERSION\t\t= 0;\n"
        f"LINE_COUNT\t= {len(mn_lines)};\n"
        f"MEMORY_SIZE\t= {prog_size + 316};\n"
        f"PROTECT\t\t= READ_WRITE;\n"
        f"TCD:  STACK_SIZE\t= 0,\n"
        f"      TASK_PRIORITY\t= 50,\n"
        f"      TIME_SLICE\t= 0,\n"
        f"      BUSY_LAMP_OFF\t= 0,\n"
        f"      ABORT_REQUEST\t= 0,\n"
        f"      PAUSE_REQUEST\t= 0;\n"
        f"DEFAULT_GROUP\t= 1,*,*,*,*;\n"
        f"CONTROL_CODE\t= 00000000 00000000;\n"
        f"/MN\n"
        f"{body}\n"
        f"/POS\n"
        f"{chr(10).join(pos_blocks)}\n"
        f"/END\n"
    )


# ----------------------------------------------------------------------------
# self test
# ----------------------------------------------------------------------------
def selftest() -> None:
    rng = np.random.default_rng(12345)
    worst = 0.0
    for _ in range(2000):
        w, p, r = (float(rng.uniform(-180, 180)) for _ in range(3))
        rot = wpr_to_rot(w, p, r)
        back = wpr_to_rot(*rot_to_fanuc_wpr(rot))
        worst = max(worst, float(np.abs(rot - back).max()))
    assert worst < 1e-9, f"WPR round trip broken (max err {worst})"

    # gimbal lock cases (pitch = +-90 deg)
    for p in (90.0, -90.0):
        rot = wpr_to_rot(0.0, p, 37.5)
        back = wpr_to_rot(*rot_to_fanuc_wpr(rot))
        assert np.abs(rot - back).max() < 1e-9, f"gimbal lock failed for pitch {p}"

    # axes sanity: rotating about X by W=90 must send +Y to +Z
    rot = wpr_to_rot(90.0, 0.0, 0.0)
    assert np.allclose(rot @ np.array([0.0, 1.0, 0.0]), [0, 0, 1], atol=1e-12)

    assert fanuc_num(1349.372) == "  1349.372", fanuc_num(1349.372)
    assert fanuc_num(0.094) == "      .094", fanuc_num(0.094)
    assert fanuc_num(-86.371) == "   -86.371", fanuc_num(-86.371)
    assert fanuc_num(-0.0) == (" " * 6) + ".000", repr(fanuc_num(-0.0))
    assert fanuc_num(180.0) == "   180.000", fanuc_num(180.0)

    # program names double as file names, so they must be plain program names
    assert normalise_prog_name("PH6SCAN") == "PH6SCAN"
    assert normalise_prog_name(" ph6scan ") == "PH6SCAN"
    for bad in ("PH6 world", "PH6WORLD 261012-142431", "1PH6", "PH6-WORLD", ""):
        try:
            normalise_prog_name(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad!r} should have been rejected")
    print(f"selftest OK (worst WPR round trip error {worst:.3e})")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input",
        help="PHASE 2/3/4/5 output JSON (default: newest PH2 output in "
        f"'{OUT_DIR_DEFAULT}')",
    )
    parser.add_argument(
        "--out-dir", default=OUT_DIR_DEFAULT, help="folder for the exported files"
    )
    parser.add_argument("--vps-key", default=DEFAULT_VPS_KEY)
    parser.add_argument(
        "--origin-mm",
        default="0,0,0",
        help="CAD coordinates of the part bounding-box minimum corner; shifts the "
        "export so the numbers are true CAD coordinates",
    )
    parser.add_argument(
        "--place",
        default="",
        help="pose 'x,y,z,w,p,r' of the CAD frame inside ROBOGUIDE, so the exported "
        "program can be run with UFRAME 0 instead of building a UFRAME by hand",
    )
    parser.add_argument(
        "--uframe",
        type=int,
        default=None,
        help="UFRAME number (default 1, or 0 when --place is used)",
    )
    parser.add_argument("--utool", type=int, default=1)
    parser.add_argument(
        "--prog-name",
        default=PROG_NAME_DEFAULT,
        help="FANUC program name; also becomes the .LS file name, so it must be a "
        f"valid program name (letters, digits, underscore). Default {PROG_NAME_DEFAULT}",
    )
    parser.add_argument("--motion", default="J", choices=["J", "j", "L", "l"])
    parser.add_argument("--speed", default="100%", help="e.g. 100%% or 250mm/sec")
    parser.add_argument(
        "--j-speed", default="100%", help="speed for joint moves (FANUC takes a percentage)"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--constant", default="", help="optional constant/instruction name")
    parser.add_argument(
        "--home-joints",
        default="",
        help="optional HOME pose as J1..J6 degrees, e.g. '0,0,-90,0,90,0'",
    )
    parser.add_argument("--rot-weight", type=float, default=0.0, help="mm per degree")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)

    if args.selftest:
        selftest()
        return 0

    try:
        origin_mm = np.array([float(v) for v in args.origin_mm.split(",")], dtype=float)
    except ValueError:
        parser.error("--origin-mm expects three comma separated numbers")
    if origin_mm.shape != (3,):
        parser.error("--origin-mm expects exactly three comma separated numbers")

    placement = None
    if args.place:
        try:
            placement = parse_pose(args.place)
        except ValueError as error:
            parser.error(str(error))

    uframe = args.uframe if args.uframe is not None else (0 if placement else 1)

    try:
        prog_name = normalise_prog_name(args.prog_name)
    except ValueError as error:
        parser.error(str(error))

    source = args.input or find_latest_output(OUT_DIR_DEFAULT)
    print(f"reading {source}")
    with open(source, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    geom = read_voxel_geometry(data)
    vps_key = args.vps_key
    if vps_key == DEFAULT_VPS_KEY and vps_key not in data:
        available = [k for k in data if k.startswith("VPs_")]
        if len(available) == 1:
            vps_key = available[0]
            print(f"'{DEFAULT_VPS_KEY}' absent, falling back to '{vps_key}'")

    points = collect_points(data, vps_key, geom, origin_mm)
    centroid = object_centroid_mm(data, geom, origin_mm)
    if placement:
        rotation_place, translation_place = placement
        points = apply_placement(points, rotation_place, translation_place)
        centroid = rotation_place @ centroid + translation_place
    points = order_points(points, centroid, rot_weight=args.rot_weight)

    home_joints = (
        [float(v) for v in args.home_joints.split(",")] if args.home_joints else None
    )
    if home_joints and len(home_joints) != 6:
        parser.error("--home-joints needs exactly 6 comma separated values")

    timestamp = datetime.now()
    stamp = timestamp.strftime("%y%m%d-%H%M%S")
    config_label = "PH6 scan points"

    ls_text = build_ls(
        points=points,
        uframe=uframe,
        utool=args.utool,
        motion=args.motion,
        speed=args.speed,
        j_speed=args.j_speed,
        config=args.config,
        constant=args.constant,
        prog_name=prog_name,
        comment=config_label,
        home_joints=home_joints,
        timestamp=timestamp,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    # the program name must equal the file name, or the controller rejects the load
    ls_path = os.path.join(args.out_dir, f"{prog_name}.LS")
    # name the companion files after the program as well, so that two runs cannot
    # silently overwrite each other's frame contract or point table
    frame_path = os.path.join(args.out_dir, f"{prog_name}_frame.json")
    csv_path = os.path.join(args.out_dir, f"{prog_name}_points.csv")

    with open(ls_path, "w", encoding="ascii", newline="\r\n") as handle:
        handle.write(ls_text)

    xyz = np.stack([pt["xyz_mm"] for pt in points])
    wpr = np.stack([pt["wpr_deg"] for pt in points])
    roundtrip = max(
        float(np.abs(pt["rotation"] - wpr_to_rot(*pt["wpr_deg"])).max()) for pt in points
    )

    frame = {
        "generated": timestamp.isoformat(timespec="seconds"),
        "source_json": os.path.abspath(source),
        "vps_key": vps_key,
        "point_count": len(points),
        "frame": {
            "origin": "object voxel-bounding-box minimum corner, shifted by origin_mm_offset",
            "origin_mm_offset": origin_mm.tolist(),
            "axes": "identical to the algorithm world axes (right handed, Z up)",
            "unit": "mm",
            "formula": "p_mm = (voxel_index - pad_voxels) * gridsize_mm + origin_mm_offset",
            "note": (
                "the part bounding-box minimum corner is exactly (0,0,0) before the "
                "shift; pass its CAD coordinates via --origin-mm to get CAD coordinates"
            ),
        },
        "voxel_geometry": {
            "gridsize_mm": geom["gridsize_mm"],
            "sample_space_extension": geom["sample_space_extension"],
            "pad_voxels": geom["pad_voxels"].tolist(),
            "pad_mm": geom["pad_mm"].tolist(),
            "expanded_grid_shape": geom["expanded_grid_shape"],
            "obj_voxel_bbox_shape": geom["obj_voxel_bbox_shape"],
            "object_bbox_size_mm": [
                float(v) * geom["gridsize_mm"] for v in geom["obj_voxel_bbox_shape"]
            ],
        },
        "algorithm_floor_plane": {
            "note": (
                "z = 0 of the algorithm world is the robot base mounting plane; "
                "the part bbox min corner sits pad_z * gridsize above it"
            ),
            "z_of_origin_above_algorithm_floor_mm": float(geom["pad_mm"][2]),
        },
        "sensor": geom["sensor"],
        "tcp_contract": {
            "definition": (
                "exported pose = algorithm VP pose = a point s_size_length mm behind "
                "the sensor front face on the optical axis; local +Z points from the "
                "sensor towards the part"
            ),
            "s_size_length_mm": geom["sensor"]["s_size_length (mm)"],
            "range_to_surface_mm": [
                geom["sensor"]["s_minRange (mm)"],
                geom["sensor"]["s_maxRange (mm)"],
            ],
            "roboguide_action": "define the UTOOL so that its TCP matches this point",
        },
        "wpr_convention": {
            "definition": "FANUC fixed-axis XYZ Euler: R = Rz(R) * Ry(P) * Rx(W)",
            "max_roundtrip_error": roundtrip,
        },
        "ordering": {
            "method": "nearest neighbour" + (" + 2-opt" if len(points) <= 2000 else ""),
            "start": "point nearest the object centroid",
            "cost": f"Euclidean TCP travel (mm) + {args.rot_weight} * optical axis angle (deg)",
            "note": "PHASE 5's joint-space order is void, the joint angles were stripped",
        },
        "program": {
            "file": os.path.basename(ls_path),
            "uframe": uframe,
            "utool": args.utool,
            "motion": args.motion.upper(),
            "speed": args.speed,
            "j_speed": args.j_speed,
            "config_string": args.config,
            "home_joints": home_joints,
        },
        "placement": {
            "applied": args.place or None,
            "meaning": (
                "pose x,y,z,w,p,r of the CAD frame inside the ROBOGUIDE world; "
                "the exported program then runs with UFRAME 0 and needs no UFRAME "
                "built by hand"
            ),
            "warning": (
                "only correct if this really is the pose of the shared CAD origin; "
                "verify by checking that the points land on the part in ROBOGUIDE"
            ),
        },
        "extents_mm": {
            "min": xyz.min(axis=0).round(3).tolist(),
            "max": xyz.max(axis=0).round(3).tolist(),
        },
    }
    with open(frame_path, "w", encoding="utf-8") as handle:
        json.dump(frame, handle, indent=2)

    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["order", "vp_key", "x_mm", "y_mm", "z_mm", "w_deg", "p_deg", "r_deg"]
        )
        for i, pt in enumerate(points):
            writer.writerow(
                [i + 1, pt["vp_key"], *(round(v, 4) for v in pt["xyz_mm"]), *(round(v, 4) for v in pt["wpr_deg"])]
            )

    print(
        f"{len(points)} viewpoints from '{vps_key}'\n"
        f"  pad          : {geom['pad_voxels'].tolist()} voxels "
        f"({geom['pad_mm'].tolist()} mm)\n"
        f"  extents      : {frame['extents_mm']['min']} .. {frame['extents_mm']['max']} mm\n"
        f"  WPR roundtrip: {roundtrip:.2e}\n"
        f"  program      : {ls_path}\n"
        f"  frame        : {frame_path}\n"
        f"  points       : {csv_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
