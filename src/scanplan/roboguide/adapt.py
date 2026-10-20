"""Bridge the planner's viewpoints onto the `.LS` writer.

The conversion itself is a few lines; the value of this module is the FRAME
CONTRACT it writes out beside the program, because that contract is what the
workshop has to reproduce in ROBOGUIDE and it is the one thing that fails
silently.

TCP: `position` is the scanner's measurement reference point - where the
coverage calculation put the scanner at the working distance. Local +Z is the
optical axis, scanner towards part. Teach the ROBOGUIDE UTOOL to match that
point; teach it wrong and every standoff is off by a constant, which shows up as
nothing at all in simulation and as a systematic measuring error on the cell.

Frame: the points are in the station CAD frame. Either keep UFRAME 1 and have
the workshop teach UFRAME 1 once to the CAD origin (the base never moves, so one
teaching serves every part), or express the CAD frame inside the ROBOGUIDE world
and run under UFRAME 0. The second is convenient but a wrong placement shifts
every point by a constant and nothing complains - always verify in ROBOGUIDE
that the points land on the part before reading anything into reachability.

Writes, per export: `<NAME>.LS`, `<NAME>_points.csv`, `<NAME>_frame.json`. The
contract file must also carry the things the export cannot know: that robot
reachability is not modelled, and that robot-arm occlusion of the tracker is not
modelled.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from scanplan.geometry import scanner_frame
from scanplan.roboguide import ls, wpr

_MISSING = object()

# FANUC program comments are limited to 16 characters, and the default below has
# to fit: the header is written verbatim into the controller.
_COMMENT = "scan viewpoints"


def _field(item, key: str, default=_MISSING):
    """Read `key` from a viewpoint dict or from an object's attributes."""
    value = item.get(key, _MISSING) if isinstance(item, dict) else getattr(item, key, _MISSING)
    if value is _MISSING:
        if default is _MISSING:
            raise ValueError(f"viewpoint is missing the '{key}' entry")
        return default
    return value


def viewpoints_to_points(viewpoints) -> list[dict]:
    """Turn the planner's viewpoints into the exporter's point table.

    The frame comes from `scanner_frame(axis, roll_deg)`: it is orthonormal with
    determinant +1 and its local +Z is the optical axis (scanner towards part),
    which is exactly the pose contract the `.LS` writer and the UTOOL use. Nothing
    else about the viewpoint survives - reachability is not this module's business.
    """
    items = list(viewpoints)
    if not items:
        raise ValueError("no viewpoints to export: the viewpoint set is empty")
    points = []
    for index, item in enumerate(items):
        key = _field(item, "vp_key", None)
        if key is None:
            key = _field(item, "id", None)
        if key is None:
            key = index + 1
        rotation = scanner_frame(
            np.asarray(_field(item, "axis"), dtype=float).reshape(3),
            float(_field(item, "roll_deg", 0.0)),
        )
        points.append(
            {
                "vp_key": key,
                "xyz_mm": np.asarray(_field(item, "position"), dtype=float).reshape(3),
                "rotation": rotation,
                "wpr_deg": wpr.rot_to_fanuc_wpr(rotation),
                "stroke_id": _field(item, "stroke_id", None),
                "role": _field(item, "role", "coverage"),
            }
        )
    return points


def apply_placement(
    points: list[dict], rotation: np.ndarray, translation: np.ndarray
) -> list[dict]:
    """Express the (already CAD frame) points in another frame."""
    rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
    translation = np.asarray(translation, dtype=float).reshape(3)
    placed = []
    for point in points:
        rotation_new = rotation @ np.asarray(point["rotation"], dtype=float).reshape(3, 3)
        moved = dict(point)
        moved.update({
            "xyz_mm": rotation @ np.asarray(point["xyz_mm"], dtype=float) + translation,
            "rotation": rotation_new,
            "wpr_deg": wpr.rot_to_fanuc_wpr(rotation_new),
        })
        placed.append(moved)
    return placed


def parse_pose(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse 'x,y,z,w,p,r' (W/P/R in degrees, FANUC convention) into a rot + t pair."""
    try:
        values = [float(value) for value in str(text).split(",")]
    except ValueError as error:
        raise ValueError(f"a pose expects six comma separated numbers, got {text!r}") from error
    if len(values) != 6:
        raise ValueError(f"a pose expects x,y,z,w,p,r, got {text!r}")
    x, y, z, w, p, r = values
    return wpr.wpr_to_rot(w, p, r), np.array([x, y, z], dtype=float)


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
    if n <= 1:
        return list(points)
    pos = np.stack([np.asarray(pt["xyz_mm"], dtype=float).reshape(3) for pt in points])
    axes = np.stack(
        [np.asarray(pt["rotation"], dtype=float).reshape(3, 3)[:, 2] for pt in points]
    )
    anchor = np.asarray(start_near, dtype=float).reshape(3)

    start = int(np.argmin(np.linalg.norm(pos - anchor, axis=1)))
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
                    cost_matrix[a, c]
                    + cost_matrix[b, d]
                    - cost_matrix[a, b]
                    - cost_matrix[c, d]
                )
                if delta < -1e-9:
                    best[i : j + 1] = reversed(best[i : j + 1])
                    improved = True
    return best


def _extents(points: list[dict]) -> dict:
    xyz = np.stack([np.asarray(pt["xyz_mm"], dtype=float).reshape(3) for pt in points])
    return {"min": xyz.min(axis=0).round(3).tolist(), "max": xyz.max(axis=0).round(3).tolist()}


def _contract_blocks(
    prog_name: str,
    *,
    uframe: int,
    utool: int,
    place,
    motion: str,
    speed: str,
    config: str,
    roundtrip: float,
    ordering: dict,
    point_count: int,
    extents: dict,
) -> dict:
    """The frame contract: everything the workshop has to reproduce in ROBOGUIDE."""
    return {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "point_count": point_count,
        "files": {
            "program": f"{prog_name}.LS",
            "points_csv": f"{prog_name}_points.csv",
            "frame_contract": f"{prog_name}_frame.json",
        },
        "frame": {
            "origin": "the station CAD frame origin (0, 0, 0): the frame the planner works in",
            "axes": "the station CAD axes, right handed, Z up",
            "unit": "mm",
            "note": (
                "positions are the planner's CAD-frame numbers and are never moved by "
                "the export; a placement is the only thing that re-expresses them"
            ),
        },
        "uframe_utool": {
            "uframe": uframe,
            "utool": utool,
            "uframe_note": (
                "UFRAME 1 is expected to be taught once to the CAD origin (the base "
                "never moves); UFRAME 0 means the placement above already put the "
                "points into the ROBOGUIDE world"
            ),
        },
        "tcp_contract": {
            "position": (
                "the scanner's measurement reference point: the point the coverage "
                "calculation placed at the working distance from the target"
            ),
            "local_z": (
                "the local +Z of every exported pose is the optical axis, pointing "
                "from the scanner towards the part"
            ),
            "roboguide_action": (
                "teach the UTOOL so that its TCP is this point, with the tool's +Z "
                "along the optical axis"
            ),
            "warning": (
                "a UTOOL taught at a constant offset from that point shifts every "
                "standoff by that offset: simulation shows nothing and the cell "
                "measures wrong"
            ),
        },
        "wpr_convention": {
            "definition": (
                "FANUC fixed-axis XYZ Euler angles, R = Rz(R) * Ry(P) * Rx(W), in degrees"
            ),
            "max_roundtrip_error": roundtrip,
            "meaning": (
                "the largest absolute element error of wpr_to_rot(rot_to_fanuc_wpr(R)) "
                "against R, over the poses actually exported here"
            ),
        },
        "placement": {
            "applied": place or None,
            "meaning": (
                "the pose 'x,y,z,w,p,r' of the CAD frame inside the ROBOGUIDE world; "
                "with a placement the program runs under UFRAME 0"
            ),
            "warning": (
                "only correct if this really is the pose of the shared CAD origin: a "
                "wrong placement shifts every point by a constant and nothing "
                "complains - verify in ROBOGUIDE that the points land on the part"
            ),
        },
        "program": {
            "prog_name": prog_name,
            "motion": motion.upper(),
            "speed": speed,
            "config_string": config,
            "config_note": (
                "the CONFIG string hints which posture solution to take; if ROBOGUIDE "
                "calls a pose unreachable, try another configuration before blaming the "
                "pose (N/F, T, and the three turn numbers)"
            ),
            "comment": _COMMENT,
        },
        "ordering": ordering,
        "not_modelled": [
            "Robot reachability, joint limits and robot collision are not modelled: "
            "the export emits Cartesian poses only, solving the joints is ROBOGUIDE's "
            "job.",
            "Occlusion of the tracker by the robot arm is not modelled: only the "
            "part, the fixture and the base are traced, so a pose whose arm stands "
            "between the tracker and the scanner is exported as usable.",
            "Corner rounding at CNT > 0 is not modelled: the scan geometry assumes the "
            "TCP follows the straight segments, and a rounded corner sweeps less on its "
            "convex side than the line it replaces.",
        ],
        "extents_mm": extents,
    }


def export(
    viewpoints,
    out_dir,
    *,
    prog_name,
    place=None,
    uframe=None,
    utool=1,
    motion="J",
    speed="100%",
    termination="FINE",
    config="N U T, 0, 0, 0",
    order=True,
) -> dict:
    """Write `<NAME>.LS`, `<NAME>_points.csv`, `<NAME>_frame.json` and return the contract.

    `place` is the optional 'x,y,z,w,p,r' pose of the CAD frame inside the
    ROBOGUIDE world; it re-expresses every point and defaults the UFRAME to 0,
    because the points then need no UFRAME taught by hand.

    `order=True` is refused for a stroke plan. Re-sorting splits each stroke into two
    unrelated moves, which silently voids the sweep the coverage figure was computed
    for - the kind of error that survives every visual check. Passing `stroke_id`
    through therefore forces the delivered order, and asking for a re-sort is an error
    rather than a silent re-sort.
    """
    name = wpr.normalise_prog_name(prog_name)
    points = viewpoints_to_points(viewpoints)

    strokes = {point["stroke_id"] for point in points if point["stroke_id"] is not None}
    if strokes and order:
        raise ValueError(
            "these viewpoints belong to a stroke plan (they carry stroke_id), so their "
            "order is the program order and depends on distance; re-sorting would split "
            "each stroke into two unrelated moves. Call export(order=False), or export "
            "the single-pose plan instead")

    if place:
        rotation_place, translation_place = parse_pose(place)
        points = apply_placement(points, rotation_place, translation_place)
    if uframe is None:
        uframe = 0 if place else 1

    if order:
        start = np.mean(
            np.stack([np.asarray(pt["xyz_mm"], dtype=float).reshape(3) for pt in points]),
            axis=0,
        )
        points = order_points(points, start)
        ordering = {
            "method": "nearest neighbour" + (" + 2-opt" if len(points) > 3 else ""),
            "start": "the point nearest the centroid of the exported points",
            "cost": "Euclidean TCP travel in mm (rot_weight defaults to 0)",
            "note": (
                "the planner's viewpoint set carries no order; this is a travel short "
                "cut, not a TSP optimum"
            ),
        }
    else:
        ordering = {
            "method": "as delivered by the planner",
            "note": "ordering was disabled for this export",
        }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ls_text = ls.build_ls(
        points,
        prog_name=name,
        uframe=uframe,
        utool=utool,
        motion=motion,
        speed=speed,
        termination=termination,
        config=config,
        comment=_COMMENT,
    )
    ls_path = out_dir / f"{name}.LS"
    csv_path = out_dir / f"{name}_points.csv"
    frame_path = out_dir / f"{name}_frame.json"
    # an .LS file is plain ASCII; replace rather than fail if a caller's comment is
    # not, so the program still loads
    ls_path.write_text(ls_text, encoding="ascii", errors="replace", newline="\r\n")

    roundtrip = max(
        float(np.abs(np.asarray(pt["rotation"]) - wpr.wpr_to_rot(*pt["wpr_deg"])).max())
        for pt in points
    )

    contract = _contract_blocks(
        name,
        uframe=uframe,
        utool=utool,
        place=place,
        motion=motion,
        speed=speed,
        config=config,
        roundtrip=roundtrip,
        ordering=ordering,
        point_count=len(points),
        extents=_extents(points),
    )
    contract["program"]["termination"] = str(termination).upper()
    contract["program"]["strokes"] = sorted(strokes) if strokes else []
    with frame_path.open("w", encoding="utf-8") as handle:
        json.dump(contract, handle, indent=2)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["order", "vp_key", "x", "y", "z", "w", "p", "r"])

        def cell(value: float) -> float:
            # `or 0.0` keeps a negative zero out of a table meant for eyeballing
            return round(float(value), 4) or 0.0

        for index, point in enumerate(points):
            writer.writerow(
                [
                    index + 1,
                    point["vp_key"],
                    *(cell(value) for value in point["xyz_mm"]),
                    *(cell(value) for value in point["wpr_deg"]),
                ]
            )

    return contract
