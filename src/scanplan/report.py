"""Write the coverage report that has to accompany any viewpoint set.

Must contain: total coverage, target area, viewpoint count; the exempted
(contact) cloud and the unreachable cloud, each with a reason per point; the
incidence-angle distribution; the uncovered samples classified by cause
(blocked by part / fixture / base, never in view, only reachable from poses that
would collide); and the uncovered regions clustered into a short list of blocks.

Must also state what it does NOT include:

* coverage is GEOMETRIC reachability only - robot reachability, joint limits and
  robot collision are decided later, in ROBOGUIDE;
* tracker visibility does not include robot-arm occlusion;
* when a measured scanner parameter is still a placeholder, say which one, so
  nobody reads the coverage figure as a prediction.

Output as a machine-readable JSON plus the human-readable summary; the figures
go in a sibling directory.

This module also owns the report's own vocabulary: the definition block, the leak
clustering and the summary. The viewpoint record itself lives in
:mod:`scanplan.viewpoints` and is re-exported here, so the JSON keys and the export
contract have exactly one definition.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .config import PlannerConfig
from .coverage import CoverageResult, attribute_uncovered
from .path import PathPlan
from .raycast import backend_name
from .scanner_model import ScannerModel
from .scene import Station
from .selection import Selection
from .sheet_split import SheetSplit
from .strokes import StrokeCoverage
from .target_surface import TargetSet, contact_summary
from .tracker import Tracker
from .viewpoints import (  # noqa: F401  (re-exported for the exporters)
    Viewpoint,
    viewpoints_from_path,
    viewpoints_from_selection,
)

_EXEMPT_REASON = "fixture_contact"
_INCIDENCE_BINS = 10
_CLUSTER_LIMIT = 12
# Rounded optical axes that count as "the same angle of view", for the diversity histogram.
_AXIS_DECIMALS = 2


def cluster_points(points, area_mm2, *, cell_mm: float = 30.0,
                   limit: int | None = _CLUSTER_LIMIT) -> list[dict]:
    """Group nearby samples into blocks, largest first.

    The point of clustering is that a reviewer can act on "one and a half dozen small
    features" but not on a list of hundreds of stray samples.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    areas = np.asarray(area_mm2, dtype=float).reshape(-1)
    if len(points) == 0:
        return []
    cells = np.floor(points / cell_mm).astype(np.int64)
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    regions: list[dict] = []
    for group in np.unique(inverse):
        selected = inverse == group
        block = points[selected]
        regions.append({
            "points": int(selected.sum()),
            "area_mm2": round(float(areas[selected].sum()), 1),
            "center": [round(float(v), 1) for v in block.mean(axis=0)],
            "size_mm": [round(float(v), 1) for v in (block.max(axis=0) - block.min(axis=0))],
        })
    regions.sort(key=lambda region: -region["area_mm2"])
    return regions[:limit] if limit is not None else regions


def build_report(*, config: PlannerConfig, model: ScannerModel, station: Station,
                 split: SheetSplit, target: TargetSet, coverage: CoverageResult,
                 selection: Selection, viewpoints,
                 tracker: Tracker | None = None, checks: dict | None = None,
                 path: PathPlan | None = None, strokes: StrokeCoverage | None = None,
                 swept_coverage_pct: float | None = None,
                 pose_only_coverage_pct: float | None = None,
                 runtime: dict | None = None, log=None) -> dict:
    """The report that has to accompany any viewpoint set (spec 7.2)."""
    say = log or (lambda _message: None)
    uncovered = ~selection.covered
    leak_points = np.asarray(target.samples.points)[uncovered]
    leak_normals = np.asarray(target.samples.normals)[uncovered]
    leak_areas = np.asarray(target.samples.area_mm2)[uncovered]
    say(f"  attributing {len(leak_points)} uncovered samples")
    leak_reasons = attribute_uncovered(leak_points, leak_normals, coverage.poses,
                                       station, model, config, obstacles=station.obstacles(),
                                       tracker_ok=coverage.tracker_ok)

    counts: dict[str, int] = {}
    for reason in leak_reasons:
        counts[str(reason)] = counts.get(str(reason), 0) + 1
    leak_reason_list = [str(reason) for reason in leak_reasons]

    angles = np.concatenate([np.degrees(np.arccos(np.clip(row, -1.0, 1.0)))
                             for row in coverage.quality]) if coverage.quality else np.zeros(0)
    limit = float(config.scanner.max_incidence_deg)
    histogram, edges = np.histogram(angles, bins=_INCIDENCE_BINS, range=(0.0, limit))

    placeholders = {
        "max_incidence_deg": config.scanner.max_incidence_deg,
        "range_mm": list(config.scanner.range_mm),
        "fov_at_standoff_mm": list(config.scanner.fov_at_standoff_mm),
        "fov_cap_mm": list(config.scanner.fov_cap_mm),
        "standoff_mm": config.scanner.standoff_mm,
        "body_sphere_r_mm": config.scanner.body_sphere_r_mm,
        "body_clearance_mm": config.scanner.body_clearance_mm,
        "tracker_half_angle_deg": config.tracker.half_angle_deg,
        "tracker_distance_mm": list(config.tracker.distance_mm),
        "tracker_model_position": ([round(float(v), 1) for v in tracker.position]
                                  if tracker is not None else None),
    }

    report = {
        "definition": {
            "name": "topological: one face of the original flat blank",
            "target": f"the {target.face} face of the blank, minus fixture contact",
            "split": "mirror partner per face -> mid-surface PCA -> local sheet normal m; "
                     "|n.m| ~ 1 = a face of the blank, ~ 0 = sheared cross-section; the "
                     "sign of n.m after orienting m says which face",
            "upper_area_mm2": round(split.areas["upper"], 1),
            "lower_area_mm2": round(split.areas["lower"], 1),
            "blanked_edge_area_mm2": round(split.areas["band"], 1),
            "material_thickness_mm": round(split.thickness_mm, 3),
            "orientation_confidence": round(split.oriented_fraction, 4),
            "not_target": ["the other face of the blank",
                           "blanked edge / sheared cross-section", _EXEMPT_REASON],
            "contact_rule": (f"distance < {config.contact_distance_mm:.1f} mm AND on the "
                             "outward side of the sheet"),
            "sampling": f"{config.target.sample_grid_mm:g} mm grid, one sample per cell, "
                        "equal area share",
            "through_axis": [round(float(v), 6) for v in split.through_axis],
            "face_naming_note": (
                "`upper` and `lower` are GEOMETRIC names: they are the two sides of the "
                "blank as the split's orientation seed happens to order them, and that seed "
                "is not anchored to the station file. Swapping the seed swaps the names "
                "while leaving both areas, the thickness and the orientation fraction "
                "identical - so check `through_axis` above (which way the split calls "
                "'up') instead of trusting the word, and prefer face='auto', which compares "
                "both sides."
            ),
        },
        "face": target.face,
        "target_face_mm2": round(target.face_area_mm2, 1),
        "target_points": int(coverage.n_targets),
        "point_area_mm2": round(
            float(target.samples.total_area_mm2 / max(coverage.n_targets, 1)), 3),
        "exempt": {**contact_summary(target), "reason": _EXEMPT_REASON},
        "viewpoints": len(viewpoints),
        "viewpoints_greedy": sum(1 for vp in viewpoints if vp.source == "greedy"),
        "viewpoints_repair": sum(1 for vp in viewpoints if vp.source == "repair"),
        "viewpoints_transition": sum(1 for vp in viewpoints if vp.role != "coverage"),
        "coverage_pct": round(selection.coverage_pct, 3),
        "pool_reach_pct": round(coverage.pool_reach_pct, 3),
        "coverage_area_mm2": round(selection.coverage_area_mm2, 1),
        "uncovered_points": int(uncovered.sum()),
        "uncovered_area_mm2": round(float(leak_areas.sum()), 1),
        "uncovered_by_reason": counts,
        "leak_reasons": leak_reason_list,
        "leak_regions": cluster_points(leak_points, leak_areas),
        "incidence_histogram": {
            "counts": [int(v) for v in histogram],
            "bin_edges_deg": [round(float(v), 3) for v in edges],
            "limit_deg": limit,
            "population": ("every (pose, sample) pair in the candidate POOL, not only the "
                           "pairs the delivered viewpoints cover - so it describes what the "
                           "pool could reach, which is the ceiling the selection works from"),
            "empty_pool": bool(len(angles) == 0),
        },
        "greedy_curve": [[count, round(pct, 3)] for count, pct in selection.curve],
        "program": _program_block(config, viewpoints, path),
        "stroke_plan": _stroke_block(strokes, path, swept_coverage_pct,
                                     pose_only_coverage_pct),
        "angle_diversity": _angle_diversity(coverage, selection.chosen) if path is not None
        else {"available": False,
              "reason": "the delivered set is single poses, so every sample is seen from "
                        "as many axes as there are poses aimed at it"},
        "parameter_provenance": _provenance(config),
        "viewpoint_records": [viewpoint.as_dict() for viewpoint in viewpoints],
        "tracker": tracker.describe() if tracker is not None else {
            "name": None, "position_mm": None,
            "note": "no tracker pose supplied: the tracker constraint was not applied",
        },
        "scope": (
            "Coverage is GEOMETRIC reachability only. Robot reachability, joint limits and "
            "robot collision are NOT modelled here: they are decided later in ROBOGUIDE, so "
            "this viewpoint set deliberately contains poses ROBOGUIDE may reject. That is "
            "expected, not a defect - import the .LS, try another CONFIG before calling a "
            "pose unreachable, and feed the rejected poses back."
        ),
        "tracker_caveat": (
            "The tracker test covers part, fixture and base only. Occlusion by the ROBOT ARM "
            "carrying the scanner is NOT modelled, because there is no robot model at this "
            "stage. Mount the tracker along the part's thin axis (along the part, not square "
            "on to its large faces) and keep it clear of the arm's normal working sector."
        ),
        "placeholders_note": (
            "`placeholders` below lists the scanner and tracker values as delivered, and "
            "`parameter_provenance` says which of them are datasheet facts, which are "
            "derived from the datasheet, and which are still unmeasured. Only the unmeasured "
            "ones can move a result: max_incidence_deg is the most sensitive, and "
            "fov_anchor_distance_mm is the one that changes the footprint everywhere at once. "
            "Do not read a coverage figure as a prediction until those are measured."
        ),
        "placeholders": placeholders,
        "config": {"source": config.source, "override": config.override, "seed": config.seed},
        "evaluation": dict(coverage.counters),
        "alignment_checks": checks,
        "runtime": {"seconds": (runtime or {}).get("seconds"),
                    "ray_backend": (runtime or {}).get("ray_backend", backend_name())},
    }
    return _plain(report)


def _program_block(config: PlannerConfig, viewpoints, path: PathPlan | None) -> dict:
    """What the workshop is handed: the motion, the corners, and what the path costs."""
    block = {
        "motion": config.motion.type,
        "termination": config.motion.termination,
        "speed_mm_s": config.motion.speed_mm_s,
        "speed_note": ("set on the floor and deliberately not optimised here: scan speed "
                       "is a site setting, not a planning variable"),
        "points": len(viewpoints),
        "coverage_points": sum(1 for vp in viewpoints if vp.role == "coverage"),
        "transition_points": sum(1 for vp in viewpoints if vp.role != "coverage"),
        "turns": len(viewpoints),
    }
    if path is None:
        block["ordering"] = ("planner order of single poses; nothing constrains it, so an "
                             "exporter may re-sort")
        return block
    block.update({
        "ordering": path.ordering,
        "path_length_mm": round(float(path.path_length_mm), 1),
        "interrupts": {"translation": int(path.interrupts_a),
                       "reorient": int(path.interrupts_b)},
        "interrupts_note": ("every one of these is the scanner idling while the head moves; "
                            "the site has accepted them"),
        "min_clearance_mm": (None if path.min_clearance_mm is None
                             else round(float(path.min_clearance_mm), 1)),
        "constraint_violations": list(path.violations),
    })
    return block


def _stroke_block(strokes: StrokeCoverage | None, path: PathPlan | None,
                  swept_coverage_pct: float | None,
                  pose_only_coverage_pct: float | None) -> dict:
    """What the stroke stage produced, and which coverage figure is the promise."""
    if strokes is None:
        return {"enabled": False,
                "note": ("stroke planning is off: the delivered set is single poses, the "
                         "same result v1 produces")}
    lengths = np.array([stroke.length_mm for stroke in strokes.strokes]) \
        if strokes.strokes else np.zeros(0)
    block = {
        "enabled": True,
        "delivered": path is not None,
        "candidates": len(strokes.strokes),
        "delivered_strokes": (None if path is None else
                              len({point.stroke_id for point in path.points
                                   if point.stroke_id is not None})),
        "length_mm": ({"min": round(float(lengths.min()), 1),
                       "median": round(float(np.median(lengths)), 1),
                       "max": round(float(lengths.max()), 1)} if len(lengths) else None),
        "min_stroke_clearance_mm": (None if strokes.min_clearance_mm is None
                                    else round(float(strokes.min_clearance_mm), 1)),
        "swept_coverage_pct": None if swept_coverage_pct is None
        else round(float(swept_coverage_pct), 3),
        "pose_only_coverage_pct": None if pose_only_coverage_pct is None
        else round(float(pose_only_coverage_pct), 3),
        "acceptance": ("promised coverage (coverage_pct) counts the strokes only. The swept "
                       "figure adds whatever the transition segments happen to pass over; it "
                       "depends on the ordering, so it is reported for the workshop and never "
                       "used for acceptance or for the fallback comparison"),
    }
    if path is None:
        block["note"] = ("strokes were generated but the delivered plan is the pose-only "
                         "one: the stroke plan did not beat it on coverage, corners and "
                         "travel, which is the guarantee the design asks for")
    return block


def _angle_diversity(coverage: CoverageResult, chosen) -> dict:
    """How many distinct optical axes see each covered sample.

    Reported, never constrained: the workshop's extra passes are hand tremor and
    experience, and the design explicitly does not ask the planner to imitate them.
    """
    if len(chosen) == 0:
        return {"available": False, "reason": "nothing was selected"}
    axes = np.array([np.asarray(coverage.poses[int(index)].axis, dtype=float)
                     for index in chosen])
    lengths = np.linalg.norm(axes, axis=1, keepdims=True)
    keys = np.round(axes / np.maximum(lengths, 1e-12), _AXIS_DECIMALS)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    looks = np.zeros(coverage.n_targets, dtype=np.int32)
    for group in range(len(unique)):
        seen = np.zeros(coverage.n_targets, dtype=bool)
        for row in np.flatnonzero(np.asarray(inverse).ravel() == group):
            seen[coverage.sets[int(chosen[int(row)])]] = True
        looks += seen
    covered = looks > 0
    histogram = np.bincount(looks[covered], minlength=2)[1:]
    return {
        "meaning": "distinct optical axes that observe each covered sample",
        "distinct_axes": int(len(unique)),
        "looks": list(range(1, len(histogram) + 1)),
        "samples": [int(value) for value in histogram],
        "mean_looks": round(float(looks[covered].mean()), 2) if covered.any() else 0.0,
        "note": "reported only; more looks is not a requirement",
    }


def _provenance(config: PlannerConfig) -> dict:
    """Which numbers are measured, which are derived, and which are still guesses.

    The report used to call every value a placeholder, which is as misleading as trusting
    them all: a reader cannot tell which number would move the coverage figure.
    """
    scanner = config.scanner
    return {
        "from_datasheet": {
            "standoff_mm": scanner.standoff_mm,
            "depth_of_field_mm": scanner.depth_of_field_mm,
            "fov_cap_mm": list(scanner.fov_cap_mm),
            "tracker_distance_mm": list(config.tracker.distance_mm),
        },
        "derived_from_datasheet": {
            "range_mm": list(scanner.range_mm),
            "tracker_half_angle_deg": config.tracker.half_angle_deg,
        },
        "unmeasured": {
            "max_incidence_deg": scanner.max_incidence_deg,
            "body_sphere_r_mm": scanner.body_sphere_r_mm,
            "body_clearance_mm": scanner.body_clearance_mm,
            "markers_omnidirectional": config.tracker.markers_omnidirectional,
            "fov_anchor_distance_mm": scanner.fov_anchor_distance_mm,
        },
        "design_choice": {
            "distances_mm": list(scanner.distances_mm),
            "tilts_deg": list(scanner.tilts_deg),
            "rolls_deg": list(scanner.rolls_deg),
            "max_stroke_mm": config.stroke.max_stroke_mm,
            "coverage_step_mm": config.stroke.coverage_step_mm,
            "check_step_mm": config.stroke.check_step_mm,
        },
        "note": ("the unmeasured block is the one to look at before believing a coverage "
                 "figure; max_incidence_deg is the most sensitive of them and "
                 "fov_anchor_distance_mm the one that moves the viewpoint count by ~20%"),
    }


def write_report(out_dir, report: dict) -> Path:
    """Write the machine-readable report."""
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "coverage_report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_viewpoints_json(path, viewpoints) -> Path:
    """Write the viewpoint set, which is the input the `.LS` export reads."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps([viewpoint.as_dict() for viewpoint in viewpoints], indent=2,
                                 ensure_ascii=False), encoding="utf-8")
    return target


def write_viewpoints_csv(path, viewpoints) -> Path:
    """The same set as a flat table, for Excel."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["order", "vp_key", "x_mm", "y_mm", "z_mm",
                         "axis_x", "axis_y", "axis_z", "roll_deg", "n_points",
                         "area_mm2", "mean_incidence_deg", "max_incidence_deg",
                         "tracker_ok", "source"])
        for order, viewpoint in enumerate(viewpoints):
            record = viewpoint.as_dict()
            writer.writerow(
                [order, f"vp_{order:04d}",
                 *[f"{v:.4f}" for v in record["position"]],
                 *[f"{v:.9f}" for v in record["axis"]], f"{record['roll_deg']:.3f}",
                 record["n_points"], f"{record['area_mm2']:.3f}",
                 f"{record['mean_incidence_deg']:.3f}",
                 f"{record['max_incidence_deg']:.3f}",
                 int(record["tracker_ok"]), record["source"]])
    return target


def write_samples_csv(path, points, normals, reasons) -> Path:
    """A point cloud with one reason per point, for eyeballing in a CAD viewer."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    normals = np.asarray(normals, dtype=float).reshape(-1, 3)
    reasons = np.asarray(reasons, dtype=object)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x_mm", "y_mm", "z_mm", "nx", "ny", "nz", "reason"])
        for point, normal, reason in zip(points, normals, reasons, strict=True):
            writer.writerow([f"{point[0]:.4f}", f"{point[1]:.4f}", f"{point[2]:.4f}",
                             f"{normal[0]:.6f}", f"{normal[1]:.6f}", f"{normal[2]:.6f}",
                             str(reason)])
    return target


def write_strokes_json(path, plan: PathPlan, coverage: CoverageResult,
                       config: PlannerConfig) -> Path:
    """The authoritative program order: every move, then every stroke summarised.

    This file, not `viewpoints.json`, is the source of truth for the sequence. It exists
    because a stroke is a PAIR of points: anything that re-sorts the flat point table
    splits a sweep into two unrelated moves, and the coverage figure that was accepted
    then describes something the robot never does.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    areas = np.asarray(coverage.area_mm2, dtype=float)
    points = []
    strokes: list[dict] = []
    seen: dict[int, int] = {}
    for order, point in enumerate(plan.points):
        row = point.coverage_index
        records = coverage.sets[row] if row is not None else np.zeros(0, dtype=np.int64)
        points.append({
            "order": order,
            "stroke_id": point.stroke_id,
            "role": point.role,
            "source": point.source,
            "position_mm": [round(float(v), 4) for v in point.position],
            "axis": [round(float(v), 9) for v in point.axis],
            "roll_deg": float(point.roll_deg),
            "n_points": int(len(records)),
            "area_mm2": round(float(areas[records].sum()), 1) if len(records) else 0.0,
        })
        if point.stroke_id is None:
            continue
        if point.stroke_id not in seen:
            seen[point.stroke_id] = len(strokes)
            strokes.append({
                "id": int(point.stroke_id),
                "source": str(point.source),
                "start_mm": [round(float(v), 4) for v in point.position],
                "end_mm": None,
                "axis": [round(float(v), 9) for v in point.axis],
                "roll_deg": float(point.roll_deg),
                "length_mm": 0.0,
                "n_points": int(len(records)),
                "area_mm2": round(float(areas[records].sum()), 1) if len(records) else 0.0,
            })
        else:
            entry = strokes[seen[point.stroke_id]]
            start = np.asarray(entry["start_mm"], dtype=float)
            entry["end_mm"] = [round(float(v), 4) for v in point.position]
            end = np.asarray(entry["end_mm"], dtype=float)
            entry["length_mm"] = round(float(np.linalg.norm(end - start)), 2)

    body = {
        "motion": {"type": config.motion.type,
                   "termination": config.motion.termination,
                   "speed_mm_s": config.motion.speed_mm_s},
        "ordering": plan.ordering,
        "turns": plan.turns,
        "path_length_mm": round(float(plan.path_length_mm), 1),
        "interrupts": {"translation": int(plan.interrupts_a),
                       "reorient": int(plan.interrupts_b)},
        "constraint_violations": list(plan.violations),
        "points": points,
        "strokes": strokes,
        "note": ("`points` is the program order and must not be re-sorted. `strokes` groups "
                 "the points that belong to one sweep; a stroke whose two ends are not "
                 "consecutive in `points` has been re-ordered and no longer scans what the "
                 "coverage report claims"),
    }
    target.write_text(json.dumps(_plain(body), indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def summary_lines(report: dict) -> list[str]:
    """The console summary, ending with the caveat a reader must not miss."""
    reasons = ", ".join(f"{key} {value}" for key, value in sorted(
        report["uncovered_by_reason"].items())) or "none"
    lines = [
        f"target face            {report['target_face_mm2'] / 1e6:.4f} m2 ({report['face']})",
        f"target samples         {report['target_points']}",
        f"exempt (clamped)       {report['exempt']['points']} samples "
        f"{report['exempt']['area_mm2'] / 1e6:.4f} m2",
        f"coverage               {report['coverage_pct']:.2f}%  "
        f"(pool could reach {report['pool_reach_pct']:.2f}%)",
        f"points                 {report['viewpoints']} "
        f"({report['viewpoints_greedy']} greedy + {report['viewpoints_repair']} repair"
        + (f" + {report['viewpoints_transition']} transition"
           if report.get("viewpoints_transition") else "") + ")",
        f"uncovered              {report['uncovered_points']} samples "
        f"{report['uncovered_area_mm2'] / 1e6:.4f} m2  [{reasons}]",
        f"max incidence          {report['placeholders']['max_incidence_deg']:.0f} deg "
        "(PLACEHOLDER)",
    ]
    path = report.get("program", {})
    if path.get("path_length_mm") is not None:
        interrupts = path.get("interrupts", {})
        lines += [
            f"motion                 {path['motion']} {path['speed_mm_s']:g} mm/s "
            f"{path['termination']}",
            f"path                   {path['turns']} corners, "
            f"{path['path_length_mm']:.0f} mm, "
            f"{interrupts.get('translation', 0)} translation + "
            f"{interrupts.get('reorient', 0)} re-orient interrupts",
        ]
    strokes = report.get("stroke_plan", {})
    if strokes.get("enabled") and strokes.get("delivered"):
        lines.append(f"strokes                "
                     f"{strokes.get('delivered_strokes') or strokes['candidates']} delivered "
                     f"of {strokes['candidates']} candidates, swept coverage "
                     f"{strokes['swept_coverage_pct']:.2f}%")
        for violation in path.get("constraint_violations", []):
            lines.append(f"CONSTRAINT             {violation}")
    lines += [
        f"ray backend            {report['runtime']['ray_backend']}",
        "NOTE: coverage is geometric reachability only - robot reachability, joint limits "
        "and robot collision are checked in ROBOGUIDE, and robot-arm occlusion of the "
        "tracker is not modelled.",
    ]
    return lines


def summary_text(report: dict) -> str:
    """`summary_lines` as one string, for a `summary.txt` beside the report."""
    return "\n".join(summary_lines(report)) + "\n"


def _plain(value):
    """Recursively convert numpy scalars and arrays so the report is JSON-serialisable."""
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
