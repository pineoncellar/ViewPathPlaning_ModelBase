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
from .raycast import backend_name
from .scanner_model import ScannerModel
from .scene import Station
from .selection import Selection
from .sheet_split import SheetSplit
from .target_surface import TargetSet, contact_summary
from .tracker import Tracker
from .viewpoints import Viewpoint, viewpoints_from_selection  # noqa: F401  (re-export)

_EXEMPT_REASON = "fixture_contact"
_INCIDENCE_BINS = 10
_CLUSTER_LIMIT = 12


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
                 runtime: dict | None = None, log=None) -> dict:
    """The report that has to accompany any viewpoint set (spec 7.2)."""
    say = log or (lambda _message: None)
    uncovered = ~selection.covered
    leak_points = np.asarray(target.samples.points)[uncovered]
    leak_normals = np.asarray(target.samples.normals)[uncovered]
    leak_areas = np.asarray(target.samples.area_mm2)[uncovered]
    say(f"  attributing {len(leak_points)} uncovered samples")
    leak_reasons = attribute_uncovered(leak_points, leak_normals, coverage.poses,
                                       station, model, config, obstacles=station.obstacles())

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
        },
        "face": target.face,
        "target_face_mm2": round(target.face_area_mm2, 1),
        "target_points": int(coverage.n_targets),
        "point_area_mm2": round(
            float(target.samples.total_area_mm2 / max(coverage.n_targets, 1)), 3),
        "exempt": {**contact_summary(target), "reason": _EXEMPT_REASON},
        "viewpoints": len(viewpoints),
        "viewpoints_greedy": len(selection.greedy),
        "viewpoints_repair": len(selection.repair),
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
            "empty_pool": bool(len(angles) == 0),
        },
        "greedy_curve": [[count, round(pct, 3)] for count, pct in selection.curve],
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
            "Every value below stands in for a measurement that has not been taken on the "
            "real device. max_incidence_deg is the most sensitive one: it is the only "
            "parameter that visibly moves the coverage figure, so do not read any coverage "
            "number as a prediction until it is measured."
        ),
        "placeholders": placeholders,
        "config": {"source": config.source, "override": config.override, "seed": config.seed},
        "evaluation": dict(coverage.counters),
        "alignment_checks": checks,
        "runtime": {"seconds": (runtime or {}).get("seconds"),
                    "ray_backend": (runtime or {}).get("ray_backend", backend_name())},
    }
    return _plain(report)


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


def summary_lines(report: dict) -> list[str]:
    """The console summary, ending with the caveat a reader must not miss."""
    reasons = ", ".join(f"{key} {value}" for key, value in sorted(
        report["uncovered_by_reason"].items())) or "none"
    return [
        f"target face            {report['target_face_mm2'] / 1e6:.4f} m2 ({report['face']})",
        f"target samples         {report['target_points']}",
        f"exempt (clamped)       {report['exempt']['points']} samples "
        f"{report['exempt']['area_mm2'] / 1e6:.4f} m2",
        f"coverage               {report['coverage_pct']:.2f}%  "
        f"(pool could reach {report['pool_reach_pct']:.2f}%)",
        f"viewpoints             {report['viewpoints']} "
        f"({report['viewpoints_greedy']} greedy + {report['viewpoints_repair']} repair)",
        f"uncovered              {report['uncovered_points']} samples "
        f"{report['uncovered_area_mm2'] / 1e6:.4f} m2  [{reasons}]",
        f"max incidence          {report['placeholders']['max_incidence_deg']:.0f} deg "
        "(PLACEHOLDER)",
        f"ray backend            {report['runtime']['ray_backend']}",
        "NOTE: coverage is geometric reachability only - robot reachability, joint limits "
        "and robot collision are checked in ROBOGUIDE, and robot-arm occlusion of the "
        "tracker is not modelled.",
    ]


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
