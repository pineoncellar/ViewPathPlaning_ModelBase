"""The stroke block and `strokes.json`: the authoritative program order.

Kept together because the prose in the report and the file the exporter reads must not
disagree about which coverage figure is the promise.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import PlannerConfig
from ..coverage import CoverageResult
from ..path import PathPlan
from ..strokes import StrokeCoverage
from .writers import plain


def stroke_block(strokes: StrokeCoverage | None, path: PathPlan | None,
                 swept_coverage_pct: float | None,
                 pose_only_coverage_pct: float | None) -> dict:
    """What the stroke stage produced, and which coverage figure is the promise."""
    if strokes is None:
        return {"enabled": False,
                "note": ("stroke planning is off: the delivered set is single poses, the "
                         "same result v1 produces")}
    lengths = np.array([stroke.length_mm for stroke in strokes.strokes]) \
        if strokes.strokes else np.zeros(0)
    delivered = None if path is None else int(path.coverage_strokes)
    block = {
        "enabled": True,
        "delivered": bool(delivered),
        "candidates": len(strokes.strokes),
        # Counted from the items actually delivered, and a delivered POSE is a zero-length
        # stroke - so this is the number of real sweeps, not the number of distinct
        # `stroke_id`s in the path.
        "delivered_strokes": delivered,
        "length_mm": ({"min": round(float(lengths.min()), 1),
                       "median": round(float(np.median(lengths)), 1),
                       "max": round(float(lengths.max()), 1)} if len(lengths) else None),
        "min_stroke_clearance_mm": (None if strokes.min_clearance_mm is None
                                    else round(float(strokes.min_clearance_mm), 1)),
        "swept_coverage_pct": None if swept_coverage_pct is None
        else round(float(swept_coverage_pct), 3),
        "pose_only_coverage_pct": None if pose_only_coverage_pct is None
        else round(float(pose_only_coverage_pct), 3),
        "acceptance": ("promised coverage (coverage_pct) counts every delivered coverage "
                       "point - a selected pose is delivered as a zero-length stroke - so "
                       "it is what the program actually sweeps. `swept_coverage_pct` adds "
                       "whatever the transition segments happen to pass over; it depends on "
                       "the ordering, so it is reported for the workshop and never used for "
                       "acceptance or for the fallback comparison"),
    }
    if not delivered:
        block["note"] = ("the delivered plan is the pose-only one: the mixed plan did not "
                         "beat it on target, corners and travel, which is the guarantee the "
                         "design asks for")
    return block


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
    target.write_text(json.dumps(plain(body), indent=2, ensure_ascii=False), encoding="utf-8")
    return target
