"""Writing a finished plan: the eight files of the deliverable.

Kept apart from the orchestration because this is the part a reviewer reads, and it
changes when the deliverable contract changes rather than when the algorithm does.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from scanview import DEFAULT_MAX_FACES, write_run_outputs

from ..report import (
    summary_text,
    write_report,
    write_samples_csv,
    write_strokes_json,
    write_viewpoints_csv,
    write_viewpoints_json,
)

if TYPE_CHECKING:
    from . import PlanResult


def write_outputs(result: PlanResult, out_dir, *, write_3d: bool = True,
                  max_faces: int = DEFAULT_MAX_FACES) -> dict[str, Path]:
    """Write the viewpoint set, both point clouds, the report and the 3D view."""
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    uncovered = ~result.selection.covered
    samples = result.target.samples
    paths = {
        "viewpoints.json": write_viewpoints_json(folder / "viewpoints.json",
                                                 result.viewpoints),
        "viewpoints.csv": write_viewpoints_csv(folder / "viewpoints.csv",
                                               result.viewpoints),
        "exempt_points.csv": write_samples_csv(
            folder / "exempt_points.csv", result.target.exempt.points,
            result.target.exempt.normals, result.target.exempt.reasons),
        "unreachable_points.csv": write_samples_csv(
            folder / "unreachable_points.csv", np.asarray(samples.points)[uncovered],
            np.asarray(samples.normals)[uncovered],
            result.report.get("leak_reasons", np.full(int(uncovered.sum()),
                                                      "uncovered", dtype=object))),
        "coverage_report.json": write_report(folder, result.report),
        "summary.txt": _write_summary(folder, result.report),
    }
    if result.path is not None:
        paths["strokes.json"] = write_strokes_json(folder / "strokes.json", result.path,
                                                   result.coverage, result.config)
    if write_3d:
        coverage_sets, chosen = _delivered_sets(result)
        paths.update(write_run_outputs(
            folder,
            part=result.station.part,
            fixture=result.station.fixture,
            base=result.station.base,
            viewpoints=result.viewpoints,
            target_points=samples.points,
            target_covered=result.selection.covered,
            coverage_sets=coverage_sets,
            chosen=chosen,
            tracker=result.tracker,
            scanner=result.config.scanner,
            max_faces=max_faces,
            stats={"face": result.report.get("face"),
                   "coverage_pct": result.report.get("coverage_pct")},
        ))
    return paths


def _delivered_sets(result: PlanResult):
    """One coverage set per DELIVERED point, so the 3D view's bit mask lines up.

    A stroke plan delivers more points than it selects items, and the viewer indexes its
    per-sample bit mask by delivered order - so the mask has to be rebuilt from the path
    rather than passed straight through from the selection.
    """
    if result.path is None:
        return result.coverage.sets, result.selection.chosen
    empty = np.zeros(0, dtype=np.int64)
    sets = [empty if point.coverage_index is None
            else result.coverage.sets[point.coverage_index] for point in result.path.points]
    return sets, list(range(len(sets)))


def _write_summary(folder: Path, report: dict) -> Path:
    path = Path(folder) / "summary.txt"
    path.write_text(summary_text(report), encoding="utf-8")
    return path
