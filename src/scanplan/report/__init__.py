"""Write the coverage report that has to accompany any viewpoint set.

Must contain: total coverage, target area, viewpoint count; the exempted
(contact) cloud and the unreachable cloud, each with a reason per point; the
incidence-angle distribution; the uncovered samples classified by cause
(blocked by part / fixture / base, never in view, only reachable from poses that
would collide); and the uncovered regions clustered into a short list of blocks.

Must also state what it does NOT include:

* coverage is GEOMETRIC reachability, filtered by the arm's joint window when
  `robot.enabled` is set - the limits themselves are enforced, ROBOT COLLISION is
  decided later, in ROBOGUIDE;
* tracker visibility does not include robot-arm occlusion;
* when a measured scanner parameter is still a placeholder, say which one, so
  nobody reads the coverage figure as a prediction.

Output as a machine-readable JSON plus the human-readable summary; the figures
go in a sibling directory.

The sections live in sibling modules - :mod:`definition` (what was targeted and what
is a guess), :mod:`coverage` (how much was covered and why not), :mod:`program`
(motion and path cost), :mod:`strokes` (the stroke stage and `strokes.json`),
:mod:`viewpoints` (the viewpoint files) and :mod:`summary` (the console text). This
module only fixes the order they appear in, so adding a metric means editing one
section rather than this assembly. The viewpoint record itself lives in
:mod:`scanplan.viewpoints` and is re-exported here, so the JSON keys and the export
contract have exactly one definition.
"""

from __future__ import annotations

from ..config import PlannerConfig
from ..coverage import CoverageResult
from ..path import PathPlan
from ..raycast import backend_name
from ..robot import RobotArm
from ..scanner_model import ScannerModel
from ..scene import Station
from ..selection import Selection
from ..sheet_split import SheetSplit
from ..strokes import StrokeCoverage
from ..target_surface import TargetSet
from ..tracker import Tracker
from ..viewpoints import (  # noqa: F401  (re-exported for the exporters)
    Viewpoint,
    viewpoints_from_path,
    viewpoints_from_selection,
)
from .coverage import angle_diversity_block, cluster_points, coverage_block
from .definition import (
    PLACEHOLDERS_NOTE,
    TRACKER_CAVEAT,
    definition_block,
    derived_block,
    placeholders_block,
    provenance,
    scope_note,
)
from .program import program_block
from .robot import robot_block
from .strokes import stroke_block, write_strokes_json
from .summary import summary_lines, summary_text
from .viewpoints import write_samples_csv, write_viewpoints_csv, write_viewpoints_json
from .writers import plain, write_report

__all__ = [
    "Viewpoint",
    "build_report",
    "cluster_points",
    "summary_lines",
    "summary_text",
    "viewpoints_from_path",
    "viewpoints_from_selection",
    "write_report",
    "write_samples_csv",
    "write_strokes_json",
    "write_viewpoints_csv",
    "write_viewpoints_json",
]


def build_report(*, config: PlannerConfig, model: ScannerModel, station: Station,
                 split: SheetSplit, target: TargetSet, coverage: CoverageResult,
                 selection: Selection, viewpoints,
                 tracker: Tracker | None = None, checks: dict | None = None,
                 path: PathPlan | None = None, strokes: StrokeCoverage | None = None,
                 robot: RobotArm | None = None,
                 swept_coverage_pct: float | None = None,
                 pose_only_coverage_pct: float | None = None,
                 runtime: dict | None = None, log=None) -> dict:
    """The report that has to accompany any viewpoint set (spec 7.2)."""
    say = log or (lambda _message: None)
    report = {
        **definition_block(config=config, target=target, split=split, coverage=coverage),
        "viewpoints": len(viewpoints),
        "viewpoints_greedy": sum(1 for vp in viewpoints if vp.source == "greedy"),
        "viewpoints_repair": sum(1 for vp in viewpoints if vp.source == "repair"),
        "viewpoints_transition": sum(1 for vp in viewpoints if vp.role != "coverage"),
        **coverage_block(config=config, model=model, station=station, target=target,
                         coverage=coverage, selection=selection, robot=robot, say=say),
        "program": program_block(config, viewpoints, path, selection),
        "stroke_plan": stroke_block(strokes, path, swept_coverage_pct,
                                    pose_only_coverage_pct),
        "angle_diversity": angle_diversity_block(coverage, selection.chosen, path),
        "robot": robot_block(robot, config),
        "parameter_provenance": provenance(config),
        "derived_parameters": derived_block(config),
        "viewpoint_records": [viewpoint.as_dict() for viewpoint in viewpoints],
        "tracker": tracker.describe() if tracker is not None else {
            "name": None, "position_mm": None,
            "note": "no tracker pose supplied: the tracker constraint was not applied",
        },
        "scope": scope_note(config),
        "tracker_caveat": TRACKER_CAVEAT,
        "placeholders_note": PLACEHOLDERS_NOTE,
        "placeholders": placeholders_block(config, tracker),
        "config": {"source": config.source, "override": config.override, "seed": config.seed},
        "evaluation": dict(coverage.counters),
        "alignment_checks": checks,
        "runtime": {"seconds": (runtime or {}).get("seconds"),
                    "ray_backend": (runtime or {}).get("ray_backend", backend_name())},
    }
    return plain(report)
