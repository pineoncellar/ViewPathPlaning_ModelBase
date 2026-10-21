"""The delivery stage: turn a selection into an ordered, connected path.

Two candidate deliverables are built here - the mixed one (poses and strokes) and the
pose-only one - and ranked on actual corners, so `path` mode cannot silently degrade
into the unverified v1 motion it exists to replace.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..config import PlannerConfig
from ..coverage import CoverageResult, evaluate
from ..path import PathPlan, build_path, prune, transition_poses
from ..scanner_model import ScannerModel
from ..scene import Station
from ..selection import Selection, select_groups
from ..strokes import Stroke, StrokeBuilder, StrokeCoverage, pose_stroke
from ..target_surface import TargetSet
from .ranking import _rank_plan

# Pruning keeps the requirement, not the achieved peak: pruning against the peak can
# never remove anything, which would make the stage pointless by construction.
PRUNE_TARGET_PCT = 99.0


def _stroke_builder(cfg: PlannerConfig, target: TargetSet, station: Station,
                    model: ScannerModel, gate) -> StrokeBuilder | None:
    """The stroke builder, or None when the delivered set is single poses."""
    if cfg.target.delivery_mode == "lean":
        # The lean deliverable is v1 plus a stopping rule: poses alone, no connectors, and
        # therefore no claim about the motion between them. Strokes cannot be part of it.
        return None
    if not cfg.stroke.enabled:
        return None
    if cfg.stroke.max_stroke_mm <= 0.0:
        return None
    return StrokeBuilder(target.samples, station, model, cfg, visible=gate)


def _group_selection(combined: CoverageResult, coverage: CoverageResult,
                     stroke_cov: StrokeCoverage, cfg: PlannerConfig, *,
                     log=None) -> Selection:
    """Set cover over poses and strokes together, priced in the configured unit.

    `corners` is the default because it is what the workshop counts: one stop for a pose,
    two for a stroke, plus the transition each one drags along. `millimetres` is the
    upstream design's pricing and is kept so the two can be compared on the same part.
    """
    stop = float(cfg.stroke.stop_cost_mm)
    transition = float(cfg.stroke.transition_corners_estimate)
    n_poses = len(coverage.poses)
    if cfg.stroke.cost_model == "corners":
        costs = [1.0 + transition] * n_poses + [2.0 + transition] * len(stroke_cov.strokes)
    else:
        costs = ([stop] * n_poses
                 + [stroke.length_mm + stop for stroke in stroke_cov.strokes])
    repair_ok = [True] * n_poses + [False] * len(stroke_cov.strokes)
    return select_groups(combined, costs, repair_ok=repair_ok,
                         coverage_target_pct=cfg.target.coverage_target_pct, log=log)


def _delivery(combined: CoverageResult, rows: Sequence[int], sources: dict[int, str],
              stroke_cov: StrokeCoverage, coverage: CoverageResult, cfg: PlannerConfig,
              station: Station, model: ScannerModel, gate, *,
              log=None) -> tuple[Selection, PathPlan]:
    """Prune and order an explicit set of chosen rows into a delivered path.

    Poses go in as zero-length strokes. That is not a trick: the module documentation of
    `strokes` already says a stroke of length zero is a viewpoint, and delivering them this
    way is what keeps the reported coverage and the program's actual sweep the same number.
    Pruning sees every row, so a pose the target does not need can go too.
    """
    n_poses = len(coverage.poses)
    say = log or (lambda _message: None)
    items: list[Stroke] = []
    item_rows: list[int] = []
    item_sets: list[np.ndarray] = []
    for row in rows:
        item_rows.append(int(row))
        item_sets.append(combined.sets[int(row)])
        if row < n_poses:
            pose = coverage.poses[int(row)]
            items.append(pose_stroke(pose.position, pose.axis, pose.roll_deg))
        else:
            items.append(stroke_cov.strokes[int(row) - n_poses])

    target_pct = cfg.target.coverage_target_pct
    kept = prune(items, item_sets, list(range(len(items))), combined.area_mm2,
                 combined.target_area_mm2,
                 target_pct=PRUNE_TARGET_PCT if target_pct is None else target_pct,
                 max_rounds=cfg.stroke.max_prune_rounds)
    if len(kept) < len(items):
        say(f"  pruning: {len(items)} selected items -> {len(kept)} while staying at "
            f"{(PRUNE_TARGET_PCT if target_pct is None else target_pct):g}%")
    kept_rows = [item_rows[index] for index in kept]
    rebuilt = _rebuild_selection(combined, kept_rows, sources, None)
    path = build_path([items[index] for index in kept], cfg=cfg, station=station,
                      model=model, coverage_index=lambda index: kept_rows[index],
                      sources=lambda index: sources.get(kept_rows[index], "greedy"),
                      visible=gate, log=log)
    return rebuilt, path


def _finish_stroke_plan(combined: CoverageResult, coverage: CoverageResult,
                        stroke_cov: StrokeCoverage, selection: Selection,
                        pose_selection: Selection, target: TargetSet, station: Station,
                        model: ScannerModel, cfg: PlannerConfig, gate,
                        *, log=None):
    """Build both deliveries, then keep the one that is better on the same scale.

    Two candidate deliverables, both ordered and connected by the same code:

    * the mixed one - the selected poses and the strokes that survived pruning;
    * the pose-only one - which is what the run would deliver without a stroke stage.

    Comparing them on ACTUAL corners rather than on "one per pose, two per stroke" is the
    point. Once the selection stops at a coverage target, the pose pool also stops there,
    so a hand-waved comparison hands the whole deliverable to poses - with no connectors -
    and `path` mode silently becomes the unverified v1 motion it exists to replace.

    The ranking is the design's guarantee ("not worse than today"): met the target first,
    then fewer corners, then less travel.
    """
    say = log or (lambda _message: None)
    target_pct = cfg.target.coverage_target_pct
    sources = {int(row): str(source)
               for row, source in zip(selection.chosen, selection.sources, strict=True)}
    mixed_sel, mixed_path = _delivery(combined, selection.chosen, sources, stroke_cov,
                                      coverage, cfg, station, model, gate, log=log)
    pose_sources = {int(row): str(source) for row, source
                    in zip(pose_selection.chosen, pose_selection.sources, strict=True)}
    pose_sel, pose_path = _delivery(combined, pose_selection.chosen, pose_sources,
                                    stroke_cov, coverage, cfg, station, model, gate,
                                    log=log)
    mixed_sel.corners = mixed_path.turns
    pose_sel.corners = pose_path.turns

    mixed_rank = _rank_plan(mixed_sel.coverage_pct, mixed_path.turns,
                            mixed_path.path_length_mm, target_pct)
    pose_rank = _rank_plan(pose_sel.coverage_pct, pose_path.turns,
                           pose_path.path_length_mm, target_pct)
    if pose_rank >= mixed_rank:
        say(f"  the mixed plan ({mixed_sel.coverage_pct:.2f}%, {mixed_path.turns} corners, "
            f"{mixed_path.path_length_mm:.0f} mm) does not beat the pose-only plan "
            f"({pose_sel.coverage_pct:.2f}%, {pose_path.turns} corners, "
            f"{pose_path.path_length_mm:.0f} mm): delivering the poses")
        return pose_sel, pose_path, _swept_pct(pose_sel, pose_path, target, station, model,
                                               cfg, combined, gate, log=log)
    say(f"  the mixed plan wins: {mixed_sel.coverage_pct:.2f}%, {mixed_path.turns} corners "
        f"against the pose-only {pose_sel.coverage_pct:.2f}%, {pose_path.turns}")
    return mixed_sel, mixed_path, _swept_pct(mixed_sel, mixed_path, target, station, model,
                                             cfg, combined, gate, log=log)


def _swept_pct(rebuilt: Selection, path: PathPlan, target: TargetSet, station: Station,
               model: ScannerModel, cfg: PlannerConfig, combined: CoverageResult, gate,
               *, log=None) -> float:
    """The coverage the path actually sweeps, transitions included.

    Reported for the workshop and never used for acceptance: it depends on the ordering,
    so a plan could "gain" coverage by flying over the part.
    """
    say = log or (lambda _message: None)
    swept_pct = rebuilt.coverage_pct
    swings = transition_poses(path, cfg)
    if swings:
        swept = evaluate(swings, target.samples, station, model, cfg, visible=gate, log=log)
        covered = rebuilt.covered.copy()
        for row in swept.sets:
            covered[row] = True
        if combined.target_area_mm2 > 0.0:
            swept_pct = float(100.0 * combined.area_mm2[covered].sum()
                              / combined.target_area_mm2)
        say(f"  the path sweeps {swept_pct:.2f}% against {rebuilt.coverage_pct:.2f}% "
            f"promised by the chosen items alone")
    return swept_pct


def _rebuild_selection(combined: CoverageResult, rows, sources: dict[int, str],
                       previous: Selection | None = None) -> Selection:
    """A `Selection` over an explicit row list, after pruning removed some rows."""
    covered = np.zeros(combined.n_targets, dtype=bool)
    for row in rows:
        covered[combined.sets[row]] = True
    area = np.asarray(combined.area_mm2, dtype=float)
    labels = [sources.get(int(row), "greedy") for row in rows]
    return Selection(
        chosen=[int(row) for row in rows],
        greedy=[int(row) for row, label in zip(rows, labels, strict=True)
                if label == "greedy"],
        repair=[int(row) for row, label in zip(rows, labels, strict=True)
                if label == "repair"],
        covered=covered, curve=[] if previous is None else list(previous.curve),
        coverage_area_mm2=float(area[covered].sum()),
        target_area_mm2=combined.target_area_mm2,
        sources=np.array(labels, dtype=object))
