"""Orchestrate the planning stage end to end.

Order: load and align the station, split the sheet, sample the target, generate
candidates, evaluate coverage, select, then export the viewpoint set and the
report.

Two ordering decisions worth keeping:

1. Filter before planning. Candidates that are unusable (occluded, out of range,
   colliding, invisible to the tracker) must be removed before set cover runs,
   otherwise the selected set leaves holes that cannot be patched.
2. The tracker constraint filters POSES, not target points, so it belongs inside
   the coverage evaluation - not as a post-filter on the result.

Choosing which face of the blank to scan lives here too. When `face` is "auto",
plan both faces and keep the one with the higher reachability; no single
geometric quantity decides it reliably.

The refinement pass is the answer to the measured finding that the viewpoint count
is limited by candidate generation rather than by set cover: after the first
selection round, the still-unreachable samples are re-aimed on a finer grid and the
selection is re-run over the enlarged pool. Whether that raised the pool's ceiling
is reported, because a residue that no amount of aiming can reach means something
different from one that a better pool would fix.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scanview import DEFAULT_MAX_FACES, write_run_outputs

from .candidates import generate, refine
from .config import PlannerConfig
from .coverage import CoverageResult, evaluate, merge
from .path import PathPlan, build_path, prune, transition_poses
from .raycast import backend_name
from .report import (
    Viewpoint,
    build_report,
    summary_text,
    viewpoints_from_path,
    viewpoints_from_selection,
    write_report,
    write_samples_csv,
    write_strokes_json,
    write_viewpoints_csv,
    write_viewpoints_json,
)
from .scanner_model import ScannerModel
from .scene import Station
from .selection import Selection, select, select_groups
from .sheet_split import SheetSplit, split_sheet
from .strokes import StrokeBuilder, StrokeCoverage, combine, merge_stroke_coverage
from .target_surface import TargetSet, build_target
from .tracker import Tracker, visible_mask

FACES = ("upper", "lower", "auto")
# Pruning keeps the requirement, not the achieved peak: pruning against the peak can
# never remove anything, which would make the stage pointless by construction.
PRUNE_TARGET_PCT = 99.0


@dataclass
class PlanResult:
    """Everything one planning run produced, so a caller can re-derive any number."""

    config: PlannerConfig
    station: Station
    split: SheetSplit
    target: TargetSet
    coverage: CoverageResult
    selection: Selection
    viewpoints: list[Viewpoint]
    report: dict
    checks: dict
    face_candidates: dict[str, float] = field(default_factory=dict)
    tracker: Tracker | None = None
    path: PathPlan | None = None
    strokes: StrokeCoverage | None = None
    pose_only_coverage_pct: float | None = None


def plan(station: Station, cfg: PlannerConfig, *, face: str | None = None,
         tracker: Tracker | None = None, split: SheetSplit | None = None,
         log: Callable[[str], None] | None = None) -> PlanResult:
    """Plan the scan of one face of the blank, and report what was missed.

    `face` defaults to the configuration; `"auto"` plans both faces and keeps the one
    with the higher coverage, which is the only reliable way to answer "which side is
    up", and records both figures in `face_candidates`.
    """
    say = log or (lambda message: print(message, flush=True))
    started = time.time()
    requested = cfg.target.face if face is None else face
    if requested not in FACES:
        raise ValueError(f"face must be one of {FACES}, not {requested!r}")

    say("station alignment checks")
    # A diagnostic, not a measurement: 40k samples per model is ample for the distances
    # and clearances it reports, and it runs on every plan including the two-face "auto"
    # run, so the default of 200k would dominate the runtime for no extra information.
    checks = station.check_alignment(seed=cfg.seed, samples=40_000)
    for warning in checks["warnings"]:
        say(f"  WARNING: {warning}")

    if split is None:
        say("splitting the blank into its two faces")
        split = split_sheet(station.part, band_deg=cfg.target.band_deg, knn=cfg.target.knn,
                            seed=cfg.seed, log=log)

    if requested == "auto":
        candidates: dict[str, PlanResult] = {}
        for option in ("upper", "lower"):
            try:
                candidates[option] = _plan_face(station, cfg, split, option, tracker,
                                               checks, started, log)
            except ValueError as error:
                say(f"  {option} face cannot be planned: {error}")
        if not candidates:
            raise ValueError("no target face could be sampled: check the split of this part")
        best = max(candidates, key=lambda name: candidates[name].selection.coverage_pct)
        result = candidates[best]
        result.face_candidates = {name: candidate.selection.coverage_pct
                                 for name, candidate in candidates.items()}
        say(f"  auto chose the {best} face")
        result.report["face_candidates"] = result.face_candidates
        return result

    return _plan_face(station, cfg, split, requested, tracker, checks, started, log)


def _plan_face(station: Station, cfg: PlannerConfig, split: SheetSplit, face: str,
               tracker: Tracker | None, checks: dict, started: float,
               log: Callable[[str], None] | None) -> PlanResult:
    """Plan one named face, from target sampling through to the report."""
    say = log or (lambda message: print(message, flush=True))
    model = ScannerModel(cfg.scanner)
    target = build_target(station.part, split, face, station, cfg, log=log)
    if len(target.samples.points) == 0:
        raise ValueError(f"the {face} target face produced no samples to scan")

    gate = None
    if tracker is not None:
        obstacles = station.obstacles()
        gate = lambda positions: visible_mask(tracker, positions, obstacles)  # noqa: E731

    say(f"generating candidates for the {face} face")
    poses = generate(target.samples.points, target.samples.normals, cfg.scanner,
                     seed=cfg.seed, log=log)
    mark = time.time()
    coverage = evaluate(poses, target.samples, station, model, cfg, visible=gate, log=log)
    say(f"  pose coverage took {time.time() - mark:.0f} s")
    mark = time.time()

    pose_selection = select(coverage, coverage.area_mm2, log=log)
    builder = _stroke_builder(cfg, target, station, model, gate)
    if builder is None:
        say("  stroke planning is off: the delivered set is single poses")
    stroke_cov: StrokeCoverage | None = None
    combined = coverage
    selection = pose_selection
    if builder is not None:
        say("growing scan strokes from the aim grid")
        stroke_cov = builder.evaluate(
            builder.generate(target.samples.points, target.samples.normals,
                             aim_grid_mm=cfg.scanner.aim_grid_mm, log=log),
            log=log)
        combined = combine(coverage, stroke_cov)
        selection = _group_selection(combined, coverage, stroke_cov, cfg, log=log)
        say(f"  stroke stage took {time.time() - mark:.0f} s")
        say(f"  pose-only reference: {len(pose_selection.chosen)} viewpoints at "
            f"{pose_selection.coverage_pct:.2f}%; strokes reach "
            f"{len(selection.chosen)} items at {selection.coverage_pct:.2f}%")

    counters = dict(coverage.counters)
    counters["refine_rounds"] = 0
    counters["refine_candidates"] = 0

    for round_index in range(cfg.scanner.refine_rounds):
        # Aim at every sample still uncovered - not only those the current pool can reach.
        # Greedy covers the whole reachable set, so the reachable-but-uncovered set is
        # always empty; what refinement exists to improve is the pool's ceiling, and the
        # samples to re-aim at are exactly the ones nothing in the pool can see.
        leftover = np.flatnonzero(~selection.covered)
        if len(leftover) == 0:
            break
        reachable = combined.pool_reach()
        say(f"refinement round {round_index + 1}: {len(leftover)} uncovered samples "
            f"({int((~reachable[leftover]).sum())} of them out of the pool's reach)")
        ceiling_before = combined.pool_reach_pct
        extra = refine(target.samples.points[leftover], target.samples.normals[leftover],
                       cfg.scanner, seed=cfg.seed + round_index, log=log)
        if not extra:
            break
        counters["refine_rounds"] = round_index + 1
        counters["refine_candidates"] += len(extra)
        coverage = merge(coverage,
                         evaluate(extra, target.samples, station, model, cfg,
                                  visible=gate, log=log))
        if builder is not None:
            # Re-grow on the enlarged pool: the design's refinement is not "add poses",
            # it is "the candidate generator was not good enough here".
            stroke_cov = merge_stroke_coverage(stroke_cov, builder.evaluate(
                builder.generate(target.samples.points[leftover],
                                 target.samples.normals[leftover],
                                 aim_grid_mm=cfg.scanner.refine_aim_grid_mm,
                                 seed_offset=len(coverage.poses), log=log), log=log))
            combined = combine(coverage, stroke_cov)
            offered = _group_selection(combined, coverage, stroke_cov, cfg, log=log)
        else:
            combined = coverage
            offered = select(coverage, coverage.area_mm2, log=log)
        selection = _better(selection, offered)
        say(f"  pool ceiling {ceiling_before:.2f}% -> {combined.pool_reach_pct:.2f}%; "
            f"keeping {len(selection.chosen)} candidates at {selection.coverage_pct:.2f}% "
            f"(this round offered {len(offered.chosen)} at {offered.coverage_pct:.2f}%)")
        if combined.pool_reach_pct <= ceiling_before + 1e-9:
            # Another round would re-aim at the same samples with the same ladder, so it
            # cannot do better; the residue is geometry, not candidate generation.
            say("  refinement gained nothing: the remaining misses are not a candidate "
                "density problem")
            break

    path: PathPlan | None = None
    swept_pct: float | None = None
    if builder is not None and stroke_cov is not None:
        selection, path, swept_pct = _finish_stroke_plan(
            combined, coverage, stroke_cov, selection, pose_selection, target, station,
            model, cfg, gate, log=log)
        if path is None:
            combined = coverage
    counters.update({key: value for key, value in combined.counters.items()})

    if path is not None:
        viewpoints = viewpoints_from_path(path, combined, target.samples)
    else:
        viewpoints = viewpoints_from_selection(selection, combined, target.samples)
    say(f"reporting: {len(viewpoints)} points cover {selection.coverage_pct:.2f}%")
    report = build_report(config=cfg, model=model, station=station, split=split,
                          target=target, coverage=combined, selection=selection,
                          viewpoints=viewpoints, tracker=tracker, checks=checks,
                          path=path, strokes=stroke_cov,
                          swept_coverage_pct=swept_pct,
                          pose_only_coverage_pct=pose_selection.coverage_pct,
                          runtime={"seconds": round(time.time() - started, 1),
                                   "ray_backend": backend_name()},
                          log=log)
    report["counters"] = counters
    return PlanResult(config=cfg, station=station, split=split, target=target,
                      coverage=combined, selection=selection, viewpoints=viewpoints,
                      report=report, checks=checks, tracker=tracker, path=path,
                      strokes=stroke_cov, pose_only_coverage_pct=pose_selection.coverage_pct)


def _stroke_builder(cfg: PlannerConfig, target: TargetSet, station: Station,
                    model: ScannerModel, gate) -> StrokeBuilder | None:
    """The stroke builder, or None when stroke planning is switched off."""
    if not cfg.stroke.enabled:
        return None
    if cfg.stroke.max_stroke_mm <= 0.0:
        return None
    return StrokeBuilder(target.samples, station, model, cfg, visible=gate)


def _group_selection(combined: CoverageResult, coverage: CoverageResult,
                     stroke_cov: StrokeCoverage, cfg: PlannerConfig, *,
                     log=None) -> Selection:
    """Set cover over poses and strokes together, priced by travel plus one stop each."""
    stop = float(cfg.stroke.stop_cost_mm)
    n_poses = len(coverage.poses)
    costs = [stop] * n_poses + [stroke.length_mm + stop for stroke in stroke_cov.strokes]
    repair_ok = [True] * n_poses + [False] * len(stroke_cov.strokes)
    return select_groups(combined, costs, repair_ok=repair_ok, log=log)


def _finish_stroke_plan(combined: CoverageResult, coverage: CoverageResult,
                        stroke_cov: StrokeCoverage, selection: Selection,
                        pose_selection: Selection, target: TargetSet, station: Station,
                        model: ScannerModel, cfg: PlannerConfig, gate,
                        *, log=None):
    """Prune, order, connect - then keep the result only if it beats the pose-only plan.

    The comparison is the design's guarantee ("not worse than today"): coverage first,
    then fewer corners, then less travel. It is what stops a bad stroke pool from being
    handed over just because it was the new idea.
    """
    say = log or (lambda _message: None)
    n_poses = len(coverage.poses)
    sources = {int(row): str(source)
               for row, source in zip(selection.chosen, selection.sources, strict=True)}
    stroke_rows = [int(row) - n_poses for row in selection.chosen if row >= n_poses]
    pose_rows = [int(row) for row in selection.chosen if row < n_poses]
    kept = prune(stroke_cov.strokes, stroke_cov.sets, stroke_rows, combined.area_mm2,
                 combined.target_area_mm2, target_pct=PRUNE_TARGET_PCT,
                 max_rounds=cfg.stroke.max_prune_rounds)
    say(f"  pruning: {len(stroke_rows)} strokes -> {len(kept)} while staying at "
        f"{PRUNE_TARGET_PCT:g}%")
    if not kept:
        say("  no stroke survived pruning: falling back to the pose-only plan")
        return pose_selection, None, None
    rows = pose_rows + [n_poses + index for index in kept]
    rebuilt = _rebuild_selection(combined, rows, sources, selection)
    path = build_path(
        [stroke_cov.strokes[index] for index in kept], cfg=cfg, station=station,
        model=model, coverage_index=lambda index: n_poses + kept[index],
        sources=lambda index: sources.get(n_poses + kept[index], "greedy"), log=log)
    if _rank_plan(rebuilt.coverage_pct, path.turns, path.path_length_mm) < _rank_plan(
            pose_selection.coverage_pct, len(pose_selection.chosen), 0.0):
        say(f"  the stroke plan ({rebuilt.coverage_pct:.2f}%, {path.turns} corners) does "
            f"not beat the pose-only plan ({pose_selection.coverage_pct:.2f}%, "
            f"{len(pose_selection.chosen)} corners): delivering the poses")
        return pose_selection, None, None

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
            f"promised by the strokes alone")
    return rebuilt, path, swept_pct


def _rebuild_selection(combined: CoverageResult, rows, sources: dict[int, str],
                       previous: Selection) -> Selection:
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
        covered=covered, curve=list(previous.curve),
        coverage_area_mm2=float(area[covered].sum()),
        target_area_mm2=combined.target_area_mm2,
        sources=np.array(labels, dtype=object))


def _rank_plan(coverage_pct: float, turns: int, length_mm: float) -> tuple:
    """Coverage, then fewer corners, then less travel - the design's ordering."""
    return (round(coverage_pct, 6), -int(turns), -float(length_mm))


def _better(first: Selection, second: Selection) -> Selection:
    """Coverage first, then fewer poses - the design's ranking of the two criteria.

    Needed because the pool a later round selects over is larger, and greedy on a larger
    pool can need MORE poses for the same coverage. That is a worse deliverable, so a round
    that does not raise the pool's ceiling must not be allowed to re-roll the answer.
    """
    return second if _rank(second) > _rank(first) else first


def _rank(selection: Selection) -> tuple[float, int]:
    return (round(selection.coverage_pct, 6), -len(selection.chosen))


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
