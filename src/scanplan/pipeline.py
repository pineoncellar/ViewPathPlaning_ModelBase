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
from collections.abc import Callable, Sequence
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
from .strokes import (
    Stroke,
    StrokeBuilder,
    StrokeCoverage,
    combine,
    merge_stroke_coverage,
    pose_stroke,
)
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
        # The same key the plan is finally judged by: meeting the target first, then
        # fewer corners. Choosing the face by coverage alone and then ranking the result by
        # corners would let the two disagree about which side is better.
        best = max(candidates, key=lambda name: _rank(candidates[name].selection,
                                                      cfg.target.coverage_target_pct))
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

    pose_selection = select(coverage, coverage.area_mm2,
                            coverage_target_pct=cfg.target.coverage_target_pct, log=log)
    builder = _stroke_builder(cfg, target, station, model, gate)
    if builder is None:
        say("  stroke planning is off: the delivered set is single poses")
    stroke_cov: StrokeCoverage | None = None
    combined = coverage
    selection = pose_selection
    # The pose count of the pool `selection` was chosen from. `combine` puts every pose
    # before every stroke, so appending poses shifts each stroke's row - see `_shift_rows`.
    selection_poses = len(coverage.poses)
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
        selection_poses = len(coverage.poses)

    counters = dict(coverage.counters)
    counters["refine_rounds"] = 0
    counters["refine_candidates"] = 0

    for round_index in range(cfg.scanner.refine_rounds):
        if (cfg.target.coverage_target_pct is not None
                and selection.coverage_pct + 1e-9 >= cfg.target.coverage_target_pct):
            # Refinement re-aims at everything still uncovered, and at a target there is
            # always something uncovered - by design, that is the area the workshop patches
            # by hand. The extra candidates mostly buy coverage the target excludes, at a
            # third of the run's ray budget each. Use `--coverage-target none` to get the
            # old "keep aiming until nothing is left" behaviour back.
            say(f"  the {cfg.target.coverage_target_pct:g}% target is met: not refining")
            break
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
            offered = select(coverage, coverage.area_mm2,
                             coverage_target_pct=cfg.target.coverage_target_pct, log=log)
        selection = _better(_shift_rows(selection, selection_poses, len(coverage.poses)),
                            offered, cfg.target.coverage_target_pct)
        selection_poses = len(coverage.poses)
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


def _shift_rows(selection: Selection, old_poses: int, new_poses: int) -> Selection:
    """Rewrite a selection's stroke rows after more poses were appended to the pool.

    `combine` puts every pose before every stroke, so adding `offset` poses moves stroke `i`
    from row `old_poses + i` to `new_poses + i`. Without this, a selection made before a
    refinement round silently points at different candidates afterwards: the measured run
    printed a mixed plan of 8 items covering 97.15% and then delivered "8 items, 0 strokes,
    33.22%", because the same row numbers now named eight poses.
    """
    offset = int(new_poses) - int(old_poses)
    if offset == 0:
        return selection

    def shift(values):
        return [int(row) if int(row) < old_poses else int(row) + offset for row in values]

    return Selection(chosen=shift(selection.chosen), greedy=shift(selection.greedy),
                     repair=shift(selection.repair), covered=selection.covered,
                     curve=list(selection.curve),
                     coverage_area_mm2=selection.coverage_area_mm2,
                     target_area_mm2=selection.target_area_mm2,
                     sources=selection.sources, corners=selection.corners)


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


def _rank_plan(coverage_pct: float, corners: int, length_mm: float,
               target_pct: float | None) -> tuple:
    """Met the target first, then fewer corners, then less travel.

    The first component is a two-way split, not a coverage ranking: once a plan meets the
    acceptance figure, more coverage buys nothing the workshop asked for, so it must not
    outrank a plan with fewer corners. Below the target the old order applies - whoever
    covers more wins - which is what keeps `coverage_target_pct = null` behaving exactly
    like v1.
    """
    if target_pct is None:
        return (round(float(coverage_pct), 6), -int(corners), -float(length_mm))
    if float(coverage_pct) + 1e-9 >= float(target_pct):
        return (1, -int(corners), -float(length_mm))
    return (0, round(float(coverage_pct), 6), -int(corners), -float(length_mm))


def _better(first: Selection, second: Selection,
            target_pct: float | None = None) -> Selection:
    """Met the target first, then fewer corners - the design's ranking of the two criteria.

    Needed because the pool a later round selects over is larger, and greedy on a larger
    pool can need MORE poses for the same coverage. That is a worse deliverable, so a round
    that does not raise the pool's ceiling must not be allowed to re-roll the answer.
    """
    return second if _rank(second, target_pct) > _rank(first, target_pct) else first


def _rank(selection: Selection, target_pct: float | None = None) -> tuple:
    """The ranking key of a selection, in the same form `_rank_plan` produces."""
    return _rank_plan(selection.coverage_pct, selection.cost_corners, 0.0, target_pct)


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
