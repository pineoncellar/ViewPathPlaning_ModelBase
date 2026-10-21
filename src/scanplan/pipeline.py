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

from .candidates import generate, refine
from .config import PlannerConfig
from .coverage import CoverageResult, evaluate, merge
from .raycast import backend_name
from .report import (
    Viewpoint,
    build_report,
    summary_text,
    viewpoints_from_selection,
    write_report,
    write_samples_csv,
    write_viewpoints_csv,
    write_viewpoints_json,
)
from .scanner_model import ScannerModel
from .scene import Station
from .selection import Selection, select
from .sheet_split import SheetSplit, split_sheet
from .target_surface import TargetSet, build_target
from .tracker import Tracker, visible_mask

FACES = ("upper", "lower", "auto")


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
    coverage = evaluate(poses, target.samples, station, model, cfg, visible=gate, log=log)
    counters = dict(coverage.counters)
    counters["refine_rounds"] = 0
    counters["refine_candidates"] = 0

    selection = select(
        coverage, coverage.area_mm2,
        max_viewpoints=cfg.scanner.max_viewpoints,
        log=log,
    )
    for round_index in range(cfg.scanner.refine_rounds):
        # The configured viewpoint budget may leave reachable samples uncovered;
        # refinement can still add candidates that cover more area per pose.
        leftover = np.flatnonzero(~selection.covered)
        if len(leftover) == 0:
            break
        reachable = coverage.pool_reach()
        say(f"refinement round {round_index + 1}: {len(leftover)} uncovered samples "
            f"({int((~reachable[leftover]).sum())} beyond the pool's reach)")
        ceiling_before = coverage.pool_reach_pct
        extra = refine(target.samples.points[leftover], target.samples.normals[leftover],
                       cfg.scanner, seed=cfg.seed + round_index, log=log)
        if not extra:
            break
        counters["refine_rounds"] = round_index + 1
        counters["refine_candidates"] += len(extra)
        coverage = merge(coverage,
                         evaluate(extra, target.samples, station, model, cfg,
                                  visible=gate, log=log))
        offered = select(
            coverage, coverage.area_mm2,
            max_viewpoints=cfg.scanner.max_viewpoints,
            log=log,
        )
        selection = _better(selection, offered)
        say(f"  pool ceiling {ceiling_before:.2f}% -> {coverage.pool_reach_pct:.2f}%; "
            f"keeping {len(selection.chosen)} poses at {selection.coverage_pct:.2f}% "
            f"(this round offered {len(offered.chosen)} at {offered.coverage_pct:.2f}%)")
        if coverage.pool_reach_pct <= ceiling_before + 1e-9:
            # Another round would re-aim at the same samples with the same ladder, so it
            # cannot do better; the residue is geometry, not candidate generation.
            say("  refinement gained nothing: the remaining misses are not a candidate "
                "density problem")
            break

    viewpoints = viewpoints_from_selection(selection, coverage, target.samples)
    say(f"reporting: {len(viewpoints)} viewpoints cover {selection.coverage_pct:.2f}%")
    report = build_report(config=cfg, model=model, station=station, split=split,
                          target=target, coverage=coverage, selection=selection,
                          viewpoints=viewpoints, tracker=tracker, checks=checks,
                          runtime={"seconds": round(time.time() - started, 1),
                                   "ray_backend": backend_name()},
                          log=log)
    report["counters"] = counters
    return PlanResult(config=cfg, station=station, split=split, target=target,
                      coverage=coverage, selection=selection, viewpoints=viewpoints,
                      report=report, checks=checks)


def _better(first: Selection, second: Selection) -> Selection:
    """Coverage first, then fewer poses - the design's ranking of the two criteria.

    Needed because the pool a later round selects over is larger, and greedy on a larger
    pool can need MORE poses for the same coverage. That is a worse deliverable, so a round
    that does not raise the pool's ceiling must not be allowed to re-roll the answer.
    """
    return second if _rank(second) > _rank(first) else first


def _rank(selection: Selection) -> tuple[float, int]:
    return (round(selection.coverage_pct, 6), -len(selection.chosen))


def write_outputs(result: PlanResult, out_dir) -> dict[str, Path]:
    """Write the viewpoint set, both point clouds and the report, and return the paths."""
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
    return paths


def _write_summary(folder: Path, report: dict) -> Path:
    path = Path(folder) / "summary.txt"
    path.write_text(summary_text(report), encoding="utf-8")
    return path
