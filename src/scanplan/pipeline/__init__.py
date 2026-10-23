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

The stages that change independently live beside this file: :mod:`refine` (the
re-aim-and-re-select rounds), :mod:`delivery` (turning a selection into an ordered
path) and :mod:`outputs` (the files of the deliverable). This module stays the
running order, so a change to one stage does not rewrite the others.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from ..candidates import generate
from ..config import PlannerConfig
from ..coverage import CoverageResult, evaluate
from ..kinematics import KinematicGate
from ..path import PathPlan
from ..raycast import backend_name
from ..report import (
    Viewpoint,
    build_report,
    viewpoints_from_path,
    viewpoints_from_selection,
)
from ..scanner_model import ScannerModel
from ..scene import Station
from ..selection import Selection, select
from ..sheet_split import SheetSplit, split_sheet
from ..strokes import StrokeCoverage, combine
from ..target_surface import TargetSet, build_target
from ..tracker import Tracker, visible_mask
from .delivery import _finish_stroke_plan, _group_selection, _stroke_builder
from .outputs import write_outputs
from .ranking import _rank
from .refine import RefineState, refine_rounds

__all__ = ["FACES", "PlanResult", "plan", "write_outputs"]

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


def _pose_tie(coverage: CoverageResult, kin) -> list[float] | None:
    """Per-pose joint room, or None when the gate is off.

    Used only to separate candidates whose covered area is equal: the choice of which of two
    equally good poses to deliver should prefer the one further from a limit, not the one
    that happened to be generated first.
    """
    if kin is None or not kin.active:
        return None
    values: list[float] = []
    for pose in coverage.poses:
        verdict = kin.check(pose.position, pose.axis, pose.roll_deg)
        values.append(verdict.solution.j5_margin_deg
                      if verdict.ok and verdict.solution is not None else -1e9)
    return values


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

    # The joint-limit gate. Off by default (`robot.enabled=false`), and when off every call
    # below is a no-op, which is what keeps the pre-2026-10-22 result reproducible.
    kin = KinematicGate(cfg.robot)
    if kin.active:
        say(f"kinematic gate on: {cfg.robot.model}, limits {kin.robot.limits_source}")

    say(f"generating candidates for the {face} face")
    poses = generate(target.samples.points, target.samples.normals, cfg.scanner,
                     seed=cfg.seed, log=log)
    mark = time.time()
    coverage = evaluate(poses, target.samples, station, model, cfg, visible=gate,
                        gate=kin, log=log)
    say(f"  pose coverage took {time.time() - mark:.0f} s")
    mark = time.time()

    pose_selection = select(coverage, coverage.area_mm2,
                            coverage_target_pct=cfg.target.coverage_target_pct,
                            tie=(_pose_tie(coverage, kin)), log=log)
    builder = _stroke_builder(cfg, target, station, model, gate, kin)
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

    state = RefineState(coverage=coverage, combined=combined, stroke_cov=stroke_cov,
                        selection=selection, selection_poses=selection_poses,
                        counters=counters)
    refine_rounds(state, builder, target, station, model, cfg, gate, kin=kin, log=log)
    coverage, combined = state.coverage, state.combined
    stroke_cov, selection = state.stroke_cov, state.selection
    counters = state.counters

    path: PathPlan | None = None
    swept_pct: float | None = None
    if builder is not None and stroke_cov is not None:
        selection, path, swept_pct = _finish_stroke_plan(
            combined, coverage, stroke_cov, selection, pose_selection, target, station,
            model, cfg, gate, kin=kin, log=log)
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
                          kinematics=kin,
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
