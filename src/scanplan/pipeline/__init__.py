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

Choosing which face of the blank to scan lives here too. `face` ("upper" / "lower" /
"auto") is a GEOMETRIC label pair, not a direction: the split orients its sheet
normal from a through-thickness axis whose sign is arbitrary, so the side the
operator pointed at can carry either name. `face_axis` is the operator's main
scan-face axis in the station frame and decides the side whenever it is given;
"auto" (plan both faces, keep the better one) is the fallback for a caller that
has no axis.

The stages that change independently live beside this file: :mod:`refine` (the
re-aim-and-re-select rounds), :mod:`delivery` (turning a selection into an ordered
path) and :mod:`outputs` (the files of the deliverable). This module stays the
running order, so a change to one stage does not rewrite the others.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

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
from .progress import ProgressHook, ProgressReporter
from .ranking import _rank
from .refine import RefineState, feature_round, refine_rounds

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


def facing_face(part, split: SheetSplit, axis) -> tuple[str, dict[str, float]]:
    """Which of the split's two faces looks along `axis`, with both projection scores.

    `upper` / `lower` are geometric names only: the split orients its mid-surface normal
    from a through-thickness axis whose SIGN is arbitrary, so on a real station the side
    the operator picked can carry either name. Station 2026-10-29 is the measured failure:
    the picked 1.15 m2 face was called `lower`, and hard-coding `face="upper"` planned a
    113 cm2 spray of back-facing facets spread over the whole part instead.

    The score is the SIGNED projected area of each face on the plane square to `axis`. The
    side the scanner can see has the larger positive projection; forming side walls
    included, because their projections cancel within one face rather than flipping it.
    """
    normals = np.asarray(part.face_normals, dtype=float)
    areas = np.asarray(part.area_faces, dtype=float)
    direction = np.asarray(axis, dtype=float).reshape(3)
    length = float(np.linalg.norm(direction))
    if length <= 1e-9:
        raise ValueError("the main scan-face axis must be a non-zero direction")
    direction = direction / length
    scores = {
        name: float((areas[mask] * (normals[mask] @ direction)).sum())
        for name, mask in (("upper", split.upper), ("lower", split.lower))
    }
    return ("upper" if scores["upper"] >= scores["lower"] else "lower"), scores


def plan(station: Station, cfg: PlannerConfig, *, face: str | None = None,
         face_axis=None, tracker: Tracker | None = None,
         split: SheetSplit | None = None,
         log: Callable[[str], None] | None = None,
         progress: ProgressHook | None = None) -> PlanResult:
    """Plan the scan of one face of the blank, and report what was missed.

    `face` defaults to the configuration; `"auto"` plans both faces and keeps the one
    with the higher coverage, and records both figures in `face_candidates`.

    `face_axis` is the main scan-face direction in the station frame (the vehicle axis the
    operator picked, carried into the measured frame). When it is given it decides which
    side of the blank is the target and the `upper` / `lower` words are only reported, not
    obeyed - see `facing_face`. "auto" is what a caller without an axis should use.

    `progress(stage, fraction, detail)` is optional and reports where the run actually
    is. The bands below are the run's real stages, not a time estimate; a caller that
    ignores it sees exactly the same planning as before.
    """
    say = log or (lambda message: print(message, flush=True))
    reporter = ProgressReporter(progress)
    started = time.time()
    requested = cfg.target.face if face is None else face
    if requested not in FACES:
        raise ValueError(f"face must be one of {FACES}, not {requested!r}")

    alignment_rep = reporter.band("segmentation", 0.0, 0.03)
    say("station alignment checks")
    # A diagnostic, not a measurement: 40k samples per model is ample for the distances
    # and clearances it reports, and it runs on every plan including the two-face "auto"
    # run, so the default of 200k would dominate the runtime for no extra information.
    checks = station.check_alignment(seed=cfg.seed, samples=40_000)
    for warning in checks["warnings"]:
        say(f"  WARNING: {warning}")
    alignment_rep(1.0, f"alignment checks: {len(checks['warnings'])} warnings")

    if split is None:
        split_rep = reporter.band("segmentation", 0.03, 0.10)
        say("splitting the blank into its two faces")
        split_rep(0.05, "splitting the blank")
        split = split_sheet(station.part, band_deg=cfg.target.band_deg, knn=cfg.target.knn,
                            seed=cfg.seed, log=log)
        split_rep(1.0, "blank split into its two faces")

    selection_record: dict | None = None
    if face_axis is not None:
        # The operator's axis is a real direction; the split's labels are not. Resolving
        # the side here keeps the split itself untouched (its seed sign is what the
        # coverage of this station was tuned on) and only changes WHICH of the two faces
        # `_plan_face` is asked for.
        chosen, scores = facing_face(station.part, split, face_axis)
        direction = np.asarray(face_axis, dtype=float).reshape(3)
        direction = direction / np.linalg.norm(direction)
        say("main scan face along "
            f"[{direction[0]:.3f}, {direction[1]:.3f}, {direction[2]:.3f}] -> the target is "
            f"the '{chosen}' face (projected area: upper "
            f"{scores['upper'] / 1e6:.4f} m2, lower {scores['lower'] / 1e6:.4f} m2)")
        if requested not in ("auto", chosen):
            say(f"  the requested '{requested}' face is the FAR side of the blank: it is "
                "the side facing away from the operator's main scan face, so it is not "
                "planned")
        selection_record = {
            "method": "signed projected area of each split face on the plane square to "
                      "the main scan-face axis",
            "face_axis": [round(float(v), 6) for v in direction],
            "requested": requested,
            "chosen": chosen,
            "projected_area_mm2": {name: round(value, 1)
                                   for name, value in scores.items()},
            "note": ("upper/lower are arbitrary labels on a real station; the axis decides "
                     "which side is planned"),
        }
        requested = chosen

    if requested == "auto":
        candidates: dict[str, PlanResult] = {}
        for option, low, high in (("upper", 0.10, 0.55), ("lower", 0.55, 1.0)):
            face_rep = reporter.band("segmentation", low, high)
            face_rep(0.0, f"planning the {option} face")
            try:
                candidates[option] = _plan_face(station, cfg, split, option, tracker,
                                               checks, started, log, progress=face_rep)
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
        if selection_record is not None:
            result.report["target_face_selection"] = selection_record
        if station.winding:
            result.report["part_winding"] = station.winding
        return result

    face_rep = reporter.band("segmentation", 0.10, 1.0)
    face_rep(0.0, f"planning the {requested} face")
    result = _plan_face(station, cfg, split, requested, tracker, checks, started, log,
                        progress=face_rep)
    if selection_record is not None:
        result.report["target_face_selection"] = selection_record
    if station.winding:
        result.report["part_winding"] = station.winding
    return result


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


def _joint_keep(kin):
    """The joint gate as a predicate for candidate generation, or None when it is off.

    Passed into `generate`/`refine` so the pool is filtered BEFORE it is capped: the cap is
    far below the natural pool, so filtering afterwards spends the ray budget on poses the
    arm can never hold (786 of 4000 on the 2026-10-28 delivery).
    """
    if kin is None or not kin.active:
        return None

    def keep(pose) -> bool:
        return bool(kin.check(pose.position, pose.axis, pose.roll_deg).ok)

    return keep


def _plan_face(station: Station, cfg: PlannerConfig, split: SheetSplit, face: str,
               tracker: Tracker | None, checks: dict, started: float,
               log: Callable[[str], None] | None,
               progress: ProgressReporter | None = None) -> PlanResult:
    """Plan one named face, from target sampling through to the report."""
    say = log or (lambda message: print(message, flush=True))
    base = progress if progress is not None else ProgressReporter()
    model = ScannerModel(cfg.scanner)
    target_rep = base.band("segmentation", 0.0, 0.10)
    target_rep(0.02, f"sampling the {face} target face")
    target = build_target(station.part, split, face, station, cfg, log=log)
    if len(target.samples.points) == 0:
        raise ValueError(f"the {face} target face produced no samples to scan")
    target_rep(1.0, f"{len(target.samples.points)} target samples")

    gate = None
    if tracker is not None:
        obstacles = station.obstacles()
        gate = lambda positions: visible_mask(tracker, positions, obstacles)  # noqa: E731

    # The joint-limit gate. ON in the shipped configuration, because a program whose points
    # the arm cannot hold is not a delivery - and the .LS export refuses such a point rather
    # than writing it. `robot.enabled=false` makes every call below a no-op, which is how a
    # pre-2026-10-27 result can still be reproduced.
    kin = KinematicGate(cfg.robot)
    if kin.active:
        say(f"kinematic gate on: {cfg.robot.model}, limits {kin.robot.limits_source}")
    else:
        say("WARNING: the joint gate is off - the delivered points are NOT checked against "
            "the arm's limits, and the .LS export will refuse to verify them")

    cand_rep = base.band("candidates", 0.10, 0.22)
    say(f"generating candidates for the {face} face")
    cand_rep(0.0, "aiming the candidate ladder")
    poses = generate(target.samples.points, target.samples.normals, cfg.scanner,
                     seed=cfg.seed, log=log, progress=cand_rep.tick("aims"),
                     lock_roll=bool(cfg.stroke.lock_roll), keep=_joint_keep(kin))
    cand_rep(1.0, f"{len(poses)} candidates")

    occ_rep = base.band("occlusion", 0.22, 0.60)
    mark = time.time()
    coverage = evaluate(poses, target.samples, station, model, cfg, visible=gate,
                        gate=kin, feature=target.features, log=log,
                        progress=occ_rep.tick("poses"))
    occ_rep(1.0, f"{len(coverage.poses)} poses in {time.time() - mark:.0f} s")
    say(f"  pose coverage took {time.time() - mark:.0f} s")
    mark = time.time()

    select_rep = base.band("occlusion", 0.60, 0.64)
    select_rep(0.2, "selecting the minimum covering set")
    pose_selection = select(coverage, coverage.area_mm2,
                            coverage_target_pct=cfg.target.coverage_target_pct,
                            feature_target_pct=cfg.target.feature_coverage_pct,
                            tie=(_pose_tie(coverage, kin)), log=log)
    select_rep(1.0, f"{len(pose_selection.chosen)} poses at "
                    f"{pose_selection.coverage_pct:.2f}%")
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
        stroke_gen_rep = base.band("strokes", 0.64, 0.74)
        say("growing scan strokes from the aim grid")
        stroke_gen_rep(0.0, "growing strokes from the aim grid")
        grown = builder.generate(target.samples.points, target.samples.normals,
                                 aim_grid_mm=cfg.scanner.aim_grid_mm, log=log,
                                 progress=stroke_gen_rep.tick("seeds"))
        stroke_gen_rep(1.0, f"{len(grown)} strokes generated")
        stroke_eval_rep = base.band("strokes", 0.74, 0.86)
        stroke_cov = builder.evaluate(grown, log=log,
                                      progress=stroke_eval_rep.tick("strokes"))
        stroke_eval_rep(1.0, f"{len(stroke_cov.strokes)} strokes kept")
        group_rep = base.band("strokes", 0.86, 0.90)
        group_rep(0.2, "grouping poses and strokes")
        combined = combine(coverage, stroke_cov)
        selection = _group_selection(combined, coverage, stroke_cov, cfg, log=log)
        group_rep(1.0, f"{len(selection.chosen)} items at {selection.coverage_pct:.2f}%")
        say(f"  stroke stage took {time.time() - mark:.0f} s")
        say(f"  pose-only reference: {len(pose_selection.chosen)} viewpoints at "
            f"{pose_selection.coverage_pct:.2f}%; strokes reach "
            f"{len(selection.chosen)} items at {selection.coverage_pct:.2f}%")
        selection_poses = len(coverage.poses)
        refine_low, refine_high = 0.90, 0.94

    if builder is None:
        # Pose-only delivery: the refine rounds are the rest of the work before the report.
        refine_low, refine_high = 0.64, 0.94

    counters = dict(coverage.counters)
    counters["refine_rounds"] = 0
    counters["refine_candidates"] = 0
    counters["feature_rounds"] = 0
    counters["feature_candidates"] = 0

    state = RefineState(coverage=coverage, combined=combined, stroke_cov=stroke_cov,
                        selection=selection, selection_poses=selection_poses,
                        counters=counters)
    # `say`, not `log`: the stage's own narration goes through the same writer the rest of
    # this function uses, so a caller that passed no logger still sees what the refinement
    # decided. Passing the raw argument silently dropped every line from both stages.
    refine_rounds(state, builder, target, station, model, cfg, gate, kin=kin, log=say,
                  progress=base.band("strokes", refine_low, refine_high - 0.02))
    feature_round(state, builder, target, station, model, cfg, gate, kin=kin, log=say,
                  progress=base.band("strokes", refine_high - 0.02, refine_high))
    coverage, combined = state.coverage, state.combined
    stroke_cov, selection = state.stroke_cov, state.selection
    counters = state.counters

    path: PathPlan | None = None
    swept_pct: float | None = None
    if builder is not None and stroke_cov is not None:
        finish_rep = base.band("strokes", 0.94, 0.97)
        finish_rep(0.05, "building the delivered path")
        selection, path, swept_pct = _finish_stroke_plan(
            combined, coverage, stroke_cov, selection, pose_selection, target, station,
            model, cfg, gate, kin=kin, log=say, progress=finish_rep)
        finish_rep(1.0, "delivered path built")
    counters.update({key: value for key, value in combined.counters.items()})

    if path is not None:
        viewpoints = viewpoints_from_path(path, combined, target.samples)
    else:
        viewpoints = viewpoints_from_selection(selection, combined, target.samples)
    say(f"reporting: {len(viewpoints)} points cover {selection.coverage_pct:.2f}%")
    report_rep = base.band("report", 0.97, 1.0)
    report_rep(0.05, "writing the coverage report")
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
    report_rep(1.0, "report ready")
    return PlanResult(config=cfg, station=station, split=split, target=target,
                      coverage=combined, selection=selection, viewpoints=viewpoints,
                      report=report, checks=checks, tracker=tracker, path=path,
                      strokes=stroke_cov, pose_only_coverage_pct=pose_selection.coverage_pct)
