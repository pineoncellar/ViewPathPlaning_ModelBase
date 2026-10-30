"""The refinement stage: re-aim at what the pool cannot see, then re-select.

Extracted from the orchestration because it is the part that changes when candidate
generation changes - the measured finding was that the viewpoint count is limited by
candidate generation rather than by set cover.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..candidates import feature_refine, refine
from ..config import PlannerConfig
from ..coverage import CoverageResult, evaluate, merge
from ..scanner_model import ScannerModel
from ..scene import Station
from ..selection import Selection, feature_percent, select
from ..strokes import StrokeBuilder, StrokeCoverage, combine, merge_stroke_coverage
from ..target_surface import TargetSet
from .delivery import _group_selection, _joint_tie, _parallel_weights
from .ranking import _better


@dataclass
class RefineState:
    """What the refinement rounds rewrite, kept explicit instead of as locals."""

    coverage: CoverageResult
    combined: CoverageResult
    stroke_cov: StrokeCoverage | None
    selection: Selection
    selection_poses: int
    counters: dict


def refine_rounds(state: RefineState, builder: StrokeBuilder | None, target: TargetSet,
                  station: Station, model: ScannerModel, cfg: PlannerConfig, gate,
                  *, kin=None, log=None, progress=None) -> None:
    """Run up to `cfg.scanner.refine_rounds` rounds against `state`, in place."""
    say = log or (lambda _message: None)
    rounds = max(int(cfg.scanner.refine_rounds), 1)
    for round_index in range(cfg.scanner.refine_rounds):
        # Each round gets its own slice of the refine band, so a run that refines twice
        # still shows movement instead of sitting on one number until the report starts.
        round_rep = (progress.band("strokes", round_index / rounds,
                                   (round_index + 1) / rounds)
                     if progress is not None else None)
        if (cfg.target.coverage_target_pct is not None
                and state.selection.coverage_pct + 1e-9 >= cfg.target.coverage_target_pct):
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
        leftover = np.flatnonzero(~state.selection.covered)
        if len(leftover) == 0:
            break
        reachable = state.combined.pool_reach()
        say(f"refinement round {round_index + 1}: {len(leftover)} uncovered samples "
            f"({int((~reachable[leftover]).sum())} of them out of the pool's reach)")
        ceiling_before = state.combined.pool_reach_pct
        if round_rep is not None:
            round_rep(0.05, f"refinement round {round_index + 1}: aiming at the leaks")
        extra = refine(target.samples.points[leftover], target.samples.normals[leftover],
                       cfg.scanner, seed=cfg.seed + round_index, log=log,
                       lock_roll=bool(cfg.stroke.lock_roll),
                       keep=(lambda pose: kin.check(pose.position, pose.axis,
                                                    pose.roll_deg).ok)
                       if kin is not None and kin.active else None,
                       progress=round_rep.band("strokes", 0.0, 0.25).tick("aims")
                       if round_rep is not None else None)
        if not extra:
            break
        state.counters["refine_rounds"] = round_index + 1
        state.counters["refine_candidates"] += len(extra)
        if round_rep is not None:
            round_rep(0.3, f"refinement round {round_index + 1}: {len(extra)} new candidates")
        state.coverage = merge(
            state.coverage,
            evaluate(extra, target.samples, station, model, cfg, visible=gate,
                     gate=kin, feature=state.coverage.feature, log=log,
                     progress=round_rep.band("strokes", 0.25, 0.60).tick("poses")
                     if round_rep is not None else None))
        if builder is not None:
            # Re-grow on the enlarged pool: the design's refinement is not "add poses",
            # it is "the candidate generator was not good enough here".
            recheck = round_rep.band("strokes", 0.60, 0.95) if round_rep is not None else None
            if recheck is not None:
                recheck(0.0, f"refinement round {round_index + 1}: re-growing strokes")
            state.stroke_cov = merge_stroke_coverage(state.stroke_cov, builder.evaluate(
                builder.generate(target.samples.points[leftover],
                                 target.samples.normals[leftover],
                                 aim_grid_mm=cfg.scanner.refine_aim_grid_mm,
                                 seed_offset=len(state.coverage.poses), log=log,
                                 progress=recheck.band("strokes", 0.0, 0.5).tick("seeds")
                                 if recheck is not None else None),
                log=log,
                progress=recheck.band("strokes", 0.5, 1.0).tick("strokes")
                if recheck is not None else None))
            state.combined = combine(state.coverage, state.stroke_cov)
            # The refinement re-selects, so it has to price the same preference the first
            # selection did: the joint-room tie-break and the skeleton weight are part of
            # the objective, not of the first pass.
            offered = _group_selection(
                state.combined, state.coverage, state.stroke_cov, cfg,
                tie=_joint_tie(state.coverage, state.stroke_cov, kin),
                weight=_parallel_weights(state.coverage, state.stroke_cov, cfg), log=log)
        else:
            state.combined = state.coverage
            offered = select(state.coverage, state.coverage.area_mm2,
                             coverage_target_pct=cfg.target.coverage_target_pct,
                             feature_target_pct=cfg.target.feature_coverage_pct,
                             tie=_joint_tie(state.coverage, state.stroke_cov, kin)
                             if state.stroke_cov is not None else None, log=log)
        state.selection = _better(
            _shift_rows(state.selection, state.selection_poses, len(state.coverage.poses),
                        state.combined),
            offered, cfg.target.coverage_target_pct)
        state.selection_poses = len(state.coverage.poses)
        say(f"  pool ceiling {ceiling_before:.2f}% -> {state.combined.pool_reach_pct:.2f}%; "
            f"keeping {len(state.selection.chosen)} candidates at "
            f"{state.selection.coverage_pct:.2f}% "
            f"(this round offered {len(offered.chosen)} at {offered.coverage_pct:.2f}%)")
        if state.combined.pool_reach_pct <= ceiling_before + 1e-9:
            # Another round would re-aim at the same samples with the same ladder, so it
            # cannot do better; the residue is geometry, not candidate generation.
            say("  refinement gained nothing: the remaining misses are not a candidate "
                "density problem")
            break
        if round_rep is not None:
            round_rep(1.0, f"refinement round {round_index + 1} done: "
                           f"{state.selection.coverage_pct:.2f}%")
    if progress is not None:
        # Close the band even when every round was skipped: a bar that stops at the
        # selection and then jumps to the report looks like a stall that never happened.
        progress(1.0, "refinement settled")


def feature_round(state: RefineState, builder: StrokeBuilder | None, target: TargetSet,
                  station: Station, model: ScannerModel, cfg: PlannerConfig, gate,
                  *, kin=None, log=None, progress=None) -> None:
    """Cover the hard leftovers - walls, bends and the rim a clamp shades - with a tilted pass.

    Runs AFTER the ordinary refinement, and for a different reason. `refine_rounds` stops as
    soon as the AREA target is met, and that is the honest reading of a site target: the
    last percent of area costs a plan full of stops. A side wall is not that kind of
    leftover. It is a few tenths of a percent of a panel, so no area figure will ever ask
    for it, and a head-on ladder cannot reach it at all - it needs a different family of
    poses, which is what this pass adds.

    Nothing is invented here: the poses come from `candidates.feature_refine` (the same
    ladder, tilted almost to the incidence limit and swept around the normal), and the
    strokes from `StrokeBuilder.generate_feature`. Both are filtered by the joint gate
    before they are capped, both are evaluated by the same coverage code, and the result is
    re-selected on the same objective. If the arm cannot hold the tilted poses, the pass
    reports it and covers nothing - it does not bend the geometry to make a number.
    """
    say = log or (lambda _message: None)
    target_pct = cfg.target.feature_coverage_pct
    if target_pct is None or target.features is None or cfg.scanner.feature_rounds <= 0:
        return
    if not np.asarray(target.features, dtype=bool).any():
        return
    area = np.asarray(state.combined.area_mm2, dtype=float)
    for round_index in range(int(cfg.scanner.feature_rounds)):
        current = feature_percent(state.combined, area, state.selection.covered)
        if current is None:
            return
        mask = np.asarray(target.features, dtype=bool)
        left = np.flatnonzero(mask & ~state.selection.covered)
        say(f"  hard features: {current:.1f}% covered against a {target_pct:g}% target, "
            f"{len(left)} samples left")
        if len(left) == 0 or current + 1e-9 >= float(target_pct):
            return
        round_rep = (progress.band("strokes", round_index / max(cfg.scanner.feature_rounds, 1),
                                   (round_index + 1) / max(cfg.scanner.feature_rounds, 1))
                     if progress is not None else None)
        if round_rep is not None:
            round_rep(0.05, f"hard-feature pass {round_index + 1}: tilting at the leftovers")
        extra = feature_refine(target.samples.points[left], target.samples.normals[left],
                               cfg.scanner, seed=cfg.seed + 100 + round_index, log=log,
                               lock_roll=bool(cfg.stroke.lock_roll),
                               keep=(lambda pose: kin.check(pose.position, pose.axis,
                                                            pose.roll_deg).ok)
                               if kin is not None and kin.active else None,
                               progress=(round_rep.band("strokes", 0.05, 0.20).tick("aims")
                                         if round_rep is not None else None))
        if not extra:
            say("  hard features: the tilted ladder produced no candidate the arm can hold")
            return
        state.counters["feature_rounds"] = round_index + 1
        state.counters["feature_candidates"] = (state.counters.get("feature_candidates", 0)
                                                + len(extra))
        state.coverage = merge(
            state.coverage,
            evaluate(extra, target.samples, station, model, cfg, visible=gate,
                     gate=kin, feature=state.coverage.feature, log=log,
                     progress=(round_rep.band("strokes", 0.20, 0.55).tick("poses")
                               if round_rep is not None else None)))
        if builder is not None:
            strokes = builder.generate_feature(
                target.samples.points[left], target.samples.normals[left],
                spacing_mm=float(cfg.scanner.feature_aim_grid_mm) * 2.0,
                seed_offset=len(state.coverage.poses), log=log,
                progress=(round_rep.band("strokes", 0.55, 0.80).tick("seeds")
                          if round_rep is not None else None))
            state.stroke_cov = merge_stroke_coverage(
                state.stroke_cov, builder.evaluate(
                    strokes, log=log,
                    progress=(round_rep.band("strokes", 0.80, 0.95).tick("strokes")
                              if round_rep is not None else None)))
            state.combined = combine(state.coverage, state.stroke_cov)
            offered = _group_selection(
                state.combined, state.coverage, state.stroke_cov, cfg,
                tie=_joint_tie(state.coverage, state.stroke_cov, kin),
                weight=_parallel_weights(state.coverage, state.stroke_cov, cfg), log=log)
        else:
            state.combined = state.coverage
            offered = select(state.combined, state.combined.area_mm2,
                             coverage_target_pct=cfg.target.coverage_target_pct,
                             feature_target_pct=target_pct,
                             tie=_joint_tie(state.coverage, state.stroke_cov, kin)
                             if state.stroke_cov is not None else None, log=log)
        shifted = _shift_rows(state.selection, state.selection_poses,
                              len(state.coverage.poses), state.combined)
        state.selection = _better(shifted, offered, cfg.target.coverage_target_pct)
        state.selection_poses = len(state.coverage.poses)
        area = np.asarray(state.combined.area_mm2, dtype=float)
        after = feature_percent(state.combined, area, state.selection.covered)
        say(f"  hard-feature pass {round_index + 1}: {current:.1f}% -> "
            f"{0.0 if after is None else after:.1f}%, "
            f"{len(state.selection.chosen)} items at {state.selection.coverage_pct:.2f}%")
        if after is None or after <= current + 1e-9:
            # Another round would aim at the same samples with the same family, so it cannot
            # do better; what is left is geometry (or the arm), not candidate generation.
            say("  hard features: the pass gained nothing - what is left is not a tilt "
                "problem")
            return
    if progress is not None:
        progress(1.0, "hard features settled")


def _shift_rows(selection: Selection, old_poses: int, new_poses: int,
                combined: CoverageResult | None = None) -> Selection:
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

    # The covered MASK does not move with the rows - it is indexed by target sample - so the
    # hard-feature figure is recomputed rather than carried over. Carrying it would rank the
    # pre-round selection on the pre-round pool's numbers and could keep a worse plan.
    feature_pct = (selection.feature_coverage_pct if combined is None
                   else feature_percent(combined, combined.area_mm2, selection.covered))
    return Selection(chosen=shift(selection.chosen), greedy=shift(selection.greedy),
                     repair=shift(selection.repair), covered=selection.covered,
                     curve=list(selection.curve),
                     coverage_area_mm2=selection.coverage_area_mm2,
                     target_area_mm2=selection.target_area_mm2,
                     sources=selection.sources, corners=selection.corners,
                     feature_coverage_pct=feature_pct,
                     feature_target_pct=selection.feature_target_pct)
