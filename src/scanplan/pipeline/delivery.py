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
from ..geometry import unit
from ..path import PathPlan, assign_branches, build_path, prune, transition_poses
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
                    model: ScannerModel, gate, kin=None) -> StrokeBuilder | None:
    """The stroke builder, or None when the delivered set is single poses."""
    if cfg.target.delivery_mode == "lean":
        # The lean deliverable is v1 plus a stopping rule: poses alone, no connectors, and
        # therefore no claim about the motion between them. Strokes cannot be part of it.
        return None
    if not cfg.stroke.enabled:
        return None
    if cfg.stroke.max_stroke_mm <= 0.0:
        return None
    return StrokeBuilder(target.samples, station, model, cfg, visible=gate, gate=kin)


def _group_selection(combined: CoverageResult, coverage: CoverageResult,
                     stroke_cov: StrokeCoverage, cfg: PlannerConfig, *,
                     tie=None, weight=None, lam: float = 0.0, exclude=None,
                     log=None) -> Selection:
    """Set cover over poses and strokes together, priced in the configured unit.

    `corners` is the default because it is what the workshop counts: one stop for a pose,
    two for a stroke, plus the transition each one drags along. `millimetres` is the
    upstream design's pricing and is kept so the two can be compared on the same part.

    `lam` raises the cost floor, which is what turns "fit the coverage into a point budget"
    into a search over one number instead of a re-run at a coarser resolution. `tie` breaks
    exact area ties by joint room, and `weight` is the skeleton preference - both are None
    unless the caller has a reason, so the default path is unchanged.
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
                         coverage_target_pct=cfg.target.coverage_target_pct,
                         tie=tie, weight=weight, lam=lam, exclude=exclude, log=log)


def _exclusion_mask(combined: CoverageResult, coverage: CoverageResult,
                    stroke_cov: StrokeCoverage, forbidden) -> np.ndarray:
    """A candidate mask with `forbidden` rows set, for the re-selection after a drop."""
    mask = np.zeros(combined.matrix().shape[0], dtype=bool)
    for row in forbidden:
        index = int(row)
        if 0 <= index < len(mask):
            mask[index] = True
    return mask


def _skeleton_axis(strokes) -> np.ndarray | None:
    """The direction the long parallel bands run along, or None when there is no majority."""
    if not strokes:
        return None
    directions = [unit(stroke.unit_direction) * max(stroke.length_mm, 1.0)
                  for stroke in strokes if stroke.length_mm > 0.0]
    if not directions:
        return None
    mean = np.sum(directions, axis=0)
    length = float(np.linalg.norm(mean))
    return mean / length if length > 1e-6 else None


def _parallel_weights(coverage: CoverageResult, stroke_cov: StrokeCoverage,
                      cfg: PlannerConfig):
    """Per-candidate skeleton preference, or None when the knob is neutral.

    The weight is a tie-break, not a policy: it may nudge the score by up to the configured
    30 percent, and it is never applied to the repair pass - the last percent of coverage
    comes from bands in other directions, and a skeleton preference that overrules them
    reaches 95 percent and then cannot patch what it skipped.
    """
    factor = float(cfg.stroke.parallel_weight)
    if abs(factor - 1.0) < 1e-12:
        return None
    axis = _skeleton_axis(stroke_cov.strokes)
    if axis is None:
        return None
    weights = [1.0] * len(coverage.poses)
    for stroke in stroke_cov.strokes:
        aligned = (stroke.length_mm > 0.0
                   and abs(float(np.dot(unit(stroke.unit_direction), axis))) > 0.9)
        weights.append(float(factor) if aligned else 1.0)
    return weights


def _joint_tie(coverage: CoverageResult, stroke_cov: StrokeCoverage, kin):
    """Per-candidate J5 room, or None when the kinematic gate is off.

    A stroke is scored by its WORSE end: the band is only as safe as the point along it
    that sits closest to a limit, and that point may well be the far end.
    """
    if kin is None or not kin.active:
        return None
    values: list[float] = []
    for pose in coverage.poses:
        verdict = kin.check(pose.position, pose.axis, pose.roll_deg)
        values.append(verdict.solution.j5_margin_deg
                      if verdict.ok and verdict.solution is not None else -1e9)
    for stroke in stroke_cov.strokes:
        ends = [kin.check(point, stroke.axis, stroke.roll_deg)
                for point in (stroke.start, stroke.end)]
        margins = [item.solution.j5_margin_deg for item in ends
                   if item.ok and item.solution is not None]
        values.append(min(margins) if len(margins) == 2 else -1e9)
    return values


def _one_delivery(combined: CoverageResult, rows: Sequence[int], sources: dict[int, str],
                  stroke_cov: StrokeCoverage, coverage: CoverageResult, cfg: PlannerConfig,
                  station: Station, model: ScannerModel, gate, *,
                  strict: bool | None = None, kin=None,
                  log=None) -> tuple[Selection, PathPlan, list[int]]:
    """Prune and order an explicit set of chosen rows into a delivered path.

    Poses go in as zero-length strokes. That is not a trick: the module documentation of
    `strokes` already says a stroke of length zero is a viewpoint, and delivering them this
    way is what keeps the reported coverage and the program's actual sweep the same number.
    Pruning sees every row, so a pose the target does not need can go too.

    Returns the rebuilt selection, the path, and the row each surviving item came from -
    the last one is what lets the caller map `path.dropped` back to selection rows.
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
    path = build_path([items[index] for index in kept], cfg=cfg, station=station,
                      model=model, coverage_index=lambda index: kept_rows[index],
                      sources=lambda index: sources.get(kept_rows[index], "greedy"),
                      visible=gate, strict=strict, kin=kin, log=log)
    # The selection is rebuilt from what the PATH actually delivers, not from what was
    # selected: an item the hard gate refused is not in the program, so counting its area
    # would make the headline figure describe a sweep that never happens. That is the same
    # defect the stroke accounting was introduced to remove, one layer down.
    delivered = [row for index, row in enumerate(kept_rows) if index not in set(path.dropped)]
    rebuilt = _rebuild_selection(combined, delivered, sources, None)
    return rebuilt, path, kept_rows


def _delivery(combined: CoverageResult, rows: Sequence[int], sources: dict[int, str],
              stroke_cov: StrokeCoverage, coverage: CoverageResult, cfg: PlannerConfig,
              station: Station, model: ScannerModel, gate, *,
              kin=None, strict: bool | None = None, rounds: int = 3,
              reselect=None, log=None) -> tuple[Selection, PathPlan]:
    """Deliver `rows`, removing any item the hard gate refuses and re-running on the rest.

    The retry is the point. "Legality is a hard constraint" cannot mean "write a violation
    and deliver the program anyway"; it has to mean "that sweep is not part of the answer",
    and then the selection has to be told, because dropping one item can make another
    redundant or reveal that the coverage target is no longer met.

    `reselect(forbidden)` re-runs the selection with those rows banned. Falling back on
    simply filtering the list is wrong and visibly so: one 800 mm band carries several
    percent of the part, and dropping it without a replacement loses that area rather than
    re-covering it - on the measured part the difference was 82.7% against 99.0%.

    Three rounds is a cap, not a target: each round is a full ordering pass, and a plan that
    needs more than three is better reported than silently ground down.
    """
    say = log or (lambda _message: None)
    hard = cfg.stroke.hard_gate if strict is None else bool(strict)
    current = list(rows)
    current_sources = dict(sources)
    forbidden: set[int] = set()
    rejected: list[int] = []
    rebuilt = _rebuild_selection(combined, current, current_sources, None)
    path = PathPlan(points=[], ordering="empty")
    for _round in range(max(1, int(rounds))):
        rebuilt, path, kept_rows = _one_delivery(
            combined, current, current_sources, stroke_cov, coverage, cfg, station, model,
            gate, strict=hard, kin=kin, log=log)
        if not path.dropped:
            break
        bad = {kept_rows[index] for index in path.dropped if index < len(kept_rows)}
        if not bad:
            break
        rejected.extend(sorted(bad))
        forbidden |= bad
        if reselect is not None:
            current, current_sources = reselect(frozenset(forbidden))
        else:
            current = [row for row in current if row not in bad]
        if not current:
            break
        say(f"  hard gate: dropping {len(bad)} item(s) and re-delivering "
            f"{len(current)}")
    if rejected:
        path.violations.append(
            f"the hard gate removed {len(rejected)} item(s) with no legal connector; "
            "the selection was re-run without them")
    if kin is not None and kin.active and path.points:
        assignment = assign_branches(path.points, kin, float(cfg.robot.max_step_deg))
        path.branches = assignment
        path.max_step_deg = assignment.max_step_deg
        if assignment.failed:
            path.violations.append(
                f"{len(assignment.failed)} delivered point(s) have no joint solution at all")
        if assignment.broken:
            path.violations.append(
                f"{len(assignment.broken)} joint step(s) exceed max_step_deg "
                f"({assignment.max_step_deg:.1f} > {cfg.robot.max_step_deg:g} deg)")
    return rebuilt, path


def _finish_stroke_plan(combined: CoverageResult, coverage: CoverageResult,
                        stroke_cov: StrokeCoverage, selection: Selection,
                        pose_selection: Selection, target: TargetSet, station: Station,
                        model: ScannerModel, cfg: PlannerConfig, gate,
                        *, kin=None, log=None):
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

    With `target.max_points` set, the mixed plan is rebuilt for a growing lambda and the
    largest one that still fits the budget is kept. Delivered points are counted, not
    estimated: connectors decide on their own how many corners a transfer needs, so any
    estimate would be a number the program then contradicts.
    """
    say = log or (lambda _message: None)
    target_pct = cfg.target.coverage_target_pct
    budget = cfg.target.max_points
    sources = {int(row): str(source)
               for row, source in zip(selection.chosen, selection.sources, strict=True)}
    pose_sources = {int(row): str(source) for row, source
                    in zip(pose_selection.chosen, pose_selection.sources, strict=True)}

    tie = _joint_tie(coverage, stroke_cov, kin)
    weight = _parallel_weights(coverage, stroke_cov, cfg)

    best = None
    verdict = ""
    lam_used = 0.0
    history: list[tuple[float, int, float]] = []
    for lam in (BUDGET_LAMBDAS if budget is not None else (0.0,)):
        offered = _group_selection(combined, coverage, stroke_cov, cfg, tie=tie,
                                   weight=weight, lam=lam, log=log)
        chosen_sources = dict(sources)
        chosen_sources.update({int(row): str(label) for row, label
                               in zip(offered.chosen, offered.sources, strict=True)})

        def reselect(forbidden, lam=lam):
            """Re-choose the same problem with the gated items banned."""
            mask = _exclusion_mask(combined, coverage, stroke_cov, forbidden)
            again = _group_selection(combined, coverage, stroke_cov, cfg, tie=tie,
                                     weight=weight, lam=lam, exclude=mask, log=log)
            return again.chosen, {int(row): str(label) for row, label
                                  in zip(again.chosen, again.sources, strict=True)}

        mixed_sel, mixed_path = _delivery(combined, offered.chosen, chosen_sources,
                                          stroke_cov, coverage, cfg, station, model, gate,
                                          kin=kin, reselect=reselect, log=log)
        mixed_sel.corners = mixed_path.turns
        history.append((float(lam), int(mixed_path.turns), float(mixed_sel.coverage_pct)))
        fits = budget is None or mixed_path.turns <= int(budget)
        if fits and (best is None
                     or (mixed_sel.coverage_pct, -mixed_path.turns)
                     > (best[0].coverage_pct, -best[1].turns)):
            best = (mixed_sel, mixed_path)
            lam_used = float(lam)
        if budget is not None and not fits:
            # More lambda means more corners; the first overflow bounds the search.
            break

    pose_sel, pose_path = _delivery(combined, pose_selection.chosen, pose_sources,
                                    stroke_cov, coverage, cfg, station, model, gate,
                                    kin=kin, log=log)
    pose_sel.corners = pose_path.turns
    if budget is not None and pose_path.turns > int(budget):
        pose_sel, pose_path = None, None

    if best is None:
        # Even the cheapest lambda overflows: this pool has no solution that fits. Drop the
        # worst value-per-point items until it does, and say so - the alternative is a
        # program over the controller's limit, which is not a deliverable.
        best = _shed_to_budget(combined, coverage, stroke_cov, cfg, station, model, gate,
                               kin=kin, tie=tie, weight=weight, log=log)
        verdict = "budget_bound"
    elif budget is not None:
        met = (target_pct is not None
               and best[0].coverage_pct + 1e-9 >= float(target_pct))
        verdict = "target_bound" if met else (
            "budget_bound" if lam_used > 0.0 else "exhausted")

    mixed_sel, mixed_path = best
    mixed_sel.corners = mixed_path.turns
    mixed_path.budget_history = history
    mixed_path.verdict = verdict
    say(f"  budget: verdict {verdict or 'unbudgeted'}, {mixed_path.turns} corners at "
        f"lambda {lam_used:g}, {len(history)} trial(s)")

    mixed_rank = _rank_plan(mixed_sel.coverage_pct, mixed_path.turns,
                            mixed_path.path_length_mm, target_pct)
    pose_rank = (_rank_plan(pose_sel.coverage_pct, pose_path.turns,
                            pose_path.path_length_mm, target_pct)
                 if pose_sel is not None else None)
    if pose_rank is not None and pose_rank >= mixed_rank:
        say(f"  the mixed plan ({mixed_sel.coverage_pct:.2f}%, {mixed_path.turns} corners, "
            f"{mixed_path.path_length_mm:.0f} mm) does not beat the pose-only plan "
            f"({pose_sel.coverage_pct:.2f}%, {pose_path.turns} corners, "
            f"{pose_path.path_length_mm:.0f} mm): delivering the poses")
        pose_path.verdict = verdict
        pose_path.budget_history = history
        return pose_sel, pose_path, _swept_pct(pose_sel, pose_path, target, station, model,
                                               cfg, combined, gate, log=log)
    say(f"  the mixed plan wins: {mixed_sel.coverage_pct:.2f}%, {mixed_path.turns} corners "
        f"against the pose-only {pose_sel.coverage_pct:.2f}%, "
        f"{pose_path.turns if pose_path is not None else 0}")
    return mixed_sel, mixed_path, _swept_pct(mixed_sel, mixed_path, target, station, model,
                                             cfg, combined, gate, log=log)


# Lambda samples for the budget search, in corner units. The scale is the cost scale, not
# the area scale: the score is gain / (cost + lambda), so lambda only has to become large
# next to a corner count (1 to 3) for the cost term to stop mattering at all.
BUDGET_LAMBDAS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)


def _shed_to_budget(combined: CoverageResult, coverage: CoverageResult,
                    stroke_cov: StrokeCoverage, cfg: PlannerConfig, station: Station,
                    model: ScannerModel, gate, *, kin, tie, weight, log=None):
    """Drop the items with the worst covered-area-per-corner until the plan fits.

    The last resort, and it is a real one: with a pool that cannot be squeezed any other
    way, the choice is between a shorter program and a program the controller rejects. The
    report calls the outcome `budget_bound` and says how much coverage went with it.
    """
    say = log or (lambda _message: None)
    offered = _group_selection(combined, coverage, stroke_cov, cfg, tie=tie, weight=weight,
                               log=log)
    rows = list(offered.chosen)
    sources = {int(row): str(label) for row, label
               in zip(offered.chosen, offered.sources, strict=True)}
    budget = int(cfg.target.max_points)
    best = None
    while rows:
        rebuilt, path = _delivery(combined, rows, sources, stroke_cov, coverage, cfg,
                                  station, model, gate, kin=kin, log=log)
        rebuilt.corners = path.turns
        best = (rebuilt, path)
        if path.turns <= budget:
            break
        # cheapest value per delivered point goes first
        area = np.asarray(combined.area_mm2, dtype=float)
        values = [(float(area[combined.sets[row]].sum()), int(row)) for row in rows]
        values.sort()
        dropped = values[0][1]
        rows = [row for row in rows if row != dropped]
        say(f"  budget too tight: dropping row {dropped} "
            f"({values[0][0]:.0f} mm2) and retrying")
    return best


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
