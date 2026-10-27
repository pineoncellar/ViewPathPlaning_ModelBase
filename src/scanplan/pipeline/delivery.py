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
from ..path import PathPlan, build_path, prune, transition_poses, unreachable_items
from ..robot import RobotArm
from ..scanner_model import ScannerModel
from ..scene import Station
from ..selection import Selection, select_groups
from ..strokes import Stroke, StrokeBuilder, StrokeCoverage, pose_stroke
from ..target_surface import TargetSet
from .ranking import _rank_plan

# Pruning keeps the requirement, not the achieved peak: pruning against the peak can
# never remove anything, which would make the stage pointless by construction.
PRUNE_TARGET_PCT = 99.0
# How many drop-and-rebuild rounds the joint screen may run. Each round drops at least one
# item, so this is a stop on pathological inputs rather than on the normal case: a path
# normally settles after one or two rounds.
REACH_ROUNDS = 12


def _stroke_builder(cfg: PlannerConfig, target: TargetSet, station: Station,
                    model: ScannerModel, gate, *, robot: RobotArm | None = None,
                    ) -> StrokeBuilder | None:
    """The stroke builder, or None when the delivered set is single poses."""
    if cfg.target.delivery_mode == "lean":
        # The lean deliverable is v1 plus a stopping rule: poses alone, no connectors, and
        # therefore no claim about the motion between them. Strokes cannot be part of it.
        return None
    if not cfg.stroke.enabled:
        return None
    if cfg.stroke.max_stroke_mm <= 0.0:
        return None
    return StrokeBuilder(target.samples, station, model, cfg, visible=gate, robot=robot)


def _group_costs(cfg: PlannerConfig, coverage: CoverageResult, stroke_cov: StrokeCoverage,
                 ) -> tuple[np.ndarray, np.ndarray]:
    """What each candidate costs, and whether the repair pass may use it.

    `corners` is the default because it is what the workshop counts: one stop for a pose,
    two for a stroke, plus the transition each one drags along. `millimetres` is the
    upstream design's pricing and is kept so the two can be compared on the same part.

    One definition, because the budget layer must price candidates exactly as the ordinary
    pass does - a different cost there would make the exchange rate mean something else.
    """
    stop = float(cfg.stroke.stop_cost_mm)
    transition = float(cfg.stroke.transition_corners_estimate)
    n_poses = len(coverage.poses)
    if cfg.stroke.cost_model == "corners":
        costs = np.array([1.0 + transition] * n_poses
                         + [2.0 + transition] * len(stroke_cov.strokes))
    else:
        costs = np.array([stop] * n_poses
                         + [stroke.length_mm + stop for stroke in stroke_cov.strokes])
    repair_ok = np.array([True] * n_poses + [False] * len(stroke_cov.strokes), dtype=bool)
    return costs, repair_ok


def _group_selection(combined: CoverageResult, coverage: CoverageResult,
                     stroke_cov: StrokeCoverage, cfg: PlannerConfig, *,
                     lam: float = 0.0, repair_limit: int | None = None,
                     log=None) -> Selection:
    """Set cover over poses and strokes together, at the configured price and rate.

    `lam` is the exchange rate the point budget bisects; at 0 this is exactly the density
    greedy of before, which is what lets the budget layer start from today's answer.
    """
    costs, repair_ok = _group_costs(cfg, coverage, stroke_cov)
    return select_groups(combined, costs, repair_ok=repair_ok, lam=float(lam),
                         repair_limit=repair_limit,
                         coverage_target_pct=cfg.target.coverage_target_pct, log=log)


def _delivery(combined: CoverageResult, rows: Sequence[int], sources: dict[int, str],
              stroke_cov: StrokeCoverage, coverage: CoverageResult, cfg: PlannerConfig,
              station: Station, model: ScannerModel, gate, *,
              robot: RobotArm | None = None, log=None) -> tuple[Selection, PathPlan]:
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
    kept_items = [items[index] for index in kept]

    def build(items_here: Sequence[Stroke], rows_here: Sequence[int]) -> PathPlan:
        return build_path(items_here, cfg=cfg, station=station, model=model,
                          coverage_index=lambda index: rows_here[index],
                          sources=lambda index: sources.get(rows_here[index], "greedy"),
                          visible=gate, robot=robot, log=log)

    # The joint screen is applied to the DELIVERED points, and a point it rejects is
    # dropped before anything is exported: carrying it in the program only moves the
    # failure to ROBOGUIDE. Dropping an item can change the connectors around it, so the
    # path is ordered and connected again and the survivors are screened again, until the
    # screen passes on every point or there is nothing left to deliver.
    path = build(kept_items, kept_rows)
    dropped_by_arm: list[int] = []
    for _round in range(REACH_ROUNDS):
        blocked = set(unreachable_items(path)) if robot is not None else set()
        if not blocked or not kept_items:
            break
        keep = [index for index in range(len(kept_items)) if index not in blocked]
        dropped_by_arm.extend(kept_rows[index] for index in sorted(blocked))
        say(f"  joint window: {len(blocked)} item(s) reach no configuration inside the "
            f"window; dropping and re-planning {len(keep)}")
        kept_items = [kept_items[index] for index in keep]
        kept_rows = [kept_rows[index] for index in keep]
        path = build(kept_items, kept_rows)
    if dropped_by_arm:
        path.joints["dropped_items"] = sorted(dropped_by_arm)
        say(f"  joint window: dropped {len(dropped_by_arm)} item(s) the arm cannot fly; "
            f"the delivered path has {path.turns} corners and "
            f"{path.joints.get('out_of_window_moves', 0)} move(s) outside the window")
    rebuilt = _rebuild_selection(combined, kept_rows, sources, None)
    return rebuilt, path


# The point budget (merge plan 5.3). Every iteration is one greedy pass over a sparse
# matrix, one ordering and one point count - all milliseconds - so a bounded number of
# them costs well under a per-cent of the run. The bound exists because "bisect until it
# converges" on a response that is only empirically monotone can run away.
_BUDGET_BISECT_STEPS = 8
_BUDGET_MAX_EVALUATIONS = 20
# Multiples of the median candidate cost the exchange rate is sampled at. The ladder is
# relative to the prices in force, so it means the same thing whether candidates are
# priced in corners (1-2) or in millimetres (hundreds).
_LAMBDA_FACTORS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)


def _deliver(combined: CoverageResult, coverage: CoverageResult,
             stroke_cov: StrokeCoverage, cfg: PlannerConfig, station: Station,
             model: ScannerModel, gate, selection: Selection, *,
             robot: RobotArm | None = None, log=None):
    """Order and connect an explicit selection - the unit the budget layer iterates."""
    sources = {int(row): str(source)
               for row, source in zip(selection.chosen, selection.sources, strict=True)}
    return _delivery(combined, selection.chosen, sources, stroke_cov, coverage, cfg,
                     station, model, gate, robot=robot, log=log)


def _bound_kind(selection: Selection, coverage: CoverageResult,
                cfg: PlannerConfig) -> str:
    """Which limit stopped the run: the target, the point cap, or the pool.

    Exactly one of three, because they ask for three different next steps: lower the
    acceptance figure, raise the point budget, or change the scan direction (merge plan
    6.5). A fourth name would only hide which of the three it was.
    """
    target = cfg.target.coverage_target_pct
    if target is not None and selection.coverage_pct + 1e-9 >= float(target):
        return "target_bound"
    if selection.coverage_pct + 1e-6 >= float(coverage.pool_reach_pct):
        return "exhausted"
    return "budget_bound" if cfg.target.max_points is not None else "exhausted"


def _budget_plan(combined: CoverageResult, coverage: CoverageResult,
                 stroke_cov: StrokeCoverage, cfg: PlannerConfig, station: Station,
                 model: ScannerModel, gate, *, robot: RobotArm | None = None, log=None):
    """The most coverage that fits in `target.max_points`, or None when no cap is set.

    The cap is turned into a search over ONE number: the exchange rate `lam` in
    `gain / (cost + lam)`. `lam = 0` is today's density greedy and produces the fewest
    points; raising it drowns the cost term and drifts towards pure area greedy, which
    produces the most coverage and the most corners. So "maximise coverage inside N
    points" becomes "find the largest lam whose delivered plan still fits", which is one
    bisection over an answer that costs milliseconds to evaluate.

    Delivered points are counted by BUILDING the path, not by adding up estimates. A
    connector is one point or two depending on the route and the lift it needed, so any
    "estimate the connector" shortcut would stop the search on a number that is not the
    one written into the program - which is how a budget turns back into silent
    truncation (merge plan 4.2).

    `repair_budget_share` caps the tail repair, because repair is one pose per leftover
    sample: inside a cap it is exactly the tail that must not eat the whole allowance.
    """
    say = log or (lambda _message: None)
    if cfg.target.max_points is None:
        return None
    budget = int(cfg.target.max_points)
    costs, _repair_ok = _group_costs(cfg, coverage, stroke_cov)
    scale = float(np.median(costs)) if len(costs) else 1.0
    scale = max(scale, 1e-6)
    repair_limit = int(np.floor(float(cfg.stroke.repair_budget_share) * budget))
    say(f"  point budget: {budget} points, exchange rate searched over "
        f"{len(_LAMBDA_FACTORS) + _BUDGET_BISECT_STEPS} candidates, repair capped at "
        f"{repair_limit} item(s)")

    evaluations: list[dict] = []
    results: list[tuple[Selection, PathPlan, dict]] = []

    def evaluate(lam: float):
        if len(evaluations) >= _BUDGET_MAX_EVALUATIONS:
            return None
        selection = _group_selection(combined, coverage, stroke_cov, cfg, lam=lam,
                                     repair_limit=repair_limit, log=None)
        rebuilt, path = _deliver(combined, coverage, stroke_cov, cfg, station, model, gate,
                                 selection, robot=robot, log=None)
        entry = {"lam": round(float(lam), 4), "turns": int(path.turns),
                 "coverage_pct": round(float(rebuilt.coverage_pct), 3),
                 "path_length_mm": round(float(path.path_length_mm), 1)}
        evaluations.append(entry)
        results.append((rebuilt, path, entry))
        return rebuilt, path, entry

    cheapest = evaluate(0.0)
    if cheapest is None:
        return None

    def deliver(selection: Selection):
        return _deliver(combined, coverage, stroke_cov, cfg, station, model, gate,
                        selection, robot=robot, log=None)

    if cheapest[1].turns > budget:
        return _trim_to_budget(cheapest, budget, evaluations, cfg, coverage, stroke_cov,
                               deliver)

    # Ladder upwards until the cap is broken, then bisect the last interval. A plan that
    # meets the acceptance figure already has no reason to grow: more coverage past the
    # target buys nothing, so the search stops there rather than at the cap.
    low, high = 0.0, None
    for factor in _LAMBDA_FACTORS:
        lam = scale * factor
        found = evaluate(lam)
        if found is None:
            break
        if found[1].turns > budget:
            high = lam
            break
        low = lam
        if (cfg.target.coverage_target_pct is not None and found[0].coverage_pct + 1e-9
                >= float(cfg.target.coverage_target_pct)):
            break
    if high is not None:
        for _ in range(_BUDGET_BISECT_STEPS):
            middle = 0.5 * (low + high)
            found = evaluate(middle)
            if found is None:
                break
            if found[1].turns > budget:
                high = middle
            else:
                low = middle

    feasible = [item for item in results if item[1].turns <= budget]
    best = max(feasible, key=lambda item: (round(item[0].coverage_pct, 9),
                                           -item[1].turns, -item[1].path_length_mm))
    record = {
        "max_points": budget,
        "repair_limit": repair_limit,
        "chosen_lam": best[2]["lam"],
        "frontier": sorted({(entry["turns"], entry["coverage_pct"]): entry
                            for entry in evaluations}.values(),
                           key=lambda entry: (entry["turns"], -entry["coverage_pct"])),
        "evaluations": len(evaluations),
        "monotone": _is_monotone(evaluations),
    }
    if not record["monotone"]:
        say("  WARNING: coverage did not rise with the exchange rate on this pool; the "
            "answer is the best plan seen, not a bisection result")
    best[0].lam = float(best[2]["lam"])
    best[0].budget = record
    best[0].bound = _bound_kind(best[0], coverage, cfg)
    say(f"  point budget met: {best[1].turns} points, {best[0].coverage_pct:.2f}% at "
        f"exchange rate {best[0].lam:g} ({len(evaluations)} evaluations)")
    return best[0], best[1], record


def _is_monotone(evaluations: Sequence[dict]) -> bool:
    """Does coverage never fall as the exchange rate rises? An empirical property only."""
    ordered = sorted(evaluations, key=lambda entry: entry["lam"])
    return all(later["coverage_pct"] + 1e-6 >= earlier["coverage_pct"]
               for earlier, later in zip(ordered, ordered[1:], strict=False))


def _trim_to_budget(cheapest, budget: int, evaluations: list[dict],
                    cfg: PlannerConfig, coverage: CoverageResult,
                    stroke_cov: StrokeCoverage, deliver):
    """No exchange rate fits the cap, so give up the cheapest area until it does.

    Reached only when the pool holds nothing inside the cap - every candidate costs more
    corners than the budget has. Reporting that instead of trimming would hand over a
    program that breaks the cap, so the tail is dropped by marginal area per corner and
    the amount of coverage given up is written into the record.
    """
    selection, path, entry = cheapest
    costs, _repair_ok = _group_costs(cfg, coverage, stroke_cov)
    areas = np.asarray(coverage.area_mm2, dtype=float)
    current, current_path = selection, path
    dropped: list[dict] = []
    steps = 0
    while current_path.turns > budget and current.chosen and steps < _BUDGET_MAX_EVALUATIONS:
        steps += 1
        rows = list(current.chosen)
        counts = np.zeros(coverage.n_targets, dtype=np.int32)
        for row in rows:
            counts[current.sets[row]] += 1
        scored = []
        for index, row in enumerate(rows):
            own = current.sets[row]
            marginal = float(areas[own[counts[own] == 1]].sum())
            scored.append((marginal / max(float(costs[row]), 1e-9), index, row, marginal))
        scored.sort()
        _value, index, row, marginal = scored[0]
        dropped.append({"row": int(row), "marginal_area_mm2": round(marginal, 1)})
        rows.pop(index)
        current = _rebuild_selection(current, rows, {int(item): "greedy" for item in rows})
        current, current_path = deliver(current)
    record = {
        "max_points": int(budget),
        "bound": "budget_bound",
        "chosen_lam": 0.0,
        "frontier": [entry],
        "evaluations": len(evaluations),
        "monotone": True,
        "dropped_for_budget": dropped,
        "coverage_given_up_mm2": round(sum(item["marginal_area_mm2"] for item in dropped), 1),
        "note": ("even the cheapest exchange rate exceeded the point cap, so the tail was "
                 "dropped by marginal area per corner; the area given up is reported "
                 "rather than hidden"),
    }
    current.budget = record
    current.bound = "budget_bound"
    return current, current_path, record


def _finish_stroke_plan(combined: CoverageResult, coverage: CoverageResult,
                        stroke_cov: StrokeCoverage, selection: Selection,
                        pose_selection: Selection, target: TargetSet, station: Station,
                        model: ScannerModel, cfg: PlannerConfig, gate,
                        *, robot: RobotArm | None = None, log=None):
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

    When `target.max_points` is set, a third candidate joins the comparison: the plan the
    budget layer found, plus the pose-only one only if it too fits the cap. The cap is a
    hard gate on what may be delivered, so an over-budget plan never wins on coverage.
    """
    say = log or (lambda _message: None)
    target_pct = cfg.target.coverage_target_pct
    sources = {int(row): str(source)
               for row, source in zip(selection.chosen, selection.sources, strict=True)}
    mixed_sel, mixed_path = _delivery(combined, selection.chosen, sources, stroke_cov,
                                      coverage, cfg, station, model, gate, robot=robot,
                                      log=log)
    pose_sources = {int(row): str(source) for row, source
                    in zip(pose_selection.chosen, pose_selection.sources, strict=True)}
    pose_sel, pose_path = _delivery(combined, pose_selection.chosen, pose_sources,
                                    stroke_cov, coverage, cfg, station, model, gate,
                                    robot=robot, log=log)
    mixed_sel.corners = mixed_path.turns
    pose_sel.corners = pose_path.turns
    budgeted = _budget_plan(combined, coverage, stroke_cov, cfg, station, model, gate,
                            robot=robot, log=log)
    if budgeted is not None:
        selection, path, _record = budgeted
        selection.corners = path.turns
        candidates = [(mixed_sel, mixed_path, "mixed"), (pose_sel, pose_path, "pose-only"),
                      (selection, path, "budgeted")]
    else:
        candidates = [(mixed_sel, mixed_path, "mixed"), (pose_sel, pose_path, "pose-only")]

    cap = None if cfg.target.max_points is None else int(cfg.target.max_points)
    allowed = [item for item in candidates if cap is None or item[1].turns <= cap]
    if not allowed:
        allowed = [candidates[0]]
    for candidate, candidate_path, label in candidates:
        if candidate_path.turns > (cap if cap is not None else np.inf):
            if label == "pose-only":
                say(f"  the pose-only plan needs {candidate_path.turns} corners and does "
                    f"not fit the {cap}-point budget: not considered")
    best = max(allowed, key=lambda item: _rank_plan(item[0].coverage_pct, item[1].turns,
                                                    item[1].path_length_mm, target_pct))
    for candidate, candidate_path, label in candidates:
        if candidate is best[0]:
            continue
        say(f"  {label} plan: {candidate.coverage_pct:.2f}%, {candidate_path.turns} "
            f"corners, {candidate_path.path_length_mm:.0f} mm")
    say(f"  delivering the {best[2]} plan: {best[0].coverage_pct:.2f}%, "
        f"{best[1].turns} corners, {best[1].path_length_mm:.0f} mm")
    if not best[0].bound:
        best[0].bound = _bound_kind(best[0], coverage, cfg)
    if budgeted is not None:
        # The search is part of what this run did, so the report carries it even when a
        # plan the budget layer did not propose wins: the cap was in force and these were
        # its candidates and their price in coverage. `delivered` says which plan the
        # numbers belong to, so nobody reads `chosen_lam` as the delivered plan's rate.
        best[0].budget = {**budgeted[2], "delivered": best[0] is budgeted[0],
                          "delivered_plan": best[2]}
    return (best[0], best[1],
            _swept_pct(best[0], best[1], target, station, model, cfg, combined, gate,
                       robot=robot, log=log))


def _swept_pct(rebuilt: Selection, path: PathPlan, target: TargetSet, station: Station,
               model: ScannerModel, cfg: PlannerConfig, combined: CoverageResult, gate,
               *, robot: RobotArm | None = None, log=None) -> float:
    """The coverage the path actually sweeps, transitions included.

    Reported for the workshop and never used for acceptance: it depends on the ordering,
    so a plan could "gain" coverage by flying over the part. Transition poses go through
    the same gates the candidates did, joint window included, so a swing that no
    configuration of the arm can fly does not add coverage it will never sweep.
    """
    say = log or (lambda _message: None)
    swept_pct = rebuilt.coverage_pct
    swings = transition_poses(path, cfg)
    if swings:
        swept = evaluate(swings, target.samples, station, model, cfg, visible=gate,
                         robot=robot, log=log)
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
