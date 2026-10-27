"""Pick the smallest set of poses that covers the target (set cover).

Greedy - repeatedly take the candidate that adds the most uncovered AREA - then
a repair pass giving each still-reachable-but-uncovered point a dedicated pose.
Greedy plus repair has proved sufficient; the expensive alternatives buy quality
per pose, not coverage, and are worth adding only once the candidate pool stops
being the limiting factor.

Cost is measured in covered area rather than point count, so that fine sampling
around a feature cannot outvote a genuinely large uncovered region.

The repair pass used to be what inflated the viewpoint count: in the measured runs,
greedy reached about 98% with roughly a hundred poses and the last ~1.5% cost
several hundred more. It now stops at the acceptance target, because a pass that adds
one pose per leftover sample returns exactly the tail the target exists to cut off.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .coverage import CoverageResult

# Guards the division when a caller prices a candidate at zero. It is not a bias term:
# a real cost here is 1 to 3 corners, so anything additive would swamp the ratio.
_COST_EPS = 1e-9


@dataclass
class Selection:
    """The chosen poses, where each came from, and what they cover."""

    chosen: list[int]
    greedy: list[int]
    repair: list[int]
    covered: np.ndarray
    curve: list[tuple[int, float]]
    coverage_area_mm2: float
    target_area_mm2: float
    sources: np.ndarray
    # What the choice is expected to cost the head, in corners, when the caller priced it.
    # `len(chosen)` is NOT that number once poses and strokes are mixed: ten strokes are
    # twenty corners and thirty poses are thirty.
    corners: int | None = None
    # The exchange rate this selection was made at (merge plan 5.3). 0 is the density
    # greedy of before, and the budget layer only leaves it when it has to.
    lam: float = 0.0
    # Why the selection stopped: `target_bound` (the acceptance figure came first),
    # `budget_bound` (the point cap did) or `exhausted` (the pool had nothing more to
    # offer). One of these three, never a fourth - see merge plan 6.5.
    bound: str = ""
    # What the point-budget layer did, for the report. Empty when no budget was set.
    budget: dict = field(default_factory=dict)

    @property
    def cost_corners(self) -> int:
        """The priced corner count, or one per item when nothing priced it."""
        return len(self.chosen) if self.corners is None else int(self.corners)

    @property
    def coverage_pct(self) -> float:
        """Area share of the target the chosen poses cover."""
        if self.target_area_mm2 <= 0.0:
            return 0.0
        return float(100.0 * self.coverage_area_mm2 / self.target_area_mm2)


def select(cov: CoverageResult, area_mm2=None, *,
           coverage_target_pct: float | None = None,
           log: Callable[[str], None] | None = None) -> Selection:
    """Greedy set cover on covered AREA, then a repair pass for the leftovers.

    Greedy takes, each round, the pose that adds the most uncovered area - one sparse
    matrix-vector product per round, which keeps this viable at the scales the design
    expects (tens of thousands of samples). Area rather than point count is what stops
    dense sampling around one feature from outvoting a large open region.

    `coverage_target_pct` stops greedy as soon as the target is met, which is the reading
    of "fewest poses that meet the target". Left at None, greedy runs until no pose adds
    new coverage, which is the behaviour of the validated round-4 run (50 poses, 99.10%,
    repair 0).

    The same target stops the repair pass, and that is deliberate: repair adds one pose per
    reachable-but-uncovered sample, so running it after a met target would hand back
    everything the target just saved - the measured run spent seven tenths of its point
    count on the last 0.74 of a percentage point. The pass stays as a net for the case where
    greedy stops short for some other reason; it reports how many points it added, so "it
    did nothing" is visible rather than assumed.
    """
    say = log or (lambda _message: None)
    if coverage_target_pct is not None and not 0.0 < coverage_target_pct <= 100.0:
        raise ValueError("coverage_target_pct must be in (0, 100]")
    matrix = cov.matrix()
    area = (np.asarray(cov.area_mm2, dtype=float) if area_mm2 is None
            else np.asarray(area_mm2, dtype=float))
    if len(area) != cov.n_targets:
        raise ValueError("one area per target sample is required")
    target_area = float(cov.target_area_mm2)

    def covered_pct() -> float:
        return 100.0 * float(area[covered].sum()) / target_area if target_area > 0.0 else 0.0

    covered = np.zeros(cov.n_targets, dtype=bool)
    greedy: list[int] = []
    curve: list[tuple[int, float]] = []
    while True:
        if coverage_target_pct is not None and covered_pct() >= coverage_target_pct:
            break
        gains = matrix @ np.where(covered, 0.0, area)
        best = int(np.argmax(gains))
        if gains[best] <= 0.0:
            break
        greedy.append(best)
        covered |= _row(matrix, best)
        curve.append((len(greedy), covered_pct()))
    say(f"  greedy: {len(greedy)} poses -> {covered_pct():.2f}%")

    repair: list[int] = []
    reachable = cov.pool_reach()
    for target in np.flatnonzero(reachable & ~covered):
        if coverage_target_pct is not None and covered_pct() >= coverage_target_pct:
            # The remainder past the target is what the workshop patches by hand. Repairing
            # it here would hand the whole point of the target back: it adds one pose per
            # reachable-but-uncovered sample, which is exactly the tail the target exists
            # to cut off.
            say(f"  repair: stopped at the {coverage_target_pct:g}% target with "
                f"{int((reachable & ~covered).sum())} reachable samples left uncovered")
            break
        if covered[target]:
            continue          # an earlier repair pose already took this one with it
        candidates = np.flatnonzero(_row(matrix, None, column=target))
        if len(candidates) == 0:
            continue
        gains = matrix[candidates] @ np.where(covered, 0.0, area)
        best = int(candidates[int(np.argmax(gains))])
        repair.append(best)
        covered |= _row(matrix, best)
    say(f"  repair: {len(repair)} poses -> {covered_pct():.2f}%"
        f"   (unreachable in the pool: {int((~reachable).sum())} samples)")

    chosen = greedy + repair
    return Selection(
        chosen=chosen, greedy=greedy, repair=repair, covered=covered, curve=curve,
        coverage_area_mm2=float(area[covered].sum()), target_area_mm2=target_area,
        sources=np.array(["greedy"] * len(greedy) + ["repair"] * len(repair), dtype=object),
    )


def select_groups(cov: CoverageResult, costs, *, area_mm2=None,
                  repair_ok=None, coverage_target_pct: float | None = None,
                  lam: float = 0.0, repair_limit: int | None = None,
                  log: Callable[[str], None] | None = None) -> Selection:
    """Set cover over candidates of MIXED kind: strokes and single poses together.

    The only difference from :func:`select` is the denominator. `costs` is whatever the
    caller wants to minimise per candidate - and that choice is the whole design decision,
    not a detail. Priced in millimetres (a stroke's length plus one stop) a stroke costs
    about nine times a pose, so the greedy pass takes scattered poses first and reaches for
    a sweep only at the tail. Priced in corners (one stop for a pose, two for a stroke) the
    two are comparable, which is what makes the planner prefer one long sweep over the
    handful of loose poses covering the same area.

    `repair_ok` restricts the repair pass (default: everything). Pass True for the
    single poses: a leftover sample is a hole to be patched point by point, and inventing
    a stroke for it would add two corners and a transition to cover one footprint.

    `lam` is the exchange rate between covered area and corners, in the same unit as
    `costs`: the score becomes `gain / (cost + lam)`. It is the one knob the point budget
    turns (merge plan 5.3). At `lam = 0` this is exactly the density greedy of before, so
    the budget layer can start from today's answer and only move away from it when it has
    to. Larger `lam` drowns the cost term, so the pass drifts towards pure area greedy:
    more coverage, more corners. That monotonicity is what makes "the largest lam that
    still fits the budget" a searchable question.

    `repair_limit` caps how many repair items may be added; the budget layer uses it to
    keep the tail - which is one pose per leftover sample - inside a point budget.
    """
    say = log or (lambda _message: None)
    if coverage_target_pct is not None and not 0.0 < coverage_target_pct <= 100.0:
        raise ValueError("coverage_target_pct must be in (0, 100]")
    matrix = cov.matrix()
    area = (np.asarray(cov.area_mm2, dtype=float) if area_mm2 is None
            else np.asarray(area_mm2, dtype=float))
    if len(area) != cov.n_targets:
        raise ValueError("one area per target sample is required")
    costs = np.asarray(costs, dtype=float).reshape(-1)
    if len(costs) != matrix.shape[0]:
        raise ValueError("one cost per candidate is required")
    allowed = (np.ones(matrix.shape[0], dtype=bool) if repair_ok is None
               else np.asarray(repair_ok, dtype=bool))
    if len(allowed) != matrix.shape[0]:
        raise ValueError("one repair flag per candidate is required")
    target_area = float(cov.target_area_mm2)

    def covered_pct() -> float:
        return 100.0 * float(area[covered].sum()) / target_area if target_area > 0.0 else 0.0

    covered = np.zeros(cov.n_targets, dtype=bool)
    greedy: list[int] = []
    curve: list[tuple[int, float]] = []
    while True:
        if coverage_target_pct is not None and covered_pct() >= coverage_target_pct:
            break
        gains = matrix @ np.where(covered, 0.0, area)
        # `max(cost, eps)` rather than `cost + 1`: the +1 was "one millimetre, ignore it",
        # which stopped being true the moment the cost became a corner count (2 or 3).
        # Adding 1 there is a 50% bias and hides the very ratio the caller asked for.
        # `lam` is that bias made explicit, and it is zero unless a point budget asks for it.
        score = gains / np.maximum(costs + float(lam), _COST_EPS)
        best = int(np.argmax(score))
        if gains[best] <= 0.0:
            break
        greedy.append(best)
        covered |= _row(matrix, best)
        curve.append((len(greedy), covered_pct()))
    say(f"  greedy over groups: {len(greedy)} candidates -> {covered_pct():.2f}%")

    repair: list[int] = []
    reachable = cov.pool_reach()
    for target in np.flatnonzero(reachable & ~covered):
        if repair_limit is not None and len(repair) >= max(0, int(repair_limit)):
            say(f"  repair over groups: stopped at the point budget with "
                f"{int((reachable & ~covered).sum())} reachable samples left uncovered")
            break
        if coverage_target_pct is not None and covered_pct() >= coverage_target_pct:
            say(f"  repair over groups: stopped at the {coverage_target_pct:g}% target with "
                f"{int((reachable & ~covered).sum())} reachable samples left uncovered")
            break
        if covered[target]:
            continue
        candidates = np.flatnonzero(_row(matrix, None, column=target) & allowed)
        if len(candidates) == 0:
            continue
        gains = matrix[candidates] @ np.where(covered, 0.0, area)
        best = int(candidates[int(np.argmax(gains))])
        repair.append(best)
        covered |= _row(matrix, best)
    say(f"  repair over groups: {len(repair)} poses -> {covered_pct():.2f}%")

    chosen = greedy + repair
    return Selection(
        chosen=chosen, greedy=greedy, repair=repair, covered=covered, curve=curve,
        coverage_area_mm2=float(area[covered].sum()), target_area_mm2=target_area,
        sources=np.array(["greedy"] * len(greedy) + ["repair"] * len(repair), dtype=object),
        corners=int(round(float(costs[chosen].sum()))) if chosen else 0,
    )


def _row(matrix, row: int | None, column: int | None = None) -> np.ndarray:
    """One row or one column of the coverage matrix, as a flat boolean array."""
    selected = matrix[[row]] if row is not None else matrix[:, [column]]
    return np.asarray(selected.todense()).ravel().astype(bool)
