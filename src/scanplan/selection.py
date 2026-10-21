"""Pick the smallest set of poses that covers the target (set cover).

Greedy - repeatedly take the candidate that adds the most uncovered AREA - then
a repair pass giving each still-reachable-but-uncovered point a dedicated pose.
Greedy plus repair has proved sufficient; the expensive alternatives buy quality
per pose, not coverage, and are worth adding only once the candidate pool stops
being the limiting factor.

Cost is measured in covered area rather than point count, so that fine sampling
around a feature cannot outvote a genuinely large uncovered region.

The repair pass is what inflates the viewpoint count: in the measured runs,
greedy reached about 98% with roughly a hundred poses and the last ~1.5% cost
several hundred more. So when the count looks too high, the fix belongs in
candidate generation. Do not silently skip the tail either - report greedy and
repair counts separately so a reviewer can see which part is compressible.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .coverage import CoverageResult


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

    @property
    def coverage_pct(self) -> float:
        """Area share of the target the chosen poses cover."""
        if self.target_area_mm2 <= 0.0:
            return 0.0
        return float(100.0 * self.coverage_area_mm2 / self.target_area_mm2)


def select(cov: CoverageResult, area_mm2=None, *,
           coverage_target_pct: float | None = None,
           max_viewpoints: int | None = None,
           log: Callable[[str], None] | None = None) -> Selection:
    """Greedy set cover on covered AREA, then a repair pass for the leftovers.

    Greedy takes, each round, the pose that adds the most uncovered area - one sparse
    matrix-vector product per round, which keeps this viable at the scales the design
    expects (tens of thousands of samples). Area rather than point count is what stops
    dense sampling around one feature from outvoting a large open region.

    `coverage_target_pct` stops greedy as soon as the target is met, which is the reading
    of "fewest poses that meet the target" and the only way repair has anything to do.
    Left at None, greedy runs until no pose adds new coverage, which is the behaviour of
    the validated round-4 run (50 poses, 99.10%, repair 0) - keeping that as the default
    is what makes this stage's numbers comparable with it. Either way the repair pass is
    what turns a reachable-but-uncovered point into a dedicated pose; it adds nothing
    when greedy finished, which is a fact worth reporting rather than hiding.
    """
    say = log or (lambda _message: None)
    if coverage_target_pct is not None and not 0.0 < coverage_target_pct <= 100.0:
        raise ValueError("coverage_target_pct must be in (0, 100]")
    if max_viewpoints is not None and max_viewpoints <= 0:
        raise ValueError("max_viewpoints must be positive")
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
        if max_viewpoints is not None and len(greedy) >= max_viewpoints:
            break
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
        if max_viewpoints is not None and len(greedy) + len(repair) >= max_viewpoints:
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


def _row(matrix, row: int | None, column: int | None = None) -> np.ndarray:
    """One row or one column of the coverage matrix, as a flat boolean array."""
    selected = matrix[[row]] if row is not None else matrix[:, [column]]
    return np.asarray(selected.todense()).ravel().astype(bool)
