"""How two candidate selections are ranked against each other.

The design's ranking, in one place because three stages ask the same question: which
face to scan (`plan`), whether a refinement round kept or lost ground, and whether the
mixed stroke plan beats the pose-only one.
"""

from __future__ import annotations

from ..selection import Selection


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
