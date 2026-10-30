"""How two candidate selections are ranked against each other.

The design's ranking, in one place because three stages ask the same question: which
face to scan (`plan`), whether a refinement round kept or lost ground, and whether the
mixed stroke plan beats the pose-only one.
"""

from __future__ import annotations

from ..selection import Selection


def _rank_plan(coverage_pct: float, corners: int, length_mm: float,
               target_pct: float | None, feature_pct: float | None = None,
               feature_target_pct: float | None = None) -> tuple:
    """Met the targets first, then fewer corners, then less travel.

    The first component is a two-way split, not a coverage ranking: once a plan meets the
    acceptance figures, more coverage buys nothing the workshop asked for, so it must not
    outrank a plan with fewer corners. Below the target the old order applies - whoever
    covers more wins - which is what keeps `coverage_target_pct = null` behaving exactly
    like v1.

    Two acceptance figures, and both must be met: the area share and the share of the hard
    features (walls, bends and the clamped rim). A plan that hits 98 % of a panel while
    leaving its walls blank has not met the bar, and ranking it equal to one that did is
    exactly how the walls kept getting dropped.
    """
    if target_pct is None and feature_target_pct is None:
        return (round(float(coverage_pct), 6), -int(corners), -float(length_mm))
    met = ((target_pct is None or float(coverage_pct) + 1e-9 >= float(target_pct))
           and (feature_target_pct is None or (feature_pct is not None
                                               and float(feature_pct) + 1e-9
                                               >= float(feature_target_pct))))
    if met:
        return (1, -int(corners), -float(length_mm))
    return (0, round(float(coverage_pct), 6),
            round(-1.0 if feature_pct is None else float(feature_pct), 6),
            -int(corners), -float(length_mm))


def meets_targets(coverage_pct: float, target_pct: float | None,
                  feature_pct: float | None = None,
                  feature_target_pct: float | None = None) -> bool:
    """Whether BOTH acceptance figures are met, with no target meaning "not required".

    One function because three stages ask it (the ranking above, the pruning guard and the
    "sweeps are required" switch), and they must not disagree about whether a plan is done.
    """
    if target_pct is None and feature_target_pct is None:
        return False
    if target_pct is not None and float(coverage_pct) + 1e-9 < float(target_pct):
        return False
    if feature_target_pct is not None and (feature_pct is None
                                           or float(feature_pct) + 1e-9
                                           < float(feature_target_pct)):
        return False
    return True


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
    return _rank_plan(selection.coverage_pct, selection.cost_corners, 0.0, target_pct,
                      selection.feature_coverage_pct, selection.feature_target_pct)
