"""Choosing the smallest covering set.

To assert: greedy prefers the candidate covering the most AREA rather than the most
points; it stops once a round adds nothing; repair covers every sample the pool can
reach and leaves the genuinely unreachable ones alone; one pose can serve two leftover
points instead of two poses; greedy and repair are counted separately and each chosen
pose is labelled with where it came from; and the coverage curve is monotone and agrees
with the chosen set.
"""

from __future__ import annotations

import numpy as np
import pytest

from scanplan.coverage import CoverageResult
from scanplan.scanner_model import ScannerPose
from scanplan.selection import select


def fake_coverage(sets, *, areas=None, n_targets=None):
    """A real CoverageResult from bare index sets, so selection is tested on its own."""
    sets = [np.asarray(row, dtype=np.int64) for row in sets]
    count = n_targets if n_targets is not None else 1 + max(
        (int(row.max()) for row in sets if len(row)), default=-1)
    area = np.ones(count) if areas is None else np.asarray(areas, dtype=float)
    poses = [ScannerPose(np.array([0.0, 0.0, 300.0 + index]),
                         np.array([0.0, 0.0, -1.0]), 0.0) for index in range(len(sets))]
    return CoverageResult(poses=poses, sets=sets,
                          quality=[np.ones(len(row)) for row in sets],
                          tracker_ok=np.ones(len(sets), dtype=bool),
                          counters=dict.fromkeys(
                              ("rejected_clearance", "rejected_tracker",
                               "rejected_no_points", "rays"), 0),
                          n_targets=count, area_mm2=area,
                          target_area_mm2=float(area.sum()))


def test_greedy_prefers_the_candidate_covering_the_most_area():
    # three one-millimetre samples against two hundred-millimetre ones
    cov = fake_coverage(sets=[np.array([0, 1, 2]), np.array([3, 4])],
                        areas=[1.0, 1.0, 1.0, 100.0, 100.0], n_targets=5)
    assert select(cov, cov.area_mm2).chosen[0] == 1


def test_greedy_runs_to_completion_and_leaves_nothing_for_repair():
    # This is the authoritative round-4 behaviour (report: greedy 50 poses -> 99.10%,
    # repair 0, pool_reach 99.096%): greedy stops only when no pose adds new coverage,
    # so by then it has covered everything the pool can reach.
    cov = fake_coverage(sets=[np.array([0, 1]), np.array([1, 2]), np.array([0, 1, 2])],
                        n_targets=3)
    selection = select(cov, cov.area_mm2, coverage_target_pct=100.0)
    assert len(selection.greedy) == 1
    assert selection.repair == []
    assert np.array_equal(selection.covered, cov.pool_reach())


def test_lambda_moves_the_choice_from_cheap_to_large():
    """`lambda` is the exchange rate between coverage and corners.

    At 0 the density greedy takes the cheap candidate first (10 units for 1 corner beats
    100 for 100); as lambda grows the cost stops mattering and the big one is taken first.
    This is the whole mechanism behind the budget search, so it is asserted directly.
    """
    from scanplan.selection import select_groups

    cov = fake_coverage(sets=[np.arange(100), np.arange(100, 110)], n_targets=110)
    cheap = select_groups(cov, [100.0, 1.0], repair_ok=[False, False])
    assert cheap.chosen[0] == 1
    large = select_groups(cov, [100.0, 1.0], repair_ok=[False, False], lam=100.0)
    assert large.chosen[0] == 0


def test_equal_areas_are_broken_by_the_declared_preference():
    """Exact ties used to be decided by candidate order, which is not a reason."""
    from scanplan.selection import select_groups

    cov = fake_coverage(sets=[np.array([0]), np.array([0])], n_targets=1)
    without = select_groups(cov, [1.0, 1.0], repair_ok=[False, False])
    assert without.chosen == [0]
    with_tie = select_groups(cov, [1.0, 1.0], repair_ok=[False, False], tie=[0.0, 5.0])
    assert with_tie.chosen == [1]


def test_the_skeleton_weight_can_flip_a_close_score_but_not_a_clear_one():
    """The weight is a nudge bounded by the configured 1.3, not a policy."""
    from scanplan.selection import select_groups

    close = fake_coverage(sets=[np.arange(1000), np.arange(1000, 2150)], n_targets=2150)
    assert select_groups(close, [1.0, 1.0],
                         repair_ok=[False, False]).chosen[0] == 1        # 1150 > 1000
    assert select_groups(close, [1.0, 1.0], repair_ok=[False, False],
                         weight=[1.3, 1.0]).chosen[0] == 0               # 1300 > 1150

    clear = fake_coverage(sets=[np.arange(1000), np.arange(1000, 3000)], n_targets=3000)
    assert select_groups(clear, [1.0, 1.0], repair_ok=[False, False],
                         weight=[1.3, 1.0]).chosen[0] == 1               # 2000 > 1300


def test_an_excluded_candidate_is_replaced_not_lost():
    """The hard gate bans a row; the selection must cover that area another way."""
    from scanplan.selection import select_groups

    cov = fake_coverage(sets=[np.arange(4), np.arange(2)], n_targets=4)
    offered = select_groups(cov, [1.0, 1.0], repair_ok=[False, False])
    assert offered.chosen == [0]
    banned = select_groups(cov, [1.0, 1.0], repair_ok=[False, False],
                           exclude=[True, False])
    assert banned.chosen == [1]
    assert banned.coverage_pct < offered.coverage_pct      # and it says so


def test_an_exclusion_mask_of_the_wrong_length_is_refused():
    from scanplan.selection import select_groups

    cov = fake_coverage(sets=[np.array([0])], n_targets=1)
    with pytest.raises(ValueError, match="exclusion"):
        select_groups(cov, [1.0], exclude=[False, True])


def test_select_still_accepts_no_tie_at_all():
    cov = fake_coverage(sets=[np.array([0, 1]), np.array([1, 2])], n_targets=3)
    assert select(cov, cov.area_mm2, tie=None).chosen
    cov = fake_coverage(sets=[np.array([0]), np.array([1]), np.array([2])], n_targets=4)
    selection = select(cov, cov.area_mm2)
    assert np.array_equal(selection.covered, cov.pool_reach())
    assert not selection.covered[3]


def test_a_met_coverage_target_hands_nothing_to_repair():
    # The target is the acceptance figure, so the leftover reachable sample is a hole the
    # workshop patches by hand. Repairing it would add one pose per leftover sample - the
    # exact tail the target exists to cut off.
    cov = fake_coverage(sets=[np.array([0, 1]), np.array([2])], n_targets=3)
    selection = select(cov, cov.area_mm2, coverage_target_pct=66.0)
    assert len(selection.greedy) == 1
    assert selection.repair == []
    assert pytest.approx(selection.coverage_pct) == pytest.approx(200.0 / 3.0)


def test_greedy_and_repair_counts_are_reported_separately():
    cov = fake_coverage(sets=[np.array([0, 1]), np.array([2]), np.array([3])], n_targets=4)
    selection = select(cov, cov.area_mm2, coverage_target_pct=50.0)
    assert len(selection.greedy) + len(selection.repair) == len(selection.chosen)
    assert list(selection.sources) == (["greedy"] * len(selection.greedy)
                                       + ["repair"] * len(selection.repair))
    assert len(selection.greedy) == 1 and selection.repair == []


def test_a_met_target_stops_before_the_tail_is_paid_for():
    # Two poses cover 90.9% and the second one carries the last 9.1%. At a 99% target the
    # tail is not bought; at 100% it is. This is the whole point-count argument in one test.
    cov = fake_coverage(sets=[np.array([0, 1]), np.array([2, 3])],
                        areas=[10.0, 10.0, 1.0, 1.0], n_targets=4)
    stopped = select(cov, cov.area_mm2, coverage_target_pct=90.0)
    assert len(stopped.chosen) == 1
    assert pytest.approx(stopped.coverage_pct) == pytest.approx(100.0 * 20.0 / 22.0)
    complete = select(cov, cov.area_mm2, coverage_target_pct=100.0)
    assert len(complete.chosen) == 2
    assert complete.coverage_pct == pytest.approx(100.0)


def test_a_coverage_target_outside_its_range_is_refused():
    cov = fake_coverage(sets=[np.array([0])], n_targets=1)
    for bad in (0.0, -1.0, 101.0):
        with pytest.raises(ValueError, match="coverage_target_pct"):
            select(cov, cov.area_mm2, coverage_target_pct=bad)


def test_repair_leaves_genuinely_unreachable_points_alone():
    cov = fake_coverage(sets=[np.array([0])], n_targets=2)
    selection = select(cov, cov.area_mm2)
    assert selection.repair == []
    assert selection.covered.tolist() == [True, False]
    # the unreachable sample keeps its area in the denominator, so the figure stays honest
    assert pytest.approx(selection.coverage_pct) == 50.0


def test_coverage_is_measured_in_area_not_in_points():
    # one large sample uncovered out of four: 75% by area even though 3 of 4 samples are in
    cov = fake_coverage(sets=[np.array([0, 1])], areas=[1.0, 1.0, 100.0, 1.0], n_targets=4)
    selection = select(cov, cov.area_mm2)
    assert pytest.approx(selection.coverage_pct, abs=1e-9) == 100.0 * 2.0 / 103.0


def test_the_curve_is_monotone_and_ends_at_the_greedy_coverage():
    cov = fake_coverage(sets=[np.array([0, 1]), np.array([2])], n_targets=4)
    selection = select(cov, cov.area_mm2)
    percentages = [value for _, value in selection.curve]
    assert percentages == sorted(percentages)
    assert selection.curve[-1][0] == len(selection.greedy)
    assert pytest.approx(percentages[-1]) == 75.0


def test_a_pool_that_covers_nothing_selects_nothing():
    cov = fake_coverage(sets=[np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)],
                        n_targets=3)
    selection = select(cov, cov.area_mm2)
    assert selection.chosen == [] and selection.greedy == [] and selection.repair == []
    assert selection.curve == [] and selection.coverage_pct == 0.0


def test_the_same_input_selects_the_same_poses():
    cov = fake_coverage(sets=[np.array([0, 1]), np.array([2, 3]), np.array([1, 2])],
                        areas=[3.0, 1.0, 1.0, 1.0], n_targets=4)
    first = select(cov, cov.area_mm2)
    second = select(cov, cov.area_mm2)
    assert first.chosen == second.chosen
    assert first.curve == second.curve

