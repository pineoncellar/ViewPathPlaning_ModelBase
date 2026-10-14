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


def test_every_reachable_sample_is_covered_once_repair_has_run():
    cov = fake_coverage(sets=[np.array([0]), np.array([1]), np.array([2])], n_targets=4)
    selection = select(cov, cov.area_mm2)
    assert np.array_equal(selection.covered, cov.pool_reach())
    assert not selection.covered[3]


def test_an_earlier_coverage_target_hands_the_rest_to_repair():
    # stopping as soon as the target is met leaves reachable points to the repair pass,
    # which is the only thing that can make repair non-empty
    cov = fake_coverage(sets=[np.array([0, 1]), np.array([2])], n_targets=3)
    selection = select(cov, cov.area_mm2, coverage_target_pct=66.0)
    assert len(selection.greedy) == 1
    assert len(selection.repair) == 1
    assert selection.covered.all()
    assert pytest.approx(selection.coverage_pct) == 100.0


def test_greedy_and_repair_counts_are_reported_separately():
    cov = fake_coverage(sets=[np.array([0, 1]), np.array([2]), np.array([3])], n_targets=4)
    selection = select(cov, cov.area_mm2, coverage_target_pct=50.0)
    assert len(selection.greedy) + len(selection.repair) == len(selection.chosen)
    assert list(selection.sources) == (["greedy"] * len(selection.greedy)
                                       + ["repair"] * len(selection.repair))
    assert len(selection.greedy) == 1 and len(selection.repair) == 2


def test_repair_adds_one_pose_for_two_points_when_one_pose_covers_both():
    # greedy takes the pose covering the two large samples and stops at its target; the
    # two small ones share the other pose, so repair adds it once rather than twice
    cov = fake_coverage(sets=[np.array([0, 1]), np.array([2, 3])],
                        areas=[10.0, 10.0, 1.0, 1.0], n_targets=4)
    selection = select(cov, cov.area_mm2, coverage_target_pct=90.0)
    assert len(selection.greedy) == 1
    assert len(selection.repair) == 1
    assert cov.sets[selection.repair[0]].tolist() == [2, 3]
    assert selection.covered.all()


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

