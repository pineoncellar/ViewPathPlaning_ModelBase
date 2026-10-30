"""The hard-feature criterion: what counts as a wall, and who is responsible for it.

The area target cannot express "scan the side walls": on a 1 m2 panel the walls are a few
tenths of a percent, so a 97 % plan can leave every one of them blank and still meet its
number. These tests pin the three places that make the second figure real - the geometric
definition, the selection that has to keep going for it, and the pruning that must not
spend it again.
"""

from __future__ import annotations

import numpy as np
import pytest

from scanplan.coverage import CoverageResult
from scanplan.path import prune
from scanplan.selection import feature_percent, select
from scanplan.target_surface import build_target, dominant_normal, feature_mask


def fake_coverage(n_targets: int, rows, feature) -> CoverageResult:
    """A coverage object with hand-written rows: the selection only needs the matrix."""
    sets = [np.asarray(row, dtype=np.int64) for row in rows]
    return CoverageResult(
        poses=[object() for _ in rows], sets=sets,
        quality=[np.ones(len(row)) for row in sets],
        tracker_ok=np.ones(len(rows), dtype=bool), counters={},
        n_targets=n_targets, area_mm2=np.ones(n_targets) * 10.0,
        target_area_mm2=float(10.0 * n_targets),
        feature=np.asarray(feature, dtype=bool),
    )


def test_a_wall_and_a_clamped_rim_are_hard_features(clamped_plate):
    """Two independent reasons, and either is enough."""
    station = clamped_plate.station
    normal_up = np.array([0.0, 0.0, 1.0])

    # (a) the rim: a sample 5 mm under the clamp pad is shaded, whatever its normal says
    rim = np.array([[70.0, 0.0, 0.35]])
    far = np.array([[0.0, 0.0, 400.0]])
    flags = feature_mask(np.vstack([rim, far]), np.vstack([normal_up, normal_up]),
                         station, wall_deg=45.0, clearance_mm=15.0, dominant=normal_up)
    assert flags.tolist() == [True, False]

    # (b) the wall: a surface whose normal leans 90 deg off the panel needs no clamp at all
    wall_normals = np.vstack([normal_up, np.array([1.0, 0.0, 0.0])])
    leaning = feature_mask(np.vstack([rim, far]), wall_normals, station,
                           wall_deg=45.0, clearance_mm=0.0, dominant=normal_up)
    assert leaning.tolist() == [False, True]


def test_the_dominant_normal_is_area_weighted():
    normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    weighted = dominant_normal(normals, np.array([10.0, 10.0, 1.0]))
    assert weighted[2] == pytest.approx(20.0 / np.sqrt(401.0), abs=1e-9)
    unweighted = dominant_normal(normals)
    assert unweighted[2] == pytest.approx(2.0 / np.sqrt(5.0), abs=1e-9)
    # a fine mesh round a flange must not outvote the panel it is attached to
    assert weighted[2] > unweighted[2]


def test_building_the_target_attaches_the_feature_mask(clamped_plate, cfg):
    target = build_target(clamped_plate.station.part, clamped_plate.split, "upper",
                          clamped_plate.station, cfg)
    assert target.features is not None
    assert len(target.features) == len(target.samples.points)
    expected = feature_mask(target.samples.points, target.samples.normals,
                            clamped_plate.station,
                            wall_deg=cfg.target.feature_wall_deg,
                            clearance_mm=cfg.target.feature_clearance_mm,
                            dominant=target.dominant_normal)
    assert np.array_equal(target.features, expected)
    # the clamp pad in this fixture shades part of the upper face, so the set is not empty
    assert target.features.any()


def test_the_selection_keeps_going_for_the_features_after_the_area_target():
    """A wall is a fraction of a percent: only its own figure will ask for it."""
    # 18 flat samples and 2 wall samples; candidate 0 takes every flat one, candidate 1 the
    # two walls.
    flat = list(range(18))
    walls = [18, 19]
    feature = np.zeros(20, dtype=bool)
    feature[walls] = True
    coverage = fake_coverage(20, [flat, walls], feature)

    targeted = select(coverage, coverage.area_mm2, coverage_target_pct=80.0,
                      feature_target_pct=100.0)
    assert targeted.chosen == [0, 1]
    assert feature_percent(coverage, coverage.area_mm2, targeted.covered) == \
        pytest.approx(100.0)

    area_only = select(coverage, coverage.area_mm2, coverage_target_pct=80.0)
    assert area_only.chosen == [0]
    assert area_only.feature_coverage_pct == pytest.approx(0.0)


def test_pruning_does_not_spend_the_feature_coverage():
    """The pruner may shave area down to the target, but not the walls with it."""
    sets = [np.arange(18, dtype=np.int64), np.array([18, 19], dtype=np.int64)]
    area = np.ones(20) * 10.0
    feature = np.zeros(20, dtype=bool)
    feature[[18, 19]] = True

    kept = prune([], sets, [0, 1], area, 200.0, target_pct=50.0, max_rounds=32,
                 feature=feature, feature_target_pct=100.0)
    assert kept == [0, 1]

    # without the feature target the same set is pruned down to the area requirement
    bare = prune([], sets, [0, 1], area, 200.0, target_pct=50.0, max_rounds=32)
    assert bare == [0]
