"""Two defects the delivered pipeline shared with an unmerged branch, and their fixes.

Both were found by reading `feature/modelbase3-update` against this tree, and neither is a
question of taste - they are cases where a number came out wrong:

* a pose whose reference point sits INSIDE the sheet still earned coverage, because
  `measure_mask` happily saw samples through the material it was buried in;
* pruning ran on a selection that already met the requirement, taking coverage away from an
  answer that was acceptable and reporting nothing, because no item was "dropped".
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from scanplan.config import load_config
from scanplan.coverage import evaluate
from scanplan.pipeline.delivery import PRUNE_SLACK_PCT, _prune_to_target
from scanplan.raycast import RayCaster
from scanplan.scanner_model import ScannerModel, ScannerPose
from scanplan.target_surface import SampleSet


def _slab() -> trimesh.Trimesh:
    """A closed 100 mm cube, so "inside" has one unambiguous meaning."""
    return trimesh.creation.box(extents=[100.0, 100.0, 100.0])


def test_the_parity_count_sees_one_crossing_from_inside_and_two_from_outside():
    """The primitive the veto stands on: an odd number of surfaces means "started inside"."""
    caster = RayCaster(_slab())
    origins = np.array([[0.0, 0.0, 0.0],      # centre: exits once
                        [0.0, 0.0, 80.0],     # outside: enters and exits
                        [0.0, 0.0, 50.0]])    # exactly on the boundary
    directions = np.tile([0.0, 0.0, 1.0], (3, 1))
    counts = caster.hit_count(origins, directions)
    assert counts[0] % 2 == 1, "the centre of the box is inside"
    assert counts[1] % 2 == 0, "80 mm above the centre is outside"
    # A miss is zero, not an error, and zero is even - the safe direction for a veto.
    assert caster.hit_count(np.array([[0.0, 0.0, 500.0]]), np.array([[0.0, 0.0, 1.0]]))[0] == 0


def test_inside_part_needs_two_of_three_axes_to_agree(clamped_plate):
    """One grazing ray must not be enough to reject a pose that is genuinely outside."""
    station = clamped_plate.station
    plate = np.asarray(station.part.bounds, dtype=float)
    middle = 0.5 * (plate[0] + plate[1])
    buried = np.array([[middle[0], middle[1], 0.5 * (plate[0][2] + plate[1][2])]])
    above = np.array([[middle[0], middle[1], plate[1][2] + 200.0]])
    outside_far = np.array([[5000.0, 5000.0, 5000.0]])

    assert station.inside_part(buried).tolist() == [True]
    assert station.inside_part(above).tolist() == [False]
    assert station.inside_part(outside_far).tolist() == [False]
    # An empty query is empty, not an error: the pool can legitimately be empty.
    assert station.inside_part(np.zeros((0, 3))).shape == (0,)


def _samples(z_mm: float) -> SampleSet:
    points = np.array([[0.0, 0.0, z_mm], [20.0, 0.0, z_mm], [0.0, 20.0, z_mm]])
    return SampleSet(points=points, normals=np.tile([0.0, 0.0, 1.0], (3, 1)),
                     area_mm2=np.ones(3), total_area_mm2=3.0)


def test_a_pose_buried_in_the_sheet_is_rejected_and_counted(clamped_plate):
    """It cannot be flown, and the samples it "sees" are on the far side of its own material."""
    cfg = load_config()
    model = ScannerModel(cfg.scanner)
    plate = np.asarray(clamped_plate.station.part.bounds, dtype=float)
    middle = 0.5 * (plate[0] + plate[1])
    buried = ScannerPose(position=np.array([middle[0], middle[1],
                                            0.5 * (plate[0][2] + plate[1][2])]),
                         axis=np.array([0.0, 0.0, 1.0]), roll_deg=0.0)
    result = evaluate([buried], _samples(plate[1][2]), clamped_plate.station, model, cfg)
    assert result.counters["rejected_inside_part"] == 1
    assert result.sets[0].size == 0, "a rejected pose must not carry a coverage set"


def test_a_pose_above_the_sheet_is_not_rejected_by_the_veto(clamped_plate):
    cfg = load_config()
    model = ScannerModel(cfg.scanner)
    plate = np.asarray(clamped_plate.station.part.bounds, dtype=float)
    middle = 0.5 * (plate[0] + plate[1])
    pose = ScannerPose(position=np.array([middle[0], middle[1],
                                          plate[1][2] + cfg.scanner.standoff_mm]),
                       axis=np.array([0.0, 0.0, -1.0]), roll_deg=0.0)
    result = evaluate([pose], _samples(plate[1][2]), clamped_plate.station, model, cfg)
    assert result.counters["rejected_inside_part"] == 0


class _Selection:
    """Just enough of a `CoverageResult` for the pruning guard to read."""

    def __init__(self, sets, n_targets, area):
        self.sets = [np.asarray(row, dtype=np.int64) for row in sets]
        self.n_targets = n_targets
        self.area_mm2 = np.asarray(area, dtype=float)
        self.target_area_mm2 = float(self.area_mm2.sum())


def _say(_message):
    pass


def test_pruning_is_skipped_when_the_selection_already_meets_the_target():
    """A set at the requirement has nothing to shave: pruning it can only undershoot."""
    combined = _Selection([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [0, 1, 2, 3, 4]],
                          n_targets=10, area=np.ones(10))
    items = ["a", "b", "c"]
    kept = _prune_to_target(items, combined.sets, [0, 1, 2], combined, load_config(),
                            limit_pct=99.9, say=_say)
    # 100% selected, 99.9 + 0.5 slack -> nothing removed, including the duplicate.
    assert kept == [0, 1, 2]


def test_pruning_still_runs_on_a_real_overshoot():
    """The guard must not disable pruning: an overshoot is exactly what it is for."""
    combined = _Selection([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [0, 1, 2, 3, 4]],
                          n_targets=10, area=np.ones(10))
    items = ["a", "b", "c"]
    kept = _prune_to_target(items, combined.sets, [0, 1, 2], combined, load_config(),
                            limit_pct=50.0, say=_say)
    assert 0 < len(kept) < 3, "the selection overshoots 50% by a wide margin, so it is pruned"


def test_the_slack_is_small_enough_to_still_prune_a_forgiving_target():
    """Documented in one place, because it is the whole width of the guard."""
    assert 0.0 < PRUNE_SLACK_PCT <= 1.0


@pytest.mark.parametrize("limit_pct,expected_all", [(90.0, False), (99.0, False)])
def test_an_overshooting_selection_is_pruned_at_either_end_of_the_range(limit_pct,
                                                                      expected_all):
    combined = _Selection([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [0, 1, 2, 3, 4]],
                          n_targets=10, area=np.ones(10))
    kept = _prune_to_target(["a", "b", "c"], combined.sets, [0, 1, 2], combined,
                            load_config(), limit_pct=limit_pct, say=_say)
    assert (len(kept) == 3) is expected_all
