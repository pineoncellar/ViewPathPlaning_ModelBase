"""Building the candidate x target coverage sets, and naming what was missed.

To assert: a pose facing away covers nothing; a plate seen head-on is fully covered;
the quality term carries the incidence cosine; the same inputs give the same sets; a
pose whose body would not fit is rejected before a single ray is cast; the tracker gate
empties a pose's set and is counted separately; the coverage matrix is sparse and agrees
with the sets; merging appends poses; and an uncovered point is attributed to the mesh
that blocked it, to it never being in view, or to the only poses that see it colliding.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from scanplan.config import load_config
from scanplan.coverage import CoverageResult, attribute_uncovered, evaluate, merge
from scanplan.scanner_model import ScannerModel, ScannerPose
from scanplan.target_surface import SampleSet

CLAMP_GAP_MM = 5.0


def head_on_samples(size_mm, *, z_mm=0.7, step_mm=10.0, area_mm2=1.0):
    """Target samples on the plate's top face, inside a square of `size_mm`.

    The grid includes its own centre, so exactly one sample sits on the optical axis of
    a pose placed straight above the origin and must come out with a cosine of 1.
    """
    coordinates = np.arange(-size_mm / 2.0, size_mm / 2.0 + step_mm / 2.0, step_mm)
    points = np.array([[x, y, z_mm] for x in coordinates for y in coordinates])
    normals = np.tile([0.0, 0.0, 1.0], (len(points), 1))
    return SampleSet(points=points, normals=normals,
                     area_mm2=np.full(len(points), area_mm2),
                     total_area_mm2=float(len(points) * area_mm2))


def upside_down_samples(size_mm, *, step_mm=10.0):
    """Samples on the plate's underside, so every pose from above faces away from them."""
    samples = head_on_samples(size_mm, z_mm=0.0, step_mm=step_mm)
    return replace(samples, normals=-samples.normals)


def pose_above(x_mm=0.0, y_mm=0.0, z_mm=300.0, axis=(0.0, 0.0, -1.0), roll_deg=0.0):
    return ScannerPose(position=np.array([x_mm, y_mm, z_mm]),
                       axis=np.array(axis, dtype=float), roll_deg=roll_deg)


def ladder_poses():
    """A small spread of poses above the plate: head-on, tilted, and rolled."""
    poses = []
    for x in (-20.0, 0.0, 20.0):
        for roll in (0.0, 90.0):
            poses.append(pose_above(x_mm=x, y_mm=10.0, roll_deg=roll))
        poses.append(pose_above(x_mm=x, y_mm=10.0, axis=(0.35, 0.0, -0.94)))
    return poses


def blind(positions):
    return np.zeros(len(positions), dtype=bool)


@pytest.fixture(scope="session")
def model_placeholder():
    return ScannerModel(load_config().scanner)


@pytest.fixture(scope="session")
def model_wide():
    """A scanner with a huge range and footprint, so only the tested rule can reject."""
    params = replace(load_config().scanner, range_mm=(1.0, 1.0e6),
                     fov_at_standoff_mm=(1.0e6, 1.0e6), fov_cap_mm=(1.0e6, 1.0e6),
                     max_incidence_deg=89.9)
    return ScannerModel(params)


def test_a_pose_facing_away_covers_nothing(clamped_plate, model_placeholder, cfg):
    facing_up = pose_above(x_mm=0.0, y_mm=0.0, axis=(0.0, 0.0, 1.0))
    result = evaluate([facing_up], head_on_samples(60.0), clamped_plate.station,
                      model_placeholder, cfg)
    assert len(result.sets[0]) == 0
    assert int(result.pool_reach().sum()) == 0
    assert result.pool_reach_pct == 0.0


def test_a_plate_seen_head_on_is_fully_covered(clamped_plate, model_placeholder, cfg):
    result = evaluate([pose_above()], head_on_samples(60.0), clamped_plate.station,
                      model_placeholder, cfg)
    assert result.pool_reach().all()
    assert pytest.approx(result.pool_reach_pct) == 100.0


def test_quality_carries_the_incidence_cosine(clamped_plate, model_placeholder, cfg):
    samples = head_on_samples(60.0)
    result = evaluate([pose_above()], samples, clamped_plate.station, model_placeholder, cfg)
    covered = result.sets[0]
    offset = samples.points[covered] - np.array([0.0, 0.0, 300.0])
    distance = np.linalg.norm(offset, axis=1)
    expected = model_placeholder.incidence_cos(offset / distance[:, None],
                                               samples.normals[covered])
    assert np.allclose(result.quality[0], expected, atol=1e-12)
    # the sample on the optical axis is seen square-on, the corners less so
    assert float(result.quality[0].max()) == pytest.approx(1.0, abs=1e-9)
    assert float(result.quality[0].min()) < 1.0


def test_the_same_inputs_give_the_same_sets(clamped_plate, model_placeholder, cfg):
    samples = head_on_samples(80.0, step_mm=5.0)
    first = evaluate(ladder_poses(), samples, clamped_plate.station, model_placeholder, cfg)
    second = evaluate(ladder_poses(), samples, clamped_plate.station, model_placeholder, cfg)
    assert [row.tolist() for row in first.sets] == [row.tolist() for row in second.sets]
    assert [row.tolist() for row in first.quality] == [row.tolist() for row in second.quality]


def test_body_clearance_rejects_a_pose_before_any_ray_is_cast(clamped_plate,
                                                             model_placeholder, cfg):
    # inside the 5 mm gap between the plate and the clamp's pad: geometrically ideal,
    # physically impossible for a 45 mm scanner body
    in_the_gap = pose_above(x_mm=70.0, y_mm=0.0, z_mm=0.7 + CLAMP_GAP_MM / 2.0)
    result = evaluate([in_the_gap], head_on_samples(60.0), clamped_plate.station,
                      model_placeholder, cfg)
    assert result.counters["rejected_clearance"] == 1
    assert result.counters["rays"] == 0
    assert len(result.sets[0]) == 0


def test_the_tracker_gate_empties_the_set_and_is_counted(clamped_plate,
                                                         model_placeholder, cfg):
    result = evaluate([pose_above()], head_on_samples(60.0), clamped_plate.station,
                      model_placeholder, cfg, visible=blind)
    assert result.counters["rejected_tracker"] == 1
    assert result.counters["rays"] == 0
    assert len(result.sets[0]) == 0
    assert result.tracker_ok.tolist() == [False]


def test_the_tracker_gate_is_reported_per_pose(clamped_plate, model_placeholder, cfg):
    only_first = lambda positions: np.arange(len(positions)) == 0  # noqa: E731
    result = evaluate([pose_above(), pose_above(x_mm=20.0)], head_on_samples(60.0),
                      clamped_plate.station, model_placeholder, cfg, visible=only_first)
    assert result.tracker_ok.tolist() == [True, False]
    assert result.counters["rejected_tracker"] == 1


def test_the_matrix_is_sparse_and_agrees_with_the_sets(clamped_plate, model_placeholder, cfg):
    result = evaluate(ladder_poses(), head_on_samples(120.0, step_mm=10.0),
                      clamped_plate.station, model_placeholder, cfg)
    matrix = result.matrix()
    assert matrix.shape == (len(result.poses), result.n_targets)
    assert matrix.dtype == bool
    assert matrix.nnz == sum(len(row) for row in result.sets)
    dense = matrix.toarray()
    for index, row in enumerate(result.sets):
        assert np.array_equal(np.flatnonzero(dense[index]), row)


def test_merge_appends_poses_and_keeps_the_target_count(clamped_plate,
                                                        model_placeholder, cfg):
    samples = head_on_samples(60.0)
    first = evaluate([pose_above()], samples, clamped_plate.station, model_placeholder, cfg)
    second = evaluate([pose_above(x_mm=20.0)], samples, clamped_plate.station,
                      model_placeholder, cfg)
    merged = merge(first, second)
    assert len(merged.poses) == len(first.poses) + len(second.poses)
    assert merged.n_targets == first.n_targets
    assert merged.pool_reach().sum() >= max(first.pool_reach().sum(), second.pool_reach().sum())
    assert merged.matrix().nnz == sum(len(row) for row in merged.sets)


def test_merge_refuses_mismatched_targets(clamped_plate, model_placeholder, cfg):
    first = evaluate([pose_above()], head_on_samples(60.0), clamped_plate.station,
                     model_placeholder, cfg)
    second = evaluate([pose_above()], head_on_samples(40.0), clamped_plate.station,
                      model_placeholder, cfg)
    with pytest.raises(ValueError, match="target"):
        merge(first, second)


def test_leak_reasons_name_the_blocking_mesh(clamped_plate, model_placeholder, cfg):
    # under the clamp's pad but seen from far above: the pad is what blocks the sight line
    point = np.array([[70.0, 0.0, 0.7]])
    normal = np.array([[0.0, 0.0, 1.0]])
    reasons = attribute_uncovered(point, normal, [pose_above(x_mm=70.0, z_mm=300.0)],
                                 clamped_plate.station, model_placeholder, cfg)
    assert reasons[0] == "blocked_by_fixture"


def test_a_point_facing_away_from_every_pose_is_reported_as_not_facing(clamped_plate,
                                                                     model_placeholder, cfg):
    """`not_facing` is the specific answer: no line in the ladder sees the back."""
    samples = upside_down_samples(120.0)
    reasons = attribute_uncovered(samples.points, samples.normals, ladder_poses(),
                                 clamped_plate.station, model_placeholder, cfg)
    assert set(reasons) == {"not_facing"}


def test_an_empty_pool_leaves_every_point_never_in_view(clamped_plate,
                                                        model_placeholder, cfg):
    """With nothing to test against, no cause can be claimed - not even "it faces away"."""
    samples = upside_down_samples(60.0)
    reasons = attribute_uncovered(samples.points, samples.normals, [],
                                 clamped_plate.station, model_placeholder, cfg)
    assert set(reasons) == {"never_in_view"}


def test_a_point_only_reachable_from_a_colliding_pose_is_named_as_such(clamped_plate,
                                                                       model_wide, cfg):
    # with a generous range the gap pose passes the geometric conditions, so the only
    # reason this point is not covered is that the scanner body cannot go there
    point = np.array([[70.0, 0.0, 0.7]])
    normal = np.array([[0.0, 0.0, 1.0]])
    in_the_gap = pose_above(x_mm=70.0, y_mm=0.0, z_mm=0.7 + CLAMP_GAP_MM / 2.0)
    reasons = attribute_uncovered(point, normal, [in_the_gap], clamped_plate.station,
                                 model_wide, cfg)
    assert reasons[0] == "only_poses_that_collide"


def test_a_point_with_a_clear_sight_line_is_reported_as_candidate_found(clamped_plate,
                                                                       model_placeholder,
                                                                       cfg):
    point = np.array([[0.0, 0.0, 0.7]])
    normal = np.array([[0.0, 0.0, 1.0]])
    reasons = attribute_uncovered(point, normal, [pose_above()], clamped_plate.station,
                                 model_placeholder, cfg)
    assert reasons[0] == "candidate_found"


def test_a_pose_the_tracker_cannot_see_names_the_tracker_not_a_candidate(
        clamped_plate, model_placeholder, cfg):
    # the pose can see the point perfectly; it is the tracker that rules the pose out, so
    # reporting "a candidate was found" hides the actual cause from whoever has to act
    point = np.array([[0.0, 0.0, 0.7]])
    normal = np.array([[0.0, 0.0, 1.0]])
    reasons = attribute_uncovered(point, normal, [pose_above()], clamped_plate.station,
                                 model_placeholder, cfg, tracker_ok=np.array([False]))
    assert reasons[0] == "blocked_by_tracker"


def test_a_pose_the_tracker_can_see_still_reports_the_geometry(clamped_plate,
                                                              model_placeholder, cfg):
    point = np.array([[0.0, 0.0, 0.7]])
    normal = np.array([[0.0, 0.0, 1.0]])
    reasons = attribute_uncovered(point, normal, [pose_above()], clamped_plate.station,
                                 model_placeholder, cfg, tracker_ok=np.array([True]))
    assert reasons[0] == "candidate_found"


def test_geometry_outranks_the_tracker_when_both_apply(clamped_plate, model_placeholder,
                                                      cfg):
    # under the clamp pad: even a pose the tracker can see is blocked by the fixture, and
    # that is the more specific answer; a tracker-invisible pose must not shadow it
    point = np.array([[70.0, 0.0, 0.7]])
    normal = np.array([[0.0, 0.0, 1.0]])
    poses = [pose_above(x_mm=70.0, z_mm=300.0)]
    reasons = attribute_uncovered(point, normal, poses, clamped_plate.station,
                                 model_placeholder, cfg, tracker_ok=np.array([False]))
    assert reasons[0] == "blocked_by_fixture"


def test_the_counters_record_what_was_rejected(clamped_plate, model_placeholder, cfg):
    result = evaluate(ladder_poses(), head_on_samples(60.0), clamped_plate.station,
                      model_placeholder, cfg)
    assert set(result.counters) == {"rejected_clearance", "rejected_tracker",
                                    "rejected_kinematics", "rejected_no_points", "rays"}
    assert result.counters["rays"] > 0
    assert result.counters["rejected_clearance"] == 0


def test_an_empty_pose_list_gives_an_empty_result(clamped_plate, model_placeholder, cfg):
    result = evaluate([], head_on_samples(60.0), clamped_plate.station, model_placeholder, cfg)
    assert isinstance(result, CoverageResult)
    assert result.poses == [] and result.sets == []
    assert result.pool_reach().sum() == 0
    assert result.matrix().shape == (0, result.n_targets)
