"""The continuous-path stage: strokes, their coverage, the ordering and the export guard.

To assert: the coverage of one straight stroke is the analytic rectangle (the only place
in the planner with a closed form); the two sampling steps behave as documented, coarse in
the middle and fine at the ends; every stroke the generator emits is legal on the segment
it claims; the path keeps each stroke's two ends adjacent and counts its interrupts; the
group selection prefers one long sweep over the poses it replaces; pruning only removes
what the target allows; the exporter refuses to re-sort a stroke plan, because that
would silently split a sweep in two; and that the program it writes keeps its sweeps
straight while the move between two sweeps is a joint reposition.

Every fixture is synthetic: customer CAD must never be committed.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from conftest import box, mesh_from_triangles, plate_triangles, write_station_file

from scanplan.config import load_config
from scanplan.coverage import CoverageResult
from scanplan.path import (
    LEVEL_JOINT,
    LEVEL_ONE_CORNER,
    LEVEL_STRAIGHT,
    LEVEL_TWO_CORNERS,
    ROLE_COVERAGE,
    ROLE_JOINT,
    ROLE_REORIENT,
    ROLE_RETRACT,
    PathPoint,
    build_path,
    order_strokes,
    pose_gap_deg,
    prune,
    transition_poses,
)
from scanplan.pipeline import plan, write_outputs
from scanplan.roboguide import adapt
from scanplan.rolls import roll_for_axis
from scanplan.scanner_model import ScannerModel, ScannerPose
from scanplan.selection import select_groups
from scanplan.strokes import (
    Stroke,
    StrokeBuilder,
    StrokeCoverage,
    combine,
    dedupe,
    pose_stroke,
    stroke_offsets,
)
from scanplan.target_surface import SampleSet

PLATE_TOP_MM = 0.7


@pytest.fixture(scope="module")
def bench(tmp_path_factory):
    """A 1400 x 700 x 0.7 mm plate on a base, with the fixture well clear underneath.

    Big on purpose: the analytic footprint is 600 x 500 mm at the standoff, so a plate
    smaller than that would clip the rectangle the test is checking.
    """
    from scanplan.scene import load_station, load_station_description

    plate = mesh_from_triangles(
        plate_triangles(-700.0, 700.0, -350.0, 350.0, 0.0, PLATE_TOP_MM, spans=14, bands=7))
    fixture = box(-100.0, 100.0, -100.0, 100.0, -80.0, -40.0)
    base = box(-800.0, 800.0, -450.0, 450.0, -100.0, -80.0)
    directory = tmp_path_factory.mktemp("stroke-bench")
    path = write_station_file(directory, plate, fixture, base)
    return load_station(load_station_description(path), cloud_seed=5)


def grid_samples(x0, x1, y0, y1, z, step) -> SampleSet:
    """A regular sample grid on a horizontal plane, equal area per sample."""
    xs = np.arange(x0, x1 + step / 2.0, step)
    ys = np.arange(y0, y1 + step / 2.0, step)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([grid_x.ravel(), grid_y.ravel(), np.full(grid_x.size, z)], axis=1)
    per_sample = float(step * step)
    return SampleSet(points=points,
                     normals=np.tile(np.array([0.0, 0.0, 1.0]), (len(points), 1)),
                     area_mm2=np.full(len(points), per_sample),
                     total_area_mm2=per_sample * len(points))


@pytest.fixture(scope="module")
def bench_cfg(cfg):
    """The shipped scanner, with range and incidence opened up.

    Both would otherwise cut the analytic rectangle: the range clips its far corners, and
    the incidence limit lands at exactly 45 deg on a sample at the rectangle's edge, which
    is a knife-edge no closed form should be checked against. What is left is the footprint
    alone, which is what the formula describes.
    """
    return replace(cfg, scanner=replace(cfg.scanner, range_mm=(50.0, 5000.0),
                                        max_incidence_deg=89.0))


def horizontal_stroke(x0, x1, y=0.0):
    """A stroke along +X at the standoff above the plate."""
    return Stroke(start=np.array([x0, y, PLATE_TOP_MM + 300.0]),
                  end=np.array([x1, y, PLATE_TOP_MM + 300.0]),
                  axis=np.array([0.0, 0.0, -1.0]))


def test_the_offsets_cover_both_ends_and_never_step_further_than_the_footprint():
    offsets = stroke_offsets(300.0, 50.0, 10.0)
    assert offsets[0] == 0.0 and offsets[-1] == 300.0
    assert len(offsets) == 7                       # ceil(300 / 50) + 1
    assert np.diff(offsets).max() <= 50.0 + 1e-9
    # a stroke shorter than one step still gets a sensible number of poses
    short = stroke_offsets(30.0, 50.0, 10.0)
    assert len(short) == 4 and short[-1] == 30.0
    assert stroke_offsets(0.0, 50.0, 10.0).tolist() == [0.0]


def test_one_stroke_covers_exactly_its_analytic_rectangle(bench, bench_cfg):
    """The closed form of design section 7.2: 600 x (L + 500) mm2.

    No incidence and no range limit reach this plate (normals are straight up, the range is
    opened up), so the only thing that decides coverage is the footprint swept along the
    line - which is exactly what the formula says, and the only place in the planner where
    an answer can be checked against arithmetic rather than against another run.
    """
    samples = grid_samples(-500.0, 500.0, -400.0, 400.0, PLATE_TOP_MM, 20.0)
    builder = StrokeBuilder(samples, bench, ScannerModel(bench_cfg.scanner), bench_cfg)
    stroke = horizontal_stroke(-150.0, 150.0)
    coverage = builder.evaluate([stroke])

    assert len(coverage.sets) == 1
    covered = np.zeros(len(samples.points), dtype=bool)
    covered[coverage.sets[0]] = True
    points = samples.points
    expected = ((np.abs(points[:, 1]) <= 300.0 + 1e-6)
                & (np.abs(points[:, 0]) <= 400.0 + 1e-6))
    assert np.array_equal(covered, expected)
    area = float(len(coverage.sets[0]) * 400.0)
    assert area == pytest.approx(600.0 * (300.0 + 500.0), rel=0.06)


def test_a_longer_stroke_covers_the_extra_length_and_nothing_else(bench, bench_cfg):
    """Doubling the stroke adds exactly the strip it travels over, never more."""
    samples = grid_samples(-500.0, 500.0, -400.0, 400.0, PLATE_TOP_MM, 20.0)
    builder = StrokeBuilder(samples, bench, ScannerModel(bench_cfg.scanner), bench_cfg)
    short = builder.evaluate([horizontal_stroke(-150.0, 150.0)]).sets[0]
    long = builder.evaluate([horizontal_stroke(-250.0, 250.0)]).sets[0]
    assert set(short.tolist()) < set(long.tolist())
    # both ends grow by 100 mm along a 600 mm wide footprint: two strips, not one
    assert len(long) - len(short) == pytest.approx(2.0 * 100.0 * 600.0 / 400.0, rel=0.05)


def test_grown_strokes_respect_the_bounds_and_the_segment_they_claim(bench, cfg):
    samples = grid_samples(-400.0, 400.0, -300.0, 300.0, PLATE_TOP_MM, 10.0)
    builder = StrokeBuilder(samples, bench, ScannerModel(cfg.scanner), cfg)
    grown = builder.generate(samples.points, samples.normals, aim_grid_mm=100.0)
    assert grown, "the generator produced nothing on a plate it can see"
    top = np.array([0.0, 0.0, 1.0])
    for stroke in grown:
        assert cfg.stroke.min_stroke_mm <= stroke.length_mm <= cfg.stroke.max_stroke_mm + 1e-6
        assert np.allclose(np.asarray(stroke.axis), -top, atol=1e-6)
        # the axis is perpendicular to the direction of travel: the head slides, it does not
        # dive towards or away from the surface
        assert abs(float(np.dot(stroke.unit_direction, stroke.axis))) < 1e-9
        fits, margin = builder.fits_at(stroke.check_positions_mm(cfg.stroke.check_step_mm))
        assert fits and margin > 0.0


def test_the_generator_does_not_offer_two_strokes_over_the_same_line():
    first = horizontal_stroke(-200.0, 200.0)
    same_line_shorter = horizontal_stroke(-100.0, 100.0)
    other_line = horizontal_stroke(-200.0, 200.0, y=60.0)
    kept = dedupe([same_line_shorter, first, other_line], tol_mm=20.0)
    assert len(kept) == 2
    assert first in kept and same_line_shorter not in kept


def test_a_short_same_pose_move_is_straight_and_a_pose_change_is_a_joint_travel(bench, cfg):
    """Rule: a joint move is inserted only where it buys something.

    A short hop between two identical poses stays straight at every level - inserting a
    joint move there would add a corner and change nothing else. A pose change is what the
    joint route exists for, and it is reported as such.
    """
    model = ScannerModel(cfg.scanner)
    assert cfg.stroke.connector_level == 2, "the joint route is the shipped default"
    same_pose = [horizontal_stroke(-150.0, 150.0, y=0.0),
                 horizontal_stroke(-150.0, 150.0, y=80.0)]
    path = build_path(same_pose, cfg=cfg, station=bench, model=model,
                      start_near=np.zeros(3))
    assert path.interrupts_a == 1 and path.interrupts_b == 0
    roles = [point.role for point in path.points]
    assert roles == [ROLE_COVERAGE, ROLE_COVERAGE, ROLE_RETRACT, ROLE_COVERAGE,
                     ROLE_COVERAGE]
    assert path.connector_levels == {LEVEL_ONE_CORNER: 1}
    # each stroke's two ends stay adjacent, which is what makes the sweep happen at all
    ids = [point.stroke_id for point in path.points]
    assert ids == [0, 0, 0, 1, 1]

    tilted = Stroke(start=np.array([0.0, 120.0, PLATE_TOP_MM + 300.0]),
                    end=np.array([200.0, 120.0, PLATE_TOP_MM + 260.0]),
                    axis=np.array([0.0, 0.4, -0.9165]))
    mixed = build_path([horizontal_stroke(-150.0, 150.0), tilted], cfg=cfg,
                       station=bench, model=model, start_near=np.zeros(3))
    assert mixed.interrupts_b == 1
    assert ROLE_JOINT in [point.role for point in mixed.points]
    assert mixed.connector_levels == {LEVEL_JOINT: 1}


def test_the_joint_route_puts_the_next_pose_on_the_joint_move(bench, cfg):
    """The L/J/L shape, and why the pre-point carries the NEXT pose.

    If the pre-point kept the previous pose, the whole turn would be left to the short
    straight approach - which is exactly the long-straight-move-with-a-turn-inside-it the
    joint route exists to avoid. The assertion is on the pose, not on the role: the role is
    a label, the pose is what the machine flies.
    """
    model = ScannerModel(cfg.scanner)
    far_apart = [horizontal_stroke(-600.0, -300.0), horizontal_stroke(300.0, 600.0)]
    path = build_path(far_apart, cfg=cfg, station=bench, model=model,
                      start_near=np.zeros(3))
    assert [point.role for point in path.points] == [
        ROLE_COVERAGE, ROLE_COVERAGE, ROLE_RETRACT, ROLE_JOINT, ROLE_COVERAGE,
        ROLE_COVERAGE]
    assert path.connector_levels == {LEVEL_JOINT: 1}
    assert path.min_clearance_mm > 0.0

    last, retract, pre_point, entry = path.points[1:5]
    # the lift keeps the pose it lifted from ...
    assert np.allclose(retract.axis, last.axis) and retract.roll_deg == last.roll_deg
    assert np.allclose(retract.position - last.position, -last.axis
                       * np.linalg.norm(retract.position - last.position))
    # ... and the pre-point already holds the next sweep's pose, so the approach that
    # follows is a straight dive along the next optical axis
    assert np.allclose(pre_point.axis, entry.axis)
    assert pre_point.roll_deg == entry.roll_deg
    approach = entry.position - pre_point.position
    assert np.allclose(approach, entry.axis * np.linalg.norm(approach))
    assert cfg.stroke.retract_clearance_mm <= np.linalg.norm(approach)

    # the threshold is the knob: raise it past the travel and the same move stays straight
    stretched = replace(cfg, stroke=replace(cfg.stroke, joint_travel_min_mm=1000.0))
    straight = build_path(far_apart, cfg=stretched, station=bench, model=model,
                          start_near=np.zeros(3))
    assert LEVEL_JOINT not in straight.connector_levels
    assert straight.connector_levels == {LEVEL_ONE_CORNER: 1}


def test_a_straight_only_site_never_gets_a_joint_move(bench, cfg):
    """`connector_level=1` caps the joint route away, whatever the geometry asks for."""
    model = ScannerModel(cfg.scanner)
    straight_only = replace(cfg, stroke=replace(cfg.stroke, connector_level=1))
    tilted = Stroke(start=np.array([0.0, 120.0, PLATE_TOP_MM + 300.0]),
                    end=np.array([200.0, 120.0, PLATE_TOP_MM + 260.0]),
                    axis=np.array([0.0, 0.4, -0.9165]))
    path = build_path([horizontal_stroke(-150.0, 150.0), tilted], cfg=straight_only,
                      station=bench, model=model, start_near=np.zeros(3))
    assert path.interrupts_b == 1, "the pose still has to change, joint move or not"
    assert ROLE_JOINT not in [point.role for point in path.points]
    assert LEVEL_JOINT not in path.connector_levels


def test_the_straight_fallback_splits_the_rotation_between_flight_and_dive(bench, cfg):
    """When the fly-in cannot spread the turn fast enough, the straight site gets two corners.

    `max_rot_deg_per_mm` is tightened past what any lift can satisfy, so the compressed
    straight shape is refused and the transition falls back to retract + re-orient + dive -
    still straight, with the rotation shared by both segments. The joint route is not a
    candidate here because the level forbids it.
    """
    model = ScannerModel(cfg.scanner)
    tight = replace(cfg, stroke=replace(cfg.stroke, connector_level=1,
                                        max_rot_deg_per_mm=0.05))
    tilted = Stroke(start=np.array([0.0, 120.0, PLATE_TOP_MM + 300.0]),
                    end=np.array([200.0, 120.0, PLATE_TOP_MM + 260.0]),
                    axis=np.array([0.0, 0.4, -0.9165]))
    path = build_path([horizontal_stroke(-150.0, 150.0), tilted], cfg=tight,
                      station=bench, model=model, start_near=np.zeros(3))
    assert path.connector_levels == {LEVEL_TWO_CORNERS: 1}
    assert [point.role for point in path.points][-4:] == [
        ROLE_RETRACT, ROLE_REORIENT, ROLE_COVERAGE, ROLE_COVERAGE]
    # the re-orient point keeps the pose it came from, which is what makes the dive carry
    # the second half of the turn
    reorient = path.points[-3]
    assert np.allclose(reorient.axis, path.points[1].axis)
    assert not np.allclose(reorient.axis, path.points[-1].axis)


def test_connector_level_zero_needs_two_identical_poses_and_a_clear_line(bench, cfg):
    """L0 delivers no corner at all, and only for a pose that does not change."""
    model = ScannerModel(cfg.scanner)
    straight = replace(cfg, stroke=replace(cfg.stroke, connector_level=0))
    same_pose = [horizontal_stroke(-150.0, 150.0, y=0.0),
                 horizontal_stroke(-150.0, 150.0, y=80.0)]
    path = build_path(same_pose, cfg=straight, station=bench, model=model,
                      start_near=np.zeros(3))
    assert [point.role for point in path.points] == [ROLE_COVERAGE] * 4
    assert path.connector_levels == {LEVEL_STRAIGHT: 1}
    # a changed pose must never take the straight route, whatever the level allows
    tilted = Stroke(start=np.array([0.0, 120.0, PLATE_TOP_MM + 300.0]),
                    end=np.array([200.0, 120.0, PLATE_TOP_MM + 260.0]),
                    axis=np.array([0.0, 0.4, -0.9165]))
    mixed = build_path([horizontal_stroke(-150.0, 150.0), tilted], cfg=straight,
                       station=bench, model=model, start_near=np.zeros(3))
    assert LEVEL_STRAIGHT not in mixed.connector_levels


def test_the_transition_poses_are_only_the_moves_between_sweeps(bench, cfg):
    model = ScannerModel(cfg.scanner)
    strokes = [horizontal_stroke(-150.0, 150.0, y=0.0),
               horizontal_stroke(-150.0, 150.0, y=80.0)]
    path = build_path(strokes, cfg=cfg, station=bench, model=model, start_near=np.zeros(3))
    swings = transition_poses(path, cfg)
    assert swings
    # nothing sampled sits on a sweep: every one of these belongs to a connector
    assert all(stroke.length_mm > 0.0 for stroke in strokes)
    assert len(swings) >= path.interrupts_a + path.interrupts_b


def test_a_pose_is_delivered_as_one_point_not_two(bench, cfg):
    """A zero-length stroke IS a pose, so it must not deliver both of its ends."""
    model = ScannerModel(cfg.scanner)
    poses = [pose_stroke(np.array([0.0, 0.0, PLATE_TOP_MM + 300.0]),
                         np.array([0.0, 0.0, -1.0])),
             pose_stroke(np.array([200.0, 0.0, PLATE_TOP_MM + 300.0]),
                         np.array([0.0, 0.0, -1.0]))]
    path = build_path(poses, cfg=cfg, station=bench, model=model, start_near=np.zeros(3))
    assert [point.role for point in path.points].count(ROLE_COVERAGE) == 2
    assert path.coverage_points == 2
    assert path.coverage_strokes == 0
    positions = [tuple(np.round(point.position, 6)) for point in path.points
                 if point.role == ROLE_COVERAGE]
    assert positions == [(0.0, 0.0, PLATE_TOP_MM + 300.0), (200.0, 0.0, PLATE_TOP_MM + 300.0)]


def test_a_zero_length_stroke_is_ordered_like_anything_else(bench, cfg):
    """It must not be special-cased: a pose sits in the sequence wherever it belongs."""
    model = ScannerModel(cfg.scanner)
    items = [horizontal_stroke(-150.0, 150.0),
             pose_stroke(np.array([600.0, 0.0, PLATE_TOP_MM + 300.0]),
                         np.array([0.0, 0.0, -1.0])),
             horizontal_stroke(-150.0, 150.0, y=80.0)]
    path = build_path(items, cfg=cfg, station=bench, model=model,
                      start_near=np.array([600.0, 0.0, 0.0]))
    ids = [point.stroke_id for point in path.points if point.role == ROLE_COVERAGE]
    # the pose is nearest the start, so it goes first despite being the middle item
    assert ids[0] == 1


def test_a_pure_roll_change_counts_as_a_rotation():
    """E7: the ordering, `_same_pose` and the rate check must agree about roll."""
    straight = PathPoint(position=np.zeros(3), axis=np.array([0.0, 0.0, -1.0]),
                         roll_deg=0.0, role=ROLE_COVERAGE, source="greedy")
    rolled = PathPoint(position=np.zeros(3), axis=np.array([0.0, 0.0, -1.0]),
                       roll_deg=30.0, role=ROLE_COVERAGE, source="greedy")
    assert pose_gap_deg(straight.axis, straight.roll_deg, rolled.axis, rolled.roll_deg) \
        == pytest.approx(30.0, abs=1e-6)
    # With the seed roll policy the frames follow their own axes, so a tilt costs the tilt.
    # (Both checks below use the policy's roll: with the bare frame reference the same tilt
    # crosses the helper axis and the delivered frame really does jump, which is the whole
    # reason the policy exists - see tests/test_rolls.py.)
    tilted = PathPoint(position=np.zeros(3), axis=np.array([0.0, 0.4472, -0.8944]),
                       roll_deg=roll_for_axis(np.array([0.0, 0.4472, -0.8944])),
                       role=ROLE_COVERAGE, source="greedy")
    level = replace(straight, roll_deg=roll_for_axis(np.array([0.0, 0.0, -1.0])))
    assert pose_gap_deg(level.axis, level.roll_deg, tilted.axis, tilted.roll_deg) \
        == pytest.approx(26.565, abs=1e-3)


def test_the_ordering_prices_a_pose_change_and_sorts_like_poses_together(bench, cfg):
    """Distance alone makes a large wrist turn look free, because it is a short move."""
    model = ScannerModel(cfg.scanner)
    flat = horizontal_stroke(-100.0, 100.0)
    nearby_tilted = Stroke(start=np.array([110.0, 0.0, PLATE_TOP_MM + 300.0]),
                           end=np.array([310.0, 0.0, PLATE_TOP_MM + 300.0]),
                           axis=np.array([0.0, 0.6, -0.8]))
    far_flat = horizontal_stroke(-100.0, 100.0, y=400.0)
    items = [flat, nearby_tilted, far_flat]
    weighted = order_strokes(items, np.zeros(3), axis_weight=20.0)
    plain = order_strokes(items, np.zeros(3), axis_weight=0.0)
    # the tilted stroke is the odd one out, so weighted ordering puts it last
    assert [index for index, _ in weighted][-1] == 1
    # ... and without the pose term it is second, because it is the closest thing there
    assert [index for index, _ in plain][1] == 1

    by_pose = replace(cfg, stroke=replace(cfg.stroke,
                                          order_axis_weight_mm_per_deg=20.0))
    path = build_path(items, cfg=by_pose, station=bench, model=model, start_near=np.zeros(3))
    ids = [point.stroke_id for point in path.points if point.role == ROLE_COVERAGE]
    # each stroke's ends stay adjacent, so the tilted one occupies the last two slots
    assert ids[-2:] == [1, 1]
    assert ids[0] == ids[1] and ids[2] == ids[3] and ids[0] != ids[2]


def test_the_group_score_prefers_one_sweep_over_the_poses_it_replaces():
    """`gain / cost` is the mechanism that makes a path out of loose poses."""
    n = 100
    area = np.ones(n, dtype=float)
    pose_sets = [np.arange(i * 25, (i + 1) * 25) for i in range(4)]
    stroke_sets = [np.arange(n)]
    poses = [ScannerPose(position=np.zeros(3), axis=np.array([0.0, 0.0, -1.0]))
             for _ in range(5)]
    coverage = CoverageResult(poses=poses, sets=pose_sets + stroke_sets,
                              quality=[np.ones(len(row)) for row in pose_sets + stroke_sets],
                              tracker_ok=np.ones(5, dtype=bool),
                              counters={}, n_targets=n, area_mm2=area,
                              target_area_mm2=float(n))
    # priced in corners: a stroke is 2, a pose is 1
    costs = [1.0] * 4 + [2.0]
    selection = select_groups(coverage, costs, repair_ok=[True] * 4 + [False])
    assert list(selection.chosen) == [4]
    assert selection.coverage_pct == pytest.approx(100.0)
    assert selection.corners == 2


def test_the_group_score_divides_by_the_cost_it_is_given():
    """`max(cost, eps)`, not `cost + 1`: an additive term would flatten the ratio.

    Two poses cover 51% and 49%. At cost 100 (millimetres) the old `+1` was noise; at cost
    1 against 2 (corners) it would turn a 2:1 preference into 3:2 and can flip the choice.
    """
    n = 100
    area = np.ones(n, dtype=float)
    big = np.arange(0, 51)
    small = np.arange(51, 100)
    poses = [ScannerPose(position=np.zeros(3), axis=np.array([0.0, 0.0, -1.0]))
             for _ in range(2)]
    coverage = CoverageResult(poses=poses, sets=[small, big],
                              quality=[np.ones(len(small)), np.ones(len(big))],
                              tracker_ok=np.ones(2, dtype=bool),
                              counters={}, n_targets=n, area_mm2=area,
                              target_area_mm2=float(n))
    # equal cost: whoever covers more is taken first
    assert select_groups(coverage, [1.0, 1.0]).chosen[0] == 1
    # the cheap one wins when the gain per corner is better: 49/1 beats 51/2
    assert select_groups(coverage, [1.0, 2.0]).chosen[0] == 0


def test_pruning_removes_only_what_is_spare():
    sets = [np.arange(0, 50), np.arange(50, 100), np.arange(0, 100)]
    area = np.ones(100, dtype=float)
    kept = prune([None, None, None], sets, [0, 1, 2], area, 100.0, target_pct=99.0,
                 max_rounds=8)
    covered = np.zeros(100, dtype=bool)
    for index in kept:
        covered[sets[index]] = True
    assert covered.all()
    assert len(kept) < 3
    # nothing is removable when every stroke carries a piece of the target alone
    assert prune([None, None], sets, [0, 1], area, 100.0, target_pct=99.0,
                 max_rounds=8) == [0, 1]


def test_combining_the_pool_and_the_strokes_keeps_both_halves_addressable():
    poses = [ScannerPose(position=np.zeros(3), axis=np.array([0.0, 0.0, -1.0]))] * 2
    pool = CoverageResult(poses=poses, sets=[np.array([0]), np.array([1])],
                          quality=[np.ones(1), np.ones(1)],
                          tracker_ok=np.ones(2, dtype=bool), counters={"rays": 2},
                          n_targets=2, area_mm2=np.ones(2), target_area_mm2=2.0)
    strokes = StrokeCoverage(strokes=[horizontal_stroke(-10.0, 10.0)],
                             sets=[np.array([0, 1])], quality=[np.ones(2)],
                             tracker_ok=np.ones(1, dtype=bool),
                             counters={"rays": 2}, n_targets=2, area_mm2=np.ones(2),
                             target_area_mm2=2.0, min_clearance_mm=100.0)
    merged = combine(pool, strokes)
    assert len(merged.poses) == 3
    assert merged.sets[2].tolist() == [0, 1]
    assert merged.pool_reach_pct == pytest.approx(100.0)
    assert merged.counters["rays"] == 4


def test_exporting_a_stroke_plan_refuses_to_re_sort(tmp_path):
    points = [
        {"id": 0, "position": [0.0, 0.0, 300.0], "axis": [0.0, 0.0, -1.0], "roll_deg": 0.0,
         "stroke_id": 0, "role": "coverage"},
        {"id": 1, "position": [200.0, 0.0, 300.0], "axis": [0.0, 0.0, -1.0], "roll_deg": 0.0,
         "stroke_id": 0, "role": "coverage"},
        {"id": 2, "position": [20.0, 120.0, 300.0], "axis": [0.0, 0.0, -1.0],
         "roll_deg": 0.0, "stroke_id": 1, "role": "coverage"},
        {"id": 3, "position": [220.0, 120.0, 300.0], "axis": [0.0, 0.0, -1.0],
         "roll_deg": 0.0, "stroke_id": 1, "role": "coverage"},
    ]
    with pytest.raises(ValueError, match="stroke"):
        adapt.export(points, tmp_path / "sorted", prog_name="SCAN_A")
    assert not (tmp_path / "sorted").exists()

    contract = adapt.export(points, tmp_path / "kept", prog_name="SCAN_A", motion="L",
                            speed="100 mm/sec", termination="CNT0", order=False)
    assert contract["ordering"]["method"] == "as delivered by the planner"
    text = (tmp_path / "kept" / "SCAN_A.LS").read_text(encoding="ascii")
    assert text.count("L P[") == len(points)
    assert "CNT0" in text and "FINE" not in text
    assert contract["program"]["termination"] == "CNT0"
    assert contract["program"]["strokes"] == [0, 1]


def test_a_stroke_plan_repositions_between_sweeps_with_a_joint_move(tmp_path):
    """A sweep stays straight; the one joint travel between sweeps is the exception.

    The `joint` role is the pre-point of the joint route: the head crossing between two
    sweeps in mid air, carrying the next pose. It is the move emitted as a joint move, at
    the percentage speed FANUC wants for one. The retract and the approach into the next
    sweep keep the straight move: those are the segments the clearance check measured, and
    a `traverse` or `reorient` point belongs to a straight-only route, so it stays `L`.
    """
    def point(index, position, role, stroke_id=0):
        return {"id": index, "position": position, "axis": [0.0, 0.0, -1.0],
                "roll_deg": 0.0, "stroke_id": stroke_id, "role": role}

    points = [
        point(0, [0.0, 0.0, 300.0], ROLE_COVERAGE),
        point(1, [200.0, 0.0, 300.0], ROLE_COVERAGE),
        point(2, [200.0, 0.0, 400.0], ROLE_RETRACT),
        point(3, [20.0, 120.0, 400.0], ROLE_JOINT, stroke_id=1),
        point(4, [20.0, 120.0, 300.0], ROLE_COVERAGE, stroke_id=1),
        point(5, [220.0, 120.0, 300.0], ROLE_COVERAGE, stroke_id=1),
    ]
    contract = adapt.export(points, tmp_path, prog_name="SCAN_B", motion="L",
                            speed="100 mm/sec", termination="CNT0", order=False)
    text = (tmp_path / "SCAN_B.LS").read_text(encoding="ascii")
    instructions = [line.split(":", 1)[1].strip().rstrip(";").strip()
                    for line in text.splitlines() if ":L P[" in line or ":J P[" in line]
    assert instructions == [
        "L P[1] 100 mm/sec CNT0",
        "L P[2] 100 mm/sec CNT0",
        "L P[3] 100 mm/sec CNT0",   # the lift, straight
        "J P[4] 100% CNT100",       # the joint travel, the only one in the program
        "L P[5] 100 mm/sec CNT0",   # the approach into the sweep
        "L P[6] 100 mm/sec CNT0",
    ]
    assert contract["program"]["motion"] == "L"
    assert contract["program"]["travel_motion"] == "J"
    assert contract["program"]["travel_speed"] == "100%"
    assert contract["program"]["travel_termination"] == "CNT100"
    assert any("joint travel move" in note for note in contract["not_modelled"])

    straight = adapt.export(points, tmp_path / "straight", prog_name="SCAN_B", motion="L",
                            speed="100 mm/sec", termination="CNT0", order=False,
                            travel_motion="L")
    assert straight["program"]["travel_motion"] == "L"
    assert "J P[" not in (tmp_path / "straight" / "SCAN_B.LS").read_text(encoding="ascii")


def test_the_shipped_configuration_delivers_strokes_and_says_so(clamped_plate):
    shipped = load_config()
    assert shipped.stroke.enabled is True, (
        "the shipped default must deliver sweeps: a pose-only program reports a coverage "
        "figure for stops the head never sweeps")
    quick = replace(shipped, scanner=replace(shipped.scanner, refine_rounds=0))
    result = plan(clamped_plate.station, quick, face="upper")
    assert result.report["stroke_plan"]["enabled"] is True
    assert result.path is not None


def test_a_stroke_run_writes_the_program_order_and_a_matching_3d_mask(
        monkeypatch, clamped_plate, cfg, tmp_path):
    """The delivered files must agree with each other about the order.

    `strokes.json` is the authority; `viewpoints.json` must be in the same order, and the
    bit mask handed to the 3D view has to have one column per DELIVERED point (a stroke
    plan delivers more points than it selects items, so passing the selection's rows
    straight through would mislabel every highlight after the first stroke).
    """
    from scanplan.pipeline import outputs

    captured: dict = {}

    def capture(folder, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(outputs, "write_run_outputs", capture)
    quick = replace(cfg, scanner=replace(cfg.scanner, refine_rounds=0),
                    stroke=replace(cfg.stroke, enabled=True))
    result = plan(clamped_plate.station, quick, face="upper")
    assert result.path is not None, "this plate is wide enough for a single sweep"
    paths = write_outputs(result, tmp_path, write_3d=True)

    body = json.loads(paths["strokes.json"].read_text(encoding="utf-8"))
    orders = [point["order"] for point in body["points"]]
    assert orders == list(range(len(orders)))
    assert body["strokes"] and body["motion"]["type"] == quick.motion.type
    assert [point["position_mm"] for point in body["points"]] == [
        [round(float(v), 4) for v in viewpoint.position] for viewpoint in result.viewpoints]
    assert len(captured["coverage_sets"]) == len(result.viewpoints)
    assert captured["chosen"] == list(range(len(result.viewpoints)))


def test_a_stroke_run_reports_the_path_and_never_covers_less(clamped_plate, cfg):
    quick = replace(cfg, scanner=replace(cfg.scanner, refine_rounds=0),
                    stroke=replace(cfg.stroke, enabled=True))
    result = plan(clamped_plate.station, quick, face="upper")
    strokes = result.report["stroke_plan"]
    assert strokes["enabled"] is True
    assert strokes["candidates"] > 0
    assert result.selection.coverage_pct >= result.pose_only_coverage_pct - 1e-9
    assert result.report["parameter_provenance"]["from_datasheet"]["standoff_mm"] == 300.0
    # In `path` mode EVERY deliverable is a path, including the pose-only one: the connectors
    # are what make the motion between points checkable, and that does not stop being true
    # because the strokes lost.
    assert result.path is not None
    program = result.report["program"]
    assert program["turns"] == len(result.viewpoints)
    assert program["motion"] == quick.motion.type
    assert program["transition_points"] == result.report["viewpoints_transition"]
    ids = [viewpoint.stroke_id for viewpoint in result.viewpoints]
    roles = [viewpoint.role for viewpoint in result.viewpoints]
    assert all(stroke is not None
               for role, stroke in zip(roles, ids, strict=True) if role == ROLE_COVERAGE)
    if not strokes["delivered"]:
        assert "did not beat" in strokes["note"]
        assert strokes["delivered_strokes"] == 0


def test_lean_mode_delivers_poses_and_promises_nothing_between_them(clamped_plate, cfg):
    """`lean` is v1 plus a stopping rule: no strokes, no connectors, no path."""
    lean = replace(cfg, scanner=replace(cfg.scanner, refine_rounds=0),
                   target=replace(cfg.target, delivery_mode="lean"),
                   stroke=replace(cfg.stroke, enabled=True))
    result = plan(clamped_plate.station, lean, face="upper")
    assert result.path is None
    assert result.strokes is None
    assert result.report["stroke_plan"]["enabled"] is False
    assert all(viewpoint.role == ROLE_COVERAGE for viewpoint in result.viewpoints)
    assert result.report["viewpoints_transition"] == 0
    # the acceptance figure still applies, which is the only difference from v1
    assert result.report["program"]["coverage_target_pct"] == lean.target.coverage_target_pct
    target = lean.target.coverage_target_pct
    if target is not None:
        assert result.selection.coverage_pct >= target - 1e-9 or \
            result.coverage.pool_reach_pct < target


def test_the_delivered_coverage_is_the_coverage_of_the_delivered_points(clamped_plate, cfg):
    """The invariant the whole accounting fix exists for.

    The reported figure used to count poses that never reached the program, so a plan could
    promise 99.7% and sweep 92%. Recomputing the union over the points that are actually
    delivered must give the same number as the report, to rounding.
    """
    quick = replace(cfg, scanner=replace(cfg.scanner, refine_rounds=0),
                    stroke=replace(cfg.stroke, enabled=True))
    result = plan(clamped_plate.station, quick, face="upper")
    assert result.path is not None
    covered = np.zeros(result.coverage.n_targets, dtype=bool)
    for point in result.path.points:
        if point.role == ROLE_COVERAGE and point.coverage_index is not None:
            covered[result.coverage.sets[point.coverage_index]] = True
    area = np.asarray(result.coverage.area_mm2, dtype=float)
    swept = 100.0 * float(area[covered].sum()) / result.coverage.target_area_mm2
    assert swept == pytest.approx(result.report["coverage_pct"], abs=1e-3)
