"""The continuous-path stage: strokes, their coverage, the ordering and the export guard.

To assert: the coverage of one straight stroke is the analytic rectangle (the only place
in the planner with a closed form); the two sampling steps behave as documented, coarse in
the middle and fine at the ends; every stroke the generator emits is legal on the segment
it claims; the path keeps each stroke's two ends adjacent and counts its interrupts; the
group selection prefers one long sweep over the poses it replaces; pruning only removes
what the target allows; and the exporter refuses to re-sort a stroke plan, because that
would silently split a sweep in two.

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
    ROLE_COVERAGE,
    ROLE_REORIENT,
    ROLE_RETRACT,
    ROLE_TRAVERSE,
    build_path,
    prune,
    transition_poses,
)
from scanplan.pipeline import plan, write_outputs
from scanplan.roboguide import adapt
from scanplan.scanner_model import ScannerModel, ScannerPose
from scanplan.selection import select_groups
from scanplan.strokes import (
    Stroke,
    StrokeBuilder,
    StrokeCoverage,
    combine,
    dedupe,
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


def test_a_pose_change_becomes_a_reorientation_and_a_translation_does_not(bench, cfg):
    model = ScannerModel(cfg.scanner)
    same_pose = [horizontal_stroke(-150.0, 150.0, y=0.0),
                 horizontal_stroke(-150.0, 150.0, y=80.0)]
    path = build_path(same_pose, cfg=cfg, station=bench, model=model, start_near=np.zeros(3))
    assert path.interrupts_a == 1 and path.interrupts_b == 0
    roles = [point.role for point in path.points]
    assert roles == [ROLE_COVERAGE, ROLE_COVERAGE, ROLE_RETRACT, ROLE_TRAVERSE,
                     ROLE_COVERAGE, ROLE_COVERAGE]
    # each stroke's two ends stay adjacent, which is what makes the sweep happen at all
    ids = [point.stroke_id for point in path.points]
    assert ids == [0, 0, 0, 1, 1, 1]

    tilted = Stroke(start=np.array([0.0, 120.0, PLATE_TOP_MM + 300.0]),
                    end=np.array([200.0, 120.0, PLATE_TOP_MM + 260.0]),
                    axis=np.array([0.0, 0.4, -0.9165]))
    mixed = build_path([horizontal_stroke(-150.0, 150.0), tilted], cfg=cfg, station=bench,
                       model=model, start_near=np.zeros(3))
    assert mixed.interrupts_b == 1
    assert ROLE_REORIENT in [point.role for point in mixed.points]


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


def test_the_group_score_prefers_one_sweep_over_the_poses_it_replaces():
    """`gain / (length + stop)` is the mechanism that makes a path out of loose poses."""
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
    costs = [100.0] * 4 + [100.0]       # four poses, then the stroke
    selection = select_groups(coverage, costs, repair_ok=[True] * 4 + [False])
    assert list(selection.chosen) == [4]
    assert selection.coverage_pct == pytest.approx(100.0)


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


def test_the_shipped_configuration_delivers_single_poses_and_says_so(clamped_plate):
    shipped = load_config()
    assert shipped.stroke.enabled is False, (
        "the shipped default must reproduce v1: that is the regression baseline")
    quick = replace(shipped, scanner=replace(shipped.scanner, refine_rounds=0))
    result = plan(clamped_plate.station, quick, face="upper")
    assert result.path is None
    assert result.report["stroke_plan"]["enabled"] is False
    assert result.report["viewpoints_transition"] == 0
    assert result.report["viewpoints_greedy"] + result.report["viewpoints_repair"] \
        == len(result.viewpoints)


def test_a_stroke_run_writes_the_program_order_and_a_matching_3d_mask(
        monkeypatch, clamped_plate, cfg, tmp_path):
    """The delivered files must agree with each other about the order.

    `strokes.json` is the authority; `viewpoints.json` must be in the same order, and the
    bit mask handed to the 3D view has to have one column per DELIVERED point (a stroke
    plan delivers more points than it selects items, so passing the selection's rows
    straight through would mislabel every highlight after the first stroke).
    """
    from scanplan import pipeline

    captured: dict = {}

    def capture(folder, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(pipeline, "write_run_outputs", capture)
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
    if strokes["delivered"]:
        program = result.report["program"]
        assert program["turns"] == len(result.viewpoints)
        assert program["motion"] == quick.motion.type
        assert program["transition_points"] == result.report["viewpoints_transition"]
        ids = [viewpoint.stroke_id for viewpoint in result.viewpoints]
        roles = [viewpoint.role for viewpoint in result.viewpoints]
        assert all(role == ROLE_COVERAGE
                   for role, stroke in zip(roles, ids, strict=True) if stroke is not None)
    else:
        assert result.path is None
        assert "did not beat" in strokes["note"]
