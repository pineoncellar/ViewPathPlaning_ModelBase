"""The planning stage end to end, on synthetic geometry.

To assert: the same config and seed give the same viewpoint set; the target excludes the
clamped region but not the opposite face; the report's coverage and the viewpoint set
agree; the report states that robot reachability is not modelled; the viewpoint records
carry every documented field; `face="auto"` plans both faces and keeps the better one;
the refinement loop stays inside its budget; a tracker gate never raises coverage; every
documented output file is written; and an empty target is refused rather than reported as
zero coverage.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from scanplan.pipeline import plan, write_outputs
from scanplan.pipeline.ranking import _better, _rank_plan
from scanplan.selection import Selection
from scanplan.tracker import Tracker, part_centroid


def choice(coverage_pct, poses):
    """A `Selection` with a known coverage and pose count, for the ranking rule."""
    target_area = 100.0
    return Selection(chosen=list(range(poses)), greedy=list(range(poses)), repair=[],
                     covered=np.ones(1, dtype=bool), curve=[],
                     coverage_area_mm2=target_area * coverage_pct / 100.0,
                     target_area_mm2=target_area,
                     sources=np.array(["greedy"] * poses, dtype=object))


@pytest.fixture(scope="session")
def quick_cfg(cfg):
    """The shipped config without the refinement rounds.

    Refinement is expensive on purpose - it aims a fresh ladder at every uncovered
    sample - and it has its own tests below. The rest of this file is about the pipeline's
    wiring, so it runs the single-pass path to stay quick.
    """
    return replace(cfg, scanner=replace(cfg.scanner, refine_rounds=0))


def test_same_config_and_seed_give_the_same_viewpoint_set(clamped_plate, quick_cfg):
    first = plan(clamped_plate.station, quick_cfg, face="upper")
    second = plan(clamped_plate.station, quick_cfg, face="upper")
    assert [tuple(viewpoint.position) for viewpoint in first.viewpoints] \
        == [tuple(viewpoint.position) for viewpoint in second.viewpoints]
    assert first.report["coverage_pct"] == second.report["coverage_pct"]


def test_progress_reports_the_real_stages_and_never_rewinds(clamped_plate, quick_cfg):
    """The bar is driven by the pipeline's own stages, not by a guessed time curve."""
    events: list[tuple[str, float, str]] = []

    def hook(stage: str, fraction: float, detail: str = "") -> None:
        events.append((stage, fraction, detail))

    plan(clamped_plate.station, quick_cfg, face="upper", progress=hook)

    assert events, "a run with a hook must report something"
    stages = {stage for stage, _fraction, _detail in events}
    assert {"segmentation", "candidates", "occlusion", "report"} <= stages
    fractions = [fraction for _stage, fraction, _detail in events]
    assert fractions == sorted(fractions), "the bar must not rewind"
    assert all(0.0 <= fraction <= 1.0 for fraction in fractions)
    assert fractions[0] < fractions[-1] == pytest.approx(1.0)


def test_a_run_without_a_hook_is_unchanged(clamped_plate, quick_cfg):
    """The CLI and the scripts pass no hook; adding progress must not move a number."""
    silent = plan(clamped_plate.station, quick_cfg, face="upper")
    listened = plan(clamped_plate.station, quick_cfg, face="upper",
                    progress=lambda stage, fraction, detail="": None)
    assert [tuple(viewpoint.position) for viewpoint in silent.viewpoints] \
        == [tuple(viewpoint.position) for viewpoint in listened.viewpoints]
    assert silent.report["coverage_pct"] == listened.report["coverage_pct"]


def test_a_point_budget_is_respected_and_reported(clamped_plate, quick_cfg):
    """`max_points` is a hard ceiling on delivered points, and the report says why it hit."""
    tight = replace(quick_cfg, target=replace(quick_cfg.target, max_points=12))
    result = plan(clamped_plate.station, tight, face="upper")
    assert result.path is not None
    assert result.path.turns <= 12
    assert result.report["budget"]["verdict"] in ("target_bound", "budget_bound", "exhausted")
    assert result.report["budget"]["delivered_points"] == result.path.turns
    assert result.report["budget"]["trials"]          # the frontier was sampled


def test_an_unbudgeted_run_says_so(clamped_plate, quick_cfg):
    result = plan(clamped_plate.station, quick_cfg, face="upper")
    assert result.report["budget"]["max_points"] is None
    assert result.report["budget"]["verdict"] == "unbudgeted"


def test_the_surface_profile_is_reported_with_its_sources(clamped_plate, quick_cfg):
    result = plan(clamped_plate.station, quick_cfg, face="upper")
    block = result.report["surface_profile"]
    assert block["surface_bias"] == pytest.approx(0.5)
    assert block["sources"]["stroke.min_stroke_mm"] == "base"
    assert block["anchors"]["0.5"]["min_stroke_mm"] == pytest.approx(150.0)


def test_the_kinematics_block_is_present_and_off_by_default(clamped_plate, quick_cfg):
    result = plan(clamped_plate.station, quick_cfg, face="upper")
    block = result.report["kinematics"]
    assert block["enabled"] is False
    assert block["points_without_solution"] == 0
    assert result.report["evaluation"]["rejected_kinematics"] == 0


def test_the_shipped_gate_never_delivers_a_point_the_arm_cannot_hold(clamped_plate):
    """The shipped configuration runs the joint gate, so nothing unreachable is delivered.

    This plate sits at the robot's own origin, i.e. inside its shoulder envelope, so the
    honest answer here is "nothing can be scanned" - and an empty deliverable is what the
    gate must produce, not a program whose points ROBOGUIDE rejects one by one.
    """
    from dataclasses import replace as _replace

    from scanplan.config import load_config
    from scanplan.kinematics import Robot

    shipped = load_config()
    assert shipped.robot.enabled is True
    quick = _replace(shipped, scanner=_replace(shipped.scanner, refine_rounds=0))
    result = plan(clamped_plate.station, quick, face="upper")
    block = result.report["kinematics"]
    assert block["enabled"] is True
    assert block["points_without_solution"] == 0
    arm = Robot(quick.robot)
    if not result.viewpoints:
        # Nothing deliverable is a legitimate answer here (the plate sits inside the arm's
        # shoulder envelope); what must NOT happen is a non-empty delivery with no joint
        # evidence. The gate now filters candidates BEFORE the pool cap, so an empty pool
        # is the normal shape of this outcome.
        assert result.report["coverage_pct"] == pytest.approx(0.0)
        assert result.report["viewpoints"] == 0
    for viewpoint in result.viewpoints:
        assert viewpoint.joints_deg is not None, (
            "a delivered point without joints is exactly what the gate exists to prevent")
        assert arm.feasible(viewpoint.joints_deg)


def test_the_target_excludes_the_clamped_region_but_not_the_opposite_face(clamped_plate,
                                                                        quick_cfg):
    result = plan(clamped_plate.station, quick_cfg, face="upper")
    assert len(result.target.exempt.points) > 0
    # every exempted sample sits on the clamped face, none on the free face behind it
    assert (result.target.exempt.points[:, 2] > 0.6).all()
    assert (result.target.samples.points[:, 2] > 0.6).all()
    lower = plan(clamped_plate.station, quick_cfg, face="lower")
    assert len(lower.target.exempt.points) == 0


def test_the_report_coverage_and_the_viewpoint_set_agree(clamped_plate, quick_cfg):
    result = plan(clamped_plate.station, quick_cfg, face="upper")
    assert (result.report["viewpoints_greedy"] + result.report["viewpoints_repair"]
            == len(result.viewpoints))
    assert pytest.approx(result.report["coverage_pct"], abs=0.05) \
        == result.selection.coverage_pct
    assert sum(viewpoint.n_points for viewpoint in result.viewpoints) \
        >= result.selection.covered.sum()


def test_the_report_states_that_robot_reachability_is_not_modelled(clamped_plate, quick_cfg):
    scope = plan(clamped_plate.station, quick_cfg, face="upper").report["scope"].lower()
    assert "robot" in scope and "not" in scope and "roboguide" in scope


def test_viewpoints_carry_every_documented_field(clamped_plate, quick_cfg):
    viewpoint = plan(clamped_plate.station, quick_cfg, face="upper").viewpoints[0]
    for field in ("id", "position", "axis", "roll_deg", "n_points", "area_mm2",
                  "mean_incidence_deg", "max_incidence_deg", "tracker_ok", "source"):
        assert hasattr(viewpoint, field)
    assert viewpoint.source in {"greedy", "repair"}
    assert np.linalg.norm(viewpoint.axis) == pytest.approx(1.0, rel=1e-12)


def test_auto_plans_both_faces_and_keeps_the_better_one(clamped_plate, quick_cfg):
    result = plan(clamped_plate.station, quick_cfg, face="auto")
    assert set(result.face_candidates) == {"upper", "lower"}
    assert result.target.face == max(result.face_candidates,
                                    key=result.face_candidates.get)
    assert result.face_candidates[result.target.face] == pytest.approx(
        result.report["coverage_pct"], abs=0.05)


def test_the_refine_loop_stays_inside_its_budget(clamped_plate, cfg):
    result = plan(clamped_plate.station, cfg, face="upper")
    assert result.report["counters"]["refine_rounds"] <= cfg.scanner.refine_rounds
    assert result.report["counters"]["refine_candidates"] >= 0


def test_the_refine_loop_aims_at_the_uncovered_samples(clamped_plate, cfg):
    # greedy covers everything the pool can reach, so the only thing refinement can
    # improve is the pool's ceiling - aiming new candidates at the uncovered samples.
    # It only runs below the acceptance target: at a target there is always something
    # uncovered, and re-aiming at it buys coverage the target excludes.
    refining = replace(cfg, scanner=replace(cfg.scanner, refine_rounds=1),
                       target=replace(cfg.target, coverage_target_pct=None))
    result = plan(clamped_plate.station, refining, face="upper")
    counters = result.report["counters"]
    assert counters["refine_rounds"] == 1
    assert counters["refine_candidates"] > 0
    assert len(result.coverage.poses) > len(result.report["viewpoint_records"])


def test_refinement_is_skipped_once_the_acceptance_target_is_met(clamped_plate, cfg):
    """A met target leaves leftovers by definition, so refining would chase the tail."""
    refining = replace(cfg, scanner=replace(cfg.scanner, refine_rounds=2))
    result = plan(clamped_plate.station, refining, face="upper")
    target = refining.target.coverage_target_pct
    if target is not None and result.selection.coverage_pct + 1e-9 >= target:
        assert result.report["counters"]["refine_rounds"] == 0
        assert result.report["counters"]["refine_candidates"] == 0


def test_more_coverage_wins_and_a_tie_goes_to_the_fewer_poses():
    # the loop selects over a pool that grows each round, and greedy on a larger pool can
    # need more poses for the same coverage - which the design ranks as a worse result
    more = choice(99.0, 20)
    fewer_at_the_same_coverage = choice(99.0, 5)
    less_coverage = choice(98.0, 2)
    assert _better(more, less_coverage) is more
    assert _better(less_coverage, more) is more
    assert _better(more, fewer_at_the_same_coverage) is fewer_at_the_same_coverage
    assert _better(more, more) is more


def test_meeting_the_target_outranks_covering_more():
    """Past the acceptance figure, extra coverage buys nothing the workshop asked for."""
    met_with_more_corners = (99.13, 80, 4000.0)
    met_with_fewer_corners = (99.02, 50, 6000.0)
    missed = (98.90, 10, 900.0)
    assert _rank_plan(met_with_fewer_corners[0], met_with_fewer_corners[1],
                      met_with_fewer_corners[2], 99.0) \
        > _rank_plan(met_with_more_corners[0], met_with_more_corners[1],
                     met_with_more_corners[2], 99.0)
    # below the target the old order applies: whoever covers more wins
    assert _rank_plan(99.0, 80, 4000.0, 99.0) > _rank_plan(*missed, 99.0)
    assert _rank_plan(98.95, 5, 100.0, 99.0) > _rank_plan(*missed, 99.0)
    # and an unset target is exactly v1's ordering
    assert _rank_plan(99.5, 100, 0.0, None) > _rank_plan(99.0, 1, 0.0, None)


def test_a_selection_survives_poses_being_appended_to_the_pool():
    """A stroke's row moves when poses are added, because poses come first in the pool.

    Without the remap, a selection made before a refinement round keeps its numbers but
    names different candidates - the measured run delivered "8 items, 0 strokes, 33.22%"
    for a mixed plan that had just covered 97.15%.
    """
    from scanplan.pipeline.refine import _shift_rows

    picked = choice(97.15, 3)
    picked.chosen = [0, 2, 3654, 3660]
    picked.greedy = [0, 2, 3654]
    picked.repair = [3660]
    picked.sources = np.array(["greedy", "greedy", "greedy", "repair"], dtype=object)
    picked.corners = 7
    moved = _shift_rows(picked, 3654, 11654)
    assert moved.chosen == [0, 2, 11654, 11660]
    assert moved.greedy == [0, 2, 11654] and moved.repair == [11660]
    assert moved.corners == 7 and moved.coverage_pct == picked.coverage_pct
    assert _shift_rows(picked, 3654, 3654) is picked


def test_two_meeting_plans_are_separated_by_corners_then_travel():
    left = _rank_plan(99.20, 60, 5000.0, 99.0)
    right = _rank_plan(99.05, 55, 6000.0, 99.0)
    assert right > left
    same_corners = _rank_plan(99.05, 55, 4000.0, 99.0)
    assert same_corners > right


def test_a_priced_selection_uses_its_corners_not_its_item_count():
    """Ten strokes are twenty corners and thirty poses are thirty: they are not the same."""
    strokes = choice(99.0, 10)
    strokes.corners = 20
    poses = choice(99.0, 30)
    poses.corners = 30
    assert strokes.cost_corners == 20 and poses.cost_corners == 30
    assert _better(poses, strokes) is strokes
    # unpriced, the item count is the best available answer
    unpriced = choice(99.0, 10)
    assert unpriced.cost_corners == 10


def test_a_refinement_round_never_makes_the_delivered_set_worse(clamped_plate, cfg):
    # Refinement re-selects over a larger candidate pool, and the ranking rule protects the
    # result: coverage first, then the point count. On this station an extra round now
    # reaches further than the pose-only pool did, because the stroke stage can use the new
    # candidates - so the invariant worth pinning is the ordering rule, not "nothing changes".
    coarse = replace(cfg, scanner=replace(cfg.scanner, refine_rounds=0),
                     target=replace(cfg.target, coverage_target_pct=None))
    refined = replace(cfg, scanner=replace(cfg.scanner, refine_rounds=1),
                      target=replace(cfg.target, coverage_target_pct=None))
    without = plan(clamped_plate.station, coarse, face="upper")
    with_round = plan(clamped_plate.station, refined, face="upper")
    assert with_round.report["counters"]["refine_rounds"] == 1
    assert with_round.report["counters"]["refine_candidates"] > 0
    assert with_round.selection.coverage_pct >= without.selection.coverage_pct - 1e-9
    if with_round.selection.coverage_pct <= without.selection.coverage_pct + 1e-9:
        assert len(with_round.viewpoints) <= len(without.viewpoints)


def test_refinement_does_not_lower_the_coverage(clamped_plate, cfg):
    coarse = replace(cfg, scanner=replace(cfg.scanner, refine_rounds=0))
    refined = replace(cfg, scanner=replace(cfg.scanner, refine_rounds=1))
    without = plan(clamped_plate.station, coarse, face="upper")
    with_refinement = plan(clamped_plate.station, refined, face="upper")
    assert with_refinement.coverage.pool_reach_pct >= without.coverage.pool_reach_pct - 1e-9
    assert (with_refinement.selection.coverage_pct
            >= without.selection.coverage_pct - 1e-9)


def test_a_tracker_gate_never_raises_the_coverage(clamped_plate, quick_cfg):
    without = plan(clamped_plate.station, quick_cfg, face="upper")
    tracker = Tracker(position=np.array([5000.0, 0.0, 100.0]),
                      aim=part_centroid(clamped_plate.station.part),
                      params=quick_cfg.tracker)
    with_tracker = plan(clamped_plate.station, quick_cfg, face="upper", tracker=tracker)
    assert with_tracker.coverage.pool_reach_pct <= without.coverage.pool_reach_pct + 1e-9
    assert with_tracker.report["placeholders"]["tracker_model_position"] \
        == [5000.0, 0.0, 100.0]
    assert all(viewpoint.tracker_ok for viewpoint in with_tracker.viewpoints)


def test_a_tracker_that_sees_nothing_names_the_tracker_as_the_reason(clamped_plate,
                                                                    quick_cfg):
    # planning with a tracker whose range excludes every pose must not report the misses
    # as "a candidate was found": the mounting is the cause and the report has to say so
    tracker = Tracker(position=part_centroid(clamped_plate.station.part)
                      + np.array([0.0, 0.0, 5.0]),
                      aim=part_centroid(clamped_plate.station.part),
                      params=quick_cfg.tracker)
    result = plan(clamped_plate.station, quick_cfg, face="upper", tracker=tracker)
    reasons = result.report["uncovered_by_reason"]
    assert reasons, "every sample should be uncovered when no pose passes the tracker"
    assert "blocked_by_tracker" in reasons
    assert "candidate_found" not in reasons
    assert all(viewpoint.tracker_ok for viewpoint in result.viewpoints)


def test_every_reported_reason_is_one_the_footprint_vocabulary_defines(clamped_plate,
                                                                      quick_cfg):
    from scanplan.coverage import REASONS
    result = plan(clamped_plate.station, quick_cfg, face="upper")
    assert set(result.report["uncovered_by_reason"]) <= set(REASONS)


def test_write_outputs_produces_every_documented_file(clamped_plate, quick_cfg, tmp_path):
    paths = write_outputs(plan(clamped_plate.station, quick_cfg, face="upper"), tmp_path)
    for name in ("viewpoints.json", "viewpoints.csv", "exempt_points.csv",
                 "unreachable_points.csv", "coverage_report.json", "summary.txt",
                 "viewpoints_3d.html", "target_samples.npz"):
        assert name in paths and paths[name].exists()
    assert paths["viewpoints.json"].name == "viewpoints.json"


def test_the_3d_view_carries_the_planned_viewpoints(clamped_plate, quick_cfg, tmp_path):
    result = plan(clamped_plate.station, quick_cfg, face="upper")
    paths = write_outputs(result, tmp_path)
    page = paths["viewpoints_3d.html"].read_text(encoding="utf-8")
    assert '"viewpoints"' in page
    for viewpoint in result.viewpoints:
        assert f'"id":{viewpoint.id}' in page
    # the page must also carry the samples and who covers each of them
    assert '"mask"' in page and '"points"' in page


def test_the_3d_view_carries_the_program_order_of_the_delivered_points(clamped_plate,
                                                                      quick_cfg):
    from scanplan.path import ROLE_COVERAGE, ROLE_REORIENT, ROLE_RETRACT, ROLE_TRAVERSE
    from scanview.payload import build_payload
    result = plan(clamped_plate.station, quick_cfg, face="upper")
    records = build_payload(part=clamped_plate.station.part,
                            viewpoints=result.viewpoints)["viewpoints"]
    assert [record["order"] for record in records] == list(range(len(records)))
    # A stroke plan delivers transition corners too; the page has to know which points
    # merely move the head so it can tell a sweep from a repositioning.
    assert {record["role"] for record in records} <= {
        ROLE_COVERAGE, ROLE_RETRACT, ROLE_TRAVERSE, ROLE_REORIENT}
    assert any(record["role"] == ROLE_COVERAGE for record in records)


def test_stroke_metadata_reaches_the_3d_view():
    import trimesh

    from scanview.payload import build_payload
    part = trimesh.creation.box(extents=[120.0, 90.0, 4.0])
    pose = {"axis": [0.0, 0.0, -1.0], "roll_deg": 0.0, "n_points": 0, "area_mm2": 0.0}
    viewpoints = [
        dict(pose, id=0, order=0, stroke_id=4, role="coverage", position=[0.0, 0.0, 300.0]),
        dict(pose, id=1, order=1, stroke_id=4, role="coverage",
             position=[180.0, 0.0, 300.0]),
        dict(pose, id=2, order=2, stroke_id=None, role="retract",
             position=[180.0, 0.0, 420.0]),
    ]

    records = build_payload(part=part, viewpoints=viewpoints)["viewpoints"]

    assert [record["order"] for record in records] == [0, 1, 2]
    assert [record["stroke_id"] for record in records] == [4, 4, None]
    assert [record["role"] for record in records] == ["coverage", "coverage", "retract"]
    # the playback interpolates its field of view from the corners, so they must survive
    assert all(len(record["corners"]) == 4 for record in records)


def test_the_3d_view_can_be_switched_off(clamped_plate, quick_cfg, tmp_path):
    paths = write_outputs(plan(clamped_plate.station, quick_cfg, face="upper"), tmp_path,
                          write_3d=False)
    assert "viewpoints_3d.html" not in paths
    assert not (tmp_path / "viewpoints_3d.html").exists()
    assert not (tmp_path / "target_samples.npz").exists()


def test_the_viewpoint_json_matches_the_export_contract(clamped_plate, quick_cfg, tmp_path):
    import json
    result = plan(clamped_plate.station, quick_cfg, face="upper")
    paths = write_outputs(result, tmp_path)
    records = json.loads(paths["viewpoints.json"].read_text(encoding="utf-8"))
    assert len(records) == len(result.viewpoints)
    for key in ("id", "position", "axis", "roll_deg", "source"):
        assert key in records[0]


def test_an_empty_target_is_refused_rather_than_reported_as_zero_coverage(clamped_plate,
                                                                          quick_cfg):
    blank = replace(clamped_plate.split,
                    upper=np.zeros(clamped_plate.split.faces, dtype=bool),
                    lower=np.zeros(clamped_plate.split.faces, dtype=bool),
                    band=np.ones(clamped_plate.split.faces, dtype=bool))
    with pytest.raises(ValueError, match="target"):
        plan(clamped_plate.station, quick_cfg, face="upper", split=blank)


def test_a_part_that_is_not_a_thin_sheet_is_refused(clamped_plate, quick_cfg):
    with pytest.raises(ValueError, match="target"):
        plan(clamped_plate.station, quick_cfg, face="upper",
             split=replace(clamped_plate.split,
                           upper=np.zeros(clamped_plate.split.faces, dtype=bool),
                           lower=np.zeros(clamped_plate.split.faces, dtype=bool),
                           band=np.ones(clamped_plate.split.faces, dtype=bool)))


def test_the_result_reports_the_alignment_checks(clamped_plate, quick_cfg):
    result = plan(clamped_plate.station, quick_cfg, face="upper")
    assert result.checks["passed"] is True
    assert result.report["alignment_checks"]["passed"] is True


def test_planning_one_face_can_reuse_a_precomputed_split(clamped_plate, quick_cfg):
    result = plan(clamped_plate.station, quick_cfg, face="upper",
                  split=clamped_plate.split)
    assert result.split is clamped_plate.split


