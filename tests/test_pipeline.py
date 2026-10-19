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

from scanplan.pipeline import _better, plan, write_outputs
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
    # improve is the pool's ceiling - aiming new candidates at the uncovered samples
    refining = replace(cfg, scanner=replace(cfg.scanner, refine_rounds=1))
    result = plan(clamped_plate.station, refining, face="upper")
    counters = result.report["counters"]
    assert counters["refine_rounds"] == 1
    assert counters["refine_candidates"] > 0
    assert len(result.coverage.poses) > len(result.report["viewpoint_records"])


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


def test_a_refinement_round_that_gains_nothing_leaves_the_viewpoint_set_alone(
        clamped_plate, cfg):
    # on this station the leak sits under the clamp, so refinement cannot raise the pool's
    # ceiling. Re-selecting over the enlarged pool then only re-rolls the same answer, and
    # greedy on more candidates can need MORE poses for the same coverage - which the design
    # ranks as a worse result (coverage first, then viewpoint count).
    coarse = replace(cfg, scanner=replace(cfg.scanner, refine_rounds=0))
    refined = replace(cfg, scanner=replace(cfg.scanner, refine_rounds=1))
    without = plan(clamped_plate.station, coarse, face="upper")
    with_round = plan(clamped_plate.station, refined, face="upper")
    assert with_round.report["counters"]["refine_rounds"] == 1
    assert with_round.report["counters"]["refine_candidates"] > 0
    assert len(with_round.viewpoints) == len(without.viewpoints)
    assert [tuple(viewpoint.position) for viewpoint in with_round.viewpoints] \
        == [tuple(viewpoint.position) for viewpoint in without.viewpoints]
    assert with_round.selection.coverage_pct == without.selection.coverage_pct


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


