"""The coverage report and the viewpoint set that has to travel with it.

To assert: clusters group neighbouring misses and report their bounding box; the
headline numbers sit at the top level and agree with the selection; the report says in
plain words what is NOT modelled (robot reachability, joint limits, robot collision, and
the arm's occlusion of the tracker); every still-placeholder parameter is listed with the
most sensitive one first; the uncovered samples are accounted for by reason and clustered;
the config source and ray backend are recorded so a figure can be traced to a run; and the
viewpoint records carry exactly the field names the export contract expects.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from scanplan.candidates import generate
from scanplan.config import RobotParams
from scanplan.coverage import attribute_uncovered, evaluate
from scanplan.kinematics import KinematicGate
from scanplan.report import (
    build_report,
    cluster_points,
    coverage_layers_block,
    summary_lines,
    viewpoints_from_selection,
    write_report,
    write_samples_csv,
)
from scanplan.scanner_model import ScannerModel
from scanplan.selection import select
from scanplan.target_surface import build_target


@pytest.fixture(scope="session")
def planned(clamped_plate, cfg):
    """A small end-to-end plan on the synthetic station, from target to viewpoints."""
    model = ScannerModel(cfg.scanner)
    target = build_target(clamped_plate.station.part, clamped_plate.split, "upper",
                          clamped_plate.station, cfg)
    poses = generate(target.samples.points, target.samples.normals, cfg.scanner,
                     seed=cfg.seed)
    coverage = evaluate(poses, target.samples, clamped_plate.station, model, cfg)
    selection = select(coverage, coverage.area_mm2)
    viewpoints = viewpoints_from_selection(selection, coverage, target.samples)
    report = build_report(config=cfg, model=model, station=clamped_plate.station,
                          split=clamped_plate.split, target=target, coverage=coverage,
                          selection=selection, viewpoints=viewpoints,
                          runtime={"seconds": 1.5, "ray_backend": "embree"})
    return {"station": clamped_plate.station, "target": target, "poses": poses,
            "coverage": coverage, "selection": selection, "viewpoints": viewpoints,
            "report": report, "config": cfg, "split": clamped_plate.split, "model": model}


def test_clusters_group_neighbours_and_sort_by_area():
    regions = cluster_points(np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 0.0], [200.0, 200.0, 0.0]]),
                             np.ones(3), cell_mm=30.0)
    assert [region["points"] for region in regions] == [2, 1]
    assert regions[0]["area_mm2"] > regions[1]["area_mm2"]


def test_a_cluster_reports_its_bounding_box_and_centre():
    regions = cluster_points(np.array([[0.0, 0.0, 0.0], [5.0, 7.0, 3.0]]), np.ones(2),
                             cell_mm=30.0)
    assert regions[0]["size_mm"] == [5.0, 7.0, 3.0]
    assert regions[0]["center"] == [2.5, 3.5, 1.5]


def test_clusters_are_capped_and_empty_input_is_handled():
    points = np.array([[index * 100.0, 0.0, 0.0] for index in range(30)])
    assert len(cluster_points(points, np.ones(30), cell_mm=30.0, limit=12)) == 12
    assert cluster_points(np.zeros((0, 3)), np.zeros(0)) == []


def test_the_headline_numbers_are_at_the_top_level(planned):
    report = planned["report"]
    assert report["target_face_mm2"] > 0 and report["target_points"] > 0
    assert report["viewpoints"] == len(planned["viewpoints"])
    assert pytest.approx(report["coverage_pct"], abs=0.05) == planned["selection"].coverage_pct
    assert (report["viewpoints_greedy"] + report["viewpoints_repair"]
            == len(planned["viewpoints"]))


def test_report_states_what_it_does_not_model(planned):
    scope = planned["report"]["scope"].lower()
    assert "geometric" in scope
    assert "robot" in scope and "not" in scope
    assert "joint limits" in scope and "collision" in scope
    caveat = planned["report"]["tracker_caveat"].lower()
    assert "arm" in caveat and "tracker" in caveat


def test_report_lists_the_placeholder_parameters_with_incidence_first(planned):
    report = planned["report"]
    placeholders = report["placeholders"]
    assert placeholders["max_incidence_deg"] == 45.0
    assert list(placeholders)[0] == "max_incidence_deg"
    assert "most sensitive" in report["placeholders_note"].lower()
    assert placeholders["tracker_model_position"] is None


def test_report_records_a_supplied_tracker_pose(planned, cfg):
    from scanplan.tracker import Tracker, part_centroid
    tracker = Tracker(position=np.array([5000.0, 0.0, 100.0]),
                      aim=part_centroid(planned["station"].part), params=cfg.tracker)
    report = build_report(config=cfg, model=planned["model"], station=planned["station"],
                          split=planned["split"], target=planned["target"],
                          coverage=planned["coverage"], selection=planned["selection"],
                          viewpoints=planned["viewpoints"], tracker=tracker)
    assert report["placeholders"]["tracker_model_position"] == [5000.0, 0.0, 100.0]
    assert report["tracker"]["name"] == "tracker"


def test_report_accounts_for_every_uncovered_sample(planned):
    report = planned["report"]
    reasons = report["uncovered_by_reason"]
    assert set(reasons) <= {"never_in_view", "blocked_by_part", "blocked_by_fixture",
                           "blocked_by_base", "only_poses_that_collide"}
    assert sum(reasons.values()) == report["uncovered_points"]
    assert len(report["leak_regions"]) <= 12
    assert report["uncovered_points"] == int((~planned["selection"].covered).sum())


def test_the_definition_block_records_the_topological_rule(planned):
    definition = planned["report"]["definition"]
    assert "topological" in definition["name"]
    assert definition["upper_area_mm2"] > 0 and definition["lower_area_mm2"] > 0
    assert definition["material_thickness_mm"] > 0
    assert "fixture_contact" in definition["not_target"]
    assert "distance" in definition["contact_rule"]


def test_the_report_records_a_witness_of_which_face_was_chosen(planned):
    # `upper`/`lower` name geometric sides, and which side gets which name depends on the
    # split's orientation seed - which nothing anchors to the station. The report has to
    # carry enough for a reviewer to check the choice rather than trust the word.
    definition = planned["report"]["definition"]
    assert len(definition["through_axis"]) == 3
    assert abs(float(np.linalg.norm(definition["through_axis"])) - 1.0) < 1e-6
    assert definition["face_naming_note"]
    assert "geometric" in definition["face_naming_note"].lower()


def test_the_exempted_region_is_reported_with_its_area(planned):
    exempt = planned["report"]["exempt"]
    assert exempt["points"] == len(planned["target"].exempt.points)
    assert exempt["area_mm2"] >= 0.0
    assert exempt["reason"] == "fixture_contact"


def test_report_records_the_config_source_and_the_ray_backend(planned):
    assert planned["report"]["config"]["source"].endswith("scanner_trackscan49.json")
    assert planned["report"]["runtime"]["ray_backend"] in {"embree", "trimesh-python"}


def test_the_incidence_histogram_spans_zero_to_the_incidence_limit(planned):
    histogram = planned["report"]["incidence_histogram"]
    assert len(histogram["counts"]) == 10
    assert histogram["bin_edges_deg"][0] == 0.0
    assert histogram["bin_edges_deg"][-1] == 45.0
    assert sum(histogram["counts"]) > 0


def test_the_acceptance_target_is_echoed_beside_the_achieved_figure(planned):
    """A coverage number without the figure it was accepted against cannot be audited."""
    report = planned["report"]
    program = report["program"]
    assert program["coverage_target_pct"] == planned["config"].target.coverage_target_pct
    assert program["delivery_mode"] == planned["config"].target.delivery_mode
    assert "target" in " ".join(summary_lines(report))


def test_the_connector_route_mix_is_reported_only_for_a_real_path(planned):
    report = planned["report"]
    # no stroke stage here, so there is no path and no routes to count
    assert "connector_levels" not in report["program"]
    assert report["program"]["ordering"].startswith("planner order of single poses")


def test_the_three_coverage_levels_are_reported_separately(planned):
    """One number cannot say whether the gap is the selection, the connector or the arm."""
    layers = planned["report"]["coverage_layers"]
    assert layers["geometric_pool_pct"] >= layers["delivered_pct"]
    assert layers["gaps_pct"]["pool_to_delivered"] >= 0.0
    # this fixture delivers single poses with no path, so the third level is UNKNOWN - and
    # that is what it must say, not a copy of the second
    assert layers["execution_verified"] is False
    assert layers["execution_verified_pct"] is None
    assert "no claim about the arm" in layers["execution_note"]


def test_with_a_path_but_no_arm_the_third_level_points_at_the_switch(planned):
    from scanplan.path import BranchAssignment, PathPlan

    path = PathPlan(points=[], joint_planned=False, branches=BranchAssignment())
    block = coverage_layers_block(config=planned["config"], coverage=planned["coverage"],
                                  selection=planned["selection"], path=path)
    assert block["execution_verified"] is False
    assert "robot.enabled is false" in block["execution_note"]


def test_the_console_says_unverified_rather_than_repeating_the_headline(planned):
    text = "\n".join(summary_lines(planned["report"]))
    assert "coverage layers" in text
    assert "execution UNVERIFIED" in text


def test_a_joint_planned_path_turns_the_third_level_into_a_number(planned):
    """With the arm checked, all three levels exist and the last equals the delivered one."""
    from scanplan.path import BranchAssignment, PathPlan

    config = replace(planned["config"],
                     robot=replace(planned["config"].robot, enabled=True))
    path = PathPlan(points=[], joint_planned=True, branches=BranchAssignment())
    block = coverage_layers_block(config=config, coverage=planned["coverage"],
                                  selection=planned["selection"], path=path)
    assert block["execution_verified"] is True
    assert block["execution_verified_pct"] == block["delivered_pct"]
    assert block["gaps_pct"]["delivered_to_execution"] == 0.0


def test_a_point_the_arm_cannot_hold_invalidates_the_third_level(planned):
    """A delivered point with no solution means the delivered figure counts unreachable area."""
    from scanplan.path import BranchAssignment, PathPlan

    config = replace(planned["config"], robot=replace(planned["config"].robot, enabled=True))
    path = PathPlan(points=[], joint_planned=True,
                    branches=BranchAssignment(failed=[3]))
    block = coverage_layers_block(config=config, coverage=planned["coverage"],
                                  selection=planned["selection"], path=path)
    assert block["execution_verified"] is False
    assert "no joint solution" in block["execution_note"]


def test_the_program_cost_separates_the_convertible_half_from_the_rest(cfg):
    """Only the linear half converts to seconds; the joint half stays in degrees.

    Converting degrees to seconds needs a joint speed, and none has been measured on this cell.
    Reporting only the convertible half would hide the cost of the repositions, so the degrees
    are reported as degrees and the note says why.
    """
    from types import SimpleNamespace

    from scanplan.path import BranchAssignment, PathPlan, PathPoint
    from scanplan.report.program import program_block

    down = np.array([0.0, 0.0, -1.0])

    def corner(x, motion):
        return PathPoint(position=np.array([x, 0.0, 0.0]), axis=down, roll_deg=0.0,
                         role="coverage", source="greedy", stroke_id=0, motion=motion)

    first = SimpleNamespace(joints_deg=(0.0, 0.0, 0.0, 0.0, -60.0, 0.0))
    second = SimpleNamespace(joints_deg=(0.0, 0.0, 0.0, 0.0, -60.0, 90.0))
    path = PathPlan(points=[corner(0.0, "J"), corner(100.0, "L"), corner(300.0, "J")],
                    branches=BranchAssignment(solutions=[first, first, second]))
    cost = program_block(cfg, [], path)["cost"]
    assert cost["linear_segments"] == 1 and cost["linear_mm"] == 100.0
    assert cost["linear_seconds_at_floor_speed"] == pytest.approx(
        100.0 / cfg.motion.speed_mm_s, abs=0.1)
    assert cost["joint_segments"] == 1
    assert cost["joint_move_deg_total"] == pytest.approx(90.0)
    assert cost["joint_move_deg_worst"] == pytest.approx(90.0)
    assert "seconds" in cost["note"]


def test_the_failure_attribution_keeps_the_three_layers_apart(planned):
    """Sample losses, item drops and refused primitives are different facts."""
    block = planned["report"]["failure_attribution"]
    assert block["sample_reasons"] == planned["report"]["uncovered_by_reason"]
    assert block["item_drop_reasons"] == {}
    assert block["connector_primitive_refusals"] == {}
    assert set(block["layers_note"]) == {"sample_reasons", "item_drop_reasons",
                                        "connector_primitive_refusals"}
    assert block["joint_gate_enabled"] is False


def _refusing_gate():
    """A gate over an arm too small to hold any pose in the fixture, so it refuses them all."""
    params = RobotParams.disabled()
    return KinematicGate(RobotParams(**{**params.__dict__, "enabled": True,
                                       "upper_arm_mm": 1.0, "forearm_mm": 2.0,
                                       "forearm_offset_mm": 1.0}))


def test_samples_only_a_joint_infeasible_pose_can_see_are_blamed_on_the_arm(planned):
    """The report must ask the gate the planner used, or the bucket names the wrong fix.

    Selection is stopped at 1% so plenty of visible samples stay uncovered, and every pose is
    refused by the gate. None of those samples may then be reported as `candidate_found`,
    which means "a usable candidate exists and was not picked" - a scoring problem - while
    the truth here is reachability, whose fix is the arm policy or the layout.
    """
    coverage = planned["coverage"]
    selection = select(coverage, coverage.area_mm2, coverage_target_pct=1.0)
    report = build_report(config=planned["config"], model=planned["model"],
                          station=planned["station"], split=planned["split"],
                          target=planned["target"], coverage=coverage,
                          selection=selection, viewpoints=[],
                          kinematics=_refusing_gate())
    reasons = report["uncovered_by_reason"]
    assert reasons.get("unreachable_joint_limit", 0) > 0
    assert reasons.get("candidate_found", 0) == 0


def test_the_report_says_which_deliverable_it_is(planned):
    strokes = planned["report"]["stroke_plan"]
    assert strokes["enabled"] is False
    assert "same result v1 produces" in strokes["note"]


def test_the_approach_block_is_reported_but_never_screens(planned):
    """It answers "does this plan ask the arm to come from underneath?" and changes nothing.

    Asserting the second half matters more than the first: a number this close to a
    constraint will be read as one, so the planner's own output must be identical with and
    without it.
    """
    report = planned["report"]
    block = report["approach"]
    assert block["report_only"] is True
    assert block["poses"]["total"] == len(planned["viewpoints"])
    for key in ("base_top_z_mm", "part_bottom_z_mm", "clearance_part_to_base_mm"):
        assert isinstance(block[key], float)
    target = block["target"]
    assert 0.0 <= target["area_facing_down_pct"] <= 100.0

    again = build_report(config=planned["config"], model=planned["model"],
                         station=planned["station"], split=planned["split"],
                         target=planned["target"], coverage=planned["coverage"],
                         selection=planned["selection"], viewpoints=planned["viewpoints"])
    for key in ("coverage_pct", "viewpoints", "uncovered_points", "pool_reach_pct"):
        assert again[key] == report[key]


def test_the_j6_block_counts_the_segments_it_says_it_counts(planned):
    """The lever arm is derived, the histogram adds up, and the worst step is where it says."""
    block = planned["report"]["kinematics"]["j6"]
    tool = np.array(planned["config"].robot.tool_xyz_mm, dtype=float)
    assert block["lever_arm_mm"] == pytest.approx(float(np.hypot(tool[0], tool[1])), abs=0.01)
    assert block["report_deg"] == planned["config"].robot.j6_report_deg
    if not block["available"]:
        assert "fewer than two" in block["reason"]
        return
    assert sum(block["histogram"].values()) == block["segments"]
    assert 1 <= block["worst_step_at"] <= block["segments"]
    assert block["dominant_segments"] <= block["segments"]


def test_the_summary_ends_with_the_caveat(planned):
    lines = summary_lines(planned["report"])
    text = "\n".join(lines)
    assert "coverage" in text.lower()
    assert "robot" in text.lower()
    assert "roboguide" in lines[-1].lower()


def test_write_report_creates_readable_json(tmp_path, planned):
    path = write_report(tmp_path, planned["report"])
    assert path.exists() and path.name == "coverage_report.json"
    assert json.loads(path.read_text(encoding="utf-8"))["viewpoints"] >= 1


def test_write_samples_csv_carries_the_reasons(tmp_path, planned):
    uncovered = ~planned["selection"].covered
    reasons = attribute_uncovered(
        planned["target"].samples.points[uncovered],
        planned["target"].samples.normals[uncovered], planned["poses"],
        planned["station"], planned["model"], planned["config"])
    path = write_samples_csv(tmp_path / "unreachable.csv",
                             planned["target"].samples.points[uncovered],
                             planned["target"].samples.normals[uncovered], reasons)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "x_mm,y_mm,z_mm,nx,ny,nz,reason"
    assert len(lines) == 1 + int(uncovered.sum())


def test_the_viewpoint_records_carry_the_export_contract_keys(planned):
    record = planned["viewpoints"][0].as_dict()
    for key in ("id", "position", "axis", "roll_deg", "n_points", "area_mm2",
                "mean_incidence_deg", "max_incidence_deg", "tracker_ok", "source"):
        assert key in record
    assert len(record["position"]) == 3 and len(record["axis"]) == 3
    assert record["source"] in {"greedy", "repair"}
    assert np.linalg.norm(record["axis"]) == pytest.approx(1.0, rel=1e-9)
    assert record["max_incidence_deg"] >= record["mean_incidence_deg"] >= 0.0


def test_a_viewpoint_that_covers_nothing_is_not_emitted(planned):
    assert all(viewpoint.n_points > 0 for viewpoint in planned["viewpoints"])


def test_the_report_is_json_serialisable(planned):
    text = json.dumps(planned["report"])
    assert "NaN" not in text and "Infinity" not in text
