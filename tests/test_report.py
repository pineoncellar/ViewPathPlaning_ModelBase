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

import numpy as np
import pytest

from scanplan.candidates import generate
from scanplan.coverage import attribute_uncovered, evaluate
from scanplan.report import (
    build_report,
    cluster_points,
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
