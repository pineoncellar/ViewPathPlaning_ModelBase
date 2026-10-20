"""The two command line entry points.

To assert: `plan` writes its outputs and returns zero; `ls` reads the viewpoint file and
writes the three export files without needing any meshes (re-exporting after a frame
change must be possible without re-planning); a bad program name, a missing mesh and an
unreadable config are usage errors with a readable message; and `--tracker-sweep` reports
the spread of the 192 placements rather than a single invented number.
"""

from __future__ import annotations

import json

import pytest
from conftest import box, write_station_file

from scanplan.cli import main


@pytest.fixture(scope="session")
def cli_station(tmp_path_factory, flat_plate, base_box):
    """A tiny station on disk, for driving the CLI the way a user would."""
    clamp = box(50.0, 90.0, -30.0, 30.0, 5.7, 30.0)
    directory = tmp_path_factory.mktemp("cli-station")
    write_station_file(directory, flat_plate, clamp, base_box)
    return directory


@pytest.fixture(scope="session")
def viewpoints_json(tmp_path_factory):
    """A hand-made viewpoint set in the delivered shape, with no meshes involved."""
    path = tmp_path_factory.mktemp("cli-ls") / "viewpoints.json"
    path.write_text(json.dumps([
        {"id": 0, "position": [10.0, 20.0, 300.0], "axis": [0.0, 0.0, -1.0],
         "roll_deg": 0.0, "n_points": 100, "area_mm2": 500.0,
         "mean_incidence_deg": 5.0, "max_incidence_deg": 12.0, "tracker_ok": True,
         "source": "greedy"},
        {"id": 1, "position": [40.0, 20.0, 300.0], "axis": [0.1, 0.0, -0.995],
         "roll_deg": 90.0, "n_points": 80, "area_mm2": 400.0,
         "mean_incidence_deg": 8.0, "max_incidence_deg": 20.0, "tracker_ok": True,
         "source": "greedy"},
    ]), encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def quick_override(tmp_path_factory):
    """An override that skips refinement, so the CLI tests are about the CLI.

    Refinement is expensive on purpose and is covered by the pipeline's own tests and by
    the real-station run; here it would only make the file slow.
    """
    path = tmp_path_factory.mktemp("cli-config") / "quick.json"
    path.write_text(json.dumps({"scanner": {"refine_rounds": 0}}), encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def sweep_override(tmp_path_factory):
    """As `quick_override`, with a coarse aim grid so the sweep ranks a small pool."""
    path = tmp_path_factory.mktemp("cli-config") / "sweep.json"
    path.write_text(json.dumps({"scanner": {"refine_rounds": 0, "aim_grid_mm": 200.0}}),
                    encoding="utf-8")
    return path


def test_plan_writes_the_outputs_and_returns_zero(cli_station, tmp_path):
    code = main(["plan", "--station", str(cli_station / "station.json"),
                 "--face", "upper", "--out", str(tmp_path)])
    assert code == 0
    assert (tmp_path / "viewpoints.json").exists()
    assert (tmp_path / "coverage_report.json").exists()
    assert (tmp_path / "viewpoints_3d.html").exists()
    assert (tmp_path / "summary.txt").read_text(encoding="utf-8")


def test_plan_can_be_driven_by_three_mesh_paths(cli_station, quick_override, tmp_path):
    code = main(["plan", "--part", str(cli_station / "part.stl"),
                 "--fixture", str(cli_station / "fixture.stl"),
                 "--base", str(cli_station / "base.stl"),
                 "--config", str(quick_override),
                 "--face", "lower", "--out", str(tmp_path)])
    assert code == 0
    report = json.loads((tmp_path / "coverage_report.json").read_text(encoding="utf-8"))
    assert report["face"] == "lower"


def test_ls_reads_the_viewpoint_file_and_writes_the_three_files(tmp_path, viewpoints_json):
    code = main(["ls", "--viewpoints", str(viewpoints_json), "--prog-name", "PART01",
                 "--out", str(tmp_path)])
    assert code == 0
    assert (tmp_path / "PART01.LS").exists()
    assert (tmp_path / "PART01_frame.json").exists()
    assert (tmp_path / "PART01_points.csv").exists()


def test_ls_needs_no_meshes_because_re_exporting_must_be_possible(tmp_path, viewpoints_json):
    assert main(["ls", "--viewpoints", str(viewpoints_json), "--prog-name", "REXPORT",
                 "--out", str(tmp_path)]) == 0
    text = (tmp_path / "REXPORT.LS").read_text(encoding="utf-8")
    assert "/PROG  REXPORT" in text
    assert "UFRAME_NUM=1" in text


def test_ls_accepts_a_placement_and_switches_to_uframe_zero(tmp_path, viewpoints_json):
    assert main(["ls", "--viewpoints", str(viewpoints_json), "--prog-name", "PLACED",
                 "--place", "0,0,0,0,0,0", "--out", str(tmp_path)]) == 0
    assert "UFRAME_NUM=0" in (tmp_path / "PLACED.LS").read_text(encoding="utf-8")


def test_view_rerenders_an_existing_run_without_planning(cli_station, viewpoints_json,
                                                         tmp_path):
    code = main(["view", "--viewpoints", str(viewpoints_json),
                 "--station", str(cli_station / "station.json"), "--out", str(tmp_path)])
    assert code == 0
    page = (tmp_path / "viewpoints_3d.html").read_text(encoding="utf-8")
    assert 'id="scene-data"' in page
    assert "http://" not in page and "https://" not in page


def test_view_needs_a_viewpoint_file(tmp_path):
    with pytest.raises(SystemExit) as caught:
        main(["view", "--viewpoints", str(tmp_path / "nope.json"), "--out", str(tmp_path)])
    assert caught.value.code == 2


def test_a_bad_program_name_is_a_usage_error(tmp_path, viewpoints_json, capsys):
    with pytest.raises(SystemExit) as caught:
        main(["ls", "--viewpoints", str(viewpoints_json), "--prog-name", "PART 01",
              "--out", str(tmp_path)])
    assert caught.value.code == 2
    assert "program name" in capsys.readouterr().err


def test_a_missing_mesh_is_a_usage_error(tmp_path):
    with pytest.raises(SystemExit) as caught:
        main(["plan", "--part", str(tmp_path / "nope.stl"), "--fixture", "f.stl",
              "--base", "b.stl", "--out", str(tmp_path)])
    assert caught.value.code == 2


def test_a_missing_viewpoint_file_is_a_usage_error(tmp_path):
    with pytest.raises(SystemExit) as caught:
        main(["ls", "--viewpoints", str(tmp_path / "nope.json"), "--prog-name", "X1",
              "--out", str(tmp_path)])
    assert caught.value.code == 2


def test_an_unreadable_config_is_a_usage_error(cli_station, tmp_path):
    broken = tmp_path / "config.json"
    broken.write_text(json.dumps({"scanner": {"max_incidence_degs": 45.0}}),
                      encoding="utf-8")
    with pytest.raises(SystemExit) as caught:
        main(["plan", "--station", str(cli_station / "station.json"),
              "--config", str(broken), "--out", str(tmp_path)])
    assert caught.value.code == 2


def test_a_config_override_is_honoured(cli_station, tmp_path):
    override = tmp_path / "override.json"
    override.write_text(json.dumps({"scanner": {"max_incidence_deg": 60.0}}),
                        encoding="utf-8")
    assert main(["plan", "--station", str(cli_station / "station.json"),
                 "--config", str(override), "--face", "upper", "--out", str(tmp_path)]) == 0
    report = json.loads((tmp_path / "coverage_report.json").read_text(encoding="utf-8"))
    assert report["placeholders"]["max_incidence_deg"] == 60.0
    assert report["config"]["override"].endswith("override.json")


def test_face_auto_records_both_faces(cli_station, quick_override, tmp_path):
    assert main(["plan", "--station", str(cli_station / "station.json"),
                 "--config", str(quick_override),
                 "--face", "auto", "--out", str(tmp_path)]) == 0
    report = json.loads((tmp_path / "coverage_report.json").read_text(encoding="utf-8"))
    assert set(report["face_candidates"]) == {"upper", "lower"}
    assert report["face"] in {"upper", "lower"}


def test_the_coverage_target_can_be_set_for_one_run(cli_station, tmp_path):
    assert main(["plan", "--station", str(cli_station / "station.json"),
                 "--face", "upper", "--coverage-target", "97.5",
                 "--out", str(tmp_path)]) == 0
    report = json.loads((tmp_path / "coverage_report.json").read_text(encoding="utf-8"))
    assert report["program"]["coverage_target_pct"] == 97.5
    assert "target 97.5%" in (tmp_path / "summary.txt").read_text(encoding="utf-8")


def test_the_coverage_target_can_be_switched_off_for_one_run(cli_station, tmp_path):
    assert main(["plan", "--station", str(cli_station / "station.json"),
                 "--face", "upper", "--coverage-target", "none",
                 "--out", str(tmp_path)]) == 0
    report = json.loads((tmp_path / "coverage_report.json").read_text(encoding="utf-8"))
    assert report["program"]["coverage_target_pct"] is None


def test_a_coverage_target_outside_its_range_is_a_usage_error(cli_station, tmp_path):
    for bad in ("0", "-1", "150", "soon"):
        with pytest.raises(SystemExit) as caught:
            main(["plan", "--station", str(cli_station / "station.json"),
                  "--face", "upper", "--coverage-target", bad, "--out", str(tmp_path)])
        assert caught.value.code == 2


def test_the_summary_prints_the_caveat(cli_station, tmp_path, capsys):
    assert main(["plan", "--station", str(cli_station / "station.json"),
                 "--face", "upper", "--out", str(tmp_path)]) == 0
    output = capsys.readouterr().out.lower()
    assert "robot" in output and "roboguide" in output


def test_the_tracker_sweep_reports_the_spread_of_the_placements(cli_station, sweep_override,
                                                              tmp_path, capsys):
    assert main(["plan", "--station", str(cli_station / "station.json"),
                 "--config", str(sweep_override),
                 "--tracker-sweep", "--out", str(tmp_path)]) == 0
    output = capsys.readouterr().out.lower()
    assert "tracker sweep" in output
    assert "best" in output and "worst" in output and "median" in output
