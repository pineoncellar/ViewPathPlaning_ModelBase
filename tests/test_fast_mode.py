"""The fast run: what it changes, what it refuses, and how loudly it says so.

To assert: an accurate run still hands the coverage stage the three meshes (the mode is
off by default and off must be bit-for-bit the old behaviour); with sight lines switched
off the mask is exactly `geometry_mask`; the preset touches only the sampling grid, the
face and the flag; a half-fast configuration is refused rather than silently reinterpreted;
and a fast run announces itself in the report, the scope note and the console summary,
while an accurate run says nothing about being fast.

The measured claim behind the preset - 3 mm/`face auto` 1724 s against 10 mm/`face auto`
223 s on the real part - is a property of the code path, not something a synthetic plate
can demonstrate, so it is recorded in `configs/fast.json` rather than asserted here.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from scanplan.config import (
    FAST_MIN_SAMPLE_GRID_MM,
    FAST_SAMPLE_GRID_MM,
    FastParams,
    fast_preset,
    override,
)
from scanplan.pipeline import plan
from scanplan.report import summary_lines
from scanplan.scanner_model import ScannerModel, ScannerPose, measure_mask
from scanplan.scene import casters_for


@pytest.fixture(scope="session")
def quick(cfg):
    """The shipped config without the refinement rounds, so `plan` stays quick."""
    return replace(cfg, scanner=replace(cfg.scanner, refine_rounds=0))


@pytest.fixture(scope="session")
def fast_cfg(quick):
    return fast_preset(quick)


def test_an_accurate_run_hands_the_coverage_stage_the_three_meshes(clamped_plate, cfg):
    """Off by default, and off means the sight lines are still tested."""
    assert cfg.fast == FastParams.disabled()
    assert sorted(casters_for(clamped_plate.station, cfg)) == ["base", "fixture", "part"]


def test_switching_off_occlusion_leaves_exactly_the_geometric_mask(clamped_plate, cfg,
                                                                 fast_cfg):
    """`nearest_across({})` answers infinity, so nothing can be blocked."""
    blind = replace(fast_cfg, fast=FastParams(enabled=True, no_occlusion=True))
    assert casters_for(clamped_plate.station, blind) == {}

    model = ScannerModel(cfg.scanner)
    pose = ScannerPose(position=np.array([0.0, 0.0, 300.0]),
                       axis=np.array([0.0, 0.0, -1.0]), roll_deg=0.0)
    points = np.asarray(clamped_plate.station.part.triangles.mean(axis=1), dtype=float)
    normals = np.asarray(clamped_plate.station.part.face_normals, dtype=float)

    without = measure_mask(model, pose, points, normals, {}, occlusion_tol_mm=0.0)
    assert np.array_equal(without, model.geometry_mask(pose, points, normals))
    # ... and the real station does block something, so the equality above is not vacuous.
    with_occlusion = measure_mask(model, pose, points, normals,
                                  clamped_plate.station.obstacles(), occlusion_tol_mm=0.0)
    assert without.sum() > with_occlusion.sum()


def test_the_preset_touches_only_the_grid_the_face_and_the_flag(cfg):
    fast = fast_preset(cfg)
    assert fast.target.sample_grid_mm == FAST_SAMPLE_GRID_MM
    assert fast.target.face == "upper"
    assert fast.fast.enabled and not fast.fast.no_occlusion
    assert fast.fast.layers == ("sample_grid", "single_face")
    assert fast.target.coverage_target_pct == cfg.target.coverage_target_pct
    assert fast.target.delivery_mode == cfg.target.delivery_mode
    assert fast.stroke == cfg.stroke and fast.scanner == cfg.scanner
    assert fast.robot == cfg.robot and fast.motion == cfg.motion


def test_the_preset_can_ask_for_no_occlusion(cfg):
    fast = fast_preset(cfg, no_occlusion=True)
    assert fast.fast.no_occlusion
    assert "no_occlusion" in fast.fast.layers


def test_a_coarser_grid_may_be_chosen_but_not_a_finer_one(cfg):
    assert fast_preset(cfg, grid=FAST_MIN_SAMPLE_GRID_MM).target.sample_grid_mm \
        == FAST_MIN_SAMPLE_GRID_MM
    with pytest.raises(ValueError, match="sample_grid_mm"):
        fast_preset(cfg, grid=FAST_MIN_SAMPLE_GRID_MM - 1.0)


def test_no_occlusion_outside_a_fast_run_is_refused(cfg):
    """Otherwise the meaning of "coverage" changes without the run saying so."""
    with pytest.raises(ValueError, match="no_occlusion"):
        override(cfg, fast={"enabled": False, "no_occlusion": True, "layers": ()})


def test_a_fast_run_at_the_shipped_grid_is_refused(cfg):
    """A "fast" run still on the 3 mm grid is slow to distrust, so it is a config error."""
    assert cfg.target.sample_grid_mm < FAST_MIN_SAMPLE_GRID_MM
    with pytest.raises(ValueError, match="sample_grid_mm"):
        override(cfg, fast={"enabled": True, "no_occlusion": False, "layers": ()})


def test_a_fast_run_says_so_in_the_report_and_on_the_console(clamped_plate, fast_cfg):
    result = plan(clamped_plate.station, fast_cfg, face="upper")
    report = result.report
    assert report["fast_mode"]["enabled"] is True
    assert report["fast_mode"]["sample_grid_mm"] == FAST_SAMPLE_GRID_MM
    assert report["fast_mode"]["occlusion_modelled"] is True
    assert "FAST RUN" in report["scope"]
    assert report["runtime"]["fast"] is True
    assert any(line.startswith("FAST MODE") for line in summary_lines(report))


def test_an_accurate_run_makes_no_such_claim(clamped_plate, quick):
    report = plan(clamped_plate.station, quick, face="upper").report
    assert report["fast_mode"] == {"enabled": False}
    assert "FAST" not in report["scope"]
    assert report["runtime"]["fast"] is False
    assert not any(line.startswith("FAST MODE") for line in summary_lines(report))


def test_a_fast_run_without_sight_lines_says_that_too(clamped_plate, quick):
    blind = fast_preset(quick, no_occlusion=True)
    report = plan(clamped_plate.station, blind, face="upper").report
    assert report["fast_mode"]["occlusion_modelled"] is False
    assert "NOT tested" in report["scope"]
    # Attribution would be inventing sight-line reasons nobody computed.
    assert set(report["uncovered_by_reason"]) <= {"not_attributed"}
