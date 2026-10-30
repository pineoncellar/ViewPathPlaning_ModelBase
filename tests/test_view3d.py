"""The 3D view: one self-contained file that carries the part and every viewpoint.

To assert: the page references no network asset, so it opens on a shop-floor PC; the
embedded buffers decode back to the same vertices; a viewpoint's goal point and
footprint follow the planner's own scanner frame and footprint; only models over the
display budget are decimated; the viewpoint records a `viewpoints.json` file holds
are accepted directly, without any planning objects; and the per-sample cover map
answers "which viewpoints see this point".
"""

from __future__ import annotations

import base64
import json
import re

import numpy as np
import pytest
from conftest import mesh_from_triangles, plate_triangles

from scanplan.geometry import scanner_frame
from scanplan.scanner_model import ScannerModel
from scanview import (
    TARGET_FILE_NAME,
    build_payload,
    coverage_mask,
    decimate_for_display,
    load_target_samples,
    save_target_samples,
    write_viewer,
)


def read_payload(path):
    """The page text and its embedded scene payload."""
    text = path.read_text(encoding="utf-8")
    match = re.search(r'<script id="scene-data" type="application/json">(.*?)</script>',
                      text, re.S)
    assert match, "the page has no embedded scene payload"
    return text, json.loads(match.group(1))


def decode(record, key, dtype):
    return np.frombuffer(base64.b64decode(record[key]), dtype=dtype)


def viewpoint(position=(0.0, 0.0, 200.0), axis=(0.0, 0.0, -1.0), **extra):
    record = {"id": 0, "position": list(position), "axis": list(axis), "roll_deg": 0.0,
              "n_points": 10, "area_mm2": 100.0, "mean_incidence_deg": 5.0,
              "max_incidence_deg": 10.0, "tracker_ok": True, "source": "greedy"}
    record.update(extra)
    return record


def test_the_page_is_one_self_contained_file(tmp_path, flat_plate, cfg):
    path = write_viewer(tmp_path / "view.html", part=flat_plate,
                        viewpoints=[viewpoint()], scanner=cfg.scanner)
    text, payload = read_payload(path)
    assert "http://" not in text and "https://" not in text
    assert "<script src" not in text.lower()
    assert payload["meshes"]["part"]["faces"] == len(flat_plate.faces)
    positions = decode(payload["meshes"]["part"], "pos", "<f4").reshape(-1, 3)
    assert positions.shape == flat_plate.vertices.shape
    assert np.allclose(positions, flat_plate.vertices, atol=1e-3)


def test_the_footprint_comes_from_the_planners_scanner_frame(flat_plate, cfg):
    axis = np.array([0.0, 0.0, -1.0])
    roll = 90.0
    position = np.array([10.0, -4.0, 400.0])
    payload = build_payload(part=flat_plate,
                            viewpoints=[viewpoint(position, axis, roll_deg=roll)],
                            scanner=cfg.scanner)
    record = payload["viewpoints"][0]

    frame = scanner_frame(axis, roll)
    width, height = ScannerModel(cfg.scanner).footprint_mm(cfg.scanner.standoff_mm)
    aim = position + axis * cfg.scanner.standoff_mm
    expected = [aim + frame[:, 0] * sx * width / 2 + frame[:, 1] * sy * height / 2
                for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))]

    assert np.allclose(record["aim"], aim, atol=1e-3)
    assert np.allclose(record["corners"], expected, atol=1e-3)
    for corner in record["corners"]:
        offset = np.asarray(corner) - position
        assert np.dot(offset, axis) == pytest.approx(cfg.scanner.standoff_mm, abs=1e-3)


def test_only_models_over_the_display_budget_are_decimated():
    dense = mesh_from_triangles(plate_triangles(-100.0, 100.0, -100.0, 100.0, 0.0, 0.7,
                                                spans=60, bands=60))
    assert len(dense.faces) > 2000
    assert decimate_for_display(dense, 10_000_000) is dense

    reduced = decimate_for_display(dense, 2000)
    assert 0 < len(reduced.faces) <= 2000
    assert len(reduced.vertices) < len(dense.vertices)


def test_target_samples_carry_their_covered_flag(flat_plate, cfg):
    points = np.array([[0.0, 0.0, 0.7], [1.0, 0.0, 0.7], [2.0, 0.0, 0.7]])
    payload = build_payload(part=flat_plate, viewpoints=[], target_points=points,
                            target_covered=[True, False, True], scanner=cfg.scanner)
    stored = decode(payload["target"], "points", "<f4").reshape(-1, 3)
    covered = decode(payload["target"], "covered", np.uint8)
    assert payload["target"]["coveredCount"] == 2
    assert payload["target"]["uncoveredCount"] == 1
    assert np.allclose(stored, points, atol=1e-5)
    assert covered.tolist() == [1, 0, 1]


def test_the_cover_map_says_which_viewpoints_see_a_sample(flat_plate, cfg):
    # pose 0 covers samples {0, 1}, pose 1 covers {1, 2}; the map must say exactly that
    mask = coverage_mask([np.array([0, 1]), np.array([1, 2])], [0, 1], 3)
    payload = build_payload(part=flat_plate,
                            viewpoints=[viewpoint(), viewpoint(id=1)],
                            target_points=np.zeros((3, 3)), target_covered=[1, 1, 1],
                            target_mask=mask, scanner=cfg.scanner)
    stored = decode(payload["target"], "mask", np.uint8).reshape(3, payload["target"]["stride"])
    assert payload["target"]["viewpoints"] >= 2
    see = [[column for column in range(2)
            if stored[row, column // 8] & (1 << (column % 8))] for row in range(3)]
    assert see == [[0], [0, 1], [1]]


def test_the_sample_sidecar_round_trips(tmp_path):
    points = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=float)
    covered = np.array([True, False])
    mask = coverage_mask([np.array([0])], [0], 2)
    path = save_target_samples(tmp_path / TARGET_FILE_NAME, points=points,
                               covered=covered, mask=mask)
    back_points, back_covered, back_mask = load_target_samples(path)
    assert np.allclose(back_points, points, atol=1e-6)
    assert back_covered.tolist() == [True, False]
    assert back_mask.tolist() == mask.tolist()


def test_the_delivered_viewpoint_records_are_accepted_as_they_are(tmp_path, flat_plate,
                                                                  cfg):
    records = [viewpoint(id=3, position=(1.0, 2.0, 3.0), source="repair")]
    payload = build_payload(part=flat_plate, viewpoints=records, scanner=cfg.scanner)
    assert payload["viewpoints"][0]["id"] == 3
    assert payload["viewpoints"][0]["source"] == "repair"
    assert len(payload["viewpoints"][0]["corners"]) == 4


def test_the_page_repaints_while_the_mouse_is_still_moving(tmp_path, flat_plate, cfg):
    """The redraw scheduler must not latch "a frame is already pending".

    A pending flag that starts true, or that is only cleared inside the frame it guards,
    turns every redraw request into a no-op - the page then looks frozen and only
    repaints when something calls `render()` directly (a pointerup). That is the failure
    this test exists to catch, and it cannot be seen in the rendered picture.
    """
    path = write_viewer(tmp_path / "view.html", part=flat_plate,
                        viewpoints=[viewpoint()], scanner=cfg.scanner)
    text = path.read_text(encoding="utf-8")
    assert "var framePending = false;" in text
    assert "needsRender" not in text
    assert 'id="fov-mode"' in text
    assert text.count("requestRender();") >= 4


def test_a_malformed_viewpoint_is_refused(flat_plate, cfg):
    with pytest.raises(ValueError, match="position"):
        build_payload(part=flat_plate,
                      viewpoints=[{"position": [1.0, 2.0], "axis": [0.0, 0.0, -1.0]}],
                      scanner=cfg.scanner)
