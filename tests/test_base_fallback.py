"""The base fallback: the shipped station file, or a plate behind the scan face."""

from __future__ import annotations

import numpy as np
import trimesh

import scanweb


def test_the_station_base_is_used_when_it_lies_under_the_assembly(tmp_path):
    part = trimesh.creation.box(extents=[100.0, 2.0, 80.0])
    fixture = trimesh.creation.box(extents=[140.0, 60.0, 20.0])
    fixture.apply_translation([0.0, 0.0, -40.0])
    part.apply_translation([0.0, 0.0, -10.0])
    station_base = trimesh.creation.box(extents=[600.0, 600.0, 40.0])
    station_base.apply_translation([0.0, 0.0, -80.0])
    (tmp_path / "base.stl").write_bytes(station_base.export(file_type="stl"))

    base, source = scanweb._default_base(part, fixture, np.array([0.0, 0.0, 1.0]), tmp_path)

    assert source == "input/base.stl"
    assert base.extents[0] == 600.0


def test_a_station_base_in_front_of_the_scan_face_is_refused(tmp_path):
    """A base exported in another frame is a wall between the head and the part."""
    part = trimesh.creation.box(extents=[100.0, 80.0, 2.0])
    fixture = trimesh.creation.box(extents=[140.0, 120.0, 20.0])
    fixture.apply_translation([0.0, 0.0, -40.0])
    part.apply_translation([0.0, 0.0, -10.0])
    misplaced = trimesh.creation.box(extents=[600.0, 600.0, 40.0])
    misplaced.apply_translation([0.0, 0.0, 800.0])      # 800 mm up the scan axis
    (tmp_path / "base.stl").write_bytes(misplaced.export(file_type="stl"))

    base, source = scanweb._default_base(part, fixture, np.array([0.0, 0.0, 1.0]), tmp_path)

    assert source.startswith("reference plate behind the scan face")
    assert "input/base.stl is in front" in source
    assert float(base.bounds[1][2]) < float(part.bounds[0][2])
