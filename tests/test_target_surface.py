"""Sampling the target face, and exempting what must not be measured.

To assert: one sample per grid cell with the per-sample areas adding up to the face;
sampling is reproducible from its seed and needs no mesh connectivity; the clamp
exemption is one-sided, so a point on the far face of the 0.7 mm sheet survives; the
exempted set is kept with its own reason so a human can check it; and the two faces
of the blank never share a sample.
"""

from __future__ import annotations

import numpy as np
import pytest

from scanplan.geometry import grid_cells
from scanplan.target_surface import (
    ScanRegion,
    build_target,
    face_mesh,
    fixture_contact_mask,
    sample_face,
)


def whole_face(mesh):
    return np.ones(len(mesh.faces), dtype=bool)


def test_samples_are_one_per_grid_cell_and_their_areas_sum_to_the_face(flat_plate):
    samples = sample_face(flat_plate, whole_face(flat_plate), grid_mm=10.0, seed=0)
    cells = grid_cells(samples.points, 10.0)
    assert len(np.unique(cells, axis=0)) == len(samples.points)
    assert pytest.approx(float(samples.area_mm2.sum()), rel=1e-9) == samples.total_area_mm2
    assert pytest.approx(samples.total_area_mm2, rel=1e-6) == float(flat_plate.area)
    assert np.allclose(samples.area_mm2, samples.area_mm2[0])


def test_a_finer_grid_yields_more_samples(flat_plate):
    coarse = sample_face(flat_plate, whole_face(flat_plate), grid_mm=20.0, seed=0)
    fine = sample_face(flat_plate, whole_face(flat_plate), grid_mm=10.0, seed=0)
    assert len(fine.points) > 3 * len(coarse.points)


def test_sampling_is_deterministic_for_a_seed(flat_plate):
    mask = whole_face(flat_plate)
    first = sample_face(flat_plate, mask, grid_mm=20.0, seed=11)
    second = sample_face(flat_plate, mask, grid_mm=20.0, seed=11)
    assert np.array_equal(first.points, second.points)
    assert np.array_equal(first.normals, second.normals)


def test_sampling_does_not_need_connectivity(folded_plate_unmerged):
    assert len(folded_plate_unmerged.vertices) == 3 * len(folded_plate_unmerged.faces)
    samples = sample_face(folded_plate_unmerged, whole_face(folded_plate_unmerged),
                          grid_mm=5.0, seed=2)
    assert len(samples.points) > 50
    assert np.allclose(np.linalg.norm(samples.normals, axis=1), 1.0)


def test_face_mesh_carries_only_the_selected_faces(flat_plate, split_flat_plate):
    upper = face_mesh(flat_plate, split_flat_plate.upper)
    assert pytest.approx(float(upper.area), rel=1e-6) == split_flat_plate.areas["upper"]
    with pytest.raises(ValueError, match="no faces"):
        face_mesh(flat_plate, np.zeros(len(flat_plate.faces), dtype=bool))


def test_contact_exemption_is_one_sided(clamped_plate):
    # probe 0 is the underside of the sheet, on the far side from the clamp; probe 1 is the
    # top face, 5 mm under the clamp's contact face. Both are the same distance from the
    # fixture cloud, so only the outward test can tell them apart.
    mask = fixture_contact_mask(clamped_plate.probes, clamped_plate.probe_normals,
                                clamped_plate.station, distance_mm=8.0)
    assert mask.tolist() == [False, True]


def test_contact_exemption_is_off_beyond_the_threshold(clamped_plate):
    mask = fixture_contact_mask(clamped_plate.probes, clamped_plate.probe_normals,
                                clamped_plate.station, distance_mm=0.5)
    assert not mask.any()


def test_the_clamped_region_comes_out_of_the_target_with_its_own_reason(clamped_plate, cfg):
    target = build_target(clamped_plate.station.part, clamped_plate.split, "upper",
                          clamped_plate.station, cfg)
    assert len(target.exempt.points) > 0
    assert set(target.exempt.reasons) == {"fixture_contact"}
    assert target.samples.reasons is None or "fixture_contact" not in set(
        target.samples.reasons)
    assert target.samples.total_area_mm2 < target.face_area_mm2


def test_the_face_area_is_reported_independently_of_what_was_exempted(
        clamped_plate, cfg, split_flat_plate):
    target = build_target(clamped_plate.station.part, split_flat_plate, "upper",
                          clamped_plate.station, cfg)
    assert pytest.approx(target.face_area_mm2, rel=1e-6) == split_flat_plate.areas["upper"]
    assert target.face == "upper"


def test_the_two_faces_of_the_blank_do_not_share_a_sample(clamped_plate, cfg,
                                                         split_flat_plate):
    upper = build_target(clamped_plate.station.part, split_flat_plate, "upper",
                         clamped_plate.station, cfg)
    lower = build_target(clamped_plate.station.part, split_flat_plate, "lower",
                         clamped_plate.station, cfg)
    assert (upper.samples.normals[:, 2] > 0.9).all()
    assert (lower.samples.normals[:, 2] < -0.9).all()
    assert upper.samples.points[:, 2].min() > lower.samples.points[:, 2].max()


def test_bare_target_params_need_an_explicit_seed(clamped_plate, cfg):
    # the advertised signature accepts TargetParams, and those carry no seed - sampling
    # draws from the global RNG, so the seed has to be supplied rather than invented
    with pytest.raises(ValueError, match="seed"):
        build_target(clamped_plate.station.part, clamped_plate.split, "upper",
                     clamped_plate.station, cfg.target)
    target = build_target(clamped_plate.station.part, clamped_plate.split, "upper",
                          clamped_plate.station, cfg.target, seed=3)
    assert len(target.samples.points) > 0


def test_only_the_upper_face_is_exempted_where_the_clamp_presses(clamped_plate, cfg,
                                                                split_flat_plate):
    upper = build_target(clamped_plate.station.part, split_flat_plate, "upper",
                         clamped_plate.station, cfg)
    lower = build_target(clamped_plate.station.part, split_flat_plate, "lower",
                         clamped_plate.station, cfg)
    assert len(upper.exempt.points) > 0
    assert len(lower.exempt.points) == 0


def pick(x0, x1, y0, y1, *, z=0.7, normal=(0.0, 0.0, 1.0), step=5.0, holes=()):
    """A picked scan face over the plate's own coordinates, as the page sends it.

    z defaults to the flat plate's top face; the fixture that builds it is in conftest.
    The page sends the picked face's own triangles, so this tiles the rectangle at `step`
    and leaves the holes untiled - exactly what clicking a face with holes produces.
    """
    origin = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0, z], dtype=float)
    triangles = []
    for x in np.arange(x0, x1 - 1e-9, step):
        for y in np.arange(y0, y1 - 1e-9, step):
            centre_x, centre_y = x + step / 2.0, y + step / 2.0
            if any(hx0 <= centre_x <= hx1 and hy0 <= centre_y <= hy1
                   for hx0, hx1, hy0, hy1 in holes):
                continue
            triangles.append([[x, y, z], [x + step, y, z], [x + step, y + step, z]])
            triangles.append([[x, y, z], [x + step, y + step, z], [x, y + step, z]])
    return ScanRegion(triangles=np.array(triangles, dtype=float),
                      normal=np.array(normal, dtype=float), origin=origin,
                      tolerance_mm=0.5)


def test_a_supplementary_face_on_the_far_side_is_added_to_the_target(clamped_plate, cfg):
    # the plate's `upper` target is the top face; picking the bottom face must ADD it,
    # not replace the top one
    plain = build_target(clamped_plate.station.part, clamped_plate.split, "upper",
                         clamped_plate.station, cfg)
    under = pick(-50.0, 50.0, -50.0, 50.0, z=0.0, normal=(0.0, 0.0, -1.0))
    target = build_target(clamped_plate.station.part, clamped_plate.split, "upper",
                          clamped_plate.station, cfg, regions=[under])
    assert pytest.approx(target.supplementary_mm2, rel=0.02) == 100.0 * 100.0
    assert target.supplementary_regions == 1
    assert target.face_area_mm2 > plain.face_area_mm2
    # the main face is untouched: every top-face sample is still there
    top = target.samples.points[:, 2] > 0.6
    assert top.sum() >= plain.samples.points.shape[0] * 0.5
    assert (target.samples.points[~top, 2] < 0.1).all()
    # nothing is exempted for being unpicked: the only exemption is the clamped region
    assert set(target.exempt.reasons) == {"fixture_contact"}


def test_a_pick_on_the_main_face_adds_nothing_instead_of_narrowing(clamped_plate, cfg):
    plain = build_target(clamped_plate.station.part, clamped_plate.split, "upper",
                         clamped_plate.station, cfg)
    region = pick(-50.0, 50.0, -50.0, 50.0)
    target = build_target(clamped_plate.station.part, clamped_plate.split, "upper",
                          clamped_plate.station, cfg, regions=[region])
    assert target.supplementary_mm2 == pytest.approx(0.0)
    assert target.face_area_mm2 == pytest.approx(plain.face_area_mm2)
    assert target.samples.total_area_mm2 == pytest.approx(plain.samples.total_area_mm2)


def test_a_pick_that_covers_nothing_is_an_error_rather_than_an_empty_plan(clamped_plate, cfg):
    region = pick(400.0, 500.0, 400.0, 500.0)
    with pytest.raises(ValueError, match="supplementary faces"):
        build_target(clamped_plate.station.part, clamped_plate.split, "upper",
                     clamped_plate.station, cfg, regions=[region])
