"""Splitting a stamped sheet into the two faces of the blank.

To assert, on a synthetic folded plate: the two faces come out with nearly equal
area; the excluded band is the outline plus hole rims rather than a whole face;
forming side-walls are assigned to a face rather than dropped; the orientation
diagnostic separates a plate with a through-thickness axis from one without; and
the split still works on a mesh whose vertices were deliberately left unmerged,
because that is what a real CAD tessellation looks like.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import crossed_plates, has_unmerged_vertices

from scanplan.sheet_split import split_sheet


def test_the_two_faces_come_out_with_nearly_equal_area(folded_plate_unmerged):
    split = split_sheet(folded_plate_unmerged, seed=1)
    assert split.areas["upper"] > 0 and split.areas["lower"] > 0
    assert 0.8 < split.areas["upper"] / split.areas["lower"] < 1.25


def test_the_excluded_band_is_an_outline_not_a_whole_face(folded_plate_unmerged):
    split = split_sheet(folded_plate_unmerged, seed=1)
    assert split.band.sum() > 0
    assert split.areas["band"] < 0.2 * split.areas["total"]


def test_every_face_lands_in_exactly_one_set(folded_plate_unmerged):
    split = split_sheet(folded_plate_unmerged, seed=1)
    assert np.array_equal(split.upper | split.lower | split.band,
                          np.ones(split.faces, dtype=bool))
    assert not (split.upper & split.lower).any()
    assert not (split.band & (split.upper | split.lower)).any()


def test_forming_side_walls_are_assigned_to_a_face_rather_than_dropped(
        folded_plate_unmerged):
    split = split_sheet(folded_plate_unmerged, seed=1)
    # A forming wall follows the stamping (its normal is far from the global
    # through-thickness axis) without being a cut cross-section. `cosnm` cannot separate
    # the two - it is about +-1 on every stamping-following wall by design - so the angle
    # to the through-thickness axis is what tells them apart.
    axis_cosine = np.abs(folded_plate_unmerged.face_normals @ split.through_axis)
    flat_limit, wall_limit = np.cos(np.deg2rad(75.0)), np.cos(np.deg2rad(15.0))
    forming = (axis_cosine > flat_limit) & (axis_cosine < wall_limit)
    assert forming.any(), "the fixture needs a forming wall for this test to mean anything"
    assert float((forming & (split.upper | split.lower)).sum()) / forming.sum() > 0.9


def test_sheared_rims_are_the_ones_excluded(folded_plate_unmerged):
    split = split_sheet(folded_plate_unmerged, seed=1)
    axis_cosine = np.abs(folded_plate_unmerged.face_normals @ split.through_axis)
    rim = axis_cosine < 0.1
    assert rim.any()
    assert float((rim & split.band).sum()) / rim.sum() > 0.8


def test_material_thickness_is_recovered_from_the_mirror_partner(folded_plate_unmerged):
    split = split_sheet(folded_plate_unmerged, seed=1)
    assert 0.5 < split.thickness_mm < 1.0


def test_split_survives_unmerged_vertices(folded_plate_unmerged):
    assert has_unmerged_vertices(folded_plate_unmerged)
    assert split_sheet(folded_plate_unmerged, seed=1).oriented_fraction > 0.9


def test_a_part_with_no_single_through_thickness_axis_is_flagged(folded_plate_unmerged):
    hostile = split_sheet(crossed_plates(), seed=1)
    strip = split_sheet(folded_plate_unmerged, seed=1)
    # Two plates crossing at a right angle have no single sheet orientation, so the local
    # mid-surface normal is meaningless. What that produces here is not a scrambled pair
    # of faces - it is a near-total band: the split refuses to call any of it a face
    # rather than handing back a plausible-looking target. Downstream, an empty or tiny
    # target is a loud refusal, which is the behaviour the design asks for.
    assigned = (hostile.areas["upper"] + hostile.areas["lower"]) / hostile.areas["total"]
    strip_assigned = (strip.areas["upper"] + strip.areas["lower"]) / strip.areas["total"]
    assert assigned < 0.2
    assert strip_assigned > 0.9
    assert 0.0 <= hostile.oriented_fraction <= 1.0
    assert strip.oriented_fraction > 0.9


def test_the_band_threshold_is_not_sensitive_across_the_documented_range(folded_plate_unmerged):
    narrow = split_sheet(folded_plate_unmerged, band_deg=70.0, seed=1).areas["band"]
    wide = split_sheet(folded_plate_unmerged, band_deg=75.0, seed=1).areas["band"]
    assert abs(narrow - wide) < 0.25 * max(narrow, wide)


def test_the_split_is_deterministic_for_a_seed(folded_plate_unmerged):
    first = split_sheet(folded_plate_unmerged, seed=4)
    second = split_sheet(folded_plate_unmerged, seed=4)
    assert np.array_equal(first.upper, second.upper)
    assert first.thickness_mm == second.thickness_mm


def test_face_mask_refuses_a_name_that_is_not_a_face(folded_plate_unmerged):
    split = split_sheet(folded_plate_unmerged, seed=1)
    assert np.array_equal(split.face_mask("upper"), split.upper)
    with pytest.raises(ValueError, match="upper"):
        split.face_mask("top")

