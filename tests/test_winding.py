"""Repairing a part whose back face was tessellated inside out.

To assert: a clean two-sided sheet is left exactly as it is; a sheet whose back face is
wound the same way as the front comes back with the two faces pointing away from each
other; and the measurement that decides it (one ray along the normal, one against it) also
reports the material thickness, so a run can say what it repaired and on what evidence.

The failure is not hypothetical: on station 20261030_100955 43 % of the uploaded part's
triangles were wound into the sheet, which made the split call the back face part of the
"upper" set, let the fixture-contact rule exempt a contiguous half of the picked face, and
left the pose ladder aiming through the panel.
"""

from __future__ import annotations

import numpy as np
import pytest

from scanplan.winding import repair_winding


def bottom_layer(triangles) -> np.ndarray:
    """The triangles of a plate that lie entirely in the z = 0 plane."""
    return np.asarray(triangles, dtype=float)[:, :, 2].max(axis=1) <= 1e-9


def reverse_bottom_layer(mesh):
    """The same plate with its back face wound inside out.

    This is the CAD failure mode: the front face is fine, the back face was tessellated
    with the opposite vertex order, so both faces end up facing the same way.
    """
    faces = np.array(mesh.faces, dtype=np.int64)
    flipped = bottom_layer(mesh.triangles)
    faces[flipped] = faces[flipped][:, ::-1]
    return type(mesh)(vertices=np.asarray(mesh.vertices).copy(), faces=faces, process=False)


def test_a_clean_sheet_is_left_alone(flat_plate):
    repaired, info = repair_winding(flat_plate)
    assert repaired is flat_plate
    assert info["flipped_faces"] == 0
    assert info["applied"] is False


def test_the_material_thickness_is_measured_from_the_two_rays(flat_plate):
    _, info = repair_winding(flat_plate)
    assert info["thickness_mm"] == pytest.approx(0.7, abs=0.05)


def test_a_back_face_wound_the_same_way_is_turned_round(flat_plate):
    broken = reverse_bottom_layer(flat_plate)
    back = bottom_layer(broken.triangles)
    assert (np.asarray(broken.face_normals)[back][:, 2] > 0.9).all(), \
        "the fixture must start with the fault in it: the back face faces up"

    repaired, info = repair_winding(broken)

    assert info["applied"] is True
    assert info["flipped_faces"] > 0
    assert info["share"] == pytest.approx(0.5, abs=0.15)
    assert info["remaining_share"] < 0.02
    normals = np.asarray(repaired.face_normals)
    back = bottom_layer(repaired.triangles)
    front = ~back
    assert (normals[back][:, 2] < -0.9).all(), "the back face must now face away"
    assert (normals[front][:, 2] > -0.9).all(), "nothing else may be turned round"


def test_the_repair_keeps_the_triangle_count_and_order(flat_plate):
    broken = reverse_bottom_layer(flat_plate)
    repaired, _ = repair_winding(broken)
    assert len(repaired.faces) == len(flat_plate.faces)
    assert np.allclose(np.asarray(repaired.vertices), np.asarray(flat_plate.vertices))
