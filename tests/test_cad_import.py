"""CAD upload conversion: mesh formats via trimesh, STEP/IGES via OpenCASCADE."""

from __future__ import annotations

import pytest
import trimesh

from scanweb.cad_import import mesh_from_bytes, ocp_available, to_stl_bytes


def test_mesh_formats_round_trip_through_stl():
    box = trimesh.creation.box(extents=[10.0, 5.0, 2.0])

    mesh = mesh_from_bytes(box.export(file_type="stl"), "part.stl")

    assert len(mesh.faces) == 12
    assert len(to_stl_bytes(mesh)) > 0


def test_obj_upload_is_read_as_a_mesh():
    box = trimesh.creation.box(extents=[10.0, 5.0, 2.0])
    payload = box.export(file_type="obj")
    if isinstance(payload, str):  # trimesh 3.x exports OBJ as text
        payload = payload.encode("utf-8")

    mesh = mesh_from_bytes(payload, "part.obj")

    assert len(mesh.faces) == 12


def test_native_prt_is_rejected_with_an_actionable_message():
    with pytest.raises(ValueError, match="STEP"):
        mesh_from_bytes(b"\x00\x01\x02", "part.prt")


def test_unknown_extension_lists_the_supported_formats():
    with pytest.raises(ValueError, match="STL"):
        mesh_from_bytes(b"data", "part.unknown")


@pytest.mark.skipif(not ocp_available(), reason="cadquery-ocp is not installed")
def test_step_file_is_tessellated_in_model_units(tmp_path):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    path = tmp_path / "box.step"
    writer = STEPControl_Writer()
    writer.Transfer(BRepPrimAPI_MakeBox(100.0, 50.0, 10.0).Shape(), STEPControl_AsIs)
    writer.Write(str(path))

    mesh = mesh_from_bytes(path.read_bytes(), "box.step")

    assert len(mesh.faces) == 12
    assert mesh.bounds[1] == pytest.approx([100.0, 50.0, 10.0])


@pytest.mark.skipif(not ocp_available(), reason="cadquery-ocp is not installed")
def test_iges_file_is_tessellated(tmp_path):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.IGESControl import IGESControl_Writer

    path = tmp_path / "box.igs"
    writer = IGESControl_Writer()
    writer.AddShape(BRepPrimAPI_MakeBox(20.0, 20.0, 20.0).Shape())
    writer.ComputeModel()
    writer.Write(str(path))

    mesh = mesh_from_bytes(path.read_bytes(), "box.igs")

    assert len(mesh.faces) == 12
