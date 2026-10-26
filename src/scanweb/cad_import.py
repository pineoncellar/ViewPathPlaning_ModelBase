"""Uploaded CAD file -> triangle mesh.

STL / OBJ / PLY / OFF / 3MF / GLB are read by trimesh. STEP / IGES / BREP go
through OpenCASCADE (`cadquery-ocp`), following `reference/IGES_to_STL.py`: each
B-rep face is tessellated directly instead of healing the shell first, which is
what the workshop IGES exports need.

Native CAD formats (`.prt`, `.sldprt`, `.catpart`, ...) have no free reader and
are rejected with an actionable message.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import trimesh

TRIMESH_SUFFIXES = {".stl", ".obj", ".ply", ".off", ".3mf", ".glb", ".gltf"}
OCP_SUFFIXES = {".step", ".stp", ".iges", ".igs", ".brep"}
NATIVE_SUFFIXES = {".prt", ".sldprt", ".catpart", ".x_t", ".x_b", ".sat"}
SUPPORTED_SUFFIXES = TRIMESH_SUFFIXES | OCP_SUFFIXES | NATIVE_SUFFIXES

#: value for an <input type="file" accept="...">
ACCEPT_ATTRIBUTE = ",".join(sorted(SUPPORTED_SUFFIXES))

FORMAT_HINT = (
    "支持 STL、OBJ、PLY、OFF、3MF、GLB、STEP/STP、IGES/IGS、BREP；"
    "PRT、SLDPRT、CATPart 等原生格式请先在 CAD 中导出为 STEP 或 IGES。"
)

_MAX_FACES = 400_000
_MAX_ATTEMPTS = 4
_ANGLE = 0.5


def ocp_available() -> bool:
    try:
        import OCP.STEPControl  # noqa: F401
    except Exception:
        return False
    return True


def suffix_of(filename: str) -> str:
    return Path(filename or "").suffix.lower()


def mesh_from_bytes(data: bytes, filename: str) -> trimesh.Trimesh:
    """Read any supported CAD/mesh file into a single triangle mesh."""
    if not data:
        raise ValueError("上传的文件为空")
    suffix = suffix_of(filename)
    if suffix in TRIMESH_SUFFIXES:
        return _trimesh_mesh(data, suffix)
    if suffix in OCP_SUFFIXES:
        return _ocp_mesh(data, suffix)
    if suffix in NATIVE_SUFFIXES:
        raise ValueError(
            f"无法直接读取 {suffix.upper()}（原生 CAD 格式）。"
            "请在 CAD 软件中导出为 STEP（.stp/.step）或 IGES（.igs/.iges）后再导入。"
        )
    raise ValueError(f"不支持的文件格式 {suffix or '(无扩展名)'}。{FORMAT_HINT}")


def to_stl_bytes(mesh: trimesh.Trimesh) -> bytes:
    return mesh.export(file_type="stl")


def _trimesh_mesh(data: bytes, suffix: str) -> trimesh.Trimesh:
    # trimesh 3.x resolves the loader from the file suffix; passing a BytesIO for
    # text formats such as OBJ fails, so round-trip through a temporary file.
    try:
        with tempfile.TemporaryDirectory(prefix="scanplan-mesh-") as folder:
            path = Path(folder) / f"model{suffix}"
            path.write_bytes(data)
            loaded = trimesh.load(path, force="mesh")
    except Exception as error:  # trimesh raises many different exception types
        raise ValueError(f"读取 {suffix.upper()} 失败：{error}") from error
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise ValueError(f"{suffix.upper()} 中没有可用的三角面")
    if not np.isfinite(loaded.vertices).all():
        raise ValueError(f"{suffix.upper()} 包含非有限坐标")
    return loaded


def _read_shape(path: Path, suffix: str):
    from OCP.IFSelect import IFSelect_RetDone

    if suffix in {".step", ".stp"}:
        from OCP.STEPControl import STEPControl_Reader

        reader = STEPControl_Reader()
        status = reader.ReadFile(str(path))
        if status != IFSelect_RetDone:
            raise ValueError(f"STEP 读取失败（状态 {status}）")
        reader.TransferRoots()
        shape = reader.OneShape()
    elif suffix in {".iges", ".igs"}:
        from OCP.IGESControl import IGESControl_Reader

        reader = IGESControl_Reader()
        status = reader.ReadFile(str(path))
        if status != IFSelect_RetDone:
            raise ValueError(f"IGES 读取失败（状态 {status}）")
        reader.TransferRoots()
        shape = reader.OneShape()
    else:
        from OCP.BRep import BRep_Builder
        from OCP.BRepTools import BRepTools
        from OCP.TopoDS import TopoDS_Shape

        shape = TopoDS_Shape()
        if not BRepTools.Read_s(shape, str(path), BRep_Builder()):
            raise ValueError("BREP 读取失败")
    if shape.IsNull():
        raise ValueError("CAD 文件不包含可读的实体")
    return shape


def _initial_deflection(shape) -> float:
    """A tessellation tolerance scaled to the model, so large parts stay usable."""
    try:
        from OCP.Bnd import Bnd_Box
        from OCP.BRepBndLib import BRepBndLib

        box = Bnd_Box()
        BRepBndLib.Add_s(shape, box)
        if box.IsVoid():
            return 0.5
        xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
        diagonal = math.dist((xmin, ymin, zmin), (xmax, ymax, zmax))
        return max(0.1, diagonal / 800.0)
    except Exception:
        return 0.5


def _tessellate(shape, deflection: float) -> tuple[np.ndarray, np.ndarray]:
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from OCP.TopLoc import TopLoc_Location

    BRepMesh_IncrementalMesh(shape, deflection, False, _ANGLE, True)

    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    offset = 0
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        try:
            face = TopoDS.Face(explorer.Current())
            location = TopLoc_Location()
            triangulation = BRep_Tool.Triangulation_s(face, location)
            if triangulation is not None and triangulation.NbTriangles() > 0:
                points = np.empty((triangulation.NbNodes(), 3), dtype=float)
                for index in range(1, triangulation.NbNodes() + 1):
                    node = triangulation.Node(index)
                    points[index - 1] = (node.X(), node.Y(), node.Z())
                if not location.IsIdentity():
                    matrix = _trsf_matrix(location.Transformation())
                    points = points @ matrix[:3, :3].T + matrix[:3, 3]

                indices = np.empty((triangulation.NbTriangles(), 3), dtype=np.int64)
                for index in range(1, triangulation.NbTriangles() + 1):
                    first, second, third = triangulation.Triangle(index).Get()
                    indices[index - 1] = (first - 1, second - 1, third - 1)

                vertices.append(points)
                faces.append(indices + offset)
                offset += len(points)
        except Exception:
            # Damaged B-rep faces are skipped rather than failing the upload.
            pass
        explorer.Next()

    if not faces:
        raise ValueError("CAD 文件中没有可三角化的面")
    return np.vstack(vertices), np.vstack(faces)


def _trsf_matrix(transformation) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    for row in range(1, 4):
        for col in range(1, 4):
            matrix[row - 1, col - 1] = transformation.Value(row, col)
    translation = transformation.TranslationPart()
    matrix[0, 3] = translation.X()
    matrix[1, 3] = translation.Y()
    matrix[2, 3] = translation.Z()
    return matrix


def _ocp_mesh(data: bytes, suffix: str) -> trimesh.Trimesh:
    if not ocp_available():
        raise ValueError(
            "读取 STEP / IGES 需要 OpenCASCADE 绑定，请先安装：pip install cadquery-ocp"
        )
    with tempfile.TemporaryDirectory(prefix="scanplan-cad-") as folder:
        path = Path(folder) / f"model{suffix}"
        path.write_bytes(data)
        shape = _read_shape(path, suffix)

    deflection = _initial_deflection(shape)
    vertices, faces = _tessellate(shape, deflection)
    attempts = 1
    while len(faces) > _MAX_FACES and attempts < _MAX_ATTEMPTS:
        deflection *= 2.0
        vertices, faces = _tessellate(shape, deflection)
        attempts += 1

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.remove_unreferenced_vertices()
    if len(mesh.faces) == 0:
        raise ValueError("CAD 文件转换后没有可用的三角面")
    return mesh
