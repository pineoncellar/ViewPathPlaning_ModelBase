"""Convert the workshop IGES station models into meshes for the viewpoint pipeline.

Why this exists
---------------
PHASE_1 voxelises *triangle meshes* through trimesh, and trimesh cannot read
IGES (nor the native NX ``.prt``). The station models were handed over as
PolyWorks IGES exports, so they have to be meshed first. This script does that
through OpenCASCADE - either directly via OCP (``--backend ocp``, the default,
which needs no healing) or via gmsh (``--backend gmsh``, which does try to heal
and therefore fails on these particular exports). It optionally decimates the
result and prints the entity information needed to tell the part, the fixture
and the base apart - the file names were changed after export and must not be
trusted blindly.

Run it in its own environment; it deliberately does not touch the pinned
pipeline environment (``pyproject.toml``):

    uv run --no-project --python 3.11 \
        --with cadquery-ocp --with gmsh --with trimesh \
        --with fast-simplification --with numpy --with matplotlib \
        python IGES_to_STL.py --input-dir "<folder with .igs>" --preview

Coordinates are passed through unchanged. The workshop IGES files are in
millimetres, so the STL is in millimetres as well; PHASE_1 then needs
``obj_mesh_scale = 0.001`` (it expects the scaled mesh in metres, because
``tm_voxelSize = 0.01`` is what makes one voxel 10 mm).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time


# ---------------------------------------------------------------------------
# IGES header / entity information
# ---------------------------------------------------------------------------
def _split_iges_params(text: str) -> list[str]:
    """Split an IGES parameter string, honouring Hollerith (nH...) values."""
    out: list[str] = []
    current = ""
    i = 0
    while i < len(text):
        char = text[i]
        if char == ",":
            out.append(current)
            current = ""
            i += 1
            continue
        if char == ";":
            out.append(current)
            break
        j = i
        while j < len(text) and text[j].isdigit():
            j += 1
        if j > i and j < len(text) and text[j] in "Hh":
            length = int(text[i:j])
            current += text[j + 1 : j + 1 + length]
            i = j + 1 + length
            continue
        current += char
        i += 1
    return out


def read_iges_header(path: str) -> dict:
    """Read the Start/Global section: original file name, producer, units."""
    with open(path, "rb") as handle:
        head = handle.read(8000)
    # IGES records are 80 characters; the terminator (LF or CRLF) is not part of
    # them, so split on newlines instead of slicing fixed 80-byte blocks.
    records = [line.decode("latin-1") for line in head.splitlines()]
    global_data = "".join(r[:72] for r in records if len(r) >= 73 and r[72] == "G")
    if not global_data:
        return {"global_parsed": False}
    params = _split_iges_params(global_data)
    units_name = params[14] if len(params) > 14 else ""
    units_flag = params[13] if len(params) > 13 else ""
    return {
        "global_parsed": True,
        "product_id_from_sender": params[2] if len(params) > 2 else "",
        "iges_file_name": params[3] if len(params) > 3 else "",
        "native_system": params[4] if len(params) > 4 else "",
        "preprocessor": params[5] if len(params) > 5 else "",
        "model_space_scale": params[12] if len(params) > 12 else "",
        "units_flag": units_flag,
        "units_name": units_name,
        "units_is_mm": str(units_flag) == "2",
    }


# ---------------------------------------------------------------------------
# meshing through gmsh / OpenCASCADE
# ---------------------------------------------------------------------------
def _mesher_attempt(
    path: str, target_edge: float, tolerance: float, heal: bool, verbose: bool = False
):
    """One meshing attempt. Raises if the geometry cannot be prepared."""
    import gmsh
    import numpy as np

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.option.setNumber("General.Verbosity", 3 if verbose else 1)
        # These models are in millimetres over a few metres, so the default
        # 1e-8 sewing tolerance is far too tight: wires then look "unfixable"
        # and the whole import aborts. 1e-3 mm is a sane industrial value.
        gmsh.option.setNumber("Geometry.Tolerance", tolerance)
        if heal:
            gmsh.option.setNumber("Geometry.OCCFixSmallEdges", 1)
            gmsh.option.setNumber("Geometry.OCCFixSmallFaces", 1)
            gmsh.option.setNumber("Geometry.OCCFixDegenerated", 1)
            gmsh.option.setNumber("Geometry.OCCSewFaces", 1)
        else:
            # welding the faces into a valid shell is what fails on slightly
            # inconsistent exports; a triangle soup per face is enough for
            # voxelisation and ray casting, so healing is optional
            gmsh.option.setNumber("Geometry.OCCFixSmallEdges", 0)
            gmsh.option.setNumber("Geometry.OCCFixSmallFaces", 0)
            gmsh.option.setNumber("Geometry.OCCFixDegenerated", 0)
            gmsh.option.setNumber("Geometry.OCCSewFaces", 0)
        gmsh.open(path)

        counts = {}
        for dim in (0, 1, 2, 3):
            counts[dim] = len(gmsh.model.getEntities(dim))
        bbox = gmsh.model.getBoundingBox(-1, -1)

        gmsh.option.setNumber("Mesh.MeshSizeMin", target_edge)
        gmsh.option.setNumber("Mesh.MeshSizeMax", target_edge)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.model.mesh.generate(2)

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        vertices = np.asarray(node_coords, dtype=float).reshape(-1, 3)

        tag_array = np.asarray(node_tags, dtype=np.int64)
        lookup = np.full(int(tag_array.max()) + 1, -1, dtype=np.int64)
        lookup[tag_array] = np.arange(len(tag_array), dtype=np.int64)

        triangles = []
        element_types, _, element_nodes = gmsh.model.mesh.getElements(2)
        for element_type, nodes in zip(element_types, element_nodes):
            corners = np.asarray(nodes, dtype=np.int64)
            if element_type == 2:  # 3-node triangle
                triangles.append(corners.reshape(-1, 3))
            elif element_type == 3:  # 4-node quad -> two triangles
                quads = corners.reshape(-1, 4)
                triangles.append(quads[:, [0, 1, 2]])
                triangles.append(quads[:, [0, 2, 3]])
        if not triangles:
            raise RuntimeError("no 2D elements were generated")
        faces = lookup[np.vstack(triangles)]
        if (faces < 0).any():
            raise RuntimeError("element node tag not found in node list")

        return vertices, faces, counts, tuple(bbox)
    finally:
        gmsh.finalize()


def _gp_trsf_to_matrix(transformation):
    """gp_Trsf -> 4x4 numpy matrix."""
    import numpy as np

    matrix = np.eye(4, dtype=float)
    for row in range(1, 4):
        for col in range(1, 4):
            matrix[row - 1, col - 1] = transformation.Value(row, col)
    translation = transformation.TranslationPart()
    matrix[0, 3] = translation.X()
    matrix[1, 3] = translation.Y()
    matrix[2, 3] = translation.Z()
    return matrix


def mesh_iges_ocp(
    path: str,
    deflection: float,
    angle: float = 0.5,
    per_face: bool = False,
    verbose: bool = False,
):
    """Mesh an IGES file with OpenCASCADE directly, without healing.

    gmsh always tries to weld the imported faces into a valid shell, which is
    exactly the step that dies on these slightly inconsistent exports ("Could
    not fix wire in surface N"). Meshing does not need a valid shell, so the
    shape is tessellated as-is; faces that still fail to mesh are counted and
    skipped instead of aborting the whole conversion.
    """
    import numpy as np
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.IGESControl import IGESControl_Reader
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from OCP.TopLoc import TopLoc_Location

    reader = IGESControl_Reader()
    status = reader.ReadFile(path)
    if status != IFSelect_RetDone:
        raise RuntimeError(f"IGESControl_Reader.ReadFile returned status {status}")
    reader.TransferRoots()
    shape = reader.OneShape()
    if shape.IsNull():
        raise RuntimeError("IGES reader produced a null shape")

    if not per_face:
        BRepMesh_IncrementalMesh(shape, deflection, False, angle, True)

    vertices: list = []
    faces: list = []
    offset = 0
    stats = {"faces_total": 0, "faces_meshed": 0, "faces_empty": 0, "faces_failed": 0}

    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        stats["faces_total"] += 1
        try:
            face = TopoDS.Face(explorer.Current())
            if per_face:
                BRepMesh_IncrementalMesh(face, deflection, False, angle, True)
            location = TopLoc_Location()
            triangulation = BRep_Tool.Triangulation_s(face, location)
            if triangulation is None or triangulation.NbTriangles() == 0:
                stats["faces_empty"] += 1
            else:
                points = np.empty((triangulation.NbNodes(), 3), dtype=float)
                for index in range(1, triangulation.NbNodes() + 1):
                    node = triangulation.Node(index)
                    points[index - 1] = (node.X(), node.Y(), node.Z())
                if not location.IsIdentity():
                    matrix = _gp_trsf_to_matrix(location.Transformation())
                    points = points @ matrix[:3, :3].T + matrix[:3, 3]

                indices = np.empty((triangulation.NbTriangles(), 3), dtype=np.int64)
                for index in range(1, triangulation.NbTriangles() + 1):
                    first, second, third = triangulation.Triangle(index).Get()
                    indices[index - 1] = (first - 1, second - 1, third - 1)

                vertices.append(points)
                faces.append(indices + offset)
                offset += len(points)
                stats["faces_meshed"] += 1
        except Exception:
            stats["faces_failed"] += 1
        explorer.Next()

    if not faces:
        raise RuntimeError(
            f"no face could be triangulated ({stats['faces_total']} faces found, "
            f"{stats['faces_empty']} without triangles, {stats['faces_failed']} errored)"
        )
    all_vertices = np.vstack(vertices)
    bbox = (
        float(all_vertices[:, 0].min()),
        float(all_vertices[:, 1].min()),
        float(all_vertices[:, 2].min()),
        float(all_vertices[:, 0].max()),
        float(all_vertices[:, 1].max()),
        float(all_vertices[:, 2].max()),
    )
    return (
        all_vertices,
        np.vstack(faces),
        stats,
        bbox,
    )


def mesh_iges(
    path: str,
    target_edge: float,
    tolerance: float = 1e-3,
    deflection: float = 1.0,
    angle: float = 0.5,
    backend: str = "auto",
    verbose: bool = False,
):
    """Mesh an IGES file, trying the requested backend with fallbacks.

    These station exports are large trimmed B-reps with slightly inconsistent
    wires, so both readers are tried with progressively more forgiving settings.
    """
    import numpy as np

    attempts = []
    if backend in ("auto", "ocp"):
        attempts.append(("ocp", None))
        attempts.append(("ocp-per-face", "per-face"))
    if backend in ("auto", "gmsh"):
        attempts.append(("gmsh", ("heal", tolerance, True)))
        attempts.append(("gmsh", ("no-heal", tolerance, False)))
        attempts.append(("gmsh", ("no-heal-loose", max(tolerance * 100.0, 0.1), False)))
    if not attempts:
        raise ValueError(f"unknown backend {backend!r}")

    last_error: Exception | None = None
    for name, gmsh_settings in attempts:
        try:
            if name.startswith("ocp"):
                vertices, faces, stats, bbox = mesh_iges_ocp(
                    path,
                    deflection,
                    angle,
                    per_face=(name == "ocp-per-face"),
                    verbose=verbose,
                )
                stats = {"backend": name, **stats}
                return vertices, faces, stats, bbox, name
            attempt_tolerance, heal = gmsh_settings[1], gmsh_settings[2]
            vertices, faces, counts, bbox = _mesher_attempt(
                path, target_edge, attempt_tolerance, heal, verbose
            )
            stats = {
                "backend": "gmsh",
                "surfaces": counts[2],
                "curves": counts[1],
                "volumes": counts[3],
                "faces_total": counts[2],
            }
            return vertices, faces, stats, bbox, f"gmsh:{gmsh_settings[0]}"
        except Exception as error:
            last_error = error
            if verbose:
                detail = gmsh_settings[0] if isinstance(gmsh_settings, tuple) else ""
                print(f"  attempt {name} {detail} failed: {error}")
    raise last_error if last_error else RuntimeError("meshing failed")


def decimate(vertices, faces, keep_ratio: float):
    """Garland-Heckbert quadric decimation; keeps `keep_ratio` of the faces."""
    import fast_simplification
    import numpy as np

    if keep_ratio >= 1.0:
        return vertices, faces
    points_out, faces_out = fast_simplification.simplify(
        np.asarray(vertices, dtype=np.float64),
        np.asarray(faces, dtype=np.int32),
        target_reduction=1.0 - keep_ratio,
    )
    return np.asarray(points_out, dtype=float), np.asarray(faces_out, dtype=np.int64)


def describe(mesh) -> dict:
    """Geometry summary used to tell part / fixture / base apart."""
    import numpy as np

    bounds = mesh.bounds
    size = bounds[1] - bounds[0]
    try:
        components = len(mesh.split(only_watertight=False))
    except Exception:
        components = None
    return {
        "triangle_count": int(len(mesh.faces)),
        "vertex_count": int(len(mesh.vertices)),
        "bbox_min_mm": [round(float(v), 2) for v in bounds[0]],
        "bbox_max_mm": [round(float(v), 2) for v in bounds[1]],
        "size_mm": [round(float(v), 2) for v in size],
        "surface_area_mm2": round(float(mesh.area), 1),
        "volume_mm3": round(float(mesh.volume), 1),
        "connected_components": components,
        "is_watertight": bool(mesh.is_watertight),
    }


# ---------------------------------------------------------------------------
# preview rendering (matplotlib, headless)
# ---------------------------------------------------------------------------
def _sample_points(mesh, limit: int):
    import numpy as np

    points = np.asarray(mesh.vertices, dtype=float)
    if len(points) > limit:
        step = len(points) // limit + 1
        points = points[::step]
    return points


def _pick_cjk_font():
    """Model file names are Chinese, so prefer a font that can draw them."""
    import matplotlib.font_manager as font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in (
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "WenQuanYi Zen Hei",
    ):
        if name in available:
            return name
    return None


def render_preview(meshes: dict, out_path: str, limit: int = 40000) -> str:
    """Three orthographic projections of all models, in the shared CAD frame."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    font = _pick_cjk_font()
    if font:
        matplotlib.rcParams["font.family"] = font
    matplotlib.rcParams["axes.unicode_minus"] = False

    projections = [("X [mm]", "Y [mm]", 0, 1), ("X [mm]", "Z [mm]", 0, 2), ("Y [mm]", "Z [mm]", 1, 2)]
    figure, axes = plt.subplots(1, 3, figsize=(21, 7))
    colours = ["#d62728", "#2ca02c", "#1f77b4", "#9467bd", "#ff7f0e", "#8c564b"]
    for axis, (label_x, label_y, ix, iy) in zip(axes, projections):
        for (name, mesh), colour in zip(meshes.items(), colours):
            points = _sample_points(mesh, limit)
            axis.scatter(points[:, ix], points[:, iy], s=0.5, c=colour, label=name, linewidths=0)
        axis.set_xlabel(label_x)
        axis.set_ylabel(label_y)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, linewidth=0.3)
    axes[0].legend(markerscale=20, loc="best")
    figure.suptitle("station models in the shared CAD frame (mm)")
    figure.tight_layout()
    figure.savefig(out_path, dpi=110)
    plt.close(figure)
    return out_path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input-dir", required=True, help="folder holding the .igs files")
    parser.add_argument("--out-dir", default=None, help="mesh output folder (default: <input-dir>/mesh)")
    parser.add_argument("--pattern", default="*.igs", help="glob for the input files")
    parser.add_argument(
        "--target-edge-mm",
        type=float,
        default=5.0,
        help="target triangle edge length handed to the OpenCASCADE mesher",
    )
    parser.add_argument(
        "--keep-ratio",
        type=float,
        default=0.1,
        help="fraction of the faces kept after decimation",
    )
    parser.add_argument(
        "--decimate-above",
        type=int,
        default=200_000,
        help="only decimate meshes with more faces than this",
    )
    parser.add_argument("--preview", action="store_true", help="render PNG projections")
    parser.add_argument("--probe", action="store_true", help="header/entity info only, no meshing")
    parser.add_argument("--verbose", action="store_true", help="let gmsh print its progress")
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "ocp", "gmsh"],
        help="mesher: ocp (OpenCASCADE directly, no healing) or gmsh",
    )
    parser.add_argument(
        "--deflection-mm",
        type=float,
        default=1.0,
        help="linear deflection for the ocp backend (model units)",
    )
    parser.add_argument(
        "--angle-rad",
        type=float,
        default=0.5,
        help="angular deflection for the ocp backend",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="OCC sewing/fixing tolerance in model units (mm), gmsh backend only",
    )
    parser.add_argument("--report", default=None, help="write the JSON report here")
    args = parser.parse_args(argv)

    # meshing a trimmed B-rep takes minutes; do not sit on the progress output
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    # Windows globbing is case insensitive, so collect by normalised name and
    # keep the original spelling for display.
    seen: dict[str, str] = {}
    for pattern in (args.pattern, "*.iges"):
        for found in glob.glob(os.path.join(args.input_dir, pattern)):
            seen.setdefault(os.path.normcase(found), found)
    files = [seen[key] for key in sorted(seen)]
    if not files:
        print(f"no files matching {args.pattern} in {args.input_dir}", file=sys.stderr)
        return 2

    out_dir = args.out_dir or os.path.join(args.input_dir, "mesh")
    report: dict = {"input_dir": os.path.abspath(args.input_dir), "files": []}

    print(f"found {len(files)} file(s) in {args.input_dir}\n")
    for index, path in enumerate(files, start=1):
        name = os.path.basename(path)
        print("=" * 78)
        print(f"[{index}/{len(files)}] {name}  ({os.path.getsize(path) / 1e6:.1f} MB)")
        entry: dict = {"file": name, "size_mb": round(os.path.getsize(path) / 1e6, 1)}

        header = read_iges_header(path)
        entry["iges_header"] = header
        print(
            f"  header      : internal name={header.get('iges_file_name')!r} "
            f"id={header.get('product_id_from_sender')!r} "
            f"units={header.get('units_name')!r} "
            f"(mm={header.get('units_is_mm')})"
        )
        print(f"                producer={header.get('preprocessor')!r}")

        if args.probe:
            report["files"].append(entry)
            continue

        started = time.time()
        try:
            vertices, faces, stats, occ_bbox, strategy = mesh_iges(
                path,
                args.target_edge_mm,
                args.tolerance,
                args.deflection_mm,
                args.angle_rad,
                args.backend,
                verbose=args.verbose,
            )
        except Exception as error:  # keep going so one bad file is not fatal
            print(f"  MESH FAILED : {type(error).__name__}: {error}")
            entry["error"] = f"{type(error).__name__}: {error}"
            report["files"].append(entry)
            continue
        entry["open_and_mesh_seconds"] = round(time.time() - started, 1)
        entry["meshing_strategy"] = strategy
        entry["mesher_stats"] = stats
        entry["occ_bbox_mm"] = [round(float(v), 2) for v in occ_bbox]
        summary = " ".join(f"{k}={v}" for k, v in stats.items() if k != "backend")
        print(
            f"  meshed by   : {strategy}  ({summary})  "
            f"in {entry['open_and_mesh_seconds']} s"
        )
        if stats.get("faces_empty", 0) or stats.get("faces_failed", 0):
            print(
                f"  WARNING     : {stats.get('faces_empty', 0)} face(s) without "
                f"triangles, {stats.get('faces_failed', 0)} face(s) failed to mesh"
            )

        import numpy as np
        import trimesh

        raw = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=True)
        # drop duplicated/degenerate faces that the surface tessellation can leave behind
        raw.update_faces(raw.nondegenerate_faces())
        raw.remove_unreferenced_vertices()
        entry["mesh_before"] = describe(raw)
        print(
            f"  meshed      : {len(raw.faces):,} triangles, {entry['mesh_before']['connected_components']} shells, "
            f"size={entry['mesh_before']['size_mm']} mm"
        )

        final = raw
        if len(raw.faces) > args.decimate_above and args.keep_ratio < 1.0:
            vertices_d, faces_d = decimate(raw.vertices, raw.faces, args.keep_ratio)
            candidate = trimesh.Trimesh(vertices=vertices_d, faces=faces_d, process=True)
            if candidate.is_volume or candidate.volume > 0:
                final = candidate
            entry["decimation"] = {
                "keep_ratio": args.keep_ratio,
                "triangles_after": int(len(final.faces)),
            }
            print(
                f"  decimated   : {len(final.faces):,} triangles "
                f"({100.0 * len(final.faces) / max(len(raw.faces), 1):.1f}% kept)"
            )
        entry["mesh_after"] = describe(final)

        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(name)[0]
        stl_path = os.path.join(out_dir, stem + ".stl")
        final.export(stl_path, file_type="stl")
        entry["output_stl"] = os.path.abspath(stl_path)
        entry["output_stl_mb"] = round(os.path.getsize(stl_path) / 1e6, 2)
        print(f"  written     : {stl_path}  ({entry['output_stl_mb']} MB)")

        report["files"].append(entry)
        report.setdefault("_meshes", {})[name] = final

    if not args.probe and args.preview and report.get("_meshes"):
        preview_path = os.path.join(out_dir, "comparison.png")
        render_preview(report["_meshes"], preview_path)
        print(f"\npreview     : {preview_path}")

    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump({k: v for k, v in report.items() if k != "_meshes"}, handle, indent=2)
        print(f"report      : {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
