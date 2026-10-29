"""Shared fixtures for the planner tests.

Keep every fixture SYNTHETIC - a folded plate of a few triangles, a box standing
in for the fixture. Customer geometry must never be committed, and the tests
have to be runnable from a fresh clone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import trimesh

from scanplan.scene import Station
from scanplan.sheet_split import SheetSplit

THICKNESS_MM = 0.7
PLATE_WIDTH_MM = 60.0


def quad(a, b, c, d, outward):
    """Two triangles for the quad a-b-c-d, wound so its normal follows `outward`."""
    a, b, c, d = (np.asarray(v, dtype=float) for v in (a, b, c, d))
    outward = np.asarray(outward, dtype=float)
    if float(np.dot(np.cross(b - a, d - a), outward)) < 0.0:
        a, b, c, d = d, c, b, a
    return [(a, b, c), (a, c, d)]


def mesh_from_triangles(triangles, welded=True):
    """A mesh from triangles, either with shared vertices or with each triangle isolated.

    The unmerged variant is what a CAD tessellation looks like: the seam points are
    not welded, so the mesh has hundreds of components instead of a handful.
    """
    if welded:
        vertices, faces, index = [], [], {}
        for triangle in triangles:
            ids = []
            for vertex in triangle:
                key = tuple(np.round(vertex, 9))
                if key not in index:
                    index[key] = len(vertices)
                    vertices.append(key)
                ids.append(index[key])
            faces.append(ids)
        return trimesh.Trimesh(vertices=np.array(vertices, dtype=float),
                               faces=np.array(faces, dtype=np.int64), process=True)

    vertices = np.array([v for triangle in triangles for v in triangle], dtype=float)
    faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def box_triangles(x0, x1, y0, y1, z0, z1):
    """Twelve triangles of an axis-aligned box, normals outward."""
    corner = {
        (0, 0, 0): (x0, y0, z0), (1, 0, 0): (x1, y0, z0),
        (1, 1, 0): (x1, y1, z0), (0, 1, 0): (x0, y1, z0),
        (0, 0, 1): (x0, y0, z1), (1, 0, 1): (x1, y0, z1),
        (1, 1, 1): (x1, y1, z1), (0, 1, 1): (x0, y1, z1),
    }
    quads = [
        (corner[0, 0, 0], corner[1, 0, 0], corner[1, 1, 0], corner[0, 1, 0], (0, 0, -1)),
        (corner[0, 0, 1], corner[1, 0, 1], corner[1, 1, 1], corner[0, 1, 1], (0, 0, 1)),
        (corner[0, 0, 0], corner[1, 0, 0], corner[1, 0, 1], corner[0, 0, 1], (0, -1, 0)),
        (corner[0, 1, 0], corner[1, 1, 0], corner[1, 1, 1], corner[0, 1, 1], (0, 1, 0)),
        (corner[0, 0, 0], corner[0, 1, 0], corner[0, 1, 1], corner[0, 0, 1], (-1, 0, 0)),
        (corner[1, 0, 0], corner[1, 1, 0], corner[1, 1, 1], corner[1, 0, 1], (1, 0, 0)),
    ]
    return [triangle for q in quads for triangle in quad(*q)]


def box(x0, x1, y0, y1, z0, z1):
    """An axis-aligned box mesh, normals outward."""
    return mesh_from_triangles(box_triangles(x0, x1, y0, y1, z0, z1))


def has_unmerged_vertices(mesh) -> bool:
    """True when no two triangles share a vertex, which is how CAD tessellations arrive.

    Tested by counting vertices rather than by splitting into components: `split()`
    reaches for networkx, which is not a dependency of this project.
    """
    return len(mesh.vertices) == 3 * len(mesh.faces)


def divided_quad(corner_00, corner_10, corner_11, corner_01, outward, spans, bands):
    """A quad split into `spans` x `bands` smaller quads, normal following `outward`."""
    corner_00 = np.asarray(corner_00, dtype=float)
    corner_10 = np.asarray(corner_10, dtype=float)
    corner_11 = np.asarray(corner_11, dtype=float)
    corner_01 = np.asarray(corner_01, dtype=float)
    triangles = []
    for i in range(spans):
        for j in range(bands):
            u0, u1 = i / spans, (i + 1) / spans
            v0, v1 = j / bands, (j + 1) / bands

            def at(u, v):
                return ((corner_00 * (1 - u) + corner_10 * u) * (1 - v)
                        + (corner_01 * (1 - u) + corner_11 * u) * v)

            triangles += quad(at(u0, v0), at(u1, v0), at(u1, v1), at(u0, v1), outward)
    return triangles


def plate_triangles(x0, x1, y0, y1, z0, z1, spans=10, bands=10):
    """A thin slab whose two blank faces are finely tessellated and whose rims are not.

    That is how a real part's tessellation is distributed, and it matters: the mesh
    resolution has to resolve the material thickness, so the rims must not dominate
    the face count.
    """
    triangles = []
    triangles += divided_quad((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
                              (0, 0, 1), spans, bands)
    triangles += divided_quad((x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                              (0, 0, -1), spans, bands)
    for *corners, outward in (
        ((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1), (0, -1, 0)),
        ((x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1), (0, 1, 0)),
        ((x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1), (-1, 0, 0)),
        ((x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1), (1, 0, 0)),
    ):
        triangles += quad(*corners, outward)
    return triangles


def crossed_plates():
    """Two thin plates crossing at a right angle, with no single through-thickness axis.

    The topology the split relies on (one local sheet normal for the whole blank) does
    not exist here, so the orientation step must flag it rather than invent an answer.
    """
    flat = box_triangles(0.0, 200.0, 0.0, 40.0, 0.0, THICKNESS_MM)
    upright = box_triangles(0.0, THICKNESS_MM, 0.0, 40.0, 0.0, 200.0)
    return mesh_from_triangles(flat + upright)


def folded_strip_triangles(thickness=THICKNESS_MM, width=PLATE_WIDTH_MM, spans=6, bands=4):
    """A thin plate folded once by 30 deg, with 3 mm-wide sheared rims all round.

    Three segments: flat, a 30 deg ramp, then flat again at the higher level. The
    side walls that the fold creates belong to a face of the blank; the rim faces
    (the cut cross-section) do not.

    The two blank faces are tessellated into a fine grid when `spans`/`bands` are
    raised, while the rims stay coarse. That is what a real CAD tessellation looks
    like, and it matters: a mesh whose rims carry a third of the triangles makes the
    paired-distances distribution say the material is 60 mm thick.
    """
    angle = np.deg2rad(30.0)
    directions = [np.array([1.0, 0.0]), np.array([np.cos(angle), np.sin(angle)]),
                  np.array([1.0, 0.0])]
    lengths = [60.0, 40.0, 40.0]
    nodes = [np.zeros(2)]
    for direction, length in zip(directions, lengths, strict=True):
        nodes.append(nodes[-1] + direction * length)

    segment_normals = [np.array([-d[1], d[0]]) for d in directions]
    node_normals = []
    for j in range(len(nodes)):
        if j == 0:
            normal = segment_normals[0]
        elif j == len(nodes) - 1:
            normal = segment_normals[-1]
        else:
            normal = segment_normals[j - 1] + segment_normals[j]
        node_normals.append(normal / np.linalg.norm(normal))

    y0, y1 = -width / 2.0, width / 2.0

    def surface_point(node_a, node_b, normal_a, normal_b, side, along, across):
        node = node_a + (node_b - node_a) * along
        normal = normal_a + (normal_b - normal_a) * along
        normal = normal / np.linalg.norm(normal)
        x, z = node + side * normal * (thickness / 2.0)
        return (x, y0 + (y1 - y0) * across, z)

    def rim_point(node, normal, side, across):
        x, z = node + side * normal * (thickness / 2.0)
        return (x, y0 + (y1 - y0) * across, z)

    def rim_quad(i, across):
        return quad(rim_point(nodes[i], node_normals[i], +1, across),
                    rim_point(nodes[i + 1], node_normals[i + 1], +1, across),
                    rim_point(nodes[i + 1], node_normals[i + 1], -1, across),
                    rim_point(nodes[i], node_normals[i], -1, across),
                    (0.0, 1.0, 0.0) if across > 0.5 else (0.0, -1.0, 0.0))

    triangles = []
    for i in range(len(directions)):
        normal = segment_normals[i]
        normal_3d = np.array([normal[0], 0.0, normal[1]])
        for side in (+1, -1):
            for span in range(spans):
                along_0, along_1 = span / spans, (span + 1) / spans
                for band in range(bands):
                    across_0, across_1 = band / bands, (band + 1) / bands
                    corners = [
                        surface_point(nodes[i], nodes[i + 1], node_normals[i],
                                      node_normals[i + 1], side, along, across)
                        for along, across in ((along_0, across_0), (along_1, across_0),
                                              (along_1, across_1), (along_0, across_1))
                    ]
                    triangles += quad(*corners, normal_3d * side)
        triangles += rim_quad(i, 0.0)
        triangles += rim_quad(i, 1.0)

    start = np.array([directions[0][0], 0.0, directions[0][1]])
    end = np.array([directions[-1][0], 0.0, directions[-1][1]])
    triangles += quad(rim_point(nodes[0], node_normals[0], +1, 0.0),
                      rim_point(nodes[0], node_normals[0], -1, 0.0),
                      rim_point(nodes[0], node_normals[0], -1, 1.0),
                      rim_point(nodes[0], node_normals[0], +1, 1.0), -start)
    last = len(nodes) - 1
    triangles += quad(rim_point(nodes[last], node_normals[last], +1, 0.0),
                      rim_point(nodes[last], node_normals[last], -1, 0.0),
                      rim_point(nodes[last], node_normals[last], -1, 1.0),
                      rim_point(nodes[last], node_normals[last], +1, 1.0), end)
    return triangles


@pytest.fixture(scope="session")
def flat_plate():
    """A 200 x 200 x 0.7 mm slab, top face at z = 0.7, with the two blank faces tessellated.

    Left as a 12-triangle box this would be useless to the split: each face contributes
    a single mid-surface point, so the local PCA has nothing to work with.
    """
    return mesh_from_triangles(plate_triangles(-100.0, 100.0, -100.0, 100.0, 0.0, THICKNESS_MM))


@pytest.fixture(scope="session")
def fixture_box():
    """A 120 x 120 x 55 mm box standing on the base: top face at z = -5, feet at z = -60."""
    return box(-60.0, 60.0, -60.0, 60.0, -60.0, -5.0)


@pytest.fixture(scope="session")
def base_box():
    """A 400 x 400 x 20 mm base, mounting plane (its top face) at z = -60."""
    return box(-200.0, 200.0, -200.0, 200.0, -80.0, -60.0)


def write_station_file(directory, part, fixture, base, *, transform_part=None,
                       transform_fixture=None, tracker_position_mm=None,
                       tracker_aim_mm=None, note="", extra=None):
    """Write the three meshes plus a station file next to them, and return its path.

    The mesh paths are relative on purpose: a station file is meant to be copied
    around with its models, and a real one lives under `input/` because it holds
    customer coordinates.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    names = {}
    for name, mesh in (("part", part), ("fixture", fixture), ("base", base)):
        path = directory / f"{name}.stl"
        mesh.export(path)
        names[name] = path.name

    identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    body = {
        "part": names["part"],
        "fixture": names["fixture"],
        "base": names["base"],
        "transform_part": transform_part or identity,
        "transform_fixture": transform_fixture or identity,
        "tracker_position_mm": tracker_position_mm,
        "tracker_aim_mm": tracker_aim_mm,
        "note": note,
    }
    if extra:
        body.update(extra)
    description = directory / "station.json"
    description.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return description


@pytest.fixture(scope="session")
def station_description_path(tmp_path_factory, folded_plate, fixture_box, base_box):
    """A station file on disk, in a directory of its own, with the three meshes beside it."""
    directory = tmp_path_factory.mktemp("station")
    return write_station_file(directory, folded_plate, fixture_box, base_box)


@pytest.fixture(scope="session")
def station(station_description_path) -> Station:
    """The loaded station every later task plans against."""
    from scanplan.scene import load_station, load_station_description

    description = load_station_description(station_description_path)
    return load_station(description, cloud_seed=3)


@pytest.fixture(scope="session")
def station_interpenetrating(tmp_path_factory, folded_plate, fixture_box, base_box):
    """The same station with the part driven 5 mm down into the clamped region."""
    from scanplan.scene import load_station, load_station_description

    directory = tmp_path_factory.mktemp("station-sunk")
    sunk = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -5.0], [0, 0, 0, 1]]
    path = write_station_file(directory, folded_plate, fixture_box, base_box,
                              transform_part=sunk)
    return load_station(load_station_description(path), cloud_seed=3)


@pytest.fixture(scope="session")
def station_fixture_sunk_into_base(tmp_path_factory, folded_plate, fixture_box, base_box):
    """A station whose fixture is driven 20 mm into the base - the frame is wrong."""
    from scanplan.scene import load_station, load_station_description

    directory = tmp_path_factory.mktemp("station-fixture-sunk")
    sunk = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -20.0], [0, 0, 0, 1]]
    path = write_station_file(directory, folded_plate, fixture_box, base_box,
                              transform_fixture=sunk)
    return load_station(load_station_description(path), cloud_seed=3)


CLAMP_FACE_GAP_MM = 5.0


@dataclass
class ClampedPlate:
    """A plate with a clamp pressing on its top face, as on the real cell.

    `probes` are two points on the same vertical, one on each face of the 0.7 mm sheet,
    so a test can show that the exemption is one-sided rather than merely distance based.
    """

    station: Station
    split: SheetSplit
    probes: np.ndarray
    probe_normals: np.ndarray


@pytest.fixture(scope="session")
def clamped_plate(tmp_path_factory, flat_plate, base_box):
    """Plate at z in [0, 0.7], a clamp reaching to 5 mm above it, base 60 mm below.

    The fixture is a C-clamp over the sheet: a column on the base, an arm across, and a
    small pad hanging down to 5 mm above the top face - which is how a real clamp
    contacts a part, through pads rather than over its whole footprint. A single
    120 x 120 block would exempt three quarters of the target face, and a pad reaching
    through the sheet would break the one-sidedness the exemption depends on.
    """
    from scanplan.scene import load_station, load_station_description
    from scanplan.sheet_split import split_sheet

    support = box_triangles(110.0, 130.0, -30.0, 30.0, -60.0, 40.0)
    arm = box_triangles(30.0, 130.0, -30.0, 30.0, 30.0, 40.0)
    pad_top = THICKNESS_MM + CLAMP_FACE_GAP_MM
    pad = box_triangles(50.0, 90.0, -30.0, 30.0, pad_top, 30.0)
    clamp = mesh_from_triangles(support + arm + pad)
    directory = tmp_path_factory.mktemp("station-clamped")
    path = write_station_file(directory, flat_plate, clamp, base_box)
    station = load_station(load_station_description(path), cloud_seed=3)
    # under the pad's footprint: the clamped face and the free face behind it
    probes = np.array([[70.0, 0.0, 0.0], [70.0, 0.0, THICKNESS_MM]])
    normals = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    return ClampedPlate(station=station, split=split_sheet(flat_plate, seed=3),
                        probes=probes, probe_normals=normals)


@pytest.fixture(scope="session")
def split_flat_plate(flat_plate):
    """The topological split of the flat plate, computed once."""
    from scanplan.sheet_split import split_sheet

    return split_sheet(flat_plate, seed=3)


@pytest.fixture(scope="session")
def cfg():
    """The shipped configuration, with the joint gate switched OFF.

    Not a convenience: every fixture in this file is synthetic geometry sitting at the
    robot's own origin - inside the shoulder envelope - so the shipped gate (on by default
    since 2026-10-27) rejects every pose and a coverage or stroke test would pass by
    measuring an empty plan. Kinematics has its own fixtures that build the gate
    explicitly (`test_kinematics.py`, `test_kinematic_gate.py`), and `test_config.py` pins
    the shipped default; this fixture is for everything else.
    """
    from dataclasses import replace

    from scanplan.config import load_config

    config = load_config()
    return replace(config, robot=replace(config.robot, enabled=False))


@pytest.fixture(scope="session")
def folded_plate():
    """The folded strip with welded vertices."""
    return mesh_from_triangles(folded_strip_triangles(), welded=True)


@pytest.fixture(scope="session")
def folded_plate_unmerged():
    """The same strip with every triangle's vertices kept separate, as CAD exports them."""
    return mesh_from_triangles(folded_strip_triangles(), welded=False)
