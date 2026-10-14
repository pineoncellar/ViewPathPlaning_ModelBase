"""Shared fixtures for the planner tests.

Keep every fixture SYNTHETIC - a folded plate of a few triangles, a box standing
in for the fixture. Customer geometry must never be committed, and the tests
have to be runnable from a fresh clone.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

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


def box(x0, x1, y0, y1, z0, z1):
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
    return mesh_from_triangles([triangle for q in quads for triangle in quad(*q)])


def folded_strip_triangles(thickness=THICKNESS_MM, width=PLATE_WIDTH_MM):
    """A thin plate folded once by 30 deg, with 3 mm-wide sheared rims all round.

    Three segments: flat, a 30 deg ramp, then flat again at the higher level. The
    side walls that the fold creates belong to a face of the blank; the rim faces
    (the cut cross-section) do not.
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

    def point(node, side, y):
        x, z = nodes[node] + side * node_normals[node] * (thickness / 2.0)
        return (x, y, z)

    triangles = []
    for i, _direction in enumerate(directions):
        normal = np.array([segment_normals[i][0], 0.0, segment_normals[i][1]])
        for side in (+1, -1):
            triangles += quad(point(i, side, y0), point(i + 1, side, y0),
                              point(i + 1, side, y1), point(i, side, y1),
                              normal * side)
        for y, outward in ((y0, (0, -1, 0)), (y1, (0, 1, 0))):
            triangles += quad(point(i, +1, y), point(i + 1, +1, y),
                              point(i + 1, -1, y), point(i, -1, y), outward)

    start = np.array([directions[0][0], 0.0, directions[0][1]])
    end = np.array([directions[-1][0], 0.0, directions[-1][1]])
    triangles += quad(point(0, +1, y0), point(0, -1, y0),
                      point(0, -1, y1), point(0, +1, y1), -start)
    last = len(nodes) - 1
    triangles += quad(point(last, +1, y0), point(last, -1, y0),
                      point(last, -1, y1), point(last, +1, y1), end)
    return triangles


@pytest.fixture(scope="session")
def flat_plate():
    """A 200 x 200 x 0.7 mm slab, top face at z = 0.7, normals welded."""
    return box(-100.0, 100.0, -100.0, 100.0, 0.0, THICKNESS_MM)


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
def station(station_description_path):
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


@pytest.fixture(scope="session")
def folded_plate():
    """The folded strip with welded vertices."""
    return mesh_from_triangles(folded_strip_triangles(), welded=True)


@pytest.fixture(scope="session")
def folded_plate_unmerged():
    """The same strip with every triangle's vertices kept separate, as CAD exports them."""
    return mesh_from_triangles(folded_strip_triangles(), welded=False)
