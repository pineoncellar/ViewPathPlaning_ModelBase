"""Load the station models and put them into ONE frame.

The station is part + fixture + base + tracker. All of them must live in a
single CAD frame before anything else runs, because every later number
(occlusion, clearances, the exported pose) is defined in that frame.

The CAD handover may arrive with the assembly placement LOST - files exported
one by one, each sitting at its own origin or rotated relative to the base. That
silently invalidates everything downstream and still looks plausible, so the
alignment has to be explicit, reproducible and self-checked: fixture feet on the
base mounting surface, part lying flat, nothing interpenetrating, fixture
footprint inside the base.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .geometry import sample_surface_seeded

IDENTITY_4X4 = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))
_MODEL_KEYS = ("part", "fixture", "base")
_OPTIONAL_KEYS = ("transform_part", "transform_fixture", "tracker_position_mm",
                  "tracker_aim_mm", "note")
_RIGID_TOL = 1e-6
_COS30 = float(np.cos(np.deg2rad(30.0)))


@dataclass(frozen=True)
class StationDescription:
    """Where the models are and how they sit in the station frame.

    The transforms are data, never code: they carry customer coordinates, so they
    live in a station file under the ignored `input/` folder rather than in `src/`.
    """

    part: Path
    fixture: Path
    base: Path
    transform_part: tuple[tuple[float, ...], ...] = IDENTITY_4X4
    transform_fixture: tuple[tuple[float, ...], ...] = IDENTITY_4X4
    tracker_position_mm: tuple[float, float, float] | None = None
    tracker_aim_mm: tuple[float, float, float] | None = None
    note: str = ""
    source: Path | None = None


@dataclass
class Station:
    """The loaded station: three meshes in one frame, plus the clearance clouds.

    The fixture and base clouds are sampled once here so the body-clearance test
    is a KD-tree query and no later module has to re-sample anything.
    """

    part: trimesh.Trimesh
    fixture: trimesh.Trimesh
    base: trimesh.Trimesh
    description: StationDescription
    fixture_cloud_mm: np.ndarray
    base_cloud_mm: np.ndarray
    fixture_tree: cKDTree
    base_tree: cKDTree

    def clearance_mm(self, points) -> np.ndarray:
        """Distance from each point to the nearest fixture or base sample."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        if len(points) == 0:
            return np.zeros(0)
        to_fixture = self.fixture_tree.query(points, k=1)[0]
        to_base = self.base_tree.query(points, k=1)[0]
        return np.minimum(to_fixture, to_base)

    def nearest_obstacle_point(self, points) -> np.ndarray:
        """The obstacle sample that `clearance_mm` measured the distance to."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        if len(points) == 0:
            return np.zeros((0, 3))
        to_fixture, fixture_index = self.fixture_tree.query(points, k=1)
        to_base, base_index = self.base_tree.query(points, k=1)
        use_fixture = (to_fixture <= to_base)[:, None]
        return np.where(use_fixture, self.fixture_cloud_mm[fixture_index],
                        self.base_cloud_mm[base_index])

    def check_alignment(self, **kwargs) -> dict:
        """The self-checks of :func:`check_alignment`, bound to this station."""
        return check_alignment(self, **kwargs)


def read_transform(values) -> np.ndarray:
    """A 4x4 rigid transform, refusing anything that is not one."""
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"a station transform must be 4x4, got shape {matrix.shape}")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=_RIGID_TOL):
        raise ValueError("the last row of a station transform must be 0,0,0,1")
    rotation = matrix[:3, :3]
    orthonormal = np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
    if not orthonormal or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError(
            "not a rigid transform: the rotation part must be orthonormal with "
            f"determinant +1, got {rotation.tolist()}"
        )
    return matrix


def load_station_description(path: str | Path) -> StationDescription:
    """Read a station file. Model paths resolve against the file, not the CWD."""
    path = Path(path).resolve()
    with open(path, encoding="utf-8") as handle:
        body = json.load(handle)

    unknown = sorted(k for k in body if k not in _MODEL_KEYS + _OPTIONAL_KEYS
                     and not k.startswith("_"))
    if unknown:
        raise ValueError(f"unknown key(s) in the station file: {unknown}")
    missing = sorted(k for k in _MODEL_KEYS if k not in body)
    if missing:
        raise ValueError(f"the station file is missing {missing}")

    transforms = {}
    for key in ("transform_part", "transform_fixture"):
        values = body.get(key, IDENTITY_4X4)
        read_transform(values)
        transforms[key] = tuple(tuple(float(v) for v in row) for row in values)

    folder = path.parent
    return StationDescription(
        part=folder / body["part"],
        fixture=folder / body["fixture"],
        base=folder / body["base"],
        transform_part=transforms["transform_part"],
        transform_fixture=transforms["transform_fixture"],
        tracker_position_mm=_triple(body.get("tracker_position_mm")),
        tracker_aim_mm=_triple(body.get("tracker_aim_mm")),
        note=str(body.get("note", "")),
        source=path,
    )


def load_station(description: StationDescription, *, cloud_seed: int,
                 fixture_cloud_points: int = 200_000,
                 base_cloud_points: int = 200_000) -> Station:
    """Load the three models, realign part and fixture, and sample the obstacles."""
    part = apply_transform(_load_mesh(description.part, "part"),
                           read_transform(description.transform_part))
    fixture = apply_transform(_load_mesh(description.fixture, "fixture"),
                              read_transform(description.transform_fixture))
    base = _load_mesh(description.base, "base")

    fixture_cloud = sample_surface_seeded(
        fixture, _cloud_points(fixture, fixture_cloud_points), cloud_seed)[0]
    base_cloud = sample_surface_seeded(
        base, _cloud_points(base, base_cloud_points), cloud_seed + 1)[0]
    return Station(
        part=part, fixture=fixture, base=base, description=description,
        fixture_cloud_mm=fixture_cloud, base_cloud_mm=base_cloud,
        fixture_tree=cKDTree(fixture_cloud), base_tree=cKDTree(base_cloud),
    )


def check_alignment(station: Station, *, base_plane_mm: float | None = None,
                    plane_tol_mm: float = 3.0, foot_tol_mm: float = 5.0,
                    samples: int = 200_000, seed: int = 7,
                    touch_tolerance_mm: float | None = None) -> dict:
    """Do the three models actually sit together the way the station file claims?

    The answer is a dict of the quantities a human would measure plus a verdict.
    `touch_tolerance_mm` is the distance below which two surfaces are treated as
    touching; by default it is the fixture cloud's own resolution, because two
    point clouds cannot resolve a separation finer than their own spacing.
    """
    part_points, _ = _cloud(station.part, samples, seed)
    fixture_points, _ = _cloud(station.fixture, samples, seed + 1)
    base_points, _ = _cloud(station.base, samples, seed + 2)

    base_plane = (float(station.base.bounds[1][2]) if base_plane_mm is None
                  else float(base_plane_mm))

    fixture_tree = cKDTree(fixture_points)
    base_tree = cKDTree(base_points)

    part_to_fixture = float(fixture_tree.query(part_points, k=1)[0].min())
    fixture_to_base = float(base_tree.query(fixture_points, k=1)[0].min())
    part_to_base = float(base_tree.query(part_points, k=1)[0].min())

    # "Feet" is the bottom band, not only its downward-facing surface: the corner
    # where a foot meets the side wall is still the foot, and leaving it in the body
    # made this check fire on a fixture that is sitting correctly.
    feet = fixture_points[:, 2] <= fixture_points[:, 2].min() + foot_tol_mm
    body = ~feet
    body_to_base = (float(base_tree.query(fixture_points[body], k=1)[0].min())
                    if body.any() else float("inf"))

    contact = np.abs(fixture_points[:, 2] - base_plane) <= plane_tol_mm
    below = int((fixture_points[:, 2] < base_plane - plane_tol_mm).sum())

    fixture_bounds = np.asarray(station.fixture.bounds)
    base_bounds = np.asarray(station.base.bounds)
    inside = bool(np.all(fixture_bounds[0][:2] >= base_bounds[0][:2])
                  and np.all(fixture_bounds[1][:2] <= base_bounds[1][:2]))

    part_bounds = np.asarray(station.part.bounds)
    extent = part_bounds[1] - part_bounds[0]
    shares = _orientation_shares(station.part)

    touch = (touch_tolerance_mm if touch_tolerance_mm is not None
             else _cloud_spacing(fixture_points))

    warnings = []
    if below:
        warnings.append(
            f"the fixture dips below the mounting plane ({below} sampled points) - "
            "the mounting plane is probably too low"
        )
    if body_to_base < 0.25:
        warnings.append(
            f"the fixture body comes within {body_to_base:.2f} mm of the base - "
            "possible interference"
        )
    if not inside:
        warnings.append("the fixture bounding box is not fully inside the base footprint")
    if part_to_base < plane_tol_mm:
        warnings.append(f"the part comes within {part_to_base:.2f} mm of the base")
    if part_to_fixture < max(0.25, touch):
        warnings.append(
            f"the part and the fixture come within {part_to_fixture:.2f} mm of each "
            f"other, below the {touch:.2f} mm the clouds can resolve - possible interference"
        )

    return {
        "part_flat_in_z": bool(extent[2] < 0.35 * max(extent[0], extent[1])),
        "part_top_face_share_pct": round(shares["z"], 2),
        "part_dominant_axis": max(("x", "y", "z"), key=lambda axis: shares[axis]),
        "min_distance_part_to_fixture_mm": round(part_to_fixture, 3),
        "min_distance_fixture_to_base_mm": round(fixture_to_base, 3),
        "min_distance_fixture_BODY_to_base_mm": round(body_to_base, 3),
        "min_distance_part_to_base_mm": round(part_to_base, 3),
        "fixture_points_in_contact_with_mounting_plane": int(contact.sum()),
        "fixture_contact_area_mm2": round(float(contact.mean() * station.fixture.area), 1),
        "fixture_points_below_mounting_plane": below,
        "fixture_footprint_fits_on_base": inside,
        "touch_tolerance_mm": round(touch, 3),
        "mounting_plane_z_mm": round(base_plane, 3),
        "point_counts": {
            "part": int(len(part_points)),
            "fixture": int(len(fixture_points)),
            "base": int(len(base_points)),
        },
        "warnings": warnings,
        "passed": not warnings,
    }


def apply_transform(mesh, matrix) -> trimesh.Trimesh:
    """A copy of `mesh` with the rigid transform applied."""
    moved = mesh.copy()
    moved.apply_transform(np.asarray(matrix, dtype=float))
    return moved


def _load_mesh(path: Path, name: str) -> trimesh.Trimesh:
    if not Path(path).exists():
        raise FileNotFoundError(f"{name} mesh not found: {path}")
    loaded = trimesh.load(str(path))
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)
    return loaded


def _triple(values):
    if values is None:
        return None
    return tuple(float(v) for v in values)


def _cloud_points(mesh, cap: int) -> int:
    return int(min(cap, max(10_000, int(float(mesh.area) / 4.0))))


def _cloud(mesh, cap: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    return sample_surface_seeded(mesh, _cloud_points(mesh, cap), seed)


def _cloud_spacing(points: np.ndarray) -> float:
    """Typical nearest-neighbour distance: the finest separation these points resolve."""
    if len(points) < 2:
        return 0.0
    distances = cKDTree(points).query(points, k=2)[0][:, 1]
    return float(np.median(distances))


def _orientation_shares(mesh) -> dict[str, float]:
    """Area share per axis and sign, plus the per-axis total, in percent."""
    normals = np.asarray(mesh.face_normals)
    areas = np.asarray(mesh.area_faces)
    total = float(areas.sum())
    shares = {}
    for index, axis in enumerate("xyz"):
        positive = float(areas[normals[:, index] >= _COS30].sum()) / total * 100.0
        negative = float(areas[normals[:, index] <= -_COS30].sum()) / total * 100.0
        shares[axis] = positive + negative
        shares[f"+{axis.upper()}"] = positive
        shares[f"-{axis.upper()}"] = negative
    return shares
