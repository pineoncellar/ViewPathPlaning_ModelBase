"""Split a stamped sheet into the two faces of its original flat blank.

The target definition is topological, not orientational:

    target   = one of the two faces of the blank
               (forming side-walls and pocket walls INCLUDED)
    excluded = the sheared (blanked / pierced) cross-section
    excluded = the other face

Rules that do NOT work, each for its own reason: `normal_z > 0` (breaks as soon
as the part is tilted); "one dihedral patch per face" (creases split a face into
several facets); an angle threshold against the part's through-thickness axis
(discards the forming side-walls, which belong to the face); a signed threshold
at 90 degrees (the side-walls are near-perpendicular, so their sign is noise);
and "drop the sheared edge then take connected components" (fails outright,
because the mesh is a patchwork - the CAD surfaces were tessellated separately
and coincident vertices were never merged).

What does work exploits the fact that the part is a thin shell: every point of a
face has a mirror point about one thickness away. Cast a ray inward, take the
pair's mid-point as a sample of the sheet MID-SURFACE, PCA those samples for the
local mid-surface normal `m`, then `|n . m| ~ 1` is a face of the blank while
`|n . m| ~ 0` is a sheared cross-section; orient `m` consistently and its sign
says which face. Purely local, so the patchwork mesh does not matter.

Self-checks: the two faces must have nearly equal area (two sides of one blank),
and the excluded band must form the outer silhouette plus the hole rims.

Known limit: orienting `m` assumes a mostly flat part. A part folded twice into
a deep U defeats the seeding step - that is the case the operator face-picker is
for. Steps up to the PCA still hold for any shape. `oriented_fraction` is how the
result says so: below about 0.9 the two faces are assigned by an arbitrary sign,
and nothing else about the output reveals that.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .geometry import sample_surface_seeded
from .raycast import RayCaster

_RAY_OFFSET_MM = 0.02
_MAX_PROPAGATION_ROUNDS = 40


@dataclass(frozen=True)
class SheetSplit:
    """The two faces of the blank, the sheared band between them, and the diagnostics.

    Areas are in mm2. `oriented_fraction` is the share of faces whose mid-surface
    normal could be oriented consistently; below roughly 0.9 the assignment of the
    two faces is not trustworthy.
    """

    upper: np.ndarray
    lower: np.ndarray
    band: np.ndarray
    cosnm: np.ndarray
    abs_cosnm: np.ndarray
    band_cos: float
    through_axis: np.ndarray
    thickness_mm: float
    areas: dict[str, float]
    oriented_fraction: float
    faces: int
    total_area_mm2: float

    def face_mask(self, face: str) -> np.ndarray:
        """The mask of `"upper"` or `"lower"`."""
        if face == "upper":
            return self.upper
        if face == "lower":
            return self.lower
        raise ValueError(f"face must be 'upper' or 'lower', not {face!r}")


def through_thickness_axis(part, *, samples: int = 200_000, seed: int = 7) -> np.ndarray:
    """The part's coarse through-thickness direction: its smallest surface spread."""
    count = int(min(samples, max(2_000, len(part.faces))))
    points, _ = sample_surface_seeded(part, count, seed)
    centred = points - points.mean(axis=0)
    _, vectors = np.linalg.eigh(np.cov(centred.T))
    return vectors[:, 0]


def split_sheet(part, *, band_deg: float = 70.0, knn: int = 24, seed_ratio: float = 0.5,
                prop_sim: float = 0.8, through_axis=None, samples: int = 200_000,
                seed: int = 7,
                log: Callable[[str], None] | None = None) -> SheetSplit:
    """Split `part`, a thin shell, into the two faces of the original flat blank."""
    say = log or (lambda _message: None)

    centres = np.asarray(part.triangles, dtype=float).mean(axis=1)
    normals = np.asarray(part.face_normals, dtype=float)
    face_areas = np.asarray(part.area_faces, dtype=float)
    face_count = len(part.faces)
    total_area = float(face_areas.sum())
    if not total_area > 0.0:
        raise ValueError("the part has no surface area to split")

    # 1. mirror partners, and the mid-surface point they imply
    caster = RayCaster(part)
    origins = centres - normals * _RAY_OFFSET_MM
    distance, hit_triangle = caster.nearest(origins, -normals)
    paired = np.flatnonzero(hit_triangle >= 0)
    if len(paired) == 0:
        raise ValueError(
            "no face found a mirror partner: this mesh does not look like a thin sheet"
        )
    partner_triangle = hit_triangle[paired]
    partner_point = origins[paired] - normals[paired] * distance[paired][:, None]
    partner_cosine = (normals[partner_triangle] * normals[paired]).sum(axis=1)
    thickness = float(np.median(distance[paired]))
    say(f"  mirror partner found for {len(paired)} / {face_count} faces "
        f"({100.0 * len(paired) / face_count:.1f}%)")
    say(f"  material thickness: median {thickness:.3f} mm, "
        f"p10 {np.percentile(distance[paired], 10):.3f}, "
        f"p90 {np.percentile(distance[paired], 90):.3f}")
    say(f"  partner anti-parallel (cos < -0.9): {100.0 * (partner_cosine < -0.9).mean():.1f}%")

    mid = centres - normals * (thickness / 2.0)
    mid[paired] = 0.5 * (centres[paired] + partner_point)

    # 2. the local mid-surface normal, which follows the stamping instead of being global
    neighbours = cKDTree(mid).query(mid, k=knn)[1]
    cloud = mid[neighbours]
    cloud = cloud - cloud.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", cloud, cloud) / knn
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    sheet_normal = eigenvectors[:, :, 0]
    planarity = eigenvalues[:, 0] / np.maximum(eigenvalues[:, 1], 1e-12)
    say(f"  mid-surface planarity w0/w1: p50 {np.percentile(planarity, 50):.4f} "
        f"p90 {np.percentile(planarity, 90):.4f}")

    # 3. orient that normal: seed where its sign is unambiguous, then propagate outwards
    axis = (through_thickness_axis(part, samples=samples, seed=seed)
            if through_axis is None else np.asarray(through_axis, dtype=float))
    axis = axis / np.linalg.norm(axis)
    projection = sheet_normal @ axis
    oriented = np.abs(projection) > seed_ratio
    sheet_normal[oriented] = np.sign(projection[oriented])[:, None] * sheet_normal[oriented]
    similar = np.abs(
        (sheet_normal[neighbours] * sheet_normal[:, None, :]).sum(axis=2)) > prop_sim
    say(f"  seeds: {int(oriented.sum())} faces ({100.0 * oriented.mean():.1f}%) "
        f"with |m . axis| > {seed_ratio}")

    rounds_used = 0
    while rounds_used < _MAX_PROPAGATION_ROUNDS and not oriented.all():
        rounds_used += 1
        votes = (sheet_normal[neighbours] * sheet_normal[:, None, :]).sum(axis=2)
        usable = similar & oriented[neighbours]
        tally = np.where(usable, np.sign(votes), 0.0).sum(axis=1)
        fresh = (~oriented) & (tally != 0)
        if not fresh.any():
            break
        sheet_normal[fresh] = np.sign(tally[fresh])[:, None] * sheet_normal[fresh]
        oriented |= fresh
    say(f"  propagation: {int(oriented.sum())} / {face_count} faces oriented "
        f"({100.0 * oriented.mean():.1f}%) in {rounds_used} rounds")

    # 4. the split itself
    cosnm = (normals * sheet_normal).sum(axis=1)
    magnitude = np.abs(cosnm)
    band_cos = float(np.cos(np.deg2rad(band_deg)))
    band = magnitude < band_cos
    upper = (~band) & (cosnm > 0)
    lower = (~band) & (cosnm < 0)

    areas = {
        "upper": float(face_areas[upper].sum()),
        "lower": float(face_areas[lower].sum()),
        "band": float(face_areas[band].sum()),
        "total": total_area,
    }
    say(f"  upper {areas['upper'] / 1e6:.4f} m2 | lower {areas['lower'] / 1e6:.4f} m2 | "
        f"sheared band {areas['band'] / 1e6:.4f} m2 of {areas['total'] / 1e6:.4f} m2")

    return SheetSplit(
        upper=upper, lower=lower, band=band, cosnm=cosnm, abs_cosnm=magnitude,
        band_cos=band_cos, through_axis=axis, thickness_mm=thickness, areas=areas,
        oriented_fraction=float(oriented.mean()), faces=face_count,
        total_area_mm2=total_area,
    )
