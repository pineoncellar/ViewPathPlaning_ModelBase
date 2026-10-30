"""Sample the target surface, and exempt what must not be measured.

Sampling: one sample per grid cell, each carrying an equal share of the face
area. Seed the RNG - mesh surface sampling draws from the global numpy state, so
an unseeded run silently changes the numbers between runs.

The sample pitch is a real trade-off rather than a detail: too coarse and the
area weighting of small features (narrow slots, tiny pockets) is wrong, and
those features are exactly where the unavoidable misses live.

Contact exemption: the clamped region is not required to be measured. The test
must be "close to the fixture AND on the outward side", because the sheet is
very thin - a plain distance test would also sweep away points on the opposite
face, which are legitimate targets. Export the exempted set so a human can
eyeball it: the CAD fixture may not match the real clamping footprint.

The sampling is a grid dedupe at the configured pitch, not a Poisson disc. The
design text says "Poisson disc at 3 mm"; a grid at the same nominal pitch is what
every validated run used, it is exactly reproducible, and it gives every sample the
same area share - which is what the area-weighted set cover consumes. A disc at
the same pitch would move the samples and make the coverage figures
non-comparable with the measured runs for no gain in reachability.

Picked scan faces are SUPPLEMENTARY: the operator picks extra surfaces in the 3D view
and they are added to the main scan face, which is never shrunk by a pick. The picks
arrive as the patch's own triangles, so the added boundary is the highlighted surface
itself - a surface on the far side of the blank, or on a flange the split assigns to the
other face, simply joins the target.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .config import PlannerConfig, TargetParams
from .geometry import grid_dedupe, sample_surface_seeded
from .scene import Station
from .sheet_split import SheetSplit

_OVERSAMPLE = 3.0
_MIN_DRAWS = 2_000

# How far a sample's normal may depart from the picked face's normal and still count as
# inside it. The page grows a pick with a tighter angle, so this only absorbs the
# difference between its triangle normals and the sampled ones.
PICKED_NORMAL_DEG = 20.0


@dataclass(frozen=True)
class SampleSet:
    """Surface samples with their normals and their share of the face area."""

    points: np.ndarray
    normals: np.ndarray
    area_mm2: np.ndarray
    total_area_mm2: float
    reasons: np.ndarray | None = None


@dataclass(frozen=True)
class TargetSet:
    """What must be measured, and what was exempted and why.

    `face_area_mm2` is the whole target: the main scan face plus any supplementary faces
    picked in the 3D view. `supplementary_mm2` says how much of it the picks added, so a
    report can tell "the face" from "the face plus what the operator asked for".
    """

    face: str
    samples: SampleSet
    exempt: SampleSet
    face_area_mm2: float
    supplementary_mm2: float = 0.0
    supplementary_regions: int = 0


@dataclass(frozen=True)
class ScanRegion:
    """One surface the operator picked in the 3D view: the triangles of that patch.

    `triangles` is an `(N, 3, 3)` array of the picked patch's triangles in the station
    frame - exactly the ones the page highlighted. A sample belongs to the picked surface
    when it faces the same way and sits within `tolerance_mm` of one of those triangles;
    nothing has to guess a spacing or a plane, and a hole nobody picked simply has no
    triangle there.

    Distance rather than a projected outline is what makes this work on real geometry: a
    picked face may be slightly curved, and its own triangles describe that curvature
    exactly, while a single plane through it would either reject the edges or accept the
    pocket bottoms behind them.
    """

    triangles: np.ndarray
    normal: np.ndarray
    origin: np.ndarray
    tolerance_mm: float = 0.5

    def mask(self, points, normals, *, tolerance_deg: float = PICKED_NORMAL_DEG) -> np.ndarray:
        """Which samples lie on this picked surface, facing the same way."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        normals = np.asarray(normals, dtype=float).reshape(-1, 3)
        if len(points) == 0:
            return np.zeros(0, dtype=bool)
        normal = np.asarray(self.normal, dtype=float)
        normal = normal / (np.linalg.norm(normal) or 1.0)
        aligned = normals @ normal >= np.cos(np.deg2rad(tolerance_deg))
        near = distance_to_triangles(points, self.triangles) <= float(self.tolerance_mm)
        return aligned & near


_NEAREST_TRIANGLES = 8


def distance_to_triangles(points, triangles) -> np.ndarray:
    """Exact distance from each point to the nearest triangle of the patch.

    The candidates come from a KD-tree over the triangles' centres, so the cost stays
    linear in the sample count; within a candidate the distance is exact (point-to-segment
    for outside points, point-to-plane for points whose projection is inside), which is what
    keeps the target boundary on the picked face instead of one spacing away from it.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    triangles = np.asarray(triangles, dtype=float)
    if len(points) == 0:
        return np.zeros(0)
    if len(triangles) == 0:
        return np.full(len(points), np.inf)
    centroids = triangles.mean(axis=1)
    count = min(_NEAREST_TRIANGLES, len(triangles))
    _, nearest = cKDTree(centroids).query(points, k=count)
    nearest = np.atleast_2d(nearest).reshape(len(points), count)
    corners = triangles[nearest]
    a, b, c = corners[:, :, 0], corners[:, :, 1], corners[:, :, 2]
    p = points[:, None, :]
    best = np.minimum.reduce((_segment_distance2(p, a, b),
                              _segment_distance2(p, b, c),
                              _segment_distance2(p, c, a)))
    edge = np.cross(b - a, c - a)
    length = np.maximum(np.linalg.norm(edge, axis=-1, keepdims=True), 1e-12)
    plane = np.abs(((p - a) * (edge / length)).sum(axis=-1))
    inside = _inside_triangle(p, a, b, c, edge)
    best = np.where(inside, np.minimum(best, plane), best)
    return np.sqrt(best.min(axis=1))


def _segment_distance2(p, start, end) -> np.ndarray:
    along = end - start
    length2 = np.maximum((along * along).sum(axis=-1), 1e-12)
    t = np.clip(((p - start) * along).sum(axis=-1) / length2, 0.0, 1.0)
    offset = p - (start + along * t[..., None])
    return (offset * offset).sum(axis=-1)


def _inside_triangle(p, a, b, c, normal) -> np.ndarray:
    def side(u, v):
        return ((np.cross(v - u, p - u) * normal).sum(axis=-1)) >= 0.0

    first = side(a, b)
    second = side(b, c)
    third = side(c, a)
    return (first & second & third) | (~first & ~second & ~third)


def region_mask(points, normals, regions: Sequence[ScanRegion] | None, *,
                tolerance_deg: float = PICKED_NORMAL_DEG) -> np.ndarray:
    """Which samples fall inside any of the picked surfaces. No picks means none of them."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if not regions:
        return np.zeros(len(points), dtype=bool)
    normals = np.asarray(normals, dtype=float).reshape(-1, 3)
    picked = np.zeros(len(points), dtype=bool)
    for region in regions:
        picked |= region.mask(points, normals, tolerance_deg=tolerance_deg)
    return picked


def region_face_mask(part, regions: Sequence[ScanRegion] | None, *,
                     tolerance_deg: float = PICKED_NORMAL_DEG) -> np.ndarray:
    """Which triangles of `part` the picked surfaces cover.

    The picks are supplementary scan faces, so they are resolved at the triangle level and
    unioned with the main face: a picked surface on the other side of the blank, or on a
    flange the split assigns elsewhere, is added to the target rather than replacing it.
    """
    mask = np.zeros(len(part.faces), dtype=bool)
    if not regions:
        return mask
    centroids = np.asarray(part.triangles, dtype=float).mean(axis=1)
    normals = np.asarray(part.face_normals, dtype=float)
    for region in regions:
        mask |= region.mask(centroids, normals, tolerance_deg=tolerance_deg)
    return mask


def face_mesh(part, mask) -> trimesh.Trimesh:
    """The geometry of the faces `mask` selects."""
    selected = np.flatnonzero(np.asarray(mask, dtype=bool))
    if len(selected) == 0:
        raise ValueError("the mask selects no faces")
    return part.submesh([selected], append=True)


def sample_face(part, mask, *, grid_mm: float, seed: int,
                samples: int | None = None,
                log: Callable[[str], None] | None = None) -> SampleSet:
    """One area-weighted sample per `grid_mm` cell of the selected faces."""
    say = log or (lambda _message: None)
    surface = face_mesh(part, mask)
    face_area = float(surface.area)
    if not face_area > 0.0:
        raise ValueError("the selected faces have no area")

    draws = samples if samples is not None else int(max(_OVERSAMPLE * face_area / grid_mm ** 2,
                                                        _MIN_DRAWS))
    points, normals = sample_surface_seeded(surface, draws, seed)
    points, normals = grid_dedupe(points, normals, grid_mm=grid_mm)
    count = len(points)
    if count == 0:
        raise ValueError(f"grid_mm={grid_mm} is too coarse for a face of {face_area:.0f} mm2")

    share = face_area / count
    say(f"  face {face_area:.0f} mm2 -> {count} samples ({share:.1f} mm2 each) on a "
        f"{grid_mm:g} mm grid")
    return SampleSet(points=points, normals=normals,
                     area_mm2=np.full(count, share), total_area_mm2=face_area)


def fixture_contact_mask(points, normals, station: Station, *,
                         distance_mm: float) -> np.ndarray:
    """Which samples sit in the clamped region.

    Both halves matter: near the fixture AND on its side of the sheet. The sheet is
    under a millimetre thick, so a plain distance test would also exempt the far face.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    normals = np.asarray(normals, dtype=float).reshape(-1, 3)
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    distance = station.clearance_mm(points)
    nearest = station.nearest_obstacle_point(points)
    outward = ((nearest - points) * normals).sum(axis=1) > 0.0
    return (distance < distance_mm) & outward


def build_target(part, split: SheetSplit, face: str, station: Station,
                 cfg: PlannerConfig | TargetParams, *, seed: int | None = None,
                 log: Callable[[str], None] | None = None,
                 regions: Sequence[ScanRegion] | None = None) -> TargetSet:
    """Sample one face of the blank, holding the clamped region out of the target.

    `seed` must be given when `cfg` is bare `TargetParams`, because those carry no seed and
    surface sampling draws from the global numpy RNG - inventing one here would make the
    result silently irreproducible.

    `regions` are the supplementary scan faces the operator picked in the 3D view. They
    are ADDED to the main scan face - the picks never shrink it. A picked surface on the
    other side of the blank or on a flange the split assigns elsewhere becomes part of the
    target, and a pick that lands on the main face simply adds nothing.
    """
    params = cfg.target if isinstance(cfg, PlannerConfig) else cfg
    if isinstance(cfg, PlannerConfig):
        resolved_seed = cfg.seed if seed is None else seed
    elif seed is None:
        raise ValueError(
            "give seed= when passing bare TargetParams: nothing else supplies one, and "
            "unseeded sampling changes the samples between runs"
        )
    else:
        resolved_seed = seed
    say = log or (lambda _message: None)
    mask = split.face_mask(face)
    if not mask.any():
        raise ValueError(f"no target faces for '{face}': the split found none of them")

    main_mask = mask.copy()
    main_area = float(np.asarray(part.area_faces, dtype=float)[main_mask].sum())
    picked_mask = region_face_mask(part, regions)
    if regions and not picked_mask.any():
        raise ValueError(
            "none of the picked supplementary faces lies on the part: pick a surface of the "
            "part, not the fixture or the base"
        )
    supplementary_mask = picked_mask & ~main_mask
    supplementary_area = float(
        np.asarray(part.area_faces, dtype=float)[supplementary_mask].sum())
    if regions:
        say(f"  supplementary scan faces: {int(supplementary_mask.sum())} triangles, "
            f"{supplementary_area:.0f} mm2 added to the {face} face "
            f"({len(regions)} picked surface(s))")
    mask = main_mask | supplementary_mask

    say(f"sampling the {face} face")
    face_area = main_area + supplementary_area
    all_samples = sample_face(part, mask, grid_mm=params.sample_grid_mm,
                              seed=resolved_seed, log=log)
    distance = params.contact_mm + params.contact_margin_mm
    contact = fixture_contact_mask(all_samples.points, all_samples.normals, station,
                                  distance_mm=distance)
    say(f"  fixture contact: {int(contact.sum())} samples "
        f"({100.0 * contact.mean():.3f}% = {contact.mean() * face_area:.0f} mm2) within "
        f"{distance:.1f} mm and on the outward side")

    required = ~contact
    if required.all():
        return TargetSet(
            face=face,
            samples=_with_reasons(all_samples, "target"),
            exempt=_empty_samples(),
            face_area_mm2=face_area,
            supplementary_mm2=supplementary_area,
            supplementary_regions=len(regions or ()),
        )

    return TargetSet(
        face=face,
        samples=_with_reasons(_select(all_samples, required), "target"),
        exempt=_with_reasons(_select(all_samples, contact), "fixture_contact"),
        face_area_mm2=face_area,
        supplementary_mm2=supplementary_area,
        supplementary_regions=len(regions or ()),
    )


def _empty_samples() -> SampleSet:
    return SampleSet(points=np.zeros((0, 3)), normals=np.zeros((0, 3)),
                     area_mm2=np.zeros(0), total_area_mm2=0.0,
                     reasons=np.zeros(0, dtype=object))


def _select(samples: SampleSet, mask: np.ndarray) -> SampleSet:
    return SampleSet(points=samples.points[mask], normals=samples.normals[mask],
                     area_mm2=samples.area_mm2[mask],
                     total_area_mm2=float(samples.area_mm2[mask].sum()))


def _with_reasons(samples: SampleSet, reason: str | np.ndarray) -> SampleSet:
    reasons = (np.full(len(samples.points), reason, dtype=object)
               if isinstance(reason, str) else np.asarray(reason, dtype=object))
    return SampleSet(points=samples.points, normals=samples.normals,
                     area_mm2=samples.area_mm2, total_area_mm2=samples.total_area_mm2,
                     reasons=reasons)


def exempt_breakdown(target: TargetSet) -> dict[str, dict[str, Any]]:
    """The exempted samples split by reason: what was left out, and how much of it."""
    if target.exempt.reasons is None or len(target.exempt.points) == 0:
        return {}
    reasons = np.asarray(target.exempt.reasons, dtype=object)
    breakdown: dict[str, dict[str, Any]] = {}
    for reason in sorted(set(reasons.tolist())):
        chosen = reasons == reason
        breakdown[str(reason)] = {
            "points": int(chosen.sum()),
            "area_mm2": round(float(target.exempt.area_mm2[chosen].sum()), 1),
        }
    return breakdown


def contact_summary(target: TargetSet, reason: str = "fixture_contact") -> dict[str, Any]:
    """The exempted region as a report entry: how many, how much area, and where."""
    if target.exempt.reasons is None or len(target.exempt.points) == 0:
        return {"points": 0, "area_mm2": 0.0, "center": None, "size_mm": None}
    chosen = np.asarray(target.exempt.reasons, dtype=object) == reason
    if not chosen.any():
        return {"points": 0, "area_mm2": 0.0, "center": None, "size_mm": None}
    points = target.exempt.points[chosen]
    return {
        "points": int(len(points)),
        "area_mm2": round(float(target.exempt.area_mm2[chosen].sum()), 1),
        "center": [round(float(v), 1) for v in points.mean(axis=0)],
        "size_mm": [round(float(v), 1) for v in (points.max(axis=0) - points.min(axis=0))],
    }

