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
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import trimesh

from .config import PlannerConfig, TargetParams
from .geometry import grid_dedupe, sample_surface_seeded
from .scene import Station
from .sheet_split import SheetSplit

_OVERSAMPLE = 3.0
_MIN_DRAWS = 2_000


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
    """What must be measured, and what was exempted and why."""

    face: str
    samples: SampleSet
    exempt: SampleSet
    face_area_mm2: float


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
                 cfg: PlannerConfig | TargetParams, *,
                 log: Callable[[str], None] | None = None) -> TargetSet:
    """Sample one face of the blank, holding the clamped region out of the target."""
    params = cfg.target if isinstance(cfg, PlannerConfig) else cfg
    say = log or (lambda _message: None)
    mask = split.face_mask(face)
    if not mask.any():
        raise ValueError(f"no target faces for '{face}': the split found none of them")

    say(f"sampling the {face} face")
    surface = face_mesh(part, mask)
    face_area = float(surface.area)
    all_samples = sample_face(part, mask, grid_mm=params.sample_grid_mm, seed=cfg.seed,
                              log=log)
    distance = params.contact_mm + params.contact_margin_mm
    contact = fixture_contact_mask(all_samples.points, all_samples.normals, station,
                                  distance_mm=distance)
    say(f"  fixture contact: {int(contact.sum())} samples "
        f"({100.0 * contact.mean():.3f}% = {contact.mean() * face_area:.0f} mm2) within "
        f"{distance:.1f} mm and on the outward side")

    if not contact.any():
        return TargetSet(
            face=face,
            samples=_with_reasons(all_samples, "target"),
            exempt=SampleSet(points=np.zeros((0, 3)), normals=np.zeros((0, 3)),
                             area_mm2=np.zeros(0), total_area_mm2=0.0,
                             reasons=np.zeros(0, dtype=object)),
            face_area_mm2=face_area,
        )

    kept = _select(all_samples, ~contact)
    exempt = _select(all_samples, contact)
    return TargetSet(
        face=face,
        samples=_with_reasons(kept, "target"),
        exempt=_with_reasons(exempt, "fixture_contact"),
        face_area_mm2=face_area,
    )


def _select(samples: SampleSet, mask: np.ndarray) -> SampleSet:
    return SampleSet(points=samples.points[mask], normals=samples.normals[mask],
                     area_mm2=samples.area_mm2[mask],
                     total_area_mm2=float(samples.area_mm2[mask].sum()))


def _with_reasons(samples: SampleSet, reason: str) -> SampleSet:
    return SampleSet(points=samples.points, normals=samples.normals,
                     area_mm2=samples.area_mm2, total_area_mm2=samples.total_area_mm2,
                     reasons=np.full(len(samples.points), reason, dtype=object))


def contact_summary(target: TargetSet) -> dict[str, Any]:
    """The exempted region as a report entry: how many, how much area, and where."""
    if len(target.exempt.points) == 0:
        return {"points": 0, "area_mm2": 0.0, "center": None, "size_mm": None}
    points = target.exempt.points
    return {
        "points": int(len(points)),
        "area_mm2": round(float(target.exempt.total_area_mm2), 1),
        "center": [round(float(v), 1) for v in points.mean(axis=0)],
        "size_mm": [round(float(v), 1) for v in (points.max(axis=0) - points.min(axis=0))],
    }
