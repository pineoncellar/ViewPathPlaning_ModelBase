"""The viewpoint set - the primary deliverable of the planning stage.

A viewpoint is a pose of the scanner REFERENCE POINT in the station CAD frame
plus the provenance needed to review it: how many target samples it covers, the
area, the incidence statistics, whether the tracker sees it, and whether it came
from the greedy or the repair pass. No joint angles: the whole point of emitting
Cartesian poses is that the controller solves them.

The reference point is where the coverage calculation placed the scanner at the
working distance from the surface. The ROBOGUIDE UTOOL has to be taught so its
TCP matches that point, otherwise every measured standoff is off by a constant.

This module is the single definition of that record: the report embeds it and the
`.LS` export reads it, so its field names are the contract the workshop sees
(spec 7.3.2). `report` and `pipeline` re-export it rather than defining their own.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .coverage import CoverageResult
from .selection import Selection
from .target_surface import SampleSet


@dataclass(frozen=True)
class Viewpoint:
    """One scanner pose as it is delivered, with the provenance a reviewer needs."""

    id: int
    position: np.ndarray
    axis: np.ndarray
    roll_deg: float
    n_points: int
    area_mm2: float
    mean_incidence_deg: float
    max_incidence_deg: float
    tracker_ok: bool
    source: str

    def as_dict(self) -> dict:
        """A plain JSON-ready mapping under the documented field names."""
        return {
            "id": int(self.id),
            "position": [round(float(v), 4) for v in self.position],
            "axis": [round(float(v), 9) for v in self.axis],
            "roll_deg": float(self.roll_deg),
            "n_points": int(self.n_points),
            "area_mm2": round(float(self.area_mm2), 3),
            "mean_incidence_deg": round(float(self.mean_incidence_deg), 3),
            "max_incidence_deg": round(float(self.max_incidence_deg), 3),
            "tracker_ok": bool(self.tracker_ok),
            "source": str(self.source),
        }


def viewpoints_from_selection(selection: Selection, coverage: CoverageResult,
                              samples: SampleSet) -> list[Viewpoint]:
    """Turn the chosen poses into the delivered records, with what each one yields."""
    areas = np.asarray(samples.area_mm2, dtype=float)
    records: list[Viewpoint] = []
    for order, index in enumerate(selection.chosen):
        covered = coverage.sets[index]
        if len(covered) == 0:
            continue
        cosines = np.clip(coverage.quality[index], -1.0, 1.0)
        angles = np.degrees(np.arccos(cosines))
        pose = coverage.poses[index]
        records.append(Viewpoint(
            id=order,
            position=np.asarray(pose.position, dtype=float),
            axis=np.asarray(pose.axis, dtype=float),
            roll_deg=float(pose.roll_deg),
            n_points=int(len(covered)),
            area_mm2=float(areas[covered].sum()),
            mean_incidence_deg=float(angles.mean()),
            max_incidence_deg=float(angles.max()),
            tracker_ok=bool(coverage.tracker_ok[index]),
            source=str(selection.sources[order]),
        ))
    return records
