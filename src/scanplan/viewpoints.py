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
    """One scanner pose as it is delivered, with the provenance a reviewer needs.

    `order` is the program order and `stroke_id` the sweep this point belongs to. Both
    exist because a stroke plan is a PATH: the file order is the motion order and must
    never be re-sorted, and two points of the same stroke have to stay neighbours or the
    sweep the coverage figure was computed for does not happen. `role` says whether the
    point scans (`coverage`) or only moves the head (`retract`, `traverse`, `reorient`).
    """

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
    order: int | None = None
    stroke_id: int | None = None
    role: str = "coverage"
    # Joint evidence, present only when the kinematic gate ran. `config` is what the `.LS`
    # writer puts in the P[n] block; the rest is what makes it reviewable. `branch` is the
    # N/F, U/D, T/F triple the config string starts with.
    config: str | None = None
    branch: str | None = None
    joints_deg: tuple[float, ...] | None = None
    j5_margin_deg: float | None = None

    def as_dict(self) -> dict:
        """A plain JSON-ready mapping under the documented field names."""
        return {
            "id": int(self.id),
            "order": int(self.id if self.order is None else self.order),
            "stroke_id": None if self.stroke_id is None else int(self.stroke_id),
            "role": str(self.role),
            "position": [round(float(v), 4) for v in self.position],
            "axis": [round(float(v), 9) for v in self.axis],
            "roll_deg": float(self.roll_deg),
            "n_points": int(self.n_points),
            "area_mm2": round(float(self.area_mm2), 3),
            "mean_incidence_deg": round(float(self.mean_incidence_deg), 3),
            "max_incidence_deg": round(float(self.max_incidence_deg), 3),
            "tracker_ok": bool(self.tracker_ok),
            "source": str(self.source),
            "config": None if self.config is None else str(self.config),
            "branch": None if self.branch is None else str(self.branch),
            "joints_deg": (None if self.joints_deg is None
                           else [round(float(v), 3) for v in self.joints_deg]),
            "j5_margin_deg": (None if self.j5_margin_deg is None
                              else round(float(self.j5_margin_deg), 3)),
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
            order=order,
        ))
    return records


def viewpoints_from_path(plan, coverage: CoverageResult,
                         samples: SampleSet) -> list[Viewpoint]:
    """The delivered records of a stroke plan, in PROGRAM order.

    Every point of the path is delivered, including the transition corners: the program
    moves through them, so leaving them out would make the file and the record disagree
    about what the robot does. A transition corner carries no coverage of its own, and
    that is shown as zero rather than hidden.

    When the kinematic gate ran, each record also carries the joint solution chosen for it
    by `path.assign_branches`. That is where the `.LS` CONFIG string comes from - one value
    per point, decided by the sequence rather than assumed - so the record and the program
    cannot disagree about which branch a point belongs to.
    """
    areas = np.asarray(samples.area_mm2, dtype=float)
    solutions = list(getattr(getattr(plan, "branches", None), "solutions", []) or [])
    records: list[Viewpoint] = []
    for order, point in enumerate(plan.points):
        solution = solutions[order] if order < len(solutions) else None
        extra = {
            "config": (solution.config if solution is not None else None),
            "branch": (solution.config.split(",")[0].strip()
                       if solution is not None else None),
            "joints_deg": (tuple(float(value) for value in solution.joints_deg)
                           if solution is not None else None),
            "j5_margin_deg": (float(solution.j5_margin_deg)
                              if solution is not None else None),
        }
        row = point.coverage_index
        if row is None or len(coverage.sets[row]) == 0:
            records.append(Viewpoint(
                id=order, position=np.asarray(point.position, dtype=float),
                axis=np.asarray(point.axis, dtype=float),
                roll_deg=float(point.roll_deg), n_points=0, area_mm2=0.0,
                mean_incidence_deg=0.0, max_incidence_deg=0.0, tracker_ok=True,
                source=str(point.source), order=order, stroke_id=point.stroke_id,
                role=str(point.role), **extra))
            continue
        covered = coverage.sets[row]
        angles = np.degrees(np.arccos(np.clip(coverage.quality[row], -1.0, 1.0)))
        records.append(Viewpoint(
            id=order, position=np.asarray(point.position, dtype=float),
            axis=np.asarray(point.axis, dtype=float), roll_deg=float(point.roll_deg),
            n_points=int(len(covered)), area_mm2=float(areas[covered].sum()),
            mean_incidence_deg=float(angles.mean()),
            max_incidence_deg=float(angles.max()),
            tracker_ok=bool(coverage.tracker_ok[row]), source=str(point.source),
            order=order, stroke_id=point.stroke_id, role=str(point.role), **extra))
    return records
