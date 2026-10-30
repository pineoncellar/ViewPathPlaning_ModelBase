"""Repair a stamped sheet's triangle winding before anything reads its normals.

Every later stage trusts `face_normals`:

* the split decides which of the blank's two faces a triangle belongs to from `n . m`,
* sampling hands `n` to the pose ladder as the optical axis the scanner has to oppose,
* the fixture-contact exemption asks whether the obstacle lies OUTSIDE the sheet along `n`,
* the feature mask asks how far `n` leans off the face's dominant direction.

A CAD export that leaves part of the back face wound the same way as the front therefore
poisons all four at once. Measured on station 20261030_100955: 0.225 m2 of the 0.575 m2
"upper" set - 39 % of the picked face - was really the back of the sheet, so the exemption
ate a contiguous half of the face (the back samples sit 0.65 mm in front of the "fixture",
which was the part itself, so the outward test was satisfied for all of them), every
remaining sample was flagged as a hard feature, and the poses aimed through the panel at a
surface the scanner cannot see. In the viewer that reads as "parts of the panel were never
recognised as the scan face".

The repair needs neither a watertight mesh nor a consistent input. A correctly wound
triangle of a thin shell has its mirror partner BEHIND it - a short ray along `-n` hits the
opposite face one thickness away - and empty space in FRONT (a ray along `+n` escapes). A
reversed triangle is the other way round, which is measurable per triangle with two rays.
Rims and forming walls escape in both directions and are left alone, which is the correct
answer for them: a sheared edge belongs to neither face.

Nothing here decides which side is "outside" in a global sense; it only makes the sheet's
own two surfaces face away from each other, which is all the four stages above need.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import trimesh

from .raycast import RayCaster

# Start the ray just off the surface so it cannot hit its own triangle.
_RAY_OFFSET_MM = 0.02
# How many measured thicknesses in front of a face still count as "the other side of the
# same sheet" rather than as a separate feature standing in the way.
_THICKNESS_WINDOW = 3.0
# A repair that would flip nearly the whole mesh describes something this rule does not
# model (a doubled panel, a solid, a plate with a lid a thickness away), so it is refused
# rather than applied to every face.
_MAX_FLIP_SHARE = 0.85


def repair_winding(part, *, log: Callable[[str], None] | None = None
                   ) -> tuple[trimesh.Trimesh, dict]:
    """Flip the triangles of `part` whose normal points INTO the sheet.

    Returns the mesh to use from now on (the same object when nothing had to change) and
    a record of the measurement: how many faces were flipped, how much area that is, the
    thickness the rule was calibrated on, and what is left over afterwards.
    """
    say = log or (lambda _message: None)
    info: dict = {"flipped_faces": 0, "flipped_area_mm2": 0.0, "share": 0.0,
                  "thickness_mm": None, "remaining_share": None, "applied": False}
    faces = np.asarray(part.faces)
    if len(faces) == 0:
        return part, info

    centres = np.asarray(part.triangles, dtype=float).mean(axis=1)
    normals = np.asarray(part.face_normals, dtype=float)
    caster = RayCaster(part)
    front, _ = caster.nearest(centres + normals * _RAY_OFFSET_MM, normals)
    behind, _ = caster.nearest(centres - normals * _RAY_OFFSET_MM, -normals)

    # One of the two directions meets the opposite face of the sheet on every face that is
    # part of a blank, so the median of the nearer hit is the material thickness.
    nearer = np.minimum(front, behind)
    measured = nearer[np.isfinite(nearer)]
    if len(measured) == 0:
        return part, info
    thickness = float(np.median(measured))
    if not thickness > 0.0:
        return part, info
    window = _THICKNESS_WINDOW * thickness

    flip = (front < window) & ~(behind < window)
    count = int(flip.sum())
    share = float(flip.mean())
    info.update(thickness_mm=round(thickness, 3), share=round(share, 4),
                flipped_faces=count,
                flipped_area_mm2=round(float(part.area_faces[flip].sum()), 1))
    if count == 0:
        return part, info
    if share > _MAX_FLIP_SHARE:
        say(f"  winding: {count} faces ({100.0 * share:.1f} %) face into the sheet, which "
            "is too many for a two-sided blank - left unchanged, check the model")
        return part, info

    reversed_faces = np.array(faces, dtype=np.int64)
    reversed_faces[flip] = reversed_faces[flip][:, ::-1]
    repaired = trimesh.Trimesh(vertices=np.asarray(part.vertices, dtype=float).copy(),
                               faces=reversed_faces, process=False)
    after = RayCaster(repaired)
    repaired_normals = np.asarray(repaired.face_normals, dtype=float)
    after_front, _ = after.nearest(centres + repaired_normals * _RAY_OFFSET_MM,
                                   repaired_normals)
    remaining = float((after_front < window).mean())
    info["remaining_share"] = round(remaining, 4)
    if remaining >= share:
        say(f"  winding: flipping {count} faces did not improve the mesh "
            f"({100.0 * remaining:.1f} % still face the sheet) - left unchanged")
        return part, info
    info["applied"] = True
    return repaired, info
