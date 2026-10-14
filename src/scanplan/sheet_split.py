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
for. Steps up to the PCA still hold for any shape.

SKELETON ONLY - no implementation yet.
"""
