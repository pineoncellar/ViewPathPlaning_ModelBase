"""The viewpoint set - the primary deliverable of the planning stage.

A viewpoint is a pose of the scanner REFERENCE POINT in the station CAD frame
plus the provenance needed to review it: how many target samples it covers, the
area, the incidence statistics, whether the tracker sees it, and whether it came
from the greedy or the repair pass. No joint angles: the whole point of emitting
Cartesian poses is that the controller solves them.

The reference point is where the coverage calculation placed the scanner at the
working distance from the surface. The ROBOGUIDE UTOOL has to be taught so its
TCP matches that point, otherwise every measured standoff is off by a constant.

SKELETON ONLY - no implementation yet.
"""
