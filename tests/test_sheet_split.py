"""Splitting a stamped sheet into the two faces of the blank.

To assert, on a synthetic folded plate: the two faces come out with nearly equal
area; the excluded band is the outline plus hole rims rather than a whole face;
forming side-walls are assigned to a face rather than dropped; and the split
still works on a mesh whose vertices were deliberately left unmerged, because
that is what a real CAD tessellation looks like.

SKELETON ONLY - no implementation yet.
"""
