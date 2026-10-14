"""Pure geometry: the scanner body frame and its tangent helpers.

`scanner_frame` is the single definition of the scanner frame. The view-field
tests, the exported pose and the ROBOGUIDE UTOOL all depend on it, so it must
stay the only place a frame is built.

Watch out for the degenerate case where a reference axis is parallel to the
optical axis: a naive cross product then vanishes.

SKELETON ONLY - no implementation yet.
"""
