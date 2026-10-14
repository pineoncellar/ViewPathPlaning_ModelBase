"""Can the tracker see the scanner from a given pose?

The tracker is fixed at the station, so a pose is usable only if the tracker has
a clear line to the SCANNER. This constrains the POSE, not the target point.

Do not confuse the two ways of scoring it. Asking, per target point, "is there a
tracker-visible pose aimed at THIS point" is a systematically pessimistic proxy
and understates coverage badly (observed: a proxy reading of 87% against a true
99.6%). The right question is whether some set of tracker-visible poses covers
the point - and one pose covers its whole footprint, so the tracker only needs
to see the scanner.

The real tracker pose is unknown until the site provides it, so sweep plausible
mounting positions and report the sensitivity rather than inventing one pose and
quoting a single number. That sweep showed azimuth dominating (mount it looking
ALONG the part's thin axis, not square-on to the large faces), height and
distance barely mattering, and the mechanism being occlusion of the scanner by
the part rather than the tracker's field of view.

Deliberate omission: robot-arm occlusion is not modelled - there is no robot
model at this stage. Say so in the report instead of implying completeness,
because on the real cell the arm holding the scanner can block the tracker.

SKELETON ONLY - no implementation yet.
"""
