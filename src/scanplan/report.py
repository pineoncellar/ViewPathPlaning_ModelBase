"""Write the coverage report that has to accompany any viewpoint set.

Must contain: total coverage, target area, viewpoint count; the exempted
(contact) cloud and the unreachable cloud, each with a reason per point; the
incidence-angle distribution; the uncovered samples classified by cause
(blocked by part / fixture / base, never in view, only reachable from poses that
would collide); and the uncovered regions clustered into a short list of blocks.

Must also state what it does NOT include:

* coverage is GEOMETRIC reachability only - robot reachability, joint limits and
  robot collision are decided later, in ROBOGUIDE;
* tracker visibility does not include robot-arm occlusion;
* when a measured scanner parameter is still a placeholder, say which one, so
  nobody reads the coverage figure as a prediction.

Output as a machine-readable JSON plus the human-readable summary; the figures
go in a sibling directory.

SKELETON ONLY - no implementation yet.
"""
