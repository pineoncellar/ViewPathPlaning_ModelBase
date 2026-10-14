"""Orchestrate the planning stage end to end.

Order: load and align the station, split the sheet, sample the target, generate
candidates, evaluate coverage, select, then export the viewpoint set and the
report.

Two ordering decisions worth keeping:

1. Filter before planning. Candidates that are unusable (occluded, out of range,
   colliding, invisible to the tracker) must be removed before set cover runs,
   otherwise the selected set leaves holes that cannot be patched.
2. The tracker constraint filters POSES, not target points, so it belongs inside
   the coverage evaluation - not as a post-filter on the result.

Choosing which face of the blank to scan lives here too. When `face` is "auto",
plan both faces and keep the one with the higher reachability; no single
geometric quantity decides it reliably.

SKELETON ONLY - no implementation yet.
"""
