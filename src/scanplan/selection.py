"""Pick the smallest set of poses that covers the target (set cover).

Greedy - repeatedly take the candidate that adds the most uncovered AREA - then
a repair pass giving each still-reachable-but-uncovered point a dedicated pose.
Greedy plus repair has proved sufficient; the expensive alternatives buy quality
per pose, not coverage, and are worth adding only once the candidate pool stops
being the limiting factor.

Cost is measured in covered area rather than point count, so that fine sampling
around a feature cannot outvote a genuinely large uncovered region.

The repair pass is what inflates the viewpoint count: in the measured runs,
greedy reached about 98% with roughly a hundred poses and the last ~1.5% cost
several hundred more. So when the count looks too high, the fix belongs in
candidate generation. Do not silently skip the tail either - report greedy and
repair counts separately so a reviewer can see which part is compressible.

SKELETON ONLY - no implementation yet.
"""
