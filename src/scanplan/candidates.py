"""Generate candidate scanner poses.

For each aim point take several working distances, tilt the optical axis a few
degrees off the surface normal, and try a couple of rolls about it - the roll
matters because the field of view is a rectangle, not a disc.

Candidates are generated without regard to occlusion or reachability, then
filtered. Filtering first and planning second is the right order: a viewpoint
discovered to be unusable after selection leaves a hole that cannot be patched
without planning again.

Density is what actually buys coverage. A pool of one pose per target point caps
coverage far below what the geometry allows, because one pose covers its whole
footprint. Measured on a real part, per-point reachability was about 96% while
the pool as a whole reached about 99.6% - so the gap between achieved coverage
and the pool's ceiling is a candidate-generation question, not a set-cover one.
Refine towards the still-uncovered region before concluding anything is
unreachable, and seed that refinement too.

SKELETON ONLY - no implementation yet.
"""
