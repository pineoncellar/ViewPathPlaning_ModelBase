"""Sample the target surface, and exempt what must not be measured.

Sampling: one sample per grid cell, each carrying an equal share of the face
area. Seed the RNG - mesh surface sampling draws from the global numpy state, so
an unseeded run silently changes the numbers between runs.

The sample pitch is a real trade-off rather than a detail: too coarse and the
area weighting of small features (narrow slots, tiny pockets) is wrong, and
those features are exactly where the unavoidable misses live.

Contact exemption: the clamped region is not required to be measured. The test
must be "close to the fixture AND on the outward side", because the sheet is
very thin - a plain distance test would also sweep away points on the opposite
face, which are legitimate targets. Export the exempted set so a human can
eyeball it: the CAD fixture may not match the real clamping footprint.

SKELETON ONLY - no implementation yet.
"""
