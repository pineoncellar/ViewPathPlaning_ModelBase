"""Load the station models and put them into ONE frame.

The station is part + fixture + base + tracker. All of them must live in a
single CAD frame before anything else runs, because every later number
(occlusion, clearances, the exported pose) is defined in that frame.

The CAD handover may arrive with the assembly placement LOST - files exported
one by one, each sitting at its own origin or rotated relative to the base. That
silently invalidates everything downstream and still looks plausible, so the
alignment has to be explicit, reproducible and self-checked: fixture feet on the
base mounting surface, part lying flat, nothing interpenetrating, fixture
footprint inside the base.

SKELETON ONLY - no implementation yet.
"""
