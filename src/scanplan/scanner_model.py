"""When can one scanner pose validly measure one surface point?

A point counts as covered by a pose only when ALL hold:
  1. inside the field of view (rectangular footprint, scaled with distance);
  2. distance within the measuring range;
  3. incidence angle within the limit;
  4. line of sight clear of part, fixture and base;
  5. the scanner body fits, including the safety clearance;
  6. the tracker can see the scanner.

The body test uses a convex envelope on purpose: it can only ever call a fitting
pose unfitting, never the reverse, so the error direction is "report more
misses", which is the safe one.

On point 3: `max_incidence_deg` is the most sensitive parameter of the planner
and is a placeholder. Never hard-code it here.

SKELETON ONLY - no implementation yet.
"""
