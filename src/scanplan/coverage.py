"""Build the candidate x target coverage sets.

For each candidate, decide which target samples it can validly measure (see
:mod:`scanplan.scanner_model`) and record them, with a quality term (incidence
angle) so that among equals the more head-on pose can win - a grazing pose is
geometrically covered but scans badly.

Implementation notes that cost real time to learn:

* use the vectorised ray intersector and ask for the FIRST hit only; the generic
  "does this ray hit anything" query is several times slower than the one
  returning the hit index and location;
* never build the incidence cosine as a full matrix product of candidates x
  target points - at realistic sizes that materialises gigabytes and the process
  appears to hang;
* do not assume the intersector's return order; a swapped tuple unpacks without
  error and yields nonsense;
* seed the RNG, or the numbers wobble between runs for no visible reason.

SKELETON ONLY - no implementation yet.
"""
