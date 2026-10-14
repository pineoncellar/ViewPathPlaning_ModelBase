"""Writer for the FANUC ASCII `.LS` program format.

Vendored from `reference/PHASE_6_Export_Roboguide.py` when implemented. That file
already solved the parts that are tedious and unforgiving:

* the `/ATTR` header must be self-consistent (PROG_SIZE, LINE_COUNT,
  MEMORY_SIZE), and the file name must match the `/PROG` name exactly, or the
  controller refuses the import with "Invalid name in /PROG section";
* positions are emitted as a `/POS` table of `P[n]` blocks that the motion
  instructions in `/MN` refer to by index.

Cartesian positions only. A joint block exists for an optional home pose, but
the planner never produces joint angles - solving them is the controller's job,
done in ROBOGUIDE.

Two details change whether a point looks reachable:

* the CONFIG string hints which of the robot's posture solutions to take. The
  same Cartesian pose can have several joint solutions, some inside the limits
  and some not, so "unreachable in ROBOGUIDE" has to be diagnosed by trying
  another configuration before blaming the planner.
* J moves are more permissive than L moves and match the "bring the scanner to a
  pose" intent, so J with FINE termination is the default for a review.

SKELETON ONLY - no implementation yet.
"""
