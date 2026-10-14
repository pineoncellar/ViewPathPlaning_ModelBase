"""Automatic scan-viewpoint planning for a FANUC robot + TrackScan-Sharp 49 scanner.

Given the CAD of a stamped sheet-metal part plus its fixture, the base and the
optical tracker of the station, produce a set of scanner viewpoints (position +
optical axis + roll) that together see the required face of the part, and export
them as a FANUC `.LS` program for reachability checking in ROBOGUIDE.

Deliberately NOT here: inverse kinematics, joint limits, robot collision,
viewpoint ordering, motion timing. The `.LS` carries Cartesian poses only.

Two invariants the whole package has to agree on:

1. Poses are in the station CAD frame, millimetres, and the scanner's local +Z
   is the optical axis pointing FROM the scanner TOWARDS the part. The
   ROBOGUIDE UTOOL must be taught to match that reference point.
2. The target surface is defined topologically: one of the two faces of the
   original flat blank, forming side-walls included, sheared edge excluded.
   See :mod:`scanplan.sheet_split`.

The pipeline is `scene -> sheet_split -> target_surface -> candidates -> coverage ->
selection -> report`, and `scanplan.cli` drives it as two subcommands.
"""
