"""Bridge the planner's viewpoints onto the `.LS` writer.

The conversion itself is a few lines; the value of this module is the FRAME
CONTRACT it writes out beside the program, because that contract is what the
workshop has to reproduce in ROBOGUIDE and it is the one thing that fails
silently.

TCP: `position` is the scanner's measurement reference point - where the
coverage calculation put the scanner at the working distance. Local +Z is the
optical axis, scanner towards part. Teach the ROBOGUIDE UTOOL to match that
point; teach it wrong and every standoff is off by a constant, which shows up as
nothing at all in simulation and as a systematic measuring error on the cell.

Frame: the points are in the station CAD frame. Either keep UFRAME 1 and have
the workshop teach UFRAME 1 once to the CAD origin (the base never moves, so one
teaching serves every part), or express the CAD frame inside the ROBOGUIDE world
and run under UFRAME 0. The second is convenient but a wrong placement shifts
every point by a constant and nothing complains - always verify in ROBOGUIDE
that the points land on the part before reading anything into reachability.

Writes, per export: `<NAME>.LS`, `<NAME>_points.csv`, `<NAME>_frame.json`. The
contract file must also carry the things the export cannot know: that robot
reachability is not modelled, and that robot-arm occlusion of the tracker is not
modelled.

SKELETON ONLY - no implementation yet.
"""
