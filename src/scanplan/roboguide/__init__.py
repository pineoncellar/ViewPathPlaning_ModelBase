"""ROBOGUIDE export adapters.

Format-specific code lives here so the planner never has to know about a
controller dialect. The planner produces a viewpoint set in the station CAD
frame; turning that into a `.LS` program is this package's problem.

`wpr` and `ls` are ported from `reference/PHASE_6_Export_Roboguide.py` rather than
rewritten, because that file already solved the FANUC ASCII details that are
tedious and unforgiving - and `adapt` is the thin bridge from the planner's own
viewpoint record onto them.
"""
