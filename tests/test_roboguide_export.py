"""End-to-end check of the `.LS` export on a hand-made viewpoint set.

To assert: the three files appear; W/P/R reproduces the frame it came from
including the gimbal-lock case; local +Z of every exported pose is the optical
axis; positions come out in the CAD frame unchanged, and moved/rotated when a
placement is given; bad program names and an empty set are refused; and the
contract file records the two things the export cannot model.

SKELETON ONLY - no implementation yet.
"""
