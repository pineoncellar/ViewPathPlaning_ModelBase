"""Command line entry point.

Planned shape (subcommands):

    scanplan plan --part P --fixture F --base B [--tracker T]
                  --config configs/scanner_trackscan49.json
                  --face auto --out output/<part>/

    scanplan ls   --viewpoints output/<part>/viewpoints.json
                  --prog-name PART01 --out output/<part>/

`plan` and the `.LS` export are separate on purpose: the ROBOGUIDE frame
contract (which UFRAME, and where its origin sits) is agreed with the workshop
and may need re-exporting without re-running the planner.

SKELETON ONLY - no implementation yet.
"""
