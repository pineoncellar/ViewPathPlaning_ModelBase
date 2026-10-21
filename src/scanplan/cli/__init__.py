"""Command line entry point.

Planned shape (subcommands):

    scanplan plan --part P --fixture F --base B [--tracker T]
                  --config configs/scanner_trackscan49.json
                  --face auto --out output/<part>/

    scanplan ls   --viewpoints output/<part>/viewpoints.json
                  --prog-name PART01 --out output/<part>/

    scanplan view --viewpoints output/<part>/viewpoints.json
                  --station input/<station>.json --out output/<part>/

`plan` and the exports are separate on purpose: the ROBOGUIDE frame contract
(which UFRAME, and where its origin sits) is agreed with the workshop and may
need re-exporting without re-running the planner - so `ls` reads the viewpoint
file and never touches the meshes. `view` is the same idea for the picture: it
re-renders the 3D view from an existing viewpoint file, so a reviewer can look
again without paying for a plan.

Each subcommand lives in its own module (`plan`, `ls`, `view`, `pick_datums`,
`web`) exposing `configure(subparsers)` for the parser and `run(args, parser)`
for the handler. Adding or changing one subcommand therefore never means editing
another; the only shared lines are the module list in `parser.COMMANDS`.
"""

from __future__ import annotations

import sys

from .parser import COMMANDS, build_parser

_COMMANDS = {module.NAME: module.run for module in COMMANDS}


def main(argv: list[str] | None = None) -> int:
    """`scanplan plan`, `ls`, `view`, `pick-datums` and `web`."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = _COMMANDS.get(args.command)
    if handler is None:
        parser.error(f"a subcommand is required: {', '.join(_COMMANDS)}")
        return 2
    return handler(args, parser)


if __name__ == "__main__":
    sys.exit(main())
