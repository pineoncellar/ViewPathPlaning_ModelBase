"""The argument parser, assembled from one `configure` call per subcommand.

`COMMANDS` is the whole seam: a new subcommand is a new module plus one entry
here. Everything else in this file is shared and should stay that way.
"""

from __future__ import annotations

import argparse

from . import ls, pick_datums, plan, view, web

COMMANDS = (plan, ls, view, pick_datums, web)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scanplan", description="Scan-viewpoint planning for a FANUC + TrackScan cell")
    subparsers = parser.add_subparsers(dest="command")
    for module in COMMANDS:
        module.configure(subparsers)
    return parser
