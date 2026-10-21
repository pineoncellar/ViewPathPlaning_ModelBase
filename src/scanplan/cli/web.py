"""`scanplan web`: the local browser interface for station setup and planning."""

from __future__ import annotations

import argparse

NAME = "web"


def configure(subparsers) -> None:
    web = subparsers.add_parser(
        NAME,
        help="start the local browser interface for station setup and path planning",
    )
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8765)


def run(args, parser: argparse.ArgumentParser) -> int:
    from scanweb import serve

    serve(args.host, args.port)
    return 0
