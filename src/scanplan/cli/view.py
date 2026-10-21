"""`scanplan view`: re-render the 3D page from an existing viewpoint file."""

from __future__ import annotations

import argparse
import json

from scanview import DEFAULT_MAX_FACES, render_run

NAME = "view"


def configure(subparsers) -> None:
    viewer = subparsers.add_parser(
        NAME, help="write a self-contained 3D view of the station and its viewpoints")
    viewer.add_argument("--viewpoints", required=True, help="viewpoints.json")
    viewer.add_argument("--station", help="station JSON (paths, transforms, tracker)")
    viewer.add_argument("--part", help="part mesh, if no --station is given")
    viewer.add_argument("--fixture", help="fixture mesh, if no --station is given")
    viewer.add_argument("--base", help="base mesh, if no --station is given")
    viewer.add_argument("--config", help="configuration override (JSON), for the FOV cone")
    viewer.add_argument("--max-faces", type=int, default=DEFAULT_MAX_FACES,
                        help="display budget per model (models above it are decimated)")
    viewer.add_argument("--out", default="output", help="output directory")


def run(args, parser: argparse.ArgumentParser) -> int:
    # Everything the page needs -- station loading, the sample sidecar, the tracker
    # pose -- lives in `scanview.render_run`; the CLI only maps flags onto it.
    try:
        path = render_run(args.viewpoints, args.out, config=args.config,
                          station=args.station, part=args.part, fixture=args.fixture,
                          base=args.base, max_faces=args.max_faces)
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
        return 2

    print(f"wrote {path.resolve()}")
    print("open it in a browser: it needs no network and no Python")
    return 0
