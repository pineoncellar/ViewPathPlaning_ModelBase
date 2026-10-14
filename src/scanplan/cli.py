"""Command line entry point.

Planned shape (subcommands):

    scanplan plan --part P --fixture F --base B [--tracker T]
                  --config configs/scanner_trackscan49.json
                  --face auto --out output/<part>/

    scanplan ls   --viewpoints output/<part>/viewpoints.json
                  --prog-name PART01 --out output/<part>/

`plan` and the `.LS` export are separate on purpose: the ROBOGUIDE frame
contract (which UFRAME, and where its origin sits) is agreed with the workshop
and may need re-exporting without re-running the planner - so `ls` reads the
viewpoint file and never touches the meshes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .config import FACES, load_config, read_json
from .raycast import backend_name
from .report import summary_lines
from .roboguide.adapt import export as export_ls
from .scene import StationDescription, load_station, load_station_description
from .tracker import (
    Tracker,
    part_centroid,
    rank_placements,
    sweep_placements,
    tracker_from_description,
)


def main(argv: list[str] | None = None) -> int:
    """`scanplan plan` and `scanplan ls`."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "plan":
        return _run_plan(args, parser)
    if args.command == "ls":
        return _run_ls(args, parser)
    parser.error("a subcommand is required: plan or ls")
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scanplan", description="Scan-viewpoint planning for a FANUC + TrackScan cell")
    subparsers = parser.add_subparsers(dest="command")

    planner = subparsers.add_parser(
        "plan", help="plan scanner viewpoints for one face of a stamped blank")
    planner.add_argument("--station", help="station JSON (paths, transforms, tracker)")
    planner.add_argument("--part", help="part mesh, if no --station is given")
    planner.add_argument("--fixture", help="fixture mesh, if no --station is given")
    planner.add_argument("--base", help="base mesh, if no --station is given")
    planner.add_argument("--config", help="configuration override (JSON)")
    planner.add_argument("--face", choices=FACES, help="which face of the blank to scan")
    planner.add_argument("--tracker", help="tracker pose 'x,y,z' if the station has none")
    planner.add_argument("--tracker-aim", help="tracker aim point 'x,y,z'")
    planner.add_argument("--tracker-sweep", action="store_true",
                         help="report the spread over the 192 candidate placements")
    planner.add_argument("--out", default="output", help="output directory")
    planner.add_argument("--quiet", action="store_true", help="only the summary")

    exporter = subparsers.add_parser(
        "ls", help="export a planned viewpoint set as a FANUC .LS program")
    exporter.add_argument("--viewpoints", required=True, help="viewpoints.json")
    exporter.add_argument("--prog-name", required=True, help="FANUC program name")
    exporter.add_argument("--out", default="output", help="output directory")
    exporter.add_argument("--place", help="pose 'x,y,z,w,p,r' of the CAD frame in ROS "
                                         "world; switches the program to UFRAME 0")
    exporter.add_argument("--uframe", type=int, help="UFRAME number (default 1, or 0 "
                                                    "with --place)")
    exporter.add_argument("--utool", type=int, default=1)
    exporter.add_argument("--motion", default="J", choices=["J", "L"],
                          help="J is the permissive default for a reachability review")
    exporter.add_argument("--speed", default="100%")
    return parser


def _run_plan(args, parser: argparse.ArgumentParser) -> int:
    from .pipeline import plan, write_outputs

    say = (lambda message: None) if args.quiet else (lambda message: print(message, flush=True))
    try:
        cfg = load_config(args.config)
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as error:
        parser.error(f"cannot read the configuration: {error}")
        return 2

    try:
        description = _station_description(args, parser)
        say(f"loading station: {description.part.parent}")
        station = load_station(description, cloud_seed=cfg.seed)
        tracker = _tracker(args, description, station, parser, cfg)
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as error:
        parser.error(str(error))
        return 2

    if args.tracker_sweep:
        _report_tracker_sweep(station, cfg, say)

    try:
        result = plan(station, cfg, face=args.face, tracker=tracker, log=say)
    except ValueError as error:
        parser.error(str(error))
        return 2

    paths = write_outputs(result, args.out)
    print()
    print("\n".join(summary_lines(result.report)))
    print()
    print(f"wrote {len(paths)} files to {Path(args.out).resolve()}")
    print("next: scanplan ls --viewpoints viewpoints.json --prog-name <NAME> --out <dir>")
    return 0


def _station_description(args, parser: argparse.ArgumentParser) -> StationDescription:
    if args.station:
        return load_station_description(args.station)
    missing = [name for name in ("part", "fixture", "base") if not getattr(args, name)]
    if missing:
        parser.error("give --station, or all three of --part, --fixture and --base "
                     f"(missing: {', '.join('--' + name for name in missing)})")
    paths = {}
    for name in ("part", "fixture", "base"):
        path = Path(getattr(args, name))
        if not path.exists():
            parser.error(f"the {name} mesh does not exist: {path}")
        paths[name] = path
    return StationDescription(part=paths["part"], fixture=paths["fixture"],
                              base=paths["base"])


def _tracker(args, description: StationDescription, station, parser,
             cfg) -> Tracker | None:
    if args.tracker:
        position = _triple(args.tracker)
        if position is None:
            parser.error("--tracker expects 'x,y,z' in millimetres")
            return None
        aim = _triple(args.tracker_aim) if args.tracker_aim else None
        if aim is None:
            aim = tuple(float(v) for v in part_centroid(station.part))
        return Tracker(position=position, aim=aim, params=cfg.tracker,
                       name="from --tracker")
    tracker = tracker_from_description(description, cfg.tracker, part=station.part)
    if tracker is None:
        print("no tracker pose given: the tracker constraint is NOT applied. Pass "
              "--tracker x,y,z, or --tracker-sweep to see what the choice costs.")
    return tracker


def _triple(text: str):
    try:
        values = [float(v) for v in text.split(",")]
    except ValueError:
        return None
    return tuple(values) if len(values) == 3 else None


def _report_tracker_sweep(station, cfg, say) -> None:
    """What does the mounting choice cost? The design asks for the spread, not a number.

    Deliberately not a full plan per placement: 192 plans would take hours, and the
    question the design answers here is how much of the candidate pool a placement keeps,
    which the visibility test answers directly.
    """
    from .candidates import generate

    say("tracker sweep: ranking the 192 documented placements against the candidate pool")
    points = np.asarray(station.part.triangles.mean(axis=1), dtype=float)
    normals = np.asarray(station.part.face_normals, dtype=float)
    poses = generate(points, normals, cfg.scanner, seed=cfg.seed)
    positions = np.array([pose.position for pose in poses], dtype=float)
    obstacles = station.obstacles()
    ranked = sorted((rank_placements(placement, positions, obstacles)
                     for placement in sweep_placements(station.part, cfg.tracker)),
                    key=lambda entry: -entry["visible_pct"])
    for label, entry in (("best", ranked[0]), ("median", ranked[len(ranked) // 2]),
                         ("worst", ranked[-1])):
        say(f"  {label:<6} {entry['visible_pct']:6.1f}% of {entry['poses']} pool poses "
            f"visible   ({entry['tracker']['name']})")
    say(f"  ray casting uses {backend_name()}")
    say("  a badly placed tracker costs coverage rather than blocking the work, so this "
        "is a mounting requirement: look ALONG the part's thin axis, not square on to "
        "its large faces (design 6.7.4)")


def _run_ls(args, parser: argparse.ArgumentParser) -> int:
    source = Path(args.viewpoints)
    if not source.exists():
        parser.error(f"viewpoint file not found: {source}")
    try:
        viewpoints = read_json(source)
    except json.JSONDecodeError as error:
        parser.error(f"{source} is not valid JSON: {error}")
    if not viewpoints:
        parser.error(f"{source} contains no viewpoints")

    try:
        contract = export_ls(viewpoints, args.out, prog_name=args.prog_name,
                             place=args.place, uframe=args.uframe, utool=args.utool,
                             motion=args.motion, speed=args.speed)
    except ValueError as error:
        parser.error(str(error))
        return 2

    print(f"wrote {args.prog_name}.LS, {args.prog_name}_points.csv and "
          f"{args.prog_name}_frame.json to {Path(args.out).resolve()}")
    print(f"UFRAME {contract['uframe_utool']['uframe']}, "
          f"UTOOL {contract['uframe_utool']['utool']}")
    print("before reading anything into reachability: confirm in ROBOGUIDE that the "
          "points land ON THE PART. A wrong frame contract moves every point by a "
          "constant and nothing complains.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
