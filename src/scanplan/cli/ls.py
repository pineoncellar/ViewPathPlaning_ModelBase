"""`scanplan ls`: export a planned viewpoint set as a FANUC `.LS` program."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..config import load_config, read_json
from ..registration import load_registration
from ..roboguide.adapt import export as export_ls
from ..roboguide.registration_report import write_registration_report
from ..roboguide.wpr import rot_to_fanuc_wpr

NAME = "ls"


def configure(subparsers) -> None:
    exporter = subparsers.add_parser(
        NAME, help="export a planned viewpoint set as a FANUC .LS program")
    exporter.add_argument("--viewpoints", required=True, help="viewpoints.json")
    exporter.add_argument("--prog-name", required=True, help="FANUC program name")
    exporter.add_argument("--out", default="output", help="output directory")
    exporter.add_argument("--place", help="pose 'x,y,z,w,p,r' of the CAD frame in the "
                                         "ROBOGUIDE world; switches the program to UFRAME 0")
    exporter.add_argument(
        "--registration",
        help="JSON with three matching CAD and robot-base datum points; registers the "
             "viewpoints to the measured part installation and uses UFRAME 0",
    )
    exporter.add_argument(
        "--max-registration-error-mm",
        type=float,
        default=3.0,
        help="maximum allowed residual for any registration point (default: 3 mm)",
    )
    exporter.add_argument("--uframe", type=int, help="UFRAME number (default 1, or 0 "
                                                    "with --place or --registration)")
    exporter.add_argument("--utool", type=int, default=1)
    exporter.add_argument("--motion", default="J", choices=["J", "L"],
                          help="J is the permissive default for a reachability review; a "
                               "stroke plan ignores this and uses the configuration")
    exporter.add_argument("--speed", default="100%")
    exporter.add_argument("--termination", default="FINE", choices=["FINE", "CNT0", "CNT100"],
                          help="a stroke plan ignores this and uses the configuration")
    exporter.add_argument("--config", help="configuration override (JSON); it is authoritative "
                                          "for motion, termination and speed when the "
                                          "viewpoint file is a stroke plan")
    exporter.add_argument("--joint-blocks", action="store_true",
                          help="write P[n] as joint angles where the viewpoint file has "
                               "them; the controller then has no branch to choose")


def run(args, parser: argparse.ArgumentParser) -> int:
    source = Path(args.viewpoints)
    if not source.exists():
        parser.error(f"viewpoint file not found: {source}")
    try:
        viewpoints = read_json(source)
    except json.JSONDecodeError as error:
        parser.error(f"{source} is not valid JSON: {error}")
    if not viewpoints:
        parser.error(f"{source} contains no viewpoints")

    registration = None
    if args.registration:
        if args.place:
            parser.error("--registration cannot be combined with --place")
        if not np.isfinite(args.max_registration_error_mm) \
                or args.max_registration_error_mm <= 0.0:
            parser.error("--max-registration-error-mm must be a positive finite number")
        try:
            registration = load_registration(args.registration)
        except (ValueError, FileNotFoundError, json.JSONDecodeError) as error:
            parser.error(f"cannot read registration: {error}")
        if args.uframe not in (None, registration.uframe):
            parser.error(
                f"registration measurements use {registration.target_frame}; "
                f"the matching FANUC frame is UFRAME {registration.uframe}"
            )
        if registration.max_error_mm > args.max_registration_error_mm:
            parser.error(
                f"registration max point error is {registration.max_error_mm:.3f} mm, "
                f"exceeding the {args.max_registration_error_mm:.3f} mm limit; "
                "check point correspondence and measurement coordinates"
            )

    stroke_plan = any(isinstance(item, dict) and item.get("stroke_id") is not None
                      for item in viewpoints)
    motion, speed, termination, order = args.motion, args.speed, args.termination, True
    try:
        # Loaded even for a pose-only export: the robot block is what the joint audit runs
        # with, and an export that cannot say whether the arm reaches its points should not
        # be written at all.
        cfg = load_config(args.config)
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as error:
        parser.error(f"cannot read the configuration: {error}")
        return 2
    if stroke_plan:
        # The configuration is authoritative here: a stroke plan's motion, termination and
        # speed are one decision, and two sources for it is how a program ends up flown at
        # a speed the coverage was not computed for.
        motion = cfg.motion.type
        termination = cfg.motion.termination
        speed = f"{cfg.motion.speed_mm_s:g} mm/sec"
        order = False
        print(f"stroke plan: {motion} {speed} {termination} from the configuration, and the "
              "planner's own order is kept (re-sorting would split strokes)")

    place = args.place
    if registration is not None:
        wpr = rot_to_fanuc_wpr(registration.rotation)
        place = ",".join(
            f"{value:.10g}" for value in (*registration.translation, *wpr)
        )

    try:
        contract = export_ls(viewpoints, args.out, prog_name=args.prog_name,
                             place=place,
                             uframe=(registration.uframe if registration is not None
                                     else args.uframe),
                             utool=args.utool,
                             motion=motion, speed=speed, termination=termination,
                             joint_blocks=bool(args.joint_blocks),
                             order=order,
                             robot=cfg.robot)
    except ValueError as error:
        parser.error(str(error))
        return 2

    if registration is not None:
        write_registration_report(args.out, contract, registration, args.registration)

    print(f"wrote {args.prog_name}.LS, {args.prog_name}_points.csv and "
          f"{args.prog_name}_frame.json to {Path(args.out).resolve()}")
    if registration is not None:
        print(f"registered CAD path to {registration.frame_label} using three datum "
              f"points (max error {registration.max_error_mm:.3f} mm)")
        print(f"registration report: {Path(args.out).resolve() / 'registration_report.json'}")
    print(f"UFRAME {contract['uframe_utool']['uframe']}, "
          f"UTOOL {contract['uframe_utool']['utool']}")
    print("before reading anything into reachability: confirm in ROBOGUIDE that the "
          "points land ON THE PART. A wrong frame contract moves every point by a "
          "constant and nothing complains.")
    return 0
