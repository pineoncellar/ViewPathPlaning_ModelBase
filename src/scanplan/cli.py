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
from .datum_picker import pick_cad_datums
from .raycast import backend_name
from .registration import (
    RigidRegistration,
    fit_three_points,
    load_registration,
    parse_measured_points,
)
from .report import summary_lines
from .roboguide.adapt import export as export_ls
from .roboguide.wpr import rot_to_fanuc_wpr
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
    if args.command == "pick-datums":
        return _run_pick_datums(args, parser)
    if args.command == "web":
        from .webui import serve

        serve(args.host, args.port)
        return 0
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
                                                    "with --place)")
    exporter.add_argument("--utool", type=int, default=1)
    exporter.add_argument("--motion", default="J", choices=["J", "L"],
                          help="J is the permissive default for a reachability review")
    exporter.add_argument("--speed", default="100%")

    picker = subparsers.add_parser(
        "pick-datums",
        help="click three CAD datum points and pair them with field measurements",
    )
    picker.add_argument("--station", required=True, help="station JSON describing the part CAD")
    picker.add_argument(
        "--measured-points",
        required=True,
        help="three measured 'x,y,z' triples separated by semicolons",
    )
    picker.add_argument(
        "--coordinate-system",
        choices=["robot_base_mm", "robot_uframe_1_mm"],
        default="robot_uframe_1_mm",
        help="coordinate frame used to measure points (default: robot_uframe_1_mm)",
    )
    picker.add_argument(
        "--guide-registration",
        help="optional previous registration JSON whose CAD datum points are shown "
             "as click guides; points must still be selected in the viewer",
    )
    picker.add_argument("--out", required=True, help="registration JSON output path")

    web = subparsers.add_parser(
        "web",
        help="start the local browser interface for station setup and path planning",
    )
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8765)
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


def _run_pick_datums(args, parser: argparse.ArgumentParser) -> int:
    import trimesh

    from .scene import apply_transform, load_station_description, read_transform

    try:
        measured = parse_measured_points(args.measured_points)
        description = load_station_description(args.station)
        reference_points = (
            load_registration(args.guide_registration).cad_points
            if args.guide_registration else None
        )
        loaded = trimesh.load(str(description.part))
        part = loaded.dump(concatenate=True) if isinstance(loaded, trimesh.Scene) else loaded
        part = apply_transform(part, read_transform(description.transform_part))
        cad = pick_cad_datums(part, reference_points=reference_points)
        registration = fit_three_points(
            cad,
            measured,
            target_frame=("robot_base" if args.coordinate_system == "robot_base_mm"
                          else "robot_uframe_1"),
            uframe=0 if args.coordinate_system == "robot_base_mm" else 1,
        )
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as error:
        parser.error(str(error))
        return 2

    measured_key = args.coordinate_system
    body = {
        "coordinate_system": args.coordinate_system,
        "units": "mm",
        "point_pairs": [
            {"cad_mm": cad_point.tolist(), measured_key: measured_point.tolist()}
            for cad_point, measured_point in zip(cad, measured, strict=True)
        ],
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(f"wrote three paired datum points to {output.resolve()}")
    print(f"registration check: RMS {registration.rms_error_mm:.3f} mm, "
          f"maximum {registration.max_error_mm:.3f} mm")
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
                             motion=args.motion, speed=args.speed)
    except ValueError as error:
        parser.error(str(error))
        return 2

    if registration is not None:
        _write_registration_outputs(args.out, contract, registration, args.registration)

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


def _write_registration_outputs(
    out_dir: str | Path,
    contract: dict,
    registration: RigidRegistration,
    source: str | Path,
) -> None:
    """Record the calibration used to express the exported path in robot-base coordinates."""
    report = {
        "method": "least-squares proper rigid transform from three paired datum points",
        "source_frame": "CAD frame used by viewpoints.json",
        "target_frame": f"{registration.frame_label} coordinates "
                        f"(UFRAME {registration.uframe}), millimetres",
        "source_file": str(Path(source).resolve()),
        "rotation_cad_to_target_frame": registration.rotation.tolist(),
        "translation_cad_origin_in_target_frame_mm": registration.translation.tolist(),
        "rms_error_mm": registration.rms_error_mm,
        "max_error_mm": registration.max_error_mm,
        "point_pairs": [
            {
                "cad_mm": cad.tolist(),
                "robot_base_mm": measured.tolist(),
                "residual_mm": residual.tolist(),
                "error_mm": float(np.linalg.norm(residual)),
            }
            for cad, measured, residual in zip(
                registration.cad_points,
                registration.measured_points,
                registration.residuals_mm,
                strict=True,
            )
        ],
        "warning": (
            "Confirm the measured points use the robot base coordinate system and "
            "that each CAD point matches the same physical datum before running."
        ),
    }
    output = Path(out_dir)
    (output / "registration_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    contract["frame"]["origin"] = (
        "CAD origin transformed into the robot base coordinate frame by the "
        "three-point registration"
    )
    contract["frame"]["axes"] = (
        "robot base axes; all exported positions and orientations are expressed "
        "in the robot base frame"
    )
    contract["frame"]["note"] = (
        f"UFRAME {registration.uframe} is used. The registration transform was fitted "
        "from the three datum-point pairs recorded in registration_report.json."
    )
    contract["placement"]["meaning"] = (
        f"the fitted rigid transform from the CAD frame into robot UFRAME "
        f"{registration.uframe}"
    )
    contract["placement"]["warning"] = (
        f"the exported positions and orientations are expressed in robot UFRAME "
        f"{registration.uframe}; confirm that this frame matches the one used to "
        "measure the three registration datum points"
    )
    contract["placement"]["registration"] = {
        "method": report["method"],
        "source_file": report["source_file"],
        "rms_error_mm": report["rms_error_mm"],
        "max_error_mm": report["max_error_mm"],
        "report": "registration_report.json",
    }
    frame_path = output / contract["files"]["frame_contract"]
    frame_path.write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    sys.exit(main())
