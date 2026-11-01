"""`scanplan plan`: plan scanner viewpoints for one face of a stamped blank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scanview import VIEWER_FILE_NAME

from ..config import (
    FACES,
    FAST_MIN_SAMPLE_GRID_MM,
    FAST_SAMPLE_GRID_MM,
    PLAN_MODES,
    ROUTE_STYLES,
    fast_preset,
    load_config,
    override,
    plan_preset,
    route_preset,
)
from ..raycast import backend_name
from ..report import summary_lines
from ..scene import StationDescription, load_station, load_station_description
from ..tracker import (
    Tracker,
    part_centroid,
    rank_placements,
    sweep_placements,
    tracker_from_description,
)

NAME = "plan"


def configure(subparsers) -> None:
    from scanview import DEFAULT_MAX_FACES

    planner = subparsers.add_parser(
        NAME, help="plan scanner viewpoints for one face of a stamped blank")
    planner.add_argument("--station", help="station JSON (paths, transforms, tracker)")
    planner.add_argument("--part", help="part mesh, if no --station is given")
    planner.add_argument("--fixture", help="fixture mesh, if no --station is given")
    planner.add_argument("--base", help="base mesh, if no --station is given")
    planner.add_argument("--config", help="configuration override (JSON)")
    planner.add_argument("--fast", action="store_true",
                         help="fast run: sample the target on a 10 mm grid and plan one "
                              "face, so an iteration takes ~2 min instead of ~29. For the "
                              "SHAPE of the path - the coverage figure is not an "
                              "acceptance result")
    planner.add_argument("--fast-grid", type=float, metavar="MM",
                         help=f"sampling grid for --fast (default "
                              f"{FAST_SAMPLE_GRID_MM:g}, minimum "
                              f"{FAST_MIN_SAMPLE_GRID_MM:g})")
    planner.add_argument("--no-occlusion", action="store_true",
                         help="inside a fast run, do not test lines of sight at all "
                              "(faster, but a pose may then 'cover' a sample through the "
                              "part); requires --fast")
    planner.add_argument("--coverage-target", dest="coverage_target",
                         help="acceptance coverage in percent, or 'none' to run until the "
                              "coverage stops growing; overrides the configuration for "
                              "this run")
    planner.add_argument("--route-style", dest="route_style", choices=ROUTE_STYLES,
                         help="which set of connector and sequence settings to plan with: "
                              "'direct' is the shipped chain (a joint move between sweeps), "
                              "'tidy' flies straight and sidesteps only when blocked, so the "
                              "program reads as fewer, straighter transitions")
    planner.add_argument("--plan-mode", dest="plan_mode", choices=PLAN_MODES,
                         help="what the selection is ALLOWED to use: 'standard' is the "
                              "shipped behaviour, 'banded' adds a dense tilted seed family, "
                              "gives up the self-imposed joint envelope shrink and prices a "
                              "band like a pose, so more of the part is swept instead of "
                              "stopped on - it costs point count and it rides the joint "
                              "limits")
    planner.add_argument("--stroke-corner-cost", dest="stroke_corner_cost", type=float,
                         metavar="CORNERS",
                         help="what one SWEEP costs the selection next to a single pose's "
                              "1.0, when stroke.cost_model is 'corners'. 2.0 is the true "
                              "count of corners a band needs and 1.0 prices it like a pose, "
                              "which sweeps more of the part at the same coverage. Range "
                              "[1.0, 2.0] is the useful part")
    planner.add_argument("--face", choices=FACES, help="which face of the blank to scan")
    planner.add_argument("--tracker", help="tracker pose 'x,y,z' if the station has none")
    planner.add_argument("--tracker-aim", help="tracker aim point 'x,y,z'")
    planner.add_argument("--tracker-sweep", action="store_true",
                         help="report the spread over the 192 candidate placements")
    planner.add_argument("--out", default="output", help="output directory")
    planner.add_argument("--no-3d", action="store_true",
                         help="skip the 3D view (the rest of the outputs are unaffected)")
    planner.add_argument("--max-faces", type=int, default=DEFAULT_MAX_FACES,
                         help="display budget per model in the 3D view")
    planner.add_argument("--quiet", action="store_true", help="only the summary")


def run(args, parser: argparse.ArgumentParser) -> int:
    from ..pipeline import plan, write_outputs

    say = (lambda message: None) if args.quiet else (lambda message: print(message, flush=True))
    try:
        cfg = load_config(args.config)
        cfg = _apply_overrides(cfg, args, parser)
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as error:
        parser.error(f"cannot read the configuration: {error}")
        return 2

    if cfg.stroke.route_style != "direct":
        say(f"route style '{cfg.stroke.route_style}': connector "
            f"{cfg.stroke.connector}, CONFIG-family change "
            f"{cfg.robot.config_change_cost_deg:g} deg, wrist flip "
            f"{cfg.robot.wrist_flip_cost_deg:g} deg - the point count and the program "
            "shape will differ from a 'direct' run on the same part")

    if cfg.stroke.plan_mode != "standard":
        say(f"plan mode '{cfg.stroke.plan_mode}': tilted seed family on "
            f"{int(cfg.stroke.side_scan_azimuths) * 2} axes every "
            f"{cfg.stroke.side_scan_spacing_mm:g} mm, joint envelope "
            f"{cfg.robot.joint_span_fraction:g} of span with "
            f"{cfg.robot.joint_margin_deg:g} deg margin, a band priced at "
            f"{cfg.stroke.stroke_corner_cost:g} corner(s) - more of the part is swept, "
            "at the cost of point count and of riding the joint limits")

    if cfg.fast.enabled:
        say(f"FAST run: {', '.join(cfg.fast.layers) or 'preset'}; "
            f"{cfg.target.sample_grid_mm:g} mm grid, face {cfg.target.face}, sight lines "
            f"{'NOT ' if cfg.fast.no_occlusion else ''}modelled - these numbers describe "
            "the SHAPE of the path, not an acceptance result")

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

    paths = write_outputs(result, args.out, write_3d=not args.no_3d,
                          max_faces=args.max_faces)
    print()
    print("\n".join(summary_lines(result.report)))
    print()
    print(f"wrote {len(paths)} files to {Path(args.out).resolve()}")
    if VIEWER_FILE_NAME in paths:
        print(f"3D view: open {paths[VIEWER_FILE_NAME].resolve()} in a browser "
              "(no network needed)")
    print("next: scanplan ls --viewpoints viewpoints.json --prog-name <NAME> --out <dir>")
    return 0


def _apply_overrides(cfg, args, parser: argparse.ArgumentParser):
    """Apply the flags that are not part of the configuration file merge.

    Order matters and is documented in the log: the JSON file goes on first (in
    `load_config`), the route style next, the fast preset after it, and an explicit
    `--coverage-target` last, so the most specific thing said on the command line always wins.
    """
    if args.fast_grid is not None and not args.fast:
        parser.error("--fast-grid only means something with --fast")
    if args.no_occlusion and not args.fast:
        parser.error("--no-occlusion only means something with --fast: switching sight "
                     "lines off is only honest inside a run that says it is fast")
    if getattr(args, "route_style", None):
        cfg = route_preset(cfg, args.route_style)
    if getattr(args, "plan_mode", None):
        # A separate axis from the route style, and applied first: the mode decides WHICH
        # items are selected, the style decides how they are connected, so an explicit
        # --stroke-corner-cost below still wins over the mode's own price.
        cfg = plan_preset(cfg, args.plan_mode)
    if getattr(args, "stroke_corner_cost", None) is not None:
        # After the route preset, so an explicit flag wins over whatever a style names -
        # this is a site's own judgement about how much of the tail it wants swept, not
        # part of a route's shape. `override` validates, and the ValueError it raises is
        # reported by the same handler that reads the configuration file.
        cfg = override(cfg, stroke={"stroke_corner_cost": float(args.stroke_corner_cost)})
    if args.fast:
        grid = FAST_SAMPLE_GRID_MM if args.fast_grid is None else float(args.fast_grid)
        if grid < FAST_MIN_SAMPLE_GRID_MM:
            parser.error(f"--fast-grid must be at least {FAST_MIN_SAMPLE_GRID_MM:g} mm; "
                         f"{grid:g} mm would cost the same time as an accurate run")
        cfg = fast_preset(cfg, grid=grid, no_occlusion=bool(args.no_occlusion))
    if args.coverage_target is not None:
        raw = str(args.coverage_target).strip().lower()
        if raw in ("none", "null", ""):
            value = None
        else:
            try:
                value = float(raw)
            except ValueError:
                parser.error(f"--coverage-target takes a percentage or 'none', not "
                             f"{args.coverage_target!r}")
                return cfg
        cfg = override(cfg, target={"coverage_target_pct": value})
    return cfg


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
    from ..candidates import generate

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
