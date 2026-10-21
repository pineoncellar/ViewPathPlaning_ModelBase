"""`scanplan pick-datums`: click three CAD datums and pair them with field measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..datum_picker import pick_cad_datums

NAME = "pick-datums"


def configure(subparsers) -> None:
    picker = subparsers.add_parser(
        NAME,
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
        choices=["robot_base_mm"],
        default="robot_base_mm",
        help="coordinate frame used to measure points; three-point registration only "
             "accepts robot_base_mm",
    )
    picker.add_argument(
        "--guide-registration",
        help="optional previous registration JSON whose CAD datum points are shown "
             "as click guides; points must still be selected in the viewer",
    )
    picker.add_argument("--out", required=True, help="registration JSON output path")


def run(args, parser: argparse.ArgumentParser) -> int:
    import trimesh

    from ..registration import fit_three_points, load_registration, parse_measured_points
    from ..scene import apply_transform, load_station_description, read_transform

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
        registration = fit_three_points(cad, measured)
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as error:
        parser.error(str(error))
        return 2

    body = {
        "coordinate_system": "robot_base_mm",
        "units": "mm",
        "point_pairs": [
            {"cad_mm": cad_point.tolist(), "robot_base_mm": measured_point.tolist()}
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
