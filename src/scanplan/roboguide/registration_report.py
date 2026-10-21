"""The calibration record that accompanies a registered `.LS` export.

Kept beside `adapt` because it edits that module's frame contract: the report and
the contract must not disagree about which frame the exported points are in.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..registration import RigidRegistration


def write_registration_report(
    out_dir: str | Path,
    contract: dict,
    registration: RigidRegistration,
    source: str | Path,
) -> None:
    """Record the calibration used to express the exported path in robot-base coordinates."""
    report = {
        "method": "least-squares proper rigid transform from three paired datum points",
        "source_frame": "CAD frame used by viewpoints.json",
        "target_frame": "robot base coordinates (UFRAME 0), millimetres",
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
    contract["uframe_utool"]["uframe_note"] = (
        "UFRAME 0 is the robot base frame the three registration datum points were "
        "measured in; the exported points are already expressed in that frame"
    )
    contract["frame"]["note"] = (
        "UFRAME 0 is the robot base frame. The registration transform was fitted "
        "from the three datum-point pairs recorded in registration_report.json."
    )
    contract["placement"]["meaning"] = (
        "the fitted rigid transform from the CAD frame into the robot base frame"
    )
    contract["placement"]["warning"] = (
        "the exported positions and orientations are expressed in the robot base "
        "frame; confirm that the three registration datum points were measured there"
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
