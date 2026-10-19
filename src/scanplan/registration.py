"""Register three CAD datum points to a measured robot coordinate frame."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import read_json


@dataclass(frozen=True)
class RigidRegistration:
    """A rigid transform from the viewpoint CAD frame to the robot base frame."""

    rotation: np.ndarray
    translation: np.ndarray
    cad_points: np.ndarray
    measured_points: np.ndarray
    residuals_mm: np.ndarray
    target_frame: str
    uframe: int

    @property
    def frame_label(self) -> str:
        return "robot base" if self.uframe == 0 else f"robot UFRAME {self.uframe}"

    @property
    def rms_error_mm(self) -> float:
        return float(np.sqrt(np.mean(np.sum(self.residuals_mm**2, axis=1))))

    @property
    def max_error_mm(self) -> float:
        return float(np.linalg.norm(self.residuals_mm, axis=1).max())


def load_registration(path: str | Path) -> RigidRegistration:
    """Load and fit the three paired points from a JSON registration file."""
    body = read_json(path)
    if not isinstance(body, dict):
        raise ValueError("registration JSON must be an object")
    target_frame, uframe = _target_frame(body)
    pairs = body.get("point_pairs")
    if pairs is not None:
        if not isinstance(pairs, list) or len(pairs) != 3:
            raise ValueError("registration point_pairs must contain exactly three point pairs")
        cad_points = []
        measured_points = []
        for index, pair in enumerate(pairs, start=1):
            if not isinstance(pair, dict):
                raise ValueError(f"registration point pair {index} must be an object")
            cad_points.append(_point(pair.get("cad_mm"), f"point pair {index} cad_mm"))
            key = f"{target_frame}_mm"
            measured_points.append(_point(pair.get(key), f"point pair {index} {key}"))
    else:
        cad_points = body.get("cad_points_mm")
        measured_points = body.get("measured_points_mm")
        if not isinstance(cad_points, list) or not isinstance(measured_points, list) \
                or len(cad_points) != 3 or len(measured_points) != 3:
            raise ValueError(
                "registration must contain three point_pairs or three cad_points_mm and "
                "measured_points_mm"
            )
        cad_points = [
            _point(point, f"CAD point {index}")
            for index, point in enumerate(cad_points, start=1)
        ]
        measured_points = [
            _point(point, f"measured point {index}")
            for index, point in enumerate(measured_points, start=1)
        ]

    return fit_three_points(
        np.asarray(cad_points), np.asarray(measured_points),
        target_frame=target_frame, uframe=uframe,
    )


def parse_measured_points(text: str) -> np.ndarray:
    """Parse three semicolon-separated XYZ triples supplied on the command line."""
    try:
        points = np.asarray(
            [[float(value.strip()) for value in item.split(",")]
             for item in str(text).split(";")],
            dtype=float,
        )
    except ValueError as error:
        raise ValueError(
            "measured points must be three 'x,y,z' triples separated by semicolons"
        ) from error
    if points.shape != (3, 3) or not np.isfinite(points).all():
        raise ValueError(
            "measured points must be exactly three finite 'x,y,z' triples "
            "separated by semicolons"
        )
    _require_non_collinear(points, "measured")
    return points


def fit_three_points(cad_points, measured_points, *,
                     target_frame: str = "robot_base", uframe: int = 0
                     ) -> RigidRegistration:
    """Find the least-squares proper rigid transform mapping CAD points to measured ones."""
    source = np.asarray(cad_points, dtype=float)
    target = np.asarray(measured_points, dtype=float)
    if source.shape != (3, 3) or target.shape != (3, 3):
        raise ValueError("three-point registration requires two arrays with shape (3, 3)")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("registration coordinates must all be finite numbers")
    _require_non_collinear(source, "CAD")
    _require_non_collinear(target, "measured")

    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    left, _, right_transpose = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(right_transpose.T @ left.T)
    rotation = right_transpose.T @ correction @ left.T
    translation = target_center - rotation @ source_center
    residuals = (source @ rotation.T + translation) - target

    return RigidRegistration(
        rotation=rotation,
        translation=translation,
        cad_points=source.copy(),
        measured_points=target.copy(),
        residuals_mm=residuals,
        target_frame=target_frame,
        uframe=uframe,
    )


def _target_frame(body: dict) -> tuple[str, int]:
    """Read the output frame from the new format or the existing field-file format."""
    if body.get("units", "mm") != "mm":
        raise ValueError("registration units must be millimetres ('mm')")
    coordinate_system = body.get("coordinate_system")
    if coordinate_system == "robot_base_mm":
        return "robot_base", 0
    match = re.fullmatch(r"robot_uframe_([1-9])_mm", str(coordinate_system))
    if match:
        return f"robot_uframe_{match.group(1)}", int(match.group(1))

    measured_frame = body.get("measured_coordinate_frame")
    match = re.fullmatch(r"robot UFRAME ([1-9])", str(measured_frame))
    if match:
        return f"robot_uframe_{match.group(1)}", int(match.group(1))
    if measured_frame == "robot base":
        return "robot_base", 0
    raise ValueError(
        "registration must identify coordinates as 'robot_base_mm', "
        "'robot_uframe_N_mm', or measured_coordinate_frame 'robot UFRAME N'"
    )


def _point(value, name: str) -> np.ndarray:
    point = np.asarray(value, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError(f"{name} must contain exactly three finite coordinates")
    return point


def _require_non_collinear(points: np.ndarray, name: str) -> None:
    centred = points - points.mean(axis=0)
    singular_values = np.linalg.svd(centred, compute_uv=False)
    if singular_values[0] <= 1e-9 or singular_values[1] <= singular_values[0] * 1e-6:
        raise ValueError(
            f"the three {name} datum points are collinear or too close together "
            "to determine a stable 3D rotation"
        )
