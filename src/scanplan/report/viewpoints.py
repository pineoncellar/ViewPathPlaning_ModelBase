"""The viewpoint files a run ships: the JSON the exporter reads and the CSV for Excel."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def write_viewpoints_json(path, viewpoints) -> Path:
    """Write the viewpoint set, which is the input the `.LS` export reads."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps([viewpoint.as_dict() for viewpoint in viewpoints], indent=2,
                                 ensure_ascii=False), encoding="utf-8")
    return target


def write_viewpoints_csv(path, viewpoints) -> Path:
    """The same set as a flat table, for Excel."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["order", "vp_key", "x_mm", "y_mm", "z_mm",
                         "axis_x", "axis_y", "axis_z", "roll_deg", "n_points",
                         "area_mm2", "mean_incidence_deg", "max_incidence_deg",
                         "tracker_ok", "source"])
        for order, viewpoint in enumerate(viewpoints):
            record = viewpoint.as_dict()
            writer.writerow(
                [order, f"vp_{order:04d}",
                 *[f"{v:.4f}" for v in record["position"]],
                 *[f"{v:.9f}" for v in record["axis"]], f"{record['roll_deg']:.3f}",
                 record["n_points"], f"{record['area_mm2']:.3f}",
                 f"{record['mean_incidence_deg']:.3f}",
                 f"{record['max_incidence_deg']:.3f}",
                 int(record["tracker_ok"]), record["source"]])
    return target


def write_samples_csv(path, points, normals, reasons) -> Path:
    """A point cloud with one reason per point, for eyeballing in a CAD viewer."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    normals = np.asarray(normals, dtype=float).reshape(-1, 3)
    reasons = np.asarray(reasons, dtype=object)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x_mm", "y_mm", "z_mm", "nx", "ny", "nz", "reason"])
        for point, normal, reason in zip(points, normals, reasons, strict=True):
            writer.writerow([f"{point[0]:.4f}", f"{point[1]:.4f}", f"{point[2]:.4f}",
                             f"{normal[0]:.6f}", f"{normal[1]:.6f}", f"{normal[2]:.6f}",
                             str(reason)])
    return target
