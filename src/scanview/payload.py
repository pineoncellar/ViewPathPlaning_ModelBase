"""The data half of the 3D view: turn one planning run into a single HTML file.

Writes ONE `.html` file with no external assets and no network access: the meshes, the
target samples and the viewpoint set are embedded as base64 buffers and drawn by the
WebGL renderer kept beside this module in `viewer.html`. That matters on the shop floor,
where the file is opened on a PC that has never seen Python - a CDN reference would
simply render nothing there.

What the page is for: a reviewer has to see WHERE the scanner poses sit relative to the
part before accepting the set, and a table of coordinates cannot answer that. The page
draws the scanner reference point, the optical axis out to the working distance and the
rectangular footprint at that distance, using the planner's own
:func:`scanplan.geometry.scanner_frame` and footprint - so the picture cannot drift from
the exported pose contract.

This package depends on `scanplan` for those two definitions and for nothing else. The
planner depends on it only where a run is written, so the view can be re-rendered,
restyled or dropped without touching the planning algorithm.

Meshes are decimated only when a model exceeds the display budget. A stamped blank is
thin enough that spatial clustering welds its two faces together, so the budget is high
by default and lowering it is a deliberate choice, not a routine one.
"""

from __future__ import annotations

import base64
import json
from functools import lru_cache
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from scanplan.config import ScannerParams, load_config
from scanplan.geometry import scanner_frame, unit
from scanplan.scanner_model import ScannerModel
from scanplan.tracker import Tracker

DEFAULT_MAX_FACES = 250_000
VIEWER_FILE_NAME = "viewpoints_3d.html"
TARGET_FILE_NAME = "target_samples.npz"
_TEMPLATE_NAME = "viewer.html"
_PLACEHOLDER_SCENE = "__SCENE_JSON__"
_PLACEHOLDER_TITLE = "__TITLE__"


def write_viewer(path, *, part, viewpoints, fixture=None, base=None,
                 target_points=None, target_covered=None,
                 target_mask: np.ndarray | None = None,
                 tracker: Tracker | None = None,
                 scanner: ScannerParams | None = None,
                 max_faces: int = DEFAULT_MAX_FACES, title: str | None = None,
                 stats: dict[str, Any] | None = None) -> Path:
    """Write the single-file 3D view and return its path.

    `viewpoints` accepts the delivered records (`Viewpoint` instances or the dicts a
    `viewpoints.json` file holds), so re-viewing an old run needs no meshes of its own.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload(part=part, viewpoints=viewpoints, fixture=fixture, base=base,
                            target_points=target_points,
                            target_covered=target_covered, target_mask=target_mask,
                            tracker=tracker,
                            scanner=scanner, max_faces=max_faces, title=title, stats=stats)
    document = (_template().replace(_PLACEHOLDER_TITLE, escape(str(payload["title"])))
                .replace(_PLACEHOLDER_SCENE, _json_for_script(payload)))
    target.write_text(document, encoding="utf-8")
    return target


def build_payload(*, part, viewpoints, fixture=None, base=None, target_points=None,
                  target_covered=None, target_mask: np.ndarray | None = None,
                  tracker: Tracker | None = None,
                  scanner: ScannerParams | None = None,
                  max_faces: int = DEFAULT_MAX_FACES, title: str | None = None,
                  stats: dict[str, Any] | None = None) -> dict:
    """Everything the page needs, as plain JSON-ready data.

    Split from :func:`write_viewer` so the payload can be asserted without parsing a
    multi-megabyte document.
    """
    meshes = {"part": mesh_payload(part, max_faces)}
    if fixture is not None:
        meshes["fixture"] = mesh_payload(fixture, max_faces)
    if base is not None:
        meshes["base"] = mesh_payload(base, max_faces)

    params = scanner if scanner is not None else load_config().scanner
    records = [_viewpoint_payload(record, params) for record in _coerce(viewpoints)]
    centre, radius = _fit(part, records, tracker)

    return {
        "title": title or "scanplan - 零件与视点集",
        "units": "mm",
        "frame": "station CAD frame",
        "fit": {"center": _round(centre), "radius": round(float(radius), 4)},
        "stats": _clean_stats(stats),
        "meshes": meshes,
        "viewpoints": records,
        "scanner": {
            "standoff_mm": float(params.standoff_mm),
            "fov_at_standoff_mm": [float(v) for v in params.fov_at_standoff_mm],
        },
        "tracker": _tracker_payload(tracker),
        "target": _target_payload(target_points, target_covered, target_mask),
    }


def mesh_payload(mesh, max_faces: int = DEFAULT_MAX_FACES) -> dict:
    """The display buffer for one mesh: positions, quantized normals, indices.

    Normals are quantized to int8 because nothing downstream needs better than a
    degree for shading, and the byte count is what decides whether a 100k-face model
    still opens smoothly in a browser.
    """
    faces = np.asarray(mesh.faces, dtype=np.int64)
    displayed = decimate_for_display(mesh, max_faces)
    vertices = np.asarray(displayed.vertices, dtype=float)
    corners = np.asarray(displayed.faces, dtype=np.int64)
    normals = _vertex_normals(displayed, vertices)

    wide = len(vertices) > 65535
    indices = corners.astype(np.uint32 if wide else np.uint16)
    return {
        "pos": _b64(np.ascontiguousarray(vertices, dtype="<f4")),
        "nrm": _b64(np.clip(np.rint(normals * 127.0), -127, 127).astype(np.int8)),
        "idx": _b64(indices),
        "indexType": "uint32" if wide else "uint16",
        "vertices": int(len(vertices)),
        "faces": int(len(corners)),
        "sourceFaces": int(len(faces)),
        "decimated": bool(len(corners) != len(faces)),
    }


def decimate_for_display(mesh, max_faces: int) -> trimesh.Trimesh:
    """A display-only copy under `max_faces`, by vertex clustering.

    Clustering is the right tool for the fixture and the base, and the wrong one for a
    thin blank: a cell larger than the material thickness merges the two faces. The
    binary search therefore only ever searches cell sizes that keep enough faces, and
    callers that care about the blank keep `max_faces` above its face count.
    """
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if max_faces <= 0 or len(faces) <= max_faces:
        return mesh

    vertices = np.asarray(mesh.vertices, dtype=float)
    span = float(np.linalg.norm(mesh.extents)) or 1.0
    low, high = span / 4000.0, span / 8.0
    best: tuple[np.ndarray, np.ndarray] | None = None
    for _ in range(18):
        cell = float(np.sqrt(low * high))
        clustered, kept = _cluster(vertices, faces, cell)
        if 0 < len(kept) <= max_faces:
            best = (clustered, kept)
            high = cell
        else:
            low = cell
        if best is not None and len(best[1]) <= max_faces * 0.85:
            break
    if best is None:
        best = _cluster(vertices, faces, high)
    return trimesh.Trimesh(vertices=best[0], faces=best[1], process=False)


def _cluster(vertices: np.ndarray, faces: np.ndarray,
             cell: float) -> tuple[np.ndarray, np.ndarray]:
    """Merge the vertices in each grid cell and drop the triangles that collapse."""
    lattice = np.floor(vertices / cell).astype(np.int64)
    lattice -= lattice.min(axis=0)
    # One integer key per cell keeps the clustering O(n log n) on a million vertices;
    # `np.unique(..., axis=0)` on the three-column array is several times slower.
    limit = max(int(lattice.max()), 1) + 1
    key = (lattice[:, 0] * limit + lattice[:, 1]) * limit + lattice[:, 2]
    _, inverse = np.unique(key, return_inverse=True)
    counts = np.bincount(inverse, minlength=int(inverse.max()) + 1)
    totals = np.zeros((len(counts), 3), dtype=float)
    np.add.at(totals, inverse, vertices)
    merged = totals / counts[:, None]

    remapped = inverse[faces]
    keeps = ((remapped[:, 0] != remapped[:, 1]) & (remapped[:, 1] != remapped[:, 2])
             & (remapped[:, 0] != remapped[:, 2]))
    remapped = remapped[keeps]
    if len(remapped) == 0:
        return merged, remapped
    _, first = np.unique(np.sort(remapped, axis=1), axis=0, return_index=True)
    return merged, remapped[np.sort(first)]


def _vertex_normals(mesh, vertices: np.ndarray) -> np.ndarray:
    normals = np.asarray(mesh.vertex_normals, dtype=float)
    if normals.shape == vertices.shape and np.isfinite(normals).all():
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        return np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)
    return np.zeros_like(vertices)


def _coerce(viewpoints) -> list[Any]:
    records = list(viewpoints)
    for record in records:
        position = _field(record, "position")
        axis = _field(record, "axis")
        if position is None or axis is None:
            raise ValueError("every viewpoint needs a position and an axis")
        if np.asarray(position, dtype=float).shape != (3,):
            raise ValueError(f"a viewpoint position must be x,y,z: {position!r}")
        if np.asarray(axis, dtype=float).shape != (3,):
            raise ValueError(f"a viewpoint axis must be x,y,z: {axis!r}")
    return records


def _field(record, name: str, default=None):
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


def _viewpoint_payload(record, params: ScannerParams) -> dict:
    position = np.asarray(_field(record, "position"), dtype=float)
    axis = unit(_field(record, "axis"))
    roll = float(_field(record, "roll_deg", 0.0) or 0.0)

    model = ScannerModel(params)
    width, height = model.footprint_mm(params.standoff_mm)
    frame = scanner_frame(axis, roll)
    aim = position + axis * float(params.standoff_mm)
    half_w, half_h = float(width) / 2.0, float(height) / 2.0
    corners = [aim + frame[:, 0] * sx * half_w + frame[:, 1] * sy * half_h
               for sx, sy in ((1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0))]

    return {
        "id": int(_field(record, "id", 0) or 0),
        "position": _round(position),
        "axis": _round(axis, 9),
        "roll_deg": roll,
        "aim": _round(aim),
        "corners": [_round(corner) for corner in corners],
        "n_points": int(_field(record, "n_points", 0) or 0),
        "area_mm2": _number(_field(record, "area_mm2")),
        "mean_incidence_deg": _number(_field(record, "mean_incidence_deg")),
        "max_incidence_deg": _number(_field(record, "max_incidence_deg")),
        "tracker_ok": bool(_field(record, "tracker_ok", True)),
        "source": str(_field(record, "source", "") or ""),
    }


def _target_payload(points, covered, mask=None) -> dict | None:
    """The sampled target surface plus, per sample, WHICH viewpoints cover it.

    `mask` is one row per sample and `ceil(viewpoints / 8)` bytes per row, bit `i` set
    when viewpoint `i` (in delivered order) can measure that sample. A bitmask rather
    than index lists because the delivered set is small and the row length is fixed:
    the whole thing is ~1.8 MB for 119k samples x 117 poses, and the page can answer
    "who sees this point" by reading one row.
    """
    if points is None:
        return None
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if covered is None:
        covered = np.zeros(len(points), dtype=bool)
    covered = np.asarray(covered, dtype=bool).reshape(-1)
    if len(covered) != len(points):
        raise ValueError("target points and their coverage mask must be the same length")

    payload = {
        "points": _b64(np.ascontiguousarray(points, dtype="<f4")),
        "covered": _b64(np.ascontiguousarray(covered, dtype=np.uint8)),
        "count": int(len(points)),
        "coveredCount": int(covered.sum()),
        "uncoveredCount": int((~covered).sum()),
        "mask": None,
        "stride": 0,
        "viewpoints": 0,
    }
    if mask is not None:
        mask = np.ascontiguousarray(mask, dtype=np.uint8)
        if mask.ndim != 2 or mask.shape[0] != len(points):
            raise ValueError("the coverage mask must have one row per target sample")
        payload.update({"mask": _b64(mask), "stride": int(mask.shape[1]),
                        "viewpoints": int(mask.shape[1] * 8)})
    return payload


def save_target_samples(path, *, points, covered, mask=None) -> Path:
    """The target samples and their per-viewpoint coverage, for `scanplan view`.

    Without this sidecar a re-rendered page could not show the sampled surface at all:
    the point cloud only ever existed in memory during the plan.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "points": np.asarray(points, dtype=np.float32).reshape(-1, 3),
        "covered": np.asarray(covered, dtype=bool).reshape(-1),
    }
    if mask is not None:
        body["mask"] = np.ascontiguousarray(mask, dtype=np.uint8)
    np.savez_compressed(target, **body)
    return target


def load_target_samples(path):
    """Read what :func:`save_target_samples` wrote; `(points, covered, mask|None)`."""
    with np.load(path, allow_pickle=False) as data:
        points = np.asarray(data["points"], dtype=float)
        covered = np.asarray(data["covered"], dtype=bool)
        mask = np.asarray(data["mask"], dtype=np.uint8) if "mask" in data else None
    return points, covered, mask


def coverage_mask(covered_sets, chosen, n_samples: int) -> np.ndarray:
    """Per-sample bitmask of which of the `chosen` poses can measure it.

    `covered_sets[i]` holds the sample indices pose `i` covers; `chosen` is the order
    the viewpoints are delivered in, so bit `k` belongs to `viewpoints[k]`.
    """
    stride = (len(chosen) + 7) // 8
    mask = np.zeros((int(n_samples), stride), dtype=np.uint8)
    for column, pose in enumerate(chosen):
        rows = np.asarray(covered_sets[pose], dtype=np.int64)
        if len(rows) == 0:
            continue
        mask[rows, column // 8] |= np.uint8(1 << (column % 8))
    return mask


def _tracker_payload(tracker: Tracker | None) -> dict | None:
    if tracker is None:
        return None
    return {
        "position": _round(tracker.position),
        "aim": _round(tracker.aim) if tracker.aim is not None else None,
    }


def _fit(part, records, tracker) -> tuple[np.ndarray, float]:
    blocks = [np.asarray(part.vertices, dtype=float)]
    if records:
        blocks.append(np.array([record["position"] for record in records], dtype=float))
        blocks.append(np.array([record["aim"] for record in records], dtype=float))
    if tracker is not None:
        blocks.append(np.asarray(tracker.position, dtype=float).reshape(1, 3))
    points = np.concatenate(blocks, axis=0)
    low, high = points.min(axis=0), points.max(axis=0)
    centre = (low + high) / 2.0
    radius = float(np.linalg.norm(high - low) / 2.0)
    return centre, max(radius, 1.0)


def _clean_stats(stats: dict[str, Any] | None) -> dict:
    if not stats:
        return {}
    cleaned = {}
    for key, value in stats.items():
        if value is None or isinstance(value, np.ndarray):
            continue
        cleaned[str(key)] = value.item() if isinstance(value, np.generic) else value
    return cleaned


def _number(value) -> float | None:
    if value is None:
        return None
    return round(float(value), 3)


def _round(values, decimals: int = 4) -> list[float]:
    return [round(float(value), decimals) for value in np.asarray(values, dtype=float)]


def _b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def _json_for_script(payload: dict) -> str:
    """JSON that is safe inside a <script> element."""
    return json.dumps(payload, separators=(",", ":"),
                      ensure_ascii=False).replace("</", "<\\/")


@lru_cache(maxsize=1)
def _template() -> str:
    return Path(__file__).with_name(_TEMPLATE_NAME).read_text(encoding="utf-8")
