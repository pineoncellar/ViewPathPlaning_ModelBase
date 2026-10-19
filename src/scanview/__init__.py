"""A self-contained 3D view of a station and its planned scanner viewpoints.

The entry points, in the order a run uses them:

* :func:`write_run_outputs` - what `scanplan plan` calls: writes the page plus the
  `target_samples.npz` sidecar that lets a re-render keep the sampled surface and
  answer "which viewpoints can measure this point";
* :func:`render_run` - what `scanplan view` calls: reads a run, loads the station and
  re-renders the page, with no planning involved;
* :func:`write_viewer` / :func:`write_viewer_from_run` - the lower-level writers, for a
  caller that already holds the meshes.

The package reads `scanplan` for the scanner frame, the field-of-view footprint and the
station loader, so the picture and the exported pose contract stay one definition.
Nothing in the planning algorithm depends on this package except the two call sites that
write or re-render a run.
"""

from __future__ import annotations

import json
from pathlib import Path

from scanplan.config import load_config, read_json
from scanplan.scene import StationDescription, load_models, load_station_description
from scanplan.tracker import tracker_from_description

from .payload import (
    DEFAULT_MAX_FACES,
    TARGET_FILE_NAME,
    VIEWER_FILE_NAME,
    build_payload,
    coverage_mask,
    decimate_for_display,
    load_target_samples,
    mesh_payload,
    save_target_samples,
    write_viewer,
)


def write_run_outputs(folder, *, part, fixture, base, viewpoints, target_points,
                      target_covered, coverage_sets, chosen, tracker=None,
                      scanner=None, max_faces: int = DEFAULT_MAX_FACES,
                      stats: dict | None = None) -> dict[str, Path]:
    """Write the page and its sample sidecar for a finished planning run.

    `coverage_sets[i]` is the target indices pose `i` can measure and `chosen` lists the
    delivered poses in the order `viewpoints` is in; together they produce the per-sample
    bitmask the page uses to answer "who sees this point".
    """
    folder = Path(folder)
    points = target_points
    mask = coverage_mask(coverage_sets, chosen, len(points))
    return {
        TARGET_FILE_NAME: save_target_samples(folder / TARGET_FILE_NAME, points=points,
                                              covered=target_covered, mask=mask),
        VIEWER_FILE_NAME: write_viewer(
            folder / VIEWER_FILE_NAME, part=part, fixture=fixture, base=base,
            viewpoints=viewpoints, target_points=points, target_covered=target_covered,
            target_mask=mask, tracker=tracker, scanner=scanner, max_faces=max_faces,
            stats=stats),
    }


def write_viewer_from_run(viewer_path, *, part, fixture, base, viewpoints,
                          samples_path=None, tracker=None, scanner=None,
                          max_faces: int = DEFAULT_MAX_FACES, title: str | None = None,
                          stats: dict | None = None) -> Path:
    """Re-render a page from a delivered viewpoint set, with its samples if they exist.

    Missing samples are not an error: an older run simply has no sidecar, and the page
    is written without the point cloud rather than inventing one.
    """
    points = covered = mask = None
    if samples_path is not None and Path(samples_path).exists():
        points, covered, mask = load_target_samples(samples_path)
    return write_viewer(viewer_path, part=part, fixture=fixture, base=base,
                        viewpoints=viewpoints, target_points=points,
                        target_covered=covered, target_mask=mask, tracker=tracker,
                        scanner=scanner, max_faces=max_faces, title=title, stats=stats)


def render_run(viewpoints_path, out_dir, *, config=None, station=None, part=None,
               fixture=None, base=None, max_faces: int = DEFAULT_MAX_FACES,
               log=None) -> Path:
    """Re-render a finished run's page: `scanplan view` is a thin wrapper over this.

    Loads only what the picture needs - the three models (already aligned by their
    station file) and the tracker pose if the station names one - so a reviewer can
    look again without paying for a plan.
    """
    say = log or (lambda message: print(message, flush=True))
    source = Path(viewpoints_path)
    if not source.exists():
        raise ValueError(f"viewpoint file not found: {source}")
    try:
        viewpoints = read_json(source)
    except json.JSONDecodeError as error:
        raise ValueError(f"{source} is not valid JSON: {error}") from error
    if not viewpoints:
        raise ValueError(f"{source} contains no viewpoints")

    cfg = load_config(config)
    description = _station_description(station, part, fixture, base)
    meshes = load_models(description)

    samples = source.with_name(TARGET_FILE_NAME)
    if not samples.exists():
        say(f"no {TARGET_FILE_NAME} beside the viewpoints: the page will show no target "
            "samples (re-run `scanplan plan` to produce them)")
        samples = None

    def writer(samples_path):
        return write_viewer_from_run(
            Path(out_dir) / VIEWER_FILE_NAME, part=meshes[0], fixture=meshes[1],
            base=meshes[2], viewpoints=viewpoints, samples_path=samples_path,
            tracker=tracker_from_description(description, cfg.tracker, part=meshes[0]),
            scanner=cfg.scanner, max_faces=max_faces, stats=_report_stats(source))

    try:
        return writer(samples)
    except (OSError, ValueError, KeyError) as error:
        if samples is None:
            raise
        say(f"cannot use {samples.name}: {error}; rendering without target samples")
        return writer(None)


def _station_description(station, part, fixture, base) -> StationDescription:
    if station:
        return load_station_description(station)
    given = {"part": part, "fixture": fixture, "base": base}
    missing = [name for name, value in given.items() if not value]
    if missing:
        raise ValueError("give --station, or all three of --part, --fixture and --base "
                         f"(missing: {', '.join('--' + name for name in missing)})")
    paths = {}
    for name, value in given.items():
        path = Path(value)
        if not path.exists():
            raise ValueError(f"the {name} mesh does not exist: {path}")
        paths[name] = path
    return StationDescription(part=paths["part"], fixture=paths["fixture"],
                              base=paths["base"])


def _report_stats(viewpoints_path: Path) -> dict:
    """The headline figures of a sibling coverage report, if the run left one."""
    report = viewpoints_path.with_name("coverage_report.json")
    if not report.exists():
        return {}
    try:
        data = read_json(report)
    except (OSError, json.JSONDecodeError):
        return {}
    return {key: data[key] for key in ("face", "coverage_pct") if data.get(key) is not None}


__all__ = [
    "DEFAULT_MAX_FACES",
    "TARGET_FILE_NAME",
    "VIEWER_FILE_NAME",
    "build_payload",
    "coverage_mask",
    "decimate_for_display",
    "load_target_samples",
    "mesh_payload",
    "render_run",
    "save_target_samples",
    "write_run_outputs",
    "write_viewer",
    "write_viewer_from_run",
]
