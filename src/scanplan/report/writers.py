"""JSON/CSV primitives shared by every report file."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def plain(value):
    """Recursively convert numpy scalars and arrays so the report is JSON-serialisable."""
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_report(out_dir, report: dict) -> Path:
    """Write the machine-readable report."""
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "coverage_report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
