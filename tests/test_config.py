"""The planner configuration, and the placeholder values it carries.

To assert: the shipped file loads with its documented placeholder values; an
override file replaces only the keys it names; the dataclasses are frozen; and a
typo in a section or key name is refused rather than silently leaving a
placeholder in place.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from scanplan.config import load_config


def test_shipped_config_loads_with_placeholder_incidence():
    cfg = load_config()
    assert cfg.scanner.max_incidence_deg == 45.0
    assert cfg.scanner.range_mm == (120.0, 480.0)
    assert cfg.scanner.distances_mm == (250.0, 300.0, 380.0)
    assert cfg.target.face == "auto"
    assert cfg.contact_distance_mm == 8.0  # 3 mm contact + 5 mm margin
    assert cfg.tracker.distance_mm == (300.0, 6000.0)
    assert cfg.seed == 7


def test_override_file_replaces_only_what_it_names(tmp_path):
    override = tmp_path / "c.json"
    override.write_text(json.dumps({"scanner": {"max_incidence_deg": 55.0}}), encoding="utf-8")
    cfg = load_config(override)
    assert cfg.scanner.max_incidence_deg == 55.0
    assert cfg.scanner.standoff_mm == 300.0  # untouched


def test_dataclasses_are_frozen():
    cfg = load_config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.scanner.max_incidence_deg = 90.0


def test_a_typo_in_a_section_name_is_rejected(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"scannner": {"max_incidence_deg": 45.0}}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown config section"):
        load_config(bad)


def test_a_typo_in_a_key_name_is_rejected(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"scanner": {"max_incidence_degs": 45.0}}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown key"):
        load_config(bad)
