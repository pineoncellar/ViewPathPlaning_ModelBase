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


def test_a_utf8_bom_is_accepted(tmp_path):
    # Windows editors and PowerShell's `Set-Content -Encoding utf8` write a BOM, and a
    # config file that refuses to load because of one is a trap, not a strictness
    override = tmp_path / "bom.json"
    override.write_bytes(b"\xef\xbb\xbf" + json.dumps(
        {"scanner": {"max_incidence_deg": 55.0}}).encode("utf-8"))
    cfg = load_config(override)
    assert cfg.scanner.max_incidence_deg == 55.0


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


def test_the_acceptance_threshold_is_a_configurable_nullable_setting(tmp_path):
    cfg = load_config()
    assert cfg.target.coverage_target_pct == 99.0
    assert cfg.target.delivery_mode == "path"
    # null is the switch back to "run until nothing new is covered", so it must survive
    # the loader rather than being coerced
    exhaust = tmp_path / "exhaust.json"
    exhaust.write_text(json.dumps({"target": {"coverage_target_pct": None}}), encoding="utf-8")
    assert load_config(exhaust).target.coverage_target_pct is None
    tuned = tmp_path / "tuned.json"
    tuned.write_text(json.dumps({"target": {"coverage_target_pct": 97}}), encoding="utf-8")
    assert load_config(tuned).target.coverage_target_pct == pytest.approx(97.0)


@pytest.mark.parametrize("value", [0.0, -1.0, 100.5, 1000.0])
def test_a_threshold_outside_its_range_is_refused_not_clamped(tmp_path, value):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"target": {"coverage_target_pct": value}}), encoding="utf-8")
    with pytest.raises(ValueError, match="coverage_target_pct"):
        load_config(bad)


def test_the_point_count_knobs_are_validated(tmp_path):
    for section, key, value, match in (
        ("target", "delivery_mode", "fast", "delivery_mode"),
        ("stroke", "cost_model", "millimetres-squared", "cost_model"),
        ("stroke", "connector_level", 3, "connector_level"),
        ("stroke", "transition_corners_estimate", -1.0, "transition_corners_estimate"),
        ("stroke", "order_axis_weight_mm_per_deg", -1.0, "order_axis_weight"),
    ):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({section: {key: value}}), encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            load_config(bad)


def test_the_shipped_profiles_are_all_loadable():
    """The three profiles the workshop runs by hand must stay valid together."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "configs"
    for name in ("stroke_on.json", "lean.json", "legacy_exhaust.json"):
        cfg = load_config(root / name)
        assert cfg.target.delivery_mode in ("path", "lean")
        assert cfg.stroke.cost_model in ("corners", "millimetres")
    lean = load_config(root / "lean.json")
    assert lean.target.delivery_mode == "lean" and lean.stroke.enabled is False
    legacy = load_config(root / "legacy_exhaust.json")
    assert legacy.target.coverage_target_pct is None
    assert legacy.stroke.connector_level == 2
    assert legacy.stroke.order_axis_weight_mm_per_deg == 0.0


def test_a_runtime_override_is_validated_like_a_file():
    from scanplan.config import override

    cfg = load_config()
    tightened = override(cfg, target={"coverage_target_pct": 97.5})
    assert tightened.target.coverage_target_pct == pytest.approx(97.5)
    assert cfg.target.coverage_target_pct == 99.0          # the original is untouched
    with pytest.raises(ValueError, match="coverage_target_pct"):
        override(cfg, target={"coverage_target_pct": 0.0})
