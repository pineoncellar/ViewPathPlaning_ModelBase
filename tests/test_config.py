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
        ("stroke", "joint_travel_min_mm", 0.0, "joint_travel_min_mm"),
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
    # the profile keeps two inserted points per transition, which is what makes its recorded
    # point-count baseline comparable - at the price of the second one being a joint travel
    assert legacy.stroke.connector_level == 2
    assert legacy.stroke.order_axis_weight_mm_per_deg == 0.0
    # the joint route is the shipped default, so a site that wants straight-only has to
    # say so; the threshold that keeps short same-pose hops straight comes with it
    shipped = load_config()
    assert shipped.stroke.connector_level == 2
    assert shipped.stroke.joint_travel_min_mm > 0.0


def test_a_runtime_override_is_validated_like_a_file():
    from scanplan.config import override

    cfg = load_config()
    tightened = override(cfg, target={"coverage_target_pct": 97.5})
    assert tightened.target.coverage_target_pct == pytest.approx(97.5)
    assert cfg.target.coverage_target_pct == 99.0          # the original is untouched
    with pytest.raises(ValueError, match="coverage_target_pct"):
        override(cfg, target={"coverage_target_pct": 0.0})


def test_the_surface_profile_pins_its_three_anchors():
    """One semantic knob, and its two ends plus the shipped middle."""
    from scanplan.config import surface_profile

    covering = surface_profile(0.0)
    shipped = surface_profile(0.5)
    regular = surface_profile(1.0)
    assert covering == {"min_gain_mm2_per_step": 50.0, "min_stroke_mm": 60.0,
                        "seed_dirs": 4, "order_axis_weight_mm_per_deg": 1.0,
                        "parallel_weight": 1.0, "repair_budget_share": 0.35}
    assert regular == {"min_gain_mm2_per_step": 600.0, "min_stroke_mm": 300.0,
                       "seed_dirs": 2, "order_axis_weight_mm_per_deg": 3.5,
                       "parallel_weight": 1.3, "repair_budget_share": 0.05}
    # the middle anchor must BE the shipped behaviour, or every existing profile changes
    assert shipped == {"min_gain_mm2_per_step": 200.0, "min_stroke_mm": 150.0,
                       "seed_dirs": 4, "order_axis_weight_mm_per_deg": 2.0,
                       "parallel_weight": 1.0, "repair_budget_share": 0.20}


def test_the_surface_profile_is_monotone_in_its_bias():
    """A higher bias never grows a band further and never keeps a shorter fragment."""
    from scanplan.config import surface_profile

    steps = [surface_profile(bias / 20.0) for bias in range(21)]
    for earlier, later in zip(steps, steps[1:], strict=False):
        assert later["min_gain_mm2_per_step"] >= earlier["min_gain_mm2_per_step"]
        assert later["min_stroke_mm"] >= earlier["min_stroke_mm"]
        assert later["seed_dirs"] <= earlier["seed_dirs"]
        assert later["order_axis_weight_mm_per_deg"] >= earlier["order_axis_weight_mm_per_deg"]
        assert later["parallel_weight"] >= earlier["parallel_weight"]
        assert later["repair_budget_share"] <= earlier["repair_budget_share"]


def test_the_shipped_config_reproduces_the_middle_anchor():
    """At the shipped bias of 0.5 every derived key must equal its factory value."""
    from scanplan.config import surface_profile

    cfg = load_config()
    assert cfg.target.surface_bias == pytest.approx(0.5)
    shipped = surface_profile(0.5)
    for key, value in shipped.items():
        assert getattr(cfg.stroke, key) == pytest.approx(value), key
    assert cfg.target.max_points is None
    assert cfg.stroke.roll_mode == "reference"


@pytest.mark.parametrize("value", [-0.1, 1.1, 2.0])
def test_a_surface_bias_outside_its_range_is_refused(tmp_path, value):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"target": {"surface_bias": value}}), encoding="utf-8")
    with pytest.raises(ValueError, match="surface_bias"):
        load_config(bad)


def test_an_explicit_key_beats_the_profile_and_is_recorded_as_such(tmp_path):
    """A profile is a default generator, not a writer: an override always wins."""
    override = tmp_path / "explicit.json"
    override.write_text(json.dumps({
        "target": {"surface_bias": 1.0},
        "stroke": {"min_gain_mm2_per_step": 123.0},
    }), encoding="utf-8")
    cfg = load_config(override)
    assert cfg.stroke.min_gain_mm2_per_step == pytest.approx(123.0)
    assert cfg.stroke.min_stroke_mm == pytest.approx(300.0)   # the profile still applied
    assert cfg.derived["min_gain_mm2_per_step"] == "override"
    assert cfg.derived["min_stroke_mm"] == "profile"
    assert cfg.surface_bias_source == "override"


def test_a_runtime_surface_bias_re_derives_the_profile():
    """A knob that left the derived keys alone would be a knob that turns nothing."""
    from scanplan.config import override

    cfg = load_config()
    covering = override(cfg, target={"surface_bias": 0.0})
    assert covering.stroke.min_gain_mm2_per_step == pytest.approx(50.0)
    assert covering.stroke.min_stroke_mm == pytest.approx(60.0)
    assert covering.stroke.seed_dirs == 4
    assert covering.derived["min_gain_mm2_per_step"].startswith("profile(")
    # and an explicit key in the same call still wins
    explicit = override(cfg, target={"surface_bias": 1.0},
                        stroke={"min_gain_mm2_per_step": 42.0})
    assert explicit.stroke.min_gain_mm2_per_step == pytest.approx(42.0)
    assert explicit.derived["min_gain_mm2_per_step"] == "override"
    with pytest.raises(ValueError, match="surface_bias"):
        override(cfg, target={"surface_bias": 1.5})


def test_the_budget_and_wrist_knobs_are_validated(tmp_path):
    for section, key, value, match in (
        ("target", "max_points", 1, "max_points"),
        ("target", "max_points", 0, "max_points"),
        ("stroke", "roll_mode", "tangent", "roll_mode"),
        ("stroke", "max_wrist_step_deg", 0.0, "max_wrist_step_deg"),
        ("stroke", "max_wrist_step_deg", 181.0, "max_wrist_step_deg"),
        ("stroke", "roll_reference_bias", -0.1, "roll_reference_bias"),
        ("stroke", "parallel_weight", 0.0, "parallel_weight"),
        ("stroke", "repair_budget_share", 1.5, "repair_budget_share"),
    ):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({section: {key: value}}), encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            load_config(bad)


def test_a_point_cap_of_null_is_a_legal_setting():
    """It is the switch that keeps the budget layer out of the run entirely."""
    cfg = load_config()
    assert cfg.target.max_points is None
