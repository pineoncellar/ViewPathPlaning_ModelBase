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


def test_the_joint_speed_ladder_prices_moves_by_size():
    """The shipped ladder is the conservative site choice, and it ends in a catch-all."""
    cfg = load_config()
    ladder = cfg.motion.joint_speed_ladder
    assert ladder == ((20.0, 50.0), (60.0, 20.0), (None, 10.0))
    assert cfg.motion.joint_speed_pct(5.0) == 50.0
    assert cfg.motion.joint_speed_pct(20.0) == 50.0
    assert cfg.motion.joint_speed_pct(20.5) == 20.0
    assert cfg.motion.joint_speed_pct(60.0) == 20.0
    assert cfg.motion.joint_speed_pct(179.0) == 10.0
    # an unmeasured move - the first of a program - takes the slowest bucket, never a guess
    assert cfg.motion.joint_speed_pct(None) == 10.0


def test_a_configuration_without_a_ladder_keeps_one_speed():
    """The old behaviour must remain the DEFAULT, so a file that never mentions the ladder
    exports the program this repository wrote before 2026-10-30."""
    from scanplan.config import MotionParams

    plain = MotionParams(type="L", termination="CNT0", speed_mm_s=100.0)
    assert plain.joint_speed_ladder == ((None, 100.0),)
    assert plain.joint_speed_pct(179.0) == 100.0
    assert plain.joint_speed_pct(None) == 100.0


@pytest.mark.parametrize("ladder, match", [
    ([], "cannot be empty"),
    ([[20.0, 50.0]], "must end with a null"),
    ([[20.0, 50.0], [60.0, 20.0]], "must end with a null"),
    ([[None, 50.0], [None, 10.0]], "not the last"),
    ([[20.0, 50.0], [20.0, 20.0], [None, 10.0]], "strictly increase"),
    ([[60.0, 50.0], [20.0, 20.0], [None, 10.0]], "strictly increase"),
    ([[0.0, 50.0], [None, 10.0]], "must be positive"),
    ([[20.0, 0.0], [None, 10.0]], "in \\(0, 100\\]"),
    ([[20.0, 150.0], [None, 10.0]], "in \\(0, 100\\]"),
])
def test_the_joint_speed_ladder_is_refused_when_it_cannot_be_read(tmp_path, ladder, match):
    """Two equal thresholds leave a bucket unreachable; no catch-all leaves big moves unpriced."""
    bad = tmp_path / "ladder.json"
    bad.write_text(json.dumps({"motion": {"type": "L", "termination": "CNT0",
                                          "speed_mm_s": 100.0,
                                          "joint_speed_ladder": ladder}}), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_config(bad)


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
    # 90.0 since 2026-10-29, down from 99.0: the pool ceiling with the do-not-ride-the-limit
    # policy is 91.19% on this station, so 99.0 was unattainable by geometry.
    assert cfg.target.coverage_target_pct == 90.0
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
    assert cfg.target.coverage_target_pct == 90.0          # the original is untouched
    with pytest.raises(ValueError, match="coverage_target_pct"):
        override(cfg, target={"coverage_target_pct": 0.0})


# -- the surface-bias table (design 2026-10-22) ------------------------------------------


def test_surface_profile_anchors_are_the_documented_ones():
    from scanplan.config import surface_profile

    zero = surface_profile(0.0)["stroke"]
    half = surface_profile(0.5)["stroke"]
    one = surface_profile(1.0)["stroke"]
    assert zero == {"min_gain_mm2_per_step": 50.0, "min_stroke_mm": 60.0, "seed_dirs": 4.0,
                    "order_axis_weight_mm_per_deg": 1.0, "parallel_weight": 1.0,
                    "repair_budget_share": 0.35}
    assert half == {"min_gain_mm2_per_step": 200.0, "min_stroke_mm": 150.0, "seed_dirs": 4.0,
                    "order_axis_weight_mm_per_deg": 2.0, "parallel_weight": 1.0,
                    "repair_budget_share": 0.20}
    assert one == {"min_gain_mm2_per_step": 600.0, "min_stroke_mm": 300.0, "seed_dirs": 2.0,
                   "order_axis_weight_mm_per_deg": 3.5, "parallel_weight": 1.3,
                   "repair_budget_share": 0.05}


def test_the_midpoint_profile_reproduces_the_shipped_values():
    """0.5 is the contract: the four pre-existing knobs must not move."""
    cfg = load_config()
    assert (cfg.stroke.min_gain_mm2_per_step, cfg.stroke.min_stroke_mm,
            cfg.stroke.seed_dirs, cfg.stroke.order_axis_weight_mm_per_deg) == \
        (200.0, 150.0, 4, 2.0)
    assert cfg.target.surface_bias == pytest.approx(0.5)
    assert cfg.target.max_points is None


def test_surface_profile_is_monotone_between_the_anchors():
    from scanplan.config import surface_profile

    values = [surface_profile(bias)["stroke"]["min_gain_mm2_per_step"]
              for bias in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert values == sorted(values)


def test_an_out_of_range_bias_is_refused_not_clamped(tmp_path):
    for bad in (-0.1, 1.1):
        path = tmp_path / "bias.json"
        path.write_text(json.dumps({"target": {"surface_bias": bad}}), encoding="utf-8")
        with pytest.raises(ValueError, match="surface_bias"):
            load_config(path)


def test_the_override_file_wins_over_the_profile(tmp_path):
    """A key written by hand must beat the derived value, and say so in the report."""
    path = tmp_path / "mixed.json"
    path.write_text(json.dumps({"target": {"surface_bias": 1.0},
                                "stroke": {"min_stroke_mm": 200.0}}), encoding="utf-8")
    cfg = load_config(path)
    assert cfg.stroke.min_gain_mm2_per_step == pytest.approx(600.0)   # from the profile
    assert cfg.stroke.min_stroke_mm == pytest.approx(200.0)           # written by hand
    assert cfg.parameter_sources["stroke.min_stroke_mm"] == "override"
    assert cfg.parameter_sources["stroke.min_gain_mm2_per_step"] == "profile(bias=1)"


def test_the_legacy_profile_turns_the_new_gates_off():
    """The regression profile must not inherit a behaviour change it cannot reproduce."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "configs"
    legacy = load_config(root / "legacy_exhaust.json")
    assert legacy.stroke.lock_roll is False
    assert legacy.stroke.scan_gate is False
    assert legacy.stroke.hard_gate is False


@pytest.mark.parametrize("budget,match", [(1, "max_points"), (0, "max_points")])
def test_a_budget_below_two_points_is_refused(tmp_path, budget, match):
    path = tmp_path / "budget.json"
    path.write_text(json.dumps({"target": {"max_points": budget}}), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_config(path)


# -- the robot block ----------------------------------------------------------------------


def test_the_robot_block_is_off_by_default_and_fully_typed():
    cfg = load_config()
    assert cfg.robot.enabled is False
    assert cfg.robot.soft_limits_deg[4] == (-125.0, 125.0)
    # nested tuples, not lists: the limit table is used as pairs by the kinematics
    assert isinstance(cfg.robot.soft_limits_deg[0], tuple)
    assert len(cfg.robot.mechanical_limits_deg) == 6


def test_a_soft_limit_wider_than_the_arm_is_refused(tmp_path):
    path = tmp_path / "robot.json"
    path.write_text(json.dumps({"robot": {
        "enabled": True,
        "soft_limits_deg": [[-200, 200], [-60, 136], [-301, 301],
                            [-720, 720], [-125, 125], [-360, 360]]}}), encoding="utf-8")
    with pytest.raises(ValueError, match="wider than the mechanical"):
        load_config(path)


def test_a_bad_joint_sign_is_refused(tmp_path):
    path = tmp_path / "robot.json"
    path.write_text(json.dumps({"robot": {"enabled": True, "j3_sign": 0}}), encoding="utf-8")
    with pytest.raises(ValueError, match="j3_sign"):
        load_config(path)


def test_j5_min_abs_must_leave_a_usable_band(tmp_path):
    """`j5_min_abs_deg` at or past half the range leaves nothing to choose from."""
    path = tmp_path / "robot.json"
    path.write_text(json.dumps({"robot": {"enabled": True, "j5_min_abs_deg": 125.0}}),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="j5_min_abs_deg"):
        load_config(path)
