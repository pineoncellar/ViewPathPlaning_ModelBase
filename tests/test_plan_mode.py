"""The two plan modes: what each expands to, and that `standard` is the identity.

`plan_mode` is a SEPARATE axis from `route_style`. A route style decides how the selected
items are connected; a plan mode decides what the selection may even choose from - which
candidate families exist, how big the joint envelope is, and what a band costs. Both can be
asked for at once and neither implies the other, so the tests come in three groups:

* the TABLE (both modes exist, `standard` restores every number it names),
* the TILTED FAMILY (one azimuth reproduces the pre-2026-11-01 seed set exactly),
* the PRICE (a band is charged what the mode says, and never below one stop).
"""

from __future__ import annotations

import numpy as np
import pytest

from scanplan.config import (
    PLAN_MODES,
    PLAN_PRESETS,
    load_config,
    override,
    plan_preset,
    route_preset,
)
from scanplan.path import ROLE_COVERAGE
from scanplan.pipeline.delivery import _group_selection
from scanplan.strokes import StrokeBuilder


def _config():
    return load_config()


# --------------------------------------------------------------------------- the table


def test_every_mode_has_a_preset_and_no_preset_is_orphaned():
    assert set(PLAN_PRESETS) == set(PLAN_MODES)


def test_standard_restores_every_number_the_mode_names():
    """Asking for `standard` on a `banded` configuration has to give back the shipped one.

    That is what lets the page or the command line say "go back to normal" without editing a
    file, and it is why the preset spells its values out instead of leaving them empty.
    """
    shipped = _config()
    banded = plan_preset(shipped, "banded")
    assert banded.stroke.plan_mode == "banded"
    back = plan_preset(banded, "standard")
    for name in ("side_scan_azimuths", "side_scan_spacing_mm", "stroke_corner_cost",
                 "plan_mode"):
        assert getattr(back.stroke, name) == getattr(shipped.stroke, name), name
    for name in ("joint_span_fraction", "joint_margin_deg"):
        assert getattr(back.robot, name) == getattr(shipped.robot, name), name


def test_banded_moves_exactly_the_three_things_it_says_it_moves():
    """One mode, one decision - but three settings, because moving any one alone does nothing.

    Measured 2026-11-01 on the track-scan station, same 10 mm grid and 99% target: the joint
    envelope shrink alone cuts the grown candidates from 216 to 43, and the tilted family at
    one azimuth every 200 mm never reaches a wall.
    """
    banded = plan_preset(_config(), "banded")
    assert banded.stroke.side_scan_azimuths > 1
    assert banded.stroke.side_scan_spacing_mm > 0.0
    assert banded.stroke.stroke_corner_cost < 2.0
    assert banded.robot.joint_span_fraction == 1.0
    assert banded.robot.joint_margin_deg == 0.0


def test_a_mode_and_a_route_style_are_independent():
    """Neither may imply the other: a style decides connection shape, a mode decides the pool."""
    config = _config()
    for mode in PLAN_MODES:
        for style in ("direct", "tidy"):
            both = plan_preset(route_preset(config, style), mode)
            assert both.stroke.plan_mode == mode
            assert both.stroke.route_style == style
            # and the mode did not disturb what the style owns
            assert both.stroke.connector == route_preset(config, style).stroke.connector


def test_an_unknown_mode_is_refused_by_name():
    with pytest.raises(ValueError, match="plan_mode"):
        plan_preset(_config(), "tidy-ish")


# --------------------------------------------------------------- the tilted seed family


class _StubBuilder:
    """`side_scan_seeds` without the rest of `StrokeBuilder`, which needs a station."""

    def __init__(self, params, standoff=300.0):
        self.params = params
        self.cfg = type("Cfg", (), {"scanner": type("S", (), {"standoff_mm": standoff})})()

    side_scan_seeds = StrokeBuilder.side_scan_seeds


def _flat_patch(count=5):
    side = np.linspace(-40.0, 40.0, count)
    grid = np.array([[x, y, 0.0] for x in side for y in side], dtype=float)
    normals = np.tile(np.array([0.0, 0.0, 1.0]), (len(grid), 1))
    return grid, normals


def test_one_azimuth_is_the_family_this_code_had_before_the_key_existed():
    """`side_scan_azimuths = 1` must reproduce the old two axes bit for bit.

    The old code laid `-tilt_axis(-normal, first * sign, tilt)` for the two signs, so the two
    axes are mirror images about the plane containing the normal and `first`: equal z, equal
    tilt off the inward normal, and their sum on the inward normal.
    """
    config = _config()
    points, normals = _flat_patch()
    builder = _StubBuilder(override(config, stroke={"side_scan_azimuths": 1,
                                                    "side_scan_spacing_mm": 0.0}).stroke)
    seeds = builder.side_scan_seeds(points, normals, aim_grid_mm=100.0)
    assert seeds, "the family must produce seeds at all"
    axes = {tuple(np.round(pose.axis, 6)) for pose, _index in seeds}
    assert len(axes) == 2, "one tangent direction, both signs"

    tilt = float(builder.params.side_scan_tilt_deg)
    inward = np.array([0.0, 0.0, -1.0])
    first, second = sorted(axes)
    one, other = np.array(first), np.array(second)
    for axis in (one, other):
        assert abs(float(np.linalg.norm(axis)) - 1.0) < 1e-5
        angle = np.degrees(np.arccos(float(np.dot(axis, inward))))
        assert angle == pytest.approx(tilt, abs=1e-4), (
            "the tilted family must look at the part, tilt degrees off the inward normal - "
            "the code it replaces negated the axis and pointed it at the back of the surface")
    # and it must actually see the surface, not the back of it
    assert float(np.dot(one, np.array([0.0, 0.0, 1.0]))) < 0.0
    # mirror images: same z, and the sum lies on the inward normal
    assert one[2] == pytest.approx(other[2], abs=1e-9)
    total = one + other
    assert total[0] == pytest.approx(0.0, abs=1e-9)
    assert total[1] == pytest.approx(0.0, abs=1e-9)
    assert total[2] < 0.0


def test_more_azimuths_put_more_axes_round_the_normal():
    config = _config()
    points, normals = _flat_patch()
    counts = {}
    for azimuths in (1, 3, 6):
        builder = _StubBuilder(override(
            config, stroke={"side_scan_azimuths": azimuths,
                            "side_scan_spacing_mm": 40.0}).stroke)
        seeds = builder.side_scan_seeds(points, normals, aim_grid_mm=100.0)
        counts[azimuths] = len({tuple(np.round(pose.axis, 6)) for pose, _ in seeds})
    assert counts[1] == 2
    assert counts[3] == 6
    assert counts[6] == 12


def test_a_finer_spacing_lays_more_seeds():
    config = _config()
    points, normals = _flat_patch()
    coarse = _StubBuilder(override(
        config, stroke={"side_scan_azimuths": 1, "side_scan_spacing_mm": 0.0}).stroke)
    fine = _StubBuilder(override(
        config, stroke={"side_scan_azimuths": 1, "side_scan_spacing_mm": 20.0}).stroke)
    assert len(fine.side_scan_seeds(points, normals, aim_grid_mm=100.0)) > \
        len(coarse.side_scan_seeds(points, normals, aim_grid_mm=100.0))


# --------------------------------------------------------------------------- the price


class _StubCoverage:
    """Two candidates and four samples: enough to see which one the price picks."""

    def __init__(self):
        self.n_targets = 4
        self.target_area_mm2 = 100.0
        self.area_mm2 = np.array([25.0, 25.0, 25.0, 25.0])

    def matrix(self):
        import scipy.sparse as sparse

        # a pose covers samples 0-1, a band covers 2-3; both are half the target
        return sparse.csr_matrix(np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=float))

    def pool_reach(self):
        return np.ones(self.n_targets, dtype=bool)

    @property
    def poses(self):
        return [object()]

    def __len__(self):
        return 1


def _band_selection(corner_cost):
    config = override(_config(), stroke={"stroke_corner_cost": corner_cost})
    return _group_selection(_StubCoverage(), _StubCoverage(),
                            type("Cov", (), {"strokes": [object()]})(), config, log=None)


def test_the_band_price_is_read_from_the_configuration():
    cheap = _band_selection(1.0)
    dear = _band_selection(2.0)
    # Same two candidates either way, and the band is charged what it was told.
    assert cheap.corners is not None and dear.corners is not None
    assert cheap.corners <= dear.corners


def test_a_band_cannot_be_priced_below_one_stop():
    config = _config()
    with pytest.raises(ValueError, match="stroke_corner_cost"):
        override(config, stroke={"stroke_corner_cost": 0.5})


def test_the_mode_owns_the_price_it_names():
    banded = plan_preset(_config(), "banded")
    assert banded.stroke.stroke_corner_cost == 1.0
    explicit = plan_preset(_config(), "banded")
    explicit = override(explicit, stroke={"stroke_corner_cost": 1.7})
    assert explicit.stroke.stroke_corner_cost == 1.7, (
        "an explicit price must still win over the mode, which is why the command line "
        "applies --stroke-corner-cost after --plan-mode")


# --------------------------------------------------------------------------- the report


def test_the_report_block_records_the_mode_and_the_numbers_it_stands_for():
    from scanplan.report.definition import route_block

    block = route_block(plan_preset(_config(), "banded"))
    assert block["plan_mode"] == "banded"
    settings = block["plan_mode_settings"]
    assert settings["side_scan_azimuths"] == 3
    assert settings["side_scan_wall_deg"] == 45.0
    assert settings["side_scan_max_seeds"] == 96
    assert settings["stroke_corner_cost"] == 1.0
    assert settings["joint_span_fraction"] == 1.0
    assert "banded" in block["caveat"]


def test_the_shipped_configuration_ships_the_standard_mode():
    """The default has to be the identity, or every stored baseline moves at once."""
    config = _config()
    assert config.stroke.plan_mode == "standard"
    assert config.stroke.side_scan_azimuths == 1
    assert config.stroke.side_scan_spacing_mm == 0.0
    assert config.stroke.stroke_corner_cost == 2.0
    assert config.robot.joint_span_fraction == 0.9
    assert config.robot.joint_margin_deg == 5.0


# --------------------------------------------------------------------------- the page


def test_the_page_can_ask_for_a_mode_and_omitting_it_keeps_the_file():
    """The web path builds a configuration per job; the mode is one of its inputs."""
    from scanweb import _delivery_config, _public_config

    public = _public_config()
    assert public["plan_mode"] == "standard"
    assert list(public["plan_modes"]) == list(PLAN_MODES)

    kept = _delivery_config({"delivery_mode": "stroke"})
    assert kept.stroke.plan_mode == "standard"

    asked = _delivery_config({"delivery_mode": "stroke", "plan_mode": "banded"})
    assert asked.stroke.plan_mode == "banded"
    assert asked.stroke.stroke_corner_cost == 1.0
    assert asked.robot.joint_span_fraction == 1.0

    # an explicit price still wins over the mode's, which is what the page's own control is
    mixed = _delivery_config({"delivery_mode": "stroke", "plan_mode": "banded",
                             "stroke_corner_cost": 1.4})
    assert mixed.stroke.stroke_corner_cost == 1.4


def test_the_page_refuses_a_mode_it_does_not_know():
    from scanweb import _delivery_config

    with pytest.raises(ValueError, match="规划模式|plan_mode"):
        _delivery_config({"delivery_mode": "stroke", "plan_mode": "wider"})


def test_a_saved_setup_does_not_store_the_sentinel():
    """`_clean_setup` must write JSON: the "keep the file" sentinel is an `object()`."""
    import json

    from scanweb import _clean_setup

    cleaned = _clean_setup({
        "point_count": 3,
        "cad_points_vehicle_mm": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        "measured_points_mm": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        "plan_mode": "banded",
    })
    assert cleaned["plan_mode"] == "banded"
    assert cleaned["stroke_corner_cost"] is None
    json.dumps(cleaned)          # must not raise

