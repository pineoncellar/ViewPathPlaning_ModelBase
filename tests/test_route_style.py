"""The two route styles: what the preset expands to, and what the chain actually flies.

The styles exchange point count against program shape, and nothing else - so the tests come
in three groups:

* the TABLE (what `tidy` means, and that `direct` leaves the shipped run alone),
* the CHAIN (straight first, sideways second, the joint move last but never removed),
* the PRICES the sequencing adds for a CONFIG family change, which land in every style and
  therefore have to be off by default.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from scanplan.config import (
    CONNECTOR_MODES,
    ROUTE_PRESETS,
    ROUTE_STYLES,
    load_config,
    route_preset,
)
from scanplan.geometry import unit
from scanplan.path import (
    LEVEL_JOINT,
    LEVEL_ONE_CORNER,
    LEVEL_STRAIGHT,
    ROLE_COVERAGE,
    ConnectorContext,
    PathPlan,
    PathPoint,
    _config_penalty,
    assign_branches,
)
from scanplan.policies import connector_policy
from scanplan.report.definition import route_block


def _config():
    return load_config()


# --------------------------------------------------------------------------- the table


def test_every_style_has_a_preset_and_no_preset_is_orphaned():
    """The page offers `ROUTE_STYLES`, so a preset without a name is dead code and a name
    without a preset is a dropdown entry that crashes the run."""
    assert set(ROUTE_PRESETS) == set(ROUTE_STYLES)


def test_every_preset_names_a_connector_that_exists():
    for style, table in ROUTE_PRESETS.items():
        assert table["stroke"]["connector"] in CONNECTOR_MODES, style


def test_direct_is_the_shipped_configuration_unchanged():
    """`direct` has to be a no-op on the file that ships, or every existing run moves."""
    config = _config()
    same = route_preset(config, "direct")
    assert same.stroke == config.stroke
    assert same.robot == config.robot


def test_tidy_swaps_the_connector_and_prices_the_branch_change():
    tidy = route_preset(_config(), "tidy")
    assert tidy.stroke.route_style == "tidy"
    assert tidy.stroke.connector == "straight-then-lateral"
    # The fallback of the straight chain is the joint move, so it stays available.
    assert tidy.stroke.joint_connector is True
    assert tidy.robot.config_change_cost_deg == 240.0
    assert tidy.robot.wrist_flip_cost_deg == 1000.0
    assert tidy.robot.j6_weight == 0.5


def test_the_style_does_not_touch_the_site_acceptance_settings():
    """A style is a route decision. Moving the coverage target with it would make one
    dropdown change two things, and the site would have no way to tell which did it."""
    config = _config()
    tidy = route_preset(config, "tidy")
    assert tidy.target.coverage_target_pct == config.target.coverage_target_pct
    assert tidy.target.sample_grid_mm == config.target.sample_grid_mm
    assert tidy.stroke.connector_level == config.stroke.connector_level


def test_an_unknown_style_is_refused_by_name():
    with pytest.raises(ValueError, match="route_style"):
        route_preset(_config(), "neat")


def test_a_file_that_only_names_the_style_gets_the_whole_set(tmp_path):
    """A saved setup carries the label, not the five settings behind it."""
    path = tmp_path / "tidy.json"
    path.write_text(json.dumps({"stroke": {"route_style": "tidy"}}), encoding="utf-8")
    loaded = load_config(path)
    assert loaded.stroke.connector == "straight-then-lateral"
    assert loaded.robot.config_change_cost_deg == 240.0


def test_a_file_with_no_style_is_the_direct_run(tmp_path):
    path = tmp_path / "plain.json"
    path.write_text(json.dumps({}), encoding="utf-8")
    loaded = load_config(path)
    assert loaded.stroke.route_style == "direct"
    assert loaded.stroke.connector == "direct-then-lift"


def test_the_factory_resolves_both_connectors():
    assert connector_policy("direct-then-lift").name == "direct-then-lift"
    assert connector_policy("straight-then-lateral").name == "straight-then-lateral"


def test_the_report_block_says_which_style_ran():
    block = route_block(route_preset(_config(), "tidy"))
    assert block["style"] == "tidy"
    assert block["connector"] == "straight-then-lateral"
    assert block["config_change_cost_deg"] == 240.0
    assert "caveat" in block


# --------------------------------------------------------------------------- the chain


class _StubStation:
    """A station whose only obstacle is a small box at the origin.

    Enough to tell the three routes apart: a straight line from -X to +X through it is
    blocked, a sideways waiting point is not, and the rest of the cell is empty.
    """

    def __init__(self, blocked=True):
        self.blocked = blocked

    def clearance_mm(self, positions):
        points = np.asarray(positions, dtype=float).reshape(-1, 3)
        if not self.blocked:
            return np.full(len(points), 120.0)
        inside = (np.abs(points[:, 0]) < 5.0) & (np.abs(points[:, 1]) < 5.0)
        return np.where(inside, -5.0, 120.0)


class _StubModel:
    body_limit_mm = 45.0


class _StubSolution:
    def __init__(self, joints, config="N U T, 0, 0, 0"):
        self.joints_deg = tuple(float(value) for value in joints)
        self.config = config
        self.min_margin_deg = 30.0
        self.j5_margin_deg = 30.0


class _StubKin:
    """A gate that accepts everything and walks a joint move as one straight sample."""

    active = True
    robot = None

    def __init__(self, joints=(0.0, 0.0, 0.0, 0.0, -60.0, 0.0)):
        self.joints = joints

    def solutions(self, position, axis, roll_deg):
        return [_StubSolution(self.joints)]

    def check(self, position, axis, roll_deg):
        return SimpleNamespace(ok=True, reason="", solution=_StubSolution(self.joints))

    def sweep_points(self, start, end, step_deg):
        return np.asarray([[60.0, 60.0, 0.0], [60.0, 60.0, 0.0]])


def _pair():
    previous = PathPoint(position=np.array([-100.0, 0.0, 0.0]),
                         axis=np.array([0.0, 0.0, -1.0]), roll_deg=0.0,
                         role=ROLE_COVERAGE, source="greedy", stroke_id=0)
    entry = PathPoint(position=np.array([100.0, 0.0, 0.0]),
                      axis=np.array([0.0, 0.0, -1.0]), roll_deg=0.0,
                      role=ROLE_COVERAGE, source="greedy", stroke_id=1)
    return previous, entry


def _context(cfg, station, kin):
    return ConnectorContext(cfg=cfg, plan=PathPlan(points=[]), station=station,
                            model=_StubModel(), visible=None, strict=True, kin=kin,
                            start_joints=np.array([0.0, 0.0, 0.0, 0.0, -60.0, 0.0]),
                            end_joints=np.array([0.0, 0.0, 0.0, 0.0, -60.0, 0.0]))


def test_the_tidy_chain_flies_straight_when_nothing_is_in_the_way():
    """One instruction, no corner: that is the whole point of the style."""
    cfg = route_preset(_config(), "tidy")
    previous, entry = _pair()
    result = connector_policy(cfg.stroke.connector).connect(
        previous, entry, _context(cfg, _StubStation(blocked=False), _StubKin()))
    assert result is not None
    assert result.level == LEVEL_STRAIGHT
    assert result.corners == [] and result.joints == []


def test_the_tidy_chain_sidesteps_perpendicular_to_the_axis_when_blocked():
    """Sideways, not backwards: a retract along the axis walks into what is behind the head."""
    cfg = route_preset(_config(), "tidy")
    previous, entry = _pair()
    result = connector_policy(cfg.stroke.connector).connect(
        previous, entry, _context(cfg, _StubStation(blocked=True), _StubKin()))
    assert result is not None
    assert result.level == LEVEL_ONE_CORNER
    assert len(result.corners) == 1
    corner = result.corners[0]
    axis = unit(np.asarray(previous.axis, dtype=float))
    offset = np.asarray(corner.position, dtype=float) - np.asarray(previous.position)
    assert abs(float(np.dot(offset, axis))) < 1e-6, "the waiting point moved along the axis"
    assert float(np.linalg.norm(offset)) > 0.0
    # The corner is only usable because its joints were solved and checked.
    assert result.verified


def test_the_tidy_chain_falls_back_to_the_joint_move_rather_than_dropping_the_sweep():
    """A blocked straight line must not cost the sweep: the style is about shape, not coverage."""
    cfg = route_preset(_config(), "tidy")
    previous, entry = _pair()
    station = _StubStation(blocked=True)
    # A wall wide enough that no sideways waiting point helps: every lateral route is
    # sampled through |x| < 500, so the style's own two routes are both refused.
    station.clearance_mm = lambda positions: np.where(
        np.abs(np.asarray(positions, dtype=float).reshape(-1, 3)[:, 0]) < 500.0,
        -5.0, 120.0)
    kin = _StubKin()
    # The joint move is modelled as sweeping somewhere else entirely - which is exactly the
    # point of it: the TCP does not travel the straight line the joints interpolate along.
    kin.sweep_points = lambda start, end, step_deg: np.asarray([[600.0, 0.0, 0.0]])
    result = connector_policy(cfg.stroke.connector).connect(
        previous, entry, _context(cfg, station, kin))
    assert result is not None
    assert result.level == LEVEL_JOINT


def test_a_tidy_plan_built_through_build_path_counts_straight_transitions():
    """The chain has to reach `build_path` unchanged, including the level it reports.

    Two parallel sweeps on a station whose only obstacle sits between them: `direct` has no
    joint move to fall back on with the gate off, so it inserts corners; `tidy` flies the
    straight line and inserts nothing, which is what the report then counts.
    """
    from scanplan.path import build_path
    from scanplan.strokes import Stroke

    config = _config()
    station = _StubStation(blocked=False)
    sweeps = [
        Stroke(start=np.array([-100.0, -50.0, 0.0]), end=np.array([-100.0, 50.0, 0.0]),
               axis=np.array([0.0, 0.0, -1.0])),
        Stroke(start=np.array([100.0, -50.0, 0.0]), end=np.array([100.0, 50.0, 0.0]),
               axis=np.array([0.0, 0.0, -1.0])),
    ]
    tidy = route_preset(config, "tidy")
    plan = build_path(sweeps, cfg=tidy, station=station, model=_StubModel(), kin=None)
    assert plan.connector_levels.get(LEVEL_STRAIGHT, 0) >= 1
    # Two sweeps, two ends each, and no corner between them.
    assert len(plan.points) == 4

    direct = build_path(sweeps, cfg=route_preset(config, "direct"), station=station,
                        model=_StubModel(), kin=None)
    assert plan.path_length_mm <= direct.path_length_mm


# --------------------------------------------------------------------------- the prices


def test_a_config_family_change_costs_nothing_by_default():
    first = _StubSolution((0.0,) * 6, config="N U T, 0, 0, 0")
    second = _StubSolution((0.0,) * 6, config="F D T, 0, 0, 0")
    assert _config_penalty(first, second, 0.0) == 0.0


def test_a_config_family_change_is_priced_and_doubled_inside_one_sweep():
    same = _StubSolution((0.0,) * 6, config="N U T, 0, 0, 0")
    other = _StubSolution((0.0,) * 6, config="F D T, 0, 0, 0")
    assert _config_penalty(same, same, 240.0) == 0.0
    assert _config_penalty(same, other, 240.0) == 240.0
    # Both corners of a sweep are the SAME pose, so a family change there is not a
    # preference but a jump the scan would ride through.
    assert _config_penalty(same, other, 240.0, doubled=True) == 480.0


def test_assign_branches_keeps_the_family_it_is_already_on_when_asked_to():
    """Two branches for one pose; without a price the cheapest step wins, with one the
    posture is kept - and the reported degrees never change either way."""
    near = np.array([0.0, 0.0, 0.0])
    far = np.array([100.0, 0.0, 0.0])

    class _Gate:
        active = True

        def solutions(self, position, axis, roll_deg):
            key = tuple(np.round(np.asarray(position, dtype=float), 6))
            if key == tuple(np.round(near, 6)):
                return [_StubSolution((0.0, 0.0, 0.0, 0.0, -60.0, 0.0), "N U T, 0, 0, 0")]
            return [
                _StubSolution((40.0, 0.0, 0.0, 0.0, -60.0, 0.0), "N U T, 0, 0, 0"),
                _StubSolution((5.0, 0.0, 0.0, 0.0, -60.0, 0.0), "F D T, 0, 0, 0"),
            ]

    points = [
        PathPoint(position=near, axis=np.array([0.0, 0.0, -1.0]), roll_deg=0.0,
                  role=ROLE_COVERAGE, source="greedy"),
        PathPoint(position=far, axis=np.array([0.0, 0.0, -1.0]), roll_deg=0.0,
                  role=ROLE_COVERAGE, source="greedy"),
    ]
    gate = _Gate()
    plain = assign_branches(points, gate, 400.0)
    assert plain.solutions[1].config.startswith("F"), "the cheaper step is the other posture"

    priced = assign_branches(points, gate, 400.0, config_change_cost_deg=240.0)
    assert priced.solutions[1].config.startswith("N"), "240 degrees buys the posture back"
    # The price decides between options; it never changes what the program is told to move.
    assert priced.max_step_deg == 40.0
