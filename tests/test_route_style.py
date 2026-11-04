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
from dataclasses import replace
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
    _arrival,
    _cartesian_leg,
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


def test_the_tidy_preset_prices_the_wrist_twist_and_direct_leaves_it_alone():
    """`tidy` is the style that promises a straighter, calmer program, so it pays for the
    wrist turn; `direct` must stay the identity, and the twist price is a site preference in
    the same sense `robot.j6_weight` is."""
    assert ROUTE_PRESETS["tidy"]["stroke"]["order_j6_weight_mm_per_deg"] > 0.0
    assert "order_j6_weight_mm_per_deg" not in ROUTE_PRESETS["direct"]["stroke"]


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

    def body_clearance_mm(self, positions, *, limit_mm=None):
        """The same box stands in for the part: the stub has no separate mesh."""
        return self.clearance_mm(positions)


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

    def follow(self, joints_start, position, axis, roll_deg, *, step_mm=10.0,
               track_start=0.0, track_end=None):
        """The straight-line walk the connector now uses to earn `LEVEL_STRAIGHT`."""
        return SimpleNamespace(ok=True, reason="", joints_deg=self.joints,
                               track_mm=track_start if track_end is None else track_end)

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


# ------------------------------------------------------- straight means Cartesian, not joint


def _leg(cfg, station, kin, entry=None):
    previous, default_entry = _pair()
    return _cartesian_leg(previous, entry if entry is not None else default_entry,
                          _context(cfg, station, kin))


def test_a_clear_line_that_keeps_the_branch_is_a_straight_leg():
    cfg = route_preset(_config(), "tidy")
    ok, margin, reason = _leg(cfg, _StubStation(blocked=False), _StubKin())
    assert ok and reason == ""
    assert margin == 120.0 - 45.0


def test_a_blocked_line_is_not_a_straight_leg():
    cfg = route_preset(_config(), "tidy")
    ok, _margin, reason = _leg(cfg, _StubStation(blocked=True), _StubKin())
    assert not ok and reason == "straight_blocked"


def test_a_straight_leg_must_spread_its_rotation_over_the_distance():
    """An `L` move turns the head as it goes, so the turn has to fit the flight."""
    cfg = route_preset(_config(), "tidy")
    previous, entry = _pair()
    rolled = replace(entry, roll_deg=90.0)
    ok, _margin, reason = _leg(cfg, _StubStation(blocked=False), _StubKin(), entry=rolled)
    assert ok, "a 90 deg twist over 200 mm is inside the measured bound"
    tight = replace(cfg, stroke=replace(cfg.stroke, max_rot_deg_per_mm=0.01))
    ok, _margin, reason = _leg(tight, _StubStation(blocked=False), _StubKin(), entry=rolled)
    assert not ok and reason == "straight_rate"


def test_a_straight_leg_that_loses_the_branch_on_the_way_is_refused():
    """Clearance is not reachability: the controller solves every point along the line."""

    class _Lost(_StubKin):
        def follow(self, joints_start, position, axis, roll_deg, *, step_mm=10.0,
                   track_start=0.0, track_end=None):
            return SimpleNamespace(ok=False, reason="branch_lost", joints_deg=self.joints,
                                   track_mm=track_start)

    cfg = route_preset(_config(), "tidy")
    ok, _margin, reason = _leg(cfg, _StubStation(blocked=False), _Lost())
    assert not ok and reason == "straight_branch_lost"


def test_the_arrival_is_written_as_the_instruction_the_connector_proved():
    """Both halves matter: an `L` where the line was proved, a `J` where it was not."""
    corner = PathPoint(position=np.zeros(3), axis=np.array([0.0, 0.0, -1.0]), roll_deg=0.0,
                       role=ROLE_COVERAGE, source="greedy", motion="J",
                       representation="joint")
    straight = _arrival(corner, LEVEL_STRAIGHT)
    assert (straight.motion, straight.representation) == ("L", "cart")
    lifted = _arrival(corner, LEVEL_ONE_CORNER)
    assert (lifted.motion, lifted.representation) == ("J", "joint")
    joint = _arrival(corner, LEVEL_JOINT)
    assert joint.motion == "J"
    # A gate-off run has no motion at all, and must keep the program default it always had.
    blind = PathPoint(position=np.zeros(3), axis=np.array([0.0, 0.0, -1.0]), roll_deg=0.0,
                      role=ROLE_COVERAGE, source="greedy")
    assert _arrival(blind, LEVEL_STRAIGHT).motion is None


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


def test_assign_branches_refuses_a_family_change_even_when_it_is_cheaper():
    """Two branches for one pose, and the cheaper step sits in the other family.

    That is a refusal now, not a price: the move into the point may be proven as a straight
    Cartesian line, and a line cannot change family mid-flight (the controller answers
    `MOTN-063`, and `roboguide.validate` refuses to write a record that disagrees with it).
    """
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
    assert plain.solutions[1].config.startswith("N"), \
        "the cheaper step is the other posture, and it is not available"
    assert plain.fallback is False
    # The price changes nothing while one family can carry the chain, and it never changes
    # what the program is told to move.
    priced = assign_branches(points, gate, 400.0, config_change_cost_deg=240.0)
    assert priced.solutions[1].config.startswith("N")
    assert priced.fallback is False
    assert priced.max_step_deg == 40.0


def test_a_point_that_only_exists_in_another_family_keeps_it():
    """The refusal is a rule with a documented way out, not a dead end.

    When no single family carries the chain, the search relaxes to the configured price and
    says so (`fallback`), because dropping the point would drop the sweep behind it.
    """
    near = np.array([0.0, 0.0, 0.0])
    far = np.array([100.0, 0.0, 0.0])

    class _Gate:
        active = True

        def solutions(self, position, axis, roll_deg):
            key = tuple(np.round(np.asarray(position, dtype=float), 6))
            if key == tuple(np.round(near, 6)):
                return [_StubSolution((0.0, 0.0, 0.0, 0.0, -60.0, 0.0), "N U T, 0, 0, 0")]
            return [_StubSolution((5.0, 0.0, 0.0, 0.0, -60.0, 0.0), "F D T, 0, 0, 0")]

    points = [
        PathPoint(position=near, axis=np.array([0.0, 0.0, -1.0]), roll_deg=0.0,
                  role=ROLE_COVERAGE, source="greedy"),
        PathPoint(position=far, axis=np.array([0.0, 0.0, -1.0]), roll_deg=0.0,
                  role=ROLE_COVERAGE, source="greedy"),
    ]
    kept = assign_branches(points, _Gate(), 400.0, config_change_cost_deg=240.0)
    assert kept.solutions[1].config.startswith("F")
    assert kept.fallback is True
