"""The two J6 charges, the roll fallback's guards, and the report that names the segments.

J6 is the axis that moves the scanner SIDEWAYS: the head sits 249 mm off the flange axis, so
one degree slides the optical origin 4.35 mm along an arc roughly parallel to the surface -
the one direction the 45 mm body sphere is blind in. A sweep is one constant pose written as
an `L` move, so J6 does not turn while the head scans; everything here is about the move
BETWEEN sweeps.

Three groups:

* the ARITHMETIC - `j6_weight` scales J6 inside the all-axis step and `j6_travel_weight`
  adds a charge on top, so they stop different things, and neither changes the degrees that
  get reported,
* the FALLBACK - a re-roll may hold J6 under the cap only when it does not cost the arm
  anything and the move is verified on both sides of the point it rewrites,
* the REPORT - `over_report` exists, is driven by `j6_report_deg`, and names `J` segments
  only.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scanplan.config import _validate_robot, load_config
from scanplan.path import (
    ROLE_COVERAGE,
    ROLE_TRAVERSE,
    PathPoint,
    _j5_crossover,
    _other_axes_deg,
    _step_cost,
    assign_branches,
    ease_over_cap_j6,
)
from scanplan.policies import ROLL_RETRY_DEG
from scanplan.report.definition import route_block
from scanplan.report.kinematics import _j6_block

REPO_ROOT = Path(__file__).resolve().parents[1]


class _Solution:
    """Just enough of a `Solution`: the joints, the posture, and `ok`."""

    ok = True

    def __init__(self, joints, config="N U T, 0, 0, 0"):
        self.joints_deg = tuple(float(value) for value in joints)
        self.config = config
        self.min_margin_deg = 30.0
        self.j5_margin_deg = 30.0


class _Gate:
    """A gate that answers a fixed list of solutions per position."""

    active = True
    robot = None

    def __init__(self, table):
        self.table = table

    def solutions(self, position, axis, roll_deg):
        key = tuple(np.round(np.asarray(position, dtype=float), 6))
        return list(self.table.get(key, []))


def _point(position, role=ROLE_COVERAGE):
    return PathPoint(position=np.asarray(position, dtype=float),
                     axis=np.array([0.0, 0.0, -1.0]), roll_deg=0.0, role=role,
                     source="greedy")


# --------------------------------------------------------------------------- the arithmetic


def test_scaling_cannot_see_a_small_twist_and_the_additive_charge_can():
    """The measurement behind the second charge.

    An arm move of 100 deg next to a 20 deg twist costs 100 whether J6 is scaled or not,
    because the all-axis step is a MAXIMUM and J6 is not the largest axis. The additive
    charge is the only one that sees the twist at all.
    """
    first = (0.0, 0.0, 0.0, 0.0, -60.0, 0.0)
    second = (100.0, 0.0, 0.0, 0.0, -60.0, 20.0)
    scaled = _step_cost(first, second, 3.0)
    carried = _step_cost(first, second, 1.0, None, 1.0)
    assert scaled == 100.0
    assert carried == pytest.approx(120.0)
    # And the plain run - both charges off - is the number the program reports.
    assert _step_cost(first, second) == 100.0


def test_the_travel_charge_is_off_by_default():
    first = (0.0, 0.0, 0.0, 0.0, -60.0, 0.0)
    second = (10.0, 0.0, 0.0, 0.0, -60.0, 170.0)
    assert _step_cost(first, second) == _step_cost(first, second, 1.0, None, 0.0)
    assert _step_cost(first, second) == 170.0


def test_travel_weight_prefers_the_branch_that_turns_j6_less():
    """Two branches at one pose, the same all-axis step, a very different turn about the
    flange - which is the one case the running charge can decide and nothing else does.

    The arm move is 150 deg and both twists stay under it, so the MAXIMUM step is 150 either
    way and the plain DP takes the first branch it is offered.
    """
    near = np.array([0.0, 0.0, 0.0])
    far = np.array([100.0, 0.0, 0.0])
    gate = _Gate({
        tuple(np.round(near, 6)): [_Solution((0.0, 0.0, 0.0, 0.0, -60.0, 0.0))],
        tuple(np.round(far, 6)): [
            _Solution((150.0, 0.0, 0.0, 0.0, -60.0, 140.0)),
            _Solution((150.0, 0.0, 0.0, 0.0, -60.0, 20.0)),
        ],
    })
    points = [_point(near), _point(far)]

    plain = assign_branches(points, gate, 400.0)
    assert plain.solutions[1].joints_deg[5] == 140.0, "no price, so the first branch wins"
    assert plain.max_step_deg == 150.0

    charged = assign_branches(points, gate, 400.0, j6_travel=1.0)
    assert charged.solutions[1].joints_deg[5] == 20.0, "the twist is now worth avoiding"
    # A price decides between options; it never changes what the program is told to move.
    assert charged.max_step_deg == plain.max_step_deg == 150.0


def test_a_negative_travel_weight_is_refused():
    config = load_config(REPO_ROOT / "configs" / "scanner_trackscan49.json")
    with pytest.raises(ValueError, match="j6_travel_weight"):
        _validate_robot(replace(config.robot, j6_travel_weight=-1.0))


# --------------------------------------------------------------------------- the fallback


def test_other_axes_deg_ignores_j6_and_j5_crossover_reads_the_sign():
    first = (0.0, 0.0, 0.0, 0.0, -60.0, 0.0)
    twisted = (0.0, 0.0, 0.0, 0.0, -60.0, 90.0)
    swung = (25.0, 0.0, 0.0, 0.0, -60.0, 0.0)
    assert _other_axes_deg(first, twisted) == 0.0
    assert _other_axes_deg(first, swung) == 25.0
    assert not _j5_crossover(first, twisted)
    assert _j5_crossover(first, (0.0, 0.0, 0.0, 0.0, 60.0, 0.0))


def _fallback_setup(*, candidates, after=None):
    """A two- or three-point path whose middle point is over the J6 cap."""
    near = np.array([0.0, 0.0, 0.0])
    middle = np.array([100.0, 0.0, 0.0])
    points = [_point(near), _point(middle, ROLE_TRAVERSE)]
    solutions = [_Solution((0.0, 0.0, 0.0, 0.0, -60.0, 0.0)),
                 _Solution((40.0, 0.0, 0.0, 0.0, -60.0, 200.0))]
    table = {tuple(np.round(middle, 6)): candidates}
    if after is not None:
        far = np.array([200.0, 0.0, 0.0])
        points.append(_point(far, ROLE_TRAVERSE))
        solutions.append(_Solution(after))
        table[tuple(np.round(far, 6))] = [_Solution(after)]
    return points, solutions, _Gate(table)


def test_a_reroll_that_only_lowers_j6_is_taken():
    points, solutions, gate = _fallback_setup(
        candidates=[_Solution((40.0, 0.0, 0.0, 0.0, -60.0, 30.0))])
    notes = ease_over_cap_j6(points, solutions, gate, 60.0)
    assert solutions[1].joints_deg[5] == 30.0
    assert any("roll re-tried" in note for note in notes)


def test_a_reroll_that_pays_with_the_arm_is_refused():
    """The failure this guard exists for: save 130 deg of wrist twist, spend 120 deg of J1."""
    points, solutions, gate = _fallback_setup(
        candidates=[_Solution((120.0, 0.0, 0.0, 0.0, -60.0, 30.0))])
    notes = ease_over_cap_j6(points, solutions, gate, 60.0)
    assert solutions[1].joints_deg[5] == 200.0, "the original corner is kept"
    assert not any("roll re-tried" in note for note in notes)
    assert any("still turn J6" in note for note in notes), "and the move is named instead"


def test_a_reroll_that_flips_j5_is_refused():
    points, solutions, gate = _fallback_setup(
        candidates=[_Solution((40.0, 0.0, 0.0, 0.0, 60.0, 30.0))])
    ease_over_cap_j6(points, solutions, gate, 60.0)
    assert solutions[1].joints_deg[4] == -60.0


def test_a_reroll_is_verified_on_both_sides_of_the_point_it_rewrites():
    """Arriving is not enough: the corner is also where the next move starts."""
    after = (60.0, 0.0, 0.0, 0.0, -60.0, 0.0)
    points, solutions, gate = _fallback_setup(
        candidates=[_Solution((40.0, 0.0, 0.0, 0.0, -60.0, 30.0))], after=after)
    seen = []

    def verify(first, second):
        seen.append((tuple(first), tuple(second)))
        return False

    ease_over_cap_j6(points, solutions, gate, 60.0, verify=verify)
    assert solutions[1].joints_deg[5] == 200.0, "an unverified move is not taken"
    assert seen, "the verifier was actually consulted"

    seen.clear()

    def arrival_only(first, second):
        seen.append(tuple(np.round(np.asarray(second, dtype=float), 6)))
        return True

    ease_over_cap_j6(points, solutions, gate, 60.0, verify=arrival_only)
    assert solutions[1].joints_deg[5] == 30.0
    assert seen[0] == (40.0, 0.0, 0.0, 0.0, -60.0, 30.0), "the arrival is checked first"
    assert seen[1] == (60.0, 0.0, 0.0, 0.0, -60.0, 0.0), "then the departure from the corner"
    assert len(seen) == 2 * len(ROLL_RETRY_DEG), "every candidate roll, on both sides"


def test_no_cap_means_no_pass_at_all():
    points, solutions, gate = _fallback_setup(
        candidates=[_Solution((40.0, 0.0, 0.0, 0.0, -60.0, 30.0))])
    assert ease_over_cap_j6(points, solutions, gate, None) == []
    assert solutions[1].joints_deg[5] == 200.0


# --------------------------------------------------------------------------- the report


def test_the_j6_block_names_the_segments_over_the_report_line():
    config = load_config()
    literal = np.zeros((3, 6))
    literal[0, 5] = 120.0     # segment 1, a J move: over the 90 deg report line
    literal[1, 5] = 150.0     # segment 2, flown as a line: not a motion the arm performs
    motions = ["J", "L", "J"]
    block = _j6_block(config, literal, literal.copy(), motions, [])
    assert block["report_deg"] == 90.0
    assert block["over_report"] == [1]
    # The cap and the report line are two different questions with the same answer shape.
    assert "over_cap" in block
    assert block["literal_total_deg"] == 120.0, "the L segment is not counted as a J move"


def test_the_j6_block_records_both_charges():
    config = load_config()
    literal = np.zeros((2, 6))
    literal[0, 5] = 10.0
    block = _j6_block(config, literal, literal.copy(), ["J"], [])
    assert block["weight"] == config.robot.j6_weight
    assert block["travel_weight"] == config.robot.j6_travel_weight


def test_the_route_block_records_the_j6_policy():
    config = load_config()
    block = route_block(config)
    for key in ("j6_weight", "j6_travel_weight", "j6_max_step_deg", "j6_report_deg"):
        assert key in block, key


def test_the_shipped_station_config_turns_the_cap_on():
    """The cap is the only knob that makes the fallback run at all, so its shipped value is
    a site decision and has to be visible here rather than implied by a default.

    It is also inert on this file, and that is recorded rather than hidden: module 10 only
    runs with `robot.enabled`, which ships false pending two unresolved site facts.
    """
    data = json.loads((REPO_ROOT / "configs" / "scanner_trackscan49.json")
                      .read_text(encoding="utf-8"))
    robot = data["robot"]
    assert robot["j6_max_step_deg"] == 60.0
    assert robot["j6_travel_weight"] == 0.0
    assert robot["j6_report_deg"] == 90.0
    assert robot["enabled"] is False, "the J6 prices are inert until the gate is on"
