"""The branch sequence prefers ONE arm configuration, and says so when it cannot.

Two solutions of the same tool pose can sit in different CONFIG families (front/back arm,
elbow up/down). Swapping families mid-program is not a joint-space hop: the controller swings
the whole arm round, and no clearance check in this library can see that swing - they only
model the tool. So the sequence search charges for it, on the measured 2026-10-28 program
where staying in one family was possible and it still changed twice (J1 167.6 deg, J3
158.8 deg).

These fixtures are hand-made solutions: the point is the CHOICE, not the inverse kinematics.
"""

from __future__ import annotations

import numpy as np

from scanplan import path as path_module
from scanplan.kinematics import Solution
from scanplan.path import PathPoint, assign_branches

NUT = "N U T, 0, 0, 0"
FDT = "F D T, 0, 0, 0"


def item(value: float, config: str) -> Solution:
    shoulder, elbow, wrist = config.split(",")[0].split()
    return Solution(joints_deg=(0.0, 0.0, 0.0, 0.0, 30.0, float(value)), shoulder=shoulder,
                    elbow=elbow, wrist_flip=wrist, config=config, j5_margin_deg=20.0,
                    min_margin_deg=6.0, ok=True)


class StubGate:
    """A gate that answers with whatever each point was given, in order."""

    active = True

    def __init__(self, options):
        self.options = options

    def solutions(self, position, axis, roll_deg):
        return self.options[int(position[0])]


def point(index: int) -> PathPoint:
    """Point `index` carries its own index in x, so the stub can answer for it."""
    return PathPoint(position=np.array([float(index), 0.0, 1000.0]),
                     axis=np.array([0.0, 0.0, -1.0]), roll_deg=0.0,
                     role="coverage", source="greedy", stroke_id=index)


def options():
    """Point 0 is only reachable in N U T; the other two are cheaper in F D T."""
    return [
        [item(0.0, NUT)],
        [item(40.0, NUT), item(10.0, FDT)],
        [item(80.0, NUT), item(20.0, FDT)],
    ]


def test_priced_on_travel_alone_the_cheapest_sequence_changes_family(monkeypatch):
    """The premise: 0 -> 10 -> 20 beats 0 -> 40 -> 80 by a mile, so travel alone switches."""
    monkeypatch.setattr(path_module, "_CONFIG_CHANGE_COST_DEG", 0.0)
    assignment = assign_branches([point(i) for i in range(3)], StubGate(options()), 40.0)
    assert assignment.histogram == {"N U T": 1, "F D T": 2}
    assert assignment.max_step_deg == 10.0


def test_a_configuration_change_is_paid_for_and_avoided():
    """With the charge in place the same points stay in the family they can all reach."""
    assignment = assign_branches([point(i) for i in range(3)], StubGate(options()), 40.0)
    assert assignment.histogram == {"N U T": 3}
    assert assignment.ok


def test_a_point_only_one_family_can_hold_still_gets_its_solution():
    """A preference, not a ban: the odd point out is delivered, not dropped."""
    only_odd = [
        [item(0.0, NUT)],
        [item(70.0, FDT)],
        [item(0.0, NUT)],
    ]
    assignment = assign_branches([point(i) for i in range(3)], StubGate(only_odd), 40.0)
    assert assignment.failed == []
    assert [solution.config.split(",")[0].strip() for solution in assignment.solutions] == [
        "N U T", "F D T", "N U T"]
    assert assignment.histogram == {"N U T": 2, "F D T": 1}
