"""Slot A (rolls) and slot B (connector) contracts, and the stage-2 filter ordering.

The tests here are about the SEAM the merge introduced: a roll is decided by a policy
rather than by a loop in the generator, and the joint gate now runs before the pool's cap
rather than during the evaluation. Both changes are invisible in a green plan, which is
exactly why they need tests that would fail if the ordering slipped back.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from scanplan.candidates import cap, generate, ladder
from scanplan.config import load_config
from scanplan.policies import DiscreteRolls, LockedRolls, connector_policy, roll_policy
from scanplan.rolls import roll_for_axis
from scanplan.scanner_model import ScannerPose


@pytest.fixture(scope="session")
def params_placeholder():
    return load_config().scanner


@pytest.fixture(scope="session")
def params_locked(params_placeholder):
    return replace(params_placeholder, roll_mode="locked")


@pytest.fixture(scope="session")
def params_discrete(params_placeholder):
    return replace(params_placeholder, roll_mode="discrete")


def test_the_shipped_default_is_the_locked_roll(params_placeholder):
    """The default has to be the convention the field run used, not the bigger search."""
    assert params_placeholder.roll_mode == "locked"
    assert roll_policy(None).name == "locked"
    assert roll_policy("discrete").name == "discrete"
    assert type(roll_policy("locked")) is LockedRolls


def test_an_unknown_roll_mode_is_named_in_the_error():
    with pytest.raises(ValueError, match="roll_mode"):
        roll_policy("sometimes")


def test_locked_rolls_emit_one_roll_per_axis(params_locked):
    """Two rows of one axis share a roll, which is what keeps J6 still between them.

    One roll per AXIS, not one per ladder: the three tilts point down different axes, so
    each gets its own roll - that is the whole point of deriving it from the axis.
    """
    point = np.array([10.0, 10.0, 0.7])
    normal = np.array([0.0, 0.0, 1.0])
    poses = ladder(point, normal, params_locked, policy=LockedRolls())
    by_axis: dict[tuple, set[float]] = {}
    for pose in poses:
        key = tuple(np.round(pose.axis, 6))
        by_axis.setdefault(key, set()).add(round(pose.roll_deg, 6))
    assert len(by_axis) == len(params_locked.tilts_deg)
    assert all(len(rolls) == 1 for rolls in by_axis.values())
    head_on = [pose for pose in poses if np.allclose(pose.axis, [0.0, 0.0, -1.0])]
    assert head_on
    assert head_on[0].roll_deg == pytest.approx(roll_for_axis(np.array([0.0, 0.0, -1.0])))


def test_discrete_rolls_still_emit_every_configured_roll(params_discrete):
    point = np.array([10.0, 10.0, 0.7])
    normal = np.array([0.0, 0.0, 1.0])
    poses = ladder(point, normal, params_discrete, policy=DiscreteRolls())
    assert {pose.roll_deg for pose in poses} == set(params_discrete.rolls_deg)


def test_a_refused_pose_is_bought_back_at_another_roll(params_locked):
    """The retry is the price of locking: a refused roll must not lose the sight line."""
    point = np.array([[10.0, 10.0, 0.7]])
    normal = np.array([[0.0, 0.0, 1.0]])
    preferred = roll_for_axis(np.array([0.0, 0.0, -1.0]))
    wanted = (preferred + 45.0) % 360.0

    def keep(pose: ScannerPose) -> bool:
        difference = (pose.roll_deg - wanted + 180.0) % 360.0 - 180.0
        return abs(difference) < 1.0

    with_retry = generate(point, normal, params_locked, seed=0, keep=keep,
                          policy=LockedRolls())
    without = generate(point, normal, params_locked, seed=0, keep=keep,
                       policy=DiscreteRolls())
    assert without == []          # the discrete rolls are 0 and 90, neither is 45
    assert with_retry
    for pose in with_retry:
        difference = (pose.roll_deg - wanted + 180.0) % 360.0 - 180.0
        assert abs(difference) < 1.0


def test_the_gate_filters_the_pool_before_the_cap(params_locked):
    """A refused pose must not spend cap budget: that is the stage-2 ordering contract."""
    point = np.array([[x, 10.0, 0.7] for x in (10.0, 60.0, 110.0)])
    normal = np.tile([0.0, 0.0, 1.0], (3, 1))

    def keep_everything(pose: ScannerPose) -> bool:
        return True

    def keep_nothing(pose: ScannerPose) -> bool:
        return False

    full = generate(point, normal, params_locked, seed=0, policy=LockedRolls())
    empty = generate(point, normal, params_locked, seed=0, keep=keep_nothing,
                     policy=LockedRolls())
    assert empty == []
    assert full
    # With the gate passing everything the pool is unchanged, so the filter is not
    # silently dropping poses of its own.
    passed = generate(point, normal, params_locked, seed=0, keep=keep_everything,
                      policy=LockedRolls())
    assert len(passed) == len(full)


def test_the_cap_keeps_every_aim_point_represented(params_locked):
    """Round-robin, not a random cut: a region with no candidate can never be covered."""
    groups = [[ScannerPose(np.array([float(i), float(j), 0.0]), np.array([0.0, 0.0, -1.0]),
                           float(90 * j)) for j in range(5)] for i in range(4)]
    picked = cap(groups, 8)
    assert len(picked) == 8
    aims = {round(float(pose.position[0])) for pose in picked}
    assert aims == {0, 1, 2, 3}


def test_the_cap_is_deterministic(params_locked):
    point = np.array([[x, y, 0.0] for x in range(0, 200, 25) for y in range(0, 100, 25)],
                     dtype=float)
    normal = np.tile([0.0, 0.0, 1.0], (len(point), 1))
    first = generate(point, normal, params_locked, seed=3, max_candidates=40,
                     policy=LockedRolls())
    second = generate(point, normal, params_locked, seed=3, max_candidates=40,
                      policy=LockedRolls())
    assert [tuple(pose.position) for pose in first] == [tuple(pose.position) for pose in second]


def test_the_shipped_connector_is_the_two_level_chain():
    policy = connector_policy(None)
    assert policy.name == "direct-then-lift"
    with pytest.raises(ValueError, match="connector"):
        connector_policy("three-level")
