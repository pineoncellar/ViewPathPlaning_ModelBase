"""Kinematics: inverse/forward round trip, branches, limits.

Every pose here is SYNTHETIC - random joint angles put through the forward model. The
measured station data used to fit the model is deliberately not in this file: it describes
the customer's cell.

The round trip is the test that matters. A sign error in one wrist axis produces joint
values that look entirely plausible and are wrong, and only putting them back through
`forward` makes that visible.
"""

from __future__ import annotations

import numpy as np
import pytest

from scanplan.config import RobotParams
from scanplan.kinematics import Robot, _interpolate_rotation, rotation_angle_deg

SPREAD = np.array([[-120.0, 120.0], [-50.0, 100.0], [-80.0, 80.0],
                   [-180.0, 180.0], [-100.0, 100.0], [-180.0, 180.0]])


@pytest.fixture(scope="module")
def robot() -> Robot:
    params = RobotParams.disabled()
    return Robot(RobotParams(**{**params.__dict__, "enabled": True}))


@pytest.fixture(scope="module")
def raw_robot() -> Robot:
    """The same arm with the site's limit policy AND the wrist-branch policy removed.

    Branch-structure and round-trip tests use this one: they are about the SOLVER (does it
    find all four arm branches, does it invert itself), and the site policy deliberately
    rejects some of those branches - the `F` shoulder lands at J3 near -175, below the
    controller's -120.71 floor, and the `F` WRIST branch cannot be written into a CONFIG
    string at all. Testing the solver through those policies would make a correct solver
    look broken.
    """
    params = RobotParams.disabled()
    return Robot(RobotParams(**{**params.__dict__, "enabled": True,
                                "soft_limits_deg": params.mechanical_limits_deg,
                                "joint_span_fraction": 1.0,
                                "wrist_flip_policy": "any"}))


@pytest.fixture(scope="module")
def any_wrist() -> Robot:
    """The SITE limits with only the wrist-branch policy removed.

    The segment-walk tests need both: the site limits, because the whole point of several of
    them is the J2 floor the pendant imposes, and both wrist branches, because they exercise
    moves that deliberately cross the wrist singularity or try to rescue a segment by taking
    the other sign.
    """
    params = RobotParams.disabled()
    return Robot(RobotParams(**{**params.__dict__, "enabled": True,
                                "wrist_flip_policy": "any"}))


def random_joints(rng, count: int) -> np.ndarray:
    low, high = SPREAD[:, 0], SPREAD[:, 1]
    return rng.uniform(low, high, size=(count, 6))


def joint_gap(first, second) -> float:
    """Largest per-axis difference, taking the 360 degree wrap into account."""
    return max(abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)
               for a, b in zip(first, second, strict=True))


def test_forward_reproduces_a_pose_its_own_inverse_computed(raw_robot):
    rng = np.random.default_rng(11)
    for joints in random_joints(rng, 40):
        position, rotation = raw_robot.forward(joints)
        solutions = raw_robot.solutions(position, rotation)
        assert solutions, f"no solution for joints {joints}"
        assert min(joint_gap(item.joints_deg, joints) for item in solutions) < 1e-3


def test_wrist_centre_matches_the_tool_offset(robot):
    """The flange origin is exactly |tool| from the TCP, in every orientation."""
    rng = np.random.default_rng(12)
    for joints in random_joints(rng, 20):
        position, _rotation = robot.forward(joints)
        wrist = robot.wrist_centre(joints)
        assert np.linalg.norm(position - wrist) == pytest.approx(
            float(np.linalg.norm(robot.tool_vector)), abs=1e-9)


def test_a_generic_pose_has_four_arm_branches(raw_robot):
    """Two shoulders times two elbows, each with two wrist signs: up to eight solutions."""
    rng = np.random.default_rng(13)
    position, rotation = raw_robot.forward(random_joints(rng, 1)[0])
    solutions = raw_robot.solutions(position, rotation)
    shoulders = {(item.shoulder, item.elbow) for item in solutions}
    assert len(solutions) >= 4
    assert len(shoulders) >= 2


def test_the_site_limits_drop_the_front_shoulder_branch(robot):
    """Most `F` poses sit near J3 = -175, and the controller's floor is -120.71.

    So the drawn-out branch simply is not offered under the site policy - which is the
    reason every `F D` point in the delivered programs was unreachable.
    """
    rng = np.random.default_rng(13)
    pose = robot.forward(random_joints(rng, 1)[0])
    offered = robot.solutions(*pose)
    assert offered
    assert all(item.shoulder == "N" for item in offered)
    assert all(item.joints_deg[2] > robot.limits[2][0] for item in offered)


def test_wrist_flip_changes_only_the_sign_of_j5(raw_robot):
    """Why changing the T letter in a CONFIG string never fixes an over-travel."""
    rng = np.random.default_rng(14)
    position, rotation = raw_robot.forward(random_joints(rng, 1)[0])
    solutions = raw_robot.solutions(position, rotation)
    grouped: dict[tuple[str, str], list[float]] = {}
    for item in solutions:
        grouped.setdefault((item.shoulder, item.elbow), []).append(item.joints_deg[4])
    for values in grouped.values():
        if len(values) == 2:
            assert abs(abs(values[0]) - abs(values[1])) < 1e-6


def test_solutions_are_sorted_by_j5_room(robot):
    rng = np.random.default_rng(15)
    position, rotation = robot.forward(random_joints(rng, 1)[0])
    margins = [item.j5_margin_deg for item in robot.solutions(position, rotation)]
    assert margins == sorted(margins, reverse=True)


def test_config_string_reads_the_joint_values(robot):
    """The letters come from the solution, not from a constant.

    The T/F letter is `T` for NEGATIVE J5: on the 2026-10-23 bank all eight poses sat at
    J5 between -9 and -124 and the pendant called every one of them `T`.
    """
    positive = robot.config_string((0, 0, 0, 0, 40, 0), "F", "D")
    negative = robot.config_string((0, 0, 0, 0, -40, 0), "F", "D")
    assert positive.startswith("F D F")
    assert negative.startswith("F D T")
    assert positive.endswith(", 0, 0, 0")


def test_turn_counts_are_reported_for_the_wrist_axes(robot):
    """The digit counts revolutions OUTSIDE the [-180, 180) base reading.

    `-63.699` is turn 0 - the pendant's own display - so `floor(J / 360)`, which said -1,
    was one too low for every negative reading in the measured bank.
    """
    assert robot.config_string((0, 0, 0, 370, 30, -370), "N", "U").endswith(", 0, 1, -1")
    assert robot.config_string((-63.699, 0, 0, 0, -40, 0), "N", "U").endswith(", 0, 0, 0")


def test_wraps_accepts_the_equivalent_reading(robot):
    """370 degrees and 10 degrees are the same physical J6, so both must be offered."""
    assert robot.wraps(5, 370.0) == pytest.approx([-350.0, 10.0])
    # J4 spans a full turn each way, so 370 is its own reading as well as 10.
    assert 10.0 in robot.wraps(3, 370.0)


def test_margins_prefer_the_roomiest_equivalent_reading(robot):
    """J4 spans +-360, so a reading of 200 has a roomier equivalent at -160."""
    roomy = robot.margins((0, 0, 0, 200.0, 30.0, 0))[3]
    assert roomy == pytest.approx(164.0)


def test_a_value_outside_every_reading_has_no_margin(robot):
    """J2 has no full turn, so 200 degrees is simply out of range."""
    assert robot.margins((0, 200.0, 0, 0, 30.0, 0))[1] == -np.inf


def test_j5_margin_prefers_a_mid_band_angle_over_zero(robot):
    """J5 = 0 is the wrist singularity, so the middle of the range is worst."""
    assert robot.j5_margin((0, 0, 0, 0, 0.0, 0)) < 0.0
    assert robot.j5_margin((0, 0, 0, 0, 60.0, 0)) > robot.j5_margin((0, 0, 0, 0, 100.0, 0))
    assert robot.j5_margin((0, 0, 0, 0, 100.0, 0)) > 0.0


def test_j5_is_rejected_three_degrees_short_of_the_documented_limit(robot):
    """J5 = -122 alarmed in ROBOGUIDE although the limit is -125.

    This is the reason `joint_span_fraction` exists: a pose three degrees from a limit is
    legal on paper and not one the machine holds. With the span at 0.9 the usable band ends
    at +-112.5, so -122 must fail even though the controller would accept the number.
    """
    assert robot.limits[4] == (-125.0, 125.0)
    assert robot.usable[4] == pytest.approx((-112.5, 112.5))
    assert robot.margins((0, 0, 0, 0, -122.0, 0))[4] < 0.0
    assert not robot.feasible((0, 0, 0, 0, -122.0, 0))
    assert robot.feasible((0, 0, 0, 0, -100.0, 0))


def test_feasible_rejects_a_point_sitting_on_the_usable_edge(robot):
    """J2's usable floor is -53.2; the controller's is -60, and neither is free."""
    assert robot.margins((0.0, -60.0, 0.0, 0.0, 64.0, 0.0))[1] < 0.0
    on_edge = (0.0, -53.2, 0.0, 0.0, 64.0, 0.0)
    assert robot.margins(on_edge)[1] == pytest.approx(0.0, abs=1e-6)
    assert not robot.feasible(on_edge)


def test_j3_floor_comes_from_the_pendant_readout(robot):
    """J3 is [-120.71, 180] on the controller, and -175 is what made every F pose fail.

    The old configuration said -185 was legal, so the planner emitted F-shoulder solutions
    with J3 near -175; the controller re-expressed those as +185 and alarmed. The floor is
    what has to be right.
    """
    assert robot.limits[2] == pytest.approx((-120.71, 180.0))
    assert robot.margins((0.0, 0.0, -175.0, 0.0, 64.0, 0.0))[2] < 0.0
    assert not robot.feasible((0.0, 0.0, -175.0, 0.0, 64.0, 0.0))
    assert robot.feasible((0.0, 0.0, -90.0, 0.0, 64.0, 0.0))


def test_usable_band_is_a_fraction_of_the_span_about_its_centre(robot):
    """J2 is [-60, 76]: centre 8, half-span 68, so 90 percent leaves [-53.2, 69.2]."""
    low, high = robot.usable[1]
    assert (low, high) == pytest.approx((-53.2, 69.2))
    # Not 90 percent of each number, which would have given [-54, 68.4]: the spans are not
    # symmetric and the fraction is about the room that exists on each side.
    assert high > 68.4


def test_span_fraction_of_one_reproduces_the_bare_limits(robot):
    params = RobotParams(**{**RobotParams.disabled().__dict__, "enabled": True,
                            "joint_span_fraction": 1.0})
    bare = Robot(params)
    for (bare_low, bare_high), (low, high) in zip(bare.usable, bare.limits, strict=True):
        assert (bare_low, bare_high) == pytest.approx((low, high))
    assert bare.limits == robot.limits


def test_feasible_accepts_a_comfortable_pose(robot):
    assert robot.feasible((0.0, 0.0, 0.0, 0.0, 64.0, 0.0))


def test_unreachable_point_has_no_solutions(robot):
    assert robot.solutions(np.array([9000.0, 0.0, 0.0]), np.eye(3)) == []


def test_limits_source_says_which_envelope_is_in_force(robot):
    """The report has to be able to say "soft" - pretending the datasheet is in force is
    exactly how a simulation passes and the machine alarms."""
    assert robot.limits_source.startswith("soft")
    params = RobotParams.disabled()
    same = RobotParams(**{**params.__dict__, "enabled": True, "soft_limits_deg":
                          params.mechanical_limits_deg})
    assert Robot(same).limits_source == "mechanical"


def test_rotation_angle_is_zero_for_identical_frames(robot):
    rotation = np.eye(3)
    assert rotation_angle_deg(rotation, rotation) == pytest.approx(0.0)


# -- segment walks: the check the gate used to be missing --------------------


def test_branch_of_round_trips_a_joint_set(any_wrist):
    """The letters have to come back from a joint set that is already being held."""
    joints = (10.0, -20.0, -10.0, 30.0, 60.0, 20.0)
    branch = any_wrist.branch_of(joints)
    assert branch is not None
    position, rotation = any_wrist.forward(joints)
    offered = {(item.shoulder, item.elbow, item.wrist_flip)
               for item in any_wrist.solutions(position, rotation)}
    assert branch in offered


def test_follow_accepts_a_zero_length_move_and_keeps_the_branch(any_wrist):
    joints = (10.0, -20.0, -10.0, 30.0, 60.0, 20.0)
    position, rotation = any_wrist.forward(joints)
    verdict = any_wrist.follow(joints, position, rotation)
    assert verdict.ok
    assert verdict.reason == "ok"
    assert any_wrist.branch_of(verdict.joints_deg) == any_wrist.branch_of(joints)


def test_follow_names_the_axis_and_where_the_segment_dies(any_wrist):
    """J2 is driven at the usable floor while both endpoints stay usable.

    This is the shape of the real failure: `X10290`'s last segment has a solution at each
    end and still dies partway along, so a per-point check passes it. The axis is reported
    0-based, the same index `margins` uses, so J2 is axis 1 and not axis 2.
    """
    start = (10.0, -46.0, -10.0, 30.0, 60.0, 20.0)
    finish = (10.0, -52.0, -10.0, 30.0, 60.0, 20.0)
    verdict = any_wrist.follow(start, *any_wrist.forward(finish), step_mm=2.0)
    assert not verdict.ok
    assert verdict.reason == "joint_limit"
    assert verdict.limiting_axis == 1
    assert 0.0 < verdict.failure_at < 1.0
    assert verdict.min_margin_deg < any_wrist.params.joint_margin_deg


def test_follow_refuses_to_cross_the_wrist_singularity(any_wrist):
    """Taking J5 from +60 to -60 walks straight through J5 = 0, where J4 and J6 collide.

    Both endpoints are comfortable and the target pose is perfectly solvable; only the
    middle of the segment is not. A verifier that looked at the endpoints alone would call
    this segment legal.
    """
    start = (10.0, -20.0, -10.0, 30.0, 60.0, 20.0)
    finish = (10.0, -20.0, -10.0, 30.0, -60.0, 20.0)
    target = any_wrist.forward(finish)
    assert any_wrist.solutions(*target), "the target itself must be solvable for this to bite"
    verdict = any_wrist.follow(start, *target, step_mm=2.0)
    assert not verdict.ok
    assert verdict.limiting_axis == 4
    assert abs(verdict.joints_deg[4]) < any_wrist.params.j5_min_abs_deg


def test_follow_does_not_switch_branch_to_rescue_a_segment(any_wrist):
    """Every sample is solved on the branch the arm started on, never on the best one.

    The controller holds the configuration it started with and raises `MOTN-063` when the
    declared and actual shapes disagree, so a verifier that re-picks the roomiest solution
    at every sample would predict a motion no controller performs.
    """
    start = (10.0, -20.0, -10.0, 30.0, 60.0, 20.0)
    finish = (10.0, -20.0, -10.0, 30.0, -60.0, 20.0)
    position, rotation = any_wrist.forward(finish)
    verdict = any_wrist.follow(start, position, rotation, step_mm=2.0)
    assert not verdict.ok
    # The other wrist sign is available at every one of those samples; taking it is exactly
    # the shortcut this refuses.
    assert any(item.wrist_flip != any_wrist.branch_of(start)[2]
               for item in any_wrist.solutions(position, rotation))


def test_follow_verdict_survives_a_change_of_step_size(robot):
    """A gross step and a fine one must agree, or the number means nothing."""
    start = (10.0, -20.0, -10.0, 30.0, 60.0, 20.0)
    finish = (10.0, -20.0, -10.0, 30.0, -60.0, 20.0)
    target = robot.forward(finish)
    fine = robot.follow(start, *target, step_mm=2.0)
    coarse = robot.follow(start, *target, step_mm=10.0)
    assert (fine.ok, fine.reason, fine.limiting_axis) == (
        coarse.ok, coarse.reason, coarse.limiting_axis)
    assert abs(fine.failure_at - coarse.failure_at) < 0.05


def test_follow_rejects_a_non_positive_step(robot):
    joints = (10.0, -20.0, -10.0, 30.0, 60.0, 20.0)
    position, rotation = robot.forward(joints)
    with pytest.raises(ValueError, match="step_mm"):
        robot.follow(joints, position, rotation, step_mm=0.0)


def test_interpolated_rotation_stays_a_rotation():
    """Half a turn is the case a naive matrix average gets wrong."""
    first = np.eye(3)
    second = np.diag([1.0, -1.0, -1.0])
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        frame = _interpolate_rotation(first, second, fraction)
        assert frame.T @ frame == pytest.approx(np.eye(3), abs=1e-9)
        assert float(np.linalg.det(frame)) == pytest.approx(1.0)
        assert rotation_angle_deg(first, frame) == pytest.approx(180.0 * fraction, abs=1e-6)


def test_the_joint_sweep_starts_after_the_first_point_and_ends_on_the_last(raw_robot):
    """The clearance check samples the MOTION, so the start pose is not one of its samples.

    Including the start would make the previous corner's clearance count twice and would let a
    zero-length move report a sample it never took; leaving the end out would drop the pose the
    arm actually has to hold.
    """
    start = (10.0, -20.0, -10.0, 30.0, 60.0, 20.0)
    end = (10.0, -20.0, -10.0, 40.0, 60.0, 20.0)
    points = raw_robot.sweep_points(start, end, step_deg=5.0)
    assert len(points) == 2                       # ceil(10 / 5) = 2
    target, _ = raw_robot.forward(end)
    assert points[-1] == pytest.approx(target, abs=1e-9)
    first, _ = raw_robot.forward(start)
    assert np.linalg.norm(points[0] - first) > 1e-6


def test_the_joint_sweep_length_follows_the_worst_axis(raw_robot):
    """One sample per `step_deg` of the largest axis change, and never zero samples."""
    start = (0.0, 0.0, 0.0, 0.0, 30.0, 0.0)
    for span, expected in ((1.0, 1), (9.9, 2), (10.0, 2), (40.0, 8)):
        end = (0.0, 0.0, 0.0, 0.0, 30.0 + span, 0.0)
        assert len(raw_robot.sweep_points(start, end, step_deg=5.0)) == expected
    with pytest.raises(ValueError, match="step_deg"):
        raw_robot.sweep_points(start, start, step_deg=0.0)


def test_the_site_policy_offers_only_the_pendant_t_branch(robot):
    """Only the negative-J5 branch is deliverable, and that is a measurement.

    Two 2026-10-29 experiments on the controller: a plan built on the F branch stopped at its
    first `L` with `MOTN-063 position configuration mismatch` (an L move cannot change the
    wrist branch, so it refused), and the same program with `F` written as a per-point CONFIG
    was refused whole at load with `ASBN-023 Configuration error`. The `F` letter is not
    loadable and the F branch is not flyable, so the planner must not choose it.
    """
    joints = (10.0, -20.0, -10.0, 30.0, -60.0, 20.0)
    found = robot.solutions(*robot.forward(joints))
    assert found, "the fixture pose must be usable for this to mean anything"
    assert all(item.wrist_flip == "T" for item in found)
    assert all(item.joints_deg[4] < 0.0 for item in found)


def test_opening_the_wrist_policy_reveals_the_other_branch(any_wrist):
    """The two branches come as a pair, which is why restricting to T costs little coverage."""
    joints = (10.0, -20.0, -10.0, 30.0, -60.0, 20.0)
    found = any_wrist.solutions(*any_wrist.forward(joints))
    signs = {item.wrist_flip: item.joints_deg[4] for item in found}
    assert set(signs) == {"T", "F"}
    assert signs["T"] < 0.0 < signs["F"]
    assert abs(abs(signs["T"]) - abs(signs["F"])) < 1e-6


