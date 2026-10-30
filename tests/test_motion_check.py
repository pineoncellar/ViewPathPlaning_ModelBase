"""Whole-move checks: the corners are not the whole story.

`assign_branches` judges the points; the controller flies the segments between them. A
reach that has to straighten, or a wrist that has to unwind, can leave the travel halfway
along while both ends sit comfortably inside it - so the move is sampled, the IK branch is
followed from where the arm already was, and a joint move (whose envelope is a convex box)
is settled by its corners alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from scanplan.config import RobotParams
from scanplan.kinematics import KinematicGate, MoveVerdict, pose_walk, slerp_axis
from scanplan.path import PathPoint, _route_allowed


def gate(**changes) -> KinematicGate:
    """The measured machine, gate on, in the workspace the fixtures sit in."""
    params = RobotParams.disabled()
    return KinematicGate(RobotParams(**{**params.__dict__, "enabled": True, **changes}))


class Pose:
    """The minimum a walk needs: a position, an optical axis and a roll."""

    def __init__(self, position, axis=(0.0, 0.0, -1.0), roll_deg: float = 0.0):
        self.position = np.asarray(position, dtype=float)
        self.axis = np.asarray(axis, dtype=float)
        self.roll_deg = float(roll_deg)


def test_a_move_is_sampled_between_its_corners_with_the_destination_included():
    first = Pose((2000.0, 0.0, 1000.0))
    second = Pose((2000.0, 200.0, 1000.0))
    samples = list(pose_walk(first, second, step_mm=50.0, step_deg=10.0))
    # the origin is already where the arm is; the destination is where it has to arrive
    assert pytest.approx(samples[-1][0]) == 1.0
    assert samples[0][0] < 1.0
    assert all(fraction > 0.0 for fraction, *_ in samples)
    # 200 mm at 50 mm per sample: four steps, five poses including the corner
    assert len(samples) == 4
    assert samples[-1][1] == pytest.approx(second.position)


def test_the_rotation_step_bounds_an_in_place_turn():
    """A re-orientation covers no millimetres, so a length-only rule would sample it once."""
    first = Pose((2000.0, 0.0, 1000.0), axis=(0.0, 0.0, -1.0))
    second = Pose((2000.0, 0.0, 1000.0), axis=(0.0, 0.0, -1.0), roll_deg=90.0)
    samples = list(pose_walk(first, second, step_mm=50.0, step_deg=10.0))
    assert len(samples) == 9                      # 90 deg at 10 deg per sample
    assert samples[-1][3] == pytest.approx(90.0)


def test_slerp_through_opposite_axes_has_a_well_defined_middle():
    """Antiparallel axes have no shortest arc; every perpendicular is equally valid."""
    middle = slerp_axis([0.0, 0.0, 1.0], [0.0, 0.0, -1.0], 0.5)
    assert float(np.linalg.norm(middle)) == pytest.approx(1.0)
    assert float(np.dot(middle, [0.0, 0.0, 1.0])) == pytest.approx(0.0, abs=1e-9)


def test_a_short_translation_is_a_legal_move_and_reports_the_branch_it_kept():
    arm = gate()
    first = Pose((2000.0, 0.0, 1000.0))
    second = Pose((2000.0, 60.0, 1000.0))
    verdict = arm.move(first, second, motion="L")
    assert verdict.ok and verdict.reason == "ok"
    assert verdict.joints_deg is not None
    assert arm.summary()["moves_checked"] == 1
    assert arm.summary()["move_samples"] >= 1


def test_a_joint_move_is_settled_by_its_corners():
    """The envelope is a box, so a straight joint line between two inside points is inside.

    Sampling it would only re-prove convexity, and the report must show that this is what
    happened rather than claiming a Cartesian check that never ran.
    """
    arm = gate()
    first = Pose((2000.0, 0.0, 1000.0))
    second = Pose((2000.0, 60.0, 1000.0))
    verdict = arm.move(first, second, motion="J")
    assert verdict.ok
    assert verdict.at_fraction is None
    assert arm.summary()["move_samples"] == 0


def test_a_move_from_an_unreachable_corner_is_not_judged():
    """A corner the arm cannot hold is reported as the corner, not as the move."""
    arm = gate()
    far = Pose((9000.0, 0.0, 1000.0))
    verdict = arm.move(far, Pose((2000.0, 0.0, 1000.0)), motion="L")
    assert not verdict.ok
    assert verdict.reason.startswith("start_")


def test_the_move_verdict_is_memoised():
    arm = gate()
    first = Pose((2000.0, 0.0, 1000.0))
    second = Pose((2000.0, 60.0, 1000.0))
    assert arm.move(first, second) is arm.move(first, second)


def test_route_legality_covers_the_segments_not_only_the_corners(cfg):
    """A gate that only objects to the MIDDLE of a move must still refuse the route."""

    class MiddleGate:
        active = True

        def check(self, position, axis, roll_deg):
            from types import SimpleNamespace
            return SimpleNamespace(ok=True, solution=None)

        def move(self, first, second, *, motion="L"):
            return MoveVerdict(False, "joint_limit", 0.5)

    def point(x: float) -> PathPoint:
        return PathPoint(position=np.array([x, 0.0, 1000.0]), axis=np.array([0.0, 0.0, -1.0]),
                         roll_deg=0.0, role="coverage", source="greedy", stroke_id=0)

    assert _route_allowed([point(100.0)], MiddleGate()) is True          # corners only
    assert _route_allowed([point(100.0)], MiddleGate(), cfg, point(0.0)) is False
