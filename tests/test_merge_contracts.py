"""The contracts and the stage-5 refusal, which are the merge's load-bearing seams.

Each test here fails if the merge is undone in a way that is invisible in a green plan:
an envelope that two stages derive separately, a connector that hands on corners without
their joint states, an export that writes a program the controller would refuse.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from scanplan.config import load_config
from scanplan.contracts import ConnectorResult, RobotEnvelope, is_delivered_point
from scanplan.kinematics import KinematicGate
from scanplan.path import PathPoint
from scanplan.roboguide import adapt, validate
from scanplan.viewpoints import Viewpoint


def _gate(enabled: bool = True) -> KinematicGate:
    params = replace(load_config().robot, enabled=enabled)
    return KinematicGate(params)


def test_the_envelope_is_one_frozen_value_from_the_gate():
    gate = _gate()
    envelope = gate.envelope
    assert envelope is gate.envelope          # the same object, not re-derived per caller
    assert envelope.enabled and envelope.limits_source
    assert envelope.band(4) == tuple(float(value) for value in gate.robot.usable[4])
    with pytest.raises(Exception):
        envelope.margin_deg = 1.0             # frozen: a stage cannot widen it


def test_a_disabled_gate_still_yields_an_envelope():
    """Downstream never branches on None, which is what keeps the stages uniform."""
    envelope = _gate(enabled=False).envelope
    assert envelope.enabled is False
    assert envelope.limits_source == "disabled"
    assert len(envelope.limits) == 6


def test_the_connector_contract_says_when_it_is_verified():
    plain = ConnectorResult(level="J-direct", margin_mm=12.0)
    assert plain.direct and plain.verified
    routed = ConnectorResult(level="L1-retract", corners=[object()], joints=[],
                             margin_mm=3.0)
    assert not routed.direct
    assert not routed.verified                # corners without checked joints are not a route


def test_both_point_types_satisfy_the_delivered_point_contract():
    point = PathPoint(position=np.zeros(3), axis=np.array([0.0, 0.0, -1.0]), roll_deg=0.0,
                      role="coverage", source="greedy", motion="J", representation="joint")
    view = Viewpoint(id=1, position=np.zeros(3), axis=np.array([0.0, 0.0, -1.0]),
                     roll_deg=0.0, n_points=1, area_mm2=1.0, mean_incidence_deg=0.0,
                     max_incidence_deg=0.0, tracker_ok=True, source="greedy",
                     motion="J", representation="joint")
    assert is_delivered_point(point)
    assert is_delivered_point(view)
    assert not is_delivered_point(object())


def _points_with_joints():
    """A two-point program that claims joint evidence and cannot fly it."""
    return [
        {"id": 1, "position": (0.0, 0.0, 300.0), "axis": (0.0, 0.0, -1.0),
         "roll_deg": 0.0, "joints_deg": (0.0, 0.0, 0.0, 0.0, -60.0, 0.0), "motion": "J"},
        {"id": 2, "position": (120.0, 0.0, 300.0), "axis": (0.0, 0.0, -1.0),
         "roll_deg": 0.0, "joints_deg": (0.0, 0.0, 0.0, 0.0, -60.0, 0.0), "motion": "J"},
    ]


def test_the_export_refuses_a_program_with_joint_evidence_it_cannot_fly(tmp_path):
    """A file the controller rejects is worse than no file: it looks delivered."""
    with pytest.raises(ValueError, match="refusing to write"):
        adapt.export(_points_with_joints(), tmp_path, prog_name="BAD", motion="J")
    assert not list(tmp_path.glob("*.LS"))


def test_the_override_writes_the_file_and_marks_it(tmp_path):
    contract = adapt.export(_points_with_joints(), tmp_path, prog_name="BAD", motion="J",
                            allow_unverified=True)
    assert (tmp_path / "BAD.LS").exists()
    assert contract["joint_audit"]["refused"] is True
    assert contract["joint_audit"]["allowed_by_override"] is True


def test_a_pose_only_file_is_not_refused_for_reachability(tmp_path):
    """Reachability of a pose-only delivery is ROBOGUIDE's job, by documented design."""
    points = [{"id": 1, "position": (0.0, 0.0, 300.0), "axis": (0.0, 0.0, -1.0),
               "roll_deg": 0.0}]
    contract = adapt.export(points, tmp_path, prog_name="POSE", motion="L")
    assert (tmp_path / "POSE.LS").exists()
    assert contract["joint_audit"]["refused"] is False


def test_a_negative_turn_digit_is_refused():
    """FANUC has no negative turn digit; ROBOGUIDE refuses the whole file (ASBN-009)."""
    points = [{"joints_deg": (0.0, 0.0, 0.0, 0.0, -60.0, 0.0),
               "config": "N U T, -1, 0, 0"}]
    report = validate.check_config_hints(points)
    assert report["ok"] is False
    assert report["failures"][0]["reason"] == "config_negative_turn"


def test_a_wrist_letter_over_the_wrong_branch_is_refused():
    points = [{"joints_deg": (0.0, 0.0, 0.0, 0.0, 120.0, 0.0), "config": "N U T, 0, 0, 0"}]
    report = validate.check_config_hints(points)
    assert report["ok"] is False
    assert report["failures"][0]["reason"] == "config_wrist_mismatch"
