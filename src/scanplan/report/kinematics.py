"""The kinematics, budget and surface-bias sections of the report.

Three blocks that all answer the same kind of question - "why does the plan look like
this?" - and that were impossible to answer before:

* `kinematics` says whether the joint gate ran, which envelope it used, where the delivered
  poses sit relative to the limits, and how far the worst joint step is. Without it a plan
  that silently dropped a third of its candidates to the joint limits is indistinguishable
  from one that simply could not reach them.
* `budget` says which of the three end states the run reached - met the coverage target,
  ran into the point ceiling, or ran out of candidates - and what each extra tolerance
  bought, so "four times the points for half a percent" is a number.
* `surface_profile` says where each derived growth/ordering threshold came from, because a
  knob that silently changes six other numbers has to be auditable.

**What "joint step" means here (schema 2).**  A `J` instruction interpolates the numbers in
the program, so its step is the literal difference of those numbers - that is what the
machine does and therefore what the report must state.  An `L` instruction moves the TCP
along a straight line and the joints follow it; the literal end-to-end difference is not a
motion the arm performs, so it is reported separately as `cartesian_segments`.  Before
schema 2 every step was measured circularly, which is correct for "how far did the wrist
physically turn" and wrong for "what will the controller do": on the reference delivery one
segment was reported as 103.9 deg while the file commanded 271.2 deg.
"""

from __future__ import annotations

import numpy as np

from ..config import SURFACE_ANCHORS, PlannerConfig

#: A step is reported in `steps_over_limit` when it is larger than this. The limit is a
#: reporting threshold, not a refusal - see `path.BranchAssignment`.
_DEFAULT_STEP_LIMIT_DEG = 40.0


def _rows(viewpoints) -> list[dict]:
    """The delivered points that carry a joint state, in program order."""
    records = [viewpoint.as_dict() for viewpoint in viewpoints]
    rows = [row for row in records if row.get("joints_deg") is not None]
    rows.sort(key=lambda row: (row.get("order") is None, row.get("order") or 0))
    return rows


def _steps(rows) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Per-segment literal and circular steps, plus what instruction reaches each point.

    `literal` is what the controller interpolates for a `J` move. `circular` is the physical
    pose change, kept because it is the honest answer to "how far did the wrist turn" and is
    the only meaningful number for a segment the program flies as a line.
    """
    joints = np.array([row["joints_deg"] for row in rows], dtype=float)
    if len(joints) < 2:
        return np.zeros((0, 6)), np.zeros((0, 6)), []
    literal = np.abs(np.diff(joints, axis=0))
    circular = np.abs((np.diff(joints, axis=0) + 180.0) % 360.0 - 180.0)
    motions = [str(row.get("motion") or "L") for row in rows[1:]]
    return literal, circular, motions


def _track_block(config, rows) -> dict | None:
    """The seventh axis, as the program used it. None when there is no rail.

    Reported for the same reason the joint steps are: "the rail helped" is a claim, and
    "it slid 2.4 m over 6 moves, worst one 1.1 m" is a measurement. `steps_over_limit` is a
    review trigger, not a refusal - the same contract as `max_step_deg`.
    """
    robot = config.robot
    if not bool(getattr(robot, "track_enabled", False)):
        return None
    values = [float(row["track_mm"]) for row in rows if row.get("track_mm") is not None]
    steps = [abs(second - first) for first, second in zip(values, values[1:], strict=False)]
    limit = float(getattr(robot, "track_max_step_mm", 0.0))
    length = float(getattr(robot, "track_length_mm", 0.0))
    margin = float(getattr(robot, "track_margin_mm", 0.0))
    return {
        "enabled": True,
        "axis": str(getattr(robot, "track_axis", "+X")),
        "length_mm": length,
        "usable_mm": [margin, length - margin],
        "margin_mm": margin,
        "points": len(values),
        "used_span_mm": ([round(min(values), 3), round(max(values), 3)]
                         if values else None),
        "travel_mm": round(float(sum(steps)), 3),
        "max_step_mm": round(float(max(steps)), 3) if steps else None,
        "max_step_at": (int(np.argmax(steps)) + 1 if steps else None),
        "max_step_limit_mm": limit,
        "steps_over_limit": [index + 1 for index, step in enumerate(steps) if step > limit],
        "program_column": "E1" if bool(getattr(robot, "track_emit_e1", True)) else None,
        "note": (
            "a linear seventh axis along the base frame's +X: it is planned with the arm "
            "(chosen per point, held through a sweep, interpolated along a straight move) "
            "and it is written into the program as the extended-axis column; the rail's own "
            "servo tuning and its collision with the fixture are not modelled"),
    }


def kinematics_block(config: PlannerConfig, viewpoints, path, kinematics=None) -> dict:
    """Joint-side evidence: envelope, J5 distribution, branches, worst step.

    Step numbers follow the instruction that reaches each point (schema 2): `J` points are
    measured by the literal numbers the program writes, `L` points are reported as a
    separate list of Cartesian segments and are not counted as joint steps at all.
    """
    robot = getattr(kinematics, "robot", None)
    envelope = getattr(kinematics, "envelope", None)
    assignment = getattr(path, "branches", None) if path is not None else None
    rows = _rows(viewpoints)
    joints = np.array([row["joints_deg"] for row in rows], dtype=float).reshape(-1, 6)

    histogram: dict[str, int] = {}
    if len(joints):
        magnitudes = np.abs(joints[:, 4])
        edges = (0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 180.0)
        for low, high in zip(edges[:-1], edges[1:], strict=True):
            count = int(((magnitudes >= low) & (magnitudes < high)).sum())
            if count:
                histogram[f"{low:.0f}-{high:.0f}"] = count

    literal, circular, motions = _steps(rows)
    limit = float(getattr(config.robot, "max_step_deg", _DEFAULT_STEP_LIMIT_DEG))
    joint_segments = [index for index, motion in enumerate(motions) if motion == "J"]
    cartesian_segments = [index for index, motion in enumerate(motions) if motion != "J"]
    if joint_segments:
        joint_steps = literal[joint_segments].max(axis=1)
        worst_at = int(joint_segments[int(np.argmax(joint_steps))]) + 1
        worst = float(joint_steps.max())
        over = [int(index) + 1 for index in joint_segments if limit < literal[index].max()]
    else:
        worst, worst_at, over = None, None, []

    block = {
        "schema": 2,
        "enabled": bool(getattr(kinematics, "active", False)),
        "model": config.robot.model,
        "envelope": envelope.describe() if envelope is not None else None,
        "limits_source": (robot.limits_source if robot is not None else None),
        "usable_limits_deg": ([list(pair) for pair in robot.usable]
                              if robot is not None else None),
        "wrist_identified": bool(config.robot.wrist_identified),
        "joint_margin_deg": config.robot.joint_margin_deg,
        "joint_span_fraction": config.robot.joint_span_fraction,
        "wrist_flip_policy": config.robot.wrist_flip_policy,
        "j5_min_abs_deg": config.robot.j5_min_abs_deg,
        "max_step_deg_limit": limit,
        "j5_histogram_deg": histogram,
        "branch_histogram": dict(getattr(assignment, "histogram", {})),
        "track": _track_block(config, rows),
        "min_margin_deg": getattr(assignment, "min_margin_deg", None),
        # Schema 2: literal, and only for the points the program reaches with a `J`.
        "max_step_deg": worst,
        "max_step_at": worst_at,
        "step_metric": (
            "literal: the difference of the joint numbers the program writes, measured only "
            "where the instruction is J. A segment flown as a line is listed separately, "
            "because its end-to-end joint difference is not a motion the arm performs"),
        "cartesian_segments": [int(index) + 1 for index in cartesian_segments],
        "max_step_circular_deg": (round(float(circular.max()), 3)
                                  if len(circular) else None),
        "points_without_solution": len(getattr(assignment, "failed", [])),
        "steps_over_limit": over,
        "joint_planned": bool(getattr(path, "joint_planned", False)) if path else False,
        "joint_connector": bool(config.stroke.joint_connector),
        "connector_failures": dict(sorted(getattr(path, "connector_failures", {}).items()))
        if path is not None else {},
        "cache": (kinematics.summary() if kinematics is not None else None),
        "config_note": (
            "the CONFIG letters are derived from the chosen joint solution, never pinned: a "
            "point with J5 > 0 is written F and a point with J5 < 0 is written T, because the "
            "letter tells the controller which wrist branch to take. A pinned letter over a "
            "solution of the other sign is what ROBOGUIDE rejects as an inconsistent "
            "configuration, so the branch filter and the letter come from the same number"),
    }
    if not config.robot.wrist_identified:
        block["joint_values_withheld"] = (
            "wrist_identified is false, so absolute joint values are not reported; the gate "
            "may only rank branches against each other")
    if viewpoints:
        records = [viewpoint.as_dict() for viewpoint in viewpoints]
        block["points_with_config"] = sum(1 for item in records if item.get("config"))
    block["j6"] = _j6_block(config, literal, circular, motions, rows)
    return block


def _j6_block(config: PlannerConfig, literal: np.ndarray, circular: np.ndarray,
              motions: list[str], rows) -> dict:
    """How much J6 moves, and how much of it the program actually commands.

    J6 is reported on its own because it is the axis that moves the scanner SIDEWAYS: the head
    sits ~250 mm off the flange axis, so one degree of J6 slides the optical origin 4.35 mm
    along an arc roughly parallel to the surface being scanned. That is the one direction the
    45 mm body sphere is blind in, so a big J6 step is the least-checked large motion in the
    program.

    `max_step_deg` is a cost in the sequencing (`robot.j6_weight`) and, above
    `robot.j6_max_step_deg`, an extra penalty - never a refusal.  Dropping a sweep to avoid a
    turn loses the coverage that sweep carried, and the delivery stage has a cheaper answer:
    re-try the point at another roll about the same axis, or slow the move down.
    """
    cap = getattr(config.robot, "j6_max_step_deg", None)
    report_deg = float(getattr(config.robot, "j6_report_deg", 90.0))
    tool = np.asarray(config.robot.tool_xyz_mm, dtype=float).reshape(3)
    lever = float(np.hypot(tool[0], tool[1]))
    block = {
        "lever_arm_mm": round(lever, 2),
        "mm_per_deg": round(lever * np.pi / 180.0, 3),
        "weight": float(getattr(config.robot, "j6_weight", 1.0)),
        "travel_weight": float(getattr(config.robot, "j6_travel_weight", 0.0)),
        "cap_deg": cap,
        "report_deg": report_deg,
        "note": (
            "lever_arm_mm is the tool offset projected onto the flange plane - the radius J6 "
            "swings the scanner origin along. `literal_*` are the numbers the program writes "
            "on J moves; `physical_*` are the pose changes and are diagnostic only. "
            "`over_cap` lists segments above robot.j6_max_step_deg, which cost extra in the "
            "sequencing, and `over_report` lists the segments above robot.j6_report_deg, "
            "which are named without being priced. Neither is a refusal: the delivery stage "
            "re-tries the roll or slows the move instead of dropping the sweep. `weight` "
            "scales J6 inside the all-axis step, `travel_weight` charges every degree of it "
            "on top - the first stops one big twist, the second stops a run of small ones"),
    }
    if len(literal) < 1:
        block["available"] = False
        block["reason"] = "fewer than two delivered points carry a joint state"
        return block
    j6_literal = literal[:, 5]
    j6_physical = circular[:, 5]
    joint_segments = [index for index, motion in enumerate(motions) if motion == "J"]
    joint_j6 = [float(j6_literal[index]) for index in joint_segments]
    edges = (0.0, 20.0, 45.0, 90.0, 180.0, 360.0)
    histogram = {}
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        count = int(((j6_physical >= low) & (j6_physical < high)).sum())
        if count:
            histogram[f"{low:.0f}-{high:.0f}"] = count
    # Both lists name segments the program reaches with a `J`, for the reason the top block
    # gives: a segment flown as a line does not perform its end-to-end joint difference, so
    # counting it here would point at a move nobody makes.
    over_cap = ([index + 1 for index in joint_segments
                 if j6_literal[index] > float(cap)] if cap is not None else [])
    over_report = [index + 1 for index in joint_segments
                   if j6_literal[index] > report_deg]
    block.update({
        "available": True,
        "segments": int(len(j6_literal)),
        "literal_total_deg": round(float(np.sum(joint_j6)), 1),
        "literal_max_deg": round(float(np.max(joint_j6)), 1) if joint_j6 else None,
        "literal_max_at": (joint_segments[int(np.argmax(joint_j6))] + 1
                           if joint_j6 else None),
        "physical_total_deg": round(float(j6_physical.sum()), 1),
        "physical_max_deg": round(float(j6_physical.max()), 1),
        "over_cap": over_cap,
        "over_report": over_report,
        "dominant_segments": int((circular.argmax(axis=1) == 5).sum()),
        "histogram": histogram,
    })
    return block


def budget_block(config: PlannerConfig, path) -> dict:
    """Which end state the run reached, and what each tolerance bought."""
    history = getattr(path, "budget_history", []) if path is not None else []
    verdict = getattr(path, "verdict", "") if path is not None else ""
    delivered = getattr(path, "turns", None) if path is not None else None
    dropped = len(getattr(path, "dropped", [])) if path is not None else 0
    return {
        "max_points": config.target.max_points,
        "delivered_points": delivered,
        "verdict": verdict or ("unbudgeted" if config.target.max_points is None
                               else "unknown"),
        "dropped_by_hard_gate": dropped,
        "trials": [{"lambda": float(lam), "points": int(points), "coverage_pct": float(pct)}
                   for lam, points, pct in history],
        "note": ("verdict is one of target_bound (reached the coverage target first), "
                 "budget_bound (ran into max_points first) or exhausted (the pool had "
                 "nothing more to offer); a fourth answer does not exist"),
    }


def failure_attribution_block(config: PlannerConfig, path, uncovered_by_reason) -> dict:
    """Why things were lost, in three separate tables because they are three separate layers.

    The old report had one sentence - "no route into it keeps clearance and rate" - covering
    clearance, turn rate, joint reach and "nothing else left". That sentence cannot be acted
    on: fixing it might mean the fixture, the connector, or the limit policy. These counters
    can.

    * `sample_reasons` is about SAMPLES: why a piece of the part is not covered. Geometric.
    * `item_drop_reasons` is about ITEMS: why a whole sweep or pose left the program.
    * `connector_primitive_refusals` is about ATTEMPTS: how often each connector primitive was
      tried and refused. A refusal is not a loss unless the item was then dropped - the joint
      move is refused on almost every transition that ends up using the retract fallback.
    """
    sample = dict(uncovered_by_reason or {})
    drops = dict(getattr(path, "drop_reasons", {}) or {})
    refusals = dict(getattr(path, "connector_failures", {}) or {})
    return {
        "sample_reasons": sample,
        "item_drop_reasons": drops,
        "connector_primitive_refusals": refusals,
        "layers_note": {
            "sample_reasons": "why a SAMPLE is uncovered; geometric, and the only one of the "
                              "three that moves the coverage figure",
            "item_drop_reasons": "why an ITEM (a sweep or a pose) left the program",
            "connector_primitive_refusals": "how often each connector primitive was tried and "
                                            "refused; a refusal is not a loss unless the item "
                                            "was then dropped",
        },
        "joint_gate_enabled": bool(config.robot.enabled),
        "note": ("with the joint gate off, `item_drop_reasons` only ever says the hard gate "
                 "found no connector - there is no arm to refuse anything"),
    }


def surface_profile_block(config: PlannerConfig) -> dict:
    """The bias in force, the values it derived, and where each one came from."""
    keys = tuple(SURFACE_ANCHORS[0][1])
    derived = {f"stroke.{key}": getattr(config.stroke, key) for key in keys}
    anchors = {f"{anchor:g}": {key: float(values[key]) for key in keys}
               for anchor, values in SURFACE_ANCHORS}
    return {
        "surface_bias": float(config.target.surface_bias),
        "effective": derived,
        "sources": dict(config.parameter_sources),
        "anchors": anchors,
        "note": ("a value written in the override file wins over the profile, and the "
                 "source column says so; 0.5 must reproduce the pre-2026-10-22 values "
                 "exactly, which is what the anchor table is pinned to"),
    }
