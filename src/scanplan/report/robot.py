"""The arm block: which robot's joints constrained the plan, and to what.

The numbers here are the ones a reviewer checks the plan against - the documented stops,
the derated window every pose had to fit in, and the degrees each end gives up - so a
"the planner rejected 40% of the candidates" figure can be traced to a limit rather than
to a hunch.
"""

from __future__ import annotations

from ..config import PlannerConfig
from ..robot import RobotArm


def robot_block(arm: RobotArm | None, config: PlannerConfig) -> dict:
    """What the joint window was, or why there was none."""
    if arm is None:
        return {
            "checked": False,
            "reason": ("robot.enabled is off in this configuration, so no pose was solved "
                       "against a joint window: reachability and joint limits are "
                       "ROBOGUIDE's to decide for this plan"),
        }
    described = arm.describe()
    margin = float(config.robot.joint_limit_margin)
    documented = arm.table_limits_deg.tolist()
    allowed = arm.allowed_table_deg.tolist()
    window = [
        {
            "joint": f"J{axis + 1}",
            "documented_deg": [round(low, 3), round(high, 3)],
            "allowed_deg": [round(allowed[axis][0], 3), round(allowed[axis][1], 3)],
            "given_up_deg": [round(allowed[axis][0] - low, 3),
                             round(high - allowed[axis][1], 3)],
        }
        for axis, (low, high) in enumerate(documented)
    ]
    return {
        "checked": True,
        "margin_note": (
            f"every documented limit is derated to {margin:.0%}, so the outermost "
            f"{100.0 * (1.0 - margin):.0f}% of each axis' travel is never used and no "
            "planned pose works against a mechanical stop"),
        "window": window,
        **described,
    }
