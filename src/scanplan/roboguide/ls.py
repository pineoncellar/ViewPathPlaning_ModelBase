"""Writer for the FANUC ASCII `.LS` program format.

Vendored from `reference/PHASE_6_Export_Roboguide.py` when implemented. That file
already solved the parts that are tedious and unforgiving:

* the `/ATTR` header must be self-consistent (PROG_SIZE, LINE_COUNT,
  MEMORY_SIZE), and the file name must match the `/PROG` name exactly, or the
  controller refuses the import with "Invalid name in /PROG section";
* positions are emitted as a `/POS` table of `P[n]` blocks that the motion
  instructions in `/MN` refer to by index.

Cartesian positions only. A joint block exists for an optional home pose, but
the planner never produces joint angles - solving them is the controller's job,
done in ROBOGUIDE.

Two details change whether a point looks reachable:

* the CONFIG string hints which of the robot's posture solutions to take. The
  same Cartesian pose can have several joint solutions, some inside the limits
  and some not, so "unreachable in ROBOGUIDE" has to be diagnosed by trying
  another configuration before blaming the planner.
* J moves are more permissive than L moves and match the "bring the scanner to a
  pose" intent, so J with FINE termination is the default for a review.
"""

from __future__ import annotations

import re
from datetime import datetime

from scanplan.roboguide.wpr import fanuc_num

# What the controller accepts in the speed field of a motion instruction: a percentage, or a
# rate with a unit. Anything else is a typo that would otherwise be written into the program
# and rejected on the machine, so it is refused here where the message can say why.
_SPEED = re.compile(
    r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>%|mm/sec|mm/s|deg/sec|deg/s|sec)?\s*$",
    re.IGNORECASE)

# A joint move takes a percentage speed; a rate with a unit is refused by the controller.
# Used when a point is written as joint angles but the program's speed is a rate.
_JOINT_BLOCK_SPEED = "50%"

# The CONFIG field, as the CONTROLLER parses it: shoulder / elbow / wrist letters, then
# three TURN digits. The digits are revolutions past the axis's own travel and are
# NON-NEGATIVE: ROBOGUIDE answers `ASBN-009 on line ..., column ... Configuration error`
# and refuses the whole file for a `-1`. Checking the shape here is what keeps this
# writer from producing a program that cannot be imported at all.
_CONFIG = re.compile(
    r"^\s*(?P<shoulder>[NF])\s+(?P<elbow>[UD])\s+(?P<wrist>[TF])\s*,\s*"
    r"(?P<turns>\d{1,2}\s*,\s*\d{1,2}\s*,\s*\d{1,2})\s*$",
    re.IGNORECASE)


def check_config(config: str) -> str:
    """Validate a FANUC CONFIG string and return it in canonical form.

    The field is written verbatim into every `P[n]` block, so a malformed one costs the
    whole import rather than one point - which is exactly what happened on 2026-10-27,
    when a negative turn digit stopped `T1.LS` at load time. Refusing it here turns that
    into an error message that names the string and says what a legal one looks like.
    """
    found = _CONFIG.match(str(config))
    if found is None:
        raise ValueError(
            f"CONFIG {config!r} is not a FANUC configuration: expected "
            "'<N|F> <U|D> <T|F>, t1, t2, t3' with three NON-NEGATIVE turn digits, e.g. "
            "'N U T, 0, 0, 0'. A negative digit (e.g. 'N U F, -1, 0, 0') makes ROBOGUIDE "
            "reject the whole program with ASBN-009 'Configuration error' at load time")
    turns = ", ".join(part.strip() for part in found["turns"].split(","))
    return (f"{found['shoulder'].upper()} {found['elbow'].upper()} "
            f"{found['wrist'].upper()}, {turns}")


def check_speed(speed: str, motion: str = "J") -> str:
    """Validate a FANUC speed field and return it normalised.

    A percentage is the only form that means the same thing for both motion types; a rate
    without a unit is ambiguous, so it is refused rather than guessed at.
    """
    text = str(speed).strip()
    found = _SPEED.match(text)
    if found is None:
        raise ValueError(f"speed {speed!r} is not a FANUC speed: use e.g. '100%' or "
                         "'250mm/sec'")
    value = float(found.group("value"))
    unit = (found.group("unit") or "").lower()
    if value <= 0.0:
        raise ValueError(f"speed {speed!r} must be positive")
    if unit == "%" and value > 100.0:
        raise ValueError(f"speed {speed!r} is over 100 percent")
    if motion.upper() == "J" and unit in ("mm/sec", "mm/s"):
        raise ValueError(f"a joint move cannot be given a linear speed ({speed!r}): "
                         "use a percentage, or switch motion.type to L")
    return text


def _pos_block(index: int, kind: str, uframe: int, utool: int, config: str, payload) -> str:
    if kind == "joint":
        joints = payload
        return (
            f"P[{index}]{{\n"
            f"   GP1:\n"
            f"\tUF : {uframe}, UT : {utool}, \n"
            f"\tJ1= {fanuc_num(joints[0])} deg,\tJ2= {fanuc_num(joints[1])} deg,"
            f"\tJ3= {fanuc_num(joints[2])} deg,\n"
            f"\tJ4= {fanuc_num(joints[3])} deg,\tJ5= {fanuc_num(joints[4])} deg,"
            f"\tJ6= {fanuc_num(joints[5])} deg\n"
            f"}};"
        )
    x, y, z, w, p, r = payload
    return (
        f"P[{index}]{{\n"
        f"   GP1:\n"
        f"\tUF : {uframe}, UT : {utool},\t\tCONFIG : '{config}',\n"
        f"\tX = {fanuc_num(x)}  mm,\tY = {fanuc_num(y)}  mm,\tZ = {fanuc_num(z)}  mm,\n"
        f"\tW = {fanuc_num(w)} deg,\tP = {fanuc_num(p)} deg,\tR = {fanuc_num(r)} deg\n"
        f"}};"
    )


def build_ls(
    points: list[dict],
    *,
    prog_name: str,
    uframe: int,
    utool: int,
    motion: str = "J",
    speed: str = "100%",
    termination: str = "FINE",
    config: str = "N U T, 0, 0, 0",
    comment: str = "scan viewpoints",
    constant: str = "",
    home_joints=None,
    joint_blocks: bool = False,
    stamp: datetime | str | None = None,
) -> str:
    """Render the whole `.LS` program.

    `points` is the exporter's point table (`xyz_mm` in mm, `wpr_deg` in degrees),
    already in the frame the UFRAME is taught to. `prog_name` must be the file
    name, or the controller refuses the import.

    `termination` matters for a continuous scan: at `FINE` the TCP stops at every point,
    at `CNT0` it rounds the corner by (almost) nothing and keeps going. A stroke plan is
    only valid at `CNT0`-like termination, because the linear segment between two points
    - the thing the coverage was computed from - is what gets flown.

    Per-point overrides. Each point may carry its own `config`, `motion`, `termination` and
    `joints_deg`; the arguments here are the defaults. That is not a convenience: one
    Cartesian pose has up to eight joint solutions and only some are inside the limits, so a
    single CONFIG string for the whole program is exactly how every point ended up pinned to
    the same branch. `joint_blocks=True` writes `P[n]` as joint angles for points that have
    them, which removes the controller's choice altogether - the most direct fix, and the
    one that needs the model to be right.
    """
    if isinstance(stamp, str):
        stamp_text = stamp
    else:
        stamp_text = (stamp or datetime.now()).strftime("%y-%m-%d  TIME %H:%M:%S")
    speed = check_speed(speed, motion)
    config = check_config(config)

    def command(index: int, point_motion: str, point_termination: str,
                point_speed: str, is_joint: bool = False) -> str:
        # a joint representation point can only be reached with a joint move, and
        # FANUC takes a percentage speed for those
        if point_motion.upper() == "L" and not is_joint:
            return f"L P[{index}] {point_speed} {point_termination.upper()}"
        return f"J P[{index}] {point_speed} FINE"

    mn_lines: list[str] = []
    pos_blocks: list[str] = []
    next_index = 1

    home_index = None
    if home_joints:
        home_index = next_index
        next_index += 1
        pos_blocks.append(_pos_block(home_index, "joint", uframe, utool, config, home_joints))

    point_indices = []
    point_specs: list[tuple[str, str, str, bool]] = []
    for pt in points:
        point_indices.append(next_index)
        point_config = check_config(pt.get("config") or config)
        point_motion = str(pt.get("motion") or motion)
        point_termination = str(pt.get("termination") or termination)
        point_speed = check_speed(str(pt.get("speed") or speed), point_motion)
        joints = pt.get("joints_deg")
        as_joint = bool(joint_blocks and joints is not None)
        if as_joint and not point_speed.endswith("%"):
            # The speed belongs to the L instruction a sweep is written with; a joint move
            # takes a percentage, and the controller refuses mm/sec on one
            point_speed = _JOINT_BLOCK_SPEED
        if as_joint:
            pos_blocks.append(
                _pos_block(next_index, "joint", uframe, utool, point_config, joints))
        else:
            pos_blocks.append(
                _pos_block(
                    next_index,
                    "cart",
                    uframe,
                    utool,
                    point_config,
                    (*pt["xyz_mm"].tolist(), *pt["wpr_deg"]),
                )
            )
        point_specs.append((point_motion, point_termination, point_speed, as_joint))
        next_index += 1

    mn_lines.append(f"{len(mn_lines) + 1:>4}:  UFRAME_NUM={uframe} ;")
    mn_lines.append(f"{len(mn_lines) + 1:>4}:  UTOOL_NUM={utool} ;")
    if constant:
        mn_lines.append(f"{len(mn_lines) + 1:>4}:  {constant} ;")

    spec_for = dict(zip(point_indices, point_specs, strict=True))
    moves: list[tuple[int, bool]] = []
    if home_index:
        moves.append((home_index, True))
    # A joint-representation point can only be reached with a joint move; asking for `L`
    # there is rejected by the controller, so the instruction follows the representation.
    moves.extend((index, spec_for[index][3]) for index in point_indices)
    if home_index:
        moves.append((home_index, True))

    for index, is_joint in moves:
        point_motion, point_termination, point_speed, _written = spec_for.get(
            index, (motion, termination, speed, is_joint))
        mn_lines.append(
            f"{len(mn_lines) + 1:>4}:"
            f"{command(index, point_motion, point_termination, point_speed, is_joint)}    ;")
    # real FANUC programs simply end after the last instruction; there is no END
    # statement in the instruction list

    body = "\n".join(mn_lines)
    prog_size = len(body) + sum(len(b) for b in pos_blocks)

    return (
        f"/PROG  {prog_name}\n"
        f"/ATTR\n"
        f"OWNER\t\t= MNEDITOR;\n"
        f'COMMENT\t\t= "{comment}";\n'
        f"PROG_SIZE\t= {prog_size};\n"
        f"CREATE\t\t= DATE {stamp_text};\n"
        f"MODIFIED\t= DATE {stamp_text};\n"
        f"FILE_NAME\t= ;\n"
        f"VERSION\t\t= 0;\n"
        f"LINE_COUNT\t= {len(mn_lines)};\n"
        f"MEMORY_SIZE\t= {prog_size + 316};\n"
        f"PROTECT\t\t= READ_WRITE;\n"
        f"TCD:  STACK_SIZE\t= 0,\n"
        f"      TASK_PRIORITY\t= 50,\n"
        f"      TIME_SLICE\t= 0,\n"
        f"      BUSY_LAMP_OFF\t= 0,\n"
        f"      ABORT_REQUEST\t= 0,\n"
        f"      PAUSE_REQUEST\t= 0;\n"
        f"DEFAULT_GROUP\t= 1,*,*,*,*;\n"
        f"CONTROL_CODE\t= 00000000 00000000;\n"
        f"/MN\n"
        f"{body}\n"
        f"/POS\n"
        f"{chr(10).join(pos_blocks)}\n"
        f"/END\n"
    )
