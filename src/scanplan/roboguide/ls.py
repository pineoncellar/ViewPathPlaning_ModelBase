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

from datetime import datetime

from scanplan.roboguide.wpr import fanuc_num


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
    config: str = "N U T, 0, 0, 0",
    comment: str = "scan viewpoints",
    constant: str = "",
    home_joints=None,
    stamp: datetime | str | None = None,
) -> str:
    """Render the whole `.LS` program.

    `points` is the exporter's point table (`xyz_mm` in mm, `wpr_deg` in degrees),
    already in the frame the UFRAME is taught to. `prog_name` must be the file
    name, or the controller refuses the import.
    """
    if isinstance(stamp, str):
        stamp_text = stamp
    else:
        stamp_text = (stamp or datetime.now()).strftime("%y-%m-%d  TIME %H:%M:%S")

    def command(index: int, is_joint: bool = False) -> str:
        # a joint representation point can only be reached with a joint move, and
        # FANUC takes a percentage speed for those
        if motion.upper() == "L" and not is_joint:
            return f"L P[{index}] {speed} FINE"
        return f"J P[{index}] {speed} FINE"

    mn_lines: list[str] = []
    pos_blocks: list[str] = []
    next_index = 1

    home_index = None
    if home_joints:
        home_index = next_index
        next_index += 1
        pos_blocks.append(_pos_block(home_index, "joint", uframe, utool, config, home_joints))

    point_indices = []
    for pt in points:
        point_indices.append(next_index)
        pos_blocks.append(
            _pos_block(
                next_index,
                "cart",
                uframe,
                utool,
                config,
                (*pt["xyz_mm"].tolist(), *pt["wpr_deg"]),
            )
        )
        next_index += 1

    mn_lines.append(f"{len(mn_lines) + 1:>4}:  UFRAME_NUM={uframe} ;")
    mn_lines.append(f"{len(mn_lines) + 1:>4}:  UTOOL_NUM={utool} ;")
    if constant:
        mn_lines.append(f"{len(mn_lines) + 1:>4}:  {constant} ;")

    moves: list[tuple[int, bool]] = []
    if home_index:
        moves.append((home_index, True))
    moves.extend((index, False) for index in point_indices)
    if home_index:
        moves.append((home_index, True))

    for index, is_joint in moves:
        mn_lines.append(f"{len(mn_lines) + 1:>4}:{command(index, is_joint)}    ;")
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
