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

A sweep and a reposition are two different moves and are emitted as two different
instructions. The straight `L` section between two points is what the coverage figure was
computed from, so it stays straight; the one point the planner labels as a joint travel
(`role` `joint`, the pre-point of a joint route) is a reposition and is emitted with
`travel_motion` (a joint move by default, at a percentage speed, because that is what a
workshop flies a reposition with). Which point is which comes from `role`, the label
`scanplan.path` writes into the point table.
"""

from __future__ import annotations

from datetime import datetime

from scanplan.roboguide.wpr import fanuc_num

# The role `scanplan.path` gives the pre-point of a joint travel: the head is brought to it
# with a joint move, carrying the next sweep's pose, and the short straight approach into
# the sweep follows it. It is the only point in an `L` program whose move is not straight -
# a `traverse` or `reorient` point belongs to the straight-only routes, so those stay `L`.
REPOSITION_ROLES = ("joint",)


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


def _is_reposition(point: dict) -> bool:
    """Whether this point exists only to carry the head between two sweeps."""
    return str(point.get("role", "")).strip().lower() in REPOSITION_ROLES


def build_ls(
    points: list[dict],
    *,
    prog_name: str,
    uframe: int,
    utool: int,
    motion: str = "J",
    speed: str = "100%",
    termination: str = "FINE",
    travel_motion: str = "J",
    travel_speed: str = "100%",
    travel_termination: str = "CNT100",
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

    `termination` matters for a continuous scan: at `FINE` the TCP stops at every point,
    at `CNT0` it rounds the corner by (almost) nothing and keeps going. A stroke plan is
    only valid at `CNT0`-like termination, because the linear segment between two points
    - the thing the coverage was computed from - is what gets flown.

    `travel_motion`, `travel_speed` and `travel_termination` are for the repositions of an
    `L` program: a point whose `role` is `joint` is the pre-point of a joint travel, the
    head crossing between two sweeps in mid air, and it is the one move that may leave the
    straight line. A `J` program is a reachability review where every point is a joint move
    already, so the travel settings are not read there; `travel_motion="L"` gives back the
    uniform straight program.

    A joint reposition does not follow the straight segment anything was measured on - the
    exclusion is what "sweeps nothing" buys, not a claim that the joint path is checked.
    """
    if isinstance(stamp, str):
        stamp_text = stamp
    else:
        stamp_text = (stamp or datetime.now()).strftime("%y-%m-%d  TIME %H:%M:%S")

    def command(index: int, is_joint: bool = False, reposition: bool = False) -> str:
        # a joint move takes a percentage speed on FANUC, never a mm/sec one, and that
        # covers both the joint representation points and a joint reposition
        if is_joint:
            return f"J P[{index}] {travel_speed} FINE"
        if motion.upper() == "L":
            # a sweep is the segment the coverage figure was computed from; a retract, or
            # the approach dive into the next sweep, is a segment the clearance check
            # measured. Both stay straight. The pre-point of a joint travel is the head
            # crossing between two sweeps in mid air, carrying the next pose - nothing is
            # claimed about what it sweeps, so it takes the travel moves, at the percentage
            # speed FANUC wants for a joint move.
            if reposition and travel_motion.upper() == "J":
                return f"J P[{index}] {travel_speed} {travel_termination.upper()}"
            return f"L P[{index}] {speed} {termination.upper()}"
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
    repositions = []
    for pt in points:
        point_indices.append(next_index)
        repositions.append(_is_reposition(pt))
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

    moves: list[tuple[int, bool, bool]] = []
    if home_index:
        moves.append((home_index, True, False))
    moves.extend((index, False, reposition)
                 for index, reposition in zip(point_indices, repositions, strict=True))
    if home_index:
        moves.append((home_index, True, False))

    for index, is_joint, reposition in moves:
        mn_lines.append(
            f"{len(mn_lines) + 1:>4}:"
            f"{command(index, is_joint=is_joint, reposition=reposition)}    ;")
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
