#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File     :   fit_robot_pose_bank.py
@Desc     :   用一批实测位姿（关节角 + 笛卡尔位姿）重新标定 R-2000iC/125L，打印可直接抄进配置的参数、残差与 CONFIG 字母比对。
@Note     :   数据本身绝不入库（客户工位），本脚本只读一个外部 JSON：
              {"poses": [{"joints": [J1..J6], "xyzwpr": [X,Y,Z,W,P,R], "config": "N U T, 0, 0, 0"}]}
              拟合前先判"是臂错了还是只换了工具"，判据见 docs/robot-calibration.md。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scanplan.config import RobotParams, load_config
from scanplan.kinematics import Robot, _ypr_rotation, rotation_angle_deg

# 自由参数：五个臂长 + 工具的六个数。关节零位/符号/腕部倾角一律钉住当前配置——
# 它们是"这台控制器的约定"，不该在一次几何拟合里被挪动（j6_zero 与工具偏航角
# 完全简并，一起放开会得到非唯一解，还会把 CONFIG 的 J6 圈数算错）。
ARM_FIELDS = ("base_offset_mm", "j2_height_mm", "upper_arm_mm", "forearm_mm",
              "forearm_offset_mm")

# 残差验收线：输入数据本身只到 0.001 mm / 0.001 deg，模型能到 0.02 mm / 0.002 deg
# 就已经是精度的下限，再小没有意义。
TOL_POSITION_MM = 0.02
TOL_ANGLE_DEG = 0.002


def rotation_vector(matrix: np.ndarray) -> np.ndarray:
    """旋转矩阵 → 旋转向量，用作姿态残差（三轴各一个数，避免欧拉角跳变）。"""
    angle = float(np.arccos(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0)))
    if angle < 1e-12:
        return np.zeros(3)
    axis = np.array([matrix[2, 1] - matrix[1, 2],
                     matrix[0, 2] - matrix[2, 0],
                     matrix[1, 0] - matrix[0, 1]])
    return axis / np.linalg.norm(axis) * angle


def wrap_deg(angle: float) -> float:
    """把角度折回 (-180, 180]，只为打印好看，不改变含义。"""
    return float((angle + 180.0) % 360.0 - 180.0)


def other_euler_branch(ypr) -> tuple[float, float, float]:
    """同一个旋转的另一个欧拉解：R = Rz(R)Ry(P)Rx(W) 时是 (W+180, 180-P, R+180)。"""
    return (wrap_deg(ypr[0] + 180.0), wrap_deg(180.0 - ypr[1]), wrap_deg(ypr[2] + 180.0))


def euler_from(matrix: np.ndarray) -> np.ndarray:
    """旋转矩阵 → (W, P, R)，FANUC 固定轴 XYZ：R = Rz(R) * Ry(P) * Rx(W)。"""
    pitch = float(np.arcsin(np.clip(-matrix[2, 0], -1.0, 1.0)))
    return np.degrees([np.arctan2(matrix[2, 1], matrix[2, 2]),
                       pitch,
                       np.arctan2(matrix[1, 0], matrix[0, 0])])


def load_bank(path: Path) -> list[dict]:
    body = json.loads(path.read_text(encoding="utf-8-sig"))
    poses = body.get("poses") if isinstance(body, dict) else body
    if not isinstance(poses, list) or len(poses) < 4:
        raise SystemExit("pose bank needs at least four poses to separate arm from tool")
    out = []
    for index, item in enumerate(poses, start=1):
        try:
            out.append({"joints": np.array(item["joints"], dtype=float).reshape(6),
                        "measured": np.array(item["xyzwpr"], dtype=float).reshape(6),
                        "config": str(item.get("config", "")).strip()})
        except (KeyError, ValueError) as error:
            raise SystemExit(f"pose {index} is malformed: {error}") from error
    return out


def build_robot(vector: np.ndarray, base: RobotParams) -> Robot:
    return Robot(replace(base, enabled=True,
                         **{name: float(vector[index])
                            for index, name in enumerate(ARM_FIELDS)},
                         tool_xyz_mm=tuple(float(value) for value in vector[5:8]),
                         tool_ypr_deg=tuple(float(value) for value in vector[8:11])))


def residuals(vector: np.ndarray, bank: list[dict], base: RobotParams,
              indices: list[int]) -> np.ndarray:
    robot = build_robot(vector, base)
    out = []
    for index in indices:
        point, frame = robot.forward(bank[index]["joints"])
        measured = bank[index]["measured"]
        out.extend(point - measured[:3])
        out.extend(np.degrees(rotation_vector(frame.T @ _ypr_rotation(measured[3:]))))
    return np.asarray(out)


# ------------------------------- 阶段一：分辨"臂错"还是"工具变" -------------------------------
def diagnose_tool_only(bank: list[dict], base: RobotParams) -> dict:
    """在法兰系里看残差是否恒定。

    臂参数一错，腕心就按位姿的关节角漂移，残差在法兰系里会随位姿变化；
    只换了工具时，残差是一个固定在法兰系的常向量 + 一个常旋转。
    """
    robot = Robot(replace(base, enabled=True))
    flange_tool = robot.tool_rotation @ robot.tool_vector
    offsets, rotations = [], []
    for item in bank:
        point, frame = robot.forward(item["joints"])
        flange = frame @ robot.tool_rotation.T
        offsets.append(flange.T @ (point - item["measured"][:3]))
        rotations.append(frame.T @ _ypr_rotation(item["measured"][3:]))
    offsets = np.array(offsets)
    mean = offsets.mean(axis=0)
    mean_rotation = np.mean(rotations, axis=0)
    u, _sigma, vt = np.linalg.svd(mean_rotation)
    rotation = u @ vt
    deviation = max(float(rotation_angle_deg(rotation, item)) for item in rotations)
    new_rotation = robot.tool_rotation @ rotation
    new_vector = new_rotation.T @ (flange_tool - mean)
    return {"offset": mean, "spread_mm": float(np.linalg.norm(offsets - mean, axis=1).max()),
            "rotation": rotation, "rotation_spread_deg": deviation,
            "tool_rotation": new_rotation, "tool_vector": new_vector}


# ------------------------------- 阶段二：联合精解 ---------------------------------------------
def refine(bank: list[dict], base: RobotParams, start: np.ndarray) -> np.ndarray:
    indices = list(range(len(bank)))
    found = least_squares(
        residuals, start, args=(bank, base, indices), method="trf",
        x_scale=[200.0, 50.0, 1000.0, 1000.0, 200.0, 100.0, 100.0, 100.0, 10.0, 10.0, 10.0],
        xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=50000)
    return found.x


def residual_table(bank: list[dict], vector: np.ndarray, base: RobotParams,
                   indices: list[int] | None = None) -> list[tuple[int, float, float]]:
    robot = build_robot(vector, base)
    rows = []
    for index in (indices if indices is not None else range(len(bank))):
        point, frame = robot.forward(bank[index]["joints"])
        measured = bank[index]["measured"]
        rows.append((index + 1, float(np.linalg.norm(point - measured[:3])),
                     rotation_angle_deg(frame, _ypr_rotation(measured[3:]))))
    return rows


def leave_one_out(bank: list[dict], base: RobotParams, vector: np.ndarray
                  ) -> list[tuple[int, float, float]]:
    """每次留一组不参与拟合，再看它能不能被预测——这才是"换成新数据还准不准"。"""
    rows = []
    for held in range(len(bank)):
        keep = [index for index in range(len(bank)) if index != held]
        fit = least_squares(residuals, vector, args=(bank, base, keep), method="trf",
                            xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=50000)
        rows.extend(residual_table(bank, fit.x, base, [held]))
    return rows


# ------------------------------- 阶段三：CONFIG 字母比对 --------------------------------------
def branch_table(bank: list[dict], vector: np.ndarray, base: RobotParams) -> list[str]:
    """位姿解出的分支 vs 示教器上的 CONFIG 字母。

    比对方法：用实测笛卡尔位姿做逆解，挑与实测关节角最接近的那个根（正常情况下
    差 < 0.01 deg）。字母对应的规则不靠猜，靠这张表数出来。
    """
    robot = build_robot(vector, base)
    lines = []
    for index, item in enumerate(bank, start=1):
        measured = item["measured"]
        found = robot.solutions(measured[:3], _ypr_rotation(measured[3:]))
        if not found:
            lines.append(f"  #{index}: no inverse solution")
            continue
        gap, best = min(((max(abs((a - b + 180.0) % 360.0 - 180.0)
                              for a, b in zip(candidate.joints_deg, item["joints"],
                                              strict=True)), candidate)
                         for candidate in found), key=lambda pair: pair[0])
        letter = "T" if best.joints_deg[4] < 0.0 else "F"
        lines.append(f"  #{index}: J5={best.joints_deg[4]:9.3f} -> {letter}   "
                     f"derived={best.shoulder}{best.elbow}{best.wrist_flip}   "
                     f"pendant={item['config'] or '(none)':20s}   "
                     f"joint gap={gap:.5f} deg")
    return lines


# ------------------------------- 主流程 -------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bank", help="pose bank JSON (joints + xyzwpr, live outside the repo)")
    parser.add_argument("--tol-mm", type=float, default=TOL_POSITION_MM)
    parser.add_argument("--tol-deg", type=float, default=TOL_ANGLE_DEG)
    parser.add_argument("--skip-loo", action="store_true", help="skip leave-one-out")
    parser.add_argument("--skip-branches", action="store_true", help="skip the CONFIG table")
    args = parser.parse_args()

    bank = load_bank(Path(args.bank))
    shipped = load_config().robot
    fallback = RobotParams.disabled()
    drift = [(name, getattr(shipped, name), getattr(fallback, name))
             for name in ("tool_xyz_mm", "tool_ypr_deg", "base_offset_mm", "upper_arm_mm",
                          "forearm_mm", "forearm_offset_mm")
             if getattr(shipped, name) != getattr(fallback, name)]
    print(f"pose bank: {len(bank)} poses")
    print(f"base model: configs/ (enabled={shipped.enabled})")
    if drift:
        print("WARNING: configs/ and RobotParams.disabled() disagree - the planner and the "
              "tests would be validating different machines:")
        for name, left, right in drift:
            print(f"  {name}: configs={left}  defaults={right}")

    diagnosis = diagnose_tool_only(bank, shipped)
    print("\n[stage 1] is this a tool change or a model error?")
    print(f"  flange-frame offset   {np.round(diagnosis['offset'], 4)} mm")
    print(f"  spread over poses     {diagnosis['spread_mm']:.4f} mm "
          f"(constant means the arm is right)")
    print(f"  tool rotation change  {rotation_angle_deg(np.eye(3), diagnosis['rotation']):.4f}"
          f" deg, spread {diagnosis['rotation_spread_deg']:.4f} deg")
    if diagnosis["spread_mm"] > 1.0:
        print("  VERDICT: the arm model does NOT explain this bank. Do not change only the")
        print("           tool; re-derive the arm hypotheses before touching the config.")
    else:
        print("  VERDICT: arm + wrist conventions reproduce every pose up to a constant")
        print("           offset and rotation in the flange frame, i.e. only the tool moved.")

    start = np.array([*[getattr(shipped, name) for name in ARM_FIELDS],
                      *diagnosis["tool_vector"], *euler_from(diagnosis["tool_rotation"])])
    vector = refine(bank, shipped, start)
    robot = build_robot(vector, shipped)

    print("\n[stage 2] fitted parameters")
    for index, name in enumerate(ARM_FIELDS):
        print(f"  {name:22s} = {vector[index]!r}")
    print(f"  {'tool_xyz_mm':22s} = {tuple(float(np.round(v, 3)) for v in vector[5:8])}")
    print(f"  {'tool_ypr_deg':22s} = {tuple(float(np.round(v, 3)) for v in vector[8:11])}")
    print(f"  {'|tool|':22s} = {np.linalg.norm(vector[5:8]):.4f}")

    flange_tool = robot.tool_rotation @ robot.tool_vector
    alternative = other_euler_branch(vector[8:11])
    print("\n  pendant cross-check - UTOOL should read:")
    print(f"    X {flange_tool[0]:.1f}  Y {flange_tool[1]:.1f}  Z {flange_tool[2]:.1f}  "
          f"W {vector[8]:.1f}  P {vector[9]:.1f}  R {vector[10]:.1f}")
    print(f"    (the same rotation, other euler branch: W {alternative[0]:.1f}  "
          f"P {alternative[1]:.1f}  R {alternative[2]:.1f})")
    print(f"    scanner frame +Z in flange coordinates: {np.round(robot.tool_rotation[:, 2], 4)}")

    rows = residual_table(bank, vector, shipped)
    print("\n[stage 3] residual per pose")
    for index, position, angle in rows:
        print(f"  #{index}: dP={position:8.5f} mm  dR={angle:9.6f} deg")
    worst_position = max(row[1] for row in rows)
    worst_angle = max(row[2] for row in rows)
    print(f"  MAX dP={worst_position:.5f} mm   MAX dR={worst_angle:.6f} deg")

    if not args.skip_loo:
        held = leave_one_out(bank, shipped, vector)
        worst_held = max(row[1] for row in held)
        print(f"\n[stage 4] leave-one-out: worst held-out dP={worst_held:.5f} mm")
        for index, position, angle in held:
            print(f"  #{index}: dP={position:8.5f} mm  dR={angle:9.6f} deg")

    if not args.skip_branches:
        print("\n[stage 5] CONFIG letters: derived vs pendant")
        for line in branch_table(bank, vector, shipped):
            print(line)

    ok = worst_position <= args.tol_mm and worst_angle <= args.tol_deg
    print(f"\n{'PASS' if ok else 'FAIL'}: tolerance {args.tol_mm} mm / {args.tol_deg} deg")
    if not ok:
        print("  A bank this bad is not a tool change. Check the UTOOL number the points were")
        print("  collected in, and whether the joint and Cartesian rows come from the same pose.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
