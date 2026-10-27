"""The console summary: the numbers a reader sees, ending with the caveat they must not miss."""

from __future__ import annotations


def summary_lines(report: dict) -> list[str]:
    """The console summary, ending with the caveat a reader must not miss."""
    reasons = ", ".join(f"{key} {value}" for key, value in sorted(
        report["uncovered_by_reason"].items())) or "none"
    lines = [
        f"target face            {report['target_face_mm2'] / 1e6:.4f} m2 ({report['face']})",
        f"target samples         {report['target_points']}",
        f"exempt (clamped)       {report['exempt']['points']} samples "
        f"{report['exempt']['area_mm2'] / 1e6:.4f} m2",
        f"coverage               {report['coverage_pct']:.2f}%  "
        f"(pool could reach {report['pool_reach_pct']:.2f}%)"
        + ("" if report.get("program", {}).get("coverage_target_pct") is None else
           f"  target {report['program']['coverage_target_pct']:g}% "
           + ("met" if report["coverage_pct"] + 1e-9
              >= report["program"]["coverage_target_pct"] else "NOT MET")),
        f"points                 {report['viewpoints']} "
        f"({report['viewpoints_greedy']} greedy + {report['viewpoints_repair']} repair"
        + (f" + {report['viewpoints_transition']} transition"
           if report.get("viewpoints_transition") else "") + ")",
        f"uncovered              {report['uncovered_points']} samples "
        f"{report['uncovered_area_mm2'] / 1e6:.4f} m2  [{reasons}]",
        f"max incidence          {report['placeholders']['max_incidence_deg']:.0f} deg "
        "(PLACEHOLDER)",
    ]
    path = report.get("program", {})
    if path.get("path_length_mm") is not None:
        interrupts = path.get("interrupts", {})
        lines += [
            f"motion                 {path['motion']} {path['speed_mm_s']:g} mm/s "
            f"{path['termination']}",
            f"path                   {path['turns']} corners, "
            f"{path['path_length_mm']:.0f} mm, "
            f"{interrupts.get('translation', 0)} translation + "
            f"{interrupts.get('reorient', 0)} re-orient interrupts",
        ]
    joints = path.get("joints")
    if joints:
        lines.append(f"joint window           {joints['checked_moves']} moves checked, "
                     f"{joints.get('out_of_window_moves', 0)} with no legal "
                     f"configuration, worst utilization "
                     f"{joints.get('worst_utilization', 0.0):.3f}")
    strokes = report.get("stroke_plan", {})
    if strokes.get("enabled") and strokes.get("delivered"):
        lines.append(f"strokes                "
                     f"{strokes.get('delivered_strokes') or strokes['candidates']} delivered "
                     f"of {strokes['candidates']} candidates, swept coverage "
                     f"{strokes['swept_coverage_pct']:.2f}%")
        for violation in path.get("constraint_violations", []):
            lines.append(f"CONSTRAINT             {violation}")
    lines += [
        f"ray backend            {report['runtime']['ray_backend']}",
        _note_line(report),
    ]
    return lines


def _note_line(report: dict) -> str:
    """The caveat a reader must not miss, in this run's terms."""
    arm = report.get("robot", {})
    if arm.get("checked"):
        return ("NOTE: coverage is geometric reachability filtered by the arm's joint "
                f"window ({arm.get('model', 'robot')}, each stop derated to "
                f"{arm.get('joint_limit_margin', 1.0):.0%}) - robot collision is checked in "
                "ROBOGUIDE, and robot-arm occlusion of the tracker is not modelled.")
    return ("NOTE: coverage is geometric reachability only - robot reachability, joint "
            "limits and robot collision are checked in ROBOGUIDE, and robot-arm occlusion "
            "of the tracker is not modelled.")


def summary_text(report: dict) -> str:
    """`summary_lines` as one string, for a `summary.txt` beside the report."""
    return "\n".join(summary_lines(report)) + "\n"
