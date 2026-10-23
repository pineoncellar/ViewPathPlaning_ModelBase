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
    strokes = report.get("stroke_plan", {})
    if strokes.get("enabled") and strokes.get("delivered"):
        lines.append(f"strokes                "
                     f"{strokes.get('delivered_strokes') or strokes['candidates']} delivered "
                     f"of {strokes['candidates']} candidates, swept coverage "
                     f"{strokes['swept_coverage_pct']:.2f}%")
        for violation in path.get("constraint_violations", []):
            lines.append(f"CONSTRAINT             {violation}")
    kinematics = report.get("kinematics", {})
    lines += [
        f"ray backend            {report['runtime']['ray_backend']}",
        ("NOTE: joint limits were screened here (see the kinematics block); collision and "
         "final reachability are still decided by ROBOGUIDE and the controller, and "
         "robot-arm occlusion of the tracker is not modelled."
         if kinematics.get("enabled") else
         "NOTE: coverage is geometric reachability only - robot reachability, joint limits "
         "and robot collision are checked in ROBOGUIDE, and robot-arm occlusion of the "
         "tracker is not modelled."),
    ]
    if kinematics.get("enabled"):
        lines.append(
            f"joints                 min margin {kinematics.get('min_margin_deg')} deg, "
            f"worst step {kinematics.get('max_step_deg')} deg "
            f"(limit {kinematics.get('max_step_deg_limit')}), "
            f"branches {kinematics.get('branch_histogram')}")
    budget = report.get("budget", {})
    if budget.get("max_points") is not None:
        lines.append(f"budget                 {budget.get('verdict')} at "
                     f"{budget.get('delivered_points')} of {budget['max_points']} points")
    return lines


def summary_text(report: dict) -> str:
    """`summary_lines` as one string, for a `summary.txt` beside the report."""
    return "\n".join(summary_lines(report)) + "\n"
