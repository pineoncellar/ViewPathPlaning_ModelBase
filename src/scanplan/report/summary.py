"""The console summary: the numbers a reader sees, ending with the caveat they must not miss."""

from __future__ import annotations


def _layers_line(report: dict) -> str:
    """One line for the three coverage levels, and an explicit refusal when one is unknown."""
    layers = report["coverage_layers"]
    pool = layers["geometric_pool_pct"]
    delivered = layers["delivered_pct"]
    execution = layers["execution_verified_pct"]
    verdict = (f"execution {execution:.2f}%" if execution is not None
               else "execution UNVERIFIED")
    return (f"coverage layers        pool {pool:.2f}% -> delivered {delivered:.2f}% "
            f"-> {verdict}")


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
        *([_layers_line(report)] if report.get("coverage_layers") else []),
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
    approach = report.get("approach") or {}
    poses = approach.get("poses") or {}
    if poses.get("available"):
        lines.append(
            f"approach               {poses['looking_up_over_0_deg']} pose(s) look up "
            f"(max {poses['max_look_up_deg']:g} deg), {poses['below_part_bottom']} below the "
            f"part bottom; {approach['target']['area_facing_down_pct']:.2f}% of the target "
            "faces down - reported, not screened")
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
        j6 = kinematics.get("j6") or {}
        if j6.get("available"):
            lines.append(
                f"J6                     total {j6['total_step_deg']} deg, "
                f"worst {j6['worst_step_deg']} deg at segment {j6['worst_step_at']}, "
                f"dominant on {j6['dominant_segments']} of {j6['segments']} segments "
                f"({j6['lever_arm_mm']:g} mm lever = {j6['mm_per_deg']:g} mm/deg) "
                f"- reported, not refused")
        program = report.get("program", {})
        levels = program.get("connector_levels") or {}
        if levels:
            lines.append(f"connector              {levels}"
                         + (f", refused {kinematics['connector_failures']}"
                            if kinematics.get("connector_failures") else ""))
    budget = report.get("budget", {})
    if budget.get("max_points") is not None:
        lines.append(f"budget                 {budget.get('verdict')} at "
                     f"{budget.get('delivered_points')} of {budget['max_points']} points")
    fast = report.get("fast_mode", {})
    if fast.get("enabled"):
        what = ", ".join(fast.get("layers") or []) or "no layer recorded"
        lines.append(
            f"FAST MODE              {what}; {fast.get('sample_grid_mm'):g} mm grid, "
            f"occlusion {'modelled' if fast.get('occlusion_modelled') else 'NOT modelled'}"
            " - these numbers are for the shape of the path, not for acceptance")
    return lines


def summary_text(report: dict) -> str:
    """`summary_lines` as one string, for a `summary.txt` beside the report."""
    return "\n".join(summary_lines(report)) + "\n"
