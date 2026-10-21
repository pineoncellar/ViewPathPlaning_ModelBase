"""Interactive cabin-side picker for matching CAD datum locations."""

from __future__ import annotations

import numpy as np

from .raycast import RayCaster


def snap_to_reference(click_xz, reference_points, *, radius_mm: float = 30.0):
    """Snap a click near a reference marker to its exact CAD point."""
    click = np.asarray(click_xz, dtype=float).reshape(2)
    references = np.asarray(reference_points, dtype=float).reshape(-1, 3)
    if radius_mm < 0.0:
        raise ValueError("reference snap radius must not be negative")
    if len(references) == 0:
        return None
    distances = np.linalg.norm(references[:, [0, 2]] - click, axis=1)
    nearest = int(np.argmin(distances))
    return references[nearest].copy() if distances[nearest] <= radius_mm else None


def pick_cad_datums(part, *, count: int = 3, reference_points=None) -> np.ndarray:
    """Let the operator click CAD datum locations in an X-Z view from the cabin side.

    Each screen click casts a ray from below the part along +Y, so the returned
    coordinate is the first cabin-facing surface point at that X/Z location.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    if count <= 0:
        raise ValueError("the number of datum points must be positive")
    references = None
    if reference_points is not None:
        references = np.asarray(reference_points, dtype=float)
        if references.shape != (count, 3) or not np.isfinite(references).all():
            raise ValueError(f"reference points must have shape ({count}, 3)")

    vertices = np.asarray(part.vertices, dtype=float)
    faces = np.asarray(part.faces, dtype=np.int64)
    normals = np.asarray(part.face_normals, dtype=float)
    visible = normals[:, 1] < 0.15
    if not visible.any():
        raise ValueError("the part has no faces visible from the cabin side (negative Y)")

    triangles = vertices[faces[visible]]
    caster = RayCaster(part)
    start_y = float(part.bounds[0, 1] - max(10.0, part.extents[1] * 0.05))
    picked: list[np.ndarray] = []

    figure, axes = plt.subplots(figsize=(12, 8))
    axes.add_collection(
        PolyCollection(
            triangles[:, :, [0, 2]],
            facecolors="#c75a45",
            edgecolors="none",
            antialiased=False,
        )
    )
    axes.autoscale_view()
    axes.set_aspect("equal", adjustable="datalim")
    axes.set_xlabel("X (mm)")
    axes.set_ylabel("Z (mm)")
    axes.set_title(
        f"Cabin-side view (Y < 0): click CAD datum 1 of {count}\n"
        + ("Click the matching cyan reference marker.\n"
           if references is not None else "")
        + "Left click selects; right click undoes the last point"
    )
    axes.grid(alpha=0.2)
    markers = axes.scatter([], [], s=90, c="#173f8a", edgecolors="white", linewidths=1.2)
    annotations = []
    if references is not None:
        axes.scatter(
            references[:, 0],
            references[:, 2],
            s=170,
            facecolors="none",
            edgecolors="#00e5ff",
            linewidths=2.0,
            label="Reference CAD datum locations",
        )
        for index, point in enumerate(references, start=1):
            axes.annotate(
                f"REFERENCE P{index}",
                (point[0], point[2]),
                xytext=(6, -14),
                textcoords="offset points",
                fontsize=8,
                color="#007080",
                weight="bold",
            )
        axes.legend(loc="upper right")

    def refresh() -> None:
        if picked:
            coordinates = np.asarray(picked)
            markers.set_offsets(coordinates[:, [0, 2]])
        else:
            markers.set_offsets(np.empty((0, 2)))
        for annotation in annotations:
            annotation.remove()
        annotations.clear()
        for index, point in enumerate(picked, start=1):
            annotations.append(
                axes.annotate(
                    f"P{index} ({point[0]:.1f}, {point[1]:.1f}, {point[2]:.1f})",
                    (point[0], point[2]),
                    xytext=(6, 8),
                    textcoords="offset points",
                    fontsize=8,
                    color="#173f8a",
                )
            )
        prompt = (
            f"Cabin-side view (Y < 0): click CAD datum "
            f"{len(picked) + 1} of {count}\n"
            + ("Click the matching cyan reference marker.\n"
               if references is not None else "")
            + "Left click selects; right click undoes the last point"
        )
        axes.set_title(
            prompt if len(picked) < count else "Three CAD datum points selected"
        )
        figure.canvas.draw_idle()

    def on_click(event) -> None:
        if event.inaxes is not axes:
            return
        if event.button == 3:
            if picked:
                picked.pop()
                print("Removed last CAD datum point.", flush=True)
                refresh()
            return
        if event.button != 1 or event.xdata is None or event.ydata is None:
            return
        if len(picked) >= count:
            return

        if references is not None:
            snapped = snap_to_reference([event.xdata, event.ydata], references)
            if snapped is not None:
                picked.append(snapped)
                print(
                    f"CAD datum {len(picked)} snapped to reference: "
                    f"{snapped[0]:.3f}, {snapped[1]:.3f}, {snapped[2]:.3f} mm",
                    flush=True,
                )
                refresh()
                if len(picked) == count:
                    plt.close(figure)
                return

        origin = np.array([[float(event.xdata), start_y, float(event.ydata)]])
        direction = np.array([[0.0, 1.0, 0.0]])
        distance, triangle = caster.nearest(origin, direction)
        if triangle[0] < 0:
            print("No part surface at that location; choose a point on the model.", flush=True)
            return
        point = origin[0] + direction[0] * distance[0]
        picked.append(point)
        print(
            f"CAD datum {len(picked)}: "
            f"{point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f} mm",
            flush=True,
        )
        refresh()
        if len(picked) == count:
            plt.close(figure)

    figure.canvas.mpl_connect("button_press_event", on_click)
    plt.show(block=True)
    if len(picked) != count:
        raise ValueError(
            f"datum picking cancelled after {len(picked)} of {count} points"
        )
    result = np.asarray(picked, dtype=float)
    _ensure_non_collinear(result)
    return result


def _ensure_non_collinear(points: np.ndarray) -> None:
    centred = points - points.mean(axis=0)
    singular_values = np.linalg.svd(centred, compute_uv=False)
    if singular_values[0] <= 1e-9 or singular_values[1] <= singular_values[0] * 1e-6:
        raise ValueError("the selected CAD datum points are collinear or too close together")
