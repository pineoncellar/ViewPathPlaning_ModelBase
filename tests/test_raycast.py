"""Ray casting, the one expensive primitive the planner depends on.

To assert: a ray hits at its true geometric distance; a miss reports infinity and
no triangle rather than zero; the nearest hit wins, and the name of the mesh that
owns it comes back with it (leak attribution needs to say WHAT blocked the line
of sight); chunking does not change the answer; empty input is empty output; and
the tuple the intersector returns is read in the documented order - a swap there
unpacks without error and produces nonsense, which is the bug this pins down.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scanplan.raycast import EMBREE_AVAILABLE, RayCaster, backend_name, nearest_across

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_hits_the_plate_at_the_expected_distance(flat_plate):
    distance, triangle = RayCaster(flat_plate).nearest(
        np.array([[5.0, 5.0, 10.0]]), np.array([[0.0, 0.0, -1.0]]))
    assert pytest.approx(float(distance[0]), abs=1e-6) == 10.0 - 0.7
    assert triangle[0] >= 0


def test_a_miss_reports_infinity_not_zero(flat_plate):
    distance, triangle = RayCaster(flat_plate).nearest(
        np.array([[5.0, 5.0, 10.0]]), np.array([[0.0, 0.0, 1.0]]))
    assert np.isinf(distance[0]) and triangle[0] == -1


def test_only_the_nearest_hit_is_reported(flat_plate, fixture_box):
    _, name, _ = nearest_across(
        {"plate": RayCaster(flat_plate), "box": RayCaster(fixture_box)},
        np.array([[5.0, 5.0, 100.0]]), np.array([[0.0, 0.0, -1.0]]))
    assert name[0] == "plate"


def test_nearest_across_names_the_blocking_mesh(fixture_box, base_box):
    distance, name, _ = nearest_across(
        {"fixture": RayCaster(fixture_box), "base": RayCaster(base_box)},
        np.array([[0.0, 0.0, 100.0]]), np.array([[0.0, 0.0, -1.0]]))
    assert name[0] == "fixture"
    assert pytest.approx(float(distance[0]), abs=1e-6) == 100.0 - (-5.0)  # fixture top face


def test_the_return_tuple_is_not_unpacked_in_the_wrong_order(flat_plate):
    # origin 7.7 mm above a plate whose top face is at z = 0.7, i.e. a 7.0 mm gap.
    # Reading the tuple as (locations, ray, tri) yields a distance built from the hit
    # coordinates (73.0, 45.0, 0.7), which is nowhere near 7.0
    distance, triangle = RayCaster(flat_plate).nearest(
        np.array([[73.0, 45.0, 7.7]]), np.array([[0.0, 0.0, -1.0]]))
    assert pytest.approx(float(distance[0]), abs=1e-6) == 7.0
    assert triangle[0] >= 0


def test_chunking_does_not_change_the_result(flat_plate):
    rng = np.random.default_rng(0)
    origins = np.column_stack([rng.uniform(-95.0, 95.0, 300), rng.uniform(-95.0, 95.0, 300),
                               np.full(300, 5.0)])
    directions = np.tile([0.0, 0.0, -1.0], (300, 1))
    one = RayCaster(flat_plate).nearest(origins, directions)
    many = RayCaster(flat_plate, chunk=17).nearest(origins, directions)
    assert np.allclose(one[0], many[0]) and np.array_equal(one[1], many[1])


def test_empty_input_returns_empty_arrays(flat_plate):
    distance, triangle = RayCaster(flat_plate).nearest(np.zeros((0, 3)), np.zeros((0, 3)))
    assert distance.shape == (0,) and triangle.shape == (0,)


def test_mismatched_input_lengths_are_refused(flat_plate):
    with pytest.raises(ValueError, match="same length"):
        RayCaster(flat_plate).nearest(np.zeros((2, 3)), np.zeros((3, 3)))


def test_the_backend_is_named_and_is_one_of_the_two_documented_ones():
    assert backend_name() in {"embree", "trimesh-python"}
    assert isinstance(EMBREE_AVAILABLE, bool)
    # the fallback has to keep working, so the accelerated path must never be assumed
    assert backend_name() == ("embree" if EMBREE_AVAILABLE else "trimesh-python")


def test_embree_is_used_whenever_the_extra_is_installed():
    if importlib.util.find_spec("embreex") is None:
        pytest.skip("the embree extra is not installed")
    # `import trimesh` already imports trimesh.ray, which caches the failed bridge as an
    # ExceptionModule when pyembree is not registered yet - and then silently uses the
    # pure-Python intersector, about 1600x slower. Something else importing trimesh first
    # must not cost the planner its fast path.
    assert backend_name() == "embree"


def test_the_pure_python_fallback_answers_when_embree_cannot_be_imported():
    # The environment stays installable without the extra (spec 9.1), so the fallback
    # is a promise, not a footnote. A subprocess is needed because the intersector is
    # chosen once at import time.
    script = (
        "import sys\n"
        "sys.modules['embreex'] = None\n"
        "import numpy as np, trimesh\n"
        "from scanplan.raycast import EMBREE_AVAILABLE, RayCaster, backend_name\n"
        "assert EMBREE_AVAILABLE is False, backend_name()\n"
        "plate = trimesh.creation.box(extents=(10.0, 10.0, 0.7))\n"
        "d, t = RayCaster(plate).nearest(np.array([[0.0, 0.0, 5.0]]),"
        " np.array([[0.0, 0.0, -1.0]]))\n"
        "print(backend_name(), round(float(d[0]), 3), int(t[0]))\n"
    )
    result = subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    backend, distance, triangle = result.stdout.split()
    assert backend == "trimesh-python"
    assert float(distance) == pytest.approx(5.0 - 0.35, abs=1e-6)
    assert int(triangle) >= 0
