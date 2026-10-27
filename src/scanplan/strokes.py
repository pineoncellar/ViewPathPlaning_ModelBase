"""Scan strokes: a straight segment the scanner travels along while it scans.

Why the planning unit changed. A viewpoint says "be here and shoot"; the workshop
does not work that way - the operator drives the head along a path and the scanner
keeps scanning on the way. A pose-only plan therefore describes something that is
never executed, and the operator has to invent the path between the poses. A
stroke describes what is actually done:

    S = (p0, p1, R)     p0, p1 in mm, R a CONSTANT orientation

`R[:, 2]` is the optical axis, exactly as in :class:`~scanplan.scanner_model.ScannerPose`.
A constant pose is what makes the swept volume exactly computable, and it is also
what makes the coverage of a stroke the union of the coverage of its sampled poses.
A stroke of length zero IS a viewpoint, which is how "turn the stroke planner off"
stays provable.

Two sampling steps, and the difference is not cosmetic:

* `coverage_step_mm` (50) decides coverage and costs one ray per target sample;
* `check_step_mm` (10) decides clearance and tracker visibility only, so it is a
  KD-tree lookup and a single ray per position.

The footprint is 500 mm long along the motion, so at 50 mm neighbouring footprints
overlap by 90% and the union of the sampled poses differs from the true swept
union only by a band of at most one step at each end. Those two ends are therefore
sampled at `check_step_mm` as well, which closes the band without paying for a fine
grid over the whole segment.

Conditions are the SAME six as for a pose; nothing new is introduced. A point is
covered by a stroke when at least one sampled pose on it satisfies all six. A
stroke is discarded when ANY sampled position fails the body-clearance or tracker
gate, because the head cannot be somewhere it does not fit - it does not merely
cover less there.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix

from .config import PlannerConfig
from .coverage import CoverageResult
from .geometry import tangents, unit
from .raycast import nearest_across
from .robot import RobotArm
from .rolls import roll_for_axis
from .scanner_model import ScannerModel, ScannerPose, measure_mask
from .scene import Station
from .target_surface import SampleSet

_EPS = 1e-9
# Below this the two strokes are the same line: directions are unit vectors, so a
# 0.1 deg difference is a genuine (very slightly) different scan.
_PARALLEL_TOL = 1e-3


@dataclass(frozen=True, eq=False)
class Stroke:
    """A straight scan segment with one constant pose.

    `seed` and `direction` are provenance only: which pool pose grew this stroke and
    which of its tangents it ran along. They are reported so a reviewer can see where
    a stroke came from; nothing downstream reads them.

    Equality is by value and defined by hand: the `__eq__` a dataclass generates for fields
    holding numpy arrays raises instead of answering.
    """

    start: np.ndarray
    end: np.ndarray
    axis: np.ndarray
    roll_deg: float = 0.0
    seed: int = -1
    direction: int = 0

    def __eq__(self, other) -> bool:
        if not isinstance(other, Stroke):
            return NotImplemented
        return (np.array_equal(np.asarray(self.start, dtype=float),
                               np.asarray(other.start, dtype=float))
                and np.array_equal(np.asarray(self.end, dtype=float),
                                   np.asarray(other.end, dtype=float))
                and np.array_equal(np.asarray(self.axis, dtype=float),
                                   np.asarray(other.axis, dtype=float))
                and abs(float(self.roll_deg) - float(other.roll_deg)) < 1e-9
                and self.seed == other.seed and self.direction == other.direction)

    @property
    def length_mm(self) -> float:
        """How far the TCP travels along the stroke."""
        return float(np.linalg.norm(np.asarray(self.end, dtype=float)
                                    - np.asarray(self.start, dtype=float)))

    @property
    def start_pose(self) -> ScannerPose:
        """The pose at the first end, which is also the pose at every other point."""
        return ScannerPose(position=np.asarray(self.start, dtype=float),
                           axis=np.asarray(self.axis, dtype=float),
                           roll_deg=float(self.roll_deg))

    @property
    def unit_direction(self) -> np.ndarray:
        """The direction of travel; zero length yields an arbitrary unit vector."""
        delta = np.asarray(self.end, dtype=float) - np.asarray(self.start, dtype=float)
        length = float(np.linalg.norm(delta))
        return delta / length if length > 0.0 else np.array([0.0, 0.0, 0.0])

    def pose_at(self, offset_mm: float) -> ScannerPose:
        """The pose `offset_mm` along the stroke from `start`."""
        position = (np.asarray(self.start, dtype=float)
                    + self.unit_direction * float(offset_mm))
        return ScannerPose(position=position, axis=np.asarray(self.axis, dtype=float),
                           roll_deg=float(self.roll_deg))

    def reversed(self) -> Stroke:
        """The same physical sweep, walked the other way - free for a path planner."""
        return Stroke(start=np.asarray(self.end, dtype=float),
                      end=np.asarray(self.start, dtype=float),
                      axis=np.asarray(self.axis, dtype=float), roll_deg=self.roll_deg,
                      seed=self.seed, direction=self.direction)

    def positions_mm(self, coverage_step_mm: float, check_step_mm: float) -> np.ndarray:
        """Where to put the head to evaluate this stroke's coverage.

        Coarse in the middle, fine within one coverage step of each end: see the
        module docstring for why the ends need it.
        """
        offsets = stroke_offsets(self.length_mm, coverage_step_mm, check_step_mm)
        if self.length_mm <= 0.0:
            return np.asarray(self.start, dtype=float).reshape(1, 3)
        step = self.unit_direction
        return np.asarray(self.start, dtype=float) + offsets[:, None] * step[None, :]

    def check_positions_mm(self, check_step_mm: float) -> np.ndarray:
        """Where to test clearance and tracker visibility: every check step, ends included."""
        length = self.length_mm
        if length <= 0.0:
            return np.asarray(self.start, dtype=float).reshape(1, 3)
        count = max(1, int(np.ceil(length / check_step_mm)))
        offsets = np.linspace(0.0, length, count + 1)
        step = self.unit_direction
        return np.asarray(self.start, dtype=float) + offsets[:, None] * step[None, :]


def pose_stroke(position, axis, roll_deg: float = 0.0) -> Stroke:
    """A single pose as the stroke of zero length that it is.

    The point of this is the accounting, not the geometry: a selected pose has to be
    handed to the path builder like anything else, or the coverage figure counts a sweep
    that never appears in the delivered program. The design's measured run promised
    99.741% and the program swept 91.98% for exactly that reason.
    """
    point = np.asarray(position, dtype=float).reshape(3)
    return Stroke(start=point, end=point.copy(),
                  axis=unit(np.asarray(axis, dtype=float).reshape(3)),
                  roll_deg=float(roll_deg))


def stroke_offsets(length_mm: float, coverage_step_mm: float,
                   check_step_mm: float) -> np.ndarray:
    """The offsets along a stroke of `length_mm` where coverage is evaluated.

    Both ends are included, and that single fact is what makes a coarse step safe: the
    footprint is convex and about 500 mm long along the motion, so the union of the
    footprints sampled at positions spaced less than that equals the continuous sweep
    exactly. Include the two ends and a 50 mm step samples the same region a continuous
    pass does, at a fifth of the ray cost; leave them out and each end loses a band one
    step wide - which is why the first version of this had to refine both caps.

    A stroke shorter than one coverage step is sampled at the cheap step, so its coverage
    still lands on a sensible number of poses rather than on one.
    """
    length = float(length_mm)
    if length <= 0.0:
        return np.zeros(1)
    step = coverage_step_mm if length > coverage_step_mm else check_step_mm
    count = max(1, int(np.ceil(length / step)))
    return np.linspace(0.0, length, count + 1)


@dataclass
class StrokeCoverage:
    """Which target samples each stroke can measure, in the shape `select` expects.

    The duck type is deliberate: `matrix()`, `sets`, `n_targets`, `area_mm2`,
    `target_area_mm2` and `pool_reach()` are exactly what the selection stage reads, so
    strokes and poses can be offered to the same greedy loop without a wrapper.
    """

    strokes: list[Stroke]
    sets: list[np.ndarray]
    quality: list[np.ndarray]
    tracker_ok: np.ndarray
    counters: dict[str, int]
    n_targets: int
    area_mm2: np.ndarray
    target_area_mm2: float
    min_clearance_mm: float | None
    _matrix: csr_matrix | None = field(default=None, repr=False, compare=False)

    @property
    def poses(self) -> list[ScannerPose]:
        """One representative pose per stroke, for the code that speaks in poses."""
        return [stroke.start_pose for stroke in self.strokes]

    def matrix(self) -> csr_matrix:
        """The stroke x sample coverage matrix, built once and kept as sparse bools."""
        if self._matrix is None:
            rows: list[int] = []
            columns: list[int] = []
            for index, row in enumerate(self.sets):
                rows.extend([index] * len(row))
                columns.extend(int(column) for column in row)
            data = np.ones(len(rows), dtype=bool)
            self._matrix = csr_matrix((data, (rows, columns)),
                                      shape=(len(self.strokes), self.n_targets), dtype=bool)
        return self._matrix

    def pool_reach(self) -> np.ndarray:
        """Which samples any stroke can measure at all."""
        reached = np.zeros(self.n_targets, dtype=bool)
        for row in self.sets:
            reached[row] = True
        return reached

    @property
    def pool_reach_pct(self) -> float:
        """Area share of the target the strokes reach - the ceiling for selection."""
        if self.target_area_mm2 <= 0.0:
            return 0.0
        return float(100.0 * self.area_mm2[self.pool_reach()].sum() / self.target_area_mm2)


_COUNTER_KEYS = ("rejected_clearance", "rejected_tracker", "rejected_joint_limit",
                 "rejected_no_points", "rejected_short", "rejected_flat", "rays",
                 "stroke_positions", "grow_rays", "broken_on_range", "broken_on_incidence",
                 "broken_on_clearance", "broken_on_tracker", "broken_on_joint_limit")


def merge_counts(first: dict[str, int], second: dict[str, int],
                 keys: Sequence[str]) -> dict[str, int]:
    """Add up two counter dictionaries over the named keys."""
    return {key: first.get(key, 0) + second.get(key, 0) for key in keys}


def merge_stroke_coverage(first: StrokeCoverage,
                          second: StrokeCoverage) -> StrokeCoverage:
    """Append the strokes of `second` to `first`, for a refinement round."""
    if first.n_targets != second.n_targets:
        raise ValueError("cannot merge stroke coverage over different target sets")
    margins = [m for m in (first.min_clearance_mm, second.min_clearance_mm)
               if m is not None]
    return StrokeCoverage(
        strokes=first.strokes + second.strokes,
        sets=first.sets + second.sets,
        quality=first.quality + second.quality,
        tracker_ok=np.concatenate([np.asarray(first.tracker_ok, dtype=bool),
                                   np.asarray(second.tracker_ok, dtype=bool)]),
        counters=merge_counts(first.counters, second.counters, _COUNTER_KEYS),
        n_targets=first.n_targets, area_mm2=first.area_mm2,
        target_area_mm2=first.target_area_mm2,
        min_clearance_mm=min(margins) if margins else None,
    )


class StrokeBuilder:
    """Grow strokes from seeds, then evaluate what they cover.

    Kept as a class rather than free functions because every step of the growth needs
    the same five objects (samples, station, model, config, tracker gate), and passing
    them through six call layers is how a growth rule ends up using a stale obstacle
    set.
    """

    def __init__(self, samples: SampleSet, station: Station, model: ScannerModel,
                 cfg: PlannerConfig, *,
                 visible: Callable[[np.ndarray], np.ndarray] | None = None,
                 obstacles: dict | None = None,
                 robot: RobotArm | None = None):
        self.samples = samples
        self.station = station
        self.model = model
        self.cfg = cfg
        self.visible = visible
        self.robot = robot
        self._obstacles = obstacles
        self.points = np.asarray(samples.points, dtype=float).reshape(-1, 3)
        self.normals = np.asarray(samples.normals, dtype=float).reshape(-1, 3)
        self.area = np.asarray(samples.area_mm2, dtype=float)
        self.params = cfg.stroke
        # Why growth stopped, per reason: a band that ends because the surface ran out of
        # measuring range is a different fact from one that ends because the head would hit
        # the fixture, and the report has to be able to tell them apart. `joint_limit` is
        # the same idea one level harder: the band ended because the arm ran out of travel.
        self.broken = dict.fromkeys(("range", "incidence", "clearance", "tracker",
                                     "joint_limit"), 0)
        self.grow_rays = 0

    @property
    def obstacles(self) -> dict:
        """The ray casters, built by the station once and reused."""
        if self._obstacles is None:
            self._obstacles = self.station.obstacles()
        return self._obstacles

    # -- coverage primitives -------------------------------------------------

    def mask_at(self, pose: ScannerPose) -> np.ndarray:
        """Which samples this one pose covers, all six conditions applied."""
        return measure_mask(self.model, pose, self.points, self.normals, self.obstacles,
                            occlusion_tol_mm=self.cfg.occlusion_tol_mm)

    def fits_at(self, positions) -> tuple[bool, float]:
        """Condition 5 over a set of positions: `(all fit, smallest margin in mm)`."""
        positions = np.asarray(positions, dtype=float).reshape(-1, 3)
        clearance = self.station.clearance_mm(positions)
        margin = float(clearance.min() - self.model.body_limit_mm)
        return bool(margin > 0.0), margin

    def tracker_sees(self, positions) -> bool:
        """Condition 6 over a set of positions, or True when no tracker was supplied."""
        if self.visible is None:
            return True
        return bool(np.all(self.visible(np.asarray(positions, dtype=float).reshape(-1, 3))))

    def segment_ok(self, start, end) -> tuple[bool, float]:
        """Legality of one short move: clearance and tracker at every check step.

        This is the TRANSITION criterion - the module docstring of `path` says why: a
        transition is not a scan, so what matters is that the head fits, not what it can
        see. Growth uses `measuring_window` on top of it.
        """
        start = np.asarray(start, dtype=float).reshape(3)
        end = np.asarray(end, dtype=float).reshape(3)
        length = float(np.linalg.norm(end - start))
        count = max(1, int(np.ceil(length / self.params.check_step_mm)))
        fractions = np.linspace(0.0, 1.0, count + 1)[:, None]
        positions = start[None, :] * (1.0 - fractions) + end[None, :] * fractions
        fits, margin = self.fits_at(positions)
        if not fits:
            return False, margin
        return self.tracker_sees(positions), margin

    def measuring_window(self, axis, roll_deg: float, position) -> tuple[bool, str]:
        """Can the head MEASURE from this position at all?

        Growth is a scan, not a transition, so a growth step has to answer the conditions
        a scan answers. `segment_ok` only covers clearance and the tracker; the measuring
        range and the incidence limit were left to the coverage mask, which evaluates them
        per target sample. That is enough to stop a band for the wrong recorded reason, but
        not to keep it legal: the mask says "this pose covers nothing here", which the gain
        rule then reads as "not worth another step", while the truth is "out of range".

        So the gate here is the head's own window: one ray along the optical axis, and the
        first hit must be inside `scanner.range_mm` and inside `max_incidence_deg`. A ray
        that misses the part entirely is NOT a failure - the footprint can still cover
        samples beside the head, and a miss is not evidence of anything.

        Returns `(ok, reason)` with `reason` empty when the window is fine.
        """
        origins = np.asarray(position, dtype=float).reshape(1, 3)
        direction = unit(np.asarray(axis, dtype=float))
        distance, names, triangles = nearest_across(self.obstacles, origins,
                                                    direction.reshape(1, 3))
        self.grow_rays += 1
        hit = float(distance[0])
        if not np.isfinite(hit):
            return True, ""
        low, high = self.model.params.range_mm
        if not low <= hit <= high:
            return False, "range"
        mesh = self.obstacles[str(names[0])].mesh
        normal = np.asarray(mesh.face_normals[int(triangles[0])], dtype=float).reshape(1, 3)
        cosine = float(self.model.incidence_cos(direction.reshape(1, 3), normal)[0])
        if cosine < self.model.cos_incidence_limit:
            return False, "incidence"
        return True, ""

    # -- generation ---------------------------------------------------------

    def seeds(self, points, normals, *, aim_grid_mm: float,
              seed_offset: int = 0) -> list[tuple[ScannerPose, int]]:
        """One head-on pose per aim cell, at the standoff, with its sample index.

        Aiming on a coarse grid rather than at every sample is what keeps the number
        of strokes from growing with the sampling density: one stroke already covers
        thousands of samples. Head-on at the standoff is the pose with the best
        incidence, i.e. the one a straight sweep is most likely to keep.
        """
        from .candidates import aim_indices

        points = np.asarray(points, dtype=float).reshape(-1, 3)
        normals = np.asarray(normals, dtype=float).reshape(-1, 3)
        if len(points) == 0:
            return []
        distance = float(self.cfg.scanner.standoff_mm)
        chosen: list[tuple[ScannerPose, int]] = []
        for index in aim_indices(points, float(aim_grid_mm)):
            normal = unit(normals[index])
            position = points[index] + normal * distance
            axis = -normal
            chosen.append((ScannerPose(position=position, axis=axis,
                                       roll_deg=self.roll_for(axis)),
                           int(index) + int(seed_offset)))
        return chosen

    def roll_for(self, axis) -> float:
        """The roll of a seed, decided here and never re-picked for a coverage point.

        The roll turns the field-of-view rectangle, so it is part of what the stroke
        covers: choosing it after the coverage evaluation would make the reported figure
        describe a sweep the program does not fly. `reference` is the qianduan2 rule that
        keeps tool x on station +X and the exported R near zero; `frame` is the old
        explicit zero, kept so a preset can still reproduce itself.
        """
        if self.params.roll_mode == "reference":
            return roll_for_axis(axis)
        return 0.0

    def grow(self, seed: ScannerPose, seed_index: int) -> list[Stroke]:
        """Grow one seed into strokes: one per tangent line, each extended both ways.

        The four directions of the design are two lines, and a line is grown in both
        directions - growing +t and -t separately and then joining the halves produces
        the same segment at twice the cost.
        """
        params = self.params
        if params.max_stroke_mm <= 0.0:
            return []
        lines = self._tangent_lines(np.asarray(seed.axis, dtype=float))
        strokes: list[Stroke] = []
        for direction_index, direction in enumerate(lines):
            covered = self.mask_at(seed)
            if not covered.any():
                continue
            ends = self._extend(seed, direction, covered)
            if ends is None:
                continue
            low, high, _covered = ends
            stroke = Stroke(start=low, end=high, axis=np.asarray(seed.axis, dtype=float),
                            roll_deg=float(seed.roll_deg), seed=int(seed_index),
                            direction=int(direction_index))
            if stroke.length_mm + _EPS < params.min_stroke_mm:
                continue
            strokes.append(stroke)
        return strokes

    def _tangent_lines(self, axis) -> list[np.ndarray]:
        """The distinct lines a stroke may run along, as unit vectors."""
        first, second = tangents(axis)
        allowed = int(self.params.seed_dirs)
        if allowed <= 0:
            return []
        lines = [first, second]
        return lines[:1] if allowed <= 2 else lines

    def _extend(self, seed: ScannerPose, direction, covered: np.ndarray):
        """Walk away from the seed in both directions for as long as it pays.

        A step is committed only when it is legal AND it bought at least
        `min_gain_mm2_per_step` of new area, so a stroke stops where the surface stops
        facing it rather than at an arbitrary length. Legality is checked every
        `extend_step_mm`, coverage every `coverage_step_mm`: that is the whole point of
        having two steps, and it is what keeps the growth inside the runtime budget.

        Legality here is the SCAN criterion, not the transition one: clearance and tracker
        (`segment_ok`) plus the head's own measuring window - range and incidence - which
        is what stops a band from growing into a region it can no longer measure. A band
        that stops for one of those reasons is counted under its own name, because "the
        surface curves away" and "the head would hit the fixture" ask for different fixes.
        """
        params = self.params
        step = float(params.extend_step_mm)
        origin = np.asarray(seed.position, dtype=float)
        low = origin.copy()
        high = origin.copy()
        # A stroke is one continuous move, so the two growth halves have to share the
        # configuration the arm is in as it crosses the seed. Solving it once and threading
        # it outwards is what makes the two halves one flyable path instead of two paths
        # that happen to meet in the middle on different elbows.
        joints = self._seed_joints(seed)
        for sign in (1.0, -1.0):
            current = origin.copy()
            side_joints = joints
            bearing = float(params.coverage_step_mm)
            while True:
                trial = current + sign * step * np.asarray(direction, dtype=float)
                # the far end does not move while this side grows, so the span is exact
                span = (float(np.linalg.norm(trial - low)) if sign > 0.0
                        else float(np.linalg.norm(high - trial)))
                if span > params.max_stroke_mm + _EPS:
                    break
                legal, _margin = self.segment_ok(current, trial)
                if not legal:
                    self.broken["clearance" if _margin <= 0.0 else "tracker"] += 1
                    break
                measurable, reason = self.measuring_window(seed.axis, seed.roll_deg, trial)
                if not measurable:
                    self.broken[reason] += 1
                    break
                if self.robot is not None:
                    side_joints = self._joint_step(seed, trial, side_joints)
                    if side_joints is None:
                        self.broken["joint_limit"] += 1
                        break
                bearing += step
                if bearing + _EPS >= params.coverage_step_mm:
                    pose = ScannerPose(position=trial,
                                       axis=np.asarray(seed.axis, dtype=float),
                                       roll_deg=float(seed.roll_deg))
                    mask = self.mask_at(pose)
                    gain = float(self.area[mask & ~covered].sum())
                    if gain < params.min_gain_mm2_per_step:
                        break
                    covered |= mask
                    bearing = 0.0
                current = trial
            if sign > 0.0:
                high = current
            else:
                low = current
        return low, high, covered

    def _seed_joints(self, seed: ScannerPose):
        """The configuration at the seed, or None when the arm cannot reach it at all."""
        if self.robot is None:
            return None
        reach = self.robot.solve(seed.frame, seed.position)
        return None if reach is None else reach.joints_deg

    def _joint_step(self, seed: ScannerPose, position, reference):
        """The configuration at `position` continuing `reference`, or None when there is
        none inside the arm's window. Called only when an arm is configured."""
        reach = self.robot.solve(seed.frame, np.asarray(position, dtype=float), reference)
        return None if reach is None else reach.joints_deg

    def generate(self, points, normals, *, aim_grid_mm: float,
                 seed_offset: int = 0,
                 log: Callable[[str], None] | None = None) -> list[Stroke]:
        """Seeds -> grown strokes -> de-duplicated, in that order."""
        say = log or (lambda _message: None)
        seeds = self.seeds(points, normals, aim_grid_mm=aim_grid_mm,
                           seed_offset=seed_offset)
        grown: list[Stroke] = []
        for seed, index in seeds:
            grown.extend(self.grow(seed, index))
        unique = dedupe(grown, self.params.dedupe_mm)
        say(f"  stroke generation: {len(seeds)} seeds -> {len(grown)} grown -> "
            f"{len(unique)} kept (<= {self.params.max_stroke_mm:g} mm each)")
        return unique

    # -- evaluation ---------------------------------------------------------

    def evaluate(self, strokes: Sequence[Stroke], *,
                 log: Callable[[str], None] | None = None) -> StrokeCoverage:
        """Coverage of every stroke, dropping the ones that cannot physically be flown."""
        say = log or (lambda _message: None)
        params = self.params
        counters = dict.fromkeys(_COUNTER_KEYS, 0)
        sets: list[np.ndarray] = []
        quality: list[np.ndarray] = []
        tracker_ok: list[bool] = []
        kept: list[Stroke] = []
        smallest_margin = np.inf

        for stroke in strokes:
            check = stroke.check_positions_mm(params.check_step_mm)
            fits, margin = self.fits_at(check)
            smallest_margin = min(smallest_margin, margin)
            if not fits:
                counters["rejected_clearance"] += 1
                continue
            if not self.tracker_sees(check):
                counters["rejected_tracker"] += 1
                continue
            positions = stroke.positions_mm(params.coverage_step_mm,
                                            params.check_step_mm)
            counters["stroke_positions"] += len(positions)
            covered = np.zeros(len(self.points), dtype=bool)
            best = np.zeros(len(self.points))
            for position in positions:
                pose = ScannerPose(position=np.asarray(position, dtype=float),
                                   axis=np.asarray(stroke.axis, dtype=float),
                                   roll_deg=float(stroke.roll_deg))
                mask = self.mask_at(pose)
                counters["rays"] += int(mask.sum())
                if not mask.any():
                    continue
                covered |= mask
                chosen = np.flatnonzero(mask)
                offset_vector = self.points[chosen] - pose.position
                distance = np.linalg.norm(offset_vector, axis=1)
                cosines = self.model.incidence_cos(
                    offset_vector / np.maximum(distance, _EPS)[:, None], self.normals[chosen])
                best[chosen] = np.maximum(best[chosen], cosines)
            index = np.flatnonzero(covered)
            if len(index) == 0:
                counters["rejected_no_points"] += 1
                continue
            kept.append(stroke)
            sets.append(index.astype(np.int64))
            quality.append(best[index])
            tracker_ok.append(True)

        counters.update({f"broken_on_{key}": int(value) for key, value in self.broken.items()})
        counters["grow_rays"] = int(self.grow_rays)
        say(f"  evaluated {len(strokes)} strokes: "
            f"{ {key: counters[key] for key in _COUNTER_KEYS} }")
        return StrokeCoverage(
            strokes=kept, sets=sets, quality=quality,
            tracker_ok=np.asarray(tracker_ok, dtype=bool), counters=counters,
            n_targets=len(self.points), area_mm2=self.area,
            target_area_mm2=float(self.samples.total_area_mm2),
            min_clearance_mm=float(smallest_margin) if np.isfinite(smallest_margin) else None,
        )


def dedupe(strokes: Sequence[Stroke], tol_mm: float) -> list[Stroke]:
    """Keep one stroke per (line, pose), the longest, so the pool has no twins.

    Two strokes count as the same when their poses agree to a hundredth of a degree
    and one millimetre, their directions are parallel, one start lies on the other's
    line, and their spans overlap. Keeping the longer of the two is the point: the
    shorter one adds coverage the longer one already has.
    """
    if not strokes:
        return []
    ordered = sorted(strokes, key=lambda stroke: -stroke.length_mm)
    kept: list[Stroke] = []
    for candidate in ordered:
        if any(_same_sweep(candidate, other, tol_mm) for other in kept):
            continue
        kept.append(candidate)
    return kept


def _same_sweep(first: Stroke, second: Stroke, tol_mm: float) -> bool:
    """Is `first` the same physical sweep as `second`, to within `tol_mm`?"""
    if abs(first.roll_deg - second.roll_deg) > 0.01:
        return False
    if abs(_cos(first.axis, second.axis) - 1.0) > 1e-6:
        return False
    first_span = first.length_mm
    second_span = second.length_mm
    if first_span <= 0.0 or second_span <= 0.0:
        return (first_span <= 0.0 and second_span <= 0.0
                and float(np.linalg.norm(np.asarray(first.start, dtype=float)
                                         - np.asarray(second.start, dtype=float))) <= tol_mm)
    direction = first.unit_direction
    origin = np.asarray(first.start, dtype=float)
    for point in (second.start, second.end):
        offset = np.asarray(point, dtype=float) - origin
        across = float(np.linalg.norm(offset - np.dot(offset, direction) * direction))
        if across > tol_mm:
            return False
    low, high = _interval(first, direction, origin)
    other_low, other_high = _interval(second, direction, origin)
    overlap = min(high, other_high) - max(low, other_low)
    return overlap > -tol_mm


def _interval(stroke: Stroke, direction, origin) -> tuple[float, float]:
    """The stroke's span projected on `direction`, measured from `origin`."""
    first = float(np.dot(np.asarray(stroke.start, dtype=float) - origin, direction))
    second = float(np.dot(np.asarray(stroke.end, dtype=float) - origin, direction))
    return (first, second) if first <= second else (second, first)


def _cos(first, second) -> float:
    return float(np.dot(unit(first), unit(second)))


def combine(pool: CoverageResult, strokes_cov: StrokeCoverage) -> CoverageResult:
    """One coverage object holding the pose pool and the strokes, strokes last.

    The selection stage indexes rows, so the two candidate kinds only have to share a
    matrix. Keeping them in ONE object is what lets the report, the leak attribution
    and the fallback comparison say "the pool could reach X%" without knowing which
    half a row came from.
    """
    if pool.n_targets != strokes_cov.n_targets:
        raise ValueError("cannot combine coverage over different target sets")
    counters = merge_counts(pool.counters, strokes_cov.counters, _COUNTER_KEYS)
    return CoverageResult(
        poses=pool.poses + strokes_cov.poses,
        sets=pool.sets + strokes_cov.sets,
        quality=pool.quality + strokes_cov.quality,
        tracker_ok=np.concatenate([np.asarray(pool.tracker_ok, dtype=bool),
                                   np.asarray(strokes_cov.tracker_ok, dtype=bool)]),
        counters=counters, n_targets=pool.n_targets, area_mm2=pool.area_mm2,
        target_area_mm2=pool.target_area_mm2,
    )
