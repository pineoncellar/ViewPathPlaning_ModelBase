# Scanner Viewpoint Planning for a FANUC Scanning Cell

Automatic generation of **scanner viewpoints** for a stamped sheet-metal part on
a FANUC robot carrying a **TrackScan-Sharp 49** optical scanner.

Given the station CAD - part, fixture, base and the optical tracker - plus the
scanner parameters, the planner produces a set of scanner poses (position,
optical axis, roll) that together image the required face of the part, and exports
them as a FANUC `.LS` program so the workshop can check reachability in ROBOGUIDE.

> **Status: early development.**
> The package layout, the interfaces and the data contracts of the export are in
> place; the planning algorithms are **not implemented yet**. Every module under
> [`src/scanplan/`](src/scanplan) currently carries a docstring stating its
> purpose and the constraints that turned out to be easy to get wrong. The design
> notes behind those constraints are kept privately, because they quote customer
> part dimensions.

## What it does

1. **Set up the station** - load the part, fixture, base and tracker and put them
   into one CAD frame, with a self-checked alignment rather than trusting the
   files.
2. **Split the target surface** topologically into the two faces of the original
   flat blank (see below) and sample the one that is to be measured.
3. **Generate candidate scanner poses** around that surface and score each one for
   field of view, range, incidence angle, occlusion, the scanner body fitting and
   tracker visibility.
4. **Choose the smallest covering set** of poses.
5. **Export** the viewpoint set, a coverage report, and a FANUC `.LS` program with
   its frame contract.

## What it deliberately does not do

Inverse kinematics, joint limits, robot collision, viewpoint ordering and motion
time optimisation. The shipped `.LS` carries **Cartesian poses only**; the
controller / ROBOGUIDE solves them. That is why no robot model is needed here, and
why the deliverable is a viewpoint set rather than a robot program.

## The target surface: a topological definition

The workpiece is a stamped sheet, so abstractly it has exactly **two** faces - the
two sides of the flat blank it was pressed from. The forming side-walls belong to
whichever face they were bent out of. The sheared (blanked / pierced) edge belongs
to neither, because it is a cut through the material rather than a surface of the
blank.

```
target   = one of the two faces of the blank
           forming side-walls and pocket walls INCLUDED
minus    = the region the fixture clamps   (not required to be measured)
excluded = the sheared (blanked / pierced) edge
excluded = the other face of the blank
```

Orientation-based rules all fail here, each for its own reason: `normal_z > 0`
breaks as soon as the part is tilted in its fixture; a dihedral-angle patch is not
a face, because creases split one face into several flat facets; a threshold
against the part's own through-thickness axis discards the forming side-walls,
which do belong to the face; and dropping the sheared edge and then taking
connected components fails outright on a real CAD tessellation, whose surfaces
were meshed separately and never had their coincident vertices merged.

[`src/scanplan/sheet_split.py`](src/scanplan/sheet_split.py) documents the method
that does work.

## Repository layout

| Path | Contents | Versioned |
|---|---|---|
| [`src/scanplan/`](src/scanplan) | The planner. Each module docstring states its purpose and the pitfalls. | yes |
| [`configs/`](configs) | Configuration, including the scanner parameters. | yes |
| [`tests/`](tests) | Tests. All fixtures are synthetic - no customer geometry. | yes |
| [`scripts/`](scripts) | Repository tooling, including the data-hygiene check. | yes |
| `input/` | **Station CAD models go here.** Contents are gitignored. | **no** |
| `output/` | Planner results. Contents are gitignored. | **no** |
| [`reference/`](reference) | The frozen original pipeline, kept as a baseline. Its manual is [`reference/README.md`](reference/README.md). | yes |
| `docs/` | Private design notes - they quote customer dimensions. | **no** |
| `samples/` | Public sample meshes and figures from the original project. | yes |

## Getting started

Requires Python 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev --extra embree
uv run pytest
```

`embree` is optional but strongly recommended: it accelerates ray casting by
roughly three orders of magnitude, which is the difference between a real part
being practical to plan and not.

There is one entry point:

```bash
# intended interface - not implemented yet
uv run scanplan plan --part P.stl --fixture F.stl --base B.stl \
                     --config configs/scanner_trackscan49.json \
                     --out output/<part>/

uv run scanplan ls   --viewpoints output/<part>/viewpoints.json \
                     --prog-name PART01 --out output/<part>/
```

`plan` and the `.LS` export are separate on purpose: the ROBOGUIDE frame contract
(which `UFRAME` to use, and where its origin sits) is agreed with the workshop and
may need re-exporting without re-planning.

### Station models

Put the CAD of the part, the fixture and the base in `input/`. Keep it out of
version control - it is customer geometry. If the models arrive as IGES exported
one file at a time, the assembly placement is lost, which is why station setup
includes an explicit, self-checked alignment step instead of trusting the files.

## Configuration

[`configs/scanner_trackscan49.json`](configs/scanner_trackscan49.json) holds the
scanner parameters. **The values are placeholders**: they come from the public
datasheet and from the geometric meaning of the words "standoff" and "depth of
field", not from measurements on the real device. The most sensitive one by far is
`max_incidence_deg` - it moves the achievable coverage more than any other number,
and it is not yet known.

## Data privacy

The station CAD, anything derived from it (meshes, point clouds, viewpoint
coordinates) and the design notes must never be committed. `input/`, `output/`,
`docs/`, `spike/` and `reference/00_Output Collection/` are all gitignored, and a
check fails if any of it ever becomes tracked:

```bash
uv run python scripts/check_repo_hygiene.py
```

Run it before committing. An ignored file can still be added later with
`git add -f`, but a pushed one cannot be un-pushed.

## Validation so far

A feasibility study was run during design, on a real workshop part and with
placeholder scanner parameters. It is the source of the constraints recorded in
the module docstrings. Two findings shaped the design:

- a fixed tracker at the station does not by itself limit coverage, provided it
  looks **along the part's thin axis** rather than square-on to the large faces -
  the mechanism is occlusion of the scanner body by the part, not the tracker's
  field of view;
- the residual misses were a couple of dozen **small** features (narrow slots,
  small pocket walls) rather than one large unreachable region, which points at
  candidate density as the next gain rather than a better covering algorithm.

Those figures depend on the placeholder `max_incidence_deg`, so they are
indicative only and are deliberately not quoted here.

## Reference pipeline

[`reference/`](reference) contains the original pipeline, frozen: `PHASE_1` ..
`PHASE_6`, the set-cover solvers and the IGES to mesh converter. It is kept as a
regression baseline, and as the source of the FANUC `.LS` writer that the new
planner ports rather than rewrites - that writer already solved the parts of the
FANUC ASCII format that are tedious and unforgiving. See
[`reference/README.md`](reference/README.md).

## Acknowledgements & prior work

This project was originally inspired by, and its core algorithmic workflow is
based on, the open-source project **ViewPathPlanning** - a model-viewpoint and
robot-path planning pipeline for a UR5e arm with a 3D scanner.

Thanks to the original author for the prototype and the exploration built during
their master's research:

- **Original repository**: [MobMonRob/ViewPathPlaning](https://github.com/MobMonRob/ViewPathPlaning)
- **License**: [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
- **Algorithmic prototype**: the original project established the complete
  Phase 1 to Phase 5 pipeline - voxel-based viewpoint generation, viewpoint set
  cover (SCP), robot base reachability with inverse kinematics, base set cover,
  and TSP path planning.

**How this repository differs.** It is no longer a fork but an independent
project. Keeping the core phase idea, the architecture, the solving chain and the
delivery format have been substantially rewritten:

| Aspect | Original project | This repository |
|---|---|---|
| Architecture | Five sequential scripts handing over JSON files | A decoupled Python package (`src/` layout); modules testable in isolation |
| Sensor | Fixed profile scanner on the arm, no external tracker | **Tracked** scanner (external optical tracker); adds a tracker-visibility constraint |
| Target surface | Every outward-facing surface of the part | **Topological**: one complete face of the original flat blank, forming side-walls included, sheared edge excluded |
| Solving chain | UR5e inverse kinematics, joint limits, base sampling, TSP | **Removed**; delivers the scanner viewpoint set only, reachability is checked in ROBOGUIDE |
| Output | Robot position coordinates (voxel units) and joint configurations as JSON | Scanner Cartesian poses, a FANUC `.LS` program, a frame contract and a coverage report |
| Ray casting | Pure Python | Embree acceleration (about three orders of magnitude) |
| Data isolation | - | Customer CAD and anything derived from it are never committed; a check enforces it |
| Original pipeline | The live implementation in the repository root | Frozen under `reference/` as a regression baseline |

*This project was originally inspired by and built upon the workflow from
**[MobMonRob/ViewPathPlaning](https://github.com/MobMonRob/ViewPathPlaning)**,
licensed under Apache-2.0.*

*While the fundamental 5-phase view-path-planning concept stems from the original
master's thesis implementation, this repository has been comprehensively
refactored and redesigned, and is maintained as an independent codebase.*

## License

Apache License 2.0 - see [LICENSE](LICENSE).
