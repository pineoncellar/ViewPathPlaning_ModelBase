# Model based ViewPathPlanning
This repository contains code to fully generate viewpoints and robot positions to completely scan the surface of a given object. The implementation uses a 6 DOF robotic arm (UR5e) mounted with a 3D Scanner (micro epsilon suface control 3D 3510-80). The aproach is divided into 5 phases, which result in a list of robot position coordinates (in voxel unit) and corresponding joint configurations, both including a sequence indicator. Output file format is JSON.


> **This is the manual of the frozen reference pipeline.**
>
> `PHASE_1`..`PHASE_6`, `SCP_src.py` and `IGES_to_STL.py` alongside this file
> are the original upstream pipeline. They are kept as a regression baseline
> and as the source of the FANUC `.LS` writer that the new planner vendors.
>
> The current project lives in [`src/scanplan/`](../src/scanplan) - see the
> [root README](../README.md).

## PHASE 1
### Description
The first phase resulst in a list of viewpoint (VP) candidates and their corresponing visible object surface (represented as voxels). The phase includes the following steps:
- Initialization: load object model (e.g. mesh)
- Step 1: Voxelize object model as ocupancy matrix. Extend the matrix as preparation for the dilation step.
- Step 2: Generate viepoint sample space via binary dilation. Select random viewpoint candidates from the sample space. (10% recommended)
- Step 3: Calculate object voxels that are within the "potential field" of the viewpoints. Use the voxels of the potential field to calculate mean viewing direction of the viepoints
- Step 4: Calculate object voxels that are within the measurement area  (MA) of the viewpoints. Only obj voxels in the potential field are used per VP in terms of dimensionality reduction.
- Step 5: Ray tracing for ocllusion culling per object voxels in MA per VP. Only object voxels that are not occluded form the viepoits perspective are included to coount as "visible surface"
### Input and configuration
- gridSize (this is the voxel size used)
- obj_mesh_scale (this is for scaling the object model before processing)
- number_of_vp_candidates_seed (if 1 or smaller than 1, it is used as percentage, else it is used as absolute number)
- sampleSpace_extension (this controls how far the smaple sae will be extended bejond the used max sensor range)
- ma_scale (scale the actual measurement area of the scanner, scaling smaller means there will be more overlap of the actual measurement areas of the VPs)
- sensor attributes (scanner dimensions, measurment areas, measurement ranges)
- obj_mesh_path (path to object model file)
- outputPath (located at the end of the code, there the output will be saved)

## PHASE 2
### Description
The second phase uses the generated VP candidates and their corresponding visible object voxels as set covering problem (SCP). The least amount of nessesary VPs need to be found that still cover the whole object surface. There are implemented the following algorithms:
- Greedy algorithm
- Naive algorithm
- Simulated annealing 
- Particle swarm optimization (with and without greedy input)
- Genetic algorithm (with and without greedy input)
- Chemical reaction optimization (with and without greedy input)

### Input and configuration
- inputPath (path to a phase 1 output file)
- algorithms (need to be commented out, if they should not be applied. Algorithm Paramters are set in file SCP_src)
- outputPath (located at the end of the code, there the output will be saved)


## PHASE 3
### Description
The third phase results in a list of robot position candidates and their corresponding reachable VPs. The phase includes the following steps:
- Step 1: Generate robot position sample space via binary dilation. Select random robot position candidates from the sample space. (10% recommended)
- Step 2: Get robot positioning orientation. The robot should always be oriented pointing away from the object with its optical axis. Thus the scanner is alway viewing away from the object in start position. This is for accessiblity purposes of the scanner while setting up the new position and it prevents invalid positions e.g. with the robots base beside the object but its rotated in a manner that the arm is under the object.
- Step 3: Calculate VP voxels that are within the "potential field" of the robots position. This is bassed on the working radius of the robot. As there could be VPs oriented in a manner that they are nevertheless reachable, this set is updated in the following step.
- Step 4: Get the robots joint configurations to reach the assigned VPs. Only VPs in the potential field are used in terms of dimensionality reduction. VPs that are not reachable will be deleted form the assigned VPS of the position. The inverse kinematics always result in 8 configurations possible. A selection of the most suitable configuration is not addressed. There will be always taken the first out of the 8 configurations.

### Input and configuration
- inputPath (path to a phase 2 output file)
- tm_rb_minRange (controls the placement of the robet, minimum distance to the object)
- rbPos_quantity_seed (controls how many candidates are taken from the sample space, if 1 or smaller than 1, it is used as percentage, else it is used as absolute number)
- vps_toUse (as there can be used multiple algorithms for the SCP of the VPs, it has to be decided with which SCP solution of VPs to work in this phase)
- robtots denavit hartenberg paramters
- outputPath (located at the end of the code, there the output will be saved)

## PHASE 4
### Description
The fourth phase uses the generated robot position candidates and their corresponding reachable VPs as SCP. The least amount of nessesary robot positions need to be found that still cover all VPs. The implemented algorithms to solve the SCP are listed in phase 2.

### Input and configuration
- inputPath (path to a phase 3 output file)
- vps_toUse (as there can be used multiple algorithms for the SCP of the VPs, it has to be decided with which SCP solution of VPs to work in this phase)
- algorithms (need to be commented out, if they should not be applied. Algorithm Paramters are set in file SCP_src)
- outputPath (located at the end of the code, there the output will be saved)

## PHASE 5
### Description
The last phase determines the minimum cost sequences of the given robot positions. Moreover the assigned VPs per robot position are also put in a sequence. For the robot positions the distance between the coordinates is considered as cost. For the VPs the summed joint configuration change to get from one VP to the next is considerd as cost. The phase includes the following steps:
- Step 1: Assign overlapping VPs to the closest robot position. If multiple robot posiiotns cover the same VP, th VPs will only be assinged to the closest robot position. One VP must not be visited multiple times.
- Step 2: Get the robot position sequence via travelling salesman problem (TSP)
- Step 3: Get the VP sequence of each robot positioin via TSP

### Input and configuration
- inputPath (path to a phase 3 output file)
- vps_toUse (as there can be used multiple algorithms for the SCP of the VPs, it has to be decided with which SCP solution of VPs to work in this phase)
- rbPos_toUse (as there can be used multiple algorithms for the SCP of the robot positions, it has to be decided with which SCP solution of robot positions to work in this phase)
- outputPath (located at the end of the code, there the output will be saved)

## PHASE 6 (export adapter)
### Description
Optional output adapter for ROBOGUIDE (FANUC) feasibility checks. It does **not** run inverse kinematics and does not modify PHASES 1..5: it reads a viewpoint set (`VPs_greedy`) from a PHASE 2/3/4/5 output file and writes the viewpoints as Cartesian positions, so that reachability, joint limits and collision checks are left to the FANUC controller inside ROBOGUIDE. The UR5e joint angles are discarded, therefore PHASES 3..5 are not needed for this route.

Three files are written to `reference/00_Output Collection`, all named after the program: the FANUC ASCII program (`<prog-name>.LS`), the frame contract plus self-check numbers (`<prog-name>_frame.json`) and a flat table (`<prog-name>_points.csv`).

The `.LS` is named after the program, and the program name inside the file is the same string, because FANUC refuses an ASCII upload whose file name is not a valid program name - a file name containing a space fails with `Invalid name in /PROG section`. `--prog-name` (default `PH6SCAN`) therefore accepts only a letter followed by up to 15 characters from `A-Z`, `0-9` and `_`, and the program is written as `<prog-name>.LS`. Do not rename the file to anything with a space in it.

Two contracts must be respected on the ROBOGUIDE side (both are repeated inside `PH6_frame*.json`):
- **Frame**: origin at the object voxel-bounding-box minimum corner, axes identical to the algorithm world axes (right handed, Z up), unit millimetres, `p_mm = (voxel_index - pad_voxels) * gridsize_mm`. The UFRAME is therefore a pure translation. The file also reports how far that origin sits above the algorithm floor plane (`z = 0`), in case a floor referenced origin is preferred. Passing the CAD coordinates of that corner via `--origin-mm` shifts the whole export, so the emitted numbers are true CAD coordinates.
- **TCP**: the exported pose is the algorithm VP pose, i.e. a point `s_size_length` (150 mm for the modelled sensor) behind the sensor front face on the optical axis, with local +Z pointing from the sensor towards the part. The UTOOL must be defined to match that point.

The arm configuration (`CONFIG`) cannot be derived from a Cartesian pose, so a default (`'N U T, 0, 0, 0'`) is written for every point. Points with multiple solutions may have to be re-solved in ROBOGUIDE.

### Input and configuration
- `--input` (PHASE 2/3/4/5 output file, default: newest `PH2 output *.json`)
- `--vps-key` (viewpoint set to export, default `VPs_greedy`)
- `--origin-mm` (`x,y,z` in CAD coordinates of the part bounding-box minimum corner, default `0,0,0`)
- `--place` (`x,y,z,w,p,r`: pose of the CAD frame inside ROBOGUIDE; emits world coordinates and uses UFRAME 0)
- `--uframe` (default `1`, or `0` when `--place` is given)
- `--rot-weight` (millimetres per degree, adds orientation change to the ordering cost, default 0)
- `--selftest` (validate the W/P/R conversion and number formatting, then exit)

```bash
uv run reference/PHASE_6_Export_Roboguide.py --input "reference/00_Output Collection/PH2 output <timestamp>.json"
```

# Preparing the station models (IGES -> mesh)

PHASE_1 voxelises *triangle meshes* through trimesh, which cannot read IGES (nor native NX `.prt`). `IGES_to_STL.py` meshes IGES assemblies through OpenCASCADE and writes binary STL, decimating large results on request. It also prints the IGES Global section (original file name, producer, units), because the file names handed over are not necessarily the names the models were exported with - the printed internal name and the per-model geometry summary are what identify them.

The default `--backend ocp` talks to OpenCASCADE directly (via `cadquery-ocp`) and meshes the imported shape **without healing it**, which is what makes it work on these exports: `--backend gmsh` attempts to weld the faces into a valid shell first and aborts with `Could not fix wire in surface N`, because a PolyWorks assembly export has thousands of trimmed surfaces with slightly inconsistent wires. Meshing needs no valid shell, so faces that still fail are counted and skipped rather than aborting the conversion. `auto` tries ocp first and falls back to gmsh.

It runs in its own environment and deliberately leaves the pinned pipeline environment alone:

```bash
uv run --no-project --python 3.11 \
    --with cadquery-ocp --with gmsh --with trimesh \
    --with fast-simplification --with numpy --with matplotlib \
    python reference/IGES_to_STL.py --input-dir <folder with .igs> \
    --preview --report <folder with .igs>/mesh/report.json
```

Meshes are written to `<input-dir>/mesh` unless `--out-dir` is given; `--keep-ratio` (default `0.1`) sets how many faces survive decimation and `--decimate-above` (default `200000`) the face count above which a mesh is decimated at all. `--deflection-mm` (default `1.0`) controls the chordal tolerance of the ocp backend. Coordinates are passed through unchanged, so an IGES in millimetres yields an STL in millimetres. `report.json` records, per model, the IGES header, the bounding box in the shared CAD frame, the triangle counts before and after decimation, and how many faces meshed.

To make the algorithm world frame *identical* to the CAD frame, neutralise the transform PHASE_1 applies to the object mesh: set `obj_mesh_scale` such that the scaled mesh is in metres (`0.001` for a millimetre mesh, because `tm_voxelSize = 0.01` is what makes one voxel 10 mm) and set the rotation in the object-loading block of `reference/PHASE_1_Get_Can_VPs.py` to identity. PHASE_1 and PHASE_2 do not require the model to be Z-up; only the robot-position sampling of PHASE_3 does, and PHASE_3 is not used on the ROBOGUIDE export route.

# Python environment (uv)

The pipeline was developed against a frozen 2022 package set (`numpy 1.23.1`, `scipy 1.8.1`, `trimesh 3.12.7`, `mujoco 2.2.0`, `ortools 9.4.1874`, ...). Those releases only ship wheels up to Python 3.10, therefore `.python-version` pins `3.10` and `requires-python` is restricted to `>=3.10,<3.11`.

```bash
uv sync                                # create .venv and install the locked versions
uv run reference/PHASE_1_Get_Can_VPs.py     # run a phase inside the environment
```

`uv sync` downloads Python 3.10 automatically if it is not installed yet.

Every phase reads its input from and writes its output to the `reference/00_Output Collection` folder of the working directory, so run the scripts from the repository root. The input file names are hard-coded at the top of each phase: point them at the output file of the previous phase (`PHASE_2`, for example, reads the `PH1 output <timestamp>.json` written by `PHASE_1`).

# Python packages
| Package | Version 
| :-------- | ------|
| absl-py         | 1.1.0 |
| cycler          | 0.11.0 |
| fonttools       | 4.37.1 |
| glfw            | 2.5.3 |
| joblib          | 1.1.0 |
| kiwisolver      | 1.4.4 |
| matplotlib      | 3.5.3 |
| mujoco          | 2.2.0 |
| numpy           | 1.23.1 |
| ortools         | 9.4.1874 |
| packaging       | 21.3 |
| Pillow          | 9.2.0 |
| pip             | 22.1.2 |
| protobuf        | 4.21.5 |
| pyglet          | 1.5.26 |
| PyOpenGL        | 3.1.6 |
| pyparsing       | 3.0.9 |
| python-dateutil | 2.8.2 |
| Rtree           | 1.0.0 |
| scikit-learn    | 1.1.2 |
| scipy           | 1.8.1 |
| setuptools      | 58.1.0 |
| Shapely         | 1.8.2 |
| six             | 1.16.0 |
| threadpoolctl   | 3.1.0 |
| trimesh         | 3.12.7 |
# Simulation Environment (mujoco version 2.2.0)
(1) The mujoco stand alone software is available here: https://mujoco.org/ <br />
(2) The mujoco package for python is available here: https://github.com/deepmind/mujoco/releases <br />
To view the sample simulation environment (folder "samples/sim env/00_sample simulation env.xml") download mujoco stand alone (reference (1)) and open the folder "bin/simulate.exe". There you can drag and drop the XML-file to start the simuation. <br />
To do further simulation steps use the python package (reference (2)). The documentation how to use the package can be found here: https://mujoco.readthedocs.io/en/latest/python.html <br />
However a sample python file, that uses the Phase 5 outputfile to move the robot can be found in the "samples/sim env" folder.
