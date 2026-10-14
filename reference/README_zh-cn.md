# 基于模型的视图路径规划（Model based ViewPathPlanning）

本仓库包含用于完整生成视点（Viewpoint, VP）与机器人位置（Robot Position）的代码，以完整扫描给定物体的表面。该实现使用一台 6 自由度机械臂（UR5e），其上搭载一台 3D 扫描仪（micro epsilon surface control 3D 3510-80）。整个方法分为 5 个阶段，最终得到的输出是一组机器人位置坐标（以体素为单位）及对应的关节构型，两者均带有顺序标识。输出文件格式为 JSON。

> **本文件是已冻结的原管线手册。**
>
> 与本文件同目录的 `PHASE_1`～`PHASE_6`、`SCP_src.py`、`IGES_to_STL.py` 是原项目的
> 流水线。它们作为回归基线保留，同时也是新规划器**移植**（而非重写）FANUC `.LS`
> 写入器的来源。
>
> 当前项目在 [`src/scanplan/`](../src/scanplan)，见[根目录中文 README](../README_zh-cn.md)。



## 阶段 1（PHASE 1）
### 描述
第一阶段生成视点（VP）候选列表及其对应的可见物体表面（以体素表示）。该阶段包含以下步骤：
- 初始化：加载物体模型（例如网格 mesh）
- 步骤 1：将物体模型体素化为占据矩阵（occupancy matrix）。扩展该矩阵，作为膨胀（dilation）步骤的准备工作。
- 步骤 2：通过二值膨胀生成视点采样空间。从采样空间中随机选择视点候选。（推荐选取 10%）
- 步骤 3：计算位于视点"势场"（potential field）内的物体体素。使用该势场内的体素计算视点的平均观察方向。
- 步骤 4：计算位于视点测量区域（Measurement Area, MA）内的物体体素。为进行降维，每个 VP 只使用其势场内的物体体素。
- 步骤 5：对每个 VP 的 MA 内每个物体体素进行光线投射（ray tracing），以进行遮挡剔除。只有从该视点视角不被遮挡的物体体素才计入"可见表面"。

### 输入与配置
- gridSize（所使用的体素尺寸）
- obj_mesh_scale（处理前对物体模型进行缩放）
- number_of_vp_candidates_seed（若小于等于 1，则作为百分比使用；否则作为绝对数量使用）
- sampleSpace_extension（控制采样空间在最大传感器范围之外扩展的程度）
- ma_scale（缩放扫描仪的实际测量区域；缩放得越小，各 VP 实际测量区域之间的重叠就越多）
- 传感器属性（扫描仪尺寸、测量区域、测量范围）
- obj_mesh_path（物体模型文件的路径）
- outputPath（位于代码末尾，输出将保存至此处）

## 阶段 2（PHASE 2）
### 描述
第二阶段将生成的 VP 候选及其对应的可见物体体素作为集合覆盖问题（Set Covering Problem, SCP）来处理。需要找出能够仍然覆盖整个物体表面的最少必要 VP 数量。实现了以下算法：
- 贪心算法（Greedy）
- 朴素算法（Naive）
- 模拟退火（Simulated Annealing）
- 粒子群优化（Particle Swarm Optimization，含贪心输入与不含贪心输入两种）
- 遗传算法（Genetic Algorithm，含贪心输入与不含贪心输入两种）
- 化学反应优化（Chemical Reaction Optimization，含贪心输入与不含贪心输入两种）

### 输入与配置
- inputPath（阶段 1 输出文件的路径）
- algorithms（若不想应用某算法，需将其注释掉。算法参数在文件 `SCP_src` 中设置）
- outputPath（位于代码末尾，输出将保存至此处）

## 阶段 3（PHASE 3）
### 描述
第三阶段生成机器人位置候选列表及其对应的可达 VP。该阶段包含以下步骤：
- 步骤 1：通过二值膨胀生成机器人位置采样空间。从采样空间中随机选择机器人位置候选。（推荐选取 10%）
- 步骤 2：确定机器人定位姿态。机器人应始终以其光轴指向背离物体的方向。这样在起始位置扫描仪始终朝远离物体的方向观察。这出于扫描仪的可达性考虑，便于在设定新位置时进行布置，并且可避免出现无效位置，例如机器人基座在物体旁边、但其旋转方式导致机械臂位于物体下方。
- 步骤 3：计算位于机器人位置"势场"内的 VP 体素。该势场基于机器人的工作半径。由于可能存在朝向合适、因而仍然可达的 VP，该集合会在下一步中更新。
- 步骤 4：计算机器人到达所分配 VP 的关节构型。为进行降维，只使用势场内的 VP。不可达的 VP 将从该位置所分配的 VP 中删除。逆运动学（inverse kinematics）通常会产生 8 种可能的构型。此处不处理最优构型的选择，始终取 8 种构型中的第一种。

### 输入与配置
- inputPath（阶段 2 输出文件的路径）
- tm_rb_minRange（控制机器人的摆放，即到物体的最小距离）
- rbPos_quantity_seed（控制从采样空间中取多少个候选；若小于等于 1，则作为百分比使用；否则作为绝对数量使用）
- vps_toUse（由于 VP 的 SCP 可使用多种算法，需决定本阶段采用哪个 VP 的 SCP 解）
- 机器人的 Denavit-Hartenberg 参数
- outputPath（位于代码末尾，输出将保存至此处）

## 阶段 4（PHASE 4）
### 描述
第四阶段将生成的机器人位置候选及其对应的可达 VP 作为 SCP 处理。需要找出能够仍然覆盖所有 VP 的最少必要机器人位置数量。所实现的 SCP 求解算法已在阶段 2 中列出。

### 输入与配置
- inputPath（阶段 3 输出文件的路径）
- vps_toUse（由于 VP 的 SCP 可使用多种算法，需决定本阶段采用哪个 VP 的 SCP 解）
- algorithms（若不想应用某算法，需将其注释掉。算法参数在文件 `SCP_src` 中设置）
- outputPath（位于代码末尾，输出将保存至此处）

## 阶段 5（PHASE 5）
### 描述
最后一个阶段确定给定机器人位置的最小成本序列。此外，每个机器人位置所分配的 VP 也进行排序。对于机器人位置，将坐标之间的距离视为成本。对于 VP，则将从一个 VP 移动到下一个 VP 所需关节构型变化的总和视为成本。该阶段包含以下步骤：
- 步骤 1：将重叠的 VP 分配给最近的机器人位置。若多个机器人位置覆盖同一个 VP，则该 VP 只会分配给最近的机器人位置。一个 VP 不得被访问多次。
- 步骤 2：通过旅行商问题（TSP）求解机器人位置序列
- 步骤 3：通过 TSP 求解每个机器人位置的 VP 序列

### 输入与配置
- inputPath（阶段 3 输出文件的路径）
- vps_toUse（由于 VP 的 SCP 可使用多种算法，需决定本阶段采用哪个 VP 的 SCP 解）
- rbPos_toUse（由于机器人位置的 SCP 可使用多种算法，需决定本阶段采用哪个机器人位置的 SCP 解）
- outputPath（位于代码末尾，输出将保存至此处）

## 阶段 6（PHASE 6，导出适配器）
### 描述
可选的 ROBOGUIDE（FANUC）可行性验证导出层。它**不做逆解**，也**不修改阶段 1～5**：读取阶段 2/3/4/5 输出文件里的视点集合（`VPs_greedy`），把视点写成笛卡尔位姿，可达性、关节限位与碰撞检查交由 ROBOGUIDE 内的 FANUC 控制器判定。UR5e 的关节角被丢弃，因此这条路线不需要跑阶段 3～5。

会在 `reference/00_Output Collection` 下写出三个文件，都以程序名命名：FANUC ASCII 程序（`<程序名>.LS`）、帧契约与自检数据（`<程序名>_frame.json`）、以及便于核对的表格（`<程序名>_points.csv`）。

`.LS` 以程序名命名，且文件内部的程序名与它完全一致——因为 FANUC 会拒绝文件名的**不是合法程序名**的 ASCII 上传：例如文件名里带空格，就会报 `Invalid name in /PROG section`。因此 `--prog-name`（默认 `PH6SCAN`）只接受"字母开头 + 最多 15 位 `A-Z`/`0-9`/`_`"，程序就写成 `<程序名>.LS`。**不要把文件重命名成带空格的名字。**

ROBOGUIDE 侧必须遵守两条契约（`PH6_frame*.json` 里也各写了一份）：
- **坐标系**：原点取零件体素包围盒的最小角点，坐标轴与算法世界轴一致（右手系、Z 朝上），单位为 mm，`p_mm = (voxel_index - pad_voxels) * gridsize_mm`。因此 UFRAME 是纯平移。文件里同时给出该原点位于算法地面平面（`z = 0`）上方多高，便于改用地面基准的原点。用 `--origin-mm` 传入该角点在 CAD 里的坐标，可以把整份导出平移到真正的 CAD 坐标。
- **TCP**：导出的位姿即算法里的视点位姿，也就是位于传感器前端面沿光轴向后 `s_size_length`（本模型为 150 mm）处的那个点，局部 +Z 由传感器指向零件。UTOOL 必须按这个点定义。

机械臂构型（`CONFIG`）无法从笛卡尔位姿反推，因此每个点位都写入默认值 `'N U T, 0, 0, 0'`。存在多解的点位可能需要在 ROBOGUIDE 里重新求解。

### 输入与配置
- `--input`（阶段 2/3/4/5 输出文件，默认取最新的 `PH2 output *.json`）
- `--vps-key`（要导出的视点集合，默认 `VPs_greedy`）
- `--origin-mm`（零件包围盒最小角点在 CAD 里的 `x,y,z` 坐标，默认 `0,0,0`）
- `--place`（CAD 坐标系在 ROBOGUIDE 里的位姿 `x,y,z,w,p,r`；输出世界坐标并使用 UFRAME 0）
- `--uframe`（默认 `1`；使用 `--place` 时为 `0`）
- `--utool`、`--prog-name`（FANUC 程序名，也是 `.LS` 文件名；字母开头 + 最多 15 位 `A-Z`/`0-9`/`_`，**不能含空格**）、`--motion`（`J`/`L`）、`--speed`、`--j-speed`、`--config`、`--home-joints`
- `--rot-weight`（毫米/度，把姿态变化计入排序代价，默认 0）
- `--selftest`（校验 W/P/R 换算与数字格式后退出）

```bash
uv run reference/PHASE_6_Export_Roboguide.py --input "reference/00_Output Collection/PH2 output <时间戳>.json"
```

# 准备工位数模（IGES → 网格）

PHASE_1 通过 trimesh 对**三角网格**做体素化，而 trimesh 读不了 IGES（也读不了 NX 原生 `.prt`）。`IGES_to_STL.py` 用 OpenCASCADE 把 IGES 装配体网格化，写出二进制 STL，并对过大的模型做简化。它同时打印 IGES 的 Global 段（原始文件名、生成器、单位），因为交接过来的文件名未必是导出时的名字——真正用来辨认的是打印出的内部文件名，以及每个模型的几何摘要。

默认的 `--backend ocp` 直接调用 OpenCASCADE（经 `cadquery-ocp`），**不做缝合修复**就直接网格化——这正是它能在这些文件上跑通的原因：`--backend gmsh` 会先把各面缝合成一个合法壳体，然后在 `Could not fix wire in surface N` 处整体失败，因为 PolyWorks 导出的装配体含有上千个修剪曲面、线框本身就轻微不一致。网格化并不需要合法壳体，所以仍然失败的面会被计数并跳过，而不是让整次转换中断。`auto` 会先试 ocp，再回退到 gmsh。

它跑在自己的环境里，不碰固定版本的流水线环境：

```bash
uv run --no-project --python 3.11 \
    --with cadquery-ocp --with gmsh --with trimesh \
    --with fast-simplification --with numpy --with matplotlib \
    python reference/IGES_to_STL.py --input-dir <存放 .igs 的目录> \
    --preview --report <存放 .igs 的目录>/mesh/report.json
```

网格默认写到 `<input-dir>/mesh`，可用 `--out-dir` 指定；`--keep-ratio`（默认 `0.1`）控制简化后保留的面数比例，`--decimate-above`（默认 `200000`）是触发简化的面数门槛，`--deflection-mm`（默认 `1.0`）是 ocp 后端的弦高容差。坐标原样透传，所以 mm 的 IGES 出来的就是 mm 的 STL。`report.json` 会逐个模型记录：IGES 头部信息、在共同 CAD 坐标系下的包围盒、简化前后的三角面数、以及成功网格化的面数。

要让**算法世界坐标系与数模坐标系完全一致**，需把 PHASE_1 对物体网格施加的变换"归零"：令 `obj_mesh_scale` 使缩放后的网格以米为单位（mm 网格取 `0.001`，因为正是 `tm_voxelSize = 0.01` 让一个体素等于 10 mm），并把 `reference/PHASE_1_Get_Can_VPs.py` 物体加载段里的旋转矩阵改成恒等。PHASE_1 与 PHASE_2 **不要求模型 Z 朝上**，只有 PHASE_3 的机器人站位采样才依赖，而导出到 ROBOGUIDE 这条路线用不到 PHASE_3。

# Python 环境（uv）

本项目基于一套 2022 年的固定依赖版本开发（`numpy 1.23.1`、`scipy 1.8.1`、`trimesh 3.12.7`、`mujoco 2.2.0`、`ortools 9.4.1874` 等）。这些版本最高只提供到 Python 3.10 的 wheel，因此 `.python-version` 固定为 `3.10`，`requires-python` 限制为 `>=3.10,<3.11`。

```bash
uv sync                                # 创建 .venv 并安装锁定版本的依赖
uv run reference/PHASE_1_Get_Can_VPs.py     # 在环境中运行某个阶段
```

若本机未安装 Python 3.10，`uv sync` 会自动下载。

各阶段均从当前工作目录下的 `reference/00_Output Collection` 文件夹读取输入并写出结果，请在仓库根目录下运行脚本。每个阶段顶部的输入文件名是硬编码的：需将其指向上一阶段的输出文件（例如 `PHASE_2` 读取 `PHASE_1` 生成的 `PH1 output <时间戳>.json`）。

# Python 依赖包
| 包名 | 版本 |
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

# 仿真环境（mujoco 2.2.0 版本）
(1) mujoco 独立软件可在此获取：https://mujoco.org/ <br />
(2) mujoco 的 Python 包可在此获取：https://github.com/deepmind/mujoco/releases <br />
若要查看示例仿真环境（文件夹 `samples/sim env/00_sample simulation env.xml`），请下载 mujoco 独立软件（见参考 (1)）并打开文件夹 `bin/simulate.exe`。在那里可将该 XML 文件拖放进去以启动仿真。 <br />
若要进行更多仿真步骤，请使用 Python 包（见参考 (2)）。该包的使用文档见此处：https://mujoco.readthedocs.io/en/latest/python.html <br />
此外，在 `samples/sim env` 文件夹中有一个示例 Python 文件，它使用阶段 5 的输出文件来驱动机器人运动。
