<!--
@File     :   tech-stack.md
@Desc     :   全项目技术栈说明：运行时、依赖分层、算法与数值技术、数据格式、前后端形态、外部系统接口、工程工具链与技术边界
@Note     :   版本号以 pyproject.toml 与 uv.lock 为准；reference/ 是冻结基线，其依赖面与主包不同，混着看会误判
-->

# 技术栈说明

这份文档回答一个问题：**这个仓库是靠哪些技术拼起来的，每一块负责什么。**
算法为什么这么选、风险清单、实测数字与验收口径不在这里，见
[architecture.md](architecture.md)、[validation.md](validation.md) 与
[stroke-history.md](stroke-history.md)。

> 本文不含客户几何与真实站位，机制部分可以外发。

---

## 1. 一页速查

| 层次 | 选型 | 版本 / 形态 |
|---|---|---|
| 语言 | Python | 3.10（本机 3.10.16；`requires-python = ">=3.10,<3.11"`） |
| 环境与依赖 | uv + `uv.lock` | uv 0.6.6；锁文件 36 个包，`requires-python == 3.10.*` |
| 打包 | Hatchling，src 布局 | wheel 收 `src/scanplan`、`src/scanview`、`src/scanweb` |
| 命令行 | argparse 子命令 | `plan` / `ls` / `view` / `pick-datums` / `web`，入口脚本 `scanplan` |
| 数值计算 | NumPy | 1.23.1 |
| 三角网格与几何 | trimesh | 3.12.7 |
| 空间索引与稀疏矩阵 | SciPy | 1.8.1（`cKDTree`、`csr_matrix`） |
| 光线求交 | Embree，经 embreex | 4.4.0（可选 extra），缺失时退化到 trimesh 纯 Python |
| 机器人运动学 | 自研解析正逆解 | 几何由 8 组实测位姿标定（≤0.02 mm / 0.002°，见 [robot-calibration.md](robot-calibration.md)）；默认关闭 |
| 2D 交互点选 | Matplotlib | 3.5.3（交互后端） |
| 3D 视图 | 手写 WebGL + 原生 JS | 单文件 `viewer.html`，无 three.js、无 CDN 依赖 |
| 本机 Web 界面 | Python 标准库 `http.server` | `ThreadingHTTPServer`，仅监听 `127.0.0.1:8765` |
| 配置 | JSON + frozen dataclass | `configs/*.json`，三层合并（出厂 → 派生表 → 覆盖件），加载时按类型注解校验 |
| 测试 | pytest | 23 个文件、354 个用例 |
| 静态检查 | ruff | `line-length = 96`、`target-version = py310`、规则 `E,F,W,I,UP,B` |
| 交付格式 | JSON / CSV / NPZ / STL / `.LS` / HTML | 见第 6 节 |
| 外部系统 | FANUC 控制器 + ROBOGUIDE、TrackScan-Sharp 49、CAD（IGES / NX `.prt`） | 见第 9 节 |

没有的东西也值得写明：没有数据库、没有容器、没有云服务、没有前端构建工具链、没有 CI。

---

## 2. 运行时与依赖管理

### 2.1 为什么钉死在 3.10

`pyproject.toml` 把 `requires-python` 卡在 `>=3.10,<3.11`，`.python-version` 写 `3.10`，
`uv.lock` 里是 `requires-python == 3.10.*`。原因不是主包挑食，而是**共用一个环境**：

- `reference/` 是冻结的原管线，开发于 2022 年那套包（`numpy 1.23.1`、`scipy 1.8.1`、
  `trimesh 3.12.7`、`mujoco 2.2.0`、`ortools 9.4.1874`…），这些版本只出到 cp310 的 wheel；
- `src/` 的新规划器本身在新版本 Python 上也能跑，但它跑在同一个 `.venv` 里。

要解除这个约束，得先把 `reference/` 迁到新版本，而不是改一行 `requires-python`。

### 2.2 环境命令

```bash
uv sync --extra dev --extra embree
uv run pytest
uv run scanplan plan --station input/<工位>.json --face auto --out output/<零件>/
```

`uv sync` 缺 Python 3.10 时会自己下载。锁文件是唯一真相，没有 `requirements.txt`。

### 2.3 打包与入口

- 构建后端 `hatchling`，包配置明确列出三个包目录（`packages = ["src/scanplan", "src/scanview", "src/scanweb"]`），
  这样 `scanplan web` 装到任何环境都能找到随包分发的 `index.html`，而不是去猜仓库根目录。
- 控制台入口：`scanplan = "scanplan.cli:main"`。
- pytest 侧用 `pythonpath = ["src"]` 直接跑源码树，不必先安装。
- ruff 侧 `target-version = "py310"`，与运行时一致。

---

## 3. 依赖分层

`pyproject.toml` 里的依赖有三个来源，混在一起看会误判"这个项目用了多少东西"。

### 3.1 主包真正 import 的

| 包 | 版本 | 用在哪 | 说明 |
|---|---|---|---|
| `numpy` | 1.23.1 | 全包 | 所有几何量、位姿、掩膜、统计 |
| `scipy` | 1.8.1 | `scene`、`sheet_split`、`coverage`、`strokes` | `cKDTree` 做邻域与去重，`csr_matrix` 存候选×样本的覆盖集与扫掠覆盖 |
| `trimesh` | 3.12.7 | `scene`、`target_surface`、`geometry`、`pick_datums`、`raycast` | STL 读写、表面采样、包围盒、光线桥接 |
| `matplotlib` | 3.5.3 | 仅 `datum_picker`（函数内延迟 import） | 基准点交互点选窗口 |

`matplotlib` 只在执行 `pick-datums` 时才会被 import，所以没有图形后端的环境照样能跑 `plan`；
交互点选窗口才需要它。

### 3.2 可选加速：embree

```toml
embree = ["rtree==1.0.0", "embreex==4.4.0", "setuptools<70"]
```

- 装了 Embree，光线求交快约**三个数量级**；没装，规划器照样能跑，只是真实零件基本等不起。
  因此它是 extra 而不是硬依赖。
- `setuptools<70` 是因为 trimesh 的 pyembree 桥要 `pkg_resources`，而 setuptools 70 之后不再提供。
- `raycast.py` 统一封装了后端切换，报告里会记下本次用的是 `embree` 还是 `trimesh-python`，
  因为两者耗时不可比。
- 一个踩过的坑写在 `raycast.py` 里：`import trimesh` 会顺带 import `trimesh.ray`，把"Embree
  加载失败"缓存成一个异常模块，之后即使补装了也不生效（实测慢 1600 倍）。

### 3.3 为冻结基线保留的依赖

| 包 | 版本 | 谁在用 |
|---|---|---|
| `ortools` | 9.4.1874 | `reference/PHASE_5_TSP.py`（TSP 排序） |
| `scikit-learn` | 1.1.2 | `reference/PHASE_1`、`PHASE_3`（`NearestNeighbors`） |
| `mujoco` / `glfw` | 2.2.0 / 2.5.3 | 上游样例仿真环境 `samples/sim env/_muj_sim.py` |
| `pyglet` / `protobuf` | 1.5.26 / 4.21.5 | 2022 冻结集的遗留项：`protobuf` 是 ortools 的传递依赖，`pyglet` 在仓库里已无人直接 import |
| `matplotlib` | 3.5.3 | 同一个包，兼作 `reference/` 的多处出图（见 3.1） |

`src/` 不使用 ortools、scikit-learn、mujoco、glfw、pyglet、protobuf。保留它们是为了让
`reference/` 与上游样例在同一环境里可复现——这是"回归基线可跑"的代价。

### 3.4 不在 pyproject 里的依赖

`reference/IGES_to_STL.py`（IGES 转网格）用到 `cadquery-ocp`（`OCP.*`）、`gmsh`、
`fast_simplification`，这些**没有写进 `pyproject.toml`**，因为主包不 import 它们。官方跑法是用
独立环境，避免污染主环境：

```bash
uv run --no-project --python 3.11 \
    --with cadquery-ocp --with gmsh --with trimesh \
    --with fast-simplification --with numpy --with matplotlib \
    python reference/IGES_to_STL.py --input-dir <igs 目录> --preview
```

默认后端 `ocp` 直接走 OpenCASCADE 且**不做缝合**——这批 PolyWorks 导出的装配有成千上万个
裁剪面，缝合必失败（`--backend gmsh` 会卡在 `Could not fix wire in surface N`）。

---

## 4. 代码组织与规模

```
src/scanplan/cli/        子命令，一个命令一个模块，只做参数与编排
src/scanplan/            算法核心：几何、采样、候选、覆盖、选型、路径、滚转、运动学、配准、报告
src/scanplan/pipeline/   一次完整规划的编排、候选补齐、交付与排序
src/scanplan/report/     报告字典的组装与落盘（JSON / CSV / summary）
src/scanplan/roboguide/  `.LS` 写出、WPR 欧拉角、帧契约、配准报告
src/scanview/            自包含 3D 网页
src/scanweb/             本机浏览器工位界面
```

| 目录 | 文件数 | 行数 |
|---|---|---|
| `src/scanplan/*.py` | 19 | 4,980 |
| `src/scanplan/cli/` | 8 | 450 |
| `src/scanplan/pipeline/` | 5 | 843 |
| `src/scanplan/report/` | 9 | 754 |
| `src/scanplan/roboguide/` | 5 | 787 |
| `src/scanview/` | 2 个 py + `viewer.html` | 478 行 Python，HTML/JS 1,284 行 |
| `src/scanweb/` | 1 个 py + `index.html` | 648 行 Python，HTML/JS 1,403 行 |
| `tests/` | 23 | 3,867 |
| `reference/` | 8 | 3,493 |
| `scripts/` | 1 | 70 |

依赖方向是单向的：算法核心不 import `cli`、`report`、`scanview`、`scanweb`；`scanview` 与
`scanweb` 只调 `scanplan` 的公开入口（`plan` / `write_outputs` / 配准 / 导出），所以换渲染器
或换界面不会碰到算法。

---

## 5. 算法与数值技术

| 技术点 | 实现方式 | 代码位置 |
|---|---|---|
| 可复现表面采样 | 每格一点、等面积份额，显式播种（trimesh 3.12.7 从全局 numpy 随机态抽样，不播种两次结果不同） | [geometry.py](../src/scanplan/geometry.py)、[target_surface.py](../src/scanplan/target_surface.py) |
| 目标面分割 | 镜像配对 → 中面 PCA → 局部片法向 `n·m` 的符号传播，拓扑定面而不是朝向定面 | [sheet_split.py](../src/scanplan/sheet_split.py) |
| 光线求交与遮挡 | Embree BVH，分块查询（`DEFAULT_CHUNK = 250_000`）压峰值内存 | [raycast.py](../src/scanplan/raycast.py) |
| 单点位姿有效性 | 视场（随距离缩放的矩形）、量程、入射角、遮挡、本体包络、跟踪器六条判据 | [scanner_model.py](../src/scanplan/scanner_model.py) |
| 候选生成 | 每个瞄准格点铺"距离 × 倾角 × 滚转"阶梯后去重；滚转必要是因为视场是矩形 | [candidates.py](../src/scanplan/candidates.py) |
| 候选×样本覆盖集 | `scipy.sparse.csr_matrix`，每行是一个候选位姿可达的样本及入射角 | [coverage.py](../src/scanplan/coverage.py) |
| 集合覆盖 | 贪心（每次选新增覆盖面积最大者）＋ 补齐（给仍可及的漏点单配位姿） | [selection.py](../src/scanplan/selection.py) |
| 由粗到细补候选 | 对残留漏点用更细的瞄准网格重生成候选，再评估、合并、重选 | [refine.py](../src/scanplan/pipeline/refine.py) |
| 路径排序 | 贪心最近邻（含进端选择）＋ 单条重定位 ＋ 2-opt 反转；成本里计转角 | [path.py](../src/scanplan/path.py) |
| 分支分配 | 同一位姿常有多组逆解，用动态规划在相邻点之间挑一组，使关节步数最小（只在机器人门打开时生效） | [path.py](../src/scanplan/path.py) |
| 扫描带 | 线段 + 恒定姿态的位姿族，覆盖按整段扫掠合并 | [strokes.py](../src/scanplan/strokes.py) |
| 扫描段合法性 | 沿段加密取样，用"这个位置还能看到量程内的样本吗"判生长是否成立（KD 球查询，不再逐点打光） | [strokes.py](../src/scanplan/strokes.py) |
| 候选池分档 | `full` = 瞄准网格 + 导轨线 + 侧扫族；`grid` = 仅瞄准网格（历史行为）；`rails` = 仅导轨线 | [strokes.py](../src/scanplan/strokes.py) |
| 滚转轴线对齐 | 扫描仪局部 +X 取光轴在零件表面的投影，相邻视点的框架因此连续（滚转数值在 helper 切换处仍会跳 ~90°，框架不跳） | [rolls.py](../src/scanplan/rolls.py) |
| 六轴正逆解 | 几何由实测位姿标定；本控制器把 J3 报成绝对臂俯仰角，故用"臂 2×2 分支 + 腕闭式解"而不是 DH 链，300 组随机位姿往返全通过 | [kinematics.py](../src/scanplan/kinematics.py) |
| 关节门 | 判"间隙 + 逐点逆解可行"，软限位与机械限位取交集；J5 余量按可用带中心算（0 是腕奇异，不是带中心） | [kinematics.py](../src/scanplan/kinematics.py)、[path.py](../src/scanplan/path.py) |
| 点数预算 | 固定 λ 网格 0…32 递增搜索，判据是 `build_path` 之后的**真实转角数**，不是估算 | [pipeline/delivery.py](../src/scanplan/pipeline/delivery.py) |
| 丢带重选 | 硬门丢带后**排除该带重跑选择**；只过滤不重选会把覆盖率从 99% 掉到 82% | [selection.py](../src/scanplan/selection.py) |
| 三点刚体配准 | Kabsch / SVD 求旋转，残差逐点回报 | [registration.py](../src/scanplan/registration.py) |
| 工位自检 | 有符号穿透判据（零件表面点落在夹具表面之后且对附近每片夹具面成立），走模型自身几何、不抽随机数 | [scene.py](../src/scanplan/scene.py) |
| 跟踪器可见性 | 从跟踪器到扫描头的射线 + 半角/距离门；判的是位姿，不是目标点 | [tracker.py](../src/scanplan/tracker.py) |
| FANUC 姿态转换 | 固定轴 XYZ 欧拉角 `R = Rz(R)·Ry(P)·Rx(W)`，角度制 | [roboguide/wpr.py](../src/scanplan/roboguide/wpr.py) |

排序规则是两段式的：达到验收覆盖率之前"覆盖面大者胜"，达到之后"转角少者胜"。这条写死在
[ranking.py](../src/scanplan/pipeline/ranking.py)，`coverage_target_pct = null` 时退化成纯覆盖优先的
v1 行为。

2026-10-22 合并进来的两组改动在这条排序之上又加了三个层，默认值都等于合并前的行为：选择打分
加"连贯性代价"（λ 乘以该候选项要付的转角数）与关节同分支偏好（`parallel_weight`）；
`target.max_points` 非空时，外层再套一轮 λ 网格搜索，把"少覆盖一点、少几个拐角"变成可以自动
求解的问题。三者的推导链与参数含义见 [配置与修改指南](configuration.md#参数说明与修改指南)。

---

## 6. 数据格式与交付物

### 6.1 输入

| 文件 | 格式 | 说明 |
|---|---|---|
| `configs/scanner_trackscan49.json` | JSON + `_comment` 字段 | 扫描仪包络、采样、验收口径、扫描带参数、机器人段；每个值都标注了出处档位 |
| `configs/station.example.json` | JSON | 工位模板：三套数模路径、配准用的 4×4 齐次变换、跟踪器位姿 |
| `configs/lean.json` / `stroke_on.json` / `legacy_exhaust.json` / `robot_gate.json` | JSON 覆盖文件 | 深合并到默认配置，未知键直接报错 |
| 数模 | STL（binary） | 零件、夹具、底座；IGES/NX `.prt` 先用 `reference/IGES_to_STL.py` 转 |

配置加载是"JSON → frozen dataclass"，用 `typing.get_type_hints` 按注解校验，路径记进
`PlannerConfig.source` / `override`，报告里可追溯。读 JSON 用 `utf-8-sig`，因为 Windows 侧的
编辑器与 PowerShell 会写 BOM。

合并分三层，优先级从低到高：**出厂文件 → `surface_bias` 派生表（`config.py` 的
`SURFACE_ANCHORS`）→ 覆盖文件**。派生表只写 `stroke` 段的六项，覆盖件里写了的键永远赢，
`config.parameter_sources` 逐键记下最终来源（`base` / `profile(bias=x)` / `override`）。

### 6.2 输出（`pipeline/write_outputs.py`）

固定六件、可选一件（`strokes.json`，仅在生成路径时写），外加 3D 视图的两个文件。

| 文件 | 格式 | 内容 |
|---|---|---|
| `viewpoints.json` | JSON | 交付位姿：位置、光轴、滚转、覆盖点数与面积、入射角、跟踪器判定；机器人门打开时另带 `config` / `branch` / `joints_deg` / `j5_margin_deg` |
| `viewpoints.csv` | CSV | 同上，给表格工具 |
| `exempt_points.csv` | CSV | 夹具接触而豁免的采样点 |
| `unreachable_points.csv` | CSV | 未覆盖采样点，附逐点原因（含 `unreachable_joint_limit` / `no_line` / `not_facing`） |
| `coverage_report.json` | JSON | 主报告：覆盖率、漏点分类与聚团、入射角直方图、贪心曲线、参数出处、工位自检、配准，以及 `kinematics` / `budget` / `surface_profile` 三个块 |
| `summary.txt` | 纯文本 | 报告的一屏摘要（含 joints 与 budget 行） |
| `strokes.json` | JSON | 仅在生成路径时写：扫描带与过渡段 |
| `viewpoints_3d.html` + `target_samples.npz` | 自包含 HTML + NumPy 存档 | 3D 视图与重渲染用的样本侧车 |

`.LS` 由 `scanplan ls` 单独写出（`<NAME>.LS`、`<NAME>_points.csv`、`<NAME>_frame.json`），
只读 `viewpoints.json`，**不需要数模**。带式计划里运动类型与终止类型逐点可以从配置带出来
（`L` + `CNT0`）；`--joint-blocks` 且视点带关节值时，`P[n]` 直接写成关节角。

### 6.3 格式选择的理由

- CSV 给现场与质检，JSON 给程序与报告阅读器，NPZ 只在重渲染时需要（体积大、被 gitignore）。
- 3D 视图是**单个 HTML 文件**，网格、视点、覆盖关系全部内嵌，评审时双击就能看，不需要服务、
  不需要装依赖、不会把数模传到任何地方。

---

## 7. 前端与交互形态

这里没有前端框架，也没有构建步骤。

### 7.1 3D 视图（`scanview/viewer.html`）

- 手写 WebGL（`canvas.getContext("webgl")`），自带相机的透视投影、网格渲染、拾取与拖拽，
  1,284 行单文件，**不依赖 three.js，不引用任何 CDN**。
- 数据由 `scanview/payload.py` 序列化成内嵌 JSON，命中着色与逐点覆盖关系用位掩码表示。
- `scanplan view` 用 `viewpoints.json` + 数模重渲染，不重跑规划；`--max-faces` 控制每个模型的
  显示面数，超出时才做顶点聚类抽稀（默认值留得高，薄板抽狠了会把正反两面焊在一起）。

### 7.2 本机 Web 界面（`scanweb/`）

- 服务端只用标准库 `http.server.ThreadingHTTPServer`，没有 Flask / FastAPI / uvloop。
- 默认绑定 `127.0.0.1:8765`，只在本机可用；页面是随包分发的 `index.html`（1,403 行，原生 JS）。
- 五个端点：`GET /api/config`（页面控件的默认值来自服务端**实际会用的**配置）、
  `POST /api/register`（配准预览）、`POST /api/solve`（跑规划并写全部交付物）、
  `GET /api/viewer`（取本次运行的 3D 页面）、`GET /api/file`（取产物）。
  所有计算都在本进程里调 `scanplan` 的公开入口，产物不离开本机，运行目录是
  `output/web_runs/<运行名>/`。

### 7.3 基准点交互点选（`datum_picker.py`）

- Matplotlib 交互窗：把零件在 X-Z 平面按背面（Y < 0）投影，用 `PolyCollection` 画三角面。
- 点击后从 Y 方向打一条射线（走同一套 `RayCaster`），取零件朝 cabin 侧的第一个表面点，所以
  返回的是真实 CAD 坐标而不是屏幕坐标；已有配准文件时，30 mm 内会吸附到原 CAD 点。
- 依赖图形后端（交互窗口），无头服务器上不可用——这是 `pick-datums` 的硬约束。

---

## 8. 工程工具链与数据合规

| 工具 | 配置 | 现状 |
|---|---|---|
| pytest | `testpaths = ["tests"]`、`pythonpath = ["src"]` | 23 个文件、354 个用例；夹具几何全部是合成的，不含客户数模 |
| ruff | `line-length = 96`、`target-version = "py310"`、规则 `E,F,W,I,UP,B` | 本地手动跑 |
| `scripts/check_repo_hygiene.py` | 70 行，查 git 索引 | 一旦 `input/`、`output/`、`docs/superpowers/`、`spike/` 或客户几何被跟踪就报错，提交前跑 |
| `.gitattributes` | `* text=auto eol=lf`，图片与 STL 标二进制 | 树里混着 Windows 与 Unix 两种换行，不定策略会让每次改动显示成整文件重写 |
| CI | 无（没有 `.github/`） | 全部在本地跑 |

没有引入类型检查器（mypy / pyright）与 pre-commit，类型信息靠 `from __future__ import annotations`
加 dataclass 维持。

数据合规是这套工具链的一部分：`input/`、`output/`、`docs/superpowers/`、`spike/`、
`**/00_Output Collection/` 以及所有派生数值（`*.npy`、`*.npz`）都在 `.gitignore` 里，
自检脚本是最后一道闸。`docs/` 下的公开文档入库，只有私密子目录被 `docs/.gitignore` 挡住。

---

## 9. 外部系统接口

### 9.1 FANUC 控制器与 `.LS`

- 写的是 FANUC ASCII 程序格式：`/ATTR`、`/PROG`、`/POS`（`P[n]` 块）、`/MN`。
- `/ATTR` 里的 `PROG_SIZE`、`LINE_COUNT`、`MEMORY_SIZE` 必须自洽，文件名必须与 `/PROG` 名完全
  一致，否则控制器以 `Invalid name in /PROG section` 拒收。
- 只有笛卡尔位姿，**默认不发关节角**。逆解、关节限位、机器人碰撞全部由 ROBOGUIDE / 控制器负责。
  例外是显式打开的机器人门：`robot.enabled` 时规划期已经解过一遍，视点文件带 `joints_deg`，此时
  `--joint-blocks` 会把 `P[n]` 写成关节角（`J` 指令），关节模型没标定（`wrist_identified=false`）
  的情况下这个开关会拒绝生效。
- 运动：`scanplan ls` 的默认是 `J` 运动 + `FINE` 终止（审查用更宽松，且符合"把扫描头送到位"的
  意图）；如果视点文件带着扫描带计划，则以配置为准（`L ... CNT0`，100 mm/s），且**逐点**可以从
  视点文件带出运动/终止类型。`CONFIG` 字符串提示控制器取哪一组姿态解（同一个笛卡尔位姿可能有多组解，
  换一个 `CONFIG` 常常就从"不可达"变成可达）——门打开时它由分支 DP 算出来，而不是猜的。
- 速度：`check_speed` 拒绝超过控制器上限的进给，不会静默截断。
- 数字格式：固定三位小数、无前导零（`.094`，`-0.0` 写成 `-.000`）。

### 9.2 坐标系契约

| 角色 | 坐标系 |
|---|---|
| `viewpoints.json` 的 `position` / `axis` | 工位数模坐标系（mm）；`position` 是扫描头测量参考点，局部 `+Z` 是光轴、指向零件 |
| `.LS` 点位 | 默认数模坐标系 + `UFRAME 1`（现场教到数模原点）；给了 `--place` 则烤成世界坐标 + `UFRAME 0` |
| `UTOOL 1` | 必须教成测量参考点，工具 `+Z` 沿光轴 |

`scanner_frame()` 是全包唯一的帧定义，覆盖判定、导出的位姿、3D 视图、ROBOGUIDE 示教都吃这一份，
避免各处各说各话。

### 9.3 扫描仪与跟踪器

- 设备：Scantech TrackScan-Sharp 49，外置光学跟踪器的跟踪式扫描头。
- 参数出处分四档写在配置与报告里：数据表直接给出（基准距 300 mm、景深 400 mm、面幅上限
  600×500 mm、跟踪距离 300–6000 mm）、由数据表推导（量程、跟踪器半角）、仍未实测
  （入射角上限、扫描头包络、视场锚点距离、标记点朝向）、设计取值（距离/倾角/滚转阶梯、扫描带步长）。
- 配置里还留了两处待定的已知偏差：`range_mm` 写的是 120/480（按推导应为 100/500）、
  `tracker.half_angle_deg` 写的是 30（按数据表反解应为 25）。改这两个数会移动所有结果，所以
  改动要作为一次独立变更来做。

### 9.4 CAD 交付

- `trimesh` 读不了 IGES，也读不了 NX `.prt`，所以数模交接必须先转 STL（OpenCASCADE 或 gmsh，
  见 3.4）。
- 逐个文件导出的 IGES 会丢装配位姿，因此 `scene.py` 有一个显式、带自检的对齐步骤：读工位文件里
  的 4×4 变换把零件与夹具摆回底座坐标系，再跑一遍自检（包围盒、最小间距、穿透、接触面积），
  不盲信文件。

---

## 10. 运行成本与硬件

- **单机、纯 CPU、无 GPU、无服务、无数据库。** 一个 `.venv` 加一个工作目录就能跑完。
- 时间量级（真实零件、3 mm 采样网格、`face=auto`）：纯单点位姿、99% 目标约 2 分钟（本机最近一次
  108.5 s / 31 点）；打开扫描带并按 `path` 模式校核点间运动，同一张面约 **10–12 分钟**（最近两次
  实测：出厂配置 12 min，其中位姿 97 s、扫描带阶段 608 s；打开机器人关节门后 10 min）。更小的
  目标面或更粗的网格会显著更快。单次 `plan` 的射线量在千万级（某次运行为 1,974 万条）。
- 打开机器人门不会改变时间量级（逆解带缓存，实测 66 个交付点的运行仍是 10 分钟量级）；
  开销主要在候选与扫描带阶段，不在运动学。
- 内存峰值靠求交分块（每次 25 万条光线）压住，不靠机器内存硬扛。
- 不装 Embree 时求交慢约三个数量级——工程上等于不可用。

---

## 11. 技术边界

明确**不使用**以下技术，而且是设计选择而非缺口：

- 机器人**碰撞**、运动时间优化——不在本仓库建模，交给 ROBOGUIDE；
- 机器人逆解与关节限位：有一套由实测位姿标定的解析模型，但**默认关闭**（`robot.enabled=false`）；
  打开后它只做交付前的筛选与 CONFIG 排序，仍不替代 ROBOGUIDE 的可达性判定；
- 机器人模型与机械臂自身遮挡——本阶段没有机器人模型，`tracker_ok` 只对零件/夹具/底座判定；
- 点云处理、测量数据后处理、扫描仪驱动——本仓库只做几何规划，不连设备；
- Web 框架、前端构建、数据库、任务队列、云服务、认证——界面只在本机，无多用户概念。

---

## 12. 已知技术债

| 项 | 影响 | 备注 |
|---|---|---|
| Python 钉在 3.10 | 无法直接用新版本工具链 | 根因是 `reference/` 的 2022 包集只出到 cp310 |
| 无 CI | 回归靠人跑 pytest | 没有 `.github/` 目录 |
| 无类型检查器 | 跨模块契约靠 docstring 与测试守着 | 模块间大量传 `dict` / dataclass |
| Matplotlib 3.5.3 | 交互点选窗口老旧，且必须有图形后端 | 也用于 `reference/` 出图，升级要两边一起看 |
| 手写 WebGL 渲染器 | 3D 视图与 Web UI 无框架、无构建，改动靠读代码 | 换来的是单文件分发、零安装 |
| 扫描仪/跟踪器参数仍是占位值 | 覆盖率数字只能作指示性参考 | 配置内自带出处注释与两处已知偏差 |
| 机器人段未标定 | `robot.enabled` 打开后仍是"参考可达性" | 两个现场缺口：软限位准确值、ROBOGUIDE 是否用数据表限位 |
| 私密设计笔记与 `spike/` 不入版本控制 | 设计笔记随工作副本走，换机器会丢 | 合规要求，不是疏漏；`docs/` 其余文档入库 |
| 无库 API 稳定承诺 | 内部模块互调频繁，对外只有 CLI | 对外入口是 `scanplan` 子命令与文件契约 |
