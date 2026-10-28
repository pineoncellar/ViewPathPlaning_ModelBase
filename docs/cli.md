# 命令行手册

`scanplan` 五个子命令（`plan` / `ls` / `view` / `pick-datums` / `web`）的用法、可加参数与配准文件格式。

## 命令行等价路径

`scanplan` 有五个子命令：`plan` / `ls` / `view` / `pick-datums` / `web`。网页端做的是同一套
事，不开界面时用这三条：

```bash
uv run scanplan plan --station input/<工位>.json \
                     --face auto --out output/<零件>/

uv run scanplan ls   --viewpoints output/<零件>/viewpoints.json \
                     --prog-name PART01 --out output/<零件>/

uv run scanplan view --viewpoints output/<零件>/viewpoints.json \
                     --station input/<工位>.json --out output/<零件>/
```

`--station` 给出三个数模的路径与它们的配准变换；也可以直接给
`--part/--fixture/--base` 三个路径，此时按单位变换处理。可以加的还有
`--config`（覆盖配置）、`--tracker x,y,z`（给出跟踪器真实位姿）、
`--tracker-sweep`（不给位姿时，报告 192 个候选摆位的差别）、`--no-3d`（不写 3D 视图）、
`--max-faces`（3D 视图里每个模型的显示面数上限）、`--quiet`。
模板见 [`configs/station.example.json`](../configs/station.example.json)。

### 快速模式：`--fast`

迭代关节模型时需要一条几分钟就能重跑一遍的路径，而不是半小时：

```bash
uv run scanplan plan --station input/<工位>.json --fast --out output/<零件>-fast/
```

`--fast` 只做两件事，都是把**已有**的键写成粗一档的值，所以报告里显示的就是实际生效值：

* `target.sample_grid_mm` → 10 mm（`--fast-grid MM` 可改，下限 8 mm）；
* `target.face` → `upper`（显式 `--face` 仍然优先）。

实测（1.05 m² 真实零件、同一条命令只改网格）：3 mm / `face auto` = 1724 s，
10 mm / `face auto` = 223 s；再降到单面约再省一半。加速来自采样点数——覆盖率成本是
`带位置数 × 采样点数`，与射线算法无关。

配套的两个开关：

* `--no-occlusion`：覆盖率阶段完全不测视线遮挡（只有和 `--fast` 同用才合法）。约再省
  10–15%，但姿态可能"隔着零件覆盖"到采样点。报告的未覆盖归因会变成 `not_attributed`。
* `--config configs/fast.json`：`--fast` 的等价预设文件，供脚本与网页端使用。

**快速模式的数字不能当验收依据**，这不是提示而是机制：报告的 `fast_mode` 块
（机读）、`scope` 文案、以及控制台的 `FAST MODE` 行都会写明，且 `runtime.fast` 为真。
引出任何数字之前，去掉 `--fast` 重跑一次。

`--fast` **不**把 `stroke.enabled` 关掉，也不改 `stroke.check_step_mm`：J5 超程发生在
扫描带的直线扫掠段上，而 `check_step_mm` 同时是关节筛选对直线移动的采样步长——
改粗它会让"检查"比"被检查的运动"还粗。

### 快速模式配哪种交付模式

同一零件、同一 seed、同一条 `--fast` 命令，只改交付模式（实测 2026-10-28）：

| 搭配 | 端到端 | 路径内容 | 能检查什么 |
|---|---|---|---|
| **快速 + 扫掠模式**（推荐） | **109 s** | 39 拐角 / 15 次 re-orientation，含 L 扫掠段 | 每个位姿 + 每条 L 扫掠段 + 连接器 |
| 快速 + 纯位姿模式 | **31 s** | 35 个孤立位姿，J 移动，无扫掠段 | 每个位姿 + J 移动的两端 |

默认用**扫掠模式**：它是唯一包含扫掠段的组合，而"腕部沿带飞不过去"这类超程只在那里出现；
109 s 已经落在迭代可接受的范围。**纯位姿 + 快速**留给一分钟级的预检——如果连位姿都大面积
不可达，那就是姿态/分支选择的问题，不必等带算完；但它**不能**当作"路径可达"的证据，而且
现场真正会飞的 `L`+`CNT0` 指令根本不在那份结果里。

加速的空间也在这里：快速模式下 109 s 里有 81 s 是扫描带阶段，再想快只能动
`coverage_step_mm` / `stroke.max_stroke_mm`，那会改变交付的带本身，不是免费的速度。

> **`--face upper` / `--face lower` 是几何名，不是物理名。** 哪一个面叫"上面"取决于
> 分面时定向种子的符号，而那个符号并没有锚定到工位文件上：交换它，两个面的名字会互换，
> 而两面面积、料厚、定向比例**全都一模一样**。报告里因此记录了 `through_axis`（分面认为
> 的"厚度方向"）和一段说明，供人核对。**默认的 `--face auto` 不受影响**，它两面各算一次
> 可达性、取高的那一面，这也是推荐用法。

`plan` 与 `.LS` 导出**刻意分开**：ROBOGUIDE 的坐标系契约（用哪个 `UFRAME`、原点落在
哪里）需要与现场商定，可能只要求重新导出而不必重跑规划。所以 `ls` 只读视点文件，
**不需要数模**。

## 命令行配准：`pick-datums`

现场可用零件上的三个定位基准点，把 CAD 路径配准到实测的机器人基坐标系。每个基准点必须提供一对
坐标：CAD/视点文件坐标中的点，以及现场在机器人基坐标系中测得的点（mm）。
如果 CAD 中的对应特征还没有坐标，可以先在模型视图中点选，再输入三组现场测量值。
下面示例中现场坐标来自机器人基坐标：

```bash
uv run scanplan pick-datums --station input/<工位>.json \
  --measured-points "x1,y1,z1;x2,y2,z2;x3,y3,z3" \
  --coordinate-system robot_base_mm \
  --out output/<零件>/registration_picked.json
```

窗口中按 P1、P2、P3 的顺序点击 CAD 上同一物理基准特征，并确保顺序与测量坐标一一对应；
视图按所选的整车坐标轴正负方向投射到模型，右键可以撤销上一个选择。若已有配准文件，可用
`--guide-registration` 将其中的 CAD 点显示为青色参考标记；仍需在窗口中逐点点击，点击标记
附近（30 mm 内）会吸附到原 CAD 点，其他位置则按表面射线拾取。点选完会输出配准 JSON，
并打印拟合误差。三对点也可以直接写入 JSON：

```json
{
  "coordinate_system": "robot_base_mm",
  "point_pairs": [
    {"cad_mm": [100.0, 200.0, 300.0], "robot_base_mm": [510.0, 220.0, 840.0]},
    {"cad_mm": [200.0, 200.0, 300.0], "robot_base_mm": [510.0, 320.0, 840.0]},
    {"cad_mm": [100.0, 280.0, 300.0], "robot_base_mm": [430.0, 220.0, 840.0]}
  ]
}
```

需要重用已有 CAD 点作为拾取参考时，可添加 `--guide-registration`：

```bash
uv run scanplan pick-datums --station input/<工位>.json \
  --measured-points "x1,y1,z1;x2,y2,z2;x3,y3,z3" \
  --coordinate-system robot_base_mm \
  --guide-registration input/previous_registration.json \
  --out output/<零件>/registration_picked.json
```

配准输入只接受机器人基坐标：`coordinate_system` 必须是 `"robot_base_mm"`，配对字段名必须是
`robot_base_mm`；也兼容 `measured_coordinate_frame: "robot base"` 配合 `cad_points_mm`、
`measured_points_mm` 这种数组格式。写成 `UFRAME 1`–`UFRAME 9` 的配准文件会被拒绝并提示
用基坐标重新量点——现场用 UFRAME 量点时请改用浏览器界面（`scanplan web`），那里可以直接
选择 `机器人 UFRAME 1` 或 `机器人基坐标 / UFRAME 0`。

```bash
uv run scanplan ls --viewpoints output/<零件>/viewpoints.json \
                   --prog-name PART01 \
                   --registration input/part_registration.json \
                   --out output/<零件>_现场配准/
```

默认要求三个点的最大配准残差不超过 3 mm；可按测量精度用
`--max-registration-error-mm` 调整。输出 `registration_report.json`，记录刚体变换和各点误差。
三个定位点必须不共线、对应同一组 CAD 特征，且现场坐标必须是机器人基坐标，不能与其他
坐标系混用。程序因此固定使用 `UFRAME 0`；导出后仍须在 ROBOGUIDE/现场核对点位。
