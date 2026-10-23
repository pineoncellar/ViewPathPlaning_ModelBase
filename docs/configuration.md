# 参数说明与修改指南

配置的三层合并规则、参数四档出处、`surface_bias` 派生表、点数上限与 λ 搜索、候选池与扫描带三道门、机器人关节门，以及改完该看哪几个文件。

[`configs/scanner_trackscan49.json`](../configs/scanner_trackscan49.json) 是唯一的出厂配置。
参数的出处分四档，文件里的注释与报告里的 `parameter_provenance` 按同一套分档写：
**数据表直接给出**（基准距、景深、最大面幅、跟踪距离）、**由数据表推导**（量程、
跟踪器半角 25°）、**仍未实测**（入射角上限、扫描头外形包络、视场锚点距离、标记点朝向）、
**设计取值**（距离/倾角/滚转阶梯、扫描带步长）。读覆盖率数字之前先看**仍未实测**那一档：
`max_incidence_deg` 的敏感度远高于其他任何一项，而 `fov_anchor_distance_mm` 一动就会
让视点数变两成。

## 怎么改：三层合并

优先级从低到高：

1. **出厂文件** [`configs/scanner_trackscan49.json`](../configs/scanner_trackscan49.json)；
2. **派生表** `config.py` 的 `SURFACE_ANCHORS`——只有 `target.surface_bias` 派生出的六个
   `stroke` 键走这一层；
3. **覆盖文件**：`--config <json>` 指定的那份。网页端不传覆盖件，它的默认值来自
   `load_config()` 读到的出厂配置。

写进覆盖文件的键永远赢，所以"把某个派生值钉死"的写法就是显式写上它。报告里的
`parameter_provenance.surface_profile.sources` 逐键标明最终来源是 `base`、
`profile(bias=x)` 还是 `override`；未知键直接报错，拼错不会静默失效。

现成的覆盖件：

| 文件 | 用途 |
|---|---|
| [`configs/lean.json`](../configs/lean.json) | 位姿优先（`delivery_mode: lean`），不生成连接点 |
| [`configs/stroke_on.json`](../configs/stroke_on.json) | 把扫描带钉住（现在就是默认，留着防默认值再变） |
| [`configs/legacy_exhaust.json`](../configs/legacy_exhaust.json) | 回归金标：毫米计价、L2 连接、跑到耗尽，三道门全关 |
| [`configs/robot_gate.json`](../configs/robot_gate.json) | 只打开机器人关节门（`robot.enabled: true`） |

## 曲面取舍：一个旋钮管六项（合并方案 A）

`target.surface_bias ∈ [0, 1]`，默认 `0.5`。它表达"覆盖率优先"还是"规整优先"：

- `0.0` 覆盖优先——继续往低产出的尾巴里长（`min_gain 50`、`min_stroke 60`、两个切线方向
  都试、排序便宜、修补可花 35% 预算）；
- `0.5` 出厂值——与合并前的数字**逐项相等**；
- `1.0` 规整优先——早停（`min_gain 600`、`min_stroke 300`、只沿一个切线方向、排序更贵、
  修补只 5%），在强曲面上牺牲几个百分点，换更少的拐角与更多平行带。

它派生的正是下面六项，三档取值抄自 `SURFACE_ANCHORS`：

| 派生键 | 作用 | bias=0.0 | bias=0.5 | bias=1.0 |
|---|---|---|---|---|
| `stroke.min_gain_mm2_per_step` | 延伸一步至少多扫多少面积才值得 | 50 | 200 | 600 |
| `stroke.min_stroke_mm` | 一条带短于这个长度就不收 | 60 | 150 | 300 |
| `stroke.seed_dirs` | 每条带允许沿几个切线方向生长（≤2 只用第一个轴，>2 两个轴都试） | 4 | 4 | 2 |
| `stroke.order_axis_weight_mm_per_deg` | 60° 换向折合多少毫米绕行 | 1.0 | 2.0 | 3.5 |
| `stroke.parallel_weight` | "与已有骨架同向"的偏好权重：只做 tie-break，最多挪动 30% 分数，修补阶段不用 | 1.0 | 1.0 | 1.3 |
| `stroke.repair_budget_share` | 收尾修补最多花掉 `max_points` 的多少比例（只在设了上限时生效） | 0.35 | 0.20 | 0.05 |

中间值按段线性插值（`seed_dirs` 取整）。刻意**不**进派生表的两个数是
`scanner.max_incidence_deg` 与 `scanner.range_mm`：让扫描头"看得更远"不是口味问题，用它们
换覆盖率只会让报告好看、扫描头却扫不到。

改法一：只给 bias，六项跟着走。

```json
{"target": {"surface_bias": 0.8}}
```

改法二：显式钉死某一项（钉了就不再跟着 bias 动，报告里记作 `override`）。

```json
{"stroke": {"min_stroke_mm": 220}}
```

## 点数硬上限与 λ 搜索

`target.max_points`（默认 `null`，整层跳过）是交付点数的硬上限。设了以后选择器在 λ 上搜
一遍：λ 是"多覆盖一点面积愿意多付几个拐角"的汇率，按固定网格 `0 / 0.5 / 1 / 2 / 4 / 8 /
16 / 32` **递增**，第一个超预算就停。判据是 `build_path` **之后**的真实转角数，不是估算。

报告 `budget` 块的 `verdict`：`target_bound`（先到覆盖率目标）、`budget_bound`（先撞点数
上限）、`exhausted`（池子没得可给了），`max_points` 为 `null` 时是 `unbudgeted`——没有第四
种答案。`trials` 记录每档 λ 试出来的点数与覆盖率，`dropped_by_hard_gate` 是被关节门丢掉的
条目数。

```json
{"target": {"max_points": 120}}
```

把上面那段存成文件（例如 `configs/budget120.json`）再 `--config`（`--config` 只吃路径，不吃
内联 JSON）：

```bash
uv run scanplan plan --station input/<工位>.json \
                     --config configs/budget120.json --out output/<零件>/
```

覆盖率目标本身是现场旋钮，可以逐次覆盖：`--coverage-target 97.5`，或 `--coverage-target none`
跑到耗尽（对应配置里的 `target.coverage_target_pct`，默认 99.0）。

## 候选池与扫描带侧的三道门

`stroke.pool_mode`（`full` / `grid` / `rails`）决定候选池怎么铺：

- `full`（默认）= 瞄准网格 + 导轨线 + 侧扫族；
- `grid` = 只有瞄准网格，历史行为（`legacy_exhaust` 用这一档）；
- `rails` = 只铺导轨线，候选池最小，适合先出一份草稿。

`stroke.side_scan_tilt_deg`（默认 45）是侧扫族的倾角：正对扫描时曲面陡的地方会掉出量程，
侧着扫能把它们捞回来。

三道门都在 `stroke` 段，默认全开：

| 键 | 关掉之后 |
|---|---|
| `lock_roll` | 种子生成时滚转自由，相邻点的姿态框架不再统一 |
| `scan_gate` | 生长只看"能不能连上"，不看"连上之后还看得见东西吗" |
| `hard_gate` | 连不进去的带不再被丢掉重选，只写一行 violation 继续交付 |

`configs/legacy_exhaust.json` 把三项全关、`pool_mode` 设成 `grid`，这是它能继续当回归金标
的前提。

## 机器人关节门（合并方案 B）

`robot` 段默认 `enabled: false`，关着与合并前逐项一致。打开后规划期就用实测的臂/腕模型
筛位姿：候选点要在限位里留出余量、过渡点要有逆解，交付点逐点带 CONFIG 与 J5 余量。

| 键 | 默认 | 含义 |
|---|---|---|
| `robot.enabled` | `false` | 总开关 |
| `robot.joint_margin_deg` | `5.0` | 每个关节离限位至少留这么多度 |
| `robot.j5_min_abs_deg` | `8.0` | 腕奇异保护：`\|J5\|` 不小于这个值 |
| `robot.max_step_deg` | `40.0` | **只报不拦**：相邻交付点的最大关节步，超限计数进报告 |
| `robot.soft_limits_deg` / `mechanical_limits_deg` | 见文件 | 两套限位取交集，报告写明用了哪套 |
| `robot.wrist_identified` | `true` | 关掉后禁止输出绝对关节值，门只能做分支排序 |
| `robot.tool_xyz_mm` / `robot.tool_ypr_deg` | 见文件 | `UTOOL 1`：法兰 → 扫描头参考点。换了工具就重标，流程见 [robot-calibration.md](robot-calibration.md) |
| `robot.base_offset_mm` / `upper_arm_mm` / `forearm_mm` / `forearm_offset_mm` / `j2_height_mm` | 见文件 | 臂几何，与图纸互证的实测值；新数据只会复现它，不会改写它 |

两条要记住的语义：

- **硬门只对两件事丢条目**：间隙 ≤ 0，或过渡点解不出逆解。`stroke.max_rot_deg_per_mm` 是
  未标定常数，**只报不拦**——拿它丢带会把覆盖率从 99% 打到 82%。
- 丢掉的条目走**排除重选**，不是只过滤；只过滤不重选同样会把覆盖率打下去。

`--joint-blocks` 让 `.LS` 的 `P[n]` 直接写关节角，**只对带 `joints_deg` 的点生效**（也就是打开
`robot.enabled` 跑出来的视点文件），没有关节值的点仍写笛卡尔位姿。三个现场缺口（软限位准确
值、CONFIG 圈数位的含义、ROBOGUIDE 是否配了数据表限位）没补齐之前，别把 `enabled`
打开当验收用；`T/F` 字母已在 2026-10-23 那批 8 组数据上核对过（`T` ↔ J5 < 0）。

```bash
uv run scanplan plan --station input/<工位>.json \
                     --config configs/robot_gate.json --out output/<零件>/
uv run scanplan ls --viewpoints output/<零件>/viewpoints.json \
                   --prog-name PART01 --joint-blocks --out output/<零件>/
```

## 改完看哪里

- `coverage_report.json`：`parameter_provenance`（每个参数的出处）、`surface_profile`
  （bias、派生出的有效值、逐键来源）、`kinematics`（限位来源、J5 直方图、分支直方图、
  min margin、最大步长与超限点数）、`budget`（verdict、每档 λ 的试验记录）；
- `summary.txt`：Joints 与 Budget 两行（没开关节门时不出现 Joints）；
- 一条守着的不变量：`coverage_pct` **等于**路径里所有覆盖点的并集；`swept_coverage_pct`
  另算过渡段顺手扫到的面积，只作参考、不参与验收；
- 网页端显示的默认值来自 `GET /api/config`，也就是服务端**实际会用的**那份配置——页面说
  扫掠，跑的就是扫掠。
