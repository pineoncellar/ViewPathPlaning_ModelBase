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

覆盖率目标本身是现场旋钮，可以逐次覆盖：`--coverage-target 87.5`，或 `--coverage-target none`
跑到耗尽（对应配置里的 `target.coverage_target_pct`，默认 90.0）。

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
| `robot.joint_margin_deg` | `5.0` | 每个关节离**可用带**至少再留这么多度 |
| `robot.joint_span_fraction` | `0.9` | 每个轴只用其量程的 90%（以中点为基准缩放），避免顶着限位运行 |
| `robot.j5_min_abs_deg` | `8.0` | 腕奇异保护：`\|J5\|` 不小于这个值 |
| `robot.max_step_deg` | `40.0` | **只报不拦**：相邻交付点的最大关节步，超限计数进报告。实测 58 段里 1 段超 180°、5 段在 150–181°（见下） |
| `robot.wrist_flip_policy` | `T` | 只允许 J5 为负的那一支腕解（见下文），因为程序里的 CONFIG 只能写 `N U T` |
| `robot.soft_limits_deg` / `mechanical_limits_deg` | 见文件 | 两套限位取交集，报告写明用了哪套 |
| `robot.wrist_identified` | `true` | 关掉后禁止输出绝对关节值，门只能做分支排序 |
| `robot.tool_xyz_mm` / `robot.tool_ypr_deg` | 见文件 | `UTOOL 1`：法兰 → 扫描头参考点。换了工具就重标，流程见 [robot-calibration.md](robot-calibration.md) |
| `robot.base_offset_mm` / `upper_arm_mm` / `forearm_mm` / `forearm_offset_mm` / `j2_height_mm` | 见文件 | 臂几何，与图纸互证的实测值；新数据只会复现它，不会改写它 |

两条要记住的语义：

- **连接器的主原语是"一个关节移动"**：从一条带的末端直接 `J` 到下一条带的起点，不插任何中间点。
  关节插值在每个轴上都是线性的，两端合法就不可能在中段出界，所以它既不需要退刀点、也不需要
  CONFIG。开关是 `stroke.joint_connector`（默认开），只在关节门打开且两端都有解时生效。
- **硬门只在没有合法连接时丢条目**。`connector_level` 那套退刀路径现在是兜底：关节直连撞到
  实体的时候才走下去。`stroke.max_rot_deg_per_mm` 只约束这些兜底路径，而且**只报不拦**。
- 丢掉的条目走**排除重选**，不是只过滤；只过滤不重选同样会把覆盖率打下去。

### 为什么把退刀点换成了关节直连

2026-10-29 之前的连接器是"退刀 + 等待 + 空走"，退刀点占整份程序的 46%。实测 99% 方案里 50 个
交付点有 7 个**根本没有逆解**，而这 7 个**全部**是退刀点 / 换向点，27 个真正的扫描点一个都没坏。
也就是说当年的"硬门丢条目"不是腕部带宽不够，是**退刀点自己**把点推进了死区。

换成关节直连之后，`T_G` 在实机上验证过：两条带之间一次 `J`，两点都到位，中间不需要等待点。
原来那句 `no route into it keeps clearance and rate` 把"蹭零件""转向太快""关节托不住"写在一起
的问题也一并拆开了，报告里现在分开计数（`connector_failures`）。

| 键 | 默认 | 含义 |
|---|---|---|
| `stroke.joint_connector` | `true` | 连接器主原语用单个 `J` 移动；关掉回到退刀路径 |
| `stroke.joint_step_deg` | `5.0` | 关节移动扫掠 TCP 路径做间隙检查的采样步长（度） |
| `stroke.max_rot_deg_per_mm` | `0.669` | 只约束退刀兜底路径。实测下界：`X10290` 有一段 173.8°/259.8 mm = 0.669 且实机跑过去了 |

### 交付程序的形态（关节门打开时）

打开关节门之后，`.LS` 不再是一份"全是笛卡尔点 + 一条运动方式"的程序，而是逐点决定的两件事：

| 点 | `motion` | `P[n]` 写法 | 为什么 |
|---|---|---|---|
| 扫描带的起点 | `J` | `J1..J6` | 关节表示是唯一能让我们选的分支真正落到控制器上的形式 |
| 扫描带的终点 | `L` | `X/Y/Z/W/P/R` | 扫掠必须是直线，覆盖率算的就是这条线段 |
| 退刀 / 空走 / 换向点（兜底路径） | `J` | `J1..J6` | 同上，且它们本来就只是"路过" |

混合表示在实机上是验证过的（`T_I`：一个关节表示点带若干笛卡尔点，导入成功、前三点到位）。
关节门关着的时候这些字段全是 `None`，导出照旧用配置文件里的全局运动方式——这是"关门等于回到
旧行为"的实现方式。

### 交付校验永远是开的

`<NAME>_frame.json` 里多了一个 `delivery_check` 块，它**不看 `robot.enabled`**：那个开关描述
的是规划期的筛选，不是这份文件能不能飞。校验器自己建一个全新的 `Robot`（不复用规划器的缓存），
按文件顺序走：关节表示点查限位与 `|J5|`，`L` 移动按起始分支沿直线走到目标，`J` 到笛卡尔点则
只解位姿本身。查不了的情况记成 `unverified`，不当通过。

它**不查碰撞**——那里没有工位数模，规划期的连接器已经带着工位查过了。



### 腕部分支只能选 T（J5 为负）

同一个位姿最多有八组逆解，其中两条只在 **J5 的符号**上不同：示教器把 J5 为负的那条叫作 `T`，
把 J5 为正的那条叫作 `F`。2026-10-29 两次实机实验把这条路堵死了：

| 程序 | 做法 | 结果 |
|---|---|---|
| `T_L` | P1 关节点（J5 = +106）→ P2 同一位置写成笛卡尔 + 常量 CONFIG `N U T` | P1 到位，P2 报 `MOTN-063 位置形态信息不一致`。示教器把 P2 解成 **J5 = −100.294** |
| `T_M` | 同上，但 P2 的 CONFIG 改成逐点 `N U F` | 整份文件**加载就被拒**：`ASBN-023 Configuration error` |

原因很直接：`L` 移动不能换构型，程序声明的是 `T`（J5 负），手臂却停在 J5 正的那一支，控制器
只能报不一致；而 `F` 这个字母连解析都过不去，所以"逐点写正确字母"这条备选路线也不存在。

结论：**规划期就只允许 T 支**，常量 `N U T` 于是永远是对的。代价实测很小（0.26 个百分点，
91.51% → 91.25%），因为两条腕解本来就成对出现。

对照组数据（同一位置、同一姿态）：

| | J4 | J5 | J6 |
|---|---|---|---|
| 我们的 F 支 | −173.502 | +100.257 | −144.242 |
| 我们的 T 支 | 6.498 | −100.257 | 35.758 |
| 示教器算出 | 6.482 | −100.294 | **38.774** |

J1–J5 都在 0.5° 以内，**J6 差 3.0°**。这不像解算差异，更像 `tool_ypr_deg` 的第三个角（绕工具
自身轴的偏转）差了约 3°：工具参考点在工具 Z 轴上，绕 Z 转错不会挪动 TCP 位置，只会让 J6 吸收掉
这个差。滚转差 3° 对扫描通常无害（视场对称），但它意味着"记录的 roll"与"实际飞的 roll"相差
3°。要确认只需一次导入：把某个笛卡尔点切到关节显示，读 J6。

### 软限位是读出来的，不是数据表

数据表（`mechanical_limits_deg`）与控制器实际执行的（`soft_limits_deg`）**不是同一组数**。
2026-10-29 直接从 `MENU → SYSTEM → Axis Limit` 抄了六行，推翻了此前两处推断：

| 轴 | 数据表 | 控制器实测 |
|---|---|---|
| J1 | `±185` | −185.00 .. 185.00 |
| J2 | `−136 .. 136` | **−60.00 .. 76.00** |
| J3 | `−301 .. 301` | **−120.71 .. 180.00** |
| J4 | `±720` | **−360.00 .. 360.00** |
| J5 | `±250` | −125.00 .. 125.00 |
| J6 | `±720` | −360.00 .. 360.00 |

J3 的下界 −120.71 是最要命的一条。旧的 −185 让规划器把 `F`（肩前）分支算成合法——那些解 J3 落在
−175 附近。写进关节块程序后，控制器把 −174.626 按 ±360 换成 +185.374，发现仍超过它自己的 +180
上界，报 `MOTN-017 (G:1, A:2,3)`。示教器上那个 +185.374 不是"另一套约定"，是越界值被等价换算之后
的结果；**`F` 分支根本不可达，是限位写宽了让它看起来可达。**

`robot.joint_span_fraction`（默认 `0.9`）是限位之外的第二道：每个轴只用其量程的 90%，以量程
**中点为基准**缩放（J2/J3 的量程明显不对称，两边各留 10% 才合理），`joint_margin_deg` 再套在里面。
起因是现场发现 J5 = −122° 就报警，而文档限位是 −125°：离限位三度的位姿，机器 hold 不住。
可用带因此是 J1 ±166.5、J2 −53.2..69.2、J3 −105.67..164.96、J4 ±324、J5 ±112.5、J6 ±324。
报告里的 `min_margin_deg` 报的是到**可用带**的距离，所以"≥ 0"意味着整条路径都在安全带内。
细节与代价见 [robot-calibration.md](robot-calibration.md)。

**不要为了让某个位姿通过而放宽这两处**。


`--joint-blocks` 让 `.LS` 的 `P[n]` 直接写关节角，**只对带 `joints_deg` 的点生效**（也就是打开
`robot.enabled` 跑出来的视点文件），没有关节值的点仍写笛卡尔位姿。关节表示的点位必须用 `J`
指令，而 `J` 只收百分比进给——`J P[1] 100 mm/sec` 会被控制器以
`ASBN-009 Undefined instruction` 拒收，所以 `--joint-speed`（默认 `100%`）与 `--speed` 分开，
传线性速度会在导出阶段就报错。

**逐点 CONFIG 默认关闭**（`--per-point-config` 才写）。原因不是保守：一份带计算结果字母
（`N U F, 0, 0, 0`）的程序被 ROBOGUIDE 以 `ASBN-023 Configuration error` 拒收，而同一份文件
把所有点换成常量 `N U T, 0, 0, 0` 就正常导入——字母只验证过"负 J5 配 T"这一半，见
[robot-calibration.md](robot-calibration.md)。想绕开 CONFIG 全部不确定性的做法是写关节块。

```bash
uv run scanplan plan --station input/<工位>.json \
                     --config configs/robot_gate.json --out output/<零件>/
uv run scanplan ls --viewpoints output/<零件>/viewpoints.json \
                   --prog-name PART01 --joint-blocks --joint-speed 20% --out output/<零件>/
```

## 快速模式（`fast` 段）

迭代关节模型时要一条"几分钟就能重跑"的路径，用它换掉每次半小时的等待。**它不是精确模式
的替代品**：数字只描述路径形状，不能当验收依据。

| 键 | 默认 | 含义 |
|---|---|---|
| `fast.enabled` | `false` | 本次运行是快速模式；报告、scope、摘要据此改口径 |
| `fast.no_occlusion` | `false` | 覆盖率阶段不做视线遮挡；必须与 `enabled` 同时为真 |
| `fast.layers` | `[]` | 预设改过哪几层（`sample_grid` / `single_face` / `no_occlusion`），进报告 |

加速不靠新算法，靠**降采样点数**：覆盖率成本是 `带位置数 × 采样点数`，所以真正被调的是
`target.sample_grid_mm`（3 → 10，实测 7.7×）和 `target.face`（`auto` → `upper`，2×）。
`scanplan plan --fast` 就是把这两个已有键写粗一档，因此报告里显示的是实际生效值，而不是
一个无法审计的开关。

两条刻意的边界：

- **不关扫描带**（`stroke.enabled` 不动）。J5 超程发生在带的直线扫掠段上，没有带的路径
  复现不了它。
- **不改 `stroke.check_step_mm`**。它同时是关节筛选对直线移动的采样步长，调粗它等于让
  检查比被检查的运动还粗。

`fast.enabled` 为真而 `sample_grid_mm` 仍小于 8 mm 会被 `_validate` 拒绝——一个还在
3 mm 网格上的"快速模式"只会更慢地被怀疑。`fast.no_occlusion` 单独打开同样被拒绝。

```bash
uv run scanplan plan --station input/<工位>.json --fast --out output/<零件>-fast/
uv run scanplan plan --station input/<工位>.json --config configs/fast.json --out output/<零件>-fast/
```

## 改完看哪里

- `coverage_report.json`：`parameter_provenance`（每个参数的出处）、`surface_profile`
  （bias、派生出的有效值、逐键来源）、`kinematics`（限位来源、J5 直方图、分支直方图、
  min margin、最大步长与超限点数）、`budget`（verdict、每档 λ 的试验记录）、
  `fast_mode`（是否快速模式、改了哪几层、视线遮挡是否建模）；
- `summary.txt`：Joints 与 Budget 两行（没开关节门时不出现 Joints）；快速模式另有一行
  `FAST MODE`；
- 一条守着的不变量：`coverage_pct` **等于**路径里所有覆盖点的并集；`swept_coverage_pct`
  另算过渡段顺手扫到的面积，只作参考、不参与验收；
- 网页端显示的默认值来自 `GET /api/config`，也就是服务端**实际会用的**那份配置——页面说
  扫掠，跑的就是扫掠。

### 三块新读的地方（2026-10-29）

**`coverage_layers` —— 同一个问题问三次。** 只报一个覆盖率数字，出了差距没人知道该改哪一层。

| 键 | 含义 | 实测 |
|---|---|---|
| `geometric_pool_pct` | 候选池几何上能到多少，完全没有机械臂参与 | 91.81 |
| `delivered_pct` | 写出来的程序实际扫到多少 | 91.25 |
| `execution_verified_pct` | 同上，但**只在**每个点都有验证过的关节状态、每条带都沿自己的直线走过之后才给数字；否则是 `null` + 原因 | 91.25 |

差距读法：`pool_to_delivered` 大 = 选带或连接器的问题；`delivered_to_execution` 大 = 机械臂
的问题。**"未验证"永远不给数字**，因为那不是"通过"。门关着时这一项直接写 `null`。

**`failure_attribution` —— 三张表，三个层次。**

| 键 | 说的是 | 实测 |
|---|---|---|
| `sample_reasons` | 为什么某个**采样点**没盖上（几何原因） | `candidate_found 909`、`blocked_by_part 47` |
| `item_drop_reasons` | 为什么整条**带/位姿**离开了程序 | `connector_no_route 5` |
| `connector_primitive_refusals` | 每种连接原语被试过几次、拒了几次 | `collision 10`、`retract 6`、`safe_plane 4` |

第三张表里的"拒绝"**不等于**丢失：关节直连被拒之后走退刀兜底，条目照样交付。把三者混成一句
话，就是原来那句 `no route into it keeps clearance and rate` 的毛病。

**`program.cost` —— 能换成秒的和不能换的分开报。**

| 键 | 实测 |
|---|---|
| `linear_segments` / `linear_mm` / `linear_seconds_at_floor_speed` | 4 段 / 2850 mm / 28.5 s |
| `joint_segments` / `joint_move_deg_total` / `joint_move_deg_worst` | 54 段 / 3462° / 239° |

58 段里有 54 段是关节移动，合计 3462°。这就是"5 个点"和"59 个点"真正的差别所在。**不换算成
秒**：换算需要关节速度，而这一点还没在实机上标定过，编一个数比不给更糟。

