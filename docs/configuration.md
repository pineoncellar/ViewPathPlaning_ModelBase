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

## 候选姿态的倾角扇（`scanner.tilt_azimuths`）

每个瞄准点先按 `distances_mm` × `tilts_deg` × `rolls_deg` 生成"梯子"，`tilt_azimuths` 决定
这架梯子在表面法线周围**转几个方位**（默认 4 → 8 个倾斜方向 + 1 个正对方向）。它存在的理由
是实测的：2026-10-28 那次交付只覆盖 57.7%，我把未覆盖的采样点逐点反解，**91% 的点其实存在
合法姿态**——只是旧的单平面梯子（只绕一个切向 ±30°）offer 不了那些方向。

关节门在**截断之前**过滤候选池（`generate(..., keep=门)`），截断本身按"每个瞄准点轮流取一个"
分配（`candidates._cap`），所以 4000 的预算不会被少数瞄准点吃光、也不会留给手臂端不住的位姿。

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
`robot` 段默认 `enabled: true`：规划期就用实测的臂/腕模型筛位姿——候选点要在限位里留出余量、
过渡点要有逆解，交付点逐点带 CONFIG；**解不出关节解的点会被当作“无合法连接”整条丢掉并重选**，
`.LS` 导出前还会再核一遍，任一点超出可用包络就**直接拒写**（不生成半个程序）。
想复现 2026-10-27 之前的结果，在覆盖文件里写 `"robot": {"enabled": false}`。

| 键 | 默认 | 含义 |
|---|---|---|
| `robot.enabled` | `true` | 总开关。关掉后交付点不再核关节，`.LS` 导出也失去关节证据（只剩一条警告） |
| `robot.joint_margin_deg` | `5.0` | **每个轴两端各预留多少度**（现场接受 5–10）。硬门要求交付点离限位 ≥ 这个值 |
| `robot.j5_min_abs_deg` | `8.0` | 腕奇异保护：`\|J5\|` 不小于这个值 |
| `robot.max_step_deg` | `40.0` | **只报不拦**：相邻交付点的最大关节步，超限计数进报告 |
| `robot.elbow_margin_deg` | `10.0` | 肘部离开自身行程两端的余量。`bend→0`（两杆折叠贴合）与 `bend→180`（伸直）是共线姿态：闭式解还会给数，控制器却直接报 `MOTN-018 位置不可达`（2026-10-29 的 PART010 就卡在 `bend = 8.4°` 的点上）。`Robot.feasible` 也用它，所以导出前的关节复核同样会拒这类点 |
| `robot.motion_check_step_mm` / `_deg` | `50` / `10` | 逐段检查一段运动时的采样步长（更紧的那个生效）；`J` 运动按凸包络只查端点 + J5 不过零，`L` 运动逐点采样 |
| `robot.limits_usage_frac` | `1.0` | 可以用数据表包络的多大比例。**现已设 1.0**：不再用“90% 量程”当软限位——那是每轴不同的余量（J1 18.5°、J2 6.8°、J4 36°），机器本身没有这个道理；余量统一交给 `joint_margin_deg`（5–10°） |
| `robot.soft_limits_deg` / `mechanical_limits_deg` | 见文件 | 数据表限位（R-2000iC/125L，reach 2655）与控制器软限位，两者取交集；报告写明用了哪套、用了多少比例 |
| `robot.wrist_identified` | `true` | 关掉后禁止输出绝对关节值，门只能做分支排序 |
| `robot.config_wrist_letter` | `"T"` | CONFIG 第三位。`"auto"` = 按 J5 符号（`T` ↔ J5 < 0，2026-10-23 实测）；`"T"`/`"F"` = 钉死。**本现场必须 `"T"`**：ROBOGUIDE 对 `N U F, 0, 0, 0` 报 `ASBN-023 Configuration error` 并丢掉整份程序。两个腕分支是同一个 TCP 姿态（J5 取反、J4/J6 转 180°），所以钉死不影响扫描几何，只影响能不能导入 |
| `robot.tool_xyz_mm` / `robot.tool_ypr_deg` | 见文件 | `UTOOL 1`：法兰 → 扫描头参考点。换了工具就重标，流程见 [robot-calibration.md](robot-calibration.md) |
| `robot.base_offset_mm` / `upper_arm_mm` / `forearm_mm` / `forearm_offset_mm` / `j2_height_mm` | 见文件 | 臂几何，与图纸互证的实测值；新数据只会复现它，不会改写它 |

两条要记住的语义：

- **CONFIG 的第三位是约束，不是标签。** `robot.config_wrist_letter` 钉成 `T` 时，求解阶段就**只接受** J5 < 0 的那支解。以前是“把 F 分支的关节值写成 `N U T`”——控制器按 T 去解，J4/J6 差 180°，解不出或超行程就报 `MOTN-018`。2026-10-29 的 PART010 有 13/27 个点是这种“字不符实”的点。
- **硬门对三件事丢条目**：间隙 ≤ 0、过渡点解不出逆解、**交付点没有任何落在可用包络内的关节解**。
  `stroke.max_rot_deg_per_mm` 是未标定常数，**只报不拦**——拿它丢带会把覆盖率从 99% 打到 82%。
- 丢掉的条目走**排除重选**，不是只过滤；只过滤不重选同样会把覆盖率打下去。
- 导出端还有一道闸：`roboguide.adapt.export(..., robot=cfg.robot)` 对每个写进 `.LS` 的点
  复核（优先用规划给的 `joints_deg`，没有就现场逆解），**超限就抛错不写文件**，并在
  `<NAME>_frame.json` 的 `joint_audit` 段留下证据与所用包络。

`--joint-blocks` 让 `.LS` 的 `P[n]` 直接写关节角，**只对带 `joints_deg` 的点生效**（也就是开过
关节门的视点文件），没有关节值的点仍写笛卡尔位姿。CONFIG 的三个圈数位现在**恒为 0**（只在轴
自身行程之外才计圈），负圈位会被 `roboguide.ls.check_config()` 直接拒绝——2026-10-27 之前就是它
让 ROBOGUIDE 以 `ASBN-009 Configuration error` 拒收整份 `T1.LS`。`T/F` 字母已在 2026-10-23 那批
8 组数据上核对过（`T` ↔ J5 < 0）；ROBOGUIDE 是否配了数据表限位仍要在现场确认。

```bash
uv run scanplan plan --station input/<工位>.json \
                     --config configs/robot_gate.json --out output/<零件>/
uv run scanplan ls --viewpoints output/<零件>/viewpoints.json \
                   --prog-name PART01 --joint-blocks --out output/<零件>/
```

## 硬特征（侧壁、折弯、夹具边）的第二条验收线

面积验收线管不了侧壁：1 m² 的面板上，一块立壁只占千分之几，97% 的计划把每一块侧壁都漏掉也照样
"达标"。所以 `target.feature_coverage_pct`（默认 `95.0`）是**独立分母**的第二条线——它只算硬特征：

- 法向偏离整面面积加权平均法向超过 `target.feature_wall_deg`（默认 45°）的样本=立壁/折弯；
- 离夹具或底座小于 `target.feature_clearance_mm`（默认 15 mm）的样本=被夹具压住的边缘。

两条线**都满足**贪心才会停；修剪（prune）也不许把硬特征覆盖率剪回去。报告 `features` 块给出
`covered_pct` / `area_mm2` / `uncovered_by_reason` / `uncovered_regions`。`feature_coverage_pct`
写 `null` 就退回"只看面积"的老行为。

面积目标已经满足、硬特征还不满足时，`scanner.feature_rounds`（默认 1）轮**斜扫补充**会跑：只对
剩下的硬特征样本，按 `scanner.feature_tilts_deg`（默认 `[30, 40]`，**必须小于
`max_incidence_deg`**）绕法向扫一圈 `scanner.feature_tilt_azimuths`（默认 12）个方位、两种符号，
瞄准网格 `scanner.feature_aim_grid_mm`（默认 15 mm，比主网格细，因为侧壁只有几毫米宽）。这一族
**只用 roll 0**：斜扫本来就是折中，再加 90° 滚转等于白送一次 90° 的 J6 转动。

```json
{"target": {"feature_coverage_pct": 95.0}}
```

## J6 与"整个过程都在行程里"

`stroke.max_j6_step_deg`（默认 `60`）是单次重定位允许的 J6 转角，`stroke.j6_step_weight`
（默认 `1.5`，旧行为是常数 0.5）是分支搜索里"一度 J6 值多少度其它轴"。扫描带内 J6 恒为零——
一条带就是一个姿态写成两个角点——所以程序里每一次 J6 都属于带间重定位，超限时写进
`CONSTRAINT` 与 `kinematics.j6_steps_over_limit`：**只报不丢**（和 `robot.max_step_deg` 同一
性质），因为没有任何控制器数据支持一个硬阈值。分支搜索对"超过 cap 的 J6"加一个远大于任何真实
代价的罚分，所以只要有别的分支就不会撞上去。

角点合法不等于**运动**合法：`robot.motion_check_step_mm`（默认 50）与
`robot.motion_check_step_deg`（默认 10）把每一段交付运动按更紧的那个步长采样，IK 从**上一点所在
的分支**一路跟下去（控制器不会每点重挑分支），中途离开可用包络就拒这条路线——间隙和行程是同一种
硬约束。`J` 运动不用采样：包络是凸盒，两端在内则整段在内（只有 J5 过零的腕奇异要单独判）；
`L` 运动必须采样，因为中间的关节值是没检查过的位姿算出来的。

报告 `kinematics.motion_check` 给出 `moves_checked` 与 IK 采样数，summary 里是一行 `moves`。

`motion.joint_speed`（默认 `"5%"`）是**不扫描的那些关节运动**的速度——扫描带方案里每一个
回缩、过渡、重定位指令都是 `J`，也是腕部真正转动的地方（单条指令里 J6 最多能转 180°）。必须
写百分比：FANUC 不接受关节指令上的 mm/sec。字段同时写进 `<NAME>_frame.json` 的
`program.joint_speed`，重定位导出沿用它。

`stroke.lock_roll` 不只管扫描带：位姿梯子也用同一个 `roll_for_axis` 约定。以前位姿池同时有
roll 0 和 90，会和 roll 90 的扫描带背对背，量出来是 74–95° 的 J6 停顿式转动；统一之后 J6 只跟
轴向变化走。

但滚转不只是"视场方不方"的问题：拟合出来的工具长 635 mm，绕光轴转一圈等于把**法兰**在 0.6 m
的圆上搬一圈，所以同一个视线、同一个位置，换一个滚转可能就是"够得着"和"够不着"的区别。
站点 20261030_105857 上锁定的那个滚转对 1.4% 的目标面不可达，报告当时说这些点"没被瞄准过"
（`not_facing`），其实视线完全正常。所以现在关节门拒掉一个位姿时，会**绕同一根光轴**再试
若干滚转（`candidates._ROLL_RETRY_DEG`，先试最小的偏转），只有全都不行才丢：

- 门第一次就接受的位姿原封不动，锁定滚转本来够用的计划逐位不变；
- 换滚转不改变扫描几何（同一条视线），只改手腕位置，代价是相邻点的 J6 差可能变大——
  实测最差 J6 从 46° 升到 104–123°（仍只报不丢），换来覆盖率 98.56% → 99.15%、
  交付点 87 → 30；
- 真的所有位姿都够不着时，漏扫原因写 `beyond_arm_reach` / `unreachable_joint_limit`，
  而不是把锅丢给候选生成。

## 回缩点：先保证"拿得住"

连接路线里的等待点（回缩点）是硬门最常卡住的地方，而它卡住时的报错原话是
"no retract corner the arm can hold"——**这句话只说症状**。2026-10-29 station 上它一口气丢了
4 个点位（8.5 个百分点的覆盖率），真正的原因分两类，现在各有对策：

1. **回缩方向只往外/往上**。沿 `-axis`（垂直向上）回缩 80 mm，在 2 m 半径的远端点位等于要求
   肘部再伸直一点——而那正是远端最缺的行程。现在 `_retract_directions` 在"沿轴后退"和
   "安全平面法向"之后**还会朝机器人方向水平拉回**（`-unit(x, y)`）：距离变短而不是变长，间隙
   照样按同一套 `_segment_margin` 判，且放在最后试，不会让已经能走的路线变差。
2. **只有比 `retract_clearance_mm` 更短的抬升才拿得住**。`_LIFT_FACTORS` 现在也往 1 以下走
   （`1, 2, 3, 4, 6, 8, 0.6, 0.35`）：回缩点只需要把头从零件上抬开，20 mm 比 80 mm 更容易在
   包络里；往上的大抬升仍然先试，所以原来能走的不会变。报告里"up to N mm of lift"用的是
   抬升阶梯的**上限**（`_MAX_LIFT_FACTOR`），不是最后一档。

硬门拒掉一条路线时，`path._refusal_note` 会在 `CONSTRAINT` 里写清**是哪一段、什么原因**
（哪个角点没有逆解 / 哪一段 `J`/`L` 中途离开包络），所以下一次看报告不必再猜。

位姿交付（99% 这类高目标下 `require_sweeps` 常常回落到位姿方案）还有一条独立规则：被硬门拒掉
的点不再"从清单里划掉"，而是**带上排除掩码重选**（`selection.select(exclude=)` +
`delivery._pose_reselect`）。一个位姿大约 3% 面积，划掉一个就是三个百分点，而重选会把它换成
另一个能连上的点。

## 改完看哪里

- `coverage_report.json`：`parameter_provenance`（每个参数的出处）、`surface_profile`
  （bias、派生出的有效值、逐键来源）、`kinematics`（限位来源、J5 直方图、分支直方图、
  min margin、最大步长与超限点数）、`budget`（verdict、每档 λ 的试验记录）；
- `summary.txt`：Joints 与 Budget 两行（没开关节门时不出现 Joints）；
- 一条守着的不变量：`coverage_pct` **等于**路径里所有覆盖点的并集；`swept_coverage_pct`
  另算过渡段顺手扫到的面积，只作参考、不参与验收；
- 网页端显示的默认值来自 `GET /api/config`，也就是服务端**实际会用的**那份配置——页面说
  扫掠，跑的就是扫掠。
