# 合并实现说明（2026-10-31）

这份文件记录 `feature/joint-reachability-merge` 相对 `feature/joint-direct-connector`
做了什么、为什么，以及**哪些还没做**。它替代一次性的会议记录：下一个人读它就能知道
每个模块为什么长这样。

分支基线：`feature/joint-direct-connector`（6b72267）。不涉及 `main` 和其他分支。

三支并列来源，下文简称：

- **A** = `feature/modelbase3-update`
- **B** = `feature/joint-audit-and-config`
- **C** = `feature/joint-direct-connector`

---

## 一、五个阶段与数据契约

管线仍然是单向的，但边界现在有名字了，写在 `src/scanplan/contracts.py`：

| 契约 | 谁产出 | 谁消费 | 为什么值得一个类型 |
|---|---|---|---|
| `RobotEnvelope` | 阶段 2 之前，`KinematicGate.envelope` | 阶段 2 的候选筛选、阶段 4 的分支选择、阶段 5 的审计与报告 | 机器人可用域只能有**一个**来源。以前它同时存在于候选筛选和排序代价里，两处各算一次就会漂移 |
| `GateVerdict` | `kinematics.check()` | 全部下游 | 布尔值 + 原因，报告才能按原因分组，而不是重跑一遍 IK |
| `ConnectorResult` | 阶段 3 的连接器 | 阶段 4 | 只给"转角点"不给"验证过的关节"的路由等于没验证；`verified` 属性把这点写死 |
| `DeliveredPoint` | 阶段 3/4 | 阶段 5 | `PathPoint` 与 `Viewpoint` 结构相同，`is_delivered_point` 是运行期检查，不强制转换 |

阶段 4 的插槽不是"再来一个策略"，而是参数：`RobotEnvelope` 是冻结的，任何阶段都改不了它。
这是 E1 那条评审意见的落地方式——**可达域不属于任何插槽**。

## 二、插槽（`src/scanplan/policies.py`）

| 插槽 | 实现 | 配置键 | 默认 |
|---|---|---|---|
| 滚转 | `LockedRolls`（光轴决定滚转 + 门拒绝时按 `ROLL_RETRY_DEG` 回退）/ `DiscreteRolls` | `scanner.roll_mode` | `locked` |
| 连接 | `ConnectorChain`（一级关节直连 → 二级抬升阶梯） | `stroke.connector` | `direct-then-lift` |

- 滚转插槽的默认是 `locked`：实测离散滚转会在带间制造 74–95° 的 J6 转动，对扫描没有收益；
  但锁定会让少数位姿不可行，所以保留固定顺序的回退（45/315/90/…），而不是一刀切。
- 连接插槽只剩两级。旧的 `L3` 安全面路线**已删除**：它的角点是插进去的点，而"7 个不可达点全是
  自插角点"正是重写连接器的原因。抬升阶梯的起步档从 1.0× 降到 0.35×/0.6×（小抬升更容易可达）。
- 两个插槽都只有**一个**生产实现。第二个实现等第二个站点需要时再加；提前建插件层等于给没人跑过的
  代码发长期维护许可证。

## 三、逐项的合并决定

### 阶段 1：几何
- 并入 B 的 `winding.py`（面片绕向修复），在 `scene.load_models` 里、任何下游读 `face_normals`
  之前执行；结果写进 `Station.winding` 和报告的 `part_winding` 块。
- **未并入**：B 的硬特征梯次覆盖、A 的交互补充面（`ScanRegion`）。原因见第五节。

### 阶段 2：候选
- 关节门从"评估期过滤"改成"`_cap` 之前过滤"（`candidates.generate(keep=...)`）。
  旧行为会把预算花在端不住的位姿上：一次交付里 786/4000 个候选是白算的。
- 候选截断从"按 seed 随机抽"改成"每个瞄准点轮流取一个"（`candidates.cap`）：
  随机抽样会让整片区域一个候选都没有，而**没有候选的区域永远不可能被覆盖**。

### 阶段 3：连接器
- 责任链压缩为两级（见上）。`_safe_plane` / `_cross_group` / `_crosses_direction` 一并删除。
- 门关时的行为保持不变（`connector_level == 0` 仍然允许同姿态直飞），
  因为"关门等于回到旧行为"是这个项目的回归口径。

### 阶段 4：分支与读数
- **圈位规则换成 B 的**：`turn_digits` 只在轴自身行程之外才计圈，负圈位返回负值由导出端拒写。
  旧规则 `floor((j+180)/360)` 会给 −284° 的 J6 写 `-1`，而 FANUC 没有负圈位，ROBOGUIDE 整份拒收（ASBN-009）。
- **J6 上限成为价格**：`robot.j6_max_step_deg`（默认 `null`，出厂配置未开启）超过即加
  `_J6_OVER_CAP_COST = 10000` 的罚分。仍然不拦——拦等于丢掉那条带的覆盖率。
- **J6 权重默认值从 1.0 提到 1.5**（B 的值）。它只在"有第二个解"时起作用，不会让不可行的变可行。
- **E3 的兜底**：`ease_over_cap_j6` 在整链 DP 之后跑，只对**非覆盖点**重试滚转（覆盖点的滚转决定
  它扫到哪些采样点，改它等于改已验收的覆盖率）；仍压不下去就记录违规，由速度梯次把它压到 10%。
- 分支过滤本来就是物理的（按 J5 符号过滤，不靠渲染字符串），这一点没有回归：
  实测导出的 4 个点 CONFIG 与 J5 符号 0 处不一致。

### 阶段 5：交付
- **导出前审计**（`adapt.export`）：
  - `validate.check_points`：独立 Robot、无规划器缓存，逐条指令重走一遍；
  - `validate.check_config_hints`：CONFIG 列与它声称的关节是否一致（负圈位、腕字母）；
  - 新增 `joint_pose_mismatch`：关节块与同一点的 XYZ 对不上，说明写的和算的不是同一个点。
- 有任何一个失败 → **不落盘任何 `.LS`**；`--allow-unverified` 才能生成"仅供审阅"的文件，
  并且 `_frame.json` 里记 `joint_audit.allowed_by_override = true`。
- 只有**声称带有joint evidence** 的文件才因可达性被拒：纯位姿交付的可达性按文档约定由 ROBOGUIDE 判定，
  在那里拦会把一条已文档化的流程悄悄废掉。CONFIG 列则永远检查，因为它总是会被写出去。
- 报告升到 `schema_version = 2`：关节步长按**指令类型**统计——`J` 用程序里写的字面值，
  `L` 另列 `cartesian_segments`（直线运动的端点关节差不代表任何真实动作）。
  `max_step_circular_deg` 保留为诊断字段。

## 四、验收

- 测试：**483 passed, 2 skipped**（基线 465，新增 `test_policies.py`、`test_merge_contracts.py`）。
- 端到端（`bench/station.json`、门开、6 mm 网格）：覆盖率 99.94%、4 个交付点、
  连接器全部 `J-direct`、最小余量 8.45°。
- **口径一致性（这次合并的核心验收项）**：
  - 报告 `kinematics.max_step_deg = 48.271` ↔ 文件字面最大步 `48.3`；
  - 报告 `j6.literal_total_deg = 90.0` ↔ 文件字面 J6 总行程 `90.0`；
  - 导出审计 `joint_audit.ok = true`，关节侧与 CONFIG 列 0 失败。
- `scanplan ls --help` 不再因为 help 文本里的裸 `%` 崩溃（C 与 B 都有这个缺陷）。

## 五、还没做的（明确记录，不假装完成了）

1. **硬特征梯次覆盖**（B）与**交互补充面 `ScanRegion`**（A）没有并入。
   两者都要改覆盖率分母、报告 schema 和 3D 视图 payload，属于"阶段 1 一整版"的改动，
   应该单独一个分支做，并与现场核对"覆盖率数字会变"。
2. **连接器"朝机器人水平拉回"第三方向**（B）没有并入；目前抬升只有沿光轴一个方向。
   小抬升档（0.35×/0.6×）已并入，它解决的是同一类问题的一半。
3. **出厂 `coverage_target_pct` 仍是 90**（C 的值）。合并后的分支只有一份出厂配置，
   所以"三支不一致"的问题在这个分支里已经消失；但**没有**改成别的值——那是站点的验收口径，
   不是代码清理。
4. `robot.j6_max_step_deg` 出厂为 `null`（未开启）。打开它只是让排序多一个价格项，
   真正让大动作变安全的是速度梯次，而梯次出厂是开的。

## 六、怎么跑

```powershell
# 环境（与三支对比实验相同：uv + Python 3.10）
uv venv --python C:\ProgramData\anaconda3\envs\bert\python.exe .venv
uv pip install --python .venv numpy==1.23.1 scipy==1.8.1 trimesh==3.12.7 rtree==1.0.0 `
    embreex==4.4.0 "setuptools<70" matplotlib==3.5.3 pytest
uv pip install --python .venv -e . --no-deps

.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\scanplan.exe plan --station ..\bench\station.json --config ..\bench\gate.json --out ..\bench\runs\merge-gate
.venv\Scripts\scanplan.exe ls --viewpoints ..\bench\runs\merge-gate\viewpoints.json --prog-name MERGE --out ..\bench\runs\merge-gate\ls
```
