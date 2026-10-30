# FANUC 扫描工位：扫描视点自动规划

针对**冲压薄板零件**，在搭载 **TrackScan-Sharp 49** 光学扫描仪的 FANUC 机器人上，自动生成
**扫描头视点**：输入工位数模（零件、夹具、底座、光学跟踪器）与扫描仪参数，输出一组扫描头
位姿（位置 + 光轴 + 滚转），使其共同覆盖零件需要测量的那一面，并导出为 FANUC `.LS` 程序，
供现场在 ROBOGUIDE 中校验可达性。

> **状态：规划器已实现，扫描仪参数仍为占位值。** [`src/scanplan/`](src/scanplan) 下各模块
> 均可单独测试、端到端跑通，并在**真实工位数模**上复现了可行性验证的结论（数字见
> [docs/validation.md](docs/validation.md)）；因为扫描仪参数仍是占位值，覆盖率只能当
> **指示性**参考。

## 它做什么

1. **工位建模** —— 载入零件、夹具、底座与跟踪器，统一到同一 CAD 坐标系；对齐过程显式且带自检，不盲信数模文件。
2. **目标面分割** —— 按**拓扑**把零件分成"原始平整铁皮的两个面"，只采样待测的那一面。
3. **生成候选视点** —— 围绕目标面生成候选位姿，逐条判定视场、量程、入射角、遮挡、扫描头能否放得下、以及跟踪器是否看得见。
4. **选取最小覆盖集** —— 在满足覆盖率前提下选出最少的视点。
5. **（默认）连成扫描带** —— 现场是**一路扫过去**，不是"停稳再拍"；`stroke.enabled` 打开时规划单元是"直线段 + 恒定姿态"，输出一条带过渡段的连续路径（`L ... CNT0`），网页端可切回纯位姿模式。
6. **导出** —— 视点集、覆盖报告、一个自包含的 3D 视图，以及 FANUC `.LS` 程序与其帧契约。

## 它明确不做

机器人碰撞与运动时间优化。导出的 `.LS` 默认**只含笛卡尔位姿**，由控制器 / ROBOGUIDE 求解，
所以交付物是视点集而不是机器人程序。

逆解与关节限位有一个例外：`robot.enabled`（默认关闭）打开后，规划期会用一套由**实测位姿
标定**的臂/腕模型先筛一遍，交付点逐点带 `CONFIG` 与 J5 余量。它只用来避免把明显不可行的
位姿交出去，不替代 ROBOGUIDE 的可达性与碰撞判定。

## 快速开始

需要 Python 3.10 与 [uv](https://docs.astral.sh/uv/)。

```bash
uv sync --extra dev --extra embree
uv run scanplan plan --station input/<工位>.json --face auto --out output/<零件>/
uv run scanplan web                 # 浏览器工位界面，默认 http://127.0.0.1:8765
```

`embree` 是可选项，但强烈建议装：它把光线求交加速约三个数量级，直接决定真实零件
"能不能在可接受时间内算完"。

其余子命令、配准与全部参数见 [docs/cli.md](docs/cli.md)；浏览器界面的完整流程见
[docs/web-ui.md](docs/web-ui.md)；配置怎么改见 [docs/configuration.md](docs/configuration.md)。
提交前请读 [CONTRIBUTING.md](CONTRIBUTING.md)：工位数模与派生数据绝不入库。

## 交付物

`plan` 在输出目录写下八个文件（`--no-3d` 时六个）：

| 文件 | 内容 |
|---|---|
| `viewpoints.json` | 视点集：位置 + 光轴 + 滚转 + 覆盖点数/面积 + 入射角 + `tracker_ok` + 来源，以及程序顺序 `order` / 所属带 `stroke_id` / 角色 `role`；关节门打开时另带 `config` / `branch` / `joints_deg` / `j5_margin_deg` |
| `viewpoints.csv` | 同一份数据，便于 Excel 目视 |
| `exempt_points.csv` | 被夹持区豁免的点云（供人工核对 CAD 夹持区与实物是否一致） |
| `unreachable_points.csv` | 漏扫点云，**每点带原因**（`blocked_by_part` / `blocked_by_fixture` / `blocked_by_base` / `blocked_by_tracker` / `never_in_view` / `only_poses_that_collide` / `unreachable_joint_limit` / `beyond_arm_reach` / `no_line` / `not_facing`） |
| `coverage_report.json` | 覆盖报告：覆盖率、视点数、入射角分布、漏扫聚类、参数出处（数据表 / 推导 / 未实测）、路径与中断统计、以及**它没有建模什么**；另含 `surface_profile` / `budget` / `kinematics` 三块（见[配置与修改指南](docs/configuration.md#改完看哪里)） |
| `summary.txt` | 上面那些数字的人读版本 |
| `viewpoints_3d.html` | **单文件 3D 视图**：零件/夹具/底座 + 视点位置、光轴与视场锥 + 已覆盖/未覆盖采样 + 程序路径与回放 |
| `target_samples.npz` | 采样点 + 逐点覆盖位掩码，供 3D 视图回答“哪些视点能扫到这个点”；`scanplan view` 重新渲染时读它 |
| `strokes.json` | **只在扫描带模式下**：程序顺序的权威来源，每条带的端点、姿态、覆盖贡献与角色 |

`ls` 写下 `.LS` + `_points.csv` + `_frame.json`（帧契约）。视点文件里带 `stroke_id`
时，`ls` **固定用配置里的 `L <速度> CNT0` 并按文件顺序下发**，并拒绝任何重排请求：
重排会把一条带的两端拆成两次互不相关的移动，而覆盖率是对那条整带算的。

帧契约（`position` 语义、+Z 光轴、`UTOOL` 教法、两条 `UFRAME` 路径）详见
[docs/frame-contract.md](docs/frame-contract.md)；3D 视图的图层与回放见
[docs/viewer.md](docs/viewer.md)。

## 目标面：拓扑定义

零件是一块冲压出来的薄铁片，所以按广义的面来说它**只有两个面**——就是它还是平整铁皮时的
两个表面。成型侧壁属于它由哪一面折出来就归哪一面；冲裁（冲孔/落料）出来的剪切断面
**两个面都不属于**；夹具夹持的区域不要求测量。基于朝向的规则在这里**全都行不通**（零件
倾斜即失效、折痕会把一个面切成多块、贯穿轴夹角会切掉侧壁、真实网格的重合顶点从未合并）。

详见 [docs/target-surface.md](docs/target-surface.md)，有效方法记录在
[`src/scanplan/sheet_split.py`](src/scanplan/sheet_split.py)。

## 报告里必须一起读的三条声明

覆盖面**只是几何可达性**。报告里会显式写明下面三件事，任何引用覆盖率数字的场合都应带上：

1. **机器人可达性默认不在内**：逆解、关节限位、机器人碰撞由现场在 ROBOGUIDE 里判，所以交付的
   `.LS` **可能包含 ROBOGUIDE 判为不可达的点位**——这是预期，不是缺陷。ROBOGUIDE 报
   不可达时**先换 `CONFIG` 构型再试**。打开 `robot.enabled` 后规划期会先用实测标定的臂/腕
   模型筛一遍并按分支排 `CONFIG`，但那仍然只是参考：它不替代 ROBOGUIDE，也不代表现场软限位
   已经标定。
2. **不含机械臂对跟踪器的遮挡**：本阶段没有机器人模型，`tracker_ok` 只对零件/夹具/底座
   判遮挡。现场架设时请同时避开机械臂的常态工作扇区。
3. **哪些参数还是占位值**：报告逐条列出，`max_incidence_deg` 排在最前。

## 文档

| 文件 | 内容 |
|---|---|
| [docs/README.md](docs/README.md) | 文档索引，先看这一页。 |
| [docs/architecture.md](docs/architecture.md) | 仓库结构与模块边界；原管线为何冻结保留，逆解与 TSP 为何移除。 |
| [docs/cli.md](docs/cli.md) | `plan` / `ls` / `view` / `pick-datums` / `web` 全参数。 |
| [docs/frame-contract.md](docs/frame-contract.md) | 帧契约：`position` 语义、+Z 光轴、`UTOOL` 教法、`UFRAME` 两条路。 |
| [docs/target-surface.md](docs/target-surface.md) | 拓扑定义，以及为什么朝向规则全失效。 |
| [docs/configuration.md](docs/configuration.md) | 参数四档出处、`surface_bias` 派生、`robot` 段、扫描带三道门、`pool_mode`。 |
| [docs/validation.md](docs/validation.md) | 验证主表、抽样波动的论证、已知局限。 |
| [docs/stroke-history.md](docs/stroke-history.md) | 记账缺陷、点数骤减四轮表、取舍记录。 |
| [docs/web-ui.md](docs/web-ui.md) | 浏览器工位界面：完整流程、CAD 导入、项目历史、局域网访问。 |
| [docs/viewer.md](docs/viewer.md) | 3D 视图的图层、回放与查询。 |
| [docs/prior-work.md](docs/prior-work.md) | 原项目 vs 本仓库对比表与致谢。 |
| [docs/tech-stack.md](docs/tech-stack.md) | 全项目技术栈。 |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 贡献流程与数据保密要求。 |

## 致谢

本项目最初的设计灵感与核心算法流程参考了开源项目
**[MobMonRob/ViewPathPlaning](https://github.com/MobMonRob/ViewPathPlaning)**（Apache-2.0），
本仓库已完全脱离它成为独立项目。两者的逐项对比与更详细的缘起见
[docs/prior-work.md](docs/prior-work.md)。

## 许可证

[Apache License 2.0](LICENSE)。
