# FANUC 扫描工位：扫描视点自动规划

针对**冲压薄板零件**，在搭载 **TrackScan-Sharp 49** 光学扫描仪的 FANUC 机器人上，
自动生成**扫描头视点**。

输入工位数模（零件、夹具、底座、光学跟踪器）与扫描仪参数，输出一组扫描头位姿
（位置 + 光轴 + 滚转），使其共同覆盖零件需要测量的那一面；并导出为 FANUC `.LS`
程序，供现场在 ROBOGUIDE 中校验可达性。

> **状态：早期开发中。**
> 包结构、模块接口与导出数据契约已就位；**规划算法尚未实现**。
> [`src/scanplan/`](src/scanplan) 下每个模块目前只有 docstring，写明它的职责、
> 以及**容易做错的地方**。这些约束背后的设计笔记为私密保存，因为其中含客户零件尺寸。

## 它做什么

1. **工位建模** —— 载入零件、夹具、底座与跟踪器，统一到同一 CAD 坐标系；对齐过程
   显式且带自检，不盲信数模文件。
2. **目标面分割** —— 按**拓扑**把零件分成"原始平整铁皮的两个面"，采样待测的那一面
   （见下节）。
3. **生成候选视点** —— 围绕目标面生成候选位姿，逐条判定视场、量程、入射角、遮挡、
   扫描头能否放得下、以及跟踪器是否看得见。
4. **选取最小覆盖集** —— 在满足覆盖率前提下选出最少的视点。
5. **导出** —— 视点集、覆盖报告，以及 FANUC `.LS` 程序与其帧契约。

## 它明确不做

逆解、关节限位、机器人碰撞、视点排序、运动时间优化。导出的 `.LS` **只含笛卡尔位姿**，
由控制器 / ROBOGUIDE 求解。这也是本方案不需要机器人模型的原因，以及为什么交付物是
视点集而不是机器人程序。

## 目标面：拓扑定义

零件是一块冲压出来的薄铁片，所以按广义的面来说它**只有两个面**——就是它还是平整铁皮时
的两个表面。成型侧壁属于它由哪一面折出来就归哪一面。冲裁（冲孔/落料）出来的剪切断面
**两个面都不属于**，因为它是"切开材料"产生的，不是铁皮的表面。

```
目标 = 原始铁皮的两个面之一
       **含**成型侧壁、含凹腔/凹槽内壁
 减去 = 夹具夹持的区域（不要求测量）
排除 = 剪切断面（冲裁/冲孔边）
排除 = 铁皮的另一面
```

基于朝向的规则在这里**全都行不通**，原因各不相同：`normal_z > 0` 在零件于夹具中倾斜时
立刻失效；二面角分割出的 patch 不等于一个面，因为折痕会把一个面切成好几块平面；用
"与零件自身贯穿轴的夹角阈值"会把成型侧壁切掉，而侧壁属于该面；而"去掉冲裁边后取连通
分量"在真实 CAD 网格上直接失败——各张曲面是分别网格化的，重合顶点从未合并。

有效的方法记录在 [`src/scanplan/sheet_split.py`](src/scanplan/sheet_split.py)。

## 仓库结构

| 路径 | 内容 | 是否进版本控制 |
|---|---|---|
| [`src/scanplan/`](src/scanplan) | 规划器。每个模块的 docstring 写明职责与坑点。 | 是 |
| [`configs/`](configs) | 配置，含扫描仪参数。 | 是 |
| [`tests/`](tests) | 测试。夹具数据全部是**合成**的，不含客户几何。 | 是 |
| [`scripts/`](scripts) | 仓库工具，含数据合规自检。 | 是 |
| `input/` | **工位数模放这里。** 内容被 gitignore。 | **否** |
| `output/` | 规划器输出。内容被 gitignore。 | **否** |
| [`reference/`](reference) | 冻结保留的原管线，作为回归基线。手册见 [`reference/MANUAL.md`](reference/MANUAL.md)。 | 是 |
| `docs/` | 私密设计文档——含客户零件尺寸。 | **否** |
| `samples/` | 原项目自带的公开样例网格与图片。 | 是 |

## 快速开始

需要 Python 3.10 与 [uv](https://docs.astral.sh/uv/)。

```bash
uv sync --extra dev --extra embree
uv run pytest
```

`embree` 是可选项，但强烈建议装：它把光线求交加速约三个数量级，直接决定真实零件
"能不能在可接受时间内算完"。

只有一个入口：

```bash
# 这是预期接口 —— 尚未实现
uv run scanplan plan --part P.stl --fixture F.stl --base B.stl \
                     --config configs/scanner_trackscan49.json \
                     --out output/<零件>/

uv run scanplan ls   --viewpoints output/<零件>/viewpoints.json \
                     --prog-name PART01 --out output/<零件>/
```

`plan` 与 `.LS` 导出**刻意分开**：ROBOGUIDE 的坐标系契约（用哪个 `UFRAME`、原点落在
哪里）需要与现场商定，可能只要求重新导出而不必重跑规划。

### 工位数模

把零件、夹具、底座的 CAD 放进 `input/`，并保证它不进版本控制——那是客户几何。
如果数模是**逐个文件导出**的 IGES，装配位姿会丢失，所以工位建模里包含一个显式、
带自检的对齐步骤，而不是直接信任文件。

## 配置

[`configs/scanner_trackscan49.json`](configs/scanner_trackscan49.json) 存放扫描仪参数。
**其中的值是占位值**：来自公开参数表，以及 "基准距""景深"这两个词的字面几何含义，
**不是**在真机上实测的。其中 `max_incidence_deg`（最大入射角）敏感度远高于其他任何
一个，而它的真值目前未知。

## 数据保密

工位数模、由它派生的一切（网格、点云、视点坐标）以及设计文档**绝不入库**。
`input/`、`output/`、`docs/`、`spike/`、`reference/00_Output Collection/` 全部被
gitignore，并且有自检脚本会在这些东西**一旦被跟踪**时报错：

```bash
uv run python scripts/check_repo_hygiene.py
```

提交前请先跑。被忽略的文件之后仍然可以用 `git add -f` 加进来，但**已经推送出去的
收不回来**。

## 目前的验证情况

设计阶段用**真实工位零件 + 占位扫描仪参数**做过一轮可行性验证，模块 docstring 里的
约束就是从那里得来的。有两条结论影响了设计：

- **固定的单台跟踪器本身不构成瓶颈**，前提是它沿**零件薄轴**方向架设，而不是正对大面
  看过去——机理是零件本体挡住了**扫描头**，不是跟踪器视场角不够；
- 残余漏扫是**二十来处小特征**（窄槽、小凹坑内壁），而不是一整片够不到的区域。这指向
  "下一步的收益在候选视点密度"，而不是"换更好的集合覆盖算法"。

这些数字依赖占位的 `max_incidence_deg`，只能当**指示性**参考，因此这里刻意不写出具体数值。

## 原管线（reference/）

[`reference/`](reference) 冻结保留原管线：`PHASE_1`～`PHASE_6`、集合覆盖求解器、以及
IGES→网格转换器。它作为回归基线保留，同时也是新规划器**移植**（而非重写）FANUC `.LS`
写入器的来源——那个写入器已经解决了 FANUC ASCII 格式里一堆繁琐且不容出错的地方。
手册见 [`reference/MANUAL.md`](reference/MANUAL.md)。

## 致谢与前序工作 (Acknowledgements & Prior Work)

本项目最初的设计灵感与核心算法流程参考了开源项目 **ViewPathPlanning**（基于 UR5e 与
3D 扫描仪的模型视点与机器人路径规划流水线）。

感谢原作者在硕士研究工作中所构建的基础原型与探索：

- **原项目仓库**：[MobMonRob/ViewPathPlaning](https://github.com/MobMonRob/ViewPathPlaning)
- **开源协议**：[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
- **算法原型**：原项目建立了 Phase 1 至 Phase 5 的完整管线（体素化视点生成、视点集合
  覆盖 SCP、机器人基座可达性逆解、基座集合覆盖及 TSP 路径规划）。

**本仓库的重构与演进说明**：

本仓库已完全脱离原 Fork 成为独立项目。我们在保留其核心阶段思想的基础上，对项目架构、
求解链路与交付形态进行了大规模重写与重构：

| 方面 | 原项目 | 本仓库 |
|---|---|---|
| 项目架构 | 五个阶段串行、以 JSON 文件交接的独立脚本 | 模块化解耦的 Python 包（`src/` 布局），各模块可单独测试 |
| 传感器 | 固定式轮廓仪，装在机械臂末端、无外部跟踪器 | **跟踪式**扫描仪（外置光学跟踪器），新增"跟踪器可见性"约束 |
| 目标面 | 零件全部朝外的外表面 | **拓扑定义**：原始铁皮的一个完整面，含成型侧壁、排除冲裁边 |
| 求解链路 | 含 UR5e 逆解、关节限位、机器人基座采样与 TSP | **移除**；只交付扫描球视点集，可达性交由 ROBOGUIDE 校验 |
| 输出 | 机器人位置坐标（体素单位）+ 关节构型，JSON | 扫描头笛卡尔位姿 + FANUC `.LS` + 帧契约 + 覆盖报告 |
| 光线求交 | 纯 Python | Embree 加速（约三个数量级） |
| 数据隔离 | — | 客户数模及派生数据永不入库，附自检脚本 |
| 原管线 | 仓库根目录下的当前实现 | 冻结保留在 `reference/`，作为回归基线 |

*This project was originally inspired by and built upon the workflow from
**[MobMonRob/ViewPathPlaning](https://github.com/MobMonRob/ViewPathPlaning)**,
licensed under Apache-2.0.*

*While the fundamental 5-phase view-path-planning concept stems from the original
master's thesis implementation, this repository has been comprehensively
refactored and redesigned, and is maintained as an independent codebase.*

## 许可

Apache License 2.0 —— 见 [LICENSE](LICENSE)。
