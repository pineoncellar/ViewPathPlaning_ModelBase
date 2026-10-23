# 架构与模块边界

仓库目录分工、工位数模的导入与对齐口径，以及原管线为什么冻结保留、哪几个阶段被整层移除。

## 仓库结构

| 路径 | 内容 | 是否进版本控制 |
|---|---|---|
| [`src/scanplan/`](../src/scanplan) | 规划器。每个模块的 docstring 写明职责与坑点。 | 是 |
| [`src/scanview/`](../src/scanview) | 3D 视图：把一次规划结果写成单文件网页（含 `viewer.html` 渲染器与逐点覆盖关系）。与规划算法解耦，只读取 `scanplan` 的扫描头坐标系与视场定义。 | 是 |
| [`src/scanweb/`](../src/scanweb) | 本机浏览器工位界面：`scanplan web` 启动的 HTTP 服务与 `index.html`。只调用 `scanplan` 的公开入口（`plan` / `write_outputs` / 配准 / 导出），页面随包分发。 | 是 |
| [`configs/`](../configs) | 配置，含扫描仪参数。 | 是 |
| [`tests/`](../tests) | 测试。夹具数据全部是**合成**的，不含客户几何。 | 是 |
| [`scripts/`](../scripts) | 仓库工具，含数据合规自检。 | 是 |
| `input/` | **工位数模放这里。** 内容被 gitignore。 | **否** |
| `output/` | 规划器输出。内容被 gitignore。 | **否** |
| [`reference/`](../reference) | 冻结保留的原管线，作为回归基线。手册见 [`reference/MANUAL.md`](../reference/MANUAL.md)。 | 是 |
| `docs/` | 公开文档：索引、架构、命令行、配置、验证与历史记录。 | 是 |
| `samples/` | 原项目自带的公开样例网格与图片。 | 是 |

## 工位数模

把零件、夹具、底座的 CAD 放进 `input/`，并保证它不进版本控制——那是客户几何。
如果数模是**逐个文件导出**的 IGES，装配位姿会丢失，所以工位建模里包含一个显式、
带自检的对齐步骤，而不是直接信任文件。

## 原管线与 `.LS` 写入器的关系

[`reference/`](../reference) 冻结保留原管线：`PHASE_1`～`PHASE_6`、集合覆盖求解器、以及
IGES→网格转换器。它作为回归基线保留，同时也是新规划器**移植**（而非重写）FANUC `.LS`
写入器的来源——那个写入器已经解决了 FANUC ASCII 格式里一堆繁琐且不容出错的地方
（固定轴 XYZ 欧拉角、万向锁分支、`/ATTR` 自洽、无前导零的数字格式、程序名合法化）。
手册见 [`reference/MANUAL.md`](../reference/MANUAL.md)。

**本路线不使用 `PHASE_3` / `PHASE_4` / `PHASE_5`**：它们是 UR5e 逆向运动学与机器人基座
采样、TSP 排序那一层，而本方案整层移除了（见根 README 的[「它明确不做」](../README.md#它明确不做)）。代码没动，只作参考。
