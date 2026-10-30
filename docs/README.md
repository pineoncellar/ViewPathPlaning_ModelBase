# 文档索引

根 [README.md](../README.md) 只做门面，深度内容都在这里。

| 文件 | 内容 |
|---|---|
| [architecture.md](architecture.md) | 仓库结构与模块边界、工位数模的导入与对齐口径、原管线为何曾冻结保留（`reference/` 与 `samples/` 已于 2026-10-30 移除）、逆解与 TSP 为何整层移除。 |
| [cli.md](cli.md) | `scanplan` 五个子命令（`plan` / `ls` / `view` / `pick-datums` / `web`）的用法与常用参数，含 `pick-datums` 配准流程与配准 JSON 格式。 |
| [frame-contract.md](frame-contract.md) | 交付位姿的帧契约：`position` 语义、+Z 光轴、`UTOOL` 教法、两条 `UFRAME` 路径。 |
| [target-surface.md](target-surface.md) | 目标面的拓扑定义，以及为什么所有基于朝向的分面规则在真实数模上都失效。 |
| [configuration.md](configuration.md) | 参数四档出处与三层合并、`surface_bias` 派生表、点数硬上限与 λ 搜索、候选池与扫描带三道门、机器人关节门。 |
| [robot-calibration.md](robot-calibration.md) | 换一批标定数据怎么办：怎么判"臂错了还是只换了工具"、四步拟合法、改哪两处配置、CONFIG 字母怎么推。 |
| [validation.md](validation.md) | 真实工位数模上的验证主表、抽样波动论证、交付默认口径下的覆盖率、工位自检报出的问题、合并后的三档实测。 |
| [stroke-history.md](stroke-history.md) | 扫描带方案的两轮实测：点数为什么没降下来、记账缺陷、以可接受覆盖率换点数的四轮表。 |
| [web-ui.md](web-ui.md) | 浏览器工位界面：从导入数模（网格格式与 STEP/IGES/BREP）、三点配准到下载 `.LS` 的完整流程，以及项目历史、进度反馈与局域网访问。 |
| [viewer.md](viewer.md) | 单文件 3D 视图的图层、视场锥与采样点查询、扫描路径回放。 |
| [prior-work.md](prior-work.md) | 原项目 ViewPathPlaning 的致谢与本仓库相对它的改动对照表。 |
| [tech-stack.md](tech-stack.md) | 全项目技术栈：运行时、依赖分层、算法与数值技术、数据格式、工程链路。 |

私有设计笔记不在本目录的公开部分，也不参与版本控制。
