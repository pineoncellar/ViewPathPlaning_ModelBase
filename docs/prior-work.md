# 致谢与前序工作 (Acknowledgements & Prior Work)

原项目 ViewPathPlaning 的缘起与致谢，以及本仓库在架构、传感器、目标面、求解链路、输出等方面相对它的全部改动。

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
