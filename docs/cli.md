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
`--config`（覆盖配置）、`--coverage-target 97.5`（面积验收线，`none` = 跑到耗尽）、
`--feature-target 95`（**硬特征**验收线：侧壁、折弯、夹具压边，`none` = 关掉这条线）、
`--tracker x,y,z`（给出跟踪器真实位姿）、
`--tracker-sweep`（不给位姿时，报告 192 个候选摆位的差别）、`--no-3d`（不写 3D 视图）、
`--max-faces`（3D 视图里每个模型的显示面数上限）、`--quiet`。
模板见 [`configs/station.example.json`](../configs/station.example.json)。

> **`--face upper` / `--face lower` 是几何名，不是物理名。** 哪一个面叫"上面"取决于
> 分面时定向种子的符号，而那个符号并没有锚定到工位文件上：交换它，两个面的名字会互换，
> 而两面面积、料厚、定向比例**全都一模一样**。报告里因此记录了 `through_axis`（分面认为
> 的"厚度方向"）和一段说明，供人核对。**默认的 `--face auto` 不受影响**，它两面各算一次
> 可达性、取高的那一面，这也是推荐用法。
>
> 网页端不走名字：它把界面里选的「整车主扫描面」轴向传给 `plan(face_axis=...)`，由两个分面
> 的**带符号投影面积**判定该扫哪一侧，判定结果写在报告的 `target_face_selection` 里。
> 命令行没有那个轴向输入，所以才需要 `auto` 或人工核对 `through_axis`。

`plan` 与 `.LS` 导出**刻意分开**：ROBOGUIDE 的坐标系契约（用哪个 `UFRAME`、原点落在
哪里）需要与现场商定，可能只要求重新导出而不必重跑规划。所以 `ls` 只读视点文件，
**不需要数模**。

> **`ls` 会先做关节复核，再写文件。** 它总是读一份配置（`--config`，默认出厂配置），用
> `robot` 段的可用包络（数据表 ∩ 控制器软限位，两端各留 `joint_margin_deg`，默认 5°）逐点检查：
> 视点带 `joints_deg` 就核关节值，
> 没有则在 UFRAME 0 下现场逆解。任何一点无解或超限就**报错退出、一个字节都不写**，
> `<NAME>_frame.json` 的 `joint_audit` 段记录用了哪套包络、几点来自规划、几点现场解得。
> 只想导出一份"审阅用、未经核"的位姿集时，用 `--config` 指一个
> `{"robot": {"enabled": false}}` 的覆盖文件——那时导出会在 `not_modelled` 里明说没核过。

`ls` 仍然要读配置：它是关节审计的依据。出厂配置里 `robot.enabled=true`，导出端会逐点复核
（视点文件带 `joints_deg` 就直接查关节，没有就在 `UFRAME 0` 下现场逆解），**任一点超出可用
包络就报错、不写 `.LS`**，错误信息会指出是哪几个 `P[n]`。只做坐标/格式检查、确实不管可达性时，
显式给一份关掉门的配置：`--config` 指向 `{"robot": {"enabled": false}}`。

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
