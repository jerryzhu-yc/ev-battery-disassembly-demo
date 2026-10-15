<div align="center">

# EV Battery Disassembly Workcell

### 基于 UR5e + MuJoCo + OrcaStudio 的双电池包自动拆解仿真工作站

面向 EV 电池包自动化拆解场景，构建包含 **输送线、UR5e 机械臂、真空吸盘、分类料箱、空托盘回收区与工业工作站外围设施** 的连续拆解仿真 Demo。

[![Sanity Check](https://github.com/jerryzhu-yc/ev-battery-disassembly-demo/actions/workflows/sanity-check.yml/badge.svg)](https://github.com/jerryzhu-yc/ev-battery-disassembly-demo/actions/workflows/sanity-check.yml)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![MuJoCo](https://img.shields.io/badge/Physics-MuJoCo-2F6F9F)
![Robot](https://img.shields.io/badge/Robot-UR5e-5B5B5B)
![Status](https://img.shields.io/badge/Status-Research%20Demo-orange)

</div>

---

## 项目简介

本项目实现一个 **双电池包连续自动拆解工作站**。

UR5e 六自由度机械臂搭载 SMC 风格真空吸盘，对到达拆解工位的电池包依次完成：

- 电池上盖拆除
- 两组汇流排拆除
- 四个电芯拆除
- 不同类别零件分类放置
- 第一包与第二包连续切换
- 拆解后的空托盘输送与回收流程

机器人控制侧结合 **Cartesian TCP 控制、6D Damped Least Squares IK、Minimum-Jerk 轨迹插值、自适应安全高度与放置结果验证**，并通过 OrcaStudio / gRPC 与 MuJoCo 仿真场景联动。

当前仓库中的 `scene_two_pack.xml` 已进一步扩展为更完整的工业工作站场景，包括安全警戒区域、防护围栏、电控柜、HMI、三色状态灯、急停站、工作站标识以及工业照明等外围设施。

> 本项目当前定位为 **robotics simulation / research demo**，重点验证拆解流程、机器人运动规划、连续生产节拍与仿真系统集成，不等同于实际工业产线控制系统。

---

## 核心特性

| 模块 | 当前实现 |
|---|---|
| Robot | UR5e 六自由度机械臂 |
| End Effector | SMC 风格真空吸盘 |
| Physics | MuJoCo |
| Simulation Frontend | OrcaStudio |
| Communication | gRPC |
| Motion Planning | Cartesian motion + 6D DLS IK |
| Trajectory | Minimum-Jerk interpolation |
| Safety Motion | 自适应安全高度 + 分段转移路径 |
| Production Flow | 双电池包连续拆解 |
| Sorting | 上盖 / 汇流排 / 电芯分类 |
| Validation | 放置位置、姿态与稳定状态验证 |
| Workcell | 输送带、分类箱、空托盘回收区、工业安全设施 |
| CI | GitHub Actions Python/XML sanity check |

---

## 工作站组成

当前仿真场景主要由以下部分组成：

```text
┌─────────────────────────────────────────────────────┐
│                 Industrial Workcell                 │
│                                                     │
│   Safety Fence / HMI / Control Cabinet / Stack Light│
│                                                     │
│        ┌──────────────────────────────┐             │
│        │          Conveyor            │             │
│        │   Pack 1        Pack 2       │             │
│        └──────────────┬───────────────┘             │
│                       │                             │
│                    UR5e +                           │
│                 Vacuum Gripper                      │
│                       │                             │
│      Cover Bin   Busbar Bin   Cell Bins             │
│                                                     │
│                     Empty Tray Bin                  │
└─────────────────────────────────────────────────────┘
```

### 工业场景元素

除核心拆解设备外，`scene_two_pack.xml` 还包含：

- 工业工作区地面
- 黄色安全警戒线
- 透明安全防护围栏
- 电气控制柜
- HMI 面板
- 红 / 黄 / 绿三色状态灯
- 急停按钮站
- 工作站标识
- 工业照明
- 输送带检测器外观模型

这些外围模型主要用于提升工作站完整性与工业场景表现，不改变现有机器人核心运动控制逻辑。

---

## 双电池包连续流程

```mermaid
flowchart LR
    A[Battery Pack 1] --> B[进入拆解工位]
    B --> C[UR5e 自动拆解]
    C --> D[零件分类放置]
    D --> E[空托盘出站]
    E --> F[Battery Pack 2 进入工位]
    F --> G[Pack 2 稳定]
    G --> H[UR5e 自动拆解]
    H --> I[第二批零件分类]
    I --> J[第二空托盘出站]
```

第二包不会重新创建一套独立拆解算法，而是通过 `Pack2Controller` 将标准零件名称自动映射到 `pack2_*` 对象，实现同一套拆解逻辑复用。

---

## 单个电池包拆解顺序

每个电池包共执行 **7 个 Pick-and-Place 任务**：

```text
battery_cover
    ↓
battery_busbar_1
    ↓
battery_busbar_2
    ↓
battery_cell_1
    ↓
battery_cell_2
    ↓
battery_cell_3
    ↓
battery_cell_4
```

---

## 单零件运动流程

```text
Prepare
   ↓
Approach
   ↓
Pick
   ↓
Attach
   ↓
Vertical Separation
   ↓
Lift
   ↓
Retract
   ↓
Arc Transfer
   ↓
Bin Clearance
   ↓
Place
   ↓
Release
   ↓
Physics Settling
   ↓
Placement Verification
   ↓
Return
```

设计目标不是简单地把 TCP 从 A 点直线移动到 B 点，而是让机械臂在电池包、机器人基座和分类箱之间采用更稳定的分段运动。

---

## 运动控制设计

### 1. 6D Damped Least Squares IK

`Controller.solve()` 使用 6D Damped Least Squares（DLS）求解 UR5e 逆运动学。

输入：

```text
目标 TCP 位置 + 固定工具姿态
```

输出：

```text
6 个 UR5e 关节角
```

求解过程同时考虑：

- TCP position error
- TCP orientation error
- Joint limits
- Workspace reachability
- 实际工程容差

---

### 2. Minimum-Jerk 轨迹

机械臂直线移动过程使用五次多项式 Minimum-Jerk 插值，使运动在起点和终点附近保持较平滑的速度与加速度变化。

```text
Current TCP
    ↓
Minimum-Jerk interpolation
    ↓
Intermediate TCP targets
    ↓
IK
    ↓
UR5e joint motion
```

---

### 3. 自适应安全高度

固定使用较高 TCP 高度会使 UR5e 在工作空间边缘位置出现 IK 不可达。

因此当前控制器会根据：

- 零件当前高度
- 当前 TCP 高度
- 分离需求
- UR5e 工作空间

动态计算：

```text
safe_high_z
approach_z
lift_z
transfer height
return height
```

从而降低：

```text
MotionError: Unreachable pose
```

出现的概率。

---

### 4. 安全转移路径

机器人从电池包区域前往地面分类箱时，不直接穿过机械臂基座附近，而采用：

```text
Current Position
      ↓
Retract to safe radius
      ↓
Arc around robot base
      ↓
Move to bin clearance pose
      ↓
Descend
```

这使路径更适合当前工作站几何布局。

---

## 零件分类规则

### Battery Cover

```text
battery_cover
    ↓
bin_cover
```

### Busbars

```text
battery_busbar_1
battery_busbar_2
    ↓
bin_busbar
```

### Cells — Batch 1

```text
battery_cell_1
battery_cell_2
battery_cell_3
battery_cell_4
    ↓
bin_cell
```

### Cells — Batch 2

```text
pack2_battery_cell_1
pack2_battery_cell_2
pack2_battery_cell_3
pack2_battery_cell_4
    ↓
bin_cell_2
```

第二批电芯使用独立分类区域，避免与第一批四个电芯发生空间重叠。

---

## 仿真与物理模型

项目采用 **运动学机器人控制 + MuJoCo 物理仿真** 的混合方式。

### Robot

UR5e 的主要运动由 Cartesian trajectory + IK 控制。

### Vacuum Gripper

当前吸附模型为理想化 attachment：

```text
TCP
 │
 └── Payload
```

它重点用于验证机器人拆解流程，而不是建立完整的真空压力、泄漏和吸附力模型。

### Released Parts

零件释放后交由 MuJoCo 处理：

- Gravity
- Contact
- Collision
- Bounce
- Sliding
- Settling

### Empty Tray

拆解后的电池基座通过输送与回收流程进入绿色空托盘回收区域。

该部分目前仍属于仿真接触参数调优项；若修改回收箱几何尺寸、位置或碰撞参数，需要重新验证托盘接触稳定性。

---

## 项目结构

```text
ev-battery-disassembly-demo/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── scene_two_pack.xml
├── ur5e_arm.xml
├── disassembly.py
├── run_orca.py
├── run_two_pack_orca.py
│
├── assets/
│   └── UR5e / scene mesh assets
│
├── tools/
│   └── make_two_pack_scene.py
│
└── .github/
    └── workflows/
        └── sanity-check.yml
```

### `scene_two_pack.xml`

工作站场景（机械臂与底座单独放在 `ur5e_arm.xml`），包含：

- 双电池包
- 输送带
- 分类料箱
- 空托盘回收箱
- 工业工作站外围设施
- MuJoCo collision geometry
- Lights / markers / safety-area visuals

### `ur5e_arm.xml`

UR5e 机械臂资产（机械臂 + 底座），在 OrcaStudio 中与场景分开导入，包含：

- UR5e
- 机器人基座
- 机器人底座（`ur5e_pedestal`）
- SMC 风格吸盘

### `disassembly.py`

核心机器人拆解控制器，负责：

- UR5e joint / TCP control
- 6D DLS IK
- Minimum-Jerk trajectory
- Adaptive safe height
- Ideal suction attachment
- Collision checking
- Pick-and-place sequencing
- Physical settling
- Placement verification

### `run_orca.py`

OrcaStudio 与 MuJoCo 控制器之间的通信桥接。

主要负责：

- gRPC connection
- MuJoCo model / data access
- State synchronization
- OrcaStudio rendering synchronization

### `run_two_pack_orca.py`

双电池包连续拆解主入口，负责：

- Batch 1 调度
- 空托盘出站
- Pack 2 入站
- Pack 2 physics settle
- Batch 2 调度
- 第二批分类位置
- 两批连续生产流程

### `tools/make_two_pack_scene.py`

开发阶段的场景生成辅助脚本。

当前正常运行 Demo 时直接使用已经提交的：

```text
scene_two_pack.xml
```

无需重新生成场景。

---

## 软件架构

```mermaid
flowchart TD
    A[OrcaStudio] <-->|gRPC| B[run_orca.py / OrcaSession]
    B --> C[run_two_pack_orca.py]
    C --> D[disassembly.py / Controller]
    D --> E[Cartesian Target]
    E --> F[DLS IK]
    F --> G[Minimum-Jerk Motion]
    G --> H[MuJoCo Model / Data]
    H --> B
```

---

## 环境要求

当前项目开发环境：

```text
Ubuntu
Conda
Python
MuJoCo
OrcaStudio
OrcaLab / OrcaGym SDK
```

推荐使用项目现有 Conda 环境：

```bash
conda activate orcalab
```

安装公开 Python 依赖：

```bash
pip install -r requirements.txt
```

`requirements.txt` 当前包含：

```text
numpy
mujoco
grpcio
```

> `orca_gym` 由 OrcaStudio / OrcaLab SDK 环境提供，因此没有通过本仓库的 `requirements.txt` 自动安装。

---

## Quick Start

### 1. Clone

```bash
git clone git@github.com:jerryzhu-yc/ev-battery-disassembly-demo.git
cd ev-battery-disassembly-demo
```

### 2. Activate Environment

```bash
conda activate orcalab
```

### 3. Load Scene

在 OrcaStudio 中依次导入**两个** XML 文件：

```text
scene_two_pack.xml    # 纯场景：输送线、电池包、分类料箱、回收箱及外围设施
ur5e_arm.xml          # UR5e 机械臂 + 底座（独立导入的资产）
```

机械臂与场景是两个独立资产，导入后 UR5e 的初始位置不一定是正确的，
需要**手动拖动 UR5e 资产（含底座）** 到拆解工位：

- 位于**输送带正前方**（正对传送带上的拆解工位）
- 位于**三个分类料箱（上盖 / 汇流排 / 电芯）的中间**，使左右两侧料箱都落在机械臂工作空间内

摆好位置后再启动仿真，并确保 gRPC 服务运行于：

```text
localhost:50051
```

> 因为机械臂是独立资产，它的命名空间与场景不同
> （场景为 `scene_two_pack_usda_`，机械臂为 `ur5e_arm_usda_`），
> 所以运行时需要分别用 `--prefix` 和 `--arm-prefix` 指定，详见下一节。

### 4. Run Two-Pack Demo

```bash
python run_two_pack_orca.py \
  --addr localhost:50051 \
  --prefix scene_two_pack_usda_ \
  --arm-prefix ur5e_arm_usda_
```

当前测试运动速度约为：

```text
Cartesian speed : 0.360 m/s
Joint speed     : 1.600 rad/s
```

加 `--camera-view` 可额外弹出一个实时窗口，显示工位上方的
`station_camera` 相机画面（俯视拆解工位）：

```bash
python run_two_pack_orca.py \
  --addr localhost:50051 \
  --prefix scene_two_pack_usda_ \
  --arm-prefix ur5e_arm_usda_ \
  --camera-view
```

`--camera-view` 可省略值（默认 `station_camera`），也可指定其他相机名，例如
`--camera-view station_camera`。该窗口只是把相机渲染出来用于观察，拆解触发
逻辑仍使用相机位置（`<camera>` 不存在时退化为 `station_camera_mount` 的位置）
做几何判定。

关于在线模式下的画面：OrcaStudio 只把**碰撞代理体**导出给 SDK（导出模型里
`nmesh=0`、`ncam=0`，既没有网格也没有 `<camera>`），所以 **SDK 的物理模型**
里渲染不出真实场景。但引擎侧另外提供了抓帧接口，可以用 `--camera-source`
选择画面来源：

- `--camera-source auto`（默认）：在线时先试 `studio`，抓帧不可用则自动退回
  `mirror`。
- `--camera-source studio`：调用 OrcaStudio 的 gRPC `GetCameraFramePNG`，把
  **当前 Studio 视口**渲染出的画面写成 PNG 再显示，因此看到的就是 OrcaStudio
  里的真实画面（含网格、材质、光照）。注意该接口没有相机名参数，抓到的是
  Studio 当前激活的视口，而不是指定的 `station_camera`。它涉及 GPU 回读，
  OrcaGym 自身记录「与渲染流并发可能触发 `AttachmentReadback::CopyExecute`
  崩溃」，因此这里把抓帧节流到约 4 Hz，并且与 `gym.render()` 在同一事件循环里
  顺序调用（不会两个 RPC 同时在飞）。
- `--camera-source mirror`：改用**本地网格模型的镜像视图**：额外加载本地带网格的
  `scene_two_pack.xml` + `ur5e_arm.xml`，每帧把在线模型的关节/自由体状态按名字
  同步进来（两个世界只差一个固定平移，偏移量从 `station_camera_mount` 的实时
  位置自动求得），再用本地的 `station_camera` 渲染。这样在线也能看到真实的
  UR5e 机械臂跟随运动，且完全不碰引擎的渲染管线。

启动时会打印一行确认信息（`mirror` 模式）：

```text
[CAMERA VIEW] mesh mirror: nmesh=20, joints=22/22, camera=model:'station_camera', offset=[...]
```

或（`studio` 模式）：

```text
[CAMERA VIEW] studio viewport capture enabled; PNG at /tmp/orca_viewport_<pid>.png
```

之后会周期性打印 `[CAMERA VIEW] frame N, ...` / `[CAMERA VIEW] studio frame N, ...`，
用来确认窗口确实在刷新。`studio` 抓帧失败会自动停用弹窗并提示改用
`--camera-source mirror`；`mirror` 在本 XML 缺失或加载失败时会退回显示碰撞
代理体的俯视视角。

---

## OrcaStudio Prefix

OrcaStudio 导入场景后可能为对象添加 namespace / prefix。

当前场景对象常用：

```text
scene_two_pack_usda_
```

如果机械臂是作为**独立资产**导入的（例如单独的 `ur5e_arm.xml`），它的
关节 / 执行器 / `suction_tcp` 会归到另一个命名空间下，例如：

```text
ur5e_arm_usda_
```

因此运行时需要分别指定两个前缀：

```bash
--prefix scene_two_pack_usda_ \
--arm-prefix ur5e_arm_usda_
```

`--arm-prefix` 只作用于机械臂的关节、执行器和 `suction_tcp`；其余场景对象
（零件、料箱、fixture、site）仍使用 `--prefix`。若机械臂与场景在同一个资产里
（单命名空间），可省略 `--arm-prefix`，此时它会沿用 `--prefix`。

可用 `python run_orca.py --addr localhost:50051 --list-names` 查看导入后的
真实名字。如果修改 XML 文件名、导入名称或 OrcaStudio scene namespace，
请同步检查这两个参数。

---

## Sanity Check

提交前可以执行：

```bash
python -m py_compile \
  disassembly.py \
  run_orca.py \
  run_two_pack_orca.py \
  tools/make_two_pack_scene.py
```

检查 XML：

```bash
python - <<'PY'
import xml.etree.ElementTree as ET

ET.parse("scene_two_pack.xml")
print("scene_two_pack.xml: OK")
PY
```

仓库同时提供：

```text
.github/workflows/sanity-check.yml
```

GitHub Actions 会在 Push / Pull Request 后执行基础 Python 与 XML 检查。

---

## 当前实现边界

本项目目前主要关注 **机器人拆解工作流与仿真验证**。

以下内容属于理想化或仍可继续工程化的部分：

- 真空吸盘采用理想化 attachment，而非完整真空压力模型
- 机器人主体采用运动学 Cartesian 控制，而非完整工业伺服控制器
- 输送带主要用于生产流程仿真
- 空托盘回收区域的 MuJoCo 接触参数仍需要随场景变化进行验证
- 工业电控柜、HMI、三色灯、急停站等当前主要作为工作站场景元素展示
- 当前项目不是实际工业安全认证系统

---

## 适用方向

该项目适合作为以下方向的仿真基础：

- EV Battery Recycling
- Robotic Disassembly
- Robot Manipulation
- Automated Sorting
- Cartesian Motion Planning
- Inverse Kinematics
- MuJoCo Simulation
- OrcaStudio Integration
- Digital Manufacturing / Digital Twin Demo

---

## Third-Party Assets

仓库包含 UR5e 相关 mesh / model assets。

若将项目用于再次分发、公开商业用途或其他正式产品，请确认相关第三方模型资源的原始许可证与 attribution 要求。

---

## Disclaimer

This repository is a robotics simulation and research demo.

It is intended for:

- simulation
- algorithm validation
- robotics research
- education
- portfolio demonstration

It is **not** a production-ready industrial robot controller or certified safety system.

---

<div align="center">

**EV Battery Disassembly Workcell — UR5e · MuJoCo · OrcaStudio**

</div>
