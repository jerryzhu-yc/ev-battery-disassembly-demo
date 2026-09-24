# EV 电池拆解 Demo

基于 ** OrcaStudio **的 EV 电池包自动拆解仿真 Demo。

项目实现了双电池包连续拆解流程：UR5e 机械臂使用真空吸盘依次完成电池上盖、汇流排和电芯的抓取，并将不同零部件分类放置到对应物料箱中。

## 功能

- UR5e 六自由度机械臂
- SMC 真空吸盘末端
- 双电池包连续拆解
- 输送带工位
- 上盖 / 汇流排 / 电芯分类放置
- Cartesian IK 运动规划
- Minimum-jerk 轨迹插值
- 自适应安全高度规划
- MuJoCo 释放后物理仿真
- 零件放置状态验证

单个电池包当前拆解顺序：

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

## 项目结构

```text
ev电池拆解demo/
├── README.md
├── scene_two_pack.xml
├── disassembly.py
├── run_orca.py
├── run_two_pack_orca.py
├── make_two_pack_scene.py        # 若保留场景生成脚本
└── assets/
    └── ...                       # UR5e / 场景模型资源
```

### scene_two_pack.xml

最终使用的 MuJoCo / OrcaStudio 场景，包含 UR5e、机械臂基座、真空吸盘、双电池包、传送带、分类箱以及相关碰撞几何。

### disassembly.py

核心拆解控制程序，主要负责：

- UR5e 关节控制
- Cartesian IK
- TCP 位姿控制
- Minimum-jerk 轨迹生成
- 真空吸附模拟
- 零件抓取、分离、搬运和释放
- 分类箱放置
- 碰撞检查
- 放置状态验证
- 自适应运动高度规划

### run_orca.py

OrcaStudio / gRPC 通信支持代码。

### run_two_pack_orca.py

双电池包连续拆解主入口。

## 环境

当前开发测试环境：

```text
Ubuntu 24.04
Python
Conda
MuJoCo
OrcaStudio
```

激活环境：

```bash
conda activate orcalab
```

主要 Python 依赖：

```text
numpy
mujoco
grpcio
```

## 运行方法

### 1. 启动 OrcaStudio

导入：

```text
scene_two_pack.xml
```

启动仿真，并确保 gRPC 服务运行在：

```text
localhost:50051
```

### 2. 进入项目目录

```bash
cd ~/桌面/ev电池拆解demo
conda activate orcalab
```

### 3. 运行双电池包拆解

```bash
python run_two_pack_orca.py \
  --addr localhost:50051 \
  --prefix scene_two_pack_usda_
```

当前测试使用的机器人速度约为：

```text
Cartesian speed : 0.360 m/s
Joint speed     : 1.600 rad/s
```

## 拆解流程

```text
安全准备位置
    ↓
接近零件
    ↓
下降到抓取位置
    ↓
吸附
    ↓
垂直分离
    ↓
抬升
    ↓
向机器人安全半径内收
    ↓
圆弧转移
    ↓
移动到分类箱
    ↓
下降放置
    ↓
释放
    ↓
物理稳定
    ↓
放置验证
    ↓
机械臂返回
```

## 自适应高度规划

UR5e 在工作空间边缘位置时，固定使用较高 TCP 高度可能导致 IK 不可达。

当前程序会根据零件高度和当前 TCP 状态动态计算安全高度，并用于：

```text
prepare
approach
lift
transfer
return
```

从而减少：

```text
MotionError: Unreachable pose
```

## 分类规则

```text
battery_cover
    → bin_cover

battery_busbar_1
battery_busbar_2
    → bin_busbar

battery_cell_1
battery_cell_2
battery_cell_3
battery_cell_4
    → bin_cell
```

## 仿真说明

当前 Demo 重点验证：

- 自动拆解任务流程
- UR5e 轨迹规划
- 多零件连续抓取
- 零部件分类
- 状态与放置结果验证

机械臂运动主要采用运动学控制，真空吸附采用理想化抓取模型；零件释放后使用 MuJoCo 物理系统进行碰撞和稳定过程仿真。

该项目适用于机器人拆解流程、自动化拆解算法以及 MuJoCo / OrcaStudio 机器人应用的仿真与研究验证。

## 注意事项

### OrcaStudio Prefix

当前场景使用：

```text
scene_two_pack_usda_
```

因此默认运行命令为：

```bash
python run_two_pack_orca.py \
  --addr localhost:50051 \
  --prefix scene_two_pack_usda_
```

如果修改 XML 文件名或 OrcaStudio 中的场景名称，prefix 可能变化。

### 场景位置修改

修改以下内容后可能需要重新调整 IK 和安全运动参数：

- UR5e 基座位置
- 电池包位置
- 传送带高度
- 分类箱位置
- 零件高度



## Disclaimer

This project is a robotics simulation and research demo intended for simulation, algorithm validation, and educational use.
