# 双轮腿机器人 ROS2 仿真工程设计文档

## 1. 概述

本工程基于赵浩阳毕设论文《基于LQR的双轮腿机器人平衡控制研究》，在 ROS2 Jazzy + Gazebo Harmonic 平台上实现双轮腿机器人的仿真，包含：

- **真实五连杆闭链机构**建模（非简化单摆杆）
- VMC + LQR 平衡控制
- 直立平衡、高度变化、移动三个仿真场景

## 2. 五连杆建模方案

### 2.1 问题

URDF 只支持树形结构，无法直接描述闭环运动链。五连杆机构 ABCDE 中，BC 和 DC 在 C 点交汇形成闭环。

### 2.2 解决方案：URDF 开链树 + Gazebo SDF 闭合约束

每条腿在 URDF 中拆分为两条独立运动链：

```
base_link
├─ hip_front_joint (ACTUATED) → upper_front → knee_front_joint (PASSIVE) → lower_front → foot_front
└─ hip_rear_joint  (ACTUATED) → upper_rear  → knee_rear_joint  (PASSIVE) → lower_rear → foot_rear → wheel
```

在 `gazebo.xacro` 中添加 SDF 闭合约束关节，将 `foot_front` 约束到 `foot_rear`，使物理引擎(DART)自动求解闭环。

### 2.3 五连杆结构

```
                C (wheel)
               / \
        l2    /   \   l3
             /     \
           B         D
           |         |
      l1   |   l5    |  l4
           |---------|
           A         E
        (front)    (rear)
```

| 参数 | 值 | 说明 |
|------|------|------|
| l1=l4 | 0.07 m | 上连杆 |
| l2=l3 | 0.147 m | 下连杆 |
| l5 | 0.123 m | 髋间距 |
| R | 0.05 m | 车轮半径 |

## 3. 物理参数

| 参数 | 值 | 来源 |
|------|------|------|
| 机体质量 | 2.19 kg | 论文表2.1 |
| 上连杆质量 | 0.039 kg | 论文(大腿) |
| 下连杆质量 | 0.072 kg | 论文(小腿) |
| 车轮质量 | 0.340 kg | 论文(含电机) |
| 机体尺寸 | 210×155×60 mm | 论文 |
| 轮距 | 0.345 m | 论文 |
| 髋关节限位 | [-1.8, 0.2] rad | Simulink |

## 4. 控制架构

### 4.1 ros2_control 配置

- `joint_state_broadcaster`: 广播所有关节状态
- `forward_command_controller (effort)`: 接收 6 个力矩命令
  - left_hip_front, left_hip_rear, left_wheel
  - right_hip_front, right_hip_rear, right_wheel

### 4.2 控制节点 balance_node (500Hz)

每个周期执行：

1. **读取传感器**: `/joint_states` → 6 关节位置/速度; `/imu/data` → 机体姿态
2. **VMC 正运动学**: `vmc.leg_pos(phi1, phi4)` → (L0, phi0)
3. **状态变量提取**:
   - θ = phi0 - π/2 + body_pitch (摆杆角)
   - x = wheel_pos × R (位移)
   - φ = -body_pitch (机体倾角)
4. **LQR 控制**: `lqr_k(L0)` → K[2×6], 计算 [T, Tp]
5. **腿长 PID**: F = Kp(L0_target - L0) - Kd·dL0 + 重力补偿
6. **Jacobian 逆映射**: `leg_conv(phi1, phi4, F, Tp)` → [T1, T2]
7. **发布力矩**: effort_controller/commands

### 4.3 LQR 增益调度

LQR 增益 K 是腿长 L0 的三次多项式函数，由 MATLAB 离线求解后用多项式拟合。
Q = diag[100, 10, 10, 10, 1000, 100], R = diag[1, 1]。

## 5. 工程结构

```
~/wlr_ws/src/
├── wlr_description/          # URDF 模型包
│   ├── urdf/
│   │   ├── wlr_robot.urdf.xacro   # 主模型
│   │   ├── leg.urdf.xacro          # 单腿宏(五连杆开链树)
│   │   ├── materials.xacro         # 颜色定义
│   │   └── gazebo.xacro            # Gazebo插件+闭链约束+ros2_control
│   ├── config/controllers.yaml     # ros2_control 配置
│   ├── launch/display.launch.py    # RViz 查看
│   └── rviz/config.rviz
│
├── wlr_gazebo/               # Gazebo 仿真包
│   ├── worlds/empty.sdf            # 空世界(DART物理引擎)
│   └── launch/sim.launch.py        # 启动仿真
│
└── wlr_controller/           # Python 控制器包
    ├── wlr_controller/
    │   ├── balance_node.py         # 主控制节点
    │   ├── vmc.py                  # VMC 正运动学
    │   ├── lqr_gains.py            # LQR 增益多项式
    │   └── leg_jacobian.py         # Jacobian 力矩转换
    ├── config/params.yaml          # 参数配置
    └── launch/control.launch.py    # 启动控制器
```

## 6. 使用说明

### 6.1 编译

```bash
cd ~/wlr_ws
colcon build --symlink-install
source install/setup.bash
```

### 6.2 场景 1: RViz 模型查看

```bash
ros2 launch wlr_description display.launch.py
```

用 joint_state_publisher_gui 拖动滑块验证五连杆关节运动。

### 6.3 场景 2: Gazebo 仿真 + 平衡控制

```bash
# 终端1: 启动 Gazebo 仿真
ros2 launch wlr_gazebo sim.launch.py

# 终端2: 启动控制器
ros2 launch wlr_controller control.launch.py
```

### 6.4 场景 3: 高度变化

```bash
ros2 topic pub --once /cmd_height std_msgs/msg/Float64 "data: 0.13"
ros2 topic pub --once /cmd_height std_msgs/msg/Float64 "data: 0.15"
```

### 6.5 场景 4: 移动

```bash
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.3}}"
```

### 6.6 监控

```bash
# 查看关节状态
ros2 topic echo /joint_states

# 绘制状态曲线
ros2 run rqt_plot rqt_plot /joint_states
```

## 7. 控制参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| lqr_tp_ratio | 0.4 | 腿部虚拟力矩缩放 |
| lqr_t_ratio | 0.8 | 车轮力矩缩放 |
| leg_length_kp | 100.0 | 腿长PD比例增益 |
| leg_length_kd | 300.0 | 腿长PD微分增益 |
| default_leg_length | 0.11 | 默认腿长 (m) |
| torque_limit | 3.0 | 关节力矩饱和限 (Nm) |

## 8. 从固件代码到仿真的移植说明

### 8.1 VMC (vmc.c → vmc.py)
- `leg_pos()`: 半角正切法求解五连杆正运动学，直接移植
- `leg_spd()`: 基于 Jacobian 的速度解算，直接移植

### 8.2 LQR (lqr_k.c → lqr_gains.py)
- MATLAB Coder 生成的 12 个三次多项式，48 个系数完整移植
- K 矩阵存储：C 代码列主序 K[state*2+output] → Python 转置为 K[output][state]

### 8.3 Jacobian (leg_conv.c → leg_jacobian.py)
- MATLAB Coder 生成的闭式 Jacobian 转置，直接移植
- 包含五连杆完整几何关系

### 8.4 主控制器 (balance_task.c + leg_control_task.c → balance_node.py)
- 状态提取、LQR 计算、腿长 PID、力矩分配逻辑完整移植
- 车轮半径从 0.038m 调整为 0.05m（仿真适配）
- 重力补偿: (3.55-0.34)*9.81/2 = 15.73N per leg
