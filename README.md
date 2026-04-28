# 双轮腿机器人控制项目（Wheel-Legged Robot）

基于 VMC + LQR + PID 的双轮腿机器人平衡控制与仿真项目，同时包含 Isaac Gym 强化学习训练框架。

## 项目简介

本项目实现了一个五连杆双轮腿机器人的完整控制系统，包括：

- **VMC（虚拟模型控制）**：五连杆正运动学 + 雅可比力矩映射
- **LQR（线性二次调节器）**：增益调度多项式，基于 MATLAB 离线计算
- **PID 控制**：腿长 PD 控制 + 重力补偿
- **强化学习**：Isaac Gym 环境下的 PPO + 残差 RL 版本

## 硬件参数

| 参数 | 值 |
|------|-----|
| 五连杆上臂长 l1=l4 | 0.07 m |
| 五连杆下臂长 l2=l3 | 0.147 m |
| 底座宽度 l5 | 0.123 m |
| 轮子半径 | 0.05 m |
| 总质量 | 3.524 kg |
| 控制频率 | 500 Hz（Gazebo）/ 1000 Hz（MuJoCo） |

## 目录结构

```
wlr_ws/
├── src/
│   ├── wlr_controller/        # 控制算法核心
│   │   ├── balance_node.py    # 主控制循环（VMC + LQR + PID）
│   │   ├── vmc.py             # 五连杆正运动学
│   │   ├── lqr_gains.py       # LQR 增益调度多项式
│   │   ├── leg_jacobian.py    # 雅可比转置力矩映射
│   │   └── config/params.yaml # 可调参数
│   ├── wlr_description/       # 机器人模型
│   │   └── urdf/              # URDF/xacro（五连杆）
│   ├── wlr_gazebo/            # Gazebo 仿真
│   │   ├── FiveBarClosure.cc  # 五连杆闭链约束插件
│   │   └── launch/sim.launch.py
│   └── wlr_mujoco/            # MuJoCo 仿真
│       ├── model/wlr_robot.xml
│       └── wlr_mujoco/mujoco_node.py
├── Wheel-Legged-Gym/          # Isaac Gym 强化学习版本（外部仓库）
└── DESIGN.md                  # 详细设计文档
```

## 控制架构

### 数据流

```
传感器读数（关节角 + IMU）
    ↓
VMC 正运动学（五连杆 → 虚拟腿 L0, phi0）
    ↓
LQR 增益调度（K 矩阵 = f(L0)）
    ↓
LQR 输出: [T_wheel, Tp_leg]
    ↓
腿长 PD 控制 → 虚拟力 F
    ↓
雅可比转置（虚拟力 → 关节力矩）
    ↓
电机执行
```

### VMC 正运动学

五连杆机构通过两个电机驱动，VMC 将关节角度映射为虚拟腿参数：

```
输入: phi1（前髋关节角）, phi4（后髋关节角）
输出: L0（虚拟腿长度）, phi0（虚拟腿角度）

计算:
  B 点位置: xb = l1*cos(phi1), yb = l1*sin(phi1)
  D 点位置: xd = l4*cos(phi4) + l5, yd = l4*sin(phi4)
  半角正切法求解 phi2
  C 点位置: xc = xb + l2*cos(phi2), yc = yb + l2*sin(phi2)
  L0 = sqrt((xc - l5/2)² + yc²)
  phi0 = atan2(yc, xc - l5/2)
```

### LQR 增益调度

LQR 增益 K 是 2×6 矩阵，每个元素是 L0 的三次多项式：

```
K[i](L0) = c0 + c1*L0 + c2*L0² + c3*L0³

状态向量: [theta, dTheta, x, dx, phi, dPhi]
输出: [T_wheel, Tp_leg]
```

其中：
- theta: 摆角（腿倾斜 + 机体俯仰）
- x: 轮子位移
- phi: 机体俯仰角

### 雅可比转置映射

将虚拟力 (F, Tp) 转换为关节力矩 (T1, T2)：

```
[T1]   [J11 J12]^-T   [F]
[T2] = [J21 J22]    × [Tp]
```

## 运行方式

### Gazebo 仿真

```bash
# 编译
cd wlr_ws
colcon build

# 运行（需要 Gazebo Harmonic）
ros2 launch wlr_gazebo sim.launch.py
```

### MuJoCo 仿真

```bash
# 编译
cd wlr_ws
colcon build --packages-select wlr_mujoco

# 无可视化运行
ros2 launch wlr_mujoco sim.launch.py

# 带可视化运行
ros2 launch wlr_mujoco sim_visual.launch.py
```

### Isaac Gym 强化学习训练

```bash
# 克隆 Wheel-Legged-Gym 仓库
cd wlr_ws
git clone https://github.com/clearlab-sustech/Wheel-Legged-Gym.git
cd Wheel-Legged-Gym
pip install -e .

# 训练原版 PPO
python wheel_legged_gym/scripts/train.py --task=wheel_legged_vmc_flat --headless

# 训练残差 RL 版本（LQR + PPO）
python wheel_legged_gym/scripts/train.py --task=wheel_legged_residual_flat --headless

# 查看训练曲线
tensorboard --logdir=./ --port=8080
```

**注意：** Isaac Gym 需要 NVIDIA GPU（RTX 2070+）和从 NVIDIA 官网下载的 Preview 4。

## 可调参数

编辑 `src/wlr_controller/config/params.yaml`：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| wheel_radius | 0.05 m | 轮子半径 |
| default_leg_length | 0.15 m | 默认腿长 |
| leg_length_kp | 100.0 | 腿长 P 增益 |
| leg_length_kd | 300.0 | 腿长 D 增益 |
| lqr_tp_ratio | 0.4 | LQR 腿力矩缩放 |
| lqr_t_ratio | 0.8 | LQR 轮子力矩缩放 |
| torque_limit | 5.0 Nm | 力矩上限 |
| control_rate | 500 Hz | 控制频率 |

## 仿真环境对比

| 特性 | Gazebo | MuJoCo | Isaac Gym |
|------|--------|--------|-----------|
| 物理引擎 | DART | MuJoCo | PhysX |
| 并行环境数 | 1 | 1 | 4096 |
| 五连杆模型 | 完整（闭链约束插件） | 简化（串联链） | 完整 |
| 控制算法 | LQR + VMC + PID | LQR + PD | PPO / 残差 RL |
| GPU 加速 | 否 | 否 | 是 |

## 相关论文

- [Learning to Walk in Minutes Using Massively Parallel Deep RL](https://arxiv.org/abs/2109.11978)
- [Virtual Model Control: An Intuitive Approach for Bipedal Dynamic Locomotion](https://ieeexplore.ieee.org/document/99497)

## 参考项目

- [Wheel-Legged-Gym](https://github.com/clearlab-sustech/Wheel-Legged-Gym) — Isaac Gym 版本
- [legged_gym](https://github.com/leggedrobotics/legged_gym) — ETH Zurich 基础框架
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl) — RL 算法库
