# 双轮腿机器人控制项目（Wheel-Legged Robot）

基于 VMC + LQR + PID 的双轮腿机器人平衡控制与仿真项目，同时包含 Isaac Gym 强化学习训练框架（含残差 RL 版本）。

## 项目简介

本项目实现了一个五连杆双轮腿机器人的完整控制系统，包括：

- **VMC（虚拟模型控制）**：五连杆正运动学 + 雅可比力矩映射
- **LQR（线性二次调节器）**：增益调度多项式，基于 MATLAB 离线计算
- **PID 控制**：腿长 PD 控制 + 重力补偿
- **强化学习**：Isaac Gym 环境下的 PPO + 残差 RL 版本
- **Sim2Real**：Teacher-Student 蒸馏 + 部署方案

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
├── DESIGN.md                  # 详细设计文档
└── README.md                  # 本文件
```

---

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

五连杆拓扑:
          C (足端/轮子)
         / \
   l2   /   \  l3
       /     \
      B       D
      |       |
  l1  |  l5   |  l4
      |-------|
      A       E
   (前髋)  (后髋)

计算过程:
  B 点位置: xb = l1*cos(phi1), yb = l1*sin(phi1)
  D 点位置: xd = l4*cos(phi4) + l5, yd = l4*sin(phi4)
  半角正切法求解 phi2（闭链约束方程）
  C 点位置: xc = xb + l2*cos(phi2), yc = yb + l2*sin(phi2)
  虚拟腿: L0 = sqrt((xc - l5/2)² + yc²)
          phi0 = atan2(yc, xc - l5/2)
```

### LQR 增益调度

LQR 增益 K 是 2×6 矩阵，每个元素是 L0 的三次多项式：

```
K[i](L0) = c0 + c1*L0 + c2*L0² + c3*L0³

状态向量: [theta, dTheta, x, dx, phi, dPhi]
输出: [T_wheel, Tp_leg]

其中:
  theta  = 腿倾斜角 + 机体俯仰角（摆角）
  dTheta = 摆角角速度
  x      = 轮子位移
  dx     = 线速度
  phi    = 机体俯仰角
  dPhi   = 俯仰角速度
```

### 雅可比转置映射

将虚拟力 (F, Tp) 转换为关节力矩 (T1, T2)：

```
雅可比矩阵:
  J = [∂end_x/∂theta1, ∂end_x/∂theta2]
      [∂end_y/∂theta1, ∂end_y/∂theta2]

力矩映射（雅可比转置）:
  [T1]   [J11 J21]^-T   [F_x]
  [T2] = [J12 J22]    × [F_y]
```

---

## LQR vs RL 对比

| 组件 | LQR 项目（本项目） | RL 项目（Wheel-Legged-Gym） |
|------|-------------------|---------------------------|
| 五连杆正运动学 | vmc.py | forward_kinematics() |
| 雅可比力矩映射 | leg_jacobian.py | VMC() |
| PD 控制器 | balance_node.py | _compute_torques() |
| LQR 状态反馈 | lqr_gains.py | 没有（RL 替代） |
| 奖励函数 | 没有 | compute_reward() |
| 域随机化 | 没有 | domain_rand |

**核心区别：LQR 是你写好的规则，RL 是机器人自己试出来的规则。**

---

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

#### 环境准备

```bash
# 前置条件（需提前安装，不包含在本仓库中）
# 1. NVIDIA GPU (RTX 2070+)
# 2. Isaac Gym Preview 4：从 https://developer.nvidia.com/isaac-gym 下载
#    进入 isaacgym/python 目录，执行 pip install -e .
# 3. CUDA 版 PyTorch：https://pytorch.org

# 验证
python -c "import isaacgym; print('Isaac Gym OK')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

#### 安装并训练

```bash
# 本仓库的 Wheel-Legged-Gym/ 已经是完整的可运行代码，无需额外克隆上游仓库

cd wlr_ws/Wheel-Legged-Gym
pip install -e .

# 训练残差 RL 版本（LQR + PPO）—— 本项目核心
python wheel_legged_gym/scripts/train.py --task=wheel_legged_residual_flat --headless

# 训练原版 PPO（纯 RL，无 LQR）—— 用于对比
python wheel_legged_gym/scripts/train.py --task=wheel_legged_vmc_flat --headless

# 查看训练曲线
tensorboard --logdir=./ --port=8080
```

#### 训练时间参考

| 迭代数 | 预期表现 |
|--------|---------|
| 0 – 500 | 机器人可能还站不稳 |
| 500 – 1000 | 能站稳但走得不好 |
| 1000 – 1500 | 能基本走起来 |
| 1500 – 2000 | 比较好的效果 |

#### 回放训练结果

```bash
# 替换 LOG_DIR 为实际日志目录名
python wheel_legged_gym/scripts/play.py --task=wheel_legged_residual_flat --load_run=LOG_DIR
```

---

## 残差 RL 架构（本项目创新）

### 核心思想

不是替换 LQR，而是在 LQR 基础上叠加 RL 改进：

```
最终力矩 = LQR 基线力矩 + residual_scale × RL 残差
```

### 为什么残差 RL 更好

- **训练更快**：RL 只需学习 LQR 的不足之处，搜索空间从"控制整个机器人"缩小到"改进 LQR"
- **部署更安全**：即使 RL 输出异常，LQR 兜底，机器人不会直接摔倒
- **可解释性**：可以清楚看到 LQR 负责什么、RL 负责什么

### 课程学习策略

```
步数 0-50k:      residual_scale = 0      纯 LQR，让机器人先学会站稳
步数 50k-250k:   residual_scale 0→1      线性增长，RL 逐步介入
步数 250k+:      residual_scale = 1      RL 全权控制残差
```

### 观测空间（31维）

```
[0:3]   base_ang_vel         IMU角速度
[3:6]   projected_gravity    重力投影
[6:9]   commands             目标速度
[9:11]  theta0               虚拟腿角度（左右）
[11:13] theta0_dot           虚拟腿角速度
[13:15] L0                   虚拟腿长度
[15:17] L0_dot               虚拟腿速度
[17:19] wheel_pos            轮子位置
[19:21] wheel_vel            轮子速度
[21:23] prev_actions         上一步残差
[23:29] lqr_state            LQR 6维状态向量
[29:31] lqr_T, lqr_Tp        LQR 输出
```

### 创建的文件

```
Wheel-Legged-Gym/wheel_legged_gym/envs/
├── __init__.py                              # 修改：注册新任务
└── wheel_legged_vmc_flat/
    ├── lqr_gpu.py                           # 新增：GPU向量化LQR+VMC+Jacobian
    ├── wheel_legged_residual_flat.py        # 新增：残差RL任务类
    └── wheel_legged_residual_flat_config.py # 新增：残差RL配置
```

---

## PPO 算法详解

### 直觉理解

想象你在教一个婴儿走路：
1. 你不会告诉他具体的肌肉收缩方案
2. 你会让他自己试：走了一步给笑脸（正奖励），摔倒了给哭脸（负奖励）
3. 他慢慢学会什么动作能得到笑脸

PPO 的核心技巧：**每次只做小幅度的策略更新**（clip 机制），防止训练崩溃。

### 训练循环

```
每次迭代（约2秒）：

阶段1: 收集数据（Rollout）
  4096 个机器人同时跑 48 步
  共收集 4096 × 48 = 约20万条数据

阶段2: 计算"优势"（GAE）
  好的动作 → 增加出现概率
  差的动作 → 降低出现概率

阶段3: 更新神经网络（5轮 × 4个batch = 20次梯度更新）
  PPO Clipped Loss 限制更新幅度

阶段4: 保存 checkpoint，写日志
```

### 数学公式

**PPO Clipped Objective：**
```
L^CLIP(θ) = E_t [min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]
```
- r_t(θ) = 新旧策略的概率比
- ε = 0.2（clip 范围）
- A_t = 优势函数（GAE 估计）

**GAE（Generalized Advantage Estimation）：**
```
δ_t = r_t + γV(s_{t+1}) - V(s_t)     (TD 误差)
A_t = Σ (γλ)^l * δ_{t+l}              (GAE)
```
- γ = 0.99（折扣因子）
- λ = 0.95（GAE 参数）

### Reward 函数

| 评分项 | 权重 | 含义 |
|--------|------|------|
| tracking_lin_vel | 1.0 | 跟着目标速度走 |
| tracking_ang_vel | 1.0 | 跟着目标转向 |
| orientation | -10.0 | 保持直立（惩罚项） |
| base_height | 1.0 | 保持合适高度 |
| torques | -0.0001 | 力矩越小越好（节能） |
| collision | -1.0 | 避免碰撞 |
| action_rate | -0.01 | 动作平滑 |
| residual_magnitude_l2 | -0.1 | 惩罚大残差（残差 RL 专有） |

### 神经网络结构

```
观测历史: 5步 × 31维 = 155维
    ↓
Encoder: Linear(155,128) → ELU → Linear(128,64) → ELU → Linear(64,3)
    输出 3 维潜在表示 z
    ↓
Actor: Linear(3,128) → ELU → Linear(128,64) → ELU → Linear(64,32) → ELU → Linear(32,2)
    输出 2 维动作 [d_T, d_Tp]
    ↓
Critic: Linear(3,256) → ELU → Linear(256,128) → ELU → Linear(128,64) → ELU → Linear(64,1)
    输出状态价值 V(s)
```

### PPO 超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| clip_param | 0.2 | PPO clip 范围 |
| entropy_coef | 0.01 | 熵奖励系数 |
| learning_rate | 1e-3 | 学习率 |
| gamma | 0.99 | 折扣因子 |
| lam | 0.95 | GAE lambda |
| num_learning_epochs | 5 | 每次更新轮数 |
| num_mini_batches | 4 | mini-batch 数 |
| num_steps_per_env | 48 | 每环境收集步数 |
| max_iterations | 2000 | 最大迭代数 |

---

## Domain Randomization（域随机化）

### 为什么需要

仿真里能站稳，实物一上就倒 → 仿真的"世界"太完美了。

训练时故意把仿真搞乱，让策略学会应对各种不确定性。

### 随机化项

| 随机化项 | 范围 | 为什么需要 |
|---------|------|-----------|
| 摩擦系数 | [0.1, 2.0] | 真实地面前后不一致 |
| 额外质量 | [-2, 3] kg | 负载变化 |
| 惯量 | [0.8, 1.2] | 建模误差 |
| 质心偏移 | ±5cm | 制造装配误差 |
| Kp 增益 | [0.9, 1.1] | 控制器参数漂移 |
| 电机力矩 | [0.9, 1.1] | 电机制造差异 |
| 动作延迟 | [0, 10] ms | CAN 通信延迟 |
| 外部推力 | 每 7 秒随机推 | 抗干扰能力 |

---

## Sim2Real：从仿真到实物

### 部署流程

```
训练好的 Student 网络
    ↓
导出 TorchScript
    ↓
部署到 Jetson / 工控机
    ↓
ROS 2 节点（100Hz 控制循环）
    ↓
CAN 总线 → 电机驱动器
```

### Teacher-Student 蒸馏

训练时 Teacher 能看到所有信息（真实质心、摩擦等），部署时 Student 只能用本体感知：

```
训练: Teacher[本体感知 + 特权信息] → 动作
蒸馏: Student[仅本体感知] → 模仿 Teacher
loss = ||Student(obs) - Teacher(obs_with_privilege)||²
```

### 关键对齐项

| 项目 | 仿真值 | 实物值 |
|------|--------|--------|
| 关节零位 | default_joint_angles | 实测编码器值 |
| 电机力矩上限 | torque_limits | 电机手册值 |
| 通信延迟 | 0-10ms | CAN 总线实测 |
| IMU 噪声 | noise_scales | 实测数据 |
| 机体质量 | base_mass | 称重实测 |

---

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

---

## 调试指南

### 训练监控

```bash
tensorboard --logdir=./ --port=8080
```

| 曲线 | 正常表现 | 有问题的表现 |
|------|---------|------------|
| Episode Reward | 从负数逐渐变正 | 一直是负数不变 |
| Episode Length | 从几十逐渐到几百 | 一直是几十 |
| Policy Loss | 逐渐下降 | 突然飙升 |
| KL Divergence | 在 0.005 附近 | 远大于 0.01 |

### 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| Reward 不增长 | 学习率太大 | 降低 learning_rate |
| 训练中途崩溃 | 策略更新太快 | 减小 clip_param |
| 能站但不走 | tracking_lin_vel 权重太小 | 增大 scale |
| 训练很慢 | 没有用 GPU | 安装 CUDA 版 PyTorch |

---

## 仿真环境对比

| 特性 | Gazebo | MuJoCo | Isaac Gym |
|------|--------|--------|-----------|
| 物理引擎 | DART | MuJoCo | PhysX |
| 并行环境数 | 1 | 1 | 4096 |
| 五连杆模型 | 完整（闭链约束插件） | 简化（串联链） | 完整 |
| 控制算法 | LQR + VMC + PID | LQR + PD | PPO / 残差 RL |
| GPU 加速 | 否 | 否 | 是 |

---

## 术语对照表

| RL 术语 | LQR 对应 | 通俗解释 |
|---------|---------|---------|
| Policy（策略） | LQR 的 K 矩阵 | 决定输出什么动作的规则 |
| Reward（奖励） | 没有对应概念 | 机器人行为的评分标准 |
| Episode | 一次运行 | 从站起到倒下的过程 |
| Observation（观测） | 传感器读数 | 机器人能看到的信息 |
| Action（动作） | LQR 输出的力矩 | 机器人执行的控制量 |
| Domain Rand | 没有对应概念 | 训练时故意把仿真搞乱 |
| GAE | 没有对应概念 | 计算动作好坏的方法 |
| Clip | 没有对应概念 | 防止策略更新太猛的安全机制 |

---

## 推荐阅读

1. Schulman et al., "Proximal Policy Optimization Algorithms", 2017 — PPO 原始论文
2. Rudin et al., "Learning to Walk in Minutes Using Massively Parallel Deep RL", 2022 — 本项目基础
3. [Wheel-Legged-Gym](https://github.com/clearlab-sustech/Wheel-Legged-Gym) — Isaac Gym 版本
4. [Isaac Gym](https://developer.nvidia.com/isaac-gym) — NVIDIA 仿真器
5. [legged_gym](https://github.com/leggedrobotics/legged_gym) — ETH Zurich 基础框架
6. [rsl_rl](https://github.com/leggedrobotics/rsl_rl) — RL 算法库

---

## 相关论文

- [Learning to Walk in Minutes Using Massively Parallel Deep RL](https://arxiv.org/abs/2109.11978)
- [Virtual Model Control: An Intuitive Approach for Bipedal Dynamic Locomotion](https://ieeexplore.ieee.org/document/99497)
- [Concurrent Training of a Control Policy and a State Estimator](https://arxiv.org/abs/2202.05481)
