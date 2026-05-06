# Wheel-Legged Robot Residual RL — 项目交接文档

> **Owner**: 赵浩阳 (Zhao Haoyang), 合肥工业大学 2024 届毕业设计
> **GitHub**: https://github.com/zhyhfut/wlr_ws
> **Date**: 2026-05-06
> **Status**: 训练 6 轮均未收敛到目标速度 (peak ~0.6 m/s, target 2.5 m/s)

---

## 0. 服务器登录信息

```
SSH: ssh -p 13834 root@connect.westc.seetacloud.com
密码: 0iafN9NJ8I0k
```

训练环境路径:
```
/root/Wheel_Legged_Gym_2/wlr_ws/Wheel-Legged-Gym/
```

Python 路径:
```
/root/miniconda3/bin/python
```

**注意**: 服务器上只有 `python2` 在 PATH 里，必须在命令前 `export PATH=/root/miniconda3/bin:$PATH`。Ninja 也需要这个 PATH。

---

## 1. 项目目标

用 PPO 强化学习训练双轮腿机器人，使其在平地/崎岖地形上以 0–2.5 m/s 的指令速度稳定行走，保持正常姿态不摔倒。

---

## 2. 机器人物理参数

| Parameter | Value |
|-----------|-------|
| Total mass | 12.28 kg |
| Base mass | 8.8 kg |
| Link lengths | l1=0.15m, l2=0.25m |
| Wheel radius | 0.0675 m |
| Track half | 0.25 m |
| COM height | ~0.25 m |
| DOFs | 6 (lf0, lf1, l_wheel, rf0, rf1, r_wheel) |
| Wheel torque limit | 15 Nm |
| Leg joint torque limit | 30 Nm |

---

## 3. 控制架构

```
RL Policy (PPO, 4-D actions)          PD Baseline (VMC)
     │                                      │
     ▼                                      ▼
[d_θ0_L, d_l0_L, d_θ0_R, d_l0_R]    theta0_ref = 0, l0_ref = 0.175
     │                                      │
     └──────────┬───────────────────────────┘
                ▼
    final target = baseline + residual_scale × action × action_scale
                │
                ▼
    VMC PD: T_leg = kp_θ(θ_ref-θ) - kd_θ*θ_dot
            F_leg = kp_l(l_ref-L) - kd_l*L_dot + feedforward_force(15N)
                │
                ▼
    VMC Jacobian transpose → [T_f0, T_f1] (per leg)
    Wheel torque = damping × (ω_target - ω_actual)  ← NON-RL, fixed PD
                │
                ▼
    final: [T_f0_L, T_f1_L, T_w_L, -T_f0_R, -T_f1_R, T_w_R]
```

**关键**: RL 不直接控制轮子力矩，只输出 4 维 VMC 参考值的偏移量。轮子由固定 PD 控制。

---

## 4. Observation (102-D) & Action (4-D)

```
Obs:
  [0:3]   base_ang_vel
  [3:6]   projected_gravity
  [6:9]   commands (v_x, v_y, ω_yaw)
  [9:11]  theta0 (virtual leg angle)
  [11:13] theta0_dot
  [13:15] L0 (virtual leg length)
  [15:17] L0_dot
  [17:19] wheel positions
  [19:21] wheel velocities
  [21:25] previous actions
  [25:102] terrain heights (77 points)

Action (4-D):
  [d_θ0_L, d_l0_L, d_θ0_R, d_l0_R]  ∈ [-1, 1]
```

---

## 5. 关键配置参数 (当前最优)

```python
# Control
feedforward_force = 15.0      # CRITICAL: 45→15, 纯PD速度 0.02→0.35 m/s (17.5×)
kp_l0 = 2000.0                # 1200→2000, 补偿低ff导致的高度下降
kp_theta = 150.0, kd_theta = 6.0
kd_l0 = 30.0
action_scale_theta = 1.0
action_scale_l0 = 0.1
l0_offset = 0.175
wheel_damping = 1.5           # parent: 0.5
lean_feedforward_gain = 0.0   # 测试过 -1.0~+1.0, 全部比纯PD差

# Residual curriculum
residual_ramp_start = 2_000 steps
residual_ramp_end   = 10_000 steps

# Reward
clip_single_reward = 5.0
tracking_sigma = 2.0
forward_speed = 4.0            # linear reward for speed
tracking_lin_vel = 2.0         # Gaussian tracking reward
orientation = -2.0
base_height = -2.0
tracking_lin_vel_enhance = 0.0 # DISABLED
nominal_state = 0.0            # DISABLED
```

---

## 6. 训练历史 (6 轮全部失败, peak ~0.6 m/s)

| Round | 关键改动 | Peak | 结果 |
|-------|---------|------|------|
| 1 | 原 VMC config (ff=45, sigma=0.25) | ~0.3 m/s | 停滞 |
| 2 | sigma=0.5, action_scale=0.5 | ~0.3 m/s | 同样停滞 |
| 3 | action_scale=1.0, [0,2.5] 固定命令 | ~0.5 m/s | 先升后降 |
| 4 | clip=5.0, sigma=2.0, forward_speed=4.0 | ~0.6 m/s | 短暂峰值后下降 |
| 5 | Flat terrain, 重调 reward | ~0.4 m/s | 同样模式 |
| 6 | ff=15, kp_l0=2000, wheel_damping=1.5 | ~0.5 m/s | iter~1700 时下降中 |

**共性**: 全部呈现 "初始上升 → 到达峰值 → 下降/振荡 → 收敛到 0.3-0.5" 的模式。

---

## 7. 根因分析

### 已解决

**A. feedforward_force = 45N 导致车轮打滑 (FIXED)**
45N 恒定向下力将重量从轮子抬起 → 轮子法向力低 → 摩擦力低 → 高速指令时严重打滑。
纯PD测试: ff=45 → 0.02 m/s, ff=15 → 0.35 m/s (17.5× 提升)。

**B. command curriculum 分母 bug (FIXED)**
`legged_robot.py` 3处用 `max_episode_length`(2000 steps) 代替 `max_episode_length_s`(20s),
导致阈值 100× 太严，命令范围永远无法扩展。

**C. wheel_vel_target 单位错误 (FIXED)**
缺少 `/wheel_radius` 和 `*track_half`，yaw 跟踪不正确。

**D. Feedforward lean 完全无效 (VERIFIED)**
测试了 -1.0 到 +1.0 的所有前馈倾斜增益，全部比纯 PD 差。

### 未解决: RL 训练停滞在 ~0.5 m/s

**H1 (最可能): 4-D action space 不够** — RL 不能控制轮子力矩。轮子是主要执行器，但 RL 只能间接通过改变身体姿态来影响它。相当于让你开车但只能调悬挂不能踩油门。

**H2: Residual curriculum 反效果** — RL 在 step 0-2000 无控制权(residual_scale=0), 此时 PD 在域随机化下性能极差(0.35→0.001 m/s), RL 观察到的数据全是噪声。

**H3: 域随机化太激进** — 摩擦 μ=0.1 相当于冰面, 任何策略在冰面上都无法跟踪速度, 极端梯度淹没正常学习信号。

**H4: 轮子 PD 阻尼太小** — damping=1.5, 消除 5 rad/s 误差只需 7.5 Nm, 电机有 15 Nm 可用但从未被充分利用。

---

## 8. 建议下一步 (给下一位工程师)

### 第一优先级: 架构修改
1. **扩展 action space 到 6-D**: 加 `[d_τ_wheel_L, d_τ_wheel_R]` 让 RL 直接控制轮子力矩 — 这是最大的架构问题
2. **移除 residual curriculum**: 从 step 0 开始 RL 就有完全控制权 (residual_scale=1)
3. **收紧域随机化**: 摩擦范围 0.3-1.5, 质量 ±1kg, 先用温和条件训练

### 第二优先级: 调参
4. 增加 wheel damping 到 3.0-5.0 (充分利用电机力矩)
5. 尝试纯线性奖励 (不要 Gaussian tracking)

---

## 9. 代码结构

```
wheel_legged_gym/
├── envs/
│   ├── base/
│   │   ├── legged_robot.py              # 基础类: 奖励, 域随机化, 终止, command curriculum
│   │   └── legged_robot_config.py
│   ├── wheel_legged/
│   │   └── wheel_legged_config.py       # URDF路径, 连杆长度, 碰撞
│   ├── wheel_legged_vmc/
│   │   ├── wheel_legged_vmc.py          # VMC基类: FK, Jacobian, theta0/L0计算
│   │   └── wheel_legged_vmc_config.py   # VMC默认参数 (kp/kd/ff)
│   └── wheel_legged_vmc_flat/
│       ├── wheel_legged_residual_flat.py         # **核心环境** (_compute_torques)
│       ├── wheel_legged_residual_flat_config.py  # **核心配置** (所有参数)
│       └── wheel_legged_vmc_flat_config.py
└── resources/robots/wl/urdf/wl.urdf
```

---

## 10. 常用命令

```bash
# 登录服务器
ssh -p 13834 root@connect.westc.seetacloud.com
# 密码: 0iafN9NJ8I0k

# 进入项目
cd /root/Wheel_Legged_Gym_2/wlr_ws/Wheel-Legged-Gym
export PATH=/root/miniconda3/bin:$PATH

# 训练
nohup python -u scripts/train.py --task wheel_legged_residual_flat --headless > logs/train.log 2>&1 &

# 查看训练进程
ps aux | grep train.py | grep -v grep

# 查看日志
tail -30 logs/train.log

# 查看最新 checkpoint
ls -lt logs/wheel_legged_residual_flat/*/model_*.pt | head -5

# 评估 checkpoint
python scripts/play.py --task wheel_legged_residual_flat --checkpoint <path>

# 停止训练
kill $(pgrep -f train.py)
```

---

## 11. 致谢

- 基础框架: [legged_gym](https://github.com/leggedrobotics/legged_gym) (ETH Zurich)
- 仿真: IsaacGym (NVIDIA)
