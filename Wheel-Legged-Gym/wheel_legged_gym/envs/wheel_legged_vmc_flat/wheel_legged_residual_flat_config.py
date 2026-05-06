"""
Configuration for Wheel-Legged Robot Residual RL (VMC PD baseline + PPO, rough terrain).

Architecture:
    final_torque = VMC(PD(theta0_ref + residual_scale * d_theta0, l0_ref + residual_scale * d_l0))

The PD baseline provides basic stabilization in VMC task space:
  - theta0_ref = 0 (upright) + RL residual
  - l0_ref = l0_offset + RL residual
  - wheel torque from command tracking (no RL residual)

RL outputs 4-D residuals [d_theta0_L, d_l0_L, d_theta0_R, d_l0_R]
applied per-leg to PD targets. Rough terrain requires independent leg control.

Key differences vs standard WheelLeggedVMCFlatCfg:
  - num_actions      = 4   (per-leg VMC reference residuals)
  - num_observations = 102 (includes 77 terrain height points)
  - Terrain is trimesh with curriculum (NOT flat plane)
  - PD gains are tuned for baseline stability (kp_theta=150)
  - Curriculum ramp on residual_scale: 0 (pure PD) -> 1 (PD + RL)
"""

from wheel_legged_gym.envs.wheel_legged_vmc_flat.wheel_legged_vmc_flat_config import (
    WheelLeggedVMCFlatCfg,
    WheelLeggedVMCFlatCfgPPO,
)


class WheelLeggedResidualFlatCfg(WheelLeggedVMCFlatCfg):

    class env(WheelLeggedVMCFlatCfg.env):
        num_observations = 102  # 25(proprio) + 77(height points)
        num_actions      = 4    # [d_theta0_L, d_l0_L, d_theta0_R, d_l0_R]
        num_privileged_obs = None  # disable asymmetric critic

    class terrain(WheelLeggedVMCFlatCfg.terrain):
        curriculum = False  # disable terrain auto-leveling — learn speed on flat first
        max_init_terrain_level = 0

    class commands(WheelLeggedVMCFlatCfg.commands):
        curriculum = True   # command curriculum: start narrow, expand as tracking improves
        class ranges(WheelLeggedVMCFlatCfg.commands.ranges):
            lin_vel_x = [0.0, 0.5]    # narrow start — curriculum expands to 2.5
            ang_vel_yaw = [-6.28, 6.28]  # wider yaw range

    class control(WheelLeggedVMCFlatCfg.control):
        # ---------- Residual RL action scaling ---------------------------------
        # RL residual shifts the PD target in VMC task space.
        action_scale_theta = 1.0   # scale on d_theta0 -> theta0_ref shift [rad]
        action_scale_l0   = 0.1    # scale on d_l0 -> l0_ref shift [m]

        # ---------- Feedforward lean ------------------------------------------
        # DISABLED after physical verification: velocity-error-based lean does
        # NOT improve speed (all gains 0.15-1.0 tested, all worse than pure PD).
        lean_feedforward_gain = 0.0

        # ---------- Residual curriculum ----------------------------------------
        # Step 0  .. ramp_start  : residual_scale = 0  (pure PD)
        # Step ramp_start .. ramp_end : linear ramp 0 -> 1
        # Step ramp_end+         : residual_scale = 1
        residual_ramp_start_steps = 2_000
        residual_ramp_end_steps   = 10_000

        # ---------- VMC PD gains (override parent) -----------------------------
        # feedforward_force reduced 45→15: verified that high ff steals wheel
        # normal force → loss of traction → wheels spin uselessly at high cmd.
        # At ff=15, pure PD reaches ~0.35 m/s vs 0.02 at ff=45 (17.5x gain).
        # kp_l0 boosted 1200→2000 to compensate for lower feedforward force
        # and keep the robot from sitting too low.
        feedforward_force = 15.0   # [N] (parent: 45)
        kp_l0            = 2000.0  # [N/m] (parent: 1200)

        # ---------- Leg length ------------------------------------------------
        l0_offset         = 0.175   # default target leg length [m]
        l0_min            = 0.10    # lower clamp [m]
        l0_max            = 0.25    # upper clamp [m]

        # ---------- Yaw -------------------------------------------------------
        yaw_torque_scale = 0.5   # differential wheel torque gain [Nm/(rad/s)]

        # Stiffness/damping: joint-level PD disabled; virtual-leg PD handles control.
        # Reduced wheel damping 3.0→1.5 for smoother acceleration ramp-up.
        stiffness = {"f0": 0.0, "f1": 0.0, "wheel": 0}
        damping   = {"f0": 0.0, "f1": 0.0, "wheel": 1.5}

    class normalization(WheelLeggedVMCFlatCfg.normalization):
        class obs_scales(WheelLeggedVMCFlatCfg.normalization.obs_scales):
            height_measurements = 5.0

    class rewards(WheelLeggedVMCFlatCfg.rewards):
        clip_single_reward = 5.0  # ceiling 0.05/step
        tracking_sigma = 2.0  # exp(-err²/2): exp(-2.5²/2)=0.044, gradient alive across full [0,2.5] range
        class scales(WheelLeggedVMCFlatCfg.rewards.scales):
            nominal_state = 0.0
            base_height = -2.0
            tracking_lin_vel = 2.0   # max 0.02/step
            forward_speed = 4.0      # linear: 4.0*vel, stronger constant gradient
            tracking_lin_vel_enhance = 0.0
            orientation = -2.0

    class noise(WheelLeggedVMCFlatCfg.noise):
        class noise_scales(WheelLeggedVMCFlatCfg.noise.noise_scales):
            l0     = 0.02   # [m]
            l0_dot = 0.10   # [m/s]


class WheelLeggedResidualFlatCfgPPO(WheelLeggedVMCFlatCfgPPO):
    """PPO training configuration for the residual RL policy."""

    class policy(WheelLeggedVMCFlatCfgPPO.policy):
        # 5 steps x 102-D obs -> encoder input
        num_encoder_obs = 510   # obs_history_length (5) * num_observations (102)
        latent_dim       = 4
        actor_hidden_dims  = [256, 128, 64]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"

    class algorithm(WheelLeggedVMCFlatCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(WheelLeggedVMCFlatCfgPPO.runner):
        experiment_name    = "wheel_legged_residual_flat"
        max_iterations     = 6000
        policy_class_name  = "ActorCriticSequence"
