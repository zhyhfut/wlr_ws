"""
Configuration for Wheel-Legged Robot Residual RL (LQR + PPO, flat terrain).

Key differences vs standard WheelLeggedVMCFlatCfg:
  - num_actions  = 2  (RL only outputs [d_T, d_Tp] residual)
  - num_observations = 31  (includes LQR state & output in obs)
  - control gains: LQR ratios + residual scaling + ramp curriculum
  - reward: adds residual_magnitude_l2 penalty
  - policy: ActorCriticSequence with 5-step obs history encoder
"""

from wheel_legged_gym.envs.wheel_legged_vmc_flat.wheel_legged_vmc_flat_config import (
    WheelLeggedVMCFlatCfg,
    WheelLeggedVMCFlatCfgPPO,
)


class WheelLeggedResidualFlatCfg(WheelLeggedVMCFlatCfg):

    class env(WheelLeggedVMCFlatCfg.env):
        num_observations = 31   # 3+3+3+2+2+2+2+2+2+2+6+2
        num_actions      = 2    # [d_T, d_Tp]
        # No privileged obs by default; set to an integer to enable teacher-student
        num_privileged_obs = None

    class control(WheelLeggedVMCFlatCfg.control):
        # ---------- LQR ratios (from firmware balance_task.c) ----------------
        lqr_t_ratio  = 0.8   # scale applied to LQR wheel torque output
        lqr_tp_ratio = 0.4   # scale applied to LQR leg lean torque output

        # ---------- Residual RL action scaling --------------------------------
        # Final torque contribution = residual_scale * action * residual_scale_*
        residual_scale_T  = 2.0   # max extra wheel torque from RL  [Nm]
        residual_scale_Tp = 1.0   # max extra leg lean torque from RL [Nm]

        # ---------- Residual curriculum ---------------------------------------
        # Step 0 -- ramp_start  : residual_scale = 0  (pure LQR)
        # Step ramp_start -- ramp_end : linear ramp 0 -> 1
        # Step ramp_end+  : residual_scale = 1
        residual_ramp_start_steps = 50_000
        residual_ramp_end_steps   = 250_000

        # ---------- Leg length ------------------------------------------------
        l0_offset         = 0.175   # default target leg length [m]
        l0_min            = 0.10    # lower clamp [m]
        l0_max            = 0.25    # upper clamp [m]
        feedforward_force = 40.0    # constant gravity-compensation force [N]

        # ---------- Yaw -------------------------------------------------------
        yaw_torque_scale = 0.5   # differential wheel torque gain [Nm/(rad/s)]

        # Residual RL does NOT use position/velocity PD targets;
        # stiffness/damping are kept at 0 so Isaac Gym PD doesn't interfere.
        stiffness = {"f0": 0.0, "f1": 0.0, "wheel": 0}
        damping   = {"f0": 0.0, "f1": 0.0, "wheel": 0.5}

    class normalization(WheelLeggedVMCFlatCfg.normalization):
        class obs_scales(WheelLeggedVMCFlatCfg.normalization.obs_scales):
            lqr_state  = 0.2    # LQR state vector is O(1-5 rad/m)
            lqr_output = 0.05   # LQR output torques are O(5-20 Nm)

    class rewards(WheelLeggedVMCFlatCfg.rewards):
        class scales(WheelLeggedVMCFlatCfg.rewards.scales):
            # Additional penalty: encourage small RL residuals
            # (negative weight = penalty, value returned by _reward_* is positive)
            residual_magnitude_l2 = -0.1

    class noise(WheelLeggedVMCFlatCfg.noise):
        class noise_scales(WheelLeggedVMCFlatCfg.noise.noise_scales):
            l0     = 0.02   # [m]
            l0_dot = 0.10   # [m/s]


class WheelLeggedResidualFlatCfgPPO(WheelLeggedVMCFlatCfgPPO):
    """PPO training configuration for the residual RL policy."""

    class policy(WheelLeggedVMCFlatCfgPPO.policy):
        # Sequence encoder: 5 steps x 31-D obs -> 3-D latent
        num_encoder_obs = 155    # obs_history_length (5) x num_observations (31)
        latent_dim       = 3
        actor_hidden_dims  = [128, 64, 32]
        critic_hidden_dims = [256, 128, 64]
        activation = "elu"

    class algorithm(WheelLeggedVMCFlatCfgPPO.algorithm):
        entropy_coef = 0.01     # encourage exploration early in training

    class runner(WheelLeggedVMCFlatCfgPPO.runner):
        experiment_name    = "wheel_legged_residual_flat"
        max_iterations     = 2000
        policy_class_name  = "ActorCriticSequence"
