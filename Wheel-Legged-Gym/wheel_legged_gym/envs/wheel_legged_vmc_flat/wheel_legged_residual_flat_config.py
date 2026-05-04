"""
Configuration for Wheel-Legged Robot Residual RL (LQR baseline + PPO, flat terrain).

Architecture:
    final_torque = VMC_PD(LQR_baseline + residual_scale * RL_residual, L0_PD)

The LQR baseline is a planar controller operating in the virtual-leg domain:
  - 6-D state  → 2-D output  [T_wheel, Tp_leg]
  - Gain-scheduled on virtual leg length L0 (cubic polynomials, pre-computed offline)
  - T_wheel → wheel drive torque (symmetric)
  - Tp_leg  → leg lean torque for pitch (SYMMETRIC across both legs)

RL outputs 2-D residuals [d_T, d_Tp] in the same virtual-torque space,
applied symmetrically to both legs. Roll stability comes from the theta0
PD controller (kp_theta=50).

Key differences vs standard WheelLeggedVMCFlatCfg:
  - num_actions      = 2   (RL only outputs [d_T, d_Tp] residual)
  - num_observations = 31  (includes LQR state & output in obs)
  - LQR gain scheduling replaces fixed PD targets
  - Curriculum ramp on residual_scale: 0 (pure LQR) -> 1 (LQR + RL)
"""

from wheel_legged_gym.envs.wheel_legged_vmc_flat.wheel_legged_vmc_flat_config import (
    WheelLeggedVMCFlatCfg,
    WheelLeggedVMCFlatCfgPPO,
)


class WheelLeggedResidualFlatCfg(WheelLeggedVMCFlatCfg):

    class env(WheelLeggedVMCFlatCfg.env):
        num_observations = 31   # 3+3+3+2+2+2+2+2+2+2+6+2
        num_actions      = 2    # [d_T, d_Tp]

    class control(WheelLeggedVMCFlatCfg.control):
        # ---------- LQR ratios ------------------------------------------------
        # Scale applied to raw LQR output before it enters the VMC pipeline.
        # Signs follow firmware convention (wheel torque is negated).
        lqr_t_ratio  = 0.8   # scale on LQR wheel torque
        lqr_tp_ratio = 0.4   # scale on LQR leg lean torque (symmetric)

        # ---------- Residual RL action scaling ---------------------------------
        # RL residual is interpreted as a virtual torque in [Nm]:
        #   total = LQR_baseline * ratio + residual_scale * action * scale
        residual_scale_T  = 2.0   # max extra wheel torque from RL  [Nm]
        residual_scale_Tp = 1.0   # max extra leg lean torque from RL [Nm]

        # ---------- Residual curriculum ----------------------------------------
        # Step 0  .. ramp_start  : residual_scale = 0  (pure LQR)
        # Step ramp_start .. ramp_end : linear ramp 0 -> 1
        # Step ramp_end+         : residual_scale = 1
        residual_ramp_start_steps = 10_000
        residual_ramp_end_steps   = 80_000

        # ---------- Leg length ------------------------------------------------
        l0_offset         = 0.175   # default target leg length [m]
        l0_min            = 0.10    # lower clamp [m]
        l0_max            = 0.25    # upper clamp [m]
        feedforward_force = 40.0    # constant gravity-compensation force [N]

        # ---------- Yaw -------------------------------------------------------
        yaw_torque_scale = 0.5   # differential wheel torque gain [Nm/(rad/s)]

        # Stiffness/damping kept at 0 for joint-level PD; virtual-leg PD
        # uses kp_theta/kd_theta/kp_l0/kd_l0 which are in parent config.
        stiffness = {"f0": 0.0, "f1": 0.0, "wheel": 0}
        damping   = {"f0": 0.0, "f1": 0.0, "wheel": 0.5}

    class normalization(WheelLeggedVMCFlatCfg.normalization):
        class obs_scales(WheelLeggedVMCFlatCfg.normalization.obs_scales):
            lqr_state  = 0.2    # LQR state is O(1-5 rad/m)
            lqr_output = 0.05   # LQR output torques are O(5-20 Nm)

    class rewards(WheelLeggedVMCFlatCfg.rewards):
        class scales(WheelLeggedVMCFlatCfg.rewards.scales):
            # Encourage small RL residuals (negative = penalty)
            residual_magnitude_l2 = -0.1

    class noise(WheelLeggedVMCFlatCfg.noise):
        class noise_scales(WheelLeggedVMCFlatCfg.noise.noise_scales):
            l0     = 0.02   # [m]
            l0_dot = 0.10   # [m/s]


class WheelLeggedResidualFlatCfgPPO(WheelLeggedVMCFlatCfgPPO):
    """PPO training configuration for the residual RL policy."""

    class policy(WheelLeggedVMCFlatCfgPPO.policy):
        # 5 steps x 31-D obs -> encoder input
        num_encoder_obs = 155   # obs_history_length (5) * num_observations (31)
        latent_dim       = 3
        actor_hidden_dims  = [128, 64, 32]
        critic_hidden_dims = [256, 128, 64]
        activation = "elu"

    class algorithm(WheelLeggedVMCFlatCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(WheelLeggedVMCFlatCfgPPO.runner):
        experiment_name    = "wheel_legged_residual_flat"
        max_iterations     = 2000
        policy_class_name  = "ActorCriticSequence"
