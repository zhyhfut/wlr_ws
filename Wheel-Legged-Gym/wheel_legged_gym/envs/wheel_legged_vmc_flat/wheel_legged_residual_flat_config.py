"""
Configuration for Wheel-Legged Robot Residual RL (VMC PD baseline + PPO, flat terrain).

Architecture:
    final_torque = VMC_PD(baseline_target + residual_scale * RL_residual)

Key differences vs standard WheelLeggedVMCFlatCfg:
  - RL outputs 6-D residuals (same action dim) added to baseline PD targets
  - Curriculum ramp on residual_scale: 0 (pure baseline) -> 1 (full RL)
  - residual_magnitude_l2 penalty encourages the policy to stay near the baseline
  - Same 27-D observation space as the original VMC policy
"""

from wheel_legged_gym.envs.wheel_legged_vmc_flat.wheel_legged_vmc_flat_config import (
    WheelLeggedVMCFlatCfg,
    WheelLeggedVMCFlatCfgPPO,
)


class WheelLeggedResidualFlatCfg(WheelLeggedVMCFlatCfg):
    """Residual RL config: VMC PD baseline + 6-D RL residual on top."""

    class control(WheelLeggedVMCFlatCfg.control):
        # ---------- Residual RL action scaling --------------------------------
        # RL actions are residuals in physical units, added to baseline targets:
        #   theta0_ref = 0              + residual_scale * action * scale_theta0
        #   L0_ref     = height_cmd     + residual_scale * action * scale_l0
        #   wheel_ref  = fwd/yaw_baseline + residual_scale * action * scale_wheel
        residual_scale_theta0 = 0.2    # [rad] per unit action
        residual_scale_l0     = 0.05   # [m]  per unit action
        residual_scale_wheel  = 5.0    # [rad/s] per unit action

        # ---------- Residual curriculum ---------------------------------------
        # Step 0 .. ramp_start      : residual_scale = 0  (pure PD baseline)
        # Step ramp_start .. ramp_end: linear ramp 0 -> 1
        # Step ramp_end+            : residual_scale = 1
        residual_ramp_start_steps = 5_000
        residual_ramp_end_steps   = 40_000

    class rewards(WheelLeggedVMCFlatCfg.rewards):
        class scales(WheelLeggedVMCFlatCfg.rewards.scales):
            # Penalize large RL residuals (negative weight = penalty)
            residual_magnitude_l2 = -0.1


class WheelLeggedResidualFlatCfgPPO(WheelLeggedVMCFlatCfgPPO):
    """PPO training configuration for the residual RL policy."""

    class runner(WheelLeggedVMCFlatCfgPPO.runner):
        experiment_name = "wheel_legged_residual_flat"
        max_iterations = 2000
