"""
Residual RL environment for the Wheel-Legged Robot (flat terrain).

Architecture:
    final_torque = VMC_PD(baseline_target + residual_scale * RL_residual)

The RL policy outputs 6-D actions interpreted as residuals to PD targets:
    actions = [d_theta0_L, d_L0_L, d_wheel_L, d_theta0_R, d_L0_R, d_wheel_R]

Baseline PD targets (pure VMC PD controller, works standalone):
    theta0_ref = 0                 (stand upright)
    L0_ref     = height_command    (from user command)
    wheel_ref  = (fwd_vel ± yaw * half_track) / wheel_radius

Curriculum for residual_scale:
    steps 0 -- ramp_start        : scale = 0   (pure baseline PD)
    steps ramp_start -- ramp_end : scale 0 -> 1
    steps ramp_end+              : scale = 1

Observation space (27-D, identical to standard VMC):
    [0 :3 ]  base_ang_vel  * ang_vel_scale
    [3 :6 ]  projected_gravity
    [6 :9 ]  commands[:, :3] * commands_scale
    [9 :11]  theta0  * dof_pos_scale
    [11:13]  theta0_dot * dof_vel_scale
    [13:15]  L0 * l0_scale
    [15:17]  L0_dot * l0_dot_scale
    [17:19]  dof_pos[:, [2,5]] * dof_pos_scale   (wheel positions)
    [19:21]  dof_vel[:, [2,5]] * dof_vel_scale   (wheel velocities)
    [21:27]  actions                               (previous 6-D residual)
"""

import torch
from torch import Tensor

from wheel_legged_gym.envs.wheel_legged_vmc.wheel_legged_vmc import LeggedRobotVMC
from .wheel_legged_residual_flat_config import WheelLeggedResidualFlatCfg

# URDF constants
WHEEL_RADIUS = 0.0675   # [m]  from wl.urdf collision cylinder
HALF_TRACK   = 0.25     # [m]  lateral offset of wheel joints in URDF


class LeggedRobotResidual(LeggedRobotVMC):
    """Extends LeggedRobotVMC: VMC PD baseline + 6-D RL residual on top."""

    def __init__(
        self,
        cfg: WheelLeggedResidualFlatCfg,
        sim_params,
        physics_engine,
        sim_device,
        headless,
    ):
        self.cfg = cfg
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    # ------------------------------------------------------------------ #
    #  Buffer initialization                                              #
    # ------------------------------------------------------------------ #

    def _init_buffers(self):
        super()._init_buffers()

        # Curriculum scale for RL residual (0 -> 1 over training)
        self.residual_scale = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )

    # ------------------------------------------------------------------ #
    #  Reset                                                              #
    # ------------------------------------------------------------------ #

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        # residual_scale is overwritten each step() by _update_residual_scale,
        # so no per-env reset needed here.

    # ------------------------------------------------------------------ #
    #  Residual-scale curriculum                                          #
    # ------------------------------------------------------------------ #

    def _update_residual_scale(self):
        """Linearly ramp residual_scale from 0 to 1 over training steps."""
        step       = self.common_step_counter
        ramp_start = self.cfg.control.residual_ramp_start_steps
        ramp_end   = self.cfg.control.residual_ramp_end_steps

        if step < ramp_start:
            scale = 0.0
        elif step < ramp_end:
            scale = float(step - ramp_start) / float(ramp_end - ramp_start)
        else:
            scale = 1.0

        self.residual_scale.fill_(scale)

    # ------------------------------------------------------------------ #
    #  Step override -- update curriculum before physics                  #
    # ------------------------------------------------------------------ #

    def step(self, actions):
        self._update_residual_scale()
        return super().step(actions)

    # ------------------------------------------------------------------ #
    #  Torque computation (VMC PD baseline + RL residual)                 #
    # ------------------------------------------------------------------ #

    def _compute_torques(self, actions: Tensor) -> Tensor:
        """
        Compute joint torques: VMC PD(baseline + residual_scale * residual).

        Args:
            actions: (num_envs, 6) = [d_theta0_L, d_L0_L, d_wheel_L,
                                       d_theta0_R, d_L0_R, d_wheel_R]

        Returns:
            torques: (num_envs, 6) = [lf0, lf1, l_wheel, rf0, rf1, r_wheel]
        """
        cfg = self.cfg.control

        # ---- Baseline PD targets --------------------------------------------
        # theta0: upright (zero)
        baseline_theta0 = torch.zeros(self.num_envs, 2, device=self.device)

        # L0: directly from height command
        baseline_L0 = self.commands[:, 2].unsqueeze(1).expand(-1, 2)

        # Wheel velocity: forward speed + differential yaw
        fwd_vel = self.commands[:, 0]
        yaw_vel = self.commands[:, 1]
        wheel_L = (fwd_vel - yaw_vel * HALF_TRACK) / WHEEL_RADIUS
        wheel_R = (fwd_vel + yaw_vel * HALF_TRACK) / WHEEL_RADIUS
        baseline_wheel = torch.stack([wheel_L, wheel_R], dim=1)

        # ---- RL residual (scaled by curriculum) -----------------------------
        rs = self.residual_scale.unsqueeze(1)  # (N, 1)
        d_theta0 = rs * actions[:, [0, 3]] * cfg.residual_scale_theta0
        d_L0     = rs * actions[:, [1, 4]] * cfg.residual_scale_l0
        d_wheel  = rs * actions[:, [2, 5]] * cfg.residual_scale_wheel

        # ---- Final PD targets -----------------------------------------------
        theta0_ref = baseline_theta0 + d_theta0
        L0_ref     = baseline_L0     + d_L0
        wheel_ref  = baseline_wheel  + d_wheel

        # ---- PD control (same gains as parent VMC) --------------------------
        self.torque_leg = (
            self.theta_kp * (theta0_ref - self.theta0)
            - self.theta_kd * self.theta0_dot
        )
        self.force_leg = (
            self.l0_kp * (L0_ref - self.L0)
            - self.l0_kd * self.L0_dot
        )
        self.torque_wheel = self.d_gains[:, [2, 5]] * (
            wheel_ref - self.dof_vel[:, [2, 5]]
        )

        # ---- VMC Jacobian: (F, T) -> joint torques --------------------------
        T1, T2 = self.VMC(
            self.force_leg + cfg.feedforward_force, self.torque_leg
        )

        # ---- Assemble -------------------------------------------------------
        # Order: [lf0, lf1, l_wheel, rf0, rf1, r_wheel]
        # Right-side VMC torques are negated (mirrored leg convention)
        torques = torch.stack(
            [
                T1[:, 0], T2[:, 0], self.torque_wheel[:, 0],
                -T1[:, 1], -T2[:, 1], self.torque_wheel[:, 1],
            ],
            dim=-1,
        )
        return torch.clip(
            torques * self.torques_scale, -self.torque_limits, self.torque_limits
        )

    # ------------------------------------------------------------------ #
    #  Reward                                                             #
    # ------------------------------------------------------------------ #

    def _reward_residual_magnitude_l2(self) -> Tensor:
        """
        Penalise large RL residuals.

        Returns (num_envs,) positive scalar; config sets scale = -0.1
        so the effective reward contribution is negative.
        """
        return torch.sum(self.actions ** 2, dim=-1)
