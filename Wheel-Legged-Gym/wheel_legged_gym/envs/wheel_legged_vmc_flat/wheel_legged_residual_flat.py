"""
Residual RL environment for the Wheel-Legged Robot (flat terrain).

Architecture:
    final_torque = LQR_baseline + residual_scale * RL_residual

The RL policy outputs 2-D actions [d_T, d_Tp]:
    d_T  : residual added to wheel torque  (symmetric, both wheels)
    d_Tp : residual added to leg lean torque (anti-symmetric: +left, -right)

Curriculum for residual_scale:
    steps 0 -- ramp_start        : scale = 0   (pure LQR, robot learns to stand)
    steps ramp_start -- ramp_end : scale 0 -> 1 (RL gradually takes over)
    steps ramp_end+              : scale = 1

Observation space (31-D):
    [0 :3 ]  base_ang_vel  * ang_vel_scale
    [3 :6 ]  projected_gravity
    [6 :9 ]  commands[:, :3] * commands_scale
    [9 :11]  theta0  * dof_pos_scale          (left, right virtual leg angle)
    [11:13]  theta0_dot * dof_vel_scale
    [13:15]  L0 * l0_scale                    (left, right virtual leg length)
    [15:17]  L0_dot * l0_dot_scale
    [17:19]  dof_pos[:, [2,5]] * dof_pos_scale (wheel positions)
    [19:21]  dof_vel[:, [2,5]] * dof_vel_scale (wheel velocities)
    [21:23]  actions                           (previous 2-D residual)
    [23:29]  lqr_state * lqr_state_scale       (6-D LQR state vector)
    [29:31]  lqr_output * lqr_output_scale     (2-D LQR torque output)
"""

import math

import torch
from torch import Tensor

from wheel_legged_gym.envs.wheel_legged_vmc.wheel_legged_vmc import LeggedRobotVMC
from .wheel_legged_residual_flat_config import WheelLeggedResidualFlatCfg
from .lqr_gpu import compute_lqr_output


class LeggedRobotResidual(LeggedRobotVMC):
    """Extends LeggedRobotVMC with gain-scheduled LQR baseline + 2-D RL residual."""

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

        # LQR buffers (populated in _compute_torques, read in compute_observations)
        self.lqr_state = torch.zeros(
            self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.lqr_output = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )

        # Per-env wheel position reference; reset to current position on episode reset
        self.wheel_pos_ref = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )

        # Curriculum scale for RL residual (0 -> 1 over training)
        self.residual_scale = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )

    # ------------------------------------------------------------------ #
    #  Reset                                                              #
    # ------------------------------------------------------------------ #

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        # Wheel position reference: set to current position so x_err starts at 0
        avg_wp = (self.dof_pos[env_ids, 2] + self.dof_pos[env_ids, 5]) * 0.5
        self.wheel_pos_ref[env_ids] = avg_wp
        self.lqr_state[env_ids] = 0.0
        self.lqr_output[env_ids] = 0.0

    # ------------------------------------------------------------------ #
    #  Residual-scale curriculum                                          #
    # ------------------------------------------------------------------ #

    def _update_residual_scale(self):
        """Linearly ramp residual_scale from 0 to 1 over training."""
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
    #  Torque computation (replaces clearlab's 6-DOF PD/VMC)             #
    # ------------------------------------------------------------------ #

    def _compute_torques(self, actions: Tensor) -> Tensor:
        """
        Compute joint torques: LQR baseline + residual_scale * RL residual.

        Args:
            actions: (num_envs, 2) = [d_T, d_Tp]  clipped residual actions

        Returns:
            torques: (num_envs, 6) = [lf0, lf1, l_wheel, rf0, rf1, r_wheel]
        """
        # ---- RL residual (scaled) --------------------------------------------
        d_T  = actions[:, 0] * self.cfg.control.residual_scale_T
        d_Tp = actions[:, 1] * self.cfg.control.residual_scale_Tp

        # ---- Wheel position reference update ---------------------------------
        # When moving: track wheel position (stop position error from growing)
        target_speed = self.commands[:, 0]
        avg_wheel_pos = (self.dof_pos[:, 2] + self.dof_pos[:, 5]) * 0.5
        moving = torch.abs(target_speed) > 0.05
        self.wheel_pos_ref = torch.where(moving, avg_wheel_pos, self.wheel_pos_ref)

        # ---- LQR baseline ----------------------------------------------------
        lqr_state, _avg_L0, lqr_out = compute_lqr_output(
            theta0=self.theta0,
            theta0_dot=self.theta0_dot,
            L0=self.L0,
            L0_dot=self.L0_dot,
            dof_pos_wheel=self.dof_pos[:, [2, 5]],
            dof_vel_wheel=self.dof_vel[:, [2, 5]],
            projected_gravity=self.projected_gravity,
            base_ang_vel=self.base_ang_vel,
            target_speed=target_speed,
            wheel_pos_ref=self.wheel_pos_ref,
        )
        self.lqr_state[:] = lqr_state
        self.lqr_output[:] = lqr_out

        lqr_T  = lqr_out[:, 0]   # raw LQR wheel torque
        lqr_Tp = lqr_out[:, 1]   # raw LQR leg lean torque

        # ---- Final wheel and leg lean torques --------------------------------
        # Firmware convention: wheel_T = -lqr_T * t_ratio
        wheel_T = -lqr_T * self.cfg.control.lqr_t_ratio + self.residual_scale * d_T

        # Left leg: +Tp, right leg: -Tp  (lean torque is antisymmetric)
        leg_Tp_L = (lqr_Tp * self.cfg.control.lqr_tp_ratio
                    + self.residual_scale * d_Tp)
        leg_Tp_R = (-lqr_Tp * self.cfg.control.lqr_tp_ratio
                    - self.residual_scale * d_Tp)

        # ---- Leg length PD -> virtual axial force ----------------------------
        l0_target = torch.clamp(
            self.commands[:, 2],
            self.cfg.control.l0_min,
            self.cfg.control.l0_max,
        )
        F_L = (
            self.l0_kp[:, 0] * (l0_target - self.L0[:, 0])
            - self.l0_kd[:, 0] * self.L0_dot[:, 0]
            + self.cfg.control.feedforward_force
        )
        F_R = (
            self.l0_kp[:, 1] * (l0_target - self.L0[:, 1])
            - self.l0_kd[:, 1] * self.L0_dot[:, 1]
            + self.cfg.control.feedforward_force
        )

        # ---- VMC Jacobian transpose: (F, Tp) -> (T1, T2) --------------------
        T1_L, T2_L = self._vmc_single(F_L, leg_Tp_L, side=0)
        T1_R, T2_R = self._vmc_single(F_R, leg_Tp_R, side=1)

        # ---- Yaw: differential wheel torque ----------------------------------
        yaw_T = self.commands[:, 1] * self.cfg.control.yaw_torque_scale

        # ---- Assemble joint torques ------------------------------------------
        # Order: [lf0, lf1, l_wheel, rf0, rf1, r_wheel]
        torques = torch.stack(
            [
                T1_L, T2_L, wheel_T + yaw_T,
                -T1_R, -T2_R, wheel_T - yaw_T,
            ],
            dim=-1,
        )
        return torch.clip(
            torques * self.torques_scale, -self.torque_limits, self.torque_limits
        )

    def _vmc_single(self, F: Tensor, Tp: Tensor, side: int):
        """
        VMC Jacobian transpose for one leg.

        Identical formula to LeggedRobotVMC.VMC() but per-leg so left/right
        can have different Tp values.

        Returns:
            T1, T2: (N,) joint torques for hip joint 0 and hip joint 1
        """
        l1 = self.cfg.asset.l1
        l2 = self.cfg.asset.l2

        theta0 = self.theta0[:, side] + math.pi / 2.0
        theta1 = self.theta1[:, side]
        theta2 = self.theta2[:, side]
        L0     = self.L0[:, side]

        t11 = (l1 * torch.sin(theta0 - theta1)
               - l2 * torch.sin(theta1 + theta2 - theta0))
        t12 = ((l1 * torch.cos(theta0 - theta1)
                - l2 * torch.cos(theta1 + theta2 - theta0))
               / (L0 + 1e-6))
        t21 = -l2 * torch.sin(theta1 + theta2 - theta0)
        t22 = (-l2 * torch.cos(theta1 + theta2 - theta0)
               / (L0 + 1e-6))

        T1 = t11 * F - t12 * Tp
        T2 = t21 * F - t22 * Tp
        return T1, T2

    # ------------------------------------------------------------------ #
    #  Observations                                                       #
    # ------------------------------------------------------------------ #

    def compute_proprioception_observations(self) -> Tensor:
        """
        31-D proprioceptive observation for residual RL.

        Compared to the standard VMC 27-D obs:
          - actions replaced with 2-D prev residual  (saves 4 dims)
          - appended: 6-D lqr_state + 2-D lqr_output  (adds 8 dims)
          net: 27 - 6 + 2 + 8 = 31
        """
        obs_scales = self.cfg.normalization.obs_scales
        obs = torch.cat(
            [
                self.base_ang_vel * obs_scales.ang_vel,              # [0:3]
                self.projected_gravity,                               # [3:6]
                self.commands[:, :3] * self.commands_scale,          # [6:9]
                self.theta0 * obs_scales.dof_pos,                    # [9:11]
                self.theta0_dot * obs_scales.dof_vel,                # [11:13]
                self.L0 * obs_scales.l0,                             # [13:15]
                self.L0_dot * obs_scales.l0_dot,                     # [15:17]
                self.dof_pos[:, [2, 5]] * obs_scales.dof_pos,        # [17:19]
                self.dof_vel[:, [2, 5]] * obs_scales.dof_vel,        # [19:21]
                self.actions,                                          # [21:23]
                self.lqr_state * obs_scales.lqr_state,               # [23:29]
                self.lqr_output * obs_scales.lqr_output,             # [29:31]
            ],
            dim=-1,
        )
        return obs

    def _get_noise_scale_vec(self, cfg):
        """Noise scale vector aligned with the 31-D observation."""
        noise_vec = torch.zeros(31, device=self.device, dtype=torch.float)
        ns  = cfg.noise.noise_scales
        nl  = cfg.noise.noise_level
        obs = cfg.normalization.obs_scales

        noise_vec[0:3]   = ns.ang_vel   * nl * obs.ang_vel
        noise_vec[3:6]   = ns.gravity   * nl
        noise_vec[6:9]   = 0.0                                   # commands
        noise_vec[9:11]  = ns.dof_pos   * nl * obs.dof_pos       # theta0
        noise_vec[11:13] = ns.dof_vel   * nl * obs.dof_vel       # theta0_dot
        noise_vec[13:15] = ns.l0        * nl * obs.l0            # L0
        noise_vec[15:17] = ns.l0_dot    * nl * obs.l0_dot        # L0_dot
        noise_vec[17:19] = ns.dof_pos   * nl * obs.dof_pos       # wheel pos
        noise_vec[19:21] = ns.dof_vel   * nl * obs.dof_vel       # wheel vel
        noise_vec[21:23] = 0.0                                   # prev actions
        noise_vec[23:29] = 0.0                                   # lqr_state (derived)
        noise_vec[29:31] = 0.0                                   # lqr_output
        return noise_vec

    # ------------------------------------------------------------------ #
    #  Reward                                                             #
    # ------------------------------------------------------------------ #

    def _reward_residual_magnitude_l2(self) -> Tensor:
        """
        Penalise large RL residuals.

        Returns (num_envs,) positive scalar; scale is set to -0.1 in config
        so the effective reward is negative.
        """
        return torch.sum(self.actions ** 2, dim=-1)
