"""
Residual RL environment for the Wheel-Legged Robot (rough terrain).

Architecture:
    final_torque = VMC(PD(theta0_ref_residual, l0_ref_residual), feedforward_force)

The PD baseline provides basic stabilization:
  - theta0_ref = 0 + residual_scale * d_theta0   (upright + RL adjustment)
  - l0_ref     = l0_offset + residual_scale * d_l0  (nominal height + RL adjustment)
  - wheel torque from command speed tracking (no RL residual on wheels)

RL outputs 4-D actions [d_theta0_L, d_l0_L, d_theta0_R, d_l0_R] as per-leg
adjustments to the PD targets in VMC task space.

Curriculum for residual_scale:
    steps 0  -- ramp_start        : scale = 0  (pure PD)
    steps ramp_start -- ramp_end : scale 0 -> 1
    steps ramp_end+               : scale = 1

Observation space (102-D):
    [0 :3 ]  base_ang_vel  * ang_vel_scale
    [3 :6 ]  projected_gravity
    [6 :9 ]  commands[:, :3] * commands_scale
    [9 :11]  theta0  * dof_pos_scale
    [11:13]  theta0_dot * dof_vel_scale
    [13:15]  L0 * l0_scale
    [15:17]  L0_dot * l0_dot_scale
    [17:19]  dof_pos[:, [2,5]] * dof_pos_scale   (wheel positions)
    [19:21]  dof_vel[:, [2,5]] * dof_vel_scale   (wheel velocities)
    [21:25]  actions                               (previous 4-D residual)
    [25:102] relative_heights * height_measurements_scale  (77 terrain height points)
"""

import math

import torch
from torch import Tensor

from wheel_legged_gym.envs.wheel_legged_vmc.wheel_legged_vmc import LeggedRobotVMC
from .wheel_legged_residual_flat_config import WheelLeggedResidualFlatCfg


class LeggedRobotResidual(LeggedRobotVMC):
    """Extends LeggedRobotVMC: PD baseline + 4-D RL residual for rough terrain."""

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
            self.num_envs, dtype=torch.float, device=self.device,
            requires_grad=False,
        )

        # Override torques buffer: parent uses num_actions (4) but we need
        # num_dof (6) since _compute_torques outputs per-DOF joint torques.
        self.torques = torch.zeros(
            self.num_envs, self.num_dof, dtype=torch.float,
            device=self.device, requires_grad=False,
        )

    # ------------------------------------------------------------------ #
    #  Reset                                                              #
    # ------------------------------------------------------------------ #

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)

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
    #  Torque computation (PD baseline with RL residual on VMC refs)     #
    # ------------------------------------------------------------------ #

    def _compute_torques(self, actions: Tensor) -> Tensor:
        """
        Compute joint torques: PD baseline with RL residual shifting VMC targets.

        Args:
            actions: (num_envs, 4) = [d_theta0_L, d_l0_L, d_theta0_R, d_l0_R]

        Returns:
            torques: (num_envs, 6) = [lf0, lf1, l_wheel, rf0, rf1, r_wheel]
        """
        cfg = self.cfg.control
        rs = self.residual_scale  # (num_envs,)

        # ---- Build VMC references from RL residuals --------------------------
        # theta0_ref: baseline is 0 (stand upright), RL adds per-leg shift
        # (num_envs, 2)  [left, right]
        theta0_ref = torch.stack(
            [
                rs * actions[:, 0] * cfg.action_scale_theta,   # left leg
                rs * actions[:, 2] * cfg.action_scale_theta,   # right leg
            ],
            dim=-1,
        )

        # l0_ref: baseline is l0_offset, RL adds per-leg shift
        l0_ref = torch.stack(
            [
                cfg.l0_offset + rs * actions[:, 1] * cfg.action_scale_l0,  # left
                cfg.l0_offset + rs * actions[:, 3] * cfg.action_scale_l0,  # right
            ],
            dim=-1,
        )

        # ---- PD control in VMC task space ------------------------------------
        # Leg torque (tangential to L0): PD on theta0
        torque_leg = (
            self.theta_kp * (theta0_ref - self.theta0)
            - self.theta_kd * self.theta0_dot
        )  # (num_envs, 2)

        # Leg force (radial along L0): PD on L0
        force_leg = (
            self.l0_kp * (l0_ref - self.L0)
            - self.l0_kd * self.L0_dot
        )  # (num_envs, 2)

        # ---- Wheel torque: track command speed (no RL residual) --------------
        target_speed = self.commands[:, 0]  # (num_envs,)
        yaw_command  = self.commands[:, 1]  # (num_envs,)

        wheel_vel_target = torch.stack(
            [
                target_speed + yaw_command,   # left wheel
                target_speed - yaw_command,   # right wheel
            ],
            dim=-1,
        )  # (num_envs, 2)

        torque_wheel = self.d_gains[:, [2, 5]] * (
            wheel_vel_target - self.dof_vel[:, [2, 5]]
        )  # (num_envs, 2)

        # ---- VMC Jacobian transpose: (F, Tp) -> (T1, T2) --------------------
        T1, T2 = self.VMC(
            force_leg + cfg.feedforward_force, torque_leg,
        )  # each (num_envs, 2)

        # ---- Assemble --------------------------------------------------------
        torques = torch.stack(
            [
                T1[:, 0], T2[:, 0], torque_wheel[:, 0],   # left:  f0, f1, wheel
                -T1[:, 1], -T2[:, 1], torque_wheel[:, 1], # right: f0, f1, wheel (negated)
            ],
            dim=-1,
        )
        return torch.clip(
            torques * self.torques_scale, -self.torque_limits, self.torque_limits
        )

    # ------------------------------------------------------------------ #
    #  Observations                                                       #
    # ------------------------------------------------------------------ #

    def compute_proprioception_observations(self) -> Tensor:
        """
        102-D proprioceptive observation for residual RL on rough terrain.

        Base VMC observation (25-D) + 77 terrain height points.
        """
        obs_scales = self.cfg.normalization.obs_scales

        # Base proprioception (same as parent VMC, 25-D with 4-D actions)
        proprio = torch.cat(
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
                self.actions,                                          # [21:25]
            ],
            dim=-1,
        )

        # Terrain height samples under and around the robot
        if self.cfg.terrain.measure_heights:
            if torch.is_tensor(self.measured_heights):
                # Relative height: how far below (or above) the base is the ground
                rel_heights = (
                    self.root_states[:, 2].unsqueeze(1) - self.measured_heights
                ).clamp(-0.2, 0.2) * obs_scales.height_measurements
            else:
                # measured_heights not populated yet (e.g. during first reset),
                # use zeros of correct shape
                rel_heights = torch.zeros(
                    self.num_envs, self.num_height_points,
                    device=self.device, dtype=torch.float,
                )
            obs = torch.cat([proprio, rel_heights], dim=-1)
        else:
            obs = proprio

        return obs

    def _get_noise_scale_vec(self, cfg):
        """Noise scale vector aligned with the 102-D observation."""
        self.add_noise = self.cfg.noise.add_noise
        noise_vec = torch.zeros(
            self.cfg.env.num_observations, device=self.device, dtype=torch.float,
        )
        ns  = cfg.noise.noise_scales
        nl  = cfg.noise.noise_level
        obs = cfg.normalization.obs_scales

        noise_vec[0:3]   = ns.ang_vel   * nl * obs.ang_vel
        noise_vec[3:6]   = ns.gravity   * nl
        noise_vec[6:9]   = 0.0                                    # commands
        noise_vec[9:11]  = ns.dof_pos   * nl * obs.dof_pos        # theta0
        noise_vec[11:13] = ns.dof_vel   * nl * obs.dof_vel        # theta0_dot
        noise_vec[13:15] = ns.l0        * nl * obs.l0             # L0
        noise_vec[15:17] = ns.l0_dot    * nl * obs.l0_dot         # L0_dot
        noise_vec[17:19] = ns.dof_pos   * nl * obs.dof_pos        # wheel pos
        noise_vec[19:21] = ns.dof_vel   * nl * obs.dof_vel        # wheel vel
        noise_vec[21:25] = 0.0                                    # prev actions
        noise_vec[25:102] = ns.height_measurements * nl * obs.height_measurements
        return noise_vec
