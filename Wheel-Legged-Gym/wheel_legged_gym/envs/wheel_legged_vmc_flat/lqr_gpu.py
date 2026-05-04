"""
GPU-vectorized LQR gain scheduling for WLR residual RL.

Ports firmware lqr_gains.py (numpy, single-env) to batched PyTorch tensors
for use in Isaac Gym with 4096 parallel environments.

State vector (6-D): [theta, dTheta, x_err, dx_err, phi, dPhi]
  theta  = avg_virtual_leg_angle + body_pitch   (pendulum from vertical) [rad]
  dTheta = avg_virtual_leg_rate  + body_dpitch  (pendulum angular rate)  [rad/s]
  x_err  = wheel position error  [m]
  dx_err = target_speed - actual_speed  [m/s]
  phi    = -body_pitch   (body lean, positive = backward) [rad]
  dPhi   = -body_dpitch  [rad/s]

Output (2-D): [T_wheel, Tp_leg]  — virtual torques in the planar model
  T_wheel = wheel drive torque  [Nm]
  Tp_leg  = symmetric leg lean torque (applied SAME sign to both legs) [Nm]

LQR: output = K(L0) @ state
K is a 2×6 gain matrix, each element a cubic polynomial in L0.
Polynomials fitted offline via MATLAB for the 5-bar linkage geometry,
but work on serial-chain URDF because the virtual-leg abstraction
(L0, theta0) is mechanism-agnostic.
"""

import torch
from torch import Tensor

# Wheel radius [m] — MUST match the URDF collision cylinder radius
WHEEL_R = 0.0675


def lqr_k_gpu(L0: Tensor) -> Tensor:
    """
    Compute gain-scheduled LQR K matrix for all envs simultaneously.

    Polynomial coefficients from MATLAB offline LQR design (firmware
    lqr_gains.py), stored column-major: K[state*2 + output].

    Args:
        L0: virtual leg length, shape (...,)   [m]

    Returns:
        K: gain matrix, shape (..., 2, 6)
           K[..., 0, :] = wheel torque coefficients (T)
           K[..., 1, :] = leg lean torque coefficients (Tp)
    """
    t2 = L0 * L0
    t3 = t2 * L0

    k_flat = torch.stack(
        [
            # ---- state 0: theta (pendulum tilt) -------------------------
            L0 * -272.72067353448853 + t2 * 634.38627807711850 - t3 * 765.00888920254624 + 8.6463291249670355,   # T
            L0 *   51.31012401349527 - t2 * 285.98949611740600 + t3 * 501.88427005007009 - 0.7958117626705045,   # Tp
            # ---- state 1: dTheta (pendulum tilt rate) -------------------
            L0 *  -34.13381909188140 + t2 *  44.82514907373783 - t3 *  69.06515169135574 + 1.0665188238702900,   # T
            L0 *    7.17572925105903 - t2 *  37.82898320537217 + t3 *  65.23341303606603 - 0.0862682014158316,   # Tp
            # ---- state 2: x_err (wheel position error) ------------------
            L0 * -220.39983489352230 + t2 * 819.38290793249166 - t3 * 1148.7584525718580 + 5.9528862606220168,  # T
            L0 *   24.81390939256314 - t2 * 176.57532274152140 + t3 *  340.9770080162276 + 0.7850268108795293,   # Tp
            # ---- state 3: dx_err (forward speed error) ------------------
            L0 * -163.64604377939659 + t2 * 584.31205773992122 - t3 *  827.1940573259287 + 3.9681623227254441,   # T
            L0 *   19.52853733265620 - t2 * 134.64430273265631 + t3 *  258.0910197564680 + 0.4753474237294207,   # Tp
            # ---- state 4: phi (body lean = -pitch) ----------------------
            L0 * -122.31770375521410 + t2 * 418.01621950621637 - t3 *  552.9703584230645 + 15.985085359180189,   # T
            L0 *   37.27038218185134 - t2 * 165.54082334299800 + t3 *  262.9043418778477 + 18.277954972584649,   # Tp
            # ---- state 5: dPhi (body lean rate = -dpitch) ---------------
            L0 *  -10.14695755036214 + t2 *  30.21637185591932 - t3 *   35.3576323577868 + 1.6363174798254820,   # T
            L0 *    3.65589048229071 - t2 *  15.31570689681319 + t3 *   23.3069872936727 + 1.8453424450483871,   # Tp
        ],
        dim=-1,
    )  # shape (..., 12)

    shape = L0.shape
    K = torch.zeros(*shape, 2, 6, device=L0.device, dtype=L0.dtype)
    for i in range(6):
        K[..., 0, i] = k_flat[..., i * 2]       # T  row
        K[..., 1, i] = k_flat[..., i * 2 + 1]   # Tp row

    return K  # (..., 2, 6)


def compute_lqr_output(
    theta0: Tensor,            # (N, 2)  virtual leg angle [rad]
    theta0_dot: Tensor,        # (N, 2)  virtual leg angular rate [rad/s]
    L0: Tensor,                # (N, 2)  virtual leg length [m]
    L0_dot: Tensor,            # (N, 2)  virtual leg length rate [m/s]
    dof_pos_wheel: Tensor,     # (N, 2)  wheel joint positions [left, right]
    dof_vel_wheel: Tensor,     # (N, 2)  wheel joint velocities [left, right]
    projected_gravity: Tensor, # (N, 3)  gravity vector in body frame
    base_ang_vel: Tensor,      # (N, 3)  angular velocity in body frame [rad/s]
    target_speed: Tensor,      # (N,)    commanded longitudinal speed [m/s]
    wheel_pos_ref: Tensor,     # (N,)    wheel position reference
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Compute LQR state vector and raw baseline output torques.

    Returns:
        lqr_state : (N, 6)   [theta, dTheta, x_err, dx_err, phi, dPhi]
        avg_L0    : (N,)     averaged virtual leg length [m]
        lqr_out   : (N, 2)   [T_wheel, Tp_leg]  raw LQR virtual torques
    """
    # ---- Average left / right -------------------------------------------
    avg_L0     = (L0[:, 0]     + L0[:, 1])    * 0.5
    avg_phi0   = (theta0[:, 0]  + theta0[:, 1]) * 0.5
    avg_dphi0  = (theta0_dot[:, 0] + theta0_dot[:, 1]) * 0.5
    avg_dL0    = (L0_dot[:, 0]  + L0_dot[:, 1]) * 0.5
    avg_wheel_pos = (dof_pos_wheel[:, 0] + dof_pos_wheel[:, 1]) * 0.5
    avg_wheel_vel = (dof_vel_wheel[:, 0] + dof_vel_wheel[:, 1]) * 0.5

    # ---- Body pitch from projected gravity ------------------------------
    # pg = R_body^T * [0, 0, -g]
    # Forward pitch α: pg ≈ [g*sin(α), 0, -g*cos(α)]
    body_pitch  = torch.atan2(projected_gravity[:, 0], -projected_gravity[:, 2])
    body_dpitch = base_ang_vel[:, 1]   # y-axis = pitch axis in body frame

    # ---- LQR state variables --------------------------------------------
    theta  = avg_phi0 + body_pitch     # total pendulum angle from vertical
    dTheta = avg_dphi0 + body_dpitch   # pendulum angular velocity

    # Wheel position error in meters
    x_err = torch.clamp(
        (avg_wheel_pos - wheel_pos_ref) * WHEEL_R, -0.3, 0.3
    )

    # Forward speed: wheel contribution + leg swing contribution
    dx = (
        avg_wheel_vel * WHEEL_R
        + avg_L0 * dTheta * torch.cos(theta)
        + avg_dL0 * torch.sin(theta)
    )
    dx_err = target_speed - dx

    # Body lean (opposite convention to pendulum tilt)
    phi  = -body_pitch
    dPhi = -body_dpitch

    lqr_state = torch.stack([theta, dTheta, x_err, dx_err, phi, dPhi], dim=-1)

    # ---- Gain-scheduled K and LQR output --------------------------------
    K = lqr_k_gpu(avg_L0)                                       # (N, 2, 6)
    lqr_out = torch.bmm(K, lqr_state.unsqueeze(-1)).squeeze(-1)  # (N, 2)

    return lqr_state, avg_L0, lqr_out
