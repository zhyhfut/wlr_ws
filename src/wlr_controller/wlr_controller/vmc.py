r"""
Five-bar linkage Virtual Model Control (VMC) - Forward Kinematics

Ported from vmc.c. Computes virtual leg position (L0, phi0) and velocity
(dL0, dAngle) from the four hip joint angles/velocities.

Five-bar topology:

          C (foot/wheel)
         / \
   l2   /   \  l3
       /     \
      B       D
      |       |
  l1  |  l5   |  l4
      |-------|
      A       E
   (front)  (rear)

A is at origin (0,0), E is at (l5, 0).
phi1 = angle of link AB from A (front hip motor).
phi4 = angle of link ED from E (rear hip motor).
L0 = distance from midpoint of AE to C.
phi0 = angle of virtual leg from midpoint of AE to C.
"""

import math

# Link lengths (meters) from vmc.c
L1 = 0.07
L2 = 0.147
L3 = 0.147
L4 = 0.07
L5 = 0.123


def leg_pos(phi1: float, phi4: float) -> tuple[float, float]:
    """
    Forward position kinematics of the five-bar linkage.

    Args:
        phi1: Front hip joint angle (rad)
        phi4: Rear hip joint angle (rad)

    Returns:
        (L0, phi0): Virtual leg length and angle
    """
    # Joint B position
    xb = L1 * math.cos(phi1)
    yb = L1 * math.sin(phi1)

    # Joint D position
    xd = L4 * math.cos(phi4) + L5
    yd = L4 * math.sin(phi4)

    # Solve for phi2 using half-angle tangent method
    A0 = 2.0 * L2 * (xd - xb)
    B0 = 2.0 * L2 * (yd - yb)
    C0 = L2 * L2 + (xd - xb) ** 2 + (yd - yb) ** 2 - L3 * L3

    disc = A0 * A0 + B0 * B0 - C0 * C0
    if disc < 0:
        disc = 0.0
    phi2 = 2.0 * math.atan2(B0 + math.sqrt(disc), A0 + C0)

    # Joint C position
    xc = xb + L2 * math.cos(phi2)
    yc = yb + L2 * math.sin(phi2)

    # Virtual leg from midpoint of AE
    mid_x = L5 / 2.0
    L0 = math.sqrt((xc - mid_x) ** 2 + yc * yc)
    phi0 = math.atan2(yc, xc - mid_x)

    return L0, phi0


def leg_spd(phi1: float, phi4: float,
            dphi1: float, dphi4: float) -> tuple[float, float]:
    """
    Forward velocity kinematics of the five-bar linkage.

    Args:
        phi1, phi4: Hip joint angles (rad)
        dphi1, dphi4: Hip joint angular velocities (rad/s)

    Returns:
        (dL0, dAngle): Virtual leg length rate and angle rate
    """
    # Recompute positions
    xb = L1 * math.cos(phi1)
    yb = L1 * math.sin(phi1)
    xd = L4 * math.cos(phi4) + L5
    yd = L4 * math.sin(phi4)

    A0 = 2.0 * L2 * (xd - xb)
    B0 = 2.0 * L2 * (yd - yb)
    C0 = L2 * L2 + (xd - xb) ** 2 + (yd - yb) ** 2 - L3 * L3

    disc = A0 * A0 + B0 * B0 - C0 * C0
    if disc < 0:
        disc = 0.0
    phi2 = 2.0 * math.atan2(B0 + math.sqrt(disc), A0 + C0)

    # phi3
    denom = (xb + L2 * math.cos(phi2) - xd) / L3
    denom = max(-1.0, min(1.0, denom))
    phi3 = math.acos(denom)

    xc = xb + L2 * math.cos(phi2)
    yc = yb + L2 * math.sin(phi2)

    # Jacobian-based velocity
    s_phi2_phi3 = math.sin(phi2 - phi3)
    if abs(s_phi2_phi3) < 1e-10:
        return 0.0, 0.0

    x_dot_c = (L1 * math.sin(phi1 - phi2) * math.sin(phi3) / s_phi2_phi3 * dphi1 +
               L4 * math.sin(phi3 - phi4) * math.sin(phi2) / s_phi2_phi3 * dphi4)
    y_dot_c = (-L1 * math.sin(phi1 - phi2) * math.cos(phi3) / s_phi2_phi3 * dphi1 -
               L4 * math.sin(phi3 - phi4) * math.cos(phi2) / s_phi2_phi3 * dphi4)

    r2 = xc * xc + yc * yc
    r = math.sqrt(r2)
    if r < 1e-10:
        return 0.0, 0.0

    dL0 = (xc * x_dot_c + yc * y_dot_c) / r
    dAngle = (xc * y_dot_c - yc * x_dot_c) / r2

    return dL0, dAngle
