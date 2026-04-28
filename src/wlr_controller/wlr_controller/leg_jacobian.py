"""
Five-bar linkage Jacobian transpose for VMC force conversion.

Ported from leg_conv.c (MATLAB Coder generated).
Converts virtual leg force (F) and torque (Tp) to physical joint torques (T1, T2).

T1 = torque at front hip (phi1), T2 = torque at rear hip (phi4).
"""

import math

# Link lengths
L1 = 0.07
L2 = 0.147
L3 = 0.147
L4 = 0.07
L5 = 0.123


def leg_conv(phi1: float, phi4: float, F: float, Tp: float) -> tuple[float, float]:
    """
    Convert virtual forces to joint torques via Jacobian transpose.

    This is a direct port of the MATLAB-generated leg_conv.c.

    Args:
        phi1: Front hip angle (rad)
        phi4: Rear hip angle (rad)
        F: Virtual leg axial force (N, positive = extension)
        Tp: Virtual leg torque (Nm)

    Returns:
        (T1, T2): Joint torques for front hip and rear hip
    """
    t2 = math.cos(phi1)
    t3 = math.cos(phi4)
    t4 = math.sin(phi1)
    t5 = math.sin(phi4)

    t8 = t2 * 0.07
    t10 = t4 * 0.07

    a = t10 - t5 * 0.07
    t54 = (t3 * 0.07 - t8) + 0.123
    t43 = t4 * 0.02058 - t5 * 0.02058
    t31 = (t3 * 0.02058 - t2 * 0.02058) + 0.036162
    t33 = a * a + t54 * t54

    disc = (t43 * t43 + t31 * t31) - t33 * t33
    if disc < 0:
        disc = 0.0
    t54_val = math.sqrt(disc)

    t41 = math.atan2(
        (t5 * 0.02058 - t4 * 0.02058) + t54_val,
        (((t5 * 0.0 - t4 * 0.0) + t31) + t33) + t54_val * 0.0
    ) * 2.0

    t43_cos = math.cos(t41)
    t33_sin = math.sin(t41)
    t46 = math.sin(phi1 - t41)

    a_val = t10 + t33_sin * 0.147

    acos_arg = ((t2 * 0.47619047619047616 - t3 * 0.47619047619047616)
                + t43_cos - 0.83673469387755106)
    acos_arg = max(-1.0, min(1.0, acos_arg))
    t3_val = math.acos(acos_arg)

    t54_new = (t8 + t43_cos * 0.147) - 0.0615
    t5_val = math.atan2(
        t4 * 0.07 + t33_sin * 0.147,
        (t4 * 0.0 + t33_sin * 0.0) + t54_new
    )

    t31_sin = math.sin(phi4 - t3_val)

    denom = math.sin(-t41 + t3_val)
    if abs(denom) < 1e-10:
        return 0.0, 0.0
    t2_inv = 1.0 / denom

    r = math.sqrt(a_val * a_val + t54_new * t54_new)
    if r < 1e-10:
        return 0.0, 0.0
    t43_inv = 1.0 / r

    t33_diff = t41 - t5_val
    t54_diff = t3_val - t5_val

    T1 = (F * t46 * t2_inv * math.sin(t54_diff) * (-0.07)
          + Tp * t46 * t2_inv * t43_inv * math.cos(t54_diff) * 0.07)
    T2 = (F * t31_sin * t2_inv * math.sin(t33_diff) * 0.07
          - Tp * t31_sin * t2_inv * t43_inv * math.cos(t33_diff) * 0.07)

    return T1, T2
