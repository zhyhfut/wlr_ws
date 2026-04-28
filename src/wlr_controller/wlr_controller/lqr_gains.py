"""
LQR gain-scheduling polynomials.

Ported from lqr_k.c (MATLAB Coder generated).
K is a 2x6 matrix: K[0,:] = wheel torque T row, K[1,:] = leg torque Tp row.
State vector: [theta, dTheta, x, dx, phi, dPhi].

Each K[i][j] is a cubic polynomial in L0 (virtual leg length).
"""

import numpy as np


def lqr_k(L0: float) -> np.ndarray:
    """
    Compute gain-scheduled LQR gain matrix K(L0).

    Args:
        L0: Current virtual leg length (m)

    Returns:
        K: 2x6 numpy array. K[0] = T gains, K[1] = Tp gains.
    """
    t2 = L0 * L0
    t3 = L0 * L0 * L0

    K = np.zeros(12)

    # Column-major storage from C code: K[i*2+j] where i=state, j=output
    # K[0] = K_T_theta
    K[0] = (L0 * -272.72067353448853 + t2 * 634.3862780771185
            - t3 * 765.00888920254624 + 8.6463291249670355)
    # K[1] = K_Tp_theta
    K[1] = (L0 * 51.310124013495269 - t2 * 285.989496117406
            + t3 * 501.88427005007009 - 0.79581176267050446)
    # K[2] = K_T_dTheta
    K[2] = (L0 * -34.1338190918814 + t2 * 44.825149073737833
            - t3 * 69.065151691355737 + 1.06651882387029)
    # K[3] = K_Tp_dTheta
    K[3] = (L0 * 7.1757292510590256 - t2 * 37.828983205372168
            + t3 * 65.233413036066025 - 0.086268201415831636)
    # K[4] = K_T_x
    K[4] = (L0 * -220.3998348935223 + t2 * 819.38290793249166
            - t3 * 1148.758452571858 + 5.9528862606220168)
    # K[5] = K_Tp_x
    K[5] = (L0 * 24.81390939256314 - t2 * 176.5753227415214
            + t3 * 340.97700801622761 + 0.7850268108795293)
    # K[6] = K_T_dx
    K[6] = (L0 * -163.64604377939659 + t2 * 584.31205773992122
            - t3 * 827.19405732592872 + 3.9681623227254441)
    # K[7] = K_Tp_dx
    K[7] = (L0 * 19.5285373326562 - t2 * 134.64430273265631
            + t3 * 258.091019756468 + 0.47534742372942068)
    # K[8] = K_T_phi
    K[8] = (L0 * -122.3177037552141 + t2 * 418.01621950621637
            - t3 * 552.97035842306445 + 15.985085359180189)
    # K[9] = K_Tp_phi
    K[9] = (L0 * 37.270382181851339 - t2 * 165.540823342998
            + t3 * 262.90434187784768 + 18.277954972584649)
    # K[10] = K_T_dPhi
    K[10] = (L0 * -10.14695755036214 + t2 * 30.21637185591932
             - t3 * 35.357632357786812 + 1.636317479825482)
    # K[11] = K_Tp_dPhi
    K[11] = (L0 * 3.65589048229071 - t2 * 15.31570689681319
             + t3 * 23.306987293672691 + 1.8453424450483871)

    # Reshape from column-major K[state*2+output] to K_matrix[output][state]
    K_matrix = np.zeros((2, 6))
    for i in range(6):
        K_matrix[0, i] = K[i * 2]      # T row
        K_matrix[1, i] = K[i * 2 + 1]  # Tp row

    return K_matrix
