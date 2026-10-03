import math
import numpy as np
from elements import rotation_pqw_to_eci, coe_to_pqw
from kepler import solve_kepler, tran_E2nju, tran_nju2E

def propagate_from_coe(a, e, i, Omega, omega, nu, t_ARRAY, steps, mu):
    """给定六要素 + 时间序列，返回 (r_array, v_array)"""
    r_ECI_ARRAY = np.zeros((steps, 3))
    v_ECI_ARRAY = np.zeros((steps, 3))
    matrix = rotation_pqw_to_eci(i = i, Omega = Omega, omega = omega)
    E0_initial = tran_nju2E(nu = nu, e = e)
    M0 = E0_initial - e * math.sin(E0_initial)
    n = math.sqrt(mu / a ** 3)

    for k in range(steps):
        t0 = t_ARRAY[k]
        M = M0 + n * t0
        temp_E = solve_kepler(M = M, e = e, maxsteps = 100)
        temp_nju = tran_E2nju(E = temp_E, e = e)
        temp_r_pqw, temp_v_pqw = coe_to_pqw(a = a, e = e, nu = temp_nju, mju = mu)
        temp_r_ECI = matrix @ temp_r_pqw
        temp_v_ECI = matrix @ temp_v_pqw
        r_ECI_ARRAY[k] = temp_r_ECI
        v_ECI_ARRAY[k] = temp_v_ECI

    z_max = np.max(np.abs(r_ECI_ARRAY[:, 2]))
    print("max |z| =", z_max, "km")
    return r_ECI_ARRAY, v_ECI_ARRAY