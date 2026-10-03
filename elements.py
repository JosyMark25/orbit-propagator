import math
import numpy as np
# from constants import mju_earth

def rotation_pqw_to_eci(i, Omega, omega):
    R = np.array([
        [math.cos(Omega)*math.cos(omega) - math.sin(Omega)*math.cos(i)*math.sin(omega),
         -math.cos(Omega)*math.sin(omega) - math.sin(Omega)*math.cos(i)*math.cos(omega),
         math.sin(Omega)*math.sin(i)],
        [math.sin(Omega)*math.cos(omega) + math.cos(Omega)*math.cos(i)*math.sin(omega),
         -math.sin(Omega)*math.sin(omega) + math.cos(Omega)*math.cos(i)*math.cos(omega),
         -math.cos(Omega)*math.sin(i)],
        [math.sin(i)*math.sin(omega),
         math.sin(i)*math.cos(omega),
         math.cos(i)]
    ], dtype=float)
    return R   # ← 不要 .T

def coe_to_pqw(a, e, nu, mju):
    p = a * (1 - e * e)
    r_mode = a * (1 - e * e) / (1 + e * math.cos(nu))
    rx = r_mode * math.cos(nu)
    ry = r_mode * math.sin(nu)
    loca = np.array([rx, ry, 0.0], dtype=float)
    speed = np.array([-math.sin(nu), e + math.cos(nu), 0.0], dtype=float)
    speed = math.sqrt(mju / p) * speed
    return loca, speed

'''def coe_to_rv(a, e, i, Omega, omega, nu, mu=mju_earth):
    """返回 ECI 下的 r, v"""
    ...

def rv_to_coe(r, v, mu=mju_earth):
    """反向：ECI 的 r, v → 六要素"""
    ...'''