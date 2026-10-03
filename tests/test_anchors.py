"""四锚点验收测试——对应《V1最终效果说明书》场景一的"机器签字"。

锚点数字 = 用户亲手验证过 / 教材对得上的已知答案，**永不改动**。
V2（数值积分）、V3（辛积分器）进来后必须过同样的四关，
届时在各自模块里放宽阈值，这里保持解析解的严格档。

勘误记录（2026-09-30 实测定稿）：
  说明书原文写 E = 0.7761…，经回代手算验证为笔误：
  E - e·sinE = 0.7761 - 0.3×0.7001 = 0.5661 ≠ 0.5（不满足开普勒方程）
  正确值为 E = 0.69125029（回代残差 = 0）。
  测试以回代残差为准（金标准），绝对值作辅助锚。
"""
import math

import numpy as np
import pytest

from constants import mju_earth, R_earth
from elements import rotation_pqw_to_eci
from kepler import solve_kepler
from propagate import propagate_from_coe
from tests.conftest import RTOL_STRICT, RTOL_TIGHT


class TestConstants:
    """常数自检：这些数字错了，后面全错。"""

    def test_mu_earth(self):
        """μ_earth = 398600.4418 km³/s²（WGS-84 值，说明书页脚数字）"""
        assert mju_earth == pytest.approx(398600.4418, abs=1e-6)

    def test_r_earth(self):
        """R_earth = 6371.0 km（平均半径，V1 暂未使用，为后续留）"""
        assert R_earth == pytest.approx(6371.0, abs=1e-6)


class TestAnchor1Matrix:
    """锚点①：§5.3 旋转矩阵 R × (1,0,0) = P̂（近地点方向单位矢量）。

    这是 8/21 numpy 验证脚本逐元素核对过的数字，也是"不要 .T"注释的来历。
    """

    def test_p_hat_hand_value(self, anchor_angles):
        """P̂ = (-0.099, 0.896, 0.433)——手算六位小数可核对档"""
        R = rotation_pqw_to_eci(**anchor_angles)
        p_hat = R @ np.array([1.0, 0.0, 0.0])
        expected = np.array([-0.099068, 0.895927, 0.433013])
        assert np.allclose(p_hat, expected, atol=1e-5), f"P̂={p_hat}"

    def test_p_hat_exact_recompute(self, anchor_angles):
        """P̂ 用三角函数独立重算——说明书 err 3e-16 那一档"""
        i, Om, om = anchor_angles["i"], anchor_angles["Omega"], anchor_angles["omega"]
        expected = np.array([
            math.cos(Om) * math.cos(om) - math.sin(Om) * math.cos(i) * math.sin(om),
            math.sin(Om) * math.cos(om) + math.cos(Om) * math.cos(i) * math.sin(om),
            math.sin(i) * math.sin(om),
        ])
        R = rotation_pqw_to_eci(**anchor_angles)
        p_hat = R @ np.array([1.0, 0.0, 0.0])
        assert np.allclose(p_hat, expected, atol=RTOL_TIGHT), \
            f"p_hat={p_hat}, expected={expected}"


class TestAnchor2Kepler:
    """锚点②：开普勒方程 M=0.5, e=0.3 → E（回代残差一票定案）。"""

    def test_residual_is_zero(self):
        """回代残差 E - e·sinE - M 应达机器精度（实测 0.0）"""
        E = solve_kepler(M=0.5, e=0.3, maxsteps=100)
        residual = E - 0.3 * math.sin(E) - 0.5
        assert abs(residual) < RTOL_STRICT, f"E={E}, residual={residual}"

    def test_absolute_value(self):
        """E = 0.69125029（说明书 0.7761 为笔误，见模块 docstring）"""
        E = solve_kepler(M=0.5, e=0.3, maxsteps=100)
        assert E == pytest.approx(0.69125029, abs=1e-7), f"E={E}"


class TestAnchor3Period:
    """锚点③：周期 T = 2π√(a³/μ)，a=8000 km。"""

    def test_period_formula(self, make_period):
        """公式重算一致性（说明书口径 7120.9 为四舍五入）"""
        T = make_period(8000.0)
        assert T == pytest.approx(2 * math.pi * math.sqrt(8000.0 ** 3 / mju_earth),
                                  rel=RTOL_STRICT)

    def test_period_absolute_value(self, make_period):
        """T = 7121.08 s ≈ 1.98 h"""
        T = make_period(8000.0)
        assert T == pytest.approx(7121.08, abs=0.01), f"T={T}"
        assert T / 3600.0 == pytest.approx(1.98, abs=0.01)


class TestAnchor4Energy:
    """锚点④：能量守恒 |Δε|/ε₀ < 1e-12（说明书验收口径）。

    实测漂移 ~8.6e-16（机器精度级），阈值留千倍余量。
    解析传播理论上严格守恒：同一条轨道上 ε = -μ/2a 与时刻无关。
    """

    def test_energy_conservation(self, default_orbit, make_period):
        """整圈能量恒等于 -μ/2a，相对漂移 < 1e-12"""
        T = make_period(default_orbit["a"])
        t = np.linspace(0.0, T, 500)
        r_arr, v_arr = propagate_from_coe(
            default_orbit["a"], default_orbit["e"], default_orbit["i"],
            default_orbit["Omega"], default_orbit["omega"], default_orbit["nu"],
            t, 500, mu=mju_earth)
        eps0 = -mju_earth / (2.0 * default_orbit["a"])
        eps = (0.5 * np.sum(v_arr ** 2, axis=1)
               - mju_earth / np.linalg.norm(r_arr, axis=1))
        drift = np.max(np.abs(eps - eps0)) / abs(eps0)
        assert drift < 1e-12, f"能量漂移={drift}"
