"""轨道要素几何层测试：旋转矩阵与平面内位置速度。

旋转矩阵是 bug 重灾区之二（转置方向、Omega/omega 搞混、行列符号错）。
用户 8/21 逐元素验证过矩阵（"不要 .T"注释即踩坑记录），锚点测试防回归；
正交性/det/列向量语义是数学性质断言——实现无关，V2/V3 重构此模块时
这些测试必须原样通过。
"""
import math

import numpy as np
import pytest

from constants import mju_earth
from elements import coe_to_pqw, rotation_pqw_to_eci
from tests.conftest import RTOL_STRICT, RTOL_TIGHT

ANGLE_SETS = [
    (0.0, 0.0, 0.0),
    (math.radians(30.0), math.radians(40.0), math.radians(60.0)),
    (math.radians(98.7), math.radians(12.3), math.radians(245.0)),
    (math.radians(150.0), math.radians(200.0), math.radians(310.0)),
]


class TestRotationMatrix:

    def test_identity_when_all_zero(self):
        """i=Omega=omega=0 时 R = I（轨道面与参考面重合）"""
        R = rotation_pqw_to_eci(i=0.0, Omega=0.0, omega=0.0)
        assert np.allclose(R, np.eye(3), atol=RTOL_STRICT), f"R={R}"

    @pytest.mark.parametrize("i,Omega,omega", ANGLE_SETS)
    def test_orthogonal(self, i, Omega, omega):
        """R^T R = I（正交性：三轴旋转不缩放不剪切）"""
        R = rotation_pqw_to_eci(i=i, Omega=Omega, omega=omega)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)

    @pytest.mark.parametrize("i,Omega,omega", ANGLE_SETS)
    def test_det_plus_one(self, i, Omega, omega):
        """det(R) = +1（纯旋转，非反射——转置错误常导致 det=-1）"""
        R = rotation_pqw_to_eci(i=i, Omega=Omega, omega=omega)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)

    @pytest.mark.parametrize("i,Omega,omega", ANGLE_SETS)
    def test_columns_unit_norm(self, i, Omega, omega):
        """三列都是单位矢量（P/Q/W 方向基）"""
        R = rotation_pqw_to_eci(i=i, Omega=Omega, omega=omega)
        for col in range(3):
            assert np.linalg.norm(R[:, col]) == pytest.approx(1.0, abs=1e-12)


    def test_anchor_first_column_p_direction(self, anchor_angles):
        """锚点：R x (1,0,0) = P_hat = (-0.099068, 0.895927, 0.433013)。

        与说明书 §5.3 一致（保留 6 位小数）；矩阵第一列 = 近地点方向
        单位矢量在惯性系的表示——8/21 用户逐元素验证过的数字。
        """
        R = rotation_pqw_to_eci(**anchor_angles)
        p_hat = R @ np.array([1.0, 0.0, 0.0])
        expected = np.array([-0.099068, 0.895927, 0.433013])
        assert np.allclose(p_hat, expected, atol=1e-5), f"P_hat={p_hat}"

    def test_first_column_matches_closed_form(self, anchor_angles):
        """第一列与闭式公式逐元素吻合（机器精度）：
        P = (cosO cosw - sinO cosi sinw, sinO cosw + cosO cosi sinw, sini sinw)
        """
        R = rotation_pqw_to_eci(**anchor_angles)
        i, O, w = anchor_angles["i"], anchor_angles["Omega"], anchor_angles["omega"]
        expected = np.array([
            math.cos(O) * math.cos(w) - math.sin(O) * math.cos(i) * math.sin(w),
            math.sin(O) * math.cos(w) + math.cos(O) * math.cos(i) * math.sin(w),
            math.sin(i) * math.sin(w)])
        assert np.allclose(R @ np.array([1.0, 0.0, 0.0]), expected, atol=1e-15)

    def test_omega_omega_swap_changes_matrix(self):
        """Omega 与 omega 互换矩阵必须不同（防 Omega/omega 混淆回归）。

        V1 的经典 bug：两个角在旋转矩阵里位置不对称，写反时
        圆/赤道特例下看不出来，一般轨道全错——此测试用不对称角度对捕获。
        """
        i, a, b = 0.5, 1.0, 0.3
        R1 = rotation_pqw_to_eci(i=i, Omega=a, omega=b)
        R2 = rotation_pqw_to_eci(i=i, Omega=b, omega=a)
        assert not np.allclose(R1, R2), "Omega/omega 互换矩阵相同 -> 两者被搞混"

    def test_third_column_is_orbit_normal(self, anchor_angles):
        """第三列 = 轨道面法向（角动量方向）闭式公式。"""
        R = rotation_pqw_to_eci(**anchor_angles)
        i, O = anchor_angles["i"], anchor_angles["Omega"]
        expected = np.array([
            math.sin(O) * math.sin(i),
            -math.cos(O) * math.sin(i),
            math.cos(i)])
        assert np.allclose(R[:, 2], expected, atol=1e-15)

    def test_equatorial_matrix_structure(self):
        """i=0 赤道轨道：第三列 = (0,0,1)（轨道面即参考面）"""
        R = rotation_pqw_to_eci(i=0.0, Omega=1.2, omega=0.7)
        assert np.allclose(R[:, 2], [0.0, 0.0, 1.0], atol=1e-15)


class TestCoeToPqw:
    """平面内位置速度：r = p/(1+e cos nu) 与 v = sqrt(mu/p)(-sin nu, e+cos nu)。

    这些公式是 Vallado 教材标准形式；测试用近地点/远地点半径、
    vis-viva 活力公式、角动量三组独立物理量交叉验证。
    """

    NU_GRID = (0.0, 0.5, 1.2, 2.0, math.pi, 4.0, 5.5)

    def test_perigee_radius(self, default_orbit):
        """近地点 nu=0：|r| = a(1-e)"""
        r, _ = coe_to_pqw(a=default_orbit["a"], e=default_orbit["e"], mju=mju_earth, nu=0.0)
        assert np.linalg.norm(r) == pytest.approx(8000.0 * 0.9, rel=RTOL_TIGHT)

    def test_apogee_radius(self, default_orbit):
        """远地点 nu=pi：|r| = a(1+e)"""
        r, _ = coe_to_pqw(a=default_orbit["a"], e=default_orbit["e"], mju=mju_earth, nu=math.pi)
        assert np.linalg.norm(r) == pytest.approx(8000.0 * 1.1, rel=RTOL_TIGHT)

    def test_perigee_speed(self, default_orbit):
        """近地点速度大小 = sqrt(mu(1+e)/(a(1-e)))（轨道速度极值点）"""
        _, v = coe_to_pqw(a=default_orbit["a"], e=default_orbit["e"], mju=mju_earth, nu=0.0)
        expected = math.sqrt(mju_earth * (1 + 0.1) / (8000.0 * (1 - 0.1)))
        assert np.linalg.norm(v) == pytest.approx(expected, rel=RTOL_TIGHT)

    def test_radial_velocity_zero_at_apses(self, default_orbit):
        """拱点（近地点/远地点）径向速度为零：r 点乘 v = 0"""
        for nu in (0.0, math.pi):
            r, v = coe_to_pqw(a=default_orbit["a"], e=default_orbit["e"], mju=mju_earth, nu=nu)
            assert abs(float(np.dot(r, v))) < 1e-9, f"nu={nu}, r.v={np.dot(r, v)}"

    def test_z_zero_in_pqw(self, default_orbit):
        """PQW 平面内：位置和速度的 z 分量为零"""
        for nu in self.NU_GRID:
            r, v = coe_to_pqw(a=default_orbit["a"], e=default_orbit["e"], mju=mju_earth, nu=nu)
            assert abs(r[2]) < RTOL_TIGHT and abs(v[2]) < RTOL_TIGHT

    @pytest.mark.parametrize("nu", NU_GRID)
    def test_vis_viva_all_nu(self, default_orbit, nu):
        """vis-viva 活力公式全轨道成立：|v|^2 = mu(2/r - 1/a)。

        一票否决速度公式：位置速度不自洽立即暴露。
        """
        r, v = coe_to_pqw(a=default_orbit["a"], e=default_orbit["e"], mju=mju_earth, nu=nu)
        r_norm = np.linalg.norm(r)
        v2 = float(np.dot(v, v))
        expected = mju_earth * (2.0 / r_norm - 1.0 / default_orbit["a"])
        assert v2 == pytest.approx(expected, rel=RTOL_TIGHT), \
            f"nu={nu}, |v|^2={v2}, expected={expected}"

    @pytest.mark.parametrize("nu", NU_GRID)
    def test_specific_energy_constant(self, default_orbit, nu):
        """比机械能恒为 -mu/2a，与 nu 无关（椭圆轨道能量守恒）"""
        r, v = coe_to_pqw(a=default_orbit["a"], e=default_orbit["e"], mju=mju_earth, nu=nu)
        eps = float(np.dot(v, v)) / 2.0 - mju_earth / np.linalg.norm(r)
        assert eps == pytest.approx(-mju_earth / (2 * 8000.0), rel=RTOL_TIGHT)
