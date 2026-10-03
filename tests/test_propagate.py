"""传播器测试：propagate_from_coe 的输出作为黑盒做物理断言。

分层设计（V1 -> V2 -> V3 可持续的核心）：
- TestConservation 守恒律（能量/角动量）是黑盒金标准——
  任何传播算法（V2 RK4 / V3 辛积分器）算出的轨道都必须满足，
  测试体只经过 propagate_from_coe 公共接口，V2 换内核不用重写；
- TestOrbitGeometry 几何性质（周期闭合/半径界/z 范围）同样是
  实现无关的物理事实；
- TestSemantics 锁 V1 语义约定（多圈 M 累计、steps 与 t_ARRAY）。

注意：解析传播的守恒精度是机器精度级（实测 ~1e-15），
V2 数值积分进来时另加数值法档位的同型测试，不要放宽本文件阈值。
"""
import math

import numpy as np
import pytest

from constants import mju_earth
from propagate import propagate_from_coe
from tests.conftest import RTOL_STRICT, RTOL_TIGHT, RTOL_LOOSE


def run_orbit(orbit, n_points=400, t_span=None, mu=mju_earth, **kwargs):
    """跑一条轨道一个周期，返回 (t, r_arr, v_arr)。

    V1.1 起 propagate 的 mu 为必填（内核无默认——地球性属于调用方），
    本文件测试均为地球轨道，helper 统一显式传 mu=mju_earth。
    V2 加 method 参数时只需在此封装处传参，各测试体不动。
    """
    from tests.conftest import PROJECT_ROOT  # noqa: F401  (锁路径)

    a = orbit["a"]
    T = 2.0 * math.pi * math.sqrt(a ** 3 / mju_earth)
    t = np.linspace(0.0, T if t_span is None else t_span, n_points)
    r_arr, v_arr = propagate_from_coe(
        a, orbit["e"], orbit["i"], orbit["Omega"], orbit["omega"],
        orbit["nu"], t, n_points, mu=mu, **kwargs)
    return t, r_arr, v_arr


class TestConservation:
    """守恒律：黑盒金标准（V2/V3 数值方法必须通过同型断言）。"""

    def test_energy_constant_default(self, default_orbit):
        """能量恒为 -mu/2a：|max drift| < 1e-12（实测 ~8.6e-16）"""
        _, r_arr, v_arr = run_orbit(default_orbit)
        eps = (0.5 * np.sum(v_arr ** 2, axis=1)
               - mju_earth / np.linalg.norm(r_arr, axis=1))
        eps0 = -mju_earth / (2.0 * default_orbit["a"])
        drift = np.max(np.abs(eps - eps0)) / abs(eps0)
        assert drift < 1e-12, f"energy drift={drift}"

    def test_angular_momentum_constant_default(self, default_orbit):
        """|r x v| 恒等于 sqrt(mu*p)：|max drift| < 1e-12（实测 ~3.9e-16）"""
        _, r_arr, v_arr = run_orbit(default_orbit)
        h = np.cross(r_arr, v_arr)
        h_norm = np.linalg.norm(h, axis=1)
        h0 = math.sqrt(mju_earth * default_orbit["a"] * (1 - default_orbit["e"] ** 2))
        drift = np.max(np.abs(h_norm - h0)) / h0
        assert drift < 1e-12, f"|h| drift={drift}"

    def test_angular_momentum_direction_fixed(self, default_orbit):
        """h 方向不随时间变（轨道面在惯性系中固定——二体问题核心性质）"""
        _, r_arr, v_arr = run_orbit(default_orbit)
        h = np.cross(r_arr, v_arr)
        h0 = h[0] / np.linalg.norm(h[0])
        for k in range(1, len(h)):
            hk = h[k] / np.linalg.norm(h[k])
            assert float(np.dot(hk, h0)) > 1.0 - 1e-9, f"step {k}: h direction changed"

    def test_energy_constant_high_ecc(self, high_ecc_orbit):
        """e=0.9 高偏心轨道能量守恒（收敛压力下仍应机器精度级）"""
        _, r_arr, v_arr = run_orbit(high_ecc_orbit, n_points=600)
        eps = (0.5 * np.sum(v_arr ** 2, axis=1)
               - mju_earth / np.linalg.norm(r_arr, axis=1))
        eps0 = -mju_earth / (2.0 * high_ecc_orbit["a"])
        drift = np.max(np.abs(eps - eps0)) / abs(eps0)
        assert drift < 1e-10, f"e=0.9 energy drift={drift}"

    def test_momentum_constant_high_ecc(self, high_ecc_orbit):
        """e=0.9 角动量守恒"""
        _, r_arr, v_arr = run_orbit(high_ecc_orbit, n_points=600)
        h_norm = np.linalg.norm(np.cross(r_arr, v_arr), axis=1)
        h0 = math.sqrt(mju_earth * high_ecc_orbit["a"] * (1 - 0.81))
        drift = np.max(np.abs(h_norm - h0)) / h0
        assert drift < 1e-10, f"e=0.9 |h| drift={drift}"


class TestOrbitGeometry:

    def test_periodic_closure(self, default_orbit):
        """整周期闭合：r(T) = r(0)（误差 km 级断言，实测 8e-12 km）"""
        t, r_arr, v_arr = run_orbit(default_orbit)
        assert np.linalg.norm(r_arr[-1] - r_arr[0]) < 1e-6, \
            f"closure err={np.linalg.norm(r_arr[-1] - r_arr[0])} km"
        assert np.linalg.norm(v_arr[-1] - v_arr[0]) < 1e-6, \
            f"velocity closure err={np.linalg.norm(v_arr[-1] - v_arr[0])} km/s"

    def test_radius_bounds(self, default_orbit):
        """半径界：a(1-e) <= |r| <= a(1+e)，全部采样点不越界"""
        _, r_arr, _ = run_orbit(default_orbit, n_points=500)
        r_norm = np.linalg.norm(r_arr, axis=1)
        assert r_norm.min() >= 8000.0 * 0.9 - 1e-6
        assert r_norm.max() <= 8000.0 * 1.1 + 1e-6

    def test_z_max_matches_reference_scan(self, default_orbit):
        """z 峰值与独立参考实现（细网格解析扫描）一致。

        注意：z_max = a(1+e)sin(i) 只是上界近似（极值点不在远地点，
        实测 4341 vs 上界 4400）；这里用独立数值参考做精确断言，
        并验证上界成立。
        """
        orbit = default_orbit
        _, r_arr, _ = run_orbit(orbit, n_points=2000)
        z_max_code = np.max(np.abs(r_arr[:, 2]))

        # 独立参考实现：几何推导闭式 z(nu) = sin(i) * p/(1+e cos nu) * sin(nu+w)
        # （不经过被测代码的任何函数，纯 numpy 向量化）
        a, e, i, w = (orbit["a"], orbit["e"], orbit["i"], orbit["omega"])
        nu = np.linspace(0.0, 2.0 * math.pi, 2_000_001)
        z_ref = np.max(np.abs(np.sin(i) * a * (1 - e ** 2)
                              / (1.0 + e * np.cos(nu)) * np.sin(nu + w)))
        assert z_max_code == pytest.approx(z_ref, rel=1e-3), \
            f"z_max_code={z_max_code}, z_ref={z_ref}"
        assert z_max_code <= a * (1 + e) * math.sin(i) + 1e-9, "z 峰值超过几何上界"

    def test_t0_at_perigee(self, default_orbit):
        """nu0=0 时 t=0 位置=近地点：|r(0)|=a(1-e)，速度全轨道最大"""
        _, r_arr, v_arr = run_orbit(default_orbit)
        assert np.linalg.norm(r_arr[0]) == pytest.approx(8000.0 * 0.9, rel=RTOL_TIGHT)
        v_norms = np.linalg.norm(v_arr, axis=1)
        assert v_norms[0] == pytest.approx(v_norms.max(), rel=1e-6)


class TestSemantics:
    """V1 语义约定：重构时这些行为变化必须显式意识到。"""

    def test_multirev_energy_conserved(self, default_orbit):
        """多圈传播（t 延伸 2.5 周期）：能量仍守恒（M 累计大数语义）"""
        T = 2.0 * math.pi * math.sqrt(default_orbit["a"] ** 3 / mju_earth)
        t = np.linspace(0.0, 2.5 * T, 600)
        r_arr, v_arr = propagate_from_coe(
            default_orbit["a"], default_orbit["e"], default_orbit["i"],
            default_orbit["Omega"], default_orbit["omega"], default_orbit["nu"],
            t, 600, mu=mju_earth)
        eps = (0.5 * np.sum(v_arr ** 2, axis=1)
               - mju_earth / np.linalg.norm(r_arr, axis=1))
        eps0 = -mju_earth / (2.0 * default_orbit["a"])
        drift = np.max(np.abs(eps - eps0)) / abs(eps0)
        assert drift < 1e-12, f"multirev energy drift={drift}"

    def test_circular_orbit_uniform_rate(self):
        """圆轨道 e=0：惯性系内角位置随时间匀速（角速度 n）"""
        a = 8000.0
        T = 2.0 * math.pi * math.sqrt(a ** 3 / mju_earth)
        n = 2.0 * math.pi / T
        t = np.linspace(0.0, T, 200)
        r_arr, v_arr = propagate_from_coe(
            a, 0.0, 0.3, 0.5, 0.7, 0.0, t, 200, mu=mju_earth)
        # 轨道面内测角速度（惯性系 x-y 投影是椭圆，投影角不匀速属正常几何）
        from elements import rotation_pqw_to_eci
        R = rotation_pqw_to_eci(0.3, 0.5, 0.7)
        r_pqw = (R.T @ r_arr.T).T          # 正交阵逆 = 转置
        theta = np.unwrap(np.arctan2(r_pqw[:, 1], r_pqw[:, 0]))
        slope = np.polyfit(t, theta, 1)[0]
        assert slope == pytest.approx(n, rel=1e-9), f"dtheta/dt={slope}, n={n}"
