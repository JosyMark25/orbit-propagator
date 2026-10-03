"""开普勒方程求解器与近点角转换测试。

solve_kepler 是 bug 重灾区之一（角度/弧度混淆、不收敛静默返回、
初值启发式选错分支）。测试策略：
1. 回代残差——开普勒方程自带"标准答案"，残差必须达到容差；
2. 参数化扫描——e 从 0 到 0.99、M 从 0 到 2π-0.01 的网格；
3. 多圈 M——V1 语义：M0 + n·t 会累计超过 2π，Newton 必须照常收敛；
4. 往返恒等——E→ν→E 必须还原（在 atan2 的值域约定内）。

已知设计问题（登记在 test_known_bugs.py，不在这里掩盖）：
- maxsteps 耗尽时静默返回未收敛解，无警告无异常。
"""
import math

import pytest

from kepler import solve_kepler, tran_E2nju, tran_nju2E
from tests.conftest import RTOL_STRICT

# 回代残差容差：solve_kepler 承诺精度（presicion=1e-9），
# 一般组合实测 1e-10~1e-9 档；测试按"达到承诺精度"收
#（锚点组合 M=0.5,e=0.3 实测 0.0 属特例）。
KEPLER_TOL = 1e-8

E_GRID = [0.0, 0.3, 0.5, 1.0, 1.5, 2.0, math.pi, 4.0, 5.0, 2 * math.pi - 0.01]
M_GRID = [0.0, 0.5, 1.0, 1.5, 2.5, math.pi, 4.0, 5.5, 2 * math.pi - 0.01]
ECC_GRID = [0.0, 0.1, 0.3, 0.7, 0.9, 0.99]


def kepler_residual(E, e, M):
    """开普勒方程残差 E - e·sinE - M（方程自带标准答案）。"""
    return E - e * math.sin(E) - M


class TestSolveKepler:

    def test_e_zero_returns_M(self):
        """圆轨道 e=0：E = M，应一步收敛（甚至零步）"""
        for M in (0.3, 1.0, 2.0, 5.0):
            E = solve_kepler(M=M, e=0.0, maxsteps=50)
            assert E == pytest.approx(M, abs=RTOL_STRICT), f"M={M}, E={E}"

    def test_m_zero_returns_zero(self):
        """M=0（近地点）：E=0 是精确解"""
        E = solve_kepler(M=0.0, e=0.7, maxsteps=50)
        assert abs(E) < RTOL_STRICT, f"E={E}"

    def test_m_pi_returns_pi(self):
        """M=π（远地点）：E=π 是精确解"""
        E = solve_kepler(M=math.pi, e=0.7, maxsteps=50)
        assert E == pytest.approx(math.pi, abs=RTOL_STRICT), f"E={E}"

    @pytest.mark.parametrize("e", ECC_GRID)
    @pytest.mark.parametrize("M", M_GRID)
    def test_residual_grid(self, e, M):
        """e×M 网格回代残差（一票定案：方程自己就是标准答案）"""
        E = solve_kepler(M=M, e=e, maxsteps=200)
        assert abs(kepler_residual(E, e, M)) < KEPLER_TOL, \
            f"e={e}, M={M}, E={E}, residual={kepler_residual(E, e, M)}"

    @pytest.mark.parametrize("M_multi", [2 * math.pi + 0.5,
                                         4 * math.pi + 2.0,
                                         10 * math.pi + 3.0])
    def test_multirev_M_converges(self, M_multi):
        """多圈累计平近点角 M > 2π：V1 语义（M0 + n·t 累计）下必须照常收敛。

        初值启发式按 M<π 分支选起点，Newton 对周期函数仍收敛——
        这条测试锁住该语义，防止未来重构时"归一化 M"破坏多圈传播。
        """
        e = 0.3
        E = solve_kepler(M=M_multi, e=e, maxsteps=200)
        assert abs(kepler_residual(E, e, M_multi)) < KEPLER_TOL, \
            f"M={M_multi}, E={E}, residual={kepler_residual(E, e, M_multi)}"

    def test_high_ecc_converges(self):
        """e=0.99 收敛压力：maxsteps 充足（与采样点数无关）时残差达标"""
        e = 0.99
        for M in (0.5, 1.5, 2.5, 3.0):
            E = solve_kepler(M=M, e=e, maxsteps=500)
            assert abs(kepler_residual(E, e, M)) < KEPLER_TOL, \
                f"e=0.99, M={M}, residual={kepler_residual(E, e, M)}"


class TestEToNu:

    def test_zero_maps_to_zero(self):
        """E=0 → ν=0（近地点）"""
        assert abs(tran_E2nju(E=0.0, e=0.3)) < RTOL_STRICT

    def test_pi_maps_to_pi(self):
        """E=π → ν=π（远地点）"""
        assert tran_E2nju(E=math.pi, e=0.3) == pytest.approx(math.pi, abs=RTOL_STRICT)

    def test_e_zero_identity(self):
        """圆轨道 e=0：ν = E"""
        for E in (0.3, 1.0, 2.5):
            assert tran_E2nju(E=E, e=0.0) == pytest.approx(E, abs=RTOL_STRICT)

    @pytest.mark.parametrize("e", ECC_GRID)
    @pytest.mark.parametrize("E", E_GRID)
    def test_roundtrip_with_nu2E(self, e, E):
        """E → ν → E 往返还原（atan2 值域 (-π, π] 约定内自洽）"""
        nu = tran_E2nju(E=E, e=e)
        E_back = tran_nju2E(nu=nu, e=e)
        assert E_back == pytest.approx(E, abs=1e-9), \
            f"e={e}, E={E}, nu={nu}, E_back={E_back}"

    def test_quadrant_3_value_convention(self):
        """E=3*pi/2 -> nu 落在 (pi, 2pi]（atan2 x2 后值域约定）。

        tan(nu/2) = sqrt((1+e)/(1-e)) tan(E/2)：E/2 在第二象限时
        nu/2 也在第二象限，nu 不折叠到负角——文档化该约定，
        防止未来拿 nu 做显示/比较时误判为 bug。
        """
        e = 0.1
        E = 3 * math.pi / 2
        nu = tran_E2nju(E=E, e=e)
        nu_ref = 2 * math.atan2(math.sqrt(1 + e) * math.sin(E / 2),
                                math.sqrt(1 - e) * math.cos(E / 2))
        assert nu == pytest.approx(nu_ref, abs=RTOL_STRICT), f"nu={nu}"
        assert math.pi < nu <= 2 * math.pi, f"nu 应落在 (pi, 2pi]，实际 {nu}"


class TestNuToE:

    def test_zero_maps_to_zero(self):
        """ν=0 → E=0"""
        assert abs(tran_nju2E(nu=0.0, e=0.7)) < RTOL_STRICT

    def test_pi_maps_to_pi(self):
        """ν=π → E=π"""
        assert tran_nju2E(nu=math.pi, e=0.7) == pytest.approx(math.pi, abs=RTOL_STRICT)

    def test_ecc_symmetry(self):
        """对称性：ν 与 -ν 对应 E 与 -E（椭圆关于拱线对称）"""
        e = 0.4
        assert tran_nju2E(nu=1.2, e=e) == pytest.approx(
            -tran_nju2E(nu=-1.2, e=e), abs=1e-12)

    def test_e_zero_identity(self):
        """圆轨道 e=0：E = ν"""
        for nu in (0.3, 1.0, 2.5):
            assert tran_nju2E(nu=nu, e=0.0) == pytest.approx(nu, abs=RTOL_STRICT)


class TestInputDomain:
    """输入域行为：坏输入必须"响"（抛异常），不允许静默 nan。"""

    def test_hyperbolic_ecc_raises(self):
        """e>1（双曲轨道）：下游 sqrt(1-e) 负数必须 ValueError，不能静默 nan。

        注：V1 域为椭圆 0≤e<1。若 V2 在 solve_kepler 层加更早的守卫
        （抛同类型异常），此测试照常通过——测"必须抛"不测"在哪抛"。
        """
        E = solve_kepler(M=0.5, e=1.5, maxsteps=100)
        with pytest.raises(ValueError):
            tran_E2nju(E=E, e=1.5)
