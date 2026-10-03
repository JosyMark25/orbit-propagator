"""已知 bug 回归防线（原 xfail 登记簿）。

V1.1（2026-10-03）三缺陷修复完毕，本文件全部条目转正为正式回归测试：
  P0  save_path 参数生效（原 visualize.py 硬编码覆盖参数）
  P1  mu 参数穿透到速度（原 coe_to_pqw 速度用全局常量混算）
  P1  Kepler 不收敛必须响（原 maxsteps 耗尽静默返回半成品）

机制不变：今后新发现的已知 bug 仍按「登记(xfail strict) -> 修复 ->
xpass 报错提醒 -> 删标记转正」的流程加入本文件。
"""
import numpy as np
import pytest

from constants import mju_earth
from propagate import propagate_from_coe
from visualize import plot_orbit_3d


def test_plot_save_path_parameter_is_respected(tmp_path):
    """plot_orbit_3d(save_path=X) 必须把文件写到 X。"""
    r_arr = np.array([[7000.0, 0.0, 0.0], [0.0, 7000.0, 0.0]])
    target = tmp_path / "orbit.png"
    try:
        plot_orbit_3d(r_arr, save_path=str(target))
    except (FileNotFoundError, OSError):
        pytest.fail("savefig 崩溃：save_path 目录不存在或不可写")
    assert target.exists() and target.stat().st_size > 1000, \
        f"参数路径未生效，文件未写到 {target}"


def test_mu_parameter_penetrates_to_speed():
    """传 mu=2*mu_earth 后 vis-viva 必须按新 mu 自洽（位置/速度同一把尺）。"""
    a, e = 8000.0, 0.1
    mu2 = 2.0 * mju_earth
    t = np.linspace(0.0, 1000.0, 50)
    r_arr, v_arr = propagate_from_coe(
        a, e, 0.3, 0.5, 0.7, 0.0, t, 50, mu=mu2)
    r_norm = np.linalg.norm(r_arr, axis=1)
    v2 = np.sum(v_arr ** 2, axis=1)
    expected = mu2 * (2.0 / r_norm - 1.0 / a)
    assert np.allclose(v2, expected, rtol=1e-9), \
        f"vis-viva 失配：速度用了全局 mu 而非传入 mu2"


def test_kepler_non_convergence_is_loud():
    """强制不收敛（e=0.99 高压 + maxsteps=2）必须抛 RuntimeError，不得静默返回。"""
    from kepler import solve_kepler

    with pytest.raises(RuntimeError, match="未收敛"):
        solve_kepler(M=1.5, e=0.99, maxsteps=2)


def test_kepler_rejects_zero_maxsteps():
    """maxsteps<1 是调用方编程错误——立即 ValueError，而非 NameError。"""
    from kepler import solve_kepler

    with pytest.raises(ValueError, match="maxsteps"):
        solve_kepler(M=1.5, e=0.1, maxsteps=0)
