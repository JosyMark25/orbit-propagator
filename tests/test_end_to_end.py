"""端到端冒烟测试：干净子进程里走完 import 链 + 全链路计算。

main.py 的交互式 input() + visualize 的硬编码路径（已登记
test_known_bugs.py）让它不可自动跑，所以 e2e 绕过 main.py，
直接在子进程里模拟它的完整数据流：
六要素输入 -> 度转弧度 -> 周期计算 -> 传播 -> 输出形状/数值检查。

这层测试防的是"文件间 import 环/路径结构破坏"这类
单元测试覆盖不到的集成失败——V2 重构目录结构时最先红的就是它。
"""
import math
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PIPELINE = """
import math
import numpy as np
from constants import mju_earth
from propagate import propagate_from_coe

# --- 模拟 main.py 的完整数据流（不含 input/plot） ---
a, e = 8000.0, 0.1
i, Omega, omega, nu0 = (math.radians(x) for x in (30.0, 40.0, 60.0, 0.0))
T = 2 * math.pi * math.sqrt(a**3 / mju_earth)
steps = 201
t_array = np.linspace(0.0, T, steps)
r_array, v_array = propagate_from_coe(
    a, e, i, Omega, omega, nu0, t_array, steps, mu=mju_earth)

assert r_array.shape == (steps, 3), f"r shape {r_array.shape}"
assert v_array.shape == (steps, 3), f"v shape {v_array.shape}"
assert np.all(np.isfinite(r_array)), "r 含非有限值"
assert np.all(np.isfinite(v_array)), "v 含非有限值"

# 终点回到起点（整周期闭合）
closure = np.linalg.norm(r_array[-1] - r_array[0])
assert closure < 1e-6, f"closure {closure} km"
print("E2E_OK", T, closure)
"""


def test_subprocess_full_pipeline():
    """干净解释器里：import 链 + 传播 + 形状/有限性/闭合全过。"""
    result = subprocess.run(
        [sys.executable, "-c", PIPELINE],
        cwd=str(PROJECT_ROOT),
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, \
        f"子进程失败: rc={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
    assert "E2E_OK" in result.stdout, f"未到达终点标记: {result.stdout}"


def test_main_module_importable():
    """main.py 本身可导入（不执行 main()，只验证 import 不炸）。"""
    result = subprocess.run(
        [sys.executable, "-c", "import main; print('MAIN_IMPORT_OK')"],
        cwd=str(PROJECT_ROOT),
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0 and "MAIN_IMPORT_OK" in result.stdout, \
        f"main 导入失败: {result.stderr}"
