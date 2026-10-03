"""测试共享配置：路径、fixture、容差常量。

设计原则（V1 → V2 → V3 可持续）：
- 测试只经过公共接口（solve_kepler / coe_to_pqw / propagate_from_coe 的输出），
  不 monkeypatch 内部函数、不断言实现细节——V2 换数值积分、V3 换辛积分器，
  这些测试原样保留，只有 xfail 登记簿里的问题被修复时才需要动标记。
- 守恒律测试（能量/角动量）是黑盒金标准：任何传播算法算出来的轨道都必须满足，
  与实现无关。
- RTOL 常量集中在这里：V2 数值积分进来时，改 RTOL_LOOSE 一处即可
  （解析传播的机器精度阈值不动，数值法另加自己的档位）。
"""
import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 容差档位（集中管理，V2 扩展点）
RTOL_STRICT = 1e-12   # 解析解内部一致性（实测机器精度级：能量漂移 ~1e-15）
RTOL_TIGHT = 1e-9     # 解析解 vs 独立参考实现
RTOL_LOOSE = 1e-7     # 高偏心轨道 / 采样敏感的断言


@pytest.fixture
def anchor_angles():
    """§5.3 锚点角度（弧度）：i=30°, Ω=40°, ω=60°。

    说明书钉死的验收角度，与《V1最终效果说明书》场景一对应。
    """
    return {
        "i": math.radians(30.0),
        "Omega": math.radians(40.0),
        "omega": math.radians(60.0),
    }


@pytest.fixture
def default_orbit():
    """标准测试轨道：a=8000 km, e=0.1, i=30°, Ω=40°, ω=60°, ν₀=0。"""
    return {
        "a": 8000.0,
        "e": 0.1,
        "i": math.radians(30.0),
        "Omega": math.radians(40.0),
        "omega": math.radians(60.0),
        "nu": 0.0,
    }


@pytest.fixture
def high_ecc_orbit():
    """高偏心轨道 e=0.9：Newton 迭代收敛压力测试。"""
    return {
        "a": 8000.0,
        "e": 0.9,
        "i": math.radians(30.0),
        "Omega": math.radians(40.0),
        "omega": math.radians(60.0),
        "nu": 0.0,
    }


@pytest.fixture
def make_period():
    """周期工厂 T = 2π√(a³/μ)。V2 换中心天体（月球）时传 mu 参数复用。"""
    from constants import mju_earth

    def _period(a, mu=mju_earth):
        return 2.0 * math.pi * math.sqrt(a ** 3 / mu)

    return _period
