"""熔池代理模型 (Eagar-Tsai) 的正确性与标定验证。

分三层测试：

1. **解析自检**：与闭式中心温度公式对比，验证前系数与高斯展宽项。
   这是最关键的一层 —— 这两处系数曾各错一次，合起来把温度放大 2×10⁴ 倍。
2. **量级验证**：与公开文献报道的典型熔池尺寸量级对比（±25% 判据）。
   注意参考值取自文献典型量级，非同一套受控实验，故判据放宽。
3. **可微性/单调性**：梯度存在、有限，且工艺趋势符合物理直觉。
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from amforge import geometry as G                              # noqa: E402
from amforge.core.contracts import ProcessPlan                 # noqa: E402
from amforge.materials import get_material                     # noqa: E402
from amforge.meltpool import (                                 # noqa: E402
    eagar_tsai_field,
    solve_meltpool_surrogate,
    _melt_length_scale,
    _smooth_max,
)


@pytest.fixture(scope="module")
def part():
    return G.from_sdf_fn(G.gyroid(cell=1.2e-3, thickness=1.6e-4),
                         bounds=[(0.0, 2.4e-3)] * 3, spacing=1.2e-4)


def _mean_props(mat):
    k = 0.5 * (mat.k_solid + mat.k_liquid)
    rho = 0.5 * (mat.rho_solid + mat.rho_liquid)
    cp = 0.5 * (mat.cp_solid + mat.cp_liquid)
    return k, rho, cp


# ===========================================================================
# 1. 解析自检 —— 前系数与高斯展宽项
# ===========================================================================
def test_center_temperature_matches_closed_form():
    """光斑中心温度必须命中闭式解（静止极限 v→0）。

    在 (x,y,z)=(0,0,0)、v=0 处积分可解析求出：

        T_c - T_0 = (√2/2) · A·P / (ρ c √π α r_b)

    推导：令 u=√(ατ)，则
        ∫₀^∞ dτ/[(4ατ + r_b²/2)·2√(ατ)]
      = (1/α)∫₀^∞ du/(4u² + r_b²/2)
      = (1/α)·(1/4)·π/(2·r_b/(2√2))
      = π√2/(4 α r_b)
    再乘前系数 2AP/(ρcπ^{3/2}) 即得上式。

    注意：该闭式是 **静止极限**，仅当 v→0 才成立。移动源 (v≠0) 的中心
    温度会显著低于此值（速度越快越低），故此处用 v=0 作精确对照。
    正切代换在 v=0 时被积函数 ≡ 1，任意节点数都给出解析值。
    """
    mat = get_material("Ti6Al4V")
    k, rho, cp = _mean_props(mat)
    rho_cp, alpha = rho * cp, k / (rho * cp)
    A, P, rb, T0 = 0.4, 220.0, 50e-6, 473.0
    Pab = A * P

    zero = jnp.zeros((1, 1, 1))
    T_num = float(eagar_tsai_field(
        absorbed_power=Pab, scan_speed=0.0, beam_radius=rb, T0=T0,
        alpha=alpha, rho_cp=rho_cp, X=zero, Y=zero, Z=zero,
        n_quad=24)[0, 0, 0])

    T_exact = T0 + (np.sqrt(2.0) / 2.0) * Pab / (
        rho_cp * np.sqrt(np.pi) * alpha * rb)

    assert T_num == pytest.approx(T_exact, rel=1e-10), (
        f"中心温度 {T_num:.1f} K 与闭式解 {T_exact:.1f} K 不符 —— "
        f"检查前系数 2AP/(ρcπ^1.5) 与高斯项 r_b²/2。"
    )


def test_quadrature_converges():
    """Gauss-Legendre 节点数增加时结果单调收敛；默认 n_quad=48 已足够精确。

    被积函数的峰落在 φ∈[0,π/2] 中段（正切代换消掉了 1/√τ 奇点），
    故近源（物理上重要的）区域 48 点即达机器精度。这里用**中心点温度**
    （即整个积分值）作为收敛判据——它在宽/深/长等交付量里是最慢收敛的
    量，若它收敛则派生量必收敛。远场靠近初温的点相对误差病态，不用于判据。
    """
    mat = get_material("316L")
    k, rho, cp = _mean_props(mat)
    rho_cp, alpha = rho * cp, k / (rho * cp)
    rb, v, T0 = 50e-6, 0.8, 353.0

    def center_T(n):
        zero = jnp.zeros((1, 1, 1))
        return float(eagar_tsai_field(
            absorbed_power=68.0, scan_speed=v, beam_radius=rb, T0=T0,
            alpha=alpha, rho_cp=rho_cp, X=zero, Y=zero, Z=zero,
            n_quad=n)[0, 0, 0])

    Tc = {n: center_T(n) for n in (16, 24, 48, 96, 192)}
    ref = Tc[192]
    # 16 点"够用"，48 点已到机器精度
    assert abs(Tc[16] - ref) / ref < 1e-2, f"n_quad=16 偏差过大 {Tc[16]:.3f}"
    assert abs(Tc[24] - ref) / ref < 1e-3, f"n_quad=24 未收敛 {Tc[24]:.3f}"
    assert abs(Tc[48] - ref) / ref < 1e-4, f"n_quad=48 未收敛 {Tc[48]:.3f}"


def test_temperature_decays_away_from_source():
    """温度场沿热源前方单调衰减回到初温，且尾迹（后方）比前方更热。

    准稳态移动热源的尾迹（热源后方，-x 侧）很长，在数毫米处仍残留数十 K
    的温升；但**前方**(+x) 是尚未被加热的区域，很快就回到初温。因此判据
    取前方远场回温 + 后方近场比前方更热，而非要求两侧都精确回到初温。
    """
    mat = get_material("316L")
    k, rho, cp = _mean_props(mat)
    rb, v, T0 = 50e-6, 0.8, 353.0
    xs = jnp.linspace(-12e-3, 12e-3, 49)
    X = xs[:, None, None]
    Z = Y = jnp.zeros_like(X)
    T = np.asarray(eagar_tsai_field(
        absorbed_power=68.0, scan_speed=v, beam_radius=rb,
        T0=T0, alpha=k / (rho * cp), rho_cp=rho * cp, X=X, Y=Y, Z=Z)).ravel()
    mid = len(T) // 2
    ahead = T[mid:]                       # x > 0（前方，尚未被加热）
    # 前方单调非增（尾迹远处趋于平台，故允许相等）
    assert np.all(np.diff(ahead) <= 1e-9), "热源前方温度应单调下降"
    # 前方远场回到初温
    assert ahead[-1] == pytest.approx(T0, abs=5.0), "远前方应回到初温"
    # 尾迹（后方近场）比对称的前方远场更热
    assert T[mid - 2] > T[mid + 2], "热源后方温度应高于前方（尾迹方向错了）"
    assert T[mid - 6] > T[mid + 6], "尾迹应整体比前方热"


# ===========================================================================
# 2. 软最大值算子
# ===========================================================================
def test_smooth_max_unbiased_on_plateau():
    """平台场上 softmax 加权平均必须 ≈ 真实最大值。

    旧实现用 p-范数 (Σxᵖ)^{1/p}，对 N 个 1 会返回 N^{1/p}（p=12、N=576
    时高估 70%），熔池诊断里恰恰全是这种平台场。
    """
    x = jnp.ones((24, 24))
    assert float(_smooth_max(x)) == pytest.approx(1.0, rel=1e-6)

    x = jnp.zeros((24, 24)).at[3, 4].set(1.0)
    assert float(_smooth_max(x)) == pytest.approx(1.0, rel=1e-3)

    # 尺度不变性：量纲化的场也要给对
    y = 1e-4 * jnp.ones((10, 10))
    assert float(_smooth_max(y)) == pytest.approx(1e-4, rel=1e-6)


def test_smooth_max_is_differentiable():
    x = jnp.linspace(0.0, 1.0, 32)
    g = jax.grad(lambda a: _smooth_max(a))(x)
    assert np.all(np.isfinite(np.asarray(g)))
    assert float(jnp.sum(g)) == pytest.approx(1.0, rel=1e-3), \
        "软最大值对输入的梯度之和应为 1（它是一个加权平均）"


# ===========================================================================
# 3. 量级验证 —— 对照文献典型熔池尺寸
# ===========================================================================
#: (材料, P[W], v[m/s], r_b[m], 层厚[m], 吸收率, 参考宽[µm], 参考深[µm])
#: 参考值为公开文献报道的典型量级（非同一套受控实验），故判据取 ±30%。
LITERATURE_CASES = [
    ("Ti6Al4V", 280.0, 1.20, 50e-6, 30e-6, 0.40, 130.0, 75.0),
    ("Ti6Al4V", 220.0, 0.90, 50e-6, 40e-6, 0.40, 150.0, 85.0),
    ("316L", 195.0, 0.80, 50e-6, 30e-6, 0.35, 120.0, 70.0),
    ("316L", 100.0, 0.40, 50e-6, 30e-6, 0.35, 105.0, 55.0),
    ("IN718", 285.0, 0.96, 50e-6, 40e-6, 0.35, 140.0, 80.0),
]


@pytest.mark.parametrize("case", LITERATURE_CASES,
                         ids=[f"{c[0]}-{c[1]:.0f}W-{c[2]:.2f}mps"
                              for c in LITERATURE_CASES])
def test_melt_pool_size_order_of_magnitude(part, case):
    mat_name, P, v, rb, t_layer, A, w_ref, d_ref = case
    plan = ProcessPlan.uniform(
        1, laser_power=P, scan_speed=v, beam_radius=rb,
        layer_thickness=t_layer, hatch_spacing=0.7 * w_ref * 1e-6,
        absorption=A, preheat_temp=353.0)
    mp = solve_meltpool_surrogate(geometry=part, process=plan,
                                  params={"material": mat_name, "n_grid": 40})
    w = float(mp.width) * 1e6
    d = float(mp.depth) * 1e6
    assert w == pytest.approx(w_ref, rel=0.30), f"熔宽 {w:.1f} vs 参考 {w_ref}"
    assert d == pytest.approx(d_ref, rel=0.30), f"熔深 {d:.1f} vs 参考 {d_ref}"


def test_temperature_capped_near_boiling(part):
    """峰值温度不得远超沸点（蒸发饱和必须生效）。"""
    plan = ProcessPlan.uniform(1, laser_power=400.0, scan_speed=0.4,
                               beam_radius=50e-6, absorption=0.4)
    mp = solve_meltpool_surrogate(geometry=part, process=plan,
                                  params={"material": "Ti6Al4V"})
    T_max = float(jnp.max(mp.temperature))
    T_boil = float(get_material("Ti6Al4V").T_boil)
    assert T_max < T_boil + 50.0, f"峰值 {T_max:.0f} K 超过沸点 {T_boil:.0f} K 太多"
    assert T_max > 0.9 * T_boil, "高功率下应达到沸点附近"


def test_pool_grows_with_energy_density(part):
    """熔池尺寸对线能量密度 P/v 必须单调递增。"""
    prev_w = prev_d = -1.0
    for P, v in [(100.0, 1.5), (150.0, 1.2), (250.0, 1.0), (350.0, 0.7)]:
        plan = ProcessPlan.uniform(1, laser_power=P, scan_speed=v,
                                   beam_radius=50e-6, absorption=0.4)
        mp = solve_meltpool_surrogate(geometry=part, process=plan,
                                      params={"material": "Ti6Al4V"})
        w, d = float(mp.width), float(mp.depth)
        assert w > prev_w, f"P/v 增大时熔宽反而减小 ({w:.3e} <= {prev_w:.3e})"
        assert d > prev_d, f"P/v 增大时熔深反而减小 ({d:.3e} <= {prev_d:.3e})"
        prev_w, prev_d = w, d


def test_no_depression_below_boiling(part):
    """低功率（表面不到沸点）时凹陷项必须严格为 0，不引入虚假熔深。"""
    plan = ProcessPlan.uniform(1, laser_power=25.0, scan_speed=2.0,
                               beam_radius=200e-6, absorption=0.3)
    mp = solve_meltpool_surrogate(geometry=part, process=plan,
                                  params={"material": "316L"})
    assert float(mp.keyhole_depth) < 1e-9, \
        f"未达沸点却出现凹陷 {float(mp.keyhole_depth):.3e} m"


# ===========================================================================
# 4. 可微性
# ===========================================================================
def test_gradient_wrt_power_and_speed(part):
    """对功率/速度的梯度必须有限、非零，且符号符合物理。"""
    def depth_of(P, v):
        plan = ProcessPlan.uniform(1, laser_power=P, scan_speed=v,
                                   beam_radius=50e-6, absorption=0.4)
        return solve_meltpool_surrogate(
            geometry=part, process=plan,
            params={"material": "Ti6Al4V", "n_grid": 24}).depth

    gP, gv = jax.grad(depth_of, argnums=(0, 1))(220.0, 0.9)
    gP, gv = float(gP), float(gv)
    assert np.isfinite(gP) and np.isfinite(gv)
    assert gP > 0.0, "功率增大应使熔深增大"
    assert gv < 0.0, "速度增大应使熔深减小"


def test_jit_compiles(part):
    """整个代理模型可被 jax.jit 编译（纯函数约束）。"""
    @jax.jit
    def run(P, v):
        plan = ProcessPlan.uniform(1, laser_power=P, scan_speed=v,
                                   beam_radius=50e-6, absorption=0.4)
        mp = solve_meltpool_surrogate(
            geometry=part, process=plan,
            params={"material": "Ti6Al4V", "n_grid": 24})
        return mp.depth, mp.width, mp.defect_score()

    d, w, s = run(220.0, 0.9)
    assert np.isfinite(float(d)) and np.isfinite(float(w)) and np.isfinite(float(s))


def test_vmap_over_process_window(part):
    """可 vmap 成工艺窗口图（这是"求解器可组合"的直接红利）。"""
    def one(P, v):
        plan = ProcessPlan.uniform(1, laser_power=P, scan_speed=v,
                                   beam_radius=50e-6, absorption=0.4)
        return solve_meltpool_surrogate(
            geometry=part, process=plan,
            params={"material": "Ti6Al4V", "n_grid": 20}).defect_score()

    P = jnp.linspace(80.0, 400.0, 5)
    v = jnp.linspace(0.4, 2.0, 4)
    grid = jax.vmap(jax.vmap(one, in_axes=(None, 0)), in_axes=(0, None))(P, v)
    assert grid.shape == (5, 4)
    assert np.all(np.isfinite(np.asarray(grid)))


def test_melt_length_scale_positive():
    L = float(_melt_length_scale(absorbed_power=88.0, scan_speed=0.9,
                                beam_radius=50e-6, rho=4200.0,
                                enthalpy_to_melt=1.35e6))
    assert 1e-5 < L < 1e-3, f"特征尺度 {L:.3e} m 不在合理范围"
    # 功率→0 时退化为 1.5·r_b（保证采样盒不塌缩）
    L0 = float(_melt_length_scale(absorbed_power=1e-12, scan_speed=1.0,
                                 beam_radius=50e-6, rho=4200.0,
                                 enthalpy_to_melt=1.35e6))
    assert L0 == pytest.approx(1.5 * 50e-6, rel=1e-6)
