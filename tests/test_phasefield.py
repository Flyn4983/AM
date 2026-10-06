"""AM 枝晶相场求解器（micro.phasefield）测试

覆盖：
  * 4 重界面能各向异性（轴对齐时主轴臂 > 对角臂，arm/diag > 1）
  * 热梯度取向（G 方向优先生长：θ₀=45° 对角取向被抑制，arm/diag < 轴对齐）
  * 可微性（固相分数对界面能各向异性强度 eps4 的梯度有限）
  * 契约合规（solve_phasefield 产出同契约 MicrostructureResult，下游零改动）
  * inverse 接线（simulate 的 micro_solver='phasefield' 分支跑通且 micro 有限）
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "True")

import jax
import jax.numpy as jnp
import pytest

from amforge.phasefield import (
    DendriteConfig, init_isothermal, run_phase_field,
    solid_fraction, arm_symmetry_4fold, solve_phasefield,
)
from amforge.core.contracts import (
    ThermalHistory, MeltPoolResult, MicrostructureResult,
)
from amforge import geometry as G
from amforge.process import heuristic_plan
from amforge.inverse import (
    simulate, optimize_dimensional, optimize_geometry_process,
)


# 轻量 RVE（本机小网格；48G 卡上可放大 grid / n_steps 做定量尖端选择）
_GRID = (48, 48)
_N = 120


def _cfg(theta0=0.0, eps4=0.15, **kw):
    return DendriteConfig(
        dx=1.0, dt=0.08, eps4=eps4, eps_k=0.05, lam=6.0, tau0=0.2,
        D=1.0, folds=4, principal_axis=theta0,
        coupled_thermal=False, bc="neumann", **kw,
    )


def test_fourfold_symmetry_axis_preferred():
    """θ₀=0（轴对齐）时主轴臂比对角臂长（arm/diag > 1），体现 4 重枝晶。"""
    cfg = _cfg(theta0=0.0)
    init = init_isothermal(_GRID, undercool=0.55, seed_radius=4.0, dx=1.0)
    phi = run_phase_field(cfg, init, n_steps=_N)["phi"]
    ratio = float(arm_symmetry_4fold(phi))
    assert ratio > 1.0, f"4 重对称性未体现: arm/diag={ratio}"


def test_g_orientation_rotation():
    """热梯度取向：枝晶主轴随 θ₀ 旋转对齐 G。

    θ₀=0°/90°（轴对齐）长出 4 臂（arm/diag > 1）；
    θ₀=45°（对角）时 4 臂落在对角的 45/135/225/315°，轴方向反而短（arm/diag < 1）。
    """
    def ratio_for(theta0):
        cfg = _cfg(theta0=theta0)
        init = init_isothermal(_GRID, 0.55, 4.0, dx=1.0)
        return float(arm_symmetry_4fold(run_phase_field(cfg, init, n_steps=_N)["phi"]))

    r0 = ratio_for(0.0)
    r90 = ratio_for(jnp.pi / 2.0)
    r45 = ratio_for(jnp.pi / 4.0)
    assert r0 > 1.0 and r90 > 1.0, "轴对齐未长出 4 臂"
    assert r45 < r0 and r45 < r90, "对角取向未受抑制（G 方向优先生长失效）"


def test_phasefield_differentiable_wrt_eps4():
    """可微性：固相分数对界面能各向异性强度 eps4 的梯度有限且非零。"""
    def sf_of_eps4(e4):
        cfg = _cfg(eps4=e4)
        init = init_isothermal(_GRID, 0.55, 4.0, dx=1.0)
        return solid_fraction(run_phase_field(cfg, init, n_steps=_N)["phi"])

    g = jax.grad(sf_of_eps4)(0.15)
    assert jnp.isfinite(g), "d(sf)/d(eps4) 非有限"
    assert abs(float(g)) > 0.0, "梯度为零（参数未连通）"


def test_semi_implicit_stable_at_large_lam():
    """半隐式在大 λ + 大 dt 下稳定，而显式 Euler 在同参数下失稳（扩散刚性越界）。

    这是"换半隐式时间格式"的价值证明：薄界面（λ 大，定量 tip selection 所需）
    下可用大时间步稳定推进，显式 Euler 受扩散数 ``dt<dx²/4`` 限制会发散。
    """
    grid = (40, 40)
    n = 40

    def run(si: bool):
        cfg = DendriteConfig(
            dx=1.0, dt=0.5, eps4=0.15, eps_k=0.05, lam=10.0, tau0=0.2,
            D=1.0, folds=4, principal_axis=0.0,
            coupled_thermal=True, semi_implicit=si, bc="neumann",
        )
        init = init_isothermal(grid, undercool=0.55, seed_radius=4.0, dx=1.0)
        out = run_phase_field(cfg, init, n_steps=n)
        return out["phi"], out["u"]

    phi_si, u_si = run(True)
    phi_ex, u_ex = run(False)
    # 半隐式：大 λ/dt 下保持物理量级（扩散刚性被隐式吸收）
    assert jnp.all(jnp.isfinite(phi_si)) and jnp.all(jnp.isfinite(u_si)), \
        "半隐式在大 λ/dt 下应稳定（有限），实际出现非有限值"
    assert float(jnp.max(jnp.abs(u_si))) < 50.0, \
        f"半隐式 u 应保持物理量级，实际 max|u|={float(jnp.max(jnp.abs(u_si))):.2e}"
    # 半隐式确有枝晶生长（固相分数远超初始晶核），非平凡演化
    sf_si = float(solid_fraction(phi_si))
    assert sf_si > 0.1, \
        f"半隐式应已发生枝晶生长，实际固相分数={sf_si:.3f}"
    # 显式 Euler：相同大 dt 下扩散项越界 → von Neumann 棋盘失稳，幅值暴涨
    # （float64 范围内不溢出为 inf，但 |u| 达 ~1e31，远超物理量级）
    assert float(jnp.max(jnp.abs(u_ex))) > 1e3, \
        f"显式 Euler 在大 λ/dt 下应失稳（幅值暴涨），实际 max|u|={float(jnp.max(jnp.abs(u_ex))):.2e}"


def _mini_thermal_meltpool(shape=(16, 16)):
    """构造最小 ThermalHistory + MeltPoolResult 供 solve_phasefield 映射。"""
    T = jnp.ones(shape) * 1700.0
    Gv = jnp.full(shape, 3.0e6)     # K/m
    Rv = jnp.full(shape, 0.05)      # m/s  -> G·R = 1.5e5 K/s
    cool = jnp.full(shape, 1.0e5)
    tam = jnp.full(shape, 0.5)
    ft = jnp.full(shape, 300.0)
    thermal = ThermalHistory(
        peak_temperature=T, cooling_rate=cool, thermal_gradient=Gv,
        solidification_rate=Rv, time_above_melt=tam, final_temperature=ft,
        spacing=1e-5, dim=2,
    )
    mp = MeltPoolResult(
        vof=jnp.ones(shape), temperature=T, liquid_fraction=jnp.ones(shape),
        velocity=jnp.zeros((*shape, 2)), pressure=jnp.zeros(shape),
        depth=jnp.array(1e-4), width=jnp.array(2e-4), length=jnp.array(3e-4),
        keyhole_depth=jnp.array(1e-4), lof_indicator=jnp.array(0.1),
        porosity_indicator=jnp.array(0.2), spatter_indicator=jnp.array(0.1),
        spacing=5e-6, dim=2,
    )
    return thermal, mp


def test_solve_phasefield_contract():
    """solve_phasefield 产出同契约 MicrostructureResult，字段有限、形状匹配。"""
    thermal, mp = _mini_thermal_meltpool()
    params = {"grid": (32, 32), "n_steps": 50, "mode": "isothermal",
              "eps4": 0.15, "folds": 4, "gradient_axis": 1}
    res = solve_phasefield(thermal=thermal, meltpool=mp, params=params)
    assert isinstance(res, MicrostructureResult)
    assert res.phi.shape == thermal.peak_temperature.shape
    assert res.grain_size.shape == thermal.peak_temperature.shape
    assert res.dim == thermal.dim
    for leaf in jax.tree_util.tree_leaves(res):
        assert jnp.all(jnp.isfinite(leaf)), "MicrostructureResult 含非有限值"
    # 柱状分数应在 [0,1]
    assert jnp.all(res.columnar_fraction >= 0.0)
    assert jnp.all(res.columnar_fraction <= 1.0)
    # 取向应沿 G：gradient_axis=1 -> θ₀=π/2
    assert jnp.isclose(float(jnp.max(res.orientation)), jnp.pi / 2.0, atol=1e-6)


def _small_geo(name="pf-demo"):
    return G.from_sdf_fn(
        lambda x: jnp.linalg.norm(x, axis=-1) - 0.4e-3,
        bounds=[(-0.5e-3, 0.5e-3)] * 3, spacing=120e-6, name=name,
    )


def test_inverse_simulate_uses_phasefield():
    """inverse.simulate 的 micro_solver='phasefield' 分支跑通且 micro 有限。"""
    geo = _small_geo()
    plan = heuristic_plan(geo, material="316L", modality="SLM")
    out = simulate(
        geo, plan, material="316L",
        params={"n_grid": 8,
                "micro": {"grid": (24, 24), "n_steps": 30, "mode": "isothermal"}},
        asbuilt_solver="buildup", micro_solver="phasefield",
    )
    micro = out["micro"]
    assert isinstance(micro, MicrostructureResult)
    for leaf in jax.tree_util.tree_leaves(micro):
        assert jnp.all(jnp.isfinite(leaf)), "phasefield micro 含非有限值"


def test_inverse_default_surrogate_unchanged():
    """回归守卫：默认 micro_solver='surrogate' 零改动（统计代理路径不变）。"""
    geo = _small_geo(name="pf-demo2")
    plan = heuristic_plan(geo, material="316L", modality="SLM")
    out = simulate(geo, plan, material="316L", params={"n_grid": 8})
    assert isinstance(out["micro"], MicrostructureResult)
    assert jnp.all(jnp.isfinite(out["micro"].grain_size))


def test_optimize_dimensional_phasefield_runs():
    """micro_solver='phasefield' 透传进 optimize_dimensional（轻量 buildup 路径，无 FEM）。"""
    geo = _small_geo(name="pf-demo3")
    plan_init = heuristic_plan(geo, material="316L", modality="SLM")
    res = optimize_dimensional(
        geo, process_init=plan_init, material="316L",
        params={"n_grid": 6, "micro": {"grid": (20, 20), "n_steps": 20,
                                       "mode": "isothermal"}},
        asbuilt_solver="buildup", micro_solver="phasefield",
        n_steps=4, learning_rate=0.05, weights=dict(geom=5.0, stress=1.0),
        verbose=False,
    )
    assert len(res["loss_history"]) == 4
    assert jnp.all(jnp.isfinite(jnp.asarray(res["loss_history"])))
    # 损失历史有限即证明 micro→constitutive→asbuilt 链路在 phasefield 下无 NaN
    for leaf in jax.tree_util.tree_leaves(res):
        assert jnp.all(jnp.isfinite(leaf))


def test_optimize_geometry_process_phasefield_runs():
    """micro_solver='phasefield' 透传进联合优化 _joint_forward（轻量 buildup 路径）。"""
    geo = _small_geo(name="pf-demo4")
    res = optimize_geometry_process(
        geo, material="316L",
        params={"n_grid": 6, "micro": {"grid": (20, 20), "n_steps": 20,
                                       "mode": "isothermal"}},
        asbuilt_solver="buildup", micro_solver="phasefield",
        n_steps=3, learning_rate=0.05, smoothness=0.02,
        constraint_weight=0.0, verbose=False,
    )
    assert len(res["loss_history"]) == 3
    assert jnp.all(jnp.isfinite(jnp.asarray(res["loss_history"])))
    # 联合优化在 phasefield 微观下可微且有限（loss 有限即证明链路连通 micro）
    for leaf in jax.tree_util.tree_leaves(res):
        assert jnp.all(jnp.isfinite(leaf))
