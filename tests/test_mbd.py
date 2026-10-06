"""P2-③ 多体 + FEM 耦合装配求解器测试（任务 #39）。

覆盖：AssemblyResult 契约合规、simulate 默认 surrogate 零回归、coupled 路径
有限 + 可微、质量矩阵 SPD、Newton 后端 guarded 跳过。
"""
import jax
import jax.numpy as jnp
import pytest
from dataclasses import replace

from amforge import geometry as G
from amforge.process import heuristic_plan
from amforge.core.contracts import AssemblyResult
from amforge.mbd import solve_assembly
from amforge.inverse import simulate
from diffmech.methods.mbd import (
    MbdChainConfig, planar_chain_mass_matrix, static_couple, solve_mbd_newton,
)


def _small_geo(name="mbd-demo"):
    sdf = -jnp.ones((8, 8)) * 1e-3
    sdf = sdf.at[3:5, 3:5].set(5e-4)            # 小凸台
    return G.PartGeometry(sdf=sdf, origin=jnp.array([0.0, 0.0]),
                          spacing=1e-3, dim=2, name=name)


def _asbuilt(geo, material="316L"):
    plan = heuristic_plan(geo, material=material, modality="SLM")
    out = simulate(geo, plan, material=material,
                   params={"n_grid": 8, "n_joints": 3})
    return out["asbuilt"]


def _cfg(n=3):
    return MbdChainConfig(mass=jnp.full(n, 1e-3), length=jnp.full(n, 1e-2),
                          com_dist=jnp.full(n, 5e-3), inertia=jnp.full(n, 1e-9))


# ---------------------------------------------------------------------------
def test_assembly_result_contract():
    """AssemblyResult 字段形状 / 有限 / assembly_score 正确。"""
    n = 3
    res = AssemblyResult(
        joint_load=jnp.ones(n), relative_displacement=jnp.full(n, 0.1),
        flexural_stiffness=jnp.full(n, 1e1), stability_margin=jnp.asarray(0.9),
        porosity=jnp.asarray(0.05), dim=2)
    assert res.joint_load.shape == (n,)
    assert res.relative_displacement.shape == (n,)
    assert res.flexural_stiffness.shape == (n,)
    assert jnp.isfinite(res.stability_margin)
    assert float(res.assembly_score()) > 0.0


def test_surrogate_default_in_simulate():
    """回归守卫：simulate 默认 mbd_solver='surrogate' 零改动，assembly 有限。"""
    geo = _small_geo()
    plan = heuristic_plan(geo, material="316L", modality="SLM")
    out = simulate(geo, plan, material="316L",
                   params={"n_grid": 8, "n_joints": 3})
    assert isinstance(out["assembly"], AssemblyResult)
    assert jnp.all(jnp.isfinite(out["assembly"].joint_load))
    assert jnp.all(jnp.isfinite(out["assembly"].relative_displacement))
    assert jnp.all(jnp.isfinite(out["assembly"].flexural_stiffness))


def test_mass_matrix_spd():
    """闭式广义质量矩阵对称正定（Featherstone/CRBA 等价）。"""
    cfg = _cfg(3)
    M = planar_chain_mass_matrix(jnp.zeros(3), cfg)
    assert M.shape == (3, 3)
    assert jnp.allclose(M, M.T, atol=1e-12)
    eigs = jnp.linalg.eigvalsh(M)
    assert jnp.all(eigs > 0), "质量矩阵应正定"


def test_mbd_coupled_finite():
    """coupled 路径：FEM↔MBD 耦合求解有限，线性化稳定裕度 > 0。"""
    geo = _small_geo()
    asbuilt = _asbuilt(geo)
    res = solve_assembly(geo, asbuilt, params={"n_joints": 3},
                         mbd_fidelity="coupled")
    assert isinstance(res, AssemblyResult)
    assert jnp.all(jnp.isfinite(res.joint_load))
    assert jnp.all(jnp.isfinite(res.relative_displacement))
    assert jnp.all(jnp.isfinite(res.flexural_stiffness))
    assert res.joint_load.shape == (3,)
    assert float(res.stability_margin) > 0.0, "线性化应稳定（min eig(M^{-1}K)>0）"


def test_mbd_coupled_differentiable():
    """coupled 耦合求解对关节柔度可微（梯度有限）。"""
    cfg = _cfg(3)

    def stab_of_scale(scale):
        K = jnp.full(3, 1e1) * scale
        _, _, s = static_couple(jnp.zeros(3), cfg, K)
        return s

    g = jax.grad(stab_of_scale)(1.0)
    assert jnp.isfinite(g), "d(stability)/d(stiffness scale) 非有限"


def test_inverse_chain_differentiable_through_mbd():
    """可微评估（任务硬要求）：梯度穿过 process→thermal→asbuilt→assembly 整链。"""
    geo = _small_geo()
    plan0 = heuristic_plan(geo, material="316L", modality="SLM")

    def scalar_of_laser_power(power):
        plan = replace(plan0, laser_power=power)
        out = simulate(geo, plan, material="316L",
                       params={"n_grid": 8, "n_joints": 3},
                       mbd_solver="coupled")
        return jnp.sum(out["assembly"].joint_load ** 2
                       + out["assembly"].relative_displacement ** 2)

    g = jax.grad(scalar_of_laser_power)(float(jnp.ravel(plan0.laser_power)[0]))
    assert jnp.isfinite(g), "整链（含 coupled MBD）梯度非有限"


def test_newton_bridge_guarded():
    """Newton 后端：未安装则抛清晰 RuntimeError；已安装则跳过（48G 验证）。"""
    try:
        import newton  # noqa: F401
    except Exception:
        available = False
    else:
        available = True

    cfg = _cfg(3)
    if not available:
        with pytest.raises(RuntimeError):
            solve_mbd_newton(chain_cfg=cfg, q0=jnp.zeros(3))
    else:
        pytest.skip("Newton 已安装：Warp 后端在 48G Ubuntu 上验证")
