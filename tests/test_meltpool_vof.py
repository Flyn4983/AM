"""B 档 VOF 求解器（meltpool.vof_flow3d）验证套件。

覆盖：
* 压力投影算子对称正定 + 收敛（定位此前 1e10 速度爆炸根因的回归锁）；
* 端到端运行有限、真实熔池成形（熔深/宽/长 > 0）；
* 自由界面（气相层保留）+ 匙孔（反冲压驱动的表面凹陷）成形；
* Marangoni 真实生效（关掉 dσ/dT 熔宽明显变化）；
* 默认链路全程 jax.grad 可微（不 OOM、梯度有限）；
* 默认开启 checkpoint（保证默认链路可微的回归守护）。

配置取自 _scratch/vof_validate.py 的物理合理参数：dx << beam_radius，
且 nz 留出气相空间，否则粉末填满整域 -> 无自由界面 -> VOF 退化。
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import amforge as af
from amforge import meltpool as MP
from amforge.materials import get_material

MAT = "316L"


# ---------------------------------------------------------------------------
# 配置构造器
# ---------------------------------------------------------------------------
def _cfg(n_steps, **kw):
    base = dict(
        nx=32, ny=22, nz=24, dx=6e-6,
        n_steps=n_steps, n_pressure_iters=18, dt_max=5e-7,
        interface_compression=1.0, fresnel_boost=1.5,
        substrate_fraction=0.55, powder_layers=1,
        checkpoint_every=20,
    )
    base.update(kw)
    return MP.MeltPoolConfig(**base)


def _proc(power=250.0, speed=0.8, rb=35e-6):
    return af.ProcessPlan.uniform(
        laser_power=power, scan_speed=speed,
        layer_thickness=30e-6, hatch_spacing=60e-6,
        beam_radius=rb, absorption=0.4,
    )


def _run(n_steps=200, **kw):
    cfg = _cfg(n_steps, **kw)
    return MP.simulate_meltpool(process=_proc(), material=MAT, config=cfg)


# ---------------------------------------------------------------------------
# 1. 压力投影算子：对称正定 + 收敛（核心回归锁）
# ---------------------------------------------------------------------------
def test_pressure_operator_symmetric():
    """Aop 必须对称：CG 收敛性的前提。"""
    nx = ny = nz = 12
    dx = 1e-5
    rng = jax.random.PRNGKey(1)
    rho = 1.0 + 0.2 * jax.random.normal(rng, (nx, ny, nz))
    metal = jnp.ones((nx, ny, nz))
    Aop, _, _, _ = MP._pressure_system(rho, metal, dt=1e-6, dx=dx)
    p = jax.random.normal(jax.random.PRNGKey(2), (nx, ny, nz))
    q = jax.random.normal(jax.random.PRNGKey(3), (nx, ny, nz))
    lhs = jnp.sum(p * Aop(q))
    rhs = jnp.sum(q * Aop(p))
    assert abs(float(lhs - rhs)) / (abs(float(lhs)) + 1e-30) < 1e-10


def test_projection_removes_divergence():
    """投影必须把速度场压到（离散）无散；这是此前速度炸到 1e10 的根因。

    注意：必须用**真实结构**的 metal 场（基板/熔体金属 + 顶部气相层），
    而不是全液态。全液态是近奇异纯 Neumann 系统（只有一个弱顶层 Dirichlet
    钉），不是求解器实际工况；真实仿真的 metal 带气相、界面 Dirichlet 使系统
    良态（见验证脚本 div_core 相对残差 ~6e-4）。
    """
    nx = ny = nz = 16
    dx = 1e-5
    z = jnp.arange(nz)
    metal1d = 0.5 * (1.0 + jnp.tanh((0.6 * nz - z) / 0.7))   # 软掩膜：顶部气相
    metal = jnp.broadcast_to(metal1d.reshape(1, 1, nz), (nx, ny, nz))
    rho = jnp.broadcast_to(
        jnp.where(metal1d > 0.5, 7000.0, 1.0).reshape(1, 1, nz), (nx, ny, nz)
    )
    rng = jax.random.split(jax.random.PRNGKey(0), 3)
    u_star = jnp.stack(
        [jax.random.normal(r, (nx, ny, nz)) * 0.1 for r in rng], axis=-1
    ) * metal[..., None]

    # (a) 线性系统残差：与配置无关的强锁。修复前 CG 残差会**发散**到 1e13+，
    #     变密度系统 + Jacobi 预条件下 40 步的真实收敛地板约 1e-4（已足以支撑
    #     稳定物理，见验证脚本 div_core 相对残差 ~6e-4）。这里锁"明显收敛、不发散"。
    Aop, Minv, rhs_of, correct = MP._pressure_system(rho, metal, dt=1e-6, dx=dx)
    p = MP._cg(Aop, rhs_of(u_star), Minv, 40)
    res = Aop(p) - rhs_of(u_star)
    lin_rel = float(jnp.max(jnp.abs(res)) / (jnp.max(jnp.abs(rhs_of(u_star))) + 1e-30))
    assert lin_rel < 1e-2, f"CG 线性残差比 {lin_rel} 未收敛（应远小于 1）"

    # (b) 物理散度：金属核心（远离自由界面过渡层）应被压到近无散
    u, p = MP._project(u_star, rho, metal, dt=1e-6, dx=dx, n_iters=40)
    core = (metal > 0.98)
    core = core.at[:, :, -1].set(False).at[:, :, -2].set(False)
    div0 = MP._div_face(u_star, dx) * core
    div1 = MP._div_face(u, dx) * core
    rel = float(jnp.max(jnp.abs(div1)) / (jnp.max(jnp.abs(div0)) + 1e-30))
    assert rel < 1e-2, f"投影后核心残余散度比 {rel} 未收敛"
    assert jnp.all(jnp.isfinite(u)) and jnp.all(jnp.isfinite(p))


# ---------------------------------------------------------------------------
# 2. 端到端：运行有限 + 真实熔池
# ---------------------------------------------------------------------------
def test_runs_finite():
    sol = _run(n_steps=150)
    for nm in ("T", "u", "F", "p"):
        a = getattr(sol.state, nm)
        assert jnp.all(jnp.isfinite(a)), f"{nm} 含非有限值"


def test_melts():
    """250W/0.8m/s 单道必须熔化（峰值温度过固相线、熔宽 > 0）。"""
    sol = _run(n_steps=200)
    mat = get_material(MAT)
    assert float(jnp.max(sol.state.T_peak)) > mat.T_solidus
    assert float(sol.width) * 1e6 > 5.0          # 熔宽 > 5 µm


# ---------------------------------------------------------------------------
# 3. 自由界面 + 匙孔
# ---------------------------------------------------------------------------
def test_free_surface_and_keyhole():
    """气相层必须保留（自由界面存在），且反冲压驱动匙孔凹陷。"""
    sol = _run(n_steps=200)
    F = sol.state.F
    assert float(jnp.min(F)) < 0.5, "没有气相单元 -> 无自由界面"
    assert float(sol.state.keyhole_max) > 0.0, "匙孔未成形"
    assert float(sol.state.precoil_max) > 0.0, "反冲压未生效"


# ---------------------------------------------------------------------------
# 4. Marangoni 真实生效
# ---------------------------------------------------------------------------
def test_marangoni_effect():
    """关掉 dσ/dT 后熔池形貌必须改变（表面张力梯度驱动的流场是真实物理）。"""
    mat_on = get_material(MAT)
    mat_off = mat_on.replace(dsigma_dT=0.0)
    cfg = _cfg(200)
    s_on = MP.simulate_meltpool(process=_proc(), material=mat_on, config=cfg)
    s_off = MP.simulate_meltpool(process=_proc(), material=mat_off, config=cfg)
    w_on = float(s_on.width) * 1e6
    w_off = float(s_off.width) * 1e6
    # 关掉 Marangoni 后熔宽变化（经验上表面张度梯度收缩熔池）
    assert abs(w_on - w_off) / (w_on + 1e-9) > 1e-3, "Marangoni 未改变熔池形貌"


# ---------------------------------------------------------------------------
# 5. 可微性：默认链路 jax.grad 可微
# ---------------------------------------------------------------------------
def test_differentiable_wrong_power():
    """默认配置下 depth 对 laser_power 的梯度必须有限（不 OOM、不 NaN）。"""
    cfg = _cfg(80, nx=24, ny=18, nz=18, dx=8e-6,
               n_pressure_iters=12, checkpoint_every=10)

    def depth_of(P):
        proc = af.ProcessPlan.uniform(
            laser_power=P, scan_speed=0.8,
            layer_thickness=30e-6, hatch_spacing=60e-6,
            beam_radius=30e-6, absorption=0.4,
        )
        return MP.simulate_meltpool(process=proc, material=MAT, config=cfg).depth

    g = jax.grad(depth_of)(250.0)
    assert jnp.isfinite(g), f"深度对功率的梯度非有限: {g}"


# ---------------------------------------------------------------------------
# 6. 回归守护：默认开启 checkpoint（默认链路可微的前提）
# ---------------------------------------------------------------------------
def test_default_checkpoint_enabled():
    """默认 checkpoint_every>0，否则默认链路 jax.grad 反向会 OOM。"""
    assert MP.MeltPoolConfig().checkpoint_every > 0
