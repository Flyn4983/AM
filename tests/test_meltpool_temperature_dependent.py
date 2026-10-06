"""meltpool.fdm 物理保真升级（A 档）的回归守卫。

覆盖两件新物理：
1. Beer-Lambert 体积吸收热源（面内高斯 × 深度指数衰减，功率守恒）。
2. 温度相关导热系数（固→液 k 跳变经液相分数线性混合），须改变熔池结果且可微。

全部走真实数值 FVM 焓法，不退化、可对工艺参数求梯度。
"""
import jax
import jax.numpy as jnp

from amforge import geometry as G
from amforge.core.contracts import ProcessPlan
from amforge.materials import get_material
from amforge.meltpool import solve_meltpool_fdm, _laser_source_fdm


# ---------------------------------------------------------------------------
# 构造器（与 test_default_solver_numerical 一致的轻量夹具）
# ---------------------------------------------------------------------------
def _geo():
    return G.from_sdf_fn(
        lambda x: jnp.linalg.norm(x, axis=-1) - 0.45e-3,
        bounds=[(-0.6e-3, 0.6e-3)] * 3, spacing=50e-6, name="td_solver",
    )


def _proc():
    return ProcessPlan.uniform(laser_power=250.0, scan_speed=0.8,
                               layer_thickness=40e-6, hatch_spacing=80e-6,
                               beam_radius=50e-6, absorption=0.4)


# ---------------------------------------------------------------------------
# 1. Beer-Lambert 体积吸收热源
# ---------------------------------------------------------------------------
def test_laser_source_power_conserving():
    """∫q dV = A_eff·P（总吸收功率守恒），且表面最强、随深度非负。"""
    n = 160
    # 高分辨盒：rb、δ 远小于盒且远大于体素，保证高斯/指数被充分采样
    # （求解器内对亚栅格光束会 r_eff=max(rb,dx) 限幅，此处用已分辨网格验证解析式）。
    rb, delta = 300e-6, 200e-6
    half = 3.0e-3
    xs = jnp.linspace(-half, half, n)
    ys = jnp.linspace(-half, half, n)
    zs = jnp.linspace(0.0, -2.5e-3, n)
    coords = jnp.stack(jnp.meshgrid(xs, ys, zs, indexing="ij"), axis=-1)
    dxy = float(xs[1] - xs[0])
    dz = abs(float(zs[1] - zs[0]))
    dv = dxy * dxy * dz

    A_eff, P = 0.4, 250.0
    src = _laser_source_fdm(coords, 0.0, rb=rb, A_eff=A_eff, P=P,
                            absorption_depth=delta, dx=dxy)

    assert jnp.all(jnp.isfinite(src)), "热源含 NaN/Inf"
    assert jnp.all(src >= 0.0), "热源出现负值"

    total = float(jnp.sum(src)) * dv
    expected = A_eff * P
    rel = abs(total - expected) / expected
    assert rel < 0.05, f"热源功率不守恒: 数值={total:.3f} 期望={expected:.3f} (rel={rel:.3f})"


def test_laser_source_exponential_depth_decay():
    """深度方向应为指数衰减（自相似：2δ 处/δ 处 ≈ (δ 处/表面)²）。"""
    n = 60
    dz = 20e-6                       # 体素远小于吸收深度，保证可分辨
    xs = jnp.linspace(-1.0e-3, 1.0e-3, 8)
    ys = jnp.linspace(-1.0e-3, 1.0e-3, 8)
    zs = -jnp.arange(n) * dz         # 0, -dz, -2dz, ...（z 向下为负）
    coords = jnp.stack(jnp.meshgrid(xs, ys, zs, indexing="ij"), axis=-1)
    delta = 200e-6                   # = 10 体素

    src = _laser_source_fdm(coords, 0.0, rb=50e-6, A_eff=0.4, P=250.0,
                            absorption_depth=delta, dx=dz)
    prof = src[4, 4, :]              # 中心柱（x=y=0）
    assert prof[0] > 0.0

    # 单调下降
    assert jnp.all(jnp.diff(prof) < 0), "热源随深度未单调衰减"

    # 指数自相似：log(q(2δ)/q(0)) ≈ 2·log(q(δ)/q(0))
    def ratio_at(zd):
        k = int(round(zd / dz))
        return prof[k] / prof[0]
    r1 = float(ratio_at(delta))
    r2 = float(ratio_at(2.0 * delta))
    # 解析期望 r1≈e^-1, r2≈e^-2 → r2 ≈ r1^2
    assert abs(jnp.log(r2) - 2.0 * jnp.log(r1)) < 0.25, \
        f"深度衰减非指数: r(δ)={r1:.3f}, r(2δ)={r2:.3f}, " \
        f"期望 r(2δ)≈r(δ)²={r1**2:.3f}"


# ---------------------------------------------------------------------------
# 2. 温度相关导热系数（固→液 k 跳变）须改变结果且可微
# ---------------------------------------------------------------------------
def test_temperature_dependent_k_changes_result():
    """有 k 跳变的材料 vs 压平 k 跳变的材料，熔池尺寸须不同（特征已激活）。"""
    mat_jump = get_material("Ti6Al4V")                 # k_solid=21, k_liquid=30
    mat_flat = mat_jump.replace(k_liquid=mat_jump.k_solid)  # 消除跳变

    kw = {"n_grid": 24, "n_steps": 80}
    r_jump = solve_meltpool_fdm(geometry=_geo(), process=_proc(),
                                params={"material": mat_jump, **kw})
    r_flat = solve_meltpool_fdm(geometry=_geo(), process=_proc(),
                                params={"material": mat_flat, **kw})

    assert jnp.isfinite(r_jump.depth) and jnp.isfinite(r_flat.depth)
    assert float(r_jump.depth) > 1e-6 and float(r_flat.depth) > 1e-6, \
        "两种材料都须真实熔化"
    assert float(r_jump.depth) != float(r_flat.depth), \
        "温度相关导热未改变熔池结果（特征疑似未激活）"


def test_meltpool_fdm_temperature_dependent_differentiable():
    """升级后整条 meltpool.fdm 仍真实熔化、数值有限、对激光功率可微。"""
    geo, proc = _geo(), _proc()
    kw = {"material": "316L", "n_grid": 24, "n_steps": 80}

    out = solve_meltpool_fdm(geometry=geo, process=proc, params=kw)
    assert jnp.all(jnp.isfinite(out.temperature)), "温度场非有限"
    assert float(out.depth) > 1e-6, f"未熔化: {float(out.depth)}"
    assert float(out.depth) < 5e-3, "熔深量级异常"

    def loss(P):
        p2 = proc.replace(laser_power=P)
        r = solve_meltpool_fdm(geometry=geo, process=p2, params=kw)
        return r.depth + r.lof_indicator * 1e-3

    g = jax.grad(loss)(300.0)
    assert jnp.isfinite(g), "梯度含 NaN/Inf（温度相关物性破坏了可微性）"
