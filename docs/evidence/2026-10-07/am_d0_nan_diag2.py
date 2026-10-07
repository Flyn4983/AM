"""D0 诊断 2：把 _smooth_defect_cost 的梯度按项拆开，定位 NaN 出在哪一项。

判据（跑前登记）：
  B1 若某一项的 grad 非有限 ⇒ 该项依赖的热场量是 NaN-梯度源；
  B2 若三项都有限而总 loss 梯度仍 NaN ⇒ NaN 来自组合（除法/取模/where 分支）；
  B3 逐项追到 ThermalHistory 的四个场量，指明是**显式热解反传**还是**契约比值**。
"""
import jax
import jax.numpy as jnp
import numpy as np

import amforge as af
from amforge.inverse import (simulate, thermal_tier, tier_bounds,
                             project_process_feasible)
from amforge.materials import get_material
from amforge.process import normalize_process, denormalize_process
import optax


def tiny_geo():
    n, s = 8, 2e-4
    xs = jnp.linspace(-s, s, n)
    X, Y = jnp.meshgrid(xs, xs, indexing="ij")
    return af.PartGeometry(sdf=jnp.maximum(jnp.abs(X), jnp.abs(Y)) - 0.6 * s,
                           origin=-s * jnp.ones(2), spacing=2 * s / (n - 1),
                           dim=2, name="tiny")


def gf(g):
    a = np.asarray(jax.tree_util.tree_leaves(g)[0], dtype=float)
    return f"finite={bool(np.isfinite(a).all())} |g|={np.linalg.norm(np.nan_to_num(a)):.3e} nan@{np.argwhere(~np.isfinite(a)).tolist()[:4]}"


geo = tiny_geo()
base = af.ProcessPlan.uniform(n_layers=3, laser_power=120.0, scan_speed=1.5)
mat = get_material("316L")
T_l = float(mat.T_liquidus)
sp = {"n_grid": 6, "max_layers": 3, "n_sub_cp": 8, "micro": {}}
sim_p = dict(sp)
sim_p["micro"] = dict(sim_p.get("micro", {}))
sim_p = thermal_tier(sim_p, geo, base, material="316L")
bnds = tier_bounds(sim_p)
nl, modality = 3, base.modality
z = project_process_feasible(normalize_process(base, bounds=bnds), material="316L",
                             n_layers=nl, modality=modality, params=sim_p)


def fields(zz):
    plan = denormalize_process(zz, n_layers=nl, modality=modality, bounds=bnds)
    o = simulate(geo, plan, material="316L", params=sim_p,
                 asbuilt_solver="buildup", micro_solver="surrogate",
                 mbd_solver="surrogate")
    return o["thermal"]


def mk(fn):
    return lambda zz: jnp.mean(fn(fields(zz)))


terms = {
    "peak_only": mk(lambda t: t.peak_temperature),
    "lof": mk(lambda t: jax.nn.softplus(T_l - jnp.atleast_1d(t.peak_temperature)) / max(T_l, 1.0)),
    "key": mk(lambda t: jax.nn.softplus(jnp.atleast_1d(t.peak_temperature) - 1.4 * T_l) / max(T_l, 1.0)),
    "por(gr_ratio)": mk(lambda t: jax.nn.softplus(1.0e3 - jnp.atleast_1d(t.gr_ratio())) / 1.0e3),
    "thermal_gradient": mk(lambda t: t.thermal_gradient),
    "solidification_rate": mk(lambda t: t.solidification_rate),
    "cooling_rate": mk(lambda t: t.cooling_rate),
    "time_above_melt": mk(lambda t: t.time_above_melt),
    "final_temperature": mk(lambda t: t.final_temperature),
}
print("=== 逐项梯度有限性（z=基准投影点）===")
for k, f in terms.items():
    v, g = jax.value_and_grad(f)(z)
    print(f"{k:22s} loss={float(v):.6e} grad[{gf(g)}]")

# gr_ratio 的分母下界：看 solidification_rate 是否 <= 1e-12 导致 max() 分支切换
t0 = fields(z)
sr = np.asarray(t0.solidification_rate, dtype=float)
tg = np.asarray(t0.thermal_gradient, dtype=float)
print("\n=== 分母/分子实况 ===")
print(f"solidification_rate: min={sr.min():.6e} n(<=1e-12)={int((sr <= 1e-12).sum())}/{sr.size}")
print(f"thermal_gradient   : min={tg.min():.6e} max={tg.max():.6e} n(==0)={int((tg == 0).sum())}/{tg.size}")
