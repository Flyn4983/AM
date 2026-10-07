"""D0 诊断：test_inverse_controller_lowers_defect_cost 的 cost_corr=nan 从哪来。

逐步打印 Adam 循环内的 loss / grad / z，以及修正后工艺仿真场的 peak/cooling/
solidification 有限性，定位 NaN 是**前向解发散**还是**代价函数 0/0**。
判据（跑前登记）：
  J1 若 z 在循环中途变非有限 ⇒ 优化器发散；
  J2 若 z 始终有限但 thermal.peak 非有限 ⇒ 显式热解在该档 CFL 下炸；
  J3 若场全部有限而 gr_ratio 非有限 ⇒ _smooth_defect_cost 的除法口径问题。
"""
import jax
import jax.numpy as jnp
import numpy as np

import amforge as af
from amforge.core.contracts import MonitoringState
from amforge.monitoring import _smooth_defect_cost
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


def fin(v):
    a = np.asarray(v, dtype=float)
    return f"finite={bool(np.isfinite(a).all())} min={np.nanmin(a):.4g} max={np.nanmax(a):.4g}"


geo = tiny_geo()
base = af.ProcessPlan.uniform(n_layers=3, laser_power=120.0, scan_speed=1.5)
mat = get_material("316L")
sp = {"n_grid": 6, "max_layers": 3, "n_sub_cp": 8, "micro": {}}

out0 = simulate(geo, base, material="316L", params=sp, asbuilt_solver="buildup",
                micro_solver="surrogate", mbd_solver="surrogate")
th = out0["thermal"]
print("=== 基准工艺前向场 ===")
print("peak        ", fin(th.peak_temperature))
print("cooling_rate", fin(th.cooling_rate))
print("thermal_grad", fin(th.thermal_gradient))
print("solidif_rate", fin(th.solidification_rate))
try:
    print("gr_ratio    ", fin(th.gr_ratio()))
except Exception as exc:
    print("gr_ratio    RAISED", type(exc).__name__, exc)
print("cost_base   ", float(_smooth_defect_cost(th, mat)))

# ---- 复现 monitor_correct_inverse 的循环，逐步打印 ----
nl = int(jnp.atleast_1d(base.laser_power).shape[0])
modality = base.modality
sim_p = dict(sp)
sim_p.setdefault("n_grid", 6)
sim_p.setdefault("max_layers", 3)
sim_p.setdefault("n_sub_cp", 8)
sim_p["micro"] = dict(sim_p.get("micro", {}))
sim_p = thermal_tier(sim_p, geo, base, material="316L")
bnds = tier_bounds(sim_p)
print("\n=== 档位钉住的静态调度 ===")
print({k: sim_p[k] for k in sorted(sim_p) if k in
       ("thermal_solver", "tier", "n_steps", "dt", "cfl", "dx", "voxel")})
print("bounds keys ", sorted(bnds) if hasattr(bnds, "keys") else bnds)

z = project_process_feasible(normalize_process(base, bounds=bnds), material="316L",
                             n_layers=nl, modality=modality, params=sim_p)
opt = optax.adam(0.08)
st = opt.init(z)


def loss_of_z(zz):
    plan = denormalize_process(zz, n_layers=nl, modality=modality, bounds=bnds)
    o = simulate(geo, plan, material="316L", params=sim_p,
                 asbuilt_solver="buildup", micro_solver="surrogate",
                 mbd_solver="surrogate")
    return _smooth_defect_cost(o["thermal"], mat), o["thermal"]


lg = jax.value_and_grad(loss_of_z, has_aux=True)
print("\n=== Adam 10 步逐站 ===")
for it in range(10):
    (loss, th_), grads = lg(z)
    upd, st = opt.update(grads, st)
    z_new = optax.apply_updates(z, upd)
    z_new = jnp.clip(z_new, 0.0, 1.0)
    z_new = project_process_feasible(z_new, material="316L", n_layers=nl,
                                     modality=modality, params=sim_p)
    pl = denormalize_process(z_new, n_layers=nl, modality=modality, bounds=bnds)
    print(f"[{it}] loss={float(loss):.6e} grad_finite={bool(np.isfinite(np.asarray(grads, float)).all())} "
          f"z_finite={bool(np.isfinite(np.asarray(z_new, float)).all())} "
          f"P={float(jnp.mean(pl.laser_power)):.1f}W v={float(jnp.mean(pl.scan_speed)):.4f} "
          f"h={float(jnp.mean(pl.hatch_spacing)):.3e} "
          f"peak_finite={bool(np.isfinite(np.asarray(th_.peak_temperature, float)).all())}")
    z = z_new

corr = denormalize_process(z, n_layers=nl, modality=modality, bounds=bnds)
out1 = simulate(geo, corr, material="316L", params=sp, asbuilt_solver="buildup",
                micro_solver="surrogate", mbd_solver="surrogate")
th1 = out1["thermal"]
print("\n=== 修正后工艺（用测试里的原始 sp 重跑，与 _smooth_cost_of 同口径）===")
print("P/V/H       ", float(jnp.mean(corr.laser_power)), float(jnp.mean(corr.scan_speed)),
      float(jnp.mean(corr.hatch_spacing)))
print("peak        ", fin(th1.peak_temperature))
print("cooling_rate", fin(th1.cooling_rate))
print("solidif_rate", fin(th1.solidification_rate))
try:
    print("gr_ratio    ", fin(th1.gr_ratio()))
except Exception as exc:
    print("gr_ratio    RAISED", type(exc).__name__, exc)
print("cost_corr   ", float(_smooth_defect_cost(th1, mat)))
