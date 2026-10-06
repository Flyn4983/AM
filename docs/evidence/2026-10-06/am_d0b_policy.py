"""D0 补充探针：resolution_policy 分层 + trace 吞错修复的证伪判据。

判据**跑前**登记（照实记分，预测错了要写清）：
  V1  strict + dx>2r → ValueError；demo + 同一夹具 → UserWarning 且解出有限峰值。
      预测：两条都成立。
  V2  demo 抬半径只影响 dx>2r 的工况：对 2r≥dx 的夹具，strict 与 demo 的峰值
      **逐位相同**（因为 r_src=max(r, dx/2)=r）。预测：max|Δpeak| = 0。
  V3  _is_traced：eager 工艺方案 → False；jax.grad 内的方案 → True。预测成立。
  V4  trace 吞错修复：名义工艺**具体**但档位真实不可行（步数预算覆盖不了曝光）时，
      即使在优化循环里也必须**报错**而不是静默放行。预测：ValueError 命中。
  V5  可微链：外层 eager 钉好档位后，jax.grad 穿过 simulate（焓解）梯度有限非零；
      且钉住的 n_steps ≥ ceil(exposure_bound/dt_rk2)（稳定）——A0 的"trace 内静默
      发散"洞就此关住。预测成立。
"""
import math
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from amforge.core.contracts import PartGeometry, ProcessPlan
from amforge.materials import get_material
from amforge.thermal_enthalpy import (solve_enthalpy_thermal, chain_schedule,
                                      _stability_limit_dx2)
from amforge.inverse import simulate, thermal_tier, _is_traced

MAT = "316L"
mat = get_material(MAT)
alpha0 = float(mat.k_solid) / (float(mat.rho_solid) * mat.cp_solid)
log = []


def geo(n=4, dx=1000e-6):
    sdf = -np.ones((n, n, n), dtype=np.float64)
    return PartGeometry(sdf=sdf, origin=jnp.zeros(3), spacing=dx, dim=3, name="tiny")


def plan(r=50e-6, v=1.0, h=200e-6, lt=200e-6, P=200.0):
    return ProcessPlan.uniform(1, laser_power=P, scan_speed=v, layer_thickness=lt,
                               hatch_spacing=h, beam_radius=r, absorption=0.4)


# --- V1 ---------------------------------------------------------------------
g, p = geo(dx=1000e-6), plan()          # dx=1000µm, 2r=100µm → dx>2r
try:
    solve_enthalpy_thermal(geometry=g, process=p,
                           params={"material": MAT, "n_steps": 40})
    v1a = "NO-RAISE(bug)"
except ValueError as e:
    v1a = f"raised: {str(e)[:52]}"
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    out = solve_enthalpy_thermal(geometry=g, process=p,
                                 params={"material": MAT, "n_steps": 40,
                                         "resolution_policy": "demo"})
    Tpk = jnp.asarray(out.peak_temperature)
    v1b = (f"warned={any(issubclass(x.category, UserWarning) for x in w)} "
           f"peak={float(jnp.max(Tpk)):.1f}K finite={bool(jnp.all(jnp.isfinite(Tpk)))}")
log.append(f"V1 strict: {v1a} | demo: {v1b}")

# --- V2 ---------------------------------------------------------------------
g2 = geo(dx=100e-6)                     # dx=100µm = 2r → 不该被抬
pk = {}
for pol in ("strict", "demo"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o = solve_enthalpy_thermal(geometry=g2, process=plan(),
                                   params={"material": MAT, "n_steps": 40,
                                           "resolution_policy": pol})
    pk[pol] = float(jnp.max(o.peak_temperature))
log.append(f"V2 strict={pk['strict']:.9f} demo={pk['demo']:.9f} "
           f"|Δ|={abs(pk['strict']-pk['demo']):.3e}")

# --- V3 ---------------------------------------------------------------------
trace_flags = [False, False]


def f_of_P(P):
    pp = plan(P=P)
    trace_flags[1] = _is_traced(pp)
    o = simulate(g2, pp, params=PINNED)
    return jnp.sum(jnp.asarray(o["thermal"].peak_temperature) ** 2)


PINNED = thermal_tier({"material": MAT}, g2, plan(), material=MAT)
trace_flags[0] = _is_traced(plan())
grad_val = jax.grad(f_of_P)(200.0)
log.append(f"V3 _is_traced eager={trace_flags[0]} in_grad={trace_flags[1]}")

# --- V4 ---------------------------------------------------------------------
g3 = geo(n=8, dx=120e-6)
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chain_schedule(g3, plan(), material=MAT, max_steps=20, fixed_n_steps=20,
                       resolution_policy="demo")
    v4 = "NO-RAISE(吞错仍在)"
except ValueError as e:
    v4 = f"raised: {str(e)[:60]}"
log.append(f"V4 {v4}")

# --- V5 ---------------------------------------------------------------------
dt_r = float(_stability_limit_dx2(jnp.asarray(100e-6), 3, alpha0, 1.0))
tp = PINNED["thermal"]
need = math.ceil(tp["exposure_bound_s"] / dt_r)
log.append(f"V5 tier={ {k: tp[k] for k in tp} } bounds={PINNED['process_bounds']}")
log.append(f"V5 grad={float(grad_val):.3e} finite={bool(jnp.isfinite(grad_val))} "
           f"n_steps={tp['n_steps']} stable_need>={need} "
           f"hole_closed={tp['n_steps'] >= need}")

print("\n".join(log))
