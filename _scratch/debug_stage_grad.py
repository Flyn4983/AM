"""逐阶段定位梯度 NaN 来源（修正：v 真正驱动工艺）。"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import amforge as af
from amforge import geometry as G
from amforge.meltpool import solve_meltpool_surrogate
from amforge.thermal import solve_thermal_history
from amforge.micro import solve_microstructure
from amforge.constitutive import solve_constitutive
from amforge.buildup import solve_buildup
from amforge.digitaltwin import solve_digital_twin
from amforge.verdict import solve_verdict
from amforge.process import denormalize_process, PROCESS_BOUNDS

part = G.from_sdf_fn(
    lambda x: jnp.linalg.norm(x, axis=-1) - 0.45e-3,
    bounds=[(-0.6e-3, 0.6e-3)] * 3, spacing=50e-6, name="dbg")

lo, hi = PROCESS_BOUNDS["scan_speed"]
nL = int(part.layer_count(40e-6))

def plan_from_v(v):
    z = jnp.zeros(len(PROCESS_BOUNDS))
    z = z.at[0].set(0.5)                                   # 功率居中
    z = z.at[1].set((jnp.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo)))
    return denormalize_process(z, n_layers=nL)

def test_grad(name, fn):
    try:
        g = jax.grad(fn)(1.0)
        gf = jnp.asarray(g)
        print(f"  {name:30s} has_nan={bool(jnp.any(jnp.isnan(gf)))} "
              f"max|g|={float(jnp.max(jnp.abs(gf))):.3e}")
    except Exception as e:
        print(f"  {name:30s} EXC: {type(e).__name__}: {str(e)[:80]}")

def f_mp(v):
    return solve_meltpool_surrogate(geometry=part, process=plan_from_v(v),
            params={"material": "316L", "n_grid": 20}).depth
def f_th(v):
    mp = solve_meltpool_surrogate(geometry=part, process=plan_from_v(v),
            params={"material": "316L", "n_grid": 20})
    return jnp.mean(solve_thermal_history(geometry=part, process=plan_from_v(v),
            meltpool=mp, params={"material": "316L"}).cooling_rate)
def f_mi(v):
    mp = solve_meltpool_surrogate(geometry=part, process=plan_from_v(v),
            params={"material": "316L", "n_grid": 20})
    th = solve_thermal_history(geometry=part, process=plan_from_v(v), meltpool=mp,
            params={"material": "316L"})
    return jnp.mean(solve_microstructure(thermal=th, meltpool=mp, params={}).grain_size)
def f_co(v):
    mp = solve_meltpool_surrogate(geometry=part, process=plan_from_v(v),
            params={"material": "316L", "n_grid": 20})
    th = solve_thermal_history(geometry=part, process=plan_from_v(v), meltpool=mp,
            params={"material": "316L"})
    mi = solve_microstructure(thermal=th, meltpool=mp, params={})
    return jnp.mean(solve_constitutive(microstructure=mi,
            params={"material": "316L"}).sigma_y0)
def f_bu(v):
    mp = solve_meltpool_surrogate(geometry=part, process=plan_from_v(v),
            params={"material": "316L", "n_grid": 20})
    th = solve_thermal_history(geometry=part, process=plan_from_v(v), meltpool=mp,
            params={"material": "316L"})
    mi = solve_microstructure(thermal=th, meltpool=mp, params={})
    co = solve_constitutive(microstructure=mi, params={"material": "316L"})
    return jnp.mean(solve_buildup(geometry=part, process=plan_from_v(v), thermal=th,
            microstructure=mi, params={"material": "316L"}).von_mises_residual())
def f_dt(v):
    mp = solve_meltpool_surrogate(geometry=part, process=plan_from_v(v),
            params={"material": "316L", "n_grid": 20})
    th = solve_thermal_history(geometry=part, process=plan_from_v(v), meltpool=mp,
            params={"material": "316L"})
    mi = solve_microstructure(thermal=th, meltpool=mp, params={})
    co = solve_constitutive(microstructure=mi, params={"material": "316L"})
    bu = solve_buildup(geometry=part, process=plan_from_v(v), thermal=th,
            microstructure=mi, params={"material": "316L"})
    return jnp.max(solve_digital_twin(asbuilt=bu, constitutive=co,
            params={"service_stress": 150e6}).von_mises)
def f_ve(v):
    mp = solve_meltpool_surrogate(geometry=part, process=plan_from_v(v),
            params={"material": "316L", "n_grid": 20})
    th = solve_thermal_history(geometry=part, process=plan_from_v(v), meltpool=mp,
            params={"material": "316L"})
    mi = solve_microstructure(thermal=th, meltpool=mp, params={})
    co = solve_constitutive(microstructure=mi, params={"material": "316L"})
    bu = solve_buildup(geometry=part, process=plan_from_v(v), thermal=th,
            microstructure=mi, params={"material": "316L"})
    dt = solve_digital_twin(asbuilt=bu, constitutive=co,
            params={"service_stress": 150e6})
    return solve_verdict(structural=dt, asbuilt=bu, constitutive=co,
            params={}).strength_safety_factor

for nm, fn in [("meltpool.depth", f_mp), ("thermal.coolrate", f_th),
               ("micro.grain", f_mi), ("constitutive.sy0", f_co),
               ("buildup.resid", f_bu), ("digitaltwin.vm", f_dt),
               ("verdict.safety", f_ve)]:
    test_grad(nm, fn)
