"""A1 决定性证据：熔化是否发生，是不是 (dx, n_steps) 的函数而不是物理的函数。

2 mm 立方、同一 heuristic_plan（P/v/r/hatch/layer_t 全固定），扫 dx × n_steps。
"""
import time
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from amforge import geometry as G
from amforge import thermal_enthalpy as TE
from amforge.process import heuristic_plan
from amforge.materials import get_material

m = get_material("316L")
T_sol, T_liq = float(m.T_solidus), float(m.T_liquidus)
print(f"316L: T_solidus={T_sol:.1f} K  T_liquidus={T_liq:.1f} K", flush=True)
print(f"{'dx_um':>6} {'nvox':>8} {'n_steps':>8} {'r_eff/dx':>9} {'peak_T':>9} "
      f"{'melted?':>8} {'melt mm3':>9} {'wall s':>7}", flush=True)

for dx_um in (100.0, 50.0, 25.0):
    dx = dx_um * 1e-6
    geo = G.from_sdf_fn(lambda x: jnp.max(jnp.abs(x), axis=-1) - 1.0e-3,
                        bounds=[(-1.0e-3, 1.0e-3)] * 3, spacing=dx, name="sweep")
    plan = heuristic_plan(geo, material="316L", modality="SLM")
    nvox = int(jnp.prod(jnp.asarray(geo.shape)))
    for n_steps in (80, 320):
        t = time.perf_counter()
        out = TE.solve_enthalpy_thermal(geometry=geo, process=plan,
                                       params=dict(n_steps=n_steps))
        jax.block_until_ready(out)
        w = time.perf_counter() - t
        pk = jnp.asarray(out.peak_temperature)
        nm = int(jnp.count_nonzero(pk > T_sol))
        nl = int(jnp.count_nonzero(pk > T_liq))
        print(f"{dx_um:6.0f} {nvox:8d} {n_steps:8d} {50.0/dx_um:9.2f} "
              f"{float(jnp.max(pk)):9.1f} {('T_liq '+str(nl)+'/T_sol '+str(nm)):>8} "
              f"{nm*dx**3*1e9:9.4f} {w:7.2f}", flush=True)
