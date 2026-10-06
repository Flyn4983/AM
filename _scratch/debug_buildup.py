"""bisect buildup 梯度 NaN。"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from amforge import geometry as G
from amforge.meltpool import solve_meltpool_surrogate
from amforge.thermal import solve_thermal_history
from amforge.micro import solve_microstructure
from amforge.buildup import solve_buildup
from amforge.process import denormalize_process, PROCESS_BOUNDS

part = G.from_sdf_fn(lambda x: jnp.linalg.norm(x, axis=-1) - 0.45e-3,
                     bounds=[(-0.6e-3, 0.6e-3)] * 3, spacing=50e-6)
lo, hi = PROCESS_BOUNDS["scan_speed"]
nL = int(part.layer_count(40e-6))

def plan(v):
    z = jnp.zeros(len(PROCESS_BOUNDS))
    z = z.at[0].set(0.5)
    z = z.at[1].set((jnp.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo)))
    return denormalize_process(z, n_layers=nL)

def base(v):
    p = plan(v)
    mp = solve_meltpool_surrogate(geometry=part, process=p,
            params={"material": "316L", "n_grid": 20})
    th = solve_thermal_history(geometry=part, process=p, meltpool=mp,
            params={"material": "316L"})
    mi = solve_microstructure(thermal=th, meltpool=mp, params={})
    return p, mp, th, mi

for out_name, pick in [
    ("sdf_def", lambda bu: jnp.max(jnp.abs(bu.sdf))),
    ("displacement", lambda bu: jnp.max(jnp.abs(bu.displacement))),
    ("resid_stress_szz", lambda bu: jnp.max(jnp.abs(bu.residual_stress[..., 2]))),
    ("resid_strain", lambda bu: jnp.max(jnp.abs(bu.residual_strain))),
    ("von_mises_residual", lambda bu: jnp.mean(bu.von_mises_residual())),
]:
    def fn(v):
        p, mp, th, mi = base(v)
        bu = solve_buildup(geometry=part, process=p, thermal=th, microstructure=mi,
                          params={"material": "316L"})
        return pick(bu)
    g = jax.grad(fn)(1.0)
    gf = jnp.asarray(g)
    print(f"  {out_name:20s} has_nan={bool(jnp.any(jnp.isnan(gf)))} "
          f"max|g|={float(jnp.max(jnp.abs(gf))):.3e}")
