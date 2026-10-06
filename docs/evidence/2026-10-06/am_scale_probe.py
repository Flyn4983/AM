"""Measure cost growth of the alpha path (forward + one jax.grad step wrt process z).

CPU-only. Purpose: quantify whether an outer process-correction loop can ever run on
the layer-activation plastic FEM forward model at industrial part scale.
"""
import time
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import amforge as af
from amforge import geometry as G
from amforge import inverse as inv
from amforge.process import heuristic_plan, denormalize_process, normalize_process

_W = dict(geom=5.0, stress=1.0, strain=0.5, defect=1.0)


def build(spacing):
    geo = G.from_sdf_fn(
        lambda x: jnp.linalg.norm(x, axis=-1) - 0.4e-3,
        bounds=[(-0.5e-3, 0.5e-3)] * 3, spacing=spacing, name="probe")
    plan = heuristic_plan(geo, material="316L", modality="SLM")
    return geo, plan


def make_loss(geo, material, params):
    z0 = normalize_process(heuristic_plan(geo, material=material, modality="SLM"))

    def loss(z):
        plan = denormalize_process(z, n_layers=int(geo.layer_count(40e-6)),
                                   modality="SLM")
        out = inv.simulate(geo, plan, material=material, params=params,
                           asbuilt_solver="plastic", constitutive="j2")
        return inv._dimensional_loss(out, geo, material, weights=_W)
    return z0, loss


print(f"{'spacing':>9} {'nvox':>5} {'layers':>7} {'fwd s':>9} {'grad s':>9} {'loss':>12}",
      flush=True)
for spacing, max_layers in [(240e-6, 2), (120e-6, 4), (80e-6, 6)]:
    geo, plan = build(spacing)
    nvox = int(geo.shape[0])
    params = dict(n_grid=8, max_layers=max_layers, n_sub_cp=8, tau_activation=1e-3)
    z0, loss = make_loss(geo, "316L", params)
    t = time.perf_counter(); L = float(loss(z0)); tf = time.perf_counter() - t
    t = time.perf_counter(); g = jax.grad(loss)(z0); tg = time.perf_counter() - t
    print(f"{spacing*1e6:8.0f}um {nvox:5d} {max_layers:7d} {tf:9.2f} {tg:9.2f} "
          f"{float(L):12.5f}  |grad|={float(jnp.linalg.norm(g)):.3e}", flush=True)
