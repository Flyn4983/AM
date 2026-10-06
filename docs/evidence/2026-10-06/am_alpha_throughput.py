"""Measure alpha-path thermal (enthalpy FVM) throughput on CPU, to size the GPU gap.

CPU-only: launched with CUDA_VISIBLE_DEVICES="". Reports voxel-steps/s so the
part-scale requirement can be divided out without guessing.
"""
import time
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from amforge import geometry as G
from amforge import thermal_enthalpy as TE
from amforge.process import heuristic_plan

print(f"{'side_mm':>8} {'dx_um':>6} {'n_axis':>7} {'nvox':>10} {'steps':>6} "
      f"{'wall s':>8} {'Mvoxstep/s':>11}", flush=True)

for side_mm, dx_um, n_steps in [(2.0, 100.0, 40), (2.0, 50.0, 40),
                                (4.0, 50.0, 40), (4.0, 25.0, 40)]:
    side = side_mm * 1e-3
    dx = dx_um * 1e-6
    half = side / 2.0
    geo = G.from_sdf_fn(
        lambda x: jnp.max(jnp.abs(x), axis=-1) - half,
        bounds=[(-side, side)] * 3, spacing=dx, name="tp")
    n_axis = int(geo.shape[0])
    nvox = int(jnp.prod(jnp.asarray(geo.shape)))
    plan = heuristic_plan(geo, material="316L", modality="SLM")
    params = dict(n_steps=n_steps)
    out = TE.solve_enthalpy_thermal(geometry=geo, process=plan, params=params)
    jax.block_until_ready(out)
    t = time.perf_counter()
    out = TE.solve_enthalpy_thermal(geometry=geo, process=plan, params=params)
    jax.block_until_ready(out)
    dt = time.perf_counter() - t
    rate = nvox * n_steps / dt / 1e6
    print(f"{side_mm:8.1f} {dx_um:6.0f} {n_axis:7d} {nvox:10d} {n_steps:6d} "
          f"{dt:8.2f} {rate:11.3f}", flush=True)
