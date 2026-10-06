"""Where does the alpha thermal step actually spend its time? CPU-only micro-bench.

Compares the three per-step costs of thermal_enthalpy: the 24-iteration enthalpy
bisection, the variable-coefficient divergence, and a minimal stencil reference.
"""
import time
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
from amforge import thermal_enthalpy as TE

N = 128
shape = (N, N, N)
H = jax.random.uniform(jax.random.PRNGKey(0), shape, dtype=jnp.float64) * 3e8
T = TE.temperature_of_enthalpy(H, rho=8000.0, cp=500.0, L=2.5e6, T_amb=300.0,
                               T_sol=1673.0, T_liq=1723.0)
alpha = TE.effective_diffusivity(T, k=15.0, rho=8000.0, cp=500.0, L=2.5e6,
                                T_sol=1673.0, T_liq=1723.0)
dx = 50e-6


def bench(name, fn, reps=5):
    fn().block_until_ready()
    t = time.perf_counter()
    for _ in range(reps):
        fn().block_until_ready()
    dt = (time.perf_counter() - t) / reps
    nvox = N ** 3
    print(f"{name:34s} {dt * 1e3:8.2f} ms   {nvox / dt / 1e9:6.2f} Gvox/s", flush=True)
    return dt


d_bisect = bench("temperature_of_enthalpy (n_iter=24)",
                 lambda: TE.temperature_of_enthalpy(
                     H, rho=8000.0, cp=500.0, L=2.5e6, T_amb=300.0,
                     T_sol=1673.0, T_liq=1723.0))
for ni in (8, 4, 1):
    bench(f"temperature_of_enthalpy (n_iter={ni})",
          lambda ni=ni: TE.temperature_of_enthalpy(
              H, rho=8000.0, cp=500.0, L=2.5e6, T_amb=300.0,
              T_sol=1673.0, T_liq=1723.0, n_iter=ni))
bench("effective_diffusivity",
      lambda: TE.effective_diffusivity(T, k=15.0, rho=8000.0, cp=500.0, L=2.5e6,
                                      T_sol=1673.0, T_liq=1723.0))
d_div = bench("_div_alpha_grad (take+concat)", lambda: TE._div_alpha_grad(H, alpha, dx))
bench("reference: alpha * laplacian via gradient",
      lambda: jnp.sum(alpha * jnp.gradient(jnp.gradient(H)[0], jnp.gradient(H)[1])[0]))

T2 = TE.temperature_of_enthalpy(H + 1e6, rho=8000.0, cp=500.0, L=2.5e6, T_amb=300.0,
                               T_sol=1673.0, T_liq=1723.0, n_iter=4)
print(f"\nbisection share of one RHS  = {d_bisect / (d_bisect + d_div):.2f}", flush=True)
print(f"SSP-RK2 => 2 RHS + 1 invert per step; per-step bisection count = 2*24+1 = 49",
      flush=True)
