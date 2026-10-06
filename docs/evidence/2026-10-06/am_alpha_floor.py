"""Floor reference: what should one 3D diffusion step cost vs what ours costs."""
import math
import time
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
from amforge import thermal_enthalpy as TE

N = 128
H = jax.random.uniform(jax.random.PRNGKey(0), (N, N, N), dtype=jnp.float64) * 3e8
a = jnp.full_like(H, 3.75e-6)
dx = 50e-6


def bench(name, fn, reps=5):
    fn().block_until_ready()
    t = time.perf_counter()
    for _ in range(reps):
        fn().block_until_ready()
    dt = (time.perf_counter() - t) / reps
    print(f"{name:38s} {dt * 1e3:8.2f} ms", flush=True)
    return dt


def lap_const(H):
    out = jnp.zeros_like(H)
    for ax in range(3):
        out = out + jnp.roll(H, 1, axis=ax) - 2.0 * H + jnp.roll(H, -1, axis=ax)
    return out / (dx * dx)


def lap_var_face(H, alpha):
    """Same math as _div_alpha_grad but with shifted-slice views instead of take+concat."""
    div = jnp.zeros_like(H)
    for ax in range(3):
        sl_c = [slice(None)] * 3
        sl_l = [slice(None)] * 3
        sl_r = [slice(None)] * 3
        sl_c[ax] = slice(1, -1)
        sl_l[ax] = slice(0, -2)
        sl_r[ax] = slice(2, None)
        af_ = 2.0 / (1.0 / alpha[tuple(sl_l)] + 1.0 / alpha[tuple(sl_c)])
        ar_ = 2.0 / (1.0 / alpha[tuple(sl_c)] + 1.0 / alpha[tuple(sl_r)])
        inner = (ar_ * (H[tuple(sl_r)] - H[tuple(sl_c)])
                 - af_ * (H[tuple(sl_c)] - H[tuple(sl_l)])) / (dx * dx)
        pad = [(0, 0)] * 3
        pad[ax] = (1, 1)
        div = div + jnp.pad(inner, pad)
    return div


d_roll = bench("const-coeff laplacian via roll (floor)", lambda: lap_const(H))
d_div = bench("_div_alpha_grad (current, take+concat)", lambda: TE._div_alpha_grad(H, a, dx))
d_new = bench("same math, shifted-slice + harmonic face alpha", lambda: lap_var_face(H, a))
print(f"\ncurrent / floor                = {d_div / d_roll:5.1f}x", flush=True)
print(f"current / shifted-slice rewrite = {d_div / d_new:5.1f}x", flush=True)
