"""Smoke test v2: does meltpool.vof_flow3d run, melt, and stay differentiable?"""
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import numpy as np
import amforge as af
from amforge import meltpool as MP
from amforge.materials import get_material

mat = get_material("316L")
cfg = MP.MeltPoolConfig(nx=20, ny=14, nz=14, n_steps=500,
                        n_pressure_iters=15, dt_max=1e-6,
                        interface_compression=1.0, fresnel_boost=1.5)
proc = af.ProcessPlan.uniform(laser_power=400.0, scan_speed=0.12,
                              layer_thickness=40e-6, hatch_spacing=80e-6,
                              beam_radius=50e-6, absorption=0.5)

sol = MP.simulate_meltpool(process=proc, material=mat, config=cfg)

def stat(name, a):
    a = jnp.asarray(a)
    print(f"  {name:12s} finite={bool(jnp.all(jnp.isfinite(a)))}  "
          f"max|·|={float(jnp.max(jnp.abs(a))):.4e}  min={float(jnp.min(a)):.4e}")
print("=== field finiteness ===")
stat("T[K]", sol.state.T)
stat("u[m/s]", sol.state.u)
stat("F", sol.state.F)
stat("p[Pa]", sol.state.p)
Tpk = float(jnp.max(sol.state.T_peak))
print(f"=== diagnostics ===  peakT={Tpk:.0f}K (T_sol={mat.T_solidus:.0f}, T_liq={mat.T_liquidus:.0f})")
print(f"  depth   = {float(sol.depth)*1e6:.1f} um")
print(f"  width   = {float(sol.width)*1e6:.1f} um")
print(f"  length  = {float(sol.length)*1e6:.1f} um")
print(f"  keyhole = {float(sol.state.keyhole_max)*1e6:.1f} um")
print(f"  vmax    = {float(sol.state.vmax_max):.3f} m/s")
print(f"  precoil = {float(sol.state.precoil_max):.2e} Pa")
print(f"  time    = {float(sol.state.time):.2e} s")
melted = bool(Tpk > mat.T_solidus)
print(f"  MELTED  = {melted}")

print("=== differentiability (grad of depth vs power) ===")
def depth_of(P):
    p2 = af.ProcessPlan.uniform(laser_power=P, scan_speed=0.12,
                                layer_thickness=40e-6, hatch_spacing=80e-6,
                                beam_radius=50e-6, absorption=0.5)
    s = MP.simulate_meltpool(process=p2, material=mat, config=cfg)
    return s.depth
try:
    g = jax.grad(depth_of)(400.0)
    print(f"  d(depth)/dP = {float(g):.3e} m/(W)  finite={bool(jnp.isfinite(g))}")
except Exception as e:
    print(f"  grad FAILED: {type(e).__name__}: {e}")
print("SMOKE DONE")
