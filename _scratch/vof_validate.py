"""B 档 VOF 物理验证：自由界面 / Marangoni / 匙孔 / 可微性。

配置要求（此前 smoke 配置的教训）：
* dx 必须 << beam_radius，且 nx*dx 要装得下几个光斑直径；
* layer_thickness/dx 决定粉层占几层，nz 必须留出**气相**空间，
  否则粉末填满整域 -> 根本没有自由界面 -> VOF 退化。
"""
import time
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import numpy as np
import amforge as af
from amforge import meltpool as MP
from amforge.materials import get_material

mat = get_material("316L")
DX = 5e-6
CFG = dict(nx=44, ny=32, nz=30, dx=DX, n_steps=400, n_pressure_iters=25,
           dt_max=5e-7, interface_compression=1.0, fresnel_boost=1.5,
           substrate_fraction=0.55, powder_layers=1)


def make_proc(power=250.0, speed=0.8, rb=30e-6):
    return af.ProcessPlan.uniform(laser_power=power, scan_speed=speed,
                                  layer_thickness=30e-6, hatch_spacing=60e-6,
                                  beam_radius=rb, absorption=0.4)


def run(power=250.0, speed=0.8, dsig=None, nsteps=None):
    m = mat if dsig is None else mat.replace(dsigma_dT=dsig)
    cfg = MP.MeltPoolConfig(**{**CFG, **({"n_steps": nsteps} if nsteps else {})})
    return MP.simulate_meltpool(process=make_proc(power, speed), material=m, config=cfg)


cfg0 = MP.MeltPoolConfig(**CFG)
k_sub = cfg0.substrate_top_index()
n_layer = max(1, int(round(30e-6 / DX)))
print(f"网格 {cfg0.nx}x{cfg0.ny}x{cfg0.nz} dx={DX*1e6:.1f}um "
      f"域={cfg0.nx*DX*1e6:.0f}x{cfg0.ny*DX*1e6:.0f}x{cfg0.nz*DX*1e6:.0f}um")
print(f"基板 z=0..{k_sub-1}  粉层 z={k_sub}..{k_sub+n_layer-1}  "
      f"气相 z={k_sub+n_layer}..{cfg0.nz-1}  (气相层数={cfg0.nz-k_sub-n_layer})")
assert cfg0.nz - k_sub - n_layer >= 4, "必须留出气相空间，否则没有自由界面"

t0 = time.time()
sol = run()
print(f"\n=== 基准 (250W, 0.8m/s)  [{time.time()-t0:.1f}s] ===")
for nm, a in [("T", sol.state.T), ("u", sol.state.u), ("F", sol.state.F), ("p", sol.state.p)]:
    a = jnp.asarray(a)
    print(f"  {nm:2s} finite={bool(jnp.all(jnp.isfinite(a)))} "
          f"max|.|={float(jnp.max(jnp.abs(a))):.4e} min={float(jnp.min(a)):.4e}")
print(f"  peakT   = {float(jnp.max(sol.state.T_peak)):.0f} K  (T_liq={mat.T_liquidus:.0f})")
print(f"  depth   = {float(sol.depth)*1e6:.2f} um")
print(f"  width   = {float(sol.width)*1e6:.2f} um")
print(f"  length  = {float(sol.length)*1e6:.2f} um")
print(f"  keyhole = {float(sol.state.keyhole_max)*1e6:.2f} um")
print(f"  vmax    = {float(sol.state.vmax_max):.3f} m/s")
print(f"  precoil = {float(sol.state.precoil_max):.3e} Pa")
print(f"  t_end   = {float(sol.state.time):.3e} s "
      f"(扫描距离 {float(sol.state.time)*0.8*1e6:.1f} um)")

# 块体液相内部散度（不可压性真实检验）
dv = jnp.abs(MP._div_face(sol.state.u, DX))
core = (sol.state.F > 0.98).astype(jnp.float64)
for c in range(3):
    core = core * (MP._shift(sol.state.F, 1, c) > 0.98) * (MP._shift(sol.state.F, -1, c) > 0.98)
core = core.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)
vmax = float(jnp.sqrt(jnp.max(jnp.sum(sol.state.u ** 2, -1))))
print(f"  div_core= {float(jnp.max(dv*core)):.3e} 1/s   (vmax/dx={vmax/DX:.3e})"
      f"  相对={float(jnp.max(dv*core))/(vmax/DX+1e-30):.3e}")

print("\n=== 匙孔正反馈：功率扫描 ===")
for P in (120.0, 250.0, 400.0):
    s = run(power=P)
    print(f"  P={P:5.0f}W  peakT={float(jnp.max(s.state.T_peak)):6.0f}K "
          f"depth={float(s.depth)*1e6:6.2f}um keyhole={float(s.state.keyhole_max)*1e6:6.2f}um "
          f"precoil={float(s.state.precoil_max):.2e}Pa vmax={float(s.state.vmax_max):7.2f}m/s")

print("\n=== Marangoni 真实生效性（关掉 dσ/dT 作对照）===")
s_on = run()
s_off = run(dsig=0.0)
for nm, s in (("dσ/dT=真实", s_on), ("dσ/dT=0  ", s_off)):
    w, d = float(s.width) * 1e6, float(s.depth) * 1e6
    print(f"  {nm}  width={w:6.2f}um depth={d:6.2f}um  W/D={w/max(d,1e-9):6.2f} "
          f"vmax={float(s.state.vmax_max):7.2f}m/s")
print(f"  dsigma_dT(316L) = {float(mat.dsigma_dT):.3e} N/(m·K)")

print("\n=== 可微性 ===")
def depth_of(P):
    return run(power=P).depth
def kh_of(P):
    return run(power=P).state.keyhole_max
for nm, f in (("depth", depth_of), ("keyhole", kh_of)):
    t = time.time()
    g = jax.grad(f)(250.0)
    print(f"  d({nm})/dP = {float(g):.4e} m/W  finite={bool(jnp.isfinite(g))} "
          f"[{time.time()-t:.1f}s]")
print("VALIDATE DONE")
