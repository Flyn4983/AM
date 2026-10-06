"""Step-by-step VOF debug: where does it blow up?"""
import sys, faulthandler
faulthandler.enable()
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import numpy as np
import amforge as af
from amforge import meltpool as MP
from amforge.materials import get_material

CLAMP = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
NSTEP = int(sys.argv[2]) if len(sys.argv) > 2 else 400

mat = get_material("316L")
cfg = MP.MeltPoolConfig(nx=20, ny=14, nz=14, n_steps=NSTEP,
                        n_pressure_iters=40, dt_max=1e-6,
                        velocity_clamp=CLAMP)
proc = af.ProcessPlan.uniform(laser_power=400.0, scan_speed=0.12,
                              layer_thickness=40e-6, hatch_spacing=80e-6,
                              beam_radius=50e-6, absorption=0.5)
traj = MP.zigzag_trajectory(speed=proc.scan_speed,
                            track_length=min(cfg.track_length, 0.9 * cfg.nx * cfg.dx),
                            n_tracks=cfg.n_tracks, hatch=proc.hatch_spacing,
                            center_x=0.5 * cfg.nx * cfg.dx, center_y=0.5 * cfg.ny * cfg.dx)
st = MP.build_initial_state(cfg, mat, proc)
print(f"clamp={CLAMP}  nsteps={NSTEP}  n_pressure_iters={cfg.n_pressure_iters}")

vmax = 0.0
for i in range(NSTEP):
    st = MP._step(st, cfg, mat, proc, traj)
    vmax = float(jnp.sqrt(jnp.max(jnp.sum(st.u ** 2, axis=-1))))
    if i % 25 == 0 or i < 3 or i == NSTEP - 1:
        pmax = float(jnp.max(jnp.abs(st.p)))
        dv = jnp.abs(MP._div_face(st.u, cfg.dx))
        # 只看"块体液相内部"（远离自由界面与域边界）的散度：软掩膜过渡区
        # 本身会引入 O(vmax/dx) 的表观散度，不代表不可压条件被破坏
        core = (st.F > 0.98).astype(jnp.float64)
        for c in range(3):
            core = core * (MP._shift(st.F, 1, c) > 0.98) * (MP._shift(st.F, -1, c) > 0.98)
        core = core.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)
        dfac = float(jnp.max(dv))
        dcore = float(jnp.max(dv * core))
        tpk = float(jnp.max(st.T_peak))
        fl = float(jnp.max(mat.liquid_fraction(st.T)))
        print(f"step {i:4d} t={float(st.time):.3e} Tpk={tpk:7.0f} flmax={fl:.3f} "
              f"vmax={vmax:.4e} pmax={pmax:.4e} div_all={dfac:.3e} div_core={dcore:.3e} "
              f"kh={float(st.keyhole_max)*1e6:6.2f}um")
    if not bool(jnp.all(jnp.isfinite(st.u))) or vmax > 1e5:
        print(f"!!! BLEW UP at step {i}  vmax={vmax:.3e}")
        break
print("DEBUG DONE")
