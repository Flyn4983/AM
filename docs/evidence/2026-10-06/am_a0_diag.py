"""A0 诊断：thermal.enthalpy 的总曝光时间 t_total 是不是 dx 与 n_steps 的函数？

代码：dt = min(stability_dt, scan_time)，scan_time = total*dx/(v*n_steps)
  => t_total = n_steps*dt = min(n_steps*stability_dt, total*dx/v)
若 stability_dt 生效，则 t_total ∝ n_steps*dx^2 —— 曝光时长被网格吃掉，
"越加密越冷"与"n_steps 越大越热"都由此产生。本脚本把 regime 直接打印出来。
"""
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
from amforge import geometry as G
from amforge.process import heuristic_plan
from amforge.materials import get_material

m = get_material("316L")
k, rho, cp = float(m.k_solid), float(m.rho_solid), float(m.cp_solid)
alpha0 = k / (rho * cp)
v = 1.0
layer_t, hatch, spot = 40e-6, 102.8e-6, 50e-6
cfl = 0.35
print(f"alpha0={alpha0:.3e} m2/s  k={k} rho={rho} cp={cp}", flush=True)
print(f"{'dx_um':>6} {'n_axis':>7} {'n_layers':>9} {'n_lines':>8} {'n_per_line':>11} "
      f"{'total':>8} {'stab_dt(s)':>11} {'scan_dt(s)':>11} {'regime':>9} "
      f"{'dt(s)':>11} {'t_total(s)':>11}", flush=True)

for dx_um in (100.0, 50.0, 25.0):
    dx = dx_um * 1e-6
    geo = G.from_sdf_fn(lambda x: jnp.max(jnp.abs(x), axis=-1) - 1.0e-3,
                        bounds=[(-1.0e-3, 1.0e-3)] * 3, spacing=dx, name="diag")
    n_axis = int(geo.shape[0])
    x_ext = y_ext = z_ext = (n_axis - 1) * dx
    n_layers = float(jnp.clip(jnp.round(layer_t / z_ext), 1, 8))
    n_lines = float(jnp.clip(jnp.round(y_ext / jnp.maximum(hatch, dx * 1.5)), 1, 16))
    n_per_line = float(jnp.clip(jnp.round(x_ext / (dx * 0.5)), 2, 64))
    total = n_layers * n_lines * n_per_line
    stab = cfl * dx * dx / (2.0 * 3 * alpha0)
    for n_steps in (80, 320, 1280):
        scan = total * dx / (v * n_steps)
        dt = min(stab, scan)
        regime = "stability" if stab < scan else "scan"
        print(f"{dx_um:6.0f} {n_axis:7d} {n_layers:9.0f} {n_lines:8.0f} "
              f"{n_per_line:11.0f} {total:8.0f} {stab:11.3e} {scan:11.3e} "
              f"{regime:>9} {dt:11.3e} {n_steps * dt:11.3e}", flush=True)

print("\n真实物理曝光时长应为 t_path = 路径长/v；本算例 1 层 16 道 x_ext=2mm →", flush=True)
L = 16 * 2.0e-3
print(f"  路径长={L*1e3:.1f} mm → t_path={L/v*1e3:.3f} ms；"
      f"而 dx=25um/n_steps=320 实际只曝光 {320*0.35*(25e-6)**2/(2*3*alpha0)*1e3:.3f} ms",
      flush=True)
