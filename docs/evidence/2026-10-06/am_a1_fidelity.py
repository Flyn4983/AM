"""A1 立项证据：thermal.enthalpy 的移动热源分辨率是否收敛。

三件事：
1. 固定几何/工艺，扫 n_steps 看 peak_T 与熔池体积是否收敛（不收敛=A1 必须做）。
2. 打印激光每步位移 vs 光斑半径（teleport 比）。
3. 打印代码里的拓扑上限（n_layers/n_lines/n_per_line 的 clip）对真实构建的截断。
CPU-only。
"""
import math
import time
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from amforge import geometry as G
from amforge import thermal_enthalpy as TE
from amforge.process import heuristic_plan
from amforge.materials import get_material

geo = G.from_sdf_fn(lambda x: jnp.max(jnp.abs(x), axis=-1) - 1.0e-3,
                    bounds=[(-1.0e-3, 1.0e-3)] * 3, spacing=100e-6, name="a1")
plan = heuristic_plan(geo, material="316L", modality="SLM")
mat = get_material("316L")
T_sol = float(getattr(mat, "T_solidus", 1673.0))
print(f"geometry shape={tuple(geo.shape)} dim={geo.dim} spacing_m={float(geo.spacing):.3e}",
      flush=True)
def sc(v):
    return float(jnp.asarray(v).reshape(-1)[0])
print(f"process: v={sc(plan.scan_speed):.3f} m/s  hatch={sc(plan.hatch_spacing):.3e} m  "
      f"layer_t={sc(plan.layer_thickness):.3e} m  power={sc(plan.laser_power):.2f} W  "
      f"spot={sc(plan.beam_radius):.3e} m  n_per_layer_arrays={jnp.asarray(plan.laser_power).shape}",
      flush=True)
print(f"\n{'n_steps':>8} {'wall s':>8} {'peak_T':>10} {'melt vox':>10} {'melt vol mm3':>13}",
      flush=True)
ref = None
for n in (20, 80, 320, 1280):
    t = time.perf_counter()
    out = TE.solve_enthalpy_thermal(geometry=geo, process=plan, params=dict(n_steps=n))
    jax.block_until_ready(out)
    dt = time.perf_counter() - t
    pk = jnp.asarray(out.peak_temperature)
    melt = int(jnp.count_nonzero(pk > T_sol))
    mv = melt * float(geo.spacing) ** 3 * 1e9
    if n == 80:
        ref = (float(jnp.max(pk)), mv)
    print(f"{n:8d} {dt:8.2f} {float(jnp.max(pk)):10.2f} {melt:10d} {mv:13.4f}", flush=True)

print("\n--- 相对 n_steps=80 的漂移 ---")
print(f"(见上表：若 20→1280 的 peak_T / 熔池体积明显变化，说明时间-路径离散未收敛)", flush=True)

# teleport 比：激光每步位移 vs 光斑半径
for label, x_ext_m, n_steps in [("本探针 (2mm)", 2.0e-3, 80),
                                 ("试片单道 (5mm)", 5.0e-3, 80),
                                 ("零件 (10mm)", 10.0e-3, 80)]:
    step_m = x_ext_m / min(64, n_steps)          # xs_full 固定 64 点缓冲
    spot = sc(plan.beam_radius)
    print(f"{label:18s} 每步位移={step_m*1e6:8.1f} um  光斑 r={spot*1e6:5.1f} um  "
          f"teleport={step_m/spot:6.1f}x", flush=True)

print("\n--- 代码拓扑上限（thermal_enthalpy.py:161/169/170）对真实构建的截断 ---")
for side_mm, layer_t in [(2.0, 30e-6), (10.0, 30e-6), (100.0, 30e-6)]:
    z_ext = side_mm * 1e-3
    need_layers = z_ext / layer_t
    hatch = sc(plan.hatch_spacing)
    need_lines = side_mm * 1e-3 / hatch
    print(f"边长 {side_mm:6.1f} mm: 需要层数={need_layers:8.0f} (clip→8)  "
          f"需要线数={need_lines:8.0f} (clip→16)  每线需要采样点="
          f"{side_mm*1e-3/(2*sc(plan.beam_radius)):8.0f} (clip→64)", flush=True)
