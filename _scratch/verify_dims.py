"""测 n_quad 对最终交付量（熔池宽/深/长、峰值温度、缺陷分数）的收敛性。"""
import sys
sys.path.insert(0, "src")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from amforge.geometry import gyroid, from_sdf_fn
from amforge.process import constant_plan
from amforge.meltpool import solve_meltpool_surrogate

part = from_sdf_fn(gyroid(cell=6e-4, thickness=1.2e-4),
                   bounds=((0, 1.2e-3), (0, 1.2e-3), (0, 6e-4)),
                   spacing=6e-5, name="gyroid")

CASES = [
    ("Ti6Al4V", 220.0, 0.9, 50e-6, 30e-6),
    ("316L", 195.0, 0.8, 45e-6, 30e-6),
    ("AlSi10Mg", 370.0, 1.3, 40e-6, 30e-6),
]

for mat, P, v, rb, lt in CASES:
    print("=" * 84)
    print(f"{mat}  P={P}W  v={v}m/s  r_b={rb*1e6:.0f}µm")
    print(f"{'n_quad':>7} {'width µm':>10} {'depth µm':>10} {'length µm':>10} "
          f"{'T_peak K':>10} {'lof':>8} {'poros':>8}")
    ref = None
    for nq in (16, 24, 32, 48, 64, 128, 256):
        plan = constant_plan(params=dict(
            laser_power=P, scan_speed=v, beam_radius=rb,
            layer_thickness=lt, n_layers=8))
        r = solve_meltpool_surrogate(
            geometry=part, process=plan,
            params={"material": mat, "n_grid": 32, "n_quad": nq, "props": "mean"})
        row = (float(r.width) * 1e6, float(r.depth) * 1e6,
               float(r.length) * 1e6, float(jnp.max(r.temperature)),
               float(r.lof_indicator), float(r.porosity_indicator))
        if ref is None and nq == 16:
            pass
        print(f"{nq:>7} {row[0]:>10.3f} {row[1]:>10.3f} {row[2]:>10.3f} "
              f"{row[3]:>10.2f} {row[4]:>8.4f} {row[5]:>8.4f}")
        if nq == 256:
            ref = row
    # 相对 256 点的偏差
    print(f"  参照 n_quad=256 的相对偏差：")
    for nq in (16, 24, 32, 48, 64, 128):
        plan = constant_plan(params=dict(
            laser_power=P, scan_speed=v, beam_radius=rb,
            layer_thickness=lt, n_layers=8))
        r = solve_meltpool_surrogate(
            geometry=part, process=plan,
            params={"material": mat, "n_grid": 32, "n_quad": nq, "props": "mean"})
        row = (float(r.width) * 1e6, float(r.depth) * 1e6,
               float(r.length) * 1e6, float(jnp.max(r.temperature)))
        rel = [abs(row[i] / ref[i] - 1) for i in range(4)]
        print(f"    n_quad={nq:>4}: w {rel[0]:.2e}  d {rel[1]:.2e}  "
              f"l {rel[2]:.2e}  T {rel[3]:.2e}")
    print()
