"""A0 J3 的**时间采样归因**探针（CPU only，CUDA_VISIBLE_DEVICES=""）。

问题：登记判据（100/50/25 三档 peak/Lx/vol 相对变化 <5%）里 Lx 极差 13.04%、
vol 极差 35.65% 双双 FAIL（/tmp/am_a0_final3.log）。粗档同时**空间**欠分辨
（dx=r，仅 2 个体素跨光斑）与**时间**欠采样（CFL 自动步数下每步激光走
6.5625e-3/66·0.8 ≈ 99µm ≈ 1.0r）。二者必须在同一 probe 里分开，否则 FAIL 的
归因写不出来。

做法：固定网格 dx，只把 n_steps 乘 2/4（dt 低于 CFL 目标 → 纯时间加密），看形态
是否向细档靠拢。

跑前登记的证伪判据（先写死，跑后打分）：
  P7 dx=100 档时间加密**只能部分**恢复形态：mult=4 时 Ly 落在 0.35~0.45mm、
     vol 落在 0.055~0.065mm³（即向 dx=50 的 0.500/0.0687 靠拢但**不超过**）。
     若 mult=4 直接把 Ly 拉到 ≥0.500 → 说明缺陷主因是时间而非空间，
     那么「dx=r 不可信」的警告归因错误，必须改写。
  P8 dx=50 档在 CFL 步数（每步 5µm=r/20）已把时间离散做足：mult=2/4 相对 mult=1
     的峰值变化 <1%、vol 变化 <2%。若超过 → CFL 自动步数不足以做时间分辨，
     suggest_n_steps 的 cfl=0.35 档位需要重新标定。
"""
import time

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from amforge import geometry as G
from amforge.core.contracts import ProcessPlan
from amforge.thermal_enthalpy import solve_enthalpy_thermal, suggest_n_steps

BX, BY, BZ = 1.2e-3, 0.6e-3, 0.4e-3
V_SCAN, ETA, R = 0.8, 0.45, 100e-6
T_LIQ = 1723.0


def coupon(dx):
    return G.from_sdf_fn(
        lambda x: jnp.maximum(jnp.maximum(jnp.abs(x[..., 0]) - BX / 2,
                                          jnp.abs(x[..., 1]) - BY / 2),
                              jnp.abs(x[..., 2]) - BZ / 2),
        bounds=[(-BX / 2, BX / 2), (-BY / 2, BY / 2), (-BZ / 2, BZ / 2)],
        spacing=dx, name="coupon")


def plan(power=600.0, radius=R):
    return ProcessPlan.uniform(1, modality="SLM", laser_power=power,
                               scan_speed=V_SCAN, layer_thickness=BZ,
                               hatch_spacing=1.4 * radius, beam_radius=radius,
                               absorption=ETA, preheat_temp=400.0)


def run(dx_um, mult):
    dx = dx_um * 1e-6
    g, pl = coupon(dx), plan()
    ns1 = suggest_n_steps(g, pl)
    ns = ns1 * mult
    t0 = time.perf_counter()
    th = solve_enthalpy_thermal(geometry=g, process=pl,
                                params=dict(material="316L", cfl=0.35,
                                            n_steps=ns))
    el = time.perf_counter() - t0
    solid = g.sdf < 0
    melted = (th.peak_temperature > T_LIQ) & solid
    c = g.coords()
    Lx = float(jnp.max(c[..., 0], where=melted, initial=0.0)
               - jnp.min(c[..., 0], where=melted, initial=0.0)) * 1e3
    Ly = float(jnp.max(c[..., 1], where=melted, initial=0.0)
               - jnp.min(c[..., 1], where=melted, initial=0.0)) * 1e3
    Lz = float(jnp.max(c[..., 2], where=melted, initial=0.0)
               - jnp.min(c[..., 2], where=melted, initial=0.0)) * 1e3
    vol = float(jnp.sum(melted)) * dx ** 3 * 1e9
    pk = float(jnp.max(th.peak_temperature))
    step_um = 5.25e-3 / ns * V_SCAN * 1e6
    print(f"  dx={dx_um:6.1f} mult={mult} ns={ns:5d} 步长={step_um:6.2f}µm"
          f"={step_um/(R*1e6):.2f}r {el:6.2f}s | peak={pk:8.1f}K"
          f" nml={int(jnp.sum(melted)):6d} Lx={Lx:6.3f} Ly={Ly:6.3f}"
          f" Lz={Lz:6.3f} vol={vol:7.4f}mm³", flush=True)
    return dict(dx=dx_um, mult=mult, ns=ns, step=step_um, peak=pk, vol=vol,
                Ly=Ly, Lx=Lx, Lz=Lz)


REF = {(50., 1): dict(peak=2670.3, Ly=0.500, vol=0.0687),
       (25., 1): dict(peak=2617.8, Ly=0.550, vol=0.0715)}

print("=== P7：dx=100 档只加密时间，形态能恢复多少？===", flush=True)
a = [run(100., m) for m in (1, 2, 4)]
print(f"  Ly {a[0]['Ly']:.3f}→{a[1]['Ly']:.3f}→{a[2]['Ly']:.3f} (dx=50 CFL 档"
      f"={REF[(50.,1)]['Ly']:.3f}) | vol {a[0]['vol']:.4f}→{a[1]['vol']:.4f}"
      f"→{a[2]['vol']:.4f} (dx=50={REF[(50.,1)]['vol']:.4f})", flush=True)
p7 = (0.35 <= a[2]["Ly"] <= 0.45) and (0.055 <= a[2]["vol"] <= 0.065)
print(f"  P7（部分恢复且不超 dx=50）=> {'PASS' if p7 else 'FAIL'}", flush=True)

print("\n=== P8：dx=50 档 CFL 步数是否已做足时间离散？===", flush=True)
b = [run(50., m) for m in (1, 2, 4)]
for r, m in zip(b, (1, 2, 4)):
    if m > 1:
        print(f"    mult={m}: Δpeak={abs(r['peak']-b[0]['peak'])/b[0]['peak']:.3%} "
              f"Δvol={abs(r['vol']-b[0]['vol'])/b[0]['vol']:.3%}", flush=True)
d_pk = max(abs(r["peak"] - b[0]["peak"]) / b[0]["peak"] for r in b[1:])
d_vl = max(abs(r["vol"] - b[0]["vol"]) / b[0]["vol"] for r in b[1:])
p8 = d_pk < 0.01 and d_vl < 0.02
print(f"  P8（Δpeak<1% 且 Δvol<2%）=> {'PASS' if p8 else 'FAIL'}", flush=True)

print("\n=== 参照：dx=25 CFL 档 + 其 mult=2（时间再翻倍）===", flush=True)
c = [run(25., 1), run(25., 2)]
print(f"  dx=25 Δpeak(1→2)={abs(c[1]['peak']-c[0]['peak'])/c[0]['peak']:.3%} "
      f"Δvol={abs(c[1]['vol']-c[0]['vol'])/c[0]['vol']:.3%}", flush=True)

print("\n=== 打分 ===", flush=True)
print(f"  P7 {'PASS' if p7 else 'FAIL'}  P8 {'PASS' if p8 else 'FAIL'}", flush=True)
print("完成。", flush=True)
