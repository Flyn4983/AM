"""A0 收尾补充证据（两项，都很便宜；跑前登记预测）。

F1 J2 缺口归因（**截断 vs 漏能**）——把逐步功率份额 frac_t 用解析高斯尾重算一遍，
   与算符实测逐一对撞。预测（跑前写定）：
       frac_t = Π_axis ½[erf((edge_hi−pos)/s) − erf((edge_lo−pos)/s)]
       s_xy = r/√2（即横向 exp(−2ρ²/r²)），s_z = dp = 1.2r；
       edge = 体素中心极值 ± dx/2（由该网格实际给出，不硬编码）。
   判据：max_t|frac_实测 − frac_解析| < 1e-4，且 mean 与 J2 的 5.51% 缺口一致。
   ⇒ 命中即证明「全域 Σ 比 ηP·t_exp 少 5.5%」**全部**来自有限域尾（z 向 1.9% +
   行程端点半束），离散算符本身零漏能（J1 已给 1e-14 级守恒）。
   注：首版预测把 z 边界当成 ±BZ/2=200µm（erf=0.98138），但网格外缘实际在
   中心极值+dx/2；本版改用真实网格边缘，属**收紧**而非放宽判据。

F2 dx>2r 护栏：需要 dx>2r=200µm 才触发（第一版误用 150µm 故未命中）。
   预测 dx=250µm 试片 → ValueError，且报错文本含「欠采样」。
"""
import math

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from amforge import geometry as G
from amforge.core.contracts import ProcessPlan
from amforge.thermal_enthalpy import (
    solve_enthalpy_thermal, suggest_n_steps, _build_scan_positions,
    _cell_integrated_source,
)

BX, BY, BZ = 1.2e-3, 0.6e-3, 0.4e-3
V, ETA, P1, R1 = 0.8, 0.45, 600.0, 100e-6


def coupon(dx):
    return G.from_sdf_fn(
        lambda x: jnp.maximum(jnp.maximum(jnp.abs(x[..., 0]) - BX / 2,
                                          jnp.abs(x[..., 1]) - BY / 2),
                              jnp.abs(x[..., 2]) - BZ / 2),
        bounds=[(-BX / 2, BX / 2), (-BY / 2, BY / 2), (-BZ / 2, BZ / 2)],
        spacing=dx, name="coupon")


def plan():
    return ProcessPlan.uniform(1, modality="SLM", laser_power=P1, scan_speed=V,
                               layer_thickness=BZ, hatch_spacing=1.4 * R1,
                               beam_radius=R1, absorption=ETA, preheat_temp=400.0)


dx = 25e-6
g, pl = coupon(dx), plan()
ns = suggest_n_steps(g, pl)
pos, path = _build_scan_positions(g.coords(), pl, jnp.asarray(dx), n_steps=ns,
                                  dim=3, solid=g.sdf < 0.5 * dx)
fr = jnp.asarray([float(jnp.sum(_cell_integrated_source(
    pos[t], g.coords(), ETA * P1, R1, 1.2 * R1, dx))) * dx ** 3 / (ETA * P1)
    for t in range(ns)])
c = g.coords()
edge = [(float(jnp.min(c[..., k])) - dx / 2, float(jnp.max(c[..., k])) + dx / 2)
        for k in range(3)]
s_xy, s_z = R1 / math.sqrt(2.0), 1.2 * R1


def cap(lo, hi, p, s):
    return 0.5 * (math.erf((hi - p) / s) - math.erf((lo - p) / s))


pred = [cap(*edge[0], float(pos[t, 0]), s_xy) * cap(*edge[1], float(pos[t, 1]), s_xy)
        * cap(*edge[2], float(pos[t, 2]), s_z) for t in range(ns)]
pred = jnp.asarray(pred)
err = float(jnp.max(jnp.abs(fr - pred)))
print(f"=== F1 逐步功率份额：算符实测 vs 解析有限域尾 (ns={ns}) ===", flush=True)
print(f"  网格边缘 x={edge[0][0]*1e3:+.4f}/{edge[0][1]*1e3:+.4f}mm "
      f"y={edge[1][0]*1e3:+.4f}/{edge[1][1]*1e3:+.4f}mm "
      f"z={edge[2][0]*1e3:+.4f}/{edge[2][1]*1e3:+.4f}mm", flush=True)
print(f"  max|实测−解析|={err:.3e} => {'PASS' if err < 1e-4 else 'FAIL'}（判据 <1e-4）",
      flush=True)
print(f"  实测 frac: min={float(jnp.min(fr)):.4f} max={float(jnp.max(fr)):.4f} "
      f"mean={float(jnp.mean(fr)):.6f} → 缺口 {1-float(jnp.mean(fr)):.4%}"
      f"（J2 实测 5.5140%）", flush=True)
print(f"  解析 z 向单轴捕获(中段)={cap(*edge[2], 0.0, s_z):.4f}；"
      f"端点步 x 向捕获≈{cap(*edge[0], edge[0][1], s_xy):.4f}（半束）"
      f" → 缺口 = 尾截断，算符零漏能", flush=True)

print("\n=== F2 dx>2r 护栏（dx=250µm, r=100µm）===", flush=True)
try:
    solve_enthalpy_thermal(geometry=coupon(250e-6), process=pl,
                           params={"material": "316L"})
    print("  FAIL：未触发", flush=True)
except ValueError as e:
    print(f"  {'PASS' if '欠' in str(e) else 'FAIL(文本)'}：{str(e)[:110]}", flush=True)

# F3 为「A0 回归测试」取实数：紧凑双道试片（体积小、步数少，适合进 CI）
#    跑前登记预测：dx=50 与 25 两档峰值差 <3%；12.5 相对 25 差 <3%。
print("\n=== F3 紧凑试片（回归夹具候选）0.6×0.3×0.2mm, 2 道 ===", flush=True)
CX, CY, CZ = 0.6e-3, 0.3e-3, 0.2e-3


def coupon_c(dx):
    return G.from_sdf_fn(
        lambda x: jnp.maximum(jnp.maximum(jnp.abs(x[..., 0]) - CX / 2,
                                          jnp.abs(x[..., 1]) - CY / 2),
                              jnp.abs(x[..., 2]) - CZ / 2),
        bounds=[(-CX / 2, CX / 2), (-CY / 2, CY / 2), (-CZ / 2, CZ / 2)],
        spacing=dx, name="coupon_c")


def plan_c(power=600.0, radius=100e-6):
    return ProcessPlan.uniform(1, modality="SLM", laser_power=power, scan_speed=V,
                               layer_thickness=CZ, hatch_spacing=1.4 * radius,
                               beam_radius=radius, absorption=ETA, preheat_temp=400.0)


f3 = []
for d_um in (50., 25., 12.5):
    gg, pp = coupon_c(d_um * 1e-6), plan_c()
    t0 = __import__("time").perf_counter()
    th = solve_enthalpy_thermal(geometry=gg, process=pp, params={"material": "316L"})
    el = __import__("time").perf_counter() - t0
    pk = th.peak_temperature
    mlt = (pk > 1723.0) & (gg.sdf < 0)
    cc = gg.coords()
    Ly = float(jnp.max(cc[..., 1], where=mlt, initial=0.) - jnp.min(cc[..., 1], where=mlt, initial=0.)) * 1e3
    Lz = float(jnp.max(cc[..., 2], where=mlt, initial=0.) - jnp.min(cc[..., 2], where=mlt, initial=0.)) * 1e3
    rec = dict(dx=d_um, ns=suggest_n_steps(gg, pp), peak=float(jnp.max(pk)),
               nml=int(jnp.sum(mlt)), Ly=Ly, Lz=Lz,
               vol=float(jnp.sum(mlt)) * (d_um * 1e-6) ** 3 * 1e9, sec=el)
    f3.append(rec)
    print(f"  dx={d_um:5.1f} ns={rec['ns']:4d} {el:5.2f}s peak={rec['peak']:8.1f}K "
          f"nml={rec['nml']:5d} Ly={Ly:6.3f} Lz={Lz:6.3f} vol={rec['vol']:7.4f}mm³", flush=True)
a, b, cc_ = f3
print(f"  峰值差 50→25={abs(a['peak']-b['peak'])/max(a['peak'],b['peak']):.2%}  "
      f"25→12.5={abs(b['peak']-cc_['peak'])/max(b['peak'],cc_['peak']):.2%}  "
      f"vol 25→12.5={abs(b['vol']-cc_['vol'])/max(b['vol'],cc_['vol']):.2%}", flush=True)
print("\n完成。", flush=True)
