"""A0 验收探针：能量守恒 + 分辨率收敛（CPU only；GPU 未授权，全程 CUDA_VISIBLE_DEVICES=""）。

预登记判据（开发日志 §24.6 与本轮开工前计划，跑前写定）：
  J1 守恒  单元体积分源全域 Σ frac = 1，相对误差 < 1e-9，且与 dx/r 比值无关。
  J2 剂量  求解器注入总能量 Σ_i,t Q·dV·dt = ηP·t_exposure，相对误差 < 1e-9。
  J3 收敛  dx = 100/50/25 µm 三档：峰值温度与熔池长度相对变化 < 5%，且三档均熔化。
J3 附加归因（同一 probe 内做，不改判据只加证据）：把 dx=100/50 两档的步数乘以 2/4
（dt 低于 CFL），看峰值是否向细档靠拢 —— 区分「空间欠分辨」与「轨迹时间欠采样」。
"""
import math
import time

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from amforge import geometry as G
from amforge.core.contracts import ProcessPlan
from amforge.materials import get_material
from amforge.thermal_enthalpy import (
    solve_enthalpy_thermal, suggest_n_steps, _build_scan_positions,
    _cell_integrated_source, _moving_source, _scan_topology,
    _stability_limit_dx2,
)

mat = get_material("316L")
ALPHA0 = float(mat.k_solid) / (float(mat.rho_solid) * float(mat.cp_solid))
T_LIQ, T_SOL, T_BOIL = float(mat.T_liquidus), float(mat.T_solidus), float(mat.T_boil)
V_SCAN = 0.8
ETA = 0.45
print(f"316L k={mat.k_solid} rho={mat.rho_solid} cp={mat.cp_solid} "
      f"alpha0={ALPHA0:.4e} Tsol={T_SOL} Tliq={T_LIQ} Tboil={T_BOIL} "
      f"latent={mat.latent_fusion}", flush=True)

# ------------------------------------------------------------------ J1
print("\n=== J1 单元体积分源：Σ frac 应恒为 1（与 dx/r 无关）===", flush=True)
N = 96
j1 = []
for r_um, dx_um in [(100., 25.), (100., 50.), (100., 100.), (100., 200.),
                    (50., 50.), (25., 100.), (10., 100.)]:
    r, dx = r_um * 1e-6, dx_um * 1e-6
    lo = -(N // 2) * dx
    g1 = jnp.linspace(lo, lo + (N - 1) * dx, N)
    cen = jnp.stack(jnp.meshgrid(g1, g1, g1, indexing="ij"), axis=-1)
    pos = jnp.array([0.37 * dx, -0.11 * dx, 0.29 * dx])
    fs = float(jnp.sum(_cell_integrated_source(pos, cen, 1.0, r, 1.2 * r, dx))
               * dx ** 3)
    ok = abs(fs - 1.0) < 1e-9
    j1.append(ok)
    print(f"  [integrated] r={r_um:5.0f} dx={dx_um:5.0f}  Σfrac={fs:.12f} "
          f"rel={abs(fs-1):.2e}  {'PASS' if ok else 'FAIL'}", flush=True)
print(f"  J1 => {'PASS' if all(j1) else 'FAIL'}", flush=True)
print("\n--- 对照：旧中点取值模型 Σ Q·dV / (ηP)（随 dx/r 漂移即离散漏能）---",
      flush=True)
for r_um, dx_um in [(100., 25.), (100., 50.), (100., 100.), (50., 100.)]:
    r, dx = r_um * 1e-6, dx_um * 1e-6
    lo = -(N // 2) * dx
    g1 = jnp.linspace(lo, lo + (N - 1) * dx, N)
    cen = jnp.stack(jnp.meshgrid(g1, g1, g1, indexing="ij"), axis=-1)
    pos = jnp.array([0.37 * dx, -0.11 * dx, 0.29 * dx])
    Q0 = 1.0 / (math.pi * r * r * math.sqrt(2 * math.pi) * 1.2 * r)
    s = float(jnp.sum(_moving_source(pos, cen, Q0, r, 1.2 * r)) * dx ** 3)
    print(f"  [point     ] r={r_um:5.0f} dx={dx_um:5.0f}  Σ/ηP={s:.6f} "
          f"偏差={abs(s-1):.2%}", flush=True)

# ------------------------------------------------------- 试片（标定级 coupon）
BX, BY, BZ = 1.2e-3, 0.6e-3, 0.4e-3       # 长×宽×厚，单道层序（layer_thickness=BZ）


def coupon(dx):
    return G.from_sdf_fn(
        lambda x: jnp.maximum(jnp.maximum(jnp.abs(x[..., 0]) - BX / 2,
                                          jnp.abs(x[..., 1]) - BY / 2),
                              jnp.abs(x[..., 2]) - BZ / 2),
        bounds=[(-BX / 2, BX / 2), (-BY / 2, BY / 2), (-BZ / 2, BZ / 2)],
        spacing=dx, name="coupon")


def plan(power, radius):
    return ProcessPlan.uniform(1, modality="SLM", laser_power=power,
                               scan_speed=V_SCAN, layer_thickness=BZ,
                               hatch_spacing=1.4 * radius, beam_radius=radius,
                               absorption=ETA, preheat_temp=400.0)


print("\n=== 拓扑（应与 dx 无关）===", flush=True)
for dx_um in (100., 50., 25.):
    g = coupon(dx_um * 1e-6)
    t = _scan_topology(g.coords(), plan(600., 100e-6), dim=3)
    print(f"  dx={dx_um:5.1f} shape={tuple(int(s) for s in g.shape)} "
          f"nvox={int(g.sdf.size):6d} n_layers={float(t[0]):.0f} "
          f"n_lines={float(t[1]):.0f} path={float(t[-1])*1e3:.3f}mm "
          f"ns_cfl={suggest_n_steps(g, plan(600., 100e-6))}", flush=True)

# ------------------------------------------------------------------ J2
print("\n=== J2 求解器注入剂量 == Σ Q·dV·dt = ηP·t_exposure（dx=25µm, r=100µm）===",
      flush=True)
P1, R1 = 600.0, 100e-6
dx = 25e-6
g = coupon(dx)
pl = plan(P1, R1)
ns = suggest_n_steps(g, pl)
positions, path_length = _build_scan_positions(g.coords(), pl, jnp.asarray(dx),
                                                n_steps=ns, dim=3)
t_exp = float(path_length) / V_SCAN
dt = t_exp / ns
msk = (g.sdf < 0).astype(jnp.float64)
ref = ETA * P1 * t_exp
raw = masked = 0.0
for t in range(ns):
    q = _cell_integrated_source(positions[t], g.coords(), ETA * P1, R1, 1.2 * R1, dx)
    raw += float(jnp.sum(q)) * dt * dx ** 3
    masked += float(jnp.sum(q * msk)) * dt * dx ** 3
ok2 = abs(raw - ref) / ref < 1e-9
print(f"  t_exposure={t_exp:.6e}s n_steps={ns} dt={dt:.6e}s ηP·t_exp={ref:.6f}J",
      flush=True)
print(f"  全域 Σ={raw:.6f}J rel_err={abs(raw-ref)/ref:.2e} => "
      f"{'PASS' if ok2 else 'FAIL'}", flush=True)
print(f"  实体内 Σ={masked:.6f}J 份额={masked/ref:.6f}"
      f"（<1 者为光束越出零件边界的真实损失，非离散误差）", flush=True)


def run(model, dx_um, power, radius, mult=1):
    dxd = dx_um * 1e-6
    gg = coupon(dxd)
    pp = plan(power, radius)
    ns_req = suggest_n_steps(gg, pp)
    n_steps = ns_req * mult
    t0 = time.perf_counter()
    th = solve_enthalpy_thermal(geometry=gg, process=pp,
                                params=dict(material="316L", source_model=model,
                                            cfl=0.35, n_steps=n_steps))
    el = time.perf_counter() - t0
    solid = gg.sdf < 0
    pk = th.peak_temperature
    melted = (pk > T_LIQ) & solid
    c = gg.coords()
    z = (jnp.max(c[..., 2], where=melted, initial=0.0)
         - jnp.min(c[..., 2], where=melted, initial=0.0))
    Lx = (jnp.max(c[..., 0], where=melted, initial=0.0)
          - jnp.min(c[..., 0], where=melted, initial=0.0))
    Ly = (jnp.max(c[..., 1], where=melted, initial=0.0)
          - jnp.min(c[..., 1], where=melted, initial=0.0))
    vol = float(jnp.sum(melted)) * dxd ** 3 * 1e9
    peak = float(jnp.max(pk))
    print(f"  [{model:10s}] dx={dx_um:5.1f} mult={mult} nvox={int(gg.sdf.size):6d} "
          f"ns={n_steps:5d} {el:7.2f}s | peak={peak:8.1f}K nml={int(jnp.sum(melted)):5d}"
          f" | Lx={float(Lx)*1e3:6.3f} Ly={float(Ly)*1e3:6.3f} depth={float(z)*1e3:6.3f}"
          f"mm vol={vol:8.3f}mm³", flush=True)
    return dict(model=model, dx=dx_um, mult=mult, ns=n_steps, peak=peak,
                nml=int(jnp.sum(melted)), Lx=float(Lx) * 1e3,
                Ly=float(Ly) * 1e3, depth=float(z) * 1e3, vol=vol, sec=el)


print("\n=== J3 三档收敛（两种源模型；P=600W r=100µm 线能量0.75J/mm）===", flush=True)
res = [run("integrated", d, P1, R1) for d in (100., 50., 25.)]
res += [run("point", d, P1, R1) for d in (100., 50., 25.)]


def score(tag):
    sel = [r for r in res if r["model"] == tag]
    for key in ("peak", "Lx", "vol"):
        v = [r[key] for r in sel]
        sp = (max(v) - min(v)) / max(v) if max(v) > 0 else 9.99
        print(f"  [{tag:10s}] {key:5s}={[round(x,3) for x in v]} 极差={sp:7.2%} "
              f"{'PASS' if sp < 0.05 else 'FAIL'}", flush=True)
    print(f"  [{tag:10s}] 三档均熔化={[r['nml']>0 for r in sel]} "
          f"{'PASS' if all(r['nml']>0 for r in sel) else 'FAIL'}", flush=True)
    sel2 = sorted(sel, key=lambda r: -r["dx"])
    print(f"  [{tag:10s}] 相邻档收敛比 (T100→T50)/(T50→T25)="
          f"{abs(sel2[0]['peak']-sel2[1]['peak'])/max(abs(sel2[1]['peak']-sel2[2]['peak']),1e-9):.2f}"
          f" （2阶收敛理论值≈4.0）", flush=True)


score("integrated")
score("point")

print("\n=== 时间采样归因：同网格加步数（dt 低于 CFL）看峰值是否靠拢细档 ===", flush=True)
res += [run("integrated", 100., P1, R1, mult=2), run("integrated", 100., P1, R1, mult=4),
        run("integrated", 50., P1, R1, mult=2), run("integrated", 50., P1, R1, mult=4)]
fine = [r for r in res if r["model"] == "integrated"]
for r in sorted(fine, key=lambda x: (-x["dx"], x["mult"])):
    print(f"    dx={r['dx']:6.1f} mult={r['mult']} ns={r['ns']:5d} "
          f"peak={r['peak']:8.1f} Lx={r['Lx']:6.3f} vol={r['vol']:8.3f}", flush=True)
print("\n完成。", flush=True)
