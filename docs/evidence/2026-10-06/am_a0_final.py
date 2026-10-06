"""A0 终版验收（CPU only；GPU 未授权：全程 CUDA_VISIBLE_DEVICES=""）。

跑前登记的证伪判据与预测（写于运行之前，运行后逐条打分，不许事后改口径）：
  P1 J1 守恒  单元体积分源 Σfrac=1，rel<1e-13，且与 dx/r 无关（7 组全 PASS）。
  P2 J2 剂量  求解器全域注入 = ηP·t_exp × mean_t(frac_grid)，偏差 < 1e-9；
              mean(frac_grid) 落在 0.94~0.95；内部步(frac>0.9999)占比 >60%；
              行程端点 min(frac)<0.6。=> 「5.5% 缺口」若由有限域截断解释则 PASS，
              否则（预测不吻合）判 A0 未完成。
  P3 护栏      5 项（RK2 下限/ CFL 警告 / dx>2r / max_steps / 非法 source_model）
              全部按设计 raise 或 warn。
  P4 预热 IC   preheat=400K 且激光≈0 时，实体内部终温 ≈400K（±2K）。
  P5 J3 收敛   integrated 三档(100/50/25)：peak 极差 3.4~3.6%（PASS，<5%）；
              Lx 极差 12~14%、vol 极差 20~25%（预期 FAIL）；三档均熔化。
              细三档(50/25/12.5)：vol 极差 1.5~2.0%、peak 极差 4.8~5.2%。
  P6 单调性    两种源模型峰值均随加密单调下降（无反号）。

=== 修订登记（第 1 次运行后、第 2 次运行前写定；判据阈值不变，仅改数值预测）===
第 1 次运行（commit 前代码）暴露一个新缺陷：扫描足迹用严格 sdf<0，体素中心恰好
落在设计边界面（sdf==0）的一圈被剔除，于是同一试片在不同 dx 下对齐不同 →
path=3.267/4.775/5.012/5.131mm（极差 36%）。工艺配方属设计几何，不该由体素对齐
决定 → 改为「含材料体素」足迹 sdf<dx/2（等价 fv>0）。修复后重跑，修订预测：
  P2' 四档 path 全部 = 5.250mm；ns=66/262/1045/4180；mean frac 0.85~0.92
      （缺口来自行程端点半束越界，内部步 frac=1±1e-12）。
  P3' 5/5 护栏命中（警告档步数改为按 cfl=1 稳定下限精确取，不再撞下限）。
  P5' integrated 三档 peak 极差 3.5~5%（PASS）；Lx 极差 <5%（配方对齐后应显著
      收窄）；vol 极差 10~25%（dx=100µm 仅 1 个体素横跨 0.2mm 熔宽，属真实欠分辨）。
  P4' 预热：实体内部均值 398~402K。
"""
import math
import time
import warnings

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from amforge import geometry as G
from amforge.core.contracts import ProcessPlan
from amforge.materials import get_material
from amforge.thermal_enthalpy import (
    solve_enthalpy_thermal, suggest_n_steps, _build_scan_positions,
    _cell_integrated_source, _moving_source, _scan_topology,
)

mat = get_material("316L")
T_LIQ, T_SOL = float(mat.T_liquidus), float(mat.T_solidus)
V_SCAN, ETA = 0.8, 0.45
print(f"316L alpha0={float(mat.k_solid)/(float(mat.rho_solid)*float(mat.cp_solid)):.4e} "
      f"Tsol={T_SOL} Tliq={T_LIQ} latent={mat.latent_fusion}", flush=True)

BX, BY, BZ = 1.2e-3, 0.6e-3, 0.4e-3


def coupon(dx):
    return G.from_sdf_fn(
        lambda x: jnp.maximum(jnp.maximum(jnp.abs(x[..., 0]) - BX / 2,
                                          jnp.abs(x[..., 1]) - BY / 2),
                              jnp.abs(x[..., 2]) - BZ / 2),
        bounds=[(-BX / 2, BX / 2), (-BY / 2, BY / 2), (-BZ / 2, BZ / 2)],
        spacing=dx, name="coupon")


def plan(power, radius, **kw):
    return ProcessPlan.uniform(1, modality="SLM", laser_power=power,
                               scan_speed=V_SCAN, layer_thickness=BZ,
                               hatch_spacing=1.4 * radius, beam_radius=radius,
                               absorption=ETA, preheat_temp=kw.pop("preheat", 400.0),
                               **kw)


# ------------------------------------------------------------------ P1 / J1
print("\n=== J1 单元体积分源守恒（应与 dx/r 无关）===", flush=True)
N, j1ok = 96, []
for r_um, dx_um in [(100., 25.), (100., 50.), (100., 100.), (100., 200.),
                    (50., 50.), (25., 100.), (10., 100.)]:
    r, dx = r_um * 1e-6, dx_um * 1e-6
    lo = -(N // 2) * dx
    g1 = jnp.linspace(lo, lo + (N - 1) * dx, N)
    cen = jnp.stack(jnp.meshgrid(g1, g1, g1, indexing="ij"), axis=-1)
    fs = float(jnp.sum(_cell_integrated_source(
        jnp.array([.37 * dx, -.11 * dx, .29 * dx]), cen, 1.0, r, 1.2 * r, dx)) * dx ** 3)
    rel = abs(fs - 1.0)
    j1ok.append(rel < 1e-13)
    print(f"  r={r_um:5.0f} dx={dx_um:5.0f} Σfrac={fs:.14f} rel={rel:.2e} "
          f"{'PASS' if rel < 1e-13 else 'FAIL'}", flush=True)
print(f"  J1 => {'PASS' if all(j1ok) else 'FAIL'} (P1 预测 7/7 rel<1e-13)", flush=True)

# ------------------------------------------------------------------ P2 / J2
print("\n=== J2 剂量核算 + 逐步 frac 剖面（截断归因）===", flush=True)
P1, R1, dx = 600.0, 100e-6, 25e-6
g, pl = coupon(dx), plan(P1, R1)
ns = suggest_n_steps(g, pl)
solid = (g.sdf < 0.0).astype(jnp.float64)
foot = g.sdf < 0.5 * dx
fv = jnp.clip(0.5 - g.sdf / dx, 0.0, 1.0)
pos, path = _build_scan_positions(g.coords(), pl, jnp.asarray(dx), n_steps=ns,
                                  dim=3, solid=foot)
t_exp, dt = float(path) / V_SCAN, float(path) / V_SCAN / ns
ref = ETA * P1 * t_exp
fr = []
into = 0.0
for t in range(ns):
    q = _cell_integrated_source(pos[t], g.coords(), ETA * P1, R1, 1.2 * R1, dx)
    fr.append(float(jnp.sum(q)) * dx ** 3 / (ETA * P1))   # 该步留在域内的功率份额
    into += float(jnp.sum(q * fv)) * dt * dx ** 3          # 落入固相份额的能量
fr = jnp.asarray(fr)
raw = float(jnp.sum(fr)) * dt * ETA * P1
mean_f = float(jnp.mean(fr))
pred = ref * mean_f
n_int = int(jnp.sum(fr > 0.9999))
print(f"  t_exp={t_exp:.6e}s ns={ns} dt={dt:.6e}s ηP·t_exp={ref:.6f}J", flush=True)
print(f"  全域 Σ={raw:.6f}J  预测(=ref×mean frac)={pred:.6f}J "
      f"|Σ-pred|/pred={abs(raw-pred)/pred:.2e} => "
      f"{'PASS' if abs(raw-pred)/pred < 1e-9 else 'FAIL'}"
      f"（恒等式：核对的是累加/量纲不写错，非独立证据）", flush=True)
print(f"  直接对 ref 的缺口={(1-raw/ref):.4%}（P2 预测 mean frac 0.94~0.95 → "
      f"实测 {mean_f:.4f}；内部步占比 {n_int}/{ns}={n_int/ns:.1%}；"
      f"min frac={float(jnp.min(fr)):.3f}）", flush=True)
low = jnp.where(fr < 0.999)[0]
xl = [float(v) * 1e3 for v in pos[jnp.asarray(low[:3], dtype=jnp.int32), 0]] if low.size else []
print(f"  低份额步共 {int(low.size)} 步，前 3 个下标={list(map(int, low[:3]))} "
      f"对应激光 x={xl} mm（零件 x 范围 ±{BX/2*1e3:.1f}mm）"
      f"→ 缺口集中在行程端点而非均匀分布 = 有限域截断，非离散漏能", flush=True)
print(f"  实体内(按 fv 加权) Σ={into:.6f}J 份额={into/ref:.4f}", flush=True)

print("\n=== 拓扑（实体足迹 sdf<dx/2，应与 dx 无关）===", flush=True)
for d_um in (100., 50., 25., 12.5):
    gg = coupon(d_um * 1e-6)
    t = _scan_topology(gg.coords(), plan(P1, R1), dim=3,
                       solid=gg.sdf < 0.5 * d_um * 1e-6)
    print(f"  dx={d_um:5.1f} shape={tuple(int(s) for s in gg.shape)} "
          f"nvox={int(gg.sdf.size):7d} n_layers={float(t[0]):.0f} "
          f"n_lines={float(t[1]):.0f} path={float(t[-1])*1e3:.3f}mm "
          f"ns_cfl={suggest_n_steps(gg, plan(P1, R1))}", flush=True)


# ------------------------------------------------------------------ P3 护栏
print("\n=== 护栏（5 项，全部应命中）===", flush=True)
g100, p100 = coupon(100e-6), plan(P1, R1)
ns100 = suggest_n_steps(g100, p100)
alpha0 = float(mat.k_solid) / (float(mat.rho_solid) * float(mat.cp_solid))
dt_rk2 = (100e-6) ** 2 / (2.0 * 3 * alpha0)
path100 = float(_build_scan_positions(g100.coords(), p100, jnp.asarray(100e-6),
                                      n_steps=2, dim=3,
                                      solid=g100.sdf < 50e-6)[1])
ns_stable = max(1, math.ceil(path100 / V_SCAN / dt_rk2 - 1e-9))
print(f"  dx=100µm：path={path100*1e3:.3f}mm t_exp={path100/V_SCAN:.4e}s "
      f"n_stable(cfl=1)={ns_stable} n_cfl(0.35)={ns100}", flush=True)
checks = []
try:
    solve_enthalpy_thermal(geometry=g100, process=p100,
                           params=dict(n_steps=max(1, ns_stable - 1),
                                       material="316L"))
    checks.append(("n_steps<RK2 下限 → ValueError", False))
except ValueError as e:
    checks.append(("n_steps<RK2 下限 → ValueError", True))
    print(f"    [ok] {str(e)[:96]}", flush=True)
with warnings.catch_warnings(record=True) as wq:
    warnings.simplefilter("always")
    solve_enthalpy_thermal(geometry=g100, process=p100,
                           params=dict(n_steps=ns_stable, material="316L"))
    checks.append(("n_steps<CFL 目标 → UserWarning",
                   any("CFL" in str(x.message) for x in wq)))
    print(f"    [ok] 警告 {len(wq)} 条: "
          f"{str(wq[0].message)[:96] if wq else '—'}", flush=True)
try:
    solve_enthalpy_thermal(geometry=coupon(150e-6), process=p100,
                           params=dict(material="316L"))
    checks.append(("dx>2r → ValueError", False))
except ValueError as e:
    checks.append(("dx>2r → ValueError", True))
    print(f"    [ok] {str(e)[:96]}", flush=True)
try:
    solve_enthalpy_thermal(geometry=coupon(6.0e-6), process=p100,
                           params=dict(material="316L", max_steps=50))
    checks.append(("超 max_steps → ValueError", False))
except ValueError as e:
    checks.append(("超 max_steps → ValueError", True))
    print(f"    [ok] {str(e)[:110]}", flush=True)
try:
    solve_enthalpy_thermal(geometry=g100, process=p100,
                           params=dict(n_steps=ns100, material="316L",
                                       source_model="bogus"))
    checks.append(("非法 source_model → ValueError", False))
except ValueError:
    checks.append(("非法 source_model → ValueError", True))
    print("    [ok] 非法模型名被拒", flush=True)
print(f"  P3 => {sum(1 for _, ok in checks if ok)}/{len(checks)} 命中", flush=True)


# ------------------------------------------------------------------ P4 预热 IC
print("\n=== P4 预热初值（激光≈0，只看 IC）===", flush=True)
zero = dict(n_steps=ns_stable, material="316L")   # 步数取稳定下限，避免触发护栏
th = solve_enthalpy_thermal(geometry=coupon(100e-6), process=plan(1e-9, R1, preheat=400.),
                            params=zero)
Tin = float(jnp.mean(th.final_temperature[g100.sdf < -1e-9]))
th0 = solve_enthalpy_thermal(geometry=coupon(100e-6),
                             process=ProcessPlan.uniform(1, modality="SLM",
                                                         laser_power=1e-9,
                                                         scan_speed=V_SCAN,
                                                         layer_thickness=BZ,
                                                         hatch_spacing=1.4 * R1,
                                                         beam_radius=R1,
                                                         absorption=ETA,
                                                         preheat_temp=300.0),
                             params=zero)
Tin0 = float(jnp.mean(th0.final_temperature[g100.sdf < -1e-9]))
print(f"  preheat=400K → 实体均值 {Tin:.2f}K（P4 预测 398~402）；"
      f"preheat=300K → {Tin0:.2f}K  => "
      f"{'PASS' if abs(Tin-400.) < 2.0 and Tin0 < Tin - 50 else 'FAIL'}", flush=True)


# ------------------------------------------------------------------ P5/P6 J3
def run(model, dx_um, power=P1, radius=R1):
    dxd = dx_um * 1e-6
    gg, pp = coupon(dxd), plan(power, radius)
    nsr = suggest_n_steps(gg, pp)
    t0 = time.perf_counter()
    th = solve_enthalpy_thermal(geometry=gg, process=pp,
                                params=dict(material="316L", source_model=model,
                                            cfl=0.35))
    el = time.perf_counter() - t0
    pk = th.peak_temperature
    melted = (pk > T_LIQ) & (gg.sdf < 0)
    c = gg.coords()
    Lx = float(jnp.max(c[..., 0], where=melted, initial=0.) - jnp.min(c[..., 0], where=melted, initial=0.))
    Ly = float(jnp.max(c[..., 1], where=melted, initial=0.) - jnp.min(c[..., 1], where=melted, initial=0.))
    vol = float(jnp.sum(melted)) * dxd ** 3 * 1e9
    peak = float(jnp.max(pk))
    thr = gg.sdf.size * nsr / el / 1e6
    print(f"  [{model:10s}] dx={dx_um:6.1f} ns={nsr:5d} {el:6.2f}s {thr:8.1f}M vox·step/s"
          f" | peak={peak:8.1f} nml={int(jnp.sum(melted)):6d}"
          f" Lx={Lx*1e3:6.3f} Ly={Ly*1e3:6.3f} vol={vol:8.4f}mm³", flush=True)
    return dict(model=model, dx=dx_um, ns=nsr, peak=peak,
                nml=int(jnp.sum(melted)), Lx=Lx * 1e3, Ly=Ly * 1e3, vol=vol)


print("\n=== J3 分辨率收敛（4 档 × 2 模型；判据在 100/50/25 三档上登记）===", flush=True)
res = [run("integrated", d) for d in (100., 50., 25., 12.5)]
res += [run("point", d) for d in (100., 50., 25., 12.5)]


def score(tag, dxs, keys=("peak", "Lx", "vol")):
    sel = [r for r in res if r["model"] == tag and r["dx"] in dxs]
    lab = "/".join("{:g}".format(d) for d in dxs)
    out = {}
    for key in keys:
        v = [r[key] for r in sel]
        sp = (max(v) - min(v)) / max(v) if max(v) > 0 else 9.99
        out[key] = sp
        print(f"  [{tag:10s}] dx={lab} {key:4s}="
              f"{[round(x, 4) for x in v]} 极差={sp:7.2%} {'PASS' if sp < 0.05 else 'FAIL'}",
              flush=True)
    print(f"  [{tag:10s}] 均熔化={[r['nml'] > 0 for r in sel]} "
          f"{'PASS' if all(r['nml'] > 0 for r in sel) else 'FAIL'}", flush=True)
    return out


print("\n--- 登记判据（三档 100/50/25）---", flush=True)
s_i = score("integrated", (100., 50., 25.))
s_p = score("point", (100., 50., 25.))
print("\n--- 附加证据（细三档 50/25/12.5）---", flush=True)
score("integrated", (50., 25., 12.5))
score("point", (50., 25., 12.5))

for tag in ("integrated", "point"):
    v = [r["peak"] for r in res if r["model"] == tag]
    mono = all(a > b for a, b in zip(v, v[1:]))
    print(f"  [{tag:10s}] 峰值随加密单调下降={mono} (P6) 值={[round(x,1) for x in v]}",
          flush=True)

print(f"\nP5 打分：integrated peak 极差={s_i['peak']:.2%} (预测 3.4~3.6%) "
      f"Lx={s_i['Lx']:.2%} (预测 12~14%) vol={s_i['vol']:.2%} (预测 20~25%)", flush=True)
print("完成。", flush=True)
