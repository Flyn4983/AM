"""#23 改前打分**第 2 轮**：把 fv 加权热容的 ablation 从 δ=0 扩到「每个 δ」（O1 规则口径）。

上一轮 `am_t23_capacity_probe.py` 的 J2 在 δ=0（零件面恰好落在体素中心）判定 fv 加权热容
把相邻档峰值差从 5.240%/1.549% 降到 3.969%/0.081%，但它自己写下了边界：**O1 规则要求的是
「每个 δ、每对相邻档 <5%」**，单档通过只是**必要不充分**。本轮补完 δ 族，不改生产代码
（monkeypatch 两个模块级 helper，`rhs` 里是调用期全局名查找 ⇒ 替换生效；由 K1 证明）。

    生产：  dH/dt = lap            + fv·s − fv·cool − fv·evap
    本探针：dH/dt = lap/max(fv,f0) + s·fv/max(fv,f0) − fv·cool − fv·evap
            ⇒ fv≥f0 的体素净为 lap/fv + s − fv·cool

夹具与 `am_t2_a0_alignment.py` **同源**（设计盒不动、栅格整体平移 δ ⇒ 等价于零件相对体素
错开 δ），δ ∈ {0, 0.25, 0.5}·dx，dx ∈ {50, 25, 12.5}µm。δ=0 与 δ=0.5·dx 有刀锋面（整层
sdf=0），δ=0.25·dx 无刀锋且表面体素 fv∈{0.25,0.75} ⇒ **这才有 fv 下限 f0 的真实灵敏度**
（δ=0 时所有表面体素 fv=0.5，f0=0.25 与 f0=0.5 逐位相同、灵敏度等于没测）。

跑前登记的证伪判据（不许事后改）：
  K1 对照：同一 (δ,dx) 下「恒等 patch」与生产解 **逐位相同**，且 patch 调用计数 >0。
     任一组失败 ⇒ 探针自身不可信，K2/K3/K4 全部不参与判据。
  K2 主判据（= O1 规则）：B=fv 加权 f0=0.5 对 **每个 δ、每对相邻档** 的 peak 相对差都 <5%，
     且 A=生产 至少一对 ≥5% ⇒ 判「域热容/剂量失配是 A0 档间漂移主因，且 fv 加权在全部对齐
     方式下都能压住」⇒ #23 按完整实现推进。
     若 B 在**任意** δ 的**任意**一对 ≥5% ⇒ 判「#23 单独不足以让 A0 转绿」；**不得**通过换
     f0、换观测量、增删 δ 或换档位来凑 PASS。f0=0.25 与 Vm 只作灵敏度记录。
  K3 下限灵敏度：报 B'(f0=0.25) 与 B(f0=0.5) 在每个 δ 的 peak 差。这是 fv 下限第一次真正
     被执行路径触及；若出现 NaN/发散 ⇒ 照实登记为 #23 第 2 条（`dt ∝ fv` 稳定性）的实测证据。
  K4 形态量：同一规则套在熔化体积 Vm 上（逐档相邻差）。若 peak 通过而 Vm 在某 δ ≥5%，照实写
     「峰值收敛但形态仍漂」——这不是判据失败，是 #23 完成后仍需面对的下一条。

诚实边界（不改）：cool/evap 仍乘 fv、`enthalpy_of_temperature` 反演仍按满体素、δ≠0 时网格
形状本身变了（nvox 报出）。设备：GPU；f64 设备无关性已在 §25.7 实测（Δpeak=0.0000），且 K1
的逐位对照在同进程内当场能抓住任何设备相关抖动。
"""
import math
import time

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from amforge import geometry as G                                          # noqa: E402
from amforge import thermal_enthalpy as TE                                 # noqa: E402
from amforge.core.contracts import (ProcessPlan, SDF_SOLID_TOL,             # noqa: E402
                                    solid_mask, solid_weight)
from amforge.materials import get_material                                 # noqa: E402
from amforge.thermal_enthalpy import (solve_enthalpy_thermal,               # noqa: E402
                                      suggest_n_steps)

EXT = (1.2e-3, 0.6e-3, 0.4e-3)
BX, BY, BZ, R, P, V = EXT[0], EXT[1], EXT[2], 100e-6, 600.0, 0.8
mat = get_material("316L")
TL, TS = float(mat.T_liquidus), float(mat.T_solidus)
TIERS = [50e-6, 25e-6, 12.5e-6]
DELTAS = [0.0, 0.25, 0.5]          # 单位：·dx


def coupon(dx, delta):
    """设计盒不动、栅格整体平移 delta（照抄 am_t2_a0_alignment.py:38-51）。"""
    bx, by, bz = EXT
    bounds = [(-bx / 2 - delta, bx / 2 + delta),
              (-by / 2 - delta, by / 2 + delta),
              (-bz / 2 - delta, bz / 2 + delta)]

    def fn(x):
        return jnp.maximum(jnp.maximum(jnp.abs(x[..., 0]) - bx / 2,
                                        jnp.abs(x[..., 1]) - by / 2),
                            jnp.abs(x[..., 2]) - bz / 2)

    return G.from_sdf_fn(lambda x: fn(np.asarray(x, dtype=np.float64)),
                         bounds=bounds, spacing=dx, name="coupon")


def plan():
    return ProcessPlan.uniform(1, modality="SLM", laser_power=P, scan_speed=V,
                               layer_thickness=BZ, hatch_spacing=1.4 * R,
                               beam_radius=R, absorption=0.45, preheat_temp=400.0)


ORIG_DIV, ORIG_SRC = TE._div_alpha_grad, TE._cell_integrated_source
CALLS = {"div": 0, "src": 0}
STATE = {"mode": "off", "f0": 1.0, "fv": None}


def patched_div(H, alpha, dx, mask=None):
    CALLS["div"] += 1
    lap = ORIG_DIV(H, alpha, dx, mask=mask)
    if STATE["mode"] == "fv":
        return lap / jnp.maximum(STATE["fv"], STATE["f0"])
    return lap


def patched_src(positions, coords, power, r_src, dp, dx):
    s = ORIG_SRC(positions, coords, power, r_src, dp, dx)
    CALLS["src"] += 1
    if STATE["mode"] == "fv":
        return s / jnp.maximum(STATE["fv"], STATE["f0"])
    return s


def solve(mode, f0, dx, frac):
    g, p = coupon(dx, frac * dx), plan()
    ns = suggest_n_steps(g, p)
    _tl = TE._scan_topology(g.coords(), p, dim=g.dim, solid=TE._footprint(g))
    t_exp = float(_tl[-1]) / V
    fv = solid_weight(g.sdf, jnp.asarray(dx, dtype=jnp.float64))
    STATE.update(mode=mode, f0=f0, fv=fv)
    CALLS.update(div=0, src=0)
    if mode == "none":
        TE._div_alpha_grad, TE._cell_integrated_source = ORIG_DIV, ORIG_SRC
    else:
        TE._div_alpha_grad, TE._cell_integrated_source = patched_div, patched_src
    t0 = time.perf_counter()
    try:
        th = solve_enthalpy_thermal(geometry=g, process=p,
                                    params=dict(material="316L",
                                                source_model="integrated",
                                                n_steps=ns))
        el = time.perf_counter() - t0
    finally:
        TE._div_alpha_grad, TE._cell_integrated_source = ORIG_DIV, ORIG_SRC
    sol = solid_mask(g.sdf) > 0.5
    pk = float(jnp.max(th.peak_temperature))
    mt = (th.peak_temperature > TL) & sol
    c = g.coords()
    Ly = float(jnp.max(c[..., 1], where=mt, initial=0.0)
               - jnp.min(c[..., 1], where=mt, initial=0.0)) * 1e3
    w = solid_weight(g.sdf, dx)
    sdf_np = np.asarray(g.sdf, dtype=np.float64)
    fv_np = np.asarray(fv, dtype=np.float64)
    Hf = TE.enthalpy_of_temperature(th.peak_temperature, rho=float(mat.rho_solid),
                                    cp=float(mat.cp_solid),
                                    L=float(mat.latent_fusion),
                                    T_amb=float(mat.T_ambient), T_sol=TS, T_liq=TL)
    surf = (fv_np > 0.0) & (fv_np < 1.0)
    return dict(pk=pk, n=int(jnp.sum(mt)), Ly=Ly,
                vm=float(jnp.sum(jnp.where(mt, w, 0.0))) * dx ** 3 * 1e9,
                finite=bool(math.isfinite(pk)),
                ns=int(ns), nvox=int(g.sdf.size),
                n_tie=int(np.sum(np.abs(sdf_np) < SDF_SOLID_TOL)),
                n_surf=int(np.sum(surf)),
                fv_min=float(np.min(fv_np[surf])) if int(np.sum(surf)) else float("nan"),
                t_exp=t_exp,
                e_mat=float(jnp.sum(fv * Hf)) * dx ** 3,
                e_full=float(jnp.sum(jnp.where(sol, Hf, 0.0))) * dx ** 3,
                calls=dict(CALLS), el=el)


MODES = [("none", 1.0), ("identity", 1.0), ("fv", 0.5), ("fv", 0.25)]
RES = {}
print(f"device = {jax.devices()[0]}", flush=True)
for mode, f0 in MODES:
    for frac in DELTAS:
        for dx in TIERS:
            r = RES[(mode, f0, frac, dx)] = solve(mode, f0, dx, frac)
            print(f"[{mode:8s} f0={f0:4.2f}] δ={frac:4.2f}·dx dx={dx*1e6:5.1f} "
                  f"nvox={r['nvox']:7d} ns={r['ns']:5d} {r['el']:6.2f}s "
                  f"peak={r['pk']:8.1f} nml={r['n']:6d} Ly={r['Ly']:6.3f}mm "
                  f"Vm={r['vm']:7.4f}mm3 刀锋={r['n_tie']:6d} 表面={r['n_surf']:6d}"
                  f"(min fv={r['fv_min']:.2f}) Efv={r['e_mat']:.4f} "
                  f"Efull={r['e_full']:.4f} calls={r['calls']}", flush=True)

bad = []
print("\n=== K1 逐位对照：生产(none) vs 恒等 patch(identity)，每个 (δ,dx) ===")
for frac in DELTAS:
    for dx in TIERS:
        a, b = RES[("none", 1.0, frac, dx)], RES[("identity", 1.0, frac, dx)]
        same = all(a[k] == b[k] for k in ("pk", "n", "Ly", "vm", "e_mat"))
        fired = b["calls"]["div"] > 0 and b["calls"]["src"] > 0
        print(f"  δ={frac:4.2f}·dx dx={dx*1e6:5.1f}: 逐位={'YES' if same else 'NO'} "
              f"(peak {a['pk']!r} vs {b['pk']!r}) 调用 div={b['calls']['div']} "
              f"src={b['calls']['src']} ⇒ {'看得见' if fired else '看不见'}", flush=True)
        if not same:
            bad.append(f"K1 δ={frac}dx dx={dx*1e6}µm：恒等 patch 未逐位复现生产解")
        if not fired:
            bad.append(f"K1 δ={frac}dx dx={dx*1e6}µm：patch 从未被调用 ⇒ ablation 不生效")
for site, r in RES.items():
    if not r["finite"]:
        mode_i, f0_i, frac_i, dx_i = site
        bad.append(f"K3 峰值非有限：mode={mode_i} f0={f0_i} δ={frac_i}·dx "
                   f"dx={dx_i*1e6}µm ⇒ 登记为 #23 第 2 条（dt∝fv 稳定性）的实测证据")


def drift(mode, f0, frac, key="pk"):
    v = [RES[(mode, f0, frac, dx)][key] for dx in TIERS]
    return [abs(v[i] - v[i + 1]) / max(v[i], v[i + 1]) * 100 for i in range(len(v) - 1)]


print("\n=== K2 主判据：相邻档 peak 相对差（50↔25, 25↔12.5）[%]，每个 δ ===")
k2_a_fail = False
k2_b_fail = []
for frac in DELTAS:
    A = drift("none", 1.0, frac)
    B = drift("fv", 0.5, frac)
    for nm, d in (("A 生产", A), ("B fv f0=0.5", B)):
        print(f"  δ={frac:4.2f}·dx  {nm:12s} = {d[0]:6.3f} / {d[1]:6.3f}", flush=True)
    if max(A) >= 5.0:
        k2_a_fail = True
    if min(B) >= 5.0 or max(B) >= 5.0:
        k2_b_fail.append((frac, B))
    if bad:
        print("     （K1 未过 ⇒ 本行不参与判据）")
if bad:
    verdict = "K1 未过 ⇒ 本轮不参与判据"
elif k2_b_fail:
    verdict = ("J3/否证分支：B 在 " + ", ".join(f"δ={f}dx({d[0]:.2f}/{d[1]:.2f}%)"
               for f, d in k2_b_fail) + " 仍 ≥5% ⇒ #23 单独不足以让 A0 转绿（不许换 "
               "f0/观测量/δ/档位凑 PASS）")
elif not k2_a_fail:
    verdict = "A 全部 <5% ⇒ 与 §26.16(e) 既有记录矛盾，需复核而不是宣布通过"
else:
    verdict = ("PASS：A 至少一对 ≥5% 且 B 在**每个 δ、每对相邻档**都 <5% ⇒ 域热容/剂量失配"
               "是 A0 档间漂移主因，fv 加权热容在所有对齐方式下都压住 ⇒ #23 按完整实现推进")
print(f"\n  ⇒ K2 结论：{verdict}")

print("\n=== K3 fv 下限灵敏度：f0=0.25 vs f0=0.5 的 peak 绝对差 [%]（只在 δ≠0 才有意义）===")
for frac in DELTAS:
    row = []
    for dx in TIERS:
        b5 = RES[("fv", 0.5, frac, dx)]["pk"]
        b25 = RES[("fv", 0.25, frac, dx)]["pk"]
        row.append(abs(b25 - b5) / max(abs(b5), abs(b25)) * 100)
    print(f"  δ={frac:4.2f}·dx: 50/25/12.5µm = {row[0]:6.3f} / {row[1]:6.3f} / "
          f"{row[2]:6.3f}（表面 fv={RES[('fv', 0.5, frac, TIERS[0])]['fv_min']:.2f}）",
          flush=True)

print("\n=== K4 形态量：相邻档 Vm 相对差 [%]（不参与判据，照实写）===")
for frac in DELTAS:
    A = drift("none", 1.0, frac, key="vm")
    B = drift("fv", 0.5, frac, key="vm")
    print(f"  δ={frac:4.2f}·dx  A = {A[0]:6.2f} / {A[1]:6.2f}   B = {B[0]:6.2f} / "
          f"{B[1]:6.2f}", flush=True)

print("\n=== 记账：Efv/Efull（域热能 vs 满体素热能，逐位同源数组 ⇒ 差只来自离散口径）===")
for frac in DELTAS:
    for dx in TIERS:
        a = RES[("none", 1.0, frac, dx)]
        print(f"  δ={frac:4.2f}·dx dx={dx*1e6:5.1f}: t_exp={a['t_exp']:.4f}s "
              f"hcool·t_exp={0.5*a['t_exp']:.2e} Efv/Efull={a['e_mat']/max(a['e_full'],1e-30):6.4f} "
              f"nvox={a['nvox']} 刀锋={a['n_tie']} 表面={a['n_surf']}", flush=True)

print(f"\nFAILS: {bad if bad else '无（K1 全组逐位等价且 patch 确在执行路径上）'}")
print(f"K2: {verdict}")
raise SystemExit(1 if bad else 0)
