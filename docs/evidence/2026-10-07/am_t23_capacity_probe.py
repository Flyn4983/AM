"""#23 前置机理探针（改前打分，**不是生产改动**）：热容按 fv 加权后，A0 的档间漂移会不会消失？

生产 `rhs` 是函数内闭包，无法整体替换，故本探针只 monkeypatch 两个模块级 helper
（`_div_alpha_grad`、`_cell_integrated_source`，二者在 `rhs` 里都是**调用期全局名查找**，
所以替换生效；由 J1 的调用计数与逐位对照共同证明）：

    生产：  dH/dt = lap              + fv·s  − fv·cool − fv·evap
    本探针：dH/dt = lap/max(fv,f0)   + s·fv/max(fv,f0)·…  ⇒ fv≥f0 时净为 lap/fv + s − fv·cool

即把 #23 第 1 条的力学内核 `dH/dt = lap/fv + Q_vol − cool` 里**占主导的两项**（扩散与源）
做成同口径，cool/evap 仍按 fv 缩放、`temperature_of_enthalpy` 仍按满体素反演 ⇒ 已知两处
不一致（弱冷却 hcool=0.5 s⁻¹ 在 ms 级曝光上是小项），照实写在下面而不是藏起来。

预注册判据（跑前写死，不许事后改）：
  J1 对照：不打 patch 的生产解 vs 打了**恒等** patch 的解，peak/nml/Ly/Vm **逐位相同**，
     且 patch 调用计数 > 0 ⇒ 打 patch 这件事本身不引入差、且 ablation 确实在执行路径上。
  J2 主判据：A=生产，B=fv 加权（f0=0.5）。若 B 的两对相邻档 peak 相对差**都 <5%** 而
     A 至少一对 ≥5% ⇒ 判"域热容/剂量失配是 A0 漂移主因"，#23 按完整实现推进。
  J3 否证：若 B 仍有任意一对 ≥5% ⇒ 判"#23 单独不足以让 A0 转绿"，须另找机制；
     **不得**通过调 f0、换观测量、换 δ 凑 PASS（f0=0.25 只作灵敏度记录，不参与判据）。
  J4 记账：报保留能 Σfv·H(Tf)·dx³ 与 ΣH(Tf)·dx³（源数组逐位相同 ⇒ 差值只来自离散），
     完整能量闭合（含 cool/evap 与 T 反演口径）留给 #23 实现时再验。

边界（照实记）：δ=0（零件面与体素中心对齐）一档、integrated 光源、dx=50/25/12.5µm；
O1 规则要求"**每个 δ**"，故 J2 通过只是**必要不充分**。运行在 GPU：f64 已证设备无关
（§25.7 Δpeak=0.0000），且 J1 的逐位对照会当场抓住任何设备相关偏差。
"""
import time

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from amforge import geometry as G                                        # noqa: E402
from amforge import thermal_enthalpy as TE                              # noqa: E402
from amforge.core.contracts import ProcessPlan, solid_mask, solid_weight  # noqa: E402
from amforge.materials import get_material                               # noqa: E402
from amforge.thermal_enthalpy import solve_enthalpy_thermal, suggest_n_steps  # noqa: E402

BX, BY, BZ, R, P, V = 1.2e-3, 0.6e-3, 0.4e-3, 100e-6, 600.0, 0.8
mat = get_material("316L")
TL, TS = float(mat.T_liquidus), float(mat.T_solidus)
TIERS = [50e-6, 25e-6, 12.5e-6]


def coupon(dx):
    return G.from_sdf_fn(
        lambda x: jnp.maximum(jnp.maximum(jnp.abs(x[..., 0]) - BX / 2,
                                          jnp.abs(x[..., 1]) - BY / 2),
                              jnp.abs(x[..., 2]) - BZ / 2),
        bounds=[(-BX / 2, BX / 2), (-BY / 2, BY / 2), (-BZ / 2, BZ / 2)],
        spacing=dx, name="c")


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


def solve(mode, f0, dx):
    g, p = coupon(dx), plan()
    ns = suggest_n_steps(g, p)
    _tl = TE._scan_topology(g.coords(), p, dim=g.dim, solid=TE._footprint(g))
    path_length = float(_tl[-1])
    STATE.update(mode=mode, f0=f0, fv=solid_weight(g.sdf, jnp.asarray(dx, dtype=jnp.float64)))
    CALLS.update(div=0, src=0)
    if mode == "none":
        TE._div_alpha_grad, TE._cell_integrated_source = ORIG_DIV, ORIG_SRC
    else:
        TE._div_alpha_grad, TE._cell_integrated_source = patched_div, patched_src
    t0 = time.perf_counter()
    th = solve_enthalpy_thermal(geometry=g, process=p,
                                params=dict(material="316L", source_model="integrated",
                                            n_steps=ns))
    el = time.perf_counter() - t0
    TE._div_alpha_grad, TE._cell_integrated_source = ORIG_DIV, ORIG_SRC
    sol = solid_mask(g.sdf) > 0.5
    mt = (th.peak_temperature > TL) & sol
    c = g.coords()
    Ly = float(jnp.max(c[..., 1], where=mt, initial=0.0)
               - jnp.min(c[..., 1], where=mt, initial=0.0)) * 1e3
    w = solid_weight(g.sdf, dx)
    fv = STATE["fv"]
    Hf = TE.enthalpy_of_temperature(th.peak_temperature, rho=float(mat.rho_solid),
                                    cp=float(mat.cp_solid), L=float(mat.latent_fusion),
                                    T_amb=float(mat.T_ambient), T_sol=TS, T_liq=TL)
    out = dict(pk=float(jnp.max(th.peak_temperature)), n=int(jnp.sum(mt)), Ly=Ly,
               vm=float(jnp.sum(jnp.where(mt, w, 0.0))) * dx ** 3 * 1e9,
               ns=int(ns), nvox=int(g.sdf.size), el=el,
               t_exp=float(path_length / V),
               e_mat=float(jnp.sum(fv * Hf)) * dx ** 3,
               e_full=float(jnp.sum(jnp.where(sol, Hf, 0.0))) * dx ** 3,
               calls=dict(CALLS))
    return out


RES = {}
for mode, f0 in (("none", 1.0), ("identity", 1.0), ("fv", 0.5), ("fv", 0.25)):
    for dx in TIERS:
        r = RES[(mode, f0, dx)] = solve(mode, f0, dx)
        print(f"[{mode:8s} f0={f0:4.2f}] dx={dx*1e6:5.1f} nvox={r['nvox']:7d} ns={r['ns']:5d} "
              f"{r['el']:6.2f}s peak={r['pk']:8.1f} nml={r['n']:6d} Ly={r['Ly']:6.3f}mm "
              f"Vm={r['vm']:7.4f}mm3 Efv={r['e_mat']:.4f} Efull={r['e_full']:.4f} "
              f"calls={r['calls']}", flush=True)

bad = []
print("\n=== J1 逐位对照：生产(none) vs 恒等 patch(identity) ===")
for dx in TIERS:
    a, b = RES[("none", 1.0, dx)], RES[("identity", 1.0, dx)]
    same = all(a[k] == b[k] for k in ("pk", "n", "Ly", "vm", "e_mat"))
    fired = b["calls"]["div"] > 0 and b["calls"]["src"] > 0
    print(f"  dx={dx*1e6:5.1f}: 逐位={'YES' if same else 'NO'} "
          f"(peak {a['pk']!r} vs {b['pk']!r})  patch 调用 div={b['calls']['div']} "
          f"src={b['calls']['src']} ⇒ {'看得见' if fired else '看不见'}")
    if not same:
        bad.append(f"J1 失败 dx={dx*1e6}µm：恒等 patch 未逐位复现生产解 ⇒ 探针自身不可信，GPU/设备回退 CPU 复核")
    if not fired:
        bad.append(f"J1 失败 dx={dx*1e6}µm：patch 从未被调用 ⇒ ablation 不可能生效")


def drift(mode, f0, key="pk"):
    v = [RES[(mode, f0, dx)][key] for dx in TIERS]
    return [abs(v[i] - v[i + 1]) / max(v[i], v[i + 1]) * 100 for i in range(len(v) - 1)]


A, B5, B25 = drift("none", 1.0), drift("fv", 0.5), drift("fv", 0.25)
print("\n=== J2/J3 相邻档 peak 相对差（50↔25, 25↔12.5）[%] ===")
for nm, d in (("A 生产（满体素热容）", A), ("B fv 加权 f0=0.5", B5),
              ("B' 灵敏度 f0=0.25", B25)):
    print(f"  {nm:24s} = {d[0]:6.3f} / {d[1]:6.3f}")
if bad:
    print("  ⇒ J1 未过 ⇒ 本行不参与判据")
elif min(B5) < 5.0 and max(A) >= 5.0:
    print("  ⇒ J2：域热容/剂量失配是主因，#23 值得完整实现（仍受 O1「每个 δ」约束）")
elif min(B5) >= 5.0:
    print("  ⇒ J3：#23 单独不足以让 A0 转绿，必须另找机制（不许靠 f0/观测量/δ 凑）")
else:
    print("  ⇒ A 与 B 都 <5%：本轮档选择与既有 §26.16(e) 记录矛盾，需复核而非宣布通过")

print("\n=== 形态量与记账（A → B，f0=0.5）===")
for dx in TIERS:
    a, b = RES[("none", 1.0, dx)], RES[("fv", 0.5, dx)]
    print(f"  dx={dx*1e6:5.1f}: peak {a['pk']:8.1f} → {b['pk']:8.1f} "
          f"({(b['pk']-a['pk'])/a['pk']*100:+6.2f}%)  nml {a['n']:6d} → {b['n']:6d}  "
          f"Ly {a['Ly']:6.3f} → {b['Ly']:6.3f}  Vm {a['vm']:7.4f} → {b['vm']:7.4f}mm3  "
          f"Efv/Efull {a['e_mat']/max(a['e_full'],1e-30):5.3f} → {b['e_mat']/max(b['e_full'],1e-30):5.3f}")

print("\n已知不一致（探针范围的诚实边界）：cool/evap 仍乘 fv、T=H 反演仍按满体素。"
      "弱冷却项的相对作用实测 = hcool·t_exposure：")
for dx in TIERS:
    a = RES[("none", 1.0, dx)]
    print(f"  dx={dx*1e6:5.1f}: t_exposure={a['t_exp']:.4f}s ⇒ hcool·t_exp="
          f"{0.5*a['t_exp']:.2e}（≪1 ⇒ cool 是小项；完整能量闭合属 #23 实现）")
print("FAILS:", bad if bad else "无（J1 逐位等价且 patch 确在执行路径上；J2/J3 结论见上）")
raise SystemExit(1 if bad else 0)
