"""#25 改前打分（第 1 轮）：**面开口度**能不能补上 #23 治不了的熔池形态（A0 assert2）？

第 4 轮 `am_t23_a0_mirror.log` 已实测：fv 加权热容让 A0 的 assert1（峰值 5.240%→3.969%）转绿，
但 assert2（`Vn=n·dx³` 熔体素计数体积）**11.040%→13.270% 反而更红** ⇒ 形态半边另有机制。
首位候选：`_div_alpha_grad` 用**二值** `solid_mask` 把任何触及非实体单元的面通量**整面**置零
（thermal_enthalpy.py:162-164），而 cut-cell FVM 的正确口径是按**面开口度**（该面被固体占的份额）
开流：内部面 fv=1 与二值掩膜**逐位相同**，只有边界层的面被改 ⇒ 这正是一个纯 O(dx) 的界面机制，
与"加密后熔体积单调增大 0.0519→0.0583→0.0677"同型。

做法（不改生产码）：monkeypatch `TE._div_alpha_grad` 为**本探针重写的同构循环**，面因子可选
    binary  ：`mask_i·mask_j`（＝生产，逐位对照用）
    min     ：`min(fv_i, fv_j)`
    product ：`fv_i·fv_j`
    avg     ：`½(fv_i+f_j)`（预期**错**：与 fv=0 的空白单元开半面通量＝往真空漏热）
再叠加第 4 轮的源/热容 patch（`src/max(fv,f0)`，f0=0.5，第 3 轮 P1 已证 f0 在实体内不参与）。
**N1 用逐位恒等把"重写"这件事本身钉死**：binary 模式必须与生产解逐位相同，否则重写循环
与生产不同构，其余结论全部作废。

夹具＝A0 测试自己那套（`_coupon` δ=0 刀锋、600W/0.8m·s⁻¹/r=100µm/η=0.45、`params={"material":"316L"}`），
另加 `am_t2_a0_alignment.py:38-51` 的整体平移 δ，以覆盖 O1 的"每个 δ"。

跑前登记判据（不许事后改）：
  N1 同构对照：V0=binary（仅重写循环，口径与生产完全相同）与生产解 **逐位相同**
     （peak/Vn/Vm 三项），且重写的 Python 调用计数 >0。任一失败 ⇒ 本轮不参与判据。
  N2 主判据（=O1 + A0 原口径）：某变体若在**每个被测 δ** 上让 **assert1 |Δpeak|<5% 且
     assert2 |ΔVn|<5%** ⇒ 该变体入选；否则不入选。**不许**换观测量（Vm/ΔH 已被 O1 否决）、
     不许放宽 5%、不许挑 δ 或挑档对。
  N3 归因：逐变体报"峰值半边 / 形态半边"各自相对 V0 的改善幅度（哪个机制治哪条），
     使 #23（热容）与 #25（开口度）的分工是**实测**出来的而不是设定的。
  N5（**本轮追加的描述性诊断，不产生入选资格、不改 N2/N4 判据**）：追加第三档 dx=12.5µm，
     用三档反解幂律 f(dx)=f*+C·dx^p，报 peak/Vn/Vm 各自的表观阶 p、外推极限 f* 与
     「让相邻档差恰好 5% 所需的粗档 dx」。目的：把"Vn 不收敛"分解成**离散缺陷**与
     **分辨率预算不够**两种可证伪的解释，而不是含糊地归给"网格太粗"。

  N4 否证：若无任何变体入选 ⇒ 判"面开口度不是 Vn 不收敛的主因"，下一条候选（T(H) 满体素反演 /
     蒸发封顶项）另开打分；**禁止**把两个机制的补丁叠加后调系数凑绿。

范围与边界（照实写）：只跑 A0 的两档 dx∈{50,25}µm（判据就是这一对），δ∈{0,0.25}·dx
（第 2 轮已证 δ=0.5·dx 处实体内 fv≡1 ⇒ 掩膜/热容两个 patch 都在该 δ 恒等，不含信息；但仍跑
V0-vs-生产的 binary 对照以证明"恒等"这件事本身）。avg 模式预期非物理，仍报数不隐藏。GPU。
"""
import math
import time

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from amforge import geometry as G                                          # noqa: E402
from amforge import thermal_enthalpy as TE                                 # noqa: E402
from amforge.core.contracts import ProcessPlan, solid_mask, solid_weight    # noqa: E402
from amforge.materials import get_material                                 # noqa: E402
from amforge.thermal_enthalpy import solve_enthalpy_thermal                # noqa: E402

EXT = (1.2e-3, 0.6e-3, 0.4e-3)
TIERS = [50e-6, 25e-6]                    # N2 判据用的**就是 A0 测试那一对**（不许换档对）
RUN_TIERS = TIERS + [12.5e-6]             # 追加第三档：只为 N5 的表观收敛阶诊断，不参与判据
DELTAS = [0.0, 0.25]
TL = float(get_material("316L").T_liquidus)
F0 = 0.5


def coupon(dx, delta):
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
    return ProcessPlan.uniform(1, modality="SLM", laser_power=600.0, scan_speed=0.8,
                               layer_thickness=EXT[2], hatch_spacing=1.4 * 100e-6,
                               beam_radius=100e-6, absorption=0.45, preheat_temp=400.0)


ORIG_DIV, ORIG_SRC = TE._div_alpha_grad, TE._cell_integrated_source
CALLS = {"div": 0, "src": 0}
STATE = {"face": "binary", "cap": False, "fv": None}


def probe_div(H, alpha, dx, mask=None):
    """与生产 `_div_alpha_grad` **同构**的循环，只把二值面掩膜换成可选开口度。

    生产式（thermal_enthalpy.py:144-171）：face_a = ½(α_i+α_j) · mask_i · mask_j。
    本函数：face_a = ½(α_i+α_j) · FACE(i,j)，FACE 由 STATE["face"] 决定；binary 时
    FACE = mask_i·mask_j ⇒ 与生产逐位相同（N1 的对照就是这一条）。
    """
    CALLS["div"] += 1
    fv = STATE["fv"]
    div = jnp.zeros_like(H)
    for ax in range(H.ndim):
        idx = jnp.arange(H.shape[ax])
        Hc = jnp.take(H, idx[:-1], axis=ax)
        Hn = jnp.take(H, idx[1:], axis=ax)
        ac = jnp.take(alpha, idx[:-1], axis=ax)
        an = jnp.take(alpha, idx[1:], axis=ax)
        face_a = 0.5 * (ac + an)
        if mask is not None:
            mode = STATE["face"]
            if mode == "binary":
                face = jnp.take(mask, idx[:-1], axis=ax) * jnp.take(mask, idx[1:], axis=ax)
            else:
                fi = jnp.take(fv, idx[:-1], axis=ax)
                fj = jnp.take(fv, idx[1:], axis=ax)
                if mode == "min":
                    face = jnp.minimum(fi, fj)
                elif mode == "product":
                    face = fi * fj
                elif mode == "avg":
                    face = 0.5 * (fi + fj)
                else:
                    raise ValueError(f"unknown face mode {mode!r}")
            face_a = face_a * face
        dH = (Hn - Hc) / dx
        flux = face_a * dH
        z = jnp.zeros_like(jnp.take(flux, idx[:1], axis=ax))
        flux_r = jnp.concatenate([flux, z], axis=ax)
        flux_l = jnp.concatenate([z, flux], axis=ax)
        div = div + (flux_r - flux_l) / dx
    if STATE["cap"]:
        div = div / jnp.maximum(fv, F0)
    return div


def probe_src(positions, coords, power, r_src, dp, dx):
    s = ORIG_SRC(positions, coords, power, r_src, dp, dx)
    CALLS["src"] += 1
    return s / jnp.maximum(STATE["fv"], F0) if STATE["cap"] else s


def solve(face, cap, dx, frac):
    g, p = coupon(dx, frac * dx), plan()
    fv = solid_weight(g.sdf, jnp.asarray(dx, dtype=jnp.float64))
    STATE.update(face=face, cap=cap, fv=fv)
    CALLS.update(div=0, src=0)
    TE._div_alpha_grad = ORIG_DIV if (face == "production") else probe_div
    TE._cell_integrated_source = ORIG_SRC if not cap else probe_src
    t0 = time.perf_counter()
    try:
        th = solve_enthalpy_thermal(geometry=g, process=p,
                                    params={"material": "316L"})
    finally:
        TE._div_alpha_grad, TE._cell_integrated_source = ORIG_DIV, ORIG_SRC
    el = time.perf_counter() - t0
    sol = solid_mask(g.sdf) > 0.5
    melted = (th.peak_temperature > TL) & sol
    n = int(jnp.sum(melted))
    pk = float(jnp.max(th.peak_temperature))
    return dict(pk=pk, n=n, vn=n * dx ** 3 * 1e9,
                vm=float(jnp.sum(jnp.where(melted, fv, 0.0))) * dx ** 3 * 1e9,
                el=el, calls=dict(CALLS), nvox=int(g.sdf.size))


VARIANTS = [("production", "binary", False),   # 生产原样（N1 的被对照方）
            ("V0", "binary", False),           # 重写循环，口径＝生产 ⇒ 必须逐位相同
            ("V1", "binary", True),            # 只改热容（#23）
            ("V2", "min", False),              # 只改开口度（#25 首位候选）
            ("V3", "min", True),               # 两者同改（口径自洽的组合）
            ("V4", "product", True),           # 开口度另一种口径
            ("V5", "avg", True)]               # 预期非物理（往真空开半面）
print(f"device = {jax.devices()[0]}", flush=True)
RES = {}
for name, face, cap in VARIANTS:
    for frac in DELTAS:
        for dx in RUN_TIERS:
            r = RES[(name, frac, dx)] = solve(
                "production" if name == "production" else face, cap, dx, frac)
            print(f"[{name:10s} face={face:7s} cap={'on ' if cap else 'off'}] "
                  f"δ={frac:4.2f}·dx dx={dx*1e6:5.1f} nvox={r['nvox']:7d} {r['el']:6.2f}s "
                  f"peak={r['pk']:9.4f} 熔体素={r['n']:6d} Vn={r['vn']:.5f}mm3 "
                  f"Vm={r['vm']:.5f}mm3 calls={r['calls']}", flush=True)

def cap_on(name):
    return {n: c for n, _f, c in VARIANTS}[name]


bad = []
print("\n=== N1 同构对照：V0（重写循环，binary）vs production，逐位 ===")
for frac in DELTAS:
    for dx in RUN_TIERS:
        a, b = RES[("production", frac, dx)], RES[("V0", frac, dx)]
        same = (a["pk"] == b["pk"]) and (a["n"] == b["n"]) and (a["vm"] == b["vm"])
        # N1 登记的是「重写调用计数 >0」。V0 的 cap=off ⇒ 只有 _div_alpha_grad 被换成重写版，
        # 源仍走 ORIG_SRC ⇒ src=0 是**设计如此**，不是"patch 没生效"。因此活路判据按变体取
        # div>0（必要）加上「cap=on 时 src>0」。这是对已登记规则的**正确实现**，不是放宽。
        fired = b["calls"]["div"] > 0 and (b["calls"]["src"] > 0 if cap_on("V0") else True)
        print(f"  δ={frac:4.2f}·dx dx={dx*1e6:5.1f}: 逐位={'YES' if same else 'NO'} "
              f"(peak {a['pk']!r} vs {b['pk']!r}, n {a['n']} vs {b['n']}) "
              f"重写 div 调用={b['calls']['div']}（源走生产实现，cap=off ⇒ src=0 属设计） ⇒ "
              f"{'重写在执行路径上' if fired else '看不见！'}", flush=True)
        if not same:
            bad.append(f"N1 δ={frac}dx dx={dx*1e6}µm：重写循环与生产不逐位相同 ⇒ 不同构，本轮作废")
        if not fired:
            bad.append(f"N1 δ={frac}dx dx={dx*1e6}µm：重写循环未被调用")


def margins(name, frac):
    c, f = RES[(name, frac, TIERS[0])], RES[(name, frac, TIERS[1])]
    d1 = abs(c["pk"] - f["pk"]) / max(c["pk"], f["pk"])
    d2 = abs(c["vn"] - f["vn"]) / max(c["vn"], f["vn"])
    d2v = abs(c["vm"] - f["vm"]) / max(c["vm"], f["vm"])
    return d1, d2, d2v


print("\n=== N1b 每个变体的 patch 活路（div 必查；cap=on 再查 src）===")
for name, face_v, cap_v in VARIANTS:
    if name == "production":
        continue
    before = len(bad)
    for frac in DELTAS:
        for dx in RUN_TIERS:
            c = RES[(name, frac, dx)]["calls"]
            ok = c["div"] > 0 and (c["src"] > 0 if cap_v else True)
            if not ok:
                bad.append(f"N1b {name} δ={frac}dx dx={dx*1e6}µm：调用计数 {c} ⇒ patch 未生效")
    n_groups = len(DELTAS) * len(RUN_TIERS)
    print(f"  {name:10s} face={face_v:7s} cap={'on ' if cap_v else 'off'}: "
          f"{n_groups} 组计数 {'全部达标 ✓' if len(bad) == before else '有失败 ✗'}"
          f"（cap=off 时 src=0 属设计，只查 div）", flush=True)



print("\n=== N2 主判据：每个 δ 上 assert1(|Δpeak|) 与 assert2(|ΔVn|) 是否都 <5% ===")
qualified = []
for name, _, _ in VARIANTS:
    row = [margins(name, frac) for frac in DELTAS]
    ok = all(d1 < 0.05 and d2 < 0.05 for d1, d2, _ in row)
    print(f"  {name:10s} δ=0:    peak {row[0][0]*100:6.3f}%  Vn {row[0][1]*100:6.3f}%  "
          f"(Vm {row[0][2]*100:6.2f}%)   δ=0.25: peak {row[1][0]*100:6.3f}%  "
          f"Vn {row[1][1]*100:6.3f}%  (Vm {row[1][2]*100:6.2f}%)  ⇒ "
          f"{'入选' if ok else '不入选'}", flush=True)
    if ok:
        qualified.append(name)
if bad:
    print("  （N1 未过 ⇒ 不参与判据）")
elif qualified:
    print(f"  ⇒ 入选变体：{qualified}（每个 δ、A0 两条判据同时 <5%）")
else:
    print("  ⇒ N4：无变体入选 ⇒ 面开口度不是 Vn 不收敛的主因；下一条候选另开打分")

print("\n=== N3 归因：各变体相对 V0 的两条余量变化（pt，正=更接近判据）===")
base = {frac: margins("V0", frac) for frac in DELTAS}
for name, _, _ in VARIANTS:
    if name == "production":
        continue
    for frac in DELTAS:
        d1, d2, _d3 = margins(name, frac)
        b1, b2, _b3 = base[frac]
        print(f"  {name:10s} δ={frac:4.2f}·dx: assert1 {(0.05-d1)*100:+7.3f} "
              f"(V0 {(0.05-b1)*100:+7.3f})  assert2 {(0.05-d2)*100:+7.3f} "
              f"(V0 {(0.05-b2)*100:+7.3f})", flush=True)

print("\n=== N5（**追加的描述性诊断，不参与 N2/N4 判据**）：三档表观收敛阶与"
      "「assert2 需要的 dx」===")
print("  模型 f(dx)=f*+C·dx^p 由三档反解：p=log((f1−f2)/(f2−f3))/log2，"
      "f*=f3+(f3−f2)/(2^p−1)；再由 |C|dx^p(1−2^{−p})/|f*|=5% 解出 dx_req。")
for frac in DELTAS:
    for name, _, _ in VARIANTS:
        if name == "production":
            continue
        out = []
        for key, lbl in (("pk", "peak"), ("vn", "Vn  "), ("vm", "Vm  ")):
            f1, f2, f3 = (RES[(name, frac, dx)][key] for dx in RUN_TIERS)
            s12, s23 = f1 - f2, f2 - f3
            if s12 * s23 <= 0 or f3 == f2:
                out.append(f"{lbl}: 非单调/同号⇒ 未进入渐近区(记 p≤0)")
                continue
            p = math.log(abs(s12 / s23)) / math.log(2.0)
            fstar = f3 + s23 / (2.0 ** p - 1.0)
            C = abs(f1 - fstar) / RUN_TIERS[0] ** p
            dx_req = (0.05 * abs(fstar) / (C * (1.0 - 2.0 ** (-p)))) ** (1.0 / p)
            out.append(f"{lbl}: p={p:5.2f} f*={fstar:9.4f} dx_req={dx_req*1e6:6.1f}µm")
        print(f"  δ={frac:4.2f}·dx {name:10s} " + " | ".join(out), flush=True)
print("  读法（dx_req = 让一对相邻档（dx 与 dx/2）相对差恰好等于 5% 的粗档 dx）：\n"
      "    · dx_req ≥ 50µm 而实测 50↔25 仍红 ⇒ 幂律模型解释不了 ⇒ 三档尚未进入渐近区，"
      "找机制而不是加网格；\n"
      "    · dx_req < 50µm ⇒ assert2 是「加密可达」的，但可达分辨率若 ≪ r/2=50µm，说明该判据"
      "实际在要求**亚光斑**分辨率 ⇒ 要么把形态口径换成可积的份额加权体积并**同步**重开 O1 "
      "打分（不许当作放宽），要么承认标定档预算够不着、把 A0 的形态半边挂到更细档位上。")

print(f"\nFAILS: {bad if bad else '无'}")
print("入选变体:", qualified if not bad else "（N1 未过）")
raise SystemExit(1 if bad else 0)
