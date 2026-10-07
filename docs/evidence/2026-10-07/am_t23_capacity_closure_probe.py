"""#23 实现后的验收探针（2026-10-07 深夜）：结构不变量 ＋ **改前登记的定量预测**是否兑现。

生产改动（`thermal_enthalpy.py`，三处）：焓态量 H 重解读为「单位**材料**体积」的焓 ⇒
右端项整体除以 cut-cell 权重 `cap_w = max(fv, 0.5)`：

    fv·dH/dt = lap + fv·(q − loss)   ⇒   dH/dt = lap/fv + (q − loss)

源/冷却/蒸发在实体内**恰好回到不乘 fv**，而**注入总剂量逐项不变**（`fv·q` 仍在分子里，
除完是 `q`）。本轮三条判据**全部在跑前登记**，不许事后改：

C1 扩散算符不产生/消灭能量（结构不变量）：对随机 H 与生产口径的面因子，
   `Σ_c dx³·lap_c` 必须到机器精度为 **0**（面因子对称 ⇒ 逐对面 telescoping，外边界绝热）。
   归一化分母取 **Σ_c dx³·|lap_c|**（被抵消项的量级），判据 |Σ|/分母 ≤ 1e-12。
   **正对照**：把某一轴向**只有一侧**的面通量乘 1.05（真正破坏"同一对面出现两次、符号相反"）
   ⇒ 同一相对残差必须 ≥ 1e-3，且比零点高 ≥1000 倍。
   没有正对照的 0 不算数（本项目纪律：一个 0 需要一个阳性对照）。
   ⚠ 实现修订登记（2026-10-07，第 2 次跑之后）：上面这条**原本写错了**——第一版把
   `fa`（整条面的因子）乘 1.05，而 `fa` 对该面的**两侧同时**生效 ⇒ telescoping 依然成立
   ⇒ 对照按构造就破坏不了被测不变量（实测：零点 1.28e-10、"对照" 1.28e-9，只差 10 倍，
   且第一版的分母 max|H|·Σmask·dx³ 也不是被抵消项的量级）。判据语义（"必须为 0"＋
   "对照必须看得见"）**不变**，只把对照换成**单侧**缩放、分母换成 Σ|lap|·dx³。
   这是自找缺陷清单里的第 6 条：一个不能失败的对照不是对照。
C2 「热容质量 == 剂量质量」（#23 的结构性陈述）：新口径下参与热容的质量
   `M_cap = Σ fv·dx³·ρ` 与剂量口径的质量 `M_dose` 之比对每一档都必须 = **1.000000**；
   改前的二值口径 `M_old = Σ mask·dx³·ρ` 必须复现已登记的 **+24.574/+12.446/+6.243%**
   （`am_t2_domain_mass.log`，δ=0 三档）⇒ 这条既是修复证明，也是探针看得见缺陷的证明。
   ⚠ `M_dose` 的定义沿用那份证据：`M_dose = Σ (fv·dx³)·ρ`（剂量按份额沉积），
   所以 C2 的新口径一侧是**恒等**——它的信息量全在"旧口径必须仍差 24.6%"这一侧，
   以及 C1 的 telescoping。照实写，不冒充独立验证。
P1 **改前登记的预测**（第 1 轮打分 `am_t25_aperture_probe.log` 的 V1＝"只改热容＋二值面"）：
   新生产解在 A0 夹具上的档对差必须落在 V1 的 **|Δpeak| 3.969%(δ=0)/4.215%(δ=0.25)** 与
   **|ΔVn| 13.270%(δ=0)/11.150%(δ=0.25)** 各 **±0.5 pt** 内（两者与 V1 的差别只在刀锋层/切割壳
   的冷却项少算 2 倍，量级 h·t_exp=3.28e-3 ⇒ 预期差在 1e-2 pt 级）。
   另外**方向**必须成立：|Δpeak| 相对生产改前的 5.240% 下降、|ΔVn| 相对 11.040% **不上降**
   （#25 第 2 轮已实测 Vn 不收敛是"计数口径＋残余场偏差"双因，#23 不治它）。
   ⇒ 若 |ΔVn| 掉到 <5%：**不是好消息**，说明改前打分与本轮实现不一致，必须回头查代码而不是
     当"A0 转绿"上报；若 |Δpeak| ≥5%：说明本轮实现 ≠ V1 口径，同样要查。
P2 峰值 argmax 的归属：`peak` 的 argmax 体素的 `fv` 必须是 **1.000**（实体内部），
   否则切割壳（0<fv<0.5、被二值面掩膜隔断扩散、本轮起带满额源）在污染峰值口径。
   实测依据：第 3 轮 `am_t23_floor_shell_probe.log` L1。
P3 绝对量级换数（不是判据，是**登记**）：本轮改了 δ=0 刀锋层的冷却/蒸发权重与壳体温，
   四档峰值与熔体积的**绝对值**会移动 ⇒ 输出对照表供 §25.2/#16 的档案数字替换。

范围：δ∈{0, 0.25}·dx × dx∈{50,25,12.5}µm，夹具＝A0 测试自己那套（1.2×0.6×0.4mm、
600W/0.8m·s⁻¹/r=100µm/η=0.45/preheat 400K、`params={"material":"316L"}`）。
C1/C2 在 CPU 钉住跑（纯结构核对，不产出性能数字）；P1–P3 在 GPU 跑（用户 10-07 指令：
计算型打分走 A6000，CPU 只作算法与验收通道）。
"""
import time

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from amforge import geometry as G                                            # noqa: E402
from amforge.core.contracts import (                                          # noqa: E402
    ProcessPlan, solid_mask, solid_weight,
)
from amforge.materials import get_material                                   # noqa: E402
from amforge import thermal_enthalpy as TE                                   # noqa: E402
from amforge.thermal_enthalpy import solve_enthalpy_thermal                  # noqa: E402

MAT = get_material("316L")
RHO, CP = float(MAT.rho_solid), float(MAT.cp_solid)
TL = float(MAT.T_liquidus)
EXT = (1.2e-3, 0.6e-3, 0.4e-3)
DELTAS = [0.0, 0.25]                     # δ 以 dx 为单位（夹具注释已固定这个约定）
TIERS = [50e-6, 25e-6]                   # P1 判据用的就是 A0 那一对档
RUN_TIERS = TIERS + [12.5e-6]            # 第三档只为 P3/趋势，不参与 P1
FLOOR = TE.CUT_CAPACITY_FLOOR
bad = []


def coupon(dx, frac):
    """A0 夹具：件尺寸恰为 dx 整数倍时 δ=0 处于刀锋工况（顶面体素中心 sdf==0）。"""
    bx, by, bz = EXT
    d = frac * dx
    bounds = [(-bx / 2 - d, bx / 2 + d), (-by / 2 - d, by / 2 + d),
              (-bz / 2 - d, bz / 2 + d)]

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


# ---------------------------------------------------------------------------
print("\n=== C1 扩散算符 telescoping：Σ_c dx³·lap_c 应为 0（正对照：不对称面因子必须非零）===",
      flush=True)
for dx in [50e-6]:
    g = coupon(dx, 0.0)
    dxa = jnp.asarray(dx, dtype=jnp.float64)
    mask = solid_mask(g.sdf).astype(jnp.float64)
    rng = np.random.default_rng(20261007)
    H = jnp.asarray(rng.normal(3.0e8, 4.0e7, size=g.sdf.shape), dtype=jnp.float64)
    alpha = jnp.full_like(H, 5.0)
    lap = TE._div_alpha_grad(H, alpha, dxa, mask=mask)
    s = float(jnp.sum(lap)) * dxa ** 3
    denom = float(jnp.sum(jnp.abs(lap))) * dxa ** 3      # 被抵消项的量级（不是 max|H|·V）
    rel = abs(s) / max(denom, 1e-30)
    print(f"  dx={dx*1e6:5.1f}µm  生产口径 Σdx³·lap = {s:.6e}  Σdx³·|lap| = {denom:.6e} "
          f"⇒ 相对残差 = {rel:.3e}（判据 ≤1e-12）")
    if rel > 1e-12:
        bad.append(f"C1 dx={dx*1e6}µm：telescoping 残差 {rel:.3e} > 1e-12")

    # 正对照：**只放大一侧**的面通量（第一版乘整条 fa ⇒ 两侧同缩放 ⇒ 抵消照旧 ⇒ 假对照）
    def asym_div(Hh, ah, dxx, mask=None):
        div = jnp.zeros_like(Hh)
        for ax in range(Hh.ndim):
            idx = jnp.arange(Hh.shape[ax])
            Hc = jnp.take(Hh, idx[:-1], axis=ax)
            Hn = jnp.take(Hh, idx[1:], axis=ax)
            ac = jnp.take(ah, idx[:-1], axis=ax)
            an = jnp.take(ah, idx[1:], axis=ax)
            fa = 0.5 * (ac + an)
            if mask is not None:
                fa = fa * jnp.take(mask, idx[:-1], axis=ax) * jnp.take(mask, idx[1:], axis=ax)
            flux = fa * (Hn - Hc) / dxx
            z = jnp.zeros_like(jnp.take(flux, idx[:1], axis=ax))
            flux_r = jnp.concatenate([flux, z], axis=ax)
            flux_l = jnp.concatenate([z, flux], axis=ax)
            if ax == 0:
                flux_r = flux_r * 1.05              # ← 单侧破坏：同一对面不再等量反号
            div = div + (flux_r - flux_l) / dxx
        return div
    lap_ctl = asym_div(H, alpha, dxa, mask=mask)
    s_ctl = float(jnp.sum(lap_ctl)) * dxa ** 3
    denom_ctl = float(jnp.sum(jnp.abs(lap_ctl))) * dxa ** 3
    rel_ctl = abs(s_ctl) / max(denom_ctl, 1e-30)
    print(f"  正对照（x 右流通量 ×1.05）：Σ = {s_ctl:.6e} Σ|lap| = {denom_ctl:.6e} "
          f"⇒ 相对残差 = {rel_ctl:.3e}（判据 ≥1e-3 且 ≥1000× 零点）")
    if rel_ctl < 1e-3:
        bad.append(f"C1 正对照哑火：单侧缩放只给出 {rel_ctl:.3e} <1e-3 ⇒ 这条检查没有效力")
    if rel_ctl < 1000.0 * max(rel, 1e-300):
        bad.append(f"C1 对照与零点只差 {rel_ctl/max(rel,1e-300):.1f} 倍（<1000）⇒ 分不出破坏与噪声")

# ---------------------------------------------------------------------------
print("\n=== C2 热容质量 == 剂量质量（新口径恒等；信息量在旧口径必须仍差 24.6%）===", flush=True)
OLD_DOMAIN_PCT = {0.0: 24.574, 1: 12.446, 2: 6.243}     # am_t2_domain_mass.log δ=0 三档
for i, dx in enumerate(RUN_TIERS):
    g = coupon(dx, 0.0)
    dxa = jnp.asarray(dx, dtype=jnp.float64)
    fv = solid_weight(g.sdf, dxa)
    mk = solid_mask(g.sdf).astype(jnp.float64)
    vol = dxa ** 3
    m_new = float(jnp.sum(fv)) * RHO * vol          # 新口径：cap = Σfv·dx³·ρ
    m_dose = float(jnp.sum(fv)) * RHO * vol         # 剂量口径：同一 Σfv·dx³（定义即如此）
    m_old = float(jnp.sum(mk)) * RHO * vol          # 改前：二值整格热容
    ratio_new = m_new / m_dose
    drift_old = (m_old / m_dose - 1.0) * 100.0
    print(f"  dx={dx*1e6:5.1f}µm  M_cap/M_dose(新)={ratio_new:.12f}  "
          f"M_cap(旧二值)/M_dose={m_old/m_dose:.6f} ⇒ 旧口径偏差 {drift_old:+.3f}% "
          f"（登记 {OLD_DOMAIN_PCT[i]:+.3f}%）")
    if abs(ratio_new - 1.0) > 1e-12:
        bad.append(f"C2 dx={dx*1e6}µm：新口径比值不为 1（{ratio_new!r}）")
    if abs(drift_old - OLD_DOMAIN_PCT[i]) > 0.05:
        bad.append(f"C2 dx={dx*1e6}µm：旧口径偏差复现 {drift_old:.3f}% ≠ 登记 {OLD_DOMAIN_PCT[i]}%")

# ---------------------------------------------------------------------------
import os                                                      # noqa: E402
if os.environ.get("T23_SKIP_SOLVE") == "1":
    # 结构性核对（C1/C2）与求解无关 ⇒ 单独在 CPU 钉住跑一遍以履行 docstring 的范围声明；
    # 求解段 P1–P3 在 GPU 那一次跑。跳过时不产生任何判据结论，只报 C1/C2。
    print(f"\nFAILS: {bad if bad else '无'}（T23_SKIP_SOLVE=1：只跑 C1/C2，P1–P3 未参与）")
    raise SystemExit(1 if bad else 0)

print("\n=== P1/P2/P3：新生产解在 A0 夹具上的档对差（GPU）===", flush=True)
print(f"device = {jax.devices()[0]}", flush=True)
V1_PRED = {("50/25", 0.0): (3.969, 13.270), ("50/25", 0.25): (4.215, 11.150)}
RES = {}
for frac in DELTAS:
    for dx in RUN_TIERS:
        g, p = coupon(dx, frac), plan()
        t0 = time.perf_counter()
        th = solve_enthalpy_thermal(geometry=g, process=p, params={"material": "316L"})
        el = time.perf_counter() - t0
        dxa = jnp.asarray(dx, dtype=jnp.float64)
        fv = solid_weight(g.sdf, dxa)
        sol = solid_mask(g.sdf) > 0.5
        melted = (th.peak_temperature > TL) & sol
        n = int(jnp.sum(melted))
        pk_flat = th.peak_temperature.ravel()
        am = int(jnp.argmax(pk_flat))
        RES[(frac, dx)] = dict(pk=float(pk_flat[am]), n=n, vn=n * dx ** 3 * 1e9,
                               vm=float(jnp.sum(jnp.where(melted, fv, 0.0))) * dx ** 3 * 1e9,
                               argmax_fv=float(fv.ravel()[am]),
                               shell_pk=float(jnp.max(th.peak_temperature * (~sol))),
                               nvox=int(g.sdf.size), el=el)
        r = RES[(frac, dx)]
        print(f"  δ={frac:4.2f}·dx dx={dx*1e6:5.1f} nvox={r['nvox']:8d} {r['el']:6.2f}s  "
              f"peak={r['pk']:9.3f}K 熔体素={r['n']:6d} Vn={r['vn']:.5f}mm³ "
              f"Vm={r['vm']:.5f}mm³  argmax fv={r['argmax_fv']:.4f}  "
              f"壳峰值={r['shell_pk']:9.3f}K", flush=True)

print("\n--- P2 峰值 argmax 归属（必须落在 fv=1 的实体单元）---")
for (frac, dx), r in RES.items():
    ok = abs(r["argmax_fv"] - 1.0) < 1e-9
    print(f"  δ={frac:4.2f}·dx dx={dx*1e6:5.1f}: argmax fv={r['argmax_fv']:.6f} "
          f"{'✓' if ok else '✗ 被切割壳/刀锋层拿走'}  壳峰值 {r['shell_pk']:.1f}K")
    if not ok:
        bad.append(f"P2 δ={frac}dx dx={dx*1e6}µm：peak argmax 的 fv={r['argmax_fv']:.4f}≠1")

print("\n--- P1 主判据：与改前登记的 V1 预测比对（±0.5pt），并核方向 ---")
for frac in DELTAS:
    c, f = RES[(frac, TIERS[0])], RES[(frac, TIERS[1])]
    d1 = abs(c["pk"] - f["pk"]) / max(c["pk"], f["pk"]) * 100.0
    d2 = abs(c["vn"] - f["vn"]) / max(c["vn"], f["vn"]) * 100.0
    p1, p2 = V1_PRED[("50/25", frac)]
    print(f"  δ={frac:4.2f}·dx：|Δpeak|={d1:6.3f}%（预测 {p1} ±0.5）  "
          f"|ΔVn|={d2:6.3f}%（预测 {p2} ±0.5）")
    for lbl, got, pred in (("Δpeak", d1, p1), ("ΔVn", d2, p2)):
        if abs(got - pred) > 0.5:
            bad.append(f"P1 δ={frac}dx：{lbl} 实测 {got:.3f}% 偏离改前预测 {pred}% 超过 0.5pt "
                       f"⇒ 实现≠打分口径，必须回头查码（不许当转绿上报）")
    if d2 < 5.0:
        bad.append(f"P1 δ={frac}dx：|ΔVn|={d2:.3f}% <5% ⇒ 与 #25 第 2 轮的『双因』结论矛盾，"
                   f"先查本轮实现与 V1 的差别，再谈 A0")
    if d1 >= 5.0:
        bad.append(f"P1 δ={frac}dx：|Δpeak|={d1:.3f}% 未达 O1（改前预测 3.97/4.22%）")

print("\n--- P3 绝对量级换数表（供 §25.2 / #16 档案替换；非判据）---")
print(f"  {'δ':>6} {'dx(µm)':>7} {'peak K':>10} {'Vn mm³':>9} {'Vm mm³':>9}")
for frac in DELTAS:
    for dx in RUN_TIERS:
        r = RES[(frac, dx)]
        print(f"  {frac:6.2f} {dx*1e6:7.1f} {r['pk']:10.3f} {r['vn']:9.5f} {r['vm']:9.5f}")
print("  参照（改前生产，δ=0）：peak 50µm=2349.0671619711457 / 25µm=2478.9673107488034 K，"
      "Vn=11.040%（`am_t2_fullsuite.log`、`am_t26_targeted.log` 逐位相同）")

print(f"\nFAILS: {bad if bad else '无'}")
raise SystemExit(1 if bad else 0)
