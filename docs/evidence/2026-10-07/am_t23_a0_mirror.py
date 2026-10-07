"""#23 打分第 4 轮（A0 镜像）：fv 加权热容**能不能**让 A0 的三条断言转绿？逐条报余量。

`tests/test_enthalpy_thermal.py:299-362` 的 A0 断言用的是**熔体素计数体积** `Vn = n·dx³`
（不是 fv 加权体积），而第 2 轮 K4 显示形态量 Vm 在 fv patch 下仍漂 7.5–14.2%。⇒ 必须把
patch 放进**测试自己的口径**里逐条打分，而不是用 peak 一条的改善宣称 A0 会绿。

夹具照抄测试：`_coupon`（bounds=设计盒 ⇒ δ=0 刀锋对齐）+ `_coupon_plan`（600W/0.8m·s⁻¹/
r=100µm/η=0.45/预热 400K）+ `params={"material":"316L"}`（步数由 CFL 自动）+ dx∈{50,25}µm
⇒ 这就是**本测试实际跑的那两个档**，不是探针另选的档。

A = 生产；B = monkeypatch `_div_alpha_grad`/`_cell_integrated_source` 成 `lap/max(fv,f0)`、
`src/max(fv,f0)`（f0=0.5；第 3 轮 P1 已证实体内 fv≥0.5 ⇒ f0 在实体内不参与，故这里 f0 取值
不改变结论）。调用计数继续报出（>0 ⇒ ablation 在执行路径上）。

跑前登记判据（不许事后改）：
  M1 逐条余量：对 A、B 各算测试的三条断言相对判据的**带符号余量**：
     assert1 |Δpeak|/max < 5%，assert2 |ΔVn|/max < 5%，assert3 pk_f > 0.95·pk_c。
  M2 结论分支：
     · B 的 assert1 绿且 assert2 红 ⇒ 判「#23 只解决峰值半边，A0 仍红；形态半边须另立机制
       （面开口度加权 + T(H) 反演口径 + 守恒重验），登记进 #23 描述」——**不许**因此把
       assert2 的判据放宽或换成 Vm（O1 规则 + 「A0 断言不许放宽」）。
     · B 的两条都绿 ⇒ 判「#23 足以让 A0 转绿」，进实现。
     · B 的 assert1 仍红 ⇒ 与第 2 轮 K2（δ=0：3.969%）矛盾 ⇒ 探针/patch 需复核，不宣布任何结论。
  M3 正对照：A 的两条必须复现测试 docstring 里登记的 5.24% / 11.04%（同夹具同口径 ⇒ 应逐位或
     近似复现；差 >0.5 个百分点说明本镜像与测试已不同源）。
"""
import time

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from amforge import geometry as G                                          # noqa: E402
from amforge import thermal_enthalpy as TE                                 # noqa: E402
from amforge.core.contracts import ProcessPlan, solid_mask, solid_weight    # noqa: E402
from amforge.materials import get_material                                 # noqa: E402
from amforge.thermal_enthalpy import solve_enthalpy_thermal                # noqa: E402

EXT = (1.2e-3, 0.6e-3, 0.4e-3)
TIERS = [50e-6, 25e-6]
TL = float(get_material("316L").T_liquidus)


def _coupon(dx, ext=EXT):
    """照抄 tests/test_enthalpy_thermal.py:_coupon。"""
    bx, by, bz = ext
    return G.from_sdf_fn(
        lambda x: jnp.maximum(jnp.maximum(jnp.abs(x[..., 0]) - bx / 2,
                                          jnp.abs(x[..., 1]) - by / 2),
                              jnp.abs(x[..., 2]) - bz / 2),
        bounds=[(-bx / 2, bx / 2), (-by / 2, by / 2), (-bz / 2, bz / 2)],
        spacing=dx, name="coupon")


def _coupon_plan(ext=EXT, power=600.0, radius=100e-6):
    """照抄 tests/test_enthalpy_thermal.py:_coupon_plan。"""
    return ProcessPlan.uniform(1, modality="SLM", laser_power=power, scan_speed=0.8,
                               layer_thickness=ext[2], hatch_spacing=1.4 * radius,
                               beam_radius=radius, absorption=0.45,
                               preheat_temp=400.0)


ORIG_DIV, ORIG_SRC = TE._div_alpha_grad, TE._cell_integrated_source
CALLS = {"div": 0, "src": 0}
STATE = {"mode": "off", "f0": 0.5, "fv": None}


def patched_div(H, alpha, dx, mask=None):
    CALLS["div"] += 1
    lap = ORIG_DIV(H, alpha, dx, mask=mask)
    return lap / jnp.maximum(STATE["fv"], STATE["f0"]) if STATE["mode"] == "fv" else lap


def patched_src(positions, coords, power, r_src, dp, dx):
    s = ORIG_SRC(positions, coords, power, r_src, dp, dx)
    CALLS["src"] += 1
    return s / jnp.maximum(STATE["fv"], STATE["f0"]) if STATE["mode"] == "fv" else s


def solve(mode, dx):
    g, pl = _coupon(dx), _coupon_plan()
    fv = solid_weight(g.sdf, jnp.asarray(dx, dtype=jnp.float64))
    STATE.update(mode=mode, fv=fv)
    CALLS.update(div=0, src=0)
    TE._div_alpha_grad, TE._cell_integrated_source = (
        (ORIG_DIV, ORIG_SRC) if mode == "none" else (patched_div, patched_src))
    t0 = time.perf_counter()
    try:
        th = solve_enthalpy_thermal(geometry=g, process=pl,
                                    params={"material": "316L"})
    finally:
        TE._div_alpha_grad, TE._cell_integrated_source = ORIG_DIV, ORIG_SRC
    el = time.perf_counter() - t0
    melted = (th.peak_temperature > TL) & (solid_mask(g.sdf) > 0.5)
    n = int(jnp.sum(melted))
    return dict(pk=float(jnp.max(th.peak_temperature)), n=n,
                 vn=n * dx ** 3 * 1e9,
                 vm=float(jnp.sum(jnp.where(melted, fv, 0.0))) * dx ** 3 * 1e9,
                 nvox=int(g.sdf.size), calls=dict(CALLS), el=el)


print(f"device = {jax.devices()[0]}", flush=True)
RES = {}
for mode in ("none", "fv"):
    for dx in TIERS:
        r = RES[(mode, dx)] = solve(mode, dx)
        print(f"[{mode:4s}] dx={dx*1e6:5.1f} nvox={r['nvox']:7d} {r['el']:6.2f}s "
              f"peak={r['pk']:9.4f} 熔体素={r['n']:6d} Vn={r['vn']:.5f}mm3 "
              f"Vm={r['vm']:.5f}mm3 calls={r['calls']}", flush=True)

bad = []
print("\n=== M1 逐条余量（相对测试判据 5%；正=通过并留余量，负=超出）===")
for mode in ("none", "fv"):
    c, f = RES[(mode, TIERS[0])], RES[(mode, TIERS[1])]
    d1 = abs(c["pk"] - f["pk"]) / max(c["pk"], f["pk"])
    d2 = abs(c["vn"] - f["vn"]) / max(c["vn"], f["vn"])
    d2v = abs(c["vm"] - f["vm"]) / max(c["vm"], f["vm"])
    a3 = f["pk"] > 0.95 * c["pk"]
    print(f"  {mode:4s}: assert1 |Δpeak|={d1*100:6.3f}% ⇒ 余量 {(0.05-d1)*100:+6.3f}pt "
          f"{'绿' if d1 < 0.05 else '红'}   assert2 |ΔVn|={d2*100:6.3f}% ⇒ 余量 "
          f"{(0.05-d2)*100:+6.3f}pt {'绿' if d2 < 0.05 else '红'}   assert3 加密不变冷 "
          f"{'绿' if a3 else '红'}   （Vm 形态 {d2v*100:.3f}%，信息项）", flush=True)
    RES[(mode, "d1")], RES[(mode, "d2")] = d1, d2
    RES[(mode, "a3")] = a3

cc = RES[("fv", TIERS[0])]["calls"]
if not (cc["div"] > 0 and cc["src"] > 0):
    bad.append(f"M2 fv：patch 调用计数={cc} ⇒ ablation 不在执行路径上，结论作废")

print("\n=== M3 正对照：A 必须复现测试 docstring 登记的 5.24% / 11.04% ===")
a1, a2 = RES[("none", "d1")], RES[("none", "d2")]
print(f"  A: assert1 {a1*100:.3f}%（登记 5.240%）、assert2 {a2*100:.3f}%（登记 11.04%）",
      flush=True)
if abs(a1 * 100 - 5.240) > 0.5 or abs(a2 * 100 - 11.04) > 0.5:
    bad.append(f"M3 失败：A 与测试 docstring 登记值差 >0.5pt（{a1*100:.3f}/{a2*100:.3f}）"
               f" ⇒ 本镜像与测试已不同源，任何结论都作废")

print("\n=== M2 结论分支 ===")
if bad:
    print("  M3/M2 未过 ⇒ 不参与判据：" + "; ".join(bad))
else:
    b1, b2 = RES[("fv", "d1")], RES[("fv", "d2")]
    if b1 < 0.05 and b2 >= 0.05:
        print(f"  ⇒ #23 只解决峰值半边（assert1 {a1*100:.2f}%→{b1*100:.2f}% 转绿；"
              f"assert2 {a2*100:.2f}%→{b2*100:.2f}% "
              f"{'仍红且更差' if b2 > a2 else '仍红'}）⇒ A0 保持红灯，形态半边须另立机制"
              f"（面开口度加权 / T(H) 反演口径 / 守恒重验），不许放宽 assert2 或改用 Vm。")
    elif b1 < 0.05 and b2 < 0.05:
        print("  ⇒ #23 足以让 A0 三条断言全绿，进完整实现。")
    else:
        print(f"  ⇒ assert1 在 patch 下仍 {b1*100:.2f}% ≥5%，与第 2 轮 K2 的 3.969% 矛盾 "
              f"⇒ 复核 patch/夹具，不宣布结论。")
        bad.append("M2：assert1 未转绿，与 K2 矛盾 ⇒ 复核")

print(f"\nFAILS: {bad if bad else '无'}")
raise SystemExit(1 if bad else 0)
