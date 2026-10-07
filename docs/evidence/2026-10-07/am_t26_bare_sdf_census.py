"""#26 裸 `sdf < 0` 消费位点普查（#19 口径收敛的收尾；fault 17 教训的第二次应用）。

背景：§26.17(c) 登记"命中 30 行/13 文件、本轮改 10 个文件"，但**命中清单 ≠ 普查**——
本探针把树内残留的**裸 `sdf < 0`**（无容差）逐位点量出来：每个夹具在
`sdf<0`（严格）与 `solid_mask = sdf<1e-12`（#19 规范）两种口径下的**实体数差**，
以及两条**断言**在两种口径下的**实际取值**（若相同 ⇒ 换口径是无操作；若不同 ⇒ 换口径
会动判据，必须单独登记）。

判据（跑前登记）：
  S1 每个位点打印 n_strict / n_tol / Δn；Δn=0 ⇒ 该位点换口径是**可证无操作**（照实写）。
  S2 正对照：Δn 必须**至少在一处非零**（否则"刀锋层被两种口径差别对待"这条一直在讲的
     事实在这些夹具上不存在，我前面的表述需要更正）。
  S3 断言取值：`:77` 的重叠和、`:125` 的 leaked 和，两种口径下各自打印；换口径后测试
     仍须为绿（若 :125 的 sup_only 集合变化导致 leaked≠0，那是**真发现**：支撑专用
     体素与容差实体壳重叠，须回改生产码而非测试）。
  S4 `test_enthalpy_thermal.py:96` 的参考求解器（测试自己手写的 rhs，与生产同构）：
     量 Δn 并**实跑**该测试两次（严格/容差）打印 peak_with_L / peak_no_L，确认
     `peak_no_L > peak_with_L+10` 的余量不因换口径而变薄（不许把断言改松）。
"""
import os
import sys

import numpy as np

R = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))          # …/AM-Qoder
sys.path.insert(0, os.path.join(R, "src"))
sys.path.insert(0, os.path.join(R, "tests"))
import jax                                          # noqa: E402

jax.config.update("jax_enable_x64", True)
from amforge.core.contracts import solid_mask       # noqa: E402
from amforge.support import support_auto            # noqa: E402


def counts(sdf):
    a = np.asarray(sdf, dtype=np.float64)
    n_strict = int((a < 0.0).sum())
    n_tol = int(np.asarray(solid_mask(a) > 0.5).sum())
    return n_strict, n_tol, n_tol - n_strict


import test_support as TS            # noqa: E402
import test_boundary as TB           # noqa: E402
import test_enthalpy_thermal as TET  # noqa: E402

SITES = []

# --- tests/test_support.py 的三处（:50 _heated_thermal 的 occ、:77 重叠和、:125 sup_only）
geo_plate = TS._box([0., 0., 2.5e-4], [2e-4, 2e-4, 2.5e-4])
geo_float = TS._box([0., 0., 3.5e-4], [2e-4, 2e-4, 1.5e-4])
for name, g in (("test_support._box(plate)", geo_plate), ("test_support._box(floating)", geo_float)):
    SITES.append((name, g.sdf))
sup = support_auto(geometry=geo_float, params={"kind": "block"})
sm = np.asarray(sup.support_mask, dtype=np.float64)
sdf_f = np.asarray(geo_float.sdf, dtype=np.float64)
strict_part = (sdf_f < 0.0).astype(np.float64)
tol_part = np.asarray(solid_mask(sdf_f), dtype=np.float64)
print("=== S1/S2 逐位点刀锋计数 ===", flush=True)
nz = 0
for name, sdf in SITES:
    a, b, d = counts(sdf)
    nz += (d != 0)
    print(f"  {name:34s} n_strict={a:6d} n_tol={b:6d} Δn={d:+6d} "
          f"⇒ {'换口径=无操作' if d == 0 else '换口径会动集合'}", flush=True)
# 支撑两处断言的实际取值（S3）
ov_strict = float((sm * strict_part).sum())
ov_tol = float((sm * (tol_part > 0.5)).sum())
only_strict = (sm > 0.5) & ~(sdf_f < 0.0)
only_tol = (sm > 0.5) & ~(tol_part > 0.5)
print(f"  :77 重叠和 严格={ov_strict} 容差={ov_tol} ⇒ {'同为 0（无操作）' if ov_strict == ov_tol == 0 else '需查'}",
      flush=True)
print(f"  :125 sup_only 单元数 严格={int(only_strict.sum())} 容差={int(only_tol.sum())} "
      f"⇒ 容差口径下检查集合{'变小' if only_tol.sum() < only_strict.sum() else '不变/变大'}"
      f"（差 {int(only_strict.sum() - only_tol.sum())} 个刀锋单元）", flush=True)
if not (ov_strict == 0.0 and ov_tol == 0.0):
    print("     ⚠ S3：支撑与零件在某一口径下重叠 ⇒ 真发现，须回改生产码", flush=True)

# --- tests/test_boundary.py:156
p_b = TB._box3d(size_mm=0.4, spacing_um=80.0)
SITES.append(("test_boundary._box3d", p_b.sdf))
# --- tests/test_enthalpy_thermal.py:96（参考求解器的 mask）
p_e = TET._part(dx=80e-6)
SITES.append(("test_enthalpy_thermal._part(80µm)", p_e.sdf))
# --- tests/test_gui_preproc.py:27（存在性检查）
from amforge.gui.preproc import build_primitive   # noqa: E402
p_g = build_primitive("sphere", length_mm=1.0, spacing_um=200.0)
SITES.append(("test_gui_preproc build_primitive sphere", p_g.sdf))
for name, sdf in SITES[2:]:
    a, b, d = counts(sdf)
    nz += (d != 0)
    print(f"  {name:40s} n_strict={a:6d} n_tol={b:6d} Δn={d:+6d} "
          f"⇒ {'换口径=无操作' if d == 0 else '换口径会动集合'}", flush=True)
print(f"\n  S2 正对照：非零 Δn 位点数 = {nz} ⇒ {'PASS（刀锋差别在这些夹具上真实存在）' if nz > 0 else 'FAIL：需更正前面所有"刀锋层被容差多算一层"的表述'}",
      flush=True)

# --- S4：参考求解器换口径后的两条峰值（实跑，判据不动）
print("\n=== S4 test_latent_heat_lowers_peak_temperature：严格 vs 容差 mask ===", flush=True)
import jax.numpy as jnp                                # noqa: E402
from amforge.materials import get_material             # noqa: E402
from amforge.thermal_enthalpy import (temperature_of_enthalpy,  # noqa: E402
                                      effective_diffusivity,
                                      _build_scan_positions,
                                      _moving_source, _div_alpha_grad)
m = get_material("316L")
rho, cp, k, L = m.rho_solid, m.cp_solid, m.k_solid, m.latent_fusion
Ts, Tl, Tamb = m.T_solidus, m.T_liquidus, m.T_ambient
dx = 80e-6
plan = TET._plan()
coords = p_e.coords()
for lab, mask, face in (
        ("现状：严格 sdf<0，面无掩膜（＝测试现写法）",
         (p_e.sdf < 0.0).astype(jnp.float64), False),
        ("换容差 solid_mask，面无掩膜",
         (jnp.asarray(solid_mask(np.asarray(p_e.sdf))) > 0.5).astype(jnp.float64), False),
        ("换容差 solid_mask + 面掩膜（＝生产同构）",
         (jnp.asarray(solid_mask(np.asarray(p_e.sdf))) > 0.5).astype(jnp.float64), True)):
    P = float(jnp.mean(plan.laser_power))
    rb = float(jnp.mean(plan.beam_radius))
    eta = float(jnp.mean(plan.absorption))
    dp = max(rb * 1.2, dx)
    Q0 = eta * P / (3.14159265 * rb * rb * (2.0 * 3.14159265) ** 0.5 * dp)
    alpha0 = k / (rho * cp)
    dt = 0.35 * dx * dx / (2.0 * 3 * alpha0)
    pos, _ = _build_scan_positions(coords, plan, dx, n_steps=80, dim=p_e.dim)
    ns = int(pos.shape[0])
    H0 = jnp.zeros_like(mask)

    def rhs(H, t, useL, mask=mask, Q0=Q0, dp=dp, dt=dt, face=face):
        T = temperature_of_enthalpy(H, rho=rho, cp=cp, L=(L if useL else 0.0),
                                    T_amb=Tamb, T_sol=Ts, T_liq=Tl)
        a = effective_diffusivity(T, k=k, rho=rho, cp=cp, L=(L if useL else 0.0),
                                  T_sol=Ts, T_liq=Tl)
        Q = _moving_source(pos[t], coords, Q0, rb, dp) * mask
        lap = (_div_alpha_grad(H, a, dx, mask=mask) if face
               else _div_alpha_grad(H, a, dx))
        return lap + Q - 0.5 * (T - Tamb) * mask

    def run(useL):
        def body(c, t):
            H, peak = c
            T = temperature_of_enthalpy(H, rho=rho, cp=cp, L=(L if useL else 0.0),
                                        T_amb=Tamb, T_sol=Ts, T_liq=Tl)
            k1 = rhs(H, t, useL)
            H1 = H + dt * k1
            k2 = rhs(H1, t, useL)
            Hn = H + 0.5 * dt * (k1 + k2)
            Tn = temperature_of_enthalpy(Hn, rho=rho, cp=cp, L=(L if useL else 0.0),
                                         T_amb=Tamb, T_sol=Ts, T_liq=Tl)
            return (Hn, jnp.maximum(peak, Tn)), None
        (_, peak), _ = jax.lax.scan(body, (H0, jnp.zeros_like(mask)), jnp.arange(ns))
        return peak

    pw, pn = float(jnp.max(run(True))), float(jnp.max(run(False)))
    print(f"  {lab:18s} mask 单元数={int(np.asarray(mask).sum()):6d}  peak_with_L={pw:8.2f}K  "
          f"peak_no_L={pn:8.2f}K  余量={pn - pw:8.2f}K（判据 >10K）⇒ "
          f"{'绿' if pn > pw + 10.0 else '红'}", flush=True)
