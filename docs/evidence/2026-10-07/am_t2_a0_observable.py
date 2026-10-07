"""A0 的"两档网格给出同一熔池"到底该用**哪个观测量**？（#19 之后的重扫）

`am_t2_a0_alignment.log` 实测：峰值 T 在同一档网格上随体素对齐抖动
J(50µm)=3.46%、J(25µm)=3.36%（**不随 dx 衰减**），而 50↔25 的跨档峰值差在三种对齐下
分别是 5.25% / 4.73% / 5.15% ⇒ "峰值差 <5%" 这条 A0 断言落在对齐抖动带里，
它的绿灯与否取决于对齐而非收敛。本脚本不改求解器，只测**哪些观测量真的跨档稳定**：

  A1 峰值 T           —— 预期不入选（上面已证伪）
  A2 熔化体积（fv 加权）Vm = Σ_{T_peak>T_liq} fv·dx³  —— #19 口径下的熔化体积
  A3 熔化体积（体素计数）Vn = #{T_peak>T_liq 且 mask}·dx³ —— 旧口径（docstring 已登记
     "约 4%/加倍 漂移"），作正对照：若它也入选，说明那条历史记录未复现
  A4 焓升 ΔH = Σ fv·(H(T_final) − H(T_preheat))·dx³ —— 剂量守恒的下游积分量

跑前登记的入选判据 O1：观测量 A 入选 ⇔ 对**每一种对齐 δ** 都有
  |A(50,δ) − A(25,δ)|/max < 5% 且 |A(25,δ) − A(12.5,δ)|/max < 5%。
即：不许挑对齐、不许挑档对。

O2（决策）：A2 与 A4 同时入选 ⇒ A0 的"同一熔池"改由它们承载，峰值 T 的对齐敏感性
   连同"域热容未按 fv 加权"（`am_t2_domain_mass.log`：刀锋对齐下域比剂量多载
   +24.6%/+12.4%/+6.2% 的体积）一起登记为后续任务 #23。
   若 A2 与 A4 都不入选 ⇒ 本轮没有可靠的收敛观测量，A0 判据保持红灯，不许换指标。
"""
import numpy as np

import jax.numpy as jnp

from amforge.geometry import from_sdf_fn
from amforge.core.contracts import ProcessPlan, solid_mask, solid_weight
from amforge.materials import get_material
from amforge.thermal_enthalpy import enthalpy_of_temperature, solve_enthalpy_thermal

EXT = (1.2e-3, 0.6e-3, 0.4e-3)
T_PRE = 400.0
OBS = ("pk", "Vm", "Vn", "dH")


def coupon(dx, delta):
    bx, by, bz = EXT
    b = [(-bx / 2 - delta, bx / 2 + delta), (-by / 2 - delta, by / 2 + delta),
         (-bz / 2 - delta, bz / 2 + delta)]

    def fn(p):
        x = np.asarray(p, dtype=np.float64)
        return np.maximum(np.maximum(np.abs(x[..., 0]) - bx / 2,
                                      np.abs(x[..., 1]) - by / 2),
                          np.abs(x[..., 2]) - bz / 2)

    return from_sdf_fn(fn, bounds=b, spacing=dx, name="coupon")


def plan():
    return ProcessPlan.uniform(1, modality="SLM", laser_power=600.0, scan_speed=0.8,
                               layer_thickness=EXT[2], hatch_spacing=1.4 * 100e-6,
                               beam_radius=100e-6, absorption=0.45,
                               preheat_temp=T_PRE)


mat = get_material("316L")
Tl, Ts, Ta = float(mat.T_liquidus), float(mat.T_solidus), float(mat.T_ambient)
rho, cp, Lk = float(mat.rho_solid), float(mat.cp_solid), float(mat.latent_fusion)

data = {}
print("=== 逐 (dx, δ) 采集 4 个观测量 ===", flush=True)
for dx_um, fracs in ((50., (0., .25, .5)), (25., (0., .25, .5)), (12.5, (0., .25))):
    dx = dx_um * 1e-6
    cell = dx ** 3
    for frac in fracs:
        g = coupon(dx, frac * dx)
        th = solve_enthalpy_thermal(geometry=g, process=plan(),
                                    params={"material": "316L"})
        fv = np.asarray(solid_weight(g.sdf, dx), dtype=np.float64)
        mk = np.asarray(solid_mask(g.sdf), dtype=np.float64)
        pkf = np.asarray(th.peak_temperature, dtype=np.float64)
        fin = np.asarray(th.final_temperature, dtype=np.float64)
        H_fin = np.asarray(enthalpy_of_temperature(fin, rho=rho, cp=cp, L=Lk,
                                                   T_amb=Ta, T_sol=Ts, T_liq=Tl),
                           dtype=np.float64)
        H_pre = float(enthalpy_of_temperature(np.float64(T_PRE), rho=rho, cp=cp,
                                              L=Lk, T_amb=Ta, T_sol=Ts, T_liq=Tl))
        liq = pkf > Tl
        rec = dict(pk=float(pkf.max()),
                   Vm=float(np.sum(np.where(liq, fv, 0.0))) * cell * 1e9,
                   Vn=float(np.sum(liq & (mk > 0.5))) * cell * 1e9,
                   dH=float(np.sum(fv * (H_fin - H_pre))) * cell)
        data[(dx_um, frac)] = rec
        print(f"  dx={dx_um:5.1f} δ={frac:4.2f}  峰值={rec['pk']:8.2f}K  "
              f"Vm(fv)={rec['Vm']:8.5f}mm³  Vn(计数)={rec['Vn']:8.5f}mm³  "
              f"ΔH={rec['dH']:.5e}J", flush=True)

print("\n=== O1：跨档相对差（对每一种 δ 都必须 <5% 才算入选） ===", flush=True)
pairs = [(50., 25.), (25., 12.5)]
qualified = {}
for name in OBS:
    worst = {}
    for (a, b) in pairs:
        rels = []
        for frac in (0., .25):
            if (a, frac) in data and (b, frac) in data:
                va, vb = data[(a, frac)][name], data[(b, frac)][name]
                rels.append(abs(va - vb) / max(abs(va), abs(vb), 1e-30))
        worst[(a, b)] = max(rels) if rels else float("nan")
    # δ=0.5 只在前两档有
    va, vb = data[(50., .5)][name], data[(25., .5)][name]
    r05 = abs(va - vb) / max(abs(va), abs(vb))
    ok = all(v < 0.05 for v in worst.values()) and r05 < 0.05
    qualified[name] = ok
    print(f"  {name:3s} max over δ∈{{0,.25}}:  50↔25 {worst[(50., 25.)] * 100:6.2f}%   "
          f"25↔12.5 {worst[(25., 12.5)] * 100:6.2f}%   |  δ=0.5 档对 {r05 * 100:6.2f}%"
          f"  ⇒ {'入选' if ok else '不入选'}", flush=True)

print("\n=== O2：决策 ===", flush=True)
fails = []
if qualified["Vm"] and qualified["dH"]:
    print("  A2+A4 入选 ⇒ '同一熔池'由 fv 加权熔化体积 + 焓升承载；峰值 T 的对齐敏感性"
          "登记为 #23（连同域热容未按 fv 加权）", flush=True)
elif not (qualified["Vm"] or qualified["dH"]):
    fails.append("A2/A4 均不入选 ⇒ 没有可靠收敛观测量，A0 判据必须保持红灯，不许换指标")
    print("  A2/A4 都不入选 ⇒ 保持红灯", flush=True)
else:
    print(f"  只入选一个 {qualified} ⇒ 决策：以更接近 5% 裕度者承载，另一项登记缺口",
          flush=True)
if qualified["Vn"]:
    print("  ⚠ 正对照未复现：旧体素计数熔体积也入选 ⇒ 原登记'约 4%/加倍漂移'需回看",
          flush=True)
print(f"\nFAILS: {fails if fails else '无（见上方决策行）'}", flush=True)
