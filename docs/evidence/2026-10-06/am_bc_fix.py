"""BC/IC 与动画测试夹具的**测量**脚本（为修测试断言取实数，不猜）。

跑前登记（判据先写，跑后打分）：
  B1 Dirichlet -Z=600K 每步强制覆盖 → 该面终温均值应 = 600.0 ± 2K。
  B2 同夹具不给 BC：该面必然偏离 600K 超过 ±20K（方向由剂量决定，不预设符号；
     旧断言「比 600 低 20K」是步数被人为压小造成的假象）。
  B3 强对流（5 面 h=200/2000 W²m/K）峰值应低于无 BC 峰值，且有限；
     预测 h=200 时降幅 1~15%，h=2000 时降幅 >15%（h/dx/(ρcp) 换算成 1/s 后
     与全局弱冷 hcool=0.5 1/s 相比：200→0.48 1/s，2000→4.8 1/s）。
  B4 预热 IC：激光≈0 时 preheat=373K 与 300K 的实体均值差落在 40~80K（初值差
     73K，弱冷时间常数 2s ≫ 曝光 → 预测 60~75K），且不高于初值。
  B5 动画夹具成本：球半径 0.4mm / dx=80µm 自动 CFL 步数下，单次求解墙钟 < 60s
     且帧数 ≥2、熔池质心跨帧移动 >0.5 体素。
"""
import math
import time

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from amforge.core.contracts import ProcessPlan
from amforge.boundary import BoundaryCondition, BoundaryCollection, face_mask
from amforge.thermal_enthalpy import solve_enthalpy_thermal, suggest_n_steps
from amforge.gui.preproc import build_primitive
from amforge.geometry import from_sdf_fn


def box(size_mm, spacing_um):
    return build_primitive("box", length_mm=size_mm, spacing_um=spacing_um)


def plan(**kw):
    base = dict(laser_power=200.0, scan_speed=1.0, layer_thickness=40e-6,
                hatch_spacing=80e-6, beam_radius=50e-6, absorption=0.4,
                preheat_temp=373.0)
    base.update(kw)
    return ProcessPlan.uniform(4, modality="SLM", **base)


for size_mm, sp_um, pw in [(1.0, 80.0, 200.0), (0.4, 80.0, 200.0)]:
    part = box(size_mm, sp_um)
    pl = plan(laser_power=pw)
    ns = suggest_n_steps(part, pl)
    print(f"\n### 夹具 size={size_mm}mm dx={sp_um}µm  shape={tuple(int(s) for s in part.shape)} "
          f"nvox={int(part.sdf.size)} ns_cfl={ns}", flush=True)
    mb = np.asarray(face_mask(part, "-Z")) > 0.5
    t0 = time.perf_counter()
    rbc = solve_enthalpy_thermal(
        geometry=part, process=pl,
        params={"boundary_conditions": BoundaryCollection(
            bcs=(BoundaryCondition("dirichlet", "-Z", 600.0),), ic=None),
            "material": "316L"})
    el = time.perf_counter() - t0
    Tb = float(np.mean(np.asarray(rbc.final_temperature)[mb]))
    print(f"  [B1] Dirichlet 面均值={Tb:.2f}K（±2K→{'PASS' if abs(Tb-600)<2 else 'FAIL'}）"
          f" 用时 {el:.1f}s", flush=True)
    t0 = time.perf_counter()
    r0 = solve_enthalpy_thermal(geometry=part, process=pl, params={"material": "316L"})
    el0 = time.perf_counter() - t0
    T0 = float(np.mean(np.asarray(r0.final_temperature)[mb]))
    pk0 = float(np.max(np.asarray(r0.peak_temperature)))
    print(f"  [B2] 无 BC 面均值={T0:.2f}K |Δ-600|={abs(T0-600):.1f}K "
          f"{'PASS' if abs(T0-600)>20 else 'FAIL'}；峰值={pk0:.1f}K 用时 {el0:.1f}s", flush=True)
    for h in (200.0, 2000.0):
        strong = BoundaryCollection(
            bcs=tuple(BoundaryCondition("convection", f, h, 293.0)
                      for f in ("+Z", "+X", "-X", "+Y", "-Y")), ic=None)
        rc = solve_enthalpy_thermal(geometry=part, process=pl,
                                    params={"boundary_conditions": strong,
                                            "material": "316L"})
        pkc = float(np.max(np.asarray(rc.peak_temperature)))
        print(f"  [B3] h={h:6.0f} 峰值={pkc:8.1f}K 相对无BC={1-pkc/pk0:7.2%} "
              f"{'PASS' if np.isfinite(pkc) and pkc < pk0 else 'FAIL'}", flush=True)
    # B4 预热 IC（激光≈0）：373K 与 300K 两次求解的实体均值差应 ≈ 初值差 73K
    # （全局弱冷时间常数 1/hcool=2s，曝光 ≪2s → 预测差 60~75K）
    pl0 = plan(laser_power=1e-9)
    rp = solve_enthalpy_thermal(geometry=part, process=pl0,
                                params={"material": "316L",
                                        "n_steps": suggest_n_steps(part, pl0)})
    sld = np.asarray(part.sdf) < -1e-9
    Tp = float(np.mean(np.asarray(rp.final_temperature)[sld]))
    pl1 = plan(laser_power=1e-9, preheat_temp=300.0)
    rp1 = solve_enthalpy_thermal(geometry=part, process=pl1,
                                 params={"material": "316L",
                                         "n_steps": suggest_n_steps(part, pl1)})
    Tp1 = float(np.mean(np.asarray(rp1.final_temperature)[sld]))
    print(f"  [B4] 实体均值 preheat=373→{Tp:.2f}K, =300→{Tp1:.2f}K 差={Tp-Tp1:.1f}K "
          f"{'PASS' if 40.0 < (Tp - Tp1) < 80.0 and Tp <= 373.0 else 'FAIL'}", flush=True)

# B5 动画夹具成本
half, dx = 0.4e-3, 80e-6
ball = from_sdf_fn(lambda x: jnp.linalg.norm(x, axis=-1) - half,
                   bounds=[(-half, half)] * 3, spacing=dx, name="ball")
apl = ProcessPlan.uniform(1, modality="SLM", laser_power=600.0, scan_speed=0.6,
                          beam_radius=60e-6, absorption=0.45,
                          layer_thickness=60e-6, hatch_spacing=120e-6,
                          preheat_temp=300.0)
ns5 = suggest_n_steps(ball, apl)
print(f"\n### 动画夹具 nvox={int(ball.sdf.size)} ns_cfl={ns5}", flush=True)
t0 = time.perf_counter()
out = solve_enthalpy_thermal(geometry=ball, process=apl,
                             params={"record_frames": True, "n_animation_frames": 12})
el5 = time.perf_counter() - t0
evo = np.asarray(out.temperature_evolution)
cen = [np.argwhere(evo[i] > 1723.0).mean(axis=0) for i in range(evo.shape[0])
       if (evo[i] > 1723.0).any()]
move = float(np.linalg.norm(cen[0] - cen[-1])) if len(cen) >= 2 else -1.0
print(f"  [B5] 墙钟={el5:.1f}s 帧={evo.shape[0]} 熔化帧={len(cen)} 质心移动={move:.2f}体素 "
      f"{'PASS' if el5 < 60 and evo.shape[0] >= 2 and move > 0.5 else 'FAIL'}", flush=True)
print("\n完成。", flush=True)
