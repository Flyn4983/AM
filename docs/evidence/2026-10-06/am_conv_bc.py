"""对流 BC 的**可观测效应**测量（为改写 test_convection 取实数；跑前登记预测）。

物理判断（先写）：h=200 W/m²K 的面散热摊到体素是 h/dx=2.5e6 W/m³K，对 ΔT≈2000K
即 5e9 W/m³；而 600W 激光聚在 ~1e-13m³ 里是 ~1e15 W/m³。所以**峰值温度不可能**
被表面对流显著压低（实测也确实 −0.04%），但**扫完之后实体平均温度**应当按
dT/dt ≈ (h/dx)ΔT/(ρcp) 下降：h=200 → ~6K，h=2000 → ~60K，h=20000 → ~数百 K。

预测（判据）：
  C1 峰值差 |Δpeak| < 1%（对流不改峰值——旧断言 peak_c < peak0 在此量级下不成立）
  C2 末态实体均值随 h 单调下降，且 h=2000 相对无 BC 下降 30~120K
  C3 全程有限（无 NaN）
"""
import time

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from amforge.core.contracts import ProcessPlan
from amforge.boundary import BoundaryCondition, BoundaryCollection, face_mask
from amforge.thermal_enthalpy import solve_enthalpy_thermal, suggest_n_steps
from amforge.gui.preproc import build_primitive

part = build_primitive("box", length_mm=0.4, spacing_um=80.0)
pl = ProcessPlan.uniform(4, modality="SLM", laser_power=200.0, scan_speed=1.0,
                         layer_thickness=40e-6, hatch_spacing=80e-6,
                         beam_radius=50e-6, absorption=0.4, preheat_temp=373.0)
ns = suggest_n_steps(part, pl)
solid = np.asarray(part.sdf) < 0.0
print(f"夹具 0.4mm dx=80µm nvox={int(part.sdf.size)} ns={ns}", flush=True)

base = solve_enthalpy_thermal(geometry=part, process=pl, params={"material": "316L"})
Tb = np.asarray(base.final_temperature)
pkb = float(np.max(np.asarray(base.peak_temperature)))
print(f"  无BC   末态实体均值={float(Tb[solid].mean()):8.2f}K 峰值={pkb:8.1f}K", flush=True)
for h in (200.0, 2000.0, 20000.0):
    bcs = BoundaryCollection(
        bcs=tuple(BoundaryCondition("convection", f, h, 293.0)
                  for f in ("+Z", "+X", "-X", "+Y", "-Y")), ic=None)
    t0 = time.perf_counter()
    r = solve_enthalpy_thermal(geometry=part, process=pl,
                               params={"boundary_conditions": bcs, "material": "316L"})
    el = time.perf_counter() - t0
    T = np.asarray(r.final_temperature)
    pk = float(np.max(np.asarray(r.peak_temperature)))
    print(f"  h={h:7.0f} 末态实体均值={float(T[solid].mean()):8.2f}K "
          f"Δ={float(T[solid].mean()) - float(Tb[solid].mean()):+8.2f}K 峰值={pk:8.1f}K "
          f"Δpeak={(pk-pkb)/pkb:+.2%} 有限={bool(np.isfinite(T).all())} {el:.1f}s", flush=True)
print("\n完成。", flush=True)
