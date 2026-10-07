"""A0 断言 `test_meltpool_converges_across_beam_resolving_grids` 在 #19 之后由绿转红
（50↔25µm 峰值差 1.96% → 5.25%，`am_t2_targeted.log`）。本探针判定它的**根因**：
是"#19 把实体掩膜改成带 ulp 容差后粗档物理变差"，还是"这条断言一直在测体素对齐
抖动，而不是网格收敛"。

夹具与测试完全同源：试片 1.2×0.6×0.4 mm、600W/0.8m·s⁻¹/r=100µm/η=0.45、
integrated 源、n_steps 由 CFL 自动推导。

做法：把设计盒相对体素栅格沿三轴整体平移 δ ∈ {0, 0.25dx, 0.5dx}。
δ=0 时**所有六个面都恰好落在体素中心上**（bz/2=200µm、dx∈{50,25} 整除）⇒ 刀锋集
非空，`sdf<0`（#19 前）与 `sdf<1e-12`（#19 后）差一整层皮；δ=0.25dx 时无刀锋 ⇒
两种掩膜**必然给同一个体**。

跑前登记的证伪判据：
  Q1 对齐抖动 J(dx) = (max_δ − min_δ)/max_δ 的峰值。
     · 若 J(50µm) ≥ 5% ⇒ 50↔25 的 5% 断言**量的是对齐抖动**（同档位相差 5% 与
       跨档差 5% 不可分），#19 不是"弄坏了收敛"；此时必须把 A0 断言升级为
       "同档抖动 + 跨档收敛"两条，并登记 fv 加权热容为后续项。
     · 若 J(50µm) < 2% ⇒ 抖动解释不了 5.25% ⇒ 是掩膜口径改变真实地动了粗档物理，
       #19 在求解器域掩膜这一处的落地方式需要回退或换 fv>0 域。
  Q2 δ=0 应落在 δ 族**之内**（不是离群点）：|peak(0) − median(peak(0.25dx),peak(0.5dx))|
     / median ≤ max(J, 5%)。若 δ=0 是离群点 ⇒ 容差方向选错了（该用 fv>0 或 fv≥0.5）。
  Q3 剂量口径不受影响：Σfv·dx³ 对设计体积的偏差随 dx 变细单调减小，且两档都 <2%
     （这是 `am_t2_prod_paths.py` P2 的同一句话，在真实求解夹具上复核一遍）。
"""
import numpy as np

import jax.numpy as jnp

from amforge.geometry import from_sdf_fn
from amforge.materials import get_material
from amforge.core.contracts import ProcessPlan, solid_weight
from amforge.thermal_enthalpy import solve_enthalpy_thermal

EXT = (1.2e-3, 0.6e-3, 0.4e-3)


def coupon(dx, delta):
    """设计盒不动、栅格整体平移 delta：等价于"零件相对体素错开 delta"。"""
    bx, by, bz = EXT
    bounds = [(-bx / 2 - delta, bx / 2 + delta),
              (-by / 2 - delta, by / 2 + delta),
              (-bz / 2 - delta, bz / 2 + delta)]

    def fn(x):
        return jnp.maximum(jnp.maximum(jnp.abs(x[..., 0]) - bx / 2,
                                        jnp.abs(x[..., 1]) - by / 2),
                            jnp.abs(x[..., 2]) - bz / 2)

    return from_sdf_fn(lambda x: fn(np.asarray(x, dtype=jnp.float64)),
                       bounds=bounds, spacing=dx, name="coupon")


def coupon_plan():
    return ProcessPlan.uniform(1, modality="SLM", laser_power=600.0, scan_speed=0.8,
                               layer_thickness=EXT[2], hatch_spacing=1.4 * 100e-6,
                               beam_radius=100e-6, absorption=0.45,
                               preheat_temp=400.0)


V_DESIGN = EXT[0] * EXT[1] * EXT[2]
fails = []
print("=== Q1/Q2：同档不同体素对齐的峰值抖动（#19 后的容差掩膜） ===", flush=True)
peaks = {}
for dx_um in (50.0, 25.0):
    dx = dx_um * 1e-6
    row = {}
    for frac in (0.0, 0.25, 0.5):
        g = coupon(dx, frac * dx)
        th = solve_enthalpy_thermal(geometry=g, process=coupon_plan(),
                                    params={"material": "316L"})
        pk = float(jnp.max(th.peak_temperature))
        n_tie = int(jnp.sum(jnp.abs(np.asarray(g.sdf, dtype=np.float64)) < 1e-12))
        fv = float(jnp.sum(solid_weight(g.sdf, g.spacing))) * dx ** 3
        row[frac] = (pk, n_tie, fv / V_DESIGN - 1.0)
        print(f"  dx={dx_um:5.1f} µm  δ={frac:4.2f}·dx  峰值={pk:8.2f} K  "
              f"刀锋体素数={n_tie:7d}  Σfv·dx³/V设计−1={row[frac][2] * 100:+6.3f}%  "
              f"形状={tuple(int(s) for s in g.shape)}", flush=True)
    vals = [row[f][0] for f in (0.0, 0.25, 0.5)]
    jit = (max(vals) - min(vals)) / max(vals)
    peaks[dx_um] = row
    med = float(np.median([row[0.25][0], row[0.5][0]]))
    dev0 = abs(row[0.0][0] - med) / max(abs(med), row[0.0][0])
    print(f"  ⇒ J({dx_um}µm) = {jit * 100:.2f}%   δ=0 相对 δ 族中位数的偏离 = "
          f"{dev0 * 100:.2f}%", flush=True)
    if dx_um == 50.0:
        if jit >= 0.05:
            print("     Q1 分支 A：抖动 ≥5% ⇒ 50↔25 的 5% 断言量的是对齐，不是收敛",
                  flush=True)
        elif jit < 0.02:
            fails.append(f"Q1 分支 B：J(50µm)={jit * 100:.2f}% <2%，"
                         f"5.25% 的跨档差不是对齐抖动 ⇒ #19 的域掩膜改动动了粗档物理")
    if dev0 > max(jit, 0.05):
        fails.append(f"Q2 dx={dx_um}：δ=0 是离群点（偏离 {dev0 * 100:.2f}% > "
                     f"max(J,5%)={max(jit, 0.05) * 100:.2f}%）⇒ 容差方向选错")

print("\n=== Q3：剂量口径（Σfv·dx³ 对设计体积）随 dx 单调收敛 ===", flush=True)
b = [abs(peaks[d][0.0][2]) for d in (50.0, 25.0)]
print(f"  dx=50/25 µm 偏差 = {peaks[50.][0.0][2] * 100:+.3f}% / "
      f"{peaks[25.][0.0][2] * 100:+.3f}% ⇒ 单调减小 "
      f"{'PASS' if b[0] >= b[1] - 1e-3 else 'FAIL'}", flush=True)
if not (b[0] >= b[1] - 1e-3):
    fails.append("Q3：Σfv 偏差不随 dx 变细单调减小")
if b[1] >= 0.02:
    fails.append(f"Q3：dx=25µm 剂量偏差 {b[1] * 100:.3f}% ≥2%")

print("\n=== 对照：把 A0 断言的两档在 δ=0.25dx（无刀锋）下重跑一遍 ===", flush=True)
pk_c = peaks[50.][0.25][0]
pk_f = peaks[25.][0.25][0]
rel = abs(pk_c - pk_f) / max(pk_c, pk_f)
print(f"  无刀锋对齐：50µm 峰值={pk_c:.2f}K  25µm 峰值={pk_f:.2f}K  差={rel * 100:.2f}% "
      f"⇒ {'<5% A0 条在同' if rel < 0.05 else '仍 ≥5%：跨档收敛本身没达标'}"
      f"（与掩膜无关的档位）成立", flush=True)

print("\n=== 记分 ===", flush=True)
print("FAILS:", fails if fails else "无", flush=True)
print("注：本探针不判 A0 断言本身红绿，只判它的**根因归属**。", flush=True)
