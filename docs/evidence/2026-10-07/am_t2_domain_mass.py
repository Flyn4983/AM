"""域热容 vs 剂量体积：解释 A0 峰值在刀锋对齐下偏低 12% 的机制（纯几何，无求解）。

跑前登记的判据：
  M1 若 `Σ solid_mask·dx³`（域=热容载体）对设计体积的偏差 **明显大于** `Σ fv·dx³`（剂量载体）
     的偏差（同一 δ/dx 下 ≥2 倍），则"域比剂量多了 O(dx) 的热容"成立 ⇒ 峰值偏低可用
     cut-cell 热容按 fv 加权来修（登记为后续任务，不在本轮做）。
  M2 该缺口应随 dx 变细而减小（O(dx) 几何误差的一阶收敛）。
"""
import numpy as np
import jax.numpy as jnp
from amforge.geometry import from_sdf_fn
from amforge.core.contracts import solid_mask, solid_weight

EXT = (1.2e-3, 0.6e-3, 0.4e-3)
V = EXT[0] * EXT[1] * EXT[2]

def coupon(dx, delta):
    bx, by, bz = EXT
    b = [(-bx/2-delta, bx/2+delta), (-by/2-delta, by/2+delta), (-bz/2-delta, bz/2+delta)]
    def fn(x):
        x = np.asarray(x, dtype=np.float64)
        return np.maximum(np.maximum(np.abs(x[...,0])-bx/2, np.abs(x[...,1])-by/2),
                          np.abs(x[...,2])-bz/2)
    return from_sdf_fn(lambda p: fn(np.asarray(p)), bounds=b, spacing=dx, name="c")

fails = []
print("=== M1：域体积(热容) vs 剂量体积(份额) 对设计体积的偏差 ===", flush=True)
prev = {}
for dx_um in (50., 25., 12.5):
    dx = dx_um*1e-6
    for frac in (0., 0.25, 0.5):
        g = coupon(dx, frac*dx)
        s = np.asarray(g.sdf, dtype=np.float64)
        cell = dx**3
        v_dom = float(np.sum(np.asarray(solid_mask(g.sdf)))) * cell
        v_dose = float(np.sum(np.asarray(solid_weight(g.sdf, dx)))) * cell
        old = float((s < 0.0).sum()) * cell
        print(f"  dx={dx_um:5.1f} δ={frac:4.2f}dx  域(mask)={v_dom/V:6.4f}×  "
              f"剂量(fv)={v_dose/V:6.4f}×  旧(sdf<0)={old/V:6.4f}×  "
              f"域−剂量={100*(v_dom/v_dose-1):+6.3f}%", flush=True)
        if frac == 0.:
            prev[dx_um] = (v_dom/v_dose - 1.0, v_dose/V - 1.0)
        if abs(v_dom/v_dose - 1) < 2*abs(v_dose/V - 1) and abs(v_dose/V-1) > 1e-3:
            fails.append(f"M1 dx={dx_um} δ={frac}: 域−剂量差 {v_dom/v_dose-1:+.4f} "
                         f"未达剂量偏差 {v_dose/V-1:+.4f} 的 2 倍")
seq = [abs(prev[d][0]) for d in (50., 25., 12.5)]
print(f"\n  M2 δ=0 的 域/剂量−1 随 dx：{[f'{x*100:+.3f}%' for x in seq]} ⇒ 单调减小 "
      f"{'PASS' if all(seq[i] >= seq[i+1]-1e-6 for i in range(2)) else 'FAIL'}", flush=True)
if not all(seq[i] >= seq[i+1]-1e-6 for i in range(2)):
    fails.append("M2 缺口不随 dx 变细单调减小")
print("\nFAILS:", fails if fails else "无 —— 域比剂量多载 O(dx) 热容成立且随 dx 收敛", flush=True)
