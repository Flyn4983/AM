"""对照 NIST AM-Bench AMB2018-02（裸板 IN625 单道）实测熔池宽/深。

参考数据（AMMT，D4sigma=170 µm -> sigma=42.5 µm -> 1/e^2 半径 r_b=85 µm）：
    Case A  137.9 W  400 mm/s   width 147.9±3.7 µm   depth 42.5±1.8 µm
    Case B  179.2 W  800 mm/s   width 123.5±6.5 µm   depth 36.0±1.9 µm
    Case C  179.2 W 1200 mm/s   width 106.0±1.4 µm   depth 29.6±0.6 µm
来源: NIST CHAL-AMB2018-02-MP-xsection, Table 2 (Lane et al. 2020, IMMI).
"""
import sys
sys.path.insert(0, "src")

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from amforge.core.contracts import ProcessPlan
from amforge.geometry import from_sdf_fn
from amforge.meltpool import solve_meltpool_surrogate

AMBENCH = [
    ("A", 137.9, 0.400, 147.9, 3.7, 42.5, 1.8),
    ("B", 179.2, 0.800, 123.5, 6.5, 36.0, 1.9),
    ("C", 179.2, 1.200, 106.0, 1.4, 29.6, 0.6),
]
RB = 85e-6          # 1/e^2 半径 = 2*sigma = D4sigma/2
T0 = 293.0          # 裸板室温起始（每道间隔 >5 min 回到环境温度）

# 几何在代理模型里只用于占位（单道解析解不依赖构型）
part = from_sdf_fn(lambda p: jnp.linalg.norm(p, axis=-1) - 1e-3,
                   bounds=((0, 4e-4), (0, 4e-4), (0, 4e-4)),
                   spacing=1e-4, name="dummy")


def run(P, v, A, n_grid=56, n_quad=64, c_dep=0.0):
    plan = ProcessPlan.uniform(1, laser_power=P, scan_speed=v, beam_radius=RB,
                               layer_thickness=40e-6, hatch_spacing=100e-6,
                               absorption=A, preheat_temp=T0)
    r = solve_meltpool_surrogate(
        geometry=part, process=plan,
        params={"material": "IN625", "n_grid": n_grid, "n_quad": n_quad,
                "props": "mean", "depression_coeff": c_dep})
    return (float(r.width) * 1e6, float(r.depth) * 1e6,
            float(r.length) * 1e6, float(r.keyhole_depth) * 1e6,
            float(jnp.max(r.temperature)))


print("=" * 92)
print("扫描吸收率 A —— 关掉凹陷项 (c_dep=0)，因三组均为传导模式 (d/w≈0.28)")
print("=" * 92)
best = None
for A in (0.25, 0.30, 0.35, 0.39, 0.45, 0.50, 0.60):
    errs_w, errs_d = [], []
    rows = []
    for cid, P, v, wr, wu, dr, du in AMBENCH:
        w, d, l, kh, Tm = run(P, v, A)
        errs_w.append(w / wr - 1)
        errs_d.append(d / dr - 1)
        rows.append((cid, w, wr, d, dr, l, Tm))
    rms = np.sqrt(np.mean(np.array(errs_w + errs_d) ** 2))
    print(f"\nA = {A:.2f}   RMS 相对误差 {rms*100:5.1f}%")
    for (cid, w, wr, d, dr, l, Tm) in rows:
        print(f"   Case {cid}: 宽 {w:6.1f} (实测 {wr:5.1f}, {w/wr-1:+6.1%})   "
              f"深 {d:5.1f} (实测 {dr:4.1f}, {d/dr-1:+6.1%})   "
              f"长 {l:6.1f}   T_max {Tm:7.0f} K")
    if best is None or rms < best[0]:
        best = (rms, A)

print()
print("=" * 92)
print(f"最优吸收率 A = {best[1]:.2f}，RMS {best[0]*100:.1f}%")
print("=" * 92)

# 细扫
print("\n细扫 A：")
for A in np.arange(0.30, 0.56, 0.02):
    errs = []
    for cid, P, v, wr, wu, dr, du in AMBENCH:
        w, d, _, _, _ = run(P, v, float(A))
        errs += [w / wr - 1, d / dr - 1]
    print(f"  A={A:.2f}  RMS {np.sqrt(np.mean(np.array(errs)**2))*100:5.2f}%  "
          f"宽偏差 {np.mean(errs[0::2])*100:+6.2f}%  "
          f"深偏差 {np.mean(errs[1::2])*100:+6.2f}%")
