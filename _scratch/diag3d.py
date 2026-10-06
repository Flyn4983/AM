"""三参数诊断：(吸收率 A, 阈值 T_thr, 扩散率倍率 fa) 能否消掉速度趋势？

若最优点仍残留"宽超预测随速度增、深欠预测随速度增"，则确认为
纯导热常物性模型的结构性精度上限，应如实记录而非继续加拟合系数。
"""
import sys
sys.path.insert(0, "src")

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from amforge.materials import get_material
from amforge.meltpool import eagar_tsai_field

AMBENCH = [("A", 137.9, 0.400, 147.9, 42.5),
           ("B", 179.2, 0.800, 123.5, 36.0),
           ("C", 179.2, 1.200, 106.0, 29.6)]
RB, T0 = 85e-6, 293.0
m = get_material("IN625")
rho = 0.5 * (m.rho_solid + m.rho_liquid)
cp = 0.5 * (m.cp_solid + m.cp_liquid)
rho_cp = rho * cp
alpha0 = 0.5 * (m.k_solid + m.k_liquid) / rho_cp

N = 320
XS = jnp.linspace(-6 * RB, 1.0 * RB, 140)


def dims(P, v, A, T_thr, alpha):
    """返回 (width, depth) [µm]，直接从原始场提等值面（线性插值）。"""
    out = []
    for axis in ("y", "z"):
        s = jnp.linspace(0.0, 4 * RB, N)
        G, S = jnp.meshgrid(XS, s, indexing="ij")
        zeros = jnp.zeros_like(G)
        Y = S if axis == "y" else zeros
        Z = zeros if axis == "y" else -S
        T = eagar_tsai_field(absorbed_power=A * P, scan_speed=v,
                             beam_radius=RB, T0=T0, alpha=alpha,
                             rho_cp=rho_cp, X=G[..., None], Y=Y[..., None],
                             Z=Z[..., None], n_quad=96)[..., 0]
        above = T >= T_thr
        idx = jnp.sum(above, axis=1)
        i0 = jnp.clip(idx - 1, 0, N - 1)
        i1 = jnp.clip(idx, 0, N - 1)
        Ta = jnp.take_along_axis(T, i0[:, None], 1)[:, 0]
        Tb = jnp.take_along_axis(T, i1[:, None], 1)[:, 0]
        frac = jnp.where(jnp.abs(Ta - Tb) > 1e-12, (Ta - T_thr) / (Ta - Tb), 0.0)
        sb = jnp.where(idx > 0, s[i0] + frac * (s[i1] - s[i0]), 0.0)
        out.append(float(jnp.max(sb)) * 1e6)
    return 2.0 * out[0], out[1]


records = []
for fa in (0.55, 0.7, 0.85, 1.0, 1.2, 1.5):
    for Tt in (1563.0, 1782.0, 2002.0, 2400.0, 2800.0):
        for A in np.arange(0.20, 0.90, 0.05):
            errs = []
            for cid, P, v, wr, dr in AMBENCH:
                w, d = dims(P, v, float(A), Tt, fa * alpha0)
                errs += [w / wr - 1, d / dr - 1]
            records.append((np.sqrt(np.mean(np.array(errs) ** 2)),
                            fa, Tt, float(A), errs))

records.sort(key=lambda r: r[0])
print(f"alpha0 = {alpha0:.4e} m^2/s")
print("=" * 96)
print("RMS 最小的 8 个参数组合")
print("=" * 96)
print(f"{'RMS%':>6} {'fa':>5} {'T_thr':>7} {'A':>5} | "
      f"{'A宽':>7} {'A深':>7} {'B宽':>7} {'B深':>7} {'C宽':>7} {'C深':>7}")
for rms, fa, Tt, A, errs in records[:8]:
    print(f"{rms*100:6.2f} {fa:5.2f} {Tt:7.0f} {A:5.2f} | " +
          " ".join(f"{e:+7.1%}" for e in errs))

print()
print("最优组合的深宽比对照：")
rms, fa, Tt, A, errs = records[0]
for cid, P, v, wr, dr in AMBENCH:
    w, d = dims(P, v, A, Tt, fa * alpha0)
    print(f"  Case {cid} (v={v*1000:4.0f} mm/s): 实测 d/w={dr/wr:.3f}  模型 d/w={d/w:.3f}")

# 单独看：能否只靠 fa 让 d/w 趋势变平？
print()
print("=" * 96)
print("d/w 随速度的趋势 vs fa（固定 A=0.4, T_thr=T_solidus）")
print("=" * 96)
print(f"{'fa':>6} {'alpha':>11} | {'A d/w':>7} {'B d/w':>7} {'C d/w':>7} {'斜率(C-A)':>10}")
for fa in (0.4, 0.55, 0.7, 0.85, 1.0, 1.3, 1.8, 2.5):
    rs = []
    for cid, P, v, wr, dr in AMBENCH:
        w, d = dims(P, v, 0.4, 1563.0, fa * alpha0)
        rs.append(d / w)
    print(f"{fa:6.2f} {fa*alpha0:11.3e} | {rs[0]:7.3f} {rs[1]:7.3f} {rs[2]:7.3f} "
          f"{rs[2]-rs[0]:+10.3f}")
print(f"{'实测':>6} {'':>11} | {42.5/147.9:7.3f} {36.0/123.5:7.3f} "
      f"{29.6/106.0:7.3f} {29.6/106.0-42.5/147.9:+10.3f}")
