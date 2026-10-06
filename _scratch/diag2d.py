"""诊断：纯导热 Eagar-Tsai 场能否同时拟合 AM-Bench 三组的宽与深？

在 (吸收率 A, 熔化阈值 T_thr) 平面上直接从原始温度场读等值面尺寸，
绕开求解器的饱和/凹陷/软指示等全部后处理，只考察**纯导热场本身**的
几何相似性。若最优点残差仍带速度趋势，则是结构性物理缺失而非标定问题。
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
k = 0.5 * (m.k_solid + m.k_liquid)
rho = 0.5 * (m.rho_solid + m.rho_liquid)
cp = 0.5 * (m.cp_solid + m.cp_liquid)
rho_cp, alpha = rho * cp, k / (rho * cp)
print(f"IN625 mean: k={k:.3g} rho={rho:.4g} cp={cp:.4g} "
      f"alpha={alpha:.4e} m^2/s  T_s={m.T_solidus} L_f/cp={m.latent_fusion/cp:.0f} K")

# 高分辨等值面提取：沿 y 轴（宽）与 z 轴（深）一维加密二分
N = 400


def half_width(P, v, A, T_thr):
    """在 y 方向找 T=T_thr 的位置（x=z=0 平面上取最大宽度处）。"""
    # 先在 x 上找最热的横截面位置（对宽度，最宽处在光斑略后）
    xs = jnp.linspace(-6 * RB, 1.0 * RB, 160)
    ys = jnp.linspace(0.0, 4 * RB, N)
    X, Y = jnp.meshgrid(xs, ys, indexing="ij")
    Z = jnp.zeros_like(X)
    T = eagar_tsai_field(absorbed_power=A * P, scan_speed=v, beam_radius=RB,
                         T0=T0, alpha=alpha, rho_cp=rho_cp,
                         X=X[..., None], Y=Y[..., None], Z=Z[..., None],
                         n_quad=96)[..., 0]
    above = T >= T_thr                              # (nx, ny)
    # 每个 x 截面的半宽 = 最后一个 above 的 y（线性插值）
    idx = jnp.sum(above, axis=1)                    # 连续区间长度
    ok = idx > 0
    yv = ys
    # 线性插值边界
    i0 = jnp.clip(idx - 1, 0, N - 1)
    i1 = jnp.clip(idx, 0, N - 1)
    T0v = jnp.take_along_axis(T, i0[:, None], 1)[:, 0]
    T1v = jnp.take_along_axis(T, i1[:, None], 1)[:, 0]
    y0 = yv[i0]
    y1 = yv[i1]
    frac = jnp.where(jnp.abs(T0v - T1v) > 1e-12, (T0v - T_thr) / (T0v - T1v), 0.0)
    yb = jnp.where(ok, y0 + frac * (y1 - y0), 0.0)
    return float(jnp.max(yb))


def depth(P, v, A, T_thr):
    """在 x 上扫描、z 方向找 T=T_thr 的最深位置（y=0）。"""
    xs = jnp.linspace(-6 * RB, 1.0 * RB, 160)
    zs = jnp.linspace(0.0, -4 * RB, N)
    X, Z = jnp.meshgrid(xs, zs, indexing="ij")
    Y = jnp.zeros_like(X)
    T = eagar_tsai_field(absorbed_power=A * P, scan_speed=v, beam_radius=RB,
                         T0=T0, alpha=alpha, rho_cp=rho_cp,
                         X=X[..., None], Y=Y[..., None], Z=Z[..., None],
                         n_quad=96)[..., 0]
    above = T >= T_thr
    idx = jnp.sum(above, axis=1)
    ok = idx > 0
    i0 = jnp.clip(idx - 1, 0, N - 1)
    i1 = jnp.clip(idx, 0, N - 1)
    T0v = jnp.take_along_axis(T, i0[:, None], 1)[:, 0]
    T1v = jnp.take_along_axis(T, i1[:, None], 1)[:, 0]
    z0, z1 = zs[i0], zs[i1]
    frac = jnp.where(jnp.abs(T0v - T1v) > 1e-12, (T0v - T_thr) / (T0v - T1v), 0.0)
    zb = jnp.where(ok, z0 + frac * (z1 - z0), 0.0)
    return float(-jnp.min(zb))


print()
print("=" * 100)
print("(A, T_thr) 网格上的 RMS 相对误差 [%]，及最优点残差趋势")
print("=" * 100)
A_grid = np.arange(0.20, 0.85, 0.05)
Tthr_grid = np.array([m.T_solidus, m.T_liquidus,
                      m.T_solidus + 0.5 * m.latent_fusion / cp,
                      m.T_solidus + m.latent_fusion / cp,
                      m.T_solidus + 1.5 * m.latent_fusion / cp,
                      2400.0, 2700.0])
print(f"{'T_thr':>8} " + " ".join(f"{a:6.2f}" for a in A_grid))
results = {}
for Tt in Tthr_grid:
    row = []
    for A in A_grid:
        errs = []
        for cid, P, v, wr, dr in AMBENCH:
            w = 2e6 * half_width(P, v, float(A), float(Tt))
            d = 1e6 * depth(P, v, float(A), float(Tt))
            errs += [w / wr - 1, d / dr - 1]
        rms = np.sqrt(np.mean(np.array(errs) ** 2))
        results[(float(Tt), float(A))] = (rms, errs)
        row.append(rms * 100)
    print(f"{Tt:8.0f} " + " ".join(f"{r:6.1f}" for r in row))

best = min(results.items(), key=lambda kv: kv[1][0])
(Tt, A), (rms, errs) = best
print()
print(f"最优: T_thr={Tt:.0f} K, A={A:.2f}, RMS={rms*100:.2f}%")
print("  逐组残差（宽, 深）:")
for i, (cid, P, v, wr, dr) in enumerate(AMBENCH):
    print(f"    Case {cid} (v={v*1000:.0f} mm/s): 宽 {errs[2*i]:+7.2%}   深 {errs[2*i+1]:+7.2%}")
print()
print("  实测与模型的深宽比对照:")
for cid, P, v, wr, dr in AMBENCH:
    w = 2e6 * half_width(P, v, A, Tt)
    d = 1e6 * depth(P, v, A, Tt)
    print(f"    Case {cid}: 实测 d/w = {dr/wr:.3f}   模型 d/w = {d/w:.3f}")
