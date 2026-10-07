"""T2 机理诊断：为什么**网格形状完全相同**，两种 dx 拼写的 peak/vol 仍差 0.53%/1.77%。

假设（写死在本文件里，跑前登记，跑后照实记分）：
  H1 刀刃共振：当 `(hi-lo)/dx` 恰为整数时，**表面体素的 SDF 恰为 0**，而掩膜用严格
     `sdf < 0`；1 ulp 的 dx 抖动把"恰为 0"推成正或负 ⇒ 一层皮体素整体进/出固体域，
     nml 与 peak 因此**离散跳变**（不是连续漂移）。
     判据：两种拼写下 `|sdf| < 1e-12` 的体素数 > 0，且 `(sdf<0)` 计数不同。
  H2 若 `|sdf|<1e-12` 计数为 0 且掩膜计数相同 ⇒ H1 被证伪，差异来自别处
     （热源分箱 / fv 权重），需继续追，本轮不下结论。

本探针**不求解热**，只做几何诊断（秒级），因此不与全量验收抢 CPU。
"""
import jax.numpy as jnp
import numpy as np

jax_x64 = __import__("jax")
jax_x64.config.update("jax_enable_x64", True)

from amforge import geometry as G

BX, BY, BZ = 1.2e-3, 0.6e-3, 0.4e-3


def _sdf(x):
    return jnp.maximum(jnp.maximum(jnp.abs(x[..., 0]) - BX / 2,
                                   jnp.abs(x[..., 1]) - BY / 2),
                       jnp.abs(x[..., 2]) - BZ / 2)


def coupon(dx_um, mul=True):
    sp = dx_um * 1e-6 if mul else float(f"{dx_um}e-6")
    return G.from_sdf_fn(_sdf, bounds=[(-BX / 2, BX / 2), (-BY / 2, BY / 2),
                                       (-BZ / 2, BZ / 2)], spacing=sp,
                         name="coupon")


for dx_um in (12.5, 25.0, 50.0):
    gm, gl = coupon(dx_um, True), coupon(dx_um, False)
    print(f"\n=== dx={dx_um} µm ===")
    for tag, g in (("mul", gm), ("lit", gl)):
        s = np.asarray(g.sdf)
        print(f"  [{tag}] dx={float(g.spacing)!r} shape={s.shape} "
              f"solid(sdf<0)={int((s < 0).sum())} "
              f"|sdf|<1e-12:{int((np.abs(s) < 1e-12).sum())} "
              f"sdf==0:{int((s == 0).sum())} "
              f"min|sdf|={np.abs(s[np.abs(s) > 0]).min() if (s != 0).any() else float('nan'):.3e}")
    a = np.asarray(gm.sdf) < 0
    b = np.asarray(gl.sdf) < 0
    flips = int((a ^ b).sum())
    print(f"  H1 掩膜翻转体素数 = {flips}（占固体 {flips / max(1, int(a.sum())) * 100:.2f}%）"
          f"  ⇒ {'H1 成立（刀刃共振）' if flips else 'H1 在本档被证伪'}")
    if flips:
        idx = np.argwhere(a ^ b)
        edge = all(i in (0, s - 1) for row in idx for i, s in zip(row, a.shape))
        print(f"    翻转位置样例（前 3，坐标轴下标）：{idx[:3].tolist()}")
        print(f"    翻转是否全在边界：{'是' if edge else '否（有内部翻转，须继续追）'}")
