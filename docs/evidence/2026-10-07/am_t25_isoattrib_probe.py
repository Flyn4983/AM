"""#25 改前打分第 2 轮：**归因**探针（不动生产码、无 monkeypatch）。

第 1 轮（`am_t25_aperture_probe.log`，42 次求解，N2 入选变体=[]）已判
「**面开口度不是 Vn 不收敛的主因**」。留下一条可证伪的分叉——Vn 的 11% 相邻档漂移是
  (i)  **观测量的口径项**：`Vn = Σ H(peak>T_liq)·dx³` 是"体素中心＋硬阈值"计数，对
        一个光滑的等温面它有 ~A·dx/2 的一阶偏差（界面单元各少算半格）。加密必然
        单调增大、符号与实测一致；此时**几何本身**在 50↔25µm 其实已经收敛；或
  (ii) **场本身**在粗档有 O(dx) 偏差（候选 2＝H↔T 反演按满体素潜热、候选 3＝evap
        封顶）：那 Vn 只是诚实的读数，该继续找求解器机制。

工具＝**亚格交叉点**量几何：沿三条轴过 argmax(peak) 的网格线，把 F−T_liq 的**最外侧
过零点**用线性插值定位（对线性场**逐位精确**，对光滑曲界面是 O(dx²)）。由此得
Lx/Ly/Lz（熔池长/宽/深），它们与被争论的 Vn 是同一物理集合的不同泛函——若几何收敛而
计数不收敛，只能归因于 (i)。

第 1 次草稿还试过一个"亚格等温面体积" θ=clip(0.5+(F−T_liq)/(|∇F|dx))；自检
（本文件 R0 段）实测它在圆界面只有 p≈0.3（斜界面单元沿法向的跨度不是 dx，未修正），
**故本轮不把它当判据**，只作为量级参考打印。

跑前登记的判据（C1/C2 不合格 ⇒ 全轮作废）：
  C1 交叉点定位器**正对照**：F=x（平面）与 F=x+y（斜平面）在 40×30 格上，解析交点与
     定位器结果相对误差 < 1e-12。不满足 ⇒ 实现有 bug。
  C2 定位器**阶对照**：F=(x−0.5)²+(y−0.5)²（圆，解析半径 r0）在 n=40/80/160 上，
     误差非零（>1e-10 ⇒ 定位器看得见曲率）**且**表观阶 1.5 ≤ p ≤ 3。
  E1 **主判据**（沿用 A0 assert2 的"每个 δ、每对相邻档相对差 <5%"，不改阈值、不挑
     δ、不挑档对）：把 |ΔVn| 换成 |ΔLx|、|ΔLy|、|ΔLz| 三条。
     · 三个_extents_在**每个 δ、每对相邻档**都 <5% ⇒ 判 (i)：几何已收敛，Vn 的漂移是
       硬阈值计数的口径项。**禁止**把 extents 当新判据；下一步是把结论＋实测系数
       呈报用户（与 #19 前"A0 判据本身是缺陷"同型，动判据需用户裁决）。
     · 任一 extent 在任一 (δ, 档对) ≥5% ⇒ 判 (ii)：场有 O(dx) 偏差 ⇒ 候选 2/3 保留为
       主因，下一轮对 H↔T 反演口径打分。
  E2 量级核对（支持性，不作红绿）：界面单元数 N_iface 与硬阈值口径上界 N_iface·dx³/2；
     要求实测 |V_θ−Vn| ≤ 该上界，且 |V_θ−Vn|/Vn 随加密单调下降。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
import jax                                          # noqa: E402

jax.config.update("jax_enable_x64", True)

from amforge import geometry as G                    # noqa: E402
from amforge.core.contracts import (ProcessPlan, solid_mask,  # noqa: E402
                                    solid_weight)
from amforge.materials import get_material           # noqa: E402
from amforge.thermal_enthalpy import solve_enthalpy_thermal  # noqa: E402

EXT = (1.2e-3, 0.6e-3, 0.4e-3)
TIERS = [50e-6, 25e-6, 12.5e-6]
DELTAS = [0.0, 0.25]
TL = float(get_material("316L").T_liquidus)
bad = []


def outermost_crossing(line, coords, level):
    """一维采样线上 F−level 的**最外侧**过零点（线性插值）；无熔化返回 None。

    返回 (lo, hi)：熔区在两端的物理坐标。对线性 F 逐位精确。
    """
    s = np.asarray(line, dtype=np.float64) - level
    c = np.asarray(coords, dtype=np.float64)
    idx = np.nonzero(s > 0.0)[0]
    if idx.size == 0:
        return None
    i, j = idx[0], idx[-1]
    # 左/外端：在 i−1（<level）与 i（>level）之间插值
    lo = c[i] if i == 0 else c[i - 1] + (level - np.float64(s[i - 1] + level)) \
        * (c[i] - c[i - 1]) / (np.float64(line[i]) - np.float64(line[i - 1]))
    hi = c[j] if j == len(c) - 1 else c[j] + (np.float64(line[j]) - level) \
        * (c[j + 1] - c[j]) / (np.float64(line[j]) - np.float64(line[j + 1]))
    return float(lo), float(hi)


def theta_frac(F, dx, level):
    """参考用的亚格体积分数（**本轮实测其阶不如声称，仅打印量级**）。"""
    grads = np.gradient(F, dx, edge_order=2)
    g = np.sqrt(sum(q * q for q in grads))
    return np.clip(0.5 + (F - level) / np.maximum(g * dx, 1e-300), 0.0, 1.0)


# ------------------------------------------------ C1/C2 定位器自检（先于任何求解）
print("=== C1 正对照：解析平面/斜平面的交点应逐位命中（半空间⇒取内端 lo） ===", flush=True)
for lab, f_of, x0 in (("F=x（轴对齐）", lambda X, Y: X, 0.5173),
                      ("F=x+y（斜 45°）", lambda X, Y: X + Y, 1.2345)):
    n0, n1, d = 40, 30, 1.0 / 37.0
    ax = np.arange(n0) * d
    y = np.arange(n1) * d
    X, Y = np.meshgrid(ax, y, indexing="ij")
    F = f_of(X, Y)
    lo_hi = outermost_crossing(F[:, n1 // 2], ax, x0)
    if lo_hi is None:
        bad.append(f"C1 {lab}：无过零点")
        continue
    analytic = x0 - f_of(0.0, y[n1 // 2])       # F=x 时 =x0；F=x+y 时 =x0−y0
    err = abs(lo_hi[0] - analytic) / max(abs(analytic), 1.0)
    print(f"  {lab}: lo={lo_hi[0]:.18e} 解析={analytic:.18e} 相对误差={err:.3e} "
          f"⇒ {'PASS' if err < 1e-12 else 'FAIL（定位器不进判据）'}", flush=True)
    if err >= 1e-12:
        bad.append(f"C1 {lab}：误差 {err:.3e} ≥ 1e-12")

print("\n=== C2 阶对照：圆界面 F=−((x−.5)²+(y−.5)²)、level=−r0² ⇒ {F>level}=圆盘 ===",
      flush=True)
r0, errs = 0.4137, []
for k in (40, 80, 160):
    d = 1.0 / (k - 1)
    ax = np.arange(k) * d
    X, Y = np.meshgrid(ax, ax, indexing="ij")
    F = -((X - 0.5) ** 2 + (Y - 0.5) ** 2)
    row = outermost_crossing(F[:, k // 2], ax, -(r0 ** 2))   # y=0.5 ⇒ 右端 0.5+r0
    err = abs(row[1] - (0.5 + r0))
    errs.append(err)
    print(f"  k={k:3d} d={d:.5f}  交点={row[1]:.12f} 解析={0.5 + r0:.12f} "
          f"绝对误差={err:.3e}", flush=True)
p0 = np.log(errs[0] / errs[1]) / np.log(2.0)
p1 = np.log(errs[1] / errs[2]) / np.log(2.0)
print(f"  ⇒ 表观阶 p={p0:.2f}/{p1:.2f}（要求 1.5–3.0）；误差非零={errs[2] > 1e-10}",
      flush=True)
if errs[2] <= 1e-10:
    bad.append("R0/C2：曲界面误差为 0 ⇒ 定位器退化")
if not (1.5 <= min(p0, p1) and max(p0, p1) <= 3.0):
    bad.append(f"C2：表观阶 {p0:.2f}/{p1:.2f} 不在 1.5–3.0")
if bad:
    print(f"\nFAILS（定位器自检未过 ⇒ 全轮作废）: {bad}", flush=True)
    sys.exit(1)


def coupon(dx, delta):
    bx, by, bz = EXT
    bounds = [(-bx / 2 - delta, bx / 2 + delta),
              (-by / 2 - delta, by / 2 + delta),
              (-bz / 2 - delta, bz / 2 + delta)]

    def fn(x):
        return np.maximum(np.maximum(np.abs(x[..., 0]) - bx / 2,
                                     np.abs(x[..., 1]) - by / 2),
                          np.abs(x[..., 2]) - bz / 2)

    return G.from_sdf_fn(lambda x: fn(np.asarray(x, dtype=np.float64)),
                         bounds=bounds, spacing=dx, name="coupon")


def plan():
    return ProcessPlan.uniform(1, modality="SLM", laser_power=600.0, scan_speed=0.8,
                               layer_thickness=EXT[2], hatch_spacing=1.4 * 100e-6,
                               beam_radius=100e-6, absorption=0.45, preheat_temp=400.0)


print("\n=== 生产求解 + 几何量（无 patch；场＝th.peak_temperature，逐体素峰值） ===",
      flush=True)
print(f"  T_liquidus={TL:.1f} K  device={jax.devices()[0]}", flush=True)
R = {}
for frac in DELTAS:
    for dx in TIERS:
        g = coupon(dx, frac * dx)          # frac 是"多少倍 dx"，不是绝对位移
        th = solve_enthalpy_thermal(geometry=g, process=plan(),
                                    params={"material": "316L"})
        F = np.asarray(th.peak_temperature, dtype=np.float64)
        sdf = np.asarray(g.sdf, dtype=np.float64)
        mask = np.asarray(solid_mask(sdf), dtype=np.float64) > 0.5
        w = np.asarray(solid_weight(sdf, dx), dtype=np.float64)
        sh = F.shape
        ax = [np.asarray(g.origin, dtype=np.float64)[d] + np.arange(sh[d]) * dx
              for d in range(3)]
        i0, j0, k0 = np.unravel_index(int(np.argmax(F * mask)), sh)
        above = (F > TL) & mask
        t = theta_frac(F, dx, TL) * mask
        vol = dx ** 3 * 1e9
        ext = {}
        for lab, line, co in (("Lx", F[:, j0, k0], ax[0]),
                              ("Ly", F[i0, :, k0], ax[1]),
                              ("Lz", F[i0, j0, :], ax[2])):
            cr = outermost_crossing(line, co, TL)
            ext[lab] = (cr[1] - cr[0]) * 1e3 if cr else 0.0      # mm
        rec = dict(nvox=int(mask.sum()), nml=int(above.sum()), argmax=(i0, j0, k0),
                   peak=float(F[i0, j0, k0]),
                   Vn=float(above.sum()) * vol, Vm=float((above * w).sum()) * vol,
                   Vt=float(t.sum()) * vol,
                   nif=int(((t > 1e-9) & (t < 1 - 1e-9)).sum()), **ext)
        R[(frac, dx)] = rec
        print(f"  δ={frac:4.2f}·dx dx={dx*1e6:5.1f} nvox={rec['nvox']:7d} "
              f"peak={rec['peak']:7.1f} 熔体素={rec['nml']:6d} "
              f"Lx={rec['Lx']:6.3f} Ly={rec['Ly']:6.3f} Lz={rec['Lz']:6.3f} mm | "
              f"Vn={rec['Vn']:.5f} Vm={rec['Vm']:.5f} Vθ={rec['Vt']:.5f} mm³ "
              f"界面单元={rec['nif']:6d}", flush=True)


def rel(a, b):
    return abs(a - b) / max(a, b)


print("\n=== E1 主判据：三个几何量在每个 δ、每对相邻档是否都 <5% ===", flush=True)
worst = {}
ok_ext = True
for frac in DELTAS:
    for (d1, d2) in zip(TIERS[:-1], TIERS[1:]):
        a, b = R[(frac, d1)], R[(frac, d2)]
        row = {lab: rel(a[lab], b[lab]) for lab in ("Lx", "Ly", "Lz")}
        row["Vn"] = rel(a["Vn"], b["Vn"])
        worst[(frac, d1)] = max(row["Lx"], row["Ly"], row["Lz"])
        print(f"  δ={frac:4.2f}·dx {d1*1e6:5.1f}↔{d2*1e6:5.1f}µm: "
              f"Lx {row['Lx']*100:6.3f}%  Ly {row['Ly']*100:6.3f}%  "
              f"Lz {row['Lz']*100:6.3f}%  ‖ Vn {row['Vn']*100:6.3f}%  ⇒ "
              f"{'几何全 <5%' if max(row['Lx'], row['Ly'], row['Lz']) < 0.05 else '几何有 ≥5%'}",
              flush=True)
        if max(row["Lx"], row["Ly"], row["Lz"]) >= 0.05:
            ok_ext = False
for lab in ("Lx", "Ly", "Lz", "Vn"):
    for frac in DELTAS:
        s = [R[(frac, dx)][lab] for dx in TIERS]
        d12, d23 = s[0] - s[1], s[1] - s[2]
        p = np.log(abs(d12 / d23)) / np.log(2.0) if d12 * d23 > 0 else float("nan")
        print(f"  {lab} δ={frac:4.2f}·dx 三档 {s[0]:.5f}→{s[1]:.5f}→{s[2]:.5f} "
              f"表观阶 p={'{:5.2f}'.format(p) if p == p else '  --（差分变号）'}",
              flush=True)
print(f"\n  ⇒ 判定：{'**几何（Lx/Ly/Lz）全部收敛而 Vn 不收敛 ⇒ 归因 (i)：硬阈值计数的口径项**' if ok_ext else '几何量本身在粗档 ≥5% ⇒ 归因 (ii)：场有 O(dx) 偏差，候选 2/3 保留'}",
      flush=True)

print("\n=== E2 量级核对：|Vθ−Vn| ≤ N_iface·dx³/2 且相对口径项随 dx 单调下降 ===",
      flush=True)
for frac in DELTAS:
    seq = []
    for dx in TIERS:
        a = R[(frac, dx)]
        d = abs(a["Vt"] - a["Vn"])
        bound = a["nif"] * dx ** 3 / 2 * 1e9
        seq.append(d / a["Vn"])
        if d > bound:
            bad.append(f"E2 δ={frac} dx={dx * 1e6}µm：|Vθ−Vn|={d:.5f} > 界面单元上界 "
                       f"{bound:.5f}")
        print(f"  δ={frac:4.2f}·dx dx={dx*1e6:5.1f}: |Vθ−Vn|={d:.5f}mm³ 上界={bound:.5f} "
              f"比值={d/bound:5.2f} 相对={d/a['Vn']*100:5.2f}% Vθ/Vn={a['Vt']/a['Vn']:.4f}",
              flush=True)
    print(f"     单调下降：{'是' if seq[0] >= seq[1] >= seq[2] else '否 ⇒ (i) 解释不完整'}",
          flush=True)

print("\n=== 记分 ===", flush=True)
print("FAILS:", bad if bad else "无", flush=True)
print("注：extents/Vθ 是**归因诊断**，不得替换 A0 的 assert2 观测量；5% 未放宽、δ 未挑、"
      "档对未挑。", flush=True)
