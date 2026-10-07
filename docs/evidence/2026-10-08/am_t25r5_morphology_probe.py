"""#25 第 5 轮：只给「液相线**等值面位置 + 体素计数**」这一半边打分（③ 代＝#23 之后；src 零改动、无 monkeypatch）

背景（`am_t25_round3_observable.log` R5、开发日志 §26.19）：#23 之后
  ΔH 2.38/0.77/4.66% ⇒ **能量侧已闭合**；peak 在 δ∈{0,.25} 是 4.21/0.08%（δ=0.5 列 5.15%）⇒ 有信息对齐达标；
  而 **Vn 13.27/6.36/1.13%、Vm 14.18/7.50/1.13%** ⇒ 形态半边不闭合。R3 已证 Vn≠Vm ⇒ 两种口径都看得见。
第 2 轮（② 代，`am_t25_isoattrib_probe.log`）归因为**双因**＝(i) 硬阈值体素计数口径 ＋ (ii) 场的 O(dx) 偏差。
本轮把两因**分开打分**：
  (i) 只换计数口径 ⇒ **Vs**：同一个 `peak_temperature` 场当**节点值**做三重线性重建，每格用 m³ 均匀子体素
      中心判据算液相份额，`Vs = Σ φ_cell·dx³`。物理、求解、实体判据**一字不动**（Vs 只换液相面一侧；
      实体边界一侧仍用中心判据，与 Vn 同——其合法性由 C0 每格打印为 0 支撑）。
  (ii) 等值面位置 ⇒ 沿用第 2 轮的「过 argmax 三轴最外侧线性插值交叉点」Lx/Ly/Lz，在 ③ 代重测。

预注册判据（写在跑之前；S1 的 **作废触发器** 任一不合格 ⇒ `sys.exit(1)` 全轮作废，不进任何物理结论）
----------------------------------------------------------------------------------
S1 估计器自检（解析场，先于任何求解）
  S1a 线性场重建精确性（F=x−a，a=0.5173，节点阵 k=41，dx=1/40）：三重线性重建对线性场应**逐位命中**
      解析值 ⇒ 取切割格 i=21 的 512 个子体素中心，要求 max|F̃−(x_sub−a)| < 1e-12
      （这是对我自己那 27 位移索引＋权重的机器精度检查，不是物理断言）。
      · **作废**：上一条不满足 ⇒ 插值实现有 bug。
      · **作废**：该格份额 ∉(0,1)、或界面格数=0 ⇒ 估计器退化成硬计数，本轮没有可解释的量。
      · **只打印不判**：闭式份额 φ=clip((x_i−a)/dx+0.5,0,1)＝0.808、硬计数给 1、m=8 子采样给 **0.750**
        （实测＝6/8。**更正**：初稿口算成 5/8＝0.625、未跑就写进文字，落盘 0 实测后改正；结论不变）
        ⇒ 单切割层的**总体积**精度是 1/m 量级而非机器精度级。初稿把"总量 <2e-3 且误差随 m 单调下降"
        写成判据，跑生产前自查算术后删除（S2–S5 一字未动；参照第 2 轮把 θ 降级为量级参考的同型处理）。
  S1b 球 F=r0²−|x−c|²（r0=0.1913，c=(.5,.5,.5)，V解析=(4/3)πr0³）在 k=21/41/81（r0/dx≈3.8/7.7/15.3）：
      · **V3b（正对照）**：max_k |Vn−V解析|/V解析 > 1e-3 ⇒ 硬计数的口径项在夹具里可见，否则作废。
      · **V4a**：|Vs−V解析| 三档**单调下降**（估计器会收敛）。
      · **V4b**：三档**平均** |Vs err| < 平均 |Vn err|（口径替换确实减小误差）。
        不用"仅最粗档"比较：r0/dx≈3.8 时整点计数有对齐运气，单档比较会误杀。
      · 只打分不作废：Vs 表观阶 p（带 1.5–3.0，第 2 轮交叉点定位器实测 2.34/1.83）、
        m 敏感性 |Vs(8)−Vs(4)|/Vs(8)（2% 为警戒线）。带外 ⇒ 记入 FAILS 并说明
        "Vs 作为**收敛体积**的解释力待定"，但 Vs 仍是同一场的**良定义泛函**，
        生产侧的口径对比（S3/S4/S5）不受影响。
S2 逐位复现（不成立 ⇒ 先查码，不记分）。拼写沿用第 3 轮的 `dx = dx_um*1e-6`、src 未动（`d7404f4..c3c7f95`
   对 `src/` 的 diff 为空），故必须等于 `am_t25_round3_observable.log` L38-45 的印值：
     peak 2461.27/2447.27/2433.15（dx=50 δ=0/.25/.5）、2563.01/2554.96/2565.18（dx=25）、2565.08/2555.81（dx=12.5）
     Vn   0.06025/0.06275/0.06400、0.06947/0.07062/0.06473、0.07418/0.07453
     Vm   0.05800/0.06244/0.06400、0.06759/0.06996/0.06473、0.07307/0.07407
   同时打印原始整数计数 nml：若某格 5 位小数不同而 nml 相同 ⇒ 属**已登记**的 dx 拼写舍入平局类
   （`am_t25_round3_observable.log` R1 的 ⚠ 段），不算复现失败；nml 也不同才算真失配。
S3 结构性预测（可失败，跑后逐条记分）
   P1 **五个** (δ, 档对)（δ∈{0,.25}×两档对 ＋ δ=0.5 的 50↔25）上全部 |ΔVs| < |ΔVn|
      （换成份额口径后每个档对差都要变小）。
   P2 ③ 代 Vn 三档差为 0.00922→0.00471（比值 1.96 ⇒ 一阶）。预测口径替换吃掉大半：
      |Vs(50)−Vs(25)| / |Vn(50)−Vn(25)| < 0.5（δ=0）。
   P2' **数量级外推**（由 S1b 的 dx² 律推出，写在跑生产之前）：S1b 实测 Vs 相对误差 −5.28/−1.43/−0.30%
      对应 r0/dx≈3.8/7.7/15.3 ⇒ |err| ≈ 0.76%·(dx/物理尺度)²。生产熔池半轴≈0.57mm ⇒ r/dx≈11.4/22.8/45.6
      ⇒ 预测 δ=0 的 |ΔVs|(50↔25) **≤0.5%**、最坏 ≤2%；若实测 ≥5% ⇒ 分支 B（等值面位置是主项）。
      ⚠ 该外推把"球"当"熔池"、并假定 dx² 律跨夹具迁移，是**带假设的**预测，命中与否都要照实记。
   P4 方向预测：S1b 显示 Vs 系统性**低估**（凹场三重线性内插偏低）且随加密单调回升 ⇒
      生产上 Vs(δ=0) 三档应**单调上升**（与 Vn 同向）。若反而下降 ⇒ 生产场非凹主导，另查。
   P3 extents 在 ③ 代必须**大于**第 2 轮 ② 代同格值（δ=0：Lx 1.144/1.168/1.174、Ly 0.546/0.567/0.566、
      Lz 0.192/0.197/0.206 mm；δ=0.25：1.149/1.157/1.172、0.558/0.563/0.589、0.195/0.198/0.215）
      ——#23 后同档 peak +4.8%、Vn +13~16%，同一液相线等值面只能外扩。出现反而变小 ⇒ 归因需重写。
S4 主判据＝**O1 原样**（5% 不放宽、δ 不挑、档对不挑、**δ=0.5 列不删**）：
   观测量入选 ⇔ δ∈{0,.25} 的两个档对 **且** δ=0.5 的 50↔25 档对全部 <5%。
   对 Vn、Vm、Vs、Lx、Ly、Lz **六个量各打一次**，两列都报（含 δ=0.5 的原判定 ＋ 剔除它的替代列）；
   若只在剔除 δ=0.5 时成立，结论必须写成「只在有信息的对齐上达标」，不许写「已闭合」。
S5 裁决表（跑前写死三选一，不许事后挑）
   A：Vs 入选而 Vn 不入选 ⇒ 剩余漂移主体＝**计数口径** ⇒ 下一步是把「熔体积改份额口径」作为
      **判据改动**呈报用户裁决（A0 断的是 Vn，不得静默换指标）。
   B：Vs 与三个 extents 全不入选 ⇒ 主体＝**场的等值面位置**随网格变，口径只是次要项 ⇒
      下一轮回求解器机制（#27 蒸发封顶、粗档源积分、H↔T 反演口径）。
   C：介于两者 ⇒ 混合，必须给每个档对的**逐项 pt 分解**
      |ΔVn| ≡ |ΔVs| ＋ |(Vn−Vs)_1 − (Vn−Vs)_2|，再决定下一步。
   任一分支都不放宽 5%、不删 δ=0.5、不改 `src/`、不换 A0 的断言量（N4：禁止叠加补丁调系数凑绿）。
"""
import itertools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
import jax                                           # noqa: E402

jax.config.update("jax_enable_x64", True)

from amforge import geometry as G                     # noqa: E402
from amforge.core.contracts import (ProcessPlan, solid_mask,  # noqa: E402
                                    solid_weight)
from amforge.materials import get_material            # noqa: E402
from amforge.thermal_enthalpy import solve_enthalpy_thermal  # noqa: E402

T0 = time.time()
EXT = (1.2e-3, 0.6e-3, 0.4e-3)
TIERS_UM = [50.0, 25.0, 12.5]
DELTAS = [0.0, 0.25, 0.5]
TL = float(get_material("316L").T_liquidus)
bad = []


def shift(F, off):
    """把 F 按整数位移 off∈{−1,0,1}³ 平移，边缘取 edge-clamp（熔区远离边界，C0 每轮验证）。"""
    n0, n1, n2 = F.shape
    P = np.pad(F, 1, mode="edge")
    return P[1 + off[0]:1 + off[0] + n0, 1 + off[1]:1 + off[1] + n1,
             1 + off[2]:1 + off[2] + n2]


def stencil_max(F):
    n0, n1, n2 = F.shape
    P = np.pad(F, 1, mode="edge")
    out = P[0:n0, 0:n1, 0:n2].copy()
    for a, b, c in itertools.product((-1, 0, 1), repeat=3):
        out = np.maximum(out, P[a + 1:a + 1 + n0, b + 1:b + 1 + n1, c + 1:c + 1 + n2])
    return out


def subcell_fraction(F, dx, level, m, debug_cell=None):
    """每格液相份额 φ = #{m³ 子体素中心：三重线性重建 F̃ > level}/m³。

    返回 (phi_full 数组, 界面格数, debug 格的 m³ 个重建值)。φ 只在「3³ 邻域最大值 ≥ level」
    的候选格上算——候选之外的格对体积**精确**贡献 0（F̃ 是 27 个节点值的凸组合）。
    """
    mx = stencil_max(F)
    cand = mx >= level
    I, J, K = np.nonzero(cand)
    phi = np.zeros(F.shape, dtype=np.float64)
    dbg = None
    if I.size == 0:
        return phi, 0, dbg
    sh = {o: shift(F, o)[I, J, K] for o in itertools.product((-1, 0, 1), repeat=3)}
    if debug_cell is not None:
        pos = int(np.flatnonzero((I == debug_cell[0]) & (J == debug_cell[1])
                                 & (K == debug_cell[2]))[0])
    u = (np.arange(m) + 0.5) / m - 0.5              # 子体素中心相对节点的位移（单位 dx）
    d = np.sign(u).astype(int)
    w = np.abs(u)
    hit = np.zeros(I.size, dtype=np.int64)
    vals = [] if debug_cell is not None else None
    for i, j, k in itertools.product(range(m), repeat=3):
        ox, oy, oz = d[i], d[j], d[k]
        val = np.zeros(I.size, dtype=np.float64)
        for ai, aj, ak in itertools.product((0, 1), repeat=3):
            wa = w[i] if ai else 1.0 - w[i]
            wb = w[j] if aj else 1.0 - w[j]
            wc = w[k] if ak else 1.0 - w[k]
            val += (wa * wb * wc) * sh[(ai * ox, aj * oy, ak * oz)]
        hit += (val > level)
        if vals is not None:
            vals.append(float(val[pos]))
    phi[I, J, K] = hit / float(m ** 3)
    return phi, int(((phi > 1e-12) & (phi < 1.0 - 1e-12)).sum()), vals


def outermost_crossing(line, coords, level):
    """一维采样线上 F−level 的**最外侧**过零点（线性插值）；对线性 F 逐位精确。"""
    s = np.asarray(line, dtype=np.float64) - level
    c = np.asarray(coords, dtype=np.float64)
    idx = np.nonzero(s > 0.0)[0]
    if idx.size == 0:
        return None
    i, j = idx[0], idx[-1]
    lo = c[i] if i == 0 else c[i - 1] + (level - np.float64(s[i - 1] + level)) \
        * (c[i] - c[i - 1]) / (np.float64(line[i]) - np.float64(line[i - 1]))
    hi = c[j] if j == len(c) - 1 else c[j] + (np.float64(line[j]) - level) \
        * (c[j + 1] - c[j]) / (np.float64(line[j]) - np.float64(line[j + 1]))
    return float(lo), float(hi)


# --------------------------------------------------------------- S1 估计器自检
print("=== S1a 线性场重建精确性：F̃ 在子体素中心必须命中解析值（机器精度）===", flush=True)
k, a = 41, 0.5173
dxp = 1.0 / (k - 1)
xs = np.arange(k) * dxp
X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
Fp = X - a
phi_closed = np.clip((xs - a) / dxp + 0.5, 0.0, 1.0)
V_closed = float(phi_closed.sum()) * dxp ** 3 * k * k
V_center = float((xs > a).sum()) * dxp ** 3 * k * k
icell = 21                                        # 切割格：闭式份额 0.808、硬计数给 1
phi8, nif8_p, vals = subcell_fraction(Fp, dxp, 0.0, 8, debug_cell=(icell, 5, 5))
Vs8 = float(phi8.sum()) * dxp ** 3
uu = (np.arange(8) + 0.5) / 8 - 0.5
combos = np.array(list(itertools.product(uu, repeat=3)))
ana = xs[icell] + combos[:, 0] * dxp - a          # 线性场解析值，与 vals 同一遍历序
err_interp = float(np.max(np.abs(np.asarray(vals) - ana)))
print(f"  切割格 i={icell}: 闭式份额={phi_closed[icell]:.4f}  Vs(m=8) 给={phi8[icell, 5, 5]:.4f}"
      f"  硬计数给={float(Fp[icell, 5, 5] > 0):.0f}", flush=True)
print(f"  重建值 vs 解析值（同一格 512 个子体素中心）最大绝对差={err_interp:.3e}"
      f"（场尺度={abs(Fp).max():.3f}）⇒ {'PASS' if err_interp < 1e-12 else 'FAIL'}", flush=True)
frac_ok = 1e-6 < phi8[icell, 5, 5] < 1 - 1e-6
print(f"  界面格数(m=8)={nif8_p}  该格份额真为分数={frac_ok}  "
      f"总量 Vs8={Vs8:.9f} V闭式={V_closed:.9f}（相对差={abs(Vs8 - V_closed) / V_closed:.3e}）"
      f" V中心计数={V_center:.9f}（相对差={abs(V_center - V_closed) / V_closed:.3e}）", flush=True)
if err_interp >= 1e-12:
    bad.append(f"S1a：线性场重建误差 {err_interp:.3e} ≥ 1e-12 ⇒ 插值机器有 bug，全轮作废")
if not (nif8_p > 0 and frac_ok):
    bad.append("S1a：估计器没有产生分数份额 ⇒ 它退化成硬计数，全轮作废")

print("\n=== S1b 球 F=r0²−|x−c|²：Vs 的精度/阶；硬计数误差＝正对照 ===", flush=True)
r0, VT = 0.1913, 4.0 / 3.0 * np.pi * 0.1913 ** 3
syn = {}
for kk in (21, 41, 81):
    d = 1.0 / (kk - 1)
    ax = np.arange(kk) * d
    Xb, Yb, Zb = np.meshgrid(ax, ax, ax, indexing="ij")
    Fb = r0 ** 2 - ((Xb - 0.5) ** 2 + (Yb - 0.5) ** 2 + (Zb - 0.5) ** 2)
    vn = float((Fb > 0.0).sum()) * d ** 3
    p4, _, _ = subcell_fraction(Fb, d, 0.0, 4)
    p8, _, _ = subcell_fraction(Fb, d, 0.0, 8)
    vs4 = float(p4.sum()) * d ** 3
    vs8 = float(p8.sum()) * d ** 3
    syn[kk] = (vn, vs4, vs8)
    print(f"  k={kk:3d} d={d:.5f}  Vn={vn:.6f}({(vn - VT) / VT * 100:+6.2f}%)  "
          f"Vs4={vs4:.6f}({(vs4 - VT) / VT * 100:+6.2f}%)  "
          f"Vs8={vs8:.6f}({(vs8 - VT) / VT * 100:+6.2f}%)  V解析={VT:.6f}", flush=True)
en = [abs(syn[kk][0] - VT) / VT for kk in (21, 41, 81)]
es = [abs(syn[kk][2] - VT) / VT for kk in (21, 41, 81)]
p0 = np.log(es[0] / es[1]) / np.log(2.0)
p1 = np.log(es[1] / es[2]) / np.log(2.0)
ins = abs(syn[81][2] - syn[81][1]) / syn[81][2]
v3b = max(en) > 1e-3
v4a = es[0] > es[1] > es[2]
v4b = sum(es) / 3.0 < sum(en) / 3.0
print(f"  V3b 硬计数误差>1e-3(正对照)={v3b}(max={max(en):.3e}) | "
      f"V4a |Vs err| 单调下降={v4a} | V4b Vs 平均误差< Vn 平均误差={v4b}"
      f"（{sum(es) / 3:.3e} vs {sum(en) / 3:.3e}）", flush=True)
print(f"  只打分：Vs 表观阶 p={p0:.2f}/{p1:.2f}（带 1.5–3.0）；"
      f"m 敏感性 |Vs8−Vs4|/Vs8={ins:.3e}（警戒 2e-2）", flush=True)
if not v3b:
    bad.append("S1b/V3b：硬计数误差 ≤1e-3 ⇒ 夹具看不见口径项，本轮作废")
if not v4a:
    bad.append(f"S1b/V4a：|Vs−V解析| 不单调（{es}）⇒ 估计器不收敛，本轮作废")
if not v4b:
    bad.append("S1b/V4b：Vs 平均误差不比硬计数小 ⇒ 口径替换不可解释，本轮作废")
if not (1.5 <= min(p0, p1) and max(p0, p1) <= 3.0):
    bad.append(f"S1b：Vs 表观阶 {p0:.2f}/{p1:.2f} 在 1.5–3.0 带外 ⇒ Vs 作为『收敛体积』的解释力"
               f"待定（生产侧口径对比仍有效，见 S5 注）")
if ins >= 2e-2:
    bad.append(f"S1b：m 敏感性 {ins:.3e} ≥ 2% ⇒ 子采样分辨率不足（生产侧用 m=8 需重估）")
if any(("作废" in s or "不单调" in s or "插值机器" in s) for s in bad):
    print(f"\nFAILS（估计器自检未过 ⇒ 全轮作废）: {bad}", flush=True)
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


print("\n=== 设备清单（长期规则：自称 GPU 实测必须含本行，写在 log 本体）===", flush=True)
print(f"  jax.devices() = {jax.devices()}", flush=True)
print(f"  platform/backend = {jax.default_backend()} / x64={jax.config.jax_enable_x64}",
      flush=True)

print("\n=== 生产求解 + 三种计数口径 + 等值面位置（无 patch；场＝th.peak_temperature）===",
      flush=True)
print(f"  T_liquidus={TL:.1f} K", flush=True)
R = {}
for dx_um in TIERS_UM:
    for frac in (DELTAS if dx_um != 12.5 else [0.0, 0.25]):   # 与第 3 轮同一采集集
        dx = dx_um * 1e-6                                      # 拼写＝第 3 轮（S2 前提）
        g = coupon(dx, frac * dx)
        th = solve_enthalpy_thermal(geometry=g, process=plan(),
                                    params={"material": "316L"})
        F = np.asarray(th.peak_temperature, dtype=np.float64)
        sdf = np.asarray(g.sdf, dtype=np.float64)
        mk = np.asarray(solid_mask(sdf), dtype=np.float64) > 0.5
        w = np.asarray(solid_weight(sdf, dx), dtype=np.float64)
        above = (F > TL) & mk
        phi4, nif4, _ = subcell_fraction(F, dx, TL, 4)
        phi8, nif8, _ = subcell_fraction(F, dx, TL, 8)
        vol = dx ** 3 * 1e9
        # 掩膜显式施加（与 Vn 同一实体判据）；同时**实测**它是否无操作，不靠断言。
        Vs4u, Vsu = float(phi4.sum()) * vol, float(phi8.sum()) * vol
        phi4 = phi4 * mk
        phi8 = phi8 * mk
        sh = F.shape
        i0, j0, k0 = np.unravel_index(int(np.argmax(F * mk)), sh)
        ax = [np.asarray(g.origin, dtype=np.float64)[d] + np.arange(sh[d]) * dx
              for d in range(3)]
        # C0：熔区是否只被液相面切割（不触实体边界、不触网格外壳）
        edge_cells = int(above.sum() - (above[1:-1, 1:-1, 1:-1]).sum())
        bnd = 0
        for o in itertools.product((-1, 0, 1), repeat=3):
            if o != (0, 0, 0):
                bnd += int((above & ~shift(mk, o)).sum())
        ext = {}
        for lab, line, co in (("Lx", F[:, j0, k0], ax[0]),
                              ("Ly", F[i0, :, k0], ax[1]),
                              ("Lz", F[i0, j0, :], ax[2])):
            cr = outermost_crossing(line, co, TL)
            ext[lab] = (cr[1] - cr[0]) * 1e3 if cr else 0.0
        rec = dict(nvox=int(mk.sum()), nml=int(above.sum()),
                   peak=float(F[i0, j0, k0]),
                   Vn=float(above.sum()) * vol, Vm=float((above * w).sum()) * vol,
                   Vs4=float(phi4.sum()) * vol, Vs=float(phi8.sum()) * vol,
                   Vs4u=Vs4u, Vsu=Vsu,
                   nif=nif8, nif4=nif4, edge=edge_cells, bnd=bnd,
                   fv_min=float(w[above].min()) if above.sum() else 1.0, **ext)
        R[(frac, dx_um)] = rec
        print(f"  δ={frac:4.2f}·dx dx={dx_um:5.1f} nvox={rec['nvox']:7d} "
              f"peak={rec['peak']:7.2f} 熔体素={rec['nml']:6d} "
              f"界面格={rec['nif']:6d} 触壳={rec['edge']} 邻非实体={rec['bnd']} "
              f"min(fv)={rec['fv_min']:.3f} | "
              f"Vn={rec['Vn']:.5f} Vm={rec['Vm']:.5f} Vs(m4)={rec['Vs4']:.5f} "
              f"Vs(m8)={rec['Vs']:.5f} mm³（加掩膜前后逐位同={rec['Vs'] == rec['Vsu']}）| "
              f"Lx={rec['Lx']:6.3f} Ly={rec['Ly']:6.3f} Lz={rec['Lz']:6.3f} mm", flush=True)

print("\n=== C0 判据：熔区只被液相面切割（触壳数与邻非实体数必须为 0）===", flush=True)
c0 = [(f, d, R[(f, d)]['edge'], R[(f, d)]['bnd']) for f in DELTAS for d in TIERS_UM
      if (f, d) in R]
for f, d, e, b in c0:
    r = R[(f, d)]
    mnoop = (r['Vs'] == r['Vsu']) and (r['Vs4'] == r['Vs4u'])
    print(f"  δ={f:4.2f}·dx dx={d:5.1f}: 触壳={e} 邻非实体={b} 掩膜对 Vs 逐位无操作={mnoop} ⇒ "
          f"{'OK（Vs 与 Vn 的差只来自液相面一侧）' if e == 0 and b == 0 and mnoop else '不 OK'}",
          flush=True)
    if e or b:
        bad.append(f"C0 δ={f} dx={d}：熔区含实体边界/外壳格（触壳={e} 邻非实体={b}）")
    if not mnoop:
        bad.append(f"C0 δ={f} dx={d}：实体掩膜改变 Vs（{r['Vsu']:.5f}→{r['Vs']:.5f}）⇒ "
                   f"Vs 与 Vn 的差别混进了实体侧口径，对比不干净")

print("\n=== S2 逐位复现（对照 am_t25_round3_observable.log L38-45）===", flush=True)
ARC = {  # (δ, dx_um): (peak, Vn, Vm) 第 3 轮印值
    (0.0, 50.0): (2461.27, 0.06025, 0.05800), (0.25, 50.0): (2447.27, 0.06275, 0.06244),
    (0.50, 50.0): (2433.15, 0.06400, 0.06400), (0.0, 25.0): (2563.01, 0.06947, 0.06759),
    (0.25, 25.0): (2554.96, 0.07062, 0.06996), (0.50, 25.0): (2565.18, 0.06473, 0.06473),
    (0.0, 12.5): (2565.08, 0.07418, 0.07307), (0.25, 12.5): (2555.81, 0.07453, 0.07407),
}
nhit = 0
for key, (ap, an, am) in ARC.items():
    r = R[key]
    dp, dn, dm = abs(r['peak'] - ap), abs(r['Vn'] - an), abs(r['Vm'] - am)
    same = dp < 5e-3 and dn < 5.1e-6 and dm < 5.1e-6
    nhit += same
    print(f"  δ={key[0]:4.2f} dx={key[1]:5.1f}: peak {r['peak']:7.2f}(Δ{dp:.3f}) "
          f"Vn {r['Vn']:.5f}(Δ{dn:.6f}) Vm {r['Vm']:.5f}(Δ{dm:.6f}) 熔体素={r['nml']} ⇒ "
          f"{'一致' if same else '不一致'}", flush=True)
    if not same:
        bad.append(f"S2 δ={key[0]} dx={key[1]}：peak Δ={dp:.3f} Vn Δ={dn:.6f} Vm Δ={dm:.6f}")
print(f"  ⇒ {nhit}/{len(ARC)} 格一致", flush=True)


def rel(a, b):
    return abs(a - b) / max(a, b)


PAIRS = [(50.0, 25.0), (25.0, 12.5)]
LAB = ("Vn", "Vm", "Vs", "Lx", "Ly", "Lz")
print("\n=== S4 主判据 O1（原样）：六个量、三列 δ、两档对；5% 不放宽、δ=0.5 不删 ===", flush=True)
rows = {}
for lab in LAB:
    per = {}
    for f in (0.0, 0.25, 0.5):
        for (d1, d2) in PAIRS:
            if (f, d2) not in R:
                continue
            per[(f, d1)] = rel(R[(f, d1)][lab], R[(f, d2)][lab])
    info = [v for (f, _), v in per.items() if f < 0.4]
    allc = list(per.values())
    rows[lab] = per
    print(f"  {lab:3s}  档对差 " + "  ".join(
        f"δ={f:4.2f}/{d1:5.1f}↔{d2:5.1f} {per[(f, d1)] * 100:6.3f}%"
        for (f, d1), d2 in ((k, k[1] / 2) for k in per)) +
        f"  | 含 δ=0.5 全列最大 {max(allc) * 100:6.3f}% ⇒ "
        f"{'入选' if max(allc) < 0.05 else '不入选'}   ‖ 剔除 δ=0.5 最大 "
        f"{max(info) * 100:6.3f}% ⇒ {'达标' if max(info) < 0.05 else '不达标'}", flush=True)

print("\n=== S3 结构性预测记分 ===", flush=True)
p1 = True
for f in (0.0, 0.25):
    for (d1, d2) in PAIRS:
        a, b = rel(R[(f, d1)]['Vs'], R[(f, d2)]['Vs']), rel(R[(f, d1)]['Vn'], R[(f, d2)]['Vn'])
        p1 &= a < b
p1 &= rel(R[(0.5, 50.0)]['Vs'], R[(0.5, 25.0)]['Vs']) < rel(R[(0.5, 50.0)]['Vn'], R[(0.5, 25.0)]['Vn'])
ratio = abs(R[(0.0, 50.0)]['Vs'] - R[(0.0, 25.0)]['Vs']) / \
    abs(R[(0.0, 50.0)]['Vn'] - R[(0.0, 25.0)]['Vn'])
dv0 = rel(R[(0.0, 50.0)]['Vs'], R[(0.0, 25.0)]['Vs'])
vs_seq = [R[(0.0, d)]['Vs'] for d in TIERS_UM]
p4 = vs_seq[0] < vs_seq[1] < vs_seq[2]
print(f"  P1（五档对全部 |ΔVs|<|ΔVn|）：{'PASS' if p1 else 'FAIL'}", flush=True)
print(f"  P2（|ΔVs|/|ΔVn| 在 δ=0 的 50↔25 < 0.5）：{ratio:.3f} ⇒ "
      f"{'PASS' if ratio < 0.5 else 'FAIL'}", flush=True)
print(f"  P2'（δ=0 的 |ΔVs|(50↔25) 预测 ≤0.5%、最坏 ≤2%）：实测 {dv0 * 100:.3f}% ⇒ "
      f"{'命中' if dv0 <= 0.005 else ('带内' if dv0 <= 0.02 else '不命中（≥5%⇒分支 B）')}",
      flush=True)
print(f"  P4（Vs δ=0 三档单调上升 {vs_seq[0]:.5f}→{vs_seq[1]:.5f}→{vs_seq[2]:.5f}）："
      f"{'PASS' if p4 else 'FAIL'}", flush=True)
if not p1:
    bad.append("P1：存在 |ΔVs| ≥ |ΔVn| 的档对")
if not p4:
    bad.append(f"P4：Vs δ=0 三档非单调上升（{vs_seq}）⇒ 生产场非凹主导，需另查")
ARC2 = {  # 第 2 轮 ② 代 extents（mm）
    (0.0, 50.0): (1.144, 0.546, 0.192), (0.0, 25.0): (1.168, 0.567, 0.197),
    (0.0, 12.5): (1.174, 0.566, 0.206), (0.25, 50.0): (1.149, 0.558, 0.195),
    (0.25, 25.0): (1.157, 0.563, 0.198), (0.25, 12.5): (1.172, 0.589, 0.215),
}
p3_ok, p3_bad = True, []
for key, (lx, ly, lz) in ARC2.items():
    r = R[key]
    for lab, now, was in (("Lx", r['Lx'], lx), ("Ly", r['Ly'], ly), ("Lz", r['Lz'], lz)):
        if now < was:
            p3_ok = False
            p3_bad.append(f"{lab} δ={key[0]} dx={key[1]}: ③ {now:.3f} < ② {was:.3f}")
print(f"  P3（③ 代 extents 全大于 ② 代同格值）：{'PASS' if p3_ok else 'FAIL'}", flush=True)
for s in p3_bad:
    print(f"     {s}", flush=True)
if not p3_ok:
    bad.append("P3：" + "；".join(p3_bad))

print("\n=== 逐项 pt 分解：|ΔVn| ＝ |ΔVs| ＋ |(Vn−Vs)₁−(Vn−Vs)₂|（S5 的裁决材料）===",
      flush=True)
for f in (0.0, 0.25, 0.5):
    for (d1, d2) in PAIRS:
        if (f, d2) not in R:
            continue
        a, b = R[(f, d1)], R[(f, d2)]
        dn = abs(a['Vn'] - b['Vn'])
        dvs = abs(a['Vs'] - b['Vs'])
        dbi = abs((a['Vn'] - a['Vs']) - (b['Vn'] - b['Vs']))
        print(f"  δ={f:4.2f}·dx {d1:5.1f}↔{d2:5.1f}µm: |ΔVn|={dn:.5f}mm³({dn / b['Vn'] * 100:6.3f}%)"
              f" = |ΔVs|{dvs:.5f}({dvs / b['Vn'] * 100:6.3f}%) + 口径项差{dbi:.5f}"
              f"({dbi / b['Vn'] * 100:6.3f}%)  口径本身 Vn−Vs：{a['Vn'] - a['Vs']:+.5f}"
              f"/{b['Vn'] - b['Vs']:+.5f}", flush=True)

print("\n=== S5 裁决 ===", flush=True)
sel_full = {lab: max(rows[lab].values()) < 0.05 for lab in LAB}
sel_info = {lab: max(v for (f, _), v in rows[lab].items() if f < 0.4) < 0.05 for lab in LAB}
print(f"  含 δ=0.5：{sel_full}", flush=True)
print(f"  剔除 δ=0.5：{sel_info}", flush=True)
if sel_full['Vs'] and not sel_full['Vn']:
    verdict = "A：Vs 入选、Vn 不入选 ⇒ 剩余漂移主体是**计数口径**（下一步＝把份额口径作为判据改动呈报用户）"
elif not sel_full['Vs'] and not any(sel_full[l] for l in ('Lx', 'Ly', 'Lz')):
    verdict = "B：Vs 与 extents 全不入选 ⇒ 主体是**场的等值面位置**随网格变（下一轮回求解器机制：#27/源积分/H↔T）"
else:
    verdict = "C：混合 ⇒ 按上面 pt 分解决定下一步（不叠加补丁、不调系数）"
print(f"  ⇒ {verdict}", flush=True)
print("  注：extents/Vs 是**归因诊断**，不得替换 A0 的 assert2 观测量（A0 断 peak 与 Vn）；"
      "5% 未放宽、δ 未挑、档对未挑、δ=0.5 列未删、src 未改。", flush=True)

print("\n=== 记分 ===", flush=True)
print("FAILS:", bad if bad else "无", flush=True)
print(f"elapsed={time.time() - T0:.0f}s", flush=True)
