# -*- coding: utf-8 -*-
"""#25 第 6 轮：把第 5 轮混在一起的**两类口径**拆开——液相面 (L) 与自由面 (S)。

第 5 轮（`am_t25r5_morphology_probe.log`，分支 A）自己抓出的漏洞：C0 预设"熔区只被液相面切割"
被实测否证（触壳熔体素 36/241/1143、邻非实体 738/3006/10449、`min(fv)=0.500`、Lx 达 1.15–1.20 mm
而试片只有 1.2 mm）⇒ 当时的口径差 `Vn−Vs` **同时含两类东西**：
  (L) 液相面口径：体素中心＋硬阈值 vs 亚格份额（第 5 轮已量化＝随加密近似减半、一阶）。
  (S) 自由面口径：熔区里的格按"中心在实体内"算（Vs）还是按"亚格交集"算（本轮 VsS）。
本轮在**同一次求解**上造一条口径梯子（液相侧固定在亚格，实体侧三档递进）：
    Vn  ＝ 中心液相 ∩ 中心实体（硬计数，A0 现断言量）
    Vs  ＝ 亚格液相 ∩ 中心实体（第 5 轮）
    Vsp ＝ 亚格液相 × fv 份额（用生产的线性 cut 份额当实体口径 ⇒ 第三副实体口径）
    VsS ＝ 亚格液相 ∩ **亚格实体**（SDF 也做三重线性重建，子体素中心同时判 F̃>TL 与 sdf̃<0）
于是 `ΔVn ≡ ΔVsS + Δ(Vs−VsS) + Δ(Vn−Vs)` 是恒等式（**代数恒真**：K2 的 1e-15 残差只校浮点结合顺序
与誊写，**不是**独立证据；真正有信息量的是 K3 的三条份额各自多大）。

跑前登记的判据（**K0 未过 ⇒ 全轮作废**；K0 只登记"正确实现必然通过"的东西）：
  K0a 双线性场（**生产约定：液相 =F>0、实体 =sdf<0**）：F=x−a（液相 x>a）、S=x−b（实体 x<b）
      ⇒ 交集＝解析平板 [a,b]，a=0.5173、b=0.8027（两个界面都**刻意不落在节点上**）、厚 0.2854。
      · 三重线性重建对**线性场是精确插值** ⇒ 子体素中心的 F̃/S̃ 与解析值差 <1e-12（机器量级）。
      · 交集份额必须**真为分数**（存在 φ_SS<φ_L 的格）⇒ 否则本轮问题（S 类口径）无从谈起。
      · 总量精度**不设拍脑袋的绝对门槛**：格是**节点中心**的，k 个节点在 y/z 上铺开的晶格面积是
        `(k·dx)²` 而不是 1（41 格 × 0.025 ＝ 1.025 ⇒ 面积因子 1.0506），所以解析值取
        `V_closed=(b−a)·(k·dx)²`；份额量子化的**严格**误差＝每个切割面 ≤dx/m ⇒
        包络 `2·(dx/m)/(b−a)`（两面），门槛就取这个包络本身，并把包络与实测两个数一起打印。
        （此处刻意不写"<1e-2"这类绝对值：那个门槛在**正确实现**上就可能触发＝第 5 轮缺陷 #15 的形状。）
  K0b **S 侧索引等价**（零容差、可失败）：把 S 换成恒负场（全域实体）⇒ φ_SS 必须与 φ_L **逐位相同**；
      换成恒正场（全域空气）⇒ φ_SS 必须**精确为 0**。这两条能抓住"S 用了错的位移/错的符号"。
      注：`φ_SS ≤ φ_L` 由构造（`liq & solid`）**必然**成立 ⇒ 只作回归哨，**不是**独立证据
      （第 5 轮的"同义反复断言"教训：不许把恒真式当判据）。
  K0c 正对照（探针必须看得见自由面）：球∩半空间 F=r0²−|x−c|²、S=x−xs（xs=0.5+0.5r0 ⇒ 实体侧
      x<xs ⇒ 切掉冠高 H=r0/2 的球冠；V解析=(4/3)πr0³−πH²(3r0−H)/3）在 k=21/41/81
      （r0/dx≈3.8/7.7/15.3）：
      · **四档梯子全对解析**：Vn/Vs/Vsp/VsS 各对 V解析、Σφ_L（无掩膜＝整球）对 V球，门槛都是
        |误差|<15%（最粗档 r0/dx=3.8 时硬计数本来就粗 ⇒ 15% 是"胡说检测"不是精度要求）。
      · 每档都存在 φ_SS<φ_L 的格，且**生产同款口径**上的自由面可见度 |Vs−VsS|/Vs > 1e-3。
        预期量级＝穿过自由面的液相 ≈A·dx/2 ⇒ 相对 ~1/(2·r0/dx)＝**3.3%（最细档）–13%（最粗档）**
        ⇒ 门槛 1e-3 比预期小 30×，正确实现不会误触发；真测出 0 就说明**本轮无法回答它的问题**。
      · 只打印不判：VsS 的表观阶、以及"S 侧切掉的体积 vs 解析球冠"——后者**刻意不设门槛**，
        因为这个差同时含 L 侧误差，两侧误差组合不出严格界（合成场里恰好大幅相消，不能当证据）。
  K0c 的跑前修正（**依据＝落盘 0 的纯 numpy 自检，未做任何生产求解**，与第 5 轮 S1a 同处置）：
      初稿把可见度写成 |Σφ_L−Σφ_SS|/Σφ_L 并把 eVs 定义为"无掩膜 Σφ_L 对 V解析"的误差。
      实测 16.8%/18.2% ⇒ 触发 VOID。查因：**Vs 按定义就不含 S 侧切割**（它只切液相面），
      拿"球−球冠"去要求它 = 把口径差当 bug ⇒ 门槛在正确实现上触发＝缺陷 #15 的形状。
      修正＝把梯子补全成四档（无掩膜那档改对 V球，生产口径那档＝亚格液相×中心掩膜，另加
      φ_L×φ_S 的乘积档），可见度改在生产同款口径上量。同一批修正另有三处（都在跑生产之前）：
      ①`sdf<0＝实体`的符号方向（初稿 S=b−x 把实体写反、交集成了半空间）；
      ②解析截面由 1×1 改为**节点中心晶格**的 `(k·dx)²`（41×0.025=1.025 ⇒ 面积因子 1.0506），
        否则 |VsS−解析| 会有 5% 的"体积差"其实是格面积口径差；
      ③K0b 初稿的 φ_SS≤φ_L 由 `liq & solid` **构造恒真** ⇒ 换成可失败的 S≡常值等价检验。
      K3 补登记分支 D（share_L 与 share_S 都 <0.2 ⇒ 残差在 share_F＝场本身漂移）。
  K1 逐位复现第 5 轮（同夹具、同 dx 拼写 `dx_um*1e-6`、src 未动）：8 格的 peak/Vn/Vs 与整数
      熔体素/界面格/触壳/邻非实体/nvox 必须等于 `am_t25r5_morphology_probe.log` L79-86 的印值
      （体积容差 5.1e-6＝第 5 位小数截断，peak 5e-3，整数逐位）。不成立 ⇒ 我的新代码动了旧路径
      ＝真 bug ⇒ 先查码、不记分。
  K2 恒等式残差 |ΔVn−(ΔVsS+Δ(Vs−VsS)+Δ(Vn−Vs))| < 1e-15 mm³（每个档对，含 δ=0.5）。
  K3 **归因记分**（本轮主判据；份额＝|该项|/|ΔVn|；四个信息档对＋δ=0.5 档对**都报**，
      判定只看信息档对，δ=0.5 列不删）：
        share_L＝|Δ(Vn−Vs)|/|ΔVn|　share_S＝|Δ(Vs−VsS)|/|ΔVn|　share_F＝|ΔVsS|/|ΔVn|
      注：三个量**带符号**相加才等于 ΔVn，所以三条份额可以合计 >1（符号相消），它们不是概率。
      A：share_S ≤ 0.2（四个信息档对全部）且 share_L ≥ 0.5 ⇒ 第 5 轮"主体＝液相面计数口径"不受影响。
      B：任一信息档对 share_S ≥ 0.2 ⇒ 两类口径**同居主因** ⇒ #29 的呈报必须加上
         "只换液相侧口径不足以闭合"。
      C：信息档对上 share_L ≤ 0.2 全部成立 ⇒ **推翻**第 5 轮分支 A ⇒ 回改 §26.21/§N/总览第 14 条。
      D（跑前补登记，代码顺序 A→C→B→D）：share_S、share_L **都** <0.2 ⇒ 两副口径都不解释 ⇒
         残差在 share_F＝**VsS 自己随 dx 漂**＝温度场/等值面位置的 O(dx) 偏差，与第 2 轮
         "δ=0.25 档 Lz=7.534%" 的分支 (ii) 相接 ⇒ #25 的战场从"计数口径"移回"场本身"。
      （0.2/0.5 是跑前写的**分类线**，不是"正确实现的可达值"——这是物理问题，允许失败。）
  K4 **域刀锋定量**：每格打印域质量 Mdom=Σfv·dx³、中心实体质量 Mcen=Σmask·dx³、nvox，
      以及熔区到六个面的最小间隙（index＝离实体壳几格；mm＝熔区最外中心离 ±EXT/2 面的余量）。
      K4b 结构一致性（**零容差、必须成立**）：`gap_index == 0 ⟺ (触壳>0 或 邻非实体>0)`。
      两个方向都推理过：熔格在实体最外层 ⇒ 要么该层就是网格外壳（触壳，边界邻居被 edge-clamp 成
      实体所以邻非实体可以是 0），要么外面那格非实体（邻非实体）；反向由"试片是盒、凸"保证。
      ⇒ 判据取 `==0` 而不是 `≤1`：熔格在实体面**内缩一格**时 gap=1 而两个触边量都可为 0，
      用 ≤1 会在正确实现上报警（又是 #15 的形状）。不成立就照实登记为定义问题，不静默改判据。
      另打印"δ=0 与 δ=0.25 同 dx 的域质量相对差 vs VsS 相对差"，只报数、不改判据。
  K5 **0.0799 mm³ 是口径无关物理量还是两类口径相消的巧合**：分母集合＝信息集
      δ∈{0,0.25}×三档＝**6 格**（第 5 轮那句"三档一致到 0.4%"用的是 δ=0 的 3 格，本轮**两列都报**）。
      spread＝(max−min)/mean。
      · spread(VsS)<2% 且 |mean(VsS)−0.0799|/0.0799<2% ⇒ 两副亚格口径下同一收敛值 ⇒ 可称口径无关
        （**限定在本夹具、本 δ 集合内**）。
      · spread(VsS)<2% 但均值偏离 >2% ⇒ 存在**另一个**收敛值，0.0799 只是"亚格液相∩中心实体"的产物
        ⇒ 档案措辞降级。
      · spread(VsS)≥2% ⇒ 口径无关的收敛值**不存在**，第 5 轮的"一致"是相消的巧合 ⇒ 回改档案三处。
      2% 的依据写在跑前：第 5 轮 Vs 在这 6 格上的极差是**实测 1.66%**（0.07930–0.08063），
      2% ＝"同一量级、不额外奖励"，跨过它必须给机制解释。

长期纪律照旧：**不放宽 5%、不改 A0 断言量、不删 δ=0.5 列、不改 `src/`、不调任何系数**
（N4：禁止把两个机制的补丁叠加后调系数凑绿）；自称 GPU 实测的 log 正文必须含设备清单行；
比物理量之前先比 `sdf.shape` 与**原始整数计数**（第 3 轮的 dx 拼写 ulp 教训）。
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
VOID = []


def shift(F, off):
    """整数位移 off∈{−1,0,1}³，边缘 edge-clamp（与第 5 轮逐字相同）。"""
    n0, n1, n2 = F.shape
    P = np.pad(F, 1, mode="edge")
    return P[1 + off[0]:1 + off[0] + n0, 1 + off[1]:1 + off[1] + n1,
             1 + off[2]:1 + off[2] + n2]


def stencil_max(F):
    """3³ 邻域最大值：候选＝三重线性重建**可能**达到 level 的格（凸组合 ⇒ 界外精确为 0）。"""
    n0, n1, n2 = F.shape
    P = np.pad(F, 1, mode="edge")
    out = P[0:n0, 0:n1, 0:n2].copy()
    for a, b, c in itertools.product((-1, 0, 1), repeat=3):
        out = np.maximum(out, P[a + 1:a + 1 + n0, b + 1:b + 1 + n1, c + 1:c + 1 + n2])
    return out


def subcell_fractions(F, S, level, m, debug_cell=None):
    """一次遍历同时得 φ_L（亚格液相）与 φ_SS（亚格液相 ∩ 亚格实体 {S̃<0}）。

    返回 (phiL, phiSS, 界面格数, debug 格的 (F̃,S̃) 列表)。两副份额都**未**加中心实体掩膜；
    掩膜是**口径选择**，由调用方显式施加（这正是本轮要量的东西）。
    """
    cand = stencil_max(F) >= level
    I, J, K = np.nonzero(cand)
    phiL = np.zeros(F.shape, dtype=np.float64)
    phiSS = np.zeros(F.shape, dtype=np.float64)
    dbg = None
    if I.size == 0:
        return phiL, phiSS, 0, dbg
    shF = {o: shift(F, o)[I, J, K] for o in itertools.product((-1, 0, 1), repeat=3)}
    shS = {o: shift(S, o)[I, J, K] for o in itertools.product((-1, 0, 1), repeat=3)}
    pos = None
    if debug_cell is not None:
        widx = np.nonzero((I == debug_cell[0]) & (J == debug_cell[1])
                          & (K == debug_cell[2]))[0]
        pos = int(widx[0]) if widx.size else None
    u = (np.arange(m) + 0.5) / m - 0.5
    d = np.sign(u).astype(int)
    w = np.abs(u)
    hitL = np.zeros(I.size, dtype=np.int64)
    hitSS = np.zeros(I.size, dtype=np.int64)
    vals = [] if pos is not None else None
    for i, j, k in itertools.product(range(m), repeat=3):
        ox, oy, oz = d[i], d[j], d[k]
        valF = np.zeros(I.size, dtype=np.float64)
        valS = np.zeros(I.size, dtype=np.float64)
        for ai, aj, ak in itertools.product((0, 1), repeat=3):
            coef = (w[i] if ai else 1.0 - w[i]) * (w[j] if aj else 1.0 - w[j]) \
                * (w[k] if ak else 1.0 - w[k])
            node = shF[(ai * ox, aj * oy, ak * oz)]
            valF += coef * node
            valS += coef * shS[(ai * ox, aj * oy, ak * oz)]
        liq = valF > level
        hitL += liq
        hitSS += liq & (valS < 0.0)
        if vals is not None:
            vals.append((float(valF[pos]), float(valS[pos])))
    phiL[I, J, K] = hitL / float(m ** 3)
    phiSS[I, J, K] = hitSS / float(m ** 3)
    nif = int(((phiL > 1e-12) & (phiL < 1.0 - 1e-12)).sum())
    return phiL, phiSS, nif, vals


# ==================================================== K0 估计器自检（先于任何生产求解）
print("=== K0a 双线性场：F=x−a（液相 x>a）、S=x−b（实体 x<b）⇒ 交集＝解析平板 [a,b] ===",
      flush=True)
k, a0, b0 = 41, 0.5173, 0.8027
dxp = 1.0 / (k - 1)
xs = np.arange(k) * dxp
Xg, Yg, Zg = np.meshgrid(xs, xs, xs, indexing="ij")
Fa, Sa = Xg - a0, Xg - b0                              # 约定：sdf<0 ＝实体 ⇒ 实体是 x<b0
AREA = (k * dxp) ** 2                                  # 节点中心晶格在 y/z 上铺开的面积（≠1）
icell = 21
pL, pSS, nifA, vals = subcell_fractions(Fa, Sa, 0.0, 8, debug_cell=(icell, 5, 5))
uu = (np.arange(8) + 0.5) / 8 - 0.5
combo = np.array(list(itertools.product(uu, repeat=3)))
xc = xs[icell] + combo[:, 0] * dxp
errF = float(np.max(np.abs(np.array([v[0] for v in vals]) - (xc - a0))))
errS = float(np.max(np.abs(np.array([v[1] for v in vals]) - (xc - b0))))
V_L = float(pL.sum()) * dxp ** 3
V_SS = float(pSS.sum()) * dxp ** 3
V_hard = float(((Fa > 0.0) & (Sa < 0.0)).sum()) * dxp ** 3
V_closed = (b0 - a0) * AREA                            # 厚 b−a、截面＝晶格面积
env = 2.0 * (dxp / 8.0) / (b0 - a0)                    # 两面量子化的**严格**相对误差包络
print(f"  重建误差：F̃={errF:.3e}  S̃={errS:.3e} ⇒ {'PASS' if max(errF, errS) < 1e-12 else 'FAIL'}",
      flush=True)
print(f"  切割自由面的格 i={icell}: φ_L={pL[icell, 5, 5]:.4f} φ_SS={pSS[icell, 5, 5]:.4f} "
      f"分数交集格数={int((pSS < pL - 1e-18).sum())}", flush=True)
print(f"  体积：VsS={V_SS:.9f} Vs={V_L:.9f} Vn(双侧硬计数)={V_hard:.9f} 解析={V_closed:.9f}"
      f"（厚={b0 - a0:.4f}×面积={AREA:.6f}）| |VsS−解析|/解析="
      f"{abs(V_SS - V_closed) / V_closed:.3e}（严格包络 2·(dx/m)/(b−a)={env:.3e}）", flush=True)
if max(errF, errS) >= 1e-12:
    VOID.append(f"K0a：线性场重建误差 F={errF:.3e} S={errS:.3e} ≥1e-12 ⇒ 插值机器有 bug")
if abs(V_SS - V_closed) / V_closed >= env:
    VOID.append(f"K0a：交集体积误差 {abs(V_SS - V_closed) / V_closed:.3e} 超出估计器的**严格**包络 "
                f"{env:.3e}")
if int((pSS < pL - 1e-18).sum()) == 0:
    VOID.append("K0a：没有 φ_SS<φ_L 的格 ⇒ 自由面口径在本夹具里不可见，本轮问题无法回答")

print("\n=== K0b S 侧索引等价：S≡负 ⇒ φ_SS 逐位＝φ_L；S≡正 ⇒ φ_SS 精确＝0 ===", flush=True)
_, pSS_all, _, _ = subcell_fractions(Fa, np.full_like(Fa, -1.0), 0.0, 8)
_, pSS_none, _, _ = subcell_fractions(Fa, np.full_like(Fa, 1.0), 0.0, 8)
eq_bits = bool(np.array_equal(pSS_all, pL))
zero_exact = bool(np.all(pSS_none == 0.0))
sentinel = int((pSS > pL + 1e-18).sum())
print(f"  全域实体：与 φ_L 逐位相同={eq_bits}   全域空气：φ_SS 精确为 0={zero_exact}"
      f"   ⇒ {'PASS' if eq_bits and zero_exact else 'FAIL'}", flush=True)
print(f"  回归哨（**由构造恒真、不作证据**）：φ_SS>φ_L 违反格数={sentinel}", flush=True)
if not eq_bits:
    VOID.append(f"K0b：S≡−1（全域实体）时 φ_SS 与 φ_L 不逐位相同 ⇒ S 侧位移/权重有 bug")
if not zero_exact:
    VOID.append("K0b：S≡+1（全域空气）时 φ_SS 非零 ⇒ 实体判据的符号方向反了")

print("\n=== K0c 正对照：球∩半空间（解析＝球−球冠）；四档口径梯子逐档对解析 ===", flush=True)
r0 = 0.1913
V_ball = 4.0 / 3.0 * np.pi * r0 ** 3
Hcap = r0 / 2.0
V_cap = np.pi * Hcap ** 2 * (3 * r0 - Hcap) / 3.0
V_true = V_ball - V_cap                                  # 实体＝平面左侧 ⇒ 去掉右球冠
syn = {}
for kk in (21, 41, 81):
    d = 1.0 / (kk - 1)
    sel = np.abs(np.arange(kk) * d - 0.5) <= r0 + 1.5 * d   # 裁到球外接盒＋2 格外壳（省算力，
    axl = (np.arange(kk) * d)[sel]                          # 且裁掉的区域 φ_L 精确＝0）
    Xb, Yb, Zb = np.meshgrid(axl, axl, axl, indexing="ij")
    Fb = r0 ** 2 - ((Xb - 0.5) ** 2 + (Yb - 0.5) ** 2 + (Zb - 0.5) ** 2)
    Sb = Xb - (0.5 + 0.5 * r0)                        # sdf<0 ⇒ 实体在平面**左**侧 ⇒ 切掉右冠
    mkc = Sb < 0.0                                        # 中心实体判据（＝生产 Vs 用的掩膜）
    qL, qSS, nifb, _ = subcell_fractions(Fb, Sb, 0.0, 8)
    qS, _, _, _ = subcell_fractions(-Sb, Fb, 0.0, 8)      # 亚格实体份额（F'=-S ⇒ liq 判据即 S̃<0）
    v = float(((Fb > 0.0) & mkc).sum()) * d ** 3          # Vn 档：中心液相 ∩ 中心实体
    ls = float((qL * mkc).sum()) * d ** 3                 # Vs 档：亚格液相 ∩ 中心实体（＝生产口径）
    vsp = float((qL * qS).sum()) * d ** 3                 # Vsp 档：亚格液相 × 亚格实体份额
    ss = float(qSS.sum()) * d ** 3                        # VsS 档：亚格液相 ∩ 亚格实体（精确交集）
    liq_only = float(qL.sum()) * d ** 3                   # 只切液相面＝整球（L 侧单独体检）
    syn[kk] = dict(Vn=v, Vs=ls, Vsp=vsp, VsS=ss, liq=liq_only,
                   strict=int((qSS < qL - 1e-18).sum()), nif=nifb,
                   eVn=(v - V_true) / V_true, eVs=(ls - V_true) / V_true,
                   eVsp=(vsp - V_true) / V_true, eVsS=(ss - V_true) / V_true,
                   eLiq=(liq_only - V_ball) / V_ball,
                   vis_mask=abs(ls - ss) / ls, vis_unmask=abs(liq_only - ss) / liq_only)
    print(f"  k={kk:3d} r0/dx={r0 / d:5.2f} 格数={Xb.size:7d} | Vn={v:.6f}({syn[kk]['eVn'] * 100:+6.2f}%) "
          f"Vs={ls:.6f}({syn[kk]['eVs'] * 100:+6.2f}%) Vsp={vsp:.6f}({syn[kk]['eVsp'] * 100:+6.2f}%) "
          f"VsS={ss:.6f}({syn[kk]['eVsS'] * 100:+6.2f}%) 解析={V_true:.6f}", flush=True)
    print(f"      L 侧单独体检：Σφ_L·dx³={liq_only:.6f} 对 V球={V_ball:.6f}（{syn[kk]['eLiq'] * 100:+.2f}%）"
          f" | 严格不等格={syn[kk]['strict']:5d} 界面格={nifb:5d} "
          f"自由面可见度：掩膜口径={syn[kk]['vis_mask']:.3e} 无掩膜={syn[kk]['vis_unmask']:.3e}",
          flush=True)
    if syn[kk]['strict'] == 0 or syn[kk]['vis_mask'] <= 1e-3:
        VOID.append(f"K0c k={kk}：看不见自由面（严格不等格={syn[kk]['strict']} "
                    f"掩膜可见度={syn[kk]['vis_mask']:.3e} ≤1e-3）⇒ 本轮问题无法回答")
    for lab in ("eVn", "eVs", "eVsp", "eVsS", "eLiq"):
        if abs(syn[kk][lab]) >= 0.15:
            VOID.append(f"K0c k={kk} {lab}：误差 {syn[kk][lab] * 100:.1f}% ≥15% ⇒ 估计器不可信")
    if int((qSS > qL + 1e-18).sum()):
        VOID.append(f"K0c k={kk}：φ_SS>φ_L ⇒ bug")
p0 = np.log(abs(syn[21]['eVsS']) / abs(syn[41]['eVsS'])) / np.log(2.0)
p1 = np.log(abs(syn[41]['eVsS']) / abs(syn[81]['eVsS'])) / np.log(2.0)
print(f"  只打印不判：VsS 误差 {syn[21]['eVsS'] * 100:+.2f}/{syn[41]['eVsS'] * 100:+.2f}/"
      f"{syn[81]['eVsS'] * 100:+.2f}% 表观阶 {p0:.2f}/{p1:.2f}；"
      f"硬计数 Vn 误差符号={np.sign([syn[kk]['eVn'] for kk in (21, 41, 81)])}（摆动＝对齐运气）",
      flush=True)
for kk in (21, 41, 81):
    rem = syn[kk]['liq'] - syn[kk]['VsS']
    print(f"  只打印不判 k={kk:3d}：S 侧切掉的体积={rem:.6f} 解析球冠={V_cap:.6f}"
          f"（{abs(rem - V_cap) / V_cap * 100:.2f}%）——**不设门槛**：该差同时含 L 侧误差，"
          f"两侧误差不可组合出严格界（本轮实测它们大幅相消）", flush=True)
if VOID:
    print(f"\nFAILS（K0 未过 ⇒ 全轮作废）: {VOID}", flush=True)
    sys.exit(1)
print("  ⇒ K0 全过（估计器可用；生产侧才开始花钱）", flush=True)


# ============================================================ 生产（与第 5 轮同一副夹具）
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
print(f"  platform/backend = {jax.default_backend()} / x64={jax.config.jax_enable_x64}", flush=True)

print("\n=== 生产求解＋口径梯子 Vn→Vs→Vsp→VsS＋域质量＋到六面间隙（无 patch、src 未动）===", flush=True)
print(f"  T_liquidus={TL:.1f} K  几何体积={EXT[0] * EXT[1] * EXT[2] * 1e9:.6f} mm³", flush=True)
R = {}
for dx_um in TIERS_UM:
    for frac in (DELTAS if dx_um != 12.5 else [0.0, 0.25]):     # 与第 5 轮同一采集集
        dx = dx_um * 1e-6                                        # 拼写＝第 5 轮（K1 前提）
        g = coupon(dx, frac * dx)
        th = solve_enthalpy_thermal(geometry=g, process=plan(),
                                    params={"material": "316L"})
        F = np.asarray(th.peak_temperature, dtype=np.float64)
        sdf = np.asarray(g.sdf, dtype=np.float64)
        mk = np.asarray(solid_mask(sdf), dtype=np.float64) > 0.5
        w = np.asarray(solid_weight(sdf, dx), dtype=np.float64)
        above = (F > TL) & mk
        vol = dx ** 3 * 1e9
        pL4, pSS4, nif4, _ = subcell_fractions(F, sdf, TL, 4)
        pL8, pSS8, nif8, _ = subcell_fractions(F, sdf, TL, 8)
        viol = int((pSS8 > pL8 + 1e-18).sum())                   # 构造恒真的回归哨（非证据）
        if viol:
            bad.append(f"K0b 哨 δ={frac} dx={dx_um}：{viol} 格 φ_SS>φ_L（未掩膜）⇒ bug")
        mL8 = pL8 * mk                                           # Vs 的口径＝中心实体掩膜
        mL4 = pL4 * mk
        sh = F.shape
        i0, j0, k0 = np.unravel_index(int(np.argmax(F * mk)), sh)
        ax = [np.asarray(g.origin, dtype=np.float64)[d] + np.arange(sh[d]) * dx
              for d in range(3)]
        edge_cells = int(above.sum() - (above[1:-1, 1:-1, 1:-1]).sum())
        bnd = 0
        for o in itertools.product((-1, 0, 1), repeat=3):
            if o != (0, 0, 0):
                bnd += int((above & ~shift(mk, o)).sum())
        gap_idx, margin_mm = [], []
        for d in range(3):
            red = tuple(i for i in range(3) if i != d)
            si = np.nonzero(mk.any(axis=red))[0]
            mi = np.nonzero(above.any(axis=red))[0]
            gap_idx += [int(mi[0] - si[0]), int(si[-1] - mi[-1])]
            co = np.abs(np.asarray(ax[d], dtype=np.float64)[mi])
            margin_mm.append(float((EXT[d] / 2 - co.max()) * 1e6))
        rec = dict(nvox=int(mk.sum()), nml=int(above.sum()), peak=float(F[i0, j0, k0]),
                   Vn=float(above.sum()) * vol, Vm=float((above * w).sum()) * vol,
                   Vs=float(mL8.sum()) * vol, Vs4=float(mL4.sum()) * vol,
                   Vsp=float((pL8 * w).sum()) * vol, VsS=float(pSS8.sum()) * vol,
                   VsS4=float(pSS4.sum()) * vol,
                   nif=nif8, nif4=nif4, edge=edge_cells, bnd=bnd, viol=viol,
                   nstrict=int((pSS8 < pL8 - 1e-18).sum()),
                   Mdom=float(w.sum()) * vol, Mcen=float(mk.sum()) * vol,
                   gap_idx=int(min(gap_idx)), margin=float(min(margin_mm)),
                   margins=margin_mm, fv_min=float(w[above].min()) if above.sum() else 1.0)
        R[(frac, dx_um)] = rec
        print(f"  δ={frac:4.2f}·dx dx={dx_um:5.1f} nvox={rec['nvox']:7d} "
              f"peak={rec['peak']:7.2f} 熔体素={rec['nml']:6d} 界面格={rec['nif']:6d} "
              f"触壳={rec['edge']} 邻非实体={rec['bnd']} 严格不等格={rec['nstrict']} "
              f"min(fv)={rec['fv_min']:.3f} |", flush=True)
        print(f"      Vn={rec['Vn']:.5f} Vs={rec['Vs']:.5f} Vsp={rec['Vsp']:.5f} "
              f"VsS={rec['VsS']:.5f} mm³（VsS(m4)={rec['VsS4']:.5f} ⇒ m 敏感性="
              f"{abs(rec['VsS'] - rec['VsS4']) / rec['VsS'] * 100:.3f}%）| "
              f"Mdom={rec['Mdom']:.5f} Mcen={rec['Mcen']:.5f} mm³ | "
              f"到面间隙={rec['gap_idx']}格 余量={rec['margin']:+.1f}µm "
              f"(三面={rec['margins'][0]:+.1f}/{rec['margins'][1]:+.1f}/{rec['margins'][2]:+.1f})",
              flush=True)

print("\n=== 自由面口径在生产夹具上的可见度（K0c 同款量，非判决）===", flush=True)
for key in sorted(R, key=lambda t: (t[1], t[0])):
    r = R[key]
    print(f"  δ={key[0]:4.2f} dx={key[1]:5.1f}: 严格不等格={r['nstrict']:6d} "
          f"|Vs−VsS|/Vs={abs(r['Vs'] - r['VsS']) / r['Vs']:.4e} "
          f"|Vsp−VsS|/VsS={abs(r['Vsp'] - r['VsS']) / r['VsS']:.4e} 构造哨 φ_SS>φ_L={r['viol']}",
          flush=True)

print("\n=== K1 逐位复现第 5 轮（其 log L79-86 印值）===", flush=True)
ARC = {  # (δ, dx_um): (peak, Vn, Vs, nml, nif, edge, bnd, nvox)
    (0.0, 50.0): (2461.27, 0.06025, 0.08030, 482, 844, 36, 0, 2925),
    (0.25, 50.0): (2447.27, 0.06275, 0.08063, 502, 876, 0, 738, 2304),
    (0.50, 50.0): (2433.15, 0.06400, 0.07678, 512, 734, 0, 486, 2304),
    (0.0, 25.0): (2563.01, 0.06947, 0.08006, 4446, 3604, 241, 0, 20825),
    (0.25, 25.0): (2554.96, 0.07062, 0.07930, 4520, 3988, 0, 3006, 18432),
    (0.50, 25.0): (2565.18, 0.06473, 0.07303, 4143, 3936, 0, 2970, 18432),
    (0.0, 12.5): (2565.08, 0.07418, 0.07995, 37982, 14484, 1143, 0, 156849),
    (0.25, 12.5): (2555.81, 0.07453, 0.07948, 38157, 15668, 0, 10449, 147456),
}
nhit = 0
for key, (ap, an, avs, anml, anif, aedge, abnd, anvox) in ARC.items():
    r = R[key]
    dp, dn, dvs = abs(r['peak'] - ap), abs(r['Vn'] - an), abs(r['Vs'] - avs)
    ints_ok = (r['nml'] == anml and r['nif'] == anif and r['edge'] == aedge
               and r['bnd'] == abnd and r['nvox'] == anvox)
    same = dp < 5e-3 and dn < 5.1e-6 and dvs < 5.1e-6 and ints_ok
    nhit += same
    print(f"  δ={key[0]:4.2f} dx={key[1]:5.1f}: peak(Δ{dp:.3f}) Vn(Δ{dn:.6f}) Vs(Δ{dvs:.6f}) "
          f"整数(熔体素/界面格/触壳/邻非实体/nvox)={'全对' if ints_ok else '有差'} ⇒ "
          f"{'一致' if same else '不一致'}", flush=True)
    if not same:
        bad.append(f"K1 δ={key[0]} dx={key[1]}：Δpeak={dp:.3f} ΔVn={dn:.6f} ΔVs={dvs:.6f} "
                   f"整数ok={ints_ok} ⇒ 新代码动了旧路径")
print(f"  ⇒ {nhit}/{len(ARC)} 格复现", flush=True)


def rel(x, y):
    return abs(x - y) / max(x, y)


PAIRS = [(50.0, 25.0), (25.0, 12.5)]
print("\n=== K2＋K3 恒等式与归因记分：ΔVn ≡ ΔVsS + Δ(Vs−VsS) + Δ(Vn−Vs) ===", flush=True)
sh_S, sh_L, sh_F = [], [], []
for f in (0.0, 0.25, 0.5):
    for (d1, d2) in PAIRS:
        if (f, d2) not in R:
            continue
        x, y = R[(f, d1)], R[(f, d2)]
        dn = x['Vn'] - y['Vn']
        pF = x['VsS'] - y['VsS']
        pS = (x['Vs'] - x['VsS']) - (y['Vs'] - y['VsS'])
        pL = (x['Vn'] - x['Vs']) - (y['Vn'] - y['Vs'])
        resid = abs(dn - (pF + pS + pL))
        sL, sS, sF = abs(pL) / abs(dn), abs(pS) / abs(dn), abs(pF) / abs(dn)
        print(f"  δ={f:4.2f}·dx {d1:5.1f}↔{d2:5.1f}: |ΔVn|={abs(dn):.5f}mm³"
              f"({abs(dn) / y['Vn'] * 100:6.3f}%) 残差={resid:.2e} ‖ 符号项：L={pL:+.5f} "
              f"S={pS:+.5f} F={pF:+.5f} ‖ share_L={sL:5.3f} share_S={sS:5.3f} share_F={sF:5.3f}",
              flush=True)
        print(f"      口径水平值 Vn−Vs：{x['Vn'] - x['Vs']:+.5f}/{y['Vn'] - y['Vs']:+.5f}　"
              f"Vs−VsS：{x['Vs'] - x['VsS']:+.5f}/{y['Vs'] - y['VsS']:+.5f}　"
              f"Vsp−VsS：{x['Vsp'] - x['VsS']:+.5f}/{y['Vsp'] - y['VsS']:+.5f}", flush=True)
        if resid >= 1e-15:
            bad.append(f"K2 δ={f} dx={d1}↔{d2}：恒等式残差 {resid:.2e} ≥1e-15 ⇒ 分解有 bug")
        if f < 0.4:
            sh_L.append(sL)
            sh_S.append(sS)
            sh_F.append(sF)

print("\n=== K3 裁决（跑前写死的 A/C/B/D，按此顺序判）===", flush=True)
print(f"  信息档对：share_S max={max(sh_S):.3f} min={min(sh_S):.3f}；"
      f"share_L min={min(sh_L):.3f} max={max(sh_L):.3f}；share_F max={max(sh_F):.3f}", flush=True)
if max(sh_S) <= 0.2 and min(sh_L) >= 0.5:
    verdict = ("A：四个信息档对上自由面口径份额都 ≤0.2 ⇒ 第 5 轮『主体＝液相面计数口径』不受影响"
               "（VsS 只是把 (S) 量出来、确认它小）")
elif max(sh_L) <= 0.2:
    verdict = "C：信息档对上 share_L 全部 ≤0.2 ⇒ **推翻**第 5 轮分支 A ⇒ 回改 §26.21/§N/总览第 14 条"
elif max(sh_S) >= 0.2:
    verdict = ("B：存在 share_S ≥0.2 的档对 ⇒ 两类口径**同居主因** ⇒ #29 的呈报必须带上"
               "『只换液相侧口径不足以闭合』")
else:
    verdict = ("D：share_S 与 share_L 在全部信息档对上都 <0.2 ⇒ 两副**计数口径**都不解释 ΔVn，"
               "残差落在 share_F（VsS 自身随 dx 漂移）⇒ 主因回到温度场/等值面位置的 O(dx) 偏差"
               "（与第 2 轮 δ=0.25 档 Lz=7.534% 的分支 (ii) 相接）⇒ #25 战场从『计数口径』移回『场本身』")
print(f"  ⇒ {verdict}", flush=True)

print("\n=== K4 域刀锋定量 ===", flush=True)
print(f"  几何体积＝{EXT[0] * EXT[1] * EXT[2] * 1e9:.6f} mm³（Mdom 应 ≤ 它，Mcen 是中心计数版）",
      flush=True)
for dx_um in TIERS_UM:
    line = f"  dx={dx_um:5.1f}:"
    for f in DELTAS:
        if (f, dx_um) not in R:
            continue
        r = R[(f, dx_um)]
        line += (f"  δ={f:4.2f} Mdom={r['Mdom']:.5f} Mcen={r['Mcen']:.5f} nvox={r['nvox']:6d}"
                 f" gap={r['gap_idx']:2d}格 余量={r['margin']:+6.1f}µm")
    print(line, flush=True)
    for f in DELTAS:
        if (f, dx_um) not in R:
            continue
        r = R[(f, dx_um)]
        touch = r['edge'] > 0 or r['bnd'] > 0
        if (r['gap_idx'] == 0) != touch:
            bad.append(f"K4b δ={f} dx={dx_um}：gap_idx={r['gap_idx']} 与 触壳/邻非实体={touch} "
                       f"不一致 ⇒ 两个量度的定义需核对（触壳用网格壳、gap 用实体壳）")
print(f"  K4b（gap_idx==0 ⟺ 触壳>0 或 邻非实体>0）："
      f"{'全部一致' if not any(s.startswith('K4b') for s in bad) else '有不一致 ⇒ 见 FAILS'}",
      flush=True)
for dx_um in TIERS_UM:
    if (0.0, dx_um) in R and (0.25, dx_um) in R:
        a_, b_ = R[(0.0, dx_um)], R[(0.25, dx_um)]
        dm, dvol = rel(a_['Mdom'], b_['Mdom']), rel(a_['VsS'], b_['VsS'])
        print(f"  跨 δ（dx={dx_um:5.1f}）：域质量相对差={dm * 100:5.2f}%（Δnvox={a_['nvox'] - b_['nvox']:+6d}）"
              f" ‖ VsS 相对差={dvol * 100:5.2f}% ‖ Vn 相对差={rel(a_['Vn'], b_['Vn']) * 100:5.2f}% ⇒ "
              f"形态差{'不超过域差量级' if dvol <= dm else '**大于**域差 ⇒ 域刀锋解释不了全部，须另查'}"
              f"（只报数，不改判据）", flush=True)

print("\n=== K5 0.0799 mm³ 是口径无关量还是相消的巧合 ===", flush=True)
INFO = [(f, d) for f in (0.0, 0.25) for d in TIERS_UM]
REF = 0.0799


def spread(vals):
    return (max(vals) - min(vals)) / (sum(vals) / len(vals))


for lab, key in (("Vs（第 5 轮：亚格液相∩中心实体）", 'Vs'),
                 ("Vsp（亚格液相×fv 份额）", 'Vsp'),
                 ("VsS（亚格液相∩亚格实体）", 'VsS')):
    v6 = [R[k][key] for k in INFO]
    v3 = [R[(0.0, d)][key] for d in TIERS_UM]
    mean6 = sum(v6) / len(v6)
    sp6, sp3 = spread(v6), spread(v3)
    dev = abs(mean6 - REF) / REF
    tag1 = "稳（<2%）" if sp6 < 0.02 else "不稳（≥2%）"
    tag2 = "与 0.0799 同值" if dev < 0.02 else f"不是同一个收敛值（偏 {dev * 100:.2f}%）"
    print(f"  {lab:28s} 6 格={' '.join(f'{x:.5f}' for x in v6)}", flush=True)
    print(f"    {'':28s} spread6={sp6 * 100:5.2f}% spread(δ=0 三档)={sp3 * 100:5.2f}% "
          f"mean6={mean6:.5f} 偏差6={dev * 100:5.2f}% ⇒ {tag1}／{tag2}", flush=True)

spS = spread([R[k]['VsS'] for k in INFO])
devS = abs(sum(R[k]['VsS'] for k in INFO) / len(INFO) - REF) / REF
spL = spread([R[k]['Vs'] for k in INFO])
if spS < 0.02 and devS < 0.02:
    k5 = ("VsS 在 6 格上既稳又与 0.0799 同值 ⇒ 该收敛值在**两副亚格口径**下都成立"
          "（限定：本夹具、δ∈{0,0.25} 这个域刀锋集合内）")
elif spS < 0.02:
    k5 = (f"VsS 稳（{spS * 100:.2f}%）但均值偏离 0.0799 达 {devS * 100:.2f}% ⇒ 存在**另一个**收敛值，"
          f"第 5 轮的 0.0799 只是『亚格液相∩中心实体』这一副口径的产物 ⇒ 档案措辞降级")
else:
    k5 = (f"VsS 的 6 格极差 {spS * 100:.2f}% ≥2% ⇒ 口径无关的收敛值**不存在**；第 5 轮"
          f"『三档一致到 0.4%』（δ=0 三格、Vs 口径）是两类口径相消的巧合 ⇒ 回改 §26.21/§N/总览第 14 条")
print(f"  ⇒ K5：{k5}", flush=True)
print(f"  对照：同 6 格上 Vs 极差={spL * 100:.2f}%（第 5 轮那句 0.4% 的分母是 δ=0 三格）", flush=True)

print("\n=== 记分 ===", flush=True)
print("FAILS:", bad if bad else "无", flush=True)
print("注：本轮不改 src/、不改 A0 断言量、不放宽 5%、不删 δ=0.5 列、不调任何系数（N4 未触碰）；"
      "Vsp/VsS 是归因诊断量，禁止替换 assert2 的观测量。", flush=True)
print(f"elapsed={time.time() - T0:.0f}s", flush=True)
