# -*- coding: utf-8 -*-
"""#25 第 7 轮：把第 6 轮剩下的那句"域刀锋解释不了全部"拆开——**cap_w 刀锋层**还是**域体积**？

第 6 轮（`am_t25r6_facesplit_probe.log`，K3＝分支 A、K4）留下的唯一不闭合项：同 dx 跨 δ 的**形态差**
（VsS 3.45/1.65/0.88% @ dx=50/25/12.5）**大于**同处的**域质量差**（Mdom 0.94/0.24/0.06%）
⇒ 域刀锋（δ=0 比 δ=0.25/0.5 多一整层表面体素）**解释不了全部**。第 6 轮写下的候选机制是：
δ=0 那层表面体素 `fv=0.5` **同时**是 #23 的 `cap_w = max(fv, CUT_CAPACITY_FLOOR=0.5)` 的**下限作用处**
⇒ 那里"每单位能量的温升"是内层的 2 倍 ⇒ 这是**热容**差异，不只是**体积**差异。本轮去判它。

**本轮不动 `src/`**：消融用 **monkeypatch** `TE.CUT_CAPACITY_FLOOR`（`try/finally` 复原），
**只诊断、不进默认链、不写进任何断言**（N4：禁止把两个机制的补丁叠加后调系数凑绿）。三臂：
    P  ＝ floor 0.5（生产，#23 现行）
    A1 ＝ floor 1.0 ⇒ `cap_w ≡ max(fv,1) = 1` ⇒ **rhs 不再被除** ＝ **#23 之前的口径**（主归因臂）
    A2 ＝ floor 0.75 ⇒ 只重判 `fv<0.75` 的格（低份额壳／空气格）
归因：`G_P`/`G_A1` ＝ 同 dx 的 |X(δ=0)−X(δ=0.25)|/max（X∈{VsS 主、Vn 次}），`D` ＝同处的域质量差
⇒ **A1 把 cap_w 这条通道整个关掉**，剩下的就是体积/几何 ⇒ `share_capw = (G_P − G_A1)/G_P`。

跑前登记的判据（**R0 任一条不过 ⇒ 全轮作废**；R0 只登记"正确实现＋真实物理必然通过"的东西）：
  R0a **确定性**（零容差）：同一臂、同一夹具连跑两次 ⇒ `peak_temperature` 与 `final_temperature`
      **逐字节相同**。不成立 ⇒ 环境不确定，本轮所有"逐位"判据全部失效 ⇒ VOID。
  R0b **消融确实生效**（正对照，**能失败**）：A1 相对 P 必须有可测差异。
      · 先验（纯算术，无求解）：`max(fv,1.0)` 与 `max(fv,0.5)` 的元素级不等格数 > 0（除 fv=1 外全部不等）；
      · 后验（小夹具求解）：`peak_temperature` 与 `final_temperature` 两个数组都**逐位不等**，
        且**全场** `max|ΔT| ≥ 0.1 K`。
      门槛依据（正确实现必过）：#23 在 δ=0 档把计算域质量与剂量的错配置换掉，**实测峰值代际差是
      ② 2349.07/2478.97/2517.98 K → ③ 2461.27/2563.01/2565.08 K**（开发日志 §26.18／#23 验收，
      同档 3.5–5.7% ＝ 80–140 K 量级）⇒ 0.1 K 比预期小**三个数量级**，正确实现不会误触发。
      **两数组都逐位相同就说明 floor 这个全局量没被读取（例如被缓存的 trace 吃掉）⇒ 本轮的"消融"是假的 ⇒ VOID。**
      【跑前修正 R0b-1（依据＝落盘 0 的 prefix 实测，见 log 的「prefix 修正」段）】原判据用 `|Δpeak| ≥ 0.1 K`，
      在 **δ=0.5** 档实测 `Δpeak=0.000 K` 而数组**逐位不等**——这不是 patch 死，而是**#23 在该档可证为恒等操作**：
      δ=0.5 时实体内 `fv≡1`（prefix 实测 `min(fv|实体)=1.000000`）⇒ 实体内右端项逐位不变 ⇒ 峰值格不动。
      ⇒ 判据改为**全场** `max|ΔT|`，并把"δ=0.5 档 Δpeak **恰为 0**"登记为**新的、能失败的预测**。
      【跑前修正 R0b-2（依据＝落盘 0 的第二轮 prefix 实测）】改完全场 `max|ΔT|` 仍在 δ=0.5 档 FAIL，实测
      `max|ΔT|=9.095e-13 K`（＝2300 K 的 4e-16 ⇒ 纯舍入）：该档改动集 279 格里 **218 格 fv=0（分子按 :840 恒 0）、
      61 格是 `fv=1−1ulp`** ⇒ **有材料且真被切开（`EPS<fv<1−EPS`）的格数=0** ⇒ A1 在该档是**算术恒等**、
      不是 patch 死。这是 T2 刀锋家族在 **cap_w 通道**的第一次现身（面落在体素面上 ⇒ 表面节点 sdf=−0.5dx
      的舍入给出 `0.5−sdf/dx = 1−ulp`；生产夹具 δ=0.5 三档实测 `fv=1−1ulp` 格数＝**96/1136/4576**，
      而 δ=0/0.25 两档此数为 0 ⇒ 该舍入层是"面恰落在体素面上"特有的产物，与 §25.7 的 1-ulp 翻皮同源）。
      ⇒ R0b 改成**两支、两侧都能失败**的判据，用 `n_far = 改动集里 `EPS<fv<1−EPS`（有材料**且**真被切）的格数`
      分派（空气格不计：它们的分子按 :840 恒为 0，改动 cap_w 不该有任何效应——若竟有效应，恒等档的
      `<1e-9 K` 那一侧会把它抓出来）：
        · `n_far > 0`（真有切割格）⇒ 必须 live（两个数组都逐位不等）**且** `max|ΔT| ≥ 0.1 K`，否则 VOID；
        · `n_far = 0`（恒等档）⇒ 必须 live-but-noise：**`max|ΔT| < 1e-9 K`**。这一侧同样能失败——
          若恒等档测出 ≥1e-9 K 的差异，就说明 floor 还经**除 cap_w 之外**的路径进入求解（真发现，登记缺陷）。
      附带收益：恒等档的 `max|ΔT|` 就是**本求解器在同夹具下的舍入噪声地板**，R3 的标量差读数低于该地板时
      不得解释为物理（prefix 实测 ≈9e-13 K）。
  R0c **消融的作用域**（两侧、**能失败**，且它检验的是源码里一句**从未被逐位检验过的论断**）：
      A2 把 `cap_w` 从 `max(fv,0.5)` 改成 `max(fv,0.75)`，元素级不等 ⟺ `fv < 0.75`。
      本轮**实测**（R0s，无求解）每个夹具的 fv 直方图，然后按下式**由测量值**给出预测（不靠猜测）：
        · 若被改动的格**全部满足 fv ≤ EPS**（＝代码自己的"无材料"边界 `fv==0`）⇒ 场必须**逐位不变**。依据＝
          `thermal_enthalpy.py:840` 的注释论断"空白单元（fv=0）分子恒为 0（面全闭 ⇒ lap=0；fv·项=0）
          除以 cap_w 仍是 0"——**这条论断此前从未被逐位检验**，它就是本轮的失败点：若空气格分子不恒为 0
          （例如被 `Q = 源 × fv` 之外的路径喂了能量），A2 就会改动结果 ⇒ 照实登记为**新缺陷**，不静默改判据；
        · 若被改动的格里**含有 fv > EPS 的格** ⇒ 场必须**逐位不等**。
      两侧都打印"改动格数（分 fv≤EPS / EPS<fv≤1e-6 / fv>1e-6）＋实测是否逐位相同"，并打印全场 `max|ΔT|`。
      【跑前修正 R0c-1】原判据的"有材料"分界取 `MAT_EPS=1e-6`（当时写下"fv=1e-17 这类巧合格分子不恒为 0，
      用 `>0` 会报警＝缺陷 #15 的形状"）。prefix 实测：9 格生产夹具与 3 档小夹具的 `EPS<fv≤1e-6` 改动格数
      **全为 0**，两种分界给出同一预测 ⇒ 该条不是靠放宽过关的；但**判据的正确分界应是代码自己的 `fv==0`
      （用 EPS）**，因为 `0<fv≤1e-6` 的格分子正比于 fv、换 cap_w 必给非零差 ⇒ 用 1e-6 当"无材料"会把
      一个真·非恒等臂误判成"应逐位不变"。故改用 EPS 判据、并把 `EPS<fv≤1e-6` 的格数**照样打印**用于诊断。
      ⚠ 同时保留 EPS=1e-9 的分带判等（避免 0.7499999999 这类浮点巧合在正确实现上报警）：prefix 实测
      δ=0.25·dx=50 档确有 **192** 格落在 `0.75−1ulp`，它们在**精确比较**的改动集里、在 **EPS 分带**的
      `[.75,1)` 带里 ⇒ 改动集一律用与代码同款的精确 `!=` 比较，分带只作读数。
  R0s **fv 直方图的结构预测**（纯几何、无求解，**能失败**）：δ=0 档实体内必有 `fv≈0.5` 的刀锋层
      （第 6 轮实测 min(fv|熔)=0.500）；δ=0.25/0.5 档实体内**不得**出现 `fv<0.75−1e-9`
      （第 6 轮实测 min(fv|熔)=0.750/1.000）。不成立 ⇒ 第 6 轮的机制叙述是错的 ⇒ VOID。
      （分带一律用 EPS=1e-9 的容差判等，避免 0.7499999999 这类浮点巧合在正确实现上报警。）
  R1 **逐位复现第 6 轮**（同夹具、同 dx 拼写 `dx_um*1e-6`、同 `subcell_fractions` 算术序、src 未动）：
      8 格的 peak/Vn/Vs/VsS/Mdom/Mcen 与整数（熔体素/界面格/触壳/邻非实体/nvox）必须等于
      `am_t25r6_facesplit_probe.log` L43-62 的印值（体积容差 5.1e-6＝第 5 位小数截断、peak 5e-3、整数逐位）。
      不成立 ⇒ 我的 patch 机器动了默认路径＝真 bug ⇒ 先查码、不记分。
      ⚠ 整数清单里**界面格**的定义＝第 6 轮的"严格部分亚格"（`0<φ_L<1`，未加中心掩膜），**不是**候选格数
      （`stencil_max(F)≥level` 的格数）——第 7 轮首跑把两者混了，R1 因此 0/8 报警（探针缺陷，非求解器）；
      候选格在本轮另列（`候选格=`），第 6 轮未印 ⇒ 不参与对表。
      **顺序要求**：R1 的 8 次求解全部在 `floor==0.5` 下完成，且打印求解前后的 `TE.CUT_CAPACITY_FLOOR` 值。
  R2 **场读数（不设门槛、不判，只把机制摆出来）**：每格按 fv 分带打印 ①格数（全网格／实体内）
      ②`Σφ_SS·dx³` 的分带熔体积与占比 ③峰值格的带归属 ④`final_temperature` 的内层/表面层 P90
      ⑤**空气格（mask=0）里的最高温**——⑤ 顺带检验 R0c 依赖的那句"空气格分子恒为 0"，
      并且是 #23 模块 docstring 里"切割壳温度不由本项物理决定"这条已知近似的**首次定量**。
  R3 **归因记分（本轮主判据）**：每个 dx 一个档对（δ=0↔δ=0.25），X∈{VsS（主）、Vn（次）}：
      `G_P = rel(X_P(δ=0), X_P(δ=0.25))`、`G_A1`、`G_A2`、`D = rel(Mdom(δ=0), Mdom(δ=0.25))`（与臂无关）
      `share_capw = (G_P − G_A1)/G_P`。分支（跑前写死的**分类线**，物理问题允许全不命中）：
        α（cap_w 通道＝超出体积的那部分的主因）：**三个 dx 全部**满足 `G_A1 ≤ max(2·D, 1.5%)`
           ⇒ #24 需加一条"刀锋层热容与对齐 δ 的耦合"；
        β（cap_w 通道不重要）：**三个 dx 全部**满足 `G_A1 ≥ 0.8·G_P` ⇒ 超出体积的那部分与 cap_w 无关
           ⇒ 归 #30（夹具触边）与 #24 的体积/吸附项处置；
        γ：其余 ⇒ 两通道同居 ⇒ 报数、两条都登记，不许只挑一条。
        （主判据看 VsS；若 VsS 与 Vn 落在不同分支 ⇒ **判 γ 并照实报**，不许挑对自己有利的量。）
      **1.5% 噪声容差的依据写在跑前**：第 6 轮实测 VsS 对子网格 m 的敏感性上界 **1.232%**
      （δ=0.25·dx=50 那格）＋印值截断 5e-6/0.077≈0.0065% ⇒ 取整到 1.5%。这**不是**"正确实现的可达值"，
      只是"不要把离散噪声算进机制"的 allowance，α 本身允许达不到。
      另报 `share_floor = (G_P − G_A2)/G_P`：A2 只重判低份额壳 ⇒ 与 A1（含全部切割格）对比可判
      "是刀锋层／低份额壳还是所有切割层"。
      ⚠ **消融覆盖面（跑前声明，防止把"测不到"读成"不存在"）**：prefix 实测 δ=0 档**全网格 `fv≤EPS` 的格数=0**
      （`nvox`＝格总数 ⇒ 实体掩膜处处为 1 ⇒ `_div_alpha_grad` 在该档**一个面都不闭合**），而 δ=0.25/0.5 有
      715/2575/9751 个空气格 ⇒ δ=0 与 δ=0.25 之间除 cap_w 之外还差**一整条边界通道**（绝热自由面在 δ=0 消失）。
      **A1/A2 都碰不到这条通道**（它由二值 `mask` 决定，不由 floor 决定）⇒ 若 R3 判 γ，残余的主候选是这条
      边界通道而非"未知的第三机制"，本轮**不**为它加臂（加臂要动 mask＝改默认链），只登记给 #24/#30。
  R4 **纪律**：`src/` 零改动；A0 断言量仍是 peak 与 **Vn**、5% 不放宽、δ=0.5 列不删、不换指标；
      消融读数（A1/A2 的任何数）**禁止**进入断言、档案现行值或吞吐分母；patch 必须 `try/finally` 复原，
      探针结束时**实测打印** `TE.CUT_CAPACITY_FLOOR == 0.5`（不成立＝污染了默认链，直接 FAILS）。

**与第 6 轮的分工**：第 6 轮把"读数口径"关单（L 侧≈全部、S 侧只在 δ=0、口径间 ≤4.37%）；本轮查的是
**场本身**为什么随 δ 动——若 α 成立，#25 的战场正式移交给 #24 的一条新子项；若 β 成立，移交 #30。
两条都**不是**"把 A0 弄绿"的路径，本轮不提出任何放宽。
"""
import contextlib
import itertools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
import jax                                              # noqa: E402

jax.config.update("jax_enable_x64", True)

from amforge import geometry as GEOM                     # noqa: E402
from amforge import thermal_enthalpy as TE               # noqa: E402
from amforge.core.contracts import (ProcessPlan, solid_mask,    # noqa: E402
                                    solid_weight)
from amforge.materials import get_material               # noqa: E402
from amforge.thermal_enthalpy import (solve_enthalpy_thermal,   # noqa: E402
                                      suggest_n_steps)

T0 = time.time()
EXT = (1.2e-3, 0.6e-3, 0.4e-3)
TIERS_UM = [50.0, 25.0, 12.5]
DELTAS = [0.0, 0.25, 0.5]
TL = float(get_material("316L").T_liquidus)
FLOOR_PROD = float(TE.CUT_CAPACITY_FLOOR)
ARMS = {"P": None, "A1": 1.0, "A2": 0.75}
EPS = 1e-9                     # fv 分带的判等容差（R0s/R2）
MAT_EPS = 1e-6                 # R0c 的"有材料"分界（依据见 docstring）
NOISE = 0.015                  # R3 的离散噪声容差（依据见 docstring）
bad = []
VOID = []
NOISE_FLOOR = []                   # 恒等档实测的 max|ΔT| ⇒ 本求解器的舍入噪声地板（R3 读数的下界）


def coupon(dx, delta):
    """与第 5/6 轮逐字相同的 A0 试片（R1 前提）。"""
    bx, by, bz = EXT
    bounds = [(-bx / 2 - delta, bx / 2 + delta),
              (-by / 2 - delta, by / 2 + delta),
              (-bz / 2 - delta, bz / 2 + delta)]

    def fn(x):
        return np.maximum(np.maximum(np.abs(x[..., 0]) - bx / 2,
                                     np.abs(x[..., 1]) - by / 2),
                          np.abs(x[..., 2]) - bz / 2)

    return GEOM.from_sdf_fn(lambda x: fn(np.asarray(x, dtype=np.float64)),
                            bounds=bounds, spacing=dx, name="coupon")


def cube(side, dx, delta):
    """R0b/R0c 的小夹具（与 #27 峰值探针同尺寸）：**只验机器，不产生物理结论**。"""
    b = side / 2.0 + delta
    bounds = [(-b, b), (-b, b), (-b, b)]

    def fn(x):
        return np.max(np.abs(x) - side / 2.0, axis=-1)

    return GEOM.from_sdf_fn(lambda x: fn(np.asarray(x, dtype=np.float64)),
                            bounds=bounds, spacing=dx, name="cube")


def plan():
    return ProcessPlan.uniform(1, modality="SLM", laser_power=600.0, scan_speed=0.8,
                               layer_thickness=EXT[2], hatch_spacing=1.4 * 100e-6,
                               beam_radius=100e-6, absorption=0.45, preheat_temp=400.0)


@contextlib.contextmanager
def floored(value):
    """monkeypatch CUT_CAPACITY_FLOOR；value=None ⇒ 不 patch（生产臂）。"""
    if value is None:
        yield
        return
    old = TE.CUT_CAPACITY_FLOOR
    TE.CUT_CAPACITY_FLOOR = value
    try:
        yield
    finally:
        TE.CUT_CAPACITY_FLOOR = old


def shift(F, off):
    """整数位移 off∈{−1,0,1}³，边缘 edge-clamp（与第 5/6 轮逐字相同）。"""
    n0, n1, n2 = F.shape
    P = np.pad(F, 1, mode="edge")
    return P[1 + off[0]:1 + off[0] + n0, 1 + off[1]:1 + off[1] + n1,
             1 + off[2]:1 + off[2] + n2]


def stencil_max(F):
    """3³ 邻域最大值：候选＝三重线性重建**可能**达到 level 的格。"""
    n0, n1, n2 = F.shape
    P = np.pad(F, 1, mode="edge")
    out = P[0:n0, 0:n1, 0:n2].copy()
    for a, b, c in itertools.product((-1, 0, 1), repeat=3):
        out = np.maximum(out, P[a + 1:a + 1 + n0, b + 1:b + 1 + n1, c + 1:c + 1 + n2])
    return out


def subcell_fractions(F, S, level, m):
    """三重线性重建＋子体素中心计数 ⇒ (φ_L, φ_SS, 候选格数)。与第 6 轮**逐字同算术序**（R1 前提）。

    φ_L＝亚格液相份额（只切液相面）；φ_SS＝亚格液相 ∩ 亚格实体（SDF 也重建，**不加中心掩膜**）。
    """
    n0, n1, n2 = F.shape
    cand = stencil_max(F) >= level
    I, J, K = np.nonzero(cand)
    phiL = np.zeros((n0, n1, n2), dtype=np.float64)
    phiSS = np.zeros((n0, n1, n2), dtype=np.float64)
    if I.size == 0:
        return phiL, phiSS, 0
    shF = {o: shift(F, o)[I, J, K] for o in itertools.product((-1, 0, 1), repeat=3)}
    shS = {o: shift(S, o)[I, J, K] for o in itertools.product((-1, 0, 1), repeat=3)}
    u = (np.arange(m) + 0.5) / m - 0.5
    d = np.sign(u).astype(int)
    w = np.abs(u)
    hitL = np.zeros(I.size, dtype=np.int64)
    hitSS = np.zeros(I.size, dtype=np.int64)
    for i, j, k in itertools.product(range(m), repeat=3):
        ox, oy, oz = d[i], d[j], d[k]
        valF = np.zeros(I.size, dtype=np.float64)
        valS = np.zeros(I.size, dtype=np.float64)
        for ai, aj, ak in itertools.product((0, 1), repeat=3):
            coef = ((w[i] if ai else 1.0 - w[i])
                    * (w[j] if aj else 1.0 - w[j])
                    * (w[k] if ak else 1.0 - w[k]))
            valF = valF + coef * shF[(ai * ox, aj * oy, ak * oz)]
            valS = valS + coef * shS[(ai * ox, aj * oy, ak * oz)]
        liq = valF > level
        hitL += liq
        hitSS += liq & (valS < 0.0)
    phiL[I, J, K] = hitL / float(m ** 3)
    phiSS[I, J, K] = hitSS / float(m ** 3)
    return phiL, phiSS, int(I.size)


BKEYS = ["=1", "[.75,1)", "==.5", "(0,.75)", "==0"]


def bands(w):
    """fv 分带（R0s/R2/R0c 共用）；判等一律用 EPS。"""
    e1 = w >= 1.0 - EPS
    b75 = (~e1) & (w >= 0.75 - EPS)
    e50 = (~e1) & (~b75) & (np.abs(w - 0.5) <= EPS)
    b50 = (~e1) & (~b75) & (~e50) & (w > EPS)
    z = ~(e1 | b75 | e50 | b50)
    return {"=1": e1, "[.75,1)": b75, "==.5": e50, "(0,.75)": b50, "==0": z}


def quantities(F, Tf, sdf, mk, w, dx, origin, pL8, pSS8, nif8):
    """一份场 → 第 6 轮同款观测量（算术序一致 ⇒ R1 可比）＋本轮新增的分带读数。"""
    vol = dx ** 3 * 1e9
    above = (F > TL) & mk
    mL8 = pL8 * mk
    sh = F.shape
    i0, j0, k0 = np.unravel_index(int(np.argmax(F * mk)), sh)
    edge_cells = int(above.sum() - (above[1:-1, 1:-1, 1:-1]).sum())
    bnd = 0
    for o in itertools.product((-1, 0, 1), repeat=3):
        if o != (0, 0, 0):
            bnd += int((above & ~shift(mk, o)).sum())
    gap_idx, margin = [], []
    for d in range(3):
        red = tuple(i for i in range(3) if i != d)
        si = np.nonzero(mk.any(axis=red))[0]
        mi = np.nonzero(above.any(axis=red))[0]
        gap_idx += [int(mi[0] - si[0]), int(si[-1] - mi[-1])]
        co = np.abs(np.asarray(origin, dtype=np.float64)[d] + np.arange(sh[d]) * dx)
        co = co[np.nonzero(above.any(axis=red))[0]]
        margin.append(float((EXT[d] / 2 - co.max()) * 1e6) if co.size else float("nan"))
    bd = bands(w)
    shell = mk & ~bd["=1"]
    # 界面格＝**严格部分**的亚格液相格（第 6 轮 `subcell_fractions` 内同款：0<φL<1，**未**加中心掩膜）；
    # 候选格＝三重线性重建可能触及 level 的格数（本轮新增读数，第 6 轮没印过 ⇒ 不参与 R1 对表）。
    nif = int(((pL8 > 1e-12) & (pL8 < 1.0 - 1e-12)).sum())
    out = dict(nml=int(above.sum()), peak=float(F[i0, j0, k0]),
               Vn=float(above.sum()) * vol, Vs=float(mL8.sum()) * vol,
               VsS=float(pSS8.sum()) * vol, nif=nif, ncand=nif8, edge=edge_cells, bnd=bnd,
               nvox=int(mk.sum()), Mdom=float(w.sum()) * vol, Mcen=float(mk.sum()) * vol,
               gap_idx=int(min(gap_idx)), margin=float(min(margin)),
               VsS_1=float(pSS8[bd["=1"]].sum()) * vol,
               VsS_sh=float(pSS8[bd["==.5"]].sum()) * vol,
               nml_1=int((above & bd["=1"]).sum()), nml_sh=int((above & bd["==.5"]).sum()),
               pkband=("=1" if bd["=1"][i0, j0, k0] else
                       "==.5" if bd["==.5"][i0, j0, k0] else
                       "[.75,1)" if bd["[.75,1)"][i0, j0, k0] else
                       "(0,.75)" if bd["(0,.75)"][i0, j0, k0] else "==0"),
               airTmax=float(Tf[~mk].max()) if (~mk).any() else float("nan"),
               TshP90=float(np.percentile(Tf[shell], 90)) if shell.any() else float("nan"),
               TcoreP90=float(np.percentile(Tf[mk & bd["=1"]], 90))
               if (mk & bd["=1"]).any() else float("nan"))
    return out


def solve_one(g_):
    """生产臂：不 patch（floor 保持 0.5）。断言的是"进求解前 floor 必须是生产值"。"""
    assert float(TE.CUT_CAPACITY_FLOOR) == FLOOR_PROD, "生产臂求解时 floor 非生产值 ⇒ patch 泄漏"
    return solve_enthalpy_thermal(geometry=g_, process=plan(), params={"material": "316L"})


def run_arm(g_, floor):
    with floored(floor):
        return solve_enthalpy_thermal(geometry=g_, process=plan(), params={"material": "316L"})


print(f"\n=== 探针自检：CUT_CAPACITY_FLOOR 生产值＝{FLOOR_PROD}；三臂＝{ARMS} ===", flush=True)
print("  cap_w 出处＝`thermal_enthalpy.py:616` 的 `cap_w = jnp.maximum(fv, CUT_CAPACITY_FLOOR)`"
      " ⇒ 模块全局、在**函数体内**读取 ⇒ monkeypatch 应当生效（R0b 去实测它，不靠推断）", flush=True)
print("  n_steps 出处＝`suggest_n_steps`／`solve_enthalpy_thermal` 内的 CFL 校核 ⇒ 只含 dx 与 cfl，"
      "**不含** floor（floor 只在 :678 的 cfl_eff 守卫里，越界才 raise、不改步数）⇒ 三臂同离散", flush=True)

# ============================================ R0s／R0c 先验：fv 直方图与 cap_w 改动集（纯几何，无求解）
print("\n=== R0s＋R0c 先验：生产夹具每格 fv 分带与 cap_w 改动集（未做任何求解）===", flush=True)
HIST = {}
for dx_um in TIERS_UM:
    for frac in DELTAS:
        dx = dx_um * 1e-6
        g_ = coupon(dx, frac * dx)
        sdf = np.asarray(g_.sdf, dtype=np.float64)
        mk = np.asarray(solid_mask(sdf), dtype=np.float64) > 0.5
        w = np.asarray(solid_weight(sdf, dx), dtype=np.float64)
        bd = bands(w)
        ch1 = np.maximum(w, 1.0) != np.maximum(w, FLOOR_PROD)
        ch2 = np.maximum(w, 0.75) != np.maximum(w, FLOOR_PROD)
        ch2_nomat = int((ch2 & (w <= EPS)).sum())
        ch2_mat = int((ch2 & (w > EPS)).sum())
        ch2_tiny = int((ch2 & (w > EPS) & (w <= MAT_EPS)).sum())
        h = dict(bd={k: int(v.sum()) for k, v in bd.items()},
                 bd_solid={k: int((v & mk).sum()) for k, v in bd.items()},
                 ch1=int(ch1.sum()), ch2=int(ch2.sum()), ch2_mat=ch2_mat,
                 ch2_nomat=ch2_nomat, ch2_tiny=ch2_tiny,
                 ch1_far=int((ch1 & (w > EPS) & (w < 1.0 - EPS)).sum()),
                 ulp1=int(((w < 1.0) & (w >= 1.0 - EPS)).sum()),
                 fv_min_solid=float(w[mk].min()), nvox=int(mk.sum()), shape=sdf.shape,
                 Mdom=float(w.sum()) * dx ** 3 * 1e9)
        HIST[(frac, dx_um)] = h
        print(f"  δ={frac:4.2f} dx={dx_um:5.1f} shape={sdf.shape} 格总数={w.size} "
              f"nvox={h['nvox']:7d} min(fv|实体)={h['fv_min_solid']:.6f} Mdom={h['Mdom']:.5f} |", flush=True)
        print("      fv 分带（全网格）：" + " ".join(f"{k}={h['bd'][k]}" for k in BKEYS), flush=True)
        print("      fv 分带（实体内）：" + " ".join(f"{k}={h['bd_solid'][k]}" for k in BKEYS), flush=True)
        print(f"      A1 判据分派：改动集={h['ch1']} 格，其中**有材料且被真切**（`EPS<fv<1−EPS`）"
              f"={h['ch1_far']} 格、`fv=1−1ulp`={h['ulp1']} 格、`fv≤EPS`（空气）="
              f"{h['ch1'] - h['ch1_far'] - h['ulp1']} 格"
              f" ⇒ R0b 走**{'有效档（须 max|ΔT|≥0.1 K）' if h['ch1_far'] > 0 else '恒等档（须 max|ΔT|<1e-9 K，同时给出噪声地板）'}**",
              flush=True)
        print(f"      cap_w 改动集：A1={h['ch1']} 格 ／ A2={h['ch2']} 格"
              f"（A2 内含材料 fv>EPS 的={h['ch2_mat']}，其中 EPS<fv≤1e-6 的={h['ch2_tiny']}，"
              f"无材料 fv≤EPS 的={h['ch2_nomat']}）⇒ A2 预测＝**"
              f"{'逐位不变' if h['ch2_mat'] == 0 else '逐位不等'}**", flush=True)

for (frac, dx_um), h in HIST.items():
    if frac == 0.0 and h["bd_solid"]["==.5"] == 0:
        bad.append(f"R0s δ=0 dx={dx_um}：实体内 fv≈0.5 的格数=0 ⇒ 第 6 轮的刀锋层机制叙述不成立")
    if frac > 0.0 and (h["bd_solid"]["==.5"] + h["bd_solid"]["(0,.75)"]) > 0:
        bad.append(f"R0s δ={frac} dx={dx_um}：实体内出现 fv<0.75 的格 "
                   f"{h['bd_solid']['==.5'] + h['bd_solid']['(0,.75)']} ⇒ 与第 6 轮 min(fv) 矛盾")
print(f"  R0s ⇒ {'PASS（δ=0 有 fv≈0.5 刀锋层；δ=0.25/0.5 实体内无 fv<0.75）' if not bad else 'FAIL'}",
      flush=True)

print("\n=== 设备清单（长期规则：自称 GPU 实测必须含本行，写在 log 本体）===", flush=True)
print(f"  jax.devices() = {jax.devices()}", flush=True)
print(f"  platform/backend = {jax.default_backend()} / x64={jax.config.jax_enable_x64}", flush=True)

# ============================================ R0a/R0b/R0c 机器验证（小夹具，秒级；数值不作物理结论）
print("\n=== R0a＋R0b＋R0c：消融机器在 0.4mm 立方 dx=80µm 上的验证（只验机器）===", flush=True)
for frac in (0.0, 0.25, 0.5):
    dx = 80e-6
    g_ = cube(0.4e-3, dx, frac * dx)
    sdf = np.asarray(g_.sdf, dtype=np.float64)
    w = np.asarray(solid_weight(sdf, dx), dtype=np.float64)
    ch1 = np.maximum(w, 1.0) != np.maximum(w, FLOOR_PROD)
    ch2 = np.maximum(w, 0.75) != np.maximum(w, FLOOR_PROD)
    ch1_mat = int((ch1 & (w > EPS)).sum())
    ch2_mat = int((ch2 & (w > EPS)).sum())
    ch2_tiny = int((ch2 & (w > EPS) & (w <= MAT_EPS)).sum())
    pred_same = (ch2_mat == 0)
    tP = solve_one(g_)
    tP2 = solve_one(g_)
    tA1 = run_arm(g_, 1.0)
    tA2 = run_arm(g_, 0.75)
    pk = np.asarray(tP.peak_temperature, dtype=np.float64)
    pk2 = np.asarray(tP2.peak_temperature, dtype=np.float64)
    fk = np.asarray(tP.final_temperature, dtype=np.float64)
    fk2 = np.asarray(tP2.final_temperature, dtype=np.float64)
    det = (pk.tobytes() == pk2.tobytes()) and (fk.tobytes() == fk2.tobytes())
    pA1 = np.asarray(tA1.peak_temperature, dtype=np.float64)
    fA1 = np.asarray(tA1.final_temperature, dtype=np.float64)
    pA2 = np.asarray(tA2.peak_temperature, dtype=np.float64)
    fA2 = np.asarray(tA2.final_temperature, dtype=np.float64)
    live_pk = pk.tobytes() != pA1.tobytes()
    live_fk = fk.tobytes() != fA1.tobytes()
    dmax1 = float(np.abs(fk - fA1).max())
    dpeak = float(abs(pk.max() - pA1.max()))
    same2 = (pk.tobytes() == pA2.tobytes()) and (fk.tobytes() == fA2.tobytes())
    dmax2 = float(np.abs(fk - fA2).max())
    ch1_far = int((ch1 & (w > EPS) & (w < 1.0 - EPS)).sum())
    ident = (ch1_far == 0)
    print(f"  δ={frac:4.2f}：R0a 确定性（连跑两次逐字节）＝"
          f"{'相同 PASS' if det else '**有差 ⇒ VOID 触发**'}｜n_steps={suggest_n_steps(g_, plan())} "
          f"peak={pk.max():.2f} K｜格数={w.size}（fv≤EPS={int((w <= EPS).sum())}、"
          f"EPS<fv<0.75={int(((w > EPS) & (w < 0.75 - EPS)).sum())}、0.75≤fv<1={ch1_mat - int(((w > EPS) & (w < 0.75 - EPS)).sum())}）",
          flush=True)
    print(f"      R0b 生效性（{'恒等档：改动集内有材料且被真切（EPS<fv<1−EPS）的格数=0 ⇒ 两侧都要过：逐位可不等但 max|ΔT| 必须 <1e-9 K' if ident else '有效档：EPS<fv<1−EPS 的格数=' + str(ch1_far) + ' ⇒ 两个数组都须逐位不等且 max|ΔT| ≥ 0.1 K'}）："
          f"cap_w 改动格数（fv<1）={int(ch1.sum())}／{w.size}（fv>EPS={ch1_mat}）｜"
          f"peak 数组逐位{'不等' if live_pk else '相同'}、final 场逐位{'不等' if live_fk else '相同'}"
          f"，全场 max|ΔT|={dmax1:.3e} K｜Δpeak(max)={dpeak:.3f} K ⇒ "
          f"{'PASS（恒等档＝#23 可证为算术恒等；顺带取噪声地板）' if (ident and dmax1 < 1e-9) else ('PASS' if (live_pk and live_fk and dmax1 >= 0.1) else '**FAIL ⇒ VOID**')}",
          flush=True)
    print(f"      R0b 附带预测（跑前登记，能失败）：δ=0.5 ⇒ Δpeak **恰为 0**（实体内 fv≡1 ⇒ cap_w 恒等），"
          f"δ=0/0.25 ⇒ Δpeak ≥ 0.1 K。实测 Δpeak={dpeak:.3f} K ⇒ "
          f"{'命中' if ((dpeak == 0.0) if frac == 0.5 else (dpeak >= 0.1)) else '**未命中 ⇒ 我对 #23 作用域的理解是错的**'}",
          flush=True)
    print(f"      R0c 作用域：A2 改动格数={int(ch2.sum())}（fv>EPS 的={ch2_mat}，其中 EPS<fv≤1e-6={ch2_tiny}）"
          f"⇒ 预测{'逐位不变' if pred_same else '逐位不等'}；实测{'两数组都逐位相同' if same2 else '至少一个不等'}"
          f"（全场 max|ΔT|={dmax2:.3e} K）⇒ {'一致 PASS' if same2 == pred_same else '**不一致 ⇒ 真发现**'}", flush=True)
    if not det:
        VOID.append(f"R0a δ={frac}：同臂两次求解不逐位相同 ⇒ 环境不确定")
    if ident:
        NOISE_FLOOR.append(dmax1)
        if dmax1 >= 1e-9:
            bad.append(f"R0b δ={frac}：恒等档（改动集内无 `fv<1−EPS` 的格）却测出 max|ΔT|={dmax1:.3e} K "
                       f"≥1e-9 K ⇒ floor 经 cap_w 之外的路径影响解 ⇒ 需重查")
    elif not (live_pk and live_fk and dmax1 >= 0.1):
        VOID.append(f"R0b δ={frac}：有效档（fv<1−EPS 的改动格={ch1_far}）但 floor→1.0 后"
                    f"（peak 逐位不等={live_pk}、final 逐位不等={live_fk}、全场 max|ΔT|={dmax1:.3e}K）"
                    f"⇒ patch 未抵达求解器")
    if not ((dpeak == 0.0) if frac == 0.5 else (dpeak >= 0.1)):
        bad.append(f"R0b 附带预测未命中 δ={frac}：Δpeak={dpeak:.3f} K ⇒ 需重查 #23 的作用域"
                   f"（是记录问题，不作 VOID：patch 生效性由 live/max|ΔT| 独立判定）")
    if same2 != pred_same:
        bad.append(f"R0c δ={frac}：预测（fv>EPS 的改动格={ch2_mat}）↔实测逐位性（same={same2}）矛盾"
                   f" ⇒ `thermal_enthalpy.py:840` 的「空气格分子恒为 0」论断不成立（或改动集判据错）")

if VOID:
    print("\nVOID（R0 触发 ⇒ 全轮作废，不进入生产梯）：", flush=True)
    for v in VOID:
        print("  - " + v, flush=True)
    sys.exit(2)
if "--prefix0" in sys.argv:
    print(f"\n[prefix0] 舍入噪声地板（恒等档 max|ΔT|，K）＝{['%.3e' % v for v in NOISE_FLOOR]}"
          f" ⇒ 低于此值的标量差**不得**解释为物理", flush=True)
    print(f"[prefix0] R0s/R0a/R0b/R0c：VOID 触发={len(VOID)}／真发现={len(bad)}"
          f"{'（照实列出）' if bad else '，全部通过'} ⇒ 前缀到此为止，生产段一行未执行。", flush=True)
    for b in bad:
        print("  - " + b, flush=True)
    sys.exit(0)

# ============================================ R1 生产梯（无 patch）＋消融梯
print("\n=== 生产梯（floor=0.5）＋消融梯（A1=1.0／A2=0.75）同一批解 ===", flush=True)
print(f"  T_liquidus={TL:.1f} K  几何体积={EXT[0] * EXT[1] * EXT[2] * 1e9:.6f} mm³"
      f"  求解前 floor={TE.CUT_CAPACITY_FLOOR}", flush=True)
R = {}
for dx_um in TIERS_UM:
    for frac in (DELTAS if dx_um != 12.5 else [0.0, 0.25]):     # 与第 5/6 轮同一采集集
        dx = dx_um * 1e-6                                       # 拼写＝第 6 轮（R1 前提）
        g_ = coupon(dx, frac * dx)
        sdf = np.asarray(g_.sdf, dtype=np.float64)
        mk = np.asarray(solid_mask(sdf), dtype=np.float64) > 0.5
        w = np.asarray(solid_weight(sdf, dx), dtype=np.float64)
        origin = np.asarray(g_.origin, dtype=np.float64)
        rec = {}
        FP = TFP = None
        for arm in ("P", "A1", "A2"):
            th = (solve_one(g_) if arm == "P" else run_arm(g_, ARMS[arm]))
            F = np.asarray(th.peak_temperature, dtype=np.float64)
            Tf = np.asarray(th.final_temperature, dtype=np.float64)
            pL8, pSS8, nif8 = subcell_fractions(F, sdf, TL, 8)
            m = quantities(F, Tf, sdf, mk, w, dx, origin, pL8, pSS8, nif8)
            if arm == "P":
                mP, FP, TFP = m, F, Tf
                m["floor"] = FLOOR_PROD
            else:
                m["floor"] = ARMS[arm]
                # 逐位性＝两个数组全部元素相同（R0c 断言的就是这个；标量代理会在"数组有差而 peak 恰等"
                # 的档上给出假结论 ⇒ δ=0.5 的 prefix 实测已经演示过一次）
                m["bit_same"] = bool(F.tobytes() == FP.tobytes() and Tf.tobytes() == TFP.tobytes())
                m["dmax_vs_P"] = float(np.abs(Tf - TFP).max())
                m["dpeak_vs_P"] = float(m["peak"] - mP["peak"])
                m["dVsS_vs_P_pct"] = float((m["VsS"] - mP["VsS"]) / mP["VsS"] * 100.0)
                m["sc_same"] = bool(abs(m["peak"] - mP["peak"]) < 1e-12
                                    and abs(m["VsS"] - mP["VsS"]) < 1e-15)
                if arm == "A1" and m["bit_same"] and HIST[(frac, dx_um)]["ch1_far"] > 0:
                    VOID.append(f"R0b 生产梯 δ={frac} dx={dx_um}：A1 与 P 逐位相同（有效档，"
                                f"fv<1−EPS 的改动格={HIST[(frac, dx_um)]['ch1_far']}）⇒ patch 在生产梯上是死的")
            rec[arm] = m
        if HIST[(frac, dx_um)]["ch1_far"] == 0:
            NOISE_FLOOR.append(rec["A1"]["dmax_vs_P"])
            if rec["A1"]["dmax_vs_P"] >= 1e-9:
                bad.append(f"R0b 生产梯 δ={frac} dx={dx_um}：恒等档（无 `fv<1−EPS` 改动格）却测出 "
                           f"max|ΔT|={rec['A1']['dmax_vs_P']:.3e} K ≥1e-9 ⇒ floor 走了 cap_w 之外的路径")
        pred_same = (HIST[(frac, dx_um)]["ch2_mat"] == 0)
        if rec["A2"]["bit_same"] != pred_same:
            bad.append(f"R0c 生产梯 δ={frac} dx={dx_um}：A2 预测（fv>EPS 的改动格="
                       f"{HIST[(frac, dx_um)]['ch2_mat']}⇒ 应{'逐位不变' if pred_same else '逐位不等'}）"
                       f"与实测逐位性 {rec['A2']['bit_same']} 矛盾 ⇒ `thermal_enthalpy.py:840` 的"
                       f"「空气格分子恒为 0」在生产夹具上不成立（真缺陷，照实登记）")
        R[(frac, dx_um)] = rec
        p, a1, a2 = rec["P"], rec["A1"], rec["A2"]
        print(f"  δ={frac:4.2f}·dx dx={dx_um:5.1f} nvox={p['nvox']:7d} 峰值格带={p['pkband']} "
              f"界面格={p['nif']} 候选格={p['ncand']} 触壳={p['edge']} 邻非实体={p['bnd']} "
              f"到面间隙={p['gap_idx']}格 余量={p['margin']:+.1f}µm |", flush=True)
        print(f"      P ＝peak{p['peak']:7.2f} Vn{p['Vn']:.5f} Vs{p['Vs']:.5f} VsS{p['VsS']:.5f} "
              f"Mdom={p['Mdom']:.5f} Mcen={p['Mcen']:.5f} 熔体素={p['nml']}", flush=True)
        print(f"      A1＝peak{a1['peak']:7.2f} Vn{a1['Vn']:.5f} VsS{a1['VsS']:.5f} "
              f"（Δpeak={a1['dpeak_vs_P']:+7.2f} K，ΔVsS={a1['dVsS_vs_P_pct']:+6.2f}%，"
              f"全场 max|ΔT|={a1['dmax_vs_P']:.3e} K，逐位{'相同' if a1['bit_same'] else '不等'}）｜"
              f"A2＝peak{a2['peak']:7.2f} Vn{a2['Vn']:.5f} VsS{a2['VsS']:.5f} "
              f"（Δpeak={a2['dpeak_vs_P']:+7.2f} K，ΔVsS={a2['dVsS_vs_P_pct']:+6.2f}%，"
              f"max|ΔT|={a2['dmax_vs_P']:.3e} K）⇒ A2 逐位与 P "
              f"{'相同' if a2['bit_same'] else '不等'}／预测"
              f"{'逐位不变' if pred_same else '逐位不等'}", flush=True)

print(f"  求解后 floor={TE.CUT_CAPACITY_FLOOR}（生产值＝{FLOOR_PROD}）⇒ "
      f"{'已复原' if float(TE.CUT_CAPACITY_FLOOR) == FLOOR_PROD else '**未复原**'}", flush=True)
if float(TE.CUT_CAPACITY_FLOOR) != FLOOR_PROD:
    bad.append(f"R4 patch 未复原：TE.CUT_CAPACITY_FLOOR={TE.CUT_CAPACITY_FLOOR}≠{FLOOR_PROD}")


def rel(x, y):
    return abs(x - y) / max(x, y)


# ============================================ R1 逐位复现第 6 轮
print("\n=== R1 逐位复现第 6 轮（其 log L43-62 印值：peak/Vn/Vs/VsS/Mdom/Mcen＋五个整数）===", flush=True)
ARC = {  # (δ, dx_um): (peak, Vn, Vs, VsS, Mdom, Mcen, nml, nif, edge, bnd, nvox)
    (0.0, 50.0): (2461.27, 0.06025, 0.08030, 0.07784, 0.29350, 0.36562, 482, 844, 36, 0, 2925),
    (0.25, 50.0): (2447.27, 0.06275, 0.08063, 0.08063, 0.29075, 0.28800, 502, 876, 0, 738, 2304),
    (0.50, 50.0): (2433.15, 0.06400, 0.07678, 0.07678, 0.28800, 0.28800, 512, 734, 0, 486, 2304),
    (0.0, 25.0): (2563.01, 0.06947, 0.08006, 0.07799, 0.28937, 0.32539, 4446, 3604, 241, 0, 20825),
    (0.25, 25.0): (2554.96, 0.07062, 0.07930, 0.07930, 0.28869, 0.28800, 4520, 3988, 0, 3006, 18432),
    (0.50, 25.0): (2565.18, 0.06473, 0.07303, 0.07303, 0.28800, 0.28800, 4143, 3936, 0, 2970, 18432),
    (0.0, 12.5): (2565.08, 0.07418, 0.07995, 0.07878, 0.28834, 0.30635, 37982, 14484, 1143, 0, 156849),
    (0.25, 12.5): (2555.81, 0.07453, 0.07948, 0.07948, 0.28817, 0.28800, 38157, 15668, 0, 10449, 147456),
}
nhit = 0
for key, a in ARC.items():
    r = R[key]["P"]
    d = [abs(r["peak"] - a[0]), abs(r["Vn"] - a[1]), abs(r["Vs"] - a[2]), abs(r["VsS"] - a[3]),
         abs(r["Mdom"] - a[4]), abs(r["Mcen"] - a[5])]
    ints_ok = (r["nml"] == a[6] and r["nif"] == a[7] and r["edge"] == a[8]
               and r["bnd"] == a[9] and r["nvox"] == a[10])
    ok = d[0] < 5e-3 and max(d[1:]) < 5.1e-6 and ints_ok
    nhit += ok
    print(f"  δ={key[0]:4.2f} dx={key[1]:5.1f}: Δpeak={d[0]:.3f} ΔVn={d[1]:.6f} ΔVs={d[2]:.6f} "
          f"ΔVsS={d[3]:.6f} ΔMdom={d[4]:.6f} ΔMcen={d[5]:.6f} 整数={'全对' if ints_ok else '有差'} ⇒ "
          f"{'一致' if ok else '不一致'}", flush=True)
    if not ok:
        bad.append(f"R1 δ={key[0]} dx={key[1]}：Δ={[round(x, 6) for x in d]} 整数ok={ints_ok}"
                   f" ⇒ patch 机器动了默认路径")
print(f"  ⇒ {nhit}/{len(ARC)} 格复现", flush=True)
if nhit != len(ARC):
    VOID.append(f"R1 只复现 {nhit}/{len(ARC)} ⇒ 本轮与第 6 轮不是同一副生产路径，记分不作数")

# ============================================ R2 场读数（不判）
print("\n=== R2 场读数：fv 分带的熔体积／峰值格归属／空气格最高温（只摆机制，不设门槛）===", flush=True)
for dx_um in TIERS_UM:
    for frac in (0.0, 0.25):
        p, h = R[(frac, dx_um)]["P"], HIST[(frac, dx_um)]
        print(f"  δ={frac:4.2f} dx={dx_um:5.1f}: VsS＝内层(fv=1){p['VsS_1']:.5f}＋刀锋层(fv≈0.5)"
              f"{p['VsS_sh']:.5f} mm³（刀锋占比 {p['VsS_sh'] / p['VsS'] * 100:5.1f}%）| 熔体素 内层="
              f"{p['nml_1']} 刀锋={p['nml_sh']}/{p['nml']} | 峰值格带={p['pkband']}", flush=True)
        print(f"      实体内 fv≈0.5 格数={h['bd_solid']['==.5']}  实体表面层（fv<1）格数="
              f"{h['bd_solid']['[.75,1)'] + h['bd_solid']['==.5'] + h['bd_solid']['(0,.75)']}  "
              f"final T 的 P90：内层={p['TcoreP90']:.1f} 表面层={p['TshP90']:.1f} K  "
              f"空气格最高 T={p['airTmax']:.1f} K（预热=400.0 ⇒ 高出即为空气格被加热的证据）", flush=True)
for dx_um in TIERS_UM:
    p, q = R[(0.0, dx_um)]["P"], R[(0.25, dx_um)]["P"]
    a0, a25 = R[(0.0, dx_um)]["A1"], R[(0.25, dx_um)]["A1"]
    print(f"  跨 δ 读数差 dx={dx_um:5.1f}: ΔVsS(P)＝{(p['VsS'] - q['VsS']) * 1e3:+8.4f}e-3 mm³；"
          f"ΔVsS(A1)＝{(a0['VsS'] - a25['VsS']) * 1e3:+8.4f}e-3；"
          f"Δpeak(P)={p['peak'] - q['peak']:+7.2f} K，Δpeak(A1)={a0['peak'] - a25['peak']:+7.2f} K", flush=True)

# ============================================ R3 归因记分＋裁决（跑前写死的 α/β/γ）
print("\n=== R3 归因记分：跨 δ（0↔0.25）同 dx 的形态差在 A1（cap_w≡1）下还剩多少 ===", flush=True)
print(f"  实测舍入噪声地板（恒等档 max|ΔT|，K）＝{['%.3e' % v for v in NOISE_FLOOR]}；"
      f"R3 用的 1.5% 离散容差与它是两回事（前者是求解器舍入、后者是 m/口径离散）", flush=True)
rows = []
for dx_um in TIERS_UM:
    for X in ("VsS", "Vn"):
        p0, p25 = R[(0.0, dx_um)]["P"][X], R[(0.25, dx_um)]["P"][X]
        a0, a25 = R[(0.0, dx_um)]["A1"][X], R[(0.25, dx_um)]["A1"][X]
        b0, b25 = R[(0.0, dx_um)]["A2"][X], R[(0.25, dx_um)]["A2"][X]
        GP, GA1, GA2 = rel(p0, p25), rel(a0, a25), rel(b0, b25)
        D = rel(R[(0.0, dx_um)]["P"]["Mdom"], R[(0.25, dx_um)]["P"]["Mdom"])
        bar = max(2.0 * D, NOISE)
        rows.append(dict(dx=dx_um, X=X, GP=GP, GA1=GA1, GA2=GA2, D=D, bar=bar))
        print(f"  {X:4s} dx={dx_um:5.1f}: G_P={GP * 100:6.3f}% G_A1={GA1 * 100:6.3f}% "
              f"G_A2={GA2 * 100:6.3f}% ‖ 域质量差 D={D * 100:5.3f}%（与臂无关）‖ α 门槛 "
              f"max(2D,1.5%)={bar * 100:5.2f}% ⇒ share_capw={(GP - GA1) / GP * 100:+6.1f}% "
              f"share_floor={(GP - GA2) / GP * 100:+6.1f}%", flush=True)


def branch(X):
    rs = [r for r in rows if r["X"] == X]
    if all(r["GA1"] <= r["bar"] for r in rs):
        return "α"
    if all(r["GA1"] >= 0.8 * r["GP"] for r in rs):
        return "β"
    return "γ"


bV, bN = branch("VsS"), branch("Vn")
final = bV if bV == bN else "γ"
print("\n=== R3 裁决（跑前写死的 α→β→γ；主判据＝VsS）===", flush=True)
print(f"  分支(VsS)＝{bV} 分支(Vn)＝{bN} ⇒ 判 **{final}**"
      f"{'' if bV == bN else '（两量分歧 ⇒ 按跑前规则判 γ，不许挑对自己有利的量）'}", flush=True)
if final == "α":
    print("  ⇒ α：关掉 cap_w 通道后跨 δ 形态差塌进 max(2·D, 1.5%) 内 ⇒ 超出体积的那部分**主要是刀锋层热容**"
          " ⇒ #24 需加一条\"刀锋层热容与对齐 δ 的耦合\"。", flush=True)
elif final == "β":
    print("  ⇒ β：A1 几乎不动跨 δ 差（三档全部 ≥0.8·G_P）⇒ 超出体积的那部分与 cap_w 无关"
          " ⇒ 归 #30（夹具触边）与 #24 的体积/吸附项处置。", flush=True)
else:
    print("  ⇒ γ：两通道同居（或 VsS/Vn 分歧）⇒ 两条都登记、以报数为主。", flush=True)
print("  ⚠ 解释边界（跑前写的）：A1＝**#23 之前的口径**，只作\"通道是否存在\"的消融开关；"
      "**α 成立绝不等于 A1 更对**（#23 的验收＝开发日志 §26.18 的预测逐位命中，未被本轮替代）。", flush=True)

print("\n=== 记分 ===", flush=True)
print("FAILS: " + ("无" if not bad and not VOID else ""), flush=True)
for b in bad:
    print("  - " + b, flush=True)
for v in VOID:
    print("  - VOID: " + v, flush=True)
print("注：本轮不改 src/、不改 A0 断言量、不放宽 5%、不删 δ=0.5 列、不调任何系数（N4 未触碰）；"
      "A1/A2 的一切读数只作消融诊断，**禁止**进入断言或档案现行值。", flush=True)
print(f"elapsed={time.time() - T0:.0f}s", flush=True)
