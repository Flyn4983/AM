# -*- coding: utf-8 -*-
"""T4（#22）光斑口径普查 ＋ **二阶矩实测**：改前跑一次、改后跑一次。

契约（`src/amforge/core/contracts.py:225-226` 的 `ProcessPlan.beam_radius`）写的是
「光斑 **1/e²** 半径」，但全树把同一个数解成两种半径：c2＝`exp(-2ρ²/r²)`（σ=r/2）与
c1＝`exp(-ρ²/r²)`（σ=r/√2）⇒ 同一个输入数在两个求解器里差 **√2**（面积差 **2×**）。
本轮的单一入口语义＝**c2**。三条外证：① 契约原文；② vendored 上游 `_refs/jaxcode` 的
全部热源行都是 c2；③ A3/NIST 基准用 `D4σ` 二阶矩描述光束，`w_{1/e²}=2σ=D4σ/2`。
⇒ 只把 **amforge 的两条默认数值求解器**（`thermal.enthalpy`、`meltpool.fdm`）改到契约口径；
diffmech 与 amforge 里已经是 c2 的位点**零改动**（动了就是换语义）。

本轮范围（刻意的窄，**只动面内**）：轴向/深度尺度（`dp = 1.2·r`）、分辨率闸门常量
（`0.5·dx`、`dx>2r`、`max(rb,dx)`、`1.5·rb`、`dx≈r/6~/10`）、经验系数（`1.2`/`1.8`/`2.0`）、
两处离散标定旋钮（`heat_scale=1.4`、`clip(0.8/v,·)`）与 A3 的 `α=0.54` **一律不改**——
它们是在旧口径下标定出来的，同批改会把「口径修复」与「重标定」混成一件事（**违 N4**）
⇒ K6 只读数，靶表移交 **#33**。可观测量（A0 的峰值/熔体积、#27 的打印值）会因面内变紧而
**合法移动**，但**阈值与指标名一字不动**，由 K7 拿 `git show HEAD:` 的 assert 行原文逐字证明。

跑法（毫秒级：只对热源函数求值＋网格积分，**不解热方程**；CPU 钉住）：

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=src JAX_ENABLE_X64=1 /tmp/amvenv/bin/python \\
        docs/evidence/2026-10-08/am_t4_spot_probe.py pre
    … am_t4_spot_probe.py post

门槛（**跑前写死**，按阶段分硬门，不过 ⇒ 非零退出）：

  K0-S 估计器自检（两阶段）：把 8 副**已知拼写**（含**修复后的正确拼写**
      `exp(-planar2/(2*sig*sig))`）喂给普查正则，必须**全部命中** ⇒ 证明这把尺能看见
      正确实现，不是只认旧写法（记忆教训：门槛必须可被正确实现达到）。
  K0  普查完备（**两阶段两种键**）：`src/**/*.py` 的每个「口径敏感行」必须命中登记表。
      pre＝**(文件,行号)** 双向对账：未登记 ⇒ VOID，登记了却普查不到 ⇒ 也 VOID。
      post＝**文本键**的前向门（未登记 ⇒ VOID）＋「行号漂移」读数：本轮要在两个文件
      顶部各加一行 import ⇒ 同文件全部登记行整体平移，行号绑定在 post **必然**失效
      （第一版 post 就红在这里——红在门上不在树上）。改成 `planar_decay(...)` 的行
      不再含 exp()/π ⇒ 天然退出命中集，故后向门在 post 降级为读数，其实质保证由
      K0b＋K4 三条 post 硬门承担。post 里 `post_marker` 只能覆盖 IN／PR 桶的行
      （CONFORMS 位点"以新写法出现"＝它被改了 ⇒ 正是本门要抓的事）。
      跳过 `def`/`class`/`@`/`#` 行与 `beam_radius=` 透传行（#24 的 K0-1 教训：修法是
      **不认定义行**，不是往登记表塞假位点）。
  K0b 文本绑定（两阶段，**双向**）：pre＝每条 marker 必须**正好在登记行**（同时兜住
      "登记表抄错行号"）。post 按桶分治：带 `post_marker` 的行（IN／PR＝本轮要改的）
      必须「**旧写法整文件消失** ∧ 新写法存在」——只改一半 ⇒ 旧串还在 ⇒ 红；
      不带的行（CONFORMS／heuristics／非光束）必须**文本仍在**（防"删掉位点来通过
      普查"）。U3：替换守护必须在两副口径同时成立。
  K1  分裂的**正对照**（只 pre 判）：逐位点用**二阶矩**实测 σ（`σ=√(⟨ρ²⟩/2)`，与拼写无关，
      量的不是"怎么写的"而是"实际多宽"）⇒ 断言「≥1 个 amforge IN 位点的 σ/(R/2) 与 √2
      之差 <1e-3，且 ≥1 个 CONFORMS 位点为 1.0」。这条在已修好的树上必然失败 ⇒ post 不判。
  K2  **内部一致性＝抓半修**（两阶段）：衰减的实测面内积分 `A_meas=Σexp(…)dA` 必须等于
      **同一文件里实际写下的归一化因子**（`π r²` 还是 `π r²/2`，从源码文本读，不由我复述）
      ⇒ 只改衰减不改归一化（或反过来）必红。另加可直接测的功率守恒：`_laser_source_fdm`、
      `gaussian_heat_source`、`moving_gaussian_source` 的 `ΣQ·dV/(ηP)` ∈ 1±5e-3。
  K3  改后统一（只 post 判）：全部 amforge IN 位点实测 σ == R/2（**统一** tol 2e-3：
      erf 档的"单元捕获份额"解析精确，但二阶矩仍按单元中心读数 ⇒ 同样吃 O(dx²)
      离散偏差，给它 1e-6 的单独严杠会红在正确实现上）；且 `exp(-2ρ²/R²)`、
      `exp(-ρ²/(2σ²))`（σ=R/2）与 `beam.planar_decay` **三副拼写逐位相同**
      （U3：替换守护必须在两副口径同时成立）。
      前置：同一位点 n=401/801 两次实测 σ 的相对差 ≥1e-3 ⇒ `未收敛` 判 FAIL
      （防"把离散偏差读成物理差"；1e-3 ＝ K3 容差的一半，比 √2＝41% 的分裂小两个
      数量级 ⇒ 既有牙又可被正确实现达到）。
  K4  单点出处（只 post 判）：amforge 的 in-scope 文件必须 import `amforge.beam`；残留
      c1 衰减（`exp(-planar2/…r*r)` 缺 2.0 系数）必须为 0；diffmech 保持内联系数
      `-2.0 * rsq / r2` 不变（依赖方向 `amforge → diffmech` ⇒ 不能反向 import）；
      且**跨包同宽**：diffmech 实测 σ 与 amforge IN 实测 σ 必须同为 R/2。
  K5  **绝对期望**（两阶段）：`D4σ=85 µm` ⇒ σ=D4/4=21.25 µm、c2 半径=D4/2=42.5 µm、
      c1 半径=√2σ。分两件事说清（第一版把这两个混了 ⇒ 自己的门槛自己过不了）：
      ① **同一个输入数**：σ_c1/σ_c2=√2、面积比恰 2×（这就是"分裂"的大小）；
      ② **描述同一个 σ**：输入数之比 R_c1/R_c2=1/√2，且两输入换算回 σ 必须逐位相等。
      ⇒ ±1ulp 换写的**稳定性**探针看不见**确定性偏差**（#32 教训），故这里给绝对数。
  K6  出域不变量**读数**（不判）：σ_z/σ_xy 长宽比、`1.2`、`0.5·dx`、`dx>2r`、`max(rb,dx)`、
      `1.5·rb`、`1.8·rb`、`heat_scale=1.4`、`clip(0.8/v)`、`α=0.54` 的在/缺 ＝ #33 靶表。
  K7  断言不动（两阶段）：两条已登记红灯的 **assert 行原文**（`git show HEAD:` vs 工作树）
      逐字相同 ⇒「A0 断言不许放宽／不许换指标」由文本证明，不由嘴证明。
"""
from __future__ import annotations

import hashlib
import math
import re
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

REPO = Path(__file__).resolve().parents[3]
PHASE = (sys.argv[1] if len(sys.argv) > 1 else "post").strip().lower()
assert PHASE in ("pre", "post"), f"phase 只能是 pre/post，收到 {PHASE!r}"

# ---------------------------------------------------------------------------
# 外部锚：A3 / NIST MDS2-3662 §2.2 的 D4σ 与 ISO 11146 的二阶矩换算
# ---------------------------------------------------------------------------
D4 = 85e-6
SIGMA_A3 = D4 / 4.0
R_C2 = 2.0 * SIGMA_A3                    # 42.5 µm —— 本轮选定的入口口径
R_C1 = math.sqrt(2.0) * SIGMA_A3         # 30.051… µm —— 旧 amforge 默认路径实际用的半径
BEAM_R = R_C2                            # 所有位点喂**同一个数**，让差值只可能来自口径
HALF = 8.0 * R_C1                        # 覆盖最宽的 c1 情形

# ---------------------------------------------------------------------------
# 普查估计器：口径敏感行 ＝ 出现光斑样符号 ∧ 进入指数/面积/体尺度
# ---------------------------------------------------------------------------
SPOT = re.compile(r"\b(beam_radius|r_b|rb|pi_r2|pi_R2|s_in|r2|dist2|planar2|rsq|r_src)\b")
MATH = re.compile(r"(exp\(|\*\* *2\b|\*\* *3\b|pi_r2|pi_R2|\* r2|/ r2|r \* r|rb \* rb"
                  r"|sqrt\(|r_src \* r_src|jnp\.pi \*|math\.pi \*|jnp\.pi \*\* 1\.5)")
PASSTH = re.compile(r"^\s*(beam_radius|r_b)\s*=")


def is_census_line(line: str) -> bool:
    s = line.strip()
    if not s or s.startswith(("def ", "async def ", "class ", "@", "#")):
        return False
    if PASSTH.match(line):
        return False
    if '"""' in line:
        return False
    return bool(SPOT.search(line)) and bool(MATH.search(line))


def census_hits() -> list[tuple[str, int, str]]:
    out = []
    for p in sorted((REPO / "src").rglob("*.py")):
        rel = str(p.relative_to(REPO))
        for i, line in enumerate(_lines(rel), 1):
            if is_census_line(line):
                out.append((rel, i, line.rstrip()))
    return out


# K0-S：估计器必须也能看见**修复后的正确拼写**（否则"全树无 c1"会被读成"尺子坏了"）
SELF_CHECK_SPELLINGS = [
    ("c1 衰减（旧默认）", "planar = jnp.exp(-planar2 / jnp.maximum(rb * rb, 1e-18))"),
    ("c1 面内积分 π r²", "norm = (jnp.pi * rb * rb) * jnp.maximum(absorption_depth, dx) + 1e-18"),
    ("c1 σ=r/√2", "s_in = jnp.maximum(r, 1e-12) / math.sqrt(2.0)"),
    ("c2 衰减＋2P/(π r²)", "q2d = (2.0 * absorbed) / pi_r2 * jnp.exp(-2.0 * rsq / r2)"),
    ("c2 半径平方来源", "r2 = cfg.beam_radius ** 2"),
    ("Eagar-Tsai 指数", "expo = -(2.0 / beam_radius ** 2) * ("),
    ("c1 point 档 Q0（**只有 r_src 词汇**的行）",
     "Q0 = eta * P / (math.pi * r_src * r_src * math.sqrt(2.0 * math.pi) * dp + 1e-18)"),
    ("**修复后的正确拼写**（beam.planar_decay）",
     "return jnp.exp(-planar2 / (2.0 * spot_sigma(beam_radius) ** 2))"),
]

# ---------------------------------------------------------------------------
# 登记表（跑前从现树逐行抄；marker＝**改前**原文片段；bucket 见 docstring）
# ---------------------------------------------------------------------------
IN_ = "IN（本轮要改到契约口径）"
CF_ = "CONFORMS（已是 c2，不许动）"
HU_ = "OUT-HEURISTIC（经验长度尺度 ⇒ #33）"
NB_ = "OUT-NOT-BEAM（同拼写但不是光束）"
PR_ = "OUT-PROSE（文档/注释行）"

REG = [
    # ---- amforge：c1 ⇒ 本轮改（IN：post 必须「旧写法整文件消失 ∧ 新写法落在登记行」）----
    dict(id="A01", f="src/amforge/thermal_enthalpy.py", ln=294, b=IN_, k="moment3d",
         marker="decay = jnp.exp(-planar2 / jnp.maximum(r * r, 1e-18)) * \\",
         post_marker="planar_decay(planar2, r)",
         note="_moving_source 的 3D 面内项（source_model=\"point\" 分支）"),
    dict(id="A02", f="src/amforge/thermal_enthalpy.py", ln=297, b=IN_, k="moment2d",
         marker="decay = jnp.exp(-planar2 / jnp.maximum(r * r, 1e-18))",
         post_marker="planar_decay(planar2, r)",
         note="_moving_source 的 2D 分支"),
    dict(id="A04", f="src/amforge/thermal_enthalpy.py", ln=652, b=IN_, k="normtext",
         marker="Q0 = eta * P / (math.pi * r_src * r_src * math.sqrt(2.0 * math.pi) * dp + 1e-18)",
         post_marker="inplane_integral(r_src)",
         note="point 档面内积分因子（含两处离散旋钮 ⇒ 不许用守恒判，只用 K2 文本对账）"),
    dict(id="A05", f="src/amforge/meltpool.py", ln=1547, b=IN_, k="momentfdm",
         marker="planar = jnp.exp(-planar2 / jnp.maximum(rb * rb, 1e-18))",
         post_marker="planar_decay(planar2, rb)",
         note="默认 meltpool.fdm 面内项（**默认熔池求解器**吃的就是这个数）"),
    dict(id="A06", f="src/amforge/meltpool.py", ln=1549, b=IN_, k="normtext",
         marker="norm = (jnp.pi * rb * rb) * jnp.maximum(absorption_depth, dx) + 1e-18",
         post_marker="inplane_integral(rb)",
         note="与 A05 同源的面内积分 π rb² ⇒ 必须同批改，否则半修"),
    # ---- amforge：同一模块里的**文字**口径（行为对/文字错，或文字随行为一起改）----
    dict(id="A07", f="src/amforge/thermal_enthalpy.py", ln=284, b=PR_, k="prose",
         marker="Q(x) = Q0·exp(−ρ²_xy/r²)·exp(−z²/(2 dp²))",
         post_marker="exp(−ρ²_xy/(2σ²))",
         note="_moving_source docstring：写的是 c1 形状，随行为一起改"),
    dict(id="A08", f="src/amforge/thermal_enthalpy.py", ln=287, b=PR_, k="prose",
         marker="∫Q dV = Q0·π r²·dp√(2π) = ηP",
         post_marker="π r²/2",
         note="面内积分声明 π r²（c1 自洽）⇒ 改后必须是 π r²/2"),
    dict(id="A09", f="src/amforge/thermal_enthalpy.py", ln=305, b=PR_, k="prose",
         marker="（等效 σ=r/√2 的高斯）",
         post_marker="σ=r/2",
         note="**文字与行为差 √2**：erf 行写 `erf(Δ/s_in)` ⇒ 实际 σ=s_in/√2=r/2，"
              "而文字宣称 σ=r/√2。实测（A03）站在我这边 ⇒ 改文字不改行为"),
    dict(id="A10", f="src/amforge/meltpool.py", ln=1537, b=PR_, k="prose",
         marker="exp(−r²/rb²) · exp(−z_d/δ) / (π·rb²·δ)",
         post_marker="2π·σ²·δ",
         note="_laser_source_fdm docstring 的形状＋归一化声明"),
    # ---- amforge：已是 c2／与口径无关（**不许动**）----
    dict(id="A03", f="src/amforge/thermal_enthalpy.py", ln=315, b=CF_, k="momenterf",
         marker="s_in = jnp.maximum(r, 1e-12) / math.sqrt(2.0)",
         note="缺省 integrated 档；实测 σ/(R/2)=1.0001 ⇒ **已合契约**（改它就是换语义）"),
    dict(id="B01", f="src/amforge/meltpool.py", ln=751, b=CF_, k="text",
         marker="r2 = ((xs[:, None] - xL) ** 2 + (ys[None, :] - yL) ** 2)",
         note="VOF 面内 ρ² 定义"),
    dict(id="B02", f="src/amforge/meltpool.py", ln=752, b=CF_, k="text",
         marker="I = (2.0 * P / (jnp.pi * r_b ** 2)) * jnp.exp(-2.0 * r2 / r_b ** 2)",
         note="VOF 表面强度 c2 ＋ 2P/(π r_b²)"),
    dict(id="B03", f="src/amforge/meltpool.py", ln=878, b=CF_, k="text",
         marker="/ process.beam_radius ** 2)",
         note="送粉质量形状 c2（自归一 ⇒ 无前系数）"),
    dict(id="B04", f="src/amforge/meltpool.py", ln=1322, b=CF_, k="text",
         marker="c2 = beam_radius ** 2 / (8.0 * alpha)",
         note="Eagar-Tsai 时标 c²=r_b²/(8α)（解析代理 ⇒ 另属铁律议题）"),
    dict(id="B05", f="src/amforge/meltpool.py", ln=1332, b=CF_, k="text",
         marker="expo = -(2.0 / beam_radius ** 2) * (", note="Eagar-Tsai 指数 c2"),
    dict(id="B06", f="src/amforge/meltpool.py", ln=1340, b=CF_, k="text",
         marker="/ (rho_cp * jnp.pi ** 1.5 * alpha * beam_radius))",
         note="Eagar-Tsai 前系数 ∝1/r_b（σ²=r_b²/4 已写进文档）"),
    dict(id="B07", f="src/amforge/thermal_enthalpy.py", ln=291, b=CF_, k="text",
         marker="planar2 = jnp.sum(planar ** 2, axis=-1)",
         note="ρ² 定义行，与口径无关（普查新见：宽词汇后才现身）"),
    dict(id="B08", f="src/amforge/meltpool.py", ln=1545, b=CF_, k="text",
         marker="planar2 = (coords[..., 0] - x_t) ** 2 + coords[..., 1] ** 2",
         note="fdm 的 ρ² 定义行，同上"),
    # ---- amforge：经验长度尺度（出域 ⇒ #33） ----
    dict(id="C01", f="src/amforge/core/contracts.py", ln=275, b=HU_, k="text",
         marker="* self.beam_radius ** 3)", note="King ΔH/h_s 判据 r³"),
    dict(id="C02", f="src/amforge/process.py", ln=216, b=HU_, k="text",
         marker="denom = jnp.sqrt(jnp.pi * alpha * scan_speed * beam_radius ** 3)",
         note="同上判据（阈值 6/25/30 是在旧口径下标的）"),
    dict(id="C03", f="src/amforge/inverse.py", ln=834, b=HU_, k="text",
         marker="w = 2.0 * plan.beam_radius * jnp.sqrt(1.0 + jnp.maximum(dH, 0.0) / 8.0)",
         note="熔宽经验式 2r√(1+ΔH/8)"),
    # ---- diffmech：上游即 c2，且被 tests_diffmech 钉住 ⇒ 零改动 ----
    dict(id="D01", f="src/diffmech/methods/am/am_thermal.py", ln=126, b=CF_, k="text",
         marker="r2 = cfg.beam_radius ** 2", note="c2 半径平方"),
    dict(id="D02", f="src/diffmech/methods/am/am_thermal.py", ln=127, b=CF_, k="text",
         marker="pi_r2 = float(np.pi) * r2", note="面内积分 π r_b²（配 2P ⇒ π r_b²/2 有效）"),
    dict(id="D03", f="src/diffmech/methods/am/am_thermal.py", ln=171, b=CF_, k="text",
         marker="q = (2.0 * absorbed) / pi_r2 * jnp.exp(-2.0 * rsq / r2)",
         note="闭包内，需 MultiLayerPath ⇒ 不独立实测"),
    dict(id="D04", f="src/diffmech/methods/am/particle_am.py", ln=665, b=CF_, k="text",
         marker="r2 = cfg.beam_radius ** 2", note="c2 半径平方"),
    dict(id="D05", f="src/diffmech/methods/am/particle_am.py", ln=666, b=CF_, k="text",
         marker="pi_r2 = jnp.pi * r2", note="面内积分 π r_b²"),
    dict(id="D06", f="src/diffmech/methods/am/particle_am.py", ln=679, b=CF_, k="text",
         marker="q = (2.0 * absorbed) / pi_r2 * jnp.exp(-2.0 * rsq / r2)", note="闭包内 ⇒ 不独立实测"),
    dict(id="D07", f="src/diffmech/methods/am/particle_am.py", ln=1188, b=CF_, k="text",
         marker="r2 = cfg.beam_radius ** 2", note="LSF 送粉形状"),
    dict(id="D08", f="src/diffmech/methods/am/particle_am.py", ln=1192, b=CF_, k="text",
         marker="pi_R2 = jnp.pi * r2", note="概率密度归一 1/(π R²)（∫exp(-2r²/R²)=πR²/2 ⇒ 与注释差 2，**登记不修**：属分布形状、不在本轮面内口径窄范围）"),
    dict(id="D09", f="src/diffmech/methods/am/particle_am.py", ln=1203, b=HU_, k="text",
         marker="ring_R2 = (1.8 * cfg.beam_radius) ** 2", note="集粉宽度经验系数 1.8 ⇒ #33"),
    dict(id="D10", f="src/diffmech/methods/am/particle_am.py", ln=1204, b=CF_, k="text",
         marker="beam_intensity = jnp.exp(-2.0 * rsq / r2)", note="需路径 ⇒ 不独立实测"),
    dict(id="D15", f="src/diffmech/methods/am/particle_am.py", ln=1205, b=HU_, k="text",
         marker="catchment = (1.0 / (jnp.pi * ring_R2)) * jnp.exp(-2.0 * rsq / ring_R2)",
         note="集粉概率密度：1/(πR²) 配 ∫exp(−2ρ²/R²)=πR²/2 ⇒ **整因子差 2**（与 D08 同族），"
              "出域 ⇒ 新立条目（形状/归一，非本轮面内口径）"),
    dict(id="D11", f="src/diffmech/methods/am/powder_bed.py", ln=675, b=HU_, k="text",
         marker="d_melt = 1.2 * rb * jnp.sqrt(jnp.maximum(ved_ / ved_ref, 1e-12))",
         note="熔深经验式 1.2 rb ⇒ #33"),
    dict(id="D12", f="src/diffmech/methods/fvm/thermal.py", ln=404, b=CF_, k="text",
         marker="pi_r2 = math.pi * r2", note="面内积分 π r_b²"),
    dict(id="D13", f="src/diffmech/methods/fvm/thermal.py", ln=409, b=CF_, k="momentdm",
         marker="q2d = (2.0 * absorbed) / pi_r2 * jnp.exp(-2.0 * rsq / r2)",
         note="gaussian_heat_source，可独立调用 ⇒ 实测"),
    dict(id="D14", f="src/diffmech/methods/fvm/thermal.py", ln=443, b=CF_, k="momentdm",
         marker="q2d = (2.0 * absorbed) / (math.pi * r2) * jnp.exp(-2.0 * rsq / r2)",
         note="moving_gaussian_source ⇒ 实测"),
    # ---- 出域：同拼写但不是光束 ----
    dict(id="E01", f="src/diffmech/preprocess/mesh_primitives.py", ln=160, b=NB_, k="text",
         marker="r2 = (cc[..., 0] - cx) ** 2 + (cc[..., 1] - cy) ** 2", note="形状原语的 r²，不吃光斑"),
    dict(id="E02", f="src/diffmech/preprocess/mesh_primitives.py", ln=178, b=NB_, k="text",
         marker="r2 = (cc[..., 0] - cx) ** 2 + (cc[..., 1] - cy) ** 2", note="同上"),
    dict(id="E03", f="src/diffmech/preprocess/mesh_primitives.py", ln=179, b=NB_, k="text",
         marker="mask = (r2 >= r_inner * r_inner) & (r2 <= r_outer * r_outer)", note="环形原语"),
    dict(id="E04", f="src/diffmech/preprocess/mesh_primitives.py", ln=203, b=NB_, k="text",
         marker="r2 = (cc[..., 0] - hx) ** 2 + (cc[..., 1] - hy) ** 2", note="六边形原语"),
    dict(id="E05", f="src/diffmech/preprocess/mesh_primitives.py", ln=250, b=NB_, k="text",
         marker="r2 = (cc[..., 0]) ** 2", note="缺口原语"),
    dict(id="E06", f="src/diffmech/preprocess/mesh_primitives.py", ln=254, b=NB_, k="text",
         marker="in_notch = r2 + (cc[..., 1]) ** 2 <= notch_radius * notch_radius", note="缺口原语"),
    dict(id="E07", f="src/diffmech/preprocess/mesh_primitives.py", ln=257, b=NB_, k="text",
         marker="r2 + (cc[..., 1] - width / 2.0) ** 2 <= notch_radius * notch_radius", note="缺口原语"),
    dict(id="E08", f="src/diffmech/preprocess/mesh_primitives.py", ln=261, b=NB_, k="text",
         marker="r2 + (cc[..., 1] + width / 2.0) ** 2 <= notch_radius * notch_radius", note="缺口原语"),
    dict(id="E09", f="src/diffmech/preprocess/mesh_primitives.py", ln=296, b=NB_, k="text",
         marker="r2 = cc[..., 0] ** 2 + cc[..., 1] ** 2", note="圆孔原语"),
    dict(id="E10", f="src/diffmech/preprocess/mesh_primitives.py", ln=304, b=NB_, k="text",
         marker="r2 = cc[..., 0] ** 2 + cc[..., 2] ** 2", note="圆孔原语"),
    dict(id="E11", f="src/diffmech/preprocess/mesh_primitives.py", ln=312, b=NB_, k="text",
         marker="r2 = cc[..., 1] ** 2 + cc[..., 2] ** 2", note="圆孔原语"),
    dict(id="E12", f="src/diffmech/preprocess/mesh_primitives.py", ln=336, b=NB_, k="text",
         marker="r2 = cc[..., 0] ** 2 + cc[..., 1] ** 2", note="圆角原语"),
    dict(id="E13", f="src/diffmech/methods/am/am_microstructure.py", ln=560, b=NB_, k="text",
         marker="w = jnp.exp(-dist2 / (2.0 * (2.0 * dx) ** 2))", note="晶粒平均核，尺度是 2·dx 不是光斑"),
    dict(id="E14", f="src/diffmech/methods/am/am_microstructure.py", ln=558, b=NB_, k="text",
         marker="dist2 = jnp.sum(diff ** 2, axis=-1) + 1e-30",
         note="同上核的 ρ² 定义行（宽词汇后普查新见 ⇒ 证明尺子变严不是变松）"),
    # ---- 计划新增的单点出处（post 必须存在；**pre 必须不存在** ⇒ K0b 两侧都判）----
    # ⚠ 新增模块自己的**口径行**同样是普查命中（Z04/Z05 是 beam.py 里真正写指数/积分的
    # 两行；Z06-Z08 是描述新旧口径的**文字**行——它们把旧写法引用了一遍，估计器按
    # 「样符号 ∧ 数学」照样抓出来）。引用不能豁免：普查的意义是"每一行带口径的行都要
    # 有归属"，所以它们登记为 planned 行，post 前向门据此认领，pre 则断言其不存在。
    dict(id="Z01", f="src/amforge/beam.py", ln=24, b=IN_, k="planned", planned=True,
         marker="def spot_sigma", note="新增模块：σ = beam_radius/2 的唯一出处"),
    dict(id="Z02", f="src/amforge/beam.py", ln=29, b=IN_, k="planned", planned=True,
         marker="def planar_decay", note="新增模块：面内衰减的唯一写法"),
    dict(id="Z03", f="src/amforge/beam.py", ln=35, b=IN_, k="planned", planned=True,
         marker="def inplane_integral", note="新增模块：面内积分 2πσ² = π r²/2 的唯一出处"),
    dict(id="Z04", f="src/amforge/beam.py", ln=32, b=IN_, k="planned", planned=True,
         marker="return jnp.exp(-planar2 / (2.0 * sig * sig))",
         note="全树唯一的面内指数写法本体（K3 逐位门量的就是它）"),
    dict(id="Z05", f="src/amforge/beam.py", ln=37, b=IN_, k="planned", planned=True,
         marker="return 2.0 * math.pi * spot_sigma(beam_radius) ** 2",
         note="全树唯一的面内积分写法本体（K2 的 inplane_integral 实测走的就是它）"),
    dict(id="Z06", f="src/amforge/beam.py", ln=5, b=PR_, k="planned", planned=True,
         marker="I(ρ) = I0 · exp(−2ρ²/r_b²)", note="beam docstring：契约的 1/e² 声明"),
    dict(id="Z07", f="src/amforge/meltpool.py", ln=1547, b=PR_, k="planned", planned=True,
         marker="历史（#22）：面内曾写成 exp(−ρ²/rb²)",
         note="改后新增的**历史**段（引用旧写法 ⇒ 估计器照样命中，归属在此）"),
    dict(id="Z08", f="src/amforge/thermal_enthalpy.py", ln=318, b=PR_, k="planned",
         planned=True, marker="`exp(−ρ²/s_in²) = exp(−ρ²/(2σ_std²))`",
         note="改后新增的 s_in/σ_std 澄清段（同上，引用不豁免）"),
]

# ---------------------------------------------------------------------------
def _lines(rel: str) -> list[str]:
    p = REPO / rel
    if not p.exists():
        return []
    return p.read_text(encoding="utf-8").split("\n")


def grid(nn: int, half: float = HALF):
    xs = jnp.linspace(-half, half, nn)
    ys = jnp.linspace(-half, half, nn)
    xx, yy = jnp.meshgrid(xs, ys, indexing="ij")
    centers = jnp.stack([xx, yy], axis=-1)
    dxy = float(xs[1] - xs[0])
    return centers, dxy


def _depth_nodes(delta: float, nz: int, span: float = 8.0):
    """Beer-Lambert 深度积分的**单元中心**节点（z 向下为负，与生产网格同号）。

    端点矩形会重复计入 z=0 的表面值 ⇒ 系统性高估 ~h/2＝6%（δ=1.2r、h=δ/8 时），
    会污染 K2 的功率守恒读数；单元中心求积的偏差是 O(h²) 且与 `exp(-z/δ)` 的
    截断（`1-exp(-8)`＝0.999665）同阶，合计 ~1e-3 ⇒ 落在 5e-3 守恒容差内、
    离 √2＝41% 的口径分裂差三个数量级（K1/K3 判的是比値，深度权重会约掉）。
    """
    dz = span * delta / nz
    return -(jnp.arange(nz) + 0.5) * dz


def sigma_and_power(q, centers, dA, etaP: float | None = None):
    """σ 由二阶矩；ηP 给定时同时给 ΣQ·dV/(ηP)。"""
    rho2 = centers[..., 0] ** 2 + centers[..., 1] ** 2
    tot = float(jnp.sum(q))
    if not (tot > 0.0) or not math.isfinite(tot):
        return float("nan"), float("nan")
    sig = math.sqrt(float(jnp.sum(rho2 * q)) / tot / 2.0)
    pwr = (tot * dA / etaP) if etaP else float("nan")
    return sig, pwr


def measure(entry) -> tuple[float, float] | None:
    r = BEAM_R
    k = entry["k"]
    try:
        centers, dxy = grid(401)
        dA = dxy * dxy
        if k == "moment2d":
            from amforge.thermal_enthalpy import _moving_source
            q = _moving_source(jnp.zeros(2), centers, 1.0, r, 1e-6)
            return sigma_and_power(q, centers, dA)
        if k == "moment3d":
            from amforge.thermal_enthalpy import _moving_source
            c3 = jnp.concatenate([centers, jnp.zeros(centers.shape[:2] + (1,))], axis=-1)
            q = _moving_source(jnp.zeros(3), c3, 1.0, r, 1e-6)
            return sigma_and_power(q, centers, dA)
        if k == "momenterf":
            from amforge.thermal_enthalpy import _cell_integrated_source
            q = _cell_integrated_source(jnp.zeros(2), centers, 1.0, r, 1e-6, dxy)
            return sigma_and_power(q, centers, dA)
        if k == "momentfdm":
            from amforge.meltpool import _laser_source_fdm
            delta = 1.2 * r                       # 与生产同源 dp=1.2·rb（出域，本轮不改）
            zs = _depth_nodes(delta, 65)
            dz = 8.0 * delta / 65.0
            acc = None
            for z in zs:
                c3 = jnp.concatenate([centers, jnp.full(centers.shape[:2] + (1,), float(z))],
                                     axis=-1)
                q = _laser_source_fdm(c3, 0.0, rb=r, A_eff=1.0, P=1.0,
                                      absorption_depth=delta, dx=dxy)
                acc = q if acc is None else acc + q
            acc = acc * dz                         # 深度积分 ⇒ 面内分布
            sig, pwr = sigma_and_power(acc, centers, dA, 1.0)
            return sig, pwr
        if k == "momentdm":
            from diffmech.methods.fvm import thermal as th
            fn = (th.gaussian_heat_source if entry["id"] == "D13"
                  else th.moving_gaussian_source)
            kw = (dict(center=(0.0, 0.0)) if entry["id"] == "D13"
                  else dict(start=(0.0, 0.0), velocity=(0.0, 0.0)))
            src = fn(power=100.0, absorption=1.0, beam_radius=float(r), **kw)
            q = jnp.asarray(src(centers, 0.0))
            return sigma_and_power(q, centers, dA, 100.0)
    except Exception as exc:
        print(f"    [{entry['id']}] 实测异常 ⇒ {type(exc).__name__}: {exc}")
        return None
    return None


MESH_DIFF: dict[str, float] = {}


def mesh_converged(entry) -> bool:
    """K3 前置：n=401 与 n=801 的 σ 相对差 <1e-3（＝K3 容差 2e-3 的一半）。

    判据不是"越严越好"：中点求积对高斯二阶矩的**离散偏差本身**是 O(dx²)≈5e-4
    （相对 σ），把它设成 1e-4 会让**任何正确实现**都判"未收敛"（记忆教训：门槛
    必须可被正确实现达到）。1e-3 既容得下真实的离散偏差，又比要抓的 √2＝41%
    分裂小两个数量级 ⇒ 仍有牙。
    """
    r = BEAM_R
    try:
        a = _sigma_at(entry, 401)
        b = _sigma_at(entry, 801)
        if a != a or b != b:
            MESH_DIFF[entry["id"]] = float("nan")
            return False
        rel = abs(a - b) / b
        MESH_DIFF[entry["id"]] = rel
        return rel < 1e-3
    except Exception:
        MESH_DIFF[entry["id"]] = float("nan")
        return False


def _sigma_at(entry, nn: int) -> float:
    r = BEAM_R
    centers, dxy = grid(nn)
    dA = dxy * dxy
    k = entry["k"]
    if k == "moment2d":
        from amforge.thermal_enthalpy import _moving_source
        q = _moving_source(jnp.zeros(2), centers, 1.0, r, 1e-6)
    elif k == "moment3d":
        from amforge.thermal_enthalpy import _moving_source
        c3 = jnp.concatenate([centers, jnp.zeros(centers.shape[:2] + (1,))], axis=-1)
        q = _moving_source(jnp.zeros(3), c3, 1.0, r, 1e-6)
    elif k == "momenterf":
        from amforge.thermal_enthalpy import _cell_integrated_source
        q = _cell_integrated_source(jnp.zeros(2), centers, 1.0, r, 1e-6, dxy)
    elif k == "momentfdm":
        from amforge.meltpool import _laser_source_fdm
        delta = 1.2 * r
        zs = _depth_nodes(delta, 33)
        dz = 8.0 * delta / 33.0
        acc = None
        for z in zs:
            c3 = jnp.concatenate([centers, jnp.full(centers.shape[:2] + (1,), float(z))],
                                 axis=-1)
            q3 = _laser_source_fdm(c3, 0.0, rb=r, A_eff=1.0, P=1.0,
                                   absorption_depth=delta, dx=dxy)
            acc = q3 if acc is None else acc + q3
        q = acc * dz
    elif k == "momentdm":
        from diffmech.methods.fvm import thermal as th
        fn = (th.gaussian_heat_source if entry["id"] == "D13"
              else th.moving_gaussian_source)
        kw = (dict(center=(0.0, 0.0)) if entry["id"] == "D13"
              else dict(start=(0.0, 0.0), velocity=(0.0, 0.0)))
        src = fn(power=100.0, absorption=1.0, beam_radius=float(r), **kw)
        q = jnp.asarray(src(centers, 0.0))
    else:
        return float("nan")
    return sigma_and_power(q, centers, dA)[0]


# ---------------------------------------------------------------------------
gates: dict[str, str] = {}


def gate(name: str, ok: bool, detail: str = ""):
    """按**门编号**记账：`name` 的第一个词（`K0-S`／`K1 pre …`／`K3 post …`）就是编号。

    上一版把整条长标题当 key ⇒ `HARD` 里的 `"K0"` 永远查不到，汇总行只能打印
    「判了=[] 缺席=全部」，等于**没有汇总**（而它自己还是绿的）。这条教训：
    汇总器必须自检"每条硬门都被登记过"，缺席即 FAIL —— 现在的 `absent` 就是这个牙。
    """
    print(f"{name:64s} {'OK' if ok else 'FAIL'}  {detail}")
    gates[name.split(maxsplit=1)[0]] = "OK" if ok else "FAIL"


def _inplane_factor():
    """实测 `amforge.beam.inplane_integral(1.0)/π`：0.5 ⇒ 该函数写的是 π r²/2。"""
    try:
        from amforge.beam import inplane_integral
        return float(inplane_integral(1.0)) / math.pi
    except Exception as exc:
        print(f"    [K2] beam.inplane_integral 不可用 ⇒ {type(exc).__name__}: {exc}")
        return float("nan")


def area_factor_claimed(text: str) -> str:
    """从**源码文本**读该归一化行实际声称的面内积分：'π r²' 还是 'π r²/2'。

    不由我复述"修好后应该是什么"：`beam.inplane_integral` 的引用靠**实测**该函数的
    返回值判定；任何未识别的写法一律回落 'π r²' ⇒ 若实际改了却没被模式认出，K2 会
    以「声称 π r²／实测 0.5」**红**掉（错的方向是吵，不是静默通过）。
    """
    t = text.replace(" ", "").replace("\t", "")
    if "inplane_integral" in t:
        f = _inplane_factor()
        if f == f:
            return "π r²/2" if abs(f - 0.5) < 1e-9 else (
                "π r²" if abs(f - 1.0) < 1e-9 else "?")
        return "?"
    pats = (r"(math\.pi|jnp\.pi)\*r_src\*r_src/2",
            r"(math\.pi|jnp\.pi)\*rb\*rb/2",
            r"0\.5\*\(?(math\.pi|jnp\.pi)\*r(src|b)?\*r(src|b)?",
            r"(math\.pi|jnp\.pi)\*r(src|b)?\*r(src|b)?\*0\.5",
            r"(math\.pi|jnp\.pi)\*r(src|b)?\*\*2(/2|\*0\.5)")
    return "π r²/2" if any(re.search(p, t) for p in pats) else "π r²"


def norm_window(e) -> tuple[str, str]:
    """K2 要读的源码片段：返回 (片段文本, 定位方式)。

    ⚠ **不能一路按登记行号读**：本轮在两个文件顶部各加一行 import，登记行号整体下移，
    post 用旧行号读到的是无关文本 ⇒ 把「已改成 inplane_integral」误读成「仍写 π r²」，
    报出 `实测 A/πR²=0.5 ⇒ 半修！` 的**假红灯**（第一版 post 就红在这里，红在门上不在
    树上）。定位规则：post 优先用 `post_marker` 找行，pre（或无 post_marker）用登记行；
    两者都找不到 ⇒ 返回空串，让 `area_factor_claimed` 回落 'π r²' 而与实测 0.5 冲突变
    红（错的方向是吵，不是静默通过）。
    """
    txt = _lines(e["f"])
    pm = e.get("post_marker")
    if PHASE == "post" and pm:
        for i, t in enumerate(txt):
            if pm in t:
                return "\n".join(txt[i:i + 3]), f"post_marker@{i + 1}"
    return "\n".join(txt[e["ln"] - 1:e["ln"] + 2]), f"登记行{e['ln']}"


print("== T4/#22 光斑口径探针 ==")
print(f"phase   = {PHASE}")
print(f"REPO    = {REPO}")
print(f"devices = {jax.devices()}")
dirty = subprocess.run(["git", "-C", str(REPO), "status", "--short", "src", "tests"],
                       capture_output=True, text=True).stdout.strip().splitlines()
print(f"dirty_src = {len(dirty)}  {dirty[:12]}")
print(f"外证  D4σ={D4:.6e} m ⇒ σ=D4/4={SIGMA_A3:.6e} m；c2 半径=D4/2={R_C2:.6e} m；"
      f"c1 半径=√2σ={R_C1:.6e} m")
print(f"实测网格  HALF={HALF:.6e} m（=8·c1 半径）、n=401（另跑 n=801 自检收敛）")

# ---- K0-S 估计器自检 ----
miss_self = [nm for nm, line in SELF_CHECK_SPELLINGS if not is_census_line(line)]
gate("K0-S 估计器自检（8 副拼写全命中，含修复后的正确拼写）", not miss_self,
     f"未命中={miss_self}")

# ---- K0 普查完备 ----
hits = census_hits()
n_planned = len([e for e in REG if e.get("planned")])
print(f"\n-- K0 普查：口径敏感命中 {len(hits)} 行；登记 {len(REG) - n_planned} 行"
      f"（＋{n_planned} 项计划新增）--")
if PHASE == "pre":
    reg_keys = {(e["f"], e["ln"]) for e in REG if not e.get("planned")}
    unreg = [h for h in hits if (h[0], h[1]) not in reg_keys]
    drift = [e["id"] for e in REG if not e.get("planned") and e["b"] != PR_
             and not any(e["f"] == h[0] and e["ln"] == h[1] for h in hits)]
    for rel, i, line in hits:
        tag = "登记" if (rel, i) in reg_keys else "★未登记"
        print(f"   [{tag}] {rel}:{i}: {line.strip()[:104]}")
    gate("K0 普查完备（未登记／行号漂 ⇒ VOID）", (not unreg) and (not drift),
         f"未登记={[f'{a}:{b}' for a, b, _ in unreg]} 登记未命中={drift}")
else:
    # **post 的键换成文本，不是行号**：本轮会在两个文件顶部各加一行 import ⇒ 同文件
    # 全部登记行整体平移，行号绑定**必然**失效（第一版 post 就是红在这里，红在门上
    # 不在树上）。前向门（"每个命中都必须已登记"）用文本仍然完整成立：
    #   ①未动的行 ⇒ 某行的 `marker`（改前原文片段）落在命中行文本里；
    #   ②改过的行 ⇒ 某行的 `post_marker` 落在命中行文本里，且**只允许** IN／PR 桶这样
    #     覆盖（CONFORMS 位点"用新写法覆盖"＝它被改了，那是本门要抓的事）。
    # 后向门（"登记了却普查不到 ⇒ VOID"）在 post 降级为**读数**：改成
    # `planar_decay(planar2, r)` 之后那行不再含 exp()/π ⇒ 天然退出命中集，这是改动的
    # 后果而非漏洞；后向的实质保证由 K0b（旧写法整文件消失 ∧ 新写法必须存在）＋
    # K4（c1 残留计数＝0 ∧ beam 单点出处 ∧ 越界文件门）三条 post 硬门承担。
    # ⚠ post 用**全集**（含 planned 行）：新增的 beam.py 与两处历史/澄清段落本身就是
    # 口径敏感行，pre 时它们不存在所以当时只按「＋N 项计划新增」读数，post 必须被认领。
    rows = list(REG)
    unmapped = []
    cover_by = []
    moved = []
    for rel, i, line in hits:
        t = line.strip()
        cand = [e["id"] for e in rows if e["f"] == rel and e["marker"] in t]
        cand2 = [e["id"] for e in rows if e["f"] == rel and e.get("post_marker")
                 and e["post_marker"] in t and e["b"] in (IN_, PR_)]
        if not cand and not cand2:
            unmapped.append(f"{rel}:{i}")
        cover_by.append((f"{rel}:{i}", (cand or cand2)[0] if (cand or cand2) else "?"))
        print(f"   [{'登记' if (cand or cand2) else '★未登记'}] {rel}:{i}: {t[:104]}")
    for e in rows:
        if e["b"] == PR_:
            continue
        ln = _lines(e["f"])
        cur = ln[e["ln"] - 1] if e["ln"] <= len(ln) else ""
        if e["marker"] not in cur and (e["marker"] in "\n".join(ln) or
                                       (e.get("post_marker") or "") in "\n".join(ln)):
            moved.append(e["id"])
    gate("K0 普查完备（post＝文本键前向门；未登记 ⇒ VOID）", not unmapped,
         f"未登记={unmapped} 行号漂移(读数)={len(moved)} 项")

# ---- K0b 文本绑定（两侧都是硬门；改后按桶分治＝U3「替换守护必须在两副口径同时成立」）----
bad_bind = []
for e in REG:
    txt = _lines(e["f"])
    whole = "\n".join(txt)
    if e.get("planned"):
        # 计划新增行：**两副口径都判**（post 必须存在∧pre 必须不存在）。只判"post 有"是
        # 半条门——它允许这些行改前就在树里，那样"新增单点出处"就成了复述现状。
        if PHASE == "post":
            if not txt or e["marker"] not in whole:
                bad_bind.append((e["id"], "post 但缺该定义/该段"))
        elif txt and e["marker"] in whole:
            bad_bind.append((e["id"], "pre 却已存在该计划新增行"))
        continue
    if PHASE == "pre":
        # 改前：marker 必须**正好在登记行**（这一条同时兜住"登记表抄错行号"）。
        cur = txt[e["ln"] - 1] if e["ln"] <= len(txt) else ""
        if e["marker"] not in cur:
            bad_bind.append((e["id"], "改前 marker 不在登记行"))
        continue
    pm = e.get("post_marker")
    if pm is not None:
        # 要改的行：旧写法必须**整文件消失** ∧ 新写法必须存在（只改一半 ⇒ 旧串仍在 ⇒ 红）。
        if e["marker"] in whole:
            bad_bind.append((e["id"], f"改后旧写法仍在文件里：{e['marker'][:40]!r}"))
        if pm not in whole:
            bad_bind.append((e["id"], f"改后新写法不存在：{pm!r}"))
    elif e["marker"] not in whole:
        # 不该动的行：文本必须仍在（防"删掉位点来通过普查"；行号漂移由 K4 的
        # `git diff --name-only` 文件级门＋本条的文本级门共同兜）。
        bad_bind.append((e["id"], "改后不该动的行消失了"))
gate(f"K0b 文本绑定（两副口径，共 {len(REG)} 项）", not bad_bind, f"异常={bad_bind}")

# ---- 实测表 ----
print("\n-- 二阶矩实测：σ_实测 与契约 σ=R/2 之比（量的不是拼写，是实际多宽）--")
measured: dict[str, tuple[float, float, float, dict]] = {}
for e in REG:
    if e.get("planned") or e["k"] in ("text", "normtext", "planned", "prose"):
        continue
    res = measure(e)
    if res is None:
        print(f"   [{e['id']}] 测不到 ⇒ 不计入 K1/K3（照实登记，不当通过）")
        continue
    sig, pwr = res
    ratio = sig / (BEAM_R / 2.0) if sig == sig else float("nan")
    measured[e["id"]] = (sig, pwr, ratio, e)
    pw = f"{pwr:.9f}" if pwr == pwr else "  —  "
    print(f"   [{e['id']}] σ={sig:.9e}  σ/(R/2)={ratio:.9f}  ΣQ·dV/(ηP)={pw}  {e['f']}:{e['ln']}")

# ---- K1（pre 专属）----
if PHASE == "pre":
    off_sqrt2 = [k for k, v in measured.items()
                 if v[3]["f"].startswith("src/amforge") and v[3]["b"] == IN_
                 and abs(v[2] - math.sqrt(2.0)) < 1e-3]
    at_one = [k for k, v in measured.items() if abs(v[2] - 1.0) < 1e-3]
    gate("K1 pre 正对照（见到 c1 位点偏 √2，且已有位点为 1.0）",
         bool(off_sqrt2) and bool(at_one), f"偏√2={off_sqrt2} 已合={at_one}")
else:
    print("K1 改后不判（pre 专属：post 的 0 分裂正是目标，判它会把'已修好'读成 FAIL）")

# ---- K2 内部一致性（抓半修）＋ 可直接测的功率守恒 ----
bad_k2 = []
for e in REG:
    if e["k"] != "normtext":
        continue
    # 读**定位行起的 3 行窗口**：改后若把归一化写成续行/换行，单行读会漏看 while
    # 声称"已改"。定位方式随口径走（见 `norm_window`：post 按文本，pre 按登记行）。
    line, where = norm_window(e)
    claim = area_factor_claimed(line)
    decay_entry = {"A04": "A01", "A06": "A05"}[e["id"]]
    if decay_entry not in measured:
        bad_k2.append((e["id"], "配对的衰减位点测不到"))
        continue
    ratio = measured[decay_entry][2]                 # σ_decay/(R/2)
    # 实测面内积分 A_meas 对「π R²」的倍数：c1 ⇒ 1.0；c2 ⇒ 0.5
    sig = measured[decay_entry][0]
    a_meas_over_piR2 = 2.0 * (sig ** 2) / (BEAM_R ** 2)
    expect = {"π r²": 1.0, "π r²/2": 0.5}[claim]
    ok = abs(a_meas_over_piR2 - expect) < 2e-3
    print(f"   [{e['id']}] 归一化（{where}）实际写着 {claim}（⇒ 要求 A/πR²={expect}）；"
          f"实测 A/πR²={a_meas_over_piR2:.9f} ⇒ {'一致' if ok else '半修！'}")
    if not ok:
        bad_k2.append((e["id"], f"声称 {claim} 但实测 A/πR²={a_meas_over_piR2:.6f}"))
for k, (sig, pwr, ratio, e) in measured.items():
    if pwr != pwr:
        continue
    if not (abs(pwr - 1.0) < 5e-3):
        bad_k2.append((k, f"功率不守恒 ΣQ·dV/(ηP)={pwr:.6f}"))
gate("K2 内部一致性（衰减实测积分 == 源码实际写的归一化因子）＋ 功率守恒", not bad_k2,
     f"异常={bad_k2}")

# ---- K3（post 专属）----
if PHASE == "post":
    ins = [k for k, v in measured.items() if v[3]["b"] == IN_]
    notconv = [k for k, v in measured.items() if not mesh_converged(v[3])]
    bad3 = []
    for k in ins:
        sig, ratio, e = measured[k][0], measured[k][2], measured[k][3]
        # 统一 2e-3：erf 档虽然"单元捕获份额"是解析精确的（Σfrac=1），但二阶矩仍按
        # **单元中心**读数 ⇒ 与中点取值档同样吃 O(dx²) 离散偏差。给它 1e-6 的单独
        # 严杠会让**正确实现**也过不了（记忆教训：门槛必须可被正确实现达到）；
        # 离散不确定性由 mesh_converged（n=401 vs 801 相对差 <1e-3）单独封顶。
        if abs(ratio - 1.0) > 2e-3:
            bad3.append((k, ratio))
    rho2 = jnp.linspace(-1.0, 1.0, 257) ** 2
    a = jnp.exp(-2.0 * rho2 / (BEAM_R ** 2))
    b = jnp.exp(-rho2 / (2.0 * (BEAM_R / 2.0) ** 2))
    from amforge.beam import planar_decay, spot_sigma
    c = planar_decay(rho2, BEAM_R)
    spell = bool(jnp.all(a == b) and jnp.all(c == a) and jnp.all(c == b))
    sigbit = bool(jnp.all(jnp.asarray(spot_sigma(BEAM_R)) == BEAM_R / 2.0))
    gate("K3 post 统一（IN 位点 σ=R/2 ＋ 三副拼写逐位相同 ＋ 网格已收敛）",
         bool(ins) and not bad3 and not notconv and spell and sigbit,
         f"IN={ins} 偏离={bad3} 未收敛={notconv} 网格差={ {k: float(f'{v:.2e}') for k, v in MESH_DIFF.items()} } "
         f"拼写逐位同={spell} max|a−c|={float(jnp.max(jnp.abs(a - c))):.3e} σ_bit={sigbit}")
else:
    print("K3 改前不判（post 专属）")

# ---- K4（post 专属）----
if PHASE == "post":
    te = "\n".join(_lines("src/amforge/thermal_enthalpy.py"))
    mp = "\n".join(_lines("src/amforge/meltpool.py"))
    uses = ("from amforge.beam import" in te) and ("from amforge.beam import" in mp)
    c1_left = re.findall(r"exp\(\s*-planar2\s*/\s*jnp\.maximum\(\s*(?:r|rb)\s*\*\s*(?:r|rb)",
                         te + "\n" + mp)
    dm_inline = all("-2.0 * rsq / r2" in _lines(e["f"])[e["ln"] - 1]
                    for e in REG if e["f"].startswith("src/diffmech") and e["b"] == CF_
                    and "rsq" in e["marker"])
    am_ratios = [v[2] for v in measured.values() if v[3]["b"] == IN_]
    dm_ratios = [v[2] for k, v in measured.items() if v[3]["f"].startswith("src/diffmech")]
    cross = bool(am_ratios) and bool(dm_ratios) and \
        all(abs(x - 1.0) < 2e-3 for x in am_ratios + dm_ratios)
    # **文件级范围门**：本轮只许碰 beam／两条默认求解器所在文件／那条手写参考＋新增的
    # 口径回归测试。
    # K0b 的文本门只保证"登记行没被删"，保证不了"没顺手改同文件的别处"⇒ 用 git 记账。
    touched = sorted({ln[3:].strip() for ln in subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain", "src", "tests"],
        capture_output=True, text=True).stdout.split("\n") if ln.strip()})
    allowed = {"src/amforge/beam.py", "src/amforge/thermal_enthalpy.py",
               "src/amforge/meltpool.py", "tests/test_enthalpy_thermal.py",
               "tests/test_beam_convention.py"}
    out_of_scope = [p for p in touched if p not in allowed]
    gate("K4 post 单点出处（amforge 经 beam；无 c1 残留；diffmech 内联不变；跨包同宽；"
         "改动文件不越界）",
         uses and not c1_left and dm_inline and cross and not out_of_scope,
         f"用beam={uses} c1残留行数={len(c1_left)} 内联保持={dm_inline} 跨包同宽={cross} "
         f"越界文件={out_of_scope} 实改={touched}")
else:
    print("K4 改前不判（post 专属）")

# ---- K5 绝对期望 ----
# 语义要分清两件事：① **同一个输入数**在 c1/c2 下的宽度比＝√2（这就是"分裂"）；
# ② 要让两者都描述 A3 的 σ=21.25µm，**输入数**必须取 R_c1=√2σ 与 R_c2=2σ ⇒ 比＝1/√2。
same_sigma_inputs = abs(R_C1 / math.sqrt(2.0) - R_C2 / 2.0) < 1e-18
input_ratio = abs(R_C1 / R_C2 - 1.0 / math.sqrt(2.0)) < 1e-12
same_number_sigma_ratio = abs((BEAM_R / math.sqrt(2.0)) / (BEAM_R / 2.0) - math.sqrt(2.0)) < 1e-12
same_number_area_ratio = abs((BEAM_R ** 2) / (2.0 * (BEAM_R / 2.0) ** 2) - 2.0) < 1e-12
c1_ratio_sqrt2 = abs(R_C1 / SIGMA_A3 - math.sqrt(2.0)) < 1e-12
sigma_is_d4over4 = abs(SIGMA_A3 - 21.25e-6) < 1e-20
c2_is_half_d4 = abs(R_C2 - 42.5e-6) < 1e-20
gate("K5 绝对期望（σ=D4/4=21.25µm、c2=D4/2=42.5µm、c1=√2σ、同数 σ 比=√2、面积比=2×）",
     all([c1_ratio_sqrt2, sigma_is_d4over4, c2_is_half_d4, same_sigma_inputs,
          input_ratio, same_number_sigma_ratio, same_number_area_ratio]),
     f"√2σ={c1_ratio_sqrt2} D4/4={sigma_is_d4over4} D4/2={c2_is_half_d4} "
     f"两输入同σ={same_sigma_inputs} R_c1/R_c2=1/√2={input_ratio} "
     f"同数σ比=√2={same_number_sigma_ratio} 面积比=2×{same_number_area_ratio}")

# ---- K6 出域读数 ----
print("\n-- K6 出域不变量读数（本轮一律不改 ⇒ #33 靶表）--")
dm_pa = "\n".join(_lines("src/diffmech/methods/am/particle_am.py"))
pb_bed = "\n".join(_lines("src/diffmech/methods/am/powder_bed.py"))
TE = "\n".join(_lines("src/amforge/thermal_enthalpy.py"))
MP = "\n".join(_lines("src/amforge/meltpool.py"))
readouts = {
    "σ_z：dp = 1.2·r（thermal_enthalpy :646）": "dp = jnp.maximum(r * 1.2, 1e-9)" in TE,
    "σ_z：dp = 1.2·r_src（point 档 :650）": "dp = jnp.maximum(r_src * 1.2, dx)" in TE,
    "σ_z：dp = max(r_eff·1.2, dx)（fdm :1631）": "dp = jnp.maximum(r_eff * 1.2, dx)" in MP,
    "r_eff = max(rb, dx)（fdm :1630）": "r_eff = jnp.maximum(rb, dx)" in MP,
    "r_src = max(r, dx)（point :649）": "r_src = jnp.maximum(r, dx)" in TE,
    "demo 档抬半径 0.5·dx（:656）": "jnp.maximum(r, 0.5 * dx)" in TE,
    "分辨率报错门 dx > 2.0·r_nom（:453）": "2.0 * r_nom" in TE,
    "熔盒尺度 max(R, 1.5·beam_radius)（:1360）": "1.5 * beam_radius" in MP,
    "point 档旋钮 heat_scale 缺省 1.4（:654）": 'p.get("heat_scale", 1.4)' in TE,
    "point 档旋钮 clip(0.8/v, 0.3, 3.0)（:653）": "jnp.clip(0.8 / jnp.maximum(_v, 1e-9), 0.3, 3.0)" in TE,
    "集粉宽度 1.8·beam_radius（diffmech :1203）": "(1.8 * cfg.beam_radius) ** 2" in dm_pa,
    "熔深经验 1.2·rb（diffmech powder_bed :675）": "d_melt = 1.2 * rb" in pb_bed,
}
for kk, v in readouts.items():
    print(f"   {'在' if v else '缺'}  {kk}")
sig_xy_old = BEAM_R / math.sqrt(2.0)
sig_xy_new = BEAM_R / 2.0
dp_val = BEAM_R * 1.2
print(f"   读数 同一输入 R={BEAM_R:.6e}：σ_xy 改前={sig_xy_old:.6e} → 改后={sig_xy_new:.6e}（÷√2）；"
      f"σ_z 由 1.2·R 决定＝{dp_val:.6e} **本轮不动** ⇒ 长宽比 σ_z/σ_xy "
      f"{dp_val / sig_xy_old:.6f} → {dp_val / sig_xy_new:.6f}（×√2）⇒ #33")

# ---- K7 assert 行原文不动 ----
def assert_pair(rel: str, needle: str):
    head = subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{rel}"],
                          capture_output=True, text=True).stdout.split("\n")
    work = _lines(rel)
    return ([l.strip() for l in head if needle in l],
            [l.strip() for l in work if needle in l])


bad7 = []
for rel, needle in (
        ("tests/test_enthalpy_thermal.py",
         "assert abs(vol_c - vol_f) / max(vol_c, vol_f) < 0.05"),
        ("tests/test_enthalpy_thermal.py",
         "assert abs(pk_c - pk_f) / max(pk_c, pk_f) < 0.05"),
        ("tests/test_enthalpy_thermal.py", "assert pk_f > pk_c - 0.05 * pk_c"),
        ("tests/test_enthalpy_thermal.py", 'assert n > 0, f"dx={dx_um}'),
        ("tests/test_boundary.py", "assert peaks[2] < peak0 - 0.02 * peak0"),
        ("tests/test_boundary.py", "assert mean0 - means[2] > 300.0"),
):
    h, w = assert_pair(rel, needle)
    if not h:
        bad7.append((rel, needle, "HEAD 里找不到该 assert ⇒ 判据自身失效"))
    elif h != w:
        bad7.append((rel, needle, f"HEAD={h} ≠ 工作树={w}"))
gate("K7 已登记红灯的 assert 行原文逐字未动（阈值/指标未换）", not bad7, f"异常={bad7}")

HARD = {"pre": ("K0-S", "K0", "K0b", "K1", "K2", "K5", "K7"),
        "post": ("K0-S", "K0", "K0b", "K2", "K3", "K4", "K5", "K7")}
judged = [k for k in HARD[PHASE] if k in gates]
failed = [k for k in judged if gates[k] != "OK"]
absent = [k for k in HARD[PHASE] if k not in gates]
print(f"\n门槛汇总（{PHASE} 硬门）: {'PASS' if not failed and not absent else 'FAIL'} "
      f"判了={judged} 失败={failed} 缺席={absent}")
print(f"探针 md5 = {hashlib.md5(Path(__file__).read_bytes()).hexdigest()}")
sys.exit(0 if not failed and not absent else 1)
