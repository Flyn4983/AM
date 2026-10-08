"""#27 蒸发封顶的处置依据：经验封顶 ``c·ss·ΔT`` vs 材料卡 Hertz–Knudsen 封顶（**只测不改**）。

红灯（登记见 `tests/test_boundary.py::test_convection_removes_energy` docstring 与 §26.20）：
    ③ 代（#23 之后）h=20000 的压峰只剩 **−0.963%**，于是守护
    `assert peaks[2] < peak0 − 0.02·peak0` 为红，pytest 实打印
    `2908.771450618092 vs 2937.060997302282`。
归因（`docs/evidence/2026-10-07/am_t27_convection_peak_probe.log` U1/U2）＝**蒸发封顶在
fv=0.5 的刀锋面刚度翻倍**（`fv·S → fv·S/cap_w = S`：注入能量不变、钳位变硬），不是散热被吞。

本轮要回答的不是"怎么让这条测试转绿"，而是**这把封顶本身该换成什么**：
现封顶 `c_evap=3.0e11`（`thermal_enthalpy.py:611`）是**按旧口径标定的自由系数**（#22 之后
半新半旧），而 `src/amforge/materials.py:198` 已实现**无自由参数**的 Hertz–Knudsen 蒸发通量
（`ṁ = 0.82·p_sat(T)·√(M/(2πRT))`，`p_sat` 走 Clausius–Clapeyron，参数全部来自材料卡
`T_boil=3090K / latent_vapor=7.45e6 / molar_mass=0.0559`）⇒ 候选替换是**外部接地**的，
符合 #27 的纪律「只能以外部靶重标定，禁止以让测试转绿为靶」。

已知的**外部靶缺口**（本轮必须先讲清）：`docs/evidence/2026-10-07/am_a3_fixture.json` 的
`benchmark.bc` ＝「顶面 Neumann alpha*u；左右前后底全绝热；**无对流、无辐射、无蒸发**」，
且 `switches_to_match_benchmark` 明确要求 `evap_coeff=0` 才能对齐基准 ⇒ **A3 不提供封顶的靶**。
所以选型只能靠"封顶自身是否有外部接地的形式"，而不是靠回归拟合。

范围与边界（照实写）：
* **`src/` 与 `tests/` 本轮一字不动**；HK 封顶用 **monkeypatch** 替换模块函数
  `TE._evap_sink`（`rhs` 在 trace 时按全局名解析 ⇒ 替换生效），诊断专用。
* CPU 钉住（`CUDA_VISIBLE_DEVICES=""`）：G0 要与 pytest 的**逐位**浮点锚点对比，
  跨设备比较无效；本探针是**归因/选型**、不是性能测量，故不含设备清单行。
* 子进程各跑一副（arm × scenario），父进程只做判据与表，不重解。

判据（**跑前写死，不许事后改**）
---------------------------------
G0 同构硬锚（任一不过 ⇒ 全轮读数不入档、不作判据）：
  G0a 生产口径（arm=tuned, h=0）必须**逐位**给出 `peak0 == 2937.060997302282`；
  G0b arm=tuned 的 h=20000 必须**逐位**给出 `peaks == 2908.771450618092`；
  G0c 末态实体均值(h=0) 必须命中 #28 登记的 ③ 代值 `2720.80 K`（±0.05，docstring 是四舍五入数）。
  ⇒ 证明探针的夹具/BC 构造与那条测试**同构**；否则 G2 量的不是同一件事。

G1 两把封顶同表 + 量纲自检（纯算术，父进程）：
  G1a **量纲自检**：HK 体密度 `S_hk(T) = ṁ(T)·L_v/dx` [W/m³] 在 `T=T_boil, dx=80µm` 处必须等于
      **独立手算值**（材料卡四数 + R + accommodation 0.82 抄成字面常数、用 `math` 而非模块函数算）
      相对误差 < 1e-9；并打印指数参数 `arg`，要求限幅 `clip(arg,−50,20)` 与 `max(T,300)`
      在自检点**不生效**（否则自检测的不是名义公式）。
      ⚠ `÷dx` 的摊法与 `boundary.py:184-186` 的既有约定同源（表面 [W/m²] → [J/m³/s]）。
  G1b 表：T ∈ {1700, 2000, 2500, 2700, 2890, 2937, 3000, 3090, 3200, 3500, 4000}，
      列 tuned / HK / 比值 / 各占**实测**峰值体源上界 `Q_peak` 的百分比；
      再列两副口径的有效值（fv=1 → `S` 两口径同；fv=0.5 新口径 → `S`；fv=0.5 旧口径 → `0.5S`）。
      `Q_peak` 由子进程用**生产热源函数** `_cell_integrated_source` 在顶面实体体素附近抽样取最大
      （上界参照，不是逐步实测——照实标注）。
  G1c **绝对预期**（把 docstring 那句「低温熔化区不介入」从断言变成**被测命题**）：
      `S_hk(2700K) > 0` 且 `S_hk(T_liquidus) > 0`，而 tuned 在 `T ≤ T_lo=2890K` **恒等于 0**。
      V1（否决条件，跑前定死）：`S_hk(T_liquidus)/Q_peak ≥ 0.1%` ⇒ HK **不是中性替换**
      （它会移动以 A3 宽度/面积为靶的标定），无论 G2a 结果如何都**不得**作为默认替换。

G2 四臂对流夹具 × 四把封顶（monkeypatch；夹具＝G0 那副）：
  arm ∈ {tuned, hk, hk_surf, off}，h ∈ {0, 200, 2000, 20000}（16 解）。
  `hk_surf`＝HK 只乘自由面体素（蒸发物理上发生在液-气界面）；`hk`＝体相摊平（对照/上限）。
  G2a **决定性判据**：arm=hk 与 arm=hk_surf 下，测试那条守护
      `peaks(h=20000) < peak0 − 2%·peak0` 是否成立——**阈值一字不动**。
  G2b 反向对照：arm=off（无封顶）的压峰幅度 ⇒ 区分"封顶钳位掩盖对流"与"夹具本来就压不动"。
  G2c 副作用登记（**未来会变的生产数字**，跑前不许参考、跑后不许追认）：
      hk / hk_surf / off 相对 tuned 的 `peak0`、末态实体均值、`Vn`、`Vm`（h=0 一臂）变化百分比。

G3 A3 侧污染（**代理，不是夹具**；#18 未落地 ⇒ 18 道版跑不了）：
  单道代理：285 W / 960 mm·s⁻¹ / r_b=42.5µm（1/e²，#22 后口径）/ α=0.54（标定档）/
  dx=40µm / `h_cool=0`（与基准对齐）；arm ∈ {tuned, hk, off}（3 解）。
  读数＝峰值、`peak > T_lo?`、代理熔宽/熔深/俯视熔面积。
  ⚠ 该夹具 BC 明写「无蒸发」⇒ 本臂**不参与**任何外部靶达标判定，只回答
  "换封顶会不会移动我们打算标定的宽度/面积读数"。

G4 能量闭合与可微性（h=0，无 BC，同初值同方程、只差封顶项）：
  `E_cap = Σ H(T_final)·fv·dx³`（`H` 用生产 `enthalpy_of_temperature`）；
  `E_inj ≡ E(h_cool=0, 无封顶) − E0`（**实测**注入能，不拿解析式充数）；
  `removed(arm) = E(off, h_cool=0.5) − E(arm, h_cool=0.5)`。
  G4a 闭合：`E_inj ≥ 0` 且对 tuned/hk/hk_surf 各自 `0 ≤ removed ≤ E(off) − E0`；
      并打印 `removed/E_inj`（＝这把封顶究竟搬走了多少注入能量）。
  G4b 可微性：`jax.grad` 对**整个 plan**（`laser_power` 叶子）在 tuned 与 hk 两臂下都必须有限；
      `n_steps` 先用 eager 的 `suggest_n_steps()` 钉好（trace 下静态形状无法从 tracer 推）。
      另检 HK 自身 `dS/dT` 在 G1b 网格上处处有限，并**定位 clip 造成的梯度冻结区**
      （`arg=−50` 以下 `S≡const`、`dS/dT=0`）——这决定替换会不会在低温区留下静默零梯度。

G5 #33 的 **15 个位点按执行覆盖**分类（`sys.settrace` 子进程，行事件只记本仓库文件）：
  位点用**源码文本**解析成当前行号（行号键会过期 ⇒ 不硬编）；命中数 ≠1 ⇒ 标 `moved/ambiguous`，
  不算命中。场景：S-default / S-point / S-demo / S-chain(`chain_schedule`) /
  S-fdm(`solve_meltpool_fdm`) / S-powderbed(`powderbed._analytic_base`，生产链) /
  S-length(直连 `_melt_length_scale`) / S-criteria(直连 `recommended_power_for_enthalpy`) /
  S-inverse(直连 `process_constraint_penalty`) / S-particle(直连 `lsf_powder_deposition_profile`，
  duck-typed paths ⇒ 只证明"被调用时这行会执行")。
  G5a **正对照**（不过 ⇒ G5 整节作废并如实打印）：S-default 必须命中
      `dp = jnp.maximum(r * 1.2, 1e-9)`。
  G5b 仪器局限照写：settrace 是**行粒度** ⇒ 同一行的三元式两分支
      （`thermal_enthalpy.py:669`）不可分，故该位点标注 `same-line-ternary`，
      其"demo 分支才执行"由 S-default(不该有最大值调用) 与 S-demo(应有) 的**取值差**旁证；
      直连场景 = `unit-only`，**不**证明生产链会调用它。

处置规则（**跑前写死，唯一允许的结论分支**）
--------------------------------------------
* G0 任一不过 ⇒ 本轮全部读数不入档，`src/`、`tests/` 一字不动。
* G2a 通过（HK 让原守护在**一字不动**的前提下成立）∧ G4a 闭合 ∧ V1 未触发
  ⇒ 判「**可采纳**：#27 以无自由参数的物理封顶替换经验封顶」，进实现轮；实现轮必须另做：
  全量回归、A3 宽度/面积外部靶核对、以及一个**关闭开关**——`am_a3_fixture.json` 要求
  封顶可置 0 才能对齐基准，而无参数公式没有"置 0"旋钮 ⇒ 必须新增显式
  `evap_model="tuned"|"hk"|"none"` 之类的选择器，否则会把 A3 对齐路径堵死。
* G2a 不通过 ⇒ 判「**换封顶不能解决**」：登记为需**外部靶**才能推进（现基准集内 A3＝无蒸发
  ⇒ 无靶），走 #29/#30 那种"等裁决/等新基准"路线；**禁止**调 `c_evap` 凑绿（N4）。
* V1 触发 ⇒ HK **不得**作默认替换（它移动 A3 靶的读数），只能作高 T 专用分支，
  且两分支间的不连续必须登记。
* 任一情况下 `tests/test_boundary.py` 的判据、阈值（两处 0.02）与 docstring 本轮一字不动。
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

# ---- G0 硬锚（pytest 打印的逐位数 + #28 登记的 ③ 代四舍五入值）----------------
ANCHOR_PEAK0 = 2937.060997302282
ANCHOR_PEAK_H = 2908.771450618092
ANCHOR_MEAN0 = (2720.80, 0.05)

HS = (0.0, 200.0, 2000.0, 20000.0)
ARMS = ("tuned", "hk", "hk_surf", "off")
T_GRID = (1700.0, 2000.0, 2500.0, 2700.0, 2890.0, 2937.0, 3000.0,
          3090.0, 3200.0, 3500.0, 4000.0)
V1_BAR_PCT = 0.1          # G1c/V1：HK 在 T_liquidus 处占峰值体源的上界（超过 ⇒ 非中性）
GUARD_PT = 0.02           # 测试那条守护的阈值，一字不动

# ---- 材料卡字面常数（G1a 的独立手算；抄自 src/amforge/materials.py:336-349）---
LIT = dict(T_boil=3090.0, latent_vapor=7.45e6, molar_mass=0.0559, T_liquidus=1723.0)
RGAS = 8.314462618
BETA_EVAP = 0.82
P_ATM = 101325.0
C_EVAP = 3.0e11           # 现生产经验系数（thermal_enthalpy.py:611）
DX_CONV = 80e-6           # 对流夹具 spacing

SOLVER_SCENARIOS = ("S-default", "S-point", "S-demo", "S-chain", "S-fdm", "S-powderbed")


# ---------------------------------------------------------------------------
# 父进程用的两把封顶（独立实现，与子进程/模块互不复用代码）
# ---------------------------------------------------------------------------
def tuned_sink(T: float, T_lo: float, T_hi: float, c: float = C_EVAP) -> float:
    x = min(max((T - T_lo) / max(T_hi - T_lo, 1.0), 0.0), 1.0)
    ss = x * x * (3.0 - 2.0 * x)
    return c * ss * max(T - T_lo, 0.0)


def hk_arg(T: float) -> float:
    return (LIT["latent_vapor"] * LIT["molar_mass"] / RGAS) * (
        1.0 / LIT["T_boil"] - 1.0 / max(T, 300.0))


def hk_sink(T: float, dx: float) -> float:
    arg = hk_arg(T)
    p_sat = P_ATM * math.exp(min(max(arg, -50.0), 20.0))
    mdot = BETA_EVAP * p_sat * math.sqrt(
        LIT["molar_mass"] / (2.0 * math.pi * RGAS * max(T, 300.0)))
    return mdot * LIT["latent_vapor"] / dx


# ---------------------------------------------------------------------------
# 子进程 A：跑一副 (arm × scenario)
# ---------------------------------------------------------------------------
CHILD = r'''
import json, math, os, sys
import numpy as np, jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import amforge.thermal_enthalpy as TE
from amforge.boundary import BoundaryCondition, BoundaryCollection
from amforge.core.contracts import ProcessPlan, solid_mask, solid_weight
from amforge.gui.preproc import build_primitive
from amforge.materials import get_material

ARM = os.environ["T27P_ARM"]
SCEN = os.environ["T27P_SCEN"]
R_GAS = 8.314462618
mat = get_material("316L")
LV = float(mat.latent_vapor); MM = float(mat.molar_mass); TB = float(mat.T_boil)
RHO = float(mat.rho_solid); CP = float(mat.cp_solid); LAT = float(mat.latent_fusion)
T_AMB = float(mat.T_ambient); T_SOL = float(mat.T_solidus); T_LIQ = float(mat.T_liquidus)
T_LO = TB - 200.0

STATE = {"dx": 1.0, "surf": 0.0}

def hk_volumetric(T, dx):
    """Hertz-Knudsen：S = mdot(T)·L_v/dx  [W/m³]（表面量÷dx，与 boundary.py 同源）。"""
    arg = (LV * MM / R_GAS) * (1.0 / TB - 1.0 / jnp.maximum(T, 300.0))
    p_sat = 101325.0 * jnp.exp(jnp.clip(arg, -50.0, 20.0))
    mdot = 0.82 * p_sat * jnp.sqrt(MM / (2.0 * jnp.pi * R_GAS * jnp.maximum(T, 300.0)))
    return mdot * LV / dx

def patched_sink(T, T_lo, T_hi, c):
    s = hk_volumetric(T, STATE["dx"])
    if ARM == "hk_surf":
        s = s * STATE["surf"]
    return s

if ARM in ("hk", "hk_surf"):
    TE._evap_sink = patched_sink              # monkeypatch：rhs 在 trace 时按全局名解析

def coeff():
    return 1e-6 if ARM == "off" else None      # None ⇒ 缺省 3.0e11（hk* 时该系数被忽略）

def fixture_conv():
    part = build_primitive("box", length_mm=0.4, spacing_um=80.0)
    plan = ProcessPlan.uniform(4, modality="SLM", laser_power=200.0, scan_speed=1.0,
                               layer_thickness=40e-6, hatch_spacing=80e-6,
                               beam_radius=50e-6, absorption=0.4, preheat_temp=373.0)
    return part, plan, (0.0, 200.0, 2000.0, 20000.0)

def fixture_a3():
    part = build_primitive("box", length_mm=0.9, spacing_um=40.0)
    plan = ProcessPlan.uniform(1, modality="SLM", laser_power=285.0, scan_speed=0.96,
                               layer_thickness=40e-6, hatch_spacing=110e-6,
                               beam_radius=4.25e-5, absorption=0.54,
                               preheat_temp=float(mat.T_ambient))
    return part, plan, (None,)

def setup(part):
    dx = float(part.spacing)
    sdf = jnp.asarray(part.sdf)
    sol = np.asarray(solid_mask(sdf) > 0.5)
    fv = np.asarray(solid_weight(sdf, jnp.asarray(dx, dtype=jnp.float64)))
    pad = np.pad(sol, 1, constant_values=False)
    nb = (pad[:-2, 1:-1, 1:-1] & pad[2:, 1:-1, 1:-1] & pad[1:-1, :-2, 1:-1]
          & pad[1:-1, 2:, 1:-1] & pad[1:-1, 1:-1, :-2] & pad[1:-1, 1:-1, 2:])
    STATE["dx"] = dx
    STATE["surf"] = jnp.asarray((sol & ~nb).astype(np.float64), dtype=jnp.float64)
    return dx, sol, fv

def energy(tf, fv, dx):
    H = np.asarray(TE.enthalpy_of_temperature(
        jnp.asarray(tf, dtype=jnp.float64), rho=RHO, cp=CP, L=LAT, T_amb=T_AMB,
        T_sol=T_SOL, T_liq=T_LIQ))
    return float(np.sum(H * fv) * dx ** 3)

def e0(plan, fv, dx):
    pre = float(np.ravel(np.asarray(plan.preheat_temp))[0])
    T_init = pre if (math.isfinite(pre) and pre > T_AMB) else T_AMB
    H = float(np.asarray(TE.enthalpy_of_temperature(
        jnp.asarray(T_init, dtype=jnp.float64), rho=RHO, cp=CP, L=LAT, T_amb=T_AMB,
        T_sol=T_SOL, T_liq=T_LIQ)))
    return H * float(np.sum(fv)) * dx ** 3

def q_peak(part, plan, dx, sol):
    """生产热源函数在顶面实体体素附近抽样取最大 ⇒ 体源**上界参照**（非逐步实测）。"""
    coords = part.coords()
    eta = float(np.ravel(np.asarray(plan.absorption))[0])
    P = float(np.ravel(np.asarray(plan.laser_power))[0])
    r = float(np.ravel(np.asarray(plan.beam_radius))[0])
    dp = max(r * 1.2, 1e-9)
    idx = np.argwhere(sol)
    ztop = int(idx[:, 2].max())
    tops = idx[idx[:, 2] == ztop]
    step = max(1, int(math.ceil(len(tops) / 60.0)))          # 抽样，控成本
    power = eta * P * 1.0                                   # integrated 档 heat_scale=1.0
    best = 0.0
    for c in tops[::step]:
        base = np.asarray(part.coords()[c[0], c[1], c[2]], dtype=np.float64)
        for kd in (-0.5, 0.0, 0.5):
            pos = base.copy()
            pos[2] = base[2] + kd * dx
            q = np.asarray(TE._cell_integrated_source(
                jnp.asarray(pos, dtype=jnp.float64), coords, power, r, dp, dx))
            best = max(best, float(q.max(initial=0.0)))
    return float(best)

def run(part, plan, h, extra=None, force_params=None):
    params = dict(force_params) if force_params else {"material": "316L"}
    c = coeff()
    if c is not None:
        params["evap_coeff"] = c
    if extra:
        params.update(extra)
    if h is not None and h > 0.0:
        strong = BoundaryCollection(bcs=tuple(
            BoundaryCondition("convection", f, h, 293.0)
            for f in ("+Z", "+X", "-X", "+Y", "-Y")), ic=None)
        params["boundary_conditions"] = strong
    return TE.solve_enthalpy_thermal(geometry=part, process=plan, params=params)

out = {"arm": ARM, "scen": SCEN, "runs": []}

if SCEN in ("conv", "inj"):
    part, plan, hs = fixture_conv()
    dx, sol, fv = setup(part)
    out.update(shape=list(part.sdf.shape), dx=dx, n_solid=int(sol.sum()),
               te_file=TE.__file__, arm_fn=TE._evap_sink.__name__,
               e0=e0(plan, fv, dx), q_peak=q_peak(part, plan, dx, sol),
               n_surf=int(np.sum(np.asarray(STATE["surf"])) > 0))
    if SCEN == "inj":                       # G4 的注入能参照：无冷却 + 无封顶
        th = run(part, plan, None, force_params={"material": "316L",
                                                 "h_cool": 0.0, "evap_coeff": 1e-6})
        tf = np.asarray(th.final_temperature)
        out["runs"].append(dict(tag="inj_ref", h=None, peak=float(np.max(th.peak_temperature)),
                                E=energy(tf, fv, dx),
                                finite=bool(np.isfinite(tf).all())))
    for h in hs:
        th = run(part, plan, h)
        pk = np.asarray(th.peak_temperature); tf = np.asarray(th.final_temperature)
        ix = np.unravel_index(int(np.argmax(pk)), pk.shape)
        melted = (pk > T_LIQ) & sol
        out["runs"].append(dict(
            h=h, peak=float(pk.max()), peak_idx=[int(v) for v in ix],
            argmax_fv=float(fv[ix]), argmax_surf=float(np.asarray(STATE["surf"])[ix]),
            mean_final_solid=float(tf[sol].mean()),
            vn=float(melted.sum()) * dx ** 3 * 1e9,
            vm=float((fv * melted).sum()) * dx ** 3 * 1e9,
            E=energy(tf, fv, dx), n_above_lo=int(((pk > T_LO) & sol).sum()),
            finite=bool(np.isfinite(pk).all() and np.isfinite(tf).all())))
    if SCEN == "conv" and ARM in ("tuned", "hk"):
        # G4b：对**整个 plan** 求 grad（laser_power 是可微叶子；n_steps 先 eager 钉好）
        ns = int(TE.suggest_n_steps(part, plan, material="316L"))
        def peak_of(pl):
            th = TE.solve_enthalpy_thermal(geometry=part, process=pl,
                                           params={"material": "316L", "n_steps": ns})
            return jnp.max(th.peak_temperature)
        pl0 = plan.replace(laser_power=jnp.asarray([200.0] * 4, dtype=jnp.float64))
        try:
            g = jax.grad(peak_of)(pl0)
            out["grad_plan"] = dict(n_steps=ns,
                                    dpeak_dP=float(np.asarray(g.laser_power)[0]),
                                    finite=bool(np.isfinite(np.asarray(g.laser_power)).all()))
        except Exception as exc:                          # 照实登记，不吞
            out["grad_plan"] = {"err": repr(exc)[:240]}

elif SCEN == "a3":
    part, plan, _ = fixture_a3()
    dx, sol, fv = setup(part)
    out.update(shape=list(part.sdf.shape), dx=dx, n_solid=int(sol.sum()),
               te_file=TE.__file__, arm_fn=TE._evap_sink.__name__,
               q_peak=q_peak(part, plan, dx, sol))
    th = run(part, plan, None, force_params={"material": "316L", "h_cool": 0.0})
    pk = np.asarray(th.peak_temperature)
    melted = (pk > T_LIQ) & sol
    top2d = melted.any(axis=2)
    i0, i1 = np.where(top2d)
    zs = np.where(melted.any(axis=(0, 1)))[0]      # np.where 单参返回元组 ⇒ 必须取 [0]
    out["runs"].append(dict(
        h=None, peak=float(pk.max()), above_lo=bool(float(pk.max()) > T_LO),
        length_mm=(float((i0.max() - i0.min() + 1) * dx * 1e3) if i0.size else 0.0),
        width_mm=(float((i1.max() - i1.min() + 1) * dx * 1e3) if i1.size else 0.0),
        depth_mm=(float((zs.max() - zs.min() + 1) * dx * 1e3) if zs.size else 0.0),
        area_top_mm2=float(top2d.sum()) * dx ** 2 * 1e6,
        vn=float(melted.sum()) * dx ** 3 * 1e9,
        E=energy(np.asarray(th.final_temperature), fv, dx),
        finite=bool(np.isfinite(pk).all())))

print("JSON::" + json.dumps(out))
'''

# ---------------------------------------------------------------------------
# 子进程 B：位点执行覆盖
# ---------------------------------------------------------------------------
SITES = (
    # (id, 相对路径, 源码文本指纹, 期望场景, 所在函数)
    ("S01-sigma", "src/amforge/thermal_enthalpy.py", "dp = jnp.maximum(r * 1.2, 1e-9)",
     "S-default", "solve_enthalpy_thermal"),
    ("S02-sigma", "src/amforge/thermal_enthalpy.py", "dp = jnp.maximum(r_src * 1.2, dx)",
     "S-point", "solve_enthalpy_thermal"),
    ("S03-sigma", "src/amforge/meltpool.py", "dp = jnp.maximum(r_eff * 1.2, dx)",
     "S-fdm", "solve_meltpool_fdm"),
    ("S04-lift", "src/amforge/meltpool.py", "r_eff = jnp.maximum(rb, dx)",
     "S-fdm", "solve_meltpool_fdm"),
    ("S05-lift", "src/amforge/thermal_enthalpy.py", "r_src = jnp.maximum(r, dx)",
     "S-point", "solve_enthalpy_thermal"),
    ("S06-lift", "src/amforge/thermal_enthalpy.py", "jnp.maximum(r, 0.5 * dx)",
     "S-demo", "solve_enthalpy_thermal[same-line-ternary]"),
    ("S07-lift", "src/amforge/thermal_enthalpy.py", "if dx > 2.0 * r_nom:",
     "S-chain", "_chain_schedule"),
    ("S08-knob", "src/amforge/meltpool.py", "return jnp.maximum(R, 1.5 * beam_radius)",
     "S-fdm", "_melt_length_scale"),
    ("S09-knob", "src/amforge/thermal_enthalpy.py", 'heat_scale", 1.4',
     "S-point", "solve_enthalpy_thermal"),
    ("S10-knob", "src/amforge/thermal_enthalpy.py",
     "jnp.clip(0.8 / jnp.maximum(_v, 1e-9), 0.3, 3.0)", "S-point", "solve_enthalpy_thermal"),
    ("S11-knob", "src/diffmech/methods/am/particle_am.py", "(1.8 * cfg.beam_radius) ** 2",
     "S-particle", "lsf_powder_deposition_profile"),
    ("S12-knob", "src/diffmech/methods/am/powder_bed.py", "d_melt = 1.2 * rb * jnp.sqrt",
     "S-powderbed", "analytic_powder_defects"),
    ("S13-crit", "src/amforge/core/contracts.py",
     "denom = jnp.sqrt(jnp.pi * diffusivity * self.scan_speed", "S-powderbed",
     "ProcessPlan.normalized_enthalpy"),
    ("S14-crit", "src/amforge/process.py",
     "denom = jnp.sqrt(jnp.pi * alpha * scan_speed * beam_radius", "S-criteria",
     "recommended_power_for_enthalpy"),
    ("S15-crit", "src/amforge/inverse.py",
     "w = 2.0 * plan.beam_radius * jnp.sqrt(1.0 + jnp.maximum(dH", "S-inverse",
     "process_constraint_penalty"),
)

CHILD_COV = r'''
import json, os, sys, types
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

OWN = tuple(os.environ["T27P_OWN"].split("|"))
HITS = set()
OWNSET = set(OWN)

def _hit(f):
    return any(f.endswith(o) for o in OWNSET)

def _local(frame, event, arg):
    if event == "line" and _hit(frame.f_code.co_filename):
        HITS.add((frame.f_code.co_filename, frame.f_lineno))
        return _local
    if event == "call":
        return _local if _hit(frame.f_code.co_filename) else None
    return None

def _global(frame, event, arg):
    if event != "call":
        return None
    if _hit(frame.f_code.co_filename):
        HITS.add((frame.f_code.co_filename, frame.f_lineno))
        return _local
    return None                       # 非本仓库文件：不追行事件（控成本）

sys.settrace(_global)

SCEN = os.environ["T27P_SCEN"]
extra = {}

def small():
    from amforge.gui.preproc import build_primitive
    from amforge.core.contracts import ProcessPlan
    part = build_primitive("box", length_mm=0.2, spacing_um=80.0)
    plan = ProcessPlan.uniform(1, modality="SLM", laser_power=200.0, scan_speed=1.0,
                               layer_thickness=40e-6, hatch_spacing=80e-6,
                               beam_radius=50e-6, absorption=0.4, preheat_temp=373.0)
    return part, plan

import amforge.thermal_enthalpy as TE

if SCEN == "S-default":
    part, plan = small()
    TE.solve_enthalpy_thermal(geometry=part, process=plan, params={"material": "316L"})
elif SCEN == "S-point":
    part, plan = small()
    TE.solve_enthalpy_thermal(geometry=part, process=plan,
                              params={"material": "316L", "source_model": "point"})
elif SCEN == "S-demo":
    part, plan = small()
    # 不传 n_steps：让 eager 自动取 CFL 档（传 6 会被稳定校核当场拒绝）
    TE.solve_enthalpy_thermal(geometry=part,
                              process=plan.replace(beam_radius=jnp.asarray([2.0e-5])),
                              params={"material": "316L",
                                      "resolution_policy": "demo"})
elif SCEN == "S-chain":
    part, plan = small()
    try:
        extra["chain"] = str(TE.chain_schedule(part, plan, material="316L"))[:100]
    except Exception as exc:
        extra["chain"] = "ERR:" + repr(exc)[:200]
elif SCEN == "S-fdm":
    from amforge.meltpool import solve_meltpool_fdm
    from amforge import geometry as G
    geo = G.from_sdf_fn(lambda x: jnp.linalg.norm(x, axis=-1) - 0.45e-3,
                        bounds=[(-0.6e-3, 0.6e-3)] * 3, spacing=50e-6, name="cov")
    from amforge.core.contracts import ProcessPlan
    proc = ProcessPlan.uniform(laser_power=250.0, scan_speed=0.8, layer_thickness=40e-6,
                               hatch_spacing=80e-6, beam_radius=50e-6, absorption=0.4)
    try:
        r = solve_meltpool_fdm(geometry=geo, process=proc,
                               params={"material": "316L", "n_grid": 24, "n_steps": 40})
        extra["fdm_depth"] = float(r.depth)
    except Exception as exc:
        extra["fdm"] = "ERR:" + repr(exc)[:240]
elif SCEN == "S-powderbed":
    from amforge.powderbed import _analytic_base
    part, plan = small()
    try:
        extra["pb"] = sorted(_analytic_base(part, plan, {"material": "316L"}).keys())
    except Exception as exc:
        extra["pb"] = "ERR:" + repr(exc)[:240]
elif SCEN == "S-length":
    from amforge.meltpool import _melt_length_scale
    extra["len"] = float(_melt_length_scale(absorbed_power=88.0, scan_speed=0.9,
                                            beam_radius=5e-5, rho=7950.0,
                                            enthalpy_to_melt=3.0e8))
elif SCEN == "S-criteria":
    import amforge.process as PR
    from amforge.materials import get_material
    extra["pr"] = float(PR.recommended_power_for_enthalpy(
        get_material("316L"), target_enthalpy=12.0, scan_speed=1.0, beam_radius=5e-5))
elif SCEN == "S-inverse":
    import amforge.inverse as INV
    try:
        pen, det = INV.process_constraint_penalty(jnp.full(9, 0.5), material="316L",
                                                  n_layers=1)
        extra["pen"] = float(pen)
    except Exception as exc:
        extra["pen"] = "ERR:" + repr(exc)[:240]
elif SCEN == "S-particle":
    from diffmech.methods.am.particle_am import lsf_powder_deposition_profile
    import diffmech.methods.am.process as P_PLAIN
    import src.diffmech.methods.am.process as P_SRC
    lay = types.SimpleNamespace(waypoints=np.array([[0.0, 0.0], [1.0e-3, 0.0]]),
                                cum_length=np.array([0.0, 1.0e-3]),
                                total_length=1.0e-3, layer_z=0.0)
    paths = types.SimpleNamespace(layers=[lay])
    pos = jnp.asarray([[0.0, 0.0], [5e-5, 0.0]])

    def _call(cfg):                                  # (Σ, 非零个数)
        v = np.asarray(lsf_powder_deposition_profile(
            pos, 0.0, paths, jnp.asarray([0.0]), jnp.asarray([1.0e-3]), cfg, dim=2))
        return (float(v.sum()), int((v > 0).sum()))

    try:
        # 守卫在 particle_am.py 里 `from src.diffmech…import LSFConfig`（:1181/:1236），
        # 而生产调用方拿到的是 `diffmech.…` 顶层别名 ⇒ 两个类对象不同 ⇒ isinstance 恒假。
        # 两副口径都跑：plain 别名＝当前生产可达路径，src 别名＝与该守卫同类的口径。
        extra["part"] = _call(P_PLAIN.LSFConfig())
        extra["part_srcalias"] = _call(P_SRC.LSFConfig())
        extra["alias_same_class"] = bool(P_PLAIN.LSFConfig is P_SRC.LSFConfig)
    except Exception as exc:
        extra["part"] = "ERR:" + repr(exc)[:240]
else:
    raise SystemExit("unknown scen " + SCEN)

sys.settrace(None)

rel = {}
for f, ln in HITS:
    for o in OWNSET:
        if f.endswith(o):
            rel.setdefault(o, set()).add(ln)
print("JSON::" + json.dumps({k: sorted(v) for k, v in rel.items()}))
print("EXTRA::" + json.dumps(extra, default=str)[:600])
'''


def coverage(scen: str, own_paths: list[str]) -> dict[str, set[int]]:
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "src"),
               CUDA_VISIBLE_DEVICES="", JAX_ENABLE_X64="1",
               T27P_SCEN=scen, T27P_OWN="|".join(own_paths))
    t0 = time.perf_counter()
    pr = subprocess.run([sys.executable, "-c", CHILD_COV], env=env,
                        capture_output=True, text=True, cwd=REPO)
    el = time.perf_counter() - t0
    line = [ln for ln in pr.stdout.splitlines() if ln.startswith("JSON::")]
    if not line:
        print(f"  [{scen}] {el:.0f}s 无读数 ⇒ rc={pr.returncode} "
              + pr.stderr[-400:].replace("\n", " | "), flush=True)
        return {}
    extra = [ln for ln in pr.stdout.splitlines() if ln.startswith("EXTRA::")]
    d = json.loads(line[0][6:])
    print(f"  [{scen}] {el:.0f}s 命中行数={sum(len(v) for v in d.values())}  "
          + (extra[0][7:][:400] if extra else ""), flush=True)
    return {k: set(v) for k, v in d.items()}


def resolve_site(relpath: str, snippet: str) -> tuple[list[int], str]:
    """按源码文本解析当前行号（行号键会过期 ⇒ 不硬编）。"""
    with open(os.path.join(REPO, relpath), encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    hits = [i + 1 for i, ln in enumerate(lines) if snippet in ln]
    if len(hits) == 1:
        return hits, "ok"
    return hits, ("moved" if not hits else f"ambig×{len(hits)}")


def child(arm: str, scen: str) -> dict:
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "src"),
               CUDA_VISIBLE_DEVICES="", JAX_ENABLE_X64="1",
               T27P_ARM=arm, T27P_SCEN=scen)
    t0 = time.perf_counter()
    pr = subprocess.run([sys.executable, "-c", CHILD], env=env,
                        capture_output=True, text=True, cwd=REPO)
    el = time.perf_counter() - t0
    if pr.returncode != 0:
        raise SystemExit(f"子进程失败（arm={arm} scen={scen}）：\n"
                         f"{pr.stdout[-2500:]}\n{pr.stderr[-3500:]}")
    line = [ln for ln in pr.stdout.splitlines() if ln.startswith("JSON::")]
    if not line:
        raise SystemExit(f"子进程无读数：\n{pr.stdout[-2500:]}\n{pr.stderr[-2000:]}")
    d = json.loads(line[0][6:])
    d["_elapsed"] = el
    print(f"    [{scen}/{arm}] {el:.0f}s shape={d['shape']} dx={d['dx']*1e6:.0f}µm "
          f"n_solid={d['n_solid']} sink_fn={d['arm_fn']}", flush=True)
    return d


def main() -> int:
    t_all = time.perf_counter()
    own = sorted({s[1] for s in SITES})
    print("== #27/#33 蒸发封顶选型探针（**只测不改**：src/ 与 tests/ 一字不动）==", flush=True)
    print(f"device = {os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
          "（CPU 钉住：G0 要逐位比 pytest 锚点 ⇒ 跨设备无效）", flush=True)

    # ---------------- G5：位点执行覆盖 --------------------------------------
    print("\n[G5] #33 的 15 位点执行覆盖（sys.settrace，行粒度，只记本仓库文件）", flush=True)
    by_scen: dict[str, dict[str, set[int]]] = {}
    for scen in SOLVER_SCENARIOS + ("S-length", "S-criteria", "S-inverse", "S-particle"):
        by_scen[scen] = coverage(scen, own)

    ln0, st0 = resolve_site(SITES[0][1], SITES[0][2])
    pos_hit = bool(ln0) and st0 == "ok" and all(
        x in by_scen["S-default"].get(SITES[0][1], set()) for x in ln0)
    print(f"\n  G5a 正对照：S-default 命中 {SITES[0][0]}（行 {ln0}，resolve={st0}）⇒ "
          f"{'PASS（仪器有效）' if pos_hit else 'FAIL ⇒ G5 整节作废'}", flush=True)

    cov_rows = []
    print(f"\n  {'site':<11}{'现树位置':<30}{'resol':<7}{'覆盖场景':<34}{'分类'}", flush=True)
    for sid, rel, snip, want, fn in SITES:
        ln, status = resolve_site(rel, snip)
        covering = [s for s, d in by_scen.items()
                    if ln and all(x in d.get(rel, set()) for x in ln)]
        solver_hits = [s for s in covering if s in SOLVER_SCENARIOS]
        if status != "ok":
            cls = "SITE-MOVED（需人工）"
        elif not covering:
            cls = "MISS（本探针场景内未执行）"
        elif solver_hits:
            cls = "live-solver"
        else:
            cls = "unit-only"
        cov_rows.append((sid, rel, ln, status, covering, cls, fn, want in covering))
        print(f"  {sid:<11}{rel.split('/')[-1] + ':' + '/'.join(map(str, ln)):<30}"
              f"{status:<7}{(','.join(covering) or '-'):<34}{cls}", flush=True)

    # ---------------- G0/G2/G4：四臂 × 四档 ---------------------------------
    print("\n[G0/G2/G4] 对流夹具（0.4mm / dx=80µm / 4 层 200W·1.0m·s⁻¹ / 5 面对流）",
          flush=True)
    res: dict[str, dict] = {}
    for arm in ARMS:
        res[arm] = child(arm, "conv")
    inj = child("off", "inj")

    tuned = res["tuned"]
    r0, r_h = tuned["runs"][0], tuned["runs"][3]
    g0a = r0["peak"] == ANCHOR_PEAK0
    g0b = r_h["peak"] == ANCHOR_PEAK_H
    g0c = abs(r0["mean_final_solid"] - ANCHOR_MEAN0[0]) <= ANCHOR_MEAN0[1]
    g0 = g0a and g0b and g0c
    print(f"\n  G0a peak0 == {ANCHOR_PEAK0!r}？实测 {r0['peak']!r} ⇒ {'PASS' if g0a else 'FAIL'}",
          flush=True)
    print(f"  G0b peaks(h=20000) == {ANCHOR_PEAK_H!r}？实测 {r_h['peak']!r} ⇒ "
          f"{'PASS' if g0b else 'FAIL'}", flush=True)
    print(f"  G0c 末态均值(h=0) == {ANCHOR_MEAN0[0]}±{ANCHOR_MEAN0[1]}？实测 "
          f"{r0['mean_final_solid']:.4f} ⇒ {'PASS' if g0c else 'FAIL'}", flush=True)
    print(f"  峰值落点 idx={r0['peak_idx']} fv={r0['argmax_fv']} 自由面={r0['argmax_surf']} "
          f"（n_solid={tuned['n_solid']}，`_evap_sink`→{tuned['arm_fn']}）", flush=True)
    if not g0:
        print("  ⇒ G0 不过：本轮全部读数**不入档、不作判据**（处置规则第 1 条）。", flush=True)

    qp = float(tuned["q_peak"])
    T_lo = LIT["T_boil"] - 200.0

    # ---------------- G1 ----------------------------------------------------
    print(f"\n[G1] 两把封顶同表（dx={DX_CONV*1e6:.0f}µm，T_lo={T_lo:.0f}K；"
          f"Q_peak={qp:.3e} W/m³＝生产热源函数在顶面附近抽样取的**上界参照**）", flush=True)
    arg_b = hk_arg(LIT["T_boil"])
    p_sat = P_ATM * math.exp(arg_b)
    mdot = BETA_EVAP * p_sat * math.sqrt(
        LIT["molar_mass"] / (2.0 * math.pi * RGAS * LIT["T_boil"]))
    hand = mdot * LIT["latent_vapor"] / DX_CONV
    got = hk_sink(LIT["T_boil"], DX_CONV)
    rel_err = abs(hand - got) / hand
    clip_ok = -50.0 < arg_b < 20.0 and LIT["T_boil"] > 300.0
    g1a = rel_err < 1e-9 and clip_ok
    print(f"  G1a 量纲自检：手算 {hand:.9e} vs 实现 {got:.9e} ⇒ 相对差 {rel_err:.3e}"
          f"（<1e-9 {'OK' if rel_err < 1e-9 else 'BAD'}）；自检点 arg={arg_b:+.6f} "
          f"限幅未生效 {'OK' if clip_ok else 'BAD(测的不是名义公式)'} ⇒ "
          f"{'PASS' if g1a else 'FAIL'}", flush=True)
    print(f"\n  {'T[K]':>6}{'tuned':>13}{'HK':>13}{'HK/tuned':>11}"
          f"{'tuned/Qp%':>11}{'HK/Qp%':>10}{'fv0.5旧':>12}{'fv0.5新':>12}{'dHK/dT':>12}", flush=True)
    dsc = {}
    for T in T_GRID:
        a = tuned_sink(T, T_lo, LIT["T_boil"])
        b = hk_sink(T, DX_CONV)
        eps = 1e-3
        dsc[T] = (hk_sink(T + eps, DX_CONV) - hk_sink(T - eps, DX_CONV)) / (2 * eps)
        print(f"  {T:>6.0f}{a:>13.3e}{b:>13.3e}"
              f"{(b / a if a > 0 else float('nan')):>11.4f}{a / qp * 100:>11.5f}"
              f"{b / qp * 100:>10.5f}{0.5 * b:>12.3e}{b:>12.3e}{dsc[T]:>12.3e}", flush=True)
    s_liq = hk_sink(LIT["T_liquidus"], DX_CONV)
    g1c = (hk_sink(2700.0, DX_CONV) > 0.0) and (s_liq > 0.0) and \
        all(tuned_sink(T, T_lo, LIT["T_boil"]) == 0.0
            for T in (1700.0, 2000.0, 2700.0, T_lo))
    v1 = s_liq / qp * 100.0 >= V1_BAR_PCT
    print(f"  G1c 绝对预期：HK(2700)={hk_sink(2700.0, DX_CONV):.3e}>0、HK(T_liq)={s_liq:.3e}>0、"
          f"tuned 在 T≤T_lo 恒 0 ⇒ {'PASS' if g1c else 'FAIL（docstring 那句不成立）'}", flush=True)
    print(f"  V1：HK(T_liq)/Q_peak = {s_liq / qp * 100:.5f}% "
          f"{'>=' if v1 else '<'} {V1_BAR_PCT}% ⇒ "
          f"{'**触发**：HK 非中性，不得作默认替换' if v1 else '未触发：正常熔化区可忽略'}",
          flush=True)
    T_freeze = 1.0 / (1.0 / LIT["T_boil"] + 50.0 * RGAS / (LIT["latent_vapor"] * LIT["molar_mass"]))
    print(f"  clip 冻结区：arg=−50 解出 T={T_freeze:.1f}K（低于它 dS/dT=0、S≡const）；"
          f"本次网格上 dHK/dT 全为有限正值 ⇒ {'OK' if all(math.isfinite(v) and v > 0 for v in dsc.values()) else 'BAD'}"
          f"；T_ambient={LIT['T_liquidus'] and 293.0:.0f}K 是否在冻结区之上＝"
          f"{'是' if 293.0 > T_freeze else '否（预热区梯度被 clip 冻结，值已达 1e-19 量级 ⇒ 无物理影响）'}",
          flush=True)

    # ---------------- G2 ----------------------------------------------------
    print("\n[G2] 四臂压峰（守护阈值 2% **一字不动**）", flush=True)
    print(f"  {'arm':<9}{'peak0':>11}{'Δh200%':>9}{'Δh2000%':>10}{'Δh20000%':>11}"
          f"{'守护':>7}{'末态Δ均值K':>13}{'n>T_lo':>8}", flush=True)
    g2a = {}
    for arm in ARMS:
        runs = res[arm]["runs"]
        p0 = runs[0]["peak"]
        dp = [(r["peak"] - p0) / p0 * 100.0 for r in runs[1:]]
        guard = runs[3]["peak"] < p0 - GUARD_PT * p0
        g2a[arm] = guard
        dm = runs[3]["mean_final_solid"] - runs[0]["mean_final_solid"]
        print(f"  {arm:<9}{p0:>11.2f}{dp[0]:>9.3f}{dp[1]:>10.3f}{dp[2]:>11.3f}"
              f"{'PASS' if guard else 'red':>7}{dm:>13.2f}{runs[0]['n_above_lo']:>8}", flush=True)
    print(f"  G2a 决定性（原守护）：hk {'满足' if g2a['hk'] else '不满足'}、"
          f"hk_surf {'满足' if g2a['hk_surf'] else '不满足'}、"
          f"tuned {'满足' if g2a['tuned'] else '不满足（＝现红灯）'}", flush=True)
    print(f"  G2b 反向对照：arm=off 的 Δpeak%(h=20000) = "
          f"{(res['off']['runs'][3]['peak'] - res['off']['runs'][0]['peak']) / res['off']['runs'][0]['peak'] * 100:.3f}%"
          f" ⇒ 无封顶时该守护 {'PASS' if g2a['off'] else 'red'}（'夹具本来就压不动' vs '封顶掩盖'的分界）",
          flush=True)
    print("\n  G2c 副作用（h=0 一臂，相对 tuned）：", flush=True)
    base = res["tuned"]["runs"][0]
    for arm in ("hk", "hk_surf", "off"):
        rr = res[arm]["runs"][0]
        print(f"    {arm:<9}" + "  ".join(
            f"{k} {(rr[k] - base[k]) / base[k] * 100:+.2f}%"
            for k in ("peak", "mean_final_solid", "vn", "vm"))
            + f"   peak={rr['peak']:.1f}K Vn={rr['vn']:.3f}mm³", flush=True)

    # ---------------- G4 ----------------------------------------------------
    print("\n[G4] 能量闭合（E = Σ H(T_final)·fv·dx³；E_inj 用 h_cool=0 无封顶**实测**）", flush=True)
    E0 = float(tuned["e0"])
    E_inj = inj["runs"][0]["E"] - E0
    off_h0 = [r for r in res["off"]["runs"] if r["h"] == 0.0][0]
    print(f"  E0(预热 373K)={E0:.6e} J；E_inj(实测, h_cool=0/无封顶)={E_inj:.6e} J ⇒ "
          f"{'PASS' if E_inj >= 0 else 'FAIL'}；该臂峰值={inj['runs'][0]['peak']:.1f}K", flush=True)
    g4a = True
    for arm in ("tuned", "hk", "hk_surf"):
        e_arm = [r for r in res[arm]["runs"] if r["h"] == 0.0][0]["E"]
        removed = off_h0["E"] - e_arm
        bound = off_h0["E"] - E0
        ok = (-1e-9 <= removed <= bound + 1e-9)
        g4a = g4a and ok
        print(f"  {arm:<9}E={e_arm:.6e} 搬走 {removed:.6e} J ＝ 注入的 "
              f"{removed / E_inj * 100:.3f}%（闭合 {'PASS' if ok else 'FAIL'}；上界 {bound:.3e}）",
              flush=True)
    gp = {a: res[a].get("grad_plan") for a in ("tuned", "hk")}
    g4b = all(isinstance(v, dict) and v.get("finite") is True for v in gp.values())
    print(f"  G4b 可微性 ∂peak/∂P：{gp} ⇒ {'PASS' if g4b else 'FAIL/ERR'}", flush=True)

    # ---------------- G3 ----------------------------------------------------
    print("\n[G3] A3 单道**代理**（285W/0.96m·s⁻¹/r_b=42.5µm/α=0.54/dx=40µm/h_cool=0）", flush=True)
    print("  ⚠ 代理：#18 未落地 ⇒ 18 道夹具版跑不了；且夹具 BC 明写「无蒸发」⇒ 本臂**不参与**"
          "外部靶判定。", flush=True)
    a3 = {arm: child(arm, "a3")["runs"][0] for arm in ("tuned", "hk", "off")}
    print(f"  {'arm':<7}{'peak[K]':>10}{'>T_lo?':>8}{'width':>8}{'depth':>8}{'length':>9}"
          f"{'area_top':>11}{'Vn':>9}{'E[J]':>12}", flush=True)
    for arm in ("tuned", "hk", "off"):
        r = a3[arm]
        print(f"  {arm:<7}{r['peak']:>10.1f}{str(r['above_lo']):>8}{r['width_mm']:>8.2f}"
              f"{r['depth_mm']:>8.2f}{r['length_mm']:>9.2f}{r['area_top_mm2']:>11.4f}"
              f"{r['vn']:>9.4f}{r['E']:>12.5e}", flush=True)
    g3_move = {}
    for key in ("width_mm", "depth_mm", "area_top_mm2", "vn"):
        a, b = a3["tuned"][key], a3["hk"][key]
        g3_move[key] = (b - a) / (a or 1e-30) * 100.0
        print(f"    {key}: tuned→hk {g3_move[key]:+.2f}% ；tuned→off "
              f"{(a3['off'][key] - a) / (a or 1e-30) * 100:+.2f}%", flush=True)

    # ---------------- 处置 --------------------------------------------------
    print("\n[处置规则套判（跑前写死）]", flush=True)
    print(f"  G0={g0}  G1a={g1a}  G1c={g1c}  V1={v1}  G4a={g4a}  G4b={g4b}  G5a={pos_hit}",
          flush=True)
    print(f"  G2a：tuned={g2a['tuned']} hk={g2a['hk']} hk_surf={g2a['hk_surf']} off={g2a['off']}",
          flush=True)
    if not g0:
        verdict = "G0 不过 ⇒ 本轮读数不入档；src/tests 一字不动。"
    elif v1:
        verdict = ("V1 触发 ⇒ HK **不得**作默认替换（会移动 A3 宽度/面积读数）；"
                   "只能作高 T 专用分支并登记分支间不连续。")
    elif g2a["hk"] or g2a["hk_surf"]:
        which = "hk" if g2a["hk"] else "hk_surf"
        verdict = (f"判「**可采纳**」：arm={which} 让原守护在一字不动的前提下成立 ∧ G4a 闭合 ∧ V1 未触发 "
                   "⇒ 进实现轮（全量回归 + A3 外部靶核对 + 新增显式 evap_model 选择器，"
                   "因为 am_a3_fixture.json 要求封顶可置 0 才能对齐基准，而无参数公式没有置 0 旋钮）。")
    else:
        verdict = ("判「**换封顶不能解决**」：体相 HK、表面专用 HK 都动不了那 2% ⇒ "
                   "登记为需外部靶才能推进（现基准集内 A3＝无蒸发 ⇒ 无靶），"
                   "走 #29/#30 式等裁决路线；**禁止**调 c_evap 凑绿（N4）。")
    print("  " + verdict, flush=True)
    n_live = sum(1 for r in cov_rows if r[5] == "live-solver")
    n_unit = sum(1 for r in cov_rows if r[5] == "unit-only")
    n_miss = sum(1 for r in cov_rows if r[5].startswith("MISS"))
    n_moved = sum(1 for r in cov_rows if r[5].startswith("SITE-MOVED"))
    n_expect = sum(1 for r in cov_rows if r[7])
    print(f"  #33 侧：15 位点＝live-solver {n_live} / unit-only {n_unit} / MISS {n_miss} / "
          f"MOVED {n_moved}；期望场景命中率 {n_expect}/15"
          + (" ⇒ 分类可用" if pos_hit else " ⇒ G5 作废") , flush=True)
    print(f"  G3 侧：换封顶对 A3 代理读数的最大移动幅度 "
          f"{max(abs(v) for v in g3_move.values()):.2f}%（>1% ⇒ 与 #33 的宽度/面积靶**必须同批**）",
          flush=True)
    print(f"\nelapsed={time.perf_counter() - t_all:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
