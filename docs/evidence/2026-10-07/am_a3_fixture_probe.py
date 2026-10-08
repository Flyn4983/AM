#!/usr/bin/env python3
"""#21 剩余项：A3 基准**夹具配置** + 光斑/吸收率口径核对。

纯 stdlib：按**文本**解析 ``src/amforge/*.py``，不 ``import amforge`` ⇒ 不碰 JAX/GPU，
不打扰正在跑的验收面（``am_d0_fullsuite2.log``）。

跑前登记的判据（先写后跑，门槛只加严）
------------------------------------------------------------
① 高斯光斑口径普查**完备**：固定正则扫出 ``src/amforge`` 里所有"面内高斯衰减"式，逐处给出
   （文件:行、指数系数 c、半径字段名）。判据 = 命中集合 == 手工枚举，**并带正对照**：
   解析器必须能同时认出 c=1 与 c=2 的**四种拼写**（直写 ``-2.0*``、σ 写法
   ``/(2.0*sig*sig)``、经 ``planar_decay(…)`` 单点出处、以及 c1 旧写法），
   且必须把非衰减行（``r2 = …`` 定义行）判为**不命中** ⇒ "现树只剩一种口径"不能靠
   解析器变瞎来达成。#22/T4 之前这条判据是"必须同时看到两种 c"（树的病）；#22 修完
   口径后它变成**过期判据**（正确实现反而过不了），本轮把它换成
   「现树 c 集合 == [2.0] ∧ 正对照仍在」——门槛内容变了，但**没有放松**（正对照是新加的牙）。
② 由系数反解 D4σ=85 µm 在两种口径下各自应填的半径（c 是**数学换算**，与树无关 ⇒ 两个
   值都留着做对照），断言二者相差 √2；夹具 JSON 的半径必须**按 (文件,口径) 分键**，
   不许留一个含糊的标量——#22 之后所有键都落在 c=2 ⇒ 键集合退化成"每个文件一个 42.5 µm"。
③ 归一化用**数值积分**核对（不抄注释）：meltpool 面源 ∫I dA == P；
   enthalpy 的 cell-integrated 份额 Σfrac == 1；并把"把 w 当 r 填"的后果量化。
④ 与基准 Eq.(1)-(4) 同条件的**开关清单**：我方默认链里每一项基准没有的物理都要
   给出可达开关的源码锚点；锚点取不到 ⇒ 记进 unreachable 并 FAIL，不许隐去。
⑤ α 的**路由**：求解器读 ``ProcessPlan.absorption``，材料卡只是构造种子 ⇒
   两档 α 是工艺级开关，不改材料卡（卡片注释禁止为凑基准改物性）。
⑥ 代价锚点：基准域 × 两种 dx 档的体素数 / 显式 dt / 30 ms 步数 / voxel-step 总量，
   与论文自报"1 道 14 min（2.6 GHz i7-10750H）"并排（N3⑤ 登记用）。
⑦ 产物落盘并**读回**核对（键齐、①条数一致、④可达项数 == 清单长度、
   半径键集合 == 由①普查导出的 (文件,口径) 对集合、且**全树只剩一种口径**：
   #22 修完后各文件的半径键必须都落在 c=2＝D4σ/2＝42.5 µm 上，若再出现两种口径
   就是口径分裂复发）。

本轮跑中自查改掉的三处**探针自身**的错（判据门槛未放松）：③ 的 ∫I dA 原用梯形法，
1e-9 的门槛被 1.2e-9 的**求积误差**（不是归一化误差）判红 ⇒ 换 Simpson，门槛收到 1e-10；
③ 的份额和式把已是米制的 dx 又乘了一次 1e-6 ⇒ Σfrac 假成 1e-14；⑦ 原把"分键数"
写成"不同口径数"（2），而实际按 (文件,口径) 分键是 3 ⇒ 键集合改为由①普查导出。

**2026-10-08（#22/T4 落地后）的第二轮更正**：② 的判据原样保留、①/⑦ 的判据换掉。原因
＝#22 把口径分裂治好之后，这颗探针**旧判据里写的正是那件病本身**（"必须同时看到两种 c"、
"两个文件口径确实不同"），于是它对着**修好**的树报 ① FAIL（ENUM 的 5 个行号全失效），
而 ⑦ 的"两种口径"却靠**解析器读不懂 σ 写法**（把 ``beam.py:32`` 的
``exp(-planar2 / (2.0*sig*sig))`` 认成 c=1）假绿——**假绿比假红危险**：它把夹具半径写成
30.052 µm，A3 一旦照填就重蹈 √2。改法＝①解析器认识全部四种拼写（直写/σ 写法/经
``planar_decay`` 调用/c1 旧写）＋非衰减行必须不命中；判据改为"现树只剩 c=2"，并配
**正对照**（把 meltpool 的 fdm 退回 c1 ⇒ ①②⑦ 三条同时红，实测见 §26.25）；
① 的站点比对改成 (站点,系数) 整张映射比对；夹具 JSON 新增 ``spot_convention`` 段，
把"半径＝D4σ/2、单点出处＝beam.py"写成数据而不是注释。
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src" / "amforge"
OUT = Path(__file__).with_name("am_a3_fixture.json")

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not ok:
        FAILS.append(name)


# ---------------------------------------------------------------- 基准侧（论文逐字）
BENCH = {
    "domain_um": (3000.0, 2500.0, 1000.0),      # Fig.2 p.4207
    "T0_C": 20.0,                                # Eq.(4) 段 p.4207
    "D4sigma_um": 85.0,                          # §2.2 p.4207
    "laser_power_W": 285.0,
    "scan_speed_mm_s": 960.0,
    "hatch_um": 110.0,
    "tracks": 18,
    "alpha_calibrated": 0.54,                    # Table 1，calibrated
    "alpha_ours": 0.39,                          # materials.py IN625 卡片
    "t_end_s": 0.030,                            # Fig.7 横轴（18 道发散扫描）
    "their_cost_min_per_track": 14.0,            # 脚注3 p.4211，2.6 GHz i7-10750H
    "bc": "顶面 Neumann alpha*u；左右前后底全绝热；无对流、无辐射、无蒸发",
}
D4 = BENCH["D4sigma_um"] * 1e-6
SIGMA = D4 / 4.0            # ISO 11146 二阶矩标准差

# 手工枚举（判据①用它反证解析器没漏没多）。**2026-10-08 重抄**：#22/T4 把 amforge 的面内
# 衰减统一到 ``beam.planar_decay``（σ 写法）之后，旧枚举的 5 个行号全部失效（import 行导致
# 整体平移＋写法改变），而旧解析器还把 ``exp(-planar2 / (2.0*sig*sig))`` 误读成 c=1
# ⇒ 把修好的单点出处又算成一种"新口径"，并把 ``thermal_enthalpy`` **整包漏掉**
# （它不再含字面 jnp.exp 衰减）。教训形态：**上游探针的口径判据会在修好之后反过来咬**——
# 判据里写着"病必须存在"的，病治好了它就红；先分清红在门上还是红在树上（同 #22 探针 N3/N6）。
ENUM = {
    ("beam.py", 32): 2.0,               # 单点出处本体（σ 写法）
    ("meltpool.py", 753): 2.0,          # VOF 表面强度（直写 -2.0*）
    ("meltpool.py", 878): 2.0,          # 送粉斑（直写 -2.0*）
    ("meltpool.py", 1553): 2.0,         # fdm 体积热源 ⇒ 经 planar_decay
    ("thermal_enthalpy.py", 301): 2.0,  # _moving_source 3D ⇒ 经 planar_decay
    ("thermal_enthalpy.py", 304): 2.0,  # _moving_source 2D ⇒ 经 planar_decay
}

# ①的正对照：解析器**必须**认得的全部拼写（含改前 c1）＋必须**不**命中的行。
# 没有这一条，"现树只剩 c=2"可能只是解析器瞎了（#22 探针 K0-S 的同一枚硬币）。
SPELLINGS = [
    ("c2 直写（diffmech/上游写法）",
     "q2d = (2.0 * absorbed) / pi_r2 * jnp.exp(-2.0 * rsq / r2)", 2.0),
    ("c2 σ 写法（beam.py 本体）",
     "return jnp.exp(-planar2 / (2.0 * sig * sig))", 2.0),
    ("c2 经单点出处调用",
     "decay = planar_decay(planar2, r) * \\", 2.0),
    ("c1 旧写法（#22 改前的默认，必须仍认得出）",
     "planar = jnp.exp(-planar2 / jnp.maximum(rb * rb, 1e-18))", 1.0),
    ("c1 直写半径平方（改前 point 档形状）",
     "q = absorbed / pi_r2 * jnp.exp(-r2 / (r * r))", 1.0),
    ("非衰减行（ρ² 定义 ⇒ 不许命中）",
     "r2 = (cc[..., 0] - cx) ** 2 + (cc[..., 1] - cy) ** 2", None),
    ("非衰减行（深度/经验尺度 ⇒ 不许命中）",
     "d_melt = 1.2 * rb * jnp.sqrt(jnp.maximum(ved_ / ved_ref, 1e-12))", None),
]

C2_DIRECT = re.compile(r"jnp\.exp\(\s*-\s*2\.0\s*\*")
C2_SIGMA = re.compile(r"jnp\.exp\(\s*-\s*(?:planar2|rsq|r2)\s*/\s*\(\s*2\.0\s*\*")
C2_VIA_BEAM = re.compile(r"(?<!def )\bplanar_decay\(")
C1_RESIDUE = re.compile(r"jnp\.exp\(\s*-\s*(?:planar2|rsq|r2|\(\(xs)[^)]*?\s*/\s*(?!2\.0)")


def classify(line: str) -> float | None:
    """把一行源码归到 c（``exp(−c·ρ²/r²)`` 的 c）；非衰减行 ⇒ None。"""
    s = line.strip()
    if not s or s.startswith(("def ", "class ", "@", "#", '"""')) or '"""' in s:
        return None
    if C2_SIGMA.search(s) or C2_DIRECT.search(s) or C2_VIA_BEAM.search(s):
        return 2.0
    return 1.0 if C1_RESIDUE.search(s) else None


def census() -> list[dict]:
    hits = []
    for f in sorted(SRC.glob("*.py")):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            coef = classify(line)
            if coef is None:
                continue
            hits.append({"site": f"{f.name}:{i}", "coef": coef, "line": line.strip()[:90]})
    return hits


print("== ① 面内高斯光斑口径普查 ==")
hits = census()
print(f"  命中 {len(hits)} 处：")
for h in hits:
    print(f"    {h['site']:<28} exp(-{h['coef']:g}·ρ²/r²)  |  {h['line']}")
got = {(h["site"].split(":")[0], int(h["site"].split(":")[1])): h["coef"] for h in hits}
# 比的是**整张映射**（站点 ∧ 系数），不只是站点集合：只比站点的话"某处退回 c1"这一条
# 要靠后面的口径检查才红，而①本身应当第一个红。
check("① 命中（站点,系数）== 手工枚举", got == ENUM,
      f"解析器多={sorted(set(got) - set(ENUM))} 少={sorted(set(ENUM) - set(got))} "
      f"系数不符={[(k, got[k], ENUM[k]) for k in set(got) & set(ENUM) if got[k] != ENUM[k]]}")
cs = sorted(set(got.values()))
check("① 现树只剩一种口径 c=2（#22/T4 的闭合断言）", cs == [2.0], f"distinct coef = {cs}")
bad_spell = [(nm, classify(src), want) for nm, src, want in SPELLINGS
             if classify(src) != want]
check("① 正对照：7 副拼写全部分对（含 c1 旧写法与非衰减行）", not bad_spell,
      f"错分={bad_spell}")

print("== ② D4σ=85 µm 反解到各口径的半径 ==")
# exp(-c ρ²/r²) 与 exp(-ρ²/(2σ²)) 等同 ⇒ r² = c·σ²·... 逐系数解：c/r² = 1/(2σ²) ⇒ r = sqrt(2c)·σ
# c 是**换算关系**，与树上现在剩哪种口径无关 ⇒ 两个值都留着（③的"误填后果"要用 c=1 做对照）。
r_of = {c: math.sqrt(2.0 * c) * SIGMA for c in (1.0, 2.0)}
for c in sorted(r_of):
    print(f"  c={c:g} ⇒ 应填半径 r = sqrt(2c)·σ = {r_of[c]*1e6:.3f} µm"
          f"{'  ← 现树用的就是这个' if c in cs else '  （历史口径，仅对照）'}")
ratio = r_of[2.0] / r_of[1.0]
check("② 两口径半径差 √2", abs(ratio - math.sqrt(2.0)) < 1e-12, f"ratio={ratio:.6f}")
check("② c=2 口径即 1/e² 半径 w=D4σ/2", abs(r_of[2.0] - D4 / 2.0) < 1e-15,
      f"{r_of[2.0]*1e6:.2f} µm")
check("② 现树唯一口径 c=2 ⇒ 夹具半径只有一个数",
      cs == [2.0] and len({round(r_of[c], 15) for c in cs}) == 1,
      f"半径 {sorted(round(r_of[c]*1e6, 3) for c in cs)} µm")

print("== ③ 归一化数值核对 ==")
P = BENCH["laser_power_W"]
w = r_of[2.0]
n = 200001                      # Simpson：200000 个等距子区间
rho_max = 12.0 * w
h_step = rho_max / (n - 1)


def _I(rho: float) -> float:
    return (2.0 * P / (math.pi * w * w)) * math.exp(-2.0 * rho * rho / (w * w)) * 2.0 * math.pi * rho


s = _I(0.0) + _I(rho_max)
for k in range(1, n - 1):
    s += (4.0 if k % 2 else 2.0) * _I(h_step * k)
s *= h_step / 3.0
rel = abs(s - P) / P
check("③ meltpool 面源 ∫I dA == P（Simpson）", rel < 1e-10,
      f"∫={s:.6f} W vs {P} W, 相对误差={rel:.2e}")


def frac_sum(r: float, dx: float, span: int = 12) -> float:
    """cell-integrated 份额：与 _cell_integrated_source 同式（s_in = r/√2）。"""
    sig = r / math.sqrt(2.0)
    tot = 0.0
    for i in range(-span, span + 1):
        c = i * dx
        tot += 0.5 * (math.erf((c + 0.5 * dx) / sig) - math.erf((c - 0.5 * dx) / sig))
    return tot


for dx_m, tag in ((r_of[1.0] / 2.0, "dx=r/2"), (r_of[1.0], "dx=r"), (2.0 * r_of[1.0], "dx=2r")):
    f1 = frac_sum(r_of[1.0], dx_m)
    print(f"  {tag:<8} dx={dx_m*1e6:.2f} µm ⇒ 单轴 Σfrac={f1:.12f} 三轴积={f1 ** 3:.12f}")
f_mid = frac_sum(r_of[1.0], r_of[1.0] / 2.0)
check("③ integrated 份额 Σfrac == 1（与 dx/r 无关）",
      abs(f_mid ** 3 - 1.0) < 1e-10, f"dx=r/2 处 {f_mid ** 3:.12f}")
area_factor = (r_of[2.0] / r_of[1.0]) ** 2
check("③ 误填后果量化", abs(area_factor - 2.0) < 1e-12,
      f"把 w={w*1e6:.1f}µm 当 c=1 口径的 r 填 ⇒ 光斑面积 ×{area_factor:.2f}、峰值强度 ×{1/area_factor:.3f}")

print("== ④ 与基准 Eq.(1)-(4) 同条件所需开关（每项要源码锚点） ==")
SWITCHES = [
    ("Newton 弱冷却（基准无）", "thermal_enthalpy.py", r'h_cool', 'params["thermal"]["h_cool"]=0'),
    ("蒸发/反冲散热封顶（基准无）", "thermal_enthalpy.py", r'evap_coeff', 'params["thermal"]["evap_coeff"]=0'),
    ("heat_scale 手工倍率（必须留 1.0）", "thermal_enthalpy.py", r'heat_scale', '保持缺省 1.0'),
    ("表面对流（基准无，FVM 链）", "meltpool.py", r'h_convection', 'MeltPoolConfig.h_convection=0'),
    ("辐射 Stefan-Boltzmann（基准无，FVM 链）", "meltpool.py", r'mat\.emissivity \* SIGMA_SB', 'AMMaterial.emissivity=0（仅对照时）'),
    ("Fresnel 增强（基准无，FVM 链）", "meltpool.py", r'fresnel_boost', 'MeltPoolConfig.fresnel_boost=0'),
    ("Beer-Lambert 遮挡（基准为面吸收）", "meltpool.py", r'laser_absorption_depth', 'MeltPoolConfig 置大 ⇒ 只顶部吸收'),
]
reached, unreachable = [], []
for label, fname, pat, knob in SWITCHES:
    txt = (SRC / fname).read_text(encoding="utf-8")
    m = re.search(pat, txt)
    if m:
        ln = txt[: m.start()].count("\n") + 1
        reached.append({"item": label, "anchor": f"{fname}:{ln}", "knob": knob})
        print(f"  {fname}:{ln:<5} {label:<34} ⇒ {knob}")
    else:
        unreachable.append(label)
        print(f"  !! 锚点缺失：{label}（{fname} 内找不到 {pat.pattern}）")
check("④ 每项开关都有可达锚点（缺失即 FAIL，不许隐去）", not unreachable,
      f"可达 {len(reached)}/{len(SWITCHES)}" + (f" 缺 {unreachable}" if unreachable else ""))

print("== ⑤ α 的路由 ==")
te = (SRC / "thermal_enthalpy.py").read_text(encoding="utf-8")
mp = (SRC / "meltpool.py").read_text(encoding="utf-8")
pr = (SRC / "process.py").read_text(encoding="utf-8")
a_te = re.search(r"eta\s*=\s*jnp\.mean\(.*process\.absorption.*\)", te)
a_mp = re.search(r"A0\s*=\s*process\.absorption", mp)
seed = re.findall(r"absorption\s*=\s*(?:jnp\.asarray\()?mat\.absorptivity", pr)
check("⑤ 两链的 α 都取自 ProcessPlan.absorption", bool(a_te) and bool(a_mp),
      f"thermal_enthalpy:{'Y' if a_te else 'N'} meltpool:{'Y' if a_mp else 'N'}")
check("⑤ 材料卡 absorptivity 只是构造种子（故 α 档位=工艺级开关）", len(seed) >= 2,
      f"process.py 内 mat.absorptivity→absorption 的种子点 {len(seed)} 处")

print("== ⑥ 代价锚点（N3⑤ 登记用） ==")
k_s = 19.86                       # 基准 c_s/k_s 式在 25-1290 °C 的区间均值（§26.12(b)）
rho_cp = 8440.0 * 591.5           # 基准 ρ 常数 × c_s 区间均值
alpha_d = k_s / rho_cp            # 热扩散率 m²/s
dx_lim = BENCH["domain_um"]
cost = []
for tag, dx_m in (("形态档 dx=r/2", r_of[1.0] / 2.0), ("strict 下界 dx=r", r_of[1.0])):
    vox = 1
    for d_um in dx_lim:
        vox *= max(1, int(round(d_um * 1e-6 / dx_m)))
    dt = dx_m ** 2 / (6.0 * alpha_d)
    steps = BENCH["t_end_s"] / dt
    vs = vox * steps
    cost.append({"tier": tag, "dx_um": dx_m * 1e6, "voxels": vox, "dt_s": dt,
                 "steps": steps, "voxel_steps": vs})
    print(f"  {tag}: dx={dx_m*1e6:.2f} µm → 体素 {vox:.3e}，dt={dt*1e6:.2f} µs，"
          f"{BENCH['t_end_s']*1e3:.0f} ms ⇒ {steps:.0f} 步，voxel-step {vs:.3e}")
theirs = BENCH["their_cost_min_per_track"] * BENCH["tracks"]
print(f"  论文自报（同基准、纯导热 FEM）：1 道 {BENCH['their_cost_min_per_track']:.0f} min"
      f"（2.6 GHz i7-10750H）⇒ 18 道 ≈ {theirs:.0f} min CPU")
check("⑥ 形态档 voxel-step 已量化并落盘（>0 且有限）",
      all(c["voxel_steps"] > 0 and math.isfinite(c["voxel_steps"]) for c in cost),
      f"形态档 {cost[0]['voxel_steps']:.3e}")

# 按 (文件, 口径系数) 分键 —— 键集合由①的普查**导出**，不手数
by_path = {f"{fn}|c={c:g}": r_of[c]
           for (fn, c) in sorted({(h["site"].split(":")[0], h["coef"]) for h in hits})}
fixture = {
    "benchmark": BENCH,
    "spot_convention": {
        "contract": "ProcessPlan.beam_radius ＝ **1/e² 半径** r_b（σ = r_b/2）",
        "single_source": "src/amforge/beam.py（spot_sigma / planar_decay / inplane_integral）",
        "beam_radius_m": r_of[2.0],
        "resolved_by": "#22/T4（2026-10-08）：改前 amforge 默认路径按 1/e 解 ⇒ 与 diffmech 差 √2",
        "evidence": "docs/evidence/2026-10-08/am_t4_spot_probe_{pre,post}.log",
    },
    "beam_radius_by_solver_m": by_path,
    "beam_census": [{"site": h["site"], "coef": h["coef"]} for h in hits],
    "normalization": {"meltpool_surface_integral_W": s, "target_W": P,
                      "rel_error": rel, "integrated_sum_frac": f_mid ** 3},
    "switches_to_match_benchmark": reached,
    "unreachable_switches": unreachable,
    "alpha_routing": {"solver_reads": "ProcessPlan.absorption",
                      "material_card_role": "construction seed only",
                      "tiers": {"reproduce": BENCH["alpha_calibrated"],
                                "predict": BENCH["alpha_ours"]}},
    "cost_anchor": {"thermal_diffusivity_m2_s": alpha_d, "tiers": cost,
                    "benchmark_their_min_per_track": BENCH["their_cost_min_per_track"]},
    "open_items": [
        "Eq.(2) 的 u 的空间分布在论文里未给显式公式 ⇒ 只能按 D4σ 口径填，"
        "并把『核内是否含 1/cos 入射角修正』留作对照档差异",
        "开关等效性要用数值跑证明（N3⑧），本探针只登记锚点",
        "#18 的 (x,y,power,gating) 折线驱动仍未实现 ⇒ 夹具暂不可跑",
    ],
}
OUT.write_text(json.dumps(fixture, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"== ⑦ 产物读回核对 {OUT.name} ({OUT.stat().st_size} B) ==")
back = json.loads(OUT.read_text(encoding="utf-8"))
check("⑦ 键齐", all(k in back for k in ("benchmark", "spot_convention",
                                      "beam_radius_by_solver_m", "beam_census",
                                      "switches_to_match_benchmark", "cost_anchor", "open_items")),
      f"键={sorted(back)}")
check("⑦ 普查条数与①一致", len(back["beam_census"]) == len(hits), f"{len(back['beam_census'])}")
check("⑦ 开关可达数 == 清单长度",
      len(back["switches_to_match_benchmark"]) == len(SWITCHES) - len(unreachable))
pairs = {(h["site"].split(":")[0], h["coef"]) for h in hits}
check("⑦ 半径按 (文件,口径) 分键，键集合由①普查导出",
      set(back["beam_radius_by_solver_m"]) == set(by_path) and len(pairs) == len(by_path),
      f"{len(by_path)} 键 / {len(pairs)} 个 (文件,口径) 对，涉及文件 "
      f"{sorted({p[0] for p in pairs})}")
check("⑦ 单一口径：夹具半径每个文件都＝D4σ/2，两种口径并存的病已闭合（#22）",
      len({c for (_f, c) in pairs}) == 1 and
      all(abs(v - D4 / 2.0) < 1e-15 for v in back["beam_radius_by_solver_m"].values()) and
      abs(back["spot_convention"]["beam_radius_m"] - D4 / 2.0) < 1e-15,
      f"键 {sorted(back['beam_radius_by_solver_m'])} ⇒ 半径 "
      f"{sorted(round(v*1e6, 3) for v in back['beam_radius_by_solver_m'].values())} µm")

print()
if FAILS:
    print(f"⇒ {len(FAILS)} 条 FAIL：{FAILS}")
    sys.exit(1)
print("⇒ 全部判据 PASS")
