#!/usr/bin/env python3
"""#21 剩余项：A3 基准**夹具配置** + 光斑/吸收率口径核对。

纯 stdlib：按**文本**解析 ``src/amforge/*.py``，不 ``import amforge`` ⇒ 不碰 JAX/GPU，
不打扰正在跑的验收面（``am_d0_fullsuite2.log``）。

跑前登记的判据（先写后跑，门槛只加严）
------------------------------------------------------------
① 高斯光斑口径普查**完备**：固定正则扫出 src 里所有"面内高斯衰减"式，逐处给出
   （文件:行、指数系数 c、半径字段名）。判据 = 命中数 == 手工枚举数，**且必须
   同时看到两种不同的 c**（只看到一种就 PASS 是假绿）。
② 由 ① 的 c 反解 D4σ=85 µm 在两种口径下各自应填的半径，断言二者相差 √2；
   夹具 JSON 必须**按求解器分开存**，不许留一个含糊的标量。
③ 归一化用**数值积分**核对（不抄注释）：meltpool 面源 ∫I dA == P；
   enthalpy 的 cell-integrated 份额 Σfrac == 1；并把"把 w 当 r 填"的后果量化。
④ 与基准 Eq.(1)-(4) 同条件的**开关清单**：我方默认链里每一项基准没有的物理都要
   给出可达开关的源码锚点；锚点取不到 ⇒ 记进 unreachable 并 FAIL，不许隐去。
⑤ α 的**路由**：求解器读 ``ProcessPlan.absorption``，材料卡只是构造种子 ⇒
   两档 α 是工艺级开关，不改材料卡（卡片注释禁止为凑基准改物性）。
⑥ 代价锚点：基准域 × 两种 dx 档的体素数 / 显式 dt / 30 ms 步数 / voxel-step 总量，
   与论文自报"1 道 14 min（2.6 GHz i7-10750H）"并排（N3⑤ 登记用）。
⑦ 产物落盘并**读回**核对（键齐、①条数一致、④可达项数 == 清单长度、
   半径键集合 == 由①普查导出的 (文件,口径) 对集合、且两个文件确实落在两种口径上）。

本轮跑中自查改掉的三处**探针自身**的错（判据门槛未放松）：③ 的 ∫I dA 原用梯形法，
1e-9 的门槛被 1.2e-9 的**求积误差**（不是归一化误差）判红 ⇒ 换 Simpson，门槛收到 1e-10；
③ 的份额和式把已是米制的 dx 又乘了一次 1e-6 ⇒ Σfrac 假成 1e-14；⑦ 原把"分键数"
写成"不同口径数"（2），而实际按 (文件,口径) 分键是 3 ⇒ 键集合改为由①普查导出。
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

# 手工枚举（判据①用它反证解析器没漏没多）
ENUM = {
    ("thermal_enthalpy.py", 268): 1.0,
    ("thermal_enthalpy.py", 271): 1.0,
    ("meltpool.py", 752): 2.0,
    ("meltpool.py", 877): 2.0,
    ("meltpool.py", 1547): 1.0,
}

def census() -> list[dict]:
    hits = []
    for f in sorted(SRC.glob("*.py")):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            m = re.search(r"jnp\.exp\(\s*-(?P<c>2\.0\s*\*?)?\s*(?P<arg>planar2|r2|\(\(xs)", line)
            if not m:
                continue
            coef = 2.0 if (m.group("c") or "").startswith("2.0") else 1.0
            hits.append({"site": f"{f.name}:{i}", "coef": coef, "line": line.strip()[:90]})
    return hits


print("== ① 面内高斯光斑口径普查 ==")
hits = census()
print(f"  命中 {len(hits)} 处：")
for h in hits:
    print(f"    {h['site']:<28} exp(-{h['coef']:g}·ρ²/r²)  |  {h['line']}")
got = {(h["site"].split(":")[0], int(h["site"].split(":")[1])): h["coef"] for h in hits}
check("① 命中集合 == 手工枚举", set(got) == set(ENUM),
      f"解析器多={sorted(set(got) - set(ENUM))} 少={sorted(set(ENUM) - set(got))}")
cs = sorted(set(got.values()))
check("① 两种口径同时被点名", cs == [1.0, 2.0], f"distinct coef = {cs}")

print("== ② D4σ=85 µm 反解到各口径的半径 ==")
# exp(-c ρ²/r²) 与 exp(-ρ²/(2σ²)) 等同 ⇒ r² = c·σ²·... 逐系数解：c/r² = 1/(2σ²) ⇒ r = sqrt(2c)·σ
r_of = {c: math.sqrt(2.0 * c) * SIGMA for c in cs}
for c in cs:
    print(f"  c={c:g} ⇒ 应填半径 r = sqrt(2c)·σ = {r_of[c]*1e6:.3f} µm")
ratio = r_of[2.0] / r_of[1.0]
check("② 两口径半径差 √2", abs(ratio - math.sqrt(2.0)) < 1e-12, f"ratio={ratio:.6f}")
check("② c=2 口径即 1/e² 半径 w=D4σ/2", abs(r_of[2.0] - D4 / 2.0) < 1e-15,
      f"{r_of[2.0]*1e6:.2f} µm")

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
check("⑦ 键齐", all(k in back for k in ("benchmark", "beam_radius_by_solver_m", "beam_census",
                                      "switches_to_match_benchmark", "cost_anchor", "open_items")))
check("⑦ 普查条数与①一致", len(back["beam_census"]) == len(hits), f"{len(back['beam_census'])}")
check("⑦ 开关可达数 == 清单长度",
      len(back["switches_to_match_benchmark"]) == len(SWITCHES) - len(unreachable))
pairs = {(h["site"].split(":")[0], h["coef"]) for h in hits}
check("⑦ 半径按 (文件,口径) 分键，键集合由①普查导出",
      set(back["beam_radius_by_solver_m"]) == set(by_path) and len(pairs) == len(by_path),
      f"{len(by_path)} 键 / {len(pairs)} 个 (文件,口径) 对，涉及文件 "
      f"{sorted({p[0] for p in pairs})}")
check("⑦ 两个文件的口径确实不同（同一 beam_radius 字段被解成两种半径）",
      len({c for (_f, c) in pairs}) == 2 and len({r_of[c] for c in r_of}) == 2,
      f"半径值 {sorted(round(v*1e6, 3) for v in r_of.values())} µm")

print()
if FAILS:
    print(f"⇒ {len(FAILS)} 条 FAIL：{FAILS}")
    sys.exit(1)
print("⇒ 全部判据 PASS")
