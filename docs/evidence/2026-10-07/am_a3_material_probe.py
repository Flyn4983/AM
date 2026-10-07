"""N2.5-③（#21）——A3 基准侧**物性口径核对**：我方 IN625 卡片 vs 基准论文 Table 1。

来源（本轮取到，一手）：Holla, Redford, Kopp, Kollmannsberger (2025) *The trace of heat:
on the predictive power of modeling transient diffusion*, Progress in Additive Manufacturing
10:4203–4215, DOI `10.1007/s40964-025-01147-9`；全文 PDF 由 NIST 公开托管
（`https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=959720`），本地留存
`_refs/benchmarks/nist_mds2-3662/holla2025_trace_of_heat_nist959720.pdf`
（5176709 B，15 页，sha256 前缀 `d139acb7c99efd26242f72d4a9254dbc`）。
**Table 1（p.4206）逐字摘录**（本脚本的"基准侧"全部来自这里，不许凭记忆填）：

    Density                ρ (kg/m³)      8440            [21]
    Solidus temperature    T_s (°C)       1290            [20]
    Liquidus temperature   T_l (°C)       1350            [20]
    Solid specific heat    c_s (J/kg°C)   410 + 276×10⁻³ T   [21]
    Liquid specific heat   c_l (J/kg°C)   766             assumed (c_s at T_l)
    Solid conductivity     k_s (W/m°C)    9.8 + 15.3×10⁻³ T  [21]
    Liquid conductivity    k_l (W/m°C)    29.5            assumed (k_s at T_l)
    Latent heat of fusion  L (J/kg)       2.8×10⁵         [20]
    Laser absorptivity     α              0.54            calibrated
    Regularization         S              1               assumed

**跑前登记的判据（先写死，再跑）**：
  ① 逐项对齐表每格给「我方 / 基准 / 相对差%」；我方**没有的口径要显式写"无此口径"**，不许留空；
  ② **单位口径自证**：断言我方 `T_solidus/T_liquidus` 按 **K** 解释后与基准同义（±1 K；°C 读法会差 273，
     故该阈值足以区分两种读法）。此条不过 ⇒ 整个对比作废（K/°C 混用会让所有差值都是假的）。
     ②b **实测偏移单独点名**：卡片若用 273 而非 273.15 偏移，会把相线整体压低 0.15 K ⇒ 单独报数。
     （第一版把容差写成 ±0.15，正好贴在实测量上，被 9.1e-14 的浮点噪声判成 FAIL ⇒ 属实现错，
     已按"阈值要能区分被排除的假设"重写；**这不是放宽门槛**：±1 K 仍能一票否决 K/°C 混淆。）
  ③ **我方卡片内部自洽**：断言我方常数 == 我方 `reference` 里所引拟合式在 [25,1290] °C 的区间均值
     （±0.5%）。不过则说明卡片是"随手填的数"，要先修卡片再谈基准；
  ④ 体热容 `ρ·c`（固/液分区）与吸收功率比 `α` **各自单独报数**，**不许合成**成单一"综合偏差"
     ——合成需要数值实验（铁律：解析折算不当作仿真）；
  ⑤ **来源歧义要点名**：基准 Table 1 的 c_l=766、k_l=29.5 按其固相式在 **T_s=1290 °C** 复算吻合
     （766.04 / 29.54），而标注写 "at T_l"（1350 °C 应给 782.6 / 30.46）⇒ 我方必须选定一个口径并记录；
  ⑥ 产物落盘（本脚本 + 日志 + `am_a3_fixture_spec.json`），#10 只引用产物。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MAT = ROOT / "src" / "amforge" / "materials.py"
OUT_JSON = HERE / "am_a3_fixture_spec.json"
fails: list[str] = []


def check(tag: str, ok: bool, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag} — {detail}", flush=True)
    if not ok:
        fails.append(tag)


# ---------------------------------------------------------------- 我方卡片（按文本解析，不 import ⇒ 不碰 JAX/GPU）
src = MAT.read_text(encoding="utf-8")
m = re.search(r'"IN625":\s*AMMaterial\((.*?)\n    \),', src, re.S)
if not m:
    raise SystemExit("FAIL：materials.py 里找不到 IN625 块（结构变了？先复核解析器，别改判据）")
block = m.group(1)
# 先取 reference 串（内部有括号与逗号，单独处理），再取标量
ref_m = re.search(r'reference="(.*?)",\n', block, re.S)
if not ref_m:
    raise SystemExit("FAIL：IN625 块里没有 reference 串（卡片结构变了？先复核解析器，别改判据）")
reference = " ".join(x.strip() for x in ref_m.group(1).splitlines() if x.strip()).replace('"', "")
# 一行可放多个字段（`T_solidus=1563.0, T_liquidus=1623.0, ...`）⇒ 先摘掉 reference 串（里面有
# `cp=405+…` 会被当成字段），再对剩余正文做全局 key=value 扫描。
body = re.sub(r'reference="(.*?)",\n', "\n", block, flags=re.S) if ref_m else block
nums: dict[str, float] = {}
for k, v in re.findall(r"\b(\w+)=(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", body):
    nums[k] = float(v)
need = ["T_solidus", "T_liquidus", "rho_solid", "rho_liquid", "cp_solid", "cp_liquid",
        "k_solid", "k_liquid", "latent_fusion", "absorptivity"]
missing = [k for k in need if k not in nums]
check("① 我方卡片字段齐备（缺项要显式登记，不许静默）", not missing,
      f"取到 {len(nums)} 个标量字段；缺 = {missing or '无'}")
if missing:
    # 解析失败就到此为止：后面每条都会拿 KeyError 冒充"判据结果"
    print(f"\n=== 记分 ===\n  判据未达项 = {fails}\n"
          "  ⇒ 解析器未取到字段 ⇒ 先修解析器，**不改判据**（§26.6-8 的处置三步）", flush=True)
    raise SystemExit(1)
print(f"    reference 原文 = {reference[:150]}…", flush=True)

# 我方 reference 自述的拟合式（从卡片文本里读出来，而不是我抄一遍）
fit = re.search(r"cp=([0-9.]+)·?\+?([0-9.]+)·\(T\[°C\]/1000\)", reference)
fitk = re.search(r"k=([0-9.]+)\+([0-9.]+)·\(T\[°C\]/1000\)", reference)
check("① 我方 reference 里的拟合式可被解析", bool(fit and fitk),
      f"cp 式 {fit.groups() if fit else None}；k 式 {fitk.groups() if fitk else None}")
cp_a, cp_b = (float(x) for x in fit.groups())
k_a, k_b = (float(x) for x in fitk.groups())

# ---------------------------------------------------------------- 基准侧（Table 1 摘录）
BEN = {
    "rho": 8440.0, "T_s_C": 1290.0, "T_l_C": 1350.0,
    "cs_a": 410.0, "cs_b": 276e-3,      # c_s = a + b·T[°C]
    "cl": 766.0, "ks_a": 9.8, "ks_b": 15.3e-3, "kl": 29.5,
    "L": 2.8e5, "alpha": 0.54, "S": 1.0,
    "spot_D4sigma_um": 85.0, "T0_C": 20.0,
    "domain_um": [3000.0, 2500.0, 1000.0],
    "bc": "顶面 Neumann αI；其余（左右前后底）**绝热 zero flux**；无对流、无辐射",
    "time": "激光命令 0.01 ms 采样（100 kHz）；熔池轮廓按 T=1290 °C 等值面，约 0.149 ms 一次",
    "scheme": "线性六面体 FEM + 层级加密；Crank–Nicolson 时间积分；Newton–Raphson 求解",
    "own_validation": "宽度平均偏差 14%、最大 20%；面积平均 11%、最大 22%（1–2 道可达 41%）",
}


def cs(T_C: float) -> float:
    return BEN["cs_a"] + BEN["cs_b"] * T_C


def ks(T_C: float) -> float:
    return BEN["ks_a"] + BEN["ks_b"] * T_C


def ours_cp(T_C: float) -> float:
    return cp_a + cp_b * (T_C / 1000.0)


def ours_k(T_C: float) -> float:
    return k_a + k_b * (T_C / 1000.0)


def mean_over(f, lo: float, hi: float) -> float:
    # 线性函数的区间均值 = 中点值（解析恒等，不是近似）
    return f((lo + hi) / 2.0)


print("\n=== 判据②：单位口径自证（我方温度是 K 还是 °C）===", flush=True)
ts_C = nums["T_solidus"] - 273.15
tl_C = nums["T_liquidus"] - 273.15
# 本判据的**目的**是排除"把 K 当 °C"这类致命混淆 ⇒ 阈值要按"能否区分两种读法"来定（1 K），
# 而不是贴着实测偏移（0.15 K）设——第一版把容差写成 ±0.15，于是 9.1e-14 的浮点噪声
# 就把一条**本来成立**的判断报成 FAIL。这属于实现错（阈值取在了被测量本身上），不是放宽门槛。
check("②a 按 **K** 解释后与基准同义（阈值 1 K；°C 读法会差 273 ⇒ 无歧义）",
      abs(ts_C - BEN["T_s_C"]) <= 1.0 and abs(tl_C - BEN["T_l_C"]) <= 1.0,
      f"{nums['T_solidus']:.1f} K → {ts_C:.2f} °C（基准 1290）；"
      f"{nums['T_liquidus']:.1f} K → {tl_C:.2f} °C（基准 1350）")
off_s = BEN["T_s_C"] - ts_C
off_l = BEN["T_l_C"] - tl_C
check("②b 实测偏移单独点名并给界（我方卡片用 **273** 偏移，非 273.15）",
      0.14 <= off_s <= 0.16 and abs(off_s - off_l) < 1e-9 and off_s <= 0.2,
      f"ΔT_s = ΔT_l = {off_s:.2f} K（{nums['T_solidus']:.1f} − 273 = "
      f"{nums['T_solidus'] - 273.0:.1f} °C 恰为整数）⇒ 我方相线比基准**低** {off_s:.2f} K，"
      f"占 T_solidus(K) 的 {off_s / nums['T_solidus'] * 100:.3f}%")

print("\n=== 判据③：我方卡片**内部**自洽（常数 == 自述拟合式的 25–1290 °C 区间均值）===", flush=True)
m_cp = mean_over(ours_cp, 25.0, 1290.0)
m_k = mean_over(ours_k, 25.0, 1290.0)
check("③ cp_solid == 自述拟合式区间均值", abs(nums["cp_solid"] - m_cp) / m_cp <= 0.005,
      f"我方 {nums['cp_solid']:.1f} vs 自述式均值 {m_cp:.1f}（差 {(nums['cp_solid']-m_cp)/m_cp*100:+.2f}%）")
check("③ k_solid == 自述拟合式区间均值", abs(nums["k_solid"] - m_k) / m_k <= 0.005,
      f"我方 {nums['k_solid']:.2f} vs 自述式均值 {m_k:.2f}（差 {(nums['k_solid']-m_k)/m_k*100:+.2f}%）")

print("\n=== 判据①＋④：逐项对齐（**各自单独报数，不合成**）===", flush=True)
rows = [
    ("ρ_solid kg/m³", nums["rho_solid"], BEN["rho"]),
    ("L 潜热 J/kg", nums["latent_fusion"], BEN["L"]),
    ("c_s 区间均值 J/kg°C", nums["cp_solid"], mean_over(cs, 25.0, 1290.0)),
    ("c_s @1290 °C J/kg°C", ours_cp(1290.0), cs(1290.0)),
    ("c_l J/kg°C", nums["cp_liquid"], BEN["cl"]),
    ("k_s 区间均值 W/m°C", nums["k_solid"], mean_over(ks, 25.0, 1290.0)),
    ("k_s @1290 °C W/m°C", ours_k(1290.0), ks(1290.0)),
    ("k_l W/m°C", nums["k_liquid"], BEN["kl"]),
    ("α 吸收率", nums["absorptivity"], BEN["alpha"]),
]
print(f"  {'量':<24}{'我方':>14}{'基准':>14}{'相对差%':>10}", flush=True)
for name, mine, theirs in rows:
    d = (mine - theirs) / theirs * 100.0
    print(f"  {name:<24}{mine:>14.4g}{theirs:>14.4g}{d:>+10.1f}", flush=True)
print("  —— 体热容 ρ·c（决定同样吸热下的温升；分区单独报）——", flush=True)
for tag, mine, theirs in (
        ("固相 ρ·c 区间均值", nums["rho_solid"] * nums["cp_solid"],
         BEN["rho"] * mean_over(cs, 25.0, 1290.0)),
        ("液相 ρ·c", nums["rho_liquid"] * nums["cp_liquid"], BEN["rho"] * BEN["cl"])):
    print(f"  {tag:<24}{mine:>14.4g}{theirs:>14.4g}{(mine-theirs)/theirs*100:>+10.1f}", flush=True)
print(f"  吸收功率比 我方/基准 = {nums['absorptivity']/BEN['alpha']:.4f}"
      f"（即我方吸热少 {(1-nums['absorptivity']/BEN['alpha'])*100:.1f}%）"
      f"；我方液相密度 {nums['rho_liquid']:.0f} vs 基准**恒定 ρ={BEN['rho']:.0f}**"
      f"（基准 Eq.1 明写 ρ 为常数）", flush=True)
check("④ 未合成综合偏差（只给分项；合成需数值实验）", True,
      "本表全部为**逐项**相对差；任何'净效应'结论须由 #10 的数值跑给出（铁律）")
print(f"  ⚠ 光斑口径（基准 D4σ = {BEN['spot_D4sigma_um']:.0f} µm）**我方尚未核对** ⇒ "
      f"登记在产物 JSON 的 `not_yet_checked`；未核对前不得称『同条件』。"
      f"（这条是**待办**，不是判据：写成判据就会永远 PASS）", flush=True)

print("\n=== 判据⑤：来源歧义点名（基准 Table 1 的 'at T_l' 标注与其数值不自洽）===", flush=True)
cl_at_ts, cl_at_tl = cs(BEN["T_s_C"]), cs(BEN["T_l_C"])
kl_at_ts, kl_at_tl = ks(BEN["T_s_C"]), ks(BEN["T_l_C"])
check("⑤ c_l=766 复算于 **T_s**（而非标注的 T_l）",
      abs(BEN["cl"] - cl_at_ts) < 0.5 and abs(BEN["cl"] - cl_at_tl) > 5.0,
      f"c_s(1290)={cl_at_ts:.2f} vs 表值 766；c_s(1350)={cl_at_tl:.2f} ⇒ 标注 'at T_l' 与数值不符")
check("⑤ k_l=29.5 复算于 **T_s**（而非标注的 T_l）",
      abs(BEN["kl"] - kl_at_ts) < 0.05 and abs(BEN["kl"] - kl_at_tl) > 0.5,
      f"k_s(1290)={kl_at_ts:.3f} vs 表值 29.5；k_s(1350)={kl_at_tl:.3f}")

print("\n=== 基准侧夹具（Table 1 + Fig.2 + §2.2 原文摘编，供 #10 直接引用）===", flush=True)
for k in ("domain_um", "bc", "time", "scheme", "own_validation"):
    print(f"  {k}: {BEN[k]}", flush=True)

spec = {
    "source": "Holla et al. 2025, Prog. Addit. Manuf. 10:4203-4215, DOI 10.1007/s40964-025-01147-9"
              "（NIST 公开托管 PDF，pub_id=959720；本地 sha256 前缀 d139acb7c99efd26242f72d4a9254dbc）"
              " + 数据集 README.txt（EOS M290 / 76×76×6 mm 实心 IN625 板 / 285 W / 960 mm/s /"
              " hatch 110 µm / skywriting 开）",
    "benchmark_table1": BEN,
    "ours_parsed_from_materials_py": nums,
    "ours_reference_string": reference,
    "ours_fitted_forms": {"cp_J_per_kgC": f"{cp_a}+{cp_b}*(T_C/1000)",
                          "k_W_per_mC": f"{k_a}+{k_b}*(T_C/1000)"},
    "unit_check": {"T_solidus_K": nums["T_solidus"], "as_C": ts_C,
                   "T_liquidus_K": nums["T_liquidus"], "as_C": tl_C,
                   "phase_line_offset_vs_benchmark_K": off_s,
                   "cause": "卡片用 273 偏移（1563−273=1290 恰为整数），非 273.15"},
    "per_item_relative_difference_pct": {name: (mine - theirs) / theirs * 100.0
                                         for name, mine, theirs in rows},
    "volumetric_heat_capacity_pct": {
        "solid_interval_mean": (nums["rho_solid"] * nums["cp_solid"])
        / (BEN["rho"] * mean_over(cs, 25.0, 1290.0)) * 100.0 - 100.0,
        "liquid": (nums["rho_liquid"] * nums["cp_liquid"]) / (BEN["rho"] * BEN["cl"]) * 100.0 - 100.0},
    "absorbed_power_ratio_ours_over_benchmark": nums["absorptivity"] / BEN["alpha"],
    "source_ambiguity": {
        "c_l_label_says": "assumed (c_s at T_l)", "c_l_value": BEN["cl"],
        "c_s_at_T_s": cl_at_ts, "c_s_at_T_l": cl_at_tl,
        "k_l_label_says": "assumed (k_s at T_l)", "k_l_value": BEN["kl"],
        "k_s_at_T_s": kl_at_ts, "k_s_at_T_l": kl_at_tl,
        "our_reading_convention": "取**数值**为准 ⇒ 液相物性 = 固相式在 **T_s=1290 °C** 之值"},
    "not_yet_checked": ["ours_beam（我方光斑/热流空间分布口径与基准 D4σ=85 µm 的对应关系）",
                        "我方默认链的对流/辐射边界如何**关闭**以匹配基准的绝热 BC（需跑，不许假设）"],
}
OUT_JSON.write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n=== 判据⑥：产物 ===\n  {OUT_JSON.name}（{OUT_JSON.stat().st_size} B）"
      f" ⇒ #10 只引用本产物，不重复读 PDF", flush=True)
# 产物要**被消费**才算落地：读回来核对键与条目，否则一个空 JSON 也能"落盘成功"。
back = json.loads(OUT_JSON.read_text(encoding="utf-8"))
check("⑥a 产物可读回，且判据①–⑤的结论都在里面",
      all(k in back for k in ("benchmark_table1", "ours_parsed_from_materials_py",
                              "per_item_relative_difference_pct", "volumetric_heat_capacity_pct",
                              "absorbed_power_ratio_ours_over_benchmark", "source_ambiguity",
                              "unit_check", "not_yet_checked"))
      and len(back["per_item_relative_difference_pct"]) == len(rows)
      and abs(back["unit_check"]["phase_line_offset_vs_benchmark_K"] - off_s) < 1e-12,
      f"键 {len(back)} 个；逐项差 {len(back['per_item_relative_difference_pct'])}/{len(rows)} 条；"
      f"待办 {len(back['not_yet_checked'])} 条（含'光斑'的条目 "
      f"{any('光斑' in s for s in back['not_yet_checked'])}）")

print("\n=== 记分 ===", flush=True)
print(f"  判据未达项 = {fails or '无'}", flush=True)
print("  ⇒ " + ("物性口径核对**闭合**（分项差与来源歧义均已量化并落盘）"
                if not fails else "本项未闭合，#21 继续阻塞 #10"), flush=True)
