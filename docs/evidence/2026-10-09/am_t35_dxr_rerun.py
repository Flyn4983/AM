#!/usr/bin/env python3
"""#35：`thermal_enthalpy.py` 的 **dx>r 欠分辨警告**里的跨 dx 对比数，在**当前树**上重测。

被核的文案（按**文本**定位，行号会过期）：
  注释：integrated 档「峰值低 15.6%（2125.6 vs 2518.0K）、熔宽低 46.7%（Ly 0.300 vs 0.563mm）、
        熔体积小 52.8%（0.0320 vs 0.0678mm³）」＋「point 源的同对比是 35.1% / 56.9%」；
  运行时 f-string（**用户可见**）：「实测 dx=r 相对 dx=r/8 档：熔宽偏低 46.7%、熔体积偏低 52.8%」。
出处＝`docs/evidence/2026-10-07/am_t2_a0_conv_rerun.log`（`TREE: head=af9fc89 dirty=30`，2026-10-07 21:05），
即 **#23 之后、#22 之前** ⇒ #22（光斑半径口径）/#24（整数取整单点化）之后必须重测才能继续引用。

跑前登记（不设"通过线"，只回答"这些数还成不成立"）：
  C1 四档 × 两模型，配方与读数定义**逐字继承** `am_t2_a0_conv_rerun.py`；
  C2 文案现值**由文件读取**（不手抄），判据＝`round(实测,1) == 文案值`（文案是一位小数，判据也是一位）；
  C3 正对照：光斑半径 ×1.05 再跑同两档 ⇒ Δpeak/ΔLy/Δvol 必须**全非零**（读数不动＝尺子失效，C2 作废）；
  C4 计数口径与 fv 加权口径**同时印**（防"两个口径混用还自洽"）；
  C5 运行时文本：真触发一次 `dx>r`（r=60µm、dx=100µm ⇒ 只落 `dx>r`、不落 `dx>2r`），把 `warnings`
     实际发出的整串按 repr 入库 ⇒ 换数改的是实测文本，不是我抄写的文本。
"""
import os
import subprocess
import sys
import time
import warnings as _w

# 本轮用户指令＝「继续，还是不动GPU」⇒ **在 import jax 之前**闸门：未钉 CPU 直接拒绝（否则
# `import jax`/`jax.devices()` 本身就会建 CUDA context，"只查一眼"也算占用）。
_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
if _cvd != "":
    raise SystemExit(f"REFUSED: CUDA_VISIBLE_DEVICES 实得 {_cvd!r}，必须是空串才允许启动本探针"
                     f"（不占 GPU 是用户指令，不是偏好）。")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

REPO = os.environ.get("AM_REPO", "/home/shy/桌面/2026年开发-显式冲击有限元-物质点法-sph-fvm+am/AM-Qoder")


def sh(*args):
    return subprocess.run(args, cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


envline = " ".join(f"{k}={os.environ.get(k, '(unset)')}"
                   for k in ("CUDA_VISIBLE_DEVICES", "JAX_ENABLE_X64", "PYTHONPATH"))
print(f"CMD: {envline} {sys.executable} {os.path.abspath(sys.argv[0])}", flush=True)
print(f"TREE: head={sh('git', 'rev-parse', '--short', 'HEAD')} "
      f"dirty={len(sh('git', 'status', '--porcelain', '--', 'src', 'tests').splitlines())}", flush=True)
print(f"DATE: {sh('date', '+%F %T %z')}", flush=True)
print(f"BACKEND: {jax.default_backend()} devices={[d.device_kind for d in jax.devices()]}", flush=True)

from amforge import geometry as G
from amforge.core.contracts import ProcessPlan, solid_mask, solid_weight
from amforge.materials import get_material
from amforge.thermal_enthalpy import solve_enthalpy_thermal, suggest_n_steps

TL = float(get_material("316L").T_liquidus)
BX, BY, BZ = 1.2e-3, 0.6e-3, 0.4e-3
R0 = 100e-6

# ---- 文案现值：从 dx>r 那一段源码里**读**出来（不手抄；分块＋唯一性自证） ----
src_txt = open(os.path.join(REPO, "src", "amforge", "thermal_enthalpy.py"), encoding="utf-8").read()
ANCHOR = "if r_c is not None and dx_c is not None and dx_c > r_c:"
assert src_txt.count(ANCHOR) == 1, f"锚点数={src_txt.count(ANCHOR)}"
i0 = src_txt.index(ANCHOR)
# ⚠ 块里另有"旧表写的是 45%/36%"一类**历史值**，若整段取会撞唯一性（干跑实测：熔宽低 出现 2 次）
# ⇒ 注释块只取到 ⚠ 之前；运行时文本单独取 `warnings.warn(` 那一次调用。
mark = src_txt.index("⚠ 2026-10-06 旧表", i0)
BLOCK_C = src_txt[i0:mark]
w_start = src_txt.index("warnings.warn(", mark)
BLOCK_W = src_txt[w_start:src_txt.index("stacklevel=2)", w_start) + len("stacklevel=2)")]
HIST = src_txt[mark:src_txt.index("能量守恒不受影响", mark)]
NEEDLES = [(BLOCK_C, "峰值低 "), (BLOCK_C, "熔宽低 "), (BLOCK_C, "熔体积小 "),
           (BLOCK_C, "point 源的同对比是 "), (BLOCK_W, "熔宽偏低 "), (BLOCK_W, "熔体积偏低 ")]
for blk, needle in NEEDLES:
    assert blk.count(needle) == 1, f"{needle!r} 出现 {blk.count(needle)} 次（块选错）"
assert HIST.count("熔宽低 ") == 1, "历史值块未成形"


def grab(blk, needle, extra_split=None):
    t = blk[blk.index(needle) + len(needle):]
    if extra_split:
        t = t.split(extra_split[0])[extra_split[1]]
    return float(t.split("%")[0])


SHIPPED = {"int_peak": grab(BLOCK_C, "峰值低 "), "int_width": grab(BLOCK_C, "熔宽低 "),
           "int_vol": grab(BLOCK_C, "熔体积小 "),
           "pt_width": grab(BLOCK_C, "point 源的同对比是 ", (" / ", 0)),
           "pt_vol": grab(BLOCK_C, "point 源的同对比是 ", (" / ", 1)),
           "warn_width": grab(BLOCK_W, "熔宽偏低 "), "warn_vol": grab(BLOCK_W, "熔体积偏低 ")}
hist_width = grab(HIST, "熔宽低 ")
print(f"SHIPPED（从源码现取，非手抄）: {SHIPPED}", flush=True)
print(f"文本读取的正对照：现取注释值={SHIPPED['int_width']}  同文件历史值(⚠ 旧表)={hist_width} "
      f"⇒ {'两者不同 ⇒ 读取器分得出现值与史值' if hist_width != SHIPPED['int_width'] else '两者相同 ⇒ 读取器无分辨力'}",
      flush=True)
print(f"运行时文本与注释文本的对比数是否同源：warn_width={SHIPPED['warn_width']} vs int_width={SHIPPED['int_width']} "
      f"warn_vol={SHIPPED['warn_vol']} vs int_vol={SHIPPED['int_vol']}", flush=True)


def coupon(dx, r=R0):
    return G.from_sdf_fn(
        lambda x: jnp.maximum(jnp.maximum(jnp.abs(x[..., 0]) - BX / 2,
                                          jnp.abs(x[..., 1]) - BY / 2),
                              jnp.abs(x[..., 2]) - BZ / 2),
        bounds=[(-BX / 2, BX / 2), (-BY / 2, BZ / 2), (-BZ / 2, BZ / 2)],
        spacing=dx, name="c")


def plan(r=R0):
    return ProcessPlan.uniform(1, modality="SLM", laser_power=600.0, scan_speed=0.8,
                               layer_thickness=BZ, hatch_spacing=1.4 * r, beam_radius=r,
                               absorption=0.45, preheat_temp=400.0)


def extent(axis_coords, sel):
    """与 am_a0_conv.py:16-17 逐字同定义（initial=0.0 一并继承，保证与旧表可比）。"""
    return float(jnp.max(axis_coords, where=sel, initial=0.0)
                 - jnp.min(axis_coords, where=sel, initial=0.0)) * 1e3


def run(model, dx, r=R0, n_steps=None):
    g, p = coupon(dx, r), plan(r)
    ns = int(n_steps if n_steps is not None else suggest_n_steps(g, p))
    t0 = time.perf_counter()
    th = solve_enthalpy_thermal(geometry=g, process=p,
                                params=dict(material="316L", source_model=model, n_steps=ns))
    el = time.perf_counter() - t0
    pk, c = th.peak_temperature, g.coords()
    mt = pk > TL
    sel = mt & (solid_mask(g.sdf) > 0.5)
    fv = solid_weight(g.sdf, dx)
    n_new = int(jnp.sum(sel))
    return dict(ns=ns, el=el, peak=float(jnp.max(pk)), n_new=n_new,
                xor=int(jnp.sum(sel ^ (mt & (g.sdf < 0.0)))),
                Lx=extent(c[..., 0], sel), Ly=extent(c[..., 1], sel),
                vol=n_new * dx ** 3 * 1e9,
                vol_fv=float(jnp.sum(jnp.where(mt, fv, 0.0))) * dx ** 3 * 1e9)


rows = {}
for model in ("integrated", "point"):
    for dx in (100e-6, 50e-6, 25e-6, 12.5e-6):
        rec = rows[(model, dx * 1e6)] = run(model, dx)
        print(f"[{model:10s}] dx={dx*1e6:6.1f} ns={rec['ns']:5d} {rec['el']:6.2f}s "
              f"peak={rec['peak']:8.1f} nml={rec['n_new']:6d} xor={rec['xor']:5d} "
              f"Lx={rec['Lx']:6.3f} Ly={rec['Ly']:6.3f} vol={rec['vol']:8.4f}mm³ "
              f"vol_fv={rec['vol_fv']:8.4f}mm³", flush=True)

print("\n=== C2/C4：dx=r(100µm) 相对 dx=r/8(12.5µm) 的偏低比 vs 文案现值 ===", flush=True)
verdicts = {}
for model, keys in (("integrated", ("int_peak", "int_width", "int_vol")),
                    ("point", ("pt_width", "pt_vol"))):
    coarse, fine = rows[(model, 100.)], rows[(model, 12.5)]
    got = {"peak": (1 - coarse["peak"] / fine["peak"]) * 100,
           "width": (1 - coarse["Ly"] / fine["Ly"]) * 100,
           "vol": (1 - coarse["vol"] / fine["vol"]) * 100,
           "vol_fv": (1 - coarse["vol_fv"] / fine["vol_fv"]) * 100}
    print(f"  [{model:10s}] peak 偏低 {got['peak']:6.2f}%  Ly 偏低 {got['width']:6.2f}%  "
          f"计数 vol 偏低 {got['vol']:6.2f}%  fv 加权 vol 偏低 {got['vol_fv']:6.2f}%", flush=True)
    print(f"    绝对值 peak {coarse['peak']:.1f} -> {fine['peak']:.1f}K   "
          f"Ly {coarse['Ly']:.3f} -> {fine['Ly']:.3f}mm   "
          f"vol {coarse['vol']:.4f} -> {fine['vol']:.4f}mm³   "
          f"vol_fv {coarse['vol_fv']:.4f} -> {fine['vol_fv']:.4f}mm³", flush=True)
    for tag in keys:
        which = "peak" if tag.endswith("peak") else ("width" if "width" in tag else "vol")
        shipped, meas = SHIPPED[tag], got[which]
        same = round(meas, 1) == shipped
        verdicts[tag] = (shipped, meas, same)
        print(f"    {tag:10s} 文案={shipped:6.2f}  实测={meas:6.2f}  "
              f"round(实测,1)==文案 -> {'仍成立' if same else '已过期'}", flush=True)

print("\n=== C3：正对照（光斑半径 ×1.05，同 ns 再跑两档 ⇒ 三个读数必须全变） ===", flush=True)
nz = tot = 0
for key in (("integrated", 100.), ("integrated", 12.5)):
    base = rows[key]
    pert = run("integrated", key[1] * 1e-6, r=R0 * 1.05, n_steps=base["ns"])
    d = {k: pert[k] - base[k] for k in ("peak", "Ly", "vol")}
    tot += len(d)
    nz += sum(1 for v in d.values() if abs(v) > 0)
    print(f"  dx={key[1]:6.1f} (ns={base['ns']})  Δpeak={d['peak']:+8.1f}K  ΔLy={d['Ly']:+6.3f}mm  "
          f"Δvol={d['vol']:+7.4f}mm³", flush=True)
print(f"  判读：非零变化 {nz}/{tot}（必须＝{tot}，否则 C2 的偏低比读数对本探针不敏感 ⇒ 结论作废）", flush=True)

print("\n=== C5：运行时警告文本（真触发一次 dx>r：r=60µm、dx=100µm、ns=40） ===", flush=True)
print("  ⚠ 首版这里写 ns=2 ⇒ 被求解器的稳定上限守卫**拒绝**（`params['n_steps']=2 不足：…至少需要 "
      "n_steps>=34`）：那条守卫是对的，错的是我图省事给的小步数。缺陷现形方式＝stderr  traceback "
      "而 C5 段落空转（正文只剩标题）⇒ 归档件必须回读 stderr，不能只看 stdout 与 rc。", flush=True)
with _w.catch_warnings(record=True) as caught:
    _w.simplefilter("always")
    run("integrated", 100e-6, r=60e-6, n_steps=40)
print(f"  捕获警告条数 = {len(caught)}", flush=True)
for x in caught:
    print(f"  category={x.category.__name__}")
    print(f"  TEXT repr: {repr(str(x.message))}", flush=True)

print("\n=== C6：与旧件（`am_t2_a0_conv_rerun.log`，其 TREE 自报 head=af9fc89 dirty=30）逐档对照 ===",
      flush=True)
import re

old_lines = open(os.path.join(REPO, "docs/evidence/2026-10-07/am_t2_a0_conv_rerun.log"),
                 encoding="utf-8").read().splitlines()
pat = re.compile(r"\[(\w+)\s*\] dx=\s*([\d.]+) ns=\s*(\d+)\s+[\d.]+s peak=\s*([\d.]+) "
                 r"nml_new=\s*(\d+).*?Ly=\s*([\d.]+) vol=\s*([\d.]+)mm")
old = {}
for ln in old_lines:
    m = pat.search(ln)
    if m:
        old[(m.group(1), float(m.group(2)))] = dict(ns=int(m.group(3)), peak=float(m.group(4)),
                                                    nml=int(m.group(5)), Ly=float(m.group(6)),
                                                    vol=float(m.group(7)))
print(f"  旧件解析出 {len(old)}/8 行（必须＝8，否则下面的对照表是空转）", flush=True)
n_same = 0
for key in sorted(rows, key=lambda k: (k[0], -k[1])):
    o = old.get(key)
    if o is None:
        print(f"  {key} 旧件无对应行 ⇒ 跳过（不猜）")
        continue
    n = rows[key]
    print(f"  [{key[0]:10s} dx={key[1]:6.1f}] ns {o['ns']:5d} -> {n['ns']:5d}   "
          f"peak {o['peak']:8.1f} -> {n['peak']:8.1f}K ({(n['peak']/o['peak']-1)*100:+6.2f}%)   "
          f"Ly {o['Ly']:6.3f} -> {n['Ly']:6.3f}mm ({(n['Ly']/o['Ly']-1)*100:+6.2f}%)   "
          f"vol {o['vol']:8.4f} -> {n['vol']:8.4f}mm³ ({(n['vol']/o['vol']-1)*100:+7.2f}%)   "
          f"nml {o['nml']:6d} -> {n['n_new']:6d}", flush=True)
    n_same += sum(1 for a, b in ((o['ns'], n['ns']), (o['peak'], n['peak'])) if a == b)
print(f"  判读：旧/新**完全相同**的 (ns, peak) 项数＝{n_same}/{2*len(old)}（若＝全部 ⇒ 本轮读数与旧件"
      f"无差 ⇒ C2 的『过期』结论不成立；实测应≠全部）", flush=True)
print("  归因边界：旧件自报 dirty=30 ⇒ 那次读数**不对应任何提交**，本轮差值只作『基线不可复现』记录，"
      "不指认由 #19/#22/#23/#24 中哪一笔造成（候选集＝af9fc89..HEAD 触及 src 的提交，见开发日志）。",
      flush=True)

allok = all(v[2] for v in verdicts.values()) and nz == tot
print(f"\nR35_VERDICT = {'文案全部仍成立' if all(v[2] for v in verdicts.values()) else '有文案已过期（见上面『已过期』行）'}"
      f"；尺子敏感={nz == tot}；整体 = {'PASS（文案无需改）' if allok else '需换数/需复核尺子'}")
