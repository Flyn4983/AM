#!/usr/bin/env python3
"""#22/T4 附带量：**改口径会把可观测量移动多少**（两副树各跑一次，同种子同夹具）。

为什么要有这份脚本：#22 把面内衰减从 1/e 改成 1/e²（σ ÷√2、面积 ÷2），这不是"重构
不动数"的那种改动——同一份工艺参数现在代表**另一束光**，熔深/熔宽/峰值必然动。台账要求
"照实记录移动量"，而 K6 只给了 σ_xy/σ_z 的**几何**读数，没给**解场**读数。

判据（跑前登记，`compare` 模式对两副 JSON 判）：
  M1 缺省档（thermal.enthalpy 的 `integrated`，即 A03 路径）改前/改后**逐位相同**
     ——本轮没动它，若这里动了就是越界改动（与探针 K4 的文件级门互为补充：K4 看
     "改了哪些文件"，这里看"没该改的文件是否真的没动数"）。
  M2 `meltpool.fdm`（A05 路径，**默认**熔池求解器）的峰值温度**上升**——束宽 ÷√2 而
     面内积分因子 π r²→2πσ²（÷2），同一 `ΣQ·dV=ηP` 下**峰值强度 ×2** ⇒ 表面更热。
     ⚠ 跑前我在这里还登记了另一半预测「熔深**变浅**」，**实测把它推翻了**（改前
     1.876464e-4 m → 改后 2.341599e-4 m，**+24.8% 更深**；n_melt 276→360）。机制：
     面内收窄只动了 σ_xy，**轴向尺度 dp=1.2·rb 一个数都没动**（出域 ⇒ #33）⇒ 表面
     温度抬高后沿同一深度尺度往**下**多导。所以"变浅"的预测错在把"束窄"当成"能量
     变少"——功率是守恒的（探针 K2 实测 0.999026960）。这条**反证**连同长宽比读数
     （1.697056→2.400000）一起构成 **#33 的量化前提**，故 M2 现在只判"峰值上升 ∧
     移动量非零"，方向照实打印，不再预判。
  M3 移动量必须是**可测的有限数**（非 NaN/Inf），并打印相对差供 §26.25 引用。

用法：
  python am_t4_observable_move.py pre              # 只测量、打印 JSON
  python am_t4_observable_move.py post             # 只测量、打印 JSON
  python am_t4_observable_move.py compare <pre.log> <post.log>   # 判 M0/M1/M2/M3

不做的事：不比吞吐（CPU 钉住 ⇒ 本脚本不含任何性能数字）；不动 A0 断言、不动阈值、
不为本轮的物理移动去补任何"凑回旧数"的旋钮。
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

PHASE = (sys.argv[1] if len(sys.argv) > 1 else "post").strip().lower()
assert PHASE in ("pre", "post", "compare"), f"phase 只能是 pre/post/compare，收到 {PHASE!r}"
REPO = Path(__file__).resolve().parents[3]

from amforge import geometry as G                                     # noqa: E402
from amforge.core.contracts import ProcessPlan, solid_mask             # noqa: E402
from amforge.materials import get_material                            # noqa: E402
from amforge.meltpool import solve_meltpool_fdm                       # noqa: E402
from amforge.thermal_enthalpy import solve_enthalpy_thermal           # noqa: E402

SELF = Path(__file__).read_bytes()
print("== T4/#22 可观测移动量探针 ==")
print(f"phase   = {PHASE}")
print(f"REPO    = {REPO}")
print(f"devices = {jax.devices()}")
print(f"本脚本 md5 = {hashlib.md5(SELF).hexdigest()}")

if PHASE == "compare":
    # ---- 跨树判据：把 pre/post 两份 JSON 并排判 M0/M1/M2/M3 ----
    assert len(sys.argv) == 4, "compare 需要两个日志路径：<pre.log> <post.log>"
    self_md5 = hashlib.md5(SELF).hexdigest()

    def load(tag: str, path: str) -> tuple[dict, str]:
        txt = Path(path).read_text(encoding="utf-8", errors="replace")
        line = [l for l in txt.splitlines() if l.startswith("JSON {")]
        assert len(line) == 1, f"{tag} 里 JSON 行数={len(line)}（应当恰好 1 行）"
        mine = [l for l in txt.splitlines() if "本脚本 md5" in l]
        assert len(mine) == 1, f"{tag} 的 md5 行={mine}"
        return json.loads(line[0][5:]), mine[0].split("=")[-1].strip()

    pre, md5_pre = load("pre", sys.argv[2])
    post, md5_post = load("post", sys.argv[3])
    bad = []
    ok_m0 = md5_pre == md5_post == self_md5
    print(f"\nM0 三份 md5 相同（同一把尺子量两副树）                        "
          f"{'OK' if ok_m0 else 'FAIL'}  pre={md5_pre} post={md5_post} self={self_md5}")
    if not ok_m0:
        bad.append("M0")

    def rel(a: float, b: float) -> float:
        return (b - a) / abs(a) if a else float("nan")

    print("\n-- M1 缺省 integrated 档（A03，本轮**不该**动）必须逐位相同 --")
    same = all(pre["enthalpy_integrated"][k] == post["enthalpy_integrated"][k]
               for k in pre["enthalpy_integrated"])
    for k in sorted(pre["enthalpy_integrated"]):
        a, b = pre["enthalpy_integrated"][k], post["enthalpy_integrated"][k]
        print(f"   {k:6s} pre={a:.17g}  post={b:.17g}  {'逐位相同' if a == b else '动了！'}")
    print(f"M1 {'OK' if same else 'FAIL'} 越界改动检测（默认热解档一个 bit 都不许动）")
    if not same:
        bad.append("M1")

    print("\n-- M2 被改路径的移动量（方向照实记 ⇒ 见 docstring 的预测被推翻）--")
    for grp, keys in (("fdm", ("peak", "depth", "n_melt", "sum_T")),
                      ("enthalpy_point", ("peak", "final"))):
        for k in keys:
            if k not in pre[grp]:
                continue
            a, b = float(pre[grp][k]), float(post[grp][k])
            print(f"   {grp}.{k:8s} {a:.10g} → {b:.10g}  Δ={rel(a, b) * 100:+.2f}%")
    peak_rel = rel(pre["fdm"]["peak"], post["fdm"]["peak"])
    m2 = peak_rel > 0.01 and abs(rel(pre["fdm"]["depth"], post["fdm"]["depth"])) > 0.01
    print(f"M2 {'OK' if m2 else 'FAIL'} fdm 峰值上升（实测 {peak_rel * 100:+.2f}%）"
          f"∧ 移动量非零（熔深方向**不预判**：实测 "
          f"{rel(pre['fdm']['depth'], post['fdm']['depth']) * 100:+.2f}%，与跑前预测的"
          f"「变浅」反号 ⇒ 记进 #33 前提）")
    if not m2:
        bad.append("M2")

    nums = [v for d in (pre, post) for g in d.values() for v in g.values()
            if isinstance(v, float)]
    m3 = all(math.isfinite(v) for v in nums)
    print(f"\nM3 {'OK' if m3 else 'FAIL'} 两副树共 {len(nums)} 个浮点读数全有限")
    if not m3:
        bad.append("M3")
    print(f"\n汇总（compare 判据 M0/M1/M2/M3）：{'PASS' if not bad else 'FAIL 失败=' + str(bad)}")
    sys.exit(0 if not bad else 1)
    print(f"\n判据汇总（compare）：{'PASS' if not bad else 'FAIL 失败=' + str(bad)}")
    sys.exit(0 if not bad else 1)


# 夹具＝`tests/test_default_solver_numerical.py::test_meltpool_fdm_runs_and_is_differentiable`
# 的现成 fixture（同一束光参数，两副树共用 ⇒ 差值只可能来自口径）。
BEAM_R = 50e-6
geo = G.from_sdf_fn(lambda x: jnp.linalg.norm(x, axis=-1) - 0.45e-3,
                    bounds=[(-0.6e-3, 0.6e-3)] * 3, spacing=50e-6,
                    name="t4_move_probe")
proc = ProcessPlan.uniform(laser_power=250.0, scan_speed=0.8, layer_thickness=40e-6,
                           hatch_spacing=80e-6, beam_radius=BEAM_R, absorption=0.4)
mat = get_material("316L")
dx = 50e-6
solid = (jnp.asarray(solid_mask(geo.sdf)) > 0.5).astype(jnp.float64)
n_solid = int(jnp.sum(solid))
print(f"夹具  R={BEAM_R:.3e} m  P={float(proc.laser_power):.1f} W  v={float(proc.scan_speed):.2f} m/s"
      f"  dx={dx:.1e} m  sdf.shape={jnp.asarray(geo.sdf).shape}  实体体素={n_solid}")

rows = {}

# ---- ① 默认熔池求解器 meltpool.fdm（A05 路径 ⇒ 本轮被改）----
# ⚠ fdm 自建 `n_grid`³ 网格 ⇒ 它的 T **不是** geo.sdf 的 25³ 栅格（实测 (24,24,24) vs
# (25,25,25)）。跨求解器比物理量之前先比 shape/原始计数（记忆铁律），故这里**不**乘
# `solid`，只报 fdm 自己栅格上的计数与形状。
out = solve_meltpool_fdm(geometry=geo, process=proc,
                         params={"material": "316L", "n_grid": 24, "n_steps": 80})
T = jnp.asarray(out.temperature)
melt = (T > mat.T_liquidus).astype(jnp.float64)
rows["fdm"] = dict(peak=float(jnp.max(T)), depth=float(out.depth),
                   lof=float(out.lof_indicator),
                   t_shape=str(tuple(T.shape)),
                   n_melt=int(jnp.sum(melt)),
                   sum_T=float(jnp.sum(T)))
print("\n-- ① meltpool.fdm（本轮改的就是它）--")
def _dump(d):
    for k, v in d.items():
        print(f"   {k:11s} = {v:.10g}" if isinstance(v, float) else f"   {k:11s} = {v}")

_dump(rows["fdm"])

# ---- ② thermal.enthalpy 缺省档 integrated（A03 路径 ⇒ 本轮**不该**动它）----
th_def = solve_enthalpy_thermal(geometry=geo, process=proc,
                                params={"material": "316L"})
rows["enthalpy_integrated"] = dict(peak=float(jnp.max(th_def.peak_temperature)),
                                   final=float(jnp.max(th_def.final_temperature)))
print("\n-- ② thermal.enthalpy 缺省 integrated 档（对照：不许动）--")
_dump(rows["enthalpy_integrated"])

# ---- ③ thermal.enthalpy 的 point 档（A01/A02/A04 路径 ⇒ 本轮被改）----
th_pt = solve_enthalpy_thermal(geometry=geo, process=proc,
                               params={"material": "316L", "source_model": "point"})
rows["enthalpy_point"] = dict(peak=float(jnp.max(th_pt.peak_temperature)),
                              final=float(jnp.max(th_pt.final_temperature)))
print("\n-- ③ thermal.enthalpy point 档（本轮改的）--")
_dump(rows["enthalpy_point"])

nums = [x for r in rows.values() for x in r.values() if isinstance(x, float)]
if not all(math.isfinite(x) for x in nums):
    print(f"\nM3 FAIL 有非有限读数（{len(nums)} 个浮点读数）")
    sys.exit(1)
print(f"\nM3 全部 {len(nums)} 个读数有限                                       OK")
print("（M1/M2 是**跨树**比对：把 pre 与 post 两份输出的同一行并排看，"
      "②必须逐位相同、①③必须移动 ⇒ 汇总见开发日志 §26.25）")
print("JSON " + json.dumps(rows, sort_keys=True))
