#!/usr/bin/env python3
"""#35 换数**之后**的验收件：证明新文案 (a) 逐字等于实测、(b) 用户可见文本已无配方实测数、
(c) 检测器本身有分辨力（跑在**换数前的提交树文本**上必须变红）。

跑前登记的判据（不达标就不算换完）：
  P1 src 的 dx>r 注释段反向解析出的 20 个 token，与 `am_t35_dxr_rerun.log` 里的 token **逐字相同**；
  P2 用户可见的 `warnings.warn` 串里**没有** `%` 字符、没有 5 个已过期数、且含证据件路径；
  P3 真跑一次 dx>r（r=60µm、dx=100µm、ns=40）⇒ 捕获到的运行时整串满足 P2，且几何事实
     （dx/r/「1.2 个体素」）按 fmt 正确渲染；
  P4 正对照：同一套检测器跑在 `git show HEAD:src/…`（＝换数**前**的提交树）上 ⇒ P1/P2 必须**变红**
     （检测器抓不到旧文本＝它对新文本的"通过"没有信息量）。
本件不动求解器数值路径，只核文本；求解器行为由同轮的定向测试与全量回归面保证。
"""
import os
import re
import subprocess
import sys

_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
if _cvd != "":
    raise SystemExit(f"REFUSED: CUDA_VISIBLE_DEVICES 实得 {_cvd!r}，必须是空串才允许启动本件"
                     f"（用户指令「继续，还是不动GPU」）。")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import warnings as _w  # noqa: E402

jax.config.update("jax_enable_x64", True)

REPO = os.environ.get("AM_REPO", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
LOGF = os.path.join(REPO, "docs/evidence/2026-10-09/am_t35_dxr_rerun.log")
REL = "src/amforge/thermal_enthalpy.py"


def sh(*args):
    # 必须 strip：git 输出自带换行 ⇒ 不 strip 会把 "TREE: head=xxx dirty=N" 戳劈成两行，
    # 而该戳是**下游解析器**（如 /tmp/am_t35_edit.py 的 `^TREE: …`）的行首锚点。首版就漏了。
    return subprocess.run(args, cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


envline = " ".join(f"{k}={os.environ.get(k, '(unset)')}"
                   for k in ("CUDA_VISIBLE_DEVICES", "JAX_ENABLE_X64", "PYTHONPATH"))
print(f"CMD: {envline} {sys.executable} {os.path.abspath(sys.argv[0])}", flush=True)
print(f"TREE: head={sh('git', 'rev-parse', '--short', 'HEAD')} "
      f"dirty={len(sh('git', 'status', '--porcelain', '--', 'src', 'tests').splitlines())}", flush=True)
print(f"DATE: {sh('date', '+%F %T %z')}", flush=True)
print(f"BACKEND: {jax.default_backend()} devices={[d.device_kind for d in jax.devices()]}", flush=True)

STALE = ("15.6", "46.7", "52.8", "35.1", "56.9")
ANCHOR = "if r_c is not None and dx_c is not None and dx_c > r_c:"
RE_SRC = (r"{name}：峰值低([\d.]+)%（([\d.]+)vs([\d.]+)K）、熔宽低([\d.]+)%（Ly([\d.]+)vs"
          r"([\d.]+)mm）、熔体积小([\d.]+)%（([\d.]+)vs([\d.]+)mm³；fv加权口径([\d.]+)%）")
KEYS = ("ip", "ic_pk", "if_pk", "iw", "ic_ly", "if_ly", "iv", "ic_v", "if_v", "ifv")
# 日志侧取数式（`%` 运算符填模型名 ⇒ 字面百分号写 `%%`）
RE_C2 = (r"\[%-10s\] peak 偏低\s+([\d.]+)%%\s+Ly 偏低\s+([\d.]+)%%\s+计数 vol 偏低\s+([\d.]+)%%"
         r"\s+fv 加权 vol 偏低\s+([\d.]+)%%")
RE_ABS = (r"绝对值 peak ([\d.]+) -> ([\d.]+)K\s+Ly ([\d.]+) -> ([\d.]+)mm\s+vol ([\d.]+) -> "
          r"([\d.]+)mm³\s+vol_fv ([\d.]+) -> ([\d.]+)mm")


def tokens_from_log():
    t = open(LOGF, encoding="utf-8").read()
    out = {}
    for name in ("integrated", "point"):
        mc = re.search(RE_C2 % name, t)
        assert mc, f"日志缺 {name} 的 C2 偏低比行"
        ma = re.search(RE_ABS, t[mc.end():])
        assert ma, f"日志缺 {name} 的绝对值行"
        vals = (mc.group(1), ma.group(1), ma.group(2), mc.group(2), ma.group(3), ma.group(4),
                mc.group(3), ma.group(5), ma.group(6), mc.group(4))
        out[name] = dict(zip(KEYS, vals))
    return out


def segs(txt):
    i0 = txt.index(ANCHOR)
    blk = txt[i0:txt.index("\n    # 初值场", i0)]
    cur = blk[:blk.index("# ⚠")]
    run = blk[blk.index("warnings.warn("):]
    return cur, run


log_tk = tokens_from_log()
work = open(os.path.join(REPO, REL), encoding="utf-8").read()
head_txt = sh("git", "show", f"HEAD:{REL}")
bad = sh("git", "status", "--porcelain", "--", REL)
print(f"\n本件对照的两份文本：工作区（换数后，git 标记 {bad[:2] or '干净'}）与 HEAD:{REL}（换数前）",
      flush=True)

print("\n=== P1：工作区注释段反向解析 vs 日志 token（逐字） ===", flush=True)
cur_w, run_w = segs(work)
flat = re.sub(r"[\s#]+", "", cur_w)
p1 = 0
for name in ("integrated", "point"):
    m = re.findall(RE_SRC.format(name=name), flat)
    assert len(m) == 1, f"P1 {name} 命中 {len(m)} 次（应＝1）"
    for k, v in zip(KEYS, m[0]):
        eq = v == log_tk[name][k]
        p1 += eq
        if not eq:
            print(f"    ✗ {name}/{k} 文本={v} 日志={log_tk[name][k]}")
print(f"  P1 逐字相同字段数 = {p1}/20（必须＝20）", flush=True)
assert p1 == 20

print("\n=== P2：工作区用户可见 warn 串 ===", flush=True)
res = [s for s in STALE if s in run_w]
print(f"  warn 串 '%' 字符数 = {run_w.count('%')}（必须＝0）；残留过期数 = {res}（必须＝空）；"
      f"含证据件路径 = {'am_t35_dxr_rerun.log' in run_w}（必须 True）", flush=True)
assert run_w.count("%") == 0 and not res and "am_t35_dxr_rerun.log" in run_w

print("\n=== P4：正对照 —— 同一套检测器跑在 HEAD（换数前）文本上 ⇒ 必须变红 ===", flush=True)
cur_h, run_h = segs(head_txt)
flat_h = re.sub(r"[\s#]+", "", cur_h)
mh = re.findall(RE_SRC.format(name="integrated"), flat_h)
hit = [s for s in STALE if s in cur_h or s in run_h]
print(f"  HEAD 注释段反向解析命中数 = {len(mh)}（应＝0：旧文案结构不同 ⇒ 解析器不是万能通过）",
      flush=True)
print(f"  HEAD 文本里的已过期数 = {len(hit)}/5（必须＝5，否则本检测器看不见旧状态）", flush=True)
print(f"  HEAD warn 串 '%' 字符数 = {run_h.count('%')}（必须＞0 ⇒ 旧的用户可见文本确实嵌了配方百分比）",
      flush=True)
assert len(mh) == 0 and len(hit) == 5 and run_h.count("%") > 0

print("\n=== P3：真触发一次 dx>r（r=60µm、dx=100µm、ns=40）⇒ 看运行时实际发出的串 ===",
      flush=True)
from amforge import geometry as G  # noqa: E402
from amforge.core.contracts import ProcessPlan  # noqa: E402
from amforge.thermal_enthalpy import solve_enthalpy_thermal  # noqa: E402

BX, BY, BZ = 1.2e-3, 0.6e-3, 0.4e-3
g = G.from_sdf_fn(
    lambda x: jnp.maximum(jnp.maximum(jnp.abs(x[..., 0]) - BX / 2, jnp.abs(x[..., 1]) - BY / 2),
                          jnp.abs(x[..., 2]) - BZ / 2),
    bounds=[(-BX / 2, BX / 2), (-BY / 2, BY / 2), (-BZ / 2, BZ / 2)], spacing=100e-6, name="c")
p = ProcessPlan.uniform(1, modality="SLM", laser_power=600.0, scan_speed=0.8, layer_thickness=BZ,
                        hatch_spacing=1.4 * 60e-6, beam_radius=60e-6, absorption=0.45,
                        preheat_temp=400.0)
with _w.catch_warnings(record=True) as caught:
    _w.simplefilter("always")
    solve_enthalpy_thermal(geometry=g, process=p,
                           params=dict(material="316L", source_model="integrated", n_steps=40))
tgt = [x for x in caught if "大于光束半径" in str(x.message)]
for x in caught:
    print(f"  category={x.category.__name__}")
    print(f"  TEXT repr: {repr(str(x.message))}", flush=True)
assert len(tgt) == 1, f"目标警告命中 {len(tgt)} 条（应＝1）"
t = str(tgt[0].message)
print(f"\n  P3 判据：含 '%' = {'%' in t}（必须 False）｜含过期数 = {[s for s in STALE if s in t]}"
      f"（必须空）｜几何事实渲染 = dx=100.0µm 在串里 {'dx=100.0µm' in t}、r=60.0µm "
      f"{'r=60.0µm' in t}、1.2 个体素 {'1.2 个体素' in t}｜含证据件路径 "
      f"{'am_t35_dxr_rerun.log' in t}", flush=True)
assert "%" not in t and not [s for s in STALE if s in t]
assert "dx=100.0µm" in t and "r=60.0µm" in t and "1.2 个体素" in t
assert "am_t35_dxr_rerun.log" in t

print("\nPASSED：P1 20/20 逐字 ＋ P2 无配方实测数 ＋ P3 运行时同判 ＋ P4 检测器对换数前文本变红。",
      flush=True)
