"""#27＋#33 代码腿**预检**（CPU 钉住，**只测不改**：src/ 与 tests/ 一字不动）。

选型轮（`am_t27_evap_sink_probe.py`，2026-10-08，tree=56ad25b）判「可采纳、进实现轮」。
但那是 **#18 落地之前**的树——本轮（2026-10-09）#18 已入库（tree=547dbf9），
`thermal_enthalpy.py` 被 #18 接线动了 5 处 ⇒ 行号漂移、默认档是否仍逐位不变**必须重量**。

预检＝把"实现轮据以动代码的前提"逐条钉成硬门；任一不过 ⇒ 停手、src/tests 一字不动、
如实登记选型轮读数已过期（不得据其实现）。

判据（跑前写死）
----------------
P1 位点漂移普查：15 个 #33 位点按**源码文本**在现树重解（复用选型探针的 SITES 元组与
   resolve_site，行号键不硬编）；命中数==1 才算 ok，并对照选型轮 log 里的旧行号打印漂移。
   全 15 位点必须仍 `ok`（若某位点 moved/ambig ⇒ 登记表过期，先修表再谈实现）。
P2 G0 锚点在当前树复现：跑 `child("tuned","conv")`（缺省经验封顶臂），要求
   peak0 与 peaks(h=20000) **逐位**等于选型轮锚点（锚点值**从归档件现取**、不手抄）。
   ⇒ 证明"红的是经验钳、不是夹具/守护前提"这一结论在 #18 后仍成立。
P3 封顶消费者普查：grep 树内 `evaporation_flux`/`recoil_pressure` 的调用点 ⇒ 必须**只有
   meltpool（FVM/VOF 链）**用 HK；`thermal_enthalpy` 仍是经验钳（这正是 #27 要补的缺口）。
P4 结构不变量（子进程 CPU 钉住，import 生产模块）：
   (a) `_evap_sink(T>T_lo, c=0) == 0`（置 0 旋钮真生效 ⇒ A3 无蒸发对齐路径可用）；
   (b) HK 体密度在正常熔化区中性：复用选型探针 `hk_sink(T_liquidus,dx)/Q_peak < 0.1%`
       （V1 复核，用选型探针同一实现与同一 Q_peak 口径 ⇒ 不新造尺子）。
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SEL_DIR = os.path.join(REPO, "docs", "evidence", "2026-10-08")
SEL_PY = os.path.join(SEL_DIR, "am_t27_evap_sink_probe.py")
SEL_LOG = os.path.join(SEL_DIR, "am_t27_evap_sink_probe.log")
REG_LOG = os.path.join(SEL_DIR, "am_t4_full_regression.log")


def load_selection_probe():
    spec = importlib.util.spec_from_file_location("t27_sel", SEL_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_anchor_from_log() -> tuple[float, float]:
    """从归档回归件的断言行现取锚点（X vs Y），不手抄。"""
    txt = open(REG_LOG, encoding="utf-8").read()
    m = re.search(r"(\d+\.\d+) vs (\d+\.\d+)", txt)
    if not m:
        raise SystemExit("P2 锚点无法从归档回归件解析 ⇒ 预检作废")
    return float(m.group(1)), float(m.group(2))  # (h20000, peak0)


def load_old_site_lines() -> dict[str, str]:
    """从选型轮 log 的 G5 表解析每个 site 当时解析到的位置（旧行号），供漂移对照。"""
    out: dict[str, str] = {}
    for ln in open(SEL_LOG, encoding="utf-8"):
        mm = re.match(r"\s{2}(S\d{2}-\w+)\s+([\w./]+:\d+)\s+(ok|moved|ambig\S*)", ln)
        if mm:
            out[mm.group(1)] = mm.group(2)
    return out


def gate(name: str, ok: bool, detail: str):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} — {detail}", flush=True)
    return ok


def main() -> int:
    t0 = time.perf_counter()
    print("== #27/#33 代码腿预检（CPU 钉住，只测不改）==", flush=True)
    print(f"tree HEAD: {subprocess.run(['git','-C',REPO,'rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()}", flush=True)
    print(f"device = '' (CUDA_VISIBLE_DEVICES 由每个子进程内部钉住)\n", flush=True)
    tp = load_selection_probe()
    results = {}

    # ---------------- P1：位点漂移普查 ----------------
    print("[P1] 15 位点按源码文本在现树重解（复用选型探针 SITES，不硬编行号）", flush=True)
    old_lines = load_old_site_lines()
    all_ok = True
    n_drift = 0
    for sid, rel, snip, want, fn in tp.SITES:
        ln, status = tp.resolve_site(rel, snip)
        old = old_lines.get(sid, "?")
        ok = status == "ok"
        all_ok = all_ok and ok
        moved = (status == "ok" and ln and f"{rel.split('/')[-1]}:{ln[0]}" != old)
        n_drift += 1 if moved else 0
        flag = "（行号漂移，文本指纹仍命中）" if moved else ""
        print(f"   {'ok' if ok else status:<6} {sid:<11} 现树 {rel.split('/')[-1]}:{'/'.join(map(str,ln)):<6}"
              f" 选型轮 {old:<24}{flag}", flush=True)
    results["P1"] = gate("P1 位点漂移普查：15 位点文本仍唯一命中", all_ok,
                         f"全 ok={all_ok}，行号漂移 {n_drift}/15（漂移无害，因判定按文本非行号）")

    # ---------------- P2：G0 锚点在当前树复现 ----------------
    print("\n[P2] 缺省经验封顶臂 child(\"tuned\",\"conv\") 在当前树复现 G0 锚点", flush=True)
    anchor_h, anchor_p0 = load_anchor_from_log()
    print(f"   锚点（从 {os.path.basename(REG_LOG)} 现取）：peak0={anchor_p0!r}  "
          f"h20000={anchor_h!r}", flush=True)
    d = tp.child("tuned", "conv")
    r0, rh = d["runs"][0], d["runs"][3]
    p0_ok = r0["peak"] == anchor_p0
    ph_ok = rh["peak"] == anchor_h
    mean_ok = abs(r0["mean_final_solid"] - tp.ANCHOR_MEAN0[0]) <= tp.ANCHOR_MEAN0[1]
    print(f"   peak0 实测 {r0['peak']!r} ⇒ {'逐位=' if p0_ok else '≠'} 锚点", flush=True)
    print(f"   h=20000 实测 {rh['peak']!r} ⇒ {'逐位=' if ph_ok else '≠'} 锚点", flush=True)
    print(f"   末态均值实测 {r0['mean_final_solid']:.4f} vs {tp.ANCHOR_MEAN0[0]}±{tp.ANCHOR_MEAN0[1]}", flush=True)
    print(f"   峰值落点 idx={r0['peak_idx']} fv={r0['argmax_fv']} 自由面={r0['argmax_surf']} "
          f"n_solid={d['n_solid']}", flush=True)
    results["P2"] = gate("P2 G0 锚点在 #18 后的当前树仍逐位复现（⇒ 选型结论对当前树有效）",
                         p0_ok and ph_ok and mean_ok,
                         f"peak0={p0_ok} h20000={ph_ok} mean={mean_ok}")

    # ---------------- P3：封顶消费者普查 ----------------
    print("\n[P3] HK 物理量消费者普查（grep 树内调用点）", flush=True)
    grep = subprocess.run(["grep", "-rn", "--include=*.py", "-E",
                           r"\.(evaporation_flux|recoil_pressure)\(", os.path.join(REPO, "src")],
                          capture_output=True, text=True).stdout.strip().splitlines()
    hits = []
    for g in grep:
        rel = g.replace(REPO + "/", "")
        fn = rel.split(":", 1)[0].split("/")[-1]
        hits.append((fn, rel))
    te_calls = [h for h in hits if h[0] == "thermal_enthalpy.py"]
    mp_calls = [h for h in hits if h[0] == "meltpool.py"]
    # materials.py 是这两个方法的**定义处**，其 evaporation_flux 内部调 recoil_pressure
    # 属定义内的自洽调用（不是"求解器消费者"）⇒ 单列，不计入越界。
    def_calls = [h for h in hits if h[0] == "materials.py"]
    other = [h for h in hits if h[0] not in
             ("thermal_enthalpy.py", "meltpool.py", "materials.py")]
    for fn, rel in hits:
        tag = "定义/自调用" if fn == "materials.py" else "消费者"
        print(f"   [{tag}] {rel}", flush=True)
    print(f"   汇总：meltpool 消费者={len(mp_calls)}  thermal_enthalpy 消费者={len(te_calls)}  "
          f"materials 定义处={len(def_calls)}  真·越界消费者={len(other)}", flush=True)
    results["P3"] = gate("P3 HK 消费者只在 meltpool；thermal_enthalpy 仍无物理封顶（缺口确认）",
                         len(te_calls) == 0 and len(mp_calls) >= 2 and len(other) == 0,
                         f"enthalpy 调用={len(te_calls)}（应 0） meltpool={len(mp_calls)}（应≥2） "
                         f"真越界={len(other)}（应 0；materials 定义处自调已单列）")

    # ---------------- P4：结构不变量（子进程 CPU 钉住） ----------------
    print("\n[P4] 结构不变量（子进程 import 生产模块）", flush=True)
    child_src = r'''
import os, json
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import amforge.thermal_enthalpy as TE
from amforge.materials import get_material
mat = get_material("316L")
TB = float(mat.T_boil); TLO = TB - 200.0; TLIQ = float(mat.T_liquidus)
# (a) 置 0 旋钮真生效：c=0 ⇒ 封顶恒 0（即便 T>>T_lo）
vals = [float(TE._evap_sink(jnp.asarray(T), TLO, TB, 0.0)) for T in (TLO+1, TB, TB+300, 4000.0)]
a_ok = all(v == 0.0 for v in vals)
print("P4a _evap_sink(c=0) @ T>T_lo 恒 0 =", a_ok, "值:", vals)
# (b) 正常熔化区 c=缺省 时经验封顶本就介入为 0（低温不干扰，#27 docstring 的命题）
below = [float(TE._evap_sink(jnp.asarray(T), TLO, TB, 3.0e11)) for T in (1700.0, 2000.0, TLIQ, TLO)]
b_ok = all(v == 0.0 for v in below)
print("P4b tuned 在 T<=T_lo 恒 0 =", b_ok, "值:", below)
print("JSON::" + json.dumps({"a_ok": a_ok, "b_ok": b_ok}))
'''
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "src"),
               CUDA_VISIBLE_DEVICES="", JAX_ENABLE_X64="1")
    pr = subprocess.run([sys.executable, "-c", child_src], env=env,
                        capture_output=True, text=True, cwd=REPO)
    print(pr.stdout, flush=True)
    if pr.returncode != 0:
        print("  子进程 stderr：\n" + pr.stderr[-800:], flush=True)
    pj = {}
    for line in pr.stdout.splitlines():
        if line.startswith("JSON::"):
            pj = json.loads(line[6:])
    p4a = bool(pj.get("a_ok"))
    p4b = bool(pj.get("b_ok"))
    # (b2) V1 复核：HK(T_liquidus)/Q_peak 用选型探针同一实现与同一 Q_peak
    qp = float(d["q_peak"])
    s_liq = tp.hk_sink(tp.LIT["T_liquidus"], tp.DX_CONV)
    v1_pct = s_liq / qp * 100.0
    v1_ok = v1_pct < tp.V1_BAR_PCT
    results["P4"] = gate("P4 结构不变量：置 0 生效 ∧ 低温不介入 ∧ HK 在熔化区中性（V1 复核）",
                         p4a and p4b and v1_ok,
                         f"c=0→0={p4a}  tuned低温0={p4b}  HK(T_liq)/Qp={v1_pct:.5f}%<{tp.V1_BAR_PCT}%={v1_ok}")

    # ---------------- 总结论 ----------------
    print("\n[预检套判]", flush=True)
    allok = all(results.values())
    if allok:
        print("  ⇒ 四项前提全部成立：**可以进入 #27 代码腿实现轮**（只加 evap_model 选择器＋无参数 HK 封顶，"
              "默认仍 tuned ⇒ 绝对数值逐位不变、规避半动口径）。默认切换与 #33 值重标定随 #10/A3 的 18 道外部靶进行。",
              flush=True)
    else:
        print("  ⇒ 有前提不过：停手，src/tests 一字不动，登记选型轮读数过期（不得据其实现）。", flush=True)
    print(f"\nelapsed={time.perf_counter()-t0:.0f}s  rc={0 if allok else 1}", flush=True)
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
