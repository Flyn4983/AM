#!/usr/bin/env python
"""#34/#36 LSF 粉源取证探针（**只测不改**：src/ 与 tests/ 一字不动）。

跑前写死的判据
--------------
K0  仪器自检（正对照必须能失败）
 K0a 同类口径（守卫可命中）下 Σ s > 0 ⇒ 夹具本身有信息量，于是"生产口径返回 0"
     是关于 import 的结论，而不是关于夹具的结论。
 K0b 同一调用重复两遍 ⇒ 逐位相同（确定性）。
K1  #36 核心
 K1a 生产口径（顶层 `diffmech…` 的 LSFConfig）⇒ Σ s == 0.0 且非零个数 == 0。
 K1b 两个 LSFConfig **不是同一个类对象**（`is` 为假）。
 K1c `_step_lsf_common` 以桩函数注入 ⇒ 生产口径下 mass/温度/activation_time
     三者逐位不变；正对照＝同类口径下质量确实增加（否则这条断言抓不到东西）。
K2  #34 的"归一因子差 2 ⇒ 数值 no-op"必须被**测过**，不许只靠代数
 K2a 本地重写式（系数照抄生产）在 6 副夹具上逐位复现生产 ⇒ 比对仪器有效。
 K2b 两处 prefactor 由 1/(πR²) 改为解析正确值 2/(πR²) ⇒ 与生产**逐位相同** ⇒ no-op 成立。
     任何一位不同 ⇒ 判 no-op 不成立，#34 的定性要改写。
 K2c 正对照：集粉环半径 1.8→1.5 ⇒ 必须**不同**（否则 K2b 的"相同"无意义）。
 K2d AST 计数：`pi_R2` 在函数体内出现次数 == 赋值次数 == 1 ⇒ 死变量；
     对照 `ring_R2` 必须被读用（出现次数 > 赋值次数）。
K3  质量记账（docstring 声称 total ṁ over all particles equals mdot_total）
 K3a 现序（先归一、后按环截断）⇒ 实测 Σ s / ṁ_total 的亏缺；> 1e-6 ⇒ 不变量被**实质**
     违反，实现轮须改截断顺序；≤ 1e-6 ⇒ 仅登记措辞问题，不改代码。
 K3c 参考序（先截断、后归一）⇒ |Σ s − ṁ_total|/ṁ_total < 1e-12。
K4  文档与代码口径差：docstring 写"≈1.5 × beam_radius"，代码用 1.8。
K5  server.py 的 `src.` **值**导入：两副口径的 am_api 是否同一模块对象；
     `_PROCESS_BOUNDS` 是否逐键同值（当前行为无害＝值相同，但模块级状态被复制）。
K6  回归网边界（先量网，再量 bug）
 K6a pyproject `testpaths` 实得；`tests/` 与 `tests_diffmech/` 各自收集数。
 K6b 全树 `src.` 前缀导入行数（含 tests_diffmech、examples）。
 K6c 收集网内引用 lsf 粉源的测试文件数（期望 0）。

处置规则（跑前写死）
--------------------
D1 K1a ∧ K1c ⇒ #36 判"真 bug、且在产品链上生效"，实现轮一次改完 3 处 import 口径。
D2 K2a ∧ K2b ∧ K2c ∧ K2d ⇒ #34 判"因子差确为 no-op"，实现轮只改**假注释＋死变量**，
   不碰任何系数。
D3 K3a 的数决定是否把 #34 扩到"截断顺序"（阈值 1e-6，跑前写死）。
D4 修复 #36 会让该路径**首次真正执行** ⇒ #34 的任何系数改动都必须与 #36 同批落地，
   否则等于把一条从未跑过的路径半改。
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = Path(__file__).resolve().parents[3]      # docs/evidence/2026-10-08/x.py ⇒ 仓库根
SRC = REPO / "src"


def _child_env():
    e = dict(os.environ)
    e["CUDA_VISIBLE_DEVICES"] = ""      # 用户 2026-10-08 指示：先不要占用 GPU
    e["JAX_ENABLE_X64"] = "1"
    e["PYTHONPATH"] = str(SRC) + os.pathsep + e.get("PYTHONPATH", "")
    return e


def child(label: str, code: str, timeout: int = 1800) -> dict:
    """在子进程里跑一段代码，回传它打印的最后一行 JSON（隔离两副 import 口径）。"""
    p = subprocess.run([sys.executable, "-c", textwrap.dedent(code)],
                       capture_output=True, text=True, cwd=str(REPO),
                       env=_child_env(), timeout=timeout)
    if p.returncode != 0:
        print(f"    [{label}] FAILED rc={p.returncode}\n{p.stderr[-1500:]}", flush=True)
        return {"__rc": p.returncode, "__err": p.stderr[-400:]}
    try:
        out = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception as exc:
        print(f"    [{label}] PARSE-ERR {exc}\n{p.stdout[-800:]}", flush=True)
        return {"__rc": 0, "__parse_err": repr(exc)[:200]}
    out["__rc"] = 0
    return out


CHILD = r"""
import json
import numpy as np
import jax.numpy as jnp
from diffmech.methods.am.process import LSFConfig as LSF_PLAIN      # 顶层口径（生产用）
from src.diffmech.methods.am.process import LSFConfig as LSF_SRC    # src. 口径（守卫认这个）
import diffmech.methods.am.particle_am as PA
from diffmech.methods.am.particle_am import lsf_powder_deposition_profile, _laser_xy_at
from diffmech.methods.am import ParticleAMState

GUARD = LSF_SRC      # 本地重写式必须与生产守卫认同一副口径，否则 K2 空转

CLOUDS = {
    "A2x2":   (2, np.array([[0.0, 0.0], [5e-5, 0.0]])),
    "grid17": (2, np.stack(np.meshgrid(np.linspace(-6e-3, 6e-3, 17),
                                      np.linspace(-6e-3, 6e-3, 17),
                                      indexing="ij")).reshape(2, -1).T),
    "line25": (2, np.stack([np.linspace(-4e-3, 4e-3, 25), np.zeros(25)], axis=1)),
    "g3d11":  (3, np.stack(np.meshgrid(np.linspace(-4e-3, 4e-3, 11),
                                      np.linspace(-4e-3, 4e-3, 11),
                                      np.linspace(-1e-3, 1e-3, 3),
                                      indexing="ij")).reshape(3, -1).T),
    "tight":  (2, np.stack(np.meshgrid(np.linspace(-1e-3, 1e-3, 9),
                                      np.linspace(-1e-3, 1e-3, 9),
                                      indexing="ij")).reshape(2, -1).T),
    "wide":   (2, np.stack(np.meshgrid(np.linspace(-2e-2, 2e-2, 21),
                                      np.linspace(-2e-2, 2e-2, 21),
                                      indexing="ij")).reshape(2, -1).T),
}

lay = type("L", (), {})()
lay.waypoints = np.array([[0.0, 0.0], [1.0e-3, 0.0]])
lay.cum_length = np.array([0.0, 1.0e-3])
lay.total_length = 1.0e-3
lay.layer_z = 0.0
paths = type("P", (), {})()
paths.layers = [lay]
STARTS = jnp.asarray([0.0])
DURS = jnp.asarray([2.0e-3])
T = 1.0e-3          # 层内正中 ⇒ 激光在 (5e-4, 0)


def _rsq(pos, laser_xy, z_active, dim):
    if dim == 2:
        return (pos[:, 0] - laser_xy[0]) ** 2 + (pos[:, 1] - z_active) ** 2
    return ((pos[:, 0] - laser_xy[0]) ** 2 + (pos[:, 1] - laser_xy[1]) ** 2
            + (pos[:, 2] - z_active) ** 2)


def local_variant(pos, t, paths, starts, durs, cfg, *, dim=3,
                  pf=1.0, ring=1.8, trunc_first=False):
    # 本地重写式：pf＝prefactor 分子（1=照抄生产，2=解析正确值）；ring＝集粉环系数
    if not isinstance(cfg, GUARD):
        return jnp.zeros(pos.shape[0])
    laser_xy, z_active, _ = _laser_xy_at(paths, starts, durs, t)
    r2 = cfg.beam_radius ** 2
    mdot_total = cfg.powder_feed_rate * cfg.deposition_efficiency
    rsq = _rsq(pos, laser_xy, z_active, dim)
    ring_R2 = (ring * cfg.beam_radius) ** 2
    beam_intensity = jnp.exp(-2.0 * rsq / r2)
    catchment = (pf / (jnp.pi * ring_R2)) * jnp.exp(-2.0 * rsq / ring_R2)
    shape = catchment * (0.2 + 0.8 * beam_intensity)
    if trunc_first:
        shape = jnp.where(rsq > 4.0 * ring_R2, 0.0, shape)
        discrete_sum = jnp.sum(shape)
        norm = jnp.where(discrete_sum > 1e-18, 1.0 / discrete_sum, 0.0)
        return mdot_total * norm * shape
    discrete_sum = jnp.sum(shape)
    norm = jnp.where(discrete_sum > 1e-18, 1.0 / discrete_sum, 0.0)
    s = mdot_total * norm * shape
    return jnp.where(rsq > 4.0 * ring_R2, 0.0, s)


def hexv(fn, key, cfg, **kw):
    # 以 16 进制逐位取回（保值的扰动比对，不用 :g 格式化）
    dim, pos = CLOUDS[key]
    v = np.asarray(fn(jnp.asarray(pos), T, paths, STARTS, DURS, cfg, dim=dim, **kw))
    return [float(x).hex() for x in v]


def stats(fn, key, cfg, **kw):
    dim, pos = CLOUDS[key]
    v = np.asarray(fn(jnp.asarray(pos), T, paths, STARTS, DURS, cfg, dim=dim, **kw))
    return float(v.sum()), int((v > 0).sum()), int(v.shape[0])


cfg_plain, cfg_src = LSF_PLAIN(), LSF_SRC()
mdot = cfg_plain.powder_feed_rate * cfg_plain.deposition_efficiency
out = {"mdot_total": mdot,
       "same_class": bool(LSF_PLAIN is LSF_SRC),
       "plain_module": LSF_PLAIN.__module__, "src_module": LSF_SRC.__module__,
       "clouds": list(CLOUDS), "rings": {"doc": 1.5, "code": 1.8}}

for k in CLOUDS:
    out.setdefault("plain", {})[k] = stats(lsf_powder_deposition_profile, k, cfg_plain)
    out.setdefault("srccfg", {})[k] = stats(lsf_powder_deposition_profile, k, cfg_src)
    a = stats(lsf_powder_deposition_profile, k, cfg_src)
    b = stats(lsf_powder_deposition_profile, k, cfg_src)
    out.setdefault("det", {})[k] = bool(a[0] == b[0] and a[1] == b[1] and a[2] == b[2])
    prod = hexv(lsf_powder_deposition_profile, k, cfg_src)
    ident = hexv(local_variant, k, cfg_src)
    dbl = hexv(local_variant, k, cfg_src, pf=2.0)
    rg = hexv(local_variant, k, cfg_src, ring=1.5)
    ref = hexv(local_variant, k, cfg_src, trunc_first=True)
    out.setdefault("k2a", {})[k] = bool(prod == ident)
    out.setdefault("k2b", {})[k] = bool(prod == dbl)
    out.setdefault("k2c", {})[k] = bool(prod != rg)
    nz_prod = int((np.array([float.fromhex(x) for x in prod]) > 0).sum())
    nz_ref = int((np.array([float.fromhex(x) for x in ref]) > 0).sum())
    s_now = sum(float.fromhex(x) for x in prod)
    s_ref = sum(float.fromhex(x) for x in ref)
    out.setdefault("k3", {})[k] = {
        "sum_now": s_now,
        "sum_ref": s_ref,
        "loss_rel": 1.0 - s_now / mdot,
        "ref_closure_rel": abs(s_ref - mdot) / mdot,
        "nz_now": nz_prod, "nz_ref": nz_ref,
        "max_r_mm": float(np.max(np.hypot(CLOUDS[k][1][:, 0], CLOUDS[k][1][:, 1])) * 1e3),
    }

# ---- K1c：产品链后果（桩函数隔离 DEM/MPM/SPH 机械步） ----------------------
pos = CLOUDS["tight"][1]
am = ParticleAMState(position=jnp.asarray(pos),
                     activation_time=jnp.full(pos.shape[0], 1e9),
                     temperature=jnp.zeros(pos.shape[0]),
                     mass=jnp.full(pos.shape[0], 1e-6),
                     layer_id=jnp.zeros(pos.shape[0], dtype=jnp.int32),
                     rho=jnp.full(pos.shape[0], 8000.0))


def stub_step(solver_state, am_state, dt, t, solver_cfg, cfg,
              paths, layer_start_times, layer_durations, *, walls, cp, alpha_T, tau):
    return solver_state, am_state


for name, cfg in (("plain", cfg_plain), ("src", cfg_src)):
    _, am2 = PA._step_lsf_common(stub_step, None, am, 1e-4, T, None, cfg,
                                 paths, STARTS, DURS, dim=2, walls=None,
                                 cp=500.0, alpha_T=1e-5, tau=1e-3)
    m0 = [float(x).hex() for x in np.asarray(am.mass)]
    m1 = [float(x).hex() for x in np.asarray(am2.mass)]
    out.setdefault("k1c", {})[name] = {
        "dmass_bits": bool(m0 == m1),
        "dmass_sum": float(np.asarray(am2.mass).sum() - np.asarray(am.mass).sum()),
        "dT_sum": float(np.asarray(am2.temperature).sum()),
        "activation_moved": bool(np.any(np.asarray(am2.activation_time)
                                        < np.asarray(am.activation_time))),
    }

print(json.dumps(out))
"""


def main() -> int:
    assert SRC.is_dir() and (REPO / "pyproject.toml").is_file(), f"仓库根推导错误：{REPO}"
    print("== #34/#36 LSF 粉源取证探针（**只测不改**：src/ 与 tests/ 一字不动）==", flush=True)
    print(f"   REPO＝{REPO}", flush=True)
    print("   device 钉 CPU（CUDA_VISIBLE_DEVICES=''，用户 2026-10-08 指示「先不要占用GPU」）",
          flush=True)

    # ---------------- K6：先量网，再量 bug ----------------------------------
    print("\n[K6] 回归网边界与全树普查（实测，不靠措辞）", flush=True)
    pyproj = (REPO / "pyproject.toml").read_text()
    tp = [l.strip() for l in pyproj.splitlines() if "testpaths" in l]
    print(f"  K6a pyproject testpaths 行＝{tp}", flush=True)
    for d in ("tests", "tests_diffmech"):
        r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", d],
                           capture_output=True, text=True, cwd=str(REPO),
                           env=_child_env(), timeout=900)
        tail = [l for l in r.stdout.splitlines() if "collected" in l or "error" in l]
        print(f"      {d}: {tail[-1].strip() if tail else 'n/a'}", flush=True)
    g = subprocess.run(["grep", "-rn", "-E",
                        r"^\s*(from src\.|import src\.|from src import)",
                        "src", "tests", "tests_diffmech", "examples", "--include=*.py"],
                       capture_output=True, text=True, cwd=str(REPO)).stdout.strip()
    lines = g.splitlines() if g else []
    print(f"  K6b 全树 `src.` 前缀导入（含 tests_diffmech/examples）＝{len(lines)} 行", flush=True)
    for l in lines:
        print(f"      {l}", flush=True)
    files_in_net = sorted(set(l.split(":")[0] for l in lines))
    print(f"      涉及文件＝{files_in_net}", flush=True)
    h1 = subprocess.run(["grep", "-rln", "-e", "lsf_powder_deposition_profile",
                         "-e", "_step_lsf_common", "-e", "step_lsf_am", "tests"],
                        capture_output=True, text=True, cwd=str(REPO)).stdout.strip()
    h2 = subprocess.run(["grep", "-rln", "-e", "lsf_powder_deposition_profile",
                         "-e", "_step_lsf_common", "-e", "step_lsf_am", "tests_diffmech"],
                        capture_output=True, text=True, cwd=str(REPO)).stdout.strip()
    n_in_net = len(h1.splitlines()) if h1 else 0
    n_out_net = len(h2.splitlines()) if h2 else 0
    print(f"  K6c 收集网（tests/）内引用 lsf 粉源的测试文件数＝{n_in_net}"
          f"；tests_diffmech（网外）＝{n_out_net} ⇒ "
          f"{'**网内零覆盖**（此路径的既有断言全在回归网之外）' if n_in_net == 0 else '网内有覆盖'}",
          flush=True)

    # ---------------- 子进程实测 --------------------------------------------
    print("\n[K0/K1/K2/K3/K1c] 子进程实测（两副 import 口径 × 6 副点云）", flush=True)
    r = child("lsf", CHILD)
    if r.get("__rc") != 0 or "clouds" not in r:
        print("  子进程失败 ⇒ 仪器无效，本轮不出结论", flush=True)
        return 1
    mdot = r["mdot_total"]
    print(f"  夹具：ṁ_total = powder_feed_rate × deposition_efficiency = {mdot:.6e} kg/s"
          f"（LSFConfig 默认 5e-4 × 0.7）", flush=True)
    print(f"  K1b 两个 LSFConfig 同类？{r['same_class']}（{r['plain_module']} vs "
          f"{r['src_module']}）⇒ {'PASS(守卫口径确实分裂)' if not r['same_class'] else 'BAD'}",
          flush=True)
    print(f"  K4 docstring 集粉环＝{r['rings']['doc']}× vs 代码＝{r['rings']['code']}× ⇒ 差 "
          f"{r['rings']['code']/r['rings']['doc']-1:.1%}；1.8 与 0.2/0.8 混合权重均属 #33 "
          f"口径耦合旋钮（本轮不动）", flush=True)
    print(f"\n  {'cloud':<8}{'N':>5}{'生产Σs':>13}{'非零':>5}{'同类Σs':>13}{'非零':>5}"
          f"{'确定':>6}{'K2a':>6}{'K2b(2×因子)':>12}{'K2c(环1.5)':>11}", flush=True)
    k0a = k0b = k1a = k2a = k2b = k2c = True
    for k in r["clouds"]:
        ps, ns, N = r["plain"][k]
        ss, sn, _ = r["srccfg"][k]
        det, a, b, c = r["det"][k], r["k2a"][k], r["k2b"][k], r["k2c"][k]
        k0a &= ss > 0.0
        k0b &= det
        k1a &= (ps == 0.0 and ns == 0)
        k2a &= a
        k2b &= b
        k2c &= c
        b_txt = "逐位同" if b else "**有差**"
        c_txt = "有差(应)" if c else "**无差**"
        print(f"  {k:<8}{N:>5}{ps:>13.4e}{ns:>5}{ss:>13.4e}{sn:>5}"
              f"{'OK' if det else 'BAD':>6}{'OK' if a else 'BAD':>6}"
              f"{b_txt:>12}{c_txt:>11}", flush=True)
    print(f"  K0a 正对照（同类口径 Σ>0 ⇒ 夹具本身有信息量）⇒ {'PASS' if k0a else 'FAIL'}",
          flush=True)
    print(f"  K0b 确定性（重复调用逐位同）⇒ {'PASS' if k0b else 'FAIL'}", flush=True)
    print(f"  K1a 生产口径 Σs≡0 ∧ 非零≡0 ⇒ {'PASS(=#36 缺陷已测到)' if k1a else 'BAD'}", flush=True)
    print(f"  K2a 本地重写式逐位复现生产（6/6）⇒ {'PASS(比对仪器有效)' if k2a else 'FAIL'}",
          flush=True)
    print(f"  K2b 解析正确值 2/(πR²) 与生产逐位相同 ⇒ "
          f"{'PASS(=#34 因子差确为 no-op)' if k2b else 'BAD(no-op 不成立 ⇒ #34 定性改写)'}",
          flush=True)
    print(f"  K2c 正对照：环系数 1.8→1.5 必须不同 ⇒ {'PASS' if k2c else 'FAIL(仪器坏了)'}",
          flush=True)

    # ---------------- K3：质量记账 ------------------------------------------
    print("\n[K3] 质量记账（docstring：\"total ṁ over all particles equals mdot_total\"）",
          flush=True)
    print(f"  {'cloud':<8}{'r_max/mm':>9}{'现序Σs':>13}{'亏缺%':>9}{'参考序Σs':>13}"
          f"{'闭合相对差':>11}{'非零(现/参)':>12}", flush=True)
    worst_loss = worst_closure = 0.0
    for k in r["clouds"]:
        d = r["k3"][k]
        worst_loss = max(worst_loss, abs(d["loss_rel"]))
        worst_closure = max(worst_closure, d["ref_closure_rel"])
        pair = f"{d['nz_now']}/{d['nz_ref']}"
        print(f"  {k:<8}{d['max_r_mm']:>9.2f}{d['sum_now']:>13.6e}"
              f"{d['loss_rel']*100:>9.4f}{d['sum_ref']:>13.6e}"
              f"{d['ref_closure_rel']:>11.2e}{pair:>12}", flush=True)
    k3b = worst_loss > 1e-6
    k3c = worst_closure < 1e-12
    print(f"  K3a 最大亏缺 {worst_loss:.4e}（阈值 1e-6 跑前写死）⇒ "
          f"{'**实质违反不变量**：实现轮须改截断顺序' if k3b else '仅措辞问题，不改代码'}",
          flush=True)
    print(f"  K3c 参考序（先截断后归一）最差闭合 {worst_closure:.2e} ⇒ "
          f"{'PASS(1e-12 内闭合)' if k3c else 'BAD'}", flush=True)

    # ---------------- K1c：产品链后果 ---------------------------------------
    print("\n[K1c] `_step_lsf_common`（桩注入 ⇒ 隔离 DEM/MPM/SPH 机械步，只考粉源）", flush=True)
    for name in ("plain", "src"):
        d = r["k1c"][name]
        print(f"  cfg 口径={name:<6} Δmass 逐位不变={str(d['dmass_bits']):<5} "
              f"ΣΔmass={d['dmass_sum']:.6e} kg ΔT 和={d['dT_sum']:.6e} K "
              f"活化前移={d['activation_moved']}", flush=True)
    prod, ctrl = r["k1c"]["plain"], r["k1c"]["src"]
    k1c = bool(prod["dmass_bits"] and prod["dmass_sum"] == 0.0
               and not prod["activation_moved"] and prod["dT_sum"] == 0.0)
    k1c_ctrl = bool((not ctrl["dmass_bits"]) and ctrl["dmass_sum"] > 0.0)
    print(f"  K1c 生产口径＝质量/温度/活化三者全不变 ⇒ "
          f"{'PASS(缺陷在产品链上生效，不只是返回值 0)' if k1c else 'BAD'}", flush=True)
    print(f"  K1c 正对照＝同类口径下质量确实增加（ΣΔ={ctrl['dmass_sum']:.3e} kg）⇒ "
          f"{'PASS' if k1c_ctrl else 'FAIL(这条断言抓不到东西)'}", flush=True)

    # ---------------- K2d：死变量与假注释（AST 计数） ------------------------
    print("\n[K2d] 死变量与假注释（AST 名字计数，非目测）", flush=True)
    fsrc = (SRC / "diffmech/methods/am/particle_am.py").read_text()
    tree = ast.parse(fsrc)
    fn = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
          and n.name == "lsf_powder_deposition_profile"][0]
    names = ("pi_R2", "ring_R2", "mdot_total", "discrete_sum", "catchment")
    occ = {t: sum(1 for n in ast.walk(fn) if isinstance(n, ast.Name) and n.id == t)
           for t in names}
    asg = {t: sum(1 for n in ast.walk(fn) if isinstance(n, ast.Assign)
                  and any(isinstance(x, ast.Name) and x.id == t for x in n.targets))
           for t in names}
    print(f"  {'name':<14}{'出现':>6}{'赋值':>6}  判定", flush=True)
    for t in names:
        verdict = "死变量（只赋值不用）" if occ[t] == asg[t] == 1 else "被读用"
        print(f"  {t:<14}{occ[t]:>6}{asg[t]:>6}  {verdict}", flush=True)
    k2d_dead = occ["pi_R2"] == asg["pi_R2"] == 1
    k2d_ctrl = occ["ring_R2"] > asg["ring_R2"] >= 1
    print(f"  K2d `pi_R2` 只被赋值一次、再无引用 ⇒ {'PASS(死变量)' if k2d_dead else 'BAD'}",
          flush=True)
    print(f"  K2d 正对照 `ring_R2` 被读用（{occ['ring_R2']}>{asg['ring_R2']}）⇒ "
          f"{'PASS(计数仪器能区分死/活)' if k2d_ctrl else 'FAIL'}", flush=True)
    doc = [l.strip() for l in fsrc.splitlines() if "π R²" in l or "Normalization" in l
           or "equals mdot_total" in l or "1.5 ×" in l]
    print(f"  相关原文（{len(doc)} 行）：", flush=True)
    for l in doc:
        print(f"      {l}", flush=True)
    print("  解析事实：∫₀^∞ exp(−2r²/R²)·2πr dr = πR²/2 ⇒ 归一 prefactor 应为 2/(πR²)，"
          "注释写的 1/(πR²) 差 2 倍", flush=True)

    # ---------------- K5：server.py 的值导入 --------------------------------
    print("\n[K5] server.py:557 的 `src.` **值**导入（与 isinstance 守卫不同类）", flush=True)
    r5 = child("amapi", r"""
import json
import diffmech.methods.am.am_api as A1
import src.diffmech.methods.am.am_api as A2
b1, b2 = A1._PROCESS_BOUNDS, A2._PROCESS_BOUNDS
srv = open("src/diffmech/server.py").read()
blk = srv.split("from diffmech.methods.am import (")[1].split(")")[0]
print(json.dumps({"same_module": A1 is A2,
                  "modnames": [A1.__name__, A2.__name__],
                  "keys_equal": sorted(b1) == sorted(b2),
                  "n_keys": len(b1),
                  "values_equal": all(b1[k] == b2[k] for k in b1),
                  "am_api_at_module_level": "am_api" in blk,
                  "uses_plain_attribute": "am_api._PROCESS_KEYS" in srv,
                  "n_local_uses": srv.count("_PROCESS_BOUNDS")}))
""", timeout=600)
    k5_dup = r5.get("same_module") is False
    k5_harmless = bool(r5.get("values_equal")) and bool(r5.get("keys_equal"))
    print(f"  两副口径的 am_api 是否同一模块对象＝{r5.get('same_module')} "
          f"（{r5.get('modnames')}）⇒ {'重复实例化确认（模块级状态被复制两份）' if k5_dup else 'BAD(未复现)'}",
          flush=True)
    print(f"  `_PROCESS_BOUNDS`：键序相同＝{r5.get('keys_equal')}、{r5.get('n_keys')} 键"
          f"逐值相同＝{r5.get('values_equal')} ⇒ 当前**行为**无害（只做成员检查）；"
          f"文件内出现 {r5.get('n_local_uses')} 次 `_PROCESS_BOUNDS`", flush=True)
    print(f"  server.py 已在模块级导入 am_api＝{r5.get('am_api_at_module_level')}、"
          f"且同文件已用 `am_api._PROCESS_KEYS` 属性口径＝{r5.get('uses_plain_attribute')}"
          f" ⇒ 修法＝删局部 import，统一改 `am_api._PROCESS_BOUNDS`", flush=True)

    # ---------------- 处置 --------------------------------------------------
    print("\n[处置]（跑前写死的 D1–D4 套判）", flush=True)
    disp = {"K0a": k0a, "K0b": k0b, "K1a": k1a, "K1b": not r["same_class"],
            "K1c_defect": k1c, "K1c_control": k1c_ctrl, "K2a": k2a, "K2b": k2b,
            "K2c": k2c, "K2d_dead": k2d_dead, "K2d_control": k2d_ctrl,
            "K3a_over_1e-6": k3b, "K3c_closure": k3c, "K5_dup": k5_dup,
            "K5_harmless": k5_harmless, "K6c_zero_in_net": n_in_net == 0,
            "worst_loss": worst_loss, "worst_closure": worst_closure}
    print("  " + json.dumps(disp, ensure_ascii=False), flush=True)
    d1 = bool(disp["K1a"] and disp["K1c_defect"] and disp["K1c_control"])
    d2 = bool(disp["K2a"] and disp["K2b"] and disp["K2c"] and disp["K2d_dead"])
    print(f"  D1 #36＝真 bug 且在产品链上生效 ⇒ {'成立' if d1 else '不成立'}"
          f" ⇒ 实现轮：3 处 import 口径一次改完＋新测试进收集网", flush=True)
    print(f"  D2 #34＝因子差确为 no-op ⇒ {'成立' if d2 else '不成立'}"
          f" ⇒ 实现轮只改假注释＋死变量，**不碰任何系数**", flush=True)
    print(f"  D3 是否把 #34 扩到截断顺序＝{'**扩**（亏缺 > 1e-6）' if k3b else '不扩（≤1e-6，仅措辞）'}",
          flush=True)
    print("  D4 #36 修复会让该路径首次真正执行 ⇒ 任何系数改动都必须与 #36 同批；"
          "本轮结论：系数一字不动", flush=True)
    print("  关闭条件（实现轮）＝①3 处 import 口径一次改完 ②新测试进 `tests/`（收集网内），"
          "含正对照「LSF 一步之后质量必须严格增大」＋「SLMConfig 必须恒零」③全量不新增红/"
          "skip/xfail ④#34 侧只改注释与死变量（＋按 D3 决定是否动截断顺序）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
