#!/usr/bin/env python
"""#39 取证探针（**只测不改**）：正向（server/GUI）步进链是否施粉。

背景（#36 修好之后剩下的问题）
------------------------------
#36 修的是 isinstance 守卫的 import 口径，使 `lsf_powder_deposition_profile` 在生产口径下真出料。
但"出料函数有输出"≠"产品会沉积"：正向逐步模拟走的是
`server.py:449 step_am_server(s.method, …, cfg=s.cfg, …)`，而 `step_am_server` 的**分发键是
method（dem/mpm/sph）**，cfg 的 kind（SLM vs LSF/Clad）在它里面**没有分支** ⇒ 若如此，
`ProcessCfg.kind="LSF"` 的用户拿到的熔覆层质量仍恒不增长，`powder_feed_rate`/
`deposition_efficiency` 在这条链上是死参数。这是**另一种机制**（路由缺失），
不是 #36 的口径分裂 ⇒ 另立任务，不混进本轮（N4：不同机制的补丁不得叠加调系数）。

⚠ 本件**不 import `diffmech.server`**：R5 实测该模块在当前环境不可导入
   （`sse_starlette` 未装，且 fastapi/sse_starlette/uvicorn 在 pyproject 与 requirements-gui.txt
   里**都没有声明**）。因此 R1/R2 用**逐行照抄 server.py 的会话构造配方**
   （`_build_geometry` 的 circle_2d 分支＋`_build_solver_state` 的 dem 分支＋`setup_particle_am`）
   复现同一副 state，再按 `run_steps_stream` 的调用形状调 `step_am_server`。
   路由那一半由 **R3 的源码普查**承载（server.py 是否引用 `step_lsf_am*`、`step_am_server` 体内
   是否有 lsf 分支），不依赖执行 server.py。

跑前写死的判据（五道，先定再看结果）
------------------------------------
R5 `import diffmech.server` 的异常类型（登记环境事实；失败**不算**本件 FAIL）。
R1 产品调用形状一步：`step_am_server(method="dem", cfg=LSFConfig)` ⇒ Σ Δmass。
   若 **== 0** ⇒ 路由缺失成立（#39 为真缺陷）；若 == ṁ·dt ⇒ 我读错了码，#39 不成立。
R2 同一副 state 走 `step_lsf_am(method="dem")` ⇒ Σ Δmass 必须 == ṁ·dt（相对差 < 1e-9）
   ⇒ 证明"物理与粉源都在，缺的只是接线"（排除"这副夹具本该不出料"的解释）。
R3 调用图普查：树内 `step_lsf_am*` 引用，server.py 命中数**单列**；
R3a `step_am_server` **体内**（AST 定界，非全文件）含 "lsf" 的行数。
R4 对称读数：SLMConfig 经 `step_lsf_am` ⇒ Σ Δmass 严格 == 0（守卫未放宽的第二处证据）。

诚实边界：本件只判"施粉与否"，不判沉积后的温度/形态是否合理（那要等 #18/#10 的靶）。
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src"

CODE = r"""
import importlib, json
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

out = {}
try:
    importlib.import_module("diffmech.server")
    out["R5"] = {"importable": True, "err": None}
except Exception as e:
    out["R5"] = {"importable": False,
                 "err": f"{type(e).__name__}: {e}"}

from diffmech.methods.am import (LSFConfig, SLMConfig, from_waypoints, multi_layer_paths,
                                 place_particles_in_sdf, sdf_circle_2d, setup_particle_am)
from diffmech.methods.am.particle_am import step_lsf_am
from diffmech.methods.am.am_api import step_am_server
from diffmech.methods.dem import DEMConfig, make_dem_state

# —— 照抄 server.py 的会话构造配方（circle_2d + dem + LSF kind）——
dim = 2
spacing = 0.05
layer_zs = np.asarray([0.0], dtype=np.float64)
sdf = sdf_circle_2d(np.asarray([0.0, 0.0]), 0.1)
bbox = [(-0.1, 0.1), (-0.1, 0.1)]
pos, lids = place_particles_in_sdf(sdf, bbox, layer_zs, dim=dim, spacing=spacing)
paths = multi_layer_paths([from_waypoints(np.array([[0.0, 0.0], [0.18, 0.0]]),
                                          layer_z=0.0)])
cfg = LSFConfig(laser_power=200.0, scan_speed=1.0, layer_thickness=40e-6,
                track_spacing=1.5e-3, beam_radius=50e-6, powder_feed_rate=5e-4,
                absorption=0.5, deposition_efficiency=0.7, preheat_temp=373.0)
solver_cfg = DEMConfig(k=5e2, gamma=0.5)
solver_state = make_dem_state(position=jnp.asarray(pos), radius=15e-3,
                              mass=1e-9, dim=dim)
problem = setup_particle_am(pos, lids, paths, cfg, dim=dim, particle_mass=1e-9)
am_state = problem.am_state
starts = jnp.asarray(problem.layer_start_times, dtype=jnp.float64)
durs = jnp.asarray(problem.layer_durations, dtype=jnp.float64)
dt = 1e-5
mdot = cfg.powder_feed_rate * cfg.deposition_efficiency
out["fixture"] = {"n_particles": int(pos.shape[0]), "mdot": mdot, "dt": dt,
                  "expect": mdot * dt, "cfg_class": type(cfg).__name__}
m_before = float(np.asarray(am_state.mass).sum())

# R1 —— 产品调用形状（与 server.py:449 run_steps_stream 一致）
o1 = step_am_server("dem", solver_state=solver_state, am_state=am_state, dt=dt, t=0.0,
                    solver_cfg=solver_cfg, cfg=cfg, paths=paths,
                    layer_start_times=problem.layer_start_times,
                    layer_durations=problem.layer_durations,
                    position0=jnp.asarray(pos))
m1 = float(np.asarray(o1["am_state"].mass).sum())
out["R1"] = {"dm": m1 - m_before, "keys": sorted(o1.keys())}

# R2 —— 同一副 state 走 LSF 步进器
_, am2 = step_lsf_am(solver_state, am_state, dt, dt, solver_cfg, cfg, paths,
                     starts, durs, "dem")
dm2 = float(np.asarray(am2.mass).sum()) - m_before
out["R2"] = {"dm": dm2, "T_sum": float(np.asarray(am2.temperature).sum()),
             "nz_new_mass": int((np.asarray(am2.mass) > 1e-9 + 1e-30).sum())}

# R4 —— SLM 配置经 LSF 步进器必须严格零
slm = SLMConfig(laser_power=200.0, scan_speed=1.0, beam_radius=50e-6,
                preheat_temp=373.0)
_, am3 = step_lsf_am(solver_state, am_state, dt, dt, solver_cfg, slm, paths,
                     starts, durs, "dem")
out["R4"] = {"dm": float(np.asarray(am3.mass).sum()) - m_before,
             "slm_has_feed": hasattr(slm, "powder_feed_rate")}
print(json.dumps(out))
"""


def main() -> int:
    print("== #39 正向步进链施粉取证（只测不改）==", flush=True)
    print(f"   REPO＝{REPO}；device 钉 CPU（用户 2026-10-08「先不要占用GPU」）", flush=True)
    dirty = subprocess.run(["git", "-c", "core.quotePath=false", "status", "--porcelain",
                            "--", "src", "tests"], capture_output=True, text=True,
                           cwd=str(REPO)).stdout.strip().splitlines()
    head = subprocess.run(["git", "log", "-1", "--format=%h"], capture_output=True,
                          text=True, cwd=str(REPO)).stdout.strip()
    print(f"  tree＝{head}；src/tests 脏项＝{len(dirty)} 行（＝#34/#36 既有改动，"
          f"本件不新增任何改动）", flush=True)

    refs = subprocess.run(["grep", "-rn", "-E", r"\bstep_lsf_am(_scan)?\b", "src", "tests",
                           "examples", "docs", "--include=*.py"],
                          capture_output=True, text=True, cwd=str(REPO)).stdout.strip().splitlines()
    prod = [l for l in refs if l.startswith("src/")]
    srv = [l for l in prod if "/server.py:" in l]
    print(f"\n[R3] `step_lsf_am*` 引用：src/ 内＝{len(prod)} 行（server.py＝{len(srv)} 行）；"
          f"tests/examples/docs＝{len(refs) - len(prod)} 行", flush=True)
    for l in prod:
        print(f"      {l}", flush=True)

    # R3a：step_am_server 体内 lsf 行数（AST 定界）
    inline = r"""
import ast, re, json
src = open('src/diffmech/methods/am/am_api.py').read()
for nd in ast.parse(src).body:
    if isinstance(nd, ast.FunctionDef) and nd.name == 'step_am_server':
        seg = ast.get_source_segment(src, nd)
        hits = [l.strip() for l in seg.splitlines() if re.search('lsf', l, re.I)]
        disp = [l.strip() for l in seg.splitlines() if re.search(r'method\s*==', l)]
        print(json.dumps({"body_lines": len(seg.splitlines()), "lsf_hits": len(hits),
                          "hits": hits[:3], "dispatch": disp[:4]}))
"""
    p3 = subprocess.run([sys.executable, "-c", inline], capture_output=True, text=True,
                        cwd=str(REPO))
    d3 = json.loads(p3.stdout.strip()) if p3.stdout.strip() else {}
    print(f"  R3a `step_am_server` 体内（AST 定界，{d3.get('body_lines')} 行）含 'lsf' 的行数"
          f"＝{d3.get('lsf_hits')}；分发键＝{d3.get('dispatch')}", flush=True)
    for h in d3.get("hits", []):
        print(f"      {h}", flush=True)

    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="", JAX_ENABLE_X64="1",
               PYTHONPATH=str(SRC) + os.pathsep + env.get("PYTHONPATH", ""))
    p = subprocess.run([sys.executable, "-c", CODE], capture_output=True, text=True,
                       cwd=str(REPO), env=env, timeout=1800)
    if p.returncode != 0:
        print(f"  子进程失败 rc={p.returncode}\n{p.stderr[-1500:]}", flush=True)
        return 1
    r = json.loads(p.stdout.strip().splitlines()[-1])
    print(f"\n[R5] `import diffmech.server` ⇒ 可导入＝{r['R5']['importable']}"
          f"；{r['R5']['err']}", flush=True)
    f = r["fixture"]
    print(f"\n  夹具（照抄 server 配方）：n_particles={f['n_particles']} cfg={f['cfg_class']} "
          f"ṁ={f['mdot']:.6e} dt={f['dt']:.1e} ⇒ ṁ·dt={f['expect']:.6e} kg", flush=True)
    exp = f["expect"]
    r1_zero = abs(r["R1"]["dm"]) < 1e-30
    r2_rel = abs(r["R2"]["dm"] - exp) / exp
    r4_zero = abs(r["R4"]["dm"]) < 1e-30
    print(f"[R1] 产品调用形状一步 Σ Δmass={r['R1']['dm']:.9e} kg（对 ṁ·dt 的比="
          f"{r['R1']['dm'] / exp:.3e}）⇒ "
          f"{'真缺陷成立：正向链不施粉' if r1_zero else '不成立：链已施粉（读码判错）'}",
          flush=True)
    print(f"[R2] 同副 state 走 LSF 步进器 Σ Δmass={r['R2']['dm']:.9e} kg，相对差 "
          f"{r2_rel:.3e}；ΔT 和={r['R2']['T_sum']:.6e} K；质量>基准的粒子数="
          f"{r['R2']['nz_new_mass']}/{f['n_particles']} ⇒ "
          f"{'PASS（粉源在真几何夹具上确实出料 ⇒ 缺的只是接线）' if r2_rel < 1e-9 else 'FAIL'}",
          flush=True)
    print(f"[R4] SLMConfig 经 LSF 步进器 Σ Δmass={r['R4']['dm']:.9e}（严格零＝{r4_zero}）；"
          f"SLMConfig 有 powder_feed_rate 属性＝{r['R4']['slm_has_feed']}", flush=True)

    route_missing = (len(srv) == 0 and d3.get("lsf_hits") == 0)
    ok = r1_zero and r2_rel < 1e-9 and r4_zero and route_missing
    print(f"\n[处置] R1 零 ∧ R2 命中 ∧ R4 零 ∧ R3(server 引用=0) ∧ R3a(体内 lsf=0) ⇒ "
          f"{'**#39 判为路由缺失型真缺陷**（与 #36 口径分裂不同机制，不并轮修）' if ok else '判定不成立，见上'}",
          flush=True)
    print(f"  ⚠ R5 的环境事实单独登记：HTTP 后端依赖（fastapi/sse_starlette/uvicorn）"
          f"在 pyproject 与 requirements-gui.txt 中均未声明。", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
