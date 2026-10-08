#!/usr/bin/env python
"""#34/#36 改后复核（实现轮的验收读数；与 `am_t34_t36_lsf_probe.py` 成对）。

为什么需要单独的复核件
----------------------
选型探针的三条本地重写变体（K2a/K2b/K2c）是**钉在改前公式**上的（先归一后截断）。
实现轮把截断挪到归一**之前** ⇒ 那些变体按设计就会不再逐位相同 ⇒ 探针不能改判据重跑
（那是"改尺子迁就结果"）。所以改后的验收交给本件＋收集网里的 `tests/test_lsf_deposition.py`。

本件量的事（参照数**由命令从探针日志解析**，不手抄）
--------------------------------------------------
C0 全树 `src.` 前缀导入普查＝0；正对照＝同一正则打在**已发布的 HEAD 副本**上必须命中。
C1 生产口径（顶层 `diffmech…` 的 LSFConfig）6 副点云上 Σ s 对 ṁ_total 闭合到 1e-12，
   且 ≤ 改前参考序实测最差闭合；并与日志『参考序Σs』列在 1e-6 内一致。
C2 非零计数与改前日志的两列（K0『同类非零』、K3『非零(现/参)』）逐一相同
   ⇒ 闭合不是因为截断失效。
C1r 生产夹具（真 `from_waypoints`/`multi_layer_paths`，非探针的桩对象）同样闭合。
C3 `_step_lsf_common` 一步：Σ Δmass == ṁ_total·dt（改前实测恒 0）；顺带读出
   max(ḟrac_deposited) 对活化前移阈值 1e-3 的位置（登记项，本件不判定）。
C4 ∂Σ s/∂feed == deposition_efficiency（解析预期；改前实测 0.0）。
C5 收集网里新测试的记分（改前同一份文件＝4 failed / 2 passed）。
C5b SLM 配置仍恒零 ⇒ 守卫没被放宽成恒真。
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
EV = Path(__file__).resolve().parent

CLOUD_KEYS = ["A2x2", "grid17", "line25", "g3d11", "tight", "wide"]

# ---------------------------------------------------------------- 参照解析
RE_K3 = re.compile(r"^\s{2}(\w+)\s+([0-9.]+)\s+([0-9.eE+-]+)\s+(-?[0-9.eE+-]+)"
                   r"\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+(\d+)/(\d+)\s*$")
RE_K0 = re.compile(r"^\s{2}(\w+)\s+(\d+)\s+([0-9.eE+-]+)\s+(\d+)\s+"
                   r"([0-9.eE+-]+)\s+(\d+)\s+OK\s+OK")


def parse_probe(log: Path) -> dict:
    """从改前探针日志里解析参照数（返回 {cloud: {nz, ref_sum}}, worst_closure, mdot）。"""
    lines = log.read_text(encoding="utf-8").splitlines()
    k3, k0, worst, mdot = {}, {}, None, None
    for l in lines:
        m = RE_K3.match(l)
        if m:
            k3[m.group(1)] = {"ref_sum": float(m.group(5)),
                              "nz_cur": int(m.group(7)), "nz_ref": int(m.group(8))}
        m = RE_K0.match(l)
        if m:
            k0[m.group(1)] = {"nz": int(m.group(6))}
        if l.strip().startswith('{"K0a"'):
            worst = json.loads(l.strip())["worst_closure"]
        m = re.search(r"=\s*([0-9.eE+-]+)\s*kg/s", l)
        if m and mdot is None:
            mdot = float(m.group(1))
    # 解析器自身的守护：缺一副点云／两列非零计数不一致／参照闭合缺失 ⇒ 立刻炸
    assert set(k3) == set(CLOUD_KEYS), f"K3 解析不全：{sorted(k3)}"
    assert set(k0) == set(CLOUD_KEYS), f"K0 解析不全：{sorted(k0)}"
    for k in CLOUD_KEYS:
        assert k3[k]["nz_cur"] == k3[k]["nz_ref"] == k0[k]["nz"], \
            f"{k} 三列非零计数互不一致 ⇒ 解析器认错表"
    assert isinstance(worst, float) and 0 < worst < 1e-12, f"worst_closure 解析异常：{worst}"
    assert isinstance(mdot, float) and mdot > 0, f"ṁ_total 解析异常：{mdot}"
    return {"ref": k3, "worst_closure": worst, "mdot_log": mdot}


# ---------------------------------------------------------------- 子进程
CODE = r"""
import json
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from diffmech.methods.am import LSFConfig, SLMConfig, from_waypoints, multi_layer_paths
from diffmech.methods.am import particle_am as PA

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
# 与探针逐字相同的桩夹具（保证与改前参照列同题）
lay = type("L", (), {})()
lay.waypoints = np.array([[0.0, 0.0], [1.0e-3, 0.0]])
lay.cum_length = np.array([0.0, 1.0e-3])
lay.total_length = 1.0e-3
lay.layer_z = 0.0
paths = type("P", (), {})()
paths.layers = [lay]
# 生产形状的真夹具
real_paths = multi_layer_paths([from_waypoints(np.array([[0.0, 0.0], [1.0e-3, 0.0]]),
                                               layer_z=0.0)])
STARTS = jnp.asarray([0.0]); DURS = jnp.asarray([2.0e-3]); T = 1.0e-3
cfg = LSFConfig()
mdot = cfg.powder_feed_rate * cfg.deposition_efficiency
out = {"mdot": mdot, "feed": cfg.powder_feed_rate,
       "eta": cfg.deposition_efficiency, "clouds": {}, "real": {}}
for k, (dim, pos) in CLOUDS.items():
    v = np.asarray(PA.lsf_powder_deposition_profile(jnp.asarray(pos), T, paths,
                                                   STARTS, DURS, cfg, dim=dim))
    out["clouds"][k] = {"N": int(v.shape[0]), "sum": float(v.sum()),
                        "nz": int((v > 0).sum()),
                        "closure_rel": abs(float(v.sum()) - mdot) / mdot,
                        "bits": [float(x).hex() for x in v][:3]}
    w = np.asarray(PA.lsf_powder_deposition_profile(jnp.asarray(pos), T, real_paths,
                                                   STARTS, DURS, cfg, dim=dim))
    out["real"][k] = {"closure_rel": abs(float(w.sum()) - mdot) / mdot,
                      "nz": int((w > 0).sum()),
                      "same_as_stub": [float(x).hex() for x in w] ==
                                      [float(x).hex() for x in v]}
slm = np.asarray(PA.lsf_powder_deposition_profile(jnp.asarray(CLOUDS["line25"][1]), T,
                                                  paths, STARTS, DURS, SLMConfig(), dim=2))
out["slm"] = {"all_zero": bool(np.all(slm == 0.0)), "N": int(slm.shape[0])}

pos = CLOUDS["tight"][1]
n = pos.shape[0]
am = PA.ParticleAMState(position=jnp.asarray(pos),
                        activation_time=jnp.full(n, 1e9), temperature=jnp.zeros(n),
                        mass=jnp.full(n, 1e-6), layer_id=jnp.zeros(n, dtype=jnp.int32),
                        rho=jnp.full(n, 8000.0))
def stub(s, a, dt, t, sc, cfg, paths, ls, ld, *, walls, cp, alpha_T, tau):
    return s, a
dt = 1e-4
_, am2 = PA._step_lsf_common(stub, None, am, dt, T, None, cfg, paths, STARTS, DURS,
                             dim=2, walls=None, cp=500.0, alpha_T=1e-5, tau=1e-3)
mass0 = np.asarray(am.mass); mass2 = np.asarray(am2.mass)
dm = float(mass2.sum() - mass0.sum())
frac = (mass2 - mass0) / (float(mass0.mean()) + 1e-18)
out["step"] = {"dm": dm, "expect": mdot * dt, "rel": abs(dm - mdot * dt) / (mdot * dt),
               "dT": float(np.asarray(am2.temperature).sum()),
               "frac_max": float(frac.max()), "move_bar": 1e-3,
               "act_moved": bool(np.any(np.asarray(am2.activation_time) < 1e9))}

import dataclasses
def total(feed):
    c = dataclasses.replace(cfg, powder_feed_rate=feed)
    return PA.lsf_powder_deposition_profile(jnp.asarray(pos), T, paths, STARTS, DURS,
                                            c, dim=2).sum()
out["grad"] = {"d_total_dfeed": float(jax.grad(total)(jnp.asarray(cfg.powder_feed_rate))),
               "expect": cfg.deposition_efficiency}
out["guard_module"] = PA.LSFConfig.__module__
print(json.dumps(out))
"""


def main() -> int:
    print("== #34/#36 改后复核（实现轮验收）==", flush=True)
    print(f"   REPO＝{REPO}", flush=True)
    print("   device 钉 CPU（CUDA_VISIBLE_DEVICES=''，用户 2026-10-08「先不要占用GPU」）",
          flush=True)
    log_cands = [EV / "am_t34_t36_lsf_probe.log", Path("/tmp/am_t34_t36_lsf_probe.log")]
    probe_log = next((p for p in log_cands if p.is_file()), None)
    if probe_log is None:
        print("  参照日志缺失 ⇒ 无法比对，FAIL", flush=True)
        return 2
    print(f"  参照来源＝{probe_log}", flush=True)
    refd = parse_probe(probe_log)
    ref = refd["ref"]
    print(f"  解析到参照：ṁ_log={refd['mdot_log']:.6e}，改前参考序最差闭合"
          f"={refd['worst_closure']:.6e}，6 副点云非零计数"
          f"={ {k: ref[k]['nz_ref'] for k in CLOUD_KEYS} }", flush=True)
    # 解析器的正对照：截断日志必须让守护炸掉（证明这道判据真能失败）
    trunc = EV / "_probe_log_truncated.tmp"
    trunc.write_text("\n".join(probe_log.read_text(
        encoding="utf-8").splitlines()[:20]), encoding="utf-8")
    try:
        parse_probe(trunc)
        ctrl = "BAD(解析器在无参照时也没炸)"
        rc_ctrl = 1
    except AssertionError as e:
        ctrl = f"PASS(截断日志⇒{str(e)[:40]})"
        rc_ctrl = 0
    finally:
        trunc.unlink(missing_ok=True)
    print(f"  解析器正对照 ⇒ {ctrl}", flush=True)

    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="", JAX_ENABLE_X64="1",
               PYTHONPATH=str(SRC) + os.pathsep + env.get("PYTHONPATH", ""))
    head = subprocess.run(["git", "log", "-1", "--format=%h"], capture_output=True,
                          text=True, cwd=str(REPO)).stdout.strip()
    dirty = subprocess.run(["git", "-c", "core.quotePath=false", "status", "--porcelain",
                            "--", "src", "tests"], capture_output=True, text=True,
                           cwd=str(REPO)).stdout.strip().splitlines()
    print(f"\n  head={head}；本轮被改文件（src/tests 脏项）＝{len(dirty)} 行", flush=True)
    for l in dirty:
        print(f"      {l}", flush=True)

    PAT = r"^\s*(from src\.|import src\.|from src import)"
    now = subprocess.run(["grep", "-rn", "-E", PAT, "src", "tests", "tests_diffmech",
                          "examples", "--include=*.py"], capture_output=True,
                         text=True, cwd=str(REPO)).stdout.strip()
    n_now = len(now.splitlines()) if now else 0
    # 正对照：同一正则打在**已发布 HEAD** 的两份原文件上必须命中（改前 3 处）
    hits_head = []
    for f in ["src/diffmech/methods/am/particle_am.py", "src/diffmech/server.py"]:
        blob = subprocess.run(["git", "show", f"HEAD:{f}"], capture_output=True,
                              text=True, cwd=str(REPO)).stdout
        hits_head.append(len([l for l in blob.splitlines() if re.match(PAT, l)]))
    n_head = sum(hits_head)
    print(f"\n[C0] 全树 `src.` 前缀导入普查：现树＝{n_now} 行；"
          f"HEAD 同正则命中＝{n_head} 行 {hits_head}（>0 ⇒ 正则与范围都有效）⇒ "
          f"{'PASS' if n_now == 0 and n_head > 0 else 'BAD'}", flush=True)

    p = subprocess.run([sys.executable, "-c", CODE], capture_output=True, text=True,
                       cwd=str(REPO), env=env, timeout=1800)
    if p.returncode != 0:
        print(f"  子进程失败 rc={p.returncode}\n{p.stderr[-1500:]}", flush=True)
        return 1
    r = json.loads(p.stdout.strip().splitlines()[-1])
    d_mdot = abs(r["mdot"] - refd["mdot_log"]) / refd["mdot_log"]
    print(f"\n  守卫实际认的类＝{r['guard_module']}；ṁ_total={r['mdot']:.6e} kg/s"
          f"（与日志夹具差 {d_mdot:.2e}）；feed={r['feed']:.6e} × η={r['eta']}", flush=True)

    print(f"\n[C1/C2] 生产口径 Σ s 与闭合（参照列由命令从日志解析）", flush=True)
    print(f"  {'cloud':<8}{'N':>5}{'Σs(改后)':>13}{'闭合相对差':>11}{'非零':>6}"
          f"{'参照Σs(日志)':>14}{'参照非零':>8}{'列间差':>10}{'判定':>8}", flush=True)
    ok1 = ok2 = ok1b = True
    for k in CLOUD_KEYS:
        d = r["clouds"][k]
        esum, enz = ref[k]["ref_sum"], ref[k]["nz_ref"]
        col = abs(d["sum"] - esum) / esum
        c1 = d["closure_rel"] < 1e-12 and d["closure_rel"] <= refd["worst_closure"] * 10
        c2 = d["nz"] == enz
        c1b = col < 1e-6
        ok1 &= c1; ok2 &= c2; ok1b &= c1b
        print(f"  {k:<8}{d['N']:>5}{d['sum']:>13.6e}{d['closure_rel']:>11.2e}"
              f"{d['nz']:>6}{esum:>14.6e}{enz:>8}{col:>10.2e}"
              f"{'OK' if c1 and c2 and c1b else 'BAD':>8}", flush=True)
    print(f"  C1 六副点云闭合全部 <1e-12 且 ≤10×改前参考序最差闭合 ⇒ "
          f"{'PASS' if ok1 else 'FAIL'}", flush=True)
    print(f"  C1b 与日志『参考序Σs』列一致（<1e-6，即同题同数）⇒ "
          f"{'PASS' if ok1b else 'FAIL'}", flush=True)
    print(f"  C2 非零计数与改前参照（两列互证）逐一相同 ⇒ {'PASS' if ok2 else 'FAIL'}"
          f"（⇒ 闭合不是因为截断失效）", flush=True)
    real_all = all(r["real"][k]["closure_rel"] < 1e-12 and r["real"][k]["nz"] > 0
                   for k in CLOUD_KEYS)
    same = sum(1 for k in CLOUD_KEYS if r["real"][k]["same_as_stub"])
    print(f"  C1r 真生产夹具（from_waypoints/multi_layer_paths）闭合＋非零 ⇒ "
          f"{'PASS' if real_all else 'FAIL'}；与桩夹具逐位相同的点云数＝{same}/6",
          flush=True)
    print(f"  C5b SLM 配置仍恒零（{r['slm']['all_zero']}，N={r['slm']['N']}）⇒ "
          f"{'PASS(守卫没放宽成恒真)' if r['slm']['all_zero'] else 'BAD'}", flush=True)

    st, gr = r["step"], r["grad"]
    c3 = st["rel"] < 1e-12 and st["dm"] > 0.0 and st["dT"] > 0.0
    c4 = abs(gr["d_total_dfeed"] - gr["expect"]) / gr["expect"] < 1e-9
    print(f"\n[C3] 一步 LSF：Σ Δmass={st['dm']:.9e} kg，预期 ṁ·dt={st['expect']:.9e} kg，"
          f"相对差 {st['rel']:.2e}；ΔT 和={st['dT']:.6e} K ⇒ "
          f"{'PASS（改前实测恒 0）' if c3 else 'FAIL'}", flush=True)
    print(f"     活化前移读数（不判定）：max ḟrac={st['frac_max']:.6e} vs 阈值 "
          f"{st['move_bar']:.0e} ⇒ 前移发生＝{st['act_moved']}"
          f"（此阈值可达性＝另立登记项）", flush=True)
    print(f"[C4] ∂Σ s/∂feed={gr['d_total_dfeed']:.12e} vs 解析预期 "
          f"deposition_efficiency={gr['expect']:.12e} ⇒ "
          f"{'PASS（改前实测 0.0）' if c4 else 'FAIL'}", flush=True)

    print(f"\n[C5] 收集网里的新测试", flush=True)
    q = subprocess.run([sys.executable, "-m", "pytest", "tests/test_lsf_deposition.py",
                        "-q", "-rA", "-p", "no:cacheprovider"],
                       capture_output=True, text=True, cwd=str(REPO), env=env,
                       timeout=1800)
    tail = [l for l in q.stdout.splitlines()
            if " passed" in l or " failed" in l or "error" in l.lower()]
    for l in tail:
        print(f"  {l.strip()}", flush=True)
    score = tail[-1].strip() if tail else "n/a"
    c5 = "failed" not in score and "error" not in score.lower() and "6 passed" in score
    print(f"  ⇒ 改后记分（改前同一份文件＝4 failed / 2 passed）：{score} ⇒ "
          f"{'PASS' if c5 else 'FAIL'}", flush=True)

    allok = (n_now == 0 and n_head > 0 and rc_ctrl == 0 and ok1 and ok1b and ok2
             and real_all and r["slm"]["all_zero"] and c3 and c4 and c5)
    print(f"\n[处置] C0∧C1∧C1b∧C1r∧C2∧C3∧C4∧C5∧C5b ⇒ "
          f"{'#34/#36 实现轮验收成立' if allok else '不成立，回到实现轮'}", flush=True)
    print(f"  全量记分见 am_t34_t36_full_regression.log（A0 两条登记红灯应逐字符未变）",
          flush=True)
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
