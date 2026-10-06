"""A0 落地后**下游链条**的代价实测（CPU only，CUDA_VISIBLE_DEVICES=""）。

背景：A0 之前 `solve_enthalpy_thermal` 有硬默认 `n_steps=80`，且 dt 取
min(稳定限, total·dx/(v·n_steps)) —— 后者使曝光时长被 dx 改写（剂量失真）。
A0 改成 dt=路径长/v/n_steps、步数由 CFL 推导后，`tests/test_dimensional_optimization.py`
有 15 条失败，两类原因：
  (i) jax.grad/jit 里 n_steps 是**静态形状**，无法从 tracer 求 ceil → 必须调用方显式给；
  (ii) eager 模式下 CFL 目标步数超 max_steps=20000 → 响亮拒绝。
本探针把"给多少步才便宜/可微"量化，供 α 标定级 vs 链条缺省求解器的决策。

跑前登记（跑后打分，不得事后改判据）：
  Q1 eager：该反演夹具（0.8mm 球、dx=120µm、20 层）稳定下限 n_stable 一次正向
     < 60s（体素少，应当很快）。
  Q2 grad：对该 n_stable 步做 jax.grad（对 laser_power）**不可行**——预期要么
     >300s 要么内存/编译爆掉。若 300s 内完成，则"显式热解直接进反演循环"成立，
     §24.3 的 3.2 年估算需要下调。
  Q3 grad：把步数压到 200（低于稳定下限的**人为档位**，仅测代价，不代表可信结果）
     时 grad 应 < 60s → 证明旧测试的"快"来自步数被人为压死，即来自剂量失真。
"""
import time
import warnings

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from amforge import geometry as G
from amforge.core.contracts import ProcessPlan
from amforge.thermal_enthalpy import solve_enthalpy_thermal, suggest_n_steps
from amforge.inverse import simulate

KW = dict(bounds=[(-0.5e-3, 0.5e-3)] * 3, spacing=120e-6, name="opt-demo")


def geo():
    return G.from_sdf_fn(lambda x: jnp.linalg.norm(x, axis=-1) - 0.4e-3, **KW)


def plan(power=900.0, speed=0.6):
    g = geo()
    return ProcessPlan.uniform(int(g.layer_count(40e-6)), modality="SLM",
                               laser_power=power, scan_speed=speed,
                               layer_thickness=60e-6, hatch_spacing=120e-6,
                               beam_radius=60e-6, absorption=0.5,
                               preheat_temp=473.0, dwell_time=0.0)


g, pl = geo(), plan()
print(f"夹具 shape={tuple(int(s) for s in g.shape)} nvox={int(g.sdf.size)} "
      f"n_layers={int(geo().layer_count(40e-6))} n_steps_cfl="
      f"{suggest_n_steps(g, pl)}", flush=True)

# n_stable（cfl=1 下限）：用 suggest 的 cfl=0.35 结果除以 0.35 反推，+2 步留余量
ns_cfl = suggest_n_steps(g, pl)
ns_stable = int(-(-ns_cfl * 0.35 // 1)) + 2
print(f"  n_cfl={ns_cfl} n_stable(≈cfl1,含余量)={ns_stable}", flush=True)

print("\n=== Q1 eager 正向（n_steps=n_stable）===", flush=True)
t0 = time.perf_counter()
with warnings.catch_warnings(record=True) as wq:
    warnings.simplefilter("always")
    th = solve_enthalpy_thermal(geometry=g, process=pl,
                                params=dict(material="316L", n_steps=ns_stable))
q1 = time.perf_counter() - t0
print(f"  {q1:.2f}s peak={float(jnp.max(th.peak_temperature)):.1f}K "
      f"警告 {len(wq)} 条 => {'PASS' if q1 < 60 else 'FAIL'}（判据 <60s）", flush=True)

print("\n=== Q3 grad @ n_steps=200（人为低压档位，只测代价）===", flush=True)
p2 = plan(power=900.0)


def scalar200(P):
    pl2 = ProcessPlan.uniform(int(g.layer_count(40e-6)), modality="SLM",
                              laser_power=P, scan_speed=0.6,
                              layer_thickness=60e-6, hatch_spacing=120e-6,
                              beam_radius=60e-6, absorption=0.5,
                              preheat_temp=473.0, dwell_time=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = solve_enthalpy_thermal(geometry=g, process=pl2,
                                   params=dict(material="316L", n_steps=200))
    return float(jnp.mean(t.final_temperature))


t0 = time.perf_counter()
grad200 = jax.grad(scalar200)(900.0)
q3 = time.perf_counter() - t0
print(f"  grad@200 用时 {q3:.2f}s dT/dP={grad200:.3e} => "
      f"{'PASS' if q3 < 60 else 'FAIL'}（判据 <60s）", flush=True)

print("\n=== Q2 grad @ n_steps=n_stable（诚实档位）——限时 300s ===", flush=True)


def scalar_full(P):
    pl3 = ProcessPlan.uniform(int(g.layer_count(40e-6)), modality="SLM",
                              laser_power=P, scan_speed=0.6,
                              layer_thickness=60e-6, hatch_spacing=120e-6,
                              beam_radius=60e-6, absorption=0.5,
                              preheat_temp=473.0, dwell_time=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = solve_enthalpy_thermal(geometry=g, process=pl3,
                                   params=dict(material="316L", n_steps=ns_stable))
    return float(jnp.mean(t.final_temperature))


t0 = time.perf_counter()
try:
    gf = jax.grad(scalar_full)(900.0)
    q2 = time.perf_counter() - t0
    print(f"  grad@{ns_stable} 用时 {q2:.2f}s dT/dP={gf:.3e} => "
          f"{'PASS(3.2年估算需下调)' if q2 < 300 else 'FAIL(超300s)'}", flush=True)
except Exception as e:
    q2 = time.perf_counter() - t0
    print(f"  grad@{ns_stable} 在 {q2:.2f}s 后异常：{type(e).__name__}: "
          f"{str(e)[:160]}", flush=True)

print("\n=== 链条 simulate() 在两种缺省下的代价（eager）===", flush=True)
for name, pr in (("enthalpy(200 步显式)", dict(n_steps=200, material="316L")),
                 ("history(Rosenthal 降阶)", dict(material="316L"))):
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = simulate(g, pl, thermal_solver=("history" if "history" in name
                                                   else "enthalpy"),
                           params=dict(pr))
        print(f"  {name:22s} {time.perf_counter()-t0:6.2f}s ok", flush=True)
    except Exception as e:
        print(f"  {name:22s} {time.perf_counter()-t0:6.2f}s "
              f"{type(e).__name__}: {str(e)[:100]}", flush=True)

print(f"\n打分：Q1 {'PASS' if q1 < 60 else 'FAIL'}  "
      f"Q2 {'PASS' if q2 < 300 else 'FAIL'}  Q3 {'PASS' if q3 < 60 else 'FAIL'}",
      flush=True)
