"""#23 打分第 3 轮（小、快）：解释上一轮 K3 的 **f0 灵敏度 0.000%**，并查「孤立切割壳」污染。

上一轮 `am_t23_capacity_probe_delta.log` 的 K3 报出 f0=0.25 与 f0=0.5 在**每个 δ、每档**的
peak 差 = 0.000%。这不是"下限不敏感"，而是两条**可证的结构**叠加，必须逐条实测确认：

  P1 `solid_weight = clip(0.5 − sdf/dx, 0, 1)` ⇒ **fv ≥ 0.5 ⟺ sdf ≤ 0** ⇒ 落在实体内
     （`solid_mask`）的体素恒有 fv ≥ 0.5，故 `max(fv, f0)` 对 f0∈{0.25,0.5} **恒等于 fv**
     ⇒ patch 在实体内部与 f0 无关。实测：报出每档「实体内 fv≤0.5 的体素数」（δ=0 应为一整层
     刀锋面 fv=0.5，δ≠0 应为 0）。
  P2 `_div_alpha_grad(..., mask=solid_mask)` 把**任何**涉及非实体体素的面通量置零
     （`thermal_enthalpy.py:162-164` 的乘积掩膜）⇒ fv<0.5 的**外部切割壳**在扩散上完全孤立，
     却在 patch 下吃到 `fv·s/max(fv,f0)`＝0.5s（f0=0.5）或 s（f0=0.25）的源 ⇒ 两组的壳温不同，
     但 `peak = jnp.maximum(peak, Tn)` 是**全网格** max。⇒ 必须验证壳会不会成为 argmax。

预注册判据：
  L1 污染判据：对每个 (δ,dx,mode,f0) 报 peak_全网格、peak_仅实体、peak_切割壳(fv<0.5)、
     argmax 处的 fv。**若 peak_全网格 > peak_仅实体** ⇒ 峰值被孤立壳占据 ⇒ 这条观测量在
     #23 实现里**必须**按实体掩膜取 max（否则"峰值温度"会报告一个物理上孤立、数值上由 f0
     决定的数）。反之若逐位相等 ⇒ 壳从未成为 argmax，peak 口径暂不受污染（壳温仍属非物理，
     照实记录其数值大小）。
  L2 f0 生效面：报 fv<0.5 的体素数（外部壳）与 fv≤0.5 的实体体素数 ⇒ 用实测解释 K3=0.000%，
     而不是留下一句"灵敏度未测"。
  L3 对照：δ=0（全表面 fv=0.5）应给出「实体内 fv≤0.5 数 >0」而 δ=0.25·dx 应 =0——若两者都 0，
     说明 P1 的推理有漏洞，探针须复核后再谈结论。

GPU；只跑 dx∈{50,25}µm × δ∈{0.25,0.5}·dx × f0∈{0.5,0.25} ＝ 8 次短求解（δ=0 由 P1 代数覆盖，
但 L3 仍实测 δ=0 一档作正对照 ⇒ +2 次）。不改生产代码（同一 monkeypatch 口径）。
"""
import time

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from amforge import geometry as G                                          # noqa: E402
from amforge import thermal_enthalpy as TE                                 # noqa: E402
from amforge.core.contracts import (ProcessPlan, SDF_SOLID_TOL,             # noqa: E402
                                    solid_mask, solid_weight)
from amforge.thermal_enthalpy import (solve_enthalpy_thermal,               # noqa: E402
                                      suggest_n_steps)

EXT = (1.2e-3, 0.6e-3, 0.4e-3)
R, P, V = 100e-6, 600.0, 0.8
TIERS = [50e-6, 25e-6]
CASES = [(0.0, 50e-6), (0.25, 50e-6), (0.5, 50e-6),
         (0.0, 25e-6), (0.25, 25e-6), (0.5, 25e-6)]   # 每档都要 δ=0 正对照（L3）
FLOORS = [0.5, 0.25]


def coupon(dx, delta):
    bx, by, bz = EXT
    bounds = [(-bx / 2 - delta, bx / 2 + delta),
              (-by / 2 - delta, by / 2 + delta),
              (-bz / 2 - delta, bz / 2 + delta)]

    def fn(x):
        return jnp.maximum(jnp.maximum(jnp.abs(x[..., 0]) - bx / 2,
                                        jnp.abs(x[..., 1]) - by / 2),
                            jnp.abs(x[..., 2]) - bz / 2)

    return G.from_sdf_fn(lambda x: fn(np.asarray(x, dtype=np.float64)),
                         bounds=bounds, spacing=dx, name="coupon")


def plan():
    return ProcessPlan.uniform(1, modality="SLM", laser_power=P, scan_speed=V,
                               layer_thickness=EXT[2], hatch_spacing=1.4 * R,
                               beam_radius=R, absorption=0.45, preheat_temp=400.0)


ORIG_DIV, ORIG_SRC = TE._div_alpha_grad, TE._cell_integrated_source
CALLS = {"div": 0, "src": 0}
STATE = {"f0": 1.0, "fv": None}


def patched_div(H, alpha, dx, mask=None):
    CALLS["div"] += 1
    return ORIG_DIV(H, alpha, dx, mask=mask) / jnp.maximum(STATE["fv"], STATE["f0"])


def patched_src(positions, coords, power, r_src, dp, dx):
    CALLS["src"] += 1
    return ORIG_SRC(positions, coords, power, r_src, dp, dx) / jnp.maximum(STATE["fv"],
                                                                          STATE["f0"])


TE._div_alpha_grad, TE._cell_integrated_source = patched_div, patched_src

print(f"device = {jax.devices()[0]}", flush=True)
bad = []
rows = []
for frac, dx in CASES:
    for f0 in FLOORS:
        g = coupon(dx, frac * dx)
        fv = solid_weight(g.sdf, jnp.asarray(dx, dtype=jnp.float64))
        STATE.update(f0=f0, fv=fv)
        CALLS.update(div=0, src=0)
        t0 = time.perf_counter()
        th = solve_enthalpy_thermal(geometry=g, process=plan(),
                                    params=dict(material="316L",
                                                source_model="integrated",
                                                n_steps=suggest_n_steps(g, plan())))
        el = time.perf_counter() - t0
        pk = np.asarray(th.peak_temperature, dtype=np.float64)
        sol = np.asarray(solid_mask(g.sdf), dtype=bool)
        fv_np = np.asarray(fv, dtype=np.float64)
        shell = (fv_np > 0.0) & (fv_np < 0.5) & ~sol   # 外部切割壳：0<fv<0.5 且非实体
        arg = np.unravel_index(int(np.argmax(pk)), pk.shape)
        pk_solid = float(pk[sol].max())
        pk_shell = float(pk[shell].max()) if int(shell.sum()) else float("-inf")
        rows.append((frac, dx, f0, float(pk.max()), pk_solid, pk_shell,
                     int(shell.sum()), int(((fv_np <= 0.5) & sol).sum()),
                     int(fv_np[arg]), float(fv_np[arg]), dict(CALLS), el))
        print(f"  δ={frac:4.2f}·dx dx={dx*1e6:5.1f} f0={f0:4.2f} {el:5.2f}s  "
              f"peak全={pk.max():8.1f} peak实={pk_solid:8.1f} peak壳={pk_shell:8.1f}  "
              f"壳体素={int(shell.sum()):6d} 实体内fv≤0.5={int(((fv_np<=0.5)&sol).sum()):6d}  "
              f"argmax处 fv={fv_np[arg]:.3f}{'(实体)' if sol[arg] else '(壳/外部!)'}  "
              f"calls={CALLS}", flush=True)

print("\n=== L1 污染判据：全网格 max 是否等于仅实体 max ===")
for (frac, dx, f0, pk_all, pk_solid, pk_shell, n_shell, n_low, _a, _fv, _c, _el) in rows:
    eq = (pk_all == pk_solid)
    print(f"  δ={frac:4.2f}·dx dx={dx*1e6:5.1f} f0={f0:4.2f}: 全网格={pk_all:8.1f} "
          f"仅实体={pk_solid:8.1f} ⇒ {'逐位相等，壳未占据 argmax' if eq else f'被壳抬高 {pk_all-pk_solid:+.1f}K'}"
          f"  壳最高温={pk_shell:8.1f}K（{'低于' if pk_shell < pk_solid else '高于'}实体峰值）",
          flush=True)
    if not eq:
        bad.append(f"L1 δ={frac}dx dx={dx*1e6}µm f0={f0}：峰值被非实体孤立壳占据 "
                   f"{pk_all - pk_solid:+.1f}K ⇒ #23 的 peak 必须按实体掩膜取 max")

print("\n=== L2/L3：K3=0.000% 的解释是否成立（实体内 fv≤0.5 只在 δ=0 的刀锋层出现）===")
for dx in TIERS:
    n0 = [r for r in rows if r[0] == 0.0 and r[1] == dx and r[2] == 0.5]
    nq = [r for r in rows if r[0] == 0.25 and r[1] == dx and r[2] == 0.5]
    if not (n0 and nq):
        print(f"  dx={dx*1e6:5.1f}: 缺 δ=0 正对照 ⇒ L3 无法判")
        bad.append(f"L3 dx={dx*1e6}µm：缺 δ=0 正对照")
        continue
    print(f"  dx={dx*1e6:5.1f}: δ=0 实体内 fv≤0.5 = {n0[0][7]} 个（一整层刀锋面）, "
          f"δ=0.25 实体内 fv≤0.5 = {nq[0][7]} 个, 外部壳体素 {n0[0][6]}/{nq[0][6]} 个",
          flush=True)
    if n0[0][7] == 0:
        bad.append(f"L3 dx={dx*1e6}µm：δ=0 实体内 fv≤0.5 计数=0 ⇒ P1 推理有漏洞，须复核")
    if nq[0][7] != 0:
        bad.append(f"L3 dx={dx*1e6}µm：δ=0.25 实体内出现 fv≤0.5={nq[0][7]} ⇒ 与 "
                   f"「fv≥0.5 ⟺ sdf≤0」矛盾")

print("\n=== 同 (δ,dx) 下 f0=0.5 vs 0.25 的壳温差（P2 的直接证据）===")
for frac, dx in CASES:
    a = [r for r in rows if r[0] == frac and r[1] == dx and r[2] == 0.5]
    b = [r for r in rows if r[0] == frac and r[1] == dx and r[2] == 0.25]
    if a and b:
        print(f"  δ={frac:4.2f}·dx dx={dx*1e6:5.1f}: peak全 {a[0][3]:8.1f} vs {b[0][3]:8.1f} "
              f"(差 {b[0][3]-a[0][3]:+.6f})  peak壳 {a[0][5]:8.1f} vs {b[0][5]:8.1f} "
              f"(差 {b[0][5]-a[0][5]:+.1f}K)", flush=True)

print(f"\nFAILS: {bad if bad else '无'}")
raise SystemExit(1 if bad else 0)
