"""A5：CPU vs GPU 实测 + f32 试点（2026-10-06，用户已授权 GPU：「可以开始自由使用GPU」）。

共享纪律：机器上另有 `newton_torch_env` 的 GPU 进程（各 <600MiB）与桌面 Xorg，
本探针只用单卡、显存占用极小（最大算例 2.2e4 体素 × f64 ≈ 0.5 MB/场），不抢占、
不常驻；被要求即归还。

**跑前登记的证伪判据**（跑后逐条打分，不得事后改）：
  G1 GPU **x64** 吞吐 **不优于** CPU x64。依据：RTX A6000 的 fp64 只有 fp32 的
     1/32（≈0.35 TFLOP/s），而本核是纯逐体素算术 + 移位 stencil，CPU 侧有 104 核。
     若 GPU x64 反而 >2× CPU → 我对"A6000 fp64 慢"的判断错了（A6000 是 1/32，
     但我们的核是带宽受限，可能吃不满 fp64 率），必须重估。
  G2 GPU **f32** 吞吐 ≥ 20× CPU x64（A6000 fp32 ≈ 38.7 TFLOP/s）。若达不到 5×
     → 说明本求解器被 host 侧开销/小算例 launch 次数卡住，GPU 收益要靠**更大算例**
     而不是换 dtype。
  G3 f32 的物理漂移很小：峰值差 <5K、熔体积相对差 <3%（剂量守恒由构造保证，
     f32 eps≈1e-7 × H~1e10 J/m³ → ΔT ~ 0.2K 量级）。若超过 → A2 需要误差补偿方案。
  G4 预测 f32 试点会**直接报错**：模块里到处写死 `dtype=jnp.float64`，
     `jax_enable_x64=False` 时 JAX 拒绝 float64。这不是 bug 而是 A2 的真实工作量清单
     （把这些位点改成 dtype 参数化）。若不报错 → G4 FAIL，说明位点比我想的少。
"""
import os
import time

import jax
import jax.numpy as jnp

from amforge import geometry as G
from amforge.core.contracts import ProcessPlan
from amforge.thermal_enthalpy import solve_enthalpy_thermal, suggest_n_steps

BX, BY, BZ = 1.2e-3, 0.6e-3, 0.4e-3
V, ETA, R, P = 0.8, 0.45, 100e-6, 600.0
T_LIQ = 1723.0


def coupon(dx):
    return G.from_sdf_fn(
        lambda x: jnp.maximum(jnp.maximum(jnp.abs(x[..., 0]) - BX / 2,
                                          jnp.abs(x[..., 1]) - BY / 2),
                              jnp.abs(x[..., 2]) - BZ / 2),
        bounds=[(-BX / 2, BX / 2), (-BY / 2, BY / 2), (-BZ / 2, BZ / 2)],
        spacing=dx, name="coupon")


def plan():
    return ProcessPlan.uniform(1, modality="SLM", laser_power=P, scan_speed=V,
                               layer_thickness=BZ, hatch_spacing=1.4 * R,
                               beam_radius=R, absorption=ETA, preheat_temp=400.0)


def metrics(th, g, dx):
    solid = g.sdf < 0
    melted = (th.peak_temperature > T_LIQ) & solid
    c = g.coords()
    Ly = float(jnp.max(c[..., 1], where=melted, initial=0.0)
               - jnp.min(c[..., 1], where=melted, initial=0.0)) * 1e3
    return dict(peak=float(jnp.max(th.peak_temperature)),
                vol=float(jnp.sum(melted)) * dx ** 3 * 1e9, Ly=Ly,
                nml=int(jnp.sum(melted)))


def bench(dev, x64, dx_um, ns, reps=3):
    """同一份工作在指定设备/dtype 上跑 reps 次，取最快一次（其余次含编译）。"""
    jax.config.update("jax_enable_x64", x64)
    d = jax.devices(dev)[0] if jax.devices(dev) else None
    if d is None:
        return None
    dx = dx_um * 1e-6
    with jax.default_device(d):
        g = coupon(dx)
        pl = plan()
        th = solve_enthalpy_thermal(geometry=g, process=pl,
                                    params=dict(material="316L", n_steps=ns))
        th.peak_temperature.block_until_ready()
        best, m = 9e9, metrics(th, g, dx)
        for _ in range(reps):
            t0 = time.perf_counter()
            th2 = solve_enthalpy_thermal(geometry=g, process=pl,
                                         params=dict(material="316L", n_steps=ns))
            th2.peak_temperature.block_until_ready()
            best = min(best, time.perf_counter() - t0)
    vs = g.sdf.size * ns
    print(f"  [{dev:3s} {'f64' if x64 else 'f32'}] dx={dx_um:5.1f} ns={ns:5d} "
          f"{best:7.2f}s {vs/best/1e6:8.2f} M vox·step/s | peak={m['peak']:8.1f} "
          f"vol={m['vol']:.4f}mm³ Ly={m['Ly']:.3f}", flush=True)
    return dict(dev=dev, x64=x64, dx=dx_um, ns=ns, sec=best, vs=vs, **m)


print("设备清单:", jax.devices("cpu")[:1], jax.devices("gpu")[:1], flush=True)
CASES = [(50., 262), (25., 1045)]

print("\n=== CPU x64（基线，与 §25.2 同工况）===", flush=True)
cpu = [bench("cpu", True, d, n) for d, n in CASES]

print("\n=== GPU x64（G1）===", flush=True)
try:
    gpu64 = [bench("gpu", True, d, n) for d, n in CASES]
except Exception as e:
    gpu64 = None
    print(f"  GPU x64 异常：{type(e).__name__}: {str(e)[:180]}", flush=True)

print("\n=== GPU f32（G2/G3/G4；A2 试点）===", flush=True)
try:
    gpu32 = [bench("gpu", False, d, n) for d, n in CASES]
    g4 = False
except Exception as e:
    gpu32 = None
    g4 = True
    print(f"  GPU f32 被拒（G4 预测命中）：{type(e).__name__}: {str(e)[:200]}",
          flush=True)

print("\n=== 打分 ===", flush=True)
if cpu[0] and gpu64 and gpu64[0]:
    r = cpu[0]["sec"] / gpu64[0]["sec"]
    print(f"  G1 GPU x64 vs CPU x64 = {r:.2f}× (>2× 则我错)  => "
          f"{'PASS' if r <= 2.0 else 'FAIL'}", flush=True)
if cpu[0] and gpu32 and gpu32[0]:
    r = cpu[1]["sec"] / gpu32[1]["sec"]
    print(f"  G2 GPU f32 vs CPU x64（dx=25 档）= {r:.2f}× => "
          f"{'PASS' if r >= 20 else ('WEAK' if r >= 5 else 'FAIL')}", flush=True)
    d_pk = abs(gpu32[1]["peak"] - cpu[1]["peak"])
    d_vl = abs(gpu32[1]["vol"] - cpu[1]["vol"]) / cpu[1]["vol"]
    print(f"  G3 f32 漂移：Δpeak={d_pk:.2f}K Δvol={d_vl:.2%} => "
          f"{'PASS' if d_pk < 5 and d_vl < 0.03 else 'FAIL'}", flush=True)
print(f"  G4 f32 试点因写死 float64 被拒 => {'PASS（A2 工作量清单成立）' if g4 else 'FAIL（位点比预想少）'}",
      flush=True)

print("\n=== 模块内写死 float64 的位点数（A2 待办规模）===", flush=True)
import re
src = open("src/amforge/thermal_enthalpy.py", encoding="utf-8").read()
print(f"  thermal_enthalpy.py: float64 出现 {len(re.findall('float64', src))} 处",
      flush=True)
print("完成。", flush=True)
