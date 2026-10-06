"""A5 续 v2：GPU 吞吐饱和点、链条小算例开销墙、同网格设备一致性、体素化 1-ulp 敏感性。

背景（v1 崩溃 + 新发现，均登记在案）：
  * v1 用 n_steps=400 做 CPU 切片，被自家稳定性护栏正确拒绝（dx=12.5µm 稳定性下限
    1463 步）——护栏有效，但探针设计错了。
  * v1 的 GPU dx=12.5µm 实测 **82.99 M vox·step/s**（156849 体素 × 4180 步，7.90s），
    且 peak=2600.1K 与 §25.2 登记的 2586.6K 不符。查明原因：§25.2 / A5 的 dx 写成
    ``dx_um * 1e-6``，本次写成字面量 ``12.5e-6``；两者相差 **1 ulp**（
    ``12.5*1e-6 = 1.2499999999999999e-05 < 1.25e-05``），于是 ``ceil(BZ/dx)`` 从 32
    跳到 33，**Z 向多出一层体素**（161602 vs 156849 体素）。同一物理零件、同一函数，
    仅仅因为参数拼写不同就得到不同网格 → 这是 §25.3「空间量子化敏感性」的更尖锐版本，
    发生在**夹具构造**层面，不是度量层面。

跑前登记的证伪判据（先写后跑）：
  U1 GPU f64 吞吐在 mul 拼写的 dx=12.5µm 网格（161602 体素 × 4180 步）上
     ≥ 25 M vox·step/s，且与 v1 字面量网格的 82.99 M 相差 <15%（网格只差一层体素，
     吞吐不该变多）。
  U2 同网格、同步数下 GPU 与 CPU 的 peak 差 <0.01K、熔化体素计数**完全相同**
     → 才能声称「换设备不改结果」；若差 >0.1K 则 A5 的 f64 一致性结论要降级。
  U3 CPU f64 在 dx=12.5µm 细网格上的吞吐 ≥ 0.8 M vox·step/s（dx=25 档实测 1.35 M）。
     若 <0.8 M → 说明细网格有 cache 损失，零件尺度外推要用更小的 CPU 值。
  U4 1-ulp 网格差（Z: 33↔34 层）造成的 peak 变化 <2%、熔化体积变化 <10%。
     若 >5% → §25.3 把形态漂移归因于「量子化敏感性较小」的说法被推翻。
  U5 反演链条小算例（约 1000 体素）GPU/CPU 加速比 <2×（启动开销墙）。
     若 ≥2× → 说明 GPU 对小算例循环也有用，A5 的「GPU 帮标定级、帮不了反演循环」结论作废。
"""
import time

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from amforge import geometry as G
from amforge.core.contracts import ProcessPlan
from amforge.thermal_enthalpy import solve_enthalpy_thermal, suggest_n_steps

BX, BY, BZ = 1.2e-3, 0.6e-3, 0.4e-3
V, ETA, R, P, T_LIQ = 0.8, 0.45, 100e-6, 600.0, 1723.0


def coupon_mul(dx_um):
    """§25.2/A5 用的拼写：dx = dx_um * 1e-6（比字面量小 1 ulp）。"""
    return G.from_sdf_fn(
        lambda x: jnp.maximum(jnp.maximum(jnp.abs(x[..., 0]) - BX / 2,
                                          jnp.abs(x[..., 1]) - BY / 2),
                              jnp.abs(x[..., 2]) - BZ / 2),
        bounds=[(-BX / 2, BX / 2), (-BY / 2, BY / 2), (-BZ / 2, BZ / 2)],
        spacing=dx_um * 1e-6, name="coupon")


def coupon_lit(dx_um):
    """v1 的拼写：字面量 ``{dx}e-6``（比 mul 大 1 ulp）。"""
    return G.from_sdf_fn(
        lambda x: jnp.maximum(jnp.maximum(jnp.abs(x[..., 0]) - BX / 2,
                                          jnp.abs(x[..., 1]) - BY / 2),
                              jnp.abs(x[..., 2]) - BZ / 2),
        bounds=[(-BX / 2, BX / 2), (-BY / 2, BY / 2), (-BZ / 2, BZ / 2)],
        spacing=float(f"{dx_um}e-6"), name="coupon")


def plan():
    return ProcessPlan.uniform(1, modality="SLM", laser_power=P, scan_speed=V,
                               layer_thickness=BZ, hatch_spacing=1.4 * R,
                               beam_radius=R, absorption=ETA, preheat_temp=400.0)


def chain_fixture():
    """inverse.simulate 的默认反演夹具（0.8mm 球、dx=120µm）。"""
    g = G.from_sdf_fn(lambda x: jnp.linalg.norm(x, axis=-1) - 0.4e-3,
                      bounds=[(-0.5e-3, 0.5e-3)] * 3, spacing=120e-6,
                      name="opt-demo")
    pl = ProcessPlan.uniform(int(g.layer_count(40e-6)), modality="SLM",
                             laser_power=900.0, scan_speed=0.6,
                             layer_thickness=60e-6, hatch_spacing=120e-6,
                             beam_radius=60e-6, absorption=0.5,
                             preheat_temp=473.0, dwell_time=0.0)
    return g, pl


def measure(dev, g, pl, ns, reps=2):
    d = jax.devices(dev)[0]
    with jax.default_device(d):
        th = solve_enthalpy_thermal(geometry=g, process=pl,
                                    params=dict(material="316L", n_steps=ns))
        th.peak_temperature.block_until_ready()
        best = 9e9
        for _ in range(reps):
            t0 = time.perf_counter()
            t2 = solve_enthalpy_thermal(geometry=g, process=pl,
                                        params=dict(material="316L", n_steps=ns))
            t2.peak_temperature.block_until_ready()
            best = min(best, time.perf_counter() - t0)
        pk = float(jnp.max(t2.peak_temperature))
        nml = int(jnp.sum((t2.peak_temperature > T_LIQ) & (g.sdf < 0)))
    dx = float(g.spacing)
    vs = g.sdf.size * ns
    print(f"  [{dev:3s}] dx={dx*1e6:6.2f} nvox={g.sdf.size:7d} ns={ns:5d} "
          f"{best:7.2f}s {vs/best/1e6:8.2f} M vox·step/s {best/ns*1e3:6.2f} ms/step | "
          f"peak={pk:8.1f} nml={nml:6d} vol={nml*dx**3*1e9:.4f}mm³", flush=True)
    return dict(dev=dev, sec=best, thr=vs / best, peak=pk, nml=nml,
                vol=nml * dx ** 3 * 1e9, nvox=g.sdf.size)


pl = plan()

print("=== T1 体素化 1-ulp 敏感性（同物理零件，两种 dx 拼写；ns 各按自身 CFL 推导）===",
      flush=True)
g_mul, g_lit = coupon_mul(12.5), coupon_lit(12.5)
ns_mul, ns_lit = suggest_n_steps(g_mul, pl), suggest_n_steps(g_lit, pl)
print(f"  mul  dx={float(g_mul.spacing)!r} shape={tuple(g_mul.sdf.shape)} "
      f"nvox={g_mul.sdf.size} ns={ns_mul}", flush=True)
print(f"  lit  dx={float(g_lit.spacing)!r} shape={tuple(g_lit.sdf.shape)} "
      f"nvox={g_lit.sdf.size} ns={ns_lit}", flush=True)
r_mul = measure("gpu", g_mul, pl, ns_mul)
r_lit = measure("gpu", g_lit, pl, ns_lit)
dpk = abs(r_mul["peak"] - r_lit["peak"]) / r_mul["peak"]
dvol = abs(r_mul["vol"] - r_lit["vol"]) / r_mul["vol"]
print(f"  U4 1-ulp 网格差 → Δpeak={dpk*100:.2f}% Δvol={dvol*100:.2f}% => "
      f"{'PASS' if dpk < 0.02 and dvol < 0.10 else 'FAIL'}", flush=True)

print("\n=== U1/U2/U3 标定级 dx=12.5µm（mul 网格，与 §25.2 同）===", flush=True)
ns_run = suggest_n_steps(g_mul, pl, cfl=0.9)     # 稳定性下限之上、且 CPU 跑得完
print(f"  ns(全精度 cfl=0.35)={ns_mul}  ns(本次同网格对比 cfl=0.9)={ns_run}", flush=True)
g_full = r_mul                                     # GPU 全程 4180 步（U1）
g_same_gpu = measure("gpu", g_mul, pl, ns_run)
c_same_cpu = measure("cpu", g_mul, pl, ns_run, reps=1)
print(f"  U1 GPU 吞吐（{ns_mul} 步全程）= {g_full['thr']/1e6:.2f} M => "
      f"{'PASS' if g_full['thr'] >= 25e6 else 'FAIL'}；"
      f"与 v1 字面量网格 {r_lit['thr']/1e6:.2f} M 相差 "
      f"{abs(g_full['thr']-r_lit['thr'])/r_lit['thr']*100:.1f}% => "
      f"{'PASS' if abs(g_full['thr']-r_lit['thr'])/r_lit['thr'] < 0.15 else 'FAIL'}",
      flush=True)
dpeak = abs(g_same_gpu["peak"] - c_same_cpu["peak"])
print(f"  U2 同网格同 ns GPU vs CPU：Δpeak={dpeak:.4f}K Δnml={g_same_gpu['nml']-c_same_cpu['nml']:+d} => "
      f"{'PASS' if dpeak < 0.01 and g_same_gpu['nml'] == c_same_cpu['nml'] else 'FAIL'}",
      flush=True)
print(f"  U3 CPU 细网格吞吐 = {c_same_cpu['thr']/1e6:.2f} M vox·step/s => "
      f"{'PASS' if c_same_cpu['thr'] >= 0.8e6 else 'FAIL'}"
      f"（GPU/CPU 同网格加速比 {g_same_gpu['thr']/c_same_cpu['thr']:.1f}×）", flush=True)

print("\n=== U5 反演链条小算例（开销墙）===", flush=True)
gc, plc = chain_fixture()
nsc = suggest_n_steps(gc, plc)
print(f"  夹具 nvox={gc.sdf.size} ns={nsc}", flush=True)
gpu_c = measure("gpu", gc, plc, nsc)
cpu_c = measure("cpu", gc, plc, nsc, reps=1)
ratio = cpu_c["sec"] / gpu_c["sec"]
print(f"  U5 GPU/CPU = {ratio:.2f}×（预测 <2×）=> {'PASS' if ratio < 2 else 'FAIL'}；"
      f"ms/step GPU={gpu_c['sec']/nsc*1e3:.2f} CPU={cpu_c['sec']/nsc*1e3:.2f}", flush=True)

print("\n=== 零件尺度外推（用实测吞吐替掉 §24.3 未实测的 100×）===", flush=True)
need = 6.4e7 * 8.3e6
for label, thr in (("CPU f64（dx=12.5µm 实测）", c_same_cpu['thr']),
                   ("GPU f64（dx=25µm 档）", 16.56e6),
                   ("GPU f64（dx=12.5µm 档实测）", g_full['thr'])):
    print(f"  {label:30s} {thr/1e6:8.2f} M vox·step/s → 一次正向 = "
          f"{need/thr/3600.0:10.1f} h = {need/thr/86400.0:8.1f} 天", flush=True)
print(f"  即 GPU 相对 CPU 省 {c_same_cpu['thr']/g_full['thr']:.0f}× 墙钟，"
      f"但绝对量级仍是「天」——指数必须来自算法（多重网格/隐式+活跃网格/准稳态·本征应变）。",
      flush=True)
print("完成。", flush=True)
