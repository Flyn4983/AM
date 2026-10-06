"""A5 续：GPU 吞吐随规模的**饱和点** + 链条小算例的**开销墙**（CPU only 授权已解除，GPU 可用）。

跑前登记的证伪判据：
  S1 GPU f64 吞吐随网格增大继续上升：dx=12.5µm（161602 体素 × 4180 步 = 6.75e8
     voxel-step）≥ **25 M vox·step/s**（A5 在 2.3e7 voxel-step 上是 16.56 M）。
     若 <20 M → 说明已近饱和，零件尺度外推要用饱和值而不是 16.6 M。
  S2 每步固定开销 ≈1.3 ms（A5 反推：dx=50 档 3250 体素×262 步只用 1.01s ⇒ 3.9ms/步
     里绝大部分是 launch 开销）→ 反演链条的 1000 体素小算例在 GPU 上**不会**比 CPU 快
     超过 2×：GPU 帮标定级（细网格长行程），帮不了小算例循环。
  S3 同 f64 下 GPU 与 CPU 的打印物理量一致（A5 已见 dx=25/50 两档 peak/vol/Ly 完全相同）
     → GPU 迁移数值安全；若 dx=12.5 档出现差异，则不能声称"换设备不改结果"。
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


def chain_fixture():
    """inverse.simulate 的默认反演夹具（0.8mm 球、dx=120µm、1000 体素、27 层）。"""
    g = G.from_sdf_fn(lambda x: jnp.linalg.norm(x, axis=-1) - 0.4e-3,
                      bounds=[(-0.5e-3, 0.5e-3)] * 3, spacing=120e-6,
                      name="opt-demo")
    pl = ProcessPlan.uniform(int(g.layer_count(40e-6)), modality="SLM",
                             laser_power=900.0, scan_speed=0.6,
                             layer_thickness=60e-6, hatch_spacing=120e-6,
                             beam_radius=60e-6, absorption=0.5,
                             preheat_temp=473.0, dwell_time=0.0)
    return g, pl


def timeit(dev, g, pl, ns, reps=2):
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
        vol = float(jnp.sum((t2.peak_temperature > T_LIQ) & (g.sdf < 0))) \
        * float(g.spacing) ** 3 * 1e9
    vs = g.sdf.size * ns
    print(f"  [{dev:3s}] nvox={g.sdf.size:7d} ns={ns:5d} {best:7.2f}s "
          f"{vs/best/1e6:8.2f} M vox·step/s  {best/ns*1e3:6.2f} ms/step | "
          f"peak={pk:8.1f} vol={vol:.4f}mm³", flush=True)
    return dict(dev=dev, sec=best, thr=vs / best, peak=pk, vol=vol,
                ms_step=best / ns * 1e3)


print("=== S1/S3：标定级 dx=12.5µm（161602 体素 × 4180 步）===", flush=True)
g12 = coupon(12.5e-6)
pl = plan()
ns12 = suggest_n_steps(g12, pl)
print(f"  suggest_n_steps(dx=12.5µm) = {ns12}", flush=True)
gpu12 = timeit("gpu", g12, pl, ns12)
cpu12 = timeit("cpu", g12, pl, 400)          # CPU 只跑 400 步切片，按吞吐外推
print(f"  CPU 吞吐（400 步切片）= {cpu12['thr']/1e6:.2f} M vox·step/s；"
      f"全 {ns12} 步外推 = {(g12.sdf.size*ns12)/cpu12['thr']:.0f}s", flush=True)
print(f"  S1 GPU≥25M => {'PASS' if gpu12['thr']>=25e6 else 'FAIL'}"
      f"（实测 {gpu12['thr']/1e6:.2f} M）  "
      f"S3 物理一致 => {'PASS' if abs(gpu12['peak']-2586.6)<0.05 else 'CHECK'}"
      f"（GPU {gpu12['peak']:.1f}K vs §25.2 CPU {2586.6}K）", flush=True)

print("\n=== S2：链条小算例（1000 体素 × 681 步）===", flush=True)
gc, plc = chain_fixture()
nsc = suggest_n_steps(gc, plc)
print(f"  反演夹具 suggest_n_steps = {nsc}", flush=True)
gpu_c = timeit("gpu", gc, plc, nsc)
cpu_c = timeit("cpu", gc, plc, nsc)
r = cpu_c["sec"] / gpu_c["sec"]
print(f"  S2 GPU/CPU 加速比 = {r:.2f}×（预测 <2×，开销墙）=> "
      f"{'PASS' if r < 2 else 'FAIL'}；ms/step GPU={gpu_c['ms_step']:.2f} "
      f"CPU={cpu_c['ms_step']:.2f}", flush=True)

print("\n=== 零件尺度外推（用实测饱和吞吐，替掉 §24.3 未实测的 100×）===", flush=True)
need = 6.4e7 * 8.3e6           # 10mm 立方 @25µm × 8.3e6 步
for label, thr in (("CPU f64（本次 400 步切片）", cpu12['thr']),
                   ("GPU f64（dx=25 档）", 16.56e6),
                   ("GPU f64（dx=12.5 档实测）", gpu12['thr'])):
    print(f"  {label:28s} {thr/1e6:7.2f} M vox·step/s → 一次正向 = "
          f"{need/thr/3600.0:9.1f} h = {need/thr/86400.0:7.1f} 天", flush=True)
print("完成。", flush=True)
