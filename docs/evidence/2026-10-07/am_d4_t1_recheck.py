"""D4 复测（T1 闭合 + §25.2 登记值刷新）——**只测正确性，不测吞吐**。

为什么这轮不测吞吐：同机另有全量验收 pytest 在跑（load avg ~9），任何 ms/步、
M vox·step/s 都会被打脏。吞吐表留到树冻结解除后单独跑（任务 #16 的 N1 步）。
夹具逐字沿用 `docs/evidence/2026-10-06/am_a5_scale2.py`（BX/BY/BZ、V=0.8、R=100µm、
P=600W、ETA=0.45、preheat=400K、material=316L、两种 dx 拼写 `dx_um*1e-6` 与
`float(f"{dx_um}e-6")`），以便与 §25.2/§25.7 已登记值直接对表。

跑前登记的证伪判据（先写后跑）：
  V1 对 dx ∈ {12.5, 25, 50} µm：两种拼写的 `sdf.shape` **完全相同**、`sdf.size` 相同。
     若仍有任一档 shape 不同 ⇒ **D4 未闭合**，本探针判 FAIL（T1 仍开放）。
  V2 同档两种拼写的 peak 与熔化体积 **逐位相同**（同网格 ⇒ 同计算 ⇒ 应无差）。
     若 |Δpeak|>0 ⇒ 说明除 ceil 之外还有拼写敏感源，登记新缺陷。
  V3 `suggest_n_steps` 两种拼写给出同步数（步数由 dx 推 CFL，网格一致则必一致）。
  V4 登记新值与旧值对照：旧 §25.2 dx=12.5 = 161602 体素 / 4180 步 / peak 2586.6 K /
     vol 0.0745 mm³；D4 的容差吸附预期把 Z 层数从 33 降回 32 ⇒ 体素数应**变小**、
     peak/vol 随之改变。若新体素数**不减反增**或与两种拼写都不同，说明吸附方向写反。
  V5 三档 peak 极差与熔池形态照实重登（旧登记：极差 3.13%，Lx 6.38%、vol 7.73% 仍漂移）；
     本探针**不声称**形态收敛，只替换数字。
"""
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from amforge import geometry as G
from amforge.core.contracts import ProcessPlan
from amforge.thermal_enthalpy import solve_enthalpy_thermal, suggest_n_steps

BX, BY, BZ = 1.2e-3, 0.6e-3, 0.4e-3
V, ETA, R, P, T_LIQ = 0.8, 0.45, 100e-6, 600.0, 1723.0
OLD = {12.5: dict(nvox=161602, ns=4180, peak=2586.6, vol=0.0745)}


def _sdf(x):
    return jnp.maximum(jnp.maximum(jnp.abs(x[..., 0]) - BX / 2,
                                   jnp.abs(x[..., 1]) - BY / 2),
                       jnp.abs(x[..., 2]) - BZ / 2)


def coupon_mul(dx_um):
    return G.from_sdf_fn(_sdf, bounds=[(-BX / 2, BX / 2), (-BY / 2, BY / 2),
                                       (-BZ / 2, BZ / 2)],
                         spacing=dx_um * 1e-6, name="coupon")


def coupon_lit(dx_um):
    return G.from_sdf_fn(_sdf, bounds=[(-BX / 2, BX / 2), (-BY / 2, BY / 2),
                                       (-BZ / 2, BZ / 2)],
                         spacing=float(f"{dx_um}e-6"), name="coupon")


def plan():
    return ProcessPlan.uniform(1, modality="SLM", laser_power=P, scan_speed=V,
                               layer_thickness=BZ, hatch_spacing=1.4 * R,
                               beam_radius=R, absorption=ETA, preheat_temp=400.0)


def solve(g, pl, ns):
    th = solve_enthalpy_thermal(geometry=g, process=pl,
                                params=dict(material="316L", n_steps=ns))
    th.peak_temperature.block_until_ready()
    dx = float(g.spacing)
    m = th.peak_temperature > T_LIQ
    if m.ndim != 3:
        raise SystemExit(f"peak_temperature 形状意外：{m.shape}（本探针按 3D 网格写）")
    nml = int(jnp.sum(m & (g.sdf < 0)))
    # Lx = 沿 x 轴（道方向）有多少个体素列出现过熔化
    lx = int(jnp.sum(m.any(axis=(1, 2))))
    return (float(jnp.max(th.peak_temperature)), nml,
            nml * dx ** 3 * 1e9, lx)


pl = plan()
fails = []
print("=== D4/T1 复测：两种 dx 拼写在同一物理零件上是否同网格、同结果 ===", flush=True)
for dx_um in (12.5, 25.0, 50.0):
    gm, gl = coupon_mul(dx_um), coupon_lit(dx_um)
    nsm, nsl = suggest_n_steps(gm, pl), suggest_n_steps(gl, pl)
    same_shape = tuple(gm.sdf.shape) == tuple(gl.sdf.shape)
    print(f"\n-- dx={dx_um} µm --")
    print(f"  mul dx={float(gm.spacing)!r} shape={tuple(gm.sdf.shape)} ns={nsm}")
    print(f"  lit dx={float(gl.spacing)!r} shape={tuple(gl.sdf.shape)} ns={nsl}")
    print(f"  V1 shape {'相同 ⇒ PASS' if same_shape else '不同 ⇒ FAIL（D4 未闭合）'}")
    print(f"  V3 n_steps {'相同 ⇒ PASS' if nsm == nsl else '不同 ⇒ FAIL'}")
    if not same_shape:
        fails.append(f"V1 dx={dx_um} shape 不同")
    if nsm != nsl:
        fails.append(f"V3 dx={dx_um} n_steps 不同")
        continue
    pm, nm, vm, lx_m = solve(gm, pl, nsm)
    p_l, n_l, v_l, lx_l = solve(gl, pl, nsl)
    print(f"  mul peak={pm:.4f} nml={nm} vol={vm:.6f} Lx={lx_m}")
    print(f"  lit peak={p_l:.4f} nml={n_l} vol={v_l:.6f} Lx={lx_l}")
    print(f"  V2 Δpeak={abs(pm-p_l):.3e} Δvol={abs(vm-v_l):.3e} "
          f"⇒ {'PASS' if pm == p_l and vm == v_l else 'FAIL'}")
    if pm != p_l or vm != v_l:
        fails.append(f"V2 dx={dx_um} 拼写间结果不同")
    if dx_um in OLD:
        o = OLD[dx_um]
        print(f"  V4 旧登记 nvox={o['nvox']} ns={o['ns']} peak={o['peak']} "
              f"vol={o['vol']} → 新 nvox={gm.sdf.size} ns={nsm} "
              f"peak={pm:.1f} vol={vm:.4f}"
              f"（体素数增删：{gm.sdf.size - o['nvox']:+d}）")
        if gm.sdf.size > o["nvox"]:
            fails.append(f"V4 dx={dx_um} 新网格比旧登记更大（{gm.sdf.size}>{o['nvox']}），"
                         f"容差吸附方向需复核")

print("\n=== 记分 ===", flush=True)
print("FAILS:", fails if fails else "无 ⇒ V1/V2/V3/V4 全 PASS", flush=True)
