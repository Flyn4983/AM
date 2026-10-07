"""T2 后续：哪一套**符号约定**能让固体域（进而让熔池度量）对 dx 拼写**不敏感**？

背景：`thermal_enthalpy.py:577` 的加热/实体掩膜用严格 `sdf < 0`；A0 当年给扫描足迹
另写了一套 `_footprint`：`sdf < dx/2`（等价 cut-cell 份额 fv>0，`:322-332`）。本轮要
**用数据决定** #19 该采用哪一套，而不是猜。

跑前登记的证伪判据：
  W1 严格掩膜 `sdf<0` 的固体集合在两种 dx 拼写间**不同**（已知：翻转一层皮）。
  W2 容差掩膜 `sdf<dx/2` 的固体集合在两种拼写间**完全相同**（体素数与逐体素布尔都相等）。
     ⇒ 若成立：#19 的修法被**实测验证**（把 `_footprint` 约定同步到热域即可拼写无关）。
     ⇒ 若不成立：容差也不够，必须改**体素中心半格偏移**或显式 cut-cell 份额，登记为未解。
  W3 熔池体积在"lit + 容差"与"mul + 严格"下应接近（同一固体集合 ⇒ 同一度量口径），
     用来检验 W2 是否真的把两套口径拉齐。
  W4 三档 dx 的 vol 极差分别在"严格"与"容差"两种口径下报告 ⇒ 直接量化
     §25.3 的"~4%/加倍 形态漂移"里有多少来自**约定**（注意：解本身仍由严格掩膜算出，
     所以这是**度量口径**的贡献，不含域变化的反馈；真正的闭环要等 #19 改完重跑）。
"""
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from amforge import geometry as G
from amforge.core.contracts import ProcessPlan
from amforge.thermal_enthalpy import solve_enthalpy_thermal, suggest_n_steps

BX, BY, BZ = 1.2e-3, 0.6e-3, 0.4e-3
V, ETA, R, P, T_LIQ = 0.8, 0.45, 100e-6, 600.0, 1723.0


def _sdf(x):
    return jnp.maximum(jnp.maximum(jnp.abs(x[..., 0]) - BX / 2,
                                   jnp.abs(x[..., 1]) - BY / 2),
                       jnp.abs(x[..., 2]) - BZ / 2)


def coupon(sp):
    return G.from_sdf_fn(_sdf, bounds=[(-BX / 2, BX / 2), (-BY / 2, BY / 2),
                                       (-BZ / 2, BZ / 2)], spacing=sp, name="coupon")


def plan():
    return ProcessPlan.uniform(1, modality="SLM", laser_power=P, scan_speed=V,
                               layer_thickness=BZ, hatch_spacing=1.4 * R,
                               beam_radius=R, absorption=ETA, preheat_temp=400.0)


pl = plan()
rows = {}
for dx_um in (12.5, 25.0, 50.0):
    sps = {"mul": dx_um * 1e-6, "lit": float(f"{dx_um}e-6")}
    for tag, sp in sps.items():
        g = coupon(sp)
        ns = suggest_n_steps(g, pl)
        dx = float(g.spacing)
        strict = jnp.asarray(g.sdf) < 0.0
        tol = jnp.asarray(g.sdf) < 0.5 * dx
        th = solve_enthalpy_thermal(geometry=g, process=pl,
                                    params=dict(material="316L", n_steps=ns))
        th.peak_temperature.block_until_ready()
        melted = th.peak_temperature > T_LIQ
        rows[(dx_um, tag)] = dict(
            nvox=g.sdf.size, ns=int(ns),
            solid_strict=int(strict.sum()), solid_tol=int(tol.sum()),
            vol_strict=float((melted & strict).sum()) * dx ** 3 * 1e9,
            vol_tol=float((melted & tol).sum()) * dx ** 3 * 1e9,
            peak=float(jnp.max(th.peak_temperature)),
            mask_tol_bool=tol, mask_strict_bool=strict,
        )
        print(f"dx={dx_um:5.1f} [{tag}] nvox={g.sdf.size} ns={ns} "
              f"solid(sdf<0)={int(strict.sum())} solid(sdf<dx/2)={int(tol.sum())} "
              f"peak={rows[(dx_um, tag)]['peak']:.4f} "
              f"vol_strict={rows[(dx_um, tag)]['vol_strict']:.6f} "
              f"vol_tol={rows[(dx_um, tag)]['vol_tol']:.6f}", flush=True)

print("\n=== W1/W2：固体集合的拼写不变性 ===", flush=True)
for dx_um in (12.5, 25.0, 50.0):
    a, b = rows[(dx_um, "mul")], rows[(dx_um, "lit")]
    d_strict = int((a["mask_strict_bool"] ^ b["mask_strict_bool"]).sum())
    d_tol = int((a["mask_tol_bool"] ^ b["mask_tol_bool"]).sum())
    print(f"  dx={dx_um:5.1f}: 严格掩膜差异={d_strict} 个 | "
          f"容差掩膜差异={d_tol} 个 | 计数 {a['solid_strict']}/{b['solid_strict']} "
          f"→ {a['solid_tol']}/{b['solid_tol']}")
    print(f"    W1 {'PASS(已知敏感)' if d_strict else 'FAIL'} | "
          f"W2 {'PASS ⇒ _footprint 约定可直接用于 #19' if d_tol == 0 else 'FAIL ⇒ 容差不足，需半格偏移/份额'}")

print("\n=== W4：三档 vol 极差按口径对比 ===", flush=True)
for key in ("vol_strict", "vol_tol"):
    vals = [rows[(d, "mul")][key] for d in (12.5, 25.0, 50.0)]
    rng = (max(vals) - min(vals)) / min(vals)
    print(f"  {key}: mul 拼写三档 = " + " / ".join(f"{v:.6f}" for v in vals) +
          f"  极差={rng * 100:.2f}%")
    vals_l = [rows[(d, "lit")][key] for d in (12.5, 25.0, 50.0)]
    rng_l = (max(vals_l) - min(vals_l)) / min(vals_l)
    print(f"  {key}: lit 拼写三档 = " + " / ".join(f"{v:.6f}" for v in vals_l) +
          f"  极差={rng_l * 100:.2f}%")
