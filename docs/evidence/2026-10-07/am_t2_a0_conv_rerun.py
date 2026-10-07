"""#16 的一半 + 警告文案的取数：用 **#19 后的口径**重跑 A0 四档收敛表（旧表见
`docs/evidence/2026-10-06/am_a0_conv.log`，那是 `sdf<0` 严格掩膜下的数）。

跑前登记（本探针不设"通过线"，只回答两个已经写在代码/文档里的具体数字该不该改）：
  W1 `thermal_enthalpy.py` 的欠分辨警告写着"实测 dx=r 档熔宽偏低 ~45%、熔体积小 ~36%"。
     这两个数来自 2026-10-06 旧口径 ⇒ 量出 #19 后**同定义**（Lx/Ly=熔化体素集合沿该轴的外包络、
     vol=计数×dx³；定义逐字继承 `am_a0_conv.py:16-18`）的对应比值。若与 45%/36% 差出
     ≥5 个百分点，则警告文案必须换数并注明出处。
  W2 §25.2/§25.7 的 J3 表（峰值 2589.4/2670.3/2617.8/2586.6K）同样是旧口径 ⇒ 本表给出新口径
     四档峰值，供 #16 替换；两口径之差在**同一 run** 里同时打印，所以"旧数不可复现"这句
     不需要再靠记忆支撑。
阳性对照：同一 run 下 `sdf<0` 与 `solid_mask` 的熔池集合异或必须**非零**（δ=0 刀锋档），
否则说明本探针看不见口径差 ⇒ W1/W2 的结论不可用。
"""
import time

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from amforge import geometry as G
from amforge.core.contracts import ProcessPlan, solid_mask, solid_weight
from amforge.materials import get_material
from amforge.thermal_enthalpy import solve_enthalpy_thermal, suggest_n_steps

TL = float(get_material("316L").T_liquidus)
BX, BY, BZ = 1.2e-3, 0.6e-3, 0.4e-3
R = 100e-6


def coupon(dx):
    return G.from_sdf_fn(
        lambda x: jnp.maximum(jnp.maximum(jnp.abs(x[..., 0]) - BX / 2,
                                          jnp.abs(x[..., 1]) - BY / 2),
                              jnp.abs(x[..., 2]) - BZ / 2),
        bounds=[(-BX / 2, BX / 2), (-BY / 2, BY / 2), (-BZ / 2, BZ / 2)],
        spacing=dx, name="c")


def plan():
    return ProcessPlan.uniform(1, modality="SLM", laser_power=600.0, scan_speed=0.8,
                               layer_thickness=BZ, hatch_spacing=1.4 * R, beam_radius=R,
                               absorption=0.45, preheat_temp=400.0)


def extent(axis_coords, sel):
    """与 am_a0_conv.py:16-17 逐字同定义（initial=0.0 一并继承，保证与旧表可比）。"""
    return float(jnp.max(axis_coords, where=sel, initial=0.0)
                 - jnp.min(axis_coords, where=sel, initial=0.0)) * 1e3


rows = {}
for model in ("integrated", "point"):
    for dx in (100e-6, 50e-6, 25e-6, 12.5e-6):
        g, p = coupon(dx), plan()
        ns = int(suggest_n_steps(g, p))
        t0 = time.perf_counter()
        th = solve_enthalpy_thermal(geometry=g, process=p,
                                    params=dict(material="316L", source_model=model,
                                                n_steps=ns))
        el = time.perf_counter() - t0
        pk, c = th.peak_temperature, g.coords()
        mt = pk > TL
        mk_new, mk_old = solid_mask(g.sdf) > 0.5, g.sdf < 0.0
        sel = mt & mk_new
        fv = solid_weight(g.sdf, dx)
        n_new, n_old = int(jnp.sum(sel)), int(jnp.sum(mt & mk_old))
        rec = dict(ns=ns, peak=float(jnp.max(pk)), n_new=n_new, n_old=n_old,
                   xor=int(jnp.sum(sel ^ (mt & mk_old))),
                   Lx=extent(c[..., 0], sel), Ly=extent(c[..., 1], sel),
                   vol=n_new * dx ** 3 * 1e9,
                   vol_fv=float(jnp.sum(jnp.where(mt, fv, 0.0))) * dx ** 3 * 1e9)
        rows[(model, dx * 1e6)] = rec
        print(f"[{model:10s}] dx={dx*1e6:6.1f} ns={ns:5d} {el:6.2f}s "
              f"peak={rec['peak']:8.1f} nml_new={n_new:6d} nml_old={n_old:6d} "
              f"xor={rec['xor']:5d} Lx={rec['Lx']:6.3f} Ly={rec['Ly']:6.3f} "
              f"vol={rec['vol']:8.4f}mm³ vol_fv={rec['vol_fv']:8.4f}mm³", flush=True)

print("\n=== W1：dx=r(100µm) 相对最细档(12.5µm)的偏低比（新口径，同 model 内比） ===", flush=True)
for model in ("integrated", "point"):
    coarse, fine = rows[(model, 100.)], rows[(model, 12.5)]
    print(f"  [{model:10s}] 熔宽 Ly 偏低 {(1 - coarse['Ly'] / fine['Ly']) * 100:6.2f}%（旧文案 45%）"
          f"  计数熔体积偏低 {(1 - coarse['vol'] / fine['vol']) * 100:6.2f}%（旧文案 36%）"
          f"  fv 加权熔体积偏低 {(1 - coarse['vol_fv'] / fine['vol_fv']) * 100:6.2f}%", flush=True)

print("\n=== W2：新口径四档峰值 vs §25.2 旧表（integrated） ===", flush=True)
old = {100.: 2589.4, 50.: 2670.3, 25.: 2617.8, 12.5: 2586.6}
for k in (100., 50., 25., 12.5):
    new = rows[("integrated", k)]["peak"]
    print(f"  dx={k:6.1f}µm  旧口径峰值={old[k]:8.1f}K  新口径={new:8.1f}K  "
          f"差={100 * (new - old[k]) / old[k]:+6.2f}%", flush=True)

bad = [k for k, v in rows.items() if v["xor"] == 0]
print(f"\n阳性对照（xor 非零）：xor==0 的档 = {bad if bad else '无 ⇒ 探针看得见口径差'}", flush=True)
