"""取数脚本（**不设判据**）：复刻红灯测试 test_meltpool_converges_across_beam_resolving_grids
在 **δ=0（刀锋对齐，即测试本身的对齐）** 下的两个观测量，并用三种实体口径各算一遍熔体积，
以便把「测试断言里的数」与「#19 后的口径」分栏写清。

为什么需要它（跑前登记）：
  红灯测试的 `melted = (peak > T_liq) & (g.sdf < 0)` 用的是**旧严格掩膜**，而 #19 后规范口径是
  `solid_mask`（容差 1e-12）。要改测试的 docstring/掩膜写法，必须先量出「换口径对这个观测量
  到底有没有影响、影响几个体素」——不许凭"应当只影响整数对齐的面"这句话就下结论。
  阳性对照：同时打印 `sdf < 0` 与 `solid_mask` 的**全域**异或体素数（此处必然非零，因为试片
  边界恰在 sdf==0 的刀锋上）；若熔池集合的异或为 0 而全域异或也为 0，则说明我这个探针根本没
  触到差异位（探针哑火），数字不可用。
"""
import jax.numpy as jnp
import numpy as np
from amforge.geometry import from_sdf_fn
from amforge.core.contracts import solid_mask, solid_weight, ProcessPlan
from amforge.materials import get_material
from amforge.thermal_enthalpy import solve_enthalpy_thermal

EXT = (1.2e-3, 0.6e-3, 0.4e-3)


def coupon(dx):
    bx, by, bz = EXT
    return from_sdf_fn(
        lambda x: jnp.maximum(jnp.maximum(jnp.abs(x[..., 0]) - bx / 2,
                                          jnp.abs(x[..., 1]) - by / 2),
                              jnp.abs(x[..., 2]) - bz / 2),
        bounds=[(-bx / 2, bx / 2), (-by / 2, by / 2), (-bz / 2, bz / 2)],
        spacing=dx, name="coupon")


pl = ProcessPlan.uniform(1, modality="SLM", laser_power=600.0, scan_speed=0.8,
                         layer_thickness=EXT[2], hatch_spacing=1.4 * 100e-6,
                         beam_radius=100e-6, absorption=0.45, preheat_temp=400.0)
mat = get_material("316L")

rows = {}
for dx_um in (50., 25.):
    dx = dx_um * 1e-6
    g = coupon(dx)
    th = solve_enthalpy_thermal(geometry=g, process=pl, params={"material": "316L"})
    liq = np.asarray(th.peak_temperature) > float(mat.T_liquidus)
    s = np.asarray(g.sdf, dtype=np.float64)
    mk_new = np.asarray(solid_mask(g.sdf)) > 0.5
    mk_old = s < 0.0
    fv = np.asarray(solid_weight(g.sdf, dx))
    cell_mm3 = dx ** 3 * 1e9
    # 全域异或（阳性对照：必须非零，否则探针哑火）
    xor_all = int(np.sum(mk_new ^ mk_old))
    a = liq & mk_old          # 测试现口径
    b = liq & mk_new          # #19 规范口径
    c = fv > 0.0              # 份额口径（fv>0 == solid_weight 支撑集）
    n_a, n_b, n_c = int(np.sum(a)), int(np.sum(b)), int(np.sum(liq & c))
    vm = float(np.sum(np.where(liq, fv, 0.0))) * cell_mm3
    rows[dx_um] = dict(pk=float(np.asarray(th.peak_temperature).max()),
                       vol_old=n_a * cell_mm3, vol_mask=n_b * cell_mm3, n_a=n_a, n_b=n_b)
    print(f"  dx={dx_um:5.1f}µm  峰值={rows[dx_um]['pk']:9.2f}K  "
          f"熔体积(sdf<0)={n_a:6d}体素/{n_a*cell_mm3:.5f}mm³  "
          f"熔体积(solid_mask)={n_b:6d}/{n_b*cell_mm3:.5f}mm³  "
          f"熔体积(fv>0∩计数)={n_c}/{vm:.5f}mm³(fv加权)  "
          f"| 熔池集合异或 old^new={int(np.sum(a ^ b))}  全域异或={xor_all}", flush=True)

print("\n=== 相邻档对差（测试的两个断言量） ===", flush=True)
for name, key in (("峰值", "pk"), ("熔体积(sdf<0)", "vol_old"), ("熔体积(solid_mask)", "vol_mask")):
    va, vb = rows[50.][key], rows[25.][key]
    print(f"  {name:18s} 50µm={va:.6f}  25µm={vb:.6f}  相对差={abs(va - vb) / max(abs(va), abs(vb)) * 100:6.2f}%",
          flush=True)
