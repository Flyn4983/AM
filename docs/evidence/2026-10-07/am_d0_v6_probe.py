"""D0 修复探针 V6（CPU）：验证 (1) 只对 laser_power 求导时档位仍能自动钉住；
(2) demo 档体素预算把昂贵网格压进闸门或给出可执行的报错；(3) 结果有限、梯度有限。"""
import time
import warnings

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from amforge import geometry as G                            # noqa: E402
from amforge.core.contracts import PartGeometry              # noqa: E402
from amforge.inverse import simulate, thermal_tier, tier_bounds  # noqa: E402
from amforge.thermal_enthalpy import chain_schedule          # noqa: E402


def _geo_sphere(r, spacing, half):
    return G.from_sdf_fn(lambda x: jnp.linalg.norm(x, axis=-1) - r,
                         bounds=[(-half, half)] * 3, spacing=spacing, name="p")


def case(tag, geo, plan, **kw):
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            s = chain_schedule(geo, plan, **kw)
    except Exception as e:  # noqa: BLE001
        print(f"{tag}: FAIL {type(e).__name__}: {str(e)[:120]}")
        return None
    nvox = int(geo.sdf.size)
    print(f"{tag}: n_steps={s['n_steps']} vox={nvox} voxel_steps={s['n_steps']*nvox:.2e} "
          f"v_lo={s['bounds']['scan_speed'][0]:.3f} warns={len(w)}")
    return s


def main():
    from amforge.core.registry import get_solver
    dim_geo = _geo_sphere(0.4e-3, 120e-6, 0.5e-3)
    e2e_geo = _geo_sphere(0.45e-3, 50e-6, 0.6e-3)
    e2e_c150 = _geo_sphere(0.45e-3, 150e-6, 0.6e-3)
    e2e_c030 = _geo_sphere(0.15e-3, 50e-6, 0.2e-3)
    plans = {n: get_solver("process.heuristic").fn(geometry=g, params=None)
             for n, g in (("dim", dim_geo), ("e2e50", e2e_geo),
                          ("e2e150", e2e_c150), ("e2e030", e2e_c030))}

    print("== 1) demo 档闸门 5e6 voxel-step ==")
    for n, g in (("dim", dim_geo), ("e2e50", e2e_geo), ("e2e150", e2e_c150),
                 ("e2e030", e2e_c030)):
        case(f"  {n} pin5e6", g, plans[n], resolution_policy="demo",
             over_budget="pin", max_voxel_steps=5_000_000)

    print("\n== 2) 只对 laser_power 求导：thermal_tier 是否自动钉档 ==")
    p = thermal_tier({}, dim_geo, plans["dim"], material="316L")
    print("  tier keys:", sorted(p.get("thermal", {})), "bounds?", bool(tier_bounds(p)))

    def loss(P):
        pl = plans["dim"].replace(laser_power=P)
        out = simulate(dim_geo, pl, material="316L",
                       params={"n_grid": 6, "max_layers": 2, "n_sub_cp": 4,
                               "tau_activation": 1e-3})
        return out["verdict"].strength_safety_factor

    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v, g = jax.value_and_grad(loss)(jnp.array(300.0))
    print(f"  value={float(v):.4f} grad={float(g):.3e} finite={np.isfinite([float(v), float(g)]).all()} "
          f"{time.time()-t0:.1f}s")

    print("\n== 3) e2e 50µm/0.9mm 在闸门下的报错信息 ==")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            simulate(e2e_geo, plans["e2e50"], material="316L",
                     params={"n_grid": 6, "max_layers": 2, "n_sub_cp": 4})
        print("  没报错（意外）")
    except Exception as e:  # noqa: BLE001
        print(f"  {type(e).__name__}: {str(e)[:400]}")


if __name__ == "__main__":
    main()
