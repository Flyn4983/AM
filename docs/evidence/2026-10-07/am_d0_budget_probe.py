"""D0 预算探针（CPU，无 GPU）：把"诚实调度"的代价在真实测试夹具上量出来。

用途：为链条档（demo）选出既不赔 dt 精度、又能让 pytest 跑得完的预算/ CFL 组合。
只调用 chain_schedule()（纯静态算式，不解热传导方程），所以本脚本秒级完成。
"""
import warnings

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from amforge import geometry as G                     # noqa: E402
from amforge.core.registry import get_solver          # noqa: E402
from amforge.thermal_enthalpy import chain_schedule   # noqa: E402

# §25.7 实测成本律：CPU f64 显式热解 0.75–2.40 M voxel-step/s（取保守 1.0 M 估墙钟）
CPU_RATE = 1.0e6


def _plan(geo):
    return get_solver("process.heuristic").fn(geometry=geo, params=None)


def _show(tag, geo, plan, *, cfl, policy, over_budget, max_steps, max_voxel_steps):
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            s = chain_schedule(geo, plan, cfl=cfl, resolution_policy=policy,
                               over_budget=over_budget, max_steps=max_steps,
                               max_voxel_steps=max_voxel_steps)
    except Exception as e:  # noqa: BLE001  探针需要看到全部失败模式
        print(f"{tag:34s} FAIL {type(e).__name__}: {str(e)[:150]}")
        return
    n, nvox = s["n_steps"], int(jnp.asarray(geo.sdf).size)
    vs = n * nvox
    print(f"{tag:34s} n_steps={n:8d} vox={nvox:7d} voxel_steps={vs:.2e} "
          f"cpu~{vs / CPU_RATE:7.1f}s t_bound={s['exposure_bound_s']:.4e}s "
          f"path={s['path_bound_m']:.4f}m v_lo={s['bounds']['scan_speed'][0]:.3f} "
          f"hatch_lo={s['bounds']['hatch_spacing'][0]*1e6:.0f}µm "
          f"lt_lo={s['bounds']['layer_thickness'][0]*1e6:.0f}µm "
          f"r_lo={s['bounds']['beam_radius'][0]*1e6:.0f}µm "
          f"warn={[str(x.message)[:28] for x in w]}")


def fixtures():
    out = {}
    # e2e_chain：0.9mm 球，dx=50µm（当前 27.6GB RSS 的那个）
    out["e2e_sphere_50um"] = G.from_sdf_fn(
        lambda x: jnp.linalg.norm(x, axis=-1) - 0.45e-3,
        bounds=[(-0.6e-3, 0.6e-3)] * 3, spacing=50e-6, name="e2e_sphere")
    # 同几何粗化到 100µm / 150µm：代价 ∝ dx^-5（n_steps∝dx^-2, 体素∝dx^-3）
    out["e2e_sphere_100um"] = G.from_sdf_fn(
        lambda x: jnp.linalg.norm(x, axis=-1) - 0.45e-3,
        bounds=[(-0.6e-3, 0.6e-3)] * 3, spacing=100e-6, name="e2e_sphere_c")
    out["e2e_sphere_150um"] = G.from_sdf_fn(
        lambda x: jnp.linalg.norm(x, axis=-1) - 0.45e-3,
        bounds=[(-0.6e-3, 0.6e-3)] * 3, spacing=150e-6, name="e2e_sphere_cc")
    # dimensional_optimization：0.8mm 球，dx=120µm
    out["dim_small_120um"] = G.from_sdf_fn(
        lambda x: jnp.linalg.norm(x, axis=-1) - 0.4e-3,
        bounds=[(-0.5e-3, 0.5e-3)] * 3, spacing=120e-6, name="opt-demo")
    # powderbed：6^3 全实体，dx=100µm
    import numpy as np
    out["powderbed_cube_100um"] = G.PartGeometry(
        sdf=-np.ones((6, 6, 6), dtype=np.float64), origin=jnp.zeros(3),
        spacing=1e-4, dim=3, name="cube")
    return out


def main():
    fx = fixtures()
    for name, geo in fx.items():
        plan = _plan(geo)
        print(f"\n### {name}  shape={geo.shape} dx={float(geo.spacing)*1e6:.0f}µm "
              f"工艺: P={float(jnp.mean(plan.laser_power)):.1f}W v={float(jnp.mean(plan.scan_speed)):.3f}m/s "
              f"hatch={float(jnp.mean(plan.hatch_spacing))*1e6:.0f}µm "
              f"lt={float(jnp.mean(plan.layer_thickness))*1e6:.0f}µm "
              f"r={float(jnp.mean(plan.beam_radius))*1e6:.0f}µm")
        # ① 完全无预算：诚实调度本身的代价
        _show("  no-budget cfl=0.35", geo, plan, cfl=0.35, policy="demo",
              over_budget="raise", max_steps=10 ** 9, max_voxel_steps=None)
        _show("  no-budget cfl=1.00", geo, plan, cfl=1.0, policy="demo",
              over_budget="raise", max_steps=10 ** 9, max_voxel_steps=None)
        # ② 链条档候选：预算 2e6 / 8e6 voxel-step，pin
        for mvs in (2_000_000, 8_000_000):
            _show(f"  pin mvs={mvs:.0e} cfl=0.35", geo, plan, cfl=0.35,
                  policy="demo", over_budget="pin", max_steps=200000,
                  max_voxel_steps=mvs)
            _show(f"  pin mvs={mvs:.0e} cfl=1.00", geo, plan, cfl=1.0,
                  policy="demo", over_budget="pin", max_steps=200000,
                  max_voxel_steps=mvs)


if __name__ == "__main__":
    main()
