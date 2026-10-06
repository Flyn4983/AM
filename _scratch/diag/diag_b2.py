"""方案 B 探索：优化器能否调稳、NSGA-II 能否产出真实前沿。

1) optimize_shape 扫 lr ∈ {0.005,0.01,0.02,0.05}，看是否能稳定把 geom_dev 降到
   baseline 以下（而非第一步发散 10x）。
2) pareto 在 _TINY(n_grid=4) vs _LIGHT(n_grid=6) 下看前沿是否塌缩成单解。
"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from amforge import geometry as G
from amforge.core.contracts import PartGeometry
from amforge.inverse import optimize_shape, pareto_optimize_geometry_process
from amforge.process import ProcessPlan, heuristic_plan, normalize_process

_GEO_KW = dict(bounds=[(-0.5e-3, 0.5e-3)] * 3, spacing=120e-6, name="opt-demo")


def _small_geo():
    return G.from_sdf_fn(lambda x: jnp.linalg.norm(x, axis=-1) - 0.4e-3, **_GEO_KW)


def _high_energy_plan(geo):
    return ProcessPlan.uniform(int(geo.layer_count(40e-6)), modality="SLM",
                               laser_power=900.0, scan_speed=0.6, layer_thickness=60e-6,
                               hatch_spacing=120e-6, beam_radius=60e-6, absorption=0.5,
                               preheat_temp=473.0, dwell_time=0.0)


geo = _small_geo()
SHP = dict(n_grid=8, max_layers=4, n_sub_cp=32, tau_activation=1e-3)

print("=" * 72)
print("EXPERIMENT 1: optimize_shape 扫 lr（看能否稳定降到 baseline 以下）")
print("=" * 72)
for lr in (0.005, 0.01, 0.02, 0.05):
    res = optimize_shape(geo, material="316L", n_steps=16, learning_rate=lr,
                         params=SHP, smoothness=0.02, verbose=False)
    gh = np.asarray(res["geom_history"])
    print(f"lr={lr:>6}: gh0={gh[0]:.5f} min={gh.min():.5f} min/gh0={gh.min()/gh[0]:.4f} "
          f"delta_max={float(jnp.max(jnp.abs(res['delta']))):.3e}")

print()
print("=" * 72)
print("EXPERIMENT 2: pareto 分辨率影响（前沿是否塌缩）")
print("=" * 72)
for tag, params in (("_TINY", dict(n_grid=4, max_layers=2, n_sub_cp=4, tau_activation=1e-3)),
                    ("_LIGHT", dict(n_grid=6, max_layers=3, n_sub_cp=8, tau_activation=1e-3))):
    res = pareto_optimize_geometry_process(geo, material="316L", params=params,
                                           algorithm="nsga2", pop_size=8, n_gen=3, seed=0,
                                           smoothness=0.02, constraint_weight=20.0,
                                           asbuilt_solver="plastic", verbose=False)
    gd = np.asarray(res["geom_dev"]); st = np.asarray(res["stress"])
    print(f"{tag}: n_solutions={len(gd)} geom_dev={np.round(gd,6)} stress={np.round(st,6)}")
    print(f"      span_g={float(np.max(gd)-np.min(gd)):.3e} span_s={float(np.max(st)-np.min(st)):.3e} "
          f"feas_min={float(np.min(res['feasible'])):.3f}")
