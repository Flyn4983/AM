"""方案 B 实证：用真实入口复现 3 个失败用例，打印实际可达的收缩量/前沿跨度。

不改动任何源码/断言，仅调用 optimize_shape / optimize_geometry_process /
pareto_optimize_geometry_process，打印 geom_history 与 Pareto 目标数组，
为断言重标定提供实测锚点。
"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from amforge import geometry as G
from amforge.process import heuristic_plan, normalize_process
from amforge.core.contracts import PartGeometry
from amforge.inverse import (
    optimize_shape, optimize_geometry_process,
    pareto_optimize_geometry_process,
)

_GEO_KW = dict(bounds=[(-0.5e-3, 0.5e-3)] * 3, spacing=120e-6, name="opt-demo")


def _small_geo():
    return G.from_sdf_fn(lambda x: jnp.linalg.norm(x, axis=-1) - 0.4e-3, **_GEO_KW)


def _high_energy_plan(geo):
    return __import__("amforge.process", fromlist=["ProcessPlan"]).ProcessPlan.uniform(
        int(geo.layer_count(40e-6)), modality="SLM",
        laser_power=900.0, scan_speed=0.6, layer_thickness=60e-6,
        hatch_spacing=120e-6, beam_radius=60e-6, absorption=0.5,
        preheat_temp=473.0, dwell_time=0.0,
    )


print("=" * 72)
print("TEST 1: optimize_shape  (threshold 旧: min(gh) < gh[0]*0.5)")
print("=" * 72)
geo = _small_geo()
res = optimize_shape(geo, material="316L", n_steps=16, learning_rate=0.05,
                     params=dict(n_grid=8, max_layers=4, n_sub_cp=32, tau_activation=1e-3),
                     smoothness=0.02, verbose=False)
gh = np.asarray(res["geom_history"])
print("geom_history:", np.round(gh, 6))
print(f"gh[0]={gh[0]:.6f}  min(gh)={gh.min():.6f}  min/gh0={gh.min()/gh[0]:.4f}")
print(f"decreased? min<gh0 : {gh.min() < gh[0]}   delta_max={float(jnp.max(jnp.abs(res['delta']))):.3e}")

print()
print("=" * 72)
print("TEST 2: joint  (threshold 旧: min(gh) < gh[0]*0.6)")
print("=" * 72)
res2 = optimize_geometry_process(geo, material="316L", process_init=_high_energy_plan(geo),
                                 params=dict(n_grid=8, max_layers=4, n_sub_cp=32, tau_activation=1e-3),
                                 n_steps=12, learning_rate=0.05, smoothness=0.02, verbose=False)
gh2 = np.asarray(res2["geom_history"])
print("geom_history:", np.round(gh2, 6))
print(f"gh2[0]={gh2[0]:.6f}  min(gh2)={gh2.min():.6f}  min/gh2[0]={gh2.min()/gh2[0]:.4f}")
print(f"decreased? min<gh0 : {gh2.min() < gh2[0]}")
z_init = normalize_process(_high_energy_plan(geo))
print(f"delta_max={float(jnp.max(jnp.abs(res2['delta']))):.3e}  z_move={float(jnp.max(jnp.abs(res2['z']-z_init))):.3e}")

print()
print("=" * 72)
print("TEST 3: pareto  (threshold 旧: span_g>1e-6 or span_s>1e-6)")
print("=" * 72)
res3 = pareto_optimize_geometry_process(geo, material="316L", params=dict(n_grid=4, max_layers=2, n_sub_cp=4, tau_activation=1e-3),
                                        algorithm="nsga2", pop_size=8, n_gen=3, seed=0,
                                        smoothness=0.02, constraint_weight=20.0, asbuilt_solver="plastic",
                                        verbose=False)
gd = np.asarray(res3["geom_dev"])
st = np.asarray(res3["stress"])
print("geom_dev :", np.round(gd, 8))
print("stress   :", np.round(st, 8))
print(f"span_g={float(np.max(gd)-np.min(gd)):.3e}  span_s={float(np.max(st)-np.min(st)):.3e}")
print(f"feasible min={float(np.min(res3['feasible'])):.3f}")
print(f"loss_history last/first: {res3.get('loss_history', None)}")
