"""方案 B 第二组：joint lr 扫描 + pareto 全种群目标分布。"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from amforge import geometry as G
from amforge.core.contracts import PartGeometry
from amforge.inverse import optimize_geometry_process, pareto_optimize_geometry_process
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
JP = dict(n_grid=8, max_layers=4, n_sub_cp=32, tau_activation=1e-3)

print("=" * 72)
print("EXPERIMENT A: joint 扫 lr（看能否调稳、降到多少）")
print("=" * 72)
for lr in (0.005, 0.01, 0.02, 0.05):
    res = optimize_geometry_process(geo, material="316L", process_init=_high_energy_plan(geo),
                                    params=JP, n_steps=12, learning_rate=lr,
                                    smoothness=0.02, verbose=False)
    gh = np.asarray(res["geom_history"])
    z_init = normalize_process(_high_energy_plan(geo))
    print(f"lr={lr:>6}: gh0={gh[0]:.5f} min={gh.min():.5f} min/gh0={gh.min()/gh[0]:.4f} "
          f"delta_max={float(jnp.max(jnp.abs(res['delta']))):.3e} z_move={float(jnp.max(jnp.abs(res['z']-z_init))):.3e}")

print()
print("=" * 72)
print("EXPERIMENT B: pareto 全种群目标分布（前沿单点是否为退化）")
print("=" * 72)
res = pareto_optimize_geometry_process(geo, material="316L", params=dict(n_grid=4, max_layers=2, n_sub_cp=4, tau_activation=1e-3),
                                       algorithm="nsga2", pop_size=8, n_gen=3, seed=0,
                                       smoothness=0.02, constraint_weight=20.0, asbuilt_solver="plastic",
                                       verbose=False)
pg = np.asarray(res["population_geom_dev"]); ps = np.asarray(res["population_stress"])
print(f"population geom_dev = {np.round(pg, 6)}")
print(f"population stress   = {np.round(ps, 6)}")
print(f"pop span_g={float(np.max(pg)-np.min(pg)):.3e}  span_s={float(np.max(ps)-np.min(ps)):.3e}")
# 设计变量是否探索（pareto_x 在前 d_u 个 u 分量 + 9 个 z 分量）
px = np.asarray(res["pareto_x"])
print(f"pareto_x shape={px.shape}")
if px.shape[0] > 1:
    print(f"  pareto_x per-individual norm: {np.round(np.linalg.norm(px, axis=1), 4)}")
    print(f"  unique designs: {len(np.unique(np.round(px, 6), axis=0))}")
else:
    print("  single design (front collapsed to 1)")
