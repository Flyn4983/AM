"""演示：工艺杠杆物理约束 + 多目标 Pareto 权衡（尺寸偏差 ↔ 残余应力）。

用轻量 FEM 参数快速跑通，输出真实前沿数值。
"""
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from amforge import geometry as G
from amforge.core.contracts import ProcessPlan
from amforge.process import normalize_process, heuristic_plan
from amforge.inverse import (
    pareto_optimize_geometry_process, process_constraint_penalty,
)

GEO_KW = dict(bounds=[(-0.5e-3, 0.5e-3)] * 3, spacing=120e-6, name="demo")
LIGHT = dict(n_grid=6, max_layers=3, n_sub_cp=8, tau_activation=1e-3)


def small_geo():
    return G.from_sdf_fn(lambda x: jnp.linalg.norm(x, axis=-1) - 0.4e-3, **GEO_KW)


def main():
    geo = small_geo()
    nl = int(geo.layer_count(40e-6))

    # --- (1) 物理约束：病态工艺 vs 可行工艺 ---
    # 高能量病态工艺
    bad = ProcessPlan.uniform(int(nl), modality="SLM", laser_power=900.0,
                             scan_speed=0.6, layer_thickness=60e-6,
                             hatch_spacing=120e-6, beam_radius=60e-6,
                             absorption=0.5, preheat_temp=473.0, dwell_time=0.0)
    pen_bad, det_bad = process_constraint_penalty(normalize_process(bad),
                                                  material="316L", n_layers=nl)
    z_ok = normalize_process(heuristic_plan(geo, material="316L", modality="SLM"))
    pen_ok, _ = process_constraint_penalty(z_ok, material="316L", n_layers=nl)
    print("== 工艺物理约束罚项 ==")
    print(f"  可行(启发式)工艺 罚项 = {float(pen_ok):.4f}")
    print(f"  病态(高能量)工艺 罚项 = {float(pen_bad):.4f}  "
          f"(ΔH/h_s={float(det_bad['normalized_enthalpy']):.1f}, "
          f"VED={float(det_bad['ved']):.2e} J/m^3)")

    # --- (2) Pareto 前沿（NSGA-II 非支配排序，替代 λ-标量化）---
    print("\n== Pareto 前沿（NSGA-II：尺寸偏差 ↔ 残余应力）==")
    res = pareto_optimize_geometry_process(
        geo, material="316L", params=LIGHT,
        algorithm="nsga2", pop_size=16, n_gen=10, seed=0,
        smoothness=0.02, constraint_weight=20.0, asbuilt_solver="plastic",
        verbose=False)
    print(f"  {'#':>3} {'geom_dev':>10} {'stress(σy)':>12} {'feasible':>9}")
    for i, (gd, st, f) in enumerate(zip(res["geom_dev"], res["stress"],
                                       res["feasible"])):
        print(f"  {i:3d} {float(gd):10.3e} {float(st):12.3e} {float(f):9.3f}")
    bc = res["best_compromise"]
    print(f"\n  utopia = (geom_dev={res['utopia'][0]:.3e}, "
          f"stress={res['utopia'][1]:.3e})")
    print(f"  best_compromise  geom_dev={bc['geom_dev']:.3e}  "
          f"stress={bc['stress']:.3e}")


if __name__ == "__main__":
    main()
