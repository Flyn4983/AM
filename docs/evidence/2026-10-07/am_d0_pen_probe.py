"""D0 诊断（无仿真，纯调度 + 罚项）：test_process_constraint_reduces_penalty_in_joint
的起点是否本来就"可行"（pen=0），从而令 pen_cons < pen_free 变成空断言。

用法：CUDA_VISIBLE_DEVICES="" PYTHONPATH=src JAX_ENABLE_X64=1 python docs/evidence/2026-10-07/am_d0_pen_probe.py
"""
import time

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from amforge import geometry as G
from amforge.core.contracts import ProcessPlan
from amforge.inverse import (process_constraint_penalty, tier_bounds, thermal_tier,
                             z_to_device_box)
from amforge.process import PROCESS_BOUNDS, heuristic_plan, normalize_process

_GEO_KW = dict(bounds=[(-0.5e-3, 0.5e-3)] * 3, spacing=120e-6, name="opt-demo")
_LIGHT = dict(n_grid=6, max_layers=3, n_sub_cp=8, tau_activation=1e-3)


def _small_geo():
    return G.from_sdf_fn(lambda x: jnp.linalg.norm(x, axis=-1) - 0.4e-3, **_GEO_KW)


def _high_energy_plan(geo):
    return ProcessPlan.uniform(
        int(geo.layer_count(40e-6)), modality="SLM",
        laser_power=900.0, scan_speed=0.6,
        layer_thickness=60e-6, hatch_spacing=120e-6,
        beam_radius=60e-6, absorption=0.5, preheat_temp=473.0, dwell_time=0.0)


def show(tag, z, *, params=None, n_layers=None, modality="SLM"):
    pen, det = process_constraint_penalty(z, material="316L", n_layers=n_layers,
                                          modality=modality, params=params)
    print(f"  {tag}: pen={float(pen):.6f}  " + "  ".join(
        f"{k}={float(v):.4g}" for k, v in det.items()))
    return float(pen)


def main():
    geo = _small_geo()
    nl = int(geo.layer_count(40e-6))
    init = _high_energy_plan(geo)
    t0 = time.time()
    params = thermal_tier(dict(_LIGHT), geo, init, material="316L",
                          thermal_solver="enthalpy")
    print(f"[tier] 用时 {time.time() - t0:.2f}s")
    print(f"[tier] n_steps={params.get('n_steps')} "
          f"exposure_bound_s={params.get('exposure_bound_s')} "
          f"spacing={params.get('dx')}")
    tb = tier_bounds(params)
    print(f"[tier] 子盒 vs 设备盒:")
    for k in ("laser_power", "scan_speed", "layer_thickness", "hatch_spacing",
              "beam_radius", "absorption"):
        print(f"   {k:16s} device={PROCESS_BOUNDS[k]}  tier={tb.get(k)}")

    print("[init 物理工艺]")
    print(f"   P={init.laser_power} v={init.scan_speed} lt={init.layer_thickness} "
          f"h={init.hatch_spacing} r={init.beam_radius} A={init.absorption}")
    z_tier = normalize_process(init, bounds=tb)
    z_dev = normalize_process(init, bounds=PROCESS_BOUNDS)
    print(f"[z] tier 坐标={np_arr(z_tier)}")
    print(f"[z] 设备坐标={np_arr(z_dev)}")
    print(f"[z] 越界分量（tier 坐标不在 [0,1]）: "
          f"{[ (i, float(v)) for i, v in enumerate(z_tier) if v < 0 or v > 1]}")

    print("[罚项] 起点（tier 坐标 + tier 边界，与优化器内部一致）:")
    show("init/tier", z_tier, params=params, n_layers=nl)
    print("[罚项] 起点（tier 坐标 + 设备边界，即测试里 process_feasibility 的读法）:")
    show("init/device", z_tier, params=None, n_layers=nl)
    print("[罚项] 起点（设备坐标 + 设备边界）:")
    show("devcoord/device", z_dev, params=None, n_layers=nl)

    print("[罚项] 启发式名义工艺（对照）:")
    hp = heuristic_plan(geo, material="316L", modality="SLM")
    show("heuristic/tier", normalize_process(hp, bounds=tb), params=params, n_layers=nl)


def np_arr(z):
    return [round(float(v), 5) for v in jnp.atleast_1d(z)]


if __name__ == "__main__":
    main()
