"""D0 诊断：test_process_constraint_reduces_penalty_in_joint 的空断言问题。

实测症状（docs/evidence/2026-10-07/am_d0_feas_subset.log）：
    AssertionError: 约束未降低罚项: pen_cons=0.000 >= pen_free=0.000
即**末步罚项比较**在该夹具上是空断言——无约束轨迹也被尺寸损失自己带进可行域。

本探针把两条轨迹的**全程罚项序列**打出来，用来判：
1. 现夹具起点（tier 子盒提升后）的罚项量级 pen_init；
2. 逐段/累计罚项是否真能把「启用软约束」与「未启用」区分开（这才是该测试想测的东西）；
3. 若把起点改成更深度的病态工艺（tier 角落的匙孔/VED 越界），pen_free 是否仍留得住
   （注意 Adam 在归一化坐标下每步位移 ≈ lr=0.05，6 步最多走 0.30，罚项未必降得完）。

用法：
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=src JAX_ENABLE_X64=1 \\
    python docs/evidence/2026-10-07/am_d0_pen_traj_probe.py
"""
import time

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from amforge import geometry as G
from amforge.core.contracts import ProcessPlan
from amforge.inverse import (optimize_geometry_process, process_constraint_penalty,
                             tier_bounds, thermal_tier)
from amforge.process import normalize_process

_GEO_KW = dict(bounds=[(-0.5e-3, 0.5e-3)] * 3, spacing=120e-6, name="opt-demo")
_LIGHT = dict(n_grid=6, max_layers=3, n_sub_cp=8, tau_activation=1e-3)
_N_STEPS = 6
_LR = 0.05


def geo():
    return G.from_sdf_fn(lambda x: jnp.linalg.norm(x, axis=-1) - 0.4e-3, **_GEO_KW)


def candidates(g, tb):
    """三个候选起点：物理值直接落在 tier 子盒内可表达的角落。"""
    v_lo, r_lo, d_lo = tb["scan_speed"][0], tb["beam_radius"][0], tb["layer_thickness"][0]
    nl = int(g.layer_count(40e-6))
    kw = dict(modality="SLM", layer_thickness=d_lo, hatch_spacing=d_lo, dwell_time=0.0,
              preheat_temp=473.0)
    return {
        "A 现夹具量级(P900/v0.6/lt60→floor/r60/A0.5)": ProcessPlan.uniform(
            nl, laser_power=900.0, scan_speed=0.6, beam_radius=60e-6,
            absorption=0.5, **kw),
        "B 中度匙孔(P1600/v0.45/r_floor/A0.6)": ProcessPlan.uniform(
            nl, laser_power=1600.0, scan_speed=0.45, beam_radius=r_lo,
            absorption=0.6, **kw),
        "C 深匙孔+超窗(P3800/v_lo/r_floor/A0.85)": ProcessPlan.uniform(
            nl, laser_power=3800.0, scan_speed=v_lo, beam_radius=r_lo,
            absorption=0.85, **kw),
    }


def hist(res):
    return [round(float(v), 4) for v in res["constraint_penalty_history"]]


def main():
    g = geo()
    params = thermal_tier(dict(_LIGHT), g, ProcessPlan.uniform(
        int(g.layer_count(40e-6)), modality="SLM", laser_power=900.0,
        scan_speed=0.6, layer_thickness=120e-6, hatch_spacing=120e-6,
        beam_radius=60e-6, absorption=0.5, preheat_temp=473.0, dwell_time=0.0),
        material="316L", thermal_solver="enthalpy")
    tb = tier_bounds(params)
    print(f"[tier] scan_speed={tb['scan_speed']} beam_radius={tb['beam_radius']} "
          f"layer_thickness={tb['layer_thickness']} hatch_spacing={tb['hatch_spacing']}")
    for name, plan in candidates(g, tb).items():
        z0 = normalize_process(plan, bounds=tb)
        pen0, det = process_constraint_penalty(z0, material="316L",
                                               n_layers=int(g.layer_count(40e-6)),
                                               modality="SLM", params=params)
        print(f"\n=== {name}")
        print(f"    pen_init={float(pen0):.4f}  " + "  ".join(
            f"{k}={float(v):.4g}" for k, v in det.items()))
        t0 = time.time()
        free = optimize_geometry_process(
            g, material="316L", process_init=plan, params=_LIGHT, n_steps=_N_STEPS,
            learning_rate=_LR, smoothness=0.02, constraint_weight=0.0,
            hard_project=False, verbose=False)
        cons = optimize_geometry_process(
            g, material="316L", process_init=plan, params=_LIGHT, n_steps=_N_STEPS,
            learning_rate=_LR, smoothness=0.02, constraint_weight=20.0,
            hard_project=False, verbose=False)
        hf, hc = hist(free), hist(cons)
        print(f"    pen_free={hf}")
        print(f"    pen_cons={hc}   （用时 {time.time() - t0:.0f}s）")
        print(f"    [判据] 末步比较 pen_cons<{hf[-1]}: {hc[-1] < hf[-1]}"
              f" | 末步可行 pen_cons<0.2: {hc[-1] < 0.2}"
              f" | 累计 sum_cons={sum(hc):.4f} < sum_free={sum(hf):.4f}: {sum(hc) < sum(hf)}"
              f" | 逐步单调 hc<=hf 全部: {all(c <= f + 1e-12 for c, f in zip(hc, hf))}")


if __name__ == "__main__":
    main()
