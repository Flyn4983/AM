"""热源标定：蒸发封顶 c_evap 标定 + 多工况峰值验证。"""
import sys
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, "src")
from amforge.geometry import from_sdf_fn
from amforge.core.contracts import ProcessPlan
from amforge.thermal_enthalpy import solve_enthalpy_thermal
from amforge.process import heuristic_plan

def part(dx=80e-6, half=0.24e-3):
    return from_sdf_fn(lambda x: jnp.linalg.norm(x, axis=-1) - half,
                       bounds=[(-half, half)] * 3, spacing=dx, name="box")

def plan(power=200.0, v=1.0, r=50e-6, eta=0.4, hatch=80e-6, lt=40e-6):
    return ProcessPlan.uniform(3, modality="SLM", laser_power=power, scan_speed=v,
                              layer_thickness=lt, hatch_spacing=hatch,
                              beam_radius=r, absorption=eta, preheat_temp=400.0)

def peak(P, v, r=50e-6, eta=0.4, cevap=3.0e11, dx=80e-6, n=80):
    th = solve_enthalpy_thermal(geometry=part(dx=dx),
                                process=plan(power=P, v=v, r=r, eta=eta),
                                params={"material": "316L", "n_steps": n,
                                        "evap_coeff": cevap})
    return float(jnp.max(th.peak_temperature)), int(jnp.sum(th.time_above_melt > 0))

print("=== c_evap 标定：高功率 _high_energy_plan(900W,0.6,r=60um,eta=0.5) ===")
for ce in [0.0, 1e11, 3e11, 1e12, 3e12]:
    pk, melt = peak(900, 0.6, r=60e-6, eta=0.5, cevap=ce)
    print(f"  c_evap={ce:.0e}  peak={pk:9.1f}K  melt_cells={melt}")

print("\n=== 低/中功率：蒸发封顶不应介入（c_evap=3e11）===")
for (P, v) in [(200, 1.0), (300, 0.8), (300, 1.0), (600, 1.0), (600, 1.2)]:
    pk, melt = peak(P, v, cevap=3e11)
    print(f"  P={P:4d} v={v:.1f}  peak={pk:8.1f}K  melt_cells={melt}")

print("\n=== 默认 heuristic_plan（optimize_shape / pareto 实际用工艺）===")
g = part(dx=80e-6)
hp = heuristic_plan(g, material="316L", modality="SLM")
th = solve_enthalpy_thermal(geometry=g, process=hp,
                            params={"material": "316L", "n_steps": 80,
                                    "evap_coeff": 3e11})
print(f"  heuristic_plan peak={float(jnp.max(th.peak_temperature)):.1f}K  "
      f"melt_cells={int(jnp.sum(th.time_above_melt>0))}  "
      f"P={float(jnp.mean(hp.laser_power)):.0f}W v={float(jnp.mean(hp.scan_speed)):.2f}")
