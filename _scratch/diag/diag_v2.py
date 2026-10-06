"""热源标定验证 v2：heat_scale=1.4 + 蒸发封顶 c_evap=3e11（均默认值）。"""
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

def peak(P, v, r=50e-6, eta=0.4, dx=80e-6, n=80):
    th = solve_enthalpy_thermal(geometry=part(dx=dx),
                                process=plan(power=P, v=v, r=r, eta=eta),
                                params={"material": "316L", "n_steps": n})
    return float(jnp.max(th.peak_temperature)), int(jnp.sum(th.time_above_melt > 0))

m = __import__("amforge.materials", fromlist=["get_material"]).get_material("316L")
print(f"材料沸点 T_boil={m.T_boil:.0f}K, T_liquidus={m.T_liquidus:.0f}K")
print("\n=== P×v 矩阵（r=50um, eta=0.4, heat_scale=1.4, evap=3e11）===")
print("  P     v    E(J/mm)    peak(K)   melt_cells")
for P in [100, 200, 300, 600, 1200]:
    for v in [0.4, 0.8, 1.0, 1.2]:
        pk, melt = peak(P, v)
        flag = "  <-- 不熔化!" if pk < m.T_liquidus else ""
        print(f"  {P:4d}  {v:.1f}  {P/v/1000:6.3f}   {pk:8.1f}   {melt:3d}{flag}")

print("\n=== 高功率 _high_energy_plan(900W,0.6,r=60um,eta=0.5) 蒸发封顶 ===")
pk, melt = peak(900, 0.6, r=60e-6, eta=0.5)
print(f"  peak={pk:.1f}K  melt_cells={melt}  (应为 ~3000K 封顶)")

print("\n=== 默认 heuristic_plan（optimize_shape/pareto 实际工艺）===")
g = part(dx=80e-6)
hp = heuristic_plan(g, material="316L", modality="SLM")
th = solve_enthalpy_thermal(geometry=g, process=hp,
                            params={"material": "316L", "n_steps": 80})
print(f"  peak={float(jnp.max(th.peak_temperature)):.1f}K  melt_cells={int(jnp.sum(th.time_above_melt>0))}  "
      f"P={float(jnp.mean(hp.laser_power)):.0f}W v={float(jnp.mean(hp.scan_speed)):.2f}")
