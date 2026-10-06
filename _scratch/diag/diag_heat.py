"""热源标定诊断：实测 solve_enthalpy_thermal 峰值温度对工艺参数的响应。"""
import sys, math
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, "src")
from amforge.geometry import from_sdf_fn
from amforge.core.contracts import ProcessPlan
from amforge.materials import get_material
from amforge.thermal_enthalpy import solve_enthalpy_thermal

def part(dx=80e-6, half=0.24e-3):
    return from_sdf_fn(lambda x: jnp.linalg.norm(x, axis=-1) - half,
                       bounds=[(-half, half)] * 3, spacing=dx, name="box")

def plan(power=200.0, v=1.0, r=50e-6, eta=0.4, hatch=80e-6, lt=40e-6):
    return ProcessPlan.uniform(3, modality="SLM", laser_power=power, scan_speed=v,
                              layer_thickness=lt, hatch_spacing=hatch,
                              beam_radius=r, absorption=eta, preheat_temp=400.0)

m = get_material("316L")
print("MAT rho,cp,k,L,Tamb,Tsol,Tliq =",
      m.rho_solid, m.cp_solid, m.k_solid, m.latent_fusion,
      m.T_ambient, m.T_solidus, m.T_liquidus)
print("物理参考：316L T_liquidus=%.0fK, 蒸气化 ~3120K\n" % m.T_liquidus)

def run(P, v, r=50e-6, eta=0.4, dx=80e-6, n=80):
    th = solve_enthalpy_thermal(geometry=part(dx=dx),
                                process=plan(power=P, v=v, r=r, eta=eta),
                                params={"material": "316L", "n_steps": n})
    pk = float(jnp.max(th.peak_temperature))
    melt = int(jnp.sum(th.time_above_melt > 0))
    return pk, melt

print("== 默认 uniform (P=200,v=1.0,r=50um,eta=0.4) 基线 ==")
pk, melt = run(200, 1.0)
print(f"  peak={pk:.1f}K  melt_cells={melt}\n")

print("== 扫描 P x v (r=50um, eta=0.4, dx=80um) ==")
print("  P     v    E(J/mm)    peak(K)   melt_cells")
for P in [100, 200, 300, 600, 1200]:
    for v in [0.4, 0.8, 1.0, 1.2]:
        pk, melt = run(P, v)
        print(f"  {P:4d}  {v:.1f}  {P/v*1e3:6.1f}   {pk:8.1f}   {melt}")

print("\n== 测试夹具工况 (P=1200,v=0.8,r=100um,eta=0.45) 敏感性 ==")
for r in [50e-6, 100e-6, 150e-6]:
    pk, melt = run(1200, 0.8, r=r, eta=0.45)
    print(f"  r={r*1e6:.0f}um  peak={pk:.1f}K  melt_cells={melt}")
