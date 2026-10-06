"""诊断：asbuilt 塑性求解对设计变量（u / 工艺）的敏感性。
区分热源峰值问题 vs 优化/asbuilt 链路问题。"""
import sys, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, ".")
sys.path.insert(0, "src")
import amforge.geometry as G
from amforge.core.contracts import PartGeometry, ProcessPlan
from amforge.thermal_enthalpy import solve_enthalpy_thermal
from amforge.process import heuristic_plan
from amforge.meltpool import solve_meltpool_surrogate
try:
    from amforge.asbuilt_plastic import solve_asbuilt_plastic
except Exception:
    from amforge.inverse import solve_asbuilt_plastic

def small_geo():
    return G.from_sdf_fn(lambda x: jnp.linalg.norm(x, axis=-1) - 0.4e-3,
                        bounds=[(-0.5e-3, 0.5e-3)] * 3, spacing=120e-6, name="opt-demo")

def thermal_of(geo, process, heat_scale=1.4):
    mp = solve_meltpool_surrogate(geometry=geo, process=process,
                                  params={"material": "316L", "n_grid": 8})
    return solve_enthalpy_thermal(geometry=geo, process=process,
                                  params={"material": "316L", "n_steps": 48, "heat_scale": heat_scale})

def asbuilt_of(nominal, thermal, process):
    eps = float(nominal.spacing)
    occ = nominal.soft_occupancy(eps=eps)
    return solve_asbuilt_plastic(
        geometry=nominal, thermal=thermal,
        params={"material": "316L", "constitutive": "j2", "max_layers": 4,
                "n_sub_cp": 32, "tau_activation": 1e-3, "process": process,
                "occupancy_override": occ})

def geom_dev(ab, geo):
    occ = geo.soft_occupancy()
    dev = jnp.abs(ab.sdf - geo.sdf) * occ
    return float(jnp.sum(dev) / jnp.sum(occ) / float(geo.spacing))

geo = small_geo()
proc = heuristic_plan(geo, material="316L", modality="SLM")
th = thermal_of(geo, proc, heat_scale=1.4)
print(f"[thermal] peak={float(jnp.max(th.peak_temperature)):.1f}K  "
      f"melt_cells={int(jnp.sum(th.time_above_melt>0))}")

# ---- (1) asbuilt 对 u（名义 SDF 修正）的敏感性 ----
nom0 = PartGeometry(sdf=geo.sdf, origin=geo.origin, spacing=float(geo.spacing),
                    dim=geo.dim, name="nom0")
ab0 = asbuilt_of(nom0, th, proc)
print(f"[u=0]      geom_dev={geom_dev(ab0, geo):.5f}  "
      f"disp_max={float(jnp.max(jnp.abs(ab0.displacement))):.3e}  "
      f"vonMises_max={float(jnp.max(ab0.von_mises_residual())):.3e}")

def loss_u(u):
    nom = PartGeometry(sdf=geo.sdf + u * float(geo.spacing), origin=geo.origin,
                       spacing=float(geo.spacing), dim=geo.dim, name="n")
    ab = asbuilt_of(nom, th, proc)
    occ = geo.soft_occupancy()
    return jnp.sum(jnp.abs(ab.sdf - geo.sdf) * occ) / jnp.sum(occ) / float(geo.spacing)

g = jax.grad(loss_u)(jnp.zeros(geo.shape))
print(f"[grad_u]   norm={float(jnp.linalg.norm(g)):.3e}  "
      f"maxabs={float(jnp.max(jnp.abs(g))):.3e}  (若≈0 → asbuilt 对 u 不敏感)")

# ---- (2) asbuilt 对不同工艺的敏感性 ----
def hep(geo):
    return ProcessPlan.uniform(int(geo.layer_count(40e-6)), modality="SLM",
                              laser_power=900.0, scan_speed=0.6, layer_thickness=60e-6,
                              hatch_spacing=120e-6, beam_radius=60e-6, absorption=0.5,
                              preheat_temp=473.0, dwell_time=0.0)
proc_hi = hep(geo)
th_hi = thermal_of(geo, proc_hi, heat_scale=1.4)
print(f"[thermal hi] peak={float(jnp.max(th_hi.peak_temperature)):.1f}K")
ab_hi = asbuilt_of(nom0, th_hi, proc_hi)
print(f"[proc=hi]   geom_dev={geom_dev(ab_hi, geo):.5f}  "
      f"disp_max={float(jnp.max(jnp.abs(ab_hi.displacement))):.3e}")
print(f"-> heuristic vs high_energy geom_dev 差 = {geom_dev(ab_hi,geo)-geom_dev(ab0,geo):+.5f}")
