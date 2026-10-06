"""Scratch: validate the asbuilt.thermomechanical_plastic bridge end-to-end."""
from __future__ import annotations
import sys, os
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from amforge.core.contracts import PartGeometry, ThermalHistory
from amforge.asbuilt_plastic import solve_asbuilt_plastic
from amforge.materials import get_material
import amforge.forge_adapter  # triggers ForgeCore registration
from forgecore.registry import REGISTRY


def make_box(Nx=8, Ny=8, Nz=6, spacing=1e-3):
    sdf = np.full((Nx, Ny, Nz), 1e-3)
    sdf[1:-1, 1:-1, 1:-1] = -1e-3  # simple solid box with 1-voxel skin
    return PartGeometry(
        sdf=jnp.asarray(sdf), origin=jnp.zeros(3), spacing=spacing, dim=3, name="box",
    )


def make_thermal(geo, peak=900.0):
    sh = geo.shape
    base = jnp.full(sh, peak)
    return ThermalHistory(
        peak_temperature=base,
        cooling_rate=jnp.full(sh, 1e3),
        thermal_gradient=jnp.full(sh, 1e3),
        solidification_rate=jnp.full(sh, 1e-2),
        time_above_melt=jnp.full(sh, 1e-3),
        final_temperature=jnp.full(sh, 300.0),
        spacing=float(geo.spacing), dim=3,
    )


geo = make_box()
th = make_thermal(geo, peak=900.0)
mat = get_material("316L")

print("=== J2 (von Mises) ===")
ab_j2 = solve_asbuilt_plastic(geometry=geo, thermal=th,
                              params={"material": "316L", "constitutive": "j2"})
vm = ab_j2.von_mises_residual()
occ = geo.occupancy
vm_in = vm[occ > 0.5]
print("finite:", bool(jnp.all(jnp.isfinite(vm))),
      " max vm (MPa):", float(jnp.max(vm_in)) / 1e6,
      " mean vm (MPa):", float(jnp.mean(vm_in)) / 1e6,
      " sigma_y (MPa):", mat.sigma_y / 1e6)
print("E*alpha*dT overshoot (MPa):", mat.E * mat.alpha_thermal * (900 - 300) / 1e6)
print("disp max (um):", float(jnp.max(jnp.linalg.norm(ab_j2.displacement, axis=-1))) * 1e6)

print("=== CPFE (FCC) ===")
ab_cp = solve_asbuilt_plastic(geometry=geo, thermal=th,
                              params={"material": "316L", "constitutive": "cp"})
vm2 = ab_cp.von_mises_residual()
vm2_in = vm2[occ > 0.5]
print("finite:", bool(jnp.all(jnp.isfinite(vm2))),
      " max vm (MPa):", float(jnp.max(vm2_in)) / 1e6,
      " mean vm (MPa):", float(jnp.mean(vm2_in)) / 1e6)
# anisotropy: sigma_xy vs sigma_xx at a part cell
rs = ab_cp.residual_stress
print("sample cell Voigt (MPa):", (rs[rs.shape[0] // 2] / 1e6).tolist())

# ---- ForgeCore routing ----
print("=== ForgeCore route ===")
specs = REGISTRY.auto("asbuilt", select={"asbuilt": "asbuilt.thermomechanical_plastic"})
print("chain:", [s.name for s in specs])
assert any(s.name == "asbuilt.thermomechanical_plastic" for s in specs)
req = REGISTRY.required_inputs("asbuilt", select={"asbuilt": "asbuilt.thermomechanical_plastic"})
print("required inputs:", req)

# ---- Differentiability w.r.t. peak temperature (proxy to process) ----
print("=== differentiable w.r.t. thermal peak ===")
def loss(peak_scalar):
    t = make_thermal(geo, peak=peak_scalar)
    ab = solve_asbuilt_plastic(geometry=geo, thermal=t,
                               params={"material": "316L", "constitutive": "j2"})
    return jnp.sum(ab.von_mises_residual()[occ > 0.5])

g = jax.grad(loss)(900.0)
print("d loss / d peak =", float(g))
assert jnp.isfinite(g) and float(g) != 0.0
print("OK: bridge validated, differentiable.")
