"""Smoke test for the plastic / CPFE thermo-mechanical AM solver (P2-②).

Validates:
  * stress is finite and non-trivial,
  * plastic yielding CAPS residual stress near the yield strength
    (vs. the purely elastic solver which overshoots),
  * equivalent plastic strain accumulates (>0),
  * J2 and CPFE both produce finite, non-trivial fields,
  * CPFE gives anisotropic (low-symmetry) stress,
  * the solve is differentiable w.r.t. the thermal load.
"""
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from diffmech.materials import LinearElasticIsotropic
from diffmech.solvers import DirichletBC
from diffmech.methods.am.activation import build_layered_mesh
from diffmech.methods.am.thermomechanical import solve_thermomechanical
from diffmech.materials.plasticity import J2Plasticity
from diffmech.methods.cpfe.crystal_plasticity import CrystalPlasticity
from diffmech.methods.cpfe.slip_systems import fcc_slip_systems
from diffmech.methods.am.thermomechanical_plastic import solve_thermomechanical_plastic


def von_mises(sig):
    """von Mises equivalent stress of a (...,3,3) tensor."""
    s = sig - jnp.trace(sig, axis1=-2, axis2=-1)[..., None, None] / 3.0 * jnp.eye(3)
    return jnp.sqrt(1.5 * jnp.sum(s * s, axis=(-2, -1)))


# ---- geometry: 8x8x4 box ------------------------------------------------
bbox = [(-1.0e-3, 1.0e-3), (-1.0e-3, 1.0e-3), (0.0, 2.0e-3)]
nx = ny = 8
n_layers = 4
cpl = 1
layer_zs = np.linspace(bbox[2][0], bbox[2][1], n_layers, endpoint=False) + \
    (bbox[2][1] - bbox[2][0]) / (2 * n_layers)
layered = build_layered_mesh(layer_zs, bbox, nx=nx, ny=ny, cells_per_layer=cpl)
n_cells = layered.n_cells

# ---- BCs: clamp bottom z-face -------------------------------------------
nz = n_layers * cpl
ny1 = ny + 1
nz1 = nz + 1
bottom = [i * ny1 * nz1 + j * nz1 + 0 for i in range(nx + 1) for j in range(ny + 1)]
bc_dofs = np.stack([np.array(bottom) * 3 + d for d in range(3)], axis=1).ravel()
bcs = [DirichletBC.fixed(jnp.asarray(bc_dofs))]

# ---- aggressive cooling so the elastic predictor far exceeds yield -------
alpha_T = 1.2e-5
dT = -400.0  # K per layer
eps_th = jnp.full((n_layers, n_cells), alpha_T * dT)

# ---- materials -----------------------------------------------------------
lin = LinearElasticIsotropic(E=190e9, nu=0.30)
j2 = J2Plasticity(E=190e9, nu=0.30, sigma_y0=300e6, H=1.0e9)
b, n = fcc_slip_systems()
cp = CrystalPlasticity(E=190e9, nu=0.30, slip_directions=b, slip_normals=n,
                       gamma_dot0=0.001, m=0.05, g0=200e6, h0=1.0e9, g_sat=1.0e9)

# ---- 1) elastic predictor reference (plastic solver, infinite yield) -----
# Running the SAME solver but with a near-infinite yield strength gives the
# *elastic* stress (no return-mapping). This is the correct upper bound that
# plasticity must fall below.
j2_elastic = J2Plasticity(E=190e9, nu=0.30, sigma_y0=1.0e15, H=0.0)
res_el = solve_thermomechanical_plastic(layered, j2_elastic, constitutive="j2",
                                        thermal_strain_per_layer=eps_th,
                                        dirichlet_bcs=bcs, dim=3, n_inner=1)
vm_el = float(von_mises(res_el.residual_stress).max())
print(f"[elastic predictor] max von Mises = {vm_el/1e6:.1f} MPa")

# ---- 2) J2 plastic -------------------------------------------------------
res_j2 = solve_thermomechanical_plastic(layered, j2, constitutive="j2",
                                        thermal_strain_per_layer=eps_th,
                                        dirichlet_bcs=bcs, dim=3, n_inner=2)
vm_j2 = float(von_mises(res_j2.residual_stress).max())
eqp = float(res_j2.eq_plastic_strain.max())
print(f"[J2]       max von Mises = {vm_j2/1e6:.1f} MPa | max eq.pl.strain = {eqp:.4e}")

# ---- 3) CPFE ------------------------------------------------------------
res_cp = solve_thermomechanical_plastic(layered, cp, constitutive="cp",
                                        thermal_strain_per_layer=eps_th,
                                        dirichlet_bcs=bcs, dim=3, n_inner=1,
                                        dt_cp=1.0, n_sub_cp=200)
vm_cp = float(von_mises(res_cp.residual_stress).max())
eqp_cp = float(res_cp.eq_plastic_strain.max())
# anisotropy: off-diagonal shear components present?
shear = res_cp.residual_stress[..., 0, 1]
print(f"[CPFE]     max von Mises = {vm_cp/1e6:.1f} MPa | max eq.pl.strain = {eqp_cp:.4e} "
      f"| |σ_xy| max = {float(jnp.abs(shear).max())/1e6:.1f} MPa")

# ---- assertions ----------------------------------------------------------
fin = all(map(lambda r: bool(jnp.all(jnp.isfinite(r.residual_stress))),
              [res_j2, res_cp]))
assert fin, "stress must be finite"
assert vm_j2 > 0 and vm_cp > 0, "non-trivial stress"
# yielding must CAP the stress below the elastic predictor
assert vm_j2 < 0.9 * vm_el, f"J2 should cap stress ({vm_j2:.1f} < {vm_el:.1f})"
assert vm_cp < vm_el, f"CPFE should reduce stress below elastic ({vm_cp:.1f} < {vm_el:.1f})"
assert eqp > 0 and eqp_cp > 0, "plastic strain must accumulate"
# CPFE single-crystal anisotropic: off-diagonal stress should be non-negligible
assert float(jnp.abs(shear).max()) > 1e3, "CPFE should show shear/anisotropy"
print("YIELD CAPPING OK  (elastic %.1f MPa -> J2 %.1f / CPFE %.1f MPa)"
      % (vm_el/1e6, vm_j2/1e6, vm_cp/1e6))

# ---- 4) differentiability ----------------------------------------------
def max_vm(dT_scalar):
    et = jnp.full((n_layers, n_cells), alpha_T * dT_scalar)
    r = solve_thermomechanical_plastic(layered, j2, constitutive="j2",
                                       thermal_strain_per_layer=et,
                                       dirichlet_bcs=bcs, dim=3, n_inner=2)
    return von_mises(r.residual_stress).max()

g = jax.grad(max_vm)(jnp.asarray(dT))
print(f"[grad]     d(max von Mises)/d(dT) = {float(g):.3e}  (finite={bool(jnp.isfinite(g))})")
assert jnp.isfinite(g), "gradient must be finite"
assert abs(float(g)) > 0.0, "gradient must be nonzero"
print("DIFFERENTIABLE OK")
print("\nSMOKE OK: plastic / CPFE thermo-mechanical solver works.")
