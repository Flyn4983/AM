"""Differentiable Crystal-Plasticity FEM (2D & 3D).

Provides:

- **Slip systems**: pre-defined crystallographic slip systems for 2D
  (single / double slip) and 3D (FCC ``{111}<110>``, BCC ``{110}<111>``),
  plus the Schmid-tensor / resolved-shear-stress helpers.
- **Constitutive model**: rate-dependent small-strain crystal plasticity with
  power-law flow and Voce isotropic hardening. The update is explicit
  (forward Euler, sub-stepped via ``jax.lax.scan``) so it is fully
  reverse-mode-AD friendly.
- **FEM solver**: small-strain CPFE using the initial-strain (eigenstrain)
  formulation. Each load step is a single linear solve against the constant
  elastic stiffness; the plastic strain enters as a load vector. The whole
  time-stepping loop is differentiable through ``jax.grad``.

All routines work for both ``dim=2`` and ``dim=3``; dimensionality is inferred
from the mesh / slip-system arrays.
"""

from diffmech.methods.cpfe.slip_systems import (
    schmid_tensors,
    resolved_shear_stress,
    double_slip_2d,
    single_slip_2d,
    fcc_slip_systems,
    bcc_slip_systems,
    von_mises_fcc_slip_systems,
)
from diffmech.methods.cpfe.crystal_plasticity import (
    CPState,
    CrystalPlasticity,
    cp_stress,
    slip_rates,
    cp_update_step,
    cp_update,
    cp_stress_and_tangent,
    elastic_tangent_voigt,
    cp_update_batched,
)
from diffmech.methods.cpfe.cpfe_solver import (
    CPFEProblem,
    CPFESolution,
    solve_cpfe,
    qp_state_at_cell,
    average_equivalent_plastic_strain,
    total_slip,
)

__all__ = [
    # slip systems
    "schmid_tensors", "resolved_shear_stress",
    "double_slip_2d", "single_slip_2d",
    "fcc_slip_systems", "bcc_slip_systems", "von_mises_fcc_slip_systems",
    # constitutive model
    "CPState", "CrystalPlasticity",
    "cp_stress", "slip_rates", "cp_update_step", "cp_update",
    "cp_stress_and_tangent", "elastic_tangent_voigt", "cp_update_batched",
    # solver
    "CPFEProblem", "CPFESolution", "solve_cpfe",
    "qp_state_at_cell", "average_equivalent_plastic_strain", "total_slip",
]
