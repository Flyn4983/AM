"""Differentiable Phase-Field methods (2D & 3D).

Provides:
    - Allen-Cahn equation (microstructure / spinodal decomposition)
    - Phase-field fracture (variational brittle fracture, Bourdin–Francfort–Marigo)

All operations use ``jnp.roll`` / ``jax.lax.scan`` so they are
``jax.jit`` and ``jax.grad`` friendly.
"""

from diffmech.methods.phase_field.phase_field import (
    # Spatial helpers
    laplacian, gradient,
    # Allen-Cahn
    AllenCahnConfig, AllenCahnState,
    double_well_potential, double_well_derivative,
    allen_cahn_rhs, step_allen_cahn, step_allen_cahn_scan,
    free_energy_allen_cahn,
    # Phase-field fracture
    FractureConfig, FractureState,
    make_fracture_state,
    degradation_function, strain_energy_density,
    fracture_rhs, step_fracture, step_fracture_scan,
    fracture_energy,
)

__all__ = [
    "laplacian", "gradient",
    "AllenCahnConfig", "AllenCahnState",
    "double_well_potential", "double_well_derivative",
    "allen_cahn_rhs", "step_allen_cahn", "step_allen_cahn_scan",
    "free_energy_allen_cahn",
    "FractureConfig", "FractureState",
    "make_fracture_state",
    "degradation_function", "strain_energy_density",
    "fracture_rhs", "step_fracture", "step_fracture_scan",
    "fracture_energy",
]
