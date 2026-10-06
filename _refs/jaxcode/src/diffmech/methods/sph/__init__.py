"""Differentiable Smoothed Particle Hydrodynamics (WCSPH, 2D & 3D).

Provides:
    - :class:`SPHState` — particle state (position, velocity, mass)
    - :class:`SPHConfig` — SPH parameters (smoothing length, EOS, viscosity)
    - :func:`cubic_kernel` / :func:`cubic_kernel_grad` — cubic spline kernel
    - :func:`compute_density` — SPH density summation
    - :func:`pressure_from_density` — Tait equation of state
    - :func:`step_sph` / :func:`step_sph_scan` — semi-implicit Euler time stepping
    - :func:`kinetic_energy` — total kinetic energy (loss primitive)

Weakly Compressible SPH with cubic-spline kernel, Tait EOS, and Monaghan
artificial viscosity.  All pairwise interactions use ``jax.vmap`` over an
``N × N`` pair matrix — fully ``jax.jit`` / ``jax.grad`` friendly.
"""

from diffmech.methods.sph.sph import (
    SPHState, SPHConfig,
    make_sph_state,
    cubic_kernel, cubic_kernel_grad,
    compute_density, pressure_from_density,
    compute_acceleration,
    step_sph, step_sph_scan,
    total_mass, kinetic_energy,
)

__all__ = [
    "SPHState", "SPHConfig",
    "make_sph_state",
    "cubic_kernel", "cubic_kernel_grad",
    "compute_density", "pressure_from_density",
    "compute_acceleration",
    "step_sph", "step_sph_scan",
    "total_mass", "kinetic_energy",
]
