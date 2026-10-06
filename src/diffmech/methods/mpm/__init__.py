"""Differentiable Material Point Method (MLS-MPM, 2D & 3D).

Provides:
    - :class:`MPMState` — particle system state (position, velocity, affine
      term, deformation gradient, volume, mass)
    - :class:`MPMConfig` — grid, material and time-stepping parameters
    - :func:`make_mpm_state` — convenience constructor
    - :func:`step_mpm` / :func:`step_mpm_scan` — MLS-MPM time stepping
      (semi-implicit Euler with B-spline P2G/G2P transfers)
    - :func:`kinetic_energy`, :func:`total_volume`, :func:`center_of_mass`
      — diagnostics suitable as optimisation objectives

Supports both PIC and APIC momentum transfer, a Neo-Hookean solid
constitutive model and a simple linear-EOS fluid.  Dimensionality is
implicit in the particle coordinate arrays: 2D for ``(N, 2)`` positions,
3D for ``(N, 3)``.
"""

from diffmech.methods.mpm.mpm import (
    MPMState, MPMConfig,
    make_mpm_state,
    step_mpm, step_mpm_scan,
    kinetic_energy, total_volume, center_of_mass,
)

__all__ = [
    "MPMState", "MPMConfig",
    "make_mpm_state",
    "step_mpm", "step_mpm_scan",
    "kinetic_energy", "total_volume", "center_of_mass",
]
