"""Differentiable Discrete Element Method (granular dynamics, 2D & 3D).

Provides:
    - :class:`DEMState` — particle system state (positions, velocities, radii, masses)
    - :class:`DEMConfig` — contact model parameters (spring-dashpot + Coulomb friction)
    - :class:`Wall` — flat boundary wall for confinement
    - :func:`step_dem` / :func:`step_dem_scan` — semi-implicit Euler time stepping
    - :func:`kinetic_energy` — total kinetic energy (loss function primitive)

The contact model is a linear spring-dashpot (Hookean normal repulsion +
viscous damping) with optional tangential Coulomb friction.  All pairwise
interactions are computed via ``jax.vmap`` over an ``N x N`` pair matrix so
the whole simulation is ``jax.jit`` and ``jax.grad`` friendly.

Dimensionality is implicit in the particle coordinates: pass ``dim=2`` arrays
for a 2D simulation, ``dim=3`` for 3D.
"""

from diffmech.methods.dem.dem import (
    DEMState, DEMConfig, Wall,
    make_dem_state,
    step_dem, step_dem_scan,
    kinetic_energy,
)

__all__ = [
    "DEMState", "DEMConfig", "Wall",
    "make_dem_state",
    "step_dem", "step_dem_scan",
    "kinetic_energy",
]
