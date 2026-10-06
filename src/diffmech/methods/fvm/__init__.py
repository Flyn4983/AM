"""Differentiable Finite Volume Method (fluid mechanics).

Provides:
    - 2D/3D Cartesian FVM grid (:class:`CartesianGrid`)
    - Scalar linear advection (upwind flux, SSP-RK2, periodic / Dirichlet BCs)
    - Compressible Euler equations (Rusanov flux, SSP-RK2, periodic /
      reflecting BCs)
    - Incompressible Stokes flow (projection method, periodic BCs, CG-based
      Poisson solve)

All hot loops use ``jnp.roll`` / ``jnp.pad`` + ``jax.lax.scan`` so they are
``jax.jit`` and ``jax.grad`` friendly. End-to-end differentiation (e.g. of a
final-time state w.r.t. an initial condition or material parameter) is
supported for every method.
"""

from diffmech.methods.fvm.grid import (
    CartesianGrid, cartesian_grid_2d, cartesian_grid_3d,
)
from diffmech.methods.fvm.advection import (
    upwind_flux, step_advection, step_advection_scan, cfl_dt as cfl_dt_advection,
)
from diffmech.methods.fvm.euler import (
    pressure, sound_speed, conservative_to_primitive,
    rusanov_flux_axis, step_euler, step_euler_scan,
    cfl_dt as cfl_dt_euler,
)
from diffmech.methods.fvm.stokes import step_stokes, step_stokes_scan
from diffmech.methods.fvm.thermal import (
    ThermalConfig,
    diffusion_rhs, step_thermal, step_thermal_scan, diffusion_dt,
    gaussian_heat_source, moving_gaussian_source,
)

__all__ = [
    "CartesianGrid", "cartesian_grid_2d", "cartesian_grid_3d",
    "upwind_flux", "step_advection", "step_advection_scan",
    "cfl_dt_advection",
    "pressure", "sound_speed", "conservative_to_primitive",
    "rusanov_flux_axis", "step_euler", "step_euler_scan",
    "cfl_dt_euler",
    "step_stokes", "step_stokes_scan",
    # thermal (heat diffusion — AM "process → thermal history" stage)
    "ThermalConfig",
    "diffusion_rhs", "step_thermal", "step_thermal_scan", "diffusion_dt",
    "gaussian_heat_source", "moving_gaussian_source",
]
