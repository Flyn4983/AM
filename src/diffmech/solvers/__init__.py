"""Solvers and optimization utilities."""

from diffmech.solvers.newton import newton_solve, newton_solve_with_jacrev
from diffmech.solvers.time_integrators import (
    explicit_euler_step, semi_implicit_euler_step,
    velocity_verlet_step, rk4_step,
)
from diffmech.solvers.boundary_conditions import (
    DirichletBC, apply_dirichlet, scatter_with_dirichlet,
)
from diffmech.optimize.inverse import (
    optimize_params_optax, optimize_params_gd,
)

__all__ = [
    "newton_solve", "newton_solve_with_jacrev",
    "explicit_euler_step", "semi_implicit_euler_step",
    "velocity_verlet_step", "rk4_step",
    "DirichletBC", "apply_dirichlet", "scatter_with_dirichlet",
    "optimize_params_optax", "optimize_params_gd",
]
