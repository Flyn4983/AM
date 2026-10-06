"""Differentiable Finite Element Method (solid mechanics).

Provides:
    - Linear static FEM solver for 2D/3D elasticity (plane strain in 2D)
    - Nonlinear FEM with Newton-Raphson for hyperelastic/plastic materials
    - Element stiffness via Gauss quadrature + autodiff of strain energy
    - Dirichlet BC support via the unified :class:`DirichletBC` interface
    - Differentiable end-to-end (so loss = functional(u) can be minimized
      w.r.t. material parameters via ``jax.grad``)

The element routine is shared across small-strain (eps = sym(grad u)) and
finite-strain (F = I + grad u) formulations; the constitutive update is
the only piece that differs.

All hot loops are written so that ``vmap``/``jax.lax.scan`` over cells can be
jitted.
"""

from diffmech.methods.fem.linear_fem import (
    assemble_stiffness, assemble_residual, solve_linear_elastic,
    compute_element_stiffness_quad4, recover_strain_stress,
)
from diffmech.methods.fem.nonlinear_fem import (
    solve_nonlinear_fem, FEMProblem, FEMState,
)

__all__ = [
    "assemble_stiffness", "assemble_residual", "solve_linear_elastic",
    "compute_element_stiffness_quad4", "recover_strain_stress",
    "solve_nonlinear_fem", "FEMProblem", "FEMState",
]
