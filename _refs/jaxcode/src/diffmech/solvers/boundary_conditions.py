"""Boundary conditions: Dirichlet constraints.

We use the standard "row/column elimination" approach for linear systems, and
the "split residual / diagonal" trick for Newton iterations on R(u) = 0:

    For unknowns with prescribed values u_D = u0:
        - In linear: K_ii = 1, K_ij = 0 (j != i), F_i = u_D
        - In Newton: du_i = u_D - u_current_i (drives the value to u_D in one step)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class DirichletBC:
    """Dirichlet boundary condition.

    Attributes
    ----------
    dofs : (n_dofs,) int array
        Global DOF indices to be constrained.
    values : (n_dofs,) float array
        Prescribed values.
    """

    dofs: jax.Array
    values: jax.Array

    @classmethod
    def fixed(cls, dofs: jax.Array, value: float = 0.0) -> "DirichletBC":
        """All dofs fixed to ``value`` (default 0)."""
        n = dofs.shape[0]
        return cls(dofs=dofs, values=jnp.full((n,), value, dtype=jnp.float64))


def apply_dirichlet_residual(
    residual: jnp.ndarray, du_guess: jnp.ndarray, bcs: list[DirichletBC]
) -> jnp.ndarray:
    """Modify residual so that Newton step ``du`` satisfies Dirichlet BCs.

    Returns residual with bc dofs replaced by ``value - current_value``,
    so the Newton update ``du = -K^{-1} R`` enforces the constraint.
    """
    out = residual
    for bc in bcs:
        out = out.at[bc.dofs].set(bc.values - du_guess[bc.dofs])
    return out


def apply_dirichlet_to_system(
    K: jnp.ndarray, F: jnp.ndarray, bcs: list[DirichletBC]
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply Dirichlet BCs to a linear system K u = F via row elimination.

    Returns modified (K, F) where bc-dofs are pinned. The system can then be
    solved with ``jnp.linalg.solve`` (small systems) or any iterative solver.

    Implementation is fully vectorised (``.at[indices].set``) so it is safe to
    call from inside ``jax.jit`` / ``jax.lax.scan`` traced code — no Python
    ``int()`` conversion of tracer leaves.
    """
    K = K.copy()
    F = F.copy()
    n = K.shape[0]
    eye = jnp.eye(n, dtype=K.dtype)
    for bc in bcs:
        # Vectorised row elimination: set the constrained rows of K to the
        # corresponding identity rows, and pin the RHS to the prescribed value.
        K = K.at[bc.dofs, :].set(eye[bc.dofs])
        F = F.at[bc.dofs].set(bc.values)
    return K, F


# Backwards-compatible aliases
def apply_dirichlet(K, F, bcs):
    return apply_dirichlet_to_system(K, F, bcs)


def scatter_with_dirichlet(
    free_solution: jnp.ndarray, bcs: list[DirichletBC], n_dofs: int
) -> jnp.ndarray:
    """Combine free-dofs solution with prescribed BC dofs into full vector."""
    out = free_solution
    for bc in bcs:
        out = out.at[bc.dofs].set(bc.values)
    return out
