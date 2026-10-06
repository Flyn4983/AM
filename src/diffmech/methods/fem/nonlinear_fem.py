"""Nonlinear FEM (large deformation, nonlinear material).

We use Total Lagrangian formulation with the deformation gradient
F = I + grad(u). Internal force at a node is

    f_int_i = sum_qp w_qp * detJ_qp * P(F_qp) : dN_i/dX_qp

where P is the first Piola-Kirchhoff stress and grad is w.r.t. reference
coordinates (here the mesh nodes). The Newton residual is

    R(u) = f_int(u) - f_ext = 0

The Jacobian (consistent tangent) is computed by autodiff of f_int w.r.t. u,
which makes the code short and always consistent with the constitutive update.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from diffmech.core import (
    Mesh, gauss_legendre_nd, get_shape_function, physical_gradient_and_det,
)
from diffmech.materials.hyperelastic import NeoHookean
from diffmech.solvers.boundary_conditions import DirichletBC, apply_dirichlet_residual
from diffmech.solvers.newton import newton_solve, newton_solve_diff


@dataclass(frozen=True)
class FEMProblem:
    """A nonlinear FEM problem specification."""

    mesh: Mesh
    material: object  # must expose ``.piola(F)`` returning (dim, dim)
    body_force: jnp.ndarray | None = None
    dirichlet_bcs: tuple = field(default_factory=tuple)
    dim: int | None = None

    @property
    def n_dofs(self) -> int:
        return self.mesh.n_nodes * (self.dim or self.mesh.dim)


@dataclass
class FEMState:
    u: jnp.ndarray  # nodal displacements
    converged: bool = False
    n_iter: int = 0


def _element_internal_force(
    coords: jnp.ndarray,
    u_e: jnp.ndarray,
    material,
    cell_type: str,
    dim: int,
) -> jnp.ndarray:
    """Compute internal force vector for one element.

    Parameters
    ----------
    coords : (n_nodes, dim) reference nodal coords
    u_e : (n_nodes * dim,) nodal displacements for this element
    """
    sf = get_shape_function(cell_type)
    n_nodes = sf.n_nodes
    if cell_type == "tri3":
        qr = gauss_legendre_nd(2, 2, reference="tri")
    elif cell_type == "quad4":
        qr = gauss_legendre_nd(2, 2, reference="quad")
    elif cell_type == "tet4":
        qr = gauss_legendre_nd(2, 3, reference="tet")
    elif cell_type == "hex8":
        qr = gauss_legendre_nd(2, 3, reference="hex")
    else:
        raise ValueError(cell_type)

    # Reshape u_e into (n_nodes, dim)
    u_node = u_e.reshape(n_nodes, dim)
    # Deformed coords
    x_def = coords + u_node

    f_int_e = jnp.zeros(n_nodes * dim, dtype=coords.dtype)

    def body(f_int_e, qp):
        xi, w = qp
        dN_dxi = sf.grad(xi)
        dN_dX, detJ = physical_gradient_and_det(dN_dxi, coords)  # ref grad
        # Displacement gradient: grad u = dN/dX^T ... actually:
        # grad u = sum_i u_i (x) dN_i/dX, in tensor form:
        # (grad u)_{a,b} = sum_i u_{i,a} dN_i/dX_b
        # Here we want F = I + grad u
        grad_u = u_node.T @ dN_dX  # (dim, dim)
        F = jnp.eye(dim, dtype=coords.dtype) + grad_u
        P = material.piola(F)  # (dim, dim) first Piola-Kirchhoff
        # f_int_i,a = w * detJ * sum_b P_{a,b} * dN_i/dX_b
        # = w * detJ * (dN/dX @ P^T)_i,a
        f_local = w * detJ * (dN_dX @ P.T).ravel()
        return f_int_e + f_local, None

    f_int_e, _ = jax.lax.scan(body, f_int_e, (qr.points, qr.weights))
    return f_int_e


def assemble_internal_force(
    mesh: Mesh, U: jnp.ndarray, material, *, dim: int | None = None
) -> jnp.ndarray:
    """Assemble internal force vector f_int(U) for the whole mesh."""
    if dim is None:
        dim = mesh.dim
    n_dofs = mesh.n_nodes * dim
    cell_coords = mesh.cell_coords
    cells = mesh.cells
    n_dpc = mesh.nodes_per_cell

    # Get u_e for each cell
    def cell_u_e(cell):
        dofs = jnp.repeat(cell * dim, dim) + jnp.tile(jnp.arange(dim), n_dpc)
        return U[dofs]

    u_per_cell = jax.vmap(cell_u_e)(cells)  # (n_cells, n_dpc*dim)

    # Compute f_int per cell (vmapped)
    f_per_cell = jax.vmap(
        lambda c, ue: _element_internal_force(c, ue, material, mesh.cell_type, dim)
    )(cell_coords, u_per_cell)

    # Assemble
    def assemble_one_cell(carry, args):
        fe, cell = args
        dofs = jnp.repeat(cell * dim, dim) + jnp.tile(jnp.arange(dim), n_dpc)
        return carry.at[dofs].add(fe), None

    F_int, _ = jax.lax.scan(assemble_one_cell, jnp.zeros(n_dofs, dtype=U.dtype),
                            (f_per_cell, cells))
    return F_int


def assemble_external_force(
    mesh: Mesh, body_force: jnp.ndarray | None, *,
    traction_bcs: list | None = None, dim: int | None = None,
) -> jnp.ndarray:
    """Assemble external force vector (body force + surface tractions)."""
    if dim is None:
        dim = mesh.dim
    n_dofs = mesh.n_nodes * dim
    F_ext = jnp.zeros(n_dofs, dtype=mesh.nodes.dtype)
    if body_force is not None:
        from diffmech.methods.fem.linear_fem import _assemble_body_force
        F_ext = F_ext + _assemble_body_force(mesh, body_force, dim=dim)
    if traction_bcs:
        # Not implemented here; would integrate over boundary faces
        pass
    return F_ext


def solve_nonlinear_fem(
    problem: FEMProblem,
    u0: jnp.ndarray | None = None,
    *,
    tol: float = 1e-8,
    max_iter: int = 30,
    differentiable: bool = False,
) -> FEMState:
    """Solve the nonlinear FEM problem via Newton-Raphson.

    The residual is R(u) = f_int(u) - f_ext, with Dirichlet BCs enforced by
    modifying the residual so that constrained dofs satisfy u = u_BC.

    Parameters
    ----------
    differentiable : bool
        If True, use :func:`newton_solve_diff` (fixed-iteration scan, supports
        ``jax.grad`` end-to-end). Slightly slower than the early-terminating
        variant.
    """
    dim = problem.dim or problem.mesh.dim
    n_dofs = problem.n_dofs
    if u0 is None:
        u0 = jnp.zeros(n_dofs, dtype=problem.mesh.nodes.dtype)

    f_ext = assemble_external_force(
        problem.mesh, problem.body_force, dim=dim
    )
    bcs = list(problem.dirichlet_bcs)

    def R(u):
        f_int = assemble_internal_force(problem.mesh, u, problem.material, dim=dim)
        r = f_int - f_ext
        # Apply Dirichlet BCs: replace residual at constrained dofs with
        # (u_target - u_current) so the Newton update drives u to u_target.
        for bc in bcs:
            r = r.at[bc.dofs].set(bc.values - u[bc.dofs])
        return r

    if differentiable:
        res = newton_solve_diff(R, u0, tol=tol, max_iter=max_iter)
    else:
        res = newton_solve(R, u0, tol=tol, max_iter=max_iter)
    return FEMState(u=res.u, converged=res.converged, n_iter=res.n_iter)
