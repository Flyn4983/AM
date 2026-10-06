"""Linear elastic FEM: element stiffness assembly and direct solver.

For each element we form the local stiffness matrix

    k_e = sum_qp w_qp * B^T(xi_qp) * C * B(xi_qp)

where ``B = dN/dx`` is the strain-displacement matrix (mapping nodal
displacements to symmetric small-strain tensor in Voigt form), and ``C`` is
the constitutive tangent.

We vmap over quadrature points and elements; assembly into the global matrix
uses ``jax.ops.segment_sum``-style scatter (here ``jnp.zeros(...).at[idx].add``
which is jit-safe).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from diffmech.core import (
    Mesh, gauss_legendre_nd, get_shape_function, physical_gradient_and_det,
)
from diffmech.core.shape_functions import quad4_grad, quad4_shape
from diffmech.materials.linear_elastic import LinearElasticIsotropic
from diffmech.solvers.boundary_conditions import DirichletBC, apply_dirichlet


def _strain_displacement_matrix(dN_dx: jnp.ndarray, dim: int) -> jnp.ndarray:
    """B matrix mapping nodal dofs to Voigt strain.

    Parameters
    ----------
    dN_dx : (n_nodes, dim) shape function gradients at a quadrature point
    dim : 2 or 3

    Returns
    -------
    B : (n_strain_components, n_nodes * dim) matrix

    For dim=2 (plane strain): strain vector = [eps_xx, eps_yy, gamma_xy]
        eps_xx = sum_i dN_i/dx * u_i
        eps_yy = sum_i dN_i/dy * v_i
        gamma_xy = sum_i (dN_i/dy * u_i + dN_i/dx * v_i)

    For dim=3: strain = [eps_xx, eps_yy, eps_zz, gamma_yz, gamma_xz, gamma_xy]
    """
    n_nodes = dN_dx.shape[0]
    if dim == 2:
        # B is (3, 2*n_nodes)
        B = jnp.zeros((3, 2 * n_nodes), dtype=dN_dx.dtype)
        for i in range(n_nodes):
            dNidx = dN_dx[i, 0]
            dNidy = dN_dx[i, 1]
            # eps_xx from u_i
            B = B.at[0, 2 * i].set(dNidx)
            # eps_yy from v_i
            B = B.at[1, 2 * i + 1].set(dNidy)
            # gamma_xy = du/dy + dv/dx
            B = B.at[2, 2 * i].set(dNidy)
            B = B.at[2, 2 * i + 1].set(dNidx)
        return B
    elif dim == 3:
        B = jnp.zeros((6, 3 * n_nodes), dtype=dN_dx.dtype)
        for i in range(n_nodes):
            dNidx = dN_dx[i, 0]
            dNidy = dN_dx[i, 1]
            dNidz = dN_dx[i, 2]
            # eps_xx, eps_yy, eps_zz
            B = B.at[0, 3 * i].set(dNidx)
            B = B.at[1, 3 * i + 1].set(dNidy)
            B = B.at[2, 3 * i + 2].set(dNidz)
            # gamma_yz = dv/dz + dw/dy
            B = B.at[3, 3 * i + 1].set(dNidz)
            B = B.at[3, 3 * i + 2].set(dNidy)
            # gamma_xz = du/dz + dw/dx
            B = B.at[4, 3 * i].set(dNidz)
            B = B.at[4, 3 * i + 2].set(dNidx)
            # gamma_xy = du/dy + dv/dx
            B = B.at[5, 3 * i].set(dNidy)
            B = B.at[5, 3 * i + 1].set(dNidx)
        return B
    raise ValueError(f"unsupported dim {dim}")


def compute_element_stiffness_quad4(
    coords: jnp.ndarray, lam: float, mu: float
) -> jnp.ndarray:
    """Local stiffness matrix (8x8) for a 2D quad4 element.

    Parameters
    ----------
    coords : (4, 2) nodal coordinates of the element
    lam, mu : Lamé constants (plane strain assumed)
    """
    qr = gauss_legendre_nd(2, 2, reference="quad")
    C = jnp.array([
        [lam + 2 * mu, lam, 0.0],
        [lam, lam + 2 * mu, 0.0],
        [0.0, 0.0, mu],
    ], dtype=coords.dtype)
    n_nodes = 4
    dim = 2
    k_e = jnp.zeros((n_nodes * dim, n_nodes * dim), dtype=coords.dtype)

    def body(k_e, qp):
        xi, w = qp
        dN_dxi = quad4_grad(xi)
        dN_dx, detJ = physical_gradient_and_det(dN_dxi, coords)
        B = _strain_displacement_matrix(dN_dx, dim)
        k_e_local = w * detJ * (B.T @ C @ B)
        return k_e + k_e_local, None

    k_e, _ = jax.lax.scan(body, k_e, (qr.points, qr.weights))
    return k_e


def assemble_stiffness(
    mesh: Mesh, mat: LinearElasticIsotropic, *, dim: int | None = None
) -> jnp.ndarray:
    """Assemble global stiffness matrix for linear elastic FEM.

    Returns (n_dofs, n_dofs) where n_dofs = n_nodes * dim.
    """
    if dim is None:
        dim = mesh.dim
    n_dofs = mesh.n_nodes * dim
    cell_coords = mesh.cell_coords  # (n_cells, n_nodes_per_cell, dim)

    # Vectorize over cells
    def cell_ke(coords):
        if mesh.cell_type == "quad4" and dim == 2:
            return compute_element_stiffness_quad4(coords, mat.lam, mat.mu)
        # Generic fallback using the same approach as quad4
        return _generic_element_stiffness(coords, mesh.cell_type, mat.lam, mat.mu, dim)

    kes = jax.vmap(cell_ke)(cell_coords)  # (n_cells, n_dpc*dim, n_dpc*dim)

    # Assemble into global K
    K = jnp.zeros((n_dofs, n_dofs), dtype=mesh.nodes.dtype)
    cells = mesh.cells
    n_dpc = mesh.nodes_per_cell

    def assemble_one_cell(carry, args):
        ke, cell = args
        # DOF indices for this cell: [2*n0, 2*n0+1, 2*n1, 2*n1+1, ...] for dim=2
        dofs = jnp.repeat(cell * dim, dim) + jnp.tile(jnp.arange(dim), n_dpc)
        # Use scatter-add
        # We need to add ke[i,j] to K[dofs[i], dofs[j]]
        # Reshape into (n_dpc*dim, n_dpc*dim) and use np.add.at semantics
        # JAX equivalent: use jax.ops with multi-dimensional index
        di, dj = jnp.meshgrid(dofs, dofs, indexing="ij")
        # We need to update K[di, dj] += ke
        return carry.at[di.ravel(), dj.ravel()].add(ke.ravel()), None

    K, _ = jax.lax.scan(assemble_one_cell, K, (kes, cells))
    return K


def _generic_element_stiffness(
    coords: jnp.ndarray, cell_type: str, lam: float, mu: float, dim: int
) -> jnp.ndarray:
    """Generic element stiffness for tri3/quad4/tet4/hex8 via shape function + quad."""
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

    if dim == 2:
        C = jnp.array([
            [lam + 2 * mu, lam, 0.0],
            [lam, lam + 2 * mu, 0.0],
            [0.0, 0.0, mu],
        ], dtype=coords.dtype)
    else:
        C = jnp.zeros((6, 6), dtype=coords.dtype)
        C = C.at[0, 0].set(lam + 2 * mu)
        C = C.at[1, 1].set(lam + 2 * mu)
        C = C.at[2, 2].set(lam + 2 * mu)
        C = C.at[3, 3].set(mu)
        C = C.at[4, 4].set(mu)
        C = C.at[5, 5].set(mu)
        for i in range(3):
            for j in range(3):
                if i != j:
                    C = C.at[i, j].set(lam)

    k_e = jnp.zeros((n_nodes * dim, n_nodes * dim), dtype=coords.dtype)

    def body(k_e, qp):
        xi, w = qp
        dN_dxi = sf.grad(xi)
        dN_dx, detJ = physical_gradient_and_det(dN_dxi, coords)
        B = _strain_displacement_matrix(dN_dx, dim)
        return k_e + w * detJ * (B.T @ C @ B), None

    k_e, _ = jax.lax.scan(body, k_e, (qr.points, qr.weights))
    return k_e


def assemble_residual(
    mesh: Mesh, U: jnp.ndarray, mat: LinearElasticIsotropic, body_force=None
) -> jnp.ndarray:
    """Internal-force residual R_int(u) = K u - F_ext (for linear elastic).

    For nonlinear materials use :func:`diffmech.methods.fem.nonlinear_fem`.
    """
    dim = mesh.dim
    K = assemble_stiffness(mesh, mat, dim=dim)
    F_ext = jnp.zeros_like(U)
    if body_force is not None:
        # Body force vector: integrate b * N over each element
        F_ext = _assemble_body_force(mesh, body_force, dim=dim)
    return K @ U - F_ext


def _assemble_body_force(mesh: Mesh, body_force: jnp.ndarray, *, dim: int) -> jnp.ndarray:
    """Assemble consistent body-force load vector.

    ``body_force`` has shape (dim,).
    """
    n_dofs = mesh.n_nodes * dim
    F = jnp.zeros(n_dofs, dtype=mesh.nodes.dtype)
    cell_coords = mesh.cell_coords
    cells = mesh.cells

    sf = get_shape_function(mesh.cell_type)
    if mesh.cell_type == "tri3":
        qr = gauss_legendre_nd(2, 2, reference="tri")
    elif mesh.cell_type == "quad4":
        qr = gauss_legendre_nd(2, 2, reference="quad")
    elif mesh.cell_type == "tet4":
        qr = gauss_legendre_nd(2, 3, reference="tet")
    elif mesh.cell_type == "hex8":
        qr = gauss_legendre_nd(2, 3, reference="hex")
    else:
        raise ValueError(mesh.cell_type)

    def cell_force(coords):
        n_nodes = sf.n_nodes
        f_e = jnp.zeros(n_nodes * dim, dtype=coords.dtype)
        def body(f_e, qp):
            xi, w = qp
            N = sf.value(xi)
            dN_dxi = sf.grad(xi)
            _, detJ = physical_gradient_and_det(dN_dxi, coords)
            # f_i,d = w * detJ * N_i * b_d
            f_local = jnp.outer(N, body_force).ravel() * w * detJ
            return f_e + f_local, None
        f_e, _ = jax.lax.scan(body, f_e, (qr.points, qr.weights))
        return f_e

    fes = jax.vmap(cell_force)(cell_coords)  # (n_cells, n_dpc*dim)

    def assemble_one_cell(carry, args):
        fe, cell = args
        n_dpc = cell.shape[0]
        dofs = jnp.repeat(cell * dim, dim) + jnp.tile(jnp.arange(dim), n_dpc)
        return carry.at[dofs].add(fe), None

    F, _ = jax.lax.scan(assemble_one_cell, F, (fes, cells))
    return F


def solve_linear_elastic(
    mesh: Mesh,
    mat: LinearElasticIsotropic,
    bcs: list[DirichletBC],
    *,
    body_force=None,
    dim: int | None = None,
) -> jnp.ndarray:
    """Solve the linear elastic static problem K u = F.

    Parameters
    ----------
    mesh : Mesh
    mat : LinearElasticIsotropic
    bcs : list of DirichletBC
        Dirichlet constraints (e.g. clamped boundary).
    body_force : (dim,) array, optional
        Constant body force per unit volume (e.g. gravity).

    Returns
    -------
    U : (n_nodes * dim,) nodal displacement vector
    """
    if dim is None:
        dim = mesh.dim
    n_dofs = mesh.n_nodes * dim
    K = assemble_stiffness(mesh, mat, dim=dim)
    F = jnp.zeros(n_dofs, dtype=mesh.nodes.dtype)
    if body_force is not None:
        F = F + _assemble_body_force(mesh, body_force, dim=dim)

    # Apply Dirichlet BCs
    K, F = apply_dirichlet(K, F, bcs)
    U = jnp.linalg.solve(K, F)
    return U


# ---------------------------------------------------------------------------
# Stress / strain recovery (post-processing from a displacement field)
# ---------------------------------------------------------------------------
_CENTROID_XI = {
    "tri3": jnp.array([1.0 / 3.0, 1.0 / 3.0]),
    "quad4": jnp.array([0.0, 0.0]),
    "tet4": jnp.array([1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0]),
    "hex8": jnp.array([0.0, 0.0, 0.0]),
}


def recover_strain_stress(
    mesh: Mesh, U: jnp.ndarray, mat: LinearElasticIsotropic,
    *, dim: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Recover per-cell strain and stress tensors from a displacement field.

    Evaluates the small-strain tensor ``ε = sym(∇u)`` and the Cauchy stress
    ``σ = λ tr(ε) I + 2 μ ε`` at each element centroid. This is the standard
    FEM post-processing step that turns a nodal displacement solution into
    engineering fields for visualisation (von Mises, principal stresses, …).

    Parameters
    ----------
    mesh : Mesh
    U : (n_nodes * dim,) nodal displacement vector.
    mat : LinearElasticIsotropic
        Provides the Lamé constants ``λ, μ``.

    Returns
    -------
    strain : (n_cells, dim, dim) symmetric strain tensor per cell.
    stress : (n_cells, dim, dim) symmetric Cauchy stress tensor per cell.

    Notes
    -----
    The recovery is differentiable, so ``jax.grad`` flows from a loss built
    on the recovered stress back to the material parameters or boundary
    conditions that produced ``U``.
    """
    if dim is None:
        dim = mesh.dim
    sf = get_shape_function(mesh.cell_type)
    xi_c = _CENTROID_XI[mesh.cell_type]
    dN_dxi = sf.grad(xi_c)  # (n_nodes_per_cell, dim) — constant for tri3/tet4,
    # but evaluated per cell for quad4/hex8 (centred, so also constant; we still
    # vmap to keep the Jacobian per cell correct).
    cell_coords = mesh.cell_coords  # (n_cells, n_nodes_per_cell, dim)
    cells = mesh.cells
    lam = mat.lam
    mu = mat.mu

    def one_cell(coords, cell):
        dN_dx, _ = physical_gradient_and_det(dN_dxi, coords)  # (npc, dim)
        B = _strain_displacement_matrix(dN_dx, dim)            # (n_strain, npc*dim)
        u_e = jnp.repeat(cell * dim, dim) + jnp.tile(jnp.arange(dim), len(cell))
        u_e = U[u_e]                                            # (npc*dim,)
        eps_voigt = B @ u_e                                    # (n_strain,)
        # Voigt → symmetric tensor (shear components carry γ = 2ε, so halve them)
        if dim == 2:
            strain = jnp.array([
                [eps_voigt[0], 0.5 * eps_voigt[2]],
                [0.5 * eps_voigt[2], eps_voigt[1]],
            ])
        else:
            strain = jnp.array([
                [eps_voigt[0], 0.5 * eps_voigt[5], 0.5 * eps_voigt[4]],
                [0.5 * eps_voigt[5], eps_voigt[1], 0.5 * eps_voigt[3]],
                [0.5 * eps_voigt[4], 0.5 * eps_voigt[3], eps_voigt[2]],
            ])
        tr = jnp.trace(strain)
        eye = jnp.eye(dim, dtype=strain.dtype)
        stress = lam * tr * eye + 2.0 * mu * strain
        return strain, stress

    strain, stress = jax.vmap(one_cell)(cell_coords, cells)
    return strain, stress
