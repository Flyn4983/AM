"""Field probes, integrals, and statistics.

Compute point values, line integrals, surface integrals, volume averages, and
time-history extractions from simulation outputs. All routines accept JAX
arrays so they can be used inside ``jax.grad`` if desired.
"""

from __future__ import annotations

from typing import Iterable

import jax
import jax.numpy as jnp
import numpy as np

from diffmech.core import Mesh
from diffmech.preprocess.mesh_tools import element_volumes


# ---------------------------------------------------------------------------
# Per-node field reshaping
# ---------------------------------------------------------------------------
def displacement_to_node_field(U: jnp.ndarray, n_nodes: int, dim: int) -> jnp.ndarray:
    """Reshape a flat DOF vector to (n_nodes, dim)."""
    return jnp.asarray(U).reshape(n_nodes, dim)


def node_field_to_dofs(field: jnp.ndarray) -> jnp.ndarray:
    """Flatten a (n_nodes, dim) per-node field to a DOF vector."""
    return jnp.asarray(field).ravel()


# ---------------------------------------------------------------------------
# Point probe (interpolated field value at an arbitrary location)
# ---------------------------------------------------------------------------
def _containing_cell(mesh: Mesh, point: np.ndarray) -> tuple[int, np.ndarray]:
    """Find the cell containing ``point`` and return (cell_id, barycentric).

    Brute-force search: iterate cells, check whether ``point`` is inside.
    Works for tri3 / quad4 / tet4 / hex8 by projecting onto a common
    barycentric / shape-function evaluation.
    """
    nodes = np.asarray(mesh.nodes)
    cells = np.asarray(mesh.cells)
    point = np.asarray(point, dtype=np.float64)
    if point.shape[0] < nodes.shape[1]:
        point = np.concatenate([point, np.zeros(nodes.shape[1] - point.shape[0])])
    point = point[:nodes.shape[1]]

    from diffmech.core import get_shape_function
    sf = get_shape_function(mesh.cell_type)
    n_nodes_per_cell = sf.n_nodes
    dim = nodes.shape[1]

    # For each cell, evaluate shape functions at the *reference* coords
    # corresponding to the physical point — we don't have an inverse
    # isoparametric map readily, so we use a Newton iteration.
    for cid in range(cells.shape[0]):
        cell_nodes = nodes[cells[cid]]
        xi = jnp.zeros(sf.dim)
        for _ in range(20):
            N = np.asarray(sf.value(xi))
            dN = np.asarray(sf.grad(xi))
            F = N @ cell_nodes - point
            J = dN.T @ cell_nodes
            try:
                dx = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                break
            xi = xi + jnp.asarray(dx)
        # Check if xi lies inside the reference element.
        inside = _inside_reference(xi, mesh.cell_type, tol=1e-6)
        if inside:
            return cid, np.asarray(xi)
    raise ValueError(f"point {point} not found in any cell of the mesh")


def _inside_reference(xi, cell_type: str, tol: float = 1e-6) -> bool:
    xi = np.asarray(xi)
    if cell_type == "tri3":
        return bool(xi[0] >= -tol and xi[1] >= -tol and
                    xi[0] + xi[1] <= 1 + tol)
    if cell_type == "quad4":
        return bool(np.all(np.abs(xi) <= 1 + tol))
    if cell_type == "tet4":
        return bool(xi[0] >= -tol and xi[1] >= -tol and xi[2] >= -tol and
                    xi[0] + xi[1] + xi[2] <= 1 + tol)
    if cell_type == "hex8":
        return bool(np.all(np.abs(xi) <= 1 + tol))
    return False


def point_probe(
    mesh: Mesh, point: jnp.ndarray, field: jnp.ndarray
) -> jnp.ndarray:
    """Interpolate a per-node field at an arbitrary physical point.

    Parameters
    ----------
    point : (dim,) coordinates
    field : (n_nodes,) or (n_nodes, k) per-node field

    Returns
    -------
    value : () or (k,) interpolated field value
    """
    from diffmech.core import get_shape_function
    sf = get_shape_function(mesh.cell_type)
    cid, xi = _containing_cell(mesh, np.asarray(point))
    N = np.asarray(sf.value(xi))
    cell_nodes = np.asarray(mesh.cells)[cid]
    field_np = np.asarray(field)
    if field_np.ndim == 1:
        return jnp.asarray(N @ field_np[cell_nodes])
    return jnp.asarray(N @ field_np[cell_nodes])


# ---------------------------------------------------------------------------
# Volume / mass integrals
# ---------------------------------------------------------------------------
def volume_average(
    mesh: Mesh, field: jnp.ndarray, *, weights: str = "uniform"
) -> jnp.ndarray:
    """Volume-weighted average of a per-cell field.

    Parameters
    ----------
    field : (n_cells,) or (n_cells, k)
    weights : ``"uniform"`` (arithmetic mean) or ``"volume"`` (volume-weighted).
    """
    field = jnp.asarray(field)
    vols = jnp.asarray(element_volumes(mesh))
    if weights == "uniform":
        w = jnp.ones_like(vols)
    elif weights == "volume":
        w = vols
    else:
        raise ValueError(f"weights must be 'uniform' or 'volume', got {weights!r}")
    w = w / jnp.sum(w)
    if field.ndim == 1:
        return jnp.sum(w * field)
    return jnp.sum(w[:, None] * field, axis=0)


def volume_integral(
    mesh: Mesh, field: jnp.ndarray
) -> jnp.ndarray:
    """Integrate a per-cell field over the volume (``Σ field_c * V_c``)."""
    field = jnp.asarray(field)
    vols = jnp.asarray(element_volumes(mesh))
    if field.ndim == 1:
        return jnp.sum(field * vols)
    return jnp.sum(field * vols[:, None], axis=0)


def total_mass(mesh: Mesh, density_per_cell: jnp.ndarray) -> jnp.ndarray:
    """Total mass = Σ ρ_c V_c."""
    return volume_integral(mesh, jnp.asarray(density_per_cell))


# ---------------------------------------------------------------------------
# Boundary-face integrals (for reactions, fluxes, etc.)
# ---------------------------------------------------------------------------
def boundary_node_field_integral(
    mesh: Mesh,
    field: jnp.ndarray,
    boundary_node_ids: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Sum of a per-node field over boundary nodes (e.g. reaction force).

    Parameters
    ----------
    boundary_node_ids : optional subset. If None, all boundary nodes are used.
    field : (n_nodes,) or (n_nodes, k) per-node field (e.g. nodal force).
    """
    field = jnp.asarray(field)
    if boundary_node_ids is None:
        from diffmech.preprocess.mesh_tools import boundary_node_ids as _bnd
        boundary_node_ids = jnp.asarray(_bnd(mesh))
    else:
        boundary_node_ids = jnp.asarray(boundary_node_ids, dtype=jnp.int32)
    if field.ndim == 1:
        return jnp.sum(field[boundary_node_ids])
    return jnp.sum(field[boundary_node_ids], axis=0)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def field_statistics(field: jnp.ndarray) -> dict[str, jnp.ndarray]:
    """Min / max / mean / L2-norm of a (n, ...) field."""
    field = jnp.asarray(field)
    return {
        "min": jnp.min(field),
        "max": jnp.max(field),
        "mean": jnp.mean(field),
        "l2": jnp.sqrt(jnp.sum(field ** 2)),
        "abs_max": jnp.max(jnp.abs(field)),
    }


# ---------------------------------------------------------------------------
# Time-history extraction
# ---------------------------------------------------------------------------
def extract_history(
    history: jnp.ndarray, *, every: int = 1
) -> jnp.ndarray:
    """Subsample a (n_steps, ...) history array.

    Parameters
    ----------
    every : keep one sample every ``every`` steps (1 = keep all).
    """
    history = jnp.asarray(history)
    return history[::every]


def history_at_dof(
    history: jnp.ndarray, dof: int
) -> jnp.ndarray:
    """Extract the value of a single DOF over time from a displacement history.

    Parameters
    ----------
    history : (n_steps, n_dofs) array
    dof : global DOF index
    """
    return jnp.asarray(history)[:, dof]


def tip_displacement_history(
    history: jnp.ndarray, tip_dofs: jnp.ndarray
) -> jnp.ndarray:
    """Average displacement of the tip dofs over time.

    Parameters
    ----------
    history : (n_steps, n_dofs)
    tip_dofs : (n_tip,) DOF indices on the "tip" (loaded) face
    """
    history = jnp.asarray(history)
    tip_dofs = jnp.asarray(tip_dofs, dtype=jnp.int32)
    return jnp.mean(history[:, tip_dofs], axis=1)
