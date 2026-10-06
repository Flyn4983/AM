"""Mesh utility helpers: boundary detection, region marking, refinement.

All routines are NumPy-based (called at *setup time* only) and return plain
arrays / :class:`~diffmech.core.Mesh` objects — they are not meant to be
called inside a hot JAX loop.
"""

from __future__ import annotations

from typing import Callable, Iterable

import jax.numpy as jnp
import numpy as np

from diffmech.core import Mesh, rectangular_mesh2d, rectangular_mesh3d


# ---------------------------------------------------------------------------
# Boundary detection
# ---------------------------------------------------------------------------
def boundary_node_ids(mesh: Mesh) -> np.ndarray:
    """Return node ids on the boundary of ``mesh``.

    A node is on the boundary iff it lies on at least one *boundary face*
    (a face that is referenced by exactly one cell). Works for any cell type
    supported by DiffMech (tri3 / quad4 / tet4 / hex8).
    """
    cells = np.asarray(mesh.cells)
    faces_of_cell = _cell_faces(mesh.cell_type)

    # Build a canonical (sorted) face signature for each face of each cell.
    n_cells = cells.shape[0]
    n_faces_per_cell = faces_of_cell.shape[0]
    all_faces = cells[:, faces_of_cell].reshape(-1, faces_of_cell.shape[1])
    all_faces_sorted = np.sort(all_faces, axis=1)
    # Convert each row to a tuple for hashing.
    keys = [tuple(row) for row in all_faces_sorted]
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True,
                                    return_counts=True)
    boundary_face_mask = counts[inverse] == 1  # face appears in only 1 cell
    boundary_nodes = np.unique(all_faces[boundary_face_mask])
    return boundary_nodes.astype(np.int64)


def _cell_faces(cell_type: str) -> np.ndarray:
    """Local node indices of every face of one cell, shape (n_faces, n_per_face)."""
    if cell_type == "tri3":
        return np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int32)
    if cell_type == "quad4":
        return np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32)
    if cell_type == "tet4":
        return np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
                        dtype=np.int32)
    if cell_type == "hex8":
        # Quad faces of a hex (matches diffmech.core.shape_functions.hex8
        # node ordering).
        return np.array([
            [0, 1, 2, 3],  # -z
            [4, 5, 6, 7],  # +z
            [0, 1, 5, 4],  # -y
            [2, 3, 7, 6],  # +y
            [1, 2, 6, 5],  # +x
            [0, 3, 7, 4],  # -x
        ], dtype=np.int32)
    raise ValueError(f"unsupported cell_type {cell_type!r}")


def nodes_on_box_face(
    mesh: Mesh,
    *,
    xmin: float | None = None,
    xmax: float | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
    zmin: float | None = None,
    zmax: float | None = None,
    tol: float = 1e-9,
) -> np.ndarray:
    """Return node ids lying on the given faces of an axis-aligned box.

    Only the non-``None`` faces are tested; passing all ``None`` returns
    every node. Typical use::

        # nodes on the left (x=0) face
        left = nodes_on_box_face(mesh, xmin=0.0)
        # nodes on the top (y=ly) face
        top = nodes_on_box_face(mesh, ymax=ly)
    """
    nodes = np.asarray(mesh.nodes)
    mask = np.ones(len(nodes), dtype=bool)
    if xmin is not None:
        mask &= np.abs(nodes[:, 0] - xmin) < tol
    if xmax is not None:
        mask &= np.abs(nodes[:, 0] - xmax) < tol
    if ymin is not None:
        mask &= np.abs(nodes[:, 1] - ymin) < tol
    if ymax is not None:
        mask &= np.abs(nodes[:, 1] - ymax) < tol
    if zmin is not None:
        mask &= np.abs(nodes[:, 2] - zmin) < tol
    if zmax is not None:
        mask &= np.abs(nodes[:, 2] - zmax) < tol
    return np.where(mask)[0].astype(np.int64)


def nodes_in_box(
    mesh: Mesh,
    *,
    xmin: float | None = None,
    xmax: float | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
    zmin: float | None = None,
    zmax: float | None = None,
) -> np.ndarray:
    """Return node ids inside an axis-aligned box (closed).

    ``None`` bounds are treated as ±inf. Useful for marking a sub-region as
    a different material.
    """
    nodes = np.asarray(mesh.nodes)
    x = nodes[:, 0]
    y = nodes[:, 1] if nodes.shape[1] > 1 else np.zeros(len(nodes))
    z = nodes[:, 2] if nodes.shape[1] > 2 else np.zeros(len(nodes))
    xmin_ = -np.inf if xmin is None else xmin
    xmax_ = np.inf if xmax is None else xmax
    ymin_ = -np.inf if ymin is None else ymin
    ymax_ = np.inf if ymax is None else ymax
    zmin_ = -np.inf if zmin is None else zmin
    zmax_ = np.inf if zmax is None else zmax
    mask = ((x >= xmin_ - 1e-15) & (x <= xmax_ + 1e-15) &
            (y >= ymin_ - 1e-15) & (y <= ymax_ + 1e-15) &
            (z >= zmin_ - 1e-15) & (z <= zmax_ + 1e-15))
    return np.where(mask)[0].astype(np.int64)


# ---------------------------------------------------------------------------
# Element geometry
# ---------------------------------------------------------------------------
def element_volumes(mesh: Mesh) -> np.ndarray:
    """Per-cell volume (2D) or area (3D ``tet4``); for ``hex8`` returns 0.

    For quad4 the volume is computed from the shoelace formula on each cell's
    corner polygon. For tri3 the signed area is ``0.5 |cross|``. For tet4 the
    signed volume is the standard determinant formula. For hex8 we decompose
    into 6 tets.
    """
    cells = np.asarray(mesh.cells)
    coords = np.asarray(mesh.nodes)
    n_cells = cells.shape[0]

    if mesh.cell_type == "tri3":
        p = coords[cells]  # (n_cells, 3, dim)
        v01 = p[:, 1] - p[:, 0]
        v02 = p[:, 2] - p[:, 0]
        cross = v01[:, 0] * v02[:, 1] - v01[:, 1] * v02[:, 0]
        return 0.5 * np.abs(cross)

    if mesh.cell_type == "quad4":
        # Shoelace on the 4-node polygon (assumes convex, CCW ordering).
        p = coords[cells]  # (n_cells, 4, dim)
        x = p[:, :, 0]
        y = p[:, :, 1]
        area = 0.5 * np.abs(np.sum(
            x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y,
            axis=1,
        ))
        return area

    if mesh.cell_type == "tet4":
        p = coords[cells]  # (n_cells, 4, 3)
        v0 = p[:, 1] - p[:, 0]
        v1 = p[:, 2] - p[:, 0]
        v2 = p[:, 3] - p[:, 0]
        vol = np.abs(np.einsum("ni,ni->n", np.cross(v1, v2), v0)) / 6.0
        return vol

    if mesh.cell_type == "hex8":
        # Decompose each hex into 6 tets and sum.
        p = coords[cells]  # (n_cells, 8, 3)
        # 6-tet Freudenthal decomposition around the body diagonal 0-6,
        # consistent with the tet4 split in rectangular_mesh3d.
        # Hex node order: 0(-,-,-) 1(+,-,-) 2(+,+,-) 3(-,+,-)
        #                 4(-,-,+) 5(+,-,+) 6(+,+,+) 7(-,+,+)
        tet_idx = np.array([
            [0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6],
            [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6],
        ], dtype=np.int32)
        vols = np.zeros(n_cells)
        for tet in tet_idx:
            pt = p[:, tet]  # (n_cells, 4, 3)
            v0 = pt[:, 1] - pt[:, 0]
            v1 = pt[:, 2] - pt[:, 0]
            v2 = pt[:, 3] - pt[:, 0]
            vols += np.abs(np.einsum("ni,ni->n", np.cross(v1, v2), v0)) / 6.0
        return vols

    raise ValueError(f"unsupported cell_type {mesh.cell_type!r}")


def total_volume(mesh: Mesh) -> float:
    """Total domain volume / area."""
    return float(np.sum(element_volumes(mesh)))


# ---------------------------------------------------------------------------
# Multi-material region assignment
# ---------------------------------------------------------------------------
def assign_cell_regions(
    mesh: Mesh,
    region_funcs: Iterable[Callable[[np.ndarray], bool]],
) -> np.ndarray:
    """Assign each cell to a region index based on its centroid.

    Parameters
    ----------
    region_funcs : iterable of callables
        Each ``f(centroid: (dim,) array) -> bool`` returns True if the cell
        belongs to that region. The first region whose predicate is True wins.
        Cells that match no predicate get region id ``-1``.

    Returns
    -------
    region_ids : (n_cells,) int array
    """
    cells = np.asarray(mesh.cells)
    coords = np.asarray(mesh.nodes)
    centroids = coords[cells].mean(axis=1)  # (n_cells, dim)
    region_ids = np.full(centroids.shape[0], -1, dtype=np.int32)
    region_funcs = list(region_funcs)
    for i, f in enumerate(region_funcs):
        matches = np.array([bool(f(c)) for c in centroids])
        region_ids = np.where(matches & (region_ids == -1), i, region_ids)
    return region_ids


def box_region(
    *, xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None,
) -> Callable[[np.ndarray], bool]:
    """A region predicate matching centroids inside an axis-aligned box."""
    def f(c: np.ndarray) -> bool:
        x = c[0]
        y = c[1] if c.shape[0] > 1 else 0.0
        z = c[2] if c.shape[0] > 2 else 0.0
        ok = True
        if xmin is not None: ok &= x >= xmin - 1e-15
        if xmax is not None: ok &= x <= xmax + 1e-15
        if ymin is not None: ok &= y >= ymin - 1e-15
        if ymax is not None: ok &= y <= ymax + 1e-15
        if zmin is not None: ok &= z >= zmin - 1e-15
        if zmax is not None: ok &= z <= zmax + 1e-15
        return ok
    return f


# ---------------------------------------------------------------------------
# Uniform refinement of structured meshes
# ---------------------------------------------------------------------------
def refine_structured_2d(mesh: Mesh, *, factor: int = 2) -> Mesh:
    """Uniformly refine a structured 2D mesh by ``factor`` in each direction.

    The original mesh must be a :class:`StructuredMesh2D` (built via
    :func:`rectangular_mesh2d`). Returns a new structured mesh with the same
    domain size and cell type but ``factor``× more cells per direction.
    """
    if not hasattr(mesh, "nx") or mesh.nx == 0:
        raise ValueError("refine_structured_2d requires a StructuredMesh2D")
    return rectangular_mesh2d(
        nx=mesh.nx * factor, ny=mesh.ny * factor,
        lx=mesh.lx, ly=mesh.ly, cell_type=mesh.cell_type,
    )


def refine_structured_3d(mesh: Mesh, *, factor: int = 2) -> Mesh:
    """Uniformly refine a structured 3D mesh by ``factor`` in each direction."""
    if not hasattr(mesh, "nx") or mesh.nx == 0:
        raise ValueError("refine_structured_3d requires a StructuredMesh3D")
    return rectangular_mesh3d(
        nx=mesh.nx * factor, ny=mesh.ny * factor, nz=mesh.nz * factor,
        lx=mesh.lx, ly=mesh.ly, lz=mesh.lz, cell_type=mesh.cell_type,
    )
