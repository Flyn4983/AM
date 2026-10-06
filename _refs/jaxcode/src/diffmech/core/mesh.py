"""Differentiable mesh data structures.

A ``Mesh`` is a lightweight, jit/vmap-friendly container holding only jax arrays.
It deliberately avoids any Python-side polymorphism so that the whole mesh can
be threaded through ``jax.jit`` and ``jax.grad`` without recompilation.

Supported cell types (kept intentionally minimal but extensible):
    - "tri3"  : 3-node triangle in 2D
    - "quad4" : 4-node quadrilateral in 2D
    - "tet4"  : 4-node tetrahedron in 3D

The mesh stores *reference* (undeformed) coordinates so that deformation
gradients are computed as F = I + grad(u) w.r.t. reference coords. This makes
large-deformation mechanics straightforward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

CellType = Literal["tri3", "quad4", "tet4", "hex8"]


def _nodes_per_cell(cell_type: CellType) -> int:
    return {"tri3": 3, "quad4": 4, "tet4": 4, "hex8": 8}[cell_type]


@dataclass(frozen=True, eq=False)
class Mesh:
    """Differentiable mesh container.

    Attributes
    ----------
    nodes : (n_nodes, dim) float array
        Reference nodal coordinates.
    cells : (n_cells, n_nodes_per_cell) int array
        Connectivity, indices into ``nodes``.
    cell_type : str
        One of "tri3", "quad4", "tet4".
    boundary_node_ids : (n_boundary,) int array, optional
        Convenience mask of nodes lying on the boundary. Auto-detected if absent
        (slow path, used only for small meshes / setup).

    Note
    ----
    ``eq=False`` is set so the frozen dataclass uses object identity for hashing
    (jax arrays are not hashable by content). JAX's jit caches by abstract
    values (shape/dtype), not by array contents, so identity hashing is the
    correct choice.
    """

    nodes: jax.Array
    cells: jax.Array
    cell_type: CellType = "tri3"
    boundary_node_ids: jax.Array | None = None

    def __post_init__(self):
        npc = _nodes_per_cell(self.cell_type)
        if self.cells.shape[1] != npc:
            raise ValueError(
                f"cell_type={self.cell_type} expects {npc} nodes/cell, "
                f"got {self.cells.shape[1]}"
            )
        if self.nodes.shape[1] not in (2, 3):
            raise ValueError(f"nodes must be (n,2) or (n,3), got {self.nodes.shape}")

    # --- derived shapes ---------------------------------------------------
    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_cells(self) -> int:
        return int(self.cells.shape[0])

    @property
    def dim(self) -> int:
        return int(self.nodes.shape[1])

    @property
    def nodes_per_cell(self) -> int:
        return _nodes_per_cell(self.cell_type)

    # --- jit-safe views ---------------------------------------------------
    @property
    def cell_coords(self) -> jax.Array:
        """(n_cells, n_nodes_per_cell, dim) nodal coords of each cell."""
        return self.nodes[self.cells]

    def cell_node_coords(self, cell_index: int | jax.Array) -> jax.Array:
        """(n_nodes_per_cell, dim) coords for a single cell."""
        return self.nodes[self.cells[cell_index]]


def _detect_boundary_nodes(cells: jax.Array, n_nodes: int, cell_type: CellType) -> jax.Array:
    """A node is on the boundary iff it belongs to at least one boundary face.

    Cheap heuristic: count how many cells each node belongs to. For a regular
    structured mesh this is approximate; for unstructured meshes it is exact only
    if the mesh is fully manifold. Used only at construction time.
    """
    npc = _nodes_per_cell(cell_type)
    # Faces: pairs/triples of node ids that form a face. For tri3/quad4 we use edges.
    if cell_type == "tri3":
        edges = jnp.array([[0, 1], [1, 2], [2, 0]])
    elif cell_type == "quad4":
        edges = jnp.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    elif cell_type == "tet4":
        # Triangle faces of a tetrahedron
        edges = jnp.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    elif cell_type == "hex8":
        # Quad faces of a hexahedron (VTK ordering, see shape_functions.hex8)
        edges = jnp.array([
            [0, 1, 2, 3],  # bottom (-z)
            [4, 5, 6, 7],  # top    (+z)
            [0, 1, 5, 4],  # front  (-y)
            [2, 3, 7, 6],  # back   (+y)
            [1, 2, 6, 5],  # right  (+x)
            [0, 3, 7, 4],  # left   (-x)
        ])
    else:
        raise ValueError(cell_type)

    npc_arr = cells.shape[1]
    # Build face list and count occurrences; boundary = count == 1
    # (For simplicity here we treat "appears in a face" — exact boundary
    # detection is more involved and only needed at setup.)
    cells_np = np.asarray(cells)
    counts = np.zeros(n_nodes, dtype=np.int32)
    np.add.at(counts, cells_np.reshape(-1), 1)
    # Heuristic: nodes whose cell-count is below the max are boundary
    max_count = counts.max() if counts.size else 0
    boundary = np.where(counts < max_count)[0]
    return jnp.asarray(boundary, dtype=jnp.int32)


@dataclass(frozen=True, eq=False)
class StructuredMesh2D(Mesh):
    """A structured rectangular mesh of ``quad4`` cells.

    Built on top of :class:`Mesh`; exists mainly to expose ``nx, ny, lx, ly``
    for analytic boundary handling and to reconstruct fields as images.
    """

    nx: int = 0
    ny: int = 0
    lx: float = 0.0
    ly: float = 0.0


@dataclass(frozen=True, eq=False)
class StructuredMesh3D(Mesh):
    """A structured box mesh of ``hex8`` cells.

    Exposes ``nx, ny, nz, lx, ly, lz`` for analytic boundary handling and
    field reshaping into 3D images.
    """

    nx: int = 0
    ny: int = 0
    nz: int = 0
    lx: float = 0.0
    ly: float = 0.0
    lz: float = 0.0


def rectangular_mesh2d(
    nx: int,
    ny: int,
    lx: float = 1.0,
    ly: float = 1.0,
    *,
    cell_type: CellType = "quad4",
    origin: tuple[float, float] = (0.0, 0.0),
) -> Mesh:
    """Construct a rectangular 2D mesh.

    Parameters
    ----------
    nx, ny : int
        Number of cells in x and y. Number of nodes is ``(nx+1)*(ny+1)``.
    lx, ly : float
        Domain size in x and y.
    cell_type : {"quad4", "tri3"}
        ``quad4`` produces a regular grid; ``tri3`` splits each quad into 2 tris.
    origin : (float, float)
        Bottom-left corner.
    """
    if nx < 1 or ny < 1:
        raise ValueError("nx, ny must be >= 1")
    if cell_type not in ("quad4", "tri3"):
        raise ValueError(f"unsupported cell_type for rectangular_mesh2d: {cell_type}")

    x = np.linspace(origin[0], origin[0] + lx, nx + 1)
    y = np.linspace(origin[1], origin[1] + ly, ny + 1)
    xx, yy = np.meshgrid(x, y, indexing="xy")  # (ny+1, nx+1)
    nodes = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float64)

    # Node id helper: (i, j) -> i * (nx+1) + j  with i in [0,ny], j in [0,nx]
    # but our ravel used row-major on (ny+1, nx+1) with yy indexing -> same.
    def nid(i, j):
        return i * (nx + 1) + j

    if cell_type == "quad4":
        cells = np.zeros((nx * ny, 4), dtype=np.int32)
        k = 0
        for i in range(ny):
            for j in range(nx):
                cells[k, 0] = nid(i, j)
                cells[k, 1] = nid(i, j + 1)
                cells[k, 2] = nid(i + 1, j + 1)
                cells[k, 3] = nid(i + 1, j)
                k += 1
    else:  # tri3: each quad -> 2 tris
        cells = np.zeros((nx * ny * 2, 3), dtype=np.int32)
        k = 0
        for i in range(ny):
            for j in range(nx):
                n0 = nid(i, j)
                n1 = nid(i, j + 1)
                n2 = nid(i + 1, j + 1)
                n3 = nid(i + 1, j)
                cells[k] = [n0, n1, n2]
                cells[k + 1] = [n0, n2, n3]
                k += 2

    cells_j = jnp.asarray(cells)
    nodes_j = jnp.asarray(nodes)
    boundary = _detect_boundary_nodes(cells_j, nodes.shape[0], cell_type)
    cls = StructuredMesh2D if cell_type == "quad4" else Mesh
    if cls is StructuredMesh2D:
        return cls(
            nodes=nodes_j, cells=cells_j, cell_type=cell_type,
            boundary_node_ids=boundary,
            nx=nx, ny=ny, lx=lx, ly=ly,
        )
    return Mesh(
        nodes=nodes_j, cells=cells_j, cell_type=cell_type,
        boundary_node_ids=boundary,
    )


def rectangular_mesh3d(
    nx: int,
    ny: int,
    nz: int,
    lx: float = 1.0,
    ly: float = 1.0,
    lz: float = 1.0,
    *,
    cell_type: CellType = "hex8",
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Mesh:
    """Construct a 3D box mesh.

    Parameters
    ----------
    nx, ny, nz : int
        Number of cells in x, y, z. Number of nodes is
        ``(nx+1)*(ny+1)*(nz+1)``.
    lx, ly, lz : float
        Domain size in x, y, z.
    cell_type : {"hex8", "tet4"}
        ``hex8`` produces a regular hex grid (8 nodes/cell);
        ``tet4`` splits each hex into 6 tets.
    origin : (float, float, float)
        Bottom-left-back corner.
    """
    if min(nx, ny, nz) < 1:
        raise ValueError("nx, ny, nz must be >= 1")
    if cell_type not in ("hex8", "tet4"):
        raise ValueError(f"unsupported cell_type for rectangular_mesh3d: {cell_type}")

    x = np.linspace(origin[0], origin[0] + lx, nx + 1)
    y = np.linspace(origin[1], origin[1] + ly, ny + 1)
    z = np.linspace(origin[2], origin[2] + lz, nz + 1)
    # indexing "ij": first axis is x (so node id = i*(ny+1)*(nz+1) + j*(nz+1) + k)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    nodes = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float64)

    def nid(i, j, k):
        # i in [0, nx], j in [0, ny], k in [0, nz]
        return i * (ny + 1) * (nz + 1) + j * (nz + 1) + k

    if cell_type == "hex8":
        cells = np.zeros((nx * ny * nz, 8), dtype=np.int32)
        idx = 0
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # Hex node ordering must match shape_functions.hex8
                    #   0:(-, -, -), 1:(+, -, -), 2:(+, +, -), 3:(-, +, -)
                    #   4:(-, -, +), 5:(+, -, +), 6:(+, +, +), 7:(-, +, +)
                    n000 = nid(i,     j,     k)      # (-,-,-)
                    n100 = nid(i + 1, j,     k)      # (+,-,-)
                    n110 = nid(i + 1, j + 1, k)      # (+,+,-)
                    n010 = nid(i,     j + 1, k)      # (-,+,-)
                    n001 = nid(i,     j,     k + 1)  # (-,-,+)
                    n101 = nid(i + 1, j,     k + 1)  # (+,-,+)
                    n111 = nid(i + 1, j + 1, k + 1)  # (+,+,+)
                    n011 = nid(i,     j + 1, k + 1)  # (-,+,+)
                    cells[idx] = [n000, n100, n110, n010,
                                  n001, n101, n111, n011]
                    idx += 1
    else:  # tet4: split each hex into 6 tets (using the 6-tet Freudenthal decomposition)
        cells = np.zeros((nx * ny * nz * 6, 4), dtype=np.int32)
        idx = 0
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    n000 = nid(i,     j,     k)
                    n100 = nid(i + 1, j,     k)
                    n110 = nid(i + 1, j + 1, k)
                    n010 = nid(i,     j + 1, k)
                    n001 = nid(i,     j,     k + 1)
                    n101 = nid(i + 1, j,     k + 1)
                    n111 = nid(i + 1, j + 1, k + 1)
                    n011 = nid(i,     j + 1, k + 1)
                    # 6-tet decomposition of a hex (consistent orientation)
                    cells[idx]     = [n000, n100, n110, n111]
                    cells[idx + 1] = [n000, n110, n010, n111]
                    cells[idx + 2] = [n000, n010, n011, n111]
                    cells[idx + 3] = [n000, n011, n001, n111]
                    cells[idx + 4] = [n000, n001, n101, n111]
                    cells[idx + 5] = [n000, n101, n100, n111]
                    idx += 6

    cells_j = jnp.asarray(cells)
    nodes_j = jnp.asarray(nodes)
    boundary = _detect_boundary_nodes(cells_j, nodes.shape[0], cell_type)
    cls = StructuredMesh3D if cell_type == "hex8" else Mesh
    if cls is StructuredMesh3D:
        return cls(
            nodes=nodes_j, cells=cells_j, cell_type=cell_type,
            boundary_node_ids=boundary,
            nx=nx, ny=ny, nz=nz, lx=lx, ly=ly, lz=lz,
        )
    return Mesh(
        nodes=nodes_j, cells=cells_j, cell_type=cell_type,
        boundary_node_ids=boundary,
    )
