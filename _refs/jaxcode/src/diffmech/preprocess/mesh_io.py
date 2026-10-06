"""Mesh import / export via :mod:`meshio`.

Bridges between external mesh formats (VTK, VTU, Gmsh ``.msh``, Abaqus
``.inp``, Exodus ``.e``, SALOME ``.med``, ANSYS ``.ans``, …) and DiffMech's
lightweight :class:`~diffmech.core.Mesh` container.

Cell-type mapping
-----------------
================  ================  ===============
meshio cell type  DiffMech type     n_nodes
================  ================  ===============
``triangle``      ``tri3``          3
``quad``          ``quad4``         4
``tetra``         ``tet4``          4
``hexahedron``    ``hex8``          8
================  ================  ===============
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import meshio
import numpy as np

from diffmech.core import Mesh, rectangular_mesh2d, rectangular_mesh3d


# ---------------------------------------------------------------------------
# Cell-type mapping
# ---------------------------------------------------------------------------
_MESHIO_TO_DIFFMECH = {
    "triangle": "tri3",
    "quad": "quad4",
    "tetra": "tet4",
    "hexahedron": "hex8",
}
_DIFFMECH_TO_MESHIO = {v: k for k, v in _MESHIO_TO_DIFFMECH.items()}


def supported_meshio_cell_types() -> tuple[str, ...]:
    """Return the meshio cell-block types DiffMech can read."""
    return tuple(_MESHIO_TO_DIFFMECH.keys())


# ---------------------------------------------------------------------------
# Import: meshio → DiffMech
# ---------------------------------------------------------------------------
def from_meshio(mesh: meshio.Mesh, *, prefer_cell_type: str | None = None) -> Mesh:
    """Convert a :class:`meshio.Mesh` to :class:`diffmech.core.Mesh`.

    Parameters
    ----------
    mesh : meshio.Mesh
        The mesh to convert. Must contain only linear cell types supported by
        DiffMech (see :func:`supported_meshio_cell_types`). Mixed meshes are
        supported by selecting one cell type (see ``prefer_cell_type``).
    prefer_cell_type : str, optional
        If the mesh contains multiple cell types, choose the one whose meshio
        name matches (e.g. ``"quad"``). If ``None`` the most numerous cell type
        is selected.

    Returns
    -------
    diffmech Mesh
        A :class:`diffmech.core.Mesh` with ``cell_type`` set to the chosen
        DiffMech cell type. ``point_data`` and ``cell_data`` are currently
        discarded — use :mod:`diffmech.postprocess.vtk` to read fields.
    """
    # Pick the dominant / preferred cell block.
    candidates = [(cb.type, cb.data) for cb in mesh.cells]
    if not candidates:
        raise ValueError("mesh contains no cells")

    supported = [c for c in candidates if c[0] in _MESHIO_TO_DIFFMECH]
    if not supported:
        raise ValueError(
            f"none of the mesh's cell types {set(c[0] for c in candidates)} "
            f"are supported; supported types: {supported_meshio_cell_types()}"
        )

    if prefer_cell_type is not None:
        match = [c for c in supported if c[0] == prefer_cell_type]
        if not match:
            raise ValueError(
                f"preferred cell type {prefer_cell_type!r} not found among "
                f"supported cells: {set(c[0] for c in supported)}"
            )
        chosen = match[0]
    else:
        # Most numerous
        chosen = max(supported, key=lambda c: len(c[1]))

    cell_type_meshio, cells = chosen
    cell_type = _MESHIO_TO_DIFFMECH[cell_type_meshio]

    nodes = np.asarray(mesh.points, dtype=np.float64)
    # meshio may give 3D coords for a 2D mesh (z=0); drop the all-zero axis.
    if nodes.shape[1] == 3 and np.allclose(nodes[:, 2], 0.0):
        nodes = nodes[:, :2]
    cells = np.asarray(cells, dtype=np.int32)

    return Mesh(
        nodes=jnp.asarray(nodes),
        cells=jnp.asarray(cells),
        cell_type=cell_type,
    )


def read_mesh(path: str | Path, *, prefer_cell_type: str | None = None) -> Mesh:
    """Read a mesh from a file (any format supported by meshio).

    Examples
    --------
    >>> mesh = read_mesh("part.msh")           # Gmsh
    >>> mesh = read_mesh("part.inp")           # Abaqus
    >>> mesh = read_mesh("part.vtu")           # VTU
    """
    m = meshio.read(str(path))
    return from_meshio(m, prefer_cell_type=prefer_cell_type)


# ---------------------------------------------------------------------------
# Export: DiffMech → meshio
# ---------------------------------------------------------------------------
def to_meshio(
    mesh: Mesh,
    *,
    point_data: dict | None = None,
    cell_data: dict | None = None,
) -> meshio.Mesh:
    """Convert a DiffMech :class:`Mesh` to a :class:`meshio.Mesh`.

    Parameters
    ----------
    mesh : Mesh
    point_data : dict of {name: (n_nodes, ...) array}, optional
        Per-node fields (e.g. displacement, velocity).
    cell_data : dict of {name: (n_cells, ...) array}, optional
        Per-cell fields (e.g. element material id, averaged stress).

    Returns
    -------
    meshio.Mesh
    """
    nodes = np.asarray(mesh.nodes, dtype=np.float64)
    cells = np.asarray(mesh.cells, dtype=np.int32)
    cell_type_meshio = _DIFFMECH_TO_MESHIO[mesh.cell_type]
    cell_block = meshio.CellBlock(cell_type_meshio, cells)

    # Force 3D coords for meshio (it always stores xyz); pad 2D with z=0.
    if nodes.shape[1] == 2:
        nodes = np.hstack([nodes, np.zeros((len(nodes), 1), dtype=np.float64)])

    # meshio expects cell_data as a *list* of arrays, one entry per cell block.
    # Since DiffMech always has exactly one cell block, wrap each value in a list.
    cell_data_meshio = None
    if cell_data:
        cell_data_meshio = {k: [np.asarray(v)] for k, v in cell_data.items()}

    return meshio.Mesh(
        points=nodes,
        cells=[cell_block],
        point_data=point_data or {},
        cell_data=cell_data_meshio,
    )


def write_mesh(
    path: str | Path,
    mesh: Mesh,
    *,
    point_data: dict | None = None,
    cell_data: dict | None = None,
) -> None:
    """Write a DiffMech mesh (+ optional fields) to a file.

    The output format is inferred from the file extension. Common choices:

    ==============  ===========================================================
    ``.vtk``       Legacy VTK (ASCII / binary)
    ``.vtu``       VTU (unstructured grid, the recommended ParaView format)
    ``.msh``       Gmsh (versions 2 or 4)
    ``.inp``       Abaqus input
    ``.e`` / ``.exo``  ExodusII
    ``.med``       SALOME MED
    ``.xdmf`` / ``.xmf``  XDMF
    ==============  ===========================================================
    """
    m = to_meshio(mesh, point_data=point_data, cell_data=cell_data)
    meshio.write(str(path), m)


# ---------------------------------------------------------------------------
# Convenience constructors re-exported here
# ---------------------------------------------------------------------------
def structured_quad_mesh_2d(nx, ny, lx=1.0, ly=1.0, *, origin=(0.0, 0.0)):
    """Alias for :func:`diffmech.core.rectangular_mesh2d` with ``quad4``."""
    return rectangular_mesh2d(nx, ny, lx, ly, cell_type="quad4", origin=origin)


def structured_tri_mesh_2d(nx, ny, lx=1.0, ly=1.0, *, origin=(0.0, 0.0)):
    """Alias for :func:`diffmech.core.rectangular_mesh2d` with ``tri3``."""
    return rectangular_mesh2d(nx, ny, lx, ly, cell_type="tri3", origin=origin)


def structured_hex_mesh_3d(nx, ny, nz, lx=1.0, ly=1.0, lz=1.0, *,
                           origin=(0.0, 0.0, 0.0)):
    """Alias for :func:`diffmech.core.rectangular_mesh3d` with ``hex8``."""
    return rectangular_mesh3d(nx, ny, nz, lx, ly, lz, cell_type="hex8",
                             origin=origin)


def structured_tet_mesh_3d(nx, ny, nz, lx=1.0, ly=1.0, lz=1.0, *,
                           origin=(0.0, 0.0, 0.0)):
    """Alias for :func:`diffmech.core.rectangular_mesh3d` with ``tet4``."""
    return rectangular_mesh3d(nx, ny, nz, lx, ly, lz, cell_type="tet4",
                             origin=origin)
