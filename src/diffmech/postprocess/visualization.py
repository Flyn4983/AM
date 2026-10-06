"""Quick matplotlib visualisation for 2D / 3D fields.

Designed for fast iteration in a notebook — *not* a publication-quality
plotting library. For production rendering, export to VTU via
:mod:`diffmech.postprocess.vtk` and open in ParaView.

All functions accept either:

- a :class:`~diffmech.core.Mesh` plus per-node / per-cell fields, or
- raw coordinate + value arrays.

2D plots use ``matplotlib``'s ``tripcolor`` / ``tricontourf`` for
unstructured meshes and ``imshow`` for structured grids. 3D plots use
scatters / slicers.
"""

from __future__ import annotations

from typing import Iterable

import jax.numpy as jnp
import numpy as np

# Lazy-import matplotlib so the module imports even when matplotlib's backend
# is unavailable (e.g. headless CI without Agg configured).
def _mpl():
    import matplotlib
    if matplotlib.get_backend().lower() not in ("agg", "module://matplotlib_inline.backend_inline"):
        # Don't override an explicitly chosen backend, but make sure one is set.
        pass
    import matplotlib.pyplot as plt
    return plt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _triangles(mesh) -> np.ndarray:
    """Return a (n_triangles, 3) triangulation of the mesh.

    For tri3 meshes this is just the cells. For quad4 each quad is split into
    two triangles. For tet4 / hex8 we project onto the xy-plane and return the
    outer faces (best-effort; 3D meshes should really be viewed in ParaView).
    """
    cells = np.asarray(mesh.cells)
    ct = mesh.cell_type
    if ct == "tri3":
        return cells
    if ct == "quad4":
        # Each quad -> two triangles (0,1,2) and (0,2,3)
        n = cells.shape[0]
        tris = np.empty((2 * n, 3), dtype=np.int32)
        tris[0::2] = cells[:, [0, 1, 2]]
        tris[1::2] = cells[:, [0, 2, 3]]
        return tris
    # 3D: project to xy and return the boundary triangle faces.
    # For a quick visual this is approximate but usually acceptable.
    from diffmech.preprocess.mesh_tools import boundary_node_ids
    from diffmech.preprocess.mesh_tools import _cell_faces
    faces = _cell_faces(ct)
    if ct == "tet4":
        return cells  # all 4 triangles per tet (overdraw is fine for scatter)
    if ct == "hex8":
        # Use the 6 quad faces, each split into 2 triangles.
        n = cells.shape[0]
        tris = np.empty((2 * 6 * n, 3), dtype=np.int32)
        for i, f in enumerate(faces):
            tris[2 * i::12] = cells[:, [f[0], f[1], f[2]]]
            tris[2 * i + 1::12] = cells[:, [f[0], f[2], f[3]]]
        return tris
    raise ValueError(f"unsupported cell_type {ct!r}")


# ---------------------------------------------------------------------------
# 2D field plots
# ---------------------------------------------------------------------------
def plot_field_2d(
    mesh,
    field,
    *,
    ax=None,
    cmap="viridis",
    title: str | None = None,
    colorbar: bool = True,
    shading: str = "gouraud",
    show_mesh: bool = False,
):
    """Plot a 2D scalar field on the mesh.

    Parameters
    ----------
    mesh : Mesh
    field : (n_nodes,) or (n_cells,) array
        Per-node fields are interpolated; per-cell fields are flat-shaded.
    shading : ``"gouraud"`` (smooth, requires per-node) or ``"flat"``
    show_mesh : draw the mesh edges on top of the field
    """
    plt = _mpl()
    nodes = np.asarray(mesh.nodes)
    if nodes.shape[1] == 3:
        nodes = nodes[:, :2]
    field = np.asarray(field).ravel()
    tris = _triangles(mesh)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    # tripcolor handles both flat and gouraud shading
    tcf = ax.tripcolor(nodes[:, 0], nodes[:, 1], tris, field,
                       shading=shading, cmap=cmap)
    if colorbar:
        plt.colorbar(tcf, ax=ax)
    if title:
        ax.set_title(title)
    ax.set_aspect("equal")
    if show_mesh:
        ax.triplot(nodes[:, 0], nodes[:, 1], tris, color="k", lw=0.3, alpha=0.5)
    return ax


def plot_contour_2d(
    mesh,
    field,
    *,
    ax=None,
    levels: int | Iterable = 20,
    cmap="viridis",
    title: str | None = None,
    colorbar: bool = True,
):
    """Filled contour plot of a 2D per-node scalar field."""
    plt = _mpl()
    nodes = np.asarray(mesh.nodes)
    if nodes.shape[1] == 3:
        nodes = nodes[:, :2]
    field = np.asarray(field).ravel()
    tris = _triangles(mesh)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    tcf = ax.tricontourf(nodes[:, 0], nodes[:, 1], tris, field,
                         levels=levels, cmap=cmap)
    if colorbar:
        plt.colorbar(tcf, ax=ax)
    if title:
        ax.set_title(title)
    ax.set_aspect("equal")
    return ax


def plot_quiver_2d(
    mesh,
    vector_field,
    *,
    ax=None,
    scale: float | None = None,
    title: str | None = None,
    color: str = "k",
    alpha: float = 1.0,
):
    """Quiver (arrow) plot of a 2D vector field at the mesh nodes.

    Parameters
    ----------
    vector_field : (n_nodes, 2) array
    scale : arrow scaling. ``None`` lets matplotlib auto-scale.
    """
    plt = _mpl()
    nodes = np.asarray(mesh.nodes)
    if nodes.shape[1] == 3:
        nodes = nodes[:, :2]
    vf = np.asarray(vector_field)
    if vf.ndim == 1:
        vf = vf.reshape(-1, 2)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    ax.quiver(nodes[:, 0], nodes[:, 1], vf[:, 0], vf[:, 1],
              scale=scale, color=color, alpha=alpha)
    if title:
        ax.set_title(title)
    ax.set_aspect("equal")
    return ax


def plot_deformed_mesh_2d(
    mesh,
    U,
    *,
    ax=None,
    scale_factor: float = 1.0,
    title: str | None = None,
    color: str = "C0",
    show_undeformed: bool = True,
):
    """Plot the deformed 2D mesh (optionally overlaying the reference mesh).

    Parameters
    ----------
    U : (n_nodes * dim,) or (n_nodes, dim) nodal displacement vector
    scale_factor : magnify displacements for visibility
    """
    plt = _mpl()
    nodes = np.asarray(mesh.nodes)
    if nodes.shape[1] == 3:
        nodes = nodes[:, :2]
    U = np.asarray(U)
    if U.ndim == 1:
        U = U.reshape(-1, nodes.shape[1])
    deformed = nodes + scale_factor * U[:, :nodes.shape[1]]
    tris = _triangles(mesh)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    if show_undeformed:
        ax.triplot(nodes[:, 0], nodes[:, 1], tris, color="gray", lw=0.3,
                   alpha=0.5, label="reference")
    ax.triplot(deformed[:, 0], deformed[:, 1], tris, color=color, lw=0.8,
               label="deformed")
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    ax.legend()
    return ax


# ---------------------------------------------------------------------------
# 3D scatter / slice plots
# ---------------------------------------------------------------------------
def plot_scatter_3d(
    mesh,
    field=None,
    *,
    ax=None,
    cmap="viridis",
    title: str | None = None,
    s: float = 8.0,
):
    """3D scatter plot of mesh nodes, coloured by an optional scalar field.

    Useful for particle methods (DEM, SPH, MPM) and for inspecting 3D FEM
    results at a glance. For full 3D visualisation use ParaView / VTU.
    """
    plt = _mpl()
    nodes = np.asarray(mesh.nodes)
    if nodes.shape[1] == 2:
        nodes = np.hstack([nodes, np.zeros((len(nodes), 1))])
    if ax is None:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
    if field is not None:
        sc = ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2],
                        c=np.asarray(field).ravel(), cmap=cmap, s=s)
        plt.colorbar(sc, ax=ax, shrink=0.6)
    else:
        ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], s=s)
    if title:
        ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    return ax


def plot_slice_3d(
    mesh,
    field,
    *,
    plane: str = "xy",
    offset: float | None = None,
    ax=None,
    cmap="viridis",
    title: str | None = None,
    colorbar: bool = True,
):
    """Plot a 2D slice of a 3D scalar field.

    Parameters
    ----------
    plane : ``"xy"``, ``"xz"`` or ``"yz"``
    offset : position of the slice along the perpendicular axis. Defaults to
        the domain mid-plane.
    """
    plt = _mpl()
    nodes = np.asarray(mesh.nodes)
    field = np.asarray(field).ravel()
    if plane == "xy":
        a1, a2, an = 0, 1, 2
    elif plane == "xz":
        a1, a2, an = 0, 2, 1
    elif plane == "yz":
        a1, a2, an = 1, 2, 0
    else:
        raise ValueError(f"plane must be 'xy', 'xz' or 'yz', got {plane!r}")
    if offset is None:
        offset = 0.5 * (nodes[:, an].min() + nodes[:, an].max())
    tol = 1e-9 + 0.01 * (nodes[:, an].max() - nodes[:, an].min())
    mask = np.abs(nodes[:, an] - offset) < tol
    if mask.sum() < 3:
        raise ValueError(
            f"slice plane {plane}={offset:g} contains fewer than 3 nodes; "
            "increase tolerance or choose a different offset"
        )
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(nodes[mask, a1], nodes[mask, a2], c=field[mask],
                    cmap=cmap, s=20)
    if colorbar:
        plt.colorbar(sc, ax=ax)
    if title:
        ax.set_title(title)
    ax.set_xlabel("xyz"[a1]); ax.set_ylabel("xyz"[a2])
    ax.set_aspect("equal")
    return ax
