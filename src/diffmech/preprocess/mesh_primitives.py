"""Common AM / mechanics test-specimen mesh primitives (2D & 3D).

The structured-mesh constructors in :mod:`diffmech.preprocess.mesh_io` only
produce axis-aligned rectangles / boxes. Real mechanical and AM benchmark
specimens — discs, rings, plates with holes, notched bars, L-brackets,
cylinders — are far more common in practice. This module provides one-liner
generators for them.

Implementation
--------------
Each generator builds a structured background mesh covering the part's
bounding box, computes each cell's centroid, and keeps only the cells whose
centroid lies inside the desired geometry (a "cell mask"). The kept cells
retain their original structured connectivity, so the result is a clean
subset of quad4 / hex8 cells (or tri3 / tet4 if a split is requested).

This keeps every primitive differentiable-friendly (the mask is computed at
construction time on numpy arrays) and consistent with the rest of DiffMech's
:class:`~diffmech.core.Mesh` containers.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from diffmech.core import Mesh, rectangular_mesh2d, rectangular_mesh3d


# ---------------------------------------------------------------------------
# Helpers: mask a structured mesh to a cell subset and renumber nodes
# ---------------------------------------------------------------------------
def _apply_cell_mask_2d(
    bg: Mesh,
    nx: int,
    ny: int,
    cell_mask: np.ndarray,
    *,
    split_tri: bool = False,
) -> Mesh:
    """Keep only the quad4 cells of a structured 2D mesh flagged by ``cell_mask``.

    ``cell_mask`` is a ``(nx, ny)`` boolean array in (i, j) = (x, y) order
    matching :func:`rectangular_mesh2d`. If ``split_tri`` is requested each
    kept quad is split into two tri3 cells.
    """
    cells = np.asarray(bg.cells).reshape(nx, ny, 4)
    kept = cells[cell_mask]                       # (n_keep, 4)
    if kept.shape[0] == 0:
        raise ValueError("cell mask kept zero cells — check the geometry")

    # Renumber nodes: keep only nodes referenced by kept cells.
    used = np.unique(kept.reshape(-1))
    remap = -np.ones(bg.n_nodes, dtype=np.int32)
    remap[used] = np.arange(used.size, dtype=np.int32)
    nodes = np.asarray(bg.nodes)[used]
    kept = remap[kept]

    if split_tri:
        # Split each kept quad [n0, n1, n2, n3] (CCW) into two tris.
        n = kept.shape[0]
        tris = np.empty((n * 2, 3), dtype=np.int32)
        tris[0::2] = kept[:, [0, 1, 2]]
        tris[1::2] = kept[:, [0, 2, 3]]
        return Mesh(
            nodes=jnp.asarray(nodes),
            cells=jnp.asarray(tris),
            cell_type="tri3",
        )
    return Mesh(
        nodes=jnp.asarray(nodes),
        cells=jnp.asarray(kept),
        cell_type="quad4",
    )


def _apply_cell_mask_3d(
    bg: Mesh,
    nx: int,
    ny: int,
    nz: int,
    cell_mask: np.ndarray,
    *,
    split_tet: bool = False,
) -> Mesh:
    """Keep only the hex8 cells of a structured 3D mesh flagged by ``cell_mask``."""
    cells = np.asarray(bg.cells).reshape(nx, ny, nz, 8)
    kept = cells[cell_mask]
    if kept.shape[0] == 0:
        raise ValueError("cell mask kept zero cells — check the geometry")

    used = np.unique(kept.reshape(-1))
    remap = -np.ones(bg.n_nodes, dtype=np.int32)
    remap[used] = np.arange(used.size, dtype=np.int32)
    nodes = np.asarray(bg.nodes)[used]
    kept = remap[kept]

    if split_tet:
        # 6-tet Freudenthal decomposition consistent with rectangular_mesh3d.
        n = kept.shape[0]
        tets = np.empty((n * 6, 4), dtype=np.int32)
        # Hex node order matches rectangular_mesh3d:
        #   0:(-,-,-) 1:(+,-,-) 2:(+,+,-) 3:(-,+,-)
        #   4:(-,-,+) 5:(+,-,+) 6:(+,+,+) 7:(-,+,+)
        tet_idx = np.array([
            [0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6],
            [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6],
        ], dtype=np.int32)
        for k, tet in enumerate(tet_idx):
            tets[k::6] = kept[:, tet]
        return Mesh(
            nodes=jnp.asarray(nodes),
            cells=jnp.asarray(tets),
            cell_type="tet4",
        )
    return Mesh(
        nodes=jnp.asarray(nodes),
        cells=jnp.asarray(kept),
        cell_type="hex8",
    )


def _cell_centroids_2d(bg: Mesh, nx: int, ny: int) -> np.ndarray:
    cells = np.asarray(bg.cells).reshape(nx, ny, 4)
    pts = np.asarray(bg.nodes)[cells]              # (nx, ny, 4, 2)
    return pts.mean(axis=2)                        # (nx, ny, 2)


def _cell_centroids_3d(bg: Mesh, nx: int, ny: int, nz: int) -> np.ndarray:
    cells = np.asarray(bg.cells).reshape(nx, ny, nz, 8)
    pts = np.asarray(bg.nodes)[cells]              # (nx, ny, nz, 8, 3)
    return pts.mean(axis=3)                        # (nx, ny, nz, 3)


# ---------------------------------------------------------------------------
# 2D primitives
# ---------------------------------------------------------------------------
def disc_mesh_2d(
    radius: float = 1.0,
    n: int = 16,
    *,
    cell_type: str = "tri3",
    center=(0.0, 0.0),
) -> Mesh:
    """A solid circular disc of given ``radius`` centred at ``center``.

    Parameters
    ----------
    radius : float
        Disc radius.
    n : int
        Number of background cells along the diameter (resolution).
    cell_type : {"tri3", "quad4"}
    center : (x, y)
    """
    cx, cy = center
    L = 2.0 * radius
    bg = rectangular_mesh2d(n, n, lx=L, ly=L, origin=(cx - radius, cy - radius))
    cc = _cell_centroids_2d(bg, n, n)
    r2 = (cc[..., 0] - cx) ** 2 + (cc[..., 1] - cy) ** 2
    mask = r2 <= radius * radius
    return _apply_cell_mask_2d(bg, n, n, mask, split_tri=(cell_type == "tri3"))


def ring_mesh_2d(
    r_inner: float = 0.5,
    r_outer: float = 1.0,
    n: int = 24,
    *,
    cell_type: str = "tri3",
    center=(0.0, 0.0),
) -> Mesh:
    """An annular ring (``r_inner <= r <= r_outer``)."""
    cx, cy = center
    L = 2.0 * r_outer
    bg = rectangular_mesh2d(n, n, lx=L, ly=L, origin=(cx - r_outer, cy - r_outer))
    cc = _cell_centroids_2d(bg, n, n)
    r2 = (cc[..., 0] - cx) ** 2 + (cc[..., 1] - cy) ** 2
    mask = (r2 >= r_inner * r_inner) & (r2 <= r_outer * r_outer)
    return _apply_cell_mask_2d(bg, n, n, mask, split_tri=(cell_type == "tri3"))


def plate_with_hole_2d(
    length: float = 4.0,
    height: float = 2.0,
    hole_radius: float = 0.5,
    nx: int = 40,
    ny: int = 20,
    *,
    cell_type: str = "quad4",
    hole_center=(0.0, 0.0),
) -> Mesh:
    """A rectangular plate with a central circular hole (tension benchmark).

    The domain spans ``[-length/2, +length/2] × [-height/2, +height/2]``.
    """
    hx, hy = hole_center
    bg = rectangular_mesh2d(
        nx, ny, lx=length, ly=height,
        origin=(-length / 2.0, -height / 2.0),
    )
    cc = _cell_centroids_2d(bg, nx, ny)
    r2 = (cc[..., 0] - hx) ** 2 + (cc[..., 1] - hy) ** 2
    mask = r2 >= hole_radius * hole_radius
    return _apply_cell_mask_2d(bg, nx, ny, mask, split_tri=(cell_type == "tri3"))


def l_bracket_mesh_2d(
    a: float = 2.0,
    b: float = 2.0,
    t: float = 1.0,
    n: int = 16,
    *,
    cell_type: str = "tri3",
) -> Mesh:
    """An L-shaped bracket made of two arms of thickness ``t``.

    The bracket occupies ``[0, b] × [0, t]`` ∪ ``[0, t] × [0, a]``
    (an "L" opening to the top-right). ``a`` is the vertical arm height,
    ``b`` the horizontal arm length, ``t`` the arm thickness (with
    ``a > t`` and ``b > t`` the two arms are distinct, giving the classic
    "L"). The occupied area is ``t * (a + b - t)``.
    """
    bg = rectangular_mesh2d(n, n, lx=b, ly=a, origin=(0.0, 0.0))
    cc = _cell_centroids_2d(bg, n, n)
    mask = (cc[..., 0] <= t) | (cc[..., 1] <= t)
    return _apply_cell_mask_2d(bg, n, n, mask, split_tri=(cell_type == "tri3"))


def notched_bar_mesh_2d(
    length: float = 4.0,
    width: float = 1.0,
    notch_radius: float = 0.25,
    nx: int = 40,
    ny: int = 12,
    *,
    cell_type: str = "quad4",
    side: str = "both",
) -> Mesh:
    """A tension bar with a U-notch (one or both sides) at mid-length.

    ``side`` is ``"both"`` (symmetric notches), ``"top"``, or ``"bottom"``.
    """
    bg = rectangular_mesh2d(
        nx, ny, lx=length, ly=width, origin=(-length / 2.0, -width / 2.0),
    )
    cc = _cell_centroids_2d(bg, nx, ny)
    # Notch centred at x=0, removing cells inside a circle of given radius
    # that also lie on the requested side(s).
    r2 = (cc[..., 0]) ** 2
    notch_top = cc[..., 1] >= 0.0
    notch_bot = cc[..., 1] <= 0.0
    if side == "both":
        in_notch = r2 + (cc[..., 1]) ** 2 <= notch_radius * notch_radius
    elif side == "top":
        in_notch = notch_top & (
            r2 + (cc[..., 1] - width / 2.0) ** 2 <= notch_radius * notch_radius
        )
    elif side == "bottom":
        in_notch = notch_bot & (
            r2 + (cc[..., 1] + width / 2.0) ** 2 <= notch_radius * notch_radius
        )
    else:
        raise ValueError(f"side must be 'both', 'top', or 'bottom', got {side!r}")
    mask = ~in_notch
    return _apply_cell_mask_2d(bg, nx, ny, mask, split_tri=(cell_type == "tri3"))


# ---------------------------------------------------------------------------
# 3D primitives
# ---------------------------------------------------------------------------
def cylinder_mesh_3d(
    radius: float = 1.0,
    height: float = 2.0,
    n_radial: int = 12,
    n_height: int = 8,
    *,
    cell_type: str = "hex8",
    axis: str = "z",
) -> Mesh:
    """A solid cylinder of given ``radius`` and ``height``.

    The cylinder axis is ``axis`` (``"x"``, ``"y"``, or ``"z"``); the default
    ``"z"`` makes it stand upright in the xy-plane.
    """
    L = 2.0 * radius
    # Use a square cross-section background of n_radial × n_radial cells
    # and n_height layers along the axis.
    if axis == "z":
        n1, n2, n3 = n_radial, n_radial, n_height
        bg = rectangular_mesh3d(
            n1, n2, n3, lx=L, ly=L, lz=height,
            origin=(-radius, -radius, 0.0),
        )
        cc = _cell_centroids_3d(bg, n1, n2, n3)
        r2 = cc[..., 0] ** 2 + cc[..., 1] ** 2
    elif axis == "y":
        n1, n2, n3 = n_radial, n_height, n_radial
        bg = rectangular_mesh3d(
            n1, n2, n3, lx=L, ly=height, lz=L,
            origin=(-radius, 0.0, -radius),
        )
        cc = _cell_centroids_3d(bg, n1, n2, n3)
        r2 = cc[..., 0] ** 2 + cc[..., 2] ** 2
    elif axis == "x":
        n1, n2, n3 = n_height, n_radial, n_radial
        bg = rectangular_mesh3d(
            n1, n2, n3, lx=height, ly=L, lz=L,
            origin=(0.0, -radius, -radius),
        )
        cc = _cell_centroids_3d(bg, n1, n2, n3)
        r2 = cc[..., 1] ** 2 + cc[..., 2] ** 2
    else:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")
    mask = r2 <= radius * radius
    return _apply_cell_mask_3d(bg, n1, n2, n3, mask, split_tet=(cell_type == "tet4"))


def plate_with_hole_3d(
    length: float = 4.0,
    height: float = 2.0,
    thickness: float = 0.5,
    hole_radius: float = 0.5,
    nx: int = 20,
    ny: int = 10,
    nz: int = 4,
    *,
    cell_type: str = "hex8",
) -> Mesh:
    """A 3D rectangular plate with a through-thickness central hole."""
    bg = rectangular_mesh3d(
        nx, ny, nz, lx=length, ly=height, lz=thickness,
        origin=(-length / 2.0, -height / 2.0, -thickness / 2.0),
    )
    cc = _cell_centroids_3d(bg, nx, ny, nz)
    r2 = cc[..., 0] ** 2 + cc[..., 1] ** 2
    mask = r2 >= hole_radius * hole_radius
    return _apply_cell_mask_3d(bg, nx, ny, nz, mask, split_tet=(cell_type == "tet4"))


__all__ = [
    "disc_mesh_2d", "ring_mesh_2d", "plate_with_hole_2d",
    "l_bracket_mesh_2d", "notched_bar_mesh_2d",
    "cylinder_mesh_3d", "plate_with_hole_3d",
]
