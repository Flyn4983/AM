"""Structured Cartesian FVM grid (2D and 3D).

The grid stores cell-centred coordinates, cell volumes, face normals/areas,
and neighbour indices in a way that is jit/vmap friendly. All geometry is
precomputed at construction time (numpy) and stored as jax arrays.

The grid is *uniform* in each direction but the cell sizes may differ between
directions (``dx != dy != dz``). Boundary faces are handled via a ghost-cell
layer: the caller supplies boundary conditions as functions that fill the
ghost cells, after which the interior update is uniform.

Indexing convention
-------------------
- Cell index in 2D: ``(i, j)`` with ``i in [0, nx)``, ``j in [0, ny)``.
- Cell index in 3D: ``(i, j, k)`` with ``i in [0, nx)``, etc.
- Flattened storage: row-major with x being the fastest axis,
  ``cell_id = i * ny * nz + j * nz + k`` (with ``nz = 1`` in 2D).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class CartesianGrid:
    """Uniform Cartesian FVM grid.

    Attributes
    ----------
    nx, ny, nz : int
        Number of cells in each direction. ``nz = 1`` for 2D grids.
    lx, ly, lz : float
        Physical domain size in each direction. ``lz = 0`` for 2D grids.
    dim : int
        2 or 3.
    cell_centers : (n_cells, dim) array
        Cartesian coordinates of each cell centre.
    cell_volume : (n_cells,) array
        Volume (or area in 2D) of each cell.
    """

    nx: int
    ny: int
    nz: int
    lx: float
    ly: float
    lz: float
    dim: int
    cell_centers: jnp.ndarray
    cell_volume: jnp.ndarray

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def dx(self) -> float:
        return self.lx / self.nx

    @property
    def dy(self) -> float:
        return self.ly / self.ny

    @property
    def dz(self) -> float:
        return self.lz / self.nz if self.dim == 3 else 0.0

    def cell_id(self, i, j, k=0):
        """Flat cell id from (i, j[, k]) indices."""
        return i * (self.ny * self.nz) + j * self.nz + k

    def reshape(self, field: jnp.ndarray) -> jnp.ndarray:
        """Reshape a flat (n_cells, ...) field to its logical shape.

        In 2D -> (nx, ny, ...); in 3D -> (nx, ny, nz, ...).
        """
        if self.dim == 2:
            return field.reshape(self.nx, self.ny, *field.shape[1:])
        return field.reshape(self.nx, self.ny, self.nz, *field.shape[1:])


def cartesian_grid_2d(
    nx: int, ny: int, lx: float = 1.0, ly: float = 1.0,
    *, origin: tuple[float, float] = (0.0, 0.0),
) -> CartesianGrid:
    """Build a 2D Cartesian FVM grid."""
    if nx < 1 or ny < 1:
        raise ValueError("nx, ny must be >= 1")
    dx, dy = lx / nx, ly / ny
    x = origin[0] + (np.arange(nx) + 0.5) * dx
    y = origin[1] + (np.arange(ny) + 0.5) * dy
    xx, yy = np.meshgrid(x, y, indexing="ij")
    centers = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float64)
    vol = np.full(nx * ny, dx * dy, dtype=np.float64)
    return CartesianGrid(
        nx=nx, ny=ny, nz=1, lx=lx, ly=ly, lz=0.0, dim=2,
        cell_centers=jnp.asarray(centers), cell_volume=jnp.asarray(vol),
    )


def cartesian_grid_3d(
    nx: int, ny: int, nz: int,
    lx: float = 1.0, ly: float = 1.0, lz: float = 1.0,
    *, origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> CartesianGrid:
    """Build a 3D Cartesian FVM grid."""
    if min(nx, ny, nz) < 1:
        raise ValueError("nx, ny, nz must be >= 1")
    dx, dy, dz = lx / nx, ly / ny, lz / nz
    x = origin[0] + (np.arange(nx) + 0.5) * dx
    y = origin[1] + (np.arange(ny) + 0.5) * dy
    z = origin[2] + (np.arange(nz) + 0.5) * dz
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    centers = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float64)
    vol = np.full(nx * ny * nz, dx * dy * dz, dtype=np.float64)
    return CartesianGrid(
        nx=nx, ny=ny, nz=nz, lx=lx, ly=ly, lz=lz, dim=3,
        cell_centers=jnp.asarray(centers), cell_volume=jnp.asarray(vol),
    )


# ---------------------------------------------------------------------------
# Neighbour / face operations.
#
# We represent the cell field as a multidimensional jax array with shape
# ``(nx, ny[, nz], n_components)``. Boundary conditions are applied by padding
# the field with ghost cells along each axis, after which the interior update
# is a uniform stencil. This keeps everything jit-friendly and differentiable.
# ---------------------------------------------------------------------------
Axis = Literal[0, 1, 2]


def _pad_ghost(field: jnp.ndarray, n_ghost: int, dim: int) -> jnp.ndarray:
    """Pad ``field`` of shape (nx, ny[, nz], ...) with ghost cells.

    The ghost cells are filled with zeros; the caller is responsible for
    overwriting them with BC values via :func:`apply_boundary_conditions`.
    """
    if dim == 2:
        return jnp.pad(field, ((n_ghost, n_ghost), (n_ghost, n_ghost), (0, 0)))
    return jnp.pad(field, ((n_ghost, n_ghost),) * 3 + ((0, 0),))
