"""Layered mesh + differentiable element-activation field.

The core idea that makes AM process simulation differentiable: instead of
the discrete "element birth/death" used by Abaqus/ANSYS (which is a hard
on/off switch — non-differentiable), we represent each element's deposited
state by a smooth **activation field**

    α(x, t) = σ((t − t_act(x)) / τ) ∈ (0, 1)

where ``t_act(x)`` is the time at which the layer containing ``x`` begins
depositing and ``τ`` is a small activation timescale. α smoothly ramps from
0 (not yet deposited, "soft") to 1 (fully deposited, full stiffness).

The element stiffness and the thermal source are both gated by α, so the
entire build history is one differentiable function of (process parameters,
geometry) → (residual stress, distortion).

This module builds, at *setup time*:
- A structured hex8 mesh covering the part bounding box, sliced into layers.
- A per-cell ``layer_id`` array mapping each cell to its build layer.
- A per-cell ``activation_time`` array (cumulative, from the scan paths).
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from diffmech.core import Mesh


@dataclass(frozen=True)
class LayeredMesh:
    """A structured hex8 mesh with per-cell layer assignment.

    Attributes
    ----------
    mesh : Mesh
        The underlying hex8 mesh (3D).
    layer_id : (n_cells,) int array
        Build-layer index of each cell (0 = bottom layer).
    activation_time : (n_cells,) float array
        Physical time at which each cell begins depositing [s].
    layer_zs : (n_layers,) float array
        z-height of each layer's mid-plane.
    n_layers : int
    """

    mesh: Mesh
    layer_id: jnp.ndarray
    activation_time: jnp.ndarray
    layer_zs: np.ndarray
    n_layers: int

    @property
    def n_cells(self) -> int:
        return self.mesh.n_cells

    @property
    def cell_centers(self) -> jnp.ndarray:
        """(n_cells, 3) cell-centre coordinates."""
        coords = self.mesh.cell_coords  # (n_cells, 8, 3)
        return jnp.mean(coords, axis=1)

    def activation_field(self, t: jnp.ndarray, tau: float = 1e-3) -> jnp.ndarray:
        """Smooth element activation α(x, t) ∈ (0, 1) at time ``t``.

        Parameters
        ----------
        t : current physical time [s].
        tau : activation timescale (small → sharp switch, large → smoother).
        """
        return jax.nn.sigmoid((t - self.activation_time) / tau)


def build_layered_mesh(
    layer_zs: np.ndarray,
    bbox,
    *,
    nx: int = 16, ny: int = 16,
    cells_per_layer: int = 2,
) -> LayeredMesh:
    """Build a structured hex8 mesh sliced into layers.

    The mesh covers the part bounding box ``bbox = [(xmin,xmax),(ymin,ymax),
    (zmin,zmax)]`` in the xy-plane and the build range in z (one or more cells
    per layer to resolve through-thickness gradients).

    Parameters
    ----------
    layer_zs : (n_layers,) array of layer mid-plane z-heights.
    bbox : bounding box [(xmin,xmax),(ymin,ymax),(zmin,zmax)].
    nx, ny : cells per direction in-plane.
    cells_per_layer : number of hex cells stacked within each layer (≥1).
    """
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bbox
    n_layers = len(layer_zs)
    nz = n_layers * cells_per_layer

    # Build a structured hex8 grid.
    xs = np.linspace(xmin, xmax, nx + 1)
    ys = np.linspace(ymin, ymax, ny + 1)
    zs = np.linspace(zmin, zmax, nz + 1)

    # Node coordinates: (nx+1)*(ny+1)*(nz+1) nodes, ordered z-fastest.
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    nodes = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float64)

    def node_id(i, j, k):
        return i * (ny + 1) * (nz + 1) + j * (nz + 1) + k

    cells = []
    layer_id_per_cell = []
    for i in range(nx):
        for j in range(ny):
            for kz in range(nz):
                # Hex8 node ordering (VTK): 8 corners of the cube.
                n0 = node_id(i, j, kz)
                n1 = node_id(i + 1, j, kz)
                n2 = node_id(i + 1, j + 1, kz)
                n3 = node_id(i, j + 1, kz)
                n4 = node_id(i, j, kz + 1)
                n5 = node_id(i + 1, j, kz + 1)
                n6 = node_id(i + 1, j + 1, kz + 1)
                n7 = node_id(i, j + 1, kz + 1)
                cells.append([n0, n1, n2, n3, n4, n5, n6, n7])
                layer_id_per_cell.append(kz // cells_per_layer)

    cells = np.asarray(cells, dtype=np.int64)
    layer_ids = np.asarray(layer_id_per_cell, dtype=np.int64)

    mesh = Mesh(
        nodes=jnp.asarray(nodes),
        cells=jnp.asarray(cells),
        cell_type="hex8",
    )
    return LayeredMesh(
        mesh=mesh,
        layer_id=jnp.asarray(layer_ids),
        activation_time=jnp.zeros(len(cells)),  # filled by assign_activation_times
        layer_zs=layer_zs,
        n_layers=n_layers,
    )


def assign_activation_times(
    layered: LayeredMesh, layer_start_times: np.ndarray,
) -> LayeredMesh:
    """Fill in the per-cell activation time from per-layer start times."""
    starts = np.asarray(layer_start_times, dtype=np.float64)
    # Per-cell activation time = its layer's start time.
    layer_ids = np.asarray(layered.layer_id)
    cell_times = starts[layer_ids]
    return LayeredMesh(
        mesh=layered.mesh,
        layer_id=layered.layer_id,
        activation_time=jnp.asarray(cell_times),
        layer_zs=layered.layer_zs,
        n_layers=layered.n_layers,
    )


def part_cell_mask(layered: LayeredMesh, sdf) -> np.ndarray:
    """Boolean mask of cells whose centre lies inside the part SDF.

    Cells outside the part are excluded from the build (they never activate).
    This is a setup-time (numpy) routine so the SDF can be evaluated pointwise.
    """
    centres = np.asarray(layered.cell_centers)
    s = np.asarray(sdf(jnp.asarray(centres)))
    return s < 0.0


__all__ = [
    "LayeredMesh",
    "build_layered_mesh", "assign_activation_times", "part_cell_mask",
]
