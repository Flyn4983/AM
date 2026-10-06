"""VTK / VTU / PVD writers for ParaView visualisation.

Three writers are provided:

- :func:`write_vtu` — write an unstructured-grid VTU file (the recommended
  ParaView format) with point and cell data. Wraps :mod:`meshio`.
- :func:`write_vtk` — write a legacy VTK file (ASCII / binary) via meshio.
- :class:`PVDSeries` — write a series of VTU files plus a master ``.pvd``
  file that ParaView opens as a time-animation.

For typical mechanics outputs (displacement field, stress field, phase
field, …), use the convenience :class:`FieldCollection` to bundle per-node
and per-cell fields, then call :func:`write_vtu`.

Example
-------
::

    from diffmech.postprocess import FieldCollection, write_vtu

    fields = FieldCollection(
        point_data={"displacement": U.reshape(n_nodes, dim),
                    "phase_field": phi.ravel()},
        cell_data={"equivalent_plastic_strain": eps_eq_per_cell},
    )
    write_vtu("solution.vtu", mesh, fields)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from diffmech.core import Mesh
from diffmech.preprocess.mesh_io import to_meshio


@dataclass
class FieldCollection:
    """Bundle of point and cell fields to attach to a mesh on export.

    Parameters
    ----------
    point_data : dict of {name: (n_nodes, ...) array}
        Per-node fields, e.g. ``"displacement"`` of shape ``(n_nodes, dim)``.
    cell_data : dict of {name: (n_cells, ...) array}
        Per-cell fields, e.g. ``"von_mises_stress"`` of shape ``(n_cells,)``.
    """

    point_data: dict = field(default_factory=dict)
    cell_data: dict = field(default_factory=dict)

    def add_point(self, name: str, values) -> "FieldCollection":
        self.point_data[name] = np.asarray(values)
        return self

    def add_cell(self, name: str, values) -> "FieldCollection":
        self.cell_data[name] = np.asarray(values)
        return self


def _split_point_cell(fields: FieldCollection | None):
    if fields is None:
        return {}, {}
    return fields.point_data, fields.cell_data


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def write_vtu(
    path: str | Path,
    mesh: Mesh,
    fields: FieldCollection | None = None,
    *,
    deformed: bool = False,
    displacement_field: str | None = None,
) -> None:
    """Write a VTU (unstructured-grid) file readable by ParaView.

    Parameters
    ----------
    path : path
        Output file (should end in ``.vtu``).
    mesh : Mesh
    fields : FieldCollection, optional
        Per-node and per-cell fields to attach.
    deformed : bool
        If True, displace the node coordinates by the ``displacement_field``
        per-node field (so ParaView shows the deformed shape). The original
        coordinates are also exported as ``reference_coords``.
    displacement_field : str, optional
        Name of the per-node displacement field to use when ``deformed=True``.
        Defaults to ``"displacement"``.
    """
    point_data, cell_data = _split_point_cell(fields)

    # Optionally deform the mesh for visualisation.
    if deformed:
        fname = displacement_field or "displacement"
        if fname not in point_data:
            raise ValueError(
                f"deformed=True requires a point-data field {fname!r}"
            )
        disp = np.asarray(point_data[fname])
        nodes = np.asarray(mesh.nodes, dtype=np.float64)
        # Broadcast displacement to the same number of spatial dims as nodes.
        if disp.ndim == 1:
            disp = disp.reshape(-1, 1)
        # Pad / truncate displacement columns to match node dim.
        ddim = disp.shape[1]
        ndim = nodes.shape[1]
        if ddim < ndim:
            disp = np.hstack([disp, np.zeros((disp.shape[0], ndim - ddim))])
        elif ddim > ndim:
            disp = disp[:, :ndim]
        # Keep the reference coordinates as an extra field.
        point_data = {**point_data, "reference_coords": nodes}
        # Build a deformed copy of the mesh.
        from diffmech.core import Mesh as _Mesh
        deformed_nodes = nodes + disp
        mesh = _Mesh(nodes=jnp.asarray(deformed_nodes),
                     cells=mesh.cells, cell_type=mesh.cell_type)

    m = to_meshio(mesh, point_data=point_data, cell_data=cell_data)
    m.write(str(path))


def write_vtk(
    path: str | Path,
    mesh: Mesh,
    fields: FieldCollection | None = None,
    *,
    binary: bool = True,
) -> None:
    """Write a legacy VTK file (ASCII if ``binary=False``)."""
    point_data, cell_data = _split_point_cell(fields)
    m = to_meshio(mesh, point_data=point_data, cell_data=cell_data)
    kwargs = {"binary": binary} if binary else {"binary": False}
    m.write(str(path), **kwargs)


# ---------------------------------------------------------------------------
# PVD time series (one .pvd master + one .vtu per timestep)
# ---------------------------------------------------------------------------
class PVDSeries:
    """Time-stepping writer that emits a ParaView ``.pvd`` animation.

    Usage
    -----
    ::

        series = PVDSeries("simulation")
        for i, (t, fields) in enumerate(time_steps):
            series.write(t, mesh, fields)
        series.close()  # writes simulation.pvd
    """

    def __init__(self, name: str, *, directory: str | Path = "."):
        self.name = name
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._frames: list[tuple[float, str]] = []

    def write(self, time: float, mesh: Mesh, fields: FieldCollection | None = None):
        """Write one frame at simulation time ``time``."""
        idx = len(self._frames)
        vtu_name = f"{self.name}_{idx:06d}.vtu"
        vtu_path = self.directory / vtu_name
        write_vtu(vtu_path, mesh, fields)
        self._frames.append((float(time), vtu_name))

    def close(self) -> Path:
        """Write the master ``.pvd`` file and return its path."""
        pvd_path = self.directory / f"{self.name}.pvd"
        lines = [
            '<?xml version="1.0"?>',
            '<VTKFile type="Collection" version="0.1" '
            'byte_order="LittleEndian">',
            "  <Collection>",
        ]
        for t, fname in self._frames:
            lines.append(
                f'    <DataSet timestep="{t:g}" group="" part="0" '
                f'file="{fname}"/>'
            )
        lines.extend(["  </Collection>", "</VTKFile>", ""])
        pvd_path.write_text("\n".join(lines))
        return pvd_path

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()
        return False


# ---------------------------------------------------------------------------
# Common per-cell field computations (helpful before export)
# ---------------------------------------------------------------------------
def nodal_displacement_field(U: jnp.ndarray, n_nodes: int, dim: int) -> np.ndarray:
    """Reshape a flat DOF vector ``U`` to ``(n_nodes, dim)``."""
    return np.asarray(U).reshape(n_nodes, dim)


def von_mises_stress_field(stress_per_cell: jnp.ndarray) -> np.ndarray:
    """Compute von Mises equivalent stress from per-cell stress tensors.

    Parameters
    ----------
    stress_per_cell : (n_cells, dim, dim) or (n_cells, 6) array
        Either a full stress tensor per cell or its Voigt vector
        ``[σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy]`` (3D) /
        ``[σ_xx, σ_yy, σ_xy]`` (2D).
    """
    s = np.asarray(stress_per_cell)
    if s.ndim == 2 and s.shape[1] in (3, 6):
        # Voigt form
        if s.shape[1] == 3:
            sxx, syy, sxy = s[:, 0], s[:, 1], s[:, 2]
            vm = np.sqrt(sxx ** 2 - sxx * syy + syy ** 2 + 3.0 * sxy ** 2)
        else:
            sxx, syy, szz, syz, sxz, sxy = (
                s[:, 0], s[:, 1], s[:, 2], s[:, 3], s[:, 4], s[:, 5]
            )
            vm = np.sqrt(0.5 * (
                (sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2 +
                6.0 * (sxy ** 2 + syz ** 2 + sxz ** 2)
            ))
        return vm
    # Full tensor form
    if s.ndim == 3:
        dim = s.shape[1]
        if dim == 2:
            sxx, syy, sxy = s[:, 0, 0], s[:, 1, 1], s[:, 0, 1]
            return np.sqrt(sxx ** 2 - sxx * syy + syy ** 2 + 3.0 * sxy ** 2)
        if dim == 3:
            sxx, syy, szz = s[:, 0, 0], s[:, 1, 1], s[:, 2, 2]
            sxy, syz, sxz = s[:, 0, 1], s[:, 1, 2], s[:, 0, 2]
            return np.sqrt(0.5 * (
                (sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2 +
                6.0 * (sxy ** 2 + syz ** 2 + sxz ** 2)
            ))
    raise ValueError(f"unrecognised stress shape {s.shape}")


def equivalent_strain_field(strain_per_cell: jnp.ndarray) -> np.ndarray:
    """Equivalent (von Mises) strain from per-cell strain tensors.

    Same shape conventions as :func:`von_mises_stress_field`.
    """
    return von_mises_stress_field(strain_per_cell) / np.sqrt(3.0)
