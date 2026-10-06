"""Fluent builder for a ready-to-solve FEM problem.

Assembling a simulation from a mesh, a material, a set of boundary
conditions and a load case is the most repetitive boilerplate in everyday
computational-mechanics work. :class:`ProblemSetup` provides a small fluent
API so a complete linear-elastic solve — with stress recovery and VTU
export — becomes a one-liner::

    from diffmech.preprocess import ProblemSetup, plate_with_hole_2d, get_material

    result = (ProblemSetup()
              .mesh(plate_with_hole_2d(nx=40, ny=40))
              .material(get_material("Ti6Al4V"))
              .tension(axis=0, strain=0.01)
              .solve()
              .export("plate.vtu")
              .summary())

The builder also supports a quick ``.preview()`` that draws the mesh and
the clamped/loaded boundaries without running a solve — handy for catching
BC mistakes early.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import jax.numpy as jnp
import numpy as np

from diffmech.core import Mesh
from diffmech.materials import LinearElasticIsotropic
from diffmech.methods.fem import solve_linear_elastic, recover_strain_stress
from diffmech.postprocess.vtk import write_vtu
from diffmech.postprocess.derived_fields import stress_field_collection
from diffmech.preprocess.load_cases import LoadCase


@dataclass
class ProblemResult:
    """Output of :meth:`ProblemSetup.solve`."""

    U: jnp.ndarray
    mesh: Mesh
    material: LinearElasticIsotropic
    load_case: LoadCase
    strain: object = None
    stress: object = None

    def fields(self):
        """Build a :class:`FieldCollection` of derived engineering fields."""
        return stress_field_collection(self.mesh, self.U, self.material)

    def export(self, path: str | Path) -> "ProblemResult":
        write_vtu(path, self.mesh, fields=self.fields())
        return self

    def summary(self) -> str:
        u = np.asarray(self.U)
        vm = None
        if self.stress is not None:
            from diffmech.postprocess.derived_fields import von_mises_from_stress
            vm = np.asarray(von_mises_from_stress(self.stress))
        lines = [
            f"Problem summary",
            f"  nodes        : {self.mesh.n_nodes}",
            f"  cells        : {self.mesh.n_cells} ({self.mesh.cell_type})",
            f"  dofs         : {u.shape[0]}",
            f"  |U|_max      : {float(np.max(np.abs(u))):.6e}",
        ]
        if vm is not None:
            lines += [
                f"  von Mises max: {float(vm.max()):.6e}",
                f"  von Mises min: {float(vm.min()):.6e}",
            ]
        s = "\n".join(lines)
        print(s)
        return s


class ProblemSetup:
    """Fluent builder for a linear-elastic FEM problem.

    Call :meth:`mesh`, :meth:`material` (or :meth:`material_E_nu`), then one
    of the load-case setters (:meth:`tension`, :meth:`compression`,
    :meth:`shear`, :meth:`bending`, :meth:`gravity_load`, :meth:`load_case`),
    and finally :meth:`solve`.
    """

    def __init__(self):
        self._mesh: Optional[Mesh] = None
        self._material: Optional[LinearElasticIsotropic] = None
        self._load_case: Optional[LoadCase] = None
        self._dim: Optional[int] = None

    # -- configuration -------------------------------------------------------
    def mesh(self, m: Mesh) -> "ProblemSetup":
        self._mesh = m
        self._dim = m.dim
        return self

    def material(self, mat) -> "ProblemSetup":
        """Use a :class:`MaterialParams` (from the AM library) or a
        :class:`LinearElasticIsotropic` directly."""
        if isinstance(mat, LinearElasticIsotropic):
            self._material = mat
        else:
            # MaterialParams / duck-typed object with E, nu
            self._material = LinearElasticIsotropic(
                E=float(getattr(mat, "E", getattr(mat, "youngs_modulus", 1.0))),
                nu=float(getattr(mat, "nu", getattr(mat, "poissons_ratio", 0.3))),
            )
        return self

    def material_E_nu(self, E: float, nu: float) -> "ProblemSetup":
        self._material = LinearElasticIsotropic(E=E, nu=nu)
        return self

    def load_case(self, case: LoadCase) -> "ProblemSetup":
        self._load_case = case
        return self

    # -- convenience load-case shortcuts ------------------------------------
    def _require_mesh(self):
        if self._mesh is None:
            raise ValueError("call .mesh(...) first")

    def tension(self, *, axis: int = 0, strain: float = 0.01) -> "ProblemSetup":
        self._require_mesh()
        from diffmech.preprocess.load_cases import uniaxial_tension
        return self._apply_lc(uniaxial_tension(self._mesh, axis=axis, strain=strain))

    def compression(self, *, axis: int = 0, strain: float = -0.01) -> "ProblemSetup":
        self._require_mesh()
        from diffmech.preprocess.load_cases import uniaxial_compression
        return self._apply_lc(uniaxial_compression(self._mesh, axis=axis, strain=strain))

    def shear(self, *, strain: float = 0.01) -> "ProblemSetup":
        self._require_mesh()
        from diffmech.preprocess.load_cases import simple_shear
        return self._apply_lc(simple_shear(self._mesh, strain=strain))

    def bending(self, *, curvature: float = 0.01) -> "ProblemSetup":
        self._require_mesh()
        from diffmech.preprocess.load_cases import bending
        return self._apply_lc(bending(self._mesh, curvature=curvature))

    def gravity_load(self, *, g: float = 9.81, direction: int = -1,
                     clamped_face_axis: int = 0,
                     clamped_face_side: str = "min") -> "ProblemSetup":
        self._require_mesh()
        from diffmech.preprocess.load_cases import gravity
        return self._apply_lc(gravity(self._mesh, g=g, direction=direction,
                                      clamped_face_axis=clamped_face_axis,
                                      clamped_face_side=clamped_face_side))

    def _apply_lc(self, case: LoadCase) -> "ProblemSetup":
        self._load_case = case
        return self

    # -- run -----------------------------------------------------------------
    def solve(self) -> ProblemResult:
        self._check()
        U = solve_linear_elastic(
            self._mesh, self._material, list(self._load_case.dirichlet_bcs),
            body_force=self._load_case.body_force, dim=self._dim,
        )
        strain, stress = recover_strain_stress(self._mesh, U, self._material,
                                               dim=self._dim)
        return ProblemResult(U=U, mesh=self._mesh, material=self._material,
                             load_case=self._load_case, strain=strain,
                             stress=stress)

    # -- preview (no solve) --------------------------------------------------
    def preview(self, path: str | Path | None = None):
        """Draw the mesh + boundary nodes without solving.

        Saves (or shows) a quick matplotlib figure highlighting clamped vs
        loaded nodes — useful for catching BC mistakes before a costly solve.
        """
        self._check()
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        nodes = np.asarray(self._mesh.nodes)
        fig, ax = plt.subplots(figsize=(5, 4.5))
        cells = np.asarray(self._mesh.cells)

        # Determine fixed nodes from the Dirichlet BCs.
        dim = self._dim
        fixed = set()
        for bc in self._load_case.dirichlet_bcs:
            dofs = np.asarray(bc.dofs)
            node_ids = (dofs // dim).astype(int)
            fixed.update(node_ids.tolist())

        all_nodes = set(range(self._mesh.n_nodes))
        loaded = all_nodes - fixed

        if self._mesh.cell_type in ("tri3", "quad4"):
            ct = self._mesh.cell_type
            if ct == "tri3":
                tris = cells
            else:
                n = cells.shape[0]
                tris = np.empty((2 * n, 3), dtype=cells.dtype)
                tris[0::2] = cells[:, [0, 1, 2]]
                tris[1::2] = cells[:, [0, 2, 3]]
            ax.triplot(nodes[:, 0], nodes[:, 1], tris, color="0.8",
                       lw=0.5, zorder=1)
        ax.scatter(nodes[list(loaded), 0], nodes[list(loaded), 1],
                   s=8, c="tab:blue", label="free / loaded", zorder=2)
        if fixed:
            ax.scatter(nodes[list(fixed), 0], nodes[list(fixed), 1],
                       s=12, c="tab:red", label="clamped", zorder=3)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{self._mesh.cell_type} mesh: "
                     f"{self._mesh.n_nodes} nodes, {self._mesh.n_cells} cells")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(path), dpi=120)
        plt.close(fig)
        return path

    def _check(self):
        if self._mesh is None:
            raise ValueError("call .mesh(...) first")
        if self._material is None:
            raise ValueError("call .material(...) or .material_E_nu(...) first")
        if self._load_case is None:
            raise ValueError("call a load-case setter (.tension/.shear/...)")


__all__ = ["ProblemSetup", "ProblemResult"]
