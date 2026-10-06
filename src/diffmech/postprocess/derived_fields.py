"""Derived engineering fields from FEM displacement solutions.

Wraps :func:`diffmech.methods.fem.recover_strain_stress` into a convenient
post-processing layer that produces the scalar/vector fields engineers
actually look at after a structural solve:

- von Mises equivalent stress
- principal stresses and directions
- hydrostatic / deviatoric stress
- equivalent (von Mises) strain

These helpers are differentiable, so they can sit inside a ``jax.grad``
loss (e.g. ``loss = max(von_mises_stress(U))`` for a stress-constrained
inverse design).

Example
-------
::

    U = solve_linear_elastic(mesh, mat, bcs)
    fc = stress_field_collection(mesh, U, mat)   # von_mises, principal, ...
    write_vtu("result.vtu", mesh, fields=fc)
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from diffmech.methods.fem import recover_strain_stress
from diffmech.postprocess.vtk import FieldCollection


def _deviatoric(stress: jnp.ndarray) -> jnp.ndarray:
    """Deviatoric part of a batch of stress tensors (n_cells, dim, dim)."""
    dim = stress.shape[-1]
    tr = jnp.trace(stress, axis1=-2, axis2=-1)        # (n_cells,)
    hydro = tr / dim                                   # (n_cells,)
    eye = jnp.eye(dim, dtype=stress.dtype)
    return stress - hydro[..., None, None] * eye


def von_mises_from_stress(stress: jnp.ndarray) -> jnp.ndarray:
    """Von Mises equivalent stress from a batch of symmetric stress tensors.

    Parameters
    ----------
    stress : (n_cells, dim, dim) array
        Per-cell Cauchy stress tensor (2D or 3D).

    Returns
    -------
    vm : (n_cells,) array of von Mises stress.

    Notes
    -----
    For 2D the plane-stress formula ``sqrt(sxx² - sxx·syy + syy² + 3·sxy²)``
    is applied to the *full* stress (so a uniaxial state ``σ_xx = σ₀`` gives
    ``vm = σ₀``). For 3D the deviatoric invariant ``vm = sqrt(3/2 · s:s_dev)``
    is used, which is the standard continuum-mechanics definition.
    """
    s = jnp.asarray(stress)
    dim = s.shape[-1]
    if dim == 2:
        sxx, syy, sxy = s[:, 0, 0], s[:, 1, 1], s[:, 0, 1]
        return jnp.sqrt(jnp.maximum(
            sxx ** 2 - sxx * syy + syy ** 2 + 3.0 * sxy ** 2, 0.0))
    dev = _deviatoric(s)
    sxx, syy, szz = dev[:, 0, 0], dev[:, 1, 1], dev[:, 2, 2]
    sxy, syz, sxz = dev[:, 0, 1], dev[:, 1, 2], dev[:, 0, 2]
    return jnp.sqrt(jnp.maximum(0.5 * (
        (sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2 +
        6.0 * (sxy ** 2 + syz ** 2 + sxz ** 2)), 0.0))


def principal_stresses(stress: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Eigenvalues (sorted desc) and eigenvectors of a batch of stress tensors.

    Returns
    -------
    values : (n_cells, dim) principal stresses σ₁ ≥ σ₂ ≥ …
    vectors : (n_cells, dim, dim) columns are the principal directions.
    """
    s = jnp.asarray(stress)
    # Symmetrise to kill any numerical asymmetry from the recovery.
    s = 0.5 * (s + jnp.swapaxes(s, -1, -2))
    vals, vecs = jnp.linalg.eigh(s)
    # eigh returns ascending order; flip to descending.
    vals = jnp.flip(vals, axis=-1)
    vecs = jnp.flip(vecs, axis=-1)
    return vals, vecs


def hydrostatic_pressure(stress: jnp.ndarray) -> jnp.ndarray:
    """Hydrostatic pressure ``p = -tr(σ)/dim`` (negative in compression)."""
    s = jnp.asarray(stress)
    dim = s.shape[-1]
    return -jnp.trace(s, axis1=-2, axis2=-1) / dim


def equivalent_strain(strain: jnp.ndarray) -> jnp.ndarray:
    """Von Mises equivalent strain ``ε_eq`` from a batch of strain tensors."""
    e = jnp.asarray(strain)
    dim = e.shape[-1]
    # ε_eq = sqrt(2/3 * ε_ij^dev : ε_ij^dev)
    dev = _deviatoric(e)
    if dim == 2:
        exx, eyy, exy = dev[:, 0, 0], dev[:, 1, 1], dev[:, 0, 1]
        return jnp.sqrt(jnp.maximum(
            2.0 / 3.0 * (exx ** 2 + eyy ** 2 + 2.0 * exy ** 2), 0.0))
    exx, eyy, ezz = dev[:, 0, 0], dev[:, 1, 1], dev[:, 2, 2]
    exy, eyz, exz = dev[:, 0, 1], dev[:, 1, 2], dev[:, 0, 2]
    return jnp.sqrt(jnp.maximum(2.0 / 3.0 * (
        exx ** 2 + eyy ** 2 + ezz ** 2 + 2.0 * (exy ** 2 + eyz ** 2 + exz ** 2)
    ), 0.0))


def stress_field_collection(mesh, U, mat, *, dim: int | None = None,
                            include_displacement: bool = True,
                            ) -> FieldCollection:
    """Build a :class:`FieldCollection` of derived fields for VTU export.

    Computes strain & stress from ``U`` and packs von Mises stress,
    principal stresses, hydrostatic pressure and equivalent strain as cell
    data, plus the nodal displacement vector as point data.

    Parameters
    ----------
    mesh, U, mat : as in :func:`recover_strain_stress`.
    include_displacement : also add the nodal displacement field.
    """
    if dim is None:
        dim = mesh.dim
    strain, stress = recover_strain_stress(mesh, U, mat, dim=dim)
    vm = von_mises_from_stress(stress)
    pvals, _ = principal_stresses(stress)
    hydro = hydrostatic_pressure(stress)
    eq_strain = equivalent_strain(strain)

    fc = FieldCollection()
    fc.add_cell("von_mises_stress", np.asarray(vm))
    fc.add_cell("principal_stress_1", np.asarray(pvals[:, 0]))
    if dim >= 2:
        fc.add_cell("principal_stress_2", np.asarray(pvals[:, 1]))
    if dim == 3:
        fc.add_cell("principal_stress_3", np.asarray(pvals[:, 2]))
    fc.add_cell("hydrostatic_pressure", np.asarray(hydro))
    fc.add_cell("equivalent_strain", np.asarray(eq_strain))

    if include_displacement:
        u_nodes = np.asarray(U).reshape(mesh.n_nodes, dim)
        fc.add_point("displacement", u_nodes)
    return fc


__all__ = [
    "von_mises_from_stress", "principal_stresses",
    "hydrostatic_pressure", "equivalent_strain",
    "stress_field_collection",
]
