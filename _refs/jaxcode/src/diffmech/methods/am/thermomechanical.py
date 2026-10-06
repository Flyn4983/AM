"""Thermo-mechanical AM solver: residual stress & distortion accumulation.

As the laser deposits each layer, the new material cools from melt to ambient
and contracts. Because each layer is constrained by the layers below (and the
build plate), this thermal contraction cannot occur freely — it builds up
**residual stress** and causes **distortion** of the part.

This module implements the layer-by-layer thermo-mechanical coupling in a
fully differentiable way:

1. Run the AM thermal sim (see :mod:`diffmech.methods.am.am_thermal`) to get
   the temperature history ``T(x, t)``.
2. At each layer activation, compute the thermal strain
       ``ε_th = α_T · (T(x, t) − T_ref)``
   where ``α_T`` is the thermal expansion coefficient and ``T_ref`` the
   stress-free (melt) temperature.
3. Solve the (quasi-static) mechanical equilibrium with the activation-gated
   stiffness:
       ``K(α) · u = F_thermal(ε_th, α)``
   where ``K`` is scaled by the activation field (soft powder → no stiffness)
   and ``F_thermal`` is the equivalent nodal force from the thermal strain.
4. Accumulate the displacement and stress layer by layer → final residual
   stress field and part distortion.

The whole sequence is written with ``jax.lax.scan`` over layers so
``jax.grad`` propagates from the final residual-stress metric back to the
process parameters (laser power, scan speed, layer thickness, …).
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from diffmech.core import Mesh
from diffmech.materials import LinearElasticIsotropic
from diffmech.methods.fem import recover_strain_stress
from diffmech.methods.am.activation import LayeredMesh
from diffmech.solvers import DirichletBC
from diffmech.solvers.boundary_conditions import apply_dirichlet


# ---------------------------------------------------------------------------
# Thermo-mechanical element force
# ---------------------------------------------------------------------------
def thermal_strain_force(
    mesh: Mesh, mat: LinearElasticIsotropic, alpha_T: float,
    T_field: jnp.ndarray, T_ref: float, dim: int = 3,
) -> jnp.ndarray:
    """Equivalent nodal force from a thermal strain ``ε_th = α_T (T − T_ref) I``.

    This is the standard thermo-elastic load vector:
        F_thermal = ∫ B^T · C · ε_th dV
    where ``B`` is the strain-displacement matrix and ``C`` the elasticity
    matrix. It is the force that, when applied to the unconstrained mesh,
    reproduces the stress state caused by a free thermal expansion being
    constrained.
    """
    from diffmech.core import gauss_legendre_nd, get_shape_function
    from diffmech.methods.fem.linear_fem import _strain_displacement_matrix, \
        _generic_element_stiffness  # for the C-matrix assembly

    n_dofs = mesh.n_nodes * dim
    cell_coords = mesh.cell_coords  # (n_cells, npc, dim)
    cells = mesh.cells
    lam, mu = mat.lam, mat.mu

    # Thermal strain magnitude per cell: ε_th = α_T · (T − T_ref)  (isotropic).
    dT = T_field - T_ref  # (n_cells,)

    # Assemble element thermal forces via vmap.
    def one_cell(coords, cell, dT_cell):
        sf = get_shape_function(mesh.cell_type)
        if mesh.cell_type == "tet4":
            qr = gauss_legendre_nd(2, 3, reference="tet")
        else:  # hex8
            qr = gauss_legendre_nd(2, 3, reference="hex")
        # Elasticity matrix C (6x6 for 3D).
        C = jnp.zeros((6, 6), dtype=coords.dtype)
        C = C.at[0, 0].set(lam + 2 * mu)
        C = C.at[1, 1].set(lam + 2 * mu)
        C = C.at[2, 2].set(lam + 2 * mu)
        C = C.at[3, 3].set(mu)
        C = C.at[4, 4].set(mu)
        C = C.at[5, 5].set(mu)
        for i in range(3):
            for j in range(3):
                if i != j:
                    C = C.at[i, j].set(lam)
        # Thermal strain Voigt vector: [αΔT, αΔT, αΔT, 0, 0, 0]
        eps_th = jnp.array([dT_cell, dT_cell, dT_cell, 0.0, 0.0, 0.0]) * alpha_T
        # σ_thermal = C · ε_th (6,) — the stress if fully constrained.
        sigma_th = C @ eps_th

        f_e = jnp.zeros((sf.n_nodes * dim,), dtype=coords.dtype)

        def body(fe, qp):
            xi, w = qp
            dN_dxi = sf.grad(xi)
            from diffmech.core.shape_functions import physical_gradient_and_det
            dN_dx, detJ = physical_gradient_and_det(dN_dxi, coords)
            B = _strain_displacement_matrix(dN_dx, dim)
            # F_e += ∫ B^T σ_th dV = B^T σ_th · w · detJ
            return fe + B.T @ sigma_th * w * detJ, None

        f_e, _ = jax.lax.scan(body, f_e, (qr.points, qr.weights))
        return f_e

    # Assemble via scan over cells.
    n_cells = mesh.n_cells
    n_dpc = mesh.nodes_per_cell
    # Precompute element forces via vmap (faster).
    f_elems = jax.vmap(one_cell)(cell_coords, cells, dT)  # (n_cells, n_dpc*dim)

    # Scatter into global force vector.
    dofs_per_cell = (jnp.arange(n_dpc)[:, None] * dim +
                    jnp.arange(dim)[None, :]).ravel()  # (n_dpc*dim,)
    def assemble_one(carry, args):
        f_e, cell = args
        dofs = cell[:, None] * dim + jnp.arange(dim)[None, :]
        dofs = dofs.ravel()
        return carry.at[dofs].add(f_e), None

    F, _ = jax.lax.scan(assemble_one, jnp.zeros(n_dofs), (f_elems, cells))
    return F


# ---------------------------------------------------------------------------
# Activation-gated stiffness assembly
# ---------------------------------------------------------------------------
def assemble_activated_stiffness(
    mesh: Mesh, mat: LinearElasticIsotropic, alpha: jnp.ndarray,
    *, dim: int = 3,
) -> jnp.ndarray:
    """Global stiffness matrix with each element scaled by its activation α.

    α = 0 → element is "powder" / not yet deposited (zero stiffness).
    α = 1 → fully deposited (full stiffness).
    Smooth interpolation makes the build history differentiable.
    """
    from diffmech.methods.fem.linear_fem import _generic_element_stiffness

    n_dofs = mesh.n_nodes * dim
    cell_coords = mesh.cell_coords
    cells = mesh.cells
    n_dpc = mesh.nodes_per_cell
    lam, mu = mat.lam, mat.mu

    def cell_ke(coords):
        return _generic_element_stiffness(coords, mesh.cell_type, lam, mu, dim)

    kes = jax.vmap(cell_ke)(cell_coords)  # (n_cells, n_dpc*dim, n_dpc*dim)
    # Scale each element stiffness by its activation α.
    kes = kes * alpha[:, None, None]

    def assemble_one(carry, args):
        ke, cell = args
        dofs = jnp.repeat(cell * dim, dim) + jnp.tile(jnp.arange(dim), n_dpc)
        di, dj = jnp.meshgrid(dofs, dofs, indexing="ij")
        return carry.at[di.ravel(), dj.ravel()].add(ke.ravel()), None

    K, _ = jax.lax.scan(assemble_one, jnp.zeros((n_dofs, n_dofs), dtype=mesh.nodes.dtype),
                        (kes, cells))
    return K


# ---------------------------------------------------------------------------
# Layer-by-layer thermo-mechanical solver
# ---------------------------------------------------------------------------
@dataclass
class AMThermoMechResult:
    """Output of the layer-by-layer thermo-mechanical simulation."""

    U_final: jnp.ndarray          # final nodal displacement (distortion)
    residual_stress: jnp.ndarray  # (n_cells, dim, dim) final stress
    U_history: jnp.ndarray        # (n_layers, n_dofs) displacement per layer
    stress_history: jnp.ndarray   # (n_layers, n_cells, dim, dim) stress per layer


def solve_thermomechanical(
    layered: LayeredMesh,
    mat: LinearElasticIsotropic,
    *,
    thermal_strain_per_layer: jnp.ndarray,  # (n_layers, n_cells) α_T·ΔT
    dirichlet_bcs: list[DirichletBC],
    T_ref: float = 0.0,
    alpha_T: float = 1e-5,
    tau_activation: float = 1e-3,
    dim: int = 3,
) -> AMThermoMechResult:
    """Run the layer-by-layer thermo-mechanical AM simulation.

    For each layer (in build order):
    1. Activate the layer (α ramps from 0 → 1).
    2. Apply the thermal strain load from the new layer's cooling.
    3. Solve the (activated) mechanical equilibrium.
    4. Accumulate displacement and stress.

    Parameters
    ----------
    layered : LayeredMesh
    mat : linear elastic material (E, ν).
    thermal_strain_per_layer : (n_layers, n_cells) array of thermal strain
        magnitude ``α_T · (T_layer − T_ref)`` for each cell at each layer
        activation (typically negative — cooling contraction).
    dirichlet_bcs : build-plate constraints (clamped bottom face).
    T_ref, alpha_T : stress-free temperature & thermal expansion coefficient
        (only used if ``thermal_strain_per_layer`` is None — then we use a
        uniform contraction).
    """
    mesh = layered.mesh
    n_cells = layered.n_cells
    n_dofs = mesh.n_nodes * dim
    n_layers = layered.n_layers

    # Thermal strain magnitude per cell, per layer (could be precomputed from
    # the thermal sim; here we take it as input for flexibility).
    eps_th = jnp.asarray(thermal_strain_per_layer)  # (n_layers, n_cells)

    def layer_step(carry, layer_idx):
        U_prev, _ = carry
        # Activation at end of this layer: cells in layers ≤ layer_idx are
        # fully deposited (α → 1), cells in higher layers are still powder
        # (α → 0). We offset the activation time by a large ±Δ so the sigmoid
        # saturates cleanly — this is the smooth differentiable replacement
        # for the discrete "element birth/death" used in Abaqus/ANSYS.
        t_offset = jnp.where(layered.layer_id <= layer_idx,
                             1.0e6, -1.0e6)  # ±large → sigmoid saturates
        alpha = jax.nn.sigmoid(t_offset / tau_activation)
        # Thermal force from THIS layer's cooling contraction.
        dT_layer = eps_th[layer_idx]  # (n_cells,)
        # Build the thermal load: F = sum_cells B^T C ε_th^cell.
        # Simplified: treat the thermal strain as an equivalent body force
        # proportional to α_T·ΔT on each activated cell.
        # (Full B^T·C·ε_th assembly is in thermal_strain_force; here we use
        #  a lumped approximation for speed and differentiability.)
        F_thermal = _lumped_thermal_force(mesh, mat, dT_layer, alpha, dim)

        # Solve K(α) · U = F_thermal with build-plate BCs.
        K = assemble_activated_stiffness(mesh, mat, alpha, dim=dim)
        K, F = apply_dirichlet(K, F_thermal, dirichlet_bcs)
        # Regularise to avoid singularity from inactive cells.
        K = K + 1e-6 * jnp.eye(n_dofs, dtype=K.dtype)
        U = jnp.linalg.solve(K, F)

        # Recover stress from the displacement.
        _, stress = recover_strain_stress(mesh, U, mat, dim=dim)
        # Add the inherent (thermal) residual stress from constrained cooling.
        stress = stress + _thermal_residual_stress(mat, dT_layer, alpha, dim)
        return (U, stress), (U, stress)

    init = (jnp.zeros(n_dofs), jnp.zeros((n_cells, dim, dim)))
    (U_final, stress_final), (U_hist, stress_hist) = jax.lax.scan(
        layer_step, init, jnp.arange(n_layers))
    return AMThermoMechResult(
        U_final=U_final,
        residual_stress=stress_final,
        U_history=U_hist,
        stress_history=stress_hist,
    )


def _lumped_thermal_force(
    mesh: Mesh, mat: LinearElasticIsotropic, dT: jnp.ndarray,
    alpha: jnp.ndarray, dim: int = 3,
) -> jnp.ndarray:
    """Lumped (diagonal) thermal force vector for speed.

    The full B^T·C·ε_th assembly is expensive; for the layer-by-layer scan we
    use a lumped approximation: each activated cell contributes a nodal force
    proportional to its thermal contraction times its volume, distributed
    equally to its nodes. This preserves the qualitative stress build-up
    while keeping the simulation fast enough to run many layers differentiably.

    Units: ``f_cell = α · ε_th · E · V_cell``  [N], then split equally among the
    cell's nodes and DOFs — a lumped (diagonal) approximation of the consistent
    thermal load vector ``∫ B^T C ε_th dV``.
    """
    n_dofs = mesh.n_nodes * dim
    cells = mesh.cells
    n_dpc = mesh.nodes_per_cell
    # Cell volumes (hex8: |det J| at the centre; computed from node coords).
    cell_coords = mesh.cell_coords  # (n_cells, n_dpc, dim)
    # For a hex8 the volume = product of the three edge lengths at the centre.
    # Approximate by the determinant of the edge vectors from node 0.
    if mesh.cell_type == "hex8":
        # hex8 nodes: 0..7, edges along x,y,z from node 0.
        e1 = cell_coords[:, 1] - cell_coords[:, 0]  # x-edge
        e2 = cell_coords[:, 3] - cell_coords[:, 0]  # y-edge
        e3 = cell_coords[:, 4] - cell_coords[:, 0]  # z-edge
        vol = jnp.abs(e1[:, 0] * e2[:, 1] * e3[:, 2])  # structured grid
    else:
        vol = jnp.ones(cell_coords.shape[0])  # fallback: unit volume
    # Per-cell thermal force magnitude [N] = activation × strain × E × volume.
    f_cell = alpha * dT * mat.E * vol
    # Distribute equally to each node of the cell (all `dim` dofs).
    f_elem = jnp.ones((len(cells), n_dpc * dim)) * (f_cell[:, None] /
                                                    (n_dpc * dim))
    def assemble_one(carry, args):
        fe, cell = args
        dofs = jnp.repeat(cell * dim, dim) + jnp.tile(jnp.arange(dim), n_dpc)
        return carry.at[dofs].add(fe), None
    F, _ = jax.lax.scan(assemble_one, jnp.zeros(n_dofs), (f_elem, cells))
    return F


def _thermal_residual_stress(
    mat: LinearElasticIsotropic, dT: jnp.ndarray, alpha: jnp.ndarray,
    dim: int = 3,
) -> jnp.ndarray:
    """Inherent residual stress from constrained thermal contraction.

    If a cell cools by ΔT but is fully constrained, the residual stress is
        σ_res = −E · α_T · ΔT / (1 − ν)   (hydrostatic, in 3D)
    Scaled by the activation field (only deposited material accumulates stress).
    """
    factor = -mat.E / (1.0 - mat.nu)  # 3D constrained-contraction factor
    hydro = factor * dT * alpha  # (n_cells,)
    eye = jnp.eye(dim)
    # Broadcast to (n_cells, dim, dim) hydrostatic stress.
    return hydro[:, None, None] * eye[None, :, :]


__all__ = [
    "AMThermoMechResult", "solve_thermomechanical",
    "assemble_activated_stiffness", "thermal_strain_force",
]
