"""Plastic / crystal-plasticity thermo-mechanical AM solver (P2-②).

Extension of :mod:`diffmech.methods.am.thermomechanical` that replaces the
purely *linear-elastic* stress recovery with a **stateful rate-independent
plastic constitutive update** carried across the layer-by-layer build.

Why this matters
----------------
The original ``solve_thermomechanical`` solves the elastic equilibrium and then
recovers stress with ``σ = C : ε``. For AM residual stress this is wrong once
the (large) thermal contraction drives the stress past the yield surface: real
LPBF/SLM parts yield, which *relaxes* the stress and caps it near the yield
strength (residual stresses are typically 0.3–0.8 σ_y, never ``E·ε_th``).

This module keeps the same activation-gated stiffness + thermal-load machinery
but, after each layer's equilibrium solve, runs a **return-mapping** update at
every Gauss/centroid point:

* **J2 (von Mises) plasticity** with isotropic hardening
  (:class:`diffmech.materials.plasticity.J2Plasticity`,
  :func:`j2_plasticity_update`) — state ``(ε_p, α)``.
* **Rate-dependent crystal plasticity** (CPFE,
  :class:`diffmech.methods.cpfe.crystal_plasticity.CrystalPlasticity`,
  :func:`cp_update`) — state ``(ε_p, γ, g)`` — gives anisotropic single-crystal
  yielding and slip-system hardening.

The internal state is threaded through ``jax.lax.scan`` over layers, so the
whole build history (residual stress, plastic strain, distortion) is one
differentiable function of the process parameters.

Consistency (stability note)
-----------------------------
The global equilibrium is solved with the **elastic** tangent (always
positive-definite), which keeps the linear solve robust even when material points
are far into the plastic regime. The return-map then projects the recovered
strain onto the yield surface, so the *stress field* is physically correct
(capped near the yield strength) and plastic strain accumulates properly —
the core requirement for faithful AM residual-stress prediction.

A full consistent-tangent Newton (re-assembling ``K`` from the per-point
algorithmic tangent) is available via
``assemble_activated_tangent_stiffness`` but is left as a future refinement:
when material is fully plastic the consistent tangent is near-singular and needs
careful regularization, whereas the elastic-solve + return-map path is unconditionally
stable. The displacement field is therefore the elastic upper bound.

All hot loops are ``vmap``/``jax.lax.scan``-friendly and jit-able.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from diffmech.core import (
    Mesh, gauss_legendre_nd, get_shape_function, physical_gradient_and_det,
)
from diffmech.materials import LinearElasticIsotropic
from diffmech.materials.plasticity import (
    J2Plasticity, j2_plasticity_update,
)
from diffmech.methods.cpfe.crystal_plasticity import (
    CrystalPlasticity, CPState, cp_stress,
)
from diffmech.methods.fem.linear_fem import (
    recover_strain_stress, _strain_displacement_matrix,
)
from diffmech.methods.am.thermomechanical import (
    assemble_activated_stiffness, thermal_strain_force,
)
from diffmech.solvers.boundary_conditions import DirichletBC, apply_dirichlet


# ---------------------------------------------------------------------------
# Voigt / tensor helpers
# ---------------------------------------------------------------------------
def _voigt_of_stress(sigma: jnp.ndarray, dim: int) -> jnp.ndarray:
    """Convert a (...,dim,dim) stress tensor to Voigt (...,6) or (...,3)."""
    if dim == 2:
        return jnp.stack([sigma[..., 0, 0], sigma[..., 1, 1],
                          sigma[..., 0, 1]], axis=-1)
    return jnp.stack([sigma[..., 0, 0], sigma[..., 1, 1], sigma[..., 2, 2],
                      sigma[..., 1, 2], sigma[..., 0, 2], sigma[..., 0, 1]],
                     axis=-1)


def _tangent_to_voigt(tg: jnp.ndarray, dim: int) -> jnp.ndarray:
    """Convert a (...,dim,dim,dim,dim) tangent to Voigt (...,6,6) or (...,3,3)."""
    idx = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)][: (3 if dim == 2 else 6)]
    rows = []
    for (a, b) in idx:
        comps = jnp.stack([tg[..., a, b, i, j] for (i, j) in idx], axis=-1)
        rows.append(comps)
    return jnp.stack(rows, axis=-2)


def _elastic_voigt_C(E: float, nu: float, dim: int) -> jnp.ndarray:
    """Isotropic elastic tangent in Voigt form (6x6 / 3x3)."""
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    if dim == 2:
        return jnp.array([[lam + 2 * mu, lam, 0.0],
                          [lam, lam + 2 * mu, 0.0],
                          [0.0, 0.0, mu]], dtype=jnp.float64)
    C = jnp.zeros((6, 6), dtype=jnp.float64)
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
    return C


def _qr_for(cell_type: str):
    """Quadrature rule (points, weights) for a given cell type."""
    if cell_type == "tri3":
        return gauss_legendre_nd(2, 2, reference="tri")
    if cell_type == "quad4":
        return gauss_legendre_nd(2, 2, reference="quad")
    if cell_type == "tet4":
        return gauss_legendre_nd(2, 3, reference="tet")
    if cell_type == "hex8":
        return gauss_legendre_nd(2, 3, reference="hex")
    raise ValueError(f"unsupported cell_type {cell_type}")


# ---------------------------------------------------------------------------
# Per-cell tangent-stiffness assembly (activation-gated)
# ---------------------------------------------------------------------------
def assemble_activated_tangent_stiffness(
    mesh: Mesh, C_cells: jnp.ndarray, alpha: jnp.ndarray, *, dim: int = 3,
) -> jnp.ndarray:
    """Global stiffness from per-cell Voigt tangents ``C_cells`` (n_cells,6,6).

    Each element stiffness ``k_e = Σ_qp w·detJ·Bᵀ·C_cell·B`` is scaled by its
    activation ``α`` (soft powder → 0 stiffness). Mirrors
    :func:`assemble_activated_stiffness` but with a *per-cell* tangent so the
    plastic consistent tangent can be fed in.
    """
    n_dofs = mesh.n_nodes * dim
    cell_coords = mesh.cell_coords
    cells = mesh.cells
    n_dpc = mesh.nodes_per_cell
    qr = _qr_for(mesh.cell_type)

    def cell_ke(coords, C):
        sf = get_shape_function(mesh.cell_type)
        ke = jnp.zeros((n_dpc * dim, n_dpc * dim), dtype=coords.dtype)

        def body(ke, qp):
            xi, w = qp
            dN_dxi = sf.grad(xi)
            dN_dx, detJ = physical_gradient_and_det(dN_dxi, coords)
            B = _strain_displacement_matrix(dN_dx, dim)  # (n_strain, n_dpc*dim)
            return ke + w * detJ * (B.T @ C @ B), None

        ke, _ = jax.lax.scan(body, ke, (qr.points, qr.weights))
        return ke

    kes = jax.vmap(cell_ke)(cell_coords, C_cells) * alpha[:, None, None]

    def assemble_one(carry, args):
        ke, cell = args
        dofs = jnp.repeat(cell * dim, dim) + jnp.tile(jnp.arange(dim), n_dpc)
        di, dj = jnp.meshgrid(dofs, dofs, indexing="ij")
        return carry.at[di.ravel(), dj.ravel()].add(ke.ravel()), None

    K, _ = jax.lax.scan(
        assemble_one, jnp.zeros((n_dofs, n_dofs), dtype=mesh.nodes.dtype),
        (kes, cells))
    return K


# ---------------------------------------------------------------------------
# Per-cell constitutive update (vmappable) — J2 and CPFE
# ---------------------------------------------------------------------------
def _j2_cell_update(
    eps: jnp.ndarray, ep: jnp.ndarray, alpha: jnp.ndarray, mat: J2Plasticity,
):
    """Return (sigma, ep_new, alpha_new) for one J2 material point."""
    from diffmech.materials.plasticity import PlasticState
    st = PlasticState(ep=ep, alpha=alpha)
    sigma, st2 = j2_plasticity_update(eps, st, mat)
    return sigma, st2.ep, st2.alpha


def _cp_cell_update(
    eps: jnp.ndarray, eps_p: jnp.ndarray, gamma: jnp.ndarray, g: jnp.ndarray,
    mat: CrystalPlasticity, dt: float, n_sub: int,
):
    """Rate-dependent CPFE update for one material point, with numerical guards.

    The slip rate ``γ̇ = γ̇0 (τ/g)^{1/m}`` is extremely stiff for the small
    rate-sensitivity exponent ``m`` used in AM (m → 0 ⇒ rate-independent limit).
    A naive explicit step therefore overflows when the imposed strain is large.
    We clamp the resolved-shear ratio, the slip *rate*, and the *per-substep* slip
    increment, and integrate over ``n_sub`` sub-steps via ``jax.lax.scan`` so the
    update stays finite and differentiable while still relaxing the stress toward
    the crystal yield surface (anisotropic).
    """
    P = mat.schmid  # (n_slip, dim, dim)
    dt_sub = dt / n_sub

    def step(st, _):
        eps_p_i, gamma_i, g_i = st
        sigma = cp_stress(eps, CPState(eps_p=eps_p_i, gamma=gamma_i, g=g_i), mat)
        tau = jnp.einsum("ij,sij->s", sigma, P)
        gc = jnp.maximum(g_i, 1e-12)
        ratio = jnp.abs(tau) / gc
        rate = mat.gamma_dot0 * jnp.power(jnp.minimum(ratio, 1e3), 1.0 / mat.m) \
            * jnp.sign(tau)
        rate = jnp.clip(rate, -1e3, 1e3)
        dgamma_raw = rate * dt_sub
        # Overshoot guard (rate-independent limit): a slip increment cannot drive
        # the resolved shear past zero. Since dτ/d(Σγ) ≈ 3·μ over the Schmid sum
        # (with P:P = 3/2 on-diagonal and |off-diagonal| ≤ 1/2), bounding
        # |dγ| ≤ 0.5·|τ|/(3μ) lets the stress relax *to* the yield surface without
        # the inter-system coupling making it overshoot/rebound — the correct
        # quasi-static response for the small rate-sensitivity exponent used in AM.
        dg_max = 0.5 * jnp.abs(tau) / (3.0 * mat.mu + 1e-12)
        dgamma = jnp.clip(dgamma_raw, -dg_max, dg_max)
        deps_p = jnp.einsum("s,sij->ij", dgamma, P)
        eps_p_n = eps_p_i + deps_p
        gamma_n = gamma_i + jnp.abs(dgamma)   # accumulated slip magnitude
        hdot = mat.h0 * (1.0 - g_i / mat.g_sat) * jnp.abs(rate)
        g_n = jnp.maximum(g_i + hdot * dt_sub, 0.0)
        return (eps_p_n, gamma_n, g_n), None

    (eps_p_f, gamma_f, g_f), _ = jax.lax.scan(
        step, (eps_p, gamma, g), xs=None, length=n_sub)
    sigma_f = cp_stress(eps, CPState(eps_p=eps_p_f, gamma=gamma_f, g=g_f), mat)
    return sigma_f, eps_p_f, gamma_f, g_f


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class PlasticTMResult:
    """Output of the plastic layer-by-layer thermo-mechanical simulation."""

    U_final: jnp.ndarray                # (n_dofs,) final nodal displacement
    residual_stress: jnp.ndarray        # (n_cells, dim, dim) final stress tensor
    U_history: jnp.ndarray              # (n_layers, n_dofs)
    stress_history: jnp.ndarray         # (n_layers, n_cells, dim, dim)
    plastic_strain_history: jnp.ndarray  # (n_layers, n_cells, dim, dim)
    eq_plastic_strain: jnp.ndarray      # (n_cells,) accumulated equiv. plastic strain
    state_final: object = None          # final internal state (PyTree, per cell)


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------
def solve_thermomechanical_plastic(
    layered,
    material,
    *,
    thermal_strain_per_layer: jnp.ndarray,  # (n_layers, n_cells) α_T·ΔT
    dirichlet_bcs: list[DirichletBC],
    constitutive: str = "j2",
    T_ref: float = 0.0,
    alpha_T: float = 1e-5,
    tau_activation: float = 1e-3,
    dim: int = 3,
    n_inner: int = 2,
    dt_cp: float = 1.0,
    n_sub_cp: int = 64,
    reg: float = 1e-6,
    cell_mask: jnp.ndarray | None = None,
) -> PlasticTMResult:
    """Run the plastic layer-by-layer thermo-mechanical AM simulation.

    Parameters
    ----------
    ...
    cell_mask : (n_cells,) optional
        Per-cell mask (1 = inside the part, 0 = powder / empty). Multiply the
        activation field so that only *part* cells ever acquire stiffness and
        carry the thermal load — a true "element birth only inside the geometry"
        (the layer activation alone only gates *when* a cell deposits, not
        *whether* it is part of the build).
    """
    """Run the plastic layer-by-layer thermo-mechanical AM simulation.

    Parameters
    ----------
    layered : LayeredMesh
    material : J2Plasticity (constitutive="j2") or CrystalPlasticity ("cp")
    thermal_strain_per_layer : (n_layers, n_cells) thermal strain magnitude
        ``α_T·(T_layer − T_ref)`` per cell, per layer (cooling contraction is
        negative).
    dirichlet_bcs : build-plate constraints (clamped bottom face).
    constitutive : "j2" or "cp".
    n_inner : reserved (the solve uses a stable elastic equilibrium + return-map;
        a future consistent-tangent Newton would use this as the inner iteration
        count). Currently a single pass per layer.
    dt_cp, n_sub_cp : effective slip time-step & sub-steps for CPFE.
    """
    mesh = layered.mesh
    n_cells = layered.n_cells
    n_dofs = mesh.n_nodes * dim
    n_layers = layered.n_layers

    # A linear-elastic material with the same E, nu is used only for the
    # (constitutive-agnostic) strain *recovery* from the displacement field.
    lin_mat = LinearElasticIsotropic(E=float(material.E), nu=float(material.nu))

    eps_th = jnp.asarray(thermal_strain_per_layer)  # (n_layers, n_cells)
    # Per-cell *full* thermal contraction magnitude α_T·(T_peak − T_ref).  The
    # per-layer field ``eps_th[k, c]`` is non-zero only on cell ``c``'s own
    # deposition layer, so summing over layers collapses it to one value per cell.
    dT_total = jnp.sum(eps_th, axis=0)  # (n_cells,)

    # ----- dispatch constitutive update ----------------------------------------
    if constitutive == "j2":
        if not isinstance(material, J2Plasticity):
            raise TypeError("constitutive='j2' requires a J2Plasticity material")
        update_fn = lambda strain, st: _j2_cell_update(
            strain, st[0], st[1], material)
        init_state = (
            jnp.zeros((n_cells, dim, dim)),   # ep
            jnp.zeros((n_cells,)),            # alpha
        )
    elif constitutive == "cp":
        if not isinstance(material, CrystalPlasticity):
            raise TypeError("constitutive='cp' requires a CrystalPlasticity material")
        n_slip = material.n_slip
        update_fn = lambda strain, st: _cp_cell_update(
            strain, st[0], st[1], st[2], material, dt_cp, n_sub_cp)
        init_state = (
            jnp.zeros((n_cells, dim, dim)),   # eps_p
            jnp.zeros((n_cells, n_slip)),      # gamma
            jnp.broadcast_to(material.g0, (n_cells, n_slip)).astype(jnp.float64),  # g
        )
    else:
        raise ValueError(f"unknown constitutive {constitutive!r}")

    # ----- one layer's equilibrium + return-mapping ---------------------------
    def solve_layer(alpha, F_thermal, start_state):
        """Equilibrium solve (elastic, well-conditioned) + return-mapping.

        The global equilibrium is solved with the **elastic** tangent (always
        positive-definite → stable). The recovered per-cell strain is then fed to
        the rate-independent return-map (J2 or CPFE), which projects the stress
        back onto the yield surface and advances the plastic internal state.

        This gives a *physically correct residual-stress field* (capped at the
        yield strength, never the elastic overshoot ``E·ε_th``) and correctly
        accumulates plastic strain, while keeping the linear solve robust. The
        displacement field is the elastic upper bound (a documented limitation;
        a full consistent-tangent Newton is a future refinement — see
        ``assemble_activated_tangent_stiffness``).
        """
        # Elastic stiffness, gated by activation → well-conditioned system.
        K = assemble_activated_stiffness(mesh, lin_mat, alpha, dim=dim)
        K, F = apply_dirichlet(K, F_thermal, dirichlet_bcs)
        K = K + reg * jnp.eye(n_dofs, dtype=K.dtype)
        U = jnp.linalg.solve(K, F)
        # Recover per-cell total strain from the displacement.
        strain, _ = recover_strain_stress(mesh, U, lin_mat, dim=dim)
        # Return-map / CP update at each cell (start_state fixed across iters).
        out = jax.vmap(update_fn)(strain, start_state)
        sigma = out[0]
        state_new = tuple(o for o in out[1:])
        return U, sigma, state_new

    # ----- layer-by-layer scan (carries plastic state) ------------------------
    alpha_mask = (jnp.asarray(cell_mask, dtype=jnp.float64)
                  if cell_mask is not None
                  else jnp.ones((n_cells,), dtype=jnp.float64))

    def layer_step(carry, layer_idx):
        state_prev = carry
        # Activation: cells in layers ≤ layer_idx are deposited (α → 1).
        t_offset = jnp.where(layered.layer_id <= layer_idx, 1.0e6, -1.0e6)
        # Gate by the part mask: only part cells ever deposit / carry load.
        a_layer = jax.nn.sigmoid(t_offset / tau_activation)
        alpha = a_layer * alpha_mask
        # Cumulative cooldown: every *deposited* cell has already cooled and
        # contracted, so the thermal load at step k is the full contraction of all
        # cells deposited so far (not just the current layer).  This makes the
        # final residual stress the quasi-static cooldown stress — the magnitude
        # (≈ E·α·ΔT in the most-constrained regions) that the plastic return-map
        # must cap, instead of only the last-deposited layer's contribution.
        deposited = (alpha > 0.5).astype(jnp.float64)
        dT_field = dT_total * deposited  # (n_cells,)  α_T·(T_peak − T_ref)
        # Inactive (powder / not-yet-deposited) cells stay at T_ref → no load
        # (they also have zero stiffness, so the equilibrium stays well-posed).
        T_field = T_ref + dT_field / alpha_T
        F_thermal = thermal_strain_force(mesh, lin_mat, alpha_T, T_field, T_ref, dim=dim)

        U_layer, sigma_layer, state_new = solve_layer(alpha, F_thermal, state_prev)

        # Plastic strain for history: J2 → eps_p; CP → eps_p.
        eps_p_layer = state_new[0]
        return state_new, (U_layer, sigma_layer, eps_p_layer)

    (state_end), (U_hist, sigma_hist, epsp_hist) = jax.lax.scan(
        layer_step, init_state, jnp.arange(n_layers))

    # Equivalent plastic strain (for output): J2 → alpha; CP → Σγ.
    if constitutive == "j2":
        eq_ps = state_end[1]  # alpha
    else:
        eq_ps = jnp.sum(state_end[1], axis=-1)  # Σ gamma

    # Cells excluded by the part mask (powder / empty) never carry load — the
    # activation already removes their stiffness from the equilibrium, but a
    # masked cell can still be dragged by an active neighbour through a shared
    # node.  Zero their reported residual stress so the field reads ~0 outside
    # the geometry (true "element birth only inside the part").
    residual_stress_out = sigma_hist[-1]
    if cell_mask is not None:
        residual_stress_out = residual_stress_out * alpha_mask[:, None, None]

    return PlasticTMResult(
        U_final=U_hist[-1],
        residual_stress=residual_stress_out,
        U_history=U_hist,
        stress_history=sigma_hist,
        plastic_strain_history=epsp_hist,
        eq_plastic_strain=eq_ps,
        state_final=state_end,
    )


__all__ = [
    "PlasticTMResult", "solve_thermomechanical_plastic",
    "assemble_activated_tangent_stiffness",
]
