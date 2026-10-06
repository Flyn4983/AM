"""Rate-dependent crystal plasticity constitutive model (small strain).

Implements a classical small-strain rate-dependent crystal plasticity model
suitable for the JAX / autodiff stack:

Constitutive relations
----------------------
Total strain decomposition::

    ε = ε_e + ε_p

Stress (Hooke's law on the elastic part)::

    σ = λ tr(ε_e) I + 2μ ε_e

Schmid law (resolved shear stress on system α)::

    τ^α = σ : P^α ,   P^α = sym(b^α ⊗ n^α)

Power-law flow rule (rate-dependent)::

    γ̇^α = γ̇0 · (|τ^α| / g^α)^(1/m) · sign(τ^α)

Plastic strain rate (flow along slip systems)::

    ε̇_p = Σ_α γ̇^α · P^α

Voce-type isotropic hardening (self-hardening per system)::

    ḡ^α = h0 · (1 − g^α / g_sat) · |γ̇^α|

State variables
---------------
- ``eps_p`` : ``(dim, dim)`` plastic strain tensor
- ``gamma`` : ``(n_slip,)`` accumulated slip per system
- ``g``     : ``(n_slip,)`` critical resolved shear stress (CRSS) per system

Integration
-----------
The constitutive update uses explicit (forward-Euler) sub-stepping via
``jax.lax.scan``, which makes it fully reverse-mode-AD friendly. The total
strain is held fixed during the sub-steps (strain is imposed by the FEM
equilibrium solve). The algorithmic tangent ``dσ/dε`` is obtained by
``jax.jacrev`` of the stress function, so it is always consistent with the
implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from diffmech.materials.linear_elastic import youngs_poisson_to_lame
from diffmech.methods.cpfe.slip_systems import schmid_tensors


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CPState:
    """Internal state variables for one material point (one quadrature point).

    Attributes
    ----------
    eps_p : (dim, dim) plastic strain tensor
    gamma : (n_slip,) accumulated slip per system
    g : (n_slip,) critical resolved shear stress per system
    """

    eps_p: jnp.ndarray
    gamma: jnp.ndarray
    g: jnp.ndarray

    def tree_flatten(self):
        return (self.eps_p, self.gamma, self.g), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


# ---------------------------------------------------------------------------
# Material
# ---------------------------------------------------------------------------
@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CrystalPlasticity:
    """Rate-dependent small-strain crystal plasticity material.

    Attributes
    ----------
    E, nu : float
        Isotropic elastic constants (Young's modulus, Poisson ratio).
    slip_directions : (n_slip, dim) array
        Unit slip directions ``b^α``.
    slip_normals : (n_slip, dim) array
        Unit slip plane normals ``n^α``.
    gamma_dot0 : float
        Reference slip rate ``γ̇0``.
    m : float
        Rate-sensitivity exponent (flow rule power is ``1/m``).
        ``m → 0`` recovers the rate-independent limit.
    g0 : float
        Initial critical resolved shear stress (same for all systems).
    h0 : float
        Hardening modulus.
    g_sat : float
        Saturation CRSS for Voce hardening. A very large value gives
        approximately linear hardening.
    """

    E: float
    nu: float
    slip_directions: jnp.ndarray
    slip_normals: jnp.ndarray
    gamma_dot0: float
    m: float
    g0: float
    h0: float
    g_sat: float = 1.0e9

    def tree_flatten(self):
        leaves = (
            self.E, self.nu,
            self.slip_directions, self.slip_normals,
            self.gamma_dot0, self.m, self.g0, self.h0, self.g_sat,
        )
        return leaves, None

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        (E, nu, b, n, gdot0, m, g0, h0, gs) = leaves
        return cls(E, nu, b, n, gdot0, m, g0, h0, gs)

    # --- derived properties ------------------------------------------------
    @property
    def lam(self) -> float:
        return youngs_poisson_to_lame(self.E, self.nu)[0]

    @property
    def mu(self) -> float:
        return youngs_poisson_to_lame(self.E, self.nu)[1]

    @property
    def n_slip(self) -> int:
        return int(np.asarray(self.slip_directions).shape[0])

    @property
    def dim(self) -> int:
        return int(np.asarray(self.slip_directions).shape[1])

    @property
    def schmid(self) -> jnp.ndarray:
        """(n_slip, dim, dim) Schmid tensors P^α = sym(b ⊗ n)."""
        return schmid_tensors(self.slip_directions, self.slip_normals)

    def initial_state(self) -> CPState:
        """Zero plastic strain / slip, uniform initial CRSS ``g0``."""
        dim = self.dim
        n = self.n_slip
        return CPState(
            eps_p=jnp.zeros((dim, dim)),
            gamma=jnp.zeros(n),
            g=jnp.full(n, self.g0),
        )


# ---------------------------------------------------------------------------
# Stress and slip-rate computation
# ---------------------------------------------------------------------------
def cp_stress(eps: jnp.ndarray, state: CPState, mat: CrystalPlasticity) -> jnp.ndarray:
    """Cauchy stress ``σ = C : (ε − ε_p)``."""
    dim = eps.shape[0]
    eps_el = eps - state.eps_p
    tr = jnp.trace(eps_el)
    eye = jnp.eye(dim, dtype=eps.dtype)
    return mat.lam * tr * eye + 2.0 * mat.mu * eps_el


def slip_rates(
    sigma: jnp.ndarray, state: CPState, mat: CrystalPlasticity
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Power-law slip rates and resolved shear stresses.

    Returns
    -------
    gamma_dot : (n_slip,) slip rates γ̇^α
    tau : (n_slip,) resolved shear stresses τ^α
    """
    P = mat.schmid  # (n_slip, dim, dim)
    tau = jnp.einsum("ij,sij->s", sigma, P)        # (n_slip,)
    g = jnp.maximum(state.g, 1e-12)
    ratio = jnp.abs(tau) / g
    # Avoid 0^0 when m is small and tau=0 by adding a tiny floor.
    rate = mat.gamma_dot0 * ratio ** (1.0 / mat.m) * jnp.sign(tau)
    return rate, tau


# ---------------------------------------------------------------------------
# Constitutive update (explicit Euler with sub-stepping)
# ---------------------------------------------------------------------------
def cp_update_step(
    eps: jnp.ndarray, state: CPState, mat: CrystalPlasticity, dt: float
) -> CPState:
    """One explicit-Euler sub-step of the CP update.

    The total strain ``eps`` is held fixed; only the internal variables evolve.
    """
    sigma = cp_stress(eps, state, mat)
    gamma_dot, _ = slip_rates(sigma, state, mat)
    dgamma = gamma_dot * dt                              # (n_slip,)
    P = mat.schmid                                       # (n_slip, dim, dim)
    deps_p = jnp.einsum("s,sij->ij", dgamma, P)         # (dim, dim)
    eps_p_new = state.eps_p + deps_p
    gamma_new = state.gamma + dgamma
    # Voce self-hardening: ḡ^α = h0 (1 − g/g_sat) |γ̇^α|
    hdot = mat.h0 * (1.0 - state.g / mat.g_sat) * jnp.abs(gamma_dot)
    g_new = jnp.maximum(state.g + hdot * dt, 0.0)
    return CPState(eps_p=eps_p_new, gamma=gamma_new, g=g_new)


def cp_update(
    eps: jnp.ndarray,
    state: CPState,
    mat: CrystalPlasticity,
    dt: float,
    n_sub: int = 1,
) -> CPState:
    """Constitutive update with sub-stepping via ``jax.lax.scan``.

    The total strain is held fixed across sub-steps. Use ``n_sub > 1`` when
    the slip-rate magnitude times ``dt`` is large (stability / accuracy).
    """
    sub_dt = dt / n_sub

    def body(s, _):
        return cp_update_step(eps, s, mat, sub_dt), None

    state_final, _ = jax.lax.scan(body, state, xs=None, length=n_sub)
    return state_final


# ---------------------------------------------------------------------------
# Algorithmic tangent (dσ/dε) via autodiff
# ---------------------------------------------------------------------------
def cp_stress_and_tangent(
    eps: jnp.ndarray, state: CPState, mat: CrystalPlasticity
):
    """Return (sigma, d_sigma_d_eps).

    The algorithmic tangent ``dσ/dε`` is computed by ``jax.jacrev`` of
    :func:`cp_stress` w.r.t. ``eps``, so it is always consistent with the
    stress implementation. For the *explicit* update the tangent reduces to
    the elastic tangent ``C`` (since ``eps_p`` is treated as fixed when the
    stress is evaluated), but the autodiff route keeps the door open for
    implicit updates.
    """
    sigma = cp_stress(eps, state, mat)
    tangent = jax.jacrev(lambda e: cp_stress(e, state, mat))(eps)  # (d,d,d,d)
    return sigma, tangent


def elastic_tangent_voigt(mat: CrystalPlasticity, dim: int) -> jnp.ndarray:
    """Voigt-form elastic tangent ``C`` (constant for small-strain CP)."""
    lam, mu = mat.lam, mat.mu
    if dim == 2:
        return jnp.array([
            [lam + 2 * mu, lam, 0.0],
            [lam, lam + 2 * mu, 0.0],
            [0.0, 0.0, mu],
        ], dtype=jnp.float64)
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


# ---------------------------------------------------------------------------
# Helpers for batched (per-quadrature-point) state
# ---------------------------------------------------------------------------
def cp_update_batched(
    eps_qp: jnp.ndarray,
    state_qp: CPState,
    mat: CrystalPlasticity,
    dt: float,
    n_sub: int = 1,
) -> CPState:
    """Vectorised CP update over many quadrature points.

    Parameters
    ----------
    eps_qp : (n_qp, dim, dim) strain at each QP
    state_qp : CPState with leading batch dim ``n_qp``
    mat : CrystalPlasticity
    """
    def single(eps, st):
        return cp_update(eps, st, mat, dt, n_sub)
    return jax.vmap(single)(eps_qp, state_qp)
