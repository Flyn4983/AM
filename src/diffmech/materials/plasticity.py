"""J2 (von Mises) plasticity with linear isotropic hardening, small strain.

Implements the *return-mapping* algorithm:

    Given trial strain increment, compute trial stress and check yield.
    If yielding, return-map to the yield surface using radial return.

State variables:
    - ``ep`` : (dim, dim) equivalent plastic strain (accumulated)
    - ``alpha`` : scalar accumulated equivalent plastic strain

Algorithmic tangent (consistent) is computed via JAX's autodiff of the
return-mapping update, so the tangent is always consistent with the
implementation, even if the algorithm is changed later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.numpy as jnp


class PlasticState(NamedTuple):
    """Internal state variables for J2 plasticity."""

    ep: jnp.ndarray  # (dim, dim) plastic strain tensor
    alpha: jnp.ndarray  # scalar accumulated equivalent plastic strain


def _initial_state(dim: int = 3) -> PlasticState:
    return PlasticState(
        ep=jnp.zeros((dim, dim)),
        alpha=jnp.array(0.0),
    )


def _deviator(s: jnp.ndarray) -> jnp.ndarray:
    """Deviatoric part of a tensor."""
    dim = s.shape[0]
    tr = jnp.trace(s)
    return s - tr / dim * jnp.eye(dim, dtype=s.dtype)


@dataclass(frozen=True)
class J2Plasticity:
    """J2 plasticity with isotropic linear hardening.

    sigma_y(alpha) = sigma_y0 + H * alpha
    """

    E: float
    nu: float
    sigma_y0: float
    H: float  # isotropic hardening modulus

    @property
    def lam(self) -> float:
        return self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

    @property
    def mu(self) -> float:
        return self.E / (2 * (1 + self.nu))

    @property
    def K(self) -> float:
        """Bulk modulus = lam + 2mu/dim."""
        return self.lam + 2 * self.mu / 3.0

    def initial_state(self, dim: int = 3) -> PlasticState:
        return _initial_state(dim)

    def trial_stress(self, eps_total: jnp.ndarray, state: PlasticState) -> jnp.ndarray:
        """Elastic predictor stress = C : (eps_total - eps_p)."""
        eps_el = eps_total - state.ep
        dim = eps_el.shape[0]
        tr = jnp.trace(eps_el)
        return self.lam * tr * jnp.eye(dim, dtype=eps_el.dtype) + 2 * self.mu * eps_el


def j2_plasticity_update(
    eps_total: jnp.ndarray,
    state: PlasticState,
    mat: J2Plasticity,
) -> tuple[jnp.ndarray, PlasticState]:
    """Return-mapping update for J2 plasticity.

    Parameters
    ----------
    eps_total : (dim, dim) total small-strain tensor at end of step
    state : PlasticState (ep, alpha) at start of step
    mat : J2Plasticity material

    Returns
    -------
    sigma : (dim, dim) updated Cauchy stress
    new_state : PlasticState (ep, alpha) at end of step
    """
    dim = eps_total.shape[0]
    eps_p_old, alpha_old = state.ep, state.alpha

    # Elastic predictor
    eps_el_tr = eps_total - eps_p_old
    tr_el = jnp.trace(eps_el_tr)
    sigma_tr = mat.lam * tr_el * jnp.eye(dim, dtype=eps_total.dtype) + 2 * mat.mu * eps_el_tr

    # Deviatoric trial stress
    s_tr = _deviator(sigma_tr)
    # Equivalent (von Mises) trial stress
    # sigma_eq = sqrt(3/2) ||s||
    norm_s_tr = jnp.sqrt(jnp.sum(s_tr * s_tr) + 1e-30)
    sigma_eq_tr = jnp.sqrt(1.5) * norm_s_tr

    # Yield function
    sigma_y = mat.sigma_y0 + mat.H * alpha_old
    f_tr = sigma_eq_tr - sigma_y

    # If f_tr <= 0, elastic step: keep state
    # Otherwise: radial return
    plastic = f_tr > 0
    # Plastic multiplier (with linear isotropic hardening)
    # dgamma = f_tr / (3*mu + H)
    dgamma = jnp.where(plastic, f_tr / (3.0 * mat.mu + mat.H), 0.0)

    # Updated deviatoric stress: s = s_tr - 2*mu*dgamma * n_hat, where
    # n_hat is the unit deviatoric direction satisfying n_hat:n_hat = 3/2.
    # Standard form: n_hat = sqrt(3/2) * s_tr / |s_tr| = (3/2) * s_tr / sigma_eq_tr
    n_hat = jnp.where(
        norm_s_tr > 1e-12,
        jnp.sqrt(1.5) * s_tr / norm_s_tr,
        jnp.zeros_like(s_tr),
    )
    s_new = s_tr - 2.0 * mat.mu * dgamma * n_hat

    # Hydrostatic stress unchanged (plastic is incompressible)
    sigma_new = s_new + (jnp.trace(sigma_tr) / dim) * jnp.eye(dim, dtype=eps_total.dtype)

    # Updated plastic strain and alpha
    dep = dgamma * n_hat  # increment of plastic strain
    eps_p_new = eps_p_old + dep
    alpha_new = alpha_old + dgamma

    new_state = PlasticState(ep=eps_p_new, alpha=alpha_new)

    # Use jnp.where to keep the elastic branch when f_tr <= 0
    sigma_final = jnp.where(plastic, sigma_new, sigma_tr)
    final_state = PlasticState(
        ep=jnp.where(plastic, eps_p_new, eps_p_old),
        alpha=jnp.where(plastic, alpha_new, alpha_old),
    )
    return sigma_final, final_state


def j2_plasticity_stress_and_tangent(
    eps_total: jnp.ndarray, state: PlasticState, mat: J2Plasticity
):
    """Return (sigma, new_state, d_sigma_d_eps).

    The algorithmic tangent is computed by autodiff of the return-mapping update.
    """
    def stress_only(eps):
        s, _ = j2_plasticity_update(eps, state, mat)
        return s

    sigma, new_state = j2_plasticity_update(eps_total, state, mat)
    # Compute the full 4th-order tangent via jax.jacrev (reverse-mode autodiff).
    # We vectorize sigma (dim*dim) and eps_total (dim*dim).
    dim = eps_total.shape[0]
    tangent = jax.jacrev(stress_only)(eps_total)  # (dim,dim,dim,dim)
    return sigma, new_state, tangent
