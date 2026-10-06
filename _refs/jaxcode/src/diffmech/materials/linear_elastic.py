"""Linear elastic isotropic material (small strain).

Stress: sigma = lambda * trace(eps) * I + 2 * mu * eps
Tangent: 4th-order elasticity tensor C_{ijkl} = lambda * delta_ij * delta_kl
                                                + mu * (delta_ik*delta_jl + delta_il*delta_jk)

We expose a voigt-form 6x6 (3D) / 3x3 (2D plane strain) tangent for use in FEM.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


def youngs_poisson_to_lame(E: float, nu: float) -> tuple[float, float]:
    """Convert (Young's modulus, Poisson ratio) to Lamé parameters."""
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lam, mu


def linear_elastic_stress(eps: jnp.ndarray, lam: float, mu: float) -> jnp.ndarray:
    """Small-strain linear elastic stress.

    Parameters
    ----------
    eps : (dim, dim) symmetric small-strain tensor
    lam, mu : Lamé constants

    Returns
    -------
    sigma : (dim, dim) Cauchy stress
    """
    dim = eps.shape[0]
    tr = jnp.trace(eps)
    eye = jnp.eye(dim, dtype=eps.dtype)
    return lam * tr * eye + 2.0 * mu * eps


def linear_elastic_tangent_3d(lam: float, mu: float) -> jnp.ndarray:
    """6x6 Voigt-form elasticity tensor (3D), ordering [xx, yy, zz, yz, xz, xy]."""
    C = jnp.zeros((6, 6), dtype=jnp.result_type(lam, mu))
    # Diagonal normal entries
    diag = jnp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    # C_ii_ii = lam + 2mu, C_ii_jj = lam (i!=j normal), C_shear = mu
    C = C.at[0, 0].set(lam + 2 * mu)
    C = C.at[1, 1].set(lam + 2 * mu)
    C = C.at[2, 2].set(lam + 2 * mu)
    C = C.at[3, 3].set(mu)
    C = C.at[4, 4].set(mu)
    C = C.at[5, 5].set(mu)
    # Off-diagonal normal couplings
    for i in range(3):
        for j in range(3):
            if i != j:
                C = C.at[i, j].set(lam)
    return C


def linear_elastic_tangent(lam: float, mu: float, dim: int = 2) -> jnp.ndarray:
    """Voigt-form elasticity tensor.

    For 2D we use plane strain. Ordering (2D): [xx, yy, xy].
    """
    if dim == 2:
        return jnp.array([
            [lam + 2 * mu, lam, 0.0],
            [lam, lam + 2 * mu, 0.0],
            [0.0, 0.0, mu],
        ])
    if dim == 3:
        return linear_elastic_tangent_3d(lam, mu)
    raise ValueError(f"dim must be 2 or 3, got {dim}")


@dataclass(frozen=True)
class LinearElasticIsotropic:
    """Material parameters for isotropic linear elasticity."""

    E: float
    nu: float

    @property
    def lame(self) -> tuple[float, float]:
        return youngs_poisson_to_lame(self.E, self.nu)

    @property
    def lam(self) -> float:
        return self.lame[0]

    @property
    def mu(self) -> float:
        return self.lame[1]

    def stress(self, eps: jnp.ndarray) -> jnp.ndarray:
        return linear_elastic_stress(eps, self.lam, self.mu)

    def tangent(self, dim: int = 2) -> jnp.ndarray:
        return linear_elastic_tangent(self.lam, self.mu, dim)
