"""Compressible Neo-Hookean hyperelastic material (finite strain).

Strain energy density (per unit reference volume):

    W(F) = mu/2 * (tr(B) - dim) - mu * ln(J) + lam/2 * (ln(J))^2

where B = F F^T, J = det(F). This form gives the correct limit as J -> 0
through the logarithmic term, and reduces to linear elasticity for small strains.

First Piola-Kirchhoff stress:

    P = mu * (F - F^{-T}) + lam * ln(J) * F^{-T}

Cauchy stress:

    sigma = (1/J) * P F^T
          = (mu/J) * (B - I) + (lam/J) * ln(J) * I
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


def neo_hookean_piola_kirchhoff(F: jnp.ndarray, lam: float, mu: float) -> jnp.ndarray:
    """First Piola-Kirchhoff stress for compressible Neo-Hookean."""
    dim = F.shape[0]
    J = jnp.linalg.det(F)
    # Add tiny regularization to keep F^{-T} finite when J -> 0 (won't normally
    # be needed in well-posed problems; mainly guards against AD blow-ups in
    # gradient descent trajectories).
    J_safe = jnp.where(jnp.abs(J) > 1e-12, J, jnp.sign(J) * 1e-12 + 1e-12)
    FinvT = jnp.linalg.inv(F).T
    logJ = jnp.log(jnp.abs(J_safe))
    return mu * (F - FinvT) + lam * logJ * FinvT


def neo_hookean_cauchy_stress(F: jnp.ndarray, lam: float, mu: float) -> jnp.ndarray:
    """Cauchy stress = (1/J) P F^T."""
    P = neo_hookean_piola_kirchhoff(F, lam, mu)
    J = jnp.linalg.det(F)
    J_safe = jnp.where(jnp.abs(J) > 1e-12, J, jnp.sign(J) * 1e-12 + 1e-12)
    return P @ F.T / J_safe


@dataclass(frozen=True)
class NeoHookean:
    """Compressible Neo-Hookean material parameters."""

    E: float
    nu: float

    @property
    def lam(self) -> float:
        return self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

    @property
    def mu(self) -> float:
        return self.E / (2 * (1 + self.nu))

    def piola(self, F: jnp.ndarray) -> jnp.ndarray:
        return neo_hookean_piola_kirchhoff(F, self.lam, self.mu)

    def cauchy(self, F: jnp.ndarray) -> jnp.ndarray:
        return neo_hookean_cauchy_stress(F, self.lam, self.mu)

    def strain_energy(self, F: jnp.ndarray) -> jnp.ndarray:
        dim = F.shape[0]
        B = F @ F.T
        J = jnp.linalg.det(F)
        J_safe = jnp.where(jnp.abs(J) > 1e-12, J, jnp.sign(J) * 1e-12 + 1e-12)
        logJ = jnp.log(jnp.abs(J_safe))
        return 0.5 * self.mu * (jnp.trace(B) - dim) - self.mu * logJ + \
               0.5 * self.lam * logJ ** 2
