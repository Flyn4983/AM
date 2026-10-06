"""Differentiable Smoothed Particle Hydrodynamics (WCSPH) for fluid dynamics.

Implements Weakly Compressible SPH with a cubic-spline kernel, Tait equation
of state, and artificial viscosity.  Works in 2D and 3D — dimensionality is
inferred from the particle coordinate arrays.

All pairwise interactions (density summation, force evaluation) use
``jax.vmap`` over an ``N × N`` pair matrix, making the whole simulation
``jax.jit`` and ``jax.grad`` friendly.

Key equations
-------------
- Density:      ``ρ_i = Σ_j m_j W(|r_ij|, h)``
- Pressure:     ``p = c₀² (ρ - ρ₀)``  (linearised Tait)
- Momentum:     ``du_i/dt = -Σ_j m_j (p_i/ρ_i² + p_j/ρ_j²) ∇W_ij
                                  + ν Σ_j m_j (u_j - u_i)/ρ_j · η_ij ∇W_ij / (|r_ij|² + 0.01h²)``
- Energy:       ``dE_i/dt = 0.5 Σ_j m_j (p_i/ρ_i² + p_j/ρ_j²) (u_i - u_j) · ∇W_ij``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Cubic spline kernel and its gradient
# ---------------------------------------------------------------------------
def _cubic_kernel_norm(dim: int, h: float) -> float:
    """Normalisation constant α_d such that ``∫W dV = 1``."""
    if dim == 2:
        return 10.0 / (7.0 * jnp.pi * h * h)
    return 1.0 / (jnp.pi * h * h * h)  # dim == 3


def cubic_kernel(r: jnp.ndarray, h: float, dim: int) -> jnp.ndarray:
    """Cubic spline kernel ``W(r, h)``.

    ``r`` is the scalar inter-particle distance.
    """
    q = r / h
    alpha = _cubic_kernel_norm(dim, h)
    W = jnp.where(q < 1.0,
                  alpha * (1.0 - 1.5 * q ** 2 + 0.75 * q ** 3),
                  jnp.where(q < 2.0,
                            alpha * 0.25 * (2.0 - q) ** 3,
                            0.0))
    return W


def cubic_kernel_grad(rij: jnp.ndarray, h: float, dim: int) -> jnp.ndarray:
    """Gradient ``∇W(r_ij, h)`` — a vector pointing from j to i.

    ``rij`` is the displacement vector ``r_i - r_j``.
    """
    r = jnp.sqrt(jnp.sum(rij ** 2) + 1e-16)  # smooth, avoids NaN at r=0
    q = r / h
    alpha = _cubic_kernel_norm(dim, h)
    # dW/dq
    dWdq = jnp.where(q < 1.0,
                     alpha * (-3.0 * q + 2.25 * q ** 2),
                     jnp.where(q < 2.0,
                               -alpha * 0.75 * (2.0 - q) ** 2,
                               0.0))
    # ∇W = dW/dr * (rij / r) = (dW/dq / h) * (rij / r)
    grad = dWdq / h * rij / r
    return grad


# ---------------------------------------------------------------------------
# State and configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SPHConfig:
    """SPH simulation parameters."""

    h: float = 0.1           # smoothing length
    rho0: float = 1000.0     # reference density
    c0: float = 50.0         # speed of sound (controls compressibility)
    nu: float = 0.0          # artificial viscosity coefficient
    gamma_eos: float = 7.0   # Tait exponent (7 for water)
    gravity: float = 0.0     # body force along last axis


@dataclass(frozen=True)
class SPHState:
    """SPH particle state.

    Attributes
    ----------
    position : (N, dim) array
    velocity : (N, dim) array
    mass : (N,) array
    """

    position: jax.Array
    velocity: jax.Array
    mass: jax.Array

    @property
    def n_particles(self) -> int:
        return int(self.position.shape[0])

    @property
    def dim(self) -> int:
        return int(self.position.shape[1])


def _sph_flatten(s: SPHState):
    return (s.position, s.velocity, s.mass), None

def _sph_unflatten(_, children):
    pos, vel, mass = children
    return SPHState(pos, vel, mass)

jax.tree_util.register_pytree_node(SPHState, _sph_flatten, _sph_unflatten)


def make_sph_state(
    position: jnp.ndarray,
    velocity: jnp.ndarray | None = None,
    mass: float | jnp.ndarray = 1.0,
) -> SPHState:
    """Convenience constructor."""
    pos = jnp.asarray(position, dtype=jnp.float64)
    N = pos.shape[0]
    vel = jnp.zeros_like(pos) if velocity is None else jnp.asarray(velocity, dtype=jnp.float64)
    m = jnp.broadcast_to(jnp.asarray(mass, dtype=jnp.float64), (N,)).copy()
    return SPHState(position=pos, velocity=vel, mass=m)


# ---------------------------------------------------------------------------
# Density and pressure
# ---------------------------------------------------------------------------
def compute_density(state: SPHState, cfg: SPHConfig) -> jnp.ndarray:
    """SPH density summation: ``ρ_i = Σ_j m_j W(|r_ij|, h)``."""
    dim = state.dim
    h = cfg.h

    def density_i(pos_i):
        def contrib(pos_j, m_j):
            r = jnp.sqrt(jnp.sum((pos_i - pos_j) ** 2) + 1e-16)
            return m_j * cubic_kernel(r, h, dim)
        # Sum over j (the last axis) — using axis=-1 is robust to the
        # extra batch dimension that the outer vmap prepends.
        return jnp.sum(jax.vmap(contrib)(state.position, state.mass), axis=-1)

    return jax.vmap(density_i)(state.position)


def pressure_from_density(rho: jnp.ndarray, cfg: SPHConfig) -> jnp.ndarray:
    """Tait equation of state: ``p = c₀² (ρ - ρ₀)`` (linearised).

    For the full Tait form use ``gamma_eos > 1``:
    ``p = c₀² ρ₀ / γ * ((ρ/ρ₀)^γ - 1)``.
    """
    if cfg.gamma_eos <= 1.0:
        return cfg.c0 ** 2 * (rho - cfg.rho0)
    return cfg.c0 ** 2 * cfg.rho0 / cfg.gamma_eos * (
        (rho / cfg.rho0) ** cfg.gamma_eos - 1.0
    )


# ---------------------------------------------------------------------------
# Momentum equation (pressure + viscosity)
# ---------------------------------------------------------------------------
def compute_acceleration(state: SPHState, cfg: SPHConfig) -> jnp.ndarray:
    """Compute the acceleration of each particle from the SPH momentum equation.

    Includes pressure gradient (symmetric form) and artificial viscosity.
    """
    dim = state.dim
    h = cfg.h
    rho = compute_density(state, cfg)
    p = pressure_from_density(rho, cfg)

    def accel_i(pos_i, vel_i, rho_i, p_i):
        def pair_force(pos_j, vel_j, m_j, rho_j, p_j):
            rij = pos_i - pos_j
            r = jnp.sqrt(jnp.sum(rij ** 2) + 1e-16)
            grad_W = cubic_kernel_grad(rij, h, dim)

            # Pressure term: -m_j * (p_i/ρ_i² + p_j/ρ_j²) * ∇W
            eps = 1e-30
            f_pressure = -m_j * (p_i / (rho_i ** 2 + eps) +
                                 p_j / (rho_j ** 2 + eps)) * grad_W

            # Artificial viscosity (Monaghan): π_ij
            vij = vel_i - vel_j
            v_dot_r = jnp.dot(vij, rij)
            pi_ij = jnp.where(v_dot_r < 0.0,
                             -cfg.nu * (2.0 * h * v_dot_r) / (r ** 2 + 0.01 * h * h),
                             0.0)
            f_visc = -m_j * pi_ij * grad_W

            return f_pressure + f_visc

        # Self-interaction (j==i) naturally gives zero force because
        # grad_W(0) = 0 (the kernel gradient vanishes at the origin).
        forces = jax.vmap(pair_force)(
            state.position, state.velocity, state.mass, rho, p
        )
        total_force = jnp.sum(forces, axis=0)   # (dim,)
        # Standard symmetric SPH form: the 1/rho_i^2 factor is already
        # inside the pressure term, so NO extra division by rho_i.
        # This guarantees pairwise force anti-symmetry and thus
        # exact momentum conservation.
        return total_force

    accel = jax.vmap(accel_i)(state.position, state.velocity, rho, p)
    # Add gravity along last axis
    g = cfg.gravity
    accel = accel.at[:, -1].add(g)
    return accel


# ---------------------------------------------------------------------------
# Time integration
# ---------------------------------------------------------------------------
def step_sph(
    state: SPHState,
    dt: float,
    cfg: SPHConfig,
) -> SPHState:
    """Advance SPH by one semi-implicit Euler step."""
    accel = compute_acceleration(state, cfg)
    v_new = state.velocity + dt * accel
    x_new = state.position + dt * v_new
    return SPHState(position=x_new, velocity=v_new, mass=state.mass)


def step_sph_scan(
    state0: SPHState,
    dt: float,
    n_steps: int,
    cfg: SPHConfig,
) -> SPHState:
    """Time-step SPH for ``n_steps`` using ``jax.lax.scan`` (differentiable)."""
    def body(carry, _):
        return step_sph(carry, dt, cfg), None
    state_final, _ = jax.lax.scan(body, state0, xs=None, length=n_steps)
    return state_final


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def total_mass(state: SPHState) -> jnp.ndarray:
    return jnp.sum(state.mass)


def kinetic_energy(state: SPHState) -> jnp.ndarray:
    return 0.5 * jnp.sum(state.mass * jnp.sum(state.velocity ** 2, axis=-1))
