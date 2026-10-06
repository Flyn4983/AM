"""Differentiable Discrete Element Method (DEM) for granular dynamics (2D & 3D).

Models a collection of soft spherical particles interacting through pairwise
contact forces.  The contact model is a linear spring-dashpot (Hookean normal
repulsion + viscous damping), optionally with tangential Coulomb friction.

All pairwise interactions are computed with ``jax.vmap`` over an ``N x N`` pair
matrix.  This is ``O(N^2)`` per step but fully ``jax.jit`` / ``jax.grad``
friendly — no sparse neighbour lists or variable-length contact arrays that
would break reverse-mode AD.

The time integrator is semi-implicit (symplectic) Euler, advanced through a
fixed number of steps with ``jax.lax.scan`` so the whole trajectory is
differentiable end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DEMConfig:
    """Material / contact parameters (static during a simulation)."""

    k: float = 1000.0       # normal contact stiffness  [N/m]
    gamma: float = 1.0      # normal viscous damping    [N·s/m]
    mu: float = 0.0         # Coulomb friction coefficient (0 = frictionless)
    kt: float = 0.0          # tangential stiffness (0 => no tangential history)
    restitution: float = 0.0  # not used directly; controlled via gamma
    gravity: float = 0.0     # body force along the last spatial axis (+z / +y)


@dataclass(frozen=True)
class DEMState:
    """Particle system state at a single instant.

    All arrays have a leading axis of size ``N`` (number of particles).

    Attributes
    ----------
    position : (N, dim) array
    velocity : (N, dim) array
    radius : (N,) array
    mass : (N,) array
    """

    position: jax.Array
    velocity: jax.Array
    radius: jax.Array
    mass: jax.Array

    @property
    def n_particles(self) -> int:
        return int(self.position.shape[0])

    @property
    def dim(self) -> int:
        return int(self.position.shape[1])


# Register as a JAX pytree so that jit/grad/scan can trace through it.
def _dem_flatten(s: DEMState):
    return (s.position, s.velocity, s.radius, s.mass), None

def _dem_unflatten(_, children):
    pos, vel, rad, mass = children
    return DEMState(pos, vel, rad, mass)

jax.tree_util.register_pytree_node(DEMState, _dem_flatten, _dem_unflatten)


def make_dem_state(
    position: jnp.ndarray,
    velocity: jnp.ndarray | None = None,
    radius: float | jnp.ndarray = 0.05,
    mass: float | jnp.ndarray = 1.0,
    dim: int | None = None,
) -> DEMState:
    """Convenience constructor with sensible defaults."""
    pos = jnp.asarray(position, dtype=jnp.float64)
    N, d = pos.shape
    if dim is None:
        dim = d
    vel = jnp.zeros_like(pos) if velocity is None else jnp.asarray(velocity, dtype=jnp.float64)
    r = jnp.broadcast_to(jnp.asarray(radius, dtype=jnp.float64), (N,)).copy()
    m = jnp.broadcast_to(jnp.asarray(mass, dtype=jnp.float64), (N,)).copy()
    return DEMState(position=pos, velocity=vel, radius=r, mass=m)


# ---------------------------------------------------------------------------
# Pairwise contact force (linear spring-dashpot + optional Coulomb friction)
# ---------------------------------------------------------------------------
def _pair_contact_force(
    pos_i: jnp.ndarray, vel_i: jnp.ndarray, r_i: jnp.ndarray, m_i: jnp.ndarray,
    pos_j: jnp.ndarray, vel_j: jnp.ndarray, r_j: jnp.ndarray, m_j: jnp.ndarray,
    cfg: DEMConfig,
) -> jnp.ndarray:
    """Force on particle *i* due to contact with particle *j*.

    Returns a zero vector when the particles are not overlapping.
    """
    delta = pos_j - pos_i                       # (dim,)  points from i to j
    # Smooth distance: adding a tiny epsilon keeps the gradient finite when
    # two particles coincide (self-interaction), preventing NaN in the
    # backward pass.  The forward-pass force is still zero for self-pairs
    # because n_hat = delta / dist = 0.
    dist = jnp.sqrt(jnp.sum(delta ** 2) + 1e-16)
    n_hat = delta / dist                         # unit normal i -> j
    overlap = r_i + r_j - dist                   # > 0 when in contact

    # Normal relative velocity (approaching => v_n < 0)
    v_rel = vel_j - vel_i
    v_n = jnp.dot(v_rel, n_hat)

    # Normal force: repulsive spring + viscous damping (only in contact).
    # The force on i points in the -n_hat direction (AWAY from j).
    f_n_mag = jnp.where(overlap > 0.0,
                        cfg.k * overlap - cfg.gamma * v_n,
                        0.0)
    f_normal = -f_n_mag * n_hat                  # pushes i away from j

    # Tangential Coulomb friction (rate-independent, no history)
    if cfg.mu > 0.0 and cfg.kt > 0.0:
        v_t_vec = v_rel - v_n * n_hat
        f_tangential = -cfg.kt * v_t_vec
        f_t_mag = jnp.sqrt(jnp.sum(f_tangential ** 2) + 1e-16)
        f_t_clipped = jnp.minimum(f_t_mag, cfg.mu * jnp.maximum(f_n_mag, 0.0))
        f_tangential = jnp.where(f_t_mag > 1e-16,
                                  f_tangential / f_t_mag * f_t_clipped,
                                  jnp.zeros_like(f_tangential))
    else:
        f_tangential = jnp.zeros_like(f_normal)

    active = overlap > 0.0
    return jnp.where(active, f_normal + f_tangential, jnp.zeros_like(f_normal))


def _compute_contact_forces(state: DEMState, cfg: DEMConfig) -> jnp.ndarray:
    """Sum of contact forces on each particle from all others (N x dim)."""
    N = state.n_particles
    # Vectorised over all (i, j) pairs.
    force_ij = jax.vmap(
        jax.vmap(
            lambda i_pos, i_vel, i_r, i_m,
                   j_pos, j_vel, j_r, j_m:
                _pair_contact_force(i_pos, i_vel, i_r, i_m,
                                    j_pos, j_vel, j_r, j_m, cfg),
            in_axes=(None, None, None, None, 0, 0, 0, 0),
        ),
        in_axes=(0, 0, 0, 0, None, None, None, None),
    )(state.position, state.velocity, state.radius, state.mass,
      state.position, state.velocity, state.radius, state.mass)
    # force_ij[i, j] = force on i from j.  Exclude self-interaction (i == j).
    eye = jnp.eye(N, dtype=force_ij.dtype)[:, :, None]
    forces = jnp.sum(force_ij * (1.0 - eye), axis=1)   # (N, dim)
    return forces


# ---------------------------------------------------------------------------
# Wall / boundary forces
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Wall:
    """A flat infinite wall defined by ``normal . x = offset``.

    Particles are confined to the half-space ``normal . x <= offset`` (the
    wall pushes particles on its *positive* side back toward the negative
    side).
    """
    normal: jnp.ndarray   # (dim,) unit normal pointing into the domain
    offset: float         # wall plane: normal . x = offset


def _wall_force(pos: jnp.ndarray, vel: jnp.ndarray, r: jnp.ndarray,
                wall: Wall, cfg: DEMConfig) -> jnp.ndarray:
    """Force on a particle from a single wall (linear spring-dashpot)."""
    penetration = jnp.dot(pos, wall.normal) - wall.offset + r
    eps = 1e-30
    active = penetration > 0.0
    v_n = jnp.dot(vel, wall.normal)
    f_mag = jnp.where(active, cfg.k * penetration - cfg.gamma * v_n, 0.0)
    return jnp.where(active, -f_mag * wall.normal, jnp.zeros_like(pos))


def _compute_wall_forces(state: DEMState, walls: list[Wall] | None,
                         cfg: DEMConfig) -> jnp.ndarray:
    if not walls:
        return jnp.zeros_like(state.position)
    per_wall = jnp.stack(
        [jax.vmap(lambda p, v, r: _wall_force(p, v, r, w, cfg))(
            state.position, state.velocity, state.radius)
         for w in walls],
        axis=0,
    )
    return jnp.sum(per_wall, axis=0)


# ---------------------------------------------------------------------------
# Time integration (semi-implicit Euler, differentiable via lax.scan)
# ---------------------------------------------------------------------------
def step_dem(
    state: DEMState,
    dt: float,
    cfg: DEMConfig,
    *,
    walls: list[Wall] | None = None,
) -> DEMState:
    """Advance the DEM system by one semi-implicit Euler step."""
    f_contact = _compute_contact_forces(state, cfg)
    f_wall = _compute_wall_forces(state, walls, cfg) if walls else jnp.zeros_like(f_contact)
    # Body force (gravity along last axis)
    g = cfg.gravity
    f_body = jnp.zeros_like(f_contact).at[:, -1].set(g * state.mass)
    f_total = f_contact + f_wall + f_body

    # Semi-implicit Euler: v_{n+1} = v_n + dt * a; x_{n+1} = x_n + dt * v_{n+1}
    accel = f_total / state.mass[:, None]
    v_new = state.velocity + dt * accel
    x_new = state.position + dt * v_new
    return DEMState(position=x_new, velocity=v_new,
                    radius=state.radius, mass=state.mass)


def step_dem_scan(
    state0: DEMState,
    dt: float,
    n_steps: int,
    cfg: DEMConfig,
    *,
    walls: list[Wall] | None = None,
) -> DEMState:
    """Time-step DEM for ``n_steps`` using ``jax.lax.scan`` (differentiable).

    ``walls`` must be a Python list (static).  If you need the wall positions
    to be differentiable, pass them as arrays inside the state instead.
    """
    def body(carry, _):
        return step_dem(carry, dt, cfg, walls=walls), None
    state_final, _ = jax.lax.scan(body, state0, xs=None, length=n_steps)
    return state_final


# ---------------------------------------------------------------------------
# Kinetic energy (useful as an optimisation objective)
# ---------------------------------------------------------------------------
def kinetic_energy(state: DEMState) -> jnp.ndarray:
    """Total kinetic energy ``sum_i 0.5 m_i |v_i|^2``."""
    return 0.5 * jnp.sum(state.mass * jnp.sum(state.velocity ** 2, axis=-1))
