"""Explicit and semi-implicit time integrators.

Each function advances a single step. They are pure functions: the state
(position, velocity, etc.) is passed in and a new state is returned, so the
caller can wrap them in ``jax.lax.scan`` for a time-stepping loop.

For mechanics:
    - velocity_verlet_step : 2nd-order symplectic, good for long-time MD / DEM
    - semi_implicit_euler_step : 1st-order symplectic, common in DEM/SPH
    - explicit_euler_step : 1st-order, mainly for testing
    - rk4_step : 4th-order, generic for ODEs (used in phase-field evolution)
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp


def explicit_euler_step(rhs: Callable, t, y, dt):
    """y_{n+1} = y_n + dt * f(t, y_n)."""
    return y + dt * rhs(t, y)


def semi_implicit_euler_step(
    compute_force: Callable,
    pos: jnp.ndarray,
    vel: jnp.ndarray,
    mass: jnp.ndarray,
    dt: float,
    *,
    damping: float = 0.0,
):
    """Semi-implicit (symplectic) Euler for second-order systems.

    v_{n+1} = v_n + dt * (F(x_n)/m - damping * v_n)
    x_{n+1} = x_n + dt * v_{n+1}

    Parameters
    ----------
    compute_force : callable, pos -> force
    pos, vel : (n, dim) arrays
    mass : (n,) or (n, 1) array
    dt : time step
    damping : viscous damping coefficient (applied to velocity)
    """
    F = compute_force(pos)
    if mass.ndim == 1:
        mass = mass[:, None]
    acc = F / mass - damping * vel
    vel_new = vel + dt * acc
    pos_new = pos + dt * vel_new
    return pos_new, vel_new


def velocity_verlet_step(
    compute_accel: Callable,
    pos: jnp.ndarray,
    vel: jnp.ndarray,
    acc: jnp.ndarray,
    dt: float,
):
    """Velocity Verlet (symplectic, 2nd-order) for second-order systems.

    Requires the *acceleration* at the start of the step (so the caller must
    keep this state across steps).

    x_{n+1} = x_n + dt*v_n + 0.5*dt^2*a_n
    a_{n+1} = F(x_{n+1}) / m
    v_{n+1} = v_n + 0.5*dt*(a_n + a_{n+1})

    Returns (pos_new, vel_new, acc_new).
    """
    pos_new = pos + dt * vel + 0.5 * dt * dt * acc
    acc_new = compute_accel(pos_new)
    vel_new = vel + 0.5 * dt * (acc + acc_new)
    return pos_new, vel_new, acc_new


def rk4_step(rhs: Callable, t, y, dt):
    """Classical RK4 for y' = f(t, y)."""
    k1 = rhs(t, y)
    k2 = rhs(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = rhs(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = rhs(t + dt, y + dt * k3)
    return y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
