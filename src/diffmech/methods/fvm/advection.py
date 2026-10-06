"""Scalar linear advection on a Cartesian FVM grid (2D & 3D).

Equation:
    du/dt + div(v * u) = 0
where ``v`` is a constant velocity vector and ``u`` is a transported scalar.

Discretisation: finite volume with upwind flux (first order), advanced in
time with SSP-RK2 (Heun's method). Boundary conditions are handled via ghost
cells filled either periodically (``jnp.roll``-based, jit-friendly) or by a
user-supplied callable.

Everything is written so that ``jax.grad`` flows through it end-to-end: the
state is a plain ``(nx, ny[, nz], 1)`` array.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from diffmech.methods.fvm.grid import CartesianGrid


def upwind_flux(u_L: jnp.ndarray, u_R: jnp.ndarray, v_n: jnp.ndarray) -> jnp.ndarray:
    """First-order upwind flux through a face with normal velocity ``v_n``."""
    return 0.5 * v_n * (u_L + u_R) - 0.5 * jnp.abs(v_n) * (u_R - u_L)


def _periodic_rhs(u: jnp.ndarray, velocity: jnp.ndarray, grid: CartesianGrid) -> jnp.ndarray:
    """Compute -div(v*u) with periodic BCs via ``jnp.roll``.

    For each axis ``d`` we form the face flux ``F_{i+1/2} = upwind(u_i, u_{i+1}, v)``
    and the cell update ``rhs_i = - sum_d (F_{i+1/2} - F_{i-1/2}) / dx_d`` where
    ``F_{i-1/2} = F_{(i-1)+1/2}`` is obtained by rolling ``F_plus`` by +1.
    """
    cell_sizes = [grid.dx, grid.dy] if grid.dim == 2 else [grid.dx, grid.dy, grid.dz]
    rhs = jnp.zeros_like(u)
    for d, (dx, v) in enumerate(zip(cell_sizes, velocity)):
        u_c = u
        u_r = jnp.roll(u, -1, axis=d)             # u[i+1]
        F_plus = upwind_flux(u_c, u_r, v)         # F_{i+1/2} = upwind(u_i, u_{i+1})
        F_minus = jnp.roll(F_plus, 1, axis=d)     # F_{i-1/2} = F_{(i-1)+1/2}
        rhs = rhs - (F_plus - F_minus) / dx
    return rhs


def _ghost_rhs(u: jnp.ndarray, velocity: jnp.ndarray, grid: CartesianGrid,
               apply_bc: Callable) -> jnp.ndarray:
    """Compute -div(v*u) using a ghost-cell padded field.

    ``apply_bc(field)`` should return ``field`` padded with one ghost cell on
    each side along every spatial axis (the trailing component axis is left
    alone). The ghost-cell values are determined by the chosen BC.
    """
    ng = 1
    u_padded = apply_bc(u)
    cell_sizes = [grid.dx, grid.dy] if grid.dim == 2 else [grid.dx, grid.dy, grid.dz]
    rhs = jnp.zeros_like(u)
    for d, (dx, v) in enumerate(zip(cell_sizes, velocity)):
        # Build three slice tuples: centre / left-neighbour / right-neighbour
        # along axis d. All other axes are full slices.
        sl_c = [slice(None)] * u_padded.ndim
        sl_l = [slice(None)] * u_padded.ndim
        sl_r = [slice(None)] * u_padded.ndim
        sl_c[d] = slice(ng, -ng)               # interior slice [ng, ng+nx)
        sl_l[d] = slice(ng - 1, -ng - 1)       # left neighbour [ng-1, ng+nx-1)
        sl_r[d] = slice(ng + 1, None) if ng + 1 < u_padded.shape[d] - ng else slice(ng + 1, -ng + 1)
        u_c = u_padded[tuple(sl_c)]
        u_l = u_padded[tuple(sl_l)]
        u_r = u_padded[tuple(sl_r)]
        # Flux through the +d face of cell i: upwind(u_c, u_r)
        F_plus = upwind_flux(u_c, u_r, v)
        # Flux through the -d face of cell i: upwind(u_l, u_c)
        F_minus = upwind_flux(u_l, u_c, v)
        rhs = rhs - (F_plus - F_minus) / dx
    return rhs


def _interior_rhs(u: jnp.ndarray, velocity: jnp.ndarray, grid: CartesianGrid,
                  *, bc, t: float) -> jnp.ndarray:
    if bc == "periodic":
        return _periodic_rhs(u, velocity, grid)
    if bc == "zero_dirichlet":
        def apply_bc(field):
            if grid.dim == 2:
                return jnp.pad(field, ((1, 1), (1, 1), (0, 0)),
                               mode="constant", constant_values=0.0)
            return jnp.pad(field, ((1, 1),) * 3 + ((0, 0),),
                           mode="constant", constant_values=0.0)
        return _ghost_rhs(u, velocity, grid, apply_bc)
    if callable(bc):
        return _ghost_rhs(u, velocity, grid, bc)
    raise ValueError(f"unknown bc {bc!r}")


def step_advection(
    u: jnp.ndarray,
    velocity: jnp.ndarray,
    grid: CartesianGrid,
    dt: float,
    *,
    bc: str | Callable = "periodic",
    order: int = 1,
    t: float = 0.0,
) -> jnp.ndarray:
    """Advance the scalar advection equation by one SSP-RK2 step."""
    del order  # currently only first-order upwind is implemented
    def rhs(state, t_eval):
        return _interior_rhs(state, velocity, grid, bc=bc, t=t_eval)
    k1 = rhs(u, t)
    u1 = u + dt * k1
    k2 = rhs(u1, t + dt)
    return u + 0.5 * dt * (k1 + k2)


def step_advection_scan(
    u0: jnp.ndarray,
    velocity: jnp.ndarray,
    grid: CartesianGrid,
    dt: float,
    n_steps: int,
    *,
    bc: str | Callable = "periodic",
) -> jnp.ndarray:
    """Time-step the advection equation with ``jax.lax.scan`` (differentiable)."""
    def body(u, _):
        return step_advection(u, velocity, grid, dt, bc=bc), None
    u_final, _ = jax.lax.scan(body, u0, xs=None, length=n_steps)
    return u_final


def cfl_dt(velocity: jnp.ndarray, grid: CartesianGrid, cfl: float = 0.5) -> float:
    """CFL-limited time step ``dt <= cfl * min_d (dx_d / |v_d|)``.

    Directions with zero velocity are ignored (do not constrain ``dt``).
    """
    import math
    cell_sizes = [grid.dx, grid.dy] if grid.dim == 2 else [grid.dx, grid.dy, grid.dz]
    v = jnp.atleast_1d(jnp.abs(jnp.asarray(velocity)))
    v_np = [float(v[i]) for i in range(len(v))]
    dt_per_axis = [math.inf if vv < 1e-30 else d / vv
                   for d, vv in zip(cell_sizes, v_np)]
    return float(cfl * min(dt_per_axis))
