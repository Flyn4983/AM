"""Compressible Euler equations on a Cartesian FVM grid (2D & 3D).

Conservative variables:
    U = (rho, rho*u, rho*v[, rho*w], E)

with total energy per unit volume ``E = p/(gamma-1) + 0.5 * rho * |u|^2``.

Equation:
    dU/dt + div(F(U)) = 0
where ``F = u * U + p * e_velocity`` (the standard Euler flux).

Discretisation: finite volume with local Lax-Friedrichs (Rusanov) flux, advanced
in time with SSP-RK2. The Rusanov flux is differentiable through JAX, so
``jax.grad`` works end-to-end.

The implementation is identical for 2D and 3D: the state array has shape
``(nx, ny[, nz], n_eq)`` with ``n_eq = dim + 2``. Each spatial axis is treated
with the same Rusanov flux, where the eigenvalue ``|v_n| + c`` is computed
pointwise.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from diffmech.methods.fvm.grid import CartesianGrid


# ---------------------------------------------------------------------------
# Thermodynamic helpers.
# ---------------------------------------------------------------------------
def pressure(U: jnp.ndarray, gamma: float, dim: int) -> jnp.ndarray:
    """Pressure from conservative state ``U``."""
    rho = U[..., 0]
    rho_u = U[..., 1:1 + dim]
    E = U[..., -1]
    kinetic = 0.5 * jnp.sum(rho_u ** 2, axis=-1) / jnp.maximum(rho, 1e-30)
    return (gamma - 1.0) * (E - kinetic)


def sound_speed(U: jnp.ndarray, gamma: float, dim: int) -> jnp.ndarray:
    """Adiabatic sound speed ``c = sqrt(gamma * p / rho)``."""
    p = pressure(U, gamma, dim)
    rho = U[..., 0]
    return jnp.sqrt(gamma * p / jnp.maximum(rho, 1e-30))


def conservative_to_primitive(U: jnp.ndarray, gamma: float, dim: int):
    """Return (rho, velocity (dim,), p) from conservative state."""
    rho = U[..., 0]
    v = U[..., 1:1 + dim] / jnp.maximum(rho[..., None], 1e-30)
    p = pressure(U, gamma, dim)
    return rho, v, p


# ---------------------------------------------------------------------------
# Rusanov flux along one axis given a unit-normal index ``axis``.
# ---------------------------------------------------------------------------
def rusanov_flux_axis(U_L: jnp.ndarray, U_R: jnp.ndarray,
                      gamma: float, dim: int, axis: int) -> jnp.ndarray:
    """Local Lax-Friedrichs (Rusanov) flux along ``axis``.

    ``U_L`` and ``U_R`` are conservative states on the two sides of a face
    whose normal points in direction ``axis``. Returns the flux vector
    ``F(U) * n_axis`` through the face.
    """
    rho_L, v_L, p_L = conservative_to_primitive(U_L, gamma, dim)
    rho_R, v_R, p_R = conservative_to_primitive(U_R, gamma, dim)
    c_L = jnp.sqrt(gamma * p_L / jnp.maximum(rho_L, 1e-30))
    c_R = jnp.sqrt(gamma * p_R / jnp.maximum(rho_R, 1e-30))
    v_n_L = v_L[..., axis]
    v_n_R = v_R[..., axis]
    # Maximum signal speed
    smax = jnp.maximum(jnp.abs(v_n_L) + c_L, jnp.abs(v_n_R) + c_R)

    # Physical flux along axis: F(U) for Euler equations.
    # F[0]   = rho * v_n
    # F[1..d]= rho * v_n * v_d + p * delta_{d, axis}
    # F[-1]  = (E + p) * v_n
    E_L = U_L[..., -1]
    E_R = U_R[..., -1]
    F_L = jnp.zeros_like(U_L)
    F_R = jnp.zeros_like(U_R)
    F_L = F_L.at[..., 0].set(rho_L * v_n_L)
    F_R = F_R.at[..., 0].set(rho_R * v_n_R)
    F_L = F_L.at[..., 1:1 + dim].set(rho_L[..., None] * v_n_L[..., None] * v_L)
    F_R = F_R.at[..., 1:1 + dim].set(rho_R[..., None] * v_n_R[..., None] * v_R)
    F_L = F_L.at[..., 1 + axis].add(p_L)
    F_R = F_R.at[..., 1 + axis].add(p_R)
    F_L = F_L.at[..., -1].set((E_L + p_L) * v_n_L)
    F_R = F_R.at[..., -1].set((E_R + p_R) * v_n_R)

    return 0.5 * (F_L + F_R) - 0.5 * smax[..., None] * (U_R - U_L)


# ---------------------------------------------------------------------------
# Spatial RHS with periodic BCs (jnp.roll based).
# ---------------------------------------------------------------------------
def _periodic_rhs(U: jnp.ndarray, gamma: float, grid: CartesianGrid) -> jnp.ndarray:
    """Compute -div(F(U)) with periodic BCs via ``jnp.roll``.

    For each axis ``d`` we form ``F_{i+1/2} = rusanov(U_i, U_{i+1})`` and
    ``F_{i-1/2} = F_{(i-1)+1/2}`` by rolling ``F_plus`` by +1.
    """
    dim = grid.dim
    cell_sizes = [grid.dx, grid.dy] if dim == 2 else [grid.dx, grid.dy, grid.dz]
    rhs = jnp.zeros_like(U)
    for d, dx in enumerate(cell_sizes):
        U_c = U
        U_r = jnp.roll(U, -1, axis=d)             # U[i+1]
        F_plus = rusanov_flux_axis(U_c, U_r, gamma, dim, d)     # F_{i+1/2}
        F_minus = jnp.roll(F_plus, 1, axis=d)                  # F_{i-1/2}
        rhs = rhs - (F_plus - F_minus) / dx
    return rhs


def _ghost_rhs(U: jnp.ndarray, gamma: float, grid: CartesianGrid, apply_bc: Callable) -> jnp.ndarray:
    """Compute -div(F(U)) with a ghost-cell padded field (general BCs)."""
    dim = grid.dim
    ng = 1
    U_padded = apply_bc(U)
    cell_sizes = [grid.dx, grid.dy] if dim == 2 else [grid.dx, grid.dy, grid.dz]
    rhs = jnp.zeros_like(U)
    for d, dx in enumerate(cell_sizes):
        sl_c = [slice(None)] * U_padded.ndim
        sl_l = [slice(None)] * U_padded.ndim
        sl_r = [slice(None)] * U_padded.ndim
        sl_c[d] = slice(ng, -ng)
        sl_l[d] = slice(ng - 1, -ng - 1)
        sl_r[d] = slice(ng + 1, None) if ng + 1 < U_padded.shape[d] - ng else slice(ng + 1, -ng + 1)
        U_c = U_padded[tuple(sl_c)]
        U_l = U_padded[tuple(sl_l)]
        U_r = U_padded[tuple(sl_r)]
        F_plus = rusanov_flux_axis(U_c, U_r, gamma, dim, d)
        F_minus = rusanov_flux_axis(U_l, U_c, gamma, dim, d)
        rhs = rhs - (F_plus - F_minus) / dx
    return rhs


def _interior_rhs(U: jnp.ndarray, gamma: float, grid: CartesianGrid, *, bc, t: float):
    if bc == "periodic":
        return _periodic_rhs(U, gamma, grid)
    if bc == "reflecting":
        # Reflecting (slip-wall): mirror the velocity component along the face
        # normal; rho, p, tangential velocity are copied.
        def apply_bc(field):
            if grid.dim == 2:
                padded = jnp.pad(field, ((1, 1), (1, 1), (0, 0)), mode="edge")
            else:
                padded = jnp.pad(field, ((1, 1),) * 3 + ((0, 0),), mode="edge")
            padded = _flip_normal_momentum(padded, grid.dim)
            return padded
        return _ghost_rhs(U, gamma, grid, apply_bc)
    if callable(bc):
        return _ghost_rhs(U, gamma, grid, bc)
    raise ValueError(f"unknown bc {bc!r}")


def _flip_normal_momentum(padded: jnp.ndarray, dim: int) -> jnp.ndarray:
    """For each boundary face, flip the sign of the normal momentum component.

    The padded array has one ghost cell on each side along every spatial axis.
    """
    out = padded
    for d in range(dim):
        # The ghost slice at index 0 (low boundary) along axis d
        low = [slice(None)] * padded.ndim
        low[d] = 0
        high = [slice(None)] * padded.ndim
        high[d] = -1
        idx = d + 1  # momentum component index
        out = out.at[tuple(low)].set(out[tuple(low)].at[..., idx].mul(-1.0))
        out = out.at[tuple(high)].set(out[tuple(high)].at[..., idx].mul(-1.0))
    return out


def step_euler(
    U: jnp.ndarray,
    gamma: float,
    grid: CartesianGrid,
    dt: float,
    *,
    bc: str | Callable = "periodic",
    t: float = 0.0,
) -> jnp.ndarray:
    """Advance the Euler equations by one SSP-RK2 step."""
    def rhs(state, t_eval):
        return _interior_rhs(state, gamma, grid, bc=bc, t=t_eval)
    k1 = rhs(U, t)
    U1 = U + dt * k1
    k2 = rhs(U1, t + dt)
    return U + 0.5 * dt * (k1 + k2)


def step_euler_scan(
    U0: jnp.ndarray,
    gamma: float,
    grid: CartesianGrid,
    dt: float,
    n_steps: int,
    *,
    bc: str | Callable = "periodic",
) -> jnp.ndarray:
    """Time-step the Euler equations with ``jax.lax.scan`` (differentiable)."""
    def body(U, _):
        return step_euler(U, gamma, grid, dt, bc=bc), None
    U_final, _ = jax.lax.scan(body, U0, xs=None, length=n_steps)
    return U_final


def cfl_dt(U: jnp.ndarray, gamma: float, grid: CartesianGrid, cfl: float = 0.5) -> float:
    """CFL-limited time step ``dt <= cfl * min_d (dx_d / (|v_d| + c)_max)``."""
    dim = grid.dim
    rho, v, p = conservative_to_primitive(U, gamma, dim)
    c = jnp.sqrt(gamma * p / jnp.maximum(rho, 1e-30))
    cell_sizes = [grid.dx, grid.dy] if dim == 2 else [grid.dx, grid.dy, grid.dz]
    # Reduce over all spatial axes (everything except the trailing velocity axis).
    spatial_axes = tuple(range(v.ndim - 1))
    max_wave = jnp.max(jnp.abs(v) + c[..., None], axis=spatial_axes)  # shape (dim,)
    max_wave_np = [float(max_wave[i]) for i in range(dim)]
    dt_per_axis = [d / max(w, 1e-30) for d, w in zip(cell_sizes, max_wave_np)]
    return float(cfl * min(dt_per_axis))
