"""Incompressible Stokes / Navier-Stokes via the projection method (2D & 3D).

Equations (low-Re, Stokes regime):
    div(u) = 0
    du/dt = -grad(p) / rho + nu * laplacian(u) + f
Discretisation:
    1. Predict: u* = u + dt * (nu * lap(u) - grad(p)/rho + f)
       (we use explicit central differences for both the viscous and pressure
       gradient terms — easily differentiable, slightly restrictive CFL).
    2. Correct: project u* onto the divergence-free space by solving a Poisson
       equation for the pressure:
           lap(p_new) = rho * div(u*) / dt
       Then ``u_new = u* - dt/rho * grad(p_new)``.

The Poisson solve uses a hand-written Conjugate Gradient (CG) iteration on the
discrete Laplacian, written as a fixed-iteration ``jax.lax.scan`` so it is
reverse-mode-AD friendly. For periodic BCs the Laplacian is singular (constant
null space); we pin the mean pressure to zero by removing the mean of the
right-hand side and projecting the solution onto zero-mean at the end.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from diffmech.methods.fvm.grid import CartesianGrid


def _laplacian_periodic(field: jnp.ndarray, grid: CartesianGrid) -> jnp.ndarray:
    """Discrete Laplacian of a scalar field with periodic BCs.

    ``field`` has shape ``(nx, ny[, nz])``. Uses 2nd-order central differences:
        lap(f) = sum_d (f_{i+1} - 2 f_i + f_{i-1}) / dx_d^2.
    """
    dim = grid.dim
    cell_sizes = [grid.dx, grid.dy] if dim == 2 else [grid.dx, grid.dy, grid.dz]
    out = jnp.zeros_like(field)
    for d, dx in enumerate(cell_sizes):
        f_p = jnp.roll(field, -1, axis=d)
        f_m = jnp.roll(field, 1, axis=d)
        out = out + (f_p - 2.0 * field + f_m) / (dx * dx)
    return out


def _grad_periodic(field: jnp.ndarray, grid: CartesianGrid) -> jnp.ndarray:
    """Gradient of a scalar field (periodic BCs), returned as ``(dim,)`` stacked.

    Output shape: ``field.shape + (dim,)``.
    """
    dim = grid.dim
    cell_sizes = [grid.dx, grid.dy] if dim == 2 else [grid.dx, grid.dy, grid.dz]
    grads = []
    for d, dx in enumerate(cell_sizes):
        f_p = jnp.roll(field, -1, axis=d)
        f_m = jnp.roll(field, 1, axis=d)
        grads.append((f_p - f_m) / (2.0 * dx))
    return jnp.stack(grads, axis=-1)


def _divergence_periodic(velocity: jnp.ndarray, grid: CartesianGrid) -> jnp.ndarray:
    """Divergence of a velocity field ``(nx, ny[, nz], dim)`` (periodic BCs)."""
    dim = grid.dim
    cell_sizes = [grid.dx, grid.dy] if dim == 2 else [grid.dx, grid.dy, grid.dz]
    out = jnp.zeros(velocity.shape[:-1], dtype=velocity.dtype)
    for d, dx in enumerate(cell_sizes):
        v_p = jnp.roll(velocity[..., d], -1, axis=d)
        v_m = jnp.roll(velocity[..., d], 1, axis=d)
        out = out + (v_p - v_m) / (2.0 * dx)
    return out


def _cg_solve(matvec: Callable[[jnp.ndarray], jnp.ndarray],
              b: jnp.ndarray, *, max_iter: int = 200,
              project: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
              ) -> jnp.ndarray:
    """Hand-written Conjugate Gradient for symmetric positive semi-definite ``A``.

    Implements the standard CG iteration as a fixed-length ``jax.lax.scan``
    so that the solve is differentiable through ``jax.grad``. No early
    termination; residuals shrink geometrically and the caller picks
    ``max_iter`` large enough. Denominators are protected against zero to
    avoid NaN when the right-hand side is (numerically) zero.

    If ``project`` is provided it is applied to ``x``, ``r`` and ``p`` after
    every iteration.  This is used for the periodic Poisson problem where the
    Laplacian has a constant null space: projecting onto zero-mean keeps the
    iterate inside the range of ``A`` (where ``A`` is strictly positive
    definite), which prevents CG from diverging due to round-off excitation
    of the null space.
    """
    if project is None:
        project = lambda v: v
    x = jnp.zeros_like(b)
    r = project(b - matvec(x))
    p = r
    rs_old = jnp.sum(r * r)
    eps = 1e-30

    def body(carry, _):
        x, p, r, rs_old = carry
        Ap = project(matvec(p))
        pAp = jnp.sum(p * Ap)
        alpha = rs_old / jnp.where(pAp > eps, pAp, eps)
        x = project(x + alpha * p)
        r = project(r - alpha * Ap)
        rs_new = jnp.sum(r * r)
        beta = rs_new / jnp.maximum(rs_old, eps)
        p = project(r + beta * p)
        return (x, p, r, rs_new), None

    (x_final, _, _, _), _ = jax.lax.scan(body, (x, p, r, rs_old),
                                          xs=None, length=max_iter)
    return x_final


def _grad_forward(field: jnp.ndarray, grid: CartesianGrid) -> jnp.ndarray:
    """Forward-difference gradient ``(field.shape + (dim,))`` with periodic BCs.

    ``g_d[i] = (f[i+1,d] - f[i]) / dx_d``.  Together with
    :func:`_div_backward` this reproduces the standard 5-point Laplacian
    exactly (``div_b(grad_f(f)) == _laplacian_periodic(f)``), which is what
    makes the projection step self-consistent.
    """
    dim = grid.dim
    cell_sizes = [grid.dx, grid.dy] if dim == 2 else [grid.dx, grid.dy, grid.dz]
    grads = []
    for d, dx in enumerate(cell_sizes):
        f_p = jnp.roll(field, -1, axis=d)
        grads.append((f_p - field) / dx)
    return jnp.stack(grads, axis=-1)


def _div_backward(velocity: jnp.ndarray, grid: CartesianGrid) -> jnp.ndarray:
    """Backward-difference divergence ``(nx, ny[, nz])`` with periodic BCs.

    ``d[v] = sum_d (v_d[i] - v_d[i-1]) / dx_d``.
    """
    dim = grid.dim
    cell_sizes = [grid.dx, grid.dy] if dim == 2 else [grid.dx, grid.dy, grid.dz]
    out = jnp.zeros(velocity.shape[:-1], dtype=velocity.dtype)
    for d, dx in enumerate(cell_sizes):
        v_m = jnp.roll(velocity[..., d], 1, axis=d)
        out = out + (velocity[..., d] - v_m) / dx
    return out


def _solve_poisson_cg(rhs: jnp.ndarray, grid: CartesianGrid,
                      *, max_iter: int = 200) -> jnp.ndarray:
    """Solve ``lap(p) = rhs`` with periodic BCs using CG.

    The discrete Laplacian ``L = _laplacian_periodic`` is negative
    semi-definite with a one-dimensional null space (the constant mode), so we
    solve the equivalent SPD problem ``-L p = -rhs`` restricted to the
    zero-mean subspace.  Every CG iterate is re-projected onto zero-mean to
    keep the iteration inside the range of ``A`` (where ``A = -L`` is strictly
    positive definite), which prevents round-off excitation of the null space
    from causing divergence.
    """
    shape = rhs.shape
    rhs_zero_mean = rhs - jnp.mean(rhs)

    def neg_lap_mat(v_flat):
        v = v_flat.reshape(shape)
        return -_laplacian_periodic(v, grid).ravel()

    def zero_mean(v_flat):
        return v_flat - jnp.mean(v_flat)

    p_flat = _cg_solve(neg_lap_mat, -rhs_zero_mean.ravel(),
                       max_iter=max_iter, project=zero_mean)
    p = p_flat.reshape(shape)
    p = p - jnp.mean(p)  # enforce zero-mean pressure (gauge)
    return p


def step_stokes(
    velocity: jnp.ndarray,
    pressure: jnp.ndarray,
    grid: CartesianGrid,
    dt: float,
    *,
    nu: float = 1.0,
    rho: float = 1.0,
    force: jnp.ndarray | None = None,
    bc: str | Callable = "periodic",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Advance the incompressible Stokes equation by one projection step.

    Parameters
    ----------
    velocity : (nx, ny[, nz], dim) array
        Cell-centred velocity field at time ``t``.
    pressure : (nx, ny[, nz]) array
        Cell-centred pressure field at time ``t``.
    grid : CartesianGrid
    dt : float
    nu : float
        Kinematic viscosity.
    rho : float
        Density (constant).
    force : (nx, ny[, nz], dim) array, optional
        Body force per unit volume.
    bc : {"periodic"}
        Currently only periodic BCs are supported (the projection step is
        well-posed only with periodic or Dirichlet velocity / Neumann pressure
        BCs; the latter requires extra bookkeeping not included here).

    Returns
    -------
    velocity_new, pressure_new
    """
    if bc != "periodic":
        raise NotImplementedError("Only periodic BCs are supported for Stokes.")

    dim = grid.dim
    # Predict: u* = u + dt * (nu * lap(u) - grad(p)/rho + f)
    # The viscous Laplacian uses the standard 2nd-order central stencil for
    # accuracy; it is a physical diffusion term and does not need to be
    # consistent with the projection operators below.
    lap_u = jnp.stack(
        [_laplacian_periodic(velocity[..., d], grid) for d in range(dim)],
        axis=-1,
    )
    grad_p = _grad_periodic(pressure, grid)
    if force is None:
        force_val = jnp.zeros_like(velocity)
    else:
        force_val = force
    u_star = velocity + dt * (nu * lap_u - grad_p / rho + force_val)

    # Pressure Poisson: lap(p_new) = rho * div(u*) / dt
    #
    # The projection step is self-consistent: we use the forward-difference
    # gradient and backward-difference divergence whose composition equals
    # the standard Laplacian (``div_b(grad_f(·)) == lap(·)``), so that
    # ``div_b(u* - dt/rho * grad_f(p)) = div_b(u*) - dt/rho * lap(p) = 0``
    # up to the CG solver tolerance.
    div_u_star = _div_backward(u_star, grid)
    rhs = rho * div_u_star / dt
    p_new = _solve_poisson_cg(rhs, grid)

    # Correct: u_new = u* - dt/rho * grad(p_new)
    grad_p_new = _grad_forward(p_new, grid)
    u_new = u_star - dt / rho * grad_p_new
    return u_new, p_new


def step_stokes_scan(
    v0: jnp.ndarray,
    p0: jnp.ndarray,
    grid: CartesianGrid,
    dt: float,
    n_steps: int,
    *,
    nu: float = 1.0,
    rho: float = 1.0,
    force: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Time-step the Stokes equation with ``jax.lax.scan`` (differentiable)."""
    def body(carry, _):
        v, p = carry
        return step_stokes(v, p, grid, dt, nu=nu, rho=rho, force=force), None
    (v_final, p_final), _ = jax.lax.scan(body, (v0, p0), xs=None, length=n_steps)
    return v_final, p_final
