"""Differentiable heat-diffusion solver on a Cartesian FVM grid (2D & 3D).

Solves the transient heat equation

    ρ c_p ∂T/∂t = ∇·(k ∇T) + Q

or, in terms of the thermal diffusivity ``α = k / (ρ c_p)``,

    ∂T/∂t = ∇·(α ∇T) + Q / (ρ c_p)

This is the AM *"process → thermal history"* stage: a laser / arc heat
source ``Q(x, t)`` deposits energy into the powder bed, the temperature
field ``T(x, t)`` evolves, and (downstream) the cooling rate drives grain
growth via the phase-field module.

Design
------
* The state is a plain ``(nx, ny[, nz], 1)`` cell-centred temperature field,
  identical in layout to the other FVM solvers, so it threads through
  ``jax.jit`` / ``jax.grad`` / ``jax.vmap`` unchanged.
* The Laplacian ``∇·(α ∇T)`` is discretised with second-order central
  differences via ghost cells; both a uniform ``alpha`` (scalar) and a
  spatially-varying ``alpha`` field are supported.
* Time integration uses SSP-RK2 (Heun), advanced through
  ``jax.lax.scan`` (:func:`step_thermal_scan`) for end-to-end
  reverse-mode-AD compatibility.
* Boundary conditions are selected per face: ``"dirichlet"`` (fixed
  temperature), ``"neumann"`` (insulated / prescribed flux), or ``"robin"``
  (convective cooling ``-k ∂T/∂n = h (T − T_inf)``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import jax
import jax.numpy as jnp

from diffmech.methods.fvm.grid import CartesianGrid


# ---------------------------------------------------------------------------
# Thermal configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ThermalConfig:
    """Parameters of the heat-diffusion problem.

    Attributes
    ----------
    alpha : float or array
        Thermal diffusivity ``k/(ρ c_p)`` [m²/s]. May be a scalar (uniform
        material) or a ``(nx, ny[, nz])`` field (spatially-varying / AM
        heterogeneous material).
    rho_cp : float
        Volumetric heat capacity ``ρ c_p`` [J/(m³·K)], used to convert a
        volumetric source ``Q`` [W/m³] into a temperature rate.
    bc : str
        Boundary condition type — one of ``"dirichlet"``, ``"neumann"``
        (insulated, zero normal gradient), or ``"robin"`` (convective).
    T_bc : float
        Boundary temperature for ``"dirichlet"`` [K].
    T_inf : float
        Ambient temperature for ``"robin"`` (and used as the far-field
        reference) [K].
    h_conv : float
        Convective heat-transfer coefficient for ``"robin"`` [W/(m²·K)].
    k_cond : float
        Thermal conductivity ``k`` [W/(m·K)], used by the Robin condition to
        form the dimensionless Biot number ``h dx / k``.
    """

    alpha: float | jnp.ndarray = 1.0
    rho_cp: float = 1.0
    bc: str = "neumann"
    T_bc: float = 300.0
    T_inf: float = 300.0
    h_conv: float = 0.0
    k_cond: float = 1.0


# ---------------------------------------------------------------------------
# Spatial operator  ∇·(α ∇T)
# ---------------------------------------------------------------------------
def _neighbour_along(field: jnp.ndarray, axis: int, dim: int) -> tuple[
    jnp.ndarray, jnp.ndarray
]:
    """Return ``(field_left, field_right)`` neighbours along ``axis``.

    Uses periodic ``jnp.roll``. ``field`` is the interior (un-padded) array of
    shape ``(nx, ny[, nz], ...)``.
    """
    f_left = jnp.roll(field, 1, axis=axis)
    f_right = jnp.roll(field, -1, axis=axis)
    return f_left, f_right


def diffusion_rhs(
    T: jnp.ndarray,
    grid: CartesianGrid,
    alpha: float | jnp.ndarray,
    *,
    bc: str = "neumann",
    T_bc: float = 300.0,
    T_inf: float = 300.0,
    h_conv: float = 0.0,
    k_cond: float = 1.0,
) -> jnp.ndarray:
    """Compute ``∇·(α ∇T)`` with the requested boundary condition.

    The Laplacian uses central differences; ``alpha`` may be a scalar or a
    spatially-varying field of shape ``(nx, ny[, nz])`` (no trailing axis).
    """
    cell_sizes = [grid.dx, grid.dy] if grid.dim == 2 else [grid.dx, grid.dy, grid.dz]
    alpha_arr = jnp.asarray(alpha, dtype=T.dtype)
    # Broadcast alpha to the field's spatial shape if it is a field.
    if alpha_arr.ndim > 0:
        alpha_int = jnp.expand_dims(alpha_arr, -1)  # (nx, ny[, nz], 1)
    else:
        alpha_int = alpha_arr

    if bc == "periodic":
        lap = jnp.zeros_like(T)
        for d, dx in enumerate(cell_sizes):
            f_left, f_right = _neighbour_along(T, d, grid.dim)
            lap = lap + (f_right - 2.0 * T + f_left) / (dx * dx)
        return alpha_int * lap

    # ---- ghost-cell BCs --------------------------------------------------
    ng = 1
    pad_spec = [(ng, ng)] * grid.dim + [(0, 0)] * (T.ndim - grid.dim)

    if bc == "neumann":
        T_pad = jnp.pad(T, pad_spec, mode="edge")
    elif bc == "dirichlet":
        # Mirror (ghost) cell so the *wall face* temperature equals ``T_bc``.
        # For a cell-centred grid the wall sits halfway between the ghost and
        # the first interior cell, hence
        #     T_face = (T_ghost + T_int) / 2 = T_bc  =>  T_ghost = 2 T_bc - T_int.
        # This is the standard "mirror" Dirichlet and pins the wall exactly,
        # unlike padding with a constant (which only sets the ghost cell itself
        # and leaves the face at (T_bc + T_int)/2).
        T_pad = jnp.pad(T, pad_spec, mode="edge")

        def _set_dirichlet_face(arr, axis, side):
            sl_g = [slice(None)] * T_pad.ndim
            sl_i = [slice(None)] * T_pad.ndim
            if side == "lo":
                sl_g[axis] = 0
                sl_i[axis] = ng
            else:
                sl_g[axis] = -1
                sl_i[axis] = -ng - 1
            interior = arr[tuple(sl_i)]
            ghost = 2.0 * T_bc - interior
            return arr.at[tuple(sl_g)].set(ghost)

        for d in range(grid.dim):
            T_pad = _set_dirichlet_face(T_pad, d, "lo")
            T_pad = _set_dirichlet_face(T_pad, d, "hi")
    elif bc == "robin":
        # Ghost-cell Robin: T_ghost = (2 T_inf + (2 dx k / h - 1) T_int) /
        #                        (2 dx k / h + 1)   for the convective face.
        # When h == 0 this reduces to a Neumann (insulated) wall.
        biot = jnp.where(h_conv > 1e-30, h_conv * grid.dx / k_cond, 0.0)  # scalar
        # Use edge padding as the base, then overwrite boundary ghost cells.
        T_pad = jnp.pad(T, pad_spec, mode="edge")
        # Convective ghost value (per face, identical on all faces for uniform h):
        #   T_g = (2 T_inf + (2/Bi - 1) T_int) / (2/Bi + 1),  Bi -> 0 => T_g = T_int (Neumann)
        inv = jnp.where(biot > 1e-30, 1.0 / biot, jnp.inf)
        coeff = jnp.where(biot > 1e-30, (2.0 * inv - 1.0) / (2.0 * inv + 1.0), 1.0)
        # Interior cell adjacent to each face.
        def _set_face(arr, axis, side):
            sl_g = [slice(None)] * T_pad.ndim
            sl_i = [slice(None)] * T_pad.ndim
            if side == "lo":
                sl_g[axis] = 0           # ghost at the low face
                sl_i[axis] = ng          # first interior cell
            else:
                sl_g[axis] = -1          # ghost at the high face
                sl_i[axis] = -ng - 1     # last interior cell
            interior = arr[tuple(sl_i)]
            ghost = coeff * interior + (1.0 - coeff) * T_inf
            return arr.at[tuple(sl_g)].set(ghost)
        for d in range(grid.dim):
            T_pad = _set_face(T_pad, d, "lo")
            T_pad = _set_face(T_pad, d, "hi")
    else:
        raise ValueError(f"unknown thermal bc {bc!r}")

    # Central differences on the padded field (interior slice).
    lap = jnp.zeros_like(T)
    for d, dx in enumerate(cell_sizes):
        sl_c = [slice(ng, -ng) if ax < grid.dim else slice(None) for ax in range(T_pad.ndim)]
        sl_l = [slice(ng, -ng) if ax < grid.dim else slice(None) for ax in range(T_pad.ndim)]
        sl_r = [slice(ng, -ng) if ax < grid.dim else slice(None) for ax in range(T_pad.ndim)]
        sl_l[d] = slice(ng - 1, -ng - 1)
        sl_r[d] = slice(ng + 1, None)
        f_left = T_pad[tuple(sl_l)]
        f_right = T_pad[tuple(sl_r)]
        lap = lap + (f_right - 2.0 * T + f_left) / (dx * dx)
    return alpha_int * lap


# ---------------------------------------------------------------------------
# Source term
# ---------------------------------------------------------------------------
def _eval_source(
    source, grid: CartesianGrid, t: float
) -> jnp.ndarray:
    """Resolve a volumetric source ``Q`` [W/m³] to a field of shape matching T.

    ``source`` may be:
      * ``None``  → zero source,
      * a scalar / jax array  → uniform source,
      * an array of shape ``(nx, ny[, nz])`` → spatially-varying source,
      * a callable ``Q(x_centers, t) -> (nx, ny[, nz])`` field.
    """
    if source is None:
        return jnp.zeros((grid.n_cells,))
    if callable(source) and not isinstance(source, jnp.ndarray):
        return jnp.asarray(source(grid.cell_centers, t)).reshape(-1)
    arr = jnp.asarray(source)
    if arr.ndim == 0:
        # NOTE: do not call ``float(arr)`` here — inside ``jax.lax.scan`` even
        # closed-over Python scalars are abstractified into tracers, and
        # ``float`` would raise ``ConcretizationTypeError``. ``jnp.full``
        # accepts a 0-d tracer as the fill value.
        return jnp.full((grid.n_cells,), arr)
    return arr.reshape(-1)


# ---------------------------------------------------------------------------
# Time stepping
# ---------------------------------------------------------------------------
def step_thermal(
    T: jnp.ndarray,
    grid: CartesianGrid,
    dt: float,
    cfg: ThermalConfig | None = None,
    *,
    alpha: float | jnp.ndarray = 1.0,
    rho_cp: float = 1.0,
    source=None,
    bc: str = "neumann",
    T_bc: float = 300.0,
    T_inf: float = 300.0,
    h_conv: float = 0.0,
    k_cond: float = 1.0,
    t: float = 0.0,
) -> jnp.ndarray:
    """Advance the heat equation by one SSP-RK2 (Heun) step.

    Parameters
    ----------
    T : (nx, ny[, nz], 1) array
        Cell-centred temperature field.
    grid : CartesianGrid
    dt : float
        Time step (use :func:`diffusion_dt` to pick a stable one).
    cfg : ThermalConfig, optional
        Packed configuration; if given its fields override the keyword args.
    source : scalar / array / callable, optional
        Volumetric heat source ``Q`` [W/m³].
    """
    if cfg is not None:
        alpha = cfg.alpha
        rho_cp = cfg.rho_cp
        bc = cfg.bc
        T_bc = cfg.T_bc
        T_inf = cfg.T_inf
        h_conv = cfg.h_conv
        k_cond = cfg.k_cond

    def rhs(state, t_eval):
        diff = diffusion_rhs(
            state, grid, alpha, bc=bc, T_bc=T_bc,
            T_inf=T_inf, h_conv=h_conv, k_cond=k_cond,
        )
        Q = _eval_source(source, grid, t_eval)
        # Q has shape (n_cells,); reshape to the field's spatial shape (+ trailing 1).
        if grid.dim == 2:
            Qf = Q.reshape(grid.nx, grid.ny, 1)
        else:
            Qf = Q.reshape(grid.nx, grid.ny, grid.nz, 1)
        return diff + Qf / rho_cp

    k1 = rhs(T, t)
    T1 = T + dt * k1
    k2 = rhs(T1, t + dt)
    return T + 0.5 * dt * (k1 + k2)


def step_thermal_scan(
    T0: jnp.ndarray,
    grid: CartesianGrid,
    dt: float,
    n_steps: int,
    cfg: ThermalConfig | None = None,
    *,
    alpha: float | jnp.ndarray = 1.0,
    rho_cp: float = 1.0,
    source=None,
    bc: str = "neumann",
    T_bc: float = 300.0,
    T_inf: float = 300.0,
    h_conv: float = 0.0,
    k_cond: float = 1.0,
    return_history: bool = False,
):
    """Time-step the heat equation with ``jax.lax.scan`` (differentiable).

    The simulation clock ``t`` is advanced by ``dt`` each step and forwarded
    to :func:`step_thermal` so that *time-dependent* sources (e.g. a moving
    Gaussian laser) are evaluated at the correct time. This is essential for
    AM thermal-history simulations where the beam position depends on ``t``.

    If ``return_history`` is True, returns the full time-history stacked along
    a new leading axis (shape ``(n_steps+1, nx, ny[, nz], 1)``); otherwise
    returns only the final state.
    """
    kw = dict(alpha=alpha, rho_cp=rho_cp, source=source, bc=bc,
              T_bc=T_bc, T_inf=T_inf, h_conv=h_conv, k_cond=k_cond)

    # Carry = (T, t): the temperature field and the current physical time.
    def body(carry, _):
        state, t = carry
        nxt = step_thermal(state, grid, dt, cfg, t=t, **kw)
        return (nxt, t + dt), nxt

    init = (T0, jnp.asarray(0.0, dtype=T0.dtype))
    (final, _), history = jax.lax.scan(body, init, xs=None, length=n_steps)
    if return_history:
        # Prepend the initial state.
        return jnp.concatenate([T0[None], history], axis=0)
    return final


# ---------------------------------------------------------------------------
# Stability limit (explicit diffusion CFL)
# ---------------------------------------------------------------------------
def diffusion_dt(
    grid: CartesianGrid,
    alpha: float | jnp.ndarray = 1.0,
    cfl: float = 0.5,
) -> float:
    """Explicit diffusion stability limit ``dt <= cfl * dx² / (2 * dim * α_max)``.

    For spatially-varying ``alpha`` the maximum diffusivity is used (so the
    bound remains safe everywhere).
    """
    dim = grid.dim
    cell_sizes = [grid.dx, grid.dy] if dim == 2 else [grid.dx, grid.dy, grid.dz]
    dx_min = min(float(d) for d in cell_sizes)
    alpha_arr = jnp.asarray(alpha)
    a_max = float(jnp.max(alpha_arr)) if alpha_arr.ndim > 0 else float(alpha_arr)
    a_max = max(a_max, 1e-30)
    return float(cfl * dx_min * dx_min / (2.0 * dim * a_max))


# ---------------------------------------------------------------------------
# AM heat-source models
# ---------------------------------------------------------------------------
def gaussian_heat_source(
    power: float,
    absorption: float,
    beam_radius: float,
    center: Sequence[float],
    *,
    depth: float | None = None,
) -> Callable[[jnp.ndarray, float], jnp.ndarray]:
    """A Gaussian (moving) laser heat source ``Q(x, t)`` [W/m³].

    The beam is a 2D Gaussian in the (x, y) plane centred at ``center``; for
    3D an optional exponential depth attenuation ``exp(-z/depth)`` models
    penetration.

    The returned callable accepts ``(cell_centers, t)`` and returns a flat
    ``(n_cells,)`` source field — this signature is what
    :func:`step_thermal` expects from a callable ``source``.

    Parameters
    ----------
    power : float
        Laser power [W].
    absorption : float
        Absorption efficiency (0..1).
    beam_radius : float
        1/e² beam radius ``r_b`` [m].
    center : (x, y) or (x, y, z)
        Beam centre (the user can move it by closing over a time-varying
        centre in a custom source).
    depth : float, optional
        Penetration depth for 3D sources [m].
    """
    center = jnp.asarray(center, dtype=jnp.float64)
    r2 = beam_radius * beam_radius
    absorbed = absorption * power
    # Peak intensity of a 1/e^2-radius Gaussian beam whose total (area-integrated)
    # power equals ``absorbed``:  I_0 = 2 P / (pi r_b^2), so that
    #   ∫∫ I_0 exp(-2 r^2 / r_b^2) dA = P.
    pi_r2 = math.pi * r2

    def source_fn(cell_centers: jnp.ndarray, t: float) -> jnp.ndarray:
        xy = cell_centers[..., :2] - center[:2]
        rsq = jnp.sum(xy * xy, axis=-1)
        q2d = (2.0 * absorbed) / pi_r2 * jnp.exp(-2.0 * rsq / r2)
        if depth is not None and cell_centers.shape[-1] == 3:
            q2d = q2d * jnp.exp(-cell_centers[..., 2] / depth) / depth
        return q2d

    return source_fn


def moving_gaussian_source(
    power: float,
    absorption: float,
    beam_radius: float,
    start: Sequence[float],
    velocity: Sequence[float],
    *,
    depth: float | None = None,
):
    """A Gaussian laser whose centre moves with constant ``velocity``.

    ``center(t) = start + velocity * t``.
    """
    start = jnp.asarray(start, dtype=jnp.float64)
    velocity = jnp.asarray(velocity, dtype=jnp.float64)

    def center_fn(t: float) -> jnp.ndarray:
        return start + velocity * t

    def source_fn(cell_centers: jnp.ndarray, t: float) -> jnp.ndarray:
        center = center_fn(t)
        xy = cell_centers[..., :2] - center[:2]
        rsq = jnp.sum(xy * xy, axis=-1)
        r2 = beam_radius * beam_radius
        absorbed = absorption * power
        # I_0 = 2 P / (pi r_b^2) — see gaussian_heat_source for the derivation.
        q2d = (2.0 * absorbed) / (math.pi * r2) * jnp.exp(-2.0 * rsq / r2)
        if depth is not None and cell_centers.shape[-1] == 3:
            q2d = q2d * jnp.exp(-cell_centers[..., 2] / depth) / depth
        return q2d

    return source_fn
