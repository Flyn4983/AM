"""Differentiable Material Point Method (MLS-MPM) for solids & fluids (2D & 3D).

Implements the *Moving Least Squares* MPM (MLS-MPM) formulation of Hu et al.
(2018, "A Moving Least Squares Material Point Method with Displacement
Discontinuity").  MLS-MPM combines the particle-in-cell (PIC) and
Affine-Particle-In-Cell (APIC) ideas with an MLS basis to give smooth,
low-dissipation transfer between the Eulerian background grid and the
Lagrangian particles, while remaining straightforward to differentiate.

The whole algorithm is written with ``jax.vmap`` / ``jax.lax.scan`` so that
``jax.grad`` flows end-to-end — this is the central feature of the
``diffmech`` suite.

Workflow per step (MLS-MPM, USL formulation):

1. **P2G** — transfer particle mass, momentum and (for APIC) affine
   velocity from particles to the regular background grid.
2. **Grid solve** — compute grid velocities (apply boundary conditions,
   external forces, optionally an equation of state for fluids).
3. **G2P** — interpolate grid velocities back to the particles and update
   particle position / affine state.
4. **(Optional) CFL / dt check.**

The implementation supports both:

- pure *PIC* transfers (cheaper, more dissipative),
- *APIC* transfers (affine momentum, much less dissipative).

Material models supplied out-of-the-box:

- Neo-Hookean hyperelastic solid (corotational Cauchy stress).
- Simple equation-of-state fluid (linear ``p = K (J - 1)``).

Dimensionality is implicit in the particle coordinate arrays: pass 2D
arrays for a 2D simulation, 3D arrays for 3D.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Quadratic B-spline weights (used for P2G / G2P).
# ---------------------------------------------------------------------------
def _bspline_weights(xp: jnp.ndarray, origin: jnp.ndarray, dx: float
                     ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Quadratic B-spline weights for a particle at ``xp`` on a grid.

    Parameters
    ----------
    xp : (N,) array
        Particle coordinates along one axis.
    origin, dx : scalar
        Grid origin and spacing along this axis.

    Returns
    -------
    base : (N,) int array
        Lowest participating node index per particle.
    weights : (N, 3) array
        Weights on the three consecutive nodes ``[base, base+1, base+2]``.
    fx : (N,) array
        Fractional position of the particle within its stencil cell.
    """
    # Particle coordinate in grid-index space
    x = (xp - origin) / dx
    # base = floor(x - 0.5)
    base_f = jnp.floor(x - 0.5)
    base = base_f.astype(jnp.int32)
    fx = x - base_f
    # Quadratic B-spline (Muller et al.).  ``fx`` may be scalar or (N,);
    # we stack along the last axis so the result is (N, 3) (or (3,) for a
    # scalar input).
    w0 = 0.5 * (1.5 - fx) ** 2
    w1 = 0.75 - (fx - 1.0) ** 2
    w2 = 0.5 * (fx - 0.5) ** 2
    weights = jnp.stack([w0, w1, w2], axis=-1)
    return base, weights, fx


def _bspline_weight_grad(fx: jnp.ndarray, dx: float) -> jnp.ndarray:
    """Gradient of the quadratic B-spline weights w.r.t. the physical coord.

    Returns ``(N, 3)`` — ``dW/dx`` for each of the three stencil nodes.
    """
    # dW/dx = (dW/dfx) / dx
    dw0 = -(1.5 - fx) / dx
    dw1 = -2.0 * (fx - 1.0) / dx
    dw2 = (fx - 0.5) / dx
    return jnp.stack([dw0, dw1, dw2], axis=-1)


def _stencil_table(dim: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return the flat stencil indices and per-axis offsets for the B-spline.

    For ``dim == 2`` returns a (9,) array of flat stencil ids (0..8) and a
    (9, dim) array of per-axis offsets in {0, 1, 2}^dim.  For ``dim == 3``
    the tables have length 27.

    The flat ordering is C-style: the last axis varies fastest, so for 2D
    the (i, j) stencil cell maps to ``k = i * 3 + j``.
    """
    if dim == 2:
        stencil = jnp.stack(jnp.meshgrid(jnp.arange(3), jnp.arange(3),
                                         indexing="ij"),
                            axis=-1).reshape(9, 2)
    else:
        stencil = jnp.stack(jnp.meshgrid(jnp.arange(3), jnp.arange(3),
                                         jnp.arange(3), indexing="ij"),
                            axis=-1).reshape(27, 3)
    k = jnp.arange(stencil.shape[0])
    return k, stencil


def _scatter_to_grid_nd(
    particle_value: jnp.ndarray,
    particle_pos: jnp.ndarray,
    grid_shape: tuple[int, ...],
    origin: jnp.ndarray,
    dx: float,
    dim: int,
) -> jnp.ndarray:
    """Accumulate a per-particle scalar/vector onto the grid via B-splines.

    ``particle_value`` has shape ``(N,)`` or ``(N, k)``; the result has
    shape ``grid_shape`` (for a scalar) or ``grid_shape + (k,)``.
    """
    N = particle_pos.shape[0]
    # Per-axis base indices and weights.
    bases = []
    weights = []
    for d in range(dim):
        b, w, _ = _bspline_weights(particle_pos[:, d], origin[d], dx)
        bases.append(b)         # (N,)
        weights.append(w)        # (N, 3)

    _, stencil = _stencil_table(dim)              # (S, dim)
    S = stencil.shape[0]                          # 9 or 27

    # Per-particle per-stencil-node flat grid index and B-spline weight.
    # node_axis_idx[n, s, d] = bases[d][n] + stencil[s, d]
    # weight[n, s] = prod_d weights[d][n, stencil[s, d]]
    flat_idx = jnp.zeros((N, S), dtype=jnp.int32)
    w_flat = jnp.ones((N, S), dtype=particle_value.dtype)
    for d in range(dim):
        node_idx_d = bases[d][:, None] + stencil[:, d][None, :]   # (N, S)
        node_idx_d = jnp.mod(node_idx_d, grid_shape[d])
        # Gather the weight for this axis at each stencil node.
        w_d = weights[d]                                            # (N, 3)
        w_gathered = w_d[:, stencil[:, d]]                         # (N, S)
        w_flat = w_flat * w_gathered
        # Accumulate into the flat index.  We multiply by the stride of axis d.
        # For 2D: flat = i * ny + j;  for 3D: flat = (i * ny + j) * nz + k.
        stride = 1
        for dd in range(dim - 1, d, -1):
            stride *= grid_shape[dd]
        flat_idx = flat_idx + node_idx_d * stride

    flat_idx_r = flat_idx.reshape(-1)                              # (N*S,)
    if particle_value.ndim == 1:
        contributions = particle_value[:, None] * w_flat            # (N, S)
        contrib_r = contributions.reshape(-1)                      # (N*S,)
        grid = jnp.zeros((int(np.prod(grid_shape)),), dtype=particle_value.dtype)
        grid = grid.at[flat_idx_r].add(contrib_r)
        return grid.reshape(*grid_shape)
    kdim = particle_value.shape[1]
    contributions = particle_value[:, None, :] * w_flat[..., None]  # (N, S, kdim)
    contrib_r = contributions.reshape(-1, kdim)                     # (N*S, kdim)
    grid = jnp.zeros((int(np.prod(grid_shape)), kdim), dtype=particle_value.dtype)
    grid = grid.at[flat_idx_r].add(contrib_r)
    return grid.reshape(*grid_shape, kdim)


def _gather_from_grid_nd(
    grid_field: jnp.ndarray,
    particle_pos: jnp.ndarray,
    origin: jnp.ndarray,
    dx: float,
    dim: int,
) -> jnp.ndarray:
    """Bilinear/B-spline gather of a grid field onto particles.

    ``grid_field`` has shape ``grid_shape + (...)`` (trailing component
    axis optional). Returns ``(N, ...)``.
    """
    N = particle_pos.shape[0]
    bases = []
    weights = []
    for d in range(dim):
        b, w, _ = _bspline_weights(particle_pos[:, d], origin[d], dx)
        bases.append(b)
        weights.append(w)
    grid_shape = grid_field.shape[:dim]
    _, stencil = _stencil_table(dim)
    S = stencil.shape[0]
    flat_idx = jnp.zeros((N, S), dtype=jnp.int32)
    w_flat = jnp.ones((N, S), dtype=grid_field.dtype)
    for d in range(dim):
        node_idx_d = bases[d][:, None] + stencil[:, d][None, :]
        node_idx_d = jnp.mod(node_idx_d, grid_shape[d])
        w_d = weights[d]
        w_gathered = w_d[:, stencil[:, d]]
        w_flat = w_flat * w_gathered
        stride = 1
        for dd in range(dim - 1, d, -1):
            stride *= grid_shape[dd]
        flat_idx = flat_idx + node_idx_d * stride
    flat_idx_r = flat_idx.reshape(-1)                              # (N*S,)
    if grid_field.ndim == dim:
        vals = grid_field.reshape(-1)[flat_idx_r].reshape(N, S)
        return jnp.sum(vals * w_flat, axis=1)
    kdim = grid_field.shape[-1]
    vals = grid_field.reshape(-1, kdim)[flat_idx_r].reshape(N, S, kdim)
    return jnp.sum(vals * w_flat[..., None], axis=1)


def _gather_stencil_displacements(
    particle_pos: jnp.ndarray,
    origin: jnp.ndarray,
    dx: float,
    dim: int,
) -> jnp.ndarray:
    """Per-particle per-stencil-node displacement ``x_node - x_p``.

    Returns an array of shape ``(N, S, dim)`` where ``S`` is the stencil
    size (9 in 2D, 27 in 3D).  The displacement is computed in physical
    units and uses the *same* B-spline stencil as the scatter/gather
    functions, so it is consistent with the weights.
    """
    N = particle_pos.shape[0]
    fx_list = []
    for d in range(dim):
        _, _, fx = _bspline_weights(particle_pos[:, d], origin[d], dx)
        fx_list.append(fx)
    _, stencil = _stencil_table(dim)
    S = stencil.shape[0]
    # Node coordinate (in grid units) relative to the particle for axis d:
    #   node_grid_x = base + stencil_offset  (integer)
    #   particle_grid_x = base + 0.5 + fx     (the particle sits at this frac)
    # So node - particle (in grid units) = stencil_offset - 0.5 - fx
    # Multiply by dx to get physical units.
    disp = jnp.zeros((N, S, dim), dtype=particle_pos.dtype)
    for d in range(dim):
        rel = (stencil[:, d][None, :].astype(particle_pos.dtype)
               - 0.5 - fx_list[d][:, None]) * dx            # (N, S)
        disp = disp.at[..., d].set(rel)
    return disp


def _stencil_info(
    particle_pos: jnp.ndarray,
    origin: jnp.ndarray,
    dx: float,
    dim: int,
    grid_shape: tuple[int, ...],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute the per-particle stencil indices, weights, displacements and
    weight gradients.

    Returns
    -------
    flat_idx : (N, S) int array
        Flat (row-major) grid node index for each stencil node.
    w_flat : (N, S) array
        B-spline weight for each stencil node.
    disp : (N, S, dim) array
        ``x_node - x_p`` in physical units.
    grad_w : (N, S, dim) array
        ``∇w`` (gradient of the B-spline weight w.r.t. physical position)
        for each stencil node.
    """
    N = particle_pos.shape[0]
    _, stencil = _stencil_table(dim)
    S = stencil.shape[0]
    bases = []
    weights = []
    fx_list = []
    for d in range(dim):
        b, w, fx = _bspline_weights(particle_pos[:, d], origin[d], dx)
        bases.append(b)
        weights.append(w)
        fx_list.append(fx)
    # Flat index and combined weight.
    flat_idx = jnp.zeros((N, S), dtype=jnp.int32)
    w_flat = jnp.ones((N, S), dtype=particle_pos.dtype)
    for d in range(dim):
        node_idx_d = bases[d][:, None] + stencil[:, d][None, :]   # (N, S)
        node_idx_d = jnp.mod(node_idx_d, grid_shape[d])
        w_flat = w_flat * weights[d][:, stencil[:, d]]
        stride = 1
        for dd in range(dim - 1, d, -1):
            stride *= grid_shape[dd]
        flat_idx = flat_idx + node_idx_d * stride
    # Displacement x_node - x_p (physical units).
    disp = jnp.zeros((N, S, dim), dtype=particle_pos.dtype)
    for d in range(dim):
        rel = (stencil[:, d][None, :].astype(particle_pos.dtype)
               - 0.5 - fx_list[d][:, None]) * dx                # (N, S)
        disp = disp.at[..., d].set(rel)
    # Weight gradient ∇w (physical units, dW/dx).
    grad_w = jnp.zeros((N, S, dim), dtype=particle_pos.dtype)
    for d in range(dim):
        dw_d = _bspline_weight_grad(fx_list[d], dx)               # (N, 3)
        # For each stencil node we need dw_d at the stencil offset along d,
        # and the *value* of w along the other axes (because
        # ∇w = (dw_d / dx) * prod_{d'≠d} w_{d'}).
        dw_gathered = dw_d[:, stencil[:, d]]                    # (N, S)
        other_w = jnp.ones((N, S), dtype=particle_pos.dtype)
        for d2 in range(dim):
            if d2 == d:
                continue
            other_w = other_w * weights[d2][:, stencil[:, d2]]
        grad_w = grad_w.at[..., d].set(dw_gathered * other_w)
    return flat_idx, w_flat, disp, grad_w


def _scatter_stencil_to_grid(
    flat_idx: jnp.ndarray,           # (N, S) int
    w_flat: jnp.ndarray,             # (N, S)
    per_node_value: jnp.ndarray,     # (N, S, k) or (N, S)
    n_nodes: int,
    kdim: int | None,
    dtype,
) -> jnp.ndarray:
    """Accumulate per-particle per-stencil-node values onto the grid.

    ``per_node_value`` already includes the B-spline weight (caller multiplies
    by ``w_flat`` if needed).  The result has shape ``(n_nodes, kdim)`` or
    ``(n_nodes,)`` depending on whether ``kdim`` is provided.
    """
    N, S = flat_idx.shape
    flat_idx_r = flat_idx.reshape(-1)
    if kdim is None:
        contrib = per_node_value.reshape(-1)
        grid = jnp.zeros((n_nodes,), dtype=dtype)
        return grid.at[flat_idx_r].add(contrib)
    contrib = per_node_value.reshape(-1, kdim)
    grid = jnp.zeros((n_nodes, kdim), dtype=dtype)
    return grid.at[flat_idx_r].add(contrib)


# ---------------------------------------------------------------------------
# State and configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MPMConfig:
    """MLS-MPM simulation parameters.

    Attributes
    ----------
    grid_origin : (dim,) array
        Coordinates of grid node (0, 0[, 0]).
    grid_shape : tuple of int
        Number of grid nodes in each spatial direction (2 or 3 entries).
    dx : float
        Grid spacing (uniform in every direction).
    dt : float
        Time step (caller chooses, no CFL enforcement inside).
    youngs_modulus, poissons_ratio : float
        Solid elasticity parameters (used for the Neo-Hookean model).
    rho0 : float
        Reference density for the fluid EOS.
    bulk_modulus : float
        Bulk modulus for the fluid EOS (``K`` in ``p = K (J - 1)``).
    gravity : float
        Body force along the last spatial axis.
    transfer : {"apic", "pic"}
        Particle-grid momentum transfer scheme.
    """

    grid_origin: tuple
    grid_shape: tuple
    dx: float
    dt: float
    youngs_modulus: float = 1.0e5
    poissons_ratio: float = 0.3
    rho0: float = 1.0
    bulk_modulus: float = 1.0e5
    gravity: float = 0.0
    transfer: str = "apic"  # "apic" or "pic"
    material: str = "solid"  # "solid" (Neo-Hookean) or "fluid" (EOS)


# Register MPMConfig as a pytree so jax.jit can trace through it.
# Numeric fields become leaves; tuples/strings are static auxiliary data.
def _mpm_cfg_flatten(cfg: MPMConfig):
    children = (cfg.dx, cfg.dt, cfg.youngs_modulus, cfg.poissons_ratio,
                cfg.rho0, cfg.bulk_modulus, cfg.gravity)
    aux = (cfg.grid_origin, cfg.grid_shape, cfg.transfer, cfg.material)
    return children, aux


def _mpm_cfg_unflatten(aux, children):
    dx, dt, E, nu, rho0, K, g = children
    grid_origin, grid_shape, transfer, material = aux
    return MPMConfig(
        grid_origin=grid_origin, grid_shape=grid_shape, dx=dx, dt=dt,
        youngs_modulus=E, poissons_ratio=nu, rho0=rho0, bulk_modulus=K,
        gravity=g, transfer=transfer, material=material,
    )


jax.tree_util.register_pytree_node(MPMConfig, _mpm_cfg_flatten, _mpm_cfg_unflatten)


@dataclass(frozen=True)
class MPMState:
    """MPM particle state.

    Attributes
    ----------
    position : (N, dim) array
        Particle positions.
    velocity : (N, dim) array
        Particle velocities.
    affine : (N, dim, dim) array
        Per-particle affine velocity matrix ``C`` (APIC).  Zero when PIC
        is used (the value is then ignored).
    deformation : (N, dim, dim) array
        Per-particle deformation gradient ``F`` (history variable for
        solids).
    volume0 : (N,) array
        Initial particle volumes.
    mass : (N,) array
        Particle masses.
    """

    position: jax.Array
    velocity: jax.Array
    affine: jax.Array
    deformation: jax.Array
    volume0: jax.Array
    mass: jax.Array

    @property
    def n_particles(self) -> int:
        return int(self.position.shape[0])

    @property
    def dim(self) -> int:
        return int(self.position.shape[1])


def _mpm_flatten(s: MPMState):
    return (s.position, s.velocity, s.affine, s.deformation,
            s.volume0, s.mass), None


def _mpm_unflatten(_, children):
    pos, vel, aff, defm, v0, m = children
    return MPMState(pos, vel, aff, defm, v0, m)


jax.tree_util.register_pytree_node(MPMState, _mpm_flatten, _mpm_unflatten)


def make_mpm_state(
    position: jnp.ndarray,
    velocity: jnp.ndarray | None = None,
    *,
    volume: float | jnp.ndarray = 1.0,
    mass: float | jnp.ndarray = 1.0,
    dim: int | None = None,
) -> MPMState:
    """Convenience constructor.

    Initialises the affine term to zero and the deformation gradient to the
    identity.  ``volume`` may be a scalar (broadcast) or a per-particle array.
    """
    pos = jnp.asarray(position, dtype=jnp.float64)
    N, d = pos.shape
    if dim is None:
        dim = d
    vel = (jnp.zeros_like(pos) if velocity is None
           else jnp.asarray(velocity, dtype=jnp.float64))
    aff = jnp.zeros((N, dim, dim), dtype=jnp.float64)
    eye = jnp.eye(dim, dtype=jnp.float64)
    defm = jnp.broadcast_to(eye, (N, dim, dim)).copy()
    v0 = jnp.broadcast_to(jnp.asarray(volume, dtype=jnp.float64), (N,)).copy()
    m = jnp.broadcast_to(jnp.asarray(mass, dtype=jnp.float64), (N,)).copy()
    return MPMState(position=pos, velocity=vel, affine=aff,
                    deformation=defm, volume0=v0, mass=m)


# ---------------------------------------------------------------------------
# Constitutive models
# ---------------------------------------------------------------------------
def _neo_hookean_cauchy_stress(F: jnp.ndarray, E: float, nu: float
                                ) -> jnp.ndarray:
    """Cauchy stress for a Neo-Hookean solid (corotational form).

    ``F`` may be a single (dim, dim) matrix or a batched ``(N, dim, dim)``.
    """
    # Lamé parameters
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    # Use SVD-free polar decomposition via F F^T
    FFt = F @ F.swapaxes(-1, -2)
    # J = det(F)
    if F.ndim == 2:
        J = jnp.linalg.det(F)
        # deviatoric part:  ``sigma_dev = mu / J * (b - I)`` where ``b = F F^T``.
        I = jnp.eye(F.shape[-1])
        sigma = mu * (FFt - I) / J + lam * jnp.log(jnp.maximum(J, 1e-12)) * I
        return sigma
    # batched
    J = jnp.linalg.det(F)
    I = jnp.eye(F.shape[-1])
    sigma = mu * (FFt - I) / J[..., None, None] + \
        lam * jnp.log(jnp.maximum(J, 1e-12))[..., None, None] * I
    return sigma


def _fluid_pressure_stress(F: jnp.ndarray, K: float) -> jnp.ndarray:
    """Pressure-only Cauchy stress for a fluid with linear EOS ``p = K (J-1)``.

    Returns ``sigma = -p I`` (compressive pressure negative).
    """
    if F.ndim == 2:
        J = jnp.linalg.det(F)
        p = K * (J - 1.0)
        return -p * jnp.eye(F.shape[-1])
    J = jnp.linalg.det(F)
    p = K * (J - 1.0)
    return -p[..., None, None] * jnp.eye(F.shape[-1])


def _cauchy_stress(state: MPMState, cfg: MPMConfig) -> jnp.ndarray:
    """Compute per-particle Cauchy stress ``(N, dim, dim)``."""
    if cfg.material == "fluid":
        return _fluid_pressure_stress(state.deformation, cfg.bulk_modulus)
    return _neo_hookean_cauchy_stress(state.deformation,
                                       cfg.youngs_modulus,
                                       cfg.poissons_ratio)


# ---------------------------------------------------------------------------
# P2G / G2P transfers
# ---------------------------------------------------------------------------
def _p2g(state: MPMState, cfg: MPMConfig) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Particle-to-grid transfer.

    Returns
    -------
    mass_grid : (nx, ny[, nz]) array
    momentum_grid : (nx, ny[, nz], dim) array
    """
    dim = state.dim
    origin = jnp.asarray(cfg.grid_origin, dtype=jnp.float64)
    grid_shape = tuple(cfg.grid_shape)
    n_nodes = int(np.prod(grid_shape))
    # Mass transfer (scalar per particle).
    mass_grid = _scatter_to_grid_nd(
        state.mass, state.position, grid_shape, origin, cfg.dx, dim,
    )
    # Momentum transfer.
    if cfg.transfer == "apic":
        # APIC: v_node = Σ_p w_ip (v_p + C_p @ (x_node - x_p))
        # momentum_node = Σ_p m_p w_ip (v_p + C_p @ d_ip)
        flat_idx, w_flat, disp, _ = _stencil_info(
            state.position, origin, cfg.dx, dim, grid_shape,
        )
        # affine_contrib[n, s, :] = C[n] @ disp[n, s, :]
        affine_contrib = jnp.einsum("nij,nsj->nsi", state.affine, disp)
        v_at_node = state.velocity[:, None, :] + affine_contrib      # (N, S, dim)
        # momentum contribution per node: m * v_at_node * w
        mom_per_node = (state.mass[:, None, None] *
                        v_at_node * w_flat[..., None])               # (N, S, dim)
        mom_flat = _scatter_stencil_to_grid(
            flat_idx, w_flat, mom_per_node, n_nodes, dim, mass_grid.dtype,
        )
        momentum_grid = mom_flat.reshape(*grid_shape, dim)
    else:  # PIC
        momentum_grid = _scatter_to_grid_nd(
            state.mass[:, None] * state.velocity, state.position,
            grid_shape, origin, cfg.dx, dim,
        )
    return mass_grid, momentum_grid


def _grid_velocity(mass_grid: jnp.ndarray, momentum_grid: jnp.ndarray
                   ) -> jnp.ndarray:
    """Compute node velocities by dividing momentum by mass (with masking)."""
    eps = 1e-30
    if mass_grid.ndim == momentum_grid.ndim - 1:
        return momentum_grid / jnp.maximum(mass_grid[..., None], eps)
    return momentum_grid / jnp.maximum(mass_grid, eps)


def _apply_grid_forces(state: MPMState, cfg: MPMConfig,
                       v_grid: jnp.ndarray) -> jnp.ndarray:
    """Apply internal (stress) and external (gravity) forces on the grid.

    Uses the unified ``_stencil_info`` helper so the same code path works
    for both 2D and 3D.  The internal force on a grid node from particle
    *p* is::

        f_node = -V_p * sigma_p @ grad_w_ip

    where ``grad_w_ip`` is the gradient of the B-spline weight.  The
    velocity update is ``dv = dt * f / m_node``.
    """
    dim = state.dim
    origin = jnp.asarray(cfg.grid_origin, dtype=jnp.float64)
    grid_shape = tuple(cfg.grid_shape)
    n_nodes = int(np.prod(grid_shape))
    sigma = _cauchy_stress(state, cfg)  # (N, dim, dim)
    # Deformed volume per particle.
    J = jnp.linalg.det(state.deformation)  # (N,)
    vol = state.volume0 * J  # (N,)
    # Unified stencil: flat_idx (N,S), w_flat (N,S), disp (N,S,dim),
    # grad_w (N,S,dim).
    flat_idx, w_flat, _, grad_w = _stencil_info(
        state.position, origin, cfg.dx, dim, grid_shape,
    )
    # Internal force per stencil node:
    #   f[n, s, i] = -V_p[n] * sum_j sigma[n, i, j] * grad_w[n, s, j]
    force_per_node = -vol[:, None, None] * jnp.einsum(
        "nij,nsj->nsi", sigma, grad_w,
    )  # (N, S, dim)
    # Scatter forces onto the grid (w_flat is unused by the helper).
    force_grid = _scatter_stencil_to_grid(
        flat_idx, w_flat, force_per_node, n_nodes, dim, v_grid.dtype,
    )  # (n_nodes, dim)
    force_grid = force_grid.reshape(*grid_shape, dim)
    # Mass grid for dividing force by node mass.
    mass_grid = _scatter_to_grid_nd(
        state.mass, state.position, grid_shape, origin, cfg.dx, dim,
    )
    eps = 1e-30
    dv = cfg.dt * force_grid / jnp.maximum(mass_grid[..., None], eps)
    v_grid = v_grid + dv
    # External gravity along last axis.
    v_grid = v_grid.at[..., -1].add(cfg.dt * cfg.gravity)
    return v_grid


def _g2p(state: MPMState, cfg: MPMConfig, v_grid: jnp.ndarray) -> MPMState:
    """Grid-to-particle transfer.

    Updates particle velocity, position, affine term and deformation
    gradient.  Uses the unified ``_stencil_info`` helper so the same code
    path works for both 2D and 3D.
    """
    dim = state.dim
    origin = jnp.asarray(cfg.grid_origin, dtype=jnp.float64)
    grid_shape = tuple(cfg.grid_shape)
    n_nodes = int(np.prod(grid_shape))
    # Unified stencil: flat_idx (N,S), w_flat (N,S), disp (N,S,dim), grad_w.
    flat_idx, w_flat, disp, _ = _stencil_info(
        state.position, origin, cfg.dx, dim, grid_shape,
    )
    N = state.n_particles
    S = flat_idx.shape[1]
    # Gather grid velocity at each stencil node: (N, S, dim).
    flat_idx_r = flat_idx.reshape(-1)  # (N*S,)
    v_nodes = v_grid.reshape(-1, dim)[flat_idx_r].reshape(N, S, dim)
    # Particle velocity: v_p = Σ_s w[s] * v_node[s].
    v_p = jnp.sum(w_flat[..., None] * v_nodes, axis=1)  # (N, dim)
    if cfg.transfer == "apic":
        # APIC affine: C = (4 / dx^2) * Σ_s w[s] * v_node[s] * disp[s]^T
        # C[n, i, j] = (4/dx^2) * Σ_s w[n,s] * v_nodes[n,s,i] * disp[n,s,j]
        C = (4.0 / (cfg.dx ** 2)) * jnp.einsum(
            "ns,nsi,nsj->nij", w_flat, v_nodes, disp,
        )  # (N, dim, dim)
    else:
        # PIC: affine term is zero (no history carried).
        C = jnp.zeros_like(state.affine)
    # Update deformation gradient: F_new = (I + dt * grad_v) @ F.
    # In APIC the affine term C approximates the velocity gradient.
    I = jnp.eye(dim, dtype=jnp.float64)
    F_new = (I + cfg.dt * C) @ state.deformation
    # Update position with new velocity.
    x_new = state.position + cfg.dt * v_p
    return MPMState(
        position=x_new,
        velocity=v_p,
        affine=C,
        deformation=F_new,
        volume0=state.volume0,
        mass=state.mass,
    )


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------
def _apply_velocity_bc(v_grid: jnp.ndarray, cfg: MPMConfig,
                        bc: str | Callable) -> jnp.ndarray:
    """Apply velocity boundary conditions on the grid.

    Currently supports:

    - ``"slip"`` — free-slip (zero normal velocity) at all domain boundaries.
    - ``"noslip"`` — zero velocity at all domain boundaries.
    - callable ``bc(v_grid, cfg)`` returning the modified grid.
    """
    if callable(bc):
        return bc(v_grid, cfg)
    dim = v_grid.ndim - 1
    if bc == "slip":
        # Free-slip: zero the normal velocity component on each boundary
        # face, leave tangential components untouched.
        out = v_grid
        # x = 0 / x = L  ->  v_x = 0
        out = out.at[0, ..., 0].set(0.0)
        out = out.at[-1, ..., 0].set(0.0)
        if dim >= 2:
            out = out.at[:, 0, ..., 1].set(0.0)
            out = out.at[:, -1, ..., 1].set(0.0)
        if dim == 3:
            out = out.at[:, :, 0, ..., 2].set(0.0)
            out = out.at[:, :, -1, ..., 2].set(0.0)
        return out
    if bc == "noslip":
        # No-slip: zero every velocity component on each boundary face.
        out = v_grid
        out = out.at[0, ...].set(0.0)
        out = out.at[-1, ...].set(0.0)
        if dim >= 2:
            out = out.at[:, 0, ...].set(0.0)
            out = out.at[:, -1, ...].set(0.0)
        if dim == 3:
            out = out.at[:, :, 0, ...].set(0.0)
            out = out.at[:, :, -1, ...].set(0.0)
        return out
    if bc == "periodic" or bc is None:
        return v_grid
    raise ValueError(f"unknown bc {bc!r}")


# ---------------------------------------------------------------------------
# Step & scan
# ---------------------------------------------------------------------------
def step_mpm(
    state: MPMState,
    cfg: MPMConfig,
    *,
    bc: str | Callable = "slip",
) -> MPMState:
    """Advance the MLS-MPM system by one time step.

    Parameters
    ----------
    state : MPMState
    cfg : MPMConfig
        Must contain ``grid_shape``, ``grid_origin``, ``dx``, ``dt`` and
        material parameters.
    bc : str or callable
        Velocity boundary condition on the grid.  See
        :func:`_apply_velocity_bc`.
    """
    mass_grid, momentum_grid = _p2g(state, cfg)
    v_grid = _grid_velocity(mass_grid, momentum_grid)
    v_grid = _apply_grid_forces(state, cfg, v_grid)
    v_grid = _apply_velocity_bc(v_grid, cfg, bc)
    new_state = _g2p(state, cfg, v_grid)
    return new_state


def step_mpm_scan(
    state0: MPMState,
    cfg: MPMConfig,
    n_steps: int,
    *,
    bc: str | Callable = "slip",
) -> MPMState:
    """Time-step MLS-MPM for ``n_steps`` via ``jax.lax.scan`` (differentiable)."""
    def body(carry, _):
        return step_mpm(carry, cfg, bc=bc), None
    state_final, _ = jax.lax.scan(body, state0, xs=None, length=n_steps)
    return state_final


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def kinetic_energy(state: MPMState) -> jnp.ndarray:
    """Total kinetic energy ``Σ 0.5 m |v|^2``."""
    return 0.5 * jnp.sum(state.mass * jnp.sum(state.velocity ** 2, axis=-1))


def total_volume(state: MPMState) -> jnp.ndarray:
    """Total particle volume in the current (deformed) configuration."""
    J = jnp.linalg.det(state.deformation)
    return jnp.sum(state.volume0 * J)


def center_of_mass(state: MPMState) -> jnp.ndarray:
    """Mass-weighted mean position."""
    total_m = jnp.sum(state.mass)
    return jnp.sum(state.mass[:, None] * state.position, axis=0) / jnp.maximum(total_m, 1e-30)
