"""Differentiable Phase-Field methods (2D & 3D).

Implements two classical phase-field models on a uniform Cartesian grid:

1. **Allen-Cahn equation** — order-parameter evolution for microstructure
   (e.g. spinodal decomposition, grain growth)::

       ∂φ/∂t = M [ κ ∇²φ − f'(φ) ]

   with double-well potential ``f(φ) = ¼ (φ²−1)²`` so
   ``f'(φ) = φ³ − φ``.

2. **Phase-field fracture** (variational brittle fracture, Bourdin–Francfort–
   Marigo regularisation)::

       ∂φ/∂t = −M · δE/δφ

   where the total energy is ::

       E[φ] = ∫ [ g(φ) · ψ_elastic + Gc · ( φ²/(2ℓ) + ℓ|∇φ|²/2 ) ] dV

   Here ``g(φ) = (1−φ)²`` degrades the elastic energy as the phase field
   ``φ → 0`` (fully broken), ``Gc`` is the critical energy release rate, and
   ``ℓ`` is the regularisation length.

Both models support 2D and 3D — dimensionality is inferred from the field
shape.  Spatial derivatives use central differences via ``jnp.roll`` (periodic
BCs) or ghost-cell padding (Neumann BCs).  Time integration uses semi-implicit
Euler (Allen-Cahn) or explicit Euler (fracture), advanced through
``jax.lax.scan`` for end-to-end ``jax.grad`` compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Laplacian helper (uniform grid, central differences)
# ---------------------------------------------------------------------------
def laplacian(field: jnp.ndarray, dx: float, dim: int,
              bc: str = "periodic") -> jnp.ndarray:
    """Discrete Laplacian ``∇²f`` via central differences.

    Parameters
    ----------
    field : (nx, ny[, nz], ...) array
        Cell-centred scalar or vector field.
    dx : float
        Uniform grid spacing (same in every direction).
    dim : int
        Spatial dimension (2 or 3).
    bc : str
        ``"periodic"`` uses ``jnp.roll``; ``"neumann"`` (zero-gradient)
        uses ghost-cell padding.

    Returns
    -------
    Array of the same shape as ``field``.
    """
    if bc == "periodic":
        lap = jnp.zeros_like(field)
        for d in range(dim):
            f_plus = jnp.roll(field, -1, axis=d)
            f_minus = jnp.roll(field, 1, axis=d)
            lap = lap + (f_plus - 2.0 * field + f_minus) / (dx * dx)
        return lap
    # Neumann (zero-gradient) ghost cells.
    ng = 1
    pad_spec = [(ng, ng)] * dim + [(0, 0)] * (field.ndim - dim)
    f_pad = jnp.pad(field, pad_spec, mode="edge")
    lap = jnp.zeros_like(field)
    for d in range(dim):
        # Spatial axes: interior [ng:-ng]; trailing component axes: full.
        sl_p = [slice(ng, -ng) if ax < dim else slice(None) for ax in range(f_pad.ndim)]
        sl_m = [slice(ng, -ng) if ax < dim else slice(None) for ax in range(f_pad.ndim)]
        sl_c = [slice(ng, -ng) if ax < dim else slice(None) for ax in range(f_pad.ndim)]
        sl_p[d] = slice(ng + 1, None)     # right neighbour
        sl_m[d] = slice(ng - 1, -ng - 1)  # left neighbour
        f_plus = f_pad[tuple(sl_p)]
        f_minus = f_pad[tuple(sl_m)]
        f_c = f_pad[tuple(sl_c)]
        lap = lap + (f_plus - 2.0 * f_c + f_minus) / (dx * dx)
    return lap


def gradient(field: jnp.ndarray, dx: float, dim: int,
             bc: str = "periodic") -> jnp.ndarray:
    """Central-difference gradient ``∇f`` — returns ``(nx, ny[, nz], dim)``."""
    grad = []
    for d in range(dim):
        if bc == "periodic":
            df = (jnp.roll(field, -1, axis=d) - jnp.roll(field, 1, axis=d)) / (2.0 * dx)
        else:
            ng = 1
            pad_spec = [(ng, ng)] * dim + [(0, 0)] * (field.ndim - dim)
            f_pad = jnp.pad(field, pad_spec, mode="edge")
            sl_p = [slice(ng, -ng) if ax < dim else slice(None) for ax in range(f_pad.ndim)]
            sl_m = [slice(ng, -ng) if ax < dim else slice(None) for ax in range(f_pad.ndim)]
            sl_p[d] = slice(ng + 1, None)     # right neighbour
            sl_m[d] = slice(ng - 1, -ng - 1)  # left neighbour
            df = (f_pad[tuple(sl_p)] - f_pad[tuple(sl_m)]) / (2.0 * dx)
        grad.append(df)
    return jnp.stack(grad, axis=-1)


# ---------------------------------------------------------------------------
# Allen-Cahn model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AllenCahnConfig:
    """Allen-Cahn simulation parameters.

    Attributes
    ----------
    dx : float
        Grid spacing (uniform).
    dt : float
        Time step.
    mobility : float
        Interface mobility ``M``.
    kappa : float
        Gradient energy coefficient ``κ``.
    bc : str
        Boundary condition: ``"periodic"`` or ``"neumann"``.
    """

    dx: float
    dt: float
    mobility: float = 1.0
    kappa: float = 1.0
    bc: str = "periodic"


# Register as pytree (bc is static aux data, numerics are leaves).
def _ac_cfg_flatten(cfg: AllenCahnConfig):
    return (cfg.dx, cfg.dt, cfg.mobility, cfg.kappa), cfg.bc


def _ac_cfg_unflatten(bc, children):
    dx, dt, M, kappa = children
    return AllenCahnConfig(dx=dx, dt=dt, mobility=M, kappa=kappa, bc=bc)


jax.tree_util.register_pytree_node(AllenCahnConfig, _ac_cfg_flatten, _ac_cfg_unflatten)


@dataclass(frozen=True)
class AllenCahnState:
    """Allen-Cahn state — just the order-parameter field.

    Attributes
    ----------
    phi : (nx, ny[, nz]) array
        Order parameter in ``[−1, 1]`` (−1 and +1 are the two phases).
    """

    phi: jax.Array

    @property
    def dim(self) -> int:
        return int(self.phi.ndim)


# Register as pytree.
def _ac_flatten(s: AllenCahnState):
    return (s.phi,), None


def _ac_unflatten(_, children):
    return AllenCahnState(phi=children[0])


jax.tree_util.register_pytree_node(AllenCahnState, _ac_flatten, _ac_unflatten)


def double_well_potential(phi: jnp.ndarray) -> jnp.ndarray:
    """``f(φ) = ¼ (φ² − 1)²``."""
    return 0.25 * (phi ** 2 - 1.0) ** 2


def double_well_derivative(phi: jnp.ndarray) -> jnp.ndarray:
    """``f'(φ) = φ³ − φ``."""
    return phi ** 3 - phi


def allen_cahn_rhs(state: AllenCahnState, cfg: AllenCahnConfig) -> jnp.ndarray:
    """Right-hand side of the Allen-Cahn equation.

    ``dφ/dt = M [ κ ∇²φ − f'(φ) ]``
    """
    lap = laplacian(state.phi, cfg.dx, state.dim, cfg.bc)
    return cfg.mobility * (cfg.kappa * lap - double_well_derivative(state.phi))


def step_allen_cahn(
    state: AllenCahnState,
    cfg: AllenCahnConfig,
) -> AllenCahnState:
    """Advance Allen-Cahn by one explicit Euler step."""
    rhs = allen_cahn_rhs(state, cfg)
    return AllenCahnState(phi=state.phi + cfg.dt * rhs)


def step_allen_cahn_scan(
    state0: AllenCahnState,
    cfg: AllenCahnConfig,
    n_steps: int,
) -> AllenCahnState:
    """Time-step Allen-Cahn for ``n_steps`` via ``jax.lax.scan`` (differentiable)."""
    def body(carry, _):
        return step_allen_cahn(carry, cfg), None
    state_final, _ = jax.lax.scan(body, state0, xs=None, length=n_steps)
    return state_final


def free_energy_allen_cahn(state: AllenCahnState, cfg: AllenCahnConfig) -> jnp.ndarray:
    """Total free energy ``∫ [f(φ) + ½ κ |∇φ|²] dV``."""
    f_bulk = double_well_potential(state.phi)
    grad_phi = gradient(state.phi, cfg.dx, state.dim, cfg.bc)
    f_grad = 0.5 * cfg.kappa * jnp.sum(grad_phi ** 2, axis=-1)
    return jnp.sum((f_bulk + f_grad) * cfg.dx ** state.dim)


# ---------------------------------------------------------------------------
# Phase-field fracture model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FractureConfig:
    """Phase-field fracture simulation parameters.

    Attributes
    ----------
    dx : float
        Grid spacing.
    dt : float
        Time step.
    Gc : float
        Critical energy release rate.
    ell : float
        Regularisation length (crack half-width).
    mobility : float
        Phase-field mobility (controls how fast φ evolves).
    bc : str
        Boundary condition for ∇² and ∇.
    irrev_threshold : float
        Strain energy threshold below which the crack does not grow
        (history variable is the maximum strain energy seen so far).
    """

    dx: float
    dt: float
    Gc: float = 1.0
    ell: float = 0.05
    mobility: float = 1.0
    bc: str = "neumann"
    irrev_threshold: float = 0.0


# Register as pytree (bc is static aux data, numerics are leaves).
def _frac_cfg_flatten(cfg: FractureConfig):
    return (cfg.dx, cfg.dt, cfg.Gc, cfg.ell, cfg.mobility, cfg.irrev_threshold), cfg.bc


def _frac_cfg_unflatten(bc, children):
    dx, dt, Gc, ell, M, irr = children
    return FractureConfig(dx=dx, dt=dt, Gc=Gc, ell=ell, mobility=M,
                          bc=bc, irrev_threshold=irr)


jax.tree_util.register_pytree_node(FractureConfig, _frac_cfg_flatten, _frac_cfg_unflatten)


@dataclass(frozen=True)
class FractureState:
    """Phase-field fracture state.

    Attributes
    ----------
    phi : (nx, ny[, nz]) array
        Phase field: 1 = intact, 0 = fully broken.
    history : (nx, ny[, nz]) array
        History field ``H`` = maximum strain energy density seen so far
        (for irreversibility).
    displacement : (nx, ny[, nz], dim) array
        Displacement field (used to compute strain energy).
    """

    phi: jax.Array
    history: jax.Array
    displacement: jax.Array

    @property
    def dim(self) -> int:
        return int(self.phi.ndim)


def _frac_flatten(s: FractureState):
    return (s.phi, s.history, s.displacement), None


def _frac_unflatten(_, children):
    phi, hist, disp = children
    return FractureState(phi=phi, history=hist, displacement=disp)


jax.tree_util.register_pytree_node(FractureState, _frac_flatten, _frac_unflatten)


def make_fracture_state(
    shape: tuple[int, ...],
    dim: int | None = None,
) -> FractureState:
    """Create an initial fracture state (fully intact, zero displacement).

    Parameters
    ----------
    shape : (nx, ny[, nz])
        Grid shape.
    dim : int, optional
        Spatial dimension (inferred from ``shape`` if omitted).
    """
    if dim is None:
        dim = len(shape)
    phi = jnp.ones(shape, dtype=jnp.float64)
    hist = jnp.zeros(shape, dtype=jnp.float64)
    disp = jnp.zeros(shape + (dim,), dtype=jnp.float64)
    return FractureState(phi=phi, history=hist, displacement=disp)


def degradation_function(phi: jnp.ndarray) -> jnp.ndarray:
    """``g(φ) = (1 − φ)²`` — degrades elastic energy as φ → 0."""
    return (1.0 - phi) ** 2


def strain_energy_density(
    displacement: jnp.ndarray,
    dx: float,
    dim: int,
    youngs_modulus: float = 1.0,
    poissons_ratio: float = 0.3,
    bc: str = "neumann",
) -> jnp.ndarray:
    """Compute the strain energy density ``ψ = ½ σ : ε``.

    Uses small-strain elasticity: ``ε = ½ (∇u + ∇uᵀ)``,
    ``σ = 2μ ε + λ tr(ε) I``.
    """
    mu = youngs_modulus / (2.0 * (1.0 + poissons_ratio))
    lam = youngs_modulus * poissons_ratio / ((1.0 + poissons_ratio) * (1.0 - 2.0 * poissons_ratio))
    grad_u = gradient(displacement, dx, dim, bc)  # (..., dim)
    eps = 0.5 * (grad_u + jnp.swapaxes(grad_u, -1, -2))
    tr_eps = jnp.trace(eps, axis1=-2, axis2=-1)  # (...)
    sigma = 2.0 * mu * eps + lam * tr_eps[..., None, None] * jnp.eye(dim)
    psi = 0.5 * jnp.sum(sigma * eps, axis=(-2, -1))
    return psi


def fracture_rhs(
    state: FractureState,
    cfg: FractureConfig,
    *,
    youngs_modulus: float = 1.0,
    poissons_ratio: float = 0.3,
) -> jnp.ndarray:
    """Right-hand side of the phase-field fracture evolution equation.

    The driving force is::

        ∂φ/∂t = −M · δE/δφ

    where::

        δE/δφ = −2(1−φ) · ψ_max  +  Gc · (φ/ℓ − ℓ ∇²φ)

    Here ``ψ_max`` is the history field (maximum strain energy seen),
    providing irreversibility.
    """
    dim = state.dim
    psi = strain_energy_density(
        state.displacement, cfg.dx, dim, youngs_modulus, poissons_ratio, cfg.bc,
    )
    # Update history: H = max(H, ψ)
    history_new = jnp.maximum(state.history, psi)
    # Driving force from elastic energy degradation.
    g_prime = -2.0 * (1.0 - state.phi)  # d/dφ [(1-φ)²] = -2(1-φ)
    elastic_force = g_prime * history_new
    # Fracture surface term: Gc * (φ/ℓ − ℓ ∇²φ)
    lap_phi = laplacian(state.phi, cfg.dx, dim, cfg.bc)
    surface_force = cfg.Gc * (state.phi / cfg.ell - cfg.ell * lap_phi)
    return -cfg.mobility * (elastic_force + surface_force)


def step_fracture(
    state: FractureState,
    cfg: FractureConfig,
    *,
    youngs_modulus: float = 1.0,
    poissons_ratio: float = 0.3,
) -> FractureState:
    """Advance the phase-field fracture model by one explicit Euler step.

    Updates ``φ`` via the evolution equation and the history field.
    The displacement is NOT updated here (caller sets it from the
    mechanical solver).
    """
    rhs = fracture_rhs(state, cfg, youngs_modulus=youngs_modulus,
                       poissons_ratio=poissons_ratio)
    phi_new = state.phi + cfg.dt * rhs
    # Clamp φ to [0, 1] to keep it physical.
    phi_new = jnp.clip(phi_new, 0.0, 1.0)
    # Update history field.
    psi = strain_energy_density(
        state.displacement, cfg.dx, state.dim,
        youngs_modulus, poissons_ratio, cfg.bc,
    )
    history_new = jnp.maximum(state.history, psi)
    return FractureState(
        phi=phi_new,
        history=history_new,
        displacement=state.displacement,
    )


def step_fracture_scan(
    state0: FractureState,
    cfg: FractureConfig,
    n_steps: int,
    *,
    youngs_modulus: float = 1.0,
    poissons_ratio: float = 0.3,
) -> FractureState:
    """Time-step fracture for ``n_steps`` via ``jax.lax.scan`` (differentiable)."""
    def body(carry, _):
        return step_fracture(carry, cfg,
                             youngs_modulus=youngs_modulus,
                             poissons_ratio=poissons_ratio), None
    state_final, _ = jax.lax.scan(body, state0, xs=None, length=n_steps)
    return state_final


def fracture_energy(state: FractureState, cfg: FractureConfig,
                    youngs_modulus: float = 1.0,
                    poissons_ratio: float = 0.3) -> jnp.ndarray:
    """Total fracture energy ``E = ∫ [g(φ) ψ + Gc (φ²/(2ℓ) + ℓ|∇φ|²/2)] dV``."""
    dim = state.dim
    psi = strain_energy_density(
        state.displacement, cfg.dx, dim, youngs_modulus, poissons_ratio, cfg.bc,
    )
    g = degradation_function(state.phi)
    elastic_term = g * psi
    grad_phi = gradient(state.phi, cfg.dx, dim, cfg.bc)
    surface_term = cfg.Gc * (state.phi ** 2 / (2.0 * cfg.ell) +
                             0.5 * cfg.ell * jnp.sum(grad_phi ** 2, axis=-1))
    return jnp.sum((elastic_term + surface_term) * cfg.dx ** dim)
