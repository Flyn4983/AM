"""AM microstructure evolution driven by the thermal history.

Couples a phase-field solidification model with the AM thermal field to
simulate the **heating → melting → cooling → solidification** microstructure
development that occurs in additive manufacturing (Selective Laser Melting,
Laser Solid Forming, laser cladding repair).

The same two extensibility entry points as the rest of the AM module —
*arbitrary complex geometry* and *user-specified laser paths / process
parameters* — drive the microstructure, because the thermal field
``T(x, t)`` that this module consumes is itself produced by the AM thermal
solver (which already depends on laser power, scan speed, layer thickness,
scan path, beam angle, …). As a result ``jax.grad`` flows end-to-end from
process parameters to microstructure metrics (grain size, morphology, …).

Physics
-------
1. **Solidification phase field** ``φ ∈ [0, 1]`` (0 = liquid, 1 = solid),
   evolved by a Kobayashi-style phase-field equation driven by the
   undercooling ``ΔT = T_melt − T(x, t)``::

       τ ∂φ/∂t = W² ∇²φ + φ(1−φ)(φ − ½ + λ·u)
       u = −ΔT / (L/c_p)        (dimensionless undercooling)

   Where ``T > T_melt`` (melt pool) the driving force pins ``φ → 0`` (liquid);
   where ``T < T_melt`` (solidified material) ``φ → 1`` (solid). The diffuse
   interface width ``W`` and relaxation time ``τ`` set the microstructure
   length / time scale.

2. **Grain-orientation field** ``θ`` (polycrystalline structure). Each cell
   carries an orientation that is *nucleated* (set) when it first solidifies
   and then coarsens by an Allen-Cahn-type equation whose mobility is
   temperature-gated (Arrhenius: fast near the melting point, frozen when
   cold — mimicking epitaxial growth from the substrate and grain growth
   during thermal cycling).

3. **Microstructure metrics**: solidified fraction, estimated grain count
   and mean grain size, and a columnar-vs-equiaxed morphology indicator from
   the temperature-gradient / solidification-velocity ratio ``G/R`` (the
   classical Hunt–Kurz morphological map).

Dimensionality
--------------
Grid-based (uniform Cartesian), works in **2D and 3D**. Dimensionality is
inferred from the field shape (``φ.ndim``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from diffmech.methods.phase_field.phase_field import laplacian, gradient


# Thermal-field callable: takes (grid_coords, t) and returns T on the grid.
# For the FEM AM thermal solver this wraps `source_fn` + a transient solver;
# for particle AM it wraps `laser_heat_on_particles` accumulated temperatures.
ThermalField = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


# ===========================================================================
# Configuration & state
# ===========================================================================
@dataclass(frozen=True)
class MicrostructureConfig:
    """Parameters for the AM microstructure evolution.

    Attributes
    ----------
    dx : float
        Uniform grid spacing [m].
    dt : float
        Time step [s].
    T_melt : float
        Melting (liquidus) temperature [K].
    L_cp : float
        Latent heat over specific heat, ``L / c_p`` [K] — sets the
        undercooling scale.
    interface_W : float
        Diffuse solid/liquid interface half-width ``W`` [m].
    tau : float
        Phase-field relaxation time ``τ`` [s].
    lam : float
        Coupling coefficient ``λ`` between undercooling and the phase field.
    mobility0 : float
        Pre-exponential grain-growth mobility ``M₀`` [m⁴/(J·s)].
    Q_over_R : float
        Activation temperature ``Q/R`` [K] for the Arrhenius mobility.
    bc : str
        Boundary condition for the Laplacian (``"periodic"`` or ``"neumann"``).
    """
    dx: float
    dt: float
    T_melt: float = 1900.0
    L_cp: float = 300.0
    interface_W: float = 1.5e-5
    tau: float = 1e-4
    lam: float = 2.0
    mobility0: float = 1.0
    Q_over_R: float = 15000.0
    bc: str = "neumann"


def _mc_cfg_flatten(cfg: MicrostructureConfig):
    return (cfg.dx, cfg.dt, cfg.T_melt, cfg.L_cp, cfg.interface_W,
            cfg.tau, cfg.lam, cfg.mobility0, cfg.Q_over_R), cfg.bc


def _mc_cfg_unflatten(bc, children):
    (dx, dt, Tm, Lcp, W, tau, lam, M0, QR) = children
    return MicrostructureConfig(dx=dx, dt=dt, T_melt=Tm, L_cp=Lcp,
                                interface_W=W, tau=tau, lam=lam,
                                mobility0=M0, Q_over_R=QR, bc=bc)


jax.tree_util.register_pytree_node(
    MicrostructureConfig, _mc_cfg_flatten, _mc_cfg_unflatten)


@dataclass(frozen=True)
class AMMicrostructureState:
    """State of the AM microstructure field.

    Attributes
    ----------
    phi : (nx, ny[, nz]) array
        Solid-fraction phase field ``φ ∈ [0, 1]`` (0 liquid, 1 solid).
    theta : (nx, ny[, nz]) array
        Grain-orientation field ``θ ∈ [0, 2π)``.
    nucleated : (nx, ny[, nz]) array
        Indicator of cells that have already nucleated a grain (0/1). Used
        to fix the orientation once a cell is solid (epitaxial growth).
    """
    phi: jax.Array
    theta: jax.Array
    nucleated: jax.Array

    @property
    def dim(self) -> int:
        return int(self.phi.ndim)


def _ms_flatten(s: AMMicrostructureState):
    return (s.phi, s.theta, s.nucleated), None


def _ms_unflatten(_, children):
    phi, theta, nuc = children
    return AMMicrostructureState(phi=phi, theta=theta, nucleated=nuc)


jax.tree_util.register_pytree_node(AMMicrostructureState,
                                   _ms_flatten, _ms_unflatten)


def make_microstructure_state(
    shape: tuple[int, ...],
    *,
    key: jax.Array | None = None,
    phi0: float = 0.0,
    substrate_layers: int = 1,
) -> AMMicrostructureState:
    """Create an initial microstructure state (liquid + solid substrate).

    In additive manufacturing solidification grows *epitaxially* from the
    build plate / previously-deposited layers, so the phase field is seeded
    solid (``φ = 1``) at the bottom ``substrate_layers`` rows of the grid.
    The undercooling then drives the solid/liquid interface upward into the
    melt pool — without this seed the Kobayashi driving force vanishes at
    ``φ = 0`` and no solid would nucleate (the classic phase-field
    metastability).

    Parameters
    ----------
    shape : (nx, ny[, nz])
        Grid shape.
    key : jax PRNG key, optional. If ``None`` a fixed key is used so results
        are reproducible.
    phi0 : initial solid fraction of the melt region (0 = fully liquid).
    substrate_layers : number of bottom rows along the build direction (the
        last spatial axis, ``z``) seeded solid as the build-plate substrate.
        Set to 0 for a fully-liquid domain.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    phi = jnp.full(shape, float(phi0), dtype=jnp.float64)
    # Seed the substrate (bottom rows) solid.
    if substrate_layers > 0:
        # The build direction is the LAST spatial axis (consistent with the
        # AM module: z is the build direction).
        build_axis = len(shape) - 1
        sl = slice(None) if substrate_layers >= shape[build_axis] \
            else slice(0, substrate_layers)
        idx = [slice(None)] * len(shape)
        idx[build_axis] = sl
        phi = phi.at[tuple(idx)].set(1.0)
    # Random initial orientation in [0, 2π) — this is the nucleation "palette";
    # the orientation only becomes meaningful once a cell solidifies.
    theta = jax.random.uniform(key, shape, minval=0.0,
                               maxval=2.0 * float(np.pi), dtype=jnp.float64)
    nucleated = (phi >= 0.5).astype(jnp.float64)
    return AMMicrostructureState(phi=phi, theta=theta, nucleated=nucleated)


# ===========================================================================
# Thermal-field coupling helpers
# ===========================================================================
def thermal_undercooling(
    T_field: jnp.ndarray, cfg: MicrostructureConfig,
) -> jnp.ndarray:
    """Dimensionless undercooling ``u = ΔT / (L/c_p)`` with ``ΔT = T_melt − T``.

    ``u > 0`` ⇒ below melting point (undercooled → solidifies);
    ``u < 0`` ⇒ above melting point (melt pool → melts).
    """
    return (cfg.T_melt - T_field) / cfg.L_cp


def grain_mobility(T_field: jnp.ndarray, cfg: MicrostructureConfig,
                   ) -> jnp.ndarray:
    """Temperature-gated Arrhenius grain-growth mobility ``M(T)``.

    Fast near ``T_melt`` (liquid/solid coexistence, thermal cycling), frozen
    when cold — this mimics epitaxial growth and grain coarsening in the
    heat-affected zone.
    """
    # Smoothly switch the mobility on only where T is between ~0.6·T_melt and
    # T_melt (the HAZ / mushy zone). Below that, grains are frozen.
    T = jnp.maximum(T_field, 0.0)
    arrhenius = jnp.exp(-cfg.Q_over_R / jnp.maximum(T, 1.0))
    # Melt cap: above T_melt the orientation is meaningless (liquid).
    melt_gate = jax.nn.sigmoid((cfg.T_melt - T) / 50.0)  # 1 below melt, 0 above
    return cfg.mobility0 * arrhenius * melt_gate


# ===========================================================================
# Phase-field solidification RHS (Kobayashi-style)
# ===========================================================================
def solidification_rhs(
    state: AMMicrostructureState,
    u: jnp.ndarray,
    cfg: MicrostructureConfig,
    *,
    T_field: jnp.ndarray | None = None,
    melt_width: float = 50.0,
    melt_strength: float = 50.0,
) -> jnp.ndarray:
    """Right-hand side of the solidification phase-field equation.

    ``τ ∂φ/∂t = W² ∇²φ + φ(1−φ)(φ − ½ + λ·u) − melt_strength · melt_rate · φ / τ``

    where ``u`` is the dimensionless undercooling. In undercooled material
    (``u > 0``, ``T < T_melt``) the Kobayashi cubic term drives ``φ → 1``
    (solidifies). The final **melt-pool dissolution** term is proportional to
    ``φ`` (not ``φ(1−φ)``) so it can drive a fully-solid cell (``φ = 1``) back
    to liquid when ``T > T_melt`` — this is the re-melting that occurs in the
    AM melt pool. It is differentiable (a smooth sigmoid gate) and removes the
    metastability of the pure Kobayashi form at ``φ = 1``. ``melt_strength``
    makes re-melting much faster than solidification kinetics so the melt pool
    stays liquid (the diffusion of solid into the melt pool is overwhelmed).

    Parameters
    ----------
    u : dimensionless undercooling from :func:`thermal_undercooling`.
    T_field : temperature field [K] (needed for the melt-pool gate). If
        ``None`` the dissolution term is omitted (pure Kobayashi).
    melt_width : temperature half-width of the melt-pool sigmoid gate [K].
    melt_strength : re-melting rate relative to the phase-field kinetics.
    """
    dim = state.dim
    lap = laplacian(state.phi, cfg.dx, dim, cfg.bc)
    phi = state.phi
    # Kobayashi double-well driving: φ(1-φ)(φ - 1/2 + λ·u).
    driving = phi * (1.0 - phi) * (phi - 0.5 + cfg.lam * u)
    rhs = (cfg.interface_W ** 2 * lap + driving) / cfg.tau
    if T_field is not None:
        # Smooth melt-pool gate: 1 above T_melt (dissolve), 0 below.
        melt_rate = jax.nn.sigmoid((T_field - cfg.T_melt) / melt_width)
        rhs = rhs - melt_strength * melt_rate * phi / cfg.tau
    return rhs


# ===========================================================================
# Grain-orientation RHS (temperature-gated Allen-Cahn coarsening)
# ===========================================================================
def orientation_rhs(
    state: AMMicrostructureState,
    T_field: jnp.ndarray,
    cfg: MicrostructureConfig,
    *,
    kappa_theta: float = 1.0,
) -> jnp.ndarray:
    """Right-hand side of the grain-orientation coarsening equation.

    The orientation field ``θ`` obeys a temperature-gated Allen-Cahn
    equation: it coarsens (reduces orientation-gradient energy) with a
    mobility ``M(T)`` that is large in the heat-affected zone and zero in
    cold, fully-solid material. Once a cell nucleates (solidifies), its
    orientation is frozen by the ``nucleated`` mask — this captures epitaxial
    growth from the substrate and competitive columnar growth.

    ``∂θ/∂t = M(T) · [ κ_θ ∇²θ ] · (1 − nucleated)``
    """
    dim = state.dim
    lap_theta = laplacian(state.theta, cfg.dx, dim, cfg.bc)
    M = grain_mobility(T_field, cfg)
    # Frozen where already nucleated (epitaxial: orientation locked at
    # solidification). ``nucleated`` ∈ {0, 1}.
    active = 1.0 - state.nucleated
    return M * kappa_theta * lap_theta * active


# ===========================================================================
# Nucleation: lock orientation at the solidification front
# ===========================================================================
def _nucleate_front(
    state: AMMicrostructureState,
    u: jnp.ndarray,
    *,
    nucleation_threshold: float = 0.5,
) -> AMMicrostructureState:
    """Mark cells as nucleated once the solid fraction crosses a threshold.

    When ``φ`` first exceeds ``nucleation_threshold`` (i.e. the cell begins to
    solidify) we set ``nucleated = 1`` so its (random) orientation is locked
    in — this is the grain-nucleation event. The orientation then propagates
    into neighbouring liquid by the coarsening equation (epitaxial growth).
    """
    newly_solid = (state.phi > nucleation_threshold).astype(state.phi.dtype)
    nucleated_new = jnp.maximum(state.nucleated, newly_solid)
    return AMMicrostructureState(
        phi=state.phi, theta=state.theta, nucleated=nucleated_new)


# ===========================================================================
# One microstructure step (coupled with the AM thermal field)
# ===========================================================================
def step_am_microstructure(
    state: AMMicrostructureState,
    cfg: MicrostructureConfig,
    T_field: jnp.ndarray,
    *,
    kappa_theta: float = 1.0,
    nucleation_threshold: float = 0.5,
    melt_width: float = 50.0,
    melt_strength: float = 50.0,
) -> AMMicrostructureState:
    """Advance the AM microstructure by one time step.

    Parameters
    ----------
    state : current microstructure state.
    cfg : microstructure config.
    T_field : temperature field on the same grid as ``state.phi`` [K], at the
        current time. This is the coupling point with the AM thermal solver
        (``am_thermal`` for the FEM path, ``particle_am`` for the particle
        path).
    kappa_theta : grain-boundary energy coefficient for the orientation field.
    nucleation_threshold : solid-fraction above which a cell is marked
        nucleated (its orientation is then frozen).
    melt_width : temperature half-width of the melt-pool sigmoid gate [K].
    melt_strength : re-melting rate relative to phase-field kinetics.
    """
    u = thermal_undercooling(T_field, cfg)
    # 1. Solidification phase field (with melt-pool dissolution driven by T).
    dphi = solidification_rhs(state, u, cfg, T_field=T_field,
                              melt_width=melt_width,
                              melt_strength=melt_strength)
    phi_new = jnp.clip(state.phi + cfg.dt * dphi, 0.0, 1.0)
    # Re-melted cells (melt pool) lose their nucleation status so they can
    # re-nucleate on the next solidification pass (AM thermal cycling).
    melt_rate = jax.nn.sigmoid((T_field - cfg.T_melt) / melt_width)
    nucleated_melt = state.nucleated * (1.0 - melt_rate)
    state_after_solid = AMMicrostructureState(
        phi=phi_new, theta=state.theta, nucleated=nucleated_melt)
    # 2. Nucleation (lock orientation at the moving front).
    state_nuc = _nucleate_front(state_after_solid, u,
                                nucleation_threshold=nucleation_threshold)
    # 3. Grain-orientation coarsening (temperature-gated).
    dtheta = orientation_rhs(state_nuc, T_field, cfg,
                             kappa_theta=kappa_theta)
    theta_new = state_nuc.theta + cfg.dt * dtheta
    return AMMicrostructureState(
        phi=phi_new, theta=theta_new, nucleated=state_nuc.nucleated)


def step_am_microstructure_scan(
    state0: AMMicrostructureState,
    cfg: MicrostructureConfig,
    thermal_fields: jnp.ndarray,
    *,
    kappa_theta: float = 1.0,
    nucleation_threshold: float = 0.5,
    melt_width: float = 50.0,
    melt_strength: float = 50.0,
) -> AMMicrostructureState:
    """Time-step the microstructure through a thermal history.

    Parameters
    ----------
    state0 : initial microstructure state.
    cfg : microstructure config.
    thermal_fields : (n_steps, nx, ny[, nz]) array of temperature fields,
        one per time step. This is the thermal history ``T(x, t)`` produced
        by the AM thermal solver — the differentiable coupling channel that
        carries the dependence on laser power, scan speed, …
    """
    def body(carry, T_t):
        s = step_am_microstructure(
            carry, cfg, T_t,
            kappa_theta=kappa_theta,
            nucleation_threshold=nucleation_threshold,
            melt_width=melt_width,
            melt_strength=melt_strength)
        return s, None
    state_final, _ = jax.lax.scan(body, state0, thermal_fields)
    return state_final


# ===========================================================================
# Microstructure metrics
# ===========================================================================
def solidified_fraction(state: AMMicrostructureState) -> jnp.ndarray:
    """Mean solid fraction ``⟨φ⟩`` (0 = all liquid, 1 = fully solid)."""
    return jnp.mean(state.phi)


def nucleated_fraction(state: AMMicrostructureState) -> jnp.ndarray:
    """Fraction of cells that have nucleated a grain."""
    return jnp.mean(state.nucleated)


def mean_grain_size(state: AMMicrostructureState, dx: float) -> jnp.ndarray:
    """Estimated mean grain size [m] from the orientation field.

    Uses the inverse of the orientation-gradient magnitude averaged over
    solidified cells: regions of nearly-constant orientation (inside a grain)
    contribute a large length; sharp orientation jumps (grain boundaries)
    are excluded. The estimate is ``dx / ⟨|∇θ|⟩`` over nucleated cells,
    clamped to a physically meaningful range.
    """
    dim = state.dim
    grad_theta = gradient(state.theta, dx, dim, "neumann")  # (..., dim)
    grad_mag = jnp.sqrt(jnp.sum(grad_theta ** 2, axis=-1) + 1e-30)
    # Only consider nucleated (solid) cells.
    weight = state.nucleated
    mean_grad = jnp.sum(grad_mag * weight) / jnp.maximum(
        jnp.sum(weight), 1.0)
    # Grain size ~ 1 / mean orientation gradient (in cells) × dx.
    grain_cells = 1.0 / jnp.maximum(mean_grad, 1e-3)
    return grain_cells * dx


def grain_count_estimate(state: AMMicrostructureState, dx: float) -> jnp.ndarray:
    """Estimate the number of distinct grains in the solidified region.

    Approximated as ``V_solid / V_grain`` where ``V_grain`` is the mean grain
    volume estimated from :func:`mean_grain_size`. This is an order-of-
    magnitude estimate suitable for differentiable optimisation (it tracks
    *trends* in grain count as process parameters vary, not absolute counts).
    """
    dim = state.dim
    gs = mean_grain_size(state, dx)
    n_solid = jnp.sum(state.nucleated)
    # Grain volume ~ gs^dim.
    grain_vol = gs ** dim
    return n_solid / jnp.maximum(grain_vol / (dx ** dim), 1.0)


def cooling_rate(T_history: jnp.ndarray, dt: float) -> jnp.ndarray:
    """Mean cooling rate ``|dT/dt|`` [K/s] over the thermal history.

    ``T_history`` has shape ``(n_steps, ...)``. Used to drive
    grain-refinement estimates (faster cooling ⇒ finer grains).
    """
    if T_history.shape[0] < 2:
        return jnp.zeros(())
    dT = T_history[1:] - T_history[:-1]
    # Only count cooling (negative dT).
    cooling = jnp.where(dT < 0.0, -dT, 0.0)
    return jnp.mean(cooling) / dt


def morphology_indicator(
    grad_T: jnp.ndarray, R: jnp.ndarray,
) -> jnp.ndarray:
    """Columnar-vs-equiaxed morphology indicator from ``G/R`` (Hunt–Kurz).

    Parameters
    ----------
    grad_T : temperature gradient ``G = |∇T|`` [K/m] at the solidification
        front.
    R : solidification-velocity [m/s].

    Returns
    -------
    indicator ∈ (−1, 1):  −1 → equiaxed (low G/R),  +1 → columnar (high G/R).

    The classical Hunt map places the columnar-to-equiaxed transition at a
    critical ``G/R``; we map it through a sigmoid centred on a typical AM
    transition value.
    """
    # Typical AM transition ~ 1e6 K·s/m⁻² (Hunt). Use a smooth sigmoid.
    transition = 1.0e6
    GR = grad_T / jnp.maximum(R, 1e-12)
    return jnp.tanh((GR - transition) / (0.5 * transition))


# ===========================================================================
# Coupling: build a thermal-field callable from the AM thermal problem
# ===========================================================================
def thermal_field_from_am(
    cell_centers: jnp.ndarray,
    source_fn: Callable,
    *,
    T0: float = 300.0,
    rho_cp: float = 4.0e6,
    k_cond: float = 20.0,
    dx: float = 1e-4,
    n_thermal_steps: int = 1,
    dt_thermal: float = 1e-4,
):
    """Build a differentiable thermal-field callable ``T(t)`` for coupling.

    Wraps the AM heat source (``source_fn(cell_centers, t)`` from
    :mod:`am_thermal`) in a simple explicit transient heat-conduction update
    so the microstructure module can consume a temperature field ``T(x, t)``
    that depends differentiably on the laser power / scan speed / path.

    The returned callable keeps an internal temperature state and returns the
    field at time ``t``. Because everything is ``jax``-based, gradients flow
    from the process parameters (captured in ``source_fn``'s closure) through
    the heat equation to the microstructure.

    Parameters
    ----------
    cell_centers : (n, dim) grid-cell centres [m].
    source_fn : AM heat source ``Q(x, t)`` [W/m³] from ``am_heat_source``.
    T0 : initial (preheat) temperature [K].
    rho_cp : volumetric heat capacity ``ρ·c_p`` [J/(m³·K)].
    k_cond : thermal conductivity [W/(m·K)].
    dx : grid spacing for the diffusion term [m].
    n_thermal_steps : thermal sub-steps per microstructure step.
    dt_thermal : thermal time step [s].
    """
    # Internal mutable state held in a list (closed over by the callable).
    state = [jnp.full(cell_centers.shape[0], float(T0), dtype=jnp.float64)]

    def laplacian_particles(T):
        # Pairwise-distance-based Laplacian (small n). For larger grids use
        # the grid-based laplacian from phase_field.
        r = cell_centers
        diff = r[:, None, :] - r[None, :, :]  # (n, n, dim)
        dist2 = jnp.sum(diff ** 2, axis=-1) + 1e-30
        # Kernel-weighted Laplacian (Gaussian, width = 2·dx).
        w = jnp.exp(-dist2 / (2.0 * (2.0 * dx) ** 2))
        w = w / jnp.maximum(jnp.sum(w, axis=1, keepdims=True), 1e-30)
        # ΔT_i = Σ_j w_ij (T_j - T_i)
        return jnp.sum(w * (T[None, :] - T[:, None]), axis=1)

    def T_at(t):
        T = state[0]
        for _ in range(n_thermal_steps):
            Q = source_fn(cell_centers, t)
            lap = laplacian_particles(T)
            dT = dt_thermal * (Q / rho_cp + k_cond / rho_cp * lap)
            T = T + dT
        state[0] = T
        return T

    return T_at


__all__ = [
    "ThermalField",
    "MicrostructureConfig",
    "AMMicrostructureState",
    "make_microstructure_state",
    "thermal_undercooling",
    "grain_mobility",
    "solidification_rhs",
    "orientation_rhs",
    "step_am_microstructure",
    "step_am_microstructure_scan",
    "solidified_fraction",
    "nucleated_fraction",
    "mean_grain_size",
    "grain_count_estimate",
    "cooling_rate",
    "morphology_indicator",
    "thermal_field_from_am",
]
