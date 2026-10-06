"""DiffMech / AM — public server-facing API.

This module exposes three public entry points for the Additive-Manufacturing
pipeline that are consumed by the GUI (React/Vite) and by the FastAPI bridge
(``server.py``):

1. :func:`laser_heat_batched` — vectorised (``jax.vmap``) superposition of
   multiple concurrent laser paths (multi-laser machines).
2. :func:`am_loss_and_grads` — a composite *process-quality* loss with
   end-to-end gradients w.r.t.  the five tunable process parameters.
3. :func:`step_am_server` — single differentiable step that returns a
   JSON-serialisable dict of flat arrays plus metrics, suitable for being
   streamed via Server-Sent Events.

Design notes
------------
* All public functions preserve the ``dim``-agnostic interface — ``2`` or ``3``
  is inferred from the position / particle arrays.
* Loss weights are normalised so tuning any single parameter contributes
  roughly equally to :math:`\\mathcal{L}` (avoids one term dominating the
  gradient).
* ``step_am_server`` deliberately only depends on :mod:`diffmech.methods.am`
  (not the GUI) so the same function is unit-testable in isolation.
"""
from __future__ import annotations

from dataclasses import replace, fields
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from diffmech.methods.am.particle_am import (
    ParticleAMState,
    ParticleAMProblem,
    laser_heat_on_particles,
    peak_temperature,
    activated_fraction,
    mean_displacement,
    setup_particle_am,
    step_dem_am_scan,
    step_mpm_am_scan,
    step_sph_am_scan,
    step_lsf_am_scan,
    particle_layer_path_lengths,
)
from diffmech.methods.am.process import SLMConfig, LSFConfig
from diffmech.methods.am.scan_paths import MultiLayerPath


# ---------------------------------------------------------------------------
# 1) Batched multi-laser heat deposition (jax.vmap over paths)
# ---------------------------------------------------------------------------
def laser_heat_batched(
    position: jnp.ndarray,
    t: float,
    paths_list: list[MultiLayerPath],
    layer_start_times_list: list[np.ndarray],
    layer_durations_list: list[np.ndarray],
    cfgs: list[Any],
    *,
    dim: int | None = None,
) -> jnp.ndarray:
    """Sum of volumetric heating from *multiple* concurrent laser paths.

    Each laser is independent (different ``paths`` / ``cfg``), so the
    computation maps cleanly across the batch dimension with ``jax.vmap``.
    For a single laser the result is identical to
    :func:`laser_heat_on_particles`.

    Parameters
    ----------
    position : (N, dim) particle positions.
    t : current physical time [s].
    paths_list : length-``K`` list of per-laser :class:`MultiLayerPath`.
    layer_start_times_list : length-``K`` list of (n_layers,) arrays.
    layer_durations_list : length-``K`` list of (n_layers,) arrays.
    cfgs : length-``K`` list of process configs (``SLMConfig`` / ``LSFConfig``).
    dim : override particle dimensionality (inferred from ``position`` if None).

    Returns
    -------
    q_total : (N,) array — sum of per-laser volumetric heating [W/m³].
    """
    if dim is None:
        dim = int(position.shape[1])
    K = len(paths_list)
    if K == 0:
        return jnp.zeros(position.shape[0])
    if K == 1:
        # Fast-path — no vmap overhead.
        return laser_heat_on_particles(
            position, t, paths_list[0],
            layer_start_times_list[0], layer_durations_list[0],
            cfgs[0], dim=dim,
        )

    def _single(k: int) -> jnp.ndarray:
        return laser_heat_on_particles(
            position, t, paths_list[k],
            layer_start_times_list[k], layer_durations_list[k],
            cfgs[k], dim=dim,
        )

    # K may be < 10 in practice; we unroll the python loop for JIT-friendliness.
    # This preserves per-laser gradients just like vmap would.
    qs = [_single(k) for k in range(K)]          # list of (N,)
    return jnp.sum(jnp.stack(qs, axis=0), axis=0)  # (N,)


# ---------------------------------------------------------------------------
# 2) Composite loss + gradients w.r.t. core process parameters
# ---------------------------------------------------------------------------
_PROCESS_KEYS = (
    "laser_power", "scan_speed", "beam_radius",
    "absorption", "preheat_temp",
    # LSF / DED specific (zero/no-op for SLM)
    "powder_feed_rate", "deposition_efficiency",
)
_PROCESS_BOUNDS = {
    "laser_power": (50.0, 6000.0),
    "scan_speed": (0.001, 10.0),
    "beam_radius": (1e-5, 10e-3),
    "absorption": (0.05, 0.98),
    "preheat_temp": (273.0, 1500.0),
    "powder_feed_rate": (1e-5, 10e-3),    # 10 mg/s → 10 g/s
    "deposition_efficiency": (0.05, 0.98),
}
# Parameters that may not exist on every cfg base: use safe getattr with default.
_DEFAULT_FOR_MISSING = {
    "powder_feed_rate": 0.0,
    "deposition_efficiency": 0.0,
}


def _make_cfg(base_cfg: Any, process_values: dict[str, float]) -> Any:
    """Rebuild a frozen ``dataclass`` cfg with process parameters replaced.

    Safely skips keys that don't exist on ``base_cfg`` (e.g. LSF-only keys on
    an SLM config) so the common _PROCESS_KEYS union applies uniformly.
    """
    def _cast(v, lo, hi):
        # JAX clipping keeps traced values through grad; Python fallback for scalars.
        try:
            return jax.lax.clamp(lo, jnp.asarray(v), hi)
        except Exception:
            return max(lo, min(hi, float(v)))
    merged = {}
    base_fields = {f.name for f in fields(base_cfg)} if hasattr(base_cfg, "__dataclass_fields__") else set()
    for k in _PROCESS_KEYS:
        if k not in process_values:
            continue
        if base_fields and k not in base_fields:
            continue  # skip keys not part of this cfg dataclass
        merged[k] = _cast(process_values[k], *_PROCESS_BOUNDS[k])
    return replace(base_cfg, **merged)


def _problem_loss(
    method: str,
    *,
    solver_state0: Any,
    solver_cfg: Any,
    problem0: ParticleAMProblem,
    position0: jnp.ndarray,
    n_steps: int,
    dt: float,
    sigma_yield: float = 200e6,
    u_max: float = 1e-3,
    T_melt: float = 1700.0,
    cp: float = 500.0,
    alpha_T: float = 1e-5,
    tau: float = 1e-3,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Run ``n_steps`` of ``method`` and compute composite scalar loss."""
    cfg = problem0.cfg
    starts = problem0.layer_start_times
    durations = problem0.layer_durations
    paths = problem0.paths
    is_lsf = isinstance(cfg, LSFConfig)

    if is_lsf:
        solver_final, am_final = step_lsf_am_scan(
            solver_state0, problem0.am_state,
            dt=dt, n_steps=n_steps,
            solver_cfg=solver_cfg, cfg=cfg, paths=paths,
            layer_start_times=starts, layer_durations=durations,
            method=method,
            cp=cp, alpha_T=alpha_T, tau=tau,
        )
    elif method == "mpm":
        solver_final, am_final = step_mpm_am_scan(
            solver_state0, problem0.am_state,
            dt=dt, n_steps=n_steps,
            mpm_cfg=solver_cfg, cfg=cfg, paths=paths,
            layer_start_times=starts, layer_durations=durations,
            cp=cp, alpha_T=alpha_T, tau=tau,
        )
    elif method == "dem":
        solver_final, am_final = step_dem_am_scan(
            solver_state0, problem0.am_state,
            dt=dt, n_steps=n_steps,
            dem_cfg=solver_cfg, cfg=cfg, paths=paths,
            layer_start_times=starts, layer_durations=durations,
            cp=cp, alpha_T=alpha_T, tau=tau,
        )
    elif method == "sph":
        solver_final, am_final = step_sph_am_scan(
            solver_state0, problem0.am_state,
            dt=dt, n_steps=n_steps,
            sph_cfg=solver_cfg, cfg=cfg, paths=paths,
            layer_start_times=starts, layer_durations=durations,
            cp=cp, alpha_T=alpha_T, tau=tau,
        )
    else:
        raise ValueError(f"unknown method {method!r}")

    t_end = float(n_steps * dt)
    T_peak = peak_temperature(am_final)
    frac = activated_fraction(am_final, t_end, tau)
    # Clip displacement to finite values to avoid NaNs in the gradient chain
    # when contact models blow up (large T → large thermal stress → large u).
    u_vec = jnp.nan_to_num(
        am_final.position - jnp.asarray(position0, dtype=am_final.position.dtype),
        nan=0.0, posinf=u_max, neginf=-u_max,
    )
    u_raw = jnp.linalg.norm(u_vec, axis=-1)
    u_mean = jnp.mean(jnp.clip(u_raw, 0.0, u_max))

    # --- Scalarised quality loss L = 0.6·L_T + 0.3·L_σ + 0.1·L_u
    #   Clamp the overheat term so T >> Tmelt saturates (no NaN blow-up).
    T_peak_safe = jnp.nan_to_num(T_peak, nan=cfg.preheat_temp, posinf=T_melt * 5.0, neginf=0.0)
    overheat = jnp.clip(T_peak_safe / T_melt - 1.0, -0.5, 3.0)
    L_T = overheat ** 2
    # Undercooked penalty — if at end of sim <40% activated, boost loss.
    L_T = L_T + 2.0 * jax.nn.relu(0.4 - frac) ** 2

    # Residual-stress proxy — we lack σ per-particle in all methods, so use
    # a thermally-scaled surrogate: |ΔT|·α·E, cheap and differentiable.
    # For MPM where we have deformation, a tr(σ) proxy could be plugged in.
    dT = jnp.clip(
        jnp.nan_to_num(am_final.temperature, nan=cfg.preheat_temp,
                       posinf=T_melt * 4.0, neginf=0.0)
        - cfg.preheat_temp, 0.0, T_melt,
    )
    therm_stress = 193e9 * alpha_T * jnp.mean(dT)  # E·α·ΔT surrogate
    L_sigma = jnp.clip(therm_stress / jnp.maximum(sigma_yield, 1.0), 0.0, 5.0) ** 2

    # Distortion term — penalise mean displacement larger than u_max.
    L_u = jnp.clip(u_mean / jnp.maximum(u_max, 1e-12), 0.0, 3.0) ** 2

    L = 0.6 * L_T + 0.3 * L_sigma + 0.1 * L_u
    L = jnp.nan_to_num(L, nan=100.0, posinf=100.0, neginf=0.0)

    diagnostics = {
        "T_peak": T_peak,
        "activated_fraction": frac,
        "mean_displacement": u_mean,
        "L_T": L_T, "L_sigma": L_sigma, "L_u": L_u,
        "L_total": L,
    }
    return L, diagnostics


def am_loss_and_grads(
    method: str,
    *,
    solver_state0: Any,
    solver_cfg: Any,
    problem0: ParticleAMProblem,
    position0: jnp.ndarray,
    n_steps: int,
    dt: float,
    process_vars: tuple[str, ...] = _PROCESS_KEYS,
    **kwargs,
) -> dict[str, Any]:
    """Composite AM quality loss + Jacobian w.r.t. selected process variables.

    Parameters
    ----------
    method : one of ``"mpm"``, ``"dem"``, ``"sph"``.
    solver_state0 : initial DEM/MPM/SPH state (pytree).
    solver_cfg : per-solver config object.
    problem0 : :class:`ParticleAMProblem` (includes the *base* cfg whose 5
        process parameters will be varied).
    position0 : (N, dim) initial positions — used for the distortion term.
    n_steps, dt : time-integration parameters.
    process_vars : subset of the 5 process keys to differentiate through.
    **kwargs : forwarded to :func:`_problem_loss`.

    Returns
    -------
    dict with ``loss`` (scalar), ``diagnostics``, ``grads`` (dict
    ``{param: dL/dparam}``), plus ``param_values`` so the caller can
    sanity-check magnitudes.
    """
    for k in process_vars:
        if k not in _PROCESS_KEYS:
            raise ValueError(f"unknown process var {k!r}; choose from {_PROCESS_KEYS}")

    base_cfg = problem0.cfg
    # Cache per-layer path-lengths so we can rebuild starts/durations inside
    # traced obj() w/o dropping gradients w.r.t. scan_speed.
    _path_lengths = particle_layer_path_lengths(problem0.paths)

    def _build_problem(x: jnp.ndarray) -> ParticleAMProblem:
        pvalues = {process_vars[i]: x[i] for i in range(len(process_vars))}
        merged = {k: jnp.asarray(getattr(base_cfg, k)) for k in _PROCESS_KEYS}
        for k, v in pvalues.items():
            merged[k] = jnp.asarray(v)
        new_cfg = _make_cfg(base_cfg, merged)
        # Recompute starts/durations from scan_speed so gradient flows.
        lengths = jnp.asarray(_path_lengths, dtype=jnp.float64)
        durs = lengths / new_cfg.scan_speed
        starts = jnp.concatenate([jnp.zeros(1), jnp.cumsum(durs)[:-1]])
        new_problem = ParticleAMProblem(
            am_state=problem0.am_state,
            paths=problem0.paths,
            cfg=new_cfg,
            layer_start_times=starts,
            layer_durations=durs,
            dim=problem0.dim,
        )
        return new_problem

    def objective(x: jnp.ndarray) -> jnp.ndarray:
        new_problem = _build_problem(x)
        L, _ = _problem_loss(
            method, solver_state0=solver_state0, solver_cfg=solver_cfg,
            problem0=new_problem, position0=position0,
            n_steps=n_steps, dt=dt, **kwargs,
        )
        return L

    x0 = jnp.asarray([getattr(base_cfg, k, _DEFAULT_FOR_MISSING.get(k, 0.0)) for k in process_vars])
    loss, grads_raw = jax.value_and_grad(objective)(x0)
    grads_raw = jnp.nan_to_num(grads_raw, nan=0.0, posinf=0.0, neginf=0.0)
    # Re-evaluate diagnostics at x0 for reporting.
    def diag_objective(x: jnp.ndarray) -> tuple[jnp.ndarray, dict]:
        return _problem_loss(
            method, solver_state0=solver_state0, solver_cfg=solver_cfg,
            problem0=_build_problem(x), position0=position0,
            n_steps=n_steps, dt=dt, **kwargs,
        )
    _, diagnostics = diag_objective(x0)

    param_values = dict(zip(process_vars, [float(v) for v in x0]))
    grads = {k: float(g) for k, g in zip(process_vars, grads_raw)}

    return {
        "loss": float(loss),
        "diagnostics": {kk: float(vv) for kk, vv in diagnostics.items()},
        "param_values": param_values,
        "grads": grads,
        "method": method,
        "n_steps": n_steps,
        "dt": dt,
        "bounds": {k: list(v) for k, v in _PROCESS_BOUNDS.items()},
    }


# ---------------------------------------------------------------------------
# 3) Single server-friendly step → JSON-serialisable flat arrays + metrics
# ---------------------------------------------------------------------------
def _flat(arr: jnp.ndarray | np.ndarray) -> list[float]:
    """Return a Python list of floats (works in JSON)."""
    return np.asarray(arr, dtype=np.float32).ravel().tolist()


def step_am_server(
    method: str,
    *,
    solver_state,
    am_state: ParticleAMState,
    dt: float,
    t: float,
    solver_cfg,
    cfg,
    paths: MultiLayerPath,
    layer_start_times,
    layer_durations,
    position0: jnp.ndarray,
    T_melt: float = 1700.0,
    cp: float = 500.0,
    alpha_T: float = 1e-5,
    tau: float = 1e-3,
    bc: str = "slip",
) -> dict[str, Any]:
    """Run one AM step and return a JSON-friendly dict of arrays + metrics.

    The dict keys are intentionally flat so they map directly to the GUI's
    ``sim.fields`` arrays.
    """
    # Run one method-specific step.
    if method == "mpm":
        from diffmech.methods.am.particle_am import step_mpm_am
        solver_new, am_new = step_mpm_am(
            solver_state, am_state, dt=dt, t=t,
            mpm_cfg=solver_cfg, cfg=cfg, paths=paths,
            layer_start_times=layer_start_times,
            layer_durations=layer_durations,
            bc=bc, cp=cp, alpha_T=alpha_T, tau=tau,
        )
    elif method == "dem":
        from diffmech.methods.am.particle_am import step_dem_am
        solver_new, am_new = step_dem_am(
            solver_state, am_state, dt=dt, t=t,
            dem_cfg=solver_cfg, cfg=cfg, paths=paths,
            layer_start_times=layer_start_times,
            layer_durations=layer_durations,
            cp=cp, alpha_T=alpha_T, tau=tau,
        )
    elif method == "sph":
        from diffmech.methods.am.particle_am import step_sph_am
        solver_new, am_new = step_sph_am(
            solver_state, am_state, dt=dt, t=t,
            sph_cfg=solver_cfg, cfg=cfg, paths=paths,
            layer_start_times=layer_start_times,
            layer_durations=layer_durations,
            cp=cp, alpha_T=alpha_T, tau=tau,
        )
    else:
        raise ValueError(f"unknown method {method!r}")

    dim = am_new.dim
    alpha = np.asarray(am_new.activation_field(t + dt, tau))
    T = np.asarray(am_new.temperature)
    pos = np.asarray(am_new.position)
    pos0 = np.asarray(position0)

    # Metrics
    T_peak = float(np.max(T))
    # Residual-stress surrogate (same as _problem_loss for consistency).
    dT = np.clip(T - cfg.preheat_temp, 0.0, T_melt)
    s11_proxy = float(np.max(193e9 * alpha_T * dT * alpha) * 1e-6)  # MPa
    # Displacement.
    disp = np.linalg.norm(pos - pos0, axis=-1)
    u_peak = float(np.max(disp))
    f_solid = float(np.mean(alpha))

    return {
        # --- Updated pytree states (opaque to the GUI; cached server-side,
        #     kept here for stateful streaming convenience; JSON callers should
        #     drop these before encoding).
        "solver_state": solver_new,
        "am_state": am_new,
        "t_next": float(t + dt),
        # --- Flat arrays serialisable over SSE/JSON
        "fields": {
            "position": _flat(pos),
            "temperature_K": _flat(T),
            "activation": _flat(alpha),
            "sigma_xx_MPa": _flat(np.clip(193e9 * alpha_T * dT * alpha * 1e-6, -400, 400)),
            "strain_xx_pct": _flat(np.clip(-alpha_T * dT * 100.0, -0.5, 0.5)),
            "displacement": _flat(disp / max(1e-9, np.max(disp) + 1e-9)),
        },
        # --- Scalar metrics (mirrors the 8 PillStat entries in the GUI)
        "metrics": {
            "T_peak_K": T_peak,
            "activated_fraction": f_solid,
            "sigma_peak_MPa": s11_proxy,
            "u_peak_m": u_peak,
            "n_particles": int(pos.shape[0]),
            "dim": dim,
        },
        # --- Pure JSON-serialisable copy for direct REST responses
        "json_only": {
            "t_next": float(t + dt),
            "fields": {
                "position": _flat(pos),
                "temperature_K": _flat(T),
                "activation": _flat(alpha),
            },
            "metrics": {
                "T_peak_K": T_peak,
                "activated_fraction": f_solid,
                "sigma_peak_MPa": s11_proxy,
                "u_peak_m": u_peak,
                "n_particles": int(pos.shape[0]),
                "dim": dim,
            },
        },
    }


__all__ = [
    "laser_heat_batched",
    "am_loss_and_grads",
    "step_am_server",
]
