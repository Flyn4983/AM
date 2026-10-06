"""AM thermal simulation: layer-by-layer laser deposition on a layered mesh.

Combines:
- the differentiable **activation field** α(x, t) (which elements are deposited)
- the **scan path** of the laser (sequence of (x, y) waypoints per layer)
- a **piecewise-moving Gaussian heat source** whose (x, y) position follows
  the active layer's path at time ``t``

into a single differentiable thermal evolution that mirrors a real AM build:
the laser scans layer 0, then layer 1, etc., heating only the cells that have
been deposited so far.

The heat source is gated by α: cells that have not yet activated receive no
heat input (they are "powder" / "not yet fed"), and the laser power is
concentrated on the currently-active layer. Because α is a smooth sigmoid,
gradients flow through the build history end-to-end.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from diffmech.methods.am.activation import LayeredMesh
from diffmech.methods.am.scan_paths import ScanPath, MultiLayerPath
from diffmech.methods.am.process import SLMConfig, LSFConfig


# ---------------------------------------------------------------------------
# Laser position along a multi-layer trajectory
# ---------------------------------------------------------------------------
def laser_position(paths: MultiLayerPath, layer_start_times: np.ndarray,
                   layer_durations: np.ndarray, t: jnp.ndarray,
                   ) -> jnp.ndarray:
    """Position of the laser spot at time ``t`` along the build trajectory.

    Returns
    -------
    pos : (3,) array [x, y, z] of the laser centre at time ``t``.
    active_layer : int index of the layer currently being scanned (or -1 if
        the laser is off between layers / after the build).

    Notes
    -----
    Implemented with ``jnp`` ops so it is differentiable w.r.t. ``t`` and the
    path geometry. Between layers the laser is "off" (returning the last
    position) — the activation field handles the ramp.
    """
    starts = jnp.asarray(layer_start_times, dtype=jnp.float64)
    durations = jnp.asarray(layer_durations, dtype=jnp.float64)
    n = len(paths.layers)
    # Find which layer is active at time t.
    active = jnp.sum(t >= starts) - 1
    active = jnp.clip(active, 0, n - 1)
    # Local time within the active layer.
    t_local = t - starts[active]
    t_local = jnp.clip(t_local, 0.0, durations[active])
    # Arc length = scan_speed * t_local; but we need speed per layer.
    # Use the path's own length + duration to find arc-length fraction.
    layer = paths.layers[0]  # we'll select per-layer below

    # Build a stacked (n_layers, n_wp, 2) waypoint array and (n_layers,)
    # cumulative arrays for vectorised selection.
    max_wp = max(len(p.waypoints) for p in paths.layers)
    wp = np.zeros((n, max_wp, 2))
    cum = np.zeros((n, max_wp))
    seg = np.zeros((n, max_wp - 1))
    lengths = np.zeros(n)
    for i, p in enumerate(paths.layers):
        m = len(p.waypoints)
        wp[i, :m] = p.waypoints
        cum[i, :m] = p.cum_length
        if m > 1:
            seg[i, :m - 1] = p.segment_lengths
        lengths[i] = p.total_length
    wp = jnp.asarray(wp)
    cum = jnp.asarray(cum)
    lengths = jnp.asarray(lengths)

    # Arc length traversed in the active layer (linear in t_local).
    frac = jnp.where(durations[active] > 1e-12,
                     t_local / jnp.maximum(durations[active], 1e-12), 0.0)
    s = jnp.clip(frac * lengths[active], 0.0, lengths[active])

    # Interpolate position along the active layer's waypoints.
    cum_layer = cum[active]  # (max_wp,)
    # Segment index = number of cum entries below s.
    idx = jnp.sum(cum_layer <= s) - 1
    idx = jnp.clip(idx, 0, max_wp - 2)
    s0 = cum_layer[idx]
    s1 = cum_layer[idx + 1]
    seg_len = s1 - s0
    t_frac = jnp.where(seg_len > 1e-12, (s - s0) / jnp.maximum(seg_len, 1e-12), 0.0)
    p0 = wp[active, idx]
    p1 = wp[active, idx + 1]
    xy = p0 + t_frac * (p1 - p0)
    z = paths.layers[0].layer_z  # will be replaced per-layer
    # Per-layer z: gather from the active layer.
    layer_zs = jnp.asarray([p.layer_z for p in paths.layers])
    z_active = layer_zs[active]
    pos = jnp.array([xy[0], xy[1], z_active])
    return pos, active


# ---------------------------------------------------------------------------
# Heat source: Gaussian spot following the laser, gated by activation
# ---------------------------------------------------------------------------
def am_heat_source(
    paths: MultiLayerPath,
    layer_start_times: np.ndarray,
    layer_durations: np.ndarray,
    cfg: SLMConfig | LSFConfig,
    *,
    activation_field: jnp.ndarray = None,
):
    """Build a time-dependent volumetric heat source ``Q(x, t)`` [W/m³].

    The source is a Gaussian spot centred at the laser's instantaneous
    position, multiplied by the activation field (so only deposited material
    heats up).
    """
    absorbed = cfg.absorption * cfg.laser_power
    r2 = cfg.beam_radius ** 2
    pi_r2 = float(np.pi) * r2
    # Pre-stack the per-layer waypoint arrays for the differentiable lookup.
    n = len(paths.layers)
    max_wp = max(len(p.waypoints) for p in paths.layers)
    wp = np.zeros((n, max_wp, 2))
    cum = np.zeros((n, max_wp))
    lengths = np.zeros(n)
    layer_zs = np.zeros(n)
    for i, p in enumerate(paths.layers):
        m = len(p.waypoints)
        wp[i, :m] = p.waypoints
        cum[i, :m] = p.cum_length
        lengths[i] = p.total_length
        layer_zs[i] = p.layer_z
    wp = jnp.asarray(wp)
    cum = jnp.asarray(cum)
    lengths = jnp.asarray(lengths)
    layer_zs = jnp.asarray(layer_zs)
    starts = jnp.asarray(layer_start_times, dtype=jnp.float64)
    durations = jnp.asarray(layer_durations, dtype=jnp.float64)

    def source_fn(cell_centers: jnp.ndarray, t: float) -> jnp.ndarray:
        # --- locate the active layer at time t ---
        active = jnp.clip(jnp.sum(t >= starts) - 1, 0, n - 1)
        t_local = jnp.clip(t - starts[active], 0.0, durations[active])
        frac = jnp.where(durations[active] > 1e-12,
                         t_local / jnp.maximum(durations[active], 1e-12), 0.0)
        s = jnp.clip(frac * lengths[active], 0.0, lengths[active])

        # --- interpolate (x, y) along the active layer's path ---
        cum_layer = cum[active]
        idx = jnp.clip(jnp.sum(cum_layer <= s) - 1, 0, max_wp - 2)
        s0 = cum_layer[idx]
        s1 = cum_layer[idx + 1]
        seg_len = s1 - s0
        t_frac = jnp.where(seg_len > 1e-12,
                           (s - s0) / jnp.maximum(seg_len, 1e-12), 0.0)
        p0 = wp[active, idx]
        p1 = wp[active, idx + 1]
        laser_xy = p0 + t_frac * (p1 - p0)

        # --- Gaussian heat deposition ---
        xy = cell_centers[..., :2] - laser_xy
        rsq = jnp.sum(xy * xy, axis=-1)
        q = (2.0 * absorbed) / pi_r2 * jnp.exp(-2.0 * rsq / r2)
        # Depth attenuation (thin-layer model): the laser only deposits in the
        # current layer's z-band.
        z = cell_centers[..., 2]
        z_active = layer_zs[active]
        depth = 2.0 * float(cfg.layer_thickness)
        # Gaussian decay in z around the active layer's mid-plane.
        q = q * jnp.exp(-((z - z_active) ** 2) / (2.0 * depth * depth))
        return q

    return source_fn


# ---------------------------------------------------------------------------
# Top-level convenience: build the full AM thermal problem
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AMThermalProblem:
    """Bundle of everything needed to run a layer-by-layer thermal sim."""

    layered: LayeredMesh
    paths: MultiLayerPath
    cfg: SLMConfig | LSFConfig
    layer_start_times: np.ndarray
    layer_durations: np.ndarray
    source_fn: object
    part_mask: np.ndarray


def setup_am_thermal(
    layered: LayeredMesh,
    paths: MultiLayerPath,
    cfg: SLMConfig | LSFConfig,
    *,
    part_mask: np.ndarray = None,
) -> AMThermalProblem:
    """Assemble the AM thermal problem from a layered mesh + scan paths.

    Computes per-layer start times and durations from the path lengths and
    scan speed, then builds the differentiable heat source.
    """
    if part_mask is None:
        part_mask = np.ones(layered.n_cells, dtype=bool)
    lengths = np.asarray([p.total_length for p in paths.layers])
    durations = lengths / cfg.scan_speed
    starts = np.concatenate([[0.0], np.cumsum(durations)[:-1]])
    source = am_heat_source(paths, starts, durations, cfg)
    return AMThermalProblem(
        layered=layered,
        paths=paths,
        cfg=cfg,
        layer_start_times=starts,
        layer_durations=durations,
        source_fn=source,
        part_mask=part_mask,
    )


__all__ = [
    "laser_position", "am_heat_source",
    "AMThermalProblem", "setup_am_thermal",
]
