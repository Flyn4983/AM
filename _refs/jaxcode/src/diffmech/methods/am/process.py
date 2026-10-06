"""Process-parameter configurations for AM simulations.

Captures the build-recipe parameters that distinguish the two AM modalities:

- :class:`SLMConfig` — **Selective Laser Melting** (a.k.a. L-PBF / Powder Bed
  Fusion): a recoater spreads a thin powder bed, the laser raster-scans each
  slice (zigzag hatch), the build plate drops by one layer thickness, repeat.
  Physics: powder → melt → rapid solidification, fine features, high residual
  stress from steep gradients.

- :class:`LSFConfig` — **Laser Solid Forming** (a.k.a. LDED / Direct Energy
  Deposition): powder or wire is fed *continuously* into the moving laser
  spot, depositing material directly onto a substrate or prior layers. No
  powder bed; larger melt pool, lower resolution, used for repair / large
  parts. Physics: synchronous mass feed, bigger melt pool, slower cooling.

Both are plain ``dataclass(frozen=True)`` so they are JAX-pytree-friendly and
can be threaded through ``jax.grad`` (you can differentiate w.r.t. laser
power, scan speed, layer thickness, …).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# SLMConfig / LSFConfig pytree registrations
# ---------------------------------------------------------------------------
# Numeric fields are leaves; bools/strings are treated as static auxiliary
# metadata so JAX.jit doesn't re-trace on every e.g. ``contour_first`` toggle.
_SLMC_NUMERIC = (
    "laser_power", "scan_speed", "layer_thickness", "hatch_spacing",
    "beam_radius", "absorption", "rotation_per_layer", "preheat_temp",
)
_LSFC_NUMERIC = (
    "laser_power", "scan_speed", "layer_thickness", "track_spacing",
    "beam_radius", "powder_feed_rate", "absorption", "deposition_efficiency",
    "preheat_temp",
)


def _slm_flatten(cfg: SLMConfig):  # noqa: F821 (registered below)
    leaves = tuple(getattr(cfg, k) for k in _SLMC_NUMERIC)
    aux = (cfg.contour_first,)
    return leaves, aux


def _slm_unflatten(aux, leaves):
    (contour_first,) = aux
    kw = dict(zip(_SLMC_NUMERIC, leaves))
    kw["contour_first"] = contour_first
    return SLMConfig(**kw)


def _lsf_flatten(cfg: LSFConfig):  # noqa: F821
    leaves = tuple(getattr(cfg, k) for k in _LSFC_NUMERIC)
    return leaves, None


def _lsf_unflatten(_aux, leaves):
    return LSFConfig(**dict(zip(_LSFC_NUMERIC, leaves)))


@dataclass(frozen=True)
class SLMConfig:
    """Selective Laser Melting (Powder Bed Fusion) process parameters.

    All quantities are SI units unless noted.

    Attributes
    ----------
    laser_power : float [W]
        Beam power (absorbed fraction handled by ``absorption``).
    scan_speed : float [m/s]
        Beam travel speed along each hatch line.
    layer_thickness : float [m]
        Powder-bed layer thickness (typical SLM: 30–60 µm).
    hatch_spacing : float [m]
        Distance between adjacent hatch lines (typical: 50–100 µm).
    beam_radius : float [m]
        1/e² beam radius (typical: 40–80 µm).
    absorption : float
        Absorbed fraction of beam power (0–1).
    contour_first : bool
        If True, scan the boundary outline before hatching the interior.
    rotation_per_layer : float [rad]
        Hatch angle rotated by this amount each layer (isotropic properties).
    preheat_temp : float [K]
        Build-plate / powder-bed preheat temperature.
    """

    laser_power: float = 200.0
    scan_speed: float = 1.0
    layer_thickness: float = 40e-6
    hatch_spacing: float = 80e-6
    beam_radius: float = 50e-6
    absorption: float = 0.5
    contour_first: bool = True
    rotation_per_layer: float = 1.5708  # 90° per layer
    preheat_temp: float = 373.0  # 100 °C preheat


@dataclass(frozen=True)
class LSFConfig:
    """Laser Solid Forming (Direct Energy Deposition) process parameters.

    Attributes
    ----------
    laser_power : float [W]
        Beam power (typically 1–3 kW, much higher than SLM).
    scan_speed : float [m/s]
        Deposition travel speed (slower than SLM, ~0.005–0.02 m/s).
    layer_thickness : float [m]
        Deposit height per pass (typical: 0.3–1 mm).
    track_spacing : float [m]
        Lateral spacing between adjacent deposition tracks.
    beam_radius : float [m]
        Effective melt-pool radius (typical: 1–3 mm).
    powder_feed_rate : float [kg/s]
        Mass flow rate of fed powder (LSF-specific).
    absorption : float
        Absorbed fraction of beam power.
    deposition_efficiency : float
        Fraction of fed powder that actually deposits (0–1).
    preheat_temp : float [K]
        Substrate preheat temperature.
    """

    laser_power: float = 1500.0
    scan_speed: float = 0.01
    layer_thickness: float = 0.5e-3
    track_spacing: float = 1.5e-3
    beam_radius: float = 2e-3
    powder_feed_rate: float = 5e-4
    absorption: float = 0.35
    deposition_efficiency: float = 0.7
    preheat_temp: float = 473.0


def layer_time(cfg: SLMConfig | LSFConfig, path_length: float) -> float:
    """Time to scan one layer of total arc-length ``path_length``.

    Neglects the (short) recoat / reposition time — only the active laser
    time, which dominates the thermal history. Differentiable w.r.t. the
    process parameters.
    """
    return path_length / cfg.scan_speed


def layer_activation_times(cfg: SLMConfig | LSFConfig,
                           layer_path_lengths) -> tuple:
    """Cumulative activation time of each layer (start, end) tuples.

    Returns
    -------
    starts, ends : (n_layers,) arrays of [s].
        ``starts[k]`` is when layer ``k`` begins depositing; ``ends[k]`` when
        it finishes (and the laser moves to the next layer).
    """
    lengths = jnp.asarray(layer_path_lengths, dtype=jnp.float64)
    durations = lengths / cfg.scan_speed
    starts = jnp.concatenate([jnp.zeros(1), jnp.cumsum(durations)[:-1]])
    ends = starts + durations
    return starts, ends


# Register pytrees *after* the dataclasses are defined so the helpers can see
# the class names via string forward-refs above.
jax.tree_util.register_pytree_node(SLMConfig, _slm_flatten, _slm_unflatten)
jax.tree_util.register_pytree_node(LSFConfig, _lsf_flatten, _lsf_unflatten)


__all__ = [
    "SLMConfig", "LSFConfig",
    "layer_time", "layer_activation_times",
]
