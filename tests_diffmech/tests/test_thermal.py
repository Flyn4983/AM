"""Tests for the differentiable thermal (heat-diffusion) FVM solver (2D & 3D).

Covers:
    - diffusion_dt stability limit (diffusion CFL)
    - Smoothing of a hot spot (max decreases, energy conserved under Neumann)
    - Steady-state source-driven 1D profile (analytic check)
    - Dirichlet BC pins boundary temperature
    - Robin (convective) cooling reduces temperature toward T_inf
    - Spatially-varying diffusivity field
    - Gaussian heat-source power integration
    - Moving Gaussian source shape
    - 2D and 3D smoke tests
    - End-to-end differentiability w.r.t. alpha
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from diffmech.methods.fvm import (
    CartesianGrid,
    cartesian_grid_2d,
    cartesian_grid_3d,
    ThermalConfig,
    diffusion_rhs,
    step_thermal,
    step_thermal_scan,
    diffusion_dt,
    gaussian_heat_source,
    moving_gaussian_source,
)


# ---------------------------------------------------------------------------
# Stability limit
# ---------------------------------------------------------------------------
def test_diffusion_dt_scales_with_dx2_and_alpha():
    g = cartesian_grid_2d(8, 8, lx=1.0, ly=1.0)
    dt1 = diffusion_dt(g, alpha=1.0, cfl=0.5)
    # dx = 1/8, dim = 2 -> dt = 0.5 * (1/8)^2 / (2*2*1) = 0.5/256 = ~0.00195
    assert dt1 == pytest.approx(0.5 * (1 / 8) ** 2 / (2 * 2 * 1.0), rel=1e-12)
    # Doubling alpha halves dt
    dt2 = diffusion_dt(g, alpha=2.0, cfl=0.5)
    assert dt2 == pytest.approx(dt1 / 2.0, rel=1e-12)
    # 3D uses dim=3 -> smaller dt
    g3 = cartesian_grid_3d(8, 8, 8, lx=1.0, ly=1.0, lz=1.0)
    dt3 = diffusion_dt(g3, alpha=1.0, cfl=0.5)
    assert dt3 == pytest.approx(0.5 * (1 / 8) ** 2 / (2 * 3 * 1.0), rel=1e-12)


def test_diffusion_dt_uses_max_alpha_for_field():
    g = cartesian_grid_2d(4, 4, lx=1.0, ly=1.0)
    alpha_field = np.ones((4, 4)) * 0.5
    alpha_field[0, 0] = 2.0  # localised spike dominates the stability bound
    dt = diffusion_dt(g, alpha=alpha_field, cfl=0.5)
    dt_expected = 0.5 * (1 / 4) ** 2 / (2 * 2 * 2.0)
    assert dt == pytest.approx(dt_expected, rel=1e-12)


# ---------------------------------------------------------------------------
# Smoothing + energy conservation (Neumann)
# ---------------------------------------------------------------------------
def test_hot_spot_smoothing_2d_neumann():
    """A localised hot spot must decay under diffusion; total heat conserved."""
    n = 32
    g = cartesian_grid_2d(n, n, lx=1.0, ly=1.0)
    T0 = jnp.full((n, n, 1), 300.0)
    T0 = T0.at[n // 2, n // 2, 0].set(2000.0)
    alpha = 1e-3
    dt = diffusion_dt(g, alpha=alpha, cfl=0.4)
    Tf = step_thermal_scan(T0, g, dt, 100, alpha=alpha, bc="neumann")
    assert float(Tf.max()) < 2000.0          # peak decays
    assert float(Tf.min()) > 300.0           # min rises (heat spreads)
    # Total heat = sum(T * cell_volume) is conserved (insulated walls).
    total0 = float(jnp.sum(T0) * g.dx * g.dy)
    totalf = float(jnp.sum(Tf) * g.dx * g.dy)
    assert totalf == pytest.approx(total0, rel=1e-3)


def test_3d_smoke_neumann():
    n = 8
    g = cartesian_grid_3d(n, n, n, lx=1.0, ly=1.0, lz=1.0)
    T0 = jnp.full((n, n, n, 1), 300.0)
    T0 = T0.at[4, 4, 4, 0].set(1500.0)
    dt = diffusion_dt(g, alpha=1e-3, cfl=0.4)
    Tf = step_thermal_scan(T0, g, dt, 20, alpha=1e-3, bc="neumann")
    assert Tf.shape == (n, n, n, 1)
    assert float(Tf.max()) < 1500.0
    assert bool(jnp.isfinite(Tf).all())


# ---------------------------------------------------------------------------
# Dirichlet BC
# ---------------------------------------------------------------------------
def test_dirichlet_pins_boundary_temperature():
    """Dirichlet walls cool the domain toward T_bc; boundary < interior."""
    n = 16
    g = cartesian_grid_2d(n, n, lx=1.0, ly=1.0)
    T0 = jnp.full((n, n, 1), 500.0)
    # Dirichlet T_bc=300 on all walls; interior will cool toward it.
    dt = diffusion_dt(g, alpha=1e-2, cfl=0.4)
    Tf = step_thermal_scan(T0, g, dt, 200, alpha=1e-2, bc="dirichlet", T_bc=300.0)
    # The wall (face) is pinned to 300 by the mirror ghost cell, so the
    # boundary *cell* (half a cell inside) must be cooler than the interior
    # and trending toward 300 — but not yet exactly 300 (only ~2 diffusion
    # times have elapsed). Check the qualitative cooling, not an exact pin.
    bnd = float(Tf[0, 0, 0])
    ctr = float(Tf[n // 2, n // 2, 0])
    assert bnd < ctr                       # boundary cooler than interior
    assert 300.0 < bnd < 500.0             # between ambient and initial
    assert ctr < 500.0                     # interior has cooled
    # All four corners (boundary) should be ~equal (symmetric BC).
    assert abs(float(Tf[0, 0, 0]) - float(Tf[-1, -1, 0])) < 1e-6


# ---------------------------------------------------------------------------
# Robin (convective) cooling
# ---------------------------------------------------------------------------
def test_robin_cools_toward_ambient():
    n = 16
    g = cartesian_grid_2d(n, n, lx=1.0, ly=1.0)
    T0 = jnp.full((n, n, 1), 1000.0)
    # Strong convection (high h) -> boundary approaches T_inf=300.
    dt = diffusion_dt(g, alpha=1e-2, cfl=0.4)
    Tf = step_thermal_scan(
        T0, g, dt, 300, alpha=1e-2, bc="robin", T_inf=300.0,
        h_conv=1e3, k_cond=1.0,
    )
    # Boundary must be cooler than the interior and trending toward 300.
    assert float(Tf[0, 0, 0]) < float(Tf[n // 2, n // 2, 0])
    assert float(Tf[0, 0, 0]) > 300.0  # not below ambient


def test_robin_with_zero_h_is_neumann():
    """h_conv=0 should reduce the Robin BC to insulated (Neumann)."""
    n = 16
    g = cartesian_grid_2d(n, n, lx=1.0, ly=1.0)
    T0 = jnp.full((n, n, 1), 1000.0)
    T0 = T0.at[8, 8, 0].set(2000.0)
    dt = diffusion_dt(g, alpha=1e-3, cfl=0.4)
    Tf_neu = step_thermal_scan(T0, g, dt, 50, alpha=1e-3, bc="neumann")
    Tf_rob = step_thermal_scan(
        T0, g, dt, 50, alpha=1e-3, bc="robin", T_inf=300.0, h_conv=0.0, k_cond=1.0,
    )
    np.testing.assert_allclose(np.asarray(Tf_neu), np.asarray(Tf_rob), atol=1e-4)


# ---------------------------------------------------------------------------
# Source-driven steady state (1D analytic)
# ---------------------------------------------------------------------------
def test_steady_state_2d_source_analytic():
    """2D heat eq with uniform source Q and Dirichlet walls (all four sides
    at T=0) has the analytic steady-state series solution

        T(x,y) = Σ_{m,n odd} (16 Q / (π^4 k m n (m^2+n^2)))
                  sin(m π x / L) sin(n π y / L).

    At steady state the numerical solution must match it.
    """
    L = 1.0
    n = 24
    g = cartesian_grid_2d(n, n, lx=L, ly=L)
    alpha = 1.0
    rho_cp = 1.0
    Q = 2.0
    k = alpha * rho_cp  # = 1
    T0 = jnp.zeros((n, n, 1))
    dt = diffusion_dt(g, alpha=alpha, cfl=0.4)
    # Long time -> steady state (run > 5 diffusion times L^2/alpha = 1).
    Tf = step_thermal_scan(
        T0, g, dt, 20000, alpha=alpha, rho_cp=rho_cp,
        source=Q, bc="dirichlet", T_bc=0.0,
    )
    centres = np.asarray(g.cell_centers).reshape(n, n, 2)
    xs = centres[:, 0, 0]
    ys = centres[0, :, 1]
    # Truncated double-sine series (9x9 odd modes is well converged).
    T_exact = np.zeros((n, n))
    for m in range(1, 20, 2):
        for nn in range(1, 20, 2):
            T_exact += (
                (16.0 * Q) / (math.pi ** 4 * k * m * nn * (m * m + nn * nn))
                * np.sin(m * math.pi * xs / L)[:, None]
                * np.sin(nn * math.pi * ys / L)[None, :]
            )
    np.testing.assert_allclose(
        np.asarray(Tf).reshape(n, n), T_exact, atol=2e-3,
    )


# ---------------------------------------------------------------------------
# Spatially-varying diffusivity
# ---------------------------------------------------------------------------
def test_spatially_varying_alpha_field():
    """A spatially-varying alpha field is accepted and shapes the diffusion."""
    n = 16
    g = cartesian_grid_2d(n, n, lx=1.0, ly=1.0)
    # Left half is a good conductor, right half an insulator.
    alpha = np.ones((n, n)) * 1e-2
    alpha[:, : n // 2] = 1e-1
    T0 = jnp.zeros((n, n, 1))
    T0 = T0.at[2, 2, 0].set(1000.0)  # hot spot in the conductor
    dt = diffusion_dt(g, alpha=alpha, cfl=0.4)
    Tf = step_thermal_scan(T0, g, dt, 50, alpha=alpha, bc="neumann")
    assert float(Tf.max()) < 1000.0
    # Heat spreads faster to the left (conductor) than to the right.
    left = float(jnp.sum(Tf[: n // 2]))
    right = float(jnp.sum(Tf[n // 2 :]))
    assert left > right


# ---------------------------------------------------------------------------
# Heat-source models
# ---------------------------------------------------------------------------
def test_gaussian_source_integrates_to_absorbed_power():
    n = 64
    g = cartesian_grid_2d(n, n, lx=1.0, ly=1.0)
    src = gaussian_heat_source(power=200.0, absorption=0.5, beam_radius=0.05,
                               center=(0.5, 0.5))
    Q = np.asarray(src(g.cell_centers, 0.0)).reshape(n, n)
    total = float(np.sum(Q) * g.dx * g.dy)
    # Total deposited power = absorption * power = 100 W
    assert total == pytest.approx(100.0, rel=1e-2)


def test_gaussian_source_centred_at_peak():
    n = 32
    g = cartesian_grid_2d(n, n, lx=1.0, ly=1.0)
    src = gaussian_heat_source(power=1.0, absorption=1.0, beam_radius=0.1,
                               center=(0.5, 0.5))
    Q = np.asarray(src(g.cell_centers, 0.0)).reshape(n, n)
    # Peak should be at the cell closest to (0.5, 0.5)
    peak = np.unravel_index(np.argmax(Q), Q.shape)
    centres = np.asarray(g.cell_centers).reshape(n, n, 2)
    peak_xy = centres[peak]
    np.testing.assert_allclose(peak_xy, [0.5, 0.5], atol=g.dx)


def test_moving_gaussian_source_travels():
    n = 32
    g = cartesian_grid_2d(n, n, lx=1.0, ly=1.0)
    src = moving_gaussian_source(
        power=1.0, absorption=1.0, beam_radius=0.05,
        start=(0.25, 0.5), velocity=(1.0, 0.0),
    )
    Q0 = np.asarray(src(g.cell_centers, 0.0)).reshape(n, n)
    Q1 = np.asarray(src(g.cell_centers, 0.25)).reshape(n, n)
    # Peak at t=0 is near x=0.25; at t=0.25 near x=0.5
    p0 = np.unravel_index(np.argmax(Q0), Q0.shape)
    p1 = np.unravel_index(np.argmax(Q1), Q1.shape)
    centres = np.asarray(g.cell_centers).reshape(n, n, 2)
    x0 = centres[p0][0]
    x1 = centres[p1][0]
    assert x1 > x0
    assert x1 == pytest.approx(0.5, abs=2 * g.dx)


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------
def test_differentiable_wrt_alpha():
    n = 16
    g = cartesian_grid_2d(n, n, lx=1.0, ly=1.0)
    T0 = jnp.zeros((n, n, 1))
    T0 = T0.at[n // 2, n // 2, 0].set(1000.0)
    dt = diffusion_dt(g, alpha=1e-3, cfl=0.4)

    def loss(alpha_):
        T = step_thermal_scan(T0, g, dt, 20, alpha=alpha_, bc="neumann")
        return jnp.mean(T * T)

    grad = jax.grad(loss)(jnp.array(1e-3))
    assert bool(jnp.isfinite(grad))
    assert float(jnp.abs(grad)) > 1e-6


def test_differentiable_wrt_source_amplitude():
    n = 16
    g = cartesian_grid_2d(n, n, lx=1.0, ly=1.0)
    T0 = jnp.full((n, n, 1), 300.0)
    dt = diffusion_dt(g, alpha=1e-3, cfl=0.4)

    def loss(Q0):
        src = gaussian_heat_source(
            power=Q0, absorption=1.0, beam_radius=0.1, center=(0.5, 0.5),
        )
        T = step_thermal_scan(
            T0, g, dt, 30, alpha=1e-3, rho_cp=1.0, source=src, bc="neumann",
        )
        return jnp.sum(T)

    grad = jax.grad(loss)(jnp.array(100.0))
    assert bool(jnp.isfinite(grad))
    assert float(grad) > 0.0   # more power -> more heat


# ---------------------------------------------------------------------------
# ThermalConfig packing
# ---------------------------------------------------------------------------
def test_thermal_config_is_used_by_step():
    n = 8
    g = cartesian_grid_2d(n, n, lx=1.0, ly=1.0)
    T0 = jnp.full((n, n, 1), 1000.0)
    cfg = ThermalConfig(alpha=1e-3, rho_cp=1.0, bc="dirichlet", T_bc=300.0)
    dt = diffusion_dt(g, alpha=cfg.alpha, cfl=0.4)
    Tf = step_thermal(T0, g, dt, cfg)
    # Dirichlet walls pull the boundary toward 300.
    assert float(Tf[0, 0, 0]) < 1000.0


def test_return_history_shapes():
    n = 8
    g = cartesian_grid_2d(n, n, lx=1.0, ly=1.0)
    T0 = jnp.full((n, n, 1), 300.0)
    dt = diffusion_dt(g, alpha=1e-3, cfl=0.4)
    hist = step_thermal_scan(T0, g, dt, 5, alpha=1e-3, bc="neumann",
                             return_history=True)
    assert hist.shape == (6, n, n, 1)
