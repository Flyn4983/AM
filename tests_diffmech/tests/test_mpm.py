"""Tests for the differentiable MLS-MPM module (2D & 3D)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from diffmech.methods.mpm import (
    MPMState, MPMConfig,
    make_mpm_state,
    step_mpm, step_mpm_scan,
    kinetic_energy, total_volume, center_of_mass,
)
from diffmech.methods.mpm.mpm import (
    _scatter_to_grid_nd, _gather_from_grid_nd,
    _bspline_weights,
    _neo_hookean_cauchy_stress, _fluid_pressure_stress,
)


# ---------------------------------------------------------------------------
# B-spline weights
# ---------------------------------------------------------------------------
def test_bspline_weights_partition_of_unity():
    """Quadratic B-spline weights must sum to 1 at every particle position."""
    origin = jnp.array([0.0])
    dx = 0.1
    for x in [0.001, 0.05, 0.1, 0.15, 0.199]:
        xp = jnp.array([x])
        _, w, _ = _bspline_weights(xp, origin, dx)
        assert abs(float(jnp.sum(w)) - 1.0) < 1e-6, (x, float(jnp.sum(w)))


# ---------------------------------------------------------------------------
# P2G / G2P consistency
# ---------------------------------------------------------------------------
def test_scatter_gather_total_mass_2d():
    """Scattering particle masses then gathering back must preserve total mass."""
    np.random.seed(1)
    N = 20
    pos = jnp.asarray(np.random.uniform(0.05, 0.95, (N, 2)))
    mass = jnp.asarray(np.random.uniform(0.5, 2.0, N))
    origin = jnp.array([0.0, 0.0])
    grid = _scatter_to_grid_nd(mass, pos, (16, 16), origin, 0.0625, 2)
    assert grid.shape == (16, 16)
    assert jnp.allclose(jnp.sum(grid), jnp.sum(mass), atol=1e-6)


def test_scatter_gather_total_mass_3d():
    np.random.seed(2)
    N = 15
    pos = jnp.asarray(np.random.uniform(0.05, 0.95, (N, 3)))
    mass = jnp.asarray(np.random.uniform(0.5, 2.0, N))
    origin = jnp.array([0.0, 0.0, 0.0])
    grid = _scatter_to_grid_nd(mass, pos, (8, 8, 8), origin, 0.125, 3)
    assert grid.shape == (8, 8, 8)
    assert jnp.allclose(jnp.sum(grid), jnp.sum(mass), atol=1e-6)


def test_gather_uniform_field_returns_constant():
    """Gathering a constant field must return the same constant at every particle."""
    np.random.seed(3)
    N = 10
    pos = jnp.asarray(np.random.uniform(0.1, 0.9, (N, 2)))
    origin = jnp.array([0.0, 0.0])
    grid_val = 3.7
    grid_field = jnp.full((16, 16, 2), grid_val)
    vals = _gather_from_grid_nd(grid_field, pos, origin, 0.0625, 2)
    assert vals.shape == (N, 2)
    assert jnp.allclose(vals, grid_val, atol=1e-5)


# ---------------------------------------------------------------------------
# Constitutive models
# ---------------------------------------------------------------------------
def test_neo_hookean_stress_at_identity_is_zero():
    """At F = I the Neo-Hookean Cauchy stress should be zero."""
    F = jnp.eye(2)
    sigma = _neo_hookean_cauchy_stress(F, E=100.0, nu=0.3)
    assert jnp.allclose(sigma, 0.0, atol=1e-6)


def test_fluid_pressure_stress_isotropic_compresses():
    """For J < 1 (compression) the fluid pressure stress should be positive."""
    F = 0.5 * jnp.eye(2)
    sigma = _fluid_pressure_stress(F, K=1000.0)
    J = jnp.linalg.det(F)
    p = 1000.0 * (J - 1.0)
    assert jnp.allclose(sigma, -p * jnp.eye(2), atol=1e-6)


# ---------------------------------------------------------------------------
# Free fall: a small cluster of particles under gravity
# ---------------------------------------------------------------------------
def _make_cluster_2d(n=3, spacing=0.05, center=(0.5, 0.5)):
    pos = []
    for i in range(n):
        for j in range(n):
            pos.append([center[0] + (i - (n - 1) / 2) * spacing,
                        center[1] + (j - (n - 1) / 2) * spacing])
    return jnp.asarray(pos)


def test_mpm_2d_free_fall_particles_fall_under_gravity():
    """A cluster of particles in free fall should move down (gravity < 0)."""
    pos = _make_cluster_2d()
    state = make_mpm_state(pos, mass=1.0)
    cfg = MPMConfig(
        grid_origin=(0.0, 0.0),
        grid_shape=(16, 16),
        dx=0.0625,
        dt=1e-3,
        youngs_modulus=1.0e4,
        poissons_ratio=0.0,
        gravity=-9.8,
    )
    new = step_mpm_scan(state, cfg, n_steps=5, bc="slip")
    # After a few steps under gravity the particles should have moved down.
    assert float(jnp.mean(new.position[:, 1])) < float(jnp.mean(state.position[:, 1]))
    assert jnp.all(jnp.isfinite(new.position))


def test_mpm_2d_volume_conserved_for_solid():
    """For a Neo-Hookean solid under free fall, total volume should be ~constant."""
    pos = _make_cluster_2d()
    state = make_mpm_state(pos, mass=1.0, volume=1.0)
    cfg = MPMConfig(
        grid_origin=(0.0, 0.0),
        grid_shape=(16, 16),
        dx=0.0625,
        dt=1e-3,
        youngs_modulus=1.0e4,
        poissons_ratio=0.3,
        gravity=0.0,  # no gravity: deformation should stay small
    )
    new = step_mpm_scan(state, cfg, n_steps=3, bc="slip")
    v0 = float(total_volume(state))
    v1 = float(total_volume(new))
    # Allow for small numerical drift due to single-step linearised update.
    assert abs(v1 - v0) < 0.05 * abs(v0) + 1e-6


def test_mpm_3d_runs_without_error():
    """A 3D MLS-MPM step should run and produce finite outputs."""
    np.random.seed(7)
    N = 10
    pos = jnp.asarray(np.random.uniform(0.1, 0.9, (N, 3)))
    state = make_mpm_state(pos, mass=1.0)
    cfg = MPMConfig(
        grid_origin=(0.0, 0.0, 0.0),
        grid_shape=(8, 8, 8),
        dx=0.125,
        dt=1e-4,
        youngs_modulus=1.0e4,
        poissons_ratio=0.3,
        gravity=-9.8,
    )
    new = step_mpm_scan(state, cfg, n_steps=3, bc="slip")
    assert new.position.shape == (N, 3)
    assert jnp.all(jnp.isfinite(new.position))
    assert jnp.all(jnp.isfinite(new.deformation))


def test_mpm_pic_transfer_runs():
    """The PIC (non-affine) transfer should also run without error."""
    pos = _make_cluster_2d()
    state = make_mpm_state(pos, mass=1.0)
    cfg = MPMConfig(
        grid_origin=(0.0, 0.0),
        grid_shape=(16, 16),
        dx=0.0625,
        dt=1e-3,
        youngs_modulus=1.0e4,
        poissons_ratio=0.3,
        gravity=-9.8,
        transfer="pic",
    )
    new = step_mpm(state, cfg, bc="slip")
    assert jnp.all(jnp.isfinite(new.position))


def test_mpm_fluid_material_runs():
    """The fluid EOS constitutive model should be selectable."""
    pos = _make_cluster_2d()
    state = make_mpm_state(pos, mass=1.0)
    cfg = MPMConfig(
        grid_origin=(0.0, 0.0),
        grid_shape=(16, 16),
        dx=0.0625,
        dt=1e-3,
        bulk_modulus=1.0e4,
        gravity=-9.8,
        material="fluid",
    )
    new = step_mpm(state, cfg, bc="slip")
    assert jnp.all(jnp.isfinite(new.position))


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------
def test_mpm_2d_differentiable_wrt_initial_position():
    """Gradient of final kinetic energy w.r.t. initial position must be finite."""
    pos = _make_cluster_2d()
    cfg = MPMConfig(
        grid_origin=(0.0, 0.0),
        grid_shape=(16, 16),
        dx=0.0625,
        dt=1e-3,
        youngs_modulus=1.0e4,
        poissons_ratio=0.3,
        gravity=-9.8,
    )

    def ke(p_init):
        state = make_mpm_state(p_init, mass=1.0)
        sf = step_mpm_scan(state, cfg, n_steps=3, bc="slip")
        return kinetic_energy(sf)

    grad = jax.grad(ke)(pos)
    assert grad.shape == pos.shape
    assert jnp.all(jnp.isfinite(grad))
    assert jnp.max(jnp.abs(grad)) > 0.0


def test_mpm_2d_differentiable_wrt_youngs_modulus():
    """Gradient of final KE w.r.t. Young's modulus must be finite."""
    pos = _make_cluster_2d()
    state = make_mpm_state(pos, mass=1.0)

    def ke(E):
        cfg = MPMConfig(
            grid_origin=(0.0, 0.0),
            grid_shape=(16, 16),
            dx=0.0625,
            dt=1e-3,
            youngs_modulus=E,
            poissons_ratio=0.3,
            gravity=-9.8,
        )
        sf = step_mpm_scan(state, cfg, n_steps=2, bc="slip")
        return kinetic_energy(sf)

    g = jax.grad(ke)(1.0e4)
    assert jnp.isfinite(g)


def test_mpm_jit_compiles():
    """The step function should JIT-compile without error."""
    pos = _make_cluster_2d()
    state = make_mpm_state(pos, mass=1.0)
    cfg = MPMConfig(
        grid_origin=(0.0, 0.0),
        grid_shape=(16, 16),
        dx=0.0625,
        dt=1e-3,
        youngs_modulus=1.0e4,
        poissons_ratio=0.3,
        gravity=-9.8,
    )
    step_jit = jax.jit(step_mpm)
    new = step_jit(state, cfg)
    assert jnp.all(jnp.isfinite(new.position))


def test_mpm_center_of_mass():
    """Center of mass should match the mass-weighted mean position."""
    pos = jnp.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    mass = jnp.asarray([1.0, 2.0, 3.0])
    state = make_mpm_state(pos, mass=mass)
    com = center_of_mass(state)
    expected = (mass[:, None] * pos).sum(axis=0) / mass.sum()
    assert jnp.allclose(com, expected, atol=1e-6)
