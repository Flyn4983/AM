"""Tests for the differentiable SPH module (2D & 3D)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from diffmech.methods.sph import (
    SPHState, SPHConfig,
    make_sph_state,
    cubic_kernel, cubic_kernel_grad,
    compute_density, pressure_from_density,
    compute_acceleration,
    step_sph, step_sph_scan,
    total_mass, kinetic_energy,
)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
def test_cubic_kernel_2d_partition_of_unity():
    """The 2D kernel integrated over the plane should be ~1."""
    h = 0.1
    dim = 2
    # Monte Carlo integration over a disk of radius 2h
    N = 100000
    key = jax.random.PRNGKey(0)
    r = 2.0 * h * jnp.sqrt(jax.random.uniform(key, (N,)))
    theta = 2.0 * jnp.pi * jax.random.uniform(jax.random.PRNGKey(1), (N,))
    x = r * jnp.cos(theta)
    y = r * jnp.sin(theta)
    W = jax.vmap(lambda ri: cubic_kernel(ri, h, dim))(jnp.sqrt(x ** 2 + y ** 2))
    # Area element for Monte Carlo: pi * (2h)^2 / N
    integral = jnp.sum(W) * (jnp.pi * (2.0 * h) ** 2 / N)
    assert float(integral) == pytest.approx(1.0, abs=0.02)


def test_cubic_kernel_3d_partition_of_unity():
    h = 0.1
    dim = 3
    N = 200000
    key = jax.random.PRNGKey(2)
    r = 2.0 * h * jnp.cbrt(jax.random.uniform(key, (N,)))
    # uniform direction
    d = jax.random.normal(jax.random.PRNGKey(3), (N, 3))
    d = d / jnp.linalg.norm(d, axis=-1, keepdims=True)
    pos = r[:, None] * d
    dist = jnp.linalg.norm(pos, axis=-1)
    W = jax.vmap(lambda ri: cubic_kernel(ri, h, dim))(dist)
    integral = jnp.sum(W) * ((4.0 / 3.0) * jnp.pi * (2.0 * h) ** 3 / N)
    assert float(integral) == pytest.approx(1.0, abs=0.03)


def test_cubic_kernel_grad_2d_radial():
    """Gradient should point radially outward (from j to i) and be negative
    for q < 1 (kernel is decreasing)."""
    h = 0.1
    dim = 2
    rij = jnp.array([0.05, 0.0])  # r/h = 0.5
    grad = cubic_kernel_grad(rij, h, dim)
    # At q=0.5, dW/dq < 0, so grad points in -rij direction (toward j)
    # Actually grad = dW/dq/h * rij/r. dW/dq < 0 at q=0.5, so grad is in -rij dir.
    assert float(grad[0]) < 0.0  # points in -x (toward j at origin)


# ---------------------------------------------------------------------------
# Density
# ---------------------------------------------------------------------------
def test_sph_2d_density_uniform():
    """Uniform particles should have approximately uniform density."""
    # Place particles on a regular grid
    nx, ny = 5, 5
    xs = jnp.linspace(0.2, 0.8, nx)
    ys = jnp.linspace(0.2, 0.8, ny)
    xx, yy = jnp.meshgrid(xs, ys, indexing="ij")
    pos = jnp.stack([xx.ravel(), yy.ravel()], axis=1)
    h = 0.15
    cfg = SPHConfig(h=h, rho0=1.0, c0=1.0, gamma_eos=1.0)
    state = make_sph_state(pos, mass=0.01)
    rho = compute_density(state, cfg)
    # Interior particles should have similar density
    assert jnp.all(rho > 0)
    assert jnp.std(rho) / jnp.mean(rho) < 0.3  # not too variable


def test_sph_3d_density_uniform():
    nx, ny, nz = 4, 4, 4
    xs = jnp.linspace(0.2, 0.8, nx)
    ys = jnp.linspace(0.2, 0.8, ny)
    zs = jnp.linspace(0.2, 0.8, nz)
    xx, yy, zz = jnp.meshgrid(xs, ys, zs, indexing="ij")
    pos = jnp.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    h = 0.2
    cfg = SPHConfig(h=h, rho0=1.0, c0=1.0, gamma_eos=1.0)
    state = make_sph_state(pos, mass=0.005)
    rho = compute_density(state, cfg)
    assert jnp.all(rho > 0)


# ---------------------------------------------------------------------------
# Pressure
# ---------------------------------------------------------------------------
def test_pressure_linear_eos():
    cfg = SPHConfig(rho0=1000.0, c0=50.0, gamma_eos=1.0)
    rho = jnp.array([1000.0, 1100.0, 900.0])
    p = pressure_from_density(rho, cfg)
    assert jnp.allclose(p, 50.0 ** 2 * (rho - 1000.0))


def test_pressure_tait_eos():
    cfg = SPHConfig(rho0=1000.0, c0=50.0, gamma_eos=7.0)
    rho = jnp.array([1000.0])  # at reference density, p = 0
    p = pressure_from_density(rho, cfg)
    assert jnp.allclose(p, 0.0, atol=1e-6)
    rho = jnp.array([1100.0])  # higher density => positive pressure
    p = pressure_from_density(rho, cfg)
    assert float(p[0]) > 0.0


# ---------------------------------------------------------------------------
# Momentum / acceleration
# ---------------------------------------------------------------------------
def test_sph_2d_pressure_pushes_apart():
    """Two particles close together should be pushed apart by pressure."""
    pos = jnp.array([[0.0, 0.0], [0.05, 0.0]])
    h = 0.1
    # rho0=0 ensures positive pressure whenever density > 0 (always repulsive).
    cfg = SPHConfig(h=h, rho0=0.0, c0=10.0, gamma_eos=1.0, nu=0.0)
    state = make_sph_state(pos, mass=0.01)
    accel = compute_acceleration(state, cfg)
    # Particle 0 should be pushed left (-x), particle 1 pushed right (+x)
    assert float(accel[0, 0]) < 0.0
    assert float(accel[1, 0]) > 0.0


def test_sph_3d_pressure_pushes_apart():
    pos = jnp.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]])
    h = 0.1
    cfg = SPHConfig(h=h, rho0=0.0, c0=10.0, gamma_eos=1.0, nu=0.0)
    state = make_sph_state(pos, mass=0.01)
    accel = compute_acceleration(state, cfg)
    assert float(accel[0, 0]) < 0.0
    assert float(accel[1, 0]) > 0.0


def test_sph_2d_momentum_conservation():
    """No external force => total momentum conserved."""
    key = jax.random.PRNGKey(10)
    N = 8
    pos = jax.random.uniform(key, (N, 2), minval=0.0, maxval=0.5)
    vel = 0.1 * jax.random.normal(jax.random.PRNGKey(11), (N, 2))
    h = 0.15
    cfg = SPHConfig(h=h, rho0=1.0, c0=5.0, gamma_eos=1.0, nu=0.1, gravity=0.0)
    state = make_sph_state(pos, velocity=vel, mass=0.01)
    state_f = step_sph_scan(state, dt=0.0005, n_steps=10, cfg=cfg)
    p0 = jnp.sum(state.mass[:, None] * state.velocity, axis=0)
    p1 = jnp.sum(state_f.mass[:, None] * state_f.velocity, axis=0)
    assert jnp.allclose(p0, p1, atol=1e-8)


def test_sph_3d_momentum_conservation():
    key = jax.random.PRNGKey(20)
    N = 6
    pos = jax.random.uniform(key, (N, 3), minval=0.0, maxval=0.5)
    vel = 0.1 * jax.random.normal(jax.random.PRNGKey(21), (N, 3))
    h = 0.2
    cfg = SPHConfig(h=h, rho0=1.0, c0=5.0, gamma_eos=1.0, nu=0.1, gravity=0.0)
    state = make_sph_state(pos, velocity=vel, mass=0.01)
    state_f = step_sph_scan(state, dt=0.0005, n_steps=8, cfg=cfg)
    p0 = jnp.sum(state.mass[:, None] * state.velocity, axis=0)
    p1 = jnp.sum(state_f.mass[:, None] * state_f.velocity, axis=0)
    assert jnp.allclose(p0, p1, atol=1e-8)


# ---------------------------------------------------------------------------
# Gravity
# ---------------------------------------------------------------------------
def test_sph_2d_free_fall():
    """A single particle under gravity accelerates downward."""
    pos = jnp.array([[0.5, 0.5]])
    h = 0.1
    cfg = SPHConfig(h=h, rho0=1.0, c0=1.0, gamma_eos=1.0, gravity=-9.81)
    state = make_sph_state(pos, mass=0.01)
    state_f = step_sph(state, dt=0.01, cfg=cfg)
    # v_y = 0 + dt * (-9.81) = -0.0981
    assert float(state_f.velocity[0, 1]) == pytest.approx(-0.0981, rel=1e-4)


def test_sph_3d_free_fall():
    pos = jnp.array([[0.5, 0.5, 0.5]])
    h = 0.1
    cfg = SPHConfig(h=h, rho0=1.0, c0=1.0, gamma_eos=1.0, gravity=-9.81)
    state = make_sph_state(pos, mass=0.01)
    state_f = step_sph(state, dt=0.01, cfg=cfg)
    assert float(state_f.velocity[0, 2]) == pytest.approx(-0.0981, rel=1e-4)


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------
def test_sph_2d_differentiable_wrt_position():
    pos = jnp.array([[0.0, 0.0], [0.05, 0.0], [0.1, 0.0]])
    h = 0.1
    cfg = SPHConfig(h=h, rho0=1.0, c0=10.0, gamma_eos=1.0, nu=0.0)

    def ke(p_init):
        state = make_sph_state(p_init, mass=0.01)
        sf = step_sph_scan(state, dt=0.001, n_steps=3, cfg=cfg)
        return kinetic_energy(sf)

    grad = jax.grad(ke)(pos)
    assert grad.shape == pos.shape
    assert jnp.all(jnp.isfinite(grad))
    assert jnp.max(jnp.abs(grad)) > 0.0


def test_sph_3d_differentiable_wrt_sound_speed():
    pos = jnp.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]])
    h = 0.1
    state = make_sph_state(pos, mass=0.01)

    def ke(c0_val):
        cfg = SPHConfig(h=h, rho0=1.0, c0=c0_val, gamma_eos=1.0, nu=0.0)
        sf = step_sph_scan(state, dt=0.001, n_steps=3, cfg=cfg)
        return kinetic_energy(sf)

    grad = jax.grad(ke)(jnp.array(10.0))
    assert jnp.all(jnp.isfinite(grad))
    assert float(grad) > 0.0  # higher c0 => more pressure => more KE


def test_sph_2d_jit_compatible():
    pos = jnp.array([[0.0, 0.0], [0.05, 0.0]])
    h = 0.1
    cfg = SPHConfig(h=h, rho0=1.0, c0=10.0, gamma_eos=1.0)
    state = make_sph_state(pos, mass=0.01)
    jitted = jax.jit(lambda s: step_sph(s, dt=0.001, cfg=cfg))
    s_new = jitted(state)
    assert jnp.all(jnp.isfinite(s_new.position))
    assert jnp.all(jnp.isfinite(s_new.velocity))
