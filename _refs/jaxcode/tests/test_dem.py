"""Tests for the differentiable DEM module (2D & 3D)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from diffmech.methods.dem import (
    DEMState, DEMConfig, Wall,
    make_dem_state, step_dem, step_dem_scan, kinetic_energy,
)


# ---------------------------------------------------------------------------
# State construction
# ---------------------------------------------------------------------------
def test_dem_state_2d_construction():
    pos = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.8]])
    s = make_dem_state(pos, radius=0.1, mass=2.0)
    assert s.n_particles == 3
    assert s.dim == 2
    assert jnp.allclose(s.radius, 0.1)
    assert jnp.allclose(s.mass, 2.0)
    assert s.velocity.shape == (3, 2)
    assert jnp.allclose(s.velocity, 0.0)


def test_dem_state_3d_construction():
    pos = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    s = make_dem_state(pos, radius=0.2)
    assert s.dim == 3
    assert s.velocity.shape == (2, 3)


# ---------------------------------------------------------------------------
# Contact physics
# ---------------------------------------------------------------------------
def test_dem_2d_no_contact_no_force():
    """Two non-overlapping particles should not exert any force."""
    pos = jnp.array([[0.0, 0.0], [5.0, 0.0]])
    s = make_dem_state(pos, radius=0.1)
    cfg = DEMConfig(k=1000.0, gamma=0.0)
    s_new = step_dem(s, dt=0.001, cfg=cfg)
    # No contact => velocity unchanged (no gravity)
    assert jnp.allclose(s_new.velocity, s.velocity, atol=1e-10)
    # Position advances by v * dt = 0
    assert jnp.allclose(s_new.position, s.position, atol=1e-10)


def test_dem_2d_head_on_collision_repulsion():
    """Two overlapping particles should repel each other."""
    pos = jnp.array([[0.0, 0.0], [0.15, 0.0]])  # overlap = 2*0.1 - 0.15 = 0.05
    s = make_dem_state(pos, radius=0.1, mass=1.0)
    cfg = DEMConfig(k=1000.0, gamma=0.0)
    s_new = step_dem(s, dt=0.001, cfg=cfg)
    # Particle 0 pushed left (-x), particle 1 pushed right (+x)
    assert float(s_new.velocity[0, 0]) < 0.0
    assert float(s_new.velocity[1, 0]) > 0.0
    # Momentum is conserved (equal masses, equal and opposite forces)
    total_p = jnp.sum(s_new.mass[:, None] * s_new.velocity, axis=0)
    assert jnp.allclose(total_p, 0.0, atol=1e-10)


def test_dem_3d_head_on_collision_repulsion():
    pos = jnp.array([[0.0, 0.0, 0.0], [0.15, 0.0, 0.0]])
    s = make_dem_state(pos, radius=0.1, mass=1.0)
    cfg = DEMConfig(k=1000.0, gamma=0.0)
    s_new = step_dem(s, dt=0.001, cfg=cfg)
    assert float(s_new.velocity[0, 0]) < 0.0
    assert float(s_new.velocity[1, 0]) > 0.0
    total_p = jnp.sum(s_new.mass[:, None] * s_new.velocity, axis=0)
    assert jnp.allclose(total_p, 0.0, atol=1e-10)


def test_dem_2d_momentum_conservation_multi_step():
    """Total momentum is conserved (no external forces) over multiple steps."""
    key = jax.random.PRNGKey(42)
    N = 5
    pos = jax.random.uniform(key, (N, 2), minval=0.0, maxval=0.5)
    vel = 0.1 * jax.random.normal(jax.random.PRNGKey(43), (N, 2))
    s = make_dem_state(pos, velocity=vel, radius=0.08, mass=1.0)
    cfg = DEMConfig(k=500.0, gamma=0.5)
    s_f = step_dem_scan(s, dt=0.0005, n_steps=20, cfg=cfg)
    p0 = jnp.sum(s.mass[:, None] * s.velocity, axis=0)
    p1 = jnp.sum(s_f.mass[:, None] * s_f.velocity, axis=0)
    assert jnp.allclose(p0, p1, atol=1e-8)


def test_dem_3d_momentum_conservation_multi_step():
    key = jax.random.PRNGKey(52)
    N = 4
    pos = jax.random.uniform(key, (N, 3), minval=0.0, maxval=0.5)
    vel = 0.1 * jax.random.normal(jax.random.PRNGKey(53), (N, 3))
    s = make_dem_state(pos, velocity=vel, radius=0.07, mass=1.0)
    cfg = DEMConfig(k=500.0, gamma=0.5)
    s_f = step_dem_scan(s, dt=0.0005, n_steps=15, cfg=cfg)
    p0 = jnp.sum(s.mass[:, None] * s.velocity, axis=0)
    p1 = jnp.sum(s_f.mass[:, None] * s_f.velocity, axis=0)
    assert jnp.allclose(p0, p1, atol=1e-8)


# ---------------------------------------------------------------------------
# Wall confinement
# ---------------------------------------------------------------------------
def test_dem_2d_wall_reflection():
    """A particle moving toward a wall should be pushed back."""
    pos = jnp.array([[0.95, 0.0]])
    vel = jnp.array([[1.0, 0.0]])
    s = make_dem_state(pos, velocity=vel, radius=0.1, mass=1.0)
    cfg = DEMConfig(k=1000.0, gamma=0.0)
    wall = Wall(normal=jnp.array([1.0, 0.0]), offset=1.0)
    s_new = step_dem(s, dt=0.001, cfg=cfg, walls=[wall])
    # The wall pushes the particle in -x
    assert float(s_new.velocity[0, 0]) < float(vel[0, 0])


def test_dem_3d_wall_reflection():
    pos = jnp.array([[0.0, 0.0, 0.95]])
    vel = jnp.array([[0.0, 0.0, 1.0]])
    s = make_dem_state(pos, velocity=vel, radius=0.1, mass=1.0)
    cfg = DEMConfig(k=1000.0, gamma=0.0)
    wall = Wall(normal=jnp.array([0.0, 0.0, 1.0]), offset=1.0)
    s_new = step_dem(s, dt=0.001, cfg=cfg, walls=[wall])
    assert float(s_new.velocity[0, 2]) < float(vel[0, 2])


# ---------------------------------------------------------------------------
# Gravity
# ---------------------------------------------------------------------------
def test_dem_2d_free_fall():
    """A single particle under gravity accelerates downward (last axis = y)."""
    pos = jnp.array([[0.5, 0.5]])
    s = make_dem_state(pos, radius=0.01, mass=1.0)
    cfg = DEMConfig(k=100.0, gamma=0.0, gravity=-9.81)
    s_new = step_dem(s, dt=0.01, cfg=cfg)
    # v_y = 0 + dt * g = -0.0981
    assert float(s_new.velocity[0, 1]) == pytest.approx(-0.0981, rel=1e-4)


def test_dem_3d_free_fall():
    pos = jnp.array([[0.5, 0.5, 0.5]])
    s = make_dem_state(pos, radius=0.01, mass=1.0)
    cfg = DEMConfig(k=100.0, gamma=0.0, gravity=-9.81)
    s_new = step_dem(s, dt=0.01, cfg=cfg)
    assert float(s_new.velocity[0, 2]) == pytest.approx(-0.0981, rel=1e-4)


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------
def test_dem_2d_differentiable_wrt_position():
    """Gradient of final kinetic energy w.r.t. initial position is finite
    and non-trivial when particles are in contact."""
    pos = jnp.array([[0.0, 0.0], [0.15, 0.0]])  # overlap
    s0 = make_dem_state(pos, radius=0.1, mass=1.0)
    cfg = DEMConfig(k=1000.0, gamma=0.0)

    def ke(p_init):
        state = make_dem_state(p_init, radius=0.1, mass=1.0)
        sf = step_dem_scan(state, dt=0.001, n_steps=5, cfg=cfg)
        return kinetic_energy(sf)

    grad = jax.grad(ke)(pos)
    assert grad.shape == pos.shape
    assert jnp.all(jnp.isfinite(grad))
    assert jnp.max(jnp.abs(grad)) > 0.0


def test_dem_3d_differentiable_wrt_stiffness():
    """Gradient of final kinetic energy w.r.t. contact stiffness is finite."""
    pos = jnp.array([[0.0, 0.0, 0.0], [0.15, 0.0, 0.0]])
    s0 = make_dem_state(pos, radius=0.1, mass=1.0)

    def ke(k_val):
        cfg = DEMConfig(k=k_val, gamma=0.0)
        sf = step_dem_scan(s0, dt=0.001, n_steps=5, cfg=cfg)
        return kinetic_energy(sf)

    grad = jax.grad(ke)(jnp.array(1000.0))
    assert jnp.all(jnp.isfinite(grad))
    # Higher stiffness => more repulsion => more KE => positive gradient
    assert float(grad) > 0.0


def test_dem_2d_differentiable_wrt_gravity():
    pos = jnp.array([[0.5, 0.9]])  # near the floor
    s0 = make_dem_state(pos, velocity=jnp.array([[0.0, 0.0]]),
                        radius=0.05, mass=1.0)
    walls = [Wall(normal=jnp.array([0.0, 1.0]), offset=1.0)]

    def ke(g_val):
        cfg = DEMConfig(k=1000.0, gamma=0.0, gravity=g_val)
        sf = step_dem_scan(s0, dt=0.001, n_steps=10, cfg=cfg, walls=walls)
        return kinetic_energy(sf)

    grad = jax.grad(ke)(jnp.array(-9.81))
    assert jnp.all(jnp.isfinite(grad))


# ---------------------------------------------------------------------------
# Energy dissipation
# ---------------------------------------------------------------------------
def test_dem_2d_damping_dissipates_energy():
    """With viscous damping, kinetic energy should decrease over time."""
    pos = jnp.array([[0.0, 0.0], [0.12, 0.0]])
    vel = jnp.array([[0.0, 0.0], [-1.0, 0.0]])
    s = make_dem_state(pos, velocity=vel, radius=0.1, mass=1.0)
    cfg = DEMConfig(k=2000.0, gamma=10.0)
    ke0 = float(kinetic_energy(s))
    s_f = step_dem_scan(s, dt=0.0001, n_steps=50, cfg=cfg)
    ke1 = float(kinetic_energy(s_f))
    assert ke1 < ke0


def test_dem_3d_jit_compatible():
    """The whole step should be jittable."""
    pos = jnp.array([[0.0, 0.0, 0.0], [0.15, 0.0, 0.0]])
    s = make_dem_state(pos, radius=0.1, mass=1.0)
    cfg = DEMConfig(k=1000.0, gamma=0.0)
    jitted = jax.jit(lambda st: step_dem(st, dt=0.001, cfg=cfg))
    s_new = jitted(s)
    assert jnp.all(jnp.isfinite(s_new.position))
    assert jnp.all(jnp.isfinite(s_new.velocity))
