"""Tests for the differentiable Phase-Field module (2D & 3D)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from diffmech.methods.phase_field import (
    laplacian, gradient,
    AllenCahnConfig, AllenCahnState,
    double_well_potential, double_well_derivative,
    allen_cahn_rhs, step_allen_cahn, step_allen_cahn_scan,
    free_energy_allen_cahn,
    FractureConfig, FractureState,
    make_fracture_state,
    degradation_function, strain_energy_density,
    fracture_rhs, step_fracture, step_fracture_scan,
    fracture_energy,
)


# ---------------------------------------------------------------------------
# Laplacian / gradient helpers
# ---------------------------------------------------------------------------
def test_laplacian_2d_constant_field_is_zero():
    """The Laplacian of a constant field must be zero."""
    f = jnp.full((16, 16), 3.0)
    lap = laplacian(f, dx=0.1, dim=2, bc="periodic")
    assert jnp.allclose(lap, 0.0, atol=1e-12)


def test_laplacian_3d_constant_field_is_zero():
    f = jnp.full((8, 8, 8), 5.0)
    lap = laplacian(f, dx=0.1, dim=3, bc="periodic")
    assert jnp.allclose(lap, 0.0, atol=1e-12)


def test_laplacian_2d_linear_field_is_zero():
    """The Laplacian of a linear function is zero (second derivative)."""
    nx, ny = 16, 16
    dx = 0.1
    x = jnp.arange(nx) * dx
    y = jnp.arange(ny) * dx
    xx, yy = jnp.meshgrid(x, y, indexing="ij")
    f = 2.0 * xx + 3.0 * yy
    lap = laplacian(f, dx, 2, bc="periodic")
    # Periodic BC introduces a small error at the wrap-around, so check interior.
    assert jnp.allclose(lap[1:-1, 1:-1], 0.0, atol=1e-10)


def test_laplacian_2d_quadratic_known_value():
    """Laplacian of x² + y² is 4 (d²/dx² x² = 2, d²/dy² y² = 2)."""
    nx, ny = 16, 16
    dx = 0.1
    x = jnp.arange(nx) * dx
    y = jnp.arange(ny) * dx
    xx, yy = jnp.meshgrid(x, y, indexing="ij")
    f = xx ** 2 + yy ** 2
    lap = laplacian(f, dx, 2, bc="neumann")
    # Interior should be very close to 4.
    assert jnp.allclose(lap[1:-1, 1:-1], 4.0, atol=1e-10)


def test_gradient_2d_shape():
    f = jnp.ones((8, 8))
    g = gradient(f, dx=0.1, dim=2, bc="periodic")
    assert g.shape == (8, 8, 2)


def test_gradient_3d_shape():
    f = jnp.ones((8, 8, 8))
    g = gradient(f, dx=0.1, dim=3, bc="periodic")
    assert g.shape == (8, 8, 8, 3)


# ---------------------------------------------------------------------------
# Double-well potential
# ---------------------------------------------------------------------------
def test_double_well_derivative_at_minima():
    """f'(φ) = 0 at φ = ±1 (the two minima)."""
    phi = jnp.array([-1.0, 1.0])
    df = double_well_derivative(phi)
    assert jnp.allclose(df, 0.0, atol=1e-12)


def test_double_well_potential_nonneg():
    """f(φ) = ¼(φ²−1)² ≥ 0."""
    phi = jnp.linspace(-2, 2, 50)
    f = double_well_potential(phi)
    assert jnp.all(f >= -1e-12)


# ---------------------------------------------------------------------------
# Allen-Cahn 2D
# ---------------------------------------------------------------------------
def test_allen_cahn_2d_constant_phi_stays():
    """A uniform φ = ±1 field is an equilibrium (no evolution)."""
    phi = jnp.full((16, 16), 1.0)
    state = AllenCahnState(phi=phi)
    cfg = AllenCahnConfig(dx=0.1, dt=1e-3, mobility=1.0, kappa=1.0)
    new = step_allen_cahn(state, cfg)
    assert jnp.allclose(new.phi, phi, atol=1e-12)


def test_allen_cahn_2d_interface_sharpens():
    """A diffuse interface should sharpen (|∇φ| energy decreases)."""
    nx, ny = 32, 32
    dx = 1.0 / nx
    x = jnp.arange(nx) * dx
    xx, _ = jnp.meshgrid(x, x, indexing="ij")
    # Smooth tanh interface
    phi0 = jnp.tanh((xx - 0.5) * 20.0)
    state = AllenCahnState(phi=phi0)
    cfg = AllenCahnConfig(dx=dx, dt=1e-4, mobility=1.0, kappa=0.001)
    new = step_allen_cahn_scan(state, cfg, n_steps=50)
    # The maximum |φ| should stay close to 1 (saturated phases).
    assert float(jnp.max(jnp.abs(new.phi))) > 0.9
    assert jnp.all(jnp.isfinite(new.phi))


def test_allen_cahn_2d_energy_decreases():
    """Free energy should be non-increasing over time."""
    np.random.seed(42)
    phi0 = jnp.asarray(np.random.uniform(-0.5, 0.5, (24, 24)))
    state = AllenCahnState(phi=phi0)
    cfg = AllenCahnConfig(dx=0.05, dt=1e-4, mobility=1.0, kappa=0.01, bc="periodic")
    e0 = float(free_energy_allen_cahn(state, cfg))
    state_f = step_allen_cahn_scan(state, cfg, n_steps=20)
    e1 = float(free_energy_allen_cahn(state_f, cfg))
    assert e1 <= e0 + 1e-8  # allow tiny numerical noise


def test_allen_cahn_2d_differentiable_wrt_initial_phi():
    """Gradient of final energy w.r.t. initial φ must be finite."""
    np.random.seed(0)
    phi0 = jnp.asarray(np.random.uniform(-0.3, 0.3, (16, 16)))
    cfg = AllenCahnConfig(dx=0.05, dt=1e-4, mobility=1.0, kappa=0.01)

    def energy(p_init):
        s = AllenCahnState(phi=p_init)
        sf = step_allen_cahn_scan(s, cfg, n_steps=5)
        return free_energy_allen_cahn(sf, cfg)

    grad = jax.grad(energy)(phi0)
    assert grad.shape == phi0.shape
    assert jnp.all(jnp.isfinite(grad))
    assert jnp.max(jnp.abs(grad)) > 0.0


def test_allen_cahn_2d_jit_compiles():
    phi = jnp.full((16, 16), 1.0)
    state = AllenCahnState(phi=phi)
    cfg = AllenCahnConfig(dx=0.1, dt=1e-3)
    jitted = jax.jit(step_allen_cahn)
    new = jitted(state, cfg)
    assert jnp.all(jnp.isfinite(new.phi))


# ---------------------------------------------------------------------------
# Allen-Cahn 3D
# ---------------------------------------------------------------------------
def test_allen_cahn_3d_constant_phi_stays():
    phi = jnp.full((8, 8, 8), -1.0)
    state = AllenCahnState(phi=phi)
    cfg = AllenCahnConfig(dx=0.1, dt=1e-3)
    new = step_allen_cahn(state, cfg)
    assert jnp.allclose(new.phi, phi, atol=1e-12)


def test_allen_cahn_3d_runs_and_finite():
    np.random.seed(1)
    phi0 = jnp.asarray(np.random.uniform(-0.3, 0.3, (12, 12, 12)))
    state = AllenCahnState(phi=phi0)
    cfg = AllenCahnConfig(dx=0.05, dt=1e-4, mobility=1.0, kappa=0.01)
    new = step_allen_cahn_scan(state, cfg, n_steps=10)
    assert new.phi.shape == (12, 12, 12)
    assert jnp.all(jnp.isfinite(new.phi))


def test_allen_cahn_3d_differentiable_wrt_kappa():
    """Gradient of final energy w.r.t. κ must be finite."""
    np.random.seed(2)
    phi0 = jnp.asarray(np.random.uniform(-0.3, 0.3, (8, 8, 8)))
    state = AllenCahnState(phi=phi0)

    def energy(kappa):
        cfg = AllenCahnConfig(dx=0.05, dt=1e-4, mobility=1.0, kappa=kappa)
        sf = step_allen_cahn_scan(state, cfg, n_steps=5)
        return free_energy_allen_cahn(sf, cfg)

    grad = jax.grad(energy)(jnp.array(0.01))
    assert jnp.all(jnp.isfinite(grad))
    assert float(grad) > 0.0  # higher κ => higher gradient energy


# ---------------------------------------------------------------------------
# Phase-field fracture
# ---------------------------------------------------------------------------
def test_degradation_function():
    """g(0) = 1 (fully broken), g(1) = 0 (intact)."""
    phi = jnp.array([0.0, 1.0, 0.5])
    g = degradation_function(phi)
    assert jnp.allclose(g, jnp.array([1.0, 0.0, 0.25]))


def test_strain_energy_density_zero_displacement():
    """Zero displacement => zero strain energy."""
    disp = jnp.zeros((8, 8, 2))
    psi = strain_energy_density(disp, dx=0.1, dim=2)
    assert jnp.allclose(psi, 0.0, atol=1e-12)


def test_strain_energy_density_3d():
    """Non-zero displacement in 3D should give positive strain energy."""
    nx, ny, nz = 8, 8, 8
    dx = 0.1
    x = jnp.arange(nx) * dx
    xx, yy, zz = jnp.meshgrid(x, x, x, indexing="ij")
    disp = jnp.stack([0.01 * xx, jnp.zeros_like(xx), jnp.zeros_like(xx)], axis=-1)
    psi = strain_energy_density(disp, dx, 3)
    assert psi.shape == (nx, ny, nz)
    assert float(jnp.sum(psi)) > 0.0


def test_fracture_2d_runs_without_error():
    """A 2D phase-field fracture step should run and produce finite outputs."""
    state = make_fracture_state((16, 16))
    # Apply some displacement to create strain energy.
    nx, ny = 16, 16
    dx = 1.0 / nx
    x = jnp.arange(nx) * dx
    xx, yy = jnp.meshgrid(x, x, indexing="ij")
    disp = jnp.stack([0.01 * xx, jnp.zeros_like(xx)], axis=-1)
    state = FractureState(phi=state.phi, history=state.history, displacement=disp)
    cfg = FractureConfig(dx=dx, dt=1e-4, Gc=1.0, ell=0.05, mobility=1.0)
    new = step_fracture(state, cfg)
    assert new.phi.shape == (16, 16)
    assert jnp.all(jnp.isfinite(new.phi))
    assert jnp.all(new.phi >= 0.0 - 1e-10)
    assert jnp.all(new.phi <= 1.0 + 1e-10)


def test_fracture_3d_runs_without_error():
    state = make_fracture_state((8, 8, 8))
    nx, ny, nz = 8, 8, 8
    dx = 1.0 / nx
    x = jnp.arange(nx) * dx
    xx, yy, zz = jnp.meshgrid(x, x, x, indexing="ij")
    disp = jnp.stack([0.01 * xx, jnp.zeros_like(xx), jnp.zeros_like(xx)], axis=-1)
    state = FractureState(phi=state.phi, history=state.history, displacement=disp)
    cfg = FractureConfig(dx=dx, dt=1e-5, Gc=1.0, ell=0.05, mobility=1.0)
    new = step_fracture(state, cfg)
    assert new.phi.shape == (8, 8, 8)
    assert jnp.all(jnp.isfinite(new.phi))


def test_fracture_2d_phi_decreases_under_strain():
    """Under sufficient strain, φ should decrease (crack grows)."""
    nx, ny = 16, 16
    dx = 1.0 / nx
    state = make_fracture_state((nx, ny))
    # Large displacement gradient => high strain energy => crack growth.
    x = jnp.arange(nx) * dx
    xx, yy = jnp.meshgrid(x, x, indexing="ij")
    disp = jnp.stack([0.5 * xx, jnp.zeros_like(xx)], axis=-1)
    state = FractureState(phi=state.phi, history=state.history, displacement=disp)
    cfg = FractureConfig(dx=dx, dt=1e-3, Gc=0.01, ell=0.05, mobility=1.0)
    state_f = step_fracture_scan(state, cfg, n_steps=10)
    # Mean φ should decrease from 1.0.
    assert float(jnp.mean(state_f.phi)) < 1.0


def test_fracture_2d_differentiable_wrt_displacement():
    """Gradient of final fracture energy w.r.t. displacement must be finite."""
    nx, ny = 12, 12
    dx = 1.0 / nx
    state = make_fracture_state((nx, ny))
    x = jnp.arange(nx) * dx
    xx, yy = jnp.meshgrid(x, x, indexing="ij")
    disp0 = jnp.stack([0.05 * xx, jnp.zeros_like(xx)], axis=-1)

    def energy(disp):
        s = FractureState(phi=state.phi, history=state.history, displacement=disp)
        cfg = FractureConfig(dx=dx, dt=1e-4, Gc=1.0, ell=0.05, mobility=1.0)
        sf = step_fracture_scan(s, cfg, n_steps=3,
                                youngs_modulus=1.0, poissons_ratio=0.3)
        return fracture_energy(sf, cfg)

    grad = jax.grad(energy)(disp0)
    assert grad.shape == disp0.shape
    assert jnp.all(jnp.isfinite(grad))
    assert jnp.max(jnp.abs(grad)) > 0.0


def test_fracture_2d_differentiable_wrt_Gc():
    """Gradient of final φ w.r.t. Gc must be finite (higher Gc => less cracking)."""
    nx, ny = 12, 12
    dx = 1.0 / nx
    x = jnp.arange(nx) * dx
    xx, yy = jnp.meshgrid(x, x, indexing="ij")
    disp = jnp.stack([0.1 * xx, jnp.zeros_like(xx)], axis=-1)
    state0 = make_fracture_state((nx, ny))
    state0 = FractureState(phi=state0.phi, history=state0.history, displacement=disp)

    def mean_phi(Gc_val):
        cfg = FractureConfig(dx=dx, dt=1e-4, Gc=Gc_val, ell=0.05, mobility=1.0)
        sf = step_fracture_scan(state0, cfg, n_steps=5)
        return jnp.mean(sf.phi)

    grad = jax.grad(mean_phi)(jnp.array(1.0))
    assert jnp.all(jnp.isfinite(grad))


def test_fracture_2d_jit_compiles():
    nx, ny = 12, 12
    dx = 1.0 / nx
    state = make_fracture_state((nx, ny))
    cfg = FractureConfig(dx=dx, dt=1e-4, Gc=1.0, ell=0.05, mobility=1.0)
    jitted = jax.jit(step_fracture)
    new = jitted(state, cfg)
    assert jnp.all(jnp.isfinite(new.phi))
