"""Tests for solvers and optimization."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from diffmech.solvers import (
    newton_solve, explicit_euler_step, semi_implicit_euler_step,
    velocity_verlet_step, rk4_step,
    DirichletBC, apply_dirichlet,
)
from diffmech.optimize import optimize_params_optax, optimize_params_gd


# --- Newton -----------------------------------------------------------------
def test_newton_solves_linear_system():
    """For R(u) = A u - b with A nonsingular, Newton converges in 1 step."""
    A = jnp.array([[4.0, 1.0], [1.0, 3.0]])
    b = jnp.array([1.0, 2.0])
    R = lambda u: A @ u - b
    res = newton_solve(R, jnp.zeros(2), tol=1e-12, max_iter=10)
    assert bool(res.converged)
    assert jnp.allclose(res.u, jnp.linalg.solve(A, b), atol=1e-10)


def test_newton_solves_nonlinear_scalar():
    """Find root of R(u) = u^2 - 4 = 0 starting at u=3; root is u=2."""
    R = lambda u: u ** 2 - 4.0
    res = newton_solve(R, jnp.array(3.0), tol=1e-12, max_iter=20)
    assert bool(res.converged)
    assert np.isclose(float(res.u), 2.0, atol=1e-9)


def test_newton_jit_compatible():
    R = lambda u: jnp.sin(u) - 0.5
    R_jit = jax.jit(R)
    res = newton_solve(R_jit, jnp.array(1.0), tol=1e-10, max_iter=20)
    assert bool(res.converged)
    # Solution: u = arcsin(0.5) = pi/6
    assert np.isclose(float(res.u), np.pi / 6, atol=1e-8)


# --- Time integrators -------------------------------------------------------
def test_explicit_euler_constant_velocity():
    """y' = v (constant) -> y(t) = y0 + v*t."""
    rhs = lambda t, y: jnp.array([2.0])
    y0 = jnp.array([1.0])
    y1 = explicit_euler_step(rhs, 0.0, y0, 0.5)
    assert jnp.allclose(y1, jnp.array([2.0]))  # 1 + 0.5*2


def test_velocity_verlet_harmonic_oscillator():
    """x'' = -x (omega=1) -> x(t) = cos(t), returns near 1 at t=2*pi."""
    omega = 1.0
    compute_accel = lambda x: -omega ** 2 * x
    x = jnp.array([1.0])
    v = jnp.array([0.0])
    a = compute_accel(x)
    dt = 0.001
    n_steps = int(round(2 * jnp.pi / dt))  # one period
    for _ in range(n_steps):
        x, v, a = velocity_verlet_step(compute_accel, x, v, a, dt)
    # After one period, x should be back near 1.0 (Verlet is symplectic, O(dt^2))
    assert jnp.allclose(x, jnp.array([1.0]), atol=1e-3)


def test_semi_implicit_euler_free_fall():
    """Constant force F = m*g -> x grows quadratically (semi-implicit Euler)."""
    g = 9.81
    F = lambda x: jnp.array([[0.0, -g]])
    m = jnp.array([1.0])
    x = jnp.array([[0.0, 0.0]])
    v = jnp.array([[0.0, 0.0]])
    dt = 0.001
    n = 1000
    t_total = n * dt
    for _ in range(n):
        x, v = semi_implicit_euler_step(F, x, v, m, dt)
    # Analytic: y = -0.5*g*t^2; semi-implicit Euler has O(dt) error per step
    # -> total error ~ 0.5*dt*g*t = 0.0049 at dt=0.001, t=1.0
    assert np.isclose(float(x[0, 1]), -0.5 * g * t_total ** 2, atol=1e-2)


def test_rk4_step_accuracy():
    """RK4 has O(dt^5) local error -> for y'=y, y(1)=e (high accuracy)."""
    rhs = lambda t, y: y
    y = jnp.array([1.0])
    dt = 0.1
    n = 10
    for _ in range(n):
        y = rk4_step(rhs, 0.0, y, dt)
    assert np.isclose(float(y[0]), np.e, rtol=1e-5)


# --- Dirichlet BCs ----------------------------------------------------------
def test_apply_dirichlet_pins_dofs():
    K = jnp.eye(3) * 2.0
    F = jnp.array([1.0, 2.0, 3.0])
    bc = DirichletBC(dofs=jnp.array([0]), values=jnp.array([5.0]))
    K2, F2 = apply_dirichlet(K, F, [bc])
    # Row 0 should be e_0 and F2[0] = 5
    assert np.isclose(float(K2[0, 0]), 1.0)
    assert np.isclose(float(K2[0, 1]), 0.0)
    assert np.isclose(float(F2[0]), 5.0)
    # Other rows unchanged
    assert np.isclose(float(K2[1, 1]), 2.0)


# --- Optimization -----------------------------------------------------------
def test_optimize_params_gd_quadratic():
    """Minimize L(theta) = (theta - 3)^2 -> theta -> 3."""
    L = lambda theta: jnp.sum((theta - 3.0) ** 2)
    theta0 = jnp.array(0.0)
    theta, _ = optimize_params_gd(L, theta0, lr=0.1, n_iter=200)
    assert np.isclose(float(theta), 3.0, atol=1e-4)


def test_optimize_params_optax_rosenbrock_2d():
    """Minimize Rosenbrock f(x,y) = (1-x)^2 + 100(y - x^2)^2, min at (1,1)."""
    def L(p):
        x, y = p[0], p[1]
        return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
    p0 = jnp.array([-1.2, 1.0])
    p_opt, hist = optimize_params_optax(L, p0, n_iter=10000, lr=1e-2)
    assert jnp.allclose(p_opt, jnp.array([1.0, 1.0]), atol=1e-2)
    assert hist[-1] < 1e-4


def test_optimize_grad_through_simulation():
    """Verify that jax.grad correctly differentiates through a forward simulation."""
    def simulate(k):
        # 1D harmonic oscillator: x(t) = cos(sqrt(k) t)
        omega = jnp.sqrt(k)
        return jnp.cos(omega * 1.0)
    # d/dk cos(sqrt(k)*1) = -sin(sqrt(k))*0.5/sqrt(k)
    # at k=1: -sin(1)*0.5 = -0.4207
    g = jax.grad(simulate)(jnp.array(1.0))
    expected = -jnp.sin(1.0) * 0.5
    assert np.isclose(float(g), float(expected), atol=1e-6)
