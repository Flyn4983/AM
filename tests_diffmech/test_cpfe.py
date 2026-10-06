"""Tests for the crystal-plasticity FEM module (2D & 3D).

Covers:
- Slip-system geometry (orthogonality, unit norms, system counts)
- Constitutive model (zero-strain → no slip, slip activation, hardening,
  stress relaxation, sub-stepping convergence, autodiff tangent)
- 2D CPFE solver (simple shear on a single-slip crystal)
- 3D CPFE solver (uniaxial tension on an FCC crystal)
- End-to-end differentiability (``jax.grad`` w.r.t. material parameters)
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from diffmech.core import rectangular_mesh2d, rectangular_mesh3d
from diffmech.solvers import DirichletBC
from diffmech.methods.cpfe import (
    CrystalPlasticity, CPState, CPFEProblem, solve_cpfe,
    cp_stress, cp_update, cp_update_step, slip_rates,
    cp_stress_and_tangent, elastic_tangent_voigt,
    schmid_tensors, resolved_shear_stress,
    double_slip_2d, single_slip_2d, fcc_slip_systems, bcc_slip_systems,
    average_equivalent_plastic_strain, total_slip,
)


# ---------------------------------------------------------------------------
# Slip-system geometry
# ---------------------------------------------------------------------------
class TestSlipSystems:
    def test_fcc_orthogonality_and_count(self):
        b, n = fcc_slip_systems()
        assert b.shape == (12, 3)
        dot = jnp.einsum("si,si->s", b, n)
        assert jnp.allclose(dot, 0.0, atol=1e-12)

    def test_fcc_unit_vectors(self):
        b, n = fcc_slip_systems()
        assert jnp.allclose(jnp.linalg.norm(b, axis=1), 1.0)
        assert jnp.allclose(jnp.linalg.norm(n, axis=1), 1.0)

    def test_bcc_orthogonality_and_count(self):
        b, n = bcc_slip_systems()
        assert b.shape == (12, 3)
        dot = jnp.einsum("si,si->s", b, n)
        assert jnp.allclose(dot, 0.0, atol=1e-12)

    def test_double_slip_2d_orthogonality(self):
        b, n = double_slip_2d(60.0)
        assert b.shape == (2, 2)
        dot = jnp.einsum("si,si->s", b, n)
        assert jnp.allclose(dot, 0.0, atol=1e-12)

    def test_schmid_tensors_symmetric(self):
        b, n = fcc_slip_systems()
        P = schmid_tensors(b, n)
        assert P.shape == (12, 3, 3)
        # Symmetry: P == P^T
        assert jnp.allclose(P, jnp.transpose(P, (0, 2, 1)))

    def test_resolved_shear_stress_single_slip(self):
        """For a single slip system b=(1,0), n=(0,1), τ = σ_xy."""
        b = jnp.array([[1.0, 0.0]])
        n = jnp.array([[0.0, 1.0]])
        P = schmid_tensors(b, n)
        sigma = jnp.array([[0.0, 5.0], [5.0, 0.0]])
        tau = resolved_shear_stress(sigma, P)
        # P = [[0, 0.5], [0.5, 0]], τ = 2 * σ_xy * 0.5 = σ_xy = 5
        assert jnp.allclose(tau, jnp.array([5.0]))


# ---------------------------------------------------------------------------
# Constitutive model
# ---------------------------------------------------------------------------
def _single_slip_x_shear_mat(**overrides):
    """Material with one slip system b=(1,0), n=(0,1) — activates on σ_xy."""
    b = jnp.array([[1.0, 0.0]])
    n = jnp.array([[0.0, 1.0]])
    defaults = dict(
        E=1000.0, nu=0.3,
        slip_directions=b, slip_normals=n,
        gamma_dot0=0.01, m=0.5, g0=1.0, h0=10.0, g_sat=100.0,
    )
    defaults.update(overrides)
    return CrystalPlasticity(**defaults)


class TestConstitutive:
    def test_zero_strain_no_slip(self):
        mat = _single_slip_x_shear_mat()
        state0 = mat.initial_state()
        eps = jnp.zeros((2, 2))
        state1 = cp_update(eps, state0, mat, dt=0.1, n_sub=5)
        assert jnp.allclose(state1.gamma, 0.0, atol=1e-20)
        assert jnp.allclose(state1.eps_p, 0.0, atol=1e-20)
        # CRSS unchanged (no slip → no hardening)
        assert jnp.allclose(state1.g, mat.g0)

    def test_shear_strain_activates_slip(self):
        """Simple shear ε_xy = γ/2 on the (1,0)/(0,1) slip system → slip > 0."""
        mat = _single_slip_x_shear_mat()
        state0 = mat.initial_state()
        gamma_shear = 0.01
        eps = jnp.array([[0.0, gamma_shear / 2.0], [gamma_shear / 2.0, 0.0]])
        state1 = cp_update(eps, state0, mat, dt=0.1, n_sub=10)
        # Slip should be positive
        assert float(state1.gamma[0]) > 0.0
        # Plastic strain should be non-zero
        assert float(jnp.max(jnp.abs(state1.eps_p))) > 0.0

    def test_hardening_increases_crss(self):
        """After slip, the CRSS g should increase (work hardening)."""
        mat = _single_slip_x_shear_mat()
        state0 = mat.initial_state()
        gamma_shear = 0.01
        eps = jnp.array([[0.0, gamma_shear / 2.0], [gamma_shear / 2.0, 0.0]])
        state1 = cp_update(eps, state0, mat, dt=0.1, n_sub=10)
        assert float(state1.g[0]) > float(state0.g[0])

    def test_stress_relaxes_with_plasticity(self):
        """Under fixed strain, stress decreases as plastic strain accumulates."""
        mat = _single_slip_x_shear_mat()
        state0 = mat.initial_state()
        gamma_shear = 0.01
        eps = jnp.array([[0.0, gamma_shear / 2.0], [gamma_shear / 2.0, 0.0]])
        sigma0 = cp_stress(eps, state0, mat)
        state1 = cp_update(eps, state0, mat, dt=0.5, n_sub=50)
        sigma1 = cp_stress(eps, state1, mat)
        # |σ_xy| should decrease (stress relaxation)
        assert float(jnp.abs(sigma1[0, 1])) < float(jnp.abs(sigma0[0, 1]))

    def test_substep_convergence(self):
        """More sub-steps → more accurate (converged) plastic strain."""
        mat = _single_slip_x_shear_mat()
        state0 = mat.initial_state()
        gamma_shear = 0.005
        eps = jnp.array([[0.0, gamma_shear / 2.0], [gamma_shear / 2.0, 0.0]])
        dt = 0.2
        s_coarse = cp_update(eps, state0, mat, dt=dt, n_sub=1)
        s_fine = cp_update(eps, state0, mat, dt=dt, n_sub=200)
        # Both should be positive; fine is the reference
        assert float(s_fine.gamma[0]) > 0.0
        # Coarse overestimates slip (explicit Euler) — check they're in the
        # same ballpark and fine is smaller (more accurate, less overshoot)
        assert float(s_fine.gamma[0]) <= float(s_coarse.gamma[0]) + 1e-6

    def test_tangent_is_elastic_for_explicit_update(self):
        """The algorithmic tangent (dσ/dε) equals the elastic tangent C."""
        mat = _single_slip_x_shear_mat()
        state0 = mat.initial_state()
        eps = jnp.array([[0.001, 0.0], [0.0, 0.0]])
        _, tangent = cp_stress_and_tangent(eps, state0, mat)
        # σ = C:(ε - ε_p), with ε_p fixed → dσ/dε = C (4th-order elastic
        # tensor).  In full-tensor convention σ_xy = 2μ·ε_xy, so the tensor
        # tangent component dσ_xy/dε_xy = 2μ (μ in Voigt/engineering form).
        assert jnp.isclose(tangent[0, 0, 0, 0], mat.lam + 2 * mat.mu, rtol=1e-6)
        assert jnp.isclose(tangent[0, 1, 0, 1], 2.0 * mat.mu, rtol=1e-6)
        # Cross coupling: dσ_xx/dε_yy = lam
        assert jnp.isclose(tangent[0, 0, 1, 1], mat.lam, rtol=1e-6)

    def test_elastic_tangent_voigt_2d(self):
        mat = _single_slip_x_shear_mat()
        C = elastic_tangent_voigt(mat, dim=2)
        assert C.shape == (3, 3)
        assert jnp.isclose(C[0, 0], mat.lam + 2 * mat.mu)
        assert jnp.isclose(C[2, 2], mat.mu)

    def test_constitutive_differentiable_wrt_g0(self):
        """Slip is differentiable w.r.t. the initial CRSS g0."""
        gamma_shear = 0.01
        eps = jnp.array([[0.0, gamma_shear / 2.0], [gamma_shear / 2.0, 0.0]])

        def slip(g0):
            mat = _single_slip_x_shear_mat(g0=g0)
            s0 = mat.initial_state()
            s1 = cp_update(eps, s0, mat, dt=0.1, n_sub=10)
            return jnp.sum(s1.gamma)

        g = jax.grad(slip)(jnp.array(1.0))
        # Lower g0 → more slip, so d(slip)/d(g0) < 0
        assert jnp.isfinite(g)
        assert float(g) < 0.0


# ---------------------------------------------------------------------------
# 2D CPFE solver
# ---------------------------------------------------------------------------
def _shear_mesh_2d(nx=2, ny=2, lx=1.0, ly=1.0):
    return rectangular_mesh2d(nx=nx, ny=ny, lx=lx, ly=ly, cell_type="quad4")


class TestCPFE2D:
    def test_simple_shear_plastic_strain_develops(self):
        """Apply simple shear on a single-slip crystal; plastic strain grows."""
        b = jnp.array([[1.0, 0.0]])   # slip in x
        n = jnp.array([[0.0, 1.0]])   # normal in y → τ = σ_xy
        mat = CrystalPlasticity(
            E=1000.0, nu=0.3,
            slip_directions=b, slip_normals=n,
            gamma_dot0=0.01, m=0.5, g0=1.0, h0=10.0, g_sat=100.0,
        )
        mesh = _shear_mesh_2d(nx=2, ny=2, lx=1.0, ly=1.0)
        nodes_y = np.asarray(mesh.nodes)[:, 1]
        bottom = jnp.where(nodes_y < 1e-9)[0]
        top = jnp.where(nodes_y > 1.0 - 1e-9)[0]
        gamma_target = 0.01
        bcs = (
            DirichletBC(dofs=jnp.concatenate([bottom * 2, bottom * 2 + 1]),
                        values=jnp.zeros(2 * len(bottom))),
            DirichletBC(dofs=jnp.concatenate([top * 2, top * 2 + 1]),
                        values=jnp.concatenate([
                            jnp.full(len(top), gamma_target, dtype=jnp.float64),
                            jnp.zeros(len(top), dtype=jnp.float64),
                        ])),
        )
        problem = CPFEProblem(mesh=mesh, material=mat, dirichlet_bcs=bcs, dim=2)
        load_factors = jnp.linspace(0.0, 1.0, 20)
        sol = solve_cpfe(problem, load_factors, dt=0.1, n_sub=5)
        # Final displacement at top should match the BC (load_factor=1)
        assert jnp.allclose(sol.u[top * 2], gamma_target, atol=1e-6)
        # Plastic strain should be non-zero
        eps_eq = average_equivalent_plastic_strain(sol.state)
        assert float(eps_eq) > 0.0

    def test_zero_load_zero_plastic_strain(self):
        """With zero load (load_factors all 0), no plastic strain develops."""
        b = jnp.array([[1.0, 0.0]])
        n = jnp.array([[0.0, 1.0]])
        mat = CrystalPlasticity(
            E=1000.0, nu=0.3,
            slip_directions=b, slip_normals=n,
            gamma_dot0=0.01, m=0.5, g0=1.0, h0=10.0, g_sat=100.0,
        )
        mesh = _shear_mesh_2d(nx=2, ny=2)
        nodes_y = np.asarray(mesh.nodes)[:, 1]
        bottom = jnp.where(nodes_y < 1e-9)[0]
        bcs = (
            DirichletBC(dofs=jnp.concatenate([bottom * 2, bottom * 2 + 1]),
                        values=jnp.zeros(2 * len(bottom))),
        )
        problem = CPFEProblem(mesh=mesh, material=mat, dirichlet_bcs=bcs, dim=2)
        load_factors = jnp.zeros(5)
        sol = solve_cpfe(problem, load_factors, dt=0.1, n_sub=5)
        assert jnp.allclose(sol.u, 0.0, atol=1e-12)
        assert jnp.allclose(sol.state.eps_p, 0.0, atol=1e-12)

    def test_2d_differentiable_wrt_g0(self):
        """End-to-end gradient of mean plastic strain w.r.t. g0."""
        b = jnp.array([[1.0, 0.0]])
        n = jnp.array([[0.0, 1.0]])
        mesh = _shear_mesh_2d(nx=2, ny=2, lx=1.0, ly=1.0)
        nodes_y = np.asarray(mesh.nodes)[:, 1]
        bottom = jnp.where(nodes_y < 1e-9)[0]
        top = jnp.where(nodes_y > 1.0 - 1e-9)[0]
        gamma_target = 0.005

        def mean_eps_eq(g0):
            mat = CrystalPlasticity(
                E=1000.0, nu=0.3,
                slip_directions=b, slip_normals=n,
                gamma_dot0=0.01, m=0.5, g0=g0, h0=10.0, g_sat=100.0,
            )
            bcs = (
                DirichletBC(dofs=jnp.concatenate([bottom * 2, bottom * 2 + 1]),
                            values=jnp.zeros(2 * len(bottom))),
                DirichletBC(dofs=jnp.concatenate([top * 2, top * 2 + 1]),
                            values=jnp.concatenate([
                                jnp.full(len(top), gamma_target, dtype=jnp.float64),
                                jnp.zeros(len(top), dtype=jnp.float64),
                            ])),
            )
            problem = CPFEProblem(mesh=mesh, material=mat, dirichlet_bcs=bcs, dim=2)
            sol = solve_cpfe(problem, jnp.linspace(0.0, 1.0, 10), dt=0.1, n_sub=3)
            return average_equivalent_plastic_strain(sol.state)

        g = jax.grad(mean_eps_eq)(jnp.array(1.0))
        assert jnp.isfinite(g)
        # Lower g0 → more slip → higher eps_eq, so d(eps_eq)/d(g0) < 0
        assert float(g) < 0.0


# ---------------------------------------------------------------------------
# 3D CPFE solver
# ---------------------------------------------------------------------------
class TestCPFE3D:
    def test_uniaxial_tension_fcc_plastic_strain(self):
        """3D uniaxial tension on an FCC crystal; plastic strain develops."""
        b_fcc, n_fcc = fcc_slip_systems()
        mat = CrystalPlasticity(
            E=1000.0, nu=0.3,
            slip_directions=b_fcc, slip_normals=n_fcc,
            gamma_dot0=0.01, m=0.5, g0=1.0, h0=10.0, g_sat=100.0,
        )
        mesh = rectangular_mesh3d(nx=1, ny=1, nz=1, lx=1.0, ly=1.0, lz=1.0,
                                   cell_type="hex8")
        nodes_x = np.asarray(mesh.nodes)[:, 0]
        left = jnp.where(nodes_x < 1e-9)[0]
        right = jnp.where(nodes_x > 1.0 - 1e-9)[0]
        eps_target = 0.01
        # Clamp left face, displace right face in x
        bcs = (
            DirichletBC(
                dofs=jnp.concatenate([left * 3, left * 3 + 1, left * 3 + 2]),
                values=jnp.zeros(3 * len(left), dtype=jnp.float64),
            ),
            DirichletBC(
                dofs=right * 3,
                values=jnp.full(len(right), eps_target, dtype=jnp.float64),
            ),
        )
        problem = CPFEProblem(mesh=mesh, material=mat, dirichlet_bcs=bcs, dim=3)
        load_factors = jnp.linspace(0.0, 1.0, 15)
        sol = solve_cpfe(problem, load_factors, dt=0.1, n_sub=5)
        # Right face x-displacement must match BC at full load
        assert jnp.allclose(sol.u[right * 3], eps_target, atol=1e-6)
        # Plastic strain should be non-zero (FCC has systems active under tension)
        eps_eq = average_equivalent_plastic_strain(sol.state)
        assert float(eps_eq) > 0.0

    def test_3d_zero_load_zero_displacement(self):
        """Zero load → zero displacement and zero plastic strain."""
        b_fcc, n_fcc = fcc_slip_systems()
        mat = CrystalPlasticity(
            E=1000.0, nu=0.3,
            slip_directions=b_fcc, slip_normals=n_fcc,
            gamma_dot0=0.01, m=0.5, g0=1.0, h0=10.0, g_sat=100.0,
        )
        mesh = rectangular_mesh3d(nx=1, ny=1, nz=1, cell_type="hex8")
        nodes_x = np.asarray(mesh.nodes)[:, 0]
        left = jnp.where(nodes_x < 1e-9)[0]
        bcs = (
            DirichletBC(
                dofs=jnp.concatenate([left * 3, left * 3 + 1, left * 3 + 2]),
                values=jnp.zeros(3 * len(left), dtype=jnp.float64),
            ),
        )
        problem = CPFEProblem(mesh=mesh, material=mat, dirichlet_bcs=bcs, dim=3)
        sol = solve_cpfe(problem, jnp.zeros(5), dt=0.1, n_sub=3)
        assert jnp.allclose(sol.u, 0.0, atol=1e-12)
        assert jnp.allclose(sol.state.gamma, 0.0, atol=1e-14)

    def test_3d_differentiable_wrt_g0(self):
        """3D end-to-end gradient w.r.t. g0 must be finite."""
        b_fcc, n_fcc = fcc_slip_systems()
        mesh = rectangular_mesh3d(nx=1, ny=1, nz=1, cell_type="hex8")
        nodes_x = np.asarray(mesh.nodes)[:, 0]
        left = jnp.where(nodes_x < 1e-9)[0]
        right = jnp.where(nodes_x > 1.0 - 1e-9)[0]
        eps_target = 0.008

        def objective(g0):
            mat = CrystalPlasticity(
                E=1000.0, nu=0.3,
                slip_directions=b_fcc, slip_normals=n_fcc,
                gamma_dot0=0.01, m=0.5, g0=g0, h0=10.0, g_sat=100.0,
            )
            bcs = (
                DirichletBC(
                    dofs=jnp.concatenate([left * 3, left * 3 + 1, left * 3 + 2]),
                    values=jnp.zeros(3 * len(left), dtype=jnp.float64),
                ),
                DirichletBC(dofs=right * 3,
                            values=jnp.full(len(right), eps_target, dtype=jnp.float64)),
            )
            problem = CPFEProblem(mesh=mesh, material=mat, dirichlet_bcs=bcs, dim=3)
            sol = solve_cpfe(problem, jnp.linspace(0.0, 1.0, 8), dt=0.1, n_sub=3)
            return average_equivalent_plastic_strain(sol.state)

        g = jax.grad(objective)(jnp.array(1.0))
        assert jnp.isfinite(g)
        assert float(g) < 0.0  # lower g0 → more slip
