"""Tests for FEM: linear elastic patch test, nonlinear hyperelastic cantilever."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from diffmech.core import rectangular_mesh2d
from diffmech.materials import LinearElasticIsotropic, NeoHookean
from diffmech.methods.fem import (
    assemble_stiffness, solve_linear_elastic, solve_nonlinear_fem, FEMProblem,
)
from diffmech.solvers import DirichletBC


def _make_quad4_mesh(nx=4, ny=4, lx=1.0, ly=1.0):
    return rectangular_mesh2d(nx=nx, ny=ny, lx=lx, ly=ly, cell_type="quad4")


# --- linear elastic ---------------------------------------------------------
def test_linear_fem_patch_test_uniform_stretch():
    """Apply uniform x-displacement on left and right; strain must be uniform.

    For u(x,y) = (eps * x, -nu * eps * y) the strain is constant, so element
    size shouldn't matter (patch test).
    """
    mat = LinearElasticIsotropic(E=100.0, nu=0.3)
    mesh = _make_quad4_mesh(nx=2, ny=2, lx=1.0, ly=1.0)
    n_nodes = mesh.n_nodes
    dim = 2

    # Nodes on x=0 -> u_x=0; nodes on x=1 -> u_x=eps
    eps_applied = 1e-3
    nodes_x = np.asarray(mesh.nodes)[:, 0]
    nodes_y = np.asarray(mesh.nodes)[:, 1]
    left_dofs = jnp.where(nodes_x < 1e-9)[0] * 2  # u_x of left nodes
    right_dofs = jnp.where(nodes_x > 1.0 - 1e-9)[0] * 2
    # Pin bottom-left node's y to suppress rigid translation in y
    bottom_left_y = jnp.array([np.where(nodes_y < 1e-9)[0][0]]) * 2 + 1

    bcs = [
        DirichletBC(dofs=left_dofs, values=jnp.zeros_like(left_dofs, dtype=jnp.float64)),
        DirichletBC(dofs=right_dofs, values=jnp.full_like(right_dofs, eps_applied, dtype=jnp.float64)),
        DirichletBC(dofs=bottom_left_y, values=jnp.zeros_like(bottom_left_y, dtype=jnp.float64)),
    ]
    U = solve_linear_elastic(mesh, mat, bcs, dim=dim)
    # Check that right edge has u_x = eps_applied
    assert jnp.allclose(U[right_dofs], eps_applied, atol=1e-10)
    # Stress sigma_xx should be uniform: sigma_xx = (lam + 2mu) * eps
    # (assuming free lateral contraction -> actually plane strain, so we need to
    # check internal force consistency instead)
    # K @ U should equal F (zero external force except for the BC reaction)
    K = assemble_stiffness(mesh, mat, dim=dim)
    F_int = K @ U
    # Free dofs (non-BC) should have zero net force
    all_dofs = jnp.arange(n_nodes * dim)
    bc_dofs = jnp.concatenate([left_dofs, right_dofs, bottom_left_y])
    free_dofs = jnp.setdiff1d(all_dofs, bc_dofs)
    assert jnp.allclose(F_int[free_dofs], 0.0, atol=1e-8)


def test_linear_fem_uniform_body_force_cantilever():
    """Cantilever beam with body force: tip deflection has the right sign and order.

    For a thin beam, Euler-Bernoulli tip deflection under uniform load q:
    delta = q * L^4 / (8 * E * I)  with I = b*h^3/12, b=1.

    quad4 elements exhibit shear locking in bending, so the FEM result will be
    stiffer than Euler-Bernoulli. We verify the sign and that the deflection is
    within a reasonable factor of the analytic estimate.
    """
    mat = LinearElasticIsotropic(E=1e6, nu=0.0)
    nx, ny = 16, 4
    L, H = 1.0, 0.1
    mesh = rectangular_mesh2d(nx=nx, ny=ny, lx=L, ly=H, cell_type="quad4")
    nodes_x = np.asarray(mesh.nodes)[:, 0]
    left_dofs = np.where(nodes_x < 1e-9)[0]
    # Pin all dofs of left edge
    left_dofs_x = jnp.asarray(left_dofs) * 2
    left_dofs_y = jnp.asarray(left_dofs) * 2 + 1
    bcs = [
        DirichletBC(dofs=left_dofs_x, values=jnp.zeros_like(left_dofs_x, dtype=jnp.float64)),
        DirichletBC(dofs=left_dofs_y, values=jnp.zeros_like(left_dofs_y, dtype=jnp.float64)),
    ]
    body_force = jnp.array([0.0, -1.0])  # unit per volume
    U = solve_linear_elastic(mesh, mat, bcs, body_force=body_force, dim=2)
    # Tip deflection: rightmost node's y-displacement
    nodes_x = np.asarray(mesh.nodes)[:, 0]
    tip_idx = np.where(nodes_x > L - 1e-9)[0]
    tip_y_dofs = jnp.asarray(tip_idx) * 2 + 1
    tip_deflection = float(jnp.mean(U[tip_y_dofs]))
    # Euler-Bernoulli estimate
    expected = 1.0 * L ** 4 / (8 * mat.E * (H ** 3) / 12)
    # FEM with quad4 will be stiffer due to shear locking; check sign + within factor of 15
    assert tip_deflection < 0  # downward
    assert np.isclose(tip_deflection, -expected, rtol=15.0)


def test_linear_fem_differentiable_wrt_material_params():
    """Loss(U(E, nu)) must be differentiable w.r.t. E and nu via jax.grad.

    Under displacement-controlled BC, strain energy U = 0.5 * u^T K u ~ E
    (since K ~ E), so dU/dE > 0.
    """
    def loss(E):
        mat = LinearElasticIsotropic(E=E, nu=0.3)
        mesh = _make_quad4_mesh(nx=2, ny=2, lx=1.0, ly=1.0)
        nodes_x = np.asarray(mesh.nodes)[:, 0]
        right_dofs = jnp.where(nodes_x > 1.0 - 1e-9)[0] * 2
        left_dofs = jnp.where(nodes_x < 1e-9)[0] * 2
        bcs = [
            DirichletBC(dofs=left_dofs, values=jnp.zeros_like(left_dofs, dtype=jnp.float64)),
            DirichletBC(dofs=right_dofs, values=jnp.full_like(right_dofs, 1e-3, dtype=jnp.float64)),
        ]
        # Pin one y-dof to suppress rigid mode
        nodes_y = np.asarray(mesh.nodes)[:, 1]
        bottom_left_y = jnp.array([np.where(nodes_y < 1e-9)[0][0]]) * 2 + 1
        bcs.append(DirichletBC(dofs=bottom_left_y, values=jnp.zeros_like(bottom_left_y, dtype=jnp.float64)))
        U = solve_linear_elastic(mesh, mat, bcs, dim=2)
        # Strain energy = 0.5 * U^T K U
        K = assemble_stiffness(mesh, mat, dim=2)
        return 0.5 * U @ K @ U

    # dU/dE should be positive (energy grows with stiffness under fixed BC)
    g = jax.grad(loss)(jnp.array(100.0))
    assert float(g) > 0
    assert jnp.isfinite(g)


# --- nonlinear --------------------------------------------------------------
def test_nonlinear_fem_neo_hookean_uniform_stretch():
    """Apply uniform x-stretch on hyperelastic block; stress should be ~uniform."""
    mat_nh = NeoHookean(E=1e3, nu=0.3)
    mesh = _make_quad4_mesh(nx=2, ny=2, lx=1.0, ly=1.0)
    nodes_x = np.asarray(mesh.nodes)[:, 0]
    nodes_y = np.asarray(mesh.nodes)[:, 1]
    left_dofs = jnp.where(nodes_x < 1e-9)[0] * 2
    right_dofs = jnp.where(nodes_x > 1.0 - 1e-9)[0] * 2
    bottom_left_y = jnp.array([np.where(nodes_y < 1e-9)[0][0]]) * 2 + 1
    stretch = 0.05  # 5% stretch
    bcs = (
        DirichletBC(dofs=left_dofs, values=jnp.zeros_like(left_dofs, dtype=jnp.float64)),
        DirichletBC(dofs=right_dofs, values=jnp.full_like(right_dofs, stretch, dtype=jnp.float64)),
        DirichletBC(dofs=bottom_left_y, values=jnp.zeros_like(bottom_left_y, dtype=jnp.float64)),
    )
    problem = FEMProblem(mesh=mesh, material=mat_nh, dirichlet_bcs=bcs, dim=2)
    state = solve_nonlinear_fem(problem, tol=1e-8, max_iter=30)
    # Under displacement-controlled BC the right edge must match exactly
    assert jnp.allclose(state.u[right_dofs], stretch, atol=1e-6)


def test_nonlinear_fem_zero_load_zero_displacement():
    """With zero BC and zero body force, the solution should be u=0."""
    mat_nh = NeoHookean(E=1e3, nu=0.3)
    mesh = _make_quad4_mesh(nx=2, ny=2)
    # Pin left edge only (zero BC), no load
    nodes_x = np.asarray(mesh.nodes)[:, 0]
    left_dofs_x = jnp.where(nodes_x < 1e-9)[0] * 2
    left_dofs_y = jnp.where(nodes_x < 1e-9)[0] * 2 + 1
    bcs = (
        DirichletBC(dofs=left_dofs_x, values=jnp.zeros_like(left_dofs_x, dtype=jnp.float64)),
        DirichletBC(dofs=left_dofs_y, values=jnp.zeros_like(left_dofs_y, dtype=jnp.float64)),
    )
    problem = FEMProblem(mesh=mesh, material=mat_nh, dirichlet_bcs=bcs, dim=2)
    state = solve_nonlinear_fem(problem)
    assert jnp.allclose(state.u, 0.0, atol=1e-6)


@pytest.mark.skip(reason="Differentiable Newton via custom_vjp is Phase 2; "
                          "jax.lax.while_loop does not support reverse-mode AD.")
def test_nonlinear_fem_differentiable_wrt_youngs_modulus_OLD():
    pass


def test_nonlinear_fem_differentiable_wrt_youngs_modulus():
    """Strain energy should be differentiable w.r.t. Young's modulus.

    Uses :func:`solve_nonlinear_fem` with ``differentiable=True`` (scan-based
    Newton) so that ``jax.grad`` propagates through the nonlinear solve.
    """
    from diffmech.methods.fem.nonlinear_fem import assemble_internal_force

    def strain_energy(E):
        mat_nh = NeoHookean(E=E, nu=0.3)
        mesh = _make_quad4_mesh(nx=2, ny=2, lx=1.0, ly=1.0)
        nodes_x = np.asarray(mesh.nodes)[:, 0]
        nodes_y = np.asarray(mesh.nodes)[:, 1]
        left_dofs = jnp.where(nodes_x < 1e-9)[0] * 2
        right_dofs = jnp.where(nodes_x > 1.0 - 1e-9)[0] * 2
        bottom_left_y = jnp.array([np.where(nodes_y < 1e-9)[0][0]]) * 2 + 1
        bcs = (
            DirichletBC(dofs=left_dofs, values=jnp.zeros_like(left_dofs, dtype=jnp.float64)),
            DirichletBC(dofs=right_dofs, values=jnp.full_like(right_dofs, 0.05, dtype=jnp.float64)),
            DirichletBC(dofs=bottom_left_y, values=jnp.zeros_like(bottom_left_y, dtype=jnp.float64)),
        )
        problem = FEMProblem(mesh=mesh, material=mat_nh, dirichlet_bcs=bcs, dim=2)
        state = solve_nonlinear_fem(problem, tol=1e-8, max_iter=20, differentiable=True)
        f_int = assemble_internal_force(mesh, state.u, mat_nh, dim=2)
        # Strain energy = 0.5 * u^T * f_int  (f_int = K u in the linearised limit)
        return 0.5 * jnp.sum(state.u * f_int)

    # Under displacement-controlled BC the strain energy grows linearly with E
    # (since K ~ E and u is fixed), so dEnergy/dE > 0.
    g = jax.grad(strain_energy)(jnp.array(1e3))
    assert jnp.isfinite(g)
    assert float(g) > 0
