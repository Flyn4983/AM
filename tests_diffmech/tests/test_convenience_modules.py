"""Tests for the convenience pre/post-processing modules.

Covers:
- :mod:`diffmech.postprocess.derived_fields` (stress/strain recovery wrappers)
- :mod:`diffmech.postprocess.animation` (GIF / PNG export)
- :mod:`diffmech.preprocess.problem_builder` (fluent ProblemSetup)
- :mod:`diffmech.methods.fem.recover_strain_stress` (the underlying recovery)
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import matplotlib
matplotlib.use("Agg")

from diffmech.core import rectangular_mesh2d, rectangular_mesh3d
from diffmech.materials import LinearElasticIsotropic
from diffmech.methods.fem import solve_linear_elastic, recover_strain_stress
from diffmech.postprocess import (
    von_mises_from_stress, principal_stresses,
    hydrostatic_pressure, equivalent_strain, stress_field_collection,
    save_field_gif, save_field_grid_png,
)
from diffmech.postprocess.derived_fields import _deviatoric
from diffmech.preprocess import (
    ProblemSetup, ProblemResult,
    plate_with_hole_2d, get_material, uniaxial_tension, gravity,
)
from diffmech.preprocess.mesh_io import structured_quad_mesh_2d, structured_hex_mesh_3d


# ---------------------------------------------------------------------------
# recover_strain_stress
# ---------------------------------------------------------------------------
def test_recover_strain_stress_uniaxial_2d():
    """Recovered strain should match the applied uniaxial strain."""
    mesh = structured_quad_mesh_2d(4, 4, lx=2.0, ly=1.0)
    mat = LinearElasticIsotropic(E=1.0, nu=0.0)
    case = uniaxial_tension(mesh, axis=0, strain=0.01)
    U = solve_linear_elastic(mesh, mat, list(case.dirichlet_bcs), dim=2)
    strain, stress = recover_strain_stress(mesh, U, mat, dim=2)
    assert strain.shape == (mesh.n_cells, 2, 2)
    assert stress.shape == (mesh.n_cells, 2, 2)
    # eps_xx should equal the applied strain (nu=0, uniform tension).
    eps_xx = np.asarray(strain)[:, 0, 0]
    np.testing.assert_allclose(eps_xx, 0.01, atol=1e-10)
    # sxx = E * eps_xx (nu=0 => plane strain = plane stress)
    sxx = np.asarray(stress)[:, 0, 0]
    np.testing.assert_allclose(sxx, 0.01, atol=1e-10)


def test_recover_strain_stress_3d_shapes():
    """3D recovery should produce (n_cells, 3, 3) tensors."""
    mesh = structured_hex_mesh_3d(2, 2, 2, lx=1.0, ly=1.0, lz=1.0)
    mat = LinearElasticIsotropic(E=100.0, nu=0.3)
    U = jnp.zeros(mesh.n_nodes * 3)
    strain, stress = recover_strain_stress(mesh, U, mat, dim=3)
    assert strain.shape == (mesh.n_cells, 3, 3)
    assert stress.shape == (mesh.n_cells, 3, 3)
    # Zero displacement => zero strain & stress.
    assert float(jnp.max(jnp.abs(strain))) < 1e-12
    assert float(jnp.max(jnp.abs(stress))) < 1e-12


def test_recover_strain_stress_is_differentiable():
    """Gradients must flow from recovered stress to Young's modulus."""
    mesh = structured_quad_mesh_2d(3, 3, lx=1.0, ly=1.0)
    case = uniaxial_tension(mesh, axis=0, strain=0.01)

    def max_vm(E):
        mat = LinearElasticIsotropic(E=E, nu=0.3)
        U = solve_linear_elastic(mesh, mat, list(case.dirichlet_bcs), dim=2)
        _, stress = recover_strain_stress(mesh, U, mat, dim=2)
        return jnp.max(von_mises_from_stress(stress))

    g = jax.grad(max_vm)(jnp.array(100.0))
    assert jnp.isfinite(g)
    # Stress should increase with stiffness.
    assert float(g) > 0.0


# ---------------------------------------------------------------------------
# derived_fields helpers
# ---------------------------------------------------------------------------
def test_von_mises_from_stress_uniaxial():
    """A uniaxial stress state should give vm == |sxx|."""
    n = 5
    s = jnp.zeros((n, 2, 2))
    s = s.at[:, 0, 0].set(3.0)
    vm = np.asarray(von_mises_from_stress(s))
    np.testing.assert_allclose(vm, 3.0, atol=1e-10)


def test_von_mises_from_stress_3d_hydrostatic_is_zero():
    """Pure hydrostatic stress has zero von Mises."""
    n = 3
    p = 5.0
    s = jnp.eye(3)[None, :, :] * p
    s = jnp.broadcast_to(s, (n, 3, 3))
    vm = np.asarray(von_mises_from_stress(s))
    np.testing.assert_allclose(vm, 0.0, atol=1e-9)


def test_deviatoric_trace_is_zero():
    """Deviatoric stress must be traceless."""
    s = jnp.array([[[3.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 1.0]]])
    dev = np.asarray(_deviatoric(s))
    np.testing.assert_allclose(np.trace(dev, axis1=1, axis2=2), 0.0, atol=1e-12)


def test_principal_stresses_sorted_descending():
    """Principal stresses should be returned in descending order."""
    s = jnp.array([[[2.0, 0.0], [0.0, 5.0]]])  # eigenvalues 2, 5
    vals, vecs = principal_stresses(s)
    vals = np.asarray(vals)
    assert vals.shape == (1, 2)
    np.testing.assert_allclose(vals[0], [5.0, 2.0], atol=1e-10)


def test_hydrostatic_pressure_sign():
    """p = -tr(sigma)/dim; compressive (negative) sigma => positive pressure."""
    s = jnp.array([[[-2.0, 0.0], [0.0, -2.0]]])
    p = np.asarray(hydrostatic_pressure(s))
    np.testing.assert_allclose(p, [2.0], atol=1e-12)


def test_equivalent_strain_scales_with_magnitude():
    """Doubling strain magnitude should increase equivalent strain."""
    e1 = jnp.array([[[0.01, 0.0], [0.0, 0.0]]])
    e2 = 2.0 * e1
    assert float(jnp.max(equivalent_strain(e2))) > float(jnp.max(equivalent_strain(e1)))


def test_stress_field_collection_contents():
    """The field collection should contain the expected engineering fields."""
    mesh = structured_quad_mesh_2d(3, 3)
    mat = LinearElasticIsotropic(E=100.0, nu=0.3)
    case = uniaxial_tension(mesh, axis=0, strain=0.01)
    U = solve_linear_elastic(mesh, mat, list(case.dirichlet_bcs), dim=2)
    fc = stress_field_collection(mesh, U, mat, dim=2)
    for name in ("von_mises_stress", "principal_stress_1", "principal_stress_2",
                 "hydrostatic_pressure", "equivalent_strain", "displacement"):
        assert name in fc.point_data or name in fc.cell_data, f"missing {name}"
    # Von Mises must be non-negative.
    assert np.min(fc.cell_data["von_mises_stress"]) >= 0.0


# ---------------------------------------------------------------------------
# animation
# ---------------------------------------------------------------------------
def test_save_field_gif(tmp_path):
    """A GIF should be produced from a stack of 2D fields."""
    fields = np.random.rand(6, 8, 8)
    out = save_field_gif(fields, tmp_path / "anim.gif", title="step {i}", fps=5)
    assert out.exists()
    assert out.stat().st_size > 0


def test_save_field_gif_accepts_list(tmp_path):
    """A list of 2D arrays should also work."""
    fields = [np.ones((5, 5)) * i for i in range(4)]
    out = save_field_gif(fields, tmp_path / "list.gif", colorbar=False)
    assert out.exists()


def test_save_field_grid_png(tmp_path):
    """A grid summary PNG should be produced."""
    fields = np.random.rand(5, 6, 6)
    out = save_field_grid_png(fields, tmp_path / "grid.png",
                             titles=[f"t={i}" for i in range(5)])
    assert out.exists()
    assert out.stat().st_size > 0


def test_save_field_gif_rejects_bad_shape(tmp_path):
    with pytest.raises(ValueError):
        save_field_gif(np.zeros((3, 4, 5, 2)), tmp_path / "bad.gif")


# ---------------------------------------------------------------------------
# ProblemSetup builder
# ---------------------------------------------------------------------------
def test_problem_setup_uniaxial_tension():
    """Full builder → solve → stress recovery should run end to end."""
    res = (ProblemSetup()
           .mesh(structured_quad_mesh_2d(5, 5, lx=2.0, ly=1.0))
           .material_E_nu(E=100.0, nu=0.0)
           .tension(axis=0, strain=0.01)
           .solve())
    assert isinstance(res, ProblemResult)
    assert res.U.shape[0] == res.mesh.n_nodes * 2
    assert res.stress.shape == (res.mesh.n_cells, 2, 2)
    # Recovered strain matches the applied strain.
    eps_xx = np.asarray(res.strain)[:, 0, 0]
    np.testing.assert_allclose(eps_xx, 0.01, atol=1e-9)


def test_problem_setup_with_am_material():
    """The builder should accept a MaterialParams from the AM library."""
    res = (ProblemSetup()
           .mesh(structured_quad_mesh_2d(4, 4))
           .material(get_material("Ti6Al4V"))
           .tension(axis=0, strain=1e-4)
           .solve())
    assert res.U.shape[0] == res.mesh.n_nodes * 2
    assert jnp.all(jnp.isfinite(res.U))


def test_problem_setup_gravity():
    """A gravity (force-controlled) load should produce a finite deflection."""
    res = (ProblemSetup()
           .mesh(structured_quad_mesh_2d(6, 3, lx=4.0, ly=1.0))
           .material_E_nu(E=200.0, nu=0.3)
           .gravity_load(g=2.0, direction=-1)
           .solve())
    assert float(jnp.max(jnp.abs(res.U))) > 0.0


def test_problem_setup_export_and_summary(tmp_path):
    """export() should write a VTU and summary() should print stats."""
    res = (ProblemSetup()
           .mesh(structured_quad_mesh_2d(3, 3))
           .material_E_nu(E=100.0, nu=0.3)
           .tension(axis=0, strain=0.01)
           .solve())
    out = res.export(tmp_path / "result.vtu")
    assert out is res
    assert (tmp_path / "result.vtu").exists()
    s = res.summary()
    assert "nodes" in s
    assert "von Mises" in s


def test_problem_setup_preview(tmp_path):
    """preview() should write a PNG without solving."""
    (ProblemSetup()
     .mesh(plate_with_hole_2d(nx=12, ny=12))
     .material_E_nu(E=1.0, nu=0.3)
     .tension(axis=0, strain=0.01)
     .preview(tmp_path / "preview.png"))
    assert (tmp_path / "preview.png").exists()


def test_problem_setup_validates_missing_config():
    """solve() should raise if mesh/material/load-case is missing."""
    with pytest.raises(ValueError):
        ProblemSetup().material_E_nu(E=1.0, nu=0.3).tension(axis=0).solve()
    with pytest.raises(ValueError):
        ProblemSetup().mesh(structured_quad_mesh_2d(2, 2)).tension(axis=0).solve()
