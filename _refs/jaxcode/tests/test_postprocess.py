"""Tests for the post-processing module: VTK writers, probes, time-series."""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from diffmech.core import rectangular_mesh2d, rectangular_mesh3d
from diffmech.materials import LinearElasticIsotropic
from diffmech.methods.fem import solve_linear_elastic
from diffmech.postprocess import (
    FieldCollection, write_vtu, write_vtk, PVDSeries,
    von_mises_stress_field, equivalent_strain_field,
    nodal_displacement_field,
    plot_field_2d, plot_contour_2d, plot_quiver_2d, plot_deformed_mesh_2d,
    plot_scatter_3d, plot_slice_3d,
    displacement_to_node_field, node_field_to_dofs,
    point_probe, volume_average, volume_integral, total_mass,
    boundary_node_field_integral, field_statistics,
    extract_history, history_at_dof, tip_displacement_history,
    HistoryLogger, TimeSeriesWriter,
    save_history_npz, load_history_npz, save_snapshot,
)
from diffmech.preprocess import (
    structured_quad_mesh_2d, structured_hex_mesh_3d,
    boundary_node_ids, nodes_on_box_face, uniaxial_tension,
)
from diffmech.solvers import DirichletBC


# Use a non-interactive matplotlib backend so tests run headless.
import matplotlib
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _solve_tension(nx=4, ny=4, strain=1e-3):
    """Solve a small uniaxial-tension problem and return (mesh, U, dim)."""
    mesh = structured_quad_mesh_2d(nx, ny, lx=1.0, ly=1.0)
    mat = LinearElasticIsotropic(E=100.0, nu=0.3)
    case = uniaxial_tension(mesh, axis=0, strain=strain, lateral_free=True)
    U = solve_linear_elastic(mesh, mat, list(case.dirichlet_bcs), dim=2)
    return mesh, U


# ---------------------------------------------------------------------------
# FieldCollection
# ---------------------------------------------------------------------------
def test_field_collection_add_point_and_cell():
    fc = FieldCollection()
    fc.add_point("u", np.zeros(5))
    fc.add_cell("region", np.zeros(3, dtype=np.int32))
    assert "u" in fc.point_data
    assert "region" in fc.cell_data


# ---------------------------------------------------------------------------
# VTK / VTU writers
# ---------------------------------------------------------------------------
def test_write_vtu_minimal(tmp_path):
    mesh = structured_quad_mesh_2d(2, 2)
    path = tmp_path / "out.vtu"
    write_vtu(path, mesh)
    assert path.exists() and path.stat().st_size > 0


def test_write_vtu_with_fields(tmp_path):
    mesh, U = _solve_tension()
    dim = 2
    disp = nodal_displacement_field(U, mesh.n_nodes, dim)
    fc = FieldCollection(
        point_data={"displacement": disp},
        cell_data={"region": np.zeros(mesh.n_cells, dtype=np.int32)},
    )
    path = tmp_path / "with_fields.vtu"
    write_vtu(path, mesh, fc)
    assert path.exists()
    # Read it back via meshio and check the fields.
    import meshio
    m = meshio.read(str(path))
    assert "displacement" in m.point_data
    assert "region" in m.cell_data


def test_write_vtu_deformed(tmp_path):
    """deformed=True should write displaced coordinates + reference_coords."""
    mesh, U = _solve_tension()
    dim = 2
    disp = nodal_displacement_field(U, mesh.n_nodes, dim)
    fc = FieldCollection(point_data={"displacement": disp})
    path = tmp_path / "deformed.vtu"
    write_vtu(path, mesh, fc, deformed=True)
    import meshio
    m = meshio.read(str(path))
    assert "reference_coords" in m.point_data
    # The deformed x at the right face should equal the prescribed displacement.
    disp_field = m.point_data["displacement"]
    nodes = np.asarray(mesh.nodes)
    right_mask = nodes[:, 0] > 1.0 - 1e-9
    np.testing.assert_allclose(
        m.points[right_mask, 0],
        nodes[right_mask, 0] + disp_field[right_mask, 0],
        atol=1e-10,
    )


def test_write_vtu_deformed_requires_displacement(tmp_path):
    mesh = structured_quad_mesh_2d(2, 2)
    with pytest.raises(ValueError):
        write_vtu(tmp_path / "x.vtu", mesh, deformed=True)


def test_write_vtk_legacy(tmp_path):
    mesh = structured_quad_mesh_2d(2, 2)
    path = tmp_path / "legacy.vtk"
    write_vtk(path, mesh, binary=False)
    assert path.exists()
    # Legacy VTK ASCII starts with "# vtk DataFile"
    text = path.read_text()
    assert "vtk DataFile" in text


def test_save_snapshot_alias(tmp_path):
    mesh = structured_quad_mesh_2d(2, 2)
    path = tmp_path / "snap.vtu"
    save_snapshot(path, mesh)
    assert path.exists()


# ---------------------------------------------------------------------------
# PVD series
# ---------------------------------------------------------------------------
def test_pvd_series_writes_master_and_frames(tmp_path):
    mesh = structured_quad_mesh_2d(2, 2)
    series = PVDSeries("anim", directory=tmp_path)
    for i, t in enumerate([0.0, 0.1, 0.2]):
        fc = FieldCollection(point_data={"step": np.full(mesh.n_nodes, i,
                                                         dtype=np.float64)})
        series.write(t, mesh, fc)
    pvd = series.close()
    assert pvd.exists()
    # Three VTU frames + one PVD master.
    vtu_files = list(tmp_path.glob("anim_*.vtu"))
    assert len(vtu_files) == 3
    text = pvd.read_text()
    assert "DataSet" in text
    assert 'timestep="0.2"' in text


def test_pvd_series_context_manager(tmp_path):
    mesh = structured_quad_mesh_2d(2, 2)
    with PVDSeries("ctx", directory=tmp_path) as series:
        series.write(0.0, mesh)
        series.write(1.0, mesh)
    assert (tmp_path / "ctx.pvd").exists()


# ---------------------------------------------------------------------------
# von Mises / equivalent strain helpers
# ---------------------------------------------------------------------------
def test_von_mises_stress_voigt_2d():
    """Uniform biaxial stress should give vm = sigma_xx."""
    stress = np.array([[100.0, 100.0, 0.0]])  # one cell, [sxx, syy, sxy]
    vm = von_mises_stress_field(stress)
    # vm = sqrt(sxx^2 - sxx*syy + syy^2 + 3*sxy^2) = sqrt(10000) = 100
    assert vm.shape == (1,)
    np.testing.assert_allclose(vm, 100.0, atol=1e-10)


def test_von_mises_stress_full_tensor_3d():
    """Hydrostatic stress should give zero von Mises."""
    p = 50.0
    stress = np.array([[[p, 0, 0], [0, p, 0], [0, 0, p]]])
    vm = von_mises_stress_field(stress)
    np.testing.assert_allclose(vm, 0.0, atol=1e-10)


def test_equivalent_strain_field_divides_by_sqrt3():
    strain = np.array([[1.0, 0.0, 0.0]])
    es = equivalent_strain_field(strain)
    np.testing.assert_allclose(es, 1.0 / np.sqrt(3.0), atol=1e-12)


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------
def test_displacement_to_node_field_roundtrip():
    U = jnp.arange(8, dtype=jnp.float64)
    field = displacement_to_node_field(U, n_nodes=4, dim=2)
    assert field.shape == (4, 2)
    back = node_field_to_dofs(field)
    np.testing.assert_allclose(np.asarray(back), np.asarray(U))


def test_point_probe_interpolates_constant_field():
    """A constant per-node field should interpolate to the same value."""
    mesh = structured_quad_mesh_2d(4, 4)
    field = jnp.full(mesh.n_nodes, 3.14)
    val = point_probe(mesh, jnp.array([0.5, 0.5]), field)
    np.testing.assert_allclose(float(val), 3.14, atol=1e-10)


def test_point_probe_interpolates_linear_field():
    """f(x,y) = x should be recovered exactly by linear shape functions."""
    mesh = structured_quad_mesh_2d(4, 4, lx=1.0, ly=1.0)
    nodes = np.asarray(mesh.nodes)
    field = jnp.asarray(nodes[:, 0])
    val = point_probe(mesh, jnp.array([0.3, 0.7]), field)
    np.testing.assert_allclose(float(val), 0.3, atol=1e-10)


def test_volume_average_uniform_weights():
    mesh = structured_quad_mesh_2d(2, 2, lx=1.0, ly=1.0)
    field = jnp.ones(mesh.n_cells)
    avg = volume_average(mesh, field, weights="uniform")
    np.testing.assert_allclose(float(avg), 1.0)


def test_volume_average_volume_weights():
    """Volume-weighted average of a constant should still be the constant."""
    mesh = structured_quad_mesh_2d(2, 2)
    field = jnp.full(mesh.n_cells, 2.5)
    avg = volume_average(mesh, field, weights="volume")
    np.testing.assert_allclose(float(avg), 2.5)


def test_volume_average_vector_field():
    mesh = structured_quad_mesh_2d(2, 2)
    field = jnp.ones((mesh.n_cells, 2))
    avg = volume_average(mesh, field)
    assert avg.shape == (2,)
    np.testing.assert_allclose(np.asarray(avg), 1.0)


def test_volume_integral():
    mesh = structured_quad_mesh_2d(2, 2, lx=1.0, ly=1.0)
    field = jnp.ones(mesh.n_cells)
    # Integral of 1 over a 1x1 square = 1
    np.testing.assert_allclose(float(volume_integral(mesh, field)), 1.0,
                              atol=1e-12)


def test_total_mass():
    mesh = structured_quad_mesh_2d(2, 2, lx=1.0, ly=1.0)
    density = jnp.full(mesh.n_cells, 7.5)  # unit thickness
    np.testing.assert_allclose(float(total_mass(mesh, density)), 7.5,
                              atol=1e-12)


def test_boundary_node_field_integral():
    mesh = structured_quad_mesh_2d(4, 4)
    bnd = boundary_node_ids(mesh)
    field = jnp.ones(mesh.n_nodes)
    total = boundary_node_field_integral(mesh, field, boundary_node_ids=bnd)
    assert float(total) == len(bnd)


def test_field_statistics():
    field = jnp.array([1.0, -2.0, 3.0, 0.0])
    stats = field_statistics(field)
    assert float(stats["min"]) == -2.0
    assert float(stats["max"]) == 3.0
    assert float(stats["mean"]) == 0.5
    assert float(stats["abs_max"]) == 3.0


# ---------------------------------------------------------------------------
# Time-history extraction
# ---------------------------------------------------------------------------
def test_extract_history_subsample():
    history = jnp.arange(10.0)
    sub = extract_history(history, every=2)
    np.testing.assert_allclose(np.asarray(sub), [0, 2, 4, 6, 8])


def test_history_at_dof():
    history = jnp.arange(12).reshape(3, 4).astype(jnp.float64)
    col = history_at_dof(history, dof=1)
    np.testing.assert_allclose(np.asarray(col), [1, 5, 9])


def test_tip_displacement_history():
    history = jnp.arange(12).reshape(3, 4).astype(jnp.float64)
    tip = tip_displacement_history(history, jnp.array([3]))
    # tip = mean over dof 3 across time = [3, 7, 11]
    np.testing.assert_allclose(np.asarray(tip), [3, 7, 11])


# ---------------------------------------------------------------------------
# HistoryLogger
# ---------------------------------------------------------------------------
def test_history_logger_save_load(tmp_path):
    log = HistoryLogger()
    for t in [0.0, 0.1, 0.2]:
        log.append(t, energy=t ** 2, u=np.array([t, 2 * t]))
    path = tmp_path / "history.npz"
    log.save(path)
    data = load_history_npz(path)
    np.testing.assert_allclose(data["time"], [0.0, 0.1, 0.2])
    np.testing.assert_allclose(data["energy"], [0.0, 0.01, 0.04])
    assert data["u"].shape == (3, 2)


def test_save_load_history_npz_roundtrip(tmp_path):
    path = tmp_path / "h.npz"
    save_history_npz(path, time=np.array([0.0, 1.0]),
                     energy=np.array([1.0, 2.0]))
    data = load_history_npz(path)
    np.testing.assert_allclose(data["time"], [0.0, 1.0])
    np.testing.assert_allclose(data["energy"], [1.0, 2.0])


# ---------------------------------------------------------------------------
# TimeSeriesWriter (combined VTU + npz)
# ---------------------------------------------------------------------------
def test_time_series_writer_writes_pvd_and_npz(tmp_path):
    mesh = structured_quad_mesh_2d(2, 2)
    with TimeSeriesWriter("run", directory=tmp_path, save_every=1) as w:
        for t in [0.0, 0.5, 1.0]:
            fc = FieldCollection(point_data={"step": np.full(mesh.n_nodes, t)})
            w.write(t, mesh, fc, energy=t * 2.0)
    assert (tmp_path / "run.pvd").exists()
    assert (tmp_path / "run.npz").exists()
    data = load_history_npz(tmp_path / "run.npz")
    np.testing.assert_allclose(data["time"], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(data["energy"], [0.0, 1.0, 2.0])
    # Three frames should have been written.
    assert len(list(tmp_path.glob("run_*.vtu"))) == 3


def test_time_series_writer_save_every(tmp_path):
    mesh = structured_quad_mesh_2d(2, 2)
    with TimeSeriesWriter("sparse", directory=tmp_path, save_every=2) as w:
        for t in range(5):
            w.write(float(t), mesh, energy=float(t))
    # Only frames at step 0, 2, 4 -> 3 VTU files
    assert len(list(tmp_path.glob("sparse_*.vtu"))) == 3
    # But all 5 entries are in the npz history.
    data = load_history_npz(tmp_path / "sparse.npz")
    assert len(data["time"]) == 5


# ---------------------------------------------------------------------------
# Visualization (smoke tests only — headless backend)
# ---------------------------------------------------------------------------
def test_plot_field_2d_smoke():
    mesh = structured_quad_mesh_2d(3, 3)
    field = np.random.rand(mesh.n_nodes)
    ax = plot_field_2d(mesh, field, title="t", colorbar=True, show_mesh=True)
    assert ax is not None


def test_plot_contour_2d_smoke():
    mesh = structured_quad_mesh_2d(3, 3)
    field = np.random.rand(mesh.n_nodes)
    ax = plot_contour_2d(mesh, field)
    assert ax is not None


def test_plot_quiver_2d_smoke():
    mesh = structured_quad_mesh_2d(3, 3)
    vf = np.random.rand(mesh.n_nodes, 2)
    ax = plot_quiver_2d(mesh, vf)
    assert ax is not None


def test_plot_deformed_mesh_2d_smoke():
    mesh = structured_quad_mesh_2d(3, 3)
    U = np.zeros((mesh.n_nodes, 2))
    U[:, 0] = 0.01 * np.asarray(mesh.nodes)[:, 0]
    ax = plot_deformed_mesh_2d(mesh, U, scale_factor=10.0)
    assert ax is not None


def test_plot_scatter_3d_smoke():
    mesh = structured_hex_mesh_3d(2, 2, 2)
    field = np.random.rand(mesh.n_nodes)
    ax = plot_scatter_3d(mesh, field)
    assert ax is not None


def test_plot_slice_3d_smoke():
    # 4x4x4 grid has z layers at 0, 1/3, 1/2, 2/3, 1 — offset=0.5 hits a layer.
    mesh = structured_hex_mesh_3d(4, 4, 4)
    field = np.random.rand(mesh.n_nodes)
    ax = plot_slice_3d(mesh, field, plane="xy", offset=0.5)
    assert ax is not None


def test_plot_slice_3d_rejects_bad_plane():
    mesh = structured_hex_mesh_3d(2, 2, 2)
    field = np.zeros(mesh.n_nodes)
    with pytest.raises(ValueError):
        plot_slice_3d(mesh, field, plane="ab")


# ---------------------------------------------------------------------------
# End-to-end: solve -> export -> reload
# ---------------------------------------------------------------------------
def test_solve_then_export_to_vtu(tmp_path):
    """A full solve should be exportable to a ParaView-readable VTU."""
    mesh, U = _solve_tension(nx=4, ny=4, strain=2e-3)
    dim = 2
    disp = nodal_displacement_field(U, mesh.n_nodes, dim)
    fc = FieldCollection(
        point_data={"displacement": disp},
        cell_data={"region": np.zeros(mesh.n_cells, dtype=np.int32)},
    )
    path = tmp_path / "tension_solution.vtu"
    write_vtu(path, mesh, fc, deformed=True)
    assert path.exists()
    # Reload and verify displacement magnitude at the right face.
    import meshio
    m = meshio.read(str(path))
    right_mask = np.asarray(mesh.nodes)[:, 0] > 1.0 - 1e-9
    np.testing.assert_allclose(
        m.point_data["displacement"][right_mask, 0], 2e-3, atol=1e-10,
    )


def test_probes_are_differentiable():
    """volume_average should be AD-friendly (used in inverse design)."""
    mesh = structured_quad_mesh_2d(2, 2, lx=1.0, ly=1.0)

    def avg_of_scaled(scale):
        field = scale * jnp.ones(mesh.n_cells)
        return volume_average(mesh, field, weights="volume")

    grad = jax.grad(avg_of_scaled)(jnp.array(2.0))
    # d/dscale [scale * 1] = 1
    np.testing.assert_allclose(float(grad), 1.0, atol=1e-10)
