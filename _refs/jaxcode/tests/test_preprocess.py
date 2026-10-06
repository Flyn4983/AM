"""Tests for the pre-processing module: mesh I/O, mesh tools, load cases."""

import numpy as np
import pytest

import jax.numpy as jnp

from diffmech.core import Mesh, rectangular_mesh2d, rectangular_mesh3d
from diffmech.preprocess import (
    read_mesh, write_mesh, from_meshio, to_meshio,
    supported_meshio_cell_types,
    structured_quad_mesh_2d, structured_tri_mesh_2d,
    structured_hex_mesh_3d, structured_tet_mesh_3d,
    boundary_node_ids, nodes_on_box_face, nodes_in_box,
    element_volumes, total_volume,
    assign_cell_regions, box_region,
    refine_structured_2d, refine_structured_3d,
    LoadCase,
    clamped_face, pin_node,
    uniaxial_tension, uniaxial_compression,
    simple_shear, biaxial, bending, gravity,
)


# ---------------------------------------------------------------------------
# Mesh I/O round-trip
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cell_type", ["quad4", "tri3"])
def test_meshio_roundtrip_2d(tmp_path, cell_type):
    """write_mesh then read_mesh should recover the same mesh (2D)."""
    mesh = structured_quad_mesh_2d(3, 2) if cell_type == "quad4" \
        else structured_tri_mesh_2d(3, 2)
    path = tmp_path / "mesh.vtu"
    write_mesh(path, mesh)
    mesh2 = read_mesh(path)
    assert mesh2.cell_type == cell_type
    assert mesh2.n_nodes == mesh.n_nodes
    assert mesh2.n_cells == mesh.n_cells
    np.testing.assert_allclose(np.asarray(mesh2.nodes),
                               np.asarray(mesh.nodes), atol=1e-12)
    np.testing.assert_array_equal(np.asarray(mesh2.cells),
                                  np.asarray(mesh.cells))


@pytest.mark.parametrize("cell_type", ["hex8", "tet4"])
def test_meshio_roundtrip_3d(tmp_path, cell_type):
    """write_mesh then read_mesh should recover the same mesh (3D)."""
    mesh = structured_hex_mesh_3d(2, 2, 2) if cell_type == "hex8" \
        else structured_tet_mesh_3d(2, 2, 2)
    path = tmp_path / "mesh3d.vtu"
    write_mesh(path, mesh)
    mesh2 = read_mesh(path)
    assert mesh2.cell_type == cell_type
    assert mesh2.n_nodes == mesh.n_nodes
    assert mesh2.n_cells == mesh.n_cells
    np.testing.assert_allclose(np.asarray(mesh2.nodes),
                               np.asarray(mesh.nodes), atol=1e-12)


def test_meshio_roundtrip_with_fields(tmp_path):
    """point_data and cell_data should survive a write/read round-trip."""
    mesh = structured_quad_mesh_2d(2, 2)
    point_data = {"temperature": np.arange(mesh.n_nodes, dtype=np.float64)}
    cell_data = {"region_id": np.zeros(mesh.n_cells, dtype=np.int32)}
    path = tmp_path / "mesh_fields.vtu"
    write_mesh(path, mesh, point_data=point_data, cell_data=cell_data)
    # Read back the raw meshio object to inspect fields (read_mesh drops them).
    import meshio
    m = meshio.read(str(path))
    assert "temperature" in m.point_data
    assert "region_id" in m.cell_data
    np.testing.assert_allclose(
        m.point_data["temperature"],
        point_data["temperature"], atol=1e-12,
    )


def test_to_meshio_pads_2d_to_3d():
    """meshio always stores xyz; a 2D mesh should be padded with z=0."""
    mesh = structured_quad_mesh_2d(2, 2)
    m = to_meshio(mesh)
    assert m.points.shape[1] == 3
    np.testing.assert_allclose(m.points[:, 2], 0.0, atol=1e-15)


def test_from_meshio_drops_zero_z_axis():
    """A 3D mesh whose z column is all zero should be returned as 2D."""
    mesh = structured_quad_mesh_2d(2, 2)
    m = to_meshio(mesh)  # padded to 3D with z=0
    mesh2 = from_meshio(m)
    assert mesh2.dim == 2
    assert mesh2.cell_type == "quad4"


def test_supported_meshio_cell_types():
    types = supported_meshio_cell_types()
    assert "triangle" in types
    assert "quad" in types
    assert "tetra" in types
    assert "hexahedron" in types


def test_write_mesh_multiple_formats(tmp_path):
    """The writer should pick the format from the file extension."""
    mesh = structured_tri_mesh_2d(2, 2)
    # Use formats that don't require optional deps (e.g. h5py for xdmf/exodus).
    for ext in (".vtu", ".vtk", ".msh", ".inp"):
        path = tmp_path / f"mesh{ext}"
        write_mesh(path, mesh)
        assert path.exists()


# ---------------------------------------------------------------------------
# Boundary detection
# ---------------------------------------------------------------------------
def test_boundary_node_ids_quad4():
    """Only the outer ring of nodes should be flagged as boundary."""
    mesh = structured_quad_mesh_2d(3, 3)  # 4x4 nodes
    bnd = boundary_node_ids(mesh)
    # Total nodes 16; interior nodes = (3-1)*(3-1) = 4 -> boundary = 12
    assert len(bnd) == 12
    # Corner nodes must be present.
    corners = {0, 3, 12, 15}
    assert corners.issubset(set(bnd.tolist()))


def test_boundary_node_ids_tri3():
    mesh = structured_tri_mesh_2d(2, 2)
    bnd = boundary_node_ids(mesh)
    # 2x2 tri3 grid has 9 nodes; the centre node (id 4) is interior to all
    # 8 triangles, so 8 nodes lie on the boundary.
    assert len(bnd) == 8
    assert 4 not in set(bnd.tolist())


def test_boundary_node_ids_hex8():
    mesh = structured_hex_mesh_3d(2, 2, 2)  # 3x3x3 = 27 nodes
    bnd = boundary_node_ids(mesh)
    # Interior node count = 1 (the centre), so boundary = 26.
    assert len(bnd) == 26


def test_nodes_on_box_face():
    mesh = structured_quad_mesh_2d(4, 3, lx=1.0, ly=1.0)
    left = nodes_on_box_face(mesh, xmin=0.0)
    right = nodes_on_box_face(mesh, xmax=1.0)
    top = nodes_on_box_face(mesh, ymax=1.0)
    bottom = nodes_on_box_face(mesh, ymin=0.0)
    # 4 cells in x -> 5 nodes per y-row
    assert len(left) == 4   # ny+1 = 4
    assert len(right) == 4
    # 3 cells in y -> 4 nodes per x-row
    assert len(top) == 5    # nx+1 = 5
    assert len(bottom) == 5
    # No overlap between left and right
    assert set(left.tolist()).isdisjoint(set(right.tolist()))


def test_nodes_in_box():
    mesh = structured_quad_mesh_2d(8, 8, lx=1.0, ly=1.0)
    # 8x8 grid has node spacing 0.125; box (0.2, 0.8) x (0.2, 0.8)
    # contains nodes at 0.25, 0.375, 0.5, 0.625, 0.75 -> 5x5 = 25 nodes.
    inner = nodes_in_box(mesh, xmin=0.2, xmax=0.8,
                         ymin=0.2, ymax=0.8)
    assert len(inner) == 25
    # All returned nodes must lie inside the box.
    nodes = np.asarray(mesh.nodes)
    assert np.all(nodes[inner, 0] >= 0.2 - 1e-15)
    assert np.all(nodes[inner, 0] <= 0.8 + 1e-15)


# ---------------------------------------------------------------------------
# Element volumes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cell_type", ["tri3", "quad4"])
def test_element_volumes_2d(cell_type):
    mesh = structured_quad_mesh_2d(2, 2, lx=2.0, ly=3.0) \
        if cell_type == "quad4" else structured_tri_mesh_2d(2, 2, lx=2.0, ly=3.0)
    vols = element_volumes(mesh)
    assert vols.shape == (mesh.n_cells,)
    if cell_type == "quad4":
        # Each quad is 1x1.5, area = 1.5
        np.testing.assert_allclose(vols, 1.5, atol=1e-12)
    else:
        # Each tri is half a quad -> area = 0.75
        np.testing.assert_allclose(vols, 0.75, atol=1e-12)
    assert abs(total_volume(mesh) - 6.0) < 1e-12  # 2*3


def test_element_volumes_hex8():
    mesh = structured_hex_mesh_3d(2, 2, 2, lx=2.0, ly=3.0, lz=4.0)
    vols = element_volumes(mesh)
    # Each cell is 1x1.5x2 = 3.0
    np.testing.assert_allclose(vols, 3.0, atol=1e-12)
    assert abs(total_volume(mesh) - 24.0) < 1e-12


def test_element_volumes_tet4():
    mesh = structured_tet_mesh_3d(1, 1, 1, lx=1.0, ly=1.0, lz=1.0)
    vols = element_volumes(mesh)
    # A unit cube split into 6 tets has total volume 1.0
    assert abs(total_volume(mesh) - 1.0) < 1e-12
    # All tets have positive volume.
    assert np.all(vols > 0.0)


# ---------------------------------------------------------------------------
# Region assignment
# ---------------------------------------------------------------------------
def test_assign_cell_regions():
    mesh = structured_quad_mesh_2d(4, 4, lx=1.0, ly=1.0)
    regions = assign_cell_regions(mesh, [
        box_region(xmax=0.5),   # region 0: left half
        box_region(xmin=0.5),   # region 1: right half
    ])
    assert regions.shape == (mesh.n_cells,)
    # Each region should contain exactly half the cells (8 each).
    assert np.sum(regions == 0) == 8
    assert np.sum(regions == 1) == 8
    # No unassigned cells.
    assert np.sum(regions == -1) == 0


# ---------------------------------------------------------------------------
# Structured refinement
# ---------------------------------------------------------------------------
def test_refine_structured_2d():
    mesh = structured_quad_mesh_2d(2, 2)
    fine = refine_structured_2d(mesh, factor=2)
    assert fine.n_cells == 4 * mesh.n_cells
    # Domain size unchanged
    np.testing.assert_allclose(np.asarray(fine.nodes).max(axis=0),
                               np.asarray(mesh.nodes).max(axis=0))


def test_refine_structured_3d():
    mesh = structured_hex_mesh_3d(2, 2, 2)
    fine = refine_structured_3d(mesh, factor=2)
    assert fine.n_cells == 8 * mesh.n_cells


# ---------------------------------------------------------------------------
# Load cases
# ---------------------------------------------------------------------------
def test_clamped_face_returns_bc():
    mesh = structured_quad_mesh_2d(4, 4)
    case = clamped_face(mesh, axis=0, side="min")
    assert isinstance(case, LoadCase)
    assert len(case.dirichlet_bcs) == 1
    bc = case.dirichlet_bcs[0]
    # All dofs of the left face are pinned (x and y for each node).
    n_left_nodes = len(nodes_on_box_face(mesh, xmin=0.0))
    assert len(bc.dofs) == 2 * n_left_nodes
    np.testing.assert_allclose(np.asarray(bc.values), 0.0)


def test_pin_node_pins_all_dofs():
    mesh = structured_quad_mesh_2d(2, 2)
    case = pin_node(mesh, node_id=0)
    assert len(case.dirichlet_bcs[0].dofs) == mesh.dim


def test_uniaxial_tension_bcs():
    mesh = structured_quad_mesh_2d(4, 4, lx=2.0, ly=1.0)
    case = uniaxial_tension(mesh, axis=0, strain=0.01)
    # Min face pinned in x, max face displaced by 0.01 * 2.0 = 0.02
    max_bc = case.dirichlet_bcs[1]
    np.testing.assert_allclose(np.asarray(max_bc.values), 0.02, atol=1e-12)


def test_uniaxial_compression_negative_strain():
    mesh = structured_quad_mesh_2d(2, 2, lx=1.0, ly=1.0)
    case = uniaxial_compression(mesh, axis=0, strain=-0.02)
    max_bc = case.dirichlet_bcs[1]
    # Displacement should be negative.
    assert np.all(np.asarray(max_bc.values) < 0.0)


def test_simple_shear_bcs():
    mesh = structured_quad_mesh_2d(4, 4, lx=1.0, ly=1.0)
    case = simple_shear(mesh, shear_axis=0, normal_axis=1, gamma=0.01)
    # Bottom face fully clamped, top face displaced in x by 0.01
    assert isinstance(case, LoadCase)
    # The third BC pins the top face's normal (y) dof to 0.
    assert len(case.dirichlet_bcs) >= 3


def test_simple_shear_rejects_same_axes():
    mesh = structured_quad_mesh_2d(2, 2)
    with pytest.raises(ValueError):
        simple_shear(mesh, shear_axis=0, normal_axis=0)


def test_biaxial_2d():
    mesh = structured_quad_mesh_2d(3, 3, lx=1.0, ly=1.0)
    case = biaxial(mesh, strain_x=0.01, strain_y=0.02)
    assert isinstance(case, LoadCase)


def test_biaxial_rejects_3d():
    mesh = structured_hex_mesh_3d(2, 2, 2)
    with pytest.raises(ValueError):
        biaxial(mesh)


def test_bending_load_case():
    mesh = structured_quad_mesh_2d(8, 4, lx=4.0, ly=1.0)
    case = bending(mesh, axis=0, thickness_axis=1, curvature=0.05)
    assert isinstance(case, LoadCase)
    # The min (left) face should be fully clamped.
    clamp_bc = case.dirichlet_bcs[0]
    n_left = len(nodes_on_box_face(mesh, xmin=0.0))
    assert len(clamp_bc.dofs) == 2 * n_left


def test_gravity_load_case_has_body_force():
    mesh = structured_quad_mesh_2d(4, 4, lx=1.0, ly=2.0)
    case = gravity(mesh, g=9.81, direction=-1)
    assert case.body_force is not None
    # gravity points along the last axis (y in 2D) with negative sign.
    body = np.asarray(case.body_force)
    assert body[-1] == pytest.approx(-9.81)
    # The clamped face BC should still be present.
    assert len(case.dirichlet_bcs) == 1


def test_load_case_dofs_are_jax_arrays():
    """BC dofs/values should be JAX arrays (so they thread through jit)."""
    mesh = structured_quad_mesh_2d(2, 2)
    case = uniaxial_tension(mesh, axis=0, strain=0.01)
    for bc in case.dirichlet_bcs:
        assert isinstance(bc.dofs, jnp.ndarray)
        assert isinstance(bc.values, jnp.ndarray)
