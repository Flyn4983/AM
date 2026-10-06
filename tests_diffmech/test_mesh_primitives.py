"""Tests for the mesh-primitives pre-processing module."""

from __future__ import annotations

import numpy as np
import pytest

import meshio

from diffmech.core import Mesh
from diffmech.preprocess import (
    disc_mesh_2d, ring_mesh_2d, plate_with_hole_2d,
    l_bracket_mesh_2d, notched_bar_mesh_2d,
    cylinder_mesh_3d, plate_with_hole_3d,
)
from diffmech.preprocess.mesh_tools import element_volumes, total_volume
from diffmech.preprocess.mesh_io import write_mesh, read_mesh


# ---------------------------------------------------------------------------
# 2D disc
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cell_type", ["tri3", "quad4"])
def test_disc_area_approximates_pi_r2(cell_type):
    r = 1.0
    m = disc_mesh_2d(radius=r, n=32, cell_type=cell_type)
    assert m.cell_type == cell_type
    assert m.dim == 2
    area = total_volume(m)
    # Staircase approximation -> ~5% error at n=32.
    assert area == pytest.approx(np.pi * r * r, rel=0.05)


def test_disc_radius_scales_area():
    m1 = disc_mesh_2d(radius=1.0, n=24)
    m2 = disc_mesh_2d(radius=2.0, n=24)
    assert total_volume(m2) == pytest.approx(4.0 * total_volume(m1), rel=0.05)


def test_disc_cells_inside_radius():
    r = 1.0
    m = disc_mesh_2d(radius=r, n=20, cell_type="quad4")
    vols = element_volumes(m)
    nodes = np.asarray(m.nodes)
    cells = np.asarray(m.cells)
    centroids = nodes[cells].mean(axis=1)
    radii = np.linalg.norm(centroids, axis=1)
    # All cell centroids should lie inside the disc (cell-mask kept only those).
    assert np.all(radii <= r + 1e-12)
    assert m.n_cells == len(vols)


# ---------------------------------------------------------------------------
# 2D ring (annulus)
# ---------------------------------------------------------------------------
def test_ring_area_approximates_pi_R2_minus_r2():
    ri, ro = 0.5, 1.0
    m = ring_mesh_2d(r_inner=ri, r_outer=ro, n=32)
    area = total_volume(m)
    expected = np.pi * (ro * ro - ri * ri)
    assert area == pytest.approx(expected, rel=0.05)


def test_ring_excludes_interior():
    ri, ro = 0.5, 1.0
    m = ring_mesh_2d(r_inner=ri, r_outer=ro, n=24, cell_type="quad4")
    nodes = np.asarray(m.nodes)
    cells = np.asarray(m.cells)
    centroids = nodes[cells].mean(axis=1)
    radii = np.linalg.norm(centroids, axis=1)
    assert np.all(radii >= ri - 1e-12)
    assert np.all(radii <= ro + 1e-12)


# ---------------------------------------------------------------------------
# Plate with hole
# ---------------------------------------------------------------------------
def test_plate_with_hole_area():
    L, H, r = 4.0, 2.0, 0.5
    m = plate_with_hole_2d(length=L, height=H, hole_radius=r, nx=60, ny=30)
    area = total_volume(m)
    expected = L * H - np.pi * r * r
    assert area == pytest.approx(expected, rel=0.02)


def test_plate_with_hole_quad4_cell_type():
    m = plate_with_hole_2d(cell_type="quad4", nx=20, ny=10)
    assert m.cell_type == "quad4"


# ---------------------------------------------------------------------------
# L-bracket
# ---------------------------------------------------------------------------
def test_l_bracket_area_formula():
    a, b, t = 2.0, 2.0, 1.0
    m = l_bracket_mesh_2d(a=a, b=b, t=t, n=24)
    area = total_volume(m)
    expected = t * (a + b - t)  # 1 * (2 + 2 - 1) = 3
    assert area == pytest.approx(expected, rel=1e-6)


def test_l_bracket_tri_count_is_double_quad():
    m_q = l_bracket_mesh_2d(cell_type="quad4", n=16)
    m_t = l_bracket_mesh_2d(cell_type="tri3", n=16)
    assert m_t.n_cells == 2 * m_q.n_cells


# ---------------------------------------------------------------------------
# Notched bar
# ---------------------------------------------------------------------------
def test_notched_bar_removes_central_cells():
    L, W, r = 4.0, 1.0, 0.25
    m = notched_bar_mesh_2d(length=L, width=W, notch_radius=r,
                            nx=40, ny=10, cell_type="quad4")
    # Area should be less than L*W (notch removed).
    area = total_volume(m)
    assert area < L * W
    # No cells should have their centroid inside the notch circle (centred at 0,0).
    nodes = np.asarray(m.nodes)
    cells = np.asarray(m.cells)
    centroids = nodes[cells].mean(axis=1)
    in_notch = (centroids[:, 0] ** 2 + centroids[:, 1] ** 2) <= r * r
    assert not np.any(in_notch)


def test_notched_bar_side_options():
    for side in ("both", "top", "bottom"):
        m = notched_bar_mesh_2d(side=side, nx=20, ny=8)
        assert m.n_cells > 0


def test_notched_bar_bad_side_raises():
    with pytest.raises(ValueError):
        notched_bar_mesh_2d(side="left")


# ---------------------------------------------------------------------------
# 3D cylinder
# ---------------------------------------------------------------------------
def test_cylinder_volume_approximates_pi_r2_h():
    r, h = 1.0, 2.0
    m = cylinder_mesh_3d(radius=r, height=h, n_radial=16, n_height=8)
    assert m.dim == 3
    assert m.cell_type == "hex8"
    vol = total_volume(m)
    assert vol == pytest.approx(np.pi * r * r * h, rel=0.05)


def test_cylinder_axis_orientation():
    for axis in ("x", "y", "z"):
        m = cylinder_mesh_3d(axis=axis, n_radial=8, n_height=4)
        assert m.dim == 3
        assert m.n_cells > 0


def test_cylinder_tet4_split():
    m = cylinder_mesh_3d(cell_type="tet4", n_radial=8, n_height=4)
    assert m.cell_type == "tet4"
    assert m.n_cells > 0


def test_cylinder_bad_axis_raises():
    with pytest.raises(ValueError):
        cylinder_mesh_3d(axis="w")


# ---------------------------------------------------------------------------
# 3D plate with hole
# ---------------------------------------------------------------------------
def test_plate_with_hole_3d_volume():
    L, H, T, r = 4.0, 2.0, 0.5, 0.5
    m = plate_with_hole_3d(length=L, height=H, thickness=T, hole_radius=r,
                           nx=20, ny=10, nz=4)
    vol = total_volume(m)
    expected = (L * H - np.pi * r * r) * T
    assert vol == pytest.approx(expected, rel=0.05)


# ---------------------------------------------------------------------------
# Integration: VTK round-trip for a primitive
# ---------------------------------------------------------------------------
def test_disc_vtk_roundtrip(tmp_path):
    m = disc_mesh_2d(radius=1.0, n=16, cell_type="tri3")
    path = tmp_path / "disc.vtu"
    write_mesh(path, m)
    m2 = read_mesh(path)
    assert m2.cell_type == m.cell_type
    assert m2.n_nodes == m.n_nodes
    assert m2.n_cells == m.n_cells


# ---------------------------------------------------------------------------
# Empty-mask safety
# ---------------------------------------------------------------------------
def test_empty_mask_raises():
    """A geometry that keeps zero cells should raise a clear error."""
    with pytest.raises(ValueError):
        # Hole larger than the plate -> everything removed
        plate_with_hole_2d(length=1.0, height=1.0, hole_radius=2.0,
                           nx=8, ny=8)
