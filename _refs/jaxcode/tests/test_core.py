"""Tests for the core layer: mesh, quadrature, shape functions."""

import jax.numpy as jnp
import numpy as np
import pytest

from diffmech.core import (
    Config, Mesh, StructuredMesh2D, rectangular_mesh2d,
    gauss_legendre_1d, gauss_legendre_nd,
    tri3_shape, tri3_grad,
    quad4_shape, quad4_grad,
    tet4_shape, tet4_grad,
)


# --- mesh --------------------------------------------------------------------
def test_rectangular_mesh_quad4():
    m = rectangular_mesh2d(nx=2, ny=3, lx=2.0, ly=3.0, cell_type="quad4")
    assert m.cell_type == "quad4"
    assert m.n_nodes == (2 + 1) * (3 + 1) == 12
    assert m.n_cells == 2 * 3 == 6
    assert m.nodes_per_cell == 4
    assert m.dim == 2
    assert isinstance(m, StructuredMesh2D)


def test_rectangular_mesh_tri3():
    m = rectangular_mesh2d(nx=1, ny=1, lx=1.0, ly=1.0, cell_type="tri3")
    assert m.cell_type == "tri3"
    assert m.n_cells == 2
    assert m.nodes_per_cell == 3


def test_mesh_jit_safe():
    """Mesh must be hashable/usable in jit context (frozen dataclass of arrays)."""
    m = rectangular_mesh2d(nx=2, ny=2, lx=1.0, ly=1.0)
    # All attributes are jax arrays or python str -> hashable
    hash(m)


# --- quadrature --------------------------------------------------------------
def test_gauss_legendre_1d_integrates_polynomial():
    x, w = gauss_legendre_1d(3)
    # int_{-1}^1 x^4 dx = 2/5 = 0.4 (3-point rule is exact up to degree 5)
    integral = jnp.sum(w * x ** 4)
    assert np.isclose(float(integral), 0.4, atol=1e-12)


def test_quad_quadrature_unit_square():
    qr = gauss_legendre_nd(2, 2, reference="quad")
    assert qr.n_points == 4
    # integral of 1 over [-1,1]^2 = 4
    integral = jnp.sum(qr.weights)
    assert np.isclose(float(integral), 4.0, atol=1e-12)


def test_tri_quadrature_unit_triangle():
    qr = gauss_legendre_nd(2, 2, reference="tri")
    assert qr.n_points == 3
    # integral of 1 over reference tri (area 1/2)
    integral = jnp.sum(qr.weights)
    assert np.isclose(float(integral), 0.5, atol=1e-12)


def test_tet_quadrature_unit_tet():
    qr = gauss_legendre_nd(2, 3, reference="tet")
    # integral of 1 over unit tet (vol 1/6)
    integral = jnp.sum(qr.weights)
    assert np.isclose(float(integral), 1 / 6, atol=1e-12)


# --- shape functions ---------------------------------------------------------
def test_tri3_partition_of_unity():
    for xi in [jnp.array([0.0, 0.0]), jnp.array([1.0, 0.0]),
               jnp.array([0.0, 1.0]), jnp.array([0.25, 0.3])]:
        N = tri3_shape(xi)
        assert np.isclose(float(jnp.sum(N)), 1.0, atol=1e-12)
        # At vertex 0: N=[1,0,0]; vertex 1: [0,1,0]; vertex 2: [0,0,1]
    N0 = tri3_shape(jnp.array([0.0, 0.0]))
    assert jnp.allclose(N0, jnp.array([1.0, 0.0, 0.0]))


def test_quad4_partition_of_unity():
    for xi in [jnp.array([-1.0, -1.0]), jnp.array([1.0, -1.0]),
               jnp.array([1.0, 1.0]), jnp.array([-1.0, 1.0]),
               jnp.array([0.0, 0.0])]:
        N = quad4_shape(xi)
        assert np.isclose(float(jnp.sum(N)), 1.0, atol=1e-12)
    N0 = quad4_shape(jnp.array([-1.0, -1.0]))
    assert jnp.allclose(N0, jnp.array([1.0, 0.0, 0.0, 0.0]))


def test_tet4_partition_of_unity():
    for xi in [jnp.array([0.0, 0.0, 0.0]), jnp.array([1.0, 0.0, 0.0]),
               jnp.array([0.0, 1.0, 0.0]), jnp.array([0.0, 0.0, 1.0]),
               jnp.array([0.1, 0.2, 0.3])]:
        N = tet4_shape(xi)
        assert np.isclose(float(jnp.sum(N)), 1.0, atol=1e-12)


# --- config ------------------------------------------------------------------
def test_config_float64_enables_x64():
    Config(dtype="float64")
    assert jnp.array(1.0).dtype == jnp.float64


def test_config_float32():
    # Reset and try float32
    cfg = Config(dtype="float32")
    assert cfg.jax_dtype == jnp.float32
