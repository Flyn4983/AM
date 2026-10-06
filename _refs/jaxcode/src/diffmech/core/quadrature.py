"""Gauss-Legendre quadrature rules for points and weights.

Rules are precomputed (NumPy via NumPy's :func:`numpy.polynomial.legendre.leggauss`)
and exposed as static jax arrays so they can be staged into ``jit``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


def gauss_legendre_1d(n: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return (points, weights) on [-1, 1]."""
    if n < 1:
        raise ValueError("n must be >= 1")
    x, w = np.polynomial.legendre.leggauss(n)
    return jnp.asarray(x, dtype=jnp.float64), jnp.asarray(w, dtype=jnp.float64)


@dataclass(frozen=True)
class QuadratureRule:
    """A quadrature rule on the reference element.

    Attributes
    ----------
    points : (n_qp, dim) array
        Quadrature points on the reference element.
    weights : (n_qp,) array
        Quadrature weights (already multiplied by the Jacobian factor for the
        reference element volume, e.g. 1/2 for tri, 4 for quad, 1/6 for tet).
    """

    points: jnp.ndarray
    weights: jnp.ndarray

    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def dim(self) -> int:
        return int(self.points.shape[1])


def gauss_legendre_nd(n_per_dim: int, dim: int, *, reference: str) -> QuadratureRule:
    """Tensor-product Gauss-Legendre rule on the reference element.

    Parameters
    ----------
    n_per_dim : int
        Number of Gauss points per dimension.
    dim : int
        Spatial dimension (2 or 3).
    reference : {"quad", "tri", "tet"}
        Reference element. For ``"quad"`` and ``"tet"`` we use tensor-product
        rules on the unit cube / simplex-appropriate mapped rule. For ``"tri"``/
        ``"tet"`` we use symmetric Dunavant rules for low orders, falling back
        to a tensor product on a split square for higher orders.
    """
    if reference in ("quad", "hex"):
        # Tensor-product Gauss-Legendre on the reference hypercube [-1,1]^dim.
        # Weights already include the reference volume (the integral of 1 over
        # [-1,1]^d equals 2^d = sum of all weights).
        x, w = gauss_legendre_1d(n_per_dim)
        grids = jnp.meshgrid(*([x] * dim), indexing="ij")
        pts = jnp.stack([g.ravel() for g in grids], axis=-1)
        if dim == 1:
            ws = w
        elif dim == 2:
            ws = jnp.einsum("i,j->ij", w, w).ravel()
        else:  # dim == 3
            ws = jnp.einsum("i,j,k->ijk", w, w, w).ravel()
        return QuadratureRule(points=pts, weights=ws)

    if reference == "tri":
        # Dunavant rules for triangle of order up to 5 (n_per_dim proxy)
        rules = _dunavant_triangle(min(n_per_dim, 5))
        pts, ws = rules
        # Reference triangle area = 1/2; multiply weights by 1/2 so that
        # sum(w) * f -> integral.
        ws = ws * 0.5
        return QuadratureRule(points=jnp.asarray(pts), weights=jnp.asarray(ws))

    if reference == "tet":
        # Keaster rules for tetrahedron
        pts, ws = _keaster_tetrahedron(min(n_per_dim, 4))
        ws = ws / 6.0  # reference tet volume
        return QuadratureRule(points=jnp.asarray(pts), weights=jnp.asarray(ws))

    raise ValueError(f"unknown reference element: {reference}")


# --- Dunavant rules on the reference triangle (vertices (0,0),(1,0),(0,1)) ---
def _dunavant_triangle(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Dunavant symmetric quadrature on the unit triangle.

    Coordinates are barycentric-free; we use the standard triangle with
    vertices (0,0), (1,0), (0,1). Weights are *unscaled* (i.e., sum to 1/2).
    """
    if order == 1:
        pts = np.array([[1 / 3, 1 / 3]])
        ws = np.array([1.0])
    elif order == 2:
        a = 2 / 3
        b = 1 / 6
        pts = np.array([
            [a, b], [b, a], [b, b],
        ])
        ws = np.array([1 / 3, 1 / 3, 1 / 3])
    elif order == 3:
        pts = np.array([
            [1 / 3, 1 / 3],
            [0.6, 0.2], [0.2, 0.6], [0.2, 0.2],
        ])
        ws = np.array([-27 / 96, 25 / 96, 25 / 96, 25 / 96])
    elif order == 4:
        a = 0.470142064105115
        b = 0.101286507323456
        pts = np.array([
            [1 / 3, 1 / 3],
            [a, a], [a, b], [b, a],
            [b, b], [b, b],  # placeholder, will fill below
        ])
        # Use a cleaner order-4 rule (5-pt):
        a1 = 0.4701420641051151
        b1 = 0.1012865073234563
        pts = np.array([
            [1 / 3, 1 / 3],
            [a1, a1], [a1, b1], [b1, a1],
            [b1, b1],
        ])
        ws = np.array([
            9 / 40,
            (155 - jnp.sqrt(15)) / 1200 * 6 * 0,
        ])
        ws = np.array([0.225, 0.132394152788506, 0.132394152788506,
                       0.132394152788506, 0.132394152788506])
        # Normalize: area = 1/2; weights should sum to 1/2.
        ws = ws * 0.5 / ws.sum()
    elif order == 5:
        # 7-point degree-5 rule
        pts = np.array([
            [1 / 3, 1 / 3],
            [0.059715871789770, 0.4701420641051151],
            [0.4701420641051151, 0.059715871789770],
            [0.4701420641051151, 0.4701420641051151],
            [0.797426985353087, 0.1012865073234563],
            [0.1012865073234563, 0.797426985353087],
            [0.1012865073234563, 0.1012865073234563],
        ])
        ws = np.array([
            0.225,
            0.1323941527885061, 0.1323941527885061, 0.1323941527885061,
            0.1259391805448271, 0.1259391805448271, 0.1259391805448271,
        ])
        ws = ws * 0.5 / ws.sum()
    else:
        raise ValueError(f"unsupported order {order}")
    return pts.astype(np.float64), ws.astype(np.float64)


def _keaster_tetrahedron(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Quadrature rules on the unit tet (vertices (0,0,0),(1,0,0),(0,1,0),(0,0,1)).

    Weights are *unscaled* (sum to 1/6).
    """
    if order == 1:
        pts = np.array([[1 / 4, 1 / 4, 1 / 4]])
        ws = np.array([1.0])
    elif order == 2:
        a = 0.5854101966249685
        b = 0.1381966011250105
        pts = np.array([
            [a, b, b], [b, a, b], [b, b, a], [b, b, b],
        ])
        ws = np.array([0.25, 0.25, 0.25, 0.25])
    elif order == 3:
        # 5-point degree-3 (Keaster)
        pts = np.array([
            [1 / 4, 1 / 4, 1 / 4],
            [1 / 2, 1 / 6, 1 / 6], [1 / 6, 1 / 2, 1 / 6],
            [1 / 6, 1 / 6, 1 / 2], [1 / 6, 1 / 6, 1 / 6],
        ])
        ws = np.array([-0.8, 0.45, 0.45, 0.45, 0.45])
        ws = ws * (1 / 6) / ws.sum()
    elif order == 4:
        # 11-point degree-4 (Keaster)
        a = 0.3994035761667992
        b = 0.1005964238332008
        c = 1 / 11
        # 1 centroid + 4 corner-shifted + 6 edge midpoints
        pts = np.array([
            [1 / 4, 1 / 4, 1 / 4],
            [a, b, b], [b, a, b], [b, b, a], [b, b, b],
            [1 / 2, 1 / 2, 0], [1 / 2, 0, 1 / 2], [0, 1 / 2, 1 / 2],
            [1 / 2, 0, 0], [0, 1 / 2, 0], [0, 0, 1 / 2],
        ])
        ws = np.array([
            -74 / 3500 * 6,
            4 / 350 * 6, 4 / 350 * 6, 4 / 350 * 6, 4 / 350 * 6,
            1 / 350 * 6, 1 / 350 * 6, 1 / 350 * 6,
            1 / 350 * 6, 1 / 350 * 6, 1 / 350 * 6,
        ])
        ws = ws * (1 / 6) / ws.sum()
    else:
        raise ValueError(f"unsupported order {order}")
    return pts.astype(np.float64), ws.astype(np.float64)
