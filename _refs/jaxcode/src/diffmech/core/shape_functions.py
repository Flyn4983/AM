"""Shape functions and their gradients on reference elements.

Reference elements used throughout DiffMech:
    - tri3  : triangle with vertices (0,0), (1,0), (0,1)
    - quad4 : square with vertices (-1,-1),(1,-1),(1,1),(-1,1)
    - tet4  : tetrahedron with vertices (0,0,0),(1,0,0),(0,1,0),(0,0,1)
    - hex8  : cube with vertices (±1, ±1, ±1)

Each ``*_shape`` returns shape-function values ``N(xi)`` of shape
``(n_nodes,)`` for a single point ``xi`` (or vmap'd ``(n_points, n_nodes)``).
Each ``*_grad`` returns the gradient ``dN/dxi`` w.r.t. the *reference*
coordinates of shape ``(n_nodes, dim)``. The caller is responsible for
multiplying by the inverse of the physical-element Jacobian to obtain
``dN/dx``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class ShapeFunction:
    """Bundles a shape function and its reference gradient + reference volume."""

    value: Callable[[jnp.ndarray], jnp.ndarray]
    grad: Callable[[jnp.ndarray], jnp.ndarray]
    n_nodes: int
    dim: int
    reference: str  # "tri", "quad", "tet"


# ------------------------------------------------------------------ tri3 ----
def tri3_shape(xi: jnp.ndarray) -> jnp.ndarray:
    """Shape functions on the reference triangle at point ``xi`` of shape (2,)."""
    xi = jnp.atleast_1d(xi)
    return jnp.array([1.0 - xi[0] - xi[1], xi[0], xi[1]])


def tri3_grad(xi: jnp.ndarray) -> jnp.ndarray:
    """Reference gradients dN/dxi for tri3, shape (3, 2)."""
    return jnp.array([
        [-1.0, -1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])


# ----------------------------------------------------------------- quad4 ----
def quad4_shape(xi: jnp.ndarray) -> jnp.ndarray:
    """Shape functions on the reference square [-1,1]^2 at ``xi`` of shape (2,)."""
    xi = jnp.atleast_1d(xi)
    x, y = xi[0], xi[1]
    return jnp.array([
        0.25 * (1 - x) * (1 - y),
        0.25 * (1 + x) * (1 - y),
        0.25 * (1 + x) * (1 + y),
        0.25 * (1 - x) * (1 + y),
    ])


def quad4_grad(xi: jnp.ndarray) -> jnp.ndarray:
    """Reference gradients dN/dxi for quad4, shape (4, 2)."""
    xi = jnp.atleast_1d(xi)
    x, y = xi[0], xi[1]
    return jnp.array([
        [-0.25 * (1 - y), -0.25 * (1 - x)],
        [0.25 * (1 - y), -0.25 * (1 + x)],
        [0.25 * (1 + y), 0.25 * (1 + x)],
        [-0.25 * (1 + y), 0.25 * (1 - x)],
    ])


# ----------------------------------------------------------------- tet4 -----
def tet4_shape(xi: jnp.ndarray) -> jnp.ndarray:
    """Shape functions on the reference tet at ``xi`` of shape (3,)."""
    xi = jnp.atleast_1d(xi)
    return jnp.array([
        1.0 - xi[0] - xi[1] - xi[2],
        xi[0], xi[1], xi[2],
    ])


def tet4_grad(xi: jnp.ndarray) -> jnp.ndarray:
    """Reference gradients dN/dxi for tet4, shape (4, 3)."""
    return jnp.array([
        [-1.0, -1.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])


# ----------------------------------------------------------------- hex8 -----
# Node ordering for the reference hex [-1,1]^3:
#   0: (-1,-1,-1)   1: (+1,-1,-1)   2: (+1,+1,-1)   3: (-1,+1,-1)
#   4: (-1,-1,+1)   5: (+1,-1,+1)   6: (+1,+1,+1)   7: (-1,+1,+1)
# (bottom face CCW, then top face CCW, matching VTK/Hexahedral convention)
def hex8_shape(xi: jnp.ndarray) -> jnp.ndarray:
    """Shape functions on the reference cube [-1,1]^3 at ``xi`` of shape (3,)."""
    xi = jnp.atleast_1d(xi)
    x, y, z = xi[0], xi[1], xi[2]
    xm, ym, zm = 1.0 - x, 1.0 - y, 1.0 - z
    xp, yp, zp = 1.0 + x, 1.0 + y, 1.0 + z
    o = 0.125
    return jnp.array([
        o * xm * ym * zm,   # 0
        o * xp * ym * zm,   # 1
        o * xp * yp * zm,   # 2
        o * xm * yp * zm,   # 3
        o * xm * ym * zp,   # 4
        o * xp * ym * zp,   # 5
        o * xp * yp * zp,   # 6
        o * xm * yp * zp,   # 7
    ])


def hex8_grad(xi: jnp.ndarray) -> jnp.ndarray:
    """Reference gradients dN/dxi for hex8, shape (8, 3)."""
    xi = jnp.atleast_1d(xi)
    x, y, z = xi[0], xi[1], xi[2]
    xm, ym, zm = 1.0 - x, 1.0 - y, 1.0 - z
    xp, yp, zp = 1.0 + x, 1.0 + y, 1.0 + z
    o = 0.125
    # dN/dx, dN/dy, dN/dz  for each node
    return jnp.array([
        [-o * ym * zm, -o * xm * zm, -o * xm * ym],   # 0
        [ o * ym * zm, -o * xp * zm, -o * xp * ym],   # 1
        [ o * yp * zm,  o * xp * zm, -o * xp * yp],   # 2
        [-o * yp * zm,  o * xm * zm, -o * xm * yp],   # 3
        [-o * ym * zp, -o * xm * zp,  o * xm * ym],   # 4
        [ o * ym * zp, -o * xp * zp,  o * xp * ym],   # 5
        [ o * yp * zp,  o * xp * zp,  o * xp * yp],   # 6
        [-o * yp * zp,  o * xm * zp,  o * xm * yp],   # 7
    ])


# ----------------------------------------------------------------- helpers --
def physical_gradient(ref_grad: jnp.ndarray, coords: jnp.ndarray) -> jnp.ndarray:
    """Map reference shape-function gradients to physical element.

    Parameters
    ----------
    ref_grad : (n_nodes, dim)
        dN/dxi at a single quadrature point.
    coords : (n_nodes, dim)
        Physical nodal coordinates of the element.

    Returns
    -------
    dN/dx : (n_nodes, dim)
    detJ : scalar (returned by :func:`physical_gradient_and_det` if needed)
    """
    dN_dxi = ref_grad  # (n_nodes, dim)
    # J = dN/dxi^T @ coords -> (dim, dim)
    J = dN_dxi.T @ coords
    Jinv = jnp.linalg.inv(J)
    return dN_dxi @ Jinv.T


def physical_gradient_and_det(ref_grad: jnp.ndarray, coords: jnp.ndarray):
    """Return (dN/dx, detJ) at a single point."""
    dN_dxi = ref_grad
    J = dN_dxi.T @ coords
    detJ = jnp.linalg.det(J)
    Jinv = jnp.linalg.inv(J)
    return dN_dxi @ Jinv.T, detJ


SHAPE_FUNCTIONS = {
    "tri3": ShapeFunction(tri3_shape, tri3_grad, 3, 2, "tri"),
    "quad4": ShapeFunction(quad4_shape, quad4_grad, 4, 2, "quad"),
    "tet4": ShapeFunction(tet4_shape, tet4_grad, 4, 3, "tet"),
    "hex8": ShapeFunction(hex8_shape, hex8_grad, 8, 3, "hex"),
}


def get_shape_function(cell_type: str) -> ShapeFunction:
    return SHAPE_FUNCTIONS[cell_type]
