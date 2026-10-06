"""Pre-defined crystallographic slip systems (2D & 3D).

A slip system α is defined by:

- a slip direction ``b^α`` (unit vector along which dislocations glide), and
- a slip plane normal ``n^α`` (unit vector normal to the slip plane),

with ``b^α · n^α = 0``.

The **Schmid tensor** is the symmetric part of the dyadic product::

    P^α = sym(b^α ⊗ n^α) = ½ (b^α ⊗ n^α + n^α ⊗ b^α)

The resolved shear stress on system α is::

    τ^α = σ : P^α

All slip-system constructors return ``(slip_directions, slip_normals)`` as
``(n_slip, dim)`` float arrays, with directions and normals normalised to
unit length.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    """Normalise each row of a (n, dim) array to unit length."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms < 1e-30, 1.0, norms)
    return arr / norms


def schmid_tensors(
    slip_directions: jnp.ndarray, slip_normals: jnp.ndarray
) -> jnp.ndarray:
    """Compute Schmid tensors ``P^α = sym(b ⊗ n)`` for all slip systems.

    Parameters
    ----------
    slip_directions : (n_slip, dim) array
    slip_normals : (n_slip, dim) array

    Returns
    -------
    P : (n_slip, dim, dim) array
        Symmetric Schmid tensors, one per slip system.
    """
    b = jnp.asarray(slip_directions)  # (n_slip, dim)
    n = jnp.asarray(slip_normals)     # (n_slip, dim)
    bn = jnp.einsum("si,sj->sij", b, n)        # (n_slip, dim, dim)
    P = 0.5 * (bn + jnp.transpose(bn, (0, 2, 1)))
    return P


def resolved_shear_stress(
    sigma: jnp.ndarray, P: jnp.ndarray
) -> jnp.ndarray:
    """Resolved shear stress ``τ^α = σ : P^α`` for all slip systems.

    Parameters
    ----------
    sigma : (dim, dim) stress tensor
    P : (n_slip, dim, dim) Schmid tensors

    Returns
    -------
    tau : (n_slip,) resolved shear stresses
    """
    return jnp.einsum("ij,sij->s", sigma, P)


# ---------------------------------------------------------------------------
# 2D slip systems
# ---------------------------------------------------------------------------
def double_slip_2d(theta_deg: float = 60.0) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Two slip systems in the (x, y) plane at angles ±θ from the x-axis.

    This is the canonical "double-slip" model used in 2D crystal-plasticity
    benchmarks (Asaro, Pierce). System 1 glides at +θ, system 2 at −θ.

    Parameters
    ----------
    theta_deg : float
        Angle (degrees) of the first slip direction from the x-axis.

    Returns
    -------
    slip_directions : (2, 2) array
    slip_normals : (2, 2) array
    """
    theta = np.deg2rad(theta_deg)
    b1 = np.array([np.cos(theta), np.sin(theta)])
    n1 = np.array([-np.sin(theta), np.cos(theta)])
    b2 = np.array([np.cos(theta), -np.sin(theta)])
    n2 = np.array([np.sin(theta), np.cos(theta)])
    dirs = np.stack([b1, b2])
    norms = np.stack([n1, n2])
    return jnp.asarray(dirs), jnp.asarray(norms)


def single_slip_2d(theta_deg: float = 45.0) -> tuple[jnp.ndarray, jnp.ndarray]:
    """A single slip system in the (x, y) plane.

    Useful for clean analytical benchmarks (e.g. simple shear on one system).
    """
    theta = np.deg2rad(theta_deg)
    b = np.array([[np.cos(theta), np.sin(theta)]])
    n = np.array([[-np.sin(theta), np.cos(theta)]])
    return jnp.asarray(b), jnp.asarray(n)


# ---------------------------------------------------------------------------
# 3D slip systems
# ---------------------------------------------------------------------------
def fcc_slip_systems() -> tuple[jnp.ndarray, jnp.ndarray]:
    """12 FCC ``{111}<110>`` slip systems.

    Four ``{111}`` slip planes, each with three ``<110>`` slip directions
    lying in the plane (12 systems total).

    Returns
    -------
    slip_directions : (12, 3) array, unit vectors
    slip_normals : (12, 3) array, unit vectors
    """
    # (plane normal, [slip directions in that plane])
    systems = [
        # (1 1 1)
        ((1, 1, 1), [(1, -1, 0), (0, 1, -1), (1, 0, -1)]),
        # (-1 1 1)
        ((-1, 1, 1), [(1, 1, 0), (0, 1, -1), (1, 0, 1)]),
        # (1 -1 1)
        ((1, -1, 1), [(1, 1, 0), (0, 1, 1), (1, 0, -1)]),
        # (1 1 -1)
        ((1, 1, -1), [(1, -1, 0), (0, 1, 1), (1, 0, 1)]),
    ]
    dirs, norms = [], []
    for plane, dlist in systems:
        n = np.array(plane, dtype=np.float64)
        n = n / np.linalg.norm(n)
        for d in dlist:
            b = np.array(d, dtype=np.float64)
            b = b / np.linalg.norm(b)
            dirs.append(b)
            norms.append(n)
    dirs = np.array(dirs)
    norms = np.array(norms)
    return jnp.asarray(dirs), jnp.asarray(norms)


def bcc_slip_systems() -> tuple[jnp.ndarray, jnp.ndarray]:
    """12 BCC ``{110}<111>`` slip systems.

    Six ``{110}`` slip planes, each with two ``<111>`` slip directions
    lying in the plane (12 systems total). This is the primary BCC family;
    ``{112}`` and ``{123}`` families can be added for higher fidelity.

    Returns
    -------
    slip_directions : (12, 3) array, unit vectors
    slip_normals : (12, 3) array, unit vectors
    """
    # All <111> directions (4 families, each ± gives the same line, so 4 unique)
    dirs_all = [(1, 1, 1), (1, -1, 1), (1, 1, -1), (1, -1, -1)]
    # All {110} planes
    planes = [
        (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1),
        (0, 1, 1), (0, 1, -1),
    ]
    dirs, norms = [], []
    for plane in planes:
        n = np.array(plane, dtype=np.float64)
        n = n / np.linalg.norm(n)
        for d in dirs_all:
            b = np.array(d, dtype=np.float64)
            if abs(np.dot(b, n)) < 1e-10:  # direction lies in the plane
                b = b / np.linalg.norm(b)
                dirs.append(b)
                norms.append(n)
    dirs = np.array(dirs)
    norms = np.array(norms)
    return jnp.asarray(dirs), jnp.asarray(norms)


def von_mises_fcc_slip_systems() -> tuple[jnp.ndarray, jnp.ndarray]:
    """FCC slip systems but augmented to give an effectively isotropic response.

    Same as :func:`fcc_slip_systems` but with directions and normals rotated
    to sample multiple crystal orientations. Currently an alias for
    :func:`fcc_slip_systems` (single orientation).
    """
    return fcc_slip_systems()
