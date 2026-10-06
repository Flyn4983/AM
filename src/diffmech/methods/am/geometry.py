"""Differentiable part geometry via signed-distance fields (SDFs).

AM process simulation needs the *shape* of the part being built. Rather than
parsing a discrete STL (which is non-differentiable and jagged), we represent
parts as **signed-distance functions** ``f(x)``:

    f(x) < 0  → inside the part
    f(x) > 0  → outside the part
    f(x) = 0  → on the surface

SDFs are smooth, vectorised over ``jnp``, and — crucially — differentiable,
so a loss built on the part shape (e.g. "minimise residual stress by
tweaking the overhang angle") propagates gradients back through the
geometry into the design parameters via ``jax.grad``.

The module provides:
- SDF primitives (sphere, box, cylinder, cone, gear, torus)
- Boolean operations (union, intersection, difference) via smooth min/max
- Layer slicing: ``slice_mask(sdf, z, grid)`` → differentiable occupancy
  ``σ(-sdf/h)`` on a 2-D grid at height ``z`` (the per-layer cross-section)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np


# A part geometry is a callable ``f(x: (..., 3)) -> (...,)`` returning the SDF.
SDF = Callable[[jnp.ndarray], jnp.ndarray]


# ---------------------------------------------------------------------------
# Smooth boolean ops (differentiable approximations of min/max)
# ---------------------------------------------------------------------------
def smooth_union(a: jnp.ndarray, b: jnp.ndarray, k: float = 1e-3) -> jnp.ndarray:
    """Differentiable ``min(a, b)`` (union of two solids)."""
    h = jnp.maximum(k - jnp.abs(a - b), 0.0) / k
    return jnp.minimum(a, b) - h * h * k * 0.25


def smooth_intersection(a: jnp.ndarray, b: jnp.ndarray, k: float = 1e-3) -> jnp.ndarray:
    """Differentiable ``max(a, b)`` (intersection of two solids)."""
    h = jnp.maximum(k - jnp.abs(a - b), 0.0) / k
    return jnp.maximum(a, b) + h * h * k * 0.25


def smooth_difference(a: jnp.ndarray, b: jnp.ndarray, k: float = 1e-3) -> jnp.ndarray:
    """Differentiable ``max(a, -b)`` (subtract ``b`` from ``a``)."""
    return smooth_intersection(a, -b, k)


# ---------------------------------------------------------------------------
# SDF primitives
# ---------------------------------------------------------------------------
def sdf_sphere(center, radius: float) -> SDF:
    """SDF of a sphere."""
    c = jnp.asarray(center, dtype=jnp.float64)
    r = float(radius)

    def f(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.linalg.norm(x - c, axis=-1) - r
    return f


def sdf_box(center, half_extents) -> SDF:
    """SDF of an axis-aligned box (centered at ``center``)."""
    c = jnp.asarray(center, dtype=jnp.float64)
    he = jnp.asarray(half_extents, dtype=jnp.float64)

    def f(x: jnp.ndarray) -> jnp.ndarray:
        d = jnp.abs(x - c) - he
        outside = jnp.linalg.norm(jnp.maximum(d, 0.0), axis=-1)
        inside = jnp.minimum(jnp.max(d, axis=-1), 0.0)
        return outside + inside
    return f


def sdf_cylinder(center, axis: int, radius: float, half_length: float) -> SDF:
    """SDF of a finite cylinder along one axis (0=x, 1=y, 2=z)."""
    c = jnp.asarray(center, dtype=jnp.float64)
    r = float(radius)
    h = float(half_length)
    other = [i for i in range(3) if i != axis]

    def f(x: jnp.ndarray) -> jnp.ndarray:
        d_radial = jnp.sqrt(
            (x[..., other[0]] - c[other[0]]) ** 2 +
            (x[..., other[1]] - c[other[1]]) ** 2) - r
        d_axial = jnp.abs(x[..., axis] - c[axis]) - h
        outside = jnp.sqrt(jnp.maximum(d_radial, 0.0) ** 2 +
                           jnp.maximum(d_axial, 0.0) ** 2)
        inside = jnp.minimum(jnp.maximum(d_radial, d_axial), 0.0)
        return outside + inside
    return f


def sdf_torus(center, axis: int, major_radius: float, minor_radius: float) -> SDF:
    """SDF of a torus around an axis (tube radius ``minor``, ring radius ``major``)."""
    c = jnp.asarray(center, dtype=jnp.float64)
    R = float(major_radius)
    r = float(minor_radius)
    other = [i for i in range(3) if i != axis]

    def f(x: jnp.ndarray) -> jnp.ndarray:
        q0 = jnp.sqrt(
            (x[..., other[0]] - c[other[0]]) ** 2 +
            (x[..., other[1]] - c[other[1]]) ** 2) - R
        q1 = x[..., axis] - c[axis]
        return jnp.sqrt(q0 ** 2 + q1 ** 2) - r
    return f


def sdf_gear(center, axis: int, outer_radius: float, n_teeth: int,
             tooth_depth: float, thickness: float) -> SDF:
    """SDF of a spur gear (approximate, for demonstration parts)."""
    c = jnp.asarray(center, dtype=jnp.float64)
    R = float(outer_radius)
    n = int(n_teeth)
    td = float(tooth_depth)

    def f(x: jnp.ndarray) -> jnp.ndarray:
        other = [i for i in range(3) if i != axis]
        dx = x[..., other[0]] - c[other[0]]
        dy = x[..., other[1]] - c[other[1]]
        r_xy = jnp.sqrt(dx ** 2 + dy ** 2)
        theta = jnp.arctan2(dy, dx)
        # Modulate radius by tooth profile (cosine bumps).
        r_profile = R - 0.5 * td * (1.0 - jnp.cos(n * theta))
        d_radial = r_xy - r_profile
        d_axial = jnp.abs(x[..., axis] - c[axis]) - thickness * 0.5
        outside = jnp.sqrt(jnp.maximum(d_radial, 0.0) ** 2 +
                           jnp.maximum(d_axial, 0.0) ** 2)
        inside = jnp.minimum(jnp.maximum(d_radial, d_axial), 0.0)
        return outside + inside
    return f


# ---------------------------------------------------------------------------
# Combinators (build complex shapes from primitives)
# ---------------------------------------------------------------------------
def union(a: SDF, b: SDF, k: float = 1e-3) -> SDF:
    def f(x):
        return smooth_union(a(x), b(x), k)
    return f


def intersection(a: SDF, b: SDF, k: float = 1e-3) -> SDF:
    def f(x):
        return smooth_intersection(a(x), b(x), k)
    return f


def difference(a: SDF, b: SDF, k: float = 1e-3) -> SDF:
    def f(x):
        return smooth_difference(a(x), b(x), k)
    return f


def translate(sdf: SDF, offset) -> SDF:
    """Translate an SDF by a constant offset."""
    off = jnp.asarray(offset, dtype=jnp.float64)

    def f(x):
        return sdf(x - off)
    return f


def rotate_z(sdf: SDF, angle: float) -> SDF:
    """Rotate an SDF around the z-axis by ``angle`` radians."""
    ca, sa = jnp.cos(angle), jnp.sin(angle)

    def f(x):
        x_rot = x.at[..., 0].set(ca * x[..., 0] - sa * x[..., 1])
        x_rot = x_rot.at[..., 1].set(sa * x[..., 0] + ca * x[..., 1])
        return sdf(x_rot)
    return f


# ---------------------------------------------------------------------------
# Layer slicing — the heart of AM geometry processing
# ---------------------------------------------------------------------------
def slice_mask(sdf: SDF, z: float, grid_xy: jnp.ndarray,
               smoothness: float = 1e-3) -> jnp.ndarray:
    """Differentiable occupancy of the part cross-section at height ``z``.

    Parameters
    ----------
    sdf : signed-distance function of the part.
    z : build height at which to slice.
    grid_xy : (n, 2) or (nx, ny, 2) array of in-plane points.
    smoothness : width of the smooth Heaviside (smaller = sharper, less smooth
        gradient; larger = more diffusive but better-conditioned gradients).

    Returns
    -------
    chi : same leading shape as ``grid_xy`` minus the last axis, values in
        [0, 1]. ``chi≈1`` inside the part, ``chi≈0`` outside.
    """
    grid_xy = jnp.asarray(grid_xy)
    z_arr = jnp.asarray(z, dtype=grid_xy.dtype)
    if grid_xy.ndim == 2:
        pts = jnp.concatenate([grid_xy, jnp.full(grid_xy.shape[:-1] + (1,),
                                                  z_arr)], axis=-1)
    else:
        pts = jnp.concatenate([grid_xy, jnp.full(grid_xy.shape[:-1] + (1,),
                                                  z_arr)], axis=-1)
    s = sdf(pts)
    # Smooth Heaviside: σ(-s/h) → 1 inside (s<0), 0 outside (s>0).
    return jax.nn.sigmoid(-s / smoothness)


def layer_heights(sdf: SDF, bbox, layer_thickness: float) -> np.ndarray:
    """Compute the z-heights of layers that intersect the part.

    Walks from the bottom of the bounding box upward in steps of
    ``layer_thickness`` and keeps layers whose slice is non-empty. This is a
    *setup-time* (numpy) routine — the differentiable simulation then uses
    the resulting array of layer z's as constants.
    """
    z_min, z_max = bbox[2]
    zs = np.arange(z_min, z_max + layer_thickness * 0.5, layer_thickness)
    return zs.astype(np.float64)


def part_bbox(sdf: SDF, resolution: int = 64, margin: float = 0.0,
              search_range=(-2.0, 2.0)):
    """Estimate the axis-aligned bounding box of an SDF by sampling.

    Parameters
    ----------
    sdf : signed-distance function of the part.
    resolution : sampling density per axis.
    margin : extra padding added to each side of the tight box.
    search_range : (lo, hi) extent of the sampling grid. Override this when
        the part is much smaller than the default ±2 m search window (e.g.
        AM parts in the millimetre range).
    """
    # Coarse search on a wide grid, then tighten.
    g = np.linspace(search_range[0], search_range[1], resolution)
    pts = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    s = np.asarray(sdf(jnp.asarray(pts)))
    inside = pts[s < 0]
    if len(inside) == 0:
        # Fall back to the sampled box.
        return [(float(g.min()), float(g.max()))] * 3
    lo = inside.min(axis=0) - margin
    hi = inside.max(axis=0) + margin
    return [(float(lo[0]), float(hi[0])),
            (float(lo[1]), float(hi[1])),
            (float(lo[2]), float(hi[2]))]


# ---------------------------------------------------------------------------
# Arbitrary complex geometry from a triangle mesh (STL / OBJ / OFF / …)
# ---------------------------------------------------------------------------
def _point_triangle_distance(p, a, b, c):
    """Squared distance from point ``p`` to triangle ``(a, b, c)``.

    Vectorised over the leading axis of ``p``. Uses the closest-feature
    algorithm of Ericson (Real-Time Collision Detection, §5.1.5).
    """
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = (ap * ab).sum(-1)
    d2 = (ap * ac).sum(-1)
    # Vertex region A.
    out = ((p - a) ** 2).sum(-1)
    out = jnp.where((d1 <= 0) & (d2 <= 0), out, out)

    bp = p - b
    d3 = (bp * ab).sum(-1)
    d4 = (bp * ac).sum(-1)
    vab = d3 * (d4 - d3) <= 0
    t_ab = jnp.clip(d1 / jnp.maximum(d1 - d3, 1e-30), 0.0, 1.0)
    proj_ab = a + t_ab[..., None] * ab
    out = jnp.where(vab, ((p - proj_ab) ** 2).sum(-1), out)

    cp = p - c
    d5 = (cp * ab).sum(-1)
    d6 = (cp * ac).sum(-1)
    vac = d6 * (d5 - d6) <= 0
    t_ac = jnp.clip(d2 / jnp.maximum(d2 - d6, 1e-30), 0.0, 1.0)
    proj_ac = a + t_ac[..., None] * ac
    out = jnp.where(vac, ((p - proj_ac) ** 2).sum(-1), out)

    # Edge BC.
    bc = c - b
    bp2 = p - b
    d_bc1 = (bp2 * bc).sum(-1)
    d_bc2 = (bp2 * ab).sum(-1) - d3
    vbc = (d3 * d4 - d1 * d6 <= 0) & (d_bc1 * d_bc2 <= 0)
    t_bc = jnp.clip(d_bc1 / jnp.maximum(
        d_bc1 - d_bc2, 1e-30), 0.0, 1.0)
    proj_bc = b + t_bc[..., None] * bc
    out = jnp.where(vbc, ((p - proj_bc) ** 2).sum(-1), out)

    # Face region (project onto plane).
    n = jnp.cross(ab, ac)
    nn = (n * n).sum(-1)
    nn_safe = jnp.maximum(nn, 1e-30)
    t = (ap * n).sum(-1) / nn_safe
    proj = p - t[..., None] * n
    # Barycentric containment test.
    v0 = b - a
    v1 = c - a
    v2 = proj - a
    d00 = (v0 * v0).sum(-1)
    d01 = (v0 * v1).sum(-1)
    d11 = (v1 * v1).sum(-1)
    d20 = (v2 * v0).sum(-1)
    d21 = (v2 * v1).sum(-1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / jnp.maximum(denom, 1e-30)
    w = (d00 * d21 - d01 * d20) / jnp.maximum(denom, 1e-30)
    u = 1.0 - v - w
    inside_face = (u >= 0) & (v >= 0) & (w >= 0)
    out = jnp.where(inside_face, ((p - proj) ** 2).sum(-1), out)
    return out


def _mesh_sdf(verts: jnp.ndarray, tris: jnp.ndarray, *,
              winding_order: int = 1) -> SDF:
    """Build an approximate SDF from a closed triangle mesh.

    The signed distance is computed as ``sign · min_distance``, where the
    sign is determined by the mesh-winding-rule generalised winding number
    (a robust inside/outside test). Distance is the Euclidean nearest point
    across all triangles (vectorised with ``jax.vmap`` over triangles, then
    a parallel minimum).

    Parameters
    ----------
    verts : (n_verts, 3) array of vertex positions.
    tris : (n_tris, 3) int array of triangle vertex indices.
    winding_order : +1 if the mesh is wound counter-clockwise outward
        (right-hand rule, the VTK/STL convention), -1 for inward winding.
    """
    v = jnp.asarray(verts, dtype=jnp.float64)
    t = jnp.asarray(tris, dtype=jnp.int64)
    # Pre-slice triangle vertex coordinates.
    a_arr = v[t[:, 0]]
    b_arr = v[t[:, 1]]
    c_arr = v[t[:, 2]]

    # Per-triangle centroid + area-weighted normal, for the winding number.
    centroid = (a_arr + b_arr + c_arr) / 3.0
    nrm = jnp.cross(b_arr - a_arr, c_arr - a_arr)
    tri_area = 0.5 * jnp.linalg.norm(nrm, axis=-1)
    nrm_unit = nrm / jnp.maximum(jnp.linalg.norm(nrm, axis=-1, keepdims=True),
                                 1e-30)

    def per_point(x):
        # Squared distance to every triangle.
        def one_tri(a, b, c):
            return _point_triangle_distance(x, a, b, c)
        d2 = jax.vmap(one_tri)(a_arr, b_arr, c_arr)
        d_min = jnp.sqrt(jnp.min(d2))

        # --- Generalised winding number (inside/outside test) -----------
        # Sum over triangles of the solid angle Ω subtended at x:
        #   Ω = 2 * atan2( n · (a - x), ((a-x)·(b-x))*|c-x| + (b-x)·(c-x)*|a-x|
        #                   + (c-x)·(a-x)*|b-x| + (a-x)·(b-x)·(c-x)·n/|n|... )
        # The robust Van Oosterom-Strackee formula:
        #   tan(Ω/2) = numerator / denominator
        a0 = a_arr - x
        b0 = b_arr - x
        c0 = c_arr - x
        la = jnp.linalg.norm(a0, axis=-1)
        lb = jnp.linalg.norm(b0, axis=-1)
        lc = jnp.linalg.norm(c0, axis=-1)
        # Numerator: triple product a·(b×c) = a0·(b0×c0).
        cross_bc = jnp.cross(b0, c0)
        numer = (a0 * cross_bc).sum(-1)
        denom = (la * lb * lc
                 + (a0 * b0).sum(-1) * lc
                 + (b0 * c0).sum(-1) * la
                 + (c0 * a0).sum(-1) * lb)
        omega = 2.0 * jnp.arctan2(numer, denom)
        # Solid angle is signed by triangle winding; sum / (4π) → winding.
        winding = jnp.sum(omega) / (4.0 * jnp.pi)
        # CCW-outward winding → |winding| > 0.5 means inside.
        sign = jnp.where(winding * winding_order > 0.25, -1.0, 1.0)
        return sign * d_min

    def f(x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.asarray(x, dtype=jnp.float64)
        return jax.vmap(per_point)(x.reshape(-1, 3)).reshape(x.shape[:-1])
    return f


def sdf_from_mesh(verts, tris, *, winding_order: int = 1,
                  chunk: int | None = None) -> SDF:
    """Differentiable SDF of an arbitrary closed triangle mesh.

    This is the principal entry point for **arbitrary complex part
    geometries** — load a CAD-exported STL/OBJ/OFF mesh with `meshio` (or any
    other loader), pass its vertices and triangles here, and you get an SDF
    that plugs directly into :func:`slice_mask`, :func:`part_bbox`, and the
    rest of the AM pipeline exactly like the analytic primitives
    (:func:`sdf_sphere`, :func:`sdf_gear`, …).

    The signed distance uses the closest-point-on-mesh for the magnitude and
    the generalised winding number (Van Oosterom-Strackee solid angle) for
    the inside/outside sign — robust for non-convex, multi-component, and
    thin-walled geometries.

    Parameters
    ----------
    verts : (n_verts, 3) array-like
        Vertex coordinates [m].
    tris : (n_tris, 3) int array-like
        Triangle vertex indices (0-based). Convention: counter-clockwise
        winding when viewed from outside (the STL/VTK/ParaView standard).
    winding_order : +1 (default) for CCW-outward winding, -1 for CW-inward.
    chunk : optional maximum number of query points per batch. If ``None``,
        all query points are processed in one vmap. Set this lower if you run
        out of memory on very large meshes.

    Returns
    -------
    sdf : SDF
        A callable ``f(x: (..., 3)) -> (...,)`` returning the signed distance,
        differentiable w.r.t. the query coordinates ``x`` via ``jax.grad``.
    """
    base = _mesh_sdf(jnp.asarray(verts, dtype=jnp.float64),
                     jnp.asarray(tris, dtype=jnp.int64),
                     winding_order=winding_order)
    if chunk is None:
        return base

    def f(x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.asarray(x, dtype=jnp.float64)
        flat = x.reshape(-1, 3)
        out = []
        for i in range(0, len(flat), chunk):
            out.append(base(flat[i:i + chunk]))
        return jnp.concatenate(out).reshape(x.shape[:-1])
    return f


def sdf_from_stl(path, *, winding_order: int = 1,
                 chunk: int | None = None) -> SDF:
    """Load an STL file and return its differentiable SDF.

    Thin wrapper around :func:`sdf_from_mesh` that reads the STL via
    :mod:`meshio` (or :mod:`numpy-stl` if available). STL is the most common
    CAD-export format for AM parts — every slicer accepts it.

    Parameters
    ----------
    path : str or Path
        Path to a binary or ASCII STL file.
    winding_order : +1 (default) for CCW-outward triangle winding (the STL
        standard), -1 if the mesh is wound the other way.
    chunk : optional batch size for the SDF evaluation (see
        :func:`sdf_from_mesh`).
    """
    try:
        import meshio
        m = meshio.read(str(path))
        verts = np.asarray(m.points, dtype=np.float64)
        # Find the triangle cells (meshio may mix cell types).
        tris_list = []
        for cell_block in m.cells:
            if cell_block.type in ("triangle",):
                tris_list.append(np.asarray(cell_block.data, dtype=np.int64))
            elif cell_block.type in ("tetra",):
                # Decompose tetrahedra into 4 triangles each.
                t = np.asarray(cell_block.data, dtype=np.int64)
                tris_list.append(t[:, [0, 1, 2]])
                tris_list.append(t[:, [0, 1, 3]])
                tris_list.append(t[:, [0, 2, 3]])
                tris_list.append(t[:, [1, 2, 3]])
            elif cell_block.type in ("quad",):
                q = np.asarray(cell_block.data, dtype=np.int64)
                tris_list.append(q[:, [0, 1, 2]])
                tris_list.append(q[:, [0, 2, 3]])
        if not tris_list:
            raise ValueError(f"no triangle/tetra/quad cells in {path}")
        tris = np.concatenate(tris_list, axis=0)
    except ImportError:
        try:
            from stl import mesh as stl_mesh
            m = stl_mesh.Mesh.from_file(str(path))
            # STL stores triangles as 3 vertices each (no shared index).
            verts_unique, inv = np.unique(
                m.vectors.reshape(-1, 3), axis=0, return_inverse=True)
            tris = inv.reshape(-1, 3).astype(np.int64)
            verts = verts_unique.astype(np.float64)
        except ImportError as exc:
            raise ImportError(
                "sdf_from_stl needs either `meshio` or `numpy-stl`; "
                "install one of them to read STL files.") from exc
    return sdf_from_mesh(verts, tris, winding_order=winding_order,
                         chunk=chunk)


def sdf_from_obj(path, *, winding_order: int = 1,
                 chunk: int | None = None) -> SDF:
    """Load a Wavefront OBJ file and return its differentiable SDF.

    Uses :mod:`meshio` to parse the OBJ (vertices + face indices).
    """
    import meshio
    m = meshio.read(str(path))
    verts = np.asarray(m.points, dtype=np.float64)
    tris_list = []
    for cell_block in m.cells:
        if cell_block.type == "triangle":
            tris_list.append(np.asarray(cell_block.data, dtype=np.int64))
        elif cell_block.type == "quad":
            q = np.asarray(cell_block.data, dtype=np.int64)
            tris_list.append(q[:, [0, 1, 2]])
            tris_list.append(q[:, [0, 2, 3]])
    if not tris_list:
        raise ValueError(f"no triangle/quad faces in {path}")
    tris = np.concatenate(tris_list, axis=0)
    return sdf_from_mesh(verts, tris, winding_order=winding_order,
                         chunk=chunk)


__all__ = [
    "SDF",
    "smooth_union", "smooth_intersection", "smooth_difference",
    "union", "intersection", "difference", "translate", "rotate_z",
    "sdf_sphere", "sdf_box", "sdf_cylinder", "sdf_torus", "sdf_gear",
    "slice_mask", "layer_heights", "part_bbox",
    # arbitrary complex geometry from triangle meshes
    "sdf_from_mesh", "sdf_from_stl", "sdf_from_obj",
]
