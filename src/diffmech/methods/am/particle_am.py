"""Particle-method AM (DEM / MPM / SPH): arbitrary geometry + custom laser paths.

Extends the AM module to **particle-based** simulation methods. The same two
extensibility entry points that the FEM-based AM module exposes —

1. *arbitrary complex geometry* from a triangle mesh (3D) or polygon (2D) via
   an SDF, and
2. *user-specified laser paths* (waypoints / G-code / CSV) —

are now available for the Discrete Element Method, the Material Point Method,
and Smoothed Particle Hydrodynamics.

Core idea
---------
Particles are pre-placed throughout the entire part volume (sampled from the
SDF, organised in build layers). Each particle carries:

- an **activation time** ``t_act`` — when the laser deposits it,
- a **temperature** ``T`` — heated by the Gaussian laser spot,
- a smooth **activation field** ``α_p(t) = σ((t − t_act) / τ) ∈ (0, 1)``.

Before ``t_act`` the particle is "powder" (zero stiffness / mass contribution,
no heat). After ``t_act`` it becomes "solid" with full properties and is heated
by the moving laser. Because ``α`` is a smooth sigmoid, ``jax.grad`` propagates
from process parameters (laser power, scan speed, …) through the entire build
history to the final particle state.

Dimensionality
--------------
Works in **2D and 3D**. Dimensionality is implicit in the particle coordinate
arrays (``(N, 2)`` → 2D, ``(N, 3)`` → 3D). In 2D the build direction is ``z``
(the second coordinate), the scan direction is ``x``, and the SDF takes
``(N, 2)`` input. In 3D the build direction is ``z`` (the third coordinate),
the scan plane is ``(x, y)``, and the SDF takes ``(N, 3)`` input.

The laser position is always derived from a :class:`MultiLayerPath` (whose
waypoints are ``(n, 2)`` = ``(x, y)``). In 2D only the ``x`` component is used;
the ``y`` component is ignored and the build-direction coordinate comes from
the layer's ``layer_z``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from diffmech.methods.am.scan_paths import MultiLayerPath
from diffmech.methods.am.process import SLMConfig, LSFConfig


SDF2D = Callable[[jnp.ndarray], jnp.ndarray]
SDF3D = Callable[[jnp.ndarray], jnp.ndarray]


# ===========================================================================
# 2D SDF primitives (for 2D particle AM)
# ===========================================================================
def sdf_circle_2d(center, radius: float) -> SDF2D:
    """SDF of a 2D circle."""
    c = jnp.asarray(center, dtype=jnp.float64)
    r = float(radius)

    def f(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.linalg.norm(x - c, axis=-1) - r
    return f


def sdf_box_2d(center, half_extents) -> SDF2D:
    """SDF of a 2D axis-aligned box."""
    c = jnp.asarray(center, dtype=jnp.float64)
    he = jnp.asarray(half_extents, dtype=jnp.float64)

    def f(x: jnp.ndarray) -> jnp.ndarray:
        d = jnp.abs(x - c) - he
        outside = jnp.linalg.norm(jnp.maximum(d, 0.0), axis=-1)
        inside = jnp.minimum(jnp.max(d, axis=-1), 0.0)
        return outside + inside
    return f


def sdf_from_polygon_2d(vertices) -> SDF2D:
    """SDF of a closed 2D polygon (arbitrary 2D geometry).

    Parameters
    ----------
    vertices : (n, 2) array of polygon vertices in counter-clockwise order.
    """
    v = jnp.asarray(vertices, dtype=jnp.float64)
    n = v.shape[0]
    # Pre-compute edge vectors.
    v_next = jnp.roll(v, -1, axis=0)
    edges = v_next - v  # (n, 2)

    def f(x: jnp.ndarray) -> jnp.ndarray:
        scalar_input = x.ndim == 1
        x = jnp.atleast_2d(x)
        # Distance from each point to each edge segment.
        def edge_dist(vi, ei, vj):
            # Point-to-segment distance (vectorised over query points).
            w = x - vi  # (N, 2)
            t = jnp.clip((w * ei).sum(-1) / jnp.maximum((ei * ei).sum(), 1e-30),
                         0.0, 1.0)
            proj = vi + t[:, None] * ei
            d2 = jnp.sum((x - proj) ** 2, axis=-1)  # (N,)
            return d2

        d2_all = jax.vmap(edge_dist)(v, edges, v_next)  # (n_edges, N)
        d_min = jnp.sqrt(jnp.min(d2_all, axis=0))  # (N,)

        # Point-in-polygon test (crossing number / ray casting).
        def crossing_count(vi, vj):
            yi, xi_ = vi[1], vi[0]
            yj, xj = vj[1], vj[0]
            yp = x[:, 1]
            xp = x[:, 0]
            cond = ((yi <= yp) & (yj > yp)) | ((yj <= yp) & (yi > yp))
            dy = yj - yi
            # Safe denominator preserving sign (avoid maximum which clips negatives).
            denom = jnp.where(jnp.abs(dy) > 1e-30, dy, 1.0)
            x_int = xi_ + (yp - yi) / denom * (xj - xi_)
            return jnp.where(cond & (xp < x_int), 1, 0)
        crossings = jnp.sum(jax.vmap(crossing_count)(v, v_next), axis=0)
        inside = (crossings % 2) == 1
        sign = jnp.where(inside, -1.0, 1.0)
        result = sign * d_min
        return result[0] if scalar_input else result

    return f


# ===========================================================================
# 3D STL parsing + triangle-mesh SDF (unsigned nearest-triangle distance +
# pseudo sign from nearest-face normal orientation)  —  pure NumPy/struct,
# no trimesh dependency required.
# ===========================================================================

def parse_binary_stl(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Parse a binary STL blob into (vertices, face_normals).

    Parameters
    ----------
    data : bytes
        Raw content of a binary STL file (80-byte header + uint32 n_faces +
        50 bytes per face: 3xf32 normal + 9xf32 vertices + uint16 attr).

    Returns
    -------
    verts : np.ndarray (F, 3, 3) float32 — triangle corner positions.
    normals : np.ndarray (F, 3) float32 — per-face outward normals.
    """
    import struct
    if len(data) < 84:
        raise ValueError("STL payload too small (< 84 bytes)")
    header = data[:80]
    n_faces = struct.unpack_from("<I", data, 80)[0]
    expected = 84 + 50 * n_faces
    if len(data) < expected:
        # ASCII STL fallback path: attempt to parse as ASCII if binary header
        # looks wrong.  We try a simple regex match for ASCII facet records.
        try:
            text = data.decode("utf-8", errors="strict").strip()
            if text.lower().startswith("solid"):
                return _parse_ascii_stl(text)
        except (UnicodeDecodeError, ValueError):
            pass
        raise ValueError(
            f"STL truncated: expected ≥ {expected} bytes, got {len(data)} "
            f"(claimed n_faces={n_faces})"
        )
    verts = np.empty((n_faces, 3, 3), dtype=np.float32)
    normals = np.empty((n_faces, 3), dtype=np.float32)
    off = 84
    for f in range(n_faces):
        chunk = data[off:off + 50]
        if len(chunk) < 50:
            raise ValueError("unexpected EOF inside STL face record")
        values = struct.unpack("<12fH", chunk)
        n = np.asarray(values[0:3], dtype=np.float32)
        v1 = values[3:6]; v2 = values[6:9]; v3 = values[9:12]
        verts[f, 0] = v1
        verts[f, 1] = v2
        verts[f, 2] = v3
        # Fallback to face normal if the STL-recorded normal is zero.
        if np.linalg.norm(n) < 1e-12:
            e1 = np.asarray(v2, dtype=np.float32) - np.asarray(v1, dtype=np.float32)
            e2 = np.asarray(v3, dtype=np.float32) - np.asarray(v1, dtype=np.float32)
            n = np.cross(e1, e2).astype(np.float32)
            nrm = np.linalg.norm(n)
            if nrm > 1e-12: n = n / nrm
        normals[f] = n
        off += 50
    return verts, normals


def _parse_ascii_stl(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Very small ASCII-STL parser — used as a fallback.  Returns (verts, normals)."""
    import re
    v_re = re.compile(r"^\s*vertex\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)")
    n_re = re.compile(r"^\s*facet\s+normal\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)")
    verts_list = []
    norms_list = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = n_re.match(lines[i])
        if m:
            n = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
            # Expect outer loop + 3 vertex lines
            tri = []
            i += 1
            while i < len(lines) and len(tri) < 3:
                mv = v_re.match(lines[i])
                if mv:
                    tri.append((float(mv.group(1)), float(mv.group(2)), float(mv.group(3))))
                i += 1
            if len(tri) == 3:
                verts_list.append(tri)
                norms_list.append(n)
        else:
            i += 1
    if not verts_list:
        raise ValueError("ASCII STL contained no facets")
    return np.asarray(verts_list, dtype=np.float32), np.asarray(norms_list, dtype=np.float32)


def _signed_distance_triangles(
    query: np.ndarray,
    verts: np.ndarray,
    normals: np.ndarray,
) -> np.ndarray:
    """CPU (NumPy) signed-distance from query points to a triangle mesh.

    Uses the geometric closest-point projection on each triangle (O(Q·F)).
    For reasonable queries (F ≤ 50k, Q ≤ 200k) this is acceptable for an
    offline geometry-preprocessing pass; it keeps us free of external
    dependencies.

    Returns ``(Q,)`` float32 signed-distance array.
    """
    Q = len(query)
    F = len(verts)
    out = np.full(Q, np.inf, dtype=np.float32)
    sign = np.ones(Q, dtype=np.float32)
    # For each query, walk the triangles.  Vectorise per-triangle over all Q.
    v0 = verts[:, 0]  # (F, 3)
    v1 = verts[:, 1]
    v2 = verts[:, 2]
    for f in range(F):
        a = v0[f]  # (3,)
        b = v1[f]
        c = v2[f]
        nf = normals[f]
        # Vectors for closest-point: standard triangle projection (Ericson, 2005).
        ab = b - a
        ac = c - a
        ap = query - a  # (Q, 3)
        # Barycentric coordinates.
        d1 = np.sum(ab * ap, axis=1)  # (Q,)
        d2 = np.sum(ac * ap, axis=1)
        # Check if P projects in vertex-region A.
        if f == 0:
            closest = np.empty((Q, 3), dtype=np.float32)
        # Compute the closest point on the triangle.  We do this via branching
        # per Voronoi region; vectorised by mask accumulation.
        # Region flags (per query):
        # * R0: d1 <= 0 & d2 <= 0 → a
        # * R1: d1 >= |ab|² & d2 <= d1 → b
        # * R2: d2 >= |ac|² & d1 <= d2 → c
        # * Edge ab
        # * Edge ac
        # * Edge bc
        # * Interior
        d1a = d1  # ap·ab
        d2a = d2  # ap·ac
        a_dot_a = 0.0
        abab = float(np.dot(ab, ab)) + 1e-20
        acac = float(np.dot(ac, ac)) + 1e-20
        abac = float(np.dot(ab, ac))
        # Compute barycentric (u, v) = closest point on parallelogram, then
        # clamp to the triangle using standard divide-by-determinant.
        denom = abab * acac - abac * abac + 1e-20
        vc = (acac * d1a - abac * d2a) / denom
        vd = (abab * d2a - abac * d1a) / denom
        # Clamp using ternary masks for triangle boundaries.
        s_mask = (vc >= 0) & (vd >= 0) & (vc + vd <= 1)
        uc = np.clip(vc, 0.0, 1.0)
        ud = np.clip(vd, 0.0, 1.0)
        # If outside the triangle interior, the clamped point lies on the
        # boundary; we do a simple 2-segment clamp fallback for s_mask==False
        # by projecting to 3 edges and taking the minimum distance.
        # Edge ab: t ∈ [0,1] closest
        ab_norm = abab
        tab = np.clip(np.sum(ab * ap, axis=1) / (ab_norm + 1e-20), 0.0, 1.0)
        cp_ab = a + ab[None, :] * tab[:, None]
        d_ab = np.sum((query - cp_ab) ** 2, axis=1)
        # Edge ac
        ac_norm = acac
        tac = np.clip(np.sum(ac * ap, axis=1) / (ac_norm + 1e-20), 0.0, 1.0)
        cp_ac = a + ac[None, :] * tac[:, None]
        d_ac = np.sum((query - cp_ac) ** 2, axis=1)
        # Edge bc
        edge_bc = c - b
        bp = query - b
        bc_norm = float(np.dot(edge_bc, edge_bc)) + 1e-20
        tbc = np.clip(np.sum(edge_bc * bp, axis=1) / bc_norm, 0.0, 1.0)
        cp_bc = b + edge_bc[None, :] * tbc[:, None]
        d_bc = np.sum((query - cp_bc) ** 2, axis=1)
        # Interior projection: a + ab·vc + ac·vd
        cp_in = a[None, :] + ab[None, :] * vc[:, None] + ac[None, :] * vd[:, None]
        d_in = np.sum((query - cp_in) ** 2, axis=1)
        # Choose per-query: interior if s_mask else min(edge distances)
        d_edge = np.minimum(np.minimum(d_ab, d_ac), d_bc)
        best_sq = np.where(s_mask, d_in, d_edge)
        cp_best = np.where(s_mask[:, None], cp_in,
                           np.where(
                               (d_ab <= d_ac)[:, None] & (d_ab <= d_bc)[:, None], cp_ab,
                               np.where((d_ac <= d_bc)[:, None], cp_ac, cp_bc),
                           ))
        is_closer = best_sq < out
        out = np.where(is_closer, best_sq, out)
        # Sign:  (p - cp) · n_face  — negative when behind the face outward
        # normal (assumes consistent outward orientation; inside → negative,
        # flip at end so inside is positive-φ inside SDF convention we want
        # (negative inside, positive outside / or the other way — we take
        # SDF signed as "negative inside the mesh" which is standard for
        # most solid CSG libraries.  We use dot(p − cp, nf) < 0 → inside → negative sign).
        outward = np.sum((query - cp_best) * nf[None, :], axis=1)
        # update sign only when we actually take this face as closest
        sign = np.where(is_closer, np.where(outward < 0, -1.0, 1.0), sign)
    out = np.sqrt(np.maximum(out, 0.0))
    # Final: negative inside, positive outside.
    return (out * sign).astype(np.float32)


def sdf_from_stl_mesh_3d(
    verts_or_path,
    normals: np.ndarray | None = None,
    *,
    center: bool = True,
    scale_to: tuple[float, float, float] | None = (0.2, 0.2, 0.08),
):
    """Build a 3D SDF ``(Q,3) → (Q,)`` from either a file path or triangles.

    Parameters
    ----------
    verts_or_path : bytes | str | np.ndarray
        - ``bytes`` — binary/ASCII STL file contents,
        - ``str`` — filesystem path to ``.stl``,
        - ``np.ndarray`` ``(F, 3, 3)`` — triangle corner positions.
    normals : np.ndarray (F, 3) | None
        Per-face normals (required only if passing raw triangle positions).
    center : bool
        Translate the mesh so its bounding box is centered at the origin.
    scale_to : (Lx, Ly, Lz) | None
        Uniformly scale the mesh so its longest side matches the maximum of
        the given extents, then clip to fit in a box of size ``scale_to``.
        ``None`` keeps the original STL units (typically mm → m; we do
        multiply by 1e-3 so mm STLs convert to metres).

    Returns
    -------
    f : callable    (Q, 3) → (Q,)  float32
    mesh_info : dict with keys ``verts``, ``normals``, ``bbox``.
    """
    # --- Load / parse ---
    if isinstance(verts_or_path, (bytes, bytearray, memoryview)):
        verts, normals = parse_binary_stl(bytes(verts_or_path))
    elif isinstance(verts_or_path, str):
        with open(verts_or_path, "rb") as fh:
            verts, normals = parse_binary_stl(fh.read())
    else:
        verts = np.asarray(verts_or_path, dtype=np.float32)
        if verts.ndim != 3 or verts.shape[1:] != (3, 3):
            raise ValueError("triangle vertices must have shape (F, 3, 3)")
        if normals is None:
            # Compute per-face normals via cross product
            e1 = verts[:, 1] - verts[:, 0]
            e2 = verts[:, 2] - verts[:, 0]
            n = np.cross(e1, e2).astype(np.float32)
            nrm = np.linalg.norm(n, axis=1, keepdims=True)
            normals = n / np.where(nrm > 1e-12, nrm, 1.0)
    F = len(verts)
    if F == 0:
        raise ValueError("empty mesh — no triangles")
    # --- Centre and scale ---
    bbox_min = verts.reshape(-1, 3).min(axis=0)
    bbox_max = verts.reshape(-1, 3).max(axis=0)
    if center:
        offset = 0.5 * (bbox_min + bbox_max)
        verts = verts - offset.astype(np.float32)
        bbox_min = bbox_min - offset
        bbox_max = bbox_max - offset
    if scale_to is not None:
        extents = bbox_max - bbox_min + 1e-12
        target_ext = np.asarray(scale_to, dtype=np.float32)
        # Uniform scale: fit the largest STL axis to the largest target axis.
        scale_factor = float(np.min(target_ext / extents))
        verts = (verts * scale_factor).astype(np.float32)
        bbox_min = bbox_min * scale_factor
        bbox_max = bbox_max * scale_factor
    else:
        # Default STL is in mm — convert to SI metres.
        verts = (verts * 1e-3).astype(np.float32)
        bbox_min = bbox_min * 1e-3
        bbox_max = bbox_max * 1e-3
    bbox = np.stack([bbox_min, bbox_max], axis=0)

    # Build a cached SDF function
    def f(points):
        scalar_input = False
        p = np.asarray(points, dtype=np.float32)
        if p.ndim == 1 and p.shape[0] == 3:
            p = p.reshape(1, 3)
            scalar_input = True
        if p.ndim != 2 or p.shape[1] != 3:
            raise ValueError(f"STL SDF expects (Q,3) input, got {p.shape}")
        d = _signed_distance_triangles(p, verts, normals)
        # Inside negative: particle placement wants negative = inside, positive = outside.
        return d[0] if scalar_input else d
    mesh_info = {
        "verts": verts, "normals": normals,
        "bbox": [[float(bbox_min[0]), float(bbox_max[0])],
                 [float(bbox_min[1]), float(bbox_max[1])],
                 [float(bbox_min[2]), float(bbox_max[2])]],
        "n_faces": int(F),
    }
    return f, mesh_info


# ===========================================================================
# Particle AM state
# ===========================================================================
@dataclass(frozen=True)
class ParticleAMState:
    """Particle state for AM simulation.

    Attributes
    ----------
    position : (N, dim) array
        Particle positions. ``dim`` is 2 or 3.
    activation_time : (N,) array
        Physical time at which each particle begins depositing [s].
    temperature : (N,) array
        Per-particle temperature [K].
    mass : (N,) array
        Full (post-activation) particle mass [kg].
    layer_id : (N,) int array
        Build-layer index of each particle.
    rho : (N,) array
        Material density [kg/m³] of each particle. Required to convert the
        volumetric laser heat source ``q`` [W/m³] into a per-particle
        temperature rise ``ΔT = q·Δt / (ρ·c_p)`` with correct units.
    """

    position: jax.Array
    activation_time: jax.Array
    temperature: jax.Array
    mass: jax.Array
    layer_id: jax.Array
    rho: jax.Array

    @property
    def n_particles(self) -> int:
        return int(self.position.shape[0])

    @property
    def dim(self) -> int:
        return int(self.position.shape[1])

    def activation_field(self, t: jnp.ndarray, tau: float = 1e-3) -> jax.Array:
        """Smooth per-particle activation ``α_p(t) ∈ (0, 1)``."""
        return jax.nn.sigmoid((t - self.activation_time) / tau)


# Register as pytree for jax.grad / lax.scan.
def _pam_flatten(s: ParticleAMState):
    return (s.position, s.activation_time, s.temperature, s.mass,
            s.layer_id, s.rho), None


def _pam_unflatten(_, children):
    pos, at, temp, mass, lid, rho = children
    return ParticleAMState(pos, at, temp, mass, lid, rho)


jax.tree_util.register_pytree_node(ParticleAMState,
                                   _pam_flatten, _pam_unflatten)


# ===========================================================================
# Particle placement from an SDF
# ===========================================================================
def place_particles_in_sdf(
    sdf,
    bbox,
    layer_zs,
    *,
    dim: int = 3,
    spacing: float = 0.05,
    margin: float = 0.0,
):
    """Place particles filling the part SDF, organised in build layers.

    Parameters
    ----------
    sdf : signed-distance function (2D or 3D).
    bbox : bounding box ``[(xmin,xmax),(ymin,ymax)]`` (2D) or
        ``[(xmin,xmax),(ymin,ymax),(zmin,zmax)]`` (3D).
    layer_zs : (n_layers,) array of layer mid-plane build-direction heights.
    dim : 2 or 3.
    spacing : nominal particle spacing [m].
    margin : extra padding inside the bbox (particles within ``margin`` of the
        bbox edge are excluded).

    Returns
    -------
    positions : (N, dim) array of particle coordinates.
    layer_id : (N,) int array of layer index per particle.
    """
    layer_zs = np.asarray(layer_zs, dtype=np.float64)
    n_layers = len(layer_zs)

    if dim == 2:
        (xmin, xmax), (zmin, zmax) = bbox
        xs = np.arange(xmin + margin, xmax - margin + spacing * 0.5, spacing)
        zs = np.arange(zmin + margin, zmax - margin + spacing * 0.5, spacing)
        positions = []
        layer_ids = []
        for k, z_layer in enumerate(layer_zs):
            # For each layer, place particles at z ≈ layer_z (thin band).
            z_band = [z_layer]  # single row of particles at layer height
            for z in z_band:
                for x in xs:
                    positions.append([x, z])
                    layer_ids.append(k)
        positions = np.asarray(positions, dtype=np.float64)
        layer_ids = np.asarray(layer_ids, dtype=np.int64)
        if len(positions) == 0:
            return positions, layer_ids
        # Filter by SDF (keep only inside the part).
        s = np.asarray(sdf(jnp.asarray(positions)))
        inside = s < 0.0
        return positions[inside], layer_ids[inside]

    # dim == 3
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bbox
    xs = np.arange(xmin + margin, xmax - margin + spacing * 0.5, spacing)
    ys = np.arange(ymin + margin, ymax - margin + spacing * 0.5, spacing)
    positions = []
    layer_ids = []
    for k, z_layer in enumerate(layer_zs):
        for x in xs:
            for y in ys:
                positions.append([x, y, z_layer])
                layer_ids.append(k)
    positions = np.asarray(positions, dtype=np.float64)
    layer_ids = np.asarray(layer_ids, dtype=np.int64)
    if len(positions) == 0:
        return positions, layer_ids
    s = np.asarray(sdf(jnp.asarray(positions)))
    inside = s < 0.0
    return positions[inside], layer_ids[inside]


# ===========================================================================
# Per-particle activation times from scan paths
# ===========================================================================
def assign_particle_activation_times(
    layer_ids: np.ndarray,
    layer_start_times: np.ndarray,
) -> np.ndarray:
    """Per-particle activation time = its layer's start time.

    Parameters
    ----------
    layer_ids : (N,) int array of layer index per particle.
    layer_start_times : (n_layers,) array of layer start times [s].  JAX traced
        values (from gradient-mode calls) are allowed.
    """
    layer_ids = jnp.asarray(layer_ids, dtype=jnp.int64)
    starts = jnp.asarray(layer_start_times, dtype=jnp.float64)
    return starts[layer_ids]


def particle_layer_times(paths: MultiLayerPath, cfg):
    """Compute per-layer start times and durations from paths + scan speed."""
    lengths = particle_layer_path_lengths(paths)
    durations = lengths / cfg.scan_speed
    starts = jnp.concatenate([jnp.zeros(1), jnp.cumsum(durations)[:-1]])
    return starts, durations


def particle_layer_path_lengths(paths: MultiLayerPath) -> jax.Array:
    """Total scan arc-length of each layer."""
    return jnp.asarray([p.total_length for p in paths.layers], dtype=jnp.float64)


# ===========================================================================
# Laser position + heat source on particles
# ===========================================================================
def _laser_xy_at(paths: MultiLayerPath, starts, durations, t):
    """Differentiable laser (x, y) position + active layer index at time ``t``."""
    n = len(paths.layers)
    max_wp = max(len(p.waypoints) for p in paths.layers)
    wp = np.zeros((n, max_wp, 2))
    cum = np.zeros((n, max_wp))
    lengths = np.zeros(n)
    layer_zs = np.zeros(n)
    for i, p in enumerate(paths.layers):
        m = len(p.waypoints)
        wp[i, :m] = p.waypoints
        cum[i, :m] = p.cum_length
        lengths[i] = p.total_length
        layer_zs[i] = p.layer_z
    wp = jnp.asarray(wp)
    cum = jnp.asarray(cum)
    lengths = jnp.asarray(lengths)
    layer_zs = jnp.asarray(layer_zs)
    starts = jnp.asarray(starts, dtype=jnp.float64)
    durations = jnp.asarray(durations, dtype=jnp.float64)

    active = jnp.clip(jnp.sum(t >= starts) - 1, 0, n - 1)
    t_local = jnp.clip(t - starts[active], 0.0, durations[active])
    frac = jnp.where(durations[active] > 1e-12,
                     t_local / jnp.maximum(durations[active], 1e-12), 0.0)
    s = jnp.clip(frac * lengths[active], 0.0, lengths[active])

    cum_layer = cum[active]
    idx = jnp.clip(jnp.sum(cum_layer <= s) - 1, 0, max_wp - 2)
    s0 = cum_layer[idx]
    s1 = cum_layer[idx + 1]
    seg_len = s1 - s0
    t_frac = jnp.where(seg_len > 1e-12,
                       (s - s0) / jnp.maximum(seg_len, 1e-12), 0.0)
    p0 = wp[active, idx]
    p1 = wp[active, idx + 1]
    laser_xy = p0 + t_frac * (p1 - p0)
    z_active = layer_zs[active]
    return laser_xy, z_active, active


def laser_heat_on_particles(
    position: jnp.ndarray,
    t: float,
    paths: MultiLayerPath,
    layer_start_times: np.ndarray,
    layer_durations: np.ndarray,
    cfg,
    *,
    dim: int = 3,
) -> jnp.ndarray:
    """Volumetric heat deposition ``Q_p`` [W/m³] on each particle at time ``t``.

    A Gaussian spot centred at the laser's instantaneous position. In 2D the
    distance is computed in the ``(x, z)`` plane (using the path's ``x``
    coordinate and the layer's ``z``). In 3D the full ``(x, y, z)`` distance
    is used.
    """
    laser_xy, z_active, _ = _laser_xy_at(paths, layer_start_times,
                                         layer_durations, t)
    absorbed = cfg.absorption * cfg.laser_power
    r2 = cfg.beam_radius ** 2
    pi_r2 = jnp.pi * r2

    if dim == 2:
        # Particles are (x, z); laser x from path, z from layer.
        dx = position[:, 0] - laser_xy[0]
        dz = position[:, 1] - z_active
        rsq = dx * dx + dz * dz
    else:
        dx = position[:, 0] - laser_xy[0]
        dy = position[:, 1] - laser_xy[1]
        dz = position[:, 2] - z_active
        rsq = dx * dx + dy * dy + dz * dz

    q = (2.0 * absorbed) / pi_r2 * jnp.exp(-2.0 * rsq / r2)
    # Depth attenuation around the active layer.
    depth = 2.0 * cfg.layer_thickness
    if dim == 2:
        q = q * jnp.exp(-((position[:, 1] - z_active) ** 2) /
                       (2.0 * depth * depth))
    else:
        q = q * jnp.exp(-((position[:, 2] - z_active) ** 2) /
                        (2.0 * depth * depth))
    return q


# ===========================================================================
# Setup: build a complete particle AM problem
# ===========================================================================
@dataclass(frozen=True)
class ParticleAMProblem:
    """Bundle of everything needed to run a particle AM simulation."""

    am_state: ParticleAMState
    paths: MultiLayerPath
    cfg: object
    layer_start_times: np.ndarray
    layer_durations: np.ndarray
    dim: int


def _pamp_flatten(p: ParticleAMProblem):
    # am_state, cfg, layer_start_times, layer_durations are leaves.
    leaves = (p.am_state, p.cfg,
              jnp.asarray(p.layer_start_times, dtype=jnp.float64),
              jnp.asarray(p.layer_durations, dtype=jnp.float64))
    aux = (p.paths, p.dim)
    return leaves, aux


def _pamp_unflatten(aux, leaves):
    am_state, cfg, starts, durs = leaves
    paths, dim = aux
    return ParticleAMProblem(
        am_state=am_state, paths=paths, cfg=cfg,
        layer_start_times=starts,
        layer_durations=durs,
        dim=dim,
    )


jax.tree_util.register_pytree_node(ParticleAMProblem,
                                   _pamp_flatten, _pamp_unflatten)


def setup_particle_am(
    positions: np.ndarray,
    layer_ids: np.ndarray,
    paths: MultiLayerPath,
    cfg,
    *,
    dim: int = 3,
    particle_mass: float = 1.0,
    density: float = 7850.0,
    preheat_temp: float | None = None,
    tau: float = 1e-3,
) -> ParticleAMProblem:
    """Assemble a particle AM problem from positions + paths + config.

    Computes per-particle activation times from the scan-path lengths and
    scan speed, initialises the temperature to the preheat temperature.
    """
    starts, durations = particle_layer_times(paths, cfg)
    act_times = assign_particle_activation_times(layer_ids, starts)
    n = len(positions)
    T0 = cfg.preheat_temp if preheat_temp is None else preheat_temp
    am_state = ParticleAMState(
        position=jnp.asarray(positions, dtype=jnp.float64),
        activation_time=jnp.asarray(act_times, dtype=jnp.float64),
        temperature=jnp.full(n, T0, dtype=jnp.float64),
        mass=jnp.full(n, float(particle_mass), dtype=jnp.float64),
        layer_id=jnp.asarray(layer_ids, dtype=jnp.int64),
        rho=jnp.full(n, float(density), dtype=jnp.float64),
    )
    return ParticleAMProblem(
        am_state=am_state,
        paths=paths,
        cfg=cfg,
        layer_start_times=starts,
        layer_durations=durations,
        dim=dim,
    )


# ===========================================================================
# AM-aware DEM step
# ===========================================================================
def step_dem_am(
    dem_state,
    am_state: ParticleAMState,
    dt: float,
    t: float,
    dem_cfg,
    cfg,
    paths: MultiLayerPath,
    layer_start_times,
    layer_durations,
    *,
    walls=None,
    cp: float = 500.0,
    alpha_T: float = 1e-5,
    tau: float = 1e-3,
):
    """One AM-aware DEM step.

    - Laser heats nearby activated particles (temperature update).
    - Activation field gates the contact stiffness (powder = soft).
    - Thermal expansion increases particle radius with temperature.
    - Standard DEM contact + wall + gravity forces advance positions.
    """
    from diffmech.methods.dem import step_dem, DEMConfig

    alpha = am_state.activation_field(t, tau)  # (N,)

    # --- laser heat deposition ---
    q = laser_heat_on_particles(am_state.position, t, paths,
                                layer_start_times, layer_durations, cfg,
                                dim=am_state.dim)
    # Only activated particles absorb heat.
    q_eff = q * alpha
    # q_eff is volumetric [W/m³]; convert to ΔT via material density ρ:
    #   ΔT = q_eff · dt / (ρ · c_p)   → units [W/m³·s / (kg/m³·J/(kg·K))] = K.
    dT = q_eff * dt / (am_state.rho * cp + 1e-30)
    T_new = am_state.temperature + dT

    # --- thermal expansion: effective radius ---
    T_ref = cfg.preheat_temp
    dT_thermal = T_new - T_ref
    # Scale radius by (1 + alpha_T * dT).
    radius_eff = dem_state.radius * (1.0 + alpha_T * jnp.maximum(dT_thermal, 0.0))

    # --- activation-gated contact stiffness ---
    # Build an effective DEM config per-step with scaled stiffness.
    # Since DEMConfig is frozen, we modify the state's effective stiffness
    # by scaling the contact force post-hoc. Instead, we use a simpler
    # approach: scale particle radii by sqrt(alpha) to weaken contact
    # for unactivated particles (geometric stiffness reduction).
    radius_contact = radius_eff * jnp.sqrt(alpha + 1e-10)
    dem_eff = type(dem_state)(
        position=dem_state.position,
        velocity=dem_state.velocity,
        radius=radius_contact,
        mass=dem_state.mass,
    )
    dem_new = step_dem(dem_eff, dt, dem_cfg, walls=walls)

    # Restore the physical radius (thermal-expanded) in the returned state.
    dem_new = type(dem_state)(
        position=dem_new.position,
        velocity=dem_new.velocity,
        radius=radius_eff,
        mass=dem_new.mass,
    )
    am_new = ParticleAMState(
        position=dem_new.position,
        activation_time=am_state.activation_time,
        temperature=T_new,
        mass=am_state.mass,
        layer_id=am_state.layer_id,
        rho=am_state.rho,
    )
    return dem_new, am_new


def step_dem_am_scan(
    dem_state0,
    am_state0: ParticleAMState,
    dt: float,
    n_steps: int,
    dem_cfg,
    cfg,
    paths: MultiLayerPath,
    layer_start_times,
    layer_durations,
    *,
    walls=None,
    cp: float = 500.0,
    alpha_T: float = 1e-5,
    tau: float = 1e-3,
):
    """Time-step DEM-AM for ``n_steps`` via ``jax.lax.scan`` (differentiable)."""
    starts = jnp.asarray(layer_start_times, dtype=jnp.float64)
    durations = jnp.asarray(layer_durations, dtype=jnp.float64)

    def body(carry, i):
        dem_s, am_s = carry
        t = i * dt
        dem_s, am_s = step_dem_am(
            dem_s, am_s, dt, t, dem_cfg, cfg, paths,
            starts, durations,
            walls=walls, cp=cp, alpha_T=alpha_T, tau=tau,
        )
        return (dem_s, am_s), None

    (dem_final, am_final), _ = jax.lax.scan(
        body, (dem_state0, am_state0), xs=jnp.arange(n_steps))
    return dem_final, am_final


# ===========================================================================
# AM-aware MPM step
# ===========================================================================
def step_mpm_am(
    mpm_state,
    am_state: ParticleAMState,
    dt: float,
    t: float,
    mpm_cfg,
    cfg,
    paths: MultiLayerPath,
    layer_start_times,
    layer_durations,
    *,
    bc="slip",
    cp: float = 500.0,
    alpha_T: float = 1e-5,
    tau: float = 1e-3,
):
    """One AM-aware MPM step.

    - Laser heats nearby activated particles (temperature update).
    - Activation field gates the Cauchy stress via deformation-gradient
      blending: ``F_eff = I + α · (F_thermal⁻¹ · F − I)`` — when ``α→0``
      the particle has identity F (zero stress = powder); when ``α→1``
      it has the full mechanical + thermal deformation.
    - Standard MPM P2G → grid solve → G2P advances positions.
    """
    from diffmech.methods.mpm.mpm import step_mpm

    alpha = am_state.activation_field(t, tau)  # (N,)

    # --- laser heat deposition ---
    q = laser_heat_on_particles(am_state.position, t, paths,
                                layer_start_times, layer_durations, cfg,
                                dim=am_state.dim)
    q_eff = q * alpha
    dT = q_eff * dt / (am_state.rho * cp + 1e-30)
    T_new = am_state.temperature + dT

    # --- activation-gated deformation gradient ---
    # Thermal expansion: F_thermal = (1 + α_T · ΔT) · I  (isotropic expansion)
    T_ref = cfg.preheat_temp
    dT_thermal = jnp.maximum(T_new - T_ref, 0.0)
    dim = mpm_state.dim
    I = jnp.eye(dim)
    # Factor by which to divide out the thermal expansion from F.
    thermal_factor = (1.0 + alpha_T * dT_thermal)[:, None, None]
    F_mech = mpm_state.deformation / jnp.maximum(thermal_factor, 1e-10)
    # Blend: α=0 → F_eff = I (powder, no stress); α=1 → F_eff = F_mech (solid).
    F_eff = I + alpha[:, None, None] * (F_mech - I)

    # Build a temporary state with the gated deformation gradient.
    mpm_eff = type(mpm_state)(
        position=mpm_state.position,
        velocity=mpm_state.velocity,
        affine=mpm_state.affine,
        deformation=F_eff,
        volume0=mpm_state.volume0,
        mass=mpm_state.mass,
    )
    # Run the standard MPM step (P2G → grid solve → G2P) with the gated F.
    mpm_new = step_mpm(mpm_eff, mpm_cfg, bc=bc)

    am_new = ParticleAMState(
        position=mpm_new.position,
        activation_time=am_state.activation_time,
        temperature=T_new,
        mass=am_state.mass,
        layer_id=am_state.layer_id,
        rho=am_state.rho,
    )
    return mpm_new, am_new


def step_mpm_am_scan(
    mpm_state0,
    am_state0: ParticleAMState,
    dt: float,
    n_steps: int,
    mpm_cfg,
    cfg,
    paths: MultiLayerPath,
    layer_start_times,
    layer_durations,
    *,
    bc="slip",
    cp: float = 500.0,
    alpha_T: float = 1e-5,
    tau: float = 1e-3,
):
    """Time-step MPM-AM for ``n_steps`` via ``jax.lax.scan`` (differentiable)."""
    starts = jnp.asarray(layer_start_times, dtype=jnp.float64)
    durations = jnp.asarray(layer_durations, dtype=jnp.float64)

    def body(carry, i):
        mpm_s, am_s = carry
        t = i * dt
        mpm_s, am_s = step_mpm_am(
            mpm_s, am_s, dt, t, mpm_cfg, cfg, paths,
            starts, durations,
            bc=bc, cp=cp, alpha_T=alpha_T, tau=tau,
        )
        return (mpm_s, am_s), None

    (mpm_final, am_final), _ = jax.lax.scan(
        body, (mpm_state0, am_state0), xs=jnp.arange(n_steps))
    return mpm_final, am_final


# ===========================================================================
# AM-aware SPH step
# ===========================================================================
def step_sph_am(
    sph_state,
    am_state: ParticleAMState,
    dt: float,
    t: float,
    sph_cfg,
    cfg,
    paths: MultiLayerPath,
    layer_start_times,
    layer_durations,
    *,
    cp: float = 500.0,
    alpha_T: float = 1e-5,
    tau: float = 1e-3,
):
    """One AM-aware SPH step.

    - Laser heats nearby activated particles (temperature update).
    - Activation field gates the mass contribution (powder = inactive).
    - Thermal expansion reduces effective density (hot = less dense).
    - Standard SPH pressure + viscosity forces advance positions.
    """
    from diffmech.methods.sph import (
        compute_density, pressure_from_density, cubic_kernel,
        cubic_kernel_grad, SPHState,
    )

    alpha = am_state.activation_field(t, tau)  # (N,)

    # --- laser heat deposition ---
    q = laser_heat_on_particles(am_state.position, t, paths,
                                layer_start_times, layer_durations, cfg,
                                dim=am_state.dim)
    q_eff = q * alpha
    dT = q_eff * dt / (am_state.rho * cp + 1e-30)
    T_new = am_state.temperature + dT

    # --- effective mass (activation-gated) ---
    mass_eff = sph_state.mass * alpha

    # --- density with gated mass ---
    dim = sph_state.dim
    h = sph_cfg.h

    def density_i(pos_i):
        def contrib(pos_j, m_j):
            r = jnp.sqrt(jnp.sum((pos_i - pos_j) ** 2) + 1e-16)
            return m_j * cubic_kernel(r, h, dim)
        return jnp.sum(jax.vmap(contrib)(sph_state.position, mass_eff),
                       axis=-1)
    rho = jax.vmap(density_i)(sph_state.position)

    # --- thermal expansion: reduce density for hot particles ---
    T_ref = cfg.preheat_temp
    dT_thermal = T_new - T_ref
    rho_thermal = rho / (1.0 + alpha_T * jnp.maximum(dT_thermal, 0.0))

    # --- pressure from thermally-modified density ---
    p = pressure_from_density(rho_thermal, sph_cfg)

    # --- acceleration (pressure + viscosity + gravity) ---
    def accel_i(pos_i, vel_i, rho_i, p_i):
        def pair_force(pos_j, vel_j, m_j, rho_j, p_j):
            rij = pos_i - pos_j
            r = jnp.sqrt(jnp.sum(rij ** 2) + 1e-16)
            grad_W = cubic_kernel_grad(rij, h, dim)
            eps = 1e-30
            f_pressure = -m_j * (p_i / (rho_i ** 2 + eps) +
                                 p_j / (rho_j ** 2 + eps)) * grad_W
            vij = vel_i - vel_j
            v_dot_r = jnp.dot(vij, rij)
            pi_ij = jnp.where(v_dot_r < 0.0,
                             -sph_cfg.nu * (2.0 * h * v_dot_r) /
                             (r ** 2 + 0.01 * h * h),
                             0.0)
            f_visc = -m_j * pi_ij * grad_W
            return f_pressure + f_visc
        forces = jax.vmap(pair_force)(
            sph_state.position, sph_state.velocity, mass_eff, rho, p)
        total_force = jnp.sum(forces, axis=0)
        return total_force

    accel = jax.vmap(accel_i)(sph_state.position, sph_state.velocity,
                              rho_thermal, p)
    # Add gravity along last axis.
    accel = accel.at[:, -1].add(sph_cfg.gravity)

    # --- semi-implicit Euler ---
    v_new = sph_state.velocity + dt * accel
    x_new = sph_state.position + dt * v_new
    sph_new = SPHState(position=x_new, velocity=v_new, mass=sph_state.mass)
    am_new = ParticleAMState(
        position=x_new,
        activation_time=am_state.activation_time,
        temperature=T_new,
        mass=am_state.mass,
        layer_id=am_state.layer_id,
        rho=am_state.rho,
    )
    return sph_new, am_new


def step_sph_am_scan(
    sph_state0,
    am_state0: ParticleAMState,
    dt: float,
    n_steps: int,
    sph_cfg,
    cfg,
    paths: MultiLayerPath,
    layer_start_times,
    layer_durations,
    *,
    cp: float = 500.0,
    alpha_T: float = 1e-5,
    tau: float = 1e-3,
):
    """Time-step SPH-AM for ``n_steps`` via ``jax.lax.scan`` (differentiable)."""
    starts = jnp.asarray(layer_start_times, dtype=jnp.float64)
    durations = jnp.asarray(layer_durations, dtype=jnp.float64)

    def body(carry, i):
        sph_s, am_s = carry
        t = i * dt
        sph_s, am_s = step_sph_am(
            sph_s, am_s, dt, t, sph_cfg, cfg, paths,
            starts, durations,
            cp=cp, alpha_T=alpha_T, tau=tau,
        )
        return (sph_s, am_s), None

    (sph_final, am_final), _ = jax.lax.scan(
        body, (sph_state0, am_state0), xs=jnp.arange(n_steps))
    return sph_final, am_final


# ===========================================================================
# Diagnostics
# ===========================================================================
def peak_temperature(am_state: ParticleAMState) -> jnp.ndarray:
    """Maximum particle temperature."""
    return jnp.max(am_state.temperature)


def activated_fraction(am_state: ParticleAMState, t: float,
                       tau: float = 1e-3) -> jnp.ndarray:
    """Fraction of particles with α > 0.5 at time ``t``."""
    alpha = am_state.activation_field(t, tau)
    return jnp.mean(alpha > 0.5)


def mean_displacement(initial_pos: jnp.ndarray,
                      am_state: ParticleAMState) -> jnp.ndarray:
    """Mean particle displacement from initial positions."""
    return jnp.mean(jnp.linalg.norm(am_state.position - initial_pos, axis=-1))


# ===========================================================================
# LSF / DED: continuous powder mass-source along the laser track
# ===========================================================================

def lsf_powder_deposition_profile(position: jnp.ndarray,
                                   t: float,
                                   paths: MultiLayerPath,
                                   layer_start_times,
                                   layer_durations,
                                   cfg,  # LSFConfig
                                   *,
                                   dim: int = 3,
                                   ) -> jnp.ndarray:
    """Spatio-temporal powder-deposition source per particle [kg/(s·m³) equivalent]

    The powder is co-axially injected into the melt pool; its effective
    spatial shape is approximated as a double Gaussian — a wider catchment
    cloud (≈ 1.5 × beam_radius) weighted by the local beam intensity so
    powder preferentially deposits right under the laser where the melt
    pool is.  The integrated mass rate over the domain is

        ṁ = powder_feed_rate * deposition_efficiency  [kg/s]

    Returns a per-particle *mass source rate* [kg/s].  Divide by particle
    volume to get a proper volumetric source, or simply add dm = s * dt
    directly to each particle's mass.
    """
    from src.diffmech.methods.am.process import LSFConfig
    # Guard against plain SLM configs — in that case the source is zero.
    if not isinstance(cfg, LSFConfig):
        return jnp.zeros(position.shape[0])

    laser_xy, z_active, active = _laser_xy_at(paths, layer_start_times,
                                               layer_durations, t)
    r2 = cfg.beam_radius ** 2
    # Total mass rate [kg/s]
    mdot_total = cfg.powder_feed_rate * cfg.deposition_efficiency
    # Normalization: ∫ 1/(π R²) exp(-2 r²/R²) over 2D = 1.
    pi_R2 = jnp.pi * r2
    if dim == 2:
        dx = position[:, 0] - laser_xy[0]
        dz = position[:, 1] - z_active
        rsq = dx * dx + dz * dz
    else:
        dx = position[:, 0] - laser_xy[0]
        dy = position[:, 1] - laser_xy[1]
        dz = position[:, 2] - z_active
        rsq = dx * dx + dy * dy + dz * dz
    # Catchment profile: wider ring radius around the beam
    ring_R2 = (1.8 * cfg.beam_radius) ** 2
    beam_intensity = jnp.exp(-2.0 * rsq / r2)
    catchment = (1.0 / (jnp.pi * ring_R2)) * jnp.exp(-2.0 * rsq / ring_R2)
    # Probability density shaped by beam intensity ∝ where the melt pool is.
    shape = catchment * (0.2 + 0.8 * beam_intensity)
    # Integral of shape across all particles in the continuous limit is
    # approximately unity; discrete sum normalization handles the discrete
    # particle case robustly so total ṁ over all particles equals mdot_total.
    discrete_sum = jnp.sum(shape)
    # If nothing is near the laser — output exactly 0 everywhere (avoid 0/0).
    norm = jnp.where(discrete_sum > 1e-18, 1.0 / discrete_sum, 0.0)
    s_per_particle = mdot_total * norm * shape
    # Suppress powder outside a finite catchment ring for numerical cleanliness.
    s_per_particle = jnp.where(rsq > 4.0 * ring_R2, 0.0, s_per_particle)
    return s_per_particle


def _step_lsf_common(
    method_step_fn,
    solver_state,
    am_state: ParticleAMState,
    dt: float, t: float, solver_cfg, cfg,
    paths, layer_start_times, layer_durations,
    *, dim, walls, cp, alpha_T, tau,
):
    """Run the method-specific AM step, then apply LSF powder mass source.

    Mass accumulation shifts ``activation_time`` earlier for particles that
    are still fully unactivated but receive a lot of powder (they effectively
    join the part as deposited).  Numerical behaviour mirrors DEM-AM exactly
    for the mechanical/thermal parts — LSF only differs by an *incremental*
    mass-addition and activation-front update.
    """
    from src.diffmech.methods.am.process import LSFConfig
    solver_new, am_new = method_step_fn(
        solver_state, am_state, dt, t, solver_cfg, cfg,
        paths, layer_start_times, layer_durations,
        walls=walls, cp=cp, alpha_T=alpha_T, tau=tau,
    )
    if not isinstance(cfg, LSFConfig):
        return solver_new, am_new

    src = lsf_powder_deposition_profile(
        am_new.position, t, paths, layer_start_times, layer_durations,
        cfg, dim=dim,
    )
    dm = src * dt
    # Add to particle mass.  For simplicity, keep volume constant so density
    # changes (a simple proxy for progressive deposition).  Full volumetric
    # growth would need remeshing in MPM/SPH and is deferred to a future pass.
    new_mass = am_new.mass + dm
    # Activate particles that accumulate powder quickly (if they were still
    # unactivated).  Use dm / m_threshold ≈ fraction of deposit to map into
    # a small negative shift of activation_time (i.e., "activates earlier").
    m0 = jnp.mean(am_state.mass) + 1e-18
    frac_deposited = dm / m0
    # New activation time = min(current activation, t + tiny_offset) where
    # powder lands.  This is JAX-safe (no loops, element-wise min).
    new_activation = jnp.minimum(
        am_new.activation_time,
        jnp.where(frac_deposited > 1e-3, t + 0.5e-3, am_new.activation_time),
    )
    # Also raise temperature slightly via powder pre-heat: the enthalpy of
    # the powder stream is taken as preheat_temp × cp × dm, so ΔT = cp·T·dm/(m·cp)
    # = T_preheat · dm / m.  Approximates the in-flight powder enthalpy.
    T_sensible = cfg.preheat_temp * dm / (new_mass + 1e-18)
    new_T = am_new.temperature + T_sensible

    am_lsf = ParticleAMState(
        position=am_new.position,
        activation_time=new_activation,
        temperature=new_T,
        mass=new_mass,
        layer_id=am_new.layer_id,
        rho=am_state.rho,
    )
    return solver_new, am_lsf


def step_lsf_am(
    solver_state,  # DEM / MPM / SPH state — inferred by kind string
    am_state: ParticleAMState,
    dt: float, t: float, solver_cfg, cfg,
    paths: MultiLayerPath,
    layer_start_times, layer_durations,
    method: str = "dem",
    *,
    walls=None, cp: float = 500.0, alpha_T: float = 1e-5, tau: float = 1e-3,
):
    """Unified LSF-aware one-step AM stepper.

    ``method`` selects the underlying solver: ``"dem"``, ``"mpm"``, or
    ``"sph"``.  After the base AM step, adds a co-axial powder mass
    source and updates particle masses / activation times / pre-heat
    enthalpy, simulating continuous DED / LSF deposition.
    """
    if method == "dem":
        base = step_dem_am
    elif method == "mpm":
        base = step_mpm_am
    elif method == "sph":
        base = step_sph_am
    else:
        raise ValueError(f"unknown solver method {method!r}")

    return _step_lsf_common(
        base, solver_state, am_state, dt, t, solver_cfg, cfg,
        paths, layer_start_times, layer_durations,
        dim=am_state.dim, walls=walls, cp=cp, alpha_T=alpha_T, tau=tau,
    )


def step_lsf_am_scan(
    solver_state0,
    am_state0: ParticleAMState,
    dt: float, n_steps: int, solver_cfg, cfg,
    paths: MultiLayerPath,
    layer_start_times, layer_durations,
    method: str = "dem",
    *,
    walls=None, cp: float = 500.0, alpha_T: float = 1e-5, tau: float = 1e-3,
):
    """Scan ``n_steps`` of :func:`step_lsf_am` via ``jax.lax.scan``."""
    starts = jnp.asarray(layer_start_times, dtype=jnp.float64)
    durations = jnp.asarray(layer_durations, dtype=jnp.float64)

    def body(carry, i):
        s_state, am_s, t_cur = carry
        t_nxt = t_cur + dt
        s_state2, am_s2 = step_lsf_am(
            s_state, am_s, dt, t_nxt, solver_cfg, cfg,
            paths, starts, durations, method,
            walls=walls, cp=cp, alpha_T=alpha_T, tau=tau,
        )
        return (s_state2, am_s2, t_nxt), (s_state2, am_s2)

    t0 = jnp.asarray(0.0, dtype=jnp.float64)
    (solver_final, am_final, _), _ = jax.lax.scan(
        body, (solver_state0, am_state0, t0), jnp.arange(n_steps),
    )
    return solver_final, am_final


__all__ = [
    # 2D SDFs
    "sdf_circle_2d", "sdf_box_2d", "sdf_from_polygon_2d",
    # state
    "ParticleAMState",
    # setup
    "ParticleAMProblem", "setup_particle_am",
    "place_particles_in_sdf",
    "assign_particle_activation_times", "particle_layer_times",
    # heat source
    "laser_heat_on_particles",
    # DEM-AM
    "step_dem_am", "step_dem_am_scan",
    # MPM-AM
    "step_mpm_am", "step_mpm_am_scan",
    # SPH-AM
    "step_sph_am", "step_sph_am_scan",
    # LSF / DED continuous powder deposition
    "lsf_powder_deposition_profile", "step_lsf_am", "step_lsf_am_scan",
    # diagnostics
    "peak_temperature", "activated_fraction", "mean_displacement",
]
