"""Standard load cases & Dirichlet BC generators for 2D / 3D meshes.

These helpers build :class:`~diffmech.solvers.DirichletBC` lists (and
optional body forces) for the most common mechanical test cases, so the user
does not need to manually select boundary nodes by coordinate each time.

The mesh is assumed to be an axis-aligned box ``[0, lx] × [0, ly]`` (2D) or
``[0, lx] × [0, ly] × [0, lz]`` (3D). All routines auto-detect the box
extent from the mesh's nodal coordinates.

Supported load cases
--------------------
- :func:`uniaxial_tension` — displacement-controlled tension along an axis
- :func:`uniaxial_compression` — negative tension
- :func:`simple_shear` — top face displaced along one axis, bottom fixed
- :func:`bending` — pure bending of a beam (left face clamped, right face rotated)
- :func:`biaxial` — symmetric tension on two perpendicular axes
- :func:`clamped_face` — fully clamped (all dofs fixed) on a box face
- :func:`pin_node` — fix all dofs of a single node (suppresses rigid modes)
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from diffmech.core import Mesh
from diffmech.solvers import DirichletBC


@dataclass(frozen=True)
class LoadCase:
    """Result of a load-case generator.

    Attributes
    ----------
    dirichlet_bcs : tuple of DirichletBC
        The Dirichlet boundary conditions to enforce.
    body_force : (dim,) array or None
        Optional body force (per unit volume), e.g. gravity.
    description : str
        Human-readable description.
    """

    dirichlet_bcs: tuple
    body_force: jnp.ndarray | None
    description: str


# ---------------------------------------------------------------------------
# Box-extent detection
# ---------------------------------------------------------------------------
def _box_extents(mesh: Mesh) -> tuple[float, ...]:
    nodes = np.asarray(mesh.nodes)
    return tuple(nodes.max(axis=0))


def _face_nodes(mesh: Mesh, axis: int, side: str) -> np.ndarray:
    """Nodes on a box face.

    axis : 0 (x), 1 (y), 2 (z)
    side : "min" or "max"
    """
    nodes = np.asarray(mesh.nodes)
    coord = nodes[:, axis]
    val = coord.min() if side == "min" else coord.max()
    return np.where(np.abs(coord - val) < 1e-9)[0].astype(np.int64)


def _bc_fixed_all_dofs(mesh: Mesh, node_ids: np.ndarray, value: float = 0.0) -> DirichletBC:
    """Pin all dofs of the given nodes to ``value``."""
    dim = mesh.dim
    n = len(node_ids)
    dofs = (np.asarray(node_ids)[:, None] * dim + np.arange(dim)[None, :]).ravel()
    return DirichletBC(
        dofs=jnp.asarray(dofs),
        values=jnp.full(dofs.shape[0], value, dtype=jnp.float64),
    )


def _bc_fixed_dofs(mesh: Mesh, node_ids: np.ndarray, dof_axis: int,
                   value: float = 0.0) -> DirichletBC:
    """Pin a single dof axis (e.g. x=0) of the given nodes."""
    dim = mesh.dim
    dofs = np.asarray(node_ids) * dim + dof_axis
    return DirichletBC(
        dofs=jnp.asarray(dofs),
        values=jnp.full(dofs.shape[0], value, dtype=jnp.float64),
    )


def _bc_prescribed_dofs(mesh: Mesh, node_ids: np.ndarray, dof_axis: int,
                        values: np.ndarray) -> DirichletBC:
    """Prescribe a vector of values for a single dof axis of the given nodes."""
    dim = mesh.dim
    dofs = np.asarray(node_ids) * dim + dof_axis
    return DirichletBC(
        dofs=jnp.asarray(dofs),
        values=jnp.asarray(values, dtype=jnp.float64),
    )


# ---------------------------------------------------------------------------
# Standard load cases
# ---------------------------------------------------------------------------
def clamped_face(mesh: Mesh, *, axis: int, side: str) -> LoadCase:
    """Fully clamp (all dofs = 0) the nodes on one box face.

    Examples
    --------
    >>> case = clamped_face(mesh, axis=0, side="min")  # clamp x=0 face
    """
    nodes = _face_nodes(mesh, axis, side)
    bc = _bc_fixed_all_dofs(mesh, nodes, value=0.0)
    return LoadCase(
        dirichlet_bcs=(bc,),
        body_force=None,
        description=f"all dofs fixed on {['x','y','z'][axis]}={side} face",
    )


def pin_node(mesh: Mesh, node_id: int) -> LoadCase:
    """Pin all dofs of a single node (suppresses rigid translation)."""
    bc = _bc_fixed_all_dofs(mesh, np.array([node_id], dtype=np.int64))
    return LoadCase(
        dirichlet_bcs=(bc,),
        body_force=None,
        description=f"node {node_id} pinned",
    )


def uniaxial_tension(
    mesh: Mesh,
    *,
    axis: int = 0,
    strain: float = 0.01,
    lateral_free: bool = True,
    pin_secondary: bool = True,
) -> LoadCase:
    """Displacement-controlled uniaxial tension along ``axis``.

    The min-face of ``axis`` is clamped in that direction (``u_axis = 0``);
    the max-face is displaced by ``strain * L_axis``. If ``lateral_free`` is
    True the lateral faces are left traction-free (only the loading-axis
    displacement is imposed). If ``pin_secondary`` is True a single node on
    the min-face has its lateral dofs pinned, suppressing rigid translation.

    Parameters
    ----------
    axis : 0 (x) or 1 (y) or 2 (z)
    strain : applied nominal strain (use negative for compression)
    """
    extents = _box_extents(mesh)
    L = extents[axis]
    disp = strain * L
    min_nodes = _face_nodes(mesh, axis, "min")
    max_nodes = _face_nodes(mesh, axis, "max")
    bcs: list[DirichletBC] = [
        _bc_fixed_dofs(mesh, min_nodes, dof_axis=axis, value=0.0),
        _bc_prescribed_dofs(mesh, max_nodes, dof_axis=axis,
                            values=np.full(len(max_nodes), disp)),
    ]
    if not lateral_free:
        # Fully clamp both min and max faces (plane-strain-ish, lateral dofs=0)
        bcs.append(_bc_fixed_all_dofs(mesh, min_nodes, value=0.0))
    if pin_secondary:
        # Pin lateral dofs of the first node on the min face
        dim = mesh.dim
        first = min_nodes[0]
        for d in range(dim):
            if d != axis:
                bcs.append(_bc_fixed_dofs(mesh, np.array([first]),
                                          dof_axis=d, value=0.0))
    return LoadCase(
        dirichlet_bcs=tuple(bcs),
        body_force=None,
        description=f"uniaxial tension {strain:.4f} along {['x','y','z'][axis]}",
    )


def uniaxial_compression(mesh: Mesh, *, axis: int = 0, strain: float = -0.01,
                         **kwargs) -> LoadCase:
    """Displacement-controlled uniaxial compression (negative strain)."""
    return uniaxial_tension(mesh, axis=axis, strain=strain, **kwargs)


def simple_shear(
    mesh: Mesh,
    *,
    shear_axis: int = 0,
    normal_axis: int = 1,
    gamma: float = 0.01,
) -> LoadCase:
    """Displacement-controlled simple shear.

    The ``normal_axis`` = min face is fully clamped (all dofs = 0); the
    ``normal_axis`` = max face is displaced by ``gamma * L_normal`` along the
    ``shear_axis`` direction (its normal-axis displacement is fixed at 0 to
    keep the spacing constant).

    Parameters
    ----------
    shear_axis : direction of the imposed tangential displacement
    normal_axis : direction normal to the sheared faces (typically y in 2D)
    gamma : applied engineering shear strain γ (Δu_shear / L_normal)
    """
    if shear_axis == normal_axis:
        raise ValueError("shear_axis and normal_axis must differ")
    extents = _box_extents(mesh)
    L = extents[normal_axis]
    disp = gamma * L
    min_nodes = _face_nodes(mesh, normal_axis, "min")
    max_nodes = _face_nodes(mesh, normal_axis, "max")
    bcs: list[DirichletBC] = [
        # Fully clamp the bottom face
        _bc_fixed_all_dofs(mesh, min_nodes, value=0.0),
        # Top face: u_shear = disp, u_normal = 0
        _bc_prescribed_dofs(mesh, max_nodes, dof_axis=shear_axis,
                            values=np.full(len(max_nodes), disp)),
        _bc_fixed_dofs(mesh, max_nodes, dof_axis=normal_axis, value=0.0),
    ]
    return LoadCase(
        dirichlet_bcs=tuple(bcs),
        body_force=None,
        description=(f"simple shear γ={gamma:.4f} "
                     f"({['x','y','z'][shear_axis]}-dir, "
                     f"normal {['x','y','z'][normal_axis]})"),
    )


def biaxial(
    mesh: Mesh,
    *,
    strain_x: float = 0.01,
    strain_y: float = 0.01,
    pin_corner: bool = True,
) -> LoadCase:
    """Symmetric biaxial tension on a 2D mesh (both x-faces and y-faces pulled).

    The min-faces in x and y are held at u=0, the max-faces are displaced by
    ``strain * L``. To suppress rigid rotation, one corner node is fully
    pinned.
    """
    if mesh.dim != 2:
        raise ValueError("biaxial load case is 2D-only")
    extents = _box_extents(mesh)
    Lx, Ly = extents[0], extents[1]
    bcs: list[DirichletBC] = [
        # x=0 face: u_x = 0
        _bc_fixed_dofs(mesh, _face_nodes(mesh, 0, "min"), dof_axis=0, value=0.0),
        # x=L face: u_x = strain_x * Lx
        _bc_prescribed_dofs(mesh, _face_nodes(mesh, 0, "max"), dof_axis=0,
                            values=np.full(len(_face_nodes(mesh, 0, "max")),
                                           strain_x * Lx)),
        # y=0 face: u_y = 0
        _bc_fixed_dofs(mesh, _face_nodes(mesh, 1, "min"), dof_axis=1, value=0.0),
        # y=L face: u_y = strain_y * Ly
        _bc_prescribed_dofs(mesh, _face_nodes(mesh, 1, "max"), dof_axis=1,
                            values=np.full(len(_face_nodes(mesh, 1, "max")),
                                           strain_y * Ly)),
    ]
    if pin_corner:
        nodes = np.asarray(mesh.nodes)
        corner = np.where((nodes[:, 0] < 1e-9) & (nodes[:, 1] < 1e-9))[0][0]
        bcs.append(_bc_fixed_all_dofs(mesh, np.array([corner])))
    return LoadCase(
        dirichlet_bcs=tuple(bcs),
        body_force=None,
        description=f"biaxial (εx={strain_x:.4f}, εy={strain_y:.4f})",
    )


def bending(
    mesh: Mesh,
    *,
    axis: int = 0,
    thickness_axis: int = 1,
    curvature: float = 0.01,
    pin_corner: bool = True,
) -> LoadCase:
    """Pure-bending load case on a beam mesh.

    The beam is along ``axis`` (length ``L``) with thickness along
    ``thickness_axis`` (height ``H``). The min-face of ``axis`` is clamped; the
    max-face of ``axis`` is given a linear displacement profile in
    ``thickness_axis`` corresponding to curvature ``κ``::

        u_axis = κ * x * H / 2 ... (rigid rotation; suppressed by clamp)
        u_thickness = -κ * x_axis² / 2  ... bending deflection

    Implementation: we apply a rotation ``θ = κ * L`` on the max-face (linear
    in ``thickness_axis`` displacement) and clamp the min-face.

    Parameters
    ----------
    curvature : beam curvature κ (1/L). Tip deflection ≈ κ * L² / 2.
    """
    extents = _box_extents(mesh)
    L = extents[axis]
    nodes = np.asarray(mesh.nodes)
    min_nodes = _face_nodes(mesh, axis, "min")
    max_nodes = _face_nodes(mesh, axis, "max")

    bcs: list[DirichletBC] = []
    # Clamp the left face fully.
    bcs.append(_bc_fixed_all_dofs(mesh, min_nodes, value=0.0))
    # On the right face apply a rotation θ = κ * L about the beam's neutral axis.
    theta = curvature * L
    # The rotation axis is the one perpendicular to both axis and thickness_axis.
    others = [a for a in (0, 1, 2) if a not in (axis, thickness_axis)]
    rot_axis = others[0] if others else 2
    # For each node on the max face:
    #   u_thickness = theta * (y - y_centroid)   (in-plane rotation)
    #   u_axis = 0 (we don't impose axial motion)
    y_coord = nodes[max_nodes, thickness_axis]
    y_centroid = y_coord.mean()
    u_thick = theta * (y_coord - y_centroid)
    bcs.append(_bc_prescribed_dofs(mesh, max_nodes, dof_axis=thickness_axis,
                                    values=u_thick))
    # Keep the right face's loading-axis displacement zero (no axial stretch).
    bcs.append(_bc_fixed_dofs(mesh, max_nodes, dof_axis=axis, value=0.0))

    if pin_corner:
        # Pin the first node on the min face's lateral (rot_axis) dof to
        # suppress out-of-plane rigid translation (matters in 3D).
        bcs.append(_bc_fixed_dofs(mesh, np.array([min_nodes[0]]),
                                  dof_axis=rot_axis, value=0.0))

    return LoadCase(
        dirichlet_bcs=tuple(bcs),
        body_force=None,
        description=(f"pure bending κ={curvature:.4f} about "
                     f"{['x','y','z'][rot_axis]}"),
    )


# ---------------------------------------------------------------------------
# Body-force / gravity load case (no Dirichlet BC except rigid-mode suppression)
# ---------------------------------------------------------------------------
def gravity(
    mesh: Mesh,
    *,
    g: float = 9.81,
    direction: int = -1,
    clamped_face_axis: int = 0,
    clamped_face_side: str = "min",
) -> LoadCase:
    """Gravity loading on a clamped beam / cantilever.

    Parameters
    ----------
    direction : -1 (last axis) or 0 / 1 / 2
        Direction of gravity (negative sign convention: gravity points down).
    """
    dim = mesh.dim
    body = np.zeros(dim, dtype=np.float64)
    axis = direction if direction >= 0 else dim - 1
    body[axis] = -g
    clamp = clamped_face(mesh, axis=clamped_face_axis, side=clamped_face_side)
    return LoadCase(
        dirichlet_bcs=clamp.dirichlet_bcs,
        body_force=jnp.asarray(body),
        description=f"gravity g={g} along {['x','y','z'][axis]}",
    )
