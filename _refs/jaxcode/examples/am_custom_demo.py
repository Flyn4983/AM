"""Arbitrary complex geometry + user-specified laser path.

Demonstrates the two principal extensibility entry points of the AM module:

1. **Arbitrary complex geometry** — instead of using the analytic SDF
   primitives (:func:`sdf_gear`, :func:`sdf_box`, …), the part is loaded from
   a **triangle mesh** (the universal CAD-export format) via
   :func:`sdf_from_mesh`. Here we synthesise a mesh procedurally (an
   "impeller-like" hub + 6 curved blades) so the example is self-contained;
   in practice you would replace the synthesiser with
   ``sdf_from_stl("my_part.stl")``.

2. **User-specified laser path** — instead of using the auto-generated
   zigzag/contour/spiral strategies, each layer's laser trajectory is supplied
   as an explicit ``(n, 2)`` waypoint array via :func:`from_waypoints`. Here
   the per-layer path is a series of "blade-sweep" arcs that deposit material
   along each blade's leading edge — a toolpath no built-in hatch strategy
   would produce.

The whole pipeline (mesh SDF → slicing → custom path → thermal →
thermo-mechanical) is differentiable end-to-end w.r.t. the laser power.

Run::

    python examples/am_custom_demo.py
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from diffmech.materials import LinearElasticIsotropic
from diffmech.methods.am import (
    # arbitrary geometry from a triangle mesh
    sdf_from_mesh,
    # geometry helpers
    part_bbox, slice_mask, layer_heights,
    # user-specified laser paths
    from_waypoints, chain_paths, multi_layer_paths,
    # process + activation + thermal + thermo-mechanical
    SLMConfig,
    build_layered_mesh, assign_activation_times, part_cell_mask,
    setup_am_thermal, solve_thermomechanical,
)
from diffmech.methods.fvm import (
    cartesian_grid_3d, step_thermal_scan, diffusion_dt,
)
from diffmech.preprocess.materials import AM_MATERIALS
from diffmech.solvers import DirichletBC
from diffmech.postprocess import FieldCollection, write_vtu


OUT_DIR = Path(__file__).resolve().parent.parent / "am_custom_out"


# ===========================================================================
# 1. Arbitrary complex geometry: a procedural impeller (hub + 6 blades)
# ===========================================================================
def _unit_cylinder_mesh(radius: float, height: float,
                        n_circ: int = 24, n_z: int = 2):
    """Closed triangle mesh of a cylinder (axis = z), CCW-outward winding."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_circ, endpoint=False)
    z_layers = np.linspace(0.0, height, n_z + 1)
    # (n_z+1, n_circ, 3) vertex grid.
    verts = np.stack([
        np.tile(radius * np.cos(theta)[None, :], (n_z + 1, 1)),
        np.tile(radius * np.sin(theta)[None, :], (n_z + 1, 1)),
        np.tile(z_layers[:, None], (1, n_circ)),
    ], axis=-1)
    # Flatten with consistent indexing: v[i, j] -> i * n_circ + j.
    verts_flat = verts.reshape(-1, 3)
    n = n_circ
    tris = []
    # Side wall: each cell (i, j)-(i+1, j)-(i+1, j+1)-(i, j+1).
    for i in range(n_z):
        for j in range(n_circ):
            jp = (j + 1) % n_circ
            a = i * n + j
            b = (i + 1) * n + j
            c = (i + 1) * n + jp
            d = i * n + jp
            # Outward-facing quads → 2 triangles each.
            tris.append([a, d, b])  # ensure CCW-outward
            tris.append([b, d, c])
    # Bottom cap (z=0): fan from the centre vertex.
    centre_bot = len(verts_flat)
    verts_flat = np.concatenate([verts_flat, [[0., 0., 0.]]], axis=0)
    for j in range(n_circ):
        jp = (j + 1) % n_circ
        # CCW viewed from below (normal -z): reverse winding.
        tris.append([0 * n + jp, 0 * n + j, centre_bot])
    # Top cap (z=height): fan from the centre vertex.
    centre_top = len(verts_flat)
    verts_flat = np.concatenate([verts_flat,
                                [[0., 0., height]]], axis=0)
    for j in range(n_circ):
        jp = (j + 1) % n_circ
        tris.append([n_z * n + j, n_z * n + jp, centre_top])
    return verts_flat.astype(np.float64), np.asarray(tris, dtype=np.int64)


def _blade_mesh(hub_radius: float, blade_length: float, height: float,
                n_span: int = 8, n_z: int = 2):
    """A thin twisted blade as a closed triangle mesh (a swept ribbon)."""
    # Blade: a thin ribbon from hub_radius to hub_radius+blade_length,
    # twisted ~30° per unit height (a stylised impeller blade).
    s = np.linspace(0.0, 1.0, n_span)
    z_layers = np.linspace(0.0, height, n_z + 1)
    r = hub_radius + s * blade_length
    twist = np.linspace(0.0, np.pi / 6, n_z + 1)  # 30° total twist
    thickness = 0.3e-3  # 0.3 mm blade thickness
    # Build a thin ribbon: leading edge at +thickness/2, trailing at -thickness/2.
    # (n_z+1, n_span, 2 edges, 3 coords)
    grid = np.zeros((n_z + 1, n_span, 2, 3))
    for i, z in enumerate(z_layers):
        ang = float(twist[i])
        ca, sa = np.cos(ang), np.sin(ang)
        for j, rj in enumerate(r):
            x = float(rj) * ca
            y = float(rj) * sa
            grid[i, j, 0, 0] = x - thickness * sa
            grid[i, j, 0, 1] = y + thickness * ca
            grid[i, j, 0, 2] = z
            grid[i, j, 1, 0] = x + thickness * sa
            grid[i, j, 1, 1] = y - thickness * ca
            grid[i, j, 1, 2] = z
    # Indexing: v(i, j, e) = i * (n_span * 2) + j * 2 + e.
    def idx(i, j, e):
        return i * (n_span * 2) + j * 2 + e
    verts = grid.reshape(-1, 3)
    tris = []
    # Side quads along the span (i, j)-(i, j+1).
    for i in range(n_z + 1):
        for j in range(n_span - 1):
            # Top edge (e=0): j → j+1.
            tris.append([idx(i, j, 0), idx(i, j + 1, 0),
                         idx(i, j + 1, 1)])
            tris.append([idx(i, j, 0), idx(i, j + 1, 1),
                         idx(i, j, 1)])
    # Side quads across z (i, j)-(i+1, j).
    for i in range(n_z):
        for j in range(n_span):
            # Leading edge (e=0): bottom face of the ribbon.
            tris.append([idx(i, j, 0), idx(i + 1, j, 0),
                         idx(i + 1, j, 1)])
            tris.append([idx(i, j, 0), idx(i + 1, j, 1),
                         idx(i, j, 1)])
    # Cap at s=0 (hub end): quad across the two edges at j=0.
    for i in range(n_z):
        tris.append([idx(i, 0, 0), idx(i + 1, 0, 1),
                     idx(i + 1, 0, 0)])
        tris.append([idx(i, 0, 0), idx(i, 0, 1),
                     idx(i + 1, 0, 1)])
    # Cap at s=1 (tip end): quad at j=n_span-1.
    for i in range(n_z):
        tris.append([idx(i, n_span - 1, 0), idx(i + 1, n_span - 1, 0),
                     idx(i + 1, n_span - 1, 1)])
        tris.append([idx(i, n_span - 1, 0), idx(i + 1, n_span - 1, 1),
                     idx(i, n_span - 1, 1)])
    return verts.astype(np.float64), np.asarray(tris, dtype=np.int64)


def impeller_mesh(n_blades: int = 6):
    """Procedural impeller mesh: central hub + n_blades twisted blades.

    Returns
    -------
    verts, tris : the combined triangle mesh (CCW-outward winding).
    """
    # Hub: 4 mm radius, 4 mm tall.
    hub_v, hub_t = _unit_cylinder_mesh(4e-3, 4e-3, n_circ=24, n_z=2)
    # 6 blades, each 6 mm long, 4 mm tall, twisted 30°.
    blade_v, blade_t = _blade_mesh(4e-3, 6e-3, 4e-3, n_span=8, n_z=2)
    all_v = [hub_v]
    all_t = [hub_t]
    offset = len(hub_v)
    for k in range(n_blades):
        # Rotate the blade around z by k * (2π / n_blades).
        ang = k * 2.0 * np.pi / n_blades
        ca, sa = np.cos(ang), np.sin(ang)
        rot = np.array([[ca, -sa, 0.0],
                        [sa, ca, 0.0],
                        [0.0, 0.0, 1.0]])
        rotated = blade_v @ rot.T
        all_v.append(rotated)
        all_t.append(blade_t + offset)
        offset += len(rotated)
    verts = np.concatenate(all_v, axis=0)
    tris = np.concatenate(all_t, axis=0)
    return verts, tris


def impeller_sdf():
    """Differentiable SDF of the impeller, built from its triangle mesh."""
    verts, tris = impeller_mesh(n_blades=6)
    return sdf_from_mesh(verts, tris, chunk=64)  # chunk for memory safety


# ===========================================================================
# 2. User-specified laser path: per-blade sweep arcs
# ===========================================================================
def blade_sweep_path(hub_radius: float, blade_length: float, z: float,
                     n_blades: int = 6, n_pts: int = 12):
    """A laser path that sweeps along each blade's leading edge.

    This is the kind of toolpath a slicer would never auto-generate from a
    cross-section mask — it requires knowledge of the part's *function*
    (deposit along each blade's aerodynamic leading edge, hub → tip).
    """
    waypoints = []
    for k in range(n_blades):
        ang = k * 2.0 * np.pi / n_blades
        ca, sa = np.cos(ang), np.sin(ang)
        # Walk from hub to tip along the blade centreline.
        s = np.linspace(0.0, 1.0, n_pts)
        r = hub_radius + s * blade_length
        xs = r * ca
        ys = r * sa
        for x, y in zip(xs, ys):
            waypoints.append([x, y])
    return np.asarray(waypoints, dtype=np.float64)


def build_user_paths(layer_zs, hub_radius=4e-3, blade_length=6e-3,
                    n_blades=6):
    """Generate per-layer user-specified laser paths (blade sweeps)."""
    paths = []
    for z in layer_zs:
        wp = blade_sweep_path(hub_radius, blade_length, float(z), n_blades)
        # The path goes hub→tip on blade 0, then jumps to blade 1's hub, etc.
        # Each blade sweep is itself a small ScanPath; chain them together.
        per_blade = np.array_split(wp, n_blades)
        sub_paths = [from_waypoints(seg, layer_z=float(z))
                     for seg in per_blade]
        path = chain_paths(*sub_paths, layer_z=float(z))
        paths.append(path)
    return multi_layer_paths(paths)


# ===========================================================================
# 3. End-to-end thermo-mechanical simulation
# ===========================================================================
def run_impeller_build(laser_power: jnp.ndarray,
                       *, n_layers: int = 4, nx: int = 8, ny: int = 8,
                       thermal_grid: int = 16, thermal_steps: int = 30):
    """Run the full impeller build with user-specified laser paths.

    Parameters
    ----------
    laser_power : scalar
        Absorbed laser power [W] — the differentiable process knob.
    """
    # 1. Arbitrary geometry from a triangle mesh.
    sdf = impeller_sdf()
    # Restrict the bbox search to ±15 mm (impeller is ~10 mm radius).
    bbox = part_bbox(sdf, resolution=24, margin=0.4e-3,
                     search_range=(-0.015, 0.015))
    layer_zs = layer_heights(sdf, bbox, layer_thickness=1e-3)[:n_layers]

    # 2. User-specified laser paths (blade sweeps — no auto-hatch used).
    paths = build_user_paths(layer_zs, hub_radius=4e-3,
                             blade_length=6e-3, n_blades=6)

    # 3. Process config (SLM-style).
    cfg = SLMConfig(
        laser_power=laser_power,
        scan_speed=0.1,
        layer_thickness=1e-3,
        hatch_spacing=0.8e-3,
        beam_radius=0.6e-3,
        absorption=0.5,
        preheat_temp=373.0,
    )

    # 4. Layered hex8 mesh + activation times.
    layered = build_layered_mesh(layer_zs, bbox, nx=nx, ny=ny,
                                 cells_per_layer=1)
    lengths = np.asarray([p.total_length for p in paths.layers])
    durations = lengths / cfg.scan_speed
    starts = np.concatenate([[0.0], np.cumsum(durations)[:-1]])
    layered = assign_activation_times(layered, starts)

    # Cells outside the impeller never activate.
    mask = part_cell_mask(layered, sdf)

    # 5. AM thermal problem with the custom path.
    am_problem = setup_am_thermal(layered, paths, cfg, part_mask=mask)

    # 6. Coarse FVM thermal sweep for peak T / cooling rate.
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bbox
    g3 = cartesian_grid_3d(thermal_grid, thermal_grid, thermal_grid,
                           lx=xmax - xmin, ly=ymax - ymin, lz=zmax - zmin,
                           origin=(xmin, ymin, zmin))
    mat = AM_MATERIALS["Ti6Al4V"]
    alpha_th = mat.alpha
    rho_cp = mat.rho_cp
    dt = diffusion_dt(g3, alpha=alpha_th, cfl=0.3)
    T0 = jnp.full((thermal_grid, thermal_grid, thermal_grid, 1),
                  cfg.preheat_temp)
    src = am_problem.source_fn

    def fvm_source(centers_jnp, t):
        return src(centers_jnp, t).reshape(g3.nx, g3.ny, g3.nz)

    T_hist = step_thermal_scan(
        T0, g3, dt, thermal_steps,
        alpha=alpha_th, rho_cp=rho_cp,
        source=fvm_source, bc="robin",
        T_inf=300.0, h_conv=20.0, k_cond=mat.k_cond,
        return_history=True,
    )
    T_peak = jnp.max(T_hist)
    dTdt = (T_hist[1:] - T_hist[:-1]) / dt
    cooling_rate = jnp.max(-dTdt)

    # 7. Thermo-mechanical coupling.
    T_ref = mat.T_melt
    T_amb = 300.0
    alpha_T = 8.6e-6
    dT_cool = -(T_ref - T_amb) * (laser_power / 200.0) * 0.5
    eps_cool = alpha_T * dT_cool

    layer_ids = np.asarray(layered.layer_id)
    eps_per_layer = jnp.zeros((layered.n_layers, layered.n_cells))
    mask_np = np.asarray(mask)
    for k in range(layered.n_layers):
        layer_cells = (layer_ids == k) & mask_np
        eps_per_layer = eps_per_layer.at[k].set(
            jnp.where(layer_cells, eps_cool, 0.0))

    z_nodes = np.asarray(layered.mesh.nodes)[:, 2]
    bottom_nodes = np.where(z_nodes <= zmin + 1e-9)[0]
    dim = 3
    bottom_dofs = jnp.asarray(
        np.concatenate([bottom_nodes * dim + d for d in range(dim)]))
    bcs = [DirichletBC.fixed(bottom_dofs, 0.0)]

    elastic = LinearElasticIsotropic(E=mat.E, nu=mat.nu)
    tm_result = solve_thermomechanical(
        layered, elastic,
        thermal_strain_per_layer=eps_per_layer,
        dirichlet_bcs=bcs,
        T_ref=float(T_ref),
        alpha_T=alpha_T,
        tau_activation=1e-3,
        dim=dim,
    )

    return {
        "layered": layered,
        "paths": paths,
        "cfg": cfg,
        "T_hist": T_hist,
        "T_peak": T_peak,
        "cooling_rate": cooling_rate,
        "tm_result": tm_result,
        "mat": mat,
        "mask": mask,
    }


def residual_stress_metric(result: dict) -> jnp.ndarray:
    stress = result["tm_result"].residual_stress
    sxx = stress[:, 0, 0]; syy = stress[:, 1, 1]; szz = stress[:, 2, 2]
    sxy = stress[:, 0, 1]; syz = stress[:, 1, 2]; sxz = stress[:, 0, 2]
    vm = jnp.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 +
                         (szz - sxx) ** 2 +
                         6.0 * (sxy ** 2 + syz ** 2 + sxz ** 2)))
    return jnp.max(jnp.abs(vm))


def distortion_metric(result: dict) -> jnp.ndarray:
    U = result["tm_result"].U_final
    n_nodes = U.shape[0] // 3
    disp = U.reshape(n_nodes, 3)
    return jnp.max(jnp.linalg.norm(disp, axis=-1))


def loss_fn(laser_power: jnp.ndarray) -> jnp.ndarray:
    return residual_stress_metric(run_impeller_build(laser_power))


# ===========================================================================
# Export
# ===========================================================================
def export_results(result: dict, out_dir: Path = OUT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    layered = result["layered"]
    mesh = layered.mesh
    mask = result["mask"]

    # 1. Custom laser path (per-layer waypoints).
    paths = result["paths"]
    all_wp = []
    for k, p in enumerate(paths.layers):
        wp = np.asarray(p.waypoints)
        z = float(p.layer_z)
        all_wp.append(np.concatenate([wp,
                                     np.full((len(wp), 1), z)], axis=1))
    wp_all = np.concatenate(all_wp, axis=0)
    fc_path = FieldCollection(
        point_data={"laser_path_order": np.arange(len(wp_all))},
    )
    from diffmech.core import rectangular_mesh2d
    mesh_path = rectangular_mesh2d(nx=1, ny=1, lx=1.0, ly=1.0,
                                   cell_type="quad4")
    # Write waypoints as a separate point cloud (VTK polydata-like).
    import meshio
    pm = meshio.Mesh(points=wp_all, cells=[("vertex",
                                            np.arange(len(wp_all))[:, None])])
    pm.write(str(out_dir / "custom_laser_path.vtu"))

    # 2. Activation field + part mask.
    act = np.asarray(layered.activation_field(
        layered.activation_time.max() + 1.0))
    fc_act = FieldCollection(
        cell_data={"activation": act,
                   "part_mask": mask.astype(np.float64)},
    )
    write_vtu(out_dir / "impeller_activation.vtu", mesh, fc_act)

    # 3. Residual stress + distortion.
    tm = result["tm_result"]
    stress = np.asarray(tm.residual_stress)
    sxx, syy, szz = stress[:, 0, 0], stress[:, 1, 1], stress[:, 2, 2]
    sxy, syz, sxz = stress[:, 0, 1], stress[:, 1, 2], stress[:, 0, 2]
    vm = np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 +
                        (szz - sxx) ** 2 +
                        6.0 * (sxy ** 2 + syz ** 2 + sxz ** 2)))
    U = np.asarray(tm.U_final)
    n_nodes = U.shape[0] // 3
    disp = U.reshape(n_nodes, 3)
    fc_tm = FieldCollection(
        point_data={"displacement": disp,
                    "displacement_magnitude": np.linalg.norm(disp, axis=-1)},
        cell_data={"von_mises_stress": vm,
                   "part_mask": mask.astype(np.float64)},
    )
    write_vtu(out_dir / "impeller_thermomechanical.vtu", mesh, fc_tm,
              deformed=True)
    return out_dir


def main():
    print("DiffMech — arbitrary geometry + user-specified laser path")
    print("=" * 70)
    print("Geometry : impeller (hub + 6 twisted blades) from a triangle mesh")
    print("Laser path: per-blade hub→tip sweep (user-supplied waypoints)")
    print()

    # Forward pass at nominal power.
    power = jnp.array(200.0)
    print(f"[1] Forward build (laser_power = {float(power):.0f} W):")
    result = run_impeller_build(power, n_layers=4, nx=8, ny=8)
    print(f"    peak temperature      T_max = {float(result['T_peak']):.1f} K")
    print(f"    peak cooling rate      Ṫ    = "
          f"{float(result['cooling_rate']):.3e} K/s")
    print(f"    peak residual stress  σ_vm = "
          f"{float(residual_stress_metric(result)):.3e} Pa")
    print(f"    peak distortion       |u|  = "
          f"{float(distortion_metric(result)):.3e} m")

    # End-to-end gradient.
    print("\n[2] End-to-end gradient dσ_vm/d(laser_power):")
    g = float(jax.grad(loss_fn)(power))
    print(f"    gradient = {g:.3e} Pa/W")

    # Export.
    print("\n[3] Exporting VTU snapshots to am_custom_out/ ...")
    out = export_results(result)
    print(f"    written to {out}")
    print("\nDone. Open the .vtu files in ParaView to inspect the build.")


if __name__ == "__main__":
    main()
