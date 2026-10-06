"""Laser Solid Forming (LSF) simulation of a complex turbine-blade part.

Demonstrates the full differentiable AM build workflow for a **Direct Energy
Deposition** process (LSF / LDED):

    SDF geometry (airfoil blade = box ∩ cylinder + root)
        → per-layer slicing
        → spiral / contour scan paths (LSF-style deposition tracks)
        → layered hex8 mesh + smooth element-activation field
        → moving-Gaussian laser heat source (large melt pool, slow cooling)
        → layer-by-layer thermo-mechanical coupling
            (residual stress + part distortion)
        → VTU export for ParaView

LSF differs from SLM in that:
- the beam radius is ~2 mm (vs ~50 µm for SLM) → much larger melt pool,
- the scan speed is ~10 mm/s (vs ~1 m/s for SLM) → slower, more energy per
  unit length,
- material is fed continuously (powder/wire) into the moving spot,
- cooling is slower → coarser microstructure, lower residual stress per layer.

The laser power is the differentiable process knob.

Run::

    python examples/am_lsf_demo.py
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from diffmech.core import rectangular_mesh2d
from diffmech.materials import LinearElasticIsotropic
from diffmech.methods.am import (
    # geometry
    sdf_box, sdf_cylinder, sdf_torus, union, intersection, difference,
    part_bbox, slice_mask, layer_heights,
    # scan paths
    spiral_hatch, contour_hatch, multi_layer_paths,
    # process
    LSFConfig,
    # activation
    build_layered_mesh, assign_activation_times, part_cell_mask,
    # thermal
    setup_am_thermal,
    # thermo-mechanical
    solve_thermomechanical,
)
from diffmech.methods.fvm import (
    CartesianGrid, cartesian_grid_3d,
    step_thermal_scan, diffusion_dt,
)
from diffmech.preprocess.materials import AM_MATERIALS
from diffmech.solvers import DirichletBC
from diffmech.postprocess import FieldCollection, write_vtu


OUT_DIR = Path(__file__).resolve().parent.parent / "am_lsf_out"


# ===========================================================================
# Part geometry: a simplified turbine blade (airfoil + root)
# ===========================================================================
def blade_geometry():
    """SDF of a simplified turbine blade: airfoil (box-profile) + root base.

    The blade is built from:
      - an airfoil section (a swept box, here simplified to an elongated
        cylinder oriented along y for the tapered leading edge),
      - a cylindrical root at the bottom.

    The whole part is ~30 mm tall, 15 mm wide, 4 mm thick.
    """
    # Airfoil: a tall thin box (the blade profile) along z.
    airfoil = sdf_box(
        center=(0.0, 0.0, 12e-3),
        half_extents=(6e-3, 1.5e-3, 12e-3),
    )
    # Taper the tip: subtract a cylinder to round the leading edge.
    tip_round = sdf_cylinder(
        center=(0.0, 0.0, 22e-3), axis=1, radius=4e-3, half_length=2e-3,
    )
    airfoil = difference(airfoil, tip_round, k=0.5e-3)

    # Root: a cylinder at the base (where the blade attaches to the disk).
    root = sdf_cylinder(
        center=(0.0, 0.0, 1e-3), axis=2, radius=5e-3, half_length=2e-3,
    )

    # Combine: union of airfoil and root.
    return union(airfoil, root, k=0.5e-3)


# ===========================================================================
# Build the LSF scan strategy: spiral / contour per layer
# ===========================================================================
def build_scan_paths(sdf, layer_zs, *, nx=24, ny=24, search_range=(-0.01, 0.01)):
    """Generate LSF deposition paths for every layer of the blade.

    LSF typically uses **spiral** or **contour** passes (not the zigzag
    raster of SLM), depositing material along the outline of each cross-
    section. For blade-like parts the contour follows the airfoil profile.
    """
    bbox = part_bbox(sdf, resolution=40, margin=0.3e-3,
                     search_range=search_range)
    (xmin, xmax), (ymin, ymax), _ = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    grid_xy = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1)

    paths = []
    for k, z in enumerate(layer_zs):
        mask = np.asarray(slice_mask(sdf, float(z), jnp.asarray(grid_xy),
                                     smoothness=0.3e-3))
        # Use contour-fill for the blade cross-section.
        path = contour_hatch(
            mask, xs, ys,
            hatch_spacing=1.5e-3,        # 1.5 mm track spacing (LSF)
            contour_offset=0.3e-3,
            scan_speed=0.01,             # 10 mm/s (LSF is slow)
            layer_z=float(z),
        )
        # Fallback to a spiral if the contour is degenerate.
        if path.total_length < 1e-6:
            cx, cy = (xmin + xmax) * 0.5, (ymin + ymax) * 0.5
            radius = min(xmax - xmin, ymax - ymin) * 0.4
            path = spiral_hatch(
                (cx, cy), radius,
                n_turns=2.0, n_points=100,
                layer_z=float(z),
            )
        paths.append(path)
    return multi_layer_paths(paths), bbox


# ===========================================================================
# Thermo-mechanical simulation
# ===========================================================================
def run_lsf_build(laser_power: jnp.ndarray,
                  *, n_layers: int = 5, nx: int = 6, ny: int = 6,
                  thermal_grid: int = 20, thermal_steps: int = 50):
    """Run the full LSF build: thermal history + thermo-mechanical coupling.

    Parameters
    ----------
    laser_power : scalar
        Beam power [W] — the differentiable process knob.
    n_layers : int
        Number of build layers to simulate.
    """
    # 1. Geometry + slicing.
    sdf = blade_geometry()
    search_range = (-0.01, 0.026)  # blade extends ~25 mm in z
    bbox = part_bbox(sdf, resolution=48, margin=0.3e-3,
                     search_range=search_range)
    layer_zs = layer_heights(sdf, bbox, layer_thickness=2e-3)[:n_layers]

    # 2. Scan paths (contour-fill per layer).
    paths, _ = build_scan_paths(sdf, layer_zs, nx=24, ny=24,
                                search_range=search_range)

    # 3. Process config (LSF). laser_power is kept as a JAX tracer.
    cfg = LSFConfig(
        laser_power=laser_power,
        scan_speed=0.01,                # 10 mm/s
        layer_thickness=2e-3,           # 2 mm deposit per pass
        track_spacing=1.5e-3,
        beam_radius=2e-3,               # 2 mm melt pool
        powder_feed_rate=5e-4,
        absorption=0.35,
        deposition_efficiency=0.7,
        preheat_temp=473.0,             # 200 °C substrate preheat
    )

    # 4. Layered hex8 mesh + activation times.
    layered = build_layered_mesh(layer_zs, bbox, nx=nx, ny=ny, cells_per_layer=1)
    lengths = np.asarray([p.total_length for p in paths.layers])
    durations = lengths / cfg.scan_speed
    starts = np.concatenate([[0.0], np.cumsum(durations)[:-1]])
    layered = assign_activation_times(layered, starts)

    # Cells outside the part never activate.
    mask = part_cell_mask(layered, sdf)

    # 5. AM thermal problem (moving Gaussian heat source on the layered mesh).
    am_problem = setup_am_thermal(layered, paths, cfg, part_mask=mask)

    # 6. Run a coarse FVM thermal sweep to get a representative peak
    #    temperature / cooling rate.
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bbox
    g3 = cartesian_grid_3d(thermal_grid, thermal_grid, thermal_grid,
                           lx=xmax - xmin, ly=ymax - ymin, lz=zmax - zmin,
                           origin=(xmin, ymin, zmin))
    mat = AM_MATERIALS["IN718"]
    alpha_th = mat.alpha
    rho_cp = mat.rho_cp
    dt = diffusion_dt(g3, alpha=alpha_th, cfl=0.3)
    T0 = jnp.full((thermal_grid, thermal_grid, thermal_grid, 1),
                  cfg.preheat_temp)
    src = am_problem.source_fn

    def fvm_source(centers_jnp, t):
        q = src(centers_jnp, t)
        return q.reshape(g3.nx, g3.ny, g3.nz)

    T_hist = step_thermal_scan(
        T0, g3, dt, thermal_steps,
        alpha=alpha_th, rho_cp=rho_cp,
        source=fvm_source, bc="robin",
        T_inf=300.0, h_conv=15.0, k_cond=mat.k_cond,
        return_history=True,
    )
    T_peak = jnp.max(T_hist)
    dTdt = (T_hist[1:] - T_hist[:-1]) / dt
    cooling_rate = jnp.max(-dTdt)

    # 7. Thermo-mechanical coupling.
    #    LSF has a larger melt pool and slower cooling than SLM, so the
    #    thermal strain per layer is larger (bigger ΔT, same α_T).
    T_ref = mat.T_melt          # stress-free at melt (IN718 ~1609 K)
    T_amb = 300.0
    alpha_T = 13.0e-6           # IN718 CTE [1/K]
    # ΔT scales with laser power: LSF delivers more energy per unit length
    # (slower scan, higher power) → larger melt pool → larger contraction.
    dT_cool = -(T_ref - T_amb) * (laser_power / 1500.0) * 0.6
    eps_cool = alpha_T * dT_cool

    # Per-layer thermal strain magnitude (n_layers, n_cells).
    layer_ids = np.asarray(layered.layer_id)
    eps_per_layer = jnp.zeros((layered.n_layers, layered.n_cells))
    mask_np = np.asarray(mask)
    for k in range(layered.n_layers):
        layer_cells = (layer_ids == k) & mask_np
        eps_per_layer = eps_per_layer.at[k].set(
            jnp.where(layer_cells, eps_cool, 0.0))

    # Build-plate clamp: fix the bottom face (z = zmin).
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


# ===========================================================================
# Metrics for inverse design
# ===========================================================================
def residual_stress_metric(result: dict) -> jnp.ndarray:
    """Von-Mises-equivalent peak residual stress [Pa]."""
    stress = result["tm_result"].residual_stress  # (n_cells, 3, 3)
    sxx = stress[:, 0, 0]
    syy = stress[:, 1, 1]
    szz = stress[:, 2, 2]
    sxy = stress[:, 0, 1]
    syz = stress[:, 1, 2]
    sxz = stress[:, 0, 2]
    vm = jnp.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 +
                         (szz - sxx) ** 2 +
                         6.0 * (sxy ** 2 + syz ** 2 + sxz ** 2)))
    return jnp.max(jnp.abs(vm))


def distortion_metric(result: dict) -> jnp.ndarray:
    """Peak nodal displacement magnitude [m]."""
    U = result["tm_result"].U_final
    n_nodes = U.shape[0] // 3
    disp = U.reshape(n_nodes, 3)
    return jnp.max(jnp.linalg.norm(disp, axis=-1))


def loss_fn(laser_power: jnp.ndarray) -> jnp.ndarray:
    """Objective: minimise residual stress (lower ⇒ better part quality)."""
    result = run_lsf_build(laser_power)
    return residual_stress_metric(result)


# ===========================================================================
# Export
# ===========================================================================
def export_results(result: dict, out_dir: Path = OUT_DIR):
    """Write VTU snapshots of the blade build."""
    out_dir.mkdir(parents=True, exist_ok=True)
    layered = result["layered"]
    mesh = layered.mesh
    mask = result["mask"]

    # 1. Thermal history (final temperature on the FVM grid).
    T_hist = result["T_hist"]
    ng = result["T_hist"].shape[1]
    mesh_t = rectangular_mesh2d(nx=ng - 1, ny=ng - 1, lx=1.0, ly=1.0,
                                cell_type="quad4")
    T_final = np.asarray(T_hist[-1, :, :, 0])
    fc_t = FieldCollection(point_data={"temperature": T_final.ravel()})
    write_vtu(out_dir / "lsf_thermal_history.vtu", mesh_t, fc_t)

    # 2. Activation field (per-cell).
    act = np.asarray(layered.activation_field(
        layered.activation_time.max() + 1.0))
    fc_act = FieldCollection(
        cell_data={"activation": act,
                   "part_mask": mask.astype(np.float64)},
    )
    write_vtu(out_dir / "lsf_activation.vtu", mesh, fc_act)

    # 3. Residual stress (von Mises) + distortion.
    tm = result["tm_result"]
    stress = np.asarray(tm.residual_stress)  # (n_cells, 3, 3)
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
    write_vtu(out_dir / "lsf_thermomechanical.vtu", mesh, fc_tm, deformed=True)

    return out_dir


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("DiffMech — LSF (Laser Solid Forming) turbine-blade build simulation")
    print("=" * 70)

    # Forward pass at nominal power.
    power = jnp.array(1500.0)
    print(f"\n[1] Forward build (laser_power = {float(power):.0f} W):")
    result = run_lsf_build(power, n_layers=5, nx=6, ny=6)
    print(f"    peak temperature      T_max = {float(result['T_peak']):.1f} K")
    print(f"    peak cooling rate      Ṫ    = {float(result['cooling_rate']):.3e} K/s")
    print(f"    peak residual stress  σ_vm = "
          f"{float(residual_stress_metric(result)):.3e} Pa")
    print(f"    peak distortion       |u|  = "
          f"{float(distortion_metric(result)):.3e} m")

    # End-to-end gradient.
    print("\n[2] End-to-end gradient dσ_vm/d(laser_power):")
    g = float(jax.grad(loss_fn)(power))
    print(f"    gradient = {g:.3e} Pa/W")

    # Export.
    print("\n[3] Exporting VTU snapshots to am_lsf_out/ ...")
    out = export_results(result)
    print(f"    written to {out}")
    print("\nDone. Open the .vtu files in ParaView to inspect the build.")


if __name__ == "__main__":
    main()
