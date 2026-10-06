"""Selective Laser Melting (SLM) simulation of a complex gear part.

Demonstrates the full differentiable AM build workflow for a **Powder Bed
Fusion** process:

    SDF geometry (spur gear)
        → per-layer slicing (differentiable occupancy)
        → zigzag hatch scan paths (rotated 67°/layer)
        → layered hex8 mesh + smooth element-activation field
        → moving-Gaussian laser heat source on an FVM grid
        → peak-temperature / cooling-rate extraction
        → layer-by-layer thermo-mechanical coupling
            (residual stress + part distortion)
        → VTU export for ParaView

The laser power is the differentiable process knob: the whole chain is
written with ``jax`` so ``jax.grad`` propagates from ``laser_power`` to the
final residual-stress / distortion metrics.

Run::

    python examples/am_slm_demo.py
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
    sdf_gear, part_bbox, slice_mask, layer_heights,
    # scan paths
    zigzag_hatch, multi_layer_paths, MultiLayerPath,
    # process
    SLMConfig,
    # activation
    build_layered_mesh, assign_activation_times, part_cell_mask,
    # thermal
    setup_am_thermal,
    # thermo-mechanical
    solve_thermomechanical,
)
from diffmech.methods.fvm import (
    CartesianGrid, cartesian_grid_3d,
    ThermalConfig, step_thermal_scan, diffusion_dt,
)
from diffmech.preprocess.materials import AM_MATERIALS
from diffmech.solvers import DirichletBC
from diffmech.postprocess import FieldCollection, write_vtu


OUT_DIR = Path(__file__).resolve().parent.parent / "am_slm_out"


# ===========================================================================
# Part geometry: a spur gear (complex AM shape)
# ===========================================================================
def gear_geometry():
    """SDF of a spur gear centred at the origin, axis = z."""
    # 12-tooth gear, 20 mm outer radius, 4 mm thick.
    return sdf_gear(
        center=(0.0, 0.0, 0.0), axis=2,
        outer_radius=10e-3, n_teeth=12,
        tooth_depth=1.5e-3, thickness=4e-3,
    )


# ===========================================================================
# Build the SLM scan strategy: zigzag per layer, rotated each layer
# ===========================================================================
def build_scan_paths(sdf, layer_zs, *, nx=24, ny=24, search_range=(-0.02, 0.02)):
    """Generate zigzag hatch paths for every layer of the gear.

    The hatch angle is rotated 67° per layer (the classic SLM "isotropic"
    rotation) — here approximated by transposing the grid on alternate layers.
    """
    bbox = part_bbox(sdf, resolution=32, margin=0.2e-3,
                     search_range=search_range)
    (xmin, xmax), (ymin, ymax), _ = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    grid_xy = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1)

    paths = []
    for k, z in enumerate(layer_zs):
        mask = np.asarray(slice_mask(sdf, float(z), jnp.asarray(grid_xy),
                                     smoothness=0.3e-3))
        # Rotate hatch direction each layer (0° / 90° alternating as a proxy
        # for the 67° rotation used in practice).
        direction = k % 2
        path = zigzag_hatch(
            mask, xs, ys,
            hatch_spacing=0.8e-3,        # 0.8 mm hatch spacing (scaled demo)
            direction=direction,
            scan_speed=1.0,
            layer_z=float(z),
            trim_margin=0.2e-3,
        )
        paths.append(path)
    return multi_layer_paths(paths), bbox


# ===========================================================================
# Thermo-mechanical simulation
# ===========================================================================
def run_slm_build(laser_power: jnp.ndarray,
                  *, n_layers: int = 4, nx: int = 6, ny: int = 6,
                  thermal_grid: int = 20, thermal_steps: int = 40):
    """Run the full SLM build: thermal history + thermo-mechanical coupling.

    Parameters
    ----------
    laser_power : scalar
        Absorbed laser power [W] — the differentiable process knob.
    n_layers : int
        Number of build layers to simulate (kept small for a fast demo).
    """
    # 1. Geometry + slicing.
    sdf = gear_geometry()
    search_range = (-0.015, 0.015)  # ±15 mm (gear is 10 mm radius)
    bbox = part_bbox(sdf, resolution=40, margin=0.3e-3,
                     search_range=search_range)
    layer_zs = layer_heights(sdf, bbox, layer_thickness=1e-3)[:n_layers]

    # 2. Scan paths (zigzag per layer).
    paths, _ = build_scan_paths(sdf, layer_zs, nx=24, ny=24,
                                search_range=search_range)

    # 3. Process config (SLM). laser_power is kept as a JAX tracer so the
    #    whole pipeline stays differentiable w.r.t. it.
    cfg = SLMConfig(
        laser_power=laser_power,
        scan_speed=0.1,                 # 100 mm/s (demo-scaled)
        layer_thickness=1e-3,
        hatch_spacing=0.8e-3,
        beam_radius=0.6e-3,
        absorption=0.5,
        preheat_temp=373.0,
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
    #    temperature / cooling rate (demonstrates the moving source).
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
    # Sample the AM heat source onto the FVM grid.
    src = am_problem.source_fn

    def fvm_source(centers_jnp, t):
        q = src(centers_jnp, t)
        return q.reshape(g3.nx, g3.ny, g3.nz)

    T_hist = step_thermal_scan(
        T0, g3, dt, thermal_steps,
        alpha=alpha_th, rho_cp=rho_cp,
        source=fvm_source, bc="robin",
        T_inf=300.0, h_conv=20.0, k_cond=mat.k_cond,
        return_history=True,
    )
    T_peak = jnp.max(T_hist)
    # Peak cooling rate after the laser passes.
    dTdt = (T_hist[1:] - T_hist[:-1]) / dt
    cooling_rate = jnp.max(-dTdt)

    # 7. Thermo-mechanical coupling.
    #    Thermal strain per layer = α_T · (T_layer − T_ref). We use a
    #    representative per-cell ΔT driven by the laser power (differentiable
    #    proxy): cells in the active layer cool from melt toward ambient.
    T_ref = mat.T_melt          # stress-free at melt
    T_amb = 300.0
    alpha_T = 8.6e-6            # Ti6Al4V CTE [1/K]
    # ΔT scales with laser power: more power → hotter melt pool → larger
    # cooling contraction. All quantities are JAX tracers (differentiable).
    dT_cool = -(T_ref - T_amb) * (laser_power / 200.0) * 0.5
    # thermal_strain_per_layer expects α_T·ΔT (dimensionless strain).
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
        alpha_T=8.6e-6,           # Ti6Al4V CTE [1/K]
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
    # Von Mises from the full 3D stress tensor.
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
    result = run_slm_build(laser_power)
    return residual_stress_metric(result)


# ===========================================================================
# Export
# ===========================================================================
def export_results(result: dict, out_dir: Path = OUT_DIR):
    """Write VTU snapshots of the gear build."""
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
    write_vtu(out_dir / "slm_thermal_history.vtu", mesh_t, fc_t)

    # 2. Activation field (which cells are part of the gear) — per-cell data.
    act = np.asarray(layered.activation_field(
        layered.activation_time.max() + 1.0))
    fc_act = FieldCollection(
        cell_data={"activation": act,
                   "part_mask": mask.astype(np.float64)},
    )
    write_vtu(out_dir / "slm_activation.vtu", mesh, fc_act)

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
    write_vtu(out_dir / "slm_thermomechanical.vtu", mesh, fc_tm, deformed=True)

    # 4. Per-layer displacement history.
    U_hist = np.asarray(tm.U_history)  # (n_layers, n_dofs)
    return out_dir


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("DiffMech — SLM (Selective Laser Melting) gear build simulation")
    print("=" * 64)

    # Forward pass at nominal power.
    power = jnp.array(200.0)
    print(f"\n[1] Forward build (laser_power = {float(power):.0f} W):")
    result = run_slm_build(power, n_layers=4, nx=6, ny=6)
    print(f"    peak temperature      T_max = {result['T_peak']:.1f} K")
    print(f"    peak cooling rate      Ṫ    = {result['cooling_rate']:.3e} K/s")
    print(f"    peak residual stress  σ_vm = "
          f"{float(residual_stress_metric(result)):.3e} Pa")
    print(f"    peak distortion       |u|  = "
          f"{float(distortion_metric(result)):.3e} m")

    # End-to-end gradient.
    print("\n[2] End-to-end gradient dσ_vm/d(laser_power):")
    g = float(jax.grad(loss_fn)(power))
    print(f"    gradient = {g:.3e} Pa/W")

    # Export.
    print("\n[3] Exporting VTU snapshots to am_slm_out/ ...")
    out = export_results(result)
    print(f"    written to {out}")
    print("\nDone. Open the .vtu files in ParaView to inspect the build.")


if __name__ == "__main__":
    main()
