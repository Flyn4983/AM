"""Particle-method AM demo: DEM / MPM / SPH with arbitrary geometry + custom
laser paths, in 2D and 3D.

This demo exercises the same two extensibility entry points as the FEM-based
AM demo (am_custom_demo.py) but using particle methods:

1. **Arbitrary geometry** —
   - 2D: an L-shaped polygon (non-convex) via :func:`sdf_from_polygon_2d`
   - 3D: a unit-cube mesh via :func:`sdf_from_mesh`

2. **User-specified laser paths** — custom waypoints (not auto-hatch)

3. **Three particle solvers** — DEM, MPM, SPH — each running a layer-by-layer
   build with activation-gated stiffness/mass and a moving Gaussian laser.

4. **End-to-end differentiability** — ``d(peak_T)/d(laser_power)`` is computed
   via ``jax.grad`` for each method.

Run::

    python examples/particle_am_demo.py
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from diffmech.methods.am import (
    # geometry
    sdf_from_mesh, sdf_from_polygon_2d, sdf_box_2d,
    # scan paths
    from_waypoints, multi_layer_paths,
    # process
    SLMConfig,
    # particle AM
    ParticleAMState, setup_particle_am, place_particles_in_sdf,
    particle_layer_times, laser_heat_on_particles,
    step_dem_am_scan, step_mpm_am_scan, step_sph_am_scan,
    peak_temperature, activated_fraction, mean_displacement,
)
from diffmech.methods.dem import DEMConfig, make_dem_state
from diffmech.methods.mpm import MPMConfig, make_mpm_state
from diffmech.methods.sph import SPHConfig, make_sph_state


OUT_DIR = Path(__file__).resolve().parent.parent / "particle_am_out"


# ===========================================================================
# 2D arbitrary geometry: L-shaped polygon (non-convex)
# ===========================================================================
def l_shape_sdf_2d():
    """L-shaped polygon SDF (arbitrary non-convex 2D geometry)."""
    verts = np.array([
        [0.0, 0.0], [0.4, 0.0], [0.4, 0.2], [0.2, 0.2],
        [0.2, 0.4], [0.0, 0.4],
    ], dtype=np.float64)
    return sdf_from_polygon_2d(verts), verts


def custom_path_2d(layer_zs):
    """User-specified laser paths: zigzag along x at each layer."""
    paths = []
    for z in layer_zs:
        # Custom waypoints: sweep left→right, then right→left.
        wp = np.array([
            [0.02, 0.0], [0.38, 0.0],
        ], dtype=np.float64)
        paths.append(from_waypoints(wp, layer_z=float(z)))
    return multi_layer_paths(paths)


# ===========================================================================
# 3D arbitrary geometry: cube mesh
# ===========================================================================
def cube_mesh_sdf():
    """Unit-cube mesh → SDF (arbitrary 3D geometry from a triangle mesh)."""
    verts = np.array([
        [0, 0, 0], [0.4, 0, 0], [0.4, 0.4, 0], [0, 0.4, 0],
        [0, 0, 0.4], [0.4, 0, 0.4], [0.4, 0.4, 0.4], [0, 0.4, 0.4],
    ], dtype=np.float64)
    tris = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
        [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3],
    ], dtype=np.int64)
    return sdf_from_mesh(verts, tris), verts, tris


def custom_path_3d(layer_zs):
    """User-specified 3D laser paths: diagonal sweep at each layer."""
    paths = []
    for z in layer_zs:
        wp = np.array([
            [0.05, 0.05], [0.35, 0.35],
        ], dtype=np.float64)
        paths.append(from_waypoints(wp, layer_z=float(z)))
    return multi_layer_paths(paths)


# ===========================================================================
# DEM-AM build (2D)
# ===========================================================================
def run_dem_am_2d(laser_power: jnp.ndarray, *, n_steps: int = 20):
    """DEM-AM: L-shaped polygon + custom path, 2D."""
    sdf, verts = l_shape_sdf_2d()
    bbox = [(0.0, 0.4), (0.0, 0.4)]
    layer_zs = np.array([0.1, 0.3])
    pos, lids = place_particles_in_sdf(
        sdf, bbox, layer_zs, dim=2, spacing=0.04)
    paths = custom_path_2d(layer_zs)
    cfg = SLMConfig(
        laser_power=laser_power, scan_speed=0.05,
        layer_thickness=0.2, beam_radius=0.05, absorption=0.5,
        preheat_temp=373.0,
    )
    problem = setup_particle_am(pos, lids, paths, cfg, dim=2,
                                particle_mass=1e-9)
    dem_state = make_dem_state(
        position=jnp.asarray(pos), radius=0.015,
        mass=1e-9, dim=2)
    dem_cfg = DEMConfig(k=5e2, gamma=0.5)
    dem_final, am_final = step_dem_am_scan(
        dem_state, problem.am_state, dt=5e-5, n_steps=n_steps,
        dem_cfg=dem_cfg, cfg=problem.cfg, paths=problem.paths,
        layer_start_times=problem.layer_start_times,
        layer_durations=problem.layer_durations,
        cp=500.0, alpha_T=1e-5, tau=1e-3,
    )
    return {
        "method": "DEM", "dim": 2, "dem": dem_final, "am": am_final,
        "problem": problem, "verts": verts, "pos0": pos,
    }


# ===========================================================================
# MPM-AM build (3D)
# ===========================================================================
def run_mpm_am_3d(laser_power: jnp.ndarray, *, n_steps: int = 10):
    """MPM-AM: cube mesh + custom path, 3D."""
    sdf, verts, tris = cube_mesh_sdf()
    bbox = [(0.0, 0.4), (0.0, 0.4), (0.0, 0.4)]
    layer_zs = np.array([0.1, 0.3])
    pos, lids = place_particles_in_sdf(
        sdf, bbox, layer_zs, dim=3, spacing=0.08)
    paths = custom_path_3d(layer_zs)
    cfg = SLMConfig(
        laser_power=laser_power, scan_speed=0.05,
        layer_thickness=0.2, beam_radius=0.06, absorption=0.5,
        preheat_temp=373.0,
    )
    problem = setup_particle_am(pos, lids, paths, cfg, dim=3,
                                particle_mass=1e-7)
    mpm_cfg = MPMConfig(
        grid_origin=(0.0, 0.0, 0.0),
        grid_shape=(6, 6, 6),
        dx=0.08, dt=5e-5,
        youngs_modulus=1e5, poissons_ratio=0.3,
    )
    mpm_state = make_mpm_state(
        position=jnp.asarray(pos), volume=0.08**3,
        mass=1e-7, dim=3)
    mpm_final, am_final = step_mpm_am_scan(
        mpm_state, problem.am_state, dt=5e-5, n_steps=n_steps,
        mpm_cfg=mpm_cfg, cfg=problem.cfg, paths=problem.paths,
        layer_start_times=problem.layer_start_times,
        layer_durations=problem.layer_durations,
        bc="slip", cp=500.0, alpha_T=1e-5, tau=1e-3,
    )
    return {
        "method": "MPM", "dim": 3, "mpm": mpm_final, "am": am_final,
        "problem": problem, "verts": verts, "pos0": pos,
    }


# ===========================================================================
# SPH-AM build (2D)
# ===========================================================================
def run_sph_am_2d(laser_power: jnp.ndarray, *, n_steps: int = 20):
    """SPH-AM: L-shaped polygon + custom path, 2D."""
    sdf, verts = l_shape_sdf_2d()
    bbox = [(0.0, 0.4), (0.0, 0.4)]
    layer_zs = np.array([0.1, 0.3])
    pos, lids = place_particles_in_sdf(
        sdf, bbox, layer_zs, dim=2, spacing=0.04)
    paths = custom_path_2d(layer_zs)
    cfg = SLMConfig(
        laser_power=laser_power, scan_speed=0.05,
        layer_thickness=0.2, beam_radius=0.05, absorption=0.5,
        preheat_temp=373.0,
    )
    problem = setup_particle_am(pos, lids, paths, cfg, dim=2,
                                particle_mass=1e-9)
    sph_cfg = SPHConfig(h=0.06, rho0=1000.0, c0=50.0, nu=0.01)
    sph_state = make_sph_state(
        position=jnp.asarray(pos), mass=1e-9)
    sph_final, am_final = step_sph_am_scan(
        sph_state, problem.am_state, dt=5e-5, n_steps=n_steps,
        sph_cfg=sph_cfg, cfg=problem.cfg, paths=problem.paths,
        layer_start_times=problem.layer_start_times,
        layer_durations=problem.layer_durations,
        cp=500.0, alpha_T=1e-5, tau=1e-3,
    )
    return {
        "method": "SPH", "dim": 2, "sph": sph_final, "am": am_final,
        "problem": problem, "verts": verts, "pos0": pos,
    }


# ===========================================================================
# Export VTU snapshots
# ===========================================================================
def export_results(result: dict, out_dir: Path = OUT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    method = result["method"]
    dim = result["dim"]
    am = result["am"]
    pos0 = result["pos0"]
    pos_final = np.asarray(am.position)
    T = np.asarray(am.temperature)
    act_time = np.asarray(am.activation_time)

    import meshio
    # Particle cloud: final positions + temperature + activation time.
    pm = meshio.Mesh(
        points=pos_final,
        cells=[("vertex", np.arange(len(pos_final))[:, None])],
        point_data={
            "temperature": T,
            "activation_time": act_time,
            "displacement": pos_final - pos0,
        },
    )
    pm.write(str(out_dir / f"{method.lower()}_am_{dim}d.vtu"))

    # Laser path (waypoints as point cloud).
    paths = result["problem"].paths
    all_wp = []
    for p in paths.layers:
        wp = np.asarray(p.waypoints)
        z = float(p.layer_z)
        if dim == 2:
            all_wp.append(np.concatenate([wp[:, :1],
                                          np.full((len(wp), 1), z)], axis=1))
        else:
            all_wp.append(np.concatenate([wp,
                                          np.full((len(wp), 1), z)], axis=1))
    wp_all = np.concatenate(all_wp, axis=0)
    pm_path = meshio.Mesh(
        points=wp_all,
        cells=[("vertex", np.arange(len(wp_all))[:, None])],
    )
    pm_path.write(str(out_dir / f"{method.lower()}_path_{dim}d.vtu"))
    return out_dir


def main():
    print("DiffMech — particle-method AM (DEM / MPM / SPH)")
    print("=" * 70)
    print("Geometry : 2D L-shaped polygon (non-convex) + 3D cube mesh")
    print("Laser   : user-specified waypoints (no auto-hatch)")
    print("Methods : DEM (2D), MPM (3D), SPH (2D)")
    print()

    power = jnp.array(200.0)

    # --- DEM-AM 2D ---
    print("[1] DEM-AM (2D, L-shaped polygon):")
    res_dem = run_dem_am_2d(power, n_steps=20)
    am = res_dem["am"]
    pos0 = res_dem["pos0"]
    print(f"    particles            N    = {am.n_particles}")
    print(f"    peak temperature     T    = {float(peak_temperature(am)):.1f} K")
    print(f"    mean displacement    |u|  = "
          f"{float(mean_displacement(jnp.asarray(pos0), am)):.3e} m")

    # Gradient
    def dem_loss(p):
        r = run_dem_am_2d(p, n_steps=20)
        return peak_temperature(r["am"])
    g_dem = float(jax.grad(dem_loss)(power))
    print(f"    dT_peak/d(power)     = {g_dem:.3e} K/W")

    # --- MPM-AM 3D ---
    print("\n[2] MPM-AM (3D, cube mesh):")
    res_mpm = run_mpm_am_3d(power, n_steps=10)
    am = res_mpm["am"]
    pos0 = res_mpm["pos0"]
    print(f"    particles            N    = {am.n_particles}")
    print(f"    peak temperature     T    = {float(peak_temperature(am)):.1f} K")
    print(f"    mean displacement    |u|  = "
          f"{float(mean_displacement(jnp.asarray(pos0), am)):.3e} m")

    def mpm_loss(p):
        r = run_mpm_am_3d(p, n_steps=10)
        return peak_temperature(r["am"])
    g_mpm = float(jax.grad(mpm_loss)(power))
    print(f"    dT_peak/d(power)     = {g_mpm:.3e} K/W")

    # --- SPH-AM 2D ---
    print("\n[3] SPH-AM (2D, L-shaped polygon):")
    res_sph = run_sph_am_2d(power, n_steps=20)
    am = res_sph["am"]
    pos0 = res_sph["pos0"]
    print(f"    particles            N    = {am.n_particles}")
    print(f"    peak temperature     T    = {float(peak_temperature(am)):.1f} K")
    print(f"    mean displacement    |u|  = "
          f"{float(mean_displacement(jnp.asarray(pos0), am)):.3e} m")

    def sph_loss(p):
        r = run_sph_am_2d(p, n_steps=20)
        return peak_temperature(r["am"])
    g_sph = float(jax.grad(sph_loss)(power))
    print(f"    dT_peak/d(power)     = {g_sph:.3e} K/W")

    # --- Export ---
    print("\n[4] Exporting VTU snapshots to particle_am_out/ ...")
    for r in [res_dem, res_mpm, res_sph]:
        out = export_results(r)
    print(f"    written to {out}")
    print("\nDone. Open the .vtu files in ParaView to inspect the builds.")


if __name__ == "__main__":
    main()
