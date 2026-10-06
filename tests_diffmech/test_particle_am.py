"""Tests for particle-method AM (DEM / MPM / SPH) with arbitrary geometry
and user-specified laser paths, in 2D and 3D.

Covers:
- 2D SDF primitives and polygon SDF
- Particle placement from SDF (2D & 3D)
- Per-particle activation times and activation field
- Laser heat source on particles
- AM-aware DEM / MPM / SPH steps (correctness + end-to-end differentiability)
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from diffmech.methods.am import (
    # 2D SDFs
    sdf_circle_2d, sdf_box_2d, sdf_from_polygon_2d,
    # 3D SDFs (reuse)
    sdf_box, sdf_sphere, sdf_from_mesh,
    # scan paths
    from_waypoints, chain_paths, multi_layer_paths, zigzag_hatch,
    # process
    SLMConfig,
    # particle AM
    ParticleAMState, setup_particle_am, place_particles_in_sdf,
    assign_particle_activation_times, particle_layer_times,
    laser_heat_on_particles,
    step_dem_am, step_dem_am_scan,
    step_mpm_am, step_mpm_am_scan,
    step_sph_am, step_sph_am_scan,
    peak_temperature, activated_fraction, mean_displacement,
)
from diffmech.methods.dem import DEMState, DEMConfig, make_dem_state
from diffmech.methods.mpm import MPMState, MPMConfig, make_mpm_state
from diffmech.methods.sph import SPHState, SPHConfig, make_sph_state


# ===========================================================================
# 2D SDF primitives
# ===========================================================================
class TestSDF2D:
    def test_circle_2d_inside_outside(self):
        sdf = sdf_circle_2d((0, 0), 1.0)
        assert float(sdf(jnp.array([0.0, 0.0]))) < 0.0
        assert float(sdf(jnp.array([2.0, 0.0]))) > 0.0

    def test_box_2d_inside_outside(self):
        sdf = sdf_box_2d((0, 0), (1, 1))
        assert float(sdf(jnp.array([0.0, 0.0]))) < 0.0
        assert float(sdf(jnp.array([2.0, 0.0]))) > 0.0

    def test_polygon_2d_inside_outside(self):
        # Square polygon from (−1,−1) to (1,1).
        verts = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]],
                         dtype=np.float64)
        sdf = sdf_from_polygon_2d(verts)
        assert float(sdf(jnp.array([0.0, 0.0]))) < 0.0   # inside
        assert float(sdf(jnp.array([2.0, 0.0]))) > 0.0   # outside

    def test_polygon_2d_arbitrary_shape(self):
        # Triangle — arbitrary non-convex test geometry.
        verts = np.array([[0, 0], [2, 0], [1, 2]], dtype=np.float64)
        sdf = sdf_from_polygon_2d(verts)
        assert float(sdf(jnp.array([1.0, 0.5]))) < 0.0   # inside
        assert float(sdf(jnp.array([3.0, 3.0]))) > 0.0   # outside

    def test_sdf_2d_differentiable(self):
        sdf = sdf_circle_2d((0, 0), 1.0)
        g = jax.grad(lambda x: sdf(jnp.array([x, 0.0])).sum())(2.0)
        assert abs(float(g) - 1.0) < 1e-6


# ===========================================================================
# Particle placement from SDF
# ===========================================================================
class TestParticlePlacement:
    def test_place_particles_2d(self):
        sdf = sdf_box_2d((0, 0.5), (0.5, 0.5))
        bbox = [(-1, 1), (0, 1)]
        layer_zs = np.array([0.25, 0.75])
        pos, lids = place_particles_in_sdf(
            sdf, bbox, layer_zs, dim=2, spacing=0.2)
        assert pos.shape[1] == 2
        assert len(pos) > 0
        assert len(lids) == len(pos)
        # All particles should be inside the SDF.
        s = np.asarray(sdf(jnp.asarray(pos)))
        assert np.all(s < 0.0)

    def test_place_particles_3d(self):
        sdf = sdf_box((0, 0, 0.5), (0.5, 0.5, 0.5))
        bbox = [(-1, 1), (-1, 1), (0, 1)]
        layer_zs = np.array([0.5])
        pos, lids = place_particles_in_sdf(
            sdf, bbox, layer_zs, dim=3, spacing=0.3)
        assert pos.shape[1] == 3
        assert len(pos) > 0
        s = np.asarray(sdf(jnp.asarray(pos)))
        assert np.all(s < 0.0)

    def test_layer_ids_match_z(self):
        sdf = sdf_box_2d((0, 0.5), (1, 0.5))
        bbox = [(-1, 1), (0, 1)]
        layer_zs = np.array([0.25, 0.75])
        pos, lids = place_particles_in_sdf(
            sdf, bbox, layer_zs, dim=2, spacing=0.2)
        # Layer 0 particles should have z near 0.25, layer 1 near 0.75.
        l0 = pos[lids == 0]
        l1 = pos[lids == 1]
        if len(l0) > 0:
            assert np.allclose(l0[:, 1], 0.25)
        if len(l1) > 0:
            assert np.allclose(l1[:, 1], 0.75)


# ===========================================================================
# Activation field
# ===========================================================================
class TestActivation:
    def test_activation_field_sigmoid(self):
        am_state = ParticleAMState(
            position=jnp.zeros((3, 2)),
            activation_time=jnp.array([0.0, 1.0, 2.0]),
            temperature=jnp.full(3, 300.0),
            mass=jnp.ones(3),
            layer_id=jnp.array([0, 1, 2]),
            rho=jnp.ones(3) * 7850.0,
        )
        # At t=0, particle 0 should be ~0.5, particles 1&2 ≈ 0.
        alpha = am_state.activation_field(0.0, tau=0.1)
        assert float(alpha[0]) > float(alpha[1])
        assert float(alpha[1]) > float(alpha[2])

    def test_activation_saturates(self):
        am_state = ParticleAMState(
            position=jnp.zeros((1, 2)),
            activation_time=jnp.array([0.0]),
            temperature=jnp.array([300.0]),
            mass=jnp.array([1.0]),
            layer_id=jnp.array([0]),
            rho=jnp.array([7850.0]),
        )
        assert float(am_state.activation_field(1e6, tau=1e-3)[0]) > 0.99
        assert float(am_state.activation_field(-1e6, tau=1e-3)[0]) < 0.01


# ===========================================================================
# Laser heat source
# ===========================================================================
class TestLaserHeat:
    def _make_paths_2d(self, layer_zs):
        paths = []
        for z in layer_zs:
            wp = np.array([[0.0, 0.0], [1.0, 0.0]])
            paths.append(from_waypoints(wp, layer_z=float(z)))
        return multi_layer_paths(paths)

    def test_heat_source_peaks_at_laser(self):
        layer_zs = np.array([0.5])
        paths = self._make_paths_2d(layer_zs)
        cfg = SLMConfig(scan_speed=1.0, beam_radius=0.1,
                        layer_thickness=0.1, laser_power=200.0)
        starts, durations = particle_layer_times(paths, cfg)
        # At t=0, laser is at (0, 0) — particle at origin should get max heat.
        positions = jnp.array([[0.0, 0.5],   # at laser
                               [1.0, 0.5]])   # far from laser
        q = laser_heat_on_particles(positions, 0.0, paths, starts,
                                    durations, cfg, dim=2)
        assert float(q[0]) > float(q[1])
        assert float(q[0]) > 0.0

    def test_heat_source_differentiable_wrt_power(self):
        layer_zs = np.array([0.5])
        paths = self._make_paths_2d(layer_zs)
        positions = jnp.array([[0.0, 0.5]])

        def heat(power):
            cfg = SLMConfig(scan_speed=1.0, beam_radius=0.1,
                           layer_thickness=0.1, laser_power=power)
            starts, durations = particle_layer_times(paths, cfg)
            return laser_heat_on_particles(positions, 0.0, paths,
                                           starts, durations, cfg,
                                           dim=2).sum()
        g = jax.grad(heat)(200.0)
        assert float(g) > 0.0


# ===========================================================================
# Setup
# ===========================================================================
class TestSetup:
    def test_setup_particle_am_2d(self):
        sdf = sdf_box_2d((0, 0.5), (0.5, 0.5))
        bbox = [(-1, 1), (0, 1)]
        layer_zs = np.array([0.25, 0.75])
        pos, lids = place_particles_in_sdf(
            sdf, bbox, layer_zs, dim=2, spacing=0.2)
        paths = multi_layer_paths([
            from_waypoints(np.array([[0.0, 0.0], [0.5, 0.0]]),
                          layer_z=0.25),
            from_waypoints(np.array([[0.0, 0.0], [0.5, 0.0]]),
                          layer_z=0.75),
        ])
        cfg = SLMConfig(scan_speed=1.0, layer_thickness=0.2,
                        beam_radius=0.1)
        problem = setup_particle_am(pos, lids, paths, cfg, dim=2)
        assert problem.dim == 2
        assert problem.am_state.n_particles == len(pos)
        assert float(problem.am_state.temperature[0]) == pytest.approx(373.0)


# ===========================================================================
# DEM-AM
# ===========================================================================
class TestDEMAM:
    def _make_problem_2d(self, laser_power=200.0):
        sdf = sdf_box_2d((0, 0.5), (0.5, 0.5))
        bbox = [(-0.6, 0.6), (0, 1)]
        layer_zs = np.array([0.25, 0.75])
        pos, lids = place_particles_in_sdf(
            sdf, bbox, layer_zs, dim=2, spacing=0.15)
        paths = multi_layer_paths([
            from_waypoints(np.array([[-0.3, 0.0], [0.3, 0.0]]),
                          layer_z=0.25),
            from_waypoints(np.array([[-0.3, 0.0], [0.3, 0.0]]),
                          layer_z=0.75),
        ])
        cfg = SLMConfig(scan_speed=0.5, layer_thickness=0.2,
                        beam_radius=0.1, laser_power=laser_power)
        problem = setup_particle_am(pos, lids, paths, cfg, dim=2,
                                    particle_mass=1e-6)
        dem_state = make_dem_state(position=jnp.asarray(pos),
                                   radius=0.05, mass=1e-6, dim=2)
        return dem_state, problem

    def test_step_dem_am_finite(self):
        dem_state, problem = self._make_problem_2d()
        dem_cfg = DEMConfig(k=1e3, gamma=1.0)
        dem_new, am_new = step_dem_am(
            dem_state, problem.am_state, dt=1e-4, t=0.0,
            dem_cfg=dem_cfg, cfg=problem.cfg, paths=problem.paths,
            layer_start_times=problem.layer_start_times,
            layer_durations=problem.layer_durations,
        )
        assert jnp.all(jnp.isfinite(dem_new.position))
        assert jnp.all(jnp.isfinite(am_new.temperature))

    def test_dem_am_temperature_increases(self):
        dem_state, problem = self._make_problem_2d(laser_power=500.0)
        dem_cfg = DEMConfig(k=1e3, gamma=1.0)
        T0 = float(problem.am_state.temperature[0])
        _, am_new = step_dem_am(
            dem_state, problem.am_state, dt=1e-3, t=0.0,
            dem_cfg=dem_cfg, cfg=problem.cfg, paths=problem.paths,
            layer_start_times=problem.layer_start_times,
            layer_durations=problem.layer_durations,
        )
        assert float(am_new.temperature[0]) > T0

    def test_dem_am_scan_differentiable(self):
        dem_state, problem = self._make_problem_2d()
        dem_cfg = DEMConfig(k=1e3, gamma=1.0)
        starts = problem.layer_start_times
        durations = problem.layer_durations

        def loss(power):
            cfg = SLMConfig(scan_speed=0.5, layer_thickness=0.2,
                           beam_radius=0.1, laser_power=power)
            _, am_final = step_dem_am_scan(
                dem_state, problem.am_state, dt=1e-4, n_steps=5,
                dem_cfg=dem_cfg, cfg=cfg, paths=problem.paths,
                layer_start_times=starts, layer_durations=durations,
            )
            return peak_temperature(am_final)

        g = jax.grad(loss)(200.0)
        assert jnp.isfinite(g)
        assert float(g) > 0.0


# ===========================================================================
# MPM-AM
# ===========================================================================
class TestMPMAM:
    def _make_problem_3d(self, laser_power=200.0):
        sdf = sdf_box((0, 0, 0.5), (0.5, 0.5, 0.5))
        bbox = [(-0.6, 0.6), (-0.6, 0.6), (0, 1)]
        layer_zs = np.array([0.5])
        pos, lids = place_particles_in_sdf(
            sdf, bbox, layer_zs, dim=3, spacing=0.2)
        paths = multi_layer_paths([
            from_waypoints(np.array([[-0.3, 0.0], [0.3, 0.0]]),
                          layer_z=0.5),
        ])
        cfg = SLMConfig(scan_speed=0.5, layer_thickness=0.3,
                        beam_radius=0.1, laser_power=laser_power)
        problem = setup_particle_am(pos, lids, paths, cfg, dim=3,
                                    particle_mass=1e-6)
        mpm_cfg = MPMConfig(
            grid_origin=(-0.6, -0.6, 0.0),
            grid_shape=(7, 7, 6),
            dx=0.2, dt=1e-4,
            youngs_modulus=1e5, poissons_ratio=0.3,
        )
        mpm_state = make_mpm_state(position=jnp.asarray(pos),
                                   volume=0.2**3, mass=1e-6, dim=3)
        return mpm_state, mpm_cfg, problem

    def test_step_mpm_am_finite(self):
        mpm_state, mpm_cfg, problem = self._make_problem_3d()
        mpm_new, am_new = step_mpm_am(
            mpm_state, problem.am_state, dt=1e-4, t=0.0,
            mpm_cfg=mpm_cfg, cfg=problem.cfg, paths=problem.paths,
            layer_start_times=problem.layer_start_times,
            layer_durations=problem.layer_durations,
        )
        assert jnp.all(jnp.isfinite(mpm_new.position))
        assert jnp.all(jnp.isfinite(am_new.temperature))

    def test_mpm_am_temperature_increases(self):
        mpm_state, mpm_cfg, problem = self._make_problem_3d(laser_power=1000.0)
        T0 = float(problem.am_state.temperature[0])
        _, am_new = step_mpm_am(
            mpm_state, problem.am_state, dt=1e-3, t=0.0,
            mpm_cfg=mpm_cfg, cfg=problem.cfg, paths=problem.paths,
            layer_start_times=problem.layer_start_times,
            layer_durations=problem.layer_durations,
        )
        # 修正热学单位后，体积热流 [W/m³] 经 ρ·c_p 转换为微小但为正的温升；
        # 检查"被光束照射的激活粒子"峰值温度上升（不再是 q≈0 的粒子[0]）。
        assert float(peak_temperature(am_new)) > T0

    def test_mpm_am_scan_differentiable(self):
        mpm_state, mpm_cfg, problem = self._make_problem_3d()
        starts = problem.layer_start_times
        durations = problem.layer_durations

        def loss(power):
            cfg = SLMConfig(scan_speed=0.5, layer_thickness=0.3,
                           beam_radius=0.1, laser_power=power)
            _, am_final = step_mpm_am_scan(
                mpm_state, problem.am_state, dt=1e-4, n_steps=3,
                mpm_cfg=mpm_cfg, cfg=cfg, paths=problem.paths,
                layer_start_times=starts, layer_durations=durations,
            )
            return peak_temperature(am_final)

        g = jax.grad(loss)(200.0)
        assert jnp.isfinite(g)
        assert float(g) > 0.0


# ===========================================================================
# SPH-AM
# ===========================================================================
class TestSPHAM:
    def _make_problem_2d(self, laser_power=200.0):
        sdf = sdf_box_2d((0, 0.5), (0.5, 0.5))
        bbox = [(-0.6, 0.6), (0, 1)]
        layer_zs = np.array([0.25, 0.75])
        pos, lids = place_particles_in_sdf(
            sdf, bbox, layer_zs, dim=2, spacing=0.15)
        paths = multi_layer_paths([
            from_waypoints(np.array([[-0.3, 0.0], [0.3, 0.0]]),
                          layer_z=0.25),
            from_waypoints(np.array([[-0.3, 0.0], [0.3, 0.0]]),
                          layer_z=0.75),
        ])
        cfg = SLMConfig(scan_speed=0.5, layer_thickness=0.2,
                        beam_radius=0.1, laser_power=laser_power)
        problem = setup_particle_am(pos, lids, paths, cfg, dim=2,
                                    particle_mass=1e-6)
        sph_cfg = SPHConfig(h=0.2, rho0=1000.0, c0=50.0, nu=0.01)
        sph_state = make_sph_state(position=jnp.asarray(pos),
                                   mass=1e-6)
        return sph_state, sph_cfg, problem

    def test_step_sph_am_finite(self):
        sph_state, sph_cfg, problem = self._make_problem_2d()
        sph_new, am_new = step_sph_am(
            sph_state, problem.am_state, dt=1e-4, t=0.0,
            sph_cfg=sph_cfg, cfg=problem.cfg, paths=problem.paths,
            layer_start_times=problem.layer_start_times,
            layer_durations=problem.layer_durations,
        )
        assert jnp.all(jnp.isfinite(sph_new.position))
        assert jnp.all(jnp.isfinite(am_new.temperature))

    def test_sph_am_temperature_increases(self):
        sph_state, sph_cfg, problem = self._make_problem_2d(laser_power=1000.0)
        T0 = float(problem.am_state.temperature[0])
        _, am_new = step_sph_am(
            sph_state, problem.am_state, dt=1e-3, t=0.0,
            sph_cfg=sph_cfg, cfg=problem.cfg, paths=problem.paths,
            layer_start_times=problem.layer_start_times,
            layer_durations=problem.layer_durations,
        )
        assert float(am_new.temperature[0]) > T0

    def test_sph_am_scan_differentiable(self):
        sph_state, sph_cfg, problem = self._make_problem_2d()
        starts = problem.layer_start_times
        durations = problem.layer_durations

        def loss(power):
            cfg = SLMConfig(scan_speed=0.5, layer_thickness=0.2,
                           beam_radius=0.1, laser_power=power)
            _, am_final = step_sph_am_scan(
                sph_state, problem.am_state, dt=1e-4, n_steps=5,
                sph_cfg=sph_cfg, cfg=cfg, paths=problem.paths,
                layer_start_times=starts, layer_durations=durations,
            )
            return peak_temperature(am_final)

        g = jax.grad(loss)(200.0)
        assert jnp.isfinite(g)
        assert float(g) > 0.0


# ===========================================================================
# Arbitrary geometry from mesh (3D) + custom path → particle AM
# ===========================================================================
class TestArbitraryGeometryParticleAM:
    def test_mesh_sdf_into_particle_am_3d(self):
        """End-to-end: mesh → SDF → particle placement → DEM-AM step."""
        # Simple box mesh (unit cube).
        verts = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], dtype=np.float64)
        tris = np.array([
            [0, 2, 1], [0, 3, 2],  # bottom
            [4, 5, 6], [4, 6, 7],  # top
            [0, 1, 5], [0, 5, 4],  # front
            [2, 3, 7], [2, 7, 6],  # back
            [1, 2, 6], [1, 6, 5],  # right
            [0, 4, 7], [0, 7, 3],  # left
        ], dtype=np.int64)
        sdf = sdf_from_mesh(verts, tris)
        bbox = [(0, 1), (0, 1), (0, 1)]
        layer_zs = np.array([0.5])
        pos, lids = place_particles_in_sdf(
            sdf, bbox, layer_zs, dim=3, spacing=0.3)
        assert len(pos) > 0
        assert pos.shape[1] == 3

        paths = multi_layer_paths([
            from_waypoints(np.array([[0.2, 0.2], [0.8, 0.8]]),
                          layer_z=0.5),
        ])
        cfg = SLMConfig(scan_speed=0.5, layer_thickness=0.5,
                        beam_radius=0.2, laser_power=200.0)
        problem = setup_particle_am(pos, lids, paths, cfg, dim=3,
                                    particle_mass=1e-3)
        dem_state = make_dem_state(position=jnp.asarray(pos),
                                   radius=0.1, mass=1e-3, dim=3)
        dem_cfg = DEMConfig(k=1e3, gamma=1.0)
        dem_new, am_new = step_dem_am(
            dem_state, problem.am_state, dt=1e-4, t=0.0,
            dem_cfg=dem_cfg, cfg=problem.cfg, paths=problem.paths,
            layer_start_times=problem.layer_start_times,
            layer_durations=problem.layer_durations,
        )
        assert jnp.all(jnp.isfinite(dem_new.position))
        assert jnp.all(jnp.isfinite(am_new.temperature))

    def test_polygon_sdf_into_particle_am_2d(self):
        """End-to-end: 2D polygon → SDF → particle placement → SPH-AM step."""
        # L-shaped polygon (arbitrary 2D geometry).
        verts = np.array([
            [0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2],
        ], dtype=np.float64)
        sdf = sdf_from_polygon_2d(verts)
        bbox = [(0, 2), (0, 2)]
        layer_zs = np.array([0.5, 1.5])
        pos, lids = place_particles_in_sdf(
            sdf, bbox, layer_zs, dim=2, spacing=0.3)
        assert len(pos) > 0
        assert pos.shape[1] == 2

        paths = multi_layer_paths([
            from_waypoints(np.array([[0.1, 0.0], [1.9, 0.0]]),
                          layer_z=0.5),
            from_waypoints(np.array([[0.1, 0.0], [0.9, 0.0]]),
                          layer_z=1.5),
        ])
        cfg = SLMConfig(scan_speed=0.5, layer_thickness=0.5,
                        beam_radius=0.2, laser_power=200.0)
        problem = setup_particle_am(pos, lids, paths, cfg, dim=2,
                                    particle_mass=1e-4)
        sph_cfg = SPHConfig(h=0.3, rho0=1000.0, c0=50.0, nu=0.01)
        sph_state = make_sph_state(position=jnp.asarray(pos),
                                   mass=1e-4)
        sph_new, am_new = step_sph_am(
            sph_state, problem.am_state, dt=1e-4, t=0.0,
            sph_cfg=sph_cfg, cfg=problem.cfg, paths=problem.paths,
            layer_start_times=problem.layer_start_times,
            layer_durations=problem.layer_durations,
        )
        assert jnp.all(jnp.isfinite(sph_new.position))
        assert jnp.all(jnp.isfinite(am_new.temperature))
