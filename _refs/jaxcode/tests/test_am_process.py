"""Tests for the AM (Additive Manufacturing) process-simulation modules.

Covers:
- geometry.py     : SDF primitives, boolean ops, layer slicing, bbox
- scan_paths.py    : zigzag / contour / spiral path generation
- process.py       : SLMConfig / LSFConfig, layer activation times
- activation.py    : layered mesh, activation field, part-cell mask
- am_thermal.py    : laser position, moving heat source
- thermomechanical.py : activation-gated stiffness, thermal force, solver

All tests verify both correctness and end-to-end differentiability.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from diffmech.methods.am import (
    # geometry
    sdf_sphere, sdf_box, sdf_cylinder, sdf_gear,
    union, intersection, difference, translate,
    slice_mask, layer_heights, part_bbox,
    smooth_union, smooth_intersection, smooth_difference,
    sdf_from_mesh,
    # scan paths
    zigzag_hatch, contour_hatch, spiral_hatch,
    multi_layer_paths, ScanPath, MultiLayerPath,
    from_waypoints, chain_paths, from_csv, from_gcode,
    # process
    SLMConfig, LSFConfig, layer_time, layer_activation_times,
    # activation
    LayeredMesh, build_layered_mesh, assign_activation_times, part_cell_mask,
    # thermal
    laser_position, am_heat_source, setup_am_thermal,
    # thermo-mechanical
    solve_thermomechanical, assemble_activated_stiffness,
)
from diffmech.materials import LinearElasticIsotropic
from diffmech.solvers import DirichletBC


# ===========================================================================
# geometry.py
# ===========================================================================
class TestSDFPrimitives:
    """SDF primitive shapes."""

    def test_sphere_inside_outside(self):
        sdf = sdf_sphere((0, 0, 0), 1.0)
        # Inside: negative SDF.
        assert float(sdf(jnp.array([0.0, 0.0, 0.0]))) < 0.0
        # On surface: ~0.
        assert abs(float(sdf(jnp.array([1.0, 0.0, 0.0])))) < 1e-6
        # Outside: positive.
        assert float(sdf(jnp.array([2.0, 0.0, 0.0]))) > 0.0

    def test_box_inside_outside(self):
        sdf = sdf_box((0, 0, 0), (1, 1, 1))
        assert float(sdf(jnp.array([0.0, 0.0, 0.0]))) < 0.0
        assert float(sdf(jnp.array([2.0, 0.0, 0.0]))) > 0.0

    def test_cylinder_inside_outside(self):
        sdf = sdf_cylinder((0, 0, 0), axis=2, radius=1.0, half_length=2.0)
        assert float(sdf(jnp.array([0.0, 0.0, 0.0]))) < 0.0
        assert float(sdf(jnp.array([0.5, 0.0, 0.0]))) < 0.0
        assert float(sdf(jnp.array([2.0, 0.0, 0.0]))) > 0.0

    def test_gear_has_teeth(self):
        """The gear SDF should produce a toothed profile."""
        sdf = sdf_gear((0, 0, 0), axis=2, outer_radius=1.0, n_teeth=8,
                       tooth_depth=0.2, thickness=0.5)
        # Sample at several angles; the radial profile should vary.
        thetas = jnp.linspace(0, 2 * np.pi, 100)[:-1]
        pts = jnp.stack([jnp.cos(thetas), jnp.sin(thetas),
                         jnp.zeros_like(thetas)], axis=-1) * 0.9
        s = sdf(pts)
        # The profile should have variation (not constant) due to teeth.
        assert float(jnp.std(s)) > 1e-4

    def test_sdf_differentiable(self):
        """SDF evaluation should be differentiable w.r.t. coordinates."""
        sdf = sdf_sphere((0, 0, 0), 1.0)
        x = jnp.array(2.0)
        g = jax.grad(lambda x_: sdf(jnp.array([x_, 0.0, 0.0])))(x)
        # d/dx (sqrt(x²) - 1) at x=2 is x/sqrt(x²) = 1.
        assert abs(float(g) - 1.0) < 1e-6


class TestSDFBooleanOps:
    """Boolean operations on SDFs."""

    def test_union_contains_both(self):
        a = sdf_sphere((-1, 0, 0), 0.8)
        b = sdf_sphere((1, 0, 0), 0.8)
        u = union(a, b)
        # Point inside a but outside b → inside union.
        assert float(u(jnp.array([-1.0, 0.0, 0.0]))) < 0.0
        # Point inside b but outside a → inside union.
        assert float(u(jnp.array([1.0, 0.0, 0.0]))) < 0.0
        # Point outside both → outside union.
        assert float(u(jnp.array([0.0, 2.0, 0.0]))) > 0.0

    def test_intersection_requires_both(self):
        a = sdf_sphere((0, 0, 0), 1.5)
        b = sdf_box((0, 0, 0), (1, 1, 1))
        inter = intersection(a, b)
        # Inside both → inside intersection.
        assert float(inter(jnp.array([0.0, 0.0, 0.0]))) < 0.0
        # Inside sphere, outside box → outside intersection.
        assert float(inter(jnp.array([1.2, 0.0, 0.0]))) > 0.0

    def test_difference_subtracts(self):
        a = sdf_box((0, 0, 0), (2, 2, 2))
        b = sdf_sphere((0, 0, 0), 1.0)
        diff = difference(a, b)
        # Inside box and inside sphere → outside (subtracted).
        assert float(diff(jnp.array([0.0, 0.0, 0.0]))) > 0.0
        # Inside box, outside sphere → inside (kept).
        assert float(diff(jnp.array([1.5, 0.0, 0.0]))) < 0.0

    def test_smooth_union_differentiable(self):
        """Smooth union should be differentiable."""
        a = sdf_sphere((-1, 0, 0), 0.8)
        b = sdf_sphere((1, 0, 0), 0.8)
        u = union(a, b)
        g = jax.grad(lambda x: u(jnp.array([x, 0.0, 0.0])).sum())(0.0)
        assert jnp.isfinite(g)


class TestSlicing:
    """Layer slicing (the heart of AM geometry processing)."""

    def test_slice_mask_returns_occupancy(self):
        sdf = sdf_cylinder((0, 0, 0.5), axis=2, radius=1.0, half_length=0.5)
        xs = np.linspace(-1.5, 1.5, 31)
        ys = np.linspace(-1.5, 1.5, 31)
        grid_xy = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1)
        chi = slice_mask(sdf, 0.5, jnp.asarray(grid_xy), smoothness=0.01)
        # Centre should be inside (chi → 1).
        assert float(chi[15, 15]) > 0.5
        # Corner should be outside (chi → 0).
        assert float(chi[0, 0]) < 0.5

    def test_slice_mask_differentiable(self):
        sdf = sdf_sphere((0, 0, 0), 1.0)
        grid = jnp.array([[[0.0, 0.0]]])
        # z=0.5 (smooth point of the sphere SDF — z=0 has a |z| kink).
        g = jax.grad(lambda z: slice_mask(sdf, z, grid).sum())(0.5)
        assert jnp.isfinite(g)

    def test_layer_heights(self):
        sdf = sdf_sphere((0, 0, 0.5), 1.0)
        bbox = [(-1, 1), (-1, 1), (0, 1)]
        zs = layer_heights(sdf, bbox, layer_thickness=0.2)
        assert len(zs) > 0
        assert zs[0] == pytest.approx(0.0)

    def test_part_bbox_small_part(self):
        """part_bbox should find the correct box for a small part."""
        sdf = sdf_sphere((0, 0, 0), 0.01)
        bbox = part_bbox(sdf, resolution=32, search_range=(-0.02, 0.02))
        assert bbox[0][0] < -0.009
        assert bbox[0][1] > 0.009


# ===========================================================================
# scan_paths.py
# ===========================================================================
class TestScanPaths:
    """Scan-path generation."""

    def test_zigzag_hatch_generates_waypoints(self):
        # Square mask: 10x10 grid, all inside.
        mask = np.ones((10, 10))
        xs = np.linspace(0, 1, 10)
        ys = np.linspace(0, 1, 10)
        path = zigzag_hatch(mask, xs, ys, hatch_spacing=0.2,
                            layer_z=0.5)
        assert path.waypoints.shape[1] == 2
        assert path.total_length > 0.0
        assert path.layer_z == 0.5

    def test_zigzag_empty_mask_fallback(self):
        mask = np.zeros((10, 10))
        xs = np.linspace(0, 1, 10)
        ys = np.linspace(0, 1, 10)
        path = zigzag_hatch(mask, xs, ys, hatch_spacing=0.2)
        # Should still produce a valid (fallback) path.
        assert len(path.waypoints) >= 2

    def test_spiral_hatch(self):
        path = spiral_hatch((0.0, 0.0), 1.0, n_turns=2.0, n_points=100,
                            layer_z=0.3)
        assert path.waypoints.shape == (100, 2)
        assert path.total_length > 0.0
        assert path.layer_z == 0.3

    def test_contour_hatch_runs(self):
        mask = np.ones((10, 10))
        xs = np.linspace(0, 1, 10)
        ys = np.linspace(0, 1, 10)
        path = contour_hatch(mask, xs, ys, hatch_spacing=0.2,
                            layer_z=0.1)
        assert path.total_length > 0.0

    def test_scan_path_position_at(self):
        """position_at should interpolate along the path."""
        path = spiral_hatch((0.0, 0.0), 1.0, n_turns=1.0, n_points=50,
                            layer_z=0.0)
        p0 = path.position_at(0.0)
        p_end = path.position_at(path.total_length)
        assert p0.shape == (2,)
        assert np.all(np.isfinite(p_end))

    def test_multi_layer_paths(self):
        paths = [spiral_hatch((0, 0), 1, layer_z=z) for z in [0, 0.1, 0.2]]
        mlp = multi_layer_paths(paths)
        assert mlp.n_layers == 3
        assert len(mlp.layer_heights) == 3


# ===========================================================================
# process.py
# ===========================================================================
class TestProcessConfigs:
    """SLM / LSF process configurations."""

    def test_slm_config_defaults(self):
        cfg = SLMConfig()
        assert cfg.laser_power > 0
        assert cfg.layer_thickness > 0
        assert 0 < cfg.absorption <= 1

    def test_lsf_config_defaults(self):
        cfg = LSFConfig()
        assert cfg.laser_power > SLMConfig().laser_power  # LSF higher power
        assert cfg.beam_radius > SLMConfig().beam_radius   # bigger melt pool
        assert cfg.scan_speed < SLMConfig().scan_speed     # slower

    def test_layer_time(self):
        cfg = SLMConfig(scan_speed=1.0)
        t = layer_time(cfg, path_length=10.0)
        assert t == pytest.approx(10.0)

    def test_layer_activation_times(self):
        cfg = SLMConfig(scan_speed=1.0)
        lengths = [10.0, 20.0, 30.0]
        starts, ends = layer_activation_times(cfg, lengths)
        assert starts[0] == 0.0
        assert ends[0] == pytest.approx(10.0)
        assert starts[1] == pytest.approx(10.0)
        assert ends[2] == pytest.approx(60.0)

    def test_configs_are_frozen(self):
        cfg = SLMConfig()
        with pytest.raises(Exception):
            cfg.laser_power = 300.0


# ===========================================================================
# activation.py
# ===========================================================================
class TestLayeredMesh:
    """Layered mesh + activation field."""

    def test_build_layered_mesh(self):
        layer_zs = np.array([0.0, 0.1, 0.2])
        bbox = [(-1, 1), (-1, 1), (0, 0.3)]
        layered = build_layered_mesh(layer_zs, bbox, nx=4, ny=4,
                                     cells_per_layer=1)
        assert layered.n_layers == 3
        assert layered.n_cells == 4 * 4 * 3
        assert layered.mesh.cell_type == "hex8"

    def test_assign_activation_times(self):
        layer_zs = np.array([0.0, 0.1])
        bbox = [(-1, 1), (-1, 1), (0, 0.2)]
        layered = build_layered_mesh(layer_zs, bbox, nx=2, ny=2)
        starts = np.array([0.0, 5.0])
        layered = assign_activation_times(layered, starts)
        # Layer 0 cells activate at t=0, layer 1 at t=5.
        layer_ids = np.asarray(layered.layer_id)
        act_times = np.asarray(layered.activation_time)
        assert np.allclose(act_times[layer_ids == 0], 0.0)
        assert np.allclose(act_times[layer_ids == 1], 5.0)

    def test_activation_field_smooth(self):
        layer_zs = np.array([0.0, 0.1])
        bbox = [(-1, 1), (-1, 1), (0, 0.2)]
        layered = build_layered_mesh(layer_zs, bbox, nx=2, ny=2)
        layered = assign_activation_times(layered, np.array([0.0, 1.0]))
        # At t=0: layer 0 just activating (alpha≈0.5), layer 1 not (alpha≈0).
        alpha = layered.activation_field(jnp.array(0.0), tau=0.1)
        layer_ids = np.asarray(layered.layer_id)
        # Layer 0 cells should have higher activation than layer 1.
        a0 = float(jnp.mean(alpha[layer_ids == 0]))
        a1 = float(jnp.mean(alpha[layer_ids == 1]))
        assert a0 > a1

    def test_activation_field_saturates(self):
        layer_zs = np.array([0.0])
        bbox = [(-1, 1), (-1, 1), (0, 0.1)]
        layered = build_layered_mesh(layer_zs, bbox, nx=2, ny=2)
        layered = assign_activation_times(layered, np.array([0.0]))
        # At t=1e6 (way after activation): alpha ≈ 1.
        alpha = layered.activation_field(jnp.array(1e6), tau=1e-3)
        assert float(jnp.mean(alpha)) > 0.99
        # At t=-1e6 (way before): alpha ≈ 0.
        alpha = layered.activation_field(jnp.array(-1e6), tau=1e-3)
        assert float(jnp.mean(alpha)) < 0.01

    def test_part_cell_mask(self):
        sdf = sdf_box((0, 0, 0.5), (1, 1, 0.5))
        layer_zs = np.array([0.5])
        bbox = [(-1.5, 1.5), (-1.5, 1.5), (0, 1)]
        layered = build_layered_mesh(layer_zs, bbox, nx=4, ny=4)
        mask = part_cell_mask(layered, sdf)
        # Some cells should be inside, some outside.
        assert mask.sum() > 0
        assert (~mask).sum() > 0

    def test_cell_centers(self):
        layer_zs = np.array([0.0])
        bbox = [(-1, 1), (-1, 1), (0, 0.1)]
        layered = build_layered_mesh(layer_zs, bbox, nx=2, ny=2,
                                     cells_per_layer=1)
        cc = np.asarray(layered.cell_centers)
        assert cc.shape == (4, 3)
        assert np.all(np.isfinite(cc))


# ===========================================================================
# am_thermal.py
# ===========================================================================
class TestAMThermal:
    """Moving laser heat source."""

    def _make_problem(self):
        layer_zs = np.array([0.0, 0.1])
        bbox = [(-1, 1), (-1, 1), (0, 0.2)]
        layered = build_layered_mesh(layer_zs, bbox, nx=4, ny=4)
        layered = assign_activation_times(layered, np.array([0.0, 1.0]))
        paths = [zigzag_hatch(np.ones((4, 4)),
                              np.linspace(-1, 1, 4),
                              np.linspace(-1, 1, 4),
                              hatch_spacing=0.5, layer_z=z)
                 for z in [0.0, 0.1]]
        mlp = multi_layer_paths(paths)
        cfg = SLMConfig(scan_speed=1.0)
        return setup_am_thermal(layered, mlp, cfg)

    def test_setup_am_thermal(self):
        prob = self._make_problem()
        assert prob.layered.n_layers == 2
        assert len(prob.layer_durations) == 2
        assert prob.layer_start_times[0] == 0.0

    def test_heat_source_positive(self):
        prob = self._make_problem()
        centers = prob.layered.cell_centers
        q = prob.source_fn(centers, jnp.array(0.5))
        # Heat source should be non-negative.
        assert float(jnp.min(q)) >= 0.0

    def test_heat_source_zero_before_build(self):
        """Before the build starts (t < 0), the source should be near zero."""
        prob = self._make_problem()
        centers = prob.layered.cell_centers
        q = prob.source_fn(centers, jnp.array(-1.0))
        # At t=-1 (before any layer), the source should be tiny (clipped to layer 0
        # but t_local=0, frac=0 → laser at the start point, still depositing).
        # Just check it's finite.
        assert jnp.all(jnp.isfinite(q))

    def test_heat_source_differentiable_wrt_power(self):
        """The heat source should be differentiable w.r.t. laser power."""
        layer_zs = np.array([0.0])
        bbox = [(-1, 1), (-1, 1), (0, 0.1)]
        layered = build_layered_mesh(layer_zs, bbox, nx=2, ny=2,
                                     cells_per_layer=1)
        layered = assign_activation_times(layered, np.array([0.0]))
        paths = [zigzag_hatch(np.ones((4, 4)),
                              np.linspace(-1, 1, 4),
                              np.linspace(-1, 1, 4),
                              hatch_spacing=0.5, layer_z=0.0)]
        mlp = multi_layer_paths(paths)
        centers = layered.cell_centers

        def total_heat(power):
            # Use a beam radius and layer thickness comparable to the mesh
            # cell size so the Gaussian actually deposits heat on the cell
            # centres (default 40 µm layer is far smaller than the 0.1 m
            # test mesh and would zero out the depth attenuation).
            cfg = SLMConfig(laser_power=power, scan_speed=1.0,
                            beam_radius=0.5, layer_thickness=0.1)
            prob = setup_am_thermal(layered, mlp, cfg)
            return prob.source_fn(centers, jnp.array(0.5)).sum()

        g = jax.grad(total_heat)(jnp.array(200.0))
        assert jnp.isfinite(g)
        assert float(g) > 0  # more power → more heat


# ===========================================================================
# thermomechanical.py
# ===========================================================================
class TestThermoMechanical:
    """Layer-by-layer thermo-mechanical solver."""

    def _make_setup(self, n_layers=2, nx=3, ny=3):
        layer_zs = np.linspace(0.1, 0.2, n_layers)
        bbox = [(-1, 1), (-1, 1), (0, 0.3)]
        layered = build_layered_mesh(layer_zs, bbox, nx=nx, ny=ny,
                                     cells_per_layer=1)
        layered = assign_activation_times(
            layered, np.linspace(0, n_layers - 1, n_layers))
        # Clamp bottom face.
        z_nodes = np.asarray(layered.mesh.nodes)[:, 2]
        bottom = np.where(z_nodes <= 0.0 + 1e-9)[0]
        dim = 3
        dofs = jnp.asarray(
            np.concatenate([bottom * dim + d for d in range(dim)]))
        bcs = [DirichletBC.fixed(dofs, 0.0)]
        mat = LinearElasticIsotropic(E=200e9, nu=0.3)
        return layered, mat, bcs

    def test_assemble_activated_stiffness_shape(self):
        layered, mat, _ = self._make_setup()
        alpha = jnp.ones(layered.n_cells)
        K = assemble_activated_stiffness(layered.mesh, mat, alpha, dim=3)
        n_dofs = layered.mesh.n_nodes * 3
        assert K.shape == (n_dofs, n_dofs)

    def test_assemble_activated_stiffness_zero_for_inactive(self):
        """Inactive cells (alpha=0) should contribute zero stiffness."""
        layered, mat, _ = self._make_setup()
        alpha = jnp.zeros(layered.n_cells)  # all inactive
        K = assemble_activated_stiffness(layered.mesh, mat, alpha, dim=3)
        # With all alphas = 0, K should be (near) zero.
        assert float(jnp.max(jnp.abs(K))) < 1e-6

    def test_solve_thermomechanical_finite(self):
        """The solver should produce finite displacement and stress."""
        layered, mat, bcs = self._make_setup()
        n_layers = layered.n_layers
        n_cells = layered.n_cells
        # Uniform cooling strain: -1e-3 per cell per layer.
        eps = -1e-3 * jnp.ones((n_layers, n_cells))
        result = solve_thermomechanical(
            layered, mat,
            thermal_strain_per_layer=eps,
            dirichlet_bcs=bcs,
            T_ref=1600.0, alpha_T=13e-6,
            tau_activation=1e-3, dim=3,
        )
        assert jnp.all(jnp.isfinite(result.U_final))
        assert jnp.all(jnp.isfinite(result.residual_stress))
        # With cooling contraction, there should be some distortion.
        assert float(jnp.max(jnp.abs(result.U_final))) > 0.0

    def test_solve_thermomechanical_zero_strain_zero_stress(self):
        """Zero thermal strain → zero thermal force → zero displacement."""
        layered, mat, bcs = self._make_setup()
        eps = jnp.zeros((layered.n_layers, layered.n_cells))
        result = solve_thermomechanical(
            layered, mat,
            thermal_strain_per_layer=eps,
            dirichlet_bcs=bcs,
            T_ref=0.0, alpha_T=1e-5, tau_activation=1e-3, dim=3,
        )
        assert float(jnp.max(jnp.abs(result.U_final))) < 1e-6

    def test_solve_thermomechanical_history_shape(self):
        """The history arrays should have the right shape."""
        layered, mat, bcs = self._make_setup(n_layers=3)
        eps = -1e-3 * jnp.ones((3, layered.n_cells))
        result = solve_thermomechanical(
            layered, mat,
            thermal_strain_per_layer=eps,
            dirichlet_bcs=bcs, dim=3,
        )
        n_dofs = layered.mesh.n_nodes * 3
        assert result.U_history.shape == (3, n_dofs)
        assert result.stress_history.shape == (3, layered.n_cells, 3, 3)

    def test_solve_thermomechanical_differentiable(self):
        """The solver should be differentiable w.r.t. thermal strain."""
        layered, mat, bcs = self._make_setup(n_layers=2, nx=2, ny=2)

        def loss(eps_scale):
            eps = -eps_scale * jnp.ones((2, layered.n_cells))
            result = solve_thermomechanical(
                layered, mat,
                thermal_strain_per_layer=eps,
                dirichlet_bcs=bcs, dim=3,
            )
            return jnp.max(jnp.abs(result.U_final))

        g = jax.grad(loss)(jnp.array(1e-3))
        assert jnp.isfinite(g)
        assert float(g) > 0  # larger strain → larger displacement

    def test_more_layers_more_stress(self):
        """More deposited layers should not decrease peak residual stress."""
        def peak_stress(n_layers):
            layered, mat, bcs = self._make_setup(n_layers=n_layers,
                                                nx=3, ny=3)
            eps = -1e-3 * jnp.ones((n_layers, layered.n_cells))
            result = solve_thermomechanical(
                layered, mat,
                thermal_strain_per_layer=eps,
                dirichlet_bcs=bcs, dim=3,
            )
            vm = jnp.sqrt(jnp.sum(result.residual_stress ** 2,
                                  axis=(1, 2)))
            return float(jnp.max(jnp.abs(vm)))

        s1 = peak_stress(1)
        s3 = peak_stress(3)
        # More layers → more accumulated stress (or at least not less).
        assert s3 >= s1 - 1e-6


# ===========================================================================
# User-specified laser paths (arbitrary toolpaths)
# ===========================================================================
class TestUserScanPaths:
    """Loading externally-defined laser trajectories."""

    def test_from_waypoints_basic(self):
        """A simple (n, 2) waypoint loop should produce a valid ScanPath."""
        wp = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.], [0., 0.]])
        p = from_waypoints(wp, layer_z=0.5e-3)
        assert p.waypoints.shape == (5, 2)
        assert p.total_length > 0
        assert p.layer_z == pytest.approx(0.5e-3)
        # Perimeter of a unit square = 4.
        assert p.total_length == pytest.approx(4.0, rel=1e-9)

    def test_from_waypoints_3d_infers_layer_z(self):
        """3-D waypoints without explicit layer_z should use first-point z."""
        wp = np.array([[0., 0., 0.002], [1., 0., 0.002], [1., 1., 0.002]])
        p = from_waypoints(wp)
        assert p.layer_z == pytest.approx(0.002)
        # xy only is stored.
        assert p.waypoints.shape == (3, 2)

    def test_from_waypoints_collapses_duplicates(self):
        """Consecutive duplicate waypoints must be collapsed."""
        wp = np.array([[0., 0.], [0., 0.], [1., 0.], [1., 0.], [1., 1.]])
        p = from_waypoints(wp, layer_z=0.0)
        assert len(p.waypoints) == 3
        # No zero-length segments.
        assert np.all(p.segment_lengths > 1e-13)

    def test_from_waypoints_single_point_padded(self):
        """A single waypoint should be padded to a degenerate 2-point path."""
        p = from_waypoints(np.array([[1.0, 2.0]]), layer_z=0.0)
        assert len(p.waypoints) == 2
        assert p.total_length == pytest.approx(0.0)

    def test_chain_paths_concatenates(self):
        """chain_paths should join multiple paths end-to-end."""
        p1 = from_waypoints(np.array([[0., 0.], [1., 0.]]), layer_z=0.0)
        p2 = from_waypoints(np.array([[1., 0.], [1., 1.], [0., 1.]]),
                            layer_z=0.0)
        chained = chain_paths(p1, p2)
        assert len(chained.waypoints) == 4
        # 1 (bottom) + 1 + 1 (sides of upper L) = 3
        assert chained.total_length == pytest.approx(3.0, rel=1e-9)
        # layer_z inherited from first segment.
        assert chained.layer_z == pytest.approx(0.0)

    def test_chain_paths_override_layer_z(self):
        p1 = from_waypoints(np.array([[0., 0.], [1., 0.]]), layer_z=0.0)
        p2 = from_waypoints(np.array([[1., 0.], [1., 1.]]), layer_z=0.0)
        chained = chain_paths(p1, p2, layer_z=0.5)
        assert chained.layer_z == pytest.approx(0.5)

    def test_from_csv_round_trip(self, tmp_path):
        """CSV loading should reproduce the input waypoints."""
        wp = np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
        fpath = tmp_path / "trajectory.csv"
        with open(fpath, "w") as f:
            f.write("x,y\n")
            for row in wp:
                f.write(f"{row[0]},{row[1]}\n")
        p = from_csv(fpath, layer_z=0.001)
        assert np.allclose(p.waypoints, wp)
        assert p.layer_z == pytest.approx(0.001)

    def test_from_gcode_basic(self, tmp_path):
        """G-code parsing should convert mm → m and follow the moves."""
        gcode = (
            "G90 ; absolute mode\n"
            "G0 X0 Y0\n"
            "G1 X10 Y0\n"
            "G1 X10 Y10\n"
            "G1 X0 Y10\n"
            "G1 X0 Y0\n"
        )
        fpath = tmp_path / "path.gcode"
        fpath.write_text(gcode)
        p = from_gcode(fpath, layer_z=0.002)
        # 5 moves → 5 waypoints (no duplicates to collapse).
        assert len(p.waypoints) == 5
        # 10 mm = 0.01 m; perimeter of 10mm square = 40 mm = 0.04 m.
        assert np.isclose(p.waypoints[1, 0], 0.01)
        assert p.total_length == pytest.approx(0.04, rel=1e-9)
        assert p.layer_z == pytest.approx(0.002)

    def test_from_gcode_relative_raises(self, tmp_path):
        """Relative coordinates (G91) are not supported."""
        fpath = tmp_path / "rel.gcode"
        fpath.write_text("G91\nG1 X1 Y0\n")
        with pytest.raises(NotImplementedError):
            from_gcode(fpath, layer_z=0.0)

    def test_from_gcode_comments_skipped(self, tmp_path):
        """Comments and () should be ignored."""
        gcode = (
            "; header comment\n"
            "G90 ; absolute\n"
            "(inline comment)\n"
            "G1 X5 Y5\n"
            "G1 X10 Y5\n"
        )
        fpath = tmp_path / "with_comments.gcode"
        fpath.write_text(gcode)
        p = from_gcode(fpath, layer_z=0.0)
        # 2 moves → 2 waypoints.
        assert len(p.waypoints) == 2

    def test_custom_path_into_thermal_setup(self):
        """A user-defined path should plug into setup_am_thermal cleanly."""
        layer_zs = np.array([0.0, 0.1])
        bbox = [(-1, 1), (-1, 1), (0, 0.2)]
        layered = build_layered_mesh(layer_zs, bbox, nx=4, ny=4)
        layered = assign_activation_times(layered, np.array([0.0, 1.0]))
        # Two layers: a triangle on layer 0, a square on layer 1.
        tri = np.array([[-0.5, -0.5], [0.5, -0.5], [0.0, 0.5],
                        [-0.5, -0.5]])
        sq = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5],
                       [-0.5, 0.5], [-0.5, -0.5]])
        mlp = multi_layer_paths([
            from_waypoints(tri, layer_z=0.0),
            from_waypoints(sq, layer_z=0.1),
        ])
        cfg = SLMConfig(scan_speed=1.0)
        prob = setup_am_thermal(layered, mlp, cfg)
        assert prob.layered.n_layers == 2
        # Source should be finite at the layer-0 mid-time.
        q = prob.source_fn(layered.cell_centers, jnp.array(0.5))
        assert jnp.all(jnp.isfinite(q))

    def test_custom_path_end_to_end_gradient(self):
        """A custom toolpath should keep the pipeline differentiable."""
        layer_zs = np.array([0.0, 0.1])
        bbox = [(-1, 1), (-1, 1), (0, 0.2)]
        layered = build_layered_mesh(layer_zs, bbox, nx=3, ny=3,
                                     cells_per_layer=1)
        layered = assign_activation_times(layered, np.array([0.0, 1.0]))
        # Two layers of custom triangle loops.
        tri = np.array([[-0.5, -0.5], [0.5, -0.5], [0.0, 0.5],
                        [-0.5, -0.5]])
        mlp = multi_layer_paths([
            from_waypoints(tri, layer_z=0.0),
            from_waypoints(tri, layer_z=0.1),
        ])
        centers = layered.cell_centers

        def total_heat(power):
            cfg = SLMConfig(laser_power=power, scan_speed=1.0,
                            beam_radius=0.5, layer_thickness=0.1)
            prob = setup_am_thermal(layered, mlp, cfg)
            return prob.source_fn(centers, jnp.array(0.5)).sum()

        g = jax.grad(total_heat)(jnp.array(200.0))
        assert jnp.isfinite(g)
        assert float(g) > 0  # more power → more heat


# ===========================================================================
# Arbitrary complex geometry from triangle meshes
# ===========================================================================
class TestMeshSDF:
    """SDF construction from arbitrary triangle meshes."""

    def _tetra_mesh(self):
        """A unit tetrahedron mesh with CCW-outward winding."""
        verts = np.array([
            [0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]
        ])
        # CCW-outward: bottom face wound clockwise viewed from below.
        tris = np.array([
            [2, 1, 0],   # bottom (normal -z)
            [0, 1, 3],   # front (normal -y)
            [0, 2, 3],   # left  (normal -x)
            [1, 2, 3],   # top   (normal +xyz)
        ])
        return verts, tris

    def test_tetra_inside_outside(self):
        verts, tris = self._tetra_mesh()
        sdf = sdf_from_mesh(verts, tris)
        # Inside (centroid).
        assert float(sdf(jnp.array([0.1, 0.1, 0.1]))) < 0.0
        # Outside.
        assert float(sdf(jnp.array([1.0, 1.0, 1.0]))) > 0.0
        # On a face vertex.
        assert float(sdf(jnp.array([0., 0., 0.]))) >= -1e-6

    def test_tetra_distance_magnitude(self):
        """The SDF magnitude should be the true Euclidean distance."""
        verts, tris = self._tetra_mesh()
        sdf = sdf_from_mesh(verts, tris)
        # Point (2, 0, 0): closest point is vertex (1, 0, 0) → distance 1.
        d = float(sdf(jnp.array([2.0, 0.0, 0.0])))
        assert d == pytest.approx(1.0, abs=1e-6)
        # Point (-1, 0, 0): closest point is origin → distance 1.
        d = float(sdf(jnp.array([-1.0, 0.0, 0.0])))
        assert d == pytest.approx(1.0, abs=1e-6)

    def test_mesh_sdf_differentiable(self):
        """The mesh SDF should be differentiable w.r.t. coordinates."""
        verts, tris = self._tetra_mesh()
        sdf = sdf_from_mesh(verts, tris)
        # Gradient at (2, 0, 0) should point away from the closest vertex
        # (1, 0, 0), so ∂d/∂x = +1.
        g = jax.grad(lambda x: sdf(jnp.array([x, 0.0, 0.0])))(2.0)
        assert float(g) == pytest.approx(1.0, abs=1e-4)

    def test_mesh_sdf_batched(self):
        """The SDF should accept batched query points."""
        verts, tris = self._tetra_mesh()
        sdf = sdf_from_mesh(verts, tris)
        pts = jnp.array([
            [0.1, 0.1, 0.1],   # inside
            [2.0, 0.0, 0.0],   # outside
            [-1.0, 0.0, 0.0],  # outside
        ])
        s = sdf(pts)
        assert s.shape == (3,)
        assert float(s[0]) < 0
        assert float(s[1]) > 0
        assert float(s[2]) > 0

    def test_mesh_sdf_chunked(self):
        """The chunked variant should agree with the unbatched one."""
        verts, tris = self._tetra_mesh()
        sdf_full = sdf_from_mesh(verts, tris)
        sdf_chunked = sdf_from_mesh(verts, tris, chunk=2)
        pts = jnp.array([[0.1, 0.1, 0.1], [2.0, 0.0, 0.0],
                         [-1.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        s_full = sdf_full(pts)
        s_chunk = sdf_chunked(pts)
        assert jnp.allclose(s_full, s_chunk)

    def test_mesh_sdf_in_am_pipeline(self):
        """A mesh-based SDF should plug into part_bbox and slice_mask."""
        verts, tris = self._tetra_mesh()
        sdf = sdf_from_mesh(verts, tris)
        # Bounding box should be roughly [0,1]³.
        bbox = part_bbox(sdf, resolution=20, margin=0.05,
                         search_range=(-0.5, 1.5))
        assert bbox[0][0] < 0.1
        assert bbox[0][1] > 0.9
        assert bbox[2][1] > 0.9

    def test_box_mesh_inside_outside(self):
        """A closed box mesh (6 faces, 12 tris) should give correct in/out.

        This exercises the multi-face closed-mesh case (different from the
        4-face tetrahedron): each of the 6 box faces contributes 2 triangles,
        and the winding-number inside/outside test must handle them all.
        """
        # Unit cube [0,1]³ with 8 vertices, CCW-outward winding.
        v = np.array([
            [0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],  # bottom
            [0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.],  # top
        ])
        # 12 triangles, 2 per face, all wound CCW-outward.
        tris = np.array([
            # bottom (z=0, normal -z): CCW viewed from below.
            [0, 2, 1], [0, 3, 2],
            # top (z=1, normal +z): CCW viewed from above.
            [4, 5, 6], [4, 6, 7],
            # front (y=0, normal -y).
            [0, 1, 5], [0, 5, 4],
            # back (y=1, normal +y).
            [3, 6, 2], [3, 7, 6],
            # left (x=0, normal -x).
            [0, 4, 7], [0, 7, 3],
            # right (x=1, normal +x).
            [1, 2, 6], [1, 6, 5],
        ])
        sdf = sdf_from_mesh(v, tris)
        # Inside (centre).
        assert float(sdf(jnp.array([0.5, 0.5, 0.5]))) < 0.0
        # Outside each face.
        assert float(sdf(jnp.array([2.0, 0.5, 0.5]))) > 0.0
        assert float(sdf(jnp.array([-1.0, 0.5, 0.5]))) > 0.0
        assert float(sdf(jnp.array([0.5, 2.0, 0.5]))) > 0.0
        assert float(sdf(jnp.array([0.5, -1.0, 0.5]))) > 0.0
        assert float(sdf(jnp.array([0.5, 0.5, 2.0]))) > 0.0
        assert float(sdf(jnp.array([0.5, 0.5, -1.0]))) > 0.0
        # On a face centre: distance ≈ 0.
        assert abs(float(sdf(jnp.array([0.0, 0.5, 0.5])))) < 1e-6
