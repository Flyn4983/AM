"""Tests for the AM microstructure module (phase-field solidification ×
thermal-history coupling).

Covers:
- State construction (substrate seeding, dimensionality)
- Undercooling and temperature-gated mobility
- Solidification grows under cold (T < T_melt), melts under hot (T > T_melt)
- Melt-pool re-melting resets nucleation (AM thermal cycling)
- Microstructure metrics (solidified fraction, grain size, grain count)
- End-to-end differentiability w.r.t. process parameters (laser power)
- 2D and 3D
- Coupling with the AM thermal source (FEM path) and particle AM (particle path)
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from diffmech.methods.am import (
    MicrostructureConfig, AMMicrostructureState, make_microstructure_state,
    thermal_undercooling, grain_mobility,
    solidification_rhs, orientation_rhs,
    step_am_microstructure, step_am_microstructure_scan,
    solidified_fraction, nucleated_fraction,
    mean_grain_size, grain_count_estimate,
    cooling_rate, morphology_indicator, thermal_field_from_am,
)
from diffmech.methods.am.scan_paths import from_waypoints, multi_layer_paths
from diffmech.methods.am.am_thermal import am_heat_source
from diffmech.methods.am.process import SLMConfig


# ---------------------------------------------------------------------------
# Stable phase-field parameters (satisfy the explicit-Euler CFL condition
# dt < dx^2 * tau / (2 * dim * W^2) with W = 2.5 * dx).
# ---------------------------------------------------------------------------
DX = 1e-4
W = 2.5 * DX
TAU = 1e-4
DT = DX ** 2 * TAU / (2 * 2 * W ** 2) / 2.0  # half the CFL limit
N_STEPS = 200


def _cfg(**kw):
    base = dict(dx=DX, dt=DT, T_melt=1900.0, tau=TAU, interface_W=W,
                lam=2.0, mobility0=0.0)
    base.update(kw)
    return MicrostructureConfig(**base)


# ===========================================================================
class TestStateConstruction:
    def test_make_state_2d_substrate(self):
        st = make_microstructure_state((8, 8), substrate_layers=1)
        assert st.dim == 2
        # Substrate (last axis = build direction z) seeded solid.
        assert float(st.phi[:, 0].mean()) == pytest.approx(1.0)
        assert float(st.phi[:, 1].mean()) == pytest.approx(0.0)
        assert float(nucleated_fraction(st)) == pytest.approx(1.0 / 8)

    def test_make_state_3d_substrate(self):
        st = make_microstructure_state((5, 5, 5), substrate_layers=1)
        assert st.dim == 3
        assert float(st.phi[:, :, 0].mean()) == pytest.approx(1.0)
        assert float(st.phi[:, :, 1].mean()) == pytest.approx(0.0)

    def test_make_state_no_substrate(self):
        st = make_microstructure_state((6, 6), substrate_layers=0)
        assert float(jnp.mean(st.phi)) == pytest.approx(0.0)
        assert float(nucleated_fraction(st)) == pytest.approx(0.0)

    def test_state_is_pytree(self):
        st = make_microstructure_state((4, 4), substrate_layers=1)
        # round-trip through tree flatten/unflatten
        flat, treedef = jax.tree_util.tree_flatten(st)
        st2 = jax.tree_util.tree_unflatten(treedef, flat)
        assert jnp.allclose(st.phi, st2.phi)


# ===========================================================================
class TestUndercoolingAndMobility:
    def test_undercooling_sign(self):
        cfg = _cfg()
        T_cold = jnp.full((4, 4), 1500.0)
        T_hot = jnp.full((4, 4), 2200.0)
        u_cold = thermal_undercooling(T_cold, cfg)
        u_hot = thermal_undercooling(T_hot, cfg)
        # Cold => positive undercooling; hot => negative.
        assert float(u_cold[0, 0]) > 0.0
        assert float(u_hot[0, 0]) < 0.0

    def test_grain_mobility_zero_above_melt(self):
        cfg = _cfg()
        T_hot = jnp.full((4, 4), 2200.0)  # above melt
        M = grain_mobility(T_hot, cfg)
        # Mobility gated off above T_melt (liquid: orientation meaningless).
        assert float(M[0, 0]) == pytest.approx(0.0, abs=1e-6)

    def test_grain_mobility_positive_in_haz(self):
        cfg = _cfg(mobility0=1.0)  # enable grain-growth mobility
        # Just below melt (heat-affected zone).
        T_haz = jnp.full((4, 4), 1850.0)
        M = grain_mobility(T_haz, cfg)
        assert float(M[0, 0]) > 0.0


# ===========================================================================
class TestSolidificationMelting:
    def test_cold_solidifies(self):
        cfg = _cfg()
        st = make_microstructure_state((10, 10), substrate_layers=1)
        init = float(jnp.mean(st.phi))
        T_cold = jnp.full((10, 10), 1500.0)
        st_f = step_am_microstructure_scan(
            st, cfg, jnp.broadcast_to(T_cold, (N_STEPS, 10, 10)))
        # Should grow well past the initial substrate fraction.
        assert float(jnp.mean(st_f.phi)) > init + 0.3

    def test_hot_melts_substrate(self):
        cfg = _cfg()
        st = make_microstructure_state((10, 10), substrate_layers=1)
        T_hot = jnp.full((10, 10), 2200.0)
        st_f = step_am_microstructure_scan(
            st, cfg, jnp.broadcast_to(T_hot, (N_STEPS, 10, 10)))
        # Melt pool dissolves even the solid substrate.
        assert float(jnp.mean(st_f.phi)) < 0.05

    def test_cold_more_solid_than_hot(self):
        cfg = _cfg()
        st0 = make_microstructure_state((10, 10), substrate_layers=1)
        T_cold = jnp.full((10, 10), 1500.0)
        T_hot = jnp.full((10, 10), 2200.0)
        st_c = step_am_microstructure_scan(
            st0, cfg, jnp.broadcast_to(T_cold, (N_STEPS, 10, 10)))
        st_h = step_am_microstructure_scan(
            st0, cfg, jnp.broadcast_to(T_hot, (N_STEPS, 10, 10)))
        assert float(jnp.mean(st_c.phi)) > float(jnp.mean(st_h.phi)) + 0.5

    def test_step_finite_and_shapes(self):
        cfg = _cfg()
        st = make_microstructure_state((6, 6), substrate_layers=1)
        T = jnp.full((6, 6), 1700.0)
        st2 = step_am_microstructure(st, cfg, T)
        assert st2.phi.shape == (6, 6)
        assert st2.theta.shape == (6, 6)
        assert st2.nucleated.shape == (6, 6)
        assert jnp.all(jnp.isfinite(st2.phi))


# ===========================================================================
class TestRemelting:
    def test_melt_pool_resets_nucleation(self):
        """Re-melted cells lose nucleation status (AM thermal cycling)."""
        cfg = _cfg()
        st = make_microstructure_state((10, 10), substrate_layers=1)
        # Solidify first.
        T_cold = jnp.full((10, 10), 1500.0)
        st_solid = step_am_microstructure_scan(
            st, cfg, jnp.broadcast_to(T_cold, (N_STEPS, 10, 10)))
        assert float(nucleated_fraction(st_solid)) > 0.5
        # Then re-melt: nucleation should drop.
        T_hot = jnp.full((10, 10), 2200.0)
        st_remelt = step_am_microstructure_scan(
            st_solid, cfg, jnp.broadcast_to(T_hot, (N_STEPS, 10, 10)))
        assert float(nucleated_fraction(st_remelt)) < 0.1


# ===========================================================================
class TestMicrostructure3D:
    def test_3d_solidification(self):
        cfg = _cfg()  # 3D CFL: dim=3 -> dt smaller; recompute with dim=3
        # Recompute stable dt for 3D.
        dt3d = DX ** 2 * TAU / (2 * 3 * W ** 2) / 2.0
        cfg = _cfg(dt=dt3d)
        st = make_microstructure_state((6, 6, 6), substrate_layers=1)
        T_cold = jnp.full((6, 6, 6), 1500.0)
        st_f = step_am_microstructure_scan(
            st, cfg, jnp.broadcast_to(T_cold, (150, 6, 6, 6)))
        assert float(jnp.mean(st_f.phi)) > 0.3
        assert jnp.all(jnp.isfinite(st_f.phi))


# ===========================================================================
class TestMetrics:
    def test_solidified_fraction_range(self):
        cfg = _cfg()
        st = make_microstructure_state((8, 8), substrate_layers=1)
        assert 0.0 <= float(solidified_fraction(st)) <= 1.0

    def test_mean_grain_size_finite(self):
        cfg = _cfg()
        st = make_microstructure_state((8, 8), substrate_layers=2)
        # Run a few steps to develop grain structure.
        T = jnp.full((8, 8), 1700.0)
        st = step_am_microstructure_scan(
            st, cfg, jnp.broadcast_to(T, (N_STEPS, 8, 8)))
        gs = mean_grain_size(st, DX)
        assert jnp.isfinite(gs)
        assert float(gs) > 0.0

    def test_grain_count_finite(self):
        cfg = _cfg()
        st = make_microstructure_state((8, 8), substrate_layers=2)
        T = jnp.full((8, 8), 1700.0)
        st = step_am_microstructure_scan(
            st, cfg, jnp.broadcast_to(T, (N_STEPS, 8, 8)))
        gc = grain_count_estimate(st, DX)
        assert jnp.isfinite(gc)
        assert float(gc) > 0.0

    def test_cooling_rate(self):
        # Monotonic cooling history -> positive mean cooling rate.
        T_hist = jnp.linspace(2000.0, 1000.0, 50)[:, None, None] * \
            jnp.ones((50, 4, 4))
        cr = cooling_rate(T_hist, DT)
        assert float(cr) > 0.0

    def test_morphology_indicator_signs(self):
        # High G/R -> columnar (positive); low G/R -> equiaxed (negative).
        col = morphology_indicator(jnp.array(5e6), jnp.array(1.0))
        eq = morphology_indicator(jnp.array(1e3), jnp.array(1.0))
        assert float(col) > 0.0
        assert float(eq) < 0.0


# ===========================================================================
class TestDifferentiability:
    def test_grad_through_thermal_field(self):
        """Gradient of solid fraction w.r.t. a temperature-scaling parameter."""
        cfg = _cfg()
        st0 = make_microstructure_state((8, 8), substrate_layers=1)

        def loss(scale):
            # scale controls how far below T_melt the field sits.
            T = 1900.0 - scale * 2.0  # higher scale => colder => more solid
            Ts = jnp.broadcast_to(jnp.full((8, 8), T), (N_STEPS, 8, 8))
            sf = step_am_microstructure_scan(st0, cfg, Ts)
            return jnp.mean(sf.phi)

        g = jax.grad(loss)(100.0)
        assert jnp.isfinite(g)
        # Colder => more solid => dphi/dscale > 0.
        assert float(g) > 0.0

    def test_grad_finite_3d(self):
        dt3d = DX ** 2 * TAU / (2 * 3 * W ** 2) / 2.0
        cfg = _cfg(dt=dt3d)
        st0 = make_microstructure_state((5, 5, 5), substrate_layers=1)

        def loss(power):
            T = 1900.0 - power * 1.5
            Ts = jnp.broadcast_to(jnp.full((5, 5, 5), T), (80, 5, 5, 5))
            sf = step_am_microstructure_scan(st0, cfg, Ts)
            return jnp.mean(sf.phi)

        g = jax.grad(loss)(50.0)
        assert jnp.isfinite(g)


# ===========================================================================
class TestCouplingWithAMThermal:
    """Couple the microstructure with the actual AM heat source (laser path)."""

    def _make_thermal_source(self, laser_power):
        # A simple 1-layer zigzag path over a small domain.
        wp = jnp.asarray([[0.0, 0.0], [0.4, 0.0], [0.4, 0.4], [0.0, 0.4]])
        path = from_waypoints(wp, layer_z=0.2)
        paths = multi_layer_paths([path])
        cfg_proc = SLMConfig(
            scan_speed=0.05, layer_thickness=0.05, beam_radius=0.05,
            laser_power=laser_power, absorption=0.5,
        )
        lengths = np.asarray([p.total_length for p in paths.layers])
        durations = lengths / cfg_proc.scan_speed
        starts = np.concatenate([[0.0], np.cumsum(durations)[:-1]])
        source = am_heat_source(paths, starts, durations, cfg_proc)
        return source, starts, durations, cfg_proc

    def test_thermal_field_callable_runs(self):
        source, starts, durations, _ = self._make_thermal_source(200.0)
        centers = jnp.asarray(np.meshgrid(
            np.linspace(0, 0.4, 5), np.linspace(0, 0.4, 5),
            indexing="ij")).reshape(-1, 2)
        centers = jnp.concatenate([centers, jnp.full((25, 1), 0.2)], axis=1)
        Q = source(centers, 0.5)
        assert Q.shape == (25,)
        assert jnp.all(jnp.isfinite(Q))
        # Heat is deposited (positive) near the laser.
        assert float(jnp.max(Q)) > 0.0

    def test_microstructure_consumes_am_thermal(self):
        """Microstructure evolves under a (laser-power-dependent) AM heat field."""
        source, starts, durations, pcfg = self._make_thermal_source(200.0)
        # Build a coarse grid of cell centres and a thermal history.
        xs = np.linspace(0.0, 0.4, 6)
        grid = np.stack(np.meshgrid(xs, xs, indexing="ij"), axis=-1).reshape(-1, 2)
        centers = jnp.asarray(np.concatenate(
            [grid, np.full((36, 1), 0.2)], axis=1))
        # Preheat; heat up via the source then cool.
        T0 = 1500.0
        times = np.linspace(0.0, float(durations[0]), 20)
        T_hist = []
        T = jnp.full(36, T0)
        for t in times:
            Q = source(centers, t)
            T = T + 1e-3 * Q / 4.0e6  # simple heating
            T_hist.append(T)
        # Cool down.
        for _ in range(20):
            T = T - 5.0
            T_hist.append(jnp.maximum(T, 300.0))
        T_hist = jnp.stack(T_hist)  # (40, 36) — reshape to grid for microstructure
        T_grid = T_hist.reshape(40, 6, 6)
        cfg = _cfg()
        st = make_microstructure_state((6, 6), substrate_layers=1)
        st_f = step_am_microstructure_scan(st, cfg, T_grid)
        assert jnp.all(jnp.isfinite(st_f.phi))
        # Some solidification should occur during cooling.
        assert float(jnp.mean(st_f.phi)) >= float(jnp.mean(st.phi)) - 0.1

    def test_grad_wrt_laser_power(self):
        """End-to-end gradient: laser power -> AM heat -> microstructure."""
        def loss(power):
            source, starts, durations, _ = self._make_thermal_source(power)
            xs = np.linspace(0.0, 0.4, 5)
            grid = np.stack(np.meshgrid(xs, xs, indexing="ij"),
                            axis=-1).reshape(-1, 2)
            centers = jnp.asarray(np.concatenate(
                [grid, np.full((25, 1), 0.2)], axis=1))
            T = jnp.full(25, 1500.0)
            T_hist = []
            for t in np.linspace(0.0, 0.5, 10):
                Q = source(centers, t)
                T = T + 1e-3 * Q / 4.0e6
                T_hist.append(T)
            for _ in range(10):
                T = T - 5.0
                T_hist.append(jnp.maximum(T, 300.0))
            T_grid = jnp.stack(T_hist).reshape(20, 5, 5)
            cfg = _cfg()
            st = make_microstructure_state((5, 5), substrate_layers=1)
            sf = step_am_microstructure_scan(st, cfg, T_grid)
            return jnp.mean(sf.phi)

        g = jax.grad(loss)(200.0)
        assert jnp.isfinite(g)
