"""Tests for the end-to-end AM digital-verification pipeline example.

These tests verify that the full "process → thermal history → microstructure
→ constitutive → structural response → service performance" chain runs
end-to-end and is differentiable through ``jax.grad`` from the AM laser power
all the way to the final fracture-energy metric.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import sys
from pathlib import Path

# Make the ``examples`` directory importable.
_EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
sys.path.insert(0, str(_EXAMPLES_DIR))

import am_pipeline as am  # noqa: E402


# ---------------------------------------------------------------------------
# Stage 0: thermal history
# ---------------------------------------------------------------------------
def test_thermal_stage_runs():
    """Stage 0: a moving laser should heat the domain above ambient."""
    cr, g, T_hist = am.thermal_history_from_laser(jnp.array(50.0),
                                                   grid=12, n_steps=30)
    assert T_hist.shape[0] == 31
    assert float(T_hist.max()) > 300.0          # heated above ambient
    assert float(cr) >= 0.0                       # cooling rate non-negative


def test_thermal_more_power_means_hotter():
    """Doubling the laser power should not decrease the peak temperature."""
    _, _, T1 = am.thermal_history_from_laser(jnp.array(50.0),
                                             grid=12, n_steps=30)
    _, _, T2 = am.thermal_history_from_laser(jnp.array(100.0),
                                             grid=12, n_steps=30)
    assert float(T2.max()) >= float(T1.max())


# ---------------------------------------------------------------------------
# Individual downstream stages
# ---------------------------------------------------------------------------
def test_microstructure_stage_runs():
    """Stage 1: Allen-Cahn evolution should produce a bounded order parameter."""
    state, cfg = am.microstructure_from_cooling_rate(jnp.array(0.0),
                                                      n_steps=10, grid=12)
    assert state.phi.shape == (12, 12)
    # Order parameter should stay bounded (Allen-Cahn is dissipative).
    assert float(jnp.max(jnp.abs(state.phi))) < 5.0


def test_grain_metric_positive():
    """The grain-refinement metric should be a positive scalar."""
    state, cfg = am.microstructure_from_cooling_rate(jnp.array(0.0),
                                                      n_steps=10, grid=12)
    G = am.grain_refinement_metric(state, cfg)
    assert float(G) > 0.0


def test_cpfe_constitutive_response_nonnegative():
    """Stage 2: CPFE should return a non-negative equivalent plastic strain."""
    G = jnp.array(0.5)
    eps_p = am.cpfe_constitutive_response(G, nx=2, ny=2)
    assert float(eps_p) >= 0.0


def test_structural_response_finite():
    """Stage 3: FEM displacement should be finite and non-zero."""
    U, mesh = am.structural_response(jnp.array(0.01), nx=4, ny=2)
    assert U.shape[0] == mesh.n_nodes * 2
    assert jnp.all(jnp.isfinite(U))
    # A gravity-loaded cantilever should deflect (non-zero displacement).
    assert float(jnp.max(jnp.abs(U))) > 0.0


def test_service_performance_positive():
    """Stage 4: fracture energy should be a positive scalar."""
    U, _ = am.structural_response(jnp.array(0.01), nx=4, ny=2)
    Ef, state = am.service_performance(U, grid=8)
    assert float(Ef) > 0.0
    # The phase field should remain in [0, 1] (intact → broken).
    assert float(jnp.min(state.phi)) >= 0.0 - 1e-9
    assert float(jnp.max(state.phi)) <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# End-to-end differentiability
# ---------------------------------------------------------------------------
def test_pipeline_forward_pass():
    """The full pipeline should run and return all stages."""
    result = am.am_pipeline(jnp.array(50.0))
    assert "cooling_rate" in result
    assert "grain_metric" in result
    assert "plastic_strain" in result
    assert "fracture_energy" in result
    assert "thermal_history" in result
    assert float(result["fracture_energy"]) > 0.0


def test_pipeline_end_to_end_gradient_is_finite():
    """dE_f / d(laser_power) should be a finite scalar.

    This is the key test: it confirms that gradients propagate through
    thermal FVM → Allen-Cahn → Hall-Petch → CPFE → FEM → phase-field fracture.
    """
    g = jax.grad(am.loss_fn)(jnp.array(50.0))
    assert jnp.isfinite(g)
    # The gradient should be non-trivial (not exactly zero) — if it were zero
    # the inverse-design stage would be unable to improve the design.
    assert float(jnp.abs(g)) > 1e-12


def test_pipeline_gradient_changes_with_laser_power():
    """The fracture energy should actually depend on the laser power.

    Comparing two nearby laser powers confirms the pipeline is not a
    constant function of the process parameter.
    """
    Ef_50 = float(am.loss_fn(jnp.array(50.0)))
    Ef_60 = float(am.loss_fn(jnp.array(60.0)))
    Ef_40 = float(am.loss_fn(jnp.array(40.0)))
    # At least one of the perturbed values should differ from the baseline.
    assert (abs(Ef_60 - Ef_50) > 1e-9) or (abs(Ef_40 - Ef_50) > 1e-9)


# ---------------------------------------------------------------------------
# Inverse design
# ---------------------------------------------------------------------------
def test_optimisation_reduces_loss():
    """Gradient descent on the laser power should not increase the loss."""
    power_opt, history = am.optimise_laser_power(initial_power=50.0,
                                                 n_iter=10, lr=2.0)
    Ef_initial = history[0][2]
    Ef_final = history[-1][2]
    # The optimised loss should be no worse than the initial loss.
    assert Ef_final <= Ef_initial + 1e-12


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def test_export_writes_files(tmp_path):
    """The export step should write VTU and NPZ files."""
    result = am.am_pipeline(jnp.array(50.0))
    power_opt, history = am.optimise_laser_power(initial_power=50.0,
                                                 n_iter=3, lr=2.0)
    out = am.export_results(result, history, out_dir=tmp_path)
    assert (tmp_path / "thermal_history.vtu").exists()
    assert (tmp_path / "microstructure.vtu").exists()
    assert (tmp_path / "structural_response.vtu").exists()
    assert (tmp_path / "fracture.vtu").exists()
    assert (tmp_path / "am_pipeline.npz").exists()
    # The npz should contain the optimisation history arrays.
    data = np.load(tmp_path / "am_pipeline.npz")
    assert "laser_power" in data.files
    assert "fracture_energy" in data.files
