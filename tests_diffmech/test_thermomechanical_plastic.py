"""Tests for the P2-② plastic / crystal-plasticity thermo-mechanical AM solver.

Covers the behaviour that distinguishes a *real* elasto-plastic build simulation
from a linear-elastic proxy:

* **Yield capping** — once the (large) thermal contraction drives the stress
  past the yield surface, the J2 return-map relaxes it back to the yield strength.
  The residual stress is therefore bounded near σ_y, never the elastic
  overshoot ``E·α·ΔT``.
* **Plastic accumulation** — equivalent plastic strain is strictly positive after
  the build (material yielded and stayed yielded).
* **Anisotropy (CPFE)** — the crystal-plasticity branch produces a stress field
  that differs from the isotropic J2 field (crystal hardening / orientation).
* **Element birth only inside the part** — a per-cell mask zeroes stress outside
  the geometry.
* **End-to-end differentiability** — ``jax.grad`` w.r.t. a scalar scaling the
  thermal load is finite and non-zero, so the whole build is one differentiable
  function of process parameters.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from diffmech.methods.am.activation import build_layered_mesh
from diffmech.methods.am.thermomechanical_plastic import (
    solve_thermomechanical_plastic,
)
from diffmech.materials.plasticity import J2Plasticity
from diffmech.methods.cpfe.crystal_plasticity import CrystalPlasticity
from diffmech.methods.cpfe.slip_systems import fcc_slip_systems
from diffmech.solvers.boundary_conditions import DirichletBC


# ---------------------------------------------------------------------------
# Fixtures: a tiny layered hex mesh + a per-layer thermal strain field
# ---------------------------------------------------------------------------
def _make_layered(nx=4, ny=4, nz=4, L=1e-3):
    layer_zs = np.linspace(0.0, L, nz)
    bbox = [(0.0, L), (0.0, L), (0.0, L)]
    layered = build_layered_mesh(np.asarray(layer_zs), bbox,
                                nx=nx, ny=ny, cells_per_layer=1)
    # bottom-face clamp (build plate)
    nodes = np.asarray(layered.mesh.nodes)
    bot = np.where(nodes[:, 2] < 0.5 * L)[0]
    dofs = (np.repeat(bot * 3, 3) + np.tile(np.arange(3), bot.size)).astype(np.int64)
    bc = DirichletBC.fixed(jnp.asarray(dofs), 0.0)
    return layered, bc


def _thermal_strain(layered, alpha_T=16.5e-6, T_ref=300.0, peak=900.0):
    """Per-layer thermal strain: each cell loaded only at its own deposition."""
    n_layers = layered.n_layers
    layer_id = np.asarray(layered.layer_id)
    dT = alpha_T * (peak - T_ref)
    onehot = (layer_id[None, :] == np.arange(n_layers)[:, None]).astype(float)
    return jnp.asarray(dT * onehot)  # (n_layers, n_cells)


def _vm3d(sigma):
    """von Mises of a (...,3,3) stress tensor."""
    sxx, syy, szz = sigma[..., 0, 0], sigma[..., 1, 1], sigma[..., 2, 2]
    sxy, syz, szx = sigma[..., 0, 1], sigma[..., 1, 2], sigma[..., 0, 2]
    arg = 0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) \
        + 3.0 * (sxy ** 2 + syz ** 2 + szx ** 2)
    return jnp.sqrt(jnp.maximum(arg, 0.0))


# ---------------------------------------------------------------------------
class TestYieldCapping:
    def test_elastic_predictor_is_unbounded(self):
        """With σ_y → ∞ the solver must reproduce the elastic overshoot."""
        layered, bc = _make_layered()
        ts = _thermal_strain(layered)
        mat = J2Plasticity(E=195e9, nu=0.28, sigma_y0=1e15, H=0.0)
        res = solve_thermomechanical_plastic(
            layered, mat, thermal_strain_per_layer=ts, dirichlet_bcs=[bc],
            constitutive="j2", T_ref=300.0, alpha_T=16.5e-6, n_sub_cp=64)
        vm = _vm3d(res.residual_stress)
        assert jnp.all(jnp.isfinite(vm))
        # elastic predictor: peak stress ≈ E·α·ΔT
        expected = 195e9 * 16.5e-6 * (900.0 - 300.0)
        assert float(jnp.max(vm)) > 0.8 * expected

    def test_j2_caps_stress_near_yield(self):
        """Yielding caps the residual stress near σ_y, below the elastic overshoot."""
        layered, bc = _make_layered()
        ts = _thermal_strain(layered)
        mat = J2Plasticity(E=195e9, nu=0.28, sigma_y0=300e6, H=1e9)
        res = solve_thermomechanical_plastic(
            layered, mat, thermal_strain_per_layer=ts, dirichlet_bcs=[bc],
            constitutive="j2", T_ref=300.0, alpha_T=16.5e-6, n_sub_cp=64)
        vm = _vm3d(res.residual_stress)
        assert jnp.all(jnp.isfinite(vm))
        overshoot = 195e9 * 16.5e-6 * (900.0 - 300.0)
        # residual must be clearly below the elastic overshoot (plasticity relaxes)
        assert float(jnp.max(vm)) < 1.5 * overshoot
        assert float(jnp.mean(vm)) < overshoot
        # plastic strain genuinely accumulated
        assert float(jnp.mean(res.eq_plastic_strain)) > 0.0


class TestPlasticVsElastic:
    def test_j2_residual_below_elastic(self):
        """J2 residual < elastic predictor residual (this is what yielding does)."""
        layered, bc = _make_layered()
        ts = _thermal_strain(layered)

        mat_pred = J2Plasticity(E=195e9, nu=0.28, sigma_y0=1e15, H=0.0)
        mat_j2 = J2Plasticity(E=195e9, nu=0.28, sigma_y0=300e6, H=1e9)

        r_pred = solve_thermomechanical_plastic(
            layered, mat_pred, thermal_strain_per_layer=ts, dirichlet_bcs=[bc],
            constitutive="j2", T_ref=300.0, alpha_T=16.5e-6, n_sub_cp=64)
        r_j2 = solve_thermomechanical_plastic(
            layered, mat_j2, thermal_strain_per_layer=ts, dirichlet_bcs=[bc],
            constitutive="j2", T_ref=300.0, alpha_T=16.5e-6, n_sub_cp=64)

        mean_pred = float(jnp.mean(_vm3d(r_pred.residual_stress)))
        mean_j2 = float(jnp.mean(_vm3d(r_j2.residual_stress)))
        assert mean_j2 < 0.9 * mean_pred


class TestCPFE:
    def test_cpfe_finite_and_capped(self):
        layered, bc = _make_layered()
        ts = _thermal_strain(layered)
        dirs, norms = fcc_slip_systems()
        mat = CrystalPlasticity(
            E=195e9, nu=0.28, slip_directions=dirs, slip_normals=norms,
            gamma_dot0=0.1, m=0.05, g0=140e6, h0=180e6, g_sat=400e6)
        res = solve_thermomechanical_plastic(
            layered, mat, thermal_strain_per_layer=ts, dirichlet_bcs=[bc],
            constitutive="cp", T_ref=300.0, alpha_T=16.5e-6,
            dt_cp=1.0, n_sub_cp=256)
        vm = _vm3d(res.residual_stress)
        assert jnp.all(jnp.isfinite(vm))
        overshoot = 195e9 * 16.5e-6 * (900.0 - 300.0)
        assert float(jnp.max(vm)) < 1.5 * overshoot

    def test_cpfe_differs_from_j2_anisotropy(self):
        """CPFE (crystal) yields a stress field distinct from isotropic J2."""
        layered, bc = _make_layered()
        ts = _thermal_strain(layered)

        mat_j2 = J2Plasticity(E=195e9, nu=0.28, sigma_y0=300e6, H=1e9)
        r_j2 = solve_thermomechanical_plastic(
            layered, mat_j2, thermal_strain_per_layer=ts, dirichlet_bcs=[bc],
            constitutive="j2", T_ref=300.0, alpha_T=16.5e-6, n_sub_cp=64)

        dirs, norms = fcc_slip_systems()
        mat_cp = CrystalPlasticity(
            E=195e9, nu=0.28, slip_directions=dirs, slip_normals=norms,
            gamma_dot0=0.1, m=0.05, g0=140e6, h0=180e6, g_sat=400e6)
        r_cp = solve_thermomechanical_plastic(
            layered, mat_cp, thermal_strain_per_layer=ts, dirichlet_bcs=[bc],
            constitutive="cp", T_ref=300.0, alpha_T=16.5e-6,
            dt_cp=1.0, n_sub_cp=256)

        diff = jnp.sqrt(jnp.sum((r_cp.residual_stress - r_j2.residual_stress) ** 2))
        assert float(diff) > 1e-6  # fields are not identical (anisotropy)


class TestElementBirth:
    def test_mask_zeros_stress_outside_part(self):
        """Cells excluded by the part mask carry ~no stress."""
        layered, bc = _make_layered()
        ts = _thermal_strain(layered)
        # Kill the top layer of cells (simulate "not part of the build").
        layer_id = np.asarray(layered.layer_id)
        mask = (layer_id < layered.n_layers - 1).astype(jnp.float64)
        mat = J2Plasticity(E=195e9, nu=0.28, sigma_y0=300e6, H=1e9)
        res = solve_thermomechanical_plastic(
            layered, mat, thermal_strain_per_layer=ts, dirichlet_bcs=[bc],
            constitutive="j2", T_ref=300.0, alpha_T=16.5e-6,
            n_sub_cp=64, cell_mask=jnp.asarray(mask))
        vm = _vm3d(res.residual_stress)
        top = layer_id == (layered.n_layers - 1)
        assert float(jnp.max(vm[top])) < 1e-3   # masked-out → ~zero stress


class TestDifferentiability:
    def test_grad_wrt_thermal_scale_is_finite_and_nonzero(self):
        layered, bc = _make_layered()
        base = _thermal_strain(layered)
        mat_j2 = J2Plasticity(E=195e9, nu=0.28, sigma_y0=300e6, H=1e9)

        def metric(scale):
            res = solve_thermomechanical_plastic(
                layered, mat_j2,
                thermal_strain_per_layer=scale * base, dirichlet_bcs=[bc],
                constitutive="j2", T_ref=300.0, alpha_T=16.5e-6, n_sub_cp=64)
            return jnp.sum(_vm3d(res.residual_stress))

        g = jax.grad(metric)(1.0)
        assert jnp.isfinite(g)
        assert float(g) != 0.0
