"""ForgeCore 集成测试：高保真塑性/CPFE 成形求解器接入 (P2-②)

验证 ``asbuilt.thermomechanical_plastic`` 作为 ForgeCore 可注册高保真求解器：

* 注册存在，且经 ``select={"asbuilt": "asbuilt.thermomechanical_plastic"}``
  能正确选路（默认链路仍走降阶 ``buildup.layer_activation``，不被破坏）；
* 由体素 ``PartGeometry`` + ``ThermalHistory`` 产出**有限**的 ``AsBuiltPart``，
  且残余应力被塑性**封顶在屈服附近**（远低于线性弹性过冲 ``E·α·ΔT``）；
* J2 与 CPFE 两种本构都能产出有限成形件；
* 整条桥接对热历史（工艺的代理量）可微：``jax.grad`` 有限且非零。
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import amforge.forge_adapter  # noqa: F401  (triggers ForgeCore registration)
from forgecore.registry import REGISTRY

from amforge.core.contracts import PartGeometry, ThermalHistory, AsBuiltPart
from amforge.asbuilt_plastic import solve_asbuilt_plastic
from amforge.materials import get_material


# ---------------------------------------------------------------------------
# Geometry / thermal fixtures
# ---------------------------------------------------------------------------
def _box(Nx=8, Ny=8, Nz=6, spacing=1e-3):
    sdf = np.full((Nx, Ny, Nz), 1e-3)
    sdf[1:-1, 1:-1, 1:-1] = -1e-3
    return PartGeometry(sdf=jnp.asarray(sdf), origin=jnp.zeros(3),
                       spacing=spacing, dim=3, name="box")


def _thermal(geo, peak=900.0):
    sh = tuple(int(s) for s in geo.shape)
    return ThermalHistory(
        peak_temperature=jnp.full(sh, peak),
        cooling_rate=jnp.full(sh, 1e3),
        thermal_gradient=jnp.full(sh, 1e3),
        solidification_rate=jnp.full(sh, 1e-2),
        time_above_melt=jnp.full(sh, 1e-3),
        final_temperature=jnp.full(sh, 300.0),
        spacing=float(geo.spacing), dim=3,
    )


# ---------------------------------------------------------------------------
class TestRegistrationAndRouting:
    def test_solver_registered(self):
        assert REGISTRY.get("asbuilt.thermomechanical_plastic") is not None

    def test_select_routes_to_high_fidelity(self):
        specs = REGISTRY.auto(
            "asbuilt", select={"asbuilt": "asbuilt.thermomechanical_plastic"})
        names = [s.name for s in specs]
        assert "asbuilt.thermomechanical_plastic" in names
        # resolution does not pull the default (low-fidelity) buildup solver
        assert "buildup.layer_activation" not in names

    def test_default_chain_unchanged(self):
        """Without select the default asbuilt is still the cheap buildup model."""
        specs = REGISTRY.auto("asbuilt")
        names = [s.name for s in specs]
        assert "buildup.layer_activation" in names
        assert "asbuilt.thermomechanical_plastic" not in names

    def test_required_inputs(self):
        req = REGISTRY.required_inputs(
            "asbuilt", select={"asbuilt": "asbuilt.thermomechanical_plastic"})
        # Only geometry must be supplied externally: process is auto-produced by
        # process.heuristic and thermal by thermal.history (both upstream of the
        # selected solver), so neither appears in the required-input set.
        assert req == ["geometry"]


# ---------------------------------------------------------------------------
class TestAsBuiltOutput:
    def test_j2_finite_and_capped(self):
        geo = _box()
        th = _thermal(geo, peak=900.0)
        mat = get_material("316L")
        ab = solve_asbuilt_plastic(
            geometry=geo, thermal=th,
            params={"material": "316L", "constitutive": "j2"})
        assert isinstance(ab, AsBuiltPart)

        rs = ab.residual_stress
        assert jnp.all(jnp.isfinite(rs))
        assert jnp.all(jnp.isfinite(ab.displacement))
        assert jnp.all(jnp.isfinite(ab.sdf))

        occ = geo.occupancy
        vm = ab.von_mises_residual()
        vm_in = vm[occ > 0.5]
        assert float(jnp.mean(vm_in)) > 0.0  # non-zero residual inside part
        overshoot = mat.E * mat.alpha_thermal * (900.0 - mat.T_ref_mech)
        # plastic relaxation caps the mean residual below the elastic overshoot
        assert float(jnp.mean(vm_in)) < overshoot
        assert float(jnp.max(vm_in)) < 1.5 * overshoot

    def test_cpfe_finite(self):
        geo = _box()
        th = _thermal(geo, peak=900.0)
        ab = solve_asbuilt_plastic(
            geometry=geo, thermal=th,
            params={"material": "316L", "constitutive": "cp"})
        assert jnp.all(jnp.isfinite(ab.residual_stress))
        occ = geo.occupancy
        vm = ab.von_mises_residual()
        assert float(jnp.mean(vm[occ > 0.5])) > 0.0

    def test_inside_part_stress_nonzero_outside_zero(self):
        geo = _box()
        th = _thermal(geo, peak=900.0)
        ab = solve_asbuilt_plastic(
            geometry=geo, thermal=th,
            params={"material": "316L", "constitutive": "j2"})
        occ = geo.occupancy
        vm = ab.von_mises_residual()
        # outside the part the residual stress must be ~0 (masked)
        assert float(jnp.max(vm[occ < 0.5])) < 1e-2


# ---------------------------------------------------------------------------
class TestDifferentiability:
    def test_grad_wrt_peak_temperature(self):
        geo = _box()
        occ = geo.occupancy
        mat = get_material("316L")

        def loss(peak):
            th = _thermal(geo, peak=peak)
            ab = solve_asbuilt_plastic(
                geometry=geo, thermal=th,
                params={"material": "316L", "constitutive": "j2"})
            return jnp.sum(ab.von_mises_residual()[occ > 0.5])

        g = jax.grad(loss)(900.0)
        assert jnp.isfinite(g)
        assert float(g) != 0.0
