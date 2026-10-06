"""Tests for the AM material library."""

from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from diffmech.preprocess.materials import (
    MaterialParams, MaterialLibrary, AM_MATERIALS, get_material,
)


# ---------------------------------------------------------------------------
# Library contents
# ---------------------------------------------------------------------------
EXPECTED_ALLOYS = {"Ti6Al4V", "IN718", "AlSi10Mg", "316L", "IN625"}


def test_library_contains_expected_alloys():
    lib = MaterialLibrary()
    for name in EXPECTED_ALLOYS:
        assert name in lib


def test_get_material_shortcut():
    ti = get_material("Ti6Al4V")
    assert isinstance(ti, MaterialParams)
    assert ti.name == "Ti6Al4V"


def test_get_unknown_material_raises():
    with pytest.raises(KeyError):
        get_material("Unobtainium")


def test_library_names_sorted():
    lib = MaterialLibrary()
    names = lib.names()
    assert names == sorted(names)
    assert len(names) == len(AM_MATERIALS)


# ---------------------------------------------------------------------------
# MaterialParams integrity
# ---------------------------------------------------------------------------
def test_ti6al4v_physical_sanity():
    ti = get_material("Ti6Al4V")
    # Young's modulus in a sensible range for Ti64 (~100-120 GPa)
    assert 90e9 < ti.E < 130e9
    assert 0.25 < ti.nu < 0.35
    assert ti.sigma_y > 800e6          # high-strength alloy
    assert ti.lattice == "hcp"
    # Thermal diffusivity positive and finite
    assert ti.alpha > 0.0
    assert ti.rho_cp > 0.0
    assert ti.T_melt > 1500.0


def test_lame_constants_consistent():
    for name in EXPECTED_ALLOYS:
        m = get_material(name)
        # mu = E / (2 (1 + nu))
        assert m.mu == pytest.approx(m.E / (2 * (1 + m.nu)), rel=1e-9)
        # lam = E nu / ((1+nu)(1-2nu))
        assert m.lam == pytest.approx(
            m.E * m.nu / ((1 + m.nu) * (1 - 2 * m.nu)), rel=1e-9,
        )


def test_alpha_equals_k_over_rho_cp():
    for name in EXPECTED_ALLOYS:
        m = get_material(name)
        assert m.alpha == pytest.approx(m.k_cond / (m.rho * m.cp), rel=1e-9)


def test_material_has_notes():
    for name in EXPECTED_ALLOYS:
        assert get_material(name).notes  # non-empty provenance


# ---------------------------------------------------------------------------
# JAX conversion + differentiability
# ---------------------------------------------------------------------------
def test_as_jax_returns_jax_arrays():
    ti = get_material("Ti6Al4V")
    tj = ti.as_jax()
    assert isinstance(tj.E, jnp.ndarray)
    assert isinstance(tj.sigma_y, jnp.ndarray)
    assert isinstance(tj.alpha, jnp.ndarray) if hasattr(tj, "alpha") or True else True
    # Values preserved
    assert float(tj.E) == pytest.approx(ti.E)


def test_material_params_differentiable():
    """Objective using material parameters must be differentiable."""
    ti = get_material("Ti6Al4V")

    def objective(E):
        # Trivial surrogate: stored elastic energy ~ E * eps^2 / 2
        eps = 1e-3
        return 0.5 * E * eps * eps

    grad = jax.grad(objective)(jnp.asarray(ti.E))
    assert bool(jnp.isfinite(grad))
    assert float(grad) == pytest.approx(0.5e-6, rel=1e-6)


# ---------------------------------------------------------------------------
# Library extension
# ---------------------------------------------------------------------------
def test_add_custom_material():
    lib = MaterialLibrary()
    n0 = len(lib)
    custom = MaterialParams(
        name="CustomAlloy", E=70e9, nu=0.33, sigma_y=200e6,
        k_cond=130.0, rho=2700.0, cp=900.0, T_melt=900.0, lattice="fcc",
    )
    lib.add(custom)
    assert "CustomAlloy" in lib
    assert len(lib) == n0 + 1
    assert lib["CustomAlloy"].E == 70e9


def test_add_anonymous_material_raises():
    lib = MaterialLibrary()
    with pytest.raises(ValueError):
        lib.add(MaterialParams())   # no name


def test_library_iteration():
    lib = MaterialLibrary()
    mats = list(lib)
    assert len(mats) == len(lib)
    assert all(isinstance(m, MaterialParams) for m in mats)
