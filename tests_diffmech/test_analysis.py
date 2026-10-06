"""Tests for the post-processing analysis module."""

from __future__ import annotations

import numpy as np
import pytest

from diffmech.postprocess import (
    l1_norm, l2_norm, linf_norm, relative_error, error_report,
    fit_convergence_rate, convergence_study, convergence_table,
    ConvergenceResult,
    compare_solutions, ComparisonReport,
    summary_report,
)


# ---------------------------------------------------------------------------
# Error norms
# ---------------------------------------------------------------------------
def test_l2_norm_zero_for_identical_fields():
    a = np.array([1.0, 2.0, 3.0])
    assert l2_norm(a, a) == pytest.approx(0.0, abs=1e-15)


def test_l2_norm_known_value():
    a = np.array([1.0, 2.0, 3.0])
    # ||a|| = sqrt((1+4+9)/3)
    assert l2_norm(a) == pytest.approx(np.sqrt(14 / 3), rel=1e-12)


def test_l1_and_linf_norms():
    a = np.array([1.0, -2.0, 3.0])
    b = np.array([1.0, 0.0, 0.0])
    # |a-b| = [0, 2, 3] -> mean=5/3, max=3
    assert l1_norm(a, b) == pytest.approx(5 / 3, rel=1e-12)
    assert linf_norm(a, b) == pytest.approx(3.0, rel=1e-12)


def test_weighted_l2_norm():
    a = np.array([0.0, 0.0, 4.0])
    b = np.zeros(3)
    w = np.array([1.0, 1.0, 9.0])  # weight the large-error cell heavily
    unweighted = l2_norm(a, b)
    weighted = l2_norm(a, b, weights=w)
    assert weighted > unweighted


def test_relative_error_zero_reference_is_inf():
    a = np.array([1.0, 0.0])
    b = np.zeros(2)
    assert relative_error(a, b) == float("inf")


def test_relative_error_norms():
    a = np.array([2.0, 0.0])
    b = np.array([1.0, 0.0])
    # l2_rel = ||a-b|| / ||b|| = 1/1 = 1
    assert relative_error(a, b, norm="l2") == pytest.approx(1.0)
    assert relative_error(a, b, norm="linf") == pytest.approx(1.0)


def test_error_report_keys():
    rep = error_report(np.array([1.0]), np.array([0.0]), name="disp")
    for k in ("name", "l1_abs", "l2_abs", "linf_abs",
              "l1_rel", "l2_rel", "linf_rel"):
        assert k in rep
    assert rep["name"] == "disp"


# ---------------------------------------------------------------------------
# Convergence fitting
# ---------------------------------------------------------------------------
def test_fit_second_order_rate():
    h = [0.1, 0.05, 0.025, 0.0125]
    e = [c * h_ ** 2 for h_, c in zip(h, [1.0] * 4)]
    p, C = fit_convergence_rate(h, e)
    assert p == pytest.approx(2.0, abs=1e-9)
    assert C == pytest.approx(1.0, abs=1e-9)


def test_fit_first_order_rate():
    h = [1.0, 0.5, 0.25, 0.125]
    e = [2.0 * h_ for h_ in h]
    p, _ = fit_convergence_rate(h, e)
    assert p == pytest.approx(1.0, abs=1e-9)


def test_fit_requires_two_points():
    with pytest.raises(ValueError):
        fit_convergence_rate([0.1], [0.01])


def test_fit_rejects_nonpositive():
    with pytest.raises(ValueError):
        fit_convergence_rate([0.1, 0.0], [0.01, 0.0])
    with pytest.raises(ValueError):
        fit_convergence_rate([0.1, -0.1], [0.01, 0.02])


def test_convergence_study_result():
    h = [0.2, 0.1, 0.05]
    e = [0.04, 0.01, 0.0025]
    res = convergence_study(h, e, norm="l2")
    assert isinstance(res, ConvergenceResult)
    assert res.rate == pytest.approx(2.0, abs=1e-9)
    assert res.h_values == h
    assert res.errors == e
    assert res.norm == "l2"


def test_convergence_table_contains_rate_and_header():
    h = [0.2, 0.1, 0.05]
    e = [0.04, 0.01, 0.0025]
    table = convergence_table(h, e, name="L2")
    assert "h" in table and "L2" in table and "rate" in table
    assert "overall rate p = 2.0000" in table


# ---------------------------------------------------------------------------
# Solution comparison
# ---------------------------------------------------------------------------
def test_compare_identical_solutions():
    a = np.array([1.0, 2.0, 3.0])
    rep = compare_solutions(a, a)
    assert rep.l2_abs == pytest.approx(0.0, abs=1e-15)
    assert rep.linf_abs == pytest.approx(0.0, abs=1e-15)
    assert rep.correlation == pytest.approx(1.0)


def test_compare_different_solutions():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 4.0, 6.0])
    rep = compare_solutions(a, b, name_a="sim", name_b="ref")
    assert rep.name_a == "sim" and rep.name_b == "ref"
    assert rep.l2_abs > 0.0
    assert rep.correlation == pytest.approx(1.0)  # perfectly correlated
    # max location should be a valid index tuple
    assert len(rep.max_location) == a.ndim


def test_compare_shape_mismatch_raises():
    with pytest.raises(ValueError):
        compare_solutions(np.zeros(3), np.zeros(4))


def test_comparison_report_to_dict():
    rep = compare_solutions(np.array([1.0]), np.array([0.0]))
    d = rep.to_dict()
    assert "l2_abs" in d and "correlation" in d


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------
def test_summary_report_contains_title_and_sections():
    rep = summary_report(
        "My Report",
        {"errors": {"l2": 0.01, "linf": 0.05},
         "notes": "all good",
         "bullets": ["a", "b"]},
    )
    assert "My Report" in rep
    assert "errors" in rep and "notes" in rep and "bullets" in rep
    # Floats are rendered in scientific notation (1.000000e-02).
    assert "l2" in rep and "1.000000e-02" in rep
    assert "- a" in rep and "- b" in rep


def test_summary_report_handles_floats_and_strings():
    rep = summary_report("T", {"sec": {"k": 1.5, "name": "foo"}})
    assert "1.500000e+00" in rep
    assert "foo" in rep
