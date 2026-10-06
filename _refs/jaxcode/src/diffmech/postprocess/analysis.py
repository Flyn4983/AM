"""Simulation-result analysis: error norms, convergence, comparison, reports.

This module bundles the post-processing utilities that are most useful for
*verifying* a simulation — computing error norms against a reference
solution, measuring convergence under mesh/time-step refinement, comparing
two runs, and emitting a human-readable summary report.

All routines accept plain NumPy or JAX arrays; nothing here is jit-restricted,
so they can be called inside ``jax.grad`` objectives (e.g. to *minimise* an
L2 error against experimental data) but are equally usable as plain Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Error norms
# ---------------------------------------------------------------------------
def l1_norm(field: np.ndarray, reference: np.ndarray | None = None,
            *, weights: np.ndarray | None = None) -> float:
    """L1 (mean-absolute) error / norm.

    If ``reference`` is None the L1 norm ``Σ |field|`` (or weighted) is
    returned; otherwise the L1 error ``Σ |field − reference|`` is returned.
    """
    f = np.asarray(field, dtype=np.float64)
    if reference is None:
        diff = f
    else:
        diff = f - np.asarray(reference, dtype=np.float64)
    if weights is None:
        return float(np.mean(np.abs(diff)))
    w = np.asarray(weights, dtype=np.float64)
    return float(np.sum(w * np.abs(diff)) / np.sum(w))


def l2_norm(field: np.ndarray, reference: np.ndarray | None = None,
            *, weights: np.ndarray | None = None) -> float:
    """L2 (root-mean-square) error / norm."""
    f = np.asarray(field, dtype=np.float64)
    diff = f if reference is None else f - np.asarray(reference, dtype=np.float64)
    sq = diff * diff
    if weights is None:
        return float(np.sqrt(np.mean(sq)))
    w = np.asarray(weights, dtype=np.float64)
    return float(np.sqrt(np.sum(w * sq) / np.sum(w)))


def linf_norm(field: np.ndarray, reference: np.ndarray | None = None) -> float:
    """L-infinity (max-absolute) error / norm."""
    f = np.asarray(field, dtype=np.float64)
    diff = f if reference is None else f - np.asarray(reference, dtype=np.float64)
    return float(np.max(np.abs(diff)))


def relative_error(field: np.ndarray, reference: np.ndarray,
                    *, norm: str = "l2") -> float:
    """Relative error ``||field − reference|| / ||reference||``.

    ``norm`` is one of ``"l1"``, ``"l2"``, ``"linf"``.
    """
    ref = np.asarray(reference, dtype=np.float64)
    denom_funcs = {"l1": l1_norm, "l2": l2_norm, "linf": linf_norm}
    if norm not in denom_funcs:
        raise ValueError(f"norm must be 'l1', 'l2', or 'linf', got {norm!r}")
    denom = denom_funcs[norm](ref)
    if denom < 1e-30:
        return float("inf")
    num = denom_funcs[norm](field, ref)
    return num / denom


def error_report(field: np.ndarray, reference: np.ndarray,
                 *, weights: np.ndarray | None = None,
                 name: str = "field") -> dict:
    """Compute L1/L2/Linf + relative errors in one call.

    Returns a dict suitable for logging or printing.
    """
    return {
        "name": name,
        "l1_abs": l1_norm(field, reference, weights=weights),
        "l2_abs": l2_norm(field, reference, weights=weights),
        "linf_abs": linf_norm(field, reference),
        "l1_rel": relative_error(field, reference, norm="l1"),
        "l2_rel": relative_error(field, reference, norm="l2"),
        "linf_rel": relative_error(field, reference, norm="linf"),
    }


# ---------------------------------------------------------------------------
# Convergence analysis
# ---------------------------------------------------------------------------
@dataclass
class ConvergenceResult:
    """Outcome of a mesh / time-step refinement convergence study."""

    h_values: list[float]              # refinement parameters (e.g. dx, dt)
    errors: list[float]               # corresponding error norms
    rate: float                       # fitted convergence rate p (E ~ C h^p)
    constant: float                   # fitted leading constant C
    norm: str                         # which error norm was used

    def __repr__(self) -> str:
        return (f"ConvergenceResult(rate={self.rate:.3f}, C={self.constant:.4g}, "
                f"norm={self.norm!r}, n_points={len(self.h_values)})")


def fit_convergence_rate(
    h_values: Sequence[float],
    errors: Sequence[float],
) -> tuple[float, float]:
    """Fit ``E = C * h^p`` (in log-log space) and return ``(p, C)``.

    ``h_values`` are the refinement parameters (e.g. mesh size dx, or time
    step dt) and ``errors`` the corresponding error norms. At least two
    points are required.
    """
    h = np.asarray(h_values, dtype=np.float64)
    e = np.asarray(errors, dtype=np.float64)
    if h.size < 2:
        raise ValueError("need at least two (h, error) points to fit a rate")
    if np.any(h <= 0) or np.any(e <= 0):
        raise ValueError("h_values and errors must all be strictly positive")
    log_h = np.log(h)
    log_e = np.log(e)
    # Linear regression: log_e = log_C + p * log_h
    A = np.vstack([np.ones_like(log_h), log_h]).T
    log_C, p = np.linalg.lstsq(A, log_e, rcond=None)[0]
    return float(p), float(np.exp(log_C))


def convergence_study(
    h_values: Sequence[float],
    errors: Sequence[float],
    *,
    norm: str = "l2",
) -> ConvergenceResult:
    """Run a full convergence study: fit rate + assemble result object."""
    p, C = fit_convergence_rate(h_values, errors)
    return ConvergenceResult(
        h_values=list(h_values), errors=list(errors),
        rate=p, constant=C, norm=norm,
    )


def convergence_table(
    h_values: Sequence[float],
    errors: Sequence[float],
    *,
    name: str = "error",
) -> str:
    """Pretty-print a (h, error, rate) convergence table as a string."""
    h = list(h_values)
    e = list(errors)
    if len(h) != len(e):
        raise ValueError("h_values and errors must have the same length")
    p, _ = fit_convergence_rate(h, e) if len(h) >= 2 else (float("nan"), 0.0)
    lines = [f"{'h':>12s}  {name:>14s}  {'rate':>8s}"]
    lines.append("-" * len(lines[0]))
    for i, (hi, ei) in enumerate(zip(h, e)):
        if i == 0:
            rate_str = "  --"
        else:
            local_p = np.log(e[i - 1] / ei) / np.log(h[i - 1] / hi) if hi > 0 else float("nan")
            rate_str = f"{local_p:8.3f}"
        lines.append(f"{hi:12.6e}  {ei:14.6e}  {rate_str}")
    lines.append("-" * len(lines[0]))
    lines.append(f"overall rate p = {p:.4f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Solution comparison
# ---------------------------------------------------------------------------
@dataclass
class ComparisonReport:
    """Pairwise comparison of two field solutions."""

    name_a: str
    name_b: str
    l2_abs: float
    linf_abs: float
    l2_rel: float
    max_location: tuple[int, ...] = ()
    correlation: float = 1.0

    def to_dict(self) -> dict:
        return {
            "name_a": self.name_a, "name_b": self.name_b,
            "l2_abs": self.l2_abs, "linf_abs": self.linf_abs,
            "l2_rel": self.l2_rel, "max_location": self.max_location,
            "correlation": self.correlation,
        }


def compare_solutions(
    field_a: np.ndarray,
    field_b: np.ndarray,
    *,
    name_a: str = "A",
    name_b: str = "B",
) -> ComparisonReport:
    """Compare two field solutions and report the dominant discrepancies."""
    a = np.asarray(field_a, dtype=np.float64)
    b = np.asarray(field_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(
            f"shape mismatch: {a.shape} vs {b.shape}; interpolate first"
        )
    diff = a - b
    max_loc = tuple(int(i) for i in np.unravel_index(np.argmax(np.abs(diff)), diff.shape))
    # Pearson correlation (flattened)
    af = a.ravel()
    bf = b.ravel()
    if af.size > 1 and np.std(af) > 1e-30 and np.std(bf) > 1e-30:
        corr = float(np.corrcoef(af, bf)[0, 1])
    else:
        corr = 1.0 if np.allclose(af, bf) else 0.0
    denom = l2_norm(b)
    l2_abs_val = l2_norm(a, b)
    l2_rel_val = l2_abs_val / denom if denom > 1e-30 else float("inf")
    return ComparisonReport(
        name_a=name_a, name_b=name_b,
        l2_abs=l2_abs_val, linf_abs=linf_norm(a, b),
        l2_rel=l2_rel_val, max_location=max_loc, correlation=corr,
    )


# ---------------------------------------------------------------------------
# Summary report (text)
# ---------------------------------------------------------------------------
def summary_report(
    title: str,
    sections: dict[str, dict | str | Iterable],
    *,
    width: int = 72,
) -> str:
    """Render a human-readable plain-text summary report.

    ``sections`` is an ordered dict of ``{section_title: content}`` where
    ``content`` is either:
      * a dict of ``{key: value}`` (rendered as a key/value table),
      * a string (rendered verbatim),
      * an iterable of strings (rendered as bullet points).
    """
    bar = "=" * width
    lines = [bar, title.center(width), bar, ""]
    for sec_title, content in sections.items():
        lines.append(f"## {sec_title}")
        lines.append("-" * width)
        if isinstance(content, str):
            lines.append(content)
        elif isinstance(content, dict):
            for k, v in content.items():
                if isinstance(v, float):
                    lines.append(f"  {k:<32s} : {v:.6e}")
                else:
                    lines.append(f"  {k:<32s} : {v}")
        else:
            try:
                for item in content:
                    lines.append(f"  - {item}")
            except TypeError:
                lines.append(f"  {content}")
        lines.append("")
    lines.append(bar)
    return "\n".join(lines)


__all__ = [
    "l1_norm", "l2_norm", "linf_norm", "relative_error", "error_report",
    "fit_convergence_rate", "convergence_study", "convergence_table",
    "ConvergenceResult",
    "compare_solutions", "ComparisonReport",
    "summary_report",
]
