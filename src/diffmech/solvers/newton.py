"""Newton-Raphson solver for nonlinear systems R(u) = 0.

Two variants:
    - :func:`newton_solve` uses ``jax.lax.while_loop`` for early termination.
      Efficient but NOT differentiable via reverse-mode AD (JAX does not allow
      ``grad`` through ``while_loop`` with dynamic stop).
    - :func:`newton_solve_diff` uses ``jax.lax.scan`` with a fixed number of
      iterations. Slightly wasteful (continues iterating after convergence)
      but supports ``jax.grad`` end-to-end. Use this when differentiating
      through the nonlinear solve (e.g. for material parameter identification).

For very large systems, ``jnp.linalg.solve`` should be replaced with a sparse
or iterative solver (e.g. ``jax.scipy.sparse.linalg.cg``).
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp


class NewtonResult(NamedTuple):
    u: jnp.ndarray
    converged: jnp.ndarray  # 0-d bool array (use bool(...) for Python bool)
    n_iter: jnp.ndarray  # 0-d int array
    final_residual_norm: jnp.ndarray  # 0-d float array


def _newton_step(R, jac, u):
    """One Newton iteration: u_new = u - J^{-1} R(u)."""
    Ru = R(u)
    Ju = jac(u)
    if u.ndim == 0:
        du = -Ru / Ju
    elif u.ndim == 1:
        Ju_mat = Ju if Ju.ndim == 2 else Ju.reshape(1, 1)
        du = jnp.linalg.solve(Ju_mat, -Ru.reshape(-1)).reshape(Ru.shape)
    else:
        du = jnp.linalg.solve(Ju, -Ru)
    return u + du


def newton_solve(
    R: Callable[[jnp.ndarray], jnp.ndarray],
    u0: jnp.ndarray,
    *,
    jac: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    tol: float = 1e-10,
    max_iter: int = 30,
    verbose: bool = False,
) -> NewtonResult:
    """Solve R(u) = 0 via Newton-Raphson (NOT differentiable; efficient).

    Use :func:`newton_solve_diff` if you need to differentiate through the solve.
    """
    if jac is None:
        jac = jax.jacrev(R)

    def cond(state):
        u, it, res_norm, converged = state
        return jnp.logical_and(jnp.logical_and(it < max_iter, ~converged),
                                res_norm > tol)

    def body(state):
        u, it, res_norm, converged = state
        u_new = _newton_step(R, jac, u)
        new_norm = jnp.linalg.norm(R(u_new))
        if verbose:
            jax.debug.print("iter {it}: res_norm = {r}", it=it, r=new_norm)
        return u_new, it + 1, new_norm, new_norm < tol

    r0_norm = jnp.linalg.norm(R(u0))
    state0 = (u0, jnp.array(0), r0_norm, r0_norm < tol)
    final = jax.lax.while_loop(cond, body, state0)
    return NewtonResult(
        u=final[0], converged=final[3],
        n_iter=final[1], final_residual_norm=final[2],
    )


def newton_solve_diff(
    R: Callable[[jnp.ndarray], jnp.ndarray],
    u0: jnp.ndarray,
    *,
    jac: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    tol: float = 1e-10,
    max_iter: int = 30,
) -> NewtonResult:
    """Differentiable Newton via ``jax.lax.scan`` with fixed iterations.

    All ``max_iter`` iterations are performed; each iteration uses ``jnp.where``
    to keep ``u`` fixed once convergence is achieved. This is slightly wasteful
    but is reverse-mode-AD-friendly (no dynamic loop bounds).
    """
    if jac is None:
        jac = jax.jacrev(R)

    def scan_step(carry, _):
        u, res_norm, converged = carry
        # Compute Newton step; if already converged, du = 0.
        u_new = _newton_step(R, jac, u)
        new_norm = jnp.linalg.norm(R(u_new))
        new_converged = jnp.logical_or(converged, new_norm < tol)
        # If already converged, keep u unchanged.
        u_out = jnp.where(converged, u, u_new)
        return (u_out, new_norm, new_converged), None

    r0 = jnp.linalg.norm(R(u0))
    init = (u0, r0, r0 < tol)
    (u_final, res_final, conv_final), _ = jax.lax.scan(
        scan_step, init, xs=None, length=max_iter,
    )
    return NewtonResult(
        u=u_final, converged=conv_final,
        n_iter=jnp.array(max_iter), final_residual_norm=res_final,
    )


def newton_solve_with_jacrev(R, u0, **kwargs):
    """Convenience alias for ``newton_solve(R, u0, jac=None, ...)``."""
    return newton_solve(R, u0, **kwargs)
