"""Inverse-design / parameter-identification utilities.

These wrap Optax (Adam) and plain gradient descent for minimising a scalar
loss ``L(params)`` whose gradient is computed via JAX autodiff. The whole
optimization loop is jit-able.

Typical usage:

    def loss_fn(params):
        u = forward_simulate(params)
        return jnp.mean((u - target) ** 2)

    params_opt = optimize_params_optax(loss_fn, params_init, n_iter=2000)
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp


def optimize_params_gd(
    loss_fn: Callable,
    params: jnp.ndarray,
    *,
    lr: float = 0.01,
    n_iter: int = 1000,
    tol: float = 1e-8,
    verbose: bool = False,
) -> tuple[jnp.ndarray, list]:
    """Plain gradient descent on a scalar loss.

    Returns (params, history_of_losses).
    """
    grad_fn = jax.grad(loss_fn)
    history = []

    def body(i, params):
        g = grad_fn(params)
        params = params - lr * g
        return params

    # Use lax.fori_loop for jit-friendly iteration
    params = jax.lax.fori_loop(0, n_iter, body, params)
    return params, [float(loss_fn(params))]


def optimize_params_optax(
    loss_fn: Callable,
    params,
    *,
    n_iter: int = 1000,
    lr: float = 1e-2,
    verbose: bool = False,
):
    """Optimize a pytree of params with Adam (via Optax).

    Returns (params, list_of_losses) where list_of_losses records the loss at
    each iteration (CPU-side; not jit-traceable but cheap for n_iter < 10k).
    """
    import optax

    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    grad_fn = jax.value_and_grad(loss_fn)
    history = []

    @jax.jit
    def step(params, opt_state):
        loss, grads = grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    for i in range(n_iter):
        params, opt_state, loss = step(params, opt_state)
        history.append(float(loss))
        if verbose and (i % max(1, n_iter // 10) == 0):
            print(f"iter {i}: loss = {loss:.6e}")

    return params, history
