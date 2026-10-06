"""Pytest configuration: enable JAX float64 globally for numerical accuracy."""

import jax

jax.config.update("jax_enable_x64", True)
# Use a single device (CPU) and avoid pre-allocation issues during tests.
jax.config.update("jax_platform_name", "cpu")
