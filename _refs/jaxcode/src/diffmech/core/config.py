"""Global configuration (dataclass, mutable for experiments).

Most DiffMech functions accept a ``config`` keyword that controls floating-point
precision, JIT behavior, and device placement. Defaults are picked to be safe
(float64 on CPU) so that physics tests are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax


@dataclass
class Config:
    """Runtime configuration for DiffMech.

    Attributes
    ----------
    dtype : str
        Either ``"float32"`` or ``"float64"``. Float64 is required for many
        solid-mechanics patch tests; we enable the X64 flag automatically when
        the user requests float64.
    jit : bool
        Whether to JIT-compile hot loops. Set to ``False`` for debugging.
    """

    dtype: str = "float64"
    jit: bool = True

    def __post_init__(self):
        if self.dtype == "float64":
            jax.config.update("jax_enable_x64", True)

    @property
    def jax_dtype(self):
        return {"float32": jax.numpy.float32, "float64": jax.numpy.float64}[self.dtype]


def default_config() -> Config:
    return Config()
