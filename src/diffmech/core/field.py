"""Lightweight field containers.

A :class:`Field` is just a thin wrapper around a jax array so that the layout
(nodal / cell / particle) is explicit and type-checkable. This avoids accidental
mix-ups (e.g., trying to scatter a cell field with nodal indexing) and gives
clean docstrings.

All fields are differentiable through JAX: gradients propagate through the
underlying array.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp

FieldLayout = Literal["nodal", "cell", "particle"]


@dataclass(frozen=True)
class Field:
    """Base container for a differentiable field.

    Attributes
    ----------
    values : jax.Array
        Shape depends on ``layout``:
            - "nodal"    : (n_nodes, n_components)
            - "cell"     : (n_cells, n_components)  per-element constant field
            - "particle" : (n_particles, n_components)
    layout : str
        One of "nodal", "cell", "particle".
    name : str
        Human-readable name (for IO and error messages).
    """

    values: jax.Array
    layout: FieldLayout
    name: str = "field"

    @property
    def shape(self):
        return self.values.shape

    @property
    def n_components(self) -> int:
        return int(self.values.shape[-1]) if self.values.ndim > 1 else 1

    def __len__(self) -> int:
        return int(self.values.shape[0])


class NodalField(Field):
    def __init__(self, values: jax.Array, name: str = "nodal_field"):
        super().__init__(values=values, layout="nodal", name=name)


class CellField(Field):
    def __init__(self, values: jax.Array, name: str = "cell_field"):
        super().__init__(values=values, layout="cell", name=name)


class ParticleField(Field):
    def __init__(self, values: jax.Array, name: str = "particle_field"):
        super().__init__(values=values, layout="particle", name=name)
