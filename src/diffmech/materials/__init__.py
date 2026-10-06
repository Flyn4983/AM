"""Differentiable constitutive models.

Each model is a *pure function* ``(strain_or_F, params, state) -> (stress, state, tangent)``.
- Small-strain models take the small-strain tensor ``eps`` (symmetric).
- Finite-strain models take the deformation gradient ``F``.
- ``state`` carries internal variables (e.g. plastic strain); pass an empty
  dict for purely elastic models.
- The returned tangent is ``d_stress / d_strain`` (small strain) or the
  algorithmic tangent (finite strain) suitable for use in a Newton solver or
  for ``jax.grad`` through the stress.
"""

from diffmech.materials.linear_elastic import (
    LinearElasticIsotropic, linear_elastic_stress, linear_elastic_tangent,
    youngs_poisson_to_lame,
)
from diffmech.materials.hyperelastic import (
    NeoHookean, neo_hookean_cauchy_stress, neo_hookean_piola_kirchhoff,
)
from diffmech.materials.plasticity import (
    J2Plasticity, j2_plasticity_update,
)

__all__ = [
    "LinearElasticIsotropic", "linear_elastic_stress", "linear_elastic_tangent",
    "youngs_poisson_to_lame",
    "NeoHookean", "neo_hookean_cauchy_stress", "neo_hookean_piola_kirchhoff",
    "J2Plasticity", "j2_plasticity_update",
]
