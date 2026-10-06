"""DiffMech: a JAX-based differentiable computational mechanics suite.

Subpackages
-----------
core          : differentiable meshes, particles, fields, quadrature, shape functions
materials     : constitutive models (linear elastic, hyperelastic, plasticity, crystal plasticity)
methods       : numerical methods (FEM, FVM, DEM, SPH, MPM, phase field, crystal plasticity FEM)
solvers       : Newton, time integrators, constraint handling
optimize      : inverse design and parameter identification via jax.grad / optax
preprocess    : mesh I/O (meshio), boundary detection, multi-material regions, load cases
postprocess   : VTK/VTU/PVD writers, matplotlib visualisation, field probes, time-series logging
"""

__version__ = "0.1.0"

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("diffmech")
except PackageNotFoundError:  # pragma: no cover - dev install
    pass

# Make subpackages importable as ``diffmech.preprocess`` / ``diffmech.postprocess``.
from diffmech import preprocess, postprocess  # noqa: E402  (re-export)

__all__ = ["__version__", "preprocess", "postprocess"]
