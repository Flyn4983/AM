"""Core building blocks: differentiable meshes, quadrature, shape functions, fields."""

from diffmech.core.mesh import (
    Mesh, StructuredMesh2D, StructuredMesh3D,
    rectangular_mesh2d, rectangular_mesh3d,
)
from diffmech.core.quadrature import gauss_legendre_1d, gauss_legendre_nd, QuadratureRule
from diffmech.core.shape_functions import (
    ShapeFunction,
    tri3_shape, tri3_grad,
    quad4_shape, quad4_grad,
    tet4_shape, tet4_grad,
    hex8_shape, hex8_grad,
    get_shape_function, physical_gradient, physical_gradient_and_det,
    SHAPE_FUNCTIONS,
)
from diffmech.core.field import Field, NodalField, CellField, ParticleField
from diffmech.core.config import Config, default_config

__all__ = [
    "Mesh", "StructuredMesh2D", "StructuredMesh3D",
    "rectangular_mesh2d", "rectangular_mesh3d",
    "gauss_legendre_1d", "gauss_legendre_nd", "QuadratureRule",
    "ShapeFunction",
    "tri3_shape", "tri3_grad",
    "quad4_shape", "quad4_grad",
    "tet4_shape", "tet4_grad",
    "hex8_shape", "hex8_grad",
    "get_shape_function", "physical_gradient", "physical_gradient_and_det",
    "SHAPE_FUNCTIONS",
    "Field", "NodalField", "CellField", "ParticleField",
    "Config", "default_config",
]
