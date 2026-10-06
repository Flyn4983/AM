"""Pre-processing: mesh I/O, mesh utilities, and standard load cases.

Provides:

- :mod:`diffmech.preprocess.mesh_io` — import/export meshes via meshio
  (VTK, VTU, Gmsh ``.msh``, Abaqus ``.inp``, Exodus, …), plus convenient
  structured-mesh constructors.
- :mod:`diffmech.preprocess.mesh_tools` — boundary detection, spatial
  selection, element volumes, multi-material region assignment, structured
  refinement.
- :mod:`diffmech.preprocess.load_cases` — generators for common mechanical
  load cases (uniaxial tension/compression, simple shear, biaxial, pure
  bending, gravity), returning ready-to-use :class:`LoadCase` objects.
"""

from diffmech.preprocess.mesh_io import (
    read_mesh, write_mesh, from_meshio, to_meshio,
    supported_meshio_cell_types,
    structured_quad_mesh_2d, structured_tri_mesh_2d,
    structured_hex_mesh_3d, structured_tet_mesh_3d,
)
from diffmech.preprocess.mesh_tools import (
    boundary_node_ids, nodes_on_box_face, nodes_in_box,
    element_volumes, total_volume,
    assign_cell_regions, box_region,
    refine_structured_2d, refine_structured_3d,
)
from diffmech.preprocess.load_cases import (
    LoadCase,
    clamped_face, pin_node,
    uniaxial_tension, uniaxial_compression,
    simple_shear, biaxial, bending, gravity,
)
from diffmech.preprocess.mesh_primitives import (
    disc_mesh_2d, ring_mesh_2d, plate_with_hole_2d,
    l_bracket_mesh_2d, notched_bar_mesh_2d,
    cylinder_mesh_3d, plate_with_hole_3d,
)
from diffmech.preprocess.materials import (
    MaterialLibrary, MaterialParams, AM_MATERIALS, get_material,
)
from diffmech.preprocess.problem_builder import ProblemSetup, ProblemResult

__all__ = [
    # mesh_io
    "read_mesh", "write_mesh", "from_meshio", "to_meshio",
    "supported_meshio_cell_types",
    "structured_quad_mesh_2d", "structured_tri_mesh_2d",
    "structured_hex_mesh_3d", "structured_tet_mesh_3d",
    # mesh_tools
    "boundary_node_ids", "nodes_on_box_face", "nodes_in_box",
    "element_volumes", "total_volume",
    "assign_cell_regions", "box_region",
    "refine_structured_2d", "refine_structured_3d",
    # mesh_primitives
    "disc_mesh_2d", "ring_mesh_2d", "plate_with_hole_2d",
    "l_bracket_mesh_2d", "notched_bar_mesh_2d",
    "cylinder_mesh_3d", "plate_with_hole_3d",
    # load_cases
    "LoadCase",
    "clamped_face", "pin_node",
    "uniaxial_tension", "uniaxial_compression",
    "simple_shear", "biaxial", "bending", "gravity",
    # materials
    "MaterialLibrary", "MaterialParams", "AM_MATERIALS", "get_material",
    # problem_builder
    "ProblemSetup", "ProblemResult",
]
