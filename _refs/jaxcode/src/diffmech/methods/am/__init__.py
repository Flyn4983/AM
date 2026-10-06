"""Additive Manufacturing (AM) process simulation.

Differentiable layer-by-layer AM process modelling covering both **Selective
Laser Melting (SLM)** and **Laser Solid Forming (LSF)**. The whole build
history — geometry slicing, scan-path generation, moving heat source, element
activation, thermo-mechanical coupling — is expressed with ``jax`` so that
``jax.grad`` propagates from process parameters (laser power, scan speed, layer
thickness, …) to residual stress and part distortion.

Pipeline
--------
1. :mod:`geometry`        — differentiable part geometry via SDFs + slicing.
2. :mod:`scan_paths`      — zigzag / contour / spiral laser trajectories.
3. :mod:`process`         — SLM / LSF process-parameter configs.
4. :mod:`activation`      — layered mesh + smooth element-activation field.
5. :mod:`am_thermal`     — moving Gaussian heat source gated by activation.
6. :mod:`thermomechanical` — layer-by-layer residual stress & distortion.
"""

from diffmech.methods.am import (
    geometry, scan_paths, process, activation, am_thermal, thermomechanical,
    particle_am, am_microstructure,
)

# Geometry
from diffmech.methods.am.geometry import (
    SDF, smooth_union, smooth_intersection, smooth_difference,
    union, intersection, difference, translate, rotate_z,
    sdf_sphere, sdf_box, sdf_cylinder, sdf_torus, sdf_gear,
    slice_mask, layer_heights, part_bbox,
    sdf_from_mesh, sdf_from_stl, sdf_from_obj,
)
# Scan paths
from diffmech.methods.am.scan_paths import (
    ScanPath, MultiLayerPath, multi_layer_paths,
    zigzag_hatch, contour_hatch, spiral_hatch,
    from_waypoints, chain_paths, from_csv, from_gcode,
)
# Process configs
from diffmech.methods.am.process import (
    SLMConfig, LSFConfig, layer_time, layer_activation_times,
)
# Activation
from diffmech.methods.am.activation import (
    LayeredMesh, build_layered_mesh, assign_activation_times, part_cell_mask,
)
# Thermal
from diffmech.methods.am.am_thermal import (
    laser_position, am_heat_source, AMThermalProblem, setup_am_thermal,
)
# Thermo-mechanical
from diffmech.methods.am.thermomechanical import (
    AMThermoMechResult, solve_thermomechanical,
    assemble_activated_stiffness, thermal_strain_force,
)
# Particle AM (DEM / MPM / SPH)
from diffmech.methods.am.particle_am import (
    sdf_circle_2d, sdf_box_2d, sdf_from_polygon_2d, sdf_from_stl_mesh_3d,
    parse_binary_stl,
    ParticleAMState, ParticleAMProblem, setup_particle_am,
    place_particles_in_sdf,
    assign_particle_activation_times, particle_layer_times,
    laser_heat_on_particles,
    step_dem_am, step_dem_am_scan,
    step_mpm_am, step_mpm_am_scan,
    step_sph_am, step_sph_am_scan,
    peak_temperature, activated_fraction, mean_displacement,
)
# AM microstructure (phase-field solidification × thermal history)
from diffmech.methods.am.am_microstructure import (
    ThermalField, MicrostructureConfig, AMMicrostructureState,
    make_microstructure_state,
    thermal_undercooling, grain_mobility,
    solidification_rhs, orientation_rhs,
    step_am_microstructure, step_am_microstructure_scan,
    solidified_fraction, nucleated_fraction,
    mean_grain_size, grain_count_estimate,
    cooling_rate, morphology_indicator, thermal_field_from_am,
)
# Public server API (gradients, batched lasers, flat output)
from diffmech.methods.am.am_api import (
    laser_heat_batched, am_loss_and_grads, step_am_server,
)

__all__ = [
    # submodules
    "geometry", "scan_paths", "process", "activation",
    "am_thermal", "thermomechanical", "particle_am", "am_microstructure",
    # geometry
    "SDF", "smooth_union", "smooth_intersection", "smooth_difference",
    "union", "intersection", "difference", "translate", "rotate_z",
    "sdf_sphere", "sdf_box", "sdf_cylinder", "sdf_torus", "sdf_gear",
    "slice_mask", "layer_heights", "part_bbox",
    "sdf_from_mesh", "sdf_from_stl", "sdf_from_obj",
    # scan paths
    "ScanPath", "MultiLayerPath", "multi_layer_paths",
    "zigzag_hatch", "contour_hatch", "spiral_hatch",
    "from_waypoints", "chain_paths", "from_csv", "from_gcode",
    # process
    "SLMConfig", "LSFConfig", "layer_time", "layer_activation_times",
    # activation
    "LayeredMesh", "build_layered_mesh", "assign_activation_times",
    "part_cell_mask",
    # thermal
    "laser_position", "am_heat_source", "AMThermalProblem", "setup_am_thermal",
    # thermo-mechanical
    "AMThermoMechResult", "solve_thermomechanical",
    "assemble_activated_stiffness", "thermal_strain_force",
    # particle AM (DEM / MPM / SPH)
    "sdf_circle_2d", "sdf_box_2d", "sdf_from_polygon_2d", "sdf_from_stl_mesh_3d", "parse_binary_stl",
    "ParticleAMState", "ParticleAMProblem", "setup_particle_am",
    "place_particles_in_sdf",
    "assign_particle_activation_times", "particle_layer_times",
    "laser_heat_on_particles",
    "step_dem_am", "step_dem_am_scan",
    "step_mpm_am", "step_mpm_am_scan",
    "step_sph_am", "step_sph_am_scan",
    "peak_temperature", "activated_fraction", "mean_displacement",
    # AM microstructure (phase-field solidification × thermal history)
    "ThermalField", "MicrostructureConfig", "AMMicrostructureState",
    "make_microstructure_state",
    "thermal_undercooling", "grain_mobility",
    "solidification_rhs", "orientation_rhs",
    "step_am_microstructure", "step_am_microstructure_scan",
    "solidified_fraction", "nucleated_fraction",
    "mean_grain_size", "grain_count_estimate",
    "cooling_rate", "morphology_indicator", "thermal_field_from_am",
    # Public server API (gradients, batched lasers, flat output)
    "laser_heat_batched", "am_loss_and_grads", "step_am_server",
]
