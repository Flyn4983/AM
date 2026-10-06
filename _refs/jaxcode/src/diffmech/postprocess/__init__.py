"""Post-processing: VTK / VTU / PVD writers, visualisation, and field probes.

Provides:

- :mod:`diffmech.postprocess.vtk` — VTU/VTK writers and a PVDSeries
  ParaView animation writer, plus von Mises / equivalent-strain field helpers.
- :mod:`diffmech.postprocess.visualization` — quick matplotlib plots for
  2D fields (contour, quiver, deformed mesh) and 3D scatter / slices.
- :mod:`diffmech.postprocess.probes` — point/line probes, volume averages,
  boundary integrals, and time-history extraction.
- :mod:`diffmech.postprocess.time_series` — combined VTU animation +
  NumPy history logger for time-stepping simulations.

All writers use :mod:`meshio` under the hood, so output is readable by
ParaView, VisIt, and any VTK-aware tool.
"""

from diffmech.postprocess.vtk import (
    FieldCollection, write_vtu, write_vtk, PVDSeries,
    von_mises_stress_field, equivalent_strain_field,
    nodal_displacement_field,
)
from diffmech.postprocess.visualization import (
    plot_field_2d, plot_contour_2d, plot_quiver_2d, plot_deformed_mesh_2d,
    plot_scatter_3d, plot_slice_3d,
)
from diffmech.postprocess.probes import (
    displacement_to_node_field, node_field_to_dofs,
    point_probe, volume_average, volume_integral, total_mass,
    boundary_node_field_integral, field_statistics,
    extract_history, history_at_dof, tip_displacement_history,
)
from diffmech.postprocess.time_series import (
    HistoryLogger, TimeSeriesWriter,
    save_history_npz, load_history_npz, save_snapshot,
)
from diffmech.postprocess.analysis import (
    l1_norm, l2_norm, linf_norm, relative_error, error_report,
    fit_convergence_rate, convergence_study, convergence_table,
    ConvergenceResult,
    compare_solutions, ComparisonReport,
    summary_report,
)
from diffmech.postprocess.derived_fields import (
    von_mises_from_stress, principal_stresses,
    hydrostatic_pressure, equivalent_strain, stress_field_collection,
)
from diffmech.postprocess.animation import (
    save_field_gif, save_field_grid_png,
)

__all__ = [
    # vtk
    "FieldCollection", "write_vtu", "write_vtk", "PVDSeries",
    "von_mises_stress_field", "equivalent_strain_field",
    "nodal_displacement_field",
    # visualization
    "plot_field_2d", "plot_contour_2d", "plot_quiver_2d", "plot_deformed_mesh_2d",
    "plot_scatter_3d", "plot_slice_3d",
    # probes
    "displacement_to_node_field", "node_field_to_dofs",
    "point_probe", "volume_average", "volume_integral", "total_mass",
    "boundary_node_field_integral", "field_statistics",
    "extract_history", "history_at_dof", "tip_displacement_history",
    # time_series
    "HistoryLogger", "TimeSeriesWriter",
    "save_history_npz", "load_history_npz", "save_snapshot",
    # analysis
    "l1_norm", "l2_norm", "linf_norm", "relative_error", "error_report",
    "fit_convergence_rate", "convergence_study", "convergence_table",
    "ConvergenceResult",
    "compare_solutions", "ComparisonReport",
    "summary_report",
    # derived_fields
    "von_mises_from_stress", "principal_stresses",
    "hydrostatic_pressure", "equivalent_strain", "stress_field_collection",
    # animation
    "save_field_gif", "save_field_grid_png",
]
