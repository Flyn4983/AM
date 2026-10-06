"""AMForge GUI 子包（PySide6 + VTK 桌面界面）。

分层设计，便于无显示环境下测试：

* :mod:`amforge.gui.preproc` —— **前处理纯逻辑层**（建几何 / 生成 hatch 扫描路径 /
  构造工艺方案 / 可打印性评估 / 配置导入导出）。只依赖 jax/numpy/amforge，
  可在无 GUI、无显示的环境里直接 pytest。
* :mod:`amforge.gui.postproc` —— **后处理/数字样机/可微优化纯逻辑层**（场量抽取 /
  ParaView 导出 / 装配摘要 / 优化聚合）。同样只依赖 numpy，可无显示测试。
* :mod:`amforge.gui.preproc_app` / :mod:`amforge.gui.postproc_app` —— **PySide6 +
  VTK 界面层**，依赖 PySide6/vtk，仅在有图形界面的机器运行。
* :mod:`amforge.gui.app` —— **统一工作台**（单窗口多标签页，集成前处理与后处理）。

该分层让 GUI 的核心算法可被单元测试守住，而不只是"能弹个窗口"。
"""

from amforge.gui.preproc import (  # noqa: F401
    build_primitive,
    build_from_stl,
    generate_hatch_paths,
    build_process_plan,
    default_process_params,
    assess,
    export_config,
    import_config,
)
from amforge.gui.postproc import (  # noqa: F401
    VolumeField,
    available_fields,
    grid_fields,
    get_field,
    scalar_summary,
    write_vtk_structured_points,
    write_field_csv,
    export_all_fields,
    assembly_summary,
    assembly_chain,
    optimization_summary,
    run_simulation,
)

__all__ = [
    "build_primitive",
    "build_from_stl",
    "generate_hatch_paths",
    "build_process_plan",
    "default_process_params",
    "assess",
    "export_config",
    "import_config",
    "VolumeField",
    "available_fields",
    "grid_fields",
    "get_field",
    "scalar_summary",
    "write_vtk_structured_points",
    "write_field_csv",
    "export_all_fields",
    "assembly_summary",
    "assembly_chain",
    "optimization_summary",
    "run_simulation",
]

