"""AMForge —— 可微分增材制造多尺度仿真软件
=========================================

一句话定位
----------
把"**工艺 → 熔池 → 组织 → 本构 → 结构 → 服役判定**"整条链做成
**一串纯函数**，于是既能像 Flow3D 那样正向做高保真熔池模拟，
也能对整条链求梯度，反向优化工艺参数乃至神经网络权重。

四大功能与模块对应
------------------
======  ============================  ==========================================
功能    需求                          实现位置
======  ============================  ==========================================
1       多数值手段模拟 AM 过程        :mod:`amforge.meltpool` (VOF/CFD 熔池)
                                      :mod:`amforge.buildup`  (逐层激活热力耦合)
                                      :mod:`amforge.geometry` (任意复杂几何 -> SDF 体素 -> 分层 -> 扫描路径)
2       可微分模拟 / 工艺反演         :mod:`amforge.inverse`  (神经工艺预测器 + 端到端闭环)
3       微观模拟 -> 本构模型          :mod:`amforge.micro`    (相场凝固/晶粒)
                                      :mod:`amforge.constitutive` (CPFE 均质化 -> 各向异性本构场)
4       数字样机数字化试验            :mod:`amforge.digitaltwin` (柔性多体动力学 + FEM 接触)
                                      :mod:`amforge.verdict`  (服役裕度判定)
======  ============================  ==========================================

"无缝链接"如何做到
------------------
所有模块只通过 :mod:`amforge.core.contracts` 里的 9 个数据契约通信，
并在 :mod:`amforge.core.registry` 声明自己"吃什么、吐什么"。
:class:`amforge.core.graph.Pipeline` 据此自动布线：

>>> import amforge as af
>>> part = af.PartGeometry.from_voxels(sdf, spacing=50e-6)      # doctest: +SKIP
>>> pipe = af.Pipeline.auto("ServiceVerdict", modality="SLM")   # doctest: +SKIP
>>> print(pipe.describe())                                     # doctest: +SKIP
>>> out = pipe.run(geometry=part)                              # doctest: +SKIP
>>> print(out["verdict"].summary())                            # doctest: +SKIP

要换求解器，不用改任何调用代码，只改一行 ``select``：

>>> pipe = af.Pipeline.auto(
...     "AsBuiltPart",
...     select={"meltpool": "meltpool.vof_flow3d"},   # 代理模型 -> 高保真 CFD
... )                                                # doctest: +SKIP

底层内核
--------
数值内核复用了同仓库的 :mod:`diffmech`（JAX 可微分计算力学套件：
FEM / FVM / DEM / SPH / MPM / 相场 / CPFE）。AMForge 不重复实现基础离散，
只做 AM 领域的物理模型、契约化封装与跨尺度编排。
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "AMForge developers"

from amforge.core import (  # noqa: F401
    CONTRACTS,
    AdapterSpec,
    AsBuiltPart,
    ConstitutiveField,
    MeltPoolResult,
    MicrostructureResult,
    PartGeometry,
    Pipeline,
    ProcessPlan,
    ServiceVerdict,
    SolverSpec,
    Step,
    StructuralResult,
    ThermalHistory,
    Unresolvable,
    find_adapters,
    find_solvers,
    get_adapter,
    get_solver,
    list_adapters,
    list_solvers,
    register_adapter,
    register_solver,
    slot_of,
    solver_table,
)

# 契约全集（含各扩展模块新增契约）统一从 contracts 导入，便于 `amforge.<Contract>` 直接访问。
from amforge.core.contracts import (  # noqa: F401
    SecondaryProcessResult,
    SupportStructure,
    SensorData,
    MonitoringState,
    ClosedLoopPlan,
)

# ---------------------------------------------------------------------------
# 内置求解器的注册
# ---------------------------------------------------------------------------
# 注册表是 import 时副作用填充的。为了让 `import amforge` 之后
# Pipeline.auto 立刻可用，这里显式导入各求解器模块。
# 用容错导入：单个模块的依赖缺失（如可选的 meshio）不应让整个包 import 失败，
# 只是对应求解器不可用——Pipeline.auto 会在选路时给出清晰的"没有求解器产出 X"。
_IMPORT_ERRORS: dict[str, str] = {}


def _load_builtin(module: str) -> None:
    import importlib

    try:
        importlib.import_module(module)
    except Exception as exc:  # pragma: no cover - 依赖缺失时的降级路径
        _IMPORT_ERRORS[module] = f"{type(exc).__name__}: {exc}"


for _m in (
    "amforge.geometry",
    "amforge.process",
    "amforge.meltpool",
    "amforge.thermal",
    "amforge.thermal_enthalpy",
    "amforge.buildup",
    "amforge.micro",
    "amforge.constitutive",
    "amforge.digitaltwin",
    "amforge.verdict",
    "amforge.inverse",
    "amforge.calibration",
    "amforge.support",
    "amforge.postprocess_secondary",
    "amforge.monitoring",
    "amforge.closedloop",
):
    _load_builtin(_m)
del _m


def import_status() -> dict[str, str]:
    """返回内置模块导入失败的原因（空字典表示全部就绪）。

    诊断用。若某个求解器"凭空消失"，先看这里。
    """
    return dict(_IMPORT_ERRORS)


__all__ = [
    "__version__",
    # 契约
    "PartGeometry",
    "ProcessPlan",
    "ThermalHistory",
    "MeltPoolResult",
    "MicrostructureResult",
    "ConstitutiveField",
    "AsBuiltPart",
    "StructuralResult",
    "ServiceVerdict",
    "SecondaryProcessResult",
    "CONTRACTS",
    # 编排
    "Pipeline",
    "Step",
    "Unresolvable",
    # 注册表
    "register_solver",
    "register_adapter",
    "get_solver",
    "get_adapter",
    "find_solvers",
    "find_adapters",
    "list_solvers",
    "list_adapters",
    "solver_table",
    "slot_of",
    "SolverSpec",
    "AdapterSpec",
    "import_status",
    "calibration",  # 模块（实例化见 amforge.calibration.calibrate_material）
    "support",      # 模块（support_auto / support_simulate）
    "SupportStructure",
    "postprocess_secondary",  # 模块（HT / HIP / machining / passthrough）
    "monitoring",  # 模块（monitor.detect / monitor.correct / ingest_raw_stream）
    "closedloop",  # 模块（run_loop 闭环驱动）
    "SensorData",
    "MonitoringState",
    "ClosedLoopPlan",
]
