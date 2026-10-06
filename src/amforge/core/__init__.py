"""AMForge 内核层：数据契约 + 求解器注册表 + 流水线编排。

三个文件的职责划分：

* :mod:`amforge.core.contracts` —— **说清楚数据长什么样**。
  9 个 frozen dataclass，全部注册为 JAX pytree，单位统一 SI。
  这是模块间唯一的通信语言，任何求解器都不许绕过它私下传数据。
* :mod:`amforge.core.registry` —— **说清楚模块吃什么吐什么**。
  声明式注册，附带 cost / modality / differentiable 元信息。
* :mod:`amforge.core.graph` —— **把模块接起来并跑通**。
  按契约类型自动布线、拓扑执行、柯里化成可求导函数。
"""

from __future__ import annotations

from amforge.core.contracts import (
    CONTRACTS,
    AsBuiltPart,
    ConstitutiveField,
    MeltPoolResult,
    MicrostructureResult,
    PartGeometry,
    ProcessPlan,
    ServiceVerdict,
    StructuralResult,
    ThermalHistory,
    register_contract,
)
from amforge.core.graph import (
    CircularDependency,
    Pipeline,
    PipelineTypeError,
    Step,
    Unresolvable,
)
from amforge.core.registry import (
    CONTRACT_OF_SLOT,
    SLOT_OF,
    STAGES,
    AdapterSpec,
    DuplicateSolver,
    SolverContractError,
    SolverNotFound,
    SolverSpec,
    clear_registry,
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

__all__ = [
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
    "CONTRACTS",
    "register_contract",
    # 注册表
    "SolverSpec",
    "AdapterSpec",
    "register_solver",
    "register_adapter",
    "get_solver",
    "get_adapter",
    "find_solvers",
    "find_adapters",
    "list_solvers",
    "list_adapters",
    "solver_table",
    "clear_registry",
    "slot_of",
    "SLOT_OF",
    "CONTRACT_OF_SLOT",
    "STAGES",
    "SolverNotFound",
    "DuplicateSolver",
    "SolverContractError",
    # 编排
    "Pipeline",
    "Step",
    "Unresolvable",
    "CircularDependency",
    "PipelineTypeError",
]
