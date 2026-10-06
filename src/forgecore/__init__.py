"""ForgeCore — AMForge 通用仿真引擎（与具体物理域解耦）。

设计要点（见 docs/AMForge-architecture.md）：
* 唯一 canonical 模型 = 一组有类型的「积木块」(Block)，可序列化、可微。
* SolverRegistry 是插件注册表；auto() 按数据端口反推 DAG（「无缝链接」核心）。
* SimulationGraph 执行 DAG，run() 正向、as_callable() 供 jax.grad 穿透整图。
* 新增物理/方法 = 新增积木块 + 注册一个求解器，核心永不硬编码某种方法。
"""

from .registry import SolverSpec, SolverRegistry, REGISTRY
from .graph import SimulationGraph
from .model import SimulationModel
from .blocks import Block
from .serialization import save_model, load_model

__all__ = [
    "SolverSpec", "SolverRegistry", "REGISTRY",
    "SimulationGraph", "SimulationModel", "Block",
    "save_model", "load_model",
]
