"""求解器注册表 (Solver Registry)
================================

AMForge 的"模块即求解器"机制在此实现。

设计目标
--------
用户要求：*"各个模块作为求解器，模块之间可以无缝链接"*。

要让链接真正"无缝"，必须回答三个问题：

1. **一个模块吃什么、吐什么？**
   由 :mod:`amforge.core.contracts` 中的 9 个数据契约给出类型级答案。
   每个求解器在注册时声明 ``consumes`` / ``produces``，这是纯声明式的，
   不依赖求解器内部实现。

2. **模块怎样被调用？**
   统一调用约定（calling convention）：所有求解器都是

       ``fn(*, <slot_1>=..., <slot_2>=..., params=<pytree>) -> Contract``

   其中 ``slot_i`` 是契约的规范槽名（见 :data:`SLOT_OF`），
   ``params`` 是该求解器的可微参数/配置 pytree。
   这样编排器只需按槽名塞关键字参数，无需知道任何求解器细节。

3. **类型不完全对齐怎么办？**
   引入 **适配器 (adapter)**：``Contract_A -> Contract_B`` 的纯函数。
   例如熔池求解器产出 :class:`MeltPoolResult`，而微观模块需要
   :class:`ThermalHistory`，两者由一个适配器桥接，而不是在求解器里硬耦合。

可微性约束
----------
注册表本身是**静态**的（Python 层字典），不参与 JAX 追踪。
求解器函数必须是纯函数：给定 ``params`` 与输入契约，输出契约；
不得依赖全局可变状态、不得在内部做 Python 层 if 判断张量值。
满足此约束后，整条流水线可被 ``jax.grad`` / ``jax.jit`` 穿透，
这是功能 2（可微分模拟）的前提。
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from amforge.core.contracts import CONTRACTS

__all__ = [
    "SLOT_OF",
    "CONTRACT_OF_SLOT",
    "slot_of",
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
    "SolverNotFound",
    "DuplicateSolver",
    "SolverContractError",
    "STAGES",
]


# ---------------------------------------------------------------------------
# 契约 <-> 槽名 映射
# ---------------------------------------------------------------------------
# 槽名是契约在流水线上下文（context dict）中的键，也是求解器的关键字参数名。
# 之所以不用类名直接当参数名，是为了让求解器签名读起来像正常的科学计算代码：
#   def solve(*, geometry, process, params): ...
# 而不是 def solve(*, PartGeometry, ProcessPlan, params)。
SLOT_OF: dict[str, str] = {
    "PartGeometry": "geometry",
    "ProcessPlan": "process",
    "SupportStructure": "support",
    "PowderBedResult": "powderbed",
    "ThermalHistory": "thermal",
    "MeltPoolResult": "meltpool",
    "MicrostructureResult": "microstructure",
    "ConstitutiveField": "constitutive",
    "AsBuiltPart": "asbuilt",
    "AssemblyResult": "assembly",
    "StructuralResult": "structural",
    "ServiceVerdict": "verdict",
    "SecondaryProcessResult": "secondary",
    "SensorData": "sensor",
    "MonitoringState": "monitoring",
    "ClosedLoopPlan": "closedloop",
}

CONTRACT_OF_SLOT: dict[str, str] = {v: k for k, v in SLOT_OF.items()}

# 流水线阶段。仅用于文档/可视化分组与冲突消解时的排序，不影响 DAG 正确性。
STAGES: tuple[str, ...] = (
    "geometry",       # STL/体素 -> PartGeometry
    "process",        # 工艺规划（含神经网络预测器）-> ProcessPlan
    "support",        # 支撑结构生成与仿真 -> SupportStructure / AsBuiltPart（模块A）
    "powderbed",      # DEM/SPH/MPM 颗粒尺度铺粉 -> PowderBedResult（P2 粉末尺度）
    "meltpool",       # Flow3D 式熔池多物理场 -> MeltPoolResult
    "thermal",        # 零件级热历史 -> ThermalHistory
    "microstructure",  # 相场 / 凝固组织 -> MicrostructureResult
    "constitutive",   # CPFE 均质化 -> ConstitutiveField
    "buildup",        # 逐层激活热力耦合 -> AsBuiltPart
    "assembly",       # 多体 + FEM 耦合装配 -> AssemblyResult（P2-③）
    "structural",     # 数字样机结构响应 -> StructuralResult
    "postprocess",    # 二次工艺（热处理/HIP/切削）-> SecondaryProcessResult（模块B）
    "verdict",        # 服役判定 -> ServiceVerdict
    "monitor",        # 在线监测闭环（传感流 -> 缺陷检测 -> 工艺修正）-> SensorData/MonitoringState/ClosedLoopPlan（模块D）
    "utility",        # 其他
)


class SolverNotFound(KeyError):
    """请求的求解器 / 适配器不存在。"""


class DuplicateSolver(ValueError):
    """同名求解器重复注册且未允许覆盖。"""


class SolverContractError(TypeError):
    """求解器声明的契约与其函数签名不一致。"""


def slot_of(contract: str | type) -> str:
    """契约类名或契约类 -> 规范槽名。"""
    name = contract if isinstance(contract, str) else contract.__name__
    try:
        return SLOT_OF[name]
    except KeyError as exc:  # pragma: no cover - 防御性
        raise SolverContractError(
            f"未知契约 {name!r}；已知契约: {sorted(SLOT_OF)}"
        ) from exc


def _check_contract_names(names: Sequence[str], where: str) -> tuple[str, ...]:
    out = []
    for n in names:
        if n not in CONTRACTS:
            raise SolverContractError(
                f"{where}: 未知契约类型 {n!r}。合法取值: {sorted(CONTRACTS)}"
            )
        out.append(n)
    return tuple(out)


# ---------------------------------------------------------------------------
# 规格对象
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SolverSpec:
    """一个求解器模块的**声明式**描述。

    Attributes
    ----------
    name
        全局唯一标识，建议 ``<阶段>.<方法>`` 形式，如 ``meltpool.vof_flow3d``。
    fn
        纯函数，签名 ``fn(*, <slots>, params) -> Contract``。
    consumes
        输入契约类名元组。编排器据此做反向链式推导。
    produces
        输出契约类名。
    stage
        所属阶段（见 :data:`STAGES`），仅用于分组与排序提示。
    modality
        适用增材工艺，如 ``("SLM", "LSF")``；``()`` 表示与工艺无关。
    differentiable
        是否支持 ``jax.grad`` 穿透。功能 2 的闭环只会挑选 ``True`` 的求解器。
    cost
        相对计算代价提示（1 = 秒级代理模型，10 = 分钟级，100 = 小时级高保真）。
        自动选路在多候选时优先低 cost，把"高保真"留给用户显式指定。
    defaults
        ``params`` 的默认值（浅层字典）。编排器会与用户传入的 params 合并。
    doc
        一句话说明。
    method
        求解方法类别，用于自动选路时区分"真实数值离散"与"闭式/解析近似"：

        * ``"numerical"``   —— FEM/FVM/MPM/SPH/DEM/CFD 等真实离散计算（默认）；
        * ``"analytical"``  —— 闭式/解析近似解（Eagar-Tsai / Rosenthal 等）；
        * ``"proxy"``       —— 统计/经验降阶代理模型（非解析，但非全物理离散）。

        用户硬原则：默认前向链路不得选 ``"analytical"``
        （见 :func:`amforge.core.graph.Pipeline.auto` 的 ``exclude_methods``）。
    """

    name: str
    fn: Callable[..., Any]
    consumes: tuple[str, ...]
    produces: str
    stage: str = "utility"
    modality: tuple[str, ...] = ()
    differentiable: bool = True
    cost: float = 1.0
    method: str = "numerical"
    defaults: Mapping[str, Any] = field(default_factory=dict)
    doc: str = ""

    # -- 便捷视图 ---------------------------------------------------------
    @property
    def in_slots(self) -> tuple[str, ...]:
        return tuple(slot_of(c) for c in self.consumes)

    @property
    def out_slot(self) -> str:
        return slot_of(self.produces)

    def signature_str(self) -> str:
        args = ", ".join(f"{s}: {c}" for s, c in zip(self.in_slots, self.consumes))
        return f"{self.name}({args}, params) -> {self.produces}"

    def merged_params(self, user_params: Mapping[str, Any] | None) -> dict[str, Any]:
        """默认参数 + 用户参数（用户优先）。"""
        out = dict(self.defaults)
        if user_params:
            out.update(user_params)
        return out

    def __call__(self, **kwargs):  # 允许直接当函数用
        return self.fn(**kwargs)


@dataclass(frozen=True)
class AdapterSpec:
    """契约间的桥接函数：``source -> target``。

    适配器承担"无缝"里最容易被忽视的一半工作：把上游求解器的富输出
    降维/重采样/统计化成下游求解器真正需要的量。例如

    * ``MeltPoolResult -> ThermalHistory``：从瞬态温度场提取峰值温度、
      冷却速率、G/R 等凝固判据；
    * ``MicrostructureResult -> ConstitutiveField``：Hall-Petch + 织构 ->
      各向异性弹塑性参数场。

    把它们单独注册，而不是写进求解器内部，好处是：
    同一个熔池求解器可以对接不同的组织模型，反之亦然。
    """

    source: str
    target: str
    fn: Callable[..., Any]
    name: str = ""
    cost: float = 0.1
    differentiable: bool = True
    doc: str = ""

    def __post_init__(self):
        if not self.name:
            object.__setattr__(
                self, "name", f"{slot_of(self.source)}->{slot_of(self.target)}"
            )

    @property
    def in_slot(self) -> str:
        return slot_of(self.source)

    @property
    def out_slot(self) -> str:
        return slot_of(self.target)

    def __call__(self, **kwargs):
        return self.fn(**kwargs)


# ---------------------------------------------------------------------------
# 全局注册表
# ---------------------------------------------------------------------------
_SOLVERS: dict[str, SolverSpec] = {}
_ADAPTERS: dict[str, AdapterSpec] = {}


def clear_registry() -> None:
    """清空注册表（仅供测试使用）。"""
    _SOLVERS.clear()
    _ADAPTERS.clear()


def _validate_signature(fn: Callable, spec_slots: Sequence[str], name: str) -> None:
    """检查函数签名是否满足统一调用约定。

    要求：所有输入槽名 + ``params`` 都能以关键字传入。
    允许函数额外接受 ``**kwargs``（此时跳过严格检查），
    这给包装第三方求解器留出余地。
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):  # pragma: no cover - 内置/C 函数
        return

    params = sig.parameters
    has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    if has_var_kw:
        return

    missing = [s for s in spec_slots if s not in params]
    if missing:
        raise SolverContractError(
            f"求解器 {name!r} 声明消费 {list(spec_slots)}，"
            f"但函数签名缺少关键字参数 {missing}。"
            f"实际签名: {name}{sig}"
        )
    if "params" not in params:
        raise SolverContractError(
            f"求解器 {name!r} 必须接受关键字参数 `params`（可为 None）。"
            f"实际签名: {name}{sig}"
        )


def register_solver(
    name: str,
    *,
    consumes: Iterable[str] | str = (),
    produces: str,
    stage: str = "utility",
    modality: Iterable[str] | str = (),
    differentiable: bool = True,
    cost: float = 1.0,
    method: str = "numerical",
    defaults: Mapping[str, Any] | None = None,
    doc: str = "",
    overwrite: bool = False,
):
    """装饰器：把一个纯函数注册为求解器模块。

    Examples
    --------
    >>> @register_solver(
    ...     "meltpool.vof",
    ...     consumes=("PartGeometry", "ProcessPlan"),
    ...     produces="MeltPoolResult",
    ...     stage="meltpool", modality="SLM", cost=100.0,
    ... )
    ... def vof_solver(*, geometry, process, params):
    ...     ...
    """
    if isinstance(consumes, str):
        consumes = (consumes,)
    if isinstance(modality, str):
        modality = (modality,)

    consumes_t = _check_contract_names(tuple(consumes), f"求解器 {name!r} consumes")
    produces_t = _check_contract_names((produces,), f"求解器 {name!r} produces")[0]

    if stage not in STAGES:
        raise SolverContractError(
            f"求解器 {name!r} 的 stage={stage!r} 非法。合法取值: {STAGES}"
        )

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        if name in _SOLVERS and not overwrite:
            raise DuplicateSolver(
                f"求解器 {name!r} 已注册（来自 "
                f"{getattr(_SOLVERS[name].fn, '__module__', '?')}）。"
                f"如需替换请传 overwrite=True。"
            )
        slots = tuple(slot_of(c) for c in consumes_t)
        _validate_signature(fn, slots, name)
        spec = SolverSpec(
            name=name,
            fn=fn,
            consumes=consumes_t,
            produces=produces_t,
            stage=stage,
            modality=tuple(modality),
            differentiable=differentiable,
            cost=float(cost),
            method=method,
            defaults=dict(defaults or {}),
            doc=doc or (inspect.getdoc(fn) or "").strip().split("\n")[0],
        )
        _SOLVERS[name] = spec
        # 让被装饰函数仍可直接调用，同时挂上 spec 便于自省
        fn.solver_spec = spec  # type: ignore[attr-defined]
        return fn

    return deco


def register_adapter(
    source: str,
    target: str,
    *,
    name: str = "",
    cost: float = 0.1,
    differentiable: bool = True,
    doc: str = "",
    overwrite: bool = False,
):
    """装饰器：注册契约桥接函数。

    被装饰函数签名为 ``fn(*, <source_slot>, params=None) -> target_contract``。
    """
    _check_contract_names((source, target), "适配器")

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        spec = AdapterSpec(
            source=source, target=target, fn=fn, name=name, cost=cost,
            differentiable=differentiable,
            doc=doc or (inspect.getdoc(fn) or "").strip().split("\n")[0],
        )
        if spec.name in _ADAPTERS and not overwrite:
            raise DuplicateSolver(
                f"适配器 {spec.name!r} 已注册。如需替换请传 overwrite=True。"
            )
        _validate_signature(fn, (spec.in_slot,), spec.name)
        _ADAPTERS[spec.name] = spec
        fn.adapter_spec = spec  # type: ignore[attr-defined]
        return fn

    return deco


# ---------------------------------------------------------------------------
# 查询接口
# ---------------------------------------------------------------------------
def get_solver(name: str) -> SolverSpec:
    try:
        return _SOLVERS[name]
    except KeyError as exc:
        near = [k for k in _SOLVERS if name.split(".")[0] in k]
        hint = f" 同阶段候选: {near}" if near else f" 已注册: {sorted(_SOLVERS)}"
        raise SolverNotFound(f"未找到求解器 {name!r}。{hint}") from exc


def get_adapter(name: str) -> AdapterSpec:
    try:
        return _ADAPTERS[name]
    except KeyError as exc:
        raise SolverNotFound(
            f"未找到适配器 {name!r}。已注册: {sorted(_ADAPTERS)}"
        ) from exc


def find_solvers(
    *,
    produces: str | None = None,
    consumes: str | None = None,
    stage: str | None = None,
    modality: str | None = None,
    differentiable: bool | None = None,
) -> list[SolverSpec]:
    """按条件筛选求解器，结果按 ``(cost, name)`` 升序，保证可复现。"""
    out = []
    for spec in _SOLVERS.values():
        if produces is not None and spec.produces != produces:
            continue
        if consumes is not None and consumes not in spec.consumes:
            continue
        if stage is not None and spec.stage != stage:
            continue
        if modality is not None and spec.modality and modality not in spec.modality:
            continue
        if differentiable is not None and spec.differentiable != differentiable:
            continue
        out.append(spec)
    return sorted(out, key=lambda s: (s.cost, s.name))


def find_adapters(
    *, source: str | None = None, target: str | None = None
) -> list[AdapterSpec]:
    out = [
        a
        for a in _ADAPTERS.values()
        if (source is None or a.source == source) and (target is None or a.target == target)
    ]
    return sorted(out, key=lambda a: (a.cost, a.name))


def list_solvers() -> list[str]:
    return sorted(_SOLVERS)


def list_adapters() -> list[str]:
    return sorted(_ADAPTERS)


def solver_table(stage: str | None = None) -> str:
    """人可读的注册表清单，用于 CLI ``amforge solvers`` 与文档生成。"""
    specs = find_solvers(stage=stage)
    if not specs:
        return "(注册表为空)"
    w_name = max(len(s.name) for s in specs)
    w_stage = max(len(s.stage) for s in specs)
    w_method = max(len(s.method) for s in specs)
    lines = [
        f"{'求解器'.ljust(w_name)}  {'阶段'.ljust(w_stage)}  "
        f"{'方法'.ljust(w_method)}  {'可微':<4} {'代价':>5}  输入 -> 输出",
        "-" * (w_name + w_stage + w_method + 46),
    ]
    for s in specs:
        ins = "+".join(s.in_slots) or "-"
        flag = "是" if s.differentiable else "否"
        lines.append(
            f"{s.name.ljust(w_name)}  {s.stage.ljust(w_stage)}  "
            f"{s.method.ljust(w_method)}  {flag:<4} {s.cost:>5.0f}  {ins} -> {s.out_slot}"
        )
    if _ADAPTERS:
        lines.append("")
        lines.append("适配器:")
        for a in find_adapters():
            lines.append(f"  {a.name:<28} {a.source} -> {a.target}")
    return "\n".join(lines)
