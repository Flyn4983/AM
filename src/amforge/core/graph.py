"""流水线编排器 (Pipeline / DAG Orchestrator)
============================================

这是"模块之间无缝链接"的执行引擎。

它做三件事
----------
1. **自动布线 (auto-wiring)**
   你只说"我要 :class:`ServiceVerdict`，手上只有 :class:`PartGeometry`"，
   :meth:`Pipeline.auto` 就按契约类型反向链式推导出一条最小代价路径：

       geometry -> process -> meltpool -> thermal -> microstructure
                -> constitutive -> asbuilt -> structural -> verdict

   多条路径可选时按 ``cost`` 取最小（同代价求解器优先）。默认**排除闭式/解析近似**
   （``exclude_methods=("analytical",)``，见用户硬原则"仿真=数值方法"），因此默认
   链路走真实数值离散求解器（如 ``meltpool.fdm`` / ``thermal.enthalpy``）；解析解
   仅在 ``select`` 强制指定或 ``exclude_methods=()`` 时才会被选中。

2. **拓扑执行**
   按依赖顺序调用求解器，把每一步的输出按槽名存进 context，
   下一步直接按槽名取用。中间不做任何隐式单位换算或形状猜测——
   契约已经把单位（SI）和语义钉死了。

3. **可微闭环入口**
   :meth:`Pipeline.function` 把整条流水线柯里化成
   ``f(params, **inputs) -> 目标契约``，可直接喂给 ``jax.grad`` / ``jax.jit``。
   功能 2（神经网络预测工艺 → 模拟 → 损失 → 反传）就架在这个入口上。

为什么不用现成的工作流框架？
--------------------------
因为需要**梯度穿透**。Airflow / Prefect 那类框架在步骤之间做序列化与
进程隔离，梯度链就断了。这里的"DAG"只是纯函数的组合，
整条链在一个 JAX 追踪上下文内完成，因此 ``jax.grad`` 能一路回传到工艺参数
乃至神经网络权重。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Mapping, Sequence

from amforge.core.contracts import CONTRACTS
from amforge.core.registry import (
    AdapterSpec,
    SolverSpec,
    find_adapters,
    find_solvers,
    get_adapter,
    get_solver,
    slot_of,
)

__all__ = [
    "Step",
    "Pipeline",
    "Unresolvable",
    "CircularDependency",
    "PipelineTypeError",
]


class Unresolvable(RuntimeError):
    """无法从已有输入推导出目标契约。"""


class CircularDependency(RuntimeError):
    """契约依赖出现环。"""


class PipelineTypeError(TypeError):
    """求解器输出类型与其声明的契约不符。"""


# ---------------------------------------------------------------------------
# 单步
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Step:
    """流水线上的一步：求解器或适配器的统一包装。"""

    spec: SolverSpec | AdapterSpec
    kind: str  # "solver" | "adapter"

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def in_slots(self) -> tuple[str, ...]:
        if self.kind == "solver":
            return self.spec.in_slots  # type: ignore[union-attr]
        return (self.spec.in_slot,)  # type: ignore[union-attr]

    @property
    def out_slot(self) -> str:
        return self.spec.out_slot

    @property
    def produces(self) -> str:
        if self.kind == "solver":
            return self.spec.produces  # type: ignore[union-attr]
        return self.spec.target  # type: ignore[union-attr]

    @property
    def cost(self) -> float:
        return self.spec.cost

    @property
    def differentiable(self) -> bool:
        return self.spec.differentiable

    def merged_params(self, user: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if self.kind == "solver":
            return self.spec.merged_params(user)  # type: ignore[union-attr]
        return dict(user) if user else None

    def __call__(self, *, params=None, **kwargs):
        return self.spec.fn(**kwargs, params=params)


def _merge_steps(*groups: Sequence[Step]) -> list[Step]:
    """按首次出现顺序去重合并，保持拓扑有效性。"""
    seen: set[str] = set()
    out: list[Step] = []
    for g in groups:
        for s in g:
            if s.name not in seen:
                seen.add(s.name)
                out.append(s)
    return out


def _canon_slot(key: str) -> str:
    """接受槽名或契约类名，统一成槽名。"""
    if key in CONTRACTS:
        return slot_of(key)
    return key


# ---------------------------------------------------------------------------
# 流水线
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Pipeline:
    """一条已解析好的求解器链。

    ``steps`` 已是拓扑序，执行时顺序调用即可。
    Pipeline 本身是**静态结构**（不含张量），因此可以安全地被闭包捕获、
    作为 ``jax.jit`` 的静态参数，不会触发重复编译。
    """

    steps: tuple[Step, ...]
    target: str = ""          # 目标契约类名
    given: tuple[str, ...] = ()  # 外部注入的槽名
    name: str = "pipeline"

    # -- 构造 -------------------------------------------------------------
    @staticmethod
    def auto(
        target: str,
        *,
        given: Iterable[str] = ("PartGeometry",),
        modality: str | None = None,
        select: Mapping[str, str] | None = None,
        exclude: Iterable[str] = (),
        require_differentiable: bool = False,
        use_adapters: bool = True,
        exclude_methods: Iterable[str] = ("analytical",),
        name: str = "",
    ) -> "Pipeline":
        """按契约类型自动布线。

        Parameters
        ----------
        target
            目标契约类名（如 ``"ServiceVerdict"``）或槽名（``"verdict"``）。
        given
            已在手的契约/槽名，作为链的起点，不再向上求解。
        modality
            ``"SLM"`` 或 ``"LSF"``。会过滤掉声明了其他工艺的求解器。
        select
            强制指定：``{槽名或契约名: 求解器名}``。用于把默认的快速代理
            替换成高保真求解器，或在多候选时消解歧义。
        exclude
            排除的求解器/适配器名集合。
        require_differentiable
            为 ``True`` 时只挑选可微求解器（功能 2 的闭环需要）。
        use_adapters
            是否允许用适配器桥接契约。关掉后要求求解器直接产出目标契约。
        exclude_methods
            默认排除的求解方法类别（见 :class:`~amforge.core.registry.SolverSpec.method`）。
            默认 ``("analytical",)`` —— **用户硬原则：默认前向链路不得走闭式/解析
            近似**，因此 Eagar-Tsai / Rosenthal 这类解析解不会被选为默认求解器；
            真实数值离散（FEM/FVM/CFD 等）成为开箱即用的默认。``select`` 仍可强制
            指定解析解（如优化闭环的低成本前向），``exclude_methods=()`` 则完全放开。

        Raises
        ------
        Unresolvable
            没有任何路径能从 ``given`` 到达 ``target``。异常信息会列出
            缺失的契约与当时可用的候选，便于定位是"缺求解器"还是"缺输入"。
        """
        target_contract = _contract_name(target)
        given_slots = {_canon_slot(g) for g in given}
        select_norm = {_canon_slot(k): v for k, v in (select or {}).items()}
        excluded = set(exclude)

        planner = _Planner(
            given=given_slots,
            modality=modality,
            select=select_norm,
            exclude=excluded,
            require_differentiable=require_differentiable,
            use_adapters=use_adapters,
            exclude_methods=tuple(exclude_methods),
        )
        steps, _cost = planner.resolve(target_contract)
        return Pipeline(
            steps=tuple(steps),
            target=target_contract,
            given=tuple(sorted(given_slots)),
            name=name or f"auto:{slot_of(target_contract)}",
        )

    @staticmethod
    def of(
        *names: str,
        given: Iterable[str] = ("PartGeometry",),
        name: str = "",
    ) -> "Pipeline":
        """由显式求解器/适配器名构造，自动做拓扑排序。

        适合已经明确知道要用哪几个模块的场景（论文复现、回归测试）。
        """
        steps: list[Step] = []
        for n in names:
            steps.append(_lookup_step(n))
        ordered = _topo_sort(steps, {_canon_slot(g) for g in given})
        return Pipeline(
            steps=tuple(ordered),
            target=ordered[-1].produces if ordered else "",
            given=tuple(sorted(_canon_slot(g) for g in given)),
            name=name or "explicit",
        )

    def then(self, solver: str | Step) -> "Pipeline":
        """在链尾追加一步，返回新 Pipeline（不可变风格）。"""
        step = solver if isinstance(solver, Step) else _lookup_step(solver)
        return replace(
            self, steps=self.steps + (step,), target=step.produces
        )

    # -- 自省 -------------------------------------------------------------
    @property
    def slots_produced(self) -> tuple[str, ...]:
        return tuple(s.out_slot for s in self.steps)

    @property
    def differentiable(self) -> bool:
        return all(s.differentiable for s in self.steps)

    @property
    def total_cost(self) -> float:
        return sum(s.cost for s in self.steps)

    def solver_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.steps)

    def param_template(self) -> dict[str, dict[str, Any]]:
        """给出 ``params`` 的骨架（含各求解器默认值），便于用户改写。"""
        return {s.name: dict(s.merged_params(None) or {}) for s in self.steps}

    def validate(self) -> None:
        """静态检查：每一步的输入是否都能由 given 或前序步骤提供。"""
        have = set(self.given)
        for step in self.steps:
            missing = [s for s in step.in_slots if s not in have]
            if missing:
                raise Unresolvable(
                    f"步骤 {step.name!r} 缺少输入槽 {missing}；"
                    f"当前可用: {sorted(have)}。"
                    f"提示：把缺失契约加入 given=，或在链中插入产出它的求解器。"
                )
            have.add(step.out_slot)

    def describe(self) -> str:
        """人可读的链路说明，直接可贴进日志或报告。"""
        lines = [
            f"Pipeline {self.name!r}  目标={self.target or '?'}  "
            f"步数={len(self.steps)}  可微={'是' if self.differentiable else '否'}  "
            f"代价≈{self.total_cost:g}",
            f"  输入: {', '.join(self.given) or '(无)'}",
        ]
        for i, s in enumerate(self.steps, 1):
            ins = " + ".join(s.in_slots) or "-"
            tag = "" if s.kind == "solver" else " [适配器]"
            grad = "" if s.differentiable else "  (不可微!)"
            lines.append(
                f"  {i:>2}. {ins:<28} -> {s.out_slot:<14} "
                f"{s.name}{tag}{grad}"
            )
        return "\n".join(lines)

    def mermaid(self) -> str:
        """输出 Mermaid flowchart 源码，用于文档/汇报。"""
        lines = ["flowchart LR"]
        for g in self.given:
            lines.append(f'  {g}(["{g}"]):::given')
        for s in self.steps:
            shape = f'{s.out_slot}["{s.out_slot}"]'
            lines.append(f"  {shape}")
            for i in s.in_slots:
                label = s.name.split(".")[-1]
                lines.append(f"  {i} -->|{label}| {s.out_slot}")
        lines.append("  classDef given fill:#1f6feb,color:#fff,stroke:none;")
        return "\n".join(lines)

    # -- 执行 -------------------------------------------------------------
    def run(
        self,
        *,
        params: Mapping[str, Mapping[str, Any]] | None = None,
        strict_types: bool = True,
        trace: list[str] | None = None,
        **inputs: Any,
    ) -> dict[str, Any]:
        """顺序执行整条链，返回 ``{槽名: 契约}`` 的 context。

        Parameters
        ----------
        params
            ``{求解器名: {参数名: 值}}``。缺省的求解器用其 ``defaults``。
            这个 pytree 就是功能 2 里被求导的对象之一。
        strict_types
            检查每步输出是否为其声明的契约类型。默认开启——
            类型错误在多物理场链里排查代价极高，早失败早省事。
        trace
            传入一个 list 时，会把每步名字追加进去，便于测试断言执行路径。
        **inputs
            起始契约，键为槽名或契约类名，如 ``geometry=part``。

        Notes
        -----
        本方法是纯函数（前提是各求解器本身纯），可放在 ``jax.jit`` 内。
        不要在这里加日志打印之外的副作用。
        """
        ctx: dict[str, Any] = {}
        for k, v in inputs.items():
            ctx[_canon_slot(k)] = v

        missing_given = [g for g in self.given if g not in ctx]
        if missing_given:
            raise Unresolvable(
                f"Pipeline {self.name!r} 声明需要输入 {list(self.given)}，"
                f"但调用时缺少 {missing_given}。"
            )

        params = params or {}
        for step in self.steps:
            kwargs = {}
            for slot in step.in_slots:
                if slot not in ctx:
                    raise Unresolvable(
                        f"执行到 {step.name!r} 时缺少输入槽 {slot!r}；"
                        f"当前 context: {sorted(ctx)}"
                    )
                kwargs[slot] = ctx[slot]
            out = step(params=step.merged_params(params.get(step.name)), **kwargs)
            if strict_types:
                expected = CONTRACTS[step.produces]
                if not isinstance(out, expected):
                    raise PipelineTypeError(
                        f"求解器 {step.name!r} 声明产出 {step.produces}，"
                        f"实际返回 {type(out).__name__}。"
                        f"请检查该求解器的 return 语句或注册时的 produces=。"
                    )
            ctx[step.out_slot] = out
            if trace is not None:
                trace.append(step.name)
        return ctx

    def __call__(self, **kw):
        return self.run(**kw)

    def function(
        self,
        *,
        target: str | None = None,
        return_context: bool = False,
    ) -> Callable[..., Any]:
        """柯里化成 ``f(params, **inputs)``，供 ``jax.grad`` / ``jax.jit`` 使用。

        Examples
        --------
        >>> pipe = Pipeline.auto("AsBuiltPart", require_differentiable=True)
        >>> f = pipe.function()
        >>> loss = lambda p: geometric_loss(f(p, geometry=geo), ideal)
        >>> g = jax.grad(loss)(params)          # 梯度一路回传到工艺参数
        """
        slot = slot_of(_contract_name(target or self.target))

        def fn(params, **inputs):
            ctx = self.run(params=params, **inputs)
            return ctx if return_context else ctx[slot]

        fn.__name__ = f"{self.name}_fn"
        fn.__doc__ = self.describe()
        return fn


def _contract_name(x: str) -> str:
    """槽名或契约类名 -> 契约类名。"""
    if x in CONTRACTS:
        return x
    from amforge.core.registry import CONTRACT_OF_SLOT

    if x in CONTRACT_OF_SLOT:
        return CONTRACT_OF_SLOT[x]
    raise Unresolvable(
        f"未知契约/槽名 {x!r}。合法契约: {sorted(CONTRACTS)}；"
        f"合法槽名: {sorted(CONTRACT_OF_SLOT)}"
    )


def _lookup_step(name: str) -> Step:
    """先查求解器，再查适配器。"""
    try:
        return Step(get_solver(name), "solver")
    except KeyError:
        return Step(get_adapter(name), "adapter")


def _topo_sort(steps: Sequence[Step], given: set[str]) -> list[Step]:
    """对显式给出的步骤做 Kahn 拓扑排序。"""
    remaining = list(steps)
    have = set(given)
    ordered: list[Step] = []
    while remaining:
        progressed = False
        for step in list(remaining):
            if all(s in have for s in step.in_slots):
                ordered.append(step)
                have.add(step.out_slot)
                remaining.remove(step)
                progressed = True
        if not progressed:
            stuck = {s.name: [i for i in s.in_slots if i not in have] for s in remaining}
            raise CircularDependency(
                f"无法拓扑排序，可能存在环或缺失输入。未满足依赖: {stuck}；"
                f"当前可用槽: {sorted(have)}"
            )
    return ordered


# ---------------------------------------------------------------------------
# 反向链式规划器
# ---------------------------------------------------------------------------
class _Planner:
    """从目标契约反向搜索最小代价求解路径。

    这是一个带记忆化的深度优先搜索：把"契约"看作节点，
    "求解器/适配器"看作超边（多输入单输出）。代价为各边 ``cost`` 之和，
    共享子图会被重复计入代价（估算偏保守），但执行时会去重，
    所以只影响选路偏好，不影响正确性。
    """

    def __init__(
        self,
        *,
        given: set[str],
        modality: str | None,
        select: Mapping[str, str],
        exclude: set[str],
        require_differentiable: bool,
        use_adapters: bool,
        exclude_methods: tuple[str, ...] = ("analytical",),
    ):
        self.given = given
        self.modality = modality
        self.select = select
        self.exclude = exclude
        self.require_diff = require_differentiable
        self.use_adapters = use_adapters
        self.exclude_methods = exclude_methods
        self._memo: dict[str, tuple[list[Step], float]] = {}
        self._stack: list[str] = []

    # -- 候选生成 ---------------------------------------------------------
    def _candidates(self, contract: str) -> list[Step]:
        slot = slot_of(contract)
        forced = self.select.get(slot)
        if forced:
            step = _lookup_step(forced)
            if step.produces != contract:
                raise Unresolvable(
                    f"select 指定 {slot!r} 用 {forced!r}，"
                    f"但它产出 {step.produces} 而非 {contract}。"
                )
            return [step]

        out = [
            Step(s, "solver")
            for s in find_solvers(
                produces=contract,
                modality=self.modality,
                differentiable=True if self.require_diff else None,
            )
            if s.name not in self.exclude
            and getattr(s, "method", "numerical") not in self.exclude_methods
        ]
        if self.use_adapters:
            out += [
                Step(a, "adapter")
                for a in find_adapters(target=contract)
                if a.name not in self.exclude
                and (a.differentiable or not self.require_diff)
            ]
        # 求解器优先于适配器（同代价时），再按 cost、name 排序保证可复现
        return sorted(out, key=lambda s: (s.cost, s.kind == "adapter", s.name))

    # -- 主递归 -----------------------------------------------------------
    def resolve(self, contract: str) -> tuple[list[Step], float]:
        slot = slot_of(contract)
        if slot in self.given:
            return [], 0.0
        if slot in self._memo:
            return self._memo[slot]
        if slot in self._stack:
            raise CircularDependency(
                f"契约依赖成环: {' -> '.join(self._stack + [slot])}"
            )

        self._stack.append(slot)
        try:
            candidates = self._candidates(contract)
            if not candidates:
                raise Unresolvable(self._explain_missing(contract))

            best: tuple[list[Step], float] | None = None
            failures: list[str] = []
            for cand in candidates:
                try:
                    sub: list[list[Step]] = []
                    for in_contract in _step_inputs(cand):
                        st, _c = self.resolve(in_contract)
                        sub.append(st)
                except (Unresolvable, CircularDependency) as exc:
                    failures.append(f"    - {cand.name}: {exc}")
                    continue
                # 合并后的步骤列表已对同名求解器去重；代价按「每个求解器只计一次」
                # 求和（共享子图不再重复计入），这才是 DAG 的真实代价。
                # 否则 thermal 同时被 buildup 与 microstructure 消费时会被重复累加，
                # 使廉价真实档（buildup）的选路代价被虚高，反而选中沉重的高保真档。
                steps = _merge_steps(*sub, [cand])
                cost = sum(s.cost for s in steps)
                if best is None or cost < best[1]:
                    best = (steps, cost)

            if best is None:
                raise Unresolvable(
                    f"目标 {contract} 有 {len(candidates)} 个候选求解器，"
                    f"但它们的输入都无法满足:\n" + "\n".join(failures)
                )
            self._memo[slot] = best
            return best
        finally:
            self._stack.pop()

    def _explain_missing(self, contract: str) -> str:
        all_producers = find_solvers(produces=contract)
        if not all_producers:
            return (
                f"没有任何求解器产出 {contract}。"
                f"需要先用 @register_solver(produces={contract!r}) 注册一个，"
                f"或把它直接放进 given=。"
            )
        reasons = []
        for s in all_producers:
            why = []
            if s.name in self.exclude:
                why.append("被 exclude 排除")
            if self.modality and s.modality and self.modality not in s.modality:
                why.append(f"仅支持 {s.modality}，与 modality={self.modality!r} 不符")
            if self.require_diff and not s.differentiable:
                why.append("不可微，但要求 require_differentiable=True")
            reasons.append(f"    - {s.name}: {'; '.join(why) or '未知'}")
        return (
            f"产出 {contract} 的求解器全部被过滤掉了:\n" + "\n".join(reasons)
        )


def _step_inputs(step: Step) -> tuple[str, ...]:
    """一步所需的输入契约类名。"""
    if step.kind == "solver":
        return step.spec.consumes  # type: ignore[union-attr]
    return (step.spec.source,)  # type: ignore[union-attr]
