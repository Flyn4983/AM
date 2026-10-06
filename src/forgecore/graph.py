"""ForgeCore 执行层：把求解器链装配成可执行的 DAG。

* run(**inputs)         正向计算，返回 {端口: 值} 的完整上下文。
* as_callable(...)      返回一个纯函数 f(*args)->target，供 jax.grad / jax.jit
                        穿透整图（可微优化「免费」获得）。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .registry import SolverSpec


class SimulationGraph:
    def __init__(self, specs: List[SolverSpec], target: str,
                 modality: str = "", params: Optional[Dict[str, Any]] = None) -> None:
        self.specs = specs
        self.target = target
        self.modality = modality
        self.params: Dict[str, Any] = dict(params or {})

    @property
    def cost(self) -> float:
        return sum(s.cost for s in self.specs)

    def required_inputs(self) -> List[str]:
        produced = {s.produces for s in self.specs}
        consumed = {p for s in self.specs for p in s.consumes}
        return sorted(consumed - produced)

    # ---- 正向执行 ------------------------------------------------------------
    def run(self, **inputs: Any) -> Dict[str, Any]:
        """正向跑完整条图。inputs 提供外部输入端口（geometry/material 等），
        可选传 params=... 覆盖默认求解参数。"""
        params = {**self.params, **inputs.pop("params", {})}
        ctx: Dict[str, Any] = dict(inputs)
        for spec in self.specs:
            ctx[spec.produces] = spec.call(ctx, params)
        return ctx

    # ---- 可微入口 ------------------------------------------------------------
    def as_callable(self, trace_ports: Sequence[str] = (),
                    fixed_inputs: Optional[Dict[str, Any]] = None,
                    params: Optional[Dict[str, Any]] = None):
        """返回一个纯函数，供 jax.grad / jax.jit 使用。

        trace_ports : 要对其求梯度的输入端口名（按此顺序成为 fn 的位置参数）。
        fixed_inputs: 非可微/常量输入端口（如 material），在调用时固定提供。
        params      : 额外覆盖的求解参数。

        例：g.as_callable(trace_ports=["geometry"])(traced_geometry) -> target_value
        """
        fixed = dict(fixed_inputs or {})
        eff = {**self.params, **(params or {})}
        specs = self.specs
        trace_ports = list(trace_ports)

        def fn(*args: Any) -> Any:
            ctx: Dict[str, Any] = dict(fixed)
            if len(args) != len(trace_ports):
                raise ValueError(
                    f"trace_ports 数量({len(trace_ports)})与入参数量({len(args)})不符")
            for name, val in zip(trace_ports, args):
                ctx[name] = val
            for spec in specs:
                ctx[spec.produces] = spec.call(ctx, eff)
            return ctx[self.target]

        return fn

    def __repr__(self) -> str:
        chain = " -> ".join(s.name for s in self.specs) or "(empty)"
        return (f"SimulationGraph(target={self.target!r}, modality={self.modality!r}, "
                f"steps={len(self.specs)}, cost={self.cost:.3g})\n  {chain}")
