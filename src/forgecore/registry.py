"""ForgeCore 编排层：求解器注册表 + 自动选路（「无缝链接」核心）。

设计（见 docs/AMForge-architecture.md）：
* 端口类型用字符串表示（如 "geometry" / "thermal" / "meltpool" / "verdict"），
  与具体物理域解耦——新增物理/方法 = 注册一个 SolverSpec，核心永不硬编码。
* SolverSpec.fn 是纯函数：fn(**{消费端口: 值}, params=dict) -> 产出值。
* auto(target, modality) 从目标端口反向链式解析出一条 DAG 拓扑序；
  无生产者的端口（geometry/material 等）被识别为「外部输入」，需由 run 提供。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


class ResolutionError(Exception):
    """选路/装配失败。"""


@dataclass
class SolverSpec:
    name: str
    produces: str
    consumes: Tuple[str, ...] = ()
    fn: Optional[Callable] = None
    stage: str = ""
    modality: Tuple[str, ...] = ()
    differentiable: bool = True
    cost: float = 1.0
    defaults: Dict[str, Any] = field(default_factory=dict)
    doc: str = ""
    tags: Tuple[str, ...] = ()

    def call(self, ctx: Dict[str, Any], params: Dict[str, Any]) -> Any:
        if self.fn is None:
            raise ResolutionError(f"solver {self.name!r} 未绑定 fn")
        kwargs = {p: ctx[p] for p in self.consumes if p in ctx}
        missing = [p for p in self.consumes if p not in ctx]
        if missing:
            raise ResolutionError(
                f"solver {self.name!r} 缺输入端口: {missing}（需由外部提供或上游产出）")
        effective = {**self.defaults, **params}
        return self.fn(**kwargs, params=effective)


class SolverRegistry:
    """插件式求解器注册表；auto() 反向链式选路。"""

    def __init__(self) -> None:
        self._by_name: Dict[str, SolverSpec] = {}
        self._by_produces: Dict[str, List[SolverSpec]] = {}

    # ---- 注册 ----------------------------------------------------------------
    def register(self, spec: SolverSpec) -> SolverSpec:
        if spec.name in self._by_name:
            raise ValueError(f"求解器名重复: {spec.name!r}")
        if not spec.fn:
            raise ValueError(f"求解器 {spec.name!r} 必须绑定 fn")
        self._by_name[spec.name] = spec
        self._by_produces.setdefault(spec.produces, []).append(spec)
        return spec

    def register_fn(self, name: str, produces: str, consumes: Sequence[str] = (),
                    *, fn: Callable, stage: str = "", modality: Sequence[str] = (),
                    differentiable: bool = True, cost: float = 1.0,
                    defaults: Optional[Dict[str, Any]] = None,
                    doc: str = "", tags: Sequence[str] = ()) -> SolverSpec:
        return self.register(SolverSpec(
            name=name, produces=produces, consumes=tuple(consumes),
            fn=fn, stage=stage, modality=tuple(modality),
            differentiable=differentiable, cost=cost,
            defaults=dict(defaults or {}), doc=doc, tags=tuple(tags)))

    # ---- 查询 ----------------------------------------------------------------
    def get(self, name: str) -> SolverSpec:
        if name not in self._by_name:
            raise ResolutionError(f"无此求解器: {name!r}")
        return self._by_name[name]

    def specs(self) -> List[SolverSpec]:
        return list(self._by_name.values())

    def producers_of(self, port: str) -> List[SolverSpec]:
        return list(self._by_produces.get(port, []))

    def list_names(self) -> List[str]:
        return sorted(self._by_name)

    # ---- 自动选路 ------------------------------------------------------------
    def _match(self, port: str, modality: str) -> List[SolverSpec]:
        cands = self.producers_of(port)
        if not cands:
            return []
        if modality:
            specific = [s for s in cands if modality in s.modality]
            if specific:
                return specific
        # 无 modality 限定或该 modality 无专属求解器时，返回全部（含通用求解器）
        return cands

    def _select(self, cands: List[SolverSpec], modality: str) -> SolverSpec:
        # 优先：命中 modality 专属；其次：非闭式/解析近似（"analytical" 标签）；
        # 再次：低成本。这样默认链路走真实数值离散求解器，而非 Eagar-Tsai /
        # Rosenthal 类解析近似（用户硬原则：仿真=数值方法）。
        def key(s: SolverSpec):
            specific = 0 if (modality and modality in s.modality) else 1
            numerical = 0 if "analytical" not in s.tags else 1
            return (specific, numerical, s.cost)
        return min(cands, key=key)

    def auto(self, target: str, modality: str = "",
             select: Optional[Dict[str, str]] = None,
             _memo: Optional[Dict[str, bool]] = None,
             _emitted: Optional[Dict[str, bool]] = None) -> List[SolverSpec]:
        """从 target 端口反向解析出一条拓扑有序的求解器链。

        返回按「先输入后产出」排序的 SolverSpec 列表；无生产者的端口作为
        外部输入（不出现在列表中，run 时需提供）。

        select : 端口 -> 求解器名 的强制映射。提供后，该端口只解析到指定
        求解器（用于显式切换高保真/代理求解器，如
        ``select={"thermal": "thermal.enthalpy"}``）。
        """
        if _memo is None:
            _memo = {}
        if _emitted is None:
            _emitted = {}
        if target in _memo:
            return []
        if select and target in select:
            spec = self.get(select[target])          # 强制指定
        else:
            cands = self._match(target, modality)
            if not cands:
                _memo[target] = True  # 外部输入端口
                return []
            spec = self._select(cands, modality)
        ordered: List[SolverSpec] = []
        for port in spec.consumes:
            ordered.extend(self.auto(port, modality, select, _memo, _emitted))
        if spec.name not in _emitted:
            _emitted[spec.name] = True
            ordered.append(spec)
        _memo[target] = True
        return ordered

    def required_inputs(self, target: str, modality: str = "",
                       select: Optional[Dict[str, str]] = None) -> List[str]:
        """auto 解析后，需要外部提供的输入端口集合。"""
        specs = self.auto(target, modality, select)
        produced = {s.produces for s in specs}
        consumed = {p for s in specs for p in s.consumes}
        return sorted(consumed - produced)

    def build(self, target: str, modality: str = "",
              params: Optional[Dict[str, Any]] = None,
              select: Optional[Dict[str, str]] = None):
        from .graph import SimulationGraph
        specs = self.auto(target, modality, select)
        return SimulationGraph(specs, target, modality, dict(params or {}))


# 全局默认注册表（所有求解器模块 import 时向其注册）
REGISTRY = SolverRegistry()
