"""ForgeCore 模型层：唯一 canonical 模型 = 一组有类型的「积木块」。

GUI 与 Python API 都是它的编辑器，导出同一份 .amf 工程文件。
模型只存配置级积木块（含端口声明）；重计算场由 run() 在求解时产出。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .blocks import Block
from .registry import REGISTRY


def ensure_registered() -> None:
    """惰性确保 AM 适配层（把 amforge 求解器注册进 REGISTRY）已被加载。"""
    if not REGISTRY.producers_of("verdict"):
        try:
            import amforge.forge_adapter  # noqa: F401
        except Exception:
            pass


class SimulationModel:
    """积木式仿真模型。例：

        m = SimulationModel("demo")
        m.add_part("part", geometry=geo)
        m.add_material("mat", material=mat316L)
        m.add_step("verdict", produces="verdict")
        m.set_modality("SLM")
        ctx = m.run(params={"material": "316L", "n_grid": 20})
    """

    def __init__(self, name: str = "model") -> None:
        self.name = name
        self.blocks: Dict[str, Block] = {}
        self.target: Optional[str] = None
        self.modality: str = ""
        self._data: Dict[str, Any] = {}          # 端口 -> 内存对象（geometry/material）
        self._data_factory: Dict[str, Dict[str, Any]] = {}  # 端口 -> 重建描述
        self._params: Dict[str, Any] = {}        # 透传给求解器的全局参数（如 material）

    # ---- 积木式 API ----------------------------------------------------------
    def add_part(self, name: str, geometry: Any = None,
                 factory: Optional[Dict[str, Any]] = None, **params: Any) -> "SimulationModel":
        self.blocks[name] = Block("part", name, params)
        if geometry is not None:
            self._data["geometry"] = geometry
        if factory is not None:
            self._data_factory["geometry"] = factory
        return self

    def add_material(self, name: str, material: Any = None,
                     factory: Optional[Dict[str, Any]] = None, **params: Any) -> "SimulationModel":
        self.blocks[name] = Block("material", name, params)
        # 材料对象经 params 透传给求解器（求解器从 params["material"] 读取）
        if material is not None:
            self._params["material"] = material
        if factory is not None:
            self._data_factory["material"] = factory
        return self

    def set_param(self, key: str, value: Any) -> "SimulationModel":
        self._params[key] = value
        return self

    def add_step(self, name: str, produces: str, **params: Any) -> "SimulationModel":
        self.blocks[name] = Block("step", name, {"produces": produces, **params})
        # 最后一个 step 即目标；也可用 set_target 显式指定
        self.target = produces
        return self

    def add_bc(self, name: str, **params: Any) -> "SimulationModel":
        self.blocks[name] = Block("bc", name, params)
        return self

    def add_ic(self, name: str, **params: Any) -> "SimulationModel":
        self.blocks[name] = Block("ic", name, params)
        return self

    def add_mesh(self, name: str, **params: Any) -> "SimulationModel":
        self.blocks[name] = Block("mesh", name, params)
        return self

    def add_solver(self, name: str, **params: Any) -> "SimulationModel":
        self.blocks[name] = Block("solver", name, params)
        return self

    def add_output(self, name: str, **params: Any) -> "SimulationModel":
        self.blocks[name] = Block("output", name, params)
        return self

    def set_target(self, produces: str) -> "SimulationModel":
        self.target = produces
        return self

    def set_modality(self, modality: str) -> "SimulationModel":
        self.modality = modality
        return self

    # ---- 装配 / 运行 ---------------------------------------------------------
    def build(self, params: Optional[Dict[str, Any]] = None):
        ensure_registered()
        if not self.target:
            raise ValueError("未设置目标端口（add_step / set_target）")
        return REGISTRY.build(self.target, self.modality, params)

    def graph(self):  # 便于调试：返回装配后的图（不运行）
        return self.build()

    def run(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        g = self.build(params)
        inputs = dict(self._data)
        merged = {**self._params, **(params or {})}
        return g.run(**inputs, params=merged)

    def attach(self, port: str, obj: Any,
               factory: Optional[Dict[str, Any]] = None) -> "SimulationModel":
        """运行时挂载内存对象（geometry/material），不入 .amf 序列化体。"""
        self._data[port] = obj
        if factory is not None:
            self._data_factory[port] = factory
        return self

    # ---- 序列化 --------------------------------------------------------------
    def save(self, path: str) -> None:
        from .serialization import save_model
        save_model(self, path)

    @classmethod
    def load(cls, path: str) -> "SimulationModel":
        from .serialization import load_model
        return load_model(path)

    def summary(self) -> str:
        ensure_registered()
        if not self.target:
            return f"SimulationModel({self.name!r}) [未设目标]"
        g = self.build()
        req = g.required_inputs()
        lines = [f"SimulationModel({self.name!r})  target={self.target!r}  modality={self.modality!r}",
                 f"  需外部输入: {req}",
                 f"  求解链路({len(g.specs)} 步, 代价 {g.cost:.3g}):"]
        for i, s in enumerate(g.specs, 1):
            lines.append(f"    {i:2d}. {s.name}  -> {s.produces}")
        return "\n".join(lines)
