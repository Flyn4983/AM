"""通用积木块：模型层的最小可序列化单元。"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Block:
    """模型树中的一个节点（Part / Material / Step / BC / IC / Mesh / Solver / Output）。

    参数只存「配置」级别的小数据（标量、字符串、短列表），
    不存重计算场；重场由求解器在 run() 时产出。可经 .to_dict() 落 .amf。
    """

    kind: str
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "version": self.version,
            "params": _serialize_params(self.params),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Block":
        return cls(
            kind=d["kind"],
            name=d["name"],
            version=d.get("version", 1),
            params=_deserialize_params(d.get("params", {})),
        )


def _serialize_params(p: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in p.items():
        if callable(v):
            # 函数（如 sdf_fn）无法 JSON 化，仅保留其名作为占位
            out[k] = {"__callable__": getattr(v, "__name__", "fn")}
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = [_scalar(x) for x in v]
        else:
            out[k] = str(v)
    return out


def _scalar(x):
    try:
        f = float(x)
        return int(f) if f.is_integer() else f
    except (TypeError, ValueError):
        return str(x)


def _deserialize_params(d: Dict[str, Any]) -> Dict[str, Any]:
    # 配置级往返；callable 占位不还原（运行时再绑定）
    return dict(d)
