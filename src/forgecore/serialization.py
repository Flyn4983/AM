"""ForgeCore 序列化：模型以 JSON 落盘为 *.amf 工程文件。

只序列化「配置级」信息（积木块、目标端口、模态、数据重建描述），
重计算场与内存对象（geometry/material 张量）不内联，而用 factory 描述重建。
加载时可经已注册的 REBUILDERS 还原内存对象；否则需用户运行时 attach。
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict

from .blocks import Block
from .model import SimulationModel

SCHEMA = "amforge.forgecore.model/v1"

# 端口 -> 重建函数（adapter 注册）。签名 fn(factory: dict) -> 内存对象
REBUILDERS: Dict[str, Callable[[Dict[str, Any]], Any]] = {}


def register_rebuilder(port: str, fn: Callable[[Dict[str, Any]], Any]) -> None:
    REBUILDERS[port] = fn


def save_model(model: SimulationModel, path: str) -> None:
    payload = {
        "schema": SCHEMA,
        "name": model.name,
        "target": model.target,
        "modality": model.modality,
        "blocks": [b.to_dict() for b in model.blocks.values()],
        "data_factories": model._data_factory,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_model(path: str) -> SimulationModel:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("schema") != SCHEMA:
        raise ValueError(f".amf schema 不匹配: {payload.get('schema')!r}")
    m = SimulationModel(payload.get("name", "model"))
    for bd in payload.get("blocks", []):
        b = Block.from_dict(bd)
        m.blocks[b.name] = b
    m.target = payload.get("target")
    m.modality = payload.get("modality", "")
    factories = payload.get("data_factories", {})
    m._data_factory = factories
    # 尝试按 factory 重建内存对象
    for port, fac in factories.items():
        fn = REBUILDERS.get(port)
        if fn is not None:
            try:
                m._data[port] = fn(fac)
            except Exception:
                pass
    return m
