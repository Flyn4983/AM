"""ForgeCore 通用引擎的端到端回归测试（AM 适配层）。

守住：
1. SimulationModel + 积木式 API 能自动选路跑通 AM 全链路；
2. 所有中间契约数值有限；
3. jax.grad 穿透整图无 NaN/Inf（对应 contracts.py / inverse.py 的 sqrt(0) 修复）；
4. .amf 工程文件序列化往返一致，重载后仍能重建并运行。
"""
import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import amforge.forge_adapter  # 触发 AM 求解器注册
from amforge import geometry as G
from forgecore import REGISTRY, SimulationModel
from forgecore.model import ensure_registered


def _make_geo():
    return G.from_sdf_fn(
        lambda x: jnp.linalg.norm(x, axis=-1) - 0.45e-3,
        bounds=[(-0.6e-3, 0.6e-3)] * 3,
        spacing=50e-6,
        name="demo",
    )


def _make_model(geo):
    ensure_registered()
    m = SimulationModel("demo")
    m.add_part("part", geometry=geo)
    m.add_material("mat", material="316L")
    m.add_step("verdict", produces="verdict")
    m.set_modality("SLM")
    return m


def test_registry_has_am_solvers():
    ensure_registered()
    names = REGISTRY.list_names()
    for n in ("process.heuristic", "meltpool.surrogate_eagar_tsai",
              "thermal.history", "micro.surrogate", "constitutive.homogenize",
              "buildup.layer_activation", "digitaltwin.rom", "verdict.service",
              "inverse.predict_process"):
        assert n in names, f"缺少求解器 {n}"


def test_mirror_costs_match_native():
    """forge_adapter 镜像进 ForgeCore 的求解器，cost 必须与原生 amforge 注册一致。

    防止 audit 标记的 ``process.constant`` 类「镜像 cost 字面量漂移」复发：
    镜像注册应把 cost 取为单一事实源（原生 spec），而非各自硬编码。
    """
    import amforge.forge_adapter  # 触发 AM 求解器注册
    import amforge.core.registry as A
    diverged = []
    for name in REGISTRY.list_names():
        if name in A._SOLVERS:
            fc = REGISTRY.get(name).cost
            am = A.get_solver(name).cost
            if fc != am:
                diverged.append((name, fc, am))
    assert not diverged, f"forge_adapter 镜像 cost 与原生注册表发散: {diverged}"


def test_auto_wiring_reaches_verdict():
    ensure_registered()
    g = REGISTRY.build("verdict", modality="SLM", params={"material": "316L", "n_grid": 20})
    produced = {s.produces for s in g.specs}
    for port in ("process", "meltpool", "thermal", "microstructure",
                 "constitutive", "asbuilt", "structural", "verdict"):
        assert port in produced, f"链路未产出 {port}"
    # 几何为外部输入
    assert "geometry" in g.required_inputs()
    # 总代价为各步之和（>0）
    assert g.cost > 0.0


def test_end_to_end_finite():
    geo = _make_geo()
    m = _make_model(geo)
    ctx = m.run(params={"n_grid": 20, "allowable_displacement": 1e-4})

    # 全部中间契约必须有限
    for port, val in ctx.items():
        leaves = jax.tree_util.tree_leaves(val)
        for leaf in leaves:
            if hasattr(leaf, "shape") and jnp.ndim(leaf) >= 0:
                arr = jnp.asarray(leaf)
                assert jnp.all(jnp.isfinite(arr)), f"端口 {port} 含非有限值"

    # 末端判定合理
    verdict = ctx["verdict"]
    assert hasattr(verdict, "strength_safety_factor")
    assert jnp.isfinite(verdict.strength_safety_factor)


def test_gradient_through_full_chain_no_nan():
    """关键回归：jax.grad 穿透整条 AM 链（含 von_mises_residual / 等效应变 sqrt）无 NaN。"""
    geo = _make_geo()
    m = _make_model(geo)
    g = m.build(params={"material": "316L", "n_grid": 20})
    f = g.as_callable(trace_ports=["geometry"])

    def loss(sdf_tracer):
        g2 = G.PartGeometry(
            sdf=sdf_tracer, origin=geo.origin, spacing=geo.spacing,
            dim=geo.dim, name=geo.name,
        )
        return f(g2).strength_safety_factor

    grad = jax.grad(loss)(geo.sdf)
    assert jnp.all(jnp.isfinite(grad)), "整链梯度含 NaN/Inf（sqrt(0) 类 bug 复发）"
    # 梯度不应全零（链路确实连接到几何）
    assert jnp.max(jnp.abs(grad)) > 0.0


def test_amf_serialization_roundtrip():
    geo = _make_geo()
    m = _make_model(geo)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "demo.amf")
        m.save(path)
        assert os.path.exists(path)

        m2 = SimulationModel.load(path)
        assert m2.target == "verdict"
        assert m2.modality == "SLM"
        # 重载后重新挂载几何（.amf 不内联张量），应能重建图并运行
        m2.attach("geometry", geo)
        g2 = m2.build(params={"material": "316L", "n_grid": 20})
        assert g2.target == "verdict"
        ctx2 = g2.run(geometry=geo, params={"material": "316L", "n_grid": 20})
        assert "verdict" in ctx2
        assert jnp.isfinite(ctx2["verdict"].strength_safety_factor)
