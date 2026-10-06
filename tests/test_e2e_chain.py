"""端到端链路与可微性回归测试 (9 契约链 / 功能 1-4 全贯通)

这些测试保护"模块之间无缝链接 + 可微分"这一核心承诺：

* 自动布线 (Pipeline.auto) 能从单一 PartGeometry 反向链式推导出完整的
  geometry -> process -> meltpool -> thermal -> microstructure ->
  constitutive -> asbuilt -> structural -> verdict 8 步路径；
* 整条链正向跑通，所有中间契约数值有限 (finite)；
* **`jax.grad` 穿透整条链不出现 NaN/Inf**（这是曾经反复炸掉功能 2 的
  sqrt(0) 逆向梯度问题，见 contracts.AsBuiltPart.von_mises_residual 与
  inverse.loss_fn 的等效应变项；本文件用断言守住它）；
* 功能 2 的可微分闭环 (inverse.train_process_predictor) 损失单调下降；
* 9 个契约各自都可由 Pipeline.auto 到达（可替换求解器 / 多候选的基础）。

运行：``pytest tests/test_e2e_chain.py -q``
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import amforge as af                                         # noqa: E402
from amforge import geometry as G                           # noqa: E402
import amforge.inverse as inv                               # noqa: E402

# 9 个统一数据契约（模块间唯一通信语言）
ALL_CONTRACTS = (
    "PartGeometry", "ProcessPlan", "ThermalHistory", "MeltPoolResult",
    "MicrostructureResult", "ConstitutiveField", "AsBuiltPart",
    "StructuralResult", "ServiceVerdict",
)


@pytest.fixture(scope="module")
def part():
    """一个 0.9mm 球（半径 0.45mm），50µm 体素 —— 任意复杂几何的代表。"""
    return G.from_sdf_fn(
        lambda x: jnp.linalg.norm(x, axis=-1) - 0.45e-3,
        bounds=[(-0.6e-3, 0.6e-3)] * 3, spacing=50e-6, name="e2e_sphere",
    )


# ===========================================================================
# 0. 导入健康检查 —— 所有内置求解器必须就绪
# ===========================================================================
def test_all_builtin_modules_imported():
    """单个模块依赖缺失不应被静默吞掉；这里明确守住 9 链所需的全部模块。"""
    bad = af.import_status()
    assert bad == {}, f"内置模块导入失败: {bad}"


def test_registry_has_full_chain_solvers():
    """注册表必须覆盖每个中间契约的至少一个可微求解器。"""
    for contract in ("ProcessPlan", "MeltPoolResult", "ThermalHistory",
                     "MicrostructureResult", "ConstitutiveField",
                     "AsBuiltPart", "StructuralResult", "ServiceVerdict"):
        specs = af.find_solvers(produces=contract, differentiable=True)
        assert specs, f"没有任何可微求解器产出 {contract}"
    # 各阶段至少有一个求解器（名字前缀即可，避免命名字面量脆化）
    for prefix in ("process.", "meltpool.", "thermal.", "micro.",
                   "constitutive.", "buildup.", "digitaltwin.",
                   "verdict.", "inverse."):
        names = af.list_solvers()
        assert any(n.startswith(prefix) for n in names), \
            f"缺少 {prefix}* 阶段求解器；已注册: {names}"


# ===========================================================================
# 1. 自动布线：从单一几何反向推导出完整 8 步路径
# ===========================================================================
def test_auto_pipeline_full_chain(part):
    pipe = af.Pipeline.auto("ServiceVerdict", modality="SLM")
    assert pipe.target == "ServiceVerdict"
    names = pipe.solver_names()
    # 必须包含从 process 到 verdict 的全部阶段
    for stage_solver in ("process.", "meltpool.", "thermal.", "micro.",
                         "constitutive.", "buildup.", "digitaltwin.",
                         "verdict."):
        assert any(stage_solver in n for n in names), \
            f"自动布线缺阶段 {stage_solver}: {names}"
    assert pipe.differentiable, "默认路径必须全可微"
    pipe.validate()  # 静态检查：每步输入都能由前序提供


def test_all_contracts_reachable(part):
    """每个中间契约都可由 Pipeline.auto 单独到达（可替换求解器的基础）。"""
    for target in ("ProcessPlan", "MeltPoolResult", "ThermalHistory",
                   "MicrostructureResult", "ConstitutiveField",
                   "AsBuiltPart", "StructuralResult", "ServiceVerdict"):
        pipe = af.Pipeline.auto(target, modality="SLM")
        assert pipe.target == target
        # 8 步完整链 + 各子目标，代价都应有限且 > 0
        assert pipe.total_cost > 0


# ===========================================================================
# 2. 正向端到端：整条链跑通且所有中间契约数值有限
# ===========================================================================
def test_pipeline_run_end_to_end(part):
    pipe = af.Pipeline.auto("ServiceVerdict", modality="SLM")
    ctx = pipe.run(geometry=part, params={"material": "316L",
                                           "n_grid": 20,
                                           "allowable_displacement": 1e-4})
    # 必须产出目标契约
    assert isinstance(ctx["verdict"], af.ServiceVerdict)

    # 中间契约全部存在且为声明类型
    expect = {
        "process": af.ProcessPlan, "meltpool": af.MeltPoolResult,
        "thermal": af.ThermalHistory, "microstructure": af.MicrostructureResult,
        "constitutive": af.ConstitutiveField, "asbuilt": af.AsBuiltPart,
        "structural": af.StructuralResult, "verdict": af.ServiceVerdict,
    }
    for slot, typ in expect.items():
        assert slot in ctx, f"缺少中间契约 {slot}"
        assert isinstance(ctx[slot], typ)

    # 关键数值量必须有限（非 NaN/Inf）
    verdict = ctx["verdict"]
    for key in ("strength_safety_factor", "stiffness_ratio",
                "fatigue_life_cycles", "wear_depth", "passed"):
        val = float(getattr(verdict, key))
        assert np.isfinite(val), f"verdict.{key} 非有限: {val}"
    asbuilt = ctx["asbuilt"]
    assert np.all(np.isfinite(np.asarray(asbuilt.residual_stress))), \
        "residual_stress 含非有限值"
    assert np.all(np.isfinite(np.asarray(asbuilt.displacement))), \
        "displacement 含非有限值"


# ===========================================================================
# 3. 可微性回归（最关键）：jax.grad 穿透整条链不得出现 NaN/Inf
#    —— 这正是之前 sqrt(0) 逆向梯度炸掉功能 2 的地方，必须守住。
# ===========================================================================
def test_loss_gradient_has_no_nan(part):
    theta = inv.mlp_init(jax.random.PRNGKey(1),
                         int(inv.geometry_features(part).shape[0]), 16,
                         n_out=len(inv.PROCESS_BOUNDS))
    loss_and_grad = jax.value_and_grad(inv.loss_fn)
    loss, grads = loss_and_grad(
        theta, part, material="316L", service_stress=150e6,
        params={"n_grid": 20, "allowable_displacement": 1e-4})
    flat = jnp.concatenate([jnp.ravel(g) for g in grads.values()])
    assert np.isfinite(float(loss)), f"loss 非有限: {float(loss)}"
    assert not bool(jnp.any(jnp.isnan(flat))), "loss 对 NN 权重的梯度含 NaN"
    assert not bool(jnp.any(jnp.isinf(flat))), "loss 对 NN 权重的梯度含 Inf"
    # 梯度不应全零（否则可微链其实没连上）
    assert float(jnp.max(jnp.abs(flat))) > 0.0, "梯度全零：可微链未真正连通"


def test_pipeline_function_gradient_has_no_nan(part):
    """通用入口 Pipeline.function() 也应能被 jax.grad 穿透且不炸。

    用 process.constant 把工艺功率作为被求导标量，跑完整链到 verdict，
    对强度安全系数求梯度。
    """
    pipe = af.Pipeline.auto(
        "ServiceVerdict", modality="SLM",
        select={"process": "process.constant"})
    f = pipe.function()  # f(params, geometry=...) -> ServiceVerdict

    def scalar(params):
        v = f(params, geometry=part)
        # 强度安全系数：越大越好，取负号做最小化目标
        return -v.strength_safety_factor

    params0 = {"process.constant": {"laser_power": jnp.asarray(200.0)}}
    grads = jax.grad(scalar)(params0)
    g = grads["process.constant"]["laser_power"]
    assert np.isfinite(float(g)), f"Pipeline 梯度非有限: {float(g)}"
    assert not bool(jnp.isnan(g)), "Pipeline 梯度含 NaN"


# ===========================================================================
# 4. 功能 2 闭环：训练使损失单调下降（可微反演真正可用）
# ===========================================================================
def test_inverse_training_converges(part):
    res = inv.train_process_predictor(
        part, material="316L", n_steps=20, learning_rate=0.03,
        n_hidden=16, seed=1, service_stress=150e6,
        params={"n_grid": 20, "allowable_displacement": 1e-4})
    hist = res["loss_history"]
    assert len(hist) == 20
    # 全程有限
    assert all(np.isfinite(h) for h in hist), "训练损失出现非有限值"
    # 单调（非严格）下降趋势：首尾下降比 > 1
    drop = hist[0] / max(hist[-1], 1e-12)
    assert drop > 1.05, f"损失未下降 (下降比={drop:.3f})"
    # 训练后整条链仍可判定
    plan = res["final_plan"]
    out = inv.simulate(part, plan, material="316L", service_stress=150e6,
                       params={"n_grid": 24, "allowable_displacement": 1e-4})
    assert isinstance(out["verdict"], af.ServiceVerdict)
    assert np.isfinite(float(out["verdict"].strength_safety_factor))


# ===========================================================================
# 5. 9 契约链全程数据契约自洽性（跨模块形态一致性）
# ===========================================================================
def test_contract_dim_consistency(part):
    """宏观物理场（thermal→verdict）应直接对齐几何栅格，保证"无缝链接"
    无需重采样；残余应力的 Voigt 分量数必须与维度匹配。

    注：MeltPoolResult 采用自身的细网格（n_grid 决定，本例 32³）是
    设计使然——下游只消费其标量汇总量（depth/width/length/temperature），
    因此不要求其栅格与几何一致；此处跳过熔池。
    """
    import jax.tree_util as jtu

    pipe = af.Pipeline.auto("ServiceVerdict", modality="SLM")
    ctx = pipe.run(geometry=part, params={"material": "316L", "n_grid": 20})

    # 残余应力 / 应力 Voigt 分量数 = 3D:6, 2D:3
    ab = ctx["asbuilt"]
    expect_comp = 6 if part.dim == 3 else 3
    assert ab.residual_stress.shape[-1] == expect_comp, \
        "residual_stress 的 Voigt 分量数应与维度匹配"
    assert ctx["structural"].stress.shape[-1] == expect_comp, \
        "structural.stress 的 Voigt 分量数应与维度匹配"

    # 宏观场（thermal/micro/constitutive/asbuilt/structural）的主空间场
    # 必须与几何栅格对齐（shape[:dim] == part.shape）
    for slot in ("thermal", "microstructure", "constitutive",
                 "asbuilt", "structural"):
        c = ctx[slot]
        aligned = False
        for leaf in jtu.tree_leaves(c):
            if hasattr(leaf, "shape") and jnp.ndim(leaf) >= part.dim:
                if tuple(leaf.shape[:part.dim]) == part.shape:
                    aligned = True
                    break
        assert aligned, f"{slot} 没有任何空间场对齐到几何栅格 {part.shape}"
