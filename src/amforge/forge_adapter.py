"""AM 适配层：把 amforge 的求解器注册进 ForgeCore 通用注册表。

端口用「插槽名」作标识（geometry / process / meltpool / thermal /
microstructure / constitutive / asbuilt / structural / verdict），
与 amforge 的契约类名解耦，使 ForgeCore 成为与具体物理域无关的通用引擎。

注册后，任意 SimulationModel 都能用统一接口跑 AM 全链路，且 jax.grad
可穿透整图（可微优化「免费」获得）。

设计要点：
* process 有两个生产者。让几何感知的 heuristic(0.5) 成本低于常量
  constant(1.0)，于是「给定几何时默认走 heuristic，缺几何时回退 constant」。
* 高保真求解器（meltpool.vof_flow3d / thermal.enthalpy 等）在此追加注册，
  并用 select={"thermal": "thermal.enthalpy"} 显式切换（不破坏默认轻量链路）。
"""
from __future__ import annotations

import inspect
from typing import Optional

from forgecore.registry import REGISTRY
from amforge.core.registry import get_solver as _get_native_solver

from amforge.meltpool import (
    solve_meltpool_surrogate, solve_meltpool_fdm, solve_meltpool_vof,
)
from amforge.thermal import solve_thermal_history
from amforge.thermal_enthalpy import solve_enthalpy_thermal
from amforge.micro import solve_microstructure
from amforge.constitutive import solve_constitutive
from amforge.buildup import solve_buildup
from amforge.asbuilt_plastic import solve_asbuilt_plastic
from amforge.support import support_auto, support_simulate
from amforge.postprocess_secondary import (
    postprocess_passthrough, ht_relax, hip_densify, subtractive,
)
from amforge.digitaltwin import solve_digital_twin
from amforge.verdict import solve_verdict
from amforge.process import _solve_constant as solve_process_constant
from amforge.process import heuristic_plan
from amforge.inverse import solve_inverse_process
from amforge.monitoring import (
    ingest_raw_stream, monitor_detect, monitor_correct,
)
from amforge.closedloop import run_loop

_DONE = False


def _native_cost(name: str, default: float) -> float:
    """取 amforge.core.registry 原生求解器的 cost（单一事实源）。

    forge_adapter 把 amforge 求解器镜像进 ForgeCore 注册表时，cost 应从原生
    spec 派生，避免两套表出现字面量漂移（audit 标记的 ``process.constant``
    类发散）。取不到时回落到显式默认。
    """
    try:
        return float(_get_native_solver(name).cost)
    except Exception:
        return default


# heuristic_plan 以 **params 展开接收参数，但只接受其签名内的命名参数；
# forgecore 的全局 params 可能含其它求解器的键（如 n_grid），需按签名过滤，
# 否则会 TypeError: unexpected keyword argument。
_HEUR_ALLOWED = tuple(
    n for n, p in inspect.signature(heuristic_plan).parameters.items()
    if n != "geometry"
)


def _solve_heuristic_adapter(*, geometry, params=None):
    p = dict(params or {})
    kw = {k: p[k] for k in _HEUR_ALLOWED if k in p}
    return heuristic_plan(geometry, **kw)


def register_am_solvers(force: bool = False) -> None:
    global _DONE
    if _DONE and not force:
        return
    R = REGISTRY

    # 工艺规划：process.constant 是 amforge.core.registry 的原生注册（见
    # process.py 的 @register_solver，cost=0.05，被 amforge.core.graph 活路径消费）。
    # 此处仅把同一求解器「镜像」进 ForgeCore 通用注册表，其 cost/doc 直接取自
    # 原生 spec（单一事实源），避免两处字面量漂移——这正是 audit 标记的
    # 「双重注册」真正要消除的发散（forge_adapter 曾写死 cost=1.0，与原生 0.05 相反，
    # 会让两套引擎的自动选路偏好不一致）。
    try:
        _const_spec = _get_native_solver("process.constant")
        _const_cost, _const_doc = float(_const_spec.cost), _const_spec.doc
    except Exception:  # pragma: no cover - 防御性兜底
        _const_cost, _const_doc = 0.05, "params 指定的均匀工艺方案（无几何依赖）。"
    R.register_fn(
        "process.constant", "process", (),
        fn=solve_process_constant, cost=_const_cost,
        defaults={"modality": "SLM"},
        doc=_const_doc,
    )
    R.register_fn(
        "process.heuristic", "process", ("geometry",),
        fn=_solve_heuristic_adapter, cost=0.5, modality=("SLM", "LSF"),
        defaults={"material": "Ti6Al4V", "modality": "SLM", "target_enthalpy": 12.0},
        doc="几何感知规则式工艺规划：悬垂/薄壁降功率 + 无量纲焓锁定工艺窗口。",
    )

    # 宏观点-线尺度
    R.register_fn(
        "meltpool.surrogate_eagar_tsai", "meltpool", ("geometry", "process"),
        fn=solve_meltpool_surrogate, cost=1.0, modality=("SLM", "LSF"),
        tags=("analytical",),
        doc="Eagar-Tsai 半解析熔池代理模型（闭式/解析近似，默认链路排除）。",
    )
    R.register_fn(
        "thermal.history", "thermal", ("geometry", "process", "meltpool"),
        fn=solve_thermal_history, cost=2.0, modality=("SLM", "LSF"),
        tags=("analytical",),
        doc="由逐道熔池累积零件级热历史（Rosenthal 闭式降阶模型，默认链路排除）。",
    )
    # 真实数值熔池（默认链路优先选这两种，而非闭式解析代理）：
    #   * meltpool.fdm      —— 便宜 3D 瞬态焓法热传导（无自由界面），默认档；
    #   * meltpool.vof_flow3d —— Flow3D 式 VOF 自由界面 CFD，高保真档（select 切换）。
    R.register_fn(
        "meltpool.fdm", "meltpool", ("geometry", "process"),
        fn=solve_meltpool_fdm, cost=_native_cost("meltpool.fdm", 8.0),
        modality=("SLM", "LSF"), tags=("numerical",),
        doc="3D 瞬态焓法热传导熔池（移动高斯热源+固液潜热）：数值 FVM，无自由界面，便宜真实档。",
    )
    R.register_fn(
        "meltpool.vof_flow3d", "meltpool", ("geometry", "process"),
        fn=solve_meltpool_vof, cost=_native_cost("meltpool.vof_flow3d", 100.0),
        modality=("SLM", "LSF"), tags=("numerical", "high-fidelity", "cfd"),
        doc="Flow3D 式 VOF 自由界面熔池 CFD（表面张力/Marangoni/反冲压力/糊状区）。",
    )
    R.register_fn(
        "thermal.enthalpy", "thermal", ("geometry", "process"),
        fn=solve_enthalpy_thermal,
        cost=_native_cost("thermal.enthalpy", 2.0), modality=("SLM", "LSF"),
        tags=("high-fidelity", "phase-change"),
        doc="enthalpy 法相变热传导（移动高斯热源 + 固液潜热），高保真热解；"
            "用 select={'thermal':'thermal.enthalpy'} 显式启用。",
    )

    # 微观 -> 本构
    R.register_fn(
        "micro.surrogate", "microstructure", ("thermal", "meltpool"),
        fn=solve_microstructure, cost=2.0, modality=("SLM", "LSF"),
        doc="冷却率幂律晶粒尺寸 + 柱状分数 + 孔隙的微观代理。",
    )
    R.register_fn(
        "constitutive.homogenize", "constitutive", ("microstructure",),
        fn=solve_constitutive, cost=1.0,
        doc="Hall-Petch 屈服 + 孔隙折减 + 各向异性的宏观本构均质化。",
    )

    # 成形 + 数字样机
    R.register_fn(
        "buildup.layer_activation", "asbuilt",
        ("geometry", "process", "thermal", "microstructure"),
        fn=solve_buildup, cost=3.0,
        doc="逐层激活：热应变/残余应力/翘曲/变形后的成形件（降级趋势模型）。",
    )
    R.register_fn(
        "asbuilt.thermomechanical_plastic", "asbuilt",
        ("geometry", "thermal"),
        fn=solve_asbuilt_plastic, cost=4.0,
        tags=("high-fidelity", "elasto-plastic", "cpfe"),
        doc="塑性/CPFE 生死单元 FEM 高保真成形（J2 各向同性硬化 或 速率相关晶体塑性）；"
            "默认链路不启用（避免 OOM），须用 select={'asbuilt':'asbuilt.thermomechanical_plastic'} "
            "显式切换。",
    )

    # 模块A：支撑结构（新增 support 阶段）
    # 代理默认档 support.auto（秒级、默认链路）：悬垂角规则自动布点支撑。
    R.register_fn(
        "support.auto", "SupportStructure", ("geometry",),
        fn=support_auto, cost=_native_cost("support.auto", 1.0),
        modality=("SLM",),
        defaults={"kind": "block", "overhang_angle_deg": 45.0, "density": 1.0},
        doc="悬垂角阈值规则自动布点支撑（block / overhang），产出 SupportStructure。",
    )
    # 高保真档 support.simulate：复用 asbuilt 逐层热-力耦合，支撑作为约束/散热边界
    # 修正成形；默认不启用，须 select={'support':'support.simulate'} 显式切换。
    R.register_fn(
        "support.simulate", "AsBuiltPart",
        ("geometry", "support", "process"),
        fn=support_simulate, cost=_native_cost("support.simulate", 50.0),
        modality=("SLM",),
        tags=("high-fidelity", "support-coupled", "elasto-plastic"),
        doc="合并几何(零件∪支撑)复用 asbuilt 逐层热-力耦合，支撑作为约束/散热边界修正成形；"
            "默认链路不启用，须用 select={'support':'support.simulate'} 显式切换。",
    )

    # 模块B：二次工艺（新增 postprocess 阶段）
    # 默认档 postprocess.passthrough（cost=0.1，直通"无二次工艺"）：是 verdict 求解
    # secondary 端口时的最低代价生产者，使默认链路等价直接用成形件（向后兼容）。
    R.register_fn(
        "postprocess.passthrough", "secondary", ("asbuilt",),
        fn=postprocess_passthrough, cost=_native_cost("postprocess.passthrough", 0.1),
        modality=(),
        doc="无二次工艺（直通）：把成形件原样包成 SecondaryProcessResult。",
    )
    # HT / HIP 高保真档（cost=5，默认链路不启用，须 select 显式切换）
    R.register_fn(
        "postprocess.heat_treatment", "secondary", ("asbuilt",),
        fn=ht_relax, cost=_native_cost("postprocess.heat_treatment", 5.0),
        modality=("SLM", "LSF"), tags=("high-fidelity", "stress-relief"),
        doc="去应力退火/固溶：Arrhenius 应力松弛+再结晶（可微闭式）。",
    )
    R.register_fn(
        "postprocess.hip", "secondary", ("asbuilt",),
        fn=hip_densify, cost=_native_cost("postprocess.hip", 5.0),
        modality=("SLM", "LSF"), tags=("high-fidelity", "densification"),
        doc="热等静压：孔隙幂律闭合→~1.0 + 高温应力释放（可微闭式）。",
    )
    # machining 确定性档（differentiable=False，几何布尔修形，须 select 显式切换）
    R.register_fn(
        "postprocess.machining", "secondary", ("asbuilt", "geometry"),
        fn=subtractive, cost=_native_cost("postprocess.machining", 2.0),
        modality=("SLM", "LSF"), differentiable=False, tags=("subtractive",),
        doc="切削精加工：余料去除+最终形位（几何布尔修形），喂给 verdict。",
    )
    R.register_fn(
        "digitaltwin.rom", "structural", ("asbuilt", "constitutive"),
        fn=solve_digital_twin, cost=4.0,
        doc="降阶数字样机：代表服役应力下的位移与应力集中。",
    )
    R.register_fn(
        "verdict.service", "verdict", ("structural", "asbuilt", "constitutive", "secondary"),
        fn=solve_verdict, cost=1.0,
        doc="强度/刚度/疲劳/磨损安全系数与服役判定（含可选二次工艺 secondary 修正）。",
    )

    # ---- 模块 D：监测闭环（monitor 阶段） -------------------------------
    R.register_fn(
        "monitor.detect", "monitoring", ("sensor", "thermal"),
        fn=monitor_detect, cost=_native_cost("monitor.detect", 3.0),
        differentiable=False, tags=("monitor", "defect-detection"),
        doc="从传感流+热历史推断未熔合/匙孔/孔隙缺陷概率。",
    )
    R.register_fn(
        "monitor.correct", "closedloop", ("monitoring", "process"),
        fn=monitor_correct, cost=_native_cost("monitor.correct", 2.0),
        differentiable=False, tags=("monitor", "closed-loop-control"),
        doc="基于缺陷状态给出修正后工艺方案（闭环喂回前向仿真）。",
    )

    # 可微逆问题（功能 2 入口）：几何 -> 工艺
    R.register_fn(
        "inverse.predict_process", "process", ("geometry",),
        fn=solve_inverse_process,
        cost=_native_cost("inverse.predict_process", 0.7), differentiable=True,
        doc="神经网络从几何预测工艺（功能2 可微闭环的可插拔求解器）。",
    )

    _DONE = True


# 导入即注册（幂等）
register_am_solvers()
