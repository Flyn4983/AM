"""可微分工艺反演（inverse —— 功能 2 的核心）
=============================================

实现用户需求中的**功能 2**：

    "具备可微分模拟能力——通过深度网络从待制造几何构型预测制造工艺
    （激光功率、扫描速度、路径等），根据工艺运行模拟模块得到模拟后构型，
    与理想构型比较（以几何尺寸差异、残余应力、应变等建损失函数），对网络
    参数求梯度反向修正并更新，迭代直至收敛。"

本模块把"几何 → 工艺"建成一个**可训练的深度网络**，把整条物理链
（工艺 → 熔池 → 热历史 → 微观 → 本构 → 成形 → 数字样机 → 判定）作为网络的
**可微前向**，于是损失对网络权重的梯度可以一路回传。这与普通"代理模型"
的本质区别：我们不是在逼近一个已存在的物理代码，而是让物理代码本身被
``jax.grad`` 穿透，成为优化器的"前向传播"。

流程
----
1. 几何特征提取：体积、悬垂分数、包围盒尺寸、层数 -> 特征向量 x。
2. 神经网络 g_θ：x -> z ∈ [0,1]^9（归一化工艺向量）。
3. ``denormalize_process``：z -> 物理有界的 ``ProcessPlan``（天然满足设备约束）。
4. 物理前向 F：ProcessPlan -> ServiceVerdict + AsBuiltPart + MeltPoolResult。
5. 损失 L(θ) = w₁·几何偏差 + w₂·残余应力 + w₃·残余应变 + w₄·缺陷。
6. ``θ ← θ − η·∇_θ L``（optax/adam），迭代直至收敛。

因为每一步都是纯 JAX 函数，``jax.grad(L)(θ)`` 自动穿过物理求解器——
这正是"无缝链接 + 可微分"的兑现。
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from typing import Any, Mapping, Sequence

from amforge.nsga2 import nsga2_minimize, NSGA2Config

from amforge.core.contracts import (
    PartGeometry, ProcessPlan, MeltPoolResult, ThermalHistory,
    MicrostructureResult, ConstitutiveField, AsBuiltPart, StructuralResult,
    ServiceVerdict, solid_weight,
)
from amforge.core.registry import register_solver
from amforge.materials import get_material
from amforge.process import (
    denormalize_process, normalize_process, heuristic_plan, clip_to_bounds,
    PROCESS_BOUNDS,
)

# 物理前向求解器（均为纯函数、已注册、可微）
from amforge.meltpool import solve_meltpool_surrogate
from amforge.thermal import solve_thermal_history
from amforge.thermal_enthalpy import solve_enthalpy_thermal, chain_schedule
from amforge.micro import solve_microstructure
from amforge.phasefield import solve_phasefield  # 高保真枝晶相场（micro.phasefield）
from amforge.mbd import solve_assembly  # 多体 + FEM 耦合装配（mbd.assembly）
from amforge.powderbed import solve_powderbed  # 粉末尺度缺陷（powder.bed）
from amforge.constitutive import solve_constitutive
from amforge.buildup import solve_buildup
from amforge.asbuilt_plastic import solve_asbuilt_plastic
from amforge.digitaltwin import solve_digital_twin
from amforge.verdict import solve_verdict


# ===========================================================================
# 1. 神经网络（纯 JAX，参数即 pytree）
# ===========================================================================
def mlp_init(rng, n_in: int, n_hidden: int, n_out: int = 9,
             n_layers: int = 2) -> dict:
    """初始化一个 (n_in -> n_hidden -> ... -> n_out) 的 MLP 参数 pytree。

    最后一层用 ``sigmoid`` 输出，使 z ∈ [0,1]，对接 ``denormalize_process``。
    """
    rngs = jax.random.split(rng, n_layers * 2)
    params: dict[str, Any] = {}
    dims = [n_in] + [n_hidden] * (n_layers - 1) + [n_out]
    for i in range(n_layers):
        key_w, key_b = rngs[2 * i], rngs[2 * i + 1]
        # Xavier/Glorot 初始化
        scale = jnp.sqrt(2.0 / (dims[i] + dims[i + 1]))
        params[f"W{i}"] = jax.random.normal(key_w, (dims[i + 1], dims[i])) * scale
        params[f"b{i}"] = jnp.zeros((dims[i + 1],))
    return params


def mlp_apply(params: dict, x: jnp.ndarray) -> jnp.ndarray:
    """前向：返回 z ∈ [0,1]^n_out。"""
    h = jnp.asarray(x, dtype=jnp.float64)
    keys = sorted(k for k in params if k.startswith("W"))
    for i, wk in enumerate(keys):
        bk = "b" + wk[1:]
        h = params[wk] @ h + params[bk]
        if i < len(keys) - 1:
            h = jnp.tanh(h)
        else:
            h = jax.nn.sigmoid(h)
    return h


# ===========================================================================
# 2. 几何特征提取
# ===========================================================================
def geometry_features(part: PartGeometry, *, nominal_layer_thickness: float = 40e-6
                      ) -> jnp.ndarray:
    """把任意复杂几何压成固定长度特征向量（网络输入）。"""
    vol = part.volume()
    try:
        overhang = part.overhang_fraction()
    except Exception:
        overhang = jnp.asarray(0.0)
    lo, hi = part.bbox()
    dims = hi - lo
    n_layers = part.layer_count(nominal_layer_thickness)
    feats = jnp.stack([
        jnp.log10(jnp.maximum(vol, 1e-12)),     # 体积（对数，跨量级）
        overhang,
        dims[0], dims[1], dims[2],              # 包围盒三向尺寸
        jnp.asarray(float(n_layers)),           # 层数
    ])
    return feats


# ===========================================================================
# 3. 物理前向：ProcessPlan -> (verdict, asbuilt, meltpool)
# ===========================================================================
def _run_thermal(name, *, geometry, process, meltpool, params):
    """宏观热学求解器分发：真实潜热相变焓法（enthalpy）支持 2D 与 3D 数值热固结。

    ``name`` 取值：
      * ``"enthalpy"`` / ``"thermal.enthalpy"``：:func:`solve_enthalpy_thermal`
        （糊状区有效扩散系数 + SSP-RK2 显式相变热传导，**真实固液相变潜热**；
        2D/3D 均为数值求解，全程对工艺参数与几何坐标可微）；
      * 其它（默认 ``"history"``）：:func:`solve_thermal_history`
        （Rosenthal 闭式降阶模型，无潜热，仅用于快速回归/教学演示）。

    两求解器同契约 ``consumes=(PartGeometry, ProcessPlan)`` → ``ThermalHistory``，
    故可无缝互换；默认链（生产/可微逆问题）使用真实相变热解。
    """
    if name in ("enthalpy", "thermal.enthalpy"):
        return solve_enthalpy_thermal(geometry=geometry, process=process, params=params)
    return solve_thermal_history(geometry=geometry, process=process,
                                 meltpool=meltpool, params=params)


_EXPLICIT_THERMAL = ("enthalpy", "thermal.enthalpy")

# 显式热解调度**真正读到**的工艺叶子：路径长只由几何 extent 与
# (layer_thickness, hatch_spacing) 决定，曝光上界再除以 scan_speed，半径下界由
# dx 定价（resolution_policy 判据）。laser_power / absorption 只改沉积能量，
# **不改时间离散**——所以 ``jax.grad`` 只对功率求导（最常见的
# ``plan.replace(laser_power=P)``）时档位照样能在 trace 入口自动钉好，
# 不该被当成"trace 内不可算"而退回未钉档的报错。
_TIER_FIELDS = ("scan_speed", "layer_thickness", "hatch_spacing", "beam_radius")


def _is_traced(plan: ProcessPlan) -> bool:
    """调度所依赖的工艺叶子里是否含 JAX tracer（= 无法在 trace 内固化静态调度）。"""
    from jax.core import Tracer
    vals = [getattr(plan, f, None) for f in _TIER_FIELDS]
    return any(v is not None and isinstance(jnp.asarray(v), Tracer) for v in vals)


# 链条/装配/GUI 档（demo）的代价闸门（§25.7 实测成本律的工程化）：一次正向显式热解
# ≈ n_steps × 体素数 voxel-step，CPU f64 实测 0.75–2.40 M/s。默认链条要能在测试与
# 交互里秒级跑完，故缺省把单次正向压在 ~5e6 voxel-step（≈2–6s CPU）。预算不够时
# 走 ``over_budget="pin"``：**抬 scan_speed 下界**把窗口压进预算（dt 精度不赔，
# 代价是更慢的扫描在本档不可达），而不是放大 dt。标定档（strict）不设体素预算，
# 预算不够就报错——标定不该悄悄缩小工艺窗口。
_DEMO_TIER_VOXEL_STEPS = 5_000_000


def thermal_tier(p: dict, geometry: PartGeometry, nominal_plan: ProcessPlan, *,
               material: str, thermal_solver: str = "enthalpy",
               resolution_policy: str = "demo") -> dict:
    """把「这张网格上显式瞬态热解的静态调度 + 与之自洽的工艺子盒」备进 params。

    D0（2026-10-06）的根因：A0 把 ``dt`` 钉成 **物理曝光/n_steps** 之后，
    ``n_steps`` 是 ``jax.lax.scan`` 的**静态形状**，只能是 trace 之外的量；而工艺
    搜索盒里任何一点都可能把曝光推长（更慢的 ``scan_speed``、更密的
    ``hatch_spacing``/更薄的 ``layer_thickness``）。所以两者必须**同时**钉：

    * ``p["thermal"]``  ← :func:`amforge.thermal_enthalpy.chain_schedule` 在盒的
      最坏角落算出的 ``n_steps`` 与 ``exposure_bound_s``（后者让求解器的三条时间
      离散校核在 ``jax.grad`` 内也能执行，不再整体跳过）；
    * ``p["process_bounds"]`` ← 同一档位给出的工艺子盒（设备盒 ∩ 网格可分辨 ∩
      步数预算内可积分）。z↔工艺 的映射用它，于是优化器**走不出**钉住的调度，
      而不是走出去后靠报错崩溃或靠跳过校核静默发散。

    调用方已显式给出 ``p["thermal"]["n_steps"]`` 时**尊重该调度**（子盒反推成"这个
    步数覆盖得住曝光"）；非显式热解档（``thermal_solver="history"`` 等）不加任何键。

    ``resolution_policy``：链条/装配/GUI 这类"跑通即可"的岗位缺省 ``"demo"``（网格
    装不下光斑时把光源半径抬到 dx/2 并警告），标定档由调用方传 ``"strict"`` 恢复
    硬报错（§25.10 的严重度分层）。**只有调度真正读到的那四个叶子**（scan_speed /
    layer_thickness / hatch_spacing / beam_radius）是 tracer 时才跳过重算——只对
    ``laser_power`` 求导的常见写法依然自动钉档。trace 内跳过时**只**沿用外层已钉好
    的档位，不再重算：真实校核错误必须在 eager 报出，不能在这里被吞掉。

    demo 档自带代价闸门 ``max_voxel_steps``（缺省 5e6）与 ``over_budget="pin"``：
    诚实调度（覆盖最坏角落曝光）在这张网格上超出闸门时，把 ``scan_speed`` 下界抬进
    闸门并警告，而不是放大 dt 或截断曝光；若连抬到设备上界都不够（部件级），
    报错并指向 §25.8 的 D2/D3。可用 ``params["thermal"]["max_voxel_steps"]`` 或
    ``params["thermal_max_voxel_steps"]`` 覆盖。
    """
    p = dict(p)
    if thermal_solver not in _EXPLICIT_THERMAL:
        return p
    tp = dict(p.get("thermal") or {})
    tp.setdefault("resolution_policy", resolution_policy)
    demo = tp["resolution_policy"] != "strict"
    pinned = tp.get("n_steps") is not None
    kw = dict(material=tp.get("material", material),
              cfl=float(tp.get("cfl", 0.35)),
              speed_slack=float(p.get("thermal_speed_slack", 2.0)),
              path_slack=float(p.get("thermal_path_slack", 2.0)),
              max_steps=int(tp.get("max_steps", p.get("thermal_max_steps", 200000))),
              max_voxel_steps=tp.get("max_voxel_steps",
                                     p.get("thermal_max_voxel_steps",
                                           _DEMO_TIER_VOXEL_STEPS if demo else None)),
              over_budget=str(tp.get("over_budget",
                                     p.get("thermal_over_budget",
                                           "pin" if demo else "raise"))),
              resolution_policy=tp["resolution_policy"])
    if pinned:
        kw["fixed_n_steps"] = int(tp["n_steps"])
    if _is_traced(nominal_plan):
        # trace 内读不到工艺值，无法用它给调度定价。退化到**这张网格上的规则式名义
        # 工艺**（heuristic，具体值）——与 :func:`loss_fn` 的钉档策略同源，于是
        # 「``jax.grad`` 直接穿透 ``simulate`` 对工艺求导」这条产品主回路不必要求
        # 调用方手工钉档。几何本身也是 tracer（形状优化）时无从定价：沿用外层已钉
        # 好的档位；若外层也没钉，求解器会因缺少静态调度而**报错**，不静默发散。
        try:
            nominal_plan = heuristic_plan(geometry, material=kw["material"],
                                          modality=nominal_plan.modality)
        except (jax.errors.ConcretizationTypeError, TypeError, ValueError):
            return p
        if pinned or _is_traced(nominal_plan):
            return p
    sched = chain_schedule(geometry, nominal_plan, **kw)
    bounds = sched.pop("bounds")
    if not pinned:
        tp["n_steps"] = sched["n_steps"]
        tp["max_steps"] = sched["max_steps"]
    tp.setdefault("exposure_bound_s", sched["exposure_bound_s"])
    p["thermal"] = tp
    p.setdefault("process_bounds", bounds)
    return p


def tier_bounds(p: Mapping[str, Any] | None):
    """取当前档位的工艺子盒（无则 ``None``=原设备盒，逐位向后兼容）。"""
    return dict(p["process_bounds"]) if (p or {}).get("process_bounds") else None


def z_to_device_box(z, *, n_layers: int = 1, bounds=None,
                    modality: str = "SLM"):
    """把**档位子盒坐标**的 z 换算回**设备盒坐标**的 z（交给外部消费者之前用）。

    ``z`` 的公开语义始终是"相对设备工艺盒 ``PROCESS_BOUNDS`` 归一化"——
    :func:`process_constraint_penalty`、:func:`process_feasibility`、报告与前端都按
    设备盒解释它。D0 之后优化器**内部**把 z 映射进"本网格分辨得出、且钉住的显式
    调度覆盖得了曝光"的那部分盒子，内部坐标与公开坐标于是分叉：把内部的 z 原样交
    出去，外部按设备盒一解释就得到一个不相干的工艺（实测 ``feas≈4e-4``）。子盒是
    设备盒的子集，故这次往返换算无损，只是将坐标对齐回文档承诺的那一套。
    """
    if not bounds:
        return z
    plan = denormalize_process(z, n_layers=n_layers, modality=modality, bounds=bounds)
    return normalize_process(plan)


def simulate(geometry: PartGeometry, process: ProcessPlan, *,
             thermal_solver: str = "enthalpy",
             material: str = "316L", service_stress: float = 150e6,
             params: Mapping[str, Any] | None = None,
             asbuilt_solver: str = "buildup",
             micro_solver: str = "surrogate",
             mbd_solver: str = "surrogate",
             powder_solver: str | None = None,
             constitutive: str = "j2") -> dict:
    """跑完整的物理链，返回末端契约供损失/判定使用。

    所有求解器都是纯函数且可微，因此本函数整体可被 ``jax.grad`` 穿透。

    ``asbuilt_solver`` 选择"制造后构型"生产者：
      - ``"buildup"``（默认）：降阶趋势模型 :func:`amforge.buildup.solve_buildup`；
      - ``"plastic"``：高保真塑性/CPFE 生死单元 FEM
        :func:`amforge.asbuilt_plastic.solve_asbuilt_plastic`
        （真实弹塑性收缩/翘曲，与 digitaltwin/verdict 同一 ``AsBuiltPart`` 契约）。
    两种路径都继续流入 digitaltwin + verdict，因此高保真 asbuilt 可无缝接入
    数字样机与可微逆问题。

    ``params["thermal"]`` 缺省时本函数按 :func:`thermal_tier` 自动为**这张网格**钉好
    显式热解的静态调度；调用方已钉好的档位一律尊重。trace 内（工艺量是 tracer）读不到
    数值，无法用报错拦下"走出可积分窗口"的请求，故把工艺**结构性投影**回该档位给出的
    子盒（被夹住的分量梯度为 0）——这是 A0 留给显式解法的那个"步数不足→扩散发散→被
    蒸发封顶伪装成可行结果"静默洞的正面对策。eager 调用仍在越界时明确报错。
    """
    p = thermal_tier(dict(params or {}), geometry, process,
                   material=(params or {}).get("material", material),
                   thermal_solver=thermal_solver)
    _bnds = tier_bounds(p)
    if _bnds and _is_traced(process):
        process = clip_to_bounds(process, bounds=_bnds)
    mat_name = p.get("material", material)
    mp_p = {"material": mat_name, "n_grid": int(p.get("n_grid", 28))}
    thermal_p = {"material": mat_name, **dict(p.get("thermal") or {})}
    micro_p = dict(p.get("micro", {}))
    const_p = {"material": mat_name}
    buildup_p = {"material": mat_name}
    dt_p = {"material": mat_name, "service_stress": service_stress,
            "allowable_displacement": float(p.get("allowable_displacement", 1e-4))}
    verdict_p = {"material": mat_name,
                 "allowable_displacement": float(p.get("allowable_displacement", 1e-4)),
                 "min_cycles": float(p.get("min_cycles", 1e5))}

    meltpool = solve_meltpool_surrogate(geometry=geometry, process=process,
                                        params=mp_p)
    thermal = _run_thermal(thermal_solver, geometry=geometry, process=process,
                           meltpool=meltpool, params=thermal_p)
    if micro_solver == "phasefield":
        # 高保真枝晶相场：含 4 重各向异性与热梯度取向，输出同契约 MicrostructureResult
        micro = solve_phasefield(thermal=thermal, meltpool=meltpool, params=micro_p)
    else:
        # 默认：统计代理微观组织模型（micro.surrogate）
        micro = solve_microstructure(thermal=thermal, meltpool=meltpool, params=micro_p)
    const = solve_constitutive(microstructure=micro, params=const_p)

    if asbuilt_solver == "plastic":
        # 高保真塑性/CPFE 生死单元 FEM —— 与 buildup 同契约 AsBuiltPart。
        # 仅需 geometry + thermal；process/micro 透传以便 2D 回退路径使用。
        plastic_p = {
            "material": mat_name,
            "constitutive": constitutive,
            "max_layers": int(p.get("max_layers", 12)),
            "n_sub_cp": int(p.get("n_sub_cp", 64)),
            "tau_activation": float(p.get("tau_activation", 1e-3)),
            "process": process,
            "microstructure": micro,
        }
        asbuilt = solve_asbuilt_plastic(
            geometry=geometry, thermal=thermal, params=plastic_p)
    else:
        asbuilt = solve_buildup(geometry=geometry, process=process, thermal=thermal,
                                microstructure=micro, params=buildup_p)

    # 装配体（多体 + FEM 耦合）：FEM 子模型(asbuilt) → 关节柔度，MBD → 质量/载荷。
    # 默认 surrogate 零回归；选 "coupled"/"newton" 走高保真（mbd.assembly）。
    mbd_p = {"n_joints": int(p.get("n_joints", 3)),
             "mbd_fidelity": mbd_solver}
    assembly = solve_assembly(geometry=geometry, asbuilt=asbuilt, params=mbd_p)

    struct = solve_digital_twin(asbuilt=asbuilt, constitutive=const, params=dt_p)
    verdict = solve_verdict(structural=struct, asbuilt=asbuilt,
                            constitutive=const, params=verdict_p)

    # 粉末床（P2 粉末尺度）：颗粒尺度缺陷前驱 → PowderBedResult。
    # 默认 None 跳过（零回归）；选 "surrogate"/"dem"/"sph"/"mpm" 走高保真。
    # 仅依赖几何静态 bbox + 工艺可微叶子，tracer-safe，可被 jax.grad 穿透。
    powderbed = None
    if powder_solver is not None and powder_solver != "none":
        powder_p = {"powder_fidelity": powder_solver, "material": mat_name,
                    "d50": float(p.get("d50", 30e-6)),
                    "packing_fraction": float(p.get("packing_fraction", 0.55))}
        powderbed = solve_powderbed(geometry=geometry, process=process,
                                    params=powder_p)

    out = {"meltpool": meltpool, "thermal": thermal, "micro": micro,
           "constitutive": const, "asbuilt": asbuilt, "assembly": assembly,
           "structural": struct, "verdict": verdict}
    if powderbed is not None:
        out["powderbed"] = powderbed
    return out


def predict_process(params_nn: dict, features: jnp.ndarray, *,
                    n_layers: int = 8, modality: str = "SLM",
                    bounds: Mapping[str, tuple[float, float]] | None = None
                    ) -> ProcessPlan:
    """神经网络 -> 物理有界工艺方案。

    ``bounds`` 给子盒时（见 :func:`thermal_tier` 的 ``process_bounds``），网络输出被
    映射进**本网格分辨得出、且钉住的显式调度覆盖得了曝光**的那部分工艺盒。
    """
    z = mlp_apply(params_nn, features)
    return denormalize_process(z, n_layers=n_layers, modality=modality,
                               bounds=bounds)


# ===========================================================================
# 4. 损失函数：模拟后构型 vs 理想构型
# ===========================================================================
def _dimensional_loss(out: dict, geometry: PartGeometry, material: str,
                      *, weights: Mapping[str, float] | None = None) -> jnp.ndarray:
    """由一次物理前向的输出 dict 计算可微尺寸损失（与 ``loss_fn`` 共用）。

    分量（均已量纲归一化，避免某一分量压倒其它）：
      (1) 几何偏差 geom_dev：变形 SDF 与理想 SDF 的法向偏差（高保真
          asbuilt 下即真实弹塑性收缩/翘曲引起的尺寸偏差）；
      (2) 残余应力：体平均 von Mises / σ_y；
      (3) 残余应变：体平均等效应变（sqrt 加护垫避免零应变处梯度爆炸）；
      (4) 缺陷：熔池综合缺陷评分；
      (5) 服役安全裕度过小的额外惩罚。
    """
    w = dict(weights or {})
    w1 = w.get("geom", 1.0)
    w2 = w.get("stress", 1.0)
    w3 = w.get("strain", 1.0)
    w4 = w.get("defect", 2.0)
    w5 = w.get("powder", 1.0)

    mat = get_material(material)
    spacing = float(geometry.spacing)
    # 本函数里的 w_vol 是**体积/面积平均权重**，一律用线性 cut 份额（唯一口径
    # contracts.solid_weight，见 #19）；不再用二值实体掩膜——掩膜在界面上偏置 O(dx)。
    w_vol = solid_weight(geometry.sdf, spacing)
    cell_vol = spacing ** geometry.dim

    # (1) 几何偏差：变形 SDF 与理想 SDF 的偏差（归一化到体素尺寸）
    sdf_def = out["asbuilt"].sdf
    geom_dev = jnp.mean(jnp.abs(sdf_def - geometry.sdf) * w_vol) / spacing

    # (2) 残余应力：体平均 von Mises / 屈服强度
    rvm = out["asbuilt"].von_mises_residual()
    stress_term = jnp.sum(rvm * w_vol) * cell_vol / \
        jnp.maximum(jnp.sum(w_vol) * cell_vol, 1e-30) / mat.sigma_y

    # (3) 残余应变：体平均等效应变（sqrt 加护垫避免零应变处梯度爆炸）
    strain_vec = out["asbuilt"].residual_strain
    strain_mag = jnp.sqrt(jnp.sum(strain_vec ** 2, axis=-1) + 1e-30)
    strain_term = jnp.sum(strain_mag * w_vol) * cell_vol / \
        jnp.maximum(jnp.sum(w_vol) * cell_vol, 1e-30)

    # (4) 缺陷：熔池综合缺陷评分
    defect_term = out["meltpool"].defect_score()

    # (4.5) 粉末尺度缺陷：颗粒尺度球化/未熔合/飞溅综合评分
    # 仅当逆问题显式启用粉末床求解器时计入（默认无该键 → 零回归）。
    powder_term = jnp.asarray(0.0)
    if "powderbed" in out:
        powder_term = out["powderbed"].defect_score()

    # 安全裕度过小的额外惩罚（强度安全系数 < 1 时显著）
    sf = out["verdict"].strength_safety_factor
    safety_penalty = jax.nn.relu(1.0 - sf)

    return (w1 * geom_dev + w2 * stress_term + w3 * strain_term
            + w4 * defect_term + w5 * powder_term + 5.0 * safety_penalty)


def loss_fn(params_nn: dict, geometry: PartGeometry, *,
            thermal_solver: str = "enthalpy",
            material: str = "316L", service_stress: float = 150e6,
            weights: Mapping[str, float] | None = None,
            params: Mapping[str, Any] | None = None,
            asbuilt_solver: str = "buildup",
            micro_solver: str = "surrogate",
            mbd_solver: str = "surrogate",
            powder_solver: str | None = None,
            constitutive: str = "j2") -> jnp.ndarray:
    """可微损失：几何偏差 + 残余应力 + 残余应变 + 缺陷。

    数值已做量纲归一化，避免某一分量压倒其它。``asbuilt_solver`` /
    ``constitutive`` 透传给 :func:`simulate`，使损失可基于高保真塑性
    成形（真实弹塑性收缩/翘曲）计算。

    ``params_nn`` 是 tracer（网络输出→工艺全为 trace 量），而显式热解的
    ``n_steps`` 必须是静态量：故档位在此用**规则式名义工艺**（几何感知
    heuristic）于 eager 钉好，网络则被限制在该档位给出的工艺子盒内取值
    （见 :func:`thermal_tier`）。
    """
    p = thermal_tier(dict(params or {}), geometry,
                   heuristic_plan(geometry, material=material, modality="SLM"),
                   material=material, thermal_solver=thermal_solver)
    feats = geometry_features(geometry)
    plan = predict_process(params_nn, feats,
                           n_layers=int(geometry.layer_count(40e-6)),
                           modality="SLM", bounds=tier_bounds(p))
    out = simulate(geometry, plan, material=material,
                   service_stress=service_stress, params=p,
                   asbuilt_solver=asbuilt_solver, micro_solver=micro_solver,
                   mbd_solver=mbd_solver, powder_solver=powder_solver,
                   constitutive=constitutive, thermal_solver=thermal_solver)
    return _dimensional_loss(out, geometry, material, weights=weights)


# ===========================================================================
# 5. 训练循环（optax / adam）
# ===========================================================================
def train_process_predictor(geometry: PartGeometry, *,
                            material: str = "316L",
                            n_steps: int = 60,
                            learning_rate: float = 0.02,
                            n_hidden: int = 16,
                            seed: int = 0,
                            service_stress: float = 150e6,
                            weights: Mapping[str, float] | None = None,
                            params: Mapping[str, Any] | None = None,
                            verbose: bool = True,
                            thermal_solver: str = "enthalpy",
                            asbuilt_solver: str = "buildup",
                            micro_solver: str = "surrogate",
                            mbd_solver: str = "surrogate",
                            powder_solver: str | None = None,
                            constitutive: str = "j2") -> dict:
    """训练"几何→工艺"网络，使模拟后构型最贴近理想构型。

    返回 ``{"params": θ, "loss_history": [...], "final_plan": ProcessPlan}``。
    设 ``asbuilt_solver="plastic"`` 即把高保真塑性成形接入训练损失。
    """
    feats = geometry_features(geometry)
    n_in = int(feats.shape[0])
    rng = jax.random.PRNGKey(seed)
    theta = mlp_init(rng, n_in, n_hidden, n_out=len(PROCESS_BOUNDS),
                     n_layers=2)

    opt = optax.adam(learning_rate)
    opt_state = opt.init(theta)
    loss_and_grad = jax.value_and_grad(loss_fn)

    loss_history = []
    for step in range(n_steps):
        loss, grads = loss_and_grad(theta, geometry, material=material,
                                    service_stress=service_stress,
                                    weights=weights, params=params,
                                    asbuilt_solver=asbuilt_solver,
                                    micro_solver=micro_solver,
                                    mbd_solver=mbd_solver,
                                    powder_solver=powder_solver,
                                    constitutive=constitutive,
                                    thermal_solver=thermal_solver)
        updates, opt_state = opt.update(grads, opt_state)
        theta = optax.apply_updates(theta, updates)
        loss_history.append(float(loss))
        if verbose and (step % max(1, n_steps // 10) == 0 or step == n_steps - 1):
            print(f"  step {step:3d}  loss = {float(loss):.4e}")

    # 训练时网络是被**档位子盒**（:func:`thermal_tier`，与 ``loss_fn`` 同一套
    # params/名义工艺）约束的：z↔工艺 用子盒映射。出训练后必须用**同一个**子盒
    # 还原 final_plan——否则用设备全盒还原会把 z 误读成另一套坐标，产出的工艺落在
    # 档位覆盖之外（实测 v=1.1e-3 m/s 而本子盒下界 0.5 m/s），下游 ``simulate`` 的
    # 曝光校核随即报错。
    tier_p = thermal_tier(dict(params or {}), geometry,
                          heuristic_plan(geometry, material=material,
                                         modality="SLM"),
                          material=material, thermal_solver=thermal_solver)
    final_plan = predict_process(theta, feats,
                                 n_layers=int(geometry.layer_count(40e-6)),
                                 modality="SLM", bounds=tier_bounds(tier_p))
    return {"params": theta, "loss_history": loss_history, "final_plan": final_plan}


# ===========================================================================
# 6. 可微尺寸优化（高保真 asbuilt 直连 digitaltwin/逆问题）
# ===========================================================================
def optimize_dimensional(geometry: PartGeometry, process_init: ProcessPlan | None = None, *,
                         thermal_solver: str = "enthalpy",
                         material: str = "316L",
                         n_steps: int = 16,
                         learning_rate: float = 0.05,
                         service_stress: float = 150e6,
                         weights: Mapping[str, float] | None = None,
                         params: Mapping[str, Any] | None = None,
                         asbuilt_solver: str = "plastic",
                         micro_solver: str = "surrogate",
                         mbd_solver: str = "surrogate",
                         powder_solver: str | None = None,
                         constitutive: str = "j2",
                         n_layers: int | None = None,
                         constraint_weight: float = 0.0,
                         hard_project: bool = True,
                         verbose: bool = True) -> dict:
    """**可微尺寸优化**：直接对归一化工艺向量 z∈[0,1]^9 做 Adam 梯度下降，
    用高保真 asbuilt（真实弹塑性收缩/翘曲）驱动的尺寸偏差作为目标，同时压低
    残余应力/应变与服役失效风险。零件构型经 digitaltwin + verdict 评估。

    这是"把高保真 asbuilt 接入逆问题做尺寸优化"的直接入口：工艺参数即设计
    变量，最小化的目标 = 模拟后构型与理想构型的尺寸/残余应力偏差。

    物理约束有两种互补手段（可同时启用，互不冲突）：

    * ``hard_project=True``（**默认**）：每步更新后把 ``z`` **硬投影**到物理
      可行域最近点（见 :func:`project_process_feasible`，log 空间凸多面体上的
      Dykstra 投影），从数学上保证最终工艺可行——这是推荐的生产设置；
    * ``constraint_weight`` > 0：在目标中叠加**软罚项**（能量密度窗口 / 道间搭接
      / 层间结合 / 匙孔抑制，见 :func:`process_constraint_penalty`）作为次级正则。

    返回 ``{"z": 优化后向量, "final_plan": ProcessPlan, "loss_history": [...],
    "constraint_penalty_history": [...]}``。
    """
    if process_init is None:
        # 几何感知启发式工艺作初值（物理合理，避免落入未熔合/匙孔病态区）
        process_init = heuristic_plan(geometry, material=material, modality="SLM")
    # 显式热解的静态调度必须在 trace 之外钉好；工艺搜索子盒与之自洽（D0）
    params = thermal_tier(dict(params or {}), geometry, process_init,
                        material=material, thermal_solver=thermal_solver)
    bnds = tier_bounds(params)
    nl = n_layers or int(geometry.layer_count(40e-6))
    z = normalize_process(process_init, bounds=bnds)
    if hard_project:
        z = project_process_feasible(z, material=material, n_layers=nl,
                                     modality="SLM", params=params)
    w = dict(weights or {"geom": 5.0, "stress": 1.0, "strain": 0.5,
                         "defect": 1.0, "powder": 1.0})

    opt = optax.adam(learning_rate)
    opt_state = opt.init(z)
    loss_history: list[float] = []
    constraint_penalty_history: list[float] = []

    def loss_of_z(zz):
        plan = denormalize_process(zz, n_layers=nl, modality="SLM", bounds=bnds)
        out = simulate(geometry, plan, material=material,
                       service_stress=service_stress, params=params,
                       asbuilt_solver=asbuilt_solver, micro_solver=micro_solver,
                       mbd_solver=mbd_solver, powder_solver=powder_solver,
                       constitutive=constitutive, thermal_solver=thermal_solver)
        loss = _dimensional_loss(out, geometry, material, weights=w)
        if constraint_weight > 0.0:
            pen, _ = process_constraint_penalty(
                zz, material=material, n_layers=nl, modality="SLM", params=params)
            loss = loss + constraint_weight * pen
        return loss

    # 注意：高保真 asbuilt 用 numpy 由几何构建 FEM 网格（geometry 为闭包常量），
    # 故不在此 jit —— jit 会把几何提升为 tracer 触发 TracerArrayConversionError。
    # 用 plain value_and_grad：几何保持具体，内层 FEM 的 jit 按形状缓存一次。
    loss_and_grad = jax.value_and_grad(loss_of_z)

    for step in range(n_steps):
        loss, grads = loss_and_grad(z)
        updates, opt_state = opt.update(grads, opt_state)
        z = optax.apply_updates(z, updates)
        z = jnp.clip(z, 0.0, 1.0)          # 保持归一化可行域
        if hard_project:
            z = project_process_feasible(z, material=material, n_layers=nl,
                                         modality="SLM", params=params)
        pen = 0.0
        if constraint_weight > 0.0:
            pen, _ = process_constraint_penalty(
                z, material=material, n_layers=nl, modality="SLM", params=params)
        loss_history.append(float(loss))
        constraint_penalty_history.append(float(pen))
        if verbose and (step % max(1, n_steps // 10) == 0 or step == n_steps - 1):
            print(f"  dim-opt step {step:3d}  loss = {float(loss):.4e}  "
                  f"constr = {float(pen):.3e}")

    final_plan = denormalize_process(z, n_layers=nl, modality="SLM", bounds=bnds)
    return {"z": z_to_device_box(z, n_layers=nl, bounds=bnds),
            "final_plan": final_plan, "loss_history": loss_history,
            "constraint_penalty_history": constraint_penalty_history}


# ===========================================================================
# 7. 可微形状/几何优化（反变形 / 预补偿）—— 把尺寸优化的设计变量
#    从"工艺参数"升级到"几何 SDF 本身"
# ===========================================================================
def optimize_shape(target: PartGeometry, *,
                   thermal_solver: str = "enthalpy",
                   material: str = "316L",
                   process: ProcessPlan | None = None,
                   n_steps: int = 20,
                   learning_rate: float = 0.01,
                   weights: Mapping[str, float] | None = None,
                   params: Mapping[str, Any] | None = None,
                   asbuilt_solver: str = "plastic",
                   constitutive: str = "j2",
                   smoothness: float = 0.02,
                   init_delta: jnp.ndarray | None = None,
                   verbose: bool = True) -> dict:
    """**可微形状/几何优化（反变形 / 预补偿）**：给定目标 CAD 构型 ``target``，
    求一个**名义（预变形）几何** ``sdf_nominal = target.sdf + delta``，使其经
    高保真增材成形后的实际构型（as-built）尽量贴合 ``target``。

    这是把"尺寸优化"的设计变量从工艺参数升级到 **几何/形状本身**：

    * 设计变量是名义 SDF 的逐体素修正场 ``delta``；为数值稳定，在**归一化体素
      单位** ``u = delta / spacing`` 上做 Adam（梯度量级被归一，避免米制单位下
      梯度过大/过小导致的发散）；
    * 目标 = as-built 与 target 的尺寸偏差（主导）+ 残余应力 + 几何光滑正则，
      全部对 ``delta`` 可微。关键是 ``AsBuiltPart.sdf = nominal_sdf + disp·n̂``
      中 ``nominal_sdf`` 项直接携带梯度，叠加 ``occupancy_override`` 通道后，
      力学位移/残余应力对 SDF 的敏感性也能回传；
    * 物理假设（增材反变形标准做法）：成形热历史由固定工艺决定并**一次预计算**
      （不随名义几何微调重算），名义几何只改变力学响应的位移场；优化即在固定
      热载荷下搜索使"变形后构型 = 目标构型"的预补偿 SDF。

    返回 ``{"delta": 补偿场[m], "nominal_geometry": PartGeometry,
    "asbuilt": AsBuiltPart, "loss_history": [...], "u_history": [...]}``。
    """
    if process is None:
        process = heuristic_plan(target, material=material, modality="SLM")

    # 一次预计算热历史（固定工艺，不随名义几何重算）
    p = thermal_tier(dict(params or {}), target, process,
                   material=material, thermal_solver=thermal_solver)
    mp_p = {"material": material, "n_grid": int(p.get("n_grid", 8))}
    thermal_p = {"material": material, **dict(p.get("thermal") or {})}
    meltpool = solve_meltpool_surrogate(geometry=target, process=process,
                                        params=mp_p)
    thermal = _run_thermal(thermal_solver, geometry=target, process=process,
                           meltpool=meltpool, params=thermal_p)

    spacing = float(target.spacing)
    shape = target.shape
    w = dict(weights or {"geom": 5.0, "stress": 1.0})
    eps = spacing  # 软占位过渡带 ~1 体素

    # 归一化变量 u = delta / spacing（体素单位，梯度量级归一）
    if init_delta is None:
        u = jnp.zeros(shape, dtype=jnp.float64)
    else:
        u = jnp.asarray(init_delta, dtype=jnp.float64) / spacing

    soft_occ_t = target.soft_occupancy(eps=eps)        # 目标实体区（具体数组）
    soft_sum = jnp.maximum(jnp.sum(soft_occ_t), 1e-12)
    mat = get_material(material)

    opt = optax.adam(learning_rate)
    opt_state = opt.init(u)
    loss_history: list[float] = []
    u_history: list[jnp.ndarray] = []
    geom_history: list[float] = []
    stress_history: list[float] = []

    def build_nominal(uu):
        nominal_sdf = target.sdf + uu * spacing
        return PartGeometry(sdf=nominal_sdf, origin=target.origin,
                            spacing=spacing, dim=target.dim, name="nominal")

    def _asbuilt_of(nominal):
        if asbuilt_solver == "plastic":
            plastic_p = {
                "material": material, "constitutive": constitutive,
                "max_layers": int(p.get("max_layers", 12)),
                "n_sub_cp": int(p.get("n_sub_cp", 64)),
                "tau_activation": float(p.get("tau_activation", 1e-3)),
                "process": process, "microstructure": None,
                # 可微软占位：让激活掩码/末态场对名义 SDF 可微
                "occupancy_override": nominal.soft_occupancy(eps=eps),
            }
            return solve_asbuilt_plastic(geometry=nominal, thermal=thermal,
                                         params=plastic_p)
        return solve_buildup(geometry=nominal, process=process,
                             thermal=thermal, microstructure=None,
                             params={"material": material})

    def shape_loss(uu):
        nominal = build_nominal(uu)
        asbuilt = _asbuilt_of(nominal)
        # (1) 尺寸偏差：as-built SDF 与 target SDF 在目标实体区偏差
        dev = jnp.abs(asbuilt.sdf - target.sdf) * soft_occ_t
        geom_dev = jnp.sum(dev) / soft_sum / spacing
        # (2) 残余应力：目标实体区体平均 von Mises / σ_y
        rvm = asbuilt.von_mises_residual()
        stress = jnp.sum(rvm * soft_occ_t) / soft_sum / mat.sigma_y
        # (3) 光滑正则：归一化修正场的总变差（保持几何可制造）。
        # 注意：用 sqrt(Σg² + ε) 而非 jnp.linalg.norm——后者在平场（gu=0）处
        # 逆向导数为 0/0 → NaN，会污染整条梯度链；加 ε 使其处处良定义。
        gu = jnp.stack(jnp.gradient(uu), axis=-1)
        tv = jnp.mean(jnp.sqrt(jnp.sum(gu ** 2, axis=-1) + 1e-12))
        total = (w.get("geom", 5.0) * geom_dev
                 + w.get("stress", 1.0) * stress
                 + smoothness * tv)
        return total, (geom_dev, stress, tv)

    loss_and_grad = jax.value_and_grad(shape_loss, has_aux=True)

    for step in range(n_steps):
        (loss, aux), grads = loss_and_grad(u)
        updates, opt_state = opt.update(grads, opt_state)
        u = optax.apply_updates(u, updates)
        loss_history.append(float(loss))
        u_history.append(u)
        geom_history.append(float(aux[0]))
        stress_history.append(float(aux[1]))
        if verbose and (step % max(1, n_steps // 10) == 0 or step == n_steps - 1):
            print(f"  shape-opt step {step:3d}  loss={float(loss):.4e}  "
                  f"geom_dev={float(aux[0]):.3e}  stress={float(aux[1]):.3e}  "
                  f"tv={float(aux[2]):.3e}")

    nominal = build_nominal(u)
    final_ab = _asbuilt_of(nominal)
    return {"delta": u * spacing, "nominal_geometry": nominal,
            "asbuilt": final_ab, "loss_history": loss_history,
            "u_history": u_history, "geom_history": geom_history,
            "stress_history": stress_history}


# ===========================================================================
# 7.5 工艺杠杆物理约束（可微罚项）+ 共享前向/目标辅助
# ===========================================================================
def process_constraint_penalty(z: jnp.ndarray, *, material: str = "316L",
                               n_layers: int = 1, modality: str = "SLM",
                               params: Mapping[str, Any] | None = None,
                               ved_window: tuple[float, float] = (3.0e10, 1.5e11),
                               overlap_factor: float = 0.9,
                               layer_factor: float = 0.6,
                               keyhole_limit: float = 30.0) -> tuple[jnp.ndarray, dict]:
    """**工艺杠杆物理约束罚项**（可微，对 z∈[0,1]⁹ 求梯度处处良定义）。

    把归一化工艺向量 ``z`` 经 :func:`denormalize_process` 还原为物理工艺方案，
    对下列**物理耦合约束**做软惩罚（全部用 ``relu``/比值，处处可微）：

    1. **体能量密度窗口** ``E_v = P/(v·h·t)`` 落在 ``[E_lo, E_hi]`` 内
       —— 偏低→未熔合(lack-of-fusion)，偏高→匙孔(keyhole)/球化；
    2. **道间搭接** ``h ≤ overlap_factor · w``，``w`` 为熔池宽度
       ``w = 2r·sqrt(1 + ΔH/h_s/8)`` —— 防止道间孔隙；
    3. **层间结合** ``t ≤ layer_factor·(2r)`` —— 防止层间未熔合；
    4. **匙孔抑制** 归一化焓 ``ΔH/h_s ≤ keyhole_limit``。

    返回 ``(penalty, details)``：``penalty`` 是标量（≥0，越接近 0 越可行），
    ``details`` 为各分项（tracer 数组，仅在非求导上下文用 ``float()`` 取数）。

    与单纯把 ``z`` clip 到 [0,1] 相比，本罚项刻画的是**参数间的物理耦合**
    （能量密度、搭接、层厚与光斑），把工艺推入"致密且稳定"的真实可行域，
    而不仅是设备标量边界。
    """
    plan = denormalize_process(z, n_layers=n_layers, modality=modality,
                               bounds=tier_bounds(params))
    mat = get_material(material)

    # (1) 体能量密度窗口
    ev = jnp.mean(jnp.atleast_1d(plan.volumetric_energy_density()))
    elo, ehi = ved_window
    pen_ved = jax.nn.relu(elo / ev - 1.0) + jax.nn.relu(ev / ehi - 1.0)

    # (2) 道间搭接：h ≤ overlap_factor · w
    dH = jnp.mean(jnp.atleast_1d(plan.normalized_enthalpy(
        rho=mat.rho_solid, cp=mat.cp_solid, T_melt=mat.T_melt,
        diffusivity=mat.diffusivity())))
    w = 2.0 * plan.beam_radius * jnp.sqrt(1.0 + jnp.maximum(dH, 0.0) / 8.0)
    h = jnp.mean(jnp.atleast_1d(plan.hatch_spacing))
    pen_overlap = jax.nn.relu(h / (overlap_factor * jnp.maximum(w, 1e-12)) - 1.0)

    # (3) 层间结合：t ≤ layer_factor·(2r)
    t = jnp.mean(jnp.atleast_1d(plan.layer_thickness))
    pen_layer = jax.nn.relu(
        t / (layer_factor * 2.0 * jnp.maximum(plan.beam_radius, 1e-12)) - 1.0)

    # (4) 匙孔抑制：ΔH/h_s ≤ keyhole_limit
    pen_key = jax.nn.relu(dH / keyhole_limit - 1.0)

    total = pen_ved + pen_overlap + pen_layer + pen_key
    # details 保持 tracer（不在此 float，避免求导时 Tracer→float 转换报错）
    details = {"ved": ev, "ved_penalty": pen_ved,
               "normalized_enthalpy": dH, "overlap_penalty": pen_overlap,
               "layer_penalty": pen_layer, "keyhole_penalty": pen_key}
    return total, details


def process_feasibility(z: jnp.ndarray, *, material: str = "316L",
                       n_layers: int = 1, modality: str = "SLM",
                       params: Mapping[str, Any] | None = None,
                       **kw) -> jnp.ndarray:
    """工艺可行性评分 ∈ (0,1]（1 = 完全满足物理约束）。

    在**非求导**上下文调用（如报告/前端展示/最优点筛选），对 ``z`` 给出
    一个直观的可行度：``1/(1 + penalty)``。
    """
    pen, _ = process_constraint_penalty(
        z, material=material, n_layers=n_layers, modality=modality,
        params=params, **kw)
    return 1.0 / (1.0 + pen)


def project_process_feasible(z: jnp.ndarray, *, material: str = "316L",
                             n_layers: int = 1, modality: str = "SLM",
                             params: Mapping[str, Any] | None = None,
                             ved_window: tuple[float, float] = (3.0e10, 1.5e11),
                             overlap_factor: float = 0.9,
                             layer_factor: float = 0.6,
                             keyhole_limit: float = 30.0,
                             n_iter: int = 60) -> jnp.ndarray:
    """**工艺杠杆硬投影**：把归一化工艺向量 ``z`` 投影到物理可行域的**最近点**。

    与 :func:`process_constraint_penalty` 的"软罚项"不同，这里是**硬约束**——
    优化器每步更新后强制把 ``z`` 拉回可行域，保证最终工艺物理可行，而非仅压低罚项。

    数学上：在 log 空间对耦合工艺变量
    ``x = (ln P, ln v, ln t, ln h, ln r, ln A)`` 构造凸多面体可行域
    （全部为线性半空间，故交集凸）：

    * VED 窗口：``ln E_lo ≤ x_P − x_v − x_t − x_h ≤ ln E_hi``；
    * 匙孔抑制：``x_A + x_P − 0.5·x_v − 1.5·x_r ≤ ln(keyhole_limit·ρ·cp·Tm·√(π α))``；
    * 层间结合：``x_t − x_r ≤ ln(layer_factor·2)``；
    * 道间搭接（保守充分）：``x_h − x_r ≤ ln(overlap_factor·2)``
      （因熔池宽 ``w ≥ 2r``，故 ``h ≤ 1.8r`` ⇒ ``h ≤ 0.9·w`` 必成立，保证真可行）；
    * 各变量设备边界 box。

    用 **Dykstra 算法**求 Euclidean(log 空间) 最近可行点：确定性、可微、对任意 ``z``
    收敛。其余 3 个线性变量（preheat/powder_feed/dwell）不参与耦合约束，原样保留。
    几何为闭包常量，故不在此 jit（与优化器外层一致，避免 TracerArrayConversionError）。

    返回可行 ``z``（长度 9，∈[0,1]⁹）。
    """
    mat = get_material(material)
    elo, ehi = ved_window
    bnds = dict(tier_bounds(params) or PROCESS_BOUNDS)
    keys = ("laser_power", "scan_speed", "layer_thickness",
            "hatch_spacing", "beam_radius", "absorption")
    los = jnp.array([bnds[k][0] for k in keys], dtype=jnp.float64)
    his = jnp.array([bnds[k][1] for k in keys], dtype=jnp.float64)
    ln_lo = jnp.log(los)
    ln_hi = jnp.log(his)

    plan = denormalize_process(z, n_layers=n_layers, modality=modality,
                               bounds=tier_bounds(params))
    phys = jnp.array(
        [jnp.mean(jnp.atleast_1d(getattr(plan, k))) for k in keys],
        dtype=jnp.float64)
    x = jnp.log(jnp.maximum(phys, 1e-12))          # log 空间坐标 [P,v,t,h,r,A]

    rho, cp, Tm = mat.rho_solid, mat.cp_solid, mat.T_melt
    alpha = mat.diffusivity()
    b_key = jnp.log(keyhole_limit * rho * cp * Tm * jnp.sqrt(jnp.pi * alpha))
    ln2 = jnp.log(2.0)

    # a·x ≤ b  （x 顺序：P,v,t,h,r,A = 0..5）
    A = jnp.stack([
        jnp.array([-1.,  1.,  1.,  1.,  0., 0.], dtype=jnp.float64),   # VED 下界
        jnp.array([ 1., -1., -1., -1.,  0., 0.], dtype=jnp.float64),   # VED 上界
        jnp.array([ 1., -0.5, 0., 0., -1.5, 1.], dtype=jnp.float64),   # 匙孔
        jnp.array([ 0.,  0.,  1.,  0., -1., 0.], dtype=jnp.float64),   # 层间结合
        jnp.array([ 0.,  0.,  0.,  1., -1., 0.], dtype=jnp.float64),   # 搭接(保守)
    ])
    b = jnp.array([
        -jnp.log(elo),
         jnp.log(ehi),
         b_key,
         jnp.log(layer_factor) + ln2,
         jnp.log(overlap_factor) + ln2,
    ], dtype=jnp.float64)

    def proj_box(y):
        return jnp.clip(y, ln_lo, ln_hi)

    def proj_half(y, a, bb):
        d = jnp.dot(a, y) - bb
        return y - (jnp.maximum(d, 0.0) / (jnp.dot(a, a) + 1e-30)) * a

    # Dykstra：box + 5 个半空间，共 6 个凸集
    xk = x
    ps = [jnp.zeros(6) for _ in range(6)]
    for _ in range(int(n_iter)):
        y = xk
        zy = proj_box(y + ps[0]); ps[0] = y + ps[0] - zy; y = zy
        for i in range(5):
            zy = proj_half(y + ps[i + 1], A[i], b[i])
            ps[i + 1] = y + ps[i + 1] - zy
            y = zy
        xk = y

    xf = xk
    new_phys = jnp.exp(xf)
    z_new = []
    for i, k in enumerate(keys):
        lo, hi = bnds[k]
        if k == "absorption":
            zi = (new_phys[i] - lo) / (hi - lo)
        else:
            zi = (xf[i] - jnp.log(lo)) / (jnp.log(hi) - jnp.log(lo))
        z_new.append(zi)
    z_new = jnp.stack(z_new)
    full = jnp.concatenate([z_new, jnp.asarray(z, dtype=jnp.float64)[6:9]])
    return jnp.clip(full, 0.0, 1.0)


def _joint_forward(target, u, z, *, material, params, asbuilt_solver,
                   micro_solver="surrogate", mbd_solver="surrogate",
                   thermal_solver="enthalpy",
                   powder_solver=None,
                   constitutive, p, spacing, eps,
                   n_layers, modality="SLM",
                   service_stress=150e6) -> dict:
    """联合优化的**共享物理前向**（与 ``optimize_geometry_process`` 同源，
    抽出为模块级函数以便 Pareto 扫描复用，行为完全一致）。

    热历史/熔池由 **target 几何 + 当前工艺 z** 决定（与名义几何 ``u`` 解耦，
    见 Pattern E）；名义几何 ``nominal = target.sdf + u·spacing`` 经高保真
    塑性 FEM 成形（``occupancy_override`` 让激活掩码/末态场对 SDF 可微）。
    """
    plan = denormalize_process(z, n_layers=n_layers, modality=modality,
                               bounds=tier_bounds(p))
    nominal = PartGeometry(sdf=target.sdf + u * spacing, origin=target.origin,
                           spacing=spacing, dim=target.dim, name="nominal")
    mp_p = {"material": material, "n_grid": int(p.get("n_grid", 8))}
    meltpool = solve_meltpool_surrogate(geometry=target, process=plan, params=mp_p)
    thermal = _run_thermal(thermal_solver, geometry=target, process=plan,
                           meltpool=meltpool,
                           params={"material": material,
                                   **dict(p.get("thermal") or {})})
    # 微观组织：默认统计代理；选 "phasefield" 走高保真枝晶相场（micro.phasefield）
    micro_p = dict(p.get("micro", {}))
    if micro_solver == "phasefield":
        micro = solve_phasefield(thermal=thermal, meltpool=meltpool, params=micro_p)
    else:
        micro = solve_microstructure(thermal=thermal, meltpool=meltpool, params=micro_p)
    const = solve_constitutive(microstructure=micro, params={"material": material})
    if asbuilt_solver == "plastic":
        plastic_p = {
            "material": material, "constitutive": constitutive,
            "max_layers": int(p.get("max_layers", 12)),
            "n_sub_cp": int(p.get("n_sub_cp", 64)),
            "tau_activation": float(p.get("tau_activation", 1e-3)),
            "process": plan, "microstructure": micro,
            "occupancy_override": nominal.soft_occupancy(eps=eps),
        }
        asbuilt = solve_asbuilt_plastic(geometry=nominal, thermal=thermal,
                                        params=plastic_p)
    else:
        asbuilt = solve_buildup(geometry=nominal, process=plan, thermal=thermal,
                               microstructure=micro, params={"material": material})
    dt_p = {"material": material, "service_stress": service_stress,
            "allowable_displacement": float(p.get("allowable_displacement", 1e-4))}
    # 装配体（多体 + FEM 耦合）：FEM 子模型(asbuilt) → 关节柔度，MBD → 质量/载荷。
    mbd_p = {"n_joints": int(p.get("n_joints", 3)),
             "mbd_fidelity": mbd_solver}
    assembly = solve_assembly(geometry=nominal, asbuilt=asbuilt, params=mbd_p)

    struct = solve_digital_twin(asbuilt=asbuilt, constitutive=const, params=dt_p)
    verdict = solve_verdict(structural=struct, asbuilt=asbuilt, constitutive=const,
                           params={"material": material,
                                   "allowable_displacement":
                                       float(p.get("allowable_displacement", 1e-4)),
                                   "min_cycles": float(p.get("min_cycles", 1e5))})

    # 粉末床（P2 粉末尺度）：仅依赖 target 静态 bbox + 工艺 z，tracer-safe。
    powderbed = None
    if powder_solver is not None and powder_solver != "none":
        powder_p = {"powder_fidelity": powder_solver, "material": material,
                    "d50": float(p.get("d50", 30e-6)),
                    "packing_fraction": float(p.get("packing_fraction", 0.55))}
        powderbed = solve_powderbed(geometry=target, process=plan, params=powder_p)

    out = {"meltpool": meltpool, "thermal": thermal, "micro": micro,
           "constitutive": const, "asbuilt": asbuilt, "assembly": assembly,
           "structural": struct, "verdict": verdict}
    if powderbed is not None:
        out["powderbed"] = powderbed
    return out


def _geom_stress_tv(out, target, w_vol, cell_vol, mat, spacing, eps, u):
    """从一次物理前向输出抽取 (geom_dev, residual_stress, TV)，供联合优化/
    Pareto 扫描共用。三者均已量纲归一化到 O(1)，且梯度对 ``u``、``z`` 连通。

    ``w_vol`` 是体积/面积平均权重（线性 cut 份额，见 :func:`solid_weight`）。
    """
    sdf_def = out["asbuilt"].sdf
    geom_dev = jnp.mean(jnp.abs(sdf_def - target.sdf) * w_vol) / spacing
    rvm = out["asbuilt"].von_mises_residual()
    stress = jnp.sum(rvm * w_vol) * cell_vol / \
        jnp.maximum(jnp.sum(w_vol) * cell_vol, 1e-30) / mat.sigma_y
    # 光滑正则：sqrt(Σg² + ε)，避免平场 0/0 梯度 NaN（Pattern D）
    gu = jnp.stack(jnp.gradient(u), axis=-1)
    tv = jnp.mean(jnp.sqrt(jnp.sum(gu ** 2, axis=-1) + 1e-12))
    return geom_dev, stress, tv


# ===========================================================================
# 8. 联合几何+工艺协同优化（反变形预补偿 + 工艺窗口同时调）
# ===========================================================================
def optimize_geometry_process(target: PartGeometry, *,
                              thermal_solver: str = "enthalpy",
                              material: str = "316L",
                              process_init: ProcessPlan | None = None,
                              init_delta: jnp.ndarray | None = None,
                              n_steps: int = 24,
                              learning_rate: float = 0.01,
                              lr_shape: float | None = None,
                              lr_process: float | None = None,
                              service_stress: float = 150e6,
                              weights: Mapping[str, float] | None = None,
                              params: Mapping[str, Any] | None = None,
                              asbuilt_solver: str = "plastic",
                              micro_solver: str = "surrogate",
                              mbd_solver: str = "surrogate",
                              powder_solver: str | None = None,
                              constitutive: str = "j2",
                              smoothness: float = 0.02,
                              constraint_weight: float = 0.0,
                              hard_project: bool = True,
                              n_layers: int | None = None,
                              verbose: bool = True) -> dict:
    """**联合几何+工艺协同优化**：把 ``optimize_shape``（反变形预补偿，设计变量
    为名义 SDF 修正场 ``delta``）与 ``optimize_dimensional``（工艺窗口调参，设计
    变量为归一化工艺向量 ``z``）合并为**一个**可微目标，用 Adam 同步更新两个
    设计变量，使"预补偿几何 + 工艺窗口"整体最优。

    设计变量是一个 pytree ``{"u": delta/spacing, "z": z∈[0,1]⁹}``：

    * ``u``（形状）：名义 SDF 修正场，归一化到体素单位做 Adam；
    * ``z``（工艺）：归一化激光功率/扫描速度/… 向量，Adam 下自动满足设备边界
      （每步 clip 到 [0,1]）。

    物理前向（每步）：

    * 工艺 ``z → ProcessPlan``，熔池/热历史由 **target 几何 + 当前工艺** 决定
      （增材反变形标准解耦：热场由工艺主导，不对名义几何 ``u`` 重算热场——
      既避免几何为 tracer 时 numpy 构建网格炸裂，也符合"固定热载荷下搜索
      预补偿 SDF"的物理设定）；
    * 名义几何 ``nominal = target.sdf + u·spacing`` 经高保真塑性 FEM 成形
      （``occupancy_override`` 让激活掩码/末态场对 SDF 可微）；
    * 残余应力/应变、熔池缺陷、服役安全由 digitaltwin + verdict 评估。

    目标 = ``_dimensional_loss``（几何偏差 + 残余应力 + 残余应变 + 缺陷 + 安全罚）
    + 光滑正则（TV，对 ``u``）。两条杠杆（补偿形状 + 工艺窗口）**联合**最小化
    残余尺寸偏差与失效风险。

    返回 dict：``delta / nominal_geometry / final_plan / asbuilt / z /
    loss_history / u_history / z_history / geom_history / stress_history``。
    """
    if process_init is None:
        process_init = heuristic_plan(target, material=material, modality="SLM")
    # 档位（静态调度 + 自洽工艺子盒）必须在 trace 外钉好（D0）
    params = thermal_tier(dict(params or {}), target, process_init,
                        material=material, thermal_solver=thermal_solver)
    z0 = normalize_process(process_init, bounds=tier_bounds(params))
    nl = n_layers or int(target.layer_count(40e-6))
    if hard_project:
        z0 = project_process_feasible(z0, material=material, n_layers=nl,
                                      modality="SLM", params=params)
    w = dict(weights or {"geom": 5.0, "stress": 1.0, "strain": 0.5,
                         "defect": 1.0, "powder": 1.0})
    p = dict(params or {})
    spacing = float(target.spacing)
    eps = spacing  # 软占位过渡带 ~1 体素
    mat = get_material(material)

    if init_delta is None:
        u0 = jnp.zeros(target.shape, dtype=jnp.float64)
    else:
        u0 = jnp.asarray(init_delta, dtype=jnp.float64) / spacing

    params_joint = {"u": u0, "z": z0}
    labels = {"u": "shape", "z": "process"}
    lr_u = lr_shape if lr_shape is not None else learning_rate
    lr_z = lr_process if lr_process is not None else learning_rate
    opt = optax.multi_transform(
        {"shape": optax.adam(lr_u), "process": optax.adam(lr_z)}, labels)
    opt_state = opt.init(params_joint)

    loss_history: list[float] = []
    u_history: list[jnp.ndarray] = []
    z_history: list[jnp.ndarray] = []
    geom_history: list[float] = []
    stress_history: list[float] = []
    constraint_penalty_history: list[float] = []

    def build_nominal(uu):
        nominal_sdf = target.sdf + uu * spacing
        return PartGeometry(sdf=nominal_sdf, origin=target.origin,
                            spacing=spacing, dim=target.dim, name="nominal")

    def _forward(uu, zz):
        return _joint_forward(target, uu, zz, material=material, params=params,
                              asbuilt_solver=asbuilt_solver, micro_solver=micro_solver,
                              mbd_solver=mbd_solver, powder_solver=powder_solver,
                              constitutive=constitutive,
                              p=p, spacing=spacing, eps=eps, n_layers=nl,
                              modality="SLM", service_stress=service_stress,
                              thermal_solver=thermal_solver)

    w_vol = solid_weight(target.sdf, spacing)   # 体积平均权重 = 线性 cut 份额（#19）
    cell_vol = spacing ** target.dim

    def joint_loss(pj):
        out = _forward(pj["u"], pj["z"])
        base = _dimensional_loss(out, target, material, weights=w)
        geom_dev, stress, tv = _geom_stress_tv(
            out, target, w_vol, cell_vol, mat, spacing, eps, pj["u"])
        # 工艺杠杆物理约束罚项（energy-density / 搭接 / 层间 / 匙孔）
        pen, _ = process_constraint_penalty(
            pj["z"], material=material, n_layers=nl, modality="SLM", params=params)
        total = base + smoothness * tv + constraint_weight * pen
        return total, (geom_dev, stress, tv, pen)

    loss_and_grad = jax.value_and_grad(joint_loss, has_aux=True)

    for step in range(n_steps):
        (loss, aux), grads = loss_and_grad(params_joint)
        updates, opt_state = opt.update(grads, opt_state)
        params_joint = optax.apply_updates(params_joint, updates)
        params_joint["z"] = jnp.clip(params_joint["z"], 0.0, 1.0)  # 设备边界
        if hard_project:
            params_joint["z"] = project_process_feasible(
                params_joint["z"], material=material, n_layers=nl,
                modality="SLM", params=params)
        loss_history.append(float(loss))
        u_history.append(params_joint["u"])
        z_history.append(params_joint["z"])
        geom_history.append(float(aux[0]))
        stress_history.append(float(aux[1]))
        constraint_penalty_history.append(float(aux[3]))
        if verbose and (step % max(1, n_steps // 10) == 0 or step == n_steps - 1):
            print(f"  joint-opt step {step:3d}  loss={float(loss):.4e}  "
                  f"geom_dev={float(aux[0]):.3e}  stress={float(aux[1]):.3e}  "
                  f"tv={float(aux[2]):.3e}  constr={float(aux[3]):.3e}")

    nominal = build_nominal(params_joint["u"])
    _b = tier_bounds(params)
    final_plan = denormalize_process(params_joint["z"], n_layers=nl,
                                     modality="SLM", bounds=_b)
    final_ab = _forward(params_joint["u"], params_joint["z"])["asbuilt"]
    return {"delta": params_joint["u"] * spacing, "nominal_geometry": nominal,
            "final_plan": final_plan, "asbuilt": final_ab,
            "z": z_to_device_box(params_joint["z"], n_layers=nl, bounds=_b),
            "loss_history": loss_history, "u_history": u_history,
            "z_history": [z_to_device_box(zz, n_layers=nl, bounds=_b)
                          for zz in z_history], "geom_history": geom_history,
            "stress_history": stress_history,
            "constraint_penalty_history": constraint_penalty_history}


# ===========================================================================
# 9. 多目标 Pareto 协同优化（尺寸偏差 ↔ 残余应力 权衡）
# ===========================================================================
def _run_lambda_point(target, lam, u0, z0, *, material, params, asbuilt_solver,
                      micro_solver, mbd_solver, powder_solver, constitutive,
                      thermal_solver="enthalpy",
                      p, spacing, eps, nl, modality, smoothness, constraint_weight,
                      n_steps, learning_rate, lr_shape, lr_process, hard_project,
                      w_vol, cell_vol, mat, verbose):
    """单点 λ-标量化联合优化（供 legacy 'lambda' 路径与 NSGA-II 播种复用）。

    对给定 ``λ`` 跑 ``n_steps`` 步 Adam（形状 ``u`` + 工艺 ``z`` 同步更新），
    返回最终 ``{"u", "z", "aux", "pen_hist"}``。与旧 ``pareto_optimize_...`` 的
    λ 循环位级一致。
    """
    params_joint = {"u": u0, "z": z0}
    labels = {"u": "shape", "z": "process"}
    lr_u = lr_shape if lr_shape is not None else learning_rate
    lr_z = lr_process if lr_process is not None else learning_rate
    opt = optax.multi_transform(
        {"shape": optax.adam(lr_u), "process": optax.adam(lr_z)}, labels)
    opt_state = opt.init(params_joint)

    def loss_fn(pj):
        out = _joint_forward(
            target, pj["u"], pj["z"], material=material, params=params,
            asbuilt_solver=asbuilt_solver, micro_solver=micro_solver,
            mbd_solver=mbd_solver, powder_solver=powder_solver,
            constitutive=constitutive,
            p=p, spacing=spacing, eps=eps, n_layers=nl,
            modality=modality, service_stress=150e6,
            thermal_solver=thermal_solver)
        gd, st, tv = _geom_stress_tv(
            out, target, w_vol, cell_vol, mat, spacing, eps, pj["u"])
        pen, _ = process_constraint_penalty(
            pj["z"], material=material, n_layers=nl, modality=modality,
            params=params)
        total = (lam * gd + (1.0 - lam) * st
                 + smoothness * tv + constraint_weight * pen)
        return total, (gd, st, tv, pen)
    loss_and_grad = jax.value_and_grad(loss_fn, has_aux=True)

    pen_hist: list[float] = []
    last_pj = params_joint
    last_aux = None
    for step in range(int(n_steps)):
        (loss, aux), grads = loss_and_grad(last_pj)
        updates, opt_state = opt.update(grads, opt_state)
        pj = optax.apply_updates(last_pj, updates)
        pj["z"] = jnp.clip(pj["z"], 0.0, 1.0)   # 设备边界
        if hard_project:
            pj["z"] = project_process_feasible(
                pj["z"], material=material, n_layers=nl,
                modality=modality, params=params)
        pen_hist.append(float(aux[3]))
        last_pj, last_aux = pj, aux
    return {"u": last_pj["u"], "z": last_pj["z"],
            "aux": last_aux, "pen_hist": pen_hist}


def _nsga_seed_population(target, *, material, process_init, lambdas,
                         n_steps, learning_rate, lr_shape, lr_process,
                         smoothness, constraint_weight, params, asbuilt_solver,
                         micro_solver, mbd_solver, powder_solver, constitutive,
                         thermal_solver="enthalpy",
                         hard_project, n_layers, modality, u_clip, pop_size,
                         seed):
    """用短 λ-扫描播种 NSGA-II 初始种群（混合初始化；仅初始化，排序仍为 NSGA-II）。

    对若干 ``λ`` 各跑短 Adam，取最终 ``(u_flat, z)`` 作为优质种子；不足 ``pop_size``
    的部分用 (启发式 z + 零 u) 加确定性微扰补齐。提升原始场上 GA 的收敛速度。
    """
    nl = int(n_layers)
    params = thermal_tier(dict(params or {}), target, process_init,
                        material=material, thermal_solver=thermal_solver)
    z0 = normalize_process(process_init, bounds=tier_bounds(params))
    p = dict(params or {})
    spacing = float(target.spacing)
    eps = spacing
    u0 = jnp.zeros(target.shape, dtype=jnp.float64)
    w_vol = solid_weight(target.sdf, spacing)   # 体积平均权重 = 线性 cut 份额（#19）
    cell_vol = spacing ** target.dim
    mat = get_material(material)
    d_u = int(np.prod(target.shape))
    seeds: list[np.ndarray] = []
    for lam in lambdas:
        r = _run_lambda_point(
            target, float(lam), u0, z0, material=material, params=params,
            asbuilt_solver=asbuilt_solver, micro_solver=micro_solver,
            mbd_solver=mbd_solver, powder_solver=powder_solver,
            constitutive=constitutive, p=p, spacing=spacing, eps=eps, nl=nl,
            modality=modality, smoothness=smoothness,
            constraint_weight=constraint_weight, n_steps=n_steps,
            learning_rate=learning_rate, lr_shape=lr_shape, lr_process=lr_process,
            hard_project=hard_project, w_vol=w_vol, cell_vol=cell_vol, mat=mat,
            verbose=False, thermal_solver=thermal_solver)
        seeds.append(np.concatenate([
            np.asarray(r["u"], dtype=np.float64).ravel(),
            np.asarray(r["z"], dtype=np.float64).ravel()]))
    rng = np.random.default_rng(int(seed) + 1)
    z_heu = np.asarray(z0, dtype=np.float64)
    while len(seeds) < int(pop_size):
        u_noise = rng.uniform(-0.3 * u_clip, 0.3 * u_clip, size=d_u)
        z_noise = np.clip(z_heu + rng.uniform(-0.05, 0.05, size=9), 0.0, 1.0)
        seeds.append(np.concatenate([u_noise, z_noise]))
    init_pop = np.stack(seeds[:int(pop_size)], axis=0)
    lb = np.concatenate([np.full(d_u, -u_clip), np.zeros(9)])
    ub = np.concatenate([np.full(d_u, u_clip), np.ones(9)])
    return np.clip(init_pop, lb, ub)


def pareto_optimize_geometry_process(target: PartGeometry, *,
                                     material: str = "316L",
                                     process_init: ProcessPlan | None = None,
                                     init_delta: jnp.ndarray | None = None,
                                     n_steps: int = 20,
                                     learning_rate: float = 0.05,
                                     lr_shape: float | None = None,
                                     lr_process: float | None = None,
                                     lambdas: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                                     weights: Mapping[str, float] | None = None,
                                     params: Mapping[str, Any] | None = None,
                                     asbuilt_solver: str = "plastic",
                                     micro_solver: str = "surrogate",
                                     mbd_solver: str = "surrogate",
                                     powder_solver: str | None = None,
                                     constitutive: str = "j2",
                                     smoothness: float = 0.02,
                                     constraint_weight: float = 10.0,
                                     hard_project: bool = True,
                                     n_layers: int | None = None,
                                     modality: str = "SLM",
                                     verbose: bool = True,
                                     thermal_solver: str = "enthalpy",
                                     algorithm: str = "nsga2",
                                     pop_size: int = 16,
                                     n_gen: int = 8,
                                     eta_c: float = 15.0,
                                     eta_m: float = 20.0,
                                     p_crossover: float = 0.9,
                                     p_mutation: float = 0.0,
                                     seed: int = 0,
                                     u_clip: float = 3.0,
                                     seed_with_lambda: bool = False,
                                     n_steps_seed: int = 4) -> dict:
    """**多目标 Pareto 协同优化**：在**尺寸偏差(geom_dev)** 与 **残余应力(stress)**
    两个目标间做权衡，生成 Pareto 前沿。

    两种算法（``algorithm``）：

    * ``"nsga2"``（**默认**，对应任务 #37）：**真 NSGA-II 非支配排序**。在联合
      设计空间 ``x = concat(u_flat, z)`` 上进化——``u`` 为名义 SDF 修正场（体素单位），
      ``z`` 为归一化工艺向量（∈[0,1]⁹）。每代变异生成子代、父子合并(2N)、快速
      非支配排序、按 (非支配秩, 拥挤度) 截断回 N。加权求和只能覆盖**凸包**，
      会漏掉非凸段；NSGA-II 直接逼近**真实（凸/非凸皆可）**Pareto 前沿。
    * ``"lambda"``（**legacy**，保留用于回归/对比）：扫描 ``λ∈[0,1]`` 做
      ``L(λ)=λ·geom_dev+(1-λ)·stress`` 标量化，每个 λ 跑一次联合优化。

    返回 dict（两算法兼容键）：``geom_dev`` / ``stress`` / ``constraint_penalty`` /
    ``feasible``：前沿各点目标值与可行度；``plans`` / ``deltas`` / ``solutions``：
    各前沿解完整输出；``utopia``：理想点 (min geom_dev, min stress)；
    ``best_compromise_index`` / ``best_compromise``：目标空间归一化后离 utopia
    最近（L2）的最优点（默认推荐方案）。NSGA-II 额外返回 ``algorithm`` /
    ``pop_size`` / ``n_gen`` / ``population_geom_dev`` / ``population_stress`` /
    ``pareto_x``；lambda 路径返回 ``lambdas``。
    """
    if process_init is None:
        process_init = heuristic_plan(target, material=material, modality=modality)
    params = thermal_tier(dict(params or {}), target, process_init,
                        material=material, thermal_solver="enthalpy")
    z0 = normalize_process(process_init, bounds=tier_bounds(params))
    nl = n_layers or int(target.layer_count(40e-6))
    p = dict(params or {})
    spacing = float(target.spacing)
    eps = spacing
    mat = get_material(material)

    if init_delta is None:
        u0 = jnp.zeros(target.shape, dtype=jnp.float64)
    else:
        u0 = jnp.asarray(init_delta, dtype=jnp.float64) / spacing

    w_vol = solid_weight(target.sdf, spacing)   # 体积平均权重 = 线性 cut 份额（#19）
    cell_vol = spacing ** target.dim

    # ------------------------------------------------------------------
    if algorithm == "lambda":
        lambda_list = [float(x) for x in lambdas]
        geom_devs: list[float] = []
        stresses: list[float] = []
        pens: list[float] = []
        feas: list[float] = []
        plans: list = []
        deltas: list = []
        sols: list[dict] = []
        for lam in lambda_list:
            r = _run_lambda_point(
                target, lam, u0, z0, material=material, params=params,
                asbuilt_solver=asbuilt_solver, micro_solver=micro_solver,
                mbd_solver=mbd_solver, powder_solver=powder_solver,
                constitutive=constitutive, p=p, spacing=spacing, eps=eps, nl=nl,
                modality=modality, smoothness=smoothness,
                constraint_weight=constraint_weight, n_steps=n_steps,
                learning_rate=learning_rate, lr_shape=lr_shape,
                lr_process=lr_process, hard_project=hard_project, w_vol=w_vol,
                cell_vol=cell_vol, mat=mat, verbose=verbose,
                thermal_solver=thermal_solver)
            out = _joint_forward(
                target, r["u"], r["z"], material=material, params=params,
                asbuilt_solver=asbuilt_solver, micro_solver=micro_solver,
                mbd_solver=mbd_solver, powder_solver=powder_solver,
                constitutive=constitutive,
                p=p, spacing=spacing, eps=eps, n_layers=nl,
                modality=modality, service_stress=150e6,
                thermal_solver=thermal_solver)
            final_plan = denormalize_process(r["z"], n_layers=nl,
                                             modality=modality,
                                             bounds=tier_bounds(params))
            nominal = PartGeometry(sdf=target.sdf + r["u"] * spacing,
                                   origin=target.origin, spacing=spacing,
                                   dim=target.dim, name="nominal")
            gd, st, tv, pen = r["aux"]
            geom_devs.append(float(gd)); stresses.append(float(st))
            pens.append(float(pen)); feas.append(float(1.0 / (1.0 + pen)))
            plans.append(final_plan); deltas.append(r["u"] * spacing)
            sols.append({"lambda": lam, "final_plan": final_plan,
                         "delta": r["u"] * spacing, "nominal_geometry": nominal,
                         "asbuilt": out["asbuilt"],
                         "z": z_to_device_box(r["z"], n_layers=nl,
                                              bounds=tier_bounds(params)),
                         "geom_dev": float(gd), "stress": float(st),
                         "constraint_penalty": float(pen),
                         "penalty_history": r["pen_hist"]})
            if verbose:
                print(f"  pareto λ={lam:.2f}  geom_dev={float(gd):.3e}  "
                      f"stress={float(st):.3e}  pen={float(pen):.3e}  "
                      f"feas={float(1.0/(1.0+pen)):.3f}")
        geom_devs_a = jnp.asarray(geom_devs)
        stresses_a = jnp.asarray(stresses)
        gmin, gmax = jnp.min(geom_devs_a), jnp.max(geom_devs_a)
        smin, smax = jnp.min(stresses_a), jnp.max(stresses_a)
        gsp = (geom_devs_a - gmin) / jnp.maximum(gmax - gmin, 1e-30)
        ssp = (stresses_a - smin) / jnp.maximum(smax - smin, 1e-30)
        best_idx = int(jnp.argmin(jnp.sqrt(gsp ** 2 + ssp ** 2)))
        return {
            "algorithm": "lambda",
            "lambdas": jnp.asarray(lambda_list),
            "geom_dev": geom_devs_a, "stress": stresses_a,
            "constraint_penalty": jnp.asarray(pens),
            "feasible": jnp.asarray(feas),
            "plans": plans, "deltas": deltas, "solutions": sols,
            "utopia": (float(gmin), float(smin)),
            "best_compromise_index": best_idx,
            "best_compromise": sols[best_idx],
        }

    # ------------------------------------------------------------------
    # NSGA-II（默认）
    if algorithm != "nsga2":
        raise ValueError(f"未知 algorithm={algorithm!r}，应为 'nsga2' 或 'lambda'")
    d_u = int(np.prod(target.shape))
    lb = np.concatenate([np.full(d_u, -float(u_clip), dtype=np.float64),
                         np.zeros(9, dtype=np.float64)])
    ub = np.concatenate([np.full(d_u, float(u_clip), dtype=np.float64),
                         np.ones(9, dtype=np.float64)])

    init_pop = None
    if seed_with_lambda:
        init_pop = _nsga_seed_population(
            target, material=material, process_init=process_init,
            lambdas=(0.0, 0.5, 1.0), n_steps=int(n_steps_seed),
            learning_rate=learning_rate, lr_shape=lr_shape, lr_process=lr_process,
            smoothness=smoothness, constraint_weight=constraint_weight,
            params=params, asbuilt_solver=asbuilt_solver, micro_solver=micro_solver,
            mbd_solver=mbd_solver, powder_solver=powder_solver,
            constitutive=constitutive, hard_project=hard_project, n_layers=nl,
            modality=modality, u_clip=float(u_clip), pop_size=int(pop_size),
            seed=int(seed))

    def _objective(x):
        x = np.asarray(x, dtype=np.float64)
        u_flat = jnp.asarray(x[:d_u].reshape(target.shape), dtype=jnp.float64)
        z = jnp.clip(jnp.asarray(x[d_u:], dtype=jnp.float64), 0.0, 1.0)
        if hard_project:
            z = project_process_feasible(
                z, material=material, n_layers=nl, modality=modality, params=params)
        out = _joint_forward(
            target, u_flat, z, material=material, params=params,
            asbuilt_solver=asbuilt_solver, micro_solver=micro_solver,
            mbd_solver=mbd_solver, powder_solver=powder_solver,
            constitutive=constitutive,
            p=p, spacing=spacing, eps=eps, n_layers=nl,
            modality=modality, service_stress=150e6,
            thermal_solver=thermal_solver)
        gd, st, tv = _geom_stress_tv(
            out, target, w_vol, cell_vol, mat, spacing, eps, u_flat)
        return jnp.array([float(gd), float(st)], dtype=jnp.float64)

    cfg = NSGA2Config(pop_size=int(pop_size), n_gen=int(n_gen), eta_c=float(eta_c),
                      eta_m=float(eta_m), p_crossover=float(p_crossover),
                      p_mut_gene=float(p_mutation), seed=int(seed),
                      u_clip=float(u_clip))
    result = nsga2_minimize(_objective, lb, ub, 2, cfg,
                            init_pop=init_pop, verbose=verbose)

    # 解码非支配前沿为完整解
    plans = []; deltas = []; sols = []; pens = []; feas = []
    for xi in result.pareto_x:
        u_flat = jnp.asarray(np.asarray(xi[:d_u], dtype=np.float64).reshape(target.shape),
                             dtype=jnp.float64)
        z = jnp.clip(jnp.asarray(xi[d_u:], dtype=jnp.float64), 0.0, 1.0)
        if hard_project:
            z = project_process_feasible(
                z, material=material, n_layers=nl, modality=modality, params=params)
        out = _joint_forward(
            target, u_flat, z, material=material, params=params,
            asbuilt_solver=asbuilt_solver, micro_solver=micro_solver,
            mbd_solver=mbd_solver, powder_solver=powder_solver,
            constitutive=constitutive,
            p=p, spacing=spacing, eps=eps, n_layers=nl,
            modality=modality, service_stress=150e6,
            thermal_solver=thermal_solver)
        final_plan = denormalize_process(z, n_layers=nl, modality=modality,
                                         bounds=tier_bounds(params))
        nominal = PartGeometry(sdf=target.sdf + u_flat * spacing,
                               origin=target.origin, spacing=spacing,
                               dim=target.dim, name="nominal")
        gd, st, tv = _geom_stress_tv(
            out, target, w_vol, cell_vol, mat, spacing, eps, u_flat)
        pen, _ = process_constraint_penalty(
            z, material=material, n_layers=nl, modality=modality, params=params)
        plans.append(final_plan)
        deltas.append(u_flat * spacing)
        sols.append({"final_plan": final_plan, "delta": u_flat * spacing,
                     "nominal_geometry": nominal, "asbuilt": out["asbuilt"],
                     "z": z_to_device_box(z, n_layers=nl, modality=modality,
                                          bounds=tier_bounds(params)),
                     "geom_dev": float(gd), "stress": float(st),
                     "constraint_penalty": float(pen), "penalty_history": None})
        pens.append(float(pen)); feas.append(float(1.0 / (1.0 + pen)))

    geom_devs_a = result.pareto_f[:, 0]
    stresses_a = result.pareto_f[:, 1]
    # 最优点：目标空间归一化后离 utopia 最近（L2 距离最小）
    pg = result.pareto_f[:, 0]; ps = result.pareto_f[:, 1]
    pgmin, pgmax = float(pg.min()), float(pg.max())
    psmin, psmax = float(ps.min()), float(ps.max())
    gsp = (pg - pgmin) / max(pgmax - pgmin, 1e-30)
    ssp = (ps - psmin) / max(psmax - psmin, 1e-30)
    best_idx = int(np.argmin(np.sqrt(gsp ** 2 + ssp ** 2)))

    return {
        "algorithm": "nsga2",
        "pop_size": int(pop_size), "n_gen": int(n_gen),
        "geom_dev": geom_devs_a, "stress": stresses_a,
        "constraint_penalty": jnp.asarray(pens),
        "feasible": jnp.asarray(feas),
        "plans": plans, "deltas": deltas, "solutions": sols,
        "population_geom_dev": result.population_f[:, 0],
        "population_stress": result.population_f[:, 1],
        "utopia": tuple(float(v) for v in result.ideal_point),
        "best_compromise_index": best_idx,
        "best_compromise": sols[best_idx],
        "pareto_x": result.pareto_x,
    }


@register_solver(
    "inverse.predict_process",
    consumes=("PartGeometry",),
    produces="ProcessPlan",
    stage="process",
    modality=(),
    differentiable=True,
    cost=1.0,
    defaults={"material": "316L", "n_hidden": 16, "seed": 0},
    doc="由几何经可训练网络预测工艺方案（功能2：可微分工艺反演的契约生产者）",
)
def solve_inverse_process(*, geometry: PartGeometry, params) -> ProcessPlan:
    """即插即用的"几何→工艺"求解器（用默认随机权重，未训练）。

    训练版见 :func:`train_process_predictor`；此注册只是让 Pipeline 在
    需要 ProcessPlan 且给定 geometry 时能选中它，体现"模块即求解器"。
    """
    feats = geometry_features(geometry)
    rng = jax.random.PRNGKey(int(params.get("seed", 0)))
    theta = mlp_init(rng, int(feats.shape[0]), int(params.get("n_hidden", 16)),
                     n_out=len(PROCESS_BOUNDS), n_layers=2)
    return predict_process(theta, feats, n_layers=int(geometry.layer_count(40e-6)))
