"""零件级热历史求解（thermal history）
=====================================

把**逐道熔池**（``MeltPoolResult``，由 ``meltpool`` 阶段的 Eagar-Tsai 代理
或 Flow3D VOF 给出）**累积**成零件级的热历史场 ``ThermalHistory``。

为什么需要这一层
----------------
熔池求解器给出的是*单道*（或单点）的瞬态温度场，而微观组织、残余应力、
数字样机关心的是*每个体素*在整个建造过程中经历的：

* ``peak_temperature``  —— 经历的最高温度（决定是否熔化/重熔）
* ``cooling_rate``      —— 凝固区间平均冷却速率 Ṫ [K/s]（控制晶粒细化）
* ``thermal_gradient``  —— 凝固前沿温度梯度 G [K/m]
* ``solidification_rate``—— 凝固前沿推进速率 R [m/s]
* ``time_above_melt``   —— 高于熔点的累计时间（重熔次数代理）
* ``final_temperature`` —— 结束时刻温度

桥接关系（Rosenthal 类简化）
----------------------------
对单道熔池，经典关系为：

    Ṫ ≈ 2·v·ΔT / d,      G ≈ ΔT / d,      R = v·w / √(w² + 4d²)

其中 v=扫描速度、d=熔深、w=熔宽、ΔT=糊状区宽度(T_liquidus−T_solidus)。
沿构建方向(z)，越靠近底部热沉越强（与基板相连），冷却更快、重熔次数更少；
越靠近顶部经历的重熔次数越多。本模块用几何位置对代表值做空间调制，
得到一个**可微**的零件级热历史场。

注意：这是*降阶*（reduced-order）热历史，不是逐层瞬态热固结求解。它足以
支撑微观组织/残余应力的趋势预测，且与 ``jax.grad`` 完全兼容——功能 2 的
可微闭环要靠它把工艺参数回传到组织与性能上。
"""

from __future__ import annotations

import jax.numpy as jnp

from amforge.core.contracts import (
    PartGeometry, ProcessPlan, MeltPoolResult, ThermalHistory,
)
from amforge.core.registry import register_solver
from amforge.materials import get_material


@register_solver(
    "thermal.history",
    consumes=("PartGeometry", "ProcessPlan", "MeltPoolResult"),
    produces="ThermalHistory",
    stage="thermal",
    modality=("SLM", "LSF"),
    differentiable=True,
    cost=2.0,
    method="analytical",
    defaults={"material": "316L"},
    doc="由逐道熔池累积得到零件级峰值温度/冷却速率/G-R 凝固判据（降阶模型）",
)
def solve_thermal_history(*, geometry: PartGeometry, process: ProcessPlan,
                          meltpool: MeltPoolResult, params) -> ThermalHistory:
    mat = get_material(params.get("material", "316L"))

    occ = geometry.occupancy.astype(jnp.float64)          # 1 在实体内部
    coords = geometry.coords()                             # (..., dim)
    spacing = float(geometry.spacing)
    z = coords[..., -1]                                    # 构建方向 = 最后一轴
    z_min = jnp.min(z)
    z_max = jnp.max(z)
    H = jnp.maximum(z_max - z_min, spacing)
    z_norm = (z - z_min) / H                              # 0=底部, 1=顶部

    # 代表熔池尺度（来自熔池契约的标量场）—— 保留为 tracer 以维持可微性
    d = jnp.asarray(meltpool.depth)                       # 熔深 [m]
    w = jnp.asarray(meltpool.width)                       # 熔宽 [m]
    Ltrack = jnp.maximum(jnp.asarray(meltpool.length),    # 熔长 [m]
                         jnp.maximum(w, d))
    Tp = jnp.max(jnp.asarray(meltpool.temperature))       # 代表峰值温度 [K]

    # 工艺量（逐层 -> 标量代表值，保持可微）
    v = jnp.mean(jnp.atleast_1d(jnp.asarray(process.scan_speed)))
    preheat = jnp.mean(jnp.atleast_1d(jnp.asarray(process.preheat_temp)))
    t_layer = jnp.asarray(process.layer_thickness)
    n_layers = jnp.asarray(process.n_layers, dtype=jnp.float64)

    Ts, Tl = mat.T_solidus, mat.T_liquidus
    dT = jnp.maximum(Tl - Ts, 1.0)                        # 糊状区宽度 [K]
    d_safe = jnp.maximum(d, 1e-7)

    # 重熔次数：某体素上方还压着多少层（底部热沉强、重熔少）
    n_reheat = jnp.clip((H - (z - z_min)) / jnp.maximum(t_layer, 1e-9),
                        0.0, n_layers)

    # 空间调制因子（可微）：
    #  * 底部(z_norm→0) 与基板相连，热沉强 -> 冷却更快
    #  * 顶部(z_norm→1) 上方已无材料，单道冷却
    cool_mod = 1.0 + 0.4 * (1.0 - z_norm)                 # 1.0~1.4

    peak = preheat + (Tp - preheat) * occ
    cooling_rate = (2.0 * v * dT / d_safe) * cool_mod * occ
    gradient = (Tp - Ts) / d_safe * occ
    # 凝固前沿推进速率：被熔池纵横比约束在 [0, v] 内
    R = v * w / jnp.sqrt(jnp.maximum(w ** 2 + 4.0 * d ** 2, 1e-18))
    time_above_melt = (Ltrack / jnp.maximum(v, 1e-9)) * (1.0 + n_reheat) * occ
    final = preheat * occ + mat.T_ambient * (1.0 - occ)

    return ThermalHistory(
        peak_temperature=jnp.asarray(peak),
        cooling_rate=jnp.asarray(cooling_rate),
        thermal_gradient=jnp.asarray(gradient),
        solidification_rate=jnp.asarray(R),
        time_above_melt=jnp.asarray(time_above_melt),
        final_temperature=jnp.asarray(final),
        spacing=spacing,
        dim=geometry.dim,
    )
