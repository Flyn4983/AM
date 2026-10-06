"""二次工艺仿真（Module B）—— 热处理 / HIP / 切削，站在 asbuilt 之上
=========================================================================

Simufact / ANSYS 的标配能力，也是当前链路在 ``buildup / structural`` 之后直接断到
``verdict`` 时缺的"中间再处理"环节。SLM 成形件通常还需一道二次工艺才交付：

* **HT（去应力退火 / 固溶热处理）**：高温下残余应力按 Arrhenius 松弛规律
  衰减（σ(t)=σ₀·exp(−t/τ)，τ=τ₀·exp(Q/RT)），并叠加再结晶附加释放；
  这是可微的闭式应力松弛模型，不重新跑 FEM。
* **HIP（热等静压）**：高温 + 等静压使封闭孔隙幂律致密化（残余孔隙随 P·t
  指数衰减），同时高温进一步释放残余应力——密度场被闭向 ~1.0。
* **machining（切削 / 精加工）**：余料去除，最终形状 = 名义几何 ∩ 成形件
  （SDF 取较大者），并给出最终形位偏差；differentiable=False（几何布尔非光滑）。

所有求解器站在 :class:`~amforge.core.contracts.AsBuiltPart` 之上，把真实物理效应
写回一个 :class:`~amforge.core.contracts.SecondaryProcessResult`：

* ``part``：处理后的成形件（残余应力/位移/应变/几何已被二次工艺改写）；
* ``treatment`` / ``residual_relief_factor`` / ``density_after`` / ``geometry_after``：
  供 ``verdict`` 做"残余应力按松弛、孔隙按密度修正"的评定。

设计要点
--------
* 全部为纯函数 + 契约通信，沿用 ``register_solver`` 统一签名，可无缝接入
  双注册表与 ``Pipeline``。
* 默认不进轻量链路：``postprocess.passthrough``（cost=0.1，直通"无二次工艺"）是
  ``verdict`` 求解 ``secondary`` 端口时的最低代价生产者；HT/HIP/machining 通过
  ``select={"secondary": "postprocess.heat_treatment"}`` 等显式切换。
* ``verdict.service`` 现有 ``consumes`` 追加 ``secondary``：有则按处理后件评定，
  无（passthrough）等价于直接用成形件——向后兼容。
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from amforge.core.contracts import (
    AsBuiltPart,
    PartGeometry,
    SecondaryProcessResult,
)
from amforge.core.registry import register_solver

_R_GAS = 8.314  # 通用气体常数 [J/(mol·K)]


def _relieved_part(asbuilt: AsBuiltPart, relief: jnp.ndarray) -> AsBuiltPart:
    """按残余应力保留比例 ``relief`` ∈ (0,1] 改写成形件的应力/应变/位移场。"""
    relief = jnp.asarray(relief, dtype=jnp.float64)
    return AsBuiltPart(
        sdf=asbuilt.sdf,
        displacement=asbuilt.displacement * relief,
        residual_stress=asbuilt.residual_stress * relief,
        residual_strain=asbuilt.residual_strain * relief,
        spacing=asbuilt.spacing,
        dim=asbuilt.dim,
    )


# ---------------------------------------------------------------------------
# 求解器 0：直通（无二次工艺）—— verdict 默认 lowest-cost 生产者
# ---------------------------------------------------------------------------
@register_solver(
    "postprocess.passthrough",
    consumes=("AsBuiltPart",),
    produces="SecondaryProcessResult",
    stage="postprocess",
    modality=(),
    differentiable=True,
    cost=0.1,
    doc="无二次工艺（直通）：把成形件原样包成 SecondaryProcessResult，供 verdict 默认消费",
)
def postprocess_passthrough(*, asbuilt: AsBuiltPart, params=None) -> SecondaryProcessResult:
    """默认「无二次工艺」节点：把成形件原样包进 SecondaryProcessResult。

    它是 ``verdict`` 求解 ``secondary`` 端口时的最低代价生产者，使默认链路
    （未显式选 HT/HIP/machining）也能产出合法的 SecondaryProcessResult，
    verdict 据此等价于直接评定成形件。
    """
    return SecondaryProcessResult(
        part=asbuilt,
        treatment="none",
        residual_relief_factor=jnp.asarray(1.0, dtype=jnp.float64),
        density_after=jnp.asarray(1.0, dtype=jnp.float64),
        geometry_after=asbuilt.sdf,
        applied_processes=(),
        params={},
    )


# ---------------------------------------------------------------------------
# 求解器 1：热处理（去应力退火 / 固溶）—— 应力松弛 + 再结晶
# ---------------------------------------------------------------------------
@register_solver(
    "postprocess.heat_treatment",
    consumes=("AsBuiltPart",),
    produces="SecondaryProcessResult",
    stage="postprocess",
    modality=("SLM", "LSF"),
    differentiable=True,
    cost=5.0,
    defaults={"temperature_K": 1200.0, "hold_time_s": 3600.0,
              "reference_temp_K": 298.0, "activation_energy_J_per_mol": 180e3,
              "melt_temp_K": 1900.0, "tau0_s": 1.0e-3},
    doc="闭式应力松弛+再结晶（残余应力按温度/时间 Arrhenius 衰减，可微）",
)
def ht_relax(*, asbuilt: AsBuiltPart, params=None) -> SecondaryProcessResult:
    """去应力退火 / 固溶热处理：残余应力按 Arrhenius 黏弹性松弛规律衰减。

    物理模型（可微闭式）
    ---------------------
    * 松弛时间 ``τ = τ₀·exp(Q / (R·T))``：温度越高松弛越快。
    * 残余应力保留比例 ``relief = exp(−t / τ)``，夹紧在 (1e-3, 1]。
    * 再结晶附加释放：当 T 高于约 0.45 Tm 时，再结晶进一步释放约一半应力
      （``relief_eff = relief·(1 − 0.5·sigmoid((T−0.45Tm)/0.05Tm))``）。
    * 改写 ``part`` 的残余应力/应变/位移场（位移部分回弹）。

    参数
    ----
    params : dict
        ``temperature_K``（退火温度，默认 1200 K）、``hold_time_s``（保温时间，
        默认 3600 s）、``activation_energy_J_per_mol``（自扩散激活能，默认 180 kJ/mol）、
        ``melt_temp_K``（熔点，默认 1900 K）、``tau0_s``（特征时间，默认 1e-3 s）。
    """
    p = dict(params or {})
    T = float(p.get("temperature_K", 1200.0))
    hold = float(p.get("hold_time_s", 3600.0))
    Q = float(p.get("activation_energy_J_per_mol", 180e3))
    Tm = float(p.get("melt_temp_K", 1900.0))

    # 应力松弛（Arrhenius）
    tau = 1.0e-3 * jnp.exp(Q / (_R_GAS * max(T, 1.0)))
    relief = jnp.exp(-hold / jnp.maximum(tau, 1e-12))
    relief = jnp.clip(relief, 1e-3, 1.0)
    # 再结晶附加释放
    recryst = jax.nn.sigmoid((T - 0.45 * Tm) / (0.05 * Tm))
    relief_eff = relief * (1.0 - 0.5 * recryst)
    relief_eff = jnp.clip(relief_eff, 1e-3, 1.0)

    part = _relieved_part(asbuilt, relief_eff)
    return SecondaryProcessResult(
        part=part,
        treatment="HT",
        residual_relief_factor=jnp.asarray(relief_eff, dtype=jnp.float64),
        density_after=jnp.asarray(1.0, dtype=jnp.float64),  # 退火不改变密度
        geometry_after=asbuilt.sdf,
        applied_processes=("HT",),
        params={"temperature_K": T, "hold_time_s": hold,
                "activation_energy_J_per_mol": Q},
    )


# ---------------------------------------------------------------------------
# 求解器 2：热等静压（HIP）—— 孔隙闭合 + 应力释放
# ---------------------------------------------------------------------------
@register_solver(
    "postprocess.hip",
    consumes=("AsBuiltPart",),
    produces="SecondaryProcessResult",
    stage="postprocess",
    modality=("SLM", "LSF"),
    differentiable=True,
    cost=5.0,
    defaults={"pressure_Pa": 1.0e8, "temperature_K": 1100.0,
              "hold_time_s": 7200.0, "initial_porosity": 0.02,
              "activation_energy_J_per_mol": 200e3, "tau0_s": 1.0e-3},
    doc="孔隙闭合：密度场→~1.0（幂律致密化，可微）+ 高温应力释放",
)
def hip_densify(*, asbuilt: AsBuiltPart, params=None) -> SecondaryProcessResult:
    """热等静压：封闭孔隙幂律致密化 + 高温残余应力释放。

    物理模型（可微闭式）
    ---------------------
    * 致密化速率随压力与温度热活化增强：``k = k₀·exp(Q/(R·T))·(P/1e8)``。
    * 残余孔隙 ``por = por₀·exp(−k·t)``，夹紧在 (1e-4, 1]；密度
      ``density_after = 1 − por``（HIP 把孔隙闭向 ~1.0）。
    * 残余应力按与 HT 同源的 Arrhenius 松弛释放（HIP 温度/压力更高，释放更强）。

    参数
    ----
    params : dict
        ``pressure_Pa``（等静压，默认 100 MPa）、``temperature_K``（默认 1100 K）、
        ``hold_time_s``（默认 7200 s）、``initial_porosity``（初始孔隙率，默认 0.02）、
        ``activation_energy_J_per_mol``（致密化激活能，默认 200 kJ/mol）。
    """
    p = dict(params or {})
    P = float(p.get("pressure_Pa", 1.0e8))
    T = float(p.get("temperature_K", 1100.0))
    hold = float(p.get("hold_time_s", 7200.0))
    init_por = float(p.get("initial_porosity", 0.02))
    Q = float(p.get("activation_energy_J_per_mol", 200e3))

    # 孔隙闭合（幂律致密化）
    k = 1.0e-6 * jnp.exp(Q / (_R_GAS * max(T, 1.0))) * (P / 1.0e8)
    por_after = init_por * jnp.exp(-k * hold)
    por_after = jnp.clip(por_after, 1e-4, 1.0)
    density_after = 1.0 - por_after

    # 高温残余应力释放（与 HT 同构，HIP 释放更强）
    tau0 = float(p.get("tau0_s", 1.0e-3))
    tau = tau0 * jnp.exp(Q / (_R_GAS * max(T, 1.0)))
    relief = jnp.exp(-hold / jnp.maximum(tau, 1e-12))
    relief = jnp.clip(relief, 1e-3, 1.0)

    part = _relieved_part(asbuilt, relief)
    return SecondaryProcessResult(
        part=part,
        treatment="HIP",
        residual_relief_factor=jnp.asarray(relief, dtype=jnp.float64),
        density_after=jnp.asarray(density_after, dtype=jnp.float64),
        geometry_after=asbuilt.sdf,  # HIP 尺寸变化极小，近似几何不变
        applied_processes=("HIP",),
        params={"pressure_Pa": P, "temperature_K": T, "hold_time_s": hold,
                "initial_porosity": init_por},
    )


# ---------------------------------------------------------------------------
# 求解器 3：切削 / 精加工（余料去除 + 最终形位）
# ---------------------------------------------------------------------------
@register_solver(
    "postprocess.machining",
    consumes=("AsBuiltPart", "PartGeometry"),
    produces="SecondaryProcessResult",
    stage="postprocess",
    modality=("SLM", "LSF"),
    differentiable=False,
    cost=2.0,
    defaults={"residual_relief_factor": 1.0},
    doc="余料去除 + 最终形位偏差（几何布尔修形），喂给 verdict",
)
def subtractive(*, asbuilt: AsBuiltPart, geometry: PartGeometry,
                params=None) -> SecondaryProcessResult:
    """切削 / 精加工：把成形件修形到名义几何，并给出最终形位。

    真实几何效应（确定性）
    -----------------------
    * 最终形状 = 名义几何 ∩ 成形件：``geometry_after = max(asbuilt.sdf, geometry.sdf)``
      （两者均以负值为实体内部，取较大者 = 取更严格的内部，即裁掉超出名义的余料、
      支撑残根与局部超差）。
    * 切削引入薄层表面残余应力，但主体残余应力近似保留；默认保留比例 1.0
      （可通过 ``residual_relief_factor`` 调整）。
    * 切削后零件视为全致密（``density_after = 1.0``）。

    differentiable=False：``max`` 几何布尔在界面处非光滑，不进入可微闭环；
    但它作为 verdict 前的一道确定性再处理完全合法（与仿真解耦的制造步骤）。

    参数
    ----
    params : dict
        ``residual_relief_factor``（切削后主体残余应力保留比例，默认 1.0）。
    """
    p = dict(params or {})
    relief = float(p.get("residual_relief_factor", 1.0))
    # 最终形状 = 名义 ∩ 成形（取更严格内部）
    geo_after = jnp.maximum(asbuilt.sdf, geometry.sdf)
    part = AsBuiltPart(
        sdf=geo_after,
        displacement=asbuilt.displacement * relief,
        residual_stress=asbuilt.residual_stress * relief,
        residual_strain=asbuilt.residual_strain * relief,
        spacing=asbuilt.spacing,
        dim=asbuilt.dim,
    )
    return SecondaryProcessResult(
        part=part,
        treatment="machining",
        residual_relief_factor=jnp.asarray(relief, dtype=jnp.float64),
        density_after=jnp.asarray(1.0, dtype=jnp.float64),
        geometry_after=geo_after,
        applied_processes=("machining",),
        params={"residual_relief_factor": relief},
    )


__all__ = [
    "postprocess_passthrough",
    "ht_relax",
    "hip_densify",
    "subtractive",
]
