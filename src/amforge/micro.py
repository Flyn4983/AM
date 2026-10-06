"""微观组织求解（microstructure）
===============================

由零件级热历史 ``ThermalHistory``（及熔池的缺陷指标）预测凝固组织
``MicrostructureResult``。

物理依据
--------
* **晶粒尺寸**：凝固组织的一次枝晶臂间距 / 晶粒尺寸与冷却速率满足幂律
  ``d ∝ Ṫ^{-n}``（经验指数 n≈0.3~0.4）。冷却越快，晶粒越细。
* **柱状 vs 等轴（CET）**：由凝固判据 ``G/R``（温度梯度 / 凝固速率）决定。
  ``G/R`` 大 -> 柱状晶（散热单向、择优生长）；``G/R`` 小 -> 等轴晶
  （成分过冷充分、形核占优）。这是增材制造"组织各向异性"的根源。
* **织构**：柱状晶沿构建方向择优取向 -> 强 ⟨001⟩/⟨100⟩ 织构，织构强度
  随柱状分数升高。
* **孔隙**：来自熔池缺陷指标（匙孔气孔 / 未熔合），直接搬移。

本模块是*统计*组织模型（给出场与体积平均），不是相场/CA 的逐晶粒演化；
后者（功能 3 的高保真路径）可在 ``micro.phasefield`` 求解器中替换，而
下游（本构、成形）接口不变——这正是"求解器可无缝替换"的体现。
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from amforge.core.contracts import (
    ThermalHistory, MeltPoolResult, MicrostructureResult,
)
from amforge.core.registry import register_solver


@register_solver(
    "micro.surrogate",
    consumes=("ThermalHistory", "MeltPoolResult"),
    produces="MicrostructureResult",
    stage="microstructure",
    modality=("SLM", "LSF"),
    differentiable=True,
    cost=2.0,
    doc="由冷却速率(G-R)预测晶粒尺寸/柱状-等轴分数/织构/孔隙（统计组织模型）",
)
def solve_microstructure(*, thermal: ThermalHistory,
                         meltpool: MeltPoolResult, params) -> MicrostructureResult:
    occ = (thermal.peak_temperature > 0.0).astype(jnp.float64)
    # 用峰值温度是否高于固相线判定"曾被熔化"（更稳健于上方 occ 来源）
    melted = (thermal.peak_temperature > 0.5 * thermal.peak_temperature.max()
              ).astype(jnp.float64)
    solid = jnp.maximum(occ, melted)

    # ---- 晶粒尺寸：d = c · Ṫ^{-n} ---------------------------------------
    CR = jnp.maximum(thermal.cooling_rate, 1.0)            # [K/s]
    n_exp = float(params.get("grain_exponent", 0.34))
    c_g = float(params.get("grain_coeff", 1.1e-3))         # [m·(K/s)^n]
    grain_size = c_g * jnp.power(CR, -n_exp)
    grain_size = jnp.clip(grain_size, 4e-6, 400e-6)        # [4µm, 400µm]
    grain_size = grain_size * solid + (1.0 - solid) * 50e-6

    # ---- 柱状/等轴：sigmoid 作用在 log(G/R) 上 --------------------------
    GR = thermal.gr_ratio()                                # G/R
    GR = jnp.maximum(GR, 1e-6)
    GR_crit = float(params.get("GR_crit", 4.5e6))          # 临界 G/R [K·s/m²]
    k_cet = float(params.get("cet_slope", 0.6))
    columnar = jax.nn.sigmoid(k_cet * (jnp.log(GR) - jnp.log(GR_crit)))
    columnar = columnar * solid                            # 只在凝固区有意义

    texture = columnar * 0.85                              # 织构强度 ≤ 0.85

    # ---- 孔隙：搬移熔池缺陷指标（标量 -> 场） ---------------------------
    por = jnp.asarray(meltpool.porosity_indicator)
    porosity = por * solid

    # ---- 取向角场：柱状区沿构建方向，等轴区随机(取常数近似) -----------
    orientation = columnar * float(params.get("build_angle", 0.0))

    # ---- 固相序参量 phi：实体区 ≈ 1 ------------------------------------
    phi = solid

    # ---- 体积平均晶粒尺寸（供本构/报表） -------------------------------
    cell = thermal.spacing ** thermal.dim
    mean_g = (jnp.sum(grain_size * solid) * cell
              / jnp.maximum(jnp.sum(solid) * cell, 1e-30))

    return MicrostructureResult(
        phi=jnp.asarray(phi),
        orientation=jnp.asarray(orientation),
        grain_size=jnp.asarray(grain_size),
        columnar_fraction=jnp.asarray(columnar),
        texture_intensity=jnp.asarray(texture),
        porosity=jnp.asarray(porosity),
        mean_grain_size=jnp.asarray(mean_g),
        spacing=thermal.spacing,
        dim=thermal.dim,
    )
