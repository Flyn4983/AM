"""逐层增材成形（buildup → AsBuiltPart）
========================================

把工艺/热历史/微观组织综合成**制造后实际构型** ``AsBuiltPart``：含热收缩与
翘曲的变形场、残余应力场、残余应变场，以及变形后的带符号距离场 ``sdf``。

此契约是功能 2（可微闭环）中与"理想构型"作比较的对象——几何尺寸偏差 +
残余应力 + 残余应变共同构成损失函数。因此本模块的全部输出都是**可微**的：
它们对上游工艺参数（经热历史、微观）有非零梯度。

物理（降阶模型）
----------------
* **热应变**：线膨胀系数 × 过冲温升 ``ε_th = α·(T_peak − T_preheat)``，是
  收缩/翘曲的根源。
* **残余应力**（淬火型）：``σ_vm ∝ E·α·ΔT``，乘以几何位置因子（约束处/自由
  端应力分布不同）。这里给出 Voigt 分量场（3D=6, 2D=3）。
* **位移/翘曲**：``u ∝ ε_th · 坐标``（悬臂式：离基板越远变形越大）。
* **变形 SDF**：沿表面法向把名义 SDF 推移 ``Σ(u·n)``，得到真实"制造后"几何。

这些都是工程上常用的趋势模型（量级正确、对工艺单调），足以驱动功能 2 的
梯度优化；要更高保真可接入 ``buildup.thermomechanical``（逐层生死单元 FEM）。
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from amforge.core.contracts import (
    PartGeometry, ProcessPlan, ThermalHistory, MicrostructureResult, AsBuiltPart,
)
from amforge.core.registry import register_solver
from amforge.materials import get_material


@register_solver(
    "buildup.layer_activation",
    consumes=("PartGeometry", "ProcessPlan", "ThermalHistory",
              "MicrostructureResult"),
    produces="AsBuiltPart",
    stage="buildup",
    modality=(),
    differentiable=True,
    cost=3.0,
    defaults={"material": "316L", "distortion_factor": 0.6},
    doc="由热历史/微观综合出制造后构型：残余应力/应变场、翘曲变形、变形SDF",
)
def solve_buildup(*, geometry: PartGeometry, process: ProcessPlan,
                  thermal: ThermalHistory, microstructure: MicrostructureResult,
                  params) -> AsBuiltPart:
    mat = get_material(params.get("material", "316L"))
    k_d = float(params.get("distortion_factor", 0.6))

    occ = geometry.occupancy.astype(jnp.float64)
    coords = geometry.coords()
    spacing = float(geometry.spacing)
    dim = geometry.dim

    # ---- 热应变 / 残余应力 ------------------------------------------------
    dT_peak = jnp.maximum(thermal.peak_temperature
                          - jnp.mean(jnp.atleast_1d(jnp.asarray(process.preheat_temp))),
                          0.0)
    eps_th = mat.alpha_thermal * dT_peak                     # 热应变场

    # 残余应力由"淬火"产生，但塑性松弛把它**限制在屈服强度附近**
    # （真实 LPBF 残余应力 ~ 0.3~0.8 σ_y，绝不会是 E·ε_th 这种量级）。
    # 这里用热应变归一化做空间调制，幅值封顶在 k_q·σ_y。
    eps_max = jnp.maximum(jnp.max(eps_th), 1e-6)
    thermal_norm = eps_th / eps_max                          # [0,1]
    z = coords[..., -1]
    z_norm = (z - jnp.min(z)) / jnp.maximum(jnp.max(z) - jnp.min(z), spacing)
    pos_factor = 1.0 - 0.3 * z_norm                         # 约束处应力高
    k_q = float(params.get("residual_fraction", 0.6))
    s_vm = k_q * mat.sigma_y * (0.4 + 0.6 * thermal_norm) * pos_factor * occ

    if dim == 3:
        # Voigt: sxx, syy, szz, sxy, syz, szx
        sxx = 0.5 * s_vm
        syy = 0.5 * s_vm
        szz = 1.0 * s_vm
        shear = 0.15 * s_vm
        residual_stress = jnp.stack([sxx, syy, szz, shear, shear, shear], axis=-1)
        # 残余应变（热应变，Voigt: 3D=6 分量）
        exx = eps_th * 0.5 * occ
        ezz = eps_th * occ
        z0 = jnp.zeros_like(exx)
        residual_strain = jnp.stack([exx, exx, ezz, z0, z0, z0], axis=-1)
    else:
        sxx = 0.6 * s_vm
        syy = 0.9 * s_vm
        sxy = 0.2 * s_vm
        residual_stress = jnp.stack([sxx, syy, sxy], axis=-1)
        residual_strain = jnp.stack(
            [eps_th * 0.5 * occ, eps_th * occ, jnp.zeros_like(eps_th)], axis=-1)

    # ---- 翘曲位移：悬臂式 u ∝ ε_th · 坐标 -------------------------------
    # 收缩沿坐标方向（离原点越远累计越大）
    displacement = -k_d * eps_th[..., None] * coords * occ[..., None]

    # ---- 变形 SDF：沿法向推移名义 SDF -----------------------------------
    # 法向由几何决定，在"工艺优化"场景下与工艺参数无关，故 stop_gradient：
    # 既符合物理（几何固定），又规避 jnp.gradient 对常量输入的 vjp 数值问题。
    grad = jnp.stack(jnp.gradient(geometry.sdf, spacing), axis=-1)   # (..., dim)
    glen = jnp.linalg.norm(grad, axis=-1, keepdims=True)
    n_hat = jax.lax.stop_gradient(grad / jnp.maximum(glen, 1e-12))
    sdf_def = geometry.sdf + jnp.sum(displacement * n_hat, axis=-1)

    return AsBuiltPart(
        sdf=jnp.asarray(sdf_def),
        displacement=jnp.asarray(displacement),
        residual_stress=jnp.asarray(residual_stress),
        residual_strain=jnp.asarray(residual_strain),
        spacing=spacing,
        dim=dim,
    )
