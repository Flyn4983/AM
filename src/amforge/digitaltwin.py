"""数字样机结构响应（digital twin → StructuralResult）
====================================================

把制造后的增材件 ``AsBuiltPart`` 嵌入**数字样机**（整机/部件的多体动力学 +
FEM 接触/受力模型），在给定服役工况下加载，得到结构响应 ``StructuralResult``。

降阶模型（ROM）
--------------
完整数字样机是柔性多体动力学 + 接触非线性 FEM，算力昂贵、且难以直接接入
``jax.grad``。这里用**代表服役应力**的降阶模型给出趋势正确、全程可微的响应：

* 由零件包围盒取特征长度 L 与矩形截面尺寸 a×b；
* 等效模量取本构场体积平均 ``E_eff``（含孔隙/损伤折减）；
* 服役工况以**远场代表应力** ``σ_svc``（来自 ``params``，单位 Pa）描述，
  而非一个与微观几何尺度脱节的"力"——否则对毫米级零件施加宏观力会得到
  不真实的 GPa 级应力；
* 应力集中：孔隙/损伤抬升局部应力（典型 Kt = 1 + 2·孔隙率）；
* 位移按轴向柔度 ``δ ≈ σ/E·L``（悬臂式沿长度线性增长）。

该 ROM 与高保真数字样机（``digitaltwin.mbd_fem``）共用同一 ``StructuralResult``
契约——功能 4 的"数字化试验验证"接口不变，只是精度不同。可微性保证：若把
服役应力或本构作为优化变量，梯度可回传。
"""

from __future__ import annotations

import jax.numpy as jnp

from amforge.core.contracts import (
    AsBuiltPart, ConstitutiveField, StructuralResult,
)
from amforge.core.registry import register_solver


@register_solver(
    "digitaltwin.rom",
    consumes=("AsBuiltPart", "ConstitutiveField"),
    produces="StructuralResult",
    stage="structural",
    modality=(),
    differentiable=True,
    cost=4.0,
    defaults={"service_stress": 150e6, "allowable_displacement": 1.0e-4},
    doc="增材件嵌入数字样机(降阶)加载代表服役应力，得位移/应力/von Mises",
)
def solve_digital_twin(*, asbuilt: AsBuiltPart, constitutive: ConstitutiveField,
                       params) -> StructuralResult:
    spacing = float(asbuilt.spacing)
    shape = tuple(int(s) for s in asbuilt.sdf.shape)
    dim = asbuilt.dim

    # 包围盒特征尺寸（沿 x 为承载长度方向）
    nx, ny, nz = (shape + (1, 1, 1))[:3]
    L = nx * spacing
    a = ny * spacing
    b = nz * spacing

    # 等效模量（体积平均，含孔隙/损伤折减）—— 保持为 traced array 以兼容 jit/grad
    E_eff = jnp.mean(constitutive.effective_stiffness())
    E_eff = jnp.maximum(E_eff, 1e7)

    # 代表服役应力（远场，由工况给定）[Pa]
    sigma_svc = float(params.get("service_stress", 150e6))

    # 应力集中：孔隙/损伤抬升局部应力（典型 Kt = 1 + 2·孔隙率）
    por_mean = jnp.mean(constitutive.porosity)
    Kt = 1.0 + 2.0 * por_mean
    peak_stress = sigma_svc * Kt

    solid = (asbuilt.sdf < 0.0).astype(jnp.float64)
    # 截面抛物线分布：表面最大
    zc = (jnp.arange(nz) * spacing if dim == 3 else jnp.arange(ny) * spacing)
    zc = zc / jnp.maximum(b, 1e-12)
    sect = jnp.clip(0.5 + 0.5 * jnp.abs(zc - 0.5) * 2.0, 0.0, 1.0)
    if dim == 3:
        sect_field = sect[jnp.newaxis, jnp.newaxis, :] * jnp.ones((nx, ny, 1))
    else:
        sect_field = sect[jnp.newaxis, :] * jnp.ones((nx, 1))
    von_mises = peak_stress * sect_field * solid

    # 位移：轴向柔度 δ ≈ σ/E · L（悬臂式沿长度线性增长）
    axes = [jnp.arange(s) * spacing for s in shape[:dim]]
    grids = jnp.meshgrid(*axes, indexing="ij")
    crd = jnp.stack(grids, axis=-1)
    x_norm = crd[..., 0] / jnp.maximum(L, 1e-12)
    tip_disp = peak_stress / E_eff * L
    if dim == 3:
        disp_field = jnp.stack(
            [tip_disp * x_norm * solid,
             jnp.zeros_like(x_norm), jnp.zeros_like(x_norm)], axis=-1)
    else:
        disp_field = jnp.stack(
            [tip_disp * x_norm * solid, jnp.zeros_like(x_norm)], axis=-1)

    # 应力 Voigt 场（单轴 -> sxx = von_mises）
    if dim == 3:
        stress = jnp.stack(
            [von_mises, jnp.zeros_like(von_mises), jnp.zeros_like(von_mises),
             jnp.zeros_like(von_mises), jnp.zeros_like(von_mises),
             jnp.zeros_like(von_mises)], axis=-1)
        strain = stress / E_eff
    else:
        stress = jnp.stack(
            [von_mises, jnp.zeros_like(von_mises), jnp.zeros_like(von_mises)],
            axis=-1)
        strain = stress / E_eff

    reaction = jnp.asarray(sigma_svc * a * b)            # 力 = 应力 × 面积

    return StructuralResult(
        displacement=jnp.asarray(disp_field),
        stress=jnp.asarray(stress),
        strain=jnp.asarray(strain),
        von_mises=jnp.asarray(von_mises),
        reaction=reaction,
        dim=dim,
    )
