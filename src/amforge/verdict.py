"""数字化试验判定（verdict → ServiceVerdict）
============================================

功能 4 的落点：**基于数字化试验验证增材件可用性**。把数字样机的结构响应
``StructuralResult``、制造后构型 ``AsBuiltPart``、本构 ``ConstitutiveField``
综合成最终判定 ``ServiceVerdict``：强度安全系数、刚度利用率、估算疲劳寿命、
磨损深度、最危险点、通过/不通过，以及一份可微的**裕度报告**。

判定量（均可微）
----------------
* ``strength_safety_factor`` = σ_y / σ_max（屈服强度裕度）
* ``stiffness_ratio``        = δ_max / δ_allow（<1 合格）
* ``fatigue_life_cycles``    = Basquin 式 N = 0.5·(σ_a/σ_f')^{1/b}
* ``wear_depth``             = k_w · (σ_max/1GPa) · (N/1e6) · 1µm
* ``critical_location``      = argmax(von Mises) 坐标
* ``passed``                 = 强度∩刚度∩疲劳 三路软逻辑乘积（可微软阈值判定）

裕度报告 ``margin_report`` 同时给出残余应力峰值与孔隙率，方便工艺优化时
直接定位"差在哪"。
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from amforge.core.contracts import (
    StructuralResult, AsBuiltPart, ConstitutiveField, ServiceVerdict,
    SecondaryProcessResult,
)
from amforge.core.registry import register_solver
from amforge.materials import get_material


@register_solver(
    "verdict.service",
    consumes=("StructuralResult", "AsBuiltPart", "ConstitutiveField",
              "SecondaryProcessResult"),
    produces="ServiceVerdict",
    stage="verdict",
    modality=(),
    differentiable=True,
    cost=1.0,
    defaults={"material": "316L",
              "allowable_displacement": 1.0e-3,
              "min_cycles": 1.0e5,
              "fatigue_strength": 1.0e9,
              "fatigue_exponent": -0.10},
    doc="综合结构响应+本构+二次工艺给出强度/刚度/疲劳/磨损判定与可微裕度报告",
)
def solve_verdict(*, structural: StructuralResult, asbuilt: AsBuiltPart,
                  constitutive: ConstitutiveField, secondary: SecondaryProcessResult = None,
                  params) -> ServiceVerdict:
    mat = get_material(params.get("material", "316L"))
    allow_disp = float(params.get("allowable_displacement", 1.0e-3))
    min_cycles = float(params.get("min_cycles", 1.0e5))
    sigma_f_prime = float(params.get("fatigue_strength", 1.0e9))
    b_exp = float(params.get("fatigue_exponent", -0.10))

    # ---- 二次工艺（可选）：有则按处理后件评定，无则等价于成形件 ----------
    eff = secondary.part if secondary is not None else asbuilt

    # ---- 强度 -------------------------------------------------------------
    sigma_y = jnp.mean(constitutive.sigma_y0)
    peak_stress = jnp.max(structural.von_mises)
    sf = sigma_y / jnp.maximum(peak_stress, 1.0)

    # ---- 刚度 -------------------------------------------------------------
    disp_mag = jnp.linalg.norm(structural.displacement, axis=-1)
    max_disp = jnp.max(disp_mag)
    stiffness_ratio = max_disp / jnp.maximum(jnp.asarray(allow_disp), 1e-12)

    # ---- 疲劳（Basquin） --------------------------------------------------
    sigma_a = jnp.maximum(peak_stress / 2.0, 1.0)
    # N = 0.5 (σ_a / σ_f')^{1/b},  b<0
    fatigue = 0.5 * jnp.power(sigma_a / jnp.maximum(sigma_f_prime, 1.0),
                              1.0 / b_exp)

    # ---- 磨损（Archard 型：与应力、循环数成正比，系数极小） --------------
    # 量级预期：亚微米~几微米/寿命。注意疲劳循环数可能极大，故系数取 1e-9。
    wear = 1.0e-9 * (peak_stress / 1.0e9) * (fatigue / 1.0e8)

    # ---- 最危险点 ---------------------------------------------------------
    idx = jnp.argmax(structural.von_mises)
    shape = structural.von_mises.shape
    multi = jnp.unravel_index(idx, shape)               # (dim,) traced array
    spacing = float(eff.spacing)
    critical = jnp.stack([multi[d] * spacing
                         for d in range(len(shape))])    # traced-safe

    # ---- 通过判定（软逻辑，可微） ----------------------------------------
    pass_strength = jax.nn.sigmoid(2.0 * (sf - 1.0))
    pass_stiff = jax.nn.sigmoid(2.0 * (1.0 - stiffness_ratio))
    pass_fatigue = jax.nn.sigmoid(
        0.5 * (jnp.log10(jnp.maximum(fatigue, 1.0))
               - jnp.log10(jnp.maximum(min_cycles, 1.0))))
    passed_soft = pass_strength * pass_stiff * pass_fatigue
    passed = (passed_soft > 0.5).astype(jnp.float64)

    # ---- 裕度报告（值保持为 traced array，便于随 ServiceVerdict 一起 jit/grad；
    #           显示时由 summary() 取 float） ---------------------------------
    residual_vm_max = jnp.max(eff.von_mises_residual())
    if secondary is not None and secondary.density_after is not None:
        porosity = 1.0 - jnp.asarray(secondary.density_after, dtype=jnp.float64)
    else:
        porosity = jnp.mean(constitutive.porosity)
    margin_report = {
        "strength_safety_factor": jnp.asarray(sf),
        "stiffness_ratio": jnp.asarray(stiffness_ratio),
        "fatigue_cycles": jnp.asarray(fatigue),
        "residual_stress_vm_max_Pa": residual_vm_max,
        "porosity": porosity,
        "anisotropy_ratio": jnp.mean(constitutive.anisotropy_ratio()),
        "residual_relief_factor": (
            jnp.asarray(secondary.relief_ratio())
            if secondary is not None else jnp.asarray(1.0)),
        "density_after": (
            jnp.asarray(secondary.density())
            if secondary is not None else jnp.asarray(1.0)),
        "passed_soft": jnp.asarray(passed_soft),
    }

    return ServiceVerdict(
        strength_safety_factor=jnp.asarray(sf),
        stiffness_ratio=jnp.asarray(stiffness_ratio),
        fatigue_life_cycles=jnp.asarray(fatigue),
        wear_depth=jnp.asarray(wear),
        critical_location=critical,
        passed=jnp.asarray(passed),
        margin_report=margin_report,
    )
