"""本构均质化（constitutive homogenization）
==========================================

把微观组织 ``MicrostructureResult`` 升格为**空间分辨的各向异性弹塑性本构
参数场** ``ConstitutiveField``——这是可行性报告点明的"微观 → 零件级本构"关键
断点。

物理依据
--------
* **屈服强度（Hall-Petch）**：``σ_y = σ_0 + k_hp / √d``。晶粒越细（冷却越快）
  屈服强度越高。这是 LPBF 零件强度普遍高于锻件的根源。
* **弹性模量（孔隙折减）**：用 Mori-Tanaka 型简化 ``E_eff ≈ E·(1−1.9p)``，
  p 为孔隙率。同时因柱状晶沿构建方向的晶界弱化，``E_build`` 略低于
  ``E_inplane``——形成横观各向同性（构建方向 Z 为对称轴）。
* **屈服（Hill48）**：横观各向同性材料的各向异性系数由 ``E_in/E_build``
  推导，使下游 FEM/数字样机可表达增材件固有的各向异性。
* **Voce 硬化**：饱和硬化增量与速率取材料基准值，可由相变/析出进一步强化。

输出是逐点场，因此能表达增材件沿高度、随工艺变化的本构不均匀性——这正是
"数据契约作为模块间唯一通信语言"的价值：微观模块只需吐出本构场，下游
（成形残余应力、数字样机）直接消费，无需关心它来自相场还是统计模型。
"""

from __future__ import annotations

import jax.numpy as jnp

from amforge.core.contracts import MicrostructureResult, ConstitutiveField
from amforge.core.registry import register_solver
from amforge.materials import get_material


@register_solver(
    "constitutive.homogenize",
    consumes=("MicrostructureResult",),
    produces="ConstitutiveField",
    stage="constitutive",
    modality=(),
    differentiable=True,
    cost=1.0,
    defaults={"material": "316L"},
    doc="由晶粒尺寸(Hall-Petch)与柱状分数(各向异性)得到空间分辨弹塑性本构场",
)
def solve_constitutive(*, microstructure: MicrostructureResult,
                       params) -> ConstitutiveField:
    mat = get_material(params.get("material", "316L"))

    d = jnp.maximum(microstructure.grain_size, 1e-9)
    col = microstructure.columnar_fraction
    por = microstructure.porosity

    # ---- Hall-Petch 屈服强度 --------------------------------------------
    sigma_y0 = mat.hall_petch_sigma0 + mat.hall_petch_k / jnp.sqrt(d)

    # ---- 弹性模量：孔隙折减 + 柱状各向异性 ------------------------------
    por_eff = jnp.clip(por, 0.0, 0.3)
    red = (1.0 - 1.9 * por_eff) * (1.0 - por_eff)        # Mori-Tanaka 型
    E_in = mat.E * red
    # 柱状区沿构建方向弱化（晶界滑移），各向异性度 0.85~1.0
    aniso = 1.0 - 0.12 * col
    E_build = E_in * aniso
    nu = mat.nu
    G_build = E_build / (2.0 * (1.0 + nu))

    # ---- Hill48 横观各向同性系数 ----------------------------------------
    # 轴 = 构建方向 Z；各向异性比 r = E_in/E_build（>1 表示面内更刚）
    r = E_in / jnp.maximum(E_build, 1e-9)
    hill_F = hill_G = hill_H = jnp.full_like(E_in, 0.5)
    # L = N = (r - 1)/2（横观各向同性标准映射），限幅避免退化
    hill_L = hill_M = hill_N = jnp.clip(0.5 * (r - 1.0), -0.4, 0.4) + 0.5

    # ---- Voce 硬化（材料基准，可后续接入析出强化） ----------------------
    hardening_sat = jnp.full_like(E_in, float(mat.hardening_sat))
    hardening_rate = jnp.full_like(E_in, float(mat.hardening_rate))

    damage = jnp.zeros_like(por)

    return ConstitutiveField(
        E_inplane=jnp.asarray(E_in),
        E_build=jnp.asarray(E_build),
        nu=jnp.asarray(nu),
        G_build=jnp.asarray(G_build),
        sigma_y0=jnp.asarray(sigma_y0),
        hardening_sat=jnp.asarray(hardening_sat),
        hardening_rate=jnp.asarray(hardening_rate),
        hill_F=jnp.asarray(hill_F),
        hill_G=jnp.asarray(hill_G),
        hill_H=jnp.asarray(hill_H),
        hill_L=jnp.asarray(hill_L),
        hill_M=jnp.asarray(hill_M),
        hill_N=jnp.asarray(hill_N),
        porosity=jnp.asarray(por),
        damage=jnp.asarray(damage),
        spacing=microstructure.spacing,
        dim=microstructure.dim,
    )
