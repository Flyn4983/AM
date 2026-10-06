"""粉末床求解器（P2 粉末尺度）：DEM 铺粉 / SPH 熔道 / MPM 剥蚀-飞溅。

把 ``diffmech`` 已有的三个**可微**颗粒内核接到 AMForge 的 ForgeCore 契约
体系上，注册为 ``powder.bed`` 高保真求解器。这是 #35（注册 DEM/SPH/MPM 粉末
床求解器到 forgecore）与 #18（高保真求解器接入 + 4 缺口）的核心闭合点：
连续介质熔池模型看不到颗粒尺度缺陷（球化 / 未熔合 / 飞溅），本求解器把
它们解析成可微指标供逆问题的 defect 罚项使用。

精度 / 速度权衡（用户的硬需求点）
----------------------------------
``powder_fidelity`` 选择器直接体现这一点，与 ``micro_solver`` / ``mbd_solver``
同源设计：

* ``"surrogate"``（默认、零回归）：用工艺无量纲数直接给出 8 项指标，快、
  全可微、零颗粒开销——供**每次**逆问题迭代廉价评估；
* ``"dem"``：DEM 铺粉重排，解析铺粉密实度 / 配位数 / 粗糙度（密实度是
  "解出来"的而非假设的）；
* ``"sph"``：SPH 熔道自由界面，解析 Plateau–Rayleigh 球化判据；
* ``"mpm"``：MPM 粉末再分布，解析剥蚀半宽 + 飞溅质量分数。

显式颗粒档（dem/sph/mpm）只解析其专属缺陷，熔融相位缺陷（其余项）仍由解析
代理补全，保证契约完整、``defect_score`` 始终有意义。每个显式档都比 surrogate
贵 1~3 个数量级，只应在关键工艺点做高保真校验（与 mbd ``coupled``/``newton``
档一致）。

tracer 安全性：严格遵守 ``diffmech/methods/am/powder_bed.py`` 模块 docstring
的 4 条硬约束——只读几何的静态 bbox 元数据（绝不读 sdf 具体值）、接触参数为
具体 float、扫描轨迹 waypoints 静态 numpy、能量 / 时序可微。因此本求解器可
被 ``jax.grad`` 穿透，``laser_power`` / ``scan_speed`` / ``layer_thickness`` 等
工艺参数的梯度照常回传。
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from typing import Any, Mapping

from amforge.core.contracts import PowderBedResult, PartGeometry, ProcessPlan
from amforge.core.registry import register_solver
from amforge.materials import get_material
from diffmech.methods.am.powder_bed import (
    PowderBedConfig,
    analytic_powder_defects,
    recoat_dem,
    melt_track_sph,
    denudation_mpm,
    slm_config_from_plan,
)


# ---------------------------------------------------------------------------
# 由几何 + 工艺构造求解配置
# ---------------------------------------------------------------------------
def _powder_config(geometry: PartGeometry, process: ProcessPlan,
                   p: Mapping[str, Any]) -> PowderBedConfig:
    """从几何 bbox（静态元数据）+ 工艺层厚构造 ``PowderBedConfig``。

    只取几何的 ``bbox()``（由 ``origin``/``spacing``/``shape`` 推导，全静态，
    不读 ``sdf`` 任何具体值），保证 tracer 几何也安全。``bed_lx``/``bed_ly``
    取成形窗口的代表性横向尺寸（非整个成形缸），``layer_thickness`` 由工艺
    驱动（可微叶子）。
    """
    lo, hi = geometry.bbox()
    lo_np = np.asarray(lo)
    hi_np = np.asarray(hi)
    bed_lx = float(hi_np[0] - lo_np[0])
    bed_ly = float(hi_np[1] - lo_np[1]) if geometry.dim >= 2 else bed_lx
    lt = jnp.ravel(jnp.asarray(process.layer_thickness))[0]
    return PowderBedConfig(
        layer_thickness=lt,
        bed_lx=jnp.asarray(bed_lx, dtype=jnp.float64),
        bed_ly=jnp.asarray(bed_ly, dtype=jnp.float64),
        d50=float(p.get("d50", 30e-6)),
        psd_sigma=float(p.get("psd_sigma", 0.25)),
        rho_powder=float(p.get("rho_powder", 7950.0)),
        nx=int(p.get("nx", 6)),
        ny=int(p.get("ny", 6)),
        nz=int(p.get("nz", 3)),
        dim=int(geometry.dim),
        seed=int(p.get("seed", 0)),
        initial_positions=p.get("initial_positions", None),
        initial_radii=p.get("initial_radii", None),
    )


def _slm_from_process(process: ProcessPlan):
    """由 ProcessPlan 的可微叶子构造 ``SLMConfig``（已注册 pytree）。"""
    return slm_config_from_plan(
        laser_power=process.laser_power,
        scan_speed=process.scan_speed,
        layer_thickness=process.layer_thickness,
        hatch_spacing=process.hatch_spacing,
        beam_radius=process.beam_radius,
        absorption=process.absorption,
        preheat_temp=process.preheat_temp,
    )


def _analytic_base(geometry: PartGeometry, process: ProcessPlan,
                   p: Mapping[str, Any]) -> dict:
    """默认解析代理：由工艺无量纲数给出全部 8 项指标（可微、零颗粒开销）。"""
    mat = get_material(p.get("material", "316L"))
    ved = jnp.mean(jnp.atleast_1d(process.volumetric_energy_density()))
    led = jnp.mean(jnp.atleast_1d(process.linear_energy_density()))
    nh = jnp.mean(jnp.atleast_1d(process.normalized_enthalpy(
        rho=mat.rho_solid, cp=mat.cp_solid, T_melt=mat.T_melt,
        diffusivity=mat.diffusivity())))
    lt = jnp.ravel(jnp.asarray(process.layer_thickness))[0]
    h = jnp.ravel(jnp.asarray(process.hatch_spacing))[0]
    rb = jnp.ravel(jnp.asarray(process.beam_radius))[0]
    return analytic_powder_defects(
        ved=ved, led=led, norm_enthalpy=nh, layer_thickness=lt,
        hatch_spacing=h, beam_radius=rb, d50=float(p.get("d50", 30e-6)),
        packing_fraction=float(p.get("packing_fraction", 0.55)),
        dim=int(geometry.dim))


def _from_dict(d: Mapping[str, Any], *, dim: int, method: str) -> PowderBedResult:
    """把内核返回的 dict 打包成 ``PowderBedResult`` 契约。"""
    return PowderBedResult(
        packing_density=d["packing_density"],
        coordination_number=d["coordination_number"],
        surface_roughness=d["surface_roughness"],
        balling_indicator=d["balling_indicator"],
        lof_indicator=d["lof_indicator"],
        spatter_fraction=d["spatter_fraction"],
        denudation_width=d["denudation_width"],
        porosity=d["porosity"],
        dim=dim, method=method,
    )


# ---------------------------------------------------------------------------
# 求解器主体（四档分发）
# ---------------------------------------------------------------------------
def solve_powderbed(geometry: PartGeometry, process: ProcessPlan, *,
                    params: Mapping[str, Any] | None = None,
                    powder_fidelity: str = "surrogate") -> PowderBedResult:
    """粉末床求解：把颗粒尺度缺陷前驱解析成可微 ``PowderBedResult``。

    Parameters
    ----------
    geometry
        待制造零件（只读静态 bbox 元数据，不读 sdf 具体值）。
    process
        工艺方案；``laser_power`` / ``scan_speed`` / ``layer_thickness`` 等
        可微叶子，梯度照常回传。
    params
        ``powder_fidelity`` 选择器与 ``dem``/``sph``/``mpm`` 各档的细粒度
        超参（步数、接触刚度等）。
    powder_fidelity
        ``surrogate``（默认）| ``dem`` | ``sph`` | ``mpm``。

    设计要点
    --------
    显式颗粒档只解析其专属缺陷，熔融相位缺陷（其余项）由解析代理补全，
    保证契约完整、``defect_score`` 始终有意义。
    """
    p = dict(params or {})
    fid = p.get("powder_fidelity", powder_fidelity)
    if fid in ("powder.bed", None, "none"):
        fid = "surrogate"

    # (0) 默认解析代理：快、全可微、零颗粒开销
    if fid == "surrogate":
        return _from_dict(_analytic_base(geometry, process, p),
                          dim=int(geometry.dim), method="surrogate")

    cfg = _powder_config(geometry, process, p)

    # (1) DEM 铺粉：解析密实度 / 配位数 / 粗糙度，覆盖解析代理的铺粉项
    if fid == "dem":
        recoat = recoat_dem(cfg,
                            initial_positions=p.get("initial_positions"),
                            initial_radii=p.get("initial_radii"),
                            **p.get("dem", {}))
        base = _analytic_base(geometry, process, p)
        return PowderBedResult(
            packing_density=recoat["packing_density"],
            coordination_number=recoat["coordination_number"],
            surface_roughness=recoat["surface_roughness"],
            balling_indicator=base["balling_indicator"],
            lof_indicator=base["lof_indicator"],
            spatter_fraction=base["spatter_fraction"],
            denudation_width=base["denudation_width"],
            porosity=base["porosity"],
            dim=int(geometry.dim), method="dem")

    # (2) SPH 熔道：解析 Plateau–Rayleigh 球化，覆盖解析代理的球化项
    if fid == "sph":
        slm = _slm_from_process(process)
        res = melt_track_sph(cfg, slm, **p.get("sph", {}))
        base = _analytic_base(geometry, process, p)
        # 球化用 SPH 解析值；若 SPH 未熔化（melt_frac≈0）则回落到解析估计
        balling = jnp.where(res["melt_fraction"] > 0.02,
                            res["balling_indicator"], base["balling_indicator"])
        return PowderBedResult(
            packing_density=base["packing_density"],
            coordination_number=base["coordination_number"],
            surface_roughness=base["surface_roughness"],
            balling_indicator=balling,
            lof_indicator=base["lof_indicator"],
            spatter_fraction=base["spatter_fraction"],
            denudation_width=base["denudation_width"],
            porosity=base["porosity"],
            dim=int(geometry.dim), method="sph")

    # (3) MPM 再分布：解析剥蚀半宽 + 飞溅质量分数，覆盖解析代理对应项
    if fid == "mpm":
        slm = _slm_from_process(process)
        res = denudation_mpm(cfg, slm, **p.get("mpm", {}))
        base = _analytic_base(geometry, process, p)
        return PowderBedResult(
            packing_density=base["packing_density"],
            coordination_number=base["coordination_number"],
            surface_roughness=base["surface_roughness"],
            balling_indicator=base["balling_indicator"],
            lof_indicator=base["lof_indicator"],
            spatter_fraction=res["spatter_fraction"],
            denudation_width=res["denudation_width"],
            porosity=base["porosity"],
            dim=int(geometry.dim), method="mpm")

    # 未知档：回落到解析代理（不抛错，保证链路不崩）
    return _from_dict(_analytic_base(geometry, process, p),
                      dim=int(geometry.dim), method="surrogate")


@register_solver(
    "powder.bed",
    consumes=("PartGeometry", "ProcessPlan"),
    produces="PowderBedResult",
    stage="powderbed",
    modality=("SLM", "LSF"),
    differentiable=True,
    cost=20.0,
    defaults={"powder_fidelity": "surrogate"},
    doc=("粉末尺度缺陷求解（P2）。DEM/SPH/MPM 颗粒方法解析球化 / 未熔合 / "
         "飞溅 + 剥蚀，surrogate 档为默认可微解析代理。powder_fidelity: "
         "surrogate(快/可微) | dem(铺粉密实) | sph(球化) | mpm(剥蚀/飞溅)。"),
)
def _solve_powderbed_registered(geometry: PartGeometry, process: ProcessPlan, *,
                                params: Mapping[str, Any] | None = None
                                ) -> PowderBedResult:
    p = dict(params or {})
    return solve_powderbed(geometry, process, params=p,
                           powder_fidelity=p.get("powder_fidelity", "surrogate"))
