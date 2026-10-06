"""边界/初值条件（BC/IC）测试（缺口 #20-②A）

守住核心算法：体素选面定位、集合序列化、焓法求解器对 BC/IC 的真实消费
（Dirichlet 基板恒温被保持、对流改变峰值、预热初值生效）。
全部为纯逻辑/数值测试，无 Qt/Vtk 依赖，可无显示运行。
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from amforge.geometry import PartGeometry
from amforge.core.contracts import ProcessPlan
from amforge.boundary import (
    BoundaryCondition, InitialCondition, BoundaryCollection, face_mask,
    initial_enthalpy_field,
)
from amforge.thermal_enthalpy import (
    solve_enthalpy_thermal, enthalpy_of_temperature,
)
from amforge.materials import get_material


def _box3d(spacing_um: float = 120.0, size_mm: float = 1.0) -> PartGeometry:
    from amforge.gui.preproc import build_primitive
    return build_primitive("box", length_mm=size_mm, spacing_um=spacing_um)


def _box2d(n: int = 13, spacing: float = 1e-4) -> PartGeometry:
    a = 0.5e-3
    xs = spacing * np.arange(n)
    ys = spacing * np.arange(n)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    sdf2 = np.maximum(np.abs(XX) - a, np.abs(YY) - a)
    return PartGeometry(
        sdf=jnp.asarray(sdf2), origin=jnp.asarray([0.0, 0.0]),
        spacing=spacing, dim=2, name="box2d",
    )


def _plan(n_layers: int = 4) -> ProcessPlan:
    return ProcessPlan.uniform(
        n_layers, modality="SLM", laser_power=200.0, scan_speed=1.0,
        layer_thickness=40e-6, hatch_spacing=80e-6, beam_radius=50e-6,
        absorption=0.4, preheat_temp=373.0,
    )


# ---------------------------------------------------------------------------
# 1. 选面定位（3D）
# ---------------------------------------------------------------------------
def test_face_mask_axis_locations_3d():
    part = _box3d()
    m_negz = np.asarray(face_mask(part, "-Z"))
    m_posx = np.asarray(face_mask(part, "+X"))
    m_all = np.asarray(face_mask(part, "all"))
    # 各轴面都应有实体表面体素
    assert m_negz.sum() > 0
    assert m_posx.sum() > 0
    assert m_all.sum() > m_negz.sum()
    # −Z 面应落在最低的「实体」z 层（盒子底面，非最低网格 z 层）
    zsum = m_all.reshape(-1, part.shape[-1]).sum(axis=0)
    z_solid_lo = int(np.argmax(zsum > 0))
    assert m_negz[..., z_solid_lo].sum() > 0
    # +X 面应落在最高的实体 x 层
    xsum = m_all.sum(axis=(1, 2))
    x_solid_hi = int(np.where(xsum > 0)[0].max())
    assert m_posx[x_solid_hi, ...].sum() > 0


# ---------------------------------------------------------------------------
# 2. 2D 几何下 ±Z 选面应自然为空（不存在 Z 向面）
# ---------------------------------------------------------------------------
def test_face_mask_z_empty_for_2d():
    part = _box2d()
    assert part.dim == 2
    assert np.asarray(face_mask(part, "+Z")).max() < 1e-6
    assert np.asarray(face_mask(part, "-Z")).max() < 1e-6
    # 但平面内面（±X）仍应非空
    assert np.asarray(face_mask(part, "+X")).sum() > 0


# ---------------------------------------------------------------------------
# 3. 集合序列化往返
# ---------------------------------------------------------------------------
def test_boundary_collection_roundtrip():
    bc = BoundaryCollection(
        bcs=(BoundaryCondition("dirichlet", "-Z", 500.0, label="基板"),
             BoundaryCondition("convection", "+Z", 15.0, 293.0)),
        ic=InitialCondition("preheat", 373.0),
    )
    bc2 = BoundaryCollection.from_dict(bc.to_dict())
    assert bc2 == bc


def test_boundary_condition_validation():
    with pytest.raises(ValueError):
        BoundaryCondition("bogus", "-Z", 1.0)
    with pytest.raises(ValueError):
        BoundaryCondition("convection", "-Z", 10.0, value2=0.0)  # 缺 T_inf


# ---------------------------------------------------------------------------
# 4. 求解器消费：Dirichlet 基板恒温被保持
# ---------------------------------------------------------------------------
def test_dirichlet_holds_base_temperature():
    # dx=80µm ≤ 2r=100µm：光斑可分辨；n_steps 交给求解器按 CFL 自动推导
    # （A0 之后 dt=曝光时长/n_steps，显式给小的步数会被判为发散而非"粗略近似"）。
    # 0.4mm 试片：ns=361，单次求解 ~1s。
    part = _box3d(size_mm=0.4, spacing_um=80.0)
    plan = _plan()
    bc = BoundaryCollection(bcs=(BoundaryCondition("dirichlet", "-Z", 600.0),),
                             ic=None)
    res = solve_enthalpy_thermal(
        geometry=part, process=plan,
        params={"boundary_conditions": bc, "material": "316L"})
    Tf = np.asarray(res.final_temperature)
    mb = np.asarray(face_mask(part, "-Z")) > 0.5
    assert np.isfinite(Tf[mb]).all()
    base_mean = float(np.mean(Tf[mb]))
    # 实测：逐体素每步强制覆盖 → 该面均值 = 600.00K（不是"落在 530~670 区间"）
    assert abs(base_mean - 600.0) < 2.0, f"Dirichlet 面均值={base_mean}"

    # 无 BC 时同面由激光剂量决定（实测 1955.9K，比基板恒温高 1356K）。
    # 守护的是「BC 真被消费」这一事实：符号取决于剂量，不预设"更低"。
    res0 = solve_enthalpy_thermal(
        geometry=part, process=plan, params={"material": "316L"})
    Tf0 = np.asarray(res0.final_temperature)
    natural = float(np.mean(Tf0[mb]))
    assert abs(natural - 600.0) > 20.0, f"无 BC 时该面={natural}，BC 未生效？"


# ---------------------------------------------------------------------------
# 5. 求解器消费：对流确实移走能量（**实测改写的判据**，见下）
# ---------------------------------------------------------------------------
def test_convection_removes_energy():
    """强对流降低末态温度；只有极强对流才降低**峰值**。

    A0 前的旧断言「h=200 W/m²K 使峰值下降」是**假**的：对流是表面通量，必须
    除以 dx 摊成体积项才与 ∂H/∂t 同量纲（旧代码少除 dx，散热被低估 1/dx≈12500
    倍）；修正后实测（0.4mm 试片 dx=80µm，5 面对流，T_inf=293K）：

        h          末态实体均值Δ    峰值Δ
        200        −8.1 K          +0.03%
        2000       −81.6 K         +0.10%
        20000      −573.7 K        −5.39%

    物理：峰值由「脉冲剂量 + 蒸发封顶」在光斑中心局部决定，表面对流在正常工艺
    量级（h≲10³）动不了它；但它扫完之后整体带走能量，末态温度显著下降。所以
    守护点应是「能量被移走」，而不是「峰值被压」。
    """
    part = _box3d(size_mm=0.4, spacing_um=80.0)
    plan = _plan()
    solid = np.asarray(part.sdf) < 0.0
    res0 = solve_enthalpy_thermal(geometry=part, process=plan,
                                  params={"material": "316L"})
    mean0 = float(np.mean(np.asarray(res0.final_temperature)[solid]))
    peak0 = float(np.max(np.asarray(res0.peak_temperature)))

    means, peaks = [], []
    for h in (200.0, 2000.0, 20000.0):
        strong = BoundaryCollection(
            bcs=tuple(BoundaryCondition("convection", f, h, 293.0)
                      for f in ("+Z", "+X", "-X", "+Y", "-Y")), ic=None)
        res = solve_enthalpy_thermal(
            geometry=part, process=plan,
            params={"boundary_conditions": strong, "material": "316L"})
        Tf = np.asarray(res.final_temperature)
        assert np.isfinite(Tf).all() and np.isfinite(res.peak_temperature).all()
        means.append(float(np.mean(Tf[solid])))
        peaks.append(float(np.max(np.asarray(res.peak_temperature))))

    # ① 单调散热：h 越大末态越冷，且跨度可测（实测 2291→2283→2210→1718 K）
    assert means == sorted(means, reverse=True), f"末态均值未随 h 单调下降：{means}"
    assert mean0 - means[1] > 30.0, f"h=2000 仅降 {mean0 - means[1]:.1f}K（散热未生效？）"
    assert mean0 - means[2] > 300.0, f"h=20000 仅降 {mean0 - means[2]:.1f}K"
    # ② 常规工艺量级对流动不了峰值（蒸发封顶/剂量决定），极强对流才能压峰
    assert abs(peaks[1] - peak0) / peak0 < 0.02, "h=2000 不应显著改变峰值"
    assert peaks[2] < peak0 - 0.02 * peak0, f"h=20000 应压低峰值：{peaks[2]} vs {peak0}"


# ---------------------------------------------------------------------------
# 6. 初值条件：预热初值场焓高于环境温度
# ---------------------------------------------------------------------------
def test_initial_condition_preheat():
    part = _box3d()
    mat = get_material("316L")
    ic_bc = BoundaryCollection(bcs=(), ic=InitialCondition("preheat", 500.0))
    H0 = initial_enthalpy_field(
        part, ic_bc, rho=mat.rho_solid, cp=mat.cp_solid, L=mat.latent_fusion,
        T_amb=mat.T_ambient, T_sol=mat.T_solidus, T_liq=mat.T_liquidus)
    H_amb = enthalpy_of_temperature(
        jnp.asarray(mat.T_ambient), rho=mat.rho_solid, cp=mat.cp_solid,
        L=mat.latent_fusion, T_amb=mat.T_ambient, T_sol=mat.T_solidus,
        T_liq=mat.T_liquidus)
    assert float(np.mean(H0)) > float(H_amb)
