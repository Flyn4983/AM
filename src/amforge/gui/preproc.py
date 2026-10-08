"""AMForge 前处理 GUI 的**纯逻辑层**（无 Qt / 无 VTK 依赖）

把所有"前处理"的计算都收敛在这里，方便在无图形界面、无显示的 CI / 沙箱里
直接用 pytest 守住正确性。界面层 :mod:`amforge.gui.preproc_app` 只负责把这些
函数的结果画出来。

覆盖能力
--------
* 几何生成：解析 SDF 体素化（球/柱/方）与 STL 导入；
* 分层切片 + **hatch 光栅扫描路径**生成（每层每条扫描线取占位连续段为一段线段）；
* 工艺方案构造（ProcessPlan.uniform 封装，带 67° 层间旋转默认值）；
* 可打印性体检（悬垂/薄壁占比）+ 体/线能量密度（VED/LED）可行性窗口；
* 配置导出 / 导入（几何 SDF 存 npz，工艺参数存 json），可直接喂给
  ``amforge.inverse.simulate`` 做后续仿真。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import amforge.geometry as G
from amforge.counts import count_floor
from amforge.core.contracts import PartGeometry, ProcessPlan
from amforge.boundary import (
    BoundaryCondition, InitialCondition, BoundaryCollection,
    face_mask, export_bc, import_bc,
)
from amforge.powder import ParticleCollection

# ---------------------------------------------------------------------------
# 1. 几何生成
# ---------------------------------------------------------------------------
DEFAULT_SPACING_UM = 100.0


def build_primitive(kind: str, length_mm: float, *, spacing_um: float = DEFAULT_SPACING_UM,
                    name: str | None = None) -> PartGeometry:
    """解析 SDF 体素化生成基础几何。

    ``kind`` ∈ {"sphere","cylinder","box"}；``length_mm`` 为特征尺寸
    （球=直径，柱=直径，方=边长）。STL 单位换算（mm→m）由调用方负责。
    """
    R = 0.5 * length_mm * 1e-3
    if kind == "sphere":
        def sdf(p):
            return jnp.linalg.norm(p, axis=-1) - R
    elif kind == "cylinder":
        H = length_mm * 1e-3
        def sdf(p):
            r = jnp.sqrt(p[..., 0] ** 2 + p[..., 1] ** 2)
            return jnp.maximum(r - R, jnp.abs(p[..., 2]) - 0.5 * H)
    elif kind == "box":
        a = 0.5 * length_mm * 1e-3
        def sdf(p):
            return jnp.maximum(jnp.maximum(jnp.abs(p[..., 0]) - a,
                                           jnp.abs(p[..., 1]) - a),
                               jnp.abs(p[..., 2]) - a)
    else:
        raise ValueError(f"未知 primitive: {kind!r}（可选 sphere/cylinder/box）")

    b = (-R * 1.4, R * 1.4)
    bounds = [b, b, b]
    return G.from_sdf_fn(sdf, bounds=bounds, spacing=spacing_um * 1e-6,
                         name=name or kind)


def build_from_stl(path: str | Path, *, spacing_um: float = DEFAULT_SPACING_UM,
                  scale: float = 1e-3, name: str | None = None) -> PartGeometry:
    """STL（通常单位为 mm）→ PartGeometry；``scale=1e-3`` 把 mm 折算成 m。"""
    return G.from_stl(str(path), spacing=spacing_um * 1e-6, scale=scale,
                      name=name or Path(path).stem)


# ---------------------------------------------------------------------------
# 2. 分层切片 + hatch 光栅扫描路径
# ---------------------------------------------------------------------------
def _runs(row: np.ndarray) -> list[tuple[int, int]]:
    """返回布尔行里连续 True 段的 (i0, i1) 索引对（闭区间）。"""
    runs: list[tuple[int, int]] = []
    i = 0
    n = len(row)
    while i < n:
        if row[i]:
            j = i
            while j + 1 < n and row[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def generate_hatch_paths(part: PartGeometry, layer_thickness: float,
                         hatch_spacing: float, *, threshold: float = 0.5) -> dict:
    """对每层切片掩膜做 raster hatch，生成扫描路径。

    返回 dict：
      - ``z_heights`` (n_layers,) 各层中面高度 [m]；
      - ``masks`` (n_layers, nx, ny) 切片占位度；
      - ``paths`` list[list[((x0,y0),(x1,y1))]]，逐层逐段的物理坐标线段 [m]；
      - ``xs``/``ys`` 切片平面坐标数组；``layer_thickness``/``hatch_spacing``。

    策略：扫描线与 X 轴平行、沿 Y 按 hatch_spacing 推进；每层方向交替
    （bidirectional raster）。注：本预览层的 hatch 为轴对齐栅格；真正熔池
    求解器内部还会按 ``scan_angle`` 旋转，此处聚焦"几何→路径"的可视化桥梁。
    """
    masks, zs = G.slice_masks(part, layer_thickness)  # (n_layers, nx, ny)
    masks = np.asarray(masks)
    sp = float(part.spacing)
    xs = np.asarray(part.origin[0]) + sp * np.arange(masks.shape[1])  # (nx,)
    ys = np.asarray(part.origin[1]) + sp * np.arange(masks.shape[2])  # (ny,)
    occ = masks > threshold

    paths: list[list[tuple[tuple[float, float], tuple[float, float]]]] = []
    for l in range(masks.shape[0]):
        m = occ[l]  # (nx, ny) bool
        layer_paths: list[tuple[tuple[float, float], tuple[float, float]]] = []
        if m.any():
            # 该层在 y 方向有占位的区间 [y_min, y_max]
            row_occ = m.any(axis=0)
            y_min, y_max = float(ys[row_occ][0]), float(ys[row_occ][-1])
            # 扫描线按 hatch_spacing 均匀铺在 [y_min, y_max]（bidirectional raster）
            n_lines = max(1, count_floor((y_max - y_min) / hatch_spacing) + 1)
            ys_lines = np.linspace(y_min, y_max, n_lines)
            for yk in ys_lines:
                j = int(np.argmin(np.abs(ys - yk)))   # 最近邻 y 行采样
                row = m[:, j]
                if not row.any():
                    continue
                for i0, i1 in _runs(row):
                    layer_paths.append(((float(xs[i0]), float(yk)),
                                        (float(xs[i1]), float(yk))))
        paths.append(layer_paths)

    return {
        "z_heights": np.asarray(zs),
        "masks": masks,
        "paths": paths,
        "xs": xs,
        "ys": ys,
        "layer_thickness": float(layer_thickness),
        "hatch_spacing": float(hatch_spacing),
        "n_layers": int(masks.shape[0]),
    }


# ---------------------------------------------------------------------------
# 3. 工艺方案构造
# ---------------------------------------------------------------------------
def default_process_params() -> dict:
    """GUI 面板的工艺默认值（SLM 常规区间）。"""
    return dict(
        laser_power=200.0,        # W
        scan_speed=1.0,           # m/s
        layer_thickness=40e-6,    # m
        hatch_spacing=80e-6,      # m
        beam_radius=50e-6,        # m
        absorption=0.4,           # -
        preheat_temp=373.0,       # K
        rotation_per_layer_deg=67.0,
        modality="SLM",
    )


def build_process_plan(n_layers: int, *, laser_power: float, scan_speed: float,
                        layer_thickness: float, hatch_spacing: float,
                        beam_radius: float, absorption: float, preheat_temp: float,
                        rotation_per_layer_deg: float = 67.0,
                        modality: str = "SLM") -> ProcessPlan:
    """由面板数值构造 ProcessPlan（封装 ProcessPlan.uniform，含层间旋转）。"""
    return ProcessPlan.uniform(
        int(n_layers), modality=modality,
        laser_power=float(laser_power), scan_speed=float(scan_speed),
        layer_thickness=float(layer_thickness), hatch_spacing=float(hatch_spacing),
        beam_radius=float(beam_radius), absorption=float(absorption),
        preheat_temp=float(preheat_temp),
        rotation_per_layer=float(rotation_per_layer_deg) * math.pi / 180.0,
    )


# ---------------------------------------------------------------------------
# 4. 可打印性体检 + 能量密度可行性窗口
# ---------------------------------------------------------------------------
def assess(part: PartGeometry, plan: ProcessPlan) -> dict:
    """几何可打印性 + 工艺能量密度的快速体检（开工前用，不做重仿真）。"""
    lt = float(jnp.mean(jnp.atleast_1d(plan.layer_thickness)))
    pr = G.printability_report(part, layer_thickness=lt)
    ved = float(jnp.mean(plan.volumetric_energy_density()))   # J/m^3
    led = float(jnp.mean(plan.linear_energy_density()))       # J/m
    # SLM 典型 VED 窗口（J/m^3）：~3e10 .. 1.2e11（文献 30–120 J/mm^3）
    ved_lo, ved_hi = 3.0e10, 1.2e11
    if ved < ved_lo:
        ved_status = "低（能量不足→未熔合风险）"
    elif ved > ved_hi:
        ved_status = "高（能量过高→匙孔/球化风险）"
    else:
        ved_status = "OK（落在典型窗口）"
    return {
        "printability": pr,
        "VED_J_m3": ved,
        "LED_J_m": led,
        "VED_status": ved_status,
    }


# ---------------------------------------------------------------------------
# 4.5 边界 / 初值条件（缺口 #20-②A）
# ---------------------------------------------------------------------------
def default_boundary_collection(part: PartGeometry, *,
                                preheat_temp: float = 373.0) -> BoundaryCollection:
    """SLM 常规默认 BC/IC：基板(−Z)恒温预热、其余表面与环境对流、全场预热初值。

    物理含义：基板（构建底板）由外部温控维持在 ``preheat_temp``；顶面(+Z)与四壁
    (±X/±Y) 通过自然/强制对流与环境(293 K)换热；仿真初始全场即处于预热温度。
    该集合可直接喂 ``solve_enthalpy_thermal(params={"boundary_conditions": bc})``。
    """
    bcs = (
        BoundaryCondition("dirichlet", "-Z", float(preheat_temp), label="基板恒温"),
        BoundaryCondition("convection", "+Z", 15.0, 293.0, label="顶面散热"),
        BoundaryCondition("convection", "+X", 12.0, 293.0, label="侧壁对流"),
        BoundaryCondition("convection", "-X", 12.0, 293.0),
        BoundaryCondition("convection", "+Y", 12.0, 293.0),
        BoundaryCondition("convection", "-Y", 12.0, 293.0),
    )
    ic = InitialCondition("preheat", float(preheat_temp))
    return BoundaryCollection(bcs=bcs, ic=ic)


def make_boundary_collection(specs: list[dict],
                             ic: dict | None = None) -> BoundaryCollection:
    """由 GUI 提交的规格字典构造 :class:`BoundaryCollection`。

    ``specs`` 每项：``{"kind","face","value","value2?","label?"}``；
    ``ic``：``{"kind","value"}`` 或 None。
    """
    bcs = tuple(BoundaryCondition(**s) for s in specs)
    init = InitialCondition(**ic) if ic else None
    return BoundaryCollection(bcs=bcs, ic=init)


# ---------------------------------------------------------------------------
# 4.6 粉末 / 粒子导入（缺口 #20-②B）
# ---------------------------------------------------------------------------
def build_particle_collection_csv(path: str | Path, *,
                                  material: str = "316L") -> ParticleCollection:
    """由 CSV/JSON 点云文件构造 :class:`ParticleCollection`（真实导入）。"""
    p = Path(path)
    if p.suffix.lower() == ".json":
        return ParticleCollection.from_json(p)
    return ParticleCollection.from_csv(p, material=material)


def generate_powder_bed(part: PartGeometry, *, layer_thickness: float,
                        d50: float = 30e-6, psd_sigma: float = 0.25,
                        n_layers: int = 1, seed: int = 0,
                        material: str = "316L") -> ParticleCollection:
    """在零件构建区**真实生成**一层粉末床点云（抖动晶格 + 对数正态 PSD）。

    与 ``diffmech.synth_powder_bed`` 同源构造；生成的点云可直接喂 DEM 评估，
    也可导出 CSV/JSON 供后续复用。
    """
    return ParticleCollection.from_part_bed(
        part, layer_thickness=float(layer_thickness), d50=float(d50),
        psd_sigma=float(psd_sigma), n_layers=int(n_layers), seed=int(seed),
        material=material)


def powderbed_params_from_collection(collection: ParticleCollection, *,
                                     n_steps: int = 120) -> dict:
    """把导入/生成的粒子集合的真实统计映射为 ``powder.bed`` 求解参数。"""
    return collection.powder_bed_params(n_steps=n_steps)


def solve_powder_from_collection(collection: ParticleCollection,
                                 part: PartGeometry, plan, *,
                                 n_steps: int = 120,
                                 use_precise: bool = False):
    """用粒子集合的真实统计跑**真实 DEM 铺粉**求解，返回 ``PowderBedResult``。

    粒子点云（d50 / PSD / 材料密度）直接驱动 ``powder.bed`` 的 DEM 档，
    得到解出的密实度 / 配位数 / 粗糙度等真实缺陷前驱指标。``use_precise=True``
    时把导入/生成的精确粒子坐标作为 DEM 沉降初始态注入求解器（opt-in）。
    """
    return collection.solve_dem(part, plan, n_steps=n_steps, use_precise=use_precise)


# ---------------------------------------------------------------------------
# 4.7 支撑结构生成（模块A 前处理入口）
# ---------------------------------------------------------------------------
def generate_support(part: PartGeometry, *, kind: str = "block",
                    overhang_angle_deg: float = 45.0,
                    density: float = 1.0) -> dict:
    """前处理 GUI 入口：为零件生成支撑结构，并返回 2D 预览切片。

    仅做几何级规则布点（``support_auto`` 代理默认档），秒级完成；高保真
    ``support.simulate`` 走 ``select={'support':'support.simulate'}`` 另行仿真。

    返回 dict：``support``（:class:`SupportStructure`）、``support_mask_2d``
    （沿 z 投影的支撑足迹掩膜，供切片视图叠加预览）、``support_mask_3d``。
    """
    from amforge.support import support_auto

    sup = support_auto(
        geometry=part,
        params={"kind": kind, "overhang_angle_deg": overhang_angle_deg,
                "density": density},
    )
    mask3 = np.asarray(sup.support_mask)
    mask2d = (mask3.reshape(-1, mask3.shape[-1]).max(axis=-1)
              .reshape(mask3.shape[:-1]))  # 沿构建方向投影成足迹
    return {
        "support": sup,
        "support_mask_2d": mask2d,
        "support_mask_3d": mask3,
        "kind": kind,
    }


# ---------------------------------------------------------------------------
# 5. 配置导入 / 导出（几何 SDF + 工艺参数）
# ---------------------------------------------------------------------------
_PROCESS_FIELDS = (
    "laser_power", "scan_speed", "layer_thickness", "hatch_spacing",
    "beam_radius", "scan_angle", "absorption", "preheat_temp",
    "powder_feed_rate", "dwell_time",
)


def export_config(part: PartGeometry, plan: ProcessPlan, out_dir: str | Path) -> tuple[str, str]:
    """导出几何 + 工艺配置到目录：``preproc_geometry.npz`` + ``preproc_process.json``。

    返回 (几何文件路径, 工艺文件路径)，可直接被 :func:`import_config` 读回，
    或喂给 ``amforge.inverse.simulate(geometry, plan, ...)``。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    geom_path = out_dir / "preproc_geometry.npz"
    np.savez(
        geom_path,
        sdf=np.asarray(part.sdf),
        origin=np.asarray(part.origin),
        spacing=np.asarray(part.spacing),
        dim=np.asarray(part.dim),
        name=np.asarray(part.name if hasattr(part, "name") else "part"),
    )
    pdata = {f: np.asarray(getattr(plan, f)).tolist() for f in _PROCESS_FIELDS}
    pdata["modality"] = plan.modality
    plan_path = out_dir / "preproc_process.json"
    plan_path.write_text(json.dumps(pdata, indent=2), encoding="utf-8")
    return str(geom_path), str(plan_path)


def import_config(out_dir: str | Path) -> tuple[PartGeometry, ProcessPlan]:
    """读回 :func:`export_config` 导出的配置。"""
    out_dir = Path(out_dir)
    d = np.load(out_dir / "preproc_geometry.npz", allow_pickle=True)
    part = PartGeometry(
        sdf=jnp.asarray(d["sdf"]),
        origin=jnp.asarray(d["origin"]),
        spacing=float(d["spacing"]),
        dim=int(d["dim"]),
        name=str(d["name"]) if "name" in d.files else "imported",
    )
    pdata = json.loads((out_dir / "preproc_process.json").read_text(encoding="utf-8"))
    modality = pdata.pop("modality", "SLM")
    plan = ProcessPlan(
        modality=modality,
        **{k: jnp.asarray(v) for k, v in pdata.items()},
    )
    return part, plan
