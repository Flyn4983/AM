"""AMForge 后处理 / 数字样机 / 可微优化 的**纯逻辑层**（无 Qt / 无 VTK 依赖）

把"仿真结果 → 可解释/可导出/可优化"的所有计算收敛在这里，方便在无图形界面、
无显示的 CI / 沙箱里直接用 pytest 守住正确性。界面层
:mod:`amforge.gui.postproc_app` 只负责把这些函数的结果画出来。

覆盖能力
--------
* **场量抽取（field extraction）**：从 :func:`amforge.inverse.simulate` 的真实
  端到端结果中，把温度 / 残余应力(vM) / 位移 / 微观组织 / 本构 等体素场抽取成
  统一的 :class:`VolumeField`（带 origin / spacing / dim 坐标信息，便于 VTK 对齐）；
* **导出**：每个体素场可导出为 ParaView 可读的 legacy ``.vtk``
  (STRUCTURED_POINTS, ASCII) 与 ``.csv``（x,y,z,value），并产出 ``manifest.json``
  （各场 min/max/mean，供后处理/可视化直接取范围）；
* **数字样机装配摘要**：从 ``AssemblyResult`` + ``ServiceVerdict`` 抽取关节载荷/相对
  位移/关节刚度/稳定性裕度/装配评分，并生成铰接链坐标供 UI 画机构；
* **可微优化聚合**：从 ``optimize_dimensional / optimize_shape /
  optimize_geometry_process / pareto_optimize_geometry_process`` 的结果中抽取
  loss 历史、参数轨迹、NSGA-II Pareto 前沿与 utopia / best_compromise。

所有函数只依赖 ``numpy``（VTK 文件用纯 numpy 手写 legacy 格式，无需 vtk 运行时），
因此既能被 UI 层调用，也能在没装 PySide6/vtk 的机器上被 pytest 守住。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from amforge.core.contracts import PartGeometry


# ===========================================================================
# 1. 体素场（带坐标信息的统一载体）
# ===========================================================================
@dataclass
class VolumeField:
    """一个对齐到几何栅格的标量体素场（温度 / 应力 / 微观 …）。

    ``data`` 形状与几何一致 ``(nx, ny, nz)`` 或 2D ``(nx, ny)``；坐标由
    ``origin`` / ``spacing`` / ``dim`` 给出，便于直接写 VTK / 画图。
    """

    name: str
    label: str
    unit: str
    data: np.ndarray
    origin: np.ndarray
    spacing: float
    dim: int
    cmap: str = "viridis"

    def min(self) -> float:
        return float(np.nanmin(self.data))

    def max(self) -> float:
        return float(np.nanmax(self.data))

    def mean(self) -> float:
        return float(np.nanmean(self.data))


# ---------------------------------------------------------------------------
# 2. 场量注册表（contract 名 → 抽取规则）
# ---------------------------------------------------------------------------
@dataclass
class _FieldSpec:
    contract: str
    attr: str | None          # 直接属性名；为 None 时用 fn
    fn: Callable[[Mapping[str, Any]], np.ndarray] | None
    label: str
    unit: str
    cmap: str


def _vm_stress(out: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(out["asbuilt"].von_mises_residual())


def _disp_mag(out: Mapping[str, Any]) -> np.ndarray:
    d = np.asarray(out["asbuilt"].displacement, dtype=np.float64)
    return np.linalg.norm(d, axis=-1)


def _strain_mag(out: Mapping[str, Any]) -> np.ndarray:
    s = np.asarray(out["asbuilt"].residual_strain, dtype=np.float64)
    return np.sqrt(np.sum(s ** 2, axis=-1) + 1e-30)


def _relaxed_vm(out: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(out["secondary"].part.von_mises_residual(), dtype=np.float64)


def _eff_stiff(out: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(out["constitutive"].effective_stiffness())


def _aniso(out: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(out["constitutive"].anisotropy_ratio())


FIELD_REGISTRY: dict[str, _FieldSpec] = {
    # —— 热历史（thermal）——
    "peak_temperature": _FieldSpec("thermal", "peak_temperature", None,
                                   "峰值温度", "K", "inferno"),
    "final_temperature": _FieldSpec("thermal", "final_temperature", None,
                                    "终场温度", "K", "inferno"),
    "cooling_rate": _FieldSpec("thermal", "cooling_rate", None,
                               "冷却速率", "K/s", "viridis"),
    "thermal_gradient": _FieldSpec("thermal", "thermal_gradient", None,
                                   "温度梯度 G", "K/m", "viridis"),
    "solidification_rate": _FieldSpec("thermal", "solidification_rate", None,
                                      "凝固速率 R", "m/s", "plasma"),
    # —— 成形 / 残余（asbuilt）——
    "von_mises_residual": _FieldSpec("asbuilt", None, _vm_stress,
                                     "残余应力(vM)", "Pa", "magma"),
    "displacement_mag": _FieldSpec("asbuilt", None, _disp_mag,
                                   "位移模", "m", "viridis"),
    "residual_strain_mag": _FieldSpec("asbuilt", None, _strain_mag,
                                      "残余应变", "-", "viridis"),
    # —— 二次工艺（secondary）——
    "relaxed_residual": _FieldSpec("secondary", None, _relaxed_vm,
                                   "松弛后残余应力(vM)", "Pa", "magma"),
    # —— 微观组织（micro）——
    "grain_size": _FieldSpec("micro", "grain_size", None,
                             "晶粒尺寸", "m", "viridis"),
    "columnar_fraction": _FieldSpec("micro", "columnar_fraction", None,
                                    "柱状晶分数", "-", "coolwarm"),
    "porosity_micro": _FieldSpec("micro", "porosity", None,
                                 "孔隙率(微观)", "-", "coolwarm"),
    "orientation": _FieldSpec("micro", "orientation", None,
                              "晶粒取向", "rad", "hsv"),
    # —— 本构（constitutive）——
    "E_eff": _FieldSpec("constitutive", None, _eff_stiff,
                        "等效模量", "Pa", "viridis"),
    "anisotropy_ratio": _FieldSpec("constitutive", None, _aniso,
                                   "各向异性比", "-", "coolwarm"),
    "sigma_y0": _FieldSpec("constitutive", "sigma_y0", None,
                           "初始屈服强度", "Pa", "viridis"),
    "porosity_constitutive": _FieldSpec("constitutive", "porosity", None,
                                        "孔隙率(本构)", "-", "coolwarm"),
}


def available_fields(out: Mapping[str, Any]) -> list[str]:
    """返回 ``out`` 中实际存在的场名（无论体素/标量，对应契约缺失则跳过）。"""
    names: list[str] = []
    for name, spec in FIELD_REGISTRY.items():
        if spec.contract in out:
            names.append(name)
    return names


def _extract_array(out: Mapping[str, Any], name: str) -> np.ndarray:
    """按注册表规则抽取原始数组（不检查形状）。"""
    spec = FIELD_REGISTRY[name]
    if spec.fn is not None:
        return np.asarray(spec.fn(out), dtype=np.float64)
    return np.asarray(getattr(out[spec.contract], spec.attr), dtype=np.float64)


def grid_fields(out: Mapping[str, Any], geometry: PartGeometry) -> list[str]:
    """返回 ``out`` 中**体素形状**（与几何栅格同形）的场名。

    部分契约字段是标量（如 ``solidification_rate`` 用熔池标量宽/深计算），
    无法作为体素场对齐到几何栅格，这类会被排除（它们由 :func:`scalar_summary` 汇总）。
    """
    gshape = tuple(geometry.shape)
    out_list: list[str] = []
    for name in available_fields(out):
        try:
            a = _extract_array(out, name)
        except Exception:
            continue
        if a.shape == gshape:
            out_list.append(name)
    return out_list


def get_field(out: Mapping[str, Any], name: str,
             geometry: PartGeometry) -> VolumeField:
    """从一次 ``simulate`` 结果中抽取名为 ``name`` 的体素场。

    坐标统一用 ``geometry`` 的 origin / spacing / dim（thermal/asbuilt/micro/
    constitutive 都在同一几何栅格上），因此写出的 VTK/CSV 能正确对齐。
    """
    if name not in FIELD_REGISTRY:
        raise KeyError(f"未知体素场 {name!r}，可选：{sorted(FIELD_REGISTRY)}")
    spec = FIELD_REGISTRY[name]
    if spec.contract not in out:
        raise KeyError(f"结果中无契约 {spec.contract!r}，无法抽取 {name!r}")
    if spec.fn is not None:
        data = np.asarray(spec.fn(out), dtype=np.float64)
    else:
        data = np.asarray(getattr(out[spec.contract], spec.attr), dtype=np.float64)
    if data.shape != tuple(geometry.shape):
        raise KeyError(
            f"场 {name!r} 在结果中为标量/非体素形状 {data.shape}，无法作为体素场 "
            f"对齐到几何 {tuple(geometry.shape)}；该量已纳入 scalar_summary。")
    return VolumeField(
        name=name, label=spec.label, unit=spec.unit, data=data,
        origin=np.asarray(geometry.origin, dtype=np.float64),
        spacing=float(geometry.spacing), dim=int(geometry.dim),
        cmap=spec.cmap,
    )


# ===========================================================================
# 3. 标量摘要（非体素指标）
# ===========================================================================
def scalar_summary(out: Mapping[str, Any]) -> dict:
    """把非体素的标量指标汇总成 dict（供 UI 摘要面板 / 报表）。"""
    s: dict[str, Any] = {}
    if "verdict" in out:
        v = out["verdict"]
        s["strength_safety_factor"] = float(v.strength_safety_factor)
        s["stiffness_ratio"] = float(v.stiffness_ratio)
        s["fatigue_life_cycles"] = float(v.fatigue_life_cycles)
        s["wear_depth_m"] = float(v.wear_depth)
        s["passed"] = bool(float(v.passed) > 0.5)
        s["margin_report"] = dict(v.margin_report)
    if "assembly" in out:
        a = out["assembly"]
        s["stability_margin"] = float(a.stability_margin)
        s["assembly_score"] = float(a.assembly_score())
        s["assembly_porosity"] = float(a.porosity)
        s["n_joints"] = int(np.asarray(a.relative_displacement).shape[0])
    if "thermal" in out:
        try:
            s["solidification_rate"] = float(out["thermal"].solidification_rate)
        except Exception:
            pass
    if "meltpool" in out:
        s["meltpool_defect_score"] = float(out["meltpool"].defect_score())
    if "powderbed" in out:
        pb = out["powderbed"]
        s["powder_defect_score"] = float(pb.defect_score())
        s["powder_recoat_quality"] = float(pb.recoat_quality())
    return s


# ===========================================================================
# 4. 导出：ParaView 可读 .vtk（legacy STRUCTURED_POINTS, ASCII）+ .csv + manifest
# ===========================================================================
def write_vtk_structured_points(field: VolumeField, path) -> str:
    """把体素场写成 ParaView 可直接打开的 legacy ``.vtk``（ASCII）。

    纯 numpy 手写，无需 vtk 运行时；坐标按 VTK 约定以 x 最快（Fortran 序）铺排。
    2D 场用 ``DIMENSIONS nx ny 1`` + ``SPACING s s 1`` 表达，ParaView 可正常读。
    """
    data = np.asarray(field.data, dtype=np.float32)
    sh = data.shape
    ndim = data.ndim
    nx, ny = int(sh[0]), int(sh[1])
    nz = int(sh[2]) if ndim == 3 else 1
    sp = float(field.spacing)
    origin = np.asarray(field.origin, dtype=np.float64)
    ox = float(origin[0]); oy = float(origin[1])
    oz = float(origin[2]) if origin.shape[0] > 2 else 0.0

    # VTK STRUCTURED_POINTS：点序为 x 最快 → numpy column-major (order='F')
    vals = data.astype(np.float32).ravel(order="F")

    lines: list[str] = []
    lines.append("# vtk DataFile Version 3.0")
    lines.append(field.label)
    lines.append("ASCII")
    lines.append("DATASET STRUCTURED_POINTS")
    lines.append(f"DIMENSIONS {nx} {ny} {nz}")
    lines.append(f"ORIGIN {ox:.12g} {oy:.12g} {oz:.12g}")
    lines.append(f"SPACING {sp:.12g} {sp:.12g} {sp:.12g}")
    npts = nx * ny * nz
    lines.append(f"POINT_DATA {npts}")
    lines.append(f"SCALARS {field.name} float 1")
    lines.append("LOOKUP_TABLE default")
    flat = " ".join(f"{v:.6g}" for v in vals)
    lines.append(flat)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def write_field_csv(field: VolumeField, path) -> str:
    """把体素场写成 ``x,y,z,value``（2D 为 ``x,y,value``）的 CSV，便于表格/绘图。"""
    data = np.asarray(field.data, dtype=np.float64)
    ndim = data.ndim
    ax = [field.origin[i] + field.spacing * np.arange(data.shape[i])
          for i in range(ndim)]
    grids = np.meshgrid(*ax, indexing="ij")
    cols = [g.ravel() for g in grids] + [data.ravel()]
    arr = np.stack(cols, axis=-1)
    header = "x y z value" if ndim == 3 else "x y value"
    np.savetxt(path, arr, delimiter=",", header=header, comments="", fmt="%.8g")
    return str(path)


def export_all_fields(out: Mapping[str, Any], geometry: PartGeometry,
                      out_dir, *, fields: Sequence[str] | None = None,
                      formats: Sequence[str] = ("vtk", "csv")) -> dict:
    """把一批体素场导出到 ``out_dir``，返回 manifest（含每个场的 min/max/mean/路径）。

    ``fields=None`` 时导出 :func:`available_fields` 给出的全部体素场。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(fields) if fields is not None else grid_fields(out, geometry)
    manifest: dict[str, Any] = {"fields": {}, "count": 0}
    for name in names:
        try:
            fld = get_field(out, name, geometry)
        except KeyError:
            continue
        entry: dict[str, Any] = {
            "label": fld.label, "unit": fld.unit, "cmap": fld.cmap,
            "dim": fld.dim, "shape": list(fld.data.shape),
            "min": fld.min(), "max": fld.max(), "mean": fld.mean(),
            "paths": {},
        }
        if "vtk" in formats:
            p = out_dir / f"{name}.vtk"
            entry["paths"]["vtk"] = write_vtk_structured_points(fld, p)
        if "csv" in formats:
            p = out_dir / f"{name}.csv"
            entry["paths"]["csv"] = write_field_csv(fld, p)
        manifest["fields"][name] = entry
        manifest["count"] += 1
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


# ===========================================================================
# 5. 数字样机装配摘要 + 铰接链坐标
# ===========================================================================
def assembly_summary(out: Mapping[str, Any]) -> dict:
    """从 ``AssemblyResult`` + ``ServiceVerdict`` 抽取装配体关键指标。"""
    s = scalar_summary(out)
    if "assembly" not in out:
        return s
    a = out["assembly"]
    s["joint_load"] = np.asarray(a.joint_load, dtype=np.float64).tolist()
    s["relative_displacement"] = np.asarray(
        a.relative_displacement, dtype=np.float64).tolist()
    s["flexural_stiffness"] = np.asarray(
        a.flexural_stiffness, dtype=np.float64).tolist()
    return s


def assembly_chain(out: Mapping[str, Any], geometry: PartGeometry) -> dict:
    """生成供 UI 画机构用的铰接链坐标。

    把 n_joints 个关节沿构建方向 (z) 均匀分布在零件包围盒内、零件 x/y 中心处；
    返回关节坐标 ``positions (n,3)`` 与相邻段的 ``links`` 索引对。
    """
    a = out.get("assembly")
    if a is None:
        return {"positions": np.zeros((0, 3)), "links": [],
                "relative_displacement": [], "flexural_stiffness": []}
    n = int(np.asarray(a.relative_displacement).shape[0])
    lo, hi = geometry.bbox()
    cx = float((lo[0] + hi[0]) / 2.0)
    cy = float((lo[1] + hi[1]) / 2.0)
    z0, z1 = float(lo[-1]), float(hi[-1])
    zs = np.linspace(z0, z1, max(n, 1))
    positions = np.stack([np.full(n, cx), np.full(n, cy), zs], axis=-1)
    links = [[i, i + 1] for i in range(n - 1)]
    return {
        "positions": positions,
        "links": links,
        "relative_displacement": np.asarray(
            a.relative_displacement, dtype=np.float64).tolist(),
        "flexural_stiffness": np.asarray(
            a.flexural_stiffness, dtype=np.float64).tolist(),
    }


# ===========================================================================
# 6. 可微优化结果聚合
# ===========================================================================
def optimization_summary(result: Mapping[str, Any], kind: str) -> dict:
    """从优化器返回 dict 中抽取 UI/报表所需的聚合量。

    ``kind`` ∈ {"dimensional","shape","joint","pareto"}。
    """
    if kind == "pareto":
        g = np.asarray(result["geom_dev"], dtype=np.float64)
        s = np.asarray(result["stress"], dtype=np.float64)
        feas = np.asarray(result["feasible"], dtype=np.float64)
        pen = np.asarray(result.get("constraint_penalty", []), dtype=np.float64)
        best = int(result.get("best_compromise_index", -1))
        agg: dict[str, Any] = {
            "algorithm": result.get("algorithm"),
            "pop_size": result.get("pop_size"),
            "n_gen": result.get("n_gen"),
            "geom_dev": g, "stress": s, "feasible": feas,
            "constraint_penalty": pen,
            "utopia": result.get("utopia"),
            "best_compromise_index": best,
            "n_front": int(g.shape[0]),
        }
        if "population_geom_dev" in result:
            agg["population_geom_dev"] = np.asarray(
                result["population_geom_dev"], dtype=np.float64)
            agg["population_stress"] = np.asarray(
                result["population_stress"], dtype=np.float64)
        return agg

    agg: dict[str, Any] = {
        "loss_history": np.asarray(result["loss_history"], dtype=np.float64),
    }
    for key in ("geom_history", "stress_history", "z_history",
                "u_history", "constraint_penalty_history"):
        if key in result:
            agg[key] = result[key]
    return agg


# ===========================================================================
# 6.5 二次工艺（模块B）纯逻辑入口
# ===========================================================================
def apply_secondary(asbuilt, *, treatment: str = "HT", geometry=None,
                    params: Mapping[str, Any] | None = None) -> dict:
    """对成形件施加二次工艺（HT / HIP / machining / none），返回结果与可视化场。

    纯逻辑、无 Qt / 无 VTK，可在 CI / 沙箱直接 pytest。界面层「二次工艺」卡片
    在后台线程调用它，把松弛后残余应力云图叠加到场浏览器。

    Parameters
    ----------
    asbuilt : AsBuiltPart
        制造后成形件（来自 ``out['asbuilt']``）。
    treatment : str
        ``"HT"``（去应力退火）/ ``"HIP"``（热等静压）/ ``"machining"``（切削）/
        ``"none"``（直通）。
    geometry : PartGeometry | None
        machining 所需名义几何（作修形参考），其余工艺可省略。
    params : dict | None
        透传给各二次工艺求解器的参数（温度/压力/保温时间/初始孔隙率等）。

    Returns
    -------
    dict
        ``{"secondary": SecondaryProcessResult, "relaxed_vm": np.ndarray,
        "treatment": str, "density_after": float, "relief_factor": float}``。
        ``relaxed_vm`` 为处理后残余应力 von Mises 场（与几何同形），供云图叠加。
    """
    from amforge.postprocess_secondary import (
        postprocess_passthrough, ht_relax, hip_densify, subtractive,
    )
    p = dict(params or {})
    t = str(treatment).lower()
    if t == "ht":
        res = ht_relax(asbuilt=asbuilt, params=p)
    elif t == "hip":
        res = hip_densify(asbuilt=asbuilt, params=p)
    elif t == "machining":
        if geometry is None:
            raise ValueError("machining 需要 geometry（名义几何）作修形参考")
        res = subtractive(asbuilt=asbuilt, geometry=geometry, params=p)
    elif t in ("none", "passthrough"):
        res = postprocess_passthrough(asbuilt=asbuilt, params=p)
    else:
        raise ValueError(f"未知二次工艺: {treatment!r}（可选 HT/HIP/machining/none）")
    relaxed_vm = np.asarray(res.part.von_mises_residual(), dtype=np.float64)
    return {
        "secondary": res,
        "relaxed_vm": relaxed_vm,
        "treatment": res.treatment,
        "density_after": float(res.density()),
        "relief_factor": float(res.relief_ratio()),
    }


# ===========================================================================
# 7. 端到端仿真入口（UI 调用；重计算委托给 inverse.simulate）
# ===========================================================================
def run_simulation(geometry: PartGeometry, plan, *,
                   material: str = "316L",
                   asbuilt_solver: str = "buildup",
                   micro_solver: str = "surrogate",
                   mbd_solver: str = "surrogate",
                   powder_solver: str | None = None,
                   constitutive: str = "j2",
                   params: Mapping[str, Any] | None = None) -> dict:
    """跑完整的物理链，返回 ``simulate`` 的输出 dict。

    这是 UI「运行仿真」按钮在纯逻辑层的落点；界面层负责在后台线程调用它，
    避免阻塞 GUI 主线程。
    """
    from amforge.inverse import simulate  # 懒加载，避免无 GUI 测试时重 import
    out = simulate(
        geometry, plan, material=material,
        asbuilt_solver=asbuilt_solver, micro_solver=micro_solver,
        mbd_solver=mbd_solver, powder_solver=powder_solver,
        constitutive=constitutive, params=params)
    return out


# ===========================================================================
# 8. 扫描动画：帧抽取 + GIF / ParaView 时间序列导出
# ===========================================================================
def extract_temperature_frames(out: Mapping[str, Any],
                               geometry: PartGeometry) -> list[VolumeField]:
    """从 ``simulate`` 结果抽取扫描过程温度演化帧序列（需求解器 ``record_frames=True``）。

    返回每帧一个 :class:`VolumeField`（与几何同形、坐标对齐），供 GUI 播放或导出。
    未开启帧记录（``out['thermal'].temperature_evolution is None``）时返回空列表。
    """
    th = out.get("thermal") if isinstance(out, dict) else out
    if th is None or getattr(th, "temperature_evolution", None) is None:
        return []
    evo = np.asarray(th.temperature_evolution, dtype=np.float64)  # (n, *gshape)
    gshape = tuple(geometry.shape)
    frames: list[VolumeField] = []
    for i in range(evo.shape[0]):
        data = evo[i]
        if data.shape != gshape:
            continue  # 防御：演化帧应与几何同形
        frames.append(VolumeField(
            name=f"temperature_t{i:03d}", label=f"温度帧 {i}", unit="K",
            data=data, origin=np.asarray(geometry.origin, dtype=np.float64),
            spacing=float(geometry.spacing), dim=int(geometry.dim),
            cmap="inferno"))
    return frames


def save_field_gif(frames: Sequence[VolumeField], path, *, fps: int = 8,
                   cmap: str = "inferno", vmin: float | None = None,
                   vmax: float | None = None) -> str:
    """把温度演化帧序列写成 GIF（matplotlib + Pillow 渲染，纯 numpy，无需 vtk）。

    ``frames`` 为 :func:`extract_temperature_frames` 返回的 :class:`VolumeField` 列表。
    惰性 import matplotlib，保持本模块默认轻量。
    """
    if not frames:
        raise ValueError("无动画帧可导出（需求解器 record_frames=True）")
    import matplotlib  # 惰性：避免拖慢无 GUI 的纯逻辑测试
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    f0 = frames[0]
    origin = np.asarray(f0.origin, dtype=np.float64)
    sp = float(f0.spacing)
    ext = [float(origin[0]), float(origin[0]) + sp * (f0.data.shape[0] - 1),
           float(origin[1]), float(origin[1]) + sp * (f0.data.shape[1] - 1)]
    if vmin is None:
        vmin = min(float(f.data.min()) for f in frames)
    if vmax is None:
        vmax = max(float(f.data.max()) for f in frames)
    fig = plt.figure(figsize=(4.2, 4.2))

    def _draw(i):
        fig.clear()
        ax = fig.add_subplot(111)
        d = frames[i].data
        if d.ndim == 3:                      # 取构建方向中间切片出 2D 云图
            k = d.shape[-1] // 2
            sl = d[..., k]
        else:
            sl = d
        im = ax.imshow(sl.T, origin="lower", extent=ext,
                       cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax)
        ax.set_title(f"扫描温度演化 帧 {i}/{len(frames) - 1}")
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="K")
        return (im,)

    ani = FuncAnimation(fig, _draw, frames=len(frames),
                        interval=1000.0 / max(1, fps), blit=False)
    path = str(path)
    if not path.lower().endswith(".gif"):
        path += ".gif"
    ani.save(path, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    return path


def save_field_pvd(frames: Sequence[VolumeField], path, *,
                   geometry: PartGeometry | None = None) -> str:
    """把温度演化帧写成 ParaView 可读的 ``.pvd`` 时间序列（每帧一个 legacy ``.vtk``）。

    纯 numpy 手写，无需 vtk 运行时；``<stem>.pvd`` 用 collection timestep 串起各帧，
    在 ParaView 中可时间轴播放。返回 ``.pvd`` 路径。
    """
    if not frames:
        raise ValueError("无动画帧可导出（需求解器 record_frames=True）")
    path = Path(path)
    if path.suffix.lower() != ".pvd":
        path = path.with_suffix(".pvd")
    stem = path.stem
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    vtk_names: list[str] = []
    for i, fld in enumerate(frames):
        vp = parent / f"{stem}_{i:04d}.vtk"
        write_vtk_structured_points(fld, vp)
        vtk_names.append(vp.name)
    lines = ['<?xml version="1.0"?>',
             '<VTKFile type="Collection" version="1.0" '
             'byte_order="LittleEndian">',
             '  <Collection>']
    for i, vp in enumerate(vtk_names):
        lines.append(f'    <DataSet timestep="{i}" group="" '
                     f'part="0" file="{vp}"/>')
    lines.append('  </Collection>')
    lines.append('</VTKFile>')
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


__all__ = [
    "VolumeField", "FIELD_REGISTRY", "available_fields", "get_field",
    "scalar_summary", "write_vtk_structured_points", "write_field_csv",
    "export_all_fields", "assembly_summary", "assembly_chain",
    "optimization_summary", "run_simulation", "apply_secondary",
    "extract_temperature_frames", "save_field_gif", "save_field_pvd",
]
