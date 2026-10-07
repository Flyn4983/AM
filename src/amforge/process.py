"""工艺规划模块 (Process Planning)
=================================

职责：产出 :class:`~amforge.core.contracts.ProcessPlan` 契约，
即"给定几何 → 用什么工艺去造它"。

本模块提供三类工艺生产者，覆盖从最简到最智能的谱系：

===========================  ========================================
求解器                        用途
===========================  ========================================
``process.constant``          直接把 ``params`` 里的标量摊成工艺方案。
                              最轻量的契约生产者，用于跑通流水线、
                              做参数扫描 (``jax.vmap``) 与梯度优化。
``process.heuristic``         **几何感知**的规则式工艺规划：
                              按悬垂率/薄壁/层数自动调功率、间距、
                              层间旋转，并用无量纲焓把工艺锁在
                              传导模式工艺窗口内。
``process.neural``            3D-CNN 几何编码器 → 工艺参数
                              （在 :mod:`amforge.inverse` 中注册，
                              功能 2 的可微闭环主角）。
===========================  ========================================

另外提供**扫描路径**生成（zigzag / 轮廓 / 螺旋 / G-code 解析）与
**工艺窗口**筛选工具 —— 前者喂给熔池求解器，后者用于可行性预筛。

可微性
------
所有函数对工艺参数（功率、速度、间距、层厚）与几何 SDF 均可微。
规则式规划里的"判断"全部用 ``jnp.where`` / ``sigmoid`` 软化，
因此 ``process.heuristic`` 也能作为可微闭环的一部分被梯度穿透。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from amforge.core.contracts import PartGeometry, ProcessPlan
from amforge.core.registry import register_solver
from amforge.geometry import overhang_fraction, thin_wall_field
from amforge.materials import AMMaterial, get_material

Array = jax.Array

__all__ = [
    "PROCESS_BOUNDS",
    "clip_to_bounds",
    "normalize_process",
    "denormalize_process",
    "constant_plan",
    "heuristic_plan",
    "process_window_score",
    "scan_process_window",
    "recommended_power_for_enthalpy",
    "ScanSegments",
    "zigzag_segments",
    "contour_segments",
    "spiral_segments",
    "segments_from_gcode",
    "layer_activation_times",
]


# ===========================================================================
# 工艺参数边界 —— 物理/设备可行域
# ===========================================================================
#: (下界, 上界)，SI 单位。来源：工业 SLM/LSF 设备典型规格。
PROCESS_BOUNDS: dict[str, tuple[float, float]] = {
    # SLM: 光纤激光 50~1000 W；LSF/DED: 500~4000 W
    "laser_power": (20.0, 4000.0),
    # SLM: 0.1~3 m/s；LSF: 0.002~0.05 m/s
    "scan_speed": (1e-3, 5.0),
    "layer_thickness": (10e-6, 1.0e-3),
    "hatch_spacing": (20e-6, 2.0e-3),
    "beam_radius": (15e-6, 1.5e-3),
    "absorption": (0.15, 0.85),
    "preheat_temp": (293.0, 1273.0),
    "powder_feed_rate": (0.0, 1e-3),
    "dwell_time": (0.0, 60.0),
}

_SCALAR_KEYS = tuple(PROCESS_BOUNDS)


def clip_to_bounds(plan: ProcessPlan, *, bounds: Mapping[str, tuple[float, float]] | None = None) -> ProcessPlan:
    """把工艺方案投影回设备可行域（可微，梯度在边界处为 0）。

    ``bounds`` 可给出**收紧后的子盒**（如
    :func:`amforge.thermal_enthalpy.chain_schedule` 返回的 ``bounds``：网格能分辨
    得出、且显式步数预算内能积分出来的那部分工艺盒）。
    """
    b = dict(bounds or PROCESS_BOUNDS)
    kw: dict[str, Any] = {}
    for k in _SCALAR_KEYS:
        lo, hi = b[k]
        kw[k] = jnp.clip(getattr(plan, k), lo, hi)
    return plan.replace(**kw)


def normalize_process(plan: ProcessPlan, *,
                      bounds: Mapping[str, tuple[float, float]] | None = None) -> Array:
    """工艺 → 归一化向量 ∈ [0,1]^9（供神经网络输出/优化器变量使用）。

    对功率、速度、层厚、间距用**对数**归一化：这些量跨 2~3 个数量级，
    线性归一化会让优化器在低值区几乎无梯度分辨力。

    ``bounds`` 给出子盒时，**必须与 :func:`denormalize_process` 用同一套**，
    否则 z↔工艺 的往返不再互逆。
    """
    b = dict(bounds or PROCESS_BOUNDS)
    out = []
    for k in _SCALAR_KEYS:
        lo, hi = b[k]
        v = jnp.mean(jnp.atleast_1d(getattr(plan, k)))
        if k in ("laser_power", "scan_speed", "layer_thickness",
                 "hatch_spacing", "beam_radius"):
            z = (jnp.log(jnp.maximum(v, 1e-12)) - np.log(lo)) / (np.log(hi) - np.log(lo))
        else:
            z = (v - lo) / (hi - lo)
        out.append(jnp.clip(z, 0.0, 1.0))
    return jnp.stack(out)


def denormalize_process(z: Array, *, n_layers: int = 1, modality: str = "SLM",
                        rotation_per_layer: float = 67.0 * np.pi / 180.0,
                        bounds: Mapping[str, tuple[float, float]] | None = None
                        ) -> ProcessPlan:
    """归一化向量 → :class:`ProcessPlan`（:func:`normalize_process` 的逆）。

    这是神经网络输出层与物理求解器之间的桥。网络输出经 ``sigmoid``
    后落在 [0,1]，本函数把它映射回带物理单位的工艺参数，
    **天然满足设备约束**，无需额外的惩罚项。

    ``bounds`` 用子盒（须与 :func:`normalize_process` 同一套）时，映射
    **天然满足该子盒**——例如显式热解档位要求 ``scan_speed ≥ v_floor``
    （否则钉住的 ``n_steps`` 覆盖不了曝光），此时靠盒子而不是靠校核报错来保证安全。
    """
    b = dict(bounds or PROCESS_BOUNDS)
    z = jnp.atleast_1d(z)
    vals: dict[str, Array] = {}
    for i, k in enumerate(_SCALAR_KEYS):
        lo, hi = b[k]
        zi = jnp.clip(z[i], 0.0, 1.0)
        if k in ("laser_power", "scan_speed", "layer_thickness",
                 "hatch_spacing", "beam_radius"):
            vals[k] = jnp.exp(np.log(lo) + zi * (np.log(hi) - np.log(lo)))
        else:
            vals[k] = lo + zi * (hi - lo)
    ang = rotation_per_layer * jnp.arange(n_layers, dtype=jnp.float64)
    return ProcessPlan(scan_angle=ang, modality=modality, **vals)


# ===========================================================================
# 1. 常量工艺（最轻量的契约生产者）
# ===========================================================================
def constant_plan(*, params: Mapping[str, Any] | None = None) -> ProcessPlan:
    """从 ``params`` 构造均匀工艺方案。"""
    p = dict(params or {})
    n_layers = int(p.pop("n_layers", 1))
    modality = str(p.pop("modality", "SLM"))
    rot = float(p.pop("rotation_per_layer", 67.0 * np.pi / 180.0))
    return ProcessPlan.uniform(n_layers, modality=modality,
                               rotation_per_layer=rot, **p)


@register_solver(
    "process.constant",
    consumes=(),
    produces="ProcessPlan",
    stage="process",
    cost=0.05,
    defaults={"modality": "SLM"},
    doc="params 指定的均匀工艺方案（可微、可 vmap 扫参）。",
)
def _solve_constant(*, params=None) -> ProcessPlan:
    return constant_plan(params=params)


# ===========================================================================
# 2. 几何感知的规则式工艺规划
# ===========================================================================
def recommended_power_for_enthasy(*a, **k):  # pragma: no cover - 兼容别名
    return recommended_power_for_enthalpy(*a, **k)


def recommended_power_for_enthalpy(
    material: AMMaterial, *, target_enthalpy: float = 12.0,
    scan_speed: float = 1.0, beam_radius: float = 50e-6,
    absorption: float | None = None,
) -> Array:
    """反解出使无量纲焓 ΔH/h_s 命中目标值的激光功率 [W]。

    无量纲焓判据（King et al. 2014）：

    .. math::
        \\frac{\\Delta H}{h_s}
        = \\frac{A P}{\\rho c_p T_m \\sqrt{\\pi \\alpha v r^3}}

    经验工艺窗口：

    * ``< 6``：能量不足 → 未熔合 (lack-of-fusion)
    * ``6 ~ 25``：**传导模式**，致密度最优（目标区）
    * ``> ~30``：匙孔模式 → 匙孔气孔、飞溅

    默认取 12（窗口中部偏下，兼顾致密与稳定）。
    """
    A = material.absorptivity if absorption is None else absorption
    hs = material.rho_solid * material.cp_solid * material.T_melt
    alpha = material.diffusivity()
    denom = jnp.sqrt(jnp.pi * alpha * scan_speed * beam_radius ** 3)
    return target_enthalpy * hs * denom / jnp.maximum(A, 1e-6)


def heuristic_plan(
    geometry: PartGeometry,
    *,
    material: AMMaterial | str = "Ti6Al4V",
    modality: str = "SLM",
    target_enthalpy: float = 12.0,
    scan_speed: float | None = None,
    beam_radius: float | None = None,
    overhang_derate: float = 0.35,
    thinwall_derate: float = 0.25,
    hatch_overlap: float = 0.35,
    rotation_per_layer: float = 67.0 * np.pi / 180.0,
    per_layer: bool = True,
) -> ProcessPlan:
    """**几何感知**的规则式工艺规划。

    规划逻辑（全部可微）：

    1. 按工艺模式给定基准速度/光斑（SLM 细而快，LSF 粗而慢）；
    2. 由无量纲焓反解基准功率，把工艺锁在传导模式窗口内；
    3. **悬垂降功率**：悬垂区下方是粉末（导热差 ~1/100），同样能量会
       过热塌陷。按悬垂体积分数整体降功率 ``overhang_derate``；
    4. **薄壁降功率**：薄壁散热路径窄，同理降功率；
    5. 扫描间距按熔池宽度的 ``1-hatch_overlap`` 取，保证道间搭接；
    6. 层厚取光斑直径的经验比例，并使层数为整数；
    7. 层间旋转 67°（非周期角，避免层间搭接缺陷对齐成贯穿弱面）；
    8. **逐层调制**：底部若干层因基板散热强而适度提功率，
       顶部因热累积而降功率（用平滑指数律，避免阶跃）。

    这些规则本身不是"最优"，而是给可微闭环（功能 2）一个
    **物理合理的初值** —— 从随机初值出发的梯度优化极易落进
    未熔合/匙孔的病态区，导致仿真崩溃或梯度爆炸。
    """
    mat = get_material(material) if isinstance(material, str) else material

    if modality.upper() == "LSF":
        v0 = 0.012 if scan_speed is None else scan_speed
        r0 = 600e-6 if beam_radius is None else beam_radius
        t_layer = 0.5e-3
        feed = 2.5e-4
    else:  # SLM
        v0 = 1.0 if scan_speed is None else scan_speed
        r0 = 50e-6 if beam_radius is None else beam_radius
        t_layer = 40e-6
        feed = 0.0

    # ⚠ 这里**不**做网格分辨率补偿（2026-10-07 D0 复盘推翻上一版）：heuristic_plan 是
    # 物理规划器，提出的是"这台设备/这种材料该怎么扫"；体素分辨不分辨得开是**数值档**
    # 的事，由 chain_schedule 的工艺子盒（hatch/lt ≥ dx、r ≥ dx/2）与求解器的
    # resolution_policy 负责——在那里拒绝或降档，并把代价如实报出来。把 dx 耦合进规划器
    # 会让"粗网格上的启发式工艺"在物理约束罚项里被记成工艺缺陷（实测 pen 0→0.667），
    # 混淆"网格太粗"与"工艺不可制造"两件事。
    P0 = recommended_power_for_enthalpy(
        mat, target_enthalpy=target_enthalpy, scan_speed=v0, beam_radius=r0)

    # --- 几何诊断 ---------------------------------------------------------
    oh = overhang_fraction(geometry, angle_threshold=45.0)
    tw = thin_wall_field(geometry)
    occ = geometry.soft_occupancy()
    thin_frac = jnp.sum(tw * occ) / jnp.maximum(jnp.sum(occ), 1e-12)

    derate = (1.0 - overhang_derate * oh) * (1.0 - thinwall_derate * thin_frac)
    P0 = P0 * jnp.clip(derate, 0.35, 1.0)

    # --- 熔池宽度估计 → 扫描间距 ------------------------------------------
    # 传导模式下 w ≈ 2r·sqrt(1 + ΔH/h_s / 8)，是 Eagar-Tsai 的一阶近似
    w_est = 2.0 * r0 * jnp.sqrt(1.0 + target_enthalpy / 8.0)
    hatch = jnp.clip(w_est * (1.0 - hatch_overlap),
                     PROCESS_BOUNDS["hatch_spacing"][0],
                     PROCESS_BOUNDS["hatch_spacing"][1])

    n_layers = geometry.layer_count(t_layer) if per_layer else 1

    # --- 逐层功率调制 -----------------------------------------------------
    if per_layer and n_layers > 1:
        k = jnp.arange(n_layers, dtype=jnp.float64) / (n_layers - 1.0)
        # 底部 +8%（基板吸热）、顶部 -12%（热累积），指数过渡
        mod = 1.08 * jnp.exp(-k / 0.15) + (1.0 - 0.12 * k) * (1.0 - jnp.exp(-k / 0.15))
        power = P0 * mod
        speed = jnp.full((n_layers,), v0)
    else:
        power = jnp.atleast_1d(P0)
        speed = jnp.atleast_1d(jnp.asarray(v0))

    ang = rotation_per_layer * jnp.arange(max(n_layers, 1), dtype=jnp.float64)

    plan = ProcessPlan(
        laser_power=power,
        scan_speed=speed,
        layer_thickness=jnp.asarray(t_layer),
        hatch_spacing=hatch,
        beam_radius=jnp.asarray(r0),
        scan_angle=ang,
        absorption=jnp.asarray(mat.absorptivity),
        preheat_temp=jnp.asarray(473.0 if modality.upper() == "SLM" else 293.0),
        powder_feed_rate=jnp.asarray(feed),
        dwell_time=jnp.asarray(0.0 if modality.upper() == "SLM" else 2.0),
        modality=modality.upper(),
    )
    return clip_to_bounds(plan)


@register_solver(
    "process.heuristic",
    consumes=("PartGeometry",),
    produces="ProcessPlan",
    stage="process",
    modality=("SLM", "LSF"),
    cost=0.5,
    defaults={"material": "Ti6Al4V", "modality": "SLM", "target_enthalpy": 12.0},
    doc="几何感知规则式工艺规划：悬垂/薄壁降功率 + 无量纲焓锁定工艺窗口。",
)
def _solve_heuristic(*, geometry, params=None) -> ProcessPlan:
    return heuristic_plan(geometry, **dict(params or {}))


# ===========================================================================
# 3. 工艺窗口筛选（可行性预筛，无需跑 CFD）
# ===========================================================================
def process_window_score(plan: ProcessPlan, material: AMMaterial | str = "Ti6Al4V",
                         *, lof_weight: float = 1.0, keyhole_weight: float = 1.0,
                         balling_weight: float = 0.5) -> dict[str, Array]:
    """零成本的工艺可行性评分（三大缺陷机制的解析判据）。

    Returns
    -------
    dict
        ``normalized_enthalpy`` / ``lof_risk`` / ``keyhole_risk`` /
        ``balling_risk`` / ``score``（越小越好）。

    判据依据
    --------
    * **未熔合**：熔深必须超过层厚，且道间必须搭接。
      用 ``ΔH/h_s < 6`` 与 ``h > w`` 双重软惩罚。
    * **匙孔**：``ΔH/h_s > 30`` 后蒸发反冲压凹陷不稳定，气泡被
      快速凝固前沿捕获成球形气孔。
    * **球化 (balling)**：熔道长宽比 ``L/w > π`` 时
      Plateau-Rayleigh 不稳定使连续熔道断裂成球。
    """
    mat = get_material(material) if isinstance(material, str) else material
    dH = plan.normalized_enthalpy(
        rho=mat.rho_solid, cp=mat.cp_solid, T_melt=mat.T_melt,
        diffusivity=mat.diffusivity())
    dH = jnp.mean(jnp.atleast_1d(dH))

    r = plan.beam_radius
    w = 2.0 * r * jnp.sqrt(1.0 + jnp.maximum(dH, 0.0) / 8.0)
    depth = 0.5 * w * (1.0 + 0.06 * jnp.maximum(dH - 6.0, 0.0))
    # Eagar-Tsai 型熔道拉长：L/w ~ 1 + Pe，Pe = v·w/(2α)
    pe = jnp.mean(jnp.atleast_1d(plan.scan_speed)) * w / (2.0 * mat.diffusivity())
    length = w * (1.0 + pe)

    sig = jax.nn.sigmoid
    lof_energy = sig((6.0 - dH) / 1.5)
    lof_depth = sig((plan.layer_thickness * 1.5 - depth) / (0.2 * plan.layer_thickness))
    lof_overlap = sig((plan.hatch_spacing - 0.95 * w) / (0.1 * w))
    lof = jnp.clip(jnp.maximum(jnp.maximum(lof_energy, lof_depth), lof_overlap), 0, 1)

    keyhole = sig((dH - 30.0) / 5.0)
    balling = sig((length / jnp.maximum(w, 1e-12) - jnp.pi) / 0.6)

    score = jnp.clip(lof_weight * lof + keyhole_weight * keyhole
                     + balling_weight * balling, 0.0, 3.0)
    return {
        "normalized_enthalpy": dH,
        "melt_width": w,
        "melt_depth": depth,
        "melt_length": length,
        "lof_risk": lof,
        "keyhole_risk": keyhole,
        "balling_risk": balling,
        "score": score,
    }


def scan_process_window(
    material: AMMaterial | str = "Ti6Al4V", *,
    powers: Sequence[float] | Array = np.linspace(50.0, 500.0, 24),
    speeds: Sequence[float] | Array = np.linspace(0.2, 2.5, 24),
    layer_thickness: float = 40e-6, hatch_spacing: float = 80e-6,
    beam_radius: float = 50e-6, modality: str = "SLM",
) -> dict[str, Array]:
    """在 P-v 平面上批量扫描工艺窗口（``jax.vmap`` 双重向量化）。

    这是"求解器可组合"的直接红利：同一个评分函数无需改一行代码
    就能被向量化成二维工艺图，用于给优化器画可行域、给用户看窗口。
    """
    mat = get_material(material) if isinstance(material, str) else material
    P = jnp.asarray(powers, dtype=jnp.float64)
    V = jnp.asarray(speeds, dtype=jnp.float64)

    def one(p, v):
        plan = ProcessPlan.uniform(
            1, modality=modality, laser_power=float("nan"), scan_speed=1.0,
            layer_thickness=layer_thickness, hatch_spacing=hatch_spacing,
            beam_radius=beam_radius, absorption=mat.absorptivity)
        plan = plan.replace(laser_power=p, scan_speed=v)
        return process_window_score(plan, mat)

    grid = jax.vmap(jax.vmap(one, in_axes=(None, 0)), in_axes=(0, None))(P, V)
    grid["powers"] = P
    grid["speeds"] = V
    return grid


# ===========================================================================
# 4. 扫描路径（喂给熔池求解器 / 逐层激活）
# ===========================================================================
@dataclass(frozen=True)
class ScanSegments:
    """扫描路径的**线段**表示（对求解器最友好的形式）。

    Attributes
    ----------
    start, end : (n_seg, 2)
        每段起止点的 XY 坐标 [m]。
    layer : (n_seg,)
        所属层索引（int）。
    jump : (n_seg,)
        1 = 空跳（激光关闭），0 = 熔化行程。
    """

    start: Array
    end: Array
    layer: Array
    jump: Array

    @property
    def n_segments(self) -> int:
        return int(self.start.shape[0])

    def lengths(self) -> Array:
        return jnp.linalg.norm(self.end - self.start, axis=-1)

    def total_length(self, *, active_only: bool = True) -> Array:
        L = self.lengths()
        if active_only:
            L = L * (1.0 - self.jump)
        return jnp.sum(L)

    def duration(self, plan: ProcessPlan, *, jump_speed: float = 5.0) -> Array:
        """按工艺速度估算总耗时 [s]（含空跳）。"""
        L = self.lengths()
        v = jnp.mean(jnp.atleast_1d(plan.scan_speed))
        t = jnp.where(self.jump > 0.5, L / jump_speed, L / jnp.maximum(v, 1e-9))
        return jnp.sum(t)

    def of_layer(self, k: int) -> "ScanSegments":
        m = np.asarray(self.layer) == k
        return ScanSegments(self.start[m], self.end[m],
                           self.layer[m], self.jump[m])


jax.tree_util.register_pytree_node(
    ScanSegments,
    lambda s: ((s.start, s.end), (s.layer, s.jump)),
    lambda aux, leaves: ScanSegments(leaves[0], leaves[1], aux[0], aux[1]),
)


def _rot2(theta: Array) -> Array:
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([[c, -s], [s, c]])


def zigzag_segments(geometry: PartGeometry, plan: ProcessPlan, *,
                    layers: Sequence[int] | None = None,
                    margin: float = 0.0) -> ScanSegments:
    """按层生成折返 (zigzag) 填充路径，含层间旋转。

    路径由包围盒生成后按层切片掩膜裁剪。对 ``hatch_spacing`` 与
    ``scan_angle`` 可微 —— 这让"扫描策略"本身也能进优化变量。
    """
    lo, hi = geometry.bbox()
    lo = np.asarray(lo)[:2] + margin
    hi = np.asarray(hi)[:2] - margin
    center = 0.5 * (lo + hi)
    diag = float(np.linalg.norm(hi - lo)) + 1e-12

    h = float(jnp.mean(jnp.atleast_1d(plan.hatch_spacing)))
    n_line = max(1, int(np.ceil(diag / max(h, 1e-9))))
    n_layers = plan.n_layers if layers is None else len(list(layers))
    idx = list(range(n_layers)) if layers is None else list(layers)

    starts, ends, lays, jumps = [], [], [], []
    ang_all = jnp.atleast_1d(plan.scan_angle)
    for k in idx:
        theta = ang_all[min(k, ang_all.shape[0] - 1)]
        R = _rot2(theta)
        offs = (jnp.arange(n_line, dtype=jnp.float64) - 0.5 * (n_line - 1)) * h
        for i in range(n_line):
            s_loc = jnp.array([-0.5 * diag, offs[i]])
            e_loc = jnp.array([+0.5 * diag, offs[i]])
            if i % 2 == 1:            # 折返
                s_loc, e_loc = e_loc, s_loc
            starts.append(R @ s_loc + jnp.asarray(center))
            ends.append(R @ e_loc + jnp.asarray(center))
            lays.append(k)
            jumps.append(0.0)

    return ScanSegments(
        start=jnp.stack(starts), end=jnp.stack(ends),
        layer=jnp.asarray(lays, dtype=jnp.int32),
        jump=jnp.asarray(jumps),
    )


def contour_segments(geometry: PartGeometry, plan: ProcessPlan, *,
                     layer: int = 0, n_offsets: int = 2,
                     n_points: int = 128) -> ScanSegments:
    """轮廓偏置 (contour / shell) 路径 —— 表面质量关键。

    在层切片上按 SDF 等值线 ``sdf = -j·h`` 提取轮廓，
    用极坐标射线求交（对 SDF 可微的软求交）。
    """
    zs = np.asarray(geometry.origin)[2] + geometry.spacing * np.arange(
        geometry.shape[-1])
    z = float(zs[min(layer, len(zs) - 1)])
    lo, hi = geometry.bbox()
    center = 0.5 * (np.asarray(lo)[:2] + np.asarray(hi)[:2])
    rmax = 0.5 * float(np.linalg.norm(np.asarray(hi)[:2] - np.asarray(lo)[:2]))
    h = float(jnp.mean(jnp.atleast_1d(plan.hatch_spacing)))

    from amforge.geometry import sample_sdf

    ang = jnp.linspace(0.0, 2.0 * jnp.pi, n_points + 1)[:-1]
    starts, ends, lays, jumps = [], [], [], []
    for j in range(n_offsets):
        target = -(j + 0.5) * h
        # 沿射线二分求 sdf == target（固定 24 次，可微且无 while）
        r_lo = jnp.zeros_like(ang)
        r_hi = jnp.full_like(ang, rmax)
        for _ in range(24):
            r_mid = 0.5 * (r_lo + r_hi)
            pts = jnp.stack([center[0] + r_mid * jnp.cos(ang),
                             center[1] + r_mid * jnp.sin(ang),
                             jnp.full_like(ang, z)], axis=-1)
            d = sample_sdf(geometry, pts)
            inside = d < target
            r_lo = jnp.where(inside, r_mid, r_lo)
            r_hi = jnp.where(inside, r_hi, r_mid)
        r = 0.5 * (r_lo + r_hi)
        pts = jnp.stack([center[0] + r * jnp.cos(ang),
                         center[1] + r * jnp.sin(ang)], axis=-1)
        starts.append(pts)
        ends.append(jnp.roll(pts, -1, axis=0))
        lays.append(np.full(n_points, layer))
        jumps.append(np.zeros(n_points))

    return ScanSegments(
        start=jnp.concatenate(starts), end=jnp.concatenate(ends),
        layer=jnp.asarray(np.concatenate(lays), dtype=jnp.int32),
        jump=jnp.asarray(np.concatenate(jumps)),
    )


def spiral_segments(geometry: PartGeometry, plan: ProcessPlan, *,
                    layer: int = 0, n_points: int = 512) -> ScanSegments:
    """阿基米德螺旋路径（回转体/圆形截面最省空跳）。"""
    lo, hi = geometry.bbox()
    center = 0.5 * (np.asarray(lo)[:2] + np.asarray(hi)[:2])
    rmax = 0.5 * float(np.linalg.norm(np.asarray(hi)[:2] - np.asarray(lo)[:2]))
    h = float(jnp.mean(jnp.atleast_1d(plan.hatch_spacing)))
    n_turn = max(1.0, rmax / max(h, 1e-9))

    t = jnp.linspace(0.0, 1.0, n_points)
    theta = 2.0 * jnp.pi * n_turn * t
    r = rmax * t
    pts = jnp.stack([center[0] + r * jnp.cos(theta),
                     center[1] + r * jnp.sin(theta)], axis=-1)
    return ScanSegments(
        start=pts[:-1], end=pts[1:],
        layer=jnp.full((n_points - 1,), layer, dtype=jnp.int32),
        jump=jnp.zeros((n_points - 1,)),
    )


def segments_from_gcode(path: str, *, scale: float = 1e-3,
                        z_tol: float = 1e-9) -> ScanSegments:
    """解析 G-code (G0/G1) 为扫描线段，按 Z 变化自动分层。

    支持商业切片软件（如 Materialise Magics / Netfabb 导出的 CLI/G-code）
    生成的真实路径 —— 这让本软件能直接复现工厂实际打印的工艺。
    """
    x = y = z = 0.0
    layer = -1
    last_z = None
    starts, ends, lays, jumps = [], [], [], []
    with open(path, "r", errors="ignore") as fh:
        for raw in fh:
            line = raw.split(";")[0].strip()
            if not line:
                continue
            tok = line.split()
            cmd = tok[0].upper()
            if cmd not in ("G0", "G00", "G1", "G01"):
                continue
            nx, ny, nz = x, y, z
            for t in tok[1:]:
                key, val = t[0].upper(), t[1:]
                try:
                    fv = float(val)
                except ValueError:
                    continue
                if key == "X":
                    nx = fv * scale
                elif key == "Y":
                    ny = fv * scale
                elif key == "Z":
                    nz = fv * scale
            if last_z is None or abs(nz - last_z) > z_tol:
                layer += 1
                last_z = nz
            if abs(nx - x) > 0 or abs(ny - y) > 0:
                starts.append((x, y))
                ends.append((nx, ny))
                lays.append(layer)
                jumps.append(1.0 if cmd in ("G0", "G00") else 0.0)
            x, y, z = nx, ny, nz

    if not starts:
        raise ValueError(f"G-code {path!r} 中未解析到任何运动指令。")
    return ScanSegments(
        start=jnp.asarray(starts), end=jnp.asarray(ends),
        layer=jnp.asarray(lays, dtype=jnp.int32),
        jump=jnp.asarray(jumps),
    )


def layer_activation_times(geometry: PartGeometry, plan: ProcessPlan, *,
                           area_fraction: Array | None = None,
                           jump_speed: float = 5.0) -> Array:
    """每层的**激活时刻** [s]（累计时间），供逐层生死单元使用。

    单层耗时 = 熔化行程 + 空跳 + 铺粉/送粉 + 层间停留。
    熔化行程长度按该层实体截面积 / 扫描间距估算，因此对
    ``hatch_spacing``、``scan_speed``、``dwell_time`` 可微。
    """
    n = plan.n_layers
    if area_fraction is None:
        occ = geometry.soft_occupancy()
        # 沿 Z 求每层面积占比，再重采样到 n 层
        az = jnp.mean(occ, axis=tuple(range(geometry.dim - 1)))
        zi = jnp.linspace(0.0, az.shape[0] - 1.0, n)
        area_fraction = jnp.interp(zi, jnp.arange(az.shape[0]), az)

    lo, hi = geometry.bbox()
    extent = hi[:2] - lo[:2]
    area = area_fraction * extent[0] * extent[1]
    track_len = area / jnp.maximum(plan.hatch_spacing, 1e-12)
    v = jnp.atleast_1d(plan.scan_speed)
    v = jnp.broadcast_to(v, (n,)) if v.shape[0] == 1 else v[:n]
    t_melt = track_len / jnp.maximum(v, 1e-12)
    t_jump = 0.15 * track_len / jump_speed
    t_recoat = 8.0 if plan.modality == "SLM" else 0.0
    dt = t_melt + t_jump + t_recoat + plan.dwell_time
    return jnp.cumsum(dt)
