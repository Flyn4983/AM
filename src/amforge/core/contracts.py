"""AMForge 模块间数据契约 (data contracts).

本模块是整个平台"**模块之间无缝链接**"的核心。每个求解器模块都不直接依赖
其它模块的内部实现，而只依赖这里定义的一组**契约对象**：

    PartGeometry   →  ProcessPlan  →  ThermalHistory / MeltPoolResult
                                   →  MicrostructureResult
                                   →  ConstitutiveField
                                   →  AsBuiltPart
                                   →  ServiceVerdict

只要一个模块声明自己 *消费* 哪些契约、*产出* 哪些契约
(见 :mod:`amforge.core.registry`)，编排器 (:mod:`amforge.core.graph`) 就能
自动把它接到管线的正确位置上，无需手写胶水代码。

设计约束
--------
1. **全部为 frozen dataclass**，数值字段是 JAX 数组 → 注册为 pytree，
   因此 ``jax.grad`` / ``jax.jit`` / ``jax.vmap`` 可以穿过任意契约对象。
2. **静态元数据**（维度、网格尺寸、材料名、方法名等）放在 ``aux`` 里，
   不参与求导，避免 JIT 反复重编译。
3. **单位一律 SI**（m, s, K, Pa, W, kg）。契约文档字符串标注每个量的单位。
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from amforge.materials import AMMaterial

Array = jnp.ndarray


# ---------------------------------------------------------------------------
# pytree 注册辅助工具
# ---------------------------------------------------------------------------
def register_contract(cls, leaf_names: Sequence[str], aux_names: Sequence[str]):
    """把一个 frozen dataclass 注册为 JAX pytree。

    ``leaf_names`` 中的字段作为可微叶子 (JAX 数组 / 标量)；
    ``aux_names`` 中的字段作为静态元数据 (必须 hashable)。
    """
    leaf_names = tuple(leaf_names)
    aux_names = tuple(aux_names)

    def _flatten(obj):
        leaves = tuple(getattr(obj, k) for k in leaf_names)
        aux = tuple(getattr(obj, k) for k in aux_names)
        return leaves, aux

    def _unflatten(aux, leaves):
        kw = dict(zip(leaf_names, leaves))
        kw.update(dict(zip(aux_names, aux)))
        return cls(**kw)

    jax.tree_util.register_pytree_node(cls, _flatten, _unflatten)
    return cls


def _as_array(x, dtype=jnp.float64):
    return jnp.asarray(x, dtype=dtype)


# ===========================================================================
# 1. 几何契约 —— 任意复杂构型
# ===========================================================================
@dataclass(frozen=True)
class PartGeometry:
    """待制造零件的**任意复杂几何**表示（体素化带符号距离场）。

    采用 SDF (signed distance field) 体素栅格作为统一几何载体，理由：

    * STL / STEP / 隐式 CSG / 点云 都能统一转成 SDF，因此"任意复杂构型"
      不受几何来源限制；
    * SDF 天然可微 —— 几何本身可以作为优化变量（拓扑/形状优化）；
    * 分层切片、扫描路径生成、生死单元激活、几何偏差比较，
      都能在同一个栅格上完成，避免跨模块的网格映射误差。

    Attributes
    ----------
    sdf : (nx, ny, nz) 或 (nx, ny)
        带符号距离场 [m]，**负值在零件内部**。
    origin : (dim,)
        栅格原点物理坐标 [m]。
    spacing : float
        体素边长 [m]（等距栅格）。
    dim : int
        2 或 3（静态）。
    name : str
        零件名（静态）。
    """

    sdf: Array
    origin: Array
    spacing: float
    dim: int = 3
    name: str = "part"

    # ---- 派生量 ----------------------------------------------------------
    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.sdf.shape)

    @property
    def occupancy(self) -> Array:
        """硬占位（0/1），负 SDF 即为实体。"""
        return (self.sdf < 0.0).astype(self.sdf.dtype)

    def soft_occupancy(self, eps: float | None = None) -> Array:
        """**可微**软占位：用 tanh 平滑 Heaviside，供梯度穿过几何比较。

        eps 默认取 1 个体素，保证界面上有 ~2 个体素的过渡带。
        """
        if eps is None:
            eps = float(self.spacing)
        return 0.5 * (1.0 - jnp.tanh(self.sdf / jnp.maximum(eps, 1e-12)))

    def volume(self) -> Array:
        """实体体积 [m^dim]（可微，用软占位积分）。"""
        cell = self.spacing ** self.dim
        return jnp.sum(self.soft_occupancy()) * cell

    def bbox(self) -> tuple[Array, Array]:
        """零件包围盒 (lo, hi) [m]。"""
        lo = self.origin
        hi = self.origin + self.spacing * (jnp.asarray(self.shape) - 1.0)
        return lo, hi

    def layer_count(self, layer_thickness: float) -> int:
        """给定层厚时的层数（沿最后一个轴 = 构建方向 Z）。"""
        height = self.spacing * (self.shape[-1] - 1)
        return max(1, int(np.ceil(float(height) / float(layer_thickness))))

    def coords(self) -> Array:
        """体素中心坐标 (..., dim) [m]。"""
        axes = [self.origin[d] + self.spacing * jnp.arange(self.shape[d])
                for d in range(self.dim)]
        grids = jnp.meshgrid(*axes, indexing="ij")
        return jnp.stack(grids, axis=-1)


register_contract(PartGeometry, ("sdf", "origin", "spacing"), ("dim", "name"))


# ===========================================================================
# 2. 工艺契约 —— 神经网络的预测目标 / 各求解器的输入
# ===========================================================================
@dataclass(frozen=True)
class ProcessPlan:
    """增材制造**工艺方案**：逐层工艺参数 + 扫描策略。

    这是"几何 → 工艺"神经网络的**输出**，也是所有物理求解器的**输入**。
    全部字段都是可微叶子，因此损失函数对工艺参数的梯度可以一路回传到
    预测网络的权重上。

    Attributes
    ----------
    laser_power : (n_layers,) 或 标量
        激光功率 [W]。
    scan_speed : (n_layers,) 或 标量
        扫描速度 [m/s]。
    layer_thickness : 标量
        层厚 [m]。
    hatch_spacing : 标量
        扫描间距 [m]。
    beam_radius : 标量
        光斑 1/e² 半径 [m]。
    scan_angle : (n_layers,)
        每层扫描方向角 [rad]（层间旋转策略）。
    absorption : 标量
        激光吸收率 [-]。
    preheat_temp : 标量
        基板/粉床预热温度 [K]。
    powder_feed_rate : 标量
        送粉率 [kg/s]（LSF/DED 专用，SLM 置 0）。
    dwell_time : 标量
        层间停留时间 [s]。
    modality : str
        ``"SLM"`` 或 ``"LSF"``（静态）。
    """

    laser_power: Array
    scan_speed: Array
    layer_thickness: Array
    hatch_spacing: Array
    beam_radius: Array
    scan_angle: Array
    absorption: Array
    preheat_temp: Array
    powder_feed_rate: Array
    dwell_time: Array
    modality: str = "SLM"

    # ---- 工艺派生指标 ----------------------------------------------------
    def volumetric_energy_density(self) -> Array:
        """体能量密度 VED = P / (v · h · t) [J/mm³ → 这里给 J/m³]。

        文献中最常用的工艺综合指标（可行性报告 4.1 节）。
        """
        denom = (self.scan_speed * self.hatch_spacing * self.layer_thickness)
        return self.laser_power / jnp.maximum(denom, 1e-30)

    def linear_energy_density(self) -> Array:
        """线能量密度 P/v [J/m]，控制熔池尺度的一阶指标。"""
        return self.laser_power / jnp.maximum(self.scan_speed, 1e-30)

    def normalized_enthalpy(self, *, rho: float, cp: float, T_melt: float,
                            diffusivity: float) -> Array:
        """无量纲焓 ΔH/h_s —— keyhole 转变的经典判据。

        ΔH/h_s = A·P / (ρ·cp·T_m · sqrt(π·α·v·r³))
        经验阈值：> ~30 进入 keyhole 模式（易产生匙孔气孔）。
        """
        hs = rho * cp * T_melt
        denom = jnp.sqrt(jnp.pi * diffusivity * self.scan_speed
                         * self.beam_radius ** 3)
        return self.absorption * self.laser_power / jnp.maximum(hs * denom, 1e-30)

    @property
    def n_layers(self) -> int:
        return int(jnp.atleast_1d(self.scan_angle).shape[0])

    def as_vector(self) -> Array:
        """把工艺方案摊平成一维向量（供优化器/网络使用）。"""
        return jnp.concatenate([jnp.atleast_1d(jnp.asarray(v)).ravel()
                                for v in (self.laser_power, self.scan_speed,
                                          self.layer_thickness, self.hatch_spacing,
                                          self.beam_radius, self.scan_angle,
                                          self.absorption, self.preheat_temp,
                                          self.powder_feed_rate, self.dwell_time)])

    def replace(self, **kw) -> "ProcessPlan":
        return replace(self, **kw)

    @staticmethod
    def uniform(n_layers: int = 1, *, modality: str = "SLM",
                laser_power: float = 200.0, scan_speed: float = 1.0,
                layer_thickness: float = 40e-6, hatch_spacing: float = 80e-6,
                beam_radius: float = 50e-6, absorption: float = 0.4,
                preheat_temp: float = 373.0, powder_feed_rate: float = 0.0,
                dwell_time: float = 0.0,
                rotation_per_layer: float = 67.0 * np.pi / 180.0) -> "ProcessPlan":
        """构造一个均匀工艺方案（默认 67° 层间旋转，工业界常用策略）。"""
        ang = rotation_per_layer * jnp.arange(n_layers, dtype=jnp.float64)
        return ProcessPlan(
            laser_power=_as_array(laser_power),
            scan_speed=_as_array(scan_speed),
            layer_thickness=_as_array(layer_thickness),
            hatch_spacing=_as_array(hatch_spacing),
            beam_radius=_as_array(beam_radius),
            scan_angle=ang,
            absorption=_as_array(absorption),
            preheat_temp=_as_array(preheat_temp),
            powder_feed_rate=_as_array(powder_feed_rate),
            dwell_time=_as_array(dwell_time),
            modality=modality,
        )


register_contract(
    ProcessPlan,
    ("laser_power", "scan_speed", "layer_thickness", "hatch_spacing",
     "beam_radius", "scan_angle", "absorption", "preheat_temp",
     "powder_feed_rate", "dwell_time"),
    ("modality",),
)


# ===========================================================================
# 2.5 粉末床契约（P2 粉末尺度）—— 熔池/热解之前的"铺粉态"
# ===========================================================================
@dataclass(frozen=True)
class PowderBedResult:
    """**粉末床铺粉态 + 粉末尺度缺陷前驱**（DEM/SPH/MPM 颗粒尺度求解产物）。

    连续介质熔池模型（:class:`MeltPoolResult`）以"均质连续金属"为出发点，
    因此**看不到粉末颗粒本身**引起的三类缺陷：

    * **球化 (balling)**：熔道在 Plateau–Rayleigh 不稳定性下断裂成球串；
    * **未熔合 (lack of fusion)**：铺粉密度不足 / 粉层过厚 → 层间未连接；
    * **飞溅 (spatter) + 剥蚀 (denudation)**：蒸气反冲 / 气流卷吸抛出颗粒。

    这三者都是**颗粒尺度**现象，只能由 DEM（铺粉重排）、SPH（熔道自由界面）、
    MPM（粉末再分布）给出。本契约把它们汇总成可微指标，供逆问题的 defect
    罚项使用 —— 即"粉末尺度缺陷"进入工艺优化目标的通路。

    Attributes
    ----------
    packing_density : 标量
        铺粉体积分数 ∈ [0,1]（典型 SLM 粉床 0.50~0.62；越低越易未熔合）。
    coordination_number : 标量
        平均配位数 [-]（颗粒平均接触数，密实床 ~6~8）。
    surface_roughness : 标量
        铺粉/熔道表面粗糙度 Ra [m]（下一层铺粉的初始不平度）。
    balling_indicator : 标量
        球化风险 ∈ [0,1]（Plateau–Rayleigh 判据 L/W > π）。
    lof_indicator : 标量
        未熔合风险 ∈ [0,1]（铺粉密度 + 层厚/熔深 + 搭接率）。
    spatter_fraction : 标量
        飞溅质量分数 ∈ [0,1]（被抛出粉床的质量 / 总质量）。
    denudation_width : 标量
        剥蚀区半宽 [m]（熔道两侧被气流卷走粉末的宽度）。
    porosity : 标量
        粉末尺度孔隙率 ∈ [0,1]（未熔合孔 + 夹带气孔）。
    dim : int
        维度（静态）。
    method : str
        产生本结果的颗粒方法（``"surrogate"`` / ``"dem"`` / ``"sph"`` /
        ``"mpm"``，静态，仅供追溯）。
    """

    packing_density: Array
    coordination_number: Array
    surface_roughness: Array
    balling_indicator: Array
    lof_indicator: Array
    spatter_fraction: Array
    denudation_width: Array
    porosity: Array
    dim: int = 3
    method: str = "surrogate"

    def defect_score(self) -> Array:
        """粉末尺度综合缺陷评分 ∈ [0,1]，越大越差。

        权重取向与 :meth:`MeltPoolResult.defect_score` 一致（未熔合权重最高，
        因为它对疲劳寿命最致命），但物理来源完全不同：这里来自颗粒尺度。
        """
        return jnp.clip(
            0.45 * self.lof_indicator
            + 0.35 * self.balling_indicator
            + 0.20 * self.spatter_fraction, 0.0, 1.0)

    def recoat_quality(self) -> Array:
        """铺粉质量 ∈ [0,1]，越大越好（密实且平整）。

        以典型可铺粉下限 0.45 为零点、理想随机密堆 0.64 为满分线性映射，
        再按粗糙度相对粉层尺度折减。
        """
        dens = jnp.clip((self.packing_density - 0.45) / (0.64 - 0.45), 0.0, 1.0)
        rough = 1.0 / (1.0 + self.surface_roughness / 3e-5)
        return dens * rough


register_contract(
    PowderBedResult,
    ("packing_density", "coordination_number", "surface_roughness",
     "balling_indicator", "lof_indicator", "spatter_fraction",
     "denudation_width", "porosity"),
    ("dim", "method"),
)


# ===========================================================================
# 3. 热历史契约
# ===========================================================================
@dataclass(frozen=True)
class ThermalHistory:
    """热历史 —— 连接"工艺"与"微观组织/残余应力"的枢纽契约。

    Attributes
    ----------
    peak_temperature : (...,) 栅格
        每点经历的峰值温度 [K]。
    cooling_rate : (...,)
        凝固区间平均冷却速率 [K/s]（正值）。
    thermal_gradient : (...,)
        凝固前沿温度梯度 G [K/m]。
    solidification_rate : (...,)
        凝固前沿推进速率 R [m/s]。
    time_above_melt : (...,)
        高于熔点的累计时间 [s]（重熔次数的代理量）。
    final_temperature : (...,)
        模拟结束时刻温度场 [K]。
    temperature_evolution : (n_frames, ...) 或 None
        扫描过程温度场时间演化帧（opt-in，由求解器 ``record_frames`` 控制）。
        仅在显式开启帧记录时存在；用于后处理动画播放 / ParaView 时间序列导出。
        默认 None：不影响任何既有计算与契约校验。
    spacing : float
        栅格体素尺寸 [m]。
    dim : int
        维度（静态）。
    """

    peak_temperature: Array
    cooling_rate: Array
    thermal_gradient: Array
    solidification_rate: Array
    time_above_melt: Array
    final_temperature: Array
    temperature_evolution: Array | None = None
    spacing: float = 1e-5
    dim: int = 3

    def gr_ratio(self) -> Array:
        """G/R —— 凝固形貌判据（大 → 柱状晶，小 → 等轴晶）。"""
        return self.thermal_gradient / jnp.maximum(self.solidification_rate, 1e-12)

    def cooling_rate_gr(self) -> Array:
        """G·R = 冷却速率 [K/s]，控制枝晶臂间距与晶粒细化程度。"""
        return self.thermal_gradient * self.solidification_rate


register_contract(
    ThermalHistory,
    ("peak_temperature", "cooling_rate", "thermal_gradient",
     "solidification_rate", "time_above_melt", "final_temperature"),
    ("spacing", "dim"),
)


# ===========================================================================
# 4. 熔池契约
# ===========================================================================
@dataclass(frozen=True)
class MeltPoolResult:
    """熔池多物理场求解结果（Flow3D 式自由界面 CFD 的输出）。

    Attributes
    ----------
    vof : (...,)
        金属体积分数 F ∈ [0,1]（VOF 自由界面）。
    temperature : (...,)
        温度场 [K]。
    liquid_fraction : (...,)
        液相分数 f_l ∈ [0,1]。
    velocity : (..., dim)
        流场速度 [m/s]。
    pressure : (...,)
        压力 [Pa]。
    depth / width / length : 标量
        熔池深度 / 宽度 / 长度 [m]。
    keyhole_depth : 标量
        匙孔深度 [m]（蒸发反冲压凹陷）。
    lof_indicator : 标量
        未熔合 (lack-of-fusion) 风险指标 ∈ [0,1]。
    porosity_indicator : 标量
        匙孔气孔风险指标 ∈ [0,1]。
    spatter_indicator : 标量
        飞溅风险指标 ∈ [0,1]。
    """

    vof: Array
    temperature: Array
    liquid_fraction: Array
    velocity: Array
    pressure: Array
    depth: Array
    width: Array
    length: Array
    keyhole_depth: Array
    lof_indicator: Array
    porosity_indicator: Array
    spatter_indicator: Array
    spacing: float = 5e-6
    dim: int = 3

    def aspect_ratio(self) -> Array:
        """熔池深宽比 —— 传导模式 (<0.5) vs 匙孔模式 (>~1) 的判别量。"""
        return self.depth / jnp.maximum(self.width, 1e-12)

    def defect_score(self) -> Array:
        """综合缺陷评分 ∈ [0,1]，越大越差（用于工艺窗口筛选）。"""
        return jnp.clip(
            0.45 * self.lof_indicator
            + 0.40 * self.porosity_indicator
            + 0.15 * self.spatter_indicator, 0.0, 1.0)


register_contract(
    MeltPoolResult,
    ("vof", "temperature", "liquid_fraction", "velocity", "pressure",
     "depth", "width", "length", "keyhole_depth",
     "lof_indicator", "porosity_indicator", "spatter_indicator"),
    ("spacing", "dim"),
)


# ===========================================================================
# 5. 微观组织契约
# ===========================================================================
@dataclass(frozen=True)
class MicrostructureResult:
    """微观组织（相场 / CA 求解结果 + 统计特征）。

    Attributes
    ----------
    phi : (...,)
        固相序参量 ∈ [0,1]。
    orientation : (...,)
        晶粒取向角 [rad]。
    grain_size : (...,)
        局部晶粒尺寸场 [m]。
    columnar_fraction : (...,)
        柱状晶体积分数 ∈ [0,1]（其余为等轴晶）。
    texture_intensity : (...,)
        织构强度 ∈ [0,1]（0=随机取向，1=单晶）。
    porosity : (...,)
        孔隙率 ∈ [0,1]。
    mean_grain_size : 标量
        体积平均晶粒尺寸 [m]。
    """

    phi: Array
    orientation: Array
    grain_size: Array
    columnar_fraction: Array
    texture_intensity: Array
    porosity: Array
    mean_grain_size: Array
    spacing: float = 1e-6
    dim: int = 3

    def hall_petch_strength(self, sigma_0: float, k_hp: float) -> Array:
        """Hall-Petch 关系：σ_y = σ_0 + k / sqrt(d)  [Pa]。"""
        d = jnp.maximum(self.grain_size, 1e-9)
        return sigma_0 + k_hp / jnp.sqrt(d)


register_contract(
    MicrostructureResult,
    ("phi", "orientation", "grain_size", "columnar_fraction",
     "texture_intensity", "porosity", "mean_grain_size"),
    ("spacing", "dim"),
)


# ===========================================================================
# 6. 本构契约 —— 微观模拟的最终产物、宏观仿真的输入
# ===========================================================================
@dataclass(frozen=True)
class ConstitutiveField:
    """**空间分辨的各向异性弹塑性本构参数场**。

    这是可行性报告点明的"关键断点"（微观组织 → 零件级本构）的产物形态：
    不是一组标量材料常数，而是**逐点的本构参数场**，因此能表达增材件
    固有的各向异性与空间不均匀性。

    弹性：横观各向同性（构建方向 Z 为对称轴），由 (E_x, E_z, nu, G_xz) 描述。
    塑性：Hill48 各向异性屈服 + Voce 硬化。

    Attributes
    ----------
    E_inplane, E_build : (...,)
        面内 / 构建方向弹性模量 [Pa]。
    nu : (...,)
        泊松比 [-]。
    G_build : (...,)
        面外剪切模量 [Pa]。
    sigma_y0 : (...,)
        初始屈服强度 [Pa]。
    hardening_sat : (...,)
        Voce 饱和硬化增量 [Pa]。
    hardening_rate : (...,)
        Voce 硬化速率 [-]。
    hill_F, hill_G, hill_H, hill_L, hill_M, hill_N : (...,)
        Hill48 各向异性系数 [-]。
    porosity : (...,)
        孔隙率 [-]（用于刚度/强度折减）。
    damage : (...,)
        初始损伤 ∈ [0,1]。
    """

    E_inplane: Array
    E_build: Array
    nu: Array
    G_build: Array
    sigma_y0: Array
    hardening_sat: Array
    hardening_rate: Array
    hill_F: Array
    hill_G: Array
    hill_H: Array
    hill_L: Array
    hill_M: Array
    hill_N: Array
    porosity: Array
    damage: Array
    spacing: float = 1e-4
    dim: int = 3

    # ---- 派生 ------------------------------------------------------------
    def effective_stiffness(self) -> Array:
        """孔隙+损伤折减后的等效模量（Mori-Tanaka 型简化）[Pa]。"""
        red = (1.0 - self.porosity) ** 2 * (1.0 - self.damage)
        return jnp.cbrt(self.E_inplane ** 2 * self.E_build) * red

    def anisotropy_ratio(self) -> Array:
        """各向异性度 E_z / E_x（增材件典型 0.85~1.0）。"""
        return self.E_build / jnp.maximum(self.E_inplane, 1e-9)

    def homogenized(self) -> dict[str, float]:
        """体积平均后的标量本构（供快速宏观仿真/报表）。"""
        m = lambda a: float(jnp.mean(a))  # noqa: E731
        return {
            "E_inplane": m(self.E_inplane), "E_build": m(self.E_build),
            "nu": m(self.nu), "G_build": m(self.G_build),
            "sigma_y0": m(self.sigma_y0),
            "hardening_sat": m(self.hardening_sat),
            "hardening_rate": m(self.hardening_rate),
            "porosity": m(self.porosity),
            "anisotropy_ratio": m(self.anisotropy_ratio()),
        }


register_contract(
    ConstitutiveField,
    ("E_inplane", "E_build", "nu", "G_build", "sigma_y0",
     "hardening_sat", "hardening_rate",
     "hill_F", "hill_G", "hill_H", "hill_L", "hill_M", "hill_N",
     "porosity", "damage"),
    ("spacing", "dim"),
)


# ===========================================================================
# 7. As-built 构型契约 —— "模拟后的构型"，用于与理想构型比较
# ===========================================================================
@dataclass(frozen=True)
class AsBuiltPart:
    """**制造后实际构型** + 残余应力/应变状态。

    这是可微闭环 (功能2) 中与"理想构型"作比较的对象：
    几何尺寸差异 + 残余应力 + 残余应变共同构成损失函数。

    Attributes
    ----------
    sdf : (...,)
        变形后（含热收缩/翘曲）的带符号距离场 [m]。
    displacement : (..., dim)
        相对名义几何的位移场 [m]。
    residual_stress : (..., n_comp)
        残余应力张量分量 [Pa]（Voigt: 3D=6, 2D=3）。
    residual_strain : (..., n_comp)
        残余塑性/热应变分量 [-]。
    """

    sdf: Array
    displacement: Array
    residual_stress: Array
    residual_strain: Array
    spacing: float = 1e-4
    dim: int = 3

    def soft_occupancy(self, eps: float | None = None) -> Array:
        if eps is None:
            eps = float(self.spacing)
        return 0.5 * (1.0 - jnp.tanh(self.sdf / jnp.maximum(eps, 1e-12)))

    def von_mises_residual(self) -> Array:
        """残余应力的 von Mises 等效值 [Pa]。

        注意：当输入残余应力为 0（零件外 / 平衡处）时，sqrt 的参数为 0，
        其逆向导数 ``1/(2√x)`` 会炸成 Inf/NaN，进而污染整条可微链。
        因此给参数加非负下界 + 极小护垫，使梯度在零点处有限。
        """
        s = self.residual_stress
        if self.dim == 3:
            sxx, syy, szz, sxy, syz, szx = [s[..., i] for i in range(6)]
            arg = (0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2
                         + (szz - sxx) ** 2)
                   + 3.0 * (sxy ** 2 + syz ** 2 + szx ** 2))
            return jnp.sqrt(jnp.maximum(arg, 0.0) + 1e-30)
        sxx, syy, sxy = s[..., 0], s[..., 1], s[..., 2]
        arg = sxx ** 2 - sxx * syy + syy ** 2 + 3.0 * sxy ** 2
        return jnp.sqrt(jnp.maximum(arg, 0.0) + 1e-30)

    def max_distortion(self) -> Array:
        """最大变形量 [m]。"""
        return jnp.max(jnp.linalg.norm(self.displacement, axis=-1))


register_contract(
    AsBuiltPart,
    ("sdf", "displacement", "residual_stress", "residual_strain"),
    ("spacing", "dim"),
)


# ===========================================================================
# 7.5 装配体契约（P2-③：多体 + FEM 耦合）—— digitaltwin/verdict 的新输入
# ===========================================================================
@dataclass(frozen=True)
class AssemblyResult:
    """**装配体（多体动力学 + FEM 耦合）求解结果**，P2-③ 高保真路径产物。

    把增材件与其支撑/基座建模为铰接链；**FEM 子模型（AsBuiltPart 变形场）提供
    关节柔度**，MBD 提供广义质量与载荷，二者在子模型边界处耦合
    （界面位移 ↔ 关节载荷）。本契约是数字样机 / 服役判定的新输入。

    Attributes
    ----------
    joint_load : (n_joints,) 关节反力 / 力矩 [N] 或 [N·m]
    relative_displacement : (n_joints,) 关节相对位移 / 转角 [m] 或 [rad]
    flexural_stiffness : (n_joints,) FEM 子模型提供的关节弯曲刚度 [N·m/rad]
    stability_margin : 标量 线性化稳定裕度（>0 稳定；趋近 0 = 临界失稳）
    porosity : 标量 从熔池 / 本构继承的孔隙风险指标 ∈ [0,1]
    dim : int
    """

    joint_load: Array
    relative_displacement: Array
    flexural_stiffness: Array
    stability_margin: Array
    porosity: Array
    dim: int = 3

    def assembly_score(self) -> Array:
        """装配质量综合评分 ∈ [0,1]，越大越可靠（稳定且低孔隙）。"""
        return jnp.clip(self.stability_margin, 0.0, 1.0) * (1.0 - self.porosity)


register_contract(
    AssemblyResult,
    ("joint_load", "relative_displacement", "flexural_stiffness",
     "stability_margin", "porosity"),
    ("dim",),
)


# ===========================================================================
# 8. 结构响应 & 服役判定契约
# ===========================================================================
@dataclass(frozen=True)
class StructuralResult:
    """数字样机中增材件的结构响应。"""

    displacement: Array
    stress: Array           # (n_cells, n_comp) Voigt [Pa]
    strain: Array
    von_mises: Array        # (n_cells,) [Pa]
    reaction: Array         # 约束反力 [N]
    dim: int = 3

    def peak_stress(self) -> Array:
        return jnp.max(self.von_mises)


register_contract(
    StructuralResult,
    ("displacement", "stress", "strain", "von_mises", "reaction"),
    ("dim",),
)


@dataclass(frozen=True)
class ServiceVerdict:
    """数字化试验的**最终判定**：增材件在服役工况下是否合格。"""

    strength_safety_factor: Array     # 强度安全系数 σ_y / σ_max
    stiffness_ratio: Array            # 变形/许用变形
    fatigue_life_cycles: Array        # 估算疲劳寿命 [循环]
    wear_depth: Array                 # 预测磨损深度 [m]
    critical_location: Array          # 最危险点坐标 [m]
    passed: Array                     # 0/1 是否通过（可微软判定）
    margin_report: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "═" * 62,
            "  增材制造件数字化试验判定报告",
            "═" * 62,
            f"  强度安全系数      : {float(self.strength_safety_factor):8.3f}",
            f"  刚度利用率        : {float(self.stiffness_ratio):8.3f}  (<1 合格)",
            f"  估算疲劳寿命      : {float(self.fatigue_life_cycles):8.3e} 循环",
            f"  预测磨损深度      : {float(self.wear_depth) * 1e6:8.3f} µm",
            f"  最危险点坐标      : "
            f"{np.array2string(np.asarray(self.critical_location), precision=4)}",
            "─" * 62,
            f"  判定结果          : "
            f"{'✅ 合格 (PASS)' if float(self.passed) > 0.5 else '❌ 不合格 (FAIL)'}",
            "═" * 62,
        ]
        for k, v in self.margin_report.items():
            lines.append(f"  · {k:26s}: {v:12.4g}")
        return "\n".join(lines)


# ServiceVerdict 不能走通用的 register_contract：它带一个 dict 字段
# margin_report（非张量叶子），必须自定义 flatten 把 dict 塞进 aux_data。
# 注意 aux_data 需可哈希，故转成排序后的 tuple。
jax.tree_util.register_pytree_node(
    ServiceVerdict,
    lambda o: (
        (o.strength_safety_factor, o.stiffness_ratio,
         o.fatigue_life_cycles, o.wear_depth,
         o.critical_location, o.passed,
         *tuple(o.margin_report.values())),
        (tuple(o.margin_report.keys()),),
    ),
    lambda aux, leaves: ServiceVerdict(
        *leaves[:6],
        margin_report=dict(zip(aux[0], leaves[6:]))),
)


# ===========================================================================
# 9. 材料标定报告契约（Module C：材料标定离线工具的输出）
# ===========================================================================
@dataclass(frozen=True)
class CalibrationReport:
    """**材料标定报告**：用实测熔池尺寸反演材料物性后的可信度凭证。

    这是用户"**实打实物理 + 误差报告**"硬约束在材料维度的落点：高保真求解器
    （vof_flow3d / enthalpy / phasefield / thermomechanical_plastic）必须给出
    "误差 ≤ X%" 的实证，否则链条再炫也不可信。本契约就是把那次实证的结论
    （拟合材料 + 逐实验残差 + RMSE + R²）打包成可序列化、可比较的对象。

    Attributes
    ----------
    base_material_name : str
        标定起点的材料名（如 ``"316L"``）；拟合只在其 ``fit_keys`` 子集上动。
    fit_keys : tuple[str, ...]
        被标定的物性字段名（如 ``("k_solid", "rho_solid", "latent_fusion")``）。
    fitted_material : AMMaterial
        标定后的材料（仅在 fit_keys 上与原材料不同，其余字段沿用）。
    experiment_ids : tuple[str, ...]
        每个实验的标签（与 measured/predicted 数组逐行对应）。
    measured_depth / predicted_depth_before / predicted_depth_after : (n,) 数组
        实测熔深；标定前/后仿真预测熔深 [m]。
    measured_width / predicted_width_before / predicted_width_after : (n,) 数组
        实测熔宽；标定前/后仿真预测熔宽 [m]。
    rmse_depth / rmse_width : 标量
        标定后熔深/熔宽的均方根误差 [m]。
    r2_depth / r2_width : 标量
        标定后熔深/熔宽的判定系数（1 = 完美拟合）。
    n_iter : int
        优化迭代步数。
    converged : bool
        优化器是否报告收敛。
    cost : 标量
        标定后最终损失（归一化残差平方均值）。
    """

    base_material_name: str
    fit_keys: tuple
    fitted_material: AMMaterial
    experiment_ids: tuple
    measured_depth: Array
    predicted_depth_before: Array
    predicted_depth_after: Array
    measured_width: Array
    predicted_width_before: Array
    predicted_width_after: Array
    rmse_depth: Array
    rmse_width: Array
    r2_depth: Array
    r2_width: Array
    n_iter: int
    converged: bool
    cost: Array

    def summary(self) -> str:
        lines = [
            "═" * 64,
            "  材料标定报告 (CalibrationReport)",
            "═" * 64,
            f"  基准材料          : {self.base_material_name}",
            f"  标定物性          : {', '.join(self.fit_keys)}",
            f"  实验数            : {len(self.experiment_ids)}",
            f"  优化器收敛        : {'是' if self.converged else '否'}  (iter={self.n_iter})",
            f"  最终损失          : {float(self.cost):.4e}",
            "─" * 64,
            f"  熔深 RMSE         : {float(self.rmse_depth) * 1e6:10.3f} µm   "
            f"R² = {float(self.r2_depth):.4f}",
            f"  熔宽 RMSE         : {float(self.rmse_width) * 1e6:10.3f} µm   "
            f"R² = {float(self.r2_width):.4f}",
            "─" * 64,
            "  拟合后物性:",
        ]
        for k in self.fit_keys:
            lines.append(f"    · {k:<16s}: {float(getattr(self.fitted_material, k)):.6g}")
        lines.append("═" * 64)
        return "\n".join(lines)


# CalibrationReport 不能走通用 register_contract：它同时含 (n,) 数组叶子与
# 一个 AMMaterial 对象（frozen dataclass，全字段 float/str → 可哈希）、
# 字符串/元组/布尔等静态元数据。自定义 flatten 把数组放叶子、其余放 aux。
jax.tree_util.register_pytree_node(
    CalibrationReport,
    lambda o: (
        (o.measured_depth, o.predicted_depth_before, o.predicted_depth_after,
         o.measured_width, o.predicted_width_before, o.predicted_width_after,
         o.rmse_depth, o.rmse_width, o.r2_depth, o.r2_width, o.cost),
        (o.base_material_name, o.fit_keys, o.fitted_material,
         o.experiment_ids, o.n_iter, o.converged),
    ),
    lambda aux, leaves: CalibrationReport(
        base_material_name=aux[0], fit_keys=aux[1], fitted_material=aux[2],
        experiment_ids=aux[3], n_iter=aux[4], converged=aux[5],
        measured_depth=leaves[0], predicted_depth_before=leaves[1],
        predicted_depth_after=leaves[2], measured_width=leaves[3],
        predicted_width_before=leaves[4], predicted_width_after=leaves[5],
        rmse_depth=leaves[6], rmse_width=leaves[7],
        r2_depth=leaves[8], r2_width=leaves[9], cost=leaves[10],
    ),
)


# ===========================================================================
# 10. 支撑结构契约（模块A：SLM 支撑自动生成与仿真）
# ===========================================================================
@dataclass(frozen=True)
class SupportStructure:
    """**支撑结构**：SLM 悬垂/薄壁区域自动生成的支撑体（block / overhang 等）。

    SLM 缺少支撑时悬垂面会塌陷、翘曲、失败——这是当前链路在 ``geometry`` 之后
    直接断到 ``powderbed`` 的最大缺口。本契约把"支撑"显式建模为一个与零件同栅格
    的占位/符号场，使后续铺粉、热-力耦合、变形修正都能把它作为**额外约束/散热边界**
    纳入计算。

    Attributes
    ----------
    support_sdf : (...,)
        支撑体的带符号距离场 [m]（负值在支撑内部，与 ``PartGeometry.sdf`` 同栅格、
        同轴，便于做并集 ``min(part_sdf, support_sdf)`` 得到「零件 ∪ 支撑」几何）。
    support_mask : (...,)
        支撑占位掩膜 ∈ [0,1]（1 = 支撑实体），是 ``support_sdf < 0`` 的硬/软占位。
    kind : str
        支撑类型（``"block"`` 整足迹块支撑 / ``"overhang"`` 仅悬垂下方支撑 /
        ``"line"`` 线支撑 / ``"cone"`` 锥支撑 / ``"skin"`` 轮廓皮支撑），静态。
    contact_area : 标量
        支撑与零件的接触面积 [m²]（决定散热与机械约束强度），静态元数据。
    volume_fraction : 标量
        支撑体积占 bounding-box 体积比 ∈ [0,1]（影响材料/时间成本），静态。
    params : dict
        生成参数（overhang_angle_deg / density / spacing 等），静态（aux）。
    """

    support_sdf: Array
    support_mask: Array
    kind: str = "block"
    contact_area: Array = None  # type: ignore[assignment]
    volume_fraction: Array = None  # type: ignore[assignment]
    params: dict = field(default_factory=dict)

    def contact_ratio(self) -> Array:
        """接触面积相对包围盒底面积的比（越大约束越强），∈ [0,1]。"""
        base = jnp.maximum(self.volume_fraction, 1e-12)
        return jnp.clip(self.contact_area / (base + 1e-12), 0.0, 1.0)

    def summary(self) -> str:
        return (
            f"SupportStructure(kind={self.kind}, "
            f"volume_fraction={float(self.volume_fraction):.4f}, "
            f"contact_area={float(self.contact_area):.4e} m^2)"
        )


# SupportStructure 的 pytree：数组叶子 = (support_sdf, support_mask,
# contact_area, volume_fraction)；aux 仅放真正静态、可哈希的元数据
# （kind 字符串 + params 字典的 (key,value) 元组）。注意 aux 必须可哈希，
# 因此 contact_area / volume_fraction 当作叶子（它们是标量数组），不能进 aux。
jax.tree_util.register_pytree_node(
    SupportStructure,
    lambda o: (
        (o.support_sdf, o.support_mask, o.contact_area, o.volume_fraction),
        (o.kind, tuple((k, v) for k, v in (o.params or {}).items())),
    ),
    lambda aux, leaves: SupportStructure(
        support_sdf=leaves[0], support_mask=leaves[1],
        contact_area=leaves[2], volume_fraction=leaves[3],
        kind=aux[0], params=dict(aux[1]),
    ),
)


# ===========================================================================
# 11. 二次工艺结果契约（模块B：热处理 / HIP / 切削）
# ===========================================================================
@dataclass(frozen=True)
class SecondaryProcessResult:
    """**二次工艺结果**：站在 ``AsBuiltPart`` 之上的热处理 / HIP / 切削处理后产物。

    SLM 成形件（``buildup / structural`` 之后）通常还需一道二次工艺才交付：
    去应力退火（HT）释放残余应力、热等静压（HIP）闭合内部孔隙、切削（machining）
    去除余料并给最终形位。这道环节此前在链路里直接断到 ``verdict``，缺中间再处理，
    导致"残余应力被高估、孔隙被低估、最终尺寸偏差不可见"。本契约把二次工艺的
    **真实物理效应**打包：

    * ``part``：处理后成形件（仍是 :class:`AsBuiltPart`，含处理后的残余应力/
      位移/应变场）；HT/HIP 对残余应力做松弛，machining 对几何做修形。
    * ``treatment``：工艺标签（``"HT"`` / ``"HIP"`` / ``"machining"`` / ``"none"``）。
    * ``residual_relief_factor``：残余应力保留比例 ∈ (0,1]（1=无松弛，越小释放越多）。
    * ``density_after``：处理后相对密度 ∈ (0,1]（HIP 把孔隙闭向 ~1）。
    * ``geometry_after``：处理后 SDF（machining 为「名义 ∩ 成形」的修形结果）。
    * ``applied_processes``：已施加工艺标签元组（可追溯）。

    ``verdict.service`` 现可选消费本契约：有则按处理后件评定（残余应力按松弛、
    孔隙按密度修正），无（passthrough）则等价于直接用成形件。
    """

    part: AsBuiltPart
    treatment: str = "none"
    residual_relief_factor: Array = None  # type: ignore[assignment]
    density_after: Array = None  # type: ignore[assignment]
    geometry_after: Array = None  # type: ignore[assignment]
    applied_processes: tuple = ()
    params: dict = field(default_factory=dict)

    def relief_ratio(self) -> Array:
        """残余应力保留比例（缺省按 1.0 视为未处理）。"""
        if self.residual_relief_factor is None:
            return jnp.asarray(1.0, dtype=jnp.float64)
        return jnp.asarray(self.residual_relief_factor, dtype=jnp.float64)

    def density(self) -> Array:
        """处理后相对密度（缺省按 1.0 视为全致密）。"""
        if self.density_after is None:
            return jnp.asarray(1.0, dtype=jnp.float64)
        return jnp.asarray(self.density_after, dtype=jnp.float64)

    def summary(self) -> str:
        return (
            f"SecondaryProcessResult(treatment={self.treatment}, "
            f"relief={float(self.relief_ratio()):.3f}, "
            f"density={float(self.density()):.4f})"
        )


# SecondaryProcessResult 的 pytree：把 part(AsBuiltPart) 的 4 个数组字段展开为叶子，
# 其余为静态元数据（treatment/applied_processes/params 可哈希、spacing/dim 来自 part）。
# 这样整张契约图（含 AsBuiltPart）可被 jax.grad/jit 穿透。
jax.tree_util.register_pytree_node(
    SecondaryProcessResult,
    lambda o: (
        (o.part.sdf, o.part.displacement, o.part.residual_stress,
         o.part.residual_strain, o.residual_relief_factor, o.density_after,
         o.geometry_after),
        (o.part.spacing, o.part.dim, o.treatment, tuple(o.applied_processes),
         tuple((k, v) for k, v in (o.params or {}).items())),
    ),
    lambda aux, leaves: SecondaryProcessResult(
        part=AsBuiltPart(
            sdf=leaves[0], displacement=leaves[1],
            residual_stress=leaves[2], residual_strain=leaves[3],
            spacing=aux[0], dim=aux[1]),
        treatment=aux[2], residual_relief_factor=leaves[4],
        density_after=leaves[5], geometry_after=leaves[6],
        applied_processes=tuple(aux[3]), params=dict(aux[4]),
    ),
)


# ===========================================================================
# 1y. 监测闭环契约 —— 模块 D（在线监测 + 工艺修正闭环）
# ===========================================================================
@dataclass(frozen=True)
class SensorData:
    """在线监测**传感数据流**：熔池影像 / 光电 / 热成像的时序信号。

    多通道时序传感信号（每通道一张 ``(n_frames, ...)`` 帧序列），是缺陷检测
    求解器 ``monitor.detect`` 的输入；也可经 ``ingest_raw_stream`` 由原始字节流
    封装而来。

    Attributes
    ----------
    frames : (n_channels, n_frames, ...) 或 (n_frames, H, W)
        多通道传感帧序列。通道顺序对应 ``channels``（如 0=光电、1=热成像、2=熔池影像）。
    time : (n_frames,)
        各帧时间戳 [s]，单调递增。
    channels : tuple[str, ...]
        信号通道名（静态）。如 ``("photodiode", "thermal", "meltpool")``。
    dt : float
        标称采样间隔 [s]（静态）。
    geometry_name : str
        关联零件名（静态），用于与几何契约对齐。
    sensor_meta : dict
        传感器/采集元数据（静态，标量值）：相机分辨率、标定矩阵、采样率等。
    """

    frames: Array
    time: Array
    channels: tuple = ()
    dt: float = 0.0
    geometry_name: str = "part"
    sensor_meta: dict = field(default_factory=dict)

    def n_frames(self) -> int:
        """帧数（取 time 长度）。"""
        return int(jnp.atleast_1d(self.time).shape[0])

    def summary(self) -> str:
        ch = ",".join(self.channels) if self.channels else "?"
        return (f"SensorData(frames={tuple(self.frames.shape)}, "
                f"n={self.n_frames()}, channels={ch})")


@dataclass(frozen=True)
class MonitoringState:
    """缺陷**监测状态**：缺陷类型概率 + 检测置信度。

    由 ``monitor.detect`` 从传感流推断，是 ``monitor.correct`` 修正工艺的输入；
    亦可作为质量指标汇入服役判定 ``ServiceVerdict``。

    Attributes
    ----------
    defect_probs : (n_defect_types,) 或 (n_frames, n_defect_types)
        各缺陷类型的发生概率（0~1）。
    defect_names : tuple[str, ...]
        缺陷类型名（静态）。默认 ``("lack_of_fusion", "keyhole", "porosity")``
        （未熔合/匙孔/孔隙）。
    confidence : (n_defect_types,) 或 (n_frames, n_defect_types)
        检测置信度（0~1）。缺省置 1。
    position : (n_frames, dim) 或 (n_frames,)
        缺陷对应的空间/时间位置（缺省置 0）。
    sensor_ref : str
        来源传感流引用（静态）。
    model_meta : dict
        检测器模型元数据（静态，标量值）：模型名、阈值、版本。
    """

    defect_probs: Array
    defect_names: tuple = ("lack_of_fusion", "keyhole", "porosity")
    confidence: Array = None  # type: ignore[assignment]
    position: Array = None  # type: ignore[assignment]
    sensor_ref: str = "stream"
    model_meta: dict = field(default_factory=dict)

    def max_defect(self) -> tuple[str, Array]:
        """返回概率最高的缺陷类型名与其概率。"""
        probs = jnp.atleast_1d(self.defect_probs)
        idx = int(jnp.argmax(probs))
        names = self.defect_names if self.defect_names else ("defect",)
        return names[min(idx, len(names) - 1)], probs[idx]

    def summary(self) -> str:
        name, p = self.max_defect()
        return (f"MonitoringState(top={name}, p={float(p):.3f}, "
                f"types={len(self.defect_names)})")


@dataclass(frozen=True)
class ClosedLoopPlan:
    """闭环**修正后工艺方案**：在基准 ``ProcessPlan`` 上叠加修正量。

    由 ``monitor.correct`` 产出，是下一轮前向仿真的输入；``closedloop.run_loop``
    反复把它喂回 ``Pipeline`` 直到质量达标。

    Attributes
    ----------
    base : ProcessPlan
        闭环起始（基准）工艺方案（可微叶子）。
    corrected : ProcessPlan
        当前迭代修正后的工艺方案（可微叶子）。
    deltas : tuple
        修正量（静态）：``((param_name, value), ...)``，value 为标量。
    iteration : int
        闭环迭代序号（静态）。
    rationale : str
        本轮修正理由（静态）。
    controller : str
        生成此方案所用的控制器类型：``"rule"``（物理规则代理）或
        ``"inverse"``（inverse 可微优化器实打实控制）。
    """

    base: ProcessPlan
    corrected: ProcessPlan
    deltas: tuple = ()
    iteration: int = 0
    rationale: str = ""
    controller: str = "rule"


# 监测闭环契约的 pytree 注册。
# 约定：frozen dataclass 中"可微叶子"为 JAX 数组，"静态元数据"为 hashable 值。
# dict 字段（sensor_meta / model_meta / params / deltas）一律以 ``tuple(items())``
# 形式存为静态 aux 并在重建时还原为 dict，以满足 jit 对 treedef 的 hashable 要求
# （与 SecondaryProcessResult.params 的处理一致）。
def _sensor_flatten(o):
    return ((o.frames, o.time),
            (tuple(o.channels), o.dt, o.geometry_name,
             tuple((k, v) for k, v in (o.sensor_meta or {}).items())))


def _sensor_unflatten(aux, leaves):
    return SensorData(frames=leaves[0], time=leaves[1],
                      channels=tuple(aux[0]), dt=aux[1], geometry_name=aux[2],
                      sensor_meta=dict(aux[3]))


def _monitor_flatten(o):
    return ((o.defect_probs, o.confidence, o.position),
            (tuple(o.defect_names), o.sensor_ref,
             tuple((k, v) for k, v in (o.model_meta or {}).items())))


def _monitor_unflatten(aux, leaves):
    return MonitoringState(defect_probs=leaves[0], defect_names=tuple(aux[0]),
                           confidence=leaves[1], position=leaves[2],
                           sensor_ref=aux[1], model_meta=dict(aux[2]))


def _closedloop_flatten(o):
    # base / corrected 均为已注册的 ProcessPlan pytree，递归展开为叶子。
    return ((o.base, o.corrected),
            (o.iteration, o.rationale, o.controller,
             tuple((k, float(v)) for k, v in o.deltas)))


def _closedloop_unflatten(aux, leaves):
    return ClosedLoopPlan(base=leaves[0], corrected=leaves[1],
                          iteration=aux[0], rationale=aux[1], controller=aux[2],
                          deltas=tuple(aux[3]))


jax.tree_util.register_pytree_node(SensorData, _sensor_flatten, _sensor_unflatten)
jax.tree_util.register_pytree_node(MonitoringState, _monitor_flatten, _monitor_unflatten)
jax.tree_util.register_pytree_node(ClosedLoopPlan, _closedloop_flatten, _closedloop_unflatten)


# ===========================================================================
# 契约注册表 —— 供 registry / graph 做类型解析
# ===========================================================================
#: **以契约类名为键**。registry / graph 的类型解析全部走这张表，
#: 因为求解器声明 ``consumes=("PartGeometry", ...)`` 用的是类名。
CONTRACTS: dict[str, type] = {
    cls.__name__: cls
    for cls in (PartGeometry, ProcessPlan, PowderBedResult, ThermalHistory,
                MeltPoolResult, MicrostructureResult, ConstitutiveField,
                AsBuiltPart, AssemblyResult, StructuralResult, ServiceVerdict,
                CalibrationReport, SupportStructure, SecondaryProcessResult,
                SensorData, MonitoringState, ClosedLoopPlan)
}

#: 以规范槽名为键的同一张表（便于按流水线上下文的 key 反查类型）。
CONTRACT_BY_SLOT: dict[str, type] = {
    "geometry": PartGeometry,
    "process": ProcessPlan,
    "support": SupportStructure,
    "powderbed": PowderBedResult,
    "thermal": ThermalHistory,
    "meltpool": MeltPoolResult,
    "microstructure": MicrostructureResult,
    "constitutive": ConstitutiveField,
    "asbuilt": AsBuiltPart,
    "assembly": AssemblyResult,
    "structural": StructuralResult,
    "secondary": SecondaryProcessResult,
    "verdict": ServiceVerdict,
    "sensor": SensorData,
    "monitoring": MonitoringState,
    "closedloop": ClosedLoopPlan,
}


__all__ = [
    "PartGeometry", "ProcessPlan", "PowderBedResult", "ThermalHistory",
    "MeltPoolResult", "MicrostructureResult", "ConstitutiveField",
    "AsBuiltPart", "AssemblyResult", "StructuralResult", "ServiceVerdict",
    "CalibrationReport", "SupportStructure", "SecondaryProcessResult",
    "SensorData", "MonitoringState", "ClosedLoopPlan",
    "CONTRACTS", "CONTRACT_BY_SLOT", "register_contract",
]
