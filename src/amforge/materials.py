"""AM 材料库 —— 熔池 CFD / 微观 / 本构 所需的全套物性
=======================================================

为什么要新建一个材料库？
------------------------
:mod:`diffmech.preprocess.materials` 已有力学与基础热学参数（E, ν, σy, k, ρ, cp），
但 Flow3D 式熔池模拟还需要一批**只在自由界面流动问题里才用到**的物性：

* 液态粘度 μ_l —— 决定 Marangoni 环流强度与熔池振荡衰减
* 表面张力 σ 与其温度系数 dσ/dT —— **Marangoni 对流的驱动源**，
  它的符号直接决定熔池是"宽而浅"还是"窄而深"，是熔池形貌的一阶控制因素
* 熔化潜热 L_f / 汽化潜热 L_v、沸点 T_b、摩尔质量 M ——
  反冲压力与匙孔 (keyhole) 形成的必要参数
* 固/液两侧分别的 k 与 cp —— 金属固液导热率能差 2~3 倍，用单值会明显失真
* 粉末堆积密度与粉床有效导热折减 —— SLM 粉床导热率只有致密体的 1%~10%，
  这是"为什么熔池热量憋在表层"的根本原因

数据说明（重要）
----------------
下列数值取自公开文献的**代表性区间中值**，用于基准算例与方法验证，
**不能用于适航/认证计算**。真实增材件物性对设备、批次、后处理极敏感。
每种合金都标注了主要文献来源，便于替换成自有实测数据。

所有物性都是 pytree 叶子 —— 因此可以对物性求梯度，
做**反演标定**（用实测熔池尺寸反推 dσ/dT 或吸收率，这是熔池模型标定的常规做法）。
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "AMMaterial",
    "AM_MATERIALS",
    "get_material",
    "list_materials",
    "material_table",
]


# ===========================================================================
# 材料物性容器
# ===========================================================================
@dataclass(frozen=True)
class AMMaterial:
    """一种增材合金的全套物性（SI 单位）。

    分组说明
    --------
    **相变温度**：``T_solidus`` / ``T_liquidus`` 给出糊状区(mushy zone)，
    ``T_boil`` 触发蒸发与反冲压力。

    **两相热物性**：固相与液相分别给值，求解时按液相分数线性混合。
    金属的 k 在熔化时通常跳变（固相靠电子+声子导热，液相以电子为主），
    对熔池深度影响显著。

    **表面张力**：``sigma0`` 是参考温度 ``T_sigma_ref`` 处的值，
    ``dsigma_dT`` 一般为负（洁净金属）。若合金含表面活性元素（如 S、O），
    dσ/dT 可能在某温度区间变正，导致 Marangoni 环流反向、熔池由浅变深——
    这是同样工艺参数下熔深散差大的经典原因之一。

    **粉末**：``packing_fraction`` 是粉床松装堆积率，
    ``powder_k_factor`` 是粉床有效导热率相对致密体的折减系数。
    """

    # -- 标识 --------------------------------------------------------------
    name: str = "generic"
    lattice: str = "fcc"          # fcc / bcc / hcp，供 CPFE 选滑移系
    reference: str = ""

    # -- 相变温度 [K] ------------------------------------------------------
    T_solidus: float = 1658.0
    T_liquidus: float = 1723.0
    T_boil: float = 3090.0
    T_ambient: float = 300.0

    # -- 密度 [kg/m³] ------------------------------------------------------
    rho_solid: float = 7950.0
    rho_liquid: float = 7000.0

    # -- 比热 [J/(kg·K)] ---------------------------------------------------
    cp_solid: float = 500.0
    cp_liquid: float = 800.0

    # -- 导热 [W/(m·K)] ----------------------------------------------------
    k_solid: float = 25.0         # 取接近熔点的高温值，而非室温值
    k_liquid: float = 25.0

    # -- 潜热 [J/kg] -------------------------------------------------------
    latent_fusion: float = 2.70e5
    latent_vapor: float = 7.45e6

    # -- 流动 --------------------------------------------------------------
    mu_liquid: float = 6.0e-3     # 动力粘度 [Pa·s]
    sigma0: float = 1.60          # 表面张力 [N/m] @ T_sigma_ref
    dsigma_dT: float = -4.3e-4    # 表面张力温度系数 [N/(m·K)]
    T_sigma_ref: float = 1723.0   # 表面张力参考温度 [K]
    beta_thermal: float = 1.0e-4  # 液态体积热膨胀系数 [1/K]（浮升力）

    # -- 光学 / 辐射 -------------------------------------------------------
    absorptivity: float = 0.35    # 1064 nm 附近的表观吸收率
    emissivity: float = 0.35
    molar_mass: float = 0.0559    # [kg/mol]，蒸发通量用

    # -- 粉末 --------------------------------------------------------------
    packing_fraction: float = 0.55
    powder_k_factor: float = 0.02  # 粉床 k_eff / k_bulk

    # -- 固体力学（供残余应力与本构模块） ---------------------------------
    E: float = 195e9              # 弹性模量 [Pa]
    nu: float = 0.28
    alpha_thermal: float = 16.5e-6  # 线膨胀系数 [1/K]
    sigma_y: float = 500e6        # 室温屈服强度 [Pa]
    hardening_sat: float = 350e6   # Voce 饱和硬化增量 [Pa]
    hardening_rate: float = 12.0   # Voce 硬化速率 [-]
    T_ref_mech: float = 300.0      # 力学参考温度（残余应力零点）

    # -- 微观组织 ----------------------------------------------------------
    hall_petch_sigma0: float = 220e6   # Hall-Petch σ0 [Pa]
    hall_petch_k: float = 0.55e6       # Hall-Petch k [Pa·m^0.5]
    gb_energy: float = 0.8             # 晶界能 [J/m²]
    nucleation_undercooling: float = 5.0   # 特征成核过冷度 [K]
    dendrite_gibbs_thomson: float = 3.0e-7  # Gibbs-Thomson 系数 [K·m]
    partition_coefficient: float = 0.9      # 溶质分配系数 k0 [-]

    # -- 晶体塑性 ----------------------------------------------------------
    crss0: float = 180e6          # 初始临界分切应力 [Pa]
    crss_sat: float = 420e6
    cp_hardening: float = 150e6
    rate_exponent: float = 0.05

    # ---- 派生量 ----------------------------------------------------------
    @property
    def T_melt(self) -> float:
        """糊状区中点温度 [K]（常用的"熔点"代表值）。"""
        return 0.5 * (self.T_solidus + self.T_liquidus)

    @property
    def mushy_range(self) -> float:
        """糊状区宽度 [K]。窄糊状区（如纯金属）数值上更硬，需要更小的 dt。"""
        return max(self.T_liquidus - self.T_solidus, 1.0)

    def diffusivity(self, liquid: bool = False) -> float:
        """热扩散系数 α = k/(ρ·cp) [m²/s]。"""
        if liquid:
            return self.k_liquid / (self.rho_liquid * self.cp_liquid)
        return self.k_solid / (self.rho_solid * self.cp_solid)

    def sigma_of_T(self, T):
        """表面张力随温度线性变化（并限幅非负）。"""
        return jnp.maximum(
            self.sigma0 + self.dsigma_dT * (T - self.T_sigma_ref), 1e-3
        )

    def liquid_fraction(self, T, *, smooth: float = 1.0):
        """液相分数 f_l(T) —— 糊状区内平滑过渡（可微）。

        用 tanh 而非线性斜坡：线性斜坡的 df_l/dT 是不连续的方波，
        在"表观比热法"里会让能量方程的雅可比不连续，牛顿迭代/梯度都变差。
        """
        w = 0.25 * self.mushy_range * smooth
        return 0.5 * (1.0 + jnp.tanh((T - self.T_melt) / w))

    def dfl_dT(self, T, *, smooth: float = 1.0):
        """df_l/dT，用于表观比热 cp_eff = cp + L_f·df_l/dT。"""
        w = 0.25 * self.mushy_range * smooth
        s = jnp.tanh((T - self.T_melt) / w)
        return 0.5 * (1.0 - s * s) / w

    def rho_of(self, fl):
        return self.rho_solid + (self.rho_liquid - self.rho_solid) * fl

    def cp_of(self, fl):
        return self.cp_solid + (self.cp_liquid - self.cp_solid) * fl

    def k_of(self, fl):
        return self.k_solid + (self.k_liquid - self.k_solid) * fl

    def recoil_pressure(self, T):
        """蒸发反冲压力 [Pa] —— Clausius-Clapeyron + Anisimov 系数 0.54。

        p_r(T) = 0.54 · p_atm · exp( L_v·M/R · (1/T_b − 1/T) )

        它是匙孔 (keyhole) 形成的直接推手：T 超过沸点后指数增长，
        把液面压出一个深坑，激光得以多次反射进入坑内（Fresnel 吸收增强），
        形成正反馈。指数项做了限幅，避免 T 很高时溢出。
        """
        R = 8.314462618
        arg = (self.latent_vapor * self.molar_mass / R) * (
            1.0 / self.T_boil - 1.0 / jnp.maximum(T, 300.0)
        )
        return 0.54 * 101325.0 * jnp.exp(jnp.clip(arg, -50.0, 20.0))

    def evaporation_flux(self, T):
        """Hertz-Knudsen 蒸发质量通量 [kg/(m²·s)]。

        ṁ = 0.82 · p_sat(T) · sqrt( M / (2π R T) )
        用于表面蒸发散热与飞溅/烟尘判据。
        """
        R = 8.314462618
        p_sat = self.recoil_pressure(T) / 0.54  # 去掉 Anisimov 系数得饱和蒸气压
        return 0.82 * p_sat * jnp.sqrt(
            self.molar_mass / (2.0 * jnp.pi * R * jnp.maximum(T, 300.0))
        )

    # ---- 工具 ------------------------------------------------------------
    def replace(self, **kw) -> "AMMaterial":
        """返回改了若干物性的新对象（用于灵敏度分析/反演标定）。"""
        return replace(self, **kw)

    def summary(self) -> str:
        return (
            f"{self.name} [{self.lattice}]  "
            f"T_s/T_l/T_b = {self.T_solidus:.0f}/{self.T_liquidus:.0f}/{self.T_boil:.0f} K\n"
            f"  ρ = {self.rho_solid:.0f}/{self.rho_liquid:.0f} kg/m³ (固/液)\n"
            f"  k = {self.k_solid:.1f}/{self.k_liquid:.1f} W/(m·K)   "
            f"cp = {self.cp_solid:.0f}/{self.cp_liquid:.0f} J/(kg·K)\n"
            f"  L_f = {self.latent_fusion/1e3:.0f} kJ/kg   "
            f"L_v = {self.latent_vapor/1e6:.2f} MJ/kg\n"
            f"  μ = {self.mu_liquid*1e3:.2f} mPa·s   σ = {self.sigma0:.3f} N/m   "
            f"dσ/dT = {self.dsigma_dT*1e4:.2f}e-4 N/(m·K)\n"
            f"  吸收率 = {self.absorptivity:.2f}   粉床 k 折减 = {self.powder_k_factor:.3f}\n"
            f"  来源: {self.reference}"
        )


# 注册为 pytree：物性可作为可微参数参与反演标定
def _mat_flatten(m: AMMaterial):
    num, aux = [], {}
    for f in fields(m):
        v = getattr(m, f.name)
        if isinstance(v, str):
            aux[f.name] = v
        else:
            num.append(v)
    aux["_order"] = tuple(f.name for f in fields(m) if not isinstance(getattr(m, f.name), str))
    return tuple(num), aux


def _mat_unflatten(aux, leaves):
    aux = dict(aux)
    order = aux.pop("_order")
    kw = dict(aux)
    kw.update(dict(zip(order, leaves)))
    return AMMaterial(**kw)


jax.tree_util.register_pytree_node(AMMaterial, _mat_flatten, _mat_unflatten)


# ===========================================================================
# 合金库
# ===========================================================================
AM_MATERIALS: dict[str, AMMaterial] = {
    # -----------------------------------------------------------------
    "Ti6Al4V": AMMaterial(
        name="Ti6Al4V",
        lattice="hcp",
        reference="Mills (2002) Recommended Values; Boivineau et al. (2006) "
                  "IJT 27:507; Ti6Al4V LPBF 常用参数集",
        T_solidus=1878.0, T_liquidus=1933.0, T_boil=3560.0,
        rho_solid=4420.0, rho_liquid=4000.0,
        cp_solid=750.0, cp_liquid=830.0,
        k_solid=21.0, k_liquid=30.0,
        latent_fusion=2.86e5, latent_vapor=9.83e6,
        mu_liquid=3.25e-3,
        sigma0=1.525, dsigma_dT=-2.80e-4, T_sigma_ref=1933.0,
        beta_thermal=1.0e-4,
        absorptivity=0.35, emissivity=0.40, molar_mass=0.0466,
        packing_fraction=0.55, powder_k_factor=0.017,
        E=110e9, nu=0.31, alpha_thermal=9.0e-6,
        sigma_y=950e6, hardening_sat=280e6, hardening_rate=14.0,
        hall_petch_sigma0=680e6, hall_petch_k=0.42e6,
        gb_energy=1.0, nucleation_undercooling=8.0,
        dendrite_gibbs_thomson=2.0e-7, partition_coefficient=0.9,
        crss0=180e6, crss_sat=450e6, cp_hardening=120e6,
    ),
    # -----------------------------------------------------------------
    "IN718": AMMaterial(
        name="IN718",
        lattice="fcc",
        reference="Mills (2002); Pottlacher et al. (2002); "
                  "Special Metals Inconel 718 数据表",
        T_solidus=1533.0, T_liquidus=1609.0, T_boil=3190.0,
        rho_solid=8190.0, rho_liquid=7400.0,
        cp_solid=620.0, cp_liquid=720.0,
        k_solid=29.6, k_liquid=29.6,
        latent_fusion=2.10e5, latent_vapor=6.40e6,
        mu_liquid=7.2e-3,
        sigma0=1.842, dsigma_dT=-3.97e-4, T_sigma_ref=1609.0,
        beta_thermal=1.0e-4,
        absorptivity=0.35, emissivity=0.30, molar_mass=0.0587,
        packing_fraction=0.55, powder_k_factor=0.020,
        E=200e9, nu=0.29, alpha_thermal=14.4e-6,
        sigma_y=1100e6, hardening_sat=350e6, hardening_rate=12.0,
        hall_petch_sigma0=760e6, hall_petch_k=0.75e6,
        gb_energy=0.8, nucleation_undercooling=5.0,
        dendrite_gibbs_thomson=3.0e-7, partition_coefficient=0.85,
        crss0=320e6, crss_sat=650e6, cp_hardening=250e6,
    ),
    # -----------------------------------------------------------------
    # IN625 存在的理由：它是 NIST AM-Bench AMB2018-02 的基准材料。该基准
    # 提供**裸板无粉**单道熔池的实测宽/深（附不确定度），与解析代理模型的
    # 物理假设完全对齐（无粉末层导热折减、无粉末表面粗糙度），因此是本项目
    # 唯一可信的定量校验锚点。切勿为了"凑上"基准而改这里的物性 —— 物性取
    # 自独立来源，模型与基准的差距应当暴露出来而不是被吸收掉。
    "IN625": AMMaterial(
        name="IN625",
        lattice="fcc",
        reference="Special Metals Inconel 625 数据表; Mills (2002); "
                  "AM-Bench AMB2018-02 建模文献常用温变拟合 "
                  "cp=405+247·(T[°C]/1000) J/kgK, k=9.5+15·(T[°C]/1000) W/mK "
                  "(固相区，取 25~1290 °C 区间均值)",
        T_solidus=1563.0, T_liquidus=1623.0, T_boil=3200.0,
        rho_solid=8440.0, rho_liquid=7700.0,
        cp_solid=567.0, cp_liquid=710.0,
        k_solid=19.4, k_liquid=30.0,
        latent_fusion=2.80e5, latent_vapor=6.40e6,
        mu_liquid=7.0e-3,
        sigma0=1.80, dsigma_dT=-3.80e-4, T_sigma_ref=1623.0,
        beta_thermal=1.0e-4,
        absorptivity=0.39, emissivity=0.30, molar_mass=0.0588,
        packing_fraction=0.55, powder_k_factor=0.020,
        E=208e9, nu=0.31, alpha_thermal=12.8e-6,
        sigma_y=490e6, hardening_sat=400e6, hardening_rate=11.0,
        hall_petch_sigma0=300e6, hall_petch_k=0.60e6,
        gb_energy=0.8, nucleation_undercooling=5.0,
        dendrite_gibbs_thomson=3.0e-7, partition_coefficient=0.87,
        crss0=180e6, crss_sat=480e6, cp_hardening=200e6,
    ),
    # -----------------------------------------------------------------
    "316L": AMMaterial(
        name="316L",
        lattice="fcc",
        reference="Mills (2002); Kim (1975) NBS; 316L LPBF 文献常用值",
        T_solidus=1658.0, T_liquidus=1723.0, T_boil=3090.0,
        rho_solid=7950.0, rho_liquid=7000.0,
        cp_solid=650.0, cp_liquid=800.0,
        k_solid=30.0, k_liquid=25.0,
        latent_fusion=2.70e5, latent_vapor=7.45e6,
        mu_liquid=6.0e-3,
        sigma0=1.60, dsigma_dT=-4.30e-4, T_sigma_ref=1723.0,
        beta_thermal=1.0e-4,
        absorptivity=0.35, emissivity=0.35, molar_mass=0.0559,
        packing_fraction=0.55, powder_k_factor=0.020,
        E=195e9, nu=0.28, alpha_thermal=16.5e-6,
        sigma_y=500e6, hardening_sat=450e6, hardening_rate=10.0,
        hall_petch_sigma0=220e6, hall_petch_k=0.55e6,
        gb_energy=0.8, nucleation_undercooling=5.0,
        dendrite_gibbs_thomson=3.0e-7, partition_coefficient=0.9,
        crss0=140e6, crss_sat=400e6, cp_hardening=180e6,
    ),
    # -----------------------------------------------------------------
    "AlSi10Mg": AMMaterial(
        name="AlSi10Mg",
        lattice="fcc",
        reference="Mills (2002); Assael et al. (2006) JPCRD 35:285; "
                  "EOS AlSi10Mg 数据表",
        T_solidus=831.0, T_liquidus=869.0, T_boil=2743.0,
        rho_solid=2670.0, rho_liquid=2400.0,
        cp_solid=1000.0, cp_liquid=1180.0,
        k_solid=150.0, k_liquid=90.0,
        latent_fusion=4.23e5, latent_vapor=1.07e7,
        mu_liquid=1.30e-3,
        sigma0=0.817, dsigma_dT=-1.55e-4, T_sigma_ref=869.0,
        beta_thermal=1.2e-4,
        # 铝合金对 1064 nm 高反射，吸收率显著低于钢/钛 —— 这是铝件难打的核心原因
        absorptivity=0.13, emissivity=0.20, molar_mass=0.0270,
        packing_fraction=0.50, powder_k_factor=0.010,
        E=70e9, nu=0.33, alpha_thermal=21.0e-6,
        sigma_y=240e6, hardening_sat=120e6, hardening_rate=15.0,
        hall_petch_sigma0=110e6, hall_petch_k=0.22e6,
        gb_energy=0.6, nucleation_undercooling=3.0,
        dendrite_gibbs_thomson=2.4e-7, partition_coefficient=0.13,
        crss0=60e6, crss_sat=180e6, cp_hardening=80e6,
    ),
}


def get_material(name: str) -> AMMaterial:
    """按名取材料（大小写与连字符不敏感）。"""
    key = name.strip().lower().replace("-", "").replace(" ", "")
    for k, v in AM_MATERIALS.items():
        if k.lower().replace("-", "") == key:
            return v
    raise KeyError(
        f"未知材料 {name!r}。可用: {sorted(AM_MATERIALS)}。"
        f"自定义材料可直接构造 AMMaterial(...)。"
    )


def list_materials() -> list[str]:
    return sorted(AM_MATERIALS)


def material_table() -> str:
    """材料库速查表（用于文档与 CLI）。"""
    rows = [
        f"{'合金':<10} {'T_l[K]':>7} {'ρ_l':>6} {'k_l':>6} {'μ[mPa·s]':>9} "
        f"{'σ[N/m]':>7} {'dσ/dT':>9} {'A':>5}",
        "-" * 70,
    ]
    for name in list_materials():
        m = AM_MATERIALS[name]
        rows.append(
            f"{name:<10} {m.T_liquidus:>7.0f} {m.rho_liquid:>6.0f} "
            f"{m.k_liquid:>6.1f} {m.mu_liquid*1e3:>9.2f} {m.sigma0:>7.3f} "
            f"{m.dsigma_dT:>9.2e} {m.absorptivity:>5.2f}"
        )
    return "\n".join(rows)
