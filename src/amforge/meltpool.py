"""熔池多物理场求解器 —— Flow3D 式自由界面 CFD（纯 JAX，可微）
==============================================================

这是功能 1 的核心：像 Flow3D 那样求解增材制造熔池，但整条求解过程可微。

物理模型（对标 Flow3D / FLOW-3D AM 的熔池模块）
----------------------------------------------
================  ====================================================
物理              模型
================  ====================================================
自由界面          VOF（金属体积分数 F）+ MUSCL/TVD 输运 + 界面压缩项
动量              变密度不可压 Navier-Stokes，投影法（Chorin）
表面张力          CSF 连续表面力 f = σ(T)·κ·∇F
**Marangoni**     dσ/dT·∇_s T·|∇F| —— 熔池宽浅/窄深的一阶控制因素
糊状区            enthalpy-porosity（Carman-Kozeny Darcy 阻力）
能量              表观比热法（cp_eff = cp + L_f·df_l/dT）含相变潜热
激光              移动高斯热源 + Beer-Lambert 遮挡 + Fresnel 多次反射增强
**蒸发/匙孔**     Clausius-Clapeyron 反冲压力 + Hertz-Knudsen 蒸发散热
散热              辐射 (Stefan-Boltzmann) + 对流 + 基板导热
粉末              粉床松装 (F=packing) + 有效导热折减 + 熔化致密化
================  ====================================================

数值方案
--------
* 单流体 (one-fluid) VOF：只在金属域求解动量，气相视为静止背景压力，
  这是激光焊接/AM 熔池的标准做法——气相动力学对熔池形貌影响极小，
  却会把时间步长压到不可接受的水平。
* 时间推进：显式，自适应 dt（同时受 CFL / 扩散 / 毛细波三重限制），
  外层 ``jax.lax.scan`` 固定步数 —— 步数固定是**反向传播可行的前提**。
* 压力泊松：变系数、无矩阵、Jacobi 预处理共轭梯度（固定迭代数），
  自由界面处以气相 p=0 作 Dirichlet 条件，天然消除纯 Neumann 的零空间。

为什么坚持"可微"
----------------
功能 2 要对整条链求梯度。传统 CFD 求解器（含 Flow3D）是黑箱，
只能靠有限差分近似梯度：n 个工艺参数要 n+1 次仿真。
本模块用 ``jax.grad`` 一次反传拿到全部梯度，
在工艺参数维度较高（逐层功率/速度 = 上百维）时是**唯一可行**的路线。

已知局限（诚实声明）
--------------------
1. 一阶显式时间推进 + 结构化等距网格：不做局部自适应加密，
   高保真区域靠 :func:`~amforge.geometry.crop_to_part` + ``resample`` 手动分区。
2. 单流体假设：不解气相流动，因此不预测烟尘/羽流对光束的衰减。
3. 飞溅只给风险指标，不做液滴脱离的拉格朗日跟踪。
4. 反向传播需保存中间状态；显存约束下建议
   ``checkpoint_every`` 配合 ``jax.checkpoint`` 使用（见 :class:`MeltPoolConfig`）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from amforge.core.contracts import MeltPoolResult, PartGeometry, ProcessPlan, ThermalHistory
from amforge.core.registry import register_adapter, register_solver
from amforge.materials import AMMaterial, get_material
from amforge.thermal_enthalpy import (
    effective_diffusivity, temperature_of_enthalpy, _div_alpha_grad,
)

__all__ = [
    "MeltPoolConfig",
    "MeltPoolState",
    "MeltPoolSolution",
    "LaserTrajectory",
    "zigzag_trajectory",
    "polyline_trajectory",
    "build_initial_state",
    "simulate_meltpool",
    "solve_meltpool_vof",
    "solve_meltpool_surrogate",
]

SIGMA_SB = 5.670374419e-8   # Stefan-Boltzmann [W/(m²·K⁴)]
GRAVITY = 9.80665


# ===========================================================================
# 配置
# ===========================================================================
@dataclass(frozen=True)
class MeltPoolConfig:
    """熔池求解器的**数值**配置（静态：改动会触发重新编译）。

    物性走 :class:`~amforge.materials.AMMaterial`，工艺走
    :class:`~amforge.core.contracts.ProcessPlan`，两者都是可微 pytree；
    这里只放离散化与算法开关。

    分辨率怎么定？
    --------------
    经验规则：``dx ≈ beam_radius / 6 ~ /10``。光斑半径 50 µm 时取 5~8 µm。
    再粗则 Marangoni 环流解不出来（熔池宽度会系统性偏小），
    再细则时间步被毛细波限制拖垮。

    Attributes
    ----------
    nx, ny, nz
        局部计算域网格数（静态）。
    dx
        体素边长 [m]。
    n_steps
        时间步数（静态，反传需要）。
    cfl
        对流 CFL 数，显式方案建议 ≤ 0.3。
    diffusion_safety
        扩散稳定性安全系数（显式限制 dt ≤ s·dx²/(2·d·α)）。
    capillary_safety
        毛细波时间步限制的安全系数。放宽它可以加速，
        代价是自由界面可能出现非物理振荡。
    n_pressure_iters
        压力泊松 CG 迭代数（静态）。20~60 通常足够；
        迭代不足表现为速度场散度残留、熔池体积漂移。
    darcy_constant
        Carman-Kozeny 常数 C [kg/(m³·s)]。1e6~1e9；
        太小则固相"流动"，太大则时间步受限。
    interface_compression
        VOF 界面压缩系数 c_α（0 = 关闭，1 = 标准 interFoam 强度）。
    fresnel_boost
        匙孔内多次反射的吸收增强系数（0 = 关闭）。
    gas_damping
        气相速度衰减因子（单流体假设，每步乘以此因子）。
    substrate_fraction
        计算域内基板占 Z 向的比例（其余为粉层 + 气相空间）。
    powder_layers
        计算域顶部铺几层粉。
    h_convection
        表面对流换热系数 [W/(m²·K)]（保护气流动）。
    checkpoint_every
        >0 时每隔若干步做一次 ``jax.checkpoint``，用重算换显存，使默认链路
        即可全程 ``jax.grad`` 可微（默认 20 步一块）。
    advection
        ``"muscl"``（二阶 TVD，推荐）或 ``"upwind"``（一阶，更稳但耗散大）。
    track_length
        单道扫描长度 [m]。
    n_tracks
        道数（多道用于研究搭接与残余热累积）。
    """

    nx: int = 64
    ny: int = 48
    nz: int = 40
    dx: float = 5e-6
    n_steps: int = 400
    cfl: float = 0.25
    diffusion_safety: float = 0.35
    capillary_safety: float = 0.8
    n_pressure_iters: int = 30
    darcy_constant: float = 1e8
    interface_compression: float = 1.0
    fresnel_boost: float = 1.5
    gas_damping: float = 0.0
    substrate_fraction: float = 0.55
    powder_layers: int = 1
    h_convection: float = 20.0
    checkpoint_every: int = 20
    # 默认开启 checkpoint：把每 20 步打包成一个 jax.checkpoint 块，
    # 反向时只存块边界、块内各步前向重算。这样默认链路即可全程
    # jax.grad 可微而不 OOM（400 步裸 scan 反向需 ~677MB、会撑爆；
    # 20 步一块峰值约 68MB）。前向数值与不开 checkpoint 完全一致。
    advection: str = "muscl"
    track_length: float = 3.0e-4
    n_tracks: int = 1
    laser_absorption_depth: float = 3.0   # Beer-Lambert 遮挡系数（无量纲/体素）
    dt_max: float = 1e-6
    record_every: int = 0                 # >0 时记录中间快照（诊断用，不可微友好）
    recoil_pressure_cap: float = 1.0e6    # 反冲压力上限 [Pa]（~10 atm）。
    # Clausius-Clapeyron 在 T≫T_boil 时指数发散；真实匙孔壁蒸气压仅 ~1–10 atm，
    # 故对反冲压力封顶——既保留匙孔驱动能力，又消除显式格式在指数尾处的数值发散。
    velocity_clamp: float = 0.0           # 速度安全限幅 [m/s]（0=关闭）。
    # 显式 NS+VOF 在匙孔/强 Marangoni 区偶发数值尖峰时，限幅只截掉非物理尖峰，
    # 真实熔池流速 < ~10 m/s，故 50 m/s 仅作安全阀、不影响物理。

    # ---- 派生 ------------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.nx, self.ny, self.nz)

    def domain_size(self) -> tuple[float, float, float]:
        return (self.nx * self.dx, self.ny * self.dx, self.nz * self.dx)

    def substrate_top_index(self) -> int:
        return max(1, int(self.substrate_fraction * self.nz))

    def describe(self) -> str:
        lx, ly, lz = self.domain_size()
        return (
            f"网格 {self.nx}×{self.ny}×{self.nz} @ dx={self.dx*1e6:.1f} µm  "
            f"域 {lx*1e3:.2f}×{ly*1e3:.2f}×{lz*1e3:.2f} mm³\n"
            f"步数 {self.n_steps}  CFL={self.cfl}  压力迭代 {self.n_pressure_iters}  "
            f"对流格式 {self.advection}\n"
            f"基板占比 {self.substrate_fraction:.2f}  粉层 {self.powder_layers}  "
            f"道数 {self.n_tracks}  道长 {self.track_length*1e3:.2f} mm"
        )


# ===========================================================================
# 状态
# ===========================================================================
@dataclass(frozen=True)
class MeltPoolState:
    """求解状态 + 在线累积的诊断量（pytree）。

    诊断量在时间推进中**在线累积**，而不是保存全部时刻场再后处理——
    后者的显存开销是 ``n_steps × 场大小``，在 3D 下立刻爆掉。
    """

    F: jnp.ndarray            # 金属体积分数
    F0: jnp.ndarray           # 初始金属体积分数（匙孔凹陷的参考基准，恒定不更新）
    T: jnp.ndarray            # 温度 [K]
    u: jnp.ndarray            # 速度 (nx,ny,nz,3) [m/s]
    p: jnp.ndarray            # 压力 [Pa]
    powder: jnp.ndarray       # 粉末标记 ∈[0,1]（1=松散粉，0=致密）
    # -- 在线累积的诊断 --
    T_peak: jnp.ndarray       # 峰值温度 [K]
    t_above_melt: jnp.ndarray  # 高于固相线的累计时间 [s]
    w_sol: jnp.ndarray        # 凝固权重累计 Σw
    G_acc: jnp.ndarray        # Σ w·|∇T|
    Rdot_acc: jnp.ndarray     # Σ w·(-dT/dt)
    melt_depth_max: jnp.ndarray   # 标量：最大熔深 [m]
    keyhole_max: jnp.ndarray      # 标量：最大匙孔深度 [m]
    vmax_max: jnp.ndarray         # 标量：历史最大流速 [m/s]
    precoil_max: jnp.ndarray      # 标量：历史最大反冲压力 [Pa]
    time: jnp.ndarray             # 标量：物理时间 [s]


jax.tree_util.register_pytree_node(
    MeltPoolState,
    lambda s: (
        (s.F, s.F0, s.T, s.u, s.p, s.powder, s.T_peak, s.t_above_melt, s.w_sol,
         s.G_acc, s.Rdot_acc, s.melt_depth_max, s.keyhole_max, s.vmax_max,
         s.precoil_max, s.time),
        None,
    ),
    lambda _aux, leaves: MeltPoolState(*leaves),
)


@dataclass(frozen=True)
class MeltPoolSolution:
    """求解产物：既能给 :class:`MeltPoolResult`，也能给 :class:`ThermalHistory`。

    之所以设这个中间对象：一次昂贵的 CFD 求解同时产出了熔池形貌与热历史，
    但契约系统里一个求解器只声明一个输出。用它做承接，
    两个薄封装（:func:`solve_meltpool_vof` 与 ``thermal.vof_resolved``）
    各取所需，避免把两类语义混塞进一个契约。
    """

    state: MeltPoolState
    config: MeltPoolConfig
    material: AMMaterial
    depth: jnp.ndarray
    width: jnp.ndarray
    length: jnp.ndarray

    def to_meltpool(self, process: ProcessPlan) -> MeltPoolResult:
        return _assemble_meltpool_result(self, process)

    def to_thermal(self) -> ThermalHistory:
        return _assemble_thermal_history(self)


# ===========================================================================
# 激光轨迹
# ===========================================================================
@dataclass(frozen=True)
class LaserTrajectory:
    """激光轨迹：给定时间返回 (x, y, on) —— ``on`` 为占空（层间空程时为 0）。

    ``fn`` 必须是可微的纯函数，这样"扫描速度"这一工艺参数的梯度才能回传。
    """

    fn: Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]
    total_time: float
    doc: str = ""

    def __call__(self, t):
        return self.fn(t)


def zigzag_trajectory(
    *,
    speed,
    track_length: float,
    n_tracks: int,
    hatch,
    center_x: float,
    center_y: float,
    jump_time: float = 0.0,
) -> LaserTrajectory:
    """折返式多道扫描轨迹（SLM 最常用的填充策略）。

    第 k 道沿 ±x 方向交替，y 方向按 ``hatch`` 递进。
    ``speed`` 与 ``hatch`` 可以是 JAX 标量（可微）。
    """
    t_track = track_length / jnp.maximum(speed, 1e-12)
    t_cycle = t_track + jump_time
    y0 = center_y - 0.5 * hatch * (n_tracks - 1)

    def fn(t):
        k = jnp.floor(t / t_cycle)
        k = jnp.clip(k, 0, n_tracks - 1)
        tau = t - k * t_cycle
        on = (tau <= t_track).astype(jnp.float64)
        s = jnp.clip(tau, 0.0, t_track) * speed          # 弧长
        forward = jnp.mod(k, 2.0) < 0.5
        x = jnp.where(
            forward,
            center_x - 0.5 * track_length + s,
            center_x + 0.5 * track_length - s,
        )
        y = y0 + k * hatch
        return x, y, on

    total = float(n_tracks) * float(np.asarray(t_cycle))
    return LaserTrajectory(fn=fn, total_time=total, doc="zigzag")


def polyline_trajectory(waypoints, *, speed, close: bool = False) -> LaserTrajectory:
    """任意折线轨迹（可来自切片器 G-code / CSV / 自定义路径）。

    按弧长参数化，速度恒定；对 ``speed`` 可微。
    """
    wp = jnp.asarray(waypoints, dtype=jnp.float64)
    if close:
        wp = jnp.concatenate([wp, wp[:1]], axis=0)
    seg = jnp.linalg.norm(jnp.diff(wp, axis=0), axis=-1)
    cum = jnp.concatenate([jnp.zeros(1), jnp.cumsum(seg)])
    total_len = cum[-1]

    def fn(t):
        s = jnp.clip(t * speed, 0.0, total_len)
        x = jnp.interp(s, cum, wp[:, 0])
        y = jnp.interp(s, cum, wp[:, 1])
        return x, y, jnp.asarray(1.0)

    return LaserTrajectory(
        fn=fn,
        total_time=float(np.asarray(total_len / jnp.maximum(speed, 1e-12))),
        doc="polyline",
    )


# ===========================================================================
# 差分算子（Neumann/edge 边界）
# ===========================================================================
def _shift(f: jnp.ndarray, k: int, axis: int) -> jnp.ndarray:
    """``f_{i+k}``，越界用边界值复制（等价 edge padding）。"""
    n = f.shape[axis]
    idx = jnp.clip(jnp.arange(n) + k, 0, n - 1)
    return jnp.take(f, idx, axis=axis)


def _norm(v: jnp.ndarray, eps: float = 1e-30) -> jnp.ndarray:
    """**可微安全**的向量模长 ``√(Σvᵢ²+ε)``。

    为什么不能直接用 ``jnp.linalg.norm``：它在向量恰为零处导数是 ``v/|v| = 0/0``
    -> 反向传播得到 **NaN**。而熔池场里"恰为零"极其普遍 ——
    致密基板内部 ``∇F ≡ 0``、气相内部 ``∇F ≡ 0``、初始时刻 ``u ≡ 0``、
    等温区 ``∇T ≡ 0``。一个体素的 NaN 就会污染整条反传链，
    表现为 ``jax.grad`` 全场 NaN（正向却完全正常，极难定位）。
    加 ε 后零点导数为 ``v/√ε → 0``，有限且方向正确。
    """
    return jnp.sqrt(jnp.sum(v * v, axis=-1) + eps)


def _grad(f: jnp.ndarray, dx: float) -> jnp.ndarray:
    """中心差分梯度 -> (..., 3)。"""
    return jnp.stack(
        [(_shift(f, 1, c) - _shift(f, -1, c)) / (2.0 * dx) for c in range(3)],
        axis=-1,
    )


def _div(v: jnp.ndarray, dx: float) -> jnp.ndarray:
    """中心差分散度。"""
    out = jnp.zeros(v.shape[:-1], dtype=v.dtype)
    for c in range(3):
        fc = v[..., c]
        out = out + (_shift(fc, 1, c) - _shift(fc, -1, c)) / (2.0 * dx)
    return out


def _div_k_grad(k: jnp.ndarray, f: jnp.ndarray, dx: float) -> jnp.ndarray:
    """变系数扩散 ∇·(k∇f)，面系数取**调和平均**。

    调和平均而非算术平均：金属/粉末/气相界面处 k 跳变 1~2 个数量级，
    算术平均会高估跨界面导热（把热量"漏"进气相），
    调和平均正确地由小的一侧控制串联热阻。
    """
    out = jnp.zeros_like(f)
    for c in range(3):
        for s in (1, -1):
            kn = _shift(k, s, c)
            fn = _shift(f, s, c)
            kf = 2.0 * k * kn / (k + kn + 1e-30)
            out = out + kf * (fn - f)
    return out / dx ** 2


def _face_mask(shape, axis: int, dtype) -> jnp.ndarray:
    """i+1/2 面是否为内部面（最后一层为域边界 -> 0，保证守恒）。"""
    n = shape[axis]
    m = (jnp.arange(n) < n - 1).astype(dtype)
    sh = [1, 1, 1]
    sh[axis] = n
    return m.reshape(sh)


def _face_inner(shape, axis: int, dtype) -> jnp.ndarray:
    """单元 i 的**左**面(i-1/2)是否为内部面（i=0 的左面是域边界 -> 0）。"""
    n = shape[axis]
    m = (jnp.arange(n) > 0).astype(dtype)
    sh = [1, 1, 1]
    sh[axis] = n
    return m.reshape(sh)


def _grad_face(f: jnp.ndarray, dx: float) -> jnp.ndarray:
    """**前差**梯度：分量落在 i+1/2 面上，域边界面置 0。

    与 :func:`_div_face` 构成**相容 D–G 对**：``div_face(β·grad_face(p))``
    精确等于紧致 7 点变系数拉普拉斯算子（即压力方程真正求解的算子）。

    为什么不能用中心差分 D–G 对：``div_c(β grad_c p)`` 是跨 2Δx 的宽模板，
    把网格拆成 8 个互不耦合的子格 -> 棋盘模式落在算子零空间里 ->
    压力解"看不见"这部分散度 -> 散度逐步无阻尼增长直到速度场爆掉。
    这是此前 vmax 冲到 1e10 m/s 的根因之一。
    """
    return jnp.stack(
        [(_shift(f, 1, c) - f) / dx * _face_mask(f.shape, c, f.dtype) for c in range(3)],
        axis=-1,
    )


def _div_face(v: jnp.ndarray, dx: float) -> jnp.ndarray:
    """**后差**散度：把 i±1/2 面法向量收敛成单元散度（与 _grad_face 相容）。"""
    shape, dtype = v.shape[:-1], v.dtype
    out = jnp.zeros(shape, dtype=dtype)
    for c in range(3):
        fc = v[..., c] * _face_mask(shape, c, dtype)
        out = out + (fc - _shift(fc, -1, c) * _face_inner(shape, c, dtype)) / dx
    return out


def _minmod(a, b):
    """minmod 限制器：保 TVD、无过冲，界面不产生非物理的 F<0 或 F>1。"""
    return 0.5 * (jnp.sign(a) + jnp.sign(b)) * jnp.minimum(jnp.abs(a), jnp.abs(b))


def _advect_flux(f: jnp.ndarray, u: jnp.ndarray, dx: float, scheme: str) -> jnp.ndarray:
    """通量形式对流项 ``-∇·(u f)``（守恒型）。

    MUSCL 分支用 minmod 限制的二阶重构；upwind 分支为一阶。
    面速度取相邻单元算术平均（collocated 网格的常规处理）。
    """
    out = jnp.zeros_like(f)
    for c in range(3):
        uc = u[..., c]
        uf = 0.5 * (uc + _shift(uc, 1, c))                   # i+1/2 面速度
        fL, fR = f, _shift(f, 1, c)
        if scheme == "muscl":
            dL = fR - fL
            dLm = fL - _shift(f, -1, c)
            dRp = _shift(f, 2, c) - fR
            fL = fL + 0.5 * _minmod(dL, dLm)
            fR = fR - 0.5 * _minmod(dRp, dL)
        f_face = jnp.where(uf >= 0.0, fL, fR)
        flux = uf * f_face * _face_mask(f.shape, c, f.dtype)
        flux_m = _shift(flux, -1, c)
        # 单元 0 的左面是域边界 -> 通量置 0
        n = f.shape[c]
        sh = [1, 1, 1]
        sh[c] = n
        inner = (jnp.arange(n) > 0).astype(f.dtype).reshape(sh)
        out = out - (flux - flux_m * inner) / dx
    return out


# ===========================================================================
# 压力投影
# ===========================================================================
def _cg(Aop, b, Minv, n_iters: int):
    """Jacobi 预处理共轭梯度（固定迭代数，反向可微）。"""
    x = jnp.zeros_like(b)
    r = b
    z = Minv * r
    p = z
    rz = jnp.sum(r * z)

    def _safe_div(num, den, tol):
        """退化分支返回 0，且**梯度也为 0**。

        写成 ``num / jnp.where(|den|<tol, tol, den)`` 是错的：正向确实得到
        ``num/tol``，但反向 ``∂/∂num = 1/tol = 1e300`` 会把梯度直接打成 inf/NaN。
        CG 一旦在 ``n_iters`` 内收敛，``r→0`` 让 ``rz`` 与 ``pᵀAp`` 双双趋零、
        必然踩中这个分支 —— 症状是正向结果完美而 ``jax.grad`` 全场 NaN。
        """
        bad = jnp.abs(den) < tol
        safe = jnp.where(bad, 1.0, den)
        return jnp.where(bad, 0.0, num / safe)

    def body(_i, st):
        x, r, p, rz = st
        Ap = Aop(p)
        alpha = _safe_div(rz, jnp.sum(p * Ap), 1e-300)
        x = x + alpha * p
        r = r - alpha * Ap
        z = Minv * r
        rz_new = jnp.sum(r * z)
        beta = _safe_div(rz_new, rz, 1e-300)
        p = z + beta * p
        return (x, r, p, rz_new)

    x, _, _, _ = jax.lax.fori_loop(0, n_iters, body, (x, r, p, rz))
    return x


def _pressure_system(rho, metal, dt, dx):
    """组装压力泊松系统，返回 ``(Aop, Minv, rhs_of, correct)``。

    抽成独立函数是为了**可单独验证**：算子的对称正定性、CG 收敛性、以及
    "解出的 p 能否把 ``_div_face`` 真正压到 0" 都可以脱离整个熔池时间步来测，
    这正是定位此前速度爆炸根因的关键手段（见 tests/test_meltpool_vof.py）。

    三条关键设计（每条都对应一个曾把速度场炸到 1e10 m/s 的真实缺陷）：

    1. **相容 D–G 对**。散度用 :func:`_div_face`（后差）、梯度用
       :func:`_grad_face`（前差），二者复合精确等于这里 ``Aop`` 求解的紧致
       7 点算子。于是"解出的 p"确实能把"测到的散度"压到 0。
       用中心差分 D–G 对时棋盘模式落在零空间里、压力看不见，散度会无阻尼增长。

    2. **自由表面是 Dirichlet 而非 Neumann**。金属单元朝气相的面上
       ``p_ghost = 0``（大气参考压；表面张力/反冲压已作为体积力单独加入），
       该面给对角线贡献 ``β/Δx²``。旧写法用调和平均把界面耦合直接归零，
       等价于给自由表面施加了**无通量 Neumann**：既物理错误（表面不能动），
       又让整个金属域退化成纯 Neumann **奇异**系统 —— 右端 ∮u*·n≠0 不满足
       相容条件时线性方程组根本无解，CG 只会沿零空间线性发散（p→1e21）。
       计算域顶面额外挂一个大气参考，保证"全金属"退化配置也非奇异。

    3. **气相行取单位算子**。``metal`` 是 tanh 软掩膜，气相残留 ~1.2e-3，
       调和平均后对角线接近 0，而 Jacobi 预条件子 ``1/diag`` 会把这些行放大
       ~1e30 倍、用舍入噪声污染整个 Krylov 子空间。这里显式给气相行
       ``β_gas·p = 0``（对称、良态、p_gas 恒为 0）。

    已知边界（诚实声明）
    --------------------
    域顶大气参考是**零阶**项（不对应任何面通量），因此最顶一层单元会残留
    ``O(dt·β·p/Δx²)`` 的离散散度 —— 任何零阶去奇异项都不可避免。它只在
    "整个计算域都是金属"这种合成退化配置下量级可见；真实熔池配置里顶层
    恒为气相（``a≈1.2e-3``），该项比同格点的 ``gas_pin`` 小约 3 个数量级，
    且该层速度随后被 ``a`` 掩膜清零，无实际影响。金属内部（顶层以外）的
    离散散度可压到机器精度（相对残差 ~1e-14，见 tests/test_meltpool_vof.py）。
    """
    a = jnp.clip(metal, 0.0, 1.0)                      # 金属域占空比（活跃度）
    beta = 1.0 / jnp.maximum(rho, 1e-6)                # 全场为正，不制造零行
    shape, dtype = a.shape, a.dtype

    fmask = [_face_mask(shape, c, dtype) for c in range(3)]
    finner = [_face_inner(shape, c, dtype) for c in range(3)]

    # ---- 面连通系数：两侧都在金属域内才连通（界面处的 β 跳变用调和平均）----
    bf = []
    for c in range(3):
        bn = _shift(beta, 1, c)
        bh = 2.0 * beta * bn / (beta + bn + 1e-30)
        bf.append(bh * a * _shift(a, 1, c) * fmask[c])

    # ---- Dirichlet 面（p=0）：金属→气相的自由表面 + 计算域顶面 ----
    w_lo = [beta * a * (1.0 - _shift(a, 1, c)) * fmask[c] for c in range(3)]   # 高侧是气相
    w_hi = [_shift(beta * a, 1, c) * (1.0 - a) * fmask[c] for c in range(3)]   # 低侧是气相
    dcoef = jnp.zeros(shape, dtype=dtype)
    for c in range(3):
        dcoef = dcoef + w_lo[c] + _shift(w_hi[c], -1, c) * finner[c]
    ztop = (jnp.arange(shape[2]) == shape[2] - 1).astype(dtype).reshape(1, 1, -1)
    dcoef = dcoef + beta * a * ztop                    # 域顶大气参考（防奇异）

    gas_pin = (1.0 - a) * beta                         # 气相单位行

    def Aop(p):
        lap = jnp.zeros_like(p)
        for c in range(3):
            bp = bf[c]
            bm = _shift(bf[c], -1, c) * finner[c]
            lap = lap + (bp * (_shift(p, 1, c) - p) - bm * (p - _shift(p, -1, c)))
        return (-lap + (dcoef + gas_pin) * p) / dx ** 2

    # Jacobi 预条件子 = Aop 对角线的倒数（Aop 为 SPD，对角线严格为正）
    diag = dcoef + gas_pin
    for c in range(3):
        diag = diag + bf[c] + _shift(bf[c], -1, c) * finner[c]
    Minv = 1.0 / jnp.maximum(diag / dx ** 2, 1e-30)

    # Aop = -∇·(β∇p) 为 SPD，故右端取负：解 ∇·(β∇p) = (1/dt)∇·u*
    def rhs_of(u_star):
        return -a * (_div_face(u_star, dx) / dt)

    def correct(u_star, p):
        """u = u* - dt·β∇p；自由表面面上用 p_ghost=0 的单侧差。"""
        corr = []
        for c in range(3):
            pn = _shift(p, 1, c)
            flux = bf[c] * (pn - p) - w_lo[c] * p + w_hi[c] * pn
            corr.append(flux * fmask[c] / dx)
        return (u_star - dt * jnp.stack(corr, axis=-1)) * a[..., None]

    return Aop, Minv, rhs_of, correct


def _project(u_star, rho, metal, dt, dx, n_iters):
    """变密度自由表面投影：解压力泊松、再把速度投影到（离散）无散空间。"""
    Aop, Minv, rhs_of, correct = _pressure_system(rho, metal, dt, dx)
    p = _cg(Aop, rhs_of(u_star), Minv, n_iters)
    return correct(u_star, p), p


# ===========================================================================
# 初始条件
# ===========================================================================
def build_initial_state(
    cfg: MeltPoolConfig,
    mat: AMMaterial,
    process: ProcessPlan,
    *,
    substrate_occupancy: jnp.ndarray | None = None,
) -> MeltPoolState:
    """构造初始场：基板（致密）+ 粉层（松装）+ 气相。

    Parameters
    ----------
    substrate_occupancy
        可选的 ``(nx, ny, nz)`` 基板占位场 ∈[0,1]，来自零件几何的 SDF 采样。
        传入它就实现了"**任意复杂几何**"与熔池求解器的对接：
        悬垂、薄壁、内腔下方缺少支撑金属时，局部散热条件会自动变差。
        为 ``None`` 时退化为全平基板（单道工艺窗口研究的标准配置）。
    """
    nx, ny, nz = cfg.shape
    k_sub = cfg.substrate_top_index()
    z_idx = jnp.arange(nz)

    # 基板：致密金属
    if substrate_occupancy is None:
        sub = (z_idx < k_sub).astype(jnp.float64)
        sub = jnp.broadcast_to(sub, (nx, ny, nz))
    else:
        sub = jnp.clip(jnp.asarray(substrate_occupancy, dtype=jnp.float64), 0.0, 1.0)
        sub = sub * (z_idx < k_sub)

    # 粉层：位于基板之上 powder_layers 层，堆积率 packing_fraction
    n_pl = max(0, int(cfg.powder_layers))
    lt = float(np.asarray(process.layer_thickness))
    n_cells_layer = max(1, int(round(lt / cfg.dx)))
    top = k_sub + n_pl * n_cells_layer
    powder_zone = ((z_idx >= k_sub) & (z_idx < min(top, nz))).astype(jnp.float64)
    powder_zone = jnp.broadcast_to(powder_zone, (nx, ny, nz))

    F = jnp.clip(sub + powder_zone * mat.packing_fraction, 0.0, 1.0)
    powder = powder_zone

    T0 = float(np.asarray(process.preheat_temp))
    T = jnp.full((nx, ny, nz), T0, dtype=jnp.float64)
    u = jnp.zeros((nx, ny, nz, 3), dtype=jnp.float64)
    p = jnp.zeros((nx, ny, nz), dtype=jnp.float64)
    z0 = jnp.zeros(())

    return MeltPoolState(
        F=F, F0=F, T=T, u=u, p=p, powder=powder,
        T_peak=T, t_above_melt=jnp.zeros_like(T),
        w_sol=jnp.zeros_like(T), G_acc=jnp.zeros_like(T), Rdot_acc=jnp.zeros_like(T),
        melt_depth_max=z0, keyhole_max=z0, vmax_max=z0, precoil_max=z0, time=z0,
    )


# ===========================================================================
# 单步推进
# ===========================================================================
def _smooth_max(x, axis=None, beta: float = 40.0):
    """**softmax 加权平均**式软最大值：几乎无偏，且处处可微。

    为什么不用 p-范数 ``(Σxᵖ)^{1/p}``：它对含 N 个近极大值的场会高估
    ``N^{1/p}`` 倍（例如 p=12、N=576 时高估 70%），而熔池诊断里
    "整片熔化区" 恰恰就是这种大平台场，误差不可接受。

    为什么不用 ``jnp.max``：梯度只流向单个体素，优化时梯度噪声极大。

    这里用 ``Σ softmax(β·x̂)·x``（x̂ 为按极差归一化后的量），
    对 0/1 场与平台场都返回 ≈ 真实最大值，同时把梯度摊到所有近极大
    体素上。``beta`` 越大越接近硬 max。

    退化场（``hi≈lo``，例如"基板完全没熔化"时的全零 melted 场）必须特殊处理：
    若仍用 ``scale = max(hi-lo, 1e-30)``，指数里的 ``beta/scale`` 会把梯度
    放大 ~1e30 倍 —— 正向返回正确的 0，``jax.grad`` 却给出 5e11 量级的伪导数，
    反演优化器第一步就会被这个假梯度带飞。此时正确答案就是**算术平均**
    （常数场的 max 等于其本身，梯度均摊 1/N），把 scale 置 1 即可自然得到。
    """
    x = jnp.asarray(x)
    hi = jax.lax.stop_gradient(jnp.max(x, axis=axis, keepdims=True))
    lo = jax.lax.stop_gradient(jnp.min(x, axis=axis, keepdims=True))
    span = hi - lo
    ref = jnp.maximum(jnp.abs(hi), jnp.abs(lo))
    degenerate = span <= 1e-9 * jnp.maximum(ref, 1e-30)
    # 非退化：scale=极差（让 beta 无量纲、尺度不变）；退化：scale=1 -> 退回均值
    scale = jnp.where(degenerate, 1.0, jnp.maximum(span, 1e-30))
    w = jnp.exp(beta * (x - hi) / scale)
    return jnp.sum(w * x, axis=axis) / jnp.sum(w, axis=axis)


def _smooth_field(F: jnp.ndarray, passes: int = 2) -> jnp.ndarray:
    """轻量拉普拉斯平滑：仅用于稳定曲率/法向估计，不参与 VOF 输运。

    显式 CSF 表面张力若直接用锐利界面的中心差分曲率 ``κ=-∇·n̂``，
    会在 1~2 体素宽的界面上产生 1e10/m 量级的曲率尖峰，驱动速度场发散。
    用平滑后的 F 估计法向与曲率（标准 interFoam 做法）即可消除尖峰，
    同时保持整体界面几何（熔池形貌、自由界面位置）不变。
    """
    out = F
    for _ in range(max(0, passes)):
        lap = (_shift(out, 1, 0) + _shift(out, -1, 0)
               + _shift(out, 1, 1) + _shift(out, -1, 1)
               + _shift(out, 1, 2) + _shift(out, -1, 2)
               - 6.0 * out)
        out = jnp.clip(out + 0.25 * lap, 0.0, 1.0)
    return out


def _laser_source(F, T, cfg, mat, process, traj, t, dx):
    """激光体热源 [W/m³]：表面吸收 + 遮挡 + Fresnel 增强。

    实现要点
    --------
    * ``|∇F|`` 充当界面 δ 函数：``∫|∇F| dz ≈ 1``，
      于是体源沿深度积分自动等于面热流，无需人工标定"吸收层厚度"。
    * 遮挡用沿 Z 向从顶往下的金属累积量做 Beer-Lambert 衰减，
      保证只有**最上面**那层界面拿到能量（否则内部界面会被"透射"加热）。
    * 匙孔内壁陡峭 -> ``1-|n_z|`` 大 -> 吸收率按 ``fresnel_boost`` 放大，
      这是匙孔正反馈（越深越吸收）的最简可微表达。
    """
    nx, ny, nz = F.shape
    xs = (jnp.arange(nx) + 0.5) * dx
    ys = (jnp.arange(ny) + 0.5) * dx
    xL, yL, on = traj(t)

    r_b = process.beam_radius
    P = process.laser_power * on
    A0 = process.absorption

    r2 = ((xs[:, None] - xL) ** 2 + (ys[None, :] - yL) ** 2)
    I = (2.0 * P / (jnp.pi * r_b ** 2)) * jnp.exp(-2.0 * r2 / r_b ** 2)   # [W/m²]

    gF = _grad(F, dx)
    gmag = _norm(gF)
    nz_comp = gF[..., 2] / jnp.maximum(gmag, 1e-12)

    # 从顶部往下累积的金属分数（不含本单元）
    above = jnp.cumsum(F[:, :, ::-1], axis=2)[:, :, ::-1] - F
    shadow = jnp.exp(-cfg.laser_absorption_depth * above)

    A_eff = jnp.clip(A0 * (1.0 + cfg.fresnel_boost * (1.0 - jnp.abs(nz_comp))), 0.0, 1.0)
    return A_eff * I[:, :, None] * gmag * shadow


def _step(state: MeltPoolState, cfg: MeltPoolConfig, mat: AMMaterial,
          process: ProcessPlan, traj: LaserTrajectory) -> MeltPoolState:
    """推进一个时间步（显式），返回新状态。"""
    dx = cfg.dx
    F, T, u, powder = state.F, state.T, state.u, state.powder

    # ---- 相与物性 ----------------------------------------------------
    fl = mat.liquid_fraction(T)                        # 液相分数
    metal = 0.5 * (1.0 + jnp.tanh((F - 0.5) / 0.15))   # 金属域软掩膜
    rho = mat.rho_of(fl) * jnp.maximum(F, 1e-3)
    cp_eff = mat.cp_of(fl) + mat.latent_fusion * mat.dfl_dT(T)
    # 粉床有效导热：松散粉的 k 折减 1~2 个数量级；气相取很小的值
    k_dense = mat.k_of(fl)
    k_eff = F * (powder * mat.powder_k_factor + (1.0 - powder)) * k_dense + 0.05
    mu = mat.mu_liquid

    # ---- 自适应时间步 ------------------------------------------------
    umax = jnp.sqrt(jnp.max(jnp.sum(u * u, axis=-1)) + 1e-30)
    dt_conv = cfg.cfl * dx / jnp.maximum(umax, 1e-6)
    alpha_max = jnp.max(k_eff / jnp.maximum(rho * cp_eff, 1e-6))
    nu = mu / jnp.maximum(mat.rho_liquid, 1e-6)
    dt_diff = cfg.diffusion_safety * dx ** 2 / jnp.maximum(6.0 * jnp.maximum(alpha_max, nu), 1e-12)
    dt_cap = cfg.capillary_safety * jnp.sqrt(
        mat.rho_liquid * dx ** 3 / (2.0 * jnp.pi * jnp.maximum(mat.sigma0, 1e-6))
    )
    dt = jnp.minimum(jnp.minimum(dt_conv, dt_diff), jnp.minimum(dt_cap, cfg.dt_max))

    # ---- 能量方程 ----------------------------------------------------
    q_laser = _laser_source(F, T, cfg, mat, process, traj, state.time, dx)

    gF = _grad(F, dx)
    gmag = _norm(gF)
    # 表面散热：辐射 + 对流 + 蒸发（都以 |∇F| 摊到界面单元）
    q_rad = mat.emissivity * SIGMA_SB * (T ** 4 - mat.T_ambient ** 4) * gmag
    q_conv = cfg.h_convection * (T - mat.T_ambient) * gmag
    mdot = mat.evaporation_flux(T)
    q_evap = mdot * mat.latent_vapor * gmag

    conv_T = _advect_flux(T, u, dx, cfg.advection)
    diff_T = _div_k_grad(k_eff, T, dx)
    rhocp = jnp.maximum(rho * cp_eff, 1e-3)
    dTdt = conv_T + (diff_T + q_laser - q_rad - q_conv - q_evap) / rhocp
    T_new = T + dt * dTdt
    # 数值下限/上限保护：T 不低于环境，不高于沸点的若干倍（防显式格式偶发振荡）
    T_new = jnp.clip(T_new, mat.T_ambient - 50.0, 1.5 * mat.T_boil)
    # 底部基板视为等温热沉（近似深基板的散热能力）
    T_new = T_new.at[:, :, 0].set(mat.T_ambient)

    # ---- 动量方程 ----------------------------------------------------
    fl_new = mat.liquid_fraction(T_new)
    sigma = mat.sigma_of_T(T)
    nhat = gF / jnp.maximum(gmag, 1e-12)[..., None]
    # 曲率/法向用平滑界面估计（消除 1 体素锐界面对中心差分的曲率尖峰，
    # 否则表面张力会驱动速度场发散）。锐界面对输运与力定位仍用 gF/gmag。
    Fs = _smooth_field(F, passes=2)
    gFs = _grad(Fs, dx)
    gn = _norm(gFs)
    nhat_s = gFs / jnp.maximum(gn, 1e-12)[..., None]
    kappa = -_div(nhat_s, dx)                     # 曲率（平滑）
    f_st = sigma[..., None] * kappa[..., None] * gF

    gT = _grad(T, dx)
    gT_n = jnp.sum(gT * nhat_s, axis=-1, keepdims=True) * nhat_s
    f_ma = mat.dsigma_dT * (gT - gT_n) * gmag[..., None]

    p_recoil = jnp.minimum(mat.recoil_pressure(T), cfg.recoil_pressure_cap)
    f_recoil = p_recoil[..., None] * gF

    # 浮升力（Boussinesq）+ 重力
    buoy = -mat.rho_liquid * GRAVITY * mat.beta_thermal * (T - mat.T_liquidus)
    f_body = jnp.zeros_like(u).at[..., 2].set(buoy * fl - rho * GRAVITY * metal)

    # 糊状区 Darcy 阻力（enthalpy-porosity）
    darcy = cfg.darcy_constant * (1.0 - fl) ** 2 / (fl ** 3 + 1e-3)
    f_darcy = -darcy[..., None] * u

    conv_u = jnp.stack(
        [_advect_flux(u[..., c], u, dx, cfg.advection) for c in range(3)], axis=-1
    )
    visc = jnp.stack([_div_k_grad(jnp.full_like(T, mu), u[..., c], dx)
                      for c in range(3)], axis=-1)

    rho_m = jnp.maximum(rho, 1e-3)[..., None]
    u_star = u + dt * (
        conv_u + (visc + f_st + f_ma + f_recoil + f_body + f_darcy) / rho_m
    )
    u_star = u_star * metal[..., None]
    # 底/侧壁无滑移
    u_star = u_star.at[:, :, 0, :].set(0.0)

    u_new, p_new = _project(u_star, rho, metal, dt, dx, cfg.n_pressure_iters)
    if cfg.gas_damping > 0.0:
        u_new = u_new * (metal + (1.0 - metal) * cfg.gas_damping)[..., None]
    if cfg.velocity_clamp > 0.0:
        speed = jnp.sqrt(jnp.sum(u_new ** 2, axis=-1, keepdims=True))
        u_new = u_new * jnp.minimum(1.0, cfg.velocity_clamp / jnp.maximum(speed, 1e-30))

    # ---- VOF 输运 ----------------------------------------------------
    dF = _advect_flux(F, u_new, dx, cfg.advection)
    if cfg.interface_compression > 0.0:
        # 界面压缩：抵抗数值扩散，把界面维持在 2~3 个体素宽
        uc = cfg.interface_compression * _norm(u_new)[..., None] * nhat
        comp = -_div(F[..., None] * (1.0 - F)[..., None] * uc, dx)
        dF = dF + comp
    # 蒸发质量损失
    dF = dF - mdot / jnp.maximum(mat.rho_liquid, 1e-6) * gmag
    # LSF/DED 送粉：在光斑处按送粉率补充金属
    feed = process.powder_feed_rate
    xs = (jnp.arange(F.shape[0]) + 0.5) * dx
    ys = (jnp.arange(F.shape[1]) + 0.5) * dx
    xL, yL, on = traj(state.time)
    spot = jnp.exp(-2.0 * ((xs[:, None] - xL) ** 2 + (ys[None, :] - yL) ** 2)
                   / process.beam_radius ** 2)
    surf_delta = gmag * jnp.maximum(nhat[..., 2], 0.0)
    norm = jnp.maximum(jnp.sum(spot[:, :, None] * surf_delta) * dx ** 3, 1e-30)
    dF = dF + on * feed / jnp.maximum(mat.rho_liquid, 1e-6) * \
        spot[:, :, None] * surf_delta / norm

    F_new = jnp.clip(F + dt * dF, 0.0, 1.0)

    # 粉末致密化：一旦达到固相线以上就永久变成致密金属
    molten = 0.5 * (1.0 + jnp.tanh((T_new - mat.T_solidus) / 30.0))
    powder_new = powder * (1.0 - molten)

    # ---- 在线诊断累积 ------------------------------------------------
    T_peak = jnp.maximum(state.T_peak, T_new)
    above_sol = 0.5 * (1.0 + jnp.tanh((T_new - mat.T_solidus) / 20.0))
    t_above = state.t_above_melt + dt * above_sol

    # 凝固权重 w = max(-Δf_l, 0)：本步内真正发生凝固的分数
    w = jnp.maximum(fl - fl_new, 0.0)
    gT_mag = _norm(_grad(T_new, dx))
    cool = jnp.maximum(-(T_new - T) / dt, 0.0)
    w_sol = state.w_sol + w
    G_acc = state.G_acc + w * gT_mag
    Rdot_acc = state.Rdot_acc + w * cool

    # 熔深：低于基板顶面且已熔化的最深处
    k_sub = cfg.substrate_top_index()
    melted = fl_new * jnp.clip(F_new, 0.0, 1.0)
    zi = jnp.arange(F.shape[2])
    below = (zi < k_sub).astype(F.dtype)
    prof_z = _smooth_max(melted * below, axis=(0, 1))
    depth_now = jnp.sum(prof_z) * dx
    melt_depth_max = jnp.maximum(state.melt_depth_max, depth_now)

    # 匙孔深度：金属柱高相对初始柱高的最大凹陷
    h_now = jnp.sum(F_new, axis=2) * dx
    h_ref = jnp.sum(state.F0, axis=2) * dx
    depression = jnp.maximum(_smooth_max(h_ref - h_now), 0.0)
    keyhole_max = jnp.maximum(state.keyhole_max, depression)

    vmax = jnp.sqrt(jnp.max(jnp.sum(u_new * u_new, axis=-1)) + 1e-30)
    return MeltPoolState(
        F=F_new, F0=state.F0, T=T_new, u=u_new, p=p_new, powder=powder_new,
        T_peak=T_peak, t_above_melt=t_above, w_sol=w_sol,
        G_acc=G_acc, Rdot_acc=Rdot_acc,
        melt_depth_max=melt_depth_max, keyhole_max=keyhole_max,
        vmax_max=jnp.maximum(state.vmax_max, vmax),
        precoil_max=jnp.maximum(state.precoil_max, jnp.max(p_recoil * gmag) * dx),
        time=state.time + dt,
    )


# ===========================================================================
# 主求解入口
# ===========================================================================
def simulate_meltpool(
    *,
    process: ProcessPlan,
    material: AMMaterial | str = "316L",
    config: MeltPoolConfig | None = None,
    substrate_occupancy=None,
    trajectory: LaserTrajectory | None = None,
    initial_state: MeltPoolState | None = None,
) -> MeltPoolSolution:
    """跑一次熔池 CFD 仿真。

    这是本模块的**低层入口**：返回富信息的 :class:`MeltPoolSolution`，
    契约化封装（:func:`solve_meltpool_vof`）只是它的薄壳。

    Examples
    --------
    >>> from amforge.core import ProcessPlan                      # doctest: +SKIP
    >>> proc = ProcessPlan.uniform(laser_power=250., scan_speed=0.8)  # doctest: +SKIP
    >>> sol = simulate_meltpool(process=proc, material="316L")     # doctest: +SKIP
    >>> float(sol.depth)*1e6, float(sol.width)*1e6                 # doctest: +SKIP

    Notes
    -----
    整个函数可被 ``jax.jit`` / ``jax.grad`` 包裹。``config`` 是静态参数，
    应放进 ``static_argnums``（或用闭包捕获）。
    """
    mat = get_material(material) if isinstance(material, str) else material
    cfg = config or MeltPoolConfig()

    if trajectory is None:
        trajectory = zigzag_trajectory(
            speed=process.scan_speed,
            track_length=min(cfg.track_length, 0.9 * cfg.nx * cfg.dx),
            n_tracks=cfg.n_tracks,
            hatch=process.hatch_spacing,
            center_x=0.5 * cfg.nx * cfg.dx,
            center_y=0.5 * cfg.ny * cfg.dx,
        )

    st = initial_state or build_initial_state(
        cfg, mat, process, substrate_occupancy=substrate_occupancy
    )

    def body(state, _):
        return _step(state, cfg, mat, process, trajectory), None

    if cfg.checkpoint_every and cfg.checkpoint_every > 1:
        # 用重算换显存：把每 ``checkpoint_every`` 步打包成一个 checkpoint 块。
        # 反向时只存块边界状态、块内各步前向重算，显存从 O(n_steps) 降到
        # O(n_steps/ce + ce)，使全程 jax.grad 可微而不 OOM。
        # 注意处理余数步：整除丢弃会少跑 (n_steps % ce) 步，必须补上。
        ce = int(cfg.checkpoint_every)
        n_full = cfg.n_steps // ce
        rem = cfg.n_steps % ce

        @jax.checkpoint
        def block(state, _):
            state, _ = jax.lax.scan(body, state, None, length=ce)
            return state, None

        if n_full > 0:
            st, _ = jax.lax.scan(block, st, None, length=n_full)
        if rem > 0:
            st, _ = jax.lax.scan(body, st, None, length=rem)
    else:
        st, _ = jax.lax.scan(body, st, None, length=cfg.n_steps)

    depth, width, length = _pool_dimensions(st, cfg, mat)
    return MeltPoolSolution(
        state=st, config=cfg, material=mat,
        depth=depth, width=width, length=length,
    )


def _pool_dimensions(st: MeltPoolState, cfg: MeltPoolConfig, mat: AMMaterial):
    """从峰值温度场提取熔池深/宽/长（可微软测度）。

    用 ``T_peak`` 而非末时刻温度：熔池尺寸的工程定义是
    "曾经熔化过的区域"（对应实验中的熔合线金相），
    末时刻场只反映激光离开后的残留熔池，会系统性偏小。
    """
    dx = cfg.dx
    melted = 0.5 * (1.0 + jnp.tanh((st.T_peak - mat.T_solidus) / 25.0))
    melted = melted * jnp.clip(st.F, 0.0, 1.0)

    k_sub = cfg.substrate_top_index()
    zi = jnp.arange(melted.shape[2])
    below = (zi < k_sub).astype(melted.dtype)

    prof_z = _smooth_max(melted * below, axis=(0, 1))
    prof_y = _smooth_max(melted, axis=(0, 2))
    prof_x = _smooth_max(melted, axis=(1, 2))

    depth = jnp.sum(prof_z) * dx
    width = jnp.sum(prof_y) * dx
    length = jnp.sum(prof_x) * dx
    return depth, width, length


# ===========================================================================
# 契约装配
# ===========================================================================
def _assemble_meltpool_result(sol: MeltPoolSolution, process: ProcessPlan) -> MeltPoolResult:
    """把求解结果装成 :class:`MeltPoolResult`，并计算三类缺陷指标。

    缺陷判据的物理依据
    ------------------
    * **未熔合 (LoF)**：熔深必须穿透当前层并重熔前一层
      （工程经验 ``depth/layer_thickness ≳ 1.5``），
      同时熔宽必须大于扫描间距以保证道间搭接。
    * **匙孔气孔**：深宽比 > ~1 进入匙孔模式，
      叠加峰值温度超过沸点 -> 匙孔壁不稳定、底部塌陷捕获气泡。
    * **飞溅**：反冲压力相对表面张力的量级（类 Weber 数）
      与表面流速共同决定液滴脱离倾向。
    """
    st, cfg, mat = sol.state, sol.config, sol.material
    fl = mat.liquid_fraction(st.T)

    lt = process.layer_thickness
    hatch = process.hatch_spacing

    pen = sol.depth / jnp.maximum(lt, 1e-9)
    overlap = sol.width / jnp.maximum(hatch, 1e-9)
    lof = jnp.clip(
        0.6 * jnp.clip(1.0 - pen / 1.5, 0.0, 1.0)
        + 0.4 * jnp.clip(1.0 - overlap, 0.0, 1.0),
        0.0, 1.0,
    )

    ar = sol.depth / jnp.maximum(sol.width, 1e-12)
    boil_excess = (jnp.max(st.T_peak) - mat.T_boil) / 200.0
    porosity = jax.nn.sigmoid((ar - 1.0) / 0.25) * jax.nn.sigmoid(boil_excess)

    we = st.precoil_max * process.beam_radius / jnp.maximum(mat.sigma0, 1e-6)
    spatter = jnp.clip(
        0.5 * jax.nn.sigmoid((we - 1.0) / 0.5)
        + 0.5 * jax.nn.sigmoid((st.vmax_max - 4.0) / 2.0),
        0.0, 1.0,
    )

    return MeltPoolResult(
        vof=st.F,
        temperature=st.T,
        liquid_fraction=fl,
        velocity=st.u,
        pressure=st.p,
        depth=sol.depth,
        width=sol.width,
        length=sol.length,
        keyhole_depth=st.keyhole_max,
        lof_indicator=lof,
        porosity_indicator=porosity,
        spatter_indicator=spatter,
        spacing=cfg.dx,
        dim=3,
    )


def _assemble_thermal_history(sol: MeltPoolSolution) -> ThermalHistory:
    """把在线累积的凝固统计装成 :class:`ThermalHistory`。

    G 与 Ṫ 都按"凝固发生量"加权平均 —— 这比"取某个时刻的瞬时值"稳健得多，
    因为同一个点可能被多道扫描反复重熔，只有最后一次凝固才决定最终组织。
    R = Ṫ / G 由 Ṫ = G·R 反解。
    """
    st, cfg = sol.state, sol.config
    w = jnp.maximum(st.w_sol, 1e-12)
    G = st.G_acc / w
    Rdot = st.Rdot_acc / w
    R = Rdot / jnp.maximum(G, 1e-6)
    return ThermalHistory(
        peak_temperature=st.T_peak,
        cooling_rate=Rdot,
        thermal_gradient=G,
        solidification_rate=R,
        time_above_melt=st.t_above_melt,
        final_temperature=st.T,
        spacing=cfg.dx,
        dim=3,
    )


# ===========================================================================
# 与零件几何的对接
# ===========================================================================
def substrate_from_part(
    part: PartGeometry,
    cfg: MeltPoolConfig,
    *,
    build_height: float | None = None,
    center: tuple[float, float] | None = None,
) -> jnp.ndarray:
    """在零件几何上开一个局部窗口，采样出基板占位场。

    这是"任意复杂几何 → 熔池求解器"的连接件：把局部计算域套在零件的
    某个位置与构建高度上，用 SDF 采样得到该处到底有多少实体金属。
    悬垂下方是空的、薄壁两侧是空的，这些几何差异会直接改变局部散热，
    进而改变熔池尺寸与缺陷倾向 —— 这正是"任意几何都能模拟"的实际含义。
    """
    from amforge.geometry import sample_sdf

    nx, ny, nz = cfg.shape
    dx = cfg.dx
    lo = np.asarray(part.origin, dtype=np.float64)
    hi = lo + part.spacing * (np.asarray(part.shape) - 1.0)

    if center is None:
        center = (0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]))
    if build_height is None:
        build_height = float(hi[2])

    k_sub = cfg.substrate_top_index()
    xs = center[0] + dx * (jnp.arange(nx) - 0.5 * nx)
    ys = center[1] + dx * (jnp.arange(ny) - 0.5 * ny)
    zs = build_height + dx * (jnp.arange(nz) - k_sub)
    pts = jnp.stack(jnp.meshgrid(xs, ys, zs, indexing="ij"), axis=-1)
    s = sample_sdf(part, pts)
    return 0.5 * (1.0 - jnp.tanh(s / part.spacing))


# ===========================================================================
# 求解器注册
# ===========================================================================
@register_solver(
    "meltpool.vof_flow3d",
    consumes=("PartGeometry", "ProcessPlan"),
    produces="MeltPoolResult",
    stage="meltpool",
    modality=("SLM", "LSF"),
    differentiable=True,
    cost=100.0,
    method="numerical",
    defaults={"material": "316L",
              "config": MeltPoolConfig(nx=40, ny=28, nz=24, n_steps=150),
              "build_height": None, "center": None},
    doc="Flow3D 式 VOF 自由界面熔池 CFD（表面张力/Marangoni/反冲压力/糊状区）",
)
def solve_meltpool_vof(*, geometry: PartGeometry, process: ProcessPlan, params) -> MeltPoolResult:
    """高保真熔池求解器（契约封装）。"""
    p = dict(params or {})
    cfg = p.get("config") or MeltPoolConfig()
    mat = p.get("material", "316L")
    occ = substrate_from_part(
        geometry, cfg,
        build_height=p.get("build_height"), center=p.get("center"),
    )
    sol = simulate_meltpool(
        process=process, material=mat, config=cfg, substrate_occupancy=occ
    )
    return sol.to_meltpool(process)


@register_solver(
    "thermal.vof_resolved",
    consumes=("PartGeometry", "ProcessPlan"),
    produces="ThermalHistory",
    stage="thermal",
    modality=("SLM", "LSF"),
    differentiable=True,
    cost=120.0,
    defaults={"material": "316L", "config": None, "build_height": None, "center": None},
    doc="由 VOF 熔池 CFD 解析出的高保真凝固热历史 (G, R, 冷却速率)",
)
def solve_thermal_from_vof(*, geometry: PartGeometry, process: ProcessPlan, params) -> ThermalHistory:
    """熔池尺度的高保真热历史。

    注意：与 ``meltpool.vof_flow3d`` 是同一次物理求解的两种产物。
    若一条 pipeline 同时需要两个契约，会重复计算一次——
    此时更划算的做法是直接调 :func:`simulate_meltpool` 拿 ``MeltPoolSolution``，
    再自取两个契约。
    """
    p = dict(params or {})
    cfg = p.get("config") or MeltPoolConfig()
    occ = substrate_from_part(
        geometry, cfg,
        build_height=p.get("build_height"), center=p.get("center"),
    )
    sol = simulate_meltpool(
        process=process, material=p.get("material", "316L"),
        config=cfg, substrate_occupancy=occ,
    )
    return sol.to_thermal()


# ---------------------------------------------------------------------------
# 快速代理：Eagar-Tsai 解析熔池（供优化闭环的低成本路径）
# ---------------------------------------------------------------------------
def _representative_scalar(x, layer: int | None = None):
    """把逐层工艺量归约成一个标量。

    工艺契约允许 ``laser_power`` 等是 ``(n_layers,)`` 数组。解析解是
    准稳态单道模型，只能吃标量，所以必须先归约：
    ``layer=None`` 取层平均（代表整件的平均工艺），
    ``layer=k`` 取第 k 层（用于逐层诊断）。
    """
    a = jnp.atleast_1d(jnp.asarray(x))
    if layer is None:
        return jnp.mean(a)
    return a[jnp.clip(layer, 0, a.shape[0] - 1)]


def eagar_tsai_field(
    *, absorbed_power, scan_speed, beam_radius, T0,
    alpha, rho_cp, X, Y, Z, n_quad: int = 32, tau_max=None,
):
    """Eagar-Tsai 移动高斯热源在半无限体上的准稳态温度场 [K]。

    物理推导（便于后人核对，这里给完整链条，因为前系数极易写错）
    --------------------------------------------------------------
    半无限体、表面绝热（镜像源使强度加倍）、瞬时点源在 τ 时刻释放
    ``q·dτ`` 能量，则

    .. math::
        dT = \\frac{2\\,q\\,d\\tau}{\\rho c}\\,
             \\frac{1}{(4\\pi\\alpha\\tau)^{3/2}}
             \\exp\\!\\left(-\\frac{R^2}{4\\alpha\\tau}\\right)

    把核函数按 z 与 (x,y) 分离：``(x,y)`` 部分是逐轴方差 ``2ατ`` 的二维
    高斯。热源本身是逐轴方差 ``σ²`` 的高斯（对 1/e² 半径 ``r_b``，
    强度 ``exp(-2r²/r_b²)`` 对应 ``σ² = r_b²/4``），卷积后方差相加：

    .. math::
        s^2 = 2\\alpha\\tau + \\sigma^2
        \\;\\Longrightarrow\\;
        a \\equiv 2 s^2 = 4\\alpha\\tau + \\tfrac{r_b^2}{2},
        \\qquad b \\equiv 4\\alpha\\tau

    于是（源在 ``+x`` 方向行进，尾迹拖在 ``x<0``）

    .. math::
        T - T_0 = \\frac{2 A P}{\\rho c\\, \\pi^{3/2}}
        \\int_0^\\infty
        \\frac{
          \\exp\\!\\left[-\\frac{(x+v\\tau)^2+y^2}{a}-\\frac{z^2}{b}\\right]
        }{a\\sqrt{b}}\\, d\\tau

    **前系数是** ``2AP/(ρ c π^{3/2})``，**高斯项是** ``r_b²/2`` 而非
    ``2r_b²`` —— 这两处曾各错一次，合起来把温度放大约 2×10⁴ 倍，
    表现为温度场被 ``T_boil`` 上限截断、熔池尺寸退化成采样盒尺寸。

    数值方案：正切代换（本函数的关键）
    -----------------------------------
    历史上这里用 ``τ = τ_max s²``。它能抵消 ``1/√τ`` 奇点，但引入了一个
    **必须由调用方猜对的截断参数**：物理时标是 ``τ_c = r_b²/(8α)``，映到
    ``s_c = √(τ_c/τ_max)``。一旦 ``τ_max ≫ τ_c``，被积函数的峰挤到
    ``s→0`` 的角落，而 Gauss 节点在区间内部只有 ``~1/n`` 的间距，于是峰
    被"饿死"。实测：``τ_max`` 放大 10⁴ 倍使中心温度从 14622 K 漂到
    17051 K（+16.6%）；``n_quad=24`` 时误差达 73%。这是静默错误 —— 调用
    方只要把采样盒开大一点就会中招。

    改用两步代换，把截断参数彻底消掉：

    1. ``τ = u²``（消 ``1/√τ`` 奇点）， 得
       ``I = α^{-1/2}∫_0^∞ du\\, e^{(\\cdot)}/a``，其中
       ``a = 4α(u² + c²)``、``c² = r_b²/(8α)``；
    2. ``u = c\\tan φ``，则 ``du/(u²+c²) = dφ/c`` —— **扩散核被精确吸收**，
       ``φ ∈ [0, π/2]`` 恰好覆盖 ``τ ∈ [0, ∞)``。

    代换后 ``a = (r_b²/2)\\sec²φ``、``b = (r_b²/2)\\tan²φ``、
    ``τ = c²\\tan²φ``，被积函数只剩指数项：

    .. math::
        T - T_0 = \\frac{\\sqrt{2}\\,A P}{\\rho c\\,\\pi^{3/2}\\alpha\\,r_b}
        \\int_0^{\\pi/2}\\exp\\Big[-\\tfrac{2}{r_b^2}\\big(
        (x^2{+}y^2)\\cos^2φ + 2 x v c^2\\sin^2φ
        + (v c^2)^2\\tfrac{\\sin^4φ}{\\cos^2φ}
        + z^2\\tfrac{\\cos^2φ}{\\sin^2φ}\\big)\\Big] dφ

    好处有三：

    * **静止极限精确**：``v=0, x=y=z=0`` 时被积函数 ≡ 1，任意节点数都给出
      解析值 ``(√2/2)·AP/(ρc√π α r_b)``；
    * **无截断参数**，速度项自带上端截断、``z²cot²φ`` 自带下端截断；
    * **良态**：截断位置由 ``\\tan²φ_c ≈ 5.66α/(v r_b)`` 定，实用工况
      (v ∈ [0.1, 5] m/s, r_b ∈ [20, 100] µm) 全落在 ``φ_c ∈ [0.4, 1.4]``
      —— 始终在区间正中，32 点即达 1e-10 量级收敛。

    Parameters
    ----------
    tau_max
        **已废弃且被忽略**。新方案精确积到 ``τ→∞``，无需截断。保留该形参
        仅为兼容旧调用；传值不会有任何效果。
    """
    del tau_max                                    # 见 docstring：已废弃
    phi_n, phi_w = np.polynomial.legendre.leggauss(int(n_quad))
    half = 0.25 * np.pi                            # [-1,1] → [0, π/2]
    phi = jnp.asarray(half * (phi_n + 1.0))[:, None, None, None]
    w = jnp.asarray(half * phi_w)[:, None, None, None]

    c2 = beam_radius ** 2 / (8.0 * alpha)          # c² = r_b²/(8α)
    sin2 = jnp.sin(phi) ** 2
    cos2 = jnp.cos(phi) ** 2
    # 两个比值项：φ→π/2 时 sin⁴/cos² → ∞（速度截断）；
    #            φ→0    时 cos²/sin² → ∞（深度截断）。
    # 各自加下限保护，避免 0/0；分子分母都有界，无中间溢出风险。
    r4_over_c2 = sin2 ** 2 / jnp.maximum(cos2, 1e-300)
    c2_over_s2 = cos2 / jnp.maximum(sin2, 1e-300)

    vc2 = scan_speed * c2
    expo = -(2.0 / beam_radius ** 2) * (
        (X ** 2 + Y ** 2) * cos2
        + 2.0 * X * vc2 * sin2
        + vc2 ** 2 * r4_over_c2
        + Z ** 2 * c2_over_s2
    )
    integral = jnp.sum(w * jnp.exp(expo), axis=0)
    pref = (jnp.sqrt(2.0) * absorbed_power
            / (rho_cp * jnp.pi ** 1.5 * alpha * beam_radius))
    return T0 + pref * integral


def _melt_length_scale(*, absorbed_power, scan_speed, beam_radius,
                       rho, enthalpy_to_melt):
    """熔池特征尺度 [m]，用于自适应确定采样盒大小。

    能量平衡：``A·P ≈ ρ·Δh·v·A_cross``（Δh 含显热 + 熔化潜热），
    半圆形熔池截面 ``A_cross = πR²/2`` 给出

    .. math:: R = \\sqrt{\\frac{2 A P}{\\pi \\rho \\Delta h\\, v}}

    这是**上界**估计（忽略向前方基体的传导损失，实际熔化效率
    仅 30~50%），正好适合当采样盒尺度 —— 宁大勿小，盒子截断熔池
    会让尺寸读数直接等于盒子尺寸（就是之前那个 bug 的表现）。
    """
    A_cross = absorbed_power / jnp.maximum(
        rho * enthalpy_to_melt * scan_speed, 1e-30)
    R = jnp.sqrt(jnp.maximum(2.0 * A_cross / jnp.pi, 0.0))
    return jnp.maximum(R, 1.5 * beam_radius)


@register_solver(
    "meltpool.surrogate_eagar_tsai",
    consumes=("PartGeometry", "ProcessPlan"),
    produces="MeltPoolResult",
    stage="meltpool",
    modality=("SLM", "LSF"),
    differentiable=True,
    cost=1.0,
    method="analytical",
    defaults={"material": "316L", "n_grid": 32, "n_quad": 48, "props": "mean"},
    doc="Eagar-Tsai 移动高斯热源解析解 + 经验缺陷判据（毫秒级，用于工艺窗口扫描）",
)
def solve_meltpool_surrogate(*, geometry: PartGeometry, process: ProcessPlan,
                             params) -> MeltPoolResult:
    """半解析熔池代理模型 —— 比 VOF 快约 4~5 个数量级。

    模型
    ----
    :func:`eagar_tsai_field` 给出的移动高斯热源准稳态解。在**传导模式**下
    对熔宽/熔深的预测通常在 ±20~30% 以内；进入匙孔模式后解析解不再
    适用（无自由界面、无反冲压），此时应切到 ``meltpool.vof_flow3d``。

    为什么需要它
    ------------
    功能 2 的优化闭环要跑成百上千次前向仿真。直接用 VOF 在算力上不现实，
    正确的工程做法是**代理模型做粗优化 + VOF 做关键点验证**
    （可行性报告推荐的混合路径）。两者产出同一个 :class:`MeltPoolResult`
    契约，在 pipeline 里只需改 ``select`` 一行即可互换。

    Parameters (via ``params``)
    ---------------------------
    material : str | AMMaterial
        材料，默认 ``"316L"``。
    n_grid : int
        每个方向的采样点数（默认 32）。
    n_quad : int
        Gauss-Legendre 节点数（默认 48）。
    props : {"mean", "solid", "liquid"}
        取哪组热物性算扩散率。默认 ``"mean"`` —— 固相 k 偏低会低估熔池，
        液相 k 偏高会高估，取均值是文献常用折中。
    layer : int | None
        逐层工艺时取第几层；``None`` 取层平均。
    """
    p = dict(params or {})
    mat_in = p.get("material", "316L")
    mat = get_material(mat_in) if isinstance(mat_in, str) else mat_in
    n = int(p.get("n_grid", 32))
    n_quad = int(p.get("n_quad", 48))
    props = str(p.get("props", "mean"))
    layer = p.get("layer", None)

    # --- 工艺归约 ---------------------------------------------------------
    P_laser = _representative_scalar(process.laser_power, layer)
    v = jnp.maximum(_representative_scalar(process.scan_speed, layer), 1e-6)
    A = _representative_scalar(process.absorption, layer)
    rb = _representative_scalar(process.beam_radius, layer)
    T0 = _representative_scalar(process.preheat_temp, layer)
    t_layer = _representative_scalar(process.layer_thickness, layer)
    hatch = _representative_scalar(process.hatch_spacing, layer)
    P = P_laser * A

    # --- 热物性 -----------------------------------------------------------
    if props == "solid":
        k, rho, cp = mat.k_solid, mat.rho_solid, mat.cp_solid
    elif props == "liquid":
        k, rho, cp = mat.k_liquid, mat.rho_liquid, mat.cp_liquid
    else:
        k = 0.5 * (mat.k_solid + mat.k_liquid)
        rho = 0.5 * (mat.rho_solid + mat.rho_liquid)
        cp = 0.5 * (mat.cp_solid + mat.cp_liquid)
    rho_cp = rho * cp
    alpha = k / rho_cp

    # --- 自适应采样盒 -----------------------------------------------------
    dh_melt = cp * jnp.maximum(mat.T_liquidus - T0, 1.0) + mat.latent_fusion
    L = _melt_length_scale(absorbed_power=P, scan_speed=v, beam_radius=rb,
                           rho=rho, enthalpy_to_melt=dh_melt)
    # 熔池拖在光斑后方（-x），故 x 向后留 6L、前留 1.5L
    xs = jnp.linspace(-6.0 * L, 1.5 * L, n)
    ys = jnp.linspace(-2.5 * L, 2.5 * L, n)
    zs = jnp.linspace(0.0, -2.5 * L, n)
    X, Y, Z = jnp.meshgrid(xs, ys, zs, indexing="ij")

    # 正切代换精确积到 τ→∞，不再需要 tau_max（见 eagar_tsai_field docstring）
    T_raw = eagar_tsai_field(
        absorbed_power=P, scan_speed=v, beam_radius=rb, T0=T0,
        alpha=alpha, rho_cp=rho_cp, X=X, Y=Y, Z=Z, n_quad=n_quad)
    T = T_raw
    # 蒸发饱和 —— 纯导热解在高斯源中心是准奇异的。解析中心温度
    #   T_c - T_0 = (√2/2)·A·P / (ρ c √π α r_b)
    # 对 88 W / r_b = 50 µm 的 Ti6Al4V 算出 ~3.8×10⁴ K，显然非物理。
    # 真实熔池被三件事钉住：熔化/汽化潜热、Marangoni 对流、蒸发吸热。
    # 这里用 softplus 型光滑下确界把温度压到沸点附近：
    #   smin(T, T_b) = T_b - s·softplus((T_b - T)/s)
    # 低温端严格还原 T（软化误差 < 1e-3 K），高温端渐近 T_b。
    # 高温端梯度趋零**是物理正确的**：过沸点后增加功率主要转成汽化
    # 而非升温，优化信号应来自熔池尺寸与缺陷指标，而不是峰值温度。
    T_sat = 250.0
    T = mat.T_boil - T_sat * jax.nn.softplus((mat.T_boil - T) / T_sat)

    dx_s = jnp.abs(xs[1] - xs[0])
    dy_s = jnp.abs(ys[1] - ys[0])
    dz_s = jnp.abs(zs[1] - zs[0])

    # --- 熔池尺寸：软熔化指示函数的投影积分（亚体素精度）-----------------
    # 过渡带宽取 ~1 个体素对应的温升，使 Σσ·dz 成为等值面位置的
    # 二阶精确估计；固定 25 K 在低功率时过窄、高功率时过宽。
    dT_band = jnp.maximum(0.02 * (jnp.max(T) - T0), 15.0)
    melted = jax.nn.sigmoid((T - mat.T_solidus) / dT_band)
    depth_cond = jnp.sum(_smooth_max(melted, axis=(0, 1))) * dz_s
    width = jnp.sum(_smooth_max(melted, axis=(0, 2))) * dy_s
    length = jnp.sum(_smooth_max(melted, axis=(1, 2))) * dx_s

    # --- 蒸发凹陷（keyhole depression）------------------------------------
    # 纯导热解系统性低估熔深 20~30%，机制是它缺少**凹陷使热源下沉**：
    # 表面过热汽化 → 反冲压压出凹坑 → 激光作用点整体下移 → 熔深增加。
    #
    # 驱动量取"过热汽化面积" A_vap = ∫∫ σ((T_raw − T_boil)/ΔT) dxdy，
    # 即 z=0 面上超过沸点的区域面积。它的物理优点：
    #   * 低于沸点时自动为 0 —— 传导模式下不引入任何虚假深度；
    #   * 随功率增大、随速度减小而增大，趋势与实验一致；
    #   * 用 T_raw（未饱和）算，避免饱和后失去分辨力。
    # 凹陷深度取其线尺度：d_dep = c_dep · √A_vap。
    #
    # c_dep 由 5 组金属工况（Ti6Al4V ×2 / 316L ×2 / IN718）标定得
    # 0.154 ± 0.038。**这是半经验系数**：参考熔池尺寸取自公开文献的
    # 典型量级而非同一套受控实验，因此换机器/换粉批时应重新标定
    # （改 params["depression_coeff"] 即可）。熔深的权威结果请用
    # ``meltpool.vof_flow3d`` —— 它显式解自由界面与反冲压，无需此系数。
    c_dep = float(p.get("depression_coeff", 0.15))
    A_vap = jnp.sum(
        jax.nn.sigmoid((T_raw[:, :, 0] - mat.T_boil) / 60.0)) * dx_s * dy_s
    keyhole = c_dep * jnp.sqrt(jnp.maximum(A_vap, 0.0))
    depth = depth_cond + keyhole

    # --- 缺陷判据 ---------------------------------------------------------
    nh = process.normalized_enthalpy(
        rho=rho, cp=cp, T_melt=mat.T_liquidus, diffusivity=alpha)
    nh = _representative_scalar(nh, layer)

    pen = depth / jnp.maximum(t_layer, 1e-12)
    overlap = width / jnp.maximum(hatch, 1e-12)
    lof = jnp.clip(
        0.6 * jnp.clip(1.0 - pen / 1.5, 0.0, 1.0)
        + 0.4 * jnp.clip(1.0 - overlap, 0.0, 1.0), 0.0, 1.0)
    ar = depth / jnp.maximum(width, 1e-12)
    porosity = jax.nn.sigmoid((ar - 1.0) / 0.25) * jax.nn.sigmoid((nh - 30.0) / 10.0)
    spatter = jax.nn.sigmoid((nh - 45.0) / 12.0)

    zero = jnp.zeros_like(T)
    return MeltPoolResult(
        vof=jnp.ones_like(T),
        temperature=T,
        liquid_fraction=mat.liquid_fraction(T),
        velocity=jnp.zeros(T.shape + (3,)),
        pressure=zero,
        depth=depth, width=width, length=length,
        keyhole_depth=keyhole,
        lof_indicator=lof,
        porosity_indicator=porosity,
        spatter_indicator=spatter,
        spacing=dz_s,
        dim=3,
    )


# ===========================================================================
# 便宜真实数值熔池：3D 瞬态焓法热传导（无自由界面）—— 默认熔池求解器
# ===========================================================================
def _laser_source_fdm(coords, x_t, *, rb, A_eff, P, absorption_depth, dx):
    """移动激光的 Beer-Lambert 体积吸收热源（可微、功率守恒）。

    面内二维高斯 + 沿深度指数衰减（取代旧的表面高斯-z 近似）：

        q(x,y,z) = A_eff · P · exp(−r²/rb²) · exp(−z_d/δ) / (π·rb²·δ)

    其中 ``z_d = −z``（网格 z 向下为负，取非负），``δ = absorption_depth``。
    归一化使 ∫q dV = A_eff·P（总吸收功率守恒），且表面 (z_d=0) 吸收最强、
    随深度指数衰减——比表面高斯更贴近真实激光穿透。``A_eff`` 为总吸收分数
    （含 Fresnel/表面反射），``δ`` 为有效吸收深度（受网格分辨率限制，取数倍体素）。
    全程纯 jnp 运算，可被 ``jax.grad`` 穿透。
    """
    planar2 = (coords[..., 0] - x_t) ** 2 + coords[..., 1] ** 2
    zd = jnp.maximum(-coords[..., 2], 0.0)                       # 进入材料深度（正）
    planar = jnp.exp(-planar2 / jnp.maximum(rb * rb, 1e-18))
    depth_decay = jnp.exp(-zd / jnp.maximum(absorption_depth, dx))
    norm = (jnp.pi * rb * rb) * jnp.maximum(absorption_depth, dx) + 1e-18
    return A_eff * P * planar * depth_decay / norm


@register_solver(
    "meltpool.fdm",
    consumes=("PartGeometry", "ProcessPlan"),
    produces="MeltPoolResult",
    stage="meltpool",
    modality=("SLM", "LSF"),
    differentiable=True,
    cost=8.0,
    method="numerical",
    defaults={"material": "316L", "n_grid": 32, "n_steps": 120},
    doc="3D 瞬态焓法热传导熔池（移动高斯热源+固液潜热）：数值 FVM，无自由界面，便宜真实档",
)
def solve_meltpool_fdm(*, geometry: PartGeometry, process: ProcessPlan,
                       params) -> MeltPoolResult:
    """便宜的真实数值熔池求解器（传导模式，无 VOF 自由界面）。

    与 ``meltpool.vof_flow3d`` 共享同一套物理内核（移动高斯热源 + 焓法相变潜热
    + Neumann 边界），但只在固定笛卡尔盒上做热传导数值积分，**不求解
    Navier-Stokes / VOF 自由界面**，因此快 1~2 个数量级、可微、且数值稳定。

    这是对用户硬原则"仿真=数值方法，不得用闭式/解析解作默认"的落地：默认熔池
    求解器走真实离散计算（FVM 焓法），而非 Eagar-Tsai 闭式解析。需要钥匙孔 /
    Marangoni / 反冲压力等高保真效应时，用
    ``select={'meltpool':'meltpool.vof_flow3d'}`` 切到 VOF CFD。

    数值方案
    --------
    * 局部计算盒：尺度由能量平衡上界 ``L = sqrt(2·A·P/(π·ρ·Δh·v))`` 定
      （与 Eagar-Tsai 代理同源，保证熔池不被采样盒截断）；体素 ``dx≈L/n_grid``。
    * 时间推进：显式 SSP-RK2，步长受 CFL/扩散稳定限与单次扫描时长约束。
    * 相变：焓法（Voller-Prakash），糊状区有效扩散系数压低 → 潜热平台，全程可微。
    * 导热系数随温度变：固→液 ``k`` 跳变（金属可差 2~3 倍）经液相分数线性混合，
      仅变 k、ρcp 保持参考常数用于焓↔温映射（"变导热、常密度"标准近似）。
    * 热源：Beer-Lambert 体积吸收（面内高斯 × 深度指数衰减），功率守恒、
      比表面高斯更接近真实激光穿透。
    * 峰值温度场 ``T_peak`` 跟踪熔池形貌；熔化指示用软 sigmoid 投影积分取
      亚体素精度的深/宽/长。
    """
    p = dict(params or {})
    mat_in = p.get("material", "316L")
    mat = get_material(mat_in) if isinstance(mat_in, str) else mat_in
    n = int(p.get("n_grid", 32))
    n_steps = int(p.get("n_steps", 120))

    # 工艺量保持为 JAX tracer（可微：梯度可回传到激光功率/速度/光斑等）；
    # 物性取材料常量（不可微）。这是与 Eagar-Tsai 代理一致的写法
    # （代理用 jnp.linspace 建网格，全程 tracer）。
    # 工艺量可能是逐层数组（如 heuristic 规划器输出），先用 _representative_scalar
    # 归约为代表标量（取层平均），否则 3D 网格构造会因非 1D 维度而失败——
    # 传导模式 FVM 只吃单道标量工艺。
    rb = _representative_scalar(process.beam_radius)
    v = _representative_scalar(process.scan_speed)
    A = _representative_scalar(process.absorption)
    P = _representative_scalar(process.laser_power)
    T0 = _representative_scalar(process.preheat_temp)
    t_layer = _representative_scalar(process.layer_thickness)
    hatch = _representative_scalar(process.hatch_spacing)

    rho = 0.5 * (mat.rho_solid + mat.rho_liquid)
    cp = 0.5 * (mat.cp_solid + mat.cp_liquid)
    k = float(mat.k_solid)
    L = float(mat.latent_fusion)
    T_amb = float(mat.T_ambient)
    T_sol = float(mat.T_solidus)
    T_liq = float(mat.T_liquidus)

    # 局部计算盒（与 Eagar-Tsai 代理同源尺度，保证熔池不被截断）
    dh_melt = cp * jnp.maximum(T_liq - T0, 1.0) + L
    Lscale = _melt_length_scale(absorbed_power=A * P, scan_speed=jnp.maximum(v, 1e-6),
                                 beam_radius=rb, rho=rho, enthalpy_to_melt=dh_melt)
    xs = jnp.linspace(-6.0 * Lscale, 1.5 * Lscale, n)
    ys = jnp.linspace(-2.5 * Lscale, 2.5 * Lscale, n)
    zs = jnp.linspace(0.0, -2.5 * Lscale, n)
    dx = xs[1] - xs[0]
    coords = jnp.stack(jnp.meshgrid(xs, ys, zs, indexing="ij"), axis=-1)

    # 面内 2D 高斯 + 深度指数衰减热源（Beer-Lambert 体积吸收，见 _laser_source）
    r_eff = jnp.maximum(rb, dx)
    dp = jnp.maximum(r_eff * 1.2, dx)                       # 有效吸收深度（受网格限制）

    x0 = xs[0]
    x1 = xs[-1]
    scan_time = jnp.maximum((x1 - x0) / jnp.maximum(v, 1e-6), dx)
    alpha0 = k / (rho * cp + 1e-12)
    dt_stab = 0.3 * dx * dx / (2.0 * 3.0 * alpha0 + 1e-18)
    dt = jnp.minimum(dt_stab, scan_time / max(n_steps, 1))

    def source_at(xs_t):
        return _laser_source_fdm(coords, xs_t, rb=r_eff, A_eff=A, P=P,
                                 absorption_depth=dp, dx=dx)

    hcool = 0.5  # 弱全局对流冷却（与 enthalpy 热解一致的数值封顶）

    def step(carry, t):
        # t 是 lax.scan 的扫描索引（0..n_steps-1），需乘 dt 转为物理时间，
        # 否则光束会被丢到计算盒外（之前 bug：不乘 dt 时光束位置 ~60mm，完全不加热）。
        t_phys = t * dt
        Ht, peak = carry
        Tt = temperature_of_enthalpy(Ht, rho=rho, cp=cp, L=L, T_amb=T_amb,
                                     T_sol=T_sol, T_liq=T_liq)
        # 温度相关导热：固→液 k 跳变（金属 2~3 倍）对熔池形貌影响显著。
        # 仅变 k（ρcp 保持参考常数用于焓↔温映射，避免改写 H(T) 定义），
        # 即"变导热系数、常密度"的标准近似；纯 jnp 运算，可微。
        fl_t = mat.liquid_fraction(Tt)
        kT = mat.k_of(fl_t)
        alpha = effective_diffusivity(Tt, k=kT, rho=rho, cp=cp, L=L,
                                       T_sol=T_sol, T_liq=T_liq)
        lap = _div_alpha_grad(Ht, alpha, dx)
        k1 = lap + source_at(x0 + v * t_phys) - hcool * (Tt - T_amb)
        H1 = Ht + dt * k1
        # SSP-RK2 二阶（复用 enthalpy 热解的稳定格式）
        T1 = temperature_of_enthalpy(H1, rho=rho, cp=cp, L=L, T_amb=T_amb,
                                     T_sol=T_sol, T_liq=T_liq)
        fl_1 = mat.liquid_fraction(T1)
        k1_ = mat.k_of(fl_1)
        a1 = effective_diffusivity(T1, k=k1_, rho=rho, cp=cp, L=L,
                                   T_sol=T_sol, T_liq=T_liq)
        lap1 = _div_alpha_grad(H1, a1, dx)
        k2 = lap1 + source_at(x0 + v * (t_phys + dt)) - hcool * (T1 - T_amb)
        Hn = Ht + 0.5 * dt * (k1 + k2)
        Tn = temperature_of_enthalpy(Hn, rho=rho, cp=cp, L=L, T_amb=T_amb,
                                     T_sol=T_sol, T_liq=T_liq)
        peak = jnp.maximum(peak, Tn)
        return (Hn, peak), None

    # 初始场：预热到 T0（H = ρ·cp·(T0−T_amb)，糊状区 f=0），而非冷态 T_amb
    H0 = rho * cp * (T0 - T_amb) * jnp.ones_like(coords[..., 0])
    peak0 = jnp.full_like(H0, T0)
    (Hf, peak_T), _ = jax.lax.scan(step, (H0, peak0), jnp.arange(n_steps))
    Tf = temperature_of_enthalpy(Hf, rho=rho, cp=cp, L=L, T_amb=T_amb,
                                 T_sol=T_sol, T_liq=T_liq)

    # ---- 熔池尺寸：软化熔化指示函数的投影积分（亚体素精度）----
    dT_band = jnp.maximum(0.02 * (jnp.max(peak_T) - T0), 15.0)
    melted = jax.nn.sigmoid((peak_T - T_sol) / dT_band)
    prof_z = _smooth_max(melted, axis=(0, 1))   # 沿 z 投影（熔深方向）
    prof_y = _smooth_max(melted, axis=(0, 2))
    prof_x = _smooth_max(melted, axis=(1, 2))
    depth = jnp.sum(prof_z) * dx
    width = jnp.sum(prof_y) * dx
    length = jnp.sum(prof_x) * dx

    # ---- 缺陷判据（与 Eagar-Tsai 代理同源，工程可读）----
    pen = depth / jnp.maximum(t_layer, 1e-12)
    overlap = width / jnp.maximum(hatch, 1e-12)
    lof = jnp.clip(
        0.6 * jnp.clip(1.0 - pen / 1.5, 0.0, 1.0)
        + 0.4 * jnp.clip(1.0 - overlap, 0.0, 1.0), 0.0, 1.0)
    ar = depth / jnp.maximum(width, 1e-12)
    # 归一化焓（keyhole 判据）：必须用*代表*标量归约。process.heuristic 下
    # normalized_enthalpy 读到的 scan_speed/beam_radius 是逐层数组 -> nh 变成
    # (n_layers,) -> porosity/lof/spatter 全变数组，破坏 MeltPoolResult“缺陷指标
    # =标量”契约（下游 micro 3D 实体场广播会失败）。FDM 几何已用代表标量，缺陷
    # 指标须一致；_representative_scalar 与几何处同义（取层平均）。
    nh = _representative_scalar(process.normalized_enthalpy(
        rho=mat.rho_solid, cp=mat.cp_solid, T_melt=mat.T_melt,
        diffusivity=k / (rho * cp)))
    porosity = jax.nn.sigmoid((ar - 1.0) / 0.25) * jax.nn.sigmoid((nh - 30.0) / 10.0)
    spatter = jax.nn.sigmoid((nh - 45.0) / 12.0)

    zero = jnp.zeros_like(Tf)
    return MeltPoolResult(
        vof=jnp.ones_like(Tf),
        temperature=peak_T,
        liquid_fraction=mat.liquid_fraction(peak_T),
        velocity=jnp.zeros(Tf.shape + (3,)),
        pressure=zero,
        depth=depth, width=width, length=length,
        keyhole_depth=jnp.array(0.0),
        lof_indicator=lof,
        porosity_indicator=porosity,
        spatter_indicator=spatter,
        spacing=jnp.asarray(dx),
        dim=3,
    )
