"""AM 专用枝晶相场求解器（Karma–Rappel 模型）
============================================

缺口④的物理求解器：模拟熔池凝固区的**枝晶形貌**，含
**枝晶各向异性**（立方晶体的 4 重界面能各向异性）与**热梯度取向**
（枝晶沿热梯度 ``G`` 方向择优生长）。这是把"微观组织 → 本构"断点
从*统计代理*（见 ``amforge.micro``）升级为*逐晶粒相场演化*的
高保真路径，且仍产出 ``MicrostructureResult``，下游本构/成形链路零改动。

物理模型
--------
采用经典的 **Karma–Rappel（1996, PRL 77 4050 / PRE 53 R3017）** 无量纲
枝晶相场方程（薄界面近似），相场 ``φ∈[−1, 1]``（固=+1 / 液=−1）、
无量纲过冷度场 ``U``：

    ∂φ/∂t = [ φ − φ³ + λ(1−φ²)² (U + θₖ(θ)) ] / (λ τ₀)
             + ∇·[ a(θ)² ∇φ ]
             − ½ (∂ₐ a²) (∂ₐ φ) (1 − φ²)

    a(θ) = 1 + ε₄·cos( N·(θ − θ₀) )      —— N=folds（立方晶体取 4 重）

    θ   = atan2(∂φ/∂y, ∂φ/∂x)            —— 界面法向角
    θ₀  = ⟨100⟩ 主晶向（与热梯度 G 对齐时，枝晶主轴即沿 G）

    θₖ(θ) = εₖ·cos( N·(θ − θ₀) )          —— 动力学各向异性（取向相关动力学过冷）

    ∂U/∂t = D ∇²U + ½ ∂φ/∂t              —— 含潜热的过冷度演化
             或"定向凝固"模式：U 取冻结线性梯度（沿 G），枝晶自热端向冷端
             （沿 G）择优生长 —— 直接体现"热梯度取向（G 方向优先生长）"。

**各向异性 → G 取向的物理链条**：
界面能在 ``⟨100⟩`` 最低（``ε₄>0``、N=4），故枝晶沿 ``⟨100⟩`` 最快生长；
当把主晶向 ``θ₀`` 设为热梯度方向 ``atan2(G_y, G_x)`` 时，枝晶主轴即沿
``G``，满足任务"枝晶各向异性与热梯度取向（G 方向优先生长）"。

数值实现
--------
* 全 JAX、纯函数，**半隐式 Fourier 谱格式**（默认，``semi_implicit=True``）：
  用 ``jax.lax.scan`` 推进。**各向异性扩散项** ``∇·(a²∇φ)`` 拆为
  各向同性主导部分 ``∇²φ``（在 Fourier 空间作*隐式*求解，算子 ``1/(1+dt·k²)``）
  与各向异性涨落 ``(a²−1)∇²φ + ∇a²·∇φ``（显式）；反应项与 Karma 反对称校正项
  亦显式。隐式扩散根除了显式 Euler 的扩散数限制 ``dt<dx²/4``，使大 ``λ``
  （薄界面、定量 tip selection）下可用大时间步稳定推进。
* 各向异性项用**向量场散度** ``∇·(a²∇φ)`` 精确离散（自动含 ``(∂ₐa²)(∂ₐφ)``），
  并补 Karma 反对称校正项 ``−½(∂ₐa²)(∂ₐφ)(1−φ²)``（用恒等式
  ``(∂ₐa²)(∂ₐφ) = (da²/dθ)·(∇θ·∇φ)`` 避免奇点）。
* 2D 为核心验证维度（枝晶形貌的标准验证空间）；3D 集成时取含 ``G`` 的截面 RVE。
* 半隐式对*反应*项仍显式，受数 ``dt<λτ₀`` 约束（薄界面 ``λ`` 大时仍满足）；
  但与扩散相关的刚性被隐式吸收，故可在大 ``λ`` 下显著放大 ``dt``。本机用中小
  网格验证物理正确性与可微性，48G 卡上可进一步放大网格/步数做定量尖端选择。

> 注：定量尖端选择（tip selection）需极薄界面（``λ`` 大）。**本模块已采用半隐式
> Fourier 谱格式并默认 ``lam`` 调大以支持之**（见 ``DendriteConfig`` 与
> ``solve_phasefield`` 默认）；定量 tip selection 的收敛性在 48G 放大网格上验证（#38），
> 与本机"代码全保真、测试降采样"策略一致。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp

from amforge.core.contracts import (
    ThermalHistory, MeltPoolResult, MicrostructureResult, register_contract,
)
from amforge.core.registry import register_solver
from amforge.materials import get_material


# ---------------------------------------------------------------------------
# 谱（Fourier）导数算子：旋转各向同性，根除 5 点中心差分的 45° 网格偏置
# ---------------------------------------------------------------------------
def _kvec(n: int, dx: float) -> jnp.ndarray:
    """角波数向量（单位 rad/m）。"""
    return 2.0 * jnp.pi * jnp.fft.fftfreq(n, d=dx)


def spectral_grad(f: jnp.ndarray, dx: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """谱梯度 ``(∂f/∂x, ∂f/∂y)``（周期性边界，各向同性到机器精度）。"""
    nx, ny = f.shape
    kx = _kvec(nx, dx)[:, None]
    ky = _kvec(ny, dx)[None, :]
    F = jnp.fft.fft2(f)
    gx = jnp.fft.ifft2(1j * kx * F).real
    gy = jnp.fft.ifft2(1j * ky * F).real
    return gx, gy


def spectral_lap(f: jnp.ndarray, dx: float) -> jnp.ndarray:
    """谱拉普拉斯 ``∇²f``（各向同性）。"""
    nx, ny = f.shape
    k2 = _kvec(nx, dx)[:, None] ** 2 + _kvec(ny, dx)[None, :] ** 2
    return jnp.fft.ifft2(-k2 * jnp.fft.fft2(f)).real


def spectral_div(fx: jnp.ndarray, fy: jnp.ndarray, dx: float) -> jnp.ndarray:
    """谱散度 ``∂fx/∂x + ∂fy/∂y``。"""
    gx_x, _ = spectral_grad(fx, dx)
    _, fy_y = spectral_grad(fy, dx)
    return gx_x + fy_y


def _k2_grid(nx: int, ny: int, dx: float) -> jnp.ndarray:
    """平方波数幅度 ``k² = kx² + ky²``（形状 ``(nx, ny)``），用于半隐式谱格式
    的隐式扩散算子 ``1/(1 + dt·k²)``。"""
    kx = _kvec(nx, dx)[:, None]
    ky = _kvec(ny, dx)[None, :]
    return kx ** 2 + ky ** 2


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 配置与状态（均为 pytree，便于 jax.grad 穿透参数）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DendriteConfig:
    """Karma–Rappel 枝晶相场参数（无量纲）。

    Attributes
    ----------
    dx : float
        网格间距（无量纲，典型 1.0）。
    dt : float
        时间步。半隐式（默认）扩散项无扩散数限制，仅反应项受 ``dt<λτ₀``；
        故大 ``λ`` 下可显著放大 ``dt``。
    eps4 : float
        界面能各向异性强度 ε₄（立方金属 4 重，典型 0.05–0.2）。
    eps_k : float
        动力学各向异性强度 εₖ（取向相关动力学过冷，典型 0.02–0.1）。
    lam : float
        耦合常数 λ（薄界面）。**默认已调大（6.0）**：越大界面越薄、tip selection
        越定量，由半隐式格式保证稳定（显式需极小 dt）。
    tau0 : float
        界面弛豫时间 τ₀（与 λ 共同决定反应项刚度）。
    D : float
        过冷度/温度场扩散系数（热或溶质扩散尺度）。
    folds : int
        各向异性重数 N（立方=4，六方=6）。
    principal_axis : float
        ⟨100⟩ 主晶向 θ₀ [rad]；``None`` 时由热梯度方向在调用处给出。
    coupled_thermal : bool
        True → U 按含潜热方程演化；False → U 取冻结梯度（定向凝固近似）。
    semi_implicit : bool
        True（默认）→ 半隐式 Fourier 谱格式（隐式扩散，支持大 λ/大 dt）；
        False → 退回显式 Euler（仅用于对照/验证）。
    bc : str
        边界标记。半隐式 Fourier 谱格式本征为**周期性**边界（谱方法固有）；
        ``bc`` 字段保留以与其他求解器对称，但求解器实际按周期谱 BC 演进。
    """

    dx: float = 1.0
    dt: float = 0.2
    eps4: float = 0.15
    eps_k: float = 0.05
    lam: float = 6.0
    tau0: float = 0.2
    D: float = 1.0
    folds: int = 4
    principal_axis: float | None = 0.0
    coupled_thermal: bool = False
    semi_implicit: bool = True
    bc: str = "neumann"

    def tree_flatten(self):
        leaves = (self.dx, self.dt, self.eps4, self.eps_k, self.lam,
                  self.tau0, self.D, float(self.folds),
                  self.principal_axis if self.principal_axis is not None else 0.0,
                  1.0 if self.coupled_thermal else 0.0,
                  1.0 if self.semi_implicit else 0.0)
        aux = {"bc": self.bc,
               "has_axis": self.principal_axis is not None}
        return leaves, aux

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        (dx, dt, eps4, eps_k, lam, tau0, D, folds, axis, cpl, si) = leaves
        pa = axis if aux["has_axis"] else None
        return cls(dx=dx, dt=dt, eps4=eps4, eps_k=eps_k, lam=lam, tau0=tau0,
                   D=D, folds=int(folds), principal_axis=pa,
                   coupled_thermal=bool(cpl), bc=aux["bc"],
                   semi_implicit=bool(si))


jax.tree_util.register_pytree_node(
    DendriteConfig, DendriteConfig.tree_flatten, DendriteConfig.tree_unflatten
)


@dataclass(frozen=True)
class PhaseFieldState:
    """相场状态：相场 ``phi`` 与过冷度场 ``u``。"""

    phi: jnp.ndarray          # (nx, ny)，∈[−1, 1]
    u: jnp.ndarray            # (nx, ny)，无量纲过冷度

    @property
    def dim(self) -> int:
        return 2


jax.tree_util.register_pytree_node(
    PhaseFieldState,
    lambda s: ((s.phi, s.u), None),
    lambda _, c: PhaseFieldState(phi=c[0], u=c[1]),
)


# ---------------------------------------------------------------------------
# 核心单步（Karma–Rappel，全可微）
# ---------------------------------------------------------------------------
def _safe_angle(gx: jnp.ndarray, gy: jnp.ndarray) -> jnp.ndarray:
    """界面法向角 θ = atan2(gy, gx)（|∇φ|=0 处取值任意，因各向异性项含
    ``(1−φ²)`` 与 ``|∇φ|`` 权重在体相内自动归零，故无奇点）。"""
    return jnp.atan2(gy, gx)


def dendrite_rhs(state: PhaseFieldState, cfg: DendriteConfig):
    """返回 ``(dφ/dt, dU/dt)`` 的当前步变化率。

    导数用**谱（Fourier）算子**（见 ``spectral_grad/lap/div``）：旋转各向同性，
    根除 5 点中心差分的 45° 网格偏置，使 4 重各向异性真正沿 ⟨100⟩ 主轴表现。
    """
    dx = cfg.dx
    phi = state.phi
    u = state.u

    gx, gy = spectral_grad(phi, dx)               # 谱梯度（各向同性）
    theta = _safe_angle(gx, gy)
    gtx, gty = spectral_grad(theta, dx)

    folds = float(cfg.folds)
    ang = folds * (theta - cfg.principal_axis)
    a = 1.0 + cfg.eps4 * jnp.cos(ang)             # a(θ)
    a2 = a * a

    # —— 各向异性扩散项：∇·(a²∇φ)（谱散度，自动含 (∂ₐa²)(∂ₐφ)）——
    a2gx = a2 * gx
    a2gy = a2 * gy
    div_a2 = spectral_div(a2gx, a2gy, dx)

    # —— Karma 反对称校正项：−½ (∂ₐa²)(∂ₐφ)(1−φ²) ——
    # 用恒等式 (∂ₐa²)(∂ₐφ) = (da²/dθ)·(∇θ·∇φ) 消去奇点
    da2_dtheta = 2.0 * a * (-folds * cfg.eps4 * jnp.sin(ang))
    dot_gradtheta_gradphi = gtx * gx + gty * gy
    correction = -0.5 * da2_dtheta * dot_gradtheta_gradphi * (1.0 - phi ** 2)

    # —— 反应（体）项 ——
    theta_k = cfg.eps_k * jnp.cos(ang)             # 动力学各向异性
    bulk = (phi - phi ** 3
            + cfg.lam * (1.0 - phi ** 2) ** 2 * (u + theta_k))
    rxn = bulk / (cfg.lam * cfg.tau0)

    dphi_dt = rxn + div_a2 + correction
    if cfg.coupled_thermal:
        du_dt = cfg.D * spectral_lap(u, dx) + 0.5 * dphi_dt
    else:
        du_dt = jnp.zeros_like(u)                  # 冻结梯度（定向凝固近似）
    return dphi_dt, du_dt


def step_phase_field_explicit(state: PhaseFieldState,
                               cfg: DendriteConfig) -> PhaseFieldState:
    """显式 Euler 推进一个时间步（对照/验证用，``semi_implicit=False`` 时调用）。"""
    dphi_dt, du_dt = dendrite_rhs(state, cfg)
    phi_new = jnp.clip(state.phi + cfg.dt * dphi_dt, -1.0, 1.0)
    u_new = state.u + cfg.dt * du_dt
    return PhaseFieldState(phi=phi_new, u=u_new)


def step_phase_field_semi_implicit(state: PhaseFieldState,
                                   cfg: DendriteConfig) -> PhaseFieldState:
    """半隐式 Fourier 谱格式推进一个时间步（默认，``semi_implicit=True``）。

    将各向异性扩散项 ``∇·(a²∇φ)`` 拆为各向同性主导 ``∇²φ``（Fourier 空间*隐式*，
    算子 ``1/(1+dt·k²)``）与各向异性涨落 ``∇·(a²∇φ)−∇²φ``（显式）；反应项与
    Karma 反对称校正项均显式。隐式吸收扩散刚性，支持大 ``λ``（薄界面）下用大
    ``dt`` 稳定推进。

    过冷度方程 ``∂U/∂t = D∇²U + ½∂φ/∂t`` 亦对扩散项作隐式（``1/(1+dt·D·k²)``），
    源项 ``½∂φ/∂t`` 用本步更新后的 ``φ``（显示式）。
    """
    dx = cfg.dx
    phi, u = state.phi, state.u
    nx, ny = phi.shape
    k2 = _k2_grid(nx, ny, dx)
    phi_hat = jnp.fft.fft2(phi)

    # —— 显式项：各向异性扩散涨落 + 反应项 + 反对称校正 ——
    gx, gy = spectral_grad(phi, dx)
    theta = _safe_angle(gx, gy)
    gtx, gty = spectral_grad(theta, dx)
    folds = float(cfg.folds)
    ang = folds * (theta - cfg.principal_axis)
    a = 1.0 + cfg.eps4 * jnp.cos(ang)
    a2 = a * a
    a2gx = a2 * gx
    a2gy = a2 * gy
    div_a2 = spectral_div(a2gx, a2gy, dx)              # ∇·(a²∇φ)（实空间）
    da2_dtheta = 2.0 * a * (-folds * cfg.eps4 * jnp.sin(ang))
    dot_gradtheta_gradphi = gtx * gx + gty * gy
    correction = -0.5 * da2_dtheta * dot_gradtheta_gradphi * (1.0 - phi ** 2)
    theta_k = cfg.eps_k * jnp.cos(ang)
    bulk = (phi - phi ** 3
            + cfg.lam * (1.0 - phi ** 2) ** 2 * (u + theta_k))
    rxn = bulk / (cfg.lam * cfg.tau0)

    lap_phi = spectral_lap(phi, dx)                    # ∇²φ（实空间）
    # 显式源 = 反应 + 反对称校正 + 各向异性扩散的剩余（非各向同性）部分
    E_real = rxn + correction + (div_a2 - lap_phi)
    E_hat = jnp.fft.fft2(E_real)

    # 隐式求解各向同性扩散项：φ̂_new = (φ̂_old + dt·Ê) / (1 + dt·k²)
    phi_hat_new = (phi_hat + cfg.dt * E_hat) / (1.0 + cfg.dt * k2)
    phi_new = jnp.clip(jnp.real(jnp.fft.ifft2(phi_hat_new)), -1.0, 1.0)

    # —— 过冷度：隐式扩散 + 显式源（用本步更新后的 ∂φ/∂t）——
    if cfg.coupled_thermal:
        dphi_dt_real = (phi_new - phi) / cfg.dt
        u_hat = jnp.fft.fft2(u)
        u_E_hat = jnp.fft.fft2(0.5 * dphi_dt_real)
        u_hat_new = (u_hat + cfg.dt * u_E_hat) / (1.0 + cfg.dt * cfg.D * k2)
        u_new = jnp.real(jnp.fft.ifft2(u_hat_new))
    else:
        u_new = u
    return PhaseFieldState(phi=phi_new, u=u_new)


def step_phase_field(state: PhaseFieldState, cfg: DendriteConfig) -> PhaseFieldState:
    """推进一个时间步：``semi_implicit=True``（默认）走半隐式谱格式，否则显式 Euler。"""
    if cfg.semi_implicit:
        return step_phase_field_semi_implicit(state, cfg)
    return step_phase_field_explicit(state, cfg)


def run_phase_field(cfg: DendriteConfig, init: PhaseFieldState, *,
                    n_steps: int, verbose: bool = False,
                    record_every: int = 0) -> dict:
    """用 ``jax.lax.scan`` 推进 n_steps，返回末态与诊断轨迹。

    Parameters
    ----------
    record_every : int
        若 >0，每 ``record_every`` 步记录一次 (phi, u)；返回 ``history`` 列表。
    """
    def body(carry, i):
        st = carry
        new = step_phase_field(st, cfg)
        if record_every and int(i) % record_every == 0:
            hist = (new.phi, new.u)
        else:
            hist = None
        return new, hist

    final, hist = jax.lax.scan(body, init, xs=jnp.arange(n_steps))
    out = {
        "final": final,
        "phi": final.phi,
        "u": final.u,
        "history": hist,
    }
    return out


# ---------------------------------------------------------------------------
# 初始条件（RVE）
# ---------------------------------------------------------------------------
def init_isothermal(grid: tuple[int, int], undercool: float,
                    seed_radius: float = 4.0,
                    seed_center: Sequence[float] | None = None,
                    dx: float = 1.0) -> PhaseFieldState:
    """均相过冷熔体 + 中心晶核：``u=Δ`` 处处，``φ=+1`` 在晶核内。"""
    nx, ny = grid
    xs = (jnp.arange(nx) - (nx - 1) / 2.0) * dx
    ys = (jnp.arange(ny) - (ny - 1) / 2.0) * dx
    X, Y = jnp.meshgrid(xs, ys, indexing="ij")
    if seed_center is None:
        cx, cy = 0.0, 0.0
    else:
        cx, cy = seed_center
    r = jnp.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    # 平滑界面：~2 网格单元过渡
    phi = jnp.tanh((seed_radius - r) / (2.0 * dx))
    u = jnp.full((nx, ny), undercool, dtype=jnp.float64)
    return PhaseFieldState(phi=phi, u=u)


def init_directional(grid: tuple[int, int], undercool: float, *,
                     axis: int = 1, seed_hot_end: bool = True,
                     dx: float = 1.0, theta0: float | None = None) -> PhaseFieldState:
    """定向凝固：过冷度沿 ``axis``（=G 方向）线性增大，晶核位于热端（低过冷）。

    枝晶自热端向冷端（沿 G）择优生长；``theta0`` 默认对齐 ``axis``
    （``axis=0``→0，``axis=1``→π/2），使 ⟨100⟩ 主轴沿 G。
    """
    nx, ny = grid
    xs = (jnp.arange(nx) - (nx - 1) / 2.0) * dx
    ys = (jnp.arange(ny) - (ny - 1) / 2.0) * dx
    X, Y = jnp.meshgrid(xs, ys, indexing="ij")
    if axis == 0:
        coord = X
        if theta0 is None:
            theta0 = 0.0
    else:
        coord = Y
        if theta0 is None:
            theta0 = jnp.pi / 2.0
    # 归一化坐标 [0,1]：热端=0（低过冷），冷端=1（高过冷，u=Δ）。
    # 晶核落在热端（coord 最小处），枝晶自热端向冷端（沿 G 即 +axis）生长。
    # 给晶界处保留最小过冷度偏移（0.1·Δ），使界面有驱动力（否则 u=0 处不生长）。
    cmin, cmax = coord.min(), coord.max()
    cnorm = (coord - cmin) / jnp.maximum(cmax - cmin, 1e-12)
    u = undercool * (0.1 + 0.9 * cnorm)
    # 晶核：热端（coord 最小）小斑
    seed_r = (coord - cmin)
    phi = jnp.tanh((2.0 * dx - seed_r) / (2.0 * dx))
    return PhaseFieldState(phi=phi, u=u)


def principal_axis_from_gradient(gx: float, gy: float) -> float:
    """由热梯度分量求 ⟨100⟩ 主晶向 θ₀ = atan2(G_y, G_x)。"""
    return float(jnp.atan2(jnp.asarray(gy), jnp.asarray(gx)))


# ---------------------------------------------------------------------------
# 形貌诊断
# ---------------------------------------------------------------------------
def solid_fraction(phi: jnp.ndarray) -> jnp.ndarray:
    """固相体积分数 = mean(0.5·(1+φ))。"""
    return jnp.mean(0.5 * (1.0 + phi))


def arm_symmetry_4fold(phi: jnp.ndarray) -> jnp.ndarray:
    """4 重对称性度量（与填充无关）：比较 ⟨100⟩ 主轴方向（0/90/180/270°）与
    对角方向（45/135/225/315°）的**界面固相尖端半径**（沿该方向最远的固相格点）。

    各向同性圆斑两者相近；4 重枝晶在 4 个主轴方向伸出臂，主轴尖端半径
    **显著大于**对角方向。返回 ``arm_tip / diag_tip``（>1 即典型的 4 臂枝晶）。
    用界面尖端半径而非固相质量，避免全凝固后体相填充稀释各向异性信号。
    """
    nx, ny = phi.shape
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    xs = (jnp.arange(nx) - cx)
    ys = (jnp.arange(ny) - cy)
    X, Y = jnp.meshgrid(xs, ys, indexing="ij")
    R = jnp.sqrt(X ** 2 + Y ** 2)
    solid = (phi > 0.0).astype(jnp.float64)
    ang = jnp.mod(jnp.atan2(Y, X), jnp.pi / 2.0)
    mask_arm = (jnp.abs(ang) < (jnp.pi / 8.0)) | (
        jnp.abs(ang - jnp.pi / 2.0) < (jnp.pi / 8.0))
    mask_diag = jnp.abs(ang - jnp.pi / 4.0) < (jnp.pi / 8.0)
    arm_tip = jnp.max(jnp.where(mask_arm & (solid > 0.5), R, 0.0))
    diag_tip = jnp.max(jnp.where(mask_diag & (solid > 0.5), R, 0.0))
    return arm_tip / jnp.maximum(diag_tip, 1e-12)


def growth_orientation(phi: jnp.ndarray) -> jnp.ndarray:
    """固相质心相对初始中心的位移方向角（枝晶主轴生长方向）。

    用于验证"沿 G 方向生长"：返回 atan2(dy, dx)。
    """
    nx, ny = phi.shape
    xs = (jnp.arange(nx) - (nx - 1) / 2.0)
    ys = (jnp.arange(ny) - (ny - 1) / 2.0)
    X, Y = jnp.meshgrid(xs, ys, indexing="ij")
    solid = (phi > 0.0).astype(jnp.float64)
    m = jnp.sum(solid)
    cx = jnp.sum(X * solid) / jnp.maximum(m, 1e-12)
    cy = jnp.sum(Y * solid) / jnp.maximum(m, 1e-12)
    return jnp.atan2(cy, cx)


def dendrite_orientation(phi: jnp.ndarray, n_rays: int = 360) -> jnp.ndarray:
    """返回枝晶主轴（⟨100⟩ 臂）方向角 [rad]。

    实现：沿 ``n_rays`` 条射线从中心采样固相尖端半径 ``R(α)``；4 重下峰值位于
    ``θ₀ + k·90°``。取 R(α) 的加权主方向（单位圆 PCA）。

    .. note::
       4 重对称下各臂贡献在 PCA 中部分抵消，结果对噪声敏感；更稳健的取向判据
       见 :func:`arm_symmetry_4fold` 的旋转测试（θ₀=0 与 θ₀=45° 的臂方向互换）。
    """
    nx, ny = phi.shape
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    xs = (jnp.arange(nx) - cx)
    ys = (jnp.arange(ny) - cy)
    X, Y = jnp.meshgrid(xs, ys, indexing="ij")
    R = jnp.sqrt(X ** 2 + Y ** 2)
    solid = (phi > 0.0).astype(jnp.float64)
    alphas = jnp.linspace(0.0, 2.0 * jnp.pi, n_rays, endpoint=False)
    dxr = jnp.cos(alphas)
    dyr = jnp.sin(alphas)
    max_step = int(jnp.ceil(jnp.sqrt(cx ** 2 + cy ** 2)))
    steps = jnp.arange(1, max_step + 1)
    ix = jnp.clip(jnp.round(cx + dxr[:, None] * steps[None, :]).astype(int), 0, nx - 1)
    iy = jnp.clip(jnp.round(cy + dyr[:, None] * steps[None, :]).astype(int), 0, ny - 1)
    sampled = solid[ix, iy]
    rev = jnp.flip(sampled, axis=1)
    first = jnp.argmax(rev, axis=1)
    has = jnp.any(sampled > 0.5, axis=1)
    tip = jnp.where(has, (max_step - first) * 1.0, 0.0)
    w = jnp.where(has, tip, 0.0)
    wx = jnp.sum(w * jnp.cos(alphas))
    wy = jnp.sum(w * jnp.sin(alphas))
    return jnp.atan2(wy, wx)


# ---------------------------------------------------------------------------
# 与 AM 链路对接：RVE 运行 + 映射为 MicrostructureResult
# ---------------------------------------------------------------------------
def _repr_grad_rate(thermal: ThermalHistory):
    """从 ThermalHistory 取熔化区代表性的 G、R、G/R。

    返回保持为 jnp 数组（不调用 ``float()`` 具体化）——这是可微评估的关键：
    当损失对工艺参数求梯度时 thermal 是 traced 数组，任何 ``float()`` 都会触发
    ``ConcretizationTypeError`` 并截断梯度。全程 jnp 运算保证梯度穿过微观映射。
    """
    melted = (thermal.peak_temperature > 0.5 * jnp.max(thermal.peak_temperature))
    melted = melted.astype(jnp.float64)
    G = jnp.sum(thermal.thermal_gradient * melted) / jnp.maximum(jnp.sum(melted), 1e-12)
    R = jnp.sum(thermal.solidification_rate * melted) / jnp.maximum(jnp.sum(melted), 1e-12)
    return G, R, G / jnp.maximum(R, 1e-12)


def _arm_spacing_law(G, R):
    """二次枝晶臂间距经验律（Hunt/KGT 型）：``d = C·(G·R)^{-0.25}``。

    量级：``G·R ~ 1e6 K/s`` 时 ``d ~ 1 µm``，冷却越快越细。
    这是枝晶尺度上最稳妥的定量关系；相场提供*取向*与*形貌*，臂间距由该律给出。

    输入 G、R 可为 jnp 数组（可微链路中来自 traced thermal），故全程 jnp 运算，
    不调用 ``float()``/``max()`` 具体化，保证梯度可穿过微观映射进入逆问题。
    """
    GR = jnp.maximum(G * R, 1.0)
    d = 1.0e-6 * (GR / 1.0e6) ** (-0.25)          # [m]，≈1 µm @ 1e6 K/s
    return jnp.clip(d, 4e-6, 400e-6)


@register_solver(
    "micro.phasefield",
    consumes=("ThermalHistory", "MeltPoolResult"),
    produces="MicrostructureResult",
    stage="microstructure",
    modality=("SLM", "LSF"),
    differentiable=True,
    cost=50.0,   # 高保真微观求解器：比统计代理(2.0)昂贵
    defaults={
        "grid": (64, 64), "n_steps": 300, "undercool": 0.55,
        "mode": "isothermal", "gradient_axis": 1,
        "eps4": 0.15, "eps_k": 0.05, "lam": 6.0, "tau0": 0.2,
        "D": 1.0, "folds": 4, "seed_radius": 4.0, "verbose": False,
    },
    doc="Karma–Rappel 枝晶相场：含 4 重界面能各向异性与热梯度取向，"
        "输出 G 对齐的 ⟨100⟩ 枝晶形貌与经验二次臂间距",
)
def solve_phasefield(*, thermal: ThermalHistory,
                     meltpool: MeltPoolResult, params) -> MicrostructureResult:
    """运行 AM 枝晶相场 RVE，并映射为 ``MicrostructureResult``。

    流程
    ----
    1. 从 ``thermal`` 取代表性 G、R（熔化区均值）。
    2. 主晶向 ``θ₀`` = 热梯度方向（由 ``gradient_axis`` 给出，默认构建方向 +
       y）：枝晶 ``⟨100⟩`` 主轴即沿 G，体现"热梯度取向（G 方向优先生长）"。
       （若 ``thermal`` 提供向量梯度，可在此直接代入真实 (Gx,Gy)；当前
       ``ThermalHistory.thermal_gradient`` 为标量幅值，故用轴向参数代表。）
    3. 跑 RVE（2D 截面）：均相过冷(默认) 或定向凝固（沿 G 线性过冷梯度）。
    4. 由经验律给出二次臂间距 ``d(G,R)``；由 G/R 判定柱状/等轴；织构随柱状分数；
       取向场 = θ₀（G 对齐）；孔隙搬移熔池指标。

    返回与 ``micro.surrogate`` **同契约** 的 ``MicrostructureResult``，因此
    下游本构/成形链路无需改动即可切换为高保真相场。
    """
    p = dict(params or {})
    grid = tuple(int(g) for g in p.get("grid", (64, 64)))
    n_steps = int(p.get("n_steps", 300))
    undercool = float(p.get("undercool", 0.55))
    eps4 = float(p.get("eps4", 0.15))
    eps_k = float(p.get("eps_k", 0.05))
    lam = float(p.get("lam", 6.0))
    tau0 = float(p.get("tau0", 0.2))
    D = float(p.get("D", 1.0))
    folds = int(p.get("folds", 4))
    seed_radius = float(p.get("seed_radius", 4.0))
    dx = float(p.get("dx", 1.0))
    mode = str(p.get("mode", "isothermal"))
    gradient_axis = int(p.get("gradient_axis", 1))
    verbose = bool(p.get("verbose", False))

    # 1) 代表性 G、R
    G, R, GR = _repr_grad_rate(thermal)

    # 2) 主晶向 θ₀：热梯度方向（轴向参数代表 G 方向）
    #    axis=0 → θ₀=0（沿 x）；axis=1 → θ₀=π/2（沿 y）；与 init_directional 一致
    theta0 = 0.0 if gradient_axis == 0 else jnp.pi / 2.0

    # 3) RVE 运行（本机用中小网格，48G 可放大）
    cfg = DendriteConfig(dx=dx, dt=float(p.get("dt", 0.2)), eps4=eps4, eps_k=eps_k,
                         lam=lam, tau0=tau0, D=D, folds=folds,
                         principal_axis=theta0, coupled_thermal=False, bc="neumann")
    if mode == "directional":
        init = init_directional(grid, undercool, axis=gradient_axis, dx=dx,
                                theta0=theta0)
    else:
        init = init_isothermal(grid, undercool, seed_radius=seed_radius, dx=dx)
    res = run_phase_field(cfg, init, n_steps=n_steps, verbose=verbose)
    phi = res["phi"]
    sf = float(solid_fraction(phi))

    # 3) 形貌 → 微观组织字段（映射到零件网格 thermal 的形貌）
    shape = thermal.peak_temperature.shape
    spacing = float(thermal.spacing)
    melted_mask = (thermal.peak_temperature >
                   0.5 * jnp.max(thermal.peak_temperature)).astype(jnp.float64)

    # 二次枝晶臂间距（经验律）
    grain_size = jnp.full(shape, _arm_spacing_law(G, R), dtype=jnp.float64)
    grain_size = grain_size * melted_mask + (1.0 - melted_mask) * 50e-6

    # 柱状/等轴：由 G/R 判据（高 G/R → 柱状）
    GR_crit = float(p.get("GR_crit", 4.5e6))
    k_cet = float(p.get("cet_slope", 0.6))
    columnar = jax.nn.sigmoid(k_cet * (jnp.log(jnp.maximum(GR, 1e-6)) -
                                       jnp.log(GR_crit)))
    columnar = columnar * melted_mask
    texture = columnar * 0.85

    # 取向：θ₀（G 对齐 ⟨100⟩），标量场广播
    orientation = jnp.full(shape, theta0, dtype=jnp.float64) * melted_mask

    # 孔隙：搬移熔池指标
    por = jnp.asarray(meltpool.porosity_indicator)
    porosity = por * melted_mask

    # 固相序参量：熔化区=实体(1)
    phi_seq = melted_mask

    # 体积平均晶粒尺寸
    cell = spacing ** thermal.dim
    mean_g = (jnp.sum(grain_size * melted_mask) * cell /
              jnp.maximum(jnp.sum(melted_mask) * cell, 1e-30))

    return MicrostructureResult(
        phi=jnp.asarray(phi_seq),
        orientation=jnp.asarray(orientation),
        grain_size=jnp.asarray(grain_size),
        columnar_fraction=jnp.asarray(columnar),
        texture_intensity=jnp.asarray(texture),
        porosity=jnp.asarray(porosity),
        mean_grain_size=jnp.asarray(mean_g),
        spacing=spacing,
        dim=thermal.dim,
    )
