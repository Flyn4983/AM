"""AM 粉末床颗粒尺度求解（DEM 铺粉 / SPH 熔道 / MPM 剥蚀-飞溅）。

为什么需要这一层
----------------
连续介质熔池模型（``MeltPoolResult``）把粉床当成**均质连续金属**，因此
在原理上**看不到颗粒本身**引起的三类缺陷：

* **球化 (balling)** —— 熔道在 Plateau–Rayleigh 不稳定性下断裂成球串；
* **未熔合 (lack of fusion)** —— 铺粉密度不足 / 粉层相对粒径过厚；
* **飞溅 (spatter) 与剥蚀 (denudation)** —— 蒸气反冲与气流卷吸抛出颗粒。

这三者只能由颗粒尺度方法给出：DEM（铺粉重排）、SPH（熔道自由界面）、
MPM（粉末再分布）。本模块把 ``diffmech`` 已有的三个**可微**颗粒内核
接到 AM 语境上，输出可微缺陷指标，供逆问题的 defect 罚项使用。

复用而非重写
------------
时间步进直接复用现成内核，本模块**不重复实现任何接触/流体/连续体物理**：

============  ==========================================  ================
物理过程      复用的步进器                                本模块新增
============  ==========================================  ================
铺粉重排      :func:`diffmech.methods.dem.step_dem_scan`  粉床生成 + 密实度/配位数/粗糙度度量
熔道自由界面  :func:`...particle_am.step_sph_am_scan`      Plateau–Rayleigh 球化判据
粉末再分布    :func:`...particle_am.step_mpm_am_scan`      剥蚀半宽 + 飞溅质量分数
============  ==========================================  ================

tracer 安全性（接入 ForgeCore 链路的硬约束）
--------------------------------------------
ForgeCore 逆问题会对 ``PartGeometry.sdf`` 与 ``ProcessPlan`` 求导，因此本
模块严格遵守：

1. **不读 ``geometry.sdf`` 的具体值**。物理上这也更正确 —— 粉床铺满整个
   成形区，而不是只铺在零件轮廓内；颗粒布点只由**静态** bbox 元数据决定。
2. **数组形状全静态**，数值可为 tracer：晶格用 ``jnp.arange(n_static)``
   生成后再乘可微间距，因此 ``layer_thickness`` 可微而形状不变。
3. **扫描轨迹的 waypoints 必须是具体 numpy**（``ScanPath`` 落在 pytree 的
   aux 侧）。故**轨迹几何静态、能量与时序可微**：``laser_power`` /
   ``scan_speed`` / ``absorption`` / ``beam_radius`` 经 ``SLMConfig``
   （已注册 pytree）进入热源，梯度照常回传。
4. **接触模型参数取具体 float**：``DEMConfig`` 内部有
   ``if cfg.mu > 0.0 and cfg.kt > 0.0`` 这类 **Python 层值分支**，传入
   tracer 会抛 ``TracerBoolConversionError``。接触刚度/阻尼本就是材料侧
   常数而非优化变量，故取具体值 —— 这是有意的设计而非妥协。

单位一律 SI（m, s, K, kg）。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from diffmech.methods.am.process import SLMConfig
from diffmech.methods.am.scan_paths import from_waypoints, multi_layer_paths
from diffmech.methods.dem import DEMConfig, DEMState, Wall, step_dem_scan

Array = jnp.ndarray

__all__ = [
    "PowderBedConfig",
    "synth_powder_bed",
    "recoater_walls",
    "recoat_dem",
    "melt_track_sph",
    "denudation_mpm",
    "single_track_path",
    "slm_config_from_plan",
    "analytic_powder_defects",
]


# ===========================================================================
# 配置
# ===========================================================================
@dataclass(frozen=True)
class PowderBedConfig:
    """粉床几何 / 颗粒参数。

    可微叶子（可作为优化变量或由 ``ProcessPlan`` 驱动）
        ``layer_thickness``、``bed_lx``、``bed_ly``。
    具体静态量（不可为 tracer）
        ``d50``、``psd_sigma``、``rho_powder``：粉末批次属性，且
        ``d50`` 要参与 DEM 接触刚度标定（见模块 docstring 第 4 条）。
        ``nx``/``ny``/``nz``/``dim``/``seed``：形状与随机种子。

    Attributes
    ----------
    layer_thickness : 标量 [m]
        名义粉层厚度（刮刀高度）。SLM 典型 20~60 µm。
    bed_lx, bed_ly : 标量 [m]
        粉床横向尺寸（取零件 bbox 的一个代表性窗口，而非整个成形缸）。
    d50 : float [m]
        粒径分布中位数。SLM 常用 15~45 µm 粉。
    psd_sigma : float [-]
        对数正态 PSD 的对数标准差（0 = 单分散）。
    rho_powder : float [kg/m³]
        颗粒材料（非表观）密度。
    nx, ny, nz : int
        初始晶格的颗粒数（静态；总数 N = nx·ny·nz）。
    dim : int
        2 或 3。
    seed : int
        PSD / 抖动的确定性随机种子。
    initial_positions : (N, dim) array or None
        可选：外部注入的初始粒子坐标（opt-in）。给定时 ``synth_powder_bed``
        直接以这些精确坐标作为 DEM 沉降初态（跳过抖动晶格采样）；未给定则
        按 nx/ny/nz + 对数正态 PSD 生成 RCP 初态（向后兼容）。
    initial_radii : (N,) array or None
        注入粒子的半径，与 ``initial_positions`` 配套。仅给坐标未给半径时，
        半径按原生 PSD 派生。坐标为具体数组（用户导入/生成的粒子），存于
        静态 aux 侧，不参与 tracer 求导。
    """

    layer_thickness: Array
    bed_lx: Array
    bed_ly: Array
    d50: float = 30e-6
    psd_sigma: float = 0.25
    rho_powder: float = 7950.0
    nx: int = 6
    ny: int = 6
    nz: int = 3
    dim: int = 3
    seed: int = 0
    # 可选：外部注入的初始粒子态（opt-in）。给定时 synth_powder_bed 直接以这些
    # 精确坐标/半径作为 DEM 沉降初态（跳过抖动晶格采样）；未给定则按 nx/ny/nz +
    # 对数正态 PSD 生成 RCP 初态（向后兼容）。坐标为具体数组（用户导入/生成的
    # 粒子），存于静态 aux 侧，不参与 tracer 求导。
    initial_positions: Any | None = None
    initial_radii: Any | None = None

    @property
    def n_particles(self) -> int:
        return int(self.nx * self.ny * (self.nz if self.dim == 3 else 1))


def _pbc_flatten(c: PowderBedConfig):
    return ((c.layer_thickness, c.bed_lx, c.bed_ly),
            (c.d50, c.psd_sigma, c.rho_powder,
             c.nx, c.ny, c.nz, c.dim, c.seed,
             c.initial_positions, c.initial_radii))


def _pbc_unflatten(aux, leaves):
    lt, lx, ly = leaves
    (d50, psd, rho, nx, ny, nz, dim, seed, init_pos, init_rad) = aux
    return PowderBedConfig(layer_thickness=lt, bed_lx=lx, bed_ly=ly,
                           d50=d50, psd_sigma=psd, rho_powder=rho,
                           nx=nx, ny=ny, nz=nz, dim=dim, seed=seed,
                           initial_positions=init_pos, initial_radii=init_rad)


jax.tree_util.register_pytree_node(PowderBedConfig, _pbc_flatten, _pbc_unflatten)


# ===========================================================================
# 1. 粉床生成（tracer-safe）
# ===========================================================================
def synth_powder_bed(cfg: PowderBedConfig) -> tuple[DEMState, Array]:
    """生成初始粉床：抖动晶格 + 对数正态粒径分布。

    物理动机：真实铺粉后的粉床既不是规则堆积也不是完全随机，而是**近似
    随机密堆 (RCP)**。这里用"规则晶格 + 亚粒径量级抖动 + 对数正态 PSD"
    构造初态，再由 DEM 在重力与刮刀约束下自行沉降到力学平衡 —— **密实度
    是解出来的，不是假设的**，这是本模块相对解析代理的核心增益。

    颗粒初始被抬升到 ``1.6 × layer_thickness``（刮刀高度之上），使沉降过程
    真实发生；否则初态即终态，DEM 退化为无意义的空转。

    Returns
    -------
    state : DEMState
        位置 / 速度 / 半径 / 质量。位置数值可为 tracer，形状恒为
        ``(N, dim)``（``N = cfg.n_particles``）。
    column_id : (N,) int
        每个颗粒的**静态**横向列索引 ∈ [0, nx·ny)，供粗糙度度量做
        拉格朗日列跟踪（见 :func:`recoat_dem`）。
    """
    dim = int(cfg.dim)
    nx, ny, nz = int(cfg.nx), int(cfg.ny), int(cfg.nz)
    if dim == 2:
        ny = 1
    n = nx * ny * nz

    # --- 外部注入的精确初始粒子态（opt-in）-----------------------------
    # 给定 initial_positions 时，直接以这些精确坐标/半径作为 DEM 沉降初态，
    # 跳过抖动晶格 + 对数正态 PSD 采样（向后兼容：未给定则走下方 RCP 路径）。
    if cfg.initial_positions is not None:
        pos = jnp.asarray(cfg.initial_positions, dtype=jnp.float64)
        if pos.ndim != 2 or int(pos.shape[-1]) != dim:
            raise ValueError(
                f"initial_positions 形状应为 (N, {dim})，收到 {tuple(pos.shape)}")
        n_inj = int(pos.shape[0])
        if cfg.initial_radii is not None:
            radius = jnp.asarray(cfg.initial_radii, dtype=jnp.float64)
            if int(radius.shape[0]) != n_inj:
                raise ValueError(
                    f"initial_radii 长度 {int(radius.shape[0])} 与 "
                    f"initial_positions 粒子数 {n_inj} 不匹配")
        else:
            key = jax.random.PRNGKey(int(cfg.seed))
            z = jax.random.normal(key, (n_inj,), dtype=jnp.float64)
            diameter = float(cfg.d50) * jnp.exp(float(cfg.psd_sigma) * z)
            diameter = jnp.clip(diameter, 0.3 * float(cfg.d50), 3.0 * float(cfg.d50))
            radius = 0.5 * diameter
        mass = float(cfg.rho_powder) * (jnp.pi / 6.0) * (2.0 * radius) ** 3
        # 拉格朗日列索引：按初始横向坐标分箱（与 RCP 路径列语义一致）
        dx = cfg.bed_lx / jnp.maximum(nx, 1)
        dy = cfg.bed_ly / jnp.maximum(ny, 1)
        cx = jnp.clip(jnp.floor(pos[:, 0] / dx), 0, max(nx - 1, 0)).astype(jnp.int32)
        if dim == 3:
            cy = jnp.clip(jnp.floor(pos[:, 1] / dy), 0, max(ny - 1, 0)).astype(jnp.int32)
            col = cx + ny * cy
        else:
            col = cx
        state = DEMState(position=pos, velocity=jnp.zeros_like(pos),
                         radius=radius, mass=mass)
        return state, col

    key = jax.random.PRNGKey(int(cfg.seed))
    k_r, k_j = jax.random.split(key)

    # --- 粒径：对数正态 PSD（d50 为中位数 ⇒ 对数正态的 scale 参数）------
    z = jax.random.normal(k_r, (n,), dtype=jnp.float64)
    diameter = float(cfg.d50) * jnp.exp(float(cfg.psd_sigma) * z)
    # 限幅到 [0.3, 3.0]·d50，避免长尾产生病态接触刚度
    diameter = jnp.clip(diameter, 0.3 * float(cfg.d50), 3.0 * float(cfg.d50))
    radius = 0.5 * diameter
    mass = float(cfg.rho_powder) * (jnp.pi / 6.0) * diameter ** 3

    # --- 晶格：形状静态（arange），间距可微（乘 tracer）-----------------
    ix = jnp.arange(nx, dtype=jnp.float64)
    iy = jnp.arange(ny, dtype=jnp.float64)
    iz = jnp.arange(nz, dtype=jnp.float64)
    dx = cfg.bed_lx / jnp.maximum(nx, 1)
    dy = cfg.bed_ly / jnp.maximum(ny, 1)
    # 竖向初始间距按 d50 给（保证初态无重叠），整体抬到刮刀之上
    dz = 1.05 * float(cfg.d50)
    z0 = 1.6 * cfg.layer_thickness

    if dim == 2:
        gx, gz = jnp.meshgrid(ix, iz, indexing="ij")
        pos = jnp.stack([(gx + 0.5) * dx, z0 + gz * dz], axis=-1)
        pos = pos.reshape(-1, 2)
        # meshgrid(indexing="ij") + reshape ⇒ x 为外层，故列索引按 x 重复 nz 次
        col = jnp.reshape(jnp.arange(nx)[:, None] * jnp.ones((1, nz), int), (-1,))
    else:
        gx, gy, gz = jnp.meshgrid(ix, iy, iz, indexing="ij")
        pos = jnp.stack([(gx + 0.5) * dx, (gy + 0.5) * dy, z0 + gz * dz],
                        axis=-1).reshape(-1, 3)
        col_grid = (jnp.arange(nx)[:, None, None] * ny
                    + jnp.arange(ny)[None, :, None]) * jnp.ones((1, 1, nz), int)
        col = jnp.reshape(col_grid, (-1,))

    # --- 亚粒径抖动：打破晶格对称，让 DEM 沉降到真实 RCP ---------------
    jitter = jax.random.uniform(k_j, pos.shape, dtype=jnp.float64,
                                minval=-0.35, maxval=0.35) * float(cfg.d50)
    pos = pos + jitter

    state = DEMState(position=pos, velocity=jnp.zeros_like(pos),
                     radius=radius, mass=mass)
    return state, col


def recoater_walls(cfg: PowderBedConfig) -> list[Wall]:
    """粉床边界：基板 + 刮刀（doctor blade）+ 四周侧壁。

    ``Wall`` 约定颗粒被限制在 ``normal · x <= offset`` 半空间内。

    * **基板**：``z >= 0``      → normal ``-e_z``, offset ``0``；
    * **刮刀**：``z <= t_layer`` → normal ``+e_z``, offset ``t_layer``
      —— 刮刀的物理作用正是**把粉层限制到名义厚度**并刮走多余粉，
      因此建模为一道水平硬约束比建模移动刀片更贴近其净效应，
      且避免把时间引入静态 wall 列表；
    * **侧壁**：限制在 ``[0, bed_lx] × [0, bed_ly]`` 窗口内。
    """
    dim = int(cfg.dim)
    walls: list[Wall] = []
    e = [jnp.zeros(dim).at[i].set(1.0) for i in range(dim)]
    # 基板 z>=0 与刮刀 z<=t
    walls.append(Wall(normal=-e[dim - 1], offset=0.0))
    walls.append(Wall(normal=e[dim - 1], offset=cfg.layer_thickness))
    # 侧壁
    walls.append(Wall(normal=-e[0], offset=0.0))
    walls.append(Wall(normal=e[0], offset=cfg.bed_lx))
    if dim == 3:
        walls.append(Wall(normal=-e[1], offset=0.0))
        walls.append(Wall(normal=e[1], offset=cfg.bed_ly))
    return walls


# ===========================================================================
# 2. DEM 铺粉：密实度 / 配位数 / 粗糙度
# ===========================================================================
def _smooth_max(x: Array, beta: float, axis=None) -> Array:
    """可微软最大值：``logsumexp(βx)/β``（β→∞ 时趋于 max）。"""
    return jax.scipy.special.logsumexp(beta * x, axis=axis) / beta


def _packing_density(state: DEMState, cfg: PowderBedConfig) -> Array:
    """铺粉体积分数 = Σ颗粒体积 / (床面积 × 实际堆高)。

    实际堆高用软最大值 ``max_p(z_p + r_p)`` 取，避免 ``jnp.max`` 在并列
    极值处的次梯度不连续。堆高下界钳到 ``0.5·d50``，防止空床时除零。
    """
    dim = int(cfg.dim)
    zt = state.position[:, -1] + state.radius
    beta = 1.0 / (0.05 * float(cfg.d50))
    h_bed = jnp.maximum(_smooth_max(zt, beta), 0.5 * float(cfg.d50))
    if dim == 2:
        v_p = jnp.sum(jnp.pi * state.radius ** 2)          # 面积（2D）
        v_box = cfg.bed_lx * h_bed
    else:
        v_p = jnp.sum((4.0 / 3.0) * jnp.pi * state.radius ** 3)
        v_box = cfg.bed_lx * cfg.bed_ly * h_bed
    return jnp.clip(v_p / jnp.maximum(v_box, 1e-30), 0.0, 1.0)


def _coordination_number(state: DEMState, cfg: PowderBedConfig) -> Array:
    """平均配位数：用 sigmoid 平滑"接触与否"，保持可微。

    ``overlap_ij = r_i + r_j - d_ij``；接触指示 ``σ(overlap / (0.02·d50))``。
    对角自接触被显式扣掉。密实随机堆积的理论值约 6~7。
    """
    pos, rad = state.position, state.radius
    d = jnp.sqrt(jnp.sum((pos[:, None, :] - pos[None, :, :]) ** 2, axis=-1)
                 + 1e-30)
    overlap = rad[:, None] + rad[None, :] - d
    eps = 0.02 * float(cfg.d50)
    contact = jax.nn.sigmoid(overlap / eps)
    n = pos.shape[0]
    contact = contact - jnp.eye(n) * jax.nn.sigmoid(
        (2.0 * rad) / eps)            # 扣掉 i==j 的自"接触"
    return jnp.mean(jnp.sum(contact, axis=1))


def _surface_roughness(state: DEMState, column_id: Array,
                       cfg: PowderBedConfig) -> Array:
    """铺粉表面粗糙度 Ra [m]：逐列顶面高度的标准差。

    列归属沿用**初始**横向索引（拉格朗日列跟踪）—— 铺粉沉降是小位移过程，
    颗粒基本不跨列迁移，因此该近似成立且让度量保持形状静态与可微。
    每列顶面高度用软最大值取。
    """
    zt = state.position[:, -1] + state.radius
    n_col = int(cfg.nx * (1 if cfg.dim == 2 else cfg.ny))
    beta = 1.0 / (0.05 * float(cfg.d50))
    onehot = jax.nn.one_hot(column_id, n_col)                  # (N, n_col)
    # 逐列 softmax 加权高度（等价于按列的软最大值，屏蔽非本列成员）
    logits = beta * zt[:, None] + jnp.log(onehot + 1e-30)
    w = jax.nn.softmax(logits, axis=0)                          # (N, n_col)
    h_col = jnp.sum(w * zt[:, None], axis=0)                    # (n_col,)
    return jnp.sqrt(jnp.mean((h_col - jnp.mean(h_col)) ** 2) + 1e-30)


def recoat_dem(cfg: PowderBedConfig, *,
               initial_positions=None, initial_radii=None,
               contact_stiffness: float = 2.0e3,
               contact_damping: float = 5.0e-4,
               friction: float = 0.35,
               tangential_stiffness: float = 4.0e2,
               gravity: float = -9.81,
               dt: float | None = None,
               n_steps: int = 240) -> dict[str, Array]:
    """DEM 铺粉：重力沉降 + 刮刀约束 → 密实度 / 配位数 / 粗糙度。

    时间步长默认按**接触振动周期**取：``dt = 0.1 · sqrt(m_min / k)``，
    这是线性弹簧-阻尼接触模型的常规稳定性判据（Rayleigh 判据的简化形式）。
    传入具体 ``dt`` 可覆盖。

    Notes
    -----
    接触参数取具体 float 而非 tracer —— 见模块 docstring 第 4 条。
    """
    if initial_positions is not None:
        cfg = replace(cfg, initial_positions=initial_positions,
                      initial_radii=initial_radii)
    state0, column_id = synth_powder_bed(cfg)
    dem_cfg = DEMConfig(k=float(contact_stiffness), gamma=float(contact_damping),
                        mu=float(friction), kt=float(tangential_stiffness),
                        gravity=float(gravity))
    if dt is None:
        m_min = float(cfg.rho_powder) * (jnp.pi / 6.0) * \
            (0.3 * float(cfg.d50)) ** 3
        dt = 0.1 * float(np.sqrt(m_min / float(contact_stiffness)))

    walls = recoater_walls(cfg)
    state = step_dem_scan(state0, float(dt), int(n_steps), dem_cfg, walls=walls)

    return {
        "packing_density": _packing_density(state, cfg),
        "coordination_number": _coordination_number(state, cfg),
        "surface_roughness": _surface_roughness(state, column_id, cfg),
        "final_state": state,
        "dt": jnp.asarray(dt),
    }


# ===========================================================================
# 3. 扫描轨迹与工艺配置桥（静态几何 / 可微能量）
# ===========================================================================
def single_track_path(cfg: PowderBedConfig, *, n_waypoints: int = 8,
                      layer_z: float | None = None):
    """粉床窗口中央的一条直线熔道（**静态 numpy** waypoints）。

    ``ScanPath.waypoints`` 落在 pytree 的 aux 侧，必须具体化；因此轨迹
    几何用静态标称值构造。能量与时序仍由 ``SLMConfig`` 的可微叶子驱动，
    梯度对 ``laser_power`` / ``scan_speed`` / ``beam_radius`` 照常回传。
    """
    lx = float(np.asarray(cfg.bed_lx))
    ly = float(np.asarray(cfg.bed_ly))
    zc = float(np.asarray(cfg.layer_thickness)) * 0.5 if layer_z is None \
        else float(layer_z)
    s = np.linspace(0.05 * lx, 0.95 * lx, int(n_waypoints))
    if int(cfg.dim) == 2:
        wp = np.stack([s, np.full_like(s, zc)], axis=-1)
    else:
        wp = np.stack([s, np.full_like(s, 0.5 * ly)], axis=-1)
    return multi_layer_paths([from_waypoints(wp, layer_z=zc)])


def slm_config_from_plan(*, laser_power, scan_speed, layer_thickness,
                         hatch_spacing, beam_radius, absorption,
                         preheat_temp) -> SLMConfig:
    """由 ``ProcessPlan`` 的可微叶子构造 ``SLMConfig``（已注册 pytree）。

    逐层数组取首层标量（粉床尺度求解针对单道单层），用 ``jnp.ravel(...)[0]``
    而非 ``float(...)`` —— 后者对 tracer 会抛
    ``TypeError: Only scalar arrays can be converted to Python scalars``。
    """
    f = lambda v: jnp.ravel(jnp.asarray(v))[0]  # noqa: E731
    return SLMConfig(
        laser_power=f(laser_power), scan_speed=f(scan_speed),
        layer_thickness=f(layer_thickness), hatch_spacing=f(hatch_spacing),
        beam_radius=f(beam_radius), absorption=f(absorption),
        contour_first=False, rotation_per_layer=0.0,
        preheat_temp=f(preheat_temp),
    )


# ===========================================================================
# 4. SPH 熔道：Plateau–Rayleigh 球化
# ===========================================================================
def melt_track_sph(cfg: PowderBedConfig, slm: SLMConfig, *,
                   T_melt: float = 1700.0,
                   cp: float = 500.0,
                   sigma_surface: float = 1.6,
                   mu_liquid: float = 6.0e-3,
                   n_steps: int = 60,
                   dt: float | None = None) -> dict[str, Array]:
    """SPH 熔道自由界面 → 球化指标（Plateau–Rayleigh）。

    物理判据
    --------
    一段长 ``L``、直径 ``W`` 的液柱在表面张力驱动下不稳定，当

    .. math:: L / W > \\pi

    时最快增长模态使其断裂成球串（Plateau–Rayleigh）。熔道即一段被基体
    部分约束的液柱，故取熔区的**纵向展布 L** 与**横向展布 W** 之比作为
    球化驱动力，并用 Ohnesorge 数修正黏性抑制：

    .. math:: Oh = \\mu / \\sqrt{\\rho \\sigma W}

    ``Oh`` 越大黏性越能抑制断裂。最终指标

    .. math:: B = \\sigma_{\\rm sig}\\!\\left[k\\left(\\frac{L/W}{\\pi(1+Oh)} - 1\\right)\\right]

    熔区用**软液相分数** ``σ((T - T_melt)/ΔT)`` 加权，保持可微。
    """
    from diffmech.methods.am.particle_am import (
        ParticleAMState, particle_layer_times, step_sph_am_scan,
    )
    from diffmech.methods.sph import SPHConfig, SPHState

    state0, _ = synth_powder_bed(cfg)
    n = state0.position.shape[0]
    paths = single_track_path(cfg)
    starts, durations = particle_layer_times(paths, slm)

    sph_cfg = SPHConfig(h=1.5 * float(cfg.d50), rho0=float(cfg.rho_powder),
                        c0=20.0, nu=0.05, gamma_eos=7.0, gravity=-9.81)
    sph0 = SPHState(position=state0.position,
                    velocity=jnp.zeros_like(state0.position),
                    mass=state0.mass)
    am0 = ParticleAMState(position=state0.position,
                          activation_time=jnp.zeros(n),
                          temperature=jnp.full(n, slm.preheat_temp),
                          mass=state0.mass,
                          layer_id=jnp.zeros(n, dtype=jnp.int64),
                          rho=jnp.full(n, float(cfg.rho_powder),
                                       dtype=jnp.float64))
    if dt is None:
        # 声速 CFL：dt = 0.2 h / c0
        dt = 0.2 * float(sph_cfg.h) / float(sph_cfg.c0)

    sph_f, am_f = step_sph_am_scan(sph0, am0, float(dt), int(n_steps),
                                   sph_cfg, slm, paths, starts, durations,
                                   cp=float(cp))

    # --- 软液相分数加权的熔区展布 -------------------------------------
    dT = 80.0
    fl = jax.nn.sigmoid((am_f.temperature - float(T_melt)) / dT)   # (N,)
    wsum = jnp.maximum(jnp.sum(fl), 1e-12)
    pos = sph_f.position
    # 纵向 = 扫描方向 (x)；横向 = 3D 取 y，2D 取 z
    ax_long, ax_tran = 0, (1 if int(cfg.dim) == 3 else 1)
    mu_l = jnp.sum(fl * pos[:, ax_long]) / wsum
    mu_t = jnp.sum(fl * pos[:, ax_tran]) / wsum
    # 2σ 展布（≈ 68% 质量宽度的两倍），加护垫保证零熔区时梯度有限
    L = 2.0 * jnp.sqrt(jnp.sum(fl * (pos[:, ax_long] - mu_l) ** 2) / wsum + 1e-30)
    W = 2.0 * jnp.sqrt(jnp.sum(fl * (pos[:, ax_tran] - mu_t) ** 2) / wsum + 1e-30)
    W = jnp.maximum(W, 0.5 * float(cfg.d50))

    Oh = float(mu_liquid) / jnp.sqrt(
        float(cfg.rho_powder) * float(sigma_surface) * W + 1e-30)
    ratio = (L / W) / (jnp.pi * (1.0 + Oh))
    balling = jax.nn.sigmoid(3.0 * (ratio - 1.0))
    # 完全未熔化时不应报球化：按熔化比例门控
    melt_frac = jnp.mean(fl)
    balling = balling * jax.nn.sigmoid((melt_frac - 0.02) / 0.02)

    return {
        "balling_indicator": jnp.clip(balling, 0.0, 1.0),
        "track_length": L, "track_width": W,
        "melt_fraction": melt_frac,
        "peak_temperature": jnp.max(am_f.temperature),
        "ohnesorge": Oh,
    }


# ===========================================================================
# 5. MPM 剥蚀 / 飞溅
# ===========================================================================
def denudation_mpm(cfg: PowderBedConfig, slm: SLMConfig, *,
                   cp: float = 500.0,
                   youngs_modulus: float = 5.0e7,
                   n_steps: int = 40,
                   dt: float | None = None) -> dict[str, Array]:
    """MPM 粉末再分布 → 剥蚀半宽 + 飞溅质量分数。

    物理图像
    --------
    金属蒸气射流在熔道上方形成低压区，卷吸两侧粉末：靠近熔道的粉末被
    横向抽走形成**剥蚀带 (denudation zone)**，动量足够大的颗粒直接脱离
    粉床成为**飞溅 (spatter)**。

    度量方式（均可微）
    ------------------
    * **剥蚀半宽**：横向位移的质量加权 RMS —— 粉末被抽离原位的横向尺度；
    * **飞溅质量分数**：竖向位置超出刮刀高度、且速度超过逃逸判据
      ``v_esc = sqrt(2 g h_bed)`` 的质量占比（双 sigmoid 软门控）。

    MPM 网格用**静态** ``grid_shape`` + 可微 ``dx``：域尺寸 ``grid_shape·dx``
    与颗粒坐标同比例缩放，故不会因粒径变化而出界。
    """
    from diffmech.methods.am.particle_am import (
        ParticleAMState, particle_layer_times, step_mpm_am_scan,
    )
    from diffmech.methods.mpm import MPMConfig, make_mpm_state

    state0, _ = synth_powder_bed(cfg)
    n = state0.position.shape[0]
    dim = int(cfg.dim)
    paths = single_track_path(cfg)
    starts, durations = particle_layer_times(paths, slm)

    dx = 2.0 * float(cfg.d50)
    lx = float(np.asarray(cfg.bed_lx))
    # T3(#24)：本包不能 import amforge.counts（依赖单向 amforge→diffmech）⇒ 按同宽 1e-9 内联。
    ratio = lx / dx
    n_grid = max(8, int(np.ceil(ratio - 1e-9 * max(1.0, abs(ratio)))) + 4)
    grid_shape = tuple([n_grid] * dim)
    origin = tuple([-2.0 * dx] * dim)
    if dt is None:
        # 弹性波 CFL：dt = 0.2 dx / c，c = sqrt(E/ρ)
        c = float(np.sqrt(float(youngs_modulus) / float(cfg.rho_powder)))
        dt = 0.2 * dx / max(c, 1e-9)

    mpm_cfg = MPMConfig(grid_origin=origin, grid_shape=grid_shape, dx=dx,
                        dt=float(dt), youngs_modulus=float(youngs_modulus),
                        poissons_ratio=0.3, rho0=float(cfg.rho_powder),
                        bulk_modulus=1.0e8, gravity=-9.81,
                        transfer="apic", material="solid")
    vol = (4.0 / 3.0) * jnp.pi * state0.radius ** 3 if dim == 3 \
        else jnp.pi * state0.radius ** 2
    mpm0 = make_mpm_state(state0.position, volume=vol, mass=state0.mass,
                          dim=dim)
    am0 = ParticleAMState(position=state0.position,
                          activation_time=jnp.zeros(n),
                          temperature=jnp.full(n, slm.preheat_temp),
                          mass=state0.mass,
                          layer_id=jnp.zeros(n, dtype=jnp.int64),
                          rho=jnp.full(n, float(cfg.rho_powder),
                                       dtype=jnp.float64))

    mpm_f, am_f = step_mpm_am_scan(mpm0, am0, float(dt), int(n_steps),
                                    mpm_cfg, slm, paths, starts, durations,
                                    bc="slip", cp=float(cp))

    disp = mpm_f.position - state0.position
    m = state0.mass
    m_tot = jnp.maximum(jnp.sum(m), 1e-30)

    # --- 剥蚀半宽：横向位移的质量加权 RMS -----------------------------
    ax_tran = 1 if dim == 3 else 0
    denudation = jnp.sqrt(
        jnp.sum(m * disp[:, ax_tran] ** 2) / m_tot + 1e-30)

    # --- 飞溅：越过刮刀高度 且 速度超逃逸判据 --------------------------
    zt = mpm_f.position[:, -1]
    h_bed = jnp.maximum(cfg.layer_thickness, 0.5 * float(cfg.d50))
    v_mag = jnp.sqrt(jnp.sum(mpm_f.velocity ** 2, axis=-1) + 1e-30)
    v_esc = jnp.sqrt(2.0 * 9.81 * h_bed)
    gate_h = jax.nn.sigmoid((zt - h_bed) / (0.2 * h_bed))
    gate_v = jax.nn.sigmoid((v_mag / jnp.maximum(v_esc, 1e-12) - 1.0) / 0.2)
    spatter = jnp.sum(m * gate_h * gate_v) / m_tot

    return {
        "denudation_width": denudation,
        "spatter_fraction": jnp.clip(spatter, 0.0, 1.0),
        "mean_displacement": jnp.sum(
            m * jnp.linalg.norm(disp, axis=-1)) / m_tot,
        "peak_temperature": jnp.max(am_f.temperature),
    }


# ===========================================================================
# 6. 解析代理（默认档）
# ===========================================================================
def analytic_powder_defects(*, ved, led, norm_enthalpy, layer_thickness,
                            hatch_spacing, beam_radius, d50: float,
                            packing_fraction: float = 0.55,
                            dim: int = 3) -> dict[str, Array]:
    """解析粉末尺度缺陷代理（快、可微、零颗粒开销）。

    这是默认档：用工艺无量纲数直接给出四项指标，供**每次**逆问题迭代
    廉价评估；颗粒尺度档（dem/sph/mpm）用于关键工艺点的高保真校验。

    判据来源
    --------
    * **铺粉密度**：粉层厚 / 粒径比 ``t/d50``。文献一致结论是 ``t/d50 < 1.5``
      时刮刀无法形成连续密实层（拱桥效应），``t/d50 > 3`` 后趋于批次表观
      密度。用 ``tanh`` 在两者间平滑过渡。
    * **未熔合**：搭接不足与层厚过大共同作用，
      ``LoF = σ[k(h/(2r_b) - 1)] ⊕ σ[k(t/d_melt - 1)]``，再按铺粉密度放大
      （疏松粉床更易未熔合）。
    * **球化**：低线能量密度 ``P/v`` → 熔道细长 → Plateau–Rayleigh。
    * **飞溅**：高无量纲焓 ``ΔH/h_s``（keyhole 阈值 ~30）→ 蒸气反冲抛料。
    """
    t = jnp.ravel(jnp.asarray(layer_thickness))[0]
    h = jnp.ravel(jnp.asarray(hatch_spacing))[0]
    rb = jnp.ravel(jnp.asarray(beam_radius))[0]
    ved_ = jnp.mean(jnp.asarray(ved))
    led_ = jnp.mean(jnp.asarray(led))
    nh_ = jnp.mean(jnp.asarray(norm_enthalpy))

    # (1) 铺粉密度：t/d50 拱桥效应
    ratio_td = t / jnp.maximum(d50, 1e-12)
    dens = packing_fraction * 0.5 * (1.0 + jnp.tanh((ratio_td - 1.5) / 0.8))
    packing_density = jnp.clip(dens + 0.06, 0.05, 0.64)
    # 配位数与密实度的经验单调关系（RCP 0.64 ↔ Z≈7）
    coordination = jnp.clip(7.0 * packing_density / 0.64, 0.0, 9.0)
    # 粗糙度 ≈ 粒径量级，随密实度下降而变差
    surface_roughness = d50 * (0.35 + 0.9 * (1.0 - packing_density / 0.64))

    # (2) 未熔合：搭接率 + 层厚/熔深
    overlap_def = jax.nn.sigmoid(3.0 * (h / jnp.maximum(2.0 * rb, 1e-12) - 1.0))
    # 熔深经验标定：d_melt ≈ 1.2·rb·(VED/VED_ref)^0.5，VED_ref = 60 J/mm³
    ved_ref = 60.0e9    # J/m³
    d_melt = 1.2 * rb * jnp.sqrt(jnp.maximum(ved_ / ved_ref, 1e-12))
    depth_def = jax.nn.sigmoid(3.0 * (t / jnp.maximum(d_melt, 1e-12) - 1.0))
    lof = 1.0 - (1.0 - overlap_def) * (1.0 - depth_def)     # 概率式或
    lof = jnp.clip(lof * (1.0 + 0.8 * (1.0 - packing_density / 0.64)),
                   0.0, 1.0)

    # (3) 球化：线能量密度过低
    led_ref = 200.0    # J/m（316L 常规窗口下限量级）
    balling = jax.nn.sigmoid(3.0 * (1.0 - led_ / led_ref))

    # (4) 飞溅：无量纲焓越过 keyhole 阈值
    spatter = jax.nn.sigmoid((nh_ - 30.0) / 8.0)
    denudation_width = (0.6 + 1.4 * spatter) * rb

    porosity = jnp.clip(0.6 * lof + 0.3 * spatter + 0.1 * balling, 0.0, 1.0) \
        * 0.08     # 缩放到物理孔隙率量级（<8%）

    return {
        "packing_density": packing_density,
        "coordination_number": coordination,
        "surface_roughness": surface_roughness,
        "balling_indicator": jnp.clip(balling, 0.0, 1.0),
        "lof_indicator": lof,
        "spatter_fraction": jnp.clip(spatter, 0.0, 1.0),
        "denudation_width": denudation_width,
        "porosity": porosity,
    }
