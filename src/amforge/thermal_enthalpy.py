"""高保真热解：enthalpy 法（焓法）相变热传导
=============================================

本模块把「连续介质热解带潜热/固液-液固相变」落地为 ForgeCore 可注册求解器
``thermal.enthalpy``，替换/补充代理级 ``thermal.history``，是 P2-①「真实模拟」
最前置的硬缺口。

物理方法
--------
采用 **enthalpy（焓）法**（Voller & Prakash 经典算法），把潜热吸收/释放嵌入
热传导方程，无需显式追踪固液界面：

    状态量 H(x,t) ：单位体积焓 [J/m³]
    ∂H/∂t = ∇·(k ∇T) + Q(x,t)            # Q 为移动高斯激光热源

其中温度与焓通过**平滑**的液相关系 f(T) 关联（避免可微性破缺）：

    H(T) = ρ c_p (T − T_amb) + ρ L f(T)
    f(T) = smoothstep(T; T_solidus, T_liquidus)   # 0=固, 1=液
    dH/dT = ρ c_p + ρ L f'(T)  ≥  ρ c_p  > 0       # 单调 → 可逆

于是 ∇T = (dT/dH) ∇H，扩散项改写为

    ∇·(k ∇T) = ∇·( α_H ∇H ),  α_H = k / (ρ c_p + ρ L f'(T))

在固/液区 f'=0 → α_H = k/(ρ c_p)（纯扩散）；在糊状区(mushy) f'>0 →
α_H 被压低，能量转入相变、温度平台出现——这正是潜热效应的数值体现，且整条
公式对 H、对工艺参数（扫描速度、功率等 tracer）**全程可微**。

输出 ThermalHistory 字段的物理来源
----------------------------------
* peak_temperature   : 逐体素峰值温度（扫描中跟踪）
* final_temperature  : 末态温度场
* cooling_rate       : 逐体素最大冷却速率（跟踪 dT/dt）
* thermal_gradient   : 末态温度梯度 |∇T|（凝固前沿 G）
* solidification_rate: R = 冷却速率 / max(G, ε)（凝固前沿推进速率）
* time_above_melt    : T > T_solidus 累计时间（重熔代理）

设计要点
--------
* 纯 JAX 实现，兼容 ``jax.jit`` / ``jax.grad`` / ``jax.lax.scan``。
* 液相关联 f 用 smoothstep，逆映射 T(H) 用固定 24 步**二分**（单调 → 收敛区间
  可预置，可微且无 Newton 在糊状区导数趋零的风险；A1 计划换成分段解析 + 2 步
  Newton 以取回 ~3.5× 速度）。
* **光源用单元体积分**（erf 可分离差分，``source_model="integrated"`` 缺省）：
  Σ_i frac_i = 1 到 1e-14 且**与 dx/r 无关**，故沉积剂量不被网格改变，r<dx 时也
  无需人为展宽光束（旧的中点取值 ``"point"`` 模型保留作 A/B 对照）。
* **时间离散步数由物理曝光时长决定**：t_exposure = 扫描路径长/速度，
  dt = t_exposure/n_steps，n_steps 由 CFL（缺省 cfl=0.35）自动推导
  （:func:`suggest_n_steps`）。步数不足只改时间分辨率、**不改能量剂量**；低于
  SSP-RK2 绝对上限直接报错（发散不会被蒸发封顶伪装成可行结果）。
* 实体表面**绝热**由跨界面通量置零实现（``_div_alpha_grad(..., mask=)``），
  空白体素不再充当恒温焓库；源/冷却/蒸发封顶按**线性 cut-cell 份额 fv** 加权，
  界面沉积份额是 O(dx²) 而非 O(dx)。
* 扫描配方（层数/道数/路径长）取自「含材料体素」足迹 ``sdf<dx/2``，与设计几何
  对齐，不随体素对齐抖动（见 :func:`_footprint`）。
* 边界：Neumann（绝热）为缺省；可经 ``params["boundary_conditions"]`` 喂入
  Dirichlet/对流/热流/绝热（对流是**表面**系数 [W/m²K]，除以 dx 摊成体积项）。
* 与 9 契约链一致：consumes (geometry, process)，produces ThermalHistory。
"""

from __future__ import annotations

import math
import warnings

import jax
import jax.numpy as jnp
from jax.scipy.special import erf

from amforge.core.contracts import PartGeometry, ProcessPlan, ThermalHistory
from amforge.core.registry import register_solver
from amforge.materials import get_material


# ---------------------------------------------------------------------------
# 相变（液相关联 f 与逆映射）—— 平滑、可微
# ---------------------------------------------------------------------------
def liquid_fraction(T, T_sol, T_liq):
    """平滑液相关系 f(T) ∈ [0,1]（smoothstep 跨糊状区）。"""
    dT = max(T_liq - T_sol, 1.0)
    s = jnp.clip((T - T_sol) / dT, 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


def d_liquid_fraction(T, T_sol, T_liq):
    """df/dT ≥ 0（smoothstep 导数，糊状区为凸包、边界处归零）。"""
    dT = max(T_liq - T_sol, 1.0)
    s = jnp.clip((T - T_sol) / dT, 0.0, 1.0)
    return 6.0 * s * (1.0 - s) / dT


def temperature_of_enthalpy(H, *, rho, cp, L, T_amb, T_sol, T_liq, n_iter=24):
    """T(H)：由单调焓-温关系反解（二分法，可微、稳定）。

    H(T) = ρ c_p (T−T_amb) + ρ L f(T) 对 T 严格单调增，故用二分法在
    [T_amb, T_amb+(H+ρL)/(ρc_p)] 内求根，不依赖导数、不会跨平段振荡，
    固定步数即可达机器精度，且全程对 H 可微。
    """
    a = rho * cp
    b = rho * L
    T_lo = jnp.full_like(H, T_amb)
    T_hi = T_amb + (H + b) / jnp.maximum(a, 1e-12)

    def body(carry, _):
        lo, hi = carry
        mid = 0.5 * (lo + hi)
        Hmid = a * (mid - T_amb) + b * liquid_fraction(mid, T_sol, T_liq)
        lo = jnp.where(Hmid < H, mid, lo)
        hi = jnp.where(Hmid < H, hi, mid)
        return (lo, hi), None

    (lo, hi), _ = jax.lax.scan(body, (T_lo, T_hi), jnp.arange(n_iter))
    return 0.5 * (lo + hi)


def enthalpy_of_temperature(T, *, rho, cp, L, T_amb, T_sol, T_liq):
    """H(T)：焓-温关系的正向映射（``temperature_of_enthalpy`` 的逆）。

        H(T) = ρ c_p (T − T_amb) + ρ L f(T)

    用于初值场（预热温度）与 Dirichlet 边界的焓强制覆盖；平滑、对 T 可微。
    """
    a = rho * cp
    b = rho * L
    return a * (T - T_amb) + b * liquid_fraction(T, T_sol, T_liq)


def effective_diffusivity(T, *, k, rho, cp, L, T_sol, T_liq):
    """糊状区被压低的有效扩散系数 α_H = k / (ρ c_p + ρ L f'(T))。"""
    denom = rho * cp + rho * L * d_liquid_fraction(T, T_sol, T_liq)
    return k / jnp.maximum(denom, 1e-12)


# ---------------------------------------------------------------------------
# 变系数扩散 ∇·(α ∇H) —— 3D/2D 通用，Neumann 边界
# ---------------------------------------------------------------------------
def _div_alpha_grad(H, alpha, dx, mask=None):
    """∇·(α∇H)，Neumann（绝热）边界。

    ``mask``（实体体素 0/1）给出时，**跨界面的通量为 0**：真实零件被真空/粉末
    包围，表面是绝热的。不给 mask 时保持原行为——网格外围的空白单元被当作
    恒温焓库（H=0→T=T_amb），实体表面按 α·H/dx 持续漏热，这是一个 **随 dx 变号
    不变的 O(dx) 冷偏**（粗网格漏得更多 → 峰值更低），A0 实测的"加密仍偏冷"
    残余阶次正是它。
    """
    div = jnp.zeros_like(H)
    ndim = H.ndim
    for ax in range(ndim):
        idx = jnp.arange(H.shape[ax])
        Hc = jnp.take(H, idx[:-1], axis=ax)          # 左邻，size N-1
        Hn = jnp.take(H, idx[1:], axis=ax)           # 右邻，size N-1
        ac = jnp.take(alpha, idx[:-1], axis=ax)
        an = jnp.take(alpha, idx[1:], axis=ax)
        face_a = 0.5 * (ac + an)
        if mask is not None:
            face_a = face_a * jnp.take(mask, idx[:-1], axis=ax) * \
                     jnp.take(mask, idx[1:], axis=ax)     # 界面通量置零=绝热
        dH = (Hn - Hc) / dx
        flux = face_a * dH                            # 面通量，size N-1
        z = jnp.zeros_like(jnp.take(flux, idx[:1], axis=ax))
        flux_r = jnp.concatenate([flux, z], axis=ax)  # 右面通量（末面=0 绝热）
        flux_l = jnp.concatenate([z, flux], axis=ax)  # 左面通量（首面=0 绝热）
        div = div + (flux_r - flux_l) / dx
    return div


# ---------------------------------------------------------------------------
# 移动高斯热源 + 分层扫描路径
# ---------------------------------------------------------------------------
def _scan_topology(coords, process, *, dim, solid=None):
    """分层 zigzag 扫描**拓扑**：层数/线数/实际道间距/物理路径长（与网格 dx 无关）。

    A0 之前此处有 ``clip(n_layers,1,8)``、``clip(n_lines,1,16)`` 两处截断，且层数
    公式写反（``layer_thickness/z_extent`` 而非 ``z_extent/layer_thickness``）——10 mm
    件需要 333 层/97 道，被压成 8 层/16 道后路径长与曝光时长双双错掉。现在拓扑
    只由几何 extent 与工艺 (layer_thickness, hatch_spacing) 决定，且**不再含 dx**。

    ``solid``（实体体素掩膜）给出时按**实体足迹**取 extent：栅格包围盒常带空白
    padding（GUI ``build_primitive`` 会外扩数格），用栅格 extent 会把路径长与曝光
    时长虚高，进而虚增沉积剂量。
    """
    x = coords[..., 0]; y = coords[..., 1]
    if solid is None:
        x_min, x_max = jnp.min(x), jnp.max(x)
        y_min, y_max = jnp.min(y), jnp.max(y)
    else:
        s = jnp.asarray(solid) > 0.5
        big = jnp.finfo(jnp.asarray(x).dtype).max / 4
        has = jnp.any(s)
        x_min = jnp.where(has, jnp.min(jnp.where(s, x, big)), jnp.min(x))
        x_max = jnp.where(has, jnp.max(jnp.where(s, x, -big)), jnp.max(x))
        y_min = jnp.where(has, jnp.min(jnp.where(s, y, big)), jnp.min(y))
        y_max = jnp.where(has, jnp.max(jnp.where(s, y, -big)), jnp.max(y))
    x_ext = jnp.maximum(x_max - x_min, 1e-9)
    y_ext = jnp.maximum(y_max - y_min, 1e-9)

    if dim >= 3:
        z = coords[..., 2]
        if solid is None:
            z_min, z_max = jnp.min(z), jnp.max(z)
        else:
            z_min = jnp.where(has, jnp.min(jnp.where(s, z, big)), jnp.min(z))
            z_max = jnp.where(has, jnp.max(jnp.where(s, z, -big)), jnp.max(z))
        z_ext = jnp.maximum(z_max - z_min, 1e-9)
        lt = jnp.mean(jnp.atleast_1d(jnp.asarray(process.layer_thickness)))
        n_layers = jnp.maximum(jnp.round(z_ext / jnp.maximum(lt, 1e-12)), 1.0)
    else:
        z_min = jnp.array(0.0)
        z_ext = jnp.array(1.0)
        n_layers = jnp.array(1.0)

    hatch = jnp.maximum(jnp.mean(jnp.atleast_1d(jnp.asarray(process.hatch_spacing))), 1e-12)
    n_lines = jnp.maximum(jnp.round(y_ext / hatch), 1.0)
    spacing = y_ext / n_lines                 # 均分后的实际道间距（≈ hatch）
    # 物理路径长 = 层数 ×（各道扫掠长度 + 道间跳距）；供求解器换算真实曝光时长。
    path_length = n_layers * (n_lines * x_ext
                              + jnp.maximum(n_lines - 1.0, 0.0) * spacing)
    return (n_layers, n_lines, x_min, y_min, z_min, x_ext, y_ext, z_ext,
            spacing, path_length)


def _build_scan_positions(coords, process, dx, *, n_steps, dim, solid=None):
    """按**归一化弧长**采样扫描位置 ``(n_steps, 3)``，并返回物理路径长。

    位置是 (layer, line, in-line fraction) 的连续函数：不再依赖 dx、不再有固定
    64 点缓冲（旧写法 ``xs_full=linspace(...,64)`` 配 ``n_per_line=round(x_ext/
    (dx*0.5))`` 会让激光只走过 ``n_per_line/64`` 的行程，行程覆盖率随网格变化）。
    第 t 步取时间区间中点 g=(t+0.5)/n_steps，使移动热源的积分与二阶时间精度对齐；
    每步激光移动 v·dt，稠密程度由 n_steps（→ CFL）决定而非网格。全程 tracer 安全，
    可对工艺参数（hatch_spacing / layer_thickness）与几何坐标求梯度。``dim==2``
    走单层面内扫描（z 恒为 0）。
    """
    (n_layers, n_lines, x_min, y_min, z_min, x_ext, y_ext, z_ext,
     spacing, path_length) = _scan_topology(coords, process, dim=dim, solid=solid)
    x_max = x_min + x_ext
    layer_h = z_ext / n_layers
    step_g = (jnp.arange(n_steps).astype(jnp.float64) + 0.5) / jnp.maximum(n_steps, 1)

    def pos_at(g):
        gl = g * n_layers
        lf = jnp.clip(jnp.floor(gl), 0.0, n_layers - 1.0)
        gj = (gl - lf) * n_lines
        jf = jnp.clip(jnp.floor(gj), 0.0, n_lines - 1.0)
        u = jnp.clip(gj - jf, 0.0, 1.0)
        # 偶数道沿 +x、奇数道反向（zigzag）；层间旋转策略 scan_angle 尚未接入
        # 路径生成（见 docs 缺口清单）。
        xc = jnp.where(jnp.mod(jf, 2.0) < 1.0, x_min + u * x_ext, x_max - u * x_ext)
        yc = y_min + (jf + 0.5) * spacing
        zc = z_min + (lf + 0.5) * layer_h if dim >= 3 else jnp.array(0.0)
        return jnp.stack([xc, yc, zc])

    positions = jax.vmap(pos_at)(step_g)
    return positions, path_length


def _moving_source(positions_t, cell_centers, Q0, r, dp):
    """单时刻体积热源：可分离 3D 高斯 Q(x) = Q0·exp(−ρ²_xy/r²)·exp(−z²/(2 dp²))。

    面内与轴向各自单调衰减（恒 ≤ Q0，无放大项），对任意几何（含厚 z 方向）
    均稳定；∫Q dV = Q0·π r²·dp√(2π) = ηP（与 Q0 归一化一致）。dim>=3 走
    面内+轴向双高斯；dim==2 退化为纯面内热斑（仍为数值热固结）。
    """
    planar = cell_centers[..., :2] - positions_t[:2]
    planar2 = jnp.sum(planar ** 2, axis=-1)
    if cell_centers.shape[-1] >= 3:
        z2 = (cell_centers[..., 2] - positions_t[2]) ** 2
        decay = jnp.exp(-planar2 / jnp.maximum(r * r, 1e-18)) * \
                jnp.exp(-z2 / jnp.maximum(2.0 * dp * dp, 1e-18))
    else:
        decay = jnp.exp(-planar2 / jnp.maximum(r * r, 1e-18))
    return Q0 * decay


def _cell_integrated_source(position_t, cell_centers, power, r, dp, dx):
    """单元体积分热源：Q_i = power · frac_i / dV，frac_i 为高斯在单元 i 上的解析积分。

    与 ``_moving_source``（中点取值）用**同一个空间形状**——面内
    ``exp(−ρ²/r²)``（等效 σ=r/√2 的高斯）+ 轴向 ``exp(−z²/(2·dp²))``（σ=dp）——
    但每个单元截获的份额按 erf 差值精确积分：

        frac_axis = ½ [erf((c+dx/2 − p)/σ) − erf((c−dx/2 − p)/σ)]

    三方向可分离相乘，故 Σ_i frac_i = 1（到机器精度），**与 dx、r 的比值无关**。
    于是粗网格不再"漏能量"、细网格不再"稀释峰值"，且 r<dx 时无需人为展宽光束
    （那是中点取值模型才需要的补丁）。物理上仍允许 Σ frac_i < 1：光束越过零件
    边界时那部分功率确实没有沉积进工件，这是真实损失而非离散误差。
    """
    s_in = jnp.maximum(r, 1e-12) / math.sqrt(2.0)
    s_z = jnp.maximum(dp, 1e-12)

    def _axis_frac(cen, pos, sig):
        return 0.5 * (erf((cen + 0.5 * dx - pos) / sig)
                      - erf((cen - 0.5 * dx - pos) / sig))

    frac = _axis_frac(cell_centers[..., 0], position_t[0], s_in) * \
        _axis_frac(cell_centers[..., 1], position_t[1], s_in)
    if cell_centers.shape[-1] >= 3:
        frac = frac * _axis_frac(cell_centers[..., 2], position_t[2], s_z)
    dV = dx ** int(cell_centers.shape[-1])
    return power * frac / dV


def _stability_limit_dx2(dx, dim, alpha0, cfl):
    """显式扩散稳定上限：dt_max = cfl·dx² / (2·dim·α₀)。"""
    return cfl * dx * dx / (2.0 * float(dim) * alpha0 + 1e-18)


def _concrete(value):
    """把可能处于 trace 中的标量固化为 Python float；固化失败返回 None。

    显式时间步数必须是静态形状（``jnp.arange(n_steps)``），因此 CFL 校核只能在
    eager 模式做；在 ``jax.grad``/``jit`` 追踪下返回 None，校核自动跳过，可微性
    与静态形状均不受影响。
    """
    try:
        return float(value)
    except jax.errors.ConcretizationTypeError:
        return None


def _footprint(geometry):
    """「含材料的体素」掩膜：``sdf < dx/2``（等价于线性 cut-cell 份额 fv>0）。

    专供**扫描足迹/路径长**推导，不用严格的 ``sdf < 0``：后者会把体素中心恰好
    落在设计边界面（``sdf == 0``）上的一圈剔掉，于是同一零件在不同 dx 下对齐
    情况不同 → 层数/道数/路径长随网格抖动（实测试片四档 path=3.267/4.775/
    5.012/5.131mm，极差 36%）。工艺配方是**设计几何**的属性，不该由体素对齐
    决定；扩散算符仍严格用 ``sdf < 0`` 的实体掩膜，两者用途不同。
    """
    dx = jnp.asarray(geometry.spacing, dtype=jnp.float64)
    return jnp.asarray(geometry.sdf, dtype=jnp.float64) < 0.5 * dx


def suggest_n_steps(geometry, process, *, material="316L", cfl=0.35):
    """CFL 稳定 + 覆盖物理曝光时长所需的显式时间步数（eager 纯函数）。

    ``solve_enthalpy_thermal`` 在 ``params`` 不给 ``n_steps`` 时自动取本值；单独暴露
    是给 trace 模式用的——``jax.grad`` 之下工艺量是 tracer，步数（静态形状）无法
    从 tracer 推出，必须先在 eager 模式算好再传入。
    """
    mat = get_material(material)
    alpha0 = float(mat.k_solid) / (float(mat.rho_solid) * float(mat.cp_solid) + 1e-12)
    coords = geometry.coords()
    solid = _footprint(geometry)
    path_length = _scan_topology(coords, process, dim=geometry.dim,
                                 solid=solid)[-1]
    v = jnp.mean(jnp.atleast_1d(jnp.asarray(process.scan_speed)))
    t_exp = float(path_length / jnp.maximum(v, 1e-9))
    dx = jnp.asarray(geometry.spacing, dtype=jnp.float64)
    dt_max = float(_stability_limit_dx2(dx, geometry.dim, alpha0, cfl))
    return max(1, int(math.ceil(t_exp / dt_max - 1e-9)))


def chain_schedule(geometry, process, *, material="316L", cfl=0.35,
                   speed_slack=2.0, path_slack=2.0, max_steps=200000,
                   fixed_n_steps=None, resolution_policy="strict"):
    """**这张网格上默认链的可用档位**：静态时间调度 + 与之自洽的工艺搜索子盒。

    动机（2026-10-06 D0）：A0 把 ``dt`` 钉在物理曝光时长上之后，``n_steps``（时间
    扫描的静态形状）必须由 ``jax.grad`` **之外**的静态量决定；而工艺搜索盒里的
    任何一点都可能把曝光推得更长（更慢的 ``scan_speed``、更密的
    ``hatch_spacing``、更薄的 ``layer_thickness``）。所以档位不能"按名义工艺算一次"
    了事，而要**在盒的最坏角落**算：

        曝光上界 = max(盒内最大扫描路径长, 名义工艺路径长) / 盒内最小扫描速度
        n_steps  = ceil(曝光上界 / dt_target)

    返回的 ``bounds`` 与 ``n_steps``/``exposure_bound_s`` 是同一套判据的两面：
    优化器无论走到哪个可达工艺，钉住的 ``n_steps`` 既覆盖曝光又满足稳定上限，
    三条时间离散校核在 trace 内**既不会误报、也不会静默放行**。

    代价上限显式可见：返回的 ``voxel_steps = n_steps × 体素数`` 即一次正向的
    voxel-step 量。若 ``n_steps`` 超 ``max_steps`` 直接报错并指出出路（粗网格 /
    提到 cfl=1 档 / 走 β 降阶），这正是开发日志 §24.3 零件尺度外推的结论——
    显式瞬态解在零件尺度不可用要**换算法**（§25.8 的 D2/D3），而不是悄悄放宽校核。

    ``fixed_n_steps`` 给定时反过来**由步数定工艺盒**：把速度下界抬到"钉住的
    ``n_steps`` 覆盖得住曝光"，用于调用方自带调度的场合（返回的 ``n_steps`` 即该值）。
    """
    from amforge.process import PROCESS_BOUNDS
    try:
        return _chain_schedule(geometry, process, material=material, cfl=cfl,
                               speed_slack=speed_slack, path_slack=path_slack,
                               max_steps=max_steps, bounds=dict(PROCESS_BOUNDS),
                               fixed_n_steps=fixed_n_steps,
                               resolution_policy=resolution_policy)
    except jax.errors.ConcretizationTypeError as e:
        raise ValueError(
            "chain_schedule() 必须在 eager 模式用**具体数值**的名义工艺调用"
            "（它的作用就是在 trace 之前把静态调度钉下来）：请在 jax.grad/jit 之外"
            "对初始工艺调用一次，再把返回值放进 params['thermal']。"
        ) from e


def _chain_schedule(geometry, process, *, material, cfl, speed_slack, path_slack,
                    max_steps, bounds, fixed_n_steps=None,
                    resolution_policy="strict"):
    mat = get_material(material)
    alpha0 = float(mat.k_solid) / (float(mat.rho_solid) * float(mat.cp_solid) + 1e-12)
    dx = float(jnp.asarray(geometry.spacing, dtype=jnp.float64))
    dt_target = float(_stability_limit_dx2(dx, geometry.dim, alpha0, cfl))
    solid = _footprint(geometry)
    coords = geometry.coords()

    def _m(v):
        return float(jnp.mean(jnp.atleast_1d(jnp.asarray(v, dtype=jnp.float64))))

    path_nom = float(_scan_topology(coords, process, dim=geometry.dim,
                                    solid=solid)[-1])
    v_nom, h_nom = _m(process.scan_speed), _m(process.hatch_spacing)
    lt_nom, r_nom = _m(process.layer_thickness), _m(process.beam_radius)

    # 子盒下界：与 eager 护栏同源（2r ≥ dx），外加"道距/层厚至少一个体素可分"，
    # 并把路径长/曝光的**可达放大**限制在 path_slack/speed_slack 内。
    r_lo = max(bounds["beam_radius"][0], 0.5 * dx)
    h_lo = max(bounds["hatch_spacing"][0], dx, h_nom / path_slack)
    lt_lo = max(bounds["layer_thickness"][0], dx, lt_nom / path_slack)
    v_lo, v_hi = max(bounds["scan_speed"][0], v_nom / speed_slack), bounds["scan_speed"][1]

    if dx > 2.0 * r_nom:  # 名义工艺本身在本网格就欠采样（r_lo 恒 ≥dx/2，故不能拿它比）
        if resolution_policy == "strict":
            raise ValueError(
                f"名义工艺在本网格上热源欠采样：dx={dx*1e6:.1f}µm > 2r="
                f"{2*r_nom*1e6:.1f}µm。请先加密网格（dx≤2r，定量熔池形态"
                f"要 dx≤r/2，见 §25.2）或放宽光斑；chain_schedule() 不为欠采样网格定价。")
        warnings.warn(
            f"演示档档位定价：名义工艺 dx={dx*1e6:.1f}µm > 2r={2*r_nom*1e6:.1f}µm，"
            f"子盒半径下界已抬到 r≥dx/2={0.5*dx*1e6:.1f}µm，解算时同样按 dx/2 投影"
            f"（能量守恒，形态/峰值不可用于标定）。要按标定档跑请加密网格或去掉 "
            f"resolution_policy='demo'。",
            stacklevel=2)

    # 路径长只由几何 extent 与 (hatch, layer_thickness) 决定，与 scan_speed 无关。
    # 最坏角落取"盒下界与实际名义工艺里更细的那个"：路径随 hatch/lt 单调变长，
    # min() 即 max(path)。名义工艺若比网格可分辨的下界还细，调度**必须**按它的真实
    # 曝光定价（否则 dt 越过稳定上限直接发散），只是形态欠分辨——单独警告。
    h_worst, lt_worst = min(h_lo, h_nom), min(lt_lo, lt_nom)
    if h_nom < dx or lt_nom < dx:
        warnings.warn(
            f"名义工艺在本网格上欠分辨：hatch_spacing={h_nom*1e6:.1f}µm、"
            f"layer_thickness={lt_nom*1e6:.1f}µm 小于体素 dx={dx*1e6:.1f}µm（道距/"
            f"层厚至少要占一个体素才解得开）。时间调度已按**实际**曝光定价所以数值"
            f"仍稳定，但逐层累积与熔池形态会失真：请加密网格或把工艺量抬到 ≥ dx。"
            f"（搜索子盒仍取 ≥ dx 的可分辨下界。）",
            stacklevel=2)

    worst = process.replace(hatch_spacing=jnp.asarray(h_worst),
                            layer_thickness=jnp.asarray(lt_worst),
                            beam_radius=jnp.asarray(max(r_nom, r_lo)))
    path_max = float(_scan_topology(coords, worst, dim=geometry.dim,
                                    solid=solid)[-1])
    # 速度下界还要保证「钉住的步数覆盖得住最坏曝光」：n_steps·dt_target ≥ path/v
    dt_budget = (int(fixed_n_steps) if fixed_n_steps else int(max_steps)) * dt_target
    v_lo = max(v_lo, path_max / max(dt_budget, 1e-30))
    if v_lo >= v_hi:
        raise ValueError(
            f"本网格的显式热解预算内没有可积的工艺窗口：需要 scan_speed ≥ "
            f"{v_lo:.3g} m/s，而设备上界只有 {v_hi:.3g} m/s（dx={dx*1e6:.1f}µm, "
            f"最坏路径长={path_max*1e3:.1f}mm, 步数预算="
            f"{int(fixed_n_steps) if fixed_n_steps else int(max_steps)}）。"
            f"出路：粗化网格／加大步数预算／走 β 降阶档（§25.8 D3）。")
    t_bound = path_max / v_lo
    n_steps = (int(fixed_n_steps) if fixed_n_steps
               else max(1, int(math.ceil(t_bound / dt_target - 1e-9))))
    nvox = int(jnp.asarray(geometry.sdf).size)
    if n_steps > max_steps:
        n_stable = max(1, int(math.ceil(t_bound / (dt_target / max(cfl, 1e-12)) - 1e-9)))
        raise ValueError(
            f"本网格 + 本工艺盒的显式热解需要 n_steps={n_steps} > max_steps="
            f"{max_steps}（dx={dx*1e6:.1f}µm, 曝光上界={t_bound:.3e}s, 体素={nvox}, "
            f"代价上界={n_steps*nvox:.2e} voxel-step）。可选出路：① 加密→**粗化**网格"
            f"或缩小单次扫描行程；② cfl→1（仍需 n_steps>={n_stable}）；③ 放宽"
            f"speed_slack/path_slack（会缩小可优化的工艺窗口）；④ 改走路线 β "
            f"（本征应变/降阶热，见 §25.8 D3）。")

    bounds["beam_radius"] = (r_lo, bounds["beam_radius"][1])
    bounds["hatch_spacing"] = (h_lo, bounds["hatch_spacing"][1])
    bounds["layer_thickness"] = (lt_lo, bounds["layer_thickness"][1])
    bounds["scan_speed"] = (v_lo, bounds["scan_speed"][1])
    return {"n_steps": n_steps, "exposure_bound_s": t_bound, "cfl": cfl,
            "max_steps": max(max_steps, n_steps), "bounds": bounds,
            "path_bound_m": path_max, "voxel_steps": n_steps * nvox}


def _evap_sink(T, T_lo, T_hi, c):
    """蒸发/反冲散热封顶项（平滑、可微）。

    当局部温度越过 ``T_lo``（接近沸点）后，按 smoothstep 平滑介入，把多余
    能量移出系统，将峰值温度封顶在蒸发温度 ``T_hi`` 量级——这正是真实 LPBF
    中高功率/低扫描速度下「蒸发反冲 + 等离子体屏蔽」把熔池峰值钳制在蒸气化
    温度附近的物理机制，也是峰值温度不可能无限增长的根本原因（物理上限 ≈
    材料沸点）。低温区（正常熔化，T<<T_lo）该项恒为 0，**完全不干扰潜热相变
    与正常熔池演化**，因此对熔化、对 jax.grad 均透明。
    """
    x = jnp.clip((T - T_lo) / jnp.maximum(T_hi - T_lo, 1.0), 0.0, 1.0)
    ss = x * x * (3.0 - 2.0 * x)
    return c * ss * jnp.maximum(T - T_lo, 0.0)


# ---------------------------------------------------------------------------
# 主求解器
# ---------------------------------------------------------------------------
@register_solver(
    "thermal.enthalpy",
    consumes=("PartGeometry", "ProcessPlan"),
    produces="ThermalHistory",
    stage="thermal",
    modality=("SLM", "LSF"),
    differentiable=True,
    cost=20.0,
    defaults={"material": "316L"},
    doc="enthalpy 法相变热传导：移动高斯热源 + 固液潜热（高保真热解）；"
        "用 select={'thermal':'thermal.enthalpy'} 显式启用。",
)
def solve_enthalpy_thermal(*, geometry, process, params=None):
    """enthalpy 法相变热传导：移动激光热源 + 固液潜热。

    注册名 ``thermal.enthalpy``；consumes (geometry, process)；produces
    ``ThermalHistory``。材料自 ``params['material']`` 解析（默认 316L）。
    """
    p = dict(params or {})
    mat = get_material(p.get("material", "316L"))

    # 边界/初值条件（缺口 #20-②A）：经 params["boundary_conditions"] 可选喂入。
    # 惰性导入以避免与 amforge.boundary 的循环依赖；缺省（None）时保持原
    # 全局弱冷却行为，逐位不变、不破坏既有测试。
    from amforge.boundary import (
        BoundaryCollection, boundary_terms, initial_enthalpy_field, face_mask,
    )
    _bc_raw = p.get("boundary_conditions")
    bcs = None
    if _bc_raw is not None:
        bcs = (_bc_raw if isinstance(_bc_raw, BoundaryCollection)
               else BoundaryCollection.from_dict(_bc_raw))

    rho = float(mat.rho_solid)
    cp = float(mat.cp_solid)
    k = float(mat.k_solid)
    L = float(mat.latent_fusion)                       # 潜热 [J/kg]
    T_amb = float(mat.T_ambient)
    T_sol = float(mat.T_solidus)
    T_liq = float(mat.T_liquidus)
    T_boil = float(mat.T_boil)                         # 蒸发/沸点（峰值封顶上限）

    # 蒸发封顶强度：温度越过 (T_boil-200K) 后平滑介入，将峰值钳制在蒸气化上限
    # 附近。c_evap 经标定使高功率（线能量≫合理窗）工况峰值落在 ~3000K 而非爆到
    # 8000K+；低温熔化区不介入，故不影响正常相变与可微性。可由 params 覆盖。
    c_evap = float(p.get("evap_coeff", 3.0e11))
    T_evap_lo = T_boil - 200.0

    spacing = jnp.asarray(geometry.spacing, dtype=jnp.float64)  # tracer 安全
    dx = spacing
    mask = (geometry.sdf < 0.0).astype(jnp.float64)    # 仅在实体内部加热
    # 线性化固相体积分数（cut-cell 份额）：|SDF|≤dx/2 的表面体素按被平面切出的
    # 份额计。用它加权**源项**（而非二值 mask）可把"界面沉积份额"从 O(dx) 一阶
    # 几何误差降到 O(dx²)——否则同一物理工况在不同网格上沉积的总剂量本身漂移。
    fv = jnp.clip(0.5 - jnp.asarray(geometry.sdf, dtype=jnp.float64) / dx, 0.0, 1.0)
    coords = geometry.coords()                          # (..., dim)

    # 工艺量（保持可微：以 tracer 形式进入热源位置/功率）
    P = jnp.mean(jnp.atleast_1d(jnp.asarray(process.laser_power)))
    r = jnp.mean(jnp.atleast_1d(jnp.asarray(process.beam_radius)))
    eta = jnp.mean(jnp.atleast_1d(jnp.asarray(process.absorption)))
    _v = jnp.mean(jnp.atleast_1d(jnp.asarray(process.scan_speed)))

    # 光源离散模型：
    #   integrated（默认）—— 单元体积分高斯，Σ_i frac_i = 1 到机器精度，故沉积
    #     能量剂量与 dx 无关，r<dx 时也**无需**人为展宽光束。不做任何网格补偿。
    #   point —— 旧的中点取值模型（含 r_eff=max(r,dx) 展宽补丁、clip(v_ref/v) 与
    #     heat_scale=1.4 两处粗网格标定旋钮），仅用于 A/B 对照与复现旧量级。
    source_model = str(p.get("source_model", "integrated"))
    if source_model not in ("integrated", "point"):
        raise ValueError(
            f"source_model 仅支持 'integrated' 或 'point'，收到 {source_model!r}")

    # 分辨率档位（D0 §25.10 的严重度分层）：
    #   strict（缺省，标定档）—— dx>2r 直接报错：峰值/熔池形态无意义，不给数字。
    #   demo（默认链/装配-GUI 档）—— 把光源半径抬到 dx/2 并**警告+记录**：能量仍守恒，
    #     但输出不代表可标定的形态。抬半径不是"补偿旋钮"，而是把欠分辨工艺投影到
    #     chain_schedule() 子盒下界（r ≥ dx/2）上，与优化器可达的那部分工艺盒一致。
    policy = str(p.get("resolution_policy", "strict"))
    if policy not in ("strict", "demo"):
        raise ValueError(
            f"resolution_policy 仅支持 'strict' 或 'demo'，收到 {policy!r}")

    # 熔深尺度（物理选择 ≈1.2 倍光束半径）
    dp = jnp.maximum(r * 1.2, 1e-9)

    if source_model == "point":
        r_src = jnp.maximum(r, dx)
        dp = jnp.maximum(r_src * 1.2, dx)
        # 热源强度归一化：∫Q dV = η P（连续意义下；离散中点取值不守恒 → 靠下面两旋钮补）
        Q0 = eta * P / (math.pi * r_src * r_src * math.sqrt(2.0 * math.pi) * dp + 1e-18)
        Q0 = Q0 * jnp.clip(0.8 / jnp.maximum(_v, 1e-9), 0.3, 3.0)
        Q0 = Q0 * float(p.get("heat_scale", 1.4))
    else:
        r_src = r if policy == "strict" else jnp.maximum(r, 0.5 * dx)
        if policy != "strict":
            dp = jnp.maximum(dp, 0.6 * dx)   # 熔深尺度与抬升后的半径同源（1.2·dx/2）
        Q0 = None
        # P/v 的一阶效应由「曝光时长 = 路径长/v」自动体现，无需再乘 v_ref/v。
        laser_power = eta * P * float(p.get("heat_scale", 1.0))

    hcool = float(p.get("h_cool", 0.5))                # 弱 Newton 冷却 [1/s]

    (n_layers, n_lines, _x_min, _y_min, _z_min, _x_ext, _y_ext, _z_ext,
     _spacing_y, path_length) = _scan_topology(coords, process, dim=geometry.dim,
                                               solid=_footprint(geometry))

    alpha0 = k / (rho * cp + 1e-12)
    cfl = float(p.get("cfl", 0.35))
    dt_rk2 = _stability_limit_dx2(dx, geometry.dim, alpha0, 1.0)   # SSP-RK2 绝对上限
    dt_target = dt_rk2 * cfl                                       # 精度目标

    # 曝光时长是**物理量**：t_exposure = 扫描路径长 / 扫描速度（层间 recoat 停留
    # 时间的冷却尚未建模，见 docs 缺口清单）。dt = t_exposure/n_steps，于是 n_steps
    # 只控制时间离散分辨率，加密网格或改步数都**不再**改写注入能量。
    # （旧写法 dt=min(stability_dt, total·dx/(v·n_steps)) 使总曝光 = n_steps·stability_dt
    #  ∝ n_steps·dx² —— 网格与步数直接改写能量剂量，实测表现为"越加密越冷"且全案
    #  不熔化；这正是 2026-10-06 A0 要修的阻塞缺陷。）
    t_exposure = path_length / jnp.maximum(_v, 1e-9)

    dt_c = _concrete(dt_target)
    dt_r = _concrete(dt_rk2)
    t_exp = _concrete(t_exposure)
    # 静态曝光上界：trace 下 ``t_exposure`` 是 tracer（含 scan_speed/hatch_spacing），
    # 但「本次调度最多要覆盖多长时间」是调用方在 eager 用 :func:`chain_schedule`
    # 钉好的**静态量**。有了它，下面三条时间离散校核就能在 ``jax.grad``/``jit`` 内
    # 照常执行——A0 之前 trace 下校核整体跳过，等于给显式解法留了一个「步数不足→
    # 扩散发散→被蒸发封顶伪装成可行结果」的静默洞。
    _bound = p.get("exposure_bound_s")
    if _bound is not None:
        t_bound = float(_bound)
    else:
        t_bound = t_exp
    _ok = (dt_c is not None and dt_r is not None and t_bound is not None)
    if _ok and t_exp is not None and t_exp > t_bound * (1.0 + 1e-9):
        raise ValueError(
            f"params['exposure_bound_s']={t_bound:.3e}s 小于本算例的实际物理曝光 "
            f"{t_exp:.3e}s：调度上界必须**覆盖**曝光，否则 dt 会越过稳定上限而发散。"
            f"请用 chain_schedule() 重算或放大 exposure_margin。")
    n_steps_cfl = max(1, int(math.ceil(t_bound / dt_c - 1e-9))) if _ok else None
    n_steps_stable = max(1, int(math.ceil(t_bound / dt_r - 1e-9))) if _ok else None

    n_steps_user = p.get("n_steps")
    if n_steps_user is None:
        if n_steps_cfl is None:
            raise ValueError(
                "未指定 params['n_steps']，且当前处于 trace（jax.grad/jit）模式下无法"
                "自动推导 CFL 步数：请在 eager 模式用 chain_schedule()（同时给出 "
                "n_steps 与 exposure_bound_s）或 suggest_n_steps() 求得后显式传入。")
        max_steps = int(p.get("max_steps", 20000))
        if n_steps_cfl > max_steps:
            raise ValueError(
                f"CFL 精度目标需要 n_steps>={n_steps_cfl}，超出预算 max_steps="
                f"{max_steps}（dx={float(dx)*1e6:.1f}µm, 路径长={float(path_length)*1e3:.1f}"
                f"mm, 曝光={t_bound:.3e}s）。显式热解在此网格/行程上不可行：请粗化网格、"
                f"缩短单次扫描行程、把 cfl 提到≤1 的下限档位（n_steps>={n_steps_stable}，"
                f"精度下降但稳定），或改走路线 β（本征应变降阶）/准稳态近似。")
        n_steps = n_steps_cfl
    else:
        n_steps = int(n_steps_user)
        if n_steps_stable is not None and n_steps < n_steps_stable:
            raise ValueError(
                f"params['n_steps']={n_steps} 不足：dt=曝光/n_steps 超过 SSP-RK2 稳定上限 "
                f"{dt_r:.3e}s（dx={float(dx)*1e6:.1f}µm, cfl=1）。覆盖 {t_bound:.3e}s 的物理"
                f"曝光至少需要 n_steps>={n_steps_stable}（精度目标 n_steps>={n_steps_cfl}，"
                f"即 cfl={cfl}）。步数不足时扩散项发散，蒸发封顶会把发散伪装成可行结果。")
        if (n_steps_cfl is not None and n_steps < n_steps_cfl
                and t_exp is not None):
            warnings.warn(
                f"params['n_steps']={n_steps} 低于 CFL 精度目标 {n_steps_cfl}"
                f"（cfl={cfl}）：仍稳定但每步激光移动 "
                f"{float(_v) * t_exp / n_steps * 1e6:.0f}µm，移动热源被时间欠采样，"
                f"峰值温度会偏低。建议 suggest_n_steps() 或提高 n_steps。",
                stacklevel=2)

    # 扫描足迹必须与上面的 _scan_topology（n_steps、path_length）用**同一个**掩膜，
    # 否则曝光时长一致而位置轨迹按另一套拓扑展开，两者错位。
    positions, _ = _build_scan_positions(coords, process, dx, n_steps=n_steps,
                                         dim=geometry.dim,
                                         solid=_footprint(geometry))
    dt = t_exposure / jnp.maximum(n_steps, 1)

    if source_model == "integrated":
        def _source_at(t):
            return _cell_integrated_source(positions[t], coords, laser_power,
                                           r_src, dp, dx)
    else:
        def _source_at(t):
            return _moving_source(positions[t], coords, Q0, r_src, dp)

    # 热源空间欠采样校核（eager）：一个体素装不下光斑直径时，峰值温度无物理意义。
    r_c, dx_c = _concrete(r), _concrete(dx)
    if r_c is not None and dx_c is not None and dx_c > 2.0 * r_c:
        if policy == "strict":
            raise ValueError(
                f"体素 dx={dx_c*1e6:.1f}µm 大于光斑直径 2r={2*r_c*1e6:.1f}µm：热源完全欠"
                f"采样，能量虽守恒但峰值/熔池尺寸无意义。请加密网格至 dx≤2r、放宽光斑，"
                f"或在明知只是要跑通链条时设 params['thermal']"
                f"['resolution_policy']='demo'。")
        warnings.warn(
            f"演示档（resolution_policy='demo'）：体素 dx={dx_c*1e6:.1f}µm > 2r="
            f"{2*r_c*1e6:.1f}µm，光源半径已抬到 dx/2={0.5*dx_c*1e6:.1f}µm 以使该网格上"
            f"解得开。能量守恒（Σfrac=1）不受影响，但**峰值与熔池形态不代表物理**，"
            f"不得用于标定或形态判据；标定档请用缺省 resolution_policy='strict'。",
            stacklevel=2)
    if r_c is not None and dx_c is not None and dx_c > r_c:
        # 实测（2026-10-06 A0 J3，试片 1.2×0.6×0.4mm、600W/0.8m·s⁻¹/r=100µm）：
        # dx=r 档横向仅 2 个体素跨过 1/e² 光斑，与 dx=r/4 档相比峰值只低 1.1%
        # （2589.4K vs 2617.8K，且两档都已撞上蒸发封顶），但**熔池形态**崩塌：
        # 熔宽低 45%（0.300 vs 0.550mm）、熔体积低 36%（0.0460 vs 0.0715mm³）。
        # 能量守恒不受影响（Σfrac=1 与 dx 无关），故只警告不报错；但取熔池形态
        # 或峰值做标定/判据时必须 dx≤r/2。
        warnings.warn(
            f"体素 dx={dx_c*1e6:.1f}µm 大于光束半径 r={r_c*1e6:.1f}µm：横向仅约 "
            f"{2*r_c/dx_c:.1f} 个体素跨过光斑，能量守恒但**熔池形态与峰值不可信**"
            f"（实测 dx=r 档熔宽偏低 ~45%、熔体积偏低 ~36%）。定量熔池形态请取 "
            f"dx≤r/2。",
            stacklevel=2)

    # 初值场：bcs 显式给了 IC 时用 BC 的 IC；否则用**工艺预热温度**（此前
    # preheat_temp 被完全忽略——本模块从不读它）。预热缺省/不高于环境温度时
    # H0 恒为 0，与旧行为逐位一致。
    pre = jnp.mean(jnp.atleast_1d(jnp.asarray(process.preheat_temp)))
    T_init = jnp.where(jnp.isfinite(pre) & (pre > T_amb), pre, T_amb)
    H0_pre = enthalpy_of_temperature(T_init, rho=rho, cp=cp, L=L, T_amb=T_amb,
                                     T_sol=T_sol, T_liq=T_liq) * mask
    if bcs is not None and bcs.ic is not None:
        H0 = initial_enthalpy_field(
            geometry, bcs, rho=rho, cp=cp, L=L, T_amb=T_amb,
            T_sol=T_sol, T_liq=T_liq)
    else:
        H0 = H0_pre

    # 面掩膜只依赖几何：在时间扫描**外**算一次。face_mask 内含全场 sdf_normal，
    # 逐步重算会让每步成本翻数倍（n_steps 现在按 CFL 决定，动则上千步）。
    bc_masks = {}
    dirichlet_override = []
    if bcs is not None:
        for bc in bcs.bcs:
            if bc.face not in bc_masks:
                bc_masks[bc.face] = face_mask(geometry, bc.face)
            if bc.kind == "dirichlet":
                Hf = enthalpy_of_temperature(
                    jnp.asarray(float(bc.value)), rho=rho, cp=cp, L=L,
                    T_amb=T_amb, T_sol=T_sol, T_liq=T_liq)
                dirichlet_override.append((bc_masks[bc.face], Hf))

    def rhs(H, t):
        T = temperature_of_enthalpy(H, rho=rho, cp=cp, L=L, T_amb=T_amb,
                                    T_sol=T_sol, T_liq=T_liq)
        alpha = effective_diffusivity(T, k=k, rho=rho, cp=cp, L=L,
                                      T_sol=T_sol, T_liq=T_liq)
        # 实体表面绝热：跨界面通量置零（mask 传入），空白单元不再充当恒温焓库
        lap = _div_alpha_grad(H, alpha, dx, mask=mask)
        # 源/散热按固相体积分数 fv 加权（不是二值 mask）：见上方 fv 定义
        Q = _source_at(t) * fv
        # 蒸发封顶：仅在 T→T_boil 时介入，把峰值钳制在蒸气化上限（不影响熔化）
        evap = _evap_sink(T, T_evap_lo, T_boil, c_evap) * fv
        if bcs is None:
            # 原行为：全局弱对流冷却（维持既有测试逐位一致）
            cool = hcool * (T - T_amb) * fv
            return lap + Q - cool - evap
        # 显式 BC：被覆盖处移除全局弱冷却，再叠加对流/热流附加项
        extra_bc, covered, _dir = boundary_terms(
            geometry, bcs, T=T, dx=dx, T_amb=T_amb, hcool=hcool, masks=bc_masks)
        base_cool = hcool * (T - T_amb) * fv * (1.0 - covered)
        return lap + Q - base_cool - evap + extra_bc

    def step(carry, t):
        H, peak, cool_rate, t_above = carry
        T = temperature_of_enthalpy(H, rho=rho, cp=cp, L=L, T_amb=T_amb,
                                    T_sol=T_sol, T_liq=T_liq)
        k1 = rhs(H, t)
        H1 = H + dt * k1
        k2 = rhs(H1, t)
        Hn = H + 0.5 * dt * (k1 + k2)                  # SSP-RK2
        # Dirichlet 边界：把选定面焓强制覆盖为固定温度对应的焓（平滑、可微）
        for (m, Hf) in dirichlet_override:
            Hn = jnp.where(m, Hf, Hn)
        Tn = temperature_of_enthalpy(Hn, rho=rho, cp=cp, L=L, T_amb=T_amb,
                                     T_sol=T_sol, T_liq=T_liq)
        peak = jnp.maximum(peak, Tn)
        cool_rate = jnp.maximum(cool_rate, jnp.maximum(T - Tn, 0.0) / dt)
        t_above = t_above + dt * (Tn > T_sol)
        return (Hn, peak, cool_rate, t_above), Tn

    init = (H0, jnp.zeros_like(mask), jnp.zeros_like(mask), jnp.zeros_like(mask))
    # 扫描动画帧（opt-in）：默认关闭，零额外开销与零契约变化；
    # 开启时从 n_steps 帧等间隔抽取 ≤ n_frames 帧，挂到 ThermalHistory。
    record_frames = bool(p.get("record_frames", False))
    n_frames = int(p.get("n_animation_frames", 16))
    if record_frames:
        scan_out, frames = jax.lax.scan(step, init, jnp.arange(n_steps))
        stride = max(1, n_steps // max(1, n_frames))
        keep = (jnp.arange(n_steps) % stride) == 0
        evolution = frames[keep]                       # (n_kept, *gshape)
    else:
        scan_out, _ = jax.lax.scan(
            lambda c, t: (step(c, t)[0], None), init, jnp.arange(n_steps))
        evolution = None
    Hf, peak_T, cool_rate, t_above = scan_out
    Tf = temperature_of_enthalpy(Hf, rho=rho, cp=cp, L=L, T_amb=T_amb,
                                 T_sol=T_sol, T_liq=T_liq)

    # 凝固前沿 G（逐体素，可作空间场）与 凝固速率 R
    grads = jnp.gradient(Tf, dx)
    G = jnp.sqrt(sum(g ** 2 for g in grads))
    R_grid = cool_rate / jnp.maximum(G, 1e-3)
    # solidification_rate 与 thermal.history 契约保持一致：工艺级标量（熔池平均 R），
    # 不进体素浏览器（由 scalar_summary 以 float() 汇总）；逐体素 G/R 可由
    # cooling_rate / thermal_gradient 两网格场自行派生。
    R_scalar = jnp.mean(R_grid)

    return ThermalHistory(
        peak_temperature=jnp.asarray(peak_T),
        cooling_rate=jnp.asarray(cool_rate),
        thermal_gradient=jnp.asarray(G),
        solidification_rate=jnp.asarray(R_scalar),
        time_above_melt=jnp.asarray(t_above),
        final_temperature=jnp.asarray(Tf),
        temperature_evolution=evolution,
        spacing=spacing,
        dim=geometry.dim,
    )
