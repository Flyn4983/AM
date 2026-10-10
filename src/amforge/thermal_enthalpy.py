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
  再整条右端项除以热容权重 ``cap_w = max(fv, CUT_CAPACITY_FLOOR)``——等价于把 ``H``
  读作**单位材料体积**焓：移走的**能量**逐项与旧口径相同（实体内 fv=1 ⇒ cap_w=1 ⇒ 逐位
  不变），差别只在切割单元不再背整格热容。刀锋层（fv=0.5）的封顶与冷却项因此**刚度翻倍**，
  峰值可能被封顶钳住而对表面对流不敏感（实测见 ``am_t27_convection_peak_probe.log``）。
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

from amforge.beam import inplane_integral, planar_decay
from amforge.core.contracts import (
    PartGeometry,
    ProcessPlan,
    ThermalHistory,
    solid_mask,
    solid_weight,
)
from amforge.core.registry import register_solver
from amforge.materials import get_material
from amforge.scan_program import gate_from_power, require_program


# ---------------------------------------------------------------------------
# cut-cell 热容（#23）
# ---------------------------------------------------------------------------
#: 焓态量 `H` 是**单位材料体积**的焓（体素内实有材料的热容＝`fv·dx³`），故扩散项要除回
#: `fv`；`fv→0` 的单元必须兜底，否则除零。下限取 **0.5** 而非更小：实测
#: `fv ≥ 0.5 ⟺ sdf ≤ 0`（`am_t23_floor_shell_probe.log` P1，1154/4610 个 `fv≤0.5` 的实体
#: 单元全部出现在 δ=0 刀锋层，δ=0.25·dx 档为 **0** 个）⇒ 该下限**永不在实体内部生效**，
#: 只兜底掩膜外的切割壳与空白单元，因此它不是标定旋钮、也不作为 `params` 暴露。
#:
#: 稳定性推论（不抬 n_steps 的依据）：`lap/fv` 把最坏单元的有效扩散数放大 `1/fv_min`，
#: 而实体内 `fv_min = 0.5` ⇒ 缺省 `cfl=0.35` 下最坏有效数 = `0.35/0.5 = 0.7 ≤ 1`
#: （SSP-RK2 绝对上限），故**不改变**已登记的时间步数与吞吐分母；越界由
#: :func:`solve_enthalpy_thermal` 里的显式校核报错，而不是静默发散。
CUT_CAPACITY_FLOOR = 0.5


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
    """单时刻体积热源：可分离 3D 高斯 Q = Q0·exp(−ρ²_xy/(2σ²))·exp(−z²/(2 dp²))，
    其中面内 σ = 输入半径的一半（契约口径＝**1/e² 半径**，唯一出处见 `amforge.beam`）。

    面内与轴向各自单调衰减（恒 ≤ Q0，无放大项），对任意几何（含厚 z 方向）
    均稳定；∫Q dV = Q0·(π r²/2)·dp√(2π) = ηP（与 Q0 归一化一致）。dim>=3 走
    面内+轴向双高斯；dim==2 退化为纯面内热斑（仍为数值热固结）。

    历史（#22）：这里曾写成 exp(−ρ²/r²)，即把输入数当 **1/e** 半径用，面内宽度比
    `_cell_integrated_source`（本模块的**缺省**档）与 diffmech 的 1/e² 写法宽 √2
    （面积宽 2×）⇒ 同一份工艺参数在两个求解器里不是同一束光。实测证据：
    `docs/evidence/2026-10-08/am_t4_spot_probe_pre.log`。
    """
    planar = cell_centers[..., :2] - positions_t[:2]
    planar2 = jnp.sum(planar ** 2, axis=-1)
    if cell_centers.shape[-1] >= 3:
        z2 = (cell_centers[..., 2] - positions_t[2]) ** 2
        decay = planar_decay(planar2, r) * \
                jnp.exp(-z2 / jnp.maximum(2.0 * dp * dp, 1e-18))
    else:
        decay = planar_decay(planar2, r)
    return Q0 * decay


def _cell_integrated_source(position_t, cell_centers, power, r, dp, dx):
    """单元体积分热源：Q_i = power · frac_i / dV，frac_i 为高斯在单元 i 上的解析积分。

    与 ``_moving_source``（中点取值）用**同一个空间形状**——面内
    ``exp(−2ρ²/r²)``（σ=r/2 的高斯，契约的 1/e² 口径）+ 轴向 ``exp(−z²/(2·dp²))``（σ=dp）——
    但每个单元截获的份额按 erf 差值精确积分：

        frac_axis = ½ [erf((c+dx/2 − p)/σ) − erf((c−dx/2 − p)/σ)]

    注意上式的 `σ` 是代码里的 `s_in`，它对应高斯标准差的 **√2 倍**：
    `exp(−ρ²/s_in²) = exp(−ρ²/(2σ_std²))` ⇒ `σ_std = s_in/√2 = r/2`。旧文字曾把这个
    写成「等效 σ=r/√2」，与本函数的**实际宽度**差 √2（也与自己声称的"和
    `_moving_source` 同形状"矛盾）；#22 用二阶矩实测把它对齐到行为上。

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

    专供**扫描足迹/路径长**推导，不用严格的实体掩膜 ``solid_mask``（``sdf < 0`` 加
    ulp 容差）：后者会把体素中心恰好落在设计边界面（``sdf == 0``）上的一圈剔掉，
    于是同一零件在不同 dx 下对齐情况不同 → 层数/道数/路径长随网格抖动（实测试片
    四档 path=3.267/4.775/5.012/5.131mm，极差 36%）。工艺配方是**设计几何**的属性，
    不该由体素对齐决定；扩散算符用 ``solid_mask`` 判实体、源项用线性 cut 份额
    ``solid_weight`` 加权（即本掩膜 == ``solid_weight(...) > 0``），两者用途不同。
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
                   max_voxel_steps=None, over_budget="raise",
                   fixed_n_steps=None, resolution_policy="strict", program=None):
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
    voxel-step 量。预算有两把尺——``max_steps``（步数）与 ``max_voxel_steps``
    （voxel-step，按网格规模反比收紧步数）。预算不够时缺省（``over_budget="raise"``，
    标定档）**报错**并指出出路（粗化网格 / cfl→1 / 加大预算 / 走 §25.8 的 D2、D3），
    因为标定档不该悄悄缩小工艺窗口；``over_budget="pin"``（链条演示档）改为把
    ``scan_speed`` 下界**抬进预算**（窗口变窄，dt 精度不赔）并警告。

    ``fixed_n_steps`` 给定时反过来**由步数定工艺盒**：把速度下界抬到"钉住的
    ``n_steps`` 覆盖得住曝光"，用于调用方自带调度的场合（返回的 ``n_steps`` 即该值）。

    ``program``（#18，opt-in）＝ ``amforge.scan_program.PathProgram`` 实例：路径长改由
    折线总长给出（与 ``hatch_spacing``/``layer_thickness`` 无关 ⇒ 最坏角落与
    ``path_slack`` 对这一支空转，曝光上界＝程序总长/速度下界），与
    ``params["thermal"]["scan_program"]`` 同一个对象。不给则逐项逐位不变。
    """
    from amforge.process import PROCESS_BOUNDS
    if program is not None:
        # 这里必然处于 eager（下面连 ConcretizationTypeError 都要兜），所以路径程序的
        # 数值域前提在**定价之前**一次判掉：给一条自相矛盾的程序（总长 0、参考功率 0
        # 却带非零功率）算出一个 n_steps，只会让它到求解器里变成 NaN 场。
        require_program(program, reference_power=process.laser_power)
    try:
        return _chain_schedule(geometry, process, material=material, cfl=cfl,
                               speed_slack=speed_slack, path_slack=path_slack,
                               max_steps=max_steps, bounds=dict(PROCESS_BOUNDS),
                               max_voxel_steps=max_voxel_steps,
                               over_budget=over_budget,
                               fixed_n_steps=fixed_n_steps,
                               resolution_policy=resolution_policy,
                               program=program)
    except jax.errors.ConcretizationTypeError as e:
        raise ValueError(
            "chain_schedule() 必须在 eager 模式用**具体数值**的名义工艺调用"
            "（它的作用就是在 trace 之前把静态调度钉下来）：请在 jax.grad/jit 之外"
            "对初始工艺调用一次，再把返回值放进 params['thermal']。"
        ) from e


def _chain_schedule(geometry, process, *, material, cfl, speed_slack, path_slack,
                    max_steps, bounds, max_voxel_steps=None, over_budget="raise",
                    fixed_n_steps=None, resolution_policy="strict", program=None):
    mat = get_material(material)
    alpha0 = float(mat.k_solid) / (float(mat.rho_solid) * float(mat.cp_solid) + 1e-12)
    dx = float(jnp.asarray(geometry.spacing, dtype=jnp.float64))
    dt_target = float(_stability_limit_dx2(dx, geometry.dim, alpha0, cfl))
    solid = _footprint(geometry)
    coords = geometry.coords()

    def _m(v):
        return float(jnp.mean(jnp.atleast_1d(jnp.asarray(v, dtype=jnp.float64))))

    path_nom = (float(program.total_length()) if program is not None else
                float(_scan_topology(coords, process, dim=geometry.dim,
                                     solid=solid)[-1]))
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

    if program is None:
        worst = process.replace(hatch_spacing=jnp.asarray(h_worst),
                                layer_thickness=jnp.asarray(lt_worst),
                                beam_radius=jnp.asarray(max(r_nom, r_lo)))
        path_max = float(_scan_topology(coords, worst, dim=geometry.dim,
                                        solid=solid)[-1])
    else:
        # 折线由调用方给定 ⇒ 路径长与 hatch_spacing/layer_thickness **无关**：上面的
        # 最坏角落与 path_slack 对程序分支都是空转，程序总长本身就是上界。
        path_max = path_nom
        warnings.warn(
            f"路径程序档位：曝光上界由折线总长 {path_nom*1e3:.1f}mm 给出，"
            f"hatch_spacing/layer_thickness 不再进入热解（也不再有最坏角落放大），"
            f"返回的工艺子盒里这两项对温度场**梯度恒零**——要优化它们就得回到缺省"
            f"解析 zigzag（去掉 params['scan_program']）。",
            stacklevel=2)
    # 代价上限（§25.7 实测成本律的工程化）：一次正向的墙钟 ≈ n_steps × 体素数。
    # 于是"预算"有两把尺：步数上限 max_steps 与 voxel-step 上限 max_voxel_steps
    # （后者按网格规模反比地收紧步数）。预算不够时**抬扫描速度下界**（缩小可优化的
    # 工艺窗口），而不是放大 dt——放大 dt 会把时间离散的精度一起赔进去。
    nvox = int(jnp.asarray(geometry.sdf).size)
    cap = int(max_steps)
    if max_voxel_steps:
        cap = min(cap, max(1, int(max_voxel_steps) // max(nvox, 1)))
    budget_steps = int(fixed_n_steps) if fixed_n_steps else cap
    v_lo_auto = v_lo
    v_need = path_max / max(budget_steps * dt_target, 1e-30)
    if over_budget == "raise" and not fixed_n_steps and v_need > v_lo_auto:
        n_need = max(1, int(math.ceil(path_max / v_lo_auto / dt_target - 1e-9)))
        n_stable = max(1, int(math.ceil(path_max / v_lo_auto
                                        / (dt_target / max(cfl, 1e-12)) - 1e-9)))
        raise ValueError(
            f"本网格 + 本工艺盒的显式热解需要 n_steps={n_need} > 预算 "
            f"{budget_steps}（dx={dx*1e6:.1f}µm, 体素={nvox}, 代价上界="
            f"{n_need*nvox:.2e} voxel-step）。可选出路：① **粗化**网格或缩短单次扫描"
            f"行程；② cfl→1（仍需 n_steps>={n_stable}）；③ 加大 max_steps / "
            f"max_voxel_steps；④ 允许 over_budget='pin'（把窗口压进预算，代价是"
            f"扫得更慢的那部分工艺不可达）；⑤ 改走 §25.8 的 D2（活跃子网格×子循环）"
            f"或 D3（本征应变降阶）。")
    v_lo = max(v_lo, v_need)
    if v_lo >= v_hi:
        raise ValueError(
            f"本网格的显式热解预算内没有可积的工艺窗口：需要 scan_speed ≥ "
            f"{v_lo:.3g} m/s，而设备上界只有 {v_hi:.3g} m/s（dx={dx*1e6:.1f}µm, "
            f"最坏路径长={path_max*1e3:.1f}mm, 步数预算={budget_steps}, "
            f"代价上界={budget_steps*nvox:.2e} voxel-step）。"
            f"出路：粗化网格／加大预算／走 §25.8 的 D2 或 D3。")
    if over_budget == "pin" and not fixed_n_steps and v_lo > v_lo_auto:
        warnings.warn(
            f"演示档预算压缩工艺窗口：scan_speed 下界由 {v_lo_auto:.3g} 抬到 "
            f"{v_lo:.3g} m/s（n_steps≤{budget_steps}, 体素={nvox}, 代价上界="
            f"{budget_steps*nvox:.2e} voxel-step）。比这更慢的扫描在本档跑不动；"
            f"要覆盖它请加大 max_steps/max_voxel_steps 或改走 §25.8 D2/D3。",
            stacklevel=2)
    t_bound = path_max / v_lo
    n_steps = (int(fixed_n_steps) if fixed_n_steps
               else max(1, int(math.ceil(t_bound / dt_target - 1e-9))))

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


def _evap_sink_hk(T, mat, dx):
    """Hertz–Knudsen 蒸发/反冲散热封顶（**无自由标定参数**，#27）。

    ``S(T) = ṁ(T) · L_v / dx``  [W/m³]，其中 ``ṁ = mat.evaporation_flux(T)`` 走
    Clausius–Clapeyron 饱和蒸气压——强度只由材料卡（``T_boil`` / ``latent_vapor`` /
    ``molar_mass``）与accommodation 0.82 决定，**不含可被标定到某个测试上的系数**。

    与经验 ``_evap_sink`` 的两点差异（选型轮 `am_t27_evap_sink_probe.log` 实测）：
    ① 让 `test_convection_removes_energy` 那条 2% 压峰守护在**阈值一字不动**下成立
    （tuned −0.963% → hk −7.553%）；② 在钳位区给反演/优化交付符号正确的梯度
    （∂peak/∂P：tuned −0.060 → hk +0.383）。低温熔化区平滑趋零（V1：HK(T_liq)/Q_peak
    ≈ 2e-5%），不干扰潜热相变。与 `meltpool.py` 的表面蒸发项同源，那里界面用 |∇F|，
    此处把 [W/m²] 摊成 [J/m³/s] 用 1/dx（与 boundary.py 的表面散热约定一致）。
    """
    return mat.evaporation_flux(T) * mat.latent_vapor / dx


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
    # 蒸发封顶的模型选择器（#27）。**缺省 "tuned" ⇒ 绝对数值逐位不变**：默认切换
    # （tuned→hk 抬峰 +14.48%）与 #33 的系数重标定只能对 A3 的 18 道**外部趋势靶**打分，
    # 在无靶时提前切换＝二次作废绝对峰值数（本项目反复登记的"半动口径"陷阱），故留给
    # #10/A3。此处仅提供三臂，供 A3 对照与外部靶核对逐臂选取：
    #   "tuned"＝经验 smoothstep·c_evap（现生产缺省）；"hk"＝无自由参数 Hertz–Knudsen；
    #   "none"＝完全关闭（与 `evap_coeff=0` 等价），供 A3「无蒸发」基准对齐。
    evap_model = str(p.get("evap_model", "tuned")).lower()
    if evap_model not in ("tuned", "hk", "none"):
        raise ValueError(
            f"evap_model 只能是 'tuned'|'hk'|'none'，收到 {evap_model!r}")

    spacing = jnp.asarray(geometry.spacing, dtype=jnp.float64)  # tracer 安全
    dx = spacing
    mask = solid_mask(geometry.sdf).astype(jnp.float64)  # 仅在实体内部加热
    # 线性化固相体积分数（cut-cell 份额）：|SDF|≤dx/2 的表面体素按被平面切出的
    # 份额计。用它加权**源项**（而非二值 mask）可把"界面沉积份额"从 O(dx) 一阶
    # 几何误差降到 O(dx²)——否则同一物理工况在不同网格上沉积的总剂量本身漂移。
    # 口径定义在全平台唯一出处 contracts.solid_weight（体积/面积一律用它）。
    fv = solid_weight(geometry.sdf, dx)
    # cut-cell 热容权重（#23）：H 是**单位材料体积**焓 ⇒ 整条右端项除以实体份额。
    # 分子里的源/冷却/蒸发已按 fv 加权（见下方 rhs），所以除完之后：
    #   实体内（fv=1）  ＝ 改前逐位相同；
    #   刀锋层（fv=0.5）＝ 扩散 ÷0.5、净源项 = fv·q/fv = q（**注入能量不变**，只是
    #     不再让半块材料带整块体素的热容）；
    #   切割壳（0<fv<0.5）由下限兜底，见 CUT_CAPACITY_FLOOR 的推论注释。
    cap_w = jnp.maximum(fv, CUT_CAPACITY_FLOOR)
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
        # 热源强度归一化：∫Q dV = Q0·(π r²/2)·dp·√(2π) = ηP（连续意义下；离散中点取值
        # 不守恒 → 靠下面两旋钮补）。面内积分因子的唯一出处＝`beam.inplane_integral`（#22）。
        Q0 = eta * P / (inplane_integral(r_src) * math.sqrt(2.0 * math.pi) * dp + 1e-18)
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

    # 路径程序（#18，**opt-in**）：``params["thermal"]["scan_program"]`` 不给就是 None，
    # 下面三处覆盖（总长/位置/功率门控）全部不执行，缺省档逐位不变。
    program = p.get("scan_program")
    if program is not None:
        require_program(program, reference_power=P)

    (n_layers, n_lines, _x_min, _y_min, _z_min, _x_ext, _y_ext, _z_ext,
     _spacing_y, path_length) = _scan_topology(coords, process, dim=geometry.dim,
                                               solid=_footprint(geometry))
    if program is not None:
        # **时间映射不被取代**：曝光时长照样 = 路径长/扫描速度，只是路径长改由折线
        # 总长给出 ⇒ 归一化弧长 g=(t+0.5)/n_steps 与缺省档同一个中点口径，
        # |∂输出/∂scan_speed| 因此仍非零（test_enthalpy_thermal.py:173 的断言不破）。
        path_length = program.total_length()

    alpha0 = k / (rho * cp + 1e-12)
    cfl = float(p.get("cfl", 0.35))
    dt_rk2 = _stability_limit_dx2(dx, geometry.dim, alpha0, 1.0)   # SSP-RK2 绝对上限
    dt_target = dt_rk2 * cfl                                       # 精度目标
    # #23 的稳定性推论校核：`lap/cap_w` 把最坏单元的有效扩散数放大 `1/fv_min`（实体内），
    # 而实体内 `fv_min = 0.5`（刀锋层）⇒ 缺省 `cfl=0.35` 下最坏有效数 `0.35/0.5 = 0.7 ≤ 1`
    # ⇒ **不抬 n_steps**、已登记的步数与吞吐分母不变。调用方若把 cfl 抬到让有效数 >1，
    # 那就是静默发散（会被蒸发封顶伪装成可行结果），此处直接报错。
    _fv_min_solid = jnp.min(jnp.where(mask > 0.5, fv, 1.0))
    cfl_eff = _concrete(cfl / jnp.maximum(_fv_min_solid, CUT_CAPACITY_FLOOR))
    if cfl_eff is not None and cfl_eff > 1.0:
        raise ValueError(
            f"cfl={cfl} 在 cut-cell 热容下等效扩散数 {cfl_eff:.3f} 超过 SSP-RK2 绝对上限 1.0："
            f"实体内最坏份额 fv_min={_concrete(_fv_min_solid):.3f}（刀锋层=0.5）。"
            f"请把 cfl 降到 ≤ {1.0 * _concrete(_fv_min_solid):.3f}，或提高 params['n_steps']。")

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
    if program is None:
        positions, _ = _build_scan_positions(coords, process, dx, n_steps=n_steps,
                                             dim=geometry.dim,
                                             solid=_footprint(geometry))
        power_gate = None
    else:
        # 折线按**同一个** g=(t+0.5)/n_steps 口径给出 (n_steps, 3) 位置与门控；门控是
        # 比值 power(t)/laser_power ⇒ 常功率列逐位＝1.0（见 scan_program 第 2 条口径）。
        positions, _pw_col = program.sample_steps(n_steps)
        power_gate = gate_from_power(_pw_col, P)
    dt = t_exposure / jnp.maximum(n_steps, 1)

    # ``power_gate is None`` 是 Python 静态判断（不是 jnp.where）：缺省分支的 jaxpr 里
    # 一个乘法都不会出现 ⇒ 默认档输出与接入前逐位相同，而门控只在程序分支乘入功率。
    if source_model == "integrated":
        def _source_at(t):
            return _cell_integrated_source(
                positions[t], coords,
                laser_power if power_gate is None else laser_power * power_gate[t],
                r_src, dp, dx)
    else:
        def _source_at(t):
            return _moving_source(
                positions[t], coords,
                Q0 if power_gate is None else Q0 * power_gate[t], r_src, dp)

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
        # 实测（2026-10-09 09:37:49 在**提交树** head=c34a2e1 dirty=0 上重跑同一夹具四档
        # dx=100/50/25/12.5µm × 两个源模型，1.2×0.6×0.4mm、600W/0.8m·s⁻¹、r=100µm、
        # `suggest_n_steps` 定步数（最细档 ns=4121），
        # docs/evidence/2026-10-09/am_t35_dxr_rerun.log）：dx=r 档横向仅 2 个体素跨过 1/e² 光斑，
        # 相对最细档 dx=r/8=12.5µm 的偏低幅度——
        #   integrated：峰值低 13.45%（2378.3 vs 2747.8K）、熔宽低 40.00%（Ly 0.300 vs 0.500mm）、
        #               熔体积小 60.99%（0.0340 vs 0.0872mm³；fv 加权口径 60.75%）。
        #   point    ：峰值低 4.08%（2836.9 vs 2957.5K）、熔宽低 20.00%（Ly 0.400 vs 0.500mm）、
        #               熔体积小 30.90%（0.1110 vs 0.1606mm³；fv 加权口径 30.39%）。
        # ⇒ 两个源的峰值与形态都不可用于标定；point 源对欠分辨**更不敏感**（熔宽 20.00% vs 40.00%），
        #   但两源的绝对量级不同轨（该件 C6：point 四档峰值相对旧件整体 +33~36%，integrated +9~12%，
        #   本轮不指认来源）⇒ 任何一侧都不可沿用另一侧或更早那张表的绝对值。
        # ⚠ 本段文案已**过期两次**：2026-10-06 写"峰值只低 1.1%、熔宽低 45%、熔体积小 36%"（那是
        #   #19 之前 `sdf<0` 严格掩膜口径，该口径连四档峰值本身都带 O(dx) 的域热容误差，见 §26.16
        #   与任务 #23）；2026-10-07 换成 15.6%/46.7%/52.8%（point 35.1%/56.9%），出自
        #   `docs/evidence/2026-10-07/am_t2_a0_conv_rerun.log`——该件 TREE 自报 head=af9fc89
        #   **dirty=30**，即那组数**不对应任何提交**、无法在提交树上复现（本轮只登记"基线不可复现"，
        #   不指认差值来自 af9fc89..HEAD 里哪一笔触及 src 的提交；候选集与逐档对照表都在上述
        #   10-09 件的 C6 段）。⇒ 自本行起**用户可见的运行时文本不再嵌配方实测数**：会腐烂的
        #   百分比只留在注释与证据件，警告串只给几何事实＋指向该件。
        # 能量守恒不受影响（Σfrac=1 与 dx 无关），故只警告不报错；但取熔池形态
        # 或峰值做标定/判据时必须 dx≤r/2。
        warnings.warn(
            f"体素 dx={dx_c*1e6:.1f}µm 大于光束半径 r={r_c*1e6:.1f}µm：横向仅约 "
            f"{2*r_c/dx_c:.1f} 个体素跨过 1/e² 光斑。能量守恒（Σfrac=1 与 dx 无关）但"
            f"**熔池形态与峰值不可信**，不得用于标定或形态判据；定量熔池形态请取 dx≤r/2。"
            f"跨 dx 的实测偏低幅度见 docs/evidence/2026-10-09/am_t35_dxr_rerun.log。",
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
        # 蒸发封顶：按 evap_model 选三臂（缺省 tuned ⇒ 与改前逐位相同，见上方选择器注释）
        if evap_model == "hk":
            evap = _evap_sink_hk(T, mat, dx) * fv
        elif evap_model == "none":
            evap = jnp.zeros_like(fv)
        else:
            evap = _evap_sink(T, T_evap_lo, T_boil, c_evap) * fv
        # —— #23：整条右端项除以 cut-cell 热容权重 cap_w（推导见 CUT_CAPACITY_FLOOR）。
        # 物理内容：`fv·dx³·dH/dt = dx³·lap + fv·dx³·(q − loss)`，即**热容按实有材料计**、
        # 而**注入剂量不变**（改前的错处是左边用了整格热容：δ=0 档计算域质量比剂量多
        # +24.574/+12.446/+6.243% @ dx=50/25/12.5µm，`am_t2_domain_mass.log`）。
        # 空白单元（fv=0）分子恒为 0（面全闭 ⇒ lap=0；fv·项=0），除以 0.5 仍是 0，不产生 NaN。
        # 切割壳（0<fv<0.5）被二值面掩膜隔断扩散，其壳体温度不由本项的物理决定——
        # 已知近似，登记见模块 docstring 与开发日志。峰值落点**不是**普适的 fv=1：A0 试片
        # 实测 argmax 落在 fv=1 的实体单元（`am_t23_floor_shell_probe.log` L1），而 0.4mm
        # 立方 / dx=80µm 的对流夹具实测落在 **fv=0.5 的刀锋面**——那里封顶项被 cap_w 放大
        # 2 倍，于是峰值由封顶刚度决定、表面对流压不动它（`am_t27_convection_peak_probe.log`）。
        if bcs is None:
            # 全局弱对流冷却仍按 fv 加权：除完 cap_w 后 fv·cool/fv = cool，移走的**能量**与
            # 改前逐项相同，只是不再让半块材料背整格热容（实体内 fv=1 处逐位不变）。
            cool = hcool * (T - T_amb) * fv
            return (lap + Q - cool - evap) / cap_w
        # 显式 BC：被覆盖处移除全局弱冷却，再叠加对流/热流附加项
        extra_bc, covered, _dir = boundary_terms(
            geometry, bcs, T=T, dx=dx, T_amb=T_amb, hcool=hcool, masks=bc_masks)
        base_cool = hcool * (T - T_amb) * fv * (1.0 - covered)
        return (lap + Q - base_cool - evap + extra_bc) / cap_w

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
    # 纯 √(Σg²) 在 ∇T≡0 处导数是 0/0 ⇒ 反传 NaN（正向却正常）。等温区/域外体素必然
    # 出现 ∇T≡0，故按 meltpool._norm 同纪律在根号内加 ε；对 G≫√ε 的正向值无影响。
    G = jnp.sqrt(sum(g ** 2 for g in grads) + 1e-30)
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
