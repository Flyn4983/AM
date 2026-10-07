"""P2-① 测试：enthalpy 法相变热解的相变物理与可微性。

守护点：
1. 液相关联 f(T) 平滑单调、糊状区 T 被夹在 [T_solidus, T_liquidus]。
2. 焓-温逆映射 T(H) 高精度（二分法，误差≈0）。
3. 潜热吸收：同热源下「含潜热」峰值温度显著低于「无潜热」（能量转入相变）。
4. 糊状区（固液共存）在不同 H 样本上真实存在。
5. jax.grad 穿透整链无 NaN/Inf（可微闭环可接高保真热解）。
6. 接入 ForgeCore 9 契约链：select={'thermal':'thermal.enthalpy'} 后
   自动布线包含该求解器，且产出有限 ThermalHistory。
"""
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from amforge.geometry import from_sdf_fn
from amforge.core.contracts import ProcessPlan, solid_mask
from amforge.materials import get_material
from amforge.thermal_enthalpy import (
    liquid_fraction, d_liquid_fraction, temperature_of_enthalpy,
    solve_enthalpy_thermal, suggest_n_steps, _build_scan_positions,
    _moving_source, _div_alpha_grad, effective_diffusivity,
    _cell_integrated_source, _scan_topology, _footprint,
)


# ---------------------------------------------------------------------------
# 公共夹具
# ---------------------------------------------------------------------------
def _part(dx=80e-6, half=0.24e-3):
    return from_sdf_fn(
        lambda x: jnp.linalg.norm(x, axis=-1) - half,
        bounds=[(-half, half)] * 3, spacing=dx, name="box",
    )


def _plan(power=1200.0):
    return ProcessPlan.uniform(3, modality="SLM", laser_power=power,
                               scan_speed=0.8, layer_thickness=80e-6,
                               hatch_spacing=120e-6, beam_radius=100e-6,
                               absorption=0.45, preheat_temp=400.0)


# ---------------------------------------------------------------------------
# 1. 液相关联 f(T) 平滑单调，糊状区温度被夹在固液线之间
# ---------------------------------------------------------------------------
def test_liquid_fraction_monotone_and_mushy_bounded():
    m = get_material("316L")
    T = jnp.linspace(300.0, 2200.0, 400)
    f = liquid_fraction(T, m.T_solidus, m.T_liquidus)
    fp = d_liquid_fraction(T, m.T_solidus, m.T_liquidus)

    # 单调、范围 [0,1]
    assert float(jnp.min(f)) >= -1e-12 and float(jnp.max(f)) <= 1.0 + 1e-12
    df = jnp.diff(f)
    assert float(jnp.min(df)) >= -1e-12, "f(T) 应单调不减"
    assert float(jnp.min(fp)) >= -1e-12, "df/dT 应 ≥ 0"

    # 糊状区内 (0<f<1) 的 T 必须夹在 [Ts, Tl]
    mushy = (f > 0.01) & (f < 0.99)
    assert bool(jnp.any(mushy)), "应存在固液共存（糊状）区"
    assert float(jnp.min(T[mushy])) >= m.T_solidus - 1.0
    assert float(jnp.max(T[mushy])) <= m.T_liquidus + 1.0


# ---------------------------------------------------------------------------
# 2. 焓-温逆映射 T(H) 高精度
# ---------------------------------------------------------------------------
def test_temperature_of_enthalpy_inverts():
    m = get_material("316L")
    rho, cp, L = m.rho_solid, m.cp_solid, m.latent_fusion
    Tamb, Ts, Tl = m.T_ambient, m.T_solidus, m.T_liquidus

    T = jnp.linspace(Tamb, 2200.0, 500)
    H = rho * cp * (T - Tamb) + rho * L * liquid_fraction(T, Ts, Tl)
    Tback = temperature_of_enthalpy(H, rho=rho, cp=cp, L=L, T_amb=Tamb,
                                    T_sol=Ts, T_liq=Tl)
    err = float(jnp.max(jnp.abs(T - Tback)))
    assert err < 1.0, f"T(H) 逆映射误差应≈0，实际 {err} K"


# ---------------------------------------------------------------------------
# 3. 潜热吸收：同热源下含潜热峰值温度更低
# ---------------------------------------------------------------------------
def test_latent_heat_lowers_peak_temperature():
    m = get_material("316L")
    rho, cp, k = m.rho_solid, m.cp_solid, m.k_solid
    L = m.latent_fusion
    Ts, Tl, Tamb = m.T_solidus, m.T_liquidus, m.T_ambient
    dx = 80e-6
    part = _part(dx=dx)
    plan = _plan()
    coords = part.coords()
    mask = (part.sdf < 0.0).astype(jnp.float64)

    P = float(jnp.mean(plan.laser_power))
    r = float(jnp.mean(plan.beam_radius))
    eta = float(jnp.mean(plan.absorption))
    dp = max(r * 1.2, dx)
    Q0 = eta * P / (3.14159265 * r * r * (2.0 * 3.14159265) ** 0.5 * dp)
    alpha0 = k / (rho * cp)
    dt = 0.35 * dx * dx / (2.0 * 3 * alpha0)
    pos, _ = _build_scan_positions(coords, plan, dx, n_steps=80, dim=part.dim)
    ns = int(pos.shape[0])
    H0 = jnp.zeros_like(mask)

    def rhs(H, t, useL):
        T = temperature_of_enthalpy(H, rho=rho, cp=cp, L=(L if useL else 0.0),
                                    T_amb=Tamb, T_sol=Ts, T_liq=Tl)
        a = effective_diffusivity(T, k=k, rho=rho, cp=cp, L=(L if useL else 0.0),
                                  T_sol=Ts, T_liq=Tl)
        Q = _moving_source(pos[t], coords, Q0, r, dp) * mask
        return _div_alpha_grad(H, a, dx) + Q - 0.5 * (T - Tamb) * mask

    def run(useL):
        def body(c, t):
            H, peak = c
            T = temperature_of_enthalpy(H, rho=rho, cp=cp, L=(L if useL else 0.0),
                                        T_amb=Tamb, T_sol=Ts, T_liq=Tl)
            k1 = rhs(H, t, useL)
            H1 = H + dt * k1
            k2 = rhs(H1, t, useL)
            Hn = H + 0.5 * dt * (k1 + k2)
            Tn = temperature_of_enthalpy(Hn, rho=rho, cp=cp, L=(L if useL else 0.0),
                                         T_amb=Tamb, T_sol=Ts, T_liq=Tl)
            return (Hn, jnp.maximum(peak, Tn)), None
        (_, peak), _ = jax.lax.scan(body, (H0, jnp.zeros_like(mask)), jnp.arange(ns))
        return peak

    peak_with_L = float(jnp.max(run(True)))
    peak_no_L = float(jnp.max(run(False)))

    # 必须真正熔化（峰值越过失相线），潜热才被触发
    assert peak_with_L > Tl, "含潜热情形应熔化"
    assert peak_no_L > peak_with_L + 10.0, \
        f"潜热应吸收能量使峰值更低：with_L={peak_with_L:.1f} no_L={peak_no_L:.1f}"


# ---------------------------------------------------------------------------
# 4. 封装求解器产出有限、且发生过熔化（time_above_melt>0）
# ---------------------------------------------------------------------------
def test_solver_produces_finite_thermal_history():
    part = _part()
    plan = _plan()
    # A0 之后 dt = 曝光时长/n_steps，步数不足会直接发散；不再手写小步数，
    # 交给求解器按 CFL 自动推导（同时也守护自动推导这条路径）。
    th = solve_enthalpy_thermal(geometry=part, process=plan,
                                params={"material": "316L"})
    for name in ("peak_temperature", "final_temperature", "cooling_rate",
                 "thermal_gradient", "solidification_rate", "time_above_melt"):
        leaf = getattr(th, name)
        assert jnp.all(jnp.isfinite(leaf)), f"{name} 应全部有限"
    assert float(jnp.max(th.peak_temperature)) > get_material("316L").T_liquidus, \
        "峰值温度应越过失相线（发生熔化）"
    assert float(jnp.sum(th.time_above_melt)) > 0.0, \
        "应存在 T > T_solidus 的体素（熔化/重熔）"


# ---------------------------------------------------------------------------
# 5. jax.grad 穿透整链无 NaN/Inf（可微优化可接高保真热解）
# ---------------------------------------------------------------------------
def test_gradient_through_enthalpy_is_finite():
    part = _part(dx=80e-6, half=0.20e-3)   # 更小网格加速
    plan = _plan(power=1000.0)
    speed = jnp.array(0.8, dtype=jnp.float64)
    # 时间步数是**静态形状**，不能从 tracer 推出：在 trace 之外按 eager 工艺量
    # 求出 CFL 步数再传入（这正是 suggest_n_steps 存在的原因）。
    ns = suggest_n_steps(part, plan)

    def loss(s):
        p = plan.replace(scan_speed=jnp.atleast_1d(s))
        th = solve_enthalpy_thermal(geometry=part, process=p,
                                    params={"material": "316L", "n_steps": ns})
        return jnp.sum(th.peak_temperature)

    g = jax.grad(loss)(speed)
    assert jnp.isfinite(g), f"梯度应有限，实际 {g}"
    assert float(jnp.abs(g)) > 0.0, "梯度不应为零（扫描速度确实影响热历史）"


# ---------------------------------------------------------------------------
# 6. 接入 ForgeCore 9 契约链（select 显式切换高保真热解）
# ---------------------------------------------------------------------------
def test_registered_and_selectable_in_forgecore():
    import amforge.forge_adapter  # 触发 AM 求解器注册进 ForgeCore
    from forgecore.registry import REGISTRY

    assert "thermal.enthalpy" in REGISTRY.list_names()

    # 自动布线（target=verdict，小写端口名）在显式 select 后必须包含该求解器
    specs = REGISTRY.auto("verdict", modality="SLM",
                          select={"thermal": "thermal.enthalpy"})
    names = [s.name for s in specs]
    assert "thermal.enthalpy" in names
    assert "thermal.history" not in names, "select 应替换为高保真热解"

    # 单独跑通 thermal.enthalpy，产出有限 ThermalHistory
    part = _part(dx=100e-6, half=0.18e-3)
    plan = _plan(power=1000.0)
    out = REGISTRY.get("thermal.enthalpy").call(
        {"geometry": part, "process": plan},
        {"material": "316L"},                       # 步数由 CFL 自动推导
    )
    assert jnp.all(jnp.isfinite(out.peak_temperature))
    assert jnp.all(jnp.isfinite(out.final_temperature))


# ===========================================================================
# A0（2026-10-06）：能量/分辨率保真回归守护
#   阻塞缺陷：旧 dt=min(稳定限, total·dx/(v·n_steps)) 使总曝光 ∝ n_steps·dx²，
#   加密网格/改步数会**改写注入能量**（实测表现为"越加密越冷、全案不熔化"）。
#   下面四条守住修复后的四条不变量。阈值全部来自同轮实测（见开发日志 §25）。
# ===========================================================================
def _coupon(dx, ext=(1.2e-3, 0.6e-3, 0.4e-3)):
    bx, by, bz = ext
    return from_sdf_fn(
        lambda x: jnp.maximum(jnp.maximum(jnp.abs(x[..., 0]) - bx / 2,
                                          jnp.abs(x[..., 1]) - by / 2),
                              jnp.abs(x[..., 2]) - bz / 2),
        bounds=[(-bx / 2, bx / 2), (-by / 2, by / 2), (-bz / 2, bz / 2)],
        spacing=dx, name="coupon",
    )


def _coupon_plan(ext=(1.2e-3, 0.6e-3, 0.4e-3), power=600.0, radius=100e-6):
    return ProcessPlan.uniform(1, modality="SLM", laser_power=power, scan_speed=0.8,
                               layer_thickness=ext[2], hatch_spacing=1.4 * radius,
                               beam_radius=radius, absorption=0.45,
                               preheat_temp=400.0)


def test_cell_integrated_source_conserves_dose():
    """单元体积分源 Σfrac=1 到机器精度，且**与 dx/r 无关**（含 r<dx 的欠分辨档）。"""
    n = 64
    for r_um, dx_um in [(100., 25.), (100., 100.), (100., 200.), (25., 100.)]:
        r, dx = r_um * 1e-6, dx_um * 1e-6
        ax = jnp.linspace(-(n // 2) * dx, -(n // 2) * dx + (n - 1) * dx, n)
        cen = jnp.stack(jnp.meshgrid(ax, ax, ax, indexing="ij"), axis=-1)
        frac = float(jnp.sum(_cell_integrated_source(
            jnp.array([0.37 * dx, -0.11 * dx, 0.29 * dx]), cen, 1.0, r, 1.2 * r, dx))
            * dx ** 3)
        assert abs(frac - 1.0) < 1e-12, f"r={r_um}µm dx={dx_um}µm Σfrac={frac!r}"


def test_scan_recipe_is_grid_independent():
    """扫描配方（层数/道数/路径长）属设计几何，不随体素对齐抖动。"""
    pl = _coupon_plan()
    ref = None
    for dx_um in (100., 50., 25.):
        dx = dx_um * 1e-6
        g = _coupon(dx)
        topo = _scan_topology(g.coords(), pl, dim=3, solid=_footprint(g))
        got = (float(topo[0]), float(topo[1]), float(topo[-1]))
        if ref is None:
            ref = got
        assert got == pytest.approx(ref, rel=1e-12, abs=1e-12), \
            f"dx={dx_um}µm 拓扑={got} 与 {ref} 不一致（路径长被网格改写）"
    assert ref[2] == pytest.approx(5.25e-3, rel=1e-12)   # 4 道 × 1.2mm + 3 × 0.15mm


def test_unstable_or_unresolved_schedule_is_rejected():
    """三道护栏：步数低于稳定下限 / 体素装不下光斑 / 未知源模型 —— 都必须响亮失败。"""
    g, pl = _coupon(100e-6), _coupon_plan()
    with pytest.raises(ValueError, match="n_steps"):
        solve_enthalpy_thermal(geometry=g, process=pl,
                               params={"material": "316L", "n_steps": 2})
    # dx=250µm > 2r=200µm：热源欠采样，峰值无物理意义
    with pytest.raises(ValueError, match="欠采样"):
        solve_enthalpy_thermal(geometry=_coupon(250e-6), process=pl,
                               params={"material": "316L"})
    with pytest.raises(ValueError, match="source_model"):
        solve_enthalpy_thermal(geometry=g, process=pl,
                               params={"material": "316L", "source_model": "bogus",
                                       "n_steps": suggest_n_steps(g, pl)})


def test_preheat_temperature_enters_initial_condition():
    """工艺预热温度进 IC（此前 preheat_temp 被本模块完全忽略）。"""
    ext = (0.2e-3, 0.2e-3, 0.2e-3)
    g = _coupon(100e-6, ext)
    solid = jnp.asarray(g.sdf) < -1e-9
    out = solve_enthalpy_thermal(
        geometry=g, process=_coupon_plan(ext, power=1e-9),
        params={"material": "316L", "n_steps": 1})
    # 容差 0.05K：T(H) 是 24 步二分（区间 ~3000K → 分辨率 ~2e-4K）+ 弱冷一步
    assert float(jnp.mean(out.final_temperature[solid])) == pytest.approx(400.0, abs=0.05)
    cold = solve_enthalpy_thermal(
        geometry=g,
        process=ProcessPlan.uniform(1, modality="SLM", laser_power=1e-9,
                                    scan_speed=0.8, layer_thickness=ext[2],
                                    hatch_spacing=1.4e-4, beam_radius=100e-6,
                                    absorption=0.45, preheat_temp=300.0),
        params={"material": "316L", "n_steps": 1})
    assert float(jnp.mean(cold.final_temperature[solid])) == pytest.approx(300.0, abs=0.05)


def test_meltpool_converges_across_beam_resolving_grids():
    """A0 的核心可验证承诺：**能分辨光斑的两档网格**给出同一熔池。

    判据（5%）**不放宽**；本测试自 #19 起为**已知红灯**，下面是它红的确切数字与归属，
    不是解释性辩护。取数条件：试片 1.2×0.6×0.4mm、600W/0.8m·s⁻¹/r=100µm/η=0.45、
    integrated 源、步数由 CFL 自动推导、CPU 钉住（CUDA_VISIBLE_DEVICES=""）。
    证据：docs/evidence/2026-10-07/am_t2_a0_test_mirror.log、am_t2_a0_observable.log、
    am_t2_a0_alignment.log、am_t2_domain_mass.log（HEAD af9fc89 + #19 工作树）。

    (1) 本测试自己的口径（δ=0：试片表面恰落在节点上，即 `_coupon` 的刀锋对齐）
          dx=50µm   峰值 2349.07K  熔体积 0.05187mm³(415 体素)
          dx=25µm   峰值 2478.97K  熔体积 0.05831mm³(3732 体素)
          dx=12.5µm 峰值 2517.98K  熔体积 0.06778mm³
        ⇒ 断言 1 峰值差 5.24%（超出判据 0.24 个百分点，pytest 在此即中止）；断言 2 单独评估为
          熔体积差 11.04%（取自镜像探针 `am_t2_a0_test_mirror.log`，同口径同对齐）⇒ 两条**同红**。
        加密不变冷（断言 3）仍通过。
        2026-10-06 登记的旧数（峰值 2670.3/2617.8K、差 1.96%；体积 0.0687/0.0715mm³、差 3.92%）
        是**旧严格掩膜 `sdf<0`** 下的数，已不可复现；旧绿里含一份口径补偿误差，见 (3)。

    (2) 换指标被跑前登记的 O1 判据否掉（`am_t2_a0_observable.py:90-125`：观测量入选
        ⇔ 对**每一种 δ** 且**每一对相邻档**都 <5%，不许挑对齐、不许挑档对）：
             观测量     50↔25(δ∈{0,.25})  25↔12.5(δ∈{0,.25})  50↔25(δ=0.5)
             峰值 T            5.24%              1.55%            5.15%
             Vm(fv 加权)      10.70%             13.49%            1.13%
             Vn(体素计数)     11.04%             13.97%            1.13%
             ΔH(焓升)          9.74%              5.32%            4.66%
        ⇒ **四项全不入选**：本轮没有可靠的网格收敛观测量，A0 保持红灯，不换指标。

    (3) 已量化的一阶成因（同一轮实测，不是推测）：#19 把实体掩膜改成容差 `sdf<1e-12` 后，
        δ=0 的刀锋面被**整层计入**计算域，而 `thermal_enthalpy` 给部分填充体素配的是
        **满体素热容**（右端项 `fv·dH/dt = lap + fv·Q − fv·cool` 里 `lap` 未除回 fv）。
        于是域热容 / 设计体积 = 1.2695×(50µm) / 1.1298×(25µm) / 1.0637×(12.5µm)，而剂量
        Σfv·dx³ 只有 1.0191 / 1.0048 / 1.0012×——**粗网格凭空多出 24.6%/12.4%/6.2% 的热容**
        （O(dx)、随加密单调消失；旧 `sdf<0` 是另一侧的 0.8785/0.9384/0.9690×，两错反向，
        所以旧档对差被补偿成"看似收敛"）。修法与前置条件登记为任务 **#23**（cut-cell 热容按
        fv 加权 + 计算域取 fv>0 + 重验 CFL/Σfrac 守恒），它是 #10/A3 峰值与熔池形态对比的
        可信度前置。
        ⚠ 照实记边界：这只解释 δ=0 一侧；熔体积 Vm/Vn 在**每一种 δ** 下都超 5%（δ=0.25 的
        50↔25 仍 9.8%），说明还至少有一个机制未被 #23 覆盖，不得宣称 #23 会转绿。

    (4) 对齐抖动量级（同档、三种 δ 的极差，`am_t2_a0_alignment.log`）：峰值 3.46%(50µm)/
        3.36%(25µm)，即与档对差同量级 ⇒ 当前噪声地板不足以把"1.55% vs 5.24%"读成收敛结论。

    dx=100µm(=r) 档不在本判据内：求解器对它发「熔池形态不可信」警告（真实空间欠分辨）。
    """
    ext = (1.2e-3, 0.6e-3, 0.4e-3)
    pl = _coupon_plan(ext)
    out = []
    for dx_um in (50., 25.):
        dx = dx_um * 1e-6
        g = _coupon(dx, ext)
        th = solve_enthalpy_thermal(geometry=g, process=pl,
                                    params={"material": "316L"})
        # 口径：#19 后统一用 solid_mask（实测对本观测量：50µm 档 0 个体素、25µm 档 28 个体素
        # =+0.76% 的差异；全域异或 901/3529 体素为阳性对照，见 am_t2_a0_test_mirror.log）
        melted = (th.peak_temperature > get_material("316L").T_liquidus) & (solid_mask(g.sdf) > 0.5)
        n = int(jnp.sum(melted))
        assert n > 0, f"dx={dx_um}µm 应发生熔化（历史缺陷：加密反而不熔）"
        out.append((float(jnp.max(th.peak_temperature)), n * dx ** 3 * 1e9))
    pk_c, vol_c = out[0]
    pk_f, vol_f = out[1]
    assert abs(pk_c - pk_f) / max(pk_c, pk_f) < 0.05, f"峰值不收敛：{pk_c} vs {pk_f}"
    assert abs(vol_c - vol_f) / max(vol_c, vol_f) < 0.05, f"熔体积不收敛：{vol_c} vs {vol_f}"
    assert pk_f > pk_c - 0.05 * pk_c, "加密网格不应显著变冷"
