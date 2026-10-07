"""高保真 asbuilt 接入 digitaltwin/逆问题 → 可微尺寸优化（功能 2 落地）

验证：
1. 高保真塑性/CPFE 成形（asbuilt.thermomechanical_plastic）能无缝接入 inverse
   的物理前向 simulate()，并继续流入 digitaltwin + verdict；
2. 基于高保真 asbuilt 的尺寸损失对工艺参数可微（jax.grad 有限且非零）；
3. optimize_dimensional 用高保真 asbuilt 直接对归一化工艺向量做 Adam 梯度下降，
   能降低"模拟后构型 vs 理想构型"的尺寸偏差（残余应力/应变 + 服役风险）；
4. 默认 asbuilt_solver 仍是降阶 buildup（不破坏默认轻量链路）。
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from amforge import geometry as G
from amforge.process import (
    heuristic_plan, denormalize_process, normalize_process, PROCESS_BOUNDS,
)
from amforge.core.contracts import PartGeometry, ProcessPlan, AsBuiltPart
from amforge.materials import get_material
from amforge.meltpool import solve_meltpool_surrogate
from amforge.thermal import solve_thermal_history
from amforge.asbuilt_plastic import solve_asbuilt_plastic
from amforge.inverse import (
    simulate, loss_fn, _dimensional_loss, optimize_dimensional, optimize_shape,
    optimize_geometry_process, geometry_features, mlp_init,
    process_constraint_penalty, process_feasibility, project_process_feasible,
    pareto_optimize_geometry_process, thermal_tier, tier_bounds,
)
from amforge.nsga2 import fast_non_dominated_sort

# 小几何：体素网格约 9^3，FEM 单元约 8^3，保证测试内 FEM 可解（不 OOM）。
_GEO_KW = dict(
    bounds=[(-0.5e-3, 0.5e-3)] * 3,
    spacing=120e-6,
    name="opt-demo",
)
_HIFI_PARAMS = dict(n_grid=8, max_layers=4, n_sub_cp=32, tau_activation=1e-3)
# 轻量 hifi 参数：大幅缩短 CPFE FEM 耗时，用于新增约束/Pareto 测试的回归（正确性
# 不依赖高保真度，仅用于快速验证可微性与可行性）。
_LIGHT = dict(n_grid=6, max_layers=3, n_sub_cp=8, tau_activation=1e-3)
# 极小参数：仅用于 NSGA-II 多目标集成冒烟（验证非支配/可行/权衡），
# 不关心绝对保真度，极致压缩 FEM 耗时。
_TINY = dict(n_grid=4, max_layers=2, n_sub_cp=4, tau_activation=1e-3)
_W = dict(geom=5.0, stress=1.0, strain=0.5, defect=1.0)


def _small_geo():
    return G.from_sdf_fn(
        lambda x: jnp.linalg.norm(x, axis=-1) - 0.4e-3, **_GEO_KW)


def _high_energy_plan(geo):
    """故意偏高能量的工艺（大激光功率/慢扫描）→ 明显尺寸偏差，给优化留出空间。"""
    return ProcessPlan.uniform(
        int(geo.layer_count(40e-6)), modality="SLM",
        laser_power=900.0, scan_speed=0.6,
        layer_thickness=60e-6, hatch_spacing=120e-6,
        beam_radius=60e-6, absorption=0.5, preheat_temp=473.0,
        dwell_time=0.0,
    )


def test_hifi_asbuilt_runs_in_simulate():
    geo = _small_geo()
    plan = heuristic_plan(geo, material="316L", modality="SLM")
    out = simulate(geo, plan, material="316L", params=_HIFI_PARAMS,
                   asbuilt_solver="plastic", constitutive="j2")
    ab = out["asbuilt"]
    assert isinstance(ab, AsBuiltPart)
    for leaf in jax.tree_util.tree_leaves(ab):
        assert jnp.all(jnp.isfinite(leaf)), "高保真 asbuilt 含非有限值"
    # 真实弹塑性收缩/翘曲应产生非零位移
    disp = ab.displacement
    assert jnp.max(jnp.abs(disp)) > 0.0, "高保真 asbuilt 未产生位移"
    # 变形 SDF 应偏离理想 SDF（发生尺寸偏差）
    assert jnp.max(jnp.abs(ab.sdf - geo.sdf)) > 0.0
    # 仍正常流入 digitaltwin + verdict
    assert jnp.isfinite(out["verdict"].strength_safety_factor)


def test_hifi_dimensional_loss_gradient_finite():
    geo = _small_geo()
    z = normalize_process(heuristic_plan(geo, material="316L", modality="SLM"))

    def loss_of_z(zz):
        plan = denormalize_process(zz, n_layers=int(geo.layer_count(40e-6)),
                                  modality="SLM")
        out = simulate(geo, plan, material="316L", params=_HIFI_PARAMS,
                       asbuilt_solver="plastic", constitutive="j2")
        return _dimensional_loss(out, geo, "316L", weights=_W)

    g = jax.grad(loss_of_z)(z)
    assert jnp.all(jnp.isfinite(g)), "高保真尺寸损失梯度含 NaN/Inf"
    assert jnp.max(jnp.abs(g)) > 0.0, "梯度全零（链路未连通工艺参数）"


def test_optimize_reduces_dimensional_loss():
    geo = _small_geo()
    plan_init = _high_energy_plan(geo)
    res = optimize_dimensional(
        geo, process_init=plan_init, material="316L",
        params=_HIFI_PARAMS, n_steps=10, learning_rate=0.08,
        weights=_W, verbose=False,
    )
    hist = res["loss_history"]
    assert len(hist) == 10
    # 优化过程中最优损失应显著低于初始损失（尺寸偏差被压低）
    assert min(hist) < hist[0] - 1e-6, f"尺寸优化未收敛下降: {hist}"
    # 最优工艺仍物理合理（有限且在设备边界内）
    plan = res["final_plan"]
    assert jnp.all(jnp.isfinite(plan.laser_power))
    assert jnp.all(plan.laser_power >= PROCESS_BOUNDS["laser_power"][0])
    assert jnp.all(plan.laser_power <= PROCESS_BOUNDS["laser_power"][1])


def test_default_simulate_still_buildup():
    """回归守卫：默认 asbuilt_solver 仍是降阶 buildup，不触发高保真 FEM。"""
    geo = _small_geo()
    plan = heuristic_plan(geo, material="316L", modality="SLM")
    out = simulate(geo, plan, material="316L", params=dict(n_grid=8))
    ab = out["asbuilt"]
    assert isinstance(ab, AsBuiltPart)
    for leaf in jax.tree_util.tree_leaves(ab):
        assert jnp.all(jnp.isfinite(leaf))


def test_loss_fn_accepts_hifi_switch():
    """loss_fn 透传 asbuilt_solver / constitutive 仍给出有限、可微损失。"""
    geo = _small_geo()
    n_in = int(geometry_features(geo).shape[0])
    rng = jax.random.PRNGKey(0)
    theta = mlp_init(rng, n_in, 8, n_out=len(PROCESS_BOUNDS), n_layers=2)
    loss = loss_fn(theta, geo, material="316L", params=_HIFI_PARAMS,
                   asbuilt_solver="plastic", constitutive="j2",
                   weights=_W)
    assert jnp.isfinite(loss)


# ---------------------------------------------------------------------------
# 几何/形状优化（反变形 / 预补偿）：设计变量从工艺参数升级为名义 SDF 修正场
# ---------------------------------------------------------------------------
def _precompute_thermal(target, *, material="316L", n_grid=8):
    mp = solve_meltpool_surrogate(geometry=target,
                                  process=heuristic_plan(target, material=material,
                                                        modality="SLM"),
                                  params={"material": material, "n_grid": n_grid})
    return solve_thermal_history(geometry=target,
                                 process=heuristic_plan(target, material=material,
                                                        modality="SLM"),
                                 meltpool=mp, params={"material": material})


def _shape_loss_u(uu, target, thermal, *, material="316L", spacing=None,
                  soft_occ_t=None, soft_sum=None):
    """与 optimize_shape 同源的形状损失（供梯度/收敛测试自包含复现）。"""
    spacing = float(target.spacing) if spacing is None else spacing
    soft_occ_t = target.soft_occupancy() if soft_occ_t is None else soft_occ_t
    soft_sum = jnp.maximum(jnp.sum(soft_occ_t), 1e-12) if soft_sum is None else soft_sum
    mat = get_material(material)
    nominal = PartGeometry(sdf=target.sdf + uu * spacing, origin=target.origin,
                           spacing=spacing, dim=target.dim, name="nominal")
    ab = solve_asbuilt_plastic(
        geometry=nominal, thermal=thermal,
        params=dict(material=material, constitutive="j2", max_layers=4,
                    n_sub_cp=32, tau_activation=1e-3, process=None,
                    occupancy_override=nominal.soft_occupancy()))
    dev = jnp.abs(ab.sdf - target.sdf) * soft_occ_t
    geom_dev = jnp.sum(dev) / soft_sum / spacing
    rvm = ab.von_mises_residual()
    stress = jnp.sum(rvm * soft_occ_t) / soft_sum / mat.sigma_y
    return 5.0 * geom_dev + 1.0 * stress


def test_shape_gradient_finite_and_connected():
    """形状损失对名义 SDF 修正场可微且梯度非零（几何->asbuilt 链路连通）。"""
    geo = _small_geo()
    thermal = _precompute_thermal(geo)
    u0 = jnp.zeros(geo.shape)
    g = jax.grad(_shape_loss_u)(u0, geo, thermal)
    assert jnp.all(jnp.isfinite(g)), "形状优化梯度含 NaN/Inf"
    assert jnp.max(jnp.abs(g)) > 0.0, "梯度全零（几何->asbuilt 链路未连通）"


def test_optimize_shape_reduces_dimensional_deviation():
    """optimize_shape 把名义几何预补偿后，as-built 对 target 的尺寸偏差显著下降。

    真实焓法（enthalpy）热学下，SDF 尺寸偏差指标在边界区对修正场 ``u`` 极敏感，
    优化器在过高学习率（早期默认 0.05）下会第一步过冲 ~10× 发散；用稳定 lr=0.01
    后预补偿可真实把尺寸偏差降到初始的 ~50% 以下（真实热学 Landscape 的物理可达区）。
    """
    geo = _small_geo()  # target = 目标 CAD
    res = optimize_shape(
        geo, material="316L", n_steps=24, learning_rate=0.01,
        params=dict(n_grid=8, max_layers=4, n_sub_cp=32, tau_activation=1e-3),
        smoothness=0.02, verbose=False,
    )
    gh = res["geom_history"]
    assert len(gh) == 24
    # 预补偿后尺寸偏差应显著低于初始（delta=0 的 target 自身 as-built）
    assert min(gh) < gh[0] * 0.5, f"形状优化未显著降低尺寸偏差: {gh}"
    # 补偿场与名义几何有限
    assert jnp.all(jnp.isfinite(res["delta"]))
    nom = res["nominal_geometry"]
    assert jnp.all(jnp.isfinite(nom.sdf))
    # as-built 有限且产生位移
    ab = res["asbuilt"]
    for leaf in jax.tree_util.tree_leaves(ab):
        assert jnp.all(jnp.isfinite(leaf))
    assert jnp.max(jnp.abs(ab.displacement)) > 0.0


def test_optimize_shape_finite_results():
    """形状优化返回有限、物理合理的补偿结果，且不破坏默认降阶链路。"""
    geo = _small_geo()
    res = optimize_shape(
        geo, material="316L", n_steps=8, learning_rate=0.05,
        params=dict(n_grid=8, max_layers=4, n_sub_cp=32),
        smoothness=0.01, verbose=False,
    )
    assert "nominal_geometry" in res and "delta" in res and "asbuilt" in res
    # 降阶 buildup 路径同样可跑（默认链路不受影响）
    res2 = optimize_shape(
        geo, material="316L", n_steps=6, asbuilt_solver="buildup",
        params=dict(n_grid=8), smoothness=0.01, verbose=False,
    )
    assert jnp.all(jnp.isfinite(res2["delta"]))
    assert jnp.all(jnp.isfinite(res2["asbuilt"].sdf))


# ---------------------------------------------------------------------------
# 联合几何+工艺协同优化：把反变形预补偿(形状变量 u) 与工艺窗口调参(工艺变量 z)
# 合并为一个可微目标，用 multi-transform Adam 同步更新两个设计变量
# ---------------------------------------------------------------------------
def _joint_geom_params():
    return dict(n_grid=8, max_layers=4, n_sub_cp=32, tau_activation=1e-3)


def test_joint_co_optimization_reduces_deviation():
    """联合优化：形状补偿 + 工艺窗口同时调，使 as-built 对 target 的尺寸偏差显著下降。

    真实焓法下两杠杆（形状 u + 工艺 z）在 lr=0.05 易过冲、仅降 ~6%；用稳定 lr=0.01
    后协同优化可真实把尺寸偏差降到初始的 ~40% 以下（物理可达区），验证两杠杆均连通。
    """
    geo = _small_geo()  # target = 目标 CAD
    res = optimize_geometry_process(
        geo, material="316L", process_init=_high_energy_plan(geo),
        params=_joint_geom_params(), n_steps=12, learning_rate=0.01,
        smoothness=0.02, verbose=False,
    )
    gh = res["geom_history"]
    assert len(gh) == 12
    # 预补偿 + 工艺窗口协同后，尺寸偏差显著低于初始（delta=0 / 高能量工艺 的 target as-built）
    assert min(gh) < gh[0] * 0.6, f"联合优化未显著降低尺寸偏差: {gh}"
    # 两个设计变量都被激活（偏离初值）—— 证明几何与工艺两条杠杆均连通
    z_init = normalize_process(_high_energy_plan(geo))
    assert jnp.max(jnp.abs(res["delta"])) > 0.0, "形状杠杆未生效（delta 全零）"
    assert jnp.max(jnp.abs(res["z"] - z_init)) > 1e-4, "工艺杠杆未生效（z 未移动）"
    # 工艺仍可行（z ∈ [0,1] 设备边界）
    assert jnp.all(res["z"] >= 0.0) and jnp.all(res["z"] <= 1.0)
    # 结果有限且产出完整契约
    ab = res["asbuilt"]
    for leaf in jax.tree_util.tree_leaves(ab):
        assert jnp.all(jnp.isfinite(leaf))
    assert isinstance(res["nominal_geometry"], PartGeometry)
    assert jnp.all(jnp.isfinite(res["final_plan"].laser_power))


def test_joint_both_levers_move_and_are_finite():
    """联合优化单步即产生有限梯度，且形状与工艺变量互不影响地更新（不 NaN）。"""
    geo = _small_geo()
    res = optimize_geometry_process(
        geo, material="316L", process_init=_high_energy_plan(geo),
        params=_joint_geom_params(), n_steps=3, learning_rate=0.05,
        smoothness=0.02, verbose=False,
    )
    # 损失有限且至少一步下降
    assert jnp.all(jnp.isfinite(jnp.asarray(res["loss_history"])))
    assert res["loss_history"][-1] <= res["loss_history"][0] + 1e-9
    # 历史记录完整、有限
    for uh in res["u_history"]:
        assert jnp.all(jnp.isfinite(uh))
    for zh in res["z_history"]:
        assert jnp.all(jnp.isfinite(zh)) and jnp.all(zh >= 0.0) and jnp.all(zh <= 1.0)


def test_joint_competitive_with_shape_only():
    """联合优化 vs 纯形状优化：两条链路均连通、均真实降低尺寸偏差。

    真实焓法下从同一 sane 初值工艺（默认 heuristic_plan，不传高能量 process_init）出发：
    纯形状与联合优化都把 as-built 尺寸偏差降到 baseline 以下，证明几何(u)与工艺(z)两杠杆
    均连通。注意联合优化目标比纯形状更丰富（几何+残余应力+残余应变+缺陷+安全罚），故其
    geom_dev 下降比**不要求**≥纯形状——本小球问题上形状单杠杆常更优；这里只验证两者均
    "真实下降"，不再强制 joint≥shape（该旧前提仅在 lr=0.05 使形状发散时侥幸成立，非真）。
    """
    geo = _small_geo()
    kw = _joint_geom_params()
    res_shape = optimize_shape(geo, material="316L", n_steps=16,
                               learning_rate=0.01,
                               params=kw, smoothness=0.02, verbose=False)
    res_joint = optimize_geometry_process(geo, material="316L",
                                          params=kw, n_steps=16, learning_rate=0.01,
                                          smoothness=0.02, verbose=False)
    ratio_shape = min(res_shape["geom_history"]) / res_shape["geom_history"][0]
    ratio_joint = min(res_joint["geom_history"]) / res_joint["geom_history"][0]
    # 两杠杆均连通：纯形状与联合优化都把尺寸偏差显著降到 baseline 以下（≥10% 下降）。
    assert ratio_shape < 0.9, \
        f"纯形状优化未显著降低尺寸偏差: ratio_shape={ratio_shape:.3f}"
    assert ratio_joint < 0.9, \
        f"联合优化未显著降低尺寸偏差: ratio_joint={ratio_joint:.3f}"


# ---------------------------------------------------------------------------
# 工艺杠杆物理约束（可微罚项）：能量密度窗口 / 道间搭接 / 层间结合 / 匙孔抑制
# ---------------------------------------------------------------------------
def _heuristic_z(geo, material="316L"):
    return normalize_process(heuristic_plan(geo, material=material, modality="SLM"))


def _deep_keyhole_plan(geo, params=_LIGHT):
    """**本档工艺子盒可表达**的深匙孔/超窗起点（不是"细工艺"起点）。

    为什么要按子盒取值（2026-10-07 D0，实测见
    ``docs/evidence/2026-10-07/am_d0_pen_traj_probe.log``）：A0 把 ``dt`` 钉成物理曝光
    /``n_steps`` 之后，链条档位给工艺盒加了下界 ``layer_thickness ≥ dx``、
    ``hatch_spacing ≥ dx``、``beam_radius ≥ dx/2``。若像 ``_high_energy_plan`` 那样用
    60µm 层厚/120µm 道距当"病态起点"，进优化器第一步就被下界抬成 120µm/60µm，剩下的
    罚项只有粗网格造成的 ``layer_penalty``（实测 ``pen_init=0.78``）——而 6 步 Adam 内
    **连不加约束的轨迹都会把它降到 0**（实测
    ``pen_free=[0.78, 0.42, 0.22, 0.08, 0, 0]``），于是"启用软约束后罚项更低"变成
    空断言（真实报错：``pen_cons=0.000 >= pen_free=0.000``）。
    本起点把越界做在**尺度无关**的 VED/匙孔两项上（功率由目标体能量密度反解、速度贴
    下界、大吸收率），使 ``pen_init`` 在任意网格下同为窗口上界的 1.67 倍越界。
    探针实测同类起点（``pen_init=3.06``）的轨迹：尺寸损失把无约束轨迹**推得更病态**
    （``pen_free=[3.06, 5.27, 8.17, 7.58, 7.12, 6.15]``），软约束则在 5 步内归零
    （``pen_cons=[3.06, 0.86, 0.21, 0.04, 0, 0]``）——前提为真，比较才可分辨。

    ⚠ 子盒必须用**贴着下界的名义工艺**定价，不能用 ``heuristic_plan``：
    ``chain_schedule`` 的 ``scan_speed`` 下界跟着**名义工艺自己的速度**走
    （实测 ``v_nom=1.0→下界 0.5``、``v_nom=0.75→0.375``、``v_nom=0.6→0.3``，即
    ``v_nom/speed_slack``），最坏角落路径长又取 ``min(h_lo, h_nom)``/
    ``min(lt_lo, lt_nom)``。启发式名义工艺给的 ``hatch≈103µm``、``lt=40µm`` 都**小于**
    本网格下界 dx=120µm，用它定价会把速度下界抬高；再"按下界造起点"造出来的其实是
    被抬快了的弱起点（实测 ``pen_init=1.80``，只剩 0.667 的网格性 ``layer_penalty``
    + 1.13 的匙孔项，6 步内被无约束轨迹自己归零 ⇒ 测试再次退化成空断言）。
    用**下界本身**当名义来定价，定价与起点才自洽。
    """
    dx = float(geo.spacing)
    nl = int(geo.layer_count(40e-6))
    floor_nominal = ProcessPlan.uniform(
        nl, modality="SLM", laser_power=1000.0, scan_speed=1.0,
        layer_thickness=dx, hatch_spacing=dx, beam_radius=0.5 * dx,
        absorption=0.4, preheat_temp=473.0, dwell_time=0.0)
    p = thermal_tier(dict(params), geo, floor_nominal, material="316L",
                     thermal_solver="enthalpy")
    tb = tier_bounds(p)
    v_lo, h_lo, lt_lo, r_lo = (tb["scan_speed"][0], tb["hatch_spacing"][0],
                               tb["layer_thickness"][0], tb["beam_radius"][0])
    # 功率由**目标体能量密度**反解（E_v = P/(v·h·t)），使起点不随网格尺度漂移：
    # 2.5e11 J/m³ = VED 窗口上界 1.5e11 的 1.67 倍（匙孔/球化那一侧）。
    ved_target = 2.5e11
    return ProcessPlan.uniform(
        nl, modality="SLM",
        laser_power=float(ved_target * v_lo * h_lo * lt_lo),
        scan_speed=v_lo, layer_thickness=lt_lo, hatch_spacing=h_lo,
        beam_radius=r_lo, absorption=0.6,
        preheat_temp=473.0, dwell_time=0.0)


def test_process_constraint_penalty_finite_and_connected():
    """工艺物理约束罚项在可行工艺上≈0、在病态工艺上显著，且梯度有限可微。"""
    geo = _small_geo()
    nl = int(geo.layer_count(40e-6))
    # 启发式工艺（物理合理）→ 罚项应接近 0
    z_ok = _heuristic_z(geo)
    pen_ok, det_ok = process_constraint_penalty(z_ok, material="316L", n_layers=nl)
    assert jnp.isfinite(pen_ok)
    assert float(pen_ok) < 0.5, f"启发式工艺不应被罚: pen={float(pen_ok)}"
    # 高能量病态工艺 → 罚项应显著大于 0
    z_bad = normalize_process(_high_energy_plan(geo))
    pen_bad, det_bad = process_constraint_penalty(z_bad, material="316L", n_layers=nl)
    assert float(pen_bad) > 0.5, f"病态工艺应被强烈惩罚: pen={float(pen_bad)}"
    # 梯度有限且非零（约束链连通 z）
    g = jax.grad(lambda zz: process_constraint_penalty(zz, material="316L",
                                                      n_layers=nl)[0])(z_bad)
    assert jnp.all(jnp.isfinite(g)), "约束罚项梯度含 NaN/Inf"
    assert jnp.max(jnp.abs(g)) > 0.0, "约束罚项梯度全零（未连通 z）"


def test_process_constraint_reduces_penalty_in_joint():
    """联合优化中启用软约束后，工艺杠杆**沿轨迹**更快进入可行域（同病态初值）。

    注：本测试固定 ``hard_project=False``，专门验证**软罚项**本身的语义
    （constraint_weight 把工艺推入可行域）。硬投影的语义由下面的
    ``test_hard_projection_*`` 系列单独覆盖；"软约束到底有没有接到 z 上"这一
    **接线语义**由 :func:`test_soft_constraint_alone_drives_process_feasible`
    用确定性构造单独覆盖。

    ⚠ 为什么不再断言"末步严格更低"（2026-10-07 D0 实测，两次踩坑）：
    末步比较在演示档是**赌损失走哪条路**，不是测约束。同一夹具族的两条无约束轨迹实测
    分别收敛到完全不同的地方——
      · 起点 P1600/v0.45：``pen_free=[3.06,5.27,8.17,7.58,7.12,6.15]``（越优化越病态）
      · 起点 P1800/v0.50：``pen_free=[3.27,1.07,0.22,0.07,0,0]``（自己就归零）
    后者让 ``pen_cons < pen_free`` 退化成 ``0.0 < 0.0`` 恒假（真实报错见
    ``am_d0_feas_subset.log``）。原因：Adam 按坐标归一化步长，``constraint_weight=20``
    在两条轨迹上只造成**首步 ~10% 的方向差**（1.066 → 0.965），随后两条轨迹几乎重合
    ——这是本项目的**已知杠杆不足**（任务 #17），不该由测试来掩盖。
    因此本测试改成四条**都可分辨**的断言：起点确实病态、首步确实分离、累计罚项确实更低、
    约束轨迹自身单调进入可行域并满足绝对可行阈（``< 0.2`` 阈值原样保留，未放宽）。
    """
    geo = _small_geo()
    kw = _LIGHT
    # 同一起点：本档可表达的深匙孔/超窗工艺（能量密度与焓双双越界）
    init = _deep_keyhole_plan(geo, params=kw)
    res_free = optimize_geometry_process(
        geo, material="316L", process_init=init, params=kw, n_steps=6,
        learning_rate=0.05, smoothness=0.02, constraint_weight=0.0,
        hard_project=False, verbose=False)
    res_cons = optimize_geometry_process(
        geo, material="316L", process_init=init, params=kw, n_steps=6,
        learning_rate=0.05, smoothness=0.02, constraint_weight=20.0,
        hard_project=False, verbose=False)
    hist_free = res_free["constraint_penalty_history"]
    hist_cons = res_cons["constraint_penalty_history"]
    # (0) 两条轨迹必须从**同一个**病态起点出发（前提，不是放宽）
    assert hist_free[0] == pytest.approx(hist_cons[0]), \
        f"对照组与实验组起点不同: {hist_free[0]} vs {hist_cons[0]}"
    assert hist_free[0] > 0.5, f"起点并不病态，比较将退化为空断言: {hist_free}"
    # (1) 首步分离：加入 20·pen 后，第一步就更朝可行域走（这才是约束自身的贡献）
    assert hist_free[1] - hist_cons[1] > 0.02, \
        f"软约束首步未见效果: free={hist_free[1]:.4f} cons={hist_cons[1]:.4f}"
    # (2) 累计罚项更低（整条轨迹更快进入可行域）
    assert sum(hist_cons) < sum(hist_free), \
        f"约束轨迹的累计罚项并不更低: {hist_cons} vs {hist_free}"
    # (3) 约束轨迹单调下降并最终**绝对**可行（阈值 0.2 未放宽）
    assert all(b <= a + 1e-9 for a, b in zip(hist_cons, hist_cons[1:])), \
        f"约束轨迹罚项非单调: {hist_cons}"
    assert hist_cons[-1] < 0.2, f"约束后工艺仍不可行: pen_cons={hist_cons[-1]:.3f}"
    # 两种情况下两条杠杆仍都被正常优化（有限、z 在设备边界内）
    assert jnp.all(jnp.isfinite(res_cons["delta"]))
    assert jnp.all(res_cons["z"] >= 0.0) and jnp.all(res_cons["z"] <= 1.0)


def test_soft_constraint_alone_drives_process_feasible():
    """**确定性**验证软约束的接线语义：目标里只剩 ``w·pen`` 时，Adam 把 z 推进可行域。

    为什么要单独一条（2026-10-07 D0）：联合/尺寸优化的轨迹由尺寸损失主导，同一族起点
    换个 20% 功率就能让无约束轨迹从"越跑越病态"翻成"自己归零"（实测见
    :func:`test_process_constraint_reduces_penalty_in_joint` 的 docstring），
    于是"约束降低了罚项"变成对随机性的抽样。本测试把尺寸损失**逐项置零**
    （``weights`` 全 0）并把 ``service_stress`` 压到极低，使安全罚
    ``relu(1-safety_factor)`` 恒为 0 ⇒ 目标函数 ≡ ``constraint_weight · pen``：
    此时罚项若不下降，只可能是约束没接到 z 上——断言因此是确定的。

    对照组取 ``constraint_weight=0``：目标恒 0 ⇒ Adam 更新为 0 ⇒ 工艺**一步都不该动**
    （``final_plan`` 与初值逐参数相同），证明"罚项下降"确实来自约束项而非别的驱动。

    ⚠ 两条实测出来的性质写在这里，免得后来人把它们当 bug 或当运气：
    ① ``optimize_dimensional`` 的 ``constraint_penalty_history[i]`` 记的是**第 i 步更新
    之后**的罚项（与联合优化器记"更新之前"不同），所以起点罚项不能取 ``hist[0]``，
    本测试因此用提交给优化器的那个 ``z`` 直接算。
    ② 纯 hinge 目标 + Adam 会**过冲**：实测序列
    ``[0.34, 0.0, 0.0, 0.52, 0.63, 0.43, 0.11, 0.0]``——罚项到达 0 后梯度消失，
    Adam 的动量继续推，于是又冲出可行域再被拉回。故本测试只断言"整体下降 + 终值可行"，
    不断言逐步单调；这也是 :func:`optimize_dimensional` 把 ``hard_project=True``
    设为缺省的实测理由（见任务 #17）。
    """
    geo = _small_geo()
    init = _deep_keyhole_plan(geo, params=_TINY)
    zero_w = dict(geom=0.0, stress=0.0, strain=0.0, defect=0.0, powder=0.0)
    kw = dict(material="316L", params=_TINY, n_steps=8, learning_rate=0.08,
              weights=zero_w, service_stress=1e-3, hard_project=False,
              verbose=False, asbuilt_solver="plastic", constitutive="j2")
    cons = optimize_dimensional(geo, process_init=init, constraint_weight=20.0, **kw)
    free = optimize_dimensional(geo, process_init=init, constraint_weight=0.0, **kw)

    # 起点必须显著不可行（直接对提交给优化器的 z 求罚项，避开记账偏移）
    p = thermal_tier(dict(_TINY), geo, init, material="316L",
                     thermal_solver="enthalpy")
    nl = int(geo.layer_count(40e-6))
    pen0, _ = process_constraint_penalty(
        normalize_process(init, bounds=tier_bounds(p)), material="316L",
        n_layers=nl, modality="SLM", params=p)
    assert float(pen0) > 0.5, f"起点应当显著不可行: pen0={float(pen0):.3f}"

    hc = cons["constraint_penalty_history"]
    lh = cons["loss_history"]
    assert hc[-1] < 0.2, f"纯约束驱动 8 步后仍不可行: {hc}"
    assert min(lh) < lh[0] - 1e-6, f"纯约束目标没有下降: {lh}"
    assert jnp.all(jnp.isfinite(jnp.asarray(lh))), "纯约束目标含 NaN/Inf"
    # 对照：目标恒 0 时优化器不该动工艺
    assert max(abs(l) for l in free["loss_history"]) == 0.0, \
        f"零权重对照组目标非恒零: {free['loss_history']}"
    for field in ("laser_power", "scan_speed", "layer_thickness",
                  "hatch_spacing", "beam_radius", "absorption"):
        got = float(jnp.mean(jnp.atleast_1d(getattr(free["final_plan"], field))))
        want = float(jnp.mean(jnp.atleast_1d(getattr(init, field))))
        assert got == pytest.approx(want, rel=1e-9), \
            f"零目标对照组不该改变 {field}: {want} -> {got}"



def test_dimensional_constraint_keeps_plan_feasible():
    """optimize_dimensional 启用软约束后，最优工艺落在物理可行域，同时尺寸损失下降。"""
    geo = _small_geo()
    init = _high_energy_plan(geo)
    res = optimize_dimensional(
        geo, process_init=init, material="316L", params=_LIGHT, n_steps=8,
        learning_rate=0.08, weights=_W, constraint_weight=20.0,
        hard_project=False, verbose=False)
    hist = res["loss_history"]
    assert min(hist) < hist[0] - 1e-6, "约束优化下尺寸损失未下降"
    # 最终工艺可行度应较高（满足能量密度/搭接/层间/匙孔约束）
    nl = int(geo.layer_count(40e-6))
    feas = process_feasibility(res["z"], material="316L", n_layers=nl, modality="SLM")
    assert float(feas) > 0.5, f"约束后工艺可行度偏低: feas={float(feas)}"
    # 设备边界与有限性
    plan = res["final_plan"]
    assert jnp.all(jnp.isfinite(plan.laser_power))
    assert jnp.all(plan.laser_power >= PROCESS_BOUNDS["laser_power"][0])
    assert jnp.all(plan.laser_power <= PROCESS_BOUNDS["laser_power"][1])


# ---------------------------------------------------------------------------
# 硬投影（解析可行流形 Dykstra 投影）：替代纯软罚项
# ---------------------------------------------------------------------------
def test_hard_projection_is_feasible_idempotent_and_finite():
    """project_process_feasible 对任意 z 输出可行、有限；对已可行点近似不动点。"""
    nl = 10
    rng = np.random.default_rng(0)
    # 随机（含病态）工艺向量
    for _ in range(5):
        z = jnp.asarray(rng.uniform(-0.3, 1.2, size=9))
        zf = project_process_feasible(z, material="316L", n_layers=nl, modality="SLM")
        # 有限 + 设备边界
        assert jnp.all(jnp.isfinite(zf))
        assert jnp.all(zf >= 0.0) and jnp.all(zf <= 1.0)
        # 投影后可行（软罚项应≈0）
        pen, _ = process_constraint_penalty(zf, material="316L", n_layers=nl, modality="SLM")
        assert float(pen) < 1e-4, f"投影未达可行: pen={float(pen):.2e}"
        # 幂等：再投影应不动
        zf2 = project_process_feasible(zf, material="316L", n_layers=nl, modality="SLM")
        assert float(jnp.max(jnp.abs(zf2 - zf))) < 1e-4, "硬投影非幂等"
    # 已可行点（显式构造，落在全部约束内部）几乎不动
    z_ok = normalize_process(ProcessPlan.uniform(
        nl, modality="SLM", laser_power=200.0, scan_speed=1.0,
        layer_thickness=40e-6, hatch_spacing=80e-6, beam_radius=50e-6,
        absorption=0.4, preheat_temp=373.0, dwell_time=0.0))
    # 先验：该点确实可行
    pen_ok, _ = process_constraint_penalty(z_ok, material="316L", n_layers=nl, modality="SLM")
    assert float(pen_ok) < 1e-4, f"测试用'可行点'其实不可行: pen={float(pen_ok):.2e}"
    zp = project_process_feasible(z_ok, material="316L", n_layers=nl, modality="SLM")
    assert float(jnp.max(jnp.abs(zp - z_ok))) < 1e-3, "可行点被硬投影移动过多"


def test_hard_projection_reduces_penalty_vs_input():
    """硬投影后软罚项不增（对病态/边界外输入更可行）。"""
    nl = 10
    # 病态：高功率 + 慢扫描 + 大层厚 → 明显越界
    z_bad = normalize_process(ProcessPlan.uniform(
        nl, modality="SLM", laser_power=3800.0, scan_speed=0.05,
        layer_thickness=900e-6, hatch_spacing=20e-6, beam_radius=15e-6,
        absorption=0.85, preheat_temp=1273.0, dwell_time=0.0))
    pen_in, _ = process_constraint_penalty(z_bad, material="316L", n_layers=nl, modality="SLM")
    zf = project_process_feasible(z_bad, material="316L", n_layers=nl, modality="SLM")
    pen_out, _ = process_constraint_penalty(zf, material="316L", n_layers=nl, modality="SLM")
    assert float(pen_out) < float(pen_in) - 1e-3, \
        f"硬投影未降低罚项: {float(pen_in):.3f} -> {float(pen_out):.3f}"


def test_hard_projection_in_joint_yields_feasible_plan():
    """联合优化启用硬投影后，最终工艺物理可行（无需软罚项），且杠杆被正常优化。"""
    geo = _small_geo()
    kw = _LIGHT
    init = _high_energy_plan(geo)
    res = optimize_geometry_process(
        geo, material="316L", process_init=init, params=kw, n_steps=10,
        learning_rate=0.05, smoothness=0.02, constraint_weight=0.0,
        hard_project=True, verbose=False)
    nl = int(geo.layer_count(40e-6))
    pen = res["constraint_penalty_history"][-1]
    feas = process_feasibility(res["z"], material="316L", n_layers=nl, modality="SLM")
    # 硬投影保证可行：软罚项≈0、可行度≈1
    assert pen < 0.2, f"硬投影后工艺仍不可行: pen={pen:.3f}"
    assert float(feas) > 0.9, f"硬投影后可行度偏低: feas={float(feas)}"
    assert jnp.all(jnp.isfinite(res["delta"]))
    assert jnp.all(res["z"] >= 0.0) and jnp.all(res["z"] <= 1.0)
    # 工艺杠杆确实被优化（偏离初值）
    z0 = normalize_process(init)
    assert float(jnp.max(jnp.abs(res["z"] - z0))) > 1e-3, "工艺杠杆未被优化"


def test_hard_projection_in_dimensional_yields_feasible_plan():
    """optimize_dimensional 启用硬投影后，最终工艺物理可行且尺寸损失下降。"""
    geo = _small_geo()
    init = _high_energy_plan(geo)
    res = optimize_dimensional(
        geo, process_init=init, material="316L", params=_LIGHT, n_steps=8,
        learning_rate=0.08, weights=_W, constraint_weight=0.0,
        hard_project=True, verbose=False)
    hist = res["loss_history"]
    assert min(hist) < hist[0] - 1e-6, "硬投影下尺寸损失未下降"
    nl = int(geo.layer_count(40e-6))
    feas = process_feasibility(res["z"], material="316L", n_layers=nl, modality="SLM")
    assert float(feas) > 0.9, f"硬投影后可行度偏低: feas={float(feas)}"
    plan = res["final_plan"]
    assert jnp.all(jnp.isfinite(plan.laser_power))
    assert jnp.all(plan.laser_power >= PROCESS_BOUNDS["laser_power"][0])
    assert jnp.all(plan.laser_power <= PROCESS_BOUNDS["laser_power"][1])


# ---------------------------------------------------------------------------
# 多目标 Pareto 协同优化：尺寸偏差 ↔ 残余应力 权衡
# ---------------------------------------------------------------------------
def test_pareto_front_nsga2_non_dominated_and_feasible():
    """NSGA-II 产出**有效非支配前沿 + 可行最优解**（对应任务 #37，替代 λ-标量化）。

    真实焓法下该小球问题的两目标（尺寸偏差 / 残余应力）高度同向——均受热收缩主导，
    故末代非支配前沿退化为**单个支配点**（无真实 Pareto 权衡）。因此本测试不要求
    "前沿权衡跨度"，而验证：(1) 前沿确实非支配、(2) 各解工艺可行、(3) NSGA-II 确实
    探索了目标空间（全种群目标有真实分布，非恒等）、(4) 最优点（best_compromise）齐全。
    """
    geo = _small_geo()
    res = pareto_optimize_geometry_process(
        geo, material="316L", params=_TINY,
        algorithm="nsga2", pop_size=8, n_gen=3, seed=0,
        smoothness=0.02, constraint_weight=20.0, asbuilt_solver="plastic",
        verbose=False)
    assert res["algorithm"] == "nsga2"
    gd = res["geom_dev"]
    st = res["stress"]
    assert jnp.all(jnp.isfinite(gd)) and jnp.all(jnp.isfinite(st))
    # 每个 Pareto 解都满足工艺物理约束（可行度应较高）
    assert jnp.all(res["feasible"] > 0.5), \
        f"Pareto 解存在不可行工艺: feasible={res['feasible']}"
    # 前沿必须是**非支配**的：整张表即第一非支配层（NSGA-II 的核心正确性）
    F = np.stack([np.asarray(gd), np.asarray(st)], axis=1)
    fronts = fast_non_dominated_sort(F)
    assert fronts[0] == list(range(F.shape[0])), \
        f"NSGA-II 前沿并非全部非支配: front0={fronts[0]}"
    # 真实焓法下前沿可能退化为单支配点（两目标同向，无权衡），故验证"种群探索"
    # 而非"前沿权衡"：全种群目标应有真实分布（证明 NSGA-II 在目标空间做了有效
    # 搜索，而非恒等塌缩）。
    pop_g = np.asarray(res["population_geom_dev"])
    pop_s = np.asarray(res["population_stress"])
    span_g = float(np.max(pop_g) - np.min(pop_g))
    span_s = float(np.max(pop_s) - np.min(pop_s))
    assert span_g > 1e-6 or span_s > 1e-6, \
        f"NSGA-II 种群未探索出目标分布: Δgeom={span_g:.2e}, Δstress={span_s:.2e}"
    # utopia 与最优点齐全
    assert isinstance(res["utopia"], tuple) and len(res["utopia"]) == 2
    bc = res["best_compromise"]
    assert "final_plan" in bc and "delta" in bc and "asbuilt" in bc
    assert jnp.all(jnp.isfinite(bc["delta"]))


def test_pareto_lambda_legacy_still_works():
    """回归守卫：重构后 legacy 'lambda' 标量化路径仍可用、仍可行。"""
    geo = _small_geo()
    res = pareto_optimize_geometry_process(
        geo, material="316L", params=_LIGHT,
        algorithm="lambda", lambdas=(0.0, 1.0), n_steps=4, learning_rate=0.05,
        smoothness=0.02, constraint_weight=20.0, asbuilt_solver="plastic",
        verbose=False)
    assert res["algorithm"] == "lambda"
    assert len(res["solutions"]) == 2
    assert len(res["lambdas"]) == 2
    gd = res["geom_dev"]
    st = res["stress"]
    assert jnp.all(jnp.isfinite(gd)) and jnp.all(jnp.isfinite(st))
    assert jnp.all(res["feasible"] > 0.5), \
        f"legacy λ 前沿存在不可行工艺: feasible={res['feasible']}"
    span_g = float(jnp.max(gd) - jnp.min(gd))
    span_s = float(jnp.max(st) - jnp.min(st))
    assert span_g > 1e-6 or span_s > 1e-6, \
        f"legacy λ 前沿未体现权衡: Δgeom={span_g:.2e}, Δstress={span_s:.2e}"
    bc = res["best_compromise"]
    assert "final_plan" in bc and "delta" in bc and "asbuilt" in bc
    assert jnp.all(jnp.isfinite(bc["delta"]))


