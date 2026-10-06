"""默认求解器必须是真实数值离散（非闭式/解析近似）回归测试。

用户硬原则：仿真 = 数值方法，默认前向链路不得走闭式/解析解
（Eagar-Tsai / Rosenthal 类）。本文件守住问题 ①+② 的修复：

* ① 熔池默认求解器由闭式解析 ``meltpool.surrogate_eagar_tsai``（Eagar-Tsai）
    切到真实数值 ``meltpool.fdm``（3D 瞬态焓法 FVM，无自由界面，便宜真实档）；
    高保真 ``meltpool.vof_flow3d``（全 VOF CFD）仍经 ``select`` 可用，但不作静默默认。
* ② 热学默认求解器由闭式降阶 ``thermal.history``（Rosenthal）自动排除，
    落到真实数值 ``thermal.enthalpy``（FVM 焓法），与 ``inverse.simulate`` 默认一致，
    两条前向路径统一。

机制：``method`` 分类 + ``exclude_methods=("analytical",)`` 贯穿
``Pipeline.auto`` 与 forgecore ``_select``，双引擎一致排除闭式/解析近似默认。

运行：``pytest tests/test_default_solver_numerical.py -q``
"""
import jax
jax.config.update("jax_enable_x64", True)

import amforge as af
import amforge.forge_adapter  # 触发 AM 求解器注册进 ForgeCore
from forgecore import REGISTRY
from forgecore.model import ensure_registered

from amforge.core.registry import get_solver
from amforge.meltpool import solve_meltpool_fdm


# ===========================================================================
# amforge Pipeline.auto：默认链路
# ===========================================================================
def test_amforge_default_meltpool_is_numerical():
    """默认熔池 = meltpool.fdm（真实数值 FVM），非 Eagar-Tsai 解析代理。"""
    pipe = af.Pipeline.auto("ServiceVerdict", modality="SLM")
    names = pipe.solver_names()
    assert "meltpool.fdm" in names, f"默认熔池应为 meltpool.fdm，实际: {names}"
    assert "meltpool.surrogate_eagar_tsai" not in names, \
        f"默认链路不应含闭式解析代理: {names}"
    assert get_solver("meltpool.fdm").method == "numerical"


def test_amforge_default_thermal_is_numerical():
    """默认热学 = thermal.enthalpy（真实数值 FVM），非 Rosenthal 解析降阶。"""
    pipe = af.Pipeline.auto("ServiceVerdict", modality="SLM")
    names = pipe.solver_names()
    assert "thermal.enthalpy" in names, f"默认热学应为 thermal.enthalpy，实际: {names}"
    assert "thermal.history" not in names, \
        f"默认链路不应含 Rosenthal 解析降阶: {names}"
    assert get_solver("thermal.enthalpy").method == "numerical"


def test_amforge_analytical_excluded_by_default_but_selectable():
    """默认排除 analytical；放开 exclude_methods 或 select 仍可强制用解析解。"""
    pipe = af.Pipeline.auto("ServiceVerdict", modality="SLM",
                            exclude_methods=("analytical",))
    assert "meltpool.surrogate_eagar_tsai" not in pipe.solver_names()
    assert "thermal.history" not in pipe.solver_names()

    # 优化闭环的低成本前向：显式 select 解析解仍可用（注册未被删除）
    pipe2 = af.Pipeline.auto(
        "ServiceVerdict", modality="SLM",
        select={"meltpool": "meltpool.surrogate_eagar_tsai",
                "thermal": "thermal.history"})
    names2 = pipe2.solver_names()
    assert "meltpool.surrogate_eagar_tsai" in names2
    assert "thermal.history" in names2


def test_amforge_high_fidelity_vof_selectable_not_default():
    """全 VOF CFD 高保真档必须经 select 显式切换，不作静默默认。"""
    pipe = af.Pipeline.auto("ServiceVerdict", modality="SLM")
    assert "meltpool.vof_flow3d" not in pipe.solver_names(), \
        "vof_flow3d 太重，不应成为静默默认"
    pipe2 = af.Pipeline.auto("ServiceVerdict", modality="SLM",
                             select={"meltpool": "meltpool.vof_flow3d"})
    assert "meltpool.vof_flow3d" in pipe2.solver_names()


# ===========================================================================
# forgecore auto：默认链路（双引擎一致性）
# ===========================================================================
def test_forgecore_default_prefers_numerical():
    ensure_registered()
    g = REGISTRY.build("verdict", modality="SLM",
                       params={"material": "316L", "n_grid": 20})
    meltpool_spec = [s for s in g.specs if s.produces == "meltpool"]
    thermal_spec = [s for s in g.specs if s.produces == "thermal"]
    assert meltpool_spec, "forgecore 链路未产出 meltpool"
    assert thermal_spec, "forgecore 链路未产出 thermal"
    assert meltpool_spec[0].name == "meltpool.fdm", \
        f"forgecore 默认熔池应为 meltpool.fdm，实际 {meltpool_spec[0].name}"
    assert thermal_spec[0].name == "thermal.enthalpy", \
        f"forgecore 默认热学应为 thermal.enthalpy，实际 {thermal_spec[0].name}"
    assert "analytical" not in meltpool_spec[0].tags
    assert "analytical" not in thermal_spec[0].tags


def test_forgecore_analytical_selectable():
    ensure_registered()
    g = REGISTRY.build("verdict", modality="SLM",
                       params={"material": "316L"},
                       select={"meltpool": "meltpool.surrogate_eagar_tsai",
                               "thermal": "thermal.history"})
    names = {s.name for s in g.specs}
    assert "meltpool.surrogate_eagar_tsai" in names
    assert "thermal.history" in names


# ===========================================================================
# 真实数值求解器的物理 + 可微性守卫（防 fdm 退化为不加热/不可微）
# ===========================================================================
def test_meltpool_fdm_runs_and_is_differentiable():
    """meltpool.fdm 必须真实加热（产生有限熔深）且对工艺参数可微。"""
    import jax.numpy as jnp
    from amforge import geometry as G
    from amforge.core.contracts import ProcessPlan

    geo = G.from_sdf_fn(
        lambda x: jnp.linalg.norm(x, axis=-1) - 0.45e-3,
        bounds=[(-0.6e-3, 0.6e-3)] * 3, spacing=50e-6, name="def_solver",
    )
    proc = ProcessPlan.uniform(laser_power=250.0, scan_speed=0.8,
                               layer_thickness=40e-6, hatch_spacing=80e-6,
                               beam_radius=50e-6, absorption=0.4)

    out = solve_meltpool_fdm(geometry=geo, process=proc,
                             params={"material": "316L", "n_grid": 24, "n_steps": 80})
    assert jnp.all(jnp.isfinite(out.temperature)), "温度场非有限"
    assert jnp.all(jnp.isfinite(out.depth)), "熔深非有限"
    # 真实数值求解器必须实际熔化（闭式/占位返回 0 会被此断言抓住）
    assert float(out.depth) > 1e-6, f"meltpool.fdm 未熔化（熔深≈0）: {float(out.depth)}"
    assert float(out.depth) < 5e-3, "熔深量级异常"

    def loss(P):
        p2 = proc.replace(laser_power=P)
        r = solve_meltpool_fdm(geometry=geo, process=p2,
                               params={"material": "316L", "n_grid": 24, "n_steps": 80})
        return r.depth + r.lof_indicator * 1e-3

    g = jax.grad(loss)(300.0)
    assert jnp.isfinite(g), "meltpool.fdm 梯度含 NaN/Inf"
