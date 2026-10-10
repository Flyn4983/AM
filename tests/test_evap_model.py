"""#27 代码腿：蒸发封顶 `evap_model` 三臂选择器的**行为**测试。

守住的是**新能力**（三臂各自语义），而不是"缺省档没被顺手改"——后者由全量回归里
`test_convection_removes_energy` 那条红灯的打印值逐字未变来守（选型轮 G0a/G0b 锚点
2937.060997302282 / 2908.771450618092）。夹具沿用 `test_boundary.py` 的 0.4mm/dx=80µm
4 层 200W·1.0m·s⁻¹ 立方——它实测把峰值推到 T_lo=T_boil-200=2890K **之上**（钳位区），
∂peak/∂P 反号这一"功能性缺陷"只有在这副夹具里才现形。

CPU 钉住（本仓测试由 pytest 环境统一跑）；无新增 skip/xfail。
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from amforge.core.contracts import ProcessPlan, solid_mask
from amforge.thermal_enthalpy import (
    solve_enthalpy_thermal, _evap_sink, _evap_sink_hk, suggest_n_steps,
)
from amforge.materials import get_material


def _clamp_fixture():
    from amforge.gui.preproc import build_primitive
    part = build_primitive("box", length_mm=0.4, spacing_um=80.0)
    plan = ProcessPlan.uniform(
        4, modality="SLM", laser_power=200.0, scan_speed=1.0,
        layer_thickness=40e-6, hatch_spacing=80e-6, beam_radius=50e-6,
        absorption=0.4, preheat_temp=373.0)
    return part, plan


def _peak(res) -> float:
    return float(np.max(np.asarray(res.peak_temperature)))


def _solve(part, plan, **params):
    p = {"material": "316L"}
    p.update(params)
    return solve_enthalpy_thermal(geometry=part, process=plan, params=p)


# ---------------------------------------------------------------------------
# 1. 选择器只认三值；越界立即抛错（在 setup 抛，不进昂贵 scan）
# ---------------------------------------------------------------------------
def test_evap_model_rejects_unknown():
    part, plan = _clamp_fixture()
    with pytest.raises(ValueError, match="evap_model"):
        _solve(part, plan, evap_model="bogus")


# ---------------------------------------------------------------------------
# 2. HK 封顶**无自由标定参数**：值只由材料卡 + accommodation 决定，与 c_evap 无关。
#    （这正是它相对经验钳的资格——#27 的红线是"禁止以让测试转绿为靶去标定系数"。）
# ---------------------------------------------------------------------------
def test_hk_is_parameter_free():
    mat = get_material("316L")
    dx = 80e-6
    T = jnp.asarray([2700.0, 3000.0, 3090.0, 3500.0], dtype=jnp.float64)
    got = np.asarray(_evap_sink_hk(T, mat, dx))
    want = np.asarray(mat.evaporation_flux(T) * mat.latent_vapor / dx)
    assert np.array_equal(got, want)          # 逐位＝材料卡量组合，不含 c_evap
    # HK(T) 在越过 T_boil 附近单调升（散热更强），且低温区有限、非负
    assert np.all(np.isfinite(got)) and np.all(got >= 0.0)
    assert got[-1] > got[0]


# ---------------------------------------------------------------------------
# 3. "none" ≡ "tuned" 且 evap_coeff=0（封顶恒零）；"tuned" 缺省 ≡ 显式 tuned。
#    两条都用**逐位**比较 ⇒ 既证明 none 提供了独立的置 0 旋钮（A3「无蒸发」对齐所需），
#    又证明不传 evap_model 走的就是原经验钳 ⇒ 默认档绝对数值不动的单元级正面证明。
# ---------------------------------------------------------------------------
def test_none_and_default_arm_semantics():
    part, plan = _clamp_fixture()
    tf_default = np.asarray(_solve(part, plan).final_temperature)
    tf_tuned = np.asarray(_solve(part, plan, evap_model="tuned").final_temperature)
    tf_none = np.asarray(_solve(part, plan, evap_model="none").final_temperature)
    tf_coeff0 = np.asarray(_solve(part, plan, evap_coeff=0.0).final_temperature)
    assert np.array_equal(tf_default, tf_tuned)      # 缺省＝tuned
    assert np.array_equal(tf_none, tf_coeff0)        # none＝把经验钳置 0
    # 但 none ≠ 缺省（缺省封顶在钳位区真的在压峰）⇒ 三臂确是可区分的行为，不是别名
    assert not np.array_equal(tf_none, tf_default)


# ---------------------------------------------------------------------------
# 4. HK 臂与 tuned 臂行为不同：在同一 clamp 夹具上 HK 抬峰（选型轮实测 +14.48%）。
#    照实登记"封顶＝沸点墙"在两副口径下都**没被证成**——HK 峰也越过 T_boil。
# ---------------------------------------------------------------------------
def test_hk_arm_differs_from_tuned():
    mat = get_material("316L")
    part, plan = _clamp_fixture()
    p_tuned = _peak(_solve(part, plan, evap_model="tuned"))
    p_hk = _peak(_solve(part, plan, evap_model="hk"))
    assert np.isfinite(p_hk)
    assert p_hk > p_tuned                       # 无参数 HK 钳得比经验钳松
    assert p_hk > float(mat.T_boil)             # 且没把峰值钉在 T_boil（诚实登记）


# ---------------------------------------------------------------------------
# 5. 核心动机＝可微性：tuned 经验钳在钳位区交付**符号相反**的 ∂peak/∂P（越大功率峰值
#    越低），HK 交付正确正号。这条把"为什么 c_evap 不该常驻默认"从审美变成功能缺陷的
#    回归锁（复现选型轮 G4b：tuned −0.060 → hk +0.383）。n_steps 先 eager 钉死（trace
#    下静态形状不能从 tracer 推）；laser_power 是可微叶子。
# ---------------------------------------------------------------------------
def test_gradient_sign_functional_defect_fixed():
    part, plan = _clamp_fixture()
    ns = int(suggest_n_steps(part, plan, material="316L"))
    pl0 = plan.replace(laser_power=jnp.asarray([200.0] * 4, dtype=jnp.float64))

    def peak_of(pl, model):
        res = solve_enthalpy_thermal(
            geometry=part, process=pl,
            params={"material": "316L", "n_steps": ns, "evap_model": model})
        return jnp.max(res.peak_temperature)

    g_tuned = float(np.asarray(
        jax.grad(lambda pl: peak_of(pl, "tuned"))(pl0).laser_power)[0])
    g_hk = float(np.asarray(
        jax.grad(lambda pl: peak_of(pl, "hk"))(pl0).laser_power)[0])
    assert np.isfinite(g_tuned) and np.isfinite(g_hk)
    assert g_tuned < 0.0 < g_hk                  # 反号 ⇒ 正是待修的功能缺陷
