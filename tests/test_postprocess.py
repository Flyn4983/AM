"""M3 二次工艺（模块B）测试：契约 / 四求解器 / 双注册表镜像 / verdict 消费 / GUI 逻辑。

覆盖：
* SecondaryProcessResult 的 pytree 往返（可被 jax.grad/jit 穿透）；
* postprocess.passthrough 直通基线（等价于直接用成形件）；
* HT 应力松弛、HIP 孔隙闭合、machining 几何修形 的真实物理效应；
* 双注册表（amforge.core.registry + forgecore REGISTRY）镜像一致性；
* Pipeline.auto(ServiceVerdict) 现消费 secondary（默认走 passthrough，select 可切换 HT/HIP）；
* verdict.service 用 secondary 评定（残余应力按松弛、密度按 HIP 修正）；
* 纯逻辑层 apply_secondary（GUI 卡片落点）。
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import amforge as af
from amforge.core.contracts import (
    AsBuiltPart, PartGeometry, StructuralResult, ConstitutiveField,
    SecondaryProcessResult,
)
from amforge.postprocess_secondary import (
    postprocess_passthrough, ht_relax, hip_densify, subtractive,
)
from amforge.verdict import solve_verdict


# ---------------------------------------------------------------------------
# 测试夹具
# ---------------------------------------------------------------------------
def _fake_asbuilt() -> AsBuiltPart:
    sh = (4, 4)
    sdf = -jnp.ones(sh)                       # 全实体
    disp = jnp.zeros((*sh, 2))
    rs = jnp.full((*sh, 3), 1.0)              # 2D 残余应力 1 Pa（sxx,syy,sxy）
    strain = jnp.zeros((*sh, 3))
    return AsBuiltPart(sdf=sdf, displacement=disp, residual_stress=rs,
                       residual_strain=strain, spacing=1e-4, dim=2)


def _fake_structural() -> StructuralResult:
    sh = (4, 4)
    vm = jnp.full(sh, 10.0)                   # 峰值 von Mises = 10 Pa
    stress = jnp.full((*sh, 3), 10.0)
    strain = jnp.zeros((*sh, 3))
    disp = jnp.zeros((*sh, 2))
    reaction = jnp.zeros((1,))
    return StructuralResult(displacement=disp, stress=stress, strain=strain,
                            von_mises=vm, reaction=reaction, dim=2)


def _fake_constitutive() -> ConstitutiveField:
    sh = (4, 4)
    return ConstitutiveField(
        E_inplane=jnp.full(sh, 200e9), E_build=jnp.full(sh, 200e9),
        nu=jnp.full(sh, 0.3), G_build=jnp.full(sh, 80e9),
        sigma_y0=jnp.full(sh, 300e6), hardening_sat=jnp.full(sh, 100e6),
        hardening_rate=jnp.full(sh, 0.1),
        hill_F=jnp.full(sh, 0.5), hill_G=jnp.full(sh, 0.5),
        hill_H=jnp.full(sh, 0.5), hill_L=jnp.full(sh, 0.5),
        hill_M=jnp.full(sh, 0.5), hill_N=jnp.full(sh, 0.5),
        porosity=jnp.full(sh, 0.01), damage=jnp.full(sh, 0.0),
        spacing=1e-4, dim=2,
    )


def _box_sdf_2d(n: int, half: float) -> jnp.ndarray:
    xs = (jnp.arange(n, dtype=jnp.float64) - (n - 1) / 2.0)
    X, Y = jnp.meshgrid(xs, xs, indexing="ij")
    qx = jnp.abs(X) - half
    qy = jnp.abs(Y) - half
    outside = jnp.sqrt(jnp.maximum(qx, 0.0) ** 2 + jnp.maximum(qy, 0.0) ** 2)
    inside = jnp.minimum(jnp.maximum(qx, qy), 0.0)
    return outside + inside                          # 盒体精确 SDF


def _box_pair():
    """名义小盒 ⊂ 成形大盒（同栅格），用于验证 machining 修形 = 交集。"""
    n = 5
    nom_sdf = _box_sdf_2d(n, 1.0)
    big_sdf = _box_sdf_2d(n, 2.0)
    geo = PartGeometry(sdf=nom_sdf, origin=jnp.zeros(2), spacing=1.0, dim=2,
                       name="nominal")
    ab = AsBuiltPart(sdf=big_sdf, displacement=jnp.zeros((n, n, 2)),
                     residual_stress=jnp.zeros((n, n, 3)),
                     residual_strain=jnp.zeros((n, n, 3)),
                     spacing=1.0, dim=2)
    return geo, ab


# ---------------------------------------------------------------------------
# 1. 契约 pytree 往返
# ---------------------------------------------------------------------------
def test_secondaryprocessresult_pytree_roundtrip():
    ab = _fake_asbuilt()
    r = SecondaryProcessResult(
        part=ab, treatment="HT",
        residual_relief_factor=jnp.asarray(0.3),
        density_after=jnp.asarray(0.99),
        geometry_after=ab.sdf, applied_processes=("HT",),
        params={"temperature_K": 1200.0},
    )
    leaves, treedef = jax.tree_util.tree_flatten(r)
    r2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert r2.treatment == "HT"
    assert r2.applied_processes == ("HT",)
    assert float(r2.relief_ratio()) == 0.3
    assert float(r2.density()) == 0.99
    np.testing.assert_array_equal(np.asarray(r2.part.sdf), np.asarray(ab.sdf))


# ---------------------------------------------------------------------------
# 2. passthrough 直通基线
# ---------------------------------------------------------------------------
def test_passthrough_wraps_asbuilt():
    ab = _fake_asbuilt()
    r = postprocess_passthrough(asbuilt=ab, params={})
    assert r.treatment == "none"
    assert float(r.relief_ratio()) == 1.0
    assert float(r.density()) == 1.0
    assert r.part is ab                       # 直通：part 即原成形件


# ---------------------------------------------------------------------------
# 3. HT 应力松弛（可微）
# ---------------------------------------------------------------------------
def test_ht_relieves_residual_stress():
    ab = _fake_asbuilt()
    r = ht_relax(asbuilt=ab, params={"temperature_K": 1500.0, "hold_time_s": 7200.0})
    assert r.treatment == "HT"
    relief = float(r.relief_ratio())
    assert 0.0 < relief < 1.0
    np.testing.assert_allclose(
        np.asarray(r.part.residual_stress),
        np.asarray(ab.residual_stress) * relief, rtol=1e-5)
    # 几何不变
    np.testing.assert_array_equal(np.asarray(r.geometry_after), np.asarray(ab.sdf))


# ---------------------------------------------------------------------------
# 4. HIP 孔隙闭合（可微）
# ---------------------------------------------------------------------------
def test_hip_closes_porosity():
    ab = _fake_asbuilt()
    r = hip_densify(asbuilt=ab, params={"initial_porosity": 0.05,
                                        "pressure_Pa": 1.5e8, "temperature_K": 1100.0,
                                        "hold_time_s": 7200.0})
    assert r.treatment == "HIP"
    density = float(r.density())
    assert 0.95 < density < 1.0               # 0.05 孔隙被闭向 ~1.0
    assert float(r.relief_ratio()) < 1.0


# ---------------------------------------------------------------------------
# 5. machining 几何修形（differentiable=False）
# ---------------------------------------------------------------------------
def test_machining_trims_to_nominal():
    geo, ab = _box_pair()
    r = subtractive(asbuilt=ab, geometry=geo, params={})
    assert r.treatment == "machining"
    # 最终形状 = 名义 ∩ 成形 = 名义（名义 ⊂ 成形）
    np.testing.assert_array_equal(np.asarray(r.geometry_after), np.asarray(geo.sdf))
    assert r.part.dim == 2


# ---------------------------------------------------------------------------
# 6. 双注册表镜像一致性
# ---------------------------------------------------------------------------
def test_mirror_postprocess_registered_in_both():
    import amforge.forge_adapter               # 触发 forgecore 镜像
    import amforge.postprocess_secondary
    import amforge.core.registry as A
    from forgecore.registry import REGISTRY

    for name in ("postprocess.passthrough", "postprocess.heat_treatment",
                 "postprocess.hip", "postprocess.machining"):
        assert name in A._SOLVERS, f"amforge 原生缺失 {name}"
        assert name in REGISTRY.list_names(), f"forgecore 缺失 {name}"
    # verdict 消费 secondary（双注册表）
    assert "SecondaryProcessResult" in A.get_solver("verdict.service").consumes
    assert "secondary" in REGISTRY.get("verdict.service").consumes
    # forgecore auto 能产出 secondary 端口
    g = REGISTRY.build("verdict", modality="SLM",
                       params={"material": "316L", "n_grid": 20})
    assert "secondary" in {s.produces for s in g.specs}


def test_mirror_costs_match_native_postprocess():
    import amforge.forge_adapter
    import amforge.core.registry as A
    from forgecore.registry import REGISTRY
    for name in ("postprocess.passthrough", "postprocess.heat_treatment",
                 "postprocess.hip", "postprocess.machining"):
        fc = REGISTRY.get(name).cost
        am = A.get_solver(name).cost
        assert fc == am, f"镜像 cost 发散 {name}: fc={fc} am={am}"


# ---------------------------------------------------------------------------
# 7. Pipeline.auto(ServiceVerdict) 现消费 secondary
# ---------------------------------------------------------------------------
def test_pipeline_serviceverdict_includes_secondary():
    pipe = af.Pipeline.auto("ServiceVerdict", modality="SLM")
    assert pipe.differentiable
    assert "postprocess.passthrough" in pipe.solver_names()   # 默认最低代价生产者
    assert "SecondaryProcessResult" in af.get_solver("verdict.service").consumes
    pipe.validate()


def test_select_secondary_switches_solver():
    pipe = af.Pipeline.auto("ServiceVerdict", modality="SLM",
                            select={"secondary": "postprocess.heat_treatment"})
    assert "postprocess.heat_treatment" in pipe.solver_names()
    assert "postprocess.passthrough" not in pipe.solver_names()


# ---------------------------------------------------------------------------
# 8. verdict 用 secondary 评定
# ---------------------------------------------------------------------------
def test_verdict_consumes_secondary():
    ab, st, co = _fake_asbuilt(), _fake_structural(), _fake_constitutive()
    v_none = solve_verdict(structural=st, asbuilt=ab, constitutive=co, params={})
    sec = ht_relax(asbuilt=ab, params={"temperature_K": 1500.0, "hold_time_s": 7200.0})
    v_ht = solve_verdict(structural=st, asbuilt=ab, constitutive=co,
                         secondary=sec, params={})
    # HT 松弛后残余应力峰值低于直通
    assert (float(v_ht.margin_report["residual_stress_vm_max_Pa"])
            < float(v_none.margin_report["residual_stress_vm_max_Pa"]))
    assert float(v_ht.margin_report["residual_relief_factor"]) < 1.0
    assert float(v_ht.margin_report["density_after"]) == 1.0     # HT 不改密度


# ---------------------------------------------------------------------------
# 9. 纯逻辑层 apply_secondary（GUI 卡片落点）
# ---------------------------------------------------------------------------
def test_apply_secondary_logic():
    from amforge.gui import postproc as PP
    ab = _fake_asbuilt()
    r = PP.apply_secondary(ab, treatment="HT",
                           params={"temperature_K": 1500.0, "hold_time_s": 7200.0})
    assert r["treatment"] == "HT"
    assert 0.0 < r["relief_factor"] < 1.0
    assert r["relaxed_vm"].shape == ab.sdf.shape
    # machining 缺 geometry 应报错
    with pytest.raises(ValueError):
        PP.apply_secondary(ab, treatment="machining")
