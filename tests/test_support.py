"""模块A 支撑结构测试（M2）

守住：support STAGE 接入、SupportStructure 契约、support.auto 规则布点、
support.simulate 复用 asbuilt 热-力耦合、双注册表镜像一致性、pytree roundtrip。
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import amforge as af
from amforge.core.contracts import (
    AsBuiltPart, PartGeometry, SupportStructure, ThermalHistory, solid_mask,
)
from amforge.core.registry import get_solver
from amforge.support import support_auto, support_simulate
from amforge.asbuilt_plastic import solve_asbuilt_plastic
from amforge.process import ProcessPlan
from forgecore.registry import REGISTRY

import amforge.forge_adapter  # 触发双注册表镜像（填充 ForgeCore REGISTRY）


# ---------------------------------------------------------------------------
# 几何辅助
# ---------------------------------------------------------------------------
def _box_sdf(center, half, p):
    q = jnp.abs(p - center) - half
    return jnp.linalg.norm(jnp.maximum(q, 0.0), axis=-1) + jnp.minimum(jnp.max(q, axis=-1), 0.0)


def _make_grid(n=6, sp=2e-4):
    xs = jnp.linspace(-3 * sp, 3 * sp, n)
    ys = xs
    zs = jnp.linspace(0.0, 5 * sp, n)
    X, Y, Z = jnp.meshgrid(xs, ys, zs, indexing="ij")
    return jnp.stack([X, Y, Z], axis=-1), sp


def _box(center, half, n=6, sp=2e-4):
    P, sp = _make_grid(n, sp)
    sdf = _box_sdf(jnp.asarray(center, dtype=jnp.float64),
                   jnp.asarray(half, dtype=jnp.float64), P)
    return PartGeometry(sdf=sdf, origin=jnp.array([-3 * sp, -3 * sp, 0.0]),
                        spacing=sp, dim=3, name="box")


def _heated_thermal(geo, amp=600.0):
    # 口径＝#19 solid_mask（实测本组夹具 Δn=+4 个刀锋单元，见 am_t26_bare_sdf_census.log S1）
    occ = (solid_mask(geo.sdf) > 0.5).astype(jnp.float64)
    peak = 373.0 + amp * occ
    sh = geo.sdf.shape
    return ThermalHistory(peak_temperature=peak, cooling_rate=jnp.full(sh, 1e3),
                         thermal_gradient=jnp.full(sh, 1e5),
                         solidification_rate=jnp.full(sh, 1e-2),
                         time_above_melt=jnp.full(sh, 0.1),
                         final_temperature=peak, spacing=float(geo.spacing), dim=geo.dim)


# ---------------------------------------------------------------------------
# support.auto：规则布点
# ---------------------------------------------------------------------------
def test_support_auto_on_plate_needs_none():
    geo = _box([0., 0., 2.5e-4], [2e-4, 2e-4, 2.5e-4])  # 底面落在基板
    sup = support_auto(geometry=geo, params={"kind": "block"})
    assert isinstance(sup, SupportStructure)
    assert float(sup.volume_fraction) < 1e-9, "落在基板上的零件不应生成支撑"
    assert float(sup.contact_area) == 0.0


def test_support_auto_floating_needs_support():
    geo = _box([0., 0., 3.5e-4], [2e-4, 2e-4, 1.5e-4])  # 悬空，底面离基板有间隙
    sup = support_auto(geometry=geo, params={"kind": "block"})
    assert float(sup.volume_fraction) > 1e-3, "悬空零件应生成支撑"
    assert float(sup.contact_area) > 0.0
    # 支撑掩膜与零件不重叠。口径＝#19 solid_mask：旧写法 `geo.sdf < 0` 是**弱判据**
    # （刀锋层在规范口径下算实体却不被本检查覆盖）。实测本夹具 Δn=+4，而重叠和在两种
    # 口径下同为 0 ⇒ 换口径为可证无操作（am_t26_bare_sdf_census.log S3）。
    assert float(jnp.sum(sup.support_mask * (solid_mask(geo.sdf) > 0.5).astype(jnp.float64))) == 0.0


def test_support_auto_overhang_subset_of_block():
    geo = _box([0., 0., 3.5e-4], [2e-4, 2e-4, 1.5e-4])
    b = support_auto(geometry=geo, params={"kind": "block"})
    o = support_auto(geometry=geo, params={"kind": "overhang"})
    # overhang 模式只在悬垂下方布点，体积占比不应多于 block
    assert float(o.volume_fraction) <= float(b.volume_fraction) + 1e-12


# ---------------------------------------------------------------------------
# support.simulate：复用 asbuilt 热-力耦合
# ---------------------------------------------------------------------------
def test_support_simulate_empty_equals_baseline():
    """空支撑时 support.simulate 必须等价于直接跑 asbuilt 塑性求解（合并几何=零件）。"""
    geo = _box([0., 0., 3.5e-4], [2e-4, 2e-4, 1.5e-4])
    th = _heated_thermal(geo)
    plan = ProcessPlan.uniform(n_layers=1, laser_power=200.0, scan_speed=1.0,
                              beam_radius=50e-6, absorption=0.4)
    ab_base = solve_asbuilt_plastic(geometry=geo, thermal=th,
                                    params={"material": "316L", "max_layers": 6})
    empty = SupportStructure(
        support_sdf=jnp.full(geo.sdf.shape, 0.5 * geo.spacing),
        support_mask=jnp.zeros(geo.sdf.shape),
        kind="block", contact_area=jnp.array(0.0), volume_fraction=jnp.array(0.0),
        params={})
    ab = support_simulate(geometry=geo, support=empty, process=plan,
                         params={"material": "316L", "max_layers": 6, "thermal": th})
    assert isinstance(ab, AsBuiltPart)
    diff = jnp.max(jnp.abs(ab.displacement - ab_base.displacement))
    assert float(diff) < 1e-12, "空支撑应与基线逐点相同"


def test_support_simulate_couples_and_zeroes_support():
    """非空支撑应改变耦合结果，且输出中支撑 voxel 的位移被遮罩为零。"""
    geo = _box([0., 0., 3.5e-4], [2e-4, 2e-4, 1.5e-4])
    th = _heated_thermal(geo)
    plan = ProcessPlan.uniform(n_layers=1, laser_power=200.0, scan_speed=1.0,
                              beam_radius=50e-6, absorption=0.4)
    sup = support_auto(geometry=geo, params={"kind": "block"})
    ab_base = solve_asbuilt_plastic(geometry=geo, thermal=th,
                                    params={"material": "316L", "max_layers": 6})
    ab = support_simulate(geometry=geo, support=sup, process=plan,
                         params={"material": "316L", "max_layers": 6, "thermal": th})
    # 支撑改变了合并几何的求解 → 与无支撑基线不同
    assert float(jnp.max(jnp.abs(ab.displacement - ab_base.displacement))) > 1e-12
    # 输出只含零件区域：支撑专用 voxel（支撑=1 且零件=0）位移应为 0
    sup_only = (sup.support_mask > 0.5) & (~(solid_mask(geo.sdf) > 0.5))
    leaked = jnp.sum(jnp.abs(ab.displacement) * sup_only[..., None])
    assert float(leaked) == 0.0


# ---------------------------------------------------------------------------
# pytree roundtrip
# ---------------------------------------------------------------------------
def test_supportstructure_pytree_roundtrip():
    geo = _box([0., 0., 3.5e-4], [2e-4, 2e-4, 1.5e-4])
    sup = support_auto(geometry=geo, params={"kind": "overhang", "density": 0.5})
    import jax
    out = jax.tree_util.tree_map(lambda x: x, sup)
    assert out.support_sdf.shape == sup.support_sdf.shape
    assert out.kind == sup.kind
    assert dict(out.params) == dict(sup.params)
    assert float(out.volume_fraction) == float(sup.volume_fraction)


# ---------------------------------------------------------------------------
# 双注册表一致性
# ---------------------------------------------------------------------------
def test_mirror_support_registered_in_both():
    # 原生 amforge 注册表
    assert get_solver("support.auto") is not None
    assert get_solver("support.simulate") is not None
    # ForgeCore 镜像注册表
    assert REGISTRY.get("support.auto") is not None
    assert REGISTRY.get("support.simulate") is not None
    # 高保真档带 high-fidelity 标签，且 cost 来自原生（单一事实源）
    spec = REGISTRY.get("support.simulate")
    assert any("high-fidelity" in str(t) for t in spec.tags)
    assert spec.cost == 50.0


def test_default_asbuilt_pipeline_avoids_support_simulate():
    """默认 AsBuiltPart 链路不应自动选高保真 support.simulate（cost=50）。"""
    pipe = af.Pipeline.auto("AsBuiltPart", modality="SLM")
    assert "support.simulate" not in pipe.solver_names()


def test_supportstructure_target_routes_to_support_auto():
    pipe = af.Pipeline.auto("SupportStructure", given=("PartGeometry",))
    assert "support.auto" in pipe.solver_names()
    assert pipe.steps[-1].produces == "SupportStructure"
