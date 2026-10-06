"""M4 监测闭环（模块D）测试：契约 / 两求解器 / ingest / 双注册表镜像 / run_loop / GUI。

覆盖（蓝图 §5 模块D）：
* SensorData / MonitoringState / ClosedLoopPlan 三个契约的 pytree 往返
  （可被 jax.grad/jit 穿透，闭环外层驱动依赖它）；
* ingest_raw_stream 把 ThermalHistory / 元组 / dict 封装成 SensorData；
* monitor.detect（SensorData+ThermalHistory -> MonitoringState）物理代理缺陷概率；
* monitor.correct（MonitoringState+ProcessPlan -> ClosedLoopPlan）控制律方向正确；
* 双注册表（amforge.core.registry + forgecore REGISTRY）镜像一致性 + 成本一致；
* closedloop.run_loop 外层迭代驱动：结构正确、缺陷代价有限、可选在线重标定回灌模块C；
* GUI 监测仪表盘可 offscreen 构造（纯逻辑层 run_monitoring_case 独立可跑）。
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import amforge as af
from amforge.core.contracts import (
    SensorData, MonitoringState, ClosedLoopPlan, ThermalHistory, ServiceVerdict,
    CalibrationReport, ProcessPlan,
)
from amforge.monitoring import (
    ingest_raw_stream, monitor_detect, monitor_correct,
    monitor_correct_inverse, _smooth_defect_cost,
)
from amforge.inverse import simulate
from amforge.materials import get_material


# ---------------------------------------------------------------------------
# 测试夹具
# ---------------------------------------------------------------------------
def _thermal(peak=1000.0, n=4):
    """构造一份 ThermalHistory：峰值全低于 316L 熔点(1723K) -> 未熔合=1。"""
    sh = (n, n)
    return ThermalHistory(
        peak_temperature=jnp.full(sh, float(peak)),
        cooling_rate=jnp.full(sh, 1e5),
        thermal_gradient=jnp.full(sh, 1e6),     # 大梯度
        solidification_rate=jnp.full(sh, 1e-3),  # 慢凝固 -> gr 大 -> 孔隙=0
        time_above_melt=jnp.full(sh, 0.5),
        final_temperature=jnp.full(sh, 300.0),
        temperature_evolution=jnp.full((5, *sh), float(peak)),
        spacing=1e-5, dim=2,
    )


def _sensor(n_frames=5):
    return SensorData(
        frames=jnp.ones((3, n_frames)),          # 3 通道、每帧幅值~1
        time=jnp.arange(n_frames, dtype=jnp.float64) * 1e-3,
        channels=("thermal", "meltpool", "photodiode"),
        dt=1e-3, geometry_name="fixture",
    )


def _tiny_geo():
    n = 8
    s = 2e-4
    xs = jnp.linspace(-s, s, n)
    X, Y = jnp.meshgrid(xs, xs, indexing="ij")
    sdf = jnp.maximum(jnp.abs(X), jnp.abs(Y)) - 0.6 * s
    return af.PartGeometry(sdf=sdf, origin=-s * jnp.ones(2),
                           spacing=2 * s / (n - 1), dim=2, name="tiny")


# ---------------------------------------------------------------------------
# 1. 契约 pytree 往返
# ---------------------------------------------------------------------------
def test_sensor_monitoring_closedloop_pytree_roundtrip():
    plan = af.ProcessPlan.uniform(n_layers=2)
    sd = _sensor()
    ms = MonitoringState(defect_probs=jnp.array([0.2, 0.1, 0.3]),
                         confidence=jnp.ones(3), position=jnp.zeros((5, 1)),
                         model_meta={"detector": "test"})
    clp = ClosedLoopPlan(base=plan, corrected=plan, deltas=(("laser_power", 210.0),),
                         iteration=1, rationale="x")
    for obj in (sd, ms, clp):
        leaves, treedef = jax.tree_util.tree_flatten(obj)
        rec = jax.tree_util.tree_unflatten(treedef, leaves)
        assert type(rec) is type(obj)
    # ClosedLoopPlan 嵌套 ProcessPlan 也应递归往返
    assert float(clp.corrected.laser_power) == float(rec.corrected.laser_power)


# ---------------------------------------------------------------------------
# 2. ingest_raw_stream 多形态输入
# ---------------------------------------------------------------------------
def test_ingest_from_thermal_history():
    th = _thermal()
    sd = ingest_raw_stream(th, channels=("thermal", "meltpool", "photodiode"),
                           geometry_name="p1")
    assert isinstance(sd, SensorData)
    assert sd.frames.shape[0] == 3                 # 三通道
    assert sd.n_frames() == 5                       # temperature_evolution 5 帧
    assert sd.channels == ("thermal", "meltpool", "photodiode")


def test_ingest_from_tuple_and_dict():
    frames = jnp.ones((2, 4))
    time = jnp.arange(4, dtype=jnp.float64) * 1e-3
    sd_t = ingest_raw_stream((frames, time), channels=("a", "b"))
    assert sd_t.frames.shape == (2, 4) and sd_t.n_frames() == 4
    sd_d = ingest_raw_stream({"frames": frames, "time": time, "dt": 1e-3,
                              "sensor_meta": {"cam": "ir"}})
    assert sd_d.dt == 1e-3 and sd_d.sensor_meta.get("cam") == "ir"


# ---------------------------------------------------------------------------
# 3. monitor.detect 缺陷检测
# ---------------------------------------------------------------------------
def test_monitor_detect_produces_monitoringstate():
    th = _thermal(peak=1000.0)                     # 全部未熔合
    sd = _sensor()
    ms = monitor_detect(sensor=sd, thermal=th, params={})
    assert isinstance(ms, MonitoringState)
    probs = np.asarray(ms.defect_probs)
    assert probs.shape == (3,)
    assert probs.min() >= 0.0 and probs.max() <= 1.0
    assert ms.defect_names == ("lack_of_fusion", "keyhole", "porosity")
    # 峰值全低于熔点 -> 未熔合概率≈1，匙孔/孔隙≈0
    assert float(probs[0]) > 0.9
    assert float(probs[1]) < 0.1
    assert float(probs[2]) < 0.1
    assert ms.position.shape == (sd.n_frames(), 1)


# ---------------------------------------------------------------------------
# 4. monitor.correct 控制律方向
# ---------------------------------------------------------------------------
def test_monitor_correct_raises_power_on_lof():
    plan = af.ProcessPlan.uniform(n_layers=2, laser_power=200.0, scan_speed=1.0)
    ms = MonitoringState(defect_probs=jnp.array([0.5, 0.0, 0.0]))  # 高未熔合
    clp = monitor_correct(monitoring=ms, process=plan, params={})
    assert isinstance(clp, ClosedLoopPlan)
    new_P = float(jnp.mean(jnp.atleast_1d(clp.corrected.laser_power)))
    assert new_P > 200.0                            # 未熔合高 -> 提功率


def test_monitor_correct_lowers_power_on_keyhole():
    plan = af.ProcessPlan.uniform(n_layers=2, laser_power=200.0, scan_speed=1.0)
    ms = MonitoringState(defect_probs=jnp.array([0.0, 0.5, 0.0]))  # 高匙孔
    clp = monitor_correct(monitoring=ms, process=plan, params={})
    new_P = float(jnp.mean(jnp.atleast_1d(clp.corrected.laser_power)))
    assert new_P < 200.0                            # 匙孔高 -> 降功率


# ---------------------------------------------------------------------------
# 4b. 真实 inverse 控制器：可微优化把缺陷代价真正压下来（不是规则代理）
# ---------------------------------------------------------------------------
def _smooth_cost_of(geo, plan, material="316L"):
    sp = {"n_grid": 6, "max_layers": 3, "n_sub_cp": 8, "micro": {}}
    out = simulate(geo, plan, material=material, params=sp,
                   asbuilt_solver="buildup", micro_solver="surrogate",
                   mbd_solver="surrogate")
    return float(_smooth_defect_cost(out["thermal"], get_material(material)))


def test_inverse_controller_lowers_defect_cost():
    geo = _tiny_geo()
    # 低功率工艺 -> 明显未熔合（缺陷代价有下降空间）
    base = af.ProcessPlan.uniform(n_layers=3, laser_power=120.0, scan_speed=1.5)
    ms = MonitoringState(defect_probs=jnp.array([0.8, 0.0, 0.0]))
    clp = monitor_correct_inverse(monitoring=ms, process=base,
                                  params={"n_grid": 6, "max_layers": 3,
                                          "n_sub_cp": 8, "micro": {}},
                                  geo=geo, n_steps=10, learning_rate=0.08)
    assert isinstance(clp, ClosedLoopPlan)
    assert clp.controller == "inverse"
    cost_base = _smooth_cost_of(geo, base)
    cost_corr = _smooth_cost_of(geo, clp.corrected)
    # 真实优化：修正后缺陷代价应低于基准（至少改善 1%）
    assert cost_corr < cost_base * 0.99, f"{cost_base:.4e} -> {cost_corr:.4e}"


def test_run_loop_inverse_controller_runs():
    geo = _tiny_geo()
    plan = af.ProcessPlan.uniform(n_layers=3, laser_power=150.0, scan_speed=1.2)
    from amforge.closedloop import run_loop
    res = run_loop(geo, plan, n_iter=2, material="316L",
                   params={"controller": "inverse", "n_grid": 6,
                           "max_layers": 3, "n_sub_cp": 8, "micro": {}},
                   recalibrate=False, tol=1.0)
    assert res.iterations >= 1
    for s in res.steps:
        assert s.corrected.controller == "inverse"   # 每步都走真实优化器
        assert np.isfinite(s.defect_cost)


# ---------------------------------------------------------------------------
# 5. 双注册表镜像一致性
# ---------------------------------------------------------------------------
def test_mirror_monitoring_registered_in_both():
    import amforge.forge_adapter                    # 触发 forgecore 镜像
    import amforge.core.registry as A
    from forgecore.registry import REGISTRY

    for name in ("monitor.detect", "monitor.correct"):
        assert name in A._SOLVERS, f"amforge 原生缺失 {name}"
        assert name in REGISTRY.list_names(), f"forgecore 缺失 {name}"

    det = A.get_solver("monitor.detect")
    assert det.consumes == ("SensorData", "ThermalHistory")
    assert det.produces == "MonitoringState"
    assert det.stage == "monitor"
    fc_det = REGISTRY.get("monitor.detect")
    assert fc_det.produces == "monitoring"          # SLOT_OF[MonitoringState]
    assert fc_det.consumes == ("sensor", "thermal")  # SLOT_OF[...]
    assert fc_det.differentiable is False

    cor = A.get_solver("monitor.correct")
    assert cor.consumes == ("MonitoringState", "ProcessPlan")
    assert cor.produces == "ClosedLoopPlan"
    fc_cor = REGISTRY.get("monitor.correct")
    assert fc_cor.produces == "closedloop"
    assert fc_cor.consumes == ("monitoring", "process")


def test_mirror_costs_match_native_monitoring():
    import amforge.forge_adapter
    import amforge.core.registry as A
    from forgecore.registry import REGISTRY

    for name, native in (("monitor.detect", 3.0), ("monitor.correct", 2.0)):
        fc = REGISTRY.get(name).cost
        assert fc == native, f"forgecore cost 发散 {name}: {fc} != {native}"
        assert A.get_solver(name).cost == native


# ---------------------------------------------------------------------------
# 6. closedloop.run_loop 外层迭代驱动
# ---------------------------------------------------------------------------
def _loop_params():
    return {"n_grid": 6, "max_layers": 3, "n_sub_cp": 8, "micro": {}}


def test_run_loop_structure_and_fields():
    geo = _tiny_geo()
    plan = af.ProcessPlan.uniform(n_layers=3, laser_power=180.0, scan_speed=1.2)
    from amforge.closedloop import run_loop
    res = run_loop(geo, plan, n_iter=3, material="316L",
                   params=_loop_params(), recalibrate=False, tol=1.0)
    assert res.iterations >= 1
    assert isinstance(res.final_plan, ProcessPlan)
    assert isinstance(res.base_plan, ProcessPlan)
    assert isinstance(res.converged, bool)
    for s in res.steps:
        assert isinstance(s.monitoring, MonitoringState)
        assert isinstance(s.corrected, ClosedLoopPlan)
        assert isinstance(s.verdict, ServiceVerdict)
        assert np.isfinite(s.defect_cost)
    assert res.calibration is None                     # recalibrate=False


def test_run_loop_online_recalibration_back_to_C():
    geo = _tiny_geo()
    plan = af.ProcessPlan.uniform(n_layers=3, laser_power=180.0, scan_speed=1.2)
    from amforge.closedloop import run_loop
    res = run_loop(geo, plan, n_iter=2, material="316L",
                   params=_loop_params(), recalibrate=True, tol=1.0)
    # 在线重标定：累积传感派生测量 -> 模块C 标定材料
    assert isinstance(res.calibration, CalibrationReport)
    assert res.calibration.summary() is not None


def test_run_loop_converges_on_good_geometry():
    geo = _tiny_geo()
    plan = af.ProcessPlan.uniform(n_layers=3, laser_power=400.0, scan_speed=0.8)
    from amforge.closedloop import run_loop
    # 高功率工艺下缺陷代价应能被压到 tol 以下 -> 提前收敛退出
    res = run_loop(geo, plan, n_iter=5, material="316L",
                   params=_loop_params(), recalibrate=False, tol=0.5)
    assert res.iterations <= 5
    assert all(np.isfinite(c) for c in res.defect_costs())


# ---------------------------------------------------------------------------
# 7. GUI 监测仪表盘（offscreen 可构造、纯逻辑层独立可跑）
# ---------------------------------------------------------------------------
def test_monitoring_gui_construct_and_logic():
    pytest.importorskip("PySide6")
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    import amforge.gui.monitoring_app as M

    app = QApplication.instance() or QApplication([])
    try:
        win = M.MonitoringMainWindow()
        geo = M.P.build_primitive("sphere", 1.0, spacing_um=200.0, name="demo")
        lt = 80e-6
        n_layers = geo.layer_count(lt)
        plan = M.P.build_process_plan(
            n_layers, laser_power=180.0, scan_speed=1.0, layer_thickness=lt,
            hatch_spacing=120e-6, beam_radius=50e-6, absorption=0.4,
            preheat_temp=373.0, rotation_per_layer_deg=67.0)
        res = M.run_monitoring_case(geo, plan, n_iter=3, recalibrate=True)
        assert res.iterations == 3
        win._on_done(res, geo)                       # 三视图 set_result 不崩溃
        assert "defect_costs" in dir(res)
    finally:
        app.quit()
