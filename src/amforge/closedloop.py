"""监测闭环编排（模块 D 外层迭代驱动）
====================================

``closedloop.run_loop`` 把"前向仿真 ↔ 在线监测 ↔ 工艺修正"卷成闭环：

    对每个迭代 i：
        1. 前向仿真 ``simulate(geo, plan)`` -> 热历史 + 服役判定；
        2. 传感流回放 ``ingest_raw_stream(thermal)`` -> SensorData；
        3. 缺陷检测   ``monitor.detect(sensor, thermal)`` -> MonitoringState；
        4. 工艺修正   ``monitor.correct(monitoring, plan)`` -> ClosedLoopPlan；
        5. （可选）在线重标定：收集传感派生测量，循环结束后回灌模块 C 标定材料；
        6. 取 ``ClosedLoopPlan.corrected`` 作为下一轮 plan。

**架构纪律**：闭环放在 ``Pipeline`` *外层*，``run_loop`` 只是反复调用前向求解器，
**不修改** ``Pipeline``/``graph`` 内核，保持纯前向 DAG 的纯洁（蓝图 §5 决策）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import jax.numpy as jnp

from amforge.core.contracts import (
    PartGeometry, ProcessPlan, SensorData, MonitoringState, ClosedLoopPlan,
    ServiceVerdict, CalibrationReport,
)
from amforge.monitoring import ingest_raw_stream, monitor_detect, monitor_correct
from amforge.inverse import simulate
from amforge.calibration import calibrate_material, CalibrationExperiment
from amforge.materials import get_material


@dataclass
class ClosedLoopStep:
    """单步闭环记录。"""

    iteration: int
    sensor: SensorData
    monitoring: MonitoringState
    corrected: ClosedLoopPlan
    verdict: ServiceVerdict
    defect_cost: float
    material: str = "316L"
    calibration: "CalibrationReport | None" = None

    def summary(self) -> str:
        name, prob = self.monitoring.max_defect()
        return (f"[{self.iteration}] top_defect={name}({float(prob):.3f}) "
                f"defect_cost={self.defect_cost:.3f} "
                f"verdict_sf={float(self.verdict.strength_safety_factor):.3f} "
                f"passed={bool(self.verdict.passed)} | {self.corrected.rationale}")


@dataclass
class ClosedLoopResult:
    """闭环总结果。"""

    steps: list[ClosedLoopStep] = field(default_factory=list)
    final_plan: "ProcessPlan | None" = None
    base_plan: "ProcessPlan | None" = None
    converged: bool = False
    iterations: int = 0
    calibration: "CalibrationReport | None" = None

    def defect_costs(self):
        return [s.defect_cost for s in self.steps]

    def summary(self) -> str:
        lines = [f"ClosedLoopResult(iterations={self.iterations}, "
                 f"converged={self.converged})"]
        for s in self.steps:
            lines.append("  " + s.summary())
        if self.calibration is not None:
            lines.append("  [online recalibration] -> "
                         + self.calibration.summary().replace("\n", " "))
        return "\n".join(lines)


def _sensor_experiment(plan: ProcessPlan, meltpool, sensor: SensorData):
    """由仿真熔池尺寸派生一条"传感测量"标定实验（含轻微传感-模型偏差以模拟真实硬件）。

    研究平台上前向仿真已用基准材料，故这里注入 ~3% 的系统偏差充当"真实传感与
    模型之间的差异"，使在线重标定产生非零修正、演示模块 C 回灌路径。
    """
    d = float(jnp.mean(jnp.atleast_1d(meltpool.depth))) * 1.03
    w = float(jnp.mean(jnp.atleast_1d(meltpool.width))) * 1.03
    exp_plan = ProcessPlan.uniform(
        n_layers=1, modality=plan.modality,
        laser_power=float(jnp.mean(jnp.atleast_1d(plan.laser_power))),
        scan_speed=float(jnp.mean(jnp.atleast_1d(plan.scan_speed))),
        layer_thickness=float(jnp.mean(jnp.atleast_1d(plan.layer_thickness))),
        hatch_spacing=float(jnp.mean(jnp.atleast_1d(plan.hatch_spacing))),
        beam_radius=float(jnp.mean(jnp.atleast_1d(plan.beam_radius))),
        absorption=float(jnp.mean(jnp.atleast_1d(plan.absorption))),
        preheat_temp=float(jnp.mean(jnp.atleast_1d(plan.preheat_temp))),
    )
    return CalibrationExperiment(process=exp_plan, measured_depth=d,
                                 measured_width=w,
                                 label=f"sensor_iter_{sensor.n_frames()}")


def run_loop(geo: PartGeometry, base_plan: ProcessPlan, monitor=None,
             n_iter: int = 3, *, material: str = "316L", params: Mapping = None,
             recalibrate: bool = False, simulate_fn: Callable | None = None,
             sensor_channels=("thermal", "meltpool", "photodiode"),
             tol: float = 0.05) -> ClosedLoopResult:
    """运行监测闭环，反复修正工艺直到缺陷代价低于 ``tol`` 或耗尽 ``n_iter``。

    参数
    ----
    geo, base_plan
        几何与起始（基准）工艺方案。
    monitor
        提供 ``detect`` / ``correct`` 的对象或模块（缺省用 ``amforge.monitoring``）。
    n_iter
        最大闭环迭代次数。
    material
        前向仿真与重标定使用的材料名。
    params
        透传给前向仿真/检测器/控制器的参数字典（含 ``n_grid`` 等网格控制）。
    recalibrate
        是否在循环结束后用累积的传感派生测量回灌模块 C（``calibrate_material``）。
    simulate_fn
        自定义前向仿真函数（缺省 ``inverse.simulate``）；签名为
        ``fn(geo, plan, *, material=, params=) -> dict``。
    tol
        缺陷代价（缺陷概率之和）收敛阈值。
    """
    mon = monitor if monitor is not None else _MonitorModule()
    sim = simulate_fn if simulate_fn is not None else simulate

    plan = base_plan
    steps: list[ClosedLoopStep] = []
    experiments: list[CalibrationExperiment] = []
    mat_name = material

    for i in range(max(1, int(n_iter))):
        step_params = dict(params or {})
        step_params["iteration"] = i
        step_params.setdefault("material", mat_name)

        out = sim(geo, plan, material=mat_name, params=step_params)
        thermal = out["thermal"]
        verdict = out["verdict"]
        meltpool = out.get("meltpool")

        sensor = ingest_raw_stream(thermal, channels=sensor_channels,
                                   geometry_name=geo.name, params=step_params)
        monitoring = mon.detect(sensor=sensor, thermal=thermal, params=step_params)
        corrected = mon.correct(monitoring=monitoring, process=plan,
                                params=step_params, geo=geo)

        defect_cost = float(jnp.sum(jnp.atleast_1d(monitoring.defect_probs)))

        if recalibrate and meltpool is not None:
            experiments.append(_sensor_experiment(plan, meltpool, sensor))

        steps.append(ClosedLoopStep(
            iteration=i, sensor=sensor, monitoring=monitoring,
            corrected=corrected, verdict=verdict, defect_cost=defect_cost,
            material=mat_name,
        ))

        plan = corrected.corrected

        if defect_cost <= tol:
            break

    # ---- 在线重标定：累积测量 -> 模块 C -----------------------------------
    calibration = None
    if recalibrate and experiments:
        calibration = calibrate_material(mat_name, experiments, n_iter=15,
                                         verbose=False)

    converged = bool(steps[-1].defect_cost <= tol) if steps else False
    return ClosedLoopResult(
        steps=steps, final_plan=plan, base_plan=base_plan,
        converged=converged, iterations=len(steps), calibration=calibration,
    )


class _MonitorModule:
    """默认监测器：转发到 ``amforge.monitoring`` 的 ``detect`` / ``correct``。"""

    detect = staticmethod(monitor_detect)
    correct = staticmethod(monitor_correct)
