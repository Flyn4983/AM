"""在线监测 + 缺陷检测（模块 D 求解器）
====================================

把"真实硬件在线传感"与"前向仿真"在闭环里对接的两只求解器：

* ``ingest_raw_stream``  —— 把原始传感采集（研究平台上由 ThermalHistory 合成，
  生产上由光电/热成像/熔池相机字节流）封装为 :class:`SensorData` 契约；
* ``monitor.detect``     —— 从传感流 + 热历史推断缺陷概率
  （未熔合/匙孔/孔隙），产出 :class:`MonitoringState`；
* ``monitor.correct``    —— 基于缺陷状态给出修正后的工艺方案，产出
  :class:`ClosedLoopPlan`（外层 ``closedloop.run_loop`` 把它喂回前向仿真）。

求解器均为非可微（``differentiable=False``）：检测器/控制器通常是 NN 或规则
控制器，不参与 ``jax.grad`` 链路；可微的"工艺→缺陷"反演仍由 ``inverse`` 承担。
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from amforge.core.contracts import (
    SensorData, MonitoringState, ClosedLoopPlan, ThermalHistory, ProcessPlan,
)
from amforge.core.registry import register_solver
from amforge.materials import get_material
from amforge.process import normalize_process, denormalize_process
from amforge.inverse import (
    simulate, project_process_feasible, thermal_tier, tier_bounds,
)


# ---------------------------------------------------------------------------
# 传感流封装：原始采集 -> SensorData
# ---------------------------------------------------------------------------
def ingest_raw_stream(raw, *, channels=("thermal", "meltpool", "photodiode"),
                      geometry_name: str = "part", params=None) -> SensorData:
    """把原始传感采集封装成 :class:`SensorData` 契约。

    研究平台默认把仿真热历史当成"实测"传感流来演示闭环；生产环境把同样的接口
    接到真实相机/光电二极管字节流即可。

    参数
    ----
    raw
        可为以下任一种：

        * :class:`ThermalHistory` —— 由温度场合成多通道传感时间序列；
        * ``(frames, time)`` 元组 —— 直接给定 ``(n_channels, n_frames)`` 帧与
          时间戳；
        * ``dict`` —— 显式 ``{"frames":..., "time":..., "dt":...}``。
    channels
        通道顺序（对应合成帧的第 0 轴）。
    """
    if isinstance(raw, ThermalHistory):
        return _thermal_to_sensor(raw, channels=channels, geometry_name=geometry_name)

    if isinstance(raw, dict):
        frames = jnp.asarray(raw["frames"], dtype=jnp.float64)
        time = jnp.asarray(raw["time"], dtype=jnp.float64)
        dt = float(raw.get("dt", 0.0))
        meta = dict(raw.get("sensor_meta", {}) or {})
        return SensorData(frames=frames, time=time, channels=tuple(channels),
                          dt=dt, geometry_name=geometry_name, sensor_meta=meta)

    # (frames, time) 元组
    frames, time = raw
    frames = jnp.asarray(frames, dtype=jnp.float64)
    time = jnp.asarray(time, dtype=jnp.float64)
    dt = float(time[1] - time[0]) if time.shape[0] > 1 else 0.0
    return SensorData(frames=frames, time=time, channels=tuple(channels),
                      dt=dt, geometry_name=geometry_name, sensor_meta={})


def _thermal_to_sensor(thermal: ThermalHistory, *, channels, geometry_name) -> SensorData:
    """由 ThermalHistory 合成多通道传感时间序列（每通道为每帧标量聚合）。

    通道约定（与蓝图一致）：
      * ``thermal``    —— 每帧平均温度（熔池热循环）；
      * ``meltpool``   —— 每帧熔合体积分数（>熔点体素占比）；
      * ``photodiode`` —— 每帧温度梯度相干（熔化/凝固跃变的代理信号）。
    """
    T_l = get_material("316L").T_liquidus  # 合成用材料熔点（仅作阈值）
    spacing = float(thermal.spacing)

    evo = thermal.temperature_evolution
    if evo is None:
        # 未开启帧记录：用峰值温度 + 轻噪声构造单帧回放
        single = thermal.peak_temperature
        evo = jnp.asarray(single, dtype=jnp.float64)[jnp.newaxis, ...]

    n_frames = int(evo.shape[0])
    flat = evo.reshape(n_frames, -1)                       # (n_frames, N)
    mean_T = jnp.mean(flat, axis=1)                        # 每帧平均温度
    melted = jnp.mean((flat > T_l).astype(jnp.float64), axis=1)  # 熔合体积分数
    grad = jnp.abs(flat[:, 1:] - flat[:, :-1])
    photodiode = jnp.mean(grad, axis=1)                    # 温度梯度相干

    # 通道堆叠为 (n_channels, n_frames)
    frames = jnp.stack(
        [mean_T / jnp.maximum(T_l, 1.0),   # 归一化热信号
         melted,
         photodiode / jnp.maximum(jnp.mean(photodiode), 1e-9)],
        axis=0,
    )
    time = jnp.arange(n_frames, dtype=jnp.float64) * spacing
    dt = float(spacing)
    meta = {"source": "thermal_history_synthesis", "T_liquidus": float(T_l),
            "spacing": spacing}
    return SensorData(frames=frames, time=time, channels=tuple(channels), dt=dt,
                      geometry_name=geometry_name, sensor_meta=meta)


# ---------------------------------------------------------------------------
# 求解器 1：缺陷检测  SensorData + ThermalHistory -> MonitoringState
# ---------------------------------------------------------------------------
@register_solver(
    "monitor.detect",
    consumes=("SensorData", "ThermalHistory"),
    produces="MonitoringState",
    stage="monitor",
    modality=(),
    differentiable=False,
    cost=3.0,
    defaults={"material": "316L", "lof_thr": 0.10, "key_thr": 0.10,
              "por_thr": 0.10},
    doc="从传感流+热历史推断未熔合/匙孔/孔隙缺陷概率（NN/统计分类缺陷指示）",
)
def monitor_detect(*, sensor: SensorData, thermal: ThermalHistory,
                   params) -> MonitoringState:
    """缺陷检测：把传感流与热历史映射为缺陷概率向量。

    采用物理启发的代理判据（研究平台可解释、零外部依赖；生产上可换成训练好的
    NN 分类器，接口不变）：

    * 未熔合（lack_of_fusion）—— 峰值温度低于熔点的体素占比；
    * 匙孔（keyhole）        —— 峰值温度显著高于熔点的过熔/深穿透占比；
    * 孔隙（porosity）       —— 低 G/R 凝固形貌（柱状过窄→气孔倾向）占比。
    """
    p = dict(params or {})
    mat = get_material(p.get("material", "316L"))
    T_l = mat.T_liquidus
    peak = jnp.atleast_1d(thermal.peak_temperature)

    # ---- 缺陷代理（全部 ∈ [0,1]） ----------------------------------------
    lof = jnp.mean((peak < T_l).astype(jnp.float64))            # 未熔合
    key = jnp.mean((peak > 1.4 * T_l).astype(jnp.float64))      # 匙孔
    gr = jnp.atleast_1d(thermal.gr_ratio())                    # G/R 凝固形貌
    por = jnp.mean((gr < 1.0e3).astype(jnp.float64))           # 孔隙倾向

    probs = jnp.clip(jnp.stack([lof, key, por]), 0.0, 1.0)

    # ---- 检测置信度：由传感信号能量估计（缺省统一置信） ------------------
    sig = jnp.mean(jnp.abs(jnp.atleast_1d(sensor.frames)))
    conf = jnp.clip(jnp.ones(3) * jnp.minimum(1.0, sig), 0.05, 1.0)

    # ---- 缺陷空间位置：以时间戳为扫描坐标（1D 位置） --------------------
    n_frames = sensor.n_frames()
    position = jnp.arange(n_frames, dtype=jnp.float64)[:, None] * float(sensor.dt)

    return MonitoringState(
        defect_probs=probs,
        defect_names=("lack_of_fusion", "keyhole", "porosity"),
        confidence=conf,
        position=position,
        sensor_ref=sensor.geometry_name,
        model_meta={"detector": "physics_proxy", "T_liquidus": float(T_l)},
    )


# ---------------------------------------------------------------------------
# 可微缺陷代价（供 inverse 控制器做梯度优化；与 monitor.detect 的物理代理同判据
# 但全程光滑可微，可被 jax.grad 穿透）
# ---------------------------------------------------------------------------
def _smooth_defect_cost(thermal: ThermalHistory, mat) -> jnp.ndarray:
    """缺陷代价的光滑可微代理（与 :func:`monitor_detect` 物理判据一致）。

    * 未熔合：峰值温度低于熔点 -> ``softplus(T_l - peak)``；
    * 匙孔：  峰值温度显著高于熔点 -> ``softplus(peak - 1.4 T_l)``；
    * 孔隙：  低 G/R 凝固形貌 -> ``softplus(1e3 - gr)``。

    全部除以各自尺度使量级 ~O(1)，便于 Adam 稳定下降。
    """
    T_l = float(mat.T_liquidus)
    peak = jnp.atleast_1d(thermal.peak_temperature)
    lof = jax.nn.softplus(T_l - peak) / jnp.maximum(T_l, 1.0)
    key = jax.nn.softplus(peak - 1.4 * T_l) / jnp.maximum(T_l, 1.0)
    gr = jnp.atleast_1d(thermal.gr_ratio())
    por = jax.nn.softplus(1.0e3 - gr) / 1.0e3
    return jnp.mean(lof) + jnp.mean(key) + jnp.mean(por)


# ---------------------------------------------------------------------------
# 真实控制器：把"缺陷代价最小化"作为目标，用 inverse 优化器反向优化工艺杠杆
# ---------------------------------------------------------------------------
def monitor_correct_inverse(*, monitoring: MonitoringState, process: ProcessPlan,
                             params, geo, material: str = "316L",
                             n_steps: int = 8, learning_rate: float = 0.05):
    """**实打实**的工艺修正控制器：可微优化而非规则代理。

    复用 :mod:`amforge.inverse` 的成熟模式——把工艺归一化到 ``z∈[0,1]^9``，
    用真实前向仿真 ``simulate`` 预测热历史，以光滑缺陷代价为目标跑 Adam 梯度
    下降，每步硬投影回设备可行域（``project_process_feasible``）。这与蓝图 §5
    "复用 digitaltwin.rom + inverse 优化器"的意图一致：用真实物理链驱动闭环控制，
    而不是手写 PID 规则。

    参数
    ----
    geo
        几何（必须）：控制器要在其上跑前向仿真评估缺陷代价。闭环 ``run_loop``
        会在调用时传入；若缺失则抛出 ``ValueError``（不改变规则代理的可用路径）。
    n_steps, learning_rate
        内层 Adam 步数与学习率。
    """
    if geo is None:
        raise ValueError(
            "inverse 控制器需要几何 geo 以运行前向仿真；未提供时请用 "
            "controller='rule' 规则代理，或在 run_loop 中由外层传入 geo。")
    mat = get_material(material)
    nl = int(jnp.atleast_1d(process.laser_power).shape[0])
    modality = process.modality

    # 透传但不把控制参数泄漏给仿真
    sim_p = {k: v for k, v in (dict(params or {})).items()
             if k not in ("controller", "iteration", "material", "geo")}
    sim_p.setdefault("n_grid", 6)
    sim_p.setdefault("max_layers", 3)
    sim_p.setdefault("n_sub_cp", 8)
    sim_p["micro"] = dict(sim_p.get("micro", {}))
    # 显式热解的静态调度必须在 trace 之外钉好（D0）；控制器只能在钉住的工艺子盒内寻优
    sim_p = thermal_tier(sim_p, geo, process, material=material)
    bnds = tier_bounds(sim_p)

    base_z = normalize_process(process, bounds=bnds)
    z = project_process_feasible(base_z, material=material, n_layers=nl,
                                 modality=modality, params=sim_p)

    opt = optax.adam(learning_rate)
    opt_state = opt.init(z)

    # 几何作为闭包常量保持具体（不 jit），与 inverse.optimize_dimensional 同纪律
    def loss_of_z(zz):
        plan = denormalize_process(zz, n_layers=nl, modality=modality, bounds=bnds)
        out = simulate(geo, plan, material=material, params=sim_p,
                       asbuilt_solver="buildup", micro_solver="surrogate",
                       mbd_solver="surrogate")
        return _smooth_defect_cost(out["thermal"], mat)

    loss_and_grad = jax.value_and_grad(loss_of_z)
    last_loss = None
    for _ in range(n_steps):
        loss, grads = loss_and_grad(z)
        updates, opt_state = opt.update(grads, opt_state)
        z = optax.apply_updates(z, updates)
        z = jnp.clip(z, 0.0, 1.0)
        z = project_process_feasible(z, material=material, n_layers=nl,
                                     modality=modality, params=sim_p)
        last_loss = float(loss)

    corrected = denormalize_process(z, n_layers=nl, modality=modality, bounds=bnds)
    P0 = float(jnp.mean(jnp.atleast_1d(process.laser_power)))
    P1 = float(jnp.mean(jnp.atleast_1d(corrected.laser_power)))
    V1 = float(jnp.mean(jnp.atleast_1d(corrected.scan_speed)))
    H1 = float(jnp.mean(jnp.atleast_1d(corrected.hatch_spacing)))
    deltas = (
        ("laser_power", P1),
        ("scan_speed", V1),
        ("hatch_spacing", H1),
    )
    rationale = (f"inverse-opt: 缺陷代价 {last_loss:.3e} (ΔP={P1 - P0:+.1f}W) "
                 f"via {n_steps} Adam steps on real forward sim")
    return ClosedLoopPlan(
        base=process, corrected=corrected, deltas=deltas,
        iteration=int((dict(params or {})).get("iteration", 0)), rationale=rationale,
        controller="inverse",
    )


# ---------------------------------------------------------------------------
# 求解器 2：工艺修正  MonitoringState + ProcessPlan -> ClosedLoopPlan
# ---------------------------------------------------------------------------
@register_solver(
    "monitor.correct",
    consumes=("MonitoringState", "ProcessPlan"),
    produces="ClosedLoopPlan",
    stage="monitor",
    modality=(),
    differentiable=False,
    cost=2.0,
    defaults={"control_gain": 0.15, "lof_thr": 0.10, "key_thr": 0.10,
              "por_thr": 0.10},
    doc="基于缺陷状态给出修正后工艺方案（控制器输出修正工艺，闭环喂回前向仿真）",
)
def monitor_correct(*, monitoring: MonitoringState, process: ProcessPlan,
                    params, geo=None) -> ClosedLoopPlan:
    """工艺修正控制器：缺陷概率高 → 朝降低方向调整工艺杠杆。

    控制器分两档（由 ``params["controller"]`` 选择）：

    * ``"rule"``（默认）—— 研究平台可解释规则控制器（物理代理）：
      未熔合偏高 → 提功率降速度；匙孔偏高 → 降功率提速度；孔隙偏高 → 调间距。
    * ``"inverse"`` —— **实打实**可微优化控制器（见 :func:`monitor_correct_inverse`）：
      把缺陷代价作为目标，用真实前向仿真 + Adam 梯度下降 + 可行域硬投影反向
      优化工艺杠杆；需要 ``geo``（闭环 ``run_loop`` 会传入）。

    修正后工艺封装进 :class:`ClosedLoopPlan`（``base`` + ``corrected`` + ``deltas``），
    由 ``closedloop.run_loop`` 取 ``corrected`` 喂回前向仿真。
    """
    p = dict(params or {})
    if p.get("controller", "rule") == "inverse":
        return monitor_correct_inverse(monitoring=monitoring, process=process,
                                       params=params, geo=geo,
                                       material=p.get("material", "316L"))

    gain = float(p.get("control_gain", 0.15))

    names = monitoring.defect_names or ("lack_of_fusion", "keyhole", "porosity")
    probs = jnp.atleast_1d(monitoring.defect_probs)
    idx = {n: i for i, n in enumerate(names)}
    lof = float(probs[idx.get("lack_of_fusion", 0)])
    key = float(probs[idx.get("keyhole", 1 % len(names))])
    por = float(probs[idx.get("porosity", min(2, len(names) - 1))])

    # ---- 标量均值工艺杠杆（控制器作用于层均值） --------------------------
    P = float(jnp.mean(jnp.atleast_1d(process.laser_power)))
    V = float(jnp.mean(jnp.atleast_1d(process.scan_speed)))
    H = float(jnp.mean(jnp.atleast_1d(process.hatch_spacing)))
    B = float(jnp.mean(jnp.atleast_1d(process.beam_radius)))

    dP = gain * (lof - 0.10) - gain * (key - 0.10)     # 增功率(未熔合)/降功率(匙孔)
    dV = -gain * (lof - 0.10) + gain * (key - 0.10)    # 降速度(未熔合)/提速度(匙孔)
    dH = -gain * (por - 0.10)                          # 调扫描间距(孔隙)

    new_P = float(jnp.clip(P * (1.0 + dP), 20.0, 2000.0))
    new_V = float(jnp.clip(V * (1.0 + dV), 0.05, 20.0))
    new_H = float(jnp.clip(H * (1.0 + dH), 10e-6, 2e-3))

    corrected = process.replace(
        laser_power=jnp.asarray(new_P, dtype=jnp.float64),
        scan_speed=jnp.asarray(new_V, dtype=jnp.float64),
        hatch_spacing=jnp.asarray(new_H, dtype=jnp.float64),
        beam_radius=jnp.asarray(B, dtype=jnp.float64),
    )
    deltas = (
        ("laser_power", new_P),
        ("scan_speed", new_V),
        ("hatch_spacing", new_H),
    )
    rationale = (f"lof={lof:.2f} key={key:.2f} por={por:.2f} -> "
                 f"dP={dP:+.3f} dV={dV:+.3f} dH={dH:+.3f}")
    return ClosedLoopPlan(
        base=process, corrected=corrected, deltas=deltas,
        iteration=int(p.get("iteration", 0)), rationale=rationale,
        controller="rule",
    )
