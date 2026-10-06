"""材料标定（Module C）—— 复用可微优化内核的离线标定工具
=========================================================

把"材料物性参数"作为设计变量，用 ``jax.grad`` + L-BFGS 最小化

    损失(θ) = mean_i [ ((d_pred_i − d_meas_i)/d_meas_i)²
                       + ((w_pred_i − w_meas_i)/w_meas_i)² ]

其中 ``d_pred / w_pred`` 是把当前材料 ``rebuild(θ)`` 代入熔池代理
:func:`amforge.meltpool.solve_meltpool_surrogate` 得到的熔深/熔宽预测。
优化结果打包成 :class:`~amforge.core.contracts.CalibrationReport`
（含拟合材料、逐实验残差、RMSE、R²）。

这与用户"**实打实物理 + 误差报告**"的硬约束直接呼应：高保真求解器
（vof_flow3d / enthalpy / phasefield / thermomechanical_plastic）必须给出
"误差 ≤ X%" 的实证，否则再炫的链路面目可疑。本模块就是那条实证的入口。

对标商业软件（Simufact / ANSYS / AdditiveLab）的材料标定工作流：用单道 /
多道熔池实测宽深反推表观吸收率、热物性、Marangoni 系数等。

设计要点
--------
* 复用现有纯 JAX 熔池代理（它消费 ``AMMaterial`` 的 k/rho/cp/latent/T_*
  字段，且整条可微）；几何参数仅占位（代理不消费几何，只为满足契约签名）。
* 被标定材料每次迭代由 ``base.replace(**{fit_keys: θ})`` 重建，梯度沿
  ``θ → 材料场 → 熔池尺寸`` 一路回传。
* **离线工具，不进流水线**：输入是实验列表而非契约槽，故无需在
  ``forge_adapter`` 注册镜像（也见蓝图文档"一致性核对清单"）。

典型用法
--------
>>> from amforge.calibration import calibrate_material, CalibrationExperiment
>>> from amforge.process import ProcessPlan
>>> plan = ProcessPlan.uniform(n_layers=1, laser_power=200.0, scan_speed=1.0,
...                            beam_radius=50e-6, absorption=0.35)
>>> exps = [CalibrationExperiment(process=plan, measured_depth=1.2e-4,
...                               measured_width=2.1e-4, label="P200v1")]
>>> report = calibrate_material("316L", exps,
...                            fit_keys=("k_solid", "rho_solid", "cp_solid",
...                                      "latent_fusion"))
>>> print(report.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax

from amforge.core.contracts import (
    CalibrationReport,
    PartGeometry,
    ProcessPlan,
)
from amforge.materials import AMMaterial, get_material
from amforge.meltpool import solve_meltpool_surrogate


# ---------------------------------------------------------------------------
# 实验与占位几何
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CalibrationExperiment:
    """一条标定实验：工艺方案 + 实测熔深/熔宽。

    Attributes
    ----------
    process : ProcessPlan
        该实验对应的激光工艺（单道一般用 ``n_layers=1``）。
    measured_depth : float
        实测熔深 [m]。
    measured_width : float
        实测熔宽 [m]。
    coupon : PartGeometry | None
        占位几何（熔池代理不消费几何，留 ``None`` 即可自动用标准占位块）。
    label : str
        实验标签（报告里逐行对应）。
    """

    process: ProcessPlan
    measured_depth: float
    measured_width: float
    coupon: "PartGeometry | None" = None
    label: str = ""


def _coupon() -> PartGeometry:
    """标准占位块（10µm 立方，3×3×3 体素）。

    熔池代理不消费几何，这里只为满足 ``solve_meltpool_surrogate`` 的契约签名
    （``geometry=`` 关键字必填）。用极小网格避免任何开销。
    """
    s = 1.0e-4
    n = 3
    xs = jnp.linspace(-s, s, n)
    X, Y, Z = jnp.meshgrid(xs, xs, xs, indexing="ij")
    sdf = jnp.maximum(jnp.maximum(jnp.abs(X), jnp.abs(Y)), jnp.abs(Z)) - 0.9 * s
    return PartGeometry(
        sdf=sdf, origin=-s * jnp.ones(3), spacing=2.0 * s / (n - 1), dim=3,
        name="calibration-coupon",
    )


def process_from_dict(d: Mapping[str, float], *, modality: str = "SLM") -> ProcessPlan:
    """从标量字典构造单道工艺方案（标定实验常用）。

    未给出的键用工业界常用默认值；吸收率/光斑取自材料库经验值，可在字典里覆盖。
    """
    return ProcessPlan.uniform(
        n_layers=1, modality=modality,
        laser_power=float(d.get("laser_power", 200.0)),
        scan_speed=float(d.get("scan_speed", 1.0)),
        layer_thickness=float(d.get("layer_thickness", 40e-6)),
        hatch_spacing=float(d.get("hatch_spacing", 80e-6)),
        beam_radius=float(d.get("beam_radius", 50e-6)),
        absorption=float(d.get("absorption", 0.35)),
        preheat_temp=float(d.get("preheat_temp", 373.0)),
        powder_feed_rate=0.0,
        dwell_time=0.0,
    )


# ---------------------------------------------------------------------------
# 可微前向：材料(θ) -> (预测熔深, 预测熔宽)
# ---------------------------------------------------------------------------
def _predict_dw_mat(mat: AMMaterial, plans: Sequence[ProcessPlan],
                    n_grid: int = 16) -> tuple[jnp.ndarray, jnp.ndarray]:
    """对每条实验给出 (熔深预测, 熔宽预测) 的 1D 数组（可微上下文内调用）。

    不在此做 ``float()`` 转换 —— 否则在 ``value_and_grad`` 的 tracer 上会触发
    ``TracerArrayConversionError``。标量转换只在报告阶段（体外）进行。

    ``n_grid`` 控制代理采样网格（默认 16³，足够参数拟合且内存友好；标定不要求
    高保真网格，避免 ``value_and_grad`` 反向传播时显存爆炸）。
    """
    geo = _coupon()
    depths, widths = [], []
    for pl in plans:
        out = solve_meltpool_surrogate(
            geometry=geo, process=pl,
            params={"material": mat, "n_grid": int(n_grid)})
        depths.append(out.depth)
        widths.append(out.width)
    return jnp.stack(depths), jnp.stack(widths)


# ---------------------------------------------------------------------------
# 标定主入口
# ---------------------------------------------------------------------------
def calibrate_material(
    base_name: str,
    experiments: Sequence[CalibrationExperiment],
    *,
    fit_keys: Sequence[str] = ("k_solid", "rho_solid", "cp_solid", "latent_fusion"),
    method: str = "adam",
    n_iter: int = 200,
    scale_by_measurement: bool = True,
    verbose: bool = True,
    n_grid: int = 16,
) -> CalibrationReport:
    """标定材料物性以匹配实测熔池尺寸。

    参数
    ----
    base_name
        基准材料名（``AM_MATERIALS`` 中的键，如 ``"316L"``）。拟合只在其
        ``fit_keys`` 子集上动，其余物性沿用基准值。
    experiments
        :class:`CalibrationExperiment` 列表（至少 1 条，建议覆盖多种工艺窗口
        以提升可辨识性）。
    fit_keys
        被标定的物性字段名。建议选对熔池尺寸敏感且物理可测的字段
        （``k_solid`` / ``rho_solid`` / ``cp_solid`` / ``latent_fusion`` /
        ``T_solidus`` / ``T_liquidus``）。
    method
        优化器标识（保留以兼容后续扩展；当前实现固定使用 ``optax.adam``
        在**对数空间**参数化下进行，因本 JAX 构建中
        ``jax.scipy.optimize`` 的 L-BFGS 仅为 experimental 不稳定别名）。
    n_iter
        最大迭代步数（Adam 步数）。
    scale_by_measurement
        是否按实测值归一化残差（默认 True，使不同量纲实验同权、提升收敛）。
    n_grid
        代理采样网格分辨率（默认 16³）——标定是参数拟合，不需要高保真网格，
        低网格可避免 ``value_and_grad`` 反向传播显存爆炸。
    verbose
        是否在结束时打印 :meth:`CalibrationReport.summary`。

    返回
    ----
    :class:`CalibrationReport`（含拟合材料 + 逐实验残差 + RMSE + R²）。
    """
    base = get_material(base_name)
    keys = tuple(fit_keys)
    n_keys = len(keys)
    if n_keys == 0:
        raise ValueError("fit_keys 不能为空")

    theta0 = jnp.array(
        [float(getattr(base, k)) for k in keys], dtype=jnp.float64)
    # 对数空间参数化（正定参数）以平衡量纲差异、利于 Adam 收敛
    log0 = jnp.log(jnp.maximum(theta0, 1e-30))
    m_d = jnp.array(
        [float(e.measured_depth) for e in experiments], dtype=jnp.float64)
    m_w = jnp.array(
        [float(e.measured_width) for e in experiments], dtype=jnp.float64)
    plans = [e.process for e in experiments]
    labels = tuple((e.label or f"exp{i}") for i, e in enumerate(experiments))

    def rebuild_from_log(log_th):
        return base.replace(**{k: jnp.exp(log_th[i]) for i, k in enumerate(keys)})

    def loss(log_th):
        ds, ws = _predict_dw_mat(rebuild_from_log(log_th), plans, n_grid=n_grid)
        if scale_by_measurement:
            rd = (ds - m_d) / jnp.maximum(m_d, 1e-12)
            rw = (ws - m_w) / jnp.maximum(m_w, 1e-12)
        else:
            rd = ds - m_d
            rw = ws - m_w
        return jnp.mean(rd * rd + rw * rw)

    # Adam（复用 inverse 的 optax 优化内核；对数空间使各物性量纲平衡，
    # 避免 k~38 与 latent_fusion~3e5 量级悬殊导致的步长失衡）。
    # 注：当前 JAX 构建中 jax.scipy.optimize 的 L-BFGS 为 experimental 别名，
    # 不稳定；optax.adam 是 inverse 已验证的稳定选择。
    # 用 jax.jit 编译 loss+grad（代理前向在 300 步内只编译一次），
    # 相比每步重追踪可提速约 1~2 个数量级。
    opt = optax.adam(learning_rate=0.1)
    log_th = log0
    opt_state = opt.init(log_th)
    loss_and_grad = jax.jit(jax.value_and_grad(loss))
    loss_history: list[float] = []
    for _ in range(int(n_iter)):
        val, grad = loss_and_grad(log_th)
        updates, opt_state = opt.update(grad, opt_state)
        log_th = optax.apply_updates(log_th, updates)
        loss_history.append(float(val))
    theta_fit = jnp.exp(log_th)
    fitted = base.replace(**{k: theta_fit[i] for i, k in enumerate(keys)})

    # ---- 体外报告（此处可安全 float 转换）------------------------------
    ds0, ws0 = _predict_dw_mat(
        base.replace(**{k: theta0[i] for i, k in enumerate(keys)}), plans,
        n_grid=n_grid)
    ds1, ws1 = _predict_dw_mat(fitted, plans, n_grid=n_grid)

    def _rmse(a, b):
        return jnp.asarray(jnp.sqrt(jnp.mean((a - b) ** 2)), dtype=jnp.float64)

    def _r2(a, b):
        ss_res = jnp.sum((b - a) ** 2)
        ss_tot = jnp.sum((b - jnp.mean(b)) ** 2)
        return jnp.asarray(1.0 - ss_res / jnp.maximum(ss_tot, 1e-30),
                           dtype=jnp.float64)

    report = CalibrationReport(
        base_material_name=base.name,
        fit_keys=keys,
        fitted_material=fitted,
        experiment_ids=labels,
        measured_depth=m_d,
        predicted_depth_before=ds0,
        predicted_depth_after=ds1,
        measured_width=m_w,
        predicted_width_before=ws0,
        predicted_width_after=ws1,
        rmse_depth=_rmse(ds1, m_d),
        rmse_width=_rmse(ws1, m_w),
        r2_depth=_r2(ds1, m_d),
        r2_width=_r2(ws1, m_w),
        n_iter=int(n_iter),
        converged=bool(loss_history and loss_history[-1] <= loss_history[0] * 0.01),
        cost=jnp.asarray(float(loss_history[-1]) if loss_history else float("nan"),
                         dtype=jnp.float64),
    )
    if verbose:
        print(report.summary())
    return report


def make_experiments_from_dicts(
    records: Sequence[Mapping[str, Any]],
    *,
    modality: str = "SLM",
) -> list[CalibrationExperiment]:
    """把 JSON / dict 列表转成实验列表（供 CLI 与批量标定使用）。

    每条记录需含 ``measured_depth`` / ``measured_width``，工艺标量（
    ``laser_power`` 等）可选（缺省用工业默认值）。
    """
    out: list[CalibrationExperiment] = []
    for rec in records:
        plan = process_from_dict(rec, modality=modality)
        out.append(CalibrationExperiment(
            process=plan,
            measured_depth=float(rec["measured_depth"]),
            measured_width=float(rec["measured_width"]),
            label=str(rec.get("label", f"exp{len(out)}")),
        ))
    return out
