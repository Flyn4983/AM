"""材料标定（Module C）测试

设计：用"已知真值材料"自洽生成实测熔池尺寸，再从未标定基准出发反演，
断言 (1) 残差大幅下降、(2) 拟合优度 R² 接近 1、(3) 报告为合法契约且可 pytree
往返。标定目标本就是"用实验数据拟合材料以匹配观测"，而非保证参数唯一可辨识
（热物性之间存在物理退化，故只强校验预测质量，不强校验单参数精确回收）。
"""
import argparse
import json
import tempfile

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from amforge.core.contracts import CalibrationReport, CONTRACTS  # noqa: E402
from amforge.materials import AMMaterial, get_material  # noqa: E402
from amforge.meltpool import solve_meltpool_surrogate  # noqa: E402
from amforge.process import ProcessPlan  # noqa: E402
from amforge.calibration import (  # noqa: E402
    CalibrationExperiment,
    _coupon,
    calibrate_material,
    make_experiments_from_dicts,
    process_from_dict,
)

FIT_KEYS = ("k_solid", "rho_solid", "cp_solid", "latent_fusion")
GT_VALS = {
    "k_solid": 38.0,
    "rho_solid": 8200.0,
    "cp_solid": 720.0,
    "latent_fusion": 3.1e5,
}


def _gt_experiments(n_grid: int = 16):
    """用已知真值材料自洽生成"实测"熔池尺寸 + 对应工艺方案。

    与标定默认 ``n_grid=16`` 保持一致，确保预测/实测在同一代理分辨率下可比。
    """
    geo = _coupon()
    base = get_material("316L")
    gt = base.replace(**GT_VALS)
    plans = []
    for P_laser, v in [(150.0, 0.8), (200.0, 1.0), (250.0, 1.2),
                      (300.0, 0.9), (180.0, 1.1)]:
        plans.append(ProcessPlan.uniform(
            n_layers=1, laser_power=P_laser, scan_speed=v,
            beam_radius=50e-6, absorption=0.35, layer_thickness=40e-6))
    exps = []
    for i, plan in enumerate(plans):
        out = solve_meltpool_surrogate(
            geometry=geo, process=plan,
            params={"material": gt, "n_grid": n_grid})
        exps.append(CalibrationExperiment(
            process=plan, measured_depth=float(out.depth),
            measured_width=float(out.width), label=f"GT{i}"))
    return exps


def test_calibration_fits_known_material():
    exps = _gt_experiments()
    report = calibrate_material(
        "316L", exps, fit_keys=FIT_KEYS, n_iter=150, verbose=False)

    # 报告是合法契约
    assert isinstance(report, CalibrationReport)
    assert "CalibrationReport" in CONTRACTS

    # 残差大幅下降（标定前的预测与实测差距被显著压缩）
    init_res = float(jnp.mean(
        (report.predicted_depth_before - report.measured_depth) ** 2
        + (report.predicted_width_before - report.measured_width) ** 2))
    final_res = float(jnp.mean(
        (report.predicted_depth_after - report.measured_depth) ** 2
        + (report.predicted_width_after - report.measured_width) ** 2))
    assert final_res < 0.1 * init_res, (
        f"标定后残差未充分下降: init={init_res:.3e} final={final_res:.3e}")

    # 拟合优度接近完美（无噪声自洽数据，模型可表示）
    assert float(report.r2_depth) > 0.9
    assert float(report.r2_width) > 0.9
    assert jnp.isfinite(report.cost)

    # 拟合材料在 fit_keys 上确实偏离基准（说明优化器动了参数）
    base = get_material("316L")
    for k in FIT_KEYS:
        assert float(getattr(report.fitted_material, k)) != pytest.approx(
            float(getattr(base, k)), rel=1e-6)


def test_calibration_report_pytree_roundtrip():
    """CalibrationReport 应可作为 pytree 叶子参与 jax 变换。"""
    exps = _gt_experiments()
    report = calibrate_material(
        "316L", exps, fit_keys=FIT_KEYS, n_iter=120, verbose=False)
    leaves = jax.tree_util.tree_leaves(report)
    # 11 个数组叶子（6 组 before/after + rmse_depth/width + r2_depth/width + cost）
    assert len(leaves) == 11
    # 重建后关键数组不变
    rebuilt = jax.tree_util.tree_map(lambda x: x, report)
    assert bool(jnp.allclose(rebuilt.measured_depth, report.measured_depth))


def test_make_experiments_from_dicts_and_process_from_dict():
    recs = [
        {"label": "a", "laser_power": 200.0, "scan_speed": 1.0,
         "measured_depth": 1.2e-4, "measured_width": 2.1e-4},
        {"label": "b", "laser_power": 250.0, "scan_speed": 1.2,
         "measured_depth": 1.5e-4, "measured_width": 2.4e-4},
    ]
    exps = make_experiments_from_dicts(recs)
    assert len(exps) == 2
    assert exps[0].label == "a"
    assert float(exps[0].process.laser_power) == pytest.approx(200.0)
    # process_from_dict 缺省也能构造
    p = process_from_dict({"laser_power": 300.0})
    assert float(p.laser_power) == pytest.approx(300.0)
    assert float(p.scan_speed) == pytest.approx(1.0)


def test_cli_calibrate_roundtrip(tmp_path):
    """CLI 路径：写实验 JSON -> cmd_calibrate -> 读报告 JSON，拟合优度高。"""
    from amforge.cli import cmd_calibrate

    exps = _gt_experiments(n_grid=16)
    records = []
    for e in exps:
        records.append({
            "label": e.label,
            "laser_power": float(e.process.laser_power),
            "scan_speed": float(e.process.scan_speed),
            "beam_radius": float(e.process.beam_radius),
            "absorption": float(e.process.absorption),
            "layer_thickness": float(e.process.layer_thickness),
            "measured_depth": e.measured_depth,
            "measured_width": e.measured_width,
        })
    spec = {
        "base_material": "316L",
        "fit_keys": list(FIT_KEYS),
        "experiments": records,
    }
    inp = tmp_path / "exps.json"
    inp.write_text(json.dumps(spec))
    outp = tmp_path / "report.json"

    args = argparse.Namespace(
        input=str(inp), output=str(outp), n_iter=150, quiet=True)
    rc = cmd_calibrate(args)
    assert rc == 0

    data = json.loads(outp.read_text())
    assert data["base_material"] == "316L"
    assert data["r2_depth"] > 0.9
    assert data["r2_width"] > 0.9
    # 拟合后物性与基准不同
    for k in FIT_KEYS:
        assert data["fitted_material"][k] != pytest.approx(
            float(getattr(get_material("316L"), k)), rel=1e-6)
