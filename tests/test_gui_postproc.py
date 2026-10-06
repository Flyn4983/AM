"""后处理纯逻辑层 (amforge.gui.postproc) 的无显示测试

只依赖 numpy，可在无 PySide6/vtk 的 CI / 沙箱里直接跑。覆盖：
* 真实端到端 simulate 结果的场量抽取（有限性 / 形状对齐 / 坐标）；
* ParaView 可读 .vtk / .csv 导出 + manifest（含一次真实回读校验）；
* 数字样机装配摘要 + 铰接链坐标；
* 可微优化聚合（用合成结果 dict 验证逻辑，避免触发慢速真实优化）。
"""

import numpy as np
import pytest

from amforge.core.contracts import PartGeometry
from amforge.gui import preproc as P
from amforge.gui import postproc as PP


def _tiny_case():
    """构造一个很小的几何 + 工艺（保证 simulate 秒级）。

    工艺与网格自洽（2026-10-06 D0）：``spacing_um=200`` 的体素装不下 50µm 光斑
    （热源完全欠采样，A0 护栏会主动拒绝），故取 ``r=100µm``（2r=dx）+ 相应
    道距/层厚。本文件只验证后处理接线与场量形状/有限性，不取熔池形态定量值。
    """
    part = P.build_primitive("sphere", 1.0, spacing_um=200.0, name="tiny")
    lt = 200e-6
    n_layers = part.layer_count(lt)
    plan = P.build_process_plan(
        n_layers, laser_power=180.0, scan_speed=1.0, layer_thickness=lt,
        hatch_spacing=280e-6, beam_radius=100e-6, absorption=0.4,
        preheat_temp=373.0, rotation_per_layer_deg=67.0)
    return part, plan


def _run_tiny():
    part, plan = _tiny_case()
    params = {"n_grid": 6, "max_layers": 3, "n_sub_cp": 8}
    out = PP.run_simulation(
        part, plan, asbuilt_solver="buildup", micro_solver="surrogate",
        mbd_solver="surrogate", params=params)
    return part, out


def test_field_extraction_finite_and_shaped():
    part, out = _run_tiny()
    names = PP.grid_fields(out, part)
    assert "peak_temperature" in names
    assert "von_mises_residual" in names
    assert "solidification_rate" not in names  # 标量场不进体素浏览器
    for name in names:
        fld = PP.get_field(out, name, part)
        assert fld.data.shape == part.shape, (name, fld.data.shape, part.shape)
        assert np.all(np.isfinite(fld.data)), name
        assert fld.spacing == float(part.spacing)


def test_unknown_field_raises():
    part, out = _run_tiny()
    with pytest.raises(KeyError):
        PP.get_field(out, "not_a_field", part)


def test_vtk_and_csv_export_roundtrip(tmp_path):
    part, out = _run_tiny()
    fld = PP.get_field(out, "peak_temperature", part)
    # 写 .vtk
    vtk_path = PP.write_vtk_structured_points(fld, tmp_path / "t.vtk")
    text = (tmp_path / "t.vtk").read_text(encoding="utf-8")
    assert "DATASET STRUCTURED_POINTS" in text
    assert f"DIMENSIONS {fld.data.shape[0]} {fld.data.shape[1]}" in text
    # 回读 POINT_DATA 标量并与原数据（x 最快 Fortran 序）比对
    lines = text.splitlines()
    idx = lines.index("LOOKUP_TABLE default") + 1
    tokens = " ".join(lines[idx:]).split()
    back = np.asarray(tokens, dtype=np.float32)
    expected = fld.data.astype(np.float32).ravel(order="F")
    assert back.shape == expected.shape
    assert np.allclose(back, expected, rtol=1e-5)

    # 写 .csv 并校验行数 = 体素数
    csv_path = PP.write_field_csv(fld, tmp_path / "t.csv")
    arr = np.loadtxt(tmp_path / "t.csv", delimiter=",", skiprows=1)
    assert arr.shape[0] == int(np.prod(fld.data.shape))


def test_export_all_fields_manifest(tmp_path):
    part, out = _run_tiny()
    manifest = PP.export_all_fields(out, part, tmp_path / "fields",
                                    fields=["peak_temperature", "von_mises_residual"])
    assert manifest["count"] == 2
    assert "peak_temperature" in manifest["fields"]
    entry = manifest["fields"]["peak_temperature"]
    assert "vtk" in entry["paths"] and "csv" in entry["paths"]
    assert np.isfinite(entry["min"]) and np.isfinite(entry["max"])
    assert (tmp_path / "fields" / "manifest.json").exists()


def test_assembly_summary_and_chain():
    part, out = _run_tiny()
    s = PP.assembly_summary(out)
    assert s["n_joints"] >= 1
    assert np.isfinite(s["stability_margin"])
    assert 0.0 <= s["assembly_score"] <= 1.0
    chain = PP.assembly_chain(out, part)
    n = s["n_joints"]
    assert chain["positions"].shape == (n, 3)
    assert len(chain["links"]) == max(n - 1, 0)
    # 关节应沿 z 分布且位于零件 x/y 中心
    assert np.allclose(chain["positions"][:, 0], chain["positions"][0, 0])


def test_optimization_summary_joint_synthetic():
    res = {
        "loss_history": [1.0, 0.8, 0.6, 0.5],
        "geom_history": [5.0, 4.0, 3.0, 2.0],
        "stress_history": [0.9, 0.8, 0.7, 0.6],
        "z_history": [np.array([0.1] * 9) for _ in range(4)],
        "constraint_penalty_history": [0.2, 0.1, 0.0, 0.0],
    }
    agg = PP.optimization_summary(res, "joint")
    assert len(agg["loss_history"]) == 4
    assert np.allclose(agg["geom_history"], [5.0, 4.0, 3.0, 2.0])
    assert "z_history" in agg


def test_optimization_summary_pareto_synthetic():
    res = {
        "algorithm": "nsga2", "pop_size": 4, "n_gen": 3,
        "geom_dev": np.array([1.0, 2.0, 0.5, 1.5]),
        "stress": np.array([0.2, 0.1, 0.4, 0.3]),
        "feasible": np.array([1.0, 1.0, 0.9, 1.0]),
        "constraint_penalty": np.array([0.0, 0.0, 0.1, 0.0]),
        "utopia": (0.5, 0.1),
        "best_compromise_index": 0,
        "population_geom_dev": np.array([1.0, 2.0, 0.5, 1.5, 0.8]),
        "population_stress": np.array([0.2, 0.1, 0.4, 0.3, 0.25]),
    }
    agg = PP.optimization_summary(res, "pareto")
    assert agg["n_front"] == 4
    assert agg["algorithm"] == "nsga2"
    assert agg["best_compromise_index"] == 0
    assert agg["population_geom_dev"].shape[0] == 5
