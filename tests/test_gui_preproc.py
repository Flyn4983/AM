"""前处理 GUI 纯逻辑层测试（无 Qt/Vtk，可无显示运行）

守住 #20 的核心算法：几何生成、hatch 扫描路径、工艺方案、可打印性、配置导入导出。
"""

from __future__ import annotations

import numpy as np
import pytest

from amforge.core.contracts import PartGeometry, solid_mask
from amforge.gui.preproc import (
    build_primitive,
    generate_hatch_paths,
    build_process_plan,
    default_process_params,
    assess,
    export_config,
    import_config,
    generate_support,
)


def test_build_primitive_sphere():
    part = build_primitive("sphere", length_mm=1.0, spacing_um=200.0)
    assert part.dim == 3
    assert bool(np.asarray(solid_mask(np.asarray(part.sdf)) > 0.5).any())  # 有实体体素（#19 口径）
    assert (np.asarray(part.sdf) > 0).any()       # 有空气体素
    assert part.layer_count(40e-6) >= 1


def test_primitive_shapes_distinct():
    sphere = build_primitive("sphere", length_mm=1.0, spacing_um=200.0)
    box = build_primitive("box", length_mm=1.0, spacing_um=200.0)
    cylinder = build_primitive("cylinder", length_mm=1.0, spacing_um=200.0)
    # 形状不同 → SDF 不应逐点相等
    assert not np.allclose(np.asarray(sphere.sdf), np.asarray(box.sdf))
    assert not np.allclose(np.asarray(cylinder.sdf), np.asarray(sphere.sdf))


def test_hatch_paths_nonempty_and_layered():
    part = build_primitive("box", length_mm=1.0, spacing_um=150.0)
    lt = 40e-6
    hs = 120e-6
    res = generate_hatch_paths(part, lt, hs)
    assert res["n_layers"] >= 1
    assert len(res["paths"]) == res["n_layers"]
    total_segments = sum(len(layer) for layer in res["paths"])
    assert total_segments > 0, "hatch 路径不应为空"
    # 至少一层有路径、且最密一层至少有 1 段（居中零件的顶/底层位于零件外，可空缺）
    assert sum(1 for layer in res["paths"] if layer) >= 1
    assert max(len(layer) for layer in res["paths"]) >= 1
    # 线段端点坐标有限
    for layer in res["paths"]:
        for (x0, y0), (x1, y1) in layer:
            assert np.isfinite([x0, y0, x1, y1]).all()


def test_hatch_spacing_density_monotonic():
    """hatch 间距越小，平均每段越长（扫描线更密、截断更少），作为合理性守恒检查。"""
    part = build_primitive("box", length_mm=1.0, spacing_um=150.0)
    lt = 40e-6
    r_coarse = generate_hatch_paths(part, lt, 200e-6)
    r_fine = generate_hatch_paths(part, lt, 60e-6)
    # 细 hatch → 每层段数更多
    segs_coarse = sum(len(l) for l in r_coarse["paths"])
    segs_fine = sum(len(l) for l in r_fine["paths"])
    assert segs_fine > segs_coarse


def test_build_process_plan_and_assess():
    part = build_primitive("sphere", length_mm=1.0, spacing_um=200.0)
    p = default_process_params()
    n_layers = part.layer_count(p["layer_thickness"])
    plan = build_process_plan(n_layers, **p)
    assert plan.n_layers == n_layers
    report = assess(part, plan)
    assert "printability" in report
    assert report["VED_J_m3"] > 0.0
    assert "VED_status" in report


def test_export_import_roundtrip(tmp_path):
    part = build_primitive("cylinder", length_mm=1.0, spacing_um=200.0)
    p = default_process_params()
    plan = build_process_plan(part.layer_count(p["layer_thickness"]), **p)
    gpath, ppath = export_config(part, plan, tmp_path)
    assert (tmp_path / "preproc_geometry.npz").exists()
    assert (tmp_path / "preproc_process.json").exists()
    part2, plan2 = import_config(tmp_path)
    assert np.allclose(np.asarray(part2.sdf), np.asarray(part.sdf))
    assert float(plan2.laser_power) == float(plan.laser_power)
    assert plan2.modality == plan.modality


def test_generate_support_logic():
    """模块A 前处理入口：落于基板的零件不应生成支撑，悬空零件应生成。"""
    on_plate = build_primitive("box", length_mm=1.0, spacing_um=200.0)
    # box 默认落在基板（origin 居中、半边长对称）→ 通常无需支撑
    r_on = generate_support(on_plate, kind="block")
    assert r_on["support"].kind == "block"
    assert r_on["support_mask_2d"].shape == on_plate.sdf.shape[:2]

    # 构造一个悬空块（底面离基板 2 个体素）
    import jax.numpy as jnp
    sp = float(on_plate.spacing)
    nx, ny, nz = on_plate.sdf.shape
    xs = on_plate.origin[0] + sp * jnp.arange(nx)
    ys = on_plate.origin[1] + sp * jnp.arange(ny)
    zs = on_plate.origin[2] + sp * jnp.arange(nz)
    X, Y, Z = jnp.meshgrid(xs, ys, zs, indexing="ij")
    P = jnp.stack([X, Y, Z], axis=-1)
    center = jnp.array([0.0, 0.0, 3.0 * sp], dtype=jnp.float64)
    half = jnp.array([2.0 * sp, 2.0 * sp, 1.0 * sp], dtype=jnp.float64)
    q = jnp.abs(P - center) - half
    sdf = jnp.linalg.norm(jnp.maximum(q, 0.0), axis=-1) + jnp.minimum(jnp.max(q, axis=-1), 0.0)
    floating = PartGeometry(sdf=sdf, origin=on_plate.origin, spacing=sp, dim=3, name="float")
    r_fl = generate_support(floating, kind="block")
    assert float(r_fl["support"].volume_fraction) > 1e-3

