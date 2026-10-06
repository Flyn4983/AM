"""粉末/粒子导入测试（缺口 #20-②B）

守住核心能力：CSV/JSON 点云真实导入与往返、from_part_bed 真实随机粉末床
生成、在真实粒子上测量的统计（d50 / PSD / 体积分数）、真实统计映射为
powder.bed 参数、以及驱动**真实 DEM 铺粉**求解得到有限且合理的密实度。
全部为纯逻辑/数值测试，无 Qt/Vtk 依赖，可无显示运行。
"""
from __future__ import annotations

import json
import numpy as np
import jax.numpy as jnp
import pytest

from amforge.core.contracts import PartGeometry, ProcessPlan
from amforge.gui import preproc as P
from amforge.powder import ParticleCollection


def _box3d(spacing_um: float = 200.0, size_mm: float = 1.0) -> PartGeometry:
    return P.build_primitive("box", length_mm=size_mm, spacing_um=spacing_um)


def _box2d(n: int = 11, spacing: float = 1e-4) -> PartGeometry:
    a = 0.5e-3
    xs = spacing * np.arange(n)
    ys = spacing * np.arange(n)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    sdf2 = np.maximum(np.abs(XX) - a, np.abs(YY) - a)
    return PartGeometry(
        sdf=jnp.asarray(sdf2), origin=jnp.asarray([0.0, 0.0]),
        spacing=spacing, dim=2, name="box2d",
    )


def _plan() -> ProcessPlan:
    return ProcessPlan.uniform(
        4, modality="SLM", laser_power=200.0, scan_speed=1.0,
        layer_thickness=40e-6, hatch_spacing=80e-6, beam_radius=50e-6,
        absorption=0.4, preheat_temp=373.0,
    )


# ---------------------------------------------------------------------------
# 1. CSV 点云导入 + 往返
# ---------------------------------------------------------------------------
def test_csv_import_roundtrip(tmp_path):
    path = tmp_path / "particles.csv"
    lines = ["# material=316L", "x,y,z,r",
             "1.0e-4,2.0e-4,3.0e-4,1.5e-5",
             "2.0e-4,3.0e-4,1.0e-4,2.0e-5"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    col = P.build_particle_collection_csv(path)
    assert col.dim == 3
    assert col.n_particles == 2
    assert col.material == "316L"
    pos = np.asarray(col.position)
    assert np.allclose(pos[0], [1.0e-4, 2.0e-4, 3.0e-4])
    assert float(np.asarray(col.radius)[1]) == pytest.approx(2.0e-5)

    out = tmp_path / "out.csv"
    col.to_csv(out)
    col2 = ParticleCollection.from_csv(out)
    assert col2 == col


def test_csv_2d_import(tmp_path):
    path = tmp_path / "p2d.csv"
    path.write_text("x,y,r\n1.0e-4,2.0e-4,1.5e-5\n", encoding="utf-8")
    col = ParticleCollection.from_csv(path)
    assert col.dim == 2
    assert col.n_particles == 1
    assert np.asarray(col.position).shape == (1, 2)


def test_generated_bed_csv_roundtrip_lossless(tmp_path):
    """真实（含抖动）粉末床经 CSV 导出再导入必须 bit-exact 相等。

    早期 ``to_csv`` 用 ``.9e`` 截断 float64 → 往返 ``__eq__`` 失真。
    现用 ``.17e`` 保证无损往返（与 JSON 全精度一致）。
    """
    part = _box3d()
    col = ParticleCollection.from_part_bed(
        part, layer_thickness=40e-6, d50=30e-6, psd_sigma=0.25,
        n_layers=1, seed=0, material="316L")
    out = tmp_path / "bed.csv"
    col.to_csv(out)
    col2 = ParticleCollection.from_csv(out)
    assert col2 == col
    # 统计亦应一致（导出无损）
    s1, s2 = col.stats(), col2.stats()
    assert s1["d50"] == pytest.approx(s2["d50"], rel=1e-12)
    assert s1["n_particles"] == s2["n_particles"]


# ---------------------------------------------------------------------------
# 2. JSON 点云导入 + 往返
# ---------------------------------------------------------------------------
def test_json_import_roundtrip(tmp_path):
    path = tmp_path / "particles.json"
    d = {"material": "316L",
         "particles": [[1.0e-4, 2.0e-4, 3.0e-4, 1.5e-5],
                       [2.0e-4, 3.0e-4, 1.0e-4, 2.0e-5]]}
    path.write_text(json.dumps(d), encoding="utf-8")
    col = ParticleCollection.from_json(path)
    assert col.dim == 3
    assert col.n_particles == 2
    assert float(np.asarray(col.radius)[0]) == pytest.approx(1.5e-5)

    out = tmp_path / "out.json"
    col.to_json(out)
    col2 = ParticleCollection.from_json(out)
    assert col2 == col


# ---------------------------------------------------------------------------
# 3. from_part_bed 真实生成
# ---------------------------------------------------------------------------
def test_from_part_bed_3d():
    part = _box3d()
    col = ParticleCollection.from_part_bed(
        part, layer_thickness=40e-6, d50=30e-6, psd_sigma=0.25,
        n_layers=1, seed=0, material="316L")
    assert col.dim == 3
    assert col.n_particles > 0
    pos = np.asarray(col.position)
    lo, hi = part.bbox()
    lo = np.asarray(lo); hi = np.asarray(hi)
    z0 = float(lo[2]); z1 = z0 + 40e-6
    assert pos[:, 2].min() >= z0 - 30e-6
    assert pos[:, 2].max() <= z1 + 30e-6
    assert pos[:, 0].min() >= float(lo[0]) - 30e-6
    assert pos[:, 0].max() <= float(hi[0]) + 30e-6
    pd = col.packing_density()
    assert 0.05 < pd < 0.9
    assert col.stats()["d50"] == pytest.approx(30e-6, rel=0.2)


def test_from_part_bed_2d():
    part = _box2d()
    col = ParticleCollection.from_part_bed(
        part, layer_thickness=40e-6, d50=30e-6, n_layers=1, seed=1)
    assert col.dim == 2
    assert col.n_particles > 0
    assert np.asarray(col.position).shape[1] == 2


def test_from_part_bed_seed_deterministic():
    part = _box3d()
    a = ParticleCollection.from_part_bed(part, d50=30e-6, seed=7)
    b = ParticleCollection.from_part_bed(part, d50=30e-6, seed=7)
    assert np.allclose(np.asarray(a.position), np.asarray(b.position))


# ---------------------------------------------------------------------------
# 4. 真实统计映射为 powder.bed 参数
# ---------------------------------------------------------------------------
def test_powder_bed_params_mapping():
    part = _box3d()
    col = ParticleCollection.from_part_bed(
        part, layer_thickness=40e-6, d50=25e-6, psd_sigma=0.3, seed=3)
    params = col.powder_bed_params(n_steps=100)
    assert params["powder_fidelity"] == "dem"
    assert params["d50"] == pytest.approx(col.stats()["d50"], rel=1e-6)
    assert params["psd_sigma"] == pytest.approx(col.stats()["psd_sigma"], rel=1e-6)
    assert params["rho_powder"] == pytest.approx(7950.0, rel=1e-6)
    assert params["dem"]["n_steps"] == 100


def test_preproc_wrappers_smoke():
    part = _box3d()
    col = P.generate_powder_bed(part, layer_thickness=40e-6, d50=30e-6)
    assert col.n_particles > 0
    params = P.powderbed_params_from_collection(col, n_steps=60)
    assert params["powder_fidelity"] == "dem"


# ---------------------------------------------------------------------------
# 5. 驱动真实 DEM 铺粉求解（实打实）
# ---------------------------------------------------------------------------
def test_solve_dem_real_powderbed():
    part = _box3d()
    plan = _plan()
    col = ParticleCollection.from_part_bed(
        part, layer_thickness=40e-6, d50=30e-6, psd_sigma=0.25, seed=0)
    res = P.solve_powder_from_collection(col, part, plan, n_steps=100)
    pd = float(np.asarray(res.packing_density))
    # 真实 DEM 解出的密实度：有限、非退化、落在大颗粒堆积的合理区间
    # （本代表性 DEM 床按零件 bbox÷2·d50 定晶格规模，沉降到 RCP 量级）
    assert np.isfinite(pd)
    assert 0.05 < pd < 0.95
    # 配位数：软接触均值（sigmoid(overlap/0.02·d50)），对粗代表性床偏低但
    # 必须有限且 > 0 —— 证明确有接触发生、DEM 解出非退化指标
    cn = float(np.asarray(res.coordination_number))
    assert np.isfinite(cn)
    assert cn > 0.0
    assert np.isfinite(float(np.asarray(res.surface_roughness)))
    assert float(np.asarray(res.surface_roughness)) > 0.0


# ---------------------------------------------------------------------------
# 6. 精确粒子坐标注入 DEM 初始态（②B 增强，opt-in 向后兼容）
# ---------------------------------------------------------------------------
def test_synth_powder_bed_injects_precise_positions():
    """synth_powder_bed 给定 initial_positions 时，精确坐标应原样成为沉降初态。"""
    from diffmech.methods.am.powder_bed import PowderBedConfig, synth_powder_bed
    pos = np.array([[1.0e-4, 5.0e-5, 2.0e-5],
                    [2.0e-4, 5.0e-5, 4.0e-5],
                    [3.0e-4, 5.0e-5, 3.0e-5]], dtype=np.float64)
    rad = np.array([15e-6, 20e-6, 18e-6], dtype=np.float64)
    cfg = PowderBedConfig(layer_thickness=jnp.asarray(40e-6),
                          bed_lx=jnp.asarray(5e-4), bed_ly=jnp.asarray(5e-4),
                          d50=30e-6, psd_sigma=0.25, rho_powder=7950.0,
                          nx=6, ny=6, nz=3, dim=3, seed=0,
                          initial_positions=pos, initial_radii=rad)
    state, col = synth_powder_bed(cfg)
    # 注入的精确坐标应原样成为沉降初态（未被抖动晶格覆盖）
    assert np.allclose(np.asarray(state.position), pos, atol=1e-15)
    assert np.allclose(np.asarray(state.radius), rad, atol=1e-15)
    # 列索引在 [0, nx*ny) 范围，供粗糙度度量
    assert int(col.max()) < 6 * 6 and int(col.min()) >= 0


def test_solve_dem_precise_injection_takes_effect():
    """精确注入生效：单粒子初始态的配位数应远低于默认密堆床（证明确实用了注入坐标）。"""
    import os
    import tempfile
    part = _box3d()
    plan = _plan()
    csv = "\n".join([
        "# name=one", "x,y,z,r",
        f"{5.0e-4:.17e},{5.0e-4:.17e},{0.0:.17e},{15e-6:.17e}",
    ])
    d = tempfile.mkdtemp()
    p = os.path.join(d, "one.csv")
    with open(p, "w", encoding="utf-8") as f:
        f.write(csv + "\n")
    col = ParticleCollection.from_csv(p)
    res_default = col.solve_dem(part, plan, n_steps=60)
    res_precise = col.solve_dem(part, plan, n_steps=60, use_precise=True)
    cz_d = float(np.asarray(res_default.coordination_number))
    cz_p = float(np.asarray(res_precise.coordination_number))
    assert np.isfinite(cz_d) and np.isfinite(cz_p)
    # 单粒子无邻居 → 配位数≈0，明显低于默认密堆床（注入确实生效）
    assert cz_p < cz_d, f"精确注入未生效：cz_p={cz_p} 未小于 cz_d={cz_d}"
    # 其余指标有限（不抛错，向后兼容默认路径）
    assert np.isfinite(float(np.asarray(res_precise.packing_density)))
    assert np.isfinite(float(np.asarray(res_precise.surface_roughness)))
