"""后处理动画（需求 ③）：扫描温度场时间演化帧抽取 + GIF / ParaView 时间序列导出。

验证点
------
* 求解器 ``record_frames`` 默认关闭时 ``temperature_evolution is None``（零额外开销）；
* 开启后返回有限、形状正确的帧序列，且熔池随扫描在帧间移动（实打实数值演化）；
* 逻辑层 ``extract_temperature_frames`` / ``save_field_gif`` / ``save_field_pvd`` 产出
  可用文件（GIF 像素动画、ParaView 可读 .pvd 时间序列）。

全部为纯 numpy / matplotlib(Agg) 计算，无需 vtk 运行时，可在 CI / 沙箱 pytest 守住。
"""

import os

import jax.numpy as jnp
import numpy as np
import pytest

from amforge.geometry import from_sdf_fn
from amforge.core.contracts import ProcessPlan
from amforge.thermal_enthalpy import solve_enthalpy_thermal
from amforge.gui import postproc as PP


def _sphere_part(dx: float = 80e-6, half: float = 0.4e-3):
    # 球填满（略小于）Bounding Box，使扫描热源路径始终落在实体内 → 真实熔化。
    return from_sdf_fn(
        lambda x: jnp.linalg.norm(x, axis=-1) - half,
        bounds=[(-half, half)] * 3, spacing=dx, name="ball",
    )


def _plan():
    return ProcessPlan.uniform(
        n_layers=1, laser_power=600.0, scan_speed=0.6, beam_radius=60e-6,
        absorption=0.45, layer_thickness=60e-6, hatch_spacing=120e-6,
    )


def _run(record_frames: bool, n_frames: int = 12):
    geo = _sphere_part()
    plan = _plan()
    out = solve_enthalpy_thermal(
        geometry=geo, process=plan,
        # 步数由 CFL 自动推导（实测 n_steps=2118）。原先这里硬传 n_steps=40：在
        # A0 之后那已低于 SSP-RK2 稳定下限，会直接 ValueError——扩散项发散时
        # 「帧在动」不再等于「物理在演化」。
        params={"record_frames": record_frames, "n_animation_frames": n_frames},
    )
    return geo, out


def test_record_frames_default_none():
    """默认不记录帧：temperature_evolution 为 None，不影响任何既有计算。"""
    geo, out = _run(False)
    assert out.temperature_evolution is None


def test_record_frames_enabled_shape_and_finite():
    """开启后帧序列形状正确、有限，温度整体在物理区间（蒸发封顶附近）。"""
    geo, out = _run(True)
    evo = out.temperature_evolution
    assert evo is not None
    gshape = tuple(geo.shape)
    assert evo.shape[0] >= 2 and evo.shape[1:] == gshape
    assert jnp.all(jnp.isfinite(evo))
    assert float(evo.min()) > 200.0          # 高于环境温度
    assert float(evo.max()) < 6000.0         # 被蒸发封顶钳在蒸气化上限附近


def test_meltpool_moves_across_frames():
    """熔池重心随扫描在帧间明显移动 —— 证明动画是真实数值演化而非静止帧。"""
    geo, out = _run(True)
    evo = np.asarray(out.temperature_evolution)
    T_liq = 1723.0
    centroids = []
    for i in range(evo.shape[0]):
        hot = np.argwhere(evo[i] > T_liq)
        if hot.size:
            centroids.append(hot.mean(axis=0))
    assert len(centroids) >= 2
    c0 = np.array(centroids[0])
    c1 = np.array(centroids[-1])
    assert np.linalg.norm(c0 - c1) > 0.5     # 跨帧明显移动（单位：体素）


def test_extract_temperature_frames():
    """逻辑层抽帧：数量与演化数组一致，每帧与几何同形、单位 K。"""
    geo, out = _run(True)
    frames = PP.extract_temperature_frames(out, geo)
    assert len(frames) == out.temperature_evolution.shape[0]
    assert frames
    for f in frames:
        assert f.data.shape == tuple(geo.shape)
        assert f.dim == 3
        assert f.unit == "K"


def test_save_field_gif(tmp_path):
    """导出 GIF：文件存在且非空（Ag 后端，无显示依赖）。"""
    geo, out = _run(True)
    frames = PP.extract_temperature_frames(out, geo)
    path = str(tmp_path / "scan.gif")
    got = PP.save_field_gif(frames, path, fps=6)
    assert os.path.exists(got)
    assert os.path.getsize(got) > 1000


def test_save_field_pvd(tmp_path):
    """导出 ParaView 时间序列：.pvd 合法集合 + 首帧 .vtk 存在。"""
    geo, out = _run(True)
    frames = PP.extract_temperature_frames(out, geo)
    path = str(tmp_path / "scan.pvd")
    got = PP.save_field_pvd(frames, path)
    assert os.path.exists(got)
    txt = open(got, encoding="utf-8").read()
    assert "<DataSet" in txt and "timestep" in txt
    stem = os.path.splitext(got)[0]
    assert os.path.exists(stem + "_0000.vtk")
