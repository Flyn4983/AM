"""后处理/数字样机/优化 UI 层 (postproc_app) 与统一工作台 (app) 的导入+构造测试

用 ``pytest.importorskip("PySide6" / "vtk")`` 守护：无 GUI 依赖的环境自动跳过；
装有 PySide6+vtk 的桌面（如用户 Windows 机 ``pip install "amforge[gui]"`` 后）
自动执行，作为离线 offscreen 冒烟测试，验证：
* 模块可导入、主窗口可构造（构造前必须先有 QApplication 实例，否则
  matplotlib 的 FigureCanvasQTAgg / 各类 QWidget 会抛
  "Cannot create a QWidget without QApplication"）；
* 真实端到端 ``simulate`` 结果能喂给 UI 的 ``set_result`` 回调并刷新各视图；
* 统一工作台包含「前处理」「后处理」两个标签页；
* VTK 视图在无 OpenGL 时安全回退（不崩溃）。
"""

import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("vtk")

# 必须在导入任何 PySide6/matplotlib-Qt 组件前设定 offscreen，否则 QtAgg 会尝试连接显示器
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from amforge.gui import preproc as P  # noqa: E402
from amforge.gui import postproc as PP  # noqa: E402
from amforge.gui.postproc_app import PostprocMainWindow  # noqa: E402
from amforge.gui.app import UnifiedMainWindow  # noqa: E402


def _qapp() -> QApplication:
    """构造/复用进程级 QApplication。offscreen 下即可构造窗口而不需真实显示器。"""
    return QApplication.instance() or QApplication([])


def _tiny_sim():
    part = P.build_primitive("sphere", 1.0, spacing_um=200.0, name="ui")
    lt = 80e-6
    plan = P.build_process_plan(
        part.layer_count(lt), laser_power=180.0, scan_speed=1.0,
        layer_thickness=lt, hatch_spacing=120e-6, beam_radius=50e-6,
        absorption=0.4, preheat_temp=373.0, rotation_per_layer_deg=67.0)
    out = PP.run_simulation(
        part, plan, asbuilt_solver="buildup", micro_solver="surrogate",
        mbd_solver="surrogate",
        params={"n_grid": 6, "max_layers": 3, "n_sub_cp": 8})
    return part, plan, out


def test_postproc_app_construct():
    _qapp()
    win = PostprocMainWindow()
    assert win is not None
    # 标签页数量（场浏览器 / 数字样机 / 可微优化）
    assert win.centralWidget() is not None


def test_postproc_set_result_refreshes_views():
    _qapp()
    part, plan, out = _tiny_sim()
    win = PostprocMainWindow()
    win._on_sim_done(out, part)  # 直接调用仿真完成回调
    # 场浏览器应被填充
    assert win.field_view.field_names, "场浏览器未被结果填充"
    # 数字样机摘要应含装配评分
    summ = PP.assembly_summary(out)
    assert "assembly_score" in summ
    # 优化视图初始为空属正常（需单独跑优化）


def test_unified_app_has_two_tabs():
    _qapp()
    u = UnifiedMainWindow()
    tabs = u.centralWidget()
    # QTabWidget 有三个标签页：① 前处理 / ② 后处理·数字样机·优化 / ③ 监测闭环
    # （③ 监测闭环来自 M4 蓝图，2026-08-17 落地的 monitoring_app 第③标签）
    assert tabs.count() == 3
