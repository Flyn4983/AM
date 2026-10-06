"""前处理 GUI 界面层冒烟测试（需要 PySide6 + vtk）

用 ``pytest.importorskip`` 跳过：在没有图形依赖的环境（CI/沙箱）里本测试自动跳过，
不会拖垮主回归套件；在有 PySide6+vtk 的桌面环境里则真正构造窗口验证可导入、可构建。

说明：本测试只构造窗口、不调用 ``show()``/``exec()``，因此在 offscreen 平台
（``QT_QPA_PLATFORM=offscreen``）即可运行，不需要真实显示器。
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("vtk")

# 无显示环境也能构造窗口
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_gui_import_and_construct():
    from amforge.gui import preproc_app

    app = None
    try:
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
        win = preproc_app.PreprocMainWindow()
        # 不调用 show()/exec()，仅验证界面可构造、控件齐全
        assert win.prim is not None
        assert win._spins  # 工艺参数自旋框已建
        assert "laser_power" in win._spins
        # 纯逻辑层可在窗口内调用
        part = preproc_app.P.build_primitive("box", 1.0, spacing_um=200.0)
        win._set_part(part)
        assert win.part is not None
        assert win.hatch is not None
        assert win.hatch["n_layers"] >= 1
    finally:
        if app is not None:
            app.quit()


def test_gui_powder_import_generate_eval(tmp_path):
    """真实驱动 ②B 面板的导入 / 生成 / DEM 评估回调（offscreen，不弹窗）。

    验证缺口 #20-②B 的 GUI 接线在交互层确实工作：导入粒子→粒子入集→
    评估真实 DEM 铺粉→结果写入信息框；生成粉末层同样落子。
    """
    from PySide6.QtWidgets import QApplication, QFileDialog
    from amforge.core.contracts import ProcessPlan
    from amforge.gui import preproc_app

    csv = tmp_path / "p.csv"
    csv.write_text(
        "x,y,z,r\n1.0e-4,2.0e-4,3.0e-4,1.5e-5\n"
        "2.0e-4,3.0e-4,1.0e-4,2.0e-5\n", encoding="utf-8")

    app = None
    real_get = QFileDialog.getOpenFileName
    try:
        app = QApplication.instance() or QApplication([])
        # 拦截文件对话框，直接喂入临时 CSV
        QFileDialog.getOpenFileName = staticmethod(
            lambda *a, **k: (str(csv), ""))

        win = preproc_app.PreprocMainWindow()
        part = preproc_app.P.build_primitive("box", 1.0, spacing_um=200.0)
        win._set_part(part)
        plan = ProcessPlan.uniform(
            4, modality="SLM", laser_power=200.0, scan_speed=1.0,
            layer_thickness=40e-6, hatch_spacing=100e-6,
            beam_radius=50e-6, absorption=0.5, preheat_temp=300.0)
        win.plan = plan

        # 1) 导入粒子回调
        win._on_import_particles()
        assert win.powder is not None
        assert win.powder.n_particles == 2
        assert "粒子=2" in win.pow_info.text()

        # 2) 真实 DEM 评估回调（解出密实度等写入 info）
        win._on_eval_powder()
        assert "粉末床评估" in win.info.toPlainText()
        assert "铺粉密度" in win.info.toPlainText()

        # 3) 生成粉末层回调
        win._on_gen_powder()
        assert win.powder is not None and win.powder.n_particles > 0
    finally:
        QFileDialog.getOpenFileName = real_get
        if app is not None:
            app.quit()
