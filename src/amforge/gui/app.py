"""AMForge 统一工作台（PySide6 + VTK）：单窗口多标签页

把 **前处理 (#20)** 与 **后处理 / 数字样机 / 可微优化 (#22)** 两个工作台
集成进同一个主窗口的标签页，形成完整的"建模 → 仿真 → 后处理 → 优化"闭环：

* 标签页 ① **前处理**：几何 / 分层 / hatch 扫描路径 / 工艺评估 / 导出配置
  （见 :mod:`amforge.gui.preproc_app.PreprocMainWindow`）。
* 标签页 ② **后处理 / 数字样机 / 可微优化**：体素场浏览器（2D 云图 + 3D 着色面）、
  数字样机装配评估、可微优化（loss 历史 / NSGA-II Pareto 前沿）
  （见 :mod:`amforge.gui.postproc_app.PostprocMainWindow`）。
* 标签页 ③ **监测闭环仪表盘**：传感流回放 + 缺陷监测 + 工艺修正记录
  （见 :mod:`amforge.gui.monitoring_app.MonitoringMainWindow`，模块 D）。

两个标签页通过「前处理导出配置 → 后处理导入配置」天然串联：前处理导出的
``preproc_geometry.npz`` + ``preproc_process.json`` 可直接被后处理标签页导入，
形成统一工程流。

仅当用户显式运行 ``amforge gui`` 时才 import 本模块（PySide6/vtk 是重依赖）。
"""

from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget  # noqa: E402

from amforge.gui.preproc_app import PreprocMainWindow
from amforge.gui.postproc_app import PostprocMainWindow
from amforge.gui.monitoring_app import MonitoringMainWindow


class UnifiedMainWindow(QMainWindow):
    """单窗口多标签页：前处理 + 后处理/数字样机/优化 + 监测闭环。"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AMForge 统一工作台 — 前处理 + 后处理/数字样机/优化 + 监测闭环")
        self.resize(1400, 900)

        pre = PreprocMainWindow()
        post = PostprocMainWindow()
        mon = MonitoringMainWindow()

        tabs = QTabWidget()
        tabs.addTab(pre.centralWidget(), "① 前处理 Pre-processing")
        tabs.addTab(post.centralWidget(), "② 后处理/数字样机/优化 Post-processing")
        tabs.addTab(mon.centralWidget(), "③ 监测闭环 Monitoring")
        tabs.setTabPosition(QTabWidget.North)
        self.setCentralWidget(tabs)

        # 保留对窗口对象的引用，防止其被 GC 影响内部状态
        self._pre = pre
        self._post = post
        self._mon = mon


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    win = UnifiedMainWindow()
    win.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
