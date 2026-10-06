"""AMForge 监测闭环仪表盘 GUI（PySide6 + matplotlib）

界面布局（主窗口 = 左控制 + 右仪表盘）：

* **左控制栏**：几何来源（示例几何 / 导入前处理配置）、工艺杠杆（功率/速度）、
  闭环次数、在线重标定开关、「运行监测闭环」。
* **右仪表盘**：
  - 标签页 ① **传感流回放**：多通道传感时间序列（热/熔池/光电相干）。
  - 标签页 ② **缺陷监测**：各迭代缺陷概率（未熔合/匙孔/孔隙）折线 + 最终柱状。
  - 标签页 ③ **工艺修正记录**：修正后功率/速度随迭代轨迹 + 文字修正日志。

设计要点（与 postproc_app 一致的纪律）
--------------------------------------
* 重计算（``closedloop.run_loop``）委托给纯逻辑层 :func:`run_monitoring_case`，
  界面只做呈现与后台线程调用，核心算法由 ``tests/test_monitoring.py`` 守住。
* 仅用 matplotlib 2D（无 VTK 3D），无 OpenGL 依赖，offscreen 可构造、可测。
* 仅当用户显式运行 ``amforge gui`` 时才 import 本模块（PySide6 是重依赖）。
"""

from __future__ import annotations

import os
import warnings

import numpy as np

from amforge.gui import preproc as P

warnings.filterwarnings("ignore")

# --- matplotlib 必须在导入 Qt 后端前设定 -------------------------------------
import matplotlib  # noqa: E402
matplotlib.use("QtAgg")  # noqa: E402
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from PySide6.QtCore import Qt, QThread, Signal  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QPushButton, QFileDialog, QSlider, QTextEdit, QSplitter,
    QTabWidget, QDoubleSpinBox, QSpinBox, QMessageBox, QSizePolicy, QCheckBox,
)

# 在桌面环境才启用 3D；本仪表盘纯 2D，无需 VTK。


# ===========================================================================
# 纯逻辑层：运行监测闭环并抽取绘图数组（numpy-only，可独立 pytest）
# ===========================================================================
def run_monitoring_case(geometry, plan, *, n_iter: int = 3, recalibrate: bool = False,
                        material: str = "316L", params: dict | None = None,
                        sensor_channels=("thermal", "meltpool", "photodiode"),
                        tol: float = 0.05, controller: str = "rule"):
    """跑监测闭环，返回 :class:`ClosedLoopResult`（绘图时再按需转 numpy）。

    ``controller`` 选择闭环控制器：``"rule"``（物理规则代理，快）或
    ``"inverse"``（inverse 可微优化器实打实控制，需跑前向仿真，较慢但真实）。
    """
    from amforge.closedloop import run_loop  # 延迟导入，避免重依赖提前加载
    opts = dict(params or {})
    opts["controller"] = controller
    opts.setdefault("n_grid", 6)
    opts.setdefault("max_layers", 3)
    opts.setdefault("n_sub_cp", 8)
    return run_loop(geometry, plan, n_iter=n_iter, material=material, params=opts,
                    recalibrate=recalibrate, sensor_channels=sensor_channels, tol=tol)


# ===========================================================================
# 后台线程：避免闭环仿真阻塞 GUI
# ===========================================================================
class MonitoringWorker(QThread):
    progress = Signal(str)
    finished = Signal(object, object)  # (ClosedLoopResult, geometry)
    errored = Signal(str)

    def __init__(self, geometry, plan, opts: dict):
        super().__init__()
        self.geometry = geometry
        self.plan = plan
        self.opts = opts

    def run(self):
        try:
            self.progress.emit("正在运行监测闭环…")
            res = run_monitoring_case(self.geometry, self.plan, **self.opts)
            self.progress.emit(f"闭环完成：{res.iterations} 步，"
                                f"收敛={res.converged}")
            self.finished.emit(res, self.geometry)
        except Exception as e:  # pragma: no cover - 依赖物理链
            self.errored.emit(repr(e))


# ===========================================================================
# 主窗口
# ===========================================================================
class MonitoringMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AMForge 监测闭环仪表盘 Monitoring Closed-Loop")
        self.resize(1280, 820)

        self.geometry = None
        self.plan = None
        self.result = None

        splitter = QSplitter(Qt.Horizontal)

        # ---- 左控制栏 ----
        left = QWidget()
        left.setFixedWidth(300)
        left_lay = QVBoxLayout(left)
        left_lay.addWidget(self._source_group())
        left_lay.addWidget(self._plan_group())
        left_lay.addWidget(self._loop_group())
        self.info = QTextEdit()
        self.info.setReadOnly(True)
        self.info.setMaximumHeight(150)
        left_lay.addWidget(QLabel("状态 / 日志"))
        left_lay.addWidget(self.info)
        left_lay.addStretch(1)

        # ---- 右仪表盘 ----
        tabs = QTabWidget()
        self.sensor_view = SensorView()
        self.defect_view = DefectView()
        self.corr_view = CorrectionView()
        tabs.addTab(self.sensor_view, "① 传感流回放")
        tabs.addTab(self.defect_view, "② 缺陷监测")
        tabs.addTab(self.corr_view, "③ 工艺修正记录")

        splitter.addWidget(left)
        splitter.addWidget(tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 3)
        self.setCentralWidget(splitter)

    # -- 几何来源 --
    def _source_group(self) -> QGroupBox:
        g = QGroupBox("几何来源 Geometry")
        lay = QGridLayout(g)
        self.btn_demo = QPushButton("构建示例几何（球）")
        self.btn_demo.clicked.connect(self._on_demo)
        lay.addWidget(self.btn_demo, 0, 0, 1, 2)
        self.btn_import = QPushButton("导入前处理配置…")
        self.btn_import.clicked.connect(self._on_import)
        lay.addWidget(self.btn_import, 1, 0, 1, 2)
        self.size_mm = QDoubleSpinBox()
        self.size_mm.setRange(0.2, 50.0)
        self.size_mm.setValue(1.0)
        self.size_mm.setSuffix(" mm")
        lay.addWidget(QLabel("尺寸"), 2, 0)
        lay.addWidget(self.size_mm, 2, 1)
        return g

    # -- 工艺杠杆 --
    def _plan_group(self) -> QGroupBox:
        g = QGroupBox("工艺杠杆 Process Levers")
        lay = QGridLayout(g)
        lay.addWidget(QLabel("功率 W"), 0, 0)
        self.power = QDoubleSpinBox()
        self.power.setRange(20.0, 2000.0)
        self.power.setValue(180.0)
        lay.addWidget(self.power, 0, 1)
        lay.addWidget(QLabel("速度 m/s"), 1, 0)
        self.speed = QDoubleSpinBox()
        self.speed.setRange(0.05, 20.0)
        self.speed.setDecimals(3)
        self.speed.setValue(1.0)
        lay.addWidget(self.speed, 1, 1)
        return g

    # -- 闭环控制 --
    def _loop_group(self) -> QGroupBox:
        g = QGroupBox("监测闭环 Closed-Loop")
        lay = QGridLayout(g)
        lay.addWidget(QLabel("迭代 n"), 0, 0)
        self.n_iter = QSpinBox()
        self.n_iter.setRange(1, 10)
        self.n_iter.setValue(3)
        lay.addWidget(self.n_iter, 0, 1)
        lay.addWidget(QLabel("控制器"), 1, 0)
        from PySide6.QtWidgets import QComboBox
        self.controller = QComboBox()
        self.controller.addItem("inverse (真实优化器)", "inverse")
        self.controller.addItem("rule (规则代理)", "rule")
        self.controller.setCurrentIndex(0)   # 默认实打实 inverse 优化器
        lay.addWidget(self.controller, 1, 1)
        self.chk_recal = QCheckBox("在线重标定材料(回灌模块C)")
        lay.addWidget(self.chk_recal, 2, 0, 1, 2)
        self.btn_run = QPushButton("运行监测闭环")
        self.btn_run.clicked.connect(self._on_run)
        lay.addWidget(self.btn_run, 3, 0, 1, 2)
        return g

    # -- 回调 --
    def _on_demo(self):
        self.geometry = P.build_primitive(
            "sphere", float(self.size_mm.value()), spacing_um=200.0, name="demo")
        lt = 80e-6
        n_layers = self.geometry.layer_count(lt)
        self.plan = P.build_process_plan(
            n_layers, laser_power=float(self.power.value()),
            scan_speed=float(self.speed.value()), layer_thickness=lt,
            hatch_spacing=120e-6, beam_radius=50e-6, absorption=0.4,
            preheat_temp=373.0, rotation_per_layer_deg=67.0)
        self.info.append(
            f"示例几何已构建：shape={tuple(self.geometry.shape)}，层数={n_layers}")

    def _on_import(self):
        d = QFileDialog.getExistingDirectory(self, "选择前处理导出目录")
        if not d:
            return
        try:
            self.geometry, self.plan = P.import_config(d)
            self.info.append(f"已导入：{d}")
        except Exception as e:
            QMessageBox.warning(self, "导入失败", repr(e))

    def _on_run(self):
        if self.geometry is None or self.plan is None:
            QMessageBox.warning(self, "提示", "请先构建或导入几何。")
            return
        opts = dict(
            n_iter=int(self.n_iter.value()),
            recalibrate=bool(self.chk_recal.isChecked()),
            material="316L",
            controller=str(self.controller.currentData()),
            params={"n_grid": 6, "max_layers": 3, "n_sub_cp": 8},
        )
        self._worker = MonitoringWorker(self.geometry, self.plan, opts)
        self._worker.progress.connect(self.info.append)
        self._worker.finished.connect(self._on_done)
        self._worker.errored.connect(lambda e: self.info.append(f"闭环失败：{e}"))
        self._worker.start()

    def _on_done(self, res, geometry):
        self.result = res
        self.geometry = geometry
        self.sensor_view.set_result(res)
        self.defect_view.set_result(res)
        self.corr_view.set_result(res)
        # 缺陷代价趋势
        costs = res.defect_costs()
        self.info.append("缺陷代价趋势: " + " -> ".join(f"{c:.3f}" for c in costs))
        if res.calibration is not None:
            self.info.append("[在线重标定] 已回灌模块 C 生成 CalibrationReport。")


# ===========================================================================
# 三个 matplotlib 视图
# ===========================================================================
class _MplView(QWidget):
    """matplotlib 视图基类：持有一个 FigureCanvas。"""

    def __init__(self, title: str = ""):
        super().__init__()
        self.fig = Figure(figsize=(7.2, 5.2))
        self.canvas = FigureCanvas(self.fig)
        lay = QVBoxLayout(self)
        lay.addWidget(self.canvas)
        self.title = title

    def _redraw(self):
        self.canvas.draw()


class SensorView(_MplView):
    def set_result(self, res):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        names = ("thermal", "meltpool", "photodiode")
        colors = ("C0", "C1", "C2")
        for i, (nm, c) in enumerate(zip(names, colors)):
            frames = np.asarray(res.steps[0].sensor.frames, dtype=np.float64)
            ch = frames[i] if frames.ndim >= 2 and frames.shape[0] > i else frames
            t = np.asarray(res.steps[0].sensor.time, dtype=np.float64)
            ax.plot(t, np.atleast_1d(ch), label=nm, color=c)
        ax.set_xlabel("时间 [s]")
        ax.set_ylabel("归一化传感信号")
        ax.set_title("传感流回放 (Sensor Stream Replay)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        self._redraw()


class DefectView(_MplView):
    def set_result(self, res):
        self.fig.clear()
        n = max(1, len(res.steps))
        probs = np.array(
            [np.asarray(s.monitoring.defect_probs, dtype=np.float64).reshape(-1)
             for s in res.steps], dtype=np.float64)
        names = res.steps[0].monitoring.defect_names
        # 左：各迭代折线
        ax1 = self.fig.add_subplot(1, 2, 1)
        for j, nm in enumerate(names):
            ax1.plot(range(n), probs[:, j] if probs.shape[1] > j else probs[:, 0],
                     marker="o", label=nm)
        ax1.set_xlabel("迭代")
        ax1.set_ylabel("缺陷概率")
        ax1.set_title("缺陷概率随迭代")
        ax1.set_ylim(0, 1)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        # 右：最终柱状
        ax2 = self.fig.add_subplot(1, 2, 2)
        final = probs[-1]
        ax2.bar([str(x) for x in names], final, color=["C0", "C1", "C2"][:len(final)])
        ax2.set_ylabel("概率")
        ax2.set_title("最终缺陷概率")
        ax2.set_ylim(0, 1)
        self.fig.tight_layout()
        self._redraw()


class CorrectionView(QWidget):
    """工艺修正记录：上=功率/速度轨迹，下=文字修正日志。"""

    def __init__(self):
        super().__init__()
        self.fig = Figure(figsize=(7.2, 3.2))
        self.canvas = FigureCanvas(self.fig)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        lay = QVBoxLayout(self)
        lay.addWidget(self.canvas, 3)
        lay.addWidget(self.log, 2)

    def set_result(self, res):
        # 轨迹
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        Pp, Vv = [], []
        for s in res.steps:
            cp = s.corrected.corrected
            Pp.append(float(np.mean(np.atleast_1d(np.asarray(cp.laser_power)))))
            Vv.append(float(np.mean(np.atleast_1d(np.asarray(cp.scan_speed)))))
        ax.plot(range(len(Pp)), Pp, "o-", color="C0", label="激光功率 W")
        ax2 = ax.twinx()
        ax2.plot(range(len(Vv)), Vv, "s-", color="C3", label="扫描速度 m/s")
        ax.set_xlabel("迭代")
        ax.set_ylabel("激光功率 [W]", color="C0")
        ax2.set_ylabel("扫描速度 [m/s]", color="C3")
        ax.set_title("修正后工艺轨迹")
        ax.grid(True, alpha=0.3)
        self.canvas.draw()
        # 文字日志
        lines = []
        for s in res.steps:
            lines.append(f"[迭代 {s.iteration}] {s.corrected.rationale}")
            lines.append(f"    缺陷代价={s.defect_cost:.3f}  "
                         f"强度安全系数={float(s.verdict.strength_safety_factor):.3f} "
                         f"passed={bool(s.verdict.passed)}")
        if res.calibration is not None:
            lines.append("[在线重标定] 已回灌模块 C -> CalibrationReport")
        self.log.setPlainText("\n".join(lines))
