"""AMForge 后处理 / 数字样机 / 可微优化 GUI（PySide6 + VTK）

界面布局（主窗口 = 左控制 + 右标签页）：

* **左控制栏**：几何来源（示例几何 / 导入前处理配置）、「运行仿真」、可微优化
  （算法选择 + 规模 + 「运行优化」）、状态区。
* **标签页 ① 场浏览器**：体素场下拉选择 + 构建方向切片滑块 + 2D 云图（matplotlib，
  始终可用）+ 3D 场着色面（VTK，无 OpenGL 时回退占位）。
* **标签页 ② 数字样机**：关节弯曲刚度柱状图 + 稳定性/装配评分 + 铰接链示意
  （由 :func:`amforge.gui.postproc.assembly_chain` 生成）。
* **标签页 ③ 可微优化**：loss 历史曲线 / NSGA-II Pareto 前沿散点（含 utopia 与
  best_compromise 标记）。

设计要点
--------
* 所有重计算（``simulate`` / 优化器）都委托给 :mod:`amforge.gui.postproc` 纯逻辑层，
  界面只负责呈现与后台线程调用，核心算法有独立 pytest 守住。
* VTK / matplotlib 在无显示环境不可用时各自回退，模块可导入、可测。
* 仅当用户显式运行 ``amforge gui`` 时才 import 本模块（PySide6/vtk 是重依赖）。
"""

from __future__ import annotations

import os
import sys
import threading
import warnings

import numpy as np

from amforge.gui import preproc as P
from amforge.gui import postproc as PP

warnings.filterwarnings("ignore")

# --- matplotlib 必须在导入 Qt 后端前设定 -------------------------------------
import matplotlib  # noqa: E402
matplotlib.use("QtAgg")  # noqa: E402
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
import matplotlib.cm as mpl_cm  # noqa: E402

from PySide6.QtCore import Qt, QThread, Signal, QTimer  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QComboBox, QPushButton, QFileDialog, QSlider, QTextEdit,
    QSplitter, QTabWidget, QDoubleSpinBox, QSpinBox, QMessageBox, QSizePolicy,
    QCheckBox,
)

VTK_AVAILABLE = True
try:
    import vtk  # noqa: E402
    from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor  # noqa: E402
except Exception as _vtk_err:  # pragma: no cover - 无 OpenGL 环境
    VTK_AVAILABLE = False
    _vtk_err_msg = str(_vtk_err)


# ===========================================================================
# 后台工作线程（避免阻塞 GUI 主线程）
# ===========================================================================
class SimWorker(QThread):
    progress = Signal(str)
    finished = Signal(object, object)   # out, geometry
    errored = Signal(str)

    def __init__(self, geometry, plan, opts):
        super().__init__()
        self.geometry = geometry
        self.plan = plan
        self.opts = opts

    def run(self):
        try:
            self.progress.emit("正在运行端到端仿真…")
            out = PP.run_simulation(self.geometry, self.plan, **self.opts)
            self.finished.emit(out, self.geometry)
        except Exception as e:  # pragma: no cover - 依赖物理链
            self.errored.emit(repr(e))


class OptWorker(QThread):
    progress = Signal(str)
    finished = Signal(dict, str)         # result, kind
    errored = Signal(str)

    def __init__(self, target, kind, opts):
        super().__init__()
        self.target = target
        self.kind = kind
        self.opts = opts

    def run(self):
        try:
            from amforge.inverse import (
                optimize_dimensional, optimize_geometry_process,
                pareto_optimize_geometry_process,
            )
            self.progress.emit(f"正在运行 {self.kind} 优化…")
            if self.kind == "dimensional":
                res = optimize_dimensional(self.target, **self.opts)
            elif self.kind == "joint":
                res = optimize_geometry_process(self.target, **self.opts)
            else:  # pareto
                res = pareto_optimize_geometry_process(self.target, **self.opts)
            self.finished.emit(res, self.kind)
        except Exception as e:  # pragma: no cover - 依赖物理链
            self.errored.emit(repr(e))


# ===========================================================================
# 标签页 ① 场浏览器
# ===========================================================================
class FieldView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.out = None
        self.geometry = None
        self.field_names: list[str] = []

        self.combo = QComboBox()
        self.combo.currentTextChanged.connect(self._on_field)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0); self.slider.setValue(0)
        self.slider.valueChanged.connect(self._on_slice)

        self.figure = Figure(figsize=(4.2, 4.2))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel("场:")); top.addWidget(self.combo, 1)
        lay.addLayout(top)
        lay.addWidget(QLabel("构建方向切片:"))
        lay.addWidget(self.slider)
        lay.addWidget(self.canvas)

        # 扫描动画（温度场时间演化；需求解器 record_frames=True 才非空）
        self.frames: list = []
        self.play_idx = 0
        self.play_timer = QTimer(self)
        self.play_timer.setInterval(120)
        self.play_timer.timeout.connect(self._on_play_tick)
        anim_box = QGroupBox("扫描温度动画")
        alay = QVBoxLayout(anim_box)
        self.btn_play = QPushButton("▶ 播放")
        self.btn_play.clicked.connect(self._on_play)
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0); self.frame_slider.setValue(0)
        self.frame_slider.valueChanged.connect(self._on_frame_slider)
        arow = QHBoxLayout()
        self.btn_gif = QPushButton("导出 GIF")
        self.btn_gif.clicked.connect(self._on_export_gif)
        self.btn_pvd = QPushButton("导出 PVD")
        self.btn_pvd.clicked.connect(self._on_export_pvd)
        arow.addWidget(self.btn_gif); arow.addWidget(self.btn_pvd)
        alay.addWidget(self.btn_play)
        alay.addWidget(QLabel("时间帧:"))
        alay.addWidget(self.frame_slider)
        alay.addLayout(arow)
        lay.addWidget(anim_box)
        self._set_anim_enabled(False)

        # 可选 3D 视图（无 VTK / 无显示环境时占位）
        # 注意：offscreen（QT_QPA_PLATFORM=offscreen）无 OpenGL 上下文，VTK 的
        # Render() 会触发原生 access violation，且无法被 Python try/except 捕获、
        # 直接终止进程；因此显式跳过 VTK 3D 视图，仅保留 2D matplotlib 云图。
        self.view3d = None
        if VTK_AVAILABLE and os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            try:
                self.view3d = Field3DSurface(self)
                lay.addWidget(self.view3d)
            except Exception:  # pragma: no cover
                self.view3d = None

    def set_result(self, out, geometry):
        self.out = out
        self.geometry = geometry
        self.frames = PP.extract_temperature_frames(out, geometry)
        self._set_anim_enabled(bool(self.frames))
        if self.frames:
            self.frame_slider.setMaximum(max(0, len(self.frames) - 1))
            self.frame_slider.setValue(0)
            self.play_idx = 0
        self.field_names = PP.grid_fields(out, geometry)
        self.combo.clear()
        self.combo.addItems(self.field_names)
        nz = geometry.shape[-1] if geometry.dim == 3 else 1
        self.slider.setMaximum(max(0, nz - 1))
        self.slider.setValue(nz // 2)
        if self.field_names:
            self._show(self.field_names[0], self.slider.value())
        if self.view3d is not None and self.field_names:
            try:
                self.view3d.set_geometry(geometry)
                fld = PP.get_field(out, self.field_names[0], geometry)
                self.view3d.set_field(fld)
            except Exception:  # pragma: no cover
                pass

    def _on_field(self, name):
        if name and self.out is not None:
            self._show(name, self.slider.value())
            if self.view3d is not None:
                try:
                    self.view3d.set_field(
                        PP.get_field(self.out, name, self.geometry))
                except Exception:  # pragma: no cover
                    pass

    def _on_slice(self, idx):
        if self.combo.currentText() and self.out is not None:
            self._show(self.combo.currentText(), idx)

    def _show(self, name, slice_idx):
        fld = PP.get_field(self.out, name, self.geometry)
        data = fld.data
        self.ax.clear()
        if fld.dim == 3:
            k = min(max(slice_idx, 0), data.shape[-1] - 1)
            sl = data[..., k]
            ext = [float(fld.origin[0]), float(fld.origin[0]) + fld.spacing * (data.shape[0] - 1),
                   float(fld.origin[1]), float(fld.origin[1]) + fld.spacing * (data.shape[1] - 1)]
            im = self.ax.imshow(sl.T, origin="lower", extent=ext, cmap=fld.cmap,
                                aspect="equal", vmin=fld.min(), vmax=fld.max())
            self.figure.colorbar(im, ax=self.ax, fraction=0.046, pad=0.04,
                                 label=f"{fld.unit}")
            self.ax.set_title(f"{fld.label}  z={float(fld.origin[-1])+fld.spacing*k*1e3:.3f} mm")
        else:
            ext = [float(fld.origin[0]), float(fld.origin[0]) + fld.spacing * (data.shape[0] - 1),
                   float(fld.origin[1]), float(fld.origin[1]) + fld.spacing * (data.shape[1] - 1)]
            im = self.ax.imshow(data.T, origin="lower", extent=ext, cmap=fld.cmap,
                                aspect="equal", vmin=fld.min(), vmax=fld.max())
            self.figure.colorbar(im, ax=self.ax, fraction=0.046, pad=0.04,
                                 label=f"{fld.unit}")
            self.ax.set_title(fld.label)
        self.ax.set_xlabel("x [m]"); self.ax.set_ylabel("y [m]")
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.canvas.draw()

    # -- 扫描温度动画 --
    def _set_anim_enabled(self, enabled: bool) -> None:
        for w in (self.btn_play, self.frame_slider, self.btn_gif, self.btn_pvd):
            w.setEnabled(enabled)

    def _show_frame(self, idx: int) -> None:
        if not self.frames:
            return
        fld = self.frames[idx]
        data = fld.data
        self.ax.clear()
        ext = [float(fld.origin[0]),
               float(fld.origin[0]) + fld.spacing * (data.shape[0] - 1),
               float(fld.origin[1]),
               float(fld.origin[1]) + fld.spacing * (data.shape[1] - 1)]
        if fld.dim == 3:
            k = min(max(self.slider.value(), 0), data.shape[-1] - 1)
            sl = data[..., k]
            im = self.ax.imshow(sl.T, origin="lower", extent=ext, cmap="inferno",
                                aspect="equal", vmin=fld.min(), vmax=fld.max())
            self.ax.set_title(
                f"温度演化 帧 {idx}/{len(self.frames)-1}  "
                f"z={float(fld.origin[-1])+fld.spacing*k*1e3:.3f} mm")
        else:
            im = self.ax.imshow(data.T, origin="lower", extent=ext, cmap="inferno",
                                aspect="equal", vmin=fld.min(), vmax=fld.max())
            self.ax.set_title(f"温度演化 帧 {idx}/{len(self.frames)-1}")
        self.ax.set_xlabel("x [m]"); self.ax.set_ylabel("y [m]")
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.figure.colorbar(im, ax=self.ax, fraction=0.046, pad=0.04, label="K")
        self.canvas.draw()

    def _on_play(self) -> None:
        if not self.frames:
            return
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.btn_play.setText("▶ 播放")
        else:
            if self.play_idx >= len(self.frames) - 1:
                self.play_idx = 0
            self.play_timer.start()
            self.btn_play.setText("⏸ 暂停")

    def _on_play_tick(self) -> None:
        if not self.frames:
            return
        self.play_idx = (self.play_idx + 1) % len(self.frames)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.play_idx)
        self.frame_slider.blockSignals(False)
        self._show_frame(self.play_idx)

    def _on_frame_slider(self, idx: int) -> None:
        self.play_idx = idx
        self._show_frame(idx)

    def _on_export_gif(self) -> None:
        if not self.frames:
            return
        d = QFileDialog.getExistingDirectory(self, "选择 GIF 导出目录")
        if not d:
            return
        path = PP.save_field_gif(self.frames, os.path.join(d, "scan_thermal.gif"))
        QMessageBox.information(self, "已导出", f"扫描温度 GIF:\n{path}")

    def _on_export_pvd(self) -> None:
        if not self.frames:
            return
        d = QFileDialog.getExistingDirectory(self, "选择 PVD 导出目录")
        if not d:
            return
        path = PP.save_field_pvd(self.frames, os.path.join(d, "scan_thermal.pvd"))
        QMessageBox.information(self, "已导出", f"ParaView 时间序列:\n{path}")


if VTK_AVAILABLE:
    class Field3DSurface(QWidget):
        """3D：把几何 SDF 面用所选体素场着色（vtkProbeFilter 采样）。无显示时回退。"""

        def __init__(self, parent=None):
            super().__init__(parent)
            self._layout = QVBoxLayout(self)
            self.vtk_widget = QVTKRenderWindowInteractor(self)
            self.renderer = vtk.vtkRenderer()
            self.renderer.SetBackground(0.12, 0.13, 0.15)
            self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
            self._layout.addWidget(self.vtk_widget)
            self._geometry = None
            self._surface_poly = None

        def set_geometry(self, geometry):
            """构建几何 SDF 等值面（实体 = SDF<0），作为着色的基底。"""
            self._geometry = geometry
            sdf = np.asarray(geometry.sdf)
            occ = (sdf < 0.0).astype(np.uint8)
            img = PP_scalar_to_image(occ, geometry.origin, geometry.spacing)
            surf = vtk.vtkFlyingEdges3D()
            surf.SetInputData(img)
            surf.SetValue(0, 0.5)
            surf.Update()
            self._surface_poly = surf.GetOutput()

        def set_field(self, fld: "PP.VolumeField"):
            if self._surface_poly is None and self._geometry is not None:
                self.set_geometry(self._geometry)
            if self._surface_poly is None:
                return
            fimg = PP_scalar_to_image(np.asarray(fld.data, dtype=np.float32),
                                     fld.origin, fld.spacing)
            probe = vtk.vtkProbeFilter()
            probe.SetInputData(self._surface_poly)
            probe.SetSourceData(fimg)
            probe.Update()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(probe.GetOutputPort())
            vmin, vmax = float(np.nanmin(fld.data)), float(np.nanmax(fld.data))
            mapper.SetScalarRange(vmin, vmax if vmax > vmin else vmin + 1.0)
            lut = vtk.vtkLookupTable()
            lut.SetNumberOfTableValues(64)
            lut.SetHueRange(0.667, 0.0)  # 蓝→红
            lut.Build()
            mapper.SetLookupTable(lut)
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            self.renderer.RemoveAllViewProps()
            self.renderer.AddActor(actor)
            self.renderer.ResetCamera()
            # 仅在真实显示环境下渲染：offscreen 无 GL 上下文时 Render() 会触发原生
            # access violation（无法被 Python try/except 捕获），故显式跳过。
            if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
                try:
                    self.vtk_widget.GetRenderWindow().Render()
                except Exception:  # pragma: no cover - 无显示
                    pass


def PP_scalar_to_image(arr: np.ndarray, origin, spacing):
    """把 numpy 数组转成 vtkImageData（用于 VTK 着色面）。"""
    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    sh = arr.shape
    img = vtk.vtkImageData()
    img.SetDimensions(sh[0], sh[1], sh[2] if arr.ndim == 3 else 1)
    img.AllocateScalars(vtk.VTK_FLOAT, 1)
    buf = np.frombuffer(img.GetPointData().GetScalars(), dtype=np.float32).reshape(-1)
    buf[:] = arr.ravel(order="F")[: buf.shape[0]]
    img.GetPointData().GetScalars().Modified()
    sp = float(spacing)
    img.SetSpacing(sp, sp, sp)
    img.SetOrigin(float(origin[0]), float(origin[1]),
                  float(origin[2]) if np.asarray(origin).shape[0] > 2 else 0.0)
    return img


# ===========================================================================
# 标签页 ② 数字样机装配
# ===========================================================================
class AssemblyView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.out = None
        self.geometry = None
        self.figure = Figure(figsize=(5.2, 4.0))
        self.canvas = FigureCanvas(self.figure)
        self.info = QTextEdit(); self.info.setReadOnly(True)
        self.info.setMaximumHeight(120)
        lay = QVBoxLayout(self)
        lay.addWidget(self.info)
        lay.addWidget(self.canvas)

    def set_result(self, out, geometry):
        self.out = out
        self.geometry = geometry
        summ = PP.assembly_summary(out)
        lines = ["数字样机 / 装配体评估", "─" * 40]
        for k in ("strength_safety_factor", "stiffness_ratio", "passed",
                 "stability_margin", "assembly_score", "assembly_porosity",
                 "n_joints", "meltpool_defect_score"):
            if k in summ:
                v = summ[k]
                lines.append(f"  {k:24s}: {v}")
        if summ.get("passed"):
            lines.append("  判定: ✅ 合格")
        else:
            lines.append("  判定: ❌ 不合格")
        self.info.setPlainText("\n".join(lines))
        self._draw()

    def _draw(self):
        self.figure.clear()
        ax1 = self.figure.add_subplot(121)
        ax2 = self.figure.add_subplot(122)
        chain = PP.assembly_chain(self.out, self.geometry)
        pos = np.asarray(chain["positions"])
        kf = np.asarray(chain["flexural_stiffness"], dtype=np.float64)
        rd = np.asarray(chain["relative_displacement"], dtype=np.float64)
        n = pos.shape[0]
        # 柱状：关节弯曲刚度
        if n:
            ax1.bar(range(n), kf, color="#3a86ff")
            ax1.set_title("关节弯曲刚度 K [N·m/rad]")
            ax1.set_xlabel("关节 #")
        else:
            ax1.text(0.5, 0.5, "无装配结果", ha="center")
        # 链示意：沿 z 画节点 + 段，节点颜色按 |相对位移|
        if n:
            rd_abs = np.abs(rd)
            norm = rd_abs / (np.max(rd_abs) + 1e-30)
            for (i, j) in chain["links"]:
                ax2.plot([pos[i, 2], pos[j, 2]],
                         [pos[i, 0], pos[j, 0]], color="#888", lw=2, zorder=1)
            sc = ax2.scatter(pos[:, 2], pos[:, 0], c=norm, cmap="autumn",
                             s=80, zorder=2)
            ax2.set_title("铰接链（节点色 = |Δ|）")
            ax2.set_xlabel("z [m]"); ax2.set_ylabel("x [m]")
            self.figure.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04,
                                 label="|相对位移|")
        else:
            ax2.text(0.5, 0.5, "无装配结果", ha="center")
        self.figure.tight_layout()
        self.canvas.draw()


# ===========================================================================
# 标签页 ③ 可微优化
# ===========================================================================
class OptimizationView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.result = None
        self.kind = "pareto"
        self.figure = Figure(figsize=(5.2, 4.0))
        self.canvas = FigureCanvas(self.figure)
        self.info = QTextEdit(); self.info.setReadOnly(True)
        self.info.setMaximumHeight(120)
        lay = QVBoxLayout(self)
        lay.addWidget(self.info)
        lay.addWidget(self.canvas)

    def set_result(self, result, kind):
        self.result = result
        self.kind = kind
        agg = PP.optimization_summary(result, kind)
        if kind == "pareto":
            lines = [f"Pareto 前沿（算法={agg.get('algorithm')}）", "─" * 40,
                     f"  前沿点数: {agg['n_front']}",
                     f"  utopia: {agg.get('utopia')}",
                     f"  best_compromise_index: {agg['best_compromise_index']}"]
        else:
            lh = np.asarray(agg["loss_history"])
            lines = [f"{kind} 优化", "─" * 40,
                     f"  步数: {len(lh)}",
                     f"  loss 初值: {lh[0]:.4e}", f"  loss 末值: {lh[-1]:.4e}",
                     f"  loss 下降: {lh[0]/max(lh[-1],1e-30):.2f}×"]
        self.info.setPlainText("\n".join(lines))
        self._draw(agg)

    def _draw(self, agg):
        self.figure.clear()
        if self.kind == "pareto":
            g = np.asarray(agg["geom_dev"], dtype=np.float64)
            s = np.asarray(agg["stress"], dtype=np.float64)
            feas = np.asarray(agg["feasible"], dtype=np.float64)
            ax = self.figure.add_subplot(111)
            sc = ax.scatter(g, s, c=feas, cmap="viridis", s=70,
                            edgecolors="#333", zorder=2)
            bi = int(agg["best_compromise_index"])
            if 0 <= bi < len(g):
                ax.scatter([g[bi]], [s[bi]], marker="*", s=260, c="#ff5a3c",
                           edgecolors="#000", zorder=3, label="best compromise")
            ut = agg.get("utopia")
            if ut is not None:
                ax.scatter([ut[0]], [ut[1]], marker="D", s=90, c="#00d26a",
                           edgecolors="#000", zorder=3, label="utopia")
            ax.set_xlabel("尺寸偏差 geom_dev")
            ax.set_ylabel("残余应力 stress")
            ax.set_title("NSGA-II Pareto 前沿")
            ax.legend(loc="best", fontsize=8)
            self.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.04,
                                 label="feasible")
        else:
            ax = self.figure.add_subplot(111)
            lh = np.asarray(agg["loss_history"], dtype=np.float64)
            ax.plot(lh, color="#3a86ff", lw=2, label="loss")
            has_geom = "geom_history" in agg
            if has_geom and agg["geom_history"] is not None:
                gh = np.asarray(agg["geom_history"], dtype=np.float64)
                if gh.ndim >= 1:
                    ax.plot(gh, color="#ff5a3c", lw=1.5, label="geom_dev",
                            alpha=0.8)
            ax.set_xlabel("step"); ax.set_ylabel("指标")
            ax.set_yscale("log")
            ax.set_title(f"{self.kind} 优化历史")
            ax.legend(loc="best", fontsize=8)
        self.figure.tight_layout()
        self.canvas.draw()


# ===========================================================================
# 主窗口
# ===========================================================================
class PostprocMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AMForge 后处理 / 数字样机 / 可微优化")
        self.resize(1280, 820)

        self.geometry = None
        self.plan = None
        self.out = None

        splitter = QSplitter(Qt.Horizontal)

        # ---- 左控制栏 ----
        left = QWidget()
        left.setFixedWidth(300)
        left_lay = QVBoxLayout(left)
        left_lay.addWidget(self._source_group())
        left_lay.addWidget(self._sim_group())
        left_lay.addWidget(self._opt_group())
        left_lay.addWidget(self._secondary_group())
        self.info = QTextEdit(); self.info.setReadOnly(True)
        self.info.setMaximumHeight(150)
        left_lay.addWidget(QLabel("状态 / 日志"))
        left_lay.addWidget(self.info)
        left_lay.addStretch(1)

        # ---- 右标签页 ----
        tabs = QTabWidget()
        self.field_view = FieldView()
        self.assembly_view = AssemblyView()
        self.opt_view = OptimizationView()
        tabs.addTab(self.field_view, "① 场浏览器")
        tabs.addTab(self.assembly_view, "② 数字样机")
        tabs.addTab(self.opt_view, "③ 可微优化")

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
        self.size_mm = QDoubleSpinBox(); self.size_mm.setRange(0.2, 50.0)
        self.size_mm.setValue(1.0); self.size_mm.setSuffix(" mm")
        lay.addWidget(QLabel("尺寸"), 2, 0)
        lay.addWidget(self.size_mm, 2, 1)
        return g

    # -- 运行仿真 --
    def _sim_group(self) -> QGroupBox:
        g = QGroupBox("仿真 Simulation")
        lay = QVBoxLayout(g)
        self.btn_run = QPushButton("运行端到端仿真")
        self.btn_run.clicked.connect(self._on_run_sim)
        lay.addWidget(self.btn_run)
        self.btn_export = QPushButton("导出场 (VTK/CSV)…")
        self.btn_export.clicked.connect(self._on_export)
        lay.addWidget(self.btn_export)
        self.chk_anim = QCheckBox("记录扫描温度动画帧")
        self.chk_anim.setChecked(True)
        lay.addWidget(self.chk_anim)
        return g

    # -- 优化 --
    def _opt_group(self) -> QGroupBox:
        g = QGroupBox("可微优化 Optimization")
        lay = QGridLayout(g)
        lay.addWidget(QLabel("类型"), 0, 0)
        self.opt_kind = QComboBox()
        self.opt_kind.addItems(["pareto", "joint", "dimensional"])
        lay.addWidget(self.opt_kind, 0, 1)
        lay.addWidget(QLabel("pop"), 1, 0)
        self.pop = QSpinBox(); self.pop.setRange(4, 64); self.pop.setValue(8)
        lay.addWidget(self.pop, 1, 1)
        lay.addWidget(QLabel("gen"), 2, 0)
        self.gen = QSpinBox(); self.gen.setRange(2, 40); self.gen.setValue(6)
        lay.addWidget(self.gen, 2, 1)
        self.btn_opt = QPushButton("运行优化")
        self.btn_opt.clicked.connect(self._on_run_opt)
        lay.addWidget(self.btn_opt, 3, 0, 1, 2)
        return g

    # -- 回调 --
    def _on_demo(self):
        self.geometry = P.build_primitive(
            "sphere", float(self.size_mm.value()), spacing_um=200.0, name="demo")
        lt = 80e-6
        n_layers = self.geometry.layer_count(lt)
        self.plan = P.build_process_plan(
            n_layers, laser_power=180.0, scan_speed=1.0, layer_thickness=lt,
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

    def _on_run_sim(self):
        if self.geometry is None or self.plan is None:
            QMessageBox.warning(self, "提示", "请先构建或导入几何。")
            return
        opts = dict(material="316L", asbuilt_solver="buildup",
                    micro_solver="surrogate", mbd_solver="surrogate",
                    params={"n_grid": 6, "max_layers": 3, "n_sub_cp": 8,
                            "record_frames": bool(self.chk_anim.isChecked())})
        self._sim_thread = SimWorker(self.geometry, self.plan, opts)
        self._sim_thread.progress.connect(self.info.append)
        self._sim_thread.finished.connect(self._on_sim_done)
        self._sim_thread.errored.connect(lambda e: self.info.append(f"仿真失败：{e}"))
        self._sim_thread.start()

    def _on_sim_done(self, out, geometry):
        self.out = out
        self.geometry = geometry
        self.field_view.set_result(out, geometry)
        self.assembly_view.set_result(out, geometry)
        self.info.append("仿真完成：场浏览器 / 数字样机 已刷新。")
        # 标量摘要
        s = PP.scalar_summary(out)
        self.info.append(f"  强度安全系数={s.get('strength_safety_factor'):.3f} "
                         f"passed={s.get('passed')}")

    def _on_export(self):
        if self.out is None or self.geometry is None:
            QMessageBox.warning(self, "提示", "请先运行仿真。")
            return
        d = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if not d:
            return
        manifest = PP.export_all_fields(self.out, self.geometry, d)
        self.info.append(f"已导出 {manifest['count']} 个体素场到 {d}"
                         f"（含 manifest.json，ParaView 可直接打开 .vtk）")

    # -- 二次工艺（模块B） --
    def _secondary_group(self) -> QGroupBox:
        g = QGroupBox("二次工艺 Secondary Process")
        lay = QVBoxLayout(g)
        row = QHBoxLayout()
        row.addWidget(QLabel("工艺:"))
        self.sec_kind = QComboBox()
        self.sec_kind.addItems(["HT", "HIP", "machining"])
        row.addWidget(self.sec_kind, 1)
        lay.addLayout(row)
        self.btn_sec = QPushButton("应用二次工艺")
        self.btn_sec.clicked.connect(self._on_apply_secondary)
        lay.addWidget(self.btn_sec)
        self.sec_info = QTextEdit(); self.sec_info.setReadOnly(True)
        self.sec_info.setMaximumHeight(90)
        lay.addWidget(self.sec_info)
        # 松弛后残余应力云图（2D matplotlib，无显示环境也可构造/绘制）
        self.sec_fig = Figure(figsize=(3.4, 3.0))
        self.sec_canvas = FigureCanvas(self.sec_fig)
        self.sec_ax = self.sec_fig.add_subplot(111)
        lay.addWidget(self.sec_canvas)
        return g

    def _on_apply_secondary(self):
        if self.out is None or self.geometry is None:
            QMessageBox.warning(self, "提示", "请先运行仿真。")
            return
        asbuilt = self.out.get("asbuilt")
        if asbuilt is None:
            QMessageBox.warning(self, "提示", "结果无 asbuilt 成形件。")
            return
        treatment = self.sec_kind.currentText()
        try:
            r = PP.apply_secondary(asbuilt, treatment=treatment,
                                   geometry=self.geometry)
        except Exception as e:  # pragma: no cover - 依赖物理链
            QMessageBox.warning(self, "二次工艺失败", repr(e))
            return
        self.sec_info.setPlainText(
            f"工艺={r['treatment']}  relief={r['relief_factor']:.3f}  "
            f"density={r['density_after']:.4f}\n"
            f"松弛后残余应力峰值(vM)={float(np.max(r['relaxed_vm'])):.3e} Pa")
        self._sec_draw(r["relaxed_vm"])

    def _sec_draw(self, vm):
        self.sec_ax.clear()
        a = np.asarray(vm, dtype=np.float64)
        sl = a[..., a.shape[-1] // 2] if a.ndim == 3 else a
        vmin, vmax = float(np.nanmin(sl)), float(np.nanmax(sl))
        if vmax <= vmin:
            vmax = vmin + 1.0
        im = self.sec_ax.imshow(sl.T, origin="lower", cmap="magma",
                                aspect="equal", vmin=vmin, vmax=vmax)
        self.sec_fig.colorbar(im, ax=self.sec_ax, fraction=0.046, pad=0.04,
                              label="Pa")
        self.sec_ax.set_title("松弛后残余应力(vM)")
        self.sec_ax.set_xticks([]); self.sec_ax.set_yticks([])
        self.sec_canvas.draw()

    def _on_run_opt(self):
        if self.geometry is None:
            QMessageBox.warning(self, "提示", "请先构建或导入几何。")
            return
        kind = self.opt_kind.currentText()
        if kind == "pareto":
            opts = dict(pop_size=int(self.pop.value()), n_gen=int(self.gen.value()),
                        n_steps=6, algorithm="nsga2", hard_project=True,
                        asbuilt_solver="buildup", micro_solver="surrogate",
                        mbd_solver="surrogate",
                        params={"n_grid": 6, "max_layers": 3})
        else:
            opts = dict(n_steps=12, asbuilt_solver="buildup",
                        micro_solver="surrogate", mbd_solver="surrogate",
                        params={"n_grid": 6, "max_layers": 3})
        self._opt_thread = OptWorker(self.geometry, kind, opts)
        self._opt_thread.progress.connect(self.info.append)
        self._opt_thread.finished.connect(self._on_opt_done)
        self._opt_thread.errored.connect(lambda e: self.info.append(f"优化失败：{e}"))
        self._opt_thread.start()

    def _on_opt_done(self, result, kind):
        self.opt_view.set_result(result, kind)
        self.info.append(f"{kind} 优化完成，见「③ 可微优化」标签页。")


# ===========================================================================
# 入口
# ===========================================================================
def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    win = PostprocMainWindow()
    win.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
