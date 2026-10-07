"""AMForge 前处理 GUI（PySide6 + VTK）

界面布局（左控制 / 中 3D / 右分层切片+扫描路径）：

* 左栏：几何源（球/柱/方 解析体素化 或 STL 导入）、工艺参数面板、
  「构建几何」「评估工艺」「导出配置」按钮与状态区。
* 中栏：VTK 三维视图，显示零件等值面（SDF=0 界面）。
* 右栏：层滑块 + matplotlib 二维切片，叠加 hatch 光栅扫描路径。

设计要点
--------
* 3D 视图用 ``QVTKRenderWindowInteractor``；若运行环境无 OpenGL 上下文（如
  无显示的 CI/沙箱），会回退为占位 QLabel，**不崩溃**，保证模块可导入、可测。
* 所有重计算都委托给 :mod:`amforge.gui.preproc` 纯逻辑层，界面只负责呈现，
  因此核心算法有独立 pytest 守住。
* 仅当用户显式运行 ``amforge gui`` 时才 import 本模块（PySide6/vtk 是重依赖）。

运行：``amforge gui`` 或 ``python -m amforge gui``。
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np

from amforge.core.contracts import SDF_SOLID_TOL
from amforge.gui import preproc as P
from amforge.boundary import (
    BoundaryCondition, BoundaryCollection, InitialCondition, face_mask,
)
from amforge.powder import ParticleCollection

warnings.filterwarnings("ignore")

# --- matplotlib 必须在导入 Qt 后端前设定 -------------------------------------
import matplotlib  # noqa: E402
matplotlib.use("QtAgg")  # noqa: E402
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QComboBox, QDoubleSpinBox, QPushButton, QFileDialog,
    QSlider, QTextEdit, QSplitter, QMessageBox, QSizePolicy, QListWidget,
    QCheckBox,
)

VTK_AVAILABLE = True
try:
    import vtk  # noqa: E402
    from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor  # noqa: E402
except Exception as _vtk_err:  # pragma: no cover - 无 OpenGL 环境
    VTK_AVAILABLE = False
    _vtk_err_msg = str(_vtk_err)


# ---------------------------------------------------------------------------
# 3D 视图（VTK，带无显示回退）
# ---------------------------------------------------------------------------
class Part3DView(QWidget):
    """零件三维等值面视图；无 VTK/OpenGL 时显示占位文字。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        # offscreen（无 OpenGL 上下文）下不创建 VTK 交互窗口：Render() 会触发原生
        # access violation 且无法被 Python try/except 捕获、直接终止进程；用占位说明替代。
        self.vtk_widget = None
        if not VTK_AVAILABLE or os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            self._placeholder = QLabel(
                "3D 视图需要 VTK + OpenGL（当前环境不可用 / offscreen）。\n"
                "二维切片 + 扫描路径视图仍可正常使用。"
            )
            self._placeholder.setAlignment(Qt.AlignCenter)
            self._layout.addWidget(self._placeholder)
            return
        try:
            self.vtk_widget = QVTKRenderWindowInteractor(self)
            self.renderer = vtk.vtkRenderer()
            self.renderer.SetBackground(0.12, 0.13, 0.15)
            self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
            self._layout.addWidget(self.vtk_widget)
        except Exception as exc:  # pragma: no cover
            self.vtk_widget = None
            self._placeholder = QLabel(f"VTK 初始化失败：{exc}")
            self._placeholder.setAlignment(Qt.AlignCenter)
            self._layout.addWidget(self._placeholder)

    def show_part(self, part) -> None:
        if self.vtk_widget is None or not VTK_AVAILABLE:
            return
        # offscreen 无 GL 上下文，跳过 VTK 渲染（Render 会原生崩溃）
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            return
        # 清掉旧 actor
        self.renderer.RemoveAllViewProps()
        sdf = np.asarray(part.sdf)
        occ = (sdf < SDF_SOLID_TOL).astype(np.uint8)   # 与求解器同口径（#19）
        nx, ny, nz = occ.shape
        img = vtk.vtkImageData()
        img.SetDimensions(nx, ny, nz)
        img.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        buf = np.frombuffer(img.GetPointData().GetScalars(), dtype=np.uint8).reshape(-1)
        buf[:] = occ.ravel()
        img.GetPointData().GetScalars().Modified()
        sp = float(part.spacing)
        img.SetSpacing(sp, sp, sp)
        img.SetOrigin(float(part.origin[0]), float(part.origin[1]), float(part.origin[2]))

        surf = vtk.vtkFlyingEdges3D()
        surf.SetInputData(img)
        surf.SetValue(0, 0.5)
        surf.Update()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(surf.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.20, 0.52, 0.90)
        self.renderer.AddActor(actor)
        self.renderer.ResetCamera()
        try:
            self.vtk_widget.GetRenderWindow().Render()
        except Exception:  # pragma: no cover - 无显示
            pass


# ---------------------------------------------------------------------------
# 二维切片 + hatch 扫描路径视图
# ---------------------------------------------------------------------------
class SliceView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(4, 4))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        lay = QVBoxLayout(self)
        lay.addWidget(self.canvas)
        self.hatch = None
        self.layer = 0
        self._highlight = None          # (nx, ny) 选面掩膜（叠加高亮）
        self._particles = None          # (xs, ys) 粉末粒子散点（叠加红点）
        self._support = None            # (nx, ny) 支撑足迹掩膜（叠加蓝层）

    def set_hatch(self, hatch: dict) -> None:
        self.hatch = hatch

    def set_highlight(self, mask2d) -> None:
        """设置当前层的选面高亮掩膜（None 取消）。"""
        self._highlight = None if mask2d is None else np.asarray(mask2d)
        if self.hatch is not None:
            self.show_layer(self.layer)

    def set_particles(self, xy) -> None:
        """设置当前层附近的粉末粒子 ``(xs, ys)`` 散点（None 取消叠加）。"""
        self._particles = None if xy is None else (np.asarray(xy[0]), np.asarray(xy[1]))
        if self.hatch is not None:
            self.show_layer(self.layer)

    def set_support(self, mask2d) -> None:
        """设置支撑足迹掩膜 (nx, ny)（沿 z 投影），叠加蓝色半透明预览（None 取消）。"""
        self._support = None if mask2d is None else np.asarray(mask2d)
        if self.hatch is not None:
            self.show_layer(self.layer)

    def show_layer(self, idx: int) -> None:
        if self.hatch is None:
            return
        n = self.hatch["n_layers"]
        idx = max(0, min(idx, n - 1))
        self.layer = idx
        mask = self.hatch["masks"][idx]
        xs = self.hatch["xs"]
        ys = self.hatch["ys"]
        self.ax.clear()
        self.ax.imshow(
            mask.T, origin="lower",
            extent=[float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())],
            cmap="gray_r", vmin=0, vmax=1, aspect="equal",
        )
        for (x0, y0), (x1, y1) in self.hatch["paths"][idx]:
            self.ax.plot([x0, x1], [y0, y1], color="#ff5a3c", linewidth=0.6)
        # 选面高亮：红色半透明叠加（仅显示该层与所选面的交集体素）
        if self._highlight is not None:
            m = np.asarray(self._highlight)
            if m.shape == mask.shape:
                red = np.zeros((*m.shape, 4))
                red[..., 0] = 1.0
                red[..., 3] = np.clip(m, 0.0, 1.0) * 0.5
                self.ax.imshow(
                    red.T, origin="lower",
                    extent=[float(xs.min()), float(xs.max()),
                            float(ys.min()), float(ys.max())],
                    alpha=0.5,
                )
        z = float(self.hatch["z_heights"][idx]) * 1e3
        self.ax.set_title(f"layer {idx}/{n-1}  z = {z:.3f} mm")
        self.ax.set_xlabel("x [m]"); self.ax.set_ylabel("y [m]")
        self.ax.set_xticks([]); self.ax.set_yticks([])
        # 支撑足迹叠加（蓝色半透明）：模块A 生成的支撑预览
        if self._support is not None:
            m = np.asarray(self._support)
            if m.shape == mask.shape:
                blue = np.zeros((*m.shape, 4))
                blue[..., 2] = 1.0
                blue[..., 3] = np.clip(m, 0.0, 1.0) * 0.5
                self.ax.imshow(
                    blue.T, origin="lower",
                    extent=[float(xs.min()), float(xs.max()),
                            float(ys.min()), float(ys.max())],
                    alpha=0.5, zorder=4,
                )
        # 粉末粒子散点（红点）：导入/生成的真实粒子点云
        if self._particles is not None:
            xs, ys = self._particles
            if len(xs) > 0:
                self.ax.scatter(xs, ys, s=6, c="#ff3b3b",
                                edgecolors="none", zorder=5, label="powder")
        self.canvas.draw()


# ---------------------------------------------------------------------------
# 主窗口
# ---------------------------------------------------------------------------
class PreprocMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AMForge 前处理 (Pre-processing)")
        self.resize(1280, 800)

        self.part = None
        self.plan = None
        self.bc = BoundaryCollection()     # 当前边界/初值条件集合
        self.bc_highlight = None           # 选面预览（面选择器名）
        self.powder = None                  # 当前导入/生成的粉末粒子集合
        self.support = None                 # 当前生成的支撑结构 (SupportStructure)

        splitter = QSplitter(Qt.Horizontal)

        # ---- 左栏控制 ----
        left = QWidget()
        left.setFixedWidth(320)
        left_lay = QVBoxLayout(left)
        left_lay.addWidget(self._geometry_group())
        left_lay.addWidget(self._process_group())
        left_lay.addWidget(self._boundary_group())
        left_lay.addWidget(self._powder_group())
        left_lay.addWidget(self._support_group())
        self.info = QTextEdit()
        self.info.setReadOnly(True)
        self.info.setMaximumHeight(180)
        left_lay.addWidget(QLabel("状态 / 评估"))
        left_lay.addWidget(self.info)
        left_lay.addStretch(1)

        # ---- 中栏 3D ----
        self.view3d = Part3DView()

        # ---- 右栏 分层切片 ----
        right = QWidget()
        right_lay = QVBoxLayout(right)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0); self.slider.setValue(0)
        self.slider.valueChanged.connect(self._on_layer)
        self.layer_label = QLabel("layer: -")
        right_lay.addWidget(self.layer_label)
        right_lay.addWidget(self.slider)
        self.slice_view = SliceView()
        right_lay.addWidget(self.slice_view)

        splitter.addWidget(left)
        splitter.addWidget(self.view3d)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 1)
        self.setCentralWidget(splitter)

    # -- 几何组 --
    def _geometry_group(self) -> QGroupBox:
        g = QGroupBox("几何 Geometry")
        lay = QGridLayout(g)
        lay.addWidget(QLabel("类型"), 0, 0)
        self.prim = QComboBox(); self.prim.addItems(["sphere", "cylinder", "box"])
        lay.addWidget(self.prim, 0, 1)
        lay.addWidget(QLabel("尺寸 mm"), 1, 0)
        self.size_mm = QDoubleSpinBox(); self.size_mm.setRange(0.2, 50.0)
        self.size_mm.setValue(1.0); self.size_mm.setSuffix(" mm")
        lay.addWidget(self.size_mm, 1, 1)
        lay.addWidget(QLabel("体素 µm"), 2, 0)
        self.spacing_um = QDoubleSpinBox(); self.spacing_um.setRange(20, 500)
        self.spacing_um.setValue(100.0); self.spacing_um.setSuffix(" µm")
        lay.addWidget(self.spacing_um, 2, 1)
        self.btn_build = QPushButton("构建几何")
        self.btn_build.clicked.connect(self._on_build)
        lay.addWidget(self.btn_build, 3, 0, 1, 2)
        self.btn_stl = QPushButton("导入 STL…")
        self.btn_stl.clicked.connect(self._on_import_stl)
        lay.addWidget(self.btn_stl, 4, 0, 1, 2)
        return g

    # -- 工艺组 --
    def _process_group(self) -> QGroupBox:
        g = QGroupBox("工艺 Process (SLM)")
        lay = QGridLayout(g)
        self._spins = {}
        rows = [
            ("laser_power", "激光功率 W", 10.0, 2000.0, 200.0),
            ("scan_speed", "扫描速度 m/s", 0.1, 10.0, 1.0),
            ("layer_thickness", "层厚 µm", 10.0, 200.0, 40.0),
            ("hatch_spacing", "扫描间距 µm", 20.0, 500.0, 80.0),
            ("beam_radius", "光斑半径 µm", 10.0, 300.0, 50.0),
            ("absorption", "吸收率", 0.1, 1.0, 0.4),
            ("preheat_temp", "预热温度 K", 293.0, 1000.0, 373.0),
            ("rotation_per_layer_deg", "层间转角 °", 0.0, 180.0, 67.0),
        ]
        for i, (key, label, lo, hi, dv) in enumerate(rows):
            lay.addWidget(QLabel(label), i, 0)
            sp = QDoubleSpinBox(); sp.setRange(lo, hi); sp.setValue(dv)
            self._spins[key] = sp
            lay.addWidget(sp, i, 1)
        self.btn_assess = QPushButton("评估工艺")
        self.btn_assess.clicked.connect(self._on_assess)
        lay.addWidget(self.btn_assess, len(rows), 0, 1, 2)
        self.btn_save = QPushButton("导出配置…")
        self.btn_save.clicked.connect(self._on_save)
        lay.addWidget(self.btn_save, len(rows) + 1, 0, 1, 2)
        return g

    # -- 边界/初值条件组（缺口 #20-②A） --
    _BC_FACE_ITEMS = ["+Z (顶面)", "-Z (基板/底面)", "+X", "-X", "+Y", "-Y", "全部表面"]
    _BC_KIND_ITEMS = ["Dirichlet 固定温度", "Convection 对流换热", "Flux 热流密度", "Adiabatic 绝热"]

    def _bc_face_code(self, text: str) -> str:
        return text.split()[0]          # "+Z (顶面)" → "+Z"

    def _bc_kind_code(self, text: str) -> str:
        return {
            "Dirichlet 固定温度": "dirichlet",
            "Convection 对流换热": "convection",
            "Flux 热流密度": "flux",
            "Adiabatic 绝热": "adiabatic",
        }[text]

    def _boundary_group(self) -> QGroupBox:
        g = QGroupBox("边界/初值条件 (BC/IC)")
        lay = QVBoxLayout(g)
        self.bc_face = QComboBox()
        self.bc_face.addItems(self._BC_FACE_ITEMS)
        lay.addWidget(QLabel("选面（几何元素）"))
        lay.addWidget(self.bc_face)
        self.bc_face.currentTextChanged.connect(self._on_bc_face_preview)

        self.bc_kind = QComboBox()
        self.bc_kind.addItems(self._BC_KIND_ITEMS)
        lay.addWidget(QLabel("边界类型"))
        lay.addWidget(self.bc_kind)
        self.bc_kind.currentTextChanged.connect(self._on_bc_kind_changed)

        self.lbl_bc_val = QLabel("固定温度 T")
        self.bc_val = QDoubleSpinBox()
        self.bc_val.setRange(50.0, 3000.0); self.bc_val.setValue(373.0)
        self.bc_val.setSuffix(" K")
        lay.addWidget(self.lbl_bc_val); lay.addWidget(self.bc_val)

        self.lbl_bc_val2 = QLabel("环境 T∞")
        self.bc_val2 = QDoubleSpinBox()
        self.bc_val2.setRange(50.0, 1200.0); self.bc_val2.setValue(293.0)
        self.bc_val2.setSuffix(" K")
        lay.addWidget(self.lbl_bc_val2); lay.addWidget(self.bc_val2)

        btn_add = QPushButton("添加条件")
        btn_add.clicked.connect(self._on_bc_add)
        lay.addWidget(btn_add)
        btn_def = QPushButton("载入默认边界")
        btn_def.clicked.connect(self._on_bc_default)
        lay.addWidget(btn_def)
        btn_clr = QPushButton("清空条件")
        btn_clr.clicked.connect(self._on_bc_clear)
        lay.addWidget(btn_clr)

        self.bc_list = QListWidget()
        lay.addWidget(QLabel("已设条件"))
        lay.addWidget(self.bc_list)

        self.ic_enable = QCheckBox("预热初值 (IC)")
        self.ic_enable.setChecked(True)
        self.ic_temp = QDoubleSpinBox()
        self.ic_temp.setRange(50.0, 1200.0); self.ic_temp.setValue(373.0)
        self.ic_temp.setSuffix(" K")
        lay.addWidget(self.ic_enable)
        lay.addWidget(QLabel("初温 T0"))
        lay.addWidget(self.ic_temp)

        self._on_bc_kind_changed(self.bc_kind.currentText())
        return g

    def _on_bc_kind_changed(self, text: str) -> None:
        self.bc_val.show(); self.bc_val2.hide()
        k = self._bc_kind_code(text)
        if k == "dirichlet":
            self.lbl_bc_val.setText("固定温度 T")
            self.bc_val.setSuffix(" K"); self.bc_val.setRange(50.0, 3000.0)
            self.bc_val.setValue(373.0)
        elif k == "convection":
            self.lbl_bc_val.setText("对流系数 h")
            self.bc_val.setSuffix(" W/m²K"); self.bc_val.setRange(0.0, 200.0)
            self.bc_val.setValue(15.0)
            self.bc_val2.show()
        elif k == "flux":
            self.lbl_bc_val.setText("热流密度 q")
            self.bc_val.setSuffix(" W/m³")
            self.bc_val.setRange(-1.0e9, 1.0e9); self.bc_val.setSingleStep(1.0e7)
            self.bc_val.setValue(0.0)
        elif k == "adiabatic":
            self.bc_val.hide(); self.lbl_bc_val.hide()
        else:
            self.lbl_bc_val.setText("主值")

    def _on_bc_face_preview(self, text: str) -> None:
        self.bc_highlight = self._bc_face_code(text)
        self._redraw_highlight()

    def _redraw_highlight(self) -> None:
        if self.part is None or self.bc_highlight is None:
            self.slice_view.set_highlight(None)
            return
        m3 = np.asarray(face_mask(self.part, self.bc_highlight))
        idx = self.slider.value() if self.hatch is not None else 0
        m2 = m3[..., idx] if m3.ndim == 3 else m3
        self.slice_view.set_highlight(m2)

    def _on_bc_add(self) -> None:
        if self.part is None:
            QMessageBox.warning(self, "提示", "请先构建几何。")
            return
        k = self._bc_kind_code(self.bc_kind.currentText())
        face = self._bc_face_code(self.bc_face.currentText())
        val = float(self.bc_val.value())
        val2 = float(self.bc_val2.value()) if k == "convection" else 0.0
        bc = BoundaryCondition(kind=k, face=face, value=val, value2=val2)
        self.bc = self.bc.with_bc(bc)
        self._refresh_bc_list()

    def _on_bc_default(self) -> None:
        if self.part is None:
            QMessageBox.warning(self, "提示", "请先构建几何。")
            return
        self.bc = P.default_boundary_collection(
            self.part, preheat_temp=float(self.ic_temp.value()))
        self._refresh_bc_list()
        self._redraw_highlight()

    def _on_bc_clear(self) -> None:
        self.bc = BoundaryCollection()
        self._refresh_bc_list()

    def _refresh_bc_list(self) -> None:
        self.bc_list.clear()
        for bc in self.bc.bcs:
            self.bc_list.addItem(bc.name)
        if self.bc.ic is not None:
            self.bc_list.addItem(f"IC: {self.bc.ic.kind} T0={self.bc.ic.value:.0f}K")

    # -- 粉末/粒子组（缺口 #20-②B） --
    def _powder_group(self) -> QGroupBox:
        g = QGroupBox("粉末/粒子 (Powder)")
        lay = QVBoxLayout(g)
        self.btn_imp_particle = QPushButton("导入粒子 (CSV/JSON)…")
        self.btn_imp_particle.clicked.connect(self._on_import_particles)
        lay.addWidget(self.btn_imp_particle)
        btn_gen = QPushButton("生成粉末层")
        btn_gen.clicked.connect(self._on_gen_powder)
        lay.addWidget(btn_gen)

        # 生成参数
        lay.addWidget(QLabel("粒径 d50 (µm)"))
        self.pow_d50 = QDoubleSpinBox(); self.pow_d50.setRange(5.0, 200.0)
        self.pow_d50.setValue(30.0); self.pow_d50.setSuffix(" µm")
        lay.addWidget(self.pow_d50)
        lay.addWidget(QLabel("PSD σ (对数正态)"))
        self.pow_psd = QDoubleSpinBox(); self.pow_psd.setRange(0.0, 1.0)
        self.pow_psd.setValue(0.25)
        lay.addWidget(self.pow_psd)
        lay.addWidget(QLabel("铺粉层数"))
        self.pow_layers = QDoubleSpinBox(); self.pow_layers.setRange(1, 20)
        self.pow_layers.setValue(1); self.pow_layers.setDecimals(0)
        lay.addWidget(self.pow_layers)

        self.pow_use_precise = QCheckBox("用导入粒子作 DEM 初始态 (精确坐标注入)")
        self.pow_use_precise.setToolTip(
            "勾选后，导入/生成的精确粒子坐标将作为 DEM 沉降的初始态进入求解器"
            "（而非仅用其 PSD 统计生成 RCP 床）；对结构化/异质点云可保留布局信息。")
        lay.addWidget(self.pow_use_precise)

        self.pow_info = QLabel("未导入粉末")
        lay.addWidget(self.pow_info)
        btn_eval = QPushButton("评估粉末床 (DEM)")
        btn_eval.clicked.connect(self._on_eval_powder)
        lay.addWidget(btn_eval)
        return g

    # -- 支撑结构组（模块A） --
    def _support_group(self) -> QGroupBox:
        g = QGroupBox("支撑结构 (Support)")
        lay = QVBoxLayout(g)
        lay.addWidget(QLabel("类型"))
        self.sup_kind = QComboBox(); self.sup_kind.addItems(["block", "overhang"])
        lay.addWidget(self.sup_kind)
        lay.addWidget(QLabel("悬垂角阈值 ° (overhang)"))
        self.sup_angle = QDoubleSpinBox(); self.sup_angle.setRange(10.0, 80.0)
        self.sup_angle.setValue(45.0); self.sup_angle.setSuffix(" °")
        lay.addWidget(self.sup_angle)
        lay.addWidget(QLabel("填充密度 0~1"))
        self.sup_density = QDoubleSpinBox(); self.sup_density.setRange(0.1, 1.0)
        self.sup_density.setValue(1.0)
        lay.addWidget(self.sup_density)
        self.btn_sup = QPushButton("生成支撑")
        self.btn_sup.clicked.connect(self._on_gen_support)
        lay.addWidget(self.btn_sup)
        self.sup_info = QLabel("未生成")
        lay.addWidget(self.sup_info)
        return g

    def _on_gen_support(self) -> None:
        if self.part is None:
            QMessageBox.warning(self, "提示", "请先构建几何。")
            return
        try:
            res = P.generate_support(
                self.part, kind=self.sup_kind.currentText(),
                overhang_angle_deg=float(self.sup_angle.value()),
                density=float(self.sup_density.value()))
        except Exception as exc:  # 求解异常兜底
            QMessageBox.warning(self, "支撑生成失败", f"{exc}")
            return
        self.support = res["support"]
        sup = res["support"]
        self.sup_info.setText(
            f"类型={sup.kind}  体积占比={float(sup.volume_fraction):.3f}  "
            f"接触面积={float(sup.contact_area):.3e} m²")
        # 预览：沿 z 投影的支撑足迹叠加到切片视图
        self.slice_view.set_support(res["support_mask_2d"])
        self.info.append(f"已生成支撑：{sup.summary()}")

    def _on_import_particles(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择粒子点云", "", "粒子文件 (*.csv *.json)")
        if not path:
            return
        try:
            col = P.build_particle_collection_csv(path)
        except Exception as exc:  # 解析失败兜底
            QMessageBox.warning(self, "导入失败", f"{exc}")
            return
        self.powder = col
        s = col.stats()
        self.pow_info.setText(
            f"粒子={s['n_particles']}，d50={s['d50']*1e6:.1f}µm，"
            f"材料={s['material']}")
        self._redraw_particles()
        self.info.append(
            f"已导入粉末：{col.name}，{s['n_particles']} 粒子，"
            f"d50={s['d50']*1e6:.1f} µm，体积分数={col.packing_density():.3f}")

    def _on_gen_powder(self) -> None:
        if self.part is None:
            QMessageBox.warning(self, "提示", "请先构建几何。")
            return
        lt = float(self._spins["layer_thickness"].value()) * 1e-6
        col = P.generate_powder_bed(
            self.part, layer_thickness=lt,
            d50=float(self.pow_d50.value()) * 1e-6,
            psd_sigma=float(self.pow_psd.value()),
            n_layers=int(self.pow_layers.value()),
            material="316L")
        self.powder = col
        s = col.stats()
        self.pow_info.setText(
            f"粒子={s['n_particles']}，d50={s['d50']*1e6:.1f}µm，"
            f"材料={s['material']}")
        self._redraw_particles()
        self.info.append(
            f"已生成粉末层：{s['n_particles']} 粒子，"
            f"d50={s['d50']*1e6:.1f} µm，体积分数={col.packing_density():.3f}")

    def _on_eval_powder(self) -> None:
        if self.powder is None:
            QMessageBox.warning(self, "提示", "请先导入或生成粉末。")
            return
        if self.part is None or self.plan is None:
            QMessageBox.warning(self, "提示", "请先构建几何并评估工艺。")
            return
        try:
            res = P.solve_powder_from_collection(
                self.powder, self.part, self.plan, n_steps=120,
                use_precise=self.pow_use_precise.isChecked())
        except Exception as exc:  # 真实 DEM 求解异常兜底
            QMessageBox.warning(self, "粉末床求解失败", f"{exc}")
            return
        self.info.append("=== 粉末床评估 (真实 DEM) ===")
        self.info.append(
            f"铺粉密度={float(res.packing_density):.3f}，"
            f"配位数={float(res.coordination_number):.2f}，"
            f"粗糙度={float(res.surface_roughness)*1e6:.2f} µm")
        self.info.append(
            f"球化风险={float(res.balling_indicator):.3f}，"
            f"未熔合={float(res.lof_indicator):.3f}，"
            f"飞溅={float(res.spatter_fraction):.3f}")

    def _redraw_particles(self) -> None:
        if self.powder is None or self.hatch is None:
            self.slice_view.set_particles(None)
            return
        idx = self.slider.value() if self.hatch is not None else 0
        z = float(np.asarray(self.hatch["z_heights"])[idx]) if self.powder.dim == 3 \
            else 0.0
        xs, ys = self.powder.slice_xy(z, tol=float(self.part.spacing) * 1.5)
        self.slice_view.set_particles((xs, ys))

    # -- 回调 --
    def _on_build(self):
        kind = self.prim.currentText()
        part = P.build_primitive(
            kind, float(self.size_mm.value()),
            spacing_um=float(self.spacing_um.value()),
        )
        self._set_part(part)

    def _on_import_stl(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 STL", "", "STL (*.stl *.STL)")
        if not path:
            return
        part = P.build_from_stl(path, spacing_um=float(self.spacing_um.value()))
        self._set_part(part)

    def _set_part(self, part):
        self.part = part
        self.bc = P.default_boundary_collection(
            part, preheat_temp=float(self.ic_temp.value()))
        self.view3d.show_part(part)
        lt = float(self._spins["layer_thickness"].value()) * 1e-6
        hs = float(self._spins["hatch_spacing"].value()) * 1e-6
        self.hatch = P.generate_hatch_paths(part, lt, hs)
        self.slice_view.set_hatch(self.hatch)
        n = self.hatch["n_layers"]
        self.slider.setMaximum(max(0, n - 1))
        self.slider.setValue(n // 2)
        self._on_layer(self.slider.value())
        self._refresh_bc_list()
        self._redraw_particles()
        self.info.append(
            f"几何已构建：shape={tuple(part.shape)}，层数={n}，"
            f"体素尺寸={float(part.spacing)*1e6:.1f} µm"
        )

    def _on_layer(self, idx):
        self.layer_label.setText(f"layer: {idx}")
        self.slice_view.show_layer(idx)
        self._redraw_highlight()
        self._redraw_particles()

    def _on_assess(self):
        if self.part is None:
            QMessageBox.warning(self, "提示", "请先构建几何。")
            return
        vals = {k: float(sp.value()) for k, sp in self._spins.items()}
        n_layers = self.part.layer_count(vals["layer_thickness"] * 1e-6)
        self.plan = P.build_process_plan(
            n_layers,
            laser_power=vals["laser_power"], scan_speed=vals["scan_speed"],
            layer_thickness=vals["layer_thickness"] * 1e-6,
            hatch_spacing=vals["hatch_spacing"] * 1e-6,
            beam_radius=vals["beam_radius"] * 1e-6,
            absorption=vals["absorption"], preheat_temp=vals["preheat_temp"],
            rotation_per_layer_deg=vals["rotation_per_layer_deg"],
        )
        rep = P.assess(self.part, self.plan)
        self.info.append("=== 工艺评估 ===")
        self.info.append(f"VED = {rep['VED_J_m3']:.3e} J/m³  ({rep['VED_status']})")
        self.info.append(f"LED = {rep['LED_J_m']:.3e} J/m")
        pr = rep["printability"]
        self.info.append(
            f"体积={pr['volume_mm3']:.2f} mm³，层数={pr['n_layers']}，"
            f"悬垂占比={pr['overhang_fraction']:.3f}，"
            f"薄壁占比={pr['thin_wall_fraction']:.3f}"
        )

    def _on_save(self):
        if self.part is None or self.plan is None:
            QMessageBox.warning(self, "提示", "请先构建几何并评估工艺。")
            return
        d = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if not d:
            return
        gpath, ppath = P.export_config(self.part, self.plan, d)
        # BC/IC：IC 由勾选框决定（预热初值）
        ic = (InitialCondition("preheat", float(self.ic_temp.value()))
              if self.ic_enable.isChecked() else None)
        bc_final = BoundaryCollection(bcs=self.bc.bcs, ic=ic)
        bcpath = P.export_bc(bc_final, d)
        lines = f"已导出：\n  {gpath}\n  {ppath}\n  {bcpath}"
        if self.powder is not None:
            powpath = self.powder.to_csv(os.path.join(str(d), "preproc_powder.csv"))
            lines += f"\n  {powpath}"
        self.info.append(lines)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    win = PreprocMainWindow()
    win.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
