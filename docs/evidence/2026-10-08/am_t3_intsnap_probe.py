# -*- coding: utf-8 -*-
"""T3（#24）普查＋「长度/间距 → 整数个数」的 1-ulp 稳定性探针：**改前跑一次、改后跑一次**。

背景：T1（D4，§25.7）只给 `geometry._grid_axes` 做了 `1e-9` 相对容差吸附。同一模式
（两个物理长度相除后取整，用来**决定域/路径/栅格有多大**）在全树还有多少处？登记时写的是
"8 处"，本轮普查（`K0`，逐条列出＋**未登记即 VOID**）实得 **in-scope 位点 11 个（15 条源码行）**
＋ **3 处属另一种舍入**（`round`，口径分裂 ⇒ 另立 #31，本轮**不统一**）＋ 一批明确出域。
这是"命中清单 ≠ 普查"的第 4 次现身（前三次见 §26.6-9／§26.17／§26.21）。
【跑后更正 CN-1】上面这个"15"是**实测**：`K0b` 在两份存档日志里都数到 **15 行**（S06 3 行、S07 3 行、
其余 9 位点各 1 行），首稿凭心算写成 14 ⇒ 属"计数要从被计量对象上读"那一族。同模式**总数 12 处**
＝这 11 处＋T1 已吸附的 `geometry._grid_axes`（`K0` 标为 `snapped`，不进 `K0b` 的 15）。
本段只是注释文本 ⇒ 记分未动：改后重跑与旧存档日志**除两行簿记外逐行相同**（`diff` 实测：差异＝
`dirty_src` 9→10 因本轮新增 `tests/test_intsnap_counts.py`、`counts.py` 两行 helper 的**行号** 41/50→49/58
因本轮补其 docstring；排除这两行后 `SCORES-IDENTICAL`）⇒ `_post.log` 已按**最终树**重捕，令树戳与所交付
状态一致（§26.20 的树戳纪律：读数必须属于它声明的那棵树）。K5 的 A0 整数块在新旧日志间仍 **diff 空**。
本段定稿后**再跑一次**：与已安装的 `_post.log` **逐字节相同**（exit=0、`diff` 输出空）⇒ 交付的探针与交付的
日志互为证据（此后任何一侧的改动都必须以重跑＋重捕的形式一并发生）。

本轮范围（刻意的窄）：**只修"同一物理量换二进制拼写 ⇒ 整数抖动"，不改任何口径语义**。
`Mcen` 整格口径与 `fv=1−1ulp` 皮层**不在本轮**——它们要动 `solid_fraction`/`solid_weight`，
会移动 **A0 的可观测量（熔体积）**，而熔体积口径正是 **#29 待用户裁决**的对象 ⇒ 登记不实现。

跑法（毫秒级、纯几何/路径/点云构造，**不解热方程**，CPU 钉住）：
    CUDA_VISIBLE_DEVICES="" JAX_ENABLE_X64=1 PYTHONPATH=src /tmp/amvenv/bin/python \\
        docs/evidence/2026-10-08/am_t3_intsnap_probe.py
阶段由 `T3_PHASE=pre|post` 指定；`auto` 时按 `src/amforge/counts.py` 是否存在判定。

跑前登记的判据（**K0 不过 ⇒ 全轮作废**；K1 是 K2 的正对照）：
  K0 **普查完备**：遍历 `src/**/*.py`，抓每一处 `ceil(`/`floor(`/`round(`/`rint(`（含裸 `round(`），
      逐条必须落在登记表里（`in-scope` / `snapped` / `time-count` / `round-family` / `index-map` /
      `int-exact`）。出现未登记位点 ⇒ **VOID**。登记表按"原文 `pre` ＋ 标记 `mark`"双路匹配，
      所以改后行文本变了也认得出（防"修完就查不到"的假完备）。
      K0b：改后每条 in-scope 取整行必须含 `1e-9`/`count_ceil`/`count_floor`，缺一条 ⇒ FAIL。
  K1 **翻转确实存在（正对照，能失败）**：每个 in-scope 位点至少一例满足
      ①同一物理量换 4 种二进制拼写后整数改变，或 ②整数不等于 `Fraction` 精确 oracle。
      改前若某位点 0 翻转，必须命中**跑前声明**的允许清单（附机制解释），否则 FAIL。
      ⇒ 作用是让 K2 不是空门：**没有可失败的门槛不算门槛**。
      【跑前修正 K1-2（S05 的**改前**整数由两套不同口径的 fudge 决定，正对照要按实测算术读）】
      夹具 5×5 格 @50µm ⇒ 单边跨度 200µm、`diag_um=√2·200=282.842712474619`。用例把
      `hatch = diag_um/q × 1e-6`，于是生产侧 `diag = ‖hi−lo‖`（≈2.8284e-4 m）里 `1e-12` 的
      **绝对** fudge 换算到比值上是 `1e-12/2.8284e-4 = 3.5355e-9`（**相对**）＝本轮带宽 1e-9 的
      3.5 倍 ⇒ **仅靠取整吸附不可能把整数拉回 q**（吸附吃的是 ±1ulp≈1e-16 相对抖动）。
      ⇒ S05 的缺陷形态不是拼写敏感而是**系统性多算一道**（改前实测 `q=4→5、q=8→9、q=20→21`
      且**四拼写全同**⇒ 0 翻转，见 `_pre.log` 的 S05 段），它属于**独立机制** ⇒ 另立 **#32**，
      本轮不与之叠加（N4）；抓它的是 K2b 预测值列，不是 K1 的扰动判据
      （K1 对它无效这件事已预注册进 `NO_FLIP_ALLOWED`）。
      ⇒ 同段 `mul6` 诊断列给出对照：`q=4` 例把 hatch 用 6 位有效数字读成 `70.7107µm`
      （物理量本身变了 **＋3.094e-7 相对**＝带宽的 309 倍）⇒ 比值 3.9999987622、整数落回 4，
      而 ±1ulp 拼写**给不出** 4 ⇒ "改物理量"与"改拼写"是两件事，前者不进判据。
      【跑前修正 K1-1（门槛不动，动的是**用例构造**；两次都是 K1 自己抓出来的探针缺陷）】
        · S02：除数是 `part.spacing` 而非用例的 `dx_um` ⇒ 首跑 oracle 与实测差 2 倍
          （`刀锋 dx=50 q=6 → o=6 lit=12`）。改：夹具 spacing 由用例 `pdx` 传入。
        · S08：首跑**唯一 0 翻转位点**。查下来不是免疫而是刀锋没搭上——`step = span_um/q` 再
          `×1e-6` 得 `9.999999999999999e-05`（比 1e-4 小 1 ulp）⇒ 生产侧比值 4.0000000000000004，
          距整数**一整个 ulp**，±1ulp 够不着 ⇒ 假稳定。改：hatch 用夹具**实测浮点跨度** `/q` 构造
          （S05 的 `diag_um` 同款）。S09 一并改，保持两族可比。
  K2 **改后稳定**（本轮主验收）：每个 in-scope 位点全部用例，把该量取
      {字面量, −1ulp, +1ulp, `µm 数值×1e-6`} 四种拼写 ⇒ 整数必须**全同**；凡 oracle 存在 ⇒
      整数必须等于 oracle。内建正对照＝`_grid_axes`（T1 已修）必须在同一判据下 PASS
      ⇒ 证明门槛**可被正确实现达到**（不是靠放宽过关）。
  K3 **吸附只吃 ulp 级抖动**：真实余量 `q+0.4` 格必须**两阶段都得 q+1**（不得欠覆盖）；
      比值高出整数 **1e-8（相对）** 时**不得**被吃掉，高出 **1e-10（相对）** 时**必须**被吃掉。
      （1e-8 与 1e-10 分居容差 1e-9 两侧 ⇒ 本条真在量容差宽度，写宽/写窄都会被抓。）
  K4 **覆盖不退化**（ceil 族）：`min(n)·step ≥ L − 1e-9·L`，除非该例本就落在容差带内
      （带内＝比值到整数的距离 ≤ 1e-9·比值）。
  K5 **A0 不动**：打印 A0 试片（1.2×0.6×0.4 mm，dx=50/25/12.5 µm）的 `shape/nvox/n_layers/
      n_lines/path_length`；改前改后两份日志这一段必须**逐行相同**（外层 diff 判定）。
      ⇒ 本轮改动**不得**成为 A0 红灯换数的理由（#16 的归因才干净）。
  K6 **口径分裂实测**（只打印、不设门槛、不修）：同一个"多少层/多少道"在 `ceil` / `round` /
      `floor+1` 三套舍入下的整数逐档列出 ⇒ 差值登记给 **#31**（与 #22 同族：同一物理量的口径
      在两求解器间不一致）。
  绑定要求：每个位点的调用/复刻都必须与登记原文仍一致（`文本绑定=True` 列），否则该位点的
      记分不作数（§26.17 的教训：探针与生产悄悄漂移）。
  退出码：pre＝K0 ∧ K1；post＝K0 ∧ K0b ∧ K2 ∧ K2b ∧ K4（任一门失败 ⇒ 退出码 1，
      见记分段【跑后修正 GATE-1】；K5/K3/K6 是读数段，由外层 diff 与 #31 台账判）。
纪律：A0 断言不许放宽；不许换指标；全量回归不得新增 skip/xfail；性能数字只能在 A6000 上取
（本探针是**整数几何**，与吞吐无关 ⇒ CPU）。
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                           # noqa: E402

from amforge import geometry as G                                 # noqa: E402
from amforge import process as PR                                 # noqa: E402
from amforge.core.contracts import PartGeometry, ProcessPlan       # noqa: E402
from amforge.geometry import _grid_axes, crop_to_part, layer_z_heights, with_baseplate
from amforge.gui.preproc import generate_hatch_paths              # noqa: E402
from amforge.powder import ParticleCollection                     # noqa: E402
from diffmech.methods.am.scan_paths import zigzag_hatch           # noqa: E402

UM = 1e-6
TOL_REL = 1e-9              # 与 _grid_axes（T1 已修）同款宽度；K3 就是来量这个宽度的
REPO = Path(__file__).resolve().parents[3]   # docs/evidence/<日期>/本文件 → 仓库根
SRC = REPO / "src"
CASES: dict[str, list[dict]] = {}


def hdr(s: str) -> None:
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78, flush=True)


def L(file: str, pre: str, mark: str | None = None, post: str | None = None) -> dict:
    """post＝改后该行**必须**含的标记（amforge 侧＝助手名，diffmech 侧＝1e-9 容差）。"""
    return dict(file=file, pre=pre, mark=mark or pre[:22], post=post)


# ===========================================================================
# 登记表（跑前写死）：sid -> 类别 + 该位点的源码行（pre＝登记原文，mark＝改后仍可认的标记）
# ===========================================================================
SITES: dict[str, dict] = {
    # ---------------- in-scope：本轮要加容差吸附（11 位点 / 14 行）----------------
    "S01_layer_count": dict(cls="in-scope", fam="ceil", called=True,
        note="层高→层数（GUI/优化/反演的 n_layers 之源）",
        lines=[L("src/amforge/core/contracts.py",
                "return max(1, int(np.ceil(float(height) / float(layer_thickness))))",
                "float(height)")]),
    "S02_with_baseplate": dict(cls="in-scope", fam="ceil", called=True,
        note="基板厚→向下扩的格数（主要散热通道厚度）",
        lines=[L("src/amforge/geometry.py",
                 "n_add = max(1, int(np.ceil(thickness / part.spacing)))", "n_add =")]),
    "S03_layer_z_heights": dict(cls="in-scope", fam="ceil", called=True,
        note="层高→切片层数（slice_masks/GUI 桥梁）",
        lines=[L("src/amforge/geometry.py", "n = max(1, int(np.ceil((hi - lo) / layer_thickness)))",
                 "(hi - lo) / layer_thickness")]),
    "S04_crop_margin": dict(cls="in-scope", fam="ceil", called=True,
        note="margin→裁剪 pad 格数（决定求解域大小）",
        lines=[L("src/amforge/geometry.py", "pad = int(np.ceil(margin / part.spacing))", "pad =")]),
    "S05_zigzag_n_line": dict(cls="in-scope", fam="ceil", called=True,
        note="对角线长→道数（diag 另带**绝对** +1e-12 fudge＝3.5e-9 相对 ⇒ 带外 ⇒ 偏高一道，"
             "属 #32，见 K1 段 K1-2 与 K2b）",
        lines=[L("src/amforge/process.py",
                 "n_line = max(1, int(np.ceil(diag / max(h, 1e-9))))", "n_line =")]),
    "S06_powder_lattice": dict(cls="in-scope", fam="floor+1", called=True,
        note="from_part_bed：床尺寸→晶格 nx/ny/nz→粒子数",
        lines=[L("src/amforge/powder.py", "nx = max(1, int(math.floor((x1 - x0) / spacing)) + 1)",
                 "nx = max(1,"),
               L("src/amforge/powder.py", "ny = max(1, int(math.floor((y1 - y0) / spacing)) + 1)",
                 "ny = max(1,"),
               L("src/amforge/powder.py",
                 "nz = max(1, int(math.floor((z1 - z0) / spacing)) + 1) if part.dim == 3 else 1",
                 "nz = max(1,")]),
    "S07_powder_bed_params": dict(cls="in-scope", fam="floor+1", called=True,
        note="powder_bed_params：DEM 网格 nx/ny（除数是 2·stats d50 ⇒ 无精确 oracle，只判稳定）",
        lines=[L("src/amforge/powder.py", "nx = max(4, int(math.floor((hi[0] - lo[0]) / sp)) + 1)",
                 "nx = max(4,"),
               L("src/amforge/powder.py", "ny = max(4, int(math.floor((hi[1] - lo[1]) / sp)) + 1)",
                 "ny = max(4,"),
               L("src/amforge/powder.py", "nz = max(2, int(math.floor((hi[1] - lo[1]) / sp)) + 1)",
                 "nz = max(2,")]),
    "S08_gui_hatch_lines": dict(cls="in-scope", fam="floor+1", called=True,
        note="GUI 预览 hatch：y 跨度→扫描线数",
        lines=[L("src/amforge/gui/preproc.py",
                 "n_lines = max(1, int(np.floor((y_max - y_min) / hatch_spacing)) + 1)",
                 "n_lines = max(1,")]),
    "S09_am_hatch_lines": dict(cls="in-scope", fam="ceil", called=True,
        note="diffmech zigzag_hatch：y 跨度→扫描线数",
        lines=[L("src/diffmech/methods/am/scan_paths.py",
                 "n_lines = max(1, int(np.ceil((y_max - y_min) / hatch_spacing)))",
                 "n_lines = max(1,")]),
    "S10_powderbed_ngrid": dict(cls="in-scope", fam="ceil", called=False,
        note="DEM 床栅格 n_grid=ceil(lx/dx)+4（复刻：调用会把整张 DEM 初态建出来，非本轮目的）",
        lines=[L("src/diffmech/methods/am/powder_bed.py",
                 "n_grid = max(8, int(np.ceil(lx / dx)) + 4)", "n_grid = max(8,")]),
    "S11_server_mpagrid": dict(cls="in-scope", fam="ceil", called=False,
        note="MPM 自动栅格 shape=ceil(span/dx)+2（复刻：位点在 FastAPI 构造内部）",
        lines=[L("src/diffmech/server.py",
                 "shape = tuple(max(3, int(np.ceil(spans[i] / dx)) + 2) for i in range(dim))",
                 "shape = tuple(max(3,")]),
    # ---------------- 出域（登记类别即计入完备性）----------------
    "S00_counts_helper": dict(cls="helper", fam="ceil+floor", called=False,
        note="T3(#24) 新增的单点吸附助手本体＝容差宽度唯一定义处 ⇒ 它的取整行也必须在册",
        lines=[L("src/amforge/counts.py",
                 "return int(math.ceil(r - TOL_REL * max(1.0, abs(r))))", "math.ceil"),
               L("src/amforge/counts.py",
                 "return int(math.floor(r + TOL_REL * max(1.0, abs(r))))", "math.floor")]),
    "grid_axes": dict(cls="snapped", fam="ceil", called=False,
        note="T1（D4）已修 ⇒ 本轮作 K2 的内建正对照",
        lines=[L("src/amforge/geometry.py",
                 "n_cells = int(np.ceil(ratio - 1e-9 * max(1.0, abs(ratio))))", "n_cells =")]),
    "time_counts": dict(cls="time-count", fam="ceil", called=False,
        note="时间步数 6 处，已有绝对 −1e-9（量纲是 s，非长度/栅格 ⇒ 不并入本轮助手）",
        lines=[L("src/amforge/thermal_enthalpy.py",
                 "return max(1, int(math.ceil(t_exp / dt_max - 1e-9)))", "n_steps"),
               L("src/amforge/thermal_enthalpy.py",
                 "n_need = max(1, int(math.ceil(path_max / v_lo_auto / dt_target - 1e-9)))", "n_need ="),
               L("src/amforge/thermal_enthalpy.py",
                 "n_stable = max(1, int(math.ceil(path_max / v_lo_auto", "n_stable ="),
               L("src/amforge/thermal_enthalpy.py",
                 "else max(1, int(math.ceil(t_bound / dt_target - 1e-9))))", "dt_target - 1e-9"),
               L("src/amforge/thermal_enthalpy.py",
                 "n_steps_cfl = max(1, int(math.ceil(t_bound / dt_c - 1e-9))) if _ok else None",
                 "n_steps_cfl ="),
               L("src/amforge/thermal_enthalpy.py",
                 "n_steps_stable = max(1, int(math.ceil(t_bound / dt_r - 1e-9))) if _ok else None",
                 "n_steps_stable =")]),
    "round_family": dict(cls="round-family", fam="round", called=False,
        note="同一『多少层/多少道』的第二/三套舍入 ⇒ 口径分裂 #31（本轮不统一）",
        lines=[L("src/amforge/thermal_enthalpy.py",
                 "n_layers = jnp.maximum(jnp.round(z_ext / jnp.maximum(lt, 1e-12)), 1.0)", "n_layers ="),
               L("src/amforge/thermal_enthalpy.py",
                 "n_lines = jnp.maximum(jnp.round(y_ext / hatch), 1.0)", "n_lines = jnp.maximum"),
               L("src/amforge/meltpool.py", "n_cells_layer = max(1, int(round(lt / cfg.dx)))",
                 "n_cells_layer =")]),
    "index_map": dict(cls="index-map", fam="floor", called=False,
        note="坐标→索引的逐点归属，不给域定尺寸 ⇒ 1 ulp 只挪一个点，不翻一整层",
        lines=[L("src/amforge/thermal_enthalpy.py",
                 "lf = jnp.clip(jnp.floor(gl), 0.0, n_layers - 1.0)", "lf ="),
               L("src/amforge/thermal_enthalpy.py",
                 "jf = jnp.clip(jnp.floor(gj), 0.0, n_lines - 1.0)", "jf ="),
               L("src/amforge/meltpool.py", "k = jnp.floor(t / t_cycle)", "k = jnp.floor"),
               L("src/amforge/asbuilt_plastic.py", "jnp.round((cell_centers - origin) / spacing),",
                 "cell_centers"),
               L("src/amforge/phasefield.py",
                 "ix = jnp.clip(jnp.round(cx + dxr[:, None] * steps[None, :]).astype(int), 0, nx - 1)",
                 "ix = jnp.clip"),
               L("src/amforge/phasefield.py",
                 "iy = jnp.clip(jnp.round(cy + dyr[:, None] * steps[None, :]).astype(int), 0, ny - 1)",
                 "iy = jnp.clip"),
               L("src/diffmech/methods/mpm/mpm.py", "base_f = jnp.floor(x - 0.5)", "base_f ="),
               L("src/diffmech/methods/am/powder_bed.py",
                 "cx = jnp.clip(jnp.floor(pos[:, 0] / dx), 0, max(nx - 1, 0)).astype(jnp.int32)",
                 "cx = jnp.clip"),
               L("src/diffmech/methods/am/powder_bed.py",
                 "cy = jnp.clip(jnp.floor(pos[:, 1] / dy), 0, max(ny - 1, 0)).astype(jnp.int32)",
                 "cy = jnp.clip")]),
    "doc_prose": dict(cls="doc-prose", fam="none", called=False,
        note="docstring/注释里提到的旧写法，不是活代码 ⇒ 不参与吸附",
        lines=[L("src/amforge/thermal_enthalpy.py",
                 "64 点缓冲（旧写法 ``xs_full=linspace(...,64)`` 配 ``n_per_line=round(x_ext/",
                 "n_per_line=round")]),
    "int_exact": dict(cls="int-exact", fam="mixed", called=False,
        note="输入已是精确整数/半整数或纯排版量 ⇒ 比值不会落在 1 ulp 内",
        lines=[L("src/amforge/phasefield.py", "max_step = int(jnp.ceil(jnp.sqrt(cx ** 2 + cy ** 2)))",
                 "max_step ="),
               L("src/diffmech/postprocess/animation.py", "nrows = int(np.ceil(n / ncols))",
                 "nrows =")]),
}
IN_SCOPE = sorted(k for k, v in SITES.items() if v["cls"] == "in-scope")
ALL_LINES = [(e["file"], e["pre"], e["mark"], sid)
             for sid, v in SITES.items() for e in v["lines"]]

# K1 允许清单：**跑前**写下（改前若这些位点 0 翻转，机制已预注册，不算门槛落空）
NO_FLIP_ALLOWED = {
    "S05_zigzag_n_line":
        "diag 带**绝对** `+1e-12` fudge，实测算术 `1e-12/2.8284e-4 = 3.5355e-9`（相对）＝带宽 1e-9 的 "
        "3.5 倍 ⇒ 比值**恒在整数上方且距整数 ≫ 1 ulp** ⇒ ±1ulp 够不着 ⇒ 0 翻转。该位点的缺陷形态是"
        "**系统性多算一道**（由 K2b 的预测值列抓，不是拼写敏感 ⇒ 正是 K1 单靠扰动会漏的那一类）",
    "S11_server_mpagrid":
        "复刻算式里 span 与 dx 由同一 `max(spans)/(n_grid−2)` 反推 ⇒ 比值贴近整数但扰动 dx 时分/子"
        "同向，预期 0 翻转（该位点的真实风险由 S10 同族代表）",
    "S07_powder_bed_params":
        "除数＝2·`stats()['d50']`（**样本中位数**，非构造整数）⇒ 比值落在整数上的先验概率低；"
        "本轮用 `pdx = q·sp/8` **造**刀锋，若造出的比值不接近整数则该例无意义（打印 rel 自查）",
}

PAT = re.compile(r"\b(?:np|jnp|math)\.(?:ceil|floor|round|rint)\s*\(|(?<![.\w])round\s*\("
                 r"|(?<![.\w])count_(?:ceil|floor)\s*\(")

# K2b 改后预测值（跑前写下，能失败）。
# 【原跑前预测（已作废，留档）】"S05 的 `diag+1e-12` 以前把整除跨度顶成 q+1（改前实测
#   q=8→9、q=20→21）；取整交给 `count_ceil` 之后必须落回设计整数 q；仍是 q+1 ⇒ 助手没接上。"
# 【跑后修正 K2b-1：作废理由＝实测算术，不是判据口径】`+1e-12` 落在 diag=2.8284e-4 m 上是
#   **3.5355e-9 相对**，而吸附带只有 1e-9 ⇒ 带外 ⇒ 改后**必然**仍是 q+1。原预测把"绝对 fudge
#   顶翻取整"（#32）误当成"拼写抖动"（#24）来吸，两条机制不同口径；把它们叠在一起、或把带宽
#   从 1e-9 写到 4e-9 去满足原预测，都属凑绿（N4）⇒ 现把期望值改成实测的 `q+1` 并**照实保留**
#   上面的原预测与本段。可失败方向：若某轮把带宽调宽（或把 fudge 悄悄挪走）⇒ 实得 `q` ⇒ 本条 FAIL。
EXPECTED_POST = {("S05_zigzag_n_line", f"整数跨度 q={q}"): q + 1 for q in (4, 8, 20)}


# ===========================================================================
# oracle 与拼写
# ===========================================================================
def F(v: float) -> Fraction:
    return Fraction(str(v))


def exact_ceil(L_um: float, s_um: float) -> int:
    r = F(L_um) / F(s_um)
    return -((-r.numerator) // r.denominator)


def exact_floor1(L_um: float, s_um: float) -> int:
    r = F(L_um) / F(s_um)
    return r.numerator // r.denominator + 1


def exact_round(L_um: float, s_um: float) -> int:
    return round(F(L_um) / F(s_um))


def spellings(x: float) -> dict[str, float]:
    """同一物理量的四种**保值**二进制拼写：字面量、−1ulp、+1ulp、`µm×1e-6` 路径（T1 成因）。

    【跑后修正 SP-1】`mul` 原写作 `float(f"{um:g}") * UM`，而 `:g` 只给 **6 位有效数字**：
    `199.999998µm` 会被截成 `200`（相对改动 1e-8 量级＝**换了物理量**，比本轮要吸的 1 ulp≈1e-16
    大十个数量级）。后果两头都错——改前在 S05/S07 造**假翻转**（K1 多计），改后在 S01/S03 的
    `带外上1e-8` 档造**假不稳定＋假"少覆盖"**（K2/K4 误报）。现 `mul` 用 `:.17g`＝精确往返，
    只引入"先除以 1e-6 再乘回来"的双舍入＝T1 的真实机制；干净十进制（12.5/25/50 µm）与旧写法同值。
    6 位截断仍作**诊断**打印（`mul6=`），不进任何判据。
    """
    lit = float(x)
    um = lit / UM
    return {"lit": lit, "m1": float(np.nextafter(lit, -np.inf)),
            "p1": float(np.nextafter(lit, np.inf)), "mul": float(f"{um:.17g}") * UM}


def trunc6(x: float) -> float:
    """6 位有效数字的 µm 读数再乘回 1e-6 —— **不是**同一物理量，只作诊断列。"""
    return float(f"{float(x) / UM:g}") * UM


def step_above(L_um: float, q: int, rel: float) -> float:
    """返回 step，使 L/step ≈ q·(1+rel)：比值**高出**整数 rel·q（K3 的两个近整数例）。"""
    return float(F(L_um) / (Fraction(q) * (Fraction(1) + F(rel))))


# ===========================================================================
# 站点调用器（每个用例的 kw 里 `div` 键＝被换拼写的那个量）
# ===========================================================================
def mk_part(dx: float, nz: int, nx: int = 5, ny: int = 5) -> PartGeometry:
    return PartGeometry(sdf=jnp.full((nx, ny, nz), -1.0, dtype=jnp.float64),
                        origin=jnp.zeros(3, dtype=jnp.float64), spacing=float(dx), dim=3)


def mk_block_part(dx: float, grid: int = 41, block: int = 21) -> PartGeometry:
    a = (grid - block) // 2
    sdf = np.full((grid, grid, grid), 1.0)
    sdf[a:a + block, a:a + block, a:a + block] = -1.0
    return PartGeometry(sdf=jnp.asarray(sdf, dtype=jnp.float64),
                        origin=jnp.zeros(3, dtype=jnp.float64), spacing=float(dx), dim=3)


def f_layer_count(dx, lt, nz):
    return int(mk_part(dx, nz).layer_count(lt))


def f_layer_z_heights(dx, lt, nz):
    return int(len(layer_z_heights(mk_part(dx, nz), lt)))


def f_baseplate(thickness, pdx=25e-6):
    p = mk_part(pdx, 4)                              # 除数＝part.spacing：用例必须把它铺成 s_um
    return int(with_baseplate(p, thickness=thickness).shape[-1]) - p.shape[-1]


def f_crop_pad(dx, margin, block=21):
    p = mk_block_part(dx)
    return (int(crop_to_part(p, margin=margin).shape[0]) - block) // 2


def f_zigzag_n_line(hatch, n_layers=1):
    part = mk_part(50e-6, 5, nx=5, ny=5)
    plan = ProcessPlan.uniform(n_layers, modality="SLM", laser_power=200.0, scan_speed=1.0,
                               layer_thickness=40e-6, hatch_spacing=hatch, beam_radius=50e-6)
    seg = PR.zigzag_segments(part, plan)
    return int(jnp.asarray(seg.start).shape[0]) // n_layers


def f_powder_count(d50, lt, n_layers=1, pdx=50e-6, pnz=9):
    part = mk_part(pdx, pnz, nx=5, ny=5)
    col = ParticleCollection.from_part_bed(part, layer_thickness=lt, d50=d50,
                                           n_layers=n_layers, seed=0)
    return int(col.n_particles)


_COL = None                                    # 固定 seed 的同一批粒子（stats d50 稳定）


def f_powder_params(pdx, d50=25e-6, pnx: int = 9, pnz: int = 5):
    global _COL
    if _COL is None:
        _COL = ParticleCollection.from_part_bed(mk_part(50e-6, 5, nx=5, ny=5),
                                                layer_thickness=40e-6, d50=25e-6,
                                                n_layers=1, seed=0)
    part = mk_part(pdx, pnz, nx=pnx, ny=pnx)
    p = _COL.powder_bed_params(n_steps=10, geometry=part)
    return int(p["nx"])


def f_gui_hatch_lines(hatch, lt=40e-6):
    part = mk_part(50e-6, 5, nx=9, ny=9)
    d = generate_hatch_paths(part, lt, hatch)
    return len({round(float(seg[0][1]), 12) for seg in d["paths"][0]})


def f_scan_hatch_lines(hatch, n=9):
    ax = np.arange(n) * 50e-6
    path = zigzag_hatch(np.ones((n, n)), ax, ax, hatch_spacing=hatch)
    return len({round(float(y), 12) for y in np.asarray(path.waypoints)[:, 1]})


# S10/S11 是**复刻**位点（真调用会把整张 DEM 初态/FastAPI 构造建出来，非本轮目的）⇒ 这两个
# lambda 必须与生产行**同式**：pre 阶段＝裸 ceil（已实测翻转），post 阶段＝改后内联式；生产行
# 到底改没改由 K0b 的"文本绑定＋标记行含 1e-9"守（复刻本身不构成对 src 的证明）。
# 宽度按阶段注入 ⇒ 同一份探针文件能分别复现 pre／post 两份日志。
TOL_REPL = [0.0]


def _snap_ceil(a: float, b: float) -> int:
    r = float(a) / float(b)
    return int(np.ceil(r - TOL_REPL[0] * max(1.0, abs(r))))


REPL = {
    "S10_powderbed_ngrid": (lambda lx, dx: max(8, _snap_ceil(lx, dx) + 4), ("lx", "dx")),
    "S11_server_mpagrid": (lambda span, dx: max(3, _snap_ceil(span, dx) + 2), ("span", "dx")),
}
FUNCS = {
    "S01_layer_count": (f_layer_count, "lt"),
    "S02_with_baseplate": (f_baseplate, "thickness"),
    "S03_layer_z_heights": (f_layer_z_heights, "lt"),
    "S04_crop_margin": (f_crop_pad, "margin"),
    "S05_zigzag_n_line": (f_zigzag_n_line, "hatch"),
    "S06_powder_lattice": (f_powder_count, "lt"),
    "S07_powder_bed_params": (f_powder_params, "pdx"),
    "S08_gui_hatch_lines": (f_gui_hatch_lines, "hatch"),
    "S09_am_hatch_lines": (f_scan_hatch_lines, "hatch"),
}


def case(tag, kw, div, oracle, L_um, s_um):
    return dict(tag=tag, kw=kw, div=div, oracle=oracle, L_um=L_um, s_um=s_um)


# ===========================================================================
# 用例：刀锋（比值恰为整数）＋ 真余量（q+0.4）＋ 容差带两侧（1e-10 / 1e-8）
# ===========================================================================
def build_cases() -> dict[str, list[dict]]:
    C: dict[str, list[dict]] = {sid: [] for sid in IN_SCOPE}

    # S01/S03 层高→层数
    for dx_um, cells, q in [(50.0, 12, 15), (25.0, 24, 12), (100.0, 8, 4), (12.5, 32, 20)]:
        Lh = dx_um * cells
        for tag, s_um, o in [("刀锋", Lh / q, exact_ceil(Lh, Lh / q)),
                             ("余量q+0.4", Lh / (q + 0.4), exact_ceil(Lh, Lh / (q + 0.4))),
                             ("带内上1e-10", step_above(Lh, q, 1e-10), None),
                             ("带外上1e-8", step_above(Lh, q, 1e-8), None)]:
            kw = dict(dx=dx_um * UM, lt=s_um * UM, nz=int(cells) + 1)
            for sid in ("S01_layer_count", "S03_layer_z_heights"):
                C[sid].append(case(f"{tag} dx={dx_um} cells={cells} q={q}", kw, "lt", o, Lh, s_um))

    # S02 基板厚→格数：除数是 part.spacing ⇒ 用例的 s_um 必须**就是**夹具的 pdx
    for dx_um, q in [(50.0, 6), (25.0, 16), (12.5, 20)]:
        Lh = dx_um * q
        C["S02_with_baseplate"].append(
            case(f"刀锋 dx={dx_um} q={q}", dict(thickness=Lh * UM, pdx=dx_um * UM), "thickness",
                 exact_ceil(Lh, dx_um), Lh, dx_um))
        C["S02_with_baseplate"].append(
            case(f"余量q+0.4 dx={dx_um}",
                 dict(thickness=(Lh + 0.4 * dx_um) * UM, pdx=dx_um * UM), "thickness",
                 exact_ceil(Lh + 0.4 * dx_um, dx_um), Lh + 0.4 * dx_um, dx_um))

    # S04 margin→pad
    for dx_um, q in [(50.0, 3), (25.0, 5), (12.5, 8)]:
        Lh = dx_um * q
        C["S04_crop_margin"].append(
            case(f"刀锋 dx={dx_um} q={q}", dict(dx=dx_um * UM, margin=Lh * UM), "margin",
                 exact_ceil(Lh, dx_um), Lh, dx_um))
        C["S04_crop_margin"].append(
            case(f"余量q+0.4 dx={dx_um}", dict(dx=dx_um * UM, margin=(Lh + 0.4 * dx_um) * UM),
                 "margin", exact_ceil(Lh + 0.4 * dx_um, dx_um), Lh + 0.4 * dx_um, dx_um))

    # S05 对角线→道数（sqrt2 ⇒ oracle 不可精确，看 rel/stable）
    part = mk_part(50e-6, 5, nx=5, ny=5)
    lo, hi = part.bbox()
    diag_um = float(np.linalg.norm(np.asarray(hi)[:2] - np.asarray(lo)[:2])) / UM
    for q in (4, 8, 20):
        C["S05_zigzag_n_line"].append(
            case(f"整数跨度 q={q}", dict(hatch=diag_um / q * UM), "hatch", None, diag_um, diag_um / q))
        C["S05_zigzag_n_line"].append(
            case(f"余量q+0.4 @q={q}", dict(hatch=diag_um / (q + 0.4) * UM), "hatch",
                 None, diag_um, diag_um / (q + 0.4)))

    # S06 粉末床粒子数：dz = n_layers·lt，sp = 2·d50 ⇒ lt=sp 时比值恰为 n_layers（刀锋）
    for sp_um, n_layers in [(50.0, 3), (50.0, 8), (40.0, 5)]:
        d50 = sp_um / 2
        C["S06_powder_lattice"].append(
            case(f"刀锋 sp={sp_um} n_layers={n_layers}",
                 dict(d50=d50 * UM, lt=sp_um * UM, n_layers=n_layers), "lt",
                 None, n_layers * sp_um, sp_um))
        C["S06_powder_lattice"].append(
            case(f"余量q+0.4 sp={sp_um} n={n_layers}",
                 dict(d50=d50 * UM, lt=(sp_um * (n_layers + 0.4) / n_layers) * UM,
                      n_layers=n_layers), "lt", None, sp_um * (n_layers + 0.4), sp_um))

    # S07 DEM 网格 nx：造 pdx 使 span = q·sp（sp＝2·stats d50，运行期实测）
    sp_eff_um = 2.0 * float(ParticleCollection.from_part_bed(
        mk_part(50e-6, 5, nx=5, ny=5), layer_thickness=40e-6, d50=25e-6, n_layers=1,
        seed=0).stats()["d50"]) / UM
    for q in (4, 8, 16):
        pdx = q * sp_eff_um / 8.0
        C["S07_powder_bed_params"].append(
            case(f"刀锋 q={q} sp_eff={sp_eff_um:.6f}µm", dict(pdx=pdx * UM), "pdx",
                 None, q * sp_eff_um, sp_eff_um))
        C["S07_powder_bed_params"].append(
            case(f"余量q+0.4 @q={q}", dict(pdx=(q + 0.4) * sp_eff_um / 8.0 * UM), "pdx",
                 None, (q + 0.4) * sp_eff_um, sp_eff_um))

    # S08/S09 hatch 线数。**刀锋必须按夹具实测的浮点跨度构造**——首跑 S08 报 0 翻转，查下来是
    # 探针缺陷而非位点免疫：`(400.0/4)*1e-6 = 9.999999999999999e-05`，比 1e-4 小 1 ulp ⇒
    # 生产侧比值 4.0000000000000004，离整数有**一整个 ulp**，±1ulp 扰动够不着 ⇒ 假稳定。
    # 改 `span_m/q` 后 double 比值恰为 q ⇒ 刀锋名副其实（`spellings` 的 mul 拼写随即翻转）。
    span8 = float(np.asarray(mk_part(50e-6, 5, nx=9, ny=9).bbox()[1])[1]
                  - np.asarray(mk_part(50e-6, 5, nx=9, ny=9).bbox()[0])[1])   # = 0.0004 m
    ax9 = np.arange(9) * 50e-6
    span9 = float(ax9[-1] - ax9[0])
    span_um = 8 * 50.0                       # 设计意图：9 个节点 ⇒ 8 个体素间隔 ⇒ 400µm
    for q in (4, 8, 10):
        C["S08_gui_hatch_lines"].append(
            case(f"刀锋 span={span8 / UM:g}µm q={q}", dict(hatch=span8 / q), "hatch",
                 exact_floor1(span_um, span_um / q), span_um, span_um / q))
        C["S08_gui_hatch_lines"].append(
            case(f"余量q+0.4 @q={q}", dict(hatch=span8 / (q + 0.4)), "hatch",
                 exact_floor1(span_um, span_um / (q + 0.4)), span_um, span_um / (q + 0.4)))
    for q in (4, 8, 16):
        C["S09_am_hatch_lines"].append(
            case(f"刀锋 span={span9 / UM:g}µm q={q}", dict(hatch=span9 / q), "hatch",
                 exact_ceil(span_um, span_um / q), span_um, span_um / q))
        C["S09_am_hatch_lines"].append(
            case(f"余量q+0.4 @q={q}", dict(hatch=span9 / (q + 0.4)), "hatch",
                 exact_ceil(span_um, span_um / (q + 0.4)), span_um, span_um / (q + 0.4)))

    # S10/S11 复刻
    for q, Lh, sp in [(6, 300.0, 50.0), (16, 400.0, 25.0), (24, 300.0, 12.5)]:
        C["S10_powderbed_ngrid"].append(
            case(f"刀锋 lx={Lh} dx={sp} q={q}", dict(lx=Lh * UM, dx=sp * UM), "dx",
                 exact_ceil(Lh, sp) + 4, Lh, sp))
        C["S10_powderbed_ngrid"].append(
            case(f"余量q+0.4 lx={Lh} dx={sp}", dict(lx=(Lh + 0.4 * sp) * UM, dx=sp * UM), "dx",
                 exact_ceil(Lh + 0.4 * sp, sp) + 4, Lh + 0.4 * sp, sp))
    for q, Lh, sp in [(6, 300.0, 50.0), (8, 600.0, 75.0)]:
        C["S11_server_mpagrid"].append(
            case(f"刀锋 span={Lh} dx={sp} q={q}", dict(span=Lh * UM, dx=sp * UM), "dx",
                 exact_ceil(Lh, sp) + 2, Lh, sp))
        C["S11_server_mpagrid"].append(
            case(f"余量q+0.4 span={Lh}", dict(span=(Lh + 0.4 * sp) * UM, dx=sp * UM), "dx",
                 exact_ceil(Lh + 0.4 * sp, sp) + 2, Lh + 0.4 * sp, sp))
    return C


# ===========================================================================
# K0 普查
# ===========================================================================
def find_site(rel: str, text: str) -> str | None:
    for f, pre, mark, sid in ALL_LINES:
        if f != rel:
            continue
        if text == pre or (mark and mark in text):
            return sid
    return None


def run_census() -> tuple[list[str], dict[str, int], list[str]]:
    hdr("K0 普查：src/**/*.py 的每一处 ceil/floor/round/rint")
    hits = []
    for p in sorted(SRC.rglob("*.py")):
        rel = str(p.relative_to(REPO))
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            # 【跑后修正 K0-1】PAT 为"吸附行改后仍可见"扩了 `count_ceil(`/`count_floor(` 一支，
            # 于是 `amforge/counts.py` 的**定义行**（`def count_ceil(ratio: float) -> int:`）也被
            # 命中——定义不是取整**位点**，把它塞进登记表等于把普查口径搞脏。⇒ 普查跳过 def/class
            # 声明行；真正的取整位点（函数体内的 `math.ceil(...)`）仍在册，S00_counts_helper 记它。
            t = line.strip()
            if t.startswith(("def ", "async def ", "class ")):
                continue
            if PAT.search(line):
                hits.append((rel, i, line.strip()))
    unreg = []
    for rel, ln, text in hits:
        sid = find_site(rel, text)
        if sid is None:
            unreg.append(f"未登记 {rel}:{ln} {text[:60]}")
            print(f"  [UNREGISTERED] {rel}:{ln}  {text[:74]}")
        else:
            print(f"  [{SITES[sid]['cls']:>11}] {sid:<24} {rel}:{ln}  {text[:52]}")
    print(f"\n全树命中 {len(hits)} 处；登记 {len(ALL_LINES)} 行；未登记 {len(unreg)} 处")

    print("\nK0b 每条 in-scope 登记行的文本绑定与吸附状态：")
    bad_snap, miss_text = [], []
    counts: dict[str, int] = {}
    for sid in IN_SCOPE:
        n_ok = 0
        for e in SITES[sid]["lines"]:
            lines = (REPO / e["file"]).read_text(encoding="utf-8").splitlines()
            live = [l.strip() for l in lines if e["mark"] in l and PAT.search(l)]
            if not live:
                miss_text.append(f"{sid}:{e['file']}:{e['pre'][:40]}")
                continue
            tok = e["post"] or ("count_ceil" if SITES[sid]["fam"] == "ceil"
                                else "count_floor")
            if e["file"].startswith("src/diffmech"):
                tok = "1e-9"
            snapped = any(tok in l for l in live)
            n_ok += int(snapped)
            print(f"   {sid:<24} {e['file'].split('/')[-1]:<18} 含标记({tok})<={snapped!s:<5} "
                  f"行={live[0][:62]}")
        counts[sid] = n_ok
        if n_ok < len(SITES[sid]["lines"]):
            bad_snap.append(sid)
    if miss_text:
        print(f"   ⚠ 登记 mark 在文件中定位失败的行 {len(miss_text)} 处：{miss_text}")
    return bad_snap, counts, unreg + miss_text


# ===========================================================================
# K1/K2（＋K3 读数）
# ===========================================================================
def run_stability() -> dict[str, dict]:
    hdr("K1/K2 同一物理量换 4 拼写（字面量/±1ulp/µm×1e-6）⇒ 整数是否全同 ＋ oracle ＋ rel")
    out: dict[str, dict] = {}
    for sid in IN_SCOPE:
        rows, flips, unst, unor = [], 0, [], []
        for c in CASES[sid]:
            bound = site_text_bound(sid)
            if sid in REPL:
                calc, keys = REPL[sid]
                vals = {}
                for name, v in spellings(c["kw"][c["div"]]).items():
                    args = {k: (v if k == c["div"] else c["kw"][k]) for k in keys}
                    vals[name] = int(calc(*[args[k] for k in keys]))
                d6 = {k: (trunc6(c["kw"][c["div"]]) if k == c["div"] else c["kw"][k])
                      for k in keys}
                mul6 = int(calc(*[d6[k] for k in keys]))
            else:
                fn, div = FUNCS[sid]
                vals = {name: int(fn(**{**c["kw"], div: v}))
                        for name, v in spellings(c["kw"][div]).items()}
                mul6 = int(fn(**{**c["kw"], div: trunc6(c["kw"][div])}))
            ints = list(vals.values())
            stable = len(set(ints)) == 1
            o_ok = c["oracle"] is None or all(v == c["oracle"] for v in ints)
            rel = c["L_um"] / c["s_um"]
            in_band = abs(rel - round(rel)) <= TOL_REL * max(1.0, abs(rel))
            if (not stable) or (not o_ok):
                flips += 1
            if not stable:
                unst.append(c["tag"])
            if not o_ok:
                unor.append(f"{c['tag']}: {sorted(set(ints))} vs oracle {c['oracle']}")
            rows.append(dict(tag=c["tag"], rel=rel, oracle=c["oracle"], vals=vals,
                             stable=stable, o_ok=o_ok, in_band=in_band, bound=bound,
                             mul6=mul6,
                             kw=c["kw"], s_um=c["s_um"], L_um=c["L_um"]))
        out[sid] = dict(rows=rows, flips=flips, unst=unst, unor=unor, n=len(rows),
                        bound=all(r["bound"] for r in rows))
        print(f"\n-- {sid} （{len(rows)} 例，异常例={flips}，文本绑定={out[sid]['bound']}）")
        for r in rows:
            v = r["vals"]
            print(f"   {r['tag']:<32} rel={r['rel']:>19.12f} band={r['in_band']!s:<5} "
                  f"o={r['oracle']!s:<5} lit={v['lit']} m1={v['m1']} p1={v['p1']} mul={v['mul']} "
                  f"stable={r['stable']!s:<5} o_ok={r['o_ok']!s:<5} mul6={r['mul6']}")
        n6 = sum(1 for r in rows if r["mul6"] not in set(r["vals"].values()))
        print(f"   （诊断：mul6≠四拼写任一值的例 = {n6}/{len(rows)} ⇒ 6 位截度改的是**物理量**"
              f"而非拼写，故不进判据，见 spellings() 【跑后修正 SP-1】）")
    return out


def site_text_bound(sid: str) -> bool:
    """调用/复刻都必须与登记原文仍一致（防探针与生产漂移＝§26.17 的教训）。"""
    for e in SITES[sid]["lines"]:
        try:
            lines = (REPO / e["file"]).read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return False
        if any(e["mark"] in l and PAT.search(l) for l in lines):
            return True
    return False


def run_k4(per_site) -> list[str]:
    hdr("K4 覆盖不退化（ceil 族：min(n)·step 不得比 L 少超过容差，除非该例在带内）")
    bad = []
    for sid, v in per_site.items():
        if SITES[sid]["fam"] != "ceil":
            continue
        for r in v["rows"]:
            n_min = min(r["vals"].values())
            deficit = r["L_um"] - n_min * r["s_um"]
            ok = r["in_band"] or deficit <= TOL_REL * r["L_um"]
            print(f"   {sid:<24} {r['tag']:<32} n_min={n_min} 少覆盖={deficit:.9g}µm "
                  f"band={r['in_band']!s:<5} {'OK' if ok else 'BAD'}")
            if not ok:
                bad.append(f"{sid}:{r['tag']}")
    return bad


def run_grid_axes_control() -> list[str]:
    hdr("K2 内建正对照：_grid_axes（T1 已修）必须通过同一判据 ⇒ 证明门槛可被正确实现达到")
    ext = 0.8e-3
    bounds = [(-ext / 2, ext / 2)] * 3
    bad = []
    for dx_um, n in [(50., 16), (25., 32), (12.5, 64), (100., 8), (20., 40)]:
        vals = {k: len(_grid_axes(bounds, v)[0]) for k, v in spellings(dx_um * UM).items()}
        ok = set(vals.values()) == {n + 1}
        print(f"   dx={dx_um}µm 期望节点数 {n + 1} → {vals}  {'OK' if ok else 'BAD'}")
        if not ok:
            bad.append(str(dx_um))
    return bad


def run_a0_block() -> None:
    hdr("K5 A0 夹具整数块（本轮改动不得移动；两阶段日志此段必须逐行相同）")
    bx, by, bz = 1.2e-3, 0.6e-3, 0.4e-3
    for dx_um in (50.0, 25.0, 12.5):
        dx = dx_um * UM
        g = G.from_sdf_fn(
            lambda x: jnp.maximum(jnp.maximum(jnp.abs(x[..., 0]) - bx / 2,
                                              jnp.abs(x[..., 1]) - by / 2),
                                  jnp.abs(x[..., 2]) - bz / 2),
            bounds=[(-bx / 2, bx / 2), (-by / 2, by / 2), (-bz / 2, bz / 2)],
            spacing=dx, name="coupon")
        plan = ProcessPlan.uniform(1, modality="SLM", laser_power=600.0, scan_speed=0.8,
                                   layer_thickness=bz, hatch_spacing=1.4e-4,
                                   beam_radius=100e-6, absorption=0.45, preheat_temp=400.0)
        extra = ""
        try:
            from amforge.thermal_enthalpy import _footprint, _scan_topology
            topo = _scan_topology(g.coords(), plan, dim=3, solid=_footprint(g))
            extra = (f" n_layers={float(topo[0]):.12g} n_lines={float(topo[1]):.12g}"
                     f" path_len={float(topo[-1]):.12g}")
        except Exception as e:                                    # noqa: BLE001
            extra = f" (topo 取数失败: {type(e).__name__}: {e})"
        print(f"   dx={dx_um:>5.1f}µm shape={tuple(int(v) for v in g.shape)} "
              f"nvox={int(jnp.asarray(g.sdf).size)}{extra}")


def run_split_block() -> None:
    hdr("K6 三套舍入口径实测（只打印 ⇒ #31，本轮不统一）")
    for tag, Lh, s in [("试片 y 跨度 0.6mm / hatch 140µm", 600.0, 140.0),
                       ("试片 y 跨度 0.6mm / hatch 150µm", 600.0, 150.0),
                       ("试片 y 跨度 0.6mm / hatch 100µm", 600.0, 100.0),
                       ("A0 高 400µm / lt 40µm", 400.0, 40.0),
                       ("A0 高 400µm / lt 60µm", 400.0, 60.0),
                       ("A0 高 400µm / lt 120µm", 400.0, 120.0)]:
        print(f"   {tag:<32} 比值={float(F(Lh) / F(s)):>17.12g}  ceil={exact_ceil(Lh, s)}  "
              f"round={exact_round(Lh, s)}  floor+1={exact_floor1(Lh, s)}")


def env_block() -> str:
    hdr("ENV")

    def sh(cmd):
        try:
            return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                                  timeout=25).stdout.strip()
        except Exception as e:                                    # noqa: BLE001
            return f"<{e}>"
    dirty = [l for l in sh(["git", "status", "--porcelain", "src", "tests"]).splitlines() if l]
    print(f"python     = {sys.version.split()[0]}   jax={jax.__version__} "
          f"x64={jax.config.jax_enable_x64}")
    print(f"HEAD       = {sh(['git', 'rev-parse', '--short', 'HEAD'])}")
    print(f"dirty_src  = {len(dirty)}  {dirty[:12]}")
    print(f"device_pin = CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<未设>')!r} "
          f"backends={jax.local_devices()}")
    return sh(["git", "rev-parse", "--short", "HEAD"])


def main() -> int:
    has_counts = (SRC / "amforge/counts.py").exists()
    phase = os.environ.get("T3_PHASE", "auto")
    if phase == "auto":
        phase = "post" if has_counts else "pre"
    head = env_block()
    print(f"\n>>> phase={phase}  counts.py 存在={has_counts}  HEAD={head}")
    TOL_REPL[0] = 0.0 if phase == "pre" else 1e-9       # 复刻位点跟随该阶段的生产式
    print(f">>> 复刻位点（S10/S11）容差宽度 = {TOL_REPL[0]:g}")
    bad_snap, snap_counts, k0_problems = run_census()
    CASES.update(build_cases())
    per_site = run_stability()
    bad_cov = run_k4(per_site)
    grid_bad = run_grid_axes_control()
    run_a0_block()
    run_split_block()

    hdr("记分")
    tot = sum(v["flips"] for v in per_site.values())
    n = sum(v["n"] for v in per_site.values())
    with_flip = sorted(s for s, v in per_site.items() if v["flips"] > 0)
    zero = sorted(s for s in per_site if s not in with_flip)
    unexplained = [s for s in zero if s not in NO_FLIP_ALLOWED]
    unbound = [s for s, v in per_site.items() if not v["bound"]]
    n_lines_in_scope = sum(len(SITES[s]["lines"]) for s in IN_SCOPE)
    print(f"K0 普查完备            : {'OK' if not k0_problems else 'VOID'}  "
          f"问题 {len(k0_problems)} 项：{k0_problems}")
    print(f"K0b in-scope 已吸附行   : {sum(snap_counts.values())}/{n_lines_in_scope}  "
          f"未吸附位点={bad_snap}  文本漂移位点={unbound}")
    if phase == "pre":
        print(f"K1 翻转正对照          : 翻转例={tot}/{n}  有翻转位点={len(with_flip)}/{len(per_site)}  "
              f"0 翻转且未预注册={unexplained}")
        for s in zero:
            why = NO_FLIP_ALLOWED.get(s)
            print(f"     · {s} 0 翻转 ⇒ {'预注册: ' + why if why else '未预注册 ⇒ K1 FAIL'}")
    else:
        print(f"K1 改后不判（改后 0 翻转正是 K2 的内容）: 翻转例={tot}/{n}"
              f"  有翻转位点={len(with_flip)}/{len(per_site)}  ⇒ 正对照证据在 `_pre.log`")
    if phase == "post":
        unst = {s: v["unst"] for s, v in per_site.items() if v["unst"]}
        unor = {s: v["unor"] for s, v in per_site.items() if v["unor"]}
        ok2 = (not unst and not unor and not grid_bad and not bad_snap
               and not unbound and not k0_problems)
        print(f"K2 改后 4 拼写同整数＋oracle: {'OK' if ok2 else 'FAIL'}  "
              f"不稳定={ {k: len(v) for k, v in unst.items()} }  "
              f"oracle不符={ {k: len(v) for k, v in unor.items()} }  "
              f"_grid_axes 失败={grid_bad}")
        for s, lst in list(unst.items()) + list(unor.items()):
            for t in lst:
                print(f"     · {s}: {t}")
        print(f"K4 覆盖不退化          : {'OK' if not bad_cov else 'FAIL'}  {bad_cov}")
        miss = []
        for (sid, tag), want in EXPECTED_POST.items():
            row = next((r for r in per_site[sid]["rows"] if r["tag"] == tag), None)
            got = None if row is None else sorted(set(row["vals"].values()))
            hit = row is not None and got == [want]
            print(f"   K2b {sid} {tag}: 期望 {want} 实得 {got} {'OK' if hit else 'FAIL'}")
            if not hit:
                miss.append(f"{sid}:{tag}")
        print(f"K2b 改后预测值命中      : {'OK' if not miss else 'FAIL'}  {miss}")
        if miss:
            ok2 = False
    else:
        print("K2 改前不判（改前必须有翻转正是 K1 的内容）")
    print("K3 读 rel/band 列：带内(1e-10)例改后须吸附到整数下侧；带外(1e-8)例须仍为 ceil")
    print("K5 A0 段：两阶段日志此段 diff 必须为空（外层判定）")
    print("K6 三套舍入差 ⇒ 登记 #31（本轮不统一）")
    # 【跑后修正 GATE-1】记分原先恒 `return 0` ⇒ "门槛"只是给人读的字符串，没有可失败面。
    # 现按阶段返回退出码：pre＝K0 完备 ∧ K1 正对照成立；post＝K0 ∧ K0b ∧ K2 ∧ K2b ∧ K4。
    # （post 阶段不再判 K1 的 0 翻转清单：改后 0 翻转正是本轮要的结论，判它＝自相矛盾。）
    if phase == "post":
        gates = {"K0": not k0_problems,
                 "K0b": not bad_snap and not unbound,
                 "K2": ok2, "K2b": not miss, "K4": not bad_cov}
    else:
        gates = {"K0": not k0_problems,
                 "K1": tot > 0 and not unexplained and len(with_flip) > 0}
    hard = [k for k, v in gates.items() if not v]
    print(f"门槛汇总（{phase} 硬门）   : {'PASS' if not hard else 'FAIL'}  "
          f"判了={list(gates)}  失败={hard}")
    return 0 if not hard else 1


if __name__ == "__main__":
    sys.exit(main())
