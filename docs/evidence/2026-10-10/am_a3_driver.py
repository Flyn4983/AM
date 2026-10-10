"""A3 外部对照驱动（任务 #10）：NIST mds2-3662「beam-on-plate」18 道顶面熔池。

基准侧事实全部**现取**，不在本件里重打一遍：
  * 工艺常数 ← ``docs/evidence/2026-10-07/am_a3_fixture.json``（域、功率、速度、hatch、
    D4σ→r_b、α 两档、T0）。
  * 轨迹 ← ``_refs/benchmarks/nist_mds2-3662/Scan Strategy Data/*.csv``，经
    ``amforge.scan_program.read_track_table``（µ 表头非 UTF-8，内部 latin-1 兜底）＋
    ``program_from_track_table``（``unit=1e-6`` 显式给，不猜单位）。
  * 靶值 ← ``docs/evidence/2026-10-07/am_a3_target_table.csv``（tidy，每格带 src_cell）。
  * dt／步数 ← 求解器自己的 ``_stability_limit_dx2``（不另写一套口径）。
  * 沉能份额 ← 求解器自己的 ``_cell_integrated_source``（G4 用的就是生产那条式子）。

开发日志 §26.7 N3 的 ①–⑨ 判据在本件的落点：①偏差分母＝**合并散差**（3 重复 × 2 操作员，
n≤6）与**中位数＋极差**两个数 → :func:`targets`；②趋势为主判据 → ``--stage full``；
③宽度/面积用 cut（线性交叉）口径并**并报布尔版** → :func:`readout_slab`；④合格线
宽 ≤14%／积 ≤11%、>20% 未达 → ``BARS``；⑤先登记 voxel-step 代价才许扩到 18 道 →
:func:`budget_gate`；⑥按列位置取数（靶表已是 tidy CSV，拼写错在解析轮已断到）；
⑦α 档位跑前二选一 → J2；⑧与基准同条件＝关掉基准没有的物理 → J5；⑨靶曲线取 Set 2、
离群两版并报 → :func:`targets`。

本轮实测把口径**新增**成下面几条（都是被数据或正文逼出来的，不是偏好）：
  J1 **逐 N 重跑**：README 的取样方式是"扫到第 N 道就停、测**最后一道**"，所以每个 N 是
     一次独立正向（把折线截到第 N 个开光段的末点），不许"跑完 18 道再取尾"。
  J2 α 档位跑前二选一：**复现档 α=0.54**（作者标定值，正文 Table 1）为本轮选用；预测档
     α=0.39（材料卡 ``absorptivity``）留作第二臂。选了哪一档写进日志，跑后不回头换。
  J3 **等温线取 T_solidus（1563.0 K ＝ 正文的 1290 °C）**，不是 T_liquidus：正文 §2.2 逐字
     写"熔池轮廓取 T=1290 °C 等值面"⇒ 换阈值就是换被测量。liquidus 版**并报**（再读一遍，
     不改判据）。
  J4 **读出时刻＝最后一道开光段内帧的逐格最大值**，不是全史 ``peak_temperature``。
     实测理由：hatch=110 µm＝2.59·r_b，而靶宽 143.7–400.5 µm ⇒ 相邻道的熔化区**本来就
     连成一片**；用全史峰值读数会把前 N−1 道的痕迹一起算进来，N 越大越离谱，而那正是
     本基准要考的 1→18 趋势。帧间隔按正文自己的抽取节律 **≤0.149 ms** 定
     （``FRAME_CADENCE_S``）。第二版 R2＝"最后一道**新**熔到的"（减去窗口前一帧已达标的
     格），两版并报。帧指标与步指标的换算**复用求解器式子** ``stride = n_steps // n_frames``
     并断言 ``len(frames) == ceil(n_steps/stride)`` ⇒ 映射错了就报红，不静默错位。
  J5 与基准同条件＝关掉基准没有的物理：``evap_model="none"``、``h_cool=0``、
     ``source_model="integrated"``、``resolution_policy="strict"``、实体表面绝热（求解器缺省）、
     IC 走 ``boundary_conditions``（基准 T₀=20 °C=293.15 K，而材料卡 ``T_ambient=300.0``，
     且 preheat 分支要求 ``preheat > T_amb`` 才生效 ⇒ 不给 IC 就会被顶成 300 K）。
     **不挂粉末档**（基准是实心板）。⚠ ``heat_scale`` 这一项**不能**留 1.0，理由见 J11。
  J6 形态档必须 ``dx ≤ r_b/2``；``dx=r_b`` 只用于接线烟测，那一条**不产出**任何形态数字。
  J7 偏差判据：宽度平均 ≤14% 达标、>20% 未达；面积 ≤11%／>20%；**必须同时报趋势**
     （1→18 单调上升到饱和的位置）——单道实测散差 p50 已达 15.7%（宽）／19.8%（积），
     分母比判据还宽，单点没有分辨力。``BARS`` 是这条的机器口径。
  J10 **时间映射的实测偏差先量化再放行**：求解器的时间轴是"弧长/扫描速度"（#18 第 1 条
     口径，不动它），而 CSV 的实测是 开光段 21.12 ms / 间隙 8.67 ms（收敛形，17 个间隙，
     中位 0.510 ms），跳段弧线 6650.7 µm ⇒ 模型间隙只有 6.93 ms ⇒ 道间冷却窗口比真实短
     约两成，方向＝后段道预热偏高。本轮**不**偷偷改：G3 只逐道判**开光段**（跳段真实速度
     含加减速，本就不该用标称速度判），间隙比值作为实测条打印并随结果同报。若趋势系统性
     偏高，预注册的修法是"把跳段弧线拉长到真实间隙时长"（功率为 0 时位置不参与能量 ⇒
     该修法不引入新物理），需另轮登记。
  J11 **能量归一化臂**：基准的边界条件是顶面 Neumann ``αI``（全部吸收功率从表面进），
     我方 ``integrated`` 源是**体热源**，束面取 ``layer_z = top_z``（顶面节点中心，比设计面
     z=0 低半个体素以内）⇒ 轴向高斯的上半叶落在实体外，G4 实测只有约 0.6 的份额进得了
     工件（这不是离散误差，是模型形式差）。两臂并报：
     ``raw``＝heat_scale 1.0（默认口径，能量少投 ~38%，作为**量级证据**保留），
     ``norm``＝heat_scale = 1/G4 份额（使 Σ注入＝α·P，与基准同一条能量陈述）。
     **主判据走 norm 臂**，跑前定死；raw 臂只用于说明这条修正的大小与方向。
  J12 **预检把两条登记闸门证伪了，改动在此登记**（两条都是"判据写错"，不是"放松判据"）：
     ① 原 G4 写作「沉能份额 ≥0.99」，与同时登记的 J11 互相矛盾——J11 事先就说了这一档
     只有约 0.6 并且**正因如此**才立 norm 臂。一条与另一条预注册判据逻辑上不可能同绿的
     闸门等于没有闸门。拆成 G4a（份额命中 :func:`expected_capture` 的闭式解，两个点阵
     相位各一次，容差 1e-4）／G4b（深度对照 ≥0.99 ⇒ 亏损只在跨顶面那一侧）／G4c（面内
     平移控件 ≤1e-3）。顶面份额实测 0.615861（本轮预检），其与闭式解的差由 G4a **现算现报**，
     不在本件里预填。
     ② 原 G5 写作「顶层满布」＝要求顶面 ``nx·ny`` 全实体，而 ``_grid_axes`` 按
     ``ceil((hi−lo)/dx)`` 铺点，域 3000×2500µm 不是 dx=21.25µm 的整数倍 ⇒ 点阵在 +x/+y
     侧各多出一圈设计域外的空节点，**任何正确实现都过不了**。改为它本来该问的：空节点
     全部落在这一圈上（并断言圈上确有空节点，防止判据退化成空集上的真），且**读数足迹**
     （末道 ±3r_b 再各留一列给线性交叉 × 该道 y 带）满布。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", "src"))

import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)   # 不开则 read_track_table 静默降 f32（实测有警告）

from amforge import geometry as G  # noqa: E402
from amforge.core.contracts import ProcessPlan, solid_mask  # noqa: E402
from amforge.materials import get_material  # noqa: E402
from amforge.scan_program import (  # noqa: E402
    gate_from_power, program_from_track_table, read_track_table,
)
from amforge.thermal_enthalpy import (  # noqa: E402
    _cell_integrated_source, _stability_limit_dx2, solve_enthalpy_thermal,
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
FIXTURE = os.path.join(REPO, "docs/evidence/2026-10-07/am_a3_fixture.json")
TARGETS = os.path.join(REPO, "docs/evidence/2026-10-07/am_a3_target_table.csv")
SCAN_DIR = os.path.join(REPO, "_refs/benchmarks/nist_mds2-3662/Scan Strategy Data")
MATERIAL = "IN625"
COL_X, COL_Y, COL_POWER, COL_TIME = 0, 1, 2, 3
#: N3⑤：全量放行的登记预算 [s]，必须拿 pilot 实测墙钟外推 ΣN 来比。
BUDGET_S = 4 * 3600.0
#: J4：帧抽取节律上限 [s]，取自正文 §2.2 的"约 0.149 ms 一次"。
FRAME_CADENCE_S = 0.149e-3
#: 帧缓存上限 [B]：超过则收紧帧数并**打印**实际节律（不许静默变粗）。
FRAME_MEM_CAP = 12e9
#: J7 的机器口径（达标线 / 未达线，%）。
BARS = dict(width_um=(14.0, 20.0), area_um2=(11.0, 20.0))


def load_fixture():
    fx = json.load(open(FIXTURE, encoding="utf-8"))["benchmark"]
    return {
        "domain_um": [float(v) for v in fx["domain_um"]],
        "T0_C": float(fx["T0_C"]),
        "laser_power_W": float(fx["laser_power_W"]),
        "scan_speed_mm_s": float(fx["scan_speed_mm_s"]),
        "hatch_um": float(fx["hatch_um"]),
        "tracks": int(fx["tracks"]),
        "alpha_calibrated": float(fx["alpha_calibrated"]),
        "alpha_ours": float(fx["alpha_ours"]),
        "D4sigma_um": float(fx["D4sigma_um"]),
    }


def beam_radius_m():
    """r_b 的唯一出处＝夹具 json 的 ``beam_radius_m``（#22/T4 定标的 1/e² 口径）。"""
    return float(json.load(open(FIXTURE, encoding="utf-8")
                            )["spot_convention"]["beam_radius_m"])


def power_runs(power_col):
    """开光段＝功率列上 ``>0`` 的极大连续行；返回 ``[(起行, 止行), …]``。"""
    on = np.asarray(power_col) > 0.0
    runs, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j + 1 < len(on) and on[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def read_strategy(name):
    table = read_track_table(os.path.join(SCAN_DIR, name))
    cols = np.asarray(table["columns"])
    return table, cols, power_runs(cols[:, COL_POWER])


def program_upto(cols, runs, n_tracks, layer_z):
    """截断到第 ``n_tracks`` 个开光段的**末点**（J1 的逐 N 停止）。"""
    stop = runs[n_tracks - 1][1]
    sub = {"columns": cols[:stop + 1], "header": [],
           "index": {i: f"col{i}" for i in range(cols.shape[1])}}
    prog = program_from_track_table(sub, x_col=COL_X, y_col=COL_Y,
                                    power_col=COL_POWER, unit=1e-6, layer_z=layer_z)
    return sub, prog


def plate_geometry(dx_m, domain_um):
    """实心板：面内＝基准域，z∈[-depth,0]，顶面在 z=0（不挂粉末层）。"""
    lx, ly, lz = [v * 1e-6 for v in domain_um]
    cx, cy, cz = lx / 2.0, ly / 2.0, -lz / 2.0

    def box(x):
        return np.maximum(
            np.maximum(np.abs(x[..., 0] - cx) - lx / 2.0,
                       np.abs(x[..., 1] - cy) - ly / 2.0),
            np.abs(x[..., 2] - cz) - lz / 2.0)

    return G.from_sdf_fn(box, bounds=[(0.0, lx), (0.0, ly), (-lz, 0.0)],
                         spacing=dx_m, name="a3-plate", chunk=None)


def axes_of(geom):
    c = np.asarray(geom.coords())
    return c[:, 0, 0, 0], c[0, :, 0, 1], c[0, 0, :, 2]


def plan_for(fx, r_b, alpha, t_ic_k):
    """工艺契约：α 走 ``process.absorption``（求解器里 ``eta`` 的唯一出处）。"""
    lz = fx["domain_um"][2] * 1e-6
    return ProcessPlan.uniform(1, modality="SLM", laser_power=fx["laser_power_W"],
                               scan_speed=fx["scan_speed_mm_s"] * 1e-3,
                               layer_thickness=lz,
                               hatch_spacing=fx["hatch_um"] * 1e-6,
                               beam_radius=r_b, absorption=alpha,
                               preheat_temp=t_ic_k, dwell_time=0.0)


def schedule(geom, prog, plan, cfl=0.35):
    """步数＝物理曝光 /(cfl×稳定 dt)，dt 口径取求解器自己的 ``_stability_limit_dx2``。"""
    mat = get_material(MATERIAL)
    alpha0 = float(mat.k_solid) / (float(mat.rho_solid) * float(mat.cp_solid) + 1e-12)
    dx = float(geom.spacing)
    dt_rk2 = float(_stability_limit_dx2(dx, geom.dim, alpha0, 1.0))
    dt_target = dt_rk2 * cfl
    v = float(np.mean(np.atleast_1d(np.asarray(plan.scan_speed, dtype=np.float64))))
    t_exp = float(prog.total_length()) / max(v, 1e-9)
    return max(1, int(math.ceil(t_exp / dt_target - 1e-9))), t_exp, dt_target, dt_rk2, alpha0


def frame_plan(n_steps, t_total, nvox):
    """按正文节律定请求帧数；返回 (帧数, stride, 实际节律, 是否未被缓存上限收紧)。

    ``stride`` 用求解器 :986 的**同一式子** ``n_steps // n_frames``，:988 的
    ``keep`` 于是给出 ``ceil(n_steps/stride)`` 帧；调用方必须拿实得帧数对回这个式子
    （:func:`readout_from_frames` 里那条断言），否则窗口映射会静默错位。
    """
    want = max(2, int(math.ceil(t_total / FRAME_CADENCE_S)))
    cap = max(2, int(FRAME_MEM_CAP // max(nvox * 8, 1)))
    n_frames = min(want, cap, n_steps)
    stride = max(1, n_steps // n_frames)
    return n_frames, stride, stride * (t_total / n_steps), want <= n_frames


def solver_params(prog, n_steps, alpha, heat_scale, t_ic_k, n_frames):
    """J5 的开关清单 + J11 的归一化臂 + J4 的帧记录，一次给齐。"""
    return dict(material=MATERIAL, n_steps=n_steps, scan_program=prog,
                source_model="integrated", resolution_policy="strict",
                heat_scale=heat_scale, h_cool=0.0, evap_model="none", cfl=0.35,
                record_frames=True, n_animation_frames=n_frames,
                # 基准 T0=20°C=293.15K，而材料卡 T_ambient=300.0K；preheat 分支要求
                # preheat>T_amb 才生效，故 IC 必须走 boundary_conditions，否则被顶成 300K。
                boundary_conditions={"bcs": [], "ic": {"kind": "uniform",
                                                       "value": t_ic_k}})


def expected_capture(z_top, dx, dp, z_beam):
    """G4a 的**独立解析期望**：体热源被顶面半空间截掉后剩下的份额。

    ``_cell_integrated_source`` 的轴向因子是相邻 erf 之差，在无限点阵上逐格相加**望远镜
    求和＝1**；实体掩膜只切掉顶面节点上表面（``z_top + dx/2``）以上那一段，面内居中时
    （光束离侧边界 ≫ σ_in）截获份额有闭式解

        C = ½ [1 + erf((z_top + dx/2 − z_beam) / dp)]

    除的是 ``dp`` 而**不是** ``dp/√2``——``_axis_frac`` 里 ``erf(Δ/s)`` 对应的标准差是
    ``s/√2``（#22 那条 √2 口径分裂正是这么咬人的）。这条式子与生产实现走的是**不同的路**
    （闭式望远镜 vs 逐格数组求和），所以它对 ``s_z``、对束面 z、对点阵相位都敏感：任何一
    处被改动，G4a 会如实报红，而不是跟着一起改。
    """
    return 0.5 * (1.0 + math.erf((z_top + 0.5 * dx - z_beam) / dp))


def deposited_fraction(geom, prog, n_steps, r_b, *, n_probe=9, z_shift=0.0,
                       x_shift=0.0, masked=True):
    """G4/J11：实体网格截获的功率份额 ``Σ frac``（用生产同款 ``_cell_integrated_source``）。

    该式返回 ``power·frac/dV``（``dV = dx**dim``），故 ``frac = q·dx³``；这里给
    ``power=1.0``，所以份额是**纯几何量**，与 η、P 无关。取多个步位的**最小值**而不是
    中点：道靠近面内边界时份额会掉，中点恰好躲过去就是假绿。``z_shift``/``x_shift``
    是**诊断探针**（整体平移束斑以分离"轴向顶面半损失"与"面内贴边"），不参与求解。
    ``masked=False`` 同样是**控件旋钮**（只给 ``am_a3_preflight_controls.py`` 用）：去掉
    实体掩膜后读到的应是**网格截断**值（≈0.80）而非 0.616，用来证明 G4 的量确实由掩膜
    决定、不是探针的常数输出。正常调用留缺省值。
    """
    pos = np.asarray(prog.sample_steps(n_steps)[0])
    if z_shift or x_shift:
        pos = pos.copy()
        pos[:, 2] += z_shift
        pos[:, 0] += x_shift
    c = np.asarray(geom.coords())
    dx = float(geom.spacing)
    dp = r_b * 1.2                      # ＝求解器 :709 在 strict+integrated 下的取值
    solid = (np.asarray(solid_mask(geom.sdf)) > 0.5 if masked
             else np.ones(np.asarray(geom.sdf).shape, dtype=bool))
    ks = sorted({int(v) for v in np.linspace(0, n_steps - 1, min(n_probe, n_steps))})
    vals = [float(np.sum(np.where(solid, np.asarray(
        _cell_integrated_source(pos[k], c, 1.0, r_b, dp, dx)), 0.0)) * dx ** 3)
        for k in ks]
    return min(vals), max(vals), ks


def capture_probes(geom, prog, n_steps, r_b):
    """一次算齐三条探针：顶面（主量）、下移 3r_b（轴向分离）、面内平移 +2mm（控件）。"""
    lo, hi, ks = deposited_fraction(geom, prog, n_steps, r_b)
    d_lo, d_hi, _ = deposited_fraction(geom, prog, n_steps, r_b, z_shift=-3.0 * r_b)
    o_lo, o_hi, _ = deposited_fraction(geom, prog, n_steps, r_b, x_shift=+2.0e-3)
    return dict(surface_lo=lo, surface_hi=hi, n_probe=len(ks), deep_lo=d_lo, deep_hi=d_hi,
                outside_lo=o_lo, outside_hi=o_hi)


def readout_slab(slab, xax, yax, *, x_track, y_lo, y_hi, r_b, T_iso, exclude=None,
                 window_r=3.0):
    """J3/J4 的读数核：顶面薄层上按**垂直于该道**方向取等温线交叉，输出宽 [µm] 与积 [µm²]。

    cut 口径＝交叉点线性插值（本项目体素布尔量实测带 0.88–1.36× 偏置，而 130–460 µm 的宽
    在 dx=21 µm 下每侧一个体素就是 ±11%，正好吃掉 14% 的通过线）；同时**并报布尔版**
    （N3③ 要求写明口径移动幅度）。``exclude`` 是 R2 用的"上一道已熔"掩膜。
    """
    dx = float(xax[1] - xax[0])
    ix = np.where(np.abs(xax - x_track) <= window_r * r_b)[0]
    if ix.size < 5:
        return dict(ok=False, why=f"±{window_r}r_b 窗内只有 {ix.size} 列（窗比光斑还窄）")
    j_lo, j_hi = int(ix[0]), int(ix[-1])
    rows = np.where((yax >= y_lo) & (yax <= y_hi))[0]
    if rows.size == 0:
        return dict(ok=False, why=f"y∈[{y_lo*1e6:.0f},{y_hi*1e6:.0f}]µm 内无行可读")
    widths, widths_bool, touched, rows_hot = [], [], 0, 0
    for iy in rows:
        col = np.asarray(slab)[j_lo:j_hi + 1, iy]
        hot = col > T_iso
        if exclude is not None:
            hot = hot & ~np.asarray(exclude)[j_lo:j_hi + 1, iy]
        if not hot.any():
            continue
        rows_hot += 1
        widths_bool.append(int(hot.sum()) * dx)
        a = int(np.argmax(hot))
        b = int(col.size - 1 - int(np.argmax(hot[::-1])))
        if a == 0 or b == col.size - 1:
            touched += 1                      # 熔池触到窗边 ⇒ 读数被截，不可用
            continue
        xl = xax[j_lo + a - 1] + dx * (T_iso - col[a - 1]) / (col[a] - col[a - 1])
        xr = xax[j_lo + b] + dx * (col[b] - T_iso) / (col[b] - col[b + 1])
        widths.append(max(0.0, float(xr - xl)))
    if not widths:
        return dict(ok=False, why=f"顶面 {rows.size} 行里无一行的交叉落在窗内"
                                  f"（有熔行={rows_hot} 触边={touched}，阈值 "
                                  f"T_iso={T_iso:.1f}K）")
    return dict(ok=True, width_um=float(max(widths)) * 1e6,
                area_um2=float(sum(widths)) * dx * 1e12,
                width_bool_um=float(max(widths_bool)) * 1e6,
                area_bool_um2=float(sum(widths_bool)) * dx * 1e12,
                n_rows=int(rows.size), rows_hot=rows_hot, touched=touched,
                cut_minus_bool_um=(float(max(widths)) - float(max(widths_bool))) * 1e6)


def window_start_step(prog, n_steps, last_run_start_row):
    """J4：最后一道开光段（**含其前沿那一段的开光爬坡**）在步指标上的起点。"""
    cum = np.asarray(prog.cumulative_length)
    arc_from = float(cum[last_run_start_row - 1]) if last_run_start_row >= 1 else 0.0
    total = float(cum[-1])
    return max(0, int(math.ceil(n_steps * arc_from / total - 0.5)))


def readout_from_frames(evo, xax, yax, zax, top_z, sub, sub_runs, prog, n_steps,
                        n_frames, r_b, T_iso_vals):
    """J4：取"最后一道开光段内的逐格最大值"作顶面读数；出 R1（全）与 R2（新熔）两版。"""
    if evo is None:
        return dict(ok=False, why="求解器未返回帧场（record_frames 未生效）")
    frames = np.asarray(evo)
    if frames.ndim != 4:
        return dict(ok=False, why=f"帧场秩应为 4 (n_frames,x,y,z)，实得 {frames.shape}")
    stride = max(1, n_steps // max(1, n_frames))      # 与求解器 :986 同式
    expect = int(math.ceil(n_steps / stride))
    if frames.shape[0] != expect:
        return dict(ok=False, why=f"帧数 {frames.shape[0]} != ceil(n_steps/stride)="
                                  f"{expect}（stride={stride}）⇒ 步↔帧映射与求解器不一致")
    iz = int(np.argmin(np.abs(np.asarray(zax) - top_z)))
    t0 = window_start_step(prog, n_steps, int(sub_runs[-1][0]))
    sel = [k for k in range(frames.shape[0]) if k * stride >= t0]
    if len(sel) < 2:
        return dict(ok=False, why=f"窗口内只有 {len(sel)} 帧（t0={t0}, stride={stride}, "
                                  f"n_frames={frames.shape[0]}）⇒ 时间采样不足")
    r1 = np.max(frames[sel[0]:sel[-1] + 1, :, :, iz], axis=0)
    # R2 的"进入本道之前"状态＝窗口前一帧；N=1 时没有前一帧，取第一帧（此时 R2≡R1 是对的）
    k_before = sel[0] - 1 if sel[0] >= 1 else sel[0]
    r0 = frames[k_before, :, :, iz]
    arr = np.asarray(sub["columns"])
    xs, ys = arr[sub_runs[-1][0], :2] * 1e-6
    xe, ye = arr[sub_runs[-1][1], :2] * 1e-6
    out = dict(ok=True, iz=iz, t0=t0, stride=stride, n_win=len(sel),
               k_before=k_before, x_track=0.5 * (xs + xe),
               y_lo=float(min(ys, ye)), y_hi=float(max(ys, ye)))
    for key, T_iso in T_iso_vals.items():
        kw = dict(x_track=out["x_track"], y_lo=out["y_lo"], y_hi=out["y_hi"],
                  r_b=r_b, T_iso=T_iso)
        out[key] = readout_slab(r1, xax, yax, **kw)
        out[key + "_R2"] = readout_slab(r1, xax, yax, exclude=(r0 > T_iso), **kw)
    return out


def targets(track, quantity, strategy):
    """N3①/⑨：Set 2 靶值按**族**（D/C）合并两操作员 × 三重复，给中位数＋极差。

    两版并报：``all`` 含作者标注的离群重复；``no_outlier`` 去掉它。离群标注的**实测位点**
    ＝ xlsx 的 ``S46``（解析轮 §26.7 N2.5 判据④打印过 ``位点列号 = [18]``），面积块从
    列 O=14 起、列序 D1,D2,D3,C1,C2,C3 ⇒ 第 18 列＝面积 **C2** ⇒ 只有 (area_um2, C2)
    被剔除，宽度块与其余重复不动。主口径用 ``all``（含离群 ⇒ 分母更宽，更保守，
    §26.12(d)①），但两版都报，不许只给抬高后的那版。
    """
    rows = open(TARGETS, encoding="utf-8").read().splitlines()
    idx = {k: i for i, k in enumerate(rows[0].split(","))}
    got = {"all": [], "no_outlier": []}
    for line in rows[1:]:
        f = line.split(",")
        if (f[idx["set"]] != "Set2" or f[idx["quantity"]] != quantity
                or f[idx["track"]] != str(track)
                or not f[idx["direction"]].startswith(strategy)):
            continue
        v = f[idx["value"]].strip()
        if not v or v.lower() == "nan":
            continue
        val = float(v)
        got["all"].append(val)
        if not (quantity == "area_um2" and f[idx["direction"]] == "C2"):
            got["no_outlier"].append(val)
    out = {}
    for kk, vv in got.items():
        if not vv:
            continue
        arr = sorted(vv)
        out[kk] = dict(n=len(arr), median=float(np.median(arr)), mean=float(np.mean(arr)),
                       lo=arr[0], hi=arr[-1],
                       spread_pct=float((arr[-1] - arr[0]) / np.median(arr) * 100.0))
    return out


def selftest_readout(r_b, T_iso):
    """preflight 的合成控件：读数核**必须**命中绝对期望，且**必须**能报红。

    C1 绝对期望：``T = T_iso + K·(H − |x−x0|)`` 的交叉点在数学上恰为 ``x0±H`` ⇒ cut 读数
        必须等于 2H、面积必须等于 2H×带长（这不是稳定性探针，是能失败的绝对判据）。
    C2 口径边界：布尔版与 cut 版的差必须 ≤ 一个体素 dx（否则插值写错了）。
    C3 能报红：把窗收窄到比熔池还小 ⇒ 必须返回 ok=False 或 touched>0。
    C4 R2 真减：造一半行的"上一道已熔"掩膜 ⇒ 有熔行数必须严格减少。
    """
    dx = 21.25e-6
    xax = np.arange(0.0, 3000e-6, dx)
    yax = np.arange(0.0, 2500e-6, dx)
    x0, H, K = 1500e-6, 90e-6, 5e6          # 期望宽 2H=180µm，斜率 5 K/µm
    prof = T_iso + K * (H - np.abs(xax - x0))
    band = np.where((yax >= 800e-6) & (yax <= 1200e-6))[0]
    y_lo, y_hi = float(yax[band[0]]), float(yax[band[-1]])
    # slab 必须是**整张顶面薄层** (nx, ny)：readout_slab 按全局 yax 行号索引，只裁带内
    # 行会越界（本轮 IndexError 实测抓到的就是这条形状契约）。带外行给阈值以下。
    slab = np.where(((yax >= y_lo) & (yax <= y_hi))[None, :], prof[:, None], T_iso - 50.0)
    exp_w, exp_area = 2 * H * 1e6, 2 * H * (band.size * dx) * 1e12
    r = readout_slab(slab, xax, yax, x_track=x0, y_lo=y_lo, y_hi=y_hi, r_b=r_b,
                     T_iso=T_iso)
    rw = readout_slab(slab, xax, yax, x_track=x0, y_lo=y_lo, y_hi=y_hi, r_b=r_b,
                      T_iso=T_iso, window_r=0.5)
    excl = np.zeros_like(slab, dtype=bool)
    excl[:, band[:max(1, band.size // 2)]] = True
    r2 = readout_slab(slab, xax, yax, x_track=x0, y_lo=y_lo, y_hi=y_hi, r_b=r_b,
                      T_iso=T_iso, exclude=excl)
    w = r.get("width_um", float("nan"))
    ar = r.get("area_um2", float("nan"))
    return [
        ("C1 cut 读数命中绝对期望（宽=2H、积=2H×带长）",
         bool(r.get("ok")) and abs(w - exp_w) < 1e-3
         and abs(ar - exp_area) / exp_area < 1e-3,
         f"读得宽={w:.6f}µm vs 期望 {exp_w:.3f}µm；积={ar:.3f} vs 期望 {exp_area:.3f} µm²"
         f"（带内 {band.size} 行，斜率 {K*1e-6:.1f}K/µm）"),
        ("C2 布尔版与 cut 版之差 ≤ dx（N3③ 口径移动的上界）",
         bool(r.get("ok")) and abs(r["cut_minus_bool_um"]) * 1e-6 <= dx + 1e-15,
         f"cut−布尔={r.get('cut_minus_bool_um', float('nan')):.3f}µm，dx={dx*1e6:.3f}µm"),
        ("C3 窗过窄必须报红（控件的控件）",
         (not rw.get("ok")) or rw.get("touched", 0) > 0,
         f"±0.5r_b 窗：ok={rw.get('ok')} touched={rw.get('touched')} "
         f"why={str(rw.get('why', ''))[:70]}"),
        ("C4 exclude 必须真的减少有熔行数（R2 语义可失败）",
         bool(r2.get("ok")) and r2["rows_hot"] < r["rows_hot"],
         f"R2 有熔行={r2.get('rows_hot')} vs R1={r.get('rows_hot')}"
         f"（掩膜覆盖 {int(excl.any(axis=0).sum())} 行）"),
    ]


def budget_gate(stage, measured_per_track_s, n_tracks_max):
    """N3⑤：全量放行必须拿**实测**每道墙钟外推 ΣN 与预算比；没给实测就不放行。"""
    if stage != "full":
        return True, f"stage={stage} 不需全量预算闸"
    if not measured_per_track_s:
        return False, ("stage=full 要求 --measured-per-track（pilot 实测每道均摊墙钟 [s]）："
                      "未登记实测代价不许开 18 道逐 N（N3⑤）")
    total = measured_per_track_s * sum(range(1, n_tracks_max + 1))
    return total <= BUDGET_S, (f"每道均摊 {measured_per_track_s:.1f}s × ΣN=1..{n_tracks_max}"
                               f"={sum(range(1, n_tracks_max + 1))} ⇒ {total:.0f}s "
                               f"vs 预算 {BUDGET_S:.0f}s")


def gap_bias(sub, prog, sub_runs, fx):
    """J10：跳段（关束）在模型里多长、在 CSV 里多长，给比值。"""
    cum = np.asarray(prog.cumulative_length)
    tc = np.asarray(sub["columns"])[:, COL_TIME]
    arc_on = float(sum(cum[b] - cum[a] for (a, b) in sub_runs))
    arc_off = float(cum[-1]) - arc_on
    if len(sub_runs) < 2:
        return dict(n_gap=0, t_model=0.0, t_csv=0.0, ratio=float("nan"),
                    note="N=1 无道间间隙 ⇒ 比值无定义")
    t_csv = float(tc[sub_runs[-1][0]] - tc[sub_runs[0][1]])
    t_model = arc_off / (fx["scan_speed_mm_s"] * 1e-3)
    return dict(n_gap=len(sub_runs) - 1, t_model=t_model, t_csv=t_csv,
                ratio=t_model / max(t_csv, 1e-15),
                note=f"跳段弧长={arc_off*1e6:.1f}µm ⇒ 模型间隙={t_model*1e3:.3f}ms，"
                     f"CSV 间隙={t_csv*1e3:.3f}ms（{len(sub_runs)-1} 个）⇒ 比值="
                     f"{t_model/max(t_csv,1e-15):.3f}；<1＝冷却窗口偏短⇒后段道预热偏高")


def toplane_readability(solid, xax, yax, top_idx, fx, sub, sub_runs, r_b, dx,
                        *, ring_shrink=1.0, x_center=None):
    """G5 的**判据本体**（单独成函数只为让控件面扰动同一条式子，不留副本）。

    返回顶面空节点的归属（设计域外的 ``ceil`` 过延伸圈 vs 圈外游荡）与**读数足迹**
    （末道 ±(3r_b＋dx)×该道 y 带，多留的一列给线性交叉）的满布情况。
    ``ring_shrink``/``x_center`` 是**控件旋钮**，只给 ``am_a3_preflight_controls.py``
    把判据推向已知失败点用；正常调用留缺省。
    """
    lx, ly = fx["domain_um"][0] * 1e-6 * ring_shrink, \
        fx["domain_um"][1] * 1e-6 * ring_shrink
    eps = dx * 1e-9
    out_ring = ((xax > lx + eps) | (xax < -eps))[:, None] \
        | ((yax > ly + eps) | (yax < -eps))[None, :]
    void_top = ~solid[:, :, top_idx]
    arr = np.asarray(sub["columns"])
    gy0, gy1 = float(arr[sub_runs[-1][0], 1] * 1e-6), float(arr[sub_runs[-1][1], 1] * 1e-6)
    gx0, gx1 = float(arr[sub_runs[-1][0], 0] * 1e-6), float(arr[sub_runs[-1][1], 0] * 1e-6)
    half = 3.0 * r_b + dx
    xc = 0.5 * (gx0 + gx1) if x_center is None else x_center
    fi = np.where(np.abs(xax - xc) <= half)[0]
    fj = np.where((yax >= min(gy0, gy1) - dx) & (yax <= max(gy0, gy1) + dx))[0]
    foot = solid[np.ix_(fi, fj, [top_idx])] if fi.size and fj.size else solid[:0, :0, :0]
    return dict(top_z_ok=int(top_idx) >= 1, n_top=int(solid[:, :, top_idx].sum()),
                n_top_all=int(solid[:, :, top_idx].size),
                frac_top=float(solid[:, :, top_idx].mean()),
                frac_vol=float(solid.mean()),
                n_void=int(void_top.sum()),
                n_ring_void=int((void_top & out_ring).sum()),
                n_stray=int((void_top & ~out_ring).sum()),
                half_um=half * 1e6, y_lo=min(gy0, gy1), y_hi=max(gy0, gy1),
                n_foot=int(foot.size), n_foot_solid=int(foot.sum()))


# smoke 档的**软闸名单**：这两条判据的对象是形态分辨率，粗网格（dx=r_b）按构造不满足；
# 对 stage=smoke 只报告不求解资格（见 run_case 里的 [SOFT] 分支），pilot／full 仍按硬闸计。
SOFT_FOR_SMOKE = ("G6 每步激光移动 ≤ 0.5·r_b（时间采样充分）",
                  "J6 形态档 dx ≤ r_b/2")


def run_case(geom, xax, yax, zax, top_z, sub, sub_runs, prog, plan, fx, r_b, alpha,
             heat_scale, arm, stage, n_tracks, strategy, T_iso_vals, n_all_runs, cap,
             ic_probe=False):
    """一个 (N, 臂) 的预检闸门＋（非 preflight 时）求解与读数。所有数字都来自现算。"""
    dx = float(geom.spacing)
    n_steps, t_exp, dt_target, dt_rk2, alpha0 = schedule(geom, prog, plan)
    nvox = int(geom.sdf.size)
    n_frames, stride, cadence, cadence_ok = frame_plan(n_steps, t_exp, nvox)
    solid = np.asarray(solid_mask(geom.sdf)) > 0.5
    top_idx = int(np.max(np.where(solid.any(axis=(0, 1)))[0]))
    gb = gap_bias(sub, prog, sub_runs, fx)
    g = []

    def add(tag, ok, detail):
        g.append((tag, bool(ok), detail))

    add("G1 截断后开光段数 == N", len(sub_runs) == n_tracks,
        f"实得 {len(sub_runs)}（全表 {n_all_runs} 段），截断行={int(sub['columns'].shape[0])}")
    pw_knots = np.unique(np.asarray(prog.power))
    duty = float(np.mean(np.asarray(gate_from_power(prog.sample_steps(256)[1],
                                                    fx["laser_power_W"]))))
    add("G2 折点功率列只取 0 或名义功率",
        bool(np.all((pw_knots < 1e-12)
                    | (np.abs(pw_knots - fx["laser_power_W"]) < 1e-9))),
        f"唯一值={np.round(pw_knots, 4).tolist()}；弧长 256 步门控均值={duty:.4f}"
        f"（关束占空比={1-duty:.4f}）")
    cum = np.asarray(prog.cumulative_length)
    tc = np.asarray(sub["columns"])[:, COL_TIME]
    spd = [float(cum[b] - cum[a]) / float(tc[b] - tc[a]) * 1e3
           for (a, b) in sub_runs if float(tc[b] - tc[a]) > 0]
    dev = max(abs(s - fx["scan_speed_mm_s"]) / fx["scan_speed_mm_s"] for s in spd)
    add("G3 每道开光段弧长/CSV 时间在名义速度 ±3% 内", dev < 0.03,
        f"{len(spd)} 段速度 min={min(spd):.1f} max={max(spd):.1f} mm/s，最大偏离 "
        f"{dev*100:.2f}%（名义 {fx['scan_speed_mm_s']:.0f}；坐标是整数 µm ⇒ 偏离含量化噪声）")
    dp_probe = r_b * 1.2                  # ＝生产 :709 在 strict+integrated 下的轴向 σ
    z_beam = float(np.asarray(prog.sample_steps(1)[0])[0, 2])
    e_surf = expected_capture(top_z, dx, dp_probe, z_beam)
    e_deep = expected_capture(top_z, dx, dp_probe, z_beam - 3.0 * r_b)
    d_surf = abs(cap["surface_lo"] - e_surf)
    d_deep = abs(cap["deep_lo"] - e_deep)
    add("G4a 截获份额命中独立解析期望（两个点阵相位各一次）",
        d_surf <= 1e-4 and d_deep <= 1e-4,
        f"顶面 实测={cap['surface_lo']:.6f} vs 解析={e_surf:.6f}（差 {d_surf:.2e}）；"
        f"下移 3r_b 实测={cap['deep_lo']:.6f} vs 解析={e_deep:.6f}（差 {d_deep:.2e}）；"
        f"束面 z={z_beam*1e6:.3f}µm 顶面节点 z={top_z*1e6:.3f}µm（设计面 z=0 ⇒ 束面比"
        f"设计面低 {abs(z_beam)*1e6:.3f}µm＝半个体素以内的相位差）")
    add("G4b 亏损**只**来自跨顶面的轴向半损失（深度对照 ≥0.99）",
        cap["deep_lo"] >= 0.99 and cap["surface_lo"] < 1.0,
        f"下移 3r_b Σfrac=[{cap['deep_lo']:.6f},{cap['deep_hi']:.6f}]，顶面="
        f"[{cap['surface_lo']:.6f},{cap['surface_hi']:.6f}] ⇒ 顶面亏损 "
        f"{100*(1-cap['surface_lo']):.2f}% 且与面内贴边无关；norm 臂 heat_scale="
        f"{1.0/max(cap['surface_lo'], 1e-9):.6f} 据此把 Σ注入补成 α·P（J11）")
    add("G4c 面内平移 +2mm 控件必须≈0（这条闸门本身可失败）",
        cap["outside_hi"] <= 1e-3,
        f"控件 Σfrac=[{cap['outside_lo']:.6f},{cap['outside_hi']:.6f}]（若探针漏掉实体"
        f"掩膜，顶面会读成网格截断值≈0.80 而非 {cap['surface_lo']:.3f} ⇒ G4a 报红）")
    # G5：原判据写作「顶层满布」＝要求顶面 nx·ny 全实体，这条**任何正确实现都过不了**：
    # ``_geometry._grid_axes`` 按 ``ceil((hi−lo)/dx)`` 铺点，域面内 3000×2500µm 不是
    # dx=21.25µm 的整数倍 ⇒ 点阵在 +x/+y 侧各多出一圈**设计域之外**的节点，它们按定义就是
    # 空的。预检把判据改成它本来该问的两件事：①顶面空节点**全部**落在这一圈里（环上确实
    # 有空节点⇒这条不是空集上的真），②**读数足迹**（末道 ±window_r·r_b × 该道 y 带，再各
    # 留一列给线性交叉）满布。
    tp = toplane_readability(solid, xax, yax, top_idx, fx, sub, sub_runs, r_b, dx)
    add("G5 顶面可读：空节点只在设计域外的 ceil 圈上，且读数足迹满布",
        tp["top_z_ok"] and tp["n_stray"] == 0 and tp["n_ring_void"] > 0
        and tp["n_foot"] == tp["n_foot_solid"],
        f"shape={geom.sdf.shape} 顶面 z={top_z*1e6:.3f}µm 顶面实体 "
        f"{tp['n_top']}/{tp['n_top_all']}={tp['frac_top']:.4f}（全体素体积占比另报 "
        f"{tp['frac_vol']:.4f}）；顶面空节点={tp['n_void']}，其中落在 ceil 圈上="
        f"{tp['n_ring_void']}（>0 才说明这条判据不是空集上的真），圈外游荡空节点="
        f"{tp['n_stray']}；读数足迹 ±{tp['half_um']:.2f}µm×y∈[{tp['y_lo']*1e6:.1f},"
        f"{tp['y_hi']*1e6:.1f}]µm 共 {tp['n_foot']} 格，实体 {tp['n_foot_solid']} 格")
    step_move = float(prog.total_length()) / n_steps
    add("G6 每步激光移动 ≤ 0.5·r_b（时间采样充分）", step_move <= 0.5 * r_b,
        f"每步 {step_move*1e6:.2f}µm={step_move/r_b:.3f}r_b；n_steps={n_steps} "
        f"曝光={t_exp*1e3:.3f}ms dt={dt_target*1e6:.4f}µs（稳定限 {dt_rk2*1e6:.4f}µs，"
        f"α₀={alpha0:.4e}m²/s）")
    add("G7 帧节律 ≤ 正文的 0.149ms 且未被缓存收紧（J4）",
        cadence_ok and cadence <= FRAME_CADENCE_S,
        f"请求帧数={n_frames} stride={stride} ⇒ 节律={cadence*1e6:.1f}µs（上限 "
        f"{FRAME_CADENCE_S*1e6:.0f}µs）；缓存上限 {FRAME_MEM_CAP/1e9:.0f}GB 折合 "
        f"{int(FRAME_MEM_CAP//(nvox*8))} 帧")
    add("J6 形态档 dx ≤ r_b/2", dx <= r_b / 2.0 + 1e-15,
        f"dx={dx*1e6:.3f}µm vs r_b/2={r_b*1e6/2:.3f}µm（不满足⇒只作接线烟测，"
        f"不产出形态数字）")
    add("J11 本臂 heat_scale 有限且与 G4 份额同向",
        math.isfinite(heat_scale) and heat_scale > 0.0,
        f"heat_scale={heat_scale:.6f}（arm={arm}；norm 臂＝1/G4 份额，raw 臂＝1.0 ⇒ "
        f"只投 {100*cap['surface_lo']:.1f}% 的 αP）")
    print(f"[{stage}/{arm}] {strategy} N={n_tracks} dx={dx*1e6:.3f}µm α={alpha} "
          f"nvox={nvox} n_steps={n_steps} voxel_steps={nvox*n_steps:.3e} "
          f"path={float(prog.total_length())*1e3:.3f}mm 折点={int(prog.points.shape[0])} "
          f"帧={n_frames}")
    # 本文件头 J6 那条口径（:41）写的是「dx=r_b 只用于接线烟测，那一条**不产出**任何形态数字」，
    # 但首版把 J6／G6 当**硬闸**用 ⇒ smoke 档按构造永远走不到求解（2026-10-10 14:09 实测：三条道次
    # 全部 `SOLVER_NOT_RUNNABLE`，零次求解、零条墙钟）。这两条判据的对象是**形态分辨率**（网格够不够细、
    # 每步激光移动够不够小），粗网格接线档对它们按构造不满足 ⇒ 对 stage=smoke 只作 [SOFT] 报告、
    # 不入 all_pass，并在下方禁用形态数字；pilot／full 仍是硬闸。这是修判据的**适用面**，不是放宽判据。
    soft_hit = []
    for tag, ok, detail in g:
        soft = stage == "smoke" and tag in SOFT_FOR_SMOKE
        if soft:
            soft_hit.append(tag)
        print(f"  [{'SOFT' if soft else ('PASS' if ok else 'FAIL')}] {tag} — {detail}")
    if stage == "smoke":
        miss = [t for t in SOFT_FOR_SMOKE if t not in soft_hit]
        assert not miss, f"smoke 软闸名单没找到这些闸门 ⇒ 名单与标签漂移：{miss}"
    print(f"  [INFO] G3b 道间间隙（J10，只报不判）— {gb['note']}")
    all_pass = all(ok for tag, ok, _ in g if tag not in soft_hit)
    if stage == "preflight":
        return dict(all_pass=all_pass, vox=nvox, n_steps=n_steps, n_gates=len(g))
    if not all_pass:
        print("  SOLVER_NOT_RUNNABLE = 预检有闸未过 ⇒ 本次不解（先修夹具/口径）")
        return dict(all_pass=False)
    t0 = time.perf_counter()
    th = solve_enthalpy_thermal(geometry=geom, process=plan,
                                params=solver_params(prog, n_steps, alpha, heat_scale,
                                                    fx["T0_C"] + 273.15, n_frames))
    th.peak_temperature.block_until_ready()
    if th.temperature_evolution is not None:
        th.temperature_evolution.block_until_ready()
    wall = time.perf_counter() - t0
    ic_k = fx["T0_C"] + 273.15
    _mc = get_material(MATERIAL)
    t_amb = float(_mc.T_ambient)
    t_sol_ref = float(_mc.T_solidus)
    pk = np.asarray(th.peak_temperature)
    nfr = None if th.temperature_evolution is None else np.asarray(
        th.temperature_evolution).shape
    print(f"  解出：wall={wall:.2f}s 帧形={nfr} min(peak)={float(pk.min()):.3f}K"
          f"（夹具 IC={ic_k:.2f}K，材料卡 T_ambient={t_amb:.2f}K）")
    # G8 把首版那条「若为 300.000K 则 IC 未生效」拆成两件**不同的**事。首版注释以为把 IC 走
    # boundary_conditions 就能绕开"被顶成 300K"——2026-10-10 14:2x 实测（/tmp/a3_ic_probe_try2.out）
    # 推翻了这个说法：``temperature_of_enthalpy`` 的二分下界写死 ``T_lo=T_amb``
    # （src/amforge/thermal_enthalpy.py:130），**任何 H≤0 的态都读回 T_amb**——293.15K→H=−3.278057e7
    # →读回 300.0000K（+6.850K），而 300/305/1563K 的往返误差都是 0.000e+00（正对照）。
    # ⇒ 正向映射接受 T<T_amb、逆向不接受＝静默的单侧不对称。另外 ``peak_temperature`` 是
    # **逐体素峰值场**（:32、:974 ``peak=jnp.maximum(peak,Tn)``，init=0），不是逐帧峰值 ⇒ 判据
    # 该问"冷体素的峰值是否等于**可表示的**初温"。
    ok8a = ic_k >= t_amb - 1e-9
    # G8b 取**实体掩膜**上的峰值最小值，不取全场：气相体素的 H0 恒为 0 ⇒ 它们的峰值就是 T_amb，
    # 一旦 ic > T_amb（E/F 斜率腿）全场 min(peak) 读到的是气相壳而不是初温 ⇒ 判据会假红。
    # 2026-10-10 14:4x 实测：--amb-k 293.15 --t0-c 26.85 时全场 min(peak)=293.15K（气相）、
    # 实体 min(peak)=300.00K（正是声明的 ic）⇒ 按实体取才是「IC 真进了场」这条问句。
    pk_solid = float(pk[solid].min()) if bool(solid.any()) else float("nan")
    ok8b = abs(pk_solid - max(ic_k, t_amb)) <= 0.1
    gap_pct = (t_amb - ic_k) / max(t_sol_ref - t_amb, 1e-9) * 100
    tail8a = ("⇒ 实际跑的初温被逆向映射顶成 T_amb，不是夹具声明的 T0" if not ok8a
              else "⇒ 初温可表示，跑的正是夹具声明的 T0")
    for tag, ok8, detail in (
            ("G8a 夹具 IC 在焓模型里可表示（ic ≥ 材料卡 T_ambient）", ok8a,
             f"ic={ic_k:.4f}K vs T_amb={t_amb:.4f}K ⇒ 缺口 {t_amb - ic_k:+.4f}K"
             f"；占「升到 T_sol 所需显焓」的 {gap_pct:.4f}%（T_sol={t_sol_ref:.1f}K）"
             f"{tail8a}"),
            ("G8b 冷体素峰值＝可表示初温（IC 真进了场，未被顶替）", ok8b,
             f"min(peak 实体)={pk_solid:.4f}K vs max(ic,T_amb)={max(ic_k, t_amb):.4f}K"
             f"（全场 min(peak)={float(pk.min()):.4f}K，含气相壳⇒只报不判；tol=0.1K；"
             f"帧形={nfr}）")):
        print(f"  [{'PASS' if ok8 else 'FAIL'}] {tag} — {detail}")
    ic_keep = False
    if not (ok8a and ok8b):
        if not ic_probe:
            print("  READOUT_FAIL = IC 不可表示（G8a）或被顶替（G8b）⇒ 本条读数作废（不比对靶值）"
                  "；这是**夹具保真**问题，不是求解崩溃")
            return dict(all_pass=False, wall=wall, ic_ok=False)
        # 敏感度探针档：G8 仍然是 FAIL（all_pass 记 False，靶值比对照常打印但不作评分），
        # 只把求解与形态读数跑完，用来量化「逆向映射把 IC 顶成 T_amb」对形态的影响有多大。
        ic_keep = True
        print("  IC_PROBE_KEEP = G8 未过但读数保留 ⇒ 本条**不是**基准读数（all_pass=False），"
              "只用于 #51 的敏感度比对；本档若产出形态数字，每行都带 [PROBE] 标记"
              f"（当前 stage={stage}：smoke 档按 J6 口径不印数字）")
    if ic_keep:
        print("  以下形态读数＝探针档（IC 已被逆向映射顶替），**不得**引作 A3 基准评分。")
    ro = readout_from_frames(th.temperature_evolution, xax, yax, zax, top_z, sub,
                             sub_runs, prog, n_steps, n_frames, r_b, T_iso_vals)
    if not ro.get("ok"):
        print(f"  READOUT_FAIL = {ro.get('why')}")
        return dict(all_pass=False, wall=wall)
    print(f"  窗口：t0={ro['t0']} stride={ro['stride']} 窗内帧={ro['n_win']} "
          f"（R2 前一帧 k={ro['k_before']}）顶层 iz={ro['iz']} "
          f"道心 x={ro['x_track']*1e6:.1f}µm y∈[{ro['y_lo']*1e6:.0f},"
          f"{ro['y_hi']*1e6:.0f}]µm")
    targs = {q: targets(n_tracks, q, strategy) for q in BARS}
    for iso_key in T_iso_vals:
        for ver in ("", "_R2"):
            rr = ro[iso_key + ver]
            if not rr.get("ok"):
                print(f"    {iso_key}{ver}: READOUT_FAIL = {rr['why']}")
                continue
            if stage == "smoke":
                # J6 未满足 ⇒ 本档**不产出任何形态数字**（本文件头 :41 的自定口径）：这里只报
                # 「读数核接得上」的布尔与计数，宽/积的具体值一律不印，避免粗网格数字被日后误引。
                print(f"    {iso_key}{ver}: 接线=True 宽有限={math.isfinite(rr['width_um'])} "
                      f"积有限={math.isfinite(rr['area_um2'])} 行数={rr['n_rows']} "
                      f"有熔={rr['rows_hot']} 触边={rr['touched']}"
                      f"（smoke 档不印形态数字）")
                continue
            line = (f"    {'[PROBE] ' if ic_keep else ''}{iso_key}{ver}: "
                    f"宽={rr['width_um']:.2f}µm "
                    f"(布尔 {rr['width_bool_um']:.2f}，差 {rr['cut_minus_bool_um']:+.2f}) "
                    f"积={rr['area_um2']:.0f}µm² (布尔 {rr['area_bool_um2']:.0f}) "
                    f"行数={rr['n_rows']} 有熔={rr['rows_hot']} 触边={rr['touched']}")
            if iso_key == "solidus" and ver == "" and ic_keep:
                # 探针档不比对靶值：靶行里的「偏差 x% ⇒ 达标」字样会被日后误引（#51 红线）。
                line += " | 靶值比对：跳过（IC_PROBE 档，all_pass 已记 False）"
            if iso_key == "solidus" and ver == "" and not ic_keep:
                for q, bar in BARS.items():
                    tv = targs[q].get("all")
                    if not tv:
                        line += f" | 靶 {q}: 无 Set2 值（不比对）"
                        continue
                    d = abs(rr[q] - tv["median"]) / tv["median"] * 100.0
                    to = targs[q].get("no_outlier")
                    d2 = (abs(rr[q] - to["median"]) / to["median"] * 100.0) if to else None
                    line += (f" | 靶 {q}: 中位 {tv['median']:.1f}（n={tv['n']}，极差 "
                             f"{tv['spread_pct']:.1f}%，{tv['lo']:.0f}–{tv['hi']:.0f}）"
                             f"⇒ 偏差 {d:.1f}%（≤{bar[0]:.0f}% 达标／>{bar[1]:.0f}% 未达）")
                    if d2 is not None:
                        line += f"，去离群版中位 {to['median']:.1f} ⇒ 偏差 {d2:.1f}%"
            print(line)
    # G9＝触边闸。首版只把 touched 印在 SOLVE_VERDICT 里、**不入** all_pass ⇒ 被试片边界截断的
    # 宽／积照样进了趋势评分（2026-10-10 14:4x pilot 实测：N=2 触边 17/56 行 ⇒ 积 116532µm²，
    # N=3 触边 33/55 行 ⇒ 积反而降到 55844µm²（道次更多、能量更多而形态更小＝物理不可能，
    # 唯一的解释是窗口被边界切走）⇒ 这两条读数是**下界**不是测量值，必须可红）。
    touch = max(int(ro[k].get("touched", 0)) for k in T_iso_vals for v in ("", "_R2"))
    n_rows = int(ro["solidus"].get("n_rows", 0))
    ok9 = touch == 0
    verdict = "OK" if ok9 else "READOUT_TOUCHED_EDGE"
    print(f"  [{'PASS' if ok9 else 'FAIL'}] G9 读数窗未被试片边界切割（触边行数==0） — "
          f"max touched={touch}／窗口行数={n_rows}（四组读数 solidus／solidus_R2／liquidus／"
          f"liquidus_R2 里取最大）⇒ "
          + ("读数完整，可进趋势评分"
             if ok9 else
             "熔池被试片边界截断 ⇒ 宽／积只是**下界**，本条不得进 18 道趋势评分；"
             "这是 #30「加大试片」裁决的直接证据"))
    print(f"  SOLVE_VERDICT = {verdict} wall={wall:.2f}s arm={arm}")
    return dict(all_pass=(not ic_keep) and ok9, wall=wall, vox=nvox, n_steps=n_steps,
                ic_ok=bool(ok8a and ok8b), touched=touch,
                readout=(None if stage == "smoke" else ro.get("solidus")),
                targets=targs, verdict=verdict)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="preflight",
                    choices=["preflight", "smoke", "pilot", "full"])
    ap.add_argument("--tracks", default="1,2,3")
    ap.add_argument("--dx", type=float, default=None,
                    help="µm；缺省＝形态档 r_b/2（smoke 档取 r_b）")
    ap.add_argument("--alpha", type=float, default=None,
                    help="缺省＝夹具 alpha_calibrated（J2 的复现档）")
    ap.add_argument("--strategy", default="C", choices=["C", "D"])
    ap.add_argument("--arms", default="norm", choices=["norm", "raw", "both"],
                    help="J11 的能量臂")
    ap.add_argument("--measured-per-track", type=float, default=None,
                    help="pilot 实测每道均摊墙钟 [s]；stage=full 必需（N3⑤）")
    ap.add_argument("--amb-k", type=float, default=None,
                    help="覆写材料卡 T_ambient [K]（夹具保真：G8a 要求基准 T0 在焓模型里可表示，"
                         "而逆向映射把下界写死在 T_ambient）；缺省＝材料卡原值")
    ap.add_argument("--t0-c", type=float, default=None,
                    help="覆写夹具声明的初温 T0 [°C]（只动 IC／预热这一条链：fx['T0_C'] 进 "
                         "plan_for 的预热、solver_params 的 IC 与 G8a 的 ic_k，物性参数一律不动）。"
                         "用于在**固定环境温度**下单独测 d(形态)/d(IC) 的斜率，两腿都可过 G8a。")
    ap.add_argument("--ic-probe", action="store_true",
                    help="**敏感度探针档**：G8a 判 IC 不可表示时按原口径作废本条基准读数；本开关"
                         "只让求解与形态读数**继续跑完**，用于量化「IC 被顶替」对形态的影响。"
                         "该档产出的任何宽度/面积都**不得**进基准比对（all_pass 仍为 False）。")
    a = ap.parse_args()
    global MATERIAL
    if a.amb_k is not None:
        import dataclasses
        from amforge.materials import AM_MATERIALS
        base = get_material(MATERIAL)
        base_name = MATERIAL                      # 覆写会改全局 MATERIAL，基卡名要先钉住
        key = f"{MATERIAL}_amb{a.amb_k:g}K"
        AM_MATERIALS[key] = base.replace(T_ambient=a.amb_k)
        changed = [f.name for f in dataclasses.fields(base)
                   if getattr(base, f.name) != getattr(AM_MATERIALS[key], f.name)]
        assert changed == ["T_ambient"], f"覆写动了不止一个字段：{changed}"
        MATERIAL = key
        print(f"材料卡覆写：注册名 {key}＝基卡 {base_name} 只换 T_ambient "
              f"{float(base.T_ambient):.4f}K→{a.amb_k:.4f}K"
              f"（{len(dataclasses.fields(base))} 字段逐字段现比，改动集＝{changed} "
              f"⇒ 其余 {len(dataclasses.fields(base)) - 1} 个含 T_sol/T_liq/ρ/cp/k/L 全同）")
    mat = get_material(MATERIAL)
    fx = load_fixture()
    if a.t0_c is not None:
        t0_base = fx["T0_C"]
        fx["T0_C"] = float(a.t0_c)
        print(f"夹具 IC 覆写：T0_C {t0_base:.4f}°C→{fx['T0_C']:.4f}°C"
              f"（ic={(fx['T0_C'] + 273.15):.4f}K；只改初温链＝plan_for 预热＋solver_params IC"
              f"＋G8a 的 ic_k，其余夹具键与物性参数不动）")
    r_b = beam_radius_m()
    alpha = a.alpha if a.alpha is not None else fx["alpha_calibrated"]
    dx_um = a.dx if a.dx is not None else (r_b * 1e6 if a.stage == "smoke"
                                          else r_b * 1e6 / 2.0)
    tracks = [int(v) for v in a.tracks.split(",")]
    if a.stage == "full":
        tracks = list(range(1, fx["tracks"] + 1))
    arms = ["norm", "raw"] if a.arms == "both" else [a.arms]
    T_iso_vals = dict(solidus=float(mat.T_solidus), liquidus=float(mat.T_liquidus))
    print(f"夹具现取：domain={fx['domain_um']}µm P={fx['laser_power_W']}W "
          f"v={fx['scan_speed_mm_s']}mm/s hatch={fx['hatch_um']}µm "
          f"r_b={r_b*1e6:.3f}µm（D4σ={fx['D4sigma_um']}µm）α 两档="
          f"{fx['alpha_calibrated']}/{fx['alpha_ours']} T0={fx['T0_C']}°C")
    print(f"材料卡现取：T_solidus={float(mat.T_solidus)}K T_liquidus={float(mat.T_liquidus)}K "
          f"T_ambient={float(mat.T_ambient)}K absorptivity={float(mat.absorptivity)} "
          f"rho/cp/k={float(mat.rho_solid)}/{float(mat.cp_solid)}/{float(mat.k_solid)}")
    print(f"本轮选用（跑前定死）：stage={a.stage} strategy={a.strategy} tracks={tracks} "
          f"α={alpha}（J2）dx={dx_um:g}µm arms={arms}（J11）"
          f"主阈值=T_solidus（J3）主时刻=末道窗内帧最大值（J4）主判据臂=norm")
    ok, why = budget_gate(a.stage, a.measured_per_track, fx["tracks"])
    print(f"N3⑤ 预算闸：{'放行' if ok else '不放行'} — {why}")
    if not ok:
        print("STAGE_DONE = aborted(预算闸)")
        return
    name = ("scanStrategyConverging.csv" if a.strategy == "C"
            else "scanStrategyDiverging.csv")
    table, cols, all_runs = read_strategy(name)
    print(f"轨迹表 {name}：列={table['header']} 行数={cols.shape[0]} "
          f"开光段={len(all_runs)}（夹具 tracks={fx['tracks']}）")
    st = selftest_readout(r_b, float(mat.T_solidus))
    print("读数核合成控件（证明读数本身能失败，preflight 也跑）：")
    for tag, rok, detail in st:
        print(f"  [{'PASS' if rok else 'FAIL'}] {tag} — {detail}")
    if not all(ok_ for _, ok_, _ in st):
        print("STAGE_DONE = aborted(读数核控件未过)")
        return
    geom = plate_geometry(dx_um * 1e-6, fx["domain_um"])
    xax, yax, zax = axes_of(geom)
    solid = np.asarray(solid_mask(geom.sdf)) > 0.5
    top_z = float(zax[int(np.max(np.where(solid.any(axis=(0, 1)))[0]))])
    plan = plan_for(fx, r_b, alpha, fx["T0_C"] + 273.15)
    results = []
    for n in tracks:
        sub, prog = program_upto(cols, all_runs, n, top_z)
        sub_runs = power_runs(np.asarray(sub["columns"])[:, COL_POWER])
        n_steps = schedule(geom, prog, plan)[0]
        cap = capture_probes(geom, prog, n_steps, r_b)
        for arm in arms:
            hs = 1.0 if arm == "raw" else 1.0 / max(cap["surface_lo"], 1e-9)
            results.append((n, arm, run_case(geom, xax, yax, zax, top_z, sub, sub_runs,
                                            prog, plan, fx, r_b, alpha, hs, arm,
                                            a.stage, n, a.strategy, T_iso_vals,
                                            len(all_runs), cap,
                                            ic_probe=a.ic_probe)))
    walls = [r["wall"] for _, _, r in results if "wall" in r]
    if walls:
        worst_n = max(n for n, _, r in results if "wall" in r)
        per_track = max(walls) / worst_n
        tot = per_track * sum(range(1, fx["tracks"] + 1))
        print(f"代价实测（可行性代价，非产品性能声明）：最大 N={worst_n} 墙钟 "
              f"{max(walls):.1f}s ⇒ 每道均摊 {per_track:.1f}s；18 道逐 N 之和 ≈ {tot:.0f}s"
              f"（预算 {BUDGET_S:.0f}s）⇒ {'在预算内' if tot <= BUDGET_S else '超预算，full 不放行'}")
    for arm in arms:
        row = [(n, r["readout"]["width_um"], r["readout"]["area_um2"])
               for n, ar_, r in results
               if ar_ == arm and isinstance(r.get("readout"), dict)
               and r["readout"].get("ok")]
        if len(row) > 1:
            ws = [w for _, w, _ in row]
            print(f"趋势（{arm} 臂，宽度 µm）："
                  f"{[(int(n), round(w, 1)) for n, w, _ in row]} "
                  f"非降={all(b >= a0 * 0.98 for a0, b in zip(ws, ws[1:]))} "
                  f"末/首={ws[-1]/ws[0]:.2f}（J7：趋势才是主判据）")
    print(f"STAGE_DONE = {a.stage} 道次={tracks} 臂={arms} "
          f"硬闸全过={all(r.get('all_pass') for _, _, r in results)}"
          + (f" 软闸={len(SOFT_FOR_SMOKE)}条/道次（J6／G6 在 dx=r_b 按构造不过⇒只作 [SOFT] 报告，"
             f"本档不产出形态数字）" if a.stage == "smoke" else ""))
    if a.stage == "preflight":
        print("PREFLIGHT_VERDICT = 见上；本件未解，零条形态数字")


if __name__ == "__main__":
    main()
