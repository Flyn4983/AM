"""路径程序（#18）：调用方给定的 ``(x, y, z, power)`` 折线 → 静态形状、可微的扫描采样。

商业软件式"给定几何/工艺 ⇒ 正向模拟"的必要一环：此前热源的走位只由工艺参数
（``hatch_spacing``/``layer_thickness``/extent）经 ``_scan_topology`` 的**解析 zigzag**
式推出，调用方无法说"激光按这条路径、这一段功率 0 W（关束）走"。本模块只提供
**路径本身**，不改任何缺省行为：``solve_enthalpy_thermal`` 在
``params["scan_program"]`` 缺省时逐位不变。

三条口径（都是被现有断言逼出来的，不是偏好）：

1. **时间映射仍是弧长↔速度**。曝光时长照旧 ``t_exposure = 总长 / scan_speed``，采样
   归一化弧长 ``g=(t+0.5)/n_steps``（与 ``_build_scan_positions`` 同一中点口径）。
   ⇒ ``test_enthalpy_thermal.py:173`` 断的 ``|∂输出/∂scan_speed| > 0`` 对本路径同样成立；
   若改用 CSV 自带的 time 列直接定时间，速度梯度恒零、那条断言立刻红。CSV 的
   时间列于是只以 :func:`implied_scan_speed` 的**诊断量**形式可见（opt-in）。
2. **功率是比值门控**，不是绝对量：``gate(t) = power(t) / process.laser_power``。
   常功率列（每点都等于 ``laser_power``）时 ``gate`` 逐位＝ ``1.0``（这条由 :meth:`PathProgram.sample`
   的差值混合写法保证，不是"舍入后差不多"）⇒ 缺省档的 ``Q0``/``laser_power`` 表达式**一字不动**；
   而缺省档本身走的是``params["scan_program"]`` 缺省分支，连乘都不乘 ⇒ 输出逐位相同。
   ``power=0`` 即关束；不做任何"平均功率"折算。
3. **纯 jnp**（无 ``np.searchsorted``/``np.interp``），所以能在 ``jax.lax.scan``/``jit``
   里用；``n_steps`` 是静态形状，折点坐标与功率列可求梯度（由
   ``tests/test_scan_program.py`` 实测钉住，不由注释声称）。

与既有实现的差别（为什么不复用）：``diffmech/methods/am/scan_paths.py`` 的
``ScanPath.position_at`` 用 ``np.searchsorted`` ⇒ 不可进 scan；
``amforge.meltpool.polyline_trajectory`` 用 ``jnp.interp`` 且把 ``total_time`` 取成
``float`` ⇒ 只能当行为参考。本模块的取段指标用 ``jnp.sum(cum <= s)``，与
``am_thermal.laser_position`` 同为树内 jnp-only 模板。
"""
from __future__ import annotations

import dataclasses
import math
import os

import jax
import jax.numpy as jnp
import numpy as np

#: 折线至少要两个折点才有"段"可走。
MIN_KNOTS = 2
#: 段长下限：只防零除，不是物理下限（与 ``beam.spot_sigma`` 的 ``1e-12`` 同性质）。
_MIN_SEG_M = 1e-30


def _concretize(*values):
    """把可能处于 trace 的标量逐个固化；任何一个固化失败就返回 ``None``。

    与 ``thermal_enthalpy._concrete`` 同口径：域校验需要 ``bool``，而 ``bool(tracer)``
    会炸，所以"能不能判"这件事本身必须先看能不能固化。
    """
    out = []
    for v in values:
        try:
            out.append(float(v))
        except jax.errors.ConcretizationTypeError:
            return None
    return out


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PathProgram:
    """一条折线程序：``points`` ``(n, 3)`` 位置 [m]，``power`` ``(n,)`` 逐折点激光功率 [W]。

    构造即校验**静态**部分（秩、行数、特征维）；数值域（有限、功率非负、总长 > 0）
    留给调用方在 eager 模式判，因为 trace 下 ``bool(tracer)`` 会炸——求解器里的
    ``require_program`` 就是干这件事的。
    """

    points: jax.Array
    power: jax.Array

    def __post_init__(self):
        pts = jnp.asarray(self.points, dtype=jnp.float64)
        pw = jnp.asarray(self.power, dtype=jnp.float64)
        if pts.ndim == 2 and pts.shape[-1] == 2:            # 允许面内 (n,2)，补 z=0
            pts = jnp.concatenate([pts, jnp.zeros((pts.shape[0], 1))], axis=-1)
        if pts.ndim != 2 or pts.shape[-1] != 3:
            raise ValueError(
                f"scan_program：points 形状须为 (n, 2) 或 (n, 3)，收到 {pts.shape}")
        if pts.shape[0] < MIN_KNOTS:
            raise ValueError(
                f"scan_program：折线至少需要 {MIN_KNOTS} 个折点，收到 {pts.shape[0]} 个")
        if pw.ndim == 0:                                     # 标量＝常功率列
            # 不用 float(pw)：那是 trace 下会炸的具体化，而常功率列正是要能对
            # laser_power 求梯度时走的分支。
            pw = jnp.broadcast_to(pw, (pts.shape[0],))
        if pw.ndim != 1 or pw.shape[0] != pts.shape[0]:
            raise ValueError(
                f"scan_program：power 须为标量或 (n,)={pts.shape[0]}，收到 {pw.shape}")
        object.__setattr__(self, "points", pts)
        object.__setattr__(self, "power", pw)

    # -- 弧长表 --------------------------------------------------------------
    @property
    def segment_lengths(self):
        """相邻折点段长 ``(n-1,)``；重复折点给出**精确 0** 段（采样时由下限兜住）。

        ``sqrt`` 在 0 处导数是无穷的 ⇒ 一个零长段就足以把 NaN 注进整条梯度链，而
        ``jnp.where`` 挡不住它（未选中分支照样被求值，``0 × NaN = NaN``）。所以写成
        双层 ``where``：内层让未选中分支的 ``sqrt`` 跑在 ``1.0`` 上（不产生 NaN），
        外层把该分支的**前值**取回 ``0.0`` ⇒ 前值仍是精确零、梯度是精确零。
        实测见 ``docs/evidence/2026-10-09/``：单层 ``where`` 版 ``∂Σpos/∂points`` 有
        非有限元，本写法全有限。
        """
        d = self.points[1:] - self.points[:-1]
        d2 = jnp.sum(d * d, axis=-1)
        return jnp.where(d2 > 0.0, jnp.sqrt(jnp.where(d2 > 0.0, d2, 1.0)), 0.0)

    @property
    def cumulative_length(self):
        """自起点累计弧长 ``(n,)``，首元素为 0。总长的**唯一出处**是本表的末元素。"""
        return jnp.concatenate(
            [jnp.zeros(1, dtype=self.segment_lengths.dtype),
             jnp.cumsum(self.segment_lengths)])

    def total_length(self):
        """折线总长 [m]（＝解析 zigzag 的 ``path_length`` 的对应物）。"""
        return self.cumulative_length[-1]

    # -- 采样 ----------------------------------------------------------------
    def sample(self, g):
        """归一化弧长 ``g``∈[0,1] 处的 ``(位置 (3,), 功率标量)``。

        段指标用 ``jnp.sum(cum <= s)`` 取（纯 jnp，可在 ``scan``/``vmap`` 内），段内
        份额线性混合 ⇒ 梯度能回流到折点坐标与功率列。``jnp.clip`` 只兜边界与 0 段，
        不改变内部任何一行的取值。
        """
        s = jnp.clip(g, 0.0, 1.0) * self.total_length()
        cum = self.cumulative_length
        last = self.points.shape[0] - 2
        idx = jnp.clip(jnp.sum(cum <= s) - 1, 0, last).astype(jnp.int32)
        seg = self.segment_lengths[idx]
        u = jnp.clip((s - cum[idx]) / jnp.maximum(seg, _MIN_SEG_M), 0.0, 1.0)
        a_p = self.points[idx]
        a_w = self.power[idx]
        # 两式都写成 ``a + u·(b − a)`` 而不是 ``(1−u)·a + u·b``：数学上同一条直线，
        # 但**常值列时逐位精确**（``b − a`` 恰为 ``0.0`` ⇒ 结果恒等于 ``a``，不经过
        # 两次舍入再相加）。这不是洁癖：功率那条因此让 ``gate`` 在常功率列时逐位＝1.0，
        # 位置那条与 ``_build_scan_positions`` 自己的写法 ``x_min + u·x_ext`` 同形
        # ⇒ 无跳段极限下两者可比到逐位（实测见 docs/evidence/2026-10-09/）。
        pos = a_p + u * (self.points[idx + 1] - a_p)
        pw = a_w + u * (self.power[idx + 1] - a_w)
        return pos, pw

    def sample_steps(self, n_steps):
        """按步中点采 ``(n_steps, 3)`` 位置与 ``(n_steps,)`` 功率（静态形状）。"""
        step_g = (jnp.arange(n_steps, dtype=jnp.float64) + 0.5) / jnp.maximum(n_steps, 1)
        pos, pw = jax.vmap(self.sample)(step_g)
        return pos, pw

    def power_gate(self, n_steps, reference_power):
        """功率门控 ``(n_steps,)``（便捷式；口径见 :func:`gate_from_power`）。"""
        _, pw = self.sample_steps(n_steps)
        return gate_from_power(pw, reference_power)

    # -- 构造器 --------------------------------------------------------------
    @classmethod
    def from_arrays(cls, points, power=None):
        """由折点数组构造；``power=None`` ⇒ 常功率列（门控恒 1，纯几何路径）。"""
        pts = jnp.asarray(points, dtype=jnp.float64)
        pw = 1.0 if power is None else power
        return cls(points=pts, power=jnp.asarray(pw, dtype=jnp.float64))

    @classmethod
    def zigzag(cls, *, x_min, y_min, z_min, x_ext, y_ext, z_ext,
               n_layers, n_lines, spacing, layer_height, power):
        """把解析 zigzag 拓扑**展开成折线**（含道间跳段），用于与 ``_scan_topology`` 对照。

        参数一律由调用方从 ``_scan_topology`` 的返回值给出——本模块**不**重写那条
        解析式，否则同一份拓扑就有了两个出处（#24/T3 的单点化口径）。``y_ext``/``z_ext``
        收下但不参与展开（y 由 ``spacing``、z 由 ``layer_height`` 决定），保留它们只为
        让调用方能直接摊平 ``_scan_topology`` 的返回值。``power`` 在这里被固化成**常
        功率列**（构造器走 Python 循环，本就在 trace 之外）；要变功率请用
        :meth:`from_arrays` 显式给列。
        """
        n_l, n_h = int(n_layers), int(n_lines)
        pts, pws = [], []
        for lf in range(n_l):
            z = float(z_min) + (lf + 0.5) * float(layer_height)
            for jf in range(n_h):
                y = float(y_min) + (jf + 0.5) * float(spacing)
                x0, x1 = (float(x_min), float(x_min) + float(x_ext)) if jf % 2 == 0 \
                    else (float(x_min) + float(x_ext), float(x_min))
                if pts:                        # 进入本层/本道前先跳位（跳段上也记功率）
                    pts.append([pts[-1][0], y, z])
                    pws.append(float(power))
                pts.append([x0, y, z])
                pws.append(float(power))
                pts.append([x1, y, z])
                pws.append(float(power))
        return cls(points=jnp.asarray(pts), power=jnp.asarray(pws))


def gate_from_power(power_col, reference_power):
    """功率门控的**唯一出处**：``gate ＝ 采样功率 / 参考功率``。

    ``reference_power`` 是工艺里的 ``laser_power``（均值）。常功率列（每点都等于参考值）
    时逐位＝ ``1.0``（由 :meth:`PathProgram.sample` 的差值混合保证，见其注释），所以
    接了程序但**没有改功率**的算例，能量表达式与缺省档同值。下界 ``_MIN_SEG_M`` 只防
    零除；"参考功率为 0 却给了非零程序功率"是矛盾输入，由 :func:`require_program`
    在 eager 模式报错。
    """
    return (jnp.asarray(power_col, dtype=jnp.float64)
            / jnp.maximum(jnp.asarray(reference_power, dtype=jnp.float64), _MIN_SEG_M))


def require_program(program, *, reference_power, name="scan_program"):
    """eager 校验：静态形状之外**所有数值域前提**都在这里一次性判掉。

    为什么要有这个函数：``PathProgram.__post_init__`` 只查静态部分（秩/行数/特征维），
    而"折点是否有限""总长是否为正""参考功率为 0 却带非零程序功率"这类**数值域**前提
    必须在求解器进入 ``lax.scan`` **之前**判掉，否则会退化成一个 NaN 场＋一句"看起来
    正常"的警告。trace 下三件事都无法固化（``bool(tracer)`` 会炸），本函数按
    :func:`_concretize` 整体跳过——同一算例必然先在 eager 过一次，报错口径仍可达。
    报错文案一律带 ``name`` ⇒ 与求解器其余守卫（``source_model``/``欠采样``/``n_steps``）
    同一条可 grep 的口径。
    """
    if not isinstance(program, PathProgram):
        raise ValueError(
            f"{name}：要接路径程序必须给 amforge.scan_program.PathProgram 实例，"
            f"收到 {type(program).__name__}。")
    if _concretize(jnp.sum(program.points), jnp.sum(program.power),
                   reference_power) is None:
        # trace（jax.grad/jit）下折点/功率/工艺功率是 tracer ⇒ 域校验整体跳过。这不是
        # "放行"：调用方（chain_schedule/GUI 装配链）在同一条算例上必然先过一次 eager，
        # 报错口径因此仍然可达；在这里抛 ConcretizationTypeError 只会把可微链打死。
        return program
    pts = np.asarray(program.points)
    pw = np.asarray(program.power)
    if not np.all(np.isfinite(pts)):
        bad = int(np.argmax(~np.isfinite(pts)))
        raise ValueError(f"{name}：折点坐标含非有限值（首个在下标 {bad}）：{pts[bad].tolist()}")
    if not np.all(np.isfinite(pw)):
        raise ValueError(f"{name}：功率列含非有限值：{pw.tolist()}")
    if float(np.min(pw)) < 0.0:
        raise ValueError(
            f"{name}：功率列不得为负（关束请写 0.0），实得最小值 {float(np.min(pw)):.6g} W")
    total = float(program.total_length())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError(
            f"{name}：折线总长必须是正有限值，实得 {total:.6g} m"
            f"（折点数 {pts.shape[0]}；全零段或重复折点会走到这里）")
    ref = float(np.mean(np.atleast_1d(np.asarray(reference_power, dtype=np.float64))))
    if ref <= 0.0 and float(np.max(pw)) > 0.0:
        raise ValueError(
            f"{name}：门控是**比值** gate=power/laser_power，而工艺 laser_power={ref:.6g} W "
            f"非正、程序功率却有 {float(np.max(pw)):.6g} W ⇒ 门控无定义。请把工艺功率设为"
            f"参考值（例如程序里的最大功率）或把程序功率列改为 0。")
    return program


def read_track_table(path, *, delimiter=","):
    """读一张轨迹表，**所有列都返回**（历史缺陷：只取前两列 ⇒ 功率/时间被静默丢掉）。

    返回 ``{"header": [...], "columns": (m, k) 的 float64 数组, "index": {列号: 列名}}``。
    列选择由调用方做，且**必须**指名列号——不做"看起来像功率"的猜测。表头由"首行是否
    含字母"判定（不收 ``skip_header``：那样就有两个互相矛盾的开关，而实测每张基准表都
    带一行表头）。
    """
    # NIST MDS2-3662 的表头是 ``x (µm),...``，µ 在那些文件里是**非 UTF-8 单字节**
    # （cp1252/latin-1）。按 utf-8 硬解会把列名变成 U+FFFD，而列名正是调用方决定
    # "哪列是功率"的依据 ⇒ 先试 utf-8，失败退 latin-1（后者不会抛，能还原 µ）。
    with open(path, "rb") as fh:
        blob = fh.read()
    try:
        text = blob.decode("utf-8")
    except UnicodeDecodeError:
        text = blob.decode("latin-1")
    raw = text.splitlines()
    header = []
    if raw:
        first = raw[0]
        if any(ch.isalpha() for ch in first):
            header = [c.strip() for c in first.split(delimiter)]
            raw = raw[1:]
    data = jnp.asarray(
        [[float(c.strip()) for c in line.split(delimiter) if c.strip() != ""]
         for line in raw if line.strip() != ""], dtype=jnp.float64)
    if data.size == 0:
        raise ValueError(f"轨迹表 {os.fspath(path)} 里没有数据行")
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    index = {i: (header[i] if i < len(header) else f"col{i}") for i in range(data.shape[1])}
    return {"header": header, "columns": data, "index": index}


def program_from_track_table(table, *, x_col=0, y_col=1, z_col=None, power_col=None,
                             unit=1.0, layer_z=0.0):
    """从 :func:`read_track_table` 的结果取列组成折线程序；``unit`` 做长度换算。

    ``unit`` 缺省 1.0＝**不猜单位**。NIST MDS2-3662 的 ``singleTrack.csv`` 头是
    ``x (µm),y (µm),laser power (W),time (seconds)`` ⇒ 那里要显式传 ``unit=1e-6``。
    功率列缺失时按常功率（门控恒 1）。
    """
    cols = table["columns"]
    need = max(c for c in (x_col, y_col, z_col, power_col) if c is not None)
    if need >= cols.shape[1]:
        raise ValueError(
            f"scan_program：轨迹表只有 {cols.shape[1]} 列，却要求第 {need} 列"
            f"（列名＝{table['index']}）")
    xy = jnp.stack([cols[:, x_col], cols[:, y_col]], axis=-1) * unit
    if z_col is None:
        pts = jnp.concatenate([xy, jnp.full((xy.shape[0], 1), float(layer_z))], axis=-1)
    else:
        pts = jnp.concatenate([xy, cols[:, z_col:z_col + 1] * unit], axis=-1)
    pw = jnp.ones((pts.shape[0],), dtype=jnp.float64) if power_col is None \
        else cols[:, power_col]
    return PathProgram(points=pts, power=pw)


def implied_scan_speed(table, program, *, time_col=None):
    """把轨迹表自带的 **时间列**换算成等效扫描速度 [m/s]（诊断量，不是求解器输入）。

    存在理由：时间轴由 ``弧长/scan_speed`` 决定（模块头第 1 条口径），而基准数据常自带
    一条自己的时间轴。让调用方看得见"要复现那张表就该用这个速度"，但**不**绕过弧长映射。
    ``program`` 已是米制（换算在 :func:`program_from_track_table` 里做过），故本函数
    不再收 ``unit``。
    """
    if time_col is None:
        raise ValueError(
            "scan_program：要读时间列必须指名 time_col（不猜哪列是时间）")
    t = table["columns"][:, time_col]
    dt = float(t[-1] - t[0])
    # 折线的 points 在 program_from_track_table 里已按 unit 换算成米，所以总长**直接**
    # 就是米；这里不再乘 unit（写成 /unit*unit 是恒等的空操作，只会让人以为它有用）。
    length = float(program.total_length())
    if dt <= 0.0:
        raise ValueError(
            f"scan_program：时间列跨度 {dt:.3e}s 非正，推不出扫描速度（表是否未排序？）")
    return length / dt


def track_power_duty(gate):
    """门控占空比（无量纲均值）——只作诊断，用于说明"功率列有多少在关束"。

    权重就是**等权**：:meth:`PathProgram.sample_steps` 取的 ``g`` 是归一化弧长上的等距
    中点，所以这些点在弧长上均匀分布，均值即弧长加权均值。改前签名多收一个 ``program``
    并按 ``segment_lengths``（**折点**口径，长度 ``n−1``）加权，而 ``gate`` 长度是
    ``n_steps`` ⇒ 两者形状不同却经广播静默错位（实测常功率列返回 31.0 而非 1.0）。
    诊断量不该有这种静默口径缝，故把 program 参数去掉。
    """
    g = jnp.asarray(gate)
    if g.ndim != 1:
        raise ValueError(f"scan_program：占空比的门控须是一维 (n_steps,)，收到 {g.shape}")
    return float(jnp.mean(g))
