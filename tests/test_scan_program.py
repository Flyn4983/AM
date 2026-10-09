"""#18 路径程序（``amforge.scan_program``）与求解器接线（``params["scan_program"]``）。

守护点（都是被现有断言或实测逼出来的，不是偏好）：

1. 弧长表自洽、零长段给**精确 0** 且梯度不掺 NaN（``sqrt`` 在 0 处导数无穷，
   单层 ``where`` 挡不住未选中分支的 ``0 × NaN``）。
2. 折线总长与解析 zigzag 的 ``path_length`` **单层**对账到 1e-12；**多层**必须**超出**
   解析值，超出量＝(跨层段 + 换层后对位段)——这条是连续折线与"索引空间瞬移"的
   真实差异（实测 Δy、Δz 在缺省档里是**瞬移**，而解析式对道间跳距计费、对跨层不计费），
   写成恒等式而不是"≈"，才能区分"实现错"与"口径差"。
3. 常功率列的门控**逐位＝ 1.0**（差值混合写法 ``a+u(b−a)``，不是"舍入后差不多"）。
4. 无跳段极限（单层单道）下折线采样位置与 ``_build_scan_positions`` **逐位相同**。
5. 纯 jnp：``jit``/``lax.scan`` 可用；对折点坐标与功率列都可求梯度。
6. 求解器接线是 **opt-in**：给定与解析拓扑同一个退化路径（单层单道）时，接了程序
   的整场输出与缺省档**逐位相同**；关束（功率列全 0）时不熔化；``|∂/∂scan_speed|>0``
   仍成立（时间映射没被取代）。
7. ``chain_schedule(program=...)`` 的曝光上界改由折线总长定价，且把"hatch/lt 梯度
   恒零"这件事**说出来**而不是静默。
"""
import dataclasses
import hashlib
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from amforge.core.contracts import ProcessPlan
from amforge.geometry import from_sdf_fn
from amforge.scan_program import (
    PathProgram, gate_from_power, implied_scan_speed, program_from_track_table,
    read_track_table, require_program, track_power_duty,
)
from amforge.thermal_enthalpy import (
    _build_scan_positions, _footprint, _scan_topology, chain_schedule,
    solve_enthalpy_thermal, suggest_n_steps,
)

DX = 150e-6


def plan(**kw):
    base = dict(laser_power=1200.0, scan_speed=0.8, hatch_spacing=150e-6,
                layer_thickness=120e-6, beam_radius=60e-6)
    base.update(kw)
    return ProcessPlan.uniform(3, modality="SLM", **base)


def box_coords(nx=5, ny=3, nz=3, dx=DX):
    """良定义 extent 的坐标网格（不用 from_sdf_fn 的球：它的边界体素正落在
    T2 的 δ=0 刀锋上，extent 随掩膜口径跳变，对账会变成对刀锋）。"""
    xs, ys, zs = (np.arange(n) * dx for n in (nx, ny, nz))
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    return jnp.asarray(np.stack([X, Y, Z], axis=-1), dtype=jnp.float64)


def zig_from_top(top, *, power=1.0):
    """把 ``_scan_topology`` 的返回值原样展开成折线（参数一律来自唯一出处）。"""
    (n_layers, n_lines, x_min, y_min, z_min, x_ext, y_ext, z_ext,
     spacing, _path) = [float(v) for v in top]
    return PathProgram.zigzag(
        x_min=x_min, y_min=y_min, z_min=z_min, x_ext=x_ext, y_ext=y_ext,
        z_ext=z_ext, n_layers=int(n_layers), n_lines=int(n_lines),
        spacing=spacing, layer_height=z_ext / float(n_layers), power=power)


# ---------------------------------------------------------------------------
# 1. 弧长表 / 零长段
# ---------------------------------------------------------------------------
def test_arc_length_table_matches_hand_arithmetic():
    prog = PathProgram(points=jnp.asarray([[0.0, 0, 0], [3.0, 4, 0], [3.0, 4, 12]]),
                       power=jnp.ones((3,)))
    seg = np.asarray(prog.segment_lengths)
    cum = np.asarray(prog.cumulative_length)
    assert seg.shape == (2,) and cum.shape == (3,)
    assert np.allclose(seg, [5.0, 12.0], rtol=0, atol=1e-12)
    assert cum[0] == 0.0, "累计表首元素必须是精确 0（采样用它定位第一段）"
    assert float(prog.total_length()) == pytest.approx(17.0, rel=1e-12), "3-4-5 + 12"


def test_duplicate_knot_gives_exact_zero_and_finite_gradient():
    """重复折点：前值**精确 0**，且梯度不得被 ``sqrt(0)`` 的无穷导数注进 NaN。"""
    dup = jnp.asarray([[0.0, 0, 0], [1.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
    prog = PathProgram(points=dup, power=jnp.ones((4,)))
    seg = np.asarray(prog.segment_lengths)
    assert seg[1] == 0.0, f"零长段须给精确 0（1e-30 下限只兜除法，不兜表值），实得 {seg[1]}"
    # 旧写法（单层 where + 直接 sqrt）在这里给出 6/12 个 NaN；本写法两侧梯度精确 ∓1
    got = np.asarray(jax.grad(lambda p: jnp.sum(p.sample(0.7)[0]))(prog).points)
    assert np.all(np.isfinite(got)), f"零长段两侧的梯度必须全有限，实得 {got.tolist()}"
    g2 = np.asarray(jax.grad(lambda p: jnp.sum(p.sample(0.3)[0] + p.sample(0.9)[0]))(prog).points)
    assert np.all(np.isfinite(g2)), f"梯度含 NaN：{g2.tolist()}"


# ---------------------------------------------------------------------------
# 2. 与解析 zigzag 的对账
# ---------------------------------------------------------------------------
def test_single_layer_total_length_equals_analytic_path():
    coords = box_coords(nx=5, ny=3, nz=1)
    top = _scan_topology(coords, plan(), dim=3, solid=jnp.ones(coords.shape[:-1]))
    prog = zig_from_top(top)
    assert float(top[0]) == 1.0, "夹具应是单层"
    ratio = float(prog.total_length()) / float(top[-1])
    assert abs(ratio - 1.0) < 1e-12, f"单道内跳距两边都计费 ⇒ 比值应≈1，实得 {ratio!r}"


@pytest.mark.parametrize("n_lines", [2, 3], ids=["even-n_lines-无对位段", "odd-n_lines-一道对位"])
def test_multilayer_excess_is_jump_plus_reposition(n_lines):
    """多层折线总长 − 解析路径长 ＝ 跨层段 + 换层后对位段（手算常数）。

    缺省档的 dwell law 在层/道边界**瞬移**（实测单步 Δy=Δz=一个间距/层厚），而解析式
    对道间跳距计费、对跨层不计费 ⇒ 连续折线**必然**更长。这条恒等式把差异写成可检验
    的量，而不是"实现错了"。
    """
    x_ext, spacing, layer_height = 1.0, 0.1, 0.2
    prog = PathProgram.zigzag(x_min=0.0, y_min=0.0, z_min=0.0, x_ext=x_ext,
                              y_ext=(n_lines - 0.5) * spacing, z_ext=2 * layer_height,
                              n_layers=2, n_lines=n_lines, spacing=spacing,
                              layer_height=layer_height, power=1.0)
    analytic = 2 * (n_lines * x_ext + (n_lines - 1) * spacing)
    cross = np.hypot(spacing * (n_lines - 1), layer_height)     # 唯一的层界跳段
    reposition = 0.0 if n_lines % 2 == 0 else x_ext             # 末道止于 x_max ⇒ 奇数道要回位
    expected = analytic + cross + reposition
    got = float(prog.total_length())
    assert abs(got - expected) <= 1e-12 * expected, (
        f"n_lines={n_lines}：总长 {got!r} ≠ 解析 {analytic} + 跨层 {cross} + 对位 {reposition}")
    # 分区校核：逐段和 == 总长（否则上面的恒等式会被舍入偏置蒙过去）
    assert abs(float(jnp.sum(prog.segment_lengths)) - got) <= 1e-15 * got


# ---------------------------------------------------------------------------
# 3. 门控
# ---------------------------------------------------------------------------
def test_constant_power_gate_is_bit_exactly_one():
    prog = PathProgram(points=jnp.asarray([[0.0, 0, 0], [1.0, 0, 0], [1.0, 1, 0]]),
                       power=jnp.full((3,), 1200.0))
    gate = np.asarray(prog.power_gate(97, 1200.0))
    assert gate.shape == (97,)
    assert np.all(gate == 1.0), f"常功率列门控须**逐位** 1.0（缺省档能量表达式才不动），" \
                                f"实得 max|g−1|={np.max(np.abs(gate - 1.0))}"


def test_gate_is_a_ratio_and_beam_off_is_zero():
    prog = PathProgram(points=jnp.asarray([[0.0, 0, 0], [1.0, 0, 0]]),
                       power=jnp.asarray([600.0, 600.0]))
    assert np.all(np.asarray(gate_from_power(prog.power, 1200.0)) == 0.5)
    off = PathProgram(points=jnp.asarray([[0.0, 0, 0], [1.0, 0, 0]]), power=jnp.zeros((2,)))
    assert np.all(np.asarray(off.power_gate(11, 1200.0)) == 0.0), "关束＝功率列写 0"


def test_duty_is_mean_of_gate_not_knot_weighted():
    """占空比只能吃门控本身：改前按**折点**口径的段长加权、形状 (n−1,) 与 (n_steps,)
    广播静默错位（实测常功率列返回 31.0 而非 1.0）。"""
    gate = jnp.asarray([1.0, 1.0, 0.0, 1.0])
    assert track_power_duty(gate) == pytest.approx(0.75)
    with pytest.raises(ValueError, match="一维"):
        track_power_duty(jnp.ones((4, 3)))


# ---------------------------------------------------------------------------
# 4. 无跳段极限：与缺省 dwell law 逐位相同
# ---------------------------------------------------------------------------
def test_no_jump_positions_are_bit_identical_to_default_dwell_law():
    coords = box_coords(nx=5, ny=3, nz=3)
    wide = plan(hatch_spacing=6.0e-3, layer_thickness=6.0e-3)   # ⇒ 单道单层（无跳段可言）
    solid = jnp.ones(coords.shape[:-1])
    top = _scan_topology(coords, wide, dim=3, solid=solid)
    assert float(top[0]) == 1.0 and float(top[1]) == 1.0, "夹具须退化成一条直线"
    prog = zig_from_top(top, power=_P(wide))
    n_steps = 137
    got, _ = _build_scan_positions(coords, wide, jnp.asarray(DX), n_steps=n_steps,
                                   dim=3, solid=solid)
    ref, _pw = prog.sample_steps(n_steps)
    assert np.array_equal(np.asarray(got), np.asarray(ref)), (
        "无跳段极限下两者须**逐位**相同，实得 max_abs="
        f"{float(np.max(np.abs(np.asarray(got) - np.asarray(ref))))}")
    assert float(prog.total_length()) == pytest.approx(float(top[-1]), rel=1e-12)


# ---------------------------------------------------------------------------
# 5. 纯 jnp：jit / lax.scan / 两条可微通道
# ---------------------------------------------------------------------------
def test_usable_inside_jit_and_lax_scan():
    prog = PathProgram(points=jnp.asarray([[0.0, 0, 0], [1.0, 0, 0], [1.0, 1, 0]]),
                       power=jnp.asarray([1.0, 2.0, 3.0]))

    @jax.jit
    def one(g):
        return prog.sample(g)[0][0]

    # 手算：两段各长 1 ⇒ 总长 2；g=0.5 → s=1.0 正落在折点 (1,0,0) 上。取段指标用
    # ``sum(cum <= s) − 1`` ⇒ 边界处选**后一段**且 u=0，两处给同一个点（连续，无跳变）。
    assert one(0.5) == pytest.approx(1.0, rel=1e-12)

    @jax.jit
    def walk(gs):
        def body(c, g):
            p, _ = prog.sample(g)
            return c + p[1], p[0]
        return jax.lax.scan(body, 0.0, gs)

    total, xs = walk(jnp.linspace(0.0, 1.0, 9))
    # y 分量：前 5 个 g（≤0.5）还在第一段 ⇒ 0；后 4 个给 0.25/0.5/0.75/1.0 ⇒ Σ=2.5
    assert float(total) == pytest.approx(2.5, rel=1e-12)
    assert float(xs[0]) == pytest.approx(0.0, abs=1e-12)
    # sample_steps 的 n_steps 是**静态形状**（与求解器里同一个口径）：必须标 static
    def stepper(p, n):
        return p.sample_steps(n)[0].shape

    assert jax.jit(stepper, static_argnums=1)(prog, 64) == (64, 3)


def test_gradient_flows_to_knots_and_to_power_column():
    pts = jnp.asarray([[0.0, 0, 0], [2.0, 0, 0], [2.0, 1, 0]])
    pw = jnp.asarray([1.0, 3.0, 3.0])
    prog = PathProgram(points=pts, power=pw)
    # 手算：段长 2 与 1 ⇒ 总长 3；g=0.5 → s=1.5 落在第一段（cum=[0,2,3]），份额 u=0.75
    pos, val = prog.sample(jnp.asarray(0.5))
    assert float(pos[0]) == pytest.approx(1.5, rel=1e-12)
    assert float(pos[1]) == 0.0
    assert float(val) == pytest.approx(1.0 + 0.75 * 2.0, rel=1e-12), "混合功率＝1+0.75·(3−1)"

    g_pts = np.asarray(jax.grad(lambda p: jnp.sum(p.sample(0.99)[0]))(prog).points)
    assert np.all(np.isfinite(g_pts)), f"折点梯度须有限：{g_pts.tolist()}"
    assert float(np.abs(g_pts).sum()) > 0.0, "位置必须对折点坐标可微（否则工艺优化接不上）"

    g_pw = np.asarray(jax.grad(lambda p: p.sample(0.5)[1])(prog).power)
    assert g_pw[1] == pytest.approx(0.75, rel=1e-9), "∂混合功率/∂折点功率 ＝ 段内份额"
    assert g_pw[0] == pytest.approx(0.25, rel=1e-9) and g_pw[2] == 0.0
    h = 1e-6
    plus = float(PathProgram(points=pts, power=pw.at[1].add(h)).sample(0.5)[1])
    minus = float(PathProgram(points=pts, power=pw.at[1].add(-h)).sample(0.5)[1])
    assert (plus - minus) / (2 * h) == pytest.approx(0.75, rel=1e-6), "中心差分对账自动微分"


# ---------------------------------------------------------------------------
# 6. eager 守卫（报错一律带 "scan_program"，与求解器其余守卫同一条可 grep 口径）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad,expect", [
    (object(), "PathProgram"),
    (None, "PathProgram"),
])
def test_guard_rejects_non_program(bad, expect):
    with pytest.raises(ValueError, match="scan_program"):
        require_program(bad, reference_power=1200.0)


def test_guard_rejects_bad_numbers():
    good = jnp.asarray([[0.0, 0, 0], [1.0, 0, 0]])
    cases = [
        (jnp.asarray([[jnp.nan, 0, 0], [1.0, 0, 0]]), jnp.ones((2,)), 1200.0, "非有限"),
        (good, jnp.asarray([1.0, jnp.inf]), 1200.0, "功率列含非有限"),
        (good, jnp.asarray([1.0, -5.0]), 1200.0, "不得为负"),
        (jnp.zeros((3, 3)), jnp.ones((3,)), 1200.0, "总长"),
        (good, jnp.asarray([600.0, 600.0]), 0.0, "比值"),
    ]
    for pts, pw, ref, key in cases:
        with pytest.raises(ValueError, match="scan_program"):
            require_program(PathProgram(points=pts, power=pw), reference_power=ref)
        with pytest.raises(ValueError, match=key):
            require_program(PathProgram(points=pts, power=pw), reference_power=ref)


def test_guard_is_trace_safe_no_op_under_grad():
    """trace 下无法固化 ⇒ 整体跳过而不是把可微链打死（前提交给 eager 侧调用方）。"""
    prog = PathProgram(points=jnp.asarray([[0.0, 0, 0], [1.0, 0, 0]]),
                       power=jnp.ones((2,)))

    def loss(p):        # p 是 tracer 化的 PathProgram
        return jnp.sum(require_program(p, reference_power=jnp.asarray(1200.0)).sample(0.5)[0])

    g = np.asarray(jax.grad(loss)(prog).points)
    assert np.all(np.isfinite(g)) and float(np.abs(g).sum()) > 0.0, \
        f"守卫在 trace 里必须透明，梯度实得 {g.tolist()}"


# ---------------------------------------------------------------------------
# 7. 轨迹表：四列全留 + 时间列只做诊断
# ---------------------------------------------------------------------------
def test_track_table_keeps_all_columns_and_decodes_mu_header(tmp_path):
    """NIST MDS2-3662 的表头 µ 是**非 UTF-8 单字节**；旧 ``from_csv`` 直接抛
    UnicodeDecodeError ⇒ 基准轨迹表读不进来。这里用同一副字节验证读得开且四列全留。"""
    csv = tmp_path / "singleTrack.csv"
    csv.write_bytes(b"x (\xb5m),y (\xb5m),laser power (W),time (seconds)\n"
                    b"500,500,285,0\n500,2500,285,0.00208\n")
    tab = read_track_table(csv)
    cols = np.asarray(tab["columns"])
    assert cols.shape == (2, 4), f"四列必须全留（历史缺陷：只取前两列），实得 {cols.shape}"
    assert "µ" in tab["header"][0], f"µ 须还原成 U+00B5，实得 {tab['header']!r}"
    assert tab["index"][2] == "laser power (W)"

    prog = program_from_track_table(tab, x_col=0, y_col=1, power_col=2, unit=1e-6)
    assert float(prog.total_length()) == pytest.approx(2.0e-3, rel=1e-12), "2000µm 竖直段"
    assert np.all(np.asarray(prog.power) == 285.0)
    speed = implied_scan_speed(tab, prog, time_col=3)
    assert speed == pytest.approx(2.0e-3 / 2.08e-3, rel=1e-12), \
        "时间列只以**诊断量**可见：弧长↔速度映射没被取代"
    with pytest.raises(ValueError, match="time_col"):
        implied_scan_speed(tab, prog)


def test_track_table_column_selection_is_explicit(tmp_path):
    """列一律按**列号**指定；越界必须报错而不是静默补零。"""
    csv = tmp_path / "two_col.csv"
    csv.write_bytes(b"500,500,285,0\n500,2500,285,1\n")
    tab = read_track_table(csv)
    assert int(np.asarray(tab["columns"]).shape[1]) == 4
    with pytest.raises(ValueError, match="scan_program"):
        program_from_track_table(tab, x_col=0, y_col=1, z_col=4)
    with pytest.raises(ValueError, match="scan_program"):
        program_from_track_table(tab, x_col=0, y_col=1, power_col=9)


# ---------------------------------------------------------------------------
# 8. 求解器接线（opt-in）
# ---------------------------------------------------------------------------
def _single_line_geometry():
    """一条直线即可覆盖拓扑的几何：hatch/lt 大到只出 1 层 1 道 ⇒ 程序与解析路径重合，
    于是"接了程序"这件事必须**逐位**复现缺省档输出。"""
    part = from_sdf_fn(lambda x: jnp.max(jnp.abs(x) - 250e-6, axis=-1),
                       bounds=[(-350e-6, 350e-6)] * 3, spacing=100e-6, name="cube")
    wide = plan(hatch_spacing=6.0e-3, layer_thickness=6.0e-3)
    top = _scan_topology(part.coords(), wide, dim=3, solid=_footprint(part))
    assert float(top[0]) == 1.0 and float(top[1]) == 1.0, "夹具须退化成单层单道"
    return part, wide, top


def _P(plan):
    """工艺里的参考功率（标量）：门控是**比值**，分母取 laser_power 的均值。"""
    return float(np.mean(np.atleast_1d(np.asarray(plan.laser_power, dtype=np.float64))))


def _fields(th):
    return {f.name: getattr(th, f.name)
            for f in dataclasses.fields(th) if getattr(th, f.name) is not None}


def _digest(a):
    return hashlib.md5(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes()).hexdigest()


def test_solver_with_equivalent_program_is_bit_identical_to_default():
    part, wide, top = _single_line_geometry()
    prog = zig_from_top(top, power=_P(wide))
    ns = suggest_n_steps(part, wide)
    base = solve_enthalpy_thermal(geometry=part, process=wide,
                                  params={"material": "316L", "n_steps": ns})
    with_prog = solve_enthalpy_thermal(geometry=part, process=wide,
                                       params={"material": "316L", "n_steps": ns,
                                               "scan_program": prog})
    fb, fw = _fields(base), _fields(with_prog)
    assert sorted(fb) == sorted(fw)
    bad = [k for k in fb if not np.array_equal(np.asarray(fb[k]), np.asarray(fw[k]))]
    assert not bad, f"opt-in 分支必须逐位复现缺省档，不符字段：{bad}"
    assert float(jnp.max(base.peak_temperature)) > 1500.0, \
        "夹具本身要真的熔过，否则上面的『逐位相同』可以靠『两边都冷』蒙过去"


def test_solver_beam_off_program_does_not_melt_and_default_does():
    part, wide, top = _single_line_geometry()
    ns = suggest_n_steps(part, wide)
    on = zig_from_top(top, power=_P(wide))
    off = PathProgram(points=on.points, power=jnp.zeros((on.points.shape[0],)))
    th = solve_enthalpy_thermal(geometry=part, process=wide,
                                params={"material": "316L", "n_steps": ns,
                                        "scan_program": off})
    assert float(jnp.max(th.peak_temperature)) < 1200.0, \
        "功率列全 0 ⇒ 门控恒 0 ⇒ 不该有任何熔化"
    assert float(jnp.sum(th.time_above_melt)) == 0.0
    base = solve_enthalpy_thermal(geometry=part, process=wide,
                                  params={"material": "316L", "n_steps": ns})
    assert float(jnp.max(base.peak_temperature)) > 1500.0, \
        "同一夹具不加门控时要真的熔（否则上面那条『不熔化』是空的）"


def test_solver_still_differentiable_wrt_scan_speed_with_program():
    part, wide, top = _single_line_geometry()
    prog = zig_from_top(top, power=_P(wide))
    ns = suggest_n_steps(part, wide)

    def loss(s):
        p = wide.replace(scan_speed=jnp.atleast_1d(s))
        th = solve_enthalpy_thermal(geometry=part, process=p,
                                    params={"material": "316L", "n_steps": ns,
                                            "scan_program": prog})
        return jnp.sum(th.peak_temperature)

    g = float(jax.grad(loss)(jnp.asarray(0.8)))
    assert np.isfinite(g), "接了程序也不能把 scan_speed 的梯度通道打断"
    assert abs(g) > 0.0, "时间映射＝弧长/scan_speed 没被取代 ⇒ |∂/∂v| 必须非零"


def test_solver_program_grads_are_finite():
    part, wide, top = _single_line_geometry()
    prog = zig_from_top(top, power=_P(wide))
    ns = max(2, suggest_n_steps(part, wide) // 8)   # 只查梯度有限性，不需要时间精度

    def loss(pts):
        p = PathProgram(points=pts, power=prog.power)
        th = solve_enthalpy_thermal(geometry=part, process=wide,
                                    params={"material": "316L", "n_steps": ns,
                                            "scan_program": p})
        return jnp.sum(th.peak_temperature)

    g = np.asarray(jax.grad(loss)(prog.points))
    assert g.shape == np.asarray(prog.points).shape
    assert np.all(np.isfinite(g)), "折点坐标→温度场的梯度必须全有限"


# ---------------------------------------------------------------------------
# 9. chain_schedule 的 program= 档位
# ---------------------------------------------------------------------------
def test_chain_schedule_prices_exposure_from_program():
    part, wide, top = _single_line_geometry()
    prog = zig_from_top(top, power=_P(wide))
    with pytest.warns(UserWarning, match="梯度恒零"):
        sched = chain_schedule(part, wide, program=prog)
    total = float(prog.total_length())
    assert sched["path_bound_m"] == pytest.approx(total, rel=1e-12), \
        "曝光上界必须由**折线总长**定价（path_slack 对程序分支是空转）"
    v_lo = sched["bounds"]["scan_speed"][0]
    assert sched["exposure_bound_s"] == pytest.approx(total / v_lo, rel=1e-12)
    assert sched["n_steps"] >= 1


def test_chain_schedule_default_unchanged_by_new_kwarg():
    part = from_sdf_fn(lambda x: jnp.linalg.norm(x, axis=-1) - 0.24e-3,
                       bounds=[(-0.24e-3, 0.24e-3)] * 3, spacing=80e-6, name="box")
    p = plan()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sched = chain_schedule(part, p, over_budget="pin")
    assert not [w for w in caught if "梯度恒零" in str(w.message)], \
        "不给 program 就不该出现程序档位的说明（opt-in 必须真是 opt-in）"
    analytic = float(_scan_topology(part.coords(), p, dim=3, solid=_footprint(part))[-1])
    v_lo = sched["bounds"]["scan_speed"][0]
    assert sched["path_bound_m"] >= analytic * (1.0 - 1e-12), \
        "缺省档的最坏角落上界不得低于名义路径"
    assert sched["exposure_bound_s"] == pytest.approx(sched["path_bound_m"] / v_lo,
                                                      rel=1e-12)
    assert sched["n_steps"] >= 1
