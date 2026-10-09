"""#18 改前打分探针：路径程序（``amforge.scan_program``）在接入求解器**之前**的可微性、
与既有解析 zigzag 的对账，以及缺省链的改前指纹（供接入后逐位比对）。

口径全部现测，不采信转述：K3 把树内另一条轨迹实现（``meltpool.polyline_trajectory``
的 ``jnp.interp`` + ``total_time=float``）亲手量出来；K2b 的期望值由折线几何手算，
不是从被测代码里回读。每个"相符"读数都配一条**能变红**的坏口径正对照。

夹具口径：K4/K5/K6 用**直接给出的坐标网格**而不是 ``from_sdf_fn`` 的球——球的边界体素
正落在 T2 的 δ=0 刀锋上（``voxel-boundary-knife-edge``），extent 会随掩膜口径跳变，
把"对账"变成"对刀锋"。拓扑对账要在良定义的 extent 上做。
"""
import dataclasses
import hashlib
import json
import math
import os

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from amforge.core.contracts import ProcessPlan
from amforge.geometry import from_sdf_fn
from amforge.meltpool import polyline_trajectory
from amforge.scan_program import (PathProgram, implied_scan_speed, program_from_track_table,
                                  read_track_table, track_power_duty)
from amforge.thermal_enthalpy import (_build_scan_positions, _footprint, _scan_topology,
                                      solve_enthalpy_thermal, suggest_n_steps)

OUT = []
REPO = "/home/shy/桌面/2026年开发-显式冲击有限元-物质点法-sph-fvm+am/AM-Qoder"
CSV = os.path.join(REPO, "_refs", "benchmarks", "nist_mds2-3662",
                   "Scan Strategy Data", "singleTrack.csv")


def say(k, v):
    OUT.append((k, v))
    print(k + " :: " + str(v), flush=True)


def arr_md5(a):
    return hashlib.md5(np.asarray(a, dtype=np.float64).tobytes()).hexdigest()


def plan(power=1200.0, speed=0.8, hatch=150e-6, lt=120e-6):
    return ProcessPlan.uniform(3, modality="SLM", laser_power=power, scan_speed=speed,
                               layer_thickness=lt, hatch_spacing=hatch,
                               beam_radius=100e-6, absorption=0.45, preheat_temp=400.0)


def box_coords(nx=5, ny=3, nz=3, dx=150e-6):
    """良定义 extent 的坐标网格：x∈[0,0.6mm]、y∈[0,0.3mm]、z∈[0,0.3mm]（dim=3）。"""
    xs = np.arange(nx) * dx
    ys = np.arange(ny) * dx
    zs = np.arange(nz) * dx
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    return jnp.asarray(np.stack([X, Y, Z], axis=-1), dtype=jnp.float64)


# ==========================================================================
# K0 环境 / 指纹
# ==========================================================================
say("K0 环境", "SRC=" + os.environ.get("SRCFINGER", "?") + " jax=" + jax.__version__
    + " x64=" + str(jax.config.jax_enable_x64) + " head="
    + os.popen("git -C " + REPO + " rev-parse --short HEAD").read().strip()
    + " dirty(src+tests)="
    + os.popen("git -C " + REPO + " -c core.quotePath=false status --porcelain -- src tests"
               " | wc -l").read().strip())

# ==========================================================================
# K1 静态构造 / 弧长表自洽（3-4-5 与 5+12=17 的手算折线）
# ==========================================================================
pts = jnp.asarray([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [3.0, 4.0, 12.0]])
prog1 = PathProgram(points=pts, power=jnp.asarray([1.0, 1.0, 1.0]))
seg, cum, tot = np.asarray(prog1.segment_lengths), np.asarray(prog1.cumulative_length), float(prog1.total_length())
say("K1a 段长/累计/总长", f"seg={list(np.round(seg, 12))} cum={list(np.round(cum, 12))} total={tot:.12f}（手算 5+12=17）")
assert abs(tot - 17.0) < 1e-12 and float(cum[0]) == 0.0
assert abs(tot - float(cum[-1])) < 1e-15, "总长的唯一出处必须是累计表末元素"
prog1b = PathProgram(points=jnp.asarray([[0.0, 0.0], [3.0, 4.0], [0.0, 0.0]]), power=1.0)
say("K1b 正对照 回头路", f"首尾直线距={float(jnp.linalg.norm(prog1b.points[-1]-prog1b.points[0])):.3f} "
                         f"折线总长={float(prog1b.total_length()):.3f}（须 10：把回去的 5 也计上）")
assert abs(float(prog1b.total_length()) - 10.0) < 1e-12
say("K1c (n,2) 补 z 与 (n,3) 同长",
    f"(n,2)={float(prog1b.total_length()):.12f} "
    f"(n,3)={float(PathProgram(points=jnp.asarray([[0., 0., 0.], [3., 4., 0.], [0., 0., 0.]]), power=1.0).total_length()):.12f}")

# ==========================================================================
# K2 梯度回流到折点坐标与功率列；段选择由"哪一格梯度恰为 0"钉住
# ==========================================================================
G2 = 0.37
PW2 = jnp.asarray([1.0, 2.0, 3.0])           # 非常值功率列，才能验出"混的是哪两段"
S2 = G2 * 17.0                    # = 6.29 → 落在第二段（下标 1）
U2 = (S2 - 5.0) / 12.0            # 段内份额 = 0.1075
prog2 = PathProgram(points=pts, power=PW2)
pos2, pw2 = prog2.sample(G2)
say("K2a 采样点", f"g={G2} s={S2:.6f} 期望段=1 份额={U2:.10f} pos={list(np.round(np.asarray(pos2), 10))} "
                  f"power={float(pw2):.10f}")
exp_pos = (1.0 - U2) * np.asarray(pts[1]) + U2 * np.asarray(pts[2])
exp_pw = (1.0 - U2) * 2.0 + U2 * 3.0
assert np.allclose(np.asarray(pos2), exp_pos, atol=1e-12), "落段/份额与手算不符"
assert abs(float(pw2) - exp_pw) < 1e-12, f"功率列的分段线性混合应为 {exp_pw}，实得 {float(pw2)}"


def _pw_of(p_pts, p_pw):
    return PathProgram(points=p_pts, power=p_pw).sample(G2)[1]


g_pw = np.asarray(jax.grad(_pw_of, argnums=(0, 1))(pts, jnp.asarray([1.0, 2.0, 3.0]))[1])
say("K2b ∂power/∂功率列", f"{list(np.round(g_pw, 12))}（手算 (0, 1−u, u)＝{[0.0, round(1-U2, 10), round(U2, 10)]}）")
assert g_pw[0] == 0.0, "第 0 折点不该进第二段 ⇒ 梯度必须恰为 0（段选择可证伪）"
assert np.allclose(g_pw, [0.0, 1.0 - U2, U2], atol=1e-12)


def _sumpos(p_pts):
    return jnp.sum(PathProgram(points=p_pts, power=jnp.asarray([1.0, 1.0, 1.0])).sample(G2)[0])


g_pts = np.asarray(jax.grad(_sumpos)(pts))
say("K2c ∂Σpos/∂points", f"有限={bool(np.all(np.isfinite(g_pts)))} 非零元={int((g_pts != 0.0).sum())}/{g_pts.size} "
                          f"max|g|={float(np.max(np.abs(g_pts))):.6e}")
assert np.all(np.isfinite(g_pts)) and float(np.max(np.abs(g_pts))) > 0.0, "折点坐标拿不到梯度"
# 中心差分独立校核（梯度非零不等于梯度正确）
h = 1e-6
fd = np.zeros_like(g_pts)
for i in range(3):
    for j in range(3):
        pp = jnp.asarray(np.asarray(pts).copy()); pm = jnp.asarray(np.asarray(pts).copy())
        pp = pp.at[i, j].add(h); pm = pm.at[i, j].add(-h)
        fd[i, j] = (_sumpos(pp) - _sumpos(pm)) / (2 * h)
say("K2d 正对照 中心差分", f"max|解析−差分|={float(np.max(np.abs(fd - g_pts))):.3e} "
                          f"相对={float(np.max(np.abs(fd - g_pts)) / max(np.max(np.abs(g_pts)), 1e-30)):.3e}（须 <1e-6）")
assert float(np.max(np.abs(fd - g_pts))) / max(float(np.max(np.abs(g_pts))), 1e-30) < 1e-6


def _stop(stop):
    def f(p_pts):
        q = p_pts if not stop else jax.lax.stop_gradient(p_pts)
        return jnp.sum(PathProgram(points=q, power=jnp.asarray([1.0, 1.0, 1.0])).sample(G2)[0])
    return float(np.max(np.abs(np.asarray(jax.grad(f)(pts)))))


say("K2e 正对照 stop_gradient", f"截断前={_stop(False):.6e} → 截断后={_stop(True):.6e}（须恰 0，否则 K2c 的『非零』是噪声）")
assert _stop(True) == 0.0

# ==========================================================================
# K3 树内另一条轨迹实现的实测（不采信"不可微"的转述，也不采信"可微"的转述）
# ==========================================================================
g_interp_fp = np.asarray(jax.grad(lambda w: jnp.interp(3.0, jnp.asarray(cum), w))(jnp.asarray(pts[:, 0])))
g_interp_xp = np.asarray(jax.grad(lambda c: jnp.interp(3.0, c, jnp.asarray(pts[:, 0])))(jnp.asarray(cum)))
say("K3a jnp.interp 梯度", f"∂/∂折点值={list(np.round(g_interp_fp, 6))}（s=3 落在 [0,5) ⇒ 手算 (0.4,0.6,0)） "
                           f"∂/∂弧长表={list(np.round(g_interp_xp, 6))}")
assert np.allclose(g_interp_fp, [0.4, 0.6, 0.0], atol=1e-12)
tr = polyline_trajectory(pts, speed=0.5)
say("K3b polyline_trajectory.total_time", f"type={type(tr.total_time).__name__} 值={tr.total_time!r}"
                                          "（float ⇒ 时长那条链在构造时就被具体化）")
assert type(tr.total_time) is float
sp = jnp.asarray(0.5)


def _poly_built(v):
    """在 trace **内**构造轨迹（优化器对工艺/折点求导时就是这个形状）。"""
    x, y, _ = polyline_trajectory(pts, speed=v).fn(1.0)
    return x + y


try:
    v3c = float(jax.grad(_poly_built)(sp))
    k3c = f"可导，grad={v3c:.6e}"
except Exception as e:  # noqa: BLE001
    k3c = f"抛 {type(e).__name__}：{str(e).splitlines()[0][:150]}"
say("K3c ∂pos(1s)/∂speed（构造在 trace 内）", k3c)
# 对照：构造在 trace 外、只对 fn 求导 ⇒ 那条"对 speed 可微"仅在这一支成立
tr0 = polyline_trajectory(pts, speed=0.5)
v3d = float(jax.grad(lambda v: jnp.sum(tr0.fn(v)[0] + tr0.fn(v)[1]))(jnp.asarray(0.5)))
v3e = float(jax.grad(lambda v: jnp.sum(tr0.fn(1.0)[0] + tr0.fn(1.0)[1]))(jnp.asarray(0.5)))
say("K3d ∂pos(t)/∂t", f"{v3d:.6e}（t 那条链仍连着构造时定格的 speed=0.5）")
say("K3e ∂pos(1s)/∂speed（构造在 trace 外）", f"{v3e:.6e}（**恒 0**：速度已在构造时折进 fn 的闭包，"
                                             "再对外部速度求导拿不到任何耦合 ⇒ 熔池求解器侧的速度梯度断在这里）")
assert v3e == 0.0
say("K3f ScanPath.position_at 的 searchsorted",
    "diffmech/methods/am/scan_paths.py:47 用 np.searchsorted ⇒ 只能 eager 逐点求值，"
    "不能进 jax.lax.scan/jit（本探针 K8 实测 PathProgram 两者皆可）")

# ==========================================================================
# K4 总长对账（单层）：折线 zigzag vs _scan_topology 的解析 path_length
# ==========================================================================
co4 = box_coords()
(nl4, nn4, xm4, ym4, zm4, xe4, ye4, ze4, sp4, pl4) = _scan_topology(co4, plan(), dim=3, solid=None)
say("K4a 拓扑现值", f"n_layers={float(nl4):.0f} n_lines={float(nn4):.0f} x_ext={float(xe4)*1e6:.3f}µm "
                    f"y_ext={float(ye4)*1e6:.3f}µm z_ext={float(ze4)*1e6:.3f}µm spacing={float(sp4)*1e6:.3f}µm "
                    f"path={float(pl4)*1e6:.6f}µm（手算 2×(2×600+1×150)=2700µm）")
assert int(nl4) == 2 and int(nn4) == 2
# 单层：把层厚抬到超过 z_ext ⇒ n_layers=1
(nl1, nn1, xm1, ym1, zm1, xe1, ye1, ze1, sp1, pl1) = _scan_topology(
    co4, plan(lt=1e-3), dim=3, solid=None)
say("K4b 单层拓扑", f"n_layers={float(nl1):.0f} n_lines={float(nn1):.0f} path={float(pl1)*1e6:.6f}µm"
                    f"（手算 2×600+1×150=1350µm）")
assert int(nl1) == 1 and int(nn1) == 2


def zig(nl, nn, xm, ym, zm, xe, ye, ze, sp, pw=1.0):
    return PathProgram.zigzag(x_min=float(xm), y_min=float(ym), z_min=float(zm),
                              x_ext=float(xe), y_ext=float(ye), z_ext=float(ze),
                              n_layers=int(nl), n_lines=int(nn), spacing=float(sp),
                              layer_height=float(ze) / float(nl), power=pw)


zz1 = zig(nl1, nn1, xm1, ym1, zm1, xe1, ye1, ze1, sp1)
rel1 = abs(float(zz1.total_length()) - float(pl1)) / float(pl1)
say("K4c 单层总长对账", f"折线={float(zz1.total_length())*1e6:.6f}µm 解析={float(pl1)*1e6:.6f}µm "
                        f"rel={rel1:.3e}（须 <1e-12） 折点数={int(np.asarray(zz1.points).shape[0])}")
assert rel1 < 1e-12, "单层 zigzag 的折线总长必须与解析 path_length 对账到 1e-12"
zz_bad = zig(nl1, nn1, xm1, ym1, zm1, xe1, ye1, ze1, float(sp1) * 1.001)
say("K4d 正对照 spacing×1.001", f"rel={abs(float(zz_bad.total_length())-float(pl1))/float(pl1):.3e}（须 >1e-6 而红）")
assert abs(float(zz_bad.total_length()) - float(pl1)) / float(pl1) > 1e-6
b4e = arr_md5(zz1.total_length())
t4e = float(zz1.total_length())
p_ulp = jnp.asarray(zz1.points)
# 先复现**探针自己踩过的那一脚**：折点 0 的 x 恰为 0.0 ⇒ math.ulp(0.0) 是最小非规格化数
# （5e-324），加上去什么也没变 ⇒ 这条正对照没有分辨力。缺陷在选分量，不在被测代码。
z0 = float(p_ulp[0, 0])
blind = arr_md5(PathProgram(points=p_ulp.at[0, 0].add(math.ulp(z0)), power=zz1.power).total_length()) == b4e
say("K4e-a 无分辨力的一脚", f"折点 0 的 x={z0:.1e}，math.ulp={math.ulp(z0):.3e}（最小非规格化）"
                            f" ⇒ 加完总长逐位不变={blind}（须 True，登记为本探针的自纠素材）")
assert blind is True
# 真正的控制：挑绝对值最大的分量，沿**它自己的量级**用 nextafter 走 1 ulp
flat = int(np.argmax(np.abs(p_ulp)))
ii, jj = divmod(flat, 3)
mag = float(p_ulp[ii, jj])
tgt = float(np.nextafter(mag, math.inf if mag >= 0 else -math.inf))
step = abs(tgt - mag)
prg_ulp = PathProgram(points=p_ulp.at[ii, jj].set(tgt), power=zz1.power)
c4e = arr_md5(prg_ulp.cumulative_length)
t4e_ulp = float(prg_ulp.total_length())
say("K4e-b 正对照 1ulp（非零分量）", f"折点[{ii}] 分量 {jj}：{mag:.12g} → {tgt:.12g}（步长 {step:.3e}）"
                                     f"弧长表 md5 变={c4e != arr_md5(zz1.cumulative_length)} "
                                     f"总长 md5 变={arr_md5(t4e_ulp) != b4e} 总长差={abs(t4e_ulp-t4e):.3e}")
assert c4e != arr_md5(zz1.cumulative_length), "1 ulp 的折点改动至少必须反映到弧长表"
# 总长为什么可以不动：被改的分量自身 ulp＝1.08e-19，而总长(1.35e-3)自身的 ulp＝2.17e-19，
# 即这条改动**恰为总长的半个 ulp** ⇒ 落进舍入平局。这不是代码钝化，是**控制选错了步长**，
# 所以再加一条把步长抬到总长 ulp 以上。
ulp_tot = math.ulp(t4e)
tgt4 = float(np.float64(mag) + np.float64(4.0 * step) * (1.0 if mag >= 0 else -1.0))
t4c = float(PathProgram(points=p_ulp.at[ii, jj].set(tgt4), power=zz1.power).total_length())
say("K4e-c 正对照 4ulp", f"步长={abs(tgt4-mag):.3e}（>总长自身 ulp={ulp_tot:.3e}）"
                         f"总长 {t4e*1e6:.9f}µm → {t4c*1e6:.9f}µm rel={abs(t4c-t4e)/t4e:.3e} "
                         f"md5 异={arr_md5(t4c) != b4e}")
assert arr_md5(t4c) != b4e and abs(t4c - t4e) > 0.0, "4 ulp 的折点改动必须反映到总长"

# ==========================================================================
# K5 多层：折线必须真的走层间行程 ⇒ 严格长于忽略它的解析式；差值有解析解释
# ==========================================================================
for n_lines_t, tag in ((2, "偶道数"), (3, "奇道数")):
    pr_t = plan(hatch=float(ye4) / n_lines_t)
    (nlx, nnx, xmx, ymx, zmx, xex, yex, zex, spx, plx) = _scan_topology(co4, pr_t, dim=3, solid=None)
    zzx = zig(nlx, nnx, xmx, ymx, zmx, xex, yex, zex, spx)
    P = np.asarray(zzx.points)
    dP = np.diff(P, axis=0)
    ln = np.linalg.norm(dP, axis=-1)
    cross = np.abs(dP[:, 2]) > 0.0                     # 换层的段（选择规则与总长不同）
    t_cross = float(np.sum(ln[cross]))
    # 每处换层之后紧跟的那一段（把 x 对回本层起扫位）在解析式里根本不存在
    x_extra = float(np.sum([ln[i] for i in range(1, len(ln))
                            if np.abs(dP[i - 1, 2]) > 0.0
                            and abs(dP[i, 1]) < 1e-18 and abs(dP[i, 2]) < 1e-18]))
    diff = float(zzx.total_length()) - float(plx)
    exp = t_cross + x_extra
    say(f"K5-{tag} 多层差值", f"n_layers={int(nlx)} n_lines={int(nnx)} 折线={float(zzx.total_length())*1e6:.4f}µm "
                              f"解析={float(plx)*1e6:.4f}µm 差={diff*1e6:.4f}µm 跨层段和={t_cross*1e6:.4f}µm "
                              f"换层后对位段和={x_extra*1e6:.4f}µm 跨层段数={int(cross.sum())}/{ln.size}")
    assert abs(diff - exp) < 1e-12 * float(plx), "差值应恰等于『跨层段＋换层后对位段』"
    assert diff > 0.0, "连续折线必含层间行程 ⇒ 严格长于忽略它的解析式"
    say(f"K5-{tag}b 分区校核", f"Σ段长={float(np.sum(ln))*1e6:.6f}µm vs total_length={float(zzx.total_length())*1e6:.6f}µm "
                               f"比值={float(zzx.total_length())/float(plx):.6f}")
    assert abs(float(np.sum(ln)) - float(zzx.total_length())) < 1e-12 * float(plx)

# ==========================================================================
# K6 采样律对账：无跳段极限（n_lines=1 且 n_layers=1）位置；有跳段时两律必分岔
# ==========================================================================
ns6 = 17
pr6 = plan(hatch=1e-3, lt=1e-3)                       # 道距/层厚都大于 extent ⇒ 各 1 道 1 层
(nl6, nn6, xm6, ym6, zm6, xe6, ye6, ze6, sp6, pl6) = _scan_topology(co4, pr6, dim=3, solid=None)
say("K6a 无跳段夹具", f"n_layers={float(nl6):.0f} n_lines={float(nn6):.0f} path={float(pl6)*1e6:.6f}µm")
assert int(nl6) == 1 and int(nn6) == 1, "夹具没落进无跳段极限，K6 失效"
pos_ref, _ = _build_scan_positions(co4, pr6, dx=jnp.asarray(150e-6), n_steps=ns6, dim=3, solid=None)
zz6 = zig(nl6, nn6, xm6, ym6, zm6, xe6, ye6, ze6, sp6)
pos_p, _ = zz6.sample_steps(ns6)
A, B = np.asarray(pos_p), np.asarray(pos_ref)
scale = max(float(np.max(np.abs(B))), 1e-30)
say("K6b 无跳段极限位置差", f"max_abs={np.abs(A-B).max():.3e}m max_rel={np.abs(A-B).max()/scale:.3e}"
                            f"（须 <1e-12） 逐位相同={bool(np.array_equal(A, B))}")
assert np.abs(A - B).max() / scale < 1e-12
pos_ref2, _ = _build_scan_positions(co4, plan(), dx=jnp.asarray(150e-6), n_steps=ns6, dim=3, solid=None)
(nl7, nn7, xm7, ym7, zm7, xe7, ye7, ze7, sp7, pl7) = _scan_topology(co4, plan(), dim=3, solid=None)
zz7 = zig(nl7, nn7, xm7, ym7, zm7, xe7, ye7, ze7, sp7)
pos_p7, _ = zz7.sample_steps(ns6)
d7 = float(np.max(np.abs(np.asarray(pos_p7) - np.asarray(pos_ref2))))
say("K6c 正对照 n_lines=" + str(int(nn7)), f"max_abs={d7:.3e}m（须显著非零 ⇒ 两种驻留律确实不同）")
assert d7 > 1e-9
prj = np.asarray(pos_ref2)
say("K6d 解析律的瞬跳", f"最大单步 Δy={float(np.abs(np.diff(prj[:, 1])).max())*1e6:.3f}µm "
                        f"Δz={float(np.abs(np.diff(prj[:, 2])).max())*1e6:.3f}µm；"
                        f"跳段占解析路径长 {float((nn7-1)*nl7*sp7)/float(pl7)*100:.2f}%"
                        f"（折线律恒速连续走，把这些行程也摊进时间）")

# ==========================================================================
# K7 功率门控：常功率列逐位 1.0（"缺省能量表达式一字不动"的前提）
# ==========================================================================
zz7P = PathProgram(points=zz7.points, power=jnp.full((int(zz7.points.shape[0]),), 1200.0))
gate_const = zz7P.power_gate(64, 1200.0)
say("K7a 常功率列门控", f"全部逐位==1.0={bool(jnp.all(gate_const == 1.0))} "
                        f"min={float(jnp.min(gate_const)):.17g} max={float(jnp.max(gate_const)):.17g} "
                        f"max|gate−1|={float(jnp.max(jnp.abs(gate_const - 1.0))):.3e}")
assert bool(jnp.all(gate_const == 1.0)), "常功率列的门控必须**逐位**等于 1.0（差值混合写法的立身之本）"
gate_off = PathProgram(points=zz7.points, power=zz7P.power.at[1].set(0.0)).power_gate(64, 1200.0)
say("K7b 正对照 第 1 折点关束", f"==1.0 的比例={float(jnp.sum(gate_off == 1.0))/64:.3f} "
                                f"含零={bool(jnp.any(gate_off == 0.0))} min={float(jnp.min(gate_off)):.6g}")
assert not bool(jnp.all(gate_off == 1.0)), "坏口径正对照失效：改功率列后门控仍恒 1"
g7 = np.asarray(jax.grad(lambda w: jnp.sum(PathProgram(points=zz7.points, power=w).power_gate(8, 1200.0)))(jnp.full((zz7.points.shape[0],), 1200.0)))
say("K7c ∂Σgate/∂功率列", f"非零元={int((g7 != 0).sum())}/{g7.size} 有限={bool(np.all(np.isfinite(g7)))}")
assert int((g7 != 0).sum()) > 0
say("K7d 门控对参考功率的梯度", f"∂Σgate/∂P={float(jax.grad(lambda P: jnp.sum(zz7P.power_gate(8, P)))(jnp.asarray(1200.0))):.6e}"
                                f"（常功率列＝P 时应为 −n_steps/P＝{-8/1200.0:.6e}，比值门的分母同样可微）")

# ==========================================================================
# K8 jit / lax.scan 安全性（opt-in 分支要能进时间扫描）
# ==========================================================================
steps_jit = jax.jit(lambda p, n: PathProgram(points=p, power=jnp.full((p.shape[0],), 1.0)).sample_steps(n),
                    static_argnums=1)   # n_steps 是**静态形状**（与求解器里同一个口径）
o8 = steps_jit(zz7.points, 8)
say("K8a jit 内可用", f"形状={tuple(np.asarray(o8[0]).shape)} 有限={bool(np.all(np.isfinite(np.asarray(o8[0]))))}")


zz5 = zig(nl4, nn4, xm4, ym4, zm4, xe4, ye4, ze4, sp4)


def scan_body(carry, _):
    s, acc = carry
    pos, pw = zz5.sample(s)
    return (s + jnp.asarray(1.0 / 64), acc + pos[0] * pw), pos[0]


(_, acc8), _ = jax.lax.scan(scan_body, (jnp.asarray(0.0), jnp.asarray(0.0)), None, length=8)
say("K8b lax.scan 内可用", f"acc={float(acc8):.6e} 有限={bool(math.isfinite(float(acc8)))}")
g8 = np.asarray(jax.grad(lambda p: jax.jit(lambda q: jnp.sum(
    PathProgram(points=q, power=jnp.full((q.shape[0],), 1.0)).sample_steps(8)[0]))(p))(zz7.points))
say("K8c jit+grad 折点", f"非零元={int((g8 != 0).sum())}/{g8.size} 有限={bool(np.all(np.isfinite(g8)))}")
assert np.all(np.isfinite(g8)) and float(np.max(np.abs(g8))) > 0
# 把 NaN 的来源单独钉住：zigzag 里必然出现**重复折点**（跳位点与本道起扫点重合），
# 零长段的 sqrt 导数无穷。先复刻改前写法证明它会红，再证明现写法全有限。
dup = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])


def _old_seg(p):
    d = p[1:] - p[:-1]
    return jnp.sum(jnp.sqrt(jnp.sum(d * d, axis=-1)))          # 改前：直接 sqrt


def _new_seg(p):
    return jnp.sum(PathProgram(points=p, power=jnp.full((p.shape[0],), 1.0)).segment_lengths)


g_old, g_new = np.asarray(jax.grad(_old_seg)(dup)), np.asarray(jax.grad(_new_seg)(dup))
say("K8d 零长段的 sqrt", f"改前 ∂Σseg/∂points 非有限元={int((~np.isfinite(g_old)).sum())}/{g_old.size} "
                         f"值={list(np.round(g_old, 4))}；现写法非有限元={int((~np.isfinite(g_new)).sum())}/{g_new.size} "
                         f"值={list(np.round(g_new, 4))}")
assert not np.all(np.isfinite(g_old)), "坏口径正对照失效：直接 sqrt 本该在零长段出 NaN"
_sgn = np.asarray([-1.0, 1.0, -1.0, 1.0])
exp_new = np.stack([_sgn, np.zeros(4), np.zeros(4)], axis=-1)   # Σ|Δx| 的逐点符号（只有 x 分量动）
assert np.all(np.isfinite(g_new)) and bool(np.all(g_new == exp_new)), \
    f"零长段两端的梯度应为 ∓1、该段贡献精确 0，实得 {g_new.tolist()}"
say("K8e 零长段的前值", f"段长={list(np.asarray(PathProgram(points=dup, power=jnp.full((4,), 1.0)).segment_lengths))}"
                        "（重复折点须给**精确 0**，否则总长对账 K4c 会带 1e-15 的偏置）")

# ==========================================================================
# K9 缺省链改前指纹（接入后必须逐位相同）
# ==========================================================================
g9 = from_sdf_fn(lambda x: jnp.linalg.norm(x, axis=-1) - 0.24e-3,
                 bounds=[(-0.24e-3, 0.24e-3)] * 3, spacing=80e-6, name="box")
p9 = plan()
ns9 = suggest_n_steps(g9, p9)
nvox9 = int(np.asarray(g9.sdf).size)
say("K9a 夹具", f"n_steps={ns9} 体素={nvox9} voxel_steps={ns9*nvox9} dx=80µm hatch=150µm lt=120µm")
th9 = solve_enthalpy_thermal(geometry=g9, process=p9, params={"material": "316L", "n_steps": ns9})
fp9 = {}
for fld in sorted(f.name for f in dataclasses.fields(th9)):
    v = getattr(th9, fld)
    if v is None or isinstance(v, (int, float, str, bool)):
        fp9[fld] = repr(v)
        continue
    a = np.asarray(v, dtype=np.float64)
    fp9[fld] = {"md5": arr_md5(a), "shape": list(a.shape), "max": float(a.max()), "sum": float(a.sum())}
say("K9b 改前指纹", json.dumps({k: (v if isinstance(v, str) else v["md5"][:16]) for k, v in fp9.items()},
                              ensure_ascii=False))
say("K9c 关键字段", json.dumps({k: fp9[k] for k in sorted(fp9) if isinstance(fp9[k], dict)},
                               ensure_ascii=False, sort_keys=True)[:1200])
with open("/tmp/_s18_pre.json", "w") as fh:
    json.dump({"n_steps": ns9, "nvox": nvox9, "fingerprint": fp9}, fh, sort_keys=True)
say("K9d 落盘", "/tmp/_s18_pre.json 字段数=" + str(len(fp9)))

# ==========================================================================
# K10 轨迹表读取：四列全留 + 时间列只做诊断；并实测旧 from_csv 丢列
# ==========================================================================
tab = read_track_table(CSV)
say("K10a 表头与列数", f"header={tab['header']} 列数={int(np.asarray(tab['columns']).shape[1])} "
                       f"行数={int(np.asarray(tab['columns']).shape[0])} index={tab['index']}")
assert int(np.asarray(tab["columns"]).shape[1]) == 4, "四列必须全留（历史缺陷：只取前两列）"
prg10 = program_from_track_table(tab, x_col=0, y_col=1, power_col=2, unit=1e-6)
say("K10b 折线", f"折点={tuple(np.asarray(prg10.points).shape)} 总长={float(prg10.total_length())*1e6:.6f}µm "
                 f"功率列={list(np.round(np.asarray(prg10.power), 6))}")
v_imp = implied_scan_speed(tab, prg10, time_col=3)
say("K10c 时间列→等效速度", f"{v_imp:.6g} m/s（手算 2000µm/2.08ms；诊断量，不绕过弧长映射）")
assert abs(v_imp - 2e-3 / 2.08e-3) < 1e-9
duty10 = track_power_duty(prg10.power_gate(32, 285.0))
say("K10d 关束占空比", f"{duty10:.6f}（表内功率恒 285W ⇒ 须 1.0）"
                       f"；正对照：把第 2 折点功率改 0 后={track_power_duty(PathProgram(points=prg10.points, power=prg10.power.at[1].set(0.0)).power_gate(32, 285.0)):.6f}")
assert abs(duty10 - 1.0) < 1e-12, f"常功率列的占空比应为 1.0，实得 {duty10}（改前的广播错位会给出 n_steps−1 量级）"
say("K10e 正对照 不猜单位", f"unit=1.0 时总长={float(program_from_track_table(tab, x_col=0, y_col=1, power_col=2).total_length()):.3f}m"
                            f"（＝按表内数字当米，2000 倍过头 ⇒ 单位必须由调用方指名）")
from diffmech.methods.am import scan_paths as sp10
try:
    p10 = sp10.from_csv(CSV, x_col=0, y_col=1, skip_header=1, delimiter=",")
    k10f = (f"waypoints 形状={tuple(np.asarray(p10.waypoints).shape)}"
            "⇒ 第 3、4 列（功率、时间）去向=静默丢弃")
    assert np.asarray(p10.waypoints).shape[-1] == 2
except Exception as e:  # noqa: BLE001
    k10f = (f"抛 {type(e).__name__}：{str(e).splitlines()[0][:110]}"
            "⇒ **连读都读不进**（np.loadtxt 按 utf-8 解表头，µ 是 0xb5）；"
            "本模块的 read_track_table 靠 utf-8→latin-1 退路把列名还原成 µ（见 K10a）")
say("K10f 旧 from_csv 对同一张表", k10f)

print("\n=== 全部断言通过 ===")
print("READOUTS=" + str(len(OUT)))
