# -*- coding: utf-8 -*-
"""整数位点容差吸附（T3 / #24，2026-10-08）回归测试。

守住的缺陷（与 T1 ``test_geometry_voxelization.py`` 同族，但普查到全树）：
「两个物理长度相除 → 取整 → 决定域/栅格/路径有多大」的位点用**裸** ``ceil``/``floor``，
于是同一零件、同一版本代码**只换参数写法**就换一个整数。改前实测（``docs/evidence/
2026-10-08/am_t3_intsnap_probe_pre.log``，84 例 × 4 拼写）：**31/84 例翻转，11 位点里 10 位有翻转**：

* 层高→层数 ``600µm/40µm``：``−1ulp`` 的层厚把比值抬成 ``15.000000000000002`` ⇒ 多一层；
* 基板 ``250µm/12.5µm``：字面量拼写 ``ceil`` 给 **21 格**（设计 20 格）；
* 3 层粉末床 ``n_layers·lt / (2·d50)``：``−1ulp`` ⇒ ``floor+1`` 掉一层 ⇒ 粒子数 100→**75**；
* hatch ``400µm/100µm``：``+1ulp`` 的 hatch ⇒ GUI **少一条扫描线**。

约定只在 ``amforge.counts`` 里定义一次（相对容差 ``1e-9·max(1,|比值|)``）；``diffmech`` 在
``amforge`` 之下（依赖单向）⇒ 那 3 处按**同宽**内联，本文件里的 ``test_diffmech_width_matches``
就是两包同宽的跨包守护（§26.17：靠文本绑定的东西一旦没人跑探针就静默漂移）。

已知**不**由本文件管的：``zigzag_segments`` 的**绝对** ``+1e-12`` fudge（3.5e-9 相对，带宽的 3.5 倍
⇒ 吸不到）在整除跨度处系统性多算一道 ⇒ 本文件把它**锁成现状**（``test_zigzag_bias_locked``），
修它归台账 #32；ceil/round/floor+1 三套口径的分裂归 #31。
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from amforge import geometry as G                                # noqa: E402
from amforge import process as PR                                # noqa: E402
from amforge.counts import TOL_REL, count_ceil, count_floor       # noqa: E402
from amforge.core.contracts import PartGeometry, ProcessPlan      # noqa: E402
from amforge.geometry import (crop_to_part, layer_z_heights,      # noqa: E402
                              with_baseplate)
from amforge.gui.preproc import generate_hatch_paths             # noqa: E402
from amforge.powder import ParticleCollection                     # noqa: E402
from diffmech.methods.am.scan_paths import zigzag_hatch           # noqa: E402

UM = 1e-6


def spellings(x: float) -> dict[str, float]:
    """同一物理量的四种**保值**二进制拼写：字面量、−1ulp、+1ulp、``µm 数值×1e-6``。

    ``µm×1e-6`` 用 ``:.17g`` 读写＝精确往返，只引入"先除以 1e-6 再乘回来"的双舍入（T1 的成因）；
    6 位有效数字的 ``:g`` 会**改物理量**（探针里 3.1e-7 相对）⇒ 不是拼写，不进测试。
    """
    lit = float(x)
    um = lit / UM
    return {"lit": lit, "m1": float(np.nextafter(lit, -np.inf)),
            "p1": float(np.nextafter(lit, np.inf)), "mul": float(f"{um:.17g}") * UM}


def _part(dx: float, nz: int, nx: int = 5, ny: int = 5) -> PartGeometry:
    return PartGeometry(sdf=jnp.full((nx, ny, nz), -1.0, dtype=jnp.float64),
                        origin=jnp.zeros(3, dtype=jnp.float64), spacing=float(dx), dim=3)


def _block_part(dx: float, grid: int = 41, block: int = 21) -> PartGeometry:
    a = (grid - block) // 2
    sdf = np.full((grid, grid, grid), 1.0)
    sdf[a:a + block, a:a + block, a:a + block] = -1.0
    return PartGeometry(sdf=jnp.asarray(sdf, dtype=jnp.float64),
                        origin=jnp.zeros(3, dtype=jnp.float64), spacing=float(dx), dim=3)


# ===========================================================================
# 助手本体：语义 + 带宽（宽度是判据，写宽/写窄都会被抓）
# ===========================================================================
def test_count_ceil_semantics():
    assert count_ceil(20.0) == 20                      # 恰整数
    assert count_ceil(20.000000000000004) == 20        # 整数上方 1 ulp ⇒ 吸附
    assert count_ceil(19.999999999999996) == 20        # 整数下方本来就该进位
    assert count_ceil(20.4) == 21                      # 真余量 ⇒ 绝不欠覆盖


def test_count_floor_semantics():
    assert count_floor(3.0) == 3
    assert count_floor(3.0 - 4.5e-16) == 3             # 整数下方 1 ulp ⇒ 吸附
    assert count_floor(3.0000000000000004) == 3        # 上方本来就该舍
    assert count_floor(2.6) == 2                        # 真缺口 ⇒ 不多算


def test_band_width_is_one_ninth_to_one_hundredth():
    """1e-10（相对）余量必须被吸掉，1e-8 必须**不**被吸掉 ⇒ 1e-9 真在量宽度。"""
    assert TOL_REL == 1e-9
    for q in (4.0, 15.0, 20.0):
        assert count_ceil(q * (1 + 1e-10)) == int(q)
        assert count_ceil(q * (1 + 1e-8)) == int(q) + 1
        assert count_floor(q * (1 - 1e-10)) == int(q)
        assert count_floor(q * (1 - 1e-8)) == int(q) - 1


# ===========================================================================
# 生产位点：刀锋跨度换拼写 ⇒ 整数必须全同，且等于设计整数
# ===========================================================================
@pytest.mark.parametrize("dx_um,cells,q", [(50.0, 12, 15), (25.0, 24, 12),
                                           (100.0, 8, 4), (12.5, 32, 20)])
def test_layer_count_and_z_heights_spelling_invariant(dx_um, cells, q):
    """S01/S03：层高→层数（``cells·dx`` 是 ``lt`` 的整数倍＝刀锋）。"""
    height = dx_um * cells * UM
    lt = height / q
    part = _part(dx_um * UM, cells + 1)
    for name, v in spellings(lt).items():
        assert int(part.layer_count(v)) == q, (name, v)
        assert len(layer_z_heights(part, v)) == q, (name, v)


@pytest.mark.parametrize("pdx_um,n_cells,q", [(25.0, 20, 20), (50.0, 6, 6),
                                              (12.5, 8, 20)])
def test_baseplate_cells_spelling_invariant(pdx_um, n_cells, q):
    """S02：基板厚→格数（除数是 ``part.spacing`` ⇒ 被换拼写的是 spacing 一侧）。"""
    thickness = pdx_um * q * UM
    for name, sp in spellings(pdx_um * UM).items():
        part = _part(sp, 4)
        assert int(with_baseplate(part, thickness=thickness).shape[-1]) - 4 == q, (name, sp)


@pytest.mark.parametrize("dx_um,q", [(50.0, 3), (25.0, 5), (12.5, 8)])
def test_crop_margin_pad_spelling_invariant(dx_um, q):
    """S04：margin→裁剪 pad 格数（求解域大小）。"""
    dx = dx_um * UM
    margin = dx * q
    for name, m in spellings(margin).items():
        p = _block_part(dx)
        got = (int(crop_to_part(p, margin=m).shape[0]) - 21) // 2
        assert got == q, (name, m, got)


@pytest.mark.parametrize("sp_um,n_layers", [(50.0, 3), (50.0, 8), (40.0, 5)])
def test_powder_lattice_spelling_invariant(sp_um, n_layers):
    """S06：``floor(n_layers·lt / 2·d50)+1``——换 1 ulp 就掉一层，粒子数随之跳。"""
    d50 = sp_um / 2 * UM
    lt = sp_um * UM
    part = _part(50e-6, 9)
    ref = int(ParticleCollection.from_part_bed(part, layer_thickness=lt, d50=d50,
                                               n_layers=n_layers, seed=0).n_particles)
    for name, v in spellings(lt).items():
        got = int(ParticleCollection.from_part_bed(part, layer_thickness=v, d50=d50,
                                                   n_layers=n_layers, seed=0).n_particles)
        assert got == ref, (name, v, got, ref)


def test_powder_lattice_drop_is_real_defect_shape():
    """正对照：把生产侧换回裸 ``floor`` ⇒ 本例确实掉一层（否则上面的门是空的）。

    改前实测 ``sp=50µm, n_layers=3`` 的 −1ulp 层厚把粒子数从 100 打到 **75**（探针 S06 段）。
    """
    sp, n_layers = 50.0, 3
    ratio = n_layers * sp / (2 * (sp / 2))
    bare = int(np.floor(float(np.nextafter(ratio, -np.inf))) + 1)
    snapped = count_floor(ratio) + 1
    assert snapped == n_layers + 1
    assert bare == n_layers, (bare, snapped)


@pytest.mark.parametrize("q", [4, 8, 10])
def test_gui_hatch_lines_spelling_invariant(q):
    """S08：GUI hatch 线数 ``floor(span/step)+1``，除数用夹具**实测**浮点跨度＝刀锋。"""
    part = _part(50e-6, 5, nx=9, ny=9)
    span = float(part.bbox()[1][1] - part.bbox()[0][1])
    ref = None
    for name, h in spellings(span / q).items():
        d = generate_hatch_paths(part, 40e-6, h)
        n = len({round(float(seg[0][1]), 12) for seg in d["paths"][0]})
        if ref is None:
            ref = n
        assert n == ref, (name, h, n, ref)
    assert ref == q + 1


@pytest.mark.parametrize("q", [4, 8, 16])
def test_diffmech_hatch_lines_spelling_invariant(q):
    """S09：``diffmech`` 侧同族位点（内联容差），跨包必须同稳定。"""
    ax = np.arange(9) * 50e-6
    span = float(ax[-1] - ax[0])
    ref = None
    for name, h in spellings(span / q).items():
        path = zigzag_hatch(np.ones((9, 9)), ax, ax, hatch_spacing=h)
        n = len({round(float(y), 12) for y in np.asarray(path.waypoints)[:, 1]})
        if ref is None:
            ref = n
        assert n == ref, (name, h, n, ref)
    assert ref == q


def test_diffmech_width_matches_amforge():
    """跨包同宽守护：``diffmech`` 的内联 1e-9 与 ``amforge.counts`` 必须同判。

    取比值**刚越过**整数（1 ulp 上方）与**真余量**（+0.4 格）两侧：前者两包都得吸附，后者两包
    都得进位。⇒ 任何一包单独改宽度都会在这里炸，而不是只在探针日志里静默不同。
    """
    ax = np.arange(9) * 50e-6
    span = float(ax[-1] - ax[0])
    for q, want in [(8, 8), (4, 4)]:
        r = span / (span / q)
        assert count_ceil(span / (span / q)) == int(want)
        path = zigzag_hatch(np.ones((9, 9)), ax, ax, hatch_spacing=span / q)
        n = len({round(float(y), 12) for y in np.asarray(path.waypoints)[:, 1]})
        assert n == count_ceil(r), (q, n, count_ceil(r))
    # 真余量：q+0.4 格 ⇒ 两包都必须多一条线（不得欠覆盖）
    step = span / 7.6
    path = zigzag_hatch(np.ones((9, 9)), ax, ax, hatch_spacing=step)
    n = len({float(y) for y in np.asarray(path.waypoints)[:, 1]})
    assert n == count_ceil(span / step) == 8


# ===========================================================================
# 扫描道数：拼写稳定（现状偏高一道＝#32，本文件锁住现状）
# ===========================================================================
@pytest.mark.parametrize("q", [4, 8, 20])
def test_zigzag_bias_locked(q):
    """S05：换拼写不改整数（本轮），但 ``diag`` 的**绝对** fudge 仍多算一道（#32）。

    实测比值 = ``q·(1+3.5355e-9)``，带宽只有 1e-9 ⇒ 吸附**不该**也**不会**吃掉它。
    #32 落地（把绝对 fudge 从计数里摘出去）时应把期望改成 ``q``，并在 CHANGELOG 指名这里。
    """
    part = _part(50e-6, 5, nx=5, ny=5)
    lo, hi = part.bbox()
    diag_um = float(np.linalg.norm(np.asarray(hi)[:2] - np.asarray(lo)[:2])) / UM
    plan_kwargs = dict(modality="SLM", laser_power=200.0, scan_speed=1.0,
                       layer_thickness=40e-6, beam_radius=50e-6)
    ints = set()
    for name, h in spellings(diag_um / q * UM).items():
        plan = ProcessPlan.uniform(1, hatch_spacing=h, **plan_kwargs)
        seg = PR.zigzag_segments(part, plan)
        ints.add(int(jnp.asarray(seg.start).shape[0]))
    assert len(ints) == 1, ints
    assert ints == {q + 1}, (sorted(ints), q)


@pytest.mark.parametrize("dx_um,cells", [(50.0, 12), (12.5, 32)])
def test_grid_axes_control_still_holds(dx_um, cells):
    """T1 正对照（``_grid_axes``）必须与新助手同判据通过 ⇒ 证明门槛可被**正确实现**达到。"""
    ext = dx_um * cells * UM
    bounds = [(-ext / 2, ext / 2)] * 3
    for name, v in spellings(dx_um * UM).items():
        assert len(G._grid_axes(bounds, v)[0]) == cells + 1, (name, v)
