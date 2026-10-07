"""体素化栅格（``geometry._grid_axes``）回归测试 — D4 / §25.7 T1。

守住的缺陷（T1，2026-10-06 实测）：``n = ceil((hi-lo)/spacing)+1`` 对 **dx 的浮点
拼写**敏感。``12.5*1e-6 = 1.2499999999999999e-05`` 比字面量 ``12.5e-6`` 小 1 ulp，
当件厚恰为 dx 的整数倍时比值从 32.0 变成 32.000000000000004，``ceil`` 便多出一格：
同一零件、同一版本代码，**只换参数写法**就多一层 Z 体素（161602 vs 156849 个体素），
实测峰值温度差 0.53%、熔化体积差 1.77%（§25.7 U4）。D4 的修复是按比例容差吸附。

本文件只做栅格几何（不解热方程），故为纯 CPU、毫秒级。
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from amforge import geometry as G                           # noqa: E402
from amforge.geometry import _grid_axes                     # noqa: E402

# 件厚 0.8mm：对下列 dx 都是整数倍（16/32/64/8/40 格），正是 T1 的触发条件
_EXTENT = 0.8e-3
_BOUNDS = [(-_EXTENT / 2, _EXTENT / 2)] * 3
_CASES = [(50e-6, 16), (25e-6, 32), (12.5e-6, 64), (100e-6, 8), (20e-6, 40)]


def _spellings(lit):
    """同一个 dx 的两种浮点拼写：字面量 vs ``µm 数值 × 1e-6``（二进制差 1 ulp）。"""
    dx_um = lit * 1e6
    mul = float(f"{dx_um:g}") * 1e-6
    # 前提检查：乘法拼写确实**偏小**，使整数倍件厚的比值越过整数（T1 的成因）。
    assert mul < lit, (lit, mul)
    assert _EXTENT / mul > _EXTENT / lit == round(_EXTENT / lit)
    return lit, mul


@pytest.mark.parametrize("lit,n_cells", _CASES)
def test_grid_axes_spelling_invariant(lit, n_cells):
    """件厚为 dx 整数倍时，1-ulp 的 dx 拼写不得改变栅格形状。"""
    lit_sp, mul_sp = _spellings(lit)
    a_lit = [len(ax) for ax in _grid_axes(_BOUNDS, lit_sp)]
    a_mul = [len(ax) for ax in _grid_axes(_BOUNDS, mul_sp)]
    assert a_lit == [n_cells + 1] * 3, a_lit
    assert a_mul == a_lit, (a_mul, a_lit)


@pytest.mark.parametrize("lit,n_cells", _CASES)
def test_from_sdf_fn_shape_invariant(lit, n_cells):
    """同一 SDF、同一 bounds，两种 dx 拼写必须给出同一 ``shape``（T1 直接复现）。"""
    lit_sp, mul_sp = _spellings(lit)
    fn = lambda x: jnp.linalg.norm(x, axis=-1) - 0.3e-3  # noqa: E731
    g_lit = G.from_sdf_fn(fn, bounds=_BOUNDS, spacing=lit_sp, name="t1")
    g_mul = G.from_sdf_fn(fn, bounds=_BOUNDS, spacing=mul_sp, name="t1")
    assert g_lit.shape == g_mul.shape == (n_cells + 1,) * 3
    assert g_lit.sdf.size == g_mul.sdf.size


def test_real_remainder_still_gets_extra_cell():
    """容差吸附只吃 1-ulp 抖动：**真的**不足一格的余量仍要多铺一格（不得欠覆盖）。"""
    dx = 50e-6
    extent = 33.4 * dx                        # 余量 0.4 格 ≫ 1e-9 相对容差
    (axis,) = _grid_axes([(-extent / 2, extent / 2)], dx)
    assert len(axis) == 35                    # ceil(33.4)=34 格 → 35 个点
    assert axis[-1] >= extent / 2 - 1e-15     # 栅格覆盖到边界，不被容差吃掉


def test_axis_nodes_are_exactly_n_cells_apart():
    """吸附后的最末节点 = lo + n·dx：整数倍时既不越过 hi，也不短一格。"""
    dx = 12.5e-6
    (axis,) = _grid_axes([(-16 * dx, 16 * dx)], dx)
    assert len(axis) == 33
    assert abs(axis[-1] - 16 * dx) < 1e-15
    assert np.all(np.diff(axis) == pytest.approx(dx, rel=1e-12))
