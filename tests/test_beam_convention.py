# -*- coding: utf-8 -*-
"""光斑口径单点出处（#22 / T4，2026-10-08）回归测试。

守住的缺陷：``ProcessPlan.beam_radius`` 契约写的是 **1/e² 半径** ``r_b``（σ=r_b/2），
但改前的 ``amforge`` 默认路径把同一个输入数解成了 **1/e** 半径（σ=r/√2）。改前实测
（``docs/evidence/2026-10-08/am_t4_spot_probe_pre.log``）：``thermal_enthalpy._moving_source``
的 2D/3D 分支与**默认**熔池求解器 ``meltpool._laser_source_fdm`` 三处 σ/(r_b/2)=**1.414214**
（面积差 2×），而 ``diffmech``/上游 JAXCode 与同文件 VOF 强度是 1.000000
⇒ 同一份工艺参数在两个求解器里根本不是同一束光。

约定只在 ``amforge.beam`` 里定义一次；本文件量的是**行为**（二阶矩、功率、逐位相同），
不是文本——探针 K0b/K4 靠文本绑定，一旦没人跑探针就静默漂移，这三条是它在 CI 里的替身。

已知**不**由本文件管的：轴向尺度 ``dp = 1.2·r`` 与 ``max(rb, dx)`` 一类经验系数是在旧面内
口径下标定的，随口径重标定会把两件事混成一件 ⇒ 台账 **#33**；``particle_am`` 送粉/集粉概率
密度里 ``1/(πR²)`` 配 ``∫exp(−2ρ²/R²)=πR²/2`` 的**整因子差 2**（D08/D15）属分布形状 ⇒ 另立。
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from amforge.beam import inplane_integral, planar_decay, spot_sigma     # noqa: E402
from amforge.meltpool import _laser_source_fdm                          # noqa: E402
from amforge.thermal_enthalpy import (                                  # noqa: E402
    _cell_integrated_source, _moving_source,
)

# 外部锚：NIST MDS2-3662 §2.2 的 D4σ=85 µm ⇒ σ=D4/4=21.25 µm，1/e² 半径=D4/2=42.5 µm。
D4 = 85e-6
SIGMA_EXT = D4 / 4.0
R = 2.0 * SIGMA_EXT
# 与生产同源的面内经验深度尺度（出域 ⇒ #33，这里只作积分权重用，不参与口径判定）
DP = 1.2 * R

REL = 2e-3          # 与探针 K3 同容差：容得下离散偏差，离 √2=41% 差两个数量级
REL_POWER = 5e-3    # 与探针 K2 同容差


def _plane(nn=201, half=8.0 * math.sqrt(2.0) * SIGMA_EXT):
    """二维单元中心网格（覆盖最宽的 c1 情形：半径 8·√2·σ）。"""
    xs = jnp.linspace(-half, half, nn)
    xx, yy = jnp.meshgrid(xs, xs, indexing="ij")
    return jnp.stack([xx, yy], axis=-1), float(xs[1] - xs[0])


def _sigma(q, centers):
    """面内二阶矩 σ = √(⟨ρ²⟩/2)：只吃分布形状，与拼写无关。"""
    rho2 = centers[..., 0] ** 2 + centers[..., 1] ** 2
    tot = float(jnp.sum(q))
    assert tot > 0.0 and math.isfinite(tot)
    return math.sqrt(float(jnp.sum(rho2 * q)) / tot / 2.0)


# ---------------------------------------------------------------------------
# 契约本身：1/e² 的含义 + 两副写法逐位相同
# ---------------------------------------------------------------------------
def test_beam_radius_is_the_one_over_e_squared_radius():
    """ρ=r_b 处强度必须是 I0/e² —— 这就是契约里那个名字的全部内容。"""
    assert float(planar_decay(R * R, R)) == jnp.exp(-2.0)
    assert float(planar_decay(0.0, R)) == 1.0
    assert abs(float(spot_sigma(R)) - R / 2.0) <= 1e-12 * R
    assert abs(float(spot_sigma(R)) - SIGMA_EXT) <= 1e-9 * SIGMA_EXT


def test_two_spellings_of_the_decay_are_bitwise_identical():
    """σ 写法与 1/e² 写法必须**逐位**相同（差一个 ulp 都不许 ⇒ 只有一处出处）。"""
    centers, _ = _plane(nn=41)
    p2 = centers[..., 0] ** 2 + centers[..., 1] ** 2
    a = planar_decay(p2, R)
    b = jnp.exp(-p2 / (2.0 * spot_sigma(R) ** 2))
    c = jnp.exp(-2.0 * p2 / (R * R))
    assert float(jnp.max(jnp.abs(a - b))) == 0.0
    assert float(jnp.max(jnp.abs(a - c))) == 0.0


def test_inplane_integral_is_half_pi_r_squared():
    """∫exp(−2ρ²/r²)dA = 2πσ² = πr²/2（旧写法 πr² 差 2 ⇒ 与旧衰减自洽、与新衰减半修）。"""
    assert abs(float(inplane_integral(R)) - math.pi * R * R / 2.0) < 1e-9 * math.pi * R * R
    centers, dxy = _plane(nn=401)
    measured = float(jnp.sum(planar_decay(
        centers[..., 0] ** 2 + centers[..., 1] ** 2, R))) * dxy * dxy
    assert abs(measured / float(inplane_integral(R)) - 1.0) < REL


# ---------------------------------------------------------------------------
# 行为：三个求解器位点 + 跨包同宽（改回 c1 ⇒ 这里立刻红）
# ---------------------------------------------------------------------------
def test_moving_source_plane_sigma_matches_contract():
    """"point" 档的 2D/3D 面内项：σ 实测 = r_b/2。"""
    centers, _ = _plane()
    q2 = _moving_source(jnp.zeros(2), centers, 1.0, R, 1e-6)
    assert abs(_sigma(q2, centers) / (R / 2.0) - 1.0) < REL
    c3 = jnp.concatenate([centers, jnp.zeros(centers.shape[:2] + (1,))], axis=-1)
    q3 = _moving_source(jnp.zeros(3), c3, 1.0, R, DP)
    assert abs(_sigma(q3, centers) / (R / 2.0) - 1.0) < REL


def test_default_meltpool_fdm_source_sigma_and_power():
    """**默认**熔池求解器的体积热源：面内宽度合契约 ∧ 总吸收功率守恒。

    功率守恒是抓"半修"的那一条：只把衰减改成 exp(−2ρ²/r²) 而不把归一化里的 πr² 换成
    2πσ²，σ 照样对，但 ∫q dV 会差 2 倍。
    """
    centers, dxy = _plane()
    dA = dxy * dxy
    nz, span = 33, 8.0
    dz = span * DP / nz
    zs = -(jnp.arange(nz) + 0.5) * dz          # 单元中心（端点矩形会重复计表面值）
    acc = None
    for z in zs:
        c3 = jnp.concatenate([centers, jnp.full(centers.shape[:2] + (1,), float(z))], axis=-1)
        q = _laser_source_fdm(c3, 0.0, rb=R, A_eff=1.0, P=1.0, absorption_depth=DP, dx=dxy)
        acc = q if acc is None else acc + q
    acc = acc * dz
    assert abs(_sigma(acc, centers) / (R / 2.0) - 1.0) < REL
    assert abs(float(jnp.sum(acc)) * dA / 1.0 - 1.0) < REL_POWER


def test_cell_integrated_default_still_conforms():
    """缺省 "integrated" 档（erf）改前**就已**是 σ=r_b/2，只是文字写错 ⇒ 不许顺手改行为。

    这里把它钉成现状：任何把 ``s_in`` 从 ``r/√2`` 挪到 ``r/2`` 的"统一拼写"改动都会把
    实测 σ 推离 r_b/2 到 √2 倍 ⇒ 本条红。
    """
    centers, dxy = _plane()
    q = _cell_integrated_source(jnp.zeros(2), centers, 1.0, R, DP, dxy)
    assert abs(_sigma(q, centers) / (R / 2.0) - 1.0) < REL


def test_diffmech_width_matches_amforge():
    """跨包同宽：``diffmech`` 的 2D 高斯与 ``amforge.beam`` 对同一输入给同一个 σ。

    依赖方向是 ``amforge → diffmech``，所以 diffmech 侧保留内联写法（探针 K4 守它不被
    反向 import 改掉）；两边同宽只能靠实测比，文本比对不到。
    """
    from diffmech.methods.fvm import thermal as th

    centers, _ = _plane()
    src = th.gaussian_heat_source(power=100.0, absorption=1.0, beam_radius=float(R),
                                 center=(0.0, 0.0))
    q_dm = jnp.asarray(src(centers, 0.0))
    sig_dm = _sigma(q_dm, centers)
    sig_af = _sigma(planar_decay(centers[..., 0] ** 2 + centers[..., 1] ** 2, R), centers)
    assert abs(sig_dm / sig_af - 1.0) < REL
    assert abs(sig_dm / (R / 2.0) - 1.0) < REL
