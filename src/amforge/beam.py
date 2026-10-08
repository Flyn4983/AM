"""光斑口径的**单点出处**（#22/T4）。

契约 ``core/contracts.py::ProcessPlan.beam_radius`` 声明的是 **1/e² 半径** ``r_b``：

    I(ρ) = I0 · exp(−2ρ²/r_b²) ,   I(r_b) = I0/e² ,   等效高斯标准差 σ = r_b/2

本模块把这条换算写成一处，求解器不得自行重写指数。历史分裂（改前实测见
``docs/evidence/2026-10-08/am_t4_spot_probe_pre.log``）：
``thermal_enthalpy._moving_source``（``source_model="point"``）与
``meltpool._laser_source_fdm``（**默认**熔池求解器）把同一个输入数解成
``exp(−ρ²/r²)``——那是 **1/e** 半径，σ = r/√2——与 ``diffmech``/上游 JAXCode 的
写法差 √2（面积差 2×）⇒ 同一份工艺参数在两个求解器里根本不是同一束光。

轴向（深度）尺度**不在**本模块范围内：``dp = 1.2·r`` 一类经验系数是在旧面内口径下
标定的，随口径一起重标定会把「修口径」与「重标定」混成一件事，另立 **#33**。
"""
from __future__ import annotations

import math

import jax.numpy as jnp


def spot_sigma(beam_radius):
    """1/e² 半径 → 高斯标准差 σ = r/2。``1e-12`` 只是防零除，不是物理下限。"""
    return jnp.maximum(beam_radius, 1e-12) / 2.0


def planar_decay(planar2, beam_radius):
    """面内衰减 exp(−ρ²/(2σ²)) ≡ exp(−2ρ²/r_b²)（两副写法逐位相同，由探针 K3 断言）。"""
    sig = spot_sigma(beam_radius)
    return jnp.exp(-planar2 / (2.0 * sig * sig))


def inplane_integral(beam_radius):
    """∫exp(−2ρ²/r²) dA = 2πσ² = π r²/2 —— 面内积分因子（体积热源归一化用）。"""
    return 2.0 * math.pi * spot_sigma(beam_radius) ** 2
