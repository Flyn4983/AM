# -*- coding: utf-8 -*-
"""LSF/DED 送粉质量累加（#36）与集粉归一口径（#34）回归测试（2026-10-08）。

守住的缺陷（改前实测 ``docs/evidence/2026-10-08/am_t34_t36_lsf_probe.log``）：
``particle_am`` 的两处 ``isinstance`` 守卫在**函数体内**写
``from src.diffmech.methods.am.process import LSFConfig``，而生产调用链上的配置对象来自
顶层 ``diffmech…`` 口径 —— 两个 ``LSFConfig`` **不是同一个类对象**（仓库根没有
``src/__init__.py``，cwd=根时命名空间导入照样成功 ⇒ 不报错，只是守卫恒假）。
改前实测：6 副点云（共 1201 个粒子）上生产口径 ``Σ s ≡ 0``、非零数 ``≡ 0``，且
``_step_lsf_common`` 之后 mass / temperature / activation_time **三者逐位不变**
⇒ LSF/DED 的熔覆层从头到尾不长质量（``am_api.py:181`` 已正确识别 LSF 并选到
``step_lsf_am_scan``，所以这条在产品链上确实被走到，只是恒零）。
同轮正对照：把配置换成 ``src.`` 拼写构造 ⇒ ``Σ s = 3.500000e-04 = ṁ_total``、非零数 2
⇒ 缺陷由 import 口径造成，不是夹具没料。

同轮量到的 #34 两件事：

* 归一 prefactor 写成 ``1/(π R²)``，而 ``∫exp(−2r²/R²)dA = π R²/2`` ⇒ 解析正确值差 2 倍；
  但下方的离散重归一使绝对 prefactor 与返回值**逐位无关**（K2b 在 6/6 副点云上逐位相同，
  K2c 用环系数 1.8→1.5 证明该比对**能失败**）。⇒ 本文件不锁系数，只锁「改了系数也不许变数」
  所依赖的那个不变量：总量必须恰为 ṁ_total。
* 环外截断发生在归一**之后** ⇒ 被丢弃的格子照样参与归一，实测 ``Σ s`` 比 ṁ_total 少
  0.0053 %（grid17）/ 0.0131 %（wide），而 docstring 明写 "equals mdot_total"
  ⇒ 现改为先截断后归一，本文件把闭合锁到 1e-12。

**不**由本文件管的：``activation_time`` 前移的门槛 ``dm/m̄ > 1e-3`` 在改前改后都未被本夹具
触发（81 粒子、单粒子 dm≈4e-10、m̄=1e-6 ⇒ 4e-4）⇒ 该腿的可触发性没有判据，另立；
集粉环系数 1.8 与 0.2/0.8 混合权重属 #33 的口径耦合旋钮（本轮一字未动）。
"""
from __future__ import annotations

import dataclasses
import re

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from diffmech.methods.am import LSFConfig, SLMConfig          # noqa: E402  生产口径
from diffmech.methods.am import from_waypoints, multi_layer_paths  # noqa: E402
from diffmech.methods.am import particle_am as PA             # noqa: E402

MDOT = LSFConfig().powder_feed_rate * LSFConfig().deposition_efficiency


def _paths():
    """单层 1 mm 直线道 ⇒ 层时长 2 ms，t=1 ms 时激光落在 (0.5 mm, 0)。"""
    return multi_layer_paths([from_waypoints(
        np.array([[0.0, 0.0], [1.0e-3, 0.0]]), layer_z=0.0)])


STARTS = jnp.asarray([0.0])
DURS = jnp.asarray([2.0e-3])
T_ON = 1.0e-3


def _cloud(half_mm: float, n: int):
    """2D 方形点云（粒子位置只用于算 rsq，不必是真的布置结果）。"""
    g = np.linspace(-half_mm * 1e-3, half_mm * 1e-3, n)
    X, Z = np.meshgrid(g, g, indexing="ij")
    return jnp.asarray(np.stack([X.ravel(), Z.ravel()], axis=1))


def _profile(cfg, pos, t=T_ON):
    return np.asarray(PA.lsf_powder_deposition_profile(
        pos, t, _paths(), STARTS, DURS, cfg, dim=2))


def test_profile_sums_exactly_to_feed_rate_when_ring_binds():
    """#36：生产口径必须真的出料；#34：先截断后归一 ⇒ Σ s 恰为 ṁ_total。"""
    pos = _cloud(20.0, 21)                      # r_max=28.3 mm ≫ 环外截止 7.2 mm
    s = _profile(LSFConfig(), pos)
    nz = int((s > 0).sum())
    assert nz > 0, "生产口径下粉源恒零 ⇒ #36 复发（isinstance 守卫又分裂了）"
    assert nz < pos.shape[0], "环外截断没生效 ⇒ 下一条断言在本夹具上是空转"
    assert abs(s.sum() - MDOT) / MDOT < 1e-12, (
        f"Σ s={s.sum():.9e} vs ṁ_total={MDOT:.9e} ⇒ 被环外丢弃的格子仍参与了归一（#34 复发）")


def test_total_mass_strictly_increases_after_one_lsf_step():
    """#36 的正对照腿：一步 LSF 之后总质量必须严格增大，且增量恰为 ṁ_total·dt。"""
    pos = _cloud(1.0, 9)
    n = pos.shape[0]
    am = PA.ParticleAMState(
        position=pos,
        activation_time=jnp.full(n, 1e9),
        temperature=jnp.zeros(n),
        mass=jnp.full(n, 1e-6),
        layer_id=jnp.zeros(n, dtype=jnp.int32),
        rho=jnp.full(n, 8000.0))

    def stub(solver_state, am_state, dt, t, solver_cfg, cfg, paths,
             layer_start_times, layer_durations, *, walls, cp, alpha_T, tau):
        return solver_state, am_state                      # 隔离机械步，只考粉源

    dt = 1e-4
    _, am2 = PA._step_lsf_common(stub, None, am, dt, T_ON, None, LSFConfig(),
                                 _paths(), STARTS, DURS, dim=2, walls=None,
                                 cp=500.0, alpha_T=1e-5, tau=1e-3)
    dm = float(np.asarray(am2.mass).sum() - np.asarray(am.mass).sum())
    assert dm > 0.0, "LSF 一步之后质量没变 ⇒ #36 复发（粉源没接进状态）"
    assert abs(dm - MDOT * dt) / (MDOT * dt) < 1e-12
    assert float(np.asarray(am2.temperature).sum()) > 0.0, "送粉预热没写进温度"
    assert np.all(np.asarray(am2.mass) >= np.asarray(am.mass)), "有粒子被扣了质量"


def test_slm_config_still_returns_exact_zeros():
    """守卫的另一侧：SLM 配置必须**恒零**（否则上面的非零断言可以是空转）。"""
    pos = _cloud(4.0, 25)
    s = _profile(SLMConfig(), pos)
    assert s.shape == (pos.shape[0],), "返回形状不等于粒子数 ⇒ 逐粒子源的定义漂了"
    assert np.all(s == 0.0), "SLM 配置也出料 ⇒ isinstance 守卫被放宽成恒真"


def test_feed_rate_gradient_hits_analytic_value():
    """可微侧给**绝对预期**：Σ s = feed×eff ⇒ ∂Σ s/∂feed 必须恰为 eff。"""
    cfg0 = LSFConfig()
    pos = _cloud(1.0, 9)
    paths = _paths()

    def total(feed):
        cfg = dataclasses.replace(cfg0, powder_feed_rate=feed)
        return PA.lsf_powder_deposition_profile(
            pos, T_ON, paths, STARTS, DURS, cfg, dim=2).sum()

    g = float(jax.grad(total)(jnp.asarray(cfg0.powder_feed_rate)))
    assert np.isfinite(g)
    assert abs(g - cfg0.deposition_efficiency) / cfg0.deposition_efficiency < 1e-9, (
        f"∂Σ s/∂feed={g:.12e} vs deposition_efficiency={cfg0.deposition_efficiency:.12e}")


def test_no_src_prefixed_import_remains_in_the_repo():
    """口径锁：全树不得再出现 ``from src.…`` 拼写（含正对照，证明扫描器能命中）。"""
    pat = re.compile(r"^\s*(?:from src\.|import src\.|from src import)", re.M)
    root = PA.__file__.rsplit("/diffmech/", 1)[0]
    hits: list[str] = []
    for rel in ("diffmech/methods/am/particle_am.py", "diffmech/server.py"):
        txt = open(f"{root}/{rel}", encoding="utf-8").read()
        hits += [f"{rel}:{i + 1}:{l.strip()}"
                 for i, l in enumerate(txt.splitlines())
                 if pat.match(l)]
    assert hits == [], f"`src.` 前缀导入复发（守卫将再次恒假）：{hits}"
    # 仪器正对照：同一扫描器必须能在合成样本上命中
    assert len(pat.findall("    from src.diffmech.methods.am.process import X\n")) == 1


def test_local_guard_uses_the_module_level_class_object():
    """结构锁：模块级 LSFConfig 必须就是生产口径的那个类对象。"""
    assert PA.LSFConfig is LSFConfig, (
        f"particle_am 模块级 LSFConfig={PA.LSFConfig.__module__} vs 生产 "
        f"{LSFConfig.__module__} ⇒ 两副口径又分裂了")
