#!/usr/bin/env python3
"""把 `tests/test_monitoring.py::test_inverse_controller_lowers_defect_cost` 的**余量**量出来。

为什么要单独跑：该断言 `cost_corr < cost_base*0.99` 只在**失败时**打印两个数（消息写在
`assert` 的 f-string 里），所以"绿"这个事实本身不含信息 —— §26.10 明确警告过"若转绿是靠
刚好降了，#17 仍然挂着"。本探针**镜像** test_monitoring.py:162-177 的构造（逐行同参数），
额外打印：代价两端值、改善百分比、以及 inverse 控制器**实际动了哪些工艺量**（#17 的直接观测量）。

判据（跑前登记）
① 复现断言：`cost_corr < cost_base*0.99` 必须成立（不成立说明本探针与测试树不同步，先修探针）；
② 打印改善余量（相对 1% 门槛超出多少），并据此判定 #17 是"有余量"还是"贴边"；
③ 打印 `corrected` 相对 `base` 的逐字段变化量（激光功率/扫描速度/…）：**若全字段变化 <0.5%
   却仍让代价降 >1%，那说明代价面对该方向的灵敏度极低**，#17 的"杠杆不足"就要换个方向重述。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))

import jax.numpy as jnp  # noqa: E402
import test_monitoring as tm  # noqa: E402
import amforge as af  # noqa: E402

t0 = time.time()
geo = tm._tiny_geo()
base = af.ProcessPlan.uniform(n_layers=3, laser_power=120.0, scan_speed=1.5)
ms = tm.MonitoringState(defect_probs=jnp.array([0.8, 0.0, 0.0]))
clp = tm.monitor_correct_inverse(monitoring=ms, process=base,
                              params={"n_grid": 6, "max_layers": 3,
                                      "n_sub_cp": 8, "micro": {}},
                              geo=geo, n_steps=10, learning_rate=0.08)
assert isinstance(clp, tm.ClosedLoopPlan) and clp.controller == "inverse"
cost_base = tm._smooth_cost_of(geo, base)
cost_corr = tm._smooth_cost_of(geo, clp.corrected)
gain = (cost_base - cost_corr) / cost_base * 100.0

print(f"cost_base = {cost_base:.6e}   cost_corr = {cost_corr:.6e}")
print(f"改善 = {gain:.3f}%   门槛 = 1.000%   余量 = {gain - 1.0:.3f} 个百分点")
print(f"① 断言复现 {'PASS' if cost_corr < cost_base * 0.99 else 'FAIL（探针与测试树不同步）'}")
print(f"② {'贴边（余量 <1 个百分点）' if gain - 1.0 < 1.0 else '有余量'}：gain={gain:.3f}%")

print("③ inverse 控制器实际动了哪些工艺量：")
worst = 0.0
for f in ("laser_power", "scan_speed", "hatch_spacing", "layer_thickness",
          "beam_radius", "absorption", "preheat_temp", "scan_angle"):
    b = jnp.atleast_1d(getattr(base, f))
    c = jnp.atleast_1d(getattr(clp.corrected, f))
    if b.shape != c.shape:
        c = jnp.resize(c, b.shape)
    d = float(jnp.max(jnp.abs(c - b)))
    rel = d / max(float(jnp.max(jnp.abs(b))), 1e-30) * 100.0
    worst = max(worst, rel) if d > 0 else worst
    flag = "未动" if d == 0 else f"Δmax={d:.4g}  相对 {rel:.3f}%"
    print(f"  {f:<15} {str(b[:3]):<28} -> {str(c[:3]):<28} {flag}")
print(f"  ⇒ 最大相对变化 {worst:.3f}%")
print(f"墙钟 {time.time() - t0:.1f} s")
