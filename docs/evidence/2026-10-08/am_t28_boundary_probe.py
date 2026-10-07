"""#28 换数取证：`test_boundary.py::test_convection_removes_energy` 的 docstring 表需要 **③ 代（#23 之后）**
的绝对读数，而档案里只有 Δmean/Δpeak（`am_t27_convection_peak_probe.log` 的 [new/on] 组）⇒ 绝对值缺格。
**缺格不填空**：本探针去量，不改被测函数、不改判据、不动 `tests/` 的任何 assert。

夹具**从测试模块本身导入**（`_box3d(size_mm=0.4, spacing_um=80.0)` / `_plan()` / 同一 `solid` 口径）
⇒ 同构性由构造保证，不靠"照着重写一遍"。

跑前登记的判据（不许跑后改）：
P1 **锚点逐位对照**：无对流的峰值 `peak0` 必须与 `am_t27_convection_peak_probe.log` U0b 的
   new 口径锚点 `2937.060997302282`（pytest 也打印过同一串）**逐位相等**。不等 ⇒ 夹具/口径不同构，
   本轮作废、先查码（这是本探针的**阳性对照**：它能失败，也能证明"我在量同一个东西"）。
P2 **Δmean 对照**：三档 h 的 `mean0 − means[i]` 必须与档案 [new/on] 的 −7.27 / −71.38 / −755.93 K
   在 ±0.01 K 内（档案只到小数 2 位 ⇒ 更严的逐位要求不成立，此处按打印精度对齐）。
P3 **Δpeak 对照**：三档 `Δpeak%` 必须与档案的 +0.042% / +0.261% / −0.963% 在 ±0.005pt 内。
P4 **设备行写进日志本体**（长期规则）；本跑**CPU 钉住**（docstring 声明的取数条件＋与 pytest 同设备）。
P5 输出＝**绝对读数**（mean0、means[3]、peak0、peaks[3] 的 `.17g`）＋两条现行守护的通过与否
   （① `abs(peaks[1]-peak0)/peak0 < 0.02` ② `peaks[2] < peak0 - 0.02*peak0`）。**只报，不改判据**：
   第②条本轮预期为 False（＝#27 的红灯，归因与处置已在 `am_t27_convection_peak_probe.log` 登记完）。
"""
import importlib.util

import numpy as np

R = "/home/shy/桌面/2026年开发-显式冲击有限元-物质点法-sph-fvm+am/AM-Qoder"
spec = importlib.util.spec_from_file_location("tb", f"{R}/tests/test_boundary.py")
tb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tb)

from amforge.core.contracts import solid_mask  # noqa: E402
from amforge.thermal_enthalpy import solve_enthalpy_thermal  # noqa: E402

try:
    import jax
    print(f"jax.devices() = {jax.devices()}", flush=True)
    print(f"default_backend = {jax.default_backend()}", flush=True)
except Exception as e:  # pragma: no cover
    print(f"device line FAILED: {type(e).__name__}: {e}", flush=True)

part = tb._box3d(size_mm=0.4, spacing_um=80.0)
plan = tb._plan()
solid = np.asarray(solid_mask(np.asarray(part.sdf)) > 0.5)
res0 = solve_enthalpy_thermal(geometry=part, process=plan, params={"material": "316L"})
mean0 = float(np.mean(np.asarray(res0.final_temperature)[solid]))
peak0 = float(np.max(np.asarray(res0.peak_temperature)))

ARC_PEAK0 = "2937.060997302282"
ARC_DMEAN = (-7.27, -71.38, -755.93)
ARC_DPEAK = (0.042, 0.261, -0.963)

fails = []
print(f"peak0={peak0!r}  ==锚点 {ARC_PEAK0}? {repr(peak0) == ARC_PEAK0}  (P1)", flush=True)
if repr(peak0) != ARC_PEAK0:
    fails.append("P1 锚点不逐位 ⇒ 夹具不同构，本轮作废")
print(f"mean0={mean0!r}  n_solid={int(solid.sum())}  grid={list(np.asarray(part.sdf).shape)}", flush=True)

means, peaks = [], []
for h in (200.0, 2000.0, 20000.0):
    strong = tb.BoundaryCollection(
        bcs=tuple(tb.BoundaryCondition("convection", f, h, 293.0)
                  for f in ("+Z", "+X", "-X", "+Y", "-Y")), ic=None)
    res = solve_enthalpy_thermal(geometry=part, process=plan,
                                 params={"boundary_conditions": strong, "material": "316L"})
    Tf = np.asarray(res.final_temperature)
    means.append(float(np.mean(Tf[solid])))
    peaks.append(float(np.max(np.asarray(res.peak_temperature))))

for i, h in enumerate((200.0, 2000.0, 20000.0)):
    dm = means[i] - mean0
    dp = (peaks[i] - peak0) / peak0 * 100.0
    ok2 = abs(dm - ARC_DMEAN[i]) <= 0.01
    ok3 = abs(dp - ARC_DPEAK[i]) <= 0.005
    print(f"h={h:7.0f}  mean={means[i]!r}  Δmean={dm:8.2f}K  (P2 {'✓' if ok2 else '✗'})   "
          f"peak={peaks[i]!r}  Δpeak={dp:+.3f}%  (P3 {'✓' if ok3 else '✗'})", flush=True)
    if not ok2:
        fails.append(f"P2 h={h:.0f} Δmean {dm:.2f} 与档案 {ARC_DMEAN[i]} 不符")
    if not ok3:
        fails.append(f"P3 h={h:.0f} Δpeak {dp:+.3f}% 与档案 {ARC_DPEAK[i]} 不符")

g1 = abs(peaks[1] - peak0) / peak0 < 0.02
g2 = peaks[2] < peak0 - 0.02 * peak0
print(f"P5 现行守护①（h=2000 不动峰值，abs<2%）= {g1}    守护②（h=20000 压峰 ≥2%）= {g2}"
      f"   ⇒ ② 为 False 即 #27 红灯，本探针**不改判据**、只登记", flush=True)
print(f"FAILS: {fails if fails else '无'}", flush=True)
print("probe exit=%d" % (1 if fails else 0), flush=True)
