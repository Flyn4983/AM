"""T3 普查：`长度/间距 -> 整数个数` 的表达在**同一物理参数的不同浮点拼写**下会不会翻一位？

D4/T1 只给 `_grid_axes` 加了容差吸附（`geometry.py:127`）。本探针按 2026-10-07 21:40
的全树普查清单，逐条把同一表达式在 4 种等价拼写 + 1 个精确有理数真值下求值，
统计**翻位数**与**与真值的 off-by-one**。纯算术，不导入 amforge（表达式逐条照抄，
文件:行号见 SITES），另附 `_grid_axes` 已修式作为**正对照**（探针必须看得见翻转）。

预注册判据（跑前写死）：
  C1 探针有效：至少一个真实参数组合在未吸附表达式上翻转，否则探针是空转。
  C2 正对照：`_grid_axes` 已吸附式在同一组网格上翻转数 = 0。
  C3 真值对账：逐条报"与 Fraction 精确真值不等"的次数（off-by-one 的另一半证据）。
"""
import math
from fractions import Fraction

SPELLS = ("字面量 e-6", "µm/1e6", "µm*1e-6", "(µm/1000)/1000")


def val(um, spelling):
    if spelling == "字面量 e-6":
        return float(f"{um}e-6")
    if spelling == "µm/1e6":
        return um / 1e6
    if spelling == "µm*1e-6":
        return um * 1e-6
    return (um / 1000.0) / 1000.0


SITES = [
    ("geometry.py:127 _grid_axes（D4 已吸附，正对照）",
     lambda L, s: int(math.ceil(L / s - 1e-9 * max(1.0, abs(L / s))))),
    ("core/contracts.py:190 layer_count",
     lambda L, s: max(1, int(math.ceil(L / s)))),
    ("geometry.py:415 add_baseplate n_add",
     lambda L, s: max(1, int(math.ceil(L / s)))),
    ("geometry.py:449 layer_z_heights n",
     lambda L, s: max(1, int(math.ceil(L / s)))),
    ("geometry.py:521 crop_to_part pad",
     lambda L, s: int(math.ceil(L / s))),
    ("process.py:501 n_line",
     lambda L, s: max(1, int(math.ceil(L / max(s, 1e-9))))),
    ("powder.py:238-240 晶格 nx/ny/nz",
     lambda L, s: max(1, int(math.floor(L / s)) + 1)),
    ("powder.py:391-396 DEM nx/ny",
     lambda L, s: max(4, int(math.floor(L / s)) + 1)),
    ("gui/preproc.py:128 n_lines",
     lambda L, s: max(1, int(math.floor(L / s)) + 1)),
]

# 真实参数对（长度 µm, 间距 µm）：试片/零件尺寸 × 层厚/体素/粒径/间距
GRID = [
    (400.0, 40.0), (400.0, 30.0), (400.0, 20.0), (400.0, 12.5), (400.0, 50.0),
    (600.0, 30.0), (600.0, 25.0), (600.0, 60.0), (600.0, 40.0),
    (1200.0, 60.0), (1200.0, 30.0), (1200.0, 12.5), (1200.0, 100.0),
    (10000.0, 50.0), (10000.0, 25.0), (10000.0, 12.5), (10000.0, 70.0),
    (2500.0, 50.0), (2500.0, 15.0), (1600.0, 40.0), (1600.0, 20.0),
    (200.0, 20.0), (200.0, 25.0), (300.0, 30.0), (300.0, 75.0), (800.0, 100.0),
]

print(f"参数对 {len(GRID)} 组 × 表达式 {len(SITES)} 条 × 拼写 {len(SPELLS)} 种\n")
hdr = f"{'表达式':52s} {'翻转组数':>10s} {'off-by-one 组数':>16s}"
print(hdr + "\n" + "-" * len(hdr))
totals = {}
for name, fn in SITES:
    flips = wrong = 0
    examples = []
    for L_um, s_um in GRID:
        got = []
        for sp in SPELLS:
            try:
                got.append(fn(val(L_um, sp), val(s_um, sp)))
            except Exception:
                got.append(None)
        truth = fn(Fraction(str(L_um)) / Fraction(10 ** 6),
                   Fraction(str(s_um)) / Fraction(10 ** 6))
        if len(set(got)) > 1:
            flips += 1
            if len(examples) < 2:
                examples.append((L_um, s_um, sorted(set(got))))
        if any(g is not None and g != truth for g in got):
            wrong += 1
    totals[name] = (flips, wrong)
    print(f"{name:52s} {flips:10d} {wrong:16d}")
    for L, s, st in examples:
        print(f"{'':52s}   例：{L:g}µm / {s:g}µm -> {st}")

FAILS = []
if totals[SITES[0][0]][0] != 0:
    FAILS.append(f"C2 正对照失败：已吸附的 _grid_axes 仍有 {totals[SITES[0][0]][0]} 组翻转 ⇒ 探针或吸附有问题")
if all(v[0] == 0 for k, v in totals.items() if k != SITES[0][0]):
    FAILS.append("C1 探针空转：未吸附表达式在真实参数对上一次都没翻转")
unfixed = [k for k, v in totals.items() if k != SITES[0][0] and v[0] > 0]
print(f"\n未吸附且实测会翻转的站点 = {len(unfixed)} 条")

print("\n=== 全部翻转参数对（ceil 族 / floor 族各一次列出，比值与真值照实）===")
for L_um, s_um in GRID:
    exact = Fraction(str(L_um)) / Fraction(str(s_um))     # 精确比值（有理数真值）
    vals = [val(L_um, sp) / val(s_um, sp) for sp in SPELLS]
    ceils = sorted({int(math.ceil(v)) for v in vals})
    floors = sorted({int(math.floor(v)) + 1 for v in vals})
    if len(ceils) > 1 or len(floors) > 1:
        print(f"  {L_um:8.1f}µm / {s_um:6.1f}µm = 精确 {exact} -> 浮点比值 "
              f"{[repr(v) for v in vals]} | ceil 族 {ceils}（真值 {math.ceil(exact)}）"
              f"| floor+1 族 {floors}（真值 {math.floor(exact) + 1}）")

print("FAILS:", FAILS if FAILS else "无（但结论是『T1 同型缺陷仍散布在未吸附站点』，需登记任务）")
raise SystemExit(1 if FAILS else 0)
