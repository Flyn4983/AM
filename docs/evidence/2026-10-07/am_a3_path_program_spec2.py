"""A3/#18 规格测量 v2（只读、stdlib、不碰 src/、不给冻结的验收环境装包）。

v1（`am_a3_path_program_spec.log`）里我的**分段假设错了两条**，照实更正后重测：
  ① 我假设"道 = 等 x 的竖直线段，跳段是跨 x 的空程"，用"单步位移 > 5×中位步长"切段。
     实测该判据把**开光道**和**关光空程**混在一起切，得到 10 段里夹着 1353/1186 µm 的怪段，
     而真正的分段依据**就在文件里**：功率列有 **850 个 0 W 点**（占 29.79 ms 的 28.5%）
     ⇒ **门控是 CSV 的一等信息**，不该用几何启发式去猜。v2 一律按 `power>0 / power==0` 切。
  ② 我在 P2 里预备的结论"CSV 只含开光轨迹、关光不在这份导出里"被自己的数据**打伪**，
     这是好事：#18 的门控不必另找来源。

跑前登记的证伪判据（v2）：
  Q1 按功率切段后，on-run 数量应收敛到一个清楚的"道数"（README 声称 18 道/策略），
     且每个 on-run 内部速度基本恒定（≈960 mm/s）。若速度在 on-run 内大幅起伏 ⇒ 说明
     时间轴不只是"位置+速度"，还含加减速，接口必须**保留时间轴**而不是重推速度。
  Q2 off-run（功率 0）里若仍有位移 ⇒ 那就是 skywriting/空程，报告其时长与位移分布；
     若 off-run 全程不动 ⇒ 关光只是暂停，#18 的"时间照走、功率为 0"仍要照做。
  Q3 相邻 on-run 的**横向间距**应给出 hatch（README 声称 110 µm）；报告实测间距分布。
  Q4 converging 与 diverging 的逐道长度应呈**镜像/单调**特征（梯形扫描的几何定义），
     并打印前 3 道给 §26.7 N2 判据②当数字靶子。
  Q5 记账核对（不是用解析解当仿真）：`285 W / 960 mm/s = 0.297 J/mm`；
     Σ(on-run 时长)×285 W 应 ≈ Σ(on-run 长度)×0.297 J/mm，两者差 <1% 才算自洽。
  Q6 `Measurements.xlsx` 是 zip：用 stdlib 读 `xl/workbook.xml` + `sharedStrings.xml`，
     定位"第 N 道熔池宽/面积"的表（sheet 名 + 表头 + 行数），**否则 A3 的被测量还没落地**。
  Q7 Sheet1 的数值单元格应 ≥36（18 道 × 至少 2 列），且能同时容纳"宽度(µm)"与"面积(µm²)"两组、
     按操作员/重复分列 ⇒ 打印整张网格作为**对比靶表的留档**；本轮只登记结构，不做对比。

登记时序（照实说，避免"先跑后写判据"的嫌疑）：Q1–Q6 在本文件**第一次执行之前**写死；第一次运行的
Q5 报了 100201% 的荒谬偏差，查明是**我的单位错误**（µm 当 mm、J/m 与 J/mm 混用），判据本身没改，
只改实现；**Q7 是在 Q6 只打印了共享字符串之后补的**（补的是"把整张网格打出来"，不是改结论的门槛）。
带单位错误的第一次运行留档为 `ABORTED_am_a3_path_program_spec2_unitbug.log`。
"""
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[3] / "_refs/benchmarks/nist_mds2-3662"
CSV_DIR = BASE / "Scan Strategy Data"
assert BASE.is_dir(), f"基准目录不存在: {BASE}"
print(f"数据源: {CSV_DIR}", flush=True)


def load(path):
    with open(path, encoding="utf-8", errors="ignore") as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    rows = np.array([[float(v) for v in ln.split(",")] for ln in lines[1:]], dtype=np.float64)
    return rows  # (x µm, y µm, P W, t s)


def runs(mask):
    """把布尔序列切成 `(值, 起, 止)` 闭区间段（止-起+1 = 该段点数）。"""
    m = np.asarray(mask, dtype=bool)
    change = np.nonzero(np.diff(m.astype(np.int8)))[0] + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change - 1, [len(m) - 1]))
    return [(bool(m[s]), int(s), int(e)) for s, e in zip(starts, ends)]


def describe(name):
    r = load(CSV_DIR / name)
    x, y, p, t = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
    on = p > 0.0
    segs = runs(on)
    print(f"\n=== {name} ===", flush=True)
    print(f"  点数={len(r)} 总时长={t[-1] * 1e3:.2f} ms 时间步={np.diff(t).min() * 1e6:.0f} µs "
          f"| 开光 {int(on.sum())} 点 / 关光 {int((~on).sum())} 点"
          f"（关光占 {(~on).sum() / len(r) * 100:.1f}%）", flush=True)
    ons = [(a, b) for f, a, b in segs if f]
    offs = [(a, b) for f, a, b in segs if not f]
    print(f"  Q1 on-run 数={len(ons)}  off-run 数={len(offs)}", flush=True)
    lens, durs, speeds, arcs = [], [], [], []
    for k, (a, b) in enumerate(ons):
        ln = float(np.hypot(x[b] - x[a], y[b] - y[a]))
        # 折线真实弧长（含 on-run 内的拐弯）
        arc = float(np.sum(np.hypot(np.diff(x[a:b + 1]), np.diff(y[a:b + 1]))))
        du = float(t[b] - t[a])
        lens.append(ln)
        arcs.append(arc)
        durs.append(du)
        speeds.append(arc * 1e-3 / du if du > 0 else np.nan)
        if k < 20:
            print(f"    on#{k+1:2d} 起=({x[a]:.0f},{y[a]:.0f}) 止=({x[b]:.0f},{y[b]:.0f}) "
                  f"直距={ln:7.1f} 弧长={arc:7.1f} µm 时长={du * 1e3:6.3f} ms "
                  f"v={speeds[-1]:6.0f} mm/s 功率={p[a:b + 1].max():.0f} W", flush=True)
    sp = np.array(speeds)
    print(f"  Q1 on-run 速度 p10/p50/p90 = {np.percentile(sp, 10):.0f}/"
          f"{np.percentile(sp, 50):.0f}/{np.percentile(sp, 90):.0f} mm/s "
          f"⇒ 恒速 {'PASS' if np.percentile(sp, 90) / np.percentile(sp, 10) < 1.5 else '不恒速（含加减速，接口须保留时间轴）'}",
          flush=True)
    print(f"  Q2 off-run 位移：", flush=True)
    off_disp, off_dur = [], []
    for k, (a, b) in enumerate(offs[:20]):
        ln = float(np.hypot(x[b] - x[a], y[b] - y[a]))
        off_disp.append(ln)
        off_dur.append(float(t[b] - t[a]))
        print(f"    off#{k+1:2d} 起=({x[a]:.0f},{y[a]:.0f}) 止=({x[b]:.0f},{y[b]:.0f}) "
              f"位移={ln:7.1f} µm 时长={off_dur[-1] * 1e3:6.3f} ms", flush=True)
    if off_disp:
        print(f"     ⇒ off 位移中位={np.median(off_disp):.1f} µm，非零位移段数="
              f"{int((np.array(off_disp) > 1.0).sum())}/{len(offs)} ⇒ "
              f"{'存在空程移动（skywriting 类）' if np.median(off_disp) > 1.0 else '关光=原地暂停'}",
              flush=True)
    # Q3 相邻 on-run 的横向间距：按各 on-run 中点连线在"排布方向"上的投影
    mid = np.array([[(x[a] + x[b]) / 2, (y[a] + y[b]) / 2] for a, b in ons])
    if len(mid) > 2:
        step = np.diff(mid, axis=0)
        dist = np.hypot(step[:, 0], step[:, 1])
        print(f"  Q3 相邻 on-run 中点距离 min/med/max = {dist.min():.1f}/{np.median(dist):.1f}/"
              f"{dist.max():.1f} µm（README 声称 hatch 110 µm）", flush=True)
    tot_len = float(np.sum(lens))
    tot_arc = float(np.sum(arcs))
    tot_on = float(np.sum(durs))
    v_mean = tot_arc * 1e-3 / tot_on if tot_on > 0 else float("nan")
    print(f"  Σ直距={tot_len:.1f} µm  Σ弧长={tot_arc:.1f} µm  Σon 时长={tot_on * 1e3:.2f} ms  "
          f"弧长平均 v={v_mean:.1f} mm/s", flush=True)
    # Q5：CSV 自身的时间轴（Σt·P）与"按弧长×名义线能量"是否自洽。
    # 名义线能量取 singleTrack.csv 的 285 W / 961.5 mm/s（同一份数据里的实测，不是 README 口头值）。
    # 单位纪律（第一版在此栽了：µm 当 mm 换算，报出 100201% 的假偏差）：
    #   285 W ÷ 0.9615 m/s = 296.4 J/m；弧长 µm → m 要乘 1e-6。
    v_nom_single = 2.0e-3 / 2.08e-3  # m/s
    line_energy = 285.0 / v_nom_single  # J/m
    e_by_time = 285.0 * tot_on
    e_by_len = tot_arc * 1e-6 * line_energy
    dev = abs(e_by_time - e_by_len) / e_by_time
    print(f"  Q5 记账：Σt·P={e_by_time:.4f} J  vs  Σ弧长({tot_arc:.1f} µm)×{line_energy:.1f} J/m="
          f"{e_by_len:.4f} J  偏差={dev * 100:.2f}% ⇒ "
          f"{'PASS（时间轴与名义速度自洽）' if dev < 0.01 else '不满足（含加减速/空程，接口须保留时间轴而不是重推速度）'}",
          flush=True)
    return dict(name=name, n_on=len(ons), lens=np.array(lens), arcs=np.array(arcs),
                durs=np.array(durs), mid=mid, tot_on=tot_on)


res = [describe(n) for n in ("scanStrategyConverging.csv", "scanStrategyDiverging.csv")]

print("\n=== Q4：两种策略的逐道长度是否镜像/单调 ===", flush=True)
a, b = res[0]["arcs"], res[1]["arcs"]
n = min(len(a), len(b))
print(f"  converging 前 8 道 = " + " ".join(f"{v:.1f}" for v in a[:8]), flush=True)
print(f"  diverging  前 8 道 = " + " ".join(f"{v:.1f}" for v in b[:8]), flush=True)
if n:
    corr = float(np.corrcoef(a[:n], b[:n])[0, 1]) if n > 2 else float("nan")
    print(f"  同序相关系数={corr:+.3f}（镜像应为负、梯形单调应先增后减）", flush=True)

print("\n=== Q6：Measurements.xlsx 内部结构（stdlib zipfile）===", flush=True)
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
with zipfile.ZipFile(BASE / "Measurements.xlsx") as z:
    names = z.namelist()
    print(f"  条目 {len(names)} 个；sheets/字符串相关: "
          + ", ".join(n for n in names if "sheet" in n or "Shared" in n or "shared" in n), flush=True)
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    rid2t = {e.get("Id"): e.get("Target") for e in rels}
    sheets = [(e.get("name"), rid2t.get(e.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")))
              for e in wb.findall("m:sheets/m:sheet", NS)]
    print(f"  Q6 sheet 数={len(sheets)}", flush=True)
    for nm, tgt in sheets:
        print(f"    - {nm}  ->  {tgt}", flush=True)
    shared = []
    if "xl/sharedStrings.xml" in names:
        ss = ET.fromstring(z.read("xl/sharedStrings.xml"))
        for si in ss.findall("m:si", NS):
            shared.append("".join(t.text or "" for t in si.findall(".//m:t", NS)))
    print(f"  共享字符串 {len(shared)} 条；前 40 条 =", flush=True)
    print("    " + " | ".join(shared[:40]), flush=True)

    print("\n=== Q7：Sheet1 数值网格（A3 的被测量到底长什么样）===", flush=True)

    def cidx(ref):
        n = 0
        for ch in re.match(r"[A-Z]+", ref).group():
            n = n * 26 + (ord(ch) - 64)
        return n - 1

    sh = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    grid, nnum = {}, 0
    for row in sh.findall("m:sheetData/m:row", NS):
        for c in row.findall("m:c", NS):
            v = c.find("m:v", NS)
            if v is None or not v.text:
                continue
            if c.get("t") == "s":
                val = shared[int(v.text)]
            else:
                val = v.text
                try:
                    float(val)
                    nnum += 1
                except ValueError:
                    pass
            grid[(int(row.get("r")), cidx(c.get("r")))] = val
    rws = sorted({k[0] for k in grid})
    cls = sorted({k[1] for k in grid})
    print(f"  Q7 网格 = {len(rws)} 行 × {len(cls)} 列，数值单元格 {nnum} 个 ⇒ "
          f"{'可读作对比靶表' if nnum >= 36 else 'FAIL：数值太少，A3 被测量未落地'}", flush=True)
    for rr in rws[:40]:
        print(f"    r{rr:2d}: " + " | ".join(str(grid.get((rr, cc), ""))[:14]
                                             for cc in cls[:16]), flush=True)

