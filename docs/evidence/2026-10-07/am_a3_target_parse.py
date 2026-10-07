"""N2.5 ①②③ —— 把 NIST mds2-3662 的 `Measurements.xlsx` 解析成 A3 对比用的**数值靶表**。

**布局是本轮实测出来的**（不是从 README 或表头名推的；列字母 0-based：A=0…T=19）：

    行 2/3    区块标题：C="Set 1"、H="Set 2"；其下一行 C/H="Operator 1"；行 24 H="Operator 2"
    行 4      C="Track #"、D="Melt Pool Width…"、F="Melt Pool Area…"；H="Track #"、I=宽度、O=面积
    行 5/26   列标签：Set1 为 D/E=F/G="Diverging/Converging"；
              Set2 为 I..N = D1,D2,**Divering(D2)**,D3,**Coverging(C1)**,Coverging(C2),C3；
                     O..T = 同序的面积六列
    **语义来自 README（`:90/:124/:126`，本轮读到）**：D1/D2/D3 与 C1/C2/C3 **不是三个方向**，
    而是同一物理量的**三次重复描图**（three repeats），且**两名操作员**各描一遍
    ⇒ 每个 (track, 量, 族∈{D,C}) 最多 **6** 个值；被测量是**末道**熔池的**最大宽度**与**顶面总面积**。
    行 6–14   Set 1（仅 Operator 1，**稀疏**）：C=Track#，D=宽 D，E=宽 C，F=积 D，G=积 C
              实测：r6 只有 Track 1 无测量；r7–r13（道 2–8）**只有 Diverging**；r14（道 18）四格齐
    行 6–23   Set 2 / Operator 1 的 Track 1–18
    行 27–44  Set 2 / Operator 2 的 Track 1–18
    行 46     S 列有文本 "Outliers" ⇒ 作者把离群值标在**面积 C2 那一列**（是文本，不进 495 个数值格）

跑前登记的判据（§26.7 N2.5，代码之前写死）：
  ① 数值单元格计数 == 495；
     ①′（本轮加严）测量格 + Track# 格 == 全表数值格 ⇒ 不存在**未建模的数值区**；
  ② 缺失格**必须是 NaN**（不是 0、不是整行丢），并**逐个点名**抽查稀疏位置；
     ②′（本轮加严）Set1 的**有值图案**必须与 README 影像清单逐道吻合 ⇒ 位置映射的独立交叉验证；
  ③ 每个 (track, 量, 方向) 给出中位数 + 极差 ⇒ 极差是后续偏差的**分母**；
     **更正（读到 README 的 repeats 语义后）**：分母按**族**（3 重复 × 2 操作员，n≤6）求，
     原先按单列跨操作员 n=2 会**低估**分母；门槛只加严、未放松；
  ④ 表头拼写错在**固定位置**被断到（`Divering`/`Coverging`），命中即证明"按位置取数"没错位；
  ⑤ 产物落盘（tidy CSV + 本日志），#10 只引用产物、不重复解析。
"""
from __future__ import annotations

import csv
import math
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2] / "_refs" / "benchmarks" / "nist_mds2-3662"
XLSX = BASE / "Measurements.xlsx"
OUT_CSV = HERE / "am_a3_target_table.csv"
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

fails: list[str] = []


def check(tag: str, ok: bool, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag} — {detail}", flush=True)
    if not ok:
        fails.append(tag)


def col_index(ref: str) -> int:
    n = 0
    for ch in re.match(r"[A-Z]+", ref).group():
        n = n * 26 + (ord(ch) - 64)
    return n - 1


with zipfile.ZipFile(XLSX) as z:
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        ss = ET.fromstring(z.read("xl/sharedStrings.xml"))
        shared = ["".join(t.text or "" for t in si.findall(".//m:t", NS))
                  for si in ss.findall("m:si", NS)]
    sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))

grid: dict[tuple[int, int], str] = {}
numeric_cells: list[tuple[int, int, float]] = []
for row in sheet.findall("m:sheetData/m:row", NS):
    r = int(row.get("r"))
    for c in row.findall("m:c", NS):
        v = c.find("m:v", NS)
        if v is None or not v.text:
            continue
        cc = col_index(c.get("r"))
        if c.get("t") == "s":
            grid[(r, cc)] = shared[int(v.text)]
            continue
        grid[(r, cc)] = v.text
        try:
            numeric_cells.append((r, cc, float(v.text)))
        except ValueError:
            pass
NUM = {(r, c): val for r, c, val in numeric_cells}


def cell(r: int, c: int) -> float:
    """按位置取数：空格/非数 ⇒ NaN（判据②：绝不把缺失当 0）。"""
    return NUM.get((r, c), math.nan)


def text(r: int, c: int) -> str:
    return str(grid.get((r, c), ""))


def suffix_digit(label: str) -> str | None:
    """从 "Diverging (D3)" / "Divering (D2)" 这类标签里取**后缀数字**。
    取不到就返回 None（让判据 FAIL 而不是 IndexError 把整个记分段带走）。"""
    m = re.search(r"\(\s*[A-Za-z]\s*([0-9]+)\s*\)", label)
    return m.group(1) if m else None


print("=== 判据④：布局与表头在固定位置命中（拼写错只断言、不筛选）===", flush=True)
check("④ 区块标题位置", text(2, 2).startswith("Set 1") and text(2, 7).startswith("Set 2"),
      f"C2={text(2, 2)!r} H2={text(2, 7)!r}")
check("④ 操作员分区位置", text(3, 2).startswith("Operator 1") and text(3, 7).startswith("Operator 1")
      and text(24, 7).startswith("Operator 2"),
      f"C3={text(3, 2)!r} H3={text(3, 7)!r} H24={text(24, 7)!r}")
check("④ Track#/量纲列族位置", text(4, 2) == "Track #" and text(4, 7) == "Track #"
      and text(4, 3).startswith("Melt Pool Width") and text(4, 5).startswith("Melt Pool Area")
      and text(4, 8).startswith("Melt Pool Width") and text(4, 14).startswith("Melt Pool Area"),
      f"C4={text(4, 2)!r} D4={text(4, 3)!r:.18} F4={text(4, 5)!r:.18} I4={text(4, 8)!r:.18} O4={text(4, 14)!r:.18}")
typos = {"J5": text(5, 9), "L5": text(5, 11), "P5": text(5, 15), "R5": text(5, 17)}
check("④ 拼写错在固定位置命中（⇒ 取数未错位）",
      typos["J5"].startswith("Divering") and typos["L5"].startswith("Coverging")
      and typos["P5"].startswith("Divering") and typos["R5"].startswith("Coverging"),
      f"{typos}")
check("④ 方向列序 D1,D2,D3,C1,C2,C3（按后缀数字读，不按列名相等）",
      [suffix_digit(text(5, 8 + j)) for j in range(6)] == ["1", "2", "3", "1", "2", "3"],
      f"宽度列后缀 = {[(text(5, 8 + j), suffix_digit(text(5, 8 + j))) for j in range(6)]}")
outl = [(r, c) for (r, c) in grid if text(r, c).strip().lower().startswith("outlier")]
print(f"    作者标注位点 = {[(f'{chr(65 + c)}{r}', text(r, c)) for r, c in outl]}", flush=True)
check("④ Outliers 标注落在面积列区（O..T=14..19），故不进数值面",
      all(14 <= c <= 19 for r, c in outl) if outl else False,
      f"位点列号 = {[c for r, c in outl]}")

# ------------------------------------------------------------------ 取数
DIRS = ["D1", "D2", "D3", "C1", "C2", "C3"]
rows_out: list[dict] = []


def add(setname, op, track, qty, direction, r, c):
    rows_out.append(dict(set=setname, operator=op, track=track, quantity=qty,
                         direction=direction, value=cell(r, c),
                         src_cell=f"{chr(65 + c)}{r}"))


# Set 2（完整 6 方向 × 宽/积）：Op1 行 6–23，Op2 行 27–44；H=Track#，I..N 宽，O..T 积
for op, r0 in (("Operator1", 6), ("Operator2", 27)):
    for i in range(18):
        r = r0 + i
        trk = cell(r, 7)
        if not (1 <= trk <= 18) or int(trk) != i + 1:
            check(f"布局：Set2/{op} Track# 连续", False, f"H{r}={trk} 期望 {i + 1}")
            continue
        for j, d in enumerate(DIRS):
            add("Set2", op, i + 1, "width_um", d, r, 8 + j)
            add("Set2", op, i + 1, "area_um2", d, r, 14 + j)

# Set 1（仅 Operator1，宽/积各只有 Diverging、Converging 两列）：行 6–14
SET1_DIR = {3: "D", 4: "C", 5: "D", 6: "C"}
for r in range(6, 15):
    trk = cell(r, 2)
    if not (1 <= trk <= 18):
        continue
    for c in (3, 4, 5, 6):
        qty = "width_um" if c in (3, 4) else "area_um2"
        add("Set1", "Operator1", int(trk), qty, SET1_DIR[c] + "1", r, c)

n_nan = sum(1 for x in rows_out if math.isnan(x["value"]))
print("\n=== 判据①②：数值面与缺失面 ===", flush=True)
check("① 数值单元格总数 == 495", len(numeric_cells) == 495, f"实测 {len(numeric_cells)}")
n_track_cells = sum(1 for (r, c) in NUM if c in (2, 7))
check("①′ 数值面**自洽**：测量格 + Track# 格 == 全表数值格（不存在未建模的数值区）",
      len(rows_out) - n_nan + n_track_cells == len(numeric_cells),
      f"测量 {len(rows_out) - n_nan} + Track#（C/H 列）{n_track_cells} vs 全表 {len(numeric_cells)}")
check("② 存在缺失且以 NaN 呈现", n_nan > 0,
      f"取数 {len(rows_out)} 格：数值 {len(rows_out) - n_nan} / NaN {n_nan}")
sparse_known = [(6, 3), (6, 4), (6, 5), (6, 6),          # Set1 道 1：整行无测量
                (7, 4), (7, 6), (8, 4), (8, 6), (9, 4),   # Set1 道 2–4：只有 Diverging
                (13, 4), (13, 6)]                         # Set1 道 8：同上
bad = [f"{chr(65 + c)}{r}" for (r, c) in sparse_known if not math.isnan(cell(r, c))]
check("② 稀疏位点逐个点名必须为 NaN", not bad, f"抽查 {len(sparse_known)} 处，非 NaN = {bad or '无'}")
# 独立交叉验证：README 的影像清单（`:112-122`）列出的 Set1 样本 =
#   单道 1 张（**无表格值**）＋ diverging (2,3,4,5,6,7,8,18) ＋ converging 仅 18 道。
# 若按位置取数真的取对了，Set1 的**有值图案**必须与该清单逐道吻合。
s1D = sorted({x["track"] for x in rows_out if x["set"] == "Set1" and x["direction"].startswith("D")
              and not math.isnan(x["value"])})
s1C = sorted({x["track"] for x in rows_out if x["set"] == "Set1" and x["direction"].startswith("C")
              and not math.isnan(x["value"])})
check("②′ Set1 有值图案 == README 影像清单（位置映射的独立交叉验证）",
      s1D == [2, 3, 4, 5, 6, 7, 8, 18] and s1C == [18],
      f"D 有值道次={s1D}（README: 2,3,4,5,6,7,8,18）；C 有值道次={s1C}（README: 18）")
zeros = [x["src_cell"] for x in rows_out if x["value"] == 0.0]
check("② 零值不是缺失的替身", not zeros, f"value==0 的位点 = {zeros or '无'}")
w = [x["value"] for x in rows_out if x["quantity"] == "width_um" and not math.isnan(x["value"])]
a = [x["value"] for x in rows_out if x["quantity"] == "area_um2" and not math.isnan(x["value"])]
check("量级 sanity（宽 80–600 µm、积 5e3–5e5 µm²）",
      min(w) >= 80 and max(w) <= 600 and min(a) >= 5e3 and max(a) <= 5e5,
      f"宽 {min(w):.1f}–{max(w):.1f} µm；积 {min(a):.0f}–{max(a):.0f} µm²")

with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
    wtr = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
    wtr.writeheader()
    wtr.writerows(rows_out)
print(f"\n=== 判据⑤：产物 ===\n  {OUT_CSV.name} → {len(rows_out)} 行长表"
      f"（每格带 src_cell 可回溯原工作表）", flush=True)

# README 实测口径（本轮读到，写死于此处以免下轮再猜）：
#   `README.txt:124/126` ⇒ Set2 的 D1/D2/D3、C1/C2/C3 是**同一物理量的三次重复测量**（three repeats）；
#   `README.txt:90`     ⇒ 两名操作员各描一遍 ⇒ 每个 (track, 量, 族) 最多 **6** 个值。
# 本轮更正：判据③原先按**单个重复列**跨操作员求极差（n=2），会把偏差**分母系统性低估**；
# 现按「族」合并 3 重复 × 2 操作员（n≤6），并另报操作员间系统差。**门槛只加严、未放松。**
FAMS = {"D": ("D1", "D2", "D3"), "C": ("C1", "C2", "C3")}
GLYPH = {"width_um": "宽", "area_um2": "积"}


def vals_for(qty: str, fam: str, t: int, op: str | None = None) -> list[float]:
    return [x["value"] for x in rows_out
            if x["set"] == "Set2" and x["track"] == t and x["quantity"] == qty
            and x["direction"] in FAMS[fam] and (op is None or x["operator"] == op)
            and not math.isnan(x["value"])]


def med_rng(vals: list[float]) -> tuple[float, float, int]:
    if not vals:
        return math.nan, math.nan, 0
    s = sorted(vals)
    k = len(s)
    # 偶数个样本取两中间值之平均（标准中位数定义）；取"上中位"会把分母系统性抬高半格
    med = s[k // 2] if k % 2 else 0.5 * (s[k // 2 - 1] + s[k // 2])
    return med, ((max(vals) - min(vals)) / med * 100.0 if med else math.nan), k


print("\n=== 判据③：靶曲线（Set2 按族合并 3 重复 × 2 操作员；中位数 / 极差% 即偏差分母）===", flush=True)
print("  track |  宽D med rng%(n)  |  宽C med rng%(n)  |  积D med rng%(n)  |  积C med rng%(n)  | 算子间系统差% 宽D 宽C 积D 积C", flush=True)
for t in range(1, 19):
    cells, gaps = [], []
    for qty in ("width_um", "area_um2"):
        for fam in ("D", "C"):
            med, rng, n = med_rng(vals_for(qty, fam, t))
            cells.append(f"{med:8.1f} {rng:5.1f}%({n})" if n else f"{'--':>8} {'--':>5}(0)")
            m1, _, _ = med_rng(vals_for(qty, fam, t, "Operator1"))
            m2, _, _ = med_rng(vals_for(qty, fam, t, "Operator2"))
            gaps.append(f"{abs(m2 - m1) / ((m1 + m2) / 2) * 100:4.1f}" if m1 and m2 else "  --")
    print(f"  {t:5d} | " + " | ".join(cells) + " | " + " ".join(gaps), flush=True)

print("\n=== 极差分布（= A3 仿真偏差的分母；须与仿真偏差同图才许声称看到物理）===", flush=True)
for qty in ("width_um", "area_um2"):
    rng_sp, op_gap = [], []
    for t in range(1, 19):
        for fam in ("D", "C"):
            _, r, n = med_rng(vals_for(qty, fam, t))
            if n >= 2 and not math.isnan(r):
                rng_sp.append(r)
            m1, _, _ = med_rng(vals_for(qty, fam, t, "Operator1"))
            m2, _, _ = med_rng(vals_for(qty, fam, t, "Operator2"))
            if m1 and m2:
                op_gap.append(abs(m2 - m1) / ((m1 + m2) / 2) * 100.0)
    for tag, sp in (("合并 6 值极差%", rng_sp), ("操作员间系统差%", op_gap)):
        sp.sort()
        if sp:
            print(f"  {qty} {tag}: 组数={len(sp)} p50={sp[len(sp) // 2]:.1f} "
                  f"p90={sp[int(len(sp) * 0.9)]:.1f} max={sp[-1]:.1f}", flush=True)

print("\n=== 单调性/饱和（趋势是被测量，单点不是）===", flush=True)
for qty in ("width_um", "area_um2"):
    for fam in ("D", "C"):
        med = [med_rng(vals_for(qty, fam, t))[0] for t in range(1, 19)]
        fin = [(i + 1, m) for i, m in enumerate(med) if not math.isnan(m)]
        if not fin:
            continue
        turn = next((t for (t, m) in fin if m >= 0.98 * max(x[1] for x in fin) and t >= 10), None)
        print(f"  {qty} {fam}: 道1={fin[0][1]:.1f} 道9={med[8]:.1f} "
              f"道18={fin[-1][1]:.1f}；首个『接近饱和』道次={turn}", flush=True)

print("\n=== 噪声地板探针（**不是判据**，只是给 A3 定标尺）===", flush=True)
print("  道 1 只有一个熔池，『发散/收敛』两组理应量到同一个对象 ⇒ 其差是噪声地板的**下界估计**。", flush=True)
for qty in ("width_um", "area_um2"):
    m_d, _, n_d = med_rng(vals_for(qty, "D", 1))
    m_c, _, n_c = med_rng(vals_for(qty, "C", 1))
    print(f"  {qty} 道1：D1={m_d:.1f} vs C1={m_c:.1f} ⇒ 相对差 "
          f"{abs(m_c - m_d) / ((m_c + m_d) / 2) * 100:.1f}%"
          f"（两组测量是否严格同一定义，数据集内**未证实** ⇒ 登记为待查，不当作已证事实）", flush=True)

print("\n=== 记分 ===", flush=True)

print(f"  判据未达项 = {fails or '无'}", flush=True)
print("  ⇒ " + ("N2.5 闭合，#10 的对比可用本靶表（仍受 N0/#19、N2/#18 前置约束）"
                if not fails else "N2.5 未闭合，#10 继续被阻塞"), flush=True)
