"""A3/#18 规格测量（只读，不碰 src/）：NIST mds2-3662 的扫描策略 CSV 到底给了什么？

为什么现在做：#18 的验收判据（§26.7 N2）写着"复算 `t_exposure`、`path_length` 与 CSV 时间轴一致"
"converging 前 3 道逐道长度/时长与 CSV 相符"——**这些靶子数字现在还不存在**，得先从 CSV 里量出来，
否则实现完无法记分。另外要诚实回答一个问题：**现有热源接口能不能只靠 CSV 就接上**。

跑前登记的证伪判据：
  P1 时间列是 10 µs 等间隔、单调、从 0 起 ⇒ 可直接作为一等时间轴（无需我方再推 `t=path/v`）。
  P2 功率列**是否出现 0**：README 说该基准 skywriting=开。若 CSV 全程 285 W 而无 0 段，
     ⇒ **CSV 只含开光轨迹，关光间隙不在这份导出里**，#18 的门控必须另找来源并写明证据缺口。
  P3 轨迹可分解成"沿 y 的直线段 + 跨 x 的跳段"，报告段数/每段长度/相邻段间距（基准声称 18 道）。
  P4 名义速度 = Σ段长/Σ时间 应接近 README 的 960 mm/s；同时报告**逐段速度分布**（若非均匀要照实说）。
  P5 converging 与 diverging 的逐道长度应呈镜像/单调特征 ⇒ 打印前 3 道作为 N2 判据②的数字靶子。
  P6 `Measurements.xlsx` 里应能定位"第 N 道熔池宽/面积"的表（sheet + 列 + 操作员/重复编码），
     否则 A3 的对比目标本身还没落地，S3 要再排后。
"""
import re
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[3] / "_refs/benchmarks/nist_mds2-3662"
CSV_DIR = BASE / "Scan Strategy Data"
assert BASE.is_dir(), f"基准目录不存在: {BASE}"
print(f"数据源: {CSV_DIR}", flush=True)


def load(path):
    with open(path, encoding="utf-8", errors="ignore") as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    head = lines[0]
    rows = np.array([[float(v) for v in ln.split(",")] for ln in lines[1:]], dtype=np.float64)
    return head, rows


def segments(rows, tol=1e-9):
    """把轨迹切成"同一条直线道"：连续点里 x 不变（或 y 不变）的极大道。"""
    xy = rows[:, :2]
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    # 跳段判据：单步位移明显大于该文件中位数步长
    med = np.median(d[d > 0])
    jump = d > max(5.0 * med, med + 50.0)
    bounds = [0] + [int(i) + 1 for i in np.nonzero(jump)[0]] + [len(xy)]
    segs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 2:
            continue
        p, q = xy[a], xy[b - 1]
        segs.append(dict(i=a, n=b - a, length=float(np.hypot(*(q - p))),
                         p=tuple(p), q=tuple(q),
                         dur=float(rows[b - 1, 3] - rows[a, 3])))
    return segs, d


for name in ("singleTrack.csv", "scanStrategyConverging.csv", "scanStrategyDiverging.csv"):
    path = CSV_DIR / name
    head, rows = load(path)
    t, pw = rows[:, 3], rows[:, 2]
    dt = np.diff(t)
    print(f"\n=== {name} ===", flush=True)
    print(f"  表头: {head!r}", flush=True)
    print(f"  点数={len(rows)} t=[{t[0]:.5f},{t[-1]:.5f}] s  功率集合={sorted(set(pw.tolist()))}", flush=True)
    print(f"  P1 dt: min={dt.min():.3e} max={dt.max():.3e} 等间隔={bool(np.allclose(dt, dt[0]))} "
          f"单调={bool(np.all(dt >= 0))} ⇒ {'PASS' if np.allclose(dt, 1e-5) else '注意'}", flush=True)
    print(f"  P2 关光(功率=0)点数={int((pw == 0).sum())} ⇒ "
          f"{'CSV 含关光段' if (pw == 0).any() else 'CSV 全程开光：门控不在这份导出里'}", flush=True)
    segs, d = segments(rows)
    tot_len = sum(s["length"] for s in segs)
    print(f"  P3 段数={len(segs)} 段长(µm)=" + " ".join(f"{s['length']:.1f}" for s in segs[:24]), flush=True)
    print(f"     Σ段长={tot_len:.1f} µm  总时长={t[-1]:.5f} s", flush=True)
    xs = sorted({round(s["p"][0]) for s in segs})
    if len(xs) > 1:
        xg = np.diff(xs)
        print(f"     各道 x（µm）= {xs[:24]}{' …' if len(xs) > 24 else ''}", flush=True)
        print(f"     相邻道 x 间距 min/med/max = {xg.min():.1f}/{np.median(xg):.1f}/"
              f"{xg.max():.1f} µm（README 声称 hatch 110 µm）", flush=True)
    v_nom = tot_len / 1e3 / t[-1] if t[-1] > 0 else float("nan")
    med = float(np.median(d[d > 0]))
    keep = d <= max(5.0 * med, med + 50.0)
    v_steps = d[keep] * 1e-3 / 1e-5  # µm→mm / 10 µs ⇒ mm/s
    print(f"  P4 名义速度=Σ长/Σ时={v_nom:.1f} mm/s | 逐步速度 p10/p50/p90="
          f"{np.percentile(v_steps, 10):.0f}/{np.percentile(v_steps, 50):.0f}/"
          f"{np.percentile(v_steps, 90):.0f} mm/s（步长中位={med:.1f} µm，"
          f"跳段 {int((~keep).sum())} 步）", flush=True)
    print(f"  P5 前 3 道（供 N2 判据②对表）:", flush=True)
    for k, s in enumerate(segs[:3]):
        print(f"      #{k+1} p={tuple(round(c) for c in s['p'])} q={tuple(round(c) for c in s['q'])} "
              f"len={s['length']:.2f} µm dur={s['dur'] * 1e3:.3f} ms "
              f"v={s['length'] * 1e-3 / s['dur']:.0f} mm/s", flush=True)

print("\n=== P6：Measurements.xlsx 的表结构（对比目标在哪里）===", flush=True)
try:
    from openpyxl import load_workbook
    wb = load_workbook(BASE / "Measurements.xlsx", read_only=True, data_only=True)
    print(f"  sheets({len(wb.sheetnames)}): {wb.sheetnames}", flush=True)
    for sn in wb.sheetnames[:4]:
        ws = wb[sn]
        print(f"  -- {sn}: dims={ws.max_row}x{ws.max_column}", flush=True)
        for r_i, row in enumerate(ws.iter_rows(min_row=1, max_row=6, values_only=True)):
            vals = [str(v)[:18] for v in row[:10]]
            print(f"     r{r_i+1}: {vals}", flush=True)
except Exception as e:  # noqa: BLE001
    print(f"  FAIL 读不到 xlsx：{type(e).__name__}: {e}", flush=True)
