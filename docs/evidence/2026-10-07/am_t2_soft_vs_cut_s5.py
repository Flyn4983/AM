"""#19 判据⑤的**续判 S5**：S4（`am_t2_soft_vs_cut.py`）在 3 个正对坐标轴的几何上按事先写死的
规则选了 `tanh(eps=dx/2)`，但**样本只有 3 个、且全是轴对齐**。本探针的目的不是"再确认一遍"，
而是**考核 S4 那条规则本身是否代表性不足**。

跑前登记的判据与**推翻条件**（先写死，后跑）：
  S5a 三个新几何都有**解析体积**（旋转是刚体运动 ⇒ 体积不变）：
      ① `box_rot`：1.2×0.6×0.4 mm 的盒子绕 (1,2,2) 轴转 37° ⇒ V = BX·BY·BZ；
      ② `sphere_small`：r=0.15 mm（直径 0.3 mm < 6·dx，**曲率主导**档）；
      ③ `cyl_rot_thin`：r=0.5、h=0.1 mm 的扁圆柱绕 x 轴转 30° ⇒ V = π r² h。
  S5b 门槛沿用 S4（不新设）：每个候选在每个几何上 `|Σw·dx³/V_true − 1| < 2%`（三档 dx 都要过）、
      偏差随 dx 变细单调减小、两拼写体积相对差 <1e-12。**任一几何不达即全局失格**（不是逐几何另选）。
  S5c **推翻规则**：只看最粗档 dx=50 的 `max|偏差|`，比较 `cut` 与 `tanh_half` 在**这 3 个新几何**上谁小：
      - 若 `tanh_half` 在 **≥2 个**新几何上比 `cut` 差 ⇒ 判 S4 的 3 样本**不具代表性**，
        改为在**合并的 6 个几何**上用**同一个统计量**（最粗档 max|偏差|）重裁，合并后的胜者为最终口径；
      - 否则（`tanh_half` 在新几何上打平或更好）⇒ **S4 的 `tanh_half` 定案，本探针之后不再允许改选**。
      注意这条规则是**对称**的：它同样可能把 S4 的结论翻掉，翻与不翻都由数字决定。
  S5d `tanh_dx`（现产线默认 `soft_occupancy()`，eps=dx）在 S4 已被判不合格；本探针**照跑**，
      只作"默认档到底差多少"的记录，不参与裁决。
"""
import re

import numpy as np

from amforge import geometry as G

BX, BY, BZ = 1.2e-3, 0.6e-3, 0.4e-3
DXS = (12.5, 25.0, 50.0)
COARSE = 50.0


def spellings(dx_um):
    return {"mul": dx_um * 1e-6, "lit": float(f"{dx_um}e-6")}


def rot_axis(axis, deg):
    """绕单位轴 axis 转 deg 度的 3×3 正交阵（Rodrigues）。"""
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    x, y, z = a
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) * c + s * K + (1.0 - c) * np.outer(a, a)


def box_sdf_local(x):
    return np.maximum(np.maximum(np.abs(x[..., 0]) - BX / 2,
                                 np.abs(x[..., 1]) - BY / 2),
                      np.abs(x[..., 2]) - BZ / 2)


def sphere_small_local(x):
    return np.linalg.norm(x, axis=-1) - 0.15e-3


def cyl_thin_local(x):
    return np.maximum(np.hypot(x[..., 0], x[..., 1]) - 0.5e-3,
                      np.abs(x[..., 2]) - 0.05e-3)


def make_rot(fn, Q):
    def f(x):
        y = np.asarray(x, dtype=np.float64) @ Q   # 等价于 Q^T·x（Q 正交）
        return fn(y)
    return f


R1 = rot_axis((1.0, 2.0, 2.0), 37.0)
R2 = rot_axis((1.0, 0.0, 0.0), 30.0)
HALF = 1.6e-3
SHAPES = {
    "box_rot(37°)": (make_rot(box_sdf_local, R1),
                     [(-HALF, HALF)] * 3, BX * BY * BZ),
    "sphere_small(r=0.15)": (sphere_small_local,
                             [(-0.4e-3, 0.4e-3)] * 3,
                             4.0 / 3.0 * np.pi * 0.15e-3 ** 3),
    "cyl_rot_thin(r=0.5,h=0.1,30°)": (make_rot(cyl_thin_local, R2),
                                      [(-HALF, HALF)] * 3,
                                      np.pi * 0.5e-3 ** 2 * 0.1e-3),
}
NAMES = ("cut", "tanh_dx", "tanh_half")


def weights(s, dx):
    return {"cut": np.clip(0.5 - s / dx, 0.0, 1.0),
            "tanh_dx": 0.5 * (1.0 - np.tanh(s / dx)),
            "tanh_half": 0.5 * (1.0 - np.tanh(s / (0.5 * dx)))}


print("=== S5a/S5b：三个新几何上的三套口径 ===", flush=True)
print(f"  旋转阵正交性核对：|R1ᵀR1−I|∞={np.max(np.abs(R1.T @ R1 - np.eye(3))):.2e} det={np.linalg.det(R1):.6f} | "
      f"|R2ᵀR2−I|∞={np.max(np.abs(R2.T @ R2 - np.eye(3))):.2e} det={np.linalg.det(R2):.6f}", flush=True)
for tag, (fn, bounds, v_true) in SHAPES.items():
    print(f"\n-- {tag}  V_true={v_true * 1e12:.4f} µm³ --", flush=True)
    for dx_um in DXS:
        per = {}
        for sp_tag, sp in spellings(dx_um).items():
            g = G.from_sdf_fn(lambda x: fn(np.asarray(x)), bounds=bounds,
                              spacing=sp, name="probe")
            s = np.asarray(g.sdf, dtype=np.float64)
            dx = float(g.spacing)
            w = weights(s, dx)
            per[sp_tag] = {n: float(w[n].sum()) * dx ** 3 for n in NAMES}
        line = f"  dx={dx_um:5.1f} µm  " + "  ".join(
            f"{n}={per['mul'][n] / v_true - 1.0:+7.3%}" for n in NAMES)
        print(line, flush=True)
        for n in NAMES:
            b = per["mul"][n] / v_true - 1.0
            rel = abs(per["mul"][n] - per["lit"][n]) / v_true
            flag = []
            if abs(b) >= 0.02:
                flag.append("S5b 偏差≥2%")
            if rel >= 1e-12:
                flag.append("S5b 拼写差≥1e-12")
            print(f"      {n:9s} 拼写差={rel:.2e}" + ("  ⇒ " + ",".join(flag) if flag else ""),
                  flush=True)

print("\n=== S1 收敛（新几何，50→25→12.5）===")
DEV = {}
for n in NAMES:
    DEV[n] = {}
    for tag, (fn, bounds, v_true) in SHAPES.items():
        seq = []
        for dx_um in (50.0, 25.0, 12.5):
            g = G.from_sdf_fn(lambda x: fn(np.asarray(x)), bounds=bounds,
                              spacing=dx_um * 1e-6, name="probe")
            s = np.asarray(g.sdf, dtype=np.float64)
            dx = float(g.spacing)
            w = weights(s, dx)[n]
            seq.append(float(w.sum()) * dx ** 3 / v_true - 1.0)
        DEV[n][tag] = seq
        mono = all(abs(seq[i]) >= abs(seq[i + 1]) - 1e-3 for i in range(2))
        print(f"  {n:9s} {tag:30s} " + " / ".join(f"{d:+8.3%}" for d in seq)
              + f" ⇒ {'PASS' if mono else 'FAIL(反弹)'}", flush=True)

print("\n=== S5c 推翻规则的执行 ===")
fails = []
qual = {n: True for n in NAMES}
for n in NAMES:
    for tag, seq in DEV[n].items():
        if any(abs(d) >= 0.02 for d in seq):
            qual[n] = False
            fails.append(f"S5b {n} {tag}: |偏差|≥2%")
        if not all(abs(seq[i]) >= abs(seq[i + 1]) - 1e-3 for i in range(2)):
            qual[n] = False
            fails.append(f"S5b {n} {tag}: 未单调收敛")

new_coarse = {n: max(abs(seq[0]) for seq in DEV[n].values()) for n in NAMES}
for n in NAMES:
    print(f"  {n:9s} 新几何合格={qual[n]}  最粗档 max|偏差|={new_coarse[n]:.4%}", flush=True)

worse = sum(1 for tag in SHAPES
            if abs(DEV['tanh_half'][tag][0]) > abs(DEV['cut'][tag][0]) + 1e-15)
print(f"  逐几何比较（tanh_half 是否比 cut 差）：", flush=True)
for tag in SHAPES:
    th, cu = abs(DEV['tanh_half'][tag][0]), abs(DEV['cut'][tag][0])
    print(f"    {tag:30s} tanh_half={th:.4%}  cut={cu:.4%}  ⇒ "
          f"{'tanh_half 更差' if th > cu else ('cut 更差' if cu > th else '同')}", flush=True)
print(f"  ⇒ 新几何上 tanh_half 比 cut 差的个数 = {worse}/3", flush=True)

# 合并 6 几何的同一统计量：**从 S4 日志逐行读**，不手抄（手抄就是又一次"归档文本自造数字"）。
S4LOG = "am_t2_soft_vs_cut.log"
s4_coarse = {}
try:
    cur_dx = None
    for ln in open(S4LOG, encoding="utf-8"):
        m = re.match(r"^-- dx=([\d.]+) µm --$", ln.strip())
        if m:
            cur_dx = float(m.group(1))
        m2 = re.match(r"^\s*(cut|tanh_dx|tanh_half)\s+偏差\s+([-+][\d.]+)%", ln)
        if m2 and cur_dx == COARSE:
            s4_coarse.setdefault(m2.group(1), []).append(float(m2.group(2)) / 100.0)
except FileNotFoundError:
    print(f"  ⚠ 读不到 {S4LOG} ⇒ 合并统计量无法计算（不是 0，是**缺数据**）", flush=True)
for n, v in s4_coarse.items():
    print(f"  S4 旧 3 几何（dx=50）{n:9s} 逐几何偏差 = "
          + " / ".join(f"{d:+.3%}" for d in v) + f"  max|·|={max(abs(d) for d in v):.4%}", flush=True)

pooled = {}
for n in NAMES:
    old = [abs(d) for d in s4_coarse.get(n, [])]
    new = [abs(DEV[n][tag][0]) for tag in SHAPES]
    if old and len(old) == 3:
        pooled[n] = max(max(old), max(new))
        print(f"  合并 6 几何（dx=50）{n:9s} max|偏差| = {pooled[n]:.4%}", flush=True)
    else:
        pooled[n] = max(new)
        print(f"  ⚠ {n}: S4 侧数据不全（读到 {len(old)} 个）⇒ 合并值只含新 3 几何 {pooled[n]:.4%}",
              flush=True)

if worse >= 2:
    verdict = ("S4 的 3 样本判为不具代表性 ⇒ 在合并 6 几何上用同一统计量重裁，"
               f"合并胜者 = **{min(pooled, key=lambda n: pooled[n] if qual[n] else 9e9)}**")
else:
    verdict = "S4 结论维持：tanh_half 定案，本探针之后不再允许改选。"
print(f"\nS5c 裁决语：{verdict}", flush=True)

print("\n=== 记分 ===")
print("FAILS:", fails if fails else "无 —— 三套口径在新几何上都未触发 S5b 门槛")
print(f"QUALIFIED: {[n for n in NAMES if qual[n]]}")
print(f"TANH_HALF_WORSE_ON: {worse}/3")
