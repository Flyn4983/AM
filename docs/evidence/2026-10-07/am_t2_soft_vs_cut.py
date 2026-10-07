"""#19 判据⑤：仓库里并存**两套**软占位口径 —— `contracts.soft_occupancy()`
（`0.5·(1−tanh(sdf/dx))`，1 体素过渡带、**双侧长尾**）与本轮定稿的**线性 cut 份额**
（`w = clip(0.5 − sdf/dx, 0, 1)`，紧支撑）。`PartGeometry.volume()` 现在走的是**前者**。

留两套就等于下一个 §26.8：同一份几何会给出两个不同的体积，而两者都"看起来对"。
所以本探针的目的**不是证明哪个更好**，而是**按事先登记的规则选一个并废掉另一个**。

跑前登记的判据（写在这里，先于任何数字）：
  S1 **精度**：三种几何 × 三档 dx 上，候选口径的 `Σw·dx³ / V_true` 偏差应 <2%，
     且随 dx 变细**单调减小**（与 §26.8(h) 的 F1 同形）。不达标的候选直接失格。
  S2 **可复现**：两种 dx 拼写（`µm×1e-6` 与 `f"{µm}e-6"`）之间，体积相对差 <1e-12。
  S3 **泄漏**：零件外部（`sdf > dx`）与内部空腔侧（`sdf < −dx`）上的权重和，
     以占总体素体积的比例报告。这一列**只作解释，不作门槛**：长尾是 tanh 的数学性质，
     如果它的偏差比 cut 小，那它就是"更好的近似"，我们不能因为它有尾巴就否掉它。
  S4 **裁决规则（先写死）**：S1、S2 都过的候选里，取 **dx=50 µm（最粗档）上 |偏差| 最小者**
     为唯一口径；落选者**不保留为第二套 helper**（要么删，要么重写成调用同一个式子的别名）。
     若两个候选在 S1/S2 上同过且最粗档 |偏差| 相差 <1e-3（即分辨不出），则取 **cut**
     （理由要写进日志：紧支撑 ⇒ 掩膜与体积共用同一个边界定义，不出现"体积算得对但掩膜多一圈"）。
     **不许事后改成偏好那版。**
"""
import numpy as np

from amforge import geometry as G

BX, BY, BZ = 1.2e-3, 0.6e-3, 0.4e-3
DXS = (12.5, 25.0, 50.0)


def spellings(dx_um):
    return {"mul": dx_um * 1e-6, "lit": float(f"{dx_um}e-6")}


def box_sdf(x):
    return np.maximum(np.maximum(np.abs(x[..., 0]) - BX / 2,
                                 np.abs(x[..., 1]) - BY / 2),
                      np.abs(x[..., 2]) - BZ / 2)


def sphere_sdf(x):
    return np.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2 + x[..., 2] ** 2) - 0.4e-3


def cyl_sdf(x):
    return np.maximum(np.hypot(x[..., 0], x[..., 1]) - 0.3e-3,
                      np.abs(x[..., 2]) - BZ / 2)


SHAPES = {
    "box(填满包围盒)": (box_sdf, [(-BX / 2, BX / 2), (-BY / 2, BY / 2), (-BZ / 2, BZ / 2)],
                        BX * BY * BZ),
    "sphere(r=0.4mm)": (sphere_sdf, [(-0.5e-3, 0.5e-3)] * 3,
                        4.0 / 3.0 * np.pi * 0.4e-3 ** 3),
    "cyl(r=0.3,h=0.4)": (cyl_sdf, [(-0.3e-3, 0.3e-3), (-0.3e-3, 0.3e-3), (-BZ / 2, BZ / 2)],
                         np.pi * 0.3e-3 ** 2 * BZ),
}


def weights(s, dx):
    """候选口径。返回 name -> w 数组。"""
    return {
        "cut": np.clip(0.5 - s / dx, 0.0, 1.0),
        "tanh_dx": 0.5 * (1.0 - np.tanh(s / dx)),
        "tanh_half": 0.5 * (1.0 - np.tanh(s / (0.5 * dx))),
    }


NAMES = ("cut", "tanh_dx", "tanh_half")
fails = []
bias = {n: {} for n in NAMES}
spell_rel = {n: [] for n in NAMES}
leak_out = {n: [] for n in NAMES}

print("=== S1/S2/S3：三套口径 × 三几何 × 三档 dx ===", flush=True)
for dx_um in DXS:
    print(f"\n-- dx={dx_um} µm --", flush=True)
    for tag, (fn, bounds, v_true) in SHAPES.items():
        acc = {}
        for sp_tag, sp in spellings(dx_um).items():
            g = G.from_sdf_fn(lambda x: fn(np.asarray(x)), bounds=bounds,
                              spacing=sp, name="probe")
            s = np.asarray(g.sdf, dtype=np.float64)
            dx = float(g.spacing)
            w = weights(s, dx)
            cell = dx ** 3
            acc[sp_tag] = {
                n: {
                    "vol": float(w[n].sum()) * cell,
                    "out": float(w[n][s > dx].sum()) * cell,
                    "in": float(w[n][s < -dx].sum()) * cell,
                } for n in NAMES
            }
        print(f"  {tag:20s} " + " | ".join(
            f"{n}: {acc['mul'][n]['vol'] / v_true:7.4f}×" for n in NAMES), flush=True)
        for n in NAMES:
            b = acc["mul"][n]["vol"] / v_true - 1.0
            bias[n].setdefault(tag, []).append(b)
            rel = abs(acc["mul"][n]["vol"] - acc["lit"][n]["vol"]) / v_true
            spell_rel[n].append(rel)
            lo = acc["mul"][n]["out"] / v_true
            leak_out[n].append(lo)
            print(f"    {n:9s} 偏差 {b * 100:+7.3f}%  拼写差 {rel:.2e}"
                  f"  体外泄漏 {lo * 100:6.3f}%  体内截断 {acc['mul'][n]['in'] / v_true * 100:6.3f}%",
                  flush=True)
            if abs(b) >= 0.02:
                fails.append(f"S1 {n} {tag} dx={dx_um}: |偏差| {abs(b)*100:.2f}% ≥2%")
            if rel >= 1e-12:
                fails.append(f"S2 {n} {tag} dx={dx_um}: 拼写差 {rel:.2e} ≥1e-12")

print("\n=== S1 收敛性：偏差应随 dx 变细单调减小（顺序 50→25→12.5）===", flush=True)
for n in NAMES:
    for tag, devs in bias[n].items():
        seq = [devs[DXS.index(d)] for d in (50.0, 25.0, 12.5)]
        mono = all(abs(seq[i]) >= abs(seq[i + 1]) - 1e-3 for i in range(len(seq) - 1))
        print(f"  {n:9s} {tag:20s} " + " / ".join(f"{d * 100:+7.3f}%" for d in seq)
              + f" ⇒ {'PASS' if mono else 'FAIL(反弹)'}", flush=True)
        if not mono:
            fails.append(f"S1 {n} 收敛 {tag}: 未单调减小")

print("\n=== S4 裁决 ===", flush=True)
qualified = [n for n in NAMES if not any(
    f.startswith(f"S1 {n} ") or f.startswith(f"S2 {n} ") for f in fails)]
coarse = {n: max(abs(v[DXS.index(50.0)]) for v in bias[n].values()) for n in NAMES}
for n in NAMES:
    print(f"  {n:9s} 合格={n in qualified}  最粗档最大|偏差|={coarse[n] * 100:.3f}%"
          f"  拼写差上界={max(spell_rel[n]):.2e}  体外泄漏上界={max(leak_out[n]) * 100:.3f}%",
          flush=True)
if not qualified:
    fails.append("S4 无候选同时过 S1/S2 ⇒ 体积口径必须由外部（解析 cut）给出，#19 重新定形")
    winner = None
else:
    ranked = sorted((coarse[n] for n in qualified))
    best = min(qualified, key=lambda n: coarse[n])
    winner = best
    if len(qualified) > 1 and (ranked[1] - ranked[0]) < 1e-3:
        winner = "cut"
        print(f"  ⇒ 合格候选在最粗档上分辨不出（最小两值之差 {ranked[1] - ranked[0]:.2e} < 1e-3）"
              f"⇒ 按 S4 后半取 cut（紧支撑 ⇒ 掩膜与体积同一边界定义）", flush=True)
    else:
        print(f"  ⇒ 按 S4 取 **{winner}**（最粗档 |偏差| 最小），其余口径**不再是第二套 helper**",
              flush=True)

print("\n=== 记分 ===", flush=True)
print("FAILS:", fails if fails else "无 —— S1/S2/S4 全部满足")
print(f"WINNER: {winner}")
