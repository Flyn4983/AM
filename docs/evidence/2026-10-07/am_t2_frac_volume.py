"""T2/#19 判据补强：布尔掩膜换成**线性 cut 份额** w=clip(0.5−sdf/dx,0,1) 后，
体积口径还剩多少一阶偏差？（`am_t2_mask_scope.py` 的延伸，纯几何、无求解）

起因（同一轮实测）：三份几何、三档 dx 上，``Σ(sdf<0)·dx³`` 给出真值的
0.878–0.999×，``Σ(sdf<dx/2)·dx³`` 给出 1.047–1.363× ⇒ **熔池体积这一列数值的口径
本身有 O(dx) 的一阶系统偏差，且方向取决于掩膜约定**。网格是**节点居中**的
（``_grid_axes`` 用 ``lo + sp·arange(n_cells+1)``，两端都算体素），所以盒子零件按体素计数
会**多出一层**（n+1 而不是 n），而严格 ``sdf<0`` 又把边界面 ties 那一层**全丢掉**：
两个偏差符号相反、量级都是 1 层 ≈ dx/尺寸，粗网格下可达 27%。

跑前登记的证伪判据：
  F1 份额求和 ``Σw·dx³`` 对三种几何、三档 dx 都应在真值的 ±2% 内，且**偏差随 dx 变细
     而单调减小**（一阶收敛）。若偏差仍像布尔掩膜那样停在 5–27%，份额方案被推翻。
  F2 份额求和在两种 dx 拼写间的相对差 < 1e-12（权重是 sdf 的连续函数，ulp 抖动只按
     1 层表面积的 ulp 距离进入，不再发生整层翻转）。若 ≥1e-6 ⇒ 说明 ulp 抖动被放大，
     份额方案同样需要 ulp 容差兜底。
  F3 严格布尔掩膜的**翻转体素数**（拼写间 xor）应集中在 sdf 近零集内；份额方案没有
     布尔翻转的概念，但仍要报告 ``Σw`` 的**逐体素最大相对变化**作对照。
"""
import numpy as np

from amforge import geometry as G

BX, BY, BZ = 1.2e-3, 0.6e-3, 0.4e-3
EPS_ABS = 1e-12


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

print("=== F1/F2：布尔掩膜 vs 线性 cut 份额的体积口径 ===", flush=True)
fails = []
for dx_um in (12.5, 25.0, 50.0):
    print(f"\n-- dx={dx_um} µm --", flush=True)
    for tag, (fn, bounds, v_true) in SHAPES.items():
        acc = {}
        for sp_tag, sp in spellings(dx_um).items():
            g = G.from_sdf_fn(lambda x: fn(np.asarray(x)), bounds=bounds,
                              spacing=sp, name="probe")
            s = np.asarray(g.sdf, dtype=np.float64)
            dx = float(g.spacing)
            w = np.clip(0.5 - s / dx, 0.0, 1.0)
            vol = {
                "strict": float((s < 0.0).sum()) * dx ** 3,
                "eps": float((s < EPS_ABS).sum()) * dx ** 3,
                "half": float((s < 0.5 * dx).sum()) * dx ** 3,
                "frac": float(w.sum()) * dx ** 3,
            }
            vol["_w"] = w
            acc[sp_tag] = vol
            print(f"  {tag:20s} [{sp_tag}] " + " ".join(
                f"{k}={vol[k] / v_true:7.4f}×"
                for k in ("strict", "eps", "half", "frac")), flush=True)
        dw = float(np.max(np.abs(acc["mul"]["_w"] - acc["lit"]["_w"])))
        print(f"    F3 份额逐体素最大变化 max|Δw| = {dw:.3e}"
              f"（布尔掩膜在同一对上会整层翻转，见上表 strict/half 列）", flush=True)
        for k in ("strict", "eps", "half", "frac"):
            rel = abs(acc["mul"][k] - acc["lit"][k]) / v_true
            if k == "frac":
                print(f"    {tag} frac 拼写间相对差 = {rel:.3e} ⇒ F2 "
                      f"{'PASS' if rel < 1e-12 else 'FAIL'}", flush=True)
                if rel >= 1e-12:
                    fails.append(f"F2 {tag} dx={dx_um}: frac 拼写差 {rel:.2e}")
            if k in ("strict", "half") and rel > 1e-3:
                print(f"    {tag} {k} 拼写间体积口径差 = {rel * 100:.3f}% "
                      f"（布尔掩膜的整层翻转）", flush=True)
        dev = abs(acc["mul"]["frac"] / v_true - 1.0)
        print(f"    {tag} frac 偏差(mul) = {(acc['mul']['frac'] / v_true - 1) * 100:+6.3f}% "
              f"⇒ F1 {'PASS' if dev < 0.02 else 'FAIL'}", flush=True)
        if dev >= 0.02:
            fails.append(f"F1 {tag} dx={dx_um}: frac 偏差 {dev*100:.2f}% ≥2%")

print("\n=== F1 收敛性：frac 偏差应随 dx 变细而减小 ===", flush=True)
for tag, (fn, bounds, v_true) in SHAPES.items():
    devs = []
    for dx_um in (50.0, 25.0, 12.5):
        g = G.from_sdf_fn(lambda x: fn(np.asarray(x)), bounds=bounds,
                          spacing=dx_um * 1e-6, name="probe")
        s = np.asarray(g.sdf, dtype=np.float64)
        dx = float(g.spacing)
        w = np.clip(0.5 - s / dx, 0.0, 1.0)
        devs.append(float(w.sum()) * dx ** 3 / v_true - 1.0)
    mono = all(abs(devs[i]) >= abs(devs[i + 1]) - 1e-3 for i in range(len(devs) - 1))
    print(f"  {tag:20s} dx=50/25/12.5 偏差 = " +
          " / ".join(f"{d * 100:+.3f}%" for d in devs) +
          f" ⇒ 单调减小 {'PASS' if mono else 'FAIL(有反弹)'}", flush=True)
    if not mono:
        fails.append(f"F1 收敛 {tag}: 偏差未随 dx 变细单调减小")

print("\n=== 记分 ===", flush=True)
print("FAILS:", fails if fails else "无 —— 线性 cut 份额把体积口径的一阶偏差压到 <2% 且随 dx 收敛，"
      "同时天然拼写无关（F2）")
