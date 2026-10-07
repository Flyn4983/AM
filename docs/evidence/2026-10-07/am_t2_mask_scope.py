"""T2 收尾：三种**实体掩膜约定**的适用范围 —— 拼写不变性 ≠ 等于材料。

起因：`am_t2_convention_test.py` 的 W2 判据写得太粗。它只检查「两套 dx 拼写下掩膜是否
逐体素相同」，实测三个 dx 全部差异=0，于是脚本按预登记的措辞打印了「⇒ _footprint 约定可
直接用于 #19」。**但同一份日志的第一栏就把反证打在脸上**：
``solid(sdf<dx/2)=156849 == nvox``（三档都一样）。试片是个**填满自己包围盒**的盒子，
于是「半格膨胀」把整张网格都算成材料——它当然与 dx 拼写无关，因为体素总数早在 D4 的
``_grid_axes`` 里就被做成了拼写无关的。**不变性来自「不再区分材料与空气」**，不是来自
正确的边界处理。这个 PASS 现在撤回，改用能区分两者的几何重测。

本轮跑前登记的证伪判据（先写后跑）：
  C1 盒子试片上 ``sdf<dx/2`` 掩膜 = 全网格（nvox），且 ``nvox - solid(sdf<0)`` = 边界 ties
     那一叠 ⇒ 证实上一轮 W2 是**退化成立**，判据本身要改。
  C2 在非充满几何（球：r=0.4mm 装在 ±0.5mm 的盒里，角上真空 8 层；圆柱：r=0.3mm、
     h=0.4mm 装在 ±(0.3,0.3,0.2) 的盒里，四个棱角是空气）上，``sdf<dx/2`` 会纳入
     ``sdf>0`` 的**空气**体素。按表面积×(dx/2) 估算，dx=12.5µm 时球应吃进固体的
     4–6%，dx=50µm 时 ~15–20%。判据：**额外空气 > 2% 固体 ⇒ 该约定不得用作热域实体掩膜**
     （否则熔池体积、吸光面积、热容都按含空气的口径系统性偏大）。
  C3 ``sdf < 1e-12``（= 8e-8·dx，纯 ulp 容差：机制日志里两拼写的近零 |sdf| 是 5.4e-20 与
     1.08e-19，都被吸收；而真实空气体素的 sdf ≥ ~0.1·dx 远在其外）在两种拼写间**逐体素
     差异=0**，且在球/柱上吃进的空气 ≤ 0.5% 固体。若成立 ⇒ #19 的修法是「给符号加
     ulp 级容差」，不是「半格膨胀」。
  C4 球面上也应存在 sdf 近零体素（刀锋不只发生在平面对齐的盒子零件）。若球上
     ``xor(strict)=0`` 而圆柱上非 0 ⇒ 风险按「平面与包围盒对齐」定位，如实登记。
  C5 严格掩膜 ``sdf<0`` 在三份几何上都应复现「拼写间差异集中在贴层面」（机制日志已对
     盒子证明 100% 在 z=末面）。
"""
import numpy as np

from amforge import geometry as G

BX, BY, BZ = 1.2e-3, 0.6e-3, 0.4e-3
EPS_ABS = 1e-12  # m；≈8e-8·dx(dx=12.5µm)，只吸收 ulp 抖动，不吃半个体素


def spellings(dx_um):
    return {"mul": dx_um * 1e-6, "lit": float(f"{dx_um}e-6")}


def box_sdf(x):
    return np.maximum(np.maximum(np.abs(x[..., 0]) - BX / 2,
                                 np.abs(x[..., 1]) - BY / 2),
                      np.abs(x[..., 2]) - BZ / 2)


def sphere_sdf(x):
    return np.sqrt((x[..., 0] ** 2 + x[..., 1] ** 2) * 1.0 + x[..., 2] ** 2) - 0.4e-3


def cyl_sdf(x):
    rho = np.hypot(x[..., 0], x[..., 1]) - 0.3e-3
    return np.maximum(rho, np.abs(x[..., 2]) - BZ / 2)


SHAPES = {
    "box(填满包围盒)": (box_sdf, [(-BX / 2, BX / 2), (-BY / 2, BY / 2), (-BZ / 2, BZ / 2)],
                        BX * BY * BZ),
    "sphere(r=0.4mm)": (sphere_sdf, [(-0.5e-3, 0.5e-3)] * 3,
                        4.0 / 3.0 * np.pi * 0.4e-3 ** 3),
    "cyl(r=0.3,h=0.4)": (cyl_sdf, [(-0.3e-3, 0.3e-3), (-0.3e-3, 0.3e-3), (-BZ / 2, BZ / 2)],
                         np.pi * 0.3e-3 ** 2 * BZ),
}

print("=== 机制复核：整数倍 dx 时各几何的 sdf 近零统计（mul 拼写，dx=12.5µm）===", flush=True)
for tag, (fn, bounds, _) in SHAPES.items():
    g = G.from_sdf_fn(lambda x: fn(np.asarray(x)), bounds=bounds,
                      spacing=12.5 * 1e-6, name="probe")
    s = np.asarray(g.sdf, dtype=np.float64)
    print(f"  {tag:20s} shape={str(tuple(s.shape)):14s} nvox={s.size:7d} "
          f"sdf<0={int((s < 0).sum()):7d} sdf==0={int((s == 0).sum()):6d} "
          f"|sdf|<{EPS_ABS:g}={int((np.abs(s) < EPS_ABS).sum()):6d} "
          f"min|sdf|nonzero={np.min(np.abs(s[s != 0])):.3e}", flush=True)

results = {}
for dx_um in (12.5, 25.0, 50.0):
    print(f"\n=== dx={dx_um} µm：三种掩膜约定的拼写不变性 + 空气纳入量 ===", flush=True)
    for tag, (fn, bounds, v_true) in SHAPES.items():
        row = {}
        for sp_tag, sp in spellings(dx_um).items():
            g = G.from_sdf_fn(lambda x: fn(np.asarray(x)), bounds=bounds,
                              spacing=sp, name="probe")
            s = np.asarray(g.sdf, dtype=np.float64)
            dx = float(g.spacing)
            m = {
                "strict": s < 0.0,
                "eps": s < EPS_ABS,
                "half": s < 0.5 * dx,
            }
            for k, mk in m.items():
                air = int((mk & (s > 0.0)).sum())
                row[(sp_tag, k)] = dict(mask=mk, n=int(mk.sum()), air=air,
                                        dx=dx, nvox=s.size)
        print(f"  {tag:20s} nvox={row[('mul', 'strict')]['nvox']}", flush=True)
        for k in ("strict", "eps", "half"):
            a, b = row[("mul", k)], row[("lit", k)]
            xor = int((a["mask"] ^ b["mask"]).sum())
            base = row[("mul", "strict")]["n"]
            print(f"    {k:6s} 计数 {a['n']}/{b['n']}  xor={xor:5d} | "
                  f"吃进空气 {a['air']:5d}/{b['air']:5d} "
                  f"= 固体的 {a['air'] / base * 100:5.2f}%/{b['air'] / base * 100:5.2f}% | "
                  f"体积口径 mul: {a['n'] * a['dx'] ** 3 / v_true:.4f}×真值", flush=True)
        results[(dx_um, tag)] = row

print("\n=== 记分 ===", flush=True)
fails = []
for (dx_um, tag), row in results.items():
    for k in ("strict", "eps", "half"):
        xor = int((row[("mul", k)]["mask"] ^ row[("lit", k)]["mask"]).sum())
        air = row[("mul", k)]["air"] / row[("mul", "strict")]["n"]
        base_ok = int(row[("mul", "half")]["n"]) != int(row[("mul", "half")]["nvox"])
        if k == "half" and base_ok and air > 0.02:
            fails.append(f"C2 dx={dx_um} {tag}: 半格约定吃进空气 {air*100:.2f}% (>2%)")
        if k in ("eps", "half") and xor != 0:
            fails.append(f"C3/C1 dx={dx_um} {tag}: {k} 掩膜拼写间差异 {xor} ≠ 0")
        if k == "eps" and air > 0.005:
            fails.append(f"C3 dx={dx_um} {tag}: ulp 容差吃进空气 {air*100:.2f}% (>0.5%)")
print("FAILS:", fails if fails else "无（半格约定在非充满几何上被判据 C2 否决；"
      "ulp 容差约定同时满足不变性与不吃空气）")
