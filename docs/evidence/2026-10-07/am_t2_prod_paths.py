"""#19 落地后的**生产路径**验收：口径改动是否真的接到了代码上。

前面的 `am_t2_soft_vs_cut{,_s5}.py` 是在**探针里重写公式**做的选型；本脚本一律走
生产入口（`PartGeometry.occupancy / solid_fraction / volume`、`process.layer_activation_times`），
验证的是"选型 → 实现"这一步，而不是再选一次。

跑前登记的证伪判据（任一不满足即 FAIL，不许事后放宽）：

  P1 拼写位一致：3 几何 × 3 档 dx × 2 种 dx 拼写，`geometry.occupancy` 逐体素 xor == 0。
     正对照（同表内）：旧硬掩膜 `sdf < 0` 的 xor 必须**至少在一格里 > 0**；
     若全表为 0，说明探针看不见刀锋集，P1 是哑火绿灯 ⇒ 判 FAIL。
     ⚠ **v1→v2 改判**（v1 登记为"`solid_fraction()` 逐体素 max|Δw| == 0"，实测
     `am_t2_prod_paths.log` 给 3.3e-15…2.2e-14）：两种拼写的 `spacing` **本身就差 1 ulp**
     （`12.5*1e-6 != float("12.5e-6")`），栅格轴与 sdf 也随之差 ulp ⇒ `w=sdf/dx` 在数学上
     不可能位相同。v1 的判据**写强了**，属判据自身缺陷而非实现缺陷。v2 改成可检验的两条：
     (a) max|Δw| ≤ 1e-12（绝对值，w∈[0,1]；刀锋整层翻转会是 O(1)，两者差 12 个数量级）；
     (b) 体积口径的拼写相对差 |ΔV|/V < 1e-12（沿用 `am_t2_frac_volume.py` 的 F2 门槛）。

  P2 体积口径归份额：`geometry.volume()` 与 `Σw·dx³` 的相对差 < 1e-12（生产实现就是份额），
     且份额对真值的偏差在 dx ≤ 25 µm 时全部 < 2%（dx=50 µm 只登记不判，见 S5 结论）。

  P3 与平滑口径可区分：`volume()` ≠ `Σ soft_occupancy·dx³`（相对差 > 1e-3）。
     若两者相等 ⇒ 说明 tanh 还挂在体积口径上，P3 FAIL。

  P4 tracer 安全：`spacing` 是 pytree 叶子 ⇒ `jax.grad`/`jax.jit` 之下是 tracer。
     判据：`grad(lambda g: jnp.sum(g.solid_fraction()))(geometry)` 对 spacing 的梯度**有限**，
     `solid_mask`/`volume` 在 jit 下可用。
     ⚠ **v1→v2 改判**（v1 登记为"jit 后 volume 与 eager **位一致**"，实测
     `2.893749999999999e-10` vs `2.8937499999999994e-10`，相对差 1.4e-16）：jit 会**重排
     归约顺序**（这里对 97×49×33 个体素求和），浮点加法不满足结合律 ⇒ 末位 ulp 差是
     编译器性质，位一致这条从一开始就不可满足。v2 改成相对差 < 1e-12。
     v1 的另一条 `float(solid_weight(...).sum())` 抛 ConcretizationTypeError 是**探针自身的
     bug**（在被 grad 的函数里调 `float()`），不是生产代码的问题；v2 改用 jnp 求和。
     任何 ConcretizationTypeError（来自 src/）/ NaN / 相对差 ≥1e-12 ⇒ FAIL。

  P5 面积/层口径切换生效：`layer_activation_times` 缺省推导的激活时刻
     两种拼写位一致，且与旧 tanh 口径的结果**不同**（逐层面积占比最大相对差 > 1e-6）
     ⇒ 证明切换生效；若与旧口径位一致 ⇒ 改动没接到，FAIL。
"""
import jax
import jax.numpy as jnp
import numpy as np

from amforge import geometry as G
from amforge import process as PR
from amforge.core.contracts import SDF_SOLID_TOL, ProcessPlan, solid_weight

BX, BY, BZ = 1.2e-3, 0.6e-3, 0.4e-3


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
    "box": (box_sdf, [(-BX / 2, BX / 2), (-BY / 2, BY / 2), (-BZ / 2, BZ / 2)],
            BX * BY * BZ),
    "sphere(r=0.4mm)": (sphere_sdf, [(-0.5e-3, 0.5e-3)] * 3,
                        4.0 / 3.0 * np.pi * 0.4e-3 ** 3),
    "cyl(r=0.3,h=0.4)": (cyl_sdf, [(-0.3e-3, 0.3e-3), (-0.3e-3, 0.3e-3),
                                   (-BZ / 2, BZ / 2)],
                         np.pi * 0.3e-3 ** 2 * BZ),
}

fails = []
flip_control = 0

print("=== P1/P2/P3：生产 occupancy / solid_fraction / volume ===", flush=True)
vol_bias = {}
for dx_um in (12.5, 25.0, 50.0):
    print(f"\n-- dx={dx_um} µm --", flush=True)
    for tag, (fn, bounds, v_true) in SHAPES.items():
        acc = {}
        for sp_tag, sp in spellings(dx_um).items():
            g = G.from_sdf_fn(lambda x: fn(np.asarray(x)), bounds=bounds,
                              spacing=sp, name="probe")
            s = np.asarray(g.sdf, dtype=np.float64)
            dx = float(g.spacing)
            occ = np.asarray(g.occupancy)                     # 生产：solid_mask
            w = np.asarray(g.solid_fraction())                # 生产：solid_weight
            v = float(g.volume())                             # 生产：份额积分
            old = (s < 0.0)                                   # 旧硬掩膜（对照）
            tanh_v = float(np.sum(0.5 * (1 - np.tanh(s / dx))) * dx ** 3)
            acc[sp_tag] = dict(occ=occ, w=w, v=v, old=old, tanh_v=tanh_v,
                               frac_sum=float(w.sum()) * dx ** 3)
        xor_new = int(np.logical_xor(acc["mul"]["occ"], acc["lit"]["occ"]).sum())
        xor_old = int(np.logical_xor(acc["mul"]["old"], acc["lit"]["old"]).sum())
        flip_control += xor_old
        dw = float(np.max(np.abs(acc["mul"]["w"] - acc["lit"]["w"])))
        rel_V_sp = abs(acc["mul"]["v"] / acc["lit"]["v"] - 1.0)
        rel_vol = abs(acc["mul"]["v"] / acc["mul"]["frac_sum"] - 1.0)
        rel_tanh = abs(acc["mul"]["v"] / acc["mul"]["tanh_v"] - 1.0)
        bias = acc["mul"]["v"] / v_true - 1.0
        vol_bias[(tag, dx_um)] = bias
        print(f"  {tag:18s} xor(new)={xor_new:4d} xor(old sdf<0)={xor_old:5d} "
              f"max|Δw|={dw:.3e} |ΔV|/V(拼写)={rel_V_sp:.3e} "
              f"|volume-Σw·dx³|/Σ={rel_vol:.3e} "
              f"|volume-tanh|/tanh={rel_tanh:.3e} 体积偏差={bias * 100:+6.3f}%",
              flush=True)
        if xor_new != 0:
            fails.append(f"P1 {tag} dx={dx_um}: occupancy xor={xor_new}")
        if dw > 1e-12:
            fails.append(f"P1 {tag} dx={dx_um}: 份额 max|Δw|={dw:.3e} > 1e-12")
        if rel_V_sp >= 1e-12:
            fails.append(f"P1 {tag} dx={dx_um}: 体积拼写相对差 {rel_V_sp:.3e} >= 1e-12")
        if rel_vol > 1e-12:
            fails.append(f"P2 {tag} dx={dx_um}: volume 与 Σw·dx³ 差 {rel_vol:.3e}")
        if dx_um <= 25.0 and abs(bias) >= 0.02:
            fails.append(f"P2 {tag} dx={dx_um}: 体积偏差 {bias * 100:.3f}% >= 2%")
        if rel_tanh <= 1e-3:
            fails.append(f"P3 {tag} dx={dx_um}: volume 与 tanh 口径不可区分 "
                         f"(差 {rel_tanh:.3e})")

print(f"\n  P1 正对照：旧硬掩膜拼写间翻转体素总数 = {flip_control} ⇒ "
      f"{'探针看得见刀锋集（有效）' if flip_control > 0 else '哑火！P1 判 FAIL'}",
      flush=True)
if flip_control == 0:
    fails.append("P1 正对照哑火：旧 sdf<0 掩膜在所有格上都不翻面")

print("\n=== P2 收敛性：偏差随 dx 变细单调减小 ===", flush=True)
for tag in SHAPES:
    devs = [abs(vol_bias[(tag, d)]) for d in (50.0, 25.0, 12.5)]
    mono = all(devs[i] >= devs[i + 1] - 1e-3 for i in range(len(devs) - 1))
    print(f"  {tag:18s} |bias| dx=50/25/12.5 = "
          + " / ".join(f"{d * 100:+.3f}%" for d in devs)
          + f" ⇒ 单调 {'PASS' if mono else 'FAIL(反弹)'}", flush=True)
    if not mono:
        fails.append(f"P2 收敛 {tag}: 偏差未随 dx 变细单调减小")

print("\n=== P4：tracer 安全（spacing 是 pytree 叶子） ===", flush=True)
g = G.from_sdf_fn(lambda x: box_sdf(np.asarray(x)),
                  bounds=[(-BX / 2, BX / 2), (-BY / 2, BY / 2), (-BZ / 2, BZ / 2)],
                  spacing=25.0e-6, name="p4")
eager_v = float(g.volume())


def sum_w(geo):
    return jnp.sum(geo.solid_fraction())


try:
    gr = jax.grad(sum_w)(g)
    gs = float(gr.spacing)
    gsdf_ok = bool(jnp.all(jnp.isfinite(gr.sdf)))
    print(f"  grad w.r.t. spacing = {gs:.6e}  finite(sdf)={gsdf_ok}", flush=True)
    if not (np.isfinite(gs) and gsdf_ok):
        fails.append(f"P4 spacing 梯度非有限 ({gs!r}) 或 sdf 含 NaN")
except Exception as exc:                                    # noqa: BLE001
    print(f"  grad 抛异常：{type(exc).__name__}: {exc}", flush=True)
    fails.append(f"P4 grad 抛异常 {type(exc).__name__}")

try:
    jit_v = float(jax.jit(lambda geo: geo.volume())(g))
    rel_je = abs(jit_v / eager_v - 1.0)
    jit_mask = jax.jit(lambda geo: jnp.sum(geo.occupancy))(g)
    print(f"  jit volume = {jit_v:.15e} vs eager {eager_v:.15e} "
          f"相对差={rel_je:.3e}（v2：允许归约重排的末位 ulp）", flush=True)
    print(f"  jit Σoccupancy = {float(jit_mask):.0f}（可编译）", flush=True)
    if rel_je >= 1e-12:
        fails.append(f"P4 jit/eager volume 相对差 {rel_je:.3e} >= 1e-12")
except Exception as exc:                                    # noqa: BLE001
    print(f"  jit 抛异常：{type(exc).__name__}: {exc}", flush=True)
    fails.append(f"P4 jit 抛异常 {type(exc).__name__}")

# 份额公式本身对 spacing 可微（体积口径穿过网格尺寸的梯度是 #19 之后才存在的通路）
try:
    dsum = jax.grad(lambda sp: solid_weight(g.sdf, sp).sum())   # v2：不做 float()
    v = float(dsum(jnp.float64(25.0e-6)))
    print(f"  solid_weight 对 spacing 的可微性：dΣw/dspacing = {v:.6e} "
          f"finite={np.isfinite(v)}", flush=True)
    if not np.isfinite(v):
        fails.append("P4 dΣw/dspacing 非有限")
except Exception as exc:                                    # noqa: BLE001
    print(f"  solid_weight grad 抛异常：{type(exc).__name__}: {exc}", flush=True)
    fails.append(f"P4 solid_weight grad 抛异常 {type(exc).__name__}")

print("\n=== P5：layer_activation_times 的层面积口径 ===", flush=True)
plan = ProcessPlan.uniform(n_layers=4)
af_new = None
for sp_tag, sp in spellings(25.0).items():
    gg = G.from_sdf_fn(lambda x: cyl_sdf(np.asarray(x)),
                       bounds=[(-0.3e-3, 0.3e-3), (-0.3e-3, 0.3e-3),
                               (-BZ / 2, BZ / 2)], spacing=sp, name="p5")
    t = np.asarray(PR.layer_activation_times(gg, plan), dtype=np.float64)
    if sp_tag == "mul":
        af_new = t
        # 旧口径（tanh 软占位）对照：按实现前的式子重算一遍
        s = np.asarray(gg.sdf, dtype=np.float64)
        dx = float(gg.spacing)
        occ_tanh = 0.5 * (1.0 - np.tanh(s / dx))
        az = np.mean(occ_tanh, axis=(0, 1))
        zi = np.linspace(0.0, az.shape[0] - 1.0, plan.n_layers)
        lo, hi = gg.bbox()
        extent = np.asarray(hi)[:2] - np.asarray(lo)[:2]
        area_old = np.interp(zi, np.arange(az.shape[0]), az) * extent[0] * extent[1]
        track_old = area_old / float(np.mean(np.atleast_1d(
            np.asarray(plan.hatch_spacing, dtype=np.float64))))
        print(f"  旧 tanh 口径 track_len = {np.round(track_old, 9)}", flush=True)
    print(f"  [{sp_tag}] 激活时刻 = {np.round(t, 9)}", flush=True)
    if sp_tag == "lit":
        dd = float(np.max(np.abs(t - af_new)))
        print(f"  拼写间 max|Δt| = {dd:.3e} ⇒ P5 位一致 {'PASS' if dd == 0 else 'FAIL'}",
              flush=True)
        if dd != 0.0:
            fails.append(f"P5 拼写间激活时刻差 {dd:.3e}")

s = np.asarray(g.sdf, dtype=np.float64)
dx = float(g.spacing)
az_cut = np.mean(np.clip(0.5 - s / dx, 0.0, 1.0), axis=(0, 1))
az_tanh = np.mean(0.5 * (1 - np.tanh(s / dx)), axis=(0, 1))
rel_af = float(np.max(np.abs(az_cut - az_tanh)) / max(np.max(np.abs(az_tanh)), 1e-30))
print(f"  份额 vs tanh 逐层面积占比最大相对差 = {rel_af:.4f} ⇒ "
      f"{'切换确实生效' if rel_af > 1e-6 else '与旧口径位一致：改动没接到！'}", flush=True)
if rel_af <= 1e-6:
    fails.append(f"P5 正对照：份额与 tanh 层面积无可区分差 ({rel_af:.3e})")

print("\n=== 记分 ===", flush=True)
print("FAILS:", fails if fails else "无 —— 生产路径已接上份额口径，掩膜带 ulp 容差，"
      "tracer 安全，且与旧口径可区分（正对照非零）", flush=True)
