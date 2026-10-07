"""#25 第 4 轮（机制侧）：δ=0.5·dx 那一列在 #23 的热容口径变更下是否**恒等**？

第 3 轮（`am_t25_round3_observable.log`）留了两条没被解释的读数：
1. **R3 附带**：δ=0.5 两档上 `Vn == Vm` 逐格相等 ⇒ 熔化集合里没有部分填充单元。
2. **本轮起飞前的档案核对**：② 代（`am_t2_a0_observable.log`，#19 之后、#23 之前）与
   ③ 代（第 3 轮，#23 之后）的 δ=0.5 那一行 **8 个打印值全部相同**
   （pk 2433.15 / 2565.18 K，Vm=Vn 0.06400 / 0.06473 mm³，ΔH 1.66710 / 1.58948 J），
   而 δ=0 与 δ=0.25 的各行**全部不同**。

若"相同"只是打印精度够不着，δ=0.5 列就是一次碰巧没动；若它是 `cap_w ≡ 1` 的**代数后果**
（IEEE：任何 `x/1.0` 逐位等于 `x`；空白格分子带 `fv` 因子 ⇒ `0/cap_w = 0` 也逐位不变），
那 δ=0.5 列就**永远不可能**判出任何热容口径问题——后者才是「δ=0.5·dx 不携带信息」这条
登记规则的**机理**。本脚本不改求解器，只做两副口径的对照。

跑前登记的判据（不许跑后改）：
  V1 结构：δ=0.5 的 `fv` 必须落在整数上 —— `max|fv − round(fv)| < 1e-12`
      （容差理由：±0.5dx 的 sdf 本身是浮点加减的产物，平局点的 ulp 抖动在第 16–17 位，
       而真切割单元的该指标是 O(0.1) 量级 ⇒ 两档差 4 个数量级，判据不模糊）。
      正对照（必须能失败）：δ=0 与 δ=0.25 上同一指标必须 **≥ 1e-12**。
  V2 受力集合：δ=0.5 上**被加热的非空单元**（`peak > T_preheat` 且 `fv > 1e-12`，
      含 fv=0.5 的刀锋面）其 `cap_w = max(fv, 0.5)` 必须**逐位等于 1.0**（`==` 而非近似），
      且该类单元数 > 0。
      正对照：δ=0 / dx=25 上必须存在 `cap_w != 1.0` 且被加热的单元，否则 V2 是空话。
  V3 主判据（逐位零杠杆）：同一夹具同一 (dx, δ=0.5)，**old**（`533c90f` ＝ #23 之前的
      整棵 src 树，`git diff --stat 533c90f d7404f4 -- src` 实测只含 `thermal_enthalpy.py`
      一个文件 52 增 3 删）与 **new**（工作树 `d7404f4` ＝ #23）各起独立子进程跑，
      四个观测量 `pk / Vm / Vn / dH` 必须以 `float.hex()` **逐位相同**；
      正对照：同一对口径在 δ=0 上必须**逐位不同**（#23 的效应实测 11.040%→13.270%）。
      ⇒ 若 δ=0.5 也不同，V3 FAILS，本轮的机理解释被否证，第 3 轮那条附带读数退回为巧合。
  V4 档案对照（非判据）：old 的 δ=0.5 读数应复现 ② 代打印值、new 的应复现 ③ 代
      （两副都打印 `float.hex` 与 `.2f/.5f` 两种口径，逐位由 V3 判、打印级由本节核对）。
  V5 设备行：`jax.devices()` 与 `nvidia-smi` 写进**日志本体**（第 3 轮 R4 的整改项）。

裁决口径也跑前写死：V1∧V2∧V3 ⇒ 结论＝「δ=0.5 列对 `cap_w` 变更是**代数恒等**的，
因此它既不能为任何观测量提供入选证据、也不能否证它们」；O1 里 δ=0.5 那一列**保留不删**
（删＝放宽判据），只在结论里标注它为何无信息。若 V3 只在部分点成立，照实分点登记。
"""
import json
import os
import subprocess
import sys

EXT = (1.2e-3, 0.6e-3, 0.4e-3)
T_PRE = 400.0
TOL = 1e-12


def h2f(h):
    """`float.hex()` 的反向：读数一律以十六进制字符串跨进程传递，保证逐位可比。"""
    return float.fromhex(h)


def coupon(dx, delta):
    import numpy as np
    from amforge.geometry import from_sdf_fn
    bx, by, bz = EXT
    b = [(-bx / 2 - delta, bx / 2 + delta), (-by / 2 - delta, by / 2 + delta),
         (-bz / 2 - delta, bz / 2 + delta)]

    def fn(p):
        x = np.asarray(p, dtype=np.float64)
        return np.maximum(np.maximum(np.abs(x[..., 0]) - bx / 2,
                                      np.abs(x[..., 1]) - by / 2),
                          np.abs(x[..., 2]) - bz / 2)

    return from_sdf_fn(fn, bounds=b, spacing=dx, name="coupon")


def plan():
    from amforge.core.contracts import ProcessPlan
    return ProcessPlan.uniform(1, modality="SLM", laser_power=600.0, scan_speed=0.8,
                               layer_thickness=EXT[2], hatch_spacing=1.4 * 100e-6,
                               beam_radius=100e-6, absorption=0.45,
                               preheat_temp=T_PRE)


def run(dx_um, frac):
    """在当前 PYTHONPATH 的 amforge 上跑一个 (dx, δ) 点，返回可逐位比较的 hex 读数。"""
    import numpy as np
    import jax
    from amforge.core.contracts import solid_mask, solid_weight
    from amforge.materials import get_material
    from amforge.thermal_enthalpy import enthalpy_of_temperature, solve_enthalpy_thermal
    import amforge.thermal_enthalpy as te

    gen = "new(#23)" if hasattr(te, "CUT_CAPACITY_FLOOR") else "old(pre-#23)"
    dx = dx_um * 1e-6
    cell = dx ** 3
    g = coupon(dx, frac * dx)
    th = solve_enthalpy_thermal(geometry=g, process=plan(),
                                params={"material": "316L"})
    mat = get_material("316L")
    Tl, Ts, Ta = float(mat.T_liquidus), float(mat.T_solidus), float(mat.T_ambient)
    rho, cp, Lk = float(mat.rho_solid), float(mat.cp_solid), float(mat.latent_fusion)
    fv = np.asarray(solid_weight(g.sdf, dx), dtype=np.float64)
    mk = np.asarray(solid_mask(g.sdf), dtype=np.float64)
    pkf = np.asarray(th.peak_temperature, dtype=np.float64)
    fin = np.asarray(th.final_temperature, dtype=np.float64)
    H_fin = np.asarray(enthalpy_of_temperature(fin, rho=rho, cp=cp, L=Lk, T_amb=Ta,
                                               T_sol=Ts, T_liq=Tl), dtype=np.float64)
    H_pre = float(enthalpy_of_temperature(np.float64(T_PRE), rho=rho, cp=cp, L=Lk,
                                          T_amb=Ta, T_sol=Ts, T_liq=Tl))
    liq = pkf > Tl
    heated = pkf > T_PRE
    cap_w = np.maximum(fv, 0.5)
    hpos = heated & (fv > TOL)          # 被加热的**非空**单元（含 fv=0.5 的刀锋面）
    n_h = int(np.count_nonzero(hpos))
    dev = float(np.max(np.abs(fv - np.round(fv))))
    cw_h = cap_w[hpos]
    return {
        "gen": gen, "dx_um": dx_um, "frac": frac,
        "shape": list(g.sdf.shape), "nvox": int(np.prod(g.sdf.shape)),
        "fv_dev": dev,
        "n_fv_partial": int(np.count_nonzero((fv > TOL) & (fv < 1.0 - TOL))),
        "n_fv_empty": int(np.count_nonzero(fv <= TOL)),
        "n_fv_full": int(np.count_nonzero(fv >= 1.0 - TOL)),
        "n_heated": int(np.count_nonzero(heated)),
        "n_heated_nonempty": n_h,
        "n_heated_capw_ne1": int(np.count_nonzero(cw_h != 1.0)),
        "capw_min": float(cw_h.min()).hex() if n_h else None,
        "capw_max": float(cw_h.max()).hex() if n_h else None,
        "pk": float(pkf.max()).hex(),
        "Vm": float(np.sum(np.where(liq, fv, 0.0)) * cell * 1e9).hex(),
        "Vn": float(np.sum(liq & (mk > 0.5)) * cell * 1e9).hex(),
        "dH": float(np.sum(fv * (H_fin - H_pre)) * cell).hex(),
        "devices": str(jax.devices()),
    }


def worker(dx_um, frac):
    print("JSON|" + json.dumps(run(float(dx_um), float(frac))), flush=True)


def parent(new_src, old_src, self_path):
    import jax

    def call(src, dx_um, frac):
        code = (f"import sys; sys.argv=['x','worker','{dx_um}','{frac}'];"
                f"exec(open({self_path!r}).read())")
        env = {"PYTHONPATH": src, "JAX_ENABLE_X64": "1", "HOME": os.environ["HOME"],
               "PATH": os.environ["PATH"], "XLA_FLAGS": os.environ.get("XLA_FLAGS", "")}
        p = subprocess.run([sys.executable, "-c", code], env=env,
                           capture_output=True, text=True)
        good = [ln for ln in p.stdout.splitlines() if ln.startswith("JSON|")]
        if p.returncode != 0 or not good:
            print("SUBPROCESS-FAIL rc=%s:\n%s" % (p.returncode, p.stderr[-1500:]),
                  flush=True)
            raise SystemExit(9)
        return json.loads(good[-1][5:])

    def row(r):
        return (f"{r['gen']:15s} dx={r['dx_um']:5.1f} δ={r['frac']:4.2f} "
                f"shape={tuple(r['shape'])} fv_dev={r['fv_dev']:.3e} "
                f"partial={r['n_fv_partial']:6d} empty={r['n_fv_empty']:6d} "
                f"full={r['n_fv_full']:7d} | heated={r['n_heated']:5d} "
                f"非空={r['n_heated_nonempty']:5d} "
                f"heated_capw≠1={r['n_heated_capw_ne1']:5d}")

    print("=== V5 设备行（写进日志本体） ===", flush=True)
    print(f"  jax.devices() = {jax.devices()}   backend={jax.default_backend()}",
          flush=True)
    fails = []

    print("\n=== V1/V2 结构段（工作树口径，五个对齐点） ===", flush=True)
    struct = {}
    for dx_um, frac in ((50., 0.5), (25., 0.5), (50., 0.), (25., 0.), (25., 0.25)):
        r = call(new_src, dx_um, frac)
        struct[(dx_um, frac)] = r
        print("  " + row(r), flush=True)
        if r["capw_min"] is not None:
            print(f"      cap_w(被加热的非空单元) ∈ [{h2f(r['capw_min']):.17g}, "
                  f"{h2f(r['capw_max']):.17g}]", flush=True)
    for dx_um in (50., 25.):
        got = struct[(dx_um, 0.5)]["fv_dev"]
        ok = got < TOL
        print(f"  V1 δ=0.5 dx={dx_um}: max|fv−round(fv)| = {got:.3e} < 1e-12 ⇒ "
              f"{'PASS' if ok else 'FAILS'}", flush=True)
        if not ok:
            fails.append(f"V1@δ=0.5,dx={dx_um}")
    ctrl = {k: struct[k]["fv_dev"] >= TOL for k in ((50., 0.), (25., 0.), (25., 0.25))}
    print(f"  V1 正对照（δ=0/dx50、δ=0/dx25、δ=0.25/dx25 有真切割单元）＝"
          f"{sum(ctrl.values())}/3 ⇒ {'PASS' if all(ctrl.values()) else 'FAILS（探针看不见切割）'}",
          flush=True)
    if not all(ctrl.values()):
        fails.append("V1-positive-control")
    for dx_um in (50., 25.):
        r = struct[(dx_um, 0.5)]
        ok = (r["n_heated_capw_ne1"] == 0) and (r["n_heated_nonempty"] > 0)
        print(f"  V2 δ=0.5 dx={dx_um}: 被加热的非空单元 {r['n_heated_nonempty']} 格，"
              f"其中 cap_w≠1.0 的 {r['n_heated_capw_ne1']} 格 ⇒ "
              f"{'PASS' if ok else 'FAILS'}", flush=True)
        if not ok:
            fails.append(f"V2@dx={dx_um}")
    rc = struct[(25., 0.)]
    ok = rc["n_heated_capw_ne1"] > 0
    print(f"  V2 正对照（δ=0/dx25 有 cap_w≠1.0 且被加热的单元）＝"
          f"{rc['n_heated_capw_ne1']} 格 ⇒ {'PASS' if ok else 'FAILS（V2 空话）'}",
          flush=True)
    if not ok:
        fails.append("V2-positive-control")

    print("\n=== V3 主判据：两副口径逐位（float.hex）对照 ===", flush=True)
    obs = ("pk", "Vm", "Vn", "dH")
    for dx_um, frac, expect in ((25., 0.5, "same"), (50., 0.5, "same"),
                                (25., 0.0, "differ"), (50., 0.0, "differ")):
        ro, rn = call(old_src, dx_um, frac), call(new_src, dx_um, frac)
        print("  " + row(ro), flush=True)
        print("  " + row(rn), flush=True)
        same = {k: (ro[k] == rn[k]) for k in obs}
        ok = all(same.values()) if expect == "same" else not any(same.values())
        print(f"  V3 dx={dx_um} δ={frac} 期望**{expect}** ⇒ "
              f"{'PASS' if ok else 'FAILS'}   "
              + " ".join(f"{k}:{'==' if same[k] else '≠'}" for k in obs), flush=True)
        if not ok:
            fails.append(f"V3@dx={dx_um},δ={frac}")
        if expect == "same" and not all(same.values()):
            # 把"约化顺序噪声"与"口径效应"分开：前者相对差 ~1e-16，后者 ≥1e-3
            rel = {k: abs(h2f(ro[k]) - h2f(rn[k])) / max(abs(h2f(rn[k])), 1e-300)
                   for k in obs if not same[k]}
            print("    差值量级（非判据，只用于归因）："
                  + "  ".join(f"{k} 相对差 {v:.3e}" for k, v in rel.items()), flush=True)
        print("    old " + " ".join(f"{k}={ro[k]}" for k in obs), flush=True)
        print("    new " + " ".join(f"{k}={rn[k]}" for k in obs), flush=True)

    print("\n=== V4 档案对照（非判据）：δ=0.5 两档对 ②/③ 代打印值 ===", flush=True)
    for src, label in ((old_src, "old→② 代"), (new_src, "new→③ 代")):
        for dx_um, want in ((50., "2433.15K 0.06400mm³ 1.66710e+00J"),
                            (25., "2565.18K 0.06473mm³ 1.58948e+00J")):
            r = call(src, dx_um, 0.5)
            print(f"  {label} dx={dx_um}：pk={h2f(r['pk']):.2f}K "
                  f"Vm={h2f(r['Vm']):.5f} Vn={h2f(r['Vn']):.5f}mm³ "
                  f"ΔH={h2f(r['dH']):.5e}J   档案 {want}", flush=True)

    print(f"\nFAILS: {fails if fails else '无（V1/V2/V3 全过 ⇒ δ=0.5 列的零杠杆成立）'}",
          flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "worker":
        worker(sys.argv[2], sys.argv[3])
    else:
        sys.exit(parent(sys.argv[1], sys.argv[2], os.path.abspath(__file__)))
