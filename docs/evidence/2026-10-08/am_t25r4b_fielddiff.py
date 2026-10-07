"""#25 第 4 轮的字段级补测（4b）：δ=0.5 列上两副口径**整场**差多少？

第 4 轮（`am_t25r4_halfcell_noop.log`）实测：δ=0.5 两档上 `pk/Vm/Vn` 逐位相同、`dH` 差 1 ulp，
但四个观测量恒等 ≠ 温度场恒等（峰值是 max、熔体积是集合计数，低位的场差看不见）。
本轮把结论建立在场上：old（`533c90f`＝#23 前）与 new（`d7404f4`＝#23）各存 `peak_temperature`
与 `final_temperature` 整场，父进程算逐元素差。

跑前登记的判据（不许跑后改）：
  F1 δ=0.5 / dx=25：`max|T_new − T_old| / max|T_old| < 1e-12`（两个场都要满足）。
  F2 正对照 δ=0 / dx=25：同一指标必须 **> 1e-3**（#23 在刀锋对齐上是真效应，实测 3.969% 级）。
      ⇒ 若 δ=0.5 也 >1e-3，F1 FAILS，"δ=0.5 非信息"退回为"只是观测量碰巧相同"。
  F3 记录差的**空间结构**（非判据）：给出 max|ΔT| 出现的单元的 fv、该单元是否在被加热集合内、
      以及 `|ΔT|>0` 的单元数 —— 用来确认差确实来自第 4 轮 V2 数到的那批 `cap_w=1−ulp` 单元。
  F4 设备行写进日志本体。
"""
import json
import os
import subprocess
import sys

EXT = (1.2e-3, 0.6e-3, 0.4e-3)
T_PRE = 400.0


def dump(dx_um, frac, out_path):
    import numpy as np
    import jax
    from amforge.core.contracts import solid_weight
    from amforge.geometry import from_sdf_fn
    from amforge.thermal_enthalpy import solve_enthalpy_thermal
    import amforge.thermal_enthalpy as te
    from amforge.core.contracts import ProcessPlan
    from amforge.materials import get_material

    bx, by, bz = EXT
    dx = float(dx_um) * 1e-6
    d = float(frac) * dx
    b = [(-bx / 2 - d, bx / 2 + d), (-by / 2 - d, by / 2 + d), (-bz / 2 - d, bz / 2 + d)]

    def fn(p):
        x = np.asarray(p, dtype=np.float64)
        return np.maximum(np.maximum(np.abs(x[..., 0]) - bx / 2,
                                      np.abs(x[..., 1]) - by / 2),
                          np.abs(x[..., 2]) - bz / 2)

    g = from_sdf_fn(fn, bounds=b, spacing=dx, name="coupon")
    pl = ProcessPlan.uniform(1, modality="SLM", laser_power=600.0, scan_speed=0.8,
                             layer_thickness=EXT[2], hatch_spacing=1.4 * 100e-6,
                             beam_radius=100e-6, absorption=0.45, preheat_temp=T_PRE)
    th = solve_enthalpy_thermal(geometry=g, process=pl, params={"material": "316L"})
    fv = np.asarray(solid_weight(g.sdf, dx), dtype=np.float64)
    np.savez(out_path,
             peak=np.asarray(th.peak_temperature, dtype=np.float64),
             final=np.asarray(th.final_temperature, dtype=np.float64),
             fv=fv)
    print("OK|" + json.dumps({"gen": "new(#23)" if hasattr(te, "CUT_CAPACITY_FLOOR")
                              else "old(pre-#23)", "devices": str(jax.devices()),
                              "path": out_path}), flush=True)


def main(new_src, old_src, self_path, workdir):
    import numpy as np
    import jax
    print("=== F4 设备行 ===", flush=True)
    print(f"  jax.devices() = {jax.devices()}   backend={jax.default_backend()}",
          flush=True)

    def call(src, dx_um, frac, tag):
        out = os.path.join(workdir, f"{tag}.npz")
        code = (f"import sys; sys.argv=['x','dump','{dx_um}','{frac}',{out!r}];"
                f"exec(open({self_path!r}).read())")
        env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
        env["PYTHONPATH"] = src
        env["JAX_ENABLE_X64"] = "1"
        p = subprocess.run([sys.executable, "-c", code], env=env,
                           capture_output=True, text=True)
        if p.returncode != 0:
            print("SUBPROCESS-FAIL:", p.stderr[-1200:], flush=True)
            raise SystemExit(9)
        return out

    fails = []
    for dx_um, frac, expect in ((25., 0.5, "tiny"), (25., 0.0, "large")):
        po = call(old_src, dx_um, frac, f"old_{int(dx_um)}_{frac}")
        pn = call(new_src, dx_um, frac, f"new_{int(dx_um)}_{frac}")
        a, b = np.load(po), np.load(pn)
        for fld in ("peak", "final"):
            ta, tb = a[fld], b[fld]
            rel = float(np.max(np.abs(tb - ta)) / max(np.max(np.abs(ta)), 1e-300))
            ndiff = int(np.count_nonzero(tb != ta))
            ok = (rel < 1e-12) if expect == "tiny" else (rel > 1e-3)
            print(f"  F1/F2 dx={dx_um} δ={frac} 场=`{fld}`：相对最大差 = {rel:.3e} "
                  f"（期望 **{expect}**）⇒ {'PASS' if ok else 'FAILS'}   "
                  f"逐元素不等的单元数 = {ndiff}/{tb.size}", flush=True)
            if not ok:
                fails.append(f"{expect}@dx={dx_um},δ={frac},{fld}")
            if expect == "tiny":
                # F3：差的空间结构
                dT = np.abs(tb - ta).ravel()
                idx = int(np.argmax(dT))
                fv_flat = a["fv"].ravel()
                cw = np.maximum(fv_flat, 0.5)
                n_capw_ne1 = int(np.count_nonzero(cw != 1.0))
                print(f"    F3 max|ΔT|={dT.ravel()[idx]:.3e}K 出现在 fv={fv_flat[idx]:.17g}"
                      f"（cap_w={cw[idx]:.17g}，该格 cap_w≠1 ⇒ {'是' if cw[idx] != 1.0 else '否'}）"
                      f"；全域 cap_w≠1 的单元数 = {n_capw_ne1}", flush=True)
                print(f"    F3 |ΔT|>0 的单元中 cap_w≠1 的占比 = "
                      f"{np.count_nonzero((dT > 0) & (cw != 1.0))}/{ndiff}", flush=True)
    print(f"\nFAILS: {fails if fails else '无（δ=0.5 整场差在机器精度内，δ=0 差为真效应）'}",
          flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "dump":
        dump(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        sys.exit(main(sys.argv[1], sys.argv[2], os.path.abspath(__file__), sys.argv[3]))
