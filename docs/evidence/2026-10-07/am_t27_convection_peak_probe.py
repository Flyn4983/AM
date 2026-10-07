"""#23 实现后的**新红灯**归因：`test_boundary.py::test_convection_removes_energy` 第 ② 条。

现象（`am_t23_impl_targeted.log`，CPU 钉住，HEAD 533c90f + 工作树 #23）：
    assert peaks[2] < peak0 − 0.02·peak0 ⇒ 2908.771450618092 vs 2937.060997302282（只差 0.96%）
改前该测试自己登记的表是 h=20000 ⇒ 峰值 **−5.39%**、末态实体均值 **−573.7 K**。
同一轮里 A0 的 assert1（峰值 5.240%）转绿、assert2 落到 **13.270%**（与改前打分 V1 登记的
13.270% 到小数第 3 位相同）⇒ 那半边是**预测命中**；这一条是**预测之外**的新红灯，必须单独归因，
不许顺手把测试数字改成实测值。

假设（本轮要证伪的对象）**H**：#23 把整条右端项除以 `cap_w=max(fv,0.5)`，于是
**蒸发封顶项在切割单元被放大 `fv/cap_w ≤ 1`→ 有效项从 `fv·S` 变回 `S`（最坏 2 倍）**，
峰值被钳得更硬；表面对流因此**动不了**峰值——即"对流压峰"从 5.39% 塌到 <1% 是**封顶变硬
掩盖**了对流，而**不是**对流散热被实现吃掉。

三条独立读数（两条口径 × 封顶开关）：
  口径 `old`＝把 `CUT_CAPACITY_FLOOR` 置 1.0（此时 `cap_w≡1`，rhs 逐项退回改前生产）；
  口径 `new`＝生产实现（0.5）。
  封顶 `on`＝缺省 `evap_coeff=3e11`；`off`＝`evap_coeff=1e-6`（把封顶项打到可忽略）。
  夹具＝测试自己那套（`build_primitive("box", 0.4mm, 80µm)` + `_plan()` 4 层 200W/1.0m·s⁻¹ +
  5 面对流 T_inf=293，h∈{0,200,2000,20000}），另取 `peak_temperature>T_liquidus` 的
  体素计数体积 Vn 与 fv 加权体积 Vm。**两副口径 × 两种封顶状态共 4 组全部实跑，不引用文档数字。**

跑前登记判据（不许事后改）：
  U0 同构对照（任一失败 ⇒ 本轮不参与判据，且不许动 `tests/`）：
     U0a `old` 口径必须复现测试 docstring 的改前表：末态均值 Δ ∈ {−8.1,−81.6,−573.7}K **±0.2K**、
         峰值 Δ ∈ {+0.03,+0.10,−5.39}% **±0.05pt**。（docstring 是四舍五入数 ⇒ 只能给带宽。）
     U0b `new` 口径必须**逐位**复现本轮 pytest 打印的两个锚点：
         peak0 == 2937.060997302282 且 peaks(h=20000) == 2908.771450618092。
         ⇒ 硬锚：证明探针的 BC/夹具构造与测试**同构**；否则下面的机制归属全部作废。
     U0c 出处对照（本轮追加的**结构性**对照，不新增数值判据、只把"两副口径真的来自两份源码"
         钉死：old 读数出自 /tmp 副本、new 出自工作树、副本里不含 `cap_w`）：任一项不成立 ⇒ 作废。
  U1 机制判据（**决定性**，对 H 的正/反证）：
     U1a 封顶确实在场：`new` 的 peak0 > T_evap_lo = T_boil−200 = 2890 K，且 argmax 单元 fv==1.0。
     U1b 封顶变硬（主判据）：`off` 组里 **new 的 |Δpeak%(h=20000)| > old 的同项**
         ⇒ 实测"对流在 new 口径下更强"，从而 5.39%→0.96% 只能由**封顶掩盖**解释。
         若反向（关掉封顶后 new 仍不比 old 敏感）⇒ 实现把对流/BC 项**吃掉**了 ⇒ 判 bug，
         **测试一字不动**，回头查 `rhs`。
     U1c 单变量对照：`on` 组内 new−old 的 Δpeak% 之差与 `off` 组内之差**符号相反**
         ⇒ 封顶开关才是那 4 个百分点的来源，而不是别的东西顺带动了峰值。
  U2 能量守护不许退化：new 口径的末态实体均值 Δ 随 h 单调下降，且 |Δ(h=20000)| ≥ old 的同项
     ⇒ 排除"散热变弱所以峰值不动"这种反向解释。（测试自身的两条均值断言本轮仍为绿。）
  U3 处置（唯一允许的测试改法，防止照新数抠门槛）：
     U0∧U1∧U2 **全过** ⇒ 才允许把 ② 第二条换成两条**口径无关**的同语义守护：
        (i) peaks 随 h 严格单调下降；(ii) h=20000 的熔化体积 Vn 相对 h=0 至少减少 **10%**，
        且 (i)(ii) 必须在 **old 与 new 两副口径下同时实测成立**（在 old 上不成立 ⇒ 该守护是
        照着 new 的数抠出来的，作废，不许写入测试）；docstring 须同时登 old/new 两列实测表。
     任一不过 ⇒ `tests/test_boundary.py` 一字不动。
     **不许**：删除该守护、把方向反掉、把 2% 改成"实测差 0.96%"这种照数定阈值、或放宽 A0。

范围与边界（照实写）：CPU 钉住（`CUDA_VISIBLE_DEVICES=""`）——U0b 要和 pytest 的**逐位**浮点对比，
跨设备比较无效；本探针只做**归因**，不是性能测量。`off` 组失去蒸发封顶后峰值可能非物理地偏高，
只作机制读数、不作物理结论，照实打印。子进程各跑一副口径（同一台机、同一确定性路径），
父进程只做判据、不重解。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OLD_SRC = "/tmp/am_t27_old_src"
OLD_HEAD = "533c90f"          # 改 #23 之前的提交（工作树里 thermal_enthalpy.py 是唯一脏文件）
HS = (0.0, 200.0, 2000.0, 20000.0)
DOC_MEAN = (-8.1, -81.6, -573.7)      # 测试 docstring 登记的改前表（末态实体均值 Δ）
DOC_PEAK = (0.03, 0.10, -5.39)        # 同表（峰值 Δ，%）
ANCHOR_PEAK0 = 2937.060997302282      # 本轮 pytest 打印（new 口径）
ANCHOR_PEAK_H = 2908.771450618092     # 同上，h=20000
T_EVAP_LO = 3090.0 - 200.0


def child(pkg_root: str, floor: float, evap: float) -> dict:
    """在子进程里跑一副口径，返回 JSON 读数。floor=1.0 ⇒ cap_w≡1 ⇒ 改前生产。"""
    code = r'''
import json, os, sys
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
FLOOR = float(os.environ["T27_FLOOR"]); EVAP = float(os.environ["T27_EVAP"])
import amforge.thermal_enthalpy as TE
from amforge.boundary import BoundaryCondition, BoundaryCollection
from amforge.core.contracts import ProcessPlan, solid_mask, solid_weight
from amforge.gui.preproc import build_primitive
from amforge.materials import get_material
TE.CUT_CAPACITY_FLOOR = FLOOR          # 必须在调用前替换模块常量（rhs 里读的就是它）
mat = get_material("316L"); TL = float(mat.T_liquidus)
part = build_primitive("box", length_mm=0.4, spacing_um=80.0)
plan = ProcessPlan.uniform(4, modality="SLM", laser_power=200.0, scan_speed=1.0,
                           layer_thickness=40e-6, hatch_spacing=80e-6, beam_radius=50e-6,
                           absorption=0.4, preheat_temp=373.0)
dx = float(part.spacing)
sol = np.asarray(solid_mask(np.asarray(part.sdf)) > 0.5)
fv = np.asarray(solid_weight(jnp.asarray(part.sdf), jnp.asarray(dx, dtype=jnp.float64)))
out = {"shape": list(part.sdf.shape), "dx": dx, "n_solid": int(sol.sum()),
       "te_file": TE.__file__, "floor_now": float(TE.CUT_CAPACITY_FLOOR), "runs": []}
def plan_of(h):
    if h == 0.0:
        return plan, {"material": "316L", "evap_coeff": EVAP}
    strong = BoundaryCollection(bcs=tuple(
        BoundaryCondition("convection", f, h, 293.0)
        for f in ("+Z", "+X", "-X", "+Y", "-Y")), ic=None)
    return plan, {"material": "316L", "boundary_conditions": strong, "evap_coeff": EVAP}
for h in (0.0, 200.0, 2000.0, 20000.0):
    p, params = plan_of(h)
    th = TE.solve_enthalpy_thermal(geometry=part, process=p, params=params)
    pk = np.asarray(th.peak_temperature); tf = np.asarray(th.final_temperature)
    assert pk.shape == fv.shape == sol.shape, (pk.shape, fv.shape, sol.shape)
    idx = np.unravel_index(int(np.argmax(pk)), pk.shape)   # 多维数组：扁平索引必须还原
    fv_peak = float(fv[idx])
    melted = (pk > TL) & sol
    out["runs"].append(dict(
        h=h, peak=float(pk.max()), peak_idx=list(int(v) for v in idx), argmax_fv=fv_peak,
        argmax_above_evap_lo=bool(float(pk.max()) > %r),
        n_above_evap_lo=int(((pk > %r) & sol).sum()),
        mean_final_solid=float(tf[sol].mean()),
        vn=float(melted.sum()) * dx ** 3 * 1e9,
        vm=float((fv * melted).sum()) * dx ** 3 * 1e9,
        peak_final=float(tf.max()), finite=bool(np.isfinite(pk).all() and np.isfinite(tf).all()),
    ))
print("JSON::" + json.dumps(out))
''' % (T_EVAP_LO, T_EVAP_LO)
    env = dict(os.environ, PYTHONPATH=pkg_root, CUDA_VISIBLE_DEVICES="",
               JAX_ENABLE_X64="1", T27_FLOOR=repr(floor), T27_EVAP=repr(evap))
    t0 = time.perf_counter()
    pr = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                        text=True, cwd=REPO)
    el = time.perf_counter() - t0
    if pr.returncode != 0:
        raise SystemExit(f"子进程失败（pkg={pkg_root} floor={floor} evap={evap}）：\n"
                         f"{pr.stdout[-2000:]}\n{pr.stderr[-3000:]}")
    line = [ln for ln in pr.stdout.splitlines() if ln.startswith("JSON::")]
    if not line:
        raise SystemExit(f"子进程无读数：\n{pr.stdout[-2000:]}\n{pr.stderr[-2000:]}")
    d = json.loads(line[0][6:])
    d["_elapsed"] = el
    return d


def make_old_src() -> str:
    """把 HEAD(改 #23 前) 的整棵 src 复制到 /tmp，只替换 thermal_enthalpy.py 为该提交版本。"""
    import shutil
    if os.path.isdir(OLD_SRC):
        shutil.rmtree(OLD_SRC)
    shutil.copytree(os.path.join(REPO, "src"), OLD_SRC,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    old_txt = subprocess.run(["git", "-C", REPO, "show", f"{OLD_HEAD}:src/amforge/thermal_enthalpy.py"],
                             capture_output=True, text=True, check=True).stdout
    with open(os.path.join(OLD_SRC, "amforge", "thermal_enthalpy.py"), "w") as f:
        f.write(old_txt)
    return OLD_SRC


def deltas(runs):
    m0 = runs[0]["mean_final_solid"]
    p0 = runs[0]["peak"]
    return [r["mean_final_solid"] - m0 for r in runs[1:]], \
        [(r["peak"] - p0) / p0 * 100.0 for r in runs[1:]]


def main():
    print(f"device = {os.environ.get('CUDA_VISIBLE_DEVICES')!r}（父进程只做判据）", flush=True)
    old = make_old_src()
    new_pkg = os.path.join(REPO, "src")
    R = {}
    for tag, pkg, floor in (("old", old, 1.0), ("new", new_pkg, 0.5)):
        for cap, evap in (("on", 3.0e11), ("off", 1.0e-6)):
            R[(tag, cap)] = child(pkg, floor, evap)
            r0 = R[(tag, cap)]["runs"][0]
            print(f"[{tag}/{cap}] te={R[(tag, cap)]['te_file']} "
                  f"grid={R[(tag, cap)]['shape']} n_solid={R[(tag, cap)]['n_solid']} "
                  f"peak0={r0['peak']:.6f} argmax_fv={r0['argmax_fv']:.3f} "
                  f">{T_EVAP_LO:.0f}K={r0['argmax_above_evap_lo']} n_evap={r0['n_above_evap_lo']} "
                  f"Vn={r0['vn']:.6f}mm3 t={R[(tag, cap)]['_elapsed']:.1f}s", flush=True)
            for r in R[(tag, cap)]["runs"][1:]:
                print(f"        h={r['h']:7.0f}: peak={r['peak']:9.4f} "
                      f"({(r['peak'] - r0['peak']) / r0['peak'] * 100:+6.3f}%) "
                      f"Δmean={r['mean_final_solid'] - r0['mean_final_solid']:+8.2f}K "
                      f"Vn={r['vn']:.6f}mm3 ({(r['vn'] - r0['vn']) / r0['vn'] * 100:+7.2f}%) "
                      f"argmax_fv={r['argmax_fv']:.3f} n_evap={r['n_above_evap_lo']} "
                      f"finite={r['finite']}", flush=True)

    bad = []
    print("\n=== U0c 出处对照（父进程硬校：两副口径真的来自两份源码）===")
    with open(os.path.join(OLD_SRC, "amforge", "thermal_enthalpy.py")) as f:
        old_txt = f.read()
    with open(os.path.join(new_pkg, "amforge", "thermal_enthalpy.py")) as f:
        new_txt = f.read()
    prov = [("old 包路径", all(R[k]["te_file"].startswith(OLD_SRC) for k in R if k[0] == "old")),
            ("new 包路径", all(R[k]["te_file"].startswith(new_pkg) for k in R if k[0] == "new")),
            ("old 源码无 cap_w", "cap_w" not in old_txt),
            ("new 源码有 cap_w", "cap_w" in new_txt),
            ("两文件仅差 #23", old_txt != new_txt)]
    for name, ok in prov:
        print(f"  {name:18s} {'✓' if ok else '✗'}")
        if not ok:
            bad.append(f"U0c {name} 不成立 ⇒ 两副口径并非同一棵树的两版本，本轮作废")
    # old 口径的 floor_now 只是探针赋上去的属性，老 rhs 从不读它（U0c 的「old 源码无 cap_w」已钉死）
    print(f"  （old 侧 floor_now={R[('old', 'on')]['floor_now']} 为探针赋值，老 rhs 不读该名）")
    print("\n=== U0a：old 口径必须复现测试 docstring 的改前表（±0.2K / ±0.05pt）===")
    dm, dp = deltas(R[("old", "on")]["runs"])
    for k, (a, b) in enumerate(zip(DOC_MEAN, DOC_PEAK)):
        ok_m = abs(dm[k] - a) <= 0.2
        ok_p = abs(dp[k] - b) <= 0.05
        print(f"  h={HS[k + 1]:7.0f}: Δmean 实测 {dm[k]:+8.2f}K vs 登记 {a:+8.1f} "
              f"{'✓' if ok_m else '✗'}   Δpeak 实测 {dp[k]:+7.3f}% vs 登记 {b:+7.2f}% "
              f"{'✓' if ok_p else '✗'}")
        if not ok_m:
            bad.append(f"U0a h={HS[k + 1]}: 末态均值 Δ {dm[k]:.2f}K 不贴合 docstring {a}K")
        if not ok_p:
            bad.append(f"U0a h={HS[k + 1]}: 峰值 Δ {dp[k]:.3f}% 不贴合 docstring {b}%")

    print("\n=== U0b：new 口径必须逐位复现 pytest 打印的两个锚点 ===")
    runs_new = R[("new", "on")]["runs"]
    for idx, name, anchor in ((0, "peak0", ANCHOR_PEAK0), (3, "peaks[h=20000]", ANCHOR_PEAK_H)):
        v = runs_new[idx]["peak"]
        ok = v == anchor
        print(f"  {name:18s} 探针 {v!r} vs pytest {anchor!r} ⇒ {'逐位相同 ✓' if ok else '不同 ✗'}")
        if not ok:
            bad.append(f"U0b {name}: 探针 {v!r} != pytest {anchor!r} ⇒ 探针与测试不同构，本轮作废")

    print("\n=== U1 机制判据 ===")
    r_new = runs_new[0]
    u1a = r_new["argmax_fv"] == 1.0 and r_new["peak"] > T_EVAP_LO
    print(f"  U1a 封顶在场（new/peak0）：peak={r_new['peak']:.4f} > {T_EVAP_LO:.0f} "
          f"{r_new['peak'] > T_EVAP_LO} 且 argmax_fv={r_new['argmax_fv']:.3f}==1.0 "
          f"{r_new['argmax_fv'] == 1.0} ⇒ {'成立' if u1a else '不成立'}")
    if not u1a:
        bad.append("U1a: new 峰值不在蒸发封顶窗内或 argmax 落在非实体单元")
    d_off_old, dp_off_old = deltas(R[("old", "off")]["runs"])
    d_off_new, dp_off_new = deltas(R[("new", "off")]["runs"])
    u1b = abs(dp_off_new[2]) > abs(dp_off_old[2])
    print(f"  U1b 封顶关闭后（off 组，h=20000 峰值 Δ）：old {dp_off_old[2]:+7.3f}%  "
          f"new {dp_off_new[2]:+7.3f}% ⇒ new 更敏感 ={'是' if u1b else '否'}"
          f"（判据：|new| > |old|，即对流在 new 口径下**更强**）")
    if not u1b:
        bad.append(f"U1b: 关掉封顶后 new 的 h 响应不比 old 强（{dp_off_new[2]:.3f}% vs "
                   f"{dp_off_old[2]:.3f}%）⇒ 对流项被实现吃掉，判 bug，不许动测试")
    dp_new_on = deltas(runs_new)[1]
    cross_on = dp_new_on[2] - dp[2]              # (new − old)，封顶 on
    cross_off = dp_off_new[2] - dp_off_old[2]    # (new − old)，封顶 off
    u1c = cross_on * cross_off < 0.0
    print(f"  U1c 单变量（口径间差 new−old，h=20000 峰值 Δ）：on 组 {cross_on:+7.3f}pt  "
          f"off 组 {cross_off:+7.3f}pt ⇒ 异号 ={'是' if u1c else '否'}"
          f"（封顶开关才是那几 pt 的来源）")
    if not u1c:
        bad.append("U1c: on/off 两组的口径间差不同号 ⇒ 封顶开关不是主因")

    print("\n=== U2 能量守护不许退化 ===")
    dm_new = deltas(runs_new)[0]
    mono = all(dm_new[i] > dm_new[i + 1] for i in range(len(dm_new) - 1))
    not_weaker = abs(dm_new[2]) >= abs(dm[2]) - 1e-9
    print(f"  new 的 Δmean = {dm_new[0]:+.2f}, {dm_new[1]:+.2f}, {dm_new[2]:+.2f} K "
          f"（old {dm[0]:+.2f}, {dm[1]:+.2f}, {dm[2]:+.2f}）⇒ 单调下降 {mono}，"
          f"|Δ(h=20000)| 不小于一 {not_weaker}")
    if not mono:
        bad.append("U2: new 口径末态均值未随 h 单调下降")
    if not not_weaker:
        bad.append("U2: new 口径散热量级小于一 ⇒ 反向解释（对流被削弱）成立")

    print("\n=== U3 处置：候选替换守护必须在**两副口径**同时成立 ===")
    def guard(tag_cap):
        rr = R[tag_cap]["runs"]
        pk = [x["peak"] for x in rr]
        vn = [x["vn"] for x in rr]
        strict = pk[1] >= pk[2] >= pk[3] and pk[3] < pk[0]
        drop = (vn[0] - vn[3]) / vn[0] * 100.0
        return strict, drop, drop >= 10.0
    allow = True
    for tag in ("old", "new"):
        s, d, ok = guard((tag, "on"))
        print(f"  (i) peaks 单调 {s}   (ii) Vn 降幅 h=20000 vs h=0 = {d:+7.3f}% ≥10% {ok}  [{tag}]")
        if not (s and ok):
            allow = False
    if not allow:
        print("  ⇒ 候选守护在某一副口径上不成立 ⇒ **照数抠门槛**，作废；测试一字不动。")
    elif bad:
        print("  ⇒ 判据未全过（见上），**测试一字不动**，回头查实现。")
    else:
        print("  ⇒ U0∧U1∧U2 全过且候选守护口径无关 ⇒ 允许按 U3 改写 ② 第二条并登两列表。")

    print(f"\nFAILS: {bad if bad else '无'}")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
