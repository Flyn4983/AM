"""支撑阈值 `bottom>=1` 与 #19 掩膜是**一对**改动：把 2×2 组合量出来，证明不能单独回退任一侧。

跑前登记的判据：
  T1（配对性）对"底面恰好落在 z=0 节点"的贴板件（刀锋档）：
     (新掩膜 solid_mask, `>=1`) 必须 **不** 判需要支撑；
     (旧掩膜 sdf<0, `>=1`) 必须 **判** 需要支撑（即"旧掩膜 + 新阈值"会造假支撑）。
     两者同时成立 ⇒ 阈值与掩膜耦合，support.py 注释里的"必须同批回退"才有依据。
  T2（不放宽）悬空件（底面离板 ≥1 体素间隙）在 (新, `>=1`) 与 (旧, `>=2`) 下都判需要支撑
     ⇒ 本次改动没有把判据调松，只是把被严格掩膜整层偷走的那一层还回去。
  T3（阳性对照）贴板件上两种掩膜给出的**列底整数必须不同**（新 0 / 旧 1）。若相同，
     说明夹具根本没落在刀锋上 ⇒ T1/T2 全部哑火，本探针不可用于支撑该结论。
另外跑一次**出厂组合**的 `support_auto`，把 volume_fraction 实测值钉在日志里（不止是布尔判定）。
"""
import jax.numpy as jnp
import numpy as np

import amforge.support as S
from amforge.core.contracts import PartGeometry, solid_mask


def box_sdf(center, half, p):
    q = np.abs(np.asarray(p) - np.asarray(center)) - np.asarray(half)
    return (np.linalg.norm(np.maximum(q, 0.0), axis=-1)
            + np.minimum(np.max(q, axis=-1), 0.0))


def make_box(cz, hz, n=6, sp=2e-4):
    xs = np.linspace(-3 * sp, 3 * sp, n)
    zs = np.linspace(0.0, 5 * sp, n)
    X, Y, Z = np.meshgrid(xs, xs, zs, indexing="ij")
    P = np.stack([X, Y, Z], axis=-1)
    return PartGeometry(sdf=jnp.asarray(box_sdf([0., 0., cz], [2e-4, 2e-4, hz], P)),
                        origin=jnp.array([-3 * sp, -3 * sp, 0.0]),
                        spacing=sp, dim=3, name="box")


def bottom_of(geo, strict):
    """列底索引（无零件列 = nz），strict=True 走 #19 前的 `sdf<0`，False 走 solid_mask。"""
    part = ((geo.sdf < 0.0) if strict else solid_mask(geo.sdf)).astype(jnp.float64)
    return S._column_bottom(part)


CASES = {"plate(底面=z0 刀锋)": (2.5e-4, 2.5e-4), "floating(离板 1 体素)": (3.5e-4, 1.5e-4)}
nz = 6
btm = {}
print("=== T3/T1/T2：列底整数 + 两种阈值下的 needs 列数 ===", flush=True)
for name, (cz, hz) in CASES.items():
    geo = make_box(cz, hz)
    for strict, mask_name in ((False, "new"), (True, "old")):
        b = bottom_of(geo, strict)
        has_part = (b < nz)
        bmin = int(jnp.min(jnp.where(has_part, b, nz)))
        for thr in (1, 2):
            needs = has_part & (b >= thr)
            ncol = int(jnp.sum(needs))
            btm[(name, mask_name, thr)] = ncol
            print(f"  {name:22s} mask={mask_name:3s} bottom(最小)={bmin:2d}  "
                  f"阈值>={thr} ⇒ needs 列数={ncol:3d}", flush=True)

print("\n=== 出厂组合的 support_auto 实测体积占比 ===", flush=True)
vf = {}
for name, (cz, hz) in CASES.items():
    geo = make_box(cz, hz)
    sup = S.support_auto(geometry=geo, params={"kind": "block"})
    vf[name] = float(sup.volume_fraction)
    print(f"  {name:22s} volume_fraction={vf[name]:.6f}  "
          f"contact_area={float(sup.contact_area):.6e}", flush=True)

fails = []
p_new1 = btm[("plate(底面=z0 刀锋)", "new", 1)]
p_old1 = btm[("plate(底面=z0 刀锋)", "old", 1)]
if not (p_new1 == 0 and p_old1 > 0):
    fails.append(f"T1 配对性未成立：plate needs(new,>=1)={p_new1}（应 0）"
                 f"，needs(old,>=1)={p_old1}（应 >0）")
f_new1 = btm[("floating(离板 1 体素)", "new", 1)]
f_old2 = btm[("floating(离板 1 体素)", "old", 2)]
if not (f_new1 > 0 and f_old2 > 0):
    fails.append(f"T2 未成立：floating needs(new,>=1)={f_new1} needs(old,>=2)={f_old2}（都应 >0）")
if vf["plate(底面=z0 刀锋)"] >= 1e-9:
    fails.append("出厂组合下贴板件仍生成支撑")
if vf["floating(离板 1 体素)"] <= 1e-3:
    fails.append("出厂组合下悬空件支撑体积过小")
print(f"\n=== 记分 ===\nFAILS: {fails if fails else '无（T1/T2/T3 全部成立）'}", flush=True)
