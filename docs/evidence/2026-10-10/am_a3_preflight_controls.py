"""#47 预检面的**控件面**：证明刚改过的 G4／G5 是真能量，不是为了让面变绿而写的装饰。

跑法（与主面同一只针，CPU 钉住，零 GPU）：

    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu PYTHONPATH=src:docs/evidence/2026-10-10 \
      /tmp/amvenv/bin/python docs/evidence/2026-10-10/am_a3_preflight_controls.py

约定：这里每条 K 的 **PASS ＝ 扰动被判据抓住**（判据报红），不是扰动本身好。
所以本面的绿色＝"控件确实能红"；主面 ``am_a3_driver.py --stage preflight`` 的绿色
才有意义。K0a/K0b 把**被证伪的旧判据**（J12）原样再跑一遍并如实打印 FAIL，
既给判据红路的活体证据，也给出"旧线任何正确实现都过不了"的量化理由。

登记的控件（跑前写死，跑后不回头改）：
  K0a 旧 G4 线「沉能份额 ≥0.99」在顶面探针上必须报红（实测量级 0.62）。
  K0b 旧 G5 线「顶面 nx·ny 全实体」必须报红，且红的原因就是 ``ceil`` 过延伸圈。
  K1  G4a 对**轴向 σ 口径**敏感：把期望式里的 dp 换成 r_b（少 1.2 倍）⇒ 差必须 ≫ 1e-4。
  K2  G4a 对**束面相位**敏感：把束面从 top_z 挪到设计面 z=0 ⇒ 差必须 ≫ 1e-4。
      （这条同时说明 J11 文里"束斑放在顶面上"到底放在哪：差半个顶面节点。）
  K3  亏损**逐笔可分配**：去掩膜的网格总和 == 顶面截获 + 顶面以上空层，三段解析式
      各自命中（这条把"38.4% 哪去了"从口号变成等式）。
  K4  G4b 的深度对照方向性：把探针**上移** 3r_b（束斑落到工件外）⇒ ≥0.99 必须报红。
  K5  G5 的圈定义敏感性：把标称域**放大**到点阵不再过延伸（3300>3017.5µm）⇒ 那圈空节点
      失去合法归属，游荡必须 >0 且圈上必须 =0。**方向记在这**：缩小域只会让圈更宽松、
      判据照样绿，量不出任何东西（本轮第一版就是这么写错的，改后才能红）。
  K6  G5 的足迹敏感性：把读数足迹中心推到 x=域宽（真实轨道之外）⇒ 足迹必须不满布。
  K7  控件的控件：K5/K6 的扰动**必须真的改变掩膜计数**，否则 K5/K6 是空跑。
  X1/X2 **标本区**（不计进 K 的绿数）：把旧 G4／旧 G5 判据原样送进同一套标签器，这两行
      按定义**必须**印成 ``[FAIL]``——主面归档正文里 ``[FAIL]=0``，那个 0 只有配上"打印式子
      确实能输出 FAIL"的活体标本才不是空话。
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "src"))
sys.path.insert(0, HERE)

import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)

import am_a3_driver as D  # noqa: E402
from amforge.core.contracts import solid_mask  # noqa: E402
from amforge.thermal_enthalpy import _cell_integrated_source  # noqa: E402

TOL = 1e-4          # 主面 G4a 的容差，这里复用同一个数，不另设一套
SENS = 10.0 * TOL   # 控件要求的**最小**偏离：扰动必须把差推到容差的 10 倍以上


def main():
    fx = D.load_fixture()
    r_b = D.beam_radius_m()
    dx = r_b / 2.0
    geom = D.plate_geometry(dx, fx["domain_um"])
    xax, yax, zax = D.axes_of(geom)
    solid = np.asarray(solid_mask(geom.sdf)) > 0.5
    top_idx = int(np.max(np.where(solid.any(axis=(0, 1)))[0]))
    top_z = float(zax[top_idx])
    tbl, cols, runs = D.read_strategy("scanStrategyConverging.csv")
    sub, prog = D.program_upto(cols, runs, 1, top_z)
    sub_runs = D.power_runs(np.asarray(sub["columns"])[:, D.COL_POWER])
    n_steps = D.schedule(geom, prog, D.plan_for(fx, r_b, fx["alpha_calibrated"],
                                                fx["T0_C"] + 273.15))[0]
    dp = r_b * 1.2
    z_beam = float(np.asarray(prog.sample_steps(1)[0])[0, 2])
    cap = D.capture_probes(geom, prog, n_steps, r_b)
    e_surf = D.expected_capture(top_z, dx, dp, z_beam)
    e_deep = D.expected_capture(top_z, dx, dp, z_beam - 3.0 * r_b)
    tp = D.toplane_readability(solid, xax, yax, top_idx, fx, sub, sub_runs, r_b, dx)
    print(f"控件面对象：dx={dx*1e6:.3f}µm n_steps={n_steps} 顶面节点 z={top_z*1e6:.3f}µm "
          f"束面 z={z_beam*1e6:.3f}µm dp={dp*1e6:.3f}µm σ_in={r_b/math.sqrt(2)*1e6:.3f}µm")
    print(f"实测探针：顶面={cap['surface_lo']:.6f} 下移3r_b={cap['deep_lo']:.6f} "
          f"面外={cap['outside_lo']:.6f}；解析：顶面={e_surf:.6f} 深度={e_deep:.6f}")
    print(f"G5 现量：顶面空节点={tp['n_void']}（圈上 {tp['n_ring_void']}／圈外 "
          f"{tp['n_stray']}）足迹 {tp['n_foot']} 格全实体="
          f"{tp['n_foot'] == tp['n_foot_solid']}")
    rows = []

    def add(tag, ok, detail):
        rows.append((tag, bool(ok), detail))

    # K0a/K0b：被 J12 证伪的**旧判据**原样再判一次，必须红——红路活体证据。
    add("K0a 旧 G4 线（份额≥0.99）现在报红 ⇒ 判据不是摆设，旧线确实过不了",
        not (cap["surface_lo"] >= 0.99),
        f"旧线判 {cap['surface_lo']:.6f} ≥ 0.99 → {cap['surface_lo'] >= 0.99}；"
        f"它和同批登记的 J11（『norm 臂＝1/G4 份额，份额≈0.6』）逻辑上不可能同绿")
    add("K0b 旧 G5 线（顶面全实体）现在报红 ⇒ 红因＝ceil 过延伸圈而非缺陷",
        not (tp["n_top"] == tp["n_top_all"]),
        f"旧线判 {tp['n_top']} == {tp['n_top_all']} → {tp['n_top'] == tp['n_top_all']}；"
        f"缺口 {tp['n_top_all'] - tp['n_top']} 格，其中 {tp['n_ring_void']} 格按定义在"
        f"设计域之外（x>{fx['domain_um'][0]}µm 或 y>{fx['domain_um'][1]}µm）")
    # K1/K2：G4a 的两个绝对期望各自对一处口径敏感。
    e1 = D.expected_capture(top_z, dx, r_b, z_beam)          # σ_z 少 1.2 倍
    d1 = abs(cap["surface_lo"] - e1)
    add("K1 期望式里 dp→r_b（少 1.2 倍）必须把差推到容差 10 倍以上",
        d1 >= SENS, f"错口径期望={e1:.6f}，与实测差={d1:.3e}（TOL={TOL:.0e}，"
                    f"要求 ≥{SENS:.0e}）")
    e2 = D.expected_capture(top_z, dx, dp, 0.0)              # 束面挪到设计面
    d2 = abs(cap["surface_lo"] - e2)
    add("K2 束面 z 由 top_z 改成设计面 0 必须把差推到容差 10 倍以上",
        d2 >= SENS, f"该相位期望={e2:.6f}，与实测量（束面={z_beam*1e6:.3f}µm）差="
                    f"{d2:.3e}；⇒ 顶面节点／设计面这 1.25µm 的相位差是**可分辨**的")
    add("K2b 反向确认：正确口径的两处相位确实命中（差 ≤TOL）",
        abs(cap["surface_lo"] - e_surf) <= TOL and abs(cap["deep_lo"] - e_deep) <= TOL,
        f"顶面差={abs(cap['surface_lo'] - e_surf):.3e}，深度差="
        f"{abs(cap['deep_lo'] - e_deep):.3e}")
    # K3：亏损逐笔可分配——掩膜损失 == 顶面以上空层的解析积分。
    pos = np.asarray(prog.sample_steps(n_steps)[0])[0]
    c = np.asarray(geom.coords())
    q = np.asarray(_cell_integrated_source(pos, c, 1.0, r_b, dp, dx))
    total_grid = float(q.sum() * dx ** 3)
    above = float(np.sum(np.where(~solid, q, 0.0)) * dx ** 3)
    inside = float(np.sum(np.where(solid, q, 0.0)) * dx ** 3)
    e_grid = 0.5 * (1.0 + math.erf((float(zax[-1]) + 0.5 * dx - z_beam) / dp))
    e_above = e_grid - e_surf
    add("K3 三段可分配：网格和／空层／实体内 各自命中解析且求和闭合",
        abs(total_grid - e_grid) <= TOL and abs(above - e_above) <= TOL
        and abs(inside - e_surf) <= TOL and abs(inside + above - total_grid) <= 1e-12,
        f"网格和={total_grid:.6f}（解析={e_grid:.6f}，格顶面 z={float(zax[-1])*1e6:.3f}"
        f"µm 的上表面）＝实体内 {inside:.6f} ＋ 空层 {above:.6f}；闭合残差="
        f"{abs(inside + above - total_grid):.2e}")
    # K4：深度对照的方向性——上移 3r_b 应落到工件外。
    up_lo, up_hi, _ = D.deposited_fraction(geom, prog, n_steps, r_b,
                                           z_shift=+3.0 * r_b)
    add("K4 探针上移 3r_b（束斑离开实体）必须让 ≥0.99 报红",
        not (up_lo >= 0.99),
        f"上移实测=[{up_lo:.6f},{up_hi:.6f}] ⇒ 与下移 3r_b 的 {cap['deep_lo']:.6f} "
        f"相差 {cap['deep_lo'] - up_lo:.3f}：对照确实量的是『是否埋在材料里』")
    # K5/K6：G5 判据本体的两个扰动（复用同一个函数，不留副本）。
    lx = fx["domain_um"][0] * 1e-6
    # K5 的扰动方向：**放大**标称域，直到点阵不再过延伸（3300>3017.5、2750>2507.5）
    # ⇒ 那圈空节点失去合法归属。缩小区只会让圈更宽松、判据照样绿，量不出东西。
    tp5 = D.toplane_readability(solid, xax, yax, top_idx, fx, sub, sub_runs, r_b, dx,
                                ring_shrink=1.1)
    add("K5 标称域放大到点阵不再过延伸 ⇒ 空节点失去归属，判据必须报红",
        tp5["n_stray"] > 0 and tp5["n_ring_void"] == 0,
        f"游荡={tp5['n_stray']}（原口径 {tp['n_stray']}），圈上={tp5['n_ring_void']}"
        f"（原 {tp['n_ring_void']}）；⇒ 『空节点必须在圈上』这半条确实在数东西")
    tp6 = D.toplane_readability(solid, xax, yax, top_idx, fx, sub, sub_runs, r_b, dx,
                                x_center=lx)
    add("K6 足迹中心推到 x=域宽 ⇒ 足迹必须不满布（判据报红）",
        tp6["n_foot_solid"] < tp6["n_foot"],
        f"足迹 {tp6['n_foot']} 格里实体 {tp6['n_foot_solid']}（原口径 "
        f"{tp['n_foot']}/{tp['n_foot_solid']}）")
    add("K7 控件的控件：K5/K6 的扰动确实改变了掩膜计数（非空跑）",
        tp5["n_stray"] != tp["n_stray"] and tp6["n_foot_solid"] != tp["n_foot_solid"],
        f"n_stray {tp['n_stray']}→{tp5['n_stray']}；足迹实体格 "
        f"{tp['n_foot_solid']}→{tp6['n_foot_solid']}")
    for tag, ok, detail in rows:
        print(f"  [{'PASS' if ok else 'FAIL'}] {tag} — {detail}")
    # 标本区：把**旧判据**原样送进同一套标签器印出来。主面归档正文里 `[FAIL]` 计数为 0，
    # 那 0 只有配上"同一条打印式子能印出 FAIL"的活体标本才算数（否则谁也不知道那个词
    # 是不是只会输出 PASS）。这两条按定义必须红，红的不是控件而是判据本身。
    spec = [("X1 旧 G4 线原样判定（份额≥0.99）", cap["surface_lo"] >= 0.99,
             f"实测份额={cap['surface_lo']:.6f} ⇒ 旧线判 False 是**正确输出**"),
            ("X2 旧 G5 线原样判定（顶面全实体）", tp["n_top"] == tp["n_top_all"],
             f"顶面实体 {tp['n_top']}/{tp['n_top_all']} ⇒ 旧线判 False 是**正确输出**")]
    print("标本区（这两行**必须**印成 [FAIL]，用来证明标签器的红路是活的）：")
    for tag, ok, detail in spec:
        print(f"  [{'PASS' if ok else 'FAIL'}] {tag} — {detail}")
    n_red = sum(1 for _, ok, _ in rows if not ok)
    n_spec_red = sum(1 for _, ok, _ in spec if not ok)
    legacy_red = [rows[0][1], rows[1][1]]   # K0a/K0b 的 ok 定义即「旧线报红」
    print(f"CONTROLS = 共 {len(rows)} 条，绿={len(rows)-n_red} 红={n_red}")
    print(f"SPECIMEN_RED = 标本区 {n_spec_red}/{len(spec)} 行印成 [FAIL]"
          f"（主面归档正文的 [FAIL] 计数见本轮两张回执表头，两处一并对）")
    print(f"LEGACY_RED = 旧 G4／旧 G5 两线均如实报红={all(legacy_red)}（红路活体证据 "
          f"{sum(legacy_red)}/2）")


if __name__ == "__main__":
    main()
