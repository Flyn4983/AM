"""AMForge 命令行入口 (``amforge ...``)
========================================

子命令
------
``info``      打印版本、依赖导入状态、求解器 / 适配器清点。
``selftest``  跑一段极轻量的端到端前向，验证「安装即可用」（几十秒）。
``pareto``    在 *尺寸偏差 ↔ 残余应力* 两个目标间做 NSGA-II 多目标优化
              （演示，默认极小网格，约 1–3 分钟）。

所有子命令都通过 :func:`main` 暴露给 ``[project.scripts]`` 里的 ``amforge``
可执行文件，也支持 ``python -m amforge``。

设计约定
--------
* 物理前向在 Python 循环里逐个体素/逐代求值（与 inverse 链的 tracer-safe 约定一致），
  不引入额外依赖。
* 仅依赖已安装的 ``amforge``（及其同源子包 ``diffmech`` / ``forgecore``），
  因此 `selftest` / `pareto` 在 ``pip install amforge`` 之后即可运行，无需源码树。
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import amforge as af  # noqa: E402
from amforge import geometry as G  # noqa: E402
from amforge import process as P  # noqa: E402
from amforge import inverse as inv  # noqa: E402
from amforge import calibration as cal  # noqa: E402


# ---------------------------------------------------------------------------
# 轻量端到端自检：被 CLI 与测试共用
# ---------------------------------------------------------------------------
def _tiny_geo():
    """一个 0.4mm 半径小球，120µm 体素 —— 足以触发全链路，又不至于 OOM。"""
    return G.from_sdf_fn(
        lambda x: jnp.linalg.norm(x, axis=-1) - 0.4e-3,
        bounds=[(-0.5e-3, 0.5e-3)] * 3,
        spacing=120e-6,
        name="cli-selftest",
    )


def run_selftest(verbose: bool = True) -> tuple[bool, str]:
    """跑一段极轻量端到端前向，返回 (是否通过, 人类可读报告)。

    覆盖「安装即可用」的最小承诺：
      * 所有内置求解器模块导入成功（import_status 为空）；
      * ``simulate`` 正向跑通，AsBuiltPart 全部叶子有限、产生非零位移；
      * 数字样机 + 服役判定正常流入（verdict 有限）。
    """
    lines: list[str] = []
    t0 = time.time()

    # 0) 导入健康检查
    bad = af.import_status()
    if bad:
        lines.append("FAIL: 内置模块导入失败:")
        for mod, why in bad.items():
            lines.append(f"    - {mod}: {why}")
        return False, "\n".join(lines)
    lines.append("ok: 所有内置求解器模块导入成功")

    # 1) 轻量前向
    geo = _tiny_geo()
    plan = P.heuristic_plan(geo, material="316L", modality="SLM")
    params = dict(n_grid=6, max_layers=3, n_sub_cp=8, tau_activation=1e-3)
    out = inv.simulate(
        geo, plan, material="316L", params=params,
        asbuilt_solver="plastic", constitutive="j2",
    )
    ab = out["asbuilt"]

    finite_ok = all(
        bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree_util.tree_leaves(ab)
    )
    if not finite_ok:
        lines.append("FAIL: AsBuiltPart 含非有限值")
        return False, "\n".join(lines)
    lines.append("ok: AsBuiltPart 全部叶子有限")

    disp = ab.displacement
    if not bool(jnp.max(jnp.abs(disp)) > 0.0):
        lines.append("FAIL: 高保真 asbuilt 未产生位移")
        return False, "\n".join(lines)
    lines.append(f"ok: 产生非零弹塑性位移 (max|u|={float(jnp.max(jnp.abs(disp))):.3e} m)")

    if not bool(jnp.isfinite(out["verdict"].strength_safety_factor)):
        lines.append("FAIL: verdict 非有限")
        return False, "\n".join(lines)
    lines.append(
        f"ok: digitaltwin+verdict 流入正常 "
        f"(安全系数={float(out['verdict'].strength_safety_factor):.3f})"
    )

    dt = time.time() - t0
    lines.append(f"selftest 通过 (耗时 {dt:.1f}s)")
    return True, "\n".join(lines)


# ---------------------------------------------------------------------------
# 子命令实现
# ---------------------------------------------------------------------------
def cmd_info(args) -> int:
    print(f"AMForge {af.__version__}")
    print(f"JAX enable_x64 = {bool(jax.config.jax_enable_x64)}")
    print()

    bad = af.import_status()
    print("[依赖导入状态]")
    if not bad:
        print("  ✓ 所有内置求解器模块就绪")
    else:
        print("  ✗ 以下模块导入失败（对应求解器不可用）:")
        for mod, why in bad.items():
            print(f"    - {mod}: {why}")
    print()

    solvers = af.list_solvers()
    print(f"[求解器] 共 {len(solvers)} 个已注册")
    # 按 stage 分组打印，便于评审快速看清能力覆盖
    by_stage: dict[str, list[str]] = {}
    for name in solvers:
        stage = name.split(".", 1)[0]
        by_stage.setdefault(stage, []).append(name)
    for stage in sorted(by_stage):
        print(f"  {stage:>12}: " + ", ".join(sorted(by_stage[stage])))
    print()

    adapters = af.list_adapters()
    print(f"[适配器] 共 {len(adapters)} 个已注册")
    for name in sorted(adapters):
        print(f"  - {name}")
    return 0 if not bad else 1


def cmd_selftest(args) -> int:
    ok, report = run_selftest(verbose=not args.quiet)
    print(report)
    return 0 if ok else 1


def cmd_pareto(args) -> int:
    geo = _tiny_geo()
    params = dict(
        n_grid=args.n_grid,
        max_layers=args.max_layers,
        n_sub_cp=args.n_sub_cp,
        tau_activation=1e-3,
    )
    t0 = time.time()
    res = inv.pareto_optimize_geometry_process(
        geo, material=args.material, params=params,
        algorithm="nsga2",
        pop_size=args.pop_size, n_gen=args.n_gen, seed=args.seed,
        smoothness=0.02, constraint_weight=20.0,
        asbuilt_solver="plastic", verbose=False,
    )
    dt = time.time() - t0

    gd = np.asarray(res["geom_dev"])
    st = np.asarray(res["stress"])
    feas = np.asarray(res["feasible"])
    print(f"NSGA-II 多目标前沿 (material={args.material}, "
          f"pop={args.pop_size}, gen={args.n_gen}, 耗时 {dt:.1f}s)")
    print(f"  {'#':>3} {'geom_dev(归一)':>14} {'stress(σy)':>12} {'feasible':>9}")
    for i in range(len(gd)):
        print(f"  {i:3d} {gd[i]:14.3e} {st[i]:12.3e} {feas[i]:9.3f}")
    bc = res["best_compromise"]
    print()
    print(f"  utopia = (geom_dev={res['utopia'][0]:.3e}, "
          f"stress={res['utopia'][1]:.3e})")
    print(f"  推荐折中方案: geom_dev={bc['geom_dev']:.3e}, "
          f"stress={bc['stress']:.3e}, feasible={feas[res['best_compromise_index']]:.3f}")
    return 0


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="amforge",
        description="AMForge — 端到端可微分增材制造仿真与数字样机 CLI",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="打印版本 / 依赖导入状态 / 求解器清点")

    st = sub.add_parser("selftest", help="轻量端到端自检（验证安装可用）")
    st.add_argument("--quiet", action="store_true", help="只输出 PASS/FAIL 结论")

    pa = sub.add_parser("pareto", help="NSGA-II 多目标优化演示（尺寸↔残余应力）")
    pa.add_argument("--material", default="316L")
    pa.add_argument("--pop-size", type=int, default=6)
    pa.add_argument("--n-gen", type=int, default=3)
    pa.add_argument("--n-grid", type=int, default=4)
    pa.add_argument("--max-layers", type=int, default=2)
    pa.add_argument("--n-sub-cp", type=int, default=4)
    pa.add_argument("--seed", type=int, default=0)

    sub.add_parser("gui", help="启动 PySide6 + VTK 统一工作台（前处理 + 后处理/数字样机/优化）")

    ca = sub.add_parser("calibrate", help="材料标定：用实测熔池尺寸反演材料物性（写 JSON 报告）")
    ca.add_argument("--input", required=True, help="实验 JSON 路径")
    ca.add_argument("--output", default=None, help="JSON 报告输出路径（默认打印到 stdout）")
    ca.add_argument("--n-iter", type=int, default=200, help="L-BFGS 最大迭代步数")
    ca.add_argument("--quiet", action="store_true", help="只输出 JSON 报告，不打印摘要")
    return p


def cmd_gui(args) -> int:
    """启动 PySide6 + VTK 统一工作台（懒加载重依赖）。"""
    from amforge.gui.app import main as gui_main

    gui_main()
    return 0


def cmd_calibrate(args) -> int:
    """材料标定：读 JSON 实验，反演材料物性，写 JSON 报告。

    JSON 格式::

        {
          "base_material": "316L",
          "fit_keys": ["k_solid", "rho_solid", "cp_solid", "latent_fusion"],
          "experiments": [
            {"label": "P200v1", "laser_power": 200, "scan_speed": 1.0,
             "beam_radius": 5.0e-5, "measured_depth": 1.2e-4,
             "measured_width": 2.1e-4},
            ...
          ]
        }
    """
    import json

    with open(args.input, "r", encoding="utf-8") as fh:
        spec = json.load(fh)

    base_material = spec.get("base_material", "316L")
    fit_keys = tuple(spec.get("fit_keys",
                             ["k_solid", "rho_solid", "cp_solid", "latent_fusion"]))
    experiments = cal.make_experiments_from_dicts(spec.get("experiments", []))

    if not experiments:
        print("错误：experiments 为空")
        return 2

    report = cal.calibrate_material(
        base_material, experiments,
        fit_keys=fit_keys, n_iter=args.n_iter,
        verbose=not args.quiet,
    )

    out = {
        "base_material": report.base_material_name,
        "fit_keys": list(report.fit_keys),
        "fitted_material": {
            k: float(getattr(report.fitted_material, k)) for k in report.fit_keys
        },
        "experiment_ids": list(report.experiment_ids),
        "measured_depth_m": [float(x) for x in report.measured_depth],
        "predicted_depth_before_m": [float(x) for x in report.predicted_depth_before],
        "predicted_depth_after_m": [float(x) for x in report.predicted_depth_after],
        "measured_width_m": [float(x) for x in report.measured_width],
        "predicted_width_before_m": [float(x) for x in report.predicted_width_before],
        "predicted_width_after_m": [float(x) for x in report.predicted_width_after],
        "rmse_depth_m": float(report.rmse_depth),
        "rmse_width_m": float(report.rmse_width),
        "r2_depth": float(report.r2_depth),
        "r2_width": float(report.r2_width),
        "n_iter": report.n_iter,
        "converged": report.converged,
        "cost": float(report.cost),
    }
    text = json.dumps(out, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"\n标定报告已写入: {args.output}")
    else:
        print("\n" + text)
    return 0


def main(argv: list[str] | None = None) -> int:
    warnings.filterwarnings("ignore")
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "info":
        return cmd_info(args)
    if args.command == "selftest":
        return cmd_selftest(args)
    if args.command == "pareto":
        return cmd_pareto(args)
    if args.command == "gui":
        return cmd_gui(args)
    if args.command == "calibrate":
        return cmd_calibrate(args)
    parser.error(f"未知子命令: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
