#!/usr/bin/env python3
"""#35 全量件的补充实测：3 条 skip 的**身份**（`-rs` 原文，不按位置反推）。

分两路（口径差异照实写进文件头）：
  路 A＝收集期 skip（模块级 importorskip）⇒ `--collect-only -rs` 就能现取，rc＝5「no tests collected」
       本身就是"收集期即跳过、不进 collected 计数"的证据；
  路 B＝测试体内 skip ⇒ 收集期不报，必须**实跑那一条 nodeid**（不跑整文件，避开慢测）。

本脚本自身的写法纪律：正文一律走 python 字面量，**不经 shell 的 `echo "…"`**——
反引号在双引号里会被当命令替换吃掉（本轮已第 6 次栽在同一条上）。
"""
import os
import subprocess
import time

REPO = "/home/shy/桌面/2026年开发-显式冲击有限元-物质点法-sph-fvm+am/AM-Qoder"
OUT = "/tmp/am_t35_skips.txt"
PY = "/tmp/amvenv/bin/python"
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_ENABLE_X64="1", PYTHONPATH="src")

FINGER = ("find src tests -name '*.py' -not -path '*__pycache__*' -print0"
          " | sort -z | xargs -0 md5sum | md5sum | cut -d' ' -f1")


def sh(*args):
    return subprocess.run(args, cwd=REPO, capture_output=True, text=True).stdout.strip()


def run(*pytest_args):
    cmd = [PY, "-m", "pytest", *pytest_args, "-p", "no:cacheprovider"]
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=ENV)
    return ("CMD: CUDA_VISIBLE_DEVICES= JAX_ENABLE_X64=1 PYTHONPATH=src "
            + " ".join([PY, "-m", "pytest", *pytest_args, "-p", "no:cacheprovider"]) + "\n"
            + p.stdout + p.stderr + f"rc={p.returncode}\n")


lines = [
    "# #35 全量件的 skip 身份实测（供 am_t35_full_assemble.py 的 G4 使用）。",
    "# 身份一律取 `-rs` 的 SKIPPED 行**原文**，不按位置反推（行宽随终端变，上一轮栽过一次）。",
    "# 分两路取，口径不同、各自标注：",
    "#   路 A＝收集期 skip（模块级 importorskip）⇒ --collect-only -rs 即可现取；",
    "#        该路 rc=5「no tests collected」正是「这两条不占 collected 计数」的直接证据。",
    "#   路 B＝测试体内 skip ⇒ 收集期不报，必须实跑那一条 nodeid（只跑一条，避开同文件的慢测）。",
    f"# DATE: {time.strftime('%F %T %z')}",
    f"# TREE: head={sh('git', 'rev-parse', '--short', 'HEAD')}"
    f" dirty={len(sh('git', 'status', '--porcelain', '--', 'src', 'tests').splitlines())}",
    f"# FINGER（启动器同口径）: {sh('bash', '-c', FINGER)}",
    "",
    "===== 路 A：收集期 skip =====",
    run("--collect-only", "-q", "-rs", "tests/test_gui_app.py", "tests/test_gui_app2.py"),
    "===== 路 B：测试体内 skip =====",
    run("-q", "-rs", "tests/test_monitoring.py::test_monitoring_gui_construct_and_logic"),
]
# 自检：三条身份必须在、两条命令必须在，否则档案会拿残缺件当真。
body = "\n".join(lines)
assert body.count("SKIPPED [1] tests/") == 3, f"SKIPPED 行应 3 条，实得 {body.count('SKIPPED [1] tests/')}"
assert "--collect-only" in body and "test_monitoring.py::test_monitoring_gui_construct_and_logic" in body
open(OUT, "w", encoding="utf-8").write(body)
print(f"已写 {OUT}（{len(body.splitlines())} 行）")
print("\n".join(x for x in body.splitlines() if x.startswith(("SKIPPED", "rc=", "1 skipped", "no tests"))))
