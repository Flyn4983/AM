"""#36 的 server.py 半边：静态名称解析取证（不 import diffmech.server，它因 #40 不可导入）。

跑前写死的判据：
  N0  仪器能炸：向属性表里塞一个不存在的名字，检查器必须报缺失（正对照）。
  N1  server.py 顶层确实把名字 am_api 绑进了模块命名空间（AST 取证，不是 grep 关键字）。
  N2  文件里每一处 am_api.<attr> 在**真模块** diffmech.methods.am.am_api 上都存在。
  N3  改前那种「值导入」写法拿到的其实是**另一个模块对象**的属性：
      diffmech.methods.am.am_api 与 src.diffmech.methods.am.am_api 必须是两个不同的
      sys.modules 条目、且 _PROCESS_BOUNDS 两个对象 not the same。
  N4  全树 src. 前缀导入＝0 行（与 C0 同一正则、同一范围，此处仅作本件的自证）。
本件只读，不改任何 src/tests 文件。
"""
from __future__ import annotations

import ast
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import diffmech.methods.am.am_api as real_api

REPO = Path(__file__).resolve().parents[3]
SERVER = REPO / "src" / "diffmech" / "server.py"
# 注意：GNU grep -E **不认** (?:…) —— 首轮用 `^\s*(?:from|import)\s+src\.` 时 grep 只报
# 「警告：表达式以 ? 开头」并返回 0 命中，于是 N4 的 HEAD 参照列（必须 >0）把它判成 FAIL。
# 这正是"尺子词汇表缺口冒充树干净"的实例，故此处改用 POSIX 捕获组 + 字符类。
CENSUS_RE = r"^[[:space:]]*(from|import)[[:space:]]+src\."

failures: list[str] = []


def sh(cmd: list[str]) -> str:
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if r.stderr.strip():
        # 尺子自己报警必须进日志：0 命中配上 stderr，读作"仪器失效"而不是"树干净"
        print(f"  [stderr@{' '.join(cmd[:2])}] {r.stderr.strip()[:180]}")
    return r.stdout


def sh_in(cmd: list[str], text: str) -> str:
    r = subprocess.run(cmd, input=text, capture_output=True, text=True)
    if r.stderr.strip():
        print(f"  [stderr@{cmd[0]}] {r.stderr.strip()[:180]}")
    return r.stdout


src = SERVER.read_text(encoding="utf-8")
tree = ast.parse(src, filename=str(SERVER))

# --- N1: 顶层绑定名 am_api -------------------------------------------------
top_names: set[str] = set()
for node in ast.walk(tree):
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        if getattr(node, "col_offset", 0) != 0:
            continue  # 只认顶层（0 缩进）的 import；函数体内的正是 #36 的病根
        for a in node.names:
            top_names.add(a.asname or a.name.split(".")[0])
n1 = "am_api" in top_names
if not n1:
    failures.append("N1")

# --- N2: 每个 am_api.<attr> 在真模块上存在 --------------------------------
attrs = sorted(
    {
        n.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute)
        and isinstance(n.value, ast.Name)
        and n.value.id == "am_api"
    }
)
missing = [a for a in attrs if not hasattr(real_api, a)]
n2 = not missing and len(attrs) > 0
if not n2:
    failures.append("N2")

# --- N0: 正对照（检查器必须能炸）------------------------------------------
bogus = [a for a in [*attrs, "_NO_SUCH_ATTRIBUTE_T36"] if not hasattr(real_api, a)]
n0 = bogus == ["_NO_SUCH_ATTRIBUTE_T36"]
if not n0:
    failures.append("N0")

# --- N3: 两种拼写是两个模块对象 -------------------------------------------
# 脚本的 sys.path[0] 是本文件所在目录，REPO 根并不在其中 ⇒ 「src. 前缀能否命名空间导入」
# 本身就是被测量的事实，必须显式把 REPO 加进去（这正是 #36 在生产里成立的环境条件）。
sys.path.insert(0, str(REPO))
try:
    src_api = __import__("src.diffmech.methods.am.am_api", fromlist=["x"])
    src_import_err = ""
except Exception as exc:  # noqa: BLE001 —— 导入失败本身就是 N3 的读数
    src_api = None
    src_import_err = f"{type(exc).__name__}: {exc}"
same_module = src_api is real_api
same_dict = src_api is not None and src_api.__dict__ is real_api.__dict__
bounds_a = real_api._PROCESS_BOUNDS
bounds_b = None if src_api is None else src_api._PROCESS_BOUNDS
n3 = src_api is not None and (not same_module) and (bounds_a is not bounds_b)
if not n3:
    failures.append("N3")

# --- N4: 全树普查 ----------------------------------------------------------
census_now = [
    ln
    for ln in sh(["grep", "-rEn", CENSUS_RE, "src", "tests", "--include=*.py"]).splitlines()
]
head_blob = sh(["git", "show", "HEAD:src/diffmech/server.py"])
hits_head = [
    ln
    for ln in sh_in(["grep", "-En", CENSUS_RE], head_blob).splitlines()
]
n4 = len(census_now) == 0 and len(hits_head) > 0
if not n4:
    failures.append("N4")

print(f"REPO={REPO}")
print(f"SERVER-MD5={hashlib.md5(src.encode()).hexdigest()[:12]}")
print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r} (钉 CPU)")
print(f"[N0] 正对照 bogus-missing==[{bogus}] => {'PASS' if n0 else 'FAIL'}")
print(f"[N1] 顶层绑定含 am_api => {n1}（顶层 import 名 {len(top_names)} 个）")
print(f"[N2] am_api.<attr> 共 {len(attrs)} 个，真模块上缺失 {missing} => {'PASS' if n2 else 'FAIL'}")
print(f"       attrs={attrs}")
print(f"[N3] 双拼写导入 err={src_import_err!r} 同一模块对象={same_module} 同一 __dict__={same_dict} "
      f"_PROCESS_BOUNDS 同一对象={bounds_a is bounds_b} 内容相等={bounds_a == bounds_b} "
      f"=> {'PASS' if n3 else 'FAIL'}")
print("       说明：改前的值导入拿到的是**另一份模块副本**里的 dict（内容同、身份异）")
print(f"[N4] 现树 src. 前缀导入={len(census_now)} 行；HEAD:server.py 同正则命中={len(hits_head)} 行 "
      f"=> {'PASS' if n4 else 'FAIL'}")
for ln in census_now[:5]:
    print(f"       残留：{ln}")
for ln in hits_head[:5]:
    print(f"       HEAD：{ln}")
print()
print("[处置] " + ("N0∧N1∧N2∧N3∧N4 => server.py 半边改名后仍可解析（静态可证，运行时等 #40）"
                if not failures else f"FAILS={failures}"))
sys.exit(1 if failures else 0)
