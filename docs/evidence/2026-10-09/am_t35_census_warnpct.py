#!/usr/bin/env python3
"""#35 出域普查的**仪器**：`src` 里所有"用户可见的运行时文本"（`warnings.warn` 的实参 ＋ `raise` 的消息）
中，**字符串字面量内嵌 `数字%`** 的位点有多少？

为什么用 AST 而不是 grep：grep 会把注释、docstring、日志格式串、`f"{x:.1%}"` 的**运行时**百分比
一并算进来（本轮已因忘 `-F` 造出 6 处假阳）。AST 只认**调用实参里的字符串常量**，且天然区分
「字面量里的死数」与「插值出来的活数」——后者正是我们要允许存在的形式。

同一件在**两个树状态**上各跑一次，构成正对照：
  工作区（换数后）⇒ 期望 0；
  `git show HEAD:src/amforge/thermal_enthalpy.py`（换数前）⇒ 期望 ≥1（否则该普查式没有分辨力，0 无意义）。
"""
import ast
import os
import re
import subprocess
import sys

REPO = os.environ.get("AM_REPO", "/home/shy/桌面/2026年开发-显式冲击有限元-物质点法-sph-fvm+am/AM-Qoder")
PCT = re.compile(r"\d+(?:\.\d+)?\s*%")


def literal_percents(node):
    """返回该调用实参里**字面量**内嵌 `数字%` 的个数（f-string 的静态片段也算字面量）。"""
    out = 0
    for arg in node.args:
        for n in ast.walk(arg):
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                out += len(PCT.findall(n.value))
    return out


def is_user_visible_call(n):
    if not isinstance(n, ast.Call):
        return False
    f = n.func
    if isinstance(f, ast.Attribute) and f.attr == "warn":
        return True
    if isinstance(f, ast.Name) and f.id == "warn":
        return True
    if isinstance(f, ast.Name) and f.id.endswith("Error"):
        return True
    return False


def census_src_dir(root):
    total_calls, hosts, files = 0, [], 0
    for dirpath, _dirs, names in os.walk(root):
        if "__pycache__" in dirpath:
            continue
        for nm in sorted(names):
            if not nm.endswith(".py"):
                continue
            p = os.path.join(dirpath, nm)
            tree = ast.parse(open(p, encoding="utf-8").read(), filename=p)
            hit_in_file = 0
            for n in ast.walk(tree):
                if is_user_visible_call(n):
                    total_calls += 1
                    k = literal_percents(n)
                    if k:
                        hit_in_file += k
                        hosts.append((os.path.relpath(p, REPO), getattr(n, "lineno", 0), k))
            files += 1
    return total_calls, hosts, files


print(f"CMD: {sys.executable} {os.path.abspath(sys.argv[0])}   # 工作区（换数后）")
print(f"TREE: head={subprocess.run(['git', '-C', REPO, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()}"
      f" dirty={len(subprocess.run(['git', '-C', REPO, 'status', '--porcelain', '--', 'src', 'tests'], capture_output=True, text=True).stdout.splitlines())}")
tc, hosts, nf = census_src_dir(os.path.join(REPO, "src"))
print(f"  扫描 src 的 .py 文件数＝{nf}")
print(f"  用户可见调用位点（warn/warn 方法/raise *Error）＝{tc}")
print(f"  其中「实参字面量内嵌 数字%」＝{sum(k for _, _, k in hosts)} 处，宿主 {len(hosts)} 行")
for h in hosts:
    print(f"    {h}")

# ---- 正对照：把 HEAD 版单独放进临时目录再跑同一函数（同一仪器、两个树状态） ----
REF = sys.argv[1] if len(sys.argv) > 1 else "HEAD"
tmp = f"/tmp/am_t35_census_{REF[:7]}"
print(f"\n  正对照的 ref＝{REF}（**不要用 HEAD 记死**：本片入库后 HEAD 就变成换数后的树，\n        届时必须显式传 c34a2e1，否则正对照会同义反复地读到换数后的文本）")
os.makedirs(tmp, exist_ok=True)
for rel in subprocess.run(["git", "-C", REPO, "diff", "--name-only", "HEAD", "--", "src"],
                          capture_output=True, text=True).stdout.split():
    txt = subprocess.run(["git", "-C", REPO, "show", f"HEAD:{rel}"], capture_output=True, text=True).stdout
    dst = os.path.join(tmp, rel.split("/", 1)[1])
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, "w", encoding="utf-8").write(txt)
    print(f"  正对照取材＝git show HEAD:{rel}（{len(txt.splitlines())} 行 → {dst}）")
tc_h, hosts_h, nf_h = census_src_dir(tmp)
print(f"  【正对照】HEAD 版：位点＝{tc_h}，字面量内嵌 数字%＝{sum(k for _, _, k in hosts_h)} 处，宿主＝{hosts_h}")
assert sum(k for _, _, k in hosts_h) >= 1, "正对照失败 ⇒ 该普查式恒 0，无分辨力，下面的 0 不能报"
print(f"  判读：工作区＝{sum(k for _, _, k in hosts)}／HEAD＝{sum(k for _, _, k in hosts_h)}"
      f" ⇒ {'换数确实把用户可见的死数清零，且仪器有分辨力' if sum(k for _, _, k in hosts) == 0 else '工作区仍有死数！'}")
