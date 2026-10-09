#!/usr/bin/env python3
"""修 #35 档案腿自己造成的文本损坏：f-string 把字面量 `[0-9a-f]{32}` 里的 `{32}` 当替换字段吃掉。

两步：① 生成器 `/tmp/am_t35_docs.py` 的**正文串**里把 `` `[0-9a-f]{32}` `` 改成 `` `[0-9a-f]{{32}}` ``
（代码里的正则不动——它们在 raw 串里，不是 f-串）；② 已落盘的三份档案把 `[0-9a-f]32` 修回
`[0-9a-f]{32}`，每个文件都印**修复前命中数**（0 也要印，否则"没坏"是空话）。
"""
import os
import subprocess
import sys

REPO = "/home/shy/桌面/2026年开发-显式冲击有限元-物质点法-sph-fvm+am/AM-Qoder"
GEN = "/tmp/am_t35_docs.py"
FILES = ["docs/开发日志.md", ".workbuddy/memory/2026-10-09.md", ".workbuddy/memory/MEMORY.md"]
BAD = "`[0-9a-f]32`"
GOOD = "`[0-9a-f]{32}`"

dry = "--write" not in sys.argv
g = open(GEN, encoding="utf-8").read()
# 只替换反引号包起来的那三处正文；代码里的 `([0-9a-f]{32})` 前后不是反引号
n_gen = g.count("`[0-9a-f]{32}`")
g2 = g.replace("`[0-9a-f]{32}`", "`[0-9a-f]{{32}}`")
print(f"生成器：待转义的正文字面量 {n_gen} 处（raw 串里的正则不受影响）")
if not dry:
    open(GEN, "w", encoding="utf-8").write(g2)
    chk = subprocess.run([sys.executable, "-W", "error::SyntaxWarning", "-m", "py_compile", GEN],
                         capture_output=True, text=True)
    print(f"生成器重编译 rc={chk.returncode}{chk.stderr[:200]}")
    assert chk.returncode == 0
    assert open(GEN, encoding="utf-8").read().count("`[0-9a-f]{{32}}`") == n_gen

print("\n=== 已落盘档案的损坏计数（修复前必须 >0，否则这次修复本身是空转）===")
rc = 0
for rel in FILES:
    p = os.path.join(REPO, rel)
    s = open(p, encoding="utf-8").read()
    c_bad, c_good = s.count(BAD), s.count(GOOD)
    print(f"  {rel:38s} 损坏={c_bad} 正确={c_good}")
    if c_bad == 0:
        print("     ⇒ 该件没有损坏（不修）")
        continue
    if not dry:
        open(p, "w", encoding="utf-8").write(s.replace(BAD, GOOD))
        back = open(p, encoding="utf-8").read()
        assert back.count(BAD) == 0 and back.count(GOOD) == c_good + c_bad, rel
        print(f"     修复后 损坏=0 正确={back.count(GOOD)}（＝{c_good}+{c_bad}）")
print("\n（预检模式，未写盘。加 --write。）" if dry else "写盘完成。")
sys.exit(rc)
