#!/usr/bin/env python3
"""表头生成器 am_s26_hdr.py 的**正对照**：每条扰动都必须让它拒绝，否则那条闸门是空转。

扰动改的是**生成器的源码副本**（sed 级），跑的是同一件（同一条 `python <file>`），
所以被验的就是落地那把仪器本身，不是另写的平行实现。判读＝rc≠0 且 stderr 里出现指定的闸门句。

为什么需要这一件：本片新加/改动的闸门有四条（回执链计数、尺子全树 glob、子条字母唯一性、
f-string 成因出处），一条"从没红过的闸门"和"坏闸门"读数相同 ⇒ 必须逐条造红。

两条**仪器自己的**坑（都是本轮实测到、不是推想）：
① 判据不能只看 rc≠0。首版把扰动副本写在 /tmp，八条**全红**——但红的原因是生成器用自身位置反推
   REPO，副本在 /tmp ⇒ git 在 `/` 上跑而失败＝环境错，不是闸门错。⇒ 副本一律落在**同一目录**，
   并加一条**未扰动基线必须 rc=0**（C0），否则所有红都不作数。
② 歧义扰动不能靠"猜一个会命中多条的子串"。首版用子串「回执」模拟出处不唯一，实测在 §26.28 只命中
   1 条子条 ⇒ 副本 rc=0＝这条正对照本身空转（被本件的"必须命中指定闸门句"判据抓出）。
   改法＝用**空模式**，按构造对每条子条都命中，歧义是必然的而不是碰上的。
"""
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
SRC = os.path.join(HERE, "am_s26_hdr.py")
text = open(SRC, encoding="utf-8").read()

CASES = [
    ("C1 回执链少一件（把生成器从链口径里删掉）",
     [(r"^CHAIN_RE = re\.compile.*$",
       'CHAIN_RE = re.compile(r"^am_s7_verify(\\d+)\\.(py|log)$|^_raw_verify(\\d+)\\.(out|err)$")')],
     "上一片回执链在 HEAD 里只 4/5"),
    ("C2 链计数口径退回 int/str 混比（归一化失效）",
     [(r"^    n = str\(n\).*$", "    pass")],
     "上一片回执链在 HEAD 里只 0/5"),
    ("C3 尺子 glob 指到不存在的目录（本件没被自己数到）",
     [(r'glob\.glob\(os\.path\.join\(REPO, "docs/evidence/\*/am_s7_verify\*\.py"\)\)',
       'glob.glob(os.path.join(REPO, "docs/evidence/NO_SUCH_DIR/am_s7_verify*.py"))')],
     "本件的尺子副本没被自己的 glob 数到"),
    ("C4 子条引用不唯一（空模式：每条子条必命中）",
     [(r'^SECLET, _SECSEG = bullet_for\(re\.escape\(PREVLOG\), "上一片回执名"\)$',
       'SECLET, _SECSEG = bullet_for("", "上一片回执名")')],
     "引用出处不唯一"),
    ("C5 子条引用查无（正则口径又变了）",
     [(r'^SECLET, _SECSEG = bullet_for\(re\.escape\(PREVLOG\), "上一片回执名"\)$',
       'SECLET, _SECSEG = bullet_for("am_s7_verify99.log", "上一片回执名")')],
     "在 §26.28 里匹配到 0 条子条"),
    ("C6 正文判读行被改坏（回执不该被当作全过归档）",
     [(r'^assert "LANDING_VERDICT = ALL CHECKS PASS" in body',
       'assert "LANDING_VERDICT = PROBLEM" in body')],
     "本件不该被当作回执归档"),
    ("C7 复跑 stdout 与归档正文不同（仪器不可复现）",
     [(r"^REPRO = rp\.stdout == body$", "REPRO = rp.stdout == body + 'x'")],
     "复跑 stdout 与归档正文不同"),
    ("C8 入库清单两个出处互证失效（body 侧 +1）",
     [(r"^assert \(NA_BODY, NM_BODY\) == \(len\(added\), len\(mods\)\), \($",
       "assert (NA_BODY, NM_BODY) == (len(added) + 1, len(mods)), (")],
     "核验件印出的入库数"),
]

ENV = dict(os.environ)
ENV["CUDA_VISIBLE_DEVICES"] = ""


def run_one(bad):
    """在同一目录落一份扰动副本、跑它、拿 rc＋合并输出，然后删掉副本。"""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8",
                                     dir=HERE, prefix="zctl_s26_hdr_") as fh:
        fh.write(bad)
        tmp = fh.name
    try:
        r = subprocess.run([sys.executable, tmp], cwd=REPO, env=ENV,
                           capture_output=True, text=True)
        return r.returncode, (r.stderr or "") + (r.stdout or "")
    finally:
        os.unlink(tmp)


# 基线＝逐字节相同的副本必须**通过**（预检模式不写盘）。没有这一步，上面所有红都可能是环境错。
BASE_RC, BASE_MSG = run_one(text)
assert BASE_RC == 0, "未扰动副本就没通过（rc=%s）⇒ 本件的正对照全部无效，先修仪器" % BASE_RC
assert "预检模式" in BASE_MSG, "未扰动副本没进预检分支 ⇒ 副本跑的不是同一条路径"
assert not [f for f in os.listdir(HERE) if f.startswith("zctl_s26_hdr_")], "扰动副本没被清掉"

rows, allred = [], True
for name, subs, expect in CASES:
    bad = text
    for pat, rep in subs:
        hits = re.findall(pat, bad, re.M)
        assert len(hits) == 1, "%s：扰动锚点在源码里命中 %d 处 ⇒ 扰动本身不可信" % (name, len(hits))
        bad = re.sub(pat, lambda m: rep, bad, count=1, flags=re.M)
    assert bad != text, name + "：扰动没改变源码 ⇒ 空转"
    rc, msg = run_one(bad)
    ok = rc != 0 and expect in msg
    allred = allred and ok
    rows.append((name, rc, ok, expect))

print("基线（未扰动副本）rc=%s 且进预检分支＝True（所有红的对照面）" % BASE_RC)

print("扰动条数＝%d｜每条判据＝rc≠0 且 stderr 含指定闸门句" % len(rows))
for name, rc, ok, expect in rows:
    print(("  " if ok else "× ") + name + "｜rc=" + str(rc) + "｜期望句「" + expect + "」"
          + ("命中" if ok else "**未命中**"))
print("CONTROL_VERDICT =", "ALL CONTROLS CAN GO RED" if allred else "PROBLEM（有闸门空转）")
sys.exit(0 if allred else 3)
