#!/usr/bin/env python3
"""表头生成器 am_s27_hdr.py 的**正对照**：每条扰动都必须让它拒绝，否则那条闸门是空转。

扰动改的是**生成器的源码副本**（sed 级），跑的是同一件（同一条 `python <file>`），
所以被验的就是落地那把仪器本身，不是另写的平行实现。判读＝rc≠0 且 stderr 里出现指定的闸门句。

为什么需要这一件：本片的生成器由上一片克隆，闸门集合没变，但**锚句换了**（LESSONLET 从
"占位符只替换"改成本轮日志里的"极性翻转不是扰动"）⇒ 换锚句这种改动最容易"取空了还不知道"。
所以 C1–C8 复验克隆过来的八条闸门，C9–C11 专门验本轮动过的那一行（取空／歧义／锚点本身不在源码里）。
一条"从没红过的闸门"和"坏闸门"读数相同 ⇒ 必须逐条造红。

三条**仪器自己的**坑（前两条实测到、第三条本轮新增，都不是推想）：
① 判据不能只看 rc≠0。首版把扰动副本写在 /tmp，八条**全红**——但红的原因是生成器用自身位置反推
   REPO，副本在 /tmp ⇒ git 在 `/` 上跑而失败＝环境错，不是闸门错。⇒ 副本一律落在**同一目录**，
   并加一条**未扰动基线必须 rc=0**（C0），否则所有红都不作数。
② 歧义扰动不能靠"猜一个会命中多条的子串"。首版用子串「回执」模拟出处不唯一，实测当轮日志只命中
   1 条子条 ⇒ 副本 rc=0＝这条正对照本身空转（被本件的"必须命中指定闸门句"判据抓出）。
   改法＝用**空模式**，按构造对每条子条都命中，歧义是必然的而不是碰上的。
③ **期望闸门句里不许写死节号**：上一件的 C5 期望句是「在 §26.28 里匹配到 0 条子条」，本片节号现取
   已是 §26.29 ⇒ 那句期望永远命中不了，红是红了、红的却不是指定闸门（本件的判据恰好会把它判成
   "红了但没命中"）。⇒ 本件的节号期望由开发日志现取最后一个 `### 26.NN` 拼出来，不手打。
"""
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
SRC = os.path.join(HERE, "am_s27_hdr.py")
text = open(SRC, encoding="utf-8").read()

# 期望句里的两个数**都从被验源码／开发日志现取**（写死就随片腐烂，见 docstring 坑③）
CHAINN = re.search(r"^CHAINN = (\d+)$", text, re.M).group(1)
DEV = open(os.path.join(REPO, "docs/开发日志.md"), encoding="utf-8").read()
SECN = re.findall(r"^### 26\.(\d+)[^\n]*$", DEV, re.M)[-1]

LESSON_ANCHOR = r'^LESSONLET, _LSEG = bullet_for\(r"极性翻转不是扰动", "对照必须是输入扰动"\)$'

CASES = [
    ("C1 回执链少一件（把生成器从链口径里删掉）",
     [(r"^CHAIN_RE = re\.compile.*$",
       'CHAIN_RE = re.compile(r"^am_s7_verify(\\d+)\\.(py|log)$|^_raw_verify(\\d+)\\.(out|err)$")')],
     "上一片回执链在 HEAD 里只 %d/%s" % (int(CHAINN) - 1, CHAINN)),
    ("C2 链计数口径退回 int/str 混比（归一化失效）",
     [(r"^    n = str\(n\).*$", "    pass")],
     "上一片回执链在 HEAD 里只 0/%s" % CHAINN),
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
     "在 §26.%s 里匹配到 0 条子条" % SECN),
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
    # C9–C10＝本轮**唯一改动的那一行**（LESSONLET 锚句）。换锚句最容易"取空了还不知道"，
    # 所以它必须既有"查无"也有"歧义"两条红路，而不只是"当前能跑通"。
    ("C9 本轮新锚句在日志里查无（取空）",
     [(LESSON_ANCHOR,
       'LESSONLET, _LSEG = bullet_for(r"这句本轮日志里没有", "对照必须是输入扰动")')],
     "在 §26.%s 里匹配到 0 条子条" % SECN),
    ("C10 本轮新锚句歧义（空模式：每条子条必命中）",
     [(LESSON_ANCHOR, 'LESSONLET, _LSEG = bullet_for("", "对照必须是输入扰动")')],
     "引用出处不唯一"),
]

ENV = dict(os.environ)
ENV["CUDA_VISIBLE_DEVICES"] = ""


def perturb(name, subs, src=None):
    """把一条扰动落到源码上；**锚点必须在源码里唯一命中**，否则这条扰动不可信（＝C11 验的就是这句）。"""
    bad = src if src is not None else text
    for pat, rep in subs:
        hits = re.findall(pat, bad, re.M)
        assert len(hits) == 1, "%s：扰动锚点在源码里命中 %d 处 ⇒ 扰动本身不可信" % (name, len(hits))
        bad = re.sub(pat, lambda m: rep, bad, count=1, flags=re.M)
    assert bad != (src if src is not None else text), name + "：扰动没改变源码 ⇒ 空转"
    return bad


def run_one(bad):
    """在同一目录落一份扰动副本、跑它、拿 rc＋合并输出，然后删掉副本。"""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8",
                                     dir=HERE, prefix="zctl_s27_hdr_") as fh:
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
assert not [f for f in os.listdir(HERE) if f.startswith("zctl_s27_hdr_")], "扰动副本没被清掉"

rows, allred = [], True
for name, subs, expect in CASES:
    rc, msg = run_one(perturb(name, subs))
    ok = rc != 0 and expect in msg
    allred = allred and ok
    rows.append((name, rc, ok, expect))

# C11＝对**本件自己**的闸门：锚点不存在时 perturb 必须拒绝。不验这条，下一轮克隆后"锚点已移位"的
# 扰动会静默地没改动任何源码（perturb 之外没人管），红就变成"跑了一遍真件"＝假红。
try:
    perturb("C11", [(r'^LESSONLET = bullet_for\("这条锚点在本件源码里不存在"\)$', "pass")])
    rows.append(("C11 锚点普查（本件自己的闸门）", 0, False, "扰动锚点在源码里命中 0 处"))
    allred = False
except AssertionError as e:
    hit = "扰动锚点在源码里命中 0 处" in str(e)
    rows.append(("C11 锚点普查（本件自己的闸门）", 3 if hit else 0, hit, "扰动锚点在源码里命中 0 处"))
    allred = allred and hit
    assert not [f for f in os.listdir(HERE) if f.startswith("zctl_s27_hdr_")], "C11 没落副本，但目录里有残留"

print("基线（未扰动副本）rc=%s 且进预检分支＝True（所有红的对照面）" % BASE_RC)
print("期望句的两个现取参数：回执链件数 CHAINN＝%s（从被验源码）｜日志节号 §26.%s（从开发日志最后一个标题）"
      % (CHAINN, SECN))

print("扰动条数＝%d｜每条判据＝rc≠0 且 stderr 含指定闸门句" % len(rows))
for name, rc, ok, expect in rows:
    print(("  " if ok else "× ") + name + "｜rc=" + str(rc) + "｜期望句「" + expect + "」"
          + ("命中" if ok else "**未命中**"))
print("CONTROL_VERDICT =", "ALL CONTROLS CAN GO RED" if allred else "PROBLEM（有闸门空转）")
sys.exit(0 if allred else 3)
