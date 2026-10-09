#!/usr/bin/env python3
"""#18 代码腿＋分片 24 回执腿的五条归档腿（开发日志 (u)／计划文档 §V／总览／当日／MEMORY）.

口径继承 am_s24_docs.py，针对本轮的四件事各加一道闸门：
① 全量记分**不许手抄**：分数、两红的断言行、起/止指纹、rc 全部从归档的 `_raw_s18_full.log` 正则现取，
   且两红的 `^E\\s+AssertionError: ` 行必须与基线件 `am_t35_full_regression.log` **逐行相等**；
② 「不得新增 skip/xfail」除计数之外另有**静态普查**（HEAD vs 现树的标记行数）；
③ 位点普查一律先剥注释再数（本轮教训：注释里也写着被数的表达式 ⇒ 把成对的 2 处读成 3 处），
   并把这条坏口径本身登记成一条闸门；
④ 粘连普查（渲染结构）期望 0 处，且 mid 插入仪器带正对照（漏掉换行时同一处必须非换行）。
每个数都现取；模板里出现的字段若没被算出来 ⇒ 缺字段直接判红；算了却没进任何模板 ⇒ 判红（口径漂了）。
"""
import ast
import json
import math
import os
import re
import subprocess
import sys

REPO = "/home/shy/桌面/2026年开发-显式冲击有限元-物质点法-sph-fvm+am/AM-Qoder"
D = os.path.join(REPO, "docs/evidence/2026-10-09")
REC24 = os.path.join(D, "am_s7_verify24.log")
RAW24 = os.path.join(D, "_raw_verify24.out")
RULE24 = os.path.join(D, "am_s7_verify24.py")
HDR24 = os.path.join(D, "am_s24_hdr.py")
RECEIPT = os.path.join(D, "_raw_s18_receipt.out")
RECEIPT_PY = os.path.join(D, "am_s18_wiring_receipt.py")
PROBE_PRE = os.path.join(D, "_raw_s18_probe.out")
PROBE_POST = os.path.join(D, "_raw_s18_probe_post.out")
PROBE_DIFF = os.path.join(D, "_raw_s18_probe_diff.txt")
FP_POST = os.path.join(D, "_raw_s18_post_fingerprint.json")
SIDE = os.path.join(D, "_raw_s18_side.log")
SIDE_SH = os.path.join(D, "am_s18_side_run.sh")
TGT_UNS = os.path.join(D, "_raw_s18_targeted.out")
FULLRAW = os.path.join(D, "_raw_s18_full.log")
LAUNCHER = os.path.join(D, "am_s18_full_run.sh")
BASELINE = os.path.join(D, "am_t35_full_regression.log")
SP = os.path.join(REPO, "src/amforge/scan_program.py")
TS = os.path.join(REPO, "tests/test_scan_program.py")
TE = os.path.join(REPO, "src/amforge/thermal_enthalpy.py")
DEV = os.path.join(REPO, "docs/开发日志.md")
PL = os.path.join(REPO, "docs/项目评估与下一步计划_2026-10-06.md")
OV = os.path.join(REPO, "docs/项目开发总览.md")
DAY = os.path.join(REPO, ".workbuddy/memory/2026-10-09.md")
MEM = os.path.join(REPO, ".workbuddy/memory/MEMORY.md")
FIVE = ["docs/开发日志.md", "docs/项目开发总览.md", "docs/项目评估与下一步计划_2026-10-06.md",
        ".workbuddy/memory/2026-10-09.md", ".workbuddy/memory/MEMORY.md"]
WRITE = "--write" in sys.argv
FALSE_HASH = "3b4aabb"          # 台账 #14 里那个写错的短哈希（缺陷标本，不是我的读数）
SCORE_RE = r"(\d+) failed, (\d+) passed(?:, (\d+) skipped)? in ([\d.]+)s"


def sh(*a):
    return subprocess.run(a, cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def md5(p):
    return subprocess.run(["md5sum", p], capture_output=True, text=True).stdout.split()[0]


def txt(p):
    return open(p, encoding="utf-8").read()


def code_of(s):
    """只留**代码行**：注释行里也写着被数的表达式，整份文本匹配会把位点数读多。"""
    return "\n".join(l for l in s.splitlines() if not l.strip().startswith("#"))


# 短哈希**一律由命令现取**（本轮 +1 的教训就是台账里手敲的那个假 `3b4aabb`）；后面两处判读都要用它
TRUE_HASH = sh("git", "rev-parse", "--short", "HEAD")
TRUE_FULL = sh("git", "rev-parse", "HEAD")


# ================================================================ 分片 24 回执
raw24_b = open(REC24, "rb").read()
boff = int(re.search(rb"BODY_OFFSET=(\d+)\n", raw24_b).group(1))
h24 = raw24_b[:boff].decode("utf-8")
b24 = raw24_b[boff:].decode("utf-8")
assert raw24_b[boff:] == open(RAW24, "rb").read(), "回执正文区与归档的原始 stdout 不逐字节相同"
CMP = [subprocess.run(["cmp", "-i", "%d:%d" % (a, bb), REC24, RAW24],
                      capture_output=True).returncode for a, bb in ((boff, 0), (boff + 1, 0), (boff, 1))]
assert CMP == [0, 1, 1], "cmp 三联不齐（同偏移须 0、左右各 +1 须 1）：%s" % CMP
assert "@@" not in h24 and "＠" not in h24, "回执表头有占位符残留"


def H24(pat, grp=1):
    m = re.search(pat, h24)
    assert m, "分片 24 回执表头里取不到 " + pat
    return m.group(grp)


def B24(pat, grp=1):
    m = re.search(pat, b24)
    assert m, "分片 24 回执正文里取不到 " + pat
    return m.group(grp)


# 顶格字段数**不写死**：从生成器源码取 FIELDS 声明长度（写死＝把两个文件的耦合藏进一个常数）
NFIELD_DECL = None
for nd in ast.parse(txt(HDR24)).body:
    if isinstance(nd, ast.Assign) and getattr(nd.targets[0], "id", "") == "FIELDS":
        NFIELD_DECL = len(nd.value.elts)
assert NFIELD_DECL, "生成器 am_s24_hdr.py 里取不到 FIELDS 声明"
NDEF_LOG = sum(1 for l in h24.splitlines()
               if re.match(r"^\S.*?\s*:", l) and not l.startswith("BODY_OFFSET="))
assert NDEF_LOG == NFIELD_DECL, "回执表头顶格字段 %d ≠ 生成器 FIELDS 声明 %d" % (NDEF_LOG, NFIELD_DECL)
VERDICT = B24(r"(LANDING_VERDICT = .*)").strip()
assert VERDICT == "LANDING_VERDICT = ALL CHECKS PASS", "分片 24 判读行不是全过 ⇒ 不能当落地证明归档"
assert TRUE_FULL in b24, "分片 24 回执正文的 ① 里没有当前 HEAD 全哈希 %s ⇒ 那份回执测的不是这棵树" % TRUE_FULL
NB = int(B24(r"blob (\d+) \+ tree"))
NP = int(B24(r"父提交 = (\d+)"))
NA = int(B24(r"新增 (\d+) \+ 修改"))
NMF = int(B24(r"新增 \d+ \+ 修改 (\d+)"))
EXPECT = int(B24(r"\+ 新增 (\d+) = (\d+)", 2))
assert NB == NP + NA == EXPECT, "闭合式在正文里不成立：%d != %d+%d != %d" % (NB, NP, NA, EXPECT)
ctlA = re.search(r"正对照 A（远端 \S+ 的 sha 首位改 f）⇒ 不符 (\d+)/(\d+)", b24)
ctlB = re.search(r"正对照 B（远端删一条路径）⇒ 本地独有 (\d+)/(\d+)，远端独有 (\d+)/(\d+)", b24)
ctlD = re.search(r"正对照 D（远端删本片一条路径）⇒ 本片缺失 (\d+)/(\d+)，SHA 不符 (\d+)/", b24)
assert ctlA and ctlB and ctlD, "四个正对照的读数取不齐"
assert int(ctlA.group(1)) >= 1 and int(ctlB.group(1)) == 1 and int(ctlD.group(1)) == 1, \
    "正对照没有一条能红 ⇒ 尺子空转"

# ================================================================ 本轮 #18 证据
rec = txt(RECEIPT)
n_total = int(re.search(r"共 (\d+) 条读数全部通过", rec).group(1))
n_pass = len(re.findall(r"^G\d+ PASS", rec, re.M))
assert (n_pass, n_total) == (19, 19), "接线回执不是 19/19：PASS 行 %d，声明 %d" % (n_pass, n_total)
assert "FAIL" not in rec, "接线回执里有 FAIL 行 ⇒ 本轮回执不能当验收证据归档"
g01 = re.search(r"G01 PASS\s*\n\s*pre=(\d+) post=(\d+)", rec)
g02 = re.search(r"相同=(\d+)/(\d+) 不同=(\[[^\]]*\])", rec)
g06 = re.search(r"peak=([\d.]+) n_steps=(\d+) nvox=(\d+)", rec)
g08 = re.search(r"pre=([0-9a-f]{32}) post=([0-9a-f]{32})", rec)
g09 = re.search(r"G09 PASS\s*\n\s*pre=(\d+) post=(\d+)", rec)
g10 = re.search(r"stderr 前 80 字='(.{0,80})'", rec)
g11 = re.search(r"机上 (\d+) 个进程，我的 (\d+) 个", rec)
g17 = re.search(r"代码行 (\d+)/\d+，含注释 (\d+) 处（正则=power_gate is None", rec)
for nm, gg in (("G01", g01), ("G02", g02), ("G06", g06), ("G08", g08), ("G09", g09),
               ("G10", g10), ("G11", g11), ("G17", g17)):
    assert gg, "回执里取不到 " + nm
assert int(g11.group(2)) == 0, "回执 G11：GPU 上有我的进程 ⇒ 与「不动 GPU」冲突"
assert g02.group(3) == "[]", "G02 的『不同』名单非空 ⇒ 缺省档不再是逐位相同"
assert g17.group(1) == "2", "G17 的成对位点数不是 2：%s" % g17.group(1)

assert open(FP_PRE_JSON := os.path.join(D, "_raw_s18_pre_fingerprint.json"), "rb").read() \
    == open(FP_POST, "rb").read(), "前后指纹 JSON 不同 ⇒ 缺省档动了"
fp = json.load(open(FP_POST, encoding="utf-8"))
fpeak = fp["fingerprint"]["peak_temperature"]
NFIELD_FP = len(fp["fingerprint"])
assert NFIELD_FP == 9, "指纹字段数不是 9，实得 %d" % NFIELD_FP

diff_spec = txt(PROBE_DIFF)
assert diff_spec.count("<") == 1 and diff_spec.count(">") == 1, "探针 pre/post 差集不是恰好一对行"
K0_PRE = re.search(r"^< (K0 环境 ::.*)$", diff_spec, re.M).group(1)
K0_POST = re.search(r"^> (K0 环境 ::.*)$", diff_spec, re.M).group(1)
assert "SRC=?" in K0_POST, "缺陷标本里那一行不含 SRC=? ⇒ 标本不是『指纹未 export』那次的运行"

# 侧翼两面（自带 CMD 的归档件）
side = txt(SIDE)
sec_col, sec_tgt = side.split("===== TARGETED 段 =====")
assert "rc_col=0" in sec_col, "collect-only 的 rc 不是 0"
assert re.search(r"^rc_tgt=1$", sec_tgt, re.M), "定向面 rc 须＝1（那条登记红还在），否则要么红丢了要么跑错了文件"
n_collect = int(re.search(r"(\d+) tests? collected", sec_col).group(1))
COL_CMD = re.search(r"^CMD: (.+)$", sec_col, re.M).group(1)
TGT_CMD = re.search(r"^CMD: (.+)$", sec_tgt, re.M).group(1)
tg_score = re.search(SCORE_RE, sec_tgt)
assert tg_score, "定向面取不到记分行"
tg_pair = re.search(r"^E\s+AssertionError: (熔体积不收敛：\S+ vs \S+)$", sec_tgt, re.M)
assert tg_pair, "定向面里那条登记红没印出熔体积读数"
n_deftest = len(re.findall(r"^def test_", txt(TS), re.M))
assert n_collect >= n_deftest, "收集数 %d < 函数数 %d" % (n_collect, n_deftest)

# ================================================================ 全量回归面
full = txt(FULLRAW)
assert open("/tmp/am_s18_full.log", "rb").read() == open(FULLRAW, "rb").read(), \
    "归档的全量 stdout 与 /tmp 原件不同 ⇒ 它不是那次运行的原件"
RC_FULL = re.search(r"^rc=(\d+)$", full, re.M).group(1)
FP_START = re.search(r"^SRCFINGER_START: ([0-9a-f]{32})$", full, re.M).group(1)
FP_END = re.search(r"^SRCFINGER_END: ([0-9a-f]{32})$", full, re.M).group(1)
D_START = re.search(r"^DATE_START: (.+)$", full, re.M).group(1)
D_END = re.search(r"^DATE_END: (.+)$", full, re.M).group(1)
TREE_LINE = re.search(r"^TREE: (.+)$", full, re.M).group(1)
CMD_LINE = re.search(r"^CMD: (.+)$", full, re.M).group(1)
assert FP_START == FP_END, "回归期间 src+tests 被改动 ⇒ 这次记分测的不是要提交的那棵树"
live_finger = subprocess.run(
    ["bash", "-c", "find src tests -name '*.py' -not -path '*__pycache__*' -print0 "
                   "| sort -z | xargs -0 md5sum | md5sum | cut -d' ' -f1"],
    cwd=REPO, capture_output=True, text=True).stdout.strip()
assert live_finger == FP_END, "现算指纹 %s ≠ 回归件止指纹 %s ⇒ 提交树不是被测树" % (live_finger, FP_END)
fs = re.search(SCORE_RE, full)
assert fs and fs.group(3) is not None, "全量记分行取不到（或没有 skipped 段）"
NF, NPA, NSI, FSEC = fs.groups()
assert RC_FULL == "1", "全量 rc 须为 1（两条登记红还在），实得 %s" % RC_FULL

base = txt(BASELINE)
bs = re.search(SCORE_RE, base)
BNF, BPA, BNSI, BSEC = bs.groups()
assert (BNF, BNSI) == ("2", "3"), "基线面的红/skip 数不是 2/3：%s/%s" % (BNF, BNSI)
assert (NF, NSI) == ("2", "3"), "现树面的红/skip 数不是 2/3：%s/%s" % (NF, NSI)
assert int(NPA) == int(BPA) + n_collect, \
    "passed %s ≠ 基线 %s ＋ 新网件收集 %s ⇒ 有测试掉出收集网，或基线件不是那一份" % (NPA, BPA, n_collect)


def asrt_lines(s):
    return re.findall(r"^E\s+AssertionError: .*$", s, re.M)


ba, fa = asrt_lines(base), asrt_lines(full)
assert len(ba) == len(fa) == 2, "两条面的 ^E AssertionError 行数不是 2/2：%d/%d" % (len(ba), len(fa))
assert ba == fa, "登记红灯的打印值变了 ⇒ 本轮改动动了物理：\n基线 %s\n现树 %s" % (ba, fa)
f_names = sorted(re.findall(r"^FAILED (\S+)", full, re.M))
b_names = sorted(re.findall(r"^FAILED (\S+)", base, re.M))
assert f_names == b_names and len(f_names) == 2, "FAILED 名单变了或不是 2 条：%s → %s" % (b_names, f_names)
pair_tgt = tg_pair.group(1)
assert pair_tgt in fa[0] or pair_tgt in fa[1], "定向面的熔体积读数不在全量面里 ⇒ 两次运行不同值"
assert "head=" + TRUE_HASH in TREE_LINE, "回归件的 TREE head %r 里没有当前 HEAD %s ⇒ 那次跑的不是这棵树" % (
    TREE_LINE, TRUE_HASH)

# 定向面的**复现**：首版无 CMD 件（⇒ 本轮为它另跑 stamped 面）与 stamped 面必须同分同读数
tgt_uns = txt(TGT_UNS)
us = re.search(SCORE_RE, tgt_uns)
up = re.search(r"^E\s+AssertionError: (熔体积不收敛：\S+ vs \S+)$", tgt_uns, re.M)
assert us and up, "无 stamp 的定向面原件取不到记分/熔体积行"
assert (us.group(1), us.group(2)) == (tg_score.group(1), tg_score.group(2)), \
    "定向面两次记分不同：%s/%s → %s/%s" % (us.group(1), us.group(2), tg_score.group(1), tg_score.group(2))
assert up.group(1) == pair_tgt, "定向面两次的熔体积读数不同：%s ≠ %s" % (up.group(1), pair_tgt)
assert not re.search(r"^CMD: ", tgt_uns, re.M), \
    "无 stamp 件里竟有 CMD 行 ⇒ 「为它另跑一枚 stamped 面」这条理由就是假的"

# skip/xfail 静态普查（pytest -q 不印 skip 名单 ⇒ 计数之外再看标记行数）
PAT = r"pytest\.mark\.(skip|xfail)"
skip_work = int(subprocess.run(["bash", "-c", "grep -rE --include=*.py '%s' tests | wc -l" % PAT],
                               cwd=REPO, capture_output=True, text=True).stdout)
skip_head = int(subprocess.run(["bash", "-c", "git grep -hE '%s' HEAD -- 'tests/*.py' | wc -l" % PAT],
                               cwd=REPO, capture_output=True, text=True).stdout)
skip_new = len(re.findall(PAT, txt(TS)))
assert skip_work == skip_head, "tests/ 的 skip/xfail 标记行数变了（HEAD %d → 现树 %d）⇒ 违反「不得新增 skip」" % (
    skip_head, skip_work)
assert skip_new == 0, "新网件里有 skip/xfail 标记 %d 处" % skip_new

# ================================================================ 代码腿读数
diffstat = sh("git", "diff", "--stat", "--", "src", "tests").splitlines()[-1].strip()
dirty_rows = [l for l in sh("git", "-c", "core.quotePath=false", "status", "--porcelain",
                            "--", "src", "tests").splitlines() if l.strip()]
n_dirty_m = sum(1 for l in dirty_rows if l[:2].strip() == "M")
n_dirty_a = sum(1 for l in dirty_rows if l[:2].strip() == "??")
assert n_dirty_m + n_dirty_a == len(dirty_rows), \
    "dirty 的改/新两分类没覆盖全部行：%d+%d ≠ %d ⇒ 注文里「N 改 M 新」那句会是假话" % (
        n_dirty_m, n_dirty_a, len(dirty_rows))
te_src, sp_src = txt(TE), txt(SP)
sites = {
    "opt-in 取值": r'program = p\.get\("scan_program"\)',
    "eager 守卫": r"require_program\(program, reference_power=P\)",
    "总长覆盖": r"path_length = program\.total_length\(\)",
    "位置覆盖": r"positions, _pw_col = program\.sample_steps\(n_steps\)",
    "门控构造": r"power_gate = gate_from_power\(_pw_col, P\)",
}
te_code = code_of(te_src)
site_counts = {k: len(re.findall(v, te_code)) for k, v in sites.items()}
assert all(v == 1 for v in site_counts.values()), "接线位点普查不齐 1：%s" % site_counts
pair_code = len(re.findall(r"power_gate is None", te_code))
pair_all = len(re.findall(r"power_gate is None", te_src))
assert (pair_code, pair_all) == (int(g17.group(1)), int(g17.group(2))), \
    "写手与接线回执对同一普查读数不一致：代码行 %d/%d、整份 %d/%d" % (
        pair_code, int(g17.group(1)), pair_all, int(g17.group(2)))


def hand(n_lines, x_ext=1.0, spacing=0.1, layer_height=0.2, n_layers=2):
    analytic = n_layers * (n_lines * x_ext + (n_lines - 1) * spacing)
    cross = math.hypot(spacing * (n_lines - 1), layer_height)
    reposition = 0.0 if n_lines % 2 == 0 else x_ext
    return analytic, cross, reposition, analytic + cross + reposition


H3, H2 = hand(3), hand(2)
ruler_md5 = md5(RULE24)
ruler_copies = sorted(f for f in os.listdir(D)
                      if re.fullmatch(r"am_s7_verify\d+\.py", f) and md5(os.path.join(D, f)) == ruler_md5)
assert "am_s7_verify24.py" in ruler_copies, "本片的尺子没被自己的 glob 数到"
GLUE = re.compile(r"。- \*\*2026-")
glue_hits = {rel: len(GLUE.findall(txt(os.path.join(REPO, rel)))) for rel in FIVE}
GLUE_TOT = sum(glue_hits.values())
assert GLUE_TOT == 0, "五份文档里仍有粘连条目 %d 处：%s" % (GLUE_TOT, glue_hits)
# 假哈希普查**分两段**（本片的教训：未分段的过期值检测器会拒绝自己写下的修复）
#   历史面＝五份文档的 **HEAD 版**，须 0 —— 这才是「没传染到文档」的证据；
#   标本面＝**工作树**，允许 >0，但每一处都必须自证是引文（同一行带「假」字标记）。
def head_of(rel):
    return subprocess.run(["git", "show", "HEAD:" + rel], cwd=REPO, capture_output=True,
                          text=True, check=True).stdout


def specimen_ok(text):
    """假哈希的每处出现都必须**与真值同段**（段＝空行切分）——这才是"我在记录一个已知错值"的结构证据。

    不用「假」这类措辞当标记：本轮五份文档里有一句写的是"短哈希写错"，措辞检测当场把它误判成传染，
    而下一轮谁换个说法（"笔误""旧值"）就会再次误报 ⇒ 标记必须取自读数本身（真值就在旁边）。
    """
    ok = bad = 0
    for para in text.split("\n\n"):
        n = para.count(FALSE_HASH)
        if n:
            ok, bad = (ok + n, bad) if TRUE_HASH in para else (ok, bad + n)
    return ok, bad


hash_head_hits = {rel: head_of(rel).count(FALSE_HASH) for rel in FIVE}
HASH_TOT = sum(hash_head_hits.values())
assert HASH_TOT == 0, "五份文档 HEAD 版已出现假短哈希 %d 处 ⇒ 真传染：%s" % (HASH_TOT, hash_head_hits)
_hash_ok = []
_hash_bad = []
_hash_work = 0
hash_doc = {}
for rel in FIVE:
    o, b = specimen_ok(txt(os.path.join(REPO, rel)))
    _hash_ok.append(o)
    _hash_bad.append(b)
    _hash_work += o + b
    hash_doc[rel] = o + b
HASH_WORK = _hash_work
UNMARKED = sum(_hash_bad)
assert UNMARKED == 0, "工作树有 %d 处引用假哈希却没有真值同段 ⇒ 传染，拒绝归档：%s" % (
    UNMARKED, {rel: n for rel, n in zip(FIVE, _hash_bad) if n})
# 正对照：把真值从各段抹掉 ⇒ 同一个检测器必须把全部标本判成传染（否则这条闸门是空转）
SPEC_CTRL = sum(specimen_ok(re.sub(re.escape(TRUE_HASH), "X", txt(os.path.join(REPO, rel))))[1]
                for rel in FIVE)
assert SPEC_CTRL == HASH_WORK == sum(_hash_ok), \
    "正对照不齐：抹掉真值后红 %d、工作树处数 %d、同段标本 %d" % (SPEC_CTRL, HASH_WORK, sum(_hash_ok))
assert TRUE_HASH != FALSE_HASH, "真假短哈希相同 ⇒ 这条普查是空转"
pairs = [(a, b) for a, b in re.findall(r"\*\*(\d+)(?:⇒(\d+))?\*\* 条", txt(OV))]
assert pairs, "总览里取不到上一条自纠计数 ⇒ 拒绝手填"
PREVCNT = int(pairs[-1][1] or pairs[-1][0])
assert PREVCNT > 0, "自纠上一条计数取回为 0 ⇒ 正则口径变了"

# 落笔时刻的 GPU 现取（回执与表头里的两个数只属于它们各自的那一刻，不能当"现在"读）
nv = [l.strip() for l in subprocess.run(["nvidia-smi", "--query-compute-apps=pid",
                                         "--format=csv,noheader"], capture_output=True,
                                        text=True).stdout.splitlines() if l.strip()]
nv_mine = [p for p in nv if "/tmp/amvenv" in
           open("/proc/%s/cmdline" % p, "rb").read().decode("utf-8", "replace")]
assert nv_mine == [], "落笔时我的解释器在卡上留有进程 %s ⇒ 与「不动 GPU」冲突，拒绝归档" % nv_mine

V = {
    "NOW": sh("date", "+%F %T %z"), "TIME_SHORT": sh("date", "+%H:%M"),
    "DATE_SHORT": sh("date", "+%Y-%m-%d"),
    "HEAD": TRUE_HASH, "HEADFULL": TRUE_FULL,
    "PARENTS": sh("git", "rev-parse", "--short", "HEAD^"),
    "DIRTY": str(len(dirty_rows)), "DIFFSTAT": diffstat,
    "NDIRTYM": str(n_dirty_m), "NDIRTYA": str(n_dirty_a),
    "NLSP": str(len(sp_src.splitlines())), "MDSP": md5(SP)[:12],
    "NLTS": str(len(txt(TS).splitlines())), "MDTS": md5(TS)[:12],
    "NLTE": str(len(te_src.splitlines())), "MDE": md5(TE)[:12],
    "NTDEF": str(n_deftest), "NCOLLECT": str(n_collect), "COLCMD": COL_CMD, "TGTCMD": TGT_CMD,
    "NREC": str(n_total), "NRECPASS": str(n_pass),
    "G01PRE": g01.group(1), "G01POST": g01.group(2),
    "SAME": g02.group(1), "SAMEDEN": g02.group(2),
    "PEAK": g06.group(1), "NSTEPS": g06.group(2), "NVOX": g06.group(3),
    "SPRE": g08.group(1), "SPOST": g08.group(2),
    "DIRTY_PRE": g09.group(1), "DIRTY_POST": g09.group(2),
    "G10HEAD": g10.group(1)[:46],
    "NVREC": g11.group(1), "NVME": g11.group(2),
    "NVNOW": str(len(nv)), "NVMENOW": str(len(nv_mine)),
    "PAIRCODE": g17.group(1), "PAIRALL": g17.group(2),
    "MDREC18": md5(RECEIPT)[:12], "MDPY18": md5(RECEIPT_PY)[:12],
    "MDPROBEPRE": md5(PROBE_PRE)[:12], "MDPROBEPOST": md5(PROBE_POST)[:12],
    "MDFPJSON": md5(FP_POST)[:12], "NFIELDFP": str(NFIELD_FP),
    "MDPEAK": fpeak["md5"][:12], "PEAKSUM": repr(fpeak["sum"]),
    "MDDIFF": md5(PROBE_DIFF)[:12], "K0PRE": K0_PRE, "K0POST": K0_POST,
    "TGPASS": tg_score.group(2), "TGFAIL": tg_score.group(1), "TGSEC": tg_score.group(4),
    "TGPAIR": pair_tgt,
    "TGUNFAIL": us.group(1), "TGUNPASS": us.group(2), "TGUNSEC": us.group(4),
    "MDTGTUNS": md5(TGT_UNS)[:12],
    "FCMD": CMD_LINE, "FTREE": TREE_LINE, "FSTART": D_START, "FEND": D_END,
    "FFP": FP_END, "FRC": RC_FULL, "NF": NF, "NPA": NPA, "NSI": NSI, "FSEC": FSEC,
    "BPA": BPA, "BSEC": BSEC, "BNF": BNF, "BNSI": BNSI,
    "BFAIL1": b_names[0].split("::")[1], "BFAIL2": b_names[1].split("::")[1],
    "ARED1": ba[0][2:].strip(), "ARED2": ba[1][2:].strip(),
    "SKIPHEAD": str(skip_head), "SKIPWORK": str(skip_work), "SKIPNEW": str(skip_new),
    "SITE1": str(site_counts["opt-in 取值"]), "SITE2": str(site_counts["eager 守卫"]),
    "SITE3": str(site_counts["总长覆盖"]), "SITE4": str(site_counts["位置覆盖"]),
    "SITE5": str(site_counts["门控构造"]),
    "ANA3": repr(H3[0]), "CROSS3": repr(H3[1]), "REP3": repr(H3[2]), "TOT3": repr(H3[3]),
    "ANA2": repr(H2[0]), "CROSS2": repr(H2[1]), "TOT2": repr(H2[3]),
    "NB": str(NB), "NP": str(NP), "NA": str(NA), "NMF": str(NMF), "EXPECT": str(EXPECT),
    "ENTRIES": B24(r"tree 条目 (\d+) = "), "NTREE": B24(r"blob \d+ \+ tree (\d+)"),
    "DL": B24(r"本地独有 (\d+)/"), "DR": B24(r"远端独有 (\d+)/"),
    "MS": B24(r"blob SHA 不符 (\d+)/"), "NSHARED": B24(r"blob SHA 不符 \d+/(\d+)"),
    "SLICE_MATCH": B24(r"MATCH (\d+)/"), "NFILES": B24(r"本片文件 (\d+) 个"),
    "CA": ctlA.group(1), "CBA": ctlA.group(2),
    "CB1": ctlB.group(1), "CDB1": ctlB.group(2), "CB2": ctlB.group(3), "CDB2": ctlB.group(4),
    "CD1": ctlD.group(1), "CDD1": ctlD.group(2), "CD2": ctlD.group(3),
    "VERDICT": VERDICT, "BOFF": str(boff),
    "CMP0": str(CMP[0]), "CMP1": str(CMP[1]), "CMP2": str(CMP[2]),
    "R24SIZE": str(os.path.getsize(REC24)), "R24MD": md5(REC24)[:12],
    "RAWM24": H24(r"md5＝([0-9a-f]{12})／"), "ERRM24": H24(r"md5＝[0-9a-f]{12}／([0-9a-f]{12})"),
    "DSTART24": H24(r"首次归档运行 (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d \+\d{4})"),
    "NRULE": str(len(ruler_copies)), "MDRULE": ruler_md5[:12],
    "NVPID": H24(r"compute 进程 (\d+) 个"), "NVME24": H24(r"所留 (\d+)/\d+ 个"),
    "RC24": H24(r"rc=(\d+)、stderr"), "ERRB24": H24(r"rc=\d+、stderr (\d+) 字节"),
    "NDEF": str(NDEF_LOG), "NFIELDDECL": str(NFIELD_DECL), "MDHDR24": md5(HDR24)[:12],
    "FULLSIZE": str(os.path.getsize(FULLRAW)), "FULLMD": md5(FULLRAW)[:12],
    "LAUNCHMD": md5(LAUNCHER)[:12], "SIDEMB": md5(SIDE)[:12], "SIDESHMD": md5(SIDE_SH)[:12],
    "GLUE_TOT": str(GLUE_TOT), "HASH_TOT": str(HASH_TOT), "FALSEH": FALSE_HASH,
    "HASH_WORK": str(HASH_WORK), "UNMARKED": str(UNMARKED), "SPECCTRL": str(SPEC_CTRL),
    "SPECDOC": "／".join("%s %d" % (rel.split("/")[-1], n) for rel, n in hash_doc.items()),
    "PREVCNT": str(PREVCNT), "CNT": str(PREVCNT + 1),
}

SEC_T = """
### 26.27 (u) #18 代码腿落地＝路径程序**可以**驱动热源（opt-in；缺省档逐位不变是两次运行**量出来的**）＋分片 24 回执入库；一条驻留律分歧、一个假短哈希落进台账（＋1）、三处仪器空转（＋0）

- **本轮约束**：用户令「继续，还是不动GPU」⇒ 每条碰 jax 的命令都内联 `CUDA_VISIBLE_DEVICES=""`
  ＋ `JAX_ENABLE_X64=1 PYTHONPATH=src` ＋ `/tmp/amvenv/bin/python`；本轮**不产生任何性能数字**
  （性能只在 GPU 上算）。树 `@@HEAD@@`（全值 `@@HEADFULL@@`，由命令现取）＋`dirty(src+tests)=@@DIRTY@@`
  （@@NDIRTYM@@ 改／@@NDIRTYA@@ 新，两分类之和＝行数，本片自己数），全量回归期间源码冻结。
- **(a) 改动清单（`git diff --stat`＝@@DIFFSTAT@@；新面另计）**
  1. **新模块** `src/amforge/scan_program.py`（@@NLSP@@ 行，md5 前 12＝@@MDSP@@）：`PathProgram`
     （`(n,3)` 折点＋`(n,)` 逐折点功率）、弧长表 `segment_lengths`/`cumulative_length`、`sample`/`sample_steps`、
     `gate_from_power`（门控的唯一出处）、`require_program`（eager 域校验、trace 下整体跳过）、
     `read_track_table`（**全部列**都返回＋utf-8→latin-1 退路）、`program_from_track_table`（`unit` 缺省 1.0＝不猜单位）、
     `implied_scan_speed`（诊断量）、`zigzag`（对照构造器）。零长段的 `sqrt` NaN 陷阱用**双层 where** 兜住
     ⇒ 前值精确 0、梯度精确 0（新位点已并入 #20 的普查）。
  2. `src/amforge/thermal_enthalpy.py`（@@NLTE@@ 行，md5 前 12＝@@MDE@@）＝接线 **5 处**（先剥注释再数，逐处 1）：
     opt-in 取值 @@SITE1@@／eager 守卫 @@SITE2@@／总长覆盖 @@SITE3@@／位置覆盖 @@SITE4@@／门控构造 @@SITE5@@，
     另有 `power_gate is None` 的**成对**缺省分支 @@PAIRCODE@@ 处；`chain_schedule(..., program=None)` 按折线总长
     **定价**曝光，并公开警告「hatch/layer_thickness 这两项对温度场梯度恒零」。
  3. **新测试** `tests/test_scan_program.py`（@@NLTS@@ 行，md5 前 12＝@@MDTS@@；`^def test_` @@NTDEF@@ 个函数、
     `--collect-only` 实收 **@@NCOLLECT@@ 条**）⇒ 进收集网。
  4. 三条口径（模块头登记，不是偏好）：**时间映射不被取代**（`t_exposure ＝ 折线总长/scan_speed` ⇒
     `|∂输出/∂scan_speed|>0` 那条 A0 断言对程序档同样成立）；**功率是比值门控** `gate＝power(t)/laser_power`
     （常功率列逐位＝1.0，混合写成 `a+u·(b−a)` 而非 `(1−u)a+ub`）；**纯 jnp**（`idx＝jnp.sum(cum<=s)`）
     ⇒ 可进 `lax.scan`/`jit`，`n_steps` 是静态形状、折点与功率列可求梯度。
- **(b) 「缺省档逐位不变」是量出来的，不是注释声称的**
  接线前探针（SRC 指纹 `@@SPRE@@`、dirty=@@DIRTY_PRE@@，`_raw_s18_probe.out` md5＝@@MDPROBEPRE@@）与接线后重跑
  （SRC `@@SPOST@@`、dirty=@@DIRTY_POST@@，`_raw_s18_probe_post.out` md5＝@@MDPROBEPOST@@）⇒ 比对发生在**两条不同源码树**之间。
  读数：@@NREC@@ 道闸门的 G01 印 pre=@@G01PRE@@／post=@@G01POST@@ 条读数，G02 印 **相同＝@@SAME@@/@@SAMEDEN@@**
  （唯一不参与比对的 K0 环境行自己就印指纹与 dirty）；ThermalHistory 的 **@@NFIELDFP@@ 个字段指纹逐字节相同**
  （`_raw_s18_post_fingerprint.json` md5＝@@MDFPJSON@@；`peak_temperature` max＝@@PEAK@@ K、sum＝@@PEAKSUM@@、
  md5 前 12＝@@MDPEAK@@；算例侧 n_steps＝@@NSTEPS@@、熔格 @@NVOX@@）。机制是**静态分支**：`power_gate is None`
  由 Python 在 trace 期判 ⇒ 缺省分支的 jaxpr 里**根本没有那次乘法**。
- **(c) 判别力（同一轮里既有"不变"也有"变了"）**
  · `test_solver_with_equivalent_program_is_bit_identical_to_default`：把同一拓扑展开成折线接进去 ⇒ 全字段与缺省档
    逐位相同，且夹具**必须真的熔过**（`peak>1500`）才允许说"相同"；
  · `test_solver_beam_off_program_does_not_melt_and_default_does`：功率列写 0 ⇒ `peak<1200` 且
    `Σ time_above_melt == 0.0`，而同夹具不加门控时 `peak>1500` ⇒ 门控**真的乘进去了**；
  · `test_solver_still_differentiable_wrt_scan_speed_with_program`：接了程序 `|∂Σpeak/∂v|` 仍非零 ⇒ 时间映射没被取代；
  · 侧翼两面（启动器 `am_s18_side_run.sh` md5＝@@SIDESHMD@@，输出 `_raw_s18_side.log` md5＝@@SIDEMB@@）
    **各自内联印 CMD** ⇒ 归档件自带口径：收集面 `@@COLCMD@@` 实收 **@@NCOLLECT@@ 条**，
    定向面 `@@TGTCMD@@` 印 **@@TGFAIL@@ failed, @@TGPASS@@ passed in @@TGSEC@@s**，红的那条印「@@TGPAIR@@」
    ⇒ 新网件没动物理。同一面对**跑过两次**：首版定向面的 stdout 没带命令行（无法证明跑的是哪两个文件），
    补 stamped 面后与首版原件（`_raw_s18_targeted.out` md5＝@@MDTGTUNS@@，@@TGUNFAIL@@ failed,
    @@TGUNPASS@@ passed in @@TGUNSEC@@s）现比＝**记分与那条红的读数两两相同**（本写手有断言），
    而首版件里确实没有 `^CMD: ` 行（同一条断言的反面）⇒ 两份都入库，判读以 stamped 那份为准。
- **(d) 全量回归面（CPU 钉住，启动器自印 stamps；档案＝`am_s18_full_run.sh` md5＝@@LAUNCHMD@@ ＋ `_raw_s18_full.log` @@FULLSIZE@@ 字节 md5＝@@FULLMD@@）**
  `@@FCMD@@`；`@@FTREE@@`；@@FSTART@@ → @@FEND@@；SRCFINGER 起＝止＝`@@FFP@@`＝本写手**现算**的提交树指纹（两边同值
  ⇒ 回归期间没动被测码）；rc＝@@FRC@@（登记红还在 ⇒ 须 1）。记分 **@@NF@@ failed, @@NPA@@ passed, @@NSI@@ skipped in @@FSEC@@s**；
  基线（#35 归档件）＝ **@@BNF@@ failed, @@BPA@@ passed, @@BNSI@@ skipped in @@BSEC@@s**，闭合式
  `@@BPA@@ ＋ 新收集 @@NCOLLECT@@ ＝ @@NPA@@` 成立 ⇒ 没有测试掉出收集网；skip 另有**静态普查**（`pytest -q` 不印名单）：
  标记行 HEAD @@SKIPHEAD@@ → 现树 @@SKIPWORK@@，新网件 @@SKIPNEW@@ 处。两条登记红的**打印值逐行相等**
  （口径＝`^E\\s+AssertionError: ` 两行，基线与现树 2/2）：①@@ARED1@@；②@@ARED2@@；名单亦未变
  （`@@BFAIL1@@`／`@@BFAIL2@@`）⇒「A0 断言不许放宽」「不许换指标」两条红线维持。
- **(e) 一处物理口径分歧（发现，不掩盖，也不改断言）**：默认档的**驻留律**在换道/换层时"瞬移"（那些步不记曝光），
  而解析式 `path_length` 对**道间跳距**计费、对**跨层迁移**不计费 ⇒ 一条**连续**折线描述同一片扫描时总长必然更长，
  超出量恰＝跨层段＋换层后对位段。手算常数（夹具 `x_ext=1.0`、`spacing=0.1`、`layer_height=0.2`、`n_layers=2`）：
  `n_lines=3` ⇒ 解析 @@ANA3@@ ＋ 跨层 @@CROSS3@@ ＋ 对位 @@REP3@@ ＝ **@@TOT3@@**；
  `n_lines=2` ⇒ 解析 @@ANA2@@ ＋ 跨层 @@CROSS2@@ ＋ 对位 0.0 ＝ **@@TOT2@@**（奇偶之差就是一道对位段）。
  已由参数化测试 `test_multilayer_excess_is_jump_plus_reposition[even/odd]` 钉住，并附分区校核 `Σ段长==总长`。
  ⇒ 含义：接了程序之后曝光时长由**真行程**决定（这正是 #18 要的），但**默认档与程序档的"速度↔时长"换算不可混用**；
  这条已写进 #10/A3 的注意事项（基准表自带时间轴只能经 `implied_scan_speed` 作诊断量看）。
- **(f) 两处既有缺陷登记为 #41（实测发现，当场**不**修）**：① `diffmech/methods/am/scan_paths.py:from_csv` 只取两列
  （功率/时间被静默丢掉），且对 A3 基准表**连读都读不进**——探针 K10f 现测 `UnicodeDecodeError：'utf-8' codec
  can't decode byte 0xb5`（µ 是非 UTF-8 单字节）；② `amforge/meltpool.py:polyline_trajectory` 在构造期
  `total_time=float(...)`（K3b 实得 type=float 值=34.0）⇒ `∂pos/∂speed` **精确 0**、`jax.grad` 内构造直接抛
  `TracerArrayConversionError`（K3c 原文）。不归 #18 修的理由与耦合面（#38 门禁网外／#25＋#33 同批重标定）写在台账里，
  两处都必须**先全树普查同一模式**再动。
- **(g) 分片 24 落地回执入库**（读数**引自回执** `am_s7_verify24.log` @@R24SIZE@@ 字节、md5＝@@R24MD@@，正文区起点 BODY_OFFSET＝@@BOFF@@）：
  远端 main＝`@@HEAD@@`＝本地 HEAD（父 `@@PARENTS@@`）；`truncated=False`、tree 条目 @@ENTRIES@@＝blob @@NB@@＋tree @@NTREE@@；
  闭合式 **远端 @@NB@@ == 父提交 @@NP@@ ＋ 新增 @@NA@@ ＝ @@EXPECT@@ -> PASS**；双向路径差集 **@@DL@@/@@NB@@** 与 **@@DR@@/@@NB@@**；
  全库共有项 SHA 不符 **@@MS@@/@@NSHARED@@**；本片单列复核 **@@SLICE_MATCH@@/@@NFILES@@ MATCH**（A @@NA@@／M @@NMF@@）；`@@VERDICT@@`。
  四个正对照同运行印出且**都能红**：A 首位改 f ⇒ 不符 @@CA@@/@@CBA@@；B 删一条路径 ⇒ 本地独有 @@CB1@@/@@CDB1@@、远端独有 @@CB2@@/@@CDB2@@；
  C 闭合式右边 +1 ⇒ 判 FAIL；D 删本片一条 ⇒ 缺失 @@CD1@@/@@CDD1@@、SHA 不符 @@CD2@@/@@CDD1@@。
  回执自身的复核：`cmp -i @@BOFF@@:0` rc＝**@@CMP0@@**，左 +1 rc＝@@CMP1@@、右 +1 rc＝@@CMP2@@（各须非 0）⇒ 正文区确是那次 stdout 的逐字节后缀，
  两侧都在仓库里；表头顶格字段 **@@NDEF@@** 行＝生成器 `FIELDS` 声明数 **@@NFIELDDECL@@**（这条**不写死 16**：写死＝把两个文件的耦合藏进常数）；
  表头生成时复跑核验件 rc＝@@RC24@@、stderr @@ERRB24@@ 字节且 stdout 与正文区逐字节相同；原始件首次归档时刻 @@DSTART24@@
  （`cp -p` ⇒ mtime 是运行时刻），md5＝@@RAWM24@@／@@ERRM24@@；表头生成器 `am_s24_hdr.py` md5＝@@MDHDR24@@；
  同一把尺子（md5 前 12＝@@MDRULE@@）本目录 glob＋md5 现算已有 **@@NRULE@@** 份。
- **(h) 自纠与仪器进步**
  · **+1（§26.27 由 @@PREVCNT@@ 到 @@CNT@@）＝任务台账 #14 的短哈希是假的**：台账写 `@@FALSEH@@`，
    `git rev-parse --short HEAD` 实得 `@@HEAD@@`（全值 `@@HEADFULL@@`）。它是**已经落进档案**的错数
    （跨会话会被下一轮当事实读）⇒ 按分界计 +1。普查（现算）＝五份文档各数一次，假值命中 **@@HASH_TOT@@ 处** ⇒ 没传染到文档；
    修法＝台账改写＋登记口径「短哈希一律由命令现取，不手敲」（本片每一个 `@@HEAD@@` 都由命令来）。
  · **+0（拦在写盘前）＝三处仪器空转/失真**：① 位点普查首版按**整份文本**匹配，把注释里那句解释也数成位点 ⇒
    `power_gate is None` 读成 @@PAIRALL@@ 而不是 @@PAIRCODE@@。修法＝`code_of()` 先剥注释，并把"坏口径整份＝@@PAIRALL@@
    ／好口径代码行＝@@PAIRCODE@@"**同时**登记成一条闸门（G19），这样以后有人加注释不会把位点数读多；
    ② "把总长覆盖行改成注释 ⇒ 普查必须变红"的正对照首版在**剥注释之后**才动手术 ⇒ 永远红不了；修法＝动完手术后重跑整条
    管道（G18，实测注释掉之后代码行命中 0）；③ 探针的 `SRCFINGER` 靠环境变量传入，忘 export 时那一行印 `SRC=?`，
    标本已入库（`_raw_s18_probe_diff.txt` md5＝@@MDDIFF@@；前『@@K0PRE@@』/ 后『@@K0POST@@』），重跑才拿到 post 指纹
    ⇒ 新登记口径：**判读件首行必须含被测树指纹，取不到就拒绝判读**（本写手按这条把 `@@FFP@@` 现算再比对）。
  · 仪器复用：`am_s24_hdr.py` 由上一片生成器 `sed` 机械克隆，sed 换了外层片号 23→24、**括号里的"分片 22"原样留下**
    ⇒ 上一片的故事被安在本片头上；读 diff 时发现（拦在写盘前 ⇒ +0），修法不是手改数字而是把那句**改成从
    `git show --name-status HEAD` 现取的清单**，并与核验件印出的 A/M 计数互证（两者不符即拒绝归档）。
  · 渲染结构普查（现算，同一粘连模式在五份文档各数一次）＝**@@GLUE_TOT@@ 处** ⇒ 上一片的 mid 修复没有复发；
    本片的 mid 腿仍带**正对照**（把插入块首字符的换行去掉，同一处下标必须非换行）⇒ 这条闸门不是空转。
    台账也是档案面：那条假短哈希就是**只写在台账里**、五份文档普查 @@HASH_TOT@@ 处才暴露的。
- **(i) GPU／钉住**：接线回执 G11 现取卡上 compute 进程 **@@NVREC@@** 个、我的解释器（`/proc/PID/cmdline` 含 `/tmp/amvenv`）
  所留 **@@NVME@@/@@NVREC@@**（须 0）；分片 24 表头生成时 **@@NVPID@@** 个、我的 **@@NVME24@@/@@NVPID@@**。全轮未占 GPU，
  读数只记录不解释。**本写手落笔时（@@NOW@@）再取一次＝@@NVNOW@@ 个、我的＝@@NVMENOW@@/@@NVNOW@@**（非 0 就拒绝写盘）
  ⇒ 那三段历史读数各属于自己那一刻，不冒充"现在"。钉住证据：探针 stderr 只有已知的插件配置错误前缀
  （G10 印 `@@G10HEAD@@…`），没有出现新的报错形态。
- **(j) 排期**：#18 关闭 ⇒ 下一块 **#27＋#33（同批，外部 18-track 靶）→ #10/A3**；#40 → #39；#41 待自己一轮
  （两处都要先普查同一模式）；#29／#30 仍等用户裁决。落笔 @@NOW@@（`date` 现取）。
"""

SEC_PLAN = """
## §V（@@NOW@@ 落笔，`date` 现取）＝**#18 关闭：路径程序能驱动热源；"opt-in 不改缺省"是两棵源码树之间的实测性质**，并把"默认档↔程序档驻留律分歧"登记成 A3 换算注意事项

- **交付物**：`src/amforge/scan_program.py`（@@NLSP@@ 行）＋ `thermal_enthalpy.py` 接线 5 处（先剥注释再数：
  opt-in/守卫/总长/位置/门控各 @@SITE1@@／@@SITE2@@／@@SITE3@@／@@SITE4@@／@@SITE5@@，缺省分支成对 @@PAIRCODE@@）
  ＋ `tests/test_scan_program.py`（@@NTDEF@@ 个函数、实收 **@@NCOLLECT@@ 条**）。三条口径：时间映射不被取代、
  功率是**比值门控**（常功率列逐位＝1.0）、纯 jnp（可进 scan/jit）。`git diff --stat`＝@@DIFFSTAT@@。
- **验收**：接线回执 **@@NREC@@/@@NREC@@ 全过**；两棵树（`@@SPRE@@`→`@@SPOST@@`）探针 **@@SAME@@/@@SAMEDEN@@** 逐字节相同、
  ThermalHistory **@@NFIELDFP@@** 个字段指纹相同（peak＝@@PEAK@@ K）；全量回归 **@@NF@@ failed, @@NPA@@ passed,
  @@NSI@@ skipped in @@FSEC@@s**＝基线 @@BPA@@＋新收集 @@NCOLLECT@@（闭合式成立），两条登记红的
  `^E\\s+AssertionError` 行与基线**逐行相等**；skip/xfail 标记行 @@SKIPHEAD@@→@@SKIPWORK@@（无新增，新网件 @@SKIPNEW@@）。
- **物理侧新认知（登记，不改断言）**：默认档驻留律在层/道边界瞬移、解析总长只对道间跳距计费 ⇒
  连续折线总长＝解析＋跨层＋对位（手算常数：n_lines=3 时 @@ANA3@@→@@TOT3@@；n_lines=2 时 @@ANA2@@→@@TOT2@@）。
  ⇒ **#10/A3 的"速度↔时长"换算在两个档位之间不可混用**；基准表自带的时间轴只能经 `implied_scan_speed` 作诊断量看。
- **另立 #41（实测发现、当场不修）**：`scan_paths.from_csv` 只取两列且对 A3 基准表 `UnicodeDecodeError 0xb5` 连读都读不进；
  `meltpool.polyline_trajectory` 构造期 `float()` ⇒ `∂pos/∂speed` 精确 0、`jax.grad` 内构造抛 `TracerArrayConversionError`。
  两处都要先全树普查同一模式（耦合 #38 与 #25/#33 重标定）。
- **自纠 §26.27 @@PREVCNT@@⇒@@CNT@@**：+1＝任务台账 #14 的短哈希写错（`@@FALSEH@@`，实为 `@@HEAD@@`；五份文档普查
  @@HASH_TOT@@ 处 ⇒ 没传染）；+0＝位点普查把注释读成位点、正对照动手术晚于剥注释＝空转、探针 SRCFINGER 未 export 印出
  `SRC=?`、sed 克隆把上一片故事安在本片头上（四条都拦在写盘前）。
- **分片 24 落地（回执入库）**：远端 main＝`@@HEAD@@`、blob **@@NB@@**、闭合式 **@@NB@@ == @@NP@@＋@@NA@@＝@@EXPECT@@**、
  双向差集 **@@DL@@/@@NB@@** 与 **@@DR@@/@@NB@@**、全库 SHA 不符 **@@MS@@/@@NSHARED@@**、本片 **@@SLICE_MATCH@@/@@NFILES@@ MATCH**
  （A @@NA@@／M @@NMF@@）⇒ 分片 23 的回执链在库。回执＝`docs/evidence/2026-10-09/am_s7_verify24.log`
  （`cmp -i @@BOFF@@:0` rc＝@@CMP0@@ ＋两侧 +1 各红；表头顶格字段数由生成器 `FIELDS` 声明现取＝@@NDEF@@）。
- **排期（就地更新）**：**#27＋#33（同批）→ #10/A3**；#40 → #39；#41 待自己一轮；#29／#30 仍等用户裁决。
  红线不变：A0 断言不放宽、#27 不换指标、无新增 skip/xfail、性能数字只在 GPU 窗口测（当前「不动 GPU」⇒ 无吞吐论断）。
"""

SEC_OV = """- **@@DATE_SHORT@@（#18 代码腿落地＋分片 24 回执入库）＝路径程序能驱动热源；"opt-in 不改缺省"由两次运行量出来**。
  新模块 `src/amforge/scan_program.py`（折线＋逐折点功率、纯 jnp 弧长采样、比值门控、`require_program` 域校验）
  ＋ `thermal_enthalpy.py` 接线 5 处（**先剥注释再数**：5 处各 1、`power_gate is None` 成对 @@PAIRCODE@@）
  ＋ 新测试 @@NCOLLECT@@ 条进收集网（函数 @@NTDEF@@ 个）。读数：接线回执 **@@NRECPASS@@/@@NREC@@ 全过**；两棵树
  （`@@SPRE@@`→`@@SPOST@@`）探针 **@@SAME@@/@@SAMEDEN@@** 逐字节相同、ThermalHistory **@@NFIELDFP@@** 字段指纹相同
  （peak＝@@PEAK@@ K）；全量 **@@NF@@ failed, @@NPA@@ passed, @@NSI@@ skipped in @@FSEC@@s**（基线 @@BPA@@＋新收集
  @@NCOLLECT@@＝@@NPA@@；两红 `^E AssertionError` 逐行相等；skip 标记行 @@SKIPHEAD@@→@@SKIPWORK@@）⇒ A0 断言未放宽、无新增 skip。
  分片 24 回执 `am_s7_verify24.log`：远端 main＝`@@HEAD@@`、tree 条目 @@ENTRIES@@＝blob **@@NB@@**＋@@NTREE@@、闭合式
  **@@NB@@ == @@NP@@＋@@NA@@＝@@EXPECT@@**、双向差集 **@@DL@@/@@NB@@** 与 **@@DR@@/@@NB@@**、SHA 不符 **@@MS@@/@@NSHARED@@**、
  本片 **@@SLICE_MATCH@@/@@NFILES@@ MATCH**、`@@VERDICT@@`（`cmp -i @@BOFF@@:0` rc＝@@CMP0@@ ＋两侧 +1 各红）。
  **四条新口径**：① 位点普查必须剥注释（整份文本把成对 @@PAIRCODE@@ 处读成 @@PAIRALL@@ 处）；② 正对照要在**动完手术后
  重跑整条仪器管道**（否则是永远红不了的空转闸门）；③ 判读件首行必须含被测树指纹，取不到（`SRC=?`）就拒绝判读；
  ④ 与生成器耦合的常数（顶格字段数）从声明现取、不写死 16（@@NDEF@@＝@@NFIELDDECL@@）⇒ §26.27 自纠
  **@@PREVCNT@@⇒@@CNT@@** 条（+1＝任务台账 #14 的短哈希 `@@FALSEH@@` 是假的，实为 `@@HEAD@@`；台账也是档案面）。
  GPU 现取：卡上 compute 进程 **@@NVREC@@** 个、我的＝**@@NVME@@/@@NVREC@@**。
"""

SEC_DAY = """
### @@TIME_SHORT@@ #18 代码腿落地（路径程序驱动热源）＋全量回归＋分片 24 回执入库
代码腿＝新模块 `scan_program.py`（@@NLSP@@ 行）＋ `thermal_enthalpy.py` 接线 5 处＋`tests/test_scan_program.py`
（@@NTDEF@@ 函数／实收 @@NCOLLECT@@ 条）；`git diff --stat`＝@@DIFFSTAT@@，dirty(src+tests)=@@DIRTY@@。
「opt-in 不改缺省」的度量：接线前探针 SRC=`@@SPRE@@`、接线后 SRC=`@@SPOST@@`（**两条不同源码树**之间），
读数 **@@SAME@@/@@SAMEDEN@@ 逐字节相同**（唯一不参与比对的 K0 行自己就印指纹），ThermalHistory **@@NFIELDFP@@** 个
字段指纹相同（peak＝@@PEAK@@ K、md5 前 12＝@@MDPEAK@@、n_steps＝@@NSTEPS@@、熔格 @@NVOX@@）；机制＝
`power_gate is None` 是 trace 期的 Python 静态判断 ⇒ 缺省分支的 jaxpr 里没有那次乘法。
接线回执 `am_s18_wiring_receipt.py`（md5＝@@MDPY18@@）**@@NRECPASS@@/@@NREC@@** 全过，原始 stdout md5＝@@MDREC18@@、stderr 0 字节。
全量回归（启动器 `am_s18_full_run.sh` md5＝@@LAUNCHMD@@，CMD＝@@FCMD@@，@@FSTART@@→@@FEND@@，rc=@@FRC@@，
SRCFINGER 起＝止＝`@@FFP@@`＝提交树现算值）＝**@@NF@@ failed, @@NPA@@ passed, @@NSI@@ skipped in @@FSEC@@s**，
基线 @@BPA@@＋新收集 @@NCOLLECT@@＝@@NPA@@（无测试掉网；skip 标记行 @@SKIPHEAD@@→@@SKIPWORK@@、新网件 @@SKIPNEW@@）；
两条登记红的 `^E AssertionError` 与基线**逐行相等**（@@ARED1@@ ／ @@ARED2@@）。定向面（@@TGTCMD@@）
＝@@TGFAIL@@ failed, @@TGPASS@@ passed in @@TGSEC@@s（红印「@@TGPAIR@@」）。
物理侧登记：默认档驻留律在层/道边界瞬移 ⇒ 连续折线总长＝解析＋跨层＋对位（n_lines=3：@@ANA3@@→@@TOT3@@；
n_lines=2：@@ANA2@@→@@TOT2@@）⇒ 默认档与程序档的速度↔时长换算不可混用（写进 #10/A3 注意事项）。
另立 **#41**（两处既有缺陷：`scan_paths.from_csv` 只取两列＋`0xb5` 读不进基准表；`polyline_trajectory` 构造期
`float()` ⇒ `∂pos/∂speed` 精确 0／grad 内抛 `TracerArrayConversionError`）。分片 24 回执入库：远端 main＝`@@HEAD@@`、
blob **@@NB@@**、闭合式 **@@NB@@ == @@NP@@＋@@NA@@＝@@EXPECT@@**、双向差集 **@@DL@@/@@NB@@** 与 **@@DR@@/@@NB@@**、
SHA 不符 **@@MS@@/@@NSHARED@@**、本片 **@@SLICE_MATCH@@/@@NFILES@@ MATCH**、`cmp -i @@BOFF@@:0` rc＝@@CMP0@@＋两侧 +1 各红；
尺子现算 @@NRULE@@ 份（前 12＝@@MDRULE@@）。自纠 §26.27 **@@PREVCNT@@⇒@@CNT@@** 条（+1＝台账短哈希 `@@FALSEH@@` 假，
实为 `@@HEAD@@`，五份文档普查 @@HASH_TOT@@ 处；+0＝位点普查没剥注释、正对照动手术晚于剥注释、SRCFINGER 未 export、
sed 把上一片故事安在本片头上）。GPU：接线回执时现取 compute 进程 @@NVREC@@ 个，我的解释器所留
**@@NVME@@/@@NVREC@@**；本写手落笔时（@@NOW@@）再取＝@@NVNOW@@ 个、我的 **@@NVMENOW@@/@@NVNOW@@**；
本轮全部 CPU 钉住，无性能数字。
"""

SEC_MEM = """
- **@@DATE_SHORT@@（#18 代码腿落地＋分片 24 回执入库）**：路径程序（折线＋逐折点功率）经
  `params["thermal"]["scan_program"]` **opt-in** 接进默认热链；缺省档"逐位不变"由**两棵源码树**的探针比对量出来
  （`@@SPRE@@`→`@@SPOST@@`，@@SAME@@/@@SAMEDEN@@ 条读数相同、ThermalHistory @@NFIELDFP@@ 字段指纹相同、peak＝@@PEAK@@ K），
  机制是 trace 期的 Python 静态分支（缺省 jaxpr 里无乘法）。全量 **@@NF@@ failed, @@NPA@@ passed,
  @@NSI@@ skipped in @@FSEC@@s**＝基线 @@BPA@@＋新收集 @@NCOLLECT@@，两红 `^E AssertionError` 逐行相等 ⇒ 红线未动。
  **新口径四条**：① 位点普查**先剥注释**（整份文本会把成对的 @@PAIRCODE@@ 处读成 @@PAIRALL@@ 处）；
  ② 正对照必须**动完手术后重跑整条仪器管道**（在剥注释之后才动手术＝永远红不了的空转闸门）；
  ③ 判读件首行必须含被测树指纹，取不到（`SRC=?`）就拒绝判读；④ 与生成器耦合的常数（顶格字段数）从声明现取、不写死 16
  ⇒ §26.27 自纠 **@@PREVCNT@@⇒@@CNT@@** 条（+1＝任务台账 #14 的短哈希 `@@FALSEH@@` 是假的，实为 `@@HEAD@@`：
  **台账也是会被跨会话当事实读的档案面**）。另立 #41（两处既有轨迹读取/可微缺陷，实测发现、当场不修）。
  排期：**#27＋#33（同批）→ #10/A3**；#40→#39；#29／#30 等裁决。
"""

# 第六腿：开发日志的**落笔后复检**补记。(h) 那段假哈希普查写的是"落笔前"的单面读数，
# 而本段自纠条目自己就引了那个假值 ⇒ 同一条命令再跑一次会把自己判红。锚点不手敲：
# 直接取开发日志**当前最后一行**（它含上一腿渲染出来的 `date` 时间戳 ⇒ 只有那次运行能对上）。
DEV_LAST = [l for l in txt(DEV).splitlines() if l.strip()][-1]
FIX_MARKER = "落笔后复检"
if FIX_MARKER not in txt(DEV):      # 补记腿还没落时才校验锚点；已落则本腿会 SKIP，末行已经是补记自己
    assert re.search(r"落笔 \d{4}-\d\d-\d\d \d\d:\d\d:\d\d \+\d{4}（`date` 现取）。$", DEV_LAST), \
        "开发日志末行不是上一条 (j) 排期句 ⇒ 结构变了，拒绝追加补记"

SEC_FIX = """
- **(k) 落笔后复检（+0，拦在提交前）＝上一条 (h) 那个假哈希普查把自己写红了 ⇒ 检测器分段**：
  (h) 里的「五份文档普查 **@@HASH_TOT@@ 处**」是**落笔前**工作树的读数（那个 0 是真的），而本段 (u) 落笔时
  按口径**故意**在自纠条目里引了那个假值 ⇒ 同一条命令再跑一次现数 **@@HASH_WORK@@ 处**
  （@@SPECDOC@@，五份各 1 处）。首版闸门按「工作树须 0」判 ⇒ 写手**拒绝自己的修复**，第二次运行直接抛错。
  修法＝把过期值检测器**分成两面**：历史面（五份文档的 `git show HEAD:` 版）须 **@@HASH_TOT@@ 处**，
  这才是「没传染到文档」的证据；工作树面允许 >0，但**每一处都必须自证是标本**——判据取读数本身：
  同一**段**（空行切分）里必须同时出现真值 `@@HEAD@@`。现数未自证处＝**@@UNMARKED@@**（非 0 即拒绝归档）。
  正对照＝把真值从各段抹掉再喂同一个检测器，必须逐处变红（现数 **@@SPECCTRL@@/@@HASH_WORK@@ 红**）
  ⇒ 这条闸门不是空转，也没有被放宽成"永远过"。
  **措辞标记是第一版的错**：首版用「假」字当标本标记，而计划文档那句写的是"短哈希写错" ⇒ 一份合法标本
  被读成传染（拦在写盘前，同一轮里发现）；标记必须取自读数（真值就在旁边），不能取自措辞。
  分界照旧：闸门件（`am_s25_docs.py`／`am_s25_msg.py`）随本片一起提交、尚未落地 ⇒ **+0**；
  而文档里那句"0 处"缺时间限定，由本补记当场补上限定，不另计。新口径入档：
  **凡"数一个已知坏值"的过期值检测器都必须分段（current／specimen），否则会拒绝自己的修复并把下一轮拦在写盘外。**
  GPU 同一落笔时刻现取：卡上 compute 进程 **@@NVNOW@@** 个、我的解释器 **@@NVMENOW@@/@@NVNOW@@**（非 0 拒绝写盘）。
  本片的提交说明由 `am_s25_msg.py` 生成，同一个普查在说明里也印成两面（@@HASH_TOT@@／@@HASH_WORK@@）。
"""

LEGS = [
    ("开发日志 (u)", DEV, "docs/开发日志.md", "### 26.27 (u)", SEC_T,
     ("  #40 → #39；#29／#30 仍等用户裁决。落笔 2026-10-09 16:25:24 +0800（`date` 现取）。", "eof")),
    ("计划文档 §V", PL, "docs/项目评估与下一步计划_2026-10-06.md", "§V（", SEC_PLAN,
     ("归档的**渲染结构**被写坏后才发现，不是数字错）。代码腿未动（SRC 指纹 4a5a257efaf3251badbdfbfb10ecec4b），下一项仍是 **#18**。", "eof")),
    ("总览", OV, "docs/项目开发总览.md", "#18 代码腿落地＋分片 24 回执入库", SEC_OV,
     ("  **21⇒22** 条。GPU 现取：卡上 compute 进程 **4** 个、我的＝**0/4**。", "mid")),
    ("当日", DAY, ".workbuddy/memory/2026-10-09.md", "#18 代码腿落地（路径程序驱动热源）＋全量回归＋分片 24 回执入库", SEC_DAY,
     ("代码腿未动（dirty=0、SRC 指纹 4a5a257efaf3251badbdfbfb10ecec4b）。GPU：现取 compute 进程 4 个，我的解释器所留\n**0/4**；本轮全部 CPU 钉住。", "eof")),
    ("MEMORY", MEM, ".workbuddy/memory/MEMORY.md", "#18 代码腿落地＋分片 24 回执入库", SEC_MEM,
     ("  本片修复并把插入模式改成先补换行 ⇒ §26.27 自纠 **21⇒22** 条。下一项 **#18**（CSV 时间轴 opt-in）。", "eof")),
    ("开发日志 (k) 补记", DEV, "docs/开发日志.md", FIX_MARKER, SEC_FIX, (DEV_LAST, "eof")),
]

ALL = "".join(s for _n, _p, _r, _m, s, _a in LEGS)
missing = sorted(set(re.findall(r"@@([A-Za-z0-9_]+)@@", ALL)) - set(V))
assert not missing, "模板有未取值的字段：" + str(missing)
unused = sorted(set(V) - set(re.findall(r"@@([A-Za-z0-9_]+)@@", ALL)))
assert not unused, "算了读数却没进任何模板（口径漂了）：" + str(unused)

report = []
for name, path, rel, marker, sec, (anchor, mode) in LEGS:
    cur = txt(path)
    if marker in cur:
        report.append("SKIP " + rel + "：标记已存在 ⇒ 该腿已落，不重复追加")
        continue
    n = cur.count(anchor)
    assert n == 1, rel + "：锚点出现 " + str(n) + " 次（须 1）⇒ 拒绝写盘：" + repr(anchor[:44])
    if mode == "eof":
        assert cur.rstrip().endswith(anchor.rstrip()), rel + "：锚点不在文件末尾 ⇒ 结构变了，拒绝盲写"
        i = len(cur)
    else:
        i = cur.index(anchor) + len(anchor)
        assert cur[i:].lstrip().startswith("---"), rel + "：插入点后不是分隔线 ⇒ 结构变了，拒绝盲写"
    rendered = ("\n" + sec) if mode == "mid" else sec
    for k, v in V.items():
        rendered = rendered.replace("@@" + k + "@@", v)
    if "@@" in rendered or "＠" in rendered:
        print("【残留PLACEHOLDER】", rel)
        for l in rendered.splitlines():
            if "@@" in l or "＠" in l:
                print("  残留行｜", l)
        raise SystemExit(3)
    new = (cur.rstrip("\n") + "\n" + rendered) if mode == "eof" else (cur[:i] + rendered + cur[i:])
    assert mode == "eof" or new[i] == "\n", rel + "：插入块首字符不是换行 ⇒ 会重演粘连"
    if mode == "mid":
        assert (cur[:i] + sec + cur[i:])[i] != "\n", "正对照失效：漏写换行的插入没被抓住 ⇒ 那条断言是空转闸门"
    assert len(GLUE.findall(new)) == 0, rel + "：本腿写完仍有粘连条目 ⇒ 拒绝落盘"
    if WRITE:
        open(path, "w", encoding="utf-8").write(new)
        back = txt(path)
        assert back == new, rel + "：写后回读 ≠ 预期全文"
        assert back.count(marker) == 1, rel + "：标记出现 " + str(back.count(marker)) + " 次"
        report.append("WRITE " + rel + "：" + str(cur.count("\n")) + " → " + str(back.count("\n"))
                      + " 行（＋" + str(back.count("\n") - cur.count("\n")) + "），标记 1 次")
    else:
        report.append("PRECHECK " + rel + "：" + str(cur.count("\n")) + " → " + str(new.count("\n"))
                      + " 行（未写盘）")

print("\n".join(report))
print("代码腿：%s｜scan_program %s 行 md5 %s｜tests %s 行 md5 %s｜solver %s 行 md5 %s｜收集 %s（函数 %s）"
      % (V["DIFFSTAT"], V["NLSP"], V["MDSP"], V["NLTS"], V["MDTS"], V["NLTE"], V["MDE"],
         V["NCOLLECT"], V["NTDEF"]))
print("缺省不变：树 %s → %s，读数 %s/%s 相同，指纹字段 %s，peak=%s n_steps=%s nvox=%s｜接线回执闸门 %s/%s"
      % (V["SPRE"], V["SPOST"], V["SAME"], V["SAMEDEN"], V["NFIELDFP"], V["PEAK"],
         V["NSTEPS"], V["NVOX"], V["NRECPASS"], V["NREC"]))
print("全量：%s failed / %s passed / %s skipped in %ss（基线 %s/%s/%s in %ss；闭合 %s＋%s＝%s；rc=%s；SRCFINGER 起止 %s；skip 标记 %s→%s 新 %s）"
      % (V["NF"], V["NPA"], V["NSI"], V["FSEC"], V["BNF"], V["BPA"], V["BNSI"], V["BSEC"],
         V["BPA"], V["NCOLLECT"], V["NPA"], V["FRC"], V["FFP"], V["SKIPHEAD"], V["SKIPWORK"], V["SKIPNEW"]))
print("分片 24：main %s｜blob %s 闭合 %s==%s+%s=%s｜差集 %s/%s、%s/%s｜SHA 不符 %s/%s｜本片 %s/%s｜BOFF %s cmp %s/%s/%s｜顶格 %s＝FIELDS %s｜尺子 %s 份"
      % (V["HEAD"], V["NB"], V["NB"], V["NP"], V["NA"], V["EXPECT"], V["DL"], V["NB"], V["DR"], V["NB"],
         V["MS"], V["NSHARED"], V["SLICE_MATCH"], V["NFILES"], V["BOFF"], V["CMP0"], V["CMP1"], V["CMP2"],
         V["NDEF"], V["NFIELDDECL"], V["NRULE"]))
print("自纠总数现值（总览末条）%s｜若再 +1 则为 %s｜台账假哈希普查分两面：历史面(HEAD) %s 处／工作树面 %s 处＝未与真值同段 %s、抹掉真值后正对照红 %s｜值＝%s，真值＝%s｜粘连普查 %s 处｜GPU 我的进程＝ %s/%s"
      % (V["PREVCNT"], V["CNT"], V["HASH_TOT"], V["HASH_WORK"], V["UNMARKED"], V["SPECCTRL"],
         V["FALSEH"], V["HEAD"], V["GLUE_TOT"],
         V["NVME"], V["NVREC"]))
if not WRITE:
    print("（预检模式：加 --write 才落盘。）")
