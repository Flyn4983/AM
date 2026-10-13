#!/usr/bin/env python3
r"""S7 分片 31 提交信息生成器：信息里每个数字都从命令／档案现取，不许手敲。

本轮＝**纯归档腿**（`git diff --name-only -- src tests` 现数 0 件）⇒ 信息写五件可核查的事：
① §26.32 登记的形制约束由「另起 §26.33」满足（节内唯一性由**暂存 blob**现验，四条锚句各恰好 1 条）；
② 分片 30 的落地半腿整条链入库（回执读数全部从回执与原始件正则现取，且**复跑** cmp 三件套）；
③ 本轮新学两条：`-uall` 的目录折叠口径、**普查仪器（NVML）坏了 ⇒ 红线改由 /proc 的 argv[0] 整词匹配承担**；
④ 本轮抓到并就地修的一处**错标**：§26.33 落笔那句把「表头区非空行数 99」填进了「顶格字段」那个槽（真值＝16）；
⑤ 本件**实跑**上一片的归档自查件，用它自己的红来登记「局部仪器不是常备尺子」。

片号与"上一片回执"一律**现取并交叉印证**（§26.27 (u)、§26.30、§26.31、§26.32 都记过机械克隆把上一片的故事安在本片头上）：
`git log -1 --format=%s` 给出 HEAD 的片号 ⇒ 本片＝它＋1；全树 glob 回执档案给出最大号角本 ⇒ 必须＝HEAD 片号；
本片的号必须**缺席**；上一片回执的**目录**由那一份回执自己被 glob 命中而定（本轮跨日期：它在 10-12，本件在 10-13）。

十条口径（① — ⑦沿用，⑧ — ⑩本轮新增）：
① 正文一律 **PLACEHOLDER＋str.replace**，不用 f-string（f-string 会把字面量里的 `{32}` 当替换字段**静默吃掉**）。
   生成后**双向**查：模板里没取值的字段拒绝，取值表里模板没用到的读数也拒绝（没用到的读数＝下一条等着的谎）。
② 引命令**原文**并把三个数闭合到 `--numstat` 逐行现算（git 对 1 用单数「1 deletion(-)」，按复数形状取数会静默折成 0）。
③ 每个 0 与它的分母同印；每条闸门都要在 `am_s31_msg_controls.py` 里被**坏输入**打红过（只看 rc≠0 的控制＝空转）。
④ **单值取数按形状类并要求恰好一次**：`search` 在多处命中时静默拿第一条 ⇒ 本件所有取数走 `one()`／`grab()`。
⑤ **局部仪器不许当常备尺子**：本件实跑上一片的**两件**局部仪器（归档自查件、上一片的提交说明生成器；后者只往 stdout 打，
   不落仓库 ⇒ 复跑无副作用），各要求它红在自己那一轮的锚句上，且红的 stderr 与已归档的复跑现场**逐字节相同**
   （两出处：本轮实跑＝档案里那次）。上一片控件面**不复跑**（它扰动的是它自己那轮的生成器副本，本轮读它的数只从档案取），
   这一点在信息里如实分开写。
⑥ **角色桶与 §26.33 登记的分组闭合**：落笔那份把 62 件分成 12 组并写了等式 `5＋4＋3＋4＋9＋3＋5＋12＋5＋1＋1＋10＝62`；
   本件按文件名模式现数同一批桶，逐组等值才算那条登记不是凭记忆写的。落笔之后才存在的三桶单独成桶，
   并与那句「最终入库数大于这里的 67」互证。
⑦ **换算出来的读数也要有第二出处**：凡把原始读数换算成人类可读量的地方，都必须与档案里那句换算逐字闭合。
⑧ 本轮：**指纹的 locale 耦合**仍是活的——同一条口径两种环境两个值 ⇒ 参与闭合的那条必须来自**不钉 `LC_ALL`** 的命令，
   另一条只断言"不同"；两个读数都印，并与 §26.32（HEAD 那版）登记的十六进制串互证。
⑨ 本轮：**落笔计数随本轮新面移动**是既有事实，不是腐烂。上一片的「未跟踪 ∉ 入库清单＝0」闸在本轮**按构造不可能为 0**
   （本件的 stdout 落 `/tmp` 而不是仓库；未跟踪面也在增长）。⇒ 本件把它改写成两条同印的读数：
   「当前未跟踪数＝入库清单 A 数」的差额 0 件＋分母，与"本轮新落的件名"清单。照抄一条本轮满足不了的闸＝把上一片的故事安在本片头上。
⑩ 本轮：**槽位接错**（⑨ 的另一面）。一个数字现取、过闸、印出，仍可能填进另一个标签的槽——§26.33 落笔的
   「顶格字段 99 行」就是「表头区非空行数」被填进了「顶格字段」那个槽（真值 16）。⇒ 本件加一条普查闸门：
   全开发日志里「顶格字段」紧跟的数字**不许不等于** FIELDS 声明数（本轮 16）；坏输入（改回 99）进控件面。
"""
import ast
import glob
import hashlib
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
DAY = os.path.basename(HERE)
E = "docs/evidence/" + DAY
INTERP = sys.executable


def sh(*args):
    r = subprocess.run(list(args), cwd=REPO, capture_output=True, text=True, check=True)
    return r.stdout


def one(pat, text, what):
    """单值取数：命中 0 处或 >1 处都拒绝（多处＝静默拿第一条＝假读数，见口径 ④）。"""
    outs = re.findall(pat, text, re.M)
    assert len(outs) == 1, "%s：模式取到 %d 处（须恰好 1）⇒ 出处不唯一或已腐烂，拒绝生成" % (what, len(outs))
    g = outs[0]
    return g[0] if isinstance(g, tuple) and len(g) == 1 else g


def grab(pat, text, what):
    """多字段取数同样要求**恰好一次**命中。"""
    outs = re.findall(pat, text, re.M)
    assert len(outs) == 1, "%s：模式取到 %d 处（须恰好 1）⇒ 出处不唯一或已腐烂，拒绝生成" % (what, len(outs))
    return outs[0]


def rb(path):
    p = path if os.path.isabs(path) else os.path.join(REPO, path)
    return open(p, "rb").read()


def md5(path, n=None):
    h = hashlib.md5(rb(path)).hexdigest()
    return h if n is None else h[:n]


def rd(path):
    return rb(path).decode("utf-8")


def b(path):
    return len(rb(path))


def run_bash(cmd, env=None):
    return subprocess.run(["bash", "-c", cmd], cwd=REPO, capture_output=True, text=True,
                          env=env if env is not None else dict(os.environ, CUDA_VISIBLE_DEVICES="")).stdout


def argv0(pid):
    try:
        blob = open("/proc/%s/cmdline" % pid, "rb").read()
    except OSError:
        return None
    return blob.split(b"\0")[0].decode("utf-8", "replace") if blob else None


def match_argv0(blob):
    return blob.split(b"\0")[0].decode("utf-8", "replace") == INTERP


# ---------- 片号：四处互证＋上一片回执目录由 glob 现取（跨日期不硬写） ----------
SUBJ = sh("git", "log", "-1", "--format=%s").strip()
HEADSLICE = int(one(r"S7 分片 (\d+)", SUBJ, "HEAD 提交说明里的片号"))
PREVSLICE, SLICE = HEADSLICE, HEADSLICE + 1
LOGS = sorted(int(re.fullmatch(r"am_s7_verify(\d+)\.log", os.path.basename(p)).group(1))
              for p in glob.glob(os.path.join(REPO, "docs/evidence/*/am_s7_verify*.log"))
              if re.fullmatch(r"am_s7_verify(\d+)\.log", os.path.basename(p)))
assert LOGS, "全树 glob 没取到任何回执档案 ⇒ 计数口径变了"
assert max(LOGS) == PREVSLICE, "最新回执号角本 %s ≠ HEAD 片号 %s ⇒ 有一片没落或片号口径变了" % (max(LOGS), PREVSLICE)
assert SLICE not in LOGS, "本轮回执 %s 已经存在 ⇒ 提交说明不能声称它将由本片入库" % SLICE
PREVSEC = int(one(r"开发日志 §26\.(\d+)", SUBJ, "HEAD 提交说明里的日志节号"))
PREVLOGS = sorted(os.path.relpath(p, REPO) for p in
                  glob.glob(os.path.join(REPO, "docs/evidence/*/am_s7_verify%d.log" % PREVSLICE)))
assert len(PREVLOGS) == 1, "上一片回执在全树命中 %d 份（须 1）：%s ⇒ 有副本，引用哪一份都不唯一" % (
    len(PREVLOGS), PREVLOGS)
CURL = PREVLOGS[0]
PREVDIR = os.path.dirname(CURL)
assert PREVDIR != E, "上一片回执与本件同目录 ⇒ 本轮的跨日期登记要改（见 §26.33 的入库面那条）"
RPY = sorted(int(re.fullmatch(r"am_s7_verify(\d+)\.py", os.path.basename(p)).group(1))
             for p in glob.glob(os.path.join(REPO, "docs/evidence/*/am_s7_verify*.py"))
             if re.fullmatch(r"am_s7_verify(\d+)\.py", os.path.basename(p)))
assert RPY, "全树尺子 glob 空 ⇒ 第四个片号出处没了"
RULER = "%s/am_s7_verify%d.py" % (PREVDIR, PREVSLICE)
HDRC = "%s/am_s%d_hdr.py" % (PREVDIR, PREVSLICE)
RAWR = "%s/_raw_verify%d.out" % (PREVDIR, PREVSLICE)
RAWR_ERR = RAWR[:-4] + ".err"
HDR_PRE = "%s/_raw_s%d_hdr_precheck.out" % (PREVDIR, PREVSLICE)
HDR_WR = "%s/_raw_s%d_hdr_write.out" % (PREVDIR, PREVSLICE)
HDRCTL = "%s/_raw_s%d_hdr_controls.out" % (PREVDIR, PREVSLICE)
HDRCTL_PY = "%s/am_s%d_hdr_controls.py" % (PREVDIR, PREVSLICE)
AFPY = "%s/am_s%d_hdr_after.py" % (PREVDIR, PREVSLICE)
AFOUT = "%s/_raw_s%d_hdr_after.out" % (PREVDIR, PREVSLICE)
AFCTL = "%s/_raw_s%d_hdr_after_controls.out" % (PREVDIR, PREVSLICE)
MSGCTL = "%s/_raw_s%d_msg_controls.out" % (PREVDIR, PREVSLICE)
LANDPY = "%s/am_s%d_docs_landing.py" % (PREVDIR, PREVSLICE)
LANDPRE = "%s/_raw_s%d_landing_precheck.out" % (PREVDIR, PREVSLICE)
LANDWR = "%s/_raw_s%d_landing_write.out" % (PREVDIR, PREVSLICE)
ARCHPY = "%s/am_s%d_archive_check.py" % (PREVDIR, PREVSLICE)
ARCHK_RAW = "%s/_raw_s%d_archive_check.out" % (PREVDIR, PREVSLICE)
ARCHK_ERR = ARCHK_RAW[:-4] + ".err"
ARCHK_A1_ERR = "%s/_raw_s%d_archive_check.attempt1-claim-space-mismatch.err" % (PREVDIR, PREVSLICE)
# 本件自己这一轮（10-13 目录）留下的面
WRITER = "%s/am_s%d_docs.py" % (E, SLICE)
WPRE = "%s/_raw_s%d_docs_precheck.out" % (E, SLICE)
WPRE_ERR = WPRE[:-4] + ".err"
WWRITE = "%s/_raw_s%d_docs_write.out" % (E, SLICE)
WWRITE_ERR = WWRITE[:-4] + ".err"
T1_OUT = "%s/_raw_s%d_docs_precheck.attempt1-selfcount-recursive.out" % (E, SLICE)
T1_ERR = T1_OUT[:-4] + ".err"
T2_ERR = "%s/_raw_s%d_docs_write.attempt2-selfguard-hits-itself.err" % (E, SLICE)
ARCHK_PIN_ERR = "%s/_raw_s%d_archive_check_rerun_in_s%d.err" % (E, PREVSLICE, SLICE)
ARCHK_PIN_OUT = ARCHK_PIN_ERR[:-4] + ".out"
# 上一片的**提交说明生成器**：本轮实跑它，要求它红在自己那一轮的跨日期闸上（第二条局部仪器现场）
PREVMSG = "%s/am_s%d_msg.py" % (PREVDIR, PREVSLICE)
MSG_PIN_ERR = "%s/_raw_s%d_msg_rerun_in_s%d.err" % (E, PREVSLICE, SLICE)
MSG_PIN_OUT = MSG_PIN_ERR[:-4] + ".out"
DAILY = ".workbuddy/memory/%s.md" % DAY
PREVDAILY = ".workbuddy/memory/%s.md" % os.path.basename(PREVDIR)
DEVLOG = "docs/开发日志.md"
SELF = "am_s%d_msg.py" % SLICE
SELFCTL = "am_s%d_msg_controls.py" % SLICE
CTLSELF = "%s/%s" % (E, SELFCTL)
REGS = sorted(os.path.relpath(p, REPO) for p in glob.glob(os.path.join(REPO, "docs/evidence/*/_raw_t*_full.log")))
assert len(REGS) == 1, "全树回归件命中 %d 份（须 1）：%s" % (len(REGS), REGS)
REG = REGS[0]
FINGER_CMD = ("find src tests -name '*.py' -not -path '*__pycache__*' -print0 "
              "| sort -z | xargs -0 md5sum | md5sum | cut -d' ' -f1")
# 「纯归档腿」那句的作用域：数的是**暂存集**里 src／tests 的改动（提交带走的就是暂存集），
# 显示串与执行 argv 由同一个常量派生，免得模板里那句命令与真正的命令两条腿分叉。
NCODE_ARGV = ("git", "diff", "--cached", "--name-only", "--", "src", "tests")
NCODE_CMD = " ".join(NCODE_ARGV)
LESSON_NEEDLE = "自审的方向性"
NEWLESSON_NEEDLE = "普查仪器没有第二出处"

RE_TALLY = re.compile(r"^(\d+) failed, (\d+) passed, (\d+) skipped in ([\d.]+)s", re.M)
RE_TREE = re.compile(r"^TREE: head=(\w+) dirty=(\d+)$", re.M)
RE_FINGER = re.compile(r"^SRCFINGER_(START|END): (\w{32})$", re.M)
RE_DATE = re.compile(r"^DATE_(START|END): (.+)$", re.M)
RE_ELINES = re.compile(r"^E\s+AssertionError: .*$", re.M)
RE_SHORTSTAT = re.compile(r"(\d+) files? changed(?:, (\d+) insertions?\(\+\))?(?:, (\d+) deletions?\(-\))?")
RE_SEC = re.compile(r"^### 26\.(\d+)[^\n]*$", re.M)
RE_BULLET = re.compile(r"^- \*\*(.+?)\*\*", re.M)

K_CLOSE = r"闭合式 远端 (\d+) == 父提交 (\d+) \+ 本片新增\(A\) (\d+) ＝ (\d+) -> (\w+)"
K_SLICE = r"本片单列复核 (\d+)/(\d+) MATCH（A (\d+)／M (\d+) 个文件）"
K_DIFF = r"双向路径差集 (\d+)/(\d+) 与 (\d+)/(\d+)"
K_SHA = r"全库 blob SHA 不符 (\d+)/(\d+)"
# 四条各捕**一个原子读数**（整段逐字节摘自回执）：早先写成多字段元组时，B／C／D 三格落到槽位上是元组，
# 渲染腿的类型闸（`isinstance(val, str)`）当场拒印＝本轮实测抓到的真 bug，不是假红。
K_CTLA = r"A 远端一条 sha 首位改 f ⇒ (不符 \d+/\d+)"
K_CTLB = r"B 远端删一条路径 ⇒ (本地独有 \d+/\d+、远端独有 \d+/\d+)"
K_CTLC = r"C 闭合式右边 \+1 ⇒ (\d+ == \d+ 判 \w+)"
K_CTLD = r"D 远端删\*\*本片\*\*一条路径 ⇒ (本片缺失 \d+/\d+、SHA 不符 \d+/\d+)"
K_BODY = r"② truncated=(\w+)，tree 条目 (\d+) = blob (\d+) \+ tree (\d+)"
K_HEAD = r"head=(\w{40}) parent=(\w{40}) dirty\(src\+tests\)=(\d+)"
K_FINGER = r"^SRC指纹 : (\w{32})"
K_MD5 = r"本件 md5 ([0-9a-f]{12})"
K_BOFF = r"BODY_OFFSET=(\d+)"
K_PREV2 = r"上一片（(\d+) 号）回执链"
K_COPIES = r"同一把尺子已有 (\d+) 份逐字节副本"
K_NDEF = r"顶格字段行数＝ (\d+) / (\d+)"
K_TRIPLE = r"cmp -i 三联（本件自己起进程跑的，不是抄生成器的）＝ (\d+) / (\d+) / (\d+)"

# §26.33「入库面」那条登记的 12 组（标签与顺序必须与本件前 12 个桶逐一对应）
BKEY_PIN = ["回执链", "表头预检与写盘两面", "表头控件面", "控件面红灯现场", "只读仪器与其各面",
            "只读仪器的控件面", "落地腿", "落地腿红灯现场", "归档自查腿", "当日纪实",
            "本片归档写入件", "本件自己的预检与写盘两面（含红灯现场）"]
# 落笔之后才存在的三桶：本轮的两件复跑现场＋本信息生成器与它的干跑控件
BKEY_SELF = ["上一片归档自查件在本轮的复跑现场", "本信息生成器与干跑控件", "本件复跑上一片信息生成器的现场"]
BUCKETS = [
    ("回执链", r"^am_s7_verify%d\.(py|log)$|^_raw_verify%d\.(out|err)$|^am_s%d_hdr\.py$" % (
        PREVSLICE, PREVSLICE, PREVSLICE)),
    ("表头预检与写盘两面", r"^_raw_s%d_hdr_(precheck|write)\.(out|err)$" % PREVSLICE),
    ("表头控件面", r"^am_s%d_hdr_controls\.py$|^_raw_s%d_hdr_controls\.(out|err)$" % (PREVSLICE, PREVSLICE)),
    ("控件面红灯现场", r"^_raw_s%d_hdr_controls\.attempt\d-[a-z0-9-]+\.(out|err)$" % PREVSLICE),
    # ⚠ 这条**必须**把裸的 `.out` 也列进去：落笔第一版只写了 `\.err` 与四个带后缀的形状，于是
    #    `_raw_s30_hdr_after.out` 落进 0 个桶、该桶现数 8 而 §26.33 那句登记的是 9。跑前拿真实未跟踪清单
    #    预演一次「逐组等值」才撞出来（桶形与等式各自内部都自洽，只有两边对齐时才红）。
    ("只读仪器与其各面", r"^am_s%d_hdr_after\.py$|^_raw_s%d_hdr_after(\.out|\.err|_pre_final\.(out|err)|"
     r"_red1\.(out|err)|_write\.(out|err))$" % (PREVSLICE, PREVSLICE)),
    ("只读仪器的控件面", r"^am_s%d_hdr_after_controls\.py$|^_raw_s%d_hdr_after_controls\.(out|err)$" % (
        PREVSLICE, PREVSLICE)),
    ("落地腿", r"^am_s%d_docs_landing\.py$|^_raw_s%d_landing_(precheck|write)\.(out|err)$" % (PREVSLICE, PREVSLICE)),
    ("落地腿红灯现场", r"^_raw_s%d_landing_(precheck|write)\.attempt\d-[a-z0-9-]+\.(out|err)$" % PREVSLICE),
    ("归档自查腿", r"^am_s%d_archive_check\.py$|^_raw_s%d_archive_check(\.attempt1-[a-z0-9-]+)?\.(out|err)$" % (
        PREVSLICE, PREVSLICE)),
    ("当日纪实", r"^%s\.md$" % DAY),
    ("本片归档写入件", r"^am_s%d_docs\.py$" % SLICE),
    ("本件自己的预检与写盘两面（含红灯现场）",
     r"^_raw_s%d_docs_(precheck|write)(\.attempt\d-[a-z0-9-]+)?\.(out|err)$" % SLICE),
    ("上一片归档自查件在本轮的复跑现场", r"^_raw_s%d_archive_check_rerun_in_s%d\.(out|err)$" % (PREVSLICE, SLICE)),
    # 本件的红灯现场用 `.attemptN-<原因>` 名（与桶 4／8／9／12 同形制），所以两个后缀都写成可选：
    # 落笔第一版只列 `_controls_red\d+`，于是第一次实跑的红灯面（本件那条基线判据写窄了＝标记句在副本
    # stdout 里命中 2 行）落进 0 个桶——桶形自己红在「逐件须落进恰好 1 桶」，而不是悄悄少一件。
    ("本信息生成器与干跑控件",
     r"^am_s%d_msg(_controls)?\.py$|^_raw_s%d_msg(_controls|_controls_red\d+)?(\.attempt\d-[a-z0-9-]+)?\.(out|err)$" % (
         SLICE, SLICE)),
    ("本件复跑上一片信息生成器的现场", r"^_raw_s%d_msg_rerun_in_s%d\.(out|err)$" % (PREVSLICE, SLICE)),
]
assert [k for k, _ in BUCKETS] == BKEY_PIN + BKEY_SELF, "桶名与桶序不一致 ⇒ 与 §26.33 分组的配对口径不可用"


def staged():
    rows = [l.split("\t") for l in sh("git", "-c", "core.quotePath=false", "diff", "--cached",
                                      "--name-status").splitlines() if l.strip()]
    kinds = sorted({r[0] for r in rows})
    assert kinds == ["A", "M"], "暂存清单里有 A/M 之外的状态字母：%s" % kinds
    return rows, sorted(r[-1] for r in rows if r[0] == "A"), sorted(r[-1] for r in rows if r[0] == "M")


def bucketize(added):
    """角色桶：一件落进 0 个或 ≥2 个桶都拒绝生成；空桶也拒绝（桶形是凭记忆写的就会空）。"""
    hits = {name: [] for name, _ in BUCKETS}
    for p in added:
        bn = os.path.basename(p)
        m = [name for name, pat in BUCKETS if re.search(pat, bn)]
        assert len(m) == 1, "%s 落进 %d 个桶（%s）⇒ 角色口径不唯一，拒绝生成" % (bn, len(m), "、".join(m) or "无")
        hits[m[0]].append(bn)
    assert sum(len(v) for v in hits.values()) == len(added), "桶和 %d ≠ 新增总数 %d" % (
        sum(len(v) for v in hits.values()), len(added))
    empty = [k for k, v in hits.items() if not v]
    assert not empty, "角色桶 %s 里一件都没有 ⇒ 那个角色的命名口径已变，桶形是凭记忆写的" % empty
    return hits


def arch_pin(body):
    """§26.33「入库面」那条里的落笔读数：角色分组等式＋未跟踪／修改闭合式＋修改面清单，全从**暂存 blob**现取。"""
    seg = one(r"按角色分组：([^；]+)；", body, "本片登记的角色分组句")
    toks = seg.split("／")
    assert len(toks) == len(BKEY_PIN), "§26.33 的角色分组 %d 项 ≠ 生成器前桶 %d 项 ⇒ 命名口径变了" % (
        len(toks), len(BKEY_PIN))
    rows = []
    for i, (t, key) in enumerate(zip(toks, BKEY_PIN)):
        lab, want, claim = grab(r"^(.+?) (\d+) \*\*(\d+)\*\*$", t, "分组项（应为「%s」）的形状" % key)
        assert lab.startswith(key), "分组第 %d 项是「%s」而非「%s」⇒ 顺序与本件桶序不一致" % (i + 1, lab, key)
        assert int(want) == int(claim), "%s：标签里的数 %s ≠ 加粗声称 %s" % (lab, want, claim)
        rows.append((lab, int(claim)))
    eq = one(r"；([^；]*?＝\d+) 与", body, "分组等式句")
    lhs = [int(x) for x in eq.split("＝")[0].split("＋")]
    assert lhs == [c for _l, c in rows], "分组等式左边 %s ≠ 各组声称 %s" % (lhs, [c for _l, c in rows])
    nu, nm, tot = grab(r"未跟踪 \*\*(\d+)\*\*＋修改 \*\*(\d+)\*\*＝\*\*(\d+)\*\* 件", body, "落笔入库面闭合式")
    nu, nm, tot = int(nu), int(nm), int(tot)
    assert nu == sum(c for _l, c in rows) and nu + nm == tot, "落笔闭合式不齐：分组和 %d／未跟踪 %d／修改 %d／总计 %d" % (
        sum(c for _l, c in rows), nu, nm, tot)
    nface, faces = grab(r"修改 (\d+) 件＝[^（\n]*（([^）]+)）", body, "修改面清单")
    # 那个括号里在「清单」之后还跟着一段解释（本轮是「；其中 2026-10-12.md 是上一片的纪实——…」），
    # 括号直到解释结束才闭合 ⇒ 直接按 `）` 切会把解释一起当成清单（第一版就这么红了）。
    # 切到第一个 `；` 为止，并要求**切出来的条数＝那句声称的件数**——对不上就红，不靠我相信分隔符的位置。
    faces = faces.split("；")[0]
    lst = sorted(faces.split("／"))
    assert int(nface) == len(lst), "修改面清单现数 %d 项 ≠ 那句声称 %s 件（切到 `；` 为止）⇒ 分隔口径变了" % (
        len(lst), nface)
    return rows, (nu, nm, tot), int(nface), lst


def shortstat_check():
    raw = sh("git", "-c", "core.quotePath=false", "diff", "--cached", "--shortstat").strip()
    sm = RE_SHORTSTAT.search(raw)
    assert sm, "shortstat 形状不认识：%r" % raw
    rows = [l.split("\t") for l in sh("git", "-c", "core.quotePath=false", "diff", "--cached",
                                      "--numstat").splitlines() if l.strip()]
    assert all(r[0] != "-" for r in rows), "numstat 里有 `-` 行（二进制／rename）⇒ 逐行求和口径不成立"
    ns = (len(rows), sum(int(r[0]) for r in rows), sum(int(r[1]) for r in rows))
    ss = tuple(int(x) if x else 0 for x in sm.groups())
    assert ss == ns, "shortstat %s ≠ numstat 现算 %s" % (ss, ns)
    return raw, ss


def fields_of(hdr_py):
    tree = ast.parse(rd(hdr_py))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "FIELDS":
            return [ast.literal_eval(e) for e in node.value.elts]
    raise AssertionError("在 %s 里取不到 FIELDS 赋值 ⇒ 顶格字段的分母不是现取的" % hdr_py)


def gate_curl(text):
    """上一片回执：读数全部现取（每条要求恰好一次），任一条不绿就拒绝引用它。"""
    remote, parent, na, expect, vc = grab(K_CLOSE, text, "回执里的闭合式")
    assert int(remote) == int(parent) + int(na) == int(expect), "闭合式不成立 %s/%s/%s" % (remote, parent, na)
    assert vc == "PASS", "闭合式判读＝%s" % vc
    verdict = one(r"^(LANDING_VERDICT = .*)$", text, "回执的判读行")
    assert verdict.endswith("ALL CHECKS PASS"), "回执判读行不是 ALL CHECKS PASS：" + verdict
    match, nfile, a, m = grab(K_SLICE, text, "回执里的单列复核")
    assert int(match) == int(nfile) and int(a) + int(m) == int(nfile), "单列复核不全绿"
    dl, dlden, dr, drden = grab(K_DIFF, text, "回执里的双向差集")
    sha, shaden = grab(K_SHA, text, "回执里的 SHA 普查")
    assert int(dl) == int(dr) == int(sha) == 0, "回执里有非零差集／SHA 不符"
    assert int(dlden) == int(drden) == int(shaden), "回执里两个分母不齐"
    tr, entries, blob, treeent = grab(K_BODY, text, "回执正文的 tree 条目行")
    assert tr == "False", "truncated=%s ⇒ 回执正文被截断，不许引用" % tr
    assert int(blob) == int(remote), "正文 blob 数 ≠ 表头闭合式左值"
    assert int(entries) == int(blob) + int(treeent), "tree 条目 ≠ blob＋tree"
    h40, p40, dirty = grab(K_HEAD, text, "回执表头的 TREE 行")
    prev2 = int(one(K_PREV2, text, "回执表头的「上一片」号"))
    assert prev2 == PREVSLICE - 1, "回执表头说上一片＝%s ≠ 上一片号 %s−1 ⇒ 连号断了" % (prev2, PREVSLICE)
    rsecs = set(re.findall(r"开发日志 §26\.(\d+)", text))
    assert rsecs == {str(PREVSEC)}, "回执里的日志节号＝%s ≠ HEAD 提交说明的 §26.%s ⇒ 两个出处不符" % (
        sorted(rsecs), PREVSEC)
    ctla, ctlb, ctlc, ctld = (grab(K_CTLA, text, "回执里的正对照 A"),
                              grab(K_CTLB, text, "回执里的正对照 B"),
                              grab(K_CTLC, text, "回执里的正对照 C"),
                              grab(K_CTLD, text, "回执里的正对照 D"))
    # 模板那句「四个正对照都能红」是**关于这四个数的断言**，不是引用回执的修辞：每个对照的第一把尺子须非零，
    # C 须判 FAIL。取 `、` 的**第一段**——后段是同一个对照里的第二把尺子，它的 0 是预期值而不是空转证据。
    for what, val in (("A", ctla), ("B", ctlb), ("D", ctld)):
        n = int(re.search(r"\d+", val.split("、")[0]).group())
        assert n >= 1, "正对照 %s 的首位不匹配数＝%d ⇒ 模板那句「四个正对照都能红」成了假话" % (what, n)
    assert ctlc.endswith("判 FAIL"), "正对照 C 的判读＝%r ⇒「四个正对照都能红」那句成了假话" % ctlc
    return dict(remote=remote, parent=parent, na=na, expect=expect, verdict=verdict,
                match=match, nfile=nfile, a=a, m=m, dl=dl, dlden=dlden, dr=dr, drden=drden,
                sha=sha, shaden=shaden, entries=entries, blob=blob, tree=treeent,
                head40=h40, parent40=p40, dirty=dirty, prev2=str(prev2), recsec=sorted(rsecs)[0],
                finger=one(K_FINGER, text, "回执表头 SRC 指纹"),
                rulermd5=one(K_MD5, text, "回执表头的尺子 md5"),
                copies=one(K_COPIES, text, "回执表头登记的尺子份数"),
                boff=one(K_BOFF, text, "回执表头 BODY_OFFSET"),
                ctla=ctla, ctlb=ctlb, ctlc=ctlc, ctld=ctld)


def cmp_triple(off):
    """复跑上一片回执的正文区比对：同偏移须 0，任一侧偏移 +1 须非 0。"""
    rcs = [subprocess.run(["cmp", "-i", "%d:%d" % (o1, o2), os.path.join(REPO, CURL),
                           os.path.join(REPO, RAWR)], capture_output=True).returncode
           for o1, o2 in ((off, 0), (off + 1, 0), (off, 1))]
    assert rcs == [0, 1, 1], "cmp 三件套复跑＝%s（须 [0,1,1]）" % rcs
    assert rb(CURL)[off:] == rb(RAWR), "回执正文区与原始 stdout 不逐字节相同（cmp 却过了？）"
    return rcs


def top_bullets(text):
    labs = RE_BULLET.findall(text)
    segs = []
    for i, let in enumerate(labs):
        start = text.index("- **%s**" % let)
        end = text.index("- **%s**" % labs[i + 1]) if i + 1 < len(labs) else len(text)
        segs.append((let, text[start:end]))
    return labs, segs


def devlog_check(ndef):
    """开发日志的**暂存 blob**（不是工作树）：最后节号＝上一节＋1，节内四条唯一性＋落笔读数由同一把尺子现验。
    `ndef` 由调用方从**唯一一处** `fields_of(HDRC)` 传入；本函数不再自己数一遍（两处各自现取就是两处各自腐烂）。"""
    text = sh("git", "show", ":" + DEVLOG)
    ms = list(RE_SEC.finditer(text))
    secn = ms[-1].group(1)
    assert int(secn) == PREVSEC + 1, "暂存 blob 的最后一节 §26.%s ≠ 上一节 §26.%s＋1 ⇒ 形制约束没落实" % (secn, PREVSEC)
    nxt = text.find("\n### ", ms[-1].end())
    body = text[ms[-1].end():(nxt if nxt != -1 else len(text))]
    labs, segs = top_bullets(body)
    assert len(labs) >= 5 and len(labs) == len(set(labs)), "§26.%s 顶级子条 %d 条且有重名标签" % (secn, len(labs))
    prevlog = "am_s7_verify%d.log" % PREVSLICE
    nd = {"回执名": len([s for _l, s in segs if prevlog in s]),
          "钉住口径": len([s for _l, s in segs if "当前「" in s]),
          "LESSON 锚句": len([s for _l, s in segs if LESSON_NEEDLE in s]),
          "新学锚句": len([s for _l, s in segs if NEWLESSON_NEEDLE in s])}
    bad = {k: v for k, v in nd.items() if v != 1}
    assert not bad, "§26.%s 节内唯一性破了：%s（各须恰好 1）⇒ 分片 %s 的表头生成器会当场红" % (secn, nd, SLICE)
    q = one(r"当前「([^」]+)」", body, "钉住口径原文")
    # 上一节的引用数与子条数：同一把切段尺子现算，并与**已归档自查件**那句读数互证（两 instrument）
    prev_body = text[ms[-2].end():ms[-1].start()]
    prev_bul = len(top_bullets(prev_body)[0])
    prev_cite = len([s for _l, s in top_bullets(prev_body)[1] if prevlog in s])
    arch_bul = int(one(r"§26\.\d+ 顶级子条＝(\d+) 条", rd(ARCHK_RAW), "自查件登记的上一节子条数"))
    assert prev_bul == arch_bul, "上一节顶级子条数：本件现算 %d ≠ 归档自查件 %d ⇒ 切段口径变了" % (prev_bul, arch_bul)
    assert prev_cite == 1, "上一节 §26.%s 里引用 %s 的顶级子条有 %d 条（须 1）⇒ 该节形制与回执表头的声称不符" % (
        PREVSEC, prevlog, prev_cite)
    head_text = sh("git", "show", "HEAD:" + DEVLOG)
    nh = len([m for m in RE_SEC.finditer(head_text) if m.group(1) == secn])
    ns = len([m for m in RE_SEC.finditer(text) if m.group(1) == secn])
    assert (nh, ns) == (0, 1), "§26.%s 在 HEAD 里 %d 次／暂存 blob 里 %d 次" % (secn, nh, ns)
    # 口径 ⑩：全日志（HEAD 版＋暂存版）里「顶格字段」紧跟的数字必须**只等于** FIELDS 声明数
    ndef = len(fields_of(os.path.join(REPO, HDRC)))
    vals = [int(x) for x in re.findall(r"顶格字段(?:行数)?[＝= ]+\*{0,2}(\d+)", head_text + "\n" + text)]
    assert len(vals) >= 5, "「顶格字段」读数只取到 %d 处 ⇒ 这条普查尺子失效" % len(vals)
    wrong = sorted({v for v in vals if v != ndef})
    assert not wrong, "「顶格字段」槽里出现了 %s（本族口径＝FIELDS 声明 %d）⇒ 有读数被填错了槽位" % (wrong, ndef)
    return secn, len(labs), nd, q, nh, ns, prev_cite, arch_bul, prev_bul, len(vals), head_text


def faces_check(modified, pinned_faces):
    """已跟踪的归档面：暂存 blob 行数 vs HEAD 行数，与 numstat 两出处闭合；只允许净增。"""
    got = sorted(os.path.basename(p) for p in modified)
    assert got == sorted(os.path.basename(p) for p in pinned_faces), "修改面 ≠ §26.33 登记的那批：%s vs %s" % (
        got, pinned_faces)
    assert PREVDAILY in modified, "修改面里没有上一片的纪实 %s ⇒ 那句「落地子条由提交之后的填值件追加」没有出处" % PREVDAILY
    assert not sh("git", "-c", "core.quotePath=false", "diff", "--name-only").strip(), \
        "还有未暂存的已跟踪改动 ⇒ 我核对的暂存 blob 不是最终要提交的那一份"
    num = {l.split("\t")[-1]: l.split("\t")[:2] for l in
           sh("git", "-c", "core.quotePath=false", "diff", "--cached", "--numstat").splitlines() if l.strip()}
    out = []
    for rel in modified:
        n_head = len(sh("git", "show", "HEAD:" + rel).splitlines())
        n_staged = len(sh("git", "show", ":" + rel).splitlines())
        add, dele = (int(x) for x in num[rel])
        assert add == n_staged - n_head, "%s：numstat 增 %d ≠ 行数差 %d" % (rel, add, n_staged - n_head)
        assert dele == 0, "%s：相对 HEAD 有 %d 行删除 ⇒ 本轮只登记新段，不改写既有内容" % (rel, dele)
        out.append((rel, n_head, n_staged, add))
    return out


def parse_reg(raw):
    t = RE_TALLY.search(raw)
    heads = dict(RE_FINGER.findall(raw))
    assert set(heads) == {"START", "END"} and heads["START"] == heads["END"], "SRCFINGER 起≠止"
    tm = RE_TREE.search(raw)
    dates = dict(RE_DATE.findall(raw))
    elines = RE_ELINES.findall(raw)
    assert len(elines) == 2, "两条红的 E 行实取 %d 条 ⇒ 判据形状已变" % len(elines)
    return dict(tally=t.group(0), nf=t.group(1), npass=t.group(2), nskip=t.group(3),
                wall=t.group(4) + "s", head=tm.group(1), dirty=tm.group(2), fs=heads["START"],
                d1=dates["START"], d2=dates["END"], e1=elines[0].strip(), e2=elines[1].strip(),
                rc=one(r"^rc=(\d+)$", raw, "回归件 rc 行"))


def finger_both():
    """口径 ⑧：两条 locale 各现算。参与闭合的那条**必须不钉 LC_ALL**，钉了的那条只断言"不同"。"""
    assert "LC_ALL" not in os.environ, "本进程环境里有 LC_ALL ⇒ 那条「默认 locale」读数就不再是默认值"
    d = run_bash(FINGER_CMD).strip()
    c = subprocess.run(["bash", "-c", FINGER_CMD], cwd=REPO, capture_output=True, text=True,
                       env=dict(os.environ, LC_ALL="C")).stdout.strip()
    for tag, v in (("默认 locale", d), ("LC_ALL=C", c)):
        assert re.fullmatch(r"[0-9a-f]{32}", v), "%s 的指纹读数不是 32 位十六进制：%r" % (tag, v)
    assert d != c, "两条 locale 现算同值 ⇒ 本轮登记的耦合已不存在，那条口径要改写"
    return d, c


def ruler_family(want):
    """尺子＝按 **md5 分组**数，不是按 glob 形状数；两个读数都印，差额必须有名字。"""
    fam = {}
    shape = []
    for p in glob.glob(os.path.join(REPO, "docs/evidence/*/am_s7_verify*.py")):
        if os.path.basename(p) == "am_s7_verify%d.py" % SLICE:
            continue  # 本片那把此刻还不存在；存在了就不该再叫「本片要抄第 N 份」
        fam.setdefault(hashlib.md5(open(p, "rb").read()).hexdigest(), []).append(p)
        shape.append(p)
    big = [v for v in fam.values() if len(v) >= 2]
    assert len(big) == 1, "逐字节副本族有 %d 组（须 1）⇒ 「同一把尺子量多片」的口径变了" % len(big)
    members = big[0]
    assert len(members) == want, "尺子族现数 %d 份（上一片回执登记 %d 份）⇒ 有人另起过一把尺子或抄丢了" % (
        len(members), want)
    assert os.path.basename(RULER) in [os.path.basename(x) for x in members], "上一片那把尺子不在族里 ⇒ 它不是逐字节副本"
    stray = sorted(os.path.basename(p) for p in shape if p not in members)
    assert stray, "同一条 glob 的形状与 md5 族完全重合 ⇒ 那句「差额有名字」本轮不成立，要改写"
    return len(members), len(shape), md5(RULER, 12), stray


def gpu_fallback():
    """本轮新学：NVML 那把尺子坏了 ⇒ 红线改由 /proc 的 argv[0] 整词匹配承担，带分母与一正一负对照。"""
    nv = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                        capture_output=True, text=True)
    assert nv.returncode != 0, "nvidia-smi 本轮又能用了（rc=%d）⇒ 「普查仪器坏了」那条新学已腐烂，要改写" % nv.returncode
    why = (nv.stdout + nv.stderr).strip().splitlines()
    assert why and "NVML" in why[0], "nvidia-smi 的失败原因换了：%s" % (why[:1],)
    drv = rd("/sys/module/nvidia/version").strip()
    # 用户态版本要**两个出处**：dpkg 状态 `ii` 的 amd64 包版本 ＋ 加载器**真正解析到**的那个文件（符号链接的目标）。
    # 第一版按 `libnvidia-compute:amd64` 匹配——真实行带系列号（`libnvidia-compute-580:amd64`）⇒ 取到 None；
    # 第二版改成 glob 目录下所有 `libnvidia-ml.so.<版本号>` 文件名——那里还留着**已卸载包**的残留文件
    # （535.288.01，dpkg 状态 `rc` 而非 `ii`）⇒ 两个出处不相等，又红。
    # 结论：盘上有哪些文件不是「此刻哪份在用」的凭据；`libnvidia-ml.so.1` 的 realpath 才是，且必须与 dpkg 的 ii 集同值。
    q = subprocess.run(["dpkg-query", "-W",
                        "-f=${Package}|${Architecture}|${db:Status-Abbrev}|${Version}\n",
                        "libnvidia-compute*"], capture_output=True, text=True)
    assert q.returncode == 0, "dpkg-query 失败（rc=%d）：%s ⇒ 用户态版本没了出处" % (q.returncode, q.stderr.strip()[:120])
    pkgs = set()
    for line in q.stdout.splitlines():
        cols = line.split("|")
        if len(cols) != 4 or cols[1] != "amd64" or cols[2] != "ii ":
            continue
        if not re.fullmatch(r"libnvidia-compute-\d+", cols[0]):
            continue
        v = cols[3].split("-")[0]
        assert re.fullmatch(r"\d+(\.\d+)+", v), "dpkg 的 ii 行版本号形状不认识：%s" % line
        pkgs.add(v)
    pkgs = sorted(pkgs)
    assert len(pkgs) == 1, ("dpkg 里 amd64／状态 ii 的 libnvidia-compute 有 %d 个版本（须恰好 1 个）：%s ⇒ "
                            "「内核模块与用户态库是两个版本」这句要重取" % (len(pkgs), pkgs))
    used = os.path.basename(os.path.realpath("/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"))
    mv = re.fullmatch(r"libnvidia-ml\.so\.(\d+\.\d+\.\d+)", used)
    assert mv, "加载器解析到的 NVML 文件名不是三段版本号（%s）⇒ 用户态版本没凭据" % used
    lib = mv.group(1)
    assert pkgs[0] == lib, "dpkg 的 ii 版本 %s ≠ 加载器解析到的 %s ⇒ 两个出处不同值，根因句要重取" % (pkgs, lib)
    # 盘上的**全部**带三段版本号文件名（含已卸载包的残留）只作旁证：在用版本必须在其中，残留不许被当成读数出处。
    on_disk = sorted({m.group(1) for m in (re.fullmatch(r"libnvidia-ml\.so\.(\d+\.\d+\.\d+)", os.path.basename(x))
                                           for x in glob.glob("/usr/lib/*/libnvidia-ml.so.*")) if m})
    # 空集判定必须在 `lib in on_disk` **之前**：后者在空集上必然红，会把「glob 出处为空」永久遮住（不可满足判据）。
    assert len(on_disk) >= 1, "盘上没有任何带三段版本号的 NVML 文件 ⇒ glob 出处为空"
    assert lib in on_disk, "在用版本 %s 不在盘上文件集 %s ⇒ realpath 与 glob 自相矛盾" % (lib, on_disk)
    assert lib.split(".")[:2] != drv.split(".")[:2], \
        "内核模块 %s 与用户态库 %s 的前两段相同 ⇒ 「两个版本不一致」这句根因本轮不成立，要改写" % (drv, lib)
    pids = [d for d in os.listdir("/proc") if d.isdigit()]
    me, myparent = os.getpid(), os.getppid()
    mine, loose, loose0 = [], [], []
    for d in pids:
        if int(d) in (me, myparent):
            continue
        try:
            blob = open("/proc/%s/cmdline" % d, "rb").read()
        except OSError:
            continue
        if not blob:
            continue
        a0 = blob.split(b"\0")[0].decode("utf-8", "replace")
        if match_argv0(blob):
            mine.append(d)
        if INTERP in blob.decode("utf-8", "replace").replace("\x00", " "):
            loose.append(d)
            loose0.append(a0)
    # 两条对照：正＝合成 argv[0] 必须是 1；负＝命令行里**含**解释器路径但 argv[0] 不是它，必须 0
    neg_blob = b"/bin/bash\0-c\0grep " + INTERP.encode() + b"\0"
    pos = match_argv0((INTERP + "\0x.py").encode())
    neg = match_argv0(neg_blob)
    assert pos is True and neg is False, "argv[0] 匹配式的两条对照失效（正 %s／负 %s）" % (pos, neg)
    loose_neg = neg_blob.decode("utf-8", "replace").replace("\x00", " ")
    assert INTERP in loose_neg and match_argv0(loose_neg.encode()) is False, "松散口径的负对照形状失效"
    # 红线本身必须是**闸**，不是读数：数到我的解释器进程就得拒绝生成信息（不是印一个非零数继续走）。
    assert not mine, ("argv[0] 整词匹配数到我的解释器进程 %s（分母＝%d 个 PID）⇒ 「GPU 全程未碰」那条红线不成立，"
                      "拒绝生成信息" % (mine, len(pids)))
    upg = len([d for d in pids if os.path.basename(argv0(d) or "") in ("unattended-upgr", "unattended-upgrade")])
    return dict(nvrc=nv.returncode, nvwhy=why[0], drv=drv, lib=lib, total=len(pids), mine=len(mine),
                stale="／".join(sorted(set(on_disk) - {lib})) or "无",
                loose=len(loose), loose0="、".join(sorted({os.path.basename(x) or "无" for x in loose0})) or "无",
                upg=upg)


def ctl_readings():
    """三面归档控件件的读数＋判读行＋继承常量，全部现取；超时不许非零（超时＝环境故障，不是判据）。"""
    msg = rd(MSGCTL)
    nc, nto = grab(r"CASES 声明＝(\d+) 条｜本件自己的锚点普查＝\d+ 条｜超时（环境故障，不判红也不判过）＝(\d+) 条",
                   msg, "提交说明控件面的声明行")
    nun = int(one(r"本轮造不出红路的闸门＝(\d+) 条", msg, "提交说明控件面 UNCOVERED"))
    hdr = rd(HDRCTL)
    hred, hgreen = grab(r"（红路 (\d+) 条＋绿路 (\d+) 条", hdr, "表头控件面预检行的红／绿")
    hself, hselfn = grab(r"本面自查＝(\d+) 条（(.+?)；每条都两个方向各验一次", hdr, "表头控件面自查条数与名字")
    hunc = int(one(r"本轮造不出红路的闸门＝(\d+) 条", hdr, "表头控件面 UNCOVERED"))
    hto = int(one(r"超时（环境故障，不判红也不判过）＝(\d+) 条", hdr, "表头控件面超时"))
    af = rd(AFCTL)
    afcase, afgreen = grab(r"CASES 声明＝(\d+) 条（全部红路）｜保持绿＝(\d+) 条", af, "只读仪器控件面声明行")
    afunc = int(one(r"本轮造不出红路的闸门＝(\d+) 条", af, "只读仪器控件面 UNCOVERED"))
    ato = int(one(r"超时（环境故障，不判红也不判过）＝(\d+) 条", af, "只读仪器控件面超时"))
    assert int(nto) == hto == ato == 0, "三面里有超时（%s／%d／%d）⇒ 不能当全过面引用" % (nto, hto, ato)
    # 表头控件面的「保持绿＝3 条」不是表里的 3 行：那件形制是 **基线单独打印、不进表**（GREENS[0]），
    # 表里只有 `GREENS[1:]` 两条 〔落笔第一版把这条写成了「G 行数＝声明数」，当场红——红得对，但它拦的是我自己写错的等式〕。
    # 所以闭合必须两侧现取：表内 G 行 ＋ 基线那一条 ＝ 声明的绿路数。
    hbase = int(one(r"｜基线＝(\d+) 条（不进表", hdr, "表头控件面基线条数声明"))
    grow = len([l for l in hdr.splitlines() if l.startswith("  G")])
    glab = len([l for l in hdr.splitlines() if "（须保持绿）" in l])
    bline = len([l for l in hdr.splitlines() if l.startswith("基线（未扰动副本")])
    assert grow == glab, "表头控件面里表内 G 行 %d 条 ≠ 带「（须保持绿）」标签的 %d 条 ⇒ 标签漏到别的行或行首形制变了" % (
        grow, glab)
    assert bline == hbase, "表头控件面基线行现数 %d 条 ≠ 那句声明 %d 条 ⇒ 「基线不进表」的口径变了" % (bline, hbase)
    assert int(hgreen) == grow + hbase, "表头控件面绿路声明 %s ≠ 表内 %d＋基线 %d ⇒ 绿路账没有第二出处" % (
        hgreen, grow, hbase)
    verd = [one(r"^(CONTROL_VERDICT = .*)$", x, "控件面判读行") for x in (msg, hdr, af)]
    assert len(set(verd)) == 1, "三面判读行不一致：%s" % set(verd)
    src = rd(HDRCTL_PY)
    return dict(ncase=int(nc), nto=int(nto), nun=nun, hred=int(hred), hgreen=int(hgreen), hself=int(hself),
                hselfn=hselfn, hunc=hunc, afcase=int(afcase), afgreen=int(afgreen), afunc=int(afunc),
                verd=verd[0], subto=int(one(r"^SUB_TIMEOUT = (\d+)$", src, "子运行上限常量")),
                envstop=int(one(r"^ENV_FAIL_STOP = (\d+)$", src, "环境故障累计常量")),
                window=int(one(r"行号落在该句在\*\*副本\*\*里的 ±(\d+) 行窗口", src, "raise 行号窗口常量")),
                msgb=b(MSGCTL), msgm=md5(MSGCTL, 12), hdrb=b(HDRCTL), hdrm=md5(HDRCTL, 12),
                afb=b(AFCTL), afm=md5(AFCTL, 12))


def archk_readings():
    """归档自查件的档案读数（不复跑它当常备尺子——那是 archk_legscope 的活）＋逐行正对照数。"""
    rep = rd(ARCHK_RAW)
    claims, miss, mden = grab(r"声称条数＝(\d+)｜缺失＝(\d+)/(\d+)", rep, "自查件的声称与缺失")
    lines = len([l for l in rep.splitlines() if l.startswith("  正对照：")])
    assert int(miss) == 0 and int(claims) == int(mden), "自查件缺失非零或分母不齐：%s／%s／%s" % (claims, miss, mden)
    assert len([l for l in ["  正对照：合成的一条"] if l.startswith("  正对照：")]) == 1, "正对照逐行定位器对合成样本失明"
    assert len([l for l in ["正对照：换了形状"] if l.startswith("  正对照：")]) == 0, "正对照定位器把非顶格样本也算进来了"
    carry = re.findall(r"承载：(\S+?)＝(\d+) 条（其中独有 (\d+) 条）", rep)
    assert len(carry) == 5, "自查件的承载行实取 %d 行（须 5，五张档案面）" % len(carry)
    top = max(carry, key=lambda x: int(x[1]))
    assert len([x for x in carry if int(x[1]) == int(top[1])]) == 1, "承重最多的一面不唯一 ⇒ 那句「最多的一面」没有所指"
    return dict(b=b(ARCHK_RAW), md5=md5(ARCHK_RAW, 12), eb=b(ARCHK_ERR), claims=int(claims), ctl=lines,
                carry=[(os.path.basename(p), int(n), int(o)) for p, n, o in carry],
                maxface=os.path.basename(top[0]), maxn=int(top[1]), maxo=int(top[2]),
                verd=one(r"^(ARCHIVE_VERDICT = .*)$", rep, "自查件判读行"),
                pyb=b(ARCHPY), pym=md5(ARCHPY, 12), a1b=b(ARCHK_A1_ERR))


def rerun_local(relpath, gate, minutes):
    """口径 ⑤ 的公共腿：**实跑**上一片的局部仪器（两件都只读／只打 stdout，不写仓库），
    要求它红在 `gate` 点名的那一句上，且 stderr 与已归档的复跑现场**逐字节相同**。
    超时＝环境故障，既不判红也不判过，直接拒绝生成信息。"""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    try:
        r = subprocess.run([INTERP, os.path.join(REPO, relpath)], cwd=REPO, capture_output=True,
                           text=True, timeout=minutes * 60, env=env)
    except subprocess.TimeoutExpired:
        raise SystemExit("%s %d 分钟没跑完 ⇒ 那是环境故障，不判红也不判过，本件拒绝生成信息" % (relpath, minutes))
    assert r.returncode == 1, "复跑 %s rc=%s（须 1＝它红在局部锚句）⇒ 它本轮仍全绿？那条「局部仪器」的口径要改写" % (
        relpath, r.returncode)
    last = r.stderr.rstrip("\n").splitlines()[-1]
    for needle in gate:
        assert needle in last, "复跑红在了别的断言上（缺「%s」）：%r ⇒ 「它红在自己那一轮的那条闸」这句不成立" % (
            needle, last[:200])
    return r, last


def archk_legscope():
    """局部仪器①＝上一片的归档自查件：它数的是**磁盘上最后一节**，LESSON 锚句是它那一轮写死的短语。"""
    r, last = rerun_local(ARCHPY, ("LESSON 锚句", "须恰好 1", "分片 %s 的生成器" % SLICE), 5)
    assert r.stdout == "", "复跑却有 stdout %d 字节 ⇒ 它已不是纯红件，现场件名与内容都要重取" % len(r.stdout.encode())
    assert rb(ARCHK_PIN_ERR) == r.stderr.encode(), \
        "本轮实跑的 stderr 与已归档复跑现场不逐字节相同 ⇒ 两次跑的不是同一条代码路径"
    assert rb(ARCHK_PIN_OUT) == r.stdout.encode(), "本轮实跑的 stdout 与已归档复跑现场不逐字节相同"
    assert b(ARCHK_PIN_OUT) == 0, "归档的复跑现场 stdout 非 0 B（%d）⇒ 那句「纯红件」要改写" % b(ARCHK_PIN_OUT)
    return dict(rc=str(r.returncode), last=last, pinb=str(b(ARCHK_PIN_ERR)), pinm=md5(ARCHK_PIN_ERR, 12),
                pinout=str(b(ARCHK_PIN_OUT)), sec=one(r"§26\.(\d+)", last, "复跑那句点名的节"),
                pinname=os.path.relpath(ARCHK_PIN_ERR, REPO)[:-4])


def msg_legscope():
    """局部仪器②＝上一片的提交说明生成器：它把 SLICE 取成 HEAD 片号＋1、把 PREVDIR 取成**它那一轮**glob 命中的目录，
    本轮 HEAD 已前移 ⇒ 它红在「上一片回执与本件同目录」那条跨日期登记闸上（落笔第一版预告它会红在桶／普查闸，
    实跑证明红点不预告 ⇒ 现场的末行必须逐字节取，不许按预期写）。"""
    r, last = rerun_local(PREVMSG, ("上一片回执与本件同目录", "跨日期登记要改"), 10)
    assert r.stdout == "", "复跑上一片信息生成器却有 stdout %d 字节 ⇒ 它这次没有早退，现场内容要重取" % len(r.stdout.encode())
    assert rb(MSG_PIN_ERR) == r.stderr.encode(), \
        "本轮实跑的 stderr 与已归档的复跑现场不逐字节相同 ⇒ 两次跑的不是同一条代码路径"
    assert rb(MSG_PIN_OUT) == r.stdout.encode() and b(MSG_PIN_OUT) == 0, \
        "归档的复跑现场 stdout 不是 0 B（%d）⇒ 那句「红在跨日期闸、没打出任何信息」要改写" % b(MSG_PIN_OUT)
    raised = one(r'File "([^"]+)", line (\d+), in', r.stderr, "复跑那行 trace 的文件与行号")
    assert os.path.basename(raised[0]) == os.path.basename(PREVMSG), \
        "复跑红点落在 %s 而非上一片生成器 ⇒ 那句归因要改写" % raised[0]
    return dict(rc=str(r.returncode), last=last, pinb=str(b(MSG_PIN_ERR)), pinm=md5(MSG_PIN_ERR, 12),
                pinout=str(b(MSG_PIN_OUT)), pinname=os.path.relpath(MSG_PIN_ERR, REPO)[:-4],
                rline=raised[1], rfile=os.path.basename(raised[0]))


def docs_specimens():
    """本片写入件留下的两条红灯现场（自指计数＋守卫打掉自己）：字节与末行现取，消息形状各自要求命中。"""
    t1 = rb(T1_ERR).decode("utf-8", "replace").rstrip("\n").splitlines()[-1]
    t2 = rb(T2_ERR).decode("utf-8", "replace").rstrip("\n").splitlines()[-1]
    assert "落笔现数" in t1 and "本节的闭合式写的是" in t1, "attempt1 的末行不像那条闭合式闸门：%r" % t1[:160]
    assert "有仪器正在跑" in t2 and os.path.basename(WRITER) in t2, "attempt2 的末行不像那条守卫：%r" % t2[:160]
    nums = grab(r"落笔现数＝(\d+)＋(\d+)＝(\d+) 处（本节的闭合式写的是 (\d+)）", t1, "attempt1 句里的三个数与写死的分母")
    assert int(nums[0]) + int(nums[1]) == int(nums[2]) and int(nums[2]) != int(nums[3]), \
        "attempt1 那句不闭合：现数 %s＋%s＝%s vs 写死 %s" % nums
    return dict(t1b=b(T1_ERR), t1m=md5(T1_ERR, 12), t1last=t1, t1o=b(T1_OUT),
                nums="＋".join(nums[:3]) + "，写死的那条＝" + nums[3],
                t2b=b(T2_ERR), t2m=md5(T2_ERR, 12), t2last=t2,
                preb=b(WPRE), prem=md5(WPRE, 12), pee=b(WPRE_ERR),
                wb=b(WWRITE), wm=md5(WWRITE, 12), we=b(WWRITE_ERR), pyb=b(WRITER), pym=md5(WRITER, 12))


def writer_verdicts():
    w = rd(WWRITE)
    v1 = one(r"^(WRITER_VERDICT = .*)$", w, "写入件判读行")
    v2 = one(r"^(FILLER_VERDICT = .*)$", w, "写入件的填值判读行")
    assert v1.endswith("LANDED, UNIQUENESS RECOUNTED ON DISK"), v1
    assert v2.endswith("DERIVED, STRUCTURE GATES PASS"), v2
    app = dict(re.findall(r"追加 (\S+?)：(\d+) B", w))
    assert len(app) == 4, "写入件登记的追加面实取 %d 张（须 4 张纯追加面）：%s" % (len(app), sorted(app))
    ins = one(r"插入 总览：条目插在 `## 7\. 文档索引` 之前（偏移 (\d+)）", w, "总览插入偏移")
    wnu, wnm, wnt = grab(r"待入库＝未跟踪 (\d+)＋修改 (\d+)＝(\d+) 处", w, "写入件 stdout 的落笔现数行")
    assert int(wnu) + int(wnm) == int(wnt), "写入件那句落笔现数不闭合：%s＋%s≠%s" % (wnu, wnm, wnt)
    return dict(v1=v1, v2=v2, app=app, ins=ins, wnu=int(wnu), wnm=int(wnm), wnt=int(wnt))


def uall_check(added, modified, pin_basenames):
    """口径 ⑨ 的**本轮真实形态**：`-uall` 的差别发生在落笔那一侧（那时整目录未跟踪），
    而生成信息时那批面已全部暂存 ⇒ 未跟踪侧对落笔清单必须是空交集，不是照抄一句「未跟踪＝入库清单」。
    〔落笔第一版写成 `当前未跟踪 == 暂存 A 清单`，当场红——`git status` 里暂存过的新文件不再显示为未跟踪，
    那条等式**按构造不可能成立**＝本轮第三次「判据自己红不了」的形状，红的是我写的闸而不是档案。〕"""
    raw = sh("git", "-c", "core.quotePath=false", "status", "--porcelain", "--untracked-files=all")
    un = sorted(l[3:] for l in raw.splitlines() if l.startswith("?? "))
    mo = sorted(l[3:] for l in raw.splitlines() if l.startswith(" M "))
    # 同一族形状：暂存过的修改面在 `git status` 里显示为 `M `（索引列），不是 ` M`（工作树列）。
    # 落笔第一版把 ` M` 行拿去等暂存 M 清单＝第二条按构造红不了的等式（本件在上一条就红了，所以它没机会自己暴露）。
    assert not mo, "工作树里有 %d 张**未暂存**的修改面（%s）⇒ 无未暂存漂移那句不成立" % (len(mo), mo)
    assert all(not p.endswith("/") for p in un), "未跟踪清单里有以 / 结尾的折叠目录行 ⇒ `-uall` 口径失效"
    unp = sorted(set(os.path.basename(p) for p in un) & set(pin_basenames))
    assert len(set(pin_basenames)) == len(pin_basenames), "落笔清单里有重名面（%d 项／%d 个名字）⇒ 按基名比对不成立" % (
        len(pin_basenames), len(set(pin_basenames)))
    assert not unp, "未跟踪面里有 %d 张属于 §26.33 的落笔清单（%s）⇒ 那张面没被暂存" % (len(unp), unp)
    gone = sorted(set(pin_basenames) - set(os.path.basename(p) for p in added))
    assert not gone, "§26.33 落笔登记的 %d 张面里有 %d 张不在暂存集（%s）⇒ 入库清单已腐烂" % (
        len(pin_basenames), len(gone), gone)
    folded = [l[3:] for l in sh("git", "-c", "core.quotePath=false", "status", "--porcelain").splitlines()
              if l.startswith("?? ") and l[3:].endswith("/")]
    sm = sorted(l[3:] for l in raw.splitlines() if l.startswith("M "))
    assert sm == modified, "`git status` 索引列的 M 面与暂存 M 清单不齐（现数 %d／清单 %d）" % (len(sm), len(modified))
    return dict(un=len(un), mo=len(mo), sm=len(sm), unp=len(unp), pin=len(pin_basenames),
                fold=len(folded), foldnames="、".join(sorted(folded)) or "无")


def selfctl_check():
    """本件对**自己的干跑控件面**只取它源码里的两样东西（绝不读它的 stdout——那是循环，见信息里诚实登记 ⑤）：
    ① 它声明的判据锚点句必须**恰好落在 1 行**（落 0 行＝那条扰动会被判空转、落 2 行＝±3 行窗口会指错闸门）；
    ② 它声明的基线可运行标记句也必须恰好 1 行，并把该行的字节数交给控件面做**逐字节相等**判据
      （分片 30 实测：标记句写窄了会让未扰动副本在打印之前就红，整轮正对照作废）。"""
    src = rd(CTLSELF)
    un = one(r'^SELF_UNUSED_NEEDLE = "(.+)"$', src, "控件面自己声明的判据锚点句")
    mark = one(r'^SELF_BASE_MARK = "(.+)"$', src, "控件面自己声明的基线可运行标记句")
    nun = len([l for l in src.splitlines() if un in l])
    nmark = len([l for l in src.splitlines() if mark in l])
    assert nun == 1, "判据锚点句「%s」在控件面源码里命中 %d 行（须 1）⇒ 那条扰动会被判空转或 ±3 窗口指错闸门" % (un, nun)
    assert nmark == 1, "基线可运行标记「%s」在控件面源码里命中 %d 行（须 1）⇒ 副本的「走到打印分支」判定不可用" % (mark, nmark)
    return dict(un=un, mark=mark, markb=str(len([l for l in src.splitlines() if mark in l][0].encode())),
                b=b(CTLSELF))


def build():
    rows, added, modified = staged()
    na, nm = len(added), len(modified)
    hits = bucketize(added)
    raw_ss, ss = shortstat_check()
    g = gate_curl(rd(CURL))
    off = int(g["boff"])
    rcs = cmp_triple(off)
    fields = fields_of(os.path.join(REPO, HDRC))
    afld = "/".join(grab(K_NDEF, rd(AFOUT), "只读仪器那句顶格字段行数"))
    assert str(len(fields)) == afld.split("/")[0] == afld.split("/")[1], \
        "顶格字段：FIELDS 声明 %d ≠ 只读仪器那句 %s" % (len(fields), afld)
    assert b(CURL) == off + b(RAWR), "回执总长 ≠ BODY_OFFSET＋原始件字节数 ⇒ 正文区边界不闭合"
    aftriple = "/".join(grab(K_TRIPLE, rd(AFOUT), "只读仪器的 cmp 三联"))
    assert aftriple == "0/1/1", "只读仪器那句三联＝%s（须 0/1/1）⇒ 那张回执不能当全绿引用" % aftriple
    secn, nbul, nd, quote, nh, ns, prev_cite, arch_bul, prev_bul, ndef_vals, head_dev = devlog_check(len(fields))
    staged_devlog = sh("git", "show", ":" + DEVLOG)
    pinrows, (pnu, pnm, ptot), pnface, pinfaces = arch_pin(staged_devlog[staged_devlog.rindex("### 26.%s" % secn):])
    assert pnface == len(pinfaces) == nm, "§26.33 登记的修改面 %s 项／声称 %s 件 ≠ 暂存 M %d 件" % (pinfaces, pnface, nm)
    assert pnm == nm, "§26.33 落笔登记的修改件数 %d ≠ 本轮暂存 M %d ⇒ 那句闭合式的左边已腐烂" % (pnm, nm)
    faces = faces_check(modified, pinfaces)
    reg = parse_reg(rd(REG))
    fd, fc = finger_both()
    hexes = set(re.findall(r"\b[0-9a-f]{32}\b", head_dev))
    assert fd in hexes and fc in hexes, "两条 locale 指纹里有 §26.32（HEAD 那版）未登记的值（默认 %s／C %s）" % (fd, fc)
    assert fd == reg["fs"] == g["finger"], "指纹四路不闭合：现算 %s／回归件 %s／上一片回执 %s" % (fd, reg["fs"], g["finger"])
    ncopies, nshape, rmd5, stray = ruler_family(int(g["copies"]))
    assert rmd5 == g["rulermd5"], "尺子 md5：现取 %s ≠ 回执登记 %s" % (rmd5, g["rulermd5"])
    gpu = gpu_fallback()
    ctl = ctl_readings()
    arch = archk_readings()
    scope = archk_legscope()
    mscope = msg_legscope()
    spec = docs_specimens()
    wv = writer_verdicts()
    u = uall_check(added, modified, sorted(bn for k in BKEY_PIN for bn in hits[k]))
    assert int(wv["wnu"]) == pnu == u["pin"], "落笔未跟踪数三处不闭合：写入件 stdout %s／§26.33 %d／前 12 桶现数 %d" % (
        wv["wnu"], pnu, u["pin"])
    assert int(wv["wnt"]) == ptot, "落笔闭合式两边不闭合：写入件 stdout %s vs §26.33 %s" % (wv["wnt"], ptot)
    sc = selfctl_check()
    selfb = md5(os.path.join(HERE, SELF), 12)
    # 模板里那句「src／tests 本轮 @@NCODE@@ 件改动」后面紧跟**写死的**「现数 0 条 ⇒ 纯归档腿有凭据」⇒ 这个读数必须过闸，
    # 不然作用域一挪（比如改成 docs）数会变大而那句话仍是 0 条＝一句印出来的谎。落笔第一版只把它当读数，没当断言。
    ncode = len(sh(*NCODE_ARGV).splitlines())
    assert ncode == 0, "src／tests 现数 %d 件改动 ⇒「纯归档腿」与「没有可比较的新测试」两句都成了假话" % ncode
    V = dict(
        SLICE=str(SLICE), PREVSLICE=str(PREVSLICE), PREV2=g["prev2"], NEXTSLICE=str(SLICE + 1),
        SECN=secn, PREVSEC=str(PREVSEC), RECSEC=g["recsec"],
        HEADD7=sh("git", "rev-parse", "HEAD").strip()[:7],
        PREVDIRD=PREVDIR + "/", EDIRD=E + "/", SELFNAME=SELF, SELFM=selfb, SELFCTL=SELFCTL,
        CURL=CURL, RULER=os.path.basename(RULER), HDRC=HDRC, RAWRN=os.path.basename(RAWR),
        RAWRB=str(b(RAWR)), RAWRM=md5(RAWR, 12), RAWRE=str(b(RAWR_ERR)),
        BOFF=str(off), TRIPLE=" / ".join(str(x) for x in rcs), CLIB=str(b(CURL)), CLINES=str(len(rb(CURL).splitlines())),
        CLM=md5(CURL, 12),
        VERD=g["verdict"], REMOTE=g["remote"], PARENTB=g["parent"], NA_PREV=g["na"], EXPECT=g["expect"],
        HEAD40=g["head40"], P40=g["parent40"], DIRTY=g["dirty"], ENTRIES=g["entries"], BLOB=g["blob"],
        TREEC=g["tree"], DL=g["dl"], DLDEN=g["dlden"], DR=g["dr"], DRDEN=g["drden"], SHA=g["sha"],
        SHADEN=g["shaden"], MATCH=g["match"], NFILE=g["nfile"], GA=g["a"], GM=g["m"],
        CTLA=g["ctla"], CTLB=g["ctlb"], CTLC=g["ctlc"], CTLD=g["ctld"],
        RFINGER=g["finger"], RULERMD5=rmd5, COPIES=str(ncopies), SHAPE=str(nshape), STRAY="、".join(stray),
        NEXTCOPY=str(ncopies + 1), NDEF=str(len(fields)), AFNDEF=afld, NDEFVALS=str(ndef_vals),
        AFTRIPLE=aftriple, AFVERDICT=one(r"^(HDR_AFTER_VERDICT = .*)$", rd(AFOUT), "只读仪器判读行"),
        AFOUTB=str(b(AFOUT)), AFWRITEB=str(b(HDR_WR)), HDRPREB=str(b(HDR_PRE)),
        SS_RAW=raw_ss, SS_F=str(ss[0]), SS_I=str(ss[1]), SS_D=str(ss[2]),
        NA=str(na), NM=str(nm), NTOT=str(na + nm),
        NCODE=str(ncode), NCODE_CMD=NCODE_CMD,
        PIN_EQ="＋".join(str(c) for _l, c in pinrows), PIN_SUM=str(pnu), PIN_NU=str(pnu), PIN_NM=str(pnm),
        PIN_TOT=str(ptot), PINFACES="／".join(pinfaces),
        BKTPIN="／".join("%s＝%d" % (k, len(hits[k])) for k in BKEY_PIN),
        BKTSELF="／".join("%s＝%d" % (k, len(hits[k])) for k in BKEY_SELF),
        UNSUM=str(sum(len(hits[k]) for k in BKEY_PIN)), UNSUMSELF=str(sum(len(hits[k]) for k in BKEY_SELF)),
        NBKT=str(len(BUCKETS)),
        FACE="；".join("%s %d→%d（＋%d）" % (os.path.basename(rel), h, s, a) for rel, h, s, a in faces),
        FACE_TOT=str(sum(a for _r, _h, _s, a in faces)),
        QUOTE=quote, NBUL=str(nbul), ND="／".join("%s＝%d" % (k, v) for k, v in nd.items()),
        NH=str(nh), NS=str(ns), PREVCITE=str(prev_cite), ARCBUL=str(arch_bul), PREVBUL=str(prev_bul),
        PREVLOGNAME="am_s7_verify%d.log" % PREVSLICE,
        REGPATH=REG, REG=reg["tally"], REGNF=reg["nf"], REGNP=reg["npass"], REGNS=reg["nskip"],
        REGW=reg["wall"], REGRC=reg["rc"], REGH=reg["head"], REGDIRTY=reg["dirty"], REGD1=reg["d1"],
        REGD2=reg["d2"], REGE1=reg["e1"], REGE2=reg["e2"],
        XF=run_bash("grep -ic 'xfail\\|xpass' %s" % REG).strip(),
        FDEFD=fd, FCNT=fc, HEXCOUNT=str(len(hexes)),
        MSGCTLB=str(ctl["msgb"]), MSGCTLM=ctl["msgm"], MSGNCASE=str(ctl["ncase"]), MSGNUNC=str(ctl["nun"]),
        HDRCTLB=str(ctl["hdrb"]), HDRCTLM=ctl["hdrm"], HDRRED=str(ctl["hred"]), HDRGREEN=str(ctl["hgreen"]),
        HDRSELF=str(ctl["hself"]), HDRSELFN=ctl["hselfn"], HDRUNC=str(ctl["hunc"]),
        AFCTLB=str(ctl["afb"]), AFCTLM=ctl["afm"], AFNCASE=str(ctl["afcase"]), AFGREEN=str(ctl["afgreen"]),
        AFUNC=str(ctl["afunc"]), CTLVERD=ctl["verd"], SUBTO=str(ctl["subto"]), ENVSTOP=str(ctl["envstop"]),
        WINDOW=str(ctl["window"]),
        ARCHPYNAME=os.path.basename(ARCHPY), ARCHPYB=str(arch["pyb"]), ARCHPYM=arch["pym"], ARCHB=str(arch["b"]),
        ARCHM=arch["md5"], ARCHEB=str(arch["eb"]), ARCHCLAIMS=str(arch["claims"]), ARCHCTL=str(arch["ctl"]),
        ARCHVERD=arch["verd"], ARCHA1B=str(arch["a1b"]),
        ARCHCARRY="／".join("%s＝%d（独 %d）" % c for c in arch["carry"]),
        ARCHMAX="%s＝%d 条（独 %d）" % (arch["maxface"], arch["maxn"], arch["maxo"]),
        SCOPE_RC=scope["rc"], SCOPE_LAST=scope["last"], SCOPE_PINB=scope["pinb"], SCOPE_PINM=scope["pinm"],
        SCOPE_PINOUT=scope["pinout"], SCOPE_SEC=scope["sec"], SCOPE_PINNAME=scope["pinname"],
        T1B=str(spec["t1b"]), T1M=spec["t1m"], T1LAST=spec["t1last"], T1O=str(spec["t1o"]), T1NUMS=spec["nums"],
        T2B=str(spec["t2b"]), T2M=spec["t2m"], T2LAST=spec["t2last"],
        WPREB=str(spec["preb"]), WPREM=spec["prem"], WFREE=str(spec["pee"]), WWB=str(spec["wb"]),
        WWM=spec["wm"], WWE=str(spec["we"]), WPYB=str(spec["pyb"]), WPYM=spec["pym"],
        WRITERNAME=os.path.basename(WRITER), WV1=wv["v1"], WV2=wv["v2"],
        WAPP="／".join("%s＝%s B" % (k, v) for k, v in sorted(wv["app"].items())), WINS=wv["ins"],
        NVRC=str(gpu["nvrc"]), NVWHY=gpu["nvwhy"], DRV=gpu["drv"], LIB=gpu["lib"], NVTOTAL=str(gpu["total"]),
        NVME=str(gpu["mine"]), NVLOOSE=str(gpu["loose"]), NVLOOSE0=gpu["loose0"], NVUPG=str(gpu["upg"]),
        NVSTALE=gpu["stale"],
        UNOW=str(u["un"]), MNOW=str(u["sm"]), MNOWT=str(u["mo"]), UNPIN=str(u["unp"]),
        NFOLD=str(u["fold"]), NFOLDN=u["foldnames"],
        WNU=str(wv["wnu"]), WNM=str(wv["wnm"]), WNT=str(wv["wnt"]),
        LANDPYB=str(b(LANDPY)), LANDPREB=str(b(LANDPRE)), LANDWRB=str(b(LANDWR)),
        DAILYNAME=os.path.basename(DAILY), DAILYLINES=str(len(rb(DAILY).splitlines())), DAILYB=str(b(DAILY)),
        DAILYM=md5(DAILY, 12), PREVDAILYNAME=os.path.basename(PREVDAILY),
        NBKTSELF=str(len(BKEY_SELF)),
        LANDRED=str(len(hits["落地腿红灯现场"])),
        ARCHA1N="_raw_s%d_archive_check.attempt1-claim-space-mismatch" % PREVSLICE,
        MSGCTLN=os.path.basename(MSGCTL), RPYMAX=str(max(RPY)), PREVMSGNAME=os.path.basename(PREVMSG),
        MSGSC_RC=mscope["rc"], MSGSC_LAST=mscope["last"], MSGSC_PINB=mscope["pinb"],
        MSGSC_PINM=mscope["pinm"], MSGSC_PINOUT=mscope["pinout"], MSGSC_PINNAME=mscope["pinname"],
        MSGSC_RLINE=mscope["rline"], MSGSC_RFILE=mscope["rfile"],
        SELFCUN=sc["un"], SELFCMARK=sc["mark"], SELFCMARKB=sc["markb"], SELFCB=str(sc["b"]),
    )
    assert sum(len(hits[k]) for k in BKEY_PIN) == pnu, "前 12 桶和 %d ≠ §26.33 落笔未跟踪数 %d" % (
        sum(len(hits[k]) for k in BKEY_PIN), pnu)
    return V


TMPL = """@@SUBJECT@@

本片＝分片 @@SLICE@@（上一片＝@@PREVSLICE@@，再上一片＝@@PREV2@@）；四个片号出处互证＝
HEAD 提交说明「S7 分片 @@PREVSLICE@@」（HEAD 现取＝@@HEADD7@@）／全树回执 glob 的最大号角本＝@@PREVSLICE@@／
上一片回执表头里那句「上一片（@@PREV2@@ 号）回执链」＝@@PREVSLICE@@−1／全树尺子 glob 的最大号角本＝@@RPYMAX@@（尺子本体 `@@RULER@@`），任一对不上就拒绝生成。
上一片回执在 **@@PREVDIRD@@**、本件在 **@@EDIRD@@**（跨日期：目录由那次 glob 命中现取，不硬写）。
信息本身由 `@@SELFNAME@@`（md5 前 12 @@SELFM@@）从命令现取，不是手打。

入库面：新增 @@NA@@ 件＋修改 @@NM@@ 件＝@@NTOT@@ 件；`git diff --cached --shortstat` 原文「@@SS_RAW@@」，
三个数已与 `--numstat` 逐行现算闭合（@@SS_F@@／@@SS_I@@／@@SS_D@@）。**src／tests 本轮 @@NCODE@@ 件改动**
（`@@NCODE_CMD@@` 现数 @@NCODE@@ 条，本件要求它恰为 0 ⇒「纯归档腿」这句有凭据）。
新增件按**角色**分 @@NBKT@@ 桶，桶和＝@@NA@@＝新增总数（一件落进 0 个或 ≥2 个桶都拒绝生成；空桶也拒绝）：@@BKTPIN@@／@@BKTSELF@@。
前 12 桶与 §26.33「入库面」那条登记的落笔分组**逐组等值**：该节原文的等式＝@@PIN_EQ@@＝@@PIN_SUM@@，
本件按文件名模式现数同一批＝@@UNSUM@@＝落笔未跟踪 @@PIN_NU@@（标签与顺序逐项对齐才算那条登记不是凭记忆写的）。
后 @@NBKTSELF@@ 桶共 @@UNSUMSELF@@ 件在落笔之后才存在
⇒ 与那句「最终入库数**大于**这里的 @@PIN_TOT@@」互证（现数 @@NTOT@@＞@@PIN_TOT@@）。
⚠ 本片自己的落地链（`am_s7_verify@@SLICE@@.py`／`_raw_verify@@SLICE@@.out`＋`.err`／`am_s@@SLICE@@_hdr.py`）此刻还不存在
（全树回执 glob 里 @@SLICE@@ 号角本缺席，本件现验），本片不预支。
计数口径＝`--untracked-files=all`，它的凭据在**落笔那一侧**：写入件归档 stdout 现取「待入库＝未跟踪 @@WNU@@＋修改 @@WNM@@＝@@WNT@@ 处」，
与 §26.33 那句等式的左边 @@PIN_NU@@、本件按模式现数的前 12 桶 @@UNSUM@@ 三处同值。到本件生成这一刻那 @@PIN_NU@@ 张面已全部暂存，
⇒ 未跟踪侧现数 @@UNOW@@ 件、其中属于落笔清单的 @@UNPIN@@ 件（须 0，分母＝@@PIN_NU@@）。
默认口径（不带 `-uall`）会把整目录未跟踪的新目录折成一行 ⇒ 生成时刻这条折叠行现数 @@NFOLD@@ 件（@@NFOLDN@@），本轮**为空**是因为无面可折。
上一片那条「未跟踪 ∉ 入库清单＝0」的闸同理：本件落笔第一版把它写成「当前未跟踪＝暂存 A 清单」，那条断言当场红
（`git status` 里暂存过的新文件不再显示为未跟踪 ⇒ 等式按构造不可能成立＝本轮第三次「判据自己红不了」的形状），
⇒ 本件把它改成上面那两条交集读数＋一条如实的空值，而不是照抄一条本轮满足不了的声称。

修改 @@NM@@ 件＝§26.33 现取的那批已跟踪档案面，三个出处同值＝暂存 M @@NM@@／落笔登记的修改数 @@PIN_NM@@／`git status` 索引列现数 @@MNOW@@。
面清单＝@@PINFACES@@；其中 `@@PREVDAILYNAME@@` 是上一片的纪实——它的落地子条由**提交之后**的填值件追加，
所以时间戳必然晚于上一片的提交时刻，因而落在本片入库面里，不是遗漏。
行数由 `git show HEAD:<面>`
与 `git show :<面>`（**暂存 blob**）两处现取：@@FACE@@；合计净增 @@FACE_TOT@@ 行，每面 `numstat 增行＝暂存行数−HEAD 行数` 且删除 0 行，
`git diff --name-only` 为空＝无未暂存漂移（`git status` 工作树列现数 @@MNOWT@@ 件，同一件事的第二把尺子）；当日纪实 `@@DAILYNAME@@` 是**新增**面（@@DAILYLINES@@ 行／@@DAILYB@@ B，md5 前 12 @@DAILYM@@）。
**形制约束的落实**：开发日志暂存 blob 里最后一节＝§26.@@SECN@@（该节在 HEAD 里 @@NH@@ 次、在暂存 blob 里 @@NS@@ 次 ⇒ 整节随本片落地）；
上一节号有两个出处互证＝HEAD 提交说明「开发日志 §26.@@PREVSEC@@」与上一片回执表头里那句「开发日志 §26.@@RECSEC@@」（集合相等才算）；
§26.@@SECN@@ 顶级子条 @@NBUL@@ 条且标签互异，其中引用上一片回执名（`@@PREVLOGNAME@@`）的恰好 1 条、带引号的钉住登记恰好 1 条、
承接的 LESSON 锚句恰好 1 条、本轮新学锚句恰好 1 条（现数＝@@ND@@）——这正是分片 @@SLICE@@ 表头生成器 `bullet_for` 的节内唯一性要求；
上一节 §26.@@PREVSEC@@ 里引用 `@@PREVLOGNAME@@` 的顶级子条＝@@PREVCITE@@ 条（须恰好 1），该节顶级子条总数本件现算＝@@PREVBUL@@ 条，
与已归档自查件原始件那句「§26.@@PREVSEC@@ 顶级子条＝@@ARCBUL@@ 条」两出处闭合（同一把切段尺子现算，不是同一条声称被引用两次）。

**本轮抓到的一处错标（＝§26.32 那条 UNCOVERED 在本轮真的撞上了）**：§26.33 落笔那句原本写「顶格字段 99 行」，而「顶格字段」这一族
既有口径全是**表头生成器 `FIELDS` 的声明数**＝@@NDEF@@（HEAD 那版＋暂存版开发日志里该标签的读数现数 @@NDEFVALS@@ 处，本件要求全部等于 @@NDEF@@）。
99 那个数是**表头区非空行数**（含缩进续行），是另一个量。⇒ 一个数字现取、过闸、印出，仍可能填进另一个标签的槽。修法三条都落实：
① 写入件 `@@WRITERNAME@@` 把那个量改名 `HDRLINES30`，另起 `NDEF30` 由三个出处现取并要求同值（`@@HDRC@@` 的 `FIELDS` ast 声明＝@@NDEF@@／只读仪器那句「@@AFNDEF@@」／回执表头里顶格行的现数）；
② 档案里那句在**未提交**的追加段内就地改标（对 HEAD 仍是净增 ⇒「每面删除 0 行」那条前提没破）；
③ **本件加一条普查闸门**：全开发日志里「顶格字段」紧跟的数字不许不等于 @@NDEF@@，坏输入（把 16 改回 99）进 `@@SELFCTL@@` 打红。

本轮新学（环境级＋口径级）＝**普查仪器坏了 ⇒ 红线换承担者，而不是换口径，也不去修共享状态**：
`nvidia-smi --query-compute-apps=pid` 本轮起不来（rc=@@NVRC@@，首行「@@NVWHY@@」）；根因由两条命令现取并对上＝内核模块
`/sys/module/nvidia/version`＝@@DRV@@ 与用户态库＝@@LIB@@ 是两个版本；用户态那个数取到**两个同值出处**＝dpkg 里状态 `ii` 的 amd64 包版本＋`libnvidia-ml.so.1` 顺符号链接解析到的文件名（盘上另有已卸载包留下的残留版本号＝@@NVSTALE@@，它不是「哪份在用」的凭据）；机器上此刻还有 @@NVUPG@@ 个 unattended-upgrade 进程在跑，那是旁证不是结论。
于是「我没占卡」这条红线失去了一直依赖的那把尺子。本轮不假装量到、也不重载内核模块（那会打断用户的 GPU 进程），
改由 `/proc/*/cmdline` 的 argv[0] **整词**匹配承担：现数我的解释器进程 **@@NVME@@／@@NVTOTAL@@** 个（分母＝同一趟扫到的 PID 数，已排除自身与父进程），
配两条对照——合成的 argv[0] 必须数出 1、命令行里**含**这条路径但 argv[0] 不是它的必须数出 0。
同一条清单上松散子串匹配现数 @@NVLOOSE@@ 个（argv[0] 依次为 @@NVLOOSE0@@）⇒ 松散匹配会把「承载这条命令的那个 shell」自己也认成我的 GPU 进程；
这条误指不靠现场运气：本件另造了一条合成命令行，松散口径 1／整词口径 0。**指纹四路闭合**：现算（默认 locale，那条命令**不钉 LC_ALL**）＝@@FDEFD@@＝回归件 SRCFINGER_START＝SRCFINGER_END＝
上一片回执表头 SRC指纹（@@RFINGER@@）；另一条钉 `LC_ALL=C` 现算＝@@FCNT@@，两值必须**不同**（相同就说明这条既有登记已腐烂）；
两个值都在 §26.32 的登记里（HEAD 那版开发日志里现数 @@HEXCOUNT@@ 个 32 位十六进制串，本件要求这两条都在集合里）。

本轮无代码腿 ⇒ 没重跑全量回归；被引用那次的凭据（逐字节从 @@REGPATH@@ 现取）：**@@REG@@**（＝@@REGNF@@ failed＋@@REGNP@@ passed＋@@REGNS@@ skipped），
墙钟 @@REGW@@，rc=@@REGRC@@（rc=1＝两条既有红灯，非本轮引入），@@REGD1@@ → @@REGD2@@，当时的树＝@@REGH@@ dirty=@@REGDIRTY@@。
两条登记红的 E 行：①@@REGE1@@；②@@REGE2@@。`grep -ic 'xfail\\|xpass'`＝@@XF@@ ⇒ A0 断言未放宽、#27 不换指标、无新增 skip/xfail
（本轮 src＋tests 改动 @@NCODE@@ 件，没有可与之比较的新测试；且被测树与那次回归的树在 src＋tests 上指纹逐字节相同，见上面那条四路闭合）。

分片 @@PREVSLICE@@ 回执入库（@@CURL@@，@@CLIB@@ B／@@CLINES@@ 行，md5 前 12 @@CLM@@；读数从回执正则现取，每条要求恰好一次命中）：
@@VERD@@；落地树 head＝@@HEAD40@@（父 @@P40@@，dirty(src+tests)=@@DIRTY@@）＝本地 HEAD；tree 条目 @@ENTRIES@@＝blob @@BLOB@@＋tree @@TREEC@@
（truncated=False）；闭合式 远端 @@REMOTE@@ == 父提交 blob @@PARENTB@@ ＋ 新增(A) @@NA_PREV@@ ＝ @@EXPECT@@ -> PASS；
双向路径差集 @@DL@@／@@DLDEN@@ 与 @@DR@@／@@DRDEN@@；全库 blob SHA 不符 @@SHA@@／@@SHADEN@@；本片单列复核 @@MATCH@@／@@NFILE@@ MATCH（A @@GA@@／M @@GM@@）。
四个正对照都能红（每格＝回执里那段**逐字节摘出**的原子读数，标签不在模板里重复一遍）：A 远端一条 sha 首位改 f ⇒ @@CTLA@@；B 远端删一条路径 ⇒ @@CTLB@@；C 闭合式右边 +1 ⇒ @@CTLC@@；
D 远端删**本片**一条路径 ⇒ @@CTLD@@。本件现验：A／B／D 各自第一段的不匹配数均 ≥1、C 那句判 FAIL，分母与每个 0 同印，且正文 blob 数与闭合式左值现比相等。
**本轮复跑**（不引用它自己的声称）：`cmp -i @@BOFF@@:0` 三件套 rc＝@@TRIPLE@@（同偏移须 0、任一侧偏移 +1 须非 0），
正文区与 `@@RAWRN@@`（@@RAWRB@@ B，md5 前 12 @@RAWRM@@，stderr @@RAWRE@@ B）逐字节相同；回执总长＝偏移＋原始件字节数（本件现验）。
表头顶格字段 @@NDEF@@ 行＝`@@HDRC@@` 的 `FIELDS`（ast 现取、不执行）声明且同序，与只读仪器那句「@@AFNDEF@@」同值。
尺子＝**逐字节副本**：md5 前 12 `@@RULERMD5@@` 这一族全树现算共 @@COPIES@@ 份（按 **md5 分组**数，同一条 glob 的形状命中 @@SHAPE@@ 份，差额有名字＝@@STRAY@@），
普查含本体 ⇒ 不是漏了自己的计数。本片要抄第 @@NEXTCOPY@@ 份。

分片 @@PREVSLICE@@ 其余入库件的读数（一律从档案正则现取，每条要求恰好一次）：表头预检面／写盘面＝@@HDRPREB@@／@@AFWRITEB@@ B
（写盘面里那句三联 0／1／1、字段 @@NDEF@@／@@NDEF@@，与本轮复跑同值）；表头控件面 `am_s@@PREVSLICE@@_hdr_controls.py` 的 stdout
@@HDRCTLB@@ B（md5 前 12 @@HDRCTLM@@，stderr 0 B）＝红路 @@HDRRED@@ 条＋保持绿 @@HDRGREEN@@ 条（绿路声明与逐行数到的 G 行现比相等）、
本面自查 @@HDRSELF@@ 条（@@HDRSELFN@@）、超时 0 条、如实登记造不出红路 @@HDRUNC@@ 条；只读仪器 `am_s@@PREVSLICE@@_hdr_after.py` 的 stdout
@@AFOUTB@@ B＝三联 @@AFTRIPLE@@、判读行「@@AFVERDICT@@」，它的控件面 stdout @@AFCTLB@@ B（md5 前 12 @@AFCTLM@@）＝红路 @@AFNCASE@@ 条＋保持绿
@@AFGREEN@@ 条、造不出红路 @@AFUNC@@ 条；提交说明控件面 stdout @@MSGCTLB@@ B（md5 前 12 @@MSGCTLM@@，stderr 0 B）＝CASES 声明 @@MSGNCASE@@ 条、
超时 0 条、造不出红路 @@MSGNUNC@@ 条 ⇒ 三面判读行现比同为「@@CTLVERD@@」。判红窗口与两个继承常量从**该件源码**现取（不重敲）＝
子运行上限 @@SUBTO@@ s／环境故障累计 @@ENVSTOP@@ 条／raise 行号 ±@@WINDOW@@ 行；超时**既不判红也不判过**、单独计数。
归档自查件 `@@ARCHPYNAME@@`（@@ARCHPYB@@ B，md5 前 12 @@ARCHPYM@@）及其 stdout @@ARCHB@@ B（md5 前 12 @@ARCHM@@）／stderr @@ARCHEB@@ B＝
声称 @@ARCHCLAIMS@@ 条全部命中原样、正对照**逐行**数到 @@ARCHCTL@@ 条 ⇒「@@ARCHVERD@@」；它的承载面读数＝@@ARCHCARRY@@（括号里「独」＝只出现在那一张面的声称条数，逐条现取），
其中承重最多的一面＝@@ARCHMAX@@——这句不是照抄那一行的数，而是由本件两条断言现取
（max 取面＋并列当场红，红掉任何一条这句话就不成立）。它自己那轮的红灯现场 `@@ARCHA1N@@`（@@ARCHA1B@@ B）一并入库。
落地腿 `am_s@@PREVSLICE@@_docs_landing.py`（@@LANDPYB@@ B）与其预检／写盘两面 @@LANDPREB@@／@@LANDWRB@@ B、以及本件按角色桶现数的 @@LANDRED@@ 件落地腿红灯现场（该桶标签与件数已在上面「入库面」那条里同印）一并入库。

⚠ **局部仪器不是常备尺子**（本轮**实跑两件**为证，不是转述；两件都只读／只打 stdout，不写仓库 ⇒ 复跑无副作用）：
① 上一片的归档自查件 `@@ARCHPYNAME@@` ⇒ rc=@@SCOPE_RC@@、stdout @@SCOPE_PINOUT@@ B，抛出的那一行原文「@@SCOPE_LAST@@」。
它数的是**磁盘上最后一节**（现在是 §26.@@SCOPE_SEC@@），而它的 LESSON 锚句是上一片那一轮写死的短语 ⇒ 这一族仪器只在自己那一轮成立，
不许当回归件复跑。本轮实跑的 stderr 与已归档的复跑现场 `@@SCOPE_PINNAME@@.err`（@@SCOPE_PINB@@ B，md5 前 12 @@SCOPE_PINM@@）
**逐字节相同**＝两出处（本轮实跑＝档案里那次），同一条代码路径。
② 上一片的提交说明生成器 `@@PREVMSGNAME@@` ⇒ rc=@@MSGSC_RC@@、stdout @@MSGSC_PINOUT@@ B，抛出点＝`@@MSGSC_RFILE@@` 的第 @@MSGSC_RLINE@@ 行，
末行原文「@@MSGSC_LAST@@」；它的 stderr 同样与已归档现场 `@@MSGSC_PINNAME@@.err`（@@MSGSC_PINB@@ B，md5 前 12 @@MSGSC_PINM@@）逐字节相同。
⚠ **落笔第一版预告它红在桶／普查闸，实跑红在另一条**：它把「上一片回执」按 HEAD 片号＋1 现取 ⇒ 本轮回执已入库、与它同目录 ⇒ 先撞跨日期那条闸。
这一族的红点**不许按预期写**，现场的末行只能逐字节取。
另有一条**已归档的标本**说明这条闸在上一轮是绿的：那张自查件的 stdout 里那句「§26.@@PREVSEC@@ 顶级子条＝@@ARCBUL@@ 条；本片回执名／钉住口径／
上一片 LESSON／本片 LESSON 各恰好 1 条且各配诱饵」——四条唯一性在它那一轮各命中 1，本轮同一把尺子在 §26.@@SECN@@ 上红在 LESSON 那条。

本片写入件的两条红灯现场（都是它自己的，不是抄上一片）：① `_raw_s@@SLICE@@_docs_precheck/write.attempt1-selfcount-recursive`
（stderr @@T1B@@ B／stdout @@T1O@@ B，md5 前 12 @@T1M@@，末行原文「@@T1LAST@@」；那句里的三个数现比＝@@T1NUMS@@）＝**自指的计数**：
预检与写盘两次落笔读出不同的未跟踪数，因为那两次运行自己的 stdout／stderr 面就在清单里。修法＝这些面**先落盘再跑**、
并把它们本身编进角色分组（于是口径 ⑥ 的分组和与现数在两次运行里同值）。② `_raw_s@@SLICE@@_docs_write.attempt2-selfguard-hits-itself`
（stderr @@T2B@@ B，md5 前 12 @@T2M@@，末行「@@T2LAST@@」）＝那条「不许动正在跑的仪器」的守卫被**本件自己的名字**打掉
（它带着 `timeout`／`bash -c` 两层 wrapper）；它在任何追加之前就停 ⇒ 五张档案面当时逐字节未被动过，这一条由写盘后的行数差复核。
修法＝自己那一支按「argv[0]＝解释器」数副本、wrapper 不参与判定，别的仪器仍按整词命中拦停。
写入件本轮的读盘面＝预检 @@WPREB@@ B（md5 前 12 @@WPREM@@／stderr @@WFREE@@ B）、写盘 @@WWB@@ B（md5 前 12 @@WWM@@／stderr @@WWE@@ B），
判读行「@@WV1@@」与「@@WV2@@」；追加四面＝@@WAPP@@，总览那条插在 `## 7. 文档索引` 之前（偏移 @@WINS@@ 字节）。
⚠ 一条如实登记的**偏离**：写入件源码（现值 `@@WRITERNAME@@`＝@@WPYB@@ B，md5 前 12 @@WPYM@@）在上面那条错标修好之后被改过
⇒ 它那两张运行面与源码不再一一对应；没有复跑它，
因为那条闭合式闸门数的是**落笔那一刻**的件数（本轮已因新面移动），复跑必然红在那句——红得对但不提供新证据。

诚实登记（本片没做什么）：① 无代码腿、无新测试、无回归重跑、无 A3 求解；② **性能数字为零**——GPU 全程未碰，
本轮**没有** NVML 那把尺子（原因与替代口径见上面那条新学），替代断言＝我的解释器进程 @@NVME@@／@@NVTOTAL@@，
该 0 的正对照＝合成的 argv[0] 必须数出 1（本件现验）；③ 钉住口径原文从暂存 blob 现取＝「@@QUOTE@@」，出处唯一性已在上面 §26.@@SECN@@ 的计数里验过；
④ **本片（分片 @@SLICE@@）自己的落地证明要到下一轮才存在**：本信息只声称分片 @@PREVSLICE@@ 那一份已落地，不预先声称本片被复核过；
⑤ 本件的干跑控件面 `@@SELFCTL@@`（@@SELFCB@@ B）此刻**已在入库清单里**（上面自件桶现数含它，且本件对它有一条「命中 1」的断言），但本信息**不引用它自己的控件读数**
（扰动条数／UNCOVERED／判读行）——控件跑的是本件的副本，本件若在控件之前读那份 stdout 就是循环；那份读数由**分片 @@NEXTSLICE@@** 的归档自查腿引用。
本件对它只取**它源码里的两个自述常量**（不是它的 stdout）：判据锚点句「@@SELFCUN@@」与基线可运行标记「@@SELFCMARK@@」，
各要求在该源码里**恰好命中 1 行**（0 行＝那条扰动会被判空转，2 行＝±3 行窗口会指错闸门；标记句那行 @@SELFCMARKB@@ B 交给控件面做逐字节相等判据）；
⑥ 上一片的提交说明控件面 `@@MSGCTLN@@`（@@MSGCTLB@@ B，md5 前 12 @@MSGCTLM@@，stderr 0 B）**本轮没有复跑**——它扰动的是它自己那一轮的
生成器副本，那些副本读的是**当下**的暂存集与档案目录 ⇒ 换轮后它红的不是判据而是现场（上面实跑两件局部仪器已经把这件事证出来了）。
本信息引用它的读数**全部取自档案那一份**（CASES／UNCOVERED／超时三条见上面「其余入库件」那段），没有一条来自本轮复跑。

排期：#30 试片尺寸裁决（A3 关键路径，当前唯一可评分面＝N=1）→ raw 臂 N≥2 ＋ `--stage full --measured-per-track`
→ #27 腿②（缺省 tuned→hk）与 #33 的 15 位点系数**同批**，只对 A3 的 1→18 外部趋势打分 → #40 → #39；
#41／#37／#38／#29 各自待办；#51 已量化（对形态 ≤0.0040%）降级不关单。GPU 仍 parked；NVML 那条环境债本轮只登记、不修。
"""


SUBJECT = ("S7 分片 %s：纯归档腿＝§26.32 登记的形制约束由「另起 §26.33」满足＋分片 %s 的落地半腿整条链"
           "（回执链／表头预检写盘与控件面／红灯现场／只读仪器与其各面／落地腿与红灯现场／归档自查腿／当日纪实／本片写入件）入库" % (
               SLICE, PREVSLICE))


def main():
    # 槽位**形制**普查（跑在昂贵的 build 之前）：模板里的每个 `@` 必须是成对槽 `@@K@@` 的一部分。
    # 这条是本轮补的：`@@K@`（尾@漏一个）既不会被 str.replace 命中，也不会被 `@@([A-Z0-9_]+)@@` 的
    # 「未取值槽」审计抓到 ⇒ 坏形制会原样印进提交说明（＝把「每个数都现取」那句话变成假话）。
    stray = re.sub(r"@@[A-Z0-9_]+@@", "", TMPL).count("@")
    assert stray == 0, "模板里有 %d 个不属于成对槽 `@@K@@` 的 `@` ⇒ 有槽位形制坏了（漏一个 @ 就静默不替换）" % stray
    v = build()
    v["SUBJECT"] = SUBJECT
    txt = TMPL
    used = set()
    for k, val in v.items():
        if "@@%s@@" % k in txt:
            assert isinstance(val, str), "槽 %s 的取值不是字符串：%r ⇒ 拒绝把 %r 印进提交说明" % (k, val, val)
            txt = txt.replace("@@%s@@" % k, val)
            used.add(k)
    left = sorted(set(re.findall(r"@@([A-Za-z0-9_]+)@@", txt)))
    assert not left, "模板里有未取值的槽：%s ⇒ 那条数没现取，拒绝生成" % left
    unused = sorted(k for k in v if k not in used)
    assert not unused, "取值表里有模板没用到的读数：%s（没用到的读数＝下一条等着的谎）" % unused
    assert "@" not in txt, "渲染后的文本里仍有 `@` ⇒ 有槽位没被替换（坏形制或取值里混进了占位符）"
    first = txt.splitlines()[0]
    assert first == SUBJECT, "主题第一行必须就是登记的那句：%r" % first[:80]
    assert len(first) <= 300, "主题第一行 %d 字符过长" % len(first)
    for l in txt.splitlines():
        assert l.count("（") == l.count("）"), "全角括号按行不成对：%r" % l[:120]
    body = txt[len(first):]
    assert body.count("`") % 2 == 0, "行内代码的反引号不成对"
    stripped = re.sub(r"`[^`]*`", "", body)
    assert stripped.count("**") % 2 == 0, "粗体标记不成对（已剔除行内代码）"
    assert "PLACEHOLDER" not in txt and "@@" not in txt, "残留占位符"
    assert txt.endswith("\n"), "尾换行"
    sys.stdout.write(txt)
    return txt


if __name__ == "__main__":
    main()
