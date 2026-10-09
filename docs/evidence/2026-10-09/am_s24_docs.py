#!/usr/bin/env python3
"""分片 23 落地回执之后的五条归档腿（开发日志 (t) ＋计划文档 ＋总览 ＋两条 .workbuddy）。

与上一轮的 am_s23_docs.py 同构，口径进步三处（都来自本轮读文件时发现的真实缺陷）：
① 所有数从**回执**正则现取（表头读数从表头取、正文读数从正文取），一个也不手填——沿用；
② 自纠计数**不手填**：从总览现文里正则取上一条计数再＋1，取不到就拒绝写盘；
③ mid 模式的插入点**必须先补一个换行**，并对上一片留下的「条目粘连」做一次前后计数的修复
   （修复前须＝普查值、修复后须＝0），因为渲染结构也是会被写坏的东西。
每腿带锚点预检（出现次数须 1、eof 须真在末尾、mid 须紧跟分隔线）、写后回读、幂等。
"""
import os
import re
import subprocess
import sys

REPO = "/home/shy/桌面/2026年开发-显式冲击有限元-物质点法-sph-fvm+am/AM-Qoder"
D = os.path.join(REPO, "docs/evidence/2026-10-09")
REC = os.path.join(D, "am_s7_verify23.log")
RAW = os.path.join(D, "_raw_verify23.out")
ERRF = os.path.join(D, "_raw_verify23.err")
RULE = os.path.join(D, "am_s7_verify23.py")
HDR = os.path.join(D, "am_s23_hdr.py")
DEV = os.path.join(REPO, "docs/开发日志.md")
OV = os.path.join(REPO, "docs/项目开发总览.md")
MK_T = "### 26.27 (t)"
GLUE = "。- **2026-10-09（S7 分片 22 落地）"
WRITE = "--write" in sys.argv


def sh(*a):
    return subprocess.run(a, cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def md5(p):
    return subprocess.run(["md5sum", p], capture_output=True, text=True).stdout.split()[0]


def cmp_rc(off_l, off_r):
    return subprocess.run(["cmp", "-i", "%d:%d" % (off_l, off_r), REC, RAW],
                          capture_output=True).returncode


# ---- 回执切分：正文区从 BODY_OFFSET 的**字节**位置起（表头是中文重的，str 下标≠字节下标）----
raw_b = open(REC, "rb").read()
boff = int(re.search(rb"BODY_OFFSET=(\d+)\n", raw_b).group(1))
head_txt = raw_b[:boff].decode("utf-8")
body_txt = raw_b[boff:].decode("utf-8")
assert raw_b[boff:] == open(RAW, "rb").read(), "回执正文区与归档的原始 stdout 不逐字节相同"
C0, C1, C2 = cmp_rc(boff, 0), cmp_rc(boff + 1, 0), cmp_rc(boff, 1)
assert (C0, C1, C2) == (0, 1, 1), "cmp 三联不齐（同偏移须 0、左右各 +1 须 1）：%s/%s/%s" % (C0, C1, C2)
assert "@@" not in head_txt and "＠" not in head_txt, "回执表头有占位符残留"


def H(pat):
    m = re.search(pat, head_txt)
    assert m, "表头里取不到 " + pat
    return m


def B(pat, grp=1):
    m = re.search(pat, body_txt)
    assert m, "正文里取不到 " + pat
    return m.group(grp)


# 同一把尺子在档案目录已有几份：glob＋md5 现算（写死它就会随片腐烂，见 (t) 第 6 条）
ruler_md5 = md5(RULE)
ruler_copies = sorted(f for f in os.listdir(D)
                      if re.fullmatch(r"am_s7_verify\d+\.py", f) and md5(os.path.join(D, f)) == ruler_md5)
assert "am_s7_verify23.py" in ruler_copies, "本片的尺子没被自己的 glob 数到 ⇒ 计数口径失真"

# 本片新增件与 /tmp 同名件的三分类（现比，不抄任何档案）
added = [r.split("\t")[-1] for r in sh("git", "-c", "core.quotePath=false", "show",
                                       "--name-status", "--format=", "HEAD").splitlines()
         if r.startswith("A\t")]
md_same, md_diff, md_absent = [], [], []
for p in added:
    t = os.path.join("/tmp", os.path.basename(p))
    if not os.path.exists(t):
        md_absent.append(p)
    elif md5(os.path.join(REPO, p)) == md5(t):
        md_same.append(p)
    else:
        md_diff.append(p)
assert len(md_same) + len(md_diff) + len(md_absent) == len(added), "三分类没覆盖全部新增件"

# 粘连普查：全库五份文档各数一次，只在总览命中（命中数写进归档前是现取的）
glue_hits = {}
for rel in ("docs/开发日志.md", "docs/项目开发总览.md", "docs/项目评估与下一步计划_2026-10-06.md",
            ".workbuddy/memory/2026-10-09.md", ".workbuddy/memory/MEMORY.md"):
    glue_hits[rel] = open(os.path.join(REPO, rel), encoding="utf-8").read().count(GLUE)
GLUE_TOT = sum(glue_hits.values())
GLUE_OV = glue_hits["docs/项目开发总览.md"]

# 自纠计数：从总览现文取上一条，再＋1（不手填）
ov_now = open(OV, encoding="utf-8").read()
prev = re.findall(r"自纠(?:仍|由 \d+ 到)? \*\*(\d+)\*\* 条", ov_now)
assert prev, "总览里取不到上一条自纠计数 ⇒ 拒绝手填"
PREVCNT = int(prev[-1])

V = {
    "NOW": sh("date", "+%F %T %z"),
    "HEADS": sh("git", "rev-parse", "--short", "HEAD"),
    "PARENTS": sh("git", "rev-parse", "--short", "HEAD^"),
    "DIRTY": str(len([l for l in sh("git", "-c", "core.quotePath=false", "status", "--porcelain",
                                    "--", "src", "tests").splitlines() if l.strip()])),
    "FINGER": H(r"SRC指纹 : ([0-9a-f]{32})").group(1),
    "RSIZE": str(os.path.getsize(REC)),
    "MDREC": md5(REC)[:12],
    "MDHDR": md5(HDR)[:12],
    "MDRULE": ruler_md5[:12],
    "NRULE": str(len(ruler_copies)),
    "RULE18": ("相同" if md5(os.path.join(D, "am_s7_verify18.py")) == ruler_md5 else "不同"),
    "RULE21": ("相同" if md5(os.path.join(D, "am_s7_verify21.py")) == ruler_md5 else "不同"),
    "RULE22": ("相同" if md5(os.path.join(D, "am_s7_verify22.py")) == ruler_md5 else "不同"),
    "RAWSZ": str(os.path.getsize(RAW)),
    "RAWMD": md5(RAW)[:12],
    "ERRSZ": str(os.path.getsize(ERRF)),
    "ERRMD": md5(ERRF)[:12],
    "BOFF": str(boff),
    "CMP0": str(C0), "CMP1": str(C1), "CMP2": str(C2),
    # 正文读数
    "ENTRIES": B(r"tree 条目 (\d+) = "),
    "NB": B(r"blob (\d+) \+ tree"),
    "NTREE": B(r"blob \d+ \+ tree (\d+)"),
    "NP": B(r"父提交 = (\d+)"),
    "NA": B(r"新增 (\d+) \+ 修改"),
    "NM": B(r"新增 \d+ \+ 修改 (\d+)"),
    "EXPECT": B(r"\+ 新增 (\d+) = (\d+)", 2),
    "NFILES": B(r"本片文件 (\d+) 个"),
    "DL": B(r"本地独有 (\d+)/"),
    "DR": B(r"远端独有 (\d+)/"),
    "MS": B(r"blob SHA 不符 (\d+)/"),
    "NSHARED": B(r"blob SHA 不符 \d+/(\d+)"),
    "SLICE_MATCH": B(r"MATCH (\d+)/"),
    "CA": B(r"正对照 A（远端 \S+ 的 sha 首位改 f）⇒ 不符 (\d+)/"),
    "CBA": B(r"正对照 A（远端 \S+ 的 sha 首位改 f）⇒ 不符 \d+/(\d+)"),
    "CB1": B(r"正对照 B（远端删一条路径）⇒ 本地独有 (\d+)/(\d+)，远端独有 (\d+)/(\d+)", 1),
    "CDB1": B(r"正对照 B（远端删一条路径）⇒ 本地独有 (\d+)/(\d+)，远端独有 (\d+)/(\d+)", 2),
    "CB2": B(r"正对照 B（远端删一条路径）⇒ 本地独有 (\d+)/(\d+)，远端独有 (\d+)/(\d+)", 3),
    "CDB2": B(r"正对照 B（远端删一条路径）⇒ 本地独有 (\d+)/(\d+)，远端独有 (\d+)/(\d+)", 4),
    "CD1": B(r"正对照 D（远端删本片一条路径）⇒ 本片缺失 (\d+)/(\d+)，SHA 不符 (\d+)/", 1),
    "CD2": B(r"正对照 D（远端删本片一条路径）⇒ 本片缺失 (\d+)/(\d+)，SHA 不符 (\d+)/", 3),
    "NBP": B(r"闭合式右边 \+1）⇒ (\d+) == (\d+) 判", 2),
    "VERDICT": B(r"(LANDING_VERDICT = .*)").strip(),
    # 表头自述读数
    "NVPID": H(r"compute 进程 (\d+) 个").group(1),
    "NVME": H(r"所留 (\d+)/\d+ 个").group(1),
    "RC2": H(r"rc=(\d+)、stderr (\d+) 字节").group(1),
    "ERRB2": H(r"rc=(\d+)、stderr (\d+) 字节").group(2),
    "NDEF": str(sum(1 for l in head_txt.splitlines()
                    if re.match(r"^\S.*?\s*:", l) and not l.startswith("BODY_OFFSET="))),
    "DSTART": H(r"首次归档运行 (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d \+\d{4})").group(1),
    "MDSAME": str(len(md_same)),
    "MDIF": str(len(md_diff)),
    "MDAB": str(len(md_absent)),
    "MDIFES": "、".join(os.path.basename(p) for p in md_diff) or "无",
    "NA_SLICE": str(len(added)),
    "GLUE_TOT": str(GLUE_TOT),
    "GLUE_OV": str(GLUE_OV),
    "GLUE_OTHER": str(GLUE_TOT - GLUE_OV),
    "PREVCNT": str(PREVCNT),
    "CNT": str(PREVCNT + 1),
}
assert V["VERDICT"] == "LANDING_VERDICT = ALL CHECKS PASS", "正文判读行不是全过 ⇒ 本轮回执不能当落地证明归档"
assert int(V["NB"]) == int(V["NP"]) + int(V["NA"]), "闭合式在正文里就不成立"
assert int(V["NDEF"]) == 16, "表头顶格字段数不是 16，实得 " + V["NDEF"]
assert V["DIRTY"] == "0", "src/tests 有改动 ⇒ 本片不是纯归档片，(t) 段那句「代码腿未动」就是假话"

SEC_T = """
### 26.27 (t) 分片 23 落地＝回执链第三环；一处"能现算的数不要写死"＋一条自纠（归档的**渲染结构**没人读）

- **本片性质**：纯归档腿（代码腿未动：`git status --porcelain -- src tests`＝@@DIRTY@@ 行，
  SRC 指纹仍＝@@FINGER@@，与上一片同 ⇒ 被测树与落地树一致）。
- **产物**：尺子 `docs/evidence/2026-10-09/am_s7_verify23.py`（md5 前 12＝@@MDRULE@@，与 18／21／22 三份
  现比＝@@RULE18@@／@@RULE21@@／@@RULE22@@；本目录 glob＋md5 现算＝同一把尺子已入库 **@@NRULE@@** 份）
  ＋落地回执 `am_s7_verify23.log`（@@RSIZE@@ 字节，md5＝@@MDREC@@，正文区起点 BODY_OFFSET＝@@BOFF@@）
  ＋表头生成器 `am_s23_hdr.py`（md5＝@@MDHDR@@）＋核验件原始 stdout/stderr（@@RAWSZ@@／@@ERRSZ@@ 字节，
  md5＝@@RAWMD@@／@@ERRMD@@，`cp -p` 归档 ⇒ mtime 仍是原次运行时刻 @@DSTART@@，不是复制时刻）。
- **落地读数（引自回执正文区，非转述）**：远端 main＝`@@HEADS@@`＝本地 HEAD（父提交 `@@PARENTS@@`）；
  `truncated=False`、tree 条目 @@ENTRIES@@＝blob @@NB@@＋tree @@NTREE@@；闭合式 **远端 @@NB@@ == 父提交 @@NP@@
  ＋ 新增 @@NA@@ ＝ @@EXPECT@@ -> PASS**；双向路径差集 **@@DL@@/@@NB@@** 与 **@@DR@@/@@NB@@**；
  全库共有项 SHA 不符 **@@MS@@/@@NSHARED@@**；本片单列复核 **@@SLICE_MATCH@@/@@NFILES@@ MATCH**
  （A @@NA@@／M @@NM@@）；`@@VERDICT@@`。
- **四个正对照（同一条运行里印出，全部能红）**：A 远端一条 sha 首位改 f ⇒ 不符 @@CA@@/@@CBA@@；
  B 远端删一条路径 ⇒ 本地独有 @@CB1@@/@@CDB1@@、远端独有 @@CB2@@/@@CDB2@@；
  C 闭合式右边 +1 ⇒ @@NB@@ == @@NBP@@ 判 FAIL；D 远端删本片一条路径 ⇒ 本片缺失 @@CD1@@/@@NFILES@@、
  SHA 不符 @@CD2@@/@@NFILES@@。
- **回执自身的复核**：仓库根跑 `cmp -i @@BOFF@@:0 docs/evidence/2026-10-09/am_s7_verify23.log
  docs/evidence/2026-10-09/_raw_verify23.out` rc＝**@@CMP0@@**（须 0），左偏移 +1 rc＝**@@CMP1@@**、
  右偏移 +1 rc＝**@@CMP2@@**（各须非 0）⇒ 正文区确是那次核验 stdout 的逐字节后缀，且两侧都在仓库里。
  表头自证：生成时复跑核验件 rc＝@@RC2@@、stderr @@ERRB2@@ 字节、顶格字段 **@@NDEF@@** 行、
  无半角成对标记残留、无全角 at 号。
- **GPU／钉住**：表头生成时现取卡上 compute 进程 **@@NVPID@@** 个，其中我的解释器（/proc/PID/cmdline 含
  /tmp/amvenv）所留 **@@NVME@@/@@NVPID@@**（须 0）；本轮全部 CPU 钉住（`CUDA_VISIBLE_DEVICES=""`）。
- **把上一轮的教训落成一处代码**（不是新缺陷 ⇒ 自纠计数 **+0**）：上一片生成器的表头里有一句写死的
  「两份档案共用同一仪器」——那是**上一片**的份数。本片由 `sed` 机械克隆上一片生成器，片号会被自动换掉、
  这种**注文里的份数**不会；我没有手改成新数字，而是加了一段 glob＋md5 现算（本轮实得 @@NRULE@@ 份，
  含本件，且断言本件自己必须被数到）。⇒ 口径：**能现算的数就不要写死；写死的文字和写死的数腐烂得一样快。**
- **自纠一条（＋1，§26.27 由 @@PREVCNT@@ 到 @@CNT@@）**：写总览那条腿时用的是"在某条目末尾之后插入"的 mid
  模式，而模板首字符没有换行 ⇒ 新条目被**粘在上一条目的行尾**（渲染时不再是独立条目）。上一片落盘后没有
  人读**渲染结构**，本片读文件才发现。普查（现算，同一模式在五份文档各数一次）＝总览 @@GLUE_OV@@ 处、
  其余四份合计 @@GLUE_OTHER@@ 处；本片把 mid 模式改成"插入前先补一个换行"，并对这一处存量做修复断言
  （修复前须＝@@GLUE_OV@@、修复后须＝0）。⇒ 计 +1 的理由是它**已经落进档案文件**（与"拦在写盘前"的 +0 分界一致）；
  新登记的口径：**归档的正确性不止是数字，还包括渲染结构与插入点位置**。
  同一条修复的**仪器**本身也错过一次：这条断言最初写成"检查插入点**前**一个字符须是换行"，而 mid 的插入点前
  正是锚点行末那个句号 ⇒ 预检当场判红（假拒）。改成"检查插入块**首字符**须是换行"，并补一条正对照
  （把换行去掉后同一处下标必须不是换行，否则这条闸门是空转）⇒ 拦在写盘前，计 **+0**。
- **同片溯源（现比）**：本片新增 @@NA_SLICE@@ 件里与 /tmp 同名件 md5 相同 @@MDSAME@@ 件、不同 @@MDIF@@ 件
  （@@MDIFES@@）、/tmp 已无同名件 @@MDAB@@ 件，三分类之和＝@@NA_SLICE@@（闭合）。只作辅助溯源。
- **自纠计数**：§26.27 ＝ **@@CNT@@** 条（上一条来自总览现文正则取回＝@@PREVCNT@@，本片＋1；
  计数本身也不手填）。
- **排期不变**：下一块＝代码腿 **#18**（路径程序驱动热源，CSV 时间轴只能做成 **opt-in**）→ #27＋#33 → #10/A3；
  #40 → #39；#29／#30 仍等用户裁决。落笔 @@NOW@@（`date` 现取）。
"""

SEC_PLAN = """
- **分片 23 落地（@@NOW@@ 现取，CPU 钉住）**：远端 main＝`@@HEADS@@`，blob **@@NB@@**＝本地、
  tree 条目 @@ENTRIES@@＝@@NB@@＋@@NTREE@@、闭合式 **@@NB@@ == @@NP@@ ＋ 新增 @@NA@@ ＝ @@EXPECT@@** -> PASS、
  双向差集 **@@DL@@/@@NB@@** 与 **@@DR@@/@@NB@@**、全库 SHA 不符 **@@MS@@/@@NSHARED@@**、
  本片 **@@SLICE_MATCH@@/@@NFILES@@ MATCH**（A @@NA@@／M @@NM@@）⇒ 分片 22 的回执与生成器都在库。
  回执＝`docs/evidence/2026-10-09/am_s7_verify23.log`。§26.27 自纠 **@@PREVCNT@@⇒@@CNT@@** 条（新增一条是
  归档的**渲染结构**被写坏后才发现，不是数字错）。代码腿未动（SRC 指纹 @@FINGER@@），下一项仍是 **#18**。
"""

SEC_OV = """- **2026-10-09（S7 分片 23 落地）＝回执链第三环，一把尺子量四片**。`am_s7_verify23.py` 与分片 18／21／22
  三份现比 md5＝**@@RULE18@@／@@RULE21@@／@@RULE22@@**（glob＋md5 现算＝同一把尺子 @@NRULE@@ 份）。
  读数（引自回执正文区）：远端 main＝`@@HEADS@@`、tree 条目 @@ENTRIES@@＝blob **@@NB@@**＋tree @@NTREE@@、
  闭合式 **@@NB@@ == @@NP@@＋@@NA@@＝@@EXPECT@@**、双向差集 **@@DL@@/@@NB@@** 与 **@@DR@@/@@NB@@**、
  全库 SHA 不符 **@@MS@@/@@NSHARED@@**、本片 **@@SLICE_MATCH@@/@@NFILES@@ MATCH**、`@@VERDICT@@`；
  正文区由 `cmp -i @@BOFF@@:0`（两侧都在仓库里）rc＝**@@CMP0@@** ＋左右各 +1 判红 证明是逐字节后缀。
  **两条新口径**：能现算的数不要写死（表头那句份数改成 glob＋md5 现算）；归档正确性含**渲染结构**
  （上一片的总览条目被粘在上一条目行尾，本片修复并把 mid 模式改成先补换行）⇒ §26.27 自纠
  **@@PREVCNT@@⇒@@CNT@@** 条。GPU 现取：卡上 compute 进程 **@@NVPID@@** 个、我的＝**@@NVME@@/@@NVPID@@**。
"""

SEC_DAY = """
### @@TIME_SHORT@@ 分片 23 落地回执（`am_s7_verify23.log`）
回执四步全过：远端 main＝`@@HEADS@@`＝本地 HEAD；tree 条目 @@ENTRIES@@＝blob **@@NB@@**＋tree @@NTREE@@；
闭合式 **@@NB@@ == @@NP@@ ＋ @@NA@@ ＝ @@EXPECT@@** -> PASS；双向差集 **@@DL@@/@@NB@@**、**@@DR@@/@@NB@@**；
全库 SHA 不符 **@@MS@@/@@NSHARED@@**；本片 **@@SLICE_MATCH@@/@@NFILES@@ MATCH**；`@@VERDICT@@`。
`cmp -i @@BOFF@@:0` rc＝@@CMP0@@、左 +1＝@@CMP1@@、右 +1＝@@CMP2@@ ⇒ 正文区逐字节＝那次 stdout，两侧都在仓库里。
尺子复用：四份核验件 md5 现比＝@@RULE18@@／@@RULE21@@／@@RULE22@@（前 12＝@@MDRULE@@），glob＋md5 现算份数＝@@NRULE@@。
仪器进步：表头里"共用仪器的份数"从写死改成 glob＋md5 现算（sed 克隆不会更新注文里的数）；总览的 mid 插入
改成先补换行＋修复上一片的**条目粘连**（普查现算：总览 @@GLUE_OV@@ 处、其余 @@GLUE_OTHER@@ 处）。
自纠计数 §26.27 **@@PREVCNT@@⇒@@CNT@@** 条（+1 落在档案文件＝渲染结构那条；份数那条拦在写盘前 ⇒ +0）。
代码腿未动（dirty=@@DIRTY@@、SRC 指纹 @@FINGER@@）。GPU：现取 compute 进程 @@NVPID@@ 个，我的解释器所留
**@@NVME@@/@@NVPID@@**；本轮全部 CPU 钉住。
"""

SEC_WBK = """
- **2026-10-09 午后（S7 分片 23 落地）**：远端 main＝`@@HEADS@@`（@@NB@@ blobs），闭合式
  `@@NB@@ == 父 @@NP@@ ＋ 新增 @@NA@@ ＝ @@EXPECT@@`、双向差集 @@DL@@/@@NB@@ 与 @@DR@@/@@NB@@、
  全库 SHA 不符 @@MS@@/@@NSHARED@@、本片 @@SLICE_MATCH@@/@@NFILES@@ MATCH ⇒ 分片 22 的回执链在库。
  回执＝`docs/evidence/2026-10-09/am_s7_verify23.log`（`cmp -i @@BOFF@@:0` rc＝@@CMP0@@ ＋两侧 +1 各红）。
  **新口径两条**：① **能现算的数不要写死**——注文里的份数（"两份档案共用同一仪器"）随片腐烂，
  已改成 glob＋md5 现算；② **归档正确性含渲染结构**——mid 插入把上一片总览条目粘连在行尾，写盘后没人读出来，
  本片修复并把插入模式改成先补换行 ⇒ §26.27 自纠 **@@PREVCNT@@⇒@@CNT@@** 条。下一项 **#18**（CSV 时间轴 opt-in）。
"""

LEGS = [
    ("开发日志 (t)", DEV, "docs/开发日志.md", MK_T, SEC_T,
     ("  #40 → #39；#29／#30 仍等用户裁决。落笔 2026-10-09 16:11:42 +0800（`date` 现取）。", "eof")),
    ("计划文档", os.path.join(REPO, "docs/项目评估与下一步计划_2026-10-06.md"),
     "docs/项目评估与下一步计划_2026-10-06.md", "分片 23 落地", SEC_PLAN,
     ("  回执＝`docs/evidence/2026-10-09/am_s7_verify22.log`。", "eof")),
    ("总览", OV, "docs/项目开发总览.md", "S7 分片 23 落地", SEC_OV,
     ("  GPU 现取：卡上 compute 进程 **4** 个、我的＝**0/4**。§26.27 自纠仍 **21** 条。", "mid")),
    ("当日", os.path.join(REPO, ".workbuddy/memory/2026-10-09.md"),
     ".workbuddy/memory/2026-10-09.md", "分片 23 落地回执", SEC_DAY,
     ("自纠计数 §26.27 仍 **21** 条。GPU：现取 compute 进程 4 个，我的解释器所留 **0/4**；本轮全部 CPU 钉住。",
      "eof")),
    ("MEMORY", os.path.join(REPO, ".workbuddy/memory/MEMORY.md"),
     ".workbuddy/memory/MEMORY.md", "S7 分片 23 落地", SEC_WBK,
     ("  改成 md5 差集现取＋只陈述\"两版逐字节不同\"。§26.27 自纠 **21** 条不变。下一项 **#18**（CSV 时间轴 opt-in）。",
      "eof")),
]

V["TIME_SHORT"] = sh("date", "+%H:%M")
ALL_TMPL = "".join(s for _n, _p, _r, _m, s, _a in LEGS)
missing = sorted(set(re.findall(r"@@([A-Za-z0-9_]+)@@", ALL_TMPL)) - set(V))
assert not missing, "模板有未取值的字段：" + str(missing)

report = []
for name, path, rel, marker, sec, (anchor, mode) in LEGS:
    txt = open(path, encoding="utf-8").read()
    if marker in txt:
        report.append("SKIP " + rel + "：标记已存在 ⇒ 该腿已落，不重复追加")
        continue
    n = txt.count(anchor)
    assert n == 1, rel + "：锚点出现 " + str(n) + " 次（须 1）⇒ 拒绝写盘：" + repr(anchor[:40])
    if mode == "eof":
        assert txt.rstrip().endswith(anchor.rstrip()), rel + "：锚点不在文件末尾 ⇒ 结构变了，拒绝盲写"
        i = len(txt)
    else:
        i = txt.index(anchor) + len(anchor)
        assert txt[i:].lstrip().startswith("---"), rel + "：插入点后不是分隔线 ⇒ 结构变了，拒绝盲写"
        g = txt.count(GLUE)
        assert g == GLUE_OV, rel + "：粘连存量与普查值不符（现读 " + str(g) + " 须 " + GLUE_OV + "）"
        txt = txt.replace(GLUE, "。\n- **2026-10-09（S7 分片 22 落地）")
        assert txt.count(GLUE) == 0, rel + "：修复后仍残留粘连"
        report.append("FIX " + rel + "：粘连 " + str(g) + " 处 → 0 处（mid 模式的存量后果）")
        i = txt.index(anchor) + len(anchor)
    rendered = ("\n" + sec) if mode == "mid" else sec
    for k, v in V.items():
        rendered = rendered.replace("@@" + k + "@@", v)
    if "@@" in rendered or "＠" in rendered:
        print("【残留下钻】", rel)
        for l in rendered.splitlines():
            if "@@" in l or "＠" in l:
                print("  残留行｜", l)
        raise SystemExit(3)
    if mode == "eof":
        new = txt.rstrip("\n") + "\n" + rendered
    else:
        new = txt[:i] + rendered + txt[i:]
    assert mode == "eof" or new[i] == "\n", rel + "：插入块首字符不是换行 ⇒ 会重演粘连"
    if mode == "mid":
        # 正对照：把插入块的换行去掉，同一处下标必须**不是**换行（否则上面那条断言是空转闸门）
        assert (txt[:i] + sec + txt[i:])[i] != "\n", "正对照失效：漏写换行的插入没被抓住 ⇒ 断言无分辨力"
    if WRITE:
        open(path, "w", encoding="utf-8").write(new)
        back = open(path, encoding="utf-8").read()
        assert back == new, rel + "：写后回读 ≠ 预期全文"
        assert back.count(marker) == 1, rel + "：标记出现 " + str(back.count(marker)) + " 次"
        assert back.count(GLUE) == 0, rel + "：写盘后仍有粘连"
        if rel.endswith("开发日志.md"):
            head = back.split(marker, 1)[0]
            assert head.count(MK_T) == 0, "(t) 标记在段前就出现 ⇒ 分段口径破了"
        report.append("WRITE " + rel + "：" + str(txt.count("\n")) + " → " + str(back.count("\n"))
                      + " 行（＋" + str(back.count("\n") - txt.count("\n")) + "），标记 1 次")
    else:
        report.append("PRECHECK " + rel + "：" + str(txt.count("\n")) + " → "
                      + str(new.count("\n")) + " 行（未写盘）")

print("\n".join(report))
print("读数：回执 %s 字节 BOFF=%s cmp=%s/%s/%s｜落地 %s blobs 闭合 %s==%s+%s=%s｜本片 %s/%s MATCH｜%s"
      % (V["RSIZE"], V["BOFF"], V["CMP0"], V["CMP1"], V["CMP2"], V["NB"], V["NB"], V["NP"],
         V["NA"], V["EXPECT"], V["SLICE_MATCH"], V["NFILES"], V["VERDICT"]))
print("尺子 md5 现算份数＝%s（前 12＝%s）｜自纠计数 %s→%s｜粘连普查 总览 %s／其余 %s"
      % (V["NRULE"], V["MDRULE"], V["PREVCNT"], V["CNT"], V["GLUE_OV"], V["GLUE_OTHER"]))
print("md5 现比 /tmp 同名件＝同 %s ／异 %s（%s）／缺 %s，共 %s｜GPU 我的进程＝ %s/%s｜顶格字段 %s｜dirty(src+tests)=%s"
      % (V["MDSAME"], V["MDIF"], V["MDIFES"], V["MDAB"], V["NA_SLICE"], V["NVME"], V["NVPID"],
         V["NDEF"], V["DIRTY"]))
if not WRITE:
    print("（预检模式：加 --write 才落盘。）")
