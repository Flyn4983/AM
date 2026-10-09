#!/usr/bin/env python3
"""分片 22 落地回执之后的五条归档腿（开发日志 (s) ＋计划文档 ＋总览 ＋两条 .workbuddy）。

与上一轮的 am_s22_docs.py 同构，口径进步两处（都是上一轮 (r) 段里立下的）：
① 所有数从**回执**里正则现取（表头读数从表头取、正文读数从正文取），一个也不手填；
② "不同件"的名字与来历一律不写死：只印 md5 差集现取到的名字，故事不进归档。
每腿带锚点预检（出现次数须 1、eof 须真在末尾、mid 须紧跟分隔线）、写后回读、幂等。
"""
import os
import re
import subprocess
import sys

REPO = "/home/shy/桌面/2026年开发-显式冲击有限元-物质点法-sph-fvm+am/AM-Qoder"
D = os.path.join(REPO, "docs/evidence/2026-10-09")
REC = os.path.join(D, "am_s7_verify22.log")
RAW = os.path.join(D, "_raw_verify22.out")
ERRF = os.path.join(D, "_raw_verify22.err")
RULE = os.path.join(D, "am_s7_verify22.py")
HDR = os.path.join(D, "am_s22_hdr.py")
DEV = os.path.join(REPO, "docs/开发日志.md")
MK_S = "### 26.27 (s)"
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

V = {
    "NOW": sh("date", "+%F %T %z"),
    "HEADS": sh("git", "rev-parse", "--short", "HEAD"),
    "HEADF": sh("git", "rev-parse", "HEAD"),
    "PARENTS": sh("git", "rev-parse", "--short", "HEAD^"),
    "RSIZE": str(os.path.getsize(REC)),
    "MDREC": md5(REC)[:12],
    "MDHDR": md5(HDR)[:12],
    "MDRULE": md5(RULE)[:12],
    "RULE18": ("逐字节相同" if md5(RULE) == md5(os.path.join(D, "am_s7_verify18.py")) else "不同"),
    "RULE21": ("逐字节相同" if md5(RULE) == md5(os.path.join(D, "am_s7_verify21.py")) else "不同"),
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
}
assert V["VERDICT"] == "LANDING_VERDICT = ALL CHECKS PASS", "正文判读行不是全过 ⇒ 本轮回执不能当落地证明归档"
assert int(V["NB"]) == int(V["NP"]) + int(V["NA"]), "闭合式在正文里就不成立"
assert V["MDIFES"] != "无" or int(V["MDIF"]) == 0, "不同件的名单与件数对不上"
assert int(V["NDEF"]) == 16, "表头顶格字段数不是 16，实得 " + V["NDEF"]

SEC_S = """
### 26.27 (s) 分片 22 落地＝回执链第二环：一把尺子量三片；上一轮的教训落成两处代码

- **产物**：尺子 `docs/evidence/2026-10-09/am_s7_verify22.py`（md5 前 12＝@@MDRULE@@，与
  `am_s7_verify18.py`／`am_s7_verify21.py` 现比＝@@RULE18@@／@@RULE21@@）＋落地回执
  `am_s7_verify22.log`（@@RSIZE@@ 字节，md5＝@@MDREC@@，正文区起点 BODY_OFFSET＝@@BOFF@@）＋表头生成器
  `am_s22_hdr.py`（md5＝@@MDHDR@@）＋核验件原始 stdout/stderr（@@RAWSZ@@／@@ERRSZ@@ 字节，
  md5＝@@RAWMD@@／@@ERRMD@@，`cp -p` 归档 ⇒ mtime 仍是原次运行时刻 @@DSTART@@）。
- **落地读数（引自回执正文区，非转述）**：远端 main＝`@@HEADS@@`＝本地 HEAD；`truncated=False`、
  tree 条目 @@ENTRIES@@＝blob @@NB@@＋tree @@NTREE@@；闭合式 **远端 @@NB@@ == 父提交 @@NP@@ ＋ 新增 @@NA@@ ＝ @@EXPECT@@ -> PASS**；
  双向路径差集 **@@DL@@/@@NB@@** 与 **@@DR@@/@@NB@@**；全库共有项 SHA 不符 **@@MS@@/@@NSHARED@@**；
  本片单列复核 **@@SLICE_MATCH@@/@@NFILES@@ MATCH**（A @@NA@@／M @@NM@@）；`@@VERDICT@@`。
- **四个正对照（同一条运行里印出，全部能红）**：A 远端一条 sha 首位改 f ⇒ 不符 @@CA@@/@@CBA@@；
  B 远端删一条路径 ⇒ 本地独有 @@CB1@@/@@CDB1@@、远端独有 @@CB2@@/@@CDB2@@；
  C 闭合式右边 +1 ⇒ @@NB@@ == @@NBP@@ 判 FAIL；D 远端删本片一条路径 ⇒ 本片缺失 @@CD1@@/@@NFILES@@、SHA 不符 @@CD2@@/@@NFILES@@。
- **回执自身的复核**：仓库根跑 `cmp -i @@BOFF@@:0 docs/evidence/2026-10-09/am_s7_verify22.log
  docs/evidence/2026-10-09/_raw_verify22.out` rc＝**@@CMP0@@**（须 0），左偏移 +1 rc＝**@@CMP1@@**、
  右偏移 +1 rc＝**@@CMP2@@**（各须非 0）⇒ 正文区确是那次核验 stdout 的逐字节后缀；
  与上一片不同，**这条复核的两侧都在仓库里**，不再依赖易失的 /tmp（上一片为此重生成过一次表头，本轮直接按这个口径写）。
  表头自证：复跑 rc＝@@RC2@@、stderr @@ERRB2@@ 字节、顶格字段 **@@NDEF@@** 行、表头无半角成对标记残留、无全角 at 号。
- **GPU／钉住**：表头生成时现取卡上 compute 进程 **@@NVPID@@** 个，其中我的解释器（/proc/PID/cmdline 含 /tmp/amvenv）
  所留 **@@NVME@@/@@NVPID@@**（须 0）。上一片那个读数是 6 个，本片 @@NVPID@@ 个——**别人程序的生命周期**，
  我只登记读数、不解释它，也不碰它。
- **上一轮的教训落成两处代码**（不是新缺陷 ⇒ 自纠计数 **+0**）：
  1. 表头里那句"唯一不同的一件是某某某"被删掉了：那是上一片的**具体故事**，本片差集现取到的是
     「同 @@MDSAME@@／异 @@MDIF@@（@@MDIFES@@）／缺 @@MDAB@@，和＝@@NA_SLICE@@」，名字换了、故事就不成立
     （这次"不同"的那件是我自己写的归档写手——五腿落盘后我又就地订正了归档那份，/tmp 留的是订正前的）。
     ⇒ 现在表头只印 md5 差集现取到的名字＋"两版逐字节不同"这条能被证明的事实，来历不进表头。
     **教训：订正注文里的具体名字也是数，也要现取。**
  2. 原始 stdout/stderr 在生成表头**之前**就抄进档案目录，表头据此写明时点，并保留"/tmp 若还有同名件则 md5 须等"
     这道条件断言（件在才判 ⇒ 临时目录被清空也不会卡住以后的重跑）。
- **同片溯源（现比）**：本片新增 @@NA_SLICE@@ 件里与 /tmp 同名件 md5 相同 @@MDSAME@@ 件、不同 @@MDIF@@ 件、
  /tmp 已无同名件 @@MDAB@@ 件，三分类之和＝@@NA_SLICE@@（闭合）。这条只作辅助溯源，落地证明以上面正文区三条同读为准。
- **自纠计数**：§26.27 仍 **21** 条（本片两处都是"把上一轮拦下的形状写进仪器"，没有错数写盘后被拦）。
- **排期不变**：下一块＝代码腿 **#18**（路径程序驱动热源，CSV 时间轴只能做成 **opt-in**）→ #27＋#33 → #10/A3；
  #40 → #39；#29／#30 仍等用户裁决。落笔 @@NOW@@（`date` 现取）。
"""

SEC_PLAN = """
- **分片 22 落地（@@NOW@@ 现取，CPU 钉住）**：远端 main＝`@@HEADS@@`，blob **@@NB@@**＝本地、
  tree 条目 @@ENTRIES@@＝@@NB@@＋@@NTREE@@、闭合式 **@@NB@@ == @@NP@@ ＋ 新增 @@NA@@ ＝ @@EXPECT@@** -> PASS、
  双向差集 **@@DL@@/@@NB@@** 与 **@@DR@@/@@NB@@**、全库 SHA 不符 **@@MS@@/@@NSHARED@@**、
  本片 **@@SLICE_MATCH@@/@@NFILES@@ MATCH**（A @@NA@@／M @@NM@@）⇒ 分片 21 的回执与两份生成器都在库。
  回执＝`docs/evidence/2026-10-09/am_s7_verify22.log`。
"""

SEC_OV = """- **2026-10-09（S7 分片 22 落地）＝回执链第二环，一把尺子量三片**。`am_s7_verify22.py` 与分片 18／21 两份
  现比 md5＝**@@RULE18@@／@@RULE21@@**（＝同一仪器）。读数（引自回执正文区）：远端 main＝`@@HEADS@@`、
  tree 条目 @@ENTRIES@@＝blob **@@NB@@**＋tree @@NTREE@@、闭合式 **@@NB@@ == @@NP@@＋@@NA@@＝@@EXPECT@@**、
  双向差集 **@@DL@@/@@NB@@** 与 **@@DR@@/@@NB@@**、全库 SHA 不符 **@@MS@@/@@NSHARED@@**、
  本片 **@@SLICE_MATCH@@/@@NFILES@@ MATCH**、`@@VERDICT@@`；正文区由 `cmp -i @@BOFF@@:0`（两侧都在仓库里）
  rc＝**@@CMP0@@** ＋左右各 +1 判红 证明是逐字节后缀。
  **仪器进步两条**：表头不再写死"不同那件是谁"（名字也现取）；核验件的原始 stdout 在生成表头前就随片归档。
  GPU 现取：卡上 compute 进程 **@@NVPID@@** 个、我的＝**@@NVME@@/@@NVPID@@**。§26.27 自纠仍 **21** 条。
"""

SEC_DAY = """
### @@TIME_SHORT@@ 分片 22 落地回执（`am_s7_verify22.log`）
回执四步全过：远端 main＝`@@HEADS@@`＝本地 HEAD；tree 条目 @@ENTRIES@@＝blob **@@NB@@**＋tree @@NTREE@@；
闭合式 **@@NB@@ == @@NP@@ ＋ @@NA@@ ＝ @@EXPECT@@** -> PASS；双向差集 **@@DL@@/@@NB@@**、**@@DR@@/@@NB@@**；
全库 SHA 不符 **@@MS@@/@@NSHARED@@**；本片 **@@SLICE_MATCH@@/@@NFILES@@ MATCH**；`@@VERDICT@@`。
`cmp -i @@BOFF@@:0` rc＝@@CMP0@@、左 +1＝@@CMP1@@、右 +1＝@@CMP2@@ ⇒ 正文区逐字节＝那次 stdout，且两侧都在仓库里。
尺子复用：三份核验件 md5 现比＝@@RULE18@@／@@RULE21@@（前 12＝@@MDRULE@@）。
仪器进步两条：表头里"不同那件是谁"改成现取（上一片那句故事对这一片不成立）；原始 stdout 在生成表头前随片归档。
同片溯源＝同 @@MDSAME@@／异 @@MDIF@@（@@MDIFES@@）／缺 @@MDAB@@（和＝@@NA_SLICE@@）。
自纠计数 §26.27 仍 **21** 条。GPU：现取 compute 进程 @@NVPID@@ 个，我的解释器所留 **@@NVME@@/@@NVPID@@**；本轮全部 CPU 钉住。
"""

SEC_WBK = """
- **2026-10-09 午后（S7 分片 22 落地）**：远端 main＝`@@HEADS@@`（@@NB@@ blobs），闭合式
  `@@NB@@ == 父 @@NP@@ ＋ 新增 @@NA@@ ＝ @@EXPECT@@`、双向差集 @@DL@@/@@NB@@ 与 @@DR@@/@@NB@@、
  全库 SHA 不符 @@MS@@/@@NSHARED@@、本片 @@SLICE_MATCH@@/@@NFILES@@ MATCH ⇒ 分片 21 的回执链全部在库。
  回执＝`docs/evidence/2026-10-09/am_s7_verify22.log`（正文区由 `cmp -i @@BOFF@@:0` rc＝@@CMP0@@ ＋两侧 +1 各红
  证明是逐字节后缀，且这条复核的两个文件都在仓库里）。
  **新口径**：订正注文里的**具体名字也是数**——上一片表头写死"唯一不同的一件是某某"，本片差集换了名字 ⇒
  改成 md5 差集现取＋只陈述"两版逐字节不同"。§26.27 自纠 **21** 条不变。下一项 **#18**（CSV 时间轴 opt-in）。
"""

LEG_NAMES = ["开发日志 (s)", "计划文档", "总览", "当日", "MEMORY"]
LEGS = [
    ("开发日志 (s)", DEV, "docs/开发日志.md", MK_S, SEC_S,
     ("  #40 → #39；#29／#30 仍等用户裁决。落笔 2026-10-09 12:08:32 +0800，就地订正后定稿于 **16:04:20 +0800**（两次 `date` 现取）。", "eof")),
    ("计划文档", os.path.join(REPO, "docs/项目评估与下一步计划_2026-10-06.md"),
     "docs/项目评估与下一步计划_2026-10-06.md", "分片 22 落地", SEC_PLAN,
     ('  无新增 skip/xfail、性能数字只在 GPU 窗口测；当前「不动 GPU」⇒ 无吞吐论断）。', "eof")),
    ("总览", os.path.join(REPO, "docs/项目开发总览.md"), "docs/项目开发总览.md",
     "S7 分片 22 落地", SEC_OV,
     ("  GPU 现取：卡上 compute 进程 6 个、我的＝**0/6**。", "mid")),
    ("当日", os.path.join(REPO, ".workbuddy/memory/2026-10-09.md"),
     ".workbuddy/memory/2026-10-09.md", "分片 22 落地回执", SEC_DAY,
     ("这条落进了档案文件 ⇒ +1，与\"拦在写盘前/仪器设计阶段\"的 6 条 ＋0 分界在此）。GPU：现取 compute 进程 6 个，"
      "我的解释器所留 **0/6**；本轮全部 CPU 钉住。", "eof")),
    ("MEMORY", os.path.join(REPO, ".workbuddy/memory/MEMORY.md"),
     ".workbuddy/memory/MEMORY.md", "S7 分片 22 落地", SEC_WBK,
     ("  下一项 **#18**（CSV 时间轴 opt-in）。", "eof")),
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
    else:
        i = txt.index(anchor) + len(anchor)
        assert txt[i:].lstrip().startswith("---"), rel + "：插入点后不是分隔线 ⇒ 结构变了，拒绝盲写"
    rendered = sec
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
        i = txt.index(anchor) + len(anchor)
        new = txt[:i] + rendered + txt[i:]
    if WRITE:
        open(path, "w", encoding="utf-8").write(new)
        back = open(path, encoding="utf-8").read()
        assert back == new, rel + "：写后回读 ≠ 预期全文"
        assert back.count(marker) == 1, rel + "：标记出现 " + str(back.count(marker)) + " 次"
        if rel.endswith("开发日志.md"):
            head = back.split(marker, 1)[0]
            assert head.count(MK_S) == 0, "(s) 标记在段前就出现 ⇒ 分段口径破了"
        report.append("WRITE " + rel + "：" + str(txt.count("\n")) + " → " + str(back.count("\n"))
                      + " 行（＋" + str(back.count("\n") - txt.count("\n")) + "），标记 1 次")
    else:
        report.append("PRECHECK " + rel + "：" + str(txt.count("\n")) + " → "
                      + str(new.count("\n")) + " 行（未写盘）")

print("\n".join(report))
print("读数：回执 %s 字节 BOFF=%s cmp=%s/%s/%s｜落地 %s blobs 闭合 %s==%s+%s=%s｜本片 %s/%s MATCH｜%s"
      % (V["RSIZE"], V["BOFF"], V["CMP0"], V["CMP1"], V["CMP2"], V["NB"], V["NB"], V["NP"],
         V["NA"], V["EXPECT"], V["SLICE_MATCH"], V["NFILES"], V["VERDICT"]))
print("md5 现比 /tmp 同名件＝同 %s ／异 %s（%s）／缺 %s，共 %s｜GPU 我的进程＝ %s/%s｜顶格字段 %s"
      % (V["MDSAME"], V["MDIF"], V["MDIFES"], V["MDAB"], V["NA_SLICE"], V["NVME"], V["NVPID"], V["NDEF"]))
if not WRITE:
    print("（预检模式：加 --write 才落盘。）")
