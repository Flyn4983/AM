#!/usr/bin/env python3
"""分片 21 回执之后的五条归档腿（开发日志 (r) ＋计划文档 ＋总览 ＋两条 .workbuddy）。

正文一律用**普通三引号字面量**＋`@@K@@` 占位，值全部由命令现取 ⇒ 不让正文穿过任何求值器
（本轮 (q) 的成因就是 f-string 吃掉了 `{32}`）。每腿都带锚点预检、写后回读、幂等（已有 (r) 段则跳过）。
"""
import os
import re
import subprocess
import sys

REPO = "/home/shy/桌面/2026年开发-显式冲击有限元-物质点法-sph-fvm+am/AM-Qoder"
D = os.path.join(REPO, "docs/evidence/2026-10-09")
REC = os.path.join(D, "am_s7_verify21.log")
RAW = os.path.join(D, "_raw_verify21.out")
WRITE = "--write" in sys.argv


def sh(*a):
    return subprocess.run(a, cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def md5(p):
    return subprocess.run(["md5sum", p], capture_output=True, text=True).stdout.split()[0]


raw_b = open(REC, "rb").read()
boff = int(re.search(rb"BODY_OFFSET=(\d+)\n", raw_b).group(1))
head_b, body_b = raw_b[:boff], raw_b[boff:]
hdr_txt, body_txt = head_b.decode("utf-8"), body_b.decode("utf-8")
assert body_b == open(RAW, "rb").read(), "回执正文区与 /tmp 原始 stdout 不逐字节相同"


def cmp_rc(l, r):
    return subprocess.run(["cmp", "-i", "%d:%d" % (l, r), REC, RAW], capture_output=True).returncode


C0, C1, C2 = cmp_rc(boff, 0), cmp_rc(boff + 1, 0), cmp_rc(boff, 1)
assert (C0, C1, C2) == (0, 1, 1), f"cmp 三联读数不齐：同偏移须 0、左右 +1 须 1，实得 {C0}/{C1}/{C2}"

V = {
    "RSIZE": str(os.path.getsize(REC)),
    "BOFF": str(boff),
    "C0": str(C0), "C1": str(C1), "C2": str(C2),
    "HEADSHORT": sh("git", "rev-parse", "--short", "HEAD"),
    "HEADFULL": sh("git", "rev-parse", "HEAD"),
    "PARENTSHORT": sh("git", "rev-parse", "--short", "HEAD^"),
    "MDREC": md5(REC)[:12],
    "MDHDR": md5(os.path.join(D, "am_s21_hdr.py"))[:12],
    "MDRULE": ("相同" if md5(os.path.join(D, "am_s7_verify21.py")) == md5(os.path.join(D, "am_s7_verify18.py"))
               else "不同"),
    "NOW": sh("date", "+%F %T %z"),
    # 正文读数：一律正则现取
    "ENTRIES": re.search(r"tree 条目 (\d+) = ", body_txt).group(1),
    "NB": re.search(r"blob (\d+) \+ tree", body_txt).group(1),
    "NTREE": re.search(r"blob \d+ \+ tree (\d+)", body_txt).group(1),
    "NP": re.search(r"父提交 = (\d+)", body_txt).group(1),
    "NA": re.search(r"新增 (\d+) \+ 修改", body_txt).group(1),
    "NM": re.search(r"新增 \d+ \+ 修改 (\d+)", body_txt).group(1),
    "EXPECT": re.search(r"\+ 新增 (\d+) = (\d+)", body_txt).group(2),
    "DL": re.search(r"本地独有 (\d+)/", body_txt).group(1),
    "DR": re.search(r"远端独有 (\d+)/", body_txt).group(1),
    "MS": re.search(r"blob SHA 不符 (\d+)/", body_txt).group(1),
    "NSHARED": re.search(r"blob SHA 不符 \d+/(\d+)", body_txt).group(1),
    "NFILES": re.search(r"本片文件 (\d+) 个", body_txt).group(1),
    "SLICE": re.search(r"MATCH (\d+)/", body_txt).group(1),
    "VERDICT": re.search(r"LANDING_VERDICT = (.+)", body_txt).group(1),
    # 表头里的现比读数（从回执表头正则现取，不重跑生成器）
    "MDSAME": re.search(r"相同 (\d+) 件／不同 (\d+) 件", hdr_txt).group(1),
    "MDIF": re.search(r"相同 \d+ 件／不同 (\d+) 件", hdr_txt).group(1),
    "MDIFES": re.search(r"不同 \d+ 件（(.+?)）", hdr_txt).group(1),
    "MDAB": re.search(r"已无同名件 (\d+) 件", hdr_txt).group(1),
    "NV": re.search(r"compute 进程 (\d+) 个", hdr_txt).group(1),
    "NVME": re.search(r"所留 (\d+)/\d+ 个", hdr_txt).group(1),
    "RC2": re.search(r"rc=(\d+)、stderr (\d+) 字节", hdr_txt).group(1),
    "ERRB2": re.search(r"rc=(\d+)、stderr (\d+) 字节", hdr_txt).group(2),
    "NDEF": str(sum(1 for l in hdr_txt.splitlines()
                    if re.match(r"^\S.*?\s*:", l) and not l.startswith("BODY_OFFSET="))),
}
assert int(V["MDSAME"]) + int(V["MDIF"]) + int(V["MDAB"]) == int(V["NA"]), "三分类之和不等于新增数"
# 档案里从未印过那句工作笔记（**真分段**：在 (r) 标记处切开，只数标记之前的那段——
# 早期写法是在追加前整份计数，靠"那时还没有 (r) 段"这个时序成立；预检重跑时 (r) 已在文件里，
# 整份计数就会把自己算进去（现测整份＝3、分段＝0），所以计数口径必须与写后那条断言一致）
DEV = os.path.join(REPO, "docs/开发日志.md")
MK_R = "### 26.27 (r)"
dev_before = open(DEV, encoding="utf-8").read().split(MK_R, 1)[0]
V["GB1"] = str(dev_before.count("逐字节副本"))
V["GB2"] = str(dev_before.count("10 件逐字节副本"))

SEC_R = """
### 26.27 (r) 分片 21 的落地回执＝一把尺子量两片；四条写盘前拦下的自身缺陷（计 +0）

- **产物**：回执 `docs/evidence/2026-10-09/am_s7_verify21.log`（@@RSIZE@@ 字节，md5 前 12＝@@MDREC@@，
  正文区起点 BODY_OFFSET＝@@BOFF@@）＋表头生成器 `am_s21_hdr.py`（md5＝@@MDHDR@@）＋尺子副本
  `am_s7_verify21.py`（与 `am_s7_verify18.py` 现比 md5＝**@@MDRULE@@**）。
- **落地读数（引自回执正文区，非转述）**：远端 main＝`@@HEADSHORT@@`＝本地 HEAD；`truncated=False`、
  tree 条目 @@ENTRIES@@＝blob @@NB@@＋tree @@NTREE@@；闭合式 **远端 @@NB@@ == 父提交 @@NP@@ ＋ 本片新增 @@NA@@ ＝ @@EXPECT@@ -> PASS**；
  双向路径差集 **@@DL@@/@@NB@@** 与 **@@DR@@/@@NB@@**；全库共有项 SHA 不符 **@@MS@@/@@NSHARED@@**；
  本片单列复核 **@@SLICE@@/@@NFILES@@ MATCH**（A @@NA@@／M @@NM@@）；`LANDING_VERDICT = @@VERDICT@@`。
- **回执自身的复核**：`cmp -i @@BOFF@@:0` rc＝**@@C0@@**（须 0），左偏移 +1 rc＝**@@C1@@**、右偏移 +1 rc＝**@@C2@@**（各须非 0）
  ⇒ 正文区确是那次核验 stdout 的逐字节后缀，而不是我另写的一份结论；回执表头自记的复跑读数＝rc **@@RC2@@**、
  stderr **@@ERRB2@@** 字节、复跑 stdout 与正文逐字节相同（生成器里有断言），顶格字段行数＝**@@NDEF@@** 行、
  表头无 ASCII 残留、无全角 at 号。
- **同片溯源（现比，不抄任何档案里的件数）**：本片新增 @@NA@@ 件里与 /tmp 同名件 md5 **相同 @@MDSAME@@ 件／不同 @@MDIF@@ 件
  （@@MDIFES@@）／/tmp 已无同名件 @@MDAB@@ 件**，三分类之和＝@@NA@@（闭合）。唯一"不同"的那件是
  `am_t35_cpu_evidence.py`：/tmp 那份是**单路判据的前一版**，档案里是产出同目录 `.log` 的**四路版**（差异见该 `.log` 的 E0 自述）。
  **这条推翻我自己本轮工作笔记里的「10 件逐字节副本」**：现比只有 @@MDSAME@@ 件。好在档案从未印过那句
  （`docs/开发日志.md` 里 `逐字节副本` 出现 **@@GB1@@** 次、`10 件逐字节副本` **@@GB2@@** 次，均为写 (r) 段之前的计数）
  ⇒ 无需更正档案，但要登记一条：**压缩摘要里的数不是实测数，凡要进档案必须现取。**
- **四处写盘前被拦下的自身缺陷**（都在"仪器／复述"侧，未污染任何档案正文 ⇒ 自纠计数 **+0**）：
  1. 表头曾按记忆手填 `ENTRIES＝672` ⇒ 改成正则从正文现取。手填数字是本工程头号禁忌（数必须来自命令）。
  2. 占位符笔误两处：一处把成对标记写成「半角双 at＋NA＋**单个** at」（少了尾部那个），这种能被"写盘前
     不得残留半角双 at"抓到；另一处误用**全角 at 号**——那是两条检查**都看不见**的（对账只认半角成对标记，
     残留检查也只认半角双 at）⇒ 补一道「表头不得含全角 at 号」。**"检查看不见的失败"才是真风险**：本条注文
     故意不写出那两种坏写法本身，否则订正段会把自己的残留检查判红（(q) 的同族），改用文字描述其形状。
  3. 控制 D 的读数取错捕获组（`group(2)` 是分母），差点把正文的「SHA 不符 0/@@NFILES@@」印成「@@SLICE@@/@@NFILES@@」
     ＝把"全过"说成"全不符"，一个**方向相反**的假信号。修＝先数清那行有几组括号，再取 `group(3)`。
  4. 模板与取值表不同步：字段改名后模板仍留着一个对不上取值表的占位符（半角双 at＋NBA＋半角双 at），
     被新增的 `missing` 对账断言（模板字段 ⊆ vals）当场拦下。
     ⇒ 复述段口径固定为：**每个数都从正文正则现取；模板字段与取值表必须对账。**
- **GPU／钉住**：生成器运行在 `CUDA_VISIBLE_DEVICES=`（空串）下，现取卡上 compute 进程 **@@NV@@** 个、
  由我的解释器（/proc/PID/cmdline 含 /tmp/amvenv）所留 **@@NVME@@/@@NV@@**；本件只跑 git 与 GitHub API，不 import jax。
- **自纠计数**：§26.27 仍为 **20** 条（(r) 的四条均被写盘前闸门拦下；与 (q) 那种"落进正文"的 +1 分界照旧）。
- **排期不变**：下一块＝**分片 22**（把回执与生成器一并入库）→ 代码腿 **#18**（路径程序驱动热源，CSV 时间轴
  只能做成 **opt-in**）→ #27＋#33 → #10/A3；#40 → #39；#29／#30 仍等用户裁决。落笔 @@NOW@@（`date` 现取）。
"""

SEC_PLAN = """
- **分片 21 落地回执（@@NOW@@ 现取，CPU 钉住）**：远端 main＝`@@HEADSHORT@@`（＝分片 21 提交本身），
  blob **@@NB@@**＝本地、tree 条目 @@ENTRIES@@＝@@NB@@＋@@NTREE@@、闭合式 **@@NB@@ == @@NP@@ ＋ 新增 @@NA@@ ＝ @@EXPECT@@** -> PASS、
  双向路径差集 **@@DL@@／@@DR@@（各 /@@NB@@）**、全库 SHA 不符 **@@MS@@/@@NSHARED@@**、本片 **@@SLICE@@/@@NFILES@@ MATCH**
  ⇒ **#35 的代码腿＋九道闸门证据件＋四路 CPU 证据件已全部在库**（`Flyn4983/AM` 公开，main＝`@@HEADSHORT@@`）。
  回执＝`docs/evidence/2026-10-09/am_s7_verify21.log`；分片 22＝这份回执与其生成器本身。
- **排期照旧**：**#18** → #27＋#33 → #10/A3；#40 → #39；#29／#30 等裁决；红线不变（A0 不放宽、#27 不换指标、
  无新增 skip/xfail、性能数字只在 GPU 窗口测；当前「不动 GPU」⇒ 无吞吐论断）。
"""

SEC_OV = """
- **2026-10-09（S7 分片 21 回执）＝落地证明复用同一把尺子**。`am_s7_verify21.py` 与分片 18 那份 **md5 现比＝@@MDRULE@@**
  （＝同一仪器量多片），读数：远端 main＝`@@HEADSHORT@@`＝本地 HEAD、tree 条目 @@ENTRIES@@＝blob **@@NB@@**＋tree @@NTREE@@、
  闭合式 **@@NB@@ == 父提交 @@NP@@ ＋ 本片新增 @@NA@@ ＝ @@EXPECT@@**、双向差集 **@@DL@@/@@NB@@** 与 **@@DR@@/@@NB@@**、
  全库 SHA 不符 **@@MS@@/@@NSHARED@@**、本片 **@@SLICE@@/@@NFILES@@ MATCH**、`LANDING_VERDICT＝@@VERDICT@@`；
  回执正文区由 `cmp -i @@BOFF@@:0`（rc＝@@C0@@，两侧各 +1 → @@C1@@/@@C2@@）证明是那次 stdout 的逐字节后缀。
  **一条口径进步**：回执表头每个数都从正文正则现取（手填 672／取错捕获组／模板与取值表不对账这三类在写盘前被拦），
  并新增「表头不得含全角 at 号」检查——全角占位符是"残留检查看不见的那类失败"。
  同片溯源现比＝与 /tmp 同名件 **同 @@MDSAME@@／异 @@MDIF@@（@@MDIFES@@）／缺 @@MDAB@@**，和＝@@NA@@；
  这推翻了我工作笔记里的「10 件逐字节副本」，档案从未印过该句（开发日志分段计数＝@@GB1@@），故无需更正。
  §26.27 自纠仍 **20** 条（四条均为写盘前拦下 ⇒ +0）。GPU 现取：卡上 compute 进程 @@NV@@ 个、我的＝**@@NVME@@/@@NV@@**。
"""

SEC_DAY = """
### 11:5x–12:0x 分片 21 落地回执（`am_s7_verify21.log`）＋分片 22 入库
回执四步全过：远端 main＝`@@HEADSHORT@@`＝本地 HEAD；tree 条目 @@ENTRIES@@＝blob **@@NB@@**＋tree @@NTREE@@；
闭合式 **@@NB@@ == @@NP@@ ＋ @@NA@@ ＝ @@EXPECT@@** -> PASS；双向差集 **@@DL@@/@@NB@@**、**@@DR@@/@@NB@@**；
全库 SHA 不符 **@@MS@@/@@NSHARED@@**；本片 **@@SLICE@@/@@NFILES@@ MATCH**；`LANDING_VERDICT＝@@VERDICT@@`。
`cmp -i @@BOFF@@:0` rc＝@@C0@@、左 +1＝@@C1@@、右 +1＝@@C2@@ ⇒ 正文区逐字节＝那次 stdout。
尺子复用：`am_s7_verify21.py` 与 `am_s7_verify18.py` md5 现比＝**@@MDRULE@@**。
四条写盘前拦下的仪器缺陷（+0）：手填 672／半角占位符少一个尾 at＋**全角 at 号**（后者残留检查看不见，已补检查）／
控制 D 取错组（差点把"0 不符"印成"@@SLICE@@ 不符"＝方向相反的假信号）／模板留着一个对不上取值表的改名残留（被对账断言拦）。
现比溯源＝同 @@MDSAME@@／异 @@MDIF@@（@@MDIFES@@）／缺 @@MDAB@@（和＝@@NA@@）⇒ **工作笔记里那句「10 件逐字节副本」是错的**，
档案从未印过它（分段计数 @@GB1@@）。教训：**压缩摘要里的数不是实测数**，进档案必须现取。
自纠计数 §26.27 仍 **20** 条。GPU：现取 compute 进程 @@NV@@ 个，我的解释器所留 **@@NVME@@/@@NV@@**；本轮全部 CPU 钉住。
"""

SEC_WBK = """
- **2026-10-09 午（S7 分片 21 回执＋分片 22）**：远端 main＝`@@HEADSHORT@@`（@@NB@@ blobs），闭合式
  `@@NB@@ == 父 @@NP@@ ＋ 新增 @@NA@@ ＝ @@EXPECT@@`、双向差集 @@DL@@/@@NB@@ 与 @@DR@@/@@NB@@、全库 SHA 不符 @@MS@@/@@NSHARED@@、
  本片 @@SLICE@@/@@NFILES@@ MATCH ⇒ #35 全块在库。回执＝`docs/evidence/2026-10-09/am_s7_verify21.log`
  （正文区由 `cmp -i @@BOFF@@:0` rc＝@@C0@@ ＋两侧 +1 各红 证明是逐字节后缀）。
  **新口径（复述段禁造数）**：回执表头每个读数都从正文正则现取；模板字段与取值表必须对账；
  残留检查要**同时**认半角成对标记与全角 at 号（后者是"检查看不见的失败"）。
  **新诚实边界**：压缩摘要/工作笔记里的数不是实测数——本轮笔记的「10 件逐字节副本」现比只有 @@MDSAME@@ 件
  （异 @@MDIF@@＝@@MDIFES@@，/tmp 存的是被取代的前一版；缺 @@MDAB@@），且该句从未进过档案。
  §26.27 自纠 **20** 条不变（本轮四条均被写盘前闸门拦下 ⇒ +0）。下一项 **#18**（CSV 时间轴 opt-in）。
"""

LEGS = [
    ("开发日志 (r)", DEV, "docs/开发日志.md", MK_R, SEC_R),
    ("计划文档 §U 尾", os.path.join(REPO, "docs/项目评估与下一步计划_2026-10-06.md"),
     "docs/项目评估与下一步计划_2026-10-06.md", "分片 21 落地回执", SEC_PLAN),
    ("总览进度段", os.path.join(REPO, "docs/项目开发总览.md"), "docs/项目开发总览.md",
     "S7 分片 21 回执", SEC_OV),
    (".workbuddy 当日", os.path.join(REPO, ".workbuddy/memory/2026-10-09.md"),
     ".workbuddy/memory/2026-10-09.md", "分片 21 落地回执", SEC_DAY),
    (".workbuddy MEMORY", os.path.join(REPO, ".workbuddy/memory/MEMORY.md"),
     ".workbuddy/memory/MEMORY.md", "S7 分片 21 回执＋分片 22", SEC_WBK),
]

ANCHORS = {
    "docs/开发日志.md": ("  **#40 → #39**；#29／#30 等用户裁决。生成器当前 md5＝a9c891466868，落笔 11:36（`date` 现取）。", "eof"),
    "docs/项目评估与下一步计划_2026-10-06.md": ('  "支持 LSF/DED 正向模拟/数字孪生"。', "eof"),
    "docs/项目开发总览.md": ("  （全量面 11:00:37 已自然结束）⇒ 归还无需停我的进程。", "mid"),
    ".workbuddy/memory/2026-10-09.md": ("教训＝正文不许穿过求值器：`echo` 会吃反引号（第 6 次），f-string 会吃花括号（第 1 次，本轮）。", "eof"),
    ".workbuddy/memory/MEMORY.md": ("教训＝正文不许穿过求值器：`echo` 会吃反引号（第 6 次），f-string 会吃花括号（第 1 次，本轮）。", "eof"),
}

report = []
for name, path, rel, marker, sec in LEGS:
    txt = open(path, encoding="utf-8").read()
    if marker in txt:
        report.append("SKIP " + rel + "：标记已存在 ⇒ 该腿已落，不重复追加")
        continue
    anchor, mode = ANCHORS[rel]
    n = txt.count(anchor)
    assert n == 1, rel + "：锚点出现 " + str(n) + " 次（须 1）⇒ 拒绝写盘：" + repr(anchor[:40])
    unknown = sorted(set(re.findall(r"@@([A-Za-z0-9_]+)@@", sec)) - set(V))
    assert not unknown, rel + "：段里有对不上取值表的占位符 " + str(unknown)
    rendered = sec
    for k, v in V.items():
        rendered = rendered.replace("@@" + k + "@@", v)
    # 严格残留闸门（半角成对标记＋全角 at 号各一条）。注文里**不写**坏写法本身——写了就会让自己的检查
    # 判红（(q) 的同族），所以这里不需要任何豁免；需要豁免的那版在预检里被自己拦下过。
    if "@@" in rendered or "＠" in rendered:
        print("【残留下钻】", rel)
        for l in rendered.splitlines():
            if "@@" in l or "＠" in l:
                print("  残留行｜", l)
        raise SystemExit(3)
    if mode == "eof":
        assert txt.rstrip().endswith(anchor.rstrip()), rel + "：锚点不在文件末尾 ⇒ 结构变了，拒绝盲写"
        new = txt.rstrip("\n") + "\n" + rendered
    else:
        i = txt.index(anchor) + len(anchor)
        assert txt[i:].lstrip().startswith("---"), rel + "：插入点之后不是分隔线 ⇒ 结构变了，拒绝盲写"
        new = txt[:i] + rendered + txt[i:]
    lines_before = txt.count("\n")
    if WRITE:
        open(path, "w", encoding="utf-8").write(new)
        back = open(path, encoding="utf-8").read()
        assert back == new, rel + "：写后回读 ≠ 预期全文"
        assert back.count(marker) == 1, rel + "：标记出现 " + str(back.count(marker)) + " 次"
        head = back.split(marker, 1)[0]
        if rel.endswith("开发日志.md"):
            assert head.count("逐字节副本") == int(V["GB1"]), "分段计数漂移 ⇒ (r) 段之前的正文被动过"
        report.append("WRITE " + rel + "：" + str(lines_before) + " → " + str(back.count("\n"))
                      + " 行（＋" + str(back.count("\n") - lines_before) + "），标记 1 次")
    else:
        report.append("PRECHECK " + rel + "：" + str(lines_before) + " → " + str(new.count("\n"))
                      + " 行（未写盘）")

print("\n".join(report))
print("读数：回执 %s 字节 BOFF=%s cmp=%s/%s/%s｜落地 %s blobs 闭合 %s==%s+%s=%s｜本片 %s/%s MATCH｜"
      "VERDICT=%s" % (V["RSIZE"], V["BOFF"], V["C0"], V["C1"], V["C2"], V["NB"], V["NB"], V["NP"],
                      V["NA"], V["EXPECT"], V["SLICE"], V["NFILES"], V["VERDICT"]))
print("现比 /tmp：同 %s／异 %s（%s）／缺 %s，和=%s｜开发日志分段计数：逐字节副本=%s，10 件逐字节副本=%s"
      % (V["MDSAME"], V["MDIF"], V["MDIFES"], V["MDAB"], V["NA"], V["GB1"], V["GB2"]))
if not WRITE:
    print("（预检模式：加 --write 才落盘。）")
