#!/usr/bin/env python3
r"""分片 31 的**填值器**（＋带守卫的一次性写盘腿）：五张档案面的每个读数都由命令现取，具名替换后渲染。

四条口径（前三条是往轮用真事故换来的，第四条是本轮新踩的，本件把它们当闸门而不是注释）：
① **具名替换＋邻接断言**——位置式 %s 在「槽位数相等＋顺序单调」两道检查同时绿的情况下仍能把散文排错位，
   所以这里一律 `%(中文键)s`，且每个值必须由 `expect_near` 钉在自己的标签后面。
② **渲染结构门**＝按行配平全角 `（）`、剥内联码后配平 `**`、写盘前查目标末字节是换行、插入锚点在整份文件
   里恰好 1 处、渲染结果里不许残留 `%(`／`%（`／`@@`。
③ **节内唯一性**（＝表头生成器 `bullet_for` 的真实约束，不是我自己加的更严的前提）＝引用上一片回执名的
   顶级子条恰好 1 条、带引号的钉住口径登记恰好 1 条、承接的 LESSON 锚句恰好 1 条、本轮新学的锚句恰好 1 条；
   每条各配一条"再复述一次就必须变成 2 处"的正对照——没有正对照的唯一性断言是恒真式。
④ **本轮新增＝未跟踪清单的折叠口径**。`git status --porcelain` 默认 `-u normal` 会把**整目录都没被跟踪**
   的那种目录折成一行（本轮实测：新建的 `docs/evidence/2026-10-13/` 在默认口径下只印出目录名一行，
   加 `-uall` 才逐条印）⇒ 件数会凭空少算、角色分组会出现"落进 0 组"的那一件。本件的所有计数一律走
   `-uall`，并且**另外断言清单里没有以 `/` 结尾的条目**（那条形状就是折叠的证据，不是件）。

volatile 读数（未入库件数、卡上进程数、落笔 date）只进本轮散文，不写进任何可复跑件的 stdout。
默认只渲染到 `/tmp/out31/`；`--write` 才落盘，且落盘前后各数一次闭合式。
"""
import ast
import glob
import hashlib
import os
import re
import subprocess
import sys

REPO = "/home/shy/桌面/2026年开发-显式冲击有限元-物质点法-sph-fvm+am/AM-Qoder"
DEVLOG = os.path.join(REPO, "docs/开发日志.md")
OVERVIEW = os.path.join(REPO, "docs/项目开发总览.md")
PLAN = os.path.join(REPO, "docs/项目评估与下一步计划_2026-10-06.md")
ANCHOR = "## 7. 文档索引"
WBIDX = os.path.join(REPO, ".workbuddy/memory/MEMORY.md")
LESSON = "自审的方向性"          # 从分片 30 承接进来的那条口径（＝表头生成器要现取的锚句）
LESSON_NEW = "普查仪器没有第二出处"   # 本轮新学的锚句（下一片的自查件按这个短语回搜）
INTERP = "/tmp/amvenv/bin/python"
SELF_NAME = os.path.basename(os.path.realpath(__file__))   # 本件自己的名字（守卫要把它从"别的仪器"里摘出去）
OUT_DIR = "/tmp/out31"
TODAY = subprocess.run(["date", "+%F"], capture_output=True, text=True).stdout.strip()
EVID_NEW = "docs/evidence/" + TODAY
DAILY = os.path.join(REPO, ".workbuddy/memory/%s.md" % TODAY)
# 写盘守卫：这些名字**正在跑**就不许动档案（按 /proc/PID/cmdline 的整词比对，不按命令行子串猜）
TOKENS = {"am_s7_verify30.py", "am_s30_hdr.py", "am_s30_msg.py", "am_s7_verify31.py",
          "am_s30_archive_check.py", "am_s30_hdr_controls.py", "am_s30_msg_controls.py",
          "am_s30_hdr_after.py", "am_s30_hdr_after_controls.py", "am_s30_docs_landing.py",
          "am_s31_docs.py", "am_s31_msg.py", "am_s31_msg_controls.py",
          "s31_fill.py", "s31_write.py"}


def sh(*args):
    """**默认环境**跑（不钉 LC_ALL）——指纹口径只在默认 locale 下与前几片可比。"""
    return subprocess.run(args, cwd=REPO, capture_output=True, text=True).stdout


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def one(needle, text, what):
    hits = re.findall(needle, text, re.M)
    assert len(hits) == 1, "%s 命中 %d 处（须恰好 1）：%s" % (what, len(hits), hits[:3])
    return hits[0]


def named(label, text):
    """具名替换＋三处残留检查（`%（` 是全角括号打错位的具名槽，静默不替换）。"""
    assert "%（" not in text, (label + "：模板里有全角括号的具名槽（那种槽不会被替换，"
                              "会原样落进档案）")
    bare = [text[max(0, m.start() - 12):m.start() + 6]
            for m in re.finditer(r"%%|%\([A-Za-z0-9_]+\)s|%", text) if m.group(0) == "%"]
    assert not bare, (label + "：模板里有裸百分号（具名替换要求每处槽都写成 %("
                      "名)s 这一形状，字面百分号要写成双写；裸百分号要么当场报错、要么把后面的字当成格式符）前 5 处＝"
                      + str(bare[:5]))
    out = text % CTX
    assert "%(" not in out, "%s：渲染后残留未替换的具名槽" % label
    assert "@@" not in out, "%s：渲染后残留 @@ 占位符" % label
    return out


# ============================ 读数：全部由命令现取 ============================
HEAD = sh("git", "rev-parse", "HEAD").strip()
HEAD7 = sh("git", "rev-parse", "--short", "HEAD").strip()
PARENT7 = sh("git", "rev-parse", "--short", "HEAD^").strip()
SUBJ = sh("git", "log", "-1", "--format=%s").strip()
CDATE = sh("git", "log", "-1", "--format=%cd", "--date=iso").strip()
TIME = sh("date", "+%F %T %z").strip()
NCODE = len([l for l in sh("git", "-c", "core.quotePath=false", "status", "--porcelain",
                           "--", "src", "tests").splitlines() if l.strip()])
assert NCODE == 0, "生产树 src＋tests 有 %d 处改动 ⇒ 本片不是纯归档腿，那句散文要先改" % NCODE
SLICE = int(one(r"S7 分片 (\d+)", SUBJ, "HEAD 提交说明里的片号")) + 1
PREVSLICE = SLICE - 1
NEXTSLICE = str(SLICE + 1)

# 上一片回执的**目录由它自己被全树 glob 命中而定**（跨日期不硬写：本轮它在 10-12，本件在 10-13）
PREVLOG = "am_s7_verify%d.log" % PREVSLICE
_curl = sorted(os.path.relpath(p, REPO) for p in
               glob.glob(os.path.join(REPO, "docs/evidence/*/" + PREVLOG))
               if re.fullmatch(os.path.join("docs", "evidence", "[^/]+", PREVLOG),
                               os.path.relpath(p, REPO)))
assert len(_curl) == 1, "上一片回执在全树命中 %d 份（须 1）：%s ⇒ 引用哪一份都不唯一" % (
    len(_curl), _curl)
CURL = _curl[0]
EVID_OLD = os.path.dirname(CURL)
assert EVID_OLD != EVID_NEW, "上一片回执与本件同目录 ⇒ 本轮的跨日期登记要改"
print("== 现取：HEAD＝%s（父 %s｜提交时刻 %s）｜片号＝%d（由 HEAD 主题现取）｜上一片回执＝%s（目录 %s 由 glob 现取）"
      % (HEAD7, PARENT7, CDATE, SLICE, PREVLOG, EVID_OLD))

st = [l.rstrip("\n") for l in sh("git", "-c", "core.quotePath=false", "status", "--porcelain",
                                 "--untracked-files=all").splitlines() if l.strip()]
UNT = sorted(l[3:] for l in st if l.startswith("??"))
MOD = sorted(l[3:] for l in st if not l.startswith("??"))
NU, NM = len(UNT), len(MOD)
# 口径 ④：折叠出来的"目录行"以 / 结尾，它不是一件文件；真出现了就说明计数用的不是 -uall
folded = [p for p in UNT if p.endswith("/")]
assert not folded, "未跟踪清单里有折叠的目录行 %s ⇒ 计数口径不是 `-uall`，件数与分组都不可信" % folded
# 67 ＝ 未跟踪 62（上一片的落地半腿链 50＋本轮新起的当日纪实 1＋本件自己 1＋本件的预检／写盘两面与其四处红灯现场）＋已跟踪档案面 5
# 这四面**必须先存在再跑**：否则本件自己的 stdout 面会把计数推着走（预检与写盘两次落笔读出不同的数＝自指的形状，attempt1 现场）
assert NU + NM == 67, "落笔现数＝%d＋%d＝%d 处（本节的闭合式写的是 67）⇒ 先确认工作树有没有被动过、本件的面是否已先落盘" % (
    NU, NM, NU + NM)

ROLES = [
    ("回执链 5", r"^am_s7_verify%d\.(py|log)$|^_raw_verify%d\.(out|err)$|^am_s%d_hdr\.py$"
     % (PREVSLICE, PREVSLICE, PREVSLICE)),
    ("表头预检与写盘两面 4", r"^_raw_s%d_hdr_(precheck|write)\.(out|err)$" % PREVSLICE),
    ("表头控件面 3", r"^am_s%d_hdr_controls\.py$|^_raw_s%d_hdr_controls\.(out|err)$"
     % (PREVSLICE, PREVSLICE)),
    ("控件面红灯现场 4", r"^_raw_s%d_hdr_controls\.attempt\d-[a-z0-9-]+\.(out|err)$" % PREVSLICE),
    ("只读仪器与其各面 9", r"^am_s%d_hdr_after\.py$|^_raw_s%d_hdr_after(_red1|_pre_final|_write)?\.(out|err)$"
     % (PREVSLICE, PREVSLICE)),
    ("只读仪器的控件面 3", r"^am_s%d_hdr_after_controls\.py$|^_raw_s%d_hdr_after_controls\.(out|err)$"
     % (PREVSLICE, PREVSLICE)),
    ("落地腿 5", r"^am_s%d_docs_landing\.py$|^_raw_s%d_landing_(precheck|write)\.(out|err)$"
     % (PREVSLICE, PREVSLICE)),
    ("落地腿红灯现场 12", r"^_raw_s%d_landing_(precheck|write)\.attempt\d-[a-z0-9-]+\.(out|err)$" % PREVSLICE),
    ("归档自查腿 5", r"^am_s%d_archive_check\.py$|^_raw_s%d_archive_check(\.attempt\d-[a-z0-9-]+)?\.(out|err)$"
     % (PREVSLICE, PREVSLICE)),
    ("当日纪实 1", r"^%s\.md$" % TODAY),
    # 本件自己在提交前也是未跟踪的一件 ⇒ 没有这一组它就会"落进 0 组"，覆盖闸门当场红（正是口径 ④ 的形状）
    ("本片归档写入件 1", r"^am_s%d_docs\.py$" % SLICE),
    # 本件自己的 stdout／stderr 两面也是待入库件；它们先落盘再跑，计数才不会在预检与写盘之间漂移（自指的那一轮＝attempt1 现场）
    ("本件自己的预检与写盘两面（含红灯现场） 10", r"^_raw_s%d_docs_(precheck|write)(\.attempt\d-[a-z0-9-]+)?\.(out|err)$" % SLICE),
]
buckets, assigned = [], []
for name, pat in ROLES:
    hit = [p for p in UNT if re.match(pat, os.path.basename(p))]
    assert hit, "%s 一组现数为 0 ⇒ 命名规则与本目录脱节（分组句就成了假话）" % name
    # 标签里那个数**也要跟现数对上**：写手在这里就查，不留给提交信息生成器（上一轮那处不配是提交前才被抓住的）
    want = int(one(r" (\d+)$", name, "分组标签尾部的登记数"))
    assert want == len(hit), "%s：标签登记 %d 件／现数 %d 件 ⇒ 分组句与磁盘不符，拒绝写盘" % (name, want, len(hit))
    buckets.append((name, len(hit)))
    assigned += hit
assert len(assigned) == len(set(assigned)) == NU, (
    "角色分组没恰好覆盖未跟踪清单（分组 %d／去重 %d／现数 %d）⇒ 有件落进 0 组或 ≥2 组"
    % (len(assigned), len(set(assigned)), NU))
BKT_TXT = "／".join("%s **%d**" % (n, c) for n, c in buckets)
BKT_SUM = "＋".join(str(c) for _, c in buckets) + "＝%d" % NU
MOD_TXT = "／".join(os.path.basename(p) for p in MOD)
assert NM == 5, "已跟踪档案面现数 %d（须 5）：%s" % (NM, MOD_TXT)
# 第五张＝上一片的纪实：它的落地子条由**提交之后**的填值件追加 ⇒ 它必然晚于上一片的提交、落在本片入库面里
PREVDAILYNAME = "%s.md" % os.path.basename(EVID_OLD)
PREVDAILY = [p for p in MOD if os.path.basename(p) == PREVDAILYNAME]
assert len(PREVDAILY) == 1, "已跟踪面里搜不到上一片的纪实 `%s`（现数 %d）⇒ 那句「落地子条由提交之后的填值件追加」没有出处" % (
    PREVDAILYNAME, len(PREVDAILY))
print("   待入库＝未跟踪 %d＋修改 %d＝%d 处｜src+tests 改动 %d｜分组（和＝%s）：%s"
      % (NU, NM, NU + NM, NCODE, BKT_SUM, BKT_TXT))

# ---- 上一片回执：读数只从档案正则现取，并逐字节复验正文区 ----
R30 = os.path.join(REPO, CURL)
r30 = open(R30, encoding="utf-8").read()
R30B, R30L = os.path.getsize(R30), len(r30.splitlines())
OFF30 = int(one(r"^BODY_OFFSET=(\d+)$", r30, "上一片回执的 BODY_OFFSET"))
head_part, body_part = r30.split("BODY_OFFSET=%d\n" % OFF30, 1)
assert len((head_part + "BODY_OFFSET=%d\n" % OFF30).encode()) == OFF30, "BODY_OFFSET 与实际表头字节数不等"
VERD30 = one(r"^(LANDING_VERDICT = .*)$", r30, "上一片回执的判读行")
CLOSURE30 = one(r"闭合式 远端 \d+ == 父提交 \d+ \+ 本片新增\(A\) \d+ ＝ \d+ -> PASS", r30, "上一片闭合式那句")
assert one(r"^TREE\s+: head=([0-9a-f]{40})", r30, "上一片 TREE 行哈希") == HEAD, \
    "上一片回执的 head ≠ 本地 HEAD ⇒ 那张贴的不是当前这棵树的落地证明"
FINGER_REC = one(r"^SRC指纹 : ([0-9a-f]{32})", r30, "上一片回执表头的 SRC指纹")
RAW30 = os.path.join(REPO, EVID_OLD, "_raw_verify%d.out" % PREVSLICE)
assert open(RAW30, encoding="utf-8").read() == body_part, "回执正文区与原始 stdout 不逐字节相同"
# ⚠ 这个量**不是**「顶格字段数」：它是表头区的非空行数（含缩进续行）。落笔第一版把它填进了「顶格字段」那个槽，
#    于是档案里出现过一句「顶格字段 99 行」（真顶格字段数＝16，见下面 AFHDRFIELDS 那条两处闭合）。
HDRLINES30 = len([l for l in head_part.splitlines() if l.strip()])
BODY30 = str(os.path.getsize(RAW30))
print("   上一片回执＝%s（%d B／%d 行＝表头 %d＋正文 %d，BODY_OFFSET 已复算）｜判读行「%s」｜正文区与原始件逐字节相同"
      % (PREVLOG, R30B, R30L, OFF30, int(BODY30), VERD30))

# ---- 节号：从开发日志现取最后一个 ### 26.NN，并验证它的片号＝上一片 ----
devtxt = open(DEVLOG, encoding="utf-8").read()
_secs = list(re.finditer(r"^### 26\.(\d+)[^\n]*$", devtxt, re.M))
PREVSEC = int(_secs[-1].group(1))
assert int(one(r"分片 (\d+)＝提交腿", _secs[-1].group(0), "最后一节标题里的片号")) == PREVSLICE, \
    "§26.%d 的标题写的片号 ≠ 上一片 %d ⇒ 节号与片号的对应变了，本节要先重推" % (PREVSEC, PREVSLICE)
SECN = PREVSEC + 1
assert devtxt.count("### 26.%d " % SECN) == 0, "开发日志里已有 §26.%d ⇒ 本节要重跑先起新名" % SECN
SUBJ1 = SUBJ.split("＋")[0]
assert SUBJ1 and SUBJ1 in SUBJ and len(SUBJ1) <= 60, "主题第一子句异常：%r" % SUBJ1

# 排期那条引用要落在**真写了它的那一节**：按短语在全树各节里现定位，不凭记忆写死节号
_bodies = [(m.group(1), devtxt[m.end():(_secs[i + 1].start() if i + 1 < len(_secs) else len(devtxt))])
           for i, m in enumerate(_secs)]
G9SEC = [n for n, b in _bodies if "唯一可评分的 N=1" in b]
assert len(G9SEC) == 1, "「唯一可评分的 N=1」落在 %d 节（须恰好 1）⇒ 那个节号没有唯一出处：%s" % (
    len(G9SEC), G9SEC)
G9SEC = "26.%s" % G9SEC[0]
# 正对照：把那句复制进**另一节**，定位器必须数出 2 节（还数 1 就说明它不看正文，节号仍是凭记忆）
_other = [i for i, (n, _b) in enumerate(_bodies) if "26.%s" % n != G9SEC][0]
_pert = list(_bodies)
_pert[_other] = (_pert[_other][0], _pert[_other][1] + "\n诱饵：那里给出唯一可评分的 N=1 的读数。")
_n = len([k for k, b in _pert if "唯一可评分的 N=1" in b])
assert _n == 2, "节定位器在写坏的样本上数出 %d 节（须 2）⇒ 它没有分辨力" % _n

# ---- 上一轮三面干跑控件面的档案读数（每条都要求恰好一次命中）----
OLD = lambda *a: os.path.join(REPO, EVID_OLD, *a)
MSGCTL = OLD("_raw_s%d_msg_controls.out" % PREVSLICE)
msg = open(MSGCTL, encoding="utf-8").read()
MSGCTLB, MSGERRB, MSGCTLM = os.path.getsize(MSGCTL), os.path.getsize(
    OLD("_raw_s%d_msg_controls.err" % PREVSLICE)), md5(MSGCTL)[:12]
MSGNCASE = int(one(r"^CASES 声明＝(\d+) 条｜本件自己的锚点普查", msg, "提交说明控件面条数"))
MSGNTO = int(one(r"超时（环境故障，不判红也不判过）＝(\d+) 条", msg, "提交说明控件面超时数"))
MSGNUNC = int(one(r"本轮造不出红路的闸门＝(\d+) 条", msg, "提交说明控件面 UNCOVERED 数"))
MSGVERD = one(r"^(CONTROL_VERDICT = .*)$", msg, "提交说明控件面判读行")

HDRCTL = OLD("_raw_s%d_hdr_controls.out" % PREVSLICE)
hdr = open(HDRCTL, encoding="utf-8").read()
HDRCTLB, HDRERRB, HDRCTLM = os.path.getsize(HDRCTL), os.path.getsize(
    OLD("_raw_s%d_hdr_controls.err" % PREVSLICE)), md5(HDRCTL)[:12]
HDRNRED = int(one(r"红路声明＝(\d+) 条", hdr, "表头控件面红路数"))
HDRNGREEN = int(one(r"保持绿＝(\d+) 条", hdr, "表头控件面绿路数"))
HDRNTO = int(one(r"超时（环境故障，不判红也不判过）＝(\d+) 条", hdr, "表头控件面超时数"))
HDRNUNC = int(one(r"本轮造不出红路的闸门＝(\d+) 条", hdr, "表头控件面 UNCOVERED 数"))
HDRNSELF = int(one(r"本面自查＝(\d+) 条", hdr, "表头控件面自查条数"))
HDRVERD = one(r"^(CONTROL_VERDICT = .*)$", hdr, "表头控件面判读行")

AFCTL = OLD("_raw_s%d_hdr_after_controls.out" % PREVSLICE)
afctl = open(AFCTL, encoding="utf-8").read()
AFCTLB, AFERRB, AFCTLM = os.path.getsize(AFCTL), os.path.getsize(
    OLD("_raw_s%d_hdr_after_controls.err" % PREVSLICE)), md5(AFCTL)[:12]
AFNCASE = int(one(r"CASES 声明＝(\d+) 条（全部红路）", afctl, "只读仪器控件面红路数"))
AFNGREEN = int(one(r"保持绿＝(\d+) 条", afctl, "只读仪器控件面绿路数"))
AFNTO = int(one(r"超时（环境故障，不判红也不判过）＝(\d+) 条", afctl, "只读仪器控件面超时数"))
AFNUNC = int(one(r"本轮造不出红路的闸门＝(\d+) 条", afctl, "只读仪器控件面 UNCOVERED 数"))
AFVERD = one(r"^(CONTROL_VERDICT = .*)$", afctl, "只读仪器控件面判读行")
assert MSGNTO == HDRNTO == AFNTO == 0, "三面里有超时（%d／%d／%d）⇒ 它不能当全过的档案引用" % (
    MSGNTO, HDRNTO, AFNTO)
# 三面判读是否同一个字符串＝现比，不写死；只有相等才许用"三面判读一致"这一句
assert MSGVERD == HDRVERD == AFVERD, "三面判读不一致（%s／%s／%s）⇒ 不能合并成一句引用" % (
    MSGVERD, HDRVERD, AFVERD)
VERD3 = MSGVERD
# 继承常量从上一片控件件**源码**现取（写死就会在别人改那两行后腐烂）
CTL_SRC = open(OLD("am_s%d_hdr_controls.py" % PREVSLICE), encoding="utf-8").read()
SUBTO = one(r"^SUB_TIMEOUT = (\d+)$", CTL_SRC, "子运行上限常量")
ENVSTOP = one(r"^ENV_FAIL_STOP = (\d+)$", CTL_SRC, "环境故障累计常量")
WINDOW = one(r"行号落在该句在\*\*副本\*\*里的 ±(\d+) 行窗口", CTL_SRC, "raise 行号窗口常量")

# ---- 只读仪器本身与落地腿、归档自查腿 ----
AFOUT = OLD("_raw_s%d_hdr_after.out" % PREVSLICE)
AFB = os.path.getsize(AFOUT)
AFVERDICT = one(r"^(HDR_AFTER_VERDICT = .*)$", open(AFOUT, encoding="utf-8").read(), "只读仪器判读行")
TRIPLE = one(r"cmp -i 三联（本件自己起进程跑的，不是抄生成器的）＝ (\d+) / (\d+) / (\d+)",
             open(AFOUT, encoding="utf-8").read(), "只读仪器的 cmp 三联")
assert TRIPLE == ("0", "1", "1"), "上一片回执的 cmp 三联＝%s（须 0/1/1）⇒ 那张回执不能当全绿引用" % (TRIPLE,)

# ---- 顶格字段数：三个出处必须同值，否则这个槽不许落笔 ----
# 上一版把「表头非空行数」（含缩进续行）填进了「顶格字段」这个槽 ⇒ 档案里写过「顶格字段 99 行」，
# 而这一族的既有口径（§26.27／29／30／31／32 五处）都是**生成器 FIELDS 声明数＝16**。三个出处：
afd = open(AFOUT, encoding="utf-8").read()
AFFIELDS = one(r"^顶格字段行数＝ (\d+) / (\d+)", afd, "只读仪器那句顶格字段行数")
_hsrc = open(OLD("am_s%d_hdr.py" % PREVSLICE), encoding="utf-8").read()
_decl = None
for _n in ast.parse(_hsrc).body:
    if isinstance(_n, ast.Assign) and getattr(_n.targets[0], "id", None) == "FIELDS":
        _decl = len(_n.value.elts)
assert _decl is not None, "在 %s 里 ast 取不到 FIELDS 赋值 ⇒ 顶格字段的分母不是现取的" % PREVSLICE
_now = len([l for l in head_part.splitlines() if l[:1] not in (" ", "\t") and ": " in l])
assert str(_decl) == AFFIELDS[0] == AFFIELDS[1] == str(_now), \
    "顶格字段数三出处不齐：FIELDS 声明 %s／只读仪器 %s／本件现数回执顶格行 %s" % (_decl, "/".join(AFFIELDS), _now)
NDEF30 = str(_decl)
print("   顶格字段＝%s 行（FIELDS 声明＝只读仪器 %s／本件现数 %s，三出处同值）｜表头非空行数（含续行）＝%d，两者不是同一个量"
      % (NDEF30, "/".join(AFFIELDS), _now, HDRLINES30))
LANDPY = OLD("am_s%d_docs_landing.py" % PREVSLICE)
LANDPRE = OLD("_raw_s%d_landing_precheck.out" % PREVSLICE)
LANDW = OLD("_raw_s%d_landing_write.out" % PREVSLICE)

ARCHPY = OLD("am_s%d_archive_check.py" % PREVSLICE)
ARCHOUT = OLD("_raw_s%d_archive_check.out" % PREVSLICE)
arch = open(ARCHOUT, encoding="utf-8").read()
ARCHPYB, ARCHPYM = os.path.getsize(ARCHPY), md5(ARCHPY)[:12]
ARCHOUTB, ARCHERRB = os.path.getsize(ARCHOUT), os.path.getsize(
    OLD("_raw_s%d_archive_check.err" % PREVSLICE))
ARCHCLAIMS, ARCHMISS = one(r"声称条数＝(\d+)｜缺失＝(\d+)/\d+", arch, "自查件声称／缺失")
# 该档里**没有**「正对照共 N 条」这种汇总句（本轮实测），所以条数只能逐行数；逐行定位器配一条合成正对照
ARCHNCTRL = len([l for l in arch.splitlines() if l.startswith("  正对照：")])
assert len([l for l in ["  正对照：合成的一条"] if l.startswith("  正对照：")]) == 1, "正对照逐行定位器失明"
assert ARCHNCTRL >= 1, "自查件里数不出正对照行（现数 %d）⇒ 那句\"正对照 N 条能红\"会渲染成假零" % ARCHNCTRL
ARCHVERD = one(r"^(ARCHIVE_VERDICT = .*)$", arch, "自查件判读行")
assert ARCHMISS == "0", "自查件报出缺失 %s 条 ⇒ 不能引它当全过的档案" % ARCHMISS
COV = [(m.group(1), int(m.group(2)), int(m.group(3)))
       for m in re.finditer(r"承载：(.+?)＝(\d+) 条（其中独有 (\d+) 条）", arch)]
assert len(COV) == 5, "承载表只有 %d 行 ⇒ 档案面少了一面" % len(COV)
DEVCOV = [c for c in COV if c[0].endswith("开发日志.md")][0]
UNIQ_TOTAL = sum(u for _f, _n, u in COV)
assert 1 <= UNIQ_TOTAL <= int(ARCHCLAIMS), "独有总数 %d 不在 [1, 声称总数 %s] 里 ⇒ 承载表自相矛盾" % (
    UNIQ_TOTAL, ARCHCLAIMS)

# ---- 本轮两次红灯现场的档案读数（时长与形状都从现场件现取，不转述）----
A1 = OLD("_raw_s%d_hdr_controls.attempt1-selfaudit-inverted.out" % PREVSLICE)
A1E = A1[:-4] + ".err"
A2 = OLD("_raw_s%d_hdr_controls.attempt2-notb-unsatisfiable.out" % PREVSLICE)
A2E = A2[:-4] + ".err"
A1B, A1EB = os.path.getsize(A1), os.path.getsize(A1E)
A2B, A2EB = os.path.getsize(A2), os.path.getsize(A2E)
A2VERD = one(r"^(CONTROL_VERDICT = .*)$", open(A2, encoding="utf-8").read(), "第二次真跑的判读行")
# 未命中行的**实际形状**是顶格的「× E6 …」（缩进的「  × 」形状在本档里现数 0 ⇒ 那是假零）
A2LINES = open(A2, encoding="utf-8").read().splitlines()
A2X = len([l for l in A2LINES if l.startswith("× ")])
A2INDENT = len([l for l in A2LINES if l.startswith("  × ")])
assert A2X >= 1, "现场件里数不出「× 」未命中行 ⇒ 那句\"判据响了却记成未命中\"没有出处"
assert len([l for l in ["× E6 合成的一条未命中行"] if l.startswith("× ")]) == 1, "× 行定位器对合成样本失明"
assert A2VERD.startswith("CONTROL_VERDICT = PROBLEM"), "现场件的判读行不是 PROBLEM ⇒ 那条标本讲的不是这次事故"
LANDPRE_N = len([p for p in UNT if re.search(r"^_raw_s%d_landing_precheck\.attempt" % PREVSLICE,
                                             os.path.basename(p))]) // 2
print("   三面控件面：提交说明 %d B（%d 条／UNCOVERED %d）｜表头 %d B（红 %d＋绿 %d／UNCOVERED %d／自查 %d）｜只读仪器 %d B（红 %d＋绿 %d／UNCOVERED %d）"
      % (MSGCTLB, MSGNCASE, MSGNUNC, HDRCTLB, HDRNRED, HDRNGREEN, HDRNUNC, HDRNSELF,
         AFCTLB, AFNCASE, AFNGREEN, AFNUNC))
print("   归档自查件 %d B（md5 %s）stdout %d B／stderr %d B＝声称 %s 条全命中、正对照逐行数到 %d 条 ⇒「%s」"
      % (ARCHPYB, ARCHPYM, ARCHOUTB, ARCHERRB, ARCHCLAIMS, ARCHNCTRL, ARCHVERD))
print("   两次红灯现场：attempt1 %d B／%d B（收面自查把正确实现判红）｜attempt2 %d B／%d B（判据恒假，「× 」行 %d 条）"
      % (A1B, A1EB, A2B, A2EB, A2X))

# ---- 尺子族：按 md5 分组数 ----
FAM = "0d20117bb82f"
allr = glob.glob(os.path.join(REPO, "docs/evidence/*/am_s7_verify*.py"))
fam_files = sorted(p for p in allr if md5(p)[:12] == FAM)
stray = sorted(os.path.basename(p) for p in allr if md5(p)[:12] != FAM)
RULE_N, RULE_GLOB = len(fam_files), len(allr)
assert RULE_N == 11, "md5 族现数 %d 份（散文写的是 11）⇒ 分组句要先改再写" % RULE_N
print("   尺子＝md5 前 12 `%s` 一族现数 %d 份（同一 glob 形状 %d 份，差额有名字＝%s）⇒ 本片抄第 %d 份"
      % (FAM, RULE_N, RULE_GLOB, "、".join(stray), RULE_N + 1))

# ---- SRC 指纹：两条 locale 各现算一次（上一轮那条新学的实证，也是它自己的靶）----
FING = ("find src tests -name '*.py' -not -path '*__pycache__*' -print0 "
        "| sort -z | xargs -0 md5sum | md5sum | cut -d' ' -f1")
FINGER_DEF = subprocess.run(["bash", "-c", FING], cwd=REPO, capture_output=True, text=True).stdout.strip()
_env = dict(os.environ, LC_ALL="C")
FINGER_C = subprocess.run(["bash", "-c", FING], cwd=REPO, capture_output=True, text=True,
                          env=_env).stdout.strip()
assert FINGER_DEF == FINGER_REC, "默认 locale 现算 %s ≠ 上一片回执 %s ⇒ 生产树内容真的变了" % (
    FINGER_DEF, FINGER_REC)
assert FINGER_C != FINGER_DEF, "两种 locale 现算相同 ⇒ 那条耦合登记没有靶了（要先确认它还存在）"
assert re.fullmatch(r"[0-9a-f]{32}", FINGER_C), "LC_ALL=C 读数不像 md5 摘要＝%r ⇒ 命令本身跑坏了" % FINGER_C
print("   SRC 指纹：默认 locale＝%s（＝上一片回执表头的值）｜LC_ALL=C＝%s ⇒ 差异只来自 sort 的 collation"
      % (FINGER_DEF, FINGER_C))

# ---- GPU：本轮的普查仪器自己坏了 ⇒ 换一条能跑的第二出处，并把"坏了"当环境事实登记 ----
nv = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                    capture_output=True, text=True)
_lines = [l.strip() for l in nv.stdout.splitlines() if l.strip()]
NVML_OK = nv.returncode == 0 and all(re.fullmatch(r"\d+", l) for l in _lines)
NVML_WHY = (_lines[0] if _lines else "").strip()
DRV = open("/sys/module/nvidia/version", encoding="utf-8").read().strip()
LIBM = re.search(r"NVML library version: ([\d.]+)", nv.stdout + nv.stderr)
LIB = LIBM.group(1) if LIBM else ""


def cmdline_raw(pid):
    """进程随时可能退出：读数失败一律当"没有这条命令行"，不让一次竞态把断言变成异常。"""
    try:
        return open("/proc/%s/cmdline" % pid, "rb").read()
    except OSError:
        return b""


def comm(pid):
    try:
        return open("/proc/%s/comm" % pid, encoding="utf-8", errors="replace").read().strip()
    except OSError:
        return ""


def argv0(pid):
    parts = cmdline_raw(pid).split(b"\0")
    return parts[0].decode("utf-8", "replace") if parts and parts[0] else None


def match_argv0(blob):
    """替代口径：整词比 argv[0]，不比命令行子串（子串会把审计它的那条命令自己也认成我的进程）。"""
    return blob.split(b"\0")[0].decode("utf-8", "replace") == INTERP


NPROCS = [d for d in os.listdir("/proc") if d.isdigit()
          and d not in (str(os.getpid()), str(os.getppid()))]
mine_proc = [d for d in NPROCS if argv0(d) == INTERP]
# 松散子串匹配会把"正在审计这条匹配式的命令"自己也认成我的 GPU 进程（本轮实测的形状，作负对照留着）
loose = [d for d in NPROCS if INTERP in (argv0(d) or "")
         or INTERP in cmdline_raw(d).decode("utf-8", "replace").replace("\x00", " ")]
upgr = [d for d in NPROCS if comm(d).startswith("unattended-upgr")]
# 匹配式自己的两条对照：合成的 argv[0] 必须数出 1， argv[0] 不是它、命令行里含着这条路径的必须数出 0
assert match_argv0((INTERP + "\0x.py").encode()) is True, "argv[0] 匹配式的正对照自己失效了"
assert match_argv0(b"/bin/bash -c grep " + INTERP.encode() + b"\0") is False, \
    "argv[0] 匹配式把子串也认成命中 ⇒ 它和松散匹配一样会误指自己"
# 误指的**形状**不靠现场运气：造一条合成命令行（argv[0]＝/bin/bash、参数里带着我的解释器）当场两种口径各数一次
FAKE_SHELL = ("/bin/bash\0-c\0cd /x && timeout 300 %s docs/evidence/%s/am_s%d_docs.py --write\0"
              % (INTERP, TODAY, SLICE)).encode("utf-8")
def loose_rule(blob):
    return INTERP in blob.decode("utf-8", "replace").replace("\x00", " ")
assert loose_rule(FAKE_SHELL) is True and match_argv0(FAKE_SHELL) is False, \
    "合成命令行没复现出「松散命中／整词不命中」的形状 ⇒ 那句\"松散口径会误指承载它的 shell\"没有出处"
LOOSE_DEMO = "1／0"
LOOSE0 = "、".join(sorted({(argv0(d) or "？") for d in loose}) or ["无"])
if NVML_OK:
    NVME, GPU_TOT = str(len([d for d in mine_proc])), str(len(NPROCS))
    GPU_WHY = "NVML 可用（rc=0）"
    assert not mine_proc, "我的解释器在跑 %s ⇒ 与「不碰 GPU」冲突" % mine_proc
else:
    assert DRV and LIB and DRV != LIB, \
        "NVML 起不来（rc=%d／首行「%s」）但驱动 %s 与库 %s 看不出版本差 ⇒ 那句根因没有出处" % (
            nv.returncode, NVML_WHY, DRV, LIB)
    NVME, GPU_TOT = str(len(mine_proc)), str(len(NPROCS))
    GPU_WHY = "NVML 不可用（rc=%d／「%s」／内核模块 %s ≠ 用户态库 %s）" % (
        nv.returncode, NVML_WHY, DRV, LIB)
    assert mine_proc == [], "替代普查数出我的解释器进程 %s ⇒ 与「不碰 GPU」冲突" % mine_proc
LOOSE_N = str(len(loose))
print("   GPU 归属：NVML 普查＝%s｜替代口径（/proc argv[0] 整词）＝我的解释器进程 %s/%s 个｜松散子串同清单现数 %s 个（argv[0]＝%s）｜合成对照＝松散 1／整词 0｜unattended-upgrade 进程 %d 个（旁证）"
      % (GPU_WHY, NVME, GPU_TOT, LOOSE_N, LOOSE0, len(upgr)))

# ---- 计划文档的补记序号：由该面现取最大值＋1（上一轮写死 5，本轮已经轮到 7）----
plantxt = open(PLAN, encoding="utf-8").read()
_w = [int(m.group(1)) for m in re.finditer(r"^### §W 补记 (\d+)", plantxt, re.M)]
assert _w, "计划文档里取不到 §W 补记 的编号 ⇒ 序号是凭记忆的"
WN = max(_w) + 1
assert plantxt.count("### §W 补记 %d" % WN) == 0, "§W 补记 %d 已存在 ⇒ 本件要重跑先起新号" % WN

# 该面自查那三条的名字（从上一片表头控件面的 stdout 现取，不转述；括注在「各验一次」之后还有下半句，所以锚到那里就停）
HDRNSELF_NAMES = one(r"本面自查＝\d+ 条（(.+?)；每条都两个方向各验一次", hdr, "表头控件面自查三条的名字")

CTX = {
    "TIME": TIME, "HEAD7": HEAD7, "PARENT7": PARENT7, "SUBJ": SUBJ1, "CDATE": CDATE,
    "SLICE": str(SLICE), "PREVSLICE": str(PREVSLICE), "NEXTSLICE": NEXTSLICE,
    "SECN": str(SECN), "PREVSEC": str(PREVSEC),
    # 散文里的节号要写成开发日志自己的形制 §26.NN（写 §33 会让读者找不到那一节）
    "SECREF": "26.%d" % SECN, "PREVSECREF": "26.%d" % PREVSEC,
    "UNT": str(NU), "MOD": str(NM), "TOT": str(NU + NM),
    "BKT": BKT_TXT, "BKTSUM": BKT_SUM, "MODTXT": MOD_TXT,
    "PREVLOG": PREVLOG, "R30B": str(R30B), "R30L": str(R30L), "OFF30": str(OFF30),
    "BODY30": BODY30, "NDEF30": NDEF30, "HDRL30": str(HDRLINES30), "VERD30": VERD30, "CLOSURE30": CLOSURE30,
    "MSGCTLB": str(MSGCTLB), "MSGCTLM": MSGCTLM, "MSGERRB": str(MSGERRB),
    "MSGNCASE": str(MSGNCASE), "MSGNUNC": str(MSGNUNC), "MSGVERD": MSGVERD,
    "HDRCTLB": str(HDRCTLB), "HDRCTLM": HDRCTLM, "HDRERRB": str(HDRERRB),
    "HDRNRED": str(HDRNRED), "HDRNGREEN": str(HDRNGREEN), "HDRNUNC": str(HDRNUNC),
    "HDRNSELF": str(HDRNSELF), "HDRNSELFN": HDRNSELF_NAMES, "HDRVERD": HDRVERD,
    "AFCTLB": str(AFCTLB), "AFCTLM": AFCTLM, "AFERRB": str(AFERRB), "AFNCASE": str(AFNCASE),
    "AFNGREEN": str(AFNGREEN), "AFNUNC": str(AFNUNC), "AFVERD": AFVERD,
    "AFB": str(AFB), "AFVERDICT": AFVERDICT, "TRIPLE": " / ".join(TRIPLE),
    "SUBTO": SUBTO, "ENVSTOP": ENVSTOP, "WINDOW": WINDOW,
    "LANDPYB": str(os.path.getsize(LANDPY)), "LANDPREB": str(os.path.getsize(LANDPRE)),
    "LANDWB": str(os.path.getsize(LANDW)), "LANDPREN": str(LANDPRE_N),
    "ARCHPYB": str(ARCHPYB), "ARCHPYM": ARCHPYM, "ARCHOUTB": str(ARCHOUTB),
    "ARCHERRB": str(ARCHERRB), "ARCHCLAIMS": ARCHCLAIMS, "ARCHNCTRL": str(ARCHNCTRL),
    "ARCHVERD": ARCHVERD, "DEVCOV": "%d（独有 %d）" % (DEVCOV[1], DEVCOV[2]),
    "COVTXT": "／".join("%s＝%d（独有 %d）" % (os.path.basename(f), n, u) for f, n, u in COV),
    "A1B": str(A1B), "A1EB": str(A1EB), "A2B": str(A2B), "A2EB": str(A2EB),
    "A2X": str(A2X), "A2INDENT": str(A2INDENT), "A2VERD": A2VERD, "VERD3": VERD3,
    "FAM": FAM, "RULE_N": str(RULE_N), "RULE_NEW": str(RULE_N + 1),
    "RULE_GLOB": str(RULE_GLOB), "STRAY": "、".join(stray),
    "FINGER_DEF": FINGER_DEF, "FINGER_C": FINGER_C,
    "NVME": NVME, "NPROCS": GPU_TOT, "GPUWHY": GPU_WHY, "LOOSE": LOOSE_N,
    "LOOSE0": LOOSE0, "LOOSEDEMO": LOOSE_DEMO,
    "DRV": DRV, "LIB": LIB, "NVRC": str(nv.returncode), "NVWHY": NVML_WHY,
    "EVIDNEW": EVID_NEW, "WN": str(WN), "NCODE": str(NCODE),
    "LESSON": LESSON, "LESSON_NEW": LESSON_NEW,
    "G9SEC": G9SEC, "PREVDAILYNAME": PREVDAILYNAME,
}

# ============================ 五张面 ============================
SEC_TITLE = ("### 26.%(SECN)s 分片 %(SLICE)s＝提交腿：分片 %(PREVSLICE)s 的落地半腿整条链入库"
             "（未跟踪 %(UNT)s＋修改 %(MOD)s＝%(TOT)s 处）＋一条**环境级**新学＝GPU 普查那把尺子（NVML）"
             "本轮坏了（%(TIME)s 现取）\n\n") % CTX

DEV_T = SEC_TITLE + """- **形制约束的落实＝本轮另起 §%(SECREF)s，不在 §%(PREVSECREF)s 内追加**：上一节落地子条末尾登记的那条可核查约束——回执表头生成器的 `bullet_for` 要求「引用上一片回执档案名」的顶级子条在**同一节内唯一**——本节由结构满足：新节号、节内该句式恰好 1 条。本条刻意**不复述那个档案名**（在本节复述就会出现第二条命中，而那正是这条闸门要拦的事，所以它同时也是「这条闸能红」的证明）。同节另外三处唯一性一并照查：带引号的钉住口径登记恰好 1 条、承接的 LESSON 锚句恰好 1 条、本轮新学的锚句恰好 1 条；写入件在落盘前对片段各数一次、落盘后再对磁盘上的整节数一次，两次不等就报错停止。
- **入库面（本条存在的唯一理由＝让分片 %(SLICE)s 的回执表头把「引用上一片回执」这件事从档案现取，而不是凭对话记忆）**：本节里唯一那条引用上一片回执名的子条就是这一条，档案名＝`%(PREVLOG)s`（%(R30B)s B／%(R30L)s 行＝表头 %(OFF30)s＋正文 %(BODY30)s，`BODY_OFFSET=%(OFF30)s` 已由写入件重解一遍并对上正文区；顶格字段 %(NDEF30)s 行（＝表头生成器 `FIELDS` 的 ast 声明数＝只读仪器那句两处读数，三个出处同值；另有表头非空行数含缩进续行＝%(HDRL30)s，两者不是同一个量，落笔第一版把后者填进了前者那个槽）；判读行「%(VERD30)s」；闭合式原文「%(CLOSURE30)s」）。落笔现数（`git status --porcelain --untracked-files=all`）＝未跟踪 **%(UNT)s**＋修改 **%(MOD)s**＝**%(TOT)s** 件。%(UNT)s 件按角色分组：%(BKT)s；%(BKTSUM)s 与 `grep -c '^??'` 现数相符（分组＝按文件名模式现数，一件落进 0 组或 ≥2 组就拒绝写盘；每组标签尾部那个登记数**也在写盘前跟现数再对一次**，不留到提交信息生成器才发现）。修改 %(MOD)s 件＝已跟踪的档案面（%(MODTXT)s；其中 `%(PREVDAILYNAME)s` 是上一片的纪实——它的落地子条由**提交之后**的填值件追加，所以时间戳必然晚于上一片的提交时刻，因而落在本片入库面里，不是遗漏）。本片自己的写入件、信息生成器与它的干跑控件件在提交前一并暂存 ⇒ 最终入库数**大于**这里的 %(TOT)s，那一部分由提交之后 `git show --name-status HEAD` 现取并写进本节的落地子条，不在这里预支。⚠ 本条里所有件数都是落笔那一刻的现取值，**下一轮必须重新现取**。
- **钉住口径与红线（当前「继续，不碰GPU」）**：本轮每条计算命令逐条内联 `CUDA_VISIBLE_DEVICES=\"\"`（含一行式命令）；**不产生任何吞吐／加速／预算数字**，墙钟只作可行性代价登记；GPU 归属本轮**没有 NVML 那把尺子**（原因与替代口径见下面「本轮新学」那条），替代断言＝`/proc/*/cmdline` 的 argv[0] **整词**匹配我的解释器 ⇒ 落笔现数我的解释器进程 **%(NVME)s** 个（分母＝同一趟扫到的 **%(NPROCS)s** 个 PID，已排除自身与父进程），该匹配式配一正一负两条对照；卡上 compute 进程总数本轮不可现取，所以不写进任何档案面，也不拿"上次那个数"顶替。每次尝试换档案名，红件先入库再复跑；A0 断言一字不动、#27 不换指标、无新增 skip/xfail。生产树 `src`＋`tests` 现数改动 **%(NCODE)s** 件 ⇒ 本片仍是纯归档腿。
- **%(LESSON)s＝从分片 %(PREVSLICE)s 的落地半腿带进本片的口径**：那一轮抓到两处判断缺陷，**两处都在控件面对它自己的判读上**，且两处都是「一条按构造红不了的判据」反过来把**正确实现**判成红。⑴ 第一次真跑：收面自查要求 md5 快照的比对函数「不一致时非空」，而它在健康路径根本不抛异常 ⇒ 标志位恒 False，整面被判红、逐条读数当场作废重跑（现场 `attempt1-selfaudit-inverted` stdout %(A1B)s B／stderr %(A1EB)s B）。修法两条都已进该面源码：自审改成正向双判据（不一致要抓得出、一致要说不报警），且逐条读数**即时落盘**（第一版把整批表攒到最后统一印，崩溃时整批丢掉，重跑要再付一次配额）。⑵ 第二次真跑：判红那条合取要求抛出行以 `SystemExit:` 打头，而 CPython 对未捕获的 `SystemExit(str)` 只把消息原样写进 stderr、连 traceback 都不打 ⇒ 恒假的合取＝闸门响了却被记成未命中（现场 `attempt2-notb-unsatisfiable` stdout %(A2B)s B／stderr %(A2EB)s B，表里未命中行的实测形状是**顶格**的「× 」＝%(A2X)s 条，而按缩进的「␣␣× 」去数现数 %(A2INDENT)s 条＝假零（定位器要按实测形状写，这一族本轮又中一次）；整面判读「%(A2VERD)s」）。修法＝把谓词从判定段抽成纯函数、按实测到的形状重写，并给它配一条**不花一次配额**的内存判定表；该面自查因此是 %(HDRNSELF)s 条（%(HDRNSELFN)s）。⇒ 一条对照必须**同时能红能绿**，而「判红的方式」自己也要被造一次红。
- **本轮新学（环境级＋口径级两条，都是仪器侧的事，不是判据缺陷也不是偏好）＝%(LESSON_NEW)s**：① **环境级**：`nvidia-smi --query-compute-apps=pid` 本轮起不来（rc=%(NVRC)s，首行「%(NVWHY)s」），根因由两条命令现取并对上——内核模块 `/sys/module/nvidia/version`＝`%(DRV)s` 与用户态 NVML 库＝`%(LIB)s` 是两个版本（机器上此刻还有 unattended-upgrade 进程在跑，那是旁证不是结论）。于是「我没占卡」这条红线**失去了它一直依赖的那把尺子**；本轮不换口径、也不假装量到，而是把这条断言改由 `/proc` 的 argv[0] 整词匹配承担（分母＝%(NPROCS)s 个 PID；正对照＝合成的那条 argv[0] 必须数出 1；负对照＝命令行里**含**这条路径但 argv[0] 不是它的进程必须数出 0——同一条清单上松散子串匹配现数 %(LOOSE)s 个（argv[0] 依次为 %(LOOSE0)s），而这个误指的形状不靠现场运气复现：本件造了一条合成命令行（argv[0]＝`/bin/bash`、参数里带着我的解释器路径），松散口径数出 1、整词口径数出 0 ⇒ 松散匹配会把「承载这条命令的那个 shell」自己也认成我的 GPU 进程，而这条对照每次跑都成立）。② **口径级**：`git status --porcelain` 默认 `-u normal` 会把**整个目录都未跟踪**的那种目录折成一行（本轮新建的 `%(EVIDNEW)s/` 在默认口径下只印目录名一行，加 `-uall` 才逐条印）⇒ 件数与角色分组会凭空少算，而"少算"在分组闸门里表现为某组现数为 0。本件所有计数一律走 `-uall`，另加一条闸门：清单里不许出现以 `/` 结尾的条目。两个读数都留在本节，供下一轮自查件回搜。
- **跑前登记的闸门（分片 %(SLICE)s 的八条，逐条可红，跑后读数从归档件现取）**：① 尺子复用＝**逐字节副本**——`cp -p` 之后 md5 前 12 必须等于 `%(FAM)s` 这一族，份数按 **md5 分组**数而不是按 glob 命中数（落笔现数 %(RULE_N)s 份／同一条 glob 形状 %(RULE_GLOB)s 份，差额有名字＝%(STRAY)s；本片抄第 %(RULE_NEW)s 份）；② 五步落地证明（`ls-remote`／`truncated` 须 false／blob＋tree 两侧计数／双向路径差集且两侧同用 `core.quotePath=false` 与 `LC_ALL=C sort`／逐文件 blob SHA 普查带分母）＋闭合式 远端＝父提交＋count(A)，四个正对照 A／B／C／D 各自能红；③ 回执表头只由生成器产出（预检面与写盘面各留原始件，`BODY_OFFSET` 由不动点迭代，写盘后**另跑一件只读仪器**做 `cmp -i 偏移:0` 三联＝0/1/1，顶格字段数与生成器 `FIELDS` 声明互证）；④ 提交说明由信息生成器产出（PLACEHOLDER＋`str.replace`、双向查「未取值的字段」与「模板没用到的读数」、shortstat 三个数闭合到 `--numstat` 逐行现算、指纹四路闭合＝**默认 locale 下**现算＝上一片回执表头 SRC 指纹）；⑤ **本轮新增**：入库清单一律 `-uall` 现数，并加一条「清单里不许出现折叠的目录行」的闸门（分组和＝未跟踪数、标签登记数＝现数，两处都配诱饵正对照）；⑥ 三面干跑控件面按「rc≠0 **且**抛出的那一行含指定闸门句 **且**行号落在该句在该**副本**里的 ±%(WINDOW)s 行窗口」判红，并如实登记造不出红路的闸门；表头面每条扰动都要复跑一次联网尺子 ⇒ 约 %(HDRNRED)s 次 GitHub 未认证 API 调用、配额 60/小时，各面分开排在配额复位之后，并继承上一片的常量（子运行上限 %(SUBTO)s s、环境故障累计 %(ENVSTOP)s 条即停，超时**既不判红也不判过**、单独计数）；⑦ 归档自查件（新追加段读数普查＋下一片锚点复验＋逐字节复跑，且 stdout 不印任何会被档案引用的自指数）。⑧ **本轮的红灯现场新增两条，都是本件自己的**：第一次跑预检时，本件自己的 stdout／stderr 面把计数推着走（预检与写盘两次落笔读出不同的未跟踪数＝自指的形状，现场件名带 `attempt1-selfcount-recursive`）⇒ 修法＝这几面**先落盘再跑**、并把它们本身编进角色分组，于是分组和与现数在两次运行里都同值；第一次跑 `--write` 时，那条「不许动正在跑的仪器」的守卫被**本件自己的名字**打掉（现场件名带 `attempt2-selfguard-hits-itself`，它在任何追加之前就停下 ⇒ 五张档案面当时逐字节未被动过，这一条由写盘后的行数差复核）⇒ 修法＝自己这一支按「argv[0]＝解释器」数副本、承载它的 wrapper 不参与判定，别的仪器仍按整词命中拦停。**本轮另有一条流程前提**：任何仪器落盘后要先 `git add` 再跑它自己的控件面（上一轮的实测：编辑一张**已暂存**的仪器会让未暂存漂移闸门先打掉未扰动基线，整轮控件作废）。
- **分片 %(SLICE)s 落地腿＝提交＋推送＋五步落地证明＋生成器回执表头＋三面控件面＋落地填值＋归档自查（%(TIME)s 现取；生产树 src＋tests 改动 %(NCODE)s、GPU 全程未碰）**：本条是**写盘前**登记的计划，不是结果——落地读数（本片 HEAD 全哈希、入库清单 A／M 计数、闭合式、回执表头字节与 `BODY_OFFSET`、三面控件面的扰动条数与 UNCOVERED）此刻**还不存在**；它们由提交之后 `git show --name-status HEAD` 与回执原始件现取，写进本节的落地子条，并交**分片 %(NEXTSLICE)s** 的归档半腿做自查。上一片的同类件在档可查，引用即从档案正则现取：提交说明控件面 stdout %(MSGCTLB)s B（md5 %(MSGCTLM)s／stderr %(MSGERRB)s B）＝扰动 %(MSGNCASE)s 条、造不出红路 %(MSGNUNC)s 条 ⇒「%(MSGVERD)s」；表头控件面 stdout %(HDRCTLB)s B（md5 %(HDRCTLM)s／stderr %(HDRERRB)s B）＝红路 %(HDRNRED)s 条、保持绿 %(HDRNGREEN)s 条、本面自查 %(HDRNSELF)s 条、造不出红路 %(HDRNUNC)s 条 ⇒「%(HDRVERD)s」；只读仪器的控件面 stdout %(AFCTLB)s B（md5 %(AFCTLM)s／stderr %(AFERRB)s B）＝红路 %(AFNCASE)s 条、保持绿 %(AFNGREEN)s 条、造不出红路 %(AFNUNC)s 条 ⇒「%(AFVERD)s」；只读仪器本体 stdout %(AFB)s B＝三联 %(TRIPLE)s、判读行「%(AFVERDICT)s」；归档自查件 %(ARCHPYB)s B（md5 %(ARCHPYM)s）及其 stdout %(ARCHOUTB)s B／stderr %(ARCHERRB)s B＝声称 %(ARCHCLAIMS)s 条全部命中原样、正对照逐行数到 %(ARCHNCTRL)s 条 ⇒「%(ARCHVERD)s」；落地腿 %(LANDPYB)s B 与其预检／写盘两面 %(LANDPREB)s／%(LANDWB)s B（含 %(LANDPREN)s 次红灯现场）一并入库。
"""

OVER_T = """- **%(TIME)s（分片 %(SLICE)s 提交腿的写盘前登记；生产树 src+tests 改动 %(NCODE)s、GPU 全程未碰）**（开发日志 §%(SECREF)s「分片 %(SLICE)s＝提交腿：分片 %(PREVSLICE)s 的落地半腿整条链入库」）：
  - **入库面**：替上一片入库整条链——`%(PREVLOG)s` **%(R30B)s** B／**%(R30L)s** 行（`BODY_OFFSET=%(OFF30)s`＝表头 %(OFF30)s＋正文 %(BODY30)s，判读行「%(VERD30)s」、闭合式「%(CLOSURE30)s」）＋提交说明控件面（stdout **%(MSGCTLB)s** B，md5 前 12 `%(MSGCTLM)s`＝扰动 **%(MSGNCASE)s** 条、造不出红路 **%(MSGNUNC)s** 条）＋表头控件面（stdout **%(HDRCTLB)s** B＝红路 **%(HDRNRED)s**／保持绿 **%(HDRNGREEN)s**／本面自查 **%(HDRNSELF)s**／造不出红路 **%(HDRNUNC)s**）＋只读仪器与其控件面（仪器 stdout **%(AFB)s** B＝三联 %(TRIPLE)s；控件面 **%(AFCTLB)s** B＝红路 %(AFNCASE)s／造不出红路 %(AFNUNC)s）＋归档自查件 **%(ARCHPYB)s** B（md5 前 12 `%(ARCHPYM)s`；声称 **%(ARCHCLAIMS)s** 条全命中、正对照 **%(ARCHNCTRL)s** 条能红）＋落地腿与两次红灯现场（%(LANDPYB)s B／预检 %(LANDPREB)s B／写盘 %(LANDWB)s B）。落笔现数＝未跟踪 **%(UNT)s**＋修改 **%(MOD)s**＝**%(TOT)s** 处，分组＝%(BKT)s。远端 main 仍为 `%(HEAD7)s`（父 `%(PARENT7)s`，提交时刻 %(CDATE)s）。
  - **两条本轮口径**：① **%(LESSON)s**（承接上一轮）——控件面两次真跑各抓到一条**按构造红不了**的判据，反过来把正确实现判红：收面自查的标志位恒 False、判红的合取要求 `SystemExit:` 前缀而 CPython 对未捕获的 `SystemExit(str)` 只写裸消息；修法＝谓词抽成纯函数＋配不花配额的内存判定表＋逐条即时落盘。② **%(LESSON_NEW)s**——`nvidia-smi` 的 NVML 本轮起不来（内核模块 `%(DRV)s` ≠ 用户态库 `%(LIB)s`，rc=%(NVRC)s），「我没占卡」那条断言改由 `/proc` argv[0] 整词匹配承担（分母 %(NPROCS)s、我的进程 %(NVME)s 个，正／负对照各一条；松散口径同一条清单现数 %(LOOSE)s 个，合成对照＝松散 1／整词 0）；同轮另发现 `git status --porcelain` 默认口径把全新目录折成一行 ⇒ 计数一律 `-uall`。
  - **仪器与排期**：尺子＝md5 前 12 `%(FAM)s` 一族现数 **%(RULE_N)s** 份（同一条 glob 形状 **%(RULE_GLOB)s** 份，差额有名字＝%(STRAY)s），本片抄第 **%(RULE_NEW)s** 份；表头控件面每条扰动复跑一次联网尺子 ⇒ 约 **%(HDRNRED)s** 次 GitHub 未认证 API 调用、配额 60/小时，各面分开排在复位之后；继承常量＝子运行上限 %(SUBTO)s s／环境故障累计 %(ENVSTOP)s 条即停（超时既不判红也不判过）。排期与红线不变（原文见 §%(PREVSECREF)s 的钉住子条与计划文档 §W 补记 %(WN)s）：**#30 试片尺寸裁决在前**、`--stage full` 暂停、A0 断言一字不动、#27 不换指标、无新增 skip/xfail、性能数字只在 GPU 窗口测（本轮 NVML 不可用，更谈不上吞吐）。本片落地读数此刻尚不存在 ⇒ 提交后现取并交**分片 %(NEXTSLICE)s** 自查。
"""

PLAN_T = """
### §W 补记 %(WN)s（%(TIME)s 落笔，`date '+%%F %%T %%z'` 现取）＝分片 %(SLICE)s 提交腿的**写盘前登记**：替分片 %(PREVSLICE)s 入库它的整条落地半腿链，并把两条**仪器侧**事实折进流程

- **外部事实只有一条**：远端 main 已在 `%(HEAD7)s`（`git ls-remote --heads origin main` 与本地 HEAD 同名，落笔时现验）；本片＝**纯归档腿**，生产树 `src`＋`tests` 改动 **%(NCODE)s** 件。入库面＝未跟踪 **%(UNT)s**＋修改 **%(MOD)s**＝**%(TOT)s** 处（`git status --porcelain --untracked-files=all` 现数；分组＝%(BKT)s）；`%(PREVLOG)s` 的判读行「%(VERD30)s」与闭合式「%(CLOSURE30)s」由档案正则现取。**本片自己的落地证明要到提交之后才存在**，本补记只登记计划与写盘前读数。
- **两条仪器侧事实（不是判据缺陷，也不是偏好）**：① **%(LESSON)s**——上一轮控件面两次真跑各抓到一条按构造红不了的判据（收面自查的标志位恒 False；判红合取要求 `SystemExit:` 前缀而 CPython 对未捕获的 `SystemExit(str)` 只把消息原样写进 stderr），两处都把**正确实现**判成红，第二次还让"闸门响了"记成"未命中"。⇒ 判据的形状要按实测重写、并且**判"红"的方式自己也要能被造一次红**。② **%(LESSON_NEW)s**——`nvidia-smi --query-compute-apps=pid` 本轮 rc=%(NVRC)s（「%(NVWHY)s」；内核模块 `%(DRV)s` 与用户态 NVML 库 `%(LIB)s` 版本不同），而这条命令正是「我的解释器没占卡」那条红线唯一的出处；本轮不换口径、不假装量到，改用 `/proc` argv[0] **整词**匹配（分母 %(NPROCS)s 个 PID、我的进程 %(NVME)s 个、正／负对照各一条，松散口径同一条清单现数 %(LOOSE)s 个，合成对照＝松散 1／整词 0）。同轮另发现 `git status --porcelain` 的默认 `-u normal` 会把全新目录折成一行 ⇒ 入库计数一律 `-uall`。
- **继承的仪器常量与代价登记**：表头控件面的三条自查（%(HDRNSELFN)s）与三道防护本轮继承（子运行上限 %(SUBTO)s s、环境故障累计 %(ENVSTOP)s 条即停、raise 行号窗口 ±%(WINDOW)s 行；超时**既不判红也不判过**、单独计数）。三面原始件在档：提交说明控件面 stdout **%(MSGCTLB)s** B（md5 前 12 `%(MSGCTLM)s`／stderr **%(MSGERRB)s** B）＝扰动 %(MSGNCASE)s 条、造不出红路 %(MSGNUNC)s 条 ⇒「%(MSGVERD)s」；表头控件面 stdout **%(HDRCTLB)s** B（md5 前 12 `%(HDRCTLM)s`／stderr **%(HDRERRB)s** B）＝红路 %(HDRNRED)s 条、保持绿 %(HDRNGREEN)s 条、超时 0 条、造不出红路 %(HDRNUNC)s 条 ⇒「%(HDRVERD)s」；只读仪器的控件面 stdout **%(AFCTLB)s** B（md5 前 12 `%(AFCTLM)s`／stderr **%(AFERRB)s** B）＝红路 %(AFNCASE)s 条、保持绿 %(AFNGREEN)s 条、造不出红路 %(AFNUNC)s 条 ⇒「%(AFVERD)s」；归档自查件 stdout **%(ARCHOUTB)s** B／stderr **%(ARCHERRB)s** B＝声称 %(ARCHCLAIMS)s 条全命中、正对照 %(ARCHNCTRL)s 条 ⇒「%(ARCHVERD)s」。两次红灯现场（attempt1 %(A1B)s B／%(A1EB)s B、attempt2 %(A2B)s B／%(A2EB)s B，判读「%(A2VERD)s」）一并入库。⚠ 表头控件面每条扰动复跑联网尺子 ⇒ 约 %(HDRNRED)s 次 GitHub 未认证 API 调用、配额 60/小时，各面分开排在复位之后（这是可行性代价登记，不是性能数字）。
- **排期与红线（不变，且依赖仍未松动）**：**#30 试片尺寸裁决在前**（G9 触边闸与「只有 N=1 可评分」的原文见 §%(G9SEC)s，节号由该短语在开发日志里现定位、并配一条"复制到另一节就要数出 2 节"的正对照，本轮未重跑 ⇒ 不复述数字）→ raw 臂 N≥2 ＋ `--stage full --measured-per-track` → **#27 腿②（缺省 tuned→hk）＋#33 的位点系数同批，且只对 A3 的 1→18 外部趋势打分** → #40→#39；#41；#37／#38；#29 等裁决；#51 已量化但**不关单**。红线：A0 断言一字不放宽、#27 不换指标、无新增 skip/xfail、性能数字只在 GPU 窗口测（本轮全程未碰 GPU，且 NVML 普查不可用 ⇒ 无任何吞吐论断）。本片回执链与本轮控件／自查件全在工作树 ⇒ 交**分片 %(NEXTSLICE)s**。

"""

DAILY_T = """
## %(TIME)s：分片 %(SLICE)s 提交腿＝替分片 %(PREVSLICE)s 入库它的整条落地半腿链（未跟踪 %(UNT)s＋修改 %(MOD)s＝%(TOT)s 处）；一条环境级新学＝GPU 普查那把尺子（NVML）坏了

- **一句话**：远端 main ＝ `%(HEAD7)s`（父 `%(PARENT7)s`，提交时刻 %(CDATE)s，`git log -1 --format=%%cd` 现取），HEAD 主题＝「%(SUBJ)s…」（片号由此现取 ⇒ 本片＝%(SLICE)s）；本节写的是**提交腿落笔前**的登记，落地读数此刻还不存在。
- **入库面（分组由文件名模式现数，和＝%(BKTSUM)s 与 `grep -c '^??'` 相符）**：%(BKT)s；另加 %(MOD)s 张已跟踪档案面（%(MODTXT)s，其中 `%(PREVDAILYNAME)s` 由上一片**提交之后**的落地填值件追加）＝未跟踪 **%(UNT)s**＋修改 **%(MOD)s**＝**%(TOT)s** 处。⚠ 本轮这条计数改用 `--untracked-files=all`：默认口径把全新的 `%(EVIDNEW)s/` 折成一行，件数会凭空少算。这些数只描述落笔那一刻，分片 %(NEXTSLICE)s 必须重新现取。
- **两条本轮口径**：① **%(LESSON)s**（承接上一轮）——控件面两次真跑各抓到一条按构造红不了的判据，反过来把正确实现判红；修法＝自审正向双判据、谓词按实测形状重写并配不花配额的内存判定表、逐条即时落盘。② **%(LESSON_NEW)s**——`nvidia-smi` 本轮 rc=%(NVRC)s（「%(NVWHY)s」，内核模块 `%(DRV)s` ≠ 用户态库 `%(LIB)s`）⇒「我没占卡」改由 `/proc` argv[0] 整词匹配承担（我的进程 %(NVME)s／分母 %(NPROCS)s，正／负对照各一条，松散口径同一条清单现数 %(LOOSE)s 个、合成对照＝松散 1／整词 0）。
- **上一片的件在档可查（引用即从档案现取，不转述）**：`%(PREVLOG)s` %(R30B)s B／%(R30L)s 行（`BODY_OFFSET=%(OFF30)s`）＝「%(VERD30)s」、闭合式「%(CLOSURE30)s」；三面控件面＝提交说明 %(MSGCTLB)s B（扰动 %(MSGNCASE)s／UNCOVERED %(MSGNUNC)s）、表头 %(HDRCTLB)s B（红 %(HDRNRED)s＋绿 %(HDRNGREEN)s／UNCOVERED %(HDRNUNC)s／自查 %(HDRNSELF)s）、只读仪器 %(AFCTLB)s B（红 %(AFNCASE)s＋绿 %(AFNGREEN)s／UNCOVERED %(AFNUNC)s），三面判读现比一致＝%(VERD3)s；只读仪器本体三联 %(TRIPLE)s＝「%(AFVERDICT)s」；归档自查件 %(ARCHPYB)s B（md5 %(ARCHPYM)s）stdout %(ARCHOUTB)s B／stderr %(ARCHERRB)s B＝声称 %(ARCHCLAIMS)s 条全命中、正对照 %(ARCHNCTRL)s 条 ⇒「%(ARCHVERD)s」；两次红灯现场 attempt1 %(A1B)s／%(A1EB)s B、attempt2 %(A2B)s／%(A2EB)s B。
- **钉住与红线**：每条计算命令逐条内联 `CUDA_VISIBLE_DEVICES=\"\"`（含一行式）；**不产生吞吐／加速／预算数字**；生产树改动 **%(NCODE)s** 件；尺子＝md5 前 12 `%(FAM)s` 一族现数 %(RULE_N)s 份（同一 glob 形状 %(RULE_GLOB)s 份，差额＝%(STRAY)s），本片抄第 %(RULE_NEW)s 份；GPU＝NVML 普查不可用，替代断言 我的解释器进程 **%(NVME)s/%(NPROCS)s**。排期与红线不变（原文见 §%(PREVSECREF)s 与计划文档 §W 补记 %(WN)s）。
"""

WBIDX_T = """- **%(TIME)s（分片 %(SLICE)s 提交腿的写盘前登记；生产树 0 改动、GPU 全程未碰）**：
  本片替上一片入库它的整条落地半腿链：`%(PREVLOG)s`（%(R30B)s B／%(R30L)s 行、`BODY_OFFSET=%(OFF30)s`，判读行「%(VERD30)s」、闭合式「%(CLOSURE30)s」）＋三面控件面（提交说明 %(MSGCTLB)s B＝扰动 %(MSGNCASE)s／UNCOVERED %(MSGNUNC)s；表头 %(HDRCTLB)s B＝红 %(HDRNRED)s＋绿 %(HDRNGREEN)s／UNCOVERED %(HDRNUNC)s／自查 %(HDRNSELF)s；只读仪器 %(AFCTLB)s B＝红 %(AFNCASE)s／UNCOVERED %(AFNUNC)s）＋只读仪器本体（stdout %(AFB)s B＝三联 %(TRIPLE)s）＋归档自查件（%(ARCHPYB)s B，声称 %(ARCHCLAIMS)s 条全命中、正对照 %(ARCHNCTRL)s 条）＋落地腿与 %(LANDPREN)s 次红灯现场。
  入库面＝未跟踪 **%(UNT)s**＋修改 **%(MOD)s**＝**%(TOT)s** 处，分组＝%(BKT)s；`src`＋`tests` 改动 **%(NCODE)s** 件 ⇒ 纯归档腿。远端 main `%(HEAD7)s`（父 `%(PARENT7)s`）。
  **两条仪器侧新学**：① **%(LESSON)s**（承接）——控件面按构造红不了的判据会把正确实现判红，「判红的方式」自己也要能被造一次红；② **%(LESSON_NEW)s**——NVML 普查本轮坏了（内核模块 `%(DRV)s` ≠ 用户态库 `%(LIB)s`）⇒「我没占卡」改由 `/proc` argv[0] 整词匹配（我的进程 %(NVME)s／分母 %(NPROCS)s，正／负对照；松散口径同一条清单现数 %(LOOSE)s 个、合成对照＝松散 1／整词 0），另加一条：`git status` 默认口径把全新目录折成一行 ⇒ 计数一律 `-uall`。
  方法账沿用：尺子按 md5 分组数＝%(RULE_N)s 份（glob 形状 %(RULE_GLOB)s 份，差额＝%(STRAY)s），本片抄第 %(RULE_NEW)s 份；表头控件面每条扰动复跑联网尺子 ⇒ 约 %(HDRNRED)s 次 API 调用、配额 60/小时，各面分开排在复位之后。
  排期不变（**#30 试片尺寸裁决在前**、`--stage full` 暂停，原文见 §%(PREVSECREF)s／§W 补记 %(WN)s）；红线：A0 断言一字不动、无新增 skip/xfail、性能数字只在 GPU 窗口测。本片落地读数此刻不存在 ⇒ 提交后现取并交**分片 %(NEXTSLICE)s**。
"""

FACES = {"DEV": ("dev", DEV_T), "OVER": ("over", OVER_T), "PLAN": ("plan", PLAN_T),
         "DAILY": ("daily", DAILY_T), "WBIDX": ("wbidx", WBIDX_T)}


def must_reject(label, tmpl):
    """named() 的负对照：坏模板必须被拦下，否则那道门是恒过的装饰。

    未取值的槽由 `%` 运算抛 KeyError（不是断言），那也是"拦下"，所以两种都算命中。
    """
    try:
        named(label, tmpl)
    except (AssertionError, KeyError):
        return True
    raise SystemExit("正对照失败：%s 这类模板本该被 named() 拦下却过了 ⇒ 那道门是装饰" % label)


assert "x % y" == named("自检/好", "x %% y"), "具名替换把双写百分号渲染错了 ⇒ 渲染结果不可信"
assert "x " + CTX["SLICE"] + " y" == named("自检/好", "x %(SLICE)s y"), "具名槽没被替换"
must_reject("自检/裸百分号", "git log --format=%cd 现取")
must_reject("自检/全角槽", "槽＝%（SLICE）s")
must_reject("自检/未替换槽", "值＝%(NO_SUCH_KEY_READING)s")
print("== named() 自检：好模板渲染逐字节正确、裸百分号／全角槽／未取值的槽各被拦下 -> PASS")

texts = {}
for k, (_, tmpl) in FACES.items():
    texts[k] = named(k, tmpl)

# ============================ 结构闸门 ============================
print("== 结构闸门")


def strip_code(s):
    return re.sub(r"`[^`\n]*`", "", s)


for k, v in texts.items():
    stripped = strip_code(v)
    bad = [i for i, l in enumerate(stripped.splitlines(), 1) if l.count("（") != l.count("）")]
    assert not bad, "%s 有 %d 行全角括号不配平（行号 %s）" % (k, len(bad), bad[:4])
    assert stripped.count("**") % 2 == 0, "%s 剥内联码后粗体标记不成对" % k
    assert v.endswith("\n"), "%s 末字节不是换行（追加会粘连）" % k
    print("   %s：%d 行｜括号按行配平｜粗体成对｜尾换行｜无残留槽 -> PASS" % (k, len(v.splitlines())))

assert texts["DEV"].count("### 26.%d " % SECN) == 1, "§26.%d 标题不在恰好 1 处" % SECN


def secrefs(v):
    """节号引用必须写成开发日志自己的形制 `§26.NN`；裸 `§33` 读者找不到那一节。

    计划文档的字母节（§W／§L／§R…）与 `§` 后不带数字的写法不在拦阻范围，所以这条门只针对数字形制。
    """
    return re.findall(r"§(?!26\.)\d+", v)


for k, v in texts.items():
    bad = secrefs(v)
    assert not bad, "%s 有裸节号 %s ⇒ 引用要写成 §26.NN，否则读者按 §33 搜不到标题" % (k, bad[:4])
_bad = secrefs(texts["DEV"].replace("§26.%d" % PREVSEC, "§%d" % PREVSEC, 1))
assert len(_bad) == 1, "裸节号检测器在写坏的样本上抓到 %d 处（须 1）⇒ 它没有分辨力" % len(_bad)
print("   五面节号形制全部＝§26.NN（检测器对自己的坏样本抓到 1 处）-> PASS")
_lab = [m.group(1) for m in re.finditer(r"^- \*\*(.+?)\*\*", texts["DEV"], re.M)]
assert len(_lab) == 7 and len(set(_lab)) == 7, "§%d 顶级子条＝%d 条、互异标签＝%d 条（须 7 且互异）" % (
    SECN, len(_lab), len(set(_lab)))

NEEDLES = (("引用上一片回执名", re.escape(PREVLOG)),
            ("钉住口径登记", r"当前「[^」]+」"),
            ("承接的 LESSON 锚句", re.escape(LESSON)),
            ("本轮新学的锚句", re.escape(LESSON_NEW)))
BAIT = {"引用上一片回执名": "`%s`" % PREVLOG, "钉住口径登记": "当前「诱饵口径」",
        LESSON: "**%s**诱饵" % LESSON, LESSON_NEW: "**%s**诱饵" % LESSON_NEW}
for what, pat in NEEDLES:
    n = len(re.findall(pat, texts["DEV"], re.M))
    assert n == 1, "%s 在 §%d 里命中 %d 处（须恰好 1）⇒ 表头生成器的节内唯一性会拒绝归档" % (what, SECN, n)
    key = what if what not in ("承接的 LESSON 锚句", "本轮新学的锚句") else (
        LESSON if what == "承接的 LESSON 锚句" else LESSON_NEW)
    bait = "\n- **诱饵**：见 %s。\n" % BAIT[key]
    assert len(re.findall(pat, texts["DEV"] + bait, re.M)) == 2, \
        "%s 的正对照空转（加一条复述后仍不是 2 处）⇒ 这条唯一性断言是恒真式" % what
print("   §%d 四条节内唯一性各恰好 1 处，且各配一条「再复述就变成 2 处」的正对照 -> PASS" % SECN)

ovtxt = open(OVERVIEW, encoding="utf-8").read()
assert ovtxt.count(ANCHOR) == 1, "总览插入锚点 `%s` 现数 %d（须 1）" % (ANCHOR, ovtxt.count(ANCHOR))
for path in (DEVLOG, OVERVIEW, PLAN, DAILY, WBIDX):
    assert open(path, "rb").read()[-1:] == b"\n", "%s 末字节不是换行 ⇒ 追加会粘连" % path
print("   总览锚点恰好 1 处；五个追加／插入目标末字节都是换行 -> PASS")


def expect_near(label, text, pairs):
    for left, val in pairs:
        assert val, "%s：值为空" % label
        assert (left + val) in text, "%s：搜不到「%s%s」⇒ 读数没钉在自己的名词后面（位置错位的形状）" % (
            label, left, val)


expect_near("DEV", texts["DEV"], [
    ("未跟踪 **", CTX["UNT"]),
    ("档案名＝`", CTX["PREVLOG"]),
    ("`BODY_OFFSET=", CTX["OFF30"]),
    ("version`＝`", CTX["DRV"]),
    ("交**分片 ", CTX["NEXTSLICE"]),
    ("分组：", BKT_TXT.split("／")[0]),
])
expect_near("OVER", texts["OVER"], [("未跟踪 **", CTX["UNT"]), ("md5 前 12 `", CTX["FAM"]),
                                    ("我的进程 ", CTX["NVME"])])
expect_near("PLAN", texts["PLAN"], [("远端 main 已在 `", CTX["HEAD7"]),
                                    ("内核模块 `", CTX["DRV"]),
                                    ("＝扰动 ", CTX["MSGNCASE"]),
                                    ("原文见 §", CTX["G9SEC"])])
expect_near("DAILY", texts["DAILY"], [("远端 main ＝ `", CTX["HEAD7"]),
                                      ("闭合式「", CTX["CLOSURE30"])])
expect_near("WBIDX", texts["WBIDX"], [("入库面＝未跟踪 **", CTX["UNT"]), ("远端 main `", CTX["HEAD7"])])
print("   邻接断言：五面各 2–6 对「标签＋值」在渲染结果里原文命中 -> PASS")

os.makedirs(OUT_DIR, exist_ok=True)
for k, v in texts.items():
    open(os.path.join(OUT_DIR, k + ".md"), "w", encoding="utf-8").write(v)
open(os.path.join(OUT_DIR, "ctx.txt"), "w", encoding="utf-8").write(
    "\n".join("%s=%s" % (kk, vv) for kk, vv in sorted(CTX.items())) + "\n")
print("== 渲染完成：%s/{DEV,OVER,PLAN,DAILY,WBIDX}.md ＋ ctx.txt" % OUT_DIR)


# ============================ 写盘腿（带守卫、一次性） ============================
def running_faces():
    """正在跑的**别的**仪器：按 `/proc/PID/cmdline` 的**整词**比对 basename，不按命令行子串猜。

    本件的名字会命中**自己**和承载自己的那条 wrapper（`bash -c`／`timeout`，它们的 argv[0] 不是解释器）⇒
    自己这一支按「argv[0]＝解释器且带着本件名」单独数副本，wrapper 不参与，否则这道守卫会把调用它的那一轮判红。
    """
    hits, copies = [], 0
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        blob = cmdline_raw(d)
        toks = {os.path.basename(a.decode("utf-8", "replace")) for a in blob.split(b"\0") if a}
        if SELF_NAME in toks:
            if blob.split(b"\0")[0].decode("utf-8", "replace") == INTERP:
                copies += 1
            continue
        if toks & TOKENS:
            hits.append((d, sorted(toks & TOKENS)))
    return hits, copies


if "--write" in sys.argv:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "写盘腿没钉 CVD 空串 ⇒ 与 ENV 口径不符"
    faces, copies = running_faces()
    assert not faces, "有别的仪器正在跑 %s ⇒ 按口径不许动它可读的档案面" % faces
    assert copies == 1, "以解释器身份跑本件的进程有 %d 个（须 1＝自己）⇒ 有两个副本同时在写档案，拒绝" % copies
    for path in (DEVLOG, OVERVIEW, PLAN, DAILY, WBIDX):
        assert open(path, "rb").read()[-1:] == b"\n", "%s 末字节不是换行 ⇒ 拒绝追加" % path
    pre = [l for l in sh("git", "-c", "core.quotePath=false", "status", "--porcelain",
                         "--untracked-files=all").splitlines() if l.strip()]
    for path, key in [(DEVLOG, "DEV"), (PLAN, "PLAN"), (WBIDX, "WBIDX")]:
        with open(path, "a", encoding="utf-8") as f:
            f.write(texts[key])
        print("   追加 %s：%d B" % (os.path.relpath(path, REPO), len(texts[key].encode("utf-8"))))
    with open(DAILY, "a", encoding="utf-8") as f:
        f.write(texts["DAILY"])
    print("   追加 %s：%d B" % (os.path.relpath(DAILY, REPO), len(texts["DAILY"].encode("utf-8"))))
    ins = open(OVERVIEW, encoding="utf-8").read()
    pos = ins.index(ANCHOR)
    with open(OVERVIEW, "w", encoding="utf-8") as f:
        f.write(ins[:pos] + texts["OVER"] + ins[pos:])
    print("   插入 总览：条目插在 `%s` 之前（偏移 %d）" % (ANCHOR, pos))
    print("== 闸门：写完回读")
    for path, key in [(DEVLOG, "DEV"), (OVERVIEW, "OVER"), (PLAN, "PLAN"), (DAILY, "DAILY"), (WBIDX, "WBIDX")]:
        now = open(path, encoding="utf-8").read()
        assert texts[key] in now, "%s 里搜不到刚写的面 ⇒ 写入没落地或被规范化" % path
        assert now.endswith("\n"), "%s 写完后末字节不是换行" % path
        print("   %s：逐字节搜到原样＋尾换行 -> PASS" % os.path.relpath(path, REPO))
    now_dev = open(DEVLOG, encoding="utf-8").read()
    m = list(re.finditer(r"^### 26\.(\d+)[^\n]*$", now_dev, re.M))
    assert int(m[-1].group(1)) == SECN, "磁盘上最后一节不是 §%d ⇒ 插入位置错了" % SECN
    seg = now_dev[m[-1].end():]
    nxt = seg.find("\n### ")
    seg = seg[:nxt] if nxt != -1 else seg
    for what, pat in NEEDLES:
        n = len(re.findall(pat, seg, re.M))
        assert n == 1, "磁盘上的 §%d 里 %s 命中 %d 处（片段数 1、整节数须 1）⇒ 回滚不了就别归档" % (
            SECN, what, n)
    print("   磁盘上的 §%d 四条唯一性各恰好 1 处（＝片段预检值）-> PASS" % SECN)
    post = [l for l in sh("git", "-c", "core.quotePath=false", "status", "--porcelain",
                          "--untracked-files=all").splitlines() if l.strip()]
    nq = len([l for l in post if l.startswith("??")])
    nm = len([l for l in post if not l.startswith("??")])
    assert nq == NU, "未跟踪件数从 %d 变成 %d ⇒ 写盘动到了不该动的地方" % (NU, nq)
    assert nm == NM, "修改面数从 %d 变成 %d ⇒ 追加写到了第五张面上" % (NM, nm)
    print("   闭合式复算：未跟踪 %d＋修改 %d＝%d 处（写盘前后同数，因五张面此前已在修改／未跟踪态）-> PASS"
          % (nq, nm, nq + nm))
    print("WRITER_VERDICT = FIVE FACES LANDED, UNIQUENESS RECOUNTED ON DISK")
else:
    print("（预检模式：未写盘。加 --write。）")
print("FILLER_VERDICT = READINGS DERIVED, STRUCTURE GATES PASS")

