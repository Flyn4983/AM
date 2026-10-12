#!/usr/bin/env python3
r"""分片 30 的**填值器**（＋带守卫的一次性写盘腿）：五张档案面的每个读数都由命令现取，具名替换后渲染。

三条口径（都是上一轮用真事故换来的，本件把它们当闸门而不是注释）：
① **具名替换＋邻接断言**——位置式 %s 在「槽位数相等＋顺序单调」两道检查同时绿的情况下仍能把散文排错位，
   所以这里一律 `%(中文键)s`，且每个值必须由 `expect_near` 钉在自己的标签后面。
② **渲染结构门**＝按行配平全角 `（）`、剥内联码后配平 `**`、写盘前查目标末字节是换行、插入锚点在整份文件
   里恰好 1 处、渲染结果里不许残留 `%(`／`%（`／`@@`。
③ **§26.32 的三条节内唯一性**（＝表头生成器 `bullet_for` 的真实约束，不是我自己加的更严的前提）＝引用上一片
   回执名的顶级子条恰好 1 条、带引号的钉住口径登记恰好 1 条、LESSON 锚句恰好 1 条；三条各配一条"再复述一次
   就必须变成 2 处"的正对照——没有正对照的唯一性断言是恒真式。

volatile 读数（未入库件数、卡上进程数、落笔 date）只进本轮散文，不写进任何可复跑件的 stdout。
默认只渲染到 `/tmp/out30/`；`--write` 才落盘，且落盘前后各数一次闭合式。
"""
import glob
import hashlib
import os
import re
import subprocess
import sys
import time

REPO = "/home/shy/桌面/2026年开发-显式冲击有限元-物质点法-sph-fvm+am/AM-Qoder"
EVID_OLD = os.path.join(REPO, "docs/evidence/2026-10-10")
DEVLOG = os.path.join(REPO, "docs/开发日志.md")
OVERVIEW = os.path.join(REPO, "docs/项目开发总览.md")
PLAN = os.path.join(REPO, "docs/项目评估与下一步计划_2026-10-06.md")
DAILY = os.path.join(REPO, ".workbuddy/memory/2026-10-12.md")
WBIDX = os.path.join(REPO, ".workbuddy/memory/MEMORY.md")
ANCHOR = "## 7. 文档索引"
PREVLOG = "am_s7_verify29.log"
LESSON = "对照的可满足性"
OUT_DIR = "/tmp/out30"
TALLY = os.path.join(OUT_DIR, "tally.txt")
# 写盘守卫：这些名字**正在跑**就不许动档案（按 /proc/PID/cmdline 的整词比对，不按命令行子串猜）
TOKENS = {"am_s7_verify30.py", "am_s30_hdr.py", "am_s30_msg.py", "am_s7_verify29.py",
          "am_s29_archive_check.py", "am_s30_hdr_controls.py", "am_s30_msg_controls.py",
          "s30_fill.py", "s30_write.py"}


def sh(*args):
    """**默认环境**跑（不钉 LC_ALL）——指纹口径只在默认 locale 下与前四片可比。"""
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

st = [l.rstrip("\n") for l in sh("git", "-c", "core.quotePath=false", "status", "--porcelain").splitlines()
      if l.strip()]
UNT = sorted(l[3:] for l in st if l.startswith("??"))
MOD = sorted(l[3:] for l in st if not l.startswith("??"))
NU, NM = len(UNT), len(MOD)
assert NU + NM == 28, "落笔现数＝%d＋%d＝%d 处（本节的闭合式写的是 28）⇒ 先确认工作树有没有被动过" % (
    NU, NM, NU + NM)

ROLES = [
    ("回执链 5", r"^am_s7_verify%d\.(py|log)$|^_raw_verify%d\.(out|err)$|^am_s%d_hdr\.py$"
     % (PREVSLICE, PREVSLICE, PREVSLICE)),
    ("表头控件面 3", r"^am_s%d_hdr_controls\.py$|^_raw_s%d_hdr_controls\.(out|err)$" % (PREVSLICE, PREVSLICE)),
    ("卡死现场 3", r"^_raw_s%d_hdr_controls\.attempt1-0byte\.(out|err)$|^_raw_s%d_hdr_hang_probe\.txt$"
     % (PREVSLICE, PREVSLICE)),
    ("表头预检与写盘两面 4", r"^_raw_s%d_hdr_(precheck|write)\.(out|err)$" % PREVSLICE),
    ("超时探针 3", r"^am_s%d_timeout_probe\.py$|^_raw_s%d_timeout_probe\.(out|err)$" % (PREVSLICE, PREVSLICE)),
    ("写盘后表头复跑 2", r"^_raw_s%d_hdr_after_write\.(out|err)$" % PREVSLICE),
    ("归档自查腿 3", r"^am_s%d_archive_check\.py$|^_raw_s%d_archive_check\.(out|err)$" % (PREVSLICE, PREVSLICE)),
    ("当日纪实 1", r"^2026-10-12\.md$"),
]
buckets, assigned = [], []
for name, pat in ROLES:
    hit = [p for p in UNT if re.match(pat, os.path.basename(p))]
    assert hit, "%s 一组现数为 0 ⇒ 命名规则与本目录脱节（分组句就成了假话）" % name
    buckets.append((name, len(hit)))
    assigned += hit
assert len(assigned) == len(set(assigned)) == NU, (
    "角色分组没恰好覆盖未跟踪清单（分组 %d／去重 %d／现数 %d）⇒ 有件落进 0 组或 ≥2 组"
    % (len(assigned), len(set(assigned)), NU))
BKT_TXT = "／".join("%s **%d**" % (n, c) for n, c in buckets)
BKT_SUM = "＋".join(str(c) for _, c in buckets) + "＝%d" % NU
MOD_TXT = "／".join(os.path.basename(p) for p in MOD)
assert NM == 4, "已跟踪档案面现数 %d（须 4）：%s" % (NM, MOD_TXT)
print("== 现取：HEAD＝%s（父 %s｜提交时刻 %s）｜待入库＝未跟踪 %d＋修改 %d＝%d 处｜src+tests 改动 %d｜片号＝%d（由 HEAD 主题现取）"
      % (HEAD7, PARENT7, CDATE, NU, NM, NU + NM, NCODE, SLICE))
print("   未跟踪分组（和＝%s）：%s" % (BKT_SUM, BKT_TXT))

# ---- 上一片回执：读数只从档案正则现取，并逐字节复验正文区 ----
R29 = os.path.join(EVID_OLD, PREVLOG)
r29 = open(R29, encoding="utf-8").read()
R29B, R29L = os.path.getsize(R29), len(r29.splitlines())
OFF29 = int(one(r"^BODY_OFFSET=(\d+)$", r29, "上一片回执的 BODY_OFFSET"))
head_part, body_part = r29.split("BODY_OFFSET=%d\n" % OFF29, 1)
assert len((head_part + "BODY_OFFSET=%d\n" % OFF29).encode()) == OFF29, "BODY_OFFSET 与实际表头字节数不等"
VERD29 = one(r"^(LANDING_VERDICT = .*)$", r29, "上一片回执的判读行")
CLOSURE29 = one(r"闭合式 远端 \d+ == 父提交 \d+ \+ 本片新增\(A\) \d+ ＝ \d+ -> PASS", r29, "上一片闭合式那句")
assert one(r"^TREE\s+: head=([0-9a-f]{40})", r29, "上一片 TREE 行哈希") == HEAD, "上一片回执的 head ≠ 本地 HEAD"
FINGER_REC = one(r"^SRC指纹 : ([0-9a-f]{32})", r29, "上一片回执表头的 SRC指纹")
RAW29 = os.path.join(EVID_OLD, "_raw_verify%d.out" % PREVSLICE)
assert open(RAW29, encoding="utf-8").read() == body_part, "回执正文区与原始 stdout 不逐字节相同"
NDEF29 = len([l for l in head_part.splitlines() if l.strip()])
print("   上一片回执＝%s（%d B／%d 行，BODY_OFFSET %d 已复算）｜判读行「%s」｜正文区与 `_raw_verify%d.out` 逐字节相同"
      % (PREVLOG, R29B, R29L, OFF29, VERD29, PREVSLICE))

# ---- 节号：从开发日志现取最后一个 ### 26.NN，并验证它的片号＝上一片 ----
devtxt = open(DEVLOG, encoding="utf-8").read()
_secs = list(re.finditer(r"^### 26\.(\d+)[^\n]*$", devtxt, re.M))
PREVSEC = int(_secs[-1].group(1))
assert int(one(r"分片 (\d+)＝提交腿", _secs[-1].group(0), "最后一节标题里的片号")) == PREVSLICE, \
    "§26.%d 的标题写的片号 ≠ 上一片 %d ⇒ 节号与片号的对应变了，本节要先重推" % (PREVSEC, PREVSLICE)
SECN = PREVSEC + 1
assert devtxt.count("### 26.%d " % SECN) == 0, "开发日志里已有 §26.%d ⇒ 本节要重跑先起新名" % SECN
NEXTSLICE = str(SLICE + 1)
# 引文要落在语法边界上：定长 [:30] 会切进子句中间（本轮主题实测就是切在「约束由」处）
SUBJ1 = SUBJ.split("＋")[0]
assert SUBJ1 and SUBJ1 in SUBJ and len(SUBJ1) <= 60, "主题第一子句异常：%r" % SUBJ1

# ---- 表头控件面 / 归档自查件 / 卡死现场 ----
CTLRAW = os.path.join(EVID_OLD, "_raw_s%d_hdr_controls.out" % PREVSLICE)
ctl = open(CTLRAW, encoding="utf-8").read()
CTLB, CTLERRB, CTLM = os.path.getsize(CTLRAW), os.path.getsize(
    os.path.join(EVID_OLD, "_raw_s%d_hdr_controls.err" % PREVSLICE)), md5(CTLRAW)[:12]
CTLVERD = one(r"^(CONTROL_VERDICT = .*)$", ctl, "表头控件面判读行")
CTLNCASE = int(one(r"CASES 声明＝(\d+) 条", ctl, "表头控件面扰动条数声明"))
CTLNUNC = int(one(r"造不出红路的闸门＝(\d+) 条", ctl, "表头控件面 UNCOVERED 数"))
CTLNSKIP = int(one(r"因超时没跑到的＝(\d+) 条", ctl, "表头控件面超时未跑数"))
CTLNTO = int(one(r"超时（环境故障，不判红也不判过）＝(\d+) 条", ctl, "表头控件面超时数"))
assert CTLNSKIP == CTLNTO == 0, "表头控件面有超时（未跑 %d／超时 %d）⇒ 它不能当全过的档案引用" % (CTLNSKIP, CTLNTO)
# 继承常量从上一片控件件**源码**现取（写死就会在别人改那两行后腐烂）
CTL_SRC = open(os.path.join(EVID_OLD, "am_s%d_hdr_controls.py" % PREVSLICE),
               encoding="utf-8").read()
SUBTO = one(r"^SUB_TIMEOUT = (\d+)$", CTL_SRC, "子运行上限常量")
ENVSTOP = one(r"^ENV_FAIL_STOP = (\d+)$", CTL_SRC, "环境故障累计常量")
WINDOW = one(r"行号落在该句源码位置的 ±(\d+) 行窗口", CTL_SRC, "raise 行号窗口常量")

ARCHPY = os.path.join(EVID_OLD, "am_s%d_archive_check.py" % PREVSLICE)
ARCHOUT = os.path.join(EVID_OLD, "_raw_s%d_archive_check.out" % PREVSLICE)
arch = open(ARCHOUT, encoding="utf-8").read()
ARCHPYB, ARCHPYM = os.path.getsize(ARCHPY), md5(ARCHPY)[:12]
ARCHOUTB, ARCHERRB = os.path.getsize(ARCHOUT), os.path.getsize(
    os.path.join(EVID_OLD, "_raw_s%d_archive_check.err" % PREVSLICE))
ARCHCLAIMS, ARCHMISS = one(r"声称条数＝(\d+)｜缺失＝(\d+)/\d+", arch, "自查件声称／缺失")
ARCHNCTRL = int(one(r"正对照共 (\d+) 条", arch, "自查件正对照条数"))
ARCHVERD = one(r"^(ARCHIVE_VERDICT = .*)$", arch, "自查件判读行")
assert ARCHMISS == "0", "自查件报出缺失 %s 条 ⇒ 不能引它当全过的档案" % ARCHMISS
COV = [(m.group(1), int(m.group(2)), int(m.group(3)))
       for m in re.finditer(r"承载：(.+?)＝(\d+) 条（其中独有 (\d+) 条）", arch)]
assert len(COV) == 5, "承载表只有 %d 行 ⇒ 档案面少了一面" % len(COV)
DEVCOV = [c for c in COV if c[0].endswith("开发日志.md")][0]
ZERO_UNIQ = [os.path.basename(f) for f, n, u in COV if u == 0]
assert len(ZERO_UNIQ) == 4, "独有＝0 的面现数 %d（上一轮登记的是 4）⇒ 承载分布变了" % len(ZERO_UNIQ)

HANG = open(os.path.join(EVID_OLD, "_raw_s%d_hdr_hang_probe.txt" % PREVSLICE), encoding="utf-8").read()
_rows = [l.split() for l in HANG.splitlines()
         if re.fullmatch(r"\d+ \d+ +\d+ S", l)]
assert _rows, "卡死现场件里抠不出 ps 行 ⇒ 那句卡了多久没有出处"
_ET = {r[0]: int(r[2]) for r in _rows}
_CMD = dict((m.group(1), m.group(2)) for m in
            re.finditer(r"^(\d+): (.*)$", HANG, re.M))
_hung = [pid for pid, c in _CMD.items() if "zctl_s%d_hdr" % PREVSLICE in c]
assert len(_hung) == 1, "按扰动副本名定位卡死进程命中 %d 个（须 1）：%s" % (len(_hung), _hung)
# 卡死的是**那条子运行**（扰动副本），不是控件面父进程：父进程起跑更早，两者 etimes 差＝副本晚起的秒数
HANGS = _ET[_hung[0]]
GAP = max(_ET.values()) - HANGS
assert GAP > 0, "扰动副本的 etimes 不小于父进程 ⇒ 主从关系判反了，这句要重推"
HMS = "%dh%02dm" % (HANGS // 3600, (HANGS % 3600) // 60)
_DEVHANG = one(r"etimes＝\*\*(\d+)\*\* s", open(DEVLOG, encoding="utf-8").read(),
               "开发日志里那句卡死时长")
assert int(_DEVHANG) == HANGS, "现取 %d s ≠ 开发日志 §26.%d 的 %s s ⇒ 两处有一处是手打的" % (
    HANGS, PREVSEC, _DEVHANG)
PROVERD = one(r"^(TIMEOUT_GUARD_VERDICT = .*)$",
              open(os.path.join(EVID_OLD, "_raw_s%d_timeout_probe.out" % PREVSLICE),
                   encoding="utf-8").read(), "超时探针判读行")
print("   表头控件面 stdout %d B（md5 %s／stderr %d B）＝扰动 %d／造不出红路 %d／超时 0 ⇒「%s」" % (
    CTLB, CTLM, CTLERRB, CTLNCASE, CTLNUNC, CTLVERD))
print("   归档自查件 %d B（md5 %s）stdout %d B／stderr %d B＝声称 %s 条全命中、正对照 %d 条 ⇒「%s」" % (
    ARCHPYB, ARCHPYM, ARCHOUTB, ARCHERRB, ARCHCLAIMS, ARCHNCTRL, ARCHVERD))
print("   卡死现场：扰动副本 PID %s 的 etimes＝%d s ⇒ %s（父进程最长 %d s，差 %d s＝副本晚起）｜与开发日志 §26.%d 现算同值｜超时探针「%s」"
      % (_hung[0], HANGS, HMS, max(_ET.values()), GAP, PREVSEC, PROVERD))

# ---- 尺子族：按 md5 分组数 ----
FAM = "0d20117bb82f"
allr = glob.glob(os.path.join(REPO, "docs/evidence/*/am_s7_verify*.py"))
fam_files = sorted(p for p in allr if md5(p)[:12] == FAM)
stray = sorted(os.path.basename(p) for p in allr if md5(p)[:12] != FAM)
RULE_N, RULE_GLOB = len(fam_files), len(allr)
assert RULE_N == 10, "md5 族现数 %d 份（散文写的是 10）⇒ 分组句要先改再写" % RULE_N
print("   尺子＝md5 前 12 `%s` 一族现数 %d 份（同一 glob 形状 %d 份，差额有名字＝%s）⇒ 本片抄第 %d 份"
      % (FAM, RULE_N, RULE_GLOB, "、".join(stray), RULE_N + 1))

# ---- SRC 指纹：两条 locale 各现算一次（本轮新学的实证，也是它自己的靶） ----
FING = ("find src tests -name '*.py' -not -path '*__pycache__*' -print0 "
        "| sort -z | xargs -0 md5sum | md5sum | cut -d' ' -f1")
FINGER_DEF = subprocess.run(["bash", "-c", FING], cwd=REPO, capture_output=True, text=True).stdout.strip()
# `LC_ALL=C find … | sort -z` 只把变量给 find，pipeline 里的 sort 仍继承 zh_CN.UTF-8 ⇒ 必须整个 shell 导出
_env = dict(os.environ, LC_ALL="C")
FINGER_C = subprocess.run(["bash", "-c", FING], cwd=REPO, capture_output=True, text=True,
                          env=_env).stdout.strip()
assert FINGER_DEF == FINGER_REC, "默认 locale 现算 %s ≠ 上一片回执 %s ⇒ 生产树内容真的变了" % (
    FINGER_DEF, FINGER_REC)
assert FINGER_C != FINGER_DEF, "两种 locale 现算相同 ⇒ 本轮那条新学没有靶了（要先确认它还存在）"
assert re.fullmatch(r"[0-9a-f]{32}", FINGER_C), "LC_ALL=C 读数不像 md5 摘要＝%r ⇒ 命令本身跑坏了" % FINGER_C
print("   SRC 指纹：默认 locale＝%s（＝四份回执表头的值）｜LC_ALL=C＝%s ⇒ 差异只来自 sort 的 collation"
      % (FINGER_DEF, FINGER_C))

# ---- GPU：分母＋能红的匹配式 ----
NVPID = [l.split(",")[0].strip() for l in sh("nvidia-smi", "--query-compute-apps=pid",
                                             "--format=csv,noheader").splitlines() if l.strip()]
mine = [p for p in NVPID if os.path.exists("/proc/%s/cmdline" % p)
        and "/tmp/amvenv" in open("/proc/%s/cmdline" % p, "rb").read().decode("utf-8", "replace")]
assert "/tmp/amvenv" in ("/tmp/amvenv/bin/python x.py".replace("\x00", " ")), "匹配式的正对照自己失效了"
assert not mine, "我的解释器在 GPU 上留有 %s ⇒ 与「不碰 GPU」冲突" % mine
GPU, NVME = str(len(NVPID)), str(len(mine))
print("   GPU 现取：卡上 compute 进程 %s 个、我的解释器所留 %s/%s（总数只作断言，不入可复跑件）" % (GPU, NVME, GPU))

CTX = {
    "TIME": TIME, "HEAD7": HEAD7, "PARENT7": PARENT7, "SUBJ": SUBJ1, "CDATE": CDATE,
    "SLICE": str(SLICE), "PREVSLICE": str(PREVSLICE), "NEXTSLICE": NEXTSLICE,
    "SECN": str(SECN), "PREVSEC": str(PREVSEC),
    # 散文里的节号要写成开发日志自己的形制 §26.NN（写 §32 会让读者找不到那一节）
    "SECREF": "26.%d" % SECN, "PREVSECREF": "26.%d" % PREVSEC,
    "UNT": str(NU), "MOD": str(NM), "TOT": str(NU + NM),
    "BKT": BKT_TXT, "BKTSUM": BKT_SUM, "MODTXT": MOD_TXT,
    "PREVLOG": PREVLOG, "R29B": str(R29B), "R29L": str(R29L), "OFF29": str(OFF29),
    "NDEF29": str(NDEF29), "VERD29": VERD29, "CLOSURE29": CLOSURE29,
    "CTLB": str(CTLB), "CTLM": CTLM, "CTLERRB": str(CTLERRB), "CTLNCASE": str(CTLNCASE),
    "CTLNUNC": str(CTLNUNC), "CTLVERD": CTLVERD,
    "SUBTO": SUBTO, "ENVSTOP": ENVSTOP, "WINDOW": WINDOW,
    "ARCHPYB": str(ARCHPYB), "ARCHPYM": ARCHPYM, "ARCHOUTB": str(ARCHOUTB),
    "ARCHERRB": str(ARCHERRB), "ARCHCLAIMS": ARCHCLAIMS, "ARCHNCTRL": str(ARCHNCTRL),
    "ARCHVERD": ARCHVERD, "DEVCOV": "%d（独有 %d）" % (DEVCOV[1], DEVCOV[2]),
    "ZERO_UNIQ": "／".join(ZERO_UNIQ),
    "HMS": HMS, "HANGS": str(HANGS), "PROVERD": PROVERD,
    "FAM": FAM, "RULE_N": str(RULE_N), "RULE_NEW": str(RULE_N + 1),
    "RULE_GLOB": str(RULE_GLOB), "STRAY": "、".join(stray),
    "FINGER_DEF": FINGER_DEF, "FINGER_C": FINGER_C,
    "GPU": GPU, "NVME": NVME, "NCODE": str(NCODE), "LESSON": LESSON,
}

# ============================ 五张面 ============================
SEC_TITLE = ("### 26.%(SECN)s 分片 %(SLICE)s＝提交腿：分片 %(PREVSLICE)s 的归档半腿整条链入库"
             "（未跟踪 %(UNT)s＋修改 %(MOD)s＝%(TOT)s 处）＋一条**口径级**新学＝`LC_ALL=C` 会把 SRC 指纹"
             "换成另一个值（%(TIME)s 现取）\n\n") % CTX

DEV_T = SEC_TITLE + """- **形制约束的落实＝本轮另起 §%(SECREF)s，不在 §%(PREVSECREF)s 内追加**：上一节落地子条末尾登记的那条可核查约束——回执表头生成器的 `bullet_for` 要求「引用上一片回执档案名」的顶级子条在**同一节内唯一**——本节由结构满足：新节号、节内该句式恰好 1 条。本条刻意**不复述那个档案名**（在本节复述就会出现第二条命中，而那正是这条闸门要拦的事，所以它同时也是「这条闸能红」的证明）。同节另外两处唯一性一并照查：带引号的钉住口径登记恰好 1 条、LESSON 锚句恰好 1 条；写入件在落盘前对片段各数一次、落盘后再对磁盘上的整节数一次，两次不等就报错停止。
- **入库面（本条存在的唯一理由＝让分片 %(SLICE)s 的回执表头把「引用上一片回执」这件事从档案现取，而不是凭对话记忆）**：本节里唯一那条引用上一片回执名的子条就是这一条，档案名＝`%(PREVLOG)s`（%(R29B)s B／%(R29L)s 行，`BODY_OFFSET=%(OFF29)s` 已由写入件重解一遍并对上正文区；判读行「%(VERD29)s」；闭合式原文「%(CLOSURE29)s」）。落笔现数（`git status --porcelain`）＝未跟踪 **%(UNT)s**＋修改 **%(MOD)s**＝**%(TOT)s** 件。%(UNT)s 件按角色分组：%(BKT)s；%(BKTSUM)s 与 `grep -c '^??'` 现数相符（分组＝按文件名模式现数，一件落进 0 组或 ≥2 组就拒绝写盘）。修改 %(MOD)s 件＝四张已跟踪档案面（%(MODTXT)s）。本片自己的信息生成器与它的干跑控件件在提交前一并暂存 ⇒ 最终入库数**大于**这里的 %(TOT)s，那一部分由提交之后 `git show --name-status HEAD` 现取并写进本节的落地子条，不在这里预支。⚠ 本条里所有件数都是落笔那一刻的现取值，**下一轮必须重新现取**。
- **钉住口径与红线（当前「继续，不碰GPU」）**：本轮每条计算命令逐条内联 `CUDA_VISIBLE_DEVICES=""`（含一行式命令）；**不产生任何吞吐／加速／预算数字**，墙钟只作可行性代价登记；卡上 compute 进程总数（落笔现取 **%(GPU)s** 个）与「我的解释器所留 **%(NVME)s** 个」只作断言、总数不写进可复跑件（它随他人作业移动）；每次尝试换档案名，红件先入库再复跑；A0 断言一字不动、#27 不换指标、无新增 skip/xfail。生产树 `src`＋`tests` 现数改动 **%(NCODE)s** 件 ⇒ 本片仍是纯归档腿。
- **%(LESSON)s＝从分片 %(PREVSLICE)s 的归档半腿带进本片的口径**：那一轮抓到两处判断缺陷，**两处都在对照自己身上**。⑴ 「把某一面换成空文本 ⇒ 该面独有声称必须判缺」这条**按构造不可能满足**——另外四面是同源摘要，独有声称现数为 0（承载分布取自该件 stdout：开发日志 %(DEVCOV)s，其余四面（%(ZERO_UNIQ)s）独有 0 条），所以那条正对照永远红不了。正确顺序＝**先量每面独有承载**，再把锚点扰动指向真有独有声称的那一面，并保留「五面全空 ⇒ %(ARCHCLAIMS)s/%(ARCHCLAIMS)s 判缺」作兜底。⑵ 计数审计里**新造的等式本身就是一条未证声称**——`sum(各面独有)＝声称总数` 当场把自己的运行判红；可证的只有「覆盖＝总数」「1 ≤ 独有总数 ≤ 总数」「独有为 0 的面数＝4」。⇒ 一条对照必须**同时能红能绿**，而它的判据里不许出现我没测过的等式。
- **本轮新学（口径级，不是判据缺陷，也不是偏好）＝`LC_ALL=C` 与 SRC 指纹是耦合的**：路径差集要 `LC_ALL=C sort` 是既有纪律，而 SRC 指纹的口径 `find src tests -name '*.py' -not -path '*__pycache__*' -print0 | sort -z | xargs -0 md5sum | md5sum` 里那个 `sort` **也吃 locale**。同一条命令在两种环境里的现算：默认 locale（`LANG=zh_CN.UTF-8`，`__init__.py` 被 collation 排到 `asbuilt_*` 之后）＝`%(FINGER_DEF)s`，正是分片 26／27／28／%(PREVSLICE)s 四份回执表头里那个值；把它放进 `LC_ALL=C` 的 shell ＝`%(FINGER_C)s`。⇒ 本轮先把它读成"指纹漂移"，追到底才发现真因是**我自己为路径差集钉的 locale 泄漏进了指纹命令**。修法**不是**换口径（换了就与前四片不可比），而是：算指纹的那条命令不许同时钉 `LC_ALL`，路径清单与指纹分两条命令各跑各的；两个读数都留在本节，供下一轮自查件回搜。
- **跑前登记的闸门（分片 %(SLICE)s 的六条，逐条可红，跑后读数从归档件现取）**：① 尺子复用＝**逐字节副本**——`cp -p` 之后 md5 前 12 必须等于 `%(FAM)s` 这一族，份数按 **md5 分组**数而不是按 glob 命中数（落笔现数 %(RULE_N)s 份／同一条 glob 形状 %(RULE_GLOB)s 份，差额有名字＝%(STRAY)s；本片抄第 %(RULE_NEW)s 份）；② 五步落地证明（`ls-remote`／`truncated` 须 false／blob＋tree 两侧计数／双向路径差集且两侧同用 `core.quotePath=false` 与 `LC_ALL=C sort`／逐文件 blob SHA 普查带分母）＋闭合式 远端＝父提交＋count(A)，四个正对照 A／B／C／D 各自能红；③ 回执表头只由生成器产出（预检面与写盘面各留原始件，`BODY_OFFSET` 由不动点迭代，写盘后**另跑一件只读仪器**做 `cmp -i 偏移:0` 三联＝0/1/1，顶格字段数与生成器 `FIELDS` 声明互证）；④ 提交说明由信息生成器产出（PLACEHOLDER＋`str.replace`、双向查「未取值的字段」与「模板没用到的读数」、shortstat 三个数闭合到 `--numstat` 逐行现算、指纹四路闭合＝**默认 locale 下**现算＝回归起＝回归止＝上一片回执表头 SRC 指纹）；⑤ 两面干跑控件面按「rc≠0 **且**抛出的那一行含指定闸门句 **且**行号落在该句源码 ±%(WINDOW)s 行窗口」判红，并如实登记造不出红路的闸门；表头面每条扰动都要复跑一次联网尺子 ⇒ 约 %(CTLNCASE)s 次 GitHub 未认证 API 调用、配额 60/小时，本轮把两面分开排在配额复位之后，并继承上一片的新常量（子运行上限 %(SUBTO)s s、环境故障累计 %(ENVSTOP)s 条即停，超时**既不判红也不判过**、单独计数）；⑥ 归档自查件（新追加段读数普查＋下一片锚点复验＋逐字节复跑，且 stdout 不印任何会被档案引用的自指数）。
- **分片 %(SLICE)s 落地腿＝提交＋推送＋五步落地证明＋生成器回执表头＋两面控件面（%(TIME)s 现取；生产树 src＋tests 改动 %(NCODE)s、GPU 全程未碰）**：本条是**写盘前**登记的计划，不是结果——落地读数（本片 HEAD 全哈希、入库清单 A／M 计数、闭合式、回执表头字节与 `BODY_OFFSET`、两面控件面的扰动条数与 UNCOVERED）此刻**还不存在**；它们由提交之后 `git show --name-status HEAD` 与回执原始件现取，写进本节的落地子条，并交**分片 %(NEXTSLICE)s** 的归档半腿做自查。上一片的同类件在档可查，引用即从档案正则现取：表头控件面 stdout %(CTLB)s B（md5 %(CTLM)s／stderr %(CTLERRB)s B）＝扰动 %(CTLNCASE)s 条、造不出红路 %(CTLNUNC)s 条 ⇒「%(CTLVERD)s」；归档自查件 %(ARCHPYB)s B（md5 %(ARCHPYM)s）及其 stdout %(ARCHOUTB)s B／stderr %(ARCHERRB)s B＝声称 %(ARCHCLAIMS)s 条全部命中原样、正对照 %(ARCHNCTRL)s 条各自能红 ⇒「%(ARCHVERD)s」；卡死现场与超时探针（那条副本 etimes %(HANGS)s s＝**%(HMS)s**，探针判读行「%(PROVERD)s」）一并入库。
"""

OVER_T = """- **%(TIME)s（分片 %(SLICE)s 提交腿的写盘前登记；生产树 src+tests 改动 %(NCODE)s、GPU 全程未碰）**（开发日志 §%(SECREF)s「分片 %(SLICE)s＝提交腿：分片 %(PREVSLICE)s 的归档半腿整条链入库」）：
  - **入库面**：替上一片入库整条链——`%(PREVLOG)s` **%(R29B)s** B／**%(R29L)s** 行（`BODY_OFFSET=%(OFF29)s`，判读行「%(VERD29)s」、闭合式「%(CLOSURE29)s」）＋表头控件面（stdout **%(CTLB)s** B，md5 前 12 `%(CTLM)s`＝扰动 **%(CTLNCASE)s** 条、造不出红路 **%(CTLNUNC)s** 条）＋归档自查件 **%(ARCHPYB)s** B（md5 前 12 `%(ARCHPYM)s`；声称 **%(ARCHCLAIMS)s** 条全命中、正对照 **%(ARCHNCTRL)s** 条能红）＋卡死现场与超时探针（那条副本 etimes **%(HANGS)s** s＝**%(HMS)s**）。落笔现数＝未跟踪 **%(UNT)s**＋修改 **%(MOD)s**＝**%(TOT)s** 处，分组＝%(BKT)s。远端 main 仍为 `%(HEAD7)s`（父 `%(PARENT7)s`，提交时刻 %(CDATE)s）。
  - **两条口径级变化**：① **%(LESSON)s**——上一轮自查件的锚点扰动**按构造红不了**（另外四面是同源摘要、独有声称现数 0；承载＝开发日志 %(DEVCOV)s），且计数审计里 `sum(各面独有)＝总数` 这条未造未证的等式当场把自己的运行判红 ⇒ 对照要同时能红能绿，判据里不许有未测等式。② **`LC_ALL=C` 与 SRC 指纹耦合**——同一条 `sort -z` 口径在默认 locale＝`%(FINGER_DEF)s`（＝分片 26／27／28／%(PREVSLICE)s 四份表头的值），钉 `LC_ALL=C` 后＝`%(FINGER_C)s` ⇒ **钉 locale 的那条命令不许同时算指纹**，两个读数一并留档。
  - **仪器与排期**：尺子＝md5 前 12 `%(FAM)s` 一族现数 **%(RULE_N)s** 份（同一条 glob 形状 **%(RULE_GLOB)s** 份，差额有名字＝%(STRAY)s），本片抄第 **%(RULE_NEW)s** 份；表头控件面每条扰动复跑一次联网尺子 ⇒ 约 **%(CTLNCASE)s** 次 GitHub 未认证 API 调用、配额 60/小时，两面分开排在复位之后；继承常量＝子运行上限 %(SUBTO)s s／环境故障累计 %(ENVSTOP)s 条即停（超时既不判红也不判过）。排期与红线不变（原文见 §%(PREVSECREF)s 的钉住子条与计划文档 §W 补记 4）：**#30 试片尺寸裁决在前**、`--stage full` 暂停、A0 断言一字不动、#27 不换指标、无新增 skip/xfail、性能数字只在 GPU 窗口测。GPU 现取（落笔）：卡上 compute 进程 **%(GPU)s** 个、我的解释器所留 **%(NVME)s/%(GPU)s**。本片落地读数此刻尚不存在 ⇒ 提交后现取并交**分片 %(NEXTSLICE)s** 自查。
"""

PLAN_T = """
### §W 补记 5（%(TIME)s 落笔，`date '+%%F %%T %%z'` 现取）＝分片 %(SLICE)s 提交腿的**写盘前登记**：替分片 %(PREVSLICE)s 入库整条归档半腿链，并把两条口径级事实折进流程

- **外部事实只有一条**：远端 main 已在 `%(HEAD7)s`（`git ls-remote --heads origin main` 与本地 HEAD 同名，落笔时现验）；本片＝**纯归档腿**，生产树 `src`＋`tests` 改动 **%(NCODE)s** 件。入库面＝未跟踪 **%(UNT)s**＋修改 **%(MOD)s**＝**%(TOT)s** 处（`git status --porcelain` 现数；分组＝%(BKT)s）；`%(PREVLOG)s` 的判读行「%(VERD29)s」与闭合式「%(CLOSURE29)s」由档案正则现取。**本片自己的落地证明要到提交之后才存在**，本补记只登记计划与写盘前读数。
- **两条口径级变化（不是判据缺陷，也不是偏好）**：① **%(LESSON)s**——上一轮锚点扰动按构造不可能红（另外四面是同源摘要、独有声称现数 0；承载＝开发日志 %(DEVCOV)s，其余四面独有 0），且计数审计里新造的等式 `sum(各面独有)＝总数` 当场把自己的运行判红 ⇒ 一条对照要同时能红能绿，判据里不许出现未测过的等式。② **`LC_ALL=C` 与 SRC 指纹耦合**——指纹命令里的 `sort -z` 吃 collation：默认 locale＝`%(FINGER_DEF)s`（＝四份回执表头值），`LC_ALL=C`＝`%(FINGER_C)s` ⇒ 钉 locale 的命令不许算指纹，两个读数一并入档。
- **继承的仪器常量与代价登记**：表头控件面上轮新加的两道防护本轮继承（子运行上限 %(SUBTO)s s、环境故障累计 %(ENVSTOP)s 条即停；超时**既不判红也不判过**、单独计数），其现场件（那条副本 etimes **%(HANGS)s** s＝**%(HMS)s**）与超时探针判读行「%(PROVERD)s」随本片入库。两面原始件在档：表头控件面 stdout **%(CTLB)s** B（md5 前 12 `%(CTLM)s`／stderr **%(CTLERRB)s** B）＝扰动 %(CTLNCASE)s 条、造不出红路 %(CTLNUNC)s 条 ⇒「%(CTLVERD)s」；自查件 stdout **%(ARCHOUTB)s** B／stderr **%(ARCHERRB)s** B＝声称 %(ARCHCLAIMS)s 条全命中、正对照 %(ARCHNCTRL)s 条 ⇒「%(ARCHVERD)s」。⚠ 表头控件面每条扰动复跑联网尺子 ⇒ 约 %(CTLNCASE)s 次 GitHub 未认证 API 调用、配额 60/小时，两面分开排在复位之后（这是可行性代价登记，不是性能数字）。
- **排期与红线（不变，且依赖仍未松动）**：**#30 试片尺寸裁决在前**（G9 触边闸与「只有 N=1 可评分」的原文见 §%(PREVSECREF)s／§26.30 与 `am_s7_verify28.log`，本轮未重跑 ⇒ 不复述数字）→ raw 臂 N≥2 ＋ `--stage full --measured-per-track` → **#27 腿②（缺省 tuned→hk）＋#33 的位点系数同批，且只对 A3 的 1→18 外部趋势打分** → #40→#39；#41；#37／#38；#29 等裁决；#51 已量化但**不关单**。红线：A0 断言一字不放宽、#27 不换指标、无新增 skip/xfail、性能数字只在 GPU 窗口测（本轮全程未碰 GPU ⇒ 无任何吞吐论断）。GPU 现取（落笔）：卡上 compute 进程 **%(GPU)s** 个、我的解释器所留 **%(NVME)s/%(GPU)s**（只作断言、不写进可复跑件）。本片回执链与本轮控件／自查件全在工作树 ⇒ 交**分片 %(NEXTSLICE)s**。

"""

DAILY_T = """
## %(TIME)s：分片 %(SLICE)s 提交腿＝替分片 %(PREVSLICE)s 入库整条归档半腿链（未跟踪 %(UNT)s＋修改 %(MOD)s＝%(TOT)s 处）；一条口径级新学＝`LC_ALL=C` 会换掉 SRC 指纹

- **一句话**：远端 main ＝ `%(HEAD7)s`（父 `%(PARENT7)s`，提交时刻 %(CDATE)s，`git log -1 --format=%%cd` 现取），HEAD 主题＝「%(SUBJ)s…」（片号由此现取 ⇒ 本片＝%(SLICE)s）；本节写的是**提交腿落笔前**的登记，落地读数此刻还不存在。
- **入库面（分组由文件名模式现数，和＝%(BKTSUM)s 与 `grep -c '^??'` 相符）**：%(BKT)s；另加四张已跟踪档案面（%(MODTXT)s）＝未跟踪 **%(UNT)s**＋修改 **%(MOD)s**＝**%(TOT)s** 处。⚠ 这些数只描述落笔那一刻，分片 %(NEXTSLICE)s 必须重新现取。
- **两条本轮口径**：① **%(LESSON)s**——上一轮自查件的锚点扰动**按构造不可能红**（另外四面是同源摘要、独有声称现数 0；承载＝开发日志 %(DEVCOV)s），且计数审计里 `sum(各面独有)＝总数` 这条未造未证的等式当场把自己的运行判红 ⇒ 对照要同时能红能绿，判据里不许有未测等式。② **`LC_ALL=C` 与 SRC 指纹耦合**——同一条指纹命令默认 locale＝`%(FINGER_DEF)s`（＝分片 26／27／28／%(PREVSLICE)s 四份表头的值），钉 `LC_ALL=C` 后＝`%(FINGER_C)s`；差异只来自 `sort -z` 的 collation。本轮先把它读成"指纹漂移"，追到才是环境耦合 ⇒ 钉 locale 的命令不许算指纹。
- **上一片的件在档可查（引用即从档案现取，不转述）**：`%(PREVLOG)s` %(R29B)s B／%(R29L)s 行（`BODY_OFFSET=%(OFF29)s`）＝「%(VERD29)s」、闭合式「%(CLOSURE29)s」；表头控件面 stdout %(CTLB)s B（md5 %(CTLM)s／stderr %(CTLERRB)s B）＝扰动 %(CTLNCASE)s 条、造不出红路 %(CTLNUNC)s 条 ⇒「%(CTLVERD)s」；归档自查件 %(ARCHPYB)s B（md5 %(ARCHPYM)s）stdout %(ARCHOUTB)s B／stderr %(ARCHERRB)s B＝声称 %(ARCHCLAIMS)s 条全命中、正对照 %(ARCHNCTRL)s 条 ⇒「%(ARCHVERD)s」；卡死现场（那条副本 etimes %(HANGS)s s＝**%(HMS)s**）与超时探针「%(PROVERD)s」一并入库。
- **钉住与红线**：每条计算命令逐条内联 `CUDA_VISIBLE_DEVICES=""`（含一行式）；**不产生吞吐／加速／预算数字**；生产树改动 **%(NCODE)s** 件；尺子＝md5 前 12 `%(FAM)s` 一族现数 %(RULE_N)s 份（同一 glob 形状 %(RULE_GLOB)s 份，差额＝%(STRAY)s），本片抄第 %(RULE_NEW)s 份；GPU 现取 卡上 **%(GPU)s** 个、我的 **%(NVME)s/%(GPU)s**。排期与红线不变（原文见 §%(PREVSECREF)s 与计划文档 §W 补记 4）。
"""

WBIDX_T = """- **%(TIME)s（分片 %(SLICE)s 提交腿的写盘前登记；生产树 0 改动、GPU 全程未碰）**：
  本片替上一片入库整条链：`%(PREVLOG)s`（%(R29B)s B／%(R29L)s 行、`BODY_OFFSET=%(OFF29)s`，判读行「%(VERD29)s」、闭合式「%(CLOSURE29)s」）＋表头控件面（stdout %(CTLB)s B＝扰动 %(CTLNCASE)s／造不出红路 %(CTLNUNC)s）＋归档自查件（%(ARCHPYB)s B，声称 %(ARCHCLAIMS)s 条全命中、正对照 %(ARCHNCTRL)s 条）＋卡死现场与超时探针（etimes %(HANGS)s s＝**%(HMS)s**）。
  入库面＝未跟踪 **%(UNT)s**＋修改 **%(MOD)s**＝**%(TOT)s** 处，分组＝%(BKT)s；`src`＋`tests` 改动 **%(NCODE)s** 件 ⇒ 纯归档腿。远端 main `%(HEAD7)s`（父 `%(PARENT7)s`）。
  **两条口径级新学**：① **%(LESSON)s**——锚点扰动要先量每面独有承载（同源摘要面独有 0 ⇒ 那条对照按构造红不了），计数审计里不许出现未测过的等式；
  ② **`LC_ALL=C` 与 SRC 指纹耦合**——同一条 `sort -z` 口径默认 locale＝`%(FINGER_DEF)s`（＝前四片表头值）、钉 `LC_ALL=C`＝`%(FINGER_C)s` ⇒ 钉 locale 的命令不许算指纹。
  方法账沿用：尺子按 md5 分组数＝%(RULE_N)s 份（glob 形状 %(RULE_GLOB)s 份，差额＝%(STRAY)s），本片抄第 %(RULE_NEW)s 份；表头控件面每条扰动复跑联网尺子 ⇒ 约 %(CTLNCASE)s 次 API 调用、配额 60/小时，两面分开排在复位之后。
  排期不变（**#30 试片尺寸裁决在前**、`--stage full` 暂停，原文见 §%(PREVSECREF)s／§W 补记 4）；红线：A0 断言一字不动、无新增 skip/xfail、性能数字只在 GPU 窗口测。本片落地读数此刻不存在 ⇒ 提交后现取并交**分片 %(NEXTSLICE)s**。
"""

FACES = {"DEV": ("dev", DEV_T), "OVER": ("over", OVER_T), "PLAN": ("plan", PLAN_T),
         "DAILY": ("daily", DAILY_T), "WBIDX": ("wbidx", WBIDX_T)}


def must_reject(label, tmpl):
    """named() 的负对照：坏模板必须被拦下，否则那道裸百分号门是恒过的装饰。"""
    try:
        named(label, tmpl)
    except AssertionError:
        return True
    raise SystemExit("正对照失败：%s 这类模板本该被 named() 拦下却过了 ⇒ 那道门是装饰" % label)


assert "x % y" == named("自检/好", "x %% y"), "具名替换把双写百分号渲染错了 ⇒ 渲染结果不可信"
assert "x " + CTX["SLICE"] + " y" == named("自检/好", "x %(SLICE)s y"), "具名槽没被替换"
must_reject("自检/裸百分号", "git log --format=%cd 现取")
must_reject("自检/全角槽", "槽＝%（SLICE）s")
print("== named() 自检：好模板渲染逐字节正确、裸百分号与全角槽各被拦下 -> PASS")

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
    """节号引用必须写成开发日志自己的形制 `§26.NN`；裸 `§32` 读者找不到那一节。

    计划文档的字母节（§W／§L／§R…）与 `§` 后不带数字的写法不在拦阻范围，所以这条门只针对数字形制。
    """
    return re.findall(r"§(?!26\.)\d+", v)


for k, v in texts.items():
    bad = secrefs(v)
    assert not bad, "%s 有裸节号 %s ⇒ 引用要写成 §26.NN，否则读者按 §32 搜不到标题" % (k, bad[:4])
# 正对照：把形制改坏，检测器必须抓到（抓不到＝这条门是装饰）
_bad = secrefs(texts["DEV"].replace("§26.%d" % PREVSEC, "§%d" % PREVSEC, 1))
assert len(_bad) == 1, "裸节号检测器在写坏的样本上抓到 %d 处（须 1）⇒ 它没有分辨力" % len(_bad)
print("   五面节号形制全部＝§26.NN（检测器对自己的坏样本抓到 1 处）-> PASS")
_lab = [m.group(1) for m in re.finditer(r"^- \*\*(.+?)\*\*", texts["DEV"], re.M)]
assert len(_lab) == 7 and len(set(_lab)) == 7, "§%d 顶级子条＝%d 条、互异标签＝%d 条（须 7 且互异）" % (
    SECN, len(_lab), len(set(_lab)))

NEEDLES = (("引用上一片回执名", re.escape(PREVLOG)),
           ("钉住口径登记", r"当前「[^」]+」"),
           ("LESSON 锚句", re.escape(LESSON)))
BAIT = {"引用上一片回执名": "`%s`" % PREVLOG, "钉住口径登记": "当前「诱饵口径」", LESSON: LESSON}
for what, pat in NEEDLES:
    n = len(re.findall(pat, texts["DEV"], re.M))
    assert n == 1, "%s 在 §%d 里命中 %d 处（须恰好 1）⇒ 表头生成器的节内唯一性会拒绝归档" % (what, SECN, n)
    bait = ("\n- **诱饵**：见 %s。\n" % (BAIT[what] if what != "LESSON 锚句" else "**%s**诱饵" % LESSON))
    assert len(re.findall(pat, texts["DEV"] + bait, re.M)) == 2, \
        "%s 的正对照空转（加一条复述后仍不是 2 处）⇒ 这条唯一性断言是恒真式" % what
print("   §%d 三条节内唯一性各恰好 1 处，且各配一条「再复述就变成 2 处」的正对照 -> PASS" % SECN)

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
    ("`BODY_OFFSET=", CTX["OFF29"]),
    ("之后）＝`", CTX["FINGER_DEF"]),
    ("交**分片 ", CTX["NEXTSLICE"]),
])
expect_near("OVER", texts["OVER"], [("未跟踪 **", CTX["UNT"]), ("md5 前 12 `", CTX["FAM"])])
expect_near("PLAN", texts["PLAN"], [("远端 main 已在 `", CTX["HEAD7"]), ("etimes **", CTX["HANGS"])])
expect_near("DAILY", texts["DAILY"], [("远端 main ＝ `", CTX["HEAD7"]), ("闭合式「", CTX["CLOSURE29"])])
expect_near("WBIDX", texts["WBIDX"], [("入库面＝未跟踪 **", CTX["UNT"]), ("远端 main `", CTX["HEAD7"])])
print("   邻接断言：五面各 2–5 对「标签＋值」在渲染结果里原文命中 -> PASS")

os.makedirs(OUT_DIR, exist_ok=True)
for k, v in texts.items():
    open(os.path.join(OUT_DIR, k + ".md"), "w", encoding="utf-8").write(v)
open(os.path.join(OUT_DIR, "ctx.txt"), "w", encoding="utf-8").write(
    "\n".join("%s=%s" % (kk, vv) for kk, vv in sorted(CTX.items())) + "\n")
print("== 渲染完成：%s/{DEV,OVER,PLAN,DAILY,WBIDX}.md ＋ ctx.txt" % OUT_DIR)


# ============================ 写盘腿（带守卫、一次性） ============================
def running_faces():
    """正在跑的仪器：按 `/proc/PID/cmdline` 的**整词**比对 basename，不按命令行子串猜。"""
    hits = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            argv = open("/proc/%s/cmdline" % d, "rb").read().split(b"\0")
        except OSError:
            continue
        toks = {os.path.basename(a.decode("utf-8", "replace")) for a in argv if a}
        if toks & TOKENS and "s30_fill.py" not in toks:
            hits.append((d, sorted(toks & TOKENS)))
    return hits


if "--write" in sys.argv:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "写盘腿没钉 CVD 空串 ⇒ 与 ENV 口径不符"
    faces = running_faces()
    assert not faces, "有仪器正在跑 %s ⇒ 按口径不许动它可读的档案面" % faces
    for path in (DEVLOG, OVERVIEW, PLAN, DAILY, WBIDX):
        assert open(path, "rb").read()[-1:] == b"\n", "%s 末字节不是换行 ⇒ 拒绝追加" % path
    pre = [l for l in sh("git", "-c", "core.quotePath=false", "status", "--porcelain").splitlines() if l.strip()]
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
    # 落盘的整节再数一次三条唯一性（片段数与整节数必须相等）
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
    print("   磁盘上的 §%d 三条唯一性各恰好 1 处（＝片段预检值）-> PASS" % SECN)
    post = [l for l in sh("git", "-c", "core.quotePath=false", "status", "--porcelain").splitlines() if l.strip()]
    nq = len([l for l in post if l.startswith("??")])
    nm = len([l for l in post if not l.startswith("??")])
    assert nq == NU + 0, "未跟踪件数从 %d 变成 %d ⇒ 写盘动到了不该动的地方" % (NU, nq)
    assert nm == NM, "修改面数从 %d 变成 %d ⇒ 追加写到了第五张面上" % (NM, nm)
    print("   闭合式复算：未跟踪 %d＋修改 %d＝%d 处（写盘前后同数，因五张面此前已在修改／未跟踪态）-> PASS"
          % (nq, nm, nq + nm))
    print("WRITER_VERDICT = FIVE FACES LANDED, UNIQUENESS RECOUNTED ON DISK")
else:
    print("（预检模式：未写盘。加 --write。）")
print("FILLER_VERDICT = READINGS DERIVED, STRUCTURE GATES PASS")
