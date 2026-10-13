#!/usr/bin/env python3
r"""分片 30 的**落地填值器**（＋带守卫的写盘腿）：提交之后那五张档案面的落地段，每个读数都由命令现取。

与本片写盘前那件 `am_s30_docs.py` 的分工＝时间轴：那件在**提交之前**落盘，所以它只能登记计划、并明写
「落地读数此刻还不存在」；本件在**提交之后**落盘，它存在的唯一理由是把那句话说过的空格填上。因此本件的每条数都取自
① `git` 现算（HEAD／父／name-status／shortstat／numstat），② 已入库档案面的正则现取（回执正文、三面控件面的 stdout），
③ 盘上字节（大小、md5、`BODY_OFFSET` 定点复解、正文区与原始件逐字节相等）。没有一条是我手敲的。

四条口径（前三条继承本片写盘前那件，第四条是本片新加的真事故）：
① **具名替换＋邻接断言**——位置式槽位在「槽数相等＋顺序单调」两道检查同时绿时仍能把散文排错位。
② **渲染结构门**——按行配平全角 `（）`、剥内联码后配平 `**`、目标末字节是换行、插入锚点整份文件恰好 1 处。
③ **节内唯一性＋正对照**——分片 31 的表头生成器要在本节里现取三条唯一出处（引用本片回执名／带引号的钉住口径
   ／本片的 LESSON 锚句）；每条都配一根「再复述一次就 +1」的 bait，也预检**落笔前节内现数**，使合并后恰好 1。
④ **易失读数只进散文**——卡上 compute 进程总数、未入库件数、落笔 date 进档案面（那里它们带日期、可被下一条覆盖），
   但**卡上那个总数不印进本件的 stdout**（本件 stdout 会被归档、会被分片 31 引用，而它随他人作业变）。

默认只渲染到 `/tmp/out30_land/`；`--write` 才落盘，且落盘前后各数一次闭合式。
"""
import ast
import hashlib
import os
import re
import subprocess
import sys

REPO = "/home/shy/桌面/2026年开发-显式冲击有限元-物质点法-sph-fvm+am/AM-Qoder"
EVID_REL = "docs/evidence/2026-10-12"
EVID = os.path.join(REPO, EVID_REL)
DEVLOG = os.path.join(REPO, "docs/开发日志.md")
OVERVIEW = os.path.join(REPO, "docs/项目开发总览.md")
PLAN = os.path.join(REPO, "docs/项目评估与下一步计划_2026-10-06.md")
DAILY = os.path.join(REPO, ".workbuddy/memory/2026-10-12.md")
WBIDX = os.path.join(REPO, ".workbuddy/memory/MEMORY.md")
FACES5 = [DEVLOG, OVERVIEW, PLAN, DAILY, WBIDX]
ANCHOR = "## 7. 文档索引"
OUT_DIR = "/tmp/out30_land"
SELF_BASE = os.path.basename(os.path.abspath(__file__))
# 写盘守卫：这些名字**正在跑**就不许动档案（按 /proc/PID/cmdline 的**整词**比对，不按命令行子串猜）
TOKENS = {"am_s30_hdr_controls.py", "am_s30_hdr_after.py", "am_s30_hdr_after_controls.py",
          "am_s30_hdr.py", "am_s30_docs.py", "am_s7_verify30.py", "am_s30_archive_check.py",
          "am_s30_msg.py", "am_s30_msg_controls.py"}
COPY_PREFIX = "zctl_s30_"            # 控件面的扰动副本在本目录里跑，名字是动态的 ⇒ 按前缀认
SPNAME = "_raw_s30_hdr_controls.attempt1-selfaudit-inverted"
SP2NAME = "_raw_s30_hdr_controls.attempt2-notb-unsatisfiable"
LESSON_PREV = "对照的可满足性"        # 上一片的 LESSON 锚句：本节已有恰好 1 条 ⇒ 本片段必须贡献 0
LESSON_NEW = "自审的方向性"           # 本片的新学锚句：整份日志里必须恰好从 0 变 1
GPU_MARK = ("AM-Qoder", "warpblast", "/tmp/amvenv", "docs/evidence")
CTX = {}


def sh(*args):
    """**默认环境**跑（不钉 LC_ALL）——指纹口径只在默认 locale 下与前四片可比。"""
    return subprocess.run(args, cwd=REPO, capture_output=True, text=True).stdout


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def one(needle, text, what):
    hits = re.findall(needle, text, re.M)
    assert len(hits) == 1, "%s 命中 %d 处（须恰好 1）：%s" % (what, len(hits), hits[:3])
    return hits[0]


def last(needle, text, what):
    hits = re.findall(needle, text, re.M)
    assert hits, "%s 现数为 0 ⇒ 该件根本没走到这一步" % what
    return hits[-1]


def named(label, text):
    """具名替换＋残留检查（`%（` 是全角括号打错位的具名槽，静默不替换）。"""
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


# ============================ 读数 1：git 现算 ============================
HEAD = sh("git", "rev-parse", "HEAD").strip()
HEAD7 = sh("git", "rev-parse", "--short", "HEAD").strip()
PAR40 = sh("git", "rev-parse", "HEAD^").strip()
PAR7 = sh("git", "rev-parse", "--short", "HEAD^").strip()
SUBJ = sh("git", "log", "-1", "--format=%s").strip()
CDATE = sh("git", "log", "-1", "--format=%cd", "--date=iso").strip()
TIME = sh("date", "+%F %T %z").strip()
NCODE = len([l for l in sh("git", "-c", "core.quotePath=false", "status", "--porcelain",
                           "--", "src", "tests").splitlines() if l.strip()])
assert NCODE == 0, "生产树 src＋tests 有 %d 处改动 ⇒ 本片不是纯归档腿，那句散文要先改" % NCODE
# 落地腿的片号＝HEAD 自己标的那个号。写盘前那件用 `+1`（它落盘时 HEAD 还是上一片）；
# 本件在提交之后跑，照抄 `+1` 会去要一份从没生成过的 am_s7_verify31.log（纸面预检的第一次真红灯）。
SLICE = int(one(r"S7 分片 (\d+)", SUBJ, "HEAD 提交说明里的片号"))
PREVSLICE, NEXTSLICE = SLICE - 1, SLICE + 1
_nums = sorted({int(x) for x in re.findall(r"分片 (\d+)", SUBJ)})
assert _nums == [PREVSLICE, SLICE], (
    "HEAD 提交说明里出现过的片号集合＝%s，须恰好＝[%d, %d]（本片／它入库的那一片）⇒ 片号方向没被提交说明双向钉住" % (
        _nums, PREVSLICE, SLICE))
for _f in ("am_s7_verify%d.log" % SLICE, "_raw_verify%d.out" % SLICE, "am_s%d_hdr.py" % SLICE):
    assert os.path.exists(os.path.join(EVID, _f)), (
        "按片号 %d 拼出的 `%s` 不在盘上 ⇒ 片号指错了对象（回执链／生成器缺一）" % (SLICE, _f))
# 写盘守卫的三类常量都是手写的名字：把它们逐一钉回现取的片号／提交日期，拼错或过期就会当场红。
assert all(str(SLICE) in t for t in TOKENS), (
    "写盘守卫的整词表里有名字不含片号 %d：%s ⇒ 那张表在守别的片" % (SLICE, sorted(t for t in TOKENS if str(SLICE) not in t)))
assert COPY_PREFIX == "zctl_s%d_" % SLICE, "扰动副本前缀＝%s，与片号 %d 拼出的名字不符" % (COPY_PREFIX, SLICE)
assert SPNAME.startswith("_raw_s%d_" % SLICE), "自审反转现场件名前缀与片号 %d 不符：%s" % (SLICE, SPNAME)
assert EVID_REL == "docs/evidence/%s" % CDATE[0:10], (
    "证据目录 %s 与 HEAD 提交日期 %s 不同日 ⇒ 本件锚的是上一轮的目录（跨日期目录要从命令现取）" % (EVID_REL, CDATE[0:10]))
_absent = sorted(t for t in TOKENS if not os.path.exists(os.path.join(EVID, t)))
assert _absent == ["am_s%d_archive_check.py" % SLICE], (
    "整词表里缺席盘上的名字＝%s，须恰好是那件**还没落盘**的归档自查腿 ⇒ 其余名字拼错就会让守卫空转" % (_absent,))
print("   守卫常量回填验：整词表 %d 个名字都带片号 %d、%d 个在盘上（缺席＝归档自查腿 %d 件）｜前缀／现场件名／证据目录同日 -> PASS"
      % (len(TOKENS), SLICE, len(TOKENS) - len(_absent), len(_absent)))
assert len(HEAD) == 40 and HEAD != PAR40, "HEAD／父哈希形状不对"

_ns = [l for l in sh("git", "show", "--name-status", "--format=", "HEAD").splitlines() if l.strip()]
_rows = [l.split("\t") for l in _ns if "\t" in l]
assert len(_rows) == len(_ns), "name-status 里有不含制表符的行"
KINDS = [r[0] for r in _rows]
CPATHS = [r[-1] for r in _rows]          # 路径取**行尾**：状态字母不是路径（上一片的真事故形状）
assert all(len(r) == 2 for r in _rows), "name-status 有非两列的行"
assert set(KINDS) <= {"A", "M"}, "HEAD 清单里有 A／M 之外的状态：%s" % sorted(set(KINDS))
NAC, NMC = KINDS.count("A"), KINDS.count("M")
NTOTC = len(CPATHS)
assert len(set(CPATHS)) == NTOTC, "HEAD 清单里有重名路径"
NSR = [l for l in sh("git", "show", "--numstat", "--format=", "HEAD").splitlines() if l.strip()]
assert len(NSR) == NTOTC, "numstat 行数 %d 与 name-status %d 不等 ⇒ 两个口径在数不同的东西" % (
    len(NSR), NTOTC)
assert all("\t" in l and l.split("\t")[0] != "-" for l in NSR), \
    "numstat 里有 `-`（二进制／rename）⇒ 本件的逐行求和口径不适用，落地段那句要改"
NS_INS = sum(int(l.split("\t")[0]) for l in NSR)
NS_DEL = sum(int(l.split("\t")[1]) for l in NSR)
SS_RAW = sh("git", "diff-tree", "--no-commit-id", "--shortstat", "-r", "HEAD").strip()
# shortstat 的 "1 deletion(-)" 是单数形制：数量词一律用 `files?`／`insertions?`，否则复数形状会把真值数成 0
SS_FILES = int(one(r"(\d+) files? changed", SS_RAW, "shortstat 的文件数"))
SS_INS = int(one(r"(\d[\d,]*) insertions?", SS_RAW, "shortstat 的插入数"))
assert "deletion" not in SS_RAW, "shortstat 里有删除段 ⇒ 零删除那句要改：%s" % SS_RAW
assert (SS_FILES, SS_INS) == (NTOTC, NS_INS), (
    "shortstat (%d, %d) 与 numstat 现算 (%d, %d) 不等 ⇒ 两把尺子对不上" % (
        SS_FILES, SS_INS, NTOTC, NS_INS))
assert NS_DEL == 0, "numstat 现算删除 %d（须 0）" % NS_DEL
assert ".workbuddy/memory/2026-%s.md" % TIME[5:10] in CPATHS, \
    "当日纪实不在 HEAD 清单里 ⇒ 上一片写盘的那张面没随提交入库"
print("== 现取：HEAD＝%s（父 %s｜提交时刻 %s）｜入库 A %d＋M %d＝%d 件｜numstat %d/%d 与 shortstat 闭合｜src+tests 改动 %d｜片号＝%d"
      % (HEAD7, PAR7, CDATE, NAC, NMC, NTOTC, NS_INS, NS_DEL, NCODE, SLICE))

# ============================ 读数 2：回执正文与盘上字节 ============================
R30LOG = "am_s7_verify%d.log" % SLICE
r30 = open(os.path.join(EVID, R30LOG), encoding="utf-8").read()
R30B, R30L = os.path.getsize(os.path.join(EVID, R30LOG)), len(r30.splitlines())
RAWM = "_raw_verify%d.out" % SLICE
ERRM = "_raw_verify%d.err" % SLICE
rawm = open(os.path.join(EVID, RAWM), encoding="utf-8").read()
ERRMB = os.path.getsize(os.path.join(EVID, ERRM))
assert ERRMB == 0, "核验件的 stderr 不是 0 字节（%d）⇒ 有警告被吞" % ERRMB
GEN = os.path.join(EVID, "am_s%d_hdr.py" % SLICE)
OFF30 = int(one(r"^BODY_OFFSET=(\d+)$", r30, "本片回执的 BODY_OFFSET"))
_marker = "BODY_OFFSET=%d\n" % OFF30
assert r30.count(_marker) == 1, "BODY_OFFSET 那行不是恰好 1 条"
_hp, _body = r30.split(_marker, 1)
assert len((_hp + _marker).encode("utf-8")) == OFF30, "BODY_OFFSET 与实际表头字节数不等（定点没收敛）"
BODYB = R30B - OFF30
assert BODYB == os.path.getsize(os.path.join(EVID, RAWM)) == len(rawm.encode("utf-8")), (
    "回执正文区字节数 %d 与原始件 %d 不等" % (BODYB, os.path.getsize(os.path.join(EVID, RAWM))))
assert _body == rawm, "回执正文区与原始件不是逐字节相同 ⇒ 判读行与正文不同源"
REC_HEAD_M = re.search(r"head=([0-9a-f]{40})", r30)
assert REC_HEAD_M, "回执表头里取不到 head= 那枚哈希"
assert REC_HEAD_M.group(1) == HEAD, "回执记的 HEAD 与盘上 HEAD 不同 ⇒ 落地段说的不是当前这棵树"
FINGER_REC = one(r"^SRC指纹\s*:\s*([0-9a-f]{32})", r30, "回执表头的 SRC 指纹")
FINGER_NOW = sh("bash", "-c", "find src tests -name '*.py' -not -path '*__pycache__*' -print0"
                 " | sort -z | xargs -0 md5sum | md5sum").split()[0]
assert FINGER_REC == FINGER_NOW, ("回执里的 SRC 指纹 %s 与盘上现算 %s 不等 ⇒ 表头与**被测树**不是同一棵"
                                  "（默认 locale 口径，与回执同）" % (FINGER_REC[:12], FINGER_NOW[:12]))
TRUNC = one(r"② truncated=(\w+)", rawm, "② 的 truncated")
TB, BB, BT = [int(x) for x in one(r"tree 条目 (\d+) = blob (\d+) \+ tree (\d+)", rawm, "② 的 tree 分解")]
LB = int(one(r"② 本地 blob = (\d+)", rawm, "② 的本地 blob"))
PB = int(one(r"② 本地 blob = \d+，父提交 = (\d+)", rawm, "② 的父提交 blob"))
RCNT = int(one(r"② 本片文件 (\d+) 个", rawm, "② 的本片文件数"))
RNEW = int(one(r"② 本片文件 \d+ 个 = 新增 (\d+)", rawm, "② 的新增数"))
RMODC = int(one(r"② 本片文件 \d+ 个 = 新增 \d+ \+ 修改 (\d+)", rawm, "② 的修改数"))
CLOS30 = one(r"^② 闭合式 .*$", rawm, "② 的闭合式整行")
DIFF3 = one(r"^③ 路径差集：.*$", rawm, "③ 的差集整行")
SHA4 = one(r"^④ 全库共有 .*$", rawm, "④ 的全库普查整行")
PER4 = one(r"^④ 本片 .*MATCH \d+/\d+$", rawm, "④ 的本片单列复核整行")
V30 = one(r"^LANDING_VERDICT = .*$", rawm, "核验件判读行")
CA = one(r"^⑤ 正对照 A.*$", rawm, "正对照 A")
CB = one(r"^⑤ 正对照 B.*$", rawm, "正对照 B")
CC = one(r"^⑤ 正对照 C.*$", rawm, "正对照 C")
CD = one(r"^⑤ 正对照 D.*$", rawm, "正对照 D")
assert BB == LB, "回执正文里远端 blob %d 与本地 blob %d 不等" % (BB, LB)
assert (RCNT, RNEW, RMODC) == (NTOTC, NAC, NMC), (
    "回执正文的本片文件数 %s 与 git 现算 总／A／M (%d,%d,%d) 不等" % ((RCNT, RNEW, RMODC), NTOTC, NAC, NMC))
assert TB == BB + BT, "tree 条目 %d ≠ blob %d ＋ tree %d" % (TB, BB, BT)
CLOSEX = BB - (PB + NAC)
assert CLOSEX == 0, "闭合式复算差 %d ≠ 0（远端 %d／父 %d／新增 A %d）" % (CLOSEX, BB, PB, NAC)
print("   回执现取：%s＝%d B／%d 行＝表头 %d＋正文 %d（逐字节 == 原始件）｜闭合差＝%d｜SRC 指纹与盘上现算同值 -> PASS"
      % (R30LOG, R30B, R30L, OFF30, BODYB, CLOSEX))

# 顶格字段数：一面从生成器的 FIELDS 声明（ast 现取，不执行它），一面从写盘后只读仪器报出的档案计数
_fl = []
for _n in ast.walk(ast.parse(open(GEN, encoding="utf-8").read())):
    if isinstance(_n, ast.Assign) and any(getattr(t, "id", None) == "FIELDS" for t in _n.targets):
        _v = ast.literal_eval(_n.value)
        assert isinstance(_v, list) and all(isinstance(x, str) for x in _v), "FIELDS 不是字符串列表"
        _fl.append(_v)
assert len(_fl) == 1, "生成器里 FIELDS 赋值有 %d 处 ⇒ 声明出处不唯一" % len(_fl)
FLDN = len(_fl[0])
after_txt = open(os.path.join(EVID, "_raw_s%d_hdr_after.out" % SLICE), encoding="utf-8").read()
AF_LAB, AF_DEC = [int(x) for x in one(r"^顶格字段行数＝ (\d+) / (\d+)", after_txt, "只读仪器的字段计数")]
assert (AF_LAB, AF_DEC) == (FLDN, FLDN), "档案字段 %d／声明 %d／本件现取 FLDN=%d 不是三处同值" % (
    AF_LAB, AF_DEC, FLDN)
CMPT = one(r"^cmp -i 三联（本件自己起进程跑的，不是抄生成器的）＝ (\d+ / \d+ / \d+)",
           after_txt, "只读仪器的 cmp 三联")
assert CMPT == "0 / 1 / 1", "cmp 三联不是 0/1/1（本件自己跑的现算值）：%s" % CMPT
HRAV = one(r"^HDR_AFTER_VERDICT = .*$", after_txt, "只读仪器判读行")

# ============================ 读数 3：三面控件面的真跑 stdout ============================
CTL = [
    ("MSG", "提交说明面", "am_s%d_msg_controls.py" % SLICE, "_raw_s%d_msg_controls.out" % SLICE,
     "_raw_s%d_msg_controls.err" % SLICE, r"^CASES 声明＝(\d+) 条｜本件自己的锚点普查", []),
    ("HDR", "表头面", "am_s%d_hdr_controls.py" % SLICE, "_raw_s%d_hdr_controls.out" % SLICE,
     "_raw_s%d_hdr_controls.err" % SLICE, r"^红路声明＝(\d+) 条", [r"红路声明＝\d+ 条｜保持绿＝(\d+) 条"]),
    ("AFTC", "只读仪器面", "am_s%d_hdr_after_controls.py" % SLICE,
     "_raw_s%d_hdr_after_controls.out" % SLICE, "_raw_s%d_hdr_after_controls.err" % SLICE,
     r"^CASES 声明＝(\d+) 条（全部红路）", [r"^(对照条数＝.*)$"]),
]
for pfx, face, script, outf, errf, dec_pat, extras in CTL:
    op, ep, sp = os.path.join(EVID, outf), os.path.join(EVID, errf), os.path.join(EVID, script)
    for p in (sp, op, ep):
        assert os.path.exists(p), "%s 缺件：%s" % (face, p)
    txt = open(op, encoding="utf-8").read()
    CTX[pfx + "_SCRIPT"] = script
    CTX[pfx + "_OUT"] = outf
    CTX[pfx + "_OUTB"] = str(os.path.getsize(op))
    CTX[pfx + "_OUTM"] = md5(op)[:12]
    CTX[pfx + "_ERRB"] = str(os.path.getsize(ep))
    assert os.path.getsize(ep) == 0, "%s 的 stderr 不是 0 字节（有警告被吞）" % face
    CTX[pfx + "_DEC"] = one(dec_pat, txt, face + " 的声明条数")
    for pat in extras:
        key = "GRN" if "保持绿" in pat else "PAIRS"
        CTX[pfx + "_" + key] = one(pat, txt, face + " 的 " + key)
    CTX[pfx + "_UNC"] = one(r"本轮造不出红路的闸门＝(\d+) 条", txt, face + " 的造不出红路数")
    CTX[pfx + "_TO"] = one(r"超时（环境故障，不判红也不判过）＝(\d+) 条", txt, face + " 的超时数")
    assert CTX[pfx + "_TO"] == "0", "%s 有 %s 条子运行超时＝环境故障，那一批读数不可用" % (
        face, CTX[pfx + "_TO"])
    ver = last(r"^(CONTROL_VERDICT = .*)$", txt, face + " 的判读行")
    assert ver.endswith("ALL CONTROLS CAN GO RED"), "%s 的判读行不是全红声明：%s" % (face, ver)
    CTX[pfx + "_VERDICT"] = ver
    print("   %s：%s %s B（md5 %s／stderr %s）｜声明 %s｜超时 %s｜造不出红路 %s｜判读＝%s"
          % (face, outf, CTX[pfx + "_OUTB"], CTX[pfx + "_OUTM"], CTX[pfx + "_ERRB"],
             CTX[pfx + "_DEC"], CTX[pfx + "_TO"], CTX[pfx + "_UNC"], ver))

# 本面第一次真跑的**自审反转**事故：现场两件按别名留在档，判据句从 stderr 现取
SPO, SPE = os.path.join(EVID, SPNAME + ".out"), os.path.join(EVID, SPNAME + ".err")
for p in (SPO, SPE):
    assert os.path.exists(p), "自审反转现场件不在档：%s" % p
SPBO, SPBE = os.path.getsize(SPO), os.path.getsize(SPE)
SPWHY = one(r"^AssertionError: (.*)$", open(SPE, encoding="utf-8").read(), "现场件里的抛出行")
assert "抓不出不一致" in SPWHY, "现场件抛出的不是那条自审反转：%s" % SPWHY
# 崩掉那一批有多少条读数＝现场件**首行自己声明**的红路＋绿路之和（从该件现取，不从我脑子里取）
_sp_first = open(SPO, encoding="utf-8").read()
_nr = one(r"^纸面预检通过＝.*?（红路 (\d+) 条＋绿路 (\d+) 条", _sp_first, "现场件首行声明的红路／绿路条数")
NLOST = int(_nr[0]) + int(_nr[1])
assert NLOST > 0, "现场件首行的两条声明数之和＝%d ⇒ 解析取到了空值" % NLOST
print("   自审反转现场：stdout %d B／stderr %d B｜抛出行「%s」｜崩掉的批＝红路 %s＋绿路 %s＝%d 条"
      % (SPBO, SPBE, SPWHY, _nr[0], _nr[1], NLOST))

# 第二次真跑的现场＝**判据侧**的同族缺陷：E6 那条闸门当场响了，却被本面自己的合取记成未命中。
# 现场两件的字节数、未命中行数、实得抛出行全部从该件现取；抛出行的**行号**从生成器源码现取。
SP2O, SP2E = os.path.join(EVID, SP2NAME + ".out"), os.path.join(EVID, SP2NAME + ".err")
for p in (SP2O, SP2E):
    assert os.path.exists(p), "第二次真跑的现场件不在档：%s" % p
SP2BO, SP2BE = os.path.getsize(SP2O), os.path.getsize(SP2E)
_sp2 = open(SP2O, encoding="utf-8").read()
SP2MISS = len([l for l in _sp2.splitlines() if l.startswith("× ")])
assert SP2MISS == 1, "现场件里有 %d 行「× 」（须恰好 1＝E6 那一条）⇒ 多出来的未命中不是本轮登记过的那处" % SP2MISS
SP2VERDICT = one(r"^CONTROL_VERDICT = .*$", _sp2, "第二次真跑的整面判读")
assert "PROBLEM" in SP2VERDICT, "现场件判读不含 PROBLEM（它当时就是收在判据空转上）：%s" % SP2VERDICT
SP2OLD = one(r"本面自查＝(\d+) 条", _sp2, "现场件声明的本面自查条数")
SP2RAISE = one(r"｜实得抛出行：(.*)$", _sp2, "未命中那行的实得抛出行")
# 那句抛出必须**逐字符等于**生成器里那条 BODY_OFFSET 的 `SystemExit(str)` 的参数：
# 等值本身就是「CPython 对未捕获的 SystemExit(str) 只打裸消息、不加 `SystemExit:` 前缀」的证据，
# 行号也从 ast 的节点上取，不用 grep 的行号（本目录里同名串可能有注释）。
_glines = open(os.path.join(EVID, "am_s%d_hdr.py" % SLICE), encoding="utf-8").read()
_se = [n for n in ast.walk(ast.parse(_glines))
       if isinstance(n, ast.Raise) and isinstance(n.exc, ast.Call)
       and getattr(n.exc.func, "id", "") == "SystemExit"
       and n.exc.args and isinstance(n.exc.args[0], ast.Constant)
       and isinstance(n.exc.args[0].value, str)]
_bo = [n for n in _se if "BODY_OFFSET" in n.exc.args[0].value]
assert len(_bo) == 1, "生成器源码里带 BODY_OFFSET 的 `raise SystemExit(str)` 不是恰好 1 条：%d 条" % len(_bo)
SP2LINE = _bo[0].lineno
assert _bo[0].exc.args[0].value == SP2RAISE, "现场件那句与生成器那条 raise 的参数不逐字符相等 ⇒ 现场不是同一个闸门：%r／%r" % (
    SP2RAISE, _bo[0].exc.args[0].value)
assert not SP2RAISE.startswith("SystemExit:"), "现场件那句自己就带前缀 ⇒ 上面那句解释要重写"
# 修复是否真吃进源码：判定段调了纯函数，且该函数自己的判定表两头都有形状可验
_ctl_src = open(os.path.join(EVID, "am_s%d_hdr_controls.py" % SLICE), encoding="utf-8").read()
assert "ok = notb_hit(msg, rc, expect, gl)" in _ctl_src, "表头控件面的 notb 分支还在原地写合取，没改成调谓词"


def table_counts(src, name):
    """从控件面源码现取判定表的三个数（总数／须收／须拒）。

    两条形制约束都是本轮实测逼出来的：① 表在 `if` 里缩进着（不是模块级），所以只扫
    `ast.parse(src).body` 会「取不到」——要整棵树走一遍；② 表里的行是 `_E6 + "\\n"` 这种**拼接**，
    `ast.literal_eval` 当场 ValueError——所以按 ast 结构数，末位必须是布尔字面量。
    """
    hits = []
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in n.targets):
            hits.append(n)
    if len(hits) != 1:
        raise AssertionError("源码里 %s 赋值 %d 处（须恰好 1）⇒ 数出来的表不是那一张" % (name, len(hits)))
    v = hits[0].value
    if not isinstance(v, ast.List):
        raise AssertionError("%s 右边不是列表字面量而是 %s ⇒ 结构计数没有靶" % (
            name, type(v).__name__))
    acc = rej = 0
    for i, row in enumerate(v.elts, 1):
        if not isinstance(row, ast.Tuple) or not row.elts:
            raise AssertionError("%s 第 %d 行不是非空元组 ⇒ 这张表的行形制变了" % (name, i))
        last = row.elts[-1]
        if not (isinstance(last, ast.Constant) and isinstance(last.value, bool)):
            raise AssertionError("%s 第 %d 行末位不是 True/False 字面量 ⇒ 分不清须收还是须拒" % (name, i))
        if last.value:
            acc += 1
        else:
            rej += 1
    return len(v.elts), acc, rej


# 这条尺子自己的正对照：纯内存、不花配额，三个方向各撞一次（漏进嵌套块／名字缺席／行形制变了）
_S1 = 'if x:\n    TT = [("a", _E + "\\n", 1, True), ("b", "y", 0, False)]\n'
assert table_counts(_S1, "TT") == (2, 1, 1), "嵌套块里的表数不出（尺子还在只扫模块级）"
for _bad_src, _why in (( 'TT = 3\n', "右边不是列表" ),
                       ( 'TT = [("a", "y", 1, 0)]\n', "末位不是布尔" ),
                       ( 'TT2 = [("a", "y", 1, True)]\n', "名字缺席" ),
                       ( 'TT = [("a", "y", 1, True)]\nTT = [("b", "y", 0, False)]\n', "同名两处赋值" )):
    try:
        table_counts(_bad_src, "TT")
    except AssertionError:
        pass
    else:
        raise AssertionError("table_counts 对坏样本（%s）没有拒绝 ⇒ 它数出来的东西不可信" % _why)
SP2TAB, SP2ACC, SP2REJ = table_counts(_ctl_src, "NOTB_TABLE")
assert SP2TAB == SP2ACC + SP2REJ and SP2ACC >= 1 and SP2REJ >= 1, \
    "notb 判定表 %d 条／须收 %d／须拒 %d ⇒ 这张表至少有一头验不到" % (SP2TAB, SP2ACC, SP2REJ)
print("   notb 判定表从源码结构现取＝%d 形状（须收 %d／须拒 %d），尺子自己的 5 个坏样本全被拒 -> PASS" % (
    SP2TAB, SP2ACC, SP2REJ))
SELFNEW = one(r"本面自查＝(\d+) 条", open(os.path.join(EVID, "_raw_s%d_hdr_controls.out" % SLICE),
                                         encoding="utf-8").read(), "本次真跑的本面自查条数")
assert int(SELFNEW) == int(SP2OLD) + 1, "本面自查从 %s 条变成 %s 条（须 ＋1＝新增那条谓词自查）" % (
    SP2OLD, SELFNEW)
print("   判据侧现场：stdout %d B／stderr %d B｜未命中 %d 行｜判读＝%s｜实得抛出行在生成器第 %d 行「%s」"
      % (SP2BO, SP2BE, SP2MISS, SP2VERDICT, SP2LINE, SP2RAISE))
print("   修复取证：notb 谓词判定表 %d 个形状（须收 %d／须拒 %d）｜本面自查 %s 条 → %s 条"
      % (SP2TAB, SP2ACC, SP2REJ, SP2OLD, SELFNEW))

# ============================ 读数 4：待入库清单（易失，只进散文） ============================
st = [l.rstrip("\n") for l in sh("git", "-c", "core.quotePath=false", "status", "--porcelain").splitlines()
      if l.strip()]
UNT = sorted(l[3:] for l in st if l.startswith("?? "))
MOD = sorted(l[3:] for l in st if not l.startswith("?? "))
NU, NMOD = len(UNT), len(MOD)
assert NU + NMOD == len(st) == len(set(st)), "清单行数与未跟踪＋修改之和不等 ⇒ 有重行或漏行"
assert len(st) == int(sh("bash", "-c",
                         "git -c core.quotePath=false status --porcelain | wc -l").strip()), \
    "本件解析的行数与 `wc -l` 现算不等 ⇒ 两把尺子在数不同的东西"
# 组名**不带**预期件数：件数一律现算（写盘前的预检与写盘时同一组会差几件，写死的数字就成了假话）。
ROLES = [
    ("回执链", r"^am_s7_verify%d\.(py|log)$|^_raw_verify%d\.(out|err)$|^am_s%d_hdr\.py$"
     % (SLICE, SLICE, SLICE)),
    ("表头预检与写盘两面", r"^_raw_s%d_hdr_(precheck|write)\.(out|err)$" % SLICE),
    ("表头控件面", r"^am_s%d_hdr_controls\.py$|^_raw_s%d_hdr_controls\.(out|err)$" % (SLICE, SLICE)),
    ("控件面红灯现场（自审反转＋判据空转两次）", r"^_raw_s%d_hdr_controls\.attempt[0-9]+-.*\.(out|err)$"
     % SLICE),
    ("写盘后只读仪器与其红路终稿两面",
     r"^am_s%d_hdr_after\.py$|^_raw_s%d_hdr_after\.(out|err)$|^_raw_s%d_hdr_after_(pre_final|red1)\.(out|err)$"
     % (SLICE, SLICE, SLICE)),
    ("写盘后表头复跑", r"^_raw_s%d_hdr_after_write\.(out|err)$" % SLICE),
    ("只读仪器的控件面", r"^am_s%d_hdr_after_controls\.py$|^_raw_s%d_hdr_after_controls\.(out|err)$"
     % (SLICE, SLICE)),
    ("落地腿（本件与其原始件）",
     r"^am_s%d_docs_landing\.py$|^_raw_s%d_landing_(precheck|write)\.(out|err)$" % (SLICE, SLICE)),
    ("落地腿的红灯现场（预检与写盘各自的失败趟）",
     r"^_raw_s%d_landing_[a-z]+\.attempt[0-9]+-.*\.(out|err)$" % SLICE),
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
MOD_TXT = "／".join(os.path.basename(p) for p in MOD) if MOD \
    else "（落笔时 0 张——五张档案面刚随上一笔提交入库；本件落盘后立刻变成 5 张）"
print("   待入库现数（易失，不作下一轮分母）：未跟踪 %d＋修改 %d＝%d 处｜分组和＝%s" % (
    NU, NMOD, NU + NMOD, BKT_SUM))


def bucket_control():
    """分组的正对照：把回执链那组的桶形改窄到认不出表头生成器，它必须**恰好少认领 1 件**。
    不少（＝改坏了还认同样多）或认 0 件（＝整组散架，等价于另一条闸门）都说明这根针没扎在要害上。"""
    name, c0 = [b for b in buckets if b[0].startswith("回执链")][0]
    pat = [p for n, p in ROLES if n == name][0]
    bad = pat.replace(r"^am_s%d_hdr\.py$" % SLICE, r"^am_s%d_hdr_NONEXIST\.py$" % SLICE)
    assert bad != pat, "桶形替换没改变字符串 ⇒ 这根针根本没扎进去"
    hit = [p for p in UNT if re.match(bad, os.path.basename(p))]
    assert len(hit) == c0 - 1, "改窄之后那组认领 %d 件（原形制 %d，须恰好少 1）" % (len(hit), c0)
    return "%s 组由 %d 件降到 %d 件（改窄 `am_s%d_hdr\\.py` 一支）" % (name, c0, len(hit), SLICE)


print("   分组断言的正对照：%s -> PASS" % bucket_control())

# ============================ 读数 5：GPU 断言（总数只进散文，不进 stdout） ============================
_pids = [l.split(",")[0].strip() for l in sh("nvidia-smi", "--query-compute-apps=pid",
                                             "--format=csv,noheader").splitlines() if l.strip()]


def _mine(pids):
    out = []
    for pid in pids:
        try:
            c = " ".join(open("/proc/%s/cmdline" % pid).read().split("\x00"))
        except OSError:
            continue
        if c and any(m in c for m in GPU_MARK):
            out.append(pid)
    return out


NGPU = len(_pids)
MYGPU = len(_mine(_pids))
SELF_MINE = len(_mine([str(os.getpid())]))
assert MYGPU == 0, "卡上有我的解释器留下的进程（%d 个）⇒ 归还口径破了" % MYGPU
assert SELF_MINE == 1, "匹配器数不出自己的 PID ⇒ 上面那个 0 是空转，不可信"
# 分母的**存在性**要能红，但「卡上 0 个进程」是合法状态（别人的作业跑完了），不许把它当成查询失败而拒绝写盘。
_nvname = sh("nvidia-smi", "--query-gpu=name", "--format=csv,noheader").strip()
assert _nvname and "\n" not in _nvname, "nvidia-smi 连卡名都取不到 ⇒ GPU 那句断言没有出处（这才是查询失败）"
print("   GPU：卡名现取＝%s｜分母 >0 与否都算合法（易失，总数不写进本件 stdout）｜我的＝0｜"
      "同一匹配器对自己的 PID 数出 %d -> PASS" % (_nvname, SELF_MINE))

# ============================ CTX 汇总 ============================
CTX.update({
    "SLICE": str(SLICE), "PREVSLICE": str(PREVSLICE), "NEXTSLICE": str(NEXTSLICE),
    "HEAD40": HEAD, "HEAD7": HEAD7, "PAR40": PAR40, "PAR7": PAR7, "CDATE": CDATE,
    "TIME": TIME, "NCODE": str(NCODE), "NAC": str(NAC), "NMC": str(NMC), "NTOTC": str(NTOTC),
    "SSRAW": SS_RAW, "NSINS": str(NS_INS), "NSDEL": str(NS_DEL),
    "R30LOG": R30LOG, "R30B": str(R30B), "R30L": str(R30L), "OFF30": str(OFF30),
    "BODYB": str(BODYB), "ERRMB": str(ERRMB), "CMPT": CMPT, "FLDN": str(FLDN), "HRAV": HRAV,
    "TRUNC": TRUNC, "TB": str(TB), "BB": str(BB), "LB": str(LB), "BT": str(BT), "PB": str(PB),
    "CLOS30": CLOS30, "DIFF3": DIFF3, "SHA4": SHA4, "PER4": PER4, "V30": V30,
    "CA": CA, "CB": CB, "CC": CC, "CD": CD, "CLOSEX": str(CLOSEX),
    "SPNAME": SPNAME, "SPBO": str(SPBO), "SPBE": str(SPBE), "SPWHY": SPWHY, "NLOST": str(NLOST),
    "SP2NAME": SP2NAME, "SP2BO": str(SP2BO), "SP2BE": str(SP2BE), "SP2MISS": str(SP2MISS),
    "SP2VERDICT": SP2VERDICT, "SP2RAISE": SP2RAISE, "SP2LINE": str(SP2LINE),
    "SP2TAB": str(SP2TAB), "SP2ACC": str(SP2ACC), "SP2REJ": str(SP2REJ),
    "SP2OLD": SP2OLD, "SELFNEW": SELFNEW,
    "NU": str(NU), "NMOD": str(NMOD), "TOT0": str(NU + NMOD), "BKT": BKT_TXT,
    "BKTSUM": BKT_SUM, "MODTXT": MOD_TXT, "NGPU": str(NGPU), "FINGER": FINGER_REC[:12],
    "LESSON_NEW": LESSON_NEW, "EVIDREL": EVID_REL, "RAWM": RAWM,
})

# ============================ 节号与补记序号：从档案现取，不写死 ============================
DEV_NOW = open(DEVLOG, encoding="utf-8").read()
_secs = list(re.finditer(r"^### 26\.(\d+)[^\n]*$", DEV_NOW, re.M))
assert _secs, "开发日志里取不到 `### 26.NN` 标题 ⇒ 节号是凭记忆的"
SECREF = _secs[-1].group(1)
assert "分片 %s" % SLICE in _secs[-1].group(0), (
    "最后一节标题不含「分片 %s」⇒ 本节的归属不是现取：%s" % (SLICE, _secs[-1].group(0)))
_seg0 = DEV_NOW[_secs[-1].end():]
_nx0 = _seg0.find("\n### ")
SEC_BODY = _seg0[:_nx0] if _nx0 != -1 else _seg0
_ptxt = open(PLAN, encoding="utf-8").read()
_pnums = [int(x) for x in re.findall(r"^### §W 补记 (\d+)", _ptxt, re.M)]
assert _pnums, "计划文档里取不到 `### §W 补记 N` 标题 ⇒ 补记序号是凭记忆的"
PNUM = str(max(_pnums) + 1)
CTX["SECREF"] = SECREF
CTX["PNUM"] = PNUM
print("   节号 §26.%s 与补记序号 §W 补记 %s 均从档案现取 -> PASS" % (SECREF, PNUM))

for _k, _v in sorted(CTX.items()):
    assert isinstance(_v, str) and _v != "" and "\n" not in _v, "%s 不是单行非空字符串：%r" % (_k, _v[:40])
print("   CTX %d 个键全部是非空单行字符串（值从命令现取，没有一个是我手敲的）-> PASS" % len(CTX))

# ============================ 五张面的模板 ============================
DEV_T = r"""- **分片 %(SLICE)s 落地腿的真读数（%(TIME)s 现取；把上一子条那句「此刻还不存在」的空格填上，值全部由命令现取）**：HEAD＝`%(HEAD40)s`（父 `%(PAR40)s`，短哈希 `%(HEAD7)s`／`%(PAR7)s`，提交时刻 `%(CDATE)s` 由 `git log -1 --format=%%cd --date=iso` 现取），推送区间 `%(PAR7)s..%(HEAD7)s`；入库面＝新增 (A) **%(NAC)s**＋修改 (M) **%(NMC)s**＝**%(NTOTC)s** 件；`git diff-tree --no-commit-id --shortstat -r HEAD` 原文＝`%(SSRAW)s`（**没有删除段**），插入数按 `--numstat` 逐行现算＝**%(NSINS)s**、删除＝**%(NSDEL)s**，两把尺子当场对账 ⇒ 本片是纯归档腿且零删除；生产树 `src`＋`tests` 改动现数 **%(NCODE)s** 件。
  - **落地证明五步＋四个正对照（逐字取自本目录的 `%(RAWM)s`，本条不许自己造数）**：① `ls-remote` refs/heads/main == 本地 HEAD -> MATCH；② `truncated=%(TRUNC)s`，tree 条目 **%(TB)s**＝blob **%(BB)s**＋tree **%(BT)s**，本地 blob＝%(LB)s、父提交 blob＝%(PB)s；③ 双向路径差集（两侧同用 `core.quotePath=false` 与 `LC_ALL=C sort`）＝`%(DIFF3)s`；④ `%(SHA4)s`；`%(PER4)s`；⑤ `%(CLOS30)s`，四个正对照各半句＝`%(CA)s`／`%(CB)s`／`%(CC)s`／`%(CD)s`；判读行 `%(V30)s`。闭合式右边那两个自变量另由 git 现取并独立复算：%(BB)s − (%(PB)s ＋ %(NAC)s) ＝ **%(CLOSEX)s**（须 0），正文里远端与本地两个 blob 也当场相等（%(BB)s＝%(LB)s）；`SRC指纹` 那枚前 12 位 `%(FINGER)s` 与**默认 locale** 下的盘上现算逐字符相同 ⇒ 表头说的就是当前这棵树。
  - **回执表头仍由生成器产出（不许手工组装）**：档案名＝`%(R30LOG)s` **%(R30B)s** B／**%(R30L)s** 行＝表头 **%(OFF30)s**＋正文 **%(BODYB)s**；`BODY_OFFSET=%(OFF30)s` 由写入件重解一遍并核对到正文区与原始件**逐字节相同**，写盘后另起一件**只读仪器**在盘上重解、并自己起进程跑 `cmp -i %(OFF30)s:0`，三联＝`%(CMPT)s`（0＝正文区与原始件相同，两侧偏移各 +1 都变红），顶格字段 **%(FLDN)s/%(FLDN)s** 与生成器 `FIELDS` 声明互证；核验件 stderr **%(ERRMB)s** B；只读仪器判读行 `%(HRAV)s`。
  - **三面干跑控件面的真跑读数（判红口径继承上一片：rc≠0 且**抛出的那一行**含指定闸门句，且 raise 行号落在该句源码 ±3 行窗口；超时既不判红也不判过、单独计数）**：%(MSG_SCRIPT)s 的 stdout＝%(MSG_OUT)s **%(MSG_OUTB)s** B（md5 前 12 `%(MSG_OUTM)s`／stderr %(MSG_ERRB)s B）｜声明 **%(MSG_DEC)s** 条｜超时 **%(MSG_TO)s** 条｜造不出红路登记 **%(MSG_UNC)s** 条 ⇒「%(MSG_VERDICT)s」；%(HDR_SCRIPT)s 的 stdout＝%(HDR_OUT)s **%(HDR_OUTB)s** B（md5 前 12 `%(HDR_OUTM)s`／stderr %(HDR_ERRB)s B）｜红路声明 **%(HDR_DEC)s** 条｜保持绿 **%(HDR_GRN)s** 条｜超时 **%(HDR_TO)s** 条｜造不出红路 **%(HDR_UNC)s** 条 ⇒「%(HDR_VERDICT)s」；%(AFTC_SCRIPT)s 的 stdout＝%(AFTC_OUT)s **%(AFTC_OUTB)s** B（md5 前 12 `%(AFTC_OUTM)s`／stderr %(AFTC_ERRB)s B）｜声明 **%(AFTC_DEC)s** 条｜%(AFTC_PAIRS)s｜超时 **%(AFTC_TO)s** 条｜造不出红路 **%(AFTC_UNC)s** 条 ⇒「%(AFTC_VERDICT)s」。
  - **%(LESSON_NEW)s＝本轮抓到的一处判断缺陷，缺陷在控件面对它自己的判读上**：表头控件面第一次真跑把整批子运行跑完之后，**收面自查**那条把自己的绿判成了红——自审写成「快照一致时 `diff_snap` 必须非空」，可 helper 在健康路径下不抛异常，于是标志位恒 False＝一条**按构造红不了**的判据反过来把正确实现判红。现场两件按别名留档：`%(EVIDREL)s/%(SPNAME)s.out` **%(SPBO)s** B／`.err` **%(SPBE)s** B，抛出行「%(SPWHY)s」。修法两条都已进该面源码：① 自审改成正向双判据（不一致要能抓出、一致要说成空），② 逐行刷盘——第一版把整批表攒到最后统一印，被这次崩溃整批丢掉，那 **%(NLOST)s** 条读数当场作废、只能重跑（条数＝现场件首行自己声明的红路＋绿路之和，从该件现取；重跑要再付一次配额，这是本轮最贵的一笔）。
  - **同一族的第二处出在判据侧＝闸门响了却被记成未命中（第二次真跑收在判据空转）**：表头控件面的 E6 把生成器 `am_s%(SLICE)s_hdr.py` 第 **%(SP2LINE)s** 行那条不动点上限砍到 1，子运行 rc=1、stderr 末行逐字符等于那条 raise 的参数「%(SP2RAISE)s」，本面第一版却判它未命中——那条合取要求抛出行以 `SystemExit:` 打头，而 CPython 对未捕获的 `SystemExit(str)` 只把消息原样写进 stderr、连 traceback 都不打 ⇒ 恒假的合取＝又是**按构造红不了**，只是这次卡的是判据那一侧而不是闸门那一侧。现场两件按别名留档：`%(EVIDREL)s/%(SP2NAME)s.out` **%(SP2BO)s** B／`.err` **%(SP2BE)s** B，表里「× 」行 **%(SP2MISS)s** 条，整面判读＝%(SP2VERDICT)s。修法两条：① 把谓词从判定段抽成纯函数、按实测到的形状重写（rc≠0＋期望句落在末条非空行＋全文无 Traceback 且无 AssertionError＋闸门句在副本恰好 1 行）；② 给谓词自己配一条**不花一次配额**的内存判定表（%(SP2TAB)s 个形状＝须收 **%(SP2ACC)s** 个＋须拒 **%(SP2REJ)s** 个，两个方向各验一遍）。本面自查因此从 **%(SP2OLD)s** 条变成 **%(SELFNEW)s** 条，两个数分别取自现场件与本次真跑的 stdout。
  - **待入库（%(TIME)s 现数，只作排期不作承诺）**：`git status --porcelain` 现算未跟踪 **%(NU)s** 件＋修改 **%(NMOD)s** 件＝**%(TOT0)s** 处，分组＝%(BKT)s（和＝%(BKTSUM)s）；修改面＝%(MODTXT)s。**最终入库数大于这里的 %(TOT0)s**——归档自查件及其原始件此刻还不存在，那一部分要等它落盘；再往后由分片 %(NEXTSLICE)s 提交时 `git show --name-status HEAD` 现取，不在这里预支。
  - **树与钉住**：%(TIME)s 现取；钉住口径**同本节上面那条带引号的登记**（本轮每条计算命令逐条内联 `CUDA_VISIBLE_DEVICES=""`，含一行式命令；**不产生任何吞吐／加速／预算数字**）；卡上 compute 进程总数（落笔现取 **%(NGPU)s** 个）与「我的解释器所留 **0** 个」只作断言，且总数不写进任何可复跑件的 stdout（它随他人作业变）；那条 0 由同一匹配器对自己的 PID 数出 1 来正对照兜住。
"""

OVER_T = r"""- **%(TIME)s（分片 %(SLICE)s 落地腿＝提交＋推送＋五步落地证明＋生成器回执表头＋三面控件面；生产树 src＋tests 改动 %(NCODE)s、GPU 全程未碰）**（开发日志 §26.%(SECREF)s「分片 %(SLICE)s＝提交腿」）：
  - **入库面**：`%(HEAD7)s`（父 `%(PAR7)s`）＝新增 (A) **%(NAC)s**＋修改 (M) **%(NMC)s**＝**%(NTOTC)s** 件；shortstat 原文「%(SSRAW)s」＋`--numstat` 现算 %(NSINS)s／%(NSDEL)s ⇒ 零删除的纯归档腿；推送区间 `%(PAR7)s..%(HEAD7)s`。
  - **落地证明五步＋四个正对照**：`truncated=%(TRUNC)s`，tree 条目 **%(TB)s**＝blob **%(BB)s**＋tree **%(BT)s**；闭合式「%(CLOS30)s」，且 %(BB)s − (%(PB)s ＋ %(NAC)s) ＝ %(CLOSEX)s 由 git 现算复对；双向路径差集「%(DIFF3)s」；全库 blob SHA 普查「%(SHA4)s」；本片单列复核「%(PER4)s」；判读行 `%(V30)s`。四个正对照（远端某条 sha 首位改 f／远端删一条路径／闭合式右边 +1／远端删本片一条路径）各自能红。
  - **回执表头**：`%(R30LOG)s` **%(R30B)s** B／**%(R30L)s** 行＝表头 **%(OFF30)s**＋正文 **%(BODYB)s**（正文区与 `%(RAWM)s` 逐字节相同，`cmp -i %(OFF30)s:0` 三联 `%(CMPT)s` 由只读仪器自己起进程跑）；顶格字段 %(FLDN)s/%(FLDN)s；核验件 stderr %(ERRMB)s B。
  - **三面控件面**：提交说明面 **%(MSG_DEC)s** 条扰动／造不出红路 **%(MSG_UNC)s** 条；表头面红路 **%(HDR_DEC)s** 条／保持绿 **%(HDR_GRN)s** 条／超时 **%(HDR_TO)s** 条／造不出红路 **%(HDR_UNC)s** 条；只读仪器面 %(AFTC_PAIRS)s／造不出红路 **%(AFTC_UNC)s** 条；三面判读一律「ALL CONTROLS CAN GO RED」。
  - **一条判断缺陷级新学＝%(LESSON_NEW)s**：控件面的收面自查按构造红不了，反过来把正确实现判红（现场 `%(SPNAME)s` 两件 %(SPBO)s／%(SPBE)s B 留档）；且"证据攒到最后统一印"在配额受限的面上一崩就整批丢 ⇒ 自审要正向双判据、每条子运行即时落盘。同族的第二处出在判据侧：E6 要求抛出行带 `SystemExit:` 前缀，而 CPython 对未捕获的 `SystemExit(str)` 只打裸消息 ⇒ 闸门响了仍记未命中，第二次真跑因此收在 %(SP2VERDICT)s（现场 `%(SP2NAME)s`，未命中行 %(SP2MISS)s 条）。
  - **排期不变**（**#30 试片尺寸裁决在前**、`--stage full` 暂停）；红线：A0 断言一字不动、#27 不换指标、无新增 skip/xfail、性能数字只在 GPU 窗口测。
"""

PLAN_T = r"""### §W 补记 %(PNUM)s（%(TIME)s 落笔，`date '+%%F %%T %%z'` 现取）＝分片 %(SLICE)s 的**落地腿**：仓库推进到 `%(HEAD7)s`／远端 blob **%(BB)s**，并把「控件面对自己判读」这类缺陷折成口径

- **外部事实只有一条**：远端 main ＝ `%(HEAD40)s`（父 `%(PAR40)s`），推送区间 `%(PAR7)s..%(HEAD7)s`；本片＝**纯归档腿**，生产树 `src`＋`tests` 改动 **%(NCODE)s** 件。入库面＝新增 (A) **%(NAC)s**＋修改 (M) **%(NMC)s**＝**%(NTOTC)s** 件；shortstat「%(SSRAW)s」与 `--numstat` 现算（%(NSINS)s 插入／%(NSDEL)s 删除）两口径当场对账。
- **落地证明的读数**（逐条从归档正文 `%(RAWM)s` 正则现取，其 stderr **%(ERRMB)s** B）：tree 条目 **%(TB)s**＝blob **%(BB)s**＋tree **%(BT)s**；%(CLOS30)s；双向差集「%(DIFF3)s」；全库 blob SHA「%(SHA4)s」；本片单列「%(PER4)s」；判读行 %(V30)s。闭合式另用 git 现算复对：%(BB)s − (%(PB)s ＋ %(NAC)s) ＝ **%(CLOSEX)s**。
- **回执表头**：`%(R30LOG)s` **%(R30B)s** B／**%(R30L)s** 行＝表头 **%(OFF30)s**＋正文 **%(BODYB)s**，`BODY_OFFSET=%(OFF30)s` 由写入件重解；三联 `cmp -i %(OFF30)s:0`＝`%(CMPT)s` 由**只读仪器**自己起进程跑（不是抄生成器的），顶格字段 %(FLDN)s/%(FLDN)s 与生成器 `FIELDS` 互证；只读仪器判读行 `%(HRAV)s`。
- **三面控件面与一条口径级变化**：提交说明面 stdout **%(MSG_OUTB)s** B（md5 前 12 `%(MSG_OUTM)s`）＝声明 **%(MSG_DEC)s** 条／造不出红路 **%(MSG_UNC)s** 条；表头面 stdout **%(HDR_OUTB)s** B（md5 前 12 `%(HDR_OUTM)s`）＝红路声明 **%(HDR_DEC)s** 条／保持绿 **%(HDR_GRN)s** 条／超时 **%(HDR_TO)s** 条／造不出红路 **%(HDR_UNC)s** 条；只读仪器面 stdout **%(AFTC_OUTB)s** B＝%(AFTC_PAIRS)s／造不出红路 **%(AFTC_UNC)s** 条。**新学＝%(LESSON_NEW)s**：那条按构造红不了的自审反过来把自己的绿判成红（现场 stdout **%(SPBO)s** B／stderr **%(SPBE)s** B 留档，抛出行「%(SPWHY)s」），而"证据攒到最后统一印"在配额受限的面上一崩就整批丢 ⇒ 自审要正向双判据、每条子运行即时落盘。**同一族的第二处出在判据侧**：E6 的合取要求抛出行带 `SystemExit:` 前缀，而 CPython 对未捕获的 `SystemExit(str)` 只把消息原样写进 stderr、连 traceback 都不打 ⇒ 那条判据按构造恒假，闸门响了（rc=1、末行就是那句）仍记未命中，第二次真跑收在 %(SP2VERDICT)s；修＝谓词抽成纯函数、按实测形状重写，再配一张 %(SP2TAB)s 形状的内存判定表（须收 %(SP2ACC)s／须拒 %(SP2REJ)s，不花一次配额）。
- **待入库与排期**：落笔现数未跟踪 **%(NU)s**＋修改 **%(NMOD)s**＝**%(TOT0)s** 处（分组和＝%(BKTSUM)s），最终入库数大于它（归档自查件此刻还不存在）；分片 %(NEXTSLICE)s 的表头生成器要在 §26.%(SECREF)s 里现取**唯一**一条引用 `%(R30LOG)s` 的子条——本节由落地段恰好提供 1 条，再复述一次就会被它拒绝。**红线不变**：A0 断言一字不动、#27 不换指标、无新增 skip/xfail、性能数字只在 GPU 窗口测（卡上 compute 进程 **%(NGPU)s** 个、我的 0 个，只作断言）。
"""

DAILY_T = r"""## %(TIME)s：分片 %(SLICE)s 落地腿＝替分片 %(PREVSLICE)s 入库整条链（A **%(NAC)s**／M **%(NMC)s**＝**%(NTOTC)s** 件）＋一条判断缺陷级新学＝%(LESSON_NEW)s

- **一句话**：远端 main ＝ `%(HEAD40)s`（父 `%(PAR40)s`，提交时刻 `%(CDATE)s`），入库 A %(NAC)s／M %(NMC)s＝%(NTOTC)s 件；shortstat「%(SSRAW)s」＋numstat %(NSINS)s／%(NSDEL)s ⇒ 纯归档腿、零删除、生产树改动 %(NCODE)s。
- **落地证明**（全从 `%(RAWM)s` 现取）：tree 条目 **%(TB)s**＝blob **%(BB)s**＋tree **%(BT)s**；「%(CLOS30)s」，git 现算复对差 **%(CLOSEX)s**；差集「%(DIFF3)s」；全库 SHA「%(SHA4)s」；本片单列「%(PER4)s」；判读行「%(V30)s」。
- **回执**：`%(R30LOG)s` %(R30B)s B／%(R30L)s 行＝表头 %(OFF30)s＋正文 %(BODYB)s（正文区逐字节＝原始件），三联 `%(CMPT)s`，顶格字段 %(FLDN)s/%(FLDN)s，stderr %(ERRMB)s B。
- **三面控件面**：%(MSG_OUT)s %(MSG_OUTB)s B／%(HDR_OUT)s %(HDR_OUTB)s B／%(AFTC_OUT)s %(AFTC_OUTB)s B；红路声明 %(HDR_DEC)s、保持绿 %(HDR_GRN)s、超时 %(HDR_TO)s；三面判读一律 ALL CONTROLS CAN GO RED。
- **事故与修法**：表头控件面第一次真跑把整批子运行跑完后被**自己的收面自查**判红（那条判据按构造红不了），且第一版把读数攒到最后统一印 ⇒ 整批丢。现场 `%(EVIDREL)s/%(SPNAME)s` 两件 %(SPBO)s／%(SPBE)s B 留档，抛出行「%(SPWHY)s」。修＝正向双判据＋逐行刷盘，重跑再付一次配额。第二次真跑又栽在同族的**判据侧**：E6 要求抛出行带 `SystemExit:` 前缀，而 CPython 对未捕获的 `SystemExit(str)` 只打裸消息 ⇒ rc=1、末行就是「%(SP2RAISE)s」（生成器第 %(SP2LINE)s 行那条 raise）却记成未命中，整面收在 %(SP2VERDICT)s。现场 `%(EVIDREL)s/%(SP2NAME)s` 两件 %(SP2BO)s／%(SP2BE)s B、未命中 %(SP2MISS)s 行。修＝谓词抽成纯函数按实测形状重写＋零配额内存判定表 %(SP2TAB)s 形状（收 %(SP2ACC)s／拒 %(SP2REJ)s），本面自查 %(SP2OLD)s 条 → %(SELFNEW)s 条。
- **待入库**：未跟踪 **%(NU)s**＋修改 **%(NMOD)s**＝**%(TOT0)s** 处（分组和＝%(BKTSUM)s）；最终入库数大于它，归档自查件此刻还不存在。**GPU**：卡上 compute 进程现取 %(NGPU)s 个、我的 0 个（同一匹配器对自己的 PID 数出 1）；本轮不产生任何吞吐／加速／预算数字。
"""

WBIDX_T = r"""- **%(TIME)s（分片 %(SLICE)s 落地腿；生产树 0 改动、GPU 全程未碰）**：
  远端 main `%(HEAD7)s`（父 `%(PAR7)s`；入库 A **%(NAC)s**／M **%(NMC)s**＝**%(NTOTC)s** 件，shortstat「%(SSRAW)s」＝零删除）。五步落地证明全绿：tree 条目 **%(TB)s**＝blob **%(BB)s**＋tree **%(BT)s**；「%(CLOS30)s」（git 现算复对差 %(CLOSEX)s）；差集 %(DIFF3)s；全库 SHA 普查 %(SHA4)s；本片单列 %(PER4)s；判读行 %(V30)s。
  回执 `%(R30LOG)s` **%(R30B)s** B／**%(R30L)s** 行＝表头 **%(OFF30)s**＋正文 **%(BODYB)s**，三联 `cmp -i %(OFF30)s:0`＝`%(CMPT)s`（只读仪器自跑），顶格字段 %(FLDN)s/%(FLDN)s。
  三面控件面：提交说明面声明 %(MSG_DEC)s／造不出红路 %(MSG_UNC)s；表头面红路 %(HDR_DEC)s／保持绿 %(HDR_GRN)s／超时 %(HDR_TO)s／造不出红路 %(HDR_UNC)s；只读仪器面 %(AFTC_PAIRS)s／超时 %(AFTC_TO)s。新学＝%(LESSON_NEW)s（自审按构造红不了＝把自己的绿判成红；判据读错输出形状＝把确确实实响了的闸门记成未命中，同族第二次出在这一侧；配额受限的面要逐行刷盘）。
  待入库＝未跟踪 **%(NU)s**＋修改 **%(NMOD)s**＝**%(TOT0)s** 处（分组和 %(BKTSUM)s；最终数大于它，归档自查件还没写）。排期不变：**#30 裁决在前**、`--stage full` 暂停；红线不变。
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
# 这条要验的是「值里带斜杠时替换逐字节不变」。期望不能手打：EVIDREL 的取值随证据目录（跨日期会变），
# 上一版把它写成 "x docs/ y" 就成了一条**按构造红不了**的对照——第三次栽在同一族，这次在我自己的门上。
assert "/" in CTX["EVIDREL"], "EVIDREL＝%s 里没有斜杠 ⇒ 这条斜杠对照是空转，换一个带斜杠的值再验" % CTX["EVIDREL"]
assert "x " + CTX["EVIDREL"] + " y" == named("自检/斜杠槽", "x %(EVIDREL)s y"), "含斜杠的值填进具名槽被改写了"
must_reject("自检/裸百分号", "git log --format=%cd 现取")
must_reject("自检/全角槽", "槽＝%（ SLICE）s")
must_reject("自检/下标槽", "槽＝%(MSG[OUT]s）")
print("== named() 自检：好模板逐字节正确，裸百分号／全角槽／下标槽三样各被拦下 -> PASS")

texts = {}
for k, (_, tmpl) in FACES.items():
    texts[k] = named(k, tmpl)

_used = set(re.findall(r"%\(([^)]+)\)s", "".join(t for _, t in FACES.values())))
_dead = sorted(set(CTX) - _used)
assert not _dead, "取到的读数没进任何一张面（＝算了没人读的字段）：%s" % _dead
_missing = sorted(_used - set(CTX))
assert not _missing, "模板里有没取到值的具名槽（`%` 替换会当场 KeyError，但先在这里点名）：%s" % _missing
print("   读数消费普查：CTX %d 个键与模板槽位 %d 个一一对应（未消费 0／缺值 0）-> PASS" % (len(CTX), len(_used)))


# ============================ 结构闸门（每道都配一根能红的针） ============================
print("== 结构闸门")


def strip_code(s):
    return re.sub(r"`[^`\n]*`", "", s)


def structure_problems(v):
    stripped = strip_code(v)
    bad = [i for i, l in enumerate(stripped.splitlines(), 1) if l.count("（") != l.count("）")]
    if stripped.count("**") % 2:
        bad.append("粗体")
    if not v.endswith("\n"):
        bad.append("尾换行")
    return bad


for k, v in texts.items():
    bad = structure_problems(v)
    assert not bad, "%s 结构不过（%s）" % (k, bad[:4])
    print("   %s：%d 行｜括号按行配平｜粗体成对｜尾换行｜无残留槽 -> PASS" % (k, len(v.splitlines())))
_p = structure_problems("a（未闭\n**单\n")
assert len(_p) == 2, "结构检测器对自己的坏样本抓到 %d 处（须 2＝一行不配平＋粗体不成对）⇒ 没分辨力：%s" % (
    len(_p), _p)
print("   结构检测器的正对照：坏样本两项全被抓到 -> PASS")


def secrefs(v):
    """节号引用必须写成开发日志自己的形制 `§26.NN`；裸 `§32` 读者找不到那一节。"""
    return re.findall(r"§(?!26\.)\d+", v)


for k, v in texts.items():
    bad = secrefs(v)
    assert not bad, "%s 有裸节号 %s ⇒ 引用要写成 §26.NN，否则读者按 §32 搜不到标题" % (k, bad[:4])
_bad = secrefs(texts["OVER"].replace("§26.%s" % SECREF, "§%s" % SECREF, 1))
assert len(_bad) == 1, "裸节号检测器抓到 %d 处（须 1）⇒ 它没有分辨力" % len(_bad)
print("   五面节号形制全部＝§26.NN（检测器对自己的坏样本抓到 1 处）-> PASS")

_lab_now = [m.group(1) for m in re.finditer(r"^- \*\*(.+?)\*\*", SEC_BODY, re.M)]
_lab_new = [m.group(1) for m in re.finditer(r"^- \*\*(.+?)\*\*", texts["DEV"], re.M)]
assert len(_lab_new) == 1, "落地片段里的顶级子条＝%d 条（须恰好 1 条，其下都是缩进子条）：%s" % (
    len(_lab_new), _lab_new)
assert not (set(_lab_now) & set(_lab_new)), "落地子条标签与本节已有标签撞名 ⇒ 按标签切段会串行"
_lab_all = _lab_now + _lab_new
assert len(_lab_all) == len(set(_lab_all)) == len(_lab_now) + 1, (
    "§26.%s 顶级子条：落笔前 %d 条、合并后应为 %d 条，现算 %d 条／互异 %d 条" % (
        SECREF, len(_lab_now), len(_lab_now) + 1, len(_lab_all), len(set(_lab_all))))
print("   §26.%s 顶级子条：落笔前 %d 条、落地片段贡献 1 条、合并后 %d 条且标签互异 -> PASS" % (
    SECREF, len(_lab_now), len(_lab_all)))

NEEDLES = (("引用本片回执名", re.escape(R30LOG), 1),
           ("带引号的钉住口径登记", r"当前「[^」]+」", 0),
           ("上一片的 LESSON 锚句", re.escape(LESSON_PREV), 0),
           ("本片 LESSON 锚句", re.escape(LESSON_NEW), 1))
BAIT = {"引用本片回执名": R30LOG, "带引号的钉住口径登记": "当前「诱饵口径」",
        "上一片的 LESSON 锚句": "**%s**诱饵" % LESSON_PREV, "本片 LESSON 锚句": "**%s**诱饵" % LESSON_NEW}
for what, pat, want in NEEDLES:
    n = len(re.findall(pat, texts["DEV"], re.M))
    assert n == want, "%s 在落地片段里命中 %d 处（须恰好 %d）⇒ 节内唯一性会被分片 %s 的生成器拒绝" % (
        what, n, want, NEXTSLICE)
    _now = len(re.findall(pat, SEC_BODY, re.M))
    assert _now + want == 1, "%s 合并后在节内现数 %d（落笔前 %d＋片段 %d，须恰好 1）" % (
        what, _now + want, _now, want)
    n2 = len(re.findall(pat, texts["DEV"] + ("\n- **诱饵**：见 %s。\n" % BAIT[what]), re.M))
    assert n2 == want + 1, "%s 的正对照空转（复述一次后现数 %d，须 %d）⇒ 这条唯一性断言是恒真式" % (
        what, n2, want + 1)
_fs = r"f-string 把 `\{32\}` 当成\*\*替换字段"
assert len(re.findall(_fs, DEV_NOW)) == 1, (
    "f-string 事故句在落笔前的整份日志里不是恰好 1 处 ⇒ 分片 %s 的生成器本来就会拒绝归档" % NEXTSLICE)
assert len(re.findall(_fs, texts["DEV"])) == 0, "落地片段不该新增 f-string 事故句"
assert len(re.findall(r"^- \*\*(.+?)\*\*.*%s" % re.escape(R30LOG), SEC_BODY, re.M)) == 0, \
    "本节落笔前已经有一整条顶级子句引用本片回执名 ⇒ 唯一性被片段之外的地方破坏，先查是谁写的"
print("   四条节内唯一性各按其形制成立（片段贡献 1／0／0／1，合并后各恰好 1），且各配一根「再复述就 +1」的正对照 -> PASS")

ovtxt = open(OVERVIEW, encoding="utf-8").read()
assert ovtxt.count(ANCHOR) == 1, "总览插入锚点 `%s` 现数 %d（须 1）" % (ANCHOR, ovtxt.count(ANCHOR))
for path in FACES5:
    assert open(path, "rb").read()[-1:] == b"\n", "%s 末字节不是换行 ⇒ 追加会粘连" % path
print("   总览锚点恰好 1 处；五张追加／插入目标末字节都是换行 -> PASS")


def expect_near(label, text, pairs):
    for left, val in pairs:
        assert val, "%s：值为空" % label
        assert (left + val) in text, "%s：搜不到「%s%s」⇒ 读数没钉在自己的名词后面（位置错位的形状）" % (
            label, left, val)


expect_near("DEV", texts["DEV"], [
    ("HEAD＝`", CTX["HEAD40"]),
    ("新增 (A) **", CTX["NAC"]),
    ("档案名＝`", CTX["R30LOG"]),
    ("三联＝`", CTX["CMPT"]),
    ("判读行 `", CTX["V30"]),
    ("未跟踪 **", CTX["NU"]),
    ("现场两件按别名留档：`", CTX["EVIDREL"] + "/" + CTX["SPNAME"]),
])
expect_near("OVER", texts["OVER"], [
    ("入库面**：`", CTX["HEAD7"]),
    ("blob **", CTX["BB"]),
    ("三联 `", CTX["CMPT"]),
    ("回执表头**：`", CTX["R30LOG"]),
])
expect_near("PLAN", texts["PLAN"], [
    ("远端 main ＝ `", CTX["HEAD40"]),
    ("其 stderr **", CTX["ERRMB"]),
    ("现场 stdout **", CTX["SPBO"]),
    ("补记 ", CTX["PNUM"]),
])
expect_near("DAILY", texts["DAILY"], [
    ("远端 main ＝ `", CTX["HEAD40"]),
    ("「", CTX["CLOS30"]),
    ("三联 `", CTX["CMPT"]),
])
expect_near("WBIDX", texts["WBIDX"], [
    ("远端 main `", CTX["HEAD7"]),
    ("回执 `", CTX["R30LOG"]),
    ("＋正文 **", CTX["BODYB"]),
])
try:
    expect_near("自检/错位", texts["DEV"], [("HEAD＝`", CTX["PAR40"])])
except AssertionError:
    pass
else:
    raise SystemExit("正对照失败：邻接断言在「父哈希被填进 HEAD 槽」时没报错 ⇒ 那道门是装饰")
print("   邻接断言：五面各 3–7 对「标签＋值」原文命中；把父哈希填进 HEAD 槽当场被拒 -> PASS")

os.makedirs(OUT_DIR, exist_ok=True)
for k, v in texts.items():
    open(os.path.join(OUT_DIR, k + ".md"), "w", encoding="utf-8").write(v)
open(os.path.join(OUT_DIR, "ctx.txt"), "w", encoding="utf-8").write(
    "\n".join("%s=%s" % (kk, vv) for kk, vv in sorted(CTX.items())) + "\n")
print("== 渲染完成：%s/{DEV,OVER,PLAN,DAILY,WBIDX}.md ＋ ctx.txt（渲染面含易失读数，但它不落仓库）" % OUT_DIR)


# ============================ 写盘腿（带守卫、一次性） ============================
def faces_in(rows):
    """纯函数：`rows＝[(pid, argv 词集合)]` → 命中守卫表的进程。拆成纯函数才配得起正对照。"""
    hits = []
    for pid, toks in rows:
        watched = {t for t in toks if t in TOKENS or t.startswith(COPY_PREFIX)}
        if watched and SELF_BASE not in toks:
            hits.append((pid, sorted(watched)))
    return hits


def running_faces():
    """正在跑的仪器：按 `/proc/PID/cmdline` 的**整词**比对 basename，不按命令行子串猜。"""
    rows = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            argv = open("/proc/%s/cmdline" % d, "rb").read().split(b"\0")
        except OSError:
            continue
        rows.append((d, {os.path.basename(a.decode("utf-8", "replace")) for a in argv if a}))
    return faces_in(rows)


# 守卫的正对照：造三个假进程（表内名／前缀名／无关名）＋一个「本件自己」的样本，四条期望都必须成立。
_ctl = faces_in([("9001", {"python", "am_s%d_hdr.py" % SLICE}),
                  ("9002", {"python", COPY_PREFIX + "perturbed.py"}),
                  ("9003", {"python", "some_other_tool.py"}),
                  ("9004", {"python", SELF_BASE, "am_s%d_hdr.py" % SLICE})])
assert [p for p, _ in _ctl] == ["9001", "9002"], (
    "写盘守卫的正对照判出 %s（须＝[9001, 9002]：表内名与前缀名各 1，无关名与本件自己都不算）⇒ 那道「有仪器在跑就拒绝写盘」的闸门红不了" % (
        [p for p, _ in _ctl],))
assert _ctl[0][1] == ["am_s%d_hdr.py" % SLICE] and _ctl[1][1] == [COPY_PREFIX + "perturbed.py"], (
    "正对照命中了名字但记错了词 ⇒ 守卫报出来的现场不可信")
print("   写盘守卫正对照：4 个样本判 2 中（表内名／前缀名各 1、无关名与自我样本各排除 1）-> PASS")


if "--write" in sys.argv:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "写盘腿没钉 CVD 空串 ⇒ 与钉住口径不符"
    faces = running_faces()
    assert not faces, "有仪器正在跑 %s ⇒ 按口径不许动它可读的档案面" % faces
    for path in FACES5:
        assert open(path, "rb").read()[-1:] == b"\n", "%s 末字节不是换行 ⇒ 拒绝追加" % path
    # 一次落地不变量：本轮片段若已在面上，说明这条写盘腿跑过第二次。追加会写进两份，而节内唯一性
    # 要到「写完回读」才炸——那时档案面已经脏了。所以在动笔之前就拒。
    for path, key in [(DEVLOG, "DEV"), (PLAN, "PLAN"), (DAILY, "DAILY"), (WBIDX, "WBIDX")]:
        assert texts[key] not in open(path, encoding="utf-8").read(), \
            "%s 里已经有本片要追加的那一段 ⇒ 这条腿已经落过一次，拒绝第二次落地" % os.path.relpath(path, REPO)
    assert texts["OVER"] not in open(OVERVIEW, encoding="utf-8").read(), \
        "总览面里已经有本片要插入的那一段 ⇒ 拒绝第二次落地"
    for path, key in [(DEVLOG, "DEV"), (PLAN, "PLAN"), (DAILY, "DAILY"), (WBIDX, "WBIDX")]:
        with open(path, "a", encoding="utf-8") as f:
            f.write(texts[key])
        print("   追加 %s：%d B" % (os.path.relpath(path, REPO), len(texts[key].encode("utf-8"))))
    ins = open(OVERVIEW, encoding="utf-8").read()
    pos = ins.index(ANCHOR)
    with open(OVERVIEW, "w", encoding="utf-8") as f:
        f.write(ins[:pos] + texts["OVER"] + ins[pos:])
    print("   插入 总览：条目插在 `%s` 之前（偏移 %d）" % (ANCHOR, pos))
    print("== 闸门：写完回读")
    for path, key in [(DEVLOG, "DEV"), (OVERVIEW, "OVER"), (PLAN, "PLAN"), (DAILY, "DAILY"),
                      (WBIDX, "WBIDX")]:
        now = open(path, encoding="utf-8").read()
        assert texts[key] in now, "%s 里搜不到刚写的面 ⇒ 写入没落地或被规范化" % path
        assert now.endswith("\n"), "%s 写完后末字节不是换行" % path
        print("   %s：逐字节搜到原样＋尾换行 -> PASS" % os.path.relpath(path, REPO))
    now_dev = open(DEVLOG, encoding="utf-8").read()
    m = list(re.finditer(r"^### 26\.(\d+)[^\n]*$", now_dev, re.M))
    assert m[-1].group(1) == SECREF, "磁盘上最后一节不是 §26.%s ⇒ 插入位置错了" % SECREF
    seg = now_dev[m[-1].end():]
    nxt = seg.find("\n### ")
    seg = seg[:nxt] if nxt != -1 else seg
    for what, pat, want in NEEDLES:
        n = len(re.findall(pat, seg, re.M))
        assert n == 1, "磁盘上的 §26.%s 里 %s 命中 %d 处（预检是落笔前 %d＋片段 %d＝1）⇒ 分片 %s 会拒绝归档" % (
            SECREF, what, n, len(re.findall(pat, SEC_BODY, re.M)), want, NEXTSLICE)
    assert len(re.findall(_fs, now_dev)) == 1, "f-string 事故句在落盘后的整份日志里不是恰好 1 处"
    _lab_disk = [x.group(1) for x in re.finditer(r"^- \*\*(.+?)\*\*", seg, re.M)]
    assert len(_lab_disk) == len(set(_lab_disk)) == len(_lab_all), (
        "磁盘上 §26.%s 顶级子条＝%d／互异＝%d（预检 %d）" % (
            SECREF, len(_lab_disk), len(set(_lab_disk)), len(_lab_all)))
    print("   磁盘上的 §26.%s：%d 条顶级子条标签互异、四条唯一性各恰好 1 处 -> PASS" % (SECREF, len(_lab_disk)))
    post = [l for l in sh("git", "-c", "core.quotePath=false", "status", "--porcelain").splitlines()
            if l.strip()]
    nq = len([l for l in post if l.startswith("?? ")])
    nm = len(post) - nq
    assert nq == NU, "未跟踪件数从 %d 变成 %d ⇒ 写盘动到了不该动的地方" % (NU, nq)
    _mpost = sorted(l[3:] for l in post if not l.startswith("?? "))
    _want = sorted(os.path.relpath(p, REPO) for p in FACES5)
    assert _mpost == _want, "写盘后被改动的面不是预期的五张：%s" % _mpost
    # 这条比较自己的正对照：`git status --porcelain` 给的是**仓库相对**路径，而 FACES5 是绝对路径，
    # 上一版直接拿两者相比＝又一条按构造红不了的判据（写盘其实已经落地，却在这行崩）。
    # 所以这里当场验「尺子有分辨力」：换成绝对路径必须不相等，少一张面也必须不相等。
    assert sorted(FACES5) != _want and _mpost != _want[:4], \
        "被改面比较没有分辨力（绝对路径或删一张仍然相等）⇒ 上面那句 PASS 不可信"
    assert nm == NMOD + 5, "修改面数从 %d 变成 %d（须恰好 +5＝五张档案面各 1 处）" % (NMOD, nm)
    print("   闭合式复算：未跟踪 %d（不变）＋修改 %d（＝落笔 %d＋刚追加的五张面）-> PASS" % (nq, nm, NMOD))
    print("WRITER_VERDICT = FIVE LANDING FACES LANDED, UNIQUENESS RECOUNTED ON DISK")
else:
    print("（预检模式：未写盘。加 --write。）")
print("LANDING_FILLER_VERDICT = READINGS DERIVED, STRUCTURE GATES PASS")
