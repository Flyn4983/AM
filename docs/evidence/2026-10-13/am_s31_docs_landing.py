#!/usr/bin/env python3
r"""分片 31 的**落地填值器**（＋带守卫的写盘腿）：提交之后那五张档案面的落地段，每个读数都由命令现取。

与本片写盘前那件 `am_s31_docs.py` 的分工＝时间轴：那件在**提交之前**落盘，所以它只能登记计划、并明写
「落地读数此刻还不存在」；本件在**提交之后**落盘，它存在的唯一理由是把那句话说过的空格填上。因此本件的每条数都取自
① `git` 现算（HEAD／父／name-status／shortstat／numstat），② 已入库档案面的正则现取（回执正文、只读仪器的 stdout、
三面控件面的 stdout 与六件红灯现场），③ 盘上字节（大小、md5、`BODY_OFFSET` 定点复解、正文区与原始件逐字节相等），
④ **ast 现取**（三面控件面的 `len(CASES)`／`len(GREENS)`／`len(UNCOVERED)`，不执行任何控件面）。没有一条是我手敲的。

五条口径（前四条继承上一片那件，第五条是本片新加的真事故）：
① **具名替换＋邻接断言**——位置式槽位在「槽数相等＋顺序单调」两道检查同时绿时仍能把散文排错位。
② **渲染结构门**——按行配平全角 `（）`、剥内联码后配平 `**`、目标末字节是换行、插入锚点整份文件恰好 1 处。
③ **节内唯一性＋正对照**——按**顶级子条**为单位数「哪一条命中这枚 needle」，因为下游那把生成器的
   `bullet_for` 问的就是「恰好一条顶级子条」；同一子条里复述三次不违例，跨到第二条才违例。每条都配一根
   「再补一条顶级子条就 +1」的 bait，也预检**落笔前节内现数**，使合并后恰好 1。
④ **易失读数只进散文**——配额余量、PID 分母、落笔 date 进档案面（那里它们带日期、可被下一条覆盖），
   但**不印进本件的 stdout**（本件 stdout 会被归档、会被分片 32 引用，而它们随环境与他人作业变）。
⑤ **归档件与源码同源要当场闭合**——一面控件面的 stdout 若比它的源码旧（编辑过已暂存的仪器却没复跑），
   条数不会自己说破。本件把三面的 `len(CASES)`／`len(GREENS)`／`len(UNCOVERED)` 用 ast 现取，再与该面归档
   stdout 自己声明的条数逐面相等；只读仪器面还多合一条 `实跑行数＝红路＋绿路＋基线`。这就是①—④之外
   本片自己被咬过的那一口：控件面的汇总行改成同族形制之后，归档件当场比源码旧。

默认只渲染到 `/tmp/out31_land/`；`--write` 才落盘，且落盘前后各数一次闭合式。
"""
import ast
import hashlib
import os
import re
import subprocess
import sys

REPO = "/home/shy/桌面/2026年开发-显式冲击有限元-物质点法-sph-fvm+am/AM-Qoder"
EVID_REL = "docs/evidence/2026-10-13"
EVID = os.path.join(REPO, EVID_REL)
DEVLOG = os.path.join(REPO, "docs/开发日志.md")
OVERVIEW = os.path.join(REPO, "docs/项目开发总览.md")
PLAN = os.path.join(REPO, "docs/项目评估与下一步计划_2026-10-06.md")
DAILY = os.path.join(REPO, ".workbuddy/memory/2026-10-13.md")
WBIDX = os.path.join(REPO, ".workbuddy/memory/MEMORY.md")
FACES5 = [DEVLOG, OVERVIEW, PLAN, DAILY, WBIDX]
ANCHOR = "## 7. 文档索引"
OUT_DIR = "/tmp/out31_land"
SELF_BASE = os.path.basename(os.path.abspath(__file__))
# 写盘守卫：这些名字**正在跑**就不许动档案（按 /proc/PID/cmdline 的**整词**比对，不按命令行子串猜）。
# 本件自己不在表里：守卫对 `SELF_BASE` 那一支本来就让路，把它列进来只会造出一条永不生效的条目。
TOKENS = {"am_s31_msg.py", "am_s31_msg_controls.py", "am_s7_verify31.py", "am_s31_hdr.py",
          "am_s31_hdr_controls.py", "am_s31_hdr_after.py", "am_s31_hdr_after_controls.py",
          "am_s31_docs.py", "am_s31_archive_check.py"}
COPY_PREFIX = "zctl_s31_"
AFTC_COPY_PREFIX = "zctl_s31_after_"
CTX = {}

# 六件红灯现场与探针的**形制名**（不是件数、不是读数）：件数一律从该件现取
SP1 = "_raw_s31_msg_controls.attempt1-mark-line-hits-twice"
SP2 = "_raw_s31_msg_controls.attempt2-twelve-controls-idled"
SP3 = "_raw_s31_hdr_controls.attempt1-quota-single-read"
SP4 = "_raw_s31_hdr_controls.attempt2-f1-interp-contam-r8-repro-idled"
SP5 = "_raw_s31_hdr_controls.probe-ruler-403"
SP6 = "_raw_s31_hdr_after_controls.attempt1-greenloop-unbound-var"
SP7 = "_raw_s31_hdr_after_controls.probe-headlen-plus2-reds-at-writeface-bytes"
# 本片承接的两枚锚句（各须在 §26.33 里恰好 1 条顶级子条命中，本片段对它们的贡献须是 0）
CARRIED = ("自审的方向性", "普查仪器没有第二出处")
LESSON_NEW = "良性扰动的等长约束"
PY3 = "/tmp/amvenv/bin/python"


def sh(*args):
    """**默认环境**跑（不钉 LC_ALL）——指纹口径只在默认 locale 下与前几片可比。"""
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


def rows(text, needle):
    return [l for l in text.splitlines() if re.match(needle, l)]


def raised(text, what):
    """异常**末行**（不是消息行）：家族形制取自 `am_s31_hdr_controls.py`，判据读的就是这一行。"""
    cand = [l for l in text.splitlines()
            if re.search(r"(AssertionError|UnboundLocalError|TypeError|ValueError|SystemExit"
                         r"|SyntaxError|FileNotFoundError|Error)\b", l) and not l.startswith("File ")]
    assert cand, "%s 里取不到异常行 ⇒ 该件不是收在一条可读的抛出上" % what
    return cand[-1].strip()


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


def list_lens(path, names=("CASES", "GREENS", "UNCOVERED")):
    """ast 现取模块级列表的**条数**（不执行该件）：这是⑤那条「归档件与源码同源」的尺子。"""
    tree = ast.parse(open(path, encoding="utf-8").read())
    got = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in names:
                    assert t.id not in got, "%s 里 %s 赋值不止一次 ⇒ 数出来的不是那一张表" % (
                        os.path.basename(path), t.id)
                    assert isinstance(node.value, ast.List), "%s 的 %s 右边不是列表字面量" % (
                        os.path.basename(path), t.id)
                    got[t.id] = len(node.value.elts)
    missing = sorted(set(names) - set(got))
    assert not missing, "%s 里取不到 %s ⇒ 该件的表形制变了，条数闭合没有靶" % (
        os.path.basename(path), "／".join(missing))
    return got


# ============================ 读数 1：git 现算 ============================
HEAD = sh("git", "rev-parse", "HEAD").strip()
HEAD7 = sh("git", "rev-parse", "--short", "HEAD").strip()
PAR40 = sh("git", "rev-parse", "HEAD^").strip()
PAR7 = sh("git", "rev-parse", "--short", "HEAD^").strip()
SUBJ = sh("git", "log", "-1", "--format=%s").strip()
CDATE = sh("git", "log", "-1", "--format=%cd", "--date=iso").strip()
TIME = sh("date", "+%F %T %z").strip()
NCODE = len([l for l in sh("git", "-c", "core.quotePath=false", "status", "--porcelain", "-uall",
                           "--", "src", "tests").splitlines() if l.strip()])
assert NCODE == 0, "生产树 src＋tests 有 %d 处改动 ⇒ 本片不是纯归档腿，那句散文要先改" % NCODE
# 落地腿的片号＝HEAD 自己标的那个号（本件在提交之后跑，照抄写盘前那件的 `+1` 会去要一份从没生成过的档案）。
SLICE = int(one(r"S7 分片 (\d+)", SUBJ, "HEAD 提交说明里的片号"))
PREVSLICE, NEXTSLICE = SLICE - 1, SLICE + 1
_nums = sorted({int(x) for x in re.findall(r"分片 (\d+)", SUBJ)})
assert _nums == [PREVSLICE, SLICE], (
    "HEAD 提交说明里出现过的片号集合＝%s，须恰好＝[%d, %d]（本片／它入库的那一片）⇒ 片号方向没被提交说明双向钉住" % (
        _nums, PREVSLICE, SLICE))
for _f in ("am_s7_verify%d.log" % SLICE, "_raw_verify%d.out" % SLICE, "am_s%d_hdr.py" % SLICE,
           "am_s%d_hdr_after.py" % SLICE, "am_s%d_hdr_after_controls.py" % SLICE):
    assert os.path.exists(os.path.join(EVID, _f)), (
        "按片号 %d 拼出的 `%s` 不在盘上 ⇒ 片号指错了对象（回执链／仪器缺一）" % (SLICE, _f))
assert all(str(SLICE) in t for t in TOKENS), (
    "写盘守卫的整词表里有名字不含片号 %d：%s ⇒ 那张表在守别的片" % (SLICE, sorted(t for t in TOKENS if str(SLICE) not in t)))
assert COPY_PREFIX == "zctl_s%d_" % SLICE, "扰动副本前缀＝%s 与片号 %d 拼出的名字不符" % (COPY_PREFIX, SLICE)
assert AFTC_COPY_PREFIX == "zctl_s%d_after_" % SLICE, "只读仪器面的副本前缀与片号 %d 不符：%s" % (
    SLICE, AFTC_COPY_PREFIX)
assert EVID_REL == "docs/evidence/%s" % CDATE[0:10], (
    "证据目录 %s 与 HEAD 提交日期 %s 不同日 ⇒ 本件锚的是上一轮的目录（跨日期目录要从命令现取）" % (EVID_REL, CDATE[0:10]))
for _n in (SP1, SP2, SP3, SP4, SP5, SP6, SP7):
    assert _n.startswith("_raw_s%d_" % SLICE), "现场件 %s 的片号前缀与本片 %d 不符" % (_n, SLICE)
_absent = sorted(t for t in TOKENS if not os.path.exists(os.path.join(EVID, t)))
assert _absent == ["am_s%d_archive_check.py" % SLICE], (
    "整词表里缺席盘上的名字＝%s，须恰好是那件**还没落盘**的归档自查腿 ⇒ 其余名字拼错就会让守卫空转" % (_absent,))
print("   守卫常量回填验：整词表 %d 个名字都带片号 %d、%d 个在盘上（缺席＝归档自查腿 1 件）｜"
      "前缀／现场件名／证据目录同日 -> PASS" % (len(TOKENS), SLICE, len(TOKENS) - len(_absent)))
assert len(HEAD) == 40 and HEAD != PAR40, "HEAD／父哈希形状不对"

_ns = [l for l in sh("git", "show", "--name-status", "--format=", "HEAD").splitlines() if l.strip()]
_r = [l.split("\t") for l in _ns if "\t" in l]
assert len(_r) == len(_ns), "name-status 里有不含制表符的行"
KINDS = [x[0] for x in _r]
CPATHS = [x[-1] for x in _r]          # 路径取**行尾**：状态字母不是路径（那是一条真事故形状）
assert all(len(x) == 2 for x in _r), "name-status 有非两列的行"
assert set(KINDS) <= {"A", "M"}, "HEAD 清单里有 A／M 之外的状态：%s" % sorted(set(KINDS))
NAC, NMC = KINDS.count("A"), KINDS.count("M")
NTOTC = len(CPATHS)
assert len(set(CPATHS)) == NTOTC, "HEAD 清单里有重名路径"
NSR = [l for l in sh("git", "show", "--numstat", "--format=", "HEAD").splitlines() if l.strip()]
assert len(NSR) == NTOTC, "numstat 行数 %d 与 name-status %d 不等 ⇒ 两个口径在数不同的东西" % (len(NSR), NTOTC)
assert all("\t" in l and l.split("\t")[0] != "-" for l in NSR), \
    "numstat 里有 `-`（二进制／rename）⇒ 本件的逐行求和口径不适用，落地段那句要改"
NS_INS = sum(int(l.split("\t")[0]) for l in NSR)
NS_DEL = sum(int(l.split("\t")[1]) for l in NSR)
SS_RAW = sh("git", "diff-tree", "--no-commit-id", "--shortstat", "-r", "HEAD").strip()
# shortstat 的 "1 insertion(+)" 是单数形制：数量词一律用 `files?`／`insertions?`，否则复数形状会把真值数成 0
SS_FILES = int(one(r"(\d+) files? changed", SS_RAW, "shortstat 的文件数"))
SS_INS = int(one(r"(\d[\d,]*) insertions?", SS_RAW, "shortstat 的插入数"))
assert "deletion" not in SS_RAW, "shortstat 里有删除段 ⇒ 零删除那句要改：%s" % SS_RAW
assert (SS_FILES, SS_INS) == (NTOTC, NS_INS), (
    "shortstat (%d, %d) 与 numstat 现算 (%d, %d) 不等 ⇒ 两把尺子对不上" % (SS_FILES, SS_INS, NTOTC, NS_INS))
assert NS_DEL == 0, "numstat 现算删除 %d（须 0）" % NS_DEL
assert ".workbuddy/memory/%s.md" % CDATE[0:10] in CPATHS, \
    "当日纪实不在 HEAD 清单里 ⇒ 上一片写盘的那张面没随提交入库"
print("== 现取：HEAD＝%s（父 %s｜提交时刻 %s）｜入库 A %d＋M %d＝%d 件｜numstat %d/%d 与 shortstat 闭合｜"
      "src+tests 改动 %d｜片号＝%d" % (HEAD7, PAR7, CDATE, NAC, NMC, NTOTC, NS_INS, NS_DEL, NCODE, SLICE))

# ============================ 读数 2：回执正文与盘上字节 ============================
RLOG = "am_s7_verify%d.log" % SLICE
raw_log = open(os.path.join(EVID, RLOG), "rb").read()
r30 = raw_log.decode()
RLINES = len(r30.splitlines())
RAWM = "_raw_verify%d.out" % SLICE
ERRM = "_raw_verify%d.err" % SLICE
rawm = open(os.path.join(EVID, RAWM), encoding="utf-8").read()
ERRMB = os.path.getsize(os.path.join(EVID, ERRM))
assert ERRMB == 0, "核验件的 stderr 不是 0 字节（%d）⇒ 有警告被吞" % ERRMB
GEN = os.path.join(EVID, "am_s%d_hdr.py" % SLICE)
OFF = int(one(r"^BODY_OFFSET=(\d+)$", r30, "本片回执的 BODY_OFFSET"))
_marker = "BODY_OFFSET=%d\n" % OFF
assert r30.count(_marker) == 1, "BODY_OFFSET 那行不是恰好 1 条"
_hp, _body = r30.split(_marker, 1)
assert len((_hp + _marker).encode("utf-8")) == OFF, "BODY_OFFSET 与实际表头字节数不等（定点没收敛）"
BODYB = len(raw_log) - OFF
assert BODYB == os.path.getsize(os.path.join(EVID, RAWM)) == len(rawm.encode("utf-8")), (
    "回执正文区字节数 %d 与原始件 %d 不等" % (BODYB, os.path.getsize(os.path.join(EVID, RAWM))))
assert _body == rawm, "回执正文区与原始件不是逐字节相同 ⇒ 判读行与正文不同源"
HDNL = len([l for l in _hp.splitlines() if l.strip()])
_mh = re.search(r"head=([0-9a-f]{40}) parent=([0-9a-f]{40})", _hp)
assert _mh, "回执表头里取不到 `head=… parent=…` 那两个哈希"
assert _mh.group(1) == HEAD and _mh.group(2) == PAR40, (
    "回执记的 HEAD／父（%s／%s）与 git 现算（%s／%s）不符 ⇒ 落地段说的不是当前这棵树" % (
        _mh.group(1)[:7], _mh.group(2)[:7], HEAD7, PAR7))
FINGER_REC = one(r"^SRC指纹\s*:\s*([0-9a-f]{32})", r30, "回执表头的 SRC 指纹")
FINGER_NOW = sh("bash", "-c", "find src tests -name '*.py' -not -path '*__pycache__*' -print0"
                 " | sort -z | xargs -0 md5sum | md5sum").split()[0]
assert FINGER_REC == FINGER_NOW, ("回执里的 SRC 指纹 %s 与盘上现算 %s 不等 ⇒ 表头与**被测树**不是同一棵"
                                  "（默认 locale 口径，与回执同）" % (FINGER_REC[:12], FINGER_NOW[:12]))
TRUNC = one(r"② truncated=(\w+)", rawm, "② 的 truncated")
TB, BB, BT = [int(x) for x in one(r"tree 条目 (\d+) = blob (\d+) \+ tree (\d+)", rawm, "② 的 tree 分解")]
LB, PB = [int(x) for x in one(r"^② 本地 blob = (\d+)，父提交 = (\d+)", rawm, "② 的本地与父提交 blob")]
RCNT, RNEW, RMODC = [int(x) for x in one(r"^② 本片文件 (\d+) 个 = 新增 (\d+) \+ 修改 (\d+)", rawm,
                                         "② 的本片文件分解")]
CLOS = one(r"^② 闭合式 .*$", rawm, "② 的闭合式整行")
DIFF3 = one(r"^③ 路径差集：.*$", rawm, "③ 的差集整行")
SHA4 = one(r"^④ 全库共有 .*$", rawm, "④ 的全库普查整行")
PER4 = one(r"^④ 本片 .*MATCH \d+/\d+$", rawm, "④ 的本片单列复核整行")
V = one(r"^LANDING_VERDICT = .*$", rawm, "核验件判读行")
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
print("   回执现取：%s＝%d B／%d 行＝表头 %d＋正文 %d（逐字节 == 原始件，表头非空行 %d）｜闭合差＝%d｜"
      "SRC 指纹与盘上现算同值 -> PASS" % (RLOG, len(raw_log), RLINES, OFF, BODYB, HDNL, CLOSEX))

# 顶格字段数：一面从生成器的 FIELDS 声明（ast 现取，不执行它），一面从写盘后只读仪器的 stdout
_fl = []
for _n in ast.walk(ast.parse(open(GEN, encoding="utf-8").read())):
    if isinstance(_n, ast.Assign) and any(getattr(t, "id", None) == "FIELDS" for t in _n.targets):
        _v = ast.literal_eval(_n.value)
        assert isinstance(_v, list) and all(isinstance(x, str) for x in _v), "FIELDS 不是字符串列表"
        _fl.append(_v)
assert len(_fl) == 1, "生成器里 FIELDS 赋值有 %d 处 ⇒ 声明出处不唯一" % len(_fl)
FLDN = len(_fl[0])
lab = "|".join(map(re.escape, _fl[0]))
HDFLD = len([l for l in _hp.splitlines() if re.match(r"^(%s)\s*:" % lab, l)])
assert HDFLD == FLDN, "回执表头按声明标签现数 %d 行 ≠ 生成器 FIELDS 声明 %d 行" % (HDFLD, FLDN)
AFT_OUT = "_raw_s%d_hdr_after.out" % SLICE
aft = open(os.path.join(EVID, AFT_OUT), encoding="utf-8").read()
AFT_AOFF = int(one(r"^BODY_OFFSET（从盘上现读）＝ (\d+)", aft, "只读仪器从盘上现读的偏移"))
assert AFT_AOFF == OFF, "只读仪器现读偏移 %d ≠ 表头定点 %d ⇒ 两把尺子读的是不同的数" % (AFT_AOFF, OFF)
CMP3 = one(r"^cmp -i 三联（本件自己起进程跑的，不是抄生成器的）＝ (\d+ / \d+ / \d+)", aft, "只读仪器的 cmp 三联")
assert CMP3 == "0 / 1 / 1", "cmp 三联不是 0/1/1（本件自己跑的现算值）：%s" % CMP3
AFT_FLD = [int(x) for x in one(r"^顶格字段行数＝ (\d+) / (\d+)", aft, "只读仪器的字段计数")]
assert AFT_FLD == [FLDN, FLDN], "档案字段 %s／声明 %d／只读仪器那对读数不同值" % (AFT_FLD, FLDN)
WC1, WC2, WC3, WFD, WDEC, WBYTES = [x for x in one(
    r"^两出处闭合＝写盘面\((\d+)／(\d+)／(\d+)，字段 (\d+)/(\d+)，(\d+) bytes\) 与本件\(盘上现解\) 同值$",
    aft, "只读仪器与写盘面那两出处")]
assert (int(WC1), int(WC2), int(WC3)) == (0, 1, 1), "写盘面三联不是 0/1/1：%s/%s/%s" % (WC1, WC2, WC3)
WF_OUT = "_raw_s%d_hdr_write.out" % SLICE
wft = open(os.path.join(EVID, WF_OUT), encoding="utf-8").read()
assert int(WBYTES) == len(raw_log) == int(one(r"^写盘完成： \S+ (\d+) bytes", wft, "写盘面报出的字节数")), (
    "写盘面、盘上现读、只读仪器三方字节数不同值")
AFT_NFACES = int(one(r"^输入面 md5 跑前后逐一相同＝True（(\d+) 张面）", aft, "只读仪器读入的面数"))
assert AFT_NFACES == 4, "只读仪器的输入面数＝%d（须 4）⇒ 那张面的形制变了" % AFT_NFACES
HRAV = one(r"^HDR_AFTER_VERDICT = .*$", aft, "只读仪器判读行")
AFTB = os.path.getsize(os.path.join(EVID, AFT_OUT))

# 只读仪器的**复现腿**：本件自己起两次进程、与归档件三方逐字节比（该件是只读件，跑它不动档案）
_tmp = []
try:
    for i in (1, 2):
        p = subprocess.run([PY3, "-B", os.path.join(EVID, "am_s%d_hdr_after.py" % SLICE)],
                           cwd=EVID, capture_output=True, text=True,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), timeout=120)
        assert p.returncode == 0, "只读仪器第 %d 趟复跑 rc=%s ⇒ 本片登记的那条复现断言不成立" % (i, p.returncode)
        q = os.path.join("/tmp", "s31_after_repro%d.out" % i)
        open(q, "w", encoding="utf-8").write(p.stdout)
        _tmp.append(q)
    REPRO = (subprocess.run(["cmp", "-s", _tmp[0], _tmp[1]]).returncode == 0
             and subprocess.run(["cmp", "-s", _tmp[0], os.path.join(EVID, AFT_OUT)]).returncode == 0)
finally:
    for q in _tmp:
        os.path.exists(q) and os.remove(q)
assert REPRO, "只读仪器复跑两趟与归档 stdout 不是三方逐字节相同 ⇒ 判读行不能代表那次运行"
print("   只读仪器（%s）：从盘上现读偏移 %d｜三联 %s｜字段 %d/%d（FIELDS 由 ast 现取自生成器）｜"
      "三方字节数同值 %d｜复跑两趟与归档逐字节相同＝True -> PASS" % (AFT_OUT, OFF, CMP3, FLDN, FLDN, len(raw_log)))

# ============================ 读数 3：三面控件面的真跑 stdout 与源码 ast 闭合 ============================
CTL = [
    ("MSG", "提交说明面", "am_s%d_msg_controls.py" % SLICE, "_raw_s%d_msg_controls.out" % SLICE,
     "_raw_s%d_msg_controls.err" % SLICE,
     r"^CASES 声明＝(\d+) 条（全部红路，其中克隆别的仪器 (\d+) 条）｜保持绿＝(\d+) 条｜"
     r"本件自己的锚点普查＝(\d+) 条｜超时（环境故障，不判红也不判过）＝(\d+) 条｜因超时没跑到的＝(\d+) 条$",
     {"DEC": 0, "CLONE": 1, "GRN": 2, "SELF": 3, "TO": 4}, "CASES", "GREENS", "UNCOVERED", None),
    ("HDR", "表头面", "am_s%d_hdr_controls.py" % SLICE, "_raw_s%d_hdr_controls.out" % SLICE,
     "_raw_s%d_hdr_controls.err" % SLICE,
     r"^红路声明＝(\d+) 条｜保持绿＝(\d+) 条（判据 rc=0 且打印出预检标记那行）｜基线＝(\d+) 条"
     r"（不进表，红在开头即整面作废）｜环境故障（不判红也不判过）＝(\d+) 条（按去向细分：无）｜本面自查＝(\d+) 条",
     {"DEC": 0, "GRN": 1, "BASE": 2, "TO": 3, "SELF": 4}, "CASES", "GREENS", "UNCOVERED", "keep"),
    ("AFTC", "只读仪器面", "am_s%d_hdr_after_controls.py" % SLICE,
     "_raw_s%d_hdr_after_controls.out" % SLICE, "_raw_s%d_hdr_after_controls.err" % SLICE,
     r"^红路声明＝(\d+) 条｜保持绿＝(\d+) 条（逐条 rc=0 且打印出判读行；含裸跑基线 1 条）｜实跑行数＝(\d+) 条｜"
     r"超时（环境故障，不判红也不判过）＝(\d+) 条（无）｜装配失败（锚点问题，不是红路）＝(\d+) 条（无）｜"
     r"造不出红路的闸门＝(\d+) 条$",
     {"DEC": 0, "GRN": 1, "ROWS": 2, "TO": 3, "ASM": 4}, "CASES", "GREENS", "UNCOVERED", "plus1"),
]
for pfx, face, script, outf, errf, dec_pat, slot, LN, GN, UN, kind in CTL:
    op, ep, sp = os.path.join(EVID, outf), os.path.join(EVID, errf), os.path.join(EVID, script)
    for p in (sp, op, ep):
        assert os.path.exists(p), "%s 缺件：%s" % (face, p)
    txt = open(op, encoding="utf-8").read()
    assert os.path.getsize(ep) == 0, "%s 的 stderr 不是 0 字节（有警告被吞）" % face
    g = one(dec_pat, txt, face + " 的声明行")
    for key, idx in slot.items():
        CTX["%s_%s" % (pfx, key)] = g[idx]
    CTX[pfx + "_SCRIPT"] = script
    CTX[pfx + "_OUT"] = outf
    CTX[pfx + "_OUTB"] = str(os.path.getsize(op))
    CTX[pfx + "_OUTM"] = md5(op)[:12]
    CTX[pfx + "_ERRB"] = str(os.path.getsize(ep))
    if "SELF" not in slot:  # 只读仪器面把那枚数写在汇总行之外的一行里
        CTX[pfx + "_SELF"] = one(r"本面自查＝(\d+) 条", txt, face + " 的本面自查条数")
    CTX[pfx + "_UNC"] = one(r"本轮造不出红路的闸门＝(\d+) 条", txt, face + " 的造不出红路数")
    ver = last(r"^(CONTROL_VERDICT = .*)$", txt, face + " 的判读行")
    assert ver.endswith("ALL CONTROLS CAN GO RED"), "%s 的判读行不是全过声明：%s" % (face, ver)
    CTX[pfx + "_VERDICT"] = ver
    assert len(rows(txt, r"^(× |！)")) == 0, "%s 的归档 stdout 里有未命中／环境行 ⇒ 该批读数不可引用" % face
    # ⑤ 归档件与源码同源：ast 现取的三条表长与 stdout 自己声明的条数逐面相等
    L = list_lens(sp)
    assert int(CTX[pfx + "_DEC"]) == L[LN], "%s：stdout 声明 %s 条 ≠ 源码 CASES %d 条 ⇒ 归档件比源码旧" % (
        face, CTX[pfx + "_DEC"], L[LN])
    assert int(CTX[pfx + "_GRN"]) == L[GN] + (1 if kind == "plus1" else 0), (
        "%s：stdout 保持绿 %s 条 ≠ 源码 GREENS %d%s ⇒ 归档件比源码旧" % (
            face, CTX[pfx + "_GRN"], L[GN], "＋基线 1" if kind == "plus1" else ""))
    assert int(CTX[pfx + "_UNC"]) == L[UN], "%s：stdout 造不出红路 %s 条 ≠ 源码 UNCOVERED %d 条" % (
        face, CTX[pfx + "_UNC"], L[UN])
    if kind == "plus1":
        assert int(CTX[pfx + "_ROWS"]) == L[LN] + L[GN] + 1, "%s：实跑 %s 行 ≠ 红 %d＋绿 %d＋基线 1" % (
            face, CTX[pfx + "_ROWS"], L[LN], L[GN])
    print("   %s：%s %s B（md5 %s／stderr %s）｜声明 %s／保持绿 %s／自查 %s／造不出红路 %s｜"
          "ast 闭合＝CASES %d·GREENS %d·UNCOVERED %d -> PASS" % (
              face, outf, CTX[pfx + "_OUTB"], CTX[pfx + "_OUTM"], CTX[pfx + "_ERRB"],
              CTX[pfx + "_DEC"], CTX[pfx + "_GRN"], CTX[pfx + "_SELF"], CTX[pfx + "_UNC"],
              L[LN], L[GN], L[UN]))
# 表头面的配额代价（等待次数／秒数是那一批的既定事实；余量两个数易失，只当现取打印、不进档案面）
hdr_txt = open(os.path.join(EVID, "_raw_s%d_hdr_controls.out" % SLICE), encoding="utf-8").read()
_HLIM = one(r"^配额闸门起点＝/rate_limit 现取余量（易失读数，不作下一轮分母）＝(\d+) 次｜单窗口上限＝(\d+) 次｜"
            r"本批子运行＝(\d+) 条", hdr_txt, "表头面的配额起点行")
CTX["HDR_WAITN"], CTX["HDR_WAITS"] = one(r"^配额闸门收面＝现取余量（易失，不作分母）＝\d+ 次｜单窗口上限（现取）＝\d+ 次｜"
                                         r"等待触发 (\d+) 次｜等待累计 (\d+) 秒$", hdr_txt, "表头面的配额收面行")
CTX["HDR_BATCH"] = _HLIM[2]
assert int(CTX["HDR_BATCH"]) == int(CTX["HDR_DEC"]) + int(CTX["HDR_GRN"]), (
    "表头面声明的基线外整批 %s 条 ≠ 红 %s＋绿 %s（那 1 条未扰动基线在配额闸门之前跑，不占本批）" % (
        CTX["HDR_BATCH"], CTX["HDR_DEC"], CTX["HDR_GRN"]))
assert CTX["MSG_TO"] == CTX["HDR_TO"] == CTX["AFTC_TO"] == CTX["AFTC_ASM"] == "0", (
    "三面里出现超时／环境／装配失败读数（%s／%s／%s／%s）⇒ 那一批不能按全过引用" % (
        CTX["MSG_TO"], CTX["HDR_TO"], CTX["AFTC_TO"], CTX["AFTC_ASM"]))
print("   配额与超时：表头面基线外整批 %s 条（跨窗口）｜等待 %s 次、累计 %s 秒（墙钟只作可行性代价）｜"
      "三面超时／环境／装配失败＝0/0/0 -> PASS" % (CTX["HDR_BATCH"], CTX["HDR_WAITN"], CTX["HDR_WAITS"]))
# 三面公共字段名的同形制普查（本件的读者按名取数，形制漂了就取不到 ⇒ 这里当场数一遍给散文作证）。
# 数的是**带赋值形的字段名所在行**：绿路逐条读数里也复述「保持绿」这个词，按出现次数数会得到 1/6/8 这种假红。
for key in ("保持绿＝", "本轮造不出红路的闸门＝", "CONTROL_VERDICT ="):
    per = [len([l for l in open(os.path.join(EVID, f), encoding="utf-8").read().splitlines() if key in l])
           for _p, _f, _s, f, _e, _d, _sl, _a, _b, _c, _k in CTL]
    assert per == [1, 1, 1], "字段名「%s」在三面归档 stdout 里按行现数 %s（须每面恰好 1 行）⇒ 形制漂了，读者按名取数会漏" % (
        key, per)
print("   三面同形制：赋值形的「保持绿＝」「本轮造不出红路的闸门＝」「CONTROL_VERDICT =」各按面现数 1/1/1 行 -> PASS")

# ============================ 读数 4：六件红灯现场与探针（每件都从该件现取） ============================
def specimen(alias, name, why_pat=None, miss=False, verdict=False):
    """一件红灯现场的读数按调用方给的 `SPn` 别名进 CTX——键名绝不由件名反推（反推会造出一批没人读的键）。"""
    op, ep = os.path.join(EVID, name + ".out"), os.path.join(EVID, name + ".err")
    for p in (op, ep):
        assert os.path.exists(p), "现场件不在档：%s" % p
    ot, et = open(op, encoding="utf-8").read(), open(ep, encoding="utf-8").read()
    CTX[alias] = name
    CTX[alias + "_B"] = str(os.path.getsize(op))
    CTX[alias + "_E"] = str(os.path.getsize(ep))
    if why_pat:
        r = raised(et if et else ot, name)
        assert re.search(why_pat, r), "现场件 %s 的抛出行不含预期形状（须/%s/），实得：%s" % (name, why_pat, r[:120])
        CTX[alias + "_WHY"] = r
    if miss:
        CTX[alias + "_MISS"] = str(len(rows(ot, r"^× ")))
    if verdict:
        CTX[alias + "_VERDICT"] = one(r"^(CONTROL_VERDICT = .*)$", ot, name + " 的整面判读")
    return ot, et


specimen("SP1", SP1, why_pat=r"基线标记句在副本 stdout 里命中")
specimen("SP2", SP2, miss=True, verdict=True)
assert int(CTX["SP2_MISS"]) == 12, "现场 %s 的「× 」行现数 %s 条（登记句写的是 12 条）⇒ 那句得按现数改" % (
    SP2, CTX["SP2_MISS"])
_sp3o, _sp3e = specimen("SP3", SP3, why_pat=r"读不到 GitHub 配额响应头")
CTX["SP3_ROWS"] = str(len(rows(_sp3o, r"^  [A-Z][A-Z]?\d+ ")))
_sp4o, _sp4e = specimen("SP4", SP4, miss=True, verdict=True)
_sp4rows = rows(_sp4o, r"^× ")
CTX["SP4_RAISES"] = "／".join(one(r"｜实得抛出行：(.*)$", l, "现场 %s 的一条实得抛出行" % SP4) for l in _sp4rows)
assert len(_sp4rows) == 2, "现场 %s 的「× 」行现数 %d（须 2＝F1 与 R8 那两条）" % (SP4, len(_sp4rows))
_sp5o, _sp5e = specimen("SP5", SP5, why_pat=r"复跑 stdout 与归档正文不同")
assert os.path.getsize(os.path.join(EVID, SP5 + ".out")) == 0, "探针 %s 的 stdout 不是 0 B ⇒ 那句「未走到打印」是假话" % SP5
# 探针那句要**逐字符**等于生成器里那条 REPRO 断言的消息，否则探针量的不是那把尺子
_gsrc = open(GEN, encoding="utf-8").read()
_mrepro = one(r'^assert REPRO, "(.*)"', _gsrc, "生成器那条复现断言")
assert "AssertionError: " + _mrepro == CTX["SP5_WHY"], "探针抛出行与生成器那条断言不逐字符相等：%s" % CTX["SP5_WHY"]
CTX["SP5_LINE"] = str([n.lineno for n in ast.walk(ast.parse(_gsrc))
                       if isinstance(n, ast.Assert) and isinstance(n.test, ast.Name)
                       and n.test.id == "REPRO"][0])
_sp6o, _sp6e = specimen("SP6", SP6, why_pat=r"UnboundLocalError")
_sp6red, _sp6grn = [int(x) for x in one(r"^纸面预检.*?（红路 (\d+) 条＋绿路 (\d+) 条", _sp6o, SP6 + " 首行声明")]
_sp6pr = len(rows(_sp6o, r"^  [AG]\d+ "))
CTX["SP6_LOST"] = str(_sp6red + _sp6grn - _sp6pr)
CTX["SP6_PR"] = str(_sp6pr)
CTX["SP6_DECL"] = "%d＋%d" % (_sp6red, _sp6grn)
assert _sp6red == int(CTX["AFTC_DEC"]), "现场 %s 声明的红路 %d 条 ≠ 终稿 %s 条 ⇒ 崩掉的那一批不是这一轮的" % (
    SP6, _sp6red, CTX["AFTC_DEC"])
_sp7o, _sp7e = specimen("SP7", SP7, why_pat=r"写盘面报出")
_w0, _w1 = [int(x) for x in one(r"^AssertionError: 写盘面报出 (\d+) 字节 ≠ 盘上现有 (\d+) 字节", _sp7o, SP7 + " 那句抛出")]
assert _w0 == len(raw_log) and _w1 == len(raw_log) + 2, (
    "探针 %s 的两枚字节数（%d／%d）与盘上现读 %d 不成「原值／+2」⇒ 那句改两字的描述不对了" % (SP7, _w0, _w1, len(raw_log)))
CTX["SP7_OFF"] = one(r"^BODY_OFFSET（从盘上现读）＝ (\d+)", _sp7o, SP7 + " 探针里的偏移")
CTX["SP7_CMP"] = one(r"^cmp -i 三联（本件自己起进程跑的，不是抄生成器的）＝ (\d+ / \d+ / \d+)", _sp7o, SP7 + " 探针里的三联")
CTX["SP7_FLD"] = " / ".join(one(r"^顶格字段行数＝ (\d+) / (\d+)", _sp7o, SP7 + " 探针里的字段计数"))
CTX["SP7_RC"] = one(r"^子进程 rc＝ (\d+)$", _sp7o, SP7 + " 的子进程 rc")
CTX["SP7_CLEAN"] = one(r"^探针目录已删＝ (True|False)$", _sp7o, SP7 + " 的清理行")
assert int(CTX["SP7_OFF"]) == OFF + 2 and CTX["SP7_CMP"] == CMP3 and CTX["SP7_FLD"] == "%d / %d" % (FLDN, FLDN), (
    "探针 %s 的形状变了（偏移 %s／三联 %s／字段 %s）⇒ 本节那句「自洽门全绿、只有跨闭合抓得住」要重写" % (
        SP7, CTX["SP7_OFF"], CTX["SP7_CMP"], CTX["SP7_FLD"]))
assert CTX["SP7_CLEAN"] == "True", "探针目录没自证已删 ⇒ 该件的临时面可能还在档案目录里"
print("   六件现场：SP1 %s／%s B｜SP2 %s B×%s 行未命中｜SP3 %s B（打印 %s 行）｜SP4 %s B×%s 行｜"
      "SP5 %s B（生成器第 %s 行那条断言逐字符）｜SP6 %s B（首行 %s、已打印 %s 行、作废 %s 条）｜"
      "SP7 %s B（偏移 %s＝表头＋2、三联仍 %s、字段仍 %s）" % (
          CTX["SP1_B"], CTX["SP1_E"], CTX["SP2_B"], CTX["SP2_MISS"], CTX["SP3_B"], CTX["SP3_ROWS"],
          CTX["SP4_B"], CTX["SP4_MISS"], CTX["SP5_B"], CTX["SP5_LINE"], CTX["SP6_B"], CTX["SP6_DECL"],
          CTX["SP6_PR"], CTX["SP6_LOST"], CTX["SP7_B"], CTX["SP7_OFF"], CTX["SP7_CMP"], CTX["SP7_FLD"]))
# 抛出串要进具名替换的模板，而模板里的裸百分号会被 `named()` 当场拒；值不参与替换，所以那里只查模板一侧。

# ============================ 读数 5：待入库清单（易失，只进散文） ============================
st = [l.rstrip("\n") for l in sh("git", "-c", "core.quotePath=false", "status", "--porcelain",
                                 "-uall").splitlines() if l.strip()]
assert not [l for l in st if l[3:].endswith("/")], "清单里有以 `/` 结尾的条目（整个目录被折成一行）⇒ 计数口径破了"
_class = {}
for l in st:
    x, y, path = l[0], l[1], l[3:]
    _class.setdefault(path, []).append("UNT" if (x, y) == ("?", "?") else
                                       "NEW" if x == "A" else "MOD" if y == "M" or x == "M" else "?")
assert not [p for p, c in _class.items() if len(set(c)) != 1 or "?" in c], "有路径落进 0 类或 ≥2 类 ⇒ 状态解析与清单不同源"
UNT = sorted(p for p, c in _class.items() if c[0] == "UNT")
NEW = sorted(p for p, c in _class.items() if c[0] == "NEW")
MODS = sorted(p for p, c in _class.items() if c[0] == "MOD")
assert len(_class) == len(st), "清单行数 %d 与去重路径数 %d 不等 ⇒ 有重行" % (len(st), len(_class))
NU, NNEW, NMOD = len(UNT), len(NEW), len(MODS)
PEND = NU + NNEW + NMOD
assert not UNT, "还有未跟踪件 %s ⇒ 本轮有仪器落盘后没先 `git add`，与登记的前提不符" % UNT[:3]
ROLES = [
    ("回执链", r"^am_s7_verify%d\.(py|log)$|^_raw_verify%d\.(out|err)$|^am_s%d_hdr\.py$" % (SLICE, SLICE, SLICE)),
    ("表头预检与写盘两面（含控件面判绿后的复跑预检）",
     r"^_raw_s%d_hdr_(precheck|write|precheck_rerun_after_controls_green)\.(out|err)$" % SLICE),
    ("表头控件面", r"^am_s%d_hdr_controls\.py$|^_raw_s%d_hdr_controls\.(out|err)$" % (SLICE, SLICE)),
    ("表头控件面的红灯现场与 403 探针",
     r"^_raw_s%d_hdr_controls\.(attempt[0-9]+-[a-z0-9-]+|probe-ruler-403)\.(out|err)$" % SLICE),
    ("只读仪器与其真跑两面", r"^am_s%d_hdr_after\.py$|^_raw_s%d_hdr_after\.(out|err)$" % (SLICE, SLICE)),
    ("只读仪器的控件面", r"^am_s%d_hdr_after_controls\.py$|^_raw_s%d_hdr_after_controls\.(out|err)$"
     % (SLICE, SLICE)),
    ("只读仪器控件面的红灯现场与等长探针",
     r"^_raw_s%d_hdr_after_controls\.(attempt[0-9]+-[a-z0-9-]+|probe-headlen[a-z0-9-]*)\.(out|err)$" % SLICE),
    ("落地腿（本件与其预检／写盘两面）",
     r"^am_s%d_docs_landing\.py$|^_raw_s%d_landing_(precheck|write)\.(out|err)$" % (SLICE, SLICE)),
]
buckets, assigned = [], []
for name, pat in ROLES:
    hit = [p for p in UNT + NEW if re.match(pat, os.path.basename(p))]
    assert hit, "%s 一组现数为 0 ⇒ 命名规则与本目录脱节（分组句就成了假话）" % name
    buckets.append((name, len(hit)))
    assigned += hit
assert len(assigned) == len(set(assigned)) == NNEW + NU, (
    "角色分组没恰好覆盖待入库清单（分组 %d／去重 %d／现数 %d）⇒ 有件落进 0 组或 ≥2 组" % (
        len(assigned), len(set(assigned)), NNEW + NU))
BKT_TXT = "／".join("%s **%d**" % (n, c) for n, c in buckets)
BKT_SUM = "＋".join(str(c) for _, c in buckets) + "＝%d" % (NNEW + NU)
MOD_TXT = "／".join(os.path.basename(p) for p in MODS) if MODS \
    else "（落笔时 %d 张——五张档案面刚随本片提交入库；本件写盘后立刻变成 %d 张）" % (NMOD, NMOD + 5)
print("   待入库现数（易失，不作下一轮分母）：未跟踪 %d＋已暂存新增 %d＋修改 %d＝%d 处｜分组和＝%s" % (
    NU, NNEW, NMOD, PEND, BKT_SUM))


def bucket_control():
    """分组的正对照：把回执链那组的桶形改窄到认不出表头生成器，它必须**恰好少认领 1 件**。
    不少（＝改坏了还认同样多）或认 0 件（＝整组散架，等价于另一条闸门）都说明这根针没扎在要害上。"""
    name, c0 = [b for b in buckets if b[0].startswith("回执链")][0]
    pat = [p for n, p in ROLES if n == name][0]
    bad = pat.replace(r"^am_s%d_hdr\.py$" % SLICE, r"^am_s%d_hdr_NONEXIST\.py$" % SLICE)
    assert bad != pat, "桶形替换没改变字符串 ⇒ 这根针根本没扎进去"
    hit = [p for p in UNT + NEW if re.match(bad, os.path.basename(p))]
    assert len(hit) == c0 - 1, "改窄之后那组认领 %d 件（原形制 %d，须恰好少 1）" % (len(hit), c0)
    return "%s 组由 %d 件降到 %d 件（改窄 `am_s%d_hdr\\.py` 一支）" % (name, c0, len(hit), SLICE)


print("   分组断言的正对照：%s -> PASS" % bucket_control())
_res = [p for p in os.listdir(EVID) if p.startswith(COPY_PREFIX) or p.startswith(AFTC_COPY_PREFIX)
        or p.startswith("zprobe_")]
assert not _res, "档案目录里还有扰动副本的残留：%s ⇒ 上一条「复跑逐字节相同」的读数不干净" % _res[:3]
print("   副本残留普查：档案目录里 `zctl_`／`zprobe_` 前缀的残留 0 件 -> PASS")

# ============================ 读数 6：GPU 红线（NVML 那把尺子坏了，替代断言＋一真两合成对照） ============================
INTERP = "/tmp/amvenv/bin/python"
_nv = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                     capture_output=True, text=True)
CTX["NVRC"] = str(_nv.returncode)
NVMSG = (_nv.stderr or _nv.stdout).splitlines()[0].strip() if (_nv.stderr or _nv.stdout).strip() else "（无输出）"
NVKERN = open("/sys/module/nvidia/version").read().strip()
_mlib = re.search(r"NVML library version: ([\d.]+)", _nv.stderr + _nv.stdout)
CTX["NVLIB"] = _mlib.group(1) if _mlib else "取不到"


def argv0(pid):
    try:
        parts = open("/proc/%s/cmdline" % pid, "rb").read().split(b"\0")
    except OSError:
        return None
    return parts[0].decode("utf-8", "replace") if parts and parts[0] else None


def match_argv0(blob):
    """替代口径：整词比 argv[0]，不比命令行子串（子串会把承载这条命令的那个 shell 自己也认成我的进程）。"""
    return blob.split(b"\0")[0].decode("utf-8", "replace") == INTERP


# 分母排除自身与父进程：本件正是用 INTERP 跑起来的，不排除＝把审计者数进被审计的清单（上一片那件同口径）
PIDS = [d for d in os.listdir("/proc") if d.isdigit() and d not in (str(os.getpid()), str(os.getppid()))]
MINE_NOW = [d for d in PIDS if argv0(d) == INTERP]
DENOM = len(PIDS)
# 三条对照：① 真 /proc 条目——本件自己的 argv[0] 必须就是 INTERP（否则下面那个 0 是读错了对象）；
# ② 合成 argv[0]＝INTERP 必须数出 1；③ argv[0]＝/bin/bash、参数里带着 INTERP 必须数出 0（松散口径会误指）
SELF_A0 = argv0(str(os.getpid()))
assert SELF_A0 == INTERP, "本件自己的 argv[0]＝%s 不是 %s ⇒ 这条替代断言数的不是我的解释器" % (SELF_A0, INTERP)
assert match_argv0((INTERP + "\0x.py").encode()) is True, "argv[0] 匹配式的正对照自己失效了 ⇒ 上面那个 0 是空转，不可信"
assert match_argv0(b"/bin/bash -c grep " + INTERP.encode() + b"\0") is False, \
    "argv[0] 匹配式把子串也认成命中 ⇒ 它和松散匹配一样会误指自己"
print("   GPU：NVML 复测 rc＝%s（首行「%s」）｜内核模块 %s／用户态库 %s｜"
      "替代断言＝argv[0] 整词匹配我的解释器，现数我的 %d 个（分母 %d 个 PID，已排除自身与父进程，只进散文）｜"
      "真条目对照（自己的 argv[0]）＝命中／合成对照＝1／0 -> PASS" % (
          CTX["NVRC"], NVMSG, NVKERN, CTX["NVLIB"], len(MINE_NOW), DENOM))
assert len(MINE_NOW) == 0, "机上还留着钉住的那个解释器在跑 %s ⇒ 归还口径破了" % MINE_NOW
CTX["NGPU0"] = str(len(MINE_NOW))
CTX["DENOM"] = str(DENOM)
CTX["NVMSG"] = NVMSG
CTX["NVKERN"] = NVKERN

# ============================ 节号与承接锚句：从档案现取，不写死 ============================
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


def top_bullets(seg):
    ms = list(re.finditer(r"^- \*\*(.+?)\*\*[^\n]*$", seg, re.M))
    out = []
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(seg)
        out.append((m.group(1), seg[m.start():end]))
    return out


_now = top_bullets(SEC_BODY)
for anchor in CARRIED:
    n = len([1 for _l, b in _now if anchor in b])
    assert n == 1, "承接锚句「%s」在 §26.%s 的顶级子条里命中 %d 条（须恰好 1，那是下游 `bullet_for` 的问法）" % (
        anchor, SECREF, n)
assert len([1 for _l, b in _now if LESSON_NEW in b]) == 0, (
    "本轮新学锚句「%s」在落笔前的 §26.%s 里已命中 ⇒ 它不是本轮新学，那句「本轮抓到」成了假话" % (LESSON_NEW, SECREF))
print("   节号 §26.%s 与补记序号 §W 补记 %s 均从档案现取｜承接锚句 %s 各恰好 1 条顶级子条、新学锚句现数 0 -> PASS" % (
    SECREF, PNUM, "／".join(CARRIED)))

CTX.update({
    "SLICE": str(SLICE), "PREVSLICE": str(PREVSLICE), "NEXTSLICE": str(NEXTSLICE),
    "HEAD40": HEAD, "HEAD7": HEAD7, "PAR40": PAR40, "PAR7": PAR7, "CDATE": CDATE,
    "TIME": TIME, "NCODE": str(NCODE), "NAC": str(NAC), "NMC": str(NMC), "NTOTC": str(NTOTC),
    "SSRAW": SS_RAW, "NSINS": str(NS_INS), "NSDEL": str(NS_DEL),
    "R31LOG": RLOG, "R31B": str(len(raw_log)), "R31L": str(RLINES), "OFF31": str(OFF),
    "BODYB": str(BODYB), "HDNL": str(HDNL), "ERRMB": str(ERRMB), "CMP3": CMP3, "FLDN": str(FLDN),
    "HRAV": HRAV, "HB": str(AFTB), "RAWM": RAWM, "WFOUT": WF_OUT, "NFACES": str(AFT_NFACES),
    "REPRO": "True", "TRUNC": TRUNC, "TB": str(TB), "BB": str(BB), "LB": str(LB), "BT": str(BT),
    "PB": str(PB), "CLOS": CLOS, "DIFF3": DIFF3, "SHA4": SHA4, "PER4": PER4, "V": V,
    "CA": CA, "CB": CB, "CC": CC, "CD": CD, "CLOSEX": str(CLOSEX), "FINGER": FINGER_REC[:12],
    "SECREF": SECREF, "PNUM": PNUM, "LESSON_NEW": LESSON_NEW, "EVIDREL": EVID_REL,
    "NU": str(NU), "NNEW": str(NNEW), "NMOD": str(NMOD), "TOT0": str(PEND), "BKT": BKT_TXT,
    "BKTSUM": BKT_SUM, "MODTXT": MOD_TXT,
    "ASTCLOSE": "／".join("%s＝%s" % (p, "OK") for p, _f, _s, _o, _e, _d, _sl, _a, _b, _c, _k in CTL),
})

for _k, _v in sorted(CTX.items()):
    assert isinstance(_v, str) and _v != "" and "\n" not in _v, "%s 不是单行非空字符串：%r" % (_k, _v[:40])
print("   CTX %d 个键全部是非空单行字符串（值从命令现取，没有一个是我手敲的）-> PASS" % len(CTX))

# ============================ 五张面的模板 ============================
DEV_T = r"""- **分片 %(SLICE)s 落地腿的真读数（%(TIME)s 现取；把上面那条计划子条里说「此刻还不存在」的那些空格逐格填上，每个数都由命令现取；本条不新增那一类以「当前」开头的钉住口径引句，也不复述上一片回执名）**：HEAD＝`%(HEAD40)s`（父 `%(PAR40)s`，短哈希 `%(HEAD7)s`／`%(PAR7)s`，提交时刻 `%(CDATE)s` 由 `git log -1 --format=%%cd --date=iso` 现取），推送区间 `%(PAR7)s..%(HEAD7)s`；入库面＝新增 (A) **%(NAC)s**＋修改 (M) **%(NMC)s**＝**%(NTOTC)s** 件；`git diff-tree --no-commit-id --shortstat -r HEAD` 原文＝`%(SSRAW)s`（**没有删除段**），插入数按 `--numstat` 逐行现算＝**%(NSINS)s**、删除＝**%(NSDEL)s**，两把尺子当场对账 ⇒ 本片是纯归档腿且零删除；生产树 `src`＋`tests` 改动现数 **%(NCODE)s** 件。
  - **落地证明五步＋四个正对照（逐字取自本目录的 `%(RAWM)s`，其 stderr **%(ERRMB)s** B；本条不许自己造数）**：① `ls-remote` refs/heads/main == 本地 HEAD -> MATCH；② `truncated=%(TRUNC)s`，tree 条目 **%(TB)s**＝blob **%(BB)s**＋tree **%(BT)s**，本地 blob＝%(LB)s、父提交 blob＝%(PB)s；③ 双向路径差集（两侧同用 `core.quotePath=false` 与 `LC_ALL=C sort`）＝`%(DIFF3)s`；④ `%(SHA4)s`；`%(PER4)s`；⑤ `%(CLOS)s`，四个正对照各半句＝`%(CA)s`／`%(CB)s`／`%(CC)s`／`%(CD)s`；判读行 `%(V)s`。闭合式右边那两个自变量另由 git 现取并独立复算：%(BB)s − (%(PB)s ＋ %(NAC)s) ＝ **%(CLOSEX)s**（须 0），正文里远端与本地两个 blob 也当场相等（%(BB)s＝%(LB)s）；回执表头里那枚 `head=`／`parent=` 与 git 现算的两个 40 位哈希逐字符相同 ⇒ 归档讲的就是当前这棵树；`SRC指纹` 那枚前 12 位 `%(FINGER)s` 与**默认 locale** 下的盘上现算逐字符相同。
  - **回执表头仍由生成器产出（不许手工组装）**：档案名＝`%(R31LOG)s` **%(R31B)s** B／**%(R31L)s** 行＝表头 **%(OFF31)s**＋正文 **%(BODYB)s**（表头里非空行 **%(HDNL)s** 行，与顶格字段 **%(FLDN)s** 行不是同一个量）；`BODY_OFFSET=%(OFF31)s` 由写入件重解一遍并核对到正文区与原始件**逐字节相同**；写盘后另起的那件**只读仪器**（stdout＝`_raw_s%(SLICE)s_hdr_after.out` **%(HB)s** B）在盘上把同一个偏移**读回来**（不是自己迭代出来的），自己起进程跑 `cmp -i %(OFF31)s:0`，三联＝`%(CMP3)s`，顶格字段 **%(FLDN)s/%(FLDN)s** 与生成器 `FIELDS` 声明互证（ast 现取，不执行生成器），写盘面那四联读数与盘上现解同值（`%(WFOUT)s` 报出的字节数、盘上现读、只读仪器三方同值＝%(R31B)s B），输入面 md5 跑前后逐一相同＝True（%(NFACES)s 张面），判读行 `%(HRAV)s`。**本件又自己起了两趟只读仪器复跑，与归档 stdout 三方逐字节相同＝%(REPRO)s**（复跑件不落仓库，跑完即删）。
  - **三面干跑控件面的真跑读数（判红口径继承上一片：rc≠0 且**抛出的那一行**含指定闸门句，且 traceback 里指向该副本的行号落在该句在**副本源码**里的 ±3 行窗口；超时／环境故障／装配失败既不判红也不判过、单独计数）**：%(MSG_SCRIPT)s 的 stdout＝%(MSG_OUT)s **%(MSG_OUTB)s** B（md5 前 12 `%(MSG_OUTM)s`／stderr %(MSG_ERRB)s B）｜CASES 声明 **%(MSG_DEC)s** 条（全部红路，其中克隆别的仪器 %(MSG_CLONE)s 条）｜保持绿 **%(MSG_GRN)s** 条｜超时 **%(MSG_TO)s** 条｜本面自查 **%(MSG_SELF)s** 条｜造不出红路 **%(MSG_UNC)s** 条 ⇒「%(MSG_VERDICT)s」；%(HDR_SCRIPT)s 的 stdout＝%(HDR_OUT)s **%(HDR_OUTB)s** B（md5 前 12 `%(HDR_OUTM)s`／stderr %(HDR_ERRB)s B）｜红路声明 **%(HDR_DEC)s** 条｜保持绿 **%(HDR_GRN)s** 条｜基线 **%(HDR_BASE)s** 条（未扰动副本，在配额闸门之前跑，不占本批）｜基线外整批 **%(HDR_BATCH)s** 条｜环境故障 **%(HDR_TO)s** 条｜本面自查 **%(HDR_SELF)s** 条｜造不出红路 **%(HDR_UNC)s** 条 ⇒「%(HDR_VERDICT)s」，该面每条扰动都要复跑一次联网尺子 ⇒ 跨配额窗口等待 **%(HDR_WAITN)s** 次、累计 **%(HDR_WAITS)s** 秒（墙钟只作可行性代价登记；配额余量那两个数易失，不写进档案面）；%(AFTC_SCRIPT)s 的 stdout＝%(AFTC_OUT)s **%(AFTC_OUTB)s** B（md5 前 12 `%(AFTC_OUTM)s`／stderr %(AFTC_ERRB)s B）｜红路声明 **%(AFTC_DEC)s** 条｜保持绿 **%(AFTC_GRN)s** 条（含裸跑基线 1 条）｜实跑行数 **%(AFTC_ROWS)s** 条｜超时 **%(AFTC_TO)s** 条｜装配失败 **%(AFTC_ASM)s** 条｜本面自查 **%(AFTC_SELF)s** 条｜造不出红路 **%(AFTC_UNC)s** 条 ⇒「%(AFTC_VERDICT)s」（该面全程零 API、零配额）。
  - **归档件与源码同源的 ast 闭合（⑤那条口径的仪器化＝本轮自己被咬过的那一口）**：三面源码里 `len(CASES)`／`len(GREENS)`／`len(UNCOVERED)` 由 ast 现取（**不执行任何控件面**），与该面归档 stdout 自己声明的条数逐面相等＝%(ASTCLOSE)s；只读仪器面另合一条 `实跑行数＝红路＋绿路＋基线`＝%(AFTC_ROWS)s。**本轮真中一次**：那面的汇总行改成与上一片同族形制之后，归档 stdout 比源码旧，而条数不会自己说破——所以这三对闭合不是装饰，是它把「编辑过已暂存的仪器却没复跑控件面」这类状态从**看不见的漂移**变成**当场红**。同族形制这一条也现数过：三面归档 stdout 里「保持绿」「本轮造不出红路的闸门」「CONTROL_VERDICT」各按面恰好 1 **行**（＝1/1/1；查的是**带赋值形的字段名**所在行——绿路的逐条读数里也在复述「保持绿」这个词，按出现次数数就会把那些复述一起数进来而成假红），而「超时」那枚在表头面叫「环境故障（不判红也不判过）」——这一处不同名正是本件按面各配一条正则的理由。
  - **本轮六处红灯现场与探针（都按别名留在 `%(EVIDREL)s`，抛出行逐字从该件现取，不是转述）**：⑴ `%(SP1)s` stdout **%(SP1_B)s** B／stderr **%(SP1_E)s** B＝「%(SP1_WHY)s」——提交说明面第一版的基线标记句在副本 stdout 里命中 2 行，判红谓词的「该句恰好一行」先响，那条对照根本没跑到自己该跑的闸门。⑵ `%(SP2)s` stdout **%(SP2_B)s** B／stderr **%(SP2_E)s** B，表里「× 」行 **%(SP2_MISS)s** 条、整面判读「%(SP2_VERDICT)s」——判据空转第三次，形状是两类：rc=0 的（那条扰动压根没让程序起）与 rc=1 但红在**别的**闸门的。⑶ `%(SP3)s` stdout **%(SP3_B)s** B／stderr **%(SP3_E)s** B（崩之前只打印出 %(SP3_ROWS)s 行）＝「%(SP3_WHY)s」——配额闸门第二次读响应头读空（403 响应不带那几枚头），本件按构造拒绝起任何子运行。⑷ `%(SP4)s` stdout **%(SP4_B)s** B／stderr **%(SP4_E)s** B，「× 」行 **%(SP4_MISS)s** 条、判读「%(SP4_VERDICT)s」，两条实得抛出行＝%(SP4_RAISES)s：F1 用系统解释器跑副本，把「我没占卡」那条红线打掉（argv[0] 整词匹配数到我的解释器 2 个），它期望的那句在副本第 647 行、压根没轮到；R8 的红被同一批里那条环境级复现闸门先吃掉。**同一条 × 行既可能是「判据空转」也可能是「环境故障」，这两件事必须分开数**——修法＝`classify()` 三去向（命中／环境／空转）＋环境那一支有界重跑一次＋备用解释器换成外来 argv[0] 命中 0 的那个，第三次真跑红路 %(HDR_DEC)s 条全命中、环境／超时 0 条。⑸ `%(SP5)s` stdout **%(SP5_B)s** B／stderr **%(SP5_E)s** B＝「%(SP5_WHY)s」，它逐字符等于生成器 `am_s%(SLICE)s_hdr.py` 第 **%(SP5_LINE)s** 行那条 `assert REPRO` 的消息（探针的 stdout 是 0 字节＝它死在打印之前）⇒ 403 在尺子那一侧的表现形式是「仪器不可复现」，不是「闸门红」，这条就是⑷那支分拣的出处。⑹ `%(SP6)s` stdout **%(SP6_B)s** B／stderr **%(SP6_E)s** B＝「%(SP6_WHY)s」，崩在绿路第一条之后（首行声明 %(SP6_DECL)s 条、实际打印 %(SP6_PR)s 行、作废 **%(SP6_LOST)s** 条）＝绿路的返回形状改成二元组而调用点没跟着改；该面零 API，重跑只花墙钟，不花配额（与上一片那笔「重跑要再付一次配额」最贵的账不同）。
  - **%(LESSON_NEW)s＝本轮抓到的一处控件面口径**：只读仪器那面的第一版把两条「良性扰动」（表头值尾加两个空格、正文区末行换成等价的另一个写法）配成绿路，两条都红，且红在 `%(SP7_WHY)s`。探针 `%(SP7)s`（stdout **%(SP7_B)s** B／stderr **%(SP7_E)s** B，子进程 rc＝%(SP7_RC)s）留下全过程：偏移由不动点重算成 %(SP7_OFF)s（＝表头原值 %(OFF31)s＋2），算术闭合、`cmp` 三联（%(SP7_CMP)s）、顶格字段（%(SP7_FLD)s）与残留检查**全绿**——这些闸门读的是档案的**自洽性**，长度改动被重算 absorbing 掉了；唯一抓住它的是那条**跨出处闭合**（写盘面当时打印的总字节数 vs 盘上现读）。⇒ 口径：在一个**跨闭合**的校验器（写盘面记下自己的读数、盘上再解一遍）面前，「良性扰动」必须**逐字节等长**，否则那条扰动测的是校验器而不是判据；本面随后把两条绿路都改成等长版（新 kinds 各自带一条 `len(new)==len(tgt)` 的当场断言），并给「换行加不加」那一类也配上等长要求。边界也要说清：`%(SP7)s` 的探针目录自证已删＝%(SP7_CLEAN)s，它不是坏输入，是**为了把这条边界写成可复查的出处**而造的一次现场。
  - **待入库（%(TIME)s 现数，只作排期不作承诺）**：`git status --porcelain -uall` 现算未跟踪 **%(NU)s**＋已暂存新增 **%(NNEW)s**＋修改 **%(NMOD)s**＝**%(TOT0)s** 处，分组＝%(BKT)s（和＝%(BKTSUM)s）；修改面＝%(MODTXT)s。**本轮按前提把每件落盘即 `git add`**（未跟踪 0 是这一条的直接读数，不是巧合：编辑一张已暂存的仪器会让未暂存漂移先打掉未扰动基线），于是分组和与现数在预检、写盘两次运行里同值。**最终入库数大于这里的 %(TOT0)s**——归档自查腿那件（`am_s%(SLICE)s_archive_check.py`）此刻还不存在；再往后由分片 %(NEXTSLICE)s 提交时 `git show --name-status HEAD` 现取，不在这里预支。
  - **树与钉住**：%(TIME)s 现取；钉住口径**同本节上面那条带引号的登记**（本轮每条计算命令逐条内联 `CUDA_VISIBLE_DEVICES=""`，含一行式命令；**不产生任何吞吐／加速／预算数字**）；NVML 那把尺子本轮复测仍起不来（rc＝%(NVRC)s，首行「%(NVMSG)s」；内核模块 %(NVKERN)s／用户态库 %(NVLIB)s），所以「我没占卡」这条断言继续由 `/proc/*/cmdline` 的 argv[0] **整词**匹配承担：现数我的解释器进程 **%(NGPU0)s** 个（分母＝同一趟扫到的 %(DENOM)s 个 PID，已排除本件自身与父进程，只进散文），并配三条对照：本件自己的 argv[0] 按同一读法必须命中、合成的 argv[0]＝解释器必须数出 1、argv[0]＝/bin/bash 而参数里带着我的路径必须数出 0；卡上 compute 进程总数本轮仍不可现取，不写进任何档案面，也不拿"上次那个数"顶替。
"""

OVER_T = r"""- **%(TIME)s（分片 %(SLICE)s 落地腿＝提交＋推送＋五步落地证明＋生成器回执表头＋三面控件面＋落地填值；生产树 src＋tests 改动 %(NCODE)s、GPU 全程未碰）**（开发日志 §26.%(SECREF)s「分片 %(SLICE)s＝提交腿」）：
  - **入库面**：`%(HEAD7)s`（父 `%(PAR7)s`）＝新增 (A) **%(NAC)s**＋修改 (M) **%(NMC)s**＝**%(NTOTC)s** 件；shortstat 原文「%(SSRAW)s」＋`--numstat` 现算 %(NSINS)s／%(NSDEL)s ⇒ 零删除的纯归档腿；推送区间 `%(PAR7)s..%(HEAD7)s`。
  - **落地证明五步＋四个正对照**：`truncated=%(TRUNC)s`，tree 条目 **%(TB)s**＝blob **%(BB)s**＋tree **%(BT)s**；闭合式「%(CLOS)s」，且 %(BB)s − (%(PB)s ＋ %(NAC)s) ＝ %(CLOSEX)s 由 git 现算复对；双向路径差集「%(DIFF3)s」；全库 blob SHA 普查「%(SHA4)s」；本片单列复核「%(PER4)s」；判读行 `%(V)s`。四个正对照（远端某条 sha 首位改 f／远端删一条路径／闭合式右边 +1／远端删本片一条路径）各自能红。
  - **回执表头**：`%(R31LOG)s` **%(R31B)s** B／**%(R31L)s** 行＝表头 **%(OFF31)s**＋正文 **%(BODYB)s**（正文区与 `%(RAWM)s` 逐字节相同，`cmp -i %(OFF31)s:0` 三联 `%(CMP3)s` 由只读仪器自己起进程跑，且本件又复跑两趟与之三方逐字节相同＝%(REPRO)s）；顶格字段 %(FLDN)s/%(FLDN)s；核验件 stderr %(ERRMB)s B。
  - **三面控件面**：提交说明面 **%(MSG_DEC)s** 条扰动／造不出红路 **%(MSG_UNC)s** 条；表头面红路 **%(HDR_DEC)s** 条／保持绿 **%(HDR_GRN)s** 条／环境故障 **%(HDR_TO)s** 条／造不出红路 **%(HDR_UNC)s** 条（跨配额窗口等待 %(HDR_WAITN)s 次、累计 %(HDR_WAITS)s 秒）；只读仪器面红路 **%(AFTC_DEC)s** 条／保持绿 **%(AFTC_GRN)s** 条／实跑 **%(AFTC_ROWS)s** 行／超时 **%(AFTC_TO)s** 条／装配失败 **%(AFTC_ASM)s** 条／造不出红路 **%(AFTC_UNC)s** 条；三面判读一律「ALL CONTROLS CAN GO RED」，且各面源码的 ast 表长与其归档 stdout 声明逐面同值（＝同源）。
  - **一条口径级新学＝%(LESSON_NEW)s**：在跨闭合的校验器（写盘面记下自己的读数、盘上再解一遍）面前，控件面的「良性扰动」必须逐字节等长——只读仪器面两条按直觉配的"无害编辑"都红在总字节数上，而偏移、算术闭合、`cmp` 三联、字段数、残留检查**全部保持绿**（探针 `%(SP7)s` 留档：偏移由不动点重算成 %(SP7_OFF)s、三联仍 %(SP7_CMP)s、字段仍 %(SP7_FLD)s）。同轮另抓到判据侧的第四处：**同一条「× 」行既可能是判据空转也可能是环境故障**（F1 用系统解释器跑副本把自己那条 GPU 红线打掉、R8 的红被环境级复现闸门吃掉），修＝三去向 `classify()`＋有界重跑；以及一把新尺子＝归档 stdout 与源码的 ast 条数闭合（它把「编辑过已暂存的仪器却没复跑控件面」从看不见的漂移变成当场红）。
  - **排期不变**（**#30 试片尺寸裁决在前**、`--stage full` 暂停）；红线：A0 断言一字不动、#27 不换指标、无新增 skip/xfail、性能数字只在 GPU 窗口测。
"""

PLAN_T = r"""### §W 补记 %(PNUM)s（%(TIME)s 落笔，`date '+%%F %%T %%z'` 现取）＝分片 %(SLICE)s 的**落地腿**：仓库推进到 `%(HEAD7)s`／远端 blob **%(BB)s**，并把「控件面的良性扰动」与「环境 vs 空转的分拣」折成口径

- **外部事实只有一条**：远端 main ＝ `%(HEAD40)s`（父 `%(PAR40)s`），推送区间 `%(PAR7)s..%(HEAD7)s`；本片＝**纯归档腿**，生产树 `src`＋`tests` 改动 **%(NCODE)s** 件。入库面＝新增 (A) **%(NAC)s**＋修改 (M) **%(NMC)s**＝**%(NTOTC)s** 件；shortstat「%(SSRAW)s」与 `--numstat` 现算（%(NSINS)s 插入／%(NSDEL)s 删除）两口径当场对账。
- **落地证明的读数**（逐条从归档正文 `%(RAWM)s` 正则现取，其 stderr **%(ERRMB)s** B）：tree 条目 **%(TB)s**＝blob **%(BB)s**＋tree **%(BT)s**；%(CLOS)s；双向差集「%(DIFF3)s」；全库 blob SHA「%(SHA4)s」；本片单列「%(PER4)s」；判读行 %(V)s。闭合式另用 git 现算复对：%(BB)s − (%(PB)s ＋ %(NAC)s) ＝ **%(CLOSEX)s**。
- **回执表头**：`%(R31LOG)s` **%(R31B)s** B／**%(R31L)s** 行＝表头 **%(OFF31)s**＋正文 **%(BODYB)s**，`BODY_OFFSET=%(OFF31)s` 由写入件重解、由只读仪器从盘上**读回来**；三联 `cmp -i %(OFF31)s:0`＝`%(CMP3)s` 由只读仪器自己起进程跑（不是抄生成器的），本件另起两趟复跑与之三方逐字节相同＝%(REPRO)s；顶格字段 %(FLDN)s/%(FLDN)s 与生成器 `FIELDS`（ast 现取）互证；只读仪器判读行 `%(HRAV)s`。
- **三面控件面与两处口径**：提交说明面 stdout **%(MSG_OUTB)s** B（md5 前 12 `%(MSG_OUTM)s`）＝CASES 声明 **%(MSG_DEC)s** 条／造不出红路 **%(MSG_UNC)s** 条；表头面 stdout **%(HDR_OUTB)s** B（md5 前 12 `%(HDR_OUTM)s`）＝红路声明 **%(HDR_DEC)s** 条／保持绿 **%(HDR_GRN)s** 条／环境故障 **%(HDR_TO)s** 条／造不出红路 **%(HDR_UNC)s** 条；只读仪器面 stdout **%(AFTC_OUTB)s** B＝红路 **%(AFTC_DEC)s** 条／保持绿 **%(AFTC_GRN)s** 条／实跑 **%(AFTC_ROWS)s** 行／装配失败 **%(AFTC_ASM)s** 条／造不出红路 **%(AFTC_UNC)s** 条。**口径一＝%(LESSON_NEW)s**：跨闭合的校验器（写盘面记字节数、盘上再解）把"无害编辑"判成坏输入，而偏移／算术／三联／字段／残留五道自洽门全绿（探针 `%(SP7)s`：重算后的偏移 %(SP7_OFF)s、三联仍 %(SP7_CMP)s）⇒ 扰动要与被验量无关就得**等长**。**口径二＝环境故障与判据空转要分开数**：F1 用系统解释器跑副本 ⇒ 打掉自己的 GPU 红线（argv[0] 整词数到我的进程 2 个），R8 的红被环境级复现闸门先吃掉（同形状的出处由探针 `%(SP5)s` 留下，那句逐字符等于生成器第 %(SP5_LINE)s 行那条 `assert REPRO`），修＝`classify()` 三去向＋有界重跑。**口径三（仪器化）＝归档 stdout 与源码的 ast 条数闭合**：三面表长与各自 stdout 声明逐面同值＝%(ASTCLOSE)s，它专门抓「编辑过已暂存的仪器却没复跑控件面」这种状态（本轮真中一次，崩溃现场 `%(SP6)s` 作废 **%(SP6_LOST)s** 条读数）。
- **待入库与排期**：落笔现数未跟踪 **%(NU)s**＋已暂存新增 **%(NNEW)s**＋修改 **%(NMOD)s**＝**%(TOT0)s** 处（分组和＝%(BKTSUM)s），最终入库数大于它（归档自查件此刻还不存在）；分片 %(NEXTSLICE)s 的落地腿要在 §26.%(SECREF)s 里现取**唯一**一条引用 `%(R31LOG)s` 的顶级子条——本节由落地段恰好提供 1 条，再复述一次就会被它的 `bullet_for` 拒绝。**红线不变**：A0 断言一字不动、#27 不换指标、无新增 skip/xfail、性能数字只在 GPU 窗口测（NVML 复测仍 rc＝%(NVRC)s，替代断言现数我的解释器 **%(NGPU0)s** 个，只作断言）。
"""

DAILY_T = r"""## %(TIME)s：分片 %(SLICE)s 落地腿＝替分片 %(PREVSLICE)s 入库整条链（A **%(NAC)s**／M **%(NMC)s**＝**%(NTOTC)s** 件）＋一条口径级新学＝%(LESSON_NEW)s

- **一句话**：远端 main ＝ `%(HEAD40)s`（父 `%(PAR40)s`，提交时刻 `%(CDATE)s`），入库 A %(NAC)s／M %(NMC)s＝%(NTOTC)s 件；shortstat「%(SSRAW)s」＋numstat %(NSINS)s／%(NSDEL)s ⇒ 纯归档腿、零删除、生产树改动 %(NCODE)s。
- **落地证明**（全从 `%(RAWM)s` 现取）：tree 条目 **%(TB)s**＝blob **%(BB)s**＋tree **%(BT)s**；「%(CLOS)s」，git 现算复对差 **%(CLOSEX)s**；差集「%(DIFF3)s」；全库 SHA「%(SHA4)s」；本片单列「%(PER4)s」；判读行「%(V)s」。
- **回执**：`%(R31LOG)s` %(R31B)s B／%(R31L)s 行＝表头 %(OFF31)s＋正文 %(BODYB)s（正文区逐字节＝原始件），三联 `%(CMP3)s`，顶格字段 %(FLDN)s/%(FLDN)s，stderr %(ERRMB)s B；只读仪器复跑两趟与归档件三方逐字节相同＝%(REPRO)s。
- **三面控件面**：%(MSG_OUT)s %(MSG_OUTB)s B／%(HDR_OUT)s %(HDR_OUTB)s B／%(AFTC_OUT)s %(AFTC_OUTB)s B；声明 %(MSG_DEC)s／红路 %(HDR_DEC)s＋%(AFTC_DEC)s、保持绿 %(MSG_GRN)s／%(HDR_GRN)s／%(AFTC_GRN)s、超时与环境 0/0/0、装配失败 0；三面判读一律 ALL CONTROLS CAN GO RED；三面源码 ast 表长与归档声明逐面同值（＝同源）。
- **事故与修法**（现场都按别名留档在本目录）：提交说明面 ⑴基线标记句命中 2 行（`%(SP1)s`，%(SP1_B)s／%(SP1_E)s B）⑵判据空转 %(SP2_MISS)s 条⇒整面 PROBLEM（`%(SP2)s`）；表头面 ⑶配额响应头读空、拒绝起任何子运行（`%(SP3)s`，只打印 %(SP3_ROWS)s 行）⑷F1 解释器污染把 GPU 红线打掉＋R8 的红被环境级复现闸门吃掉（`%(SP4)s`，未命中 %(SP4_MISS)s 条）⇒ 修＝`classify()` 三去向（命中／环境／空转）＋有界重跑＋备用解释器换外来 argv[0] 命中 0 的那个，第三次真跑 %(HDR_DEC)s 条全命中；⑸同形状的出处探针 `%(SP5)s`（stdout %(SP5_B)s B／stderr %(SP5_E)s B）逐字符等于生成器第 %(SP5_LINE)s 行那条 `assert REPRO`；只读仪器面 ⑹绿路返回形状改二元组而调用点没改，崩在第 %(SP6_PR)s 行、作废 **%(SP6_LOST)s** 条（`%(SP6)s`，零 API ⇒ 只花墙钟）。**新学＝%(LESSON_NEW)s**：表头值尾加两个空格这类"无害编辑"红在写盘面记下的总字节数，而偏移（重算成 %(SP7_OFF)s）、`cmp` 三联（仍 %(SP7_CMP)s）、字段数（仍 %(SP7_FLD)s）全绿——跨闭合的校验器面前，良性扰动必须等长（探针 `%(SP7)s` 留档）。
- **待入库**：未跟踪 **%(NU)s**＋已暂存新增 **%(NNEW)s**＋修改 **%(NMOD)s**＝**%(TOT0)s** 处（分组和＝%(BKTSUM)s）；最终入库数大于它，归档自查腿那件此刻还不存在。**GPU**：NVML 复测仍 rc＝%(NVRC)s（首行「%(NVMSG)s」；内核 %(NVKERN)s／用户态 %(NVLIB)s），替代断言＝argv[0] 整词匹配，现数我的解释器 %(NGPU0)s 个（分母 %(DENOM)s 个 PID，已排除自身与父进程，只进散文）＋三条对照（自己的真条目必须命中／合成 argv[0] 数出 1／参数里带着路径的那条数出 0）；本轮不产生任何吞吐／加速／预算数字。
"""

WBIDX_T = r"""- **%(TIME)s（分片 %(SLICE)s 落地腿；生产树 0 改动、GPU 全程未碰）**：
  远端 main `%(HEAD7)s`（父 `%(PAR7)s`；入库 A **%(NAC)s**／M **%(NMC)s**＝**%(NTOTC)s** 件，shortstat「%(SSRAW)s」＝零删除）。五步落地证明全绿：tree 条目 **%(TB)s**＝blob **%(BB)s**＋tree **%(BT)s**；「%(CLOS)s」（git 现算复对差 %(CLOSEX)s）；差集 %(DIFF3)s；全库 SHA 普查 %(SHA4)s；本片单列 %(PER4)s；判读行 %(V)s。
  回执 `%(R31LOG)s` **%(R31B)s** B／**%(R31L)s** 行＝表头 **%(OFF31)s**＋正文 **%(BODYB)s**，三联 `cmp -i %(OFF31)s:0`＝`%(CMP3)s`（只读仪器自跑，本件另复跑两趟三方同字节），顶格字段 %(FLDN)s/%(FLDN)s。
  三面控件面：提交说明面声明 %(MSG_DEC)s／造不出红路 %(MSG_UNC)s；表头面红路 %(HDR_DEC)s／保持绿 %(HDR_GRN)s／环境 %(HDR_TO)s／造不出红路 %(HDR_UNC)s；只读仪器面红路 %(AFTC_DEC)s／保持绿 %(AFTC_GRN)s／实跑 %(AFTC_ROWS)s 行／超时 %(AFTC_TO)s／装配失败 %(AFTC_ASM)s／造不出红路 %(AFTC_UNC)s。新学＝%(LESSON_NEW)s（跨闭合校验器面前良性扰动必须等长：偏移／三联／字段全绿，只有写盘面字节数抓住它）；同轮两处口径＝环境故障与判据空转要 `classify()` 分开数、归档 stdout 与源码的 ast 条数要逐面闭合（它抓住了「编辑过已暂存的仪器却没复跑控件面」）。
  待入库＝未跟踪 **%(NU)s**＋已暂存新增 **%(NNEW)s**＋修改 **%(NMOD)s**＝**%(TOT0)s** 处（分组和 %(BKTSUM)s；最终数大于它，归档自查腿还没写）。排期不变：**#30 裁决在前**、`--stage full` 暂停；红线不变。
"""

FACES = {"DEV": DEV_T, "OVER": OVER_T, "PLAN": PLAN_T, "DAILY": DAILY_T, "WBIDX": WBIDX_T}
print("== named() 自检与模板渲染")


def must_reject(label, tmpl):
    """named() 的负对照：坏模板必须被拦下，否则那道裸百分号门是恒过的装饰。
    缺席的槽在替换那一步是 KeyError，形状错的槽是 AssertionError——两种都算「拦下」。"""
    try:
        named(label, tmpl)
    except (AssertionError, KeyError):
        return True
    raise SystemExit("正对照失败：%s 这类模板本该被 named() 拦下却过了 ⇒ 那道门是装饰" % label)


assert "x % y" == named("自检/好", "x %% y"), "具名替换把双写百分号渲染错了 ⇒ 渲染结果不可信"
assert "x " + CTX["SLICE"] + " y" == named("自检/好", "x %(SLICE)s y"), "具名槽没被替换"
assert "/" in CTX["EVIDREL"], "EVIDREL＝%s 里没有斜杠 ⇒ 这条斜杠对照是空转，换一个带斜杠的值再验" % CTX["EVIDREL"]
assert "x " + CTX["EVIDREL"] + " y" == named("自检/斜杠槽", "x %(EVIDREL)s y"), "含斜杠的值填进具名槽被改写了"
must_reject("自检/裸百分号", "git log --format=%cd 现取")
must_reject("自检/全角槽", "槽＝%（ SLICE）s")
must_reject("自检/下标槽", "槽＝%(MSG[OUT]s）")
must_reject("自检/坏槽", "槽＝%(NO_SUCH_KEY)s")
print("   named()：好模板逐字节正确，裸百分号／全角槽／下标槽／缺席槽四样各被拦下 -> PASS")

texts = {}
for k, tmpl in FACES.items():
    texts[k] = named(k, tmpl)

_used = set(re.findall(r"%\(([^)]+)\)s", "".join(FACES.values())))
_dead = sorted(set(CTX) - _used)
assert not _dead, "取到的读数没进任何一张面（＝算了没人读的字段）：%s" % _dead[:8]
_missing = sorted(_used - set(CTX))
assert not _missing, "模板里有没取到值的具名槽（`%` 替换会当场 KeyError，但先在这里点名）：%s" % _missing[:8]
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

_lab_now = [b[0] for b in _now]
_lab_new = [m.group(1) for m in re.finditer(r"^- \*\*(.+?)\*\*", texts["DEV"], re.M)]
assert len(_lab_new) == 1, "落地片段里的顶级子条＝%d 条（须恰好 1 条，其下都是缩进子条）：%s" % (
    len(_lab_new), _lab_new)
assert not (set(_lab_now) & set(_lab_new)), "落地子条标签与本节已有标签撞名 ⇒ 按标签切段会串行"
_lab_all = _lab_now + _lab_new
assert len(_lab_all) == len(set(_lab_all)) == len(_lab_now) + 1, (
    "§26.%s 顶级子条：落笔前 %d 条、合并后应为 %d 条，现算 %d 条／互异 %d 条" % (
        SECREF, len(_lab_now), len(_lab_now) + 1, len(_lab_all), len(set(_lab_all))))
_sub_now = len(rows(SEC_BODY, r"^  - \*\*"))
_sub_new = len(rows(texts["DEV"], r"^  - \*\*"))
assert _sub_new >= 8, "落地片段的缩进子条只有 %d 条（本轮登记的是 9 条）⇒ 有一段没渲染出来" % _sub_new
print("   §26.%s 顶级子条：落笔前 %d 条、落地片段贡献 1 条（其下缩进子条 %d 条，本节原有 %d 条）且标签互异 -> PASS" % (
    SECREF, len(_lab_now), _sub_new, _sub_now))

# 节内唯一性按**顶级子条**为单位查（下游 `bullet_for` 的问法），每条都配一根「再补一条顶级子条就 +1」的 bait
NEEDLES = (("引用本片回执名", RLOG, 1),
           ("上一片回执名（本节已有恰好 1 条，本片段必须贡献 0）", "am_s7_verify%d.log" % PREVSLICE, 0),
           ("带引号的钉住口径登记", r"当前「[^」]+」", 0),
           ("承接锚句：自审的方向性", CARRIED[0], 0),
           ("承接锚句：普查仪器没有第二出处", CARRIED[1], 0),
           ("本轮新学锚句", LESSON_NEW, 1))
BAIT = {"引用本片回执名": "\n- **诱饵子条**：见 %s。\n" % RLOG,
        "上一片回执名（本节已有恰好 1 条，本片段必须贡献 0）": "\n- **诱饵子条**：见 am_s7_verify%d.log。\n" % PREVSLICE,
        "带引号的钉住口径登记": "\n- **诱饵**：当前「诱饵口径」\n",
        "承接锚句：自审的方向性": "\n- **诱饵**：自审的方向性\n",
        "承接锚句：普查仪器没有第二出处": "\n- **诱饵**：普查仪器没有第二出处\n",
        "本轮新学锚句": "\n- **诱饵**：%s\n" % LESSON_NEW}
_merged = SEC_BODY + texts["DEV"]
for what, pat, want in NEEDLES:
    rx = re.escape(pat) if not pat.startswith("当前「") else pat
    frag = len([1 for _l, b in top_bullets(texts["DEV"]) if re.search(rx, b, re.M)])
    assert frag == want, "%s 在落地片段的顶级子条里命中 %d 条（须恰好 %d）⇒ 节内唯一性会被下游 `bullet_for` 拒绝" % (
        what, frag, want)
    _now_n = len([1 for _l, b in _now if re.search(rx, b, re.M)])
    assert _now_n + want == 1, "%s 合并后在节内现数 %d（落笔前 %d＋片段 %d，须恰好 1）" % (
        what, _now_n + want, _now_n, want)
    n2 = len([1 for _l, b in top_bullets(_merged + BAIT[what]) if re.search(rx, b, re.M)])
    assert n2 == 1 + 1, "%s 的正对照空转（补一条诱饵顶级子条后现数 %d，须 2）⇒ 这条唯一性断言是恒真式" % (what, n2)
_fs = r"f-string 把 `\{32\}` 当成\*\*替换字段"
assert len(re.findall(_fs, DEV_NOW)) == 1, (
    "f-string 事故句在落笔前的整份日志里不是恰好 1 处 ⇒ 分片 %s 的生成器本来就会拒绝归档" % NEXTSLICE)
assert len(re.findall(_fs, texts["DEV"])) == 0, "落地片段不该新增 f-string 事故句"
assert len([1 for _l, b in _now if RLOG in b]) == 0, (
    "本节落笔前已经有顶级子条引用本片回执名 ⇒ 唯一性被片段之外的地方破坏，先查是谁写的")
print("   六条节内唯一性按顶级子条各按其形制成立（片段贡献 1／0／0／0／0／1，合并后各恰好 1），"
      "且各配一根「再补一条顶级子条就 +1」的正对照 -> PASS")

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
    ("档案名＝`", CTX["R31LOG"]),
    ("三联＝`", CTX["CMP3"]),
    ("判读行 `", CTX["V"]),
    ("未跟踪 **", CTX["NU"]),
    ("探针 `", CTX["SP7"]),
    ("现数我的解释器进程 **", CTX["NGPU0"]),
])
expect_near("OVER", texts["OVER"], [
    ("入库面**：`", CTX["HEAD7"]),
    ("blob **", CTX["BB"]),
    ("三联 `", CTX["CMP3"]),
    ("回执表头**：`", CTX["R31LOG"]),
])
expect_near("PLAN", texts["PLAN"], [
    ("远端 main ＝ `", CTX["HEAD40"]),
    ("其 stderr **", CTX["ERRMB"]),
    ("探针 `", CTX["SP7"]),
    ("补记 ", CTX["PNUM"]),
])
expect_near("DAILY", texts["DAILY"], [
    ("远端 main ＝ `", CTX["HEAD40"]),
    ("「", CTX["CLOS"]),
    ("三联 `", CTX["CMP3"]),
])
expect_near("WBIDX", texts["WBIDX"], [
    ("远端 main `", CTX["HEAD7"]),
    ("回执 `", CTX["R31LOG"]),
    ("＋正文 **", CTX["BODYB"]),
])
try:
    expect_near("自检/错位", texts["DEV"], [("HEAD＝`", CTX["PAR40"])])
except AssertionError:
    pass
else:
    raise SystemExit("正对照失败：邻接断言在「父哈希被填进 HEAD 槽」时没报错 ⇒ 那道门是装饰")
print("   邻接断言：五面各 3–8 对「标签＋值」原文命中；把父哈希填进 HEAD 槽当场被拒 -> PASS")

os.makedirs(OUT_DIR, exist_ok=True)
for k, v in texts.items():
    open(os.path.join(OUT_DIR, k + ".md"), "w", encoding="utf-8").write(v)
open(os.path.join(OUT_DIR, "ctx.txt"), "w", encoding="utf-8").write(
    "\n".join("%s=%s" % (kk, vv) for kk, vv in sorted(CTX.items())) + "\n")
print("== 渲染完成：%s/{DEV,OVER,PLAN,DAILY,WBIDX}.md ＋ ctx.txt（渲染面含易失读数，但它不落仓库）" % OUT_DIR)


# ============================ 写盘腿（带守卫、一次性） ============================
def faces_in(rows_):
    """纯函数：`rows_＝[(pid, argv 词集合)]` → 命中守卫表的进程。拆成纯函数才配得起正对照。"""
    hits = []
    for pid, toks in rows_:
        watched = {t for t in toks if t in TOKENS or t.startswith(COPY_PREFIX)
                   or t.startswith(AFTC_COPY_PREFIX)}
        if watched and SELF_BASE not in toks:
            hits.append((pid, sorted(watched)))
    return hits


def running_faces():
    """正在跑的仪器：按 `/proc/PID/cmdline` 的**整词**比对 basename，不按命令行子串猜。"""
    rr = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            argv = open("/proc/%s/cmdline" % d, "rb").read().split(b"\0")
        except OSError:
            continue
        rr.append((d, {os.path.basename(a.decode("utf-8", "replace")) for a in argv if a}))
    return faces_in(rr)


_ctl = faces_in([("9001", {"python", "am_s%d_hdr.py" % SLICE}),
                 ("9002", {"python", COPY_PREFIX + "perturbed.py"}),
                 ("9003", {"python", "some_other_tool.py"}),
                 ("9004", {"python", SELF_BASE, "am_s%d_hdr.py" % SLICE}),
                 ("9005", {"python", AFTC_COPY_PREFIX + "copy.py"})])
assert [p for p, _ in _ctl] == ["9001", "9002", "9005"], (
    "写盘守卫的正对照判出 %s（须＝[9001, 9002, 9005]：表内名 1＋两个前缀族各 1，无关名与自我样本各排除 1）"
    "⇒ 那道「有仪器在跑就拒绝写盘」的闸门红不了" % ([p for p, _ in _ctl],))
assert _ctl[0][1] == ["am_s%d_hdr.py" % SLICE] and _ctl[1][1] == [COPY_PREFIX + "perturbed.py"], (
    "正对照命中了名字但记错了词 ⇒ 守卫报出来的现场不可信")
print("   写盘守卫正对照：5 个样本判 3 中（表内名／两种副本前缀各 1、无关名与自我样本各排除 1）-> PASS")

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
    _merged_now = top_bullets(seg)
    for what, pat, want in NEEDLES:
        rx = re.escape(pat) if not pat.startswith("当前「") else pat
        n = len([1 for _l, b in _merged_now if re.search(rx, b, re.M)])
        assert n == 1, "磁盘上的 §26.%s 里 %s 命中 %d 条顶级子条（预检＝落笔前 %d＋片段 %d＝1）⇒ 分片 %s 会拒绝归档" % (
            SECREF, what, n, len([1 for _l, b in _now if re.search(rx, b, re.M)]), want, NEXTSLICE)
    assert len(re.findall(_fs, now_dev)) == 1, "f-string 事故句在落盘后的整份日志里不是恰好 1 处"
    _lab_disk = [b[0] for b in _merged_now]
    assert len(_lab_disk) == len(set(_lab_disk)) == len(_lab_all), (
        "磁盘上 §26.%s 顶级子条＝%d／互异＝%d（预检 %d）" % (
            SECREF, len(_lab_disk), len(set(_lab_disk)), len(_lab_all)))
    print("   磁盘上的 §26.%s：%d 条顶级子条标签互异、六条唯一性各恰好 1 条顶级子条 -> PASS" % (
        SECREF, len(_lab_disk)))
    post = [l for l in sh("git", "-c", "core.quotePath=false", "status", "--porcelain", "-uall").splitlines()
            if l.strip()]
    _pc = {}
    for l in post:
        _pc.setdefault(l[3:], []).append("UNT" if l[:2] == "??" else "NEW" if l[0] == "A" else "MOD")
    nq = len([p for p, c in _pc.items() if c == ["UNT"]])
    nn = len([p for p, c in _pc.items() if c == ["NEW"]])
    nm = len(post) - nq - nn
    assert (nq, nn) == (NU, NNEW), "未跟踪／已暂存新增从 (%d, %d) 变成 (%d, %d) ⇒ 写盘动到了不该动的地方" % (
        NU, NNEW, nq, nn)
    _mpost = sorted(l[3:] for l in post if l[:2] not in ("?? ",) and l[0] != "A")
    _want = sorted(os.path.relpath(p, REPO) for p in FACES5)
    assert _mpost == _want, "写盘后被改动的面不是预期的五张：%s" % _mpost
    # 这条分类自己的正对照：`git status --porcelain` 给的是**仓库相对**路径，而 FACES5 是绝对路径，
    # 上面那句已经把它们换算过一次；这里再验分类**不是**「清单里的一切」——本件那四张已暂存的原始面
    # 若被当成修改面混进来，`_mpost == _want` 会红，但那正是分类解析写错时最可能出现的形状。
    assert not [p for p in _mpost if p.startswith(EVID_REL + "/_raw_s%d_landing" % SLICE)], \
        "被改面里混进了本轮的已暂存新增原始面 ⇒ 状态列解析与清单不同源，上面那句 PASS 不可信"
    assert nm == NMOD + 5, "修改面数从 %d 变成 %d（须恰好 +5＝五张档案面各 1 处）" % (NMOD, nm)
    print("   闭合式复算：未跟踪 %d（不变）＋已暂存新增 %d（不变）＋修改 %d（＝落笔 %d＋刚追加的五张面）-> PASS" % (
        nq, nn, nm, NMOD))
    print("WRITER_VERDICT = FIVE LANDING FACES LANDED, UNIQUENESS RECOUNTED ON DISK")
else:
    print("（预检模式：未写盘。加 --write。）")
print("LANDING_FILLER_VERDICT = READINGS DERIVED, STRUCTURE GATES PASS")
