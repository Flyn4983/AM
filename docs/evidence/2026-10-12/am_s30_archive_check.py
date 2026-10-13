#!/usr/bin/env python3
r"""归档自查（分片 30 的落地半腿）：本轮**五张档案面**里每一条落地声称，由命令现取一次再回搜原样。

本件与 `am_s30_docs_landing.py`（写手）的分工＝尺子同源：下面「现取」那一段的**命令与正则**逐字节切自写手
（读数 1／2／3／自审现场三段），所以「档案里写的」与「本件重新现取的」是同一把尺子；改了格式两边一起红，
不会出现一个改了一个没改。写手自己那套结构闸门在**落笔前**跑，本件在**落笔后**跑，两趟之间隔着真实写盘。

范围照实说清：本件普查「落地读数」这张清单（HEAD／父／入库 A-M-合计／shortstat／numstat／生产树改动／
blob＋tree 条目／闭合式与 git 复对差／三组带分母的 0／四个正对照／判读行／回执三段字节／`cmp` 三联／
顶格字段与 `FIELDS` 互证／核验件 stderr／尺子 md5 与份数／SRC 指纹／三面控件面各自的字节-md5-声明-保持绿-
超时-UNCOVERED-判读行／自审反转现场两件与崩掉的批／五面各自的专有句）。**不是**对归档散文做全量出处普查。

三条口径（都是前面几片实测到的形状）：
① volatile 读数（未入库件数、卡上 compute 进程数、落笔 `date`）**只断言、不入 stdout**，否则「本件复跑逐字节
   相同」做不到；散文里那些带日期的数由「面内自洽」那条检查（分组和＝合计）核，不由现取核。
② 每个 0 配一条**同一次运行里能红**的正对照：五内存扰动（读数加尾记号／一位哈希扰动／逐面抽空白／全集抽
   空白／承载量互证）＋节内四条唯一性各自的诱饵。抽空白那一条**按测量选面**（先数每面的独有声称数，只对有
   独有的面要求它红），不硬写「只有开发日志承载独有声称」——那是上一片的测量结果，不是本轮的事实。
③ 分片 31 的表头生成器要在 §26.%s 里现取四条唯一出处（引用本片回执名／带引号的钉住口径／上一片 LESSON／
   本片 LESSON），本件在盘上重数一次，并各配一根「再复述就 +1」的诱饵。

运行＝`CUDA_VISIBLE_DEVICES="" /tmp/amvenv/bin/python -B docs/evidence/2026-10-12/am_s30_archive_check.py`
（不 import jax：只跑 git／md5sum／cmp／nvidia-smi／读文件。）
"""
import ast
import glob
import hashlib
import os
import re
import subprocess

REPO = "/home/shy/桌面/2026年开发-显式冲击有限元-物质点法-sph-fvm+am/AM-Qoder"
EVID_REL = "docs/evidence/2026-10-12"
D = os.path.join(REPO, EVID_REL)
DEVLOG = os.path.join(REPO, "docs/开发日志.md")
OVERVIEW = os.path.join(REPO, "docs/项目开发总览.md")
PLAN = os.path.join(REPO, "docs/项目评估与下一步计划_2026-10-06.md")
DAILY = os.path.join(REPO, ".workbuddy/memory/2026-10-12.md")
WBIDX = os.path.join(REPO, ".workbuddy/memory/MEMORY.md")
FACE_NAMES = ["docs/开发日志.md", "docs/项目开发总览.md", "docs/项目评估与下一步计划_2026-10-06.md",
              ".workbuddy/memory/2026-10-12.md", ".workbuddy/memory/MEMORY.md"]
FACE_PATHS = [DEVLOG, OVERVIEW, PLAN, DAILY, WBIDX]
SELF_CLAIM = "docs/evidence/2026-10-12/am_s30_archive_check.py"
LESSON_NEW = "自审的方向性"
LESSON_PREV = "对照的可满足性"


def sh(*a):
    return subprocess.run(a, cwd=REPO, capture_output=True, text=True).stdout


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def one(needle, text, what):
    hits = re.findall(needle, text, re.M)
    assert len(hits) == 1, "%s 命中 %d 处（须恰好 1）：%s" % (what, len(hits), hits[:3])
    return hits[0]


# ============================ 现取（与写手同源） ============================
HEAD = sh("git", "rev-parse", "HEAD").strip()
HEAD7 = sh("git", "rev-parse", "--short", "HEAD").strip()
PAR40 = sh("git", "rev-parse", "HEAD^").strip()
PAR7 = sh("git", "rev-parse", "--short", "HEAD^").strip()
SUBJ = sh("git", "log", "-1", "--format=%s").strip()
SLICE = int(one(r"S7 分片 (\d+)", SUBJ, "HEAD 提交说明里的片号"))
PREVSLICE, NEXTSLICE = str(SLICE - 1), str(SLICE + 1)
assert sorted({int(x) for x in re.findall(r"分片 (\d+)", SUBJ)}) == [SLICE - 1, SLICE], (
    "HEAD 提交说明里的片号集合不＝[%d, %d] ⇒ 本件的片号方向没被双向钉住" % (SLICE - 1, SLICE))
R30LOG = "am_s7_verify%d.log" % SLICE
RAWM = "_raw_verify%d.out" % SLICE
ERRM = "_raw_verify%d.err" % SLICE
GEN = os.path.join(D, "am_s%d_hdr.py" % SLICE)

_ns = [l for l in sh("git", "show", "--name-status", "--format=", "HEAD").splitlines() if l.strip()]
_rows = [l.split("\t") for l in _ns if "\t" in l]
assert len(_rows) == len(_ns) and all(len(r) == 2 for r in _rows), "name-status 行形制不是「状态字母 TAB 路径」"
KINDS = [r[0] for r in _rows]
CPATHS = [r[-1] for r in _rows]
assert set(KINDS) <= {"A", "M"}, "HEAD 清单里有 A／M 之外的状态：%s" % sorted(set(KINDS))
NA, NM = KINDS.count("A"), KINDS.count("M")
NTOT = len(CPATHS)
NSR = [l for l in sh("git", "show", "--numstat", "--format=", "HEAD").splitlines() if l.strip()]
assert len(NSR) == NTOT and all("\t" in l and l.split("\t")[0] != "-" for l in NSR), (
    "numstat 行数／形状与 name-status 不对齐（含 `-` 就说明有二进制或 rename，本件的求和口径不适用）")
NS_INS = sum(int(l.split("\t")[0]) for l in NSR)
NS_DEL = sum(int(l.split("\t")[1]) for l in NSR)
SS = sh("git", "diff-tree", "--no-commit-id", "--shortstat", "-r", "HEAD").strip()
SS_FILES = int(one(r"(\d+) files? changed", SS, "shortstat 的文件数"))
SS_INS = int(one(r"(\d[\d,]*) insertions?", SS, "shortstat 的插入数"))
assert (SS_FILES, SS_INS) == (NTOT, NS_INS) and NS_DEL == 0 and "deletion" not in SS, (
    "shortstat (%d, %d) 与 numstat 现算 (%d, %d)／删除 %d 不闭合 ⇒ 「零删除的纯归档腿」那句是假话" % (
        SS_FILES, SS_INS, NTOT, NS_INS, NS_DEL))
NCODE = len([l for l in sh("git", "-c", "core.quotePath=false", "status", "--porcelain",
                           "--", "src", "tests").splitlines() if l.strip()])
assert NCODE == 0, "生产树 src＋tests 有 %d 处改动 ⇒ 本轮不是纯归档腿" % NCODE

r30 = open(os.path.join(D, R30LOG), encoding="utf-8").read()
R30B, R30L = os.path.getsize(os.path.join(D, R30LOG)), len(r30.splitlines())
rawm = open(os.path.join(D, RAWM), encoding="utf-8").read()
ERRMB = os.path.getsize(os.path.join(D, ERRM))
assert ERRMB == 0, "核验件 stderr %d B（非 0）⇒ 有警告被吞" % ERRMB
OFF = int(one(r"^BODY_OFFSET=(\d+)$", r30, "本片回执的 BODY_OFFSET"))
_marker = "BODY_OFFSET=%d\n" % OFF
assert r30.count(_marker) == 1, "BODY_OFFSET 那行不是恰好 1 条"
_hp, _body = r30.split(_marker, 1)
assert len((_hp + _marker).encode("utf-8")) == OFF, "BODY_OFFSET 与实际表头字节数不等 ⇒ 定点没收敛"
BODYB = R30B - OFF
assert _body == rawm, "回执正文区与原始件不逐字节相等 ⇒ 表头说的不是这份 stdout"
CMP = [subprocess.run(["cmp", "-i", i, os.path.join(D, R30LOG), os.path.join(D, RAWM)],
                      cwd=REPO, capture_output=True).returncode
       for i in ("%d:0" % OFF, "%d:0" % (OFF + 1), "%d:1" % OFF)]
assert CMP == [0, 1, 1], "cmp 三联不齐（本件自己起进程跑的）：%s" % CMP
CMPT = " / ".join(str(x) for x in CMP)
FINGER_REC = one(r"^SRC指纹\s*:\s*([0-9a-f]{32})", r30, "回执表头的 SRC 指纹")
FINGER_NOW = sh("bash", "-c", "find src tests -name '*.py' -not -path '*__pycache__*' -print0"
                 " | sort -z | xargs -0 md5sum | md5sum").split()[0]
assert FINGER_REC == FINGER_NOW, "表头 SRC 指纹 %s 与此刻盘上现算 %s 不等 ⇒ 表头与树不是同一棵" % (
    FINGER_REC[:12], FINGER_NOW[:12])
_fl = []
for _n in ast.parse(open(GEN, encoding="utf-8").read()).body:
    if isinstance(_n, ast.Assign) and any(getattr(t, "id", None) == "FIELDS" for t in _n.targets):
        _fl.append(ast.literal_eval(_n.value))
assert len(_fl) == 1 and isinstance(_fl[0], list), "生成器里 FIELDS 声明不是恰好 1 处字符串列表"
FLDN = len(_fl[0])
after_txt = open(os.path.join(D, "_raw_s%d_hdr_after.out" % SLICE), encoding="utf-8").read()
AF_LAB, AF_DEC = [int(x) for x in one(r"^顶格字段行数＝ (\d+) / (\d+)", after_txt, "只读仪器的字段计数")]
assert (AF_LAB, AF_DEC) == (FLDN, FLDN), "档案字段 %d／声明 %d／本件现算 %d 三处不同值" % (
    AF_LAB, AF_DEC, FLDN)
HRAV = one(r"^HDR_AFTER_VERDICT = .*$", after_txt, "只读仪器判读行")

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
DL = int(one(r"本地独有 (\d+)/", DIFF3, "③ 本地独有"))
DR = int(one(r"远端独有 (\d+)/", DIFF3, "③ 远端独有"))
MSS = int(one(r"blob SHA 不符 (\d+)/", SHA4, "④ 全库 SHA 不符"))
MSD = int(one(r"blob SHA 不符 \d+/(\d+)", SHA4, "④ 的全库分母"))
MIS = int(one(r"远端缺失 (\d+)/", PER4, "④ 本片缺失"))
MMATCH = int(one(r"MATCH (\d+)/", PER4, "④ 本片 MATCH"))
MDEN = int(one(r"MATCH \d+/(\d+)", PER4, "④ 本片分母"))
assert BB == LB and TB == BB + BT, "远端 blob %d ≠ 本地 %d 或条目式不闭合（%d ≠ %d＋%d）" % (
    BB, LB, TB, BB, BT)
assert (RCNT, RNEW, RMODC) == (NTOT, NA, NM), "回执正文的本片文件数与 git 现算 总／A／M 不等"
CLOSEX = BB - (PB + NA)
assert CLOSEX == 0, "闭合式 远端 %d −（父 %d ＋ 新增 %d）＝ %d（须 0）" % (BB, PB, NA, CLOSEX)
assert (DL, DR, MSS, MIS) == (0, 0, 0, 0), "正文里那四个 0 此刻不是 0：%s" % ((DL, DR, MSS, MIS),)
assert MMATCH == MDEN == NTOT and MSD == BB, "本片 MATCH %d/%d 与全库分母 %d 对不上现算总数 %d" % (
    MMATCH, MDEN, MSD, NTOT)
ruler = md5(os.path.join(D, "am_s7_verify%d.py" % SLICE))
NCOPIES = len([p for p in glob.glob(os.path.join(REPO, "docs/evidence/*/am_s7_verify*.py"))
               if md5(p) == ruler])
NDIR = len([p for p in glob.glob(os.path.join(D, "am_s7_verify*.py")) if md5(p) == ruler])

# ---------------- 三面控件面（与写手同一批正则） ----------------
CTL = [
    ("MSG", "提交说明面", "am_s%d_msg_controls.py" % SLICE, "_raw_s%d_msg_controls.out" % SLICE,
     "_raw_s%d_msg_controls.err" % SLICE, r"^CASES 声明＝(\d+) 条｜本件自己的锚点普查", []),
    ("HDR", "表头面", "am_s%d_hdr_controls.py" % SLICE, "_raw_s%d_hdr_controls.out" % SLICE,
     "_raw_s%d_hdr_controls.err" % SLICE, r"^红路声明＝(\d+) 条", [r"红路声明＝\d+ 条｜保持绿＝(\d+) 条"]),
    ("AFTC", "只读仪器面", "am_s%d_hdr_after_controls.py" % SLICE,
     "_raw_s%d_hdr_after_controls.out" % SLICE, "_raw_s%d_hdr_after_controls.err" % SLICE,
     r"^CASES 声明＝(\d+) 条（全部红路）", [r"^(对照条数＝.*)$"]),
]
C = {}
for pfx, face, script, outf, errf, dec_pat, extras in CTL:
    op, ep = os.path.join(D, outf), os.path.join(D, errf)
    txt = open(op, encoding="utf-8").read()
    C[pfx + "_SCRIPT"], C[pfx + "_OUT"] = script, outf
    C[pfx + "_OUTB"], C[pfx + "_OUTM"] = str(os.path.getsize(op)), md5(op)[:12]
    C[pfx + "_ERRB"] = str(os.path.getsize(ep))
    assert os.path.getsize(ep) == 0, "%s 的 stderr 非 0 字节" % face
    C[pfx + "_DEC"] = one(dec_pat, txt, face + " 的声明条数")
    for pat in extras:
        C[pfx + ("_GRN" if "保持绿" in pat else "_PAIRS")] = one(pat, txt, face + " 的附加条数")
    C[pfx + "_UNC"] = one(r"本轮造不出红路的闸门＝(\d+) 条", txt, face + " 的造不出红路数")
    C[pfx + "_TO"] = one(r"超时（环境故障，不判红也不判过）＝(\d+) 条", txt, face + " 的超时数")
    assert C[pfx + "_TO"] == "0", "%s 有 %s 条超时＝环境故障 ⇒ 那一批读数不能当全过归档" % (
        face, C[pfx + "_TO"])
    ver = [l for l in txt.splitlines() if re.match(r"^CONTROL_VERDICT = .*", l)][-1]
    assert ver.endswith("ALL CONTROLS CAN GO RED"), "%s 判读行不是全红：%s" % (face, ver)
    assert not [l for l in txt.splitlines() if l.startswith("× ")], "%s 里有未命中的对照，不能写成全红" % face
    C[pfx + "_VERDICT"] = ver

SPNAME = "_raw_s%d_hdr_controls.attempt1-selfaudit-inverted" % SLICE
SPO, SPE = os.path.join(D, SPNAME + ".out"), os.path.join(D, SPNAME + ".err")
SPBO, SPBE = os.path.getsize(SPO), os.path.getsize(SPE)
SPWHY = one(r"^AssertionError: (.*)$", open(SPE, encoding="utf-8").read(), "现场件里的抛出行")
assert "抓不出不一致" in SPWHY, "现场件抛出的不是那条自审反转：%s" % SPWHY
_nr = one(r"^纸面预检通过＝.*?（红路 (\d+) 条＋绿路 (\d+) 条", open(SPO, encoding="utf-8").read(),
          "现场件首行声明的红路／绿路条数")
NLOST = str(int(_nr[0]) + int(_nr[1]))

# ---------------- 判据侧那处缺陷的现场（E6＝表头控件面第二次真跑） ----------------
SP2NAME = "_raw_s%d_hdr_controls.attempt2-notb-unsatisfiable" % SLICE
SP2O, SP2E = os.path.join(D, SP2NAME + ".out"), os.path.join(D, SP2NAME + ".err")
SP2BO, SP2BE = os.path.getsize(SP2O), os.path.getsize(SP2E)
_sp2 = open(SP2O, encoding="utf-8").read()
SP2MISS = len([l for l in _sp2.splitlines() if l.startswith("× ")])
assert SP2MISS == 1, "现场件的「× 」行＝%d（须恰好 1＝E6 那一条）" % SP2MISS
SP2VERDICT = one(r"^CONTROL_VERDICT = .*$", _sp2, "现场件的整面判读")
assert "PROBLEM" in SP2VERDICT, "现场件判读不含 PROBLEM（它就是收在判据空转上那一次）：%s" % SP2VERDICT
SP2RAISE = one(r"｜实得抛出行：(.*)$", _sp2, "未命中那行的实得抛出行")
HDR_LIVE = open(os.path.join(D, "_raw_s%d_hdr_controls.out" % SLICE), encoding="utf-8").read()
_live_miss = len([l for l in HDR_LIVE.splitlines() if l.startswith("× ")])
assert _live_miss == 0 and SP2MISS == 1, (
    "本次真跑「× 」行＝%d、现场件＝%d（须 0 与 1 成对）⇒ 那个 0 没有能红的搭档" % (_live_miss, SP2MISS))
_gh = open(os.path.join(D, "am_s%d_hdr.py" % SLICE), encoding="utf-8").read()
_bo = [n for n in ast.walk(ast.parse(_gh))
       if isinstance(n, ast.Raise) and isinstance(n.exc, ast.Call)
       and getattr(n.exc.func, "id", "") == "SystemExit"
       and n.exc.args and isinstance(n.exc.args[0], ast.Constant)
       and isinstance(n.exc.args[0].value, str) and "BODY_OFFSET" in n.exc.args[0].value]
assert len(_bo) == 1, "生成器里带 BODY_OFFSET 的 `raise SystemExit(str)` 不是恰好 1 条：%d" % len(_bo)
SP2LINE = str(_bo[0].lineno)
assert _bo[0].exc.args[0].value == SP2RAISE, "现场那句与生成器那条 raise 的参数不逐字符相等：%r／%r" % (
    SP2RAISE, _bo[0].exc.args[0].value)
assert not SP2RAISE.startswith("SystemExit:"), "现场那句自己就带前缀 ⇒「CPython 只打裸消息」这句是假话"


def table_counts(src, name):
    """按 ast 结构数判定表：整棵树走一遍（那张表缩进在 `if` 里）＋末位必须是布尔字面量。

    两个约束都是本轮实测逼出来的：只扫 `.body` 会「取不到」，`literal_eval` 会撞
    `_E6 + "\\n"` 这种拼接行当场 ValueError（这两条与落地件里那条同名尺子各自独立实现，
    两边都要能拒绝坏样本，否则数出来的 7／2／5 没有出处）。
    """
    hits = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    if len(hits) != 1:
        raise AssertionError("源码里 %s 赋值 %d 处（须恰好 1）⇒ 数出来的不是那一张表" % (name, len(hits)))
    v = hits[0].value
    if not isinstance(v, ast.List):
        raise AssertionError("%s 右边不是列表字面量而是 %s ⇒ 结构计数没有靶" % (name, type(v).__name__))
    acc = rej = 0
    for i, row in enumerate(v.elts, 1):
        if not isinstance(row, ast.Tuple) or not row.elts:
            raise AssertionError("%s 第 %d 行不是非空元组 ⇒ 行形制变了" % (name, i))
        last = row.elts[-1]
        if not (isinstance(last, ast.Constant) and isinstance(last.value, bool)):
            raise AssertionError("%s 第 %d 行末位不是 True/False ⇒ 分不清须收须拒" % (name, i))
        if last.value:
            acc += 1
        else:
            rej += 1
    return len(v.elts), acc, rej


# 这把尺子自己的正对照（纯内存、零配额）：嵌套块里的表要数得出，四种坏样本都要当场拒绝
assert table_counts('if x:\n    TT = [("a", _E + "\\n", 1, True), ("b", "y", 0, False)]\n',
                    "TT") == (2, 1, 1), "嵌套块里的表数不出 ⇒ 这把尺子还在只扫模块级"
for _bad, _why in (('TT = 3\n', "右边不是列表"), ('TT = [("a", "y", 1, 0)]\n', "末位不是布尔"),
                   ('TT2 = [("a", "y", 1, True)]\n', "名字缺席"),
                   ('TT = [("a", "y", 1, True)]\nTT = [("b", "y", 0, False)]\n', "同名两处赋值")):
    try:
        table_counts(_bad, "TT")
    except AssertionError:
        pass
    else:
        raise AssertionError("table_counts 没拒绝坏样本（%s）⇒ 它数出来的三个数不可信" % _why)
_ctl_src = open(os.path.join(D, "am_s%d_hdr_controls.py" % SLICE), encoding="utf-8").read()
SP2TAB, SP2ACC, SP2REJ = [str(x) for x in table_counts(_ctl_src, "NOTB_TABLE")]
assert int(SP2TAB) == int(SP2ACC) + int(SP2REJ) and int(SP2ACC) >= 1 and int(SP2REJ) >= 1, \
    "notb 判定表 %s 条／须收 %s／须拒 %s ⇒ 这张表有一头验不到" % (SP2TAB, SP2ACC, SP2REJ)
assert "ok = notb_hit(msg, rc, expect, gl)" in _ctl_src, "判定段还在原地写合取（没调谓词）⇒ 修复没吃进源码"
SELF_OLD = one(r"本面自查＝(\d+) 条", _sp2, "现场件声明的本面自查条数")
SELF_NEW = one(r"本面自查＝(\d+) 条", HDR_LIVE, "本次真跑的本面自查条数")
assert int(SELF_NEW) == int(SELF_OLD) + 1, "本面自查 %s 条 → %s 条（须恰好 ＋1＝新增那条谓词自查）" % (
    SELF_OLD, SELF_NEW)

# ============================ 五张档案面 ============================
TEXTS = dict(zip(FACE_NAMES, [open(p, encoding="utf-8").read() for p in FACE_PATHS]))
assert len(TEXTS) == 5, "档案面不到 5 张"
devtxt = TEXTS[FACE_NAMES[0]]
_secs = list(re.finditer(r"^### 26\.(\d+)[^\n]*$", devtxt, re.M))
assert _secs, "开发日志里取不到 `### 26.NN` ⇒ 节号是凭记忆的"
SECREF = _secs[-1].group(1)
assert "分片 %d" % SLICE in _secs[-1].group(0), "最后一节标题不含「分片 %d」：%s" % (
    SLICE, _secs[-1].group(0))
_seg = devtxt[_secs[-1].end():]
_nx = _seg.find("\n### ")
SEC_BODY = _seg[:_nx] if _nx != -1 else _seg
CLAIMS = [
    ("HEAD 全哈希", "HEAD＝`%s`（父 `%s`" % (HEAD, PAR40)),
    ("推送区间", "`%s..%s`" % (PAR7, HEAD7)),
    ("入库面 A／M／合计", "新增 (A) **%d**＋修改 (M) **%d**＝**%d** 件" % (NA, NM, NTOT)),
    ("shortstat 原文", "原文＝`%s`（**没有删除段**）" % SS),
    ("numstat 现算", "逐行现算＝**%d**、删除＝**%d**" % (NS_INS, NS_DEL)),
    ("生产树改动", "改动现数 **%d** 件" % NCODE),
    ("truncated", "`truncated=%s`" % TRUNC),
    ("blob＋tree 条目", "tree 条目 **%d**＝blob **%d**＋tree **%d**" % (TB, BB, BT)),
    ("本地／父提交 blob", "本地 blob＝%d、父提交 blob＝%d" % (LB, PB)),
    ("双向差集", "＝`%s`" % DIFF3),
    ("全库 SHA 普查", "`%s`" % SHA4),
    ("本片单列复核", "`%s`" % PER4),
    ("闭合式", "`%s`" % CLOS30),
    ("正对照 A", "`%s`" % CA),
    ("正对照 B", "`%s`" % CB),
    ("正对照 C", "`%s`" % CC),
    ("正对照 D", "`%s`" % CD),
    ("落地证明判读行", "判读行 `%s`" % V30),
    ("闭合式 git 复对", "%d − (%d ＋ %d) ＝ **%d**（须 0）" % (BB, PB, NA, CLOSEX)),
    ("远端与本地 blob 相等", "（%d＝%d）" % (BB, LB)),
    ("SRC 指纹前 12", "那枚前 12 位 `%s`" % FINGER_REC[:12]),
    ("回执三段字节", "档案名＝`%s` **%d** B／**%d** 行＝表头 **%d**＋正文 **%d**" % (
        R30LOG, R30B, R30L, OFF, BODYB)),
    ("cmp 三联", "三联＝`%s`" % CMPT),
    ("顶格字段与声明互证", "顶格字段 **%d/%d**" % (FLDN, FLDN)),
    ("核验件 stderr", "核验件 stderr **%d** B" % ERRMB),
    ("只读仪器判读行", "只读仪器判读行 `%s`" % HRAV),
    ("提交说明面字节", "%s 的 stdout＝%s **%s** B（md5 前 12 `%s`／stderr %s B）" % (
        C["MSG_SCRIPT"], C["MSG_OUT"], C["MSG_OUTB"], C["MSG_OUTM"], C["MSG_ERRB"])),
    ("提交说明面条数", "声明 **%s** 条｜超时 **%s** 条｜造不出红路登记 **%s** 条 ⇒「%s」" % (
        C["MSG_DEC"], C["MSG_TO"], C["MSG_UNC"], C["MSG_VERDICT"])),
    ("表头面字节与条数", "%s 的 stdout＝%s **%s** B（md5 前 12 `%s`／stderr %s B）｜红路声明 **%s** 条｜"
     "保持绿 **%s** 条｜超时 **%s** 条｜造不出红路 **%s** 条 ⇒「%s」" % (
         C["HDR_SCRIPT"], C["HDR_OUT"], C["HDR_OUTB"], C["HDR_OUTM"], C["HDR_ERRB"], C["HDR_DEC"],
         C["HDR_GRN"], C["HDR_TO"], C["HDR_UNC"], C["HDR_VERDICT"])),
    ("只读仪器面字节与条数", "%s 的 stdout＝%s **%s** B（md5 前 12 `%s`／stderr %s B）｜声明 **%s** 条｜%s｜"
     "超时 **%s** 条｜造不出红路 **%s** 条 ⇒「%s」" % (
         C["AFTC_SCRIPT"], C["AFTC_OUT"], C["AFTC_OUTB"], C["AFTC_OUTM"], C["AFTC_ERRB"], C["AFTC_DEC"],
         C["AFTC_PAIRS"], C["AFTC_TO"], C["AFTC_UNC"], C["AFTC_VERDICT"])),
    ("自审反转现场两件", "`%s/%s.out` **%s** B／`.err` **%s** B，抛出行「%s」" % (
        EVID_REL, SPNAME, SPBO, SPBE, SPWHY)),
    ("崩掉的批", "那 **%s** 条读数当场作废" % NLOST),
    ("判据侧现场两件", "`%s/%s.out` **%d** B／`.err` **%d** B，表里「× 」行 **%d** 条，整面判读＝%s" % (
        EVID_REL, SP2NAME, SP2BO, SP2BE, SP2MISS, SP2VERDICT)),
    ("判据侧的抛出出处", "第 **%s** 行那条不动点上限砍到 1，子运行 rc=1、stderr 末行逐字符等于那条 raise"
     " 的参数「%s」" % (SP2LINE, SP2RAISE)),
    ("notb 判定表", "（%s 个形状＝须收 **%s** 个＋须拒 **%s** 个，两个方向各验一遍）" % (
        SP2TAB, SP2ACC, SP2REJ)),
    ("本面自查条数的变化", "本面自查因此从 **%s** 条变成 **%s** 条" % (SELF_OLD, SELF_NEW)),
]
FACE_ONLY = [
    ("总览：三面控件面摘要句", "**三面控件面**：提交说明面 **%s** 条扰动／造不出红路 **%s** 条；表头面红路 "
     "**%s** 条／保持绿 **%s** 条／超时 **%s** 条／造不出红路 **%s** 条" % (
         C["MSG_DEC"], C["MSG_UNC"], C["HDR_DEC"], C["HDR_GRN"], C["HDR_TO"], C["HDR_UNC"])),
    ("计划：补记标题的落地腿定性句", "＝分片 %d 的**落地腿**：仓库推进到 `%s`／远端 blob **%d**" % (
        SLICE, HEAD7, BB)),
    ("纪实：替上一片入库那句", "落地腿＝替分片 %s 入库整条链（A **%d**／M **%d**＝**%d** 件）＋一条判断缺陷级"
     "新学＝%s" % (PREVSLICE, NA, NM, NTOT, LESSON_NEW)),
    ("索引：shortstat 内联句", "入库 A **%d**／M **%d**＝**%d** 件，shortstat「%s」＝零删除）" % (
        NA, NM, NTOT, SS)),
    ("索引：新学锚句", "新学＝%s（自审按构造红不了＝把自己的绿判成红" % LESSON_NEW),
]
CLAIMS += FACE_ONLY

# ---------------- 先量每面的独有声称数，再据此选「抽空白」的那根针 ----------------


def detector(claims, texts):
    return [(lab, lit) for lab, lit in claims if not any(lit in t for t in texts.values())]


MISSING = detector(CLAIMS, TEXTS)
assert not MISSING, "有 %d 条现取读数在档案里搜不到原样：%s" % (
    len(MISSING), "、".join("%s=%r" % (l, s[:44]) for l, s in MISSING))
_carriers = dict((lab, [n for n in FACE_NAMES if lit in TEXTS[n]]) for lab, lit in CLAIMS)
assert all(_carriers.values()), "有声称在五面里都搜不到"
COVERAGE = [(_n, len([lab for lab, c in _carriers.items() if _n in c]),
             len([lab for lab, c in _carriers.items() if c == [_n]])) for _n in FACE_NAMES]
COVERED = len([lab for lab, c in _carriers.items() if c])
assert COVERED == len(CLAIMS), "只有 %d/%d 条声称能在五面里搜到" % (COVERED, len(CLAIMS))
UNIQ_TOTAL = sum(u for _, _, u in COVERAGE)
assert 1 <= UNIQ_TOTAL <= len(CLAIMS), "独有数之和 %d 不在 1..%d" % (UNIQ_TOTAL, len(CLAIMS))
for _lab, _lit in CLAIMS:
    _n = len(_carriers[_lab])
    assert 1 <= _n <= 5, "%s 命中 %d 面（应当 1..5）" % (_lab, _n)

# ============================ 正对照：五内存扰动＋唯一性诱饵（全部不写盘） ============================
CTRL = []
_c1 = detector([(l, s + "◇") for l, s in CLAIMS], TEXTS)
assert len(_c1) == len(CLAIMS), "读数扰动后仍有 %d 条没判缺 ⇒ 回搜是空转" % (len(CLAIMS) - len(_c1))
CTRL.append("读数扰动（每条尾巴加一个档案里不可能出现的记号）⇒ 判缺 %d/%d" % (len(_c1), len(CLAIMS)))
_alt = HEAD[:-1] + ("0" if HEAD[-1] != "0" else "1")
_c2 = detector([("HEAD 全哈希", _alt)], TEXTS)
assert len(_c2) == 1 and _c2[0][0] == "HEAD 全哈希", "一位哈希扰动没让那一条红：%s" % _c2
CTRL.append("一位哈希扰动（HEAD 末位换值）⇒ 判缺 1/1")
_uniq_faces = [_n for _n, _c, _u in COVERAGE if _u > 0]
assert _uniq_faces, "五面全无独有声称 ⇒ 声称全同源，逐面抽空白这根针没有靶"
for _n in _uniq_faces:
    _c3 = detector(CLAIMS, dict(TEXTS, **{_n: ""}))
    _u = len([lab for lab, c in _carriers.items() if c == [_n]])
    assert len(_c3) == _u, "%s 抽成空文本判缺 %d 条，而它独有 %d 条 ⇒ 独有数与缺失数必须相等" % (
        _n, len(_c3), _u)
CTRL.append("逐面抽空白（只对独有数>0 的面要求红，实测 %d 面：\'%s\'）⇒ 各面缺失＝各自独有数" % (
    len(_uniq_faces), "／".join(_uniq_faces)))
_c4 = detector(CLAIMS, dict((n, "") for n in FACE_NAMES))
assert len(_c4) == len(CLAIMS), "五面全空却只判缺 %d 条 ⇒ detector 不依赖档案文本" % len(_c4)
CTRL.append("全集扰动（五面全换成空文本）⇒ 判缺 %d/%d" % (len(_c4), len(CLAIMS)))
_c5 = detector([("入库面 A／M／合计", "新增 (A) **%d**＋修改 (M) **9**＝**%d** 件" % (NA, NTOT))], TEXTS)
assert len(_c5) == 1, "把 M 改成档案里不存在的数字却没判缺 ⇒ 那条声称的三位数之间没有互相校验" % ()
CTRL.append("单值扰动（把 M 换成 %d 与现算 %d 不同的一个数）⇒ 判缺 1/1" % (9 if NM != 9 else 8, NM))

# ---------------- 分片 %s 的表头生成器要用的四条节内唯一性 ----------------
NEEDLES = (("引用本片回执名", re.escape(R30LOG), 1),
           ("带引号的钉住口径登记", r"当前「[^」]+」", 1),
           ("上一片的 LESSON 锚句", re.escape(LESSON_PREV), 1),
           ("本片的 LESSON 锚句", re.escape(LESSON_NEW), 1))
BAIT = {"引用本片回执名": R30LOG, "带引号的钉住口径登记": "当前「诱饵口径」",
        "上一片的 LESSON 锚句": "**%s**诱饵" % LESSON_PREV, "本片的 LESSON 锚句": "**%s**诱饵" % LESSON_NEW}
for what, pat, want in NEEDLES:
    n = len(re.findall(pat, SEC_BODY, re.M))
    assert n == want, "%s 在磁盘上的 §26.%s 里命中 %d 处（须恰好 %d）⇒ 分片 %s 的生成器会拒绝归档" % (
        what, SECREF, n, want, NEXTSLICE)
    n2 = len(re.findall(pat, SEC_BODY + ("\n- **诱饵**：见 %s。\n" % BAIT[what]), re.M))
    assert n2 == want + 1, "%s 的诱饵没造出 %d 处（现数 %d）⇒ 这条唯一性计数是恒真式" % (
        what, want + 1, n2)
CTRL.append("节内四条唯一性各自的诱饵（内存里再复述一次⇒各 +1）⇒ 同一判据当场可红")
_fs = r"f-string 把 `\{32\}` 当成\*\*替换字段"
assert len(re.findall(_fs, devtxt)) == 1, "f-string 事故句在整份日志里不是恰好 1 处 ⇒ 下一片的生成器本来就会拒绝"
CTRL.append("f-string 事故句全日志恰好 1 处（下一片生成器的硬前提）")
_lab = [m.group(1) for m in re.finditer(r"^- \*\*(.+?)\*\*", SEC_BODY, re.M)]
assert len(_lab) == len(set(_lab)) == 8, "§26.%s 顶级子条＝%d 条／互异 %d 条（须 7＋本片 1＝8）" % (
    SECREF, len(_lab), len(set(_lab)))
CTRL.append("§26.%s 顶级子条 8 条且标签互异（写手登记 7＋落地 1）" % SECREF)

# ---------------- 面内自洽：易失读数不现取，只核它自己那行的加法 ----------------
# 形制（从写手模板 DEV_T 的「待入库」子条现取）：未跟踪 **NU** 件＋修改 **NMOD** 件＝**TOT0** 处，
# 分组＝甲 **a**／乙 **b**／…（和＝a＋b＋…＝NU）。**分组和闭合的是「未跟踪」那一项，不是合计**
# ——修改面不进分组（本件写盘前把闭合式踩红过一回，就是这个数对错了对象）。
_pat = (r"现算未跟踪 \*\*(\d+)\*\* 件＋修改 \*\*(\d+)\*\* 件＝\*\*(\d+)\*\* 处，"
        r"分组＝(.*?)（和＝([0-9＋＝]+)）")
_h = re.findall(_pat, SEC_BODY)
assert len(_h) == 1, "§26.%s 里「待入库」那句形制命中 %d 处（须恰好 1）⇒ 下一片的现取会取错行" % (
    SECREF, len(_h))
_g1, _g2, _g3, _grp, _sum = _h[0]
assert int(_g1) + int(_g2) == int(_g3), "待入库那句自己的加法不闭合：%s＋%s≠%s" % (_g1, _g2, _g3)
_bnums = [int(x) for x in re.findall(r"\*\*(\d+)\*\*", _grp)]
_snums = [int(x) for x in re.findall(r"\d+", _sum)]
assert len(_snums) == len(_bnums) + 1 and sum(_snums[:-1]) == _snums[-1] == int(_g1), (
    "分组和 %s 的尾数 %s 应＝未跟踪 %s，且前 %d 项之和要等于它" % (
        _sum, _snums[-1:] or "（无）", _g1, len(_bnums)))
assert _bnums == _snums[:-1] and sum(_bnums) == int(_g1), "分组逐项 %s 与和式逐项 %s 不同 ⇒ 两处有一处是抄的" % (
    _bnums, _snums[:-1])
CTRL.append("待入库那句的**面内自洽**（未跟踪＋修改＝合计；各组和＝未跟踪项，逐组双向对）⇒ 通过")
# 诱饵：把合计改一个不可能出现的数 ⇒ 同一条自洽判据必须红（在内存里，不动盘）
_bait2 = _g3 + "9"
assert not (int(_g1) + int(_g2) == int(_bait2) and int(_g1) != int(_g3)), "合计诱饵没让加法判缺 ⇒ 那条自洽是恒真"
CTRL.append("合计诱饵（把 TOT0 尾巴加一位）⇒ 同一加法当场判缺")

# ---------------- volatile 读数：只断言，不入 stdout ----------------
st = [l for l in sh("git", "-c", "core.quotePath=false", "status", "--porcelain").splitlines() if l.strip()]
stat = {}
for l in st:
    stat[l[3:].strip()] = l[:2].strip()
for p in FACE_PATHS:
    rel = os.path.relpath(p, REPO)
    assert stat.get(rel) == "M", "%s 现在不是 (M)：%r ⇒ 落地写盘没留在工作树里" % (rel, stat.get(rel))
for p in (SELF_CLAIM,
          "docs/evidence/2026-10-12/_raw_s%d_archive_check.out" % SLICE,
          "docs/evidence/2026-10-12/_raw_s%d_archive_check.err" % SLICE):
    assert stat.get(p) == "??", "%s 现在不是未跟踪：%r" % (p, stat.get(p))
_pids = [l.split(",")[0].strip() for l in sh("nvidia-smi", "--query-compute-apps=pid",
                                             "--format=csv,noheader").splitlines() if l.strip()]


def _mine(pids):
    out = []
    for pid in pids:
        try:
            c = " ".join(open("/proc/%s/cmdline" % pid).read().split("\x00"))
        except OSError:
            continue
        if c and any(mk in c for mk in ("AM-Qoder", "warpblast", "/tmp/amvenv", "docs/evidence")):
            out.append(pid)
    return out


assert _mine(_pids) == [], "卡上有我的解释器留下的进程（%d 个）⇒ 归还口径破了" % len(_mine(_pids))
assert len(_mine([str(os.getpid())])) == 1, "匹配器数不出自己的 PID ⇒ 上面那个 0 是空转"
assert sh("nvidia-smi", "--query-gpu=name", "--format=csv,noheader").strip(), "nvidia-smi 连卡名都取不到"
for _n, _p in zip(FACE_NAMES, FACE_PATHS):
    assert open(_p, "rb").read()[-1:] == b"\n", "%s 末字节不是换行 ⇒ 下一次追加会粘连" % _n
    assert os.path.getsize(_p) > 0, "%s 是空文件" % _n
assert os.path.getsize(os.path.join(D, "_raw_s%d_landing_write.err" % SLICE)) == 0, (
    "写手那趟的 stderr 非 0 字节 ⇒ 有警告被吞")
print("归档自查（分片 %d 落地腿）＝把五张档案面里每条落地声称由命令现取再回搜原样" % SLICE)
print("  声称条数＝%d｜缺失＝0/%d｜每条命中面数 1..5 全部通过" % (len(CLAIMS), len(CLAIMS)))
for _c in CTRL:
    print("  正对照：%s" % _c)
for _n, _c, _u in COVERAGE:
    print("  承载：%s＝%d 条（其中独有 %d 条）" % (_n, _c, _u))
print("  §26.%s 顶级子条＝%d 条；本片回执名／钉住口径／上一片 LESSON／本片 LESSON 各恰好 1 条且各配诱饵" % (
    SECREF, len(_lab)))
print("  回执正文区与 `%s` 逐字节相等（`cmp -i %d:0` 三联 %s 由本件自跑）；BODY_OFFSET 定点复解＝%d；"
      "SRC 指纹与此刻盘上现算同值" % (RAWM, OFF, CMPT, OFF))
print("  尺子 `am_s7_verify%d.py` 全树 %d 份逐字节相同（本目录 %d 份）" % (SLICE, NCOPIES, NDIR))
print("  volatile 读数（未入库件数、卡上进程数、落笔 date）按口径只断言、不入 stdout ⇒ 本件可复跑逐字节相同")
print("  闭合式复核：远端 %d −（父 %d ＋ 新增 %d）＝%d；入库清单 %d 件与 numstat %d 行对齐、删除 %d" % (
    BB, PB, NA, CLOSEX, NTOT, len(NSR), NS_DEL))
print("ARCHIVE_VERDICT = ALL CLAIMS LIVE AND PRESENT, ALL CONTROLS CAN GO RED")
