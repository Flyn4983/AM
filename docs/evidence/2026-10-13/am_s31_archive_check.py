#!/usr/bin/env python3
r"""归档自查（分片 31 的落地半腿收尾件）：本轮**五张档案面**里每一条落地声称，由命令现取一次再回搜原样。

本件与 `am_s31_docs_landing.py`（写手）的分工＝尺子同源：下面「现取」那一段的**命令与正则**逐字节切自写手
（读数 1／2／3／4／6 五段），所以「档案里写的」与「本件重新现取的」是同一把尺子；改了格式两边一起红，
不会出现一个改了一个没改。写手自己那套结构闸门在**落笔前**跑，本件在**落笔后**跑，两趟之间隔着真实写盘。

范围照实说清：本件普查「落地读数」这张清单（HEAD／父／提交时刻／入库 A-M-合计／shortstat／numstat／
生产树改动／blob＋tree 条目／闭合式与 git 复对差／五个带分母的 0／四个正对照／判读行／回执三段字节／
`cmp` 三联／顶格字段与 `FIELDS` 互证／三方字节同值／只读仪器的复现腿／三面控件面各自的字节-md5-声明-
保持绿-基线-整批-超时-自查-造不出红路-判读行／ast 表长闭合／三面同形制的按行现数／六件红灯现场与探针
每件自己的字节-抛出行-未命中行数-作废条数／GPU 那把坏尺子的 rc-首行-内核-用户态／替代断言现数的 0／
五面各自的专有句）。**不是**对归档散文做全量出处普查。

四条口径（前三条继承上一片，第四条本片新增）：
① volatile 读数（未入库件数、PID 分母、落笔 `date`）**只断言、不入 stdout**，否则「本件复跑逐字节相同」做不到；
   散文里那些带日期的数由「面内自洽」那条检查（未跟踪＋新增＋修改＝合计；分组和＝新增＋未跟踪）核，不由现取核。
② 每个 0 配一条**同一次运行里能红**的正对照：五内存扰动（读数加尾记号／一位哈希扰动／逐面抽空白／全集抽
   空白／单值扰动）＋节内六条唯一性各自的诱饵＋分组针＋`argv[0]` 匹配器的一正一负合成对照。
③ 下一片的表头生成器要在 §26.%s 里现取**唯一**一条顶级子条（引用本片回执名／上一片回执名／带引号的钉住
   口径／两枚承接锚句／本片 LESSON），本件在盘上重数一次（**按顶级子条**，与下游 `bullet_for` 同问法），
   并各配一根「再复述就 +1」的诱饵。
④ **本片新增**：落笔那句「最终入库数大于这里的 %(TOT0预告)s」是一条**会被下一片证伪**的预测，本件在盘上
   把那个数解出来、与此刻的待入库现数比一次（此刻数就须大于它），并配一根「把预告值 +100 就红」的对照——
   上一片那次复跑红在 `SECREF` 随节号漂（本件的节号是现取的，不是写死的）。

运行＝`CUDA_VISIBLE_DEVICES="" /tmp/amvenv/bin/python -B docs/evidence/2026-10-13/am_s31_archive_check.py`
（不 import jax：只跑 git／md5sum／cmp／nvidia-smi／只读仪器复跑／读文件。）
"""
import ast
import glob
import hashlib
import os
import re
import subprocess

REPO = "/home/shy/桌面/2026年开发-显式冲击有限元-物质点法-sph-fvm+am/AM-Qoder"
EVID_REL = "docs/evidence/2026-10-13"
D = os.path.join(REPO, EVID_REL)
DEVLOG = os.path.join(REPO, "docs/开发日志.md")
OVERVIEW = os.path.join(REPO, "docs/项目开发总览.md")
PLAN = os.path.join(REPO, "docs/项目评估与下一步计划_2026-10-06.md")
DAILY = os.path.join(REPO, ".workbuddy/memory/2026-10-13.md")
WBIDX = os.path.join(REPO, ".workbuddy/memory/MEMORY.md")
FACE_NAMES = ["docs/开发日志.md", "docs/项目开发总览.md", "docs/项目评估与下一步计划_2026-10-06.md",
              ".workbuddy/memory/2026-10-13.md", ".workbuddy/memory/MEMORY.md"]
FACE_PATHS = [DEVLOG, OVERVIEW, PLAN, DAILY, WBIDX]
PY3 = "/tmp/amvenv/bin/python"
INTERP = "/tmp/amvenv/bin/python"
SELF_CLAIM = "docs/evidence/2026-10-13/am_s31_archive_check.py"
LESSON_NEW = "良性扰动的等长约束"
CARRIED = ("自审的方向性", "普查仪器没有第二出处")
SP1 = "_raw_s31_msg_controls.attempt1-mark-line-hits-twice"
SP2 = "_raw_s31_msg_controls.attempt2-twelve-controls-idled"
SP3 = "_raw_s31_hdr_controls.attempt1-quota-single-read"
SP4 = "_raw_s31_hdr_controls.attempt2-f1-interp-contam-r8-repro-idled"
SP5 = "_raw_s31_hdr_controls.probe-ruler-403"
SP6 = "_raw_s31_hdr_after_controls.attempt1-greenloop-unbound-var"
SP7 = "_raw_s31_hdr_after_controls.probe-headlen-plus2-reds-at-writeface-bytes"
COPY_PREFIX = "zctl_s31_"
AFTC_COPY_PREFIX = "zctl_s31_after_"


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
    """异常**末行**（不是消息行）：家族形制取自写手，判据读的就是这一行。"""
    cand = [l for l in text.splitlines()
            if re.search(r"(AssertionError|UnboundLocalError|TypeError|ValueError|SystemExit"
                         r"|SyntaxError|FileNotFoundError|Error)\b", l) and not l.startswith("File ")]
    assert cand, "%s 里取不到异常行 ⇒ 该件不是收在一条可读的抛出上" % what
    return cand[-1].strip()


def list_lens(path, names=("CASES", "GREENS", "UNCOVERED")):
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
    assert not missing, "%s 里取不到 %s ⇒ 该件的表形制变了" % (
        os.path.basename(path), "／".join(missing))
    return got


def top_bullets(seg):
    ms = list(re.finditer(r"^- \*\*(.+?)\*\*[^\n]*$", seg, re.M))
    out = []
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(seg)
        out.append((m.group(1), seg[m.start():end]))
    return out


# ============================ 读数 1：git 现算（切自写手） ============================
HEAD = sh("git", "rev-parse", "HEAD").strip()
HEAD7 = sh("git", "rev-parse", "--short", "HEAD").strip()
PAR40 = sh("git", "rev-parse", "HEAD^").strip()
PAR7 = sh("git", "rev-parse", "--short", "HEAD^").strip()
SUBJ = sh("git", "log", "-1", "--format=%s").strip()
CDATE = sh("git", "log", "-1", "--format=%cd", "--date=iso").strip()
NCODE = len([l for l in sh("git", "-c", "core.quotePath=false", "status", "--porcelain", "-uall",
                           "--", "src", "tests").splitlines() if l.strip()])
assert NCODE == 0, "生产树 src＋tests 有 %d 处改动 ⇒ 本片不是纯归档腿" % NCODE
SLICE = int(one(r"S7 分片 (\d+)", SUBJ, "HEAD 提交说明里的片号"))
PREVSLICE, NEXTSLICE = str(SLICE - 1), str(SLICE + 1)
_nums = sorted({int(x) for x in re.findall(r"分片 (\d+)", SUBJ)})
assert _nums == [SLICE - 1, SLICE], (
    "HEAD 提交说明里的片号集合＝%s，须恰好＝[%d, %d]（本片／它入库的那一片）⇒ 片号方向没被双向钉住" % (
        _nums, SLICE - 1, SLICE))
for _f in ("am_s7_verify%d.log" % SLICE, "_raw_verify%d.out" % SLICE, "am_s%d_hdr.py" % SLICE,
           "am_s%d_hdr_after.py" % SLICE, "am_s%d_hdr_after_controls.py" % SLICE,
           "am_s%d_docs_landing.py" % SLICE):
    assert os.path.exists(os.path.join(D, _f)), "按片号 %d 拼出的 `%s` 不在盘上 ⇒ 片号指错了对象" % (SLICE, _f)
assert EVID_REL == "docs/evidence/%s" % CDATE[0:10], (
    "证据目录 %s 与 HEAD 提交日期 %s 不同日 ⇒ 本件锚的是上一轮的目录" % (EVID_REL, CDATE[0:10]))
for _n in (SP1, SP2, SP3, SP4, SP5, SP6, SP7):
    assert _n.startswith("_raw_s%d_" % SLICE), "现场件 %s 的片号前缀与本片 %d 不符" % (_n, SLICE)

_ns = [l for l in sh("git", "show", "--name-status", "--format=", "HEAD").splitlines() if l.strip()]
_r = [l.split("\t") for l in _ns if "\t" in l]
assert len(_r) == len(_ns) and all(len(x) == 2 for x in _r), "name-status 行形制不是「状态字母 TAB 路径」"
KINDS = [x[0] for x in _r]
CPATHS = [x[-1] for x in _r]
assert set(KINDS) <= {"A", "M"}, "HEAD 清单里有 A／M 之外的状态：%s" % sorted(set(KINDS))
NAC, NMC = KINDS.count("A"), KINDS.count("M")
NTOTC = len(CPATHS)
assert len(set(CPATHS)) == NTOTC, "HEAD 清单里有重名路径"
NSR = [l for l in sh("git", "show", "--numstat", "--format=", "HEAD").splitlines() if l.strip()]
assert len(NSR) == NTOTC, "numstat 行数 %d 与 name-status %d 不等 ⇒ 两个口径在数不同的东西" % (len(NSR), NTOTC)
assert all("\t" in l and l.split("\t")[0] != "-" for l in NSR), \
    "numstat 里有 `-`（二进制／rename）⇒ 逐行求和口径不适用"
NS_INS = sum(int(l.split("\t")[0]) for l in NSR)
NS_DEL = sum(int(l.split("\t")[1]) for l in NSR)
SS_RAW = sh("git", "diff-tree", "--no-commit-id", "--shortstat", "-r", "HEAD").strip()
SS_FILES = int(one(r"(\d+) files? changed", SS_RAW, "shortstat 的文件数"))
SS_INS = int(one(r"(\d[\d,]*) insertions?", SS_RAW, "shortstat 的插入数"))
assert "deletion" not in SS_RAW, "shortstat 里有删除段 ⇒ 「零删除」那句是假话：%s" % SS_RAW
assert (SS_FILES, SS_INS) == (NTOTC, NS_INS) and NS_DEL == 0, (
    "shortstat (%d, %d) 与 numstat 现算 (%d, %d)／删除 %d 不闭合" % (
        SS_FILES, SS_INS, NTOTC, NS_INS, NS_DEL))
assert ".workbuddy/memory/%s.md" % CDATE[0:10] in CPATHS, "当日纪实不在 HEAD 清单里 ⇒ 上一片写盘那张面没入库"

# ============================ 读数 2：回执正文与盘上字节（切自写手） ============================
RLOG = "am_s7_verify%d.log" % SLICE
raw_log = open(os.path.join(D, RLOG), "rb").read()
r30 = raw_log.decode()
RLINES = len(r30.splitlines())
RAWM = "_raw_verify%d.out" % SLICE
ERRM = "_raw_verify%d.err" % SLICE
rawm = open(os.path.join(D, RAWM), encoding="utf-8").read()
ERRMB = os.path.getsize(os.path.join(D, ERRM))
assert ERRMB == 0, "核验件的 stderr 不是 0 字节（%d）⇒ 有警告被吞" % ERRMB
GEN = os.path.join(D, "am_s%d_hdr.py" % SLICE)
OFF = int(one(r"^BODY_OFFSET=(\d+)$", r30, "本片回执的 BODY_OFFSET"))
_marker = "BODY_OFFSET=%d\n" % OFF
assert r30.count(_marker) == 1, "BODY_OFFSET 那行不是恰好 1 条"
_hp, _body = r30.split(_marker, 1)
assert len((_hp + _marker).encode("utf-8")) == OFF, "BODY_OFFSET 与实际表头字节数不等（定点没收敛）"
BODYB = len(raw_log) - OFF
assert _body == rawm, "回执正文区与原始件不逐字节相等 ⇒ 表头说的不是这份 stdout"
HDNL = len([l for l in _hp.splitlines() if l.strip()])
_mh = re.search(r"head=([0-9a-f]{40}) parent=([0-9a-f]{40})", _hp)
assert _mh and _mh.group(1) == HEAD and _mh.group(2) == PAR40, (
    "回执记的 HEAD／父与 git 现算不符 ⇒ 归档讲的不是当前这棵树")
FINGER_REC = one(r"^SRC指纹\s*:\s*([0-9a-f]{32})", r30, "回执表头的 SRC 指纹")
FINGER_NOW = sh("bash", "-c", "find src tests -name '*.py' -not -path '*__pycache__*' -print0"
                 " | sort -z | xargs -0 md5sum | md5sum").split()[0]
assert FINGER_REC == FINGER_NOW, "表头 SRC 指纹 %s 与盘上现算 %s 不等 ⇒ 表头与树不是同一棵" % (
    FINGER_REC[:12], FINGER_NOW[:12])
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
DL = int(one(r"本地独有 (\d+)/", DIFF3, "③ 本地独有"))
DR = int(one(r"远端独有 (\d+)/", DIFF3, "③ 远端独有"))
MSS = int(one(r"blob SHA 不符 (\d+)/", SHA4, "④ 全库 SHA 不符"))
MIS = int(one(r"远端缺失 (\d+)/", PER4, "④ 本片缺失"))
MMATCH, MDEN = [int(x) for x in one(r"MATCH (\d+)/(\d+)", PER4, "④ 本片 MATCH 与分母")]
assert BB == LB and TB == BB + BT, "远端 blob %d ≠ 本地 %d 或条目式不闭合" % (BB, LB)
assert (RCNT, RNEW, RMODC) == (NTOTC, NAC, NMC), "回执正文的本片文件数与 git 现算 总／A／M 不等"
CLOSEX = BB - (PB + NAC)
assert CLOSEX == 0, "闭合式 远端 %d −（父 %d ＋ 新增 %d）＝ %d（须 0）" % (BB, PB, NAC, CLOSEX)
assert (DL, DR, MSS, MIS, MMATCH - MDEN) == (0, 0, 0, 0, 0), (
    "正文里那五个 0 此刻不是 0（差集／全库 SHA／本片缺失／MATCH−分母）：%s" % ((DL, DR, MSS, MIS, MMATCH, MDEN),))
assert MMATCH == NTOTC, "本片 MATCH %d 与 git 现算入库数 %d 不等" % (MMATCH, NTOTC)
ruler = md5(os.path.join(D, "am_s7_verify%d.py" % SLICE))
NCOPIES = len([p for p in glob.glob(os.path.join(REPO, "docs/evidence/*/am_s7_verify*.py")) if md5(p) == ruler])
assert NCOPIES >= 11, "尺子的逐字节副本全树只有 %d 份 ⇒ 分片 31 登记的那一族（≥11）散了" % NCOPIES

# 顶格字段数：一面从生成器的 FIELDS 声明（ast 现取），一面从只读仪器的 stdout
_fl = []
for _n in ast.walk(ast.parse(open(GEN, encoding="utf-8").read())):
    if isinstance(_n, ast.Assign) and any(getattr(t, "id", None) == "FIELDS" for t in _n.targets):
        _v = ast.literal_eval(_n.value)
        assert isinstance(_v, list) and all(isinstance(x, str) for x in _v), "FIELDS 不是字符串列表"
        _fl.append(_v)
assert len(_fl) == 1, "生成器里 FIELDS 赋值有 %d 处 ⇒ 声明出处不唯一" % len(_fl)
FLDN = len(_fl[0])
AFT_OUT = "_raw_s%d_hdr_after.out" % SLICE
aft = open(os.path.join(D, AFT_OUT), encoding="utf-8").read()
AFT_AOFF = int(one(r"^BODY_OFFSET（从盘上现读）＝ (\d+)", aft, "只读仪器从盘上现读的偏移"))
assert AFT_AOFF == OFF, "只读仪器现读偏移 %d ≠ 表头定点 %d ⇒ 两把尺子读的是不同的数" % (AFT_AOFF, OFF)
CMP3 = one(r"^cmp -i 三联（本件自己起进程跑的，不是抄生成器的）＝ (\d+ / \d+ / \d+)", aft, "只读仪器的 cmp 三联")
assert CMP3 == "0 / 1 / 1", "cmp 三联不是 0/1/1：%s" % CMP3
CMP_SELF = [subprocess.run(["cmp", "-i", i, os.path.join(D, RLOG), os.path.join(D, RAWM)],
                           cwd=REPO, capture_output=True).returncode
            for i in ("%d:0" % OFF, "%d:0" % (OFF + 1), "%d:1" % OFF)]
assert CMP_SELF == [0, 1, 1], "本件自己起的 cmp 三联不齐：%s" % CMP_SELF
assert " / ".join(str(x) for x in CMP_SELF) == CMP3, "本件自跑的三联与归档件现取的三联不同值 ⇒ 两边读的不是同一对档案"
AFT_FLD = [int(x) for x in one(r"^顶格字段行数＝ (\d+) / (\d+)", aft, "只读仪器的字段计数")]
assert AFT_FLD == [FLDN, FLDN], "档案字段 %s／声明 %d 不同值" % (AFT_FLD, FLDN)
WC1, WC2, WC3, WFD, WDEC, WBYTES = [x for x in one(
    r"^两出处闭合＝写盘面\((\d+)／(\d+)／(\d+)，字段 (\d+)/(\d+)，(\d+) bytes\) 与本件\(盘上现解\) 同值$",
    aft, "只读仪器与写盘面那两出处")]
assert (int(WC1), int(WC2), int(WC3)) == (0, 1, 1), "写盘面三联不是 0/1/1"
WF_OUT = "_raw_s%d_hdr_write.out" % SLICE
wft = open(os.path.join(D, WF_OUT), encoding="utf-8").read()
assert int(WBYTES) == len(raw_log) == int(one(r"^写盘完成： \S+ (\d+) bytes", wft, "写盘面报出的字节数")), (
    "写盘面、盘上现读、只读仪器三方字节数不同值")
AFT_NFACES = int(one(r"^输入面 md5 跑前后逐一相同＝True（(\d+) 张面）", aft, "只读仪器读入的面数"))
assert AFT_NFACES == 4, "只读仪器的输入面数＝%d（须 4）⇒ 那张面的形制变了" % AFT_NFACES
HRAV = one(r"^HDR_AFTER_VERDICT = .*$", aft, "只读仪器判读行")
AFTB = os.path.getsize(os.path.join(D, AFT_OUT))

# 复现腿：**本件自己**起两趟只读仪器，与归档件三方逐字节比（写手那趟已经跑过，这里是第二次独立复现）
_tmp = []
try:
    for i in (1, 2):
        p = subprocess.run([PY3, "-B", os.path.join(D, "am_s%d_hdr_after.py" % SLICE)],
                           cwd=D, capture_output=True, text=True,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), timeout=180)
        assert p.returncode == 0, "只读仪器复跑 rc=%s ⇒ 归档那句「可复现」不成立" % p.returncode
        q = os.path.join("/tmp", "s31_after_repro_check%d.out" % i)
        open(q, "w", encoding="utf-8").write(p.stdout)
        _tmp.append(q)
    REPRO = (subprocess.run(["cmp", "-s", _tmp[0], _tmp[1]]).returncode == 0
             and subprocess.run(["cmp", "-s", _tmp[0], os.path.join(D, AFT_OUT)]).returncode == 0)
finally:
    for q in _tmp:
        os.path.exists(q) and os.remove(q)
assert REPRO is True, "本件复跑两趟与归档 stdout 不是三方逐字节相同 ⇒ 那句 True 是假话"
assert os.path.getsize(os.path.join(D, "_raw_s%d_landing_write.err" % SLICE)) == 0, "写腿 stderr 非 0"
assert os.path.getsize(os.path.join(D, "_raw_s%d_landing_precheck.err" % SLICE)) == 0, "预检 stderr 非 0"
_res = [p for p in os.listdir(D) if p.startswith(COPY_PREFIX) or p.startswith(AFTC_COPY_PREFIX)
        or p.startswith("zprobe_")]
assert not _res, "档案目录里还有扰动副本的残留：%s" % _res[:3]

# ============================ 读数 3：三面控件面（与写手同一批正则） ============================
CTL = [
    ("MSG", "提交说明面", "am_s%d_msg_controls.py" % SLICE, "_raw_s%d_msg_controls.out" % SLICE,
     "_raw_s%d_msg_controls.err" % SLICE,
     r"^CASES 声明＝(\d+) 条（全部红路，其中克隆别的仪器 (\d+) 条）｜保持绿＝(\d+) 条｜"
     r"本件自己的锚点普查＝(\d+) 条｜超时（环境故障，不判红也不判过）＝(\d+) 条｜因超时没跑到的＝(\d+) 条$",
     {"DEC": 0, "CLONE": 1, "GRN": 2, "SELF": 3, "TO": 4}, "keep"),
    ("HDR", "表头面", "am_s%d_hdr_controls.py" % SLICE, "_raw_s%d_hdr_controls.out" % SLICE,
     "_raw_s%d_hdr_controls.err" % SLICE,
     r"^红路声明＝(\d+) 条｜保持绿＝(\d+) 条（判据 rc=0 且打印出预检标记那行）｜基线＝(\d+) 条"
     r"（不进表，红在开头即整面作废）｜环境故障（不判红也不判过）＝(\d+) 条（按去向细分：无）｜本面自查＝(\d+) 条",
     {"DEC": 0, "GRN": 1, "BASE": 2, "TO": 3, "SELF": 4}, "keep"),
    ("AFTC", "只读仪器面", "am_s%d_hdr_after_controls.py" % SLICE,
     "_raw_s%d_hdr_after_controls.out" % SLICE, "_raw_s%d_hdr_after_controls.err" % SLICE,
     r"^红路声明＝(\d+) 条｜保持绿＝(\d+) 条（逐条 rc=0 且打印出判读行；含裸跑基线 1 条）｜实跑行数＝(\d+) 条｜"
     r"超时（环境故障，不判红也不判过）＝(\d+) 条（无）｜装配失败（锚点问题，不是红路）＝(\d+) 条（无）｜"
     r"造不出红路的闸门＝(\d+) 条$",
     {"DEC": 0, "GRN": 1, "ROWS": 2, "TO": 3, "ASM": 4}, "plus1"),
]
C = {}
for pfx, face, script, outf, errf, dec_pat, slot, kind in CTL:
    op, ep = os.path.join(D, outf), os.path.join(D, errf)
    txt = open(op, encoding="utf-8").read()
    sp = os.path.join(D, script)
    g = one(dec_pat, txt, face + " 的汇总行")
    for key, idx in slot.items():
        C["%s_%s" % (pfx, key)] = g[idx]
    C[pfx + "_SCRIPT"], C[pfx + "_OUT"] = script, outf
    C[pfx + "_OUTB"] = str(os.path.getsize(op))
    C[pfx + "_OUTM"] = md5(op)[:12]
    C[pfx + "_ERRB"] = str(os.path.getsize(ep))
    assert os.path.getsize(ep) == 0, "%s 的 stderr 非 0 字节" % face
    assert len(rows(txt, r"^(× |！)")) == 0, "%s 的归档 stdout 里有未命中／环境行 ⇒ 那批读数不可引用" % face
    ver = last(r"^(CONTROL_VERDICT = .*)$", txt, face + " 的判读行")
    assert ver.endswith("ALL CONTROLS CAN GO RED"), "%s 判读行不是全过声明：%s" % (face, ver)
    C[pfx + "_VERDICT"] = ver
    C[pfx + "_UNC"] = one(r"本轮造不出红路的闸门＝(\d+) 条", txt, face + " 的造不出红路数")
    L = list_lens(sp)
    assert int(C[pfx + "_DEC"]) == L["CASES"], "%s：stdout 声明 %s ≠ 源码 CASES %d ⇒ 归档件比源码旧" % (
        face, C[pfx + "_DEC"], L["CASES"])
    assert int(C[pfx + "_GRN"]) == L["GREENS"] + (1 if kind == "plus1" else 0), (
        "%s：stdout 保持绿 %s ≠ 源码 GREENS %d ⇒ 归档件比源码旧" % (face, C[pfx + "_GRN"], L["GREENS"]))
    assert int(C[pfx + "_UNC"]) == L["UNCOVERED"], "%s：stdout 造不出红路 %s ≠ 源码 UNCOVERED %d" % (
        face, C[pfx + "_UNC"], L["UNCOVERED"])
    if kind == "plus1":
        assert int(C[pfx + "_ROWS"]) == L["CASES"] + L["GREENS"] + 1, "%s：实跑 %s ≠ 红＋绿＋基线" % (
            face, C[pfx + "_ROWS"])
    if "SELF" not in slot:
        C[pfx + "_SELF"] = one(r"本面自查＝(\d+) 条", txt, face + " 的本面自查条数")
ASTCLOSE = "／".join("%s＝OK" % p for p, _f, _s, _o, _e, _d, _sl, _k in CTL)
PER_LINE = []
for key in ("保持绿＝", "本轮造不出红路的闸门＝", "CONTROL_VERDICT ="):
    per = [len([l for l in open(os.path.join(D, f), encoding="utf-8").read().splitlines() if key in l])
           for _p, _f, _s, f, _e, _d, _sl, _k in CTL]
    assert per == [1, 1, 1], "字段名「%s」在三面归档 stdout 里按行现数 %s（须每面恰好 1 行）⇒ 形制漂了" % (key, per)
    PER_LINE = per
assert len(PER_LINE) == 3, "按行现数没攒够三枚（现数 %s）⇒ 那句「＝1/1/1」没有出处" % PER_LINE
hdr_txt = open(os.path.join(D, "_raw_s%d_hdr_controls.out" % SLICE), encoding="utf-8").read()
_HLIM = one(r"^配额闸门起点＝/rate_limit 现取余量（易失读数，不作下一轮分母）＝(\d+) 次｜单窗口上限＝(\d+) 次｜"
            r"本批子运行＝(\d+) 条", hdr_txt, "表头面的配额起点行")
C["HDR_BATCH"] = _HLIM[2]
assert int(C["HDR_BATCH"]) == int(C["HDR_DEC"]) + int(C["HDR_GRN"]), (
    "表头面基线外整批 %s ≠ 红 %s＋绿 %s" % (C["HDR_BATCH"], C["HDR_DEC"], C["HDR_GRN"]))
_w = one(r"^配额闸门收面＝现取余量（易失，不作分母）＝\d+ 次｜单窗口上限（现取）＝\d+ 次｜等待触发 (\d+) 次｜"
         r"等待累计 (\d+) 秒$", hdr_txt, "表头面的配额收面行")
C["HDR_WAITN"], C["HDR_WAITS"] = _w
assert C["MSG_TO"] == C["HDR_TO"] == C["AFTC_TO"] == C["AFTC_ASM"] == "0", (
    "三面里出现超时／环境／装配失败读数 ⇒ 那一批不能按全过引用")

# ============================ 读数 4：六件红灯现场与探针（切自写手） ============================
def specimen(alias, name, why_pat=None, miss=False, verdict=False):
    op, ep = os.path.join(D, name + ".out"), os.path.join(D, name + ".err")
    for p in (op, ep):
        assert os.path.exists(p), "现场件不在档：%s" % p
    ot, et = open(op, encoding="utf-8").read(), open(ep, encoding="utf-8").read()
    S[alias] = name
    S[alias + "_B"] = str(os.path.getsize(op))
    S[alias + "_E"] = str(os.path.getsize(ep))
    if why_pat:
        r = raised(et if et else ot, name)
        assert re.search(why_pat, r), "现场件 %s 的抛出行不含预期形状（须/%s/），实得：%s" % (name, why_pat, r[:120])
        S[alias + "_WHY"] = r
    if miss:
        S[alias + "_MISS"] = str(len(rows(ot, r"^× ")))
    if verdict:
        S[alias + "_VERDICT"] = one(r"^(CONTROL_VERDICT = .*)$", ot, name + " 的整面判读")
    return ot, et


S = {}
specimen("SP1", SP1, why_pat=r"基线标记句在副本 stdout 里命中")
specimen("SP2", SP2, miss=True, verdict=True)
assert S["SP2_MISS"] == "12", "现场 %s 的「× 」行现数 %s 条（档案那句写的是 12 条）" % (SP2, S["SP2_MISS"])
_sp3o, _sp3e = specimen("SP3", SP3, why_pat=r"读不到 GitHub 配额响应头")
S["SP3_ROWS"] = str(len(rows(_sp3o, r"^  [A-Z][A-Z]?\d+ ")))
_sp4o, _sp4e = specimen("SP4", SP4, miss=True, verdict=True)
_sp4rows = rows(_sp4o, r"^× ")
S["SP4_RAISES"] = "／".join(one(r"｜实得抛出行：(.*)$", l, "现场 %s 的一条实得抛出行" % SP4) for l in _sp4rows)
assert len(_sp4rows) == 2, "现场 %s 的「× 」行现数 %d（须 2＝F1 与 R8 那两条）" % (SP4, len(_sp4rows))
assert S["SP4_MISS"] == "2", "现场 %s 未命中数现数 %s（档案写 2）" % (SP4, S["SP4_MISS"])
_sp5o, _sp5e = specimen("SP5", SP5, why_pat=r"复跑 stdout 与归档正文不同")
assert os.path.getsize(os.path.join(D, SP5 + ".out")) == 0, "探针 %s 的 stdout 不是 0 B ⇒ 那句「死在打印之前」是假话" % SP5
_gsrc = open(GEN, encoding="utf-8").read()
_mrepro = one(r'^assert REPRO, "(.*)"', _gsrc, "生成器那条复现断言")
assert "AssertionError: " + _mrepro == S["SP5_WHY"], "探针抛出行与生成器那条断言不逐字符相等：%s" % S["SP5_WHY"]
S["SP5_LINE"] = str([n.lineno for n in ast.walk(ast.parse(_gsrc))
                     if isinstance(n, ast.Assert) and isinstance(n.test, ast.Name)
                     and n.test.id == "REPRO"][0])
_sp6o, _sp6e = specimen("SP6", SP6, why_pat=r"UnboundLocalError")
_sp6red, _sp6grn = [int(x) for x in one(r"^纸面预检.*?（红路 (\d+) 条＋绿路 (\d+) 条", _sp6o, SP6 + " 首行声明")]
_sp6pr = len(rows(_sp6o, r"^  [AG]\d+ "))
S["SP6_LOST"] = str(_sp6red + _sp6grn - _sp6pr)
S["SP6_PR"] = str(_sp6pr)
S["SP6_DECL"] = "%d＋%d" % (_sp6red, _sp6grn)
assert _sp6red == int(C["AFTC_DEC"]), "现场 %s 声明的红路 %d ≠ 终稿 %s 条 ⇒ 崩掉那批不是这一轮的" % (
    SP6, _sp6red, C["AFTC_DEC"])
_sp7o, _sp7e = specimen("SP7", SP7, why_pat=r"写盘面报出")
_w0, _w1 = [int(x) for x in one(r"^AssertionError: 写盘面报出 (\d+) 字节 ≠ 盘上现有 (\d+) 字节", _sp7o, SP7 + " 那句抛出")]
assert _w0 == len(raw_log) and _w1 == len(raw_log) + 2, (
    "探针 %s 的两枚字节数（%d／%d）与盘上现读 %d 不成「原值／+2」" % (SP7, _w0, _w1, len(raw_log)))
S["SP7_OFF"] = one(r"^BODY_OFFSET（从盘上现读）＝ (\d+)", _sp7o, SP7 + " 探针里的偏移")
S["SP7_CMP"] = one(r"^cmp -i 三联（本件自己起进程跑的，不是抄生成器的）＝ (\d+ / \d+ / \d+)", _sp7o,
                   SP7 + " 探针里的三联")
S["SP7_FLD"] = " / ".join(one(r"^顶格字段行数＝ (\d+) / (\d+)", _sp7o, SP7 + " 探针里的字段计数"))
S["SP7_RC"] = one(r"^子进程 rc＝ (\d+)$", _sp7o, SP7 + " 的子进程 rc")
S["SP7_CLEAN"] = one(r"^探针目录已删＝ (True|False)$", _sp7o, SP7 + " 的清理行")
assert int(S["SP7_OFF"]) == OFF + 2 and S["SP7_CMP"] == CMP3 and S["SP7_FLD"] == "%d / %d" % (FLDN, FLDN), (
    "探针 %s 的形状变了（偏移 %s／三联 %s／字段 %s）⇒ 那句「自洽门全绿、只有跨闭合抓得住」要重写" % (
        SP7, S["SP7_OFF"], S["SP7_CMP"], S["SP7_FLD"]))
assert S["SP7_CLEAN"] == "True", "探针目录没自证已删 ⇒ 该件的临时面可能还在档案目录里"

# ============================ 读数 6：GPU 红线（切自写手：坏尺子＋argv[0] 替代断言） ============================
_nv = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                     capture_output=True, text=True)
NVRC = str(_nv.returncode)
NVMSG = (_nv.stderr or _nv.stdout).splitlines()[0].strip() if (_nv.stderr or _nv.stdout).strip() else "（无输出）"
NVKERN = open("/sys/module/nvidia/version").read().strip()
_mlib = re.search(r"NVML library version: ([\d.]+)", _nv.stderr + _nv.stdout)
NVLIB = _mlib.group(1) if _mlib else "取不到"


def argv0(pid):
    try:
        parts = open("/proc/%s/cmdline" % pid, "rb").read().split(b"\0")
    except OSError:
        return None
    return parts[0].decode("utf-8", "replace") if parts and parts[0] else None


def match_argv0(blob):
    return blob.split(b"\0")[0].decode("utf-8", "replace") == INTERP


PIDS = [d for d in os.listdir("/proc") if d.isdigit() and d not in (str(os.getpid()), str(os.getppid()))]
MINE_NOW = [d for d in PIDS if argv0(d) == INTERP]
assert argv0(str(os.getpid())) == INTERP, "本件自己的 argv[0] 不是 %s ⇒ 下面那个 0 数错了对象" % INTERP
assert match_argv0((INTERP + "\0x.py").encode()) is True, "匹配式的正对照自己失效 ⇒ 那个 0 是空转"
assert match_argv0(b"/bin/bash -c grep " + INTERP.encode() + b"\0") is False, "匹配式把子串也认成命中 ⇒ 会误指自己"
assert len(MINE_NOW) == 0, "机上还留着钉住的那个解释器在跑 %s ⇒ 归还口径破了" % MINE_NOW

# ============================ 节号与落地段：从档案现取 ============================
devtxt = open(DEVLOG, encoding="utf-8").read()
_secs = list(re.finditer(r"^### 26\.(\d+)[^\n]*$", devtxt, re.M))
assert _secs, "开发日志里取不到 `### 26.NN` ⇒ 节号是凭记忆的"
SECREF = _secs[-1].group(1)
assert "分片 %s" % SLICE in _secs[-1].group(0), "最后一节标题不含「分片 %s」：%s" % (SLICE, _secs[-1].group(0))
_seg = devtxt[_secs[-1].end():]
_nx = _seg.find("\n### ")
SEC_BODY = _seg[:_nx] if _nx != -1 else _seg
BULLETS = top_bullets(SEC_BODY)
assert len(BULLETS) == 8, "§26.%s 顶级子条＝%d 条（须 7 条登记＋1 条落地＝8）" % (SECREF, len(BULLETS))
assert len({b[0] for b in BULLETS}) == len(BULLETS), "§26.%s 顶级子条标签有重名 ⇒ 按标签切段会串行" % SECREF
_ptxt = open(PLAN, encoding="utf-8").read()
_pnums = [int(x) for x in re.findall(r"^### §W 补记 (\d+)", _ptxt, re.M)]
assert _pnums, "计划文档里取不到 `### §W 补记 N` 标题 ⇒ 补记序号是凭记忆的"
PNUM = str(max(_pnums))

# ============================ 五张档案面与声称清单 ============================
TEXTS = dict(zip(FACE_NAMES, [open(p, encoding="utf-8").read() for p in FACE_PATHS]))
assert len(TEXTS) == 5, "档案面不到 5 张"
CLAIMS = [
    ("HEAD 全哈希与父", "HEAD＝`%s`（父 `%s`，短哈希 `%s`／`%s`" % (HEAD, PAR40, HEAD7, PAR7)),
    ("提交时刻与取法", "提交时刻 `%s` 由 `git log -1 --format=%%cd --date=iso` 现取" % CDATE),
    ("推送区间", "推送区间 `%s..%s`" % (PAR7, HEAD7)),
    ("入库面 A／M／合计", "入库面＝新增 (A) **%d**＋修改 (M) **%d**＝**%d** 件" % (NAC, NMC, NTOTC)),
    ("shortstat 原文", "原文＝`%s`（**没有删除段**）" % SS_RAW),
    ("numstat 现算", "逐行现算＝**%d**、删除＝**%d**" % (NS_INS, NS_DEL)),
    ("生产树改动", "改动现数 **%d** 件" % NCODE),
    ("truncated", "`truncated=%s`" % TRUNC),
    ("blob＋tree 条目", "tree 条目 **%d**＝blob **%d**＋tree **%d**" % (TB, BB, BT)),
    ("本地／父提交 blob", "本地 blob＝%d、父提交 blob＝%d" % (LB, PB)),
    ("双向差集", "＝`%s`" % DIFF3),
    ("全库 SHA 普查", "`%s`" % SHA4),
    ("本片单列复核", "`%s`" % PER4),
    ("闭合式", "`%s`" % CLOS),
    ("正对照 A", "`%s`" % CA),
    ("正对照 B", "`%s`" % CB),
    ("正对照 C", "`%s`" % CC),
    ("正对照 D", "`%s`" % CD),
    ("落地证明判读行", "判读行 `%s`" % V),
    ("闭合式 git 复对", "%d − (%d ＋ %d) ＝ **%d**（须 0）" % (BB, PB, NAC, CLOSEX)),
    ("远端与本地 blob 相等", "（%d＝%d）" % (BB, LB)),
    ("SRC 指纹前 12", "那枚前 12 位 `%s`" % FINGER_REC[:12]),
    ("回执三段字节", "档案名＝`%s` **%d** B／**%d** 行＝表头 **%d**＋正文 **%d**（表头里非空行 **%d** 行，"
     "与顶格字段 **%d** 行不是同一个量）" % (RLOG, len(raw_log), RLINES, OFF, BODYB, HDNL, FLDN)),
    ("BODY_OFFSET 定点", "`BODY_OFFSET=%d` 由写入件重解一遍并核对到正文区与原始件**逐字节相同**" % OFF),
    ("只读仪器字节", "（stdout＝`%s` **%d** B）" % (AFT_OUT, AFTB)),
    ("cmp 三联", "自己起进程跑 `cmp -i %d:0`，三联＝`%s`" % (OFF, CMP3)),
    ("顶格字段与声明互证", "顶格字段 **%d/%d** 与生成器 `FIELDS` 声明互证（ast 现取，不执行生成器）" % (FLDN, FLDN)),
    ("三方字节同值", "报出的字节数、盘上现读、只读仪器三方同值＝%d B）" % len(raw_log)),
    ("输入面张数", "输入面 md5 跑前后逐一相同＝True（%d 张面）" % AFT_NFACES),
    ("只读仪器判读行", "判读行 `%s`" % HRAV),
    ("复现腿三方", "**本件又自己起了两趟只读仪器复跑，与归档 stdout 三方逐字节相同＝True**"),
    ("提交说明面字节与条数", "%s 的 stdout＝%s **%s** B（md5 前 12 `%s`／stderr %s B）｜CASES 声明 **%s** 条"
     "（全部红路，其中克隆别的仪器 %s 条）｜保持绿 **%s** 条｜超时 **%s** 条｜本面自查 **%s** 条｜"
     "造不出红路 **%s** 条 ⇒「%s」" % (C["MSG_SCRIPT"], C["MSG_OUT"], C["MSG_OUTB"], C["MSG_OUTM"],
                                      C["MSG_ERRB"], C["MSG_DEC"], C["MSG_CLONE"], C["MSG_GRN"],
                                      C["MSG_TO"], C["MSG_SELF"], C["MSG_UNC"], C["MSG_VERDICT"])),
    ("表头面字节与条数", "%s 的 stdout＝%s **%s** B（md5 前 12 `%s`／stderr %s B）｜红路声明 **%s** 条｜"
     "保持绿 **%s** 条｜基线 **%s** 条（未扰动副本，在配额闸门之前跑，不占本批）｜基线外整批 **%s** 条｜"
     "环境故障 **%s** 条｜本面自查 **%s** 条｜造不出红路 **%s** 条 ⇒「%s」" % (
         C["HDR_SCRIPT"], C["HDR_OUT"], C["HDR_OUTB"], C["HDR_OUTM"], C["HDR_ERRB"], C["HDR_DEC"],
         C["HDR_GRN"], C["HDR_BASE"], C["HDR_BATCH"], C["HDR_TO"], C["HDR_SELF"], C["HDR_UNC"],
         C["HDR_VERDICT"])),
    ("表头面配额墙钟", "跨配额窗口等待 **%s** 次、累计 **%s** 秒" % (C["HDR_WAITN"], C["HDR_WAITS"])),
    ("只读仪器面字节与条数", "%s 的 stdout＝%s **%s** B（md5 前 12 `%s`／stderr %s B）｜红路声明 **%s** 条｜"
     "保持绿 **%s** 条（含裸跑基线 1 条）｜实跑行数 **%s** 条｜超时 **%s** 条｜装配失败 **%s** 条｜"
     "本面自查 **%s** 条｜造不出红路 **%s** 条 ⇒「%s」" % (
         C["AFTC_SCRIPT"], C["AFTC_OUT"], C["AFTC_OUTB"], C["AFTC_OUTM"], C["AFTC_ERRB"], C["AFTC_DEC"],
         C["AFTC_GRN"], C["AFTC_ROWS"], C["AFTC_TO"], C["AFTC_ASM"], C["AFTC_SELF"], C["AFTC_UNC"],
         C["AFTC_VERDICT"])),
    ("ast 表长闭合", "与该面归档 stdout 自己声明的条数逐面相等＝%s；只读仪器面另合一条 "
     "`实跑行数＝红路＋绿路＋基线`＝%s" % (ASTCLOSE, C["AFTC_ROWS"])),
    ("三面同形制按行现数", "三面归档 stdout 里「保持绿」「本轮造不出红路的闸门」「CONTROL_VERDICT」各按面"
     "恰好 1 **行**（＝%s" % "/".join(str(x) for x in PER_LINE),),
    ("现场⑴字节与抛出", "`%s` stdout **%s** B／stderr **%s** B＝「%s」" % (SP1, S["SP1_B"], S["SP1_E"], S["SP1_WHY"])),
    ("现场⑵字节与未命中", "`%s` stdout **%s** B／stderr **%s** B，表里「× 」行 **%s** 条、整面判读「%s」" % (
        SP2, S["SP2_B"], S["SP2_E"], S["SP2_MISS"], S["SP2_VERDICT"])),
    ("现场⑶字节与打印行数", "`%s` stdout **%s** B／stderr **%s** B（崩之前只打印出 %s 行）＝「%s」" % (
        SP3, S["SP3_B"], S["SP3_E"], S["SP3_ROWS"], S["SP3_WHY"])),
    ("现场⑷未命中与两条抛出", "`%s` stdout **%s** B／stderr **%s** B，「× 」行 **%s** 条、判读「%s」，"
     "两条实得抛出行＝%s" % (SP4, S["SP4_B"], S["SP4_E"], S["SP4_MISS"], S["SP4_VERDICT"], S["SP4_RAISES"])),
    ("现场⑸探针与生成器行号", "`%s` stdout **%s** B／stderr **%s** B＝「%s」，它逐字符等于生成器 `am_s%d_hdr.py` "
     "第 **%s** 行那条 `assert REPRO` 的消息" % (SP5, S["SP5_B"], S["SP5_E"], S["SP5_WHY"], SLICE, S["SP5_LINE"])),
    ("现场⑹作废条数", "`%s` stdout **%s** B／stderr **%s** B＝「%s」，崩在绿路第一条之后（首行声明 %s 条、"
     "实际打印 %s 行、作废 **%s** 条）" % (SP6, S["SP6_B"], S["SP6_E"], S["SP6_WHY"], S["SP6_DECL"],
                                        S["SP6_PR"], S["SP6_LOST"])),
    ("第三次真跑全命中", "第三次真跑红路 %s 条全命中、环境／超时 0 条" % C["HDR_DEC"]),
    ("LESSON 那句抛出", "两条都红，且红在 `%s`" % S["SP7_WHY"]),
    ("探针⑺字节与 rc", "探针 `%s`（stdout **%s** B／stderr **%s** B，子进程 rc＝%s）留下全过程" % (
        SP7, S["SP7_B"], S["SP7_E"], S["SP7_RC"])),
    ("探针⑺重算后的三件套", "偏移由不动点重算成 %s（＝表头原值 %d＋2），算术闭合、`cmp` 三联（%s）、顶格字段（%s）"
     "与残留检查**全绿**" % (S["SP7_OFF"], OFF, S["SP7_CMP"], S["SP7_FLD"])),
    ("探针⑺目录自证已删", "的探针目录自证已删＝%s" % S["SP7_CLEAN"]),
    ("坏尺子的四条读数", "NVML 那把尺子本轮复测仍起不来（rc＝%s，首行「%s」；内核模块 %s／用户态库 %s）" % (
        NVRC, NVMSG, NVKERN, NVLIB)),
    ("替代断言现数的 0", "现数我的解释器进程 **%d** 个" % len(MINE_NOW)),
]
FACE_ONLY = [
    ("总览：三面控件面摘要句", "**三面控件面**：提交说明面 **%s** 条扰动／造不出红路 **%s** 条；表头面红路 "
     "**%s** 条／保持绿 **%s** 条／环境故障 **%s** 条／造不出红路 **%s** 条（跨配额窗口等待 %s 次、累计 %s 秒）" % (
         C["MSG_DEC"], C["MSG_UNC"], C["HDR_DEC"], C["HDR_GRN"], C["HDR_TO"], C["HDR_UNC"],
         C["HDR_WAITN"], C["HDR_WAITS"])),
    ("总览：新学标题句", "- **一条口径级新学＝%s**：在跨闭合的校验器" % LESSON_NEW),
    ("计划：补记标题的落地腿定性句", "＝分片 %d 的**落地腿**：仓库推进到 `%s`／远端 blob **%d**" % (SLICE, HEAD7, BB)),
    ("计划：只读仪器判读行句", "只读仪器判读行 `%s`" % HRAV),
    ("纪实：替上一片入库那句", "落地腿＝替分片 %s 入库整条链（A **%d**／M **%d**＝**%d** 件）＋一条口径级新学＝%s" % (
        PREVSLICE, NAC, NMC, NTOTC, LESSON_NEW)),
    ("纪实：坏尺子复测句", "NVML 复测仍 rc＝%s（首行「%s」；内核 %s／用户态 %s）" % (NVRC, NVMSG, NVKERN, NVLIB)),
    ("索引：shortstat 内联句", "入库 A **%d**／M **%d**＝**%d** 件，shortstat「%s」＝零删除）" % (
        NAC, NMC, NTOTC, SS_RAW)),
    ("索引：新学锚句", "新学＝%s（跨闭合校验器面前良性扰动必须等长" % LESSON_NEW),
]
CLAIMS += FACE_ONLY


def detector(claims, texts):
    return [(lab, lit) for lab, lit in claims if not any(lit in t for t in texts.values())]


MISSING = detector(CLAIMS, TEXTS)
assert not MISSING, "有 %d 条现取读数在档案里搜不到原样：%s" % (
    len(MISSING), "、".join("%s=%r" % (l, s[:44]) for l, s in MISSING))
_carriers = dict((lab, [n for n in FACE_NAMES if lit in TEXTS[n]]) for lab, lit in CLAIMS)
assert all(_carriers.values()), "有声称在五面里都搜不到"
COVERAGE = [(_n, len([lab for lab, c in _carriers.items() if _n in c]),
             len([lab for lab, c in _carriers.items() if c == [_n]])) for _n in FACE_NAMES]
assert len([lab for lab, c in _carriers.items() if c]) == len(CLAIMS), "有声称一条都没搜到"
UNIQ_TOTAL = sum(u for _, _, u in COVERAGE)
assert 1 <= UNIQ_TOTAL <= len(CLAIMS), "独有数之和 %d 不在 1..%d" % (UNIQ_TOTAL, len(CLAIMS))
for _lab, _lit in CLAIMS:
    assert 1 <= len(_carriers[_lab]) <= 5, "%s 命中 %d 面（应当 1..5）" % (_lab, len(_carriers[_lab]))

# ============================ 正对照：五内存扰动＋唯一性诱饵（全部不写盘） ============================
CTRL = []
_c1 = detector([(l, s + "◇") for l, s in CLAIMS], TEXTS)
assert len(_c1) == len(CLAIMS), "读数扰动后仍有 %d 条没判缺 ⇒ 回搜是空转" % (len(CLAIMS) - len(_c1))
CTRL.append("读数扰动（每条尾巴加一个档案里不可能出现的记号）⇒ 判缺 %d/%d" % (len(_c1), len(CLAIMS)))
_alt = HEAD[:-1] + ("0" if HEAD[-1] != "0" else "1")
_c2 = detector([("HEAD 全哈希与父", "HEAD＝`%s`（父 `%s`，短哈希 `%s`／`%s`" % (_alt, PAR40, HEAD7, PAR7))], TEXTS)
assert len(_c2) == 1 and _c2[0][0] == "HEAD 全哈希与父", "一位哈希扰动没让那一条红：%s" % _c2
CTRL.append("一位哈希扰动（HEAD 末位换值）⇒ 判缺 1/1")
_uniq_faces = [_n for _n, _c, _u in COVERAGE if _u > 0]
assert _uniq_faces, "五面全无独有声称 ⇒ 逐面抽空白这根针没有靶"
for _n in _uniq_faces:
    _c3 = detector(CLAIMS, dict(TEXTS, **{_n: ""}))
    _u = len([lab for lab, c in _carriers.items() if c == [_n]])
    assert len(_c3) == _u, "%s 抽成空文本判缺 %d 条，而它独有 %d 条 ⇒ 两者必须相等" % (_n, len(_c3), _u)
CTRL.append("逐面抽空白（只对独有数>0 的面要求红，实测 %d 面）⇒ 各面缺失＝各自独有数" % len(_uniq_faces))
_c4 = detector(CLAIMS, dict((n, "") for n in FACE_NAMES))
assert len(_c4) == len(CLAIMS), "五面全空却只判缺 %d 条 ⇒ detector 不依赖档案文本" % len(_c4)
CTRL.append("全集扰动（五面全换成空文本）⇒ 判缺 %d/%d" % (len(_c4), len(CLAIMS)))
_alt_m = 9 if NMC != 9 else 8
_c5 = detector([("入库面 A／M／合计", "入库面＝新增 (A) **%d**＋修改 (M) **%d**＝**%d** 件" % (NAC, _alt_m, NTOTC))],
               TEXTS)
assert len(_c5) == 1, "把 M 换成 %d（现算 %d）却没判缺 ⇒ 那三位数之间没有互相校验" % (_alt_m, NMC)
CTRL.append("单值扰动（把 M 换成与现算不同的一个数）⇒ 判缺 1/1")
_bump = int(C["AFTC_DEC"]) + 1
_c6 = detector([("只读仪器面字节与条数", "%s 的 stdout＝%s **%s** B（md5 前 12 `%s`／stderr %s B）｜红路声明 **%d** 条｜"
                 "保持绿 **%s** 条（含裸跑基线 1 条）｜实跑行数 **%s** 条｜超时 **%s** 条｜装配失败 **%s** 条｜"
                 "本面自查 **%s** 条｜造不出红路 **%s** 条 ⇒「%s」" % (
                     C["AFTC_SCRIPT"], C["AFTC_OUT"], C["AFTC_OUTB"], C["AFTC_OUTM"], C["AFTC_ERRB"], _bump,
                     C["AFTC_GRN"], C["AFTC_ROWS"], C["AFTC_TO"], C["AFTC_ASM"], C["AFTC_SELF"],
                     C["AFTC_UNC"], C["AFTC_VERDICT"]))], TEXTS)
assert len(_c6) == 1, "把只读仪器面的红路声明 +1 却没判缺 ⇒ 那一长句是整段回搜、没逐格校验" % ()
CTRL.append("逐格扰动（只读仪器面红路声明 +1，其余十七格原样）⇒ 判缺 1/1")

# ---------------- 下一片生成器要用的六条节内唯一性（按顶级子条，与下游同问法） ----------------
NEEDLES = (("引用本片回执名", RLOG, 1),
           ("引用上一片回执名", "am_s7_verify%d.log" % (SLICE - 1), 1),
           ("带引号的钉住口径登记", r"当前「[^」]+」", 1),
           ("承接锚句：自审的方向性", CARRIED[0], 1),
           ("承接锚句：普查仪器没有第二出处", CARRIED[1], 1),
           ("本轮新学锚句", LESSON_NEW, 1))
BAIT = {"引用本片回执名": "\n- **诱饵子条**：见 %s。\n" % RLOG,
        "引用上一片回执名": "\n- **诱饵子条**：见 am_s7_verify%d.log。\n" % (SLICE - 1),
        "带引号的钉住口径登记": "\n- **诱饵**：当前「诱饵口径」\n",
        "承接锚句：自审的方向性": "\n- **诱饵**：%s\n" % CARRIED[0],
        "承接锚句：普查仪器没有第二出处": "\n- **诱饵**：%s\n" % CARRIED[1],
        "本轮新学锚句": "\n- **诱饵**：%s\n" % LESSON_NEW}
for what, pat, want in NEEDLES:
    rx = re.escape(pat) if not pat.startswith("当前「") else pat
    n = len([1 for _l, b in BULLETS if re.search(rx, b, re.M)])
    assert n == want, "%s 在磁盘上的 §26.%s 里命中 %d 条顶级子条（须恰好 %d）⇒ 分片 %s 的生成器会拒绝归档" % (
        what, SECREF, n, want, NEXTSLICE)
    n2 = len([1 for _l, b in top_bullets(SEC_BODY + BAIT[what]) if re.search(rx, b, re.M)])
    assert n2 == want + 1, "%s 的诱饵没造出 %d 处（现数 %d）⇒ 这条唯一性计数是恒真式" % (what, want + 1, n2)
CTRL.append("节内六条唯一性各自的诱饵（内存里再补一条顶级子条⇒各 +1）⇒ 同一判据当场可红")
_fs = r"f-string 把 `\{32\}` 当成\*\*替换字段"
assert len(re.findall(_fs, devtxt)) == 1, "f-string 事故句在整份日志里不是恰好 1 处 ⇒ 下一片生成器本来就会拒绝"
CTRL.append("f-string 事故句全日志恰好 1 处（下一片生成器的硬前提）")

# ---------------- 面内自洽：易失读数不现取，只核它自己那行的加法 ----------------
# 形制（从写手 DEV_T 的「待入库」子条）：未跟踪 **a**＋已暂存新增 **b**＋修改 **c**＝**d** 处，
# 分组＝甲 **g1**／…（和＝g1＋…＝h）。**分组和闭合的是「未跟踪＋已暂存新增」那一项**（角色分组只数这两类），
# 与写手那句 `len(assigned) == NNEW + NU` 同源；修改面不进分组。
_pat = (r"现算未跟踪 \*\*(\d+)\*\*＋已暂存新增 \*\*(\d+)\*\*＋修改 \*\*(\d+)\*\*＝\*\*(\d+)\*\* 处，"
        r"分组＝(.*?)（和＝([0-9＋＝]+)）")
_h = re.findall(_pat, SEC_BODY)
assert len(_h) == 1, "§26.%s 里「待入库」那句形制命中 %d 处（须恰好 1）⇒ 下一片的现取会取错行" % (SECREF, len(_h))
_g1, _g2, _g3, _g4, _grp, _sum = _h[0]
assert int(_g1) + int(_g2) + int(_g3) == int(_g4), "待入库那句自己的三项加法不闭合：%s＋%s＋%s≠%s" % (
    _g1, _g2, _g3, _g4)
_bnums = [int(x) for x in re.findall(r"\*\*(\d+)\*\*", _grp)]
_snums = [int(x) for x in re.findall(r"\d+", _sum)]
assert len(_bnums) >= 8, "分组逐项只有 %d 组（本轮登记 8 组）⇒ 那段被截了" % len(_bnums)
assert len(_snums) == len(_bnums) + 1 and sum(_snums[:-1]) == _snums[-1] == int(_g1) + int(_g2), (
    "分组和 %s 的尾数应＝未跟踪＋新增（%s＋%s），且前 %d 项之和要等于它" % (_sum, _g1, _g2, len(_bnums)))
assert _bnums == _snums[:-1], "分组逐项 %s 与和式逐项 %s 不同 ⇒ 两处有一处是抄的" % (_bnums, _snums[:-1])
CTRL.append("待入库那句的**面内自洽**（三类和＝合计；各组和＝未跟踪＋新增，逐组双向对）⇒ 通过")
_bait3 = str(int(_g2) + 100)
assert int(_g1) + int(_bait3) + int(_g3) != int(_g4), "把新增改成 %s 之后加法仍然闭合 ⇒ 那条自洽抓不到" % _bait3
CTRL.append("合计诱饵（把「已暂存新增」改成 %s，与现数不成加法）⇒ 同一判据当场判缺" % _bait3)

# ---------------- 落笔那句预测的兑现：此刻的待入库现数必须大于它 ----------------
st = [l for l in sh("git", "-c", "core.quotePath=false", "status", "--porcelain", "-uall").splitlines() if l.strip()]
assert not [l for l in st if l[3:].endswith("/")], "清单里有折叠的目录行 ⇒ `-uall` 没生效"
stat = {}
for l in st:
    stat[l[3:].strip()] = l[:2].strip()
PEND_NOW = len(stat)
assert PEND_NOW > int(_g4), (
    "此刻待入库 %d 处 ≤ 落笔那句预告的 %d 处 ⇒ 「最终入库数大于这里的」那句成了假话" % (PEND_NOW, int(_g4)))
CTRL.append("落笔那句「最终入库数大于这里的 %s」的兑现核对（此刻待入库现数须严格大于它）⇒ 通过" % _g4)
assert not (PEND_NOW > int(_g4) + 100), "把预告值抬到 +100 之后同一条判据仍然绿 ⇒ 这个大小比较没有分辨力"
for p in FACE_PATHS:
    rel = os.path.relpath(p, REPO)
    assert stat.get(rel) == "M", "%s 现在不是 (M)：%r ⇒ 落地写盘没留在工作树里" % (rel, stat.get(rel))
for p in (SELF_CLAIM,
          "docs/evidence/2026-10-13/_raw_s%d_archive_check.out" % SLICE,
          "docs/evidence/2026-10-13/_raw_s%d_archive_check.err" % SLICE):
    assert stat.get(p) == "??", "%s 现在不是未跟踪：%r ⇒ 本件要在入库前跑" % (p, stat.get(p))
assert int(one(r"§26\.%s 里现取\*\*唯一\*\*一条引用 `%s` 的顶级子条——本节由落地段恰好提供 (\d+) 条" % (
    SECREF, RLOG), _ptxt, "计划文档里那句关于本片节面的「恰好提供 N 条」")) == 1, (
    "计划文档里那句「落地段恰好提供 1 条」现数不是 1 ⇒ 下一片的 `bullet_for` 会拒绝")

# ---------------- 最后一批纯盘形制断言（volatile：只断言，不入 stdout） ----------------
for _n, _p in zip(FACE_NAMES, FACE_PATHS):
    assert open(_p, "rb").read()[-1:] == b"\n", "%s 末字节不是换行 ⇒ 下一次追加会粘连" % _n
    assert os.path.getsize(_p) > 0, "%s 是空文件" % _n
assert open(os.path.join(D, "_raw_s%d_landing_precheck.out" % SLICE), encoding="utf-8").read().endswith(
    "LANDING_FILLER_VERDICT = READINGS DERIVED, STRUCTURE GATES PASS\n"), "写手预检面的末行不是那句判读"
assert one(r"^(WRITER_VERDICT = .*)$",
           open(os.path.join(D, "_raw_s%d_landing_write.out" % SLICE), encoding="utf-8").read(),
           "写腿判读行") == "WRITER_VERDICT = FIVE LANDING FACES LANDED, UNIQUENESS RECOUNTED ON DISK", \
    "写腿判读行形制变了"

print("归档自查（分片 %d 落地腿）＝把五张档案面里每条落地声称由命令现取再回搜原样" % SLICE)
print("  声称条数＝%d｜缺失＝0/%d｜每条命中面数 1..5 全部通过" % (len(CLAIMS), len(CLAIMS)))
for _c in CTRL:
    print("  正对照：%s" % _c)
for _n, _c, _u in COVERAGE:
    print("  承载：%s＝%d 条（其中独有 %d 条）" % (_n, _c, _u))
print("  §26.%s 顶级子条＝%d 条；本片回执名／上一片回执名／钉住口径／两枚承接锚句／本片 LESSON 各恰好 1 条且各配诱饵" % (
    SECREF, len(BULLETS)))
print("  回执正文区与 `%s` 逐字节相等（`cmp -i %d:0` 三联 %s 由归档件自跑、本件另起两趟只读仪器复现）；"
      "BODY_OFFSET 定点复解＝%d；SRC 指纹与此刻盘上现算同值" % (RAWM, OFF, CMP3, OFF))
print("  尺子 `am_s7_verify%d.py` 全树 %d 份逐字节相同；三面 ast 表长闭合＝%s" % (SLICE, NCOPIES, ASTCLOSE))
print("  volatile 读数（待入库件数、PID 分母、落笔 date）按口径只断言、不入 stdout ⇒ 本件可复跑逐字节相同")
print("  闭合式复核：远端 %d −（父 %d ＋ 新增 %d）＝%d；入库清单 %d 件与 numstat %d 行对齐、删除 %d" % (
    BB, PB, NAC, CLOSEX, NTOTC, len(NSR), NS_DEL))
print("ARCHIVE_VERDICT = ALL CLAIMS LIVE AND PRESENT, ALL CONTROLS CAN GO RED")
