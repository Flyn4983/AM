#!/usr/bin/env python3
"""归档腿自查（分片 29）：本轮**五张档案面**里每一条落地声称，必须由命令现取一次再回搜原样。

范围照实说清：本件普查的是"落地读数"这张清单（39 条：HEAD／父提交／入库面 A-M-合计／shortstat／
blob＋tree 条目／闭合式／三组带分母的 0／四个正对照／判读行／回执三段字节／cmp 三联／顶格字段与
`FIELDS` 互证／尺子 md5 与两份数／SRC 指纹／两面控件面的字节-md5-条数-UNCOVERED／预检与写盘两件原始件／
卡死现场件与时长与起跑差与第一次尝试那对 0 字节／守卫两个常量与子运行次数／守卫正对照源码与原始件与复跑），
**不是**对归档散文做全量出处普查——那是 `am_s29_hdr.py` 对表头面做的事，本件管散文面。

取值命令与**字面串格式**逐字节切自本轮填值器 /tmp/s29_head.py（22–45／71–180／209–255 三段）⇒
"档案里写的"与"自查重新现取的"用的是同一把尺子，改了格式两边一起红，不会出现一个改了一个没改。

三条口径：① volatile 读数（未入库件数、卡上 compute 进程数、落笔 `date`）**只断言、不入 stdout**，
否则"复跑逐字节相同"不可能成立；② 每个 0 配一条**同一次运行里能红**的正对照（四条内存扰动＋
一条回执名唯一性诱饵，全部不写盘）；③ 上一片已入库那张回执的生成器要用的三条唯一性锚句
（上一片回执名／钉住口径「当前「」」／LESSON 锚句）在追加后仍各自恰好 1 条，本件自己那条落地子条
一个字都不复述——复述即让那把已入库的尺子再也复跑不出来。

运行＝`CUDA_VISIBLE_DEVICES= /tmp/amvenv/bin/python docs/evidence/2026-10-10/am_s29_archive_check.py`
（不 import jax：只跑 git／md5sum／cmp／nvidia-smi／读文件。）
"""
import glob
import hashlib
import os
import re
import subprocess

REPO = "/home/shy/桌面/2026年开发-显式冲击有限元-物质点法-sph-fvm+am/AM-Qoder"
D = os.path.join(REPO, "docs/evidence/2026-10-10")
LOG = os.path.join(D, "am_s7_verify29.log")
RAWO = os.path.join(D, "_raw_verify29.out")
RAWE = os.path.join(D, "_raw_verify29.err")
HDR = os.path.join(D, "am_s29_hdr.py")
PRE = os.path.join(D, "_raw_s29_hdr_precheck.out")
WRI = os.path.join(D, "_raw_s29_hdr_write.out")
CTLO = os.path.join(D, "_raw_s29_hdr_controls.out")
MSGO = os.path.join(D, "_raw_s29_msg_controls.out")
DEVLOG = os.path.join(REPO, "docs/开发日志.md")
SLICE, PREV = "29", "28"
LABELS = (r"生成时刻|CMD|ENV|脚本|TREE|SRC指纹|GPU|RUN|判读|四个正对照|分母口径|正文区|复核|"
          r"同片对照|用途|教训")


def sh(*a):
    return subprocess.run(a, cwd=REPO, capture_output=True, text=True, check=True).stdout


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()



HEAD = sh("git", "rev-parse", "HEAD").strip()
PARENT = sh("git", "rev-parse", "HEAD^").strip()
ROWS = [r.split("\t") for r in sh("git", "-c", "core.quotePath=false", "show", "--name-status",
                                  "--format=", "HEAD").splitlines() if r.strip()]
NA = sum(1 for r in ROWS if r[0] == "A")
NM = sum(1 for r in ROWS if r[0] == "M")
NTOT = len(ROWS)
assert NA + NM == NTOT, "入库清单里有 A／M 之外的状态字母"
SS = sh("git", "diff-tree", "--no-commit-id", "--shortstat", "-r", "HEAD").strip()
body = open(RAWO, encoding="utf-8").read()
assert "LANDING_VERDICT = ALL CHECKS PASS" in body, "回执正文不是全过"
assert os.path.getsize(RAWE) == 0, "回执 stderr 非 0 字节"
logtxt = open(LOG, encoding="utf-8").read()
OFF = re.search(r"^BODY_OFFSET=(\d+)$", logtxt, re.M).group(1)
RAWSZ, LOGSZ = str(os.path.getsize(RAWO)), str(os.path.getsize(LOG))
LINES = str(len(logtxt.splitlines()))
assert int(OFF) + int(RAWSZ) == int(LOGSZ), "表头＋正文 ≠ 回执"
PREB, PREMD5 = str(os.path.getsize(PRE)), md5(PRE)[:12]
WRB, WRMD5 = str(os.path.getsize(WRI)), md5(WRI)[:12]
NFIELD = len([l for l in logtxt[:int(OFF)].splitlines() if re.match(r"^(%s) *:" % LABELS, l)])
NFIELDS = len(re.findall(r'"([^"]+)"', re.search(r"^FIELDS = \[(.*?)\]", open(HDR, encoding="utf-8").read(),
                                                 re.M | re.S).group(1)))
CMP = [subprocess.run(["cmp", "-i", i, LOG, RAWO], cwd=REPO, capture_output=True).returncode
       for i in ("%s:0" % OFF, "%d:0" % (int(OFF) + 1), "%s:1" % OFF)]
assert CMP == [0, 1, 1], "cmp 三联不齐：%s" % CMP
ruler = md5(os.path.join(D, "am_s7_verify%s.py" % SLICE))
NCOPIES = len([p for p in glob.glob(os.path.join(REPO, "docs/evidence/*/am_s7_verify*.py"))
               if md5(p) == ruler])
NDIR = len([p for p in glob.glob(os.path.join(D, "am_s7_verify*.py")) if md5(p) == ruler])
FINGER = re.search(r"^SRC指纹 *: *([0-9a-f]{32})", logtxt, re.M).group(1)
now_finger = subprocess.run(["bash", "-c", "find src tests -name '*.py' -not -path '*__pycache__*' "
                             "-print0 | sort -z | xargs -0 md5sum | md5sum | cut -d' ' -f1"],
                            cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
assert FINGER == now_finger, "表头 SRC 指纹与此刻现算不同 ⇒ 生产树动过"
ctxt = open(CTLO, encoding="utf-8").read()
mtxt = open(MSGO, encoding="utf-8").read()
CTLB, CTLMD5 = str(os.path.getsize(CTLO)), md5(CTLO)[:12]
MSGB, MSGMD5 = str(os.path.getsize(MSGO)), md5(MSGO)[:12]
CTLN = int(re.search(r"扰动条数＝(\d+)", ctxt).group(1))
MSGN = int(re.search(r"扰动条数＝(\d+)", mtxt).group(1))
CTLCOV = int(re.search(r"本轮造不出红路的闸门＝(\d+) 条", ctxt).group(1))
MSGCOV = int(re.search(r"本轮造不出红路的闸门＝(\d+) 条", mtxt).group(1))
for nm, txt in (("表头面", ctxt), ("信息面", mtxt)):
    assert "CONTROL_VERDICT = ALL CONTROLS CAN GO RED" in txt, nm + "判读行不是全红"
    assert not [l for l in txt.splitlines() if l.startswith("× ")], nm + "有未命中，不能写成全红"
g = lambda pat: re.search(pat, body).group(1)
ENTRIES, NB, NTREE = re.search(r"tree 条目 (\d+) = blob (\d+) \+ tree (\d+)", body).groups()
NP, DL, DR = g(r"父提交 = (\d+)"), g(r"本地独有 (\d+)/"), g(r"远端独有 (\d+)/")
MS, NSH, SLM, NFL = g(r"blob SHA 不符 (\d+)/"), g(r"blob SHA 不符 \d+/(\d+)"), g(r"MATCH (\d+)/"), g(r"本片 (\d+) 文件")
CA = g(r"正对照 A（远端 \S+ 的 sha 首位改 f）⇒ 不符 (\d+)/")
CTLA_T = re.search(r"正对照 A（远端 (\S+) 的 sha 首位改 f）", body).group(1)
_CB = re.search(r"正对照 B（远端删一条路径）⇒ 本地独有 (\d+)/(\d+)，远端独有 (\d+)/(\d+)", body)
_CC = re.search(r"正对照 C（闭合式右边 \+1）⇒ (\d+) == (\d+) 判", body)
_CD = re.search(r"正对照 D（远端删本片一条路径）⇒ 本片缺失 (\d+)/(\d+)", body)
assert _CB and _CC and _CD and int(CA) == 1 and int(_CB.group(1)) == 1 and int(_CD.group(1)) == 1 \
    and int(_CC.group(2)) == int(NB) + 1, "正对照读数本身不对"
assert int(DL) == 0 and int(DR) == 0 and int(MS) == 0 and int(SLM) == int(NFL), "正文里的 0 现在不是 0 了"
assert int(NB) == int(NP) + NA and int(ENTRIES) == int(NB) + int(NTREE), "闭合式／条目式不闭合"
st = sh("git", "-c", "core.quotePath=false", "status", "--porcelain").splitlines()
UNT, MOD = len([l for l in st if l.startswith("??")]), len([l for l in st if l[:2] not in ("??", "")])
nv = [l.strip() for l in subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                                        capture_output=True, text=True).stdout.splitlines() if l.strip()]
nv_mine = [p for p in nv if "/tmp/amvenv" in open("/proc/%s/cmdline" % p, "rb").read().decode("utf-8", "replace")]
assert nv_mine == [], "我的解释器在卡上留有进程 %s" % nv_mine
DIRTY_S = len([l for l in sh("git", "-c", "core.quotePath=false", "status", "--porcelain",
                             "--", "src", "tests").splitlines() if l.strip()])
assert DIRTY_S == 0, "生产树 src+tests 有 %d 处改动" % DIRTY_S
TIME = sh("date", "+%F %T %z").strip()

# ---------------- 本轮新学：控件面卡死 2h08m 的现场、守卫与守卫的正对照 ----------------
# 三条规矩：① 时长与"64s 内已跑完基线＋7 条"这类读数**从现场件里现解析**，不从我脑子里取；
# ② 现场件本身也是档案面，它的字节／md5 要进 CLAIMS（归档自查件会回搜）；③ 守卫的正对照原始件
#    必须带自己的判读行，否则"加了 timeout="只是一句散文。
CTLS = os.path.join(D, "am_s29_hdr_controls.py")
ctlsrc = open(CTLS, encoding="utf-8").read()
HP = os.path.join(D, "_raw_s29_hdr_hang_probe.txt")
hptxt = open(HP, encoding="utf-8").read()
HPB, HPMD5 = str(os.path.getsize(HP)), md5(HP)[:12]
PT = os.path.join(D, "am_s29_timeout_probe.py")
PRO = os.path.join(D, "_raw_s29_timeout_probe.out")
PROE = os.path.join(D, "_raw_s29_timeout_probe.err")
PROB, PROMD5 = str(os.path.getsize(PRO)), md5(PRO)[:12]
assert os.path.getsize(PROE) == 0, "守卫正对照件的 stderr 非 0 字节"
PROVERD = re.search(r"^(TIMEOUT_GUARD_VERDICT = .+)$", open(PRO, encoding="utf-8").read(), re.M).group(1)
# 现场件里 ps 那一段：`PID PPID ETIMES STAT`
_prows = dict((int(m.group(1)), int(m.group(3)))
              for m in re.finditer(r"^(\d+) (\d+) +(\d+) S$", hptxt, re.M))
_ctlpid = int(re.search(r"^(\d+) (\d+) +(\d+) S$", hptxt, re.M).group(1))
_hangpid = [int(m.group(1)) for m in re.finditer(r"^(\d+) %d +(\d+) S$" % _ctlpid, hptxt, re.M)][0]
assert "zctl_s29_hdr_" in hptxt.split("## cmdline")[1].split("## 那条 ssh")[0].split(
    "\n%d: " % _hangpid)[1].splitlines()[0], "现场件里那条卡死的子进程不是扰动副本"
GAP = _prows[_ctlpid] - _prows[_hangpid]          # 控件面起跑→那条副本起跑之间跑完的条数所用时间
HANGS = _prows[_hangpid]                            # 那条副本（含其孙代）卡了多久
HMS = "%dh%02dm" % (HANGS // 3600, (HANGS % 3600) // 60)
assert GAP > 0 and HANGS > 3600, "现场件里的 etimes 读数量级不对（GAP=%s HANGS=%s）" % (GAP, HANGS)
A1B = str(os.path.getsize(os.path.join(D, "_raw_s29_hdr_controls.attempt1-0byte.out")))
A2B = str(os.path.getsize(os.path.join(D, "_raw_s29_hdr_controls.attempt1-0byte.err")))
assert A1B == A2B == "0", "第一次尝试那对原始件不是 0 字节 ⇒ 现场件里 `wc -c` 的读数是假的"
SUBTMO = re.search(r"^SUB_TIMEOUT = (\d+)$", ctlsrc, re.M).group(1)
ENVSTOP = re.search(r"^ENV_FAIL_STOP = (\d+)$", ctlsrc, re.M).group(1)
assert ctlsrc.count("⑨ **子运行不许无上限地等**") == 1 and "九条**仪器自己的**坑" in ctlsrc, \
    "控件面没把这条新学登记成坑⑨（或 docstring 条数没改）"
# 重跑那次的子运行次数／上限／超时计数**从控件面 stdout 现取**，并与源码里的常量互证（两个来源）
_NC = re.search(r"CASES 声明＝(\d+) 条｜每条子运行上限＝(\d+)s｜因超时没跑到的＝(\d+) 条｜"
                r"超时（环境故障，不判红也不判过）＝(\d+) 条", ctxt)
assert _NC, "控件面 stdout 里没有那条超时口径的现取行 ⇒ 守卫的登记没有出处"
NCASES, NLIM, NSKIP, NTO = _NC.groups()
assert NLIM == SUBTMO, "控件面 stdout 的子运行上限（%s）≠ 源码里的 SUB_TIMEOUT（%s）" % (NLIM, SUBTMO)
assert NSKIP == NTO == "0", "这一版控件面有超时（跳过 %s／超时 %s）⇒ 它的 raw 不能当全过的档案" % (NSKIP, NTO)
NSUB = str(int(NCASES) + 1)                         # 基线 1 次＋每条 CASES 一次

CLAIMS = [
    ("HEAD 全哈希", HEAD),
    ("父提交短哈希", "父 `%s`" % PARENT[:7]),
    ("推送区间", "%s..%s" % (PARENT[:7], HEAD[:7])),
    ("入库面 A／M／合计", "新增 (A) **%s**＋修改 (M) **%s**＝**%s** 件" % (NA, NM, NTOT)),
    ("blob＋tree 条目", "tree 条目 **%s**＝blob **%s**＋tree **%s**" % (ENTRIES, NB, NTREE)),
    ("闭合式", "远端 %s == 父提交 %s ＋ 本片新增(A) %s ＝ %s -> PASS" % (NB, NP, NA, NB)),
    ("双向差集", "0/%s" % NB),
    ("全库 SHA 不符", "0/%s" % NSH),
    ("本片单列复核", "%s/%s MATCH" % (SLM, NFL)),
    ("正对照 A", "不符 **%s/%s**" % (CA, NB)),
    ("正对照 A 靶", CTLA_T),
    ("正对照 B", "本地独有 **%s/%s**" % (_CB.group(1), _CB.group(2))),
    ("正对照 C", "**%s == %s**" % (_CC.group(1), _CC.group(2))),
    ("正对照 D", "缺失 **%s/%s**" % (_CD.group(1), _CD.group(2))),
    ("判读行", "LANDING_VERDICT = ALL CHECKS PASS"),
    ("回执三段字节", "表头 **%s**＋正文 **%s**" % (OFF, RAWSZ)),
    ("回执字节与行数", "**%s** B／**%s** 行" % (LOGSZ, LINES)),
    ("cmp 三联", "cmp -i %s:0" % OFF),
    ("顶格字段与声明互证", "顶格字段 **%s** 行＝生成器 `FIELDS` 声明 **%s**" % (NFIELD, NFIELDS)),
    ("尺子 md5 前 12", ruler[:12]),
    ("尺子全树份数", "全树 **%s** 份" % NCOPIES),
    ("尺子本目录旧口径", "本目录旧口径 **%s** 份" % NDIR),
    ("SRC 指纹", FINGER),
    ("表头控件字节与 md5", "**%s** B（md5 前 12 `%s`" % (CTLB, CTLMD5)),
    ("表头控件条数", "**%s** 条扰动、未命中 **0** 条、登记造不出红路 **%s** 条" % (CTLN, CTLCOV)),
    ("表头控件判读", "CONTROL_VERDICT = ALL CONTROLS CAN GO RED"),
    ("提交说明面字节与 md5", "**%s** B（md5 前 12 `%s`" % (MSGB, MSGMD5)),
    ("提交说明面条数", "**%s** 条扰动全红、未扰动基线先 rc=0" % MSGN),
    ("预检面原始件", "预检面原始件 `_raw_s%s_hdr_precheck.out` **%s** B（md5 前 12 `%s`" % (SLICE, PREB, PREMD5)),
    ("写盘面原始件", "写盘面 `_raw_s%s_hdr_write.out` **%s** B（md5 前 12 `%s`" % (SLICE, WRB, WRMD5)),
    ("卡死现场件", "`_raw_s29_hdr_hang_probe.txt` **%s** B（md5 前 12 `%s`）" % (HPB, HPMD5)),
    ("卡死时长", "卡了 **%s**（现场件里那条卡死副本的 etimes＝**%s** s）" % (HMS, HANGS)),
    ("起跑差", "控件面起跑 **%s** s 之后那条副本才起（这段时间里跑完了基线与前面若干条；"
              "到底是第几条没有入档凭据 ⇒ 不写成数字）" % GAP),
    ("第一次尝试的原始件", "那对原始件 **%s** B／**%s** B，改名保留为 "
                        "`_raw_s29_hdr_controls.attempt1-0byte.{out,err}`" % (A1B, A2B)),
    ("守卫常量", "`SUB_TIMEOUT`＝**%s** s、`ENV_FAIL_STOP`＝**%s** 条" % (SUBTMO, ENVSTOP)),
    ("子运行次数", "基线 1＋CASES **%s**＝**%s** 次子运行，本轮超时 **0** 条、因超时没跑到的 **0** 条"
                % (NCASES, NSUB)),
    ("守卫正对照源码", "`am_s29_timeout_probe.py` **%s** B（md5 前 12 `%s`）"
                 % (str(os.path.getsize(PT)), md5(PT)[:12])),
    ("守卫正对照原始件", "`_raw_s29_timeout_probe.out` **%s** B（md5 前 12 `%s`，stderr **0** B），"
                    "判读行 %s" % (PROB, PROMD5, PROVERD)),
    ("守卫正对照复跑", "同件复跑第二趟与第一趟**逐字节相同**（`cmp` rc＝**0**）"),
]




# ============================ 归档自查腿（分片 29）============================
# 上面那一段「取值命令＋字面串格式」与本轮填值器**同源**（逐字节从 /tmp/s29_head.py 的 22–45／71–180／209–255
# 三段切进来），所以本件不是"复述散文"，而是**用同一批命令重新现取一次**再去档案里搜原样。
# 三条规矩（都来自上一片自查件实测到的形状，不是偏好）：
# ① volatile 读数**只断言、不入 stdout**：未入库件数随本件自己的归档件移动、卡上 compute 进程数随他人作业
#    移动、落笔时刻 `date` 每次都变 ⇒ 一旦印出来，"复跑逐字节相同"这条就永远做不到。
# ② 每个 0 都要在同一次运行里配一条**能红**的正对照：这里用四条内存扰动（读数加尾／改一位哈希／
#    抽掉一张档案面／把锚点所在面换成空串），全部不写盘。
# ③ 已入库那把表头生成器的**唯一性锚句**必须跟着复查：§26.31 里引用**上一片**回执名（28 号那份）的
#    顶级子条须恰好 1 条；这条判据的"可通过性"由一条诱饵证明（再造一条引用⇒2 条⇒同一函数必须红）。
FACE_M = ["docs/开发日志.md", "docs/项目开发总览.md", "docs/项目评估与下一步计划_2026-10-06.md",
          ".workbuddy/memory/MEMORY.md"]
FACE_NEW = [".workbuddy/memory/2026-10-12.md",
            "docs/evidence/2026-10-10/am_s29_archive_check.py",
            "docs/evidence/2026-10-10/_raw_s29_archive_check.out",
            "docs/evidence/2026-10-10/_raw_s29_archive_check.err",
            "docs/evidence/2026-10-10/_raw_s29_hdr_after_write.out",
            "docs/evidence/2026-10-10/_raw_s29_hdr_after_write.err"]
DEVLOGF = DEVLOG
OVERF = os.path.join(REPO, "docs/项目开发总览.md")
PLANF = os.path.join(REPO, "docs/项目评估与下一步计划_2026-10-06.md")
DAILYF = os.path.join(REPO, ".workbuddy/memory/2026-10-12.md")
WBIDXF = os.path.join(REPO, ".workbuddy/memory/MEMORY.md")
FACE_NAMES = ["docs/开发日志.md", "docs/项目开发总览.md", "docs/项目评估与下一步计划_2026-10-06.md",
              ".workbuddy/memory/2026-10-12.md", ".workbuddy/memory/MEMORY.md"]
FACE_PATHS = [DEVLOGF, OVERF, PLANF, DAILYF, WBIDXF]
TEXTS = dict(zip(FACE_NAMES, [open(p, encoding="utf-8").read() for p in FACE_PATHS]))
assert len(TEXTS) == 5, "档案面不到 5 张"
PREVLOG = "am_s7_verify%s.log" % PREV


def detector(claims, texts):
    """返回"现取到的字面串在档案里搜不到原样"的那几张声称；搜不到＝写手手抄，本件存在的唯一理由。"""
    return [(lab, lit) for lab, lit in claims
            if not any(lit in t for t in texts.values())]


MISSING = detector(CLAIMS, TEXTS)
assert not MISSING, "有 %d 条现取读数在档案里搜不到原样：%s" % (
    len(MISSING), "、".join("%s=%r" % (l, s[:40]) for l, s in MISSING))
# 每条声称各命中在哪些面上（命中数本身不入 stdout，只做断言：至少一面、至多五面）
for _lab, _lit in CLAIMS:
    _n = sum(1 for _t in TEXTS.values() if _lit in _t)
    assert 1 <= _n <= 5, "%s 命中 %d 面（应当 1..5）" % (_lab, _n)

# ---------------- 四条正对照：全部在内存里扰动，不写盘 ----------------
CTRL = []
# (1) 读数扰动：给每条声称的字面串尾巴上加一个档案里绝不可能出现的记号 ⇒ 全部 39 条必须判缺
_c1 = detector([(l, s + "◇") for l, s in CLAIMS], TEXTS)
assert len(_c1) == len(CLAIMS), "扰动读数后仍有 %d 条没判缺 ⇒ 回搜是空转" % (len(CLAIMS) - len(_c1))
CTRL.append("读数扰动（每条尾巴加一个不可能出现的记号）⇒ 判缺 %d/%d" % (len(_c1), len(CLAIMS)))
# (2) 一位哈希扰动：只改 HEAD 的最后一个十六进制位 ⇒ 恰好那一条判缺
_alt = HEAD[:-1] + ("0" if HEAD[-1] != "0" else "1")
_c2 = detector([("HEAD 全哈希", _alt)], TEXTS)
assert len(_c2) == 1 and _c2[0][0] == "HEAD 全哈希", "一位哈希扰动没让那一条红：%s" % _c2
CTRL.append("一位哈希扰动（HEAD 末位换值）⇒ 判缺 1/1")
# (3) 档案子集扰动：只留总览一张 ⇒ 至少一条判缺（缺的是别的面上才有的那几条）
_c3 = detector(CLAIMS, {FACE_NAMES[1]: TEXTS[FACE_NAMES[1]]})
assert _c3, "抽掉四面只留总览却零缺失 ⇒ 声称并没有分散在各面，普查面过窄"
CTRL.append("档案子集扰动（只留总览）⇒ 判缺 %d/%d" % (len(_c3), len(CLAIMS)))
# (4) 锚点扰动：把**开发日志**那一面换成空文本 ⇒ 只有它承载的那几条必须判缺。
#     为什么不用当日纪现：本件先量过五面各自的"独有声称数"（下面 COVERAGE 那行），结果是
#     **只有开发日志承载独有声称**（其余四面＝同一批 token 的同源摘要），
#     所以"抽掉任意一面都能红"是一条**结构上不可能红**的假闸门——那是判据坏了，不是档案坏了。
_c4 = detector(CLAIMS, dict(TEXTS, **{FACE_NAMES[0]: ""}))
assert _c4, "把开发日志换成空文本却零缺失 ⇒ 没有任何声称只在它里面（普查面选错）"
CTRL.append("锚点扰动（开发日志换成空文本）⇒ 判缺 %d/%d" % (len(_c4), len(CLAIMS)))
# (5) 全集扰动：五面全换成空文本 ⇒ 39 条**全部**判缺（证明绿灯依赖档案文本，不是恒真）
_c5 = detector(CLAIMS, dict((n, "") for n in FACE_NAMES))
assert len(_c5) == len(CLAIMS), "五面全空却只判缺 %d 条 ⇒ detector 不依赖档案文本" % len(_c5)
CTRL.append("全集扰动（五面全换成空文本）⇒ 判缺 %d/%d" % (len(_c5), len(CLAIMS)))
# 每面承载量与"独有声称数"（稳定读数，不是 volatile）：同源摘要这件事要留在档里，不能靠我记
_carriers = dict((lab, [n for n in FACE_NAMES if lit in TEXTS[n]]) for lab, lit in CLAIMS)
assert all(_carriers.values()), "有声称在五面里都搜不到"
COVERAGE = [(_n, len([lab for lab, c in _carriers.items() if _n in c]),
             len([lab for lab, c in _carriers.items() if c == [_n]])) for _n in FACE_NAMES]
COVERED = len([lab for lab, c in _carriers.items() if c])
assert COVERED == len(CLAIMS), "只有 %d/%d 条声称能在五面里搜到" % (COVERED, len(CLAIMS))
UNIQ_TOTAL = sum(u for _, _, u in COVERAGE)
assert 1 <= UNIQ_TOTAL <= len(CLAIMS), "独有数之和 %d 不在 1..%d（至少一条要只长在开发日志里，(4) 才红得了）" % (
    UNIQ_TOTAL, len(CLAIMS))
assert [u for _, _, u in COVERAGE].count(0) == 4, "独有数为 0 的面不是四面 ⇒ 摘要形制变了，(4) 的依据要重量"
CTRL.append("承载量互证（五面全覆盖 %d/%d，独有只长在开发日志那面）⇒ 通过" % (COVERED, len(CLAIMS)))

# ---------------- 已入库表头生成器要用的三条唯一性锚句 ----------------
devtxt = TEXTS[FACE_NAMES[0]]
_secs = re.findall(r"^### 26\.\d+ .*$", devtxt, re.M)
SECN = re.search(r"### 26\.(\d+)", _secs[-1]).group(1)
SEC_BODY = devtxt.split(_secs[-1])[-1]
NEXT_SEC = re.split(r"^### ", devtxt.split(_secs[-1])[-1], maxsplit=1, flags=re.M)[0]


def top_bullets(body):
    """按**本轮形制**切顶级子条：行首 `- **标签**`，其下缩进行并入上一条。"""
    segs = re.split(r"(?m)^- \*\*(.+?)\*\*", body)
    return list(zip(segs[1::2], segs[2::2]))


BULLETS = top_bullets(NEXT_SEC)
PREVCITE = len([1 for a, b in BULLETS if PREVLOG in (a + b)])
assert PREVCITE == 1, "§26.%s 里引用上一片回执名 %s 的顶级子条有 %d 条（须恰好 1）" % (
    SECN, PREVLOG, PREVCITE)
QUOTED = len([1 for a, b in BULLETS if re.search(r"当前「", a + b)])
assert QUOTED == 1, "§26.%s 里带「当前「…」」的钉住口径登记有 %d 条（须恰好 1）" % (SECN, QUOTED)
LESSON = len([1 for a, b in BULLETS if "判据的可通过性" in (a + b)])
assert LESSON == 1, "§26.%s 里 LESSON 锚句有 %d 条（须恰好 1）" % (SECN, LESSON)
# 本件新追加那一条自己**不得**复述这三句（否则已入库那张回执的生成器再也复跑不出来）
own = [b for b in BULLETS if "分片 %s 落地腿＝提交＋推送" % SLICE in b[0]]
assert len(own) == 1, "本轮落地子条在 §26.%s 里不是恰好 1 条：%d" % (SECN, len(own))
assert PREVLOG not in own[0][0] + own[0][1], "本轮落地子条复述了上一片回执名 ⇒ 作废已入库回执的复跑"
assert "当前「" not in own[0][0] + own[0][1] and "判据的可通过性" not in own[0][0] + own[0][1], \
    "本轮落地子条复述了钉住口径／LESSON 锚句 ⇒ 唯一性闸门被本轮自己踩红"
# 诱饵：在内存里再造一条引用上一片回执名的子条 ⇒ 同一条判据必须变红
_bait = NEXT_SEC + "\n- **诱饵**：见 %s。\n" % PREVLOG
assert len([1 for a, b in top_bullets(_bait) if PREVLOG in (a + b)]) == 2, "诱饵没造出 2 条 ⇒ 计数口径坏了"
CTRL.append("回执名唯一性诱饵（内存里再引一条→2 条）⇒ 同一判据当场可红")

# ---------------- volatile 读数：只断言，不入 stdout ----------------
st_lines = sh("git", "-c", "core.quotePath=false", "status", "--porcelain").splitlines()
stat = {}
for l in st_lines:
    if l.strip():
        stat[l[3:].strip()] = l[:2].strip()
for p in FACE_M:
    assert stat.get(p) == "M", "%s 现在不是 (M)：%r" % (p, stat.get(p))
for p in FACE_NEW:
    assert stat.get(p) == "??", "%s 现在不是未跟踪：%r" % (p, stat.get(p))
assert len(FACE_M) == 4 and len(FACE_NEW) == 6, "本轮闭合式的两个分量本身要先对上"
assert sh("git", "-c", "core.quotePath=false", "status", "--porcelain", "--", "src", "tests").strip() == "", \
    "生产树 src+tests 不干净"
assert nv_mine == [] and DIRTY_S == 0, "GPU／生产树前提不成立"
assert re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} [+-]\d{4}$", TIME), "落笔时刻不是 ISO 格式"
assert os.path.getsize(PROE) == 0 and os.path.getsize(RAWE) == 0, "有 stderr 非 0 字节 ⇒ 有警告被吞"

print("归档自查（分片 %s）＝把本轮五张档案面里的每条落地声称由命令现取再回搜原样" % SLICE)
print("  声称条数＝%d｜缺失＝0/%d｜每条命中面数 1..5 全部通过" % (len(CLAIMS), len(CLAIMS)))
print("  正对照共 %d 条（(1)..(5) 五内存扰动＋承载量互证＋回执名唯一性诱饵），全部在同一次运行里各自能红" % len(CTRL))
for _c in CTRL:
    print("  正对照：%s" % _c)
for _n, _c, _u in COVERAGE:
    print("  承载：%s＝%d 条（其中独有 %d 条）" % (_n, _c, _u))
print("  §26.%s 顶级子条＝%d 条；上一片回执名／钉住口径／LESSON 锚句各自恰好 1 条；本轮落地子条不复述它们"
      % (SECN, len(BULLETS)))
print("  五张档案面都在位（四面 (M)＋一面新增）；生产树 src+tests 干净；GPU 上我的解释器所留进程＝0")
print("  volatile 读数（未入库件数、卡上进程数、落笔 date）按口径只断言、不入 stdout ⇒ 本件可复跑逐字节相同")
print("ARCHIVE_VERDICT = ALL CLAIMS LIVE AND PRESENT, ALL CONTROLS CAN GO RED")
