#!/usr/bin/env python3
"""归档腿自查（分片 28）：本轮四份文档新追加的那几段里，每条**落地读数**必须由命令现取并在文中原样出现。

范围照实说清：本件普查的是"落地读数"这一张清单（HEAD／父提交／入库面 A-M-合计／shortstat 原文／
blob＋tree 条目／闭合式／三组带分母的 0／四个正对照读数／判读行／回执三段字节／cmp 三联／顶格字段数与
`FIELDS` 声明数／尺子份数与 md5／SRC 指纹／两面控件面的字节-md5-条数-UNCOVERED-判读行），
**不是**对归档散文做全量出处普查——那是 `am_a3_hdr_audit.py` 对表头面做的事。查法＝每个读数由命令现取，
再按"本轮写在文里的那种说法"拼成字面串去搜；搜不到即拒绝＝写手手抄最常犯的错。

② 复查**下一片那把表头生成器要用的锚点**，并按本轮真实形状改口径：`bullet_for` 要求的是
**同一节内唯一**，不是整份文件唯一。§26.30 里引用本轮回执名的行首子条现有 **2** 条（15:0x 轮为表头埋的
前向引用＋本轮落地子条），所以这一条不能写成「须 1」——写成须 1 就是一条永远红的假闸门。改法＝把"2 条"
当作现取读数钉住，并**要求登记句本身在文里**（那条 ⚠ 说明分片 29 要么另起 §26.31、要么先并引用），
配一条内存扰动＝再造一条引用回执名的子条（3 条）⇒ 同一函数必须变红。钉住口径「当前「…」」与
LESSONLET 锚句仍须各自唯一、f-string 事故句全日志恰好 1 次、事故标本句只在一份生成器里。

③ volatile 读数一律**只断言不入 stdout**（未入库件数随本件自己的归档件移动、卡上 compute 进程总数随他人
作业移动、落笔时刻 `date`），否则"归档件复跑逐字节相同"这条不可能成立。

④ 本轮新增的口径（控件面判"红"按抛出行、多条同抛一句＝共因环境故障、预检面对 BODY_OFFSET 盲目）
都是**散文**，不在本件普查范围；它们各自的凭据在 `_raw_s28_hdr_controls.out` 与 §26.30 那条登记里。

⑤ 自指的那五个读数（本件的段数／声称条数／扰动条数／原始件字节／本件 md5）走**不动点**：它们由总览的收尾
填值从「本件的 stdout／原始件／本件源码」现取，而 stdout **不打印**这些值本身（只打印固定措辞），
所以填完值再跑不会改变 stdout ⇒ 三次复跑逐字节相同与"档案里引用了原始件字节数"可以同时成立。
唯一随阶段变的一趟＝原始件尚不存在的头一趟（自查腿跳过、不影响任何数字）；归档件取原始件已存在之后的那次运行。

运行＝`CUDA_VISIBLE_DEVICES= /tmp/amvenv/bin/python docs/evidence/2026-10-10/am_s28_archive_check.py`
（本件不 import jax：只跑 git／md5sum／cmp／nvidia-smi。）
"""
import glob
import hashlib
import os
import re
import subprocess

REPO = "/home/shy/桌面/2026年开发-显式冲击有限元-物质点法-sph-fvm+am/AM-Qoder"
D = os.path.join(REPO, "docs/evidence/2026-10-10")
LOG = os.path.join(D, "am_s7_verify28.log")
RAWO = os.path.join(D, "_raw_verify28.out")
RAWE = os.path.join(D, "_raw_verify28.err")
HDR = os.path.join(D, "am_s28_hdr.py")
CTLO = os.path.join(D, "_raw_s28_hdr_controls.out")
MSGO = os.path.join(D, "_raw_s28_msg_controls.out")
PILOT = os.path.join(D, "_raw_a3_pilot_amb_g9_norm_t123.out")
DEVLOG = os.path.join(REPO, "docs/开发日志.md")
DAILY = os.path.join(REPO, ".workbuddy/memory/2026-10-10.md")
PLAN = os.path.join(REPO, "docs/项目评估与下一步计划_2026-10-06.md")
OVER = os.path.join(REPO, "docs/项目开发总览.md")
SLICE = "28"
PREV = "27"
LABELS = (r"生成时刻|CMD|ENV|脚本|TREE|SRC指纹|GPU|RUN|判读|四个正对照|分母口径|正文区|复核|"
          r"同片对照|用途|教训")
CHAIN = ["am_s7_verify%s.py" % SLICE, "am_s7_verify%s.log" % SLICE, "_raw_verify%s.out" % SLICE,
         "_raw_verify%s.err" % SLICE, "am_s%s_hdr.py" % SLICE]


def sh(*a):
    return subprocess.run(a, cwd=REPO, capture_output=True, text=True, check=True).stdout


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def block(path, start, stop=None):
    """取文件里"本轮新追加的那一段"：起始标记必须唯一命中，到 stop（或文件末）为止；`**` 原样保留。"""
    s = open(path, encoding="utf-8").read()
    m = list(re.finditer(start, s, re.M))
    assert len(m) == 1, "起始标记 %r 在 %s 里命中 %d 处 ⇒ 无法定位本轮那段" % (start, path, len(m))
    seg = s[m[0].start():]
    if stop:
        k = re.search(stop, seg[len(m[0].group(0)):], re.M)
        assert k, "在 %s 那段之后找不到终止边界 %r" % (path, stop)
        seg = seg[:len(m[0].group(0)) + k.start()]
    return seg


# ---------- 现取：读数一个都不许手打 ----------
HEAD = sh("git", "rev-parse", "HEAD").strip()
PARENT = sh("git", "rev-parse", "HEAD^").strip()
ROWS = [r.split("\t") for r in sh("git", "-c", "core.quotePath=false", "show", "--name-status",
                                  "--format=", "HEAD").splitlines() if r.strip()]
NA = sum(1 for r in ROWS if r[0] == "A")
NM = sum(1 for r in ROWS if r[0] == "M")
NTOT = len(ROWS)
assert NA + NM == NTOT, "入库清单里有 A／M 之外的状态字母 ⇒ 取列口径破了"
SS = sh("git", "diff-tree", "--no-commit-id", "--shortstat", "-r", "HEAD").strip()
body = open(RAWO, encoding="utf-8").read()
assert "LANDING_VERDICT = ALL CHECKS PASS" in body, "回执正文不是全过 ⇒ 不能当落地证明归档"
assert os.path.getsize(RAWE) == 0, "回执正文的 stderr 非 0 字节 ⇒ 有警告被吞"
logtxt = open(LOG, encoding="utf-8").read()
OFF = re.search(r"^BODY_OFFSET=(\d+)$", logtxt, re.M).group(1)
RAWSZ, LOGSZ = str(os.path.getsize(RAWO)), str(os.path.getsize(LOG))
LINES = str(len(logtxt.splitlines()))
assert int(OFF) + int(RAWSZ) == int(LOGSZ), "表头＋正文 ≠ 回执（%s＋%s≠%s）" % (OFF, RAWSZ, LOGSZ)
NFIELD = len([l for l in logtxt[:int(OFF)].splitlines() if re.match(r"^(%s) *:" % LABELS, l)])
NFIELDS = len(re.findall(r'"([^"]+)"', re.search(r"^FIELDS = \[(.*?)\]",
                                                 open(HDR, encoding="utf-8").read(),
                                                 re.M | re.S).group(1)))
CMP = [subprocess.run(["cmp", "-i", i, LOG, RAWO], cwd=REPO, capture_output=True).returncode
       for i in ("%s:0" % OFF, "%d:0" % (int(OFF) + 1), "%s:1" % OFF)]
ruler = md5(os.path.join(D, "am_s7_verify%s.py" % SLICE))
NCOPIES = len([p for p in glob.glob(os.path.join(REPO, "docs/evidence/*/am_s7_verify*.py")) if md5(p) == ruler])
NDIR = len([p for p in glob.glob(os.path.join(D, "am_s7_verify*.py")) if md5(p) == ruler])
FINGER = re.search(r"^SRC指纹 *: *([0-9a-f]{32})", logtxt, re.M).group(1)
now_finger = subprocess.run(["bash", "-c", "find src tests -name '*.py' -not -path '*__pycache__*' "
                             "-print0 | sort -z | xargs -0 md5sum | md5sum | cut -d' ' -f1"],
                            cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
assert FINGER == now_finger, "表头 SRC 指纹与此刻现算不同 ⇒ 生产树在落笔后动过，本轮归档作废"
ctxt = open(CTLO, encoding="utf-8").read()
mtxt = open(MSGO, encoding="utf-8").read()
CTLB, CTLMD5 = str(os.path.getsize(CTLO)), md5(CTLO)[:12]
MSGB, MSGMD5 = str(os.path.getsize(MSGO)), md5(MSGO)[:12]
CTLN = int(re.search(r"扰动条数＝(\d+)", ctxt).group(1))
MSGN = int(re.search(r"扰动条数＝(\d+)", mtxt).group(1))
CTLCOV = int(re.search(r"本轮造不出红路的闸门＝(\d+) 条", ctxt).group(1))
MSGCOV = int(re.search(r"本轮造不出红路的闸门＝(\d+) 条", mtxt).group(1))
CTLVERD = re.search(r"^(CONTROL_VERDICT = .+)$", ctxt, re.M).group(1)
MSGVERD = re.search(r"^(CONTROL_VERDICT = .+)$", mtxt, re.M).group(1)
assert CTLVERD == MSGVERD == "CONTROL_VERDICT = ALL CONTROLS CAN GO RED", "两面控件面的判读行没全红"
assert len([l for l in ctxt.splitlines() if l.startswith("× ")]) == 0, "表头控件面有未命中，不能写成全红"
assert len([l for l in mtxt.splitlines() if l.startswith("× ")]) == 0, "提交说明面有未命中，不能写成全红"
g = lambda pat: re.search(pat, body).group(1)
ENTRIES, NB, NTREE = re.search(r"tree 条目 (\d+) = blob (\d+) \+ tree (\d+)", body).groups()
NP, DL, DR = g(r"父提交 = (\d+)"), g(r"本地独有 (\d+)/"), g(r"远端独有 (\d+)/")
MS, NSH, SLM, NFL = g(r"blob SHA 不符 (\d+)/"), g(r"blob SHA 不符 \d+/(\d+)"), g(r"MATCH (\d+)/"), g(r"本片 (\d+) 文件")
CA = g(r"正对照 A（远端 \S+ 的 sha 首位改 f）⇒ 不符 (\d+)/")
CTLA_T = re.search(r"正对照 A（远端 (\S+) 的 sha 首位改 f）", body).group(1)
_CB = re.search(r"正对照 B（远端删一条路径）⇒ 本地独有 (\d+)/(\d+)，远端独有 (\d+)/(\d+)", body)
_CC = re.search(r"正对照 C（闭合式右边 \+1）⇒ (\d+) == (\d+) 判", body)
_CD = re.search(r"正对照 D（远端删本片一条路径）⇒ 本片缺失 (\d+)/(\d+)", body)
assert _CB and _CC and _CD, "正文 ⑤ 段的对照读数形状不认识 ⇒ 拼不出可查的字面串"
assert int(CA) == 1 and int(_CB.group(1)) == 1 and int(_CD.group(1)) == 1 \
    and int(_CC.group(2)) == int(NB) + 1, "正对照的读数本身不对 ⇒ 归档里那句「能红」没凭据"
assert (int(_CB.group(2)), int(_CB.group(3)), int(_CB.group(4)), int(_CD.group(2)), int(_CC.group(1))) == \
    (int(NB), 0, int(NB) - 1, int(NFL), int(NB)), "对照的分母口径与正文不符"
assert int(DL) == 0 and int(DR) == 0 and int(MS) == 0 and int(SLM) == int(NFL), \
    "正文里那几个 0 现在不是 0 了 ⇒ 被测树/远端已移动，本轮归档声称全部作废"
assert int(NB) == int(NP) + NA and int(ENTRIES) == int(NB) + int(NTREE), "闭合式／条目式现在不闭合"
ptxt = open(PILOT, encoding="utf-8").read()
TOUCH = re.findall(r"max touched=(\d+)／窗口行数=(\d+)", ptxt)
assert len(TOUCH) == 3 and TOUCH[0][0] == "0", "pilot 归档件的 G9 触边行数不认识（须 3 条且 N=1 为 0）"
G9 = (TOUCH[1][0], TOUCH[2][0])

# ---------- ① 读数清单：字面串按"本轮写进文里的那种说法"拼，值全部现取 ----------
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
    ("尺子全树份数", "现算共 **%s** 份" % NCOPIES),
    ("尺子本目录旧口径", "只 glob 本目录的旧口径此处只数出 **%s** 份" % NDIR),
    ("SRC 指纹", FINGER),
    ("表头控件字节与 md5", "**%s** B（md5 前 12 `%s`" % (CTLB, CTLMD5)),
    ("表头控件条数", "**%s** 条扰动、未命中 **0** 条、登记造不出红路 **%s** 条" % (CTLN, CTLCOV)),
    ("表头控件判读", "CONTROL_VERDICT = ALL CONTROLS CAN GO RED"),
    ("提交说明面字节与 md5", "**%s** B（md5 前 12 `%s`" % (MSGB, MSGMD5)),
    ("提交说明面条数", "**%s** 条扰动全红、未扰动基线先 rc=0" % MSGN),
]
DEV_BLOCK = block(DEVLOG, r"^- \*\*分片 28 落地腿")


def missing_claims(claims, text):
    """返回文中搜不到原样的声称（空＝每条读数都在归档里落了原样）。"""
    return [(n, v) for n, v in claims if v not in text]


# 另外三份档案从**同一套 token** 渲染，但句式各自收缩 ⇒ 各按各的写法取子集（值仍全部现取）。
# 子集里必须留**名值对**而不是值：missing_claims 是按 (名, 值) 解包的，只传值会把字符串当行内元组拆。
PAIRS = dict((n, (n, v)) for n, v in CLAIMS)
SHORTSTAT = [("shortstat 原文", SS)]
SELFCLAIMS = []
RAWC = os.path.join(D, "_raw_s28_archive_check.out")
if os.path.exists(RAWC):
    # 自查件自己的声称：值从**已归档的原始件**里 grep（第一跑没有原始件 ⇒ 这一腿如实跳过）
    rtxt = open(RAWC, encoding="utf-8").read()
    SELFCLAIMS = [
        ("自查件段数", "%s 段新追加" % re.search(r"新追加段数＝(\d+)", rtxt).group(1)),
        ("自查件声称条数", "%s 条落地声称" % re.search(r"现取声称 (\d+) 条", rtxt).group(1)),
        ("自查件扰动条数", "%s 条读数扰动" % re.search(r"内存扰动 (\d+) 条读数闸门", rtxt).group(1)),
        ("自查件原始件字节", "原始件 `_raw_s28_archive_check.out` %s B" % os.path.getsize(RAWC)),
        ("自查件 md5 前 12", "`am_s28_archive_check.py`（md5 前 12 `%s`"
                            % md5(os.path.join(D, "am_s28_archive_check.py"))[:12]),
    ]
    # 「五形」这句是打进 stdout 的**固定措辞**，必须有断言钉住，否则加了第六形措辞就烂在档里
    assert len(SELFCLAIMS) == 5, "自查件自身声称不是五形（%d）⇒ stdout 里那句「五形」成假话" % len(SELFCLAIMS)
OTHER = {"当日纪实": (block(DAILY, r"^## \d{4}-\d\d-\d\d \d\d:\d\d:\d\d \+\d{4} 分片 28 落地腿"),
                      [PAIRS[k] for k in ("入库面 A／M／合计", "闭合式", "双向差集",
                                          "blob＋tree 条目", "回执三段字节", "cmp 三联",
                                          "尺子 md5 前 12", "表头控件判读")]),
         "计划文档": (block(PLAN, r"^### §W 补记 3"),
                      [PAIRS[k] for k in ("入库面 A／M／合计", "闭合式", "双向差集",
                                          "本片单列复核", "回执三段字节", "cmp 三联", "blob＋tree 条目",
                                          "顶格字段与声明互证", "尺子 md5 前 12")]
                      + SHORTSTAT
                      + [("G9 触边读数", "N=1 触边 **0**、N=2 **%s**、N=3 **%s**" % G9)]),
         "总览条目": (block(OVER, r"^- \*\*2026-10-10（分片 28 落地腿", stop=r"^## 7\. 文档索引$"),
                          [PAIRS[k] for k in ("HEAD 全哈希", "入库面 A／M／合计", "闭合式",
                                              "双向差集", "本片单列复核", "正对照 A",
                                              "正对照 C", "正对照 D", "判读行", "回执三段字节", "cmp 三联",
                                              "顶格字段与声明互证", "尺子 md5 前 12", "表头控件判读")]
                      + SHORTSTAT + SELFCLAIMS)}
ALLOWED_TOKENS = {"CHKMD5", "CHKB", "NSEC", "NCLAIM", "NPERT", "UNT", "MOD", "TOT"}

miss_dev = missing_claims(CLAIMS, DEV_BLOCK)
assert not miss_dev, "开发日志本轮那段里搜不到这些读数原样：%s" % miss_dev
miss_other = {k: missing_claims(v, t) for k, (t, v) in OTHER.items()}
bad_other = {k: m for k, m in miss_other.items() if m}
assert not bad_other, "另外三份文档里缺读数：%s" % bad_other
assert NFIELD == NFIELDS, "表头顶格字段行数 %s ≠ `FIELDS` 声明数 %s ⇒ 两文件耦合断了" % (NFIELD, NFIELDS)
assert CMP == [0, 1, 1], "cmp 三联不齐（同偏移须 0、两侧各 +1 须 1）：%s" % CMP
tok = {k: sorted(set(re.findall(r"@@([A-Za-z0-9_]+)@@", t))) for k, (t, v) in OTHER.items()}
tok["开发日志"] = sorted(set(re.findall(r"@@([A-Za-z0-9_]+)@@", DEV_BLOCK)))
bad_tok = {k: v for k, v in tok.items() if not set(v) <= ALLOWED_TOKENS}
assert not bad_tok, "本轮那段里有非收尾清单内的占位符：%s" % bad_tok

# ---------- ② 下一片的锚点（按本轮真实形制：节内引用数＝现取，不写死"须 1"）----------
DEV = open(DEVLOG, encoding="utf-8").read()


def last_section(dev):
    secs = list(re.finditer(r"^### 26\.(\d+)[^\n]*$", dev, re.M))
    assert secs, "开发日志里取不到 `### 26.NN` 标题 ⇒ 节号是凭记忆的"
    tail = dev[secs[-1].end():]
    k = tail.find("\n### ")
    return secs[-1].group(1), (tail[:k] if k != -1 else tail)


def bullets(seg):
    letters = [m.group(1) for m in re.finditer(r"^- \*\*(.+?)\*\*", seg, re.M)]
    out = []
    for i, l in enumerate(letters):
        s = seg.index("- **%s**" % l)
        e = seg.index("- **%s**" % letters[i + 1]) if i + 1 < len(letters) else len(seg)
        out.append((l, seg[s:e]))
    return out


def anchor_failures(dev):
    secn, sec = last_section(dev)
    bs = bullets(sec)
    assert len([l for l, _ in bs]) == len(set(l for l, _ in bs)), "§26.%s 有同名子条标签" % secn
    bad = []
    hit = [l for l, s in bs if "am_s7_verify%s.log" % SLICE in s]
    if len(hit) != NCITE:
        bad.append("本轮回执名被子条 %s 引用（本轮现数＝%s 条）" % (hit, NCITE))
    reg = [l for l, s in bs if "⚠ 下一片的形制约束" in s]
    if len(reg) != 1:
        bad.append("⚠ 那句形制登记命中子条 %s（须恰好 1 条）" % reg)
    for what, pat in (("钉住口径「当前「…」」", r"当前「([^」]+)」"),
                      ("LESSONLET 锚句", "判据的可通过性")):
        h = [l for l, s in bs if re.search(pat, s)]
        if len(h) != 1:
            bad.append("%s 命中子条 %s（须唯一）" % (what, h))
    n = len(list(re.finditer(r"f-string 把 `\{32\}` 当成\*\*替换字段", dev)))
    if n != 1:
        bad.append("f-string 事故句在整份日志里出现 %d 次（须 1）" % n)
    return bad, secn


NCITE = len([l for l, s in bullets(last_section(DEV)[1]) if "am_s7_verify%s.log" % SLICE in s])
assert NCITE >= 1, "§26.%s 里没有子条引用本轮回执名" % last_section(DEV)[0]
anch, SECN = anchor_failures(DEV)
assert not anch, "下一片要用的锚点被本轮碰坏：%s" % anch
spec = [os.path.relpath(p, REPO) for p in glob.glob(os.path.join(REPO, "docs/evidence/*/am_s*_hdr.py"))
        if "本片的实物证据＝**改之前**那句" in open(p, encoding="utf-8").read()]
assert len(spec) == 1, "含事故标本句的生成器有 %d 份（须恰好 1）：%s" % (len(spec), spec)
assert not glob.glob(os.path.join(D, "zctl_*")), "扰动副本没被清掉"
DIRTY_S = len([l for l in sh("git", "-c", "core.quotePath=false", "status", "--porcelain",
                             "--", "src", "tests").splitlines() if l.strip()])
assert DIRTY_S == 0, "生产树 src+tests 有 %d 处改动 ⇒ 本轮归档腿不该碰求解器" % DIRTY_S
OWNCHAIN = len([p for _, p in ROWS if os.path.basename(p) in CHAIN])
assert OWNCHAIN == 0, "本片回执链里有 %d 件已在 HEAD ⇒ 「留给分片 29」那句成了假话" % OWNCHAIN
PREVCHAIN = len([p for _, p in ROWS if os.path.basename(p) in
                 [c.replace(SLICE, PREV) for c in CHAIN]])
st = sh("git", "-c", "core.quotePath=false", "status", "--porcelain").splitlines()
UNT, MOD = len([l for l in st if l.startswith("??")]), len([l for l in st if l[:2] not in ("??", "")])
nv = [l.strip() for l in subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                                        capture_output=True, text=True).stdout.splitlines() if l.strip()]
nv_mine = [p for p in nv if "/tmp/amvenv" in open("/proc/%s/cmdline" % p, "rb").read().decode("utf-8", "replace")]
assert nv_mine == [], "我的解释器在卡上留有进程 %s ⇒ 与「不动 GPU」冲突" % nv_mine

# ---------- ③ 内存扰动：同一个函数必须返回失败 ----------
CL = dict(CLAIMS)


def swap(name, new):
    """把某条声称的字面串换成假值；先断言它**本轮确实在文里**（没有目标＝空转的红路）。"""
    lit = CL[name]
    assert lit in DEV_BLOCK, "扰动目标「%s」不在本轮那段里 ⇒ 这条红路没有靶" % name
    return "%s 换成假值" % name, DEV_BLOCK.replace(lit, new)


PERT = [
    swap("闭合式", "远端 %s == 父提交 %s ＋ 本片新增(A) %s ＝ %s -> PASS" % (NB, NP, NA, int(NB) + 1)),
    swap("回执三段字节", "表头 **%s**＋正文 **%s**" % (int(OFF) + 1, RAWSZ)),
    swap("尺子全树份数", "现算共 **%s** 份" % (NCOPIES - 1)),
    swap("SRC 指纹", FINGER[:-1] + "a"),
    swap("顶格字段与声明互证", "顶格字段 **%s** 行＝生成器 `FIELDS` 声明 **%s**" % (NFIELD, NFIELDS + 1)),
    swap("正对照 A", "不符 **0/%s**" % NB),
    swap("表头控件字节与 md5", "**%s** B（md5 前 12 `%s`" % (CTLB, CTLMD5[:-1] + "a")),
    swap("本片单列复核", "%s/%s MATCH" % (int(SLM) - 1, NFL)),
    swap("正对照 D", "缺失 **%s/%s**" % (_CD.group(1), int(_CD.group(2)) + 1)),
]
for name, mutated in PERT:
    got = missing_claims(CLAIMS, mutated)
    assert got, name + "：扰动后仍全绿 ⇒ 该闸门空转"
# 三份档案各自的子集也必须能红（否则子集只是"顺带过了"，没有靶）
for k, (t, v) in OTHER.items():
    assert v, "%s：子集是空表 ⇒ 这一腿没有闸门" % k
    lit = dict(v)["闭合式"] if "闭合式" in dict(v) else v[0][1]
    got = missing_claims(v, t.replace(lit, lit[:-3] + "X)"))
    assert got, "%s：把闭合式／首个读数改坏后仍全绿 ⇒ 该子集空转" % k
_secn, _sec = last_section(DEV)
# 锚点红路 ①＝**再造一条**引用本轮回执名的子条（本轮真实形状是 2 条，所以 3 条必须红）
bait = _sec + ("\n- **诱饵**：这一子条也引用一次 am_s7_verify%s.log，且另起一处「当前「不动 GPU」」口径\n" % SLICE)
bad2, _ = anchor_failures(DEV.replace(_sec, bait, 1))
assert bad2, "把本轮回执名再抄进第三个子条后锚点闸门没变红 ⇒ 唯一性闸门空转"
assert len([m for m in bad2 if "被子条" in m]) == 1, "锚点扰动没抓到回执引用数：%s" % bad2
assert len([m for m in bad2 if "钉住口径" in m]) == 1, "锚点扰动没抓到钉住口径唯一性：%s" % bad2
# 锚点红路 ②＝把 ⚠ 那句登记抹掉（读数还在，但下一片的读者拿不到口径）
bad3, _ = anchor_failures(DEV.replace("⚠ 下一片的形制约束", "⚠ 下一片无形制约束", 1))
assert len([m for m in bad3 if "形制登记" in m]) == 1, "抹掉登记句后锚点闸门没变红：%s" % bad3
NANCH = 2

print("新追加段数＝%d 段（开发日志本轮子条＋%s；各段起始标记各唯一命中 1 处）"
      % (1 + len(OTHER), "、".join(OTHER)))
print("开发日志那段：现取声称 %d 条全部命中原样｜HEAD＝%s 父＝%s｜A／M／合计＝%s／%s／%s"
      % (len(CLAIMS), HEAD[:7], PARENT[:7], NA, NM, NTOT))
print("另外三份各查 %s 条读数，缺失 0 条" % "/".join(str(len(v)) for k, (t, v) in OTHER.items()))
print("落地证明现算：tree 条目 %s＝blob %s＋tree %s｜闭合式 %s==%s＋%s｜差集 %s/%s 与 %s/%s｜SHA 不符 %s/%s｜本片 %s/%s MATCH"
      % (ENTRIES, NB, NTREE, NB, NP, NA, DL, NB, DR, NB, MS, NSH, SLM, NFL))
print("正对照现算：A 不符 %s/%s（靶 %s）｜B %s/%s＋%s/%s｜C %s==%s｜D %s/%s"
      % (CA, NB, CTLA_T, _CB.group(1), _CB.group(2), _CB.group(3), _CB.group(4),
         _CC.group(1), _CC.group(2), _CD.group(1), _CD.group(2)))
print("回执 %s B／%s 行＝表头 %s＋正文 %s｜cmp 三联＝%s｜顶格字段 %s＝FIELDS 声明 %s"
      % (LOGSZ, LINES, OFF, RAWSZ, CMP, NFIELD, NFIELDS))
print("尺子全树副本 %s 份（本目录口径 %s，md5 前 12 %s）｜SRC 指纹 %s｜表头控件 %s B／%s 条／UNCOVERED %s｜"
      "提交说明 %s B／%s 条／UNCOVERED %s" % (NCOPIES, NDIR, ruler[:12], FINGER, CTLB, CTLN, CTLCOV,
                                              MSGB, MSGN, MSGCOV))
print("下一片锚点：§26.%s 里 am_s7_verify%s.log 被 %s 条子条引用（本轮真实形状，登记句在文里）｜标本生成器＝%s｜"
      "zctl 残留 0" % (SECN, SLICE, NCITE, spec[0]))
print("入库节奏：本片回执链 5 件在 HEAD 清单里 %s/5 件 ⇒ 交分片 %s｜上一片链在 HEAD 里 %s/5 件（已入库）"
      % (OWNCHAIN, int(SLICE) + 1, PREVCHAIN))
print("内存扰动 %d 条读数闸门＋%d 条档案子集扰动＋%d 条锚点闸门全部变红"
      % (len(PERT), len(OTHER), NANCH))
print("收尾占位符：四段残留的 token 全落在收尾清单 %d 项内（各段明细随填值那一趟变＝volatile，只断言不入 stdout）"
      % len(ALLOWED_TOKENS))
print("自查件自身声称＝五形（段数／声称条数／读数扰动条数／原始件字节／本件 md5 前 12），值全部 grep 自本件的归档原始件；"
      "原始件不存在的头一趟此腿跳过")
print("钉住：我的解释器在卡上留有 0 个 compute 进程（断言，总数不入档）；生产树 src+tests 改动 0 件")
print("ARCHIVE_VERDICT = LANDING READINGS AND NEXT-SLICE ANCHORS PASS")
