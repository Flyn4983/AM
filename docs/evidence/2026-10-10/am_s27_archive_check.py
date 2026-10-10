#!/usr/bin/env python3
"""归档腿自查：本轮四份文档新追加的句子里，每条**落地读数**必须由命令现取、并在文中原样出现。

范围照实说清：本件普查的是"落地读数"这一张清单（HEAD／父提交／入库面 A-M-合计／shortstat 原文／
blob-tree 条目／闭合式／三组带分母的 0／回执三段字节／cmp 三联／顶格字段数与 `FIELDS` 声明数／
尺子份数与 md5／SRC 指纹／控件扰动数／判读行）。**不是**对归档散文做全量出处普查——那是
`am_a3_hdr_audit.py` 对表头面做的事（句式多样、数字与叙述混排，一趟做不全，硬做只会造出一条
看着绿却什么都没验的闸门）。查法＝每个读数**由命令现取**，再按"本轮写在文里的那种说法"拼成字面串去搜；
搜不到即拒绝＝写手手抄最常犯的错（把上一片的数当这一片的、或把随片腐烂的句子原样留下）。

另两件事：
② 复查**下一片那把表头生成器要用的锚点**：`am_s7_verify27.log` 在本轮节里必须唯一被子条 (m) 引用、
   钉住口径「当前「…」」与 LESSONLET 锚句必须各自唯一、f-string 事故句全日志恰好 1 次、
   事故标本句只在一份生成器里 ⇒ 本轮追加若碰坏任一条，分片 28 会当场拒绝归档；与其那时再查，不如现在红。
③ 每条闸门配一条**内存里**的扰动（把某读数从文本改掉／再往第二个子条抄一遍回执名），必须让
   **同一个函数**返回失败＝闸门能红不是摆设。扰动只改内存字符串，不落盘、不改文档。
④ stdout 会被 `cp -p` 当原始件归档 ⇒ ** volatile 读数一律不进 stdout**：卡上 compute 进程总数随他人作业移动、
   未入库件数随本件自己的归档件移动，两者都只做断言（我的解释器须 0、本片链在 HEAD 须 0 件），印出来的
   只有稳定读数 ⇒ 归档件复跑逐字节相同这条才可能成立（本轮实测三次复跑同值，见日志）。

运行＝`CUDA_VISIBLE_DEVICES= /tmp/amvenv/bin/python docs/evidence/2026-10-10/am_s27_archive_check.py`
（本件不 import jax：只跑 git／md5sum／cmp／nvidia-smi；带空串是习惯性钉住而不是必需，照实说明。）
"""
import glob
import os
import re
import subprocess

REPO = "/home/shy/桌面/2026年开发-显式冲击有限元-物质点法-sph-fvm+am/AM-Qoder"
D = os.path.join(REPO, "docs/evidence/2026-10-10")
LOG = os.path.join(D, "am_s7_verify27.log")
RAWO = os.path.join(D, "_raw_verify27.out")
RAWE = os.path.join(D, "_raw_verify27.err")
HDR = os.path.join(D, "am_s27_hdr.py")
CTLRAW = os.path.join(D, "_raw_s27_hdr_controls.out")
MSGRAW = os.path.join(D, "_raw_s27_msg_controls.out")
DEVLOG = os.path.join(REPO, "docs/开发日志.md")
DAILY = os.path.join(REPO, ".workbuddy/memory/2026-10-10.md")
PLAN = os.path.join(REPO, "docs/项目评估与下一步计划_2026-10-06.md")
OVER = os.path.join(REPO, "docs/项目开发总览.md")
LABELS = (r"生成时刻|CMD|ENV|脚本|TREE|SRC指纹|GPU|RUN|判读|四个正对照|分母口径|正文区|复核|"
          r"同片对照|用途|教训")


def sh(*args):
    return subprocess.run(args, cwd=REPO, capture_output=True, text=True, check=True).stdout


def md5(path):
    return subprocess.run(["md5sum", path], capture_output=True,
                          text=True).stdout.split()[0]


def block(path, start, stop=None):
    """取文件里"本轮新追加的那一段"：起始标记必须唯一命中，到 stop（或文件末）为止。"""
    s = open(path, encoding="utf-8").read()
    m = list(re.finditer(start, s, re.M))
    assert len(m) == 1, "起始标记 %r 在 %s 里命中 %d 处 ⇒ 无法定位本轮那段" % (start, path, len(m))
    seg = s[m[0].start():]
    if stop:
        k = re.search(stop, seg[m[0].end() - m[0].start():], re.M)
        assert k, "在 %s 那段之后找不到终止边界 %r" % (path, stop)
        seg = seg[:m[0].end() - m[0].start() + k.start()]
    return seg.replace("**", "")  # 只剥强调标记，其余逐字节按原文（写手用不用加粗不是读数）


# ---------- 现取：读数一个都不许手打 ----------
HEAD = sh("git", "rev-parse", "HEAD").strip()
PARENT = sh("git", "rev-parse", "HEAD^").strip()
SUBJ = sh("git", "log", "-1", "--format=%s").strip()
SLICE = re.search(r"S7 分片 (\d+)", SUBJ).group(1)
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
assert int(OFF) + int(RAWSZ) == int(LOGSZ), "表头＋正文 ≠ 回执（%s＋%s≠%s）" % (OFF, RAWSZ, LOGSZ)
NFIELD = len([l for l in logtxt[:int(OFF)].splitlines() if re.match(r"^(%s) *:" % LABELS, l)])
NFIELDS = len(re.findall(r'"([^"]+)"', re.search(r"^FIELDS = \[(.*?)\]",
                                                 open(HDR, encoding="utf-8").read(),
                                                 re.M | re.S).group(1)))
CMP = [subprocess.run(["cmp", "-i", i, LOG, RAWO], cwd=REPO,
                      capture_output=True).returncode
       for i in ("%s:0" % OFF, "%d:0" % (int(OFF) + 1), "%s:1" % OFF)]
ruler = md5(os.path.join(D, "am_s7_verify%s.py" % SLICE))
NCOPIES = len([p for p in glob.glob(os.path.join(REPO, "docs/evidence/*/am_s7_verify*.py"))
               if md5(p) == ruler])
FINGER = subprocess.run(["bash", "-c", "find src tests -name '*.py' -not -path '*__pycache__*' "
                         "-print0 | sort -z | xargs -0 md5sum | md5sum | cut -d' ' -f1"],
                        cwd=REPO, capture_output=True, text=True).stdout.strip()
NPERT = len(re.findall(r"^  C\d+ ", open(CTLRAW, encoding="utf-8").read(), re.M))
NBASE = int(re.search(r"真件基线 (\d+) 条全过", open(MSGRAW, encoding="utf-8").read()).group(1))
NCTLBAD = int(re.search(r"坏件 (\d+) 条", open(MSGRAW, encoding="utf-8").read()).group(1))
g = lambda pat: re.search(pat, body).group(1)
NB, NTREE, ENTRIES = g(r"blob (\d+) \+ tree"), g(r"\+ tree (\d+)"), g(r"tree 条目 (\d+)")
NP, DL, DR = g(r"父提交 = (\d+)"), g(r"本地独有 (\d+)/"), g(r"远端独有 (\d+)/")
MS, NSH, SLM, NFL = g(r"blob SHA 不符 (\d+)/"), g(r"blob SHA 不符 \d+/(\d+)"), g(r"MATCH (\d+)/"), g(r"本片 (\d+) 文件")
CA = g(r"正对照 A（远端 \S+ 的 sha 首位改 f）⇒ 不符 (\d+)/")
_CB = re.search(r"正对照 B（远端删一条路径）⇒ 本地独有 (\d+)/(\d+)，远端独有 (\d+)/(\d+)", body)
_CC = re.search(r"正对照 C（闭合式右边 \+1）⇒ (\d+) == (\d+) 判", body)
_CD = re.search(r"正对照 D（远端删本片一条路径）⇒ 本片缺失 (\d+)/(\d+)", body)
assert _CB and _CC and _CD, "正文 ⑤ 段的三条对照读数形状不认识 ⇒ 拼不出可查的字面串"
assert int(CA) == 1 and int(_CB.group(1)) == 1 and int(_CD.group(1)) == 1 and int(_CC.group(2)) == int(NB) + 1, \
    "正对照的读数本身不对（A/B/D 须各 1、C 须＝blob+1）⇒ 归档里那句「能红」没凭据"
assert int(_CB.group(2)) == int(NB) and int(_CB.group(3)) == 0 and int(_CB.group(4)) == int(NB) - 1, \
    "正对照 B 的分母口径与正文不符 ⇒ 本件对 B 的字面串拼法要重新现取"
assert int(_CD.group(2)) == int(NFL), "正对照 D 的分母 ≠ 本片文件数"
assert int(DL) == 0 and int(DR) == 0 and int(MS) == 0 and int(SLM) == int(NFL), \
    "正文里那几个 0 现在不是 0 了 ⇒ 被测树/远端已移动，本轮归档声称全部作废"

# ---------- ① 读数清单：字面串按"本轮写进文里的那种说法"拼，值全部现取 ----------
CLAIMS = [
    ("HEAD 全哈希", HEAD),
    ("父提交全哈希", PARENT),
    ("推送区间", "%s..%s" % (PARENT[:7], HEAD[:7])),
    ("入库面 A／M／合计", "新增 (A) %s ＋ 修改 (M) %s＝%s 件" % (NA, NM, NTOT)),
    ("shortstat 原文", SS),
    ("blob＋tree 条目", "tree 条目 %s＝blob %s＋tree %s" % (ENTRIES, NB, NTREE)),
    ("闭合式", "远端 %s == 父提交 %s ＋ 本片新增(A) %s ＝ %s -> PASS" % (NB, NP, NA, NB)),
    ("双向差集", "0/%s" % NB),
    ("全库 SHA 不符", "0/%s" % NSH),
    ("本片单列复核", "%s/%s MATCH" % (SLM, NFL)),
    ("正对照 A", "不符 %s/%s" % (CA, NB)),
    ("正对照 B", "本地独有 %s/%s、远端独有 %s/%s" % (_CB.group(1), _CB.group(2), _CB.group(3), _CB.group(4))),
    ("正对照 C", "%s == %s 判 FAIL" % (_CC.group(1), _CC.group(2))),
    ("正对照 D", "缺失 %s/%s" % (_CD.group(1), _CD.group(2))),
    ("判读行", "LANDING_VERDICT = ALL CHECKS PASS"),
    ("回执三段字节", "%s 字节＝表头 %s＋正文 %s" % (LOGSZ, OFF, RAWSZ)),
    ("cmp 三联", "cmp -i %s:0" % OFF),
    ("顶格字段与声明互证", "顶格字段 %s 行＝生成器 `FIELDS` 声明 %s" % (NFIELD, NFIELDS)),
    ("尺子 md5 前 12", ruler[:12]),
    ("尺子全树份数", "现算 %s 份逐字节副本" % NCOPIES),
    ("SRC 指纹", FINGER),
    ("表头控件扰动数", "%s/%s 条扰动" % (NPERT, NPERT)),
    ("表头控件判读", "CONTROL_VERDICT = ALL CONTROLS CAN GO RED"),
    ("提交说明面基线与坏件", "真件基线 %s 条全过／坏件 %s 条全拒" % (NBASE, NCTLBAD)),
    ("提交说明面判读", "CONTROL_VERDICT = ALL DRY-RUN GATES CAN GO RED"),
]
DEV_BLOCK = block(DEVLOG, r"^\*\*\(m\) 分片 27 落地腿")


def missing_claims(claims, text):
    """返回文中搜不到原样的声称（空＝每条读数都在归档里落了原样）。"""
    return [(n, v) for n, v in claims if v not in text]


OTHER = {"当日纪实": (block(DAILY, r"^### 13:35 分片 27 落地腿"),
                      [("HEAD 短哈希", HEAD[:7]), ("blob", NB), ("回执字节", LOGSZ),
                       ("尺子份数", "现算 %s 份逐字节副本" % NCOPIES), ("父提交 blob", NP)]),
         "计划文档": (block(PLAN, r"^### §W 补记 2"),
                      [("HEAD 短哈希", HEAD[:7]), ("blob", NB), ("闭合式左半", "远端 %s == 父提交 %s" % (NB, NP)),
                       ("回执字节", LOGSZ), ("cmp 三联", "cmp -i %s:0" % OFF)]),
         "总览条目": (block(OVER, r"^- \*\*2026-10-10（分片 27 落地腿＝提交＋推送", stop=r"^---$"),
                      [("HEAD 短哈希", HEAD[:7]), ("blob", NB), ("回执字节", LOGSZ),
                       ("顶格字段与声明互证", "顶格字段 %s 行＝生成器 `FIELDS` 声明 %s" % (NFIELD, NFIELDS)),
                       ("尺子份数", "现算 %s 份逐字节副本" % NCOPIES)])}

miss_dev = missing_claims(CLAIMS, DEV_BLOCK)
assert not miss_dev, "开发日志本轮那段里搜不到这些读数原样：%s" % miss_dev
miss_other = {k: missing_claims(v, t) for k, (t, v) in OTHER.items()}
bad_other = {k: m for k, m in miss_other.items() if m}
assert not bad_other, "另外三份文档里缺读数：%s" % bad_other
assert NFIELD == NFIELDS, "表头顶格字段行数 %s ≠ `FIELDS` 声明数 %s ⇒ 两文件耦合断了" % (NFIELD, NFIELDS)
assert CMP == [0, 1, 1], "cmp 三联不齐（同偏移须 0、两侧各 +1 须 1）：%s" % CMP

# ---------- ② 下一片的锚点 ----------
DEV = open(DEVLOG, encoding="utf-8").read()


def last_section(dev):
    secs = list(re.finditer(r"^### 26\.(\d+)[^\n]*$", dev, re.M))
    assert secs, "开发日志里取不到 `### 26.NN` 标题 ⇒ 节号是凭记忆的"
    tail = dev[secs[-1].end():]
    k = tail.find("\n### ")
    return secs[-1].group(1), (tail[:k] if k != -1 else tail)


def bullets_with(seg, pattern):
    letters = [m.group(1) for m in re.finditer(r"^\*\*\((\w)\)[^\n]*$", seg, re.M)]
    out = []
    for i, l in enumerate(letters):
        s = seg.index("**(%s)" % l)
        e = seg.index("**(%s)" % letters[i + 1]) if i + 1 < len(letters) else len(seg)
        if re.search(pattern, seg[s:e]):
            out.append(l)
    return out


def anchor_failures(dev):
    secn, sec = last_section(dev)
    bad = []
    hit = bullets_with(sec, re.escape("am_s7_verify%s.log" % SLICE))
    if hit != ["m"]:
        bad.append("本轮回执名被子条 %s 引用（须恰好 ['m']）" % hit)
    for what, pat in (("钉住口径「当前「…」」", r"当前「([^」]+)」"),
                      ("LESSONLET 锚句", "极性翻转不是扰动")):
        h = bullets_with(sec, pat)
        if len(h) != 1:
            bad.append("%s 命中子条 %s（须唯一）" % (what, h))
    n = len(list(re.finditer(r"f-string 把 `\{32\}` 当成\*\*替换字段", dev)))
    if n != 1:
        bad.append("f-string 事故句在整份日志里出现 %d 次（须 1）" % n)
    return bad, secn


anch, SECN = anchor_failures(DEV)
assert not anch, "下一片要用的锚点被本轮碰坏：%s" % anch
spec = [os.path.relpath(p, REPO) for p in glob.glob(os.path.join(REPO, "docs/evidence/*/am_s*_hdr.py"))
        if "本片的实物证据＝**改之前**那句" in open(p, encoding="utf-8").read()]
assert len(spec) == 1, "含事故标本句的生成器有 %d 份（须恰好 1）：%s" % (len(spec), spec)
assert not glob.glob(os.path.join(D, "zctl_*")), "扰动副本没被清掉"
assert not re.search(r"@@[A-Za-z0-9_]+@@", DEV_BLOCK), "本轮那段里有占位符残留"
DIRTY_S = len([l for l in sh("git", "-c", "core.quotePath=false", "status", "--porcelain",
                             "--", "src", "tests").splitlines() if l.strip()])
assert DIRTY_S == 0, "生产树 src+tests 有 %d 处改动 ⇒ 本轮归档腿不该碰求解器" % DIRTY_S
# 未入库件数会随本件自己的归档件移动 ⇒ 不入 stdout（见 docstring ④），这里只查回执链在 HEAD 里必须 0 件
OWNCHAIN = len([p for _, p in ROWS
                if re.fullmatch(r"am_s7_verify%s\.(py|log)|_raw_verify%s\.(out|err)|am_s%s_hdr\.py"
                                % (SLICE, SLICE, SLICE), os.path.basename(p))])
assert OWNCHAIN == 0, "本片回执链里有 %d 件已在 HEAD ⇒ 「留给分片 28」那句成了假话" % OWNCHAIN
nv = [l.strip() for l in subprocess.run(["nvidia-smi", "--query-compute-apps=pid",
                                         "--format=csv,noheader"], capture_output=True,
                                        text=True).stdout.splitlines() if l.strip()]
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
    swap("回执三段字节", "%s 字节＝表头 %s＋正文 %s" % (LOGSZ, OFF, int(RAWSZ) + 1)),
    swap("尺子全树份数", "现算 %s 份逐字节副本" % (NCOPIES - 1)),
    swap("SRC 指纹", FINGER[:-1] + "a"),
    swap("shortstat 原文", SS.split(", ")[0]),
    swap("顶格字段与声明互证", "顶格字段 %s 行＝生成器 `FIELDS` 声明 %s" % (NFIELD, NFIELDS + 1)),
    swap("正对照 A", "不符 0/%s" % NB),
]
for name, mutated in PERT:
    got = missing_claims(CLAIMS, mutated)
    assert got, name + "：扰动后仍全绿 ⇒ 该闸门空转"
_secn, _sec = last_section(DEV)
# 红路＝**新增一个子条**再引用同一份回执（往 (m) 内部抄一遍不算扰动：分段按子条字母，同一条内多次引用仍是"唯一"）
bait = _sec + "\n**(n) 诱饵：这一子条也引用一次 am_s7_verify%s.log，且另起一处「当前「不动 GPU」」口径\n" % SLICE
bad2, _ = anchor_failures(DEV.replace(_sec, bait, 1))
assert bad2, "把本轮回执名再抄进第二个子条后锚点闸门没变红 ⇒ 唯一性闸门空转"
assert len(bad2) == 2, "锚点扰动应同时抓坏回执唯一性与钉住口径唯一性，实得 %s" % bad2

print("本轮四段新追加＝开发日志 (m)、%s（起始标记各唯一命中 1 处）" % "、".join(OTHER))
print("开发日志那段：现取声称 %d 条全部命中原样｜HEAD＝%s 父＝%s｜A／M／合计＝%s／%s／%s"
      % (len(CLAIMS), HEAD[:7], PARENT[:7], NA, NM, NTOT))
print("另外三份各查 %s 条读数，缺失 0 条" % "/".join(str(len(v)) for k, (t, v) in OTHER.items()))
print("落地证明现算：tree 条目 %s＝blob %s＋tree %s｜闭合式 %s==%s＋%s｜差集 %s/%s 与 %s/%s｜SHA 不符 %s/%s｜本片 %s/%s MATCH"
      % (ENTRIES, NB, NTREE, NB, NP, NA, DL, NB, DR, NSH, MS, NSH, SLM, NFL))
print("回执 %s B＝表头 %s＋正文 %s｜cmp 三联＝%s｜顶格字段 %s＝FIELDS 声明 %s" % (LOGSZ, OFF, RAWSZ, CMP, NFIELD, NFIELDS))
print("尺子全树副本 %s 份（md5 前 12 %s）｜SRC 指纹 %s｜表头控件扰动 %s 条｜提交说明面基线 %s／坏件 %s"
      % (NCOPIES, ruler[:12], FINGER, NPERT, NBASE, NCTLBAD))
print("下一片锚点：§26.%s 里 am_s7_verify%s.log 唯一被子条 ['m'] 引用｜标本生成器＝%s｜zctl 残留 0" % (SECN, SLICE, spec[0]))
print("入库节奏：本片回执链 5 件（am_s7_verify%s.py／.log、_raw_verify%s.out／.err、am_s%s_hdr.py）"
      "在 HEAD 清单里 %s/5 件 ⇒ 交分片 %s 入库" % (SLICE, SLICE, SLICE, OWNCHAIN, int(SLICE) + 1))
print("内存扰动 %d 条读数闸门＋1 条锚点闸门全部变红" % len(PERT))
print("钉住：我的解释器在卡上留有 0 个 compute 进程（断言，总数不入档）；生产树 src+tests 改动 0 件")
print("ARCHIVE_VERDICT = LANDING READINGS AND NEXT-SLICE ANCHORS PASS")
