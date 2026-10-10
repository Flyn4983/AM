#!/usr/bin/env python3
"""本片落地核验件的**表头生成器**：表头每字段由命令现取，正文区逐字节等于核验件的原始 stdout。

片号、上一片回执名、本轮日志节号**都从命令现取**（HEAD 提交说明／本目录 glob／开发日志正则），
因为上一片 §26.27 (u) 记过一次实物事故：机械克隆只换得动"含旧片号的那一处"，其余注文原样留下
⇒ 上一片的故事被安在本片头上。文件名里的片号与 HEAD 的片号互相印证，不符就拒绝归档。

为什么不用 f-string 承载正文（§26.27 (q) 那一片就是为此记了一条自纠）：f-string 会把字面量里的 `{32}`
当替换字段**静默吃掉**。⇒ 本件一律用 **PLACEHOLDER ＋ str.replace**，值只从命令读数来。

自证四条：① 字段标签只允许出现在**行首**（粘连闸门，沿用 am_s18_hdr.py 的规则）；
② BODY_OFFSET 用**不动点迭代**求（写这一行本身会改变偏移量），不收敛即拒绝；
③ 写盘后实跑 `cmp -i <正文起点>:0`，rc≠0 就抛；
④ 表头不得残留 PLACEHOLDER。
"""
import os
import re
import subprocess
import sys

D = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(D)))
RAW = os.path.join(D, "_raw_verify25.out")
ERR = os.path.join(D, "_raw_verify25.err")
OUT = os.path.join(D, "am_s7_verify25.log")
SCRIPT_REL = "docs/evidence/2026-10-09/am_s7_verify25.py"
FINGER_CMD = ("find src tests -name '*.py' -not -path '*__pycache__*' -print0 "
              "| sort -z | xargs -0 md5sum | md5sum | cut -d' ' -f1")


def sh(*args):
    return subprocess.run(args, cwd=REPO, capture_output=True, text=True, check=True).stdout


body = open(RAW, encoding="utf-8").read()
# 复跑核验件：既取 rc/stderr，又要求 stdout 与归档正文**逐字节相同**（＝仪器可复现的正对照）
ENV = dict(os.environ)
ENV["CUDA_VISIBLE_DEVICES"] = ""
rp = subprocess.run([sys.executable, os.path.join(REPO, SCRIPT_REL)], cwd=REPO, env=ENV,
                    capture_output=True, text=True)
rc_txt = str(rp.returncode)
err_bytes = len(rp.stderr.encode())
REPRO = rp.stdout == body
assert REPRO, "复跑 stdout 与归档正文不同 ⇒ 仪器不可复现，本件的判读行不能代表那次运行"

FIELDS = ["生成时刻", "CMD", "ENV", "脚本", "TREE", "SRC指纹", "GPU", "RUN", "判读",
          "四个正对照", "分母口径", "正文区", "复核", "同片对照", "用途", "教训"]

TMPL = """生成时刻: @@NOW@@
CMD     : 正文那条运行＝CUDA_VISIBLE_DEVICES= /tmp/amvenv/bin/python docs/evidence/2026-10-09/am_s7_verify25.py
          （stdout→_raw_verify25.out，stderr→_raw_verify25.err，两条都在本件里被复跑并逐字节比对；
          这两件在生成本表头**之前**就用 cp -p 逐字节抄进本目录（与本件同一次提交入库），md5＝@@RAWM@@／@@ERRM@@，
          mtime 保留 ⇒ 下面"首次归档运行"那个时刻是原次运行时刻，不是复制时刻）
          本表头由同解释器跑 am_s25_hdr.py 生成，它内部又用同一环境复跑了上面那条命令一次。
ENV     : CUDA_VISIBLE_DEVICES=（空串＝CPU 钉住，用户令「继续，还是不动GPU」＋「现在需要把GPU归还给我的其他程序」）；
          本件**不 import jax**（只跑 git／GitHub API），带空串是习惯性钉住而不是必需，照实说明。
脚本    : @@SCRIPT@@（＝分片 18 那把尺子的**逐字节副本**：与 am_s7_verify18.py 现比 md5＝@@MDEQ@@，
          本件 md5 @@MD5PY@@；本目录 glob＋md5 现算＝同一把尺子已有 @@NRULE@@ 份档案（含本件），
          多片共用同一仪器，不是"为这一片重写一把尺子"。这个数**现算**，因为写死它就会随片腐烂）
TREE    : head=@@HEAD@@ parent=@@PARENT@@ dirty(src+tests)=@@DIRTY@@
SRC指纹 : @@FINGER@@（口径与全量回归件一致 ⇒ 核验时被测树就是本片落地的那棵树）
GPU     : 本表头生成时现取 nvidia-smi compute 进程 @@NVPID@@ 个，其中我的解释器（/proc/PID/cmdline 含 /tmp/amvenv）
          所留 @@NVME@@/@@NVPID@@ 个（须 0）；口径与全量回归件的归还段相同。
RUN     : 表头生成时**复跑**核验件：rc=@@RC@@、stderr @@ERRB@@ 字节（0＝没有一条警告被吞），且复跑 stdout 与
          下面的正文区**逐字节相同**（本件写入前有断言）⇒ 判读行与正文是同一次运行的口径，非两次拼接。
          首次归档运行 @@DATE_START@@（其 stderr 留档案 @@ERRF@@ 字节），复跑于 @@DATE_END@@。
判读    : 远端 main == 本地 HEAD；tree 条目 @@ENTRIES@@ = blob @@NB@@ + tree @@NTREE@@；
          闭合式 远端 @@NB@@ == 父提交 @@NP@@ + 本片新增(A) @@NA@@ ＝ @@EXPECT@@ -> PASS；
          双向路径差集 @@DL@@/@@NB@@ 与 @@DR@@/@@NB@@；全库 blob SHA 不符 @@MS@@/@@NSHARED@@；
          本片单列复核 @@SLICE_MATCH@@/@@NFILES@@ MATCH（A @@NA@@／M @@NM@@ 个文件）。
四个正对照: A 远端一条 sha 首位改 f ⇒ 不符 @@CA@@/@@CBA@@；B 远端删一条路径 ⇒ 本地独有 @@CB1@@/@@CDB1@@、远端独有 @@CB2@@/@@CDB2@@；
          C 闭合式右边 +1 ⇒ @@NB@@ == @@NBP@@ 判 FAIL；D 远端删**本片**一条路径 ⇒ 本片缺失 @@CD1@@/@@NFILES@@、SHA 不符 @@CD2@@/@@NFILES@@。
          四条读数一律从正文 ⑤ 段正则现取（本表头不许自己造数），逐条都能红、所以不是空转。
分母口径: 每个 0 都与其分母同印（@@DL@@/@@NB@@、@@MS@@/@@NSHARED@@、@@SLICE_MATCH@@/@@NFILES@@），
          没有"光秃秃的 0"。
正文区  : 下面 BODY_OFFSET 之后＝核验件原始 stdout，逐字节未改（含 ①〜⑤ 与 LANDING_VERDICT 行）。
复核    : 在仓库根跑 cmp -i @@BOFF@@:0 @@OUTREL@@ @@RAWREL@@   （rc=0 才算正文区逐字节相同；两侧偏移各 +1 必须变红）
同片对照: 本片新增 @@NA@@ 件，与 /tmp **同名件现比 md5**（本表头自己数，不抄任何一份档案里的件数）＝
          相同 @@MDSAME@@ 件／不同 @@MDIF@@ 件（@@MDIFES@@）／/tmp 已无同名件 @@MDAB@@ 件（临时件被清理或当初换了名）。
          @@MDNOTE@@⇒ 这条只作辅助溯源，落地证明以正文区为准。
用途    : S7 分片 @@SLICE@@ 的落地回执（片号由 HEAD 提交说明现取，并与本件文件名 am_s7_verify@@SLICE@@.py
          互证，两边不符即拒绝归档 ⇒ 机械克隆不再能把上一片的故事安在本片头上）。入库清单由
          `git show --name-status HEAD` 现取（本表头自己数、自己列名）＝
          新增 @@NAA@@ 件（@@ADDBAS@@）＋修改 @@NMOD@@ 件（@@MODBAS@@）；逐条 SHA 复核在正文区 ④ 段。
          本轮开发日志 §26.27 (@@SEC@@) 引用的回执是**上一片**的 @@PREVLOG@@（@@PREVSLICE@@ 号）；
          对本件（am_s7_verify@@SLICE@@.log）的日志引用要到下一轮才出现，本表头不预先声称它已被引用。
教训    : ① 仪器复用＝同一把尺子量多片，比对才有效；② 闭合式的右边含本片新增数，所以它随片而变、不是恒等式：
          分片 18 的档案印「@@S18@@」、本片印「远端 @@NB@@ == 父提交 @@NP@@ ＋ 新增 @@NA@@ ＝ @@EXPECT@@」，
          两条各自被控制 C（右边 +1）证过"能红"；但闭合式只数**个数**，看不见"少一个＋多一个"的形状，
          所以必须与双向路径差集、全库 SHA 不符两条**同读**，三条一起才构成落地证明；③ 表头不用 f-string
          承载（§26.27 (q) 的成因），改用 PLACEHOLDER＋replace，写盘后再查残留；④ 本件由上一片的生成器
          **机械克隆**（sed 只换得出文件名里的片号）而来 ⇒ 凡"来历／片号／节号字母／份数"写死在注文里的
          都会随片腐烂，diff 不会自动改它。事故标本**从档案里现取**（不是凭记忆转述）：上一份生成器
          `am_s24_hdr.py` 的"教训"行原文印着「@@SPEC24@@」——同一句里外层片号 @@SPEC24OUT@@、括号里
          @@SPEC24IN@@，两个数不同 ⇒ 那句故事说的是 @@SPEC24IN@@ 号那一片，而写它的生成器是 @@SPEC24OUT@@ 号自己的。
          本片的防法不是手改文字，而是三条**现取＋互证**闸门：(a) 片号从 HEAD 提交说明现取，并与本件文件名
          互证（两边都＝@@SLICE@@）；(b) 日志节号从开发日志正则现取（＝@@SEC@@），"本轮引用哪份回执"由本目录
          glob 现取（＝@@PREVLOG@@）；(c) 入库清单从 `git show --name-status HEAD` 现取，并与正文区 ④ 段的
          A／M 计数互证（@@NAA@@/@@NMOD@@）。教训一句话：**数字一律现取，文字也要；不能现取的文字就只能引用
          一份能被命令查到的档案原文。**
BODY_OFFSET=@@BOFF@@
"""

def md5(path):
    return subprocess.run(["md5sum", path], capture_output=True, text=True).stdout.split()[0]


nv = [l.strip() for l in subprocess.run(["nvidia-smi", "--query-compute-apps=pid",
                                         "--format=csv,noheader"], capture_output=True,
                                        text=True).stdout.splitlines() if l.strip()]
nv_mine = [p for p in nv
           if "/tmp/amvenv" in open(f"/proc/{p}/cmdline", "rb").read().decode("utf-8", "replace")]

# 同一把尺子在本目录已有几份（glob＋md5 现算）：写死"两份/三份"会随片腐烂，所以让它自己数
ruler_md5 = md5(os.path.join(REPO, SCRIPT_REL))
ruler_copies = sorted(f for f in os.listdir(D)
                      if re.fullmatch(r"am_s7_verify\d+\.py", f)
                      and md5(os.path.join(D, f)) == ruler_md5)
assert os.path.basename(SCRIPT_REL) in ruler_copies, "本件的尺子副本没被自己的 glob 数到 ⇒ 计数口径失真"

# 本片新增件里有多少是 /tmp 原件的逐字节副本（现比现取，不抄开发日志的数）
rows = [r.split("\t") for r in sh("git", "-c", "core.quotePath=false", "show",
                                  "--name-status", "--format=", "HEAD").splitlines() if r.strip()]
added = [r[-1] for r in rows if r[0] == "A"]
mods = [r[-1] for r in rows if r[0] == "M"]
assert len(added) + len(mods) == len(rows), "入库清单里有 A/M 之外的状态字母 ⇒ 上面的取列口径破了"
md_same, md_diff, md_absent = [], [], []
for p in added:
    t = os.path.join("/tmp", os.path.basename(p))
    if not os.path.exists(t):
        md_absent.append(p)
    elif md5(os.path.join(REPO, p)) == md5(t):
        md_same.append(p)
    else:
        md_diff.append(p)
assert len(md_same) + len(md_diff) + len(md_absent) == len(added), "三分类没覆盖全部新增件 ⇒ 有一条既不算同也不算异也不算缺"
# "不同"的那件**不写死来历**：上一片把它写成一句具体故事，这一片的差集若换了名字，那句故事就成了假话
# ⇒ 读数（件名、件数）一律现取，注文只说"两版逐字节不同"这一件能被 md5 证明的事实。
if md_diff:
    MDNOTE = ("其中逐字节**不同**的是 " + "、".join(os.path.basename(p) for p in md_diff)
              + "：/tmp 那份与档案那份是两个版本（差异由 md5 现比判出，来历不在本表头里编）。")
else:
    MDNOTE = "没有一件不同 ⇒ /tmp 现存的同名件全部逐字节相同。"

# 入库清单两个出处互证：正文区 ④ 段（核验件从远端树＋git 现算）与本表头（本地 git show）
NA_BODY = int(re.search(r"新增 (\d+) \+ 修改", body).group(1))
NM_BODY = int(re.search(r"新增 \d+ \+ 修改 (\d+)", body).group(1))
assert (NA_BODY, NM_BODY) == (len(added), len(mods)), (
    "核验件印出的入库数 %s/%s ≠ 本地 git show 实得 %s/%s ⇒ 有一侧的清单口径变了，拒绝归档"
    % (NA_BODY, NM_BODY, len(added), len(mods)))

# ---------- 片号／节号／上一片回执／事故标本：三条闸门都靠现取，防机械克隆张冠李戴 ----------
SUBJ = sh("git", "log", "-1", "--format=%s").strip()
SLICE = re.search(r"S7 分片 (\d+)", SUBJ).group(1)
assert os.path.basename(SCRIPT_REL) == "am_s7_verify%s.py" % SLICE, \
    "HEAD 提交说明的片号 %s ≠ 本件所验尺子 %s ⇒ 上一片的故事安在本片头上，拒绝归档" % (
        SLICE, SCRIPT_REL)
DEVLOG = os.path.join(REPO, "docs/开发日志.md")
SECS = re.findall(r"### 26\.27 \((\w)\)", open(DEVLOG, encoding="utf-8").read())
assert SECS, "开发日志里取不到 26.27 的节号 ⇒ 那条引用是凭记忆的"
SEC = SECS[-1]
RCPTS = sorted(int(re.fullmatch(r"am_s7_verify(\d+)\.log", f).group(1))
               for f in os.listdir(D) if re.fullmatch(r"am_s7_verify(\d+)\.log", f))
assert len(RCPTS) == len(set(RCPTS)), "回执档案里有重号角本"
PREVSLICE = max(n for n in RCPTS if n != int(SLICE))
PREVLOG = "am_s7_verify%d.log" % PREVSLICE
assert PREVLOG in os.listdir(D), "取到的上一片回执 %s 不在本目录" % PREVLOG
assert int(SLICE) == PREVSLICE + 1, \
    "本片 %s 与上一片回执 %s 不连号 ⇒ 有一片没落，或片号口径变了" % (SLICE, PREVSLICE)
# 事故标本**从档案里现取**（不凭记忆转述）：上一份生成器的"教训"行原文
HDR24 = open(os.path.join(D, "am_s24_hdr.py"), encoding="utf-8").read()
_cap = re.search(r"本片的实物证据＝\*\*改之前\*\*那句「(.+?)」", HDR24)
assert _cap, "上一份生成器里取不到那句标本原文 ⇒ 这条引用没凭据，拒绝"
SPEC24 = _cap.group(1)
_g24 = re.search(r"分片 (\d+)（分片 \*\*(\d+)\*\*", SPEC24)
assert _g24, "标本那句里没有「外层片号＋括号里旧片号」两个数 ⇒ 标本不是那次事故：" + SPEC24
SPEC24OUT, SPEC24IN = _g24.group(1), _g24.group(2)
assert SPEC24OUT != SPEC24IN, "标本那句两个数相同 ⇒ 事故不存在，这条引用是空转"

vals = {
    "SLICE": SLICE, "SEC": SEC, "PREVLOG": PREVLOG, "PREVSLICE": str(PREVSLICE),
    "SPEC24": SPEC24, "SPEC24OUT": SPEC24OUT, "SPEC24IN": SPEC24IN,
    "NOW": sh("date", "+%F %T %z").strip(),
    "SCRIPT": SCRIPT_REL,
    "MD5PY": md5(os.path.join(REPO, SCRIPT_REL))[:12],
    "NRULE": str(len(ruler_copies)),
    "MDEQ": ("相同" if md5(os.path.join(REPO, SCRIPT_REL)) == md5(os.path.join(D, "am_s7_verify18.py"))
             else "不同 ⇒ 尺子被改过"),
    "HEAD": sh("git", "rev-parse", "HEAD").strip(),
    "PARENT": sh("git", "rev-parse", "HEAD^").strip(),
    "DIRTY": str(len([l for l in sh("git", "-c", "core.quotePath=false", "status", "--porcelain",
                                    "--", "src", "tests").splitlines() if l.strip()])),
    "FINGER": subprocess.run(["bash", "-c", FINGER_CMD], cwd=REPO, capture_output=True,
                             text=True).stdout.strip(),
    "NVPID": str(len(nv)),
    "NVME": str(len(nv_mine)),
    "RC": rc_txt,
    "ERRB": str(err_bytes),
    "ERRF": str(os.path.getsize(ERR)),
    "DATE_START": sh("date", "-r", RAW, "+%F %T %z").strip(),
    "DATE_END": sh("date", "+%F %T %z").strip(),
    "ENTRIES": re.search(r"tree 条目 (\d+) = ", body).group(1),
    "NB": re.search(r"blob (\d+) \+ tree", body).group(1),
    "NTREE": re.search(r"blob \d+ \+ tree (\d+)", body).group(1),
    "NP": re.search(r"父提交 = (\d+)", body).group(1),
    "NA": re.search(r"新增 (\d+) \+ 修改", body).group(1),
    "NM": re.search(r"新增 \d+ \+ 修改 (\d+)", body).group(1),
    "EXPECT": re.search(r"\+ 新增 (\d+) = (\d+)", body).group(2),
    "NBP": re.search(r"闭合式右边 \+1）⇒ (\d+) == (\d+) 判", body).group(2),
    "CA": re.search(r"正对照 A（远端 \S+ 的 sha 首位改 f）⇒ 不符 (\d+)/(\d+)", body).group(1),
    "CBA": re.search(r"正对照 A（远端 \S+ 的 sha 首位改 f）⇒ 不符 (\d+)/(\d+)", body).group(2),
    "CB1": re.search(r"正对照 B（远端删一条路径）⇒ 本地独有 (\d+)/(\d+)，远端独有 (\d+)/(\d+)", body).group(1),
    "CDB1": re.search(r"正对照 B（远端删一条路径）⇒ 本地独有 (\d+)/(\d+)，远端独有 (\d+)/(\d+)", body).group(2),
    "CB2": re.search(r"正对照 B（远端删一条路径）⇒ 本地独有 (\d+)/(\d+)，远端独有 (\d+)/(\d+)", body).group(3),
    "CDB2": re.search(r"正对照 B（远端删一条路径）⇒ 本地独有 (\d+)/(\d+)，远端独有 (\d+)/(\d+)", body).group(4),
    "CD1": re.search(r"正对照 D（远端删本片一条路径）⇒ 本片缺失 (\d+)/(\d+)，SHA 不符 (\d+)/", body).group(1),
    "CD2": re.search(r"正对照 D（远端删本片一条路径）⇒ 本片缺失 (\d+)/(\d+)，SHA 不符 (\d+)/", body).group(3),
    "DL": re.search(r"本地独有 (\d+)/", body).group(1),
    "DR": re.search(r"远端独有 (\d+)/", body).group(1),
    "MS": re.search(r"blob SHA 不符 (\d+)/", body).group(1),
    "NSHARED": re.search(r"blob SHA 不符 \d+/(\d+)", body).group(1),
    "SLICE_MATCH": re.search(r"MATCH (\d+)/", body).group(1),
    "NFILES": re.search(r"本片文件 (\d+) 个", body).group(1),
    "MDSAME": str(len(md_same)),
    "MDIF": str(len(md_diff)),
    "MDIFES": "、".join(os.path.basename(p) for p in md_diff) or "无",
    "MDAB": str(len(md_absent)),
    "S18": re.search(r"② 闭合式 (.+?) -> PASS",
                     open(os.path.join(D, "am_s7_verify18.log"), encoding="utf-8").read()).group(1),
    "OUTREL": "docs/evidence/2026-10-09/am_s7_verify25.log",
    "RAWREL": "docs/evidence/2026-10-09/_raw_verify25.out",
    "RAWM": md5(RAW)[:12],
    "ERRM": md5(ERR)[:12],
    "MDNOTE": MDNOTE,
    "NAA": str(len(added)),
    "NMOD": str(len(mods)),
    "ADDBAS": "、".join(os.path.basename(p) for p in added),
    "MODBAS": "、".join(os.path.basename(p) for p in mods),
    "BOFF": "0",
}
# 原始件的来历：/tmp 里那份若还在，必须与归档件逐字节相同（cp -p 的凭据），否则 CMD 行的"逐字节归档"是假话
for _n in ("_raw_verify25.out", "_raw_verify25.err"):
    _t = os.path.join("/tmp", _n)
    if os.path.exists(_t):
        assert md5(_t) == md5(os.path.join(D, _n)), "/tmp 与归档的 " + _n + " 不同 ⇒ 归档件不是那次运行的原始 stdout"
assert "LANDING_VERDICT = ALL CHECKS PASS" in body, "正文里没有全过的判读行 ⇒ 本件不该被当作回执归档"
assert rc_txt == "0" and err_bytes == 0, "复跑核验件 rc/stderr 不齐 ⇒ 与正文不同一次运行，拒绝归档"
assert nv_mine == [], f"我的解释器在 GPU 上留有进程 {nv_mine} ⇒ 与「不动 GPU」冲突，拒绝归档"
missing = sorted(set(re.findall(r"@@([A-Za-z0-9_]+)@@", TMPL)) - set(vals))
assert not missing, "模板有未取值的字段：" + str(missing)

def render():
    h = TMPL
    for k, v in vals.items():
        h = h.replace("@@" + k + "@@", v)
    return h

# BODY_OFFSET 必须等于"表头字节数"，而表头含这一行自身 ⇒ 迭代到不动点
for _ in range(12):
    cand = len(render().encode())
    if cand == int(vals["BOFF"]):
        break
    vals["BOFF"] = str(cand)
else:
    raise SystemExit("BODY_OFFSET 未收敛 ⇒ 拒绝归档")
hdr = render()
assert "@@" not in hdr, "表头残留 PLACEHOLDER"


def gate(h):
    """字段标签必须**顶行首且紧跟冒号**；缩进行只能是续行；顶格非字段行只允许 BODY_OFFSET。

    口径收窄成「行首标签＋冒号」而不是「行内出现标签」：正文里有「本片单列复核 27/27」这类词组，
    用"包含"判会把自己的正确表头判红（写本件时实测到，故在此注明口径）。
    """
    lines = [l for l in h.splitlines() if l.strip()]
    lab = "|".join(map(re.escape, FIELDS))
    defs = [l for l in lines if re.match(r"^(%s)\s*:" % lab, l)]
    mis = [l for l in lines if re.match(r"^\s+(?:%s)\s*:" % lab, l)]
    loose = [l for l in lines if not l[0].isspace() and l not in defs
             and not l.startswith("BODY_OFFSET=")]
    assert not mis, "缩进处出现「标签＋冒号」（漏了续行缩进？）：" + str(mis[:2])
    assert not loose, "顶格却不像字段行：" + str(loose[:2])
    assert len(defs) == len(FIELDS), "顶格字段行数 %d ≠ FIELDS 声明 %d ⇒ 有字段漏写或改名" % (
        len(defs), len(FIELDS))
    return len(defs)


assert sys.executable.startswith("/tmp/amvenv"), f"解释器不是 /tmp/amvenv 而是 {sys.executable} ⇒ CMD 行描述失真"
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "生成器自身未带空串 CVD ⇒ 与 ENV 行的描述不符"
gate(hdr)

if "--write" not in sys.argv:
    print(hdr)
    print("（预检模式：未写盘。加 --write。）")
    print("md5 现比 /tmp 同名件＝同", len(md_same), "／异", len(md_diff), "／缺", len(md_absent),
          "，共", len(added), "｜GPU 我的进程＝", len(nv_mine), "/", len(nv))
    raise SystemExit(0)

open(OUT, "w", encoding="utf-8").write(hdr + body)
back = open(OUT, "rb").read()
off = int(re.search(rb"BODY_OFFSET=(\d+)", back).group(1))
assert back[off:] == body.encode(), "正文区逐字节复核失败"
head_txt = back[:off].decode()
assert "@@" not in head_txt, "表头残留 PLACEHOLDER"
assert "＠" not in head_txt, "表头含全角 at 号 ⇒ 那种占位符会静默不替换"
n_defs = gate(head_txt)
assert int(vals["BOFF"]) == off == len(head_txt.encode()), "BODY_OFFSET ≠ 实际表头字节数"
c1 = subprocess.run(["cmp", "-i", str(off) + ":0", OUT, RAW], capture_output=True, text=True)
c2 = subprocess.run(["cmp", "-i", str(off + 1) + ":0", OUT, RAW], capture_output=True, text=True)
c3 = subprocess.run(["cmp", "-i", str(off) + ":1", OUT, RAW], capture_output=True, text=True)
print("写盘完成：", OUT, len(back), "bytes")
print("cmp 复核 rc（同偏移须 0）＝", c1.returncode, "｜左 +1 须非 0＝", c2.returncode,
      "｜右 +1 须非 0＝", c3.returncode, "｜顶格字段行数＝", n_defs, "/", len(FIELDS))
print("md5 现比 /tmp 同名件＝同", len(md_same), "／异", len(md_diff), "／缺", len(md_absent),
      "，共", len(added), "｜GPU 我的进程＝", len(nv_mine), "/", len(nv),
      "｜复跑 stdout 与正文逐字节相同＝", REPRO)
assert c1.returncode == 0 and c2.returncode != 0 and c3.returncode != 0
