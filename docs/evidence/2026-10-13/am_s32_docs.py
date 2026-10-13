#!/usr/bin/env python3
r"""分片 32 的**填值器**（＋带守卫的一次性写盘腿）：五张档案面的每个读数都由命令现取，具名替换后渲染。

本轮的起点与往轮不同，所以本件的分类层是重写过的（不是抄上一版的字面）：
① **同日轮**——上一片的回执 `am_s7_verify31.log` 与本件在**同一个** `docs/evidence/2026-10-13/`，
   上一版那两条「上一片回执与本件不同目录」的前提（`assert EVID_OLD != EVID_NEW`）本轮**为假**。
   链归属因此只能由**文件名里的片号**切分，不能由目录切分；本件把这一点做成可红的对照（`chain_split_control`）。
② **上一片的链此刻是「已暂存新增」而不是「未跟踪」**——本轮按流程前提每件落盘即 `git add`，
   所以只扫 `^??` 的旧分组尺子对整条链会数出**每组 0 件**（＝假零，不是"没有件"）；本件按三去向分类
   （未跟踪／已暂存新增／已跟踪改动）并让分组扫「未跟踪＋已暂存新增」的并集，配一条旧口径的对照
   （`staged_census_control`：旧尺子对每组现数 0，新尺子逐组非 0）。
③ **具名替换＋邻接断言**、**渲染结构门**（按行配平全角 `（）`、剥内联码后配平 `**`、目标末字节是换行、
   锚点整份文件恰好 1 处、渲染结果不残留 `%(`／`%（`／`@@`）、**节内唯一性按顶级子条数**（每条配诱饵）
   ——这四条是往轮用真事故换来的，本轮逐条继承，不做修改。

volatile 读数（待入库件数、卡上进程数、落笔 date）只进本轮散文，不写进任何可复跑件的 stdout。
默认只渲染到 `/tmp/out32/`；`--write` 才落盘，且落盘前后各数一次闭合式。
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
LESSON = "普查仪器没有第二出处"       # 从分片 31 承接进来的那条口径（＝表头生成器要现取的锚句）
LESSON_NEW = "同日轮没有目录级证据"    # 本轮新学的锚句（下一片的自查件按这个短语回搜）
INTERP = "/tmp/amvenv/bin/python"
SELF_NAME = os.path.basename(os.path.realpath(__file__))
OUT_DIR = "/tmp/out32"
TODAY = subprocess.run(["date", "+%F"], capture_output=True, text=True).stdout.strip()
EVID_NEW = "docs/evidence/" + TODAY
DAILY = os.path.join(REPO, ".workbuddy/memory/%s.md" % TODAY)
# 写盘守卫：这些名字**正在跑**就不许动档案（按 /proc/PID/cmdline 的 basename **整词**比对，不按命令行子串猜）
TOKENS = {"am_s7_verify31.py", "am_s31_hdr.py", "am_s31_msg.py", "am_s31_archive_check.py",
          "am_s31_hdr_controls.py", "am_s31_msg_controls.py", "am_s31_hdr_after.py",
          "am_s31_hdr_after_controls.py", "am_s31_docs_landing.py",
          "am_s7_verify32.py", "am_s32_hdr.py", "am_s32_docs.py", "am_s32_msg.py",
          "am_s32_msg_controls.py", "am_s32_hdr_controls.py", "am_s32_hdr_after.py",
          "am_s32_hdr_after_controls.py", "am_s32_docs_landing.py", "am_s32_archive_check.py"}


def sh(*args):
    """**默认环境**跑（不钉 LC_ALL）——指纹口径只在默认 locale 下与前几片可比。"""
    return subprocess.run(args, cwd=REPO, capture_output=True, text=True).stdout


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def one(needle, text, what):
    hits = re.findall(needle, text, re.M)
    assert len(hits) == 1, "%s 命中 %d 处（须恰好 1）：%s" % (what, len(hits), hits[:3])
    return hits[0]


def line_of(needle, text, what):
    """把一条汇总读数**锁到整行**再取字段：同一份档里那个词可能在别处被复述（按出现次数数就会串面）。"""
    hits = re.findall(needle, text, re.M)
    assert len(hits) == 1, "%s 命中 %d 行（须恰好 1）：%s" % (what, len(hits), [h[:60] for h in hits[:3]])
    return hits[0]


def field(line, name, what):
    m = re.search(r"%s＝(\d+) 条" % re.escape(name), line)
    assert m, "%s：锁到的那行里取不到「%s＝N 条」⇒ 该面的汇总行形制变了" % (what, name)
    return int(m.group(1))


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
print("== 现取：HEAD＝%s（父 %s｜提交时刻 %s）｜片号＝%d（由 HEAD 主题现取）" % (HEAD7, PARENT7, CDATE, SLICE))

# ---- 上一片回执：全树 glob 命中必须恰好 1；同日轮里"唯一"来自片号而不是目录 ----
PREVLOG = "am_s7_verify%d.log" % PREVSLICE


def receipt_glob(key):
    pat = "am_s7_verify%s.log" % key
    return sorted(os.path.relpath(p, REPO) for p in glob.glob(os.path.join(REPO, "docs/evidence/*/" + pat)))


_curl = receipt_glob(str(PREVSLICE))
assert len(_curl) == 1, "上一片回执在全树命中 %d 份（须 1）：%s ⇒ 引用哪一份都不唯一" % (
    len(_curl), _curl)
CURL = _curl[0]
EVID_OLD = os.path.dirname(CURL)
SAME_DAY = EVID_OLD == EVID_NEW
print("   上一片回执＝%s（目录 %s 由 glob 现取）｜同日轮＝%s" % (PREVLOG, EVID_OLD, SAME_DAY))


def chain_split_control():
    """同日轮的正对照：**去掉片号**的那把 glob 必须变成多命中 ⇒ 证明"片号才是唯一的钥匙"。

    上一版靠「上一片回执在另一个日期目录」这一环境事实保唯一；本轮该事实为假，所以唯一性必须由**命名**承担，
    而一条从不检查歧义的命名规则是看不见的——这条对照就是在它身上扎一针。
    """
    wide = sorted(os.path.relpath(p, REPO) for p in
                  glob.glob(os.path.join(REPO, "docs/evidence/*/am_s7_verify*.log")))
    assert len(wide) >= 2, "全树回执共 %d 份（须 ≥2）⇒ 那条「不带片号就不唯一」的说法本轮没有靶" % len(wide)
    assert len(wide) > len(_curl), "不带片号与带片号命中同样多（%d）⇒ 片号那把钥匙本轮并不必要，要改写" % len(wide)
    return "%d 份（带片号的同一条 glob＝%d 份）" % (len(wide), len(_curl))


SPLITCTRL = chain_split_control()
# 上一版仪器里那条把「上一片回执与本件不同目录」当前提的闸门，本轮实测**为假**（本件的这一行就是它的现取读数）
OLDGATE = (EVID_OLD != EVID_NEW)
assert SAME_DAY and OLDGATE is False, "本轮不是同日轮（旧闸门＝%s）⇒ 本件的分类层要按目录切链，那套口径是另一件事" % OLDGATE

# ---- 待入库面：三去向分类（本轮重写的那一层） ----
st = [l.rstrip("\n") for l in sh("git", "-c", "core.quotePath=false", "status", "--porcelain",
                                 "--untracked-files=all").splitlines() if l.strip()]
_class = {}
for l in st:
    x, y, p = l[0], l[1], l[3:]
    k = "UNT" if (x, y) == ("?", "?") else ("NEW" if x == "A" else ("MOD" if "M" in (x, y) else "OTHER"))
    assert k != "OTHER", "清单里出现不认识的形状：%r ⇒ 分类层要按实测重写，先别落笔" % l
    _class.setdefault(p, []).append(k)
dup = [p for p, v in _class.items() if len(v) > 1]
assert not dup, "路径重复出现：%s ⇒ 清单行数与去重数不等" % dup[:3]
UNT = sorted(p for p, v in _class.items() if v == ["UNT"])
NEW = sorted(p for p, v in _class.items() if v == ["NEW"])
MOD = sorted(p for p, v in _class.items() if v == ["MOD"])
folded = [p for p in UNT + NEW + MOD if p.endswith("/")]
assert not folded, "清单里有折叠的目录行 %s ⇒ 计数口径不是 `-uall`，件数与分组都不可信" % folded
NU, NA, NM = len(UNT), len(NEW), len(MOD)
PENDING = UNT + NEW           # 分组尺子扫这一条并集
assert NU == 0, "还有未跟踪件 %s ⇒ 有仪器落盘后没先 `git add`，与本轮登记的流程前提不符" % [
    os.path.basename(p) for p in UNT][:3]
# 49 ＝ 已暂存新增 44（上一片落地半腿链 38＋本片归档写入件与信息生成器 2＋本件的预检／写盘两面 4）＋已跟踪档案面 5
# 这六件**必须先存在再跑**（先落盘、先 `git add`）：否则本件自己的 stdout 面会把计数推着走＝上一轮 attempt1 那个自指形状
assert NA + NM == 49, "落笔现数＝已暂存新增 %d＋修改 %d＝%d 处（本节的闭合式写的是 49）⇒ 先确认工作树有没有被动过、本件的面是否已先落盘" % (
    NA, NM, NA + NM)

ROLES = [
    ("回执链", r"^am_s7_verify%s\.(py|log)$|^_raw_verify%s\.(out|err)$|^am_s%s_hdr\.py$"
     % (PREVSLICE, PREVSLICE, PREVSLICE)),
    ("表头预检与写盘两面（含控件面判绿后的复跑预检）",
     r"^_raw_s%s_hdr_(precheck|write|precheck_rerun_after_controls_green)\.(out|err)$" % PREVSLICE),
    ("表头控件面", r"^am_s%s_hdr_controls\.py$|^_raw_s%s_hdr_controls\.(out|err)$"
     % (PREVSLICE, PREVSLICE)),
    ("表头控件面的红灯现场与 403 探针",
     r"^_raw_s%s_hdr_controls\.(attempt[0-9]+-[a-z0-9-]+|probe-ruler-403)\.(out|err)$" % PREVSLICE),
    ("只读仪器与其真跑两面", r"^am_s%s_hdr_after\.py$|^_raw_s%s_hdr_after\.(out|err)$"
     % (PREVSLICE, PREVSLICE)),
    ("只读仪器的控件面", r"^am_s%s_hdr_after_controls\.py$|^_raw_s%s_hdr_after_controls\.(out|err)$"
     % (PREVSLICE, PREVSLICE)),
    ("只读仪器控件面的红灯现场与等长探针",
     r"^_raw_s%s_hdr_after_controls\.(attempt[0-9]+-[a-z0-9-]+|probe-headlen[a-z0-9-]*)\.(out|err)$"
     % PREVSLICE),
    ("落地腿（本件与其预检／写盘两面）",
     r"^am_s%s_docs_landing\.py$|^_raw_s%s_landing_(precheck|write)\.(out|err)$" % (PREVSLICE, PREVSLICE)),
    ("归档自查腿", r"^am_s%s_archive_check\.py$|^_raw_s%s_archive_check(\.attempt[0-9]+-[a-z0-9-]+)?\.(out|err)$"
     % (PREVSLICE, PREVSLICE)),
    ("本片归档写入件与信息生成器", r"^am_s%d_docs\.py$|^am_s%d_msg\.py$" % (SLICE, SLICE)),
    # 本件自己的两面（含将来可能出现的红灯现场）也是待入库件；它们先落盘再跑，两次运行的计数才同值
    ("本件自己的预检与写盘两面（含红灯现场）",
     r"^_raw_s%d_docs_(precheck|write)(\.attempt[0-9]+-[a-z0-9-]+)?\.(out|err)$" % SLICE),
]
WANT = {"回执链": 5, "表头预检与写盘两面（含控件面判绿后的复跑预检）": 6, "表头控件面": 3,
        "表头控件面的红灯现场与 403 探针": 6, "只读仪器与其真跑两面": 3, "只读仪器的控件面": 3,
        "只读仪器控件面的红灯现场与等长探针": 4, "落地腿（本件与其预检／写盘两面）": 5,
        "归档自查腿": 3, "本片归档写入件与信息生成器": 2,
        "本件自己的预检与写盘两面（含红灯现场）": 4}
buckets, assigned = [], []
for name, pat in ROLES:
    hit = [p for p in PENDING if re.match(pat, os.path.basename(p))]
    assert hit, "%s 一组现数为 0 ⇒ 命名规则与本目录脱节（分组句就成了假话）" % name
    assert WANT[name] == len(hit), "%s：登记 %d 件／现数 %d 件 ⇒ 分组句与磁盘不符，拒绝写盘" % (
        name, WANT[name], len(hit))
    buckets.append((name, len(hit)))
    assigned += hit
assert len(assigned) == len(set(assigned)) == len(PENDING), (
    "角色分组没恰好覆盖待入库并集（分组 %d／去重 %d／现数 %d）⇒ 有件落进 0 组或 ≥2 组" % (
        len(assigned), len(set(assigned)), len(PENDING)))
BKT_TXT = "／".join("%s **%d**" % (n, c) for n, c in buckets)
BKT_SUM = "＋".join(str(c) for _, c in buckets) + "＝%d" % len(PENDING)
MOD_TXT = "／".join(os.path.basename(p) for p in MOD)
assert NM == 5, "已跟踪档案面现数 %d（须 5）：%s" % (NM, MOD_TXT)


def staged_census_control():
    """旧口径（只扫 `^??`）在本轮树上对**每一组**都现数 0 ⇒ 那不是"没有件"，是分类层把链看成空气。

    这一族（假零）本轮第二次落在我自己的仪器上：上一轮的形状是折叠目录行，这一轮的形状是已暂存新增。
    """
    zeros = [n for n, pat in ROLES if not [p for p in UNT if re.match(pat, os.path.basename(p))]]
    assert len(zeros) == len(ROLES), "旧尺子在 %d／%d 组上数出了非零 ⇒ 「每组都是假零」这句本轮不成立" % (
        len(zeros), len(ROLES))
    # 反向也要能红：把「回执链」的桶形改窄到认不出表头生成器，它必须**恰好少认领 1 件**
    name, c0 = buckets[0]
    pat = ROLES[0][1]
    bad = pat.replace(r"^am_s%d_hdr\.py$" % PREVSLICE, r"^am_s%d_hdr_NONEXIST\.py$" % PREVSLICE)
    assert bad != pat, "桶形替换没改变字符串 ⇒ 这根针根本没扎进去"
    hit = [p for p in PENDING if re.match(bad, os.path.basename(p))]
    assert len(hit) == c0 - 1, "改窄之后那组认领 %d 件（原形制 %d，须恰好少 1）" % (len(hit), c0)
    return "%d／%d 组在旧口径（只扫 `^??`）下现数 0；「%s」组桶形改窄后由 %d 件降到 %d 件" % (
        len(zeros), len(ROLES), name, c0, len(hit))


CENSUSCTRL = staged_census_control()
print("   待入库＝未跟踪 %d＋已暂存新增 %d＋修改 %d＝%d 处｜src+tests 改动 %d｜分组（和＝%s）：%s"
      % (NU, NA, NM, NA + NM, NCODE, BKT_SUM, BKT_TXT))
print("   两条对照：%s -> PASS" % CENSUSCTRL)

# ---- 上一片回执的读数（只从档案正则现取，并逐字节复验正文区） ----
RPREV = os.path.join(REPO, CURL)
r31 = open(RPREV, encoding="utf-8").read()
RCPTB, RCPTL = os.path.getsize(RPREV), len(r31.splitlines())
OFFR = int(one(r"^BODY_OFFSET=(\d+)$", r31, "上一片回执的 BODY_OFFSET"))
head_part, body_part = r31.split("BODY_OFFSET=%d\n" % OFFR, 1)
assert len((head_part + "BODY_OFFSET=%d\n" % OFFR).encode()) == OFFR, "BODY_OFFSET 与实际表头字节数不等"
VERDR = one(r"^(LANDING_VERDICT = .*)$", r31, "上一片回执的判读行")
CLOSURER = one(r"闭合式 远端 \d+ == 父提交 \d+ \+ 本片新增\(A\) \d+ ＝ \d+ -> PASS", r31, "上一片闭合式那句")
assert one(r"^TREE\s+: head=([0-9a-f]{40})", r31, "上一片 TREE 行哈希") == HEAD, \
    "上一片回执的 head ≠ 本地 HEAD ⇒ 那张贴的不是当前这棵树的落地证明"
FINGER_REC = one(r"^SRC指纹 : ([0-9a-f]{32})", r31, "上一片回执表头的 SRC指纹")
RAWR = os.path.join(REPO, EVID_OLD, "_raw_verify%d.out" % PREVSLICE)
assert open(RAWR, encoding="utf-8").read() == body_part, "回执正文区与原始 stdout 不逐字节相同"
# ⚠ 这个量**不是**「顶格字段数」：它是表头区的非空行数（含缩进续行）；两量之差在往轮被填错过槽位
HDRLINESR = len([l for l in head_part.splitlines() if l.strip()])
BODYR = str(os.path.getsize(RAWR))
PINREADER = line_of(r"^ENV\s+:.*$", r31, "回执表头里那句「钉住口径从档案现取」")
PREVSEC_R = int(one(r"开发日志 §26\.(\d+)", PINREADER, "回执表头 ENV 行里的节号"))
print("   上一片回执＝%s（%d B／%d 行＝表头 %d＋正文 %d，BODY_OFFSET 已复算）｜判读行「%s」｜正文区与原始件逐字节相同"
      % (PREVLOG, RCPTB, RCPTL, OFFR, int(BODYR), VERDR))

# ---- 节号：开发日志现取最后一个 ### 26.NN，并与回执表头那枚同值（两条出处） ----
devtxt = open(DEVLOG, encoding="utf-8").read()
_secs = list(re.finditer(r"^### 26\.(\d+)[^\n]*$", devtxt, re.M))
PREVSEC = int(_secs[-1].group(1))
assert int(one(r"分片 (\d+)＝提交腿", _secs[-1].group(0), "最后一节标题里的片号")) == PREVSLICE, \
    "§26.%d 的标题写的片号 ≠ 上一片 %d ⇒ 节号与片号的对应变了，本节要先重推" % (PREVSEC, PREVSLICE)
# 本轮的**承接新学**在这条上就现场兑现一次：HEAD 主题里没有「开发日志 §」那枚针（上一族仪器的唯一出处）
SUBJ_SEC = re.findall(r"开发日志 §26\.(\d+)", SUBJ)
assert not SUBJ_SEC, ("HEAD 主题里现在能取到节号（%s）⇒ 那条「主题针不可靠」的承接口径要改写："
                      "它本轮又有了第二出处") % SUBJ_SEC
assert PREVSEC == PREVSEC_R, "开发日志现取的最后一节 §26.%d ≠ 上一片回执表头 ENV 行的 §26.%d ⇒ 两出处不同值" % (
    PREVSEC, PREVSEC_R)
print("   节号两条出处闭合：开发日志最后一节＝§26.%d＝上一片回执 ENV 行那枚；HEAD 主题里该针命中 0 处（承接那条的本轮实证）"
      % PREVSEC)
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
_other = [i for i, (n, _b) in enumerate(_bodies) if "26.%s" % n != G9SEC][0]
_pert = list(_bodies)
_pert[_other] = (_pert[_other][0], _pert[_other][1] + "\n诱饵：那里给出唯一可评分的 N=1 的读数。")
_n = len([k for k, b in _pert if "唯一可评分的 N=1" in b])
assert _n == 2, "节定位器在写坏的样本上数出 %d 节（须 2）⇒ 它没有分辨力" % _n

OLD = lambda *a: os.path.join(REPO, EVID_OLD, *a)

# ---- 上一片三面干跑控件面的档案读数（汇总行先锁到整行，再从该行取字段） ----
MSGCTL = OLD("_raw_s%d_msg_controls.out" % PREVSLICE)
msg = open(MSGCTL, encoding="utf-8").read()
MSGCTLB, MSGERRB, MSGCTLM = os.path.getsize(MSGCTL), os.path.getsize(
    OLD("_raw_s%d_msg_controls.err" % PREVSLICE)), md5(MSGCTL)[:12]
msgsum = line_of(r"^CASES 声明＝\d+ 条（全部红路.*$", msg, "提交说明控件面汇总行")
MSGNCASE = int(one(r"^CASES 声明＝(\d+) 条", msgsum, "提交说明控件面条数"))
MSGNGREEN = field(msgsum, "保持绿", "提交说明控件面")
MSGNSELF = field(msgsum, "本件自己的锚点普查", "提交说明控件面")
MSGNTO = field(msgsum, "超时（环境故障，不判红也不判过）", "提交说明控件面")
MSGNUNC = int(one(r"^本轮造不出红路的闸门＝(\d+) 条", msg, "提交说明控件面 UNCOVERED 数"))
MSGVERD = one(r"^(CONTROL_VERDICT = .*)$", msg, "提交说明控件面判读行")

HDRCTL = OLD("_raw_s%d_hdr_controls.out" % PREVSLICE)
hdr = open(HDRCTL, encoding="utf-8").read()
HDRCTLB, HDRERRB, HDRCTLM = os.path.getsize(HDRCTL), os.path.getsize(
    OLD("_raw_s%d_hdr_controls.err" % PREVSLICE)), md5(HDRCTL)[:12]
hdrsum = line_of(r"^红路声明＝\d+ 条｜保持绿.*$", hdr, "表头控件面汇总行")
HDRNRED = int(one(r"^红路声明＝(\d+) 条", hdrsum, "表头控件面红路数"))
HDRNGREEN = field(hdrsum, "保持绿", "表头控件面")
HDRNENV = field(hdrsum, "环境故障（不判红也不判过）", "表头控件面")
HDRNSELF = field(hdrsum, "本面自查", "表头控件面")
HDRNUNC = int(one(r"^本轮造不出红路的闸门＝(\d+) 条", hdr, "表头控件面 UNCOVERED 数"))
HDRVERD = one(r"^(CONTROL_VERDICT = .*)$", hdr, "表头控件面判读行")

AFCTL = OLD("_raw_s%d_hdr_after_controls.out" % PREVSLICE)
afctl = open(AFCTL, encoding="utf-8").read()
AFCTLB, AFERRB, AFCTLM = os.path.getsize(AFCTL), os.path.getsize(
    OLD("_raw_s%d_hdr_after_controls.err" % PREVSLICE)), md5(AFCTL)[:12]
aftcsum = line_of(r"^红路声明＝\d+ 条｜保持绿.*$", afctl, "只读仪器控件面汇总行")
AFNCASE = int(one(r"^红路声明＝(\d+) 条", aftcsum, "只读仪器控件面红路数"))
AFNGREEN = field(aftcsum, "保持绿", "只读仪器控件面")
AFNRUN = field(aftcsum, "实跑行数", "只读仪器控件面")
AFNTO = field(aftcsum, "超时（环境故障，不判红也不判过）", "只读仪器控件面")
AFNASS = field(aftcsum, "装配失败（锚点问题，不是红路）", "只读仪器控件面")
AFNUNC = field(aftcsum, "造不出红路的闸门", "只读仪器控件面")
AFNSELF = int(one(r"本面自查＝(\d+) 条", afctl, "只读仪器控件面自查条数"))
AFVERD = one(r"^(CONTROL_VERDICT = .*)$", afctl, "只读仪器控件面判读行")
assert HDRNENV == AFNTO == 0, "表头面环境故障／只读仪器面超时不为零（%d／%d）⇒ 它不能当全过的档案引用" % (
    HDRNENV, AFNTO)
assert MSGNTO == 0, "提交说明控件面超时 %d 条 ⇒ 它不能当全过的档案引用" % MSGNTO
# 三面判读是否同一个字符串＝现比，不写死；只有相等才许用"三面判读一致"这一句
assert MSGVERD == HDRVERD == AFVERD, "三面判读不一致（%s／%s／%s）⇒ 不能合并成一句引用" % (
    MSGVERD, HDRVERD, AFVERD)
VERD3 = MSGVERD
# 继承常量从上一片控件件**源码**现取（写死就会在别人改那两行后腐烂）
CTL_SRC = open(OLD("am_s%d_hdr_controls.py" % PREVSLICE), encoding="utf-8").read()
SUBTO = one(r"^SUB_TIMEOUT = (\d+)$", CTL_SRC, "子运行上限常量")
ENVSTOP = one(r"^ENV_FAIL_STOP = (\d+)$", CTL_SRC, "环境故障累计常量")
WINDOW = one(r"行号落在该句在\*\*副本\*\*里的 ±(\d+) 行窗口", CTL_SRC, "raise 行号窗口常量")
# 该面的配额代价那两枚：只在该面自己印了的时候才引（本轮实测它的形制是「环境故障重跑＝N 次」）
HDRRERUN = int(one(r"^环境故障重跑＝(\d+) 次", hdr, "表头控件面环境重跑次数"))
HDRWAITN = int(one(r"等待触发 (\d+) 次", hdr, "表头控件面配额等待次数"))
HDRWAITSEC = int(one(r"等待累计 (\d+) 秒", hdr, "表头控件面配额等待秒数（墙钟代价，不是性能数字）"))

# ---- 只读仪器本身与落地腿、归档自查腿 ----
AFOUT = OLD("_raw_s%d_hdr_after.out" % PREVSLICE)
afd = open(AFOUT, encoding="utf-8").read()
AFB = os.path.getsize(AFOUT)
AFVERDICT = one(r"^(HDR_AFTER_VERDICT = .*)$", afd, "只读仪器判读行")
TRIPLE = one(r"cmp -i 三联（本件自己起进程跑的，不是抄生成器的）＝ (\d+) / (\d+) / (\d+)",
             afd, "只读仪器的 cmp 三联")
assert TRIPLE == ("0", "1", "1"), "上一片回执的 cmp 三联＝%s（须 0/1/1）⇒ 那张回执不能当全绿引用" % (TRIPLE,)

# ---- 顶格字段数：三个出处必须同值，否则这个槽不许落笔 ----
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
NDEFR = str(_decl)
print("   顶格字段＝%s 行（FIELDS 声明＝只读仪器 %s／本件现数 %s，三出处同值）｜表头非空行数（含续行）＝%d，两者不是同一个量"
      % (NDEFR, "/".join(AFFIELDS), _now, HDRLINESR))
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
# 自查件那句"面内自洽"的正对照本轮仍在档，且它数的是**三类和**——本件的分类层必须与它同口径
assert "三类和＝合计" in arch, "自查件里没有那句「三类和＝合计」的正对照 ⇒ 本件三去向分类的档案出处变了，要先重取"

# ---- 上一片那七处红灯现场／探针：名字由 glob 现取，字节数与首条抛出行由现场件现取 ----
SCENES = sorted(glob.glob(OLD("_raw_s%d_*.attempt*.out" % PREVSLICE))
                + glob.glob(OLD("_raw_s%d_*.probe-*.out" % PREVSLICE)))
assert len(SCENES) >= 2, "glob 现取到的现场件只有 %d 份 ⇒ 那句「上一轮有 N 处红灯现场」会渲染成假零" % len(SCENES)
ATT_N = len([p for p in SCENES if ".attempt" in os.path.basename(p)])
PRB_N = len(SCENES) - ATT_N
RE_RAISE = re.compile(r"^(AssertionError|UnboundLocalError|ValueError|SystemExit|TypeError)", re.M)


def scene_txt(p):
    """现场件的一行摘要＝**逐字节从该件现取**：stderr 有内容就取它那条抛出行；
    stderr 为空时，探针会把子进程的抛出行**回显在 stdout**，控件面则留下判读行＋顶格「× 」行。

    三类形状不合并成一个数（承接上一轮那条：一条未命中行既可能是判据空转也可能是环境故障）。
    """
    stem = os.path.basename(p)[:-4]
    err = p[:-4] + ".err"
    ob, eb = os.path.getsize(p), os.path.getsize(err)
    body = open(p, encoding="utf-8").read()
    etxt = open(err, encoding="utf-8").read() if eb else ""
    if eb:
        line = RE_RAISE.search(etxt)
        assert line, "%s 的 stderr 有 %d B 却数不出抛出行 ⇒ 现场形状变了，本件的摘要要按实测重写" % (stem, eb)
        tag = "抛出「%s」" % line.group(0)
    elif RE_RAISE.search(body):
        tag = "回显抛出「%s」" % RE_RAISE.search(body).group(0)
    else:
        xl = len([l for l in body.splitlines() if l.startswith("× ")])
        tag = "判读「%s」＋「× 」行 %d 条" % (one(r"^(CONTROL_VERDICT = .*)$", body, "现场判读行"), xl)
    return "`%s` %d／%d B＝%s" % (stem, ob, eb, tag)


SCENETXT = "｜".join(scene_txt(p) for p in SCENES)
print("   上一轮红灯现场与探针＝glob 现取 %d 处（attempt %d／probe %d），逐处的字节数与抛出句从现场件现取"
      % (len(SCENES), ATT_N, PRB_N))

# ---- 尺子族：按 md5 分组数；期望份数由上一节那句"本片抄第 N 份"现取（两出处闭合） ----
FAM = "0d20117bb82f"
allr = glob.glob(os.path.join(REPO, "docs/evidence/*/am_s7_verify*.py"))
fam_files = sorted(p for p in allr if md5(p)[:12] == FAM)
stray = sorted(os.path.basename(p) for p in allr if md5(p)[:12] != FAM)
RULE_N, RULE_GLOB = len(fam_files), len(allr)
prev_body = devtxt[_secs[-1].end():]
RULE_WANT = int(one(r"本片抄第 (\d+) 份", prev_body, "上一节登记的「本片抄第 N 份」"))
assert os.path.basename(OLD("am_s7_verify%d.py" % PREVSLICE)) in [os.path.basename(x) for x in fam_files], \
    "上一片那把尺子不在 md5 族里 ⇒ 它不是逐字节副本，那句「同一把尺子量多片」是假话"
assert RULE_N == RULE_WANT, "md5 族现数 %d 份 ≠ 上一节登记的「本片抄第 %d 份」⇒ 分组句与两出处都要重取" % (
    RULE_N, RULE_WANT)
assert stray, "同一条 glob 形状与 md5 族完全重合 ⇒ 那句「差额有名字」本轮不成立，要改写"
print("   尺子＝md5 前 12 `%s` 一族现数 %d 份（＝上一节那句「本片抄第 %d 份」）｜同一 glob 形状 %d 份，差额有名字＝%s ⇒ 本片抄第 %d 份"
      % (FAM, RULE_N, RULE_WANT, RULE_GLOB, "、".join(stray), RULE_N + 1))

# ---- SRC 指纹：两条 locale 各现算一次（上一轮那条新学的实证，也是它自己的靶） ----
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


# ---- GPU：NVML 那把尺子仍在坏（本轮复测），替代口径由 /proc 的 argv[0] 整词匹配承担 ----
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


nv = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                    capture_output=True, text=True)
_lines = [l.strip() for l in nv.stdout.splitlines() if l.strip()]
NVML_OK = nv.returncode == 0 and all(re.fullmatch(r"\d+", l) for l in _lines)
NVML_WHY = (_lines[0] if _lines else "").strip()
DRV = open("/sys/module/nvidia/version", encoding="utf-8").read().strip()
LIBM = re.search(r"NVML library version: ([\d.]+)", nv.stdout + nv.stderr)
LIB = LIBM.group(1) if LIBM else ""
REC_NVRC = int(one(r"nvidia-smi \*\*仍起不来\*\*（rc=(\d+)", line_of(r"^GPU\s+:.*$", r31, "回执表头的 GPU 行"),
                  "上一片回执表头记的 nvidia-smi rc"))

NPROCS = [d for d in os.listdir("/proc") if d.isdigit()
          and d not in (str(os.getpid()), str(os.getppid()))]
mine_proc = [d for d in NPROCS if argv0(d) == INTERP]
loose = [d for d in NPROCS if INTERP in (argv0(d) or "")
         or INTERP in cmdline_raw(d).decode("utf-8", "replace").replace("\x00", " ")]
upgr = [d for d in NPROCS if comm(d).startswith("unattended-upgr")]
assert match_argv0((INTERP + "\0x.py").encode()) is True, "argv[0] 匹配式的正对照自己失效了"
assert match_argv0(b"/bin/bash -c grep " + INTERP.encode() + b"\0") is False, \
    "argv[0] 匹配式把子串也认成命中 ⇒ 它和松散匹配一样会误指自己"
FAKE_SHELL = ("/bin/bash\0-c\0cd /x && timeout 300 %s docs/evidence/%s/am_s%d_docs.py --write\0"
              % (INTERP, TODAY, SLICE)).encode("utf-8")


def loose_rule(blob):
    return INTERP in blob.decode("utf-8", "replace").replace("\x00", " ")


assert loose_rule(FAKE_SHELL) is True and match_argv0(FAKE_SHELL) is False, \
    "合成命令行没复现出「松散命中／整词不命中」的形状 ⇒ 那句\"松散口径会误指承载它的 shell\"没有出处"
LOOSE_DEMO = "1／0"
LOOSE0 = "、".join(sorted({(argv0(d) or "？") for d in loose}) or ["无"])
assert not NVML_OK, "nvidia-smi 本轮又能用了（rc=%d）⇒ 那条「普查仪器仍坏着」的承接登记已腐烂，要改写" % nv.returncode
assert DRV and LIB and DRV != LIB, \
    "NVML 起不来（rc=%d／首行「%s」）但驱动 %s 与库 %s 看不出版本差 ⇒ 那句根因没有出处" % (
        nv.returncode, NVML_WHY, DRV, LIB)
assert nv.returncode == REC_NVRC, "本轮 nvidia-smi 的 rc=%d ≠ 上一片回执表头记的 %d ⇒ 那条环境事实的形状变了" % (
    nv.returncode, REC_NVRC)
NVME, GPU_TOT = str(len(mine_proc)), str(len(NPROCS))
GPU_WHY = "NVML 仍不可用（rc=%d／「%s」／内核模块 %s ≠ 用户态库 %s）" % (
    nv.returncode, NVML_WHY, DRV, LIB)
assert mine_proc == [], "替代普查数出我的解释器进程 %s ⇒ 与「不碰 GPU」冲突" % mine_proc
LOOSE_N = str(len(loose))
print("   GPU 归属：%s｜替代口径（/proc argv[0] 整词）＝我的解释器进程 %s/%s 个｜松散子串同清单现数 %s 个（argv[0]＝%s）｜合成对照＝松散 1／整词 0｜unattended-upgrade 进程 %d 个（旁证）"
      % (GPU_WHY, NVME, GPU_TOT, LOOSE_N, LOOSE0, len(upgr)))

# ---- 计划文档的补记序号：由该面现取最大值＋1 ----
plantxt = open(PLAN, encoding="utf-8").read()
_w = [int(m.group(1)) for m in re.finditer(r"^### §W 补记 (\d+)", plantxt, re.M)]
assert _w, "计划文档里取不到 §W 补记 的编号 ⇒ 序号是凭记忆的"
WN = max(_w) + 1
assert plantxt.count("### §W 补记 %d" % WN) == 0, "§W 补记 %d 已存在 ⇒ 本件要重跑先起新号" % WN

# 该面自查那几条的名字（从上一片表头控件面的 stdout 现取，不转述；括注在「各验一次」之后还有下半句，所以锚到那里就停）
HDRNSELF_NAMES = one(r"本面自查＝\d+ 条（(.+?)；每条都两个方向各验一次", hdr, "表头控件面自查条目的名字")
# 上一节登记给本片的形制要求：**从 §26.(本片−1) 那一节的正文正则现取**，不凭对话记忆复述
PREV_BODY = dict((n, b) for n, b in _bodies)[str(PREVSEC)]
REQUIRE_PAT = r"回执表头生成器的 `bullet_for` 要求「[^」]+」的顶级子条在\*\*同一节内唯一\*\*"
REQUIRE = one("(" + REQUIRE_PAT + ")", PREV_BODY, "上一节登记的形制约束原文")
# 正对照：同一条针在**整份开发日志**里必须命中多节（每节的形制条都写它）⇒ 证明"必须 scoped 到上一节"不是装饰
_multi = len(re.findall(REQUIRE_PAT, devtxt))
assert _multi >= 2, "该针在全日志只命中 %d 处 ⇒ scoped 到上一节这一步本轮没有靶（要么针写歪了，要么形制条消失了）" % _multi
# 反向对照：承接口径那条说「只有单一出处的读数会腐烂」——它的靶是 HEAD 主题那枚节号针，本轮实测 0 命中；
# 这里给该检测器配一条合成正对照（把针喂给一个**带**节号的主题），它必须数出 1，否则那句"命中 0 处"是尺子坏了。
_probe = "S7 分片 31：…；开发日志 §26.32"
assert re.findall(r"开发日志 §26\.(\d+)", _probe) == ["32"], "节号针的合成正对照失明 ⇒ 上面那个 0 是假零"

CTX = {
    "TIME": TIME, "HEAD7": HEAD7, "PARENT7": PARENT7, "SUBJ": SUBJ1, "CDATE": CDATE,
    "SLICE": str(SLICE), "PREVSLICE": str(PREVSLICE), "NEXTSLICE": NEXTSLICE,
    "SECN": str(SECN), "PREVSEC": str(PREVSEC),
    "SECREF": "26.%d" % SECN, "PREVSECREF": "26.%d" % PREVSEC,
    "UNT": str(NU), "NEW": str(NA), "MOD": str(NM), "TOT": str(NA + NM),
    "BKT": BKT_TXT, "BKTSUM": BKT_SUM, "MODTXT": MOD_TXT, "NGRP": str(len(ROLES)),
    "SPLITCTRL": SPLITCTRL, "CENSUSCTRL": CENSUSCTRL,
    "PREVLOG": PREVLOG, "RCPTB": str(RCPTB), "RCPTL": str(RCPTL), "OFFR": str(OFFR),
    "BODYR": BODYR, "NDEFR": NDEFR, "HDRLR": str(HDRLINESR), "VERDR": VERDR, "CLOSURER": CLOSURER,
    "MSGCTLB": str(MSGCTLB), "MSGCTLM": MSGCTLM, "MSGERRB": str(MSGERRB),
    "MSGNCASE": str(MSGNCASE), "MSGNGREEN": str(MSGNGREEN), "MSGNUNC": str(MSGNUNC),
    "MSGNSELF": str(MSGNSELF), "MSGVERD": MSGVERD,
    "HDRCTLB": str(HDRCTLB), "HDRCTLM": HDRCTLM, "HDRERRB": str(HDRERRB),
    "HDRNRED": str(HDRNRED), "HDRNGREEN": str(HDRNGREEN), "HDRNUNC": str(HDRNUNC),
    "HDRNSELF": str(HDRNSELF), "HDRNSELFN": HDRNSELF_NAMES, "HDRVERD": HDRVERD,
    "HDRRERUN": str(HDRRERUN), "HDRWAITN": str(HDRWAITN), "HDRWAITSEC": str(HDRWAITSEC),
    "AFCTLB": str(AFCTLB), "AFCTLM": AFCTLM, "AFERRB": str(AFERRB), "AFNCASE": str(AFNCASE),
    "AFNGREEN": str(AFNGREEN), "AFNUNC": str(AFNUNC), "AFNRUN": str(AFNRUN),
    "AFNSELF": str(AFNSELF), "AFVERD": AFVERD,
    "AFB": str(AFB), "AFVERDICT": AFVERDICT, "TRIPLE": " / ".join(TRIPLE),
    "SUBTO": SUBTO, "ENVSTOP": ENVSTOP, "WINDOW": WINDOW,
    "LANDPYB": str(os.path.getsize(LANDPY)), "LANDPREB": str(os.path.getsize(LANDPRE)),
    "LANDWB": str(os.path.getsize(LANDW)),
    "ARCHPYB": str(ARCHPYB), "ARCHPYM": ARCHPYM, "ARCHOUTB": str(ARCHOUTB),
    "ARCHERRB": str(ARCHERRB), "ARCHCLAIMS": ARCHCLAIMS, "ARCHNCTRL": str(ARCHNCTRL),
    "ARCHVERD": ARCHVERD, "DEVCOV": "%d（独有 %d）" % (DEVCOV[1], DEVCOV[2]),
    "COVTXT": "／".join("%s＝%d（独有 %d）" % (os.path.basename(f), n, u) for f, n, u in COV),
    "SCENEN": str(len(SCENES)), "ATTN": str(ATT_N), "PRBN": str(PRB_N), "SCENETXT": SCENETXT,
    "VERD3": VERD3,
    "FAM": FAM, "RULE_N": str(RULE_N), "RULE_NEW": str(RULE_N + 1),
    "RULE_GLOB": str(RULE_GLOB), "STRAY": "、".join(stray),
    "FINGER_DEF": FINGER_DEF, "FINGER_C": FINGER_C,
    "NVME": NVME, "NPROCS": GPU_TOT, "GPUWHY": GPU_WHY, "LOOSE": LOOSE_N,
    "LOOSE0": LOOSE0, "LOOSEDEMO": LOOSE_DEMO, "UPGR": str(len(upgr)),
    "DRV": DRV, "LIB": LIB, "NVRC": str(nv.returncode), "NVWHY": NVML_WHY, "RECNVRC": str(REC_NVRC),
    "EVIDNEW": EVID_NEW, "EVIDOLD": EVID_OLD, "WN": str(WN), "NCODE": str(NCODE),
    "LESSON": LESSON, "LESSON_NEW": LESSON_NEW, "REQUIRE": REQUIRE,
    "G9SEC": G9SEC, "DAILYNAME": os.path.basename(DAILY),
}

# ============================ 五张面 ============================
SEC_TITLE = ("### 26.%(SECN)s 分片 %(SLICE)s＝提交腿：分片 %(PREVSLICE)s 的落地半腿整条链入库"
             "（已暂存新增 %(NEW)s＋修改 %(MOD)s＝%(TOT)s 处）＋一条**口径级**新学："
             "同日轮的链归属只能由文件名的片号切分，目录不给唯一性（%(TIME)s 现取）\n\n") % CTX

DEV_T = SEC_TITLE + """- **形制约束的落实＝本轮另起 §%(SECREF)s，不在 §%(PREVSECREF)s 内追加**：上一节落地子条末尾登记的那条可核查约束——回执表头生成器的 `bullet_for` 要求「引用上一片回执档案名」的顶级子条在**同一节内唯一**——本节由结构满足：新节号、节内该句式恰好 1 条；那句话在档案里的原文由本件从 §%(PREVSECREF)s **正则现取**（`REQUIRE`＝「%(REQUIRE)s」），不是凭对话记忆复述。本条刻意**不复述那个档案名**（在本节复述就会出现第二条命中，而那正是这条闸门要拦的事，所以它同时也是「这条闸能红」的证明）。同节另外三处唯一性一并照查：带引号的钉住口径登记恰好 1 条、承接的 LESSON 锚句恰好 1 条、本轮新学的锚句恰好 1 条；写入件在落盘前对片段各数一次、落盘后再对磁盘上的整节数一次，两次不等就报错停止。
- **入库面（本条存在的唯一理由＝让分片 %(SLICE)s 的回执表头把「引用上一片回执」这件事从档案现取，而不是凭对话记忆）**：本节里唯一那条引用上一片回执名的子条就是这一条，档案名＝`%(PREVLOG)s`（%(RCPTB)s B／%(RCPTL)s 行＝表头 %(OFFR)s＋正文 %(BODYR)s，`BODY_OFFSET=%(OFFR)s` 已由写入件重解一遍并对上正文区；顶格字段 %(NDEFR)s 行（＝表头生成器 `FIELDS` 的 ast 声明数＝只读仪器那句两处读数，三个出处同值；另有表头非空行数含缩进续行＝%(HDRLR)s，两者不是同一个量）；判读行「%(VERDR)s」；闭合式原文「%(CLOSURER)s」；它与本件同在 `%(EVIDOLD)s/`＝**同日轮**，该目录由全树 glob 现取，而"恰好 1 份"这件事本轮**不是**由目录给的：同一条不带片号的 glob 现取到 %(SPLITCTRL)s ⇒ 唯一性来自文件名里的片号）。落笔现数（`git status --porcelain --untracked-files=all` 按三去向分类）＝未跟踪 **%(UNT)s**＋已暂存新增 **%(NEW)s**＋修改 **%(MOD)s**＝**%(TOT)s** 处。%(NEW)s 件按角色分 %(NGRP)s 组：%(BKT)s；%(BKTSUM)s 与「未跟踪＋已暂存新增」的并集现数相符（分组＝按文件名模式现数，一件落进 0 组或 ≥2 组就拒绝写盘；每组标签尾部那个登记数**也在写盘前跟现数再对一次**，不留到提交信息生成器才发现）。未跟踪 **%(UNT)s** 是本轮那条流程前提的直接读数（每件落盘即 `git add`），不是巧合；也正因如此，上一片的整条链此刻**不在** `^??` 里——旧的只扫未跟踪的分组尺子在这一棵树上：%(CENSUSCTRL)s。修改 %(MOD)s 件＝已跟踪的档案面（%(MODTXT)s；同日轮里 `%(DAILYNAME)s` 这一张**既是上一片的纪实也是本片的纪实**，上一版那句"另有一张上一片的纪实文件"在本轮不成立，链的分界只能看段落的片号与时间戳）。本片自己的干跑控件件与其原始面在提交前一并暂存 ⇒ 最终入库数**大于**这里的 %(TOT)s，那一部分由提交之后 `git show --name-status HEAD` 现取并写进本节的落地子条，不在这里预支。⚠ 本条里所有件数都是落笔那一刻的现取值，**下一轮必须重新现取**。
- **钉住口径与红线（当前「继续，不碰GPU」）**：本轮每条计算命令逐条内联 `CUDA_VISIBLE_DEVICES=\"\"`（含一行式命令）；**不产生任何吞吐／加速／预算数字**，墙钟只作可行性代价登记；GPU 归属本轮**仍然没有 NVML 那把尺子**（%(GPUWHY)s，与上一片回执表头那枚 rc=%(RECNVRC)s 同值＝本轮复测对上了一次），替代断言＝`/proc/*/cmdline` 的 argv[0] **整词**匹配我的解释器 ⇒ 落笔现数我的解释器进程 **%(NVME)s** 个（分母＝同一趟扫到的 **%(NPROCS)s** 个 PID，已排除自身与父进程），该匹配式配一正一负两条对照，另配一条**不靠现场运气**的合成对照（argv[0]＝`/bin/bash`、参数里带着我的解释器路径：两种口径现数＝%(LOOSEDEMO)s（松散子串／整词）；同一条清单上松散子串匹配现数 %(LOOSE)s 个，argv[0] 依次为 %(LOOSE0)s）；卡上 compute 进程总数本轮不可现取，所以不写进任何档案面，也不拿"上次那个数"顶替。机器上此刻还有 %(UPGR)s 个 unattended-upgrade 进程在跑，那是旁证不是结论。每次尝试换档案名，红件先入库再复跑；A0 断言一字不动、#27 不换指标、无新增 skip/xfail。生产树 `src`＋`tests` 现数改动 **%(NCODE)s** 件 ⇒ 本片仍是纯归档腿。
- **%(LESSON)s＝从分片 %(PREVSLICE)s 的落地半腿带进本片的口径，本轮当场再兑现一次**：那一轮的那条新学说「一条只有**单一出处**的仪器读数会在下一片腐烂」，它自己的出处是一枚**正则针**；本片改造上一片的信息生成器时，那枚「从 HEAD 提交说明里取节号」的针在本轮主题上命中 **0** 处（`git log -1 --format=%%s` 现取，主题里没有「开发日志 §」那枚字样），照抄下来就会让下一片的表头生成器拿一个空读数当依据。修法已进本件：**两出处互证**——节号由开发日志最后一个 `### 26.NN` 现取（＝§%(PREVSECREF)s），并与上一片回执表头 `ENV` 行那句「钉住口径从档案现取＝开发日志 §26.NN」的读数当场比相等，两值不等就拒绝写盘；同时给这条口径配一条**反向对照**：一旦 HEAD 主题又能取到那枚针，本件就红（那说明"主题不可靠"这句话本轮不成立，登记要改写）。⇒ 凡进入散文的读数都要问第二遍"它还有没有别的出处"；只有单一出处的，那个出处本身要带一条能红的对照。
- **本轮新学（口径级两条，都是仪器侧的事，不是判据缺陷也不是偏好）＝%(LESSON_NEW)s**：① **同日轮的链归属**：本片与上一片同处 `%(EVIDOLD)s/`，上一版仪器里那两条把"上一片回执在另一个日期目录"当前提的闸门（`assert EVID_OLD != EVID_NEW`）本轮**为假**；本件把它换成了命名键——回执由 `docs/evidence/*/am_s7_verify<片号>.log` 的 glob 现取并要求恰好 1，再配一条对照证明这个 1 不来自目录：去掉片号的同一条 glob 现取 %(SPLITCTRL)s。② **暂存态的链**：本轮每件落盘即 `git add`（那条流程前提的产物），于是上一片的整条链此刻是**已暂存新增**而不是未跟踪；沿用上一版"分组只扫 `^??`"的尺子会给**每一组**都数出 0（%(CENSUSCTRL)s），而"某组为 0"正是那把尺子自己判红的形状——也就是说，一条**错的分类层**会被一条**对的闸门**表现为"命名规则与磁盘脱节"，把人引向改桶形而不是改分类。本件的分类层重写为三去向（未跟踪／已暂存新增／已跟踪改动，含 `MM`），分组扫「未跟踪＋已暂存新增」的并集，并留一条把桶形改窄恰好少认领 1 件的正对照。两个读数都留在本节，供下一轮自查件回搜。
- **跑前登记的闸门（分片 %(SLICE)s 的八条，逐条可红，跑后读数从归档件现取）**：① 尺子复用＝**逐字节副本**——`cp -p` 之后 md5 前 12 必须等于 `%(FAM)s` 这一族，份数按 **md5 分组**数而不是按 glob 命中数（落笔现数 %(RULE_N)s 份＝上一节那句「本片抄第 %(RULE_N)s 份」，同一条 glob 形状 %(RULE_GLOB)s 份，差额有名字＝%(STRAY)s；本片抄第 %(RULE_NEW)s 份）；② 五步落地证明（`ls-remote`／`truncated` 须 false／blob＋tree 两侧计数／双向路径差集且两侧同用 `core.quotePath=false` 与 `LC_ALL=C sort`／逐文件 blob SHA 普查带分母）＋闭合式 远端＝父提交＋count(A)，四个正对照 A／B／C／D 各自能红；③ 回执表头只由生成器产出（预检面与写盘面各留原始件，`BODY_OFFSET` 由不动点迭代，写盘后**另跑一件只读仪器**做 `cmp -i 偏移:0` 三联＝0/1/1，顶格字段数与生成器 `FIELDS` 声明互证）；④ 提交说明由信息生成器产出（PLACEHOLDER＋`str.replace`、双向查「未取值的字段」与「模板没用到的读数」、shortstat 三个数闭合到 `--numstat` 逐行现算、指纹四路闭合＝**默认 locale 下**现算＝上一片回执表头 SRC 指纹）；⑤ 入库清单按三去向分类现数（并集覆盖＝每件恰好落进 1 组、标签登记数＝现数、旧口径每组现数 0 那条对照在档），仍要求清单里不许出现以 `/` 结尾的折叠行；⑥ 三面干跑控件面按「rc≠0 **且**抛出的那一行含指定闸门句 **且**行号落在该句在该**副本**里的 ±%(WINDOW)s 行窗口」判红，且**一条 × 行要分三去向**（命中／环境／空转，本轮承接上一片那处：同一条未命中行既可能是判据空转也可能是环境故障，混数就会把环境当红线），并如实登记造不出红路的闸门；表头面每条扰动都要复跑一次联网尺子 ⇒ 约 %(HDRNRED)s 次 GitHub 未认证 API 调用、配额 60/小时，各面分开排在配额复位之后（上一片那一面实测等待 %(HDRWAITN)s 次、累计 %(HDRWAITSEC)s 秒墙钟，只作可行性代价登记、不是性能数字），并继承上一片的常量（子运行上限 %(SUBTO)s s、环境故障累计 %(ENVSTOP)s 条即停，超时**既不判红也不判过**、单独计数）；⑦ **良性扰动必须逐字节等长**（上一片实测：把表头某值改长 2 字节，偏移由不动点重算 absorbing 掉了长度差，`cmp` 三联、顶格字段、残留检查全绿，唯一抓住它的是"写盘面打印的总字节数 vs 盘上现读"那条**跨出处闭合** ⇒ 在跨闭合校验器面前不等长的扰动测的是校验器不是判据）；⑧ 归档自查件（新追加段读数普查＋下一片锚点复验＋逐字节复跑，且 stdout 不印任何会被档案引用的自指数）。**流程前提两条一并继承**：任何仪器落盘后先 `git add` 再跑它自己的控件面（编辑一张已暂存的仪器会让未暂存漂移闸门先打掉未扰动基线）；本件自己的预检／写盘两面先落盘再跑（否则本件的 stdout 会把计数推着走＝往轮 attempt1 那个自指形状）。
- **分片 %(SLICE)s 落地腿＝提交＋推送＋五步落地证明＋生成器回执表头＋三面控件面＋落地填值＋归档自查（%(TIME)s 现取；生产树 src＋tests 改动 %(NCODE)s、GPU 全程未碰）**：本条是**写盘前**登记的计划，不是结果——落地读数（本片 HEAD 全哈希、入库清单 A／M 计数、闭合式、回执表头字节与 `BODY_OFFSET`、三面控件面的扰动条数与 UNCOVERED）此刻**还不存在**；它们由提交之后 `git show --name-status HEAD` 与回执原始件现取，写进本节的落地子条，并交**分片 %(NEXTSLICE)s** 的归档半腿做自查。上一片的同类件在档可查，引用即从档案正则现取（汇总读数一律先**锁到整行**再取字段，因为「保持绿」「造不出红路的闸门」这些词在逐条读数里会被复述）：提交说明控件面 stdout %(MSGCTLB)s B（md5 %(MSGCTLM)s／stderr %(MSGERRB)s B）＝红路 %(MSGNCASE)s 条、保持绿 %(MSGNGREEN)s 条、本面自查 %(MSGNSELF)s 条、造不出红路 %(MSGNUNC)s 条 ⇒「%(MSGVERD)s」；表头控件面 stdout %(HDRCTLB)s B（md5 %(HDRCTLM)s／stderr %(HDRERRB)s B）＝红路 %(HDRNRED)s 条、保持绿 %(HDRNGREEN)s 条、环境故障 0 条、本面自查 %(HDRNSELF)s 条（%(HDRNSELFN)s）、造不出红路 %(HDRNUNC)s 条、环境故障重跑 %(HDRRERUN)s 次 ⇒「%(HDRVERD)s」；只读仪器的控件面 stdout %(AFCTLB)s B（md5 %(AFCTLM)s／stderr %(AFERRB)s B）＝红路 %(AFNCASE)s 条、保持绿 %(AFNGREEN)s 条、实跑 %(AFNRUN)s 条、本面自查 %(AFNSELF)s 条、造不出红路 %(AFNUNC)s 条 ⇒「%(AFVERD)s」；只读仪器本体 stdout %(AFB)s B＝三联 %(TRIPLE)s、判读行「%(AFVERDICT)s」；归档自查件 %(ARCHPYB)s B（md5 %(ARCHPYM)s）及其 stdout %(ARCHOUTB)s B／stderr %(ARCHERRB)s B＝声称 %(ARCHCLAIMS)s 条全部命中原样、正对照逐行数到 %(ARCHNCTRL)s 条、五面承载（%(COVTXT)s）⇒「%(ARCHVERD)s」；落地腿 %(LANDPYB)s B 与其预检／写盘两面 %(LANDPREB)s／%(LANDWB)s B 一并入库。上一轮那 %(SCENEN)s 处红灯现场与探针（attempt %(ATTN)s／probe %(PRBN)s）逐件按名入库，字节数与抛出行现取：%(SCENETXT)s。
"""

OVER_T = """- **%(TIME)s（分片 %(SLICE)s 提交腿的写盘前登记；生产树 src+tests 改动 %(NCODE)s、GPU 全程未碰）**（开发日志 §%(SECREF)s「分片 %(SLICE)s＝提交腿：分片 %(PREVSLICE)s 的落地半腿整条链入库」）：
  - **入库面**：替上一片入库整条链——`%(PREVLOG)s` **%(RCPTB)s** B／**%(RCPTL)s** 行（`BODY_OFFSET=%(OFFR)s`＝表头 %(OFFR)s＋正文 %(BODYR)s，判读行「%(VERDR)s」、闭合式「%(CLOSURER)s」）＋提交说明控件面（stdout **%(MSGCTLB)s** B，md5 前 12 `%(MSGCTLM)s`＝红路 **%(MSGNCASE)s** 条、造不出红路 **%(MSGNUNC)s** 条）＋表头控件面（stdout **%(HDRCTLB)s** B＝红路 **%(HDRNRED)s**／保持绿 **%(HDRNGREEN)s**／本面自查 **%(HDRNSELF)s**／造不出红路 **%(HDRNUNC)s**）＋只读仪器与其控件面（仪器 stdout **%(AFB)s** B＝三联 %(TRIPLE)s；控件面 **%(AFCTLB)s** B＝红路 %(AFNCASE)s／实跑 %(AFNRUN)s／造不出红路 %(AFNUNC)s）＋归档自查件 **%(ARCHPYB)s** B（md5 前 12 `%(ARCHPYM)s`；声称 **%(ARCHCLAIMS)s** 条全命中、正对照 **%(ARCHNCTRL)s** 条能红）＋落地腿与上一轮 %(SCENEN)s 处红灯现场／探针。落笔现数＝未跟踪 **%(UNT)s**＋已暂存新增 **%(NEW)s**＋修改 **%(MOD)s**＝**%(TOT)s** 处，分 %(NGRP)s 组＝%(BKT)s。远端 main 仍为 `%(HEAD7)s`（父 `%(PARENT7)s`，提交时刻 %(CDATE)s）。
  - **两条本轮口径**：① **%(LESSON)s**（承接上一轮，本轮当场兑现）——上一片信息生成器那枚「从 HEAD 主题取节号」的针在本轮主题上命中 0 处；本件改为**开发日志现取＋回执表头 ENV 行**两出处互证，并配一条反向对照（主题又能取到就该句改写）。② **%(LESSON_NEW)s**——本片与上一片同在 `%(EVIDOLD)s/`，那条「上一片回执在另一日期目录」的前提本轮为假 ⇒ 链归属改由**文件名的片号**切分（不带片号的同一条 glob 现取 %(SPLITCTRL)s）；同轮另一处：每件落盘即 `git add` 让上一片整条链变成**已暂存新增**，只扫 `^??` 的旧分组尺子对每组都数出 0（%(CENSUSCTRL)s）⇒ 分类层重写为未跟踪／已暂存新增／已跟踪改动三去向。
  - **仪器与排期**：尺子＝md5 前 12 `%(FAM)s` 一族现数 **%(RULE_N)s** 份（＝上一节登记的「本片抄第 %(RULE_N)s 份」；同一条 glob 形状 **%(RULE_GLOB)s** 份，差额有名字＝%(STRAY)s），本片抄第 **%(RULE_NEW)s** 份；表头控件面每条扰动复跑一次联网尺子 ⇒ 约 **%(HDRNRED)s** 次 GitHub 未认证 API 调用、配额 60/小时，各面分开排在复位之后；继承常量＝子运行上限 %(SUBTO)s s／环境故障累计 %(ENVSTOP)s 条即停／raise 行号窗口 ±%(WINDOW)s 行（超时既不判红也不判过）。排期与红线不变（原文见 §%(PREVSECREF)s 的钉住子条与计划文档 §W 补记 %(WN)s）：**#30 试片尺寸裁决在前**、`--stage full` 暂停、A0 断言一字不动、#27 不换指标、无新增 skip/xfail、性能数字只在 GPU 窗口测（本轮 NVML 仍不可用，替代断言＝我的解释器进程现数 **%(NVME)s** 个／同一趟扫到 %(NPROCS)s 个 PID，更谈不上吞吐）。本片落地读数此刻尚不存在 ⇒ 提交后现取并交**分片 %(NEXTSLICE)s** 自查。
"""

PLAN_T = """
### §W 补记 %(WN)s（%(TIME)s 落笔，`date '+%%F %%T %%z'` 现取）＝分片 %(SLICE)s 提交腿的**写盘前登记**：替分片 %(PREVSLICE)s 入库它的整条落地半腿链，并把两条**仪器侧**事实折进流程

- **外部事实只有一条**：远端 main 已在 `%(HEAD7)s`（`git ls-remote --heads origin main` 与本地 HEAD 同名，落笔时现验）；本片＝**纯归档腿**，生产树 `src`＋`tests` 改动 **%(NCODE)s** 件。入库面＝未跟踪 **%(UNT)s**＋已暂存新增 **%(NEW)s**＋修改 **%(MOD)s**＝**%(TOT)s** 处（`git status --porcelain --untracked-files=all` 按三去向分类现数；分 %(NGRP)s 组＝%(BKT)s）；`%(PREVLOG)s` 的判读行「%(VERDR)s」与闭合式「%(CLOSURER)s」由档案正则现取。**本片自己的落地证明要到提交之后才存在**，本补记只登记计划与写盘前读数。
- **两条仪器侧事实（不是判据缺陷，也不是偏好）**：① **%(LESSON)s**——只有单一出处的读数会在下一片腐烂；本轮的现场是上一片信息生成器那枚「从 HEAD 主题现取节号」的针（主题里没有「开发日志 §」），本件因此把节号改为**开发日志最后一个 `### 26.NN`** 与**上一片回执表头 ENV 行那枚**两出处互证，并加一条反向对照（该针若又能取到，登记就得改写）。② **%(LESSON_NEW)s**——本片与上一片同处 `%(EVIDOLD)s/`（上一版仪器里 `assert EVID_OLD != EVID_NEW` 那两条闸门本轮为假）⇒ 链归属只能由文件名的片号切分，唯一性由 glob 现取（不带片号的那一条现取 %(SPLITCTRL)s）；同轮另一处：每件落盘即 `git add` ⇒ 上一片的链此刻是**已暂存新增**，只扫未跟踪的旧分组尺子逐组现数 0（%(CENSUSCTRL)s），所以分类层重写为三去向。
- **继承的仪器常量与代价登记**：表头控件面的 %(HDRNSELF)s 条自查（%(HDRNSELFN)s）与三道防护本轮继承（子运行上限 %(SUBTO)s s、环境故障累计 %(ENVSTOP)s 条即停、raise 行号窗口 ±%(WINDOW)s 行；超时**既不判红也不判过**、单独计数；一条 × 行分三去向）。三面原始件在档：提交说明控件面 stdout **%(MSGCTLB)s** B（md5 前 12 `%(MSGCTLM)s`／stderr **%(MSGERRB)s** B）＝红路 %(MSGNCASE)s 条、保持绿 %(MSGNGREEN)s 条、造不出红路 %(MSGNUNC)s 条 ⇒「%(MSGVERD)s」；表头控件面 stdout **%(HDRCTLB)s** B（md5 前 12 `%(HDRCTLM)s`／stderr **%(HDRERRB)s** B）＝红路 %(HDRNRED)s 条、保持绿 %(HDRNGREEN)s 条、环境故障 0 条、重跑 %(HDRRERUN)s 次、造不出红路 %(HDRNUNC)s 条 ⇒「%(HDRVERD)s」；只读仪器的控件面 stdout **%(AFCTLB)s** B（md5 前 12 `%(AFCTLM)s`／stderr **%(AFERRB)s** B）＝红路 %(AFNCASE)s 条、保持绿 %(AFNGREEN)s 条、实跑 %(AFNRUN)s 条、造不出红路 %(AFNUNC)s 条 ⇒「%(AFVERD)s」；只读仪器本体 stdout **%(AFB)s** B＝三联 %(TRIPLE)s；归档自查件 stdout **%(ARCHOUTB)s** B／stderr **%(ARCHERRB)s** B＝声称 %(ARCHCLAIMS)s 条全命中、正对照 %(ARCHNCTRL)s 条 ⇒「%(ARCHVERD)s」。上一轮 %(SCENEN)s 处红灯现场／探针（attempt %(ATTN)s／probe %(PRBN)s）一并入库，逐件字节数与抛出行现取：%(SCENETXT)s。⚠ 表头控件面每条扰动复跑联网尺子 ⇒ 约 %(HDRNRED)s 次 GitHub 未认证 API 调用、配额 60/小时，各面分开排在复位之后（这是可行性代价登记，不是性能数字）。
- **排期与红线（不变，且依赖仍未松动）**：**#30 试片尺寸裁决在前**（G9 触边闸与「只有 N=1 可评分」的原文见 §%(G9SEC)s，节号由该短语在开发日志里现定位、并配一条"复制到另一节就要数出 2 节"的正对照，本轮未重跑 ⇒ 不复述数字）→ raw 臂 N≥2 ＋ `--stage full --measured-per-track` → **#27 腿②（缺省 tuned→hk）＋#33 的位点系数同批，且只对 A3 的 1→18 外部趋势打分** → #40→#39；#41；#37／#38；#29 等裁决；#51 已量化但**不关单**。红线：A0 断言一字不放宽、#27 不换指标、无新增 skip/xfail、性能数字只在 GPU 窗口测（本轮全程未碰 GPU，且 NVML 普查仍不可用：rc=%(RECNVRC)s／内核模块 %(DRV)s ≠ 用户态库 %(LIB)s ⇒ 无任何吞吐论断）。本片回执链与本轮控件／自查件全在工作树 ⇒ 交**分片 %(NEXTSLICE)s**。

"""

DAILY_T = """
## %(TIME)s：分片 %(SLICE)s 提交腿＝替分片 %(PREVSLICE)s 入库它的整条落地半腿链（已暂存新增 %(NEW)s＋修改 %(MOD)s＝%(TOT)s 处）；一条口径级新学＝%(LESSON_NEW)s

- **一句话**：远端 main ＝ `%(HEAD7)s`（父 `%(PARENT7)s`，提交时刻 %(CDATE)s，`git log -1 --format=%%cd` 现取），HEAD 主题＝「%(SUBJ)s…」（片号由此现取 ⇒ 本片＝%(SLICE)s）；本节写的是**提交腿落笔前**的登记，落地读数此刻还不存在。
- **入库面（分 %(NGRP)s 组由文件名模式现数，和＝%(BKTSUM)s 与「未跟踪＋已暂存新增」并集相符）**：%(BKT)s；另 %(MOD)s 张已跟踪档案面（%(MODTXT)s；同日轮里 `%(DAILYNAME)s` 既是上一片的纪实也是本片的纪实）＝未跟踪 **%(UNT)s**＋已暂存新增 **%(NEW)s**＋修改 **%(MOD)s**＝**%(TOT)s** 处。⚠ 未跟踪那个 0 是本轮那条流程前提（每件落盘即 `git add`）的直接读数；也正因如此，上一片的链在 `^??` 里数不出来——旧口径的分组尺子：%(CENSUSCTRL)s。这些数只描述落笔那一刻，分片 %(NEXTSLICE)s 必须重新现取。
- **两条本轮口径**：① **%(LESSON)s**（承接上一轮并当场兑现）——上一片信息生成器那枚「从 HEAD 主题取节号」的针本轮命中 0 处；本件改为开发日志现取＋上一片回执表头 ENV 行两出处互证，并配一条反向对照。② **%(LESSON_NEW)s**——本片与上一片同在 `%(EVIDOLD)s/`，`assert EVID_OLD != EVID_NEW` 那类闸门本轮为假 ⇒ 链归属改由文件名片号切分（不带片号的 glob 现取 %(SPLITCTRL)s）；分类层同时重写为三去向，因为只扫未跟踪的尺子对上一片整条链逐组数出 0。
- **上一片的件在档可查（引用即从档案现取，不转述）**：`%(PREVLOG)s` %(RCPTB)s B／%(RCPTL)s 行（`BODY_OFFSET=%(OFFR)s`）＝「%(VERDR)s」、闭合式「%(CLOSURER)s」；三面控件面＝提交说明 %(MSGCTLB)s B（红 %(MSGNCASE)s＋绿 %(MSGNGREEN)s／UNCOVERED %(MSGNUNC)s）、表头 %(HDRCTLB)s B（红 %(HDRNRED)s＋绿 %(HDRNGREEN)s／UNCOVERED %(HDRNUNC)s／自查 %(HDRNSELF)s／重跑 %(HDRRERUN)s）、只读仪器 %(AFCTLB)s B（红 %(AFNCASE)s＋绿 %(AFNGREEN)s／实跑 %(AFNRUN)s／UNCOVERED %(AFNUNC)s），三面判读现比一致＝%(VERD3)s；只读仪器本体三联 %(TRIPLE)s＝「%(AFVERDICT)s」；归档自查件 %(ARCHPYB)s B（md5 %(ARCHPYM)s）stdout %(ARCHOUTB)s B／stderr %(ARCHERRB)s B＝声称 %(ARCHCLAIMS)s 条全命中、正对照 %(ARCHNCTRL)s 条、五面承载＝%(COVTXT)s ⇒「%(ARCHVERD)s」；上一轮 %(SCENEN)s 处红灯现场／探针逐件按名入库（attempt %(ATTN)s／probe %(PRBN)s）。
- **钉住与红线**：每条计算命令逐条内联 `CUDA_VISIBLE_DEVICES=\"\"`（含一行式）；**不产生吞吐／加速／预算数字**；生产树改动 **%(NCODE)s** 件；尺子＝md5 前 12 `%(FAM)s` 一族现数 %(RULE_N)s 份（同一 glob 形状 %(RULE_GLOB)s 份，差额＝%(STRAY)s），本片抄第 %(RULE_NEW)s 份；GPU＝NVML 普查仍不可用（%(GPUWHY)s，与上一片回执表头那枚 rc=%(RECNVRC)s 同值），替代断言 我的解释器进程 **%(NVME)s/%(NPROCS)s**，合成对照＝松散 1／整词 0（同清单松散现数 %(LOOSE)s 个）。排期与红线不变（原文见 §%(PREVSECREF)s 与计划文档 §W 补记 %(WN)s）。
"""

WBIDX_T = """- **%(TIME)s（分片 %(SLICE)s 提交腿的写盘前登记；生产树 0 改动、GPU 全程未碰）**：
  本片替上一片入库它的整条落地半腿链：`%(PREVLOG)s`（%(RCPTB)s B／%(RCPTL)s 行、`BODY_OFFSET=%(OFFR)s`，判读行「%(VERDR)s」、闭合式「%(CLOSURER)s」）＋三面控件面（提交说明 %(MSGCTLB)s B＝红 %(MSGNCASE)s／UNCOVERED %(MSGNUNC)s；表头 %(HDRCTLB)s B＝红 %(HDRNRED)s＋绿 %(HDRNGREEN)s／UNCOVERED %(HDRNUNC)s／自查 %(HDRNSELF)s；只读仪器 %(AFCTLB)s B＝红 %(AFNCASE)s／实跑 %(AFNRUN)s／UNCOVERED %(AFNUNC)s）＋只读仪器本体（stdout %(AFB)s B＝三联 %(TRIPLE)s）＋归档自查件（%(ARCHPYB)s B，声称 %(ARCHCLAIMS)s 条全命中、正对照 %(ARCHNCTRL)s 条）＋落地腿＋上一轮 %(SCENEN)s 处红灯现场／探针。
  入库面＝未跟踪 **%(UNT)s**＋已暂存新增 **%(NEW)s**＋修改 **%(MOD)s**＝**%(TOT)s** 处，分 %(NGRP)s 组＝%(BKT)s；`src`＋`tests` 改动 **%(NCODE)s** 件 ⇒ 纯归档腿。远端 main `%(HEAD7)s`（父 `%(PARENT7)s`）。
  **两条仪器侧新学**：① **%(LESSON)s**（承接＋本轮兑现）——只有单一出处的读数会腐烂：上一片那枚「从 HEAD 主题取节号」的针本轮命中 0 处，改为两出处互证＋反向对照；② **%(LESSON_NEW)s**——本片与上一片同在 `%(EVIDOLD)s/` ⇒ 链归属只能由文件名的片号切分（不带片号的 glob 现取 %(SPLITCTRL)s），且每件落盘即 `git add` 让上一片链变成已暂存新增 ⇒ 只扫 `^??` 的旧分组尺子逐组现数 0（%(CENSUSCTRL)s），分类层重写为三去向。
  方法账沿用：尺子按 md5 分组数＝%(RULE_N)s 份（glob 形状 %(RULE_GLOB)s 份，差额＝%(STRAY)s），本片抄第 %(RULE_NEW)s 份；表头控件面每条扰动复跑联网尺子 ⇒ 约 %(HDRNRED)s 次 API 调用、配额 60/小时，各面分开排在复位之后；良性扰动必须逐字节等长；一条未命中行分三去向。
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
print("   §%d 顶级子条＝%d 条且标签互异 -> PASS" % (SECN, len(_lab)))

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
    ("已暂存新增 **", CTX["NEW"]),
    ("档案名＝`", CTX["PREVLOG"]),
    ("`BODY_OFFSET=", CTX["OFFR"]),
    ("内核模块 ", CTX["DRV"]),
    ("交**分片 ", CTX["NEXTSLICE"]),
    ("分 " + CTX["NGRP"] + " 组：", CTX["BKT"]),
    ("（attempt ", CTX["ATTN"]),
    ("＝三联 ", CTX["TRIPLE"]),
])
expect_near("OVER", texts["OVER"], [("已暂存新增 **", CTX["NEW"]), ("md5 前 12 `", CTX["FAM"]),
                                    ("我的解释器进程现数 **", CTX["NVME"])])
expect_near("PLAN", texts["PLAN"], [("远端 main 已在 `", CTX["HEAD7"]),
                                    ("内核模块 ", CTX["DRV"]),
                                    ("＝红路 ", CTX["MSGNCASE"]),
                                    ("原文见 §", CTX["G9SEC"])])
expect_near("DAILY", texts["DAILY"], [("远端 main ＝ `", CTX["HEAD7"]),
                                      ("闭合式「", CTX["CLOSURER"]),
                                      ("重跑 ", CTX["HDRRERUN"])])
expect_near("WBIDX", texts["WBIDX"], [("入库面＝未跟踪 **", CTX["UNT"]), ("远端 main `", CTX["HEAD7"])])
print("   邻接断言：五面各 3–7 对「标签＋值」在渲染结果里原文命中 -> PASS")

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
    hits, copies = [], []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        blob = cmdline_raw(d)
        toks = {os.path.basename(a.decode("utf-8", "replace")) for a in blob.split(b"\0") if a}
        if SELF_NAME in toks:
            if blob.split(b"\0")[0].decode("utf-8", "replace") == INTERP:
                copies.append(d)
            continue
        if toks & TOKENS:
            hits.append((d, sorted(toks & TOKENS)))
    return hits, copies


if "--write" in sys.argv:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "写盘腿没钉 CVD 空串 ⇒ 与 ENV 口径不符"
    faces, copies = running_faces()
    assert not faces, "有别的仪器正在跑 %s ⇒ 按口径不许动它可读的档案面" % faces
    assert len(copies) == 1, "以解释器身份跑本件的进程有 %d 个（须 1＝自己）⇒ 有两个副本同时在写档案，拒绝" % len(copies)
    for path in (DEVLOG, OVERVIEW, PLAN, DAILY, WBIDX):
        assert open(path, "rb").read()[-1:] == b"\n", "%s 末字节不是换行 ⇒ 拒绝追加" % path
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
    _c = {}
    for l in post:
        x, y, p = l[0], l[1], l[3:]
        _c.setdefault(p, "UNT" if (x, y) == ("?", "?") else ("NEW" if x == "A" else
                                                            ("MOD" if "M" in (x, y) else "OTHER")))
    nq = len([p for p, k in _c.items() if k == "UNT"])
    na = len([p for p, k in _c.items() if k == "NEW"])
    nm = len([p for p, k in _c.items() if k == "MOD"])
    assert nq == NU, "未跟踪件数从 %d 变成 %d ⇒ 写盘动到了不该动的地方" % (NU, nq)
    assert na == NA, "已暂存新增从 %d 变成 %d ⇒ 写盘过程中有人新建／暂存了文件" % (NA, na)
    assert nm == NM, "修改面数从 %d 变成 %d ⇒ 追加写到了第五张面之外" % (NM, nm)
    print("   闭合式复算：未跟踪 %d＋已暂存新增 %d＋修改 %d＝%d 处（写盘前后同数，因五张面此前已在暂存态）-> PASS"
          % (nq, na, nm, na + nm))
    print("WRITER_VERDICT = FIVE FACES LANDED, UNIQUENESS RECOUNTED ON DISK")
else:
    print("（预检模式：未写盘。加 --write。）")
print("FILLER_VERDICT = READINGS DERIVED, STRUCTURE GATES PASS")
