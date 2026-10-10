#!/usr/bin/env python3
"""分片 27 落地腿的归档写手（开发日志 (l)／计划文档 §W 补记／总览新条目＋就地更正／当日／MEMORY）。

口径继承 `am_s25_docs.py`，本轮的四件事各带一道闸门：
① **表头读数不许手抄**：审计面与正对照面的每条判读行由**复跑**取得（rc／stderr／锚定计数／
   普查的总数与三类红旗），复跑拿不到同一读数就拒绝写盘（"归档时绿、现在不绿"＝一份不能重现的回执）；
   普查行里的**类别分布**故意不复现——出处池＝整个证据目录，目录每多一件同一个数就可能换类，
   因此它不是可声称的读数，只进本件的 stdout；
② **闭合式自己也是一条声称**：本轮改表头使 4860／14260 三个字数过期 ⇒ 就地更正，且过期值检测器
   **分三面**（历史面＝HEAD 版／改前工作树面／**改后全文面**），并配"诱饵闭合式"（子串命中＝词边界＋诱饵）
   与"抹掉真值"的正对照——上一轮正是因为检测器不分面而**拒绝了自己的修复**；
   全文面依赖本写手自己追加的那些句子（它们就带着旧值引用）⇒ 该面的读数**迭代到不动点**才进模板；
③ **标本标记取自读数不取自措辞**：任何一句引用旧值都必须与真值**同段**（空行切分），否则判传染；
④ 分片 26 回执的入库读数全部从 `am_s7_verify26.log` 现取（BODY_OFFSET 不动点、cmp 三联、
   生成器 `FIELDS` 声明数、尺子全树 md5 份数），一字不引对话记忆。

模板一律 PLACEHOLDER＋`str.replace`（f-string 会把字面量里的 `{32}` 当替换字段静默吃掉＝§26.27 (q)）；
每条腿的插入块前必须是**空行**（单个换行＝把新条目粘到上一行上，§26.29 (k) ① 就是它），正对照＝少写一个换行
时同一条断言必须变红；写后回读逐字节相等；幂等（标记已存在即 SKIP）。
"""
import ast
import datetime
import glob
import os
import re
import subprocess
import sys

REPO = "/home/shy/桌面/2026年开发-显式冲击有限元-物质点法-sph-fvm+am/AM-Qoder"
D = os.path.join(REPO, "docs/evidence/2026-10-10")
DEV = os.path.join(REPO, "docs/开发日志.md")
PL = os.path.join(REPO, "docs/项目评估与下一步计划_2026-10-06.md")
OV = os.path.join(REPO, "docs/项目开发总览.md")
DAY = os.path.join(REPO, ".workbuddy/memory/2026-10-10.md")
MEM = os.path.join(REPO, ".workbuddy/memory/MEMORY.md")
FIVE = ["docs/开发日志.md", "docs/项目开发总览.md", "docs/项目评估与下一步计划_2026-10-06.md",
        ".workbuddy/memory/2026-10-10.md", ".workbuddy/memory/MEMORY.md"]
REC26 = os.path.join(D, "am_s7_verify26.log")
RAW26 = os.path.join(D, "_raw_verify26.out")
HDR26PY = os.path.join(D, "am_s26_hdr.py")
RULE26 = os.path.join(D, "am_s7_verify26.py")
AUDIT = os.path.join(D, "am_a3_hdr_audit.py")
AUDIT_RAW = os.path.join(D, "_raw_a3_hdr_audit.out")
CTL = os.path.join(D, "am_a3_hdr_audit_controls.py")
CTL_RAW = os.path.join(D, "_raw_a3_hdr_audit_controls.out")
HDR_MAIN = os.path.join(D, "am_a3_preflight_hdr.txt")
RAW_MAIN = os.path.join(D, "_raw_a3_preflight.out")
LOG_MAIN = os.path.join(D, "am_a3_preflight.log")
HDR_CTL = os.path.join(D, "am_a3_preflight_controls_hdr.txt")
RAW_CTL = os.path.join(D, "_raw_a3_preflight_controls.out")
LOG_CTL = os.path.join(D, "am_a3_preflight_controls.log")
WRITE = "--write" in sys.argv
STALE, TRUEV = "4860", None       # TRUEV 由 stat 现取
FINGER_CMD = ("find src tests -name '*.py' -not -path '*__pycache__*' -print0 "
              "| sort -z | xargs -0 md5sum | md5sum | cut -d' ' -f1")


def sh(*a):
    return subprocess.run(a, cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def md5(p):
    return subprocess.run(["md5sum", p], capture_output=True,
                          text=True).stdout.split()[0]


def txt(p):
    return open(p, encoding="utf-8").read()


def size(p):
    return str(os.path.getsize(p))


def head_of(rel):
    return subprocess.run(["git", "show", "HEAD:" + rel], cwd=REPO, capture_output=True,
                          text=True, check=True).stdout


def wb_count(text, val):
    """词边界命中（`grep -cE '(^|[^0-9])4860([^0-9]|$'` 的同一口径）：整串子串会把 `0.944860` 也数进来。"""
    return len(re.findall(r"(?<![0-9])" + val + r"(?![0-9])", text))


# ============================================================ 现取：树／钉住／脏面
HEAD = sh("git", "rev-parse", "--short", "HEAD")
HEADFULL = sh("git", "rev-parse", "HEAD")
PARENTS = sh("git", "rev-parse", "--short", "HEAD^")
por = [l for l in sh("git", "-c", "core.quotePath=false", "status", "--porcelain").splitlines() if l.strip()]
DIRTY_T, DIRTY_E = len(por), len([l for l in por if "docs/evidence/" in l])
DIRTY_S = len([l for l in por if re.search(r"(^| )(src|tests)/", l)])
assert DIRTY_S == 0, "生产树（src/＋tests/）有 %d 处改动 ⇒ 本轮回执腿不该碰求解器，拒绝归档" % DIRTY_S
FINGER = subprocess.run(["bash", "-c", FINGER_CMD], cwd=REPO, capture_output=True,
                        text=True).stdout.strip()
nv = [l.strip() for l in subprocess.run(["nvidia-smi", "--query-compute-apps=pid",
                                         "--format=csv,noheader"], capture_output=True,
                                        text=True).stdout.splitlines() if l.strip()]
nv_mine = [p for p in nv if "/tmp/amvenv" in open("/proc/%s/cmdline" % p, "rb").read()
           .decode("utf-8", "replace")]
assert nv_mine == [], "落笔时我的解释器在卡上留有进程 %s ⇒ 与「不动 GPU」冲突，拒绝写盘" % nv_mine

# ============================================================ 现取：两面表头／正文／回执的字节闭合
HDRSZ, RAWSZ, LOGSZ = (size(HDR_MAIN), size(RAW_MAIN), size(LOG_MAIN))
CHDRSZ, CRAWSZ, CLOGSZ = size(HDR_CTL), size(RAW_CTL), size(LOG_CTL)
assert int(HDRSZ) + int(RAWSZ) == int(LOGSZ), "主面 表头＋正文 ≠ 回执（%s＋%s≠%s）⇒ 复现式破了" % (
    HDRSZ, RAWSZ, LOGSZ)
assert int(CHDRSZ) + int(CRAWSZ) == int(CLOGSZ), "控件面 表头＋正文 ≠ 回执"
for a, b in ((HDR_MAIN, RAW_MAIN), (HDR_CTL, RAW_CTL)):
    r = subprocess.run(["bash", "-c", "cat '%s' '%s' | cmp - '%s'" % (a, b,
                   LOG_MAIN if a == HDR_MAIN else LOG_CTL)], capture_output=True)
    assert r.returncode == 0, "cat＋cmp 复核失败：" + a
TRUEV = HDRSZ                      # 就地更正的目标值＝表头实际字节数（现取，不手打）

# ============================================================ 现取：复跑审计件（读数必须能重现）
ENV = dict(os.environ)
ENV["CUDA_VISIBLE_DEVICES"] = ""
ENV["JAX_PLATFORMS"] = "cpu"


CENSUS = r"表头数字 (\d+) 个＝(.*?)／无出处 (\d+)／跨面无归属 (\d+)／无出处十六进制读数 (\d+)"


def census_nums(lines):
    """普查行里**能重现**的那四个数：总数＋三类红旗。

    类别分布（B/C/D/S/X 各几个）**故意不比**：审计件的出处池＝整个证据目录，本轮每往目录里
    新写一件（含本写手自己），同一个数字就可能从 B 类改判 C 类 ⇒ 分布是一条随输入面移动的读数，
    把它当"复现"来断言会造出一条永远对不齐的闸门；红旗与总数由表头正文钉住，才是可复现的。
    """
    out = []
    for l in lines:
        if "溯源普查" in l:
            m = re.search(CENSUS, l)
            assert m, "普查行取不到汇总：%r" % l[:70]
            out.append((m.group(1), m.group(3), m.group(4), m.group(5)))
    return out


def rerun(script, archived):
    r = subprocess.run([sys.executable, script], cwd=REPO, env=ENV,
                       capture_output=True, text=True)
    assert r.returncode == 0, "%s 复跑 rc=%s ⇒ 本轮的判读行不能代表这台仪器现在说什么" % (
        os.path.basename(script), r.returncode)
    assert not r.stderr, "%s 复跑 stderr 非空（%d 字节）⇒ 有警告被吞" % (
        os.path.basename(script), len(r.stderr.encode()))
    old, fresh = txt(archived).splitlines(), r.stdout.splitlines()
    # 只要求**判读类**行重现（含 mtime／dirty／时刻／池分布的行是状态相关的，不参与重现断言）
    stable = [l for l in old if re.search(r"^\s+(行首|不锚|零声称|原始件|cat )", l)
              or "本面判读" in l or l.startswith("AUDIT_VERDICT")
              or l.startswith("CONTROL_VERDICT") or "未扰动基线" in l
              or l.startswith("  ") and "｜rc=" in l]
    lost = [l for l in stable if l not in fresh]
    assert not lost, "%s 的 %d 条判读行在复跑里没重现：%s" % (
        os.path.basename(script), len(lost), lost[:2])
    co, cf = census_nums(old), census_nums(fresh)
    assert len(co) == len(cf), "%s 的普查行数在复跑里变了（归档 %d → 现跑 %d）" % (
        script, len(co), len(cf))
    if co:   # 控件面不印普查行；审计面的"两面各一条"由下面 au_lines==2 那条断言钉住
        assert len(co) == 2, "%s 印了 %d 条普查行（审计面须 2）" % (script, len(co))
        assert co == cf, "%s 的普查红旗在复跑里变了（归档 %s → 现跑 %s）⇒ 归档件不能代表现在" % (
            os.path.basename(script), co, cf)
    return r.stdout


AU = rerun(AUDIT, AUDIT_RAW)
CT = rerun(CTL, CTL_RAW)

au_lines = [l for l in AU.splitlines() if "溯源普查" in l]
assert len(au_lines) == 2, "审计面应有两面普查汇总，现取 %d 行" % len(au_lines)
(MAIN_C, CTL_C) = [re.findall(CENSUS, l)[0] for l in au_lines]
AUN, AUCLS, AUU, AUA, AUH = MAIN_C
CTN, CTCLS, CTU, CTA, CTH = CTL_C
assert (AUU, AUA, AUH, CTU, CTA, CTH) == ("0", "0", "0", "0", "0", "0"), \
    "两面普查里有红旗非 0：主面 %s/%s/%s／控件面 %s/%s/%s" % (AUU, AUA, AUH, CTU, CTA, CTH)
assert int(AUN) >= 1 and int(CTN) >= 1, "表头数字普查取回 0 ⇒ 句子切分口径变了"
AU_PASS = int(re.findall(r"行首 \[PASS\] 行数\s+实测=(\d+)", AU)[0])
AU_FAIL = int(re.findall(r"行首 \[FAIL\] 行数\s+实测=(\d+)", AU)[0])
CT_PASS = int(re.findall(r"行首 \[PASS\] 行数\s+实测=(\d+)", AU)[-1])
CT_FAIL = int(re.findall(r"行首 \[FAIL\] 行数\s+实测=(\d+)", AU)[-1])
AU_NA = int(re.findall(r"不锚标签位的 \[FAIL\] 行数\s+实测=(\d+)", AU)[0])
CT_NA = int(re.findall(r"不锚标签位的 \[FAIL\] 行数\s+实测=(\d+)", AU)[-1])
assert (AU_PASS, AU_FAIL, CT_PASS, CT_FAIL, AU_NA, CT_NA) == (37, 0, 10, 2, 0, 4), \
    "两面锚定计数变了（%s/%s/%s/%s＋不锚 %s/%s）⇒ §26.29 (b) 那几句记分要一起改" % (
        AU_PASS, AU_FAIL, CT_PASS, CT_FAIL, AU_NA, CT_NA)
AUVERD = re.search(r"AUDIT_VERDICT = .*", AU).group(0)
CTLN = int(re.search(r"扰动条数＝(\d+)", CT).group(1))
CTLVERD = re.search(r"CONTROL_VERDICT = .*", CT).group(0)
CTLB = re.search(r"C0 未扰动基线 rc=(\d+)", CT).group(1)
CTLRED = len(re.findall(r"^  C\d+ .*命中$", CT, re.M))
assert CTLB == "0" and CTLRED == CTLN, "正对照基线 rc=%s、命中 %d/%d 条 ⇒ 有闸门空转" % (
    CTLB, CTLRED, CTLN)

# ============================================================ 现取：分片 26 回执（入库那一份）
raw26 = open(REC26, "rb").read()
boff = int(re.search(rb"BODY_OFFSET=(\d+)\n", raw26).group(1))
h26, b26 = raw26[:boff].decode("utf-8"), raw26[boff:].decode("utf-8")
assert raw26[boff:] == open(RAW26, "rb").read(), "回执正文区 ≠ 归档的原始 stdout"
CMP = [subprocess.run(["cmp", "-i", "%d:%d" % (o, z), REC26, RAW26],
                      capture_output=True).returncode for o, z in ((boff, 0), (boff + 1, 0), (boff, 1))]
assert CMP == [0, 1, 1], "cmp 三联不齐（同偏移须 0、两侧 +1 须 1）：%s" % CMP
assert "@@" not in h26 and "＠" not in h26, "分片 26 表头有占位符残留"


def H26(pat, grp=1):
    m = re.search(pat, h26)
    assert m, "分片 26 回执表头取不到 " + pat
    return m.group(grp)


def B26(pat, grp=1):
    m = re.search(pat, b26)
    assert m, "分片 26 回执正文取不到 " + pat
    return m.group(grp)


VERD26 = B26(r"(LANDING_VERDICT = .*)").strip()
assert VERD26 == "LANDING_VERDICT = ALL CHECKS PASS", "分片 26 判读行不是全过 ⇒ 不能当落地证明归档"
assert HEADFULL in b26, "回执正文 ① 里没有当前 HEAD 全哈希 ⇒ 那份回执测的不是这棵树"
NB, NP = int(B26(r"blob (\d+) \+ tree")), int(B26(r"父提交 = (\d+)"))
NA, NMF = int(B26(r"新增 (\d+) \+ 修改")), int(B26(r"新增 \d+ \+ 修改 (\d+)"))
EXPECT = int(B26(r"\+ 新增 (\d+) = (\d+)", 2))
assert NB == NP + NA == EXPECT, "闭合式在正文里不成立：%d ≠ %d＋%d ≠ %d" % (NB, NP, NA, EXPECT)
DL, DR = B26(r"本地独有 (\d+)/"), B26(r"远端独有 (\d+)/")
MS, NSHARED = B26(r"blob SHA 不符 (\d+)/"), B26(r"blob SHA 不符 \d+/(\d+)")
SM, NFILES = B26(r"MATCH (\d+)/"), B26(r"本片文件 (\d+) 个")
cA = re.search(r"正对照 A（远端 \S+ 的 sha 首位改 f）⇒ 不符 (\d+)/(\d+)", b26)
cB = re.search(r"正对照 B（远端删一条路径）⇒ 本地独有 (\d+)/(\d+)，远端独有 (\d+)/(\d+)", b26)
cD = re.search(r"正对照 D（远端删本片一条路径）⇒ 本片缺失 (\d+)/(\d+)，SHA 不符 (\d+)/", b26)
assert cA and cB and cD, "四个正对照读数取不齐"
assert int(cA.group(1)) >= 1 and cB.group(1) == "1" and cD.group(1) == "1", "正对照没有一条能红 ⇒ 尺子空转"
assert (DL, DR, MS) == ("0", "0", "0"), "落地差集／SHA 非 0：%s/%s/%s" % (DL, DR, MS)
NDEF26 = sum(1 for l in h26.splitlines() if re.match(r"^\S.*?\s*:", l) and not l.startswith("BODY_OFFSET="))
NFIELDDECL = None
for nd in ast.parse(txt(HDR26PY)).body:
    if isinstance(nd, ast.Assign) and getattr(nd.targets[0], "id", "") == "FIELDS":
        NFIELDDECL = len(nd.value.elts)
assert NFIELDDECL, "生成器 am_s26_hdr.py 里取不到 FIELDS 声明"
assert NDEF26 == NFIELDDECL, "回执顶格字段 %d ≠ 生成器 FIELDS 声明 %d" % (NDEF26, NFIELDDECL)
ruler_md5 = md5(RULE26)
ruler_copies = sorted(p for p in (os.path.relpath(x, REPO) for x in
                                  glob.glob(os.path.join(REPO, "docs/evidence/*/am_s7_verify*.py")))
                      if re.fullmatch(os.path.join("docs", "evidence", "[^/]+", r"am_s7_verify\d+\.py"), p)
                      and md5(os.path.join(REPO, p)) == ruler_md5)
assert os.path.relpath(RULE26, REPO) in ruler_copies, "分片 26 的尺子没被全树 glob 数到"
SRC26 = H26(r"SRC指纹 : ([0-9a-f]{32})")
assert SRC26 == FINGER, "分片 26 回执表头 SRC 指纹 %s ≠ 本轮现算 %s ⇒ 审计与归档面对的不是同一棵 src+tests 树" % (
    SRC26, FINGER)

# ============================================================ 现取：过期字节数声称（三面检测器）
DOCS = {rel: txt(os.path.join(REPO, rel)) for rel in FIVE}
STALE_HEAD = sum(wb_count(head_of(rel), STALE) for rel in FIVE)
STALE_PRE = sum(wb_count(DOCS[rel], STALE) for rel in FIVE)
DECOY = sum(DOCS[rel].count("0.944860") for rel in FIVE)
NAIVE = sum(DOCS[rel].count(STALE) for rel in FIVE)
assert NAIVE == STALE_PRE + DECOY, "诱饵闭合式破了：子串 %d ≠ 词边界 %d＋诱饵 %d ⇒ 口径不是我说的那两套" % (
    NAIVE, STALE_PRE, DECOY)
assert STALE_PRE >= 1, "改前工作树取不到过期值 ⇒ 本轮没有要更正的东西，检测器是空转"

REPL = [
    ("docs/开发日志.md", r"表头偏移 \d+ ⇒", "表头偏移 %s ⇒" % TRUEV, 1),
    ("docs/开发日志.md", r"\d+＋9400=\d+／3411＋2873=6284 现取",
     "%s＋%s=%s／%s＋%s=%s 现取" % (HDRSZ, RAWSZ, LOGSZ, CHDRSZ, CRAWSZ, CLOGSZ), 1),
    ("docs/项目开发总览.md", r"表头偏移 \d+）", "表头偏移 %s）" % TRUEV, 1),
]
NEW = dict(DOCS)
for rel, pat, rep, want in REPL:
    got = re.findall(pat, NEW[rel], re.M)
    assert len(got) == want, "%s：更正锚点命中 %d 处（须 %d）⇒ 拒绝盲写：%r" % (rel, len(got), want, pat)
    NEW[rel] = re.sub(pat, lambda m: rep, NEW[rel], count=want, flags=re.M)

# ============================================================ 待追加的各腿
SEC_DEV = """
**(l) 分片 27 落地腿＝把上一轮登记的"手工表头债"折成一台**溯源审计仪器**（不重写已归档字节，改判"表头里每个读数必须有出处"）；三处 +0**


- **债怎么折**：§26.29 (i) 写明本轮两张 A3 面表头**没**走 `am_s26_hdr.py` 那把生成器（片号字段绑定 HEAD 提交说明，而当时本片尚未提交）⇒ 少了"字段只从命令现取"这层保护。本轮**不**重写表头字节（重写会让 (k) 里那些 md5／字节数声称再次过期），改为一件 `docs/evidence/2026-10-10/am_a3_hdr_audit.py`：把两面表头逐句拆开，每个读数现查五类出处（C 命令读数／D 器件常量／B 本面正文／X 跨面正文且引用句须自带出处词／S 结构式重算），查不到即 rc=3。现数＝主面表头数字 @@AUN@@ 个、控件面 @@CTN@@ 个，两面**无出处 0／跨面无归属 0／无出处 hex 0**；`@@AUVERD@@`（rc=0、stderr 0 字节，原始 stdout 归档 `_raw_a3_hdr_audit.out`）。本写手**复跑**该件并要求判读行逐条重现（断言在写手里）⇒ 这些数不是那次运行的化石。**复现口径本轮收窄了一次**（拦在写盘前＝+0）：普查行里的**类别分布**（B／C／D／S／X 各几个）随证据目录内容移动——出处池就是整个目录，本轮每新写一件，同一个数就可能从 B 类改判 C 类；把分布当"复现"来断言会造出一条对不齐的闸门，故只钉**总数＋三类红旗**（它们由表头正文钉住），类别分布只进本写手的 stdout、**不入档**。

- **当场抓到一处真错（跨面引用被四舍五入）**：主面表头那句「写错就差 `1.33e-02`」与正文读数 `1.331e-02`（控件面 K2 行原样）不一致 ⇒ 表头改一字（@@HDRSZ@@ 字节，原 4860），回执重建＝@@HDRSZ@@＋@@RAWSZ@@＝@@LOGSZ@@（`cat hdr 原始件 | cmp 回执` 两面 rc 均 0），且正文原始件 md5 仍＝@@RAWMD@@（逐字节未动）⇒ 变的只有表头那句。

- **仪器自己的空转闸门（+0）＝极性翻转不是扰动**：审计件首版有个 `HDR_INJECT` 环境变量，它只把判读从"须 0 红"翻成"须 >0 红"、**不注入任何坏输入** ⇒ 一条能把自己说成绿的对照不是对照，已删（旧命令带这个变量如今直接拒绝运行而不是静默放行）。红路改由正对照面 `am_a3_hdr_audit_controls.py` 真造：十条**输入**扰动（表头的 `zctl_` 副本；回执按"新表头＋原正文"重建，好让 `cat` 复现式不跟着一起红），判据＝rc≠0 **且**命中指定闸门句，未扰动基线先 rc=@@CTLBRC@@ 且印全过判读行 ⇒ 现数 **@@CTLRED@@/@@CTLN@@ 条各自命中自己的闸门**（`@@CTLVERD@@`）。副本一律落**本件自己的目录**（逃到 /tmp ⇒ 生成器反推 REPO=/、git 当场失败＝环境错，§26.28 已为此记过一条）。

- **配套的一处池加固（仪器不许给自己开后门）**：`zctl_*` 副本原先会被 `D/*` 的 md5 池数进去 ⇒ 副本自己的 md5 能把表头里无出处的读数洗白，现从 C／HEX 两池一并排除；出处匹配一律**精确**命中（子串命中＝给十六进制串里的数字段放行）。

- **过期值检测器本轮的宿主换成了"字节数"**：改表头使 §26.29 (i) 与总览那条里的「表头偏移 @@STALE@@」「@@STALE@@＋9400＝@@LOGOLD@@」再次作废 ⇒ 就地更正 **@@NREPL@@ 处**（开发日志 2／总览 1），新值全部由 `stat` 现取＝@@TRUEV@@／@@HDRSZ@@＋@@RAWSZ@@＝@@LOGSZ@@。分三面读：历史面（五份文档的 `git show HEAD:` 版）词边界命中 **@@STALEH@@** 处——§26.29 与那条总览条目**尚未提交** ⇒ 这些过期值从没落进档案，本条按分界计 **+0** 而不是 +1；改前工作树 **@@STALEP@@** 处；**改后全文**面（＝落盘那一份，含本段自己引用的旧值）按"未自证"判 **@@UNMARKED@@** 处、
自证标本 **@@SPECWORK@@** 处（每处引用旧值的**同段**必须含真值 @@TRUEV@@；这一面的读数依赖本写手将要写下的句子
⇒ 迭代到不动点才进模板，见本件 `detector`）。诱饵闭合式＝子串命中 @@NAIVE@@ ＝ 词边界 @@STALEP@@ ＋ 形如 `0.944860` 的诱饵 @@DECOY@@ ⇒ 词边界这条口径不是装饰。正对照＝把真值从各段抹掉再喂同一个检测器，必须逐处变红（现数 **@@SPECCTRL@@/@@SPECWORK@@**）。

- **分片 26 回执入库**（读数**引自回执** `docs/evidence/2026-10-10/am_s7_verify26.log` @@R26SIZE@@ 字节、md5＝@@R26MD@@、正文区起点 BODY_OFFSET＝@@BOFF26@@）：远端 main＝`@@HEAD26@@`＝本地 HEAD（父 `@@PARENT26@@`）；`truncated=False`、tree 条目 @@ENTRIES@@＝blob @@NB@@＋tree @@NTREE@@；闭合式 **远端 @@NB@@ == 父提交 @@NP@@ ＋ 新增(A) @@NA@@ ＝ @@EXPECT@@ -> PASS**；双向路径差集 **@@DL@@/@@NB@@** 与 **@@DR@@/@@NB@@**；全库共有项 blob SHA 不符 **@@MS@@/@@NSHARED@@**；本片单列复核 **@@SM@@/@@NFILES@@ MATCH**（A @@NA@@／M @@NMF@@）；`@@VERD26@@`。四个正对照同一条运行里都能红：A 远端一条 sha 首位改 f ⇒ 不符 @@CA@@/@@CBA@@；B 删一条路径 ⇒ 本地独有 @@CB1@@/@@CDB1@@、远端独有 @@CB2@@/@@CDB2@@；C 闭合式右边 +1 ⇒ 判 FAIL；D 删**本片**一条 ⇒ 缺失 @@CD1@@/@@NFILES@@。回执自身复核：`cmp -i @@BOFF26@@:0` rc＝@@CMP0@@、左 +1＝@@CMP1@@、右 +1＝@@CMP2@@；表头顶格字段 **@@NDEF26@@** 行＝生成器 `FIELDS` 声明 **@@NFIELDDECL26@@**（不写死，写死＝把两文件的耦合藏进常数）；同一把尺子（md5 前 12＝@@MDRULE@@）**全树**现算已有 **@@NRULE26@@** 份逐字节副本。⇒ 本片（分片 27）自己的落地证明要到下一轮才存在，本段不预先声称它已被引用。

- **树与钉住**：本写手落笔（@@NOW@@，`date` 现取）现算 dirty total=@@DIRTYT@@（其中 docs/evidence/ **@@DIRTYE@@**、src+tests **@@DIRTYS@@**＝0）⇒ 生产树未动；SRC 指纹现算 `@@FINGER@@` **等于**分片 26 回执表头那一条（`@@SRC26@@`）⇒ 本轮的审计、更正与归档面对的还是同一棵 src+tests 树。卡上 compute 进程 **@@NV@@** 个、我的解释器（按 `comm` 起头 python 且 cmdline 含 `/tmp/amvenv`、排除自身）**@@NVMINE@@/@@NV@@**（非 0 即拒绝写盘）；全轮 CPU 钉住，本轮不产生任何性能数字。

- **排期**：A3 阶梯 `--stage smoke`（dx=r_b，按 J6 只验接线、不产出形态数字）→ `--stage pilot --tracks 1,2,3 --arms both` → `--stage full --measured-per-track`（ΣN=1..18 比 @@BUDGET@@ s，N3⑤）；**#27 腿②（缺省 tuned→hk）＋#33 的 15 位点系数（含 σ_z）仍只能对 A3 的 1→18 外部趋势打分**；#40→#39；#41 待自己一轮；#29／#30 等用户裁决。红线不变：A0 断言一字不放宽、#27 不换指标、无新增 skip/xfail。自纠计数：§26.27 仍 **@@PREVCNT@@** 条（本轮三处都拦在写盘前／未提交 ⇒ +0）。
"""

SEC_PLAN = """
### §W 补记（@@NOW@@ 落笔，`date` 现取）＝分片 27 落地腿：手工表头的债折成一台**溯源审计件**，并就地更正三处随改表头过期的字节数


- **审计件**＝`docs/evidence/2026-10-10/am_a3_hdr_audit.py`：两面表头逐句现查五类出处（C 命令读数／D 器件常量／B 本面正文／X 跨面正文＋出处词／S 结构式重算），无出处即 rc=3。现数：主面表头数字 @@AUN@@ 个、控件面 @@CTN@@ 个，两面**无出处／跨面无归属／无出处 hex 都是 0**；`@@AUVERD@@`。它当场抓到主面表头把控件面 K2 的 `1.331e-02` 四舍五入成 `1.33e-02` ⇒ 表头改一字（@@HDRSZ@@ 字节）、回执重建（@@HDRSZ@@＋@@RAWSZ@@＝@@LOGSZ@@，`cat|cmp` 两面 rc=0），正文原始件 md5 未动（@@RAWMD@@）。

- **仪器纪律两条**：① **极性翻转不是扰动**——首版的 `HDR_INJECT` 只翻判读极性、不注入坏输入，已删；红路交给 `am_a3_hdr_audit_controls.py` 的十条**输入**扰动（基线先 rc=@@CTLBRC@@，现数 **@@CTLRED@@/@@CTLN@@ 各命中自己的闸门句**，`@@CTLVERD@@`）；② 池里不许出现仪器自己的脚手架——`zctl_*` 副本从 C／HEX 池排除，出处一律精确命中。

- **过期值检测器分三面**：历史面（HEAD 版五份文档）**@@STALEH@@** 处、改前工作树 **@@STALEP@@** 处、改后全文未自证 **@@UNMARKED@@** 处；诱饵闭合式 子串 @@NAIVE@@＝词边界 @@STALEP@@＋诱饵 @@DECOY@@；正对照（抹掉真值）红 **@@SPECCTRL@@/@@SPECWORK@@**。因 §26.29／总览那条**尚未提交**，本轮计 **自纠 +0**（§26.27 仍 **@@PREVCNT@@** 条）。

- **分片 26 回执入库**：远端 main＝`@@HEAD26@@`、tree 条目 @@ENTRIES@@＝blob **@@NB@@**＋tree @@NTREE@@、闭合式 **@@NB@@ == @@NP@@＋@@NA@@＝@@EXPECT@@**、双向差集 **@@DL@@/@@NB@@** 与 **@@DR@@/@@NB@@**、全库 SHA 不符 **@@MS@@/@@NSHARED@@**、本片 **@@SM@@/@@NFILES@@ MATCH**、`cmp -i @@BOFF26@@:0` rc＝@@CMP0@@＋两侧 +1 各红、尺子全树 **@@NRULE26@@** 份。

- **排期与红线**：`--stage smoke`→`pilot --arms both`→`full --measured-per-track`（ΣN=1..18 比 @@BUDGET@@ s）；**#27 腿②＋#33 与 #10/A3 同批**（外部 18-track 趋势是唯一打分面）；#40→#39；#41 待自己一轮；#29／#30 等裁决。A0 断言不放宽、#27 不换指标、无新增 skip/xfail、性能数字只在 GPU 窗口测。本轮生产树 `src/`＋`tests/` 改动 **@@DIRTYS@@** 件，落笔时卡上 compute 进程 **@@NV@@** 个、我的 **@@NVMINE@@/@@NV@@**。
"""

SEC_OV = """- **@@DATE_SHORT@@（分片 27 落地腿＋#47 手工表头的债折成审计仪器，CPU 钉住、不占 GPU；生产树 0 改动）＝表头里每个数字**必须有出处**，出处由命令现查而不是由我背书**。
  新仪器 `docs/evidence/2026-10-10/am_a3_hdr_audit.py`（五类出处 C／D／B／X＋出处词／S 结构式重算，无出处即 rc=3）
  ＋正对照面 `am_a3_hdr_audit_controls.py`：主面表头数字 **@@AUN@@** 个、控件面 **@@CTN@@** 个，
  无出处／跨面无归属／无出处 hex **全 0**，`@@AUVERD@@`；十条输入扰动 **@@CTLRED@@/@@CTLN@@** 各命中自己的闸门句（基线先 rc=@@CTLBRC@@）。
  当场抓到一处真错＝主面表头把控件面 K2 的 `1.331e-02` 四舍五入成 `1.33e-02` ⇒ 表头改一字（**@@HDRSZ@@** B，原 4860）、
  回执重建（@@HDRSZ@@＋@@RAWSZ@@＝@@LOGSZ@@，`cat|cmp` 两面 rc=0），正文原始件 md5 未动＝@@RAWMD@@。
  两条仪器纪律：**极性翻转不是扰动**（`HDR_INJECT` 只翻判读极性、已删）；池里不许出现仪器自己的脚手架
  （`zctl_*` 副本从 C／HEX 池排除、出处一律精确命中）。过期值检测器分三面（历史面 @@STALEH@@／改前 @@STALEP@@／
  改后全文未自证 @@UNMARKED@@）＋诱饵闭合式（子串 @@NAIVE@@＝词边界 @@STALEP@@＋诱饵 @@DECOY@@）＋抹掉真值的正对照
  （红 @@SPECCTRL@@/@@SPECWORK@@）⇒ 就地更正 @@NREPL@@ 处。分片 26 回执 `am_s7_verify26.log`：远端 main＝`@@HEAD26@@`、
  blob **@@NB@@**、闭合式 **@@NB@@ == @@NP@@＋@@NA@@＝@@EXPECT@@**、双向差集 **@@DL@@/@@NB@@**与**@@DR@@/@@NB@@**、
  SHA 不符 **@@MS@@/@@NSHARED@@**、本片 **@@SM@@/@@NFILES@@ MATCH**、尺子全树 **@@NRULE26@@** 份。
  自纠 §26.27 仍 **@@PREVCNT@@** 条（本轮三处都拦在未提交状态 ⇒ +0）。GPU 现取：**@@NV@@** 个、我的 **@@NVMINE@@/@@NV@@**。
"""

SEC_DAY = """
### @@TIME_SHORT@@ 分片 27 落地腿＝表头数值溯源审计件＋控件面＋就地更正过期字节数（CPU 钉住，生产树 0 改动）
把 §26.29 (i) 登记的"两张 A3 面表头没走生成器"折成一台**审计件**（`am_a3_hdr_audit.py`：五类出处 C／D／B／X＋出处词／S 重算，
无出处 rc=3）而不是重写已归档字节。现数：主面表头数字 @@AUN@@ 个／控件面 @@CTN@@ 个，两面**无出处／跨面无归属／无出处 hex 全 0**，
`@@AUVERD@@`（rc=0、stderr 0 字节；本写手复跑并要求判读行逐条重现）。抓到一处真错：主面表头把控件面 K2 的 `1.331e-02`
写成四舍五入的 `1.33e-02` ⇒ 表头改一字（@@HDRSZ@@ B）、回执重建（@@HDRSZ@@＋@@RAWSZ@@＝@@LOGSZ@@，`cat|cmp` 两面 rc=0），
正文原始件 md5 未动（@@RAWMD@@）。仪器纪律两条：**极性翻转不是扰动**（首版 `HDR_INJECT` 只翻判读极性、不注入坏输入 ⇒ 已删，
旧命令现在直接拒绝运行）；红路交给 `am_a3_hdr_audit_controls.py` 十条**输入**扰动（表头 `zctl_` 副本＋回执按"新表头＋原正文"
重建，使 `cat` 式不连带变红），基线先 rc=@@CTLBRC@@、现数 **@@CTLRED@@/@@CTLN@@** 各命中自己的闸门句；`zctl_*` 从 C／HEX 池排除
＝仪器不许拿自己的脚手架给自己洗白。过期值检测器本轮宿主是"字节数"：三面读数 历史面 @@STALEH@@／改前 @@STALEP@@／
改后全文未自证 @@UNMARKED@@，诱饵闭合式 子串 @@NAIVE@@＝词边界 @@STALEP@@＋诱饵 @@DECOY@@，抹掉真值后正对照红 **@@SPECCTRL@@/@@SPECWORK@@**
⇒ 就地更正 @@NREPL@@ 处（开发日志 2／总览 1）。分片 26 回执入库：远端 main＝`@@HEAD26@@`、blob **@@NB@@**、闭合式
**@@NB@@ == @@NP@@＋@@NA@@＝@@EXPECT@@**、差集 **@@DL@@/@@NB@@**与**@@DR@@/@@NB@@**、SHA 不符 **@@MS@@/@@NSHARED@@**、
本片 **@@SM@@/@@NFILES@@ MATCH**、`cmp -i @@BOFF26@@:0` rc＝@@CMP0@@＋两侧 +1 各红、同一把尺子全树 **@@NRULE26@@** 份。
树与钉住：dirty total=@@DIRTYT@@（evidence @@DIRTYE@@／src+tests **@@DIRTYS@@**），SRC 指纹现算 `@@FINGER@@`＝回执表头那一条
`@@SRC26@@`；卡上 compute 进程 **@@NV@@** 个、我的 **@@NVMINE@@/@@NV@@**；全轮 CPU 钉住，无性能数字。自纠 §26.27 仍
**@@PREVCNT@@** 条（本轮三处都拦在未提交状态 ⇒ +0）。排期：`--stage smoke`→`--stage pilot --tracks 1,2,3 --arms both`→
`--stage full --measured-per-track`（ΣN=1..18 比 @@BUDGET@@ s）；**#27 腿②＋#33 与 #10/A3 同批**；#40→#39；#41 待自己一轮；
#29／#30 等裁决。红线不变。
"""

SEC_MEM = """

- **@@DATE_SHORT@@（分片 27 落地腿＝表头读数改由**审计件**现查出处；生产树 0 改动）**：#47 那两张手工表头的债
  （§26.29 (i)）不靠重写偿还，靠 `docs/evidence/2026-10-10/am_a3_hdr_audit.py` 把每个读数现查五类出处
  （C 命令读数／D 器件常量／B 本面正文／X 跨面正文须自带出处词／S 结构式重算），无出处⇒rc=3。现数主面表头数字
  @@AUN@@ 个、控件面 @@CTN@@ 个，三类红旗全 0，`@@AUVERD@@`；**当场抓到主面表头把控件面 K2 的 `1.331e-02` 四舍五入成
  `1.33e-02`**⇒ 表头改一字（@@HDRSZ@@ B）＋回执重建（@@HDRSZ@@＋@@RAWSZ@@＝@@LOGSZ@@，`cat|cmp` rc=0），正文 md5 未动。
  **两条新口径入档**：① **极性翻转不是扰动**（`HDR_INJECT` 只把"须 0 红"翻成"须 >0 红"、不注入坏输入＝一条能把自己
  说成绿的对照，已删）；② 出处池里不许出现仪器自己的脚手架（`zctl_*` 副本从 C／HEX 池排除、一律精确命中）。
  正对照面十条输入扰动 **@@CTLRED@@/@@CTLN@@** 各命中自己的闸门句（未扰动基线先 rc=@@CTLBRC@@）；改表头使「表头偏移 4860」
  一类字数过期⇒ 过期值检测器分三面（历史面 @@STALEH@@／改前 @@STALEP@@／改后全文未自证 @@UNMARKED@@）＋诱饵闭合式
  （子串 @@NAIVE@@＝词边界 @@STALEP@@＋诱饵 @@DECOY@@）＋抹掉真值正对照（红 @@SPECCTRL@@/@@SPECWORK@@），就地更正 @@NREPL@@ 处。
  分片 26 回执入库：远端 main `@@HEAD26@@`、blob **@@NB@@**、闭合式 **@@NB@@==@@NP@@＋@@NA@@＝@@EXPECT@@**、双向差集
  **@@DL@@/@@NB@@**与**@@DR@@/@@NB@@**、SHA 不符 **@@MS@@/@@NSHARED@@**、本片 **@@SM@@/@@NFILES@@ MATCH**、尺子全树 **@@NRULE26@@** 份。
  自纠 §26.27 仍 **@@PREVCNT@@** 条（本轮三处都拦在未提交状态 ⇒ +0）。排期：A3 `smoke`→`pilot`→`full`（预算 @@BUDGET@@ s），
  **#27 腿②＋#33 与 #10/A3 同批**；#40→#39；#29／#30 等裁决。GPU 现取（@@NOW@@）：**@@NV@@** 个、我的 **@@NVMINE@@/@@NV@@**。
"""

BUDGET = "14400"
_m = re.search(r"^BUDGET_S = ([0-9. *]+)$", txt(os.path.join(D, "am_a3_driver.py")), re.M)
assert _m, "驱动器里取不到 BUDGET_S 的定义式"
BUDGET = str(int(eval(_m.group(1), {"__builtins__": {}}, {})))

pairs = re.findall(r"\*\*(\d+)(?:⇒(\d+))?\*\* 条", DOCS["docs/项目开发总览.md"])
assert pairs, "总览里取不到上一条自纠计数 ⇒ 拒绝手填"
PREVCNT = int(pairs[-1][1] or pairs[-1][0])
assert PREVCNT > 0, "自纠计数取回 0 ⇒ 正则口径变了"
OVN = len(re.findall(r"^- \*\*20\d\d-\d\d-\d\d（", DOCS["docs/项目开发总览.md"], re.M))

_m2 = re.search(r"\d+＋9400=(\d+)／", DOCS["docs/开发日志.md"])
assert _m2, "开发日志里取不到那句旧的字节能合式"
LOGOLD = _m2.group(1)

V = {
    "NOW": sh("date", "+%F %T %z"), "TIME_SHORT": sh("date", "+%H:%M"),
    "DATE_SHORT": sh("date", "+%Y-%m-%d"),
    "HEAD26": H26(r"head=([0-9a-f]{7,40})", 1)[:7], "PARENT26": H26(r"parent=([0-9a-f]{7,40})", 1)[:7],
    "ENTRIES": B26(r"tree 条目 (\d+) = "), "NTREE": B26(r"blob \d+ \+ tree (\d+)"),
    "NB": str(NB), "NP": str(NP), "NA": str(NA), "NMF": str(NMF), "EXPECT": str(EXPECT),
    "DL": DL, "DR": DR, "MS": MS, "NSHARED": NSHARED, "SM": SM, "NFILES": NFILES,
    "CA": cA.group(1), "CBA": cA.group(2), "CB1": cB.group(1), "CDB1": cB.group(2),
    "CB2": cB.group(3), "CDB2": cB.group(4), "CD1": cD.group(1),
    "VERD26": VERD26, "BOFF26": str(boff), "CMP0": str(CMP[0]), "CMP1": str(CMP[1]),
    "CMP2": str(CMP[2]), "NDEF26": str(NDEF26), "NFIELDDECL26": str(NFIELDDECL),
    "R26SIZE": size(REC26), "R26MD": md5(REC26)[:12], "MDRULE": ruler_md5[:12],
    "NRULE26": str(len(ruler_copies)), "SRC26": SRC26,
    "AUN": AUN, "CTN": CTN,
    "AUVERD": AUVERD, "CTLN": str(CTLN), "CTLRED": str(CTLRED), "CTLBRC": CTLB,
    "CTLVERD": CTLVERD,
    "HDRSZ": HDRSZ, "RAWSZ": RAWSZ, "LOGSZ": LOGSZ, "RAWMD": md5(RAW_MAIN)[:12],
    "TRUEV": TRUEV, "STALE": STALE, "LOGOLD": LOGOLD, "NREPL": str(len(REPL)),
    "STALEH": str(STALE_HEAD), "STALEP": str(STALE_PRE), "DECOY": str(DECOY),
    "NAIVE": str(NAIVE),
    "DIRTYT": str(DIRTY_T), "DIRTYE": str(DIRTY_E), "DIRTYS": str(DIRTY_S),
    "FINGER": FINGER, "NV": str(len(nv)), "NVMINE": str(len(nv_mine)),
    "PREVCNT": str(PREVCNT), "BUDGET": BUDGET,
}


def specimen_ok(text):
    """引用旧值的每一处都必须**同段**含真值（标记取自读数，不取自措辞＝§26.27 (k) 的教训）。"""
    ok = bad = 0
    for para in re.split(r"\n\s*\n", text):
        n = wb_count(para, STALE)
        if n:
            ok, bad = (ok + n, bad) if TRUEV in para else (ok, bad + n)
    return ok, bad


def detector(texts):
    """三面里唯一需要迭代的**全文面**：未自证／自证标本／抹掉真值后红。texts 必须是最终落盘那一份。"""
    sp = [specimen_ok(s) for s in texts]
    return (sum(b for _, b in sp), sum(o for o, _ in sp),
            sum(specimen_ok(re.sub(re.escape(TRUEV), "X", s))[1] for s in texts))

GLUE = re.compile(r"。- \*\*2026-")
DEV_LAST = [l for l in DOCS["docs/开发日志.md"].splitlines() if l.strip()][-1]
_ov0 = DOCS["docs/项目开发总览.md"]
# 时间线**不在文件末尾**：它后面还有 `---`＋`## 7. 文档索引`。锚点＝该边界之前最后一条非空行。
_mbound = re.search(r"\n---\n+## 7\. 文档索引", _ov0)
assert _mbound, "总览里取不到 `---`＋`## 7. 文档索引` 边界 ⇒ 结构变了，mid 插入没有落点"
anchor_ov = [l for l in _ov0[:_mbound.start()].splitlines() if l.strip()][-1]
assert anchor_ov.lstrip().startswith(("- **", "**")), \
    "总览时间线末行不是条目行：%r" % anchor_ov[:60]
i_ov = _ov0.index(anchor_ov) + len(anchor_ov)
assert _ov0[i_ov:].startswith("\n\n---"), "总览末条之后不是 空行＋--- ⇒ 结构变了"

LEGS = [
    ("开发日志 (l)", "docs/开发日志.md", "**(l) 分片 27 落地腿", SEC_DEV, "eof", DEV_LAST),
    ("计划文档 §W 补记", "docs/项目评估与下一步计划_2026-10-06.md", "### §W 补记", SEC_PLAN, "eof",
     [l for l in DOCS["docs/项目评估与下一步计划_2026-10-06.md"].splitlines() if l.strip()][-1]),
    ("总览", "docs/项目开发总览.md", "分片 27 落地腿", SEC_OV, "mid", anchor_ov),
    ("当日", ".workbuddy/memory/2026-10-10.md", "分片 27 落地腿＝表头数值溯源审计件", SEC_DAY, "eof",
     [l for l in DOCS[".workbuddy/memory/2026-10-10.md"].splitlines() if l.strip()][-1]),
    ("MEMORY", ".workbuddy/memory/MEMORY.md", "分片 27 落地腿＝表头读数改由", SEC_MEM, "eof",
     [l for l in DOCS[".workbuddy/memory/MEMORY.md"].splitlines() if l.strip()][-1]),
]

ALL = SEC_DEV + SEC_PLAN + SEC_OV + SEC_DAY + SEC_MEM
tpl = set(re.findall(r"@@([A-Za-z0-9_]+)@@", ALL))
V["UNMARKED"] = V["SPECWORK"] = V["SPECCTRL"] = "0"     # 种子值，下面迭代到不动点
missing = sorted(tpl - set(V))
assert not missing, "模板有未取值的字段：" + str(missing)
unused = sorted(set(V) - tpl)
assert not unused, "算了读数却没进任何模板（口径漂了）：" + str(unused)

NEW0 = dict(NEW)          # 就地更正之后、追加之前的那一版（追加会往里面再塞旧值引用）


def blank_before(text, start):
    """块首之前必须是**空行**：单个换行＝把新条目粘在上一行尾部＝§26.29 (k) ① 的那次事故。"""
    return text[start - 1] == "\n" and text[start - 2] == "\n"


def render(Vloc):
    fin, rep = dict(NEW0), []
    for name, rel, marker, sec, mode, anchor in LEGS:
        cur = fin[rel]
        if marker in cur:
            rep.append("SKIP " + rel + "：标记已存在 ⇒ 该腿已落，不重复追加")
            continue
        n = cur.count(anchor)
        assert n == 1, rel + "：锚点出现 " + str(n) + " 次（须 1）⇒ 拒绝写盘：" + repr(anchor[:44])
        block = sec.strip("\n")
        for k, v in Vloc.items():
            block = block.replace("@@" + k + "@@", v)
        if "@@" in block or "＠" in block:
            print("【残留PLACEHOLDER】", rel)
            for l in block.splitlines():
                if "@@" in l or "＠" in l:
                    print("  残留行｜", l)
            raise SystemExit(3)
        if mode == "eof":
            assert cur.rstrip().endswith(anchor.rstrip()), rel + "：锚点不在文件末尾 ⇒ 结构变了，拒绝盲写"
            pre = cur.rstrip("\n")
            new, start = pre + "\n\n" + block + "\n", len(pre) + 2
            ctrl, cstart = pre + "\n" + block + "\n", len(pre) + 1
        else:
            i = cur.index(anchor) + len(anchor)
            new, start = cur[:i] + "\n\n" + block + cur[i:], i + 2
            ctrl, cstart = cur[:i] + "\n" + block + cur[i:], i + 1
            assert new[start:].startswith(block + "\n\n"), rel + "：块体之后不是空行＋--- ⇒ 总览的边界变了"
        assert blank_before(new, start), rel + "：块首之前不是空行 ⇒ 会重演粘连"
        assert not blank_before(ctrl, cstart), rel + "：正对照失效——少写一个换行的插入没被同一条断言抓住"
        assert len(GLUE.findall(new)) == 0, rel + "：本腿写完仍有粘连条目 ⇒ 拒绝落盘"
        assert specimen_ok(new)[1] == 0, rel + "：本腿写入了未自证的旧字节数"
        fin[rel] = new
        rep.append("%s %s：%d → %d 行（＋%d），标记 1 次｜更正腿另计 %d 处" % (
            "WRITE" if WRITE else "PRECHECK", rel, cur.count("\n"), new.count("\n"),
            new.count("\n") - cur.count("\n"), len([r for r in REPL if r[0] == rel])))
    return fin, rep


Vloc, FIN, report = dict(V), None, None
for _it in range(6):
    FIN, report = render(Vloc)
    UNMARKED, SPECWORK, SPECCTRL = detector(FIN.values())
    if (Vloc["UNMARKED"], Vloc["SPECWORK"], Vloc["SPECCTRL"]) == (
            str(UNMARKED), str(SPECWORK), str(SPECCTRL)):
        break
    Vloc.update({"UNMARKED": str(UNMARKED), "SPECWORK": str(SPECWORK), "SPECCTRL": str(SPECCTRL)})
else:
    raise SystemExit("全文面读数不收敛（模板里的标本数依赖它自己写下的句子）⇒ 拒绝归档")
V = Vloc
assert UNMARKED == 0, "有 %d 处引用旧字节数却没在同一句里带上真值 ⇒ 是传染不是标本，拒绝归档" % UNMARKED
assert SPECCTRL == SPECWORK >= 1, "抹掉真值的正对照不齐（红 %d／标本 %d）⇒ 这条检测器空转" % (
    SPECCTRL, SPECWORK)
OVF = FIN["docs/项目开发总览.md"]
assert len(re.findall(r"^- \*\*20\d\d-\d\d-\d\d（", OVF, re.M)) == OVN + 1, \
    "总览顶层条目 %d → %d 不是恰好 +1 ⇒ mid 插入把结构改了" % (
        OVN, len(re.findall(r"^- \*\*20\d\d-\d\d-\d\d（", OVF, re.M)))
assert OVF.count("\n\n---\n") == DOCS["docs/项目开发总览.md"].count("\n\n---\n"), "总览的 --- 边界数变了"

if WRITE:
    for rel, new in FIN.items():
        if new != DOCS[rel]:
            open(os.path.join(REPO, rel), "w", encoding="utf-8").write(new)
            assert txt(os.path.join(REPO, rel)) == new, rel + "：写后回读 ≠ 预期全文"
    for rel in FIVE:
        assert wb_count(txt(os.path.join(REPO, rel)), TRUEV) >= 1, rel + "：写后取不到真字节数"
    after = detector([txt(os.path.join(REPO, rel)) for rel in FIVE])
    assert after == (UNMARKED, SPECWORK, SPECCTRL), \
        "写后全文的三面读数 %s ≠ 归档句里声称的 %s/%s/%s ⇒ 落盘的文档不是被检测的那一份" % (
            after, UNMARKED, SPECWORK, SPECCTRL)

print("\n".join(report))
print("审计件复跑：%s｜主面表头数字 %s 个（%s）／控件面 %s 个｜正对照 %s/%s 条命中，基线 rc=%s"
      % (AUVERD, V["AUN"], AUCLS, V["CTN"], CTLRED, CTLN, CTLB))
print("字节闭合：主面 %s＋%s=%s｜控件面 %s＋%s=%s｜更正锚点 %d 处（旧值 词边界 历史面 %s／改前 %s／诱饵 %s／子串 %s）"
      % (HDRSZ, RAWSZ, LOGSZ, CHDRSZ, CRAWSZ, CLOGSZ, len(REPL),
         STALE_HEAD, STALE_PRE, DECOY, NAIVE))
print("过期值检测器三面：历史面 %s／改前工作树 %s／改后全文未自证 %s｜标本 %s／抹掉真值后红 %s｜自纠上一条 %s（本轮 +0）"
      % (STALE_HEAD, STALE_PRE, UNMARKED, SPECWORK, SPECCTRL, PREVCNT))
print("分片 26：main %s｜blob %s 闭合 %s==%s+%s=%s｜差集 %s/%s、%s/%s｜SHA 不符 %s/%s｜本片 %s/%s｜BOFF %s cmp %s/%s/%s｜顶格 %s＝FIELDS %s｜尺子 %s 份"
      % (V["HEAD26"], NB, NB, NP, NA, EXPECT, DL, NB, DR, NB, MS, NSHARED, SM, NFILES,
         boff, CMP[0], CMP[1], CMP[2], NDEF26, NFIELDDECL, V["NRULE26"]))
print("树：HEAD %s（父 %s）｜dirty total %s／evidence %s／src+tests %s｜SRC 指纹 %s＝回执那一条 %s｜GPU %s 个、我的 %s/%s"
      % (HEAD, PARENTS, DIRTY_T, DIRTY_E, DIRTY_S, FINGER, SRC26, len(nv), len(nv_mine), len(nv)))
if not WRITE:
    print("（预检模式：加 --write 才落盘。）")
