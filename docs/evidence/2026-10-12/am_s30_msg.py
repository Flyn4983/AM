#!/usr/bin/env python3
r"""S7 分片 30 提交信息生成器：信息里的每个数字都从命令／档案现取，不许手敲。

本轮＝**纯归档腿**（`git diff --cached --name-only -- src tests` 现数 0 件）⇒ 写的是三件可核查的事：
① §26.31 登记的形制约束由「另起 §26.32」满足（节内唯一性由**暂存 blob** 现验）；
② 分片 29 的归档半腿整条链入库（读数全部从回执与原始件正则现取，且**复跑** cmp 三件套）；
③ 本轮新学的口径（`LC_ALL=C` 与 SRC 指纹耦合）由两条 locale 各现算一次、并与 §26.32 登记的两个值互证。

片号与"上一片回执"一律**现取并交叉印证**（§26.27 (u)、§26.30、§26.31 都记过机械克隆把上一片的故事安在本片头上）：
`git log -1 --format=%s` 给出 HEAD 的片号 ⇒ 本片＝它＋1；全树 glob 回执档案给出最大号角本 ⇒ 必须＝HEAD 片号；
上一片回执的**目录**由那一份回执自己被 glob 命中而定（本轮跨日期：它在 10-10，本件在 10-12 ⇒ 日期目录不许硬写）。

八条口径（沿用＋本轮新增）：
① 正文一律 **PLACEHOLDER＋str.replace**，不用 f-string（f-string 会把字面量里的 `{32}` 当替换字段
   **静默吃掉**，§26.27 (q) 记过一条自纠）。生成后**双向**查：模板里没取值的字段拒绝，取值表里模板
   没用到的读数也拒绝（没用到的读数＝下一条等着的谎）。
② 引命令**原文**并把三个数闭合到 `--numstat` 逐行现算（git 对 1 用单数「1 deletion(-)」，按复数形状
   取数会把它静默折成 0）。
③ 每个 0 与它的分母同印；每条闸门都要在 `am_s30_msg_controls.py` 里被**坏输入**打红过（只看 rc≠0
   的控制＝空转，本件按"抛出的那一行含闸门句"判）。
④ **单值取数必须按形状类并要求恰好一次**：`(\S+)` 从中文句子里会把「；」和后面的词一起吞掉，
   `search` 在多处命中时会静默拿到第一条 ⇒ 本件所有取数走 `one()`／`grab()`（findall 后断言 len==1）。
⑤ **局部仪器不许当常备尺子**：`am_s29_archive_check.py` 的「上一节引用数」闸是按**它自己那一轮**的
   最后一个节写的 ⇒ §26.32 一开它必然红。本件现跑一次并要求它红在那句，且**红的末行与已归档的标本
   逐字节相同**（两出处：本轮实跑 ＝ 档案里那次），作为"这条闸是活的"的正对照。
⑥ 本轮新增：**角色桶与 §26.32 登记的分组闭合**。落笔那份把 24 件分成 8 组并写了等式
   `5＋3＋3＋4＋3＋2＋3＋1＝24`；本件按文件名模式现数同一批桶，逐组等值才算这条登记不是凭记忆写的。
   本件自己的两件（归档写入件与标本／信息生成器与干跑控件）在落笔之后才存在 ⇒ 单独两桶，
   并与该节那句「最终入库数大于这里的 28」互证（`A 现数 > 落笔 A`）。
⑦ 本轮新增：**指纹的 locale 耦合**。同一条口径在两种环境里是两个值 ⇒ 参与四路闭合的那条必须来自
   **不钉 `LC_ALL`** 的命令，另一条只断言"不同"；两个读数都印出来，并与 §26.32 里那两个十六进制串互证。
⑧ 本轮新增：**换算出来的读数也要有第二出处**。第一版把 etimes 用 `divmod(hangs, 3600)` 写成「时, 分」，
   于是 7717 s 渲染成 `2h517m`（余数 **517** 是**秒**不是分，形状看着像"分"就静通过了），而档案里落笔的是 `2h08m`：
   凡是把原始读数换算成人类可读量的地方，都必须与档案里那句换算**逐字闭合**（本件现取 §26.32 的
   `etimes … s＝**…h…m**` 那一条并要求恰好一次命中）。同一轮还纠出两处同形：闭合式里的"父提交"在回执中
   是 **blob 数**而不是那枚 SHA 的前 7 位（本件对回执里那个读数加了一条 `\d+` 形状断言，档案口径一变就红；
   但"把哪个字段填进哪一格"这件事没有机器可审——填错槽位仍会静默渲染，那条已如实进控件面的 UNCOVERED）；
   承载面读数里"最多的一面"改成由 max 断言现取，而不是照抄那一行。
"""
import ast
import glob
import hashlib
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
DAY = os.path.basename(HERE)
E = "docs/evidence/" + DAY


def sh(*args):
    r = subprocess.run(list(args), cwd=REPO, capture_output=True, text=True, check=True)
    return r.stdout


def one(pat, text, what):
    """单值取数：命中 0 处或 >1 处都拒绝（多处＝静默拿第一条＝假读数，见口径 ④）。"""
    outs = re.findall(pat, text, re.M)
    assert len(outs) == 1, "%s：模式取到 %d 处（须恰好 1）⇒ 出处不唯一或已腐烂，拒绝生成" % (what, len(outs))
    g = outs[0]
    return g[0] if isinstance(g, tuple) and len(g) == 1 else g


def grab(pat, text, what):
    """多字段取数同样要求**恰好一次命中**。"""
    outs = re.findall(pat, text, re.M)
    assert len(outs) == 1, "%s：模式取到 %d 处（须恰好 1）⇒ 出处不唯一或已腐烂，拒绝生成" % (what, len(outs))
    return outs[0]


def rb(path):
    p = path if os.path.isabs(path) else os.path.join(REPO, path)
    return open(p, "rb").read()


def md5(path, n=None):
    h = hashlib.md5(rb(path)).hexdigest()
    return h if n is None else h[:n]


def rd(path):
    return rb(path).decode("utf-8")


# ---------- 片号：三处互证＋上一片回执的目录由 glob 现取（跨日期不硬写） ----------
SUBJ = sh("git", "log", "-1", "--format=%s").strip()
HEADSLICE = int(one(r"S7 分片 (\d+)", SUBJ, "HEAD 提交说明里的片号"))
PREVSLICE, SLICE = HEADSLICE, HEADSLICE + 1
LOGS = sorted(int(re.fullmatch(r"am_s7_verify(\d+)\.log", os.path.basename(p)).group(1))
              for p in glob.glob(os.path.join(REPO, "docs/evidence/*/am_s7_verify*.log"))
              if re.fullmatch(r"am_s7_verify(\d+)\.log", os.path.basename(p)))
assert LOGS, "全树 glob 没取到任何回执档案 ⇒ 计数口径变了"
assert max(LOGS) == PREVSLICE, "最新回执号角本 %s ≠ HEAD 片号 %s ⇒ 有一片没落或片号口径变了" % (
    max(LOGS), PREVSLICE)
assert SLICE not in LOGS, "本轮回执 %s 已经存在 ⇒ 提交说明不能声称它将由本片入库" % SLICE
PREVSEC = int(one(r"开发日志 §26\.(\d+)", SUBJ, "HEAD 提交说明里的日志节号"))
PREVLOGS = sorted(os.path.relpath(p, REPO) for p in glob.glob(os.path.join(REPO, "docs/evidence/*/am_s7_verify%d.log" % PREVSLICE)))
assert len(PREVLOGS) == 1, "上一片回执在全树命中 %d 份（须 1）：%s ⇒ 有副本，引用哪一份都不唯一" % (
    len(PREVLOGS), PREVLOGS)
CURL = PREVLOGS[0]
PREVDIR = os.path.dirname(CURL)
assert PREVDIR != E, "上一片回执与本件同目录 ⇒ 本轮的跨日期登记要改（见 §26.32 的入库面那条）"
RULER = "%s/am_s7_verify%d.py" % (PREVDIR, PREVSLICE)
HDRC = "%s/am_s%d_hdr.py" % (PREVDIR, PREVSLICE)
RAWR = "%s/_raw_verify%d.out" % (PREVDIR, PREVSLICE)
RAWR_ERR = RAWR[:-4] + ".err"
HDRCTL = "%s/am_s%d_hdr_controls.py" % (PREVDIR, PREVSLICE)
HDRCTL_RAW = "%s/_raw_s%d_hdr_controls.out" % (PREVDIR, PREVSLICE)
HDRPRE = "%s/_raw_s%d_hdr_precheck.out" % (PREVDIR, PREVSLICE)
HDRW = "%s/_raw_s%d_hdr_write.out" % (PREVDIR, PREVSLICE)
AFTERW = "%s/_raw_s%d_hdr_after_write.out" % (PREVDIR, PREVSLICE)
HANG = "%s/_raw_s%d_hdr_hang_probe.txt" % (PREVDIR, PREVSLICE)
ZERO_OUT = "%s/_raw_s%d_hdr_controls.attempt1-0byte.out" % (PREVDIR, PREVSLICE)
ZERO_ERR = ZERO_OUT[:-4] + ".err"
TPROBE = "%s/am_s%d_timeout_probe.py" % (PREVDIR, PREVSLICE)
TPROBE_RAW = "%s/_raw_s%d_timeout_probe.out" % (PREVDIR, PREVSLICE)
ARCHK = "%s/am_s%d_archive_check.py" % (PREVDIR, PREVSLICE)
ARCHK_RAW = "%s/_raw_s%d_archive_check.out" % (PREVDIR, PREVSLICE)
ARCHK_PIN = "%s/_raw_s%d_archive_check_rerun_in_s%d" % (E, PREVSLICE, SLICE)
DOCS = "%s/am_s%d_docs.py" % (E, SLICE)
DOCS_SPEC = "%s/_raw_s%d_docs_rerun_after_selfcopy" % (E, SLICE)
DAILY = ".workbuddy/memory/%s.md" % DAY
DEVLOG = "docs/开发日志.md"
SELF = "am_s%d_msg.py" % SLICE
REGS = sorted(os.path.relpath(p, REPO) for p in glob.glob(os.path.join(REPO, "docs/evidence/*/_raw_t*_full.log")))
assert len(REGS) == 1, "全树回归件命中 %d 份（须 1）：%s" % (len(REGS), REGS)
REG = REGS[0]
FACTSS = sorted(os.path.relpath(p, REPO) for p in glob.glob(os.path.join(REPO, "docs/evidence/*/am_t*_facts.txt")))
assert len(FACTSS) == 1, "全树 facts 命中 %d 份（须 1）：%s" % (len(FACTSS), FACTSS)
FACTS = FACTSS[0]
FINGER_CMD = ("find src tests -name '*.py' -not -path '*__pycache__*' -print0 "
              "| sort -z | xargs -0 md5sum | md5sum | cut -d' ' -f1")
ENV = dict(os.environ)
ENV["CUDA_VISIBLE_DEVICES"] = ""

RE_TALLY = re.compile(r"^(\d+) failed, (\d+) passed, (\d+) skipped in ([\d.]+)s", re.M)
RE_TREE = re.compile(r"^TREE: head=(\w+) dirty=(\d+)$", re.M)
RE_FINGER = re.compile(r"^SRCFINGER_(START|END): (\w{32})$", re.M)
RE_DATE = re.compile(r"^DATE_(START|END): (.+)$", re.M)
RE_ELINES = re.compile(r"^E\s+AssertionError: .*$", re.M)
RE_SHORTSTAT = re.compile(r"(\d+) files? changed(?:, (\d+) insertions?\(\+\))?(?:, (\d+) deletions?\(-\))?")
RE_SEC = re.compile(r"^### 26\.(\d+)[^\n]*$", re.M)
RE_BULLET = re.compile(r"^- \*\*(.+?)\*\*", re.M)

# 上一片回执的读数（每条都要求恰好一次）
K_CLOSE = r"闭合式 远端 (\d+) == 父提交 (\d+) \+ 本片新增\(A\) (\d+) ＝ (\d+) -> (\w+)"
K_SLICE = r"本片单列复核 (\d+)/(\d+) MATCH（A (\d+)／M (\d+) 个文件）"
K_DIFF = r"双向路径差集 (\d+)/(\d+) 与 (\d+)/(\d+)"
K_SHA = r"全库 blob SHA 不符 (\d+)/(\d+)"
K_CTLA = r"A 远端一条 sha 首位改 f ⇒ 不符 (\d+)/(\d+)"
K_CTLB = r"本地独有 (\d+)/(\d+)、远端独有 (\d+)/(\d+)"
K_CTLC = r"C 闭合式右边 \+1 ⇒ (\d+) == (\d+) 判 (\w+)"
K_CTLD = r"D 远端删\*\*本片\*\*一条路径 ⇒ 本片缺失 (\d+)/(\d+)、SHA 不符 (\d+)/(\d+)"
K_BODY = r"② truncated=(\w+)，tree 条目 (\d+) = blob (\d+) \+ tree (\d+)"
K_HEAD = r"head=(\w{40}) parent=(\w{40}) dirty\(src\+tests\)=(\d+)"
K_FINGER = r"^SRC指纹 : (\w{32})"
K_MD5 = r"本件 md5 ([0-9a-f]{12})"
K_BOFF = r"BODY_OFFSET=(\d+)"
K_PREV2 = r"上一片（(\d+) 号）回执链"

# 新增件按**角色**分桶：桶和必须＝新增总数，一件落进 0 个或 ≥2 个桶都拒绝生成。
# 桶名与桶形全部由现取的片号插值（不硬写上一片号 ⇒ 片号推错了会让所有件「落不进桶」而红，而不是静默沿用上一片的故事）
# 前 8 桶与 §26.32「入库面」那条登记的分组**逐组等值**（口径 ⑥）；后 2 桶本片自件，落笔时还不存在。
BKEY_PIN = ["回执链", "表头控件面", "卡死现场", "表头预检与写盘两面", "超时探针", "写盘后表头复跑",
            "归档自查腿", "当日纪实"]
BKEY_SELF = ["本片归档写入件与标本", "本信息生成器与干跑控件"]
BUCKETS = [
    ("回执链", r"^am_s7_verify%d\.(py|log)$|^_raw_verify%d\.(out|err)$|^am_s%d_hdr\.py$" % (
        PREVSLICE, PREVSLICE, PREVSLICE)),
    ("表头控件面", r"^am_s%d_hdr_controls\.py$|^_raw_s%d_hdr_controls\.(out|err)$" % (PREVSLICE, PREVSLICE)),
    ("卡死现场", r"^_raw_s%d_hdr_hang_probe\.txt$|^_raw_s%d_hdr_controls\.attempt1-0byte\.(out|err)$" % (
        PREVSLICE, PREVSLICE)),
    ("表头预检与写盘两面", r"^_raw_s%d_hdr_(precheck|write)\.(out|err)$" % PREVSLICE),
    ("超时探针", r"^am_s%d_timeout_probe\.py$|^_raw_s%d_timeout_probe\.(out|err)$" % (PREVSLICE, PREVSLICE)),
    ("写盘后表头复跑", r"^_raw_s%d_hdr_after_write\.(out|err)$" % PREVSLICE),
    ("归档自查腿", r"^am_s%d_archive_check\.py$|^_raw_s%d_archive_check\.(out|err)$" % (PREVSLICE, PREVSLICE)),
    ("当日纪实", r"^%s$" % re.escape(os.path.basename(DAILY))),
    ("本片归档写入件与标本", r"^am_s%d_docs\.py$|^_raw_s%d_docs_rerun_after_selfcopy\.(out|err)$"
     r"|^_raw_s%d_archive_check_rerun_in_s%d\.(out|err)$" % (SLICE, SLICE, PREVSLICE, SLICE)),
    ("本信息生成器与干跑控件", r"^am_s%d_msg(_controls)?\.py$|^_raw_s%d_msg_controls(_red\d+)?\.(out|err)$" % (
        SLICE, SLICE)),
]
assert [k for k, _ in BUCKETS] == BKEY_PIN + BKEY_SELF, "桶名与桶序不一致 ⇒ 与 §26.32 分组的配对口径不可用"


def staged():
    rows = [l.split("\t") for l in sh("git", "-c", "core.quotePath=false", "diff", "--cached",
                                      "--name-status").splitlines() if l.strip()]
    kinds = sorted({r[0] for r in rows})
    assert kinds == ["A", "M"], "暂存清单里有 A/M 之外的状态字母：%s" % kinds
    return rows, [r[-1] for r in rows if r[0] == "A"], [r[-1] for r in rows if r[0] == "M"]


def bucketize(added):
    hits = {name: [] for name, _ in BUCKETS}
    for p in added:
        b = os.path.basename(p)
        m = [name for name, pat in BUCKETS if re.search(pat, b)]
        assert len(m) == 1, "%s 落进 %d 个桶 ⇒ 角色口径不唯一，拒绝生成" % (b, len(m))
        hits[m[0]].append(b)
    assert sum(len(v) for v in hits.values()) == len(added), "桶和 %d ≠ 新增总数 %d" % (
        sum(len(v) for v in hits.values()), len(added))
    empty = [k for k, v in hits.items() if not v]
    assert not empty, "角色桶 %s 里一件都没有 ⇒ 那个角色的命名口径已变，桶形是凭记忆写的" % empty
    return hits


def arch_pin(body):
    """§26.32「入库面」那条里的落笔读数：角色分组等式＋未跟踪／修改闭合式＋修改面清单，全从**暂存 blob**现取。"""
    seg = one(r"按角色分组：([^；]+)；", body, "本片登记的角色分组句")
    toks = seg.split("／")
    assert len(toks) == len(BKEY_PIN), "§26.32 的角色分组 %d 项 ≠ 生成器前桶 %d 项 ⇒ 命名口径变了" % (
        len(toks), len(BKEY_PIN))
    rows = []
    for t, key in zip(toks, BKEY_PIN):
        lab, want, claim = grab(r"^(.+?) (\d+) \*\*(\d+)\*\*$", t, "分组项（应为「%s」）的形状" % key)
        assert lab.startswith(key), "分组第 %d 项是「%s」而非「%s」⇒ 顺序与本件桶序不一致" % (
            len(rows) + 1, lab, key)
        assert int(want) == int(claim), "%s：标签里的数 %s ≠ 加粗声称 %s" % (lab, want, claim)
        rows.append((lab, int(claim)))
    eq = one(r"；([^；]*?＝\d+) 与", body, "分组等式句")
    lhs = [int(x) for x in eq.split("＝")[0].split("＋")]
    assert lhs == [c for _l, c in rows], "分组等式左边 %s ≠ 各组声称 %s" % (lhs, [c for _l, c in rows])
    nu, nm, tot = grab(r"未跟踪 \*\*(\d+)\*\*＋修改 \*\*(\d+)\*\*＝\*\*(\d+)\*\* 件", body, "落笔入库面闭合式")
    nu, nm, tot = int(nu), int(nm), int(tot)
    assert nu == sum(c for _l, c in rows) and nu + nm == tot, "落笔闭合式不齐：分组和 %d／未跟踪 %d／修改 %d／总计 %d" % (
        sum(c for _l, c in rows), nu, nm, tot)
    nface, faces = grab(r"修改 (\d+) 件＝[^（\n]*（([^）]+)）", body, "修改面清单")
    return rows, (nu, nm, tot), int(nface), sorted(faces.split("／"))


def shortstat_check():
    raw = sh("git", "-c", "core.quotePath=false", "diff", "--cached", "--shortstat").strip()
    sm = RE_SHORTSTAT.search(raw)
    assert sm, "shortstat 形状不认识：%r" % raw
    rows = [l.split("\t") for l in sh("git", "-c", "core.quotePath=false", "diff", "--cached",
                                      "--numstat").splitlines() if l.strip()]
    assert all(r[0] != "-" for r in rows), "numstat 里有 `-` 行（二进制／rename）⇒ 逐行求和口径不成立"
    ns = (len(rows), sum(int(r[0]) for r in rows), sum(int(r[1]) for r in rows))
    ss = tuple(int(x) if x else 0 for x in sm.groups())
    assert ss == ns, "shortstat %s ≠ numstat 现算 %s" % (ss, ns)
    return raw, ss


def fields_of(hdr_py):
    tree = ast.parse(rd(hdr_py))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "FIELDS":
            return [ast.literal_eval(e) for e in node.value.elts]
    raise AssertionError("在 %s 里取不到 FIELDS 赋值 ⇒ 顶格字段的分母不是现取的" % hdr_py)


def gate_curl(text):
    """上一片回执：读数全部现取（每条要求恰好一次），任一条不绿就拒绝引用它。"""
    remote, parent, na, expect, vc = grab(K_CLOSE, text, "回执里的闭合式")
    assert int(remote) == int(parent) + int(na) == int(expect), "闭合式不成立 %s/%s/%s" % (
        remote, parent, na)
    assert vc == "PASS", "闭合式判读＝%s" % vc
    verdict = one(r"LANDING_VERDICT = [^\n]*", text, "落地判读行")
    assert verdict.strip().endswith("ALL CHECKS PASS"), "回执判读行不是 ALL CHECKS PASS：" + verdict
    match, nfile, a, m = grab(K_SLICE, text, "回执里的单列复核")
    assert int(match) == int(nfile) and int(a) + int(m) == int(nfile), "单列复核不全绿"
    dl, dlden, dr, drden = grab(K_DIFF, text, "回执里的双向差集")
    sha, shaden = grab(K_SHA, text, "回执里的 SHA 普查")
    assert int(dl) == int(dr) == int(sha) == 0, "回执里有非零差集／SHA 不符"
    tr, entries, blob, treeent = grab(K_BODY, text, "回执正文的 tree 条目行")
    assert tr == "False", "truncated=%s ⇒ 回执正文被截断，不许引用" % tr
    assert int(blob) == int(remote), "正文 blob 数 ≠ 表头闭合式左值"
    assert int(entries) == int(blob) + int(treeent), "tree 条目 ≠ blob＋tree"
    h40, p40, dirty = grab(K_HEAD, text, "回执表头的 TREE 行")
    prev2 = int(one(K_PREV2, text, "回执表头的「上一片」号"))
    assert prev2 == PREVSLICE - 1, "回执表头说上一片＝%s ≠ 本片号 %s−1 ⇒ 连号断了" % (prev2, PREVSLICE)
    rsecs = re.findall(r"开发日志 §26\.(\d+)", text)
    assert rsecs and set(rsecs) == {str(PREVSEC)}, "回执表头里的日志节号＝%s ≠ HEAD 提交说明的 §26.%s ⇒ 两个出处不符" % (
        sorted(set(rsecs)), PREVSEC)
    rsec = sorted(set(rsecs))[0]
    return dict(remote=remote, parent=parent, na=na, expect=expect, verdict=verdict.strip(),
                match=match, nfile=nfile, a=a, m=m, dl=dl, dlden=dlden, dr=dr, drden=drden,
                sha=sha, shaden=shaden, entries=entries, blob=blob, tree=treeent,
                head40=h40, parent40=p40, dirty=dirty, prev2=str(prev2), recsec=rsec,
                finger=one(K_FINGER, text, "回执表头 SRC 指纹"),
                rulermd5=one(K_MD5, text, "回执表头的尺子 md5"),
                boff=one(K_BOFF, text, "回执表头 BODY_OFFSET"),
                ctla=grab(K_CTLA, text, "回执里的正对照 A"),
                ctlb=grab(K_CTLB, text, "回执里的正对照 B"),
                ctlc=grab(K_CTLC, text, "回执里的正对照 C"),
                ctld=grab(K_CTLD, text, "回执里的正对照 D"))


def cmp_triple(off):
    rcs = [subprocess.run(["cmp", "-i", "%d:%d" % (o1, o2), os.path.join(REPO, CURL),
                           os.path.join(REPO, RAWR)], capture_output=True).returncode
           for o1, o2 in ((off, 0), (off + 1, 0), (off, 1))]
    assert rcs == [0, 1, 1], "cmp 三件套复跑＝%s（须 [0,1,1]）" % rcs
    assert rb(CURL)[off:] == rb(RAWR), "回执正文区与原始 stdout 不逐字节相同（cmp 却过了？）"
    return rcs


def top_bullets(text):
    labs = RE_BULLET.findall(text)
    segs = []
    for i, let in enumerate(labs):
        start = text.index("- **%s**" % let)
        end = text.index("- **%s**" % labs[i + 1]) if i + 1 < len(labs) else len(text)
        segs.append((let, text[start:end]))
    return labs, segs


def devlog_check():
    """开发日志的**暂存 blob**（不是工作树）：最后节号＝上一节＋1，节内三条唯一性＋落笔读数由同一把尺子现验。"""
    text = sh("git", "show", ":" + DEVLOG)
    ms = list(RE_SEC.finditer(text))
    secn = ms[-1].group(1)
    assert int(secn) == PREVSEC + 1, "暂存 blob 的最后一节 §26.%s ≠ 上一节 §26.%s＋1 ⇒ 形制约束没落实" % (
        secn, PREVSEC)
    nxt = text.find("\n### ", ms[-1].end())
    body = text[ms[-1].end():(nxt if nxt != -1 else len(text))]
    labs, segs = top_bullets(body)
    assert len(labs) >= 5 and len(labs) == len(set(labs)), "§26.%s 顶级子条 %d 条且有重名标签" % (
        secn, len(labs))
    prevlog = "am_s7_verify%d.log" % PREVSLICE
    n = {k: len([s for _l, s in segs if k in s]) for k in (prevlog, "当前「", "对照的可满足性")}
    assert n[prevlog] == 1, "§26.%s 里引用上一片回执名的顶级子条有 %d 条（须 1）⇒ 表头生成器会当场红" % (
        secn, n[prevlog])
    assert n["当前「"] == 1 and n["对照的可满足性"] == 1, "节内唯一性破了：%s" % n
    q = one(r"当前「([^」]+)」", body, "钉住口径原文")
    # 上一节的引用数：同一把切段尺子现算，并与**已归档的自查件原始件**里那句读数互证（两 instrument）
    prev_body = text[ms[-2].end():ms[-1].start()]
    prev_bul = len(top_bullets(prev_body)[0])
    prev_cite = len([s for _l, s in top_bullets(prev_body)[1] if prevlog in s])
    arch_bul = int(one(r"§26\.\d+ 顶级子条＝(\d+) 条", rd(ARCHK_RAW), "自查件登记的上一节子条数"))
    assert prev_bul == arch_bul, "上一节顶级子条数：本件现算 %d ≠ 归档自查件 %d ⇒ 切段口径变了" % (
        prev_bul, arch_bul)
    assert prev_cite == 1, "上一节 §26.%s 里引用 %s 的顶级子条有 %d 条（须 1）⇒ 该节形制与回执表头的声称不符" % (
        PREVSEC, prevlog, prev_cite)
    head_text = sh("git", "show", "HEAD:" + DEVLOG)
    nh = len([m for m in RE_SEC.finditer(head_text) if m.group(1) == secn])
    ns = len([m for m in RE_SEC.finditer(text) if m.group(1) == secn])
    assert (nh, ns) == (0, 1), "§26.%s 在 HEAD 里 %d 次／暂存 blob 里 %d 次" % (secn, nh, ns)
    return secn, labs, n[prevlog], q, nh, ns, prev_cite, arch_bul, prev_bul, body


def faces_check(modified):
    """已跟踪的归档面：暂存 blob 行数 vs HEAD 行数，与 numstat 两出处闭合；只允许净增。"""
    assert sorted(os.path.basename(p) for p in modified) == PINFACES, "修改面 ≠ §26.32 登记的那批：%s" % sorted(
        os.path.basename(p) for p in modified)
    assert len(modified) == NPFACE, "修改面 %d 件 ≠ 登记 %d 件" % (len(modified), NPFACE)
    assert not sh("git", "-c", "core.quotePath=false", "diff", "--name-only").strip(), \
        "还有未暂存的已跟踪改动 ⇒ 我核对的暂存 blob 不是最终要提交的那一份"
    num = {l.split("\t")[-1]: l.split("\t")[:2] for l in
           sh("git", "-c", "core.quotePath=false", "diff", "--cached", "--numstat").splitlines() if l.strip()}
    out = []
    for rel in sorted(modified):
        n_head = len(sh("git", "show", "HEAD:" + rel).splitlines())
        n_staged = len(sh("git", "show", ":" + rel).splitlines())
        add, dele = (int(x) for x in num[rel])
        assert add == n_staged - n_head, "%s：numstat 增 %d ≠ 行数差 %d" % (rel, add, n_staged - n_head)
        assert dele == 0, "%s：相对 HEAD 有 %d 行删除 ⇒ 本轮只登记新段，不改写既有内容" % (rel, dele)
        out.append((rel, n_head, n_staged, add))
    return out


def parse_reg(raw):
    t = RE_TALLY.search(raw)
    heads = dict(RE_FINGER.findall(raw))
    assert set(heads) == {"START", "END"} and heads["START"] == heads["END"], "SRCFINGER 起≠止"
    tm = RE_TREE.search(raw)
    dates = dict(RE_DATE.findall(raw))
    elines = RE_ELINES.findall(raw)
    assert len(elines) == 2, "两条红的 E 行实取 %d 条 ⇒ 判据形状已变" % len(elines)
    return dict(tally=t.group(0), nf=t.group(1), npass=t.group(2), nskip=t.group(3),
                wall=t.group(4) + "s", head=tm.group(1), dirty=tm.group(2),
                fs=heads["START"], d1=dates["START"], d2=dates["END"],
                e1=elines[0].strip(), e2=elines[1].strip(),
                rc=one(r"^rc=(\d+)$", raw, "回归件 rc 行"))


def finger_both():
    """口径 ⑦：两条 locale 各现算。参与闭合的那条**必须不钉 LC_ALL**；钉了的那条只断言"不同"。"""
    assert "LC_ALL" not in os.environ, "本进程环境里有 LC_ALL ⇒ 那条「默认 locale」读数就不再是默认值"
    d = subprocess.run(["bash", "-c", FINGER_CMD], cwd=REPO, capture_output=True, text=True).stdout.strip()
    c = subprocess.run(["bash", "-c", FINGER_CMD], cwd=REPO, capture_output=True, text=True,
                       env=dict(os.environ, LC_ALL="C")).stdout.strip()
    for tag, v in (("默认 locale", d), ("LC_ALL=C", c)):
        assert re.fullmatch(r"[0-9a-f]{32}", v), "%s 的指纹读数不是 32 位十六进制：%r" % (tag, v)
    assert d != c, "两条 locale 现算同值 ⇒ 本轮登记的耦合已不存在，那条新学要改写"
    return d, c


def archk_readings():
    """归档自查件的档案读数（不复跑它——它是局部仪器）＋它的原始件字节／md5／判读行。"""
    rep = rd(ARCHK_RAW)
    claims, miss, mden = grab(r"声称条数＝(\d+)｜缺失＝(\d+)/(\d+)", rep, "自查件的声称与缺失")
    ctl = int(one(r"正对照共 (\d+) 条", rep, "自查件的正对照条数"))
    lines = len([l for l in rep.splitlines() if l.startswith("  正对照：")])
    assert ctl == lines, "自查件正对照：登记 %d 条 ≠ 逐行数到 %d 行 ⇒ 该行形状变了" % (ctl, lines)
    carry = re.findall(r"承载：(\S+?)＝(\d+) 条（其中独有 (\d+) 条）", rep)
    assert len(carry) == 5, "自查件的承载行实取 %d 行（须 5，五张档案面）" % len(carry)
    return dict(b=len(rb(ARCHK_RAW)), md5=md5(ARCHK_RAW, 12), eb=len(rb(ARCHK_RAW[:-4] + ".err")),
                claims=int(claims), miss=int(miss), mden=int(mden), ctl=ctl, lines=lines,
                carry=[(p, int(n), int(o)) for p, n, o in carry],
                verdict=one(r"ARCHIVE_VERDICT = [^\n]*", rep, "自查件判读行").strip(),
                pyb=len(rb(ARCHK)), pymd5=md5(ARCHK, 12))


def archk_legscope():
    """正对照（口径 ⑤）：现跑上一轮的归档自查件 ⇒ 必须**红在**引用数那句，且红的末行与已归档标本**逐字节相同**。"""
    r = subprocess.run([sys.executable, os.path.join(REPO, ARCHK)], cwd=REPO, env=ENV,
                       capture_output=True, text=True)
    assert r.returncode != 0, "%s 复跑 rc=0 ⇒ 它没受本节影响，口径 ⑤ 要改写" % ARCHK
    assert not r.stdout, "局部仪器复跑竟有 stdout（%d B）⇒ 归档件里那份不是同一条运行" % len(r.stdout.encode())
    last = [l for l in r.stderr.splitlines() if l.strip()][-1]
    assert "顶级子条有 0 条（须恰好 1）" in last, "抛出的那一行不是引用数闸门：%s" % last[:160]
    assert ("§26.%s" % SECN) in last, "红的那句没落在本节：%s" % last[:160]
    spec = [l for l in rd(ARCHK_PIN + ".err").splitlines() if l.strip()][-1]
    assert last == spec, "本轮实跑末行 ≠ 已归档标本末行 ⇒ 标本不是同一条代码路径"
    assert not rb(ARCHK_PIN + ".out"), "标本的 stdout 非 0（%d B）⇒ volatile 读数进了档案" % len(
        rb(ARCHK_PIN + ".out"))
    assert not rb(DOCS_SPEC + ".out"), "写入件标本的 stdout 非 0 ⇒ 同上"
    return r.returncode, last


def docs_specimens():
    """本片自己的两件标本（写入件＋它的自抄复跑）读数：复跑拒绝的那句里含落笔闭合式那个数。"""
    err = rd(DOCS_SPEC + ".err")
    nu, nm, tot, pin = grab(r"落笔现数＝(\d+)＋(\d+)＝(\d+) 处（本节的闭合式写的是 (\d+)）", err, "写入件复跑的拒绝句")
    assert int(nu) + int(nm) == int(tot), "标本句里的闭合式自己不齐：%s＋%s≠%s" % (nu, nm, tot)
    assert pin == str(PINTOT), "标本句钉的闭合式＝%s ≠ 从 §26.32 现取的总计 %s" % (pin, PINTOT)
    return dict(pyb=len(rb(DOCS)), pymd5=md5(DOCS, 12), eb=len(rb(DOCS_SPEC + ".err")),
                md5=md5(DOCS_SPEC + ".err", 12), nu=int(nu), nm=int(nm), tot=int(tot), pin=pin)


def ctl_readings():
    """上一片表头控件面：判红口径的三个常量从**它的源码**现取，读数从它的 stdout 现取，两出处互证。"""
    src = rd(HDRCTL)
    sub = int(one(r"^SUB_TIMEOUT = (\d+)$", src, "子运行上限常量"))
    stop = int(one(r"^ENV_FAIL_STOP = (\d+)$", src, "环境故障累计常量"))
    win = int(one(r"行号落在该句源码位置的 ±(\d+) 行窗口", src, "raise 行号窗口常量"))
    out = rd(HDRCTL_RAW)
    rows = int(one(r"^扰动条数＝(\d+)", out, "表头控件面扰动条数"))
    decl, cap, miss, to = grab(r"CASES 声明＝(\d+) 条｜每条子运行上限＝(\d+)s｜因超时没跑到的＝(\d+) 条｜超时（环境故障，不判红也不判过）＝(\d+) 条",
                               out, "表头控件面的常量行")
    assert int(cap) == sub, "控件面 stdout 的上限 %s ≠ 源码常量 %s ⇒ 常量与运行不是同一份" % (cap, sub)
    assert int(decl) + 1 == rows, "CASES 声明 %s＋基线 1 ≠ 子运行条数 %s ⇒ 那条等式的形状变了" % (decl, rows)
    assert int(miss) == 0 and int(to) == 0, "归档那份控件面有 %s 条超时／%s 条没跑到 ⇒ 它不是全过的那一次" % (
        miss, to)
    probe = rd(TPROBE_RAW)
    pcap, pcap2 = grab(r"探针里压到 (\d+)s，真件里的上限是 (\d+)s", probe, "超时探针的上限对照行")
    assert int(pcap2) == sub, "探针 stdout 的真件上限 %s ≠ 源码常量 %s ⇒ 探针读的是平行实现" % (pcap2, sub)
    pverd = one(r"TIMEOUT_GUARD_VERDICT = [^\n]*", probe, "超时探针判读行").strip()
    return dict(sub=sub, stop=stop, win=win, rows=rows, decl=int(decl), miss=int(miss), to=int(to),
                unc=one(r"本轮造不出红路的闸门＝(\d+) 条", out, "表头控件面 UNCOVERED"),
                base=one(r"^基线（未扰动副本）[^\n]*", out, "表头控件面基线行").strip(),
                verdict=one(r"CONTROL_VERDICT = [^\n]*", out, "表头控件面判读行").strip(),
                b=len(rb(HDRCTL_RAW)), md5=md5(HDRCTL_RAW, 12), eb=len(rb(HDRCTL_RAW[:-4] + ".err")),
                pyb=len(rb(HDRCTL)), pymd5=md5(HDRCTL, 12),
                pprobe=int(pcap), pverd=pverd, ppyb=len(rb(TPROBE)), pmd5=md5(TPROBE, 12),
                pb=len(rb(TPROBE_RAW)), pmd5raw=md5(TPROBE_RAW, 12), peb=len(rb(TPROBE_RAW[:-4] + ".err")))


def hang_readings():
    """卡死现场：那条**扰动副本**的 etimes 按身份定位（cmdline 含 zctl_s29_hdr），不按 max(etimes)。"""
    txt = rd(HANG)
    ps = re.findall(r"^(\d+)\s+(\d+)\s+(\d+)\s+S\s*$", txt, re.M)
    assert len(ps) == 5, "现场件的 ps 行实取 %d 条（须 5）⇒ 取法形状变了" % len(ps)
    cmd = dict(re.findall(r"^(\d+): (.*)$", txt, re.M))
    hung = [pid for pid, c in cmd.items() if "zctl_s%d_hdr" % PREVSLICE in c]
    assert len(hung) == 1, "按身份找到 %d 条扰动副本（须 1）⇒ 现场件里副本不唯一" % len(hung)
    et = {pid: int(e) for pid, _p, e in ps}
    hangs = et[hung[0]]
    gap = max(et.values()) - hangs
    assert gap > 0, "副本 etimes 已是最大 ⇒ 分不清「副本」与「父控件面」，身份取法要重验"
    h, m = hangs // 3600, (hangs % 3600) // 60
    return dict(hangs=hangs, hms="%dh%02dm" % (h, m), gap=gap, stray=one(
        r"^zctl_s\d+_hdr_\S+", txt, "扰动副本残留名"),
        b=len(rb(HANG)), md5=md5(HANG, 12),
        zb=len(rb(ZERO_OUT)), eb=len(rb(ZERO_ERR)),
        ssh=one(r"ESTAB\s+\d+\s+\d+\s+\S+\s+(\S+:22)", txt, "那条 ssh 的远端地址"))


def gpu_census():
    """我的 GPU 进程数＝0，带分母＋能红的正对照（按 /proc/PID/cmdline 判，不按命令行形状猜）。"""
    pids = [l.strip() for l in sh("nvidia-smi", "--query-compute-apps=pid",
                                  "--format=csv,noheader").splitlines() if l.strip()]

    def mine(cmds):
        return [c for c in cmds if "/tmp/amvenv" in c]

    real = mine([open("/proc/%s/cmdline" % p, "rb").read().decode("utf-8", "replace") for p in pids])
    ctl = mine(["/tmp/amvenv/bin/python fake.py"])          # 正对照：同一把匹配式必须抓到它
    assert len(ctl) == 1, "GPU 匹配式的正对照没命中 ⇒ 那个 0 是空转"
    assert real == [], "我的解释器在卡上留有进程 %s ⇒ 与钉住口径冲突" % real
    return len(pids), len(real)


def build():
    global PINFACES, NPFACE, PINTOT, SECN
    assert sys.executable.startswith("/tmp/amvenv"), "解释器不是 /tmp/amvenv 而是 %s" % sys.executable
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "生成器自身没带空串 CVD ⇒ 钉住口径失真"
    assert os.environ.get("JAX_PLATFORMS") in (None, "cpu"), "环境里 JAX_PLATFORMS＝%s" % os.environ.get("JAX_PLATFORMS")
    head_now = sh("git", "rev-parse", "HEAD").strip()
    assert re.fullmatch(r"[0-9a-f]{40}", head_now), "现取 HEAD 不是 40 位十六进制：%r" % head_now
    rows, added, modified = staged()
    buckets = bucketize(added)
    n_code = len([l for l in sh("git", "-c", "core.quotePath=false", "diff", "--cached", "--name-only",
                                "--", "src", "tests").splitlines() if l.strip()])
    assert n_code == 0, "src/tests 现数 %d 条改动 ≠ 0 ⇒『纯归档腿』成了假话" % n_code
    sraw, ss = shortstat_check()
    curl = gate_curl(rd(CURL))
    assert re.fullmatch(r"\d+", curl["parent"]), \
        "闭合式里「父提交」的读数 %r 不是十进制条数 ⇒ 档案那一句的单位已不是 blob 条数，本件的引用方式要重审（口径 ⑧）" % curl["parent"]
    assert sh("git", "rev-parse", "HEAD").strip() == curl["head40"], \
        "本地 HEAD ≠ 回执所验的树 %s ⇒ 提交说明引用的落地证明不属于当前 HEAD" % curl["head40"][:7]
    cmp_rcs = cmp_triple(int(curl["boff"]))
    hdr_fields = fields_of(HDRC)
    head_lines = rb(CURL)[:int(curl["boff"])].decode("utf-8").splitlines()
    lab = "|".join(map(re.escape, hdr_fields))
    n_defs = len([l for l in head_lines if re.match(r"^(%s)\s*:" % lab, l)])
    ind = [l for l in head_lines if re.match(r"^\s+(?:%s)\s*:" % lab, l)]
    assert not ind, "回执表头有缩进的标签行 ⇒ 顶格口径破了"
    assert n_defs == len(hdr_fields), "顶格字段 %d ≠ FIELDS 声明 %d" % (n_defs, len(hdr_fields))
    fam = md5(RULER, 12)
    assert fam == curl["rulermd5"], "尺子 md5 %s ≠ 回执表头声称 %s ⇒ 不是同一把" % (fam, curl["rulermd5"])
    alls = glob.glob(os.path.join(REPO, "docs/evidence/*/am_s7_verify*.py"))
    group = sorted(os.path.relpath(p, REPO) for p in alls if md5(p, 12) == fam)
    globn = len(alls)
    assert os.path.relpath(os.path.join(REPO, RULER), REPO) in group, "尺子不在自己的普查结果里 ⇒ 计数漏了本体"
    n_group = len(group)
    strays = sorted(os.path.basename(p) for p in alls if md5(p, 12) != fam)
    reg = parse_reg(rd(REG))
    fp_def, fp_c = finger_both()
    assert len({fp_def, reg["fs"], curl["finger"]}) == 1, "指纹四路不齐：现算 %s／回归 %s／回执 %s" % (
        fp_def, reg["fs"], curl["finger"])
    assert curl["dirty"] == "0", "回执表头 dirty(src+tests)=%s ≠ 0" % curl["dirty"]
    secn, labs, ncite, gpuquot, nh, ns, prevcite, archbul, prevbul, body = devlog_check()
    SECN = secn
    groups, (pnu, pnm, ptot), npface, pinfaces = arch_pin(body)
    PINFACES, NPFACE, PINTOT = pinfaces, npface, ptot
    assert pnu == sum(len(buckets[k]) for k, _ in BUCKETS if k in BKEY_PIN), \
        "落笔分组和 %d ≠ 本件现数前 8 桶 %d ⇒ 桶形与登记不一致（口径 ⑥）" % (
            pnu, sum(len(buckets[k]) for k, _ in BUCKETS if k in BKEY_PIN))
    for (glabel, gclaim), key in zip(groups, BKEY_PIN):
        assert len(buckets[key]) == gclaim, "%s：§26.%s 登记 %d 件 ≠ 本件按文件名现数 %d 件" % (
            glabel, secn, gclaim, len(buckets[key]))
    assert len(added) > pnu, "新增现数 %d ≤ 落笔 %d ⇒ 该节那句「最终入库数大于这里的」成了假话" % (len(added), pnu)
    hexes = re.findall(r"`([0-9a-f]{32})`", body)
    assert fp_def in hexes and fp_c in hexes, "§26.%s 里没登记本件现算的两个 locale 读数：%s" % (secn, hexes)
    assert hexes.index(fp_def) < hexes.index(fp_c), "§26.%s 里两个 32 位读数的顺序不是「默认在前、C 在后」" % secn
    faces = faces_check(modified)
    daily_in_a = [p for p in added if p == DAILY]
    assert len(daily_in_a) == 1, "当日纪实 %s 不在新增清单（命中 %d）⇒ 与 §26.%s 的分组口径不符" % (
        DAILY, len(daily_in_a), secn)
    ctlself = [p for p in added if os.path.basename(p) == "am_s%d_msg_controls.py" % SLICE]
    assert len(ctlself) == 1, "本件的干跑控件面不在入库清单（命中 %d）⇒ 末尾那句「按文件名算进自件桶」成了假话" % len(ctlself)
    CTLSELF = ctlself[0]
    arch = archk_readings()
    assert sorted(p for p, _n, _o in arch["carry"]) == sorted([f[0] for f in faces] + [DAILY]), \
        "自查件承载行的五张面 ≠ 本件现取的（四张已跟踪＋当日纪实）"
    top = max(arch["carry"], key=lambda c: c[1])
    n_at_max = len([c for c in arch["carry"] if c[1] == top[1]])
    assert n_at_max == 1, "承重最多的一面有 %d 张并列 ⇒「最多的一面」这句话没有唯一所指" % n_at_max
    assert top[0] == DEVLOG, "承重最多的一面不是开发日志（现算 %s＝%d 条）⇒ 信息里那句「承重最多的一面＝开发日志」的口径变了" % (
        top[0], top[1])
    carry_top, carry_top_n, carry_top_only = top
    arcrc, arcline = archk_legscope()
    docsp = docs_specimens()
    ctl = ctl_readings()
    hang = hang_readings()
    reg_s, reg_hms = grab(r"etimes (\d+) s＝\*\*(\d+h\d+m)\*\*", body, "§26.%s 里登记的卡死时长" % secn)
    assert (int(reg_s), reg_hms) == (hang["hangs"], hang["hms"]), \
        "本件由现场件 etimes 现算 %d s＝%s ≠ §26.%s 登记的 %s s＝%s ⇒ 换算与落笔那份不是同一个数（口径级自纠：divmod 只到小时）" % (
            hang["hangs"], hang["hms"], secn, reg_s, reg_hms)
    nv, nvme = gpu_census()
    facts = dict(l.split("=", 1) for l in rd(FACTS).splitlines() if "=" in l)
    pre = dict(b=len(rb(HDRPRE)), md5=md5(HDRPRE, 12))
    wr = dict(b=len(rb(HDRW)), md5=md5(HDRW, 12),
              cmps="/".join(grab(r"cmp 复核 rc（同偏移须 0）＝ (\d+) ｜左 \+1 须非 0＝ (\d+) ｜右 \+1 须非 0＝ (\d+)",
                                 rd(HDRW), "写盘面的 cmp 三联")),
              fields="/".join(grab(r"顶格字段行数＝ (\d+) / (\d+)", rd(HDRW), "写盘面的字段数")))
    assert wr["cmps"] == "/".join(str(x) for x in cmp_rcs), "写盘面那次 cmp 三联 %s ≠ 本轮复跑 %s ⇒ 正文区与原始件的关系动过" % (
        wr["cmps"], "/".join(str(x) for x in cmp_rcs))
    assert wr["fields"] == "%d/%d" % (n_defs, len(hdr_fields)), "写盘面字段数 %s ≠ 本件现取 %d/%d" % (
        wr["fields"], n_defs, len(hdr_fields))
    aw = dict(b=len(rb(AFTERW)), md5=md5(AFTERW, 12))
    untracked_now = [l[3:] for l in sh("git", "-c", "core.quotePath=false", "status",
                                       "--porcelain").splitlines() if l.startswith("??")]
    left = [os.path.basename(p) for p in untracked_now if os.path.basename(p) not in
            {os.path.basename(x) for x in added}]
    V = {
        "SLICE": str(SLICE), "PREVSLICE": str(PREVSLICE), "NEXTSLICE": str(SLICE + 1), "SECN": secn,
        "E": E, "NBULLET": str(len(labs)), "NBUCKETK": str(len(BUCKETS)),
        "BUCKETSUMK": str(sum(len(v) for v in buckets.values())),
        "PREVSEC": str(PREVSEC), "PREVCITE": str(prevcite), "ARCHKBUL": str(archbul),
        "PREVBUGL": str(prevbul),
        "PREV2": curl["prev2"], "RECSEC": curl["recsec"], "NCITE": str(ncite), "GPUQUOT": gpuquot,
        "SECH": str(nh), "SECSTAGED": str(ns),
        "NA": str(len(added)), "NM": str(len(modified)), "NTOT": str(len(rows)),
        "NCODE": str(n_code),
        "SHORTSTAT": sraw, "SS_F": str(ss[0]), "SS_I": str(ss[1]), "SS_D": str(ss[2]),
        "PNU": str(pnu), "PNM": str(pnm), "PTOT": str(ptot),
        "GRP": "／".join("%s＝%d" % (k, len(buckets[k])) for k, _ in BUCKETS),
        "GRPSUM": str(sum(len(buckets[k]) for k, _ in BUCKETS if k in BKEY_PIN)),
        "SELFGRPSUM": str(sum(len(buckets[k]) for k, _ in BUCKETS if k in BKEY_SELF)),
        "LEFT": str(len(left)), "LEFTN": "／".join(sorted(left)) or "无",
        "FACES": "／".join("%s %d→%d（＋%d）" % (os.path.basename(f[0]), f[1], f[2], f[3]) for f in faces),
        "ADDLINE": str(sum(f[3] for f in faces)),
        "DAILY": DAILY, "DAILYL": str(len(rb(DAILY).decode("utf-8").splitlines())), "DAILYB": str(len(rb(DAILY))),
        "HEAD": curl["head40"][:7], "HEAD40": curl["head40"],
        "PCOUNT": curl["parent"], "PARENT40": curl["parent40"], "REMOTE": curl["remote"], "EXPECT": curl["expect"],
        "ENTRIES": curl["entries"], "BLOB": curl["blob"], "TREEENT": curl["tree"],
        "MATCH": curl["match"], "NFILE": curl["nfile"], "AFILE": curl["a"], "MFILE": curl["m"],
        "DL": curl["dl"], "DLDEN": curl["dlden"], "DR": curl["dr"], "DRDEN": curl["drden"],
        "SHA": curl["sha"], "SHADEN": curl["shaden"], "VERDICT": curl["verdict"],
        "CA0": curl["ctla"][0], "CA1": curl["ctla"][1], "CB1": curl["ctlb"][0],
        "CB1D": curl["ctlb"][1], "CB2": curl["ctlb"][2], "CB2D": curl["ctlb"][3],
        "CC_L": curl["ctlc"][0], "CC_R": curl["ctlc"][1], "CC_V": curl["ctlc"][2],
        "CD1": curl["ctld"][0], "CD1D": curl["ctld"][1], "CD2": curl["ctld"][2],
        "CD2D": curl["ctld"][3], "BOFF": curl["boff"],
        "CMP": "/".join(str(x) for x in cmp_rcs), "NDEF": str(n_defs),
        "NFIELDS": str(len(hdr_fields)), "HDRLOG": CURL, "HDRLOGMD5": md5(CURL, 12),
        "HDRLOGBASE": os.path.basename(CURL), "PREVDIR": PREVDIR,
        "HDRLOGSZ": str(len(rb(CURL))), "RAWRB": str(len(rb(RAWR))),
        "RAWRMD5": md5(RAWR, 12), "RAWRERR": str(len(rb(RAWR_ERR))),
        "FAM": fam, "NGROUP": str(n_group), "NGROUPN": str(n_group + 1), "NGLOB": str(globn),
        "STRAY": "、".join(strays) or "无",
        "FPDEF": fp_def, "FPC": fp_c, "REG": REG, "TALLY": reg["tally"], "RC": reg["rc"],
        "NF": reg["nf"], "NP": reg["npass"], "NS": reg["nskip"], "WALL": reg["wall"],
        "REGHEAD": reg["head"], "REGDIRTY": reg["dirty"], "D1": reg["d1"], "D2": reg["d2"],
        "E1": reg["e1"], "E2": reg["e2"], "XFAIL": facts["XFAIL"], "COLLECT": facts["COLLECT"],
        "ARCHK": ARCHK, "ARCHKMD5": arch["pymd5"], "ARCHKB": str(arch["b"]),
        "ARCHKMD5RAW": arch["md5"], "ARCHKERR": str(arch["eb"]), "ARCHKCLAIM": str(arch["claims"]),
        "ARCHKMISS": "%d/%d" % (arch["miss"], arch["mden"]), "ARCHKCTL": str(arch["ctl"]),
        "ARCHKCTLLINE": str(arch["lines"]),
        "ARCHKVERD": arch["verdict"], "ARCHKPYB": str(arch["pyb"]),
        "CARRYDEV": str(carry_top_n), "CARRYDEVONLY": str(carry_top_only),
        "CARRYN": str(len(arch["carry"])),
        "CARRYL": "／".join("%s＝%d（独 %d）" % (os.path.basename(p), n, o) for p, n, o in arch["carry"]),
        "ARCRC": str(arcrc), "ARCLINE": arcline.strip(),
        "ARCHKPIN": ARCHK_PIN + ".err", "ARCHKPINB": str(len(rb(ARCHK_PIN + ".err"))),
        "DOCS": DOCS, "DOCSMD5": docsp["pymd5"], "DOCSB": str(docsp["pyb"]),
        "DOCSSPEC": DOCS_SPEC + ".err", "DOCSSPECB": str(docsp["eb"]),
        "DOCSSPECMD5": docsp["md5"], "SPECNU": str(docsp["nu"]), "SPECNM": str(docsp["nm"]),
        "SPECTOT": str(docsp["tot"]), "SPECN": str(docsp["pin"]),
        "HDRCTL": HDRCTL, "HDRCTLMD5": ctl["pymd5"], "HDRCTLB": str(ctl["b"]),
        "HDRCTLROWS": str(ctl["rows"]), "HDRCTLDECL": str(ctl["decl"]),
        "HDRCTLBASE": ctl["base"], "HDRCTLUNC": ctl["unc"], "HDRCTLVERD": ctl["verdict"],
        "HDRCTLERR": str(ctl["eb"]), "HDRCTLMD5RAW": ctl["md5"],
        "SUBTO": str(ctl["sub"]), "ENVSTOP": str(ctl["stop"]), "WINDOW": str(ctl["win"]),
        "CTLMISS": str(ctl["miss"]), "CTLTO": str(ctl["to"]),
        "TPROBE": TPROBE, "TPROBEMB": str(ctl["ppyb"]), "TPROBEMD5": ctl["pmd5"],
        "TPCAP": str(ctl["pprobe"]), "TPVERD": ctl["pverd"], "TPRAWB": str(ctl["pb"]),
        "TPRAWMD5": ctl["pmd5raw"], "TPRAWERR": str(ctl["peb"]),
        "HANG": HANG, "HANGB": str(hang["b"]), "HANGMD5": hang["md5"],
        "HANGES": str(hang["hangs"]), "HANGHM": hang["hms"], "HANGGAP": str(hang["gap"]),
        "HANGSTRAY": hang["stray"], "HANGSSH": hang["ssh"],
        "ZEROB": str(hang["zb"]), "ZEROEB": str(hang["eb"]),
        "PREB": str(pre["b"]), "PREMD5": pre["md5"], "WRB": str(wr["b"]), "WRMD5": wr["md5"],
        "WRCMP": wr["cmps"], "WRFLD": wr["fields"], "AWB": str(aw["b"]),
        "AWMD5": aw["md5"],
        "NV": str(nv), "NVME": str(nvme), "SELF": SELF, "SELFMD5": md5(os.path.join(HERE, SELF), 12),
        "CTLSELF": CTLSELF,
        "NOW": sh("date", "+%F %T %z").strip(), "SUBJ": SUBJ.split("：")[0],
    }
    # 键表自查：读 **`__file__`**（不是按文件名找原件）⇒ 干跑控件的扰动副本审到的是它自己。
    selftree = ast.parse(open(__file__, encoding="utf-8").read())
    litk = None
    for node in ast.walk(selftree):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "V":
            litk = node.value.keys
    assert litk is not None, "本件源码里取不到 V 的字面字典 ⇒ 键表自查的锚点已腐烂，这条闸门是空转"
    names = [k.value for k in litk
             if k is not None and isinstance(k, ast.Constant) and isinstance(k.value, str)]
    odd = len(litk) - len(names)
    assert odd == 0, "V 的字面里有非字符串常量键／**解包（Python 静默 last-wins）%d 处 ⇒ 键表不可审" % odd
    dup = sorted({n for n in names if names.count(n) > 1})
    assert not dup, "V 里有重复字面键（后值静默覆盖前值）：" + str(dup)
    assert sorted(names) == sorted(V), "字面键表 ≠ 实得键集合（有键从字面量之外塞进来或漏了）：" + str(
        sorted(set(names) ^ set(V)))
    used = set(re.findall(r"@@([A-Za-z0-9_]+)@@", TMPL))
    missing = sorted(used - set(V))
    assert not missing, "模板有未取值的字段：" + str(missing)
    unused = sorted(set(V) - used)
    assert not unused, "取值表里有模板没用到的读数（＝下一条等着的谎）：" + str(unused)
    msg = TMPL
    for k, v in V.items():
        msg = msg.replace("@@" + k + "@@", str(v))
    assert "@@" not in msg and "＠" not in msg, "信息里残留占位符 ⇒ 替换没走完"
    counts = dict(total=len(rows), added=len(added), modified=len(modified), code_changes=n_code,
                  bucket_pin=sum(len(buckets[k]) for k, _ in BUCKETS if k in BKEY_PIN),
                  bucket_self=sum(len(buckets[k]) for k, _ in BUCKETS if k in BKEY_SELF),
                  left=len(left), ruler_group=n_group, ruler_glob=globn, cmp_rcs=cmp_rcs,
                  header_fields=n_defs, section=secn, bullets=len(labs), shortstat=ss,
                  finger_def=fp_def[:12], finger_c=fp_c[:12], gpu_mine=nvme, gpu_total=nv,
                  archk_rc=arcrc, msg_bytes=len(msg.encode("utf-8")))
    return msg, counts


TMPL = """S7 分片 @@SLICE@@：纯归档腿＝§26.@@PREVSEC@@ 登记的形制约束由「另起 §26.@@SECN@@」满足＋分片 @@PREVSLICE@@ 的归档半腿整条链（回执链／表头控件面／卡死现场与超时探针／预检写盘与复跑三面／归档自查腿／当日纪实）入库；开发日志 §26.@@SECN@@

本片＝分片 @@SLICE@@（上一片＝@@PREVSLICE@@，再上一片＝@@PREV2@@）；四个片号出处互证：HEAD 提交说明「@@SUBJ@@」（HEAD 现取＝@@HEAD@@）／全树回执 glob 的最大号角本／
回执表头里那句「上一片（@@PREV2@@ 号）」＝@@PREVSLICE@@−1／文件名里的片号，任一对不上就拒绝生成。上一片回执在 **@@PREVDIR@@**、本件在 **@@E@@**（跨日期，目录由那次 glob 命中现取而非硬写）。
信息本身由 `@@SELF@@`（md5 @@SELFMD5@@，生成于 @@NOW@@）从命令现取，不是手打。

入库面：新增 @@NA@@ 件＋修改 @@NM@@ 件＝@@NTOT@@ 件；`git diff --cached --shortstat` 原文「@@SHORTSTAT@@」，
三个数已与 `--numstat` 逐行现算闭合（@@SS_F@@／@@SS_I@@／@@SS_D@@）。**src／tests 本轮 0 件改动**
（`git diff --cached --name-only -- src tests` 现数 @@NCODE@@ 条 ⇒「纯归档腿」这句有凭据）。
新增件按**角色**分 @@NBUCKETK@@ 桶，桶和＝@@BUCKETSUMK@@＝新增总数（一件落进 0 个或 ≥2 个桶都拒绝生成）：
@@GRP@@。前 8 桶与 §26.@@SECN@@「入库面」那条登记的落笔分组**逐组等值**（落笔未跟踪 @@PNU@@＋修改 @@PNM@@＝@@PTOT@@，
前 8 桶现数 @@GRPSUM@@＝落笔 @@PNU@@），本件自件两桶 @@SELFGRPSUM@@ 件在落笔之后才存在 ⇒ 与那句「最终入库数大于这里的 @@PTOT@@」互证（现数 @@NTOT@@>@@PTOT@@）。
⚠ 本片自己的落地链（`am_s7_verify@@SLICE@@.py`／`_raw_verify@@SLICE@@.out`＋`.err`／`am_s@@SLICE@@_hdr.py`）此刻还不存在
（回执 glob 里 @@SLICE@@ 号角本缺席，本件现验）；本件生成这一刻 `git status --porcelain` 里「未跟踪且不在入库清单」的件＝@@LEFT@@ 件（@@LEFTN@@）⇒ 一并按既有节奏交分片 @@NEXTSLICE@@ 入库，本片不预支。
修改 @@NM@@ 件＝§26.@@SECN@@ 现取的那四张已跟踪档案面（清单也从该节括号里现取，不硬写），行数由 `git show HEAD:<面>` 与 `git show :<面>`（**暂存 blob**）两处现取：
@@FACES@@；合计净增 @@ADDLINE@@ 行，每面 `numstat 增行 ＝ 暂存行数 − HEAD 行数` 且删除 0 行，
`git diff --name-only` 为空＝无未暂存漂移；当日纪实 @@DAILY@@ 是**新增**面（@@DAILYL@@ 行／@@DAILYB@@ B）。
**形制约束的落实**：开发日志暂存 blob 里最后一节＝§26.@@SECN@@（该节在 HEAD 里 @@SECH@@ 次、暂存 blob 里 @@SECSTAGED@@ 次 ⇒ 整节随本片落地）；
上一节号有两个出处互证＝HEAD 提交说明「开发日志 §26.@@PREVSEC@@」与上一片回执表头里那句「开发日志 §26.@@RECSEC@@」（集合相等才算）；
顶级子条 @@NBULLET@@ 条标签互异，其中引用上一片回执名的恰好 @@NCITE@@ 条、带引号的钉住登记恰好 1 条、§26.@@SECN@@ 那条「对照的可满足性」恰好 1 条
——这正是分片 @@SLICE@@ 表头生成器 `bullet_for` 的节内唯一性要求；§26.@@PREVSEC@@ 那一节里引用 @@HDRLOGBASE@@ 的顶级子条＝@@PREVCITE@@ 条（表头生成器要求恰好 1），该节顶级子条总数本件现算＝@@PREVBUGL@@ 条，与已归档自查件原始件那句「§26.@@PREVSEC@@ 顶级子条＝@@ARCHKBUL@@ 条」两出处闭合（同一把切段尺子，不是同一条声称）。

本轮新学（口径级）＝**`LC_ALL=C` 会把 SRC 指纹换成另一个值**：同一条口径在本件里跑两遍（一条不钉 locale、一条钉），
现算得 默认 locale＝@@FPDEF@@、`LC_ALL=C`＝@@FPC@@，两值必须**不同**（相同就说明这条新学已腐烂）；参与闭合的那条只能是**不钉 locale** 的那一条。
两个读数也都出现在 §26.@@SECN@@ 的登记里（该节里两个 32 位十六进制串现取，顺序＝默认在前、C 在后，本件现验）。
指纹四路闭合：现算（默认 locale）＝@@FPDEF@@＝回归件 SRCFINGER_START＝SRCFINGER_END＝上一片回执表头 SRC指纹；任一路断裂即拒绝生成。

本轮无代码腿 ⇒ 没重跑全量回归；被引用那次的凭据（逐字节从 @@REG@@ 现取）：**@@TALLY@@**（＝@@NF@@ failed＋@@NP@@ passed＋@@NS@@ skipped），
墙钟 @@WALL@@，rc=@@RC@@（rc=1＝两条既有红灯，非本轮引入），@@D1@@ → @@D2@@，当时的树＝@@REGHEAD@@ dirty=@@REGDIRTY@@。
两条登记红的 E 行：①@@E1@@；②@@E2@@。`grep -ic 'xfail|xpass'`＝@@XFAIL@@、`--collect-only`＝@@COLLECT@@
⇒ A0 断言未放宽、#27 未换指标、无新增 skip/xfail（本轮无新测试可与之比）。

分片 @@PREVSLICE@@ 回执入库（@@HDRLOG@@，@@HDRLOGSZ@@ B，md5 @@HDRLOGMD5@@；读数从回执正则现取，每条要求恰好一次命中）：
@@VERDICT@@；落地树 head＝@@HEAD40@@（父 @@PARENT40@@）＝本地 HEAD；tree 条目 @@ENTRIES@@＝blob @@BLOB@@＋tree @@TREEENT@@
（truncated=False）；闭合式 远端 @@REMOTE@@ == 父提交 blob @@PCOUNT@@ ＋ 新增(A) @@AFILE@@ ＝ @@EXPECT@@ -> PASS；
双向路径差集 @@DL@@/@@DLDEN@@ 与 @@DR@@/@@DRDEN@@；全库 blob SHA 不符 @@SHA@@/@@SHADEN@@；
本片单列复核 @@MATCH@@/@@NFILE@@ MATCH（A @@AFILE@@／M @@MFILE@@）。四个正对照都能红：A 远端一条 sha 首位改 f ⇒ 不符 @@CA0@@/@@CA1@@；
B 远端删一条 ⇒ 本地独有 @@CB1@@/@@CB1D@@、远端独有 @@CB2@@/@@CB2D@@；C 闭合式右边 +1 ⇒ @@CC_L@@ == @@CC_R@@ 判 @@CC_V@@；
D 远端删**本片**一条 ⇒ 本片缺失 @@CD1@@/@@CD1D@@、SHA 不符 @@CD2@@/@@CD2D@@。
**本轮复跑**（不引用它自己的声称）：`cmp -i @@BOFF@@:0` 三件套 rc＝@@CMP@@（同偏移须 0、任一侧 +1 须非 0），
正文区与 `_raw_verify@@PREVSLICE@@.out`（@@RAWRB@@ B，md5 @@RAWRMD5@@，stderr @@RAWRERR@@ B）逐字节相同；
表头顶格字段 @@NDEF@@ 行＝`am_s@@PREVSLICE@@_hdr.py` 的 `FIELDS`（ast 现取、不执行）声明 @@NFIELDS@@ 行且同序。
尺子＝**逐字节副本**：md5 前 12 `@@FAM@@` 这一族全树现算共 @@NGROUP@@ 份（按 md5 分组；同一条 glob 的形状命中是
@@NGLOB@@ 份，差额有名字＝@@STRAY@@），且普查结果里含本体 ⇒ 不是漏了自己的计数。本片要抄第 @@NGROUPN@@ 份。

分片 @@PREVSLICE@@ 其余入库件的读数（一律从档案正则现取，每条要求恰好一次）：表头预检面 @@PREB@@ B（md5 @@PREMD5@@）／写盘面 @@WRB@@ B（md5 @@WRMD5@@，
其中 cmp 三联＝@@WRCMP@@、与本轮复跑同值、顶格字段＝@@WRFLD@@）／写盘后复跑面 @@AWB@@ B（md5 @@AWMD5@@）；表头控件面 `@@HDRCTL@@`（md5 @@HDRCTLMD5@@）的 stdout
@@HDRCTLB@@ B（md5 @@HDRCTLMD5RAW@@，stderr @@HDRCTLERR@@ B）＝原始件那句「扰动条数＝@@HDRCTLROWS@@」（＝基线 1＋CASES 声明 @@HDRCTLDECL@@，本件现验），
本轮超时 @@CTLTO@@ 条、因超时没跑到的 @@CTLMISS@@ 条、
如实登记造不出红路 @@HDRCTLUNC@@ 条 ⇒ @@HDRCTLVERD@@；它的基线行原文「@@HDRCTLBASE@@」。判红窗口与两个继承常量从**该件源码**现取（不重敲）＝
子运行上限 @@SUBTO@@ s／环境故障累计 @@ENVSTOP@@ 条／raise 行号 ±@@WINDOW@@ 行——其中上限那条与探针 stdout 的「真件里的上限是 @@SUBTO@@s」**同值**（两出处互证）。
卡死现场 `@@HANG@@`（@@HANGB@@ B，md5 @@HANGMD5@@）：那条**扰动副本**的 etimes＝@@HANGES@@ s＝@@HANGHM@@（按 cmdline 身份定位，不按 max(etimes)，
父控件面比它长 @@HANGGAP@@ s），其 ssh 仍持 ESTAB 到 @@HANGSSH@@，那对原始件 @@ZEROB@@ B／@@ZEROEB@@ B 改名保留为 `.attempt1-0byte.{out,err}`，副本残留名＝@@HANGSTRAY@@。
超时探针 `@@TPROBE@@`（@@TPROBEMB@@ B，md5 @@TPROBEMD5@@）stdout @@TPRAWB@@ B（md5 @@TPRAWMD5@@，stderr @@TPRAWERR@@ B）：探针里压到 @@TPCAP@@s ⇒ @@TPVERD@@。
归档自查件 `@@ARCHK@@`（@@ARCHKPYB@@ B，md5 @@ARCHKMD5@@）及其原始件（stdout @@ARCHKB@@ B，md5 @@ARCHKMD5RAW@@，stderr @@ARCHKERR@@ B）＝
声称 @@ARCHKCLAIM@@ 条、缺失 @@ARCHKMISS@@、正对照 @@ARCHKCTL@@ 条（逐行数到 @@ARCHKCTLLINE@@ 行）⇒ @@ARCHKVERD@@；
它的承载面读数（@@CARRYN@@ 张面，括号里「独」＝只出现在那一张面的声称条数，逐条现取）＝@@CARRYL@@，其中承重最多的一面＝开发日志 @@CARRYDEV@@ 条／独 @@CARRYDEVONLY@@ 条——这句不是照抄那一行的数，而是由本件两条断言现取（「承重最多的一面＝开发日志」＋并列当场红，红掉任何一条这句话就不成立）。

⚠ **局部仪器不是常备尺子**（本轮实测，两出处闭合）：`am_s@@PREVSLICE@@_archive_check.py` 的「上一节引用数」闸按它自己那一轮的最后一节写，
§26.@@SECN@@ 一开就红。本件现跑一次为证：rc=@@ARCRC@@，抛出的那一行「@@ARCLINE@@」，且该行与已归档标本 `@@ARCHKPIN@@`（@@ARCHKPINB@@ B）的末行**逐字节相同**
⇒ 标本与本轮实跑是同一条代码路径，该声称的适用范围只到落地那一轮，不许当回归件复跑。
本片自己的归档写入件 `@@DOCS@@`（@@DOCSB@@ B，md5 @@DOCSMD5@@）也留下一条**自纠标本**：`@@DOCSSPEC@@`（@@DOCSSPECB@@ B，md5 @@DOCSSPECMD5@@）
里那句「落笔现数＝@@SPECNU@@＋@@SPECNM@@＝@@SPECTOT@@ 处（本节的闭合式写的是 @@SPECN@@）」＝把写入件自己 `cp -p` 进档案目录之后，那条闸门**如实拒绝**了复跑
（@@SPECN@@ 与本件从 §26.@@SECN@@ 现取的落笔总计同值）⇒ 闸没被放宽，只是它数的那个数在落笔之后动了。

诚实登记（本片没做什么）：① 无代码腿、无新测试、无回归重跑、无 A3 求解；② **性能数字为零**——GPU 全程未碰，
卡上 compute 进程 @@NV@@ 个、我的解释器（读 /proc/PID/cmdline 含 /tmp/amvenv）＝@@NVME@@/@@NV@@，
该 0 的正对照＝同一把匹配式喂一条假 cmdline 必须命中（现验 1/1）；③ 钉住口径原文从暂存 blob 现取＝「@@GPUQUOT@@」，
出处唯一性已在上面 §26.@@SECN@@ 的计数里验过；④ **本片（分片 @@SLICE@@）自己的落地证明要到下一轮才存在**：
本信息只声称分片 @@PREVSLICE@@ 那一份已落地，不预先声称本片被复核过；⑤ 本件的干跑控件面 `@@CTLSELF@@` 此刻**已在入库清单里**（上面自件桶现数含它，且本件对它有一条「命中 1」的断言），
但本信息**不引用它自己的控件读数**（扰动条数／UNCOVERED／判读行）——控件跑的是本件的副本，本件若在控件之前读那份 stdout 就是循环；
那份读数由**分片 @@NEXTSLICE@@** 的归档自查腿引用。
排期：#30 试片尺寸裁决（A3 关键路径，当前唯一可评分面＝N=1）→ raw 臂 N≥2 ＋ `--stage full --measured-per-track`
→ #27 腿②（缺省 tuned→hk）与 #33 的 15 位点系数**同批**，只对 A3 的 1→18 外部趋势打分 → #40 → #39；
#41／#37／#38／#29 各自待办；#51 已量化（对形态 ≤0.0040%）降级不关单。
"""

if __name__ == "__main__":
    m, c = build()
    open("/tmp/am_s%d_msg.txt" % SLICE, "w", encoding="utf-8").write(m)
    print(m)
    print("---- 现取核对 ----")
    for k, v in c.items():
        print("%s=%s" % (k, v))
