#!/usr/bin/env python3
r"""S7 分片 29 提交信息生成器：信息里的每个数字都从命令／档案现取，不许手敲。

本轮＝**纯归档腿**（`git diff --cached --name-only -- src tests` 现数 0 件）⇒ 不写"改了哪个求解器"，
写的是两件可核查的事：① §26.30 登记的形制约束由「另起 §26.31」满足（节内唯一性由**暂存 blob** 现验）；
② 分片 28 的回执链／两面控件面／归档自查件入库（读数全部从回执与原始件正则现取，且**复跑** cmp 三件套）。

片号与"上一片回执"一律**现取并交叉印证**（§26.27 (u) 与 §26.30 都记过机械克隆把上一片的故事安在本片头上）：
`git log -1 --format=%s` 给出 HEAD 的片号 ⇒ 本片＝它＋1；全树 glob 回执档案给出最大号角本 ⇒ 必须＝HEAD 片号；
文件名里的片号必须＝同一个数；三处任一对不上就拒绝生成。

五条口径（沿用＋本轮新增）：
① 正文一律 **PLACEHOLDER＋str.replace**，不用 f-string（f-string 会把字面量里的 `{32}` 当替换字段
   **静默吃掉**，§26.27 (q) 记过一条自纠）。生成后**双向**查：模板里没取值的字段拒绝，取值表里模板
   没用到的读数也拒绝（没用到的读数＝下一条等着的谎）。
② 引命令**原文**并把三个数闭合到 `--numstat` 逐行现算（git 对 1 用单数「1 deletion(-)」，按复数形状
   取数会把它静默折成 0）。
③ 每个 0 与它的分母同印；每条闸门都要在 `am_s29_msg_controls.py` 里被**坏输入**打红过（只看 rc≠0
   的控制＝空转，本件按"抛出的那一行含闸门句"判）。
④ **单值取数必须按形状类并要求恰好一次**：`(\S+)` 从中文句子里会把「；」和后面的词一起吞掉，
   `search` 在多处命中时会静默拿到第一条 ⇒ 本件所有取数走 `one()`（findall 后断言 len==1）。
⑤ 本轮新增：**局部仪器不许当常备尺子**。`am_s28_archive_check.py` 的「下一片锚点」闸是按**它自己那一轮**
   的最后一个节写的 ⇒ 本节一开（§26.31 出现）它必然红在那句。这不是档案坏了，是那句声称的范围只在落地的
   那一轮。本件把它**现跑一次并要求它红在那句**，作为"这条闸是活的"的正对照，同时在信息里如实写成局部仪器。
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


def rb(path):
    p = path if os.path.isabs(path) else os.path.join(REPO, path)
    return open(p, "rb").read()


def md5(path, n=None):
    h = hashlib.md5(rb(path)).hexdigest()
    return h if n is None else h[:n]


def rd(path):
    return rb(path).decode("utf-8")


# ---------- 片号：三处互证 ----------
SUBJ = sh("git", "log", "-1", "--format=%s").strip()
HEADSLICE = int(one(r"S7 分片 (\d+)", SUBJ, "HEAD 提交说明里的片号"))
PREVSLICE, SLICE = HEADSLICE, HEADSLICE + 1
RCPT = sorted(int(re.fullmatch(r"am_s7_verify(\d+)\.log", os.path.basename(p)).group(1))
              for p in glob.glob(os.path.join(REPO, "docs/evidence/*/am_s7_verify*.log"))
              if re.fullmatch(r"am_s7_verify(\d+)\.log", os.path.basename(p)))
assert RCPT, "全树 glob 没取到任何回执档案 ⇒ 计数口径变了"
assert max(RCPT) == PREVSLICE, "最新回执号角本 %s ≠ HEAD 片号 %s ⇒ 有一片没落或片号口径变了" % (
    max(RCPT), PREVSLICE)
assert SLICE not in RCPT, "本轮回执 %s 已经存在 ⇒ 提交说明不能声称它将由本片入库" % SLICE
PREVSEC = int(one(r"开发日志 §26\.(\d+)", SUBJ, "HEAD 提交说明里的日志节号"))

CURL = "%s/am_s7_verify%d.log" % (E, PREVSLICE)
RULER = "%s/am_s7_verify%d.py" % (E, PREVSLICE)
HDRC = "%s/am_s%d_hdr.py" % (E, PREVSLICE)
RAWR = "%s/_raw_verify%d.out" % (E, PREVSLICE)
RAWR_ERR = RAWR[:-4] + ".err"
HDRCTL = "%s/am_s%d_hdr_controls.py" % (E, PREVSLICE)
HDRCTL_RAW = "%s/_raw_s%d_hdr_controls.out" % (E, PREVSLICE)
HDRPRE = "%s/_raw_s%d_hdr_precheck.out" % (E, PREVSLICE)
HDRW = "%s/_raw_s%d_hdr_write.out" % (E, PREVSLICE)
ARCHK = "%s/am_s%d_archive_check.py" % (E, PREVSLICE)
ARCHK_RAW = "%s/_raw_s%d_archive_check.out" % (E, PREVSLICE)
REG = E + "/_raw_t27_full.log"
FACTS = E + "/am_t27_facts.txt"
DEVLOG = "docs/开发日志.md"
SELF = "am_s%d_msg.py" % SLICE
FIVE = [DEVLOG, "docs/项目评估与下一步计划_2026-10-06.md", "docs/项目开发总览.md",
        ".workbuddy/memory/2026-10-10.md", ".workbuddy/memory/MEMORY.md"]
FINGER_CMD = ("find src tests -name '*.py' -not -path '*__pycache__*' -print0 "
              "| sort -z | xargs -0 md5sum | md5sum | cut -d' ' -f1")
ENV = dict(os.environ)
ENV["CUDA_VISIBLE_DEVICES"] = ""
ENV["JAX_PLATFORMS"] = "cpu"

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

# 新增件按**角色**分桶：桶和必须＝新增总数，一件落进 0 个或 ≥2 个桶都拒绝生成。
# 桶名与桶形全部由现取的片号插值（不硬写上一片号 ⇒ 片号推错了会让所有件「落不进桶」而红，而不是静默沿用上一片的故事）
BUCKETS = [
    ("分片 %d 回执链" % PREVSLICE,
     r"^am_s7_verify%d\.(py|log)$|^_raw_verify%d\.(out|err)$|^am_s%d_hdr\.py$" % (PREVSLICE, PREVSLICE, PREVSLICE)),
    ("分片 %d 表头控件面" % PREVSLICE,
     r"^am_s%d_hdr_controls\.py$|^_raw_s%d_hdr_controls\.(out|err)$" % (PREVSLICE, PREVSLICE)),
    ("分片 %d 表头预检与写盘两面" % PREVSLICE,
     r"^_raw_s%d_hdr_(precheck|write)\.(out|err)$" % PREVSLICE),
    ("分片 %d 归档自查腿" % PREVSLICE,
     r"^am_s%d_archive_check\.py$|^_raw_s%d_archive_check\.(out|err)$" % (PREVSLICE, PREVSLICE)),
    ("本信息生成器（分片 %d）与干跑控件" % SLICE,
     r"^am_s%d_msg(_controls)?\.py$|^_raw_s%d_msg_controls(_red\d+)?\.(out|err)$" % (SLICE, SLICE)),
]
(B_CH, B_CTL, B_PRE, B_ARCH, B_SELF) = [b[0] for b in BUCKETS]


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


def grab(pat, text, what):
    """多字段取数同样要求**恰好一次命中**（多处＝静默拿第一条＝假读数）。"""
    outs = re.findall(pat, text, re.M)
    assert len(outs) == 1, "%s：模式取到 %d 处（须恰好 1）⇒ 出处不唯一或已腐烂，拒绝生成" % (what, len(outs))
    return outs[0]


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
    return dict(remote=remote, parent=parent, na=na, expect=expect, verdict=verdict.strip(),
                match=match, nfile=nfile, a=a, m=m, dl=dl, dlden=dlden, dr=dr, drden=drden,
                sha=sha, shaden=shaden, entries=entries, blob=blob, tree=treeent,
                head40=h40, parent40=p40, dirty=dirty,
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
    """开发日志的**暂存 blob**（不是工作树）：最后节号＝上一节＋1，节内三条唯一性由表头生成器同口径现验。"""
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
    n = {k: len([s for _l, s in segs if k in s]) for k in
         (prevlog, "当前「", "判据的可通过性")}
    assert n[prevlog] == 1, "§26.%s 里引用上一片回执名的顶级子条有 %d 条（须 1）⇒ 表头生成器会当场红" % (
        secn, n[prevlog])
    assert n["当前「"] == 1 and n["判据的可通过性"] == 1, "节内唯一性破了：%s" % n
    q = one(r"当前「([^」]+)」", body, "钉住口径原文")
    # 上一节的引用数：同一把切段尺子现算，并与**已归档的自查件原始件**里那句读数互证（两 instrument）
    prev_body = text[ms[-2].end():ms[-1].start()]
    prev_cite = len([s for _l, s in top_bullets(prev_body)[1] if prevlog in s])
    arch_cite = int(one(r"am_s7_verify%d\.log 被 (\d+) 条子条引用" % PREVSLICE,
                        rd(ARCHK_RAW), "自查件登记的上一节引用数"))
    assert prev_cite == arch_cite, "上一节引用数：本件现算 %d ≠ 归档自查件 %d ⇒ 切段口径变了" % (
        prev_cite, arch_cite)
    assert prev_cite >= 2, "上一节 §26.%s 只有 %d 条顶级子条引用回执名 ⇒「必须另起一节」这句成了假话（1 条就该回去追加）" % (
        PREVSEC, prev_cite)
    head_text = sh("git", "show", "HEAD:" + DEVLOG)
    nh = len([m for m in RE_SEC.finditer(head_text) if m.group(1) == secn])
    ns = len([m for m in RE_SEC.finditer(text) if m.group(1) == secn])
    assert (nh, ns) == (0, 1), "§26.%s 在 HEAD 里 %d 次／暂存 blob 里 %d 次" % (secn, nh, ns)
    return secn, labs, n[prevlog], q, nh, ns, prev_cite


def faces_check(modified):
    """五张归档面：暂存 blob 行数 vs HEAD 行数，与 numstat 两出处闭合；只允许净增。"""
    assert sorted(modified) == sorted(FIVE), "修改面 ≠ 五张归档面：%s" % sorted(modified)
    assert not sh("git", "-c", "core.quotePath=false", "diff", "--name-only").strip(), \
        "还有未暂存的已跟踪改动 ⇒ 我核对的暂存 blob 不是最终要提交的那一份"
    num = {l.split("\t")[-1]: l.split("\t")[:2] for l in
           sh("git", "-c", "core.quotePath=false", "diff", "--cached", "--numstat").splitlines() if l.strip()}
    out = []
    for rel in FIVE:
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


def archk_readings():
    """归档自查件的档案读数（不复跑它——它是局部仪器）＋它的原始件字节／md5／判读行。"""
    rep = rd(ARCHK_RAW)
    return dict(b=len(rb(ARCHK_RAW)), md5=md5(ARCHK_RAW, 12), eb=len(rb(ARCHK_RAW[:-4] + ".err")),
                claims=int(one(r"现取声称 (\d+) 条", rep, "自查件的声称条数")),
                segs=int(one(r"新追加段数＝(\d+) 段", rep, "自查件的段数")),
                perturb=int(one(r"内存扰动 (\d+) 条读数闸门", rep, "自查件的读数扰动条数")),
                subset=int(one(r"＋(\d+) 条档案子集扰动", rep, "自查件的子集扰动条数")),
                anchor=int(one(r"＋(\d+) 条锚点闸门", rep, "自查件的锚点扰动条数")),
                verdict=one(r"ARCHIVE_VERDICT = [^\n]*", rep, "自查件判读行").strip(),
                pyb=len(rb(ARCHK)), pymd5=md5(ARCHK, 12))


def archk_legscope():
    """正对照：现跑上一轮的归档自查件 ⇒ 必须**红在**「下一片锚点」那句（口径 ⑤ 的实测依据）。"""
    r = subprocess.run([sys.executable, os.path.join(REPO, ARCHK)], cwd=REPO, env=ENV,
                       capture_output=True, text=True)
    assert r.returncode != 0, "am_s%d_archive_check.py 复跑 rc=0 ⇒ 它没受本节影响，口径 ⑤ 要改写" % PREVSLICE
    assert not r.stdout, "局部仪器复跑竟有 stdout（%d B）⇒ 归档件里那份不是同一条运行" % len(r.stdout.encode())
    last = [l for l in r.stderr.splitlines() if l.strip()][-1]
    assert "锚点被本轮碰坏" in last, "抛出的那一行不是锚点闸门：%s" % last[:160]
    return r.returncode, last


def gpu_census():
    """我的 GPU 进程数＝0，带分母＋能红的匹配式正对照（按 /proc/PID/cmdline 判，不按命令行形状猜）。"""
    pids = [l.strip() for l in sh("nvidia-smi", "--query-compute-apps=pid",
                                  "--format=csv,noheader").splitlines() if l.strip()]
    def mine(cmds):
        outs = []
        for c in cmds:
            if "/tmp/amvenv" in c:
                outs.append(c)
        return outs
    real = mine([open("/proc/%s/cmdline" % p, "rb").read().decode("utf-8", "replace") for p in pids])
    ctl = mine(["/tmp/amvenv/bin/python fake.py"])          # 正对照：同一把匹配式必须抓到它
    assert len(ctl) == 1, "GPU 匹配式的正对照没命中 ⇒ 那个 0 是空转"
    assert real == [], "我的解释器在卡上留有进程 %s ⇒ 与钉住口径冲突" % real
    return len(pids), len(real)


def build():
    assert sys.executable.startswith("/tmp/amvenv"), "解释器不是 /tmp/amvenv 而是 %s" % sys.executable
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "生成器自身没带空串 CVD ⇒ 钉住口径失真"
    head_now = sh("git", "rev-parse", "HEAD").strip()
    assert re.fullmatch(r"[0-9a-f]{40}", head_now), "现取 HEAD 不是 40 位十六进制：%r ⇒ 后面的等式没有可比的东西" % head_now
    rows, added, modified = staged()
    buckets = bucketize(added)
    n_code = len([l for l in sh("git", "-c", "core.quotePath=false", "diff", "--cached", "--name-only",
                                "--", "src", "tests").splitlines() if l.strip()])
    assert n_code == 0, "src/tests 现数 %d 条改动 ≠ 0 ⇒『纯归档腿』成了假话" % n_code
    sraw, ss = shortstat_check()
    curl = gate_curl(rd(CURL))
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
    finger_now = subprocess.run(["bash", "-c", FINGER_CMD], cwd=REPO, capture_output=True,
                                text=True).stdout.strip()
    assert len({finger_now, reg["fs"], curl["finger"]}) == 1, "指纹四路不齐：现树 %s／回归 %s／回执 %s" % (
        finger_now, reg["fs"], curl["finger"])
    assert curl["dirty"] == "0", "回执表头 dirty(src+tests)=%s ≠ 0" % curl["dirty"]
    secn, labs, ncite, gpuquot, nh, ns, prevcite = devlog_check()
    faces = faces_check(modified)
    arch = archk_readings()
    arcrc, arcline = archk_legscope()
    nv, nvme = gpu_census()
    facts = dict(l.split("=", 1) for l in rd(FACTS).splitlines() if "=" in l)
    hdr_ctl = dict(b=len(rb(HDRCTL_RAW)), md5=md5(HDRCTL_RAW, 12), eb=len(rb(HDRCTL_RAW[:-4] + ".err")),
                   rows=int(one(r"^扰动条数＝(\d+)", rd(HDRCTL_RAW), "表头控件面扰动条数")),
                   unc=one(r"本轮造不出红路的闸门＝(\d+) 条", rd(HDRCTL_RAW), "表头控件面 UNCOVERED"),
                   base=one(r"^基线（未扰动副本）[^\n]*", rd(HDRCTL_RAW), "表头控件面基线行").strip(),
                   verdict=one(r"CONTROL_VERDICT = [^\n]*", rd(HDRCTL_RAW),
                               "表头控件面判读行").strip(),
                   pyb=len(rb(HDRCTL)), pymd5=md5(HDRCTL, 12))
    pre = dict(b=len(rb(HDRPRE)), md5=md5(HDRPRE, 12))
    wr = dict(b=len(rb(HDRW)), md5=md5(HDRW, 12))
    untracked_now = [l[3:] for l in sh("git", "-c", "core.quotePath=false", "status",
                                       "--porcelain").splitlines() if l.startswith("??")]
    left = [os.path.basename(p) for p in untracked_now if os.path.basename(p) not in
            {os.path.basename(x) for x in added}]
    V = {
        "SLICE": str(SLICE), "PREVSLICE": str(PREVSLICE), "NEXTSLICE": str(SLICE + 1), "SECN": secn,
        "NBULLET": str(len(labs)),
        "PREVSEC": str(PREVSEC), "PREVCITE": str(prevcite),
        "NCITE": str(ncite), "GPUQUOT": gpuquot, "SECH": str(nh), "SECSTAGED": str(ns),
        "NA": str(len(added)), "NM": str(len(modified)), "NTOT": str(len(rows)),
        "NCODE": str(n_code),
        "SHORTSTAT": sraw, "SS_F": str(ss[0]), "SS_I": str(ss[1]), "SS_D": str(ss[2]),
        "NCH": str(len(buckets[B_CH])), "NCTL": str(len(buckets[B_CTL])),
        "NPRE": str(len(buckets[B_PRE])),
        "NARCH": str(len(buckets[B_ARCH])),
        "NMSG": str(len(buckets[B_SELF])),
        "BUCKETSUM": str(sum(len(v) for v in buckets.values())),
        "LEFT": str(len(left)), "LEFTN": "／".join(sorted(left)) or "无",
        "DL0": str(faces[0][1]), "DL1": str(faces[0][2]), "DLD": str(faces[0][3]),
        "PL0": str(faces[1][1]), "PL1": str(faces[1][2]), "PLD": str(faces[1][3]),
        "OV0": str(faces[2][1]), "OV1": str(faces[2][2]), "OVD": str(faces[2][3]),
        "DY0": str(faces[3][1]), "DY1": str(faces[3][2]), "DYD": str(faces[3][3]),
        "MEM0": str(faces[4][1]), "MEM1": str(faces[4][2]), "MEMD": str(faces[4][3]),
        "ADDLINE": str(sum(f[3] for f in faces)),
        "HEAD": curl["head40"][:7], "HEAD40": curl["head40"], "PARENT": curl["parent40"][:7],
        "PARENT40": curl["parent40"], "REMOTE": curl["remote"], "EXPECT": curl["expect"],
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
        "HDRLOGBASE": os.path.basename(CURL),
        "HDRLOGSZ": str(len(rb(CURL))), "RAWRB": str(len(rb(RAWR))),
        "RAWRMD5": md5(RAWR, 12), "RAWRERR": str(len(rb(RAWR_ERR))),
        "FAM": fam, "NGROUP": str(n_group), "NGLOB": str(globn), "STRAY": "、".join(strays) or "无",
        "FINGER": finger_now, "REG": REG, "TALLY": reg["tally"], "RC": reg["rc"],
        "NF": reg["nf"], "NP": reg["npass"], "NS": reg["nskip"], "WALL": reg["wall"],
        "REGHEAD": reg["head"], "REGDIRTY": reg["dirty"], "D1": reg["d1"], "D2": reg["d2"],
        "E1": reg["e1"], "E2": reg["e2"], "XFAIL": facts["XFAIL"], "COLLECT": facts["COLLECT"],
        "ARCHK": ARCHK, "ARCHKMD5": arch["pymd5"], "ARCHKB": str(arch["b"]),
        "ARCHKMD5RAW": arch["md5"], "ARCHKERR": str(arch["eb"]), "ARCHKCLAIM": str(arch["claims"]),
        "ARCHKSEG": str(arch["segs"]), "ARCHKPERT": str(arch["perturb"]),
        "ARCHKSUB": str(arch["subset"]), "ARCHKANC": str(arch["anchor"]),
        "ARCHKVERD": arch["verdict"], "ARCHKPYB": str(arch["pyb"]),
        "ARCRC": str(arcrc), "ARCLINE": arcline.strip(),
        "HDRCTL": HDRCTL, "HDRCTLMD5": hdr_ctl["pymd5"], "HDRCTLB": str(hdr_ctl["b"]),
        "HDRCTLROWS": str(hdr_ctl["rows"]), "HDRCTLBASE": hdr_ctl["base"],
        "HDRCTLUNC": hdr_ctl["unc"], "HDRCTLVERD": hdr_ctl["verdict"],
        "HDRCTLERR": str(hdr_ctl["eb"]), "HDRCTLMD5RAW": hdr_ctl["md5"],
        "PREB": str(pre["b"]), "PREMD5": pre["md5"], "WRB": str(wr["b"]), "WRMD5": wr["md5"],
        "NV": str(nv), "NVME": str(nvme), "SELF": SELF, "SELFMD5": md5(os.path.join(HERE, SELF), 12),
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
                  bucket_sum=sum(len(v) for v in buckets.values()), left=len(left),
                  ruler_group=n_group, ruler_glob=globn, cmp_rcs=cmp_rcs, header_fields=n_defs,
                  section=secn, bullets=len(labs), shortstat=ss, finger=finger_now[:12],
                  gpu_mine=nvme, gpu_total=nv, archk_rc=arcrc)
    return msg, counts


TMPL = """S7 分片 @@SLICE@@：纯归档腿＝§26.@@PREVSEC@@ 登记的形制约束由「另起 §26.@@SECN@@」满足＋分片 @@PREVSLICE@@ 的回执链／表头控件面／预检与写盘两面／归档自查件入库；开发日志 §26.@@SECN@@

本片＝分片 @@SLICE@@（上一片＝@@PREVSLICE@@）；三个片号出处互证：HEAD 提交说明「@@SUBJ@@」（HEAD 现取＝@@HEAD@@）／全树回执 glob 的最大号角本／
文件名里的片号，任一对不上就拒绝生成。信息本身由 `@@SELF@@`（md5 @@SELFMD5@@，生成于 @@NOW@@）从命令现取，不是手打。

入库面：新增 @@NA@@ 件＋修改 @@NM@@ 件＝@@NTOT@@ 件；`git diff --cached --shortstat` 原文「@@SHORTSTAT@@」，
三个数已与 `--numstat` 逐行现算闭合（@@SS_F@@／@@SS_I@@／@@SS_D@@）。**src／tests 本轮 0 件改动**
（`git diff --cached --name-only -- src tests` 现数 @@NCODE@@ 条 ⇒「纯归档腿」这句有凭据）。
新增件按**角色**分五桶，桶和＝@@BUCKETSUM@@＝新增总数（一件落进 0 个或 ≥2 个桶都拒绝生成）：
分片 @@PREVSLICE@@ 回执链 @@NCH@@ 件／表头控件面 @@NCTL@@ 件／表头预检与写盘两面 @@NPRE@@ 件／归档自查腿 @@NARCH@@ 件／
本信息生成器与干跑控件 @@NMSG@@ 件。⚠ 本片自己的落地链（`am_s7_verify@@SLICE@@.py`／`_raw_verify@@SLICE@@.out`＋`.err`／
`am_s@@SLICE@@_hdr.py`）此刻还不存在（回执 glob 里 @@SLICE@@ 号角本缺席，本件现验）；本件生成这一刻`git status --porcelain` 里
「未跟踪且不在入库清单」的件＝@@LEFT@@ 件（@@LEFTN@@）⇒ 两者一并按既有节奏交分片 @@NEXTSLICE@@ 入库，本片不预支。
修改 @@NM@@ 件＝五张归档面，行数由 `git show HEAD:<面>` 与 `git show :<面>`（**暂存 blob**）两处现取：
开发日志 @@DL0@@→@@DL1@@（＋@@DLD@@）、计划文档 @@PL0@@→@@PL1@@（＋@@PLD@@）、总览 @@OV0@@→@@OV1@@（＋@@OVD@@）、
当日纪实 @@DY0@@→@@DY1@@（＋@@DYD@@）、MEMORY 索引 @@MEM0@@→@@MEM1@@（＋@@MEMD@@）；合计净增 @@ADDLINE@@ 行，
每面 `numstat 增行 ＝ 暂存行数 − HEAD 行数` 且删除 0 行，`git diff --name-only` 为空＝无未暂存漂移。
**形制约束的落实**：开发日志暂存 blob 里最后一节＝§26.@@SECN@@（该节在 HEAD 里 @@SECH@@ 次、暂存 blob 里 @@SECSTAGED@@ 次 ⇒ 整节随本片落地）；
顶级子条 @@NBULLET@@ 条标签互异，其中引用上一片回执名的恰好 @@NCITE@@ 条、带引号的钉住登记恰好 1 条、LESSON 锚句恰好 1 条
——这正是分片 @@SLICE@@ 表头生成器 `bullet_for` 的节内唯一性要求；§26.@@PREVSEC@@ 那一节有 @@PREVCITE@@ 条顶级子条引用 @@HDRLOGBASE@@（唯一性要求恰好 1 条）⇒ 那一节结构上回不去，本轮另起新节。

本轮无代码腿 ⇒ 没重跑全量回归；引用它的凭据＝**指纹四路闭合**：现算 src+tests 指纹＝@@FINGER@@＝回归件
SRCFINGER_START＝SRCFINGER_END＝上一片回执表头 SRC指纹（四路任一路断裂即拒绝生成）。被引用的那次：**@@TALLY@@**
（＝@@NF@@ failed＋@@NP@@ passed＋@@NS@@ skipped），墙钟 @@WALL@@，rc=@@RC@@（rc=1＝两条既有红灯，非本轮引入），
@@D1@@ → @@D2@@，当时的树＝@@REGHEAD@@ dirty=@@REGDIRTY@@（内容级＝现树；那次跑在当时的未提交工作树上）。
两条登记红的 E 行（逐字节从 @@REG@@ 现取）：①@@E1@@；②@@E2@@。`grep -ic 'xfail|xpass'`＝@@XFAIL@@、
`--collect-only`＝@@COLLECT@@ ⇒ A0 断言未放宽、#27 未换指标、无新增 skip/xfail（本轮无新测试可与之比）。

分片 @@PREVSLICE@@ 回执入库（@@HDRLOG@@，@@HDRLOGSZ@@ B，md5 @@HDRLOGMD5@@；读数从回执正则现取，每条要求恰好一次命中）：
@@VERDICT@@；落地树 head＝@@HEAD40@@（父 @@PARENT40@@）＝本地 HEAD；tree 条目 @@ENTRIES@@＝blob @@BLOB@@＋tree @@TREEENT@@
（truncated=False）；闭合式 远端 @@REMOTE@@ == 父提交 @@PARENT@@ ＋ 新增(A) @@AFILE@@ ＝ @@EXPECT@@ -> PASS；
双向路径差集 @@DL@@/@@DLDEN@@ 与 @@DR@@/@@DRDEN@@；全库 blob SHA 不符 @@SHA@@/@@SHADEN@@；
本片单列复核 @@MATCH@@/@@NFILE@@ MATCH（A @@AFILE@@／M @@MFILE@@）。四个正对照都能红：A 远端一条 sha 首位改 f ⇒ 不符 @@CA0@@/@@CA1@@；
B 远端删一条 ⇒ 本地独有 @@CB1@@/@@CB1D@@、远端独有 @@CB2@@/@@CB2D@@；C 闭合式右边 +1 ⇒ @@CC_L@@ == @@CC_R@@ 判 @@CC_V@@；
D 远端删**本片**一条 ⇒ 本片缺失 @@CD1@@/@@CD1D@@、SHA 不符 @@CD2@@/@@CD2D@@。
**本轮复跑**（不引用它自己的声称）：`cmp -i @@BOFF@@:0` 三件套 rc＝@@CMP@@（同偏移须 0、任一侧 +1 须非 0），
正文区与 `_raw_verify@@PREVSLICE@@.out`（@@RAWRB@@ B，md5 @@RAWRMD5@@，stderr @@RAWRERR@@ B）逐字节相同；
表头顶格字段 @@NDEF@@ 行＝`am_s@@PREVSLICE@@_hdr.py` 的 `FIELDS`（ast 现取、不执行）声明 @@NFIELDS@@ 行且同序。
尺子＝**逐字节副本**：md5 前 12 `@@FAM@@` 这一族全树现算共 @@NGROUP@@ 份（按 md5 分组；同一条 glob 的形状命中是
@@NGLOB@@ 份，差额有名字＝@@STRAY@@），且普查结果里含本体 ⇒ 不是漏了自己的计数。

分片 @@PREVSLICE@@ 的三面原始件一并入库：表头预检面 @@PREB@@ B（md5 @@PREMD5@@）／写盘面 @@WRB@@ B（md5 @@WRMD5@@）／
表头控件面 `@@HDRCTL@@`（md5 @@HDRCTLMD5@@）的 stdout @@HDRCTLB@@ B（md5 @@HDRCTLMD5RAW@@，stderr @@HDRCTLERR@@ B）＝
扰动 @@HDRCTLROWS@@ 条、如实登记造不出红路 @@HDRCTLUNC@@ 条 ⇒ @@HDRCTLVERD@@；它的基线行原文「@@HDRCTLBASE@@」。
归档自查件 `@@ARCHK@@`（@@ARCHKPYB@@ B，md5 @@ARCHKMD5@@）及其原始件（stdout @@ARCHKB@@ B，md5 @@ARCHKMD5RAW@@，
stderr @@ARCHKERR@@ B）：新追加段 @@ARCHKSEG@@ 段、开发日志那段现取声称 @@ARCHKCLAIM@@ 条全部命中原样、
扰动＝@@ARCHKPERT@@ 条读数闸门＋@@ARCHKSUB@@ 条档案子集＋@@ARCHKANC@@ 条锚点闸门 ⇒ @@ARCHKVERD@@。
⚠ **它是局部仪器，不是常备尺子**（本轮实测）：它的「下一片锚点」闸按**它那一轮**的最后一个节写，§26.@@SECN@@ 一开就红——
本件现跑一次为证（rc=@@ARCRC@@，抛出的那一行「@@ARCLINE@@」）⇒ 该声称的适用范围只到落地那一轮，不许当回归件复跑。

诚实登记（本片没做什么）：① 无代码腿、无新测试、无回归重跑、无 A3 求解；② **性能数字为零**——GPU 全程未碰，
卡上 compute 进程 @@NV@@ 个、我的解释器（读 /proc/PID/cmdline 含 /tmp/amvenv）＝@@NVME@@/@@NV@@，
该 0 的正对照＝同一把匹配式喂一条假 cmdline 必须命中（现验 1/1）；③ 钉住口径原文从暂存 blob 现取＝「@@GPUQUOT@@」，
出处唯一性已在上面 §26.@@SECN@@ 的计数里验过；④ **本片（分片 @@SLICE@@）自己的落地证明要到下一轮才存在**：
本信息只声称分片 @@PREVSLICE@@ 那一份已落地，不预先声称本片被复核过。
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
