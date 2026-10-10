#!/usr/bin/env python3
"""S7 分片 27 提交信息生成器：信息里的每个数字都从命令／档案现取，不许手敲。

本轮**没有代码腿**（src/tests 0 件改动）⇒ 本件不写"改了哪个求解器"那段，改为把
"在网回归仍是分片 26 那一份、且它量的就是现树"这件事**用指纹证出来**：
现算 src+tests 指纹 ＝＝ 回归件 SRCFINGER_START ＝＝ SRCFINGER_END ＝＝ 上一片回执表头的 SRC指纹，
四路同值才允许在信息里引用那次回归；四路里任一路断裂即拒绝生成（不是降级成叙述）。

口径两处与上一片不同，都是本轮学到的：
① 正文一律 **PLACEHOLDER＋str.replace**，不用 f-string（§26.28 (q) 记过 f-string 把字面量 `{32}`
   当替换字段**静默吃掉**；表头生成器早已是这个口径，信息生成器本轮才跟上）⇒ 生成后查 `@@` 残留，
   并**双向**查：模板里没取值的字段要拒绝，取值表里没被模板用到的读数也要拒绝（没用到的读数＝下一条谎）。
② 复现声称只钉**结构上钉得住的读数**：审计件普查行里的"类别分布"随证据目录内容移动（出处池＝整个目录），
   本轮新写几十件就可能让同一个数从 B 类改判 C 类 ⇒ 把分布当"复现"来断言会造出一条对不齐的闸门。
   只比**总数＋三类红旗**（§26.29 (l)）。
③ 命令原文与"重建的原文"两选一时长者**必错**：shortstat 在 1 条时是单数「1 deletion(-)」，按复数形状
   取数 ⇒ 可选分组静默不参与 ⇒ `or "0"` 折成一个像样的 0（本件首版就是这样，信息里写着"0 deletions"）。
   改为**引命令原文**＋把三个数闭合到 `--numstat` 逐行现算（`shortstat_check`），并在正文里说出口。

正则与闸门全部放模块级常量／函数：本件可被 `am_s27_msg_controls.py` import 而无副作用
（`build()` 不在 import 时执行），干跑用的是**同一把**探测器，不是另写的平行实现。
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

REG = "docs/evidence/" + DAY + "/_raw_t27_full.log"          # 在网全量回归（分片 26 那一次）
FACTS = "docs/evidence/" + DAY + "/am_t27_facts.txt"
CURL = "docs/evidence/" + DAY + "/am_s7_verify26.log"         # 上一片回执（本轮入库）
RULER = "docs/evidence/" + DAY + "/am_s7_verify26.py"         # 现用尺子（本轮入库）
HDR26 = "docs/evidence/" + DAY + "/am_s26_hdr.py"             # 上一片表头生成器（FIELDS 出处）
RAW26 = "docs/evidence/" + DAY + "/_raw_verify26.out"         # 上一片回执的原始 stdout
AUDIT = "docs/evidence/" + DAY + "/am_a3_hdr_audit.py"
AUDIT_RAW = "docs/evidence/" + DAY + "/_raw_a3_hdr_audit.out"
CTL = "docs/evidence/" + DAY + "/am_a3_hdr_audit_controls.py"
CTL_RAW = "docs/evidence/" + DAY + "/_raw_a3_hdr_audit_controls.out"
WRITER = "docs/evidence/" + DAY + "/am_s27_docs.py"
WRITER_RAW = "docs/evidence/" + DAY + "/_raw_s27_docs.out"
SELF = "am_s27_msg.py"
DEVLOG = "docs/开发日志.md"
FIVE = ["docs/开发日志.md", "docs/项目评估与下一步计划_2026-10-06.md", "docs/项目开发总览.md",
        ".workbuddy/memory/2026-10-10.md", ".workbuddy/memory/MEMORY.md"]
FINGER_CMD = ("find src tests -name '*.py' -not -path '*__pycache__*' -print0 "
              "| sort -z | xargs -0 md5sum | md5sum | cut -d' ' -f1")

ENV = dict(os.environ)
ENV["CUDA_VISIBLE_DEVICES"] = ""
ENV["JAX_PLATFORMS"] = "cpu"

RE_TALLY = re.compile(r"^(\d+) failed, (\d+) passed, (\d+) skipped in ([\d.]+)s", re.M)
RE_RC = re.compile(r"^rc=(\d+)$", re.M)
RE_TREE = re.compile(r"^TREE: head=(\w+) dirty=(\d+)$", re.M)
RE_FINGER = re.compile(r"^SRCFINGER_(START|END): (\w{32})$", re.M)
RE_DATE = re.compile(r"^DATE_(START|END): (.+)$", re.M)
RE_ELINES = re.compile(r"^E\s+AssertionError: .*$", re.M)
# 回执表头里的来历读数（正文走 RE_B_BODY／RE_C_CLOSE…，表头走这几条 ⇒ 两出处互证）
RE_C_HEAD = re.compile(r"head=(\w{40}) parent=(\w{40}) dirty\(src\+tests\)=(\d+)")
RE_C_FINGER = re.compile(r"^SRC指纹 : (\w{32})", re.M)
RE_C_RULERMD5 = re.compile(r"本件 md5 ([0-9a-f]{12})")
RE_C_COPIES = re.compile(r"同一把尺子已有 (\d+) 份逐字节副本")
RE_C_CLOSE = re.compile(r"闭合式 远端 (\d+) == 父提交 (\d+) \+ 本片新增\(A\) (\d+) ＝ (\d+) -> (\w+)")
RE_C_DIFF = re.compile(r"双向路径差集 (\d+)/(\d+) 与 (\d+)/(\d+)")
RE_C_SHA = re.compile(r"全库 blob SHA 不符 (\d+)/(\d+)")
RE_C_SLICE = re.compile(r"本片单列复核 (\d+)/(\d+) MATCH（A (\d+)／M (\d+) 个文件）")
RE_C_CTLA = re.compile(r"A 远端一条 sha 首位改 f ⇒ 不符 (\d+)/(\d+)")
RE_C_CTLB = re.compile(r"本地独有 (\d+)/(\d+)、远端独有 (\d+)/(\d+)")
RE_C_CTLC = re.compile(r"C 闭合式右边 \+1 ⇒ (\d+) == (\d+) 判 (\w+)")
RE_C_CTLD = re.compile(r"D 远端删\*\*本片\*\*一条路径 ⇒ 本片缺失 (\d+)/(\d+)、SHA 不符 (\d+)/(\d+)")
RE_C_VERDICT = re.compile(r"LANDING_VERDICT = .*", re.M)
RE_C_BOFF = re.compile(r"BODY_OFFSET=(\d+)")
RE_B_BODY = re.compile(r"② truncated=(\w+)，tree 条目 (\d+) = blob (\d+) \+ tree (\d+)")
# 写手 stdout 里那四条汇总行：信息引用它们＝引用**那次运行**，所以逐条与本生成器的现算对账
RE_W_BYTES = re.compile(r"^字节闭合：主面 (\d+)＋(\d+)=(\d+)｜控件面 (\d+)＋(\d+)=(\d+)｜更正锚点 (\d+) 处"
                        r"（旧值 词边界 历史面 (\d+)／改前 (\d+)／诱饵 (\d+)／子串 (\d+)）", re.M)
RE_W_DET = re.compile(r"^过期值检测器三面：历史面 (\d+)／改前工作树 (\d+)／改后全文未自证 (\d+)"
                      r"｜标本 (\d+)／抹掉真值后红 (\d+)｜自纠上一条 (\d+)（本轮 \+(\d+)）", re.M)
RE_W_26 = re.compile(r"^分片 26：main (\w{7})｜blob (\d+) 闭合 (\d+)==(\d+)\+(\d+)=(\d+)｜差集 (\d+)/(\d+)、(\d+)/(\d+)"
                     r"｜SHA 不符 (\d+)/(\d+)｜本片 (\d+)/(\d+)｜BOFF (\d+) cmp (\d+)/(\d+)/(\d+)"
                     r"｜顶格 (\d+)＝FIELDS (\d+)｜尺子 (\d+) 份", re.M)
RE_W_TREE = re.compile(r"^树：HEAD (\w{7})（父 (\w{7})）｜dirty total (\d+)／evidence (\d+)／src\+tests (\d+)"
                       r"｜SRC 指纹 (\w{32})＝回执那一条 (\w{32})｜GPU (\d+) 个、我的 (\d+)/(\d+)", re.M)
RE_CENSUS = re.compile(r"表头数字 (\d+) 个＝.*?／无出处 (\d+)／跨面无归属 (\d+)／无出处十六进制读数 (\d+)")
RE_AUDIT_V = re.compile(r"AUDIT_VERDICT = .*", re.M)
RE_CTL_ROW = re.compile(r"^  (C\d+) .*?｜面＝(主面|控件面)｜rc=(\d+)｜真红行 (\d+) 条（普查汇总行不计）"
                        r"｜期望句「(.*)」(命中|\*\*未命中\*\*)$", re.M)
RE_CTL_C0 = re.compile(r"^C0 未扰动基线 rc=0 且判读行全过＝True", re.M)
RE_CTL_V = re.compile(r"CONTROL_VERDICT = .*", re.M)
RE_CTL_N = re.compile(r"^扰动条数＝(\d+)", re.M)
RE_WRITE_ROW = re.compile(r"^WRITE (.+?)：(\d+) → (\d+) 行（＋(\d+)），标记 1 次｜更正腿另计 (\d+) 处", re.M)
# git 对 1 用**单数**（"1 deletion(-)"），形状必须 files?／insertions?／deletions? 都放开；
# 只按复数取数 ⇒ 第三组静默不参与 ⇒ 首版把 1 删成 0（靠下面的 numstat 闭合才露出来）。
RE_SHORTSTAT = re.compile(r"(\d+) files? changed(?:, (\d+) insertions?\(\+\))?(?:, (\d+) deletions?\(-\))?")
RE_SEC = re.compile(r"^### 26\.(\d+)[^\n]*$", re.M)
RE_BULLET = re.compile(r"^\*\*\((\w)\)[^\n]*$", re.M)

# 新增件按**角色**分桶（桶和必须＝新增总数；任何一件落不进桶、或落进两个桶，都拒绝生成）
BUCKETS = [
    ("分片 26 回执链", r"^am_s7_verify26\.(py|log)$|^_raw_verify26\.(out|err)$|^am_s26_hdr\.py$"),
    ("分片 26 表头控件面", r"^am_s26_hdr_controls\.(py|log)$|^_raw_s26_hdr_controls\.(out|err)$"),
    ("A3 预检面（腿①）", r"^am_a3_driver\.py$|^am_a3_run\.sh$|^am_a3_preflight(_controls)?\.log$"
                         r"|^am_a3_preflight(_controls)?_hdr\.txt$|^am_a3_preflight_controls\.py$"
                         r"|^_raw_a3_preflight(_controls)?\.(out|err)$"),
    ("表头溯源审计仪器", r"^am_a3_hdr_audit(_controls)?\.py$|^_raw_a3_hdr_audit(_controls)?\.(out|err)$"),
    ("归档写手", r"^am_s27_docs\.py$|^_raw_s27_docs(_precheck)?\.(out|err)$"),
    ("信息生成器与干跑控件", r"^am_s27_msg(_controls)?\.py$|^_raw_s27_msg_controls\.(out|err)$"),
]


def sh(*args):
    r = subprocess.run(list(args), cwd=REPO, capture_output=True, text=True, check=True)
    return r.stdout


def rb(path):
    p = path if os.path.isabs(path) else os.path.join(REPO, path)
    return open(p, "rb").read()


def md5(path, n=None):
    h = hashlib.md5(rb(path)).hexdigest()
    return h if n is None else h[:n]


def rd(path):
    return rb(path).decode("utf-8")


def staged():
    rows = [l.split("\t") for l in sh("git", "-c", "core.quotePath=false", "diff", "--cached",
                                      "--name-status").splitlines() if l.strip()]
    added = [r[-1] for r in rows if r[0] == "A"]
    modified = [r[-1] for r in rows if r[0] == "M"]
    assert len(added) + len(modified) == len(rows), "暂存清单里有 A/M 之外的状态字母"
    return rows, added, modified


def bucketize(added):
    hits = {name: [] for name, _ in BUCKETS}
    for p in added:
        b = os.path.basename(p)
        m = [name for name, pat in BUCKETS if re.search(pat, b)]
        assert len(m) == 1, "%s 落进 %d 个桶 ⇒ 角色口径不唯一，拒绝生成" % (b, len(m))
        hits[m[0]].append(b)
    assert sum(len(v) for v in hits.values()) == len(added), "桶和 ≠ 新增总数"
    return hits


def parse_reg(raw):
    t = RE_TALLY.search(raw)
    assert t, "回归件里取不到汇总行"
    tm = RE_TREE.search(raw)
    heads = dict(RE_FINGER.findall(raw))
    assert set(heads) == {"START", "END"}, "SRCFINGER 只取到 %s" % sorted(heads)
    dates = dict(RE_DATE.findall(raw))
    elines = RE_ELINES.findall(raw)
    assert len(elines) == 2, "两条红的 E 行实取 %d 条 ⇒ 判据形状已变，不许静默通过" % len(elines)
    assert heads["START"] == heads["END"], "SRCFINGER 起≠止 ⇒「回归期间源码一字未动」不成立"
    return dict(tally=t.group(0), nf=int(t.group(1)), npass=int(t.group(2)), nskip=int(t.group(3)),
                wall=t.group(4) + "s", head=tm.group(1), dirty=tm.group(2), fs=heads["START"],
                fe=heads["END"], d1=dates["START"], d2=dates["END"], elines=elines)


def fields_of(hdr_py):
    """从上一片表头生成器的源码里取 FIELDS 声明（ast 解析，**不执行**它）＝回执"顶格字段行数"的分母。"""
    tree = ast.parse(rd(hdr_py))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "FIELDS":
            return [ast.literal_eval(e) for e in node.value.elts]
    raise AssertionError("在 %s 里取不到 FIELDS 赋值 ⇒ 分母不是现取的" % hdr_py)


def gate_curl(text):
    """上一片回执的闸门：读数全部现取，任一条不绿就拒绝引用它。"""
    close = RE_C_CLOSE.search(text)
    assert close, "回执里取不到闭合式"
    remote, parent, na, expect, verdict_close = close.groups()
    assert int(remote) == int(parent) + int(na) == int(expect), \
        "闭合式在信息生成器里不成立（%s/%s/%s/%s）⇒ 不许引用" % (remote, parent, na, expect)
    assert verdict_close == "PASS", "闭合式判读＝%s" % verdict_close
    v = RE_C_VERDICT.search(text).group(0)
    assert v.endswith("ALL CHECKS PASS"), "回执判读行不是 ALL CHECKS PASS：" + v
    slc = RE_C_SLICE.search(text)
    match, nfile, a, m = (int(x) for x in slc.groups())
    assert match == nfile and a + m == nfile, "单列复核不是全绿：%s" % slc.group(0)
    diff, sha = RE_C_DIFF.search(text), RE_C_SHA.search(text)
    assert int(diff.group(1)) == int(diff.group(3)) == int(sha.group(1)) == 0, "回执里有非零差集／SHA 不符"
    body = RE_B_BODY.search(text)
    assert body.group(1) == "False", "回执正文 truncated=%s" % body.group(1)
    assert int(body.group(3)) == int(remote), "正文的 blob 数与表头闭合式左值不等 ⇒ 两出处不同"
    assert int(body.group(2)) == int(body.group(3)) + int(body.group(4)), "tree 条目 ≠ blob＋tree"
    return dict(remote=remote, parent=parent, na=na, expect=expect, verdict=v, entries=body.group(2),
                blob=body.group(3), tree=body.group(4), dl=diff.group(1), dlden=diff.group(2),
                dr=diff.group(3), drden=diff.group(4), sha=sha.group(1), shaden=sha.group(2),
                match=match, nfile=nfile, a=a, m=m,
                ctl_a=RE_C_CTLA.search(text).groups(), ctl_b=RE_C_CTLB.search(text).groups(),
                ctl_c=RE_C_CTLC.search(text).groups(), ctl_d=RE_C_CTLD.search(text).groups(),
                head40=RE_C_HEAD.search(text).group(1), parent40=RE_C_HEAD.search(text).group(2),
                dirty_st=RE_C_HEAD.search(text).group(3), finger=RE_C_FINGER.search(text).group(1),
                ruler_md5=RE_C_RULERMD5.search(text).group(1), copies=RE_C_COPIES.search(text).group(1),
                boff=RE_C_BOFF.search(text).group(1))


def run_pinned(rel):
    r = subprocess.run([sys.executable, os.path.join(REPO, rel)], cwd=REPO, env=ENV,
                       capture_output=True, text=True)
    assert r.returncode == 0, "%s 复跑 rc=%s ⇒ 判读行不能代表这台仪器现在说什么" % (rel, r.returncode)
    assert not r.stderr, "%s 复跑 stderr 非空（%d 字节）⇒ 有警告被吞" % (rel, len(r.stderr.encode()))
    return r.stdout


def census_pinned(text):
    out = [m.groups() for m in RE_CENSUS.finditer(text)]
    assert len(out) == 2, "审计件应印两面普查行，现取 %d 条" % len(out)
    return out


def audit_check():
    """复跑审计件：只把**总数＋三类红旗**当复现（类别分布随目录移动，见 docstring ②）。"""
    fresh = run_pinned(AUDIT)
    f, a = census_pinned(fresh), census_pinned(rd(AUDIT_RAW))
    assert f == a, "普查的四个钉住读数复跑与档案不一致：档案 %s → 复跑 %s" % (a, f)
    assert all(x[1] == x[2] == x[3] == "0" for x in f), "三类红旗里有非零：%s" % f
    vf, va = RE_AUDIT_V.search(fresh).group(0), RE_AUDIT_V.search(rd(AUDIT_RAW)).group(0)
    assert vf == va == "AUDIT_VERDICT = ALL HEADER READINGS PROVENANCE-PASS", "判读行变了：%s／%s" % (vf, va)
    return f, vf


def controls_check():
    """复跑审计控件件：十条扰动**逐条**要在自己的闸门上红，且基线 C0 先绿。"""
    fresh = run_pinned(CTL)
    rows = RE_CTL_ROW.findall(fresh)
    n = int(RE_CTL_N.search(fresh).group(1))
    assert RE_CTL_C0.search(fresh), "控件面缺 C0 未扰动基线行 ⇒ 所有红都不作数"
    vf, va = RE_CTL_V.search(fresh).group(0), RE_CTL_V.search(rd(CTL_RAW)).group(0)
    assert vf == va, "控件面判读行与档案不同：%s ≠ %s" % (vf, va)
    assert vf.endswith("ALL TEN PERTURBATIONS GO RED ON THEIR OWN GATE"), vf
    assert len(rows) == n == 10, "扰动条数声称 %d、实取 %d 行" % (n, len(rows))
    for cid, face, rc, red, sent, hit in rows:
        assert rc == "3" and hit == "命中", "%s（%s）没在自己的闸门上红：rc=%s hit=%s" % (cid, face, rc, hit)
    assert not [p for p in os.listdir(HERE) if p.startswith("zctl_")], "扰动副本没被清掉 ⇒ 会污染工作树"
    return rows


def writer_check(modified):
    """归档腿的两出处互证＋一笔**必须说出口的意外**：写手的"改前"是工作树，不是 HEAD。

    本轮的特殊之处：上一轮（§26.@@SECN@@ (a)–(k)）的归档改动**没被任何提交带走** ⇒
    写手声称的 `旧行数` 比 HEAD 版本多。若这里断言"旧行数＝HEAD 实得"就会把**正确**的仪器判红
    （首版正是这样，被干跑控件抓出）。改成可加的两段分解：
    `HEAD → 暂存 blob 的净增` ＝ `本轮写手增量` ＋ `动手前就在工作树里的未提交部分`，
    三段都由命令现取，且"未提交部分确实来自上一轮"用"HEAD 里没有本节、暂存 blob 里有"证明。
    """
    rep = rd(WRITER_RAW)
    rows = RE_WRITE_ROW.findall(rep)
    assert len(rows) == 5, "写手报告应覆盖五张面，现取 %d 行" % len(rows)
    assert [r[0] for r in rows] == FIVE, "写手覆盖的面与五张归档面不一致：%s" % [r[0] for r in rows]
    assert not sh("git", "-c", "core.quotePath=false", "diff", "--name-only").strip(), \
        "还有未暂存的改动 ⇒ 我核对的暂存 blob 不是最终要提交的那一份"
    num = {l.split("\t")[-1]: l.split("\t")[:2] for l in
           sh("git", "-c", "core.quotePath=false", "diff", "--cached", "--numstat").splitlines() if l.strip()}
    out = []
    for rel, old, new, delta, corr in rows:
        assert int(new) - int(old) == int(delta), "%s：写手声称＋%s 与 %s→%s 不等" % (rel, delta, old, new)
        n_head = len(sh("git", "show", "HEAD:" + rel).splitlines())
        n_staged = len(sh("git", "show", ":" + rel).splitlines())
        assert n_staged == int(new), "%s：新行数 %s ≠ 暂存 blob 实得 %d ⇒ 两出处不等" % (rel, new, n_staged)
        add, dele = (int(x) for x in num[rel])
        assert add - dele == n_staged - n_head, "%s：numstat 净增 %d ≠ 行数差 %d" % (
            rel, add - dele, n_staged - n_head)
        since_head = n_staged - n_head
        assert since_head >= int(delta), "%s：对 HEAD 净增 %d < 本轮增量 %s ⇒ 分解不成立" % (
            rel, since_head, delta)
        assert rel in modified, rel + " 不在暂存的修改清单里"
        out.append(dict(rel=rel, old=old, new=new, delta=int(delta), corr=int(corr),
                        head=n_head, since_head=since_head, pre=since_head - int(delta),
                        add=add, dele=dele))
    return out, rep


def cmp_check(curl, hdr_fields):
    """回执的"正文区逐字节"由本轮**重跑** cmp 三件套证明（同偏移须 0、任一侧 +1 须非 0）。"""
    off = int(curl["boff"])
    rcs = [subprocess.run(["cmp", "-i", "%d:%d" % (o1, o2), os.path.join(REPO, CURL),
                           os.path.join(REPO, RAW26)], capture_output=True).returncode
           for o1, o2 in ((off, 0), (off + 1, 0), (off, 1))]
    assert rcs == [0, 1, 1], "cmp 三件套复跑＝%s（须 [0,1,1]）⇒ 回执正文区不是那次运行的逐字节后缀" % rcs
    body = open(os.path.join(REPO, RAW26), "rb").read()
    raw = rb(CURL)
    assert raw[off:] == body, "回执正文区与 _raw_verify26.out 不是逐字节相同（cmp 却过了？）"
    lines = raw[:off].decode("utf-8").splitlines()          # 表头＝**字节**偏移之前那一段
    lab = "|".join(map(re.escape, hdr_fields))
    defs = [l for l in lines if re.match(r"^(%s)\s*:" % lab, l)]
    mis = [l for l in lines if re.match(r"^\s+(?:%s)\s*:" % lab, l)]
    assert not mis, "回执表头有缩进的标签＋冒号行：" + str(mis[:1])
    assert len(defs) == len(hdr_fields), "回执表头顶格字段 %d ≠ FIELDS 声明 %d" % (
        len(defs), len(hdr_fields))
    return rcs, len(defs), len(hdr_fields)


def devlog_check():
    """开发日志（**暂存 blob**，不是工作树）里最后一个 `### 26.NN` ＝本轮节号；该节子条字母须连续。"""
    text = sh("git", "show", ":" + DEVLOG)
    ms = list(RE_SEC.finditer(text))
    assert ms, "取不到 `### 26.NN` 标题 ⇒ 节号是凭记忆的"
    secn = ms[-1].group(1)
    body_start = ms[-1].end()
    nxt = text.find("\n### ", body_start)
    body = text[body_start:(nxt if nxt != -1 else len(text))]
    letters = RE_BULLET.findall(body)
    assert letters == [chr(ord("a") + i) for i in range(len(letters))], \
        "§26.%s 的子条字母不连续：%s" % (secn, letters)
    assert "am_s7_verify26.log" in body, "§26.%s 里没有引用上一片回执名 ⇒ 入库句不在了" % secn
    assert "当前「不动 GPU」" in body, "§26.%s 里取不到钉住口径原文" % secn
    assert "极性翻转不是扰动" in body, "§26.%s 里取不到本轮教训原文" % secn
    # "写手的改前工作树里那部分未提交归档确实属于上一轮"＝命令可查：HEAD 里没有本节，暂存 blob 里有
    head_text = sh("git", "show", "HEAD:" + DEVLOG)
    n_head = len([m for m in RE_SEC.finditer(head_text) if m.group(1) == secn])
    n_staged = len([m for m in RE_SEC.finditer(text) if m.group(1) == secn])
    assert (n_head, n_staged) == (0, 1), "§26.%s 在 HEAD 里 %d 次／暂存 blob 里 %d 次 ⇒『本节随本片整体落地』不成立" % (
        secn, n_head, n_staged)
    return secn, letters, n_head, n_staged


def w26_cross_check(rep, curl, copies, n_defs, n_fields, cmp_rcs):
    """写手那轮印出的"分片 26"行 vs 本生成器现算：21 个字段逐位相等才允许引用那句话。"""
    m = RE_W_26.search(rep)
    assert m, "写手 stdout 里取不到『分片 26：』汇总行"
    want = [curl["head40"][:7], curl["remote"], curl["remote"], curl["parent"], curl["na"], curl["expect"],
            curl["dl"], curl["dlden"], curl["dr"], curl["drden"], curl["sha"], curl["shaden"],
            str(curl["match"]), str(curl["nfile"]), curl["boff"],
            str(cmp_rcs[0]), str(cmp_rcs[1]), str(cmp_rcs[2]), str(n_defs), str(n_fields), str(copies)]
    names = ["main", "blob", "闭左", "父", "新增", "闭右", "dl", "dl分母", "dr", "dr分母", "sha", "sha分母",
             "match", "nfile", "boff", "cmp0", "cmp1", "cmp2", "顶格", "FIELDS", "尺子份数"]
    got = list(m.groups())
    assert len(got) == len(want), "对账字段数变了（写手 %d／生成器 %d）⇒ 汇总行形状不再可信" % (len(got), len(want))
    bad = [(names[i], got[i], want[i]) for i in range(len(want)) if got[i] != want[i]]
    assert not bad, "写手那轮的读数与本轮现算不符：" + str(bad)
    return len(bad)


def shortstat_check():
    """shortstat 的三个数必须等于 numstat 逐行现算之和（S 类结构重算）。

    为什么不能只信形状正则：git 在 1 条时使用单数拼写，正则少写一个 `?` 就会让那一组**静默
    不参与**、取到 None，再被 `or "0"` 折成一个看起来完全合理的 0。闭合到 numstat 才让这种
    漏项变成红。二进制／重命名行在 numstat 里是 `-`，求和会跳过 ⇒ 先拒绝这种行。
    """
    raw = sh("git", "diff", "--cached", "--shortstat").strip()
    sm = RE_SHORTSTAT.search(raw)
    assert sm, "shortstat 形状不认识：%r" % raw
    ns = [l.split("\t") for l in sh("git", "-c", "core.quotePath=false", "diff", "--cached",
                                    "--numstat").splitlines() if l.strip()]
    dash = [r[-1] for r in ns if "-" in (r[0], r[1])]
    assert not dash, "numstat 里有 `-` 行（二进制／rename）%s ⇒ 逐行求和口径不成立" % dash
    files, ins, dels = int(sm.group(1)), int(sm.group(2)), int(sm.group(3) or 0)
    assert files == len(ns), "shortstat 文件数 %d ≠ numstat 现算行数 %d" % (files, len(ns))
    assert ins == sum(int(r[0]) for r in ns), \
        "shortstat 增加数 %d ≠ numstat 现算 %d" % (ins, sum(int(r[0]) for r in ns))
    assert dels == sum(int(r[1]) for r in ns), \
        "shortstat 删除数 %d ≠ numstat 现算 %d ⇒ 单复数取数漏项／分组没吃到" % (
            dels, sum(int(r[1]) for r in ns))
    return raw, files, ins, dels


def build():
    assert sys.executable.startswith("/tmp/amvenv"), "解释器不是 /tmp/amvenv 而是 %s" % sys.executable
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "生成器自身未带空串 CVD ⇒ 钉住口径失真"
    rows, added, modified = staged()
    buckets = bucketize(added)
    assert not [p for p in added + modified if p.split("/")[0] in ("src", "tests")], \
        "暂存集里有 src／tests 改动 ⇒『本轮无代码腿』这句成了假话，拒绝生成"
    n_code = len([l for l in sh("git", "-c", "core.quotePath=false", "diff", "--cached", "--name-only",
                                "--", "src", "tests").splitlines() if l.strip()])
    assert n_code == 0, "src/tests 现数 %d 条改动 ≠ 0" % n_code
    sraw, ss_files, ss_ins, ss_dels = shortstat_check()
    curl = gate_curl(rd(CURL))
    reg = parse_reg(rd(REG))
    facts = dict(l.split("=", 1) for l in rd(FACTS).splitlines() if "=" in l)
    finger_now = subprocess.run(["bash", "-c", FINGER_CMD], cwd=REPO, capture_output=True,
                                text=True).stdout.strip()
    # 四路指纹闭合：现树 == 回归起 == 回归止 == 上一片回执表头 ⇒「引用那次回归」才有凭据
    assert len({finger_now, reg["fs"], reg["fe"], curl["finger"]}) == 1, \
        "指纹四路不齐：现树 %s／回归起 %s／回归止 %s／回执 %s" % (
            finger_now, reg["fs"], reg["fe"], curl["finger"])
    assert int(curl["dirty_st"]) == 0, "回执表头 dirty(src+tests)=%s ≠ 0" % curl["dirty_st"]
    copies = sum(1 for p in glob.glob(os.path.join(REPO, "docs/evidence/*/am_s7_verify*.py"))
                 if hashlib.md5(open(p, "rb").read()).hexdigest() == md5(RULER))
    assert copies >= 2, "尺子副本普查落空（glob 没命中 ⇒ 计数会假绿）"
    assert md5(RULER, 12) == curl["ruler_md5"], "尺子 md5 与回执表头声称的不符 ⇒ 不是同一把"
    census, audit_verdict = audit_check()
    ctl_rows = controls_check()
    wrows, wrep = writer_check(modified)
    hdr_fields = fields_of(HDR26)
    cmp_rcs, n_defs, n_fields = cmp_check(curl, hdr_fields)
    assert str(copies) == curl["copies"], "现算尺子份数 %d ≠ 回执表头声称 %s" % (copies, curl["copies"])
    secn, letters, sech, secs = devlog_check()
    w26_cross_check(wrep, curl, copies, n_defs, n_fields, cmp_rcs)
    head_now = sh("git", "rev-parse", "HEAD").strip()
    assert curl["head40"] == head_now, "回执的落地树 %s ≠ 本地 HEAD %s" % (curl["head40"], head_now)
    dirty_src = len([l for l in sh("git", "-c", "core.quotePath=false", "status", "--porcelain",
                                   "--", "src", "tests").splitlines() if l.strip()])
    wt = RE_W_TREE.search(wrep)
    assert wt, "写手 stdout 里取不到『树：』汇总行"
    assert wt.group(6) == wt.group(7) == finger_now, "写手那轮的 SRC 指纹两路不同或 ≠ 现树 ⇒ 回归引用断裂"
    assert wt.group(5) == str(dirty_src), "写手那轮 dirty(src+tests)=%s ≠ 现算 %d" % (wt.group(5), dirty_src)
    corr_total = sum(r["corr"] for r in wrows)
    since_head = sum(r["since_head"] for r in wrows)
    writer_total = sum(r["delta"] for r in wrows)
    pre_total = sum(r["pre"] for r in wrows)
    assert writer_total + pre_total == since_head, "五面行数分解不闭合：%d＋%d ≠ %d" % (
        writer_total, pre_total, since_head)
    wb, wd = RE_W_BYTES.search(wrep), RE_W_DET.search(wrep)
    assert wb and wd, "写手 stdout 里取不到字节闭合／检测器汇总行"
    assert corr_total == int(wb.group(7)), "五面更正腿之和 %d ≠ 汇总行声称的锚点处数 %s" % (
        corr_total, wb.group(7))
    nv = [l.strip() for l in subprocess.run(["nvidia-smi", "--query-compute-apps=pid",
                                             "--format=csv,noheader"], capture_output=True,
                                            text=True).stdout.splitlines() if l.strip()]
    nv_mine = [p for p in nv if "/tmp/amvenv" in open("/proc/%s/cmdline" % p, "rb").read()
               .decode("utf-8", "replace")]
    assert nv_mine == [], "我的解释器在 GPU 上留有进程 %s ⇒ 与「不动 GPU」冲突" % nv_mine
    byrel = {r["rel"]: r for r in wrows}
    b = buckets
    V = {
        "SLICE": "27", "PREVSLICE": "26", "SECN": secn, "NBULLET": str(len(letters)),
        "BULLETS": letters[0] + "–" + letters[-1],
        "NA": str(len(added)), "NM": str(len(modified)), "NTOT": str(len(rows)),
        "SHORTSTAT": sraw,
        "NCODE": str(n_code), "DIRTYSRC": str(dirty_src),
        "NCH26": str(len(b["分片 26 回执链"])), "NCTL26": str(len(b["分片 26 表头控件面"])),
        "NA3": str(len(b["A3 预检面（腿①）"])), "NAUD": str(len(b["表头溯源审计仪器"])),
        "NDOC": str(len(b["归档写手"])), "NMSG": str(len(b["信息生成器与干跑控件"])),
        "BUCKETSUM": str(sum(len(v) for v in b.values())),
        "HEAD": head_now[:7], "PARENT": curl["parent"][:7], "PARENT40": curl["parent40"],
        "REMOTE": curl["remote"], "EXPECT": curl["expect"], "ENTRIES": curl["entries"],
        "BLOB": curl["blob"], "TREEENT": curl["tree"], "NFILE": str(curl["nfile"]),
        "MATCH": str(curl["match"]), "A26": str(curl["a"]), "M26": str(curl["m"]),
        "DL": curl["dl"], "DLDEN": curl["dlden"], "DR": curl["dr"], "DRDEN": curl["drden"],
        "SHA": curl["sha"], "SHADEN": curl["shaden"], "VERDICT": curl["verdict"],
        "CA0": curl["ctl_a"][0], "CA1": curl["ctl_a"][1], "CB1": curl["ctl_b"][0],
        "CB1D": curl["ctl_b"][1], "CB2": curl["ctl_b"][2], "CB2D": curl["ctl_b"][3],
        "CC_L": curl["ctl_c"][0], "CC_R": curl["ctl_c"][1], "CC_V": curl["ctl_c"][2],
        "CD1": curl["ctl_d"][0], "CD1D": curl["ctl_d"][1], "CD2": curl["ctl_d"][2],
        "CD2D": curl["ctl_d"][3], "BOFF": curl["boff"], "CMP": "/".join(str(x) for x in cmp_rcs),
        "NDEF": str(n_defs), "NFIELDS": str(n_fields), "COPIES": str(copies),
        "NEXTCOPIES": str(copies + 1), "RULERMD5": md5(RULER, 12),
        "FINGER": finger_now, "REG": REG, "REGMD5": md5(REG, 12),
        "REGHEAD": reg["head"], "REGDIRTY": reg["dirty"], "TALLY": reg["tally"],
        "RC": RE_RC.search(rd(REG)).group(1), "NF": str(reg["nf"]), "NP": str(reg["npass"]),
        "NS": str(reg["nskip"]), "WALL": reg["wall"], "D1": reg["d1"], "D2": reg["d2"],
        "E1": reg["elines"][0].strip(), "E2": reg["elines"][1].strip(),
        "XFAIL": facts["XFAIL"], "COLLECT": facts["COLLECT"],
        "AUDITN1": census[0][0], "AUDITN2": census[1][0], "AUDITVERDICT": audit_verdict,
        "AUDITREL": AUDIT, "CTLREL": CTL, "CTLN": str(len(ctl_rows)),
        "CTLMAIN": str(sum(1 for r in ctl_rows if r[1] == "主面")),
        "CTLCTL": str(sum(1 for r in ctl_rows if r[1] == "控件面")),
        "DL0": byrel[FIVE[0]]["old"], "DL1": byrel[FIVE[0]]["new"], "DLD": byrel[FIVE[0]]["delta"],
        "DC": str(byrel[FIVE[0]]["corr"]),
        "PL0": byrel[FIVE[1]]["old"], "PL1": byrel[FIVE[1]]["new"], "PLD": byrel[FIVE[1]]["delta"],
        "PC": str(byrel[FIVE[1]]["corr"]),
        "OV0": byrel[FIVE[2]]["old"], "OV1": byrel[FIVE[2]]["new"], "OVD": byrel[FIVE[2]]["delta"],
        "OC": str(byrel[FIVE[2]]["corr"]),
        "DY0": byrel[FIVE[3]]["old"], "DY1": byrel[FIVE[3]]["new"], "DYD": byrel[FIVE[3]]["delta"],
        "YC": str(byrel[FIVE[3]]["corr"]),
        "MEM0": byrel[FIVE[4]]["old"], "MEM1": byrel[FIVE[4]]["new"], "MEMD": byrel[FIVE[4]]["delta"],
        "MC": str(byrel[FIVE[4]]["corr"]),
        "CORRTOT": str(corr_total), "SECH": str(sech), "SECSTAGED": str(secs),
        "SINCEHEAD": str(since_head), "WRITERTOT": str(writer_total), "PRETOT": str(pre_total),
        "BC1": wb.group(1), "BC2": wb.group(2), "BC3": wb.group(3), "BC4": wb.group(4),
        "BC5": wb.group(5), "BC6": wb.group(6), "BC7": wb.group(7), "BC8": wb.group(8),
        "BC9": wb.group(9), "BC10": wb.group(10), "BC11": wb.group(11),
        "DET1": wd.group(1), "DET2": wd.group(2), "DET3": wd.group(3), "DET4": wd.group(4),
        "DET5": wd.group(5), "DET6": wd.group(6), "DET7": wd.group(7),
        "WT1": wt.group(1), "WT2": wt.group(2), "WT3": wt.group(3), "WT4": wt.group(4),
        "WT5": wt.group(5), "WT8": wt.group(8), "WT9": wt.group(9), "WT10": wt.group(10),
        "NV": str(len(nv)), "NVME": str(len(nv_mine)), "NOW": sh("date", "+%F %T %z").strip(),
        "WRAWMD5": md5(WRITER_RAW, 12), "ARAWMD5": md5(AUDIT_RAW, 12), "CRAWMD5": md5(CTL_RAW, 12),
        "AUDITMD5": md5(AUDIT, 12), "CTLMD5": md5(CTL, 12), "WRITERMD5": md5(WRITER, 12),
        "CURLMD5": md5(CURL, 12), "SELFMD5": md5(os.path.join(HERE, SELF), 12), "WRITERREL": WRITER,
        "CURL": CURL, "SELF": SELF,
    }
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
                  bucket_sum=sum(len(v) for v in b.values()), ruler_copies=copies,
                  census_main=census[0], census_ctl=census[1], ctl_rows=len(ctl_rows),
                  writer_faces=len(wrows), cmp_rcs=cmp_rcs, header_fields=n_defs,
                  corrections=corr_total, bullets=len(letters), section=secn,
                  shortstat=(ss_files, ss_ins, ss_dels),
                  finger=finger_now[:12], gpu_mine=len(nv_mine), gpu_total=len(nv))
    return msg, counts


TMPL = """S7 分片 27：归档腿＝把 §26.29 (i) 登记的"手工表头债"折成**表头数值溯源审计仪器**（五类出处闸门＋十条可红输入扰动）＋分片 26 的落地回执链入库；开发日志 §26.@@SECN@@

本片＝分片 @@SLICE@@，上一片＝分片 @@PREVSLICE@@；信息本身由 `@@SELF@@`（md5 @@SELFMD5@@，生成于 @@NOW@@）从命令现取，
不是手打。入库面：新增 @@NA@@ 件＋修改 @@NM@@ 件＝@@NTOT@@ 件（`git diff --cached --shortstat` 原文
「@@SHORTSTAT@@」，三个数已与 `--numstat` 逐行现算闭合；首版在这里按复数形状取数，把 git 的单数
「1 deletion(-)」静默读成 0，靠这条闭合才露出来）。**src／tests 本轮 0 件改动**
（`git diff --cached --name-only -- src tests` 现数 @@NCODE@@ 条；工作树 src+tests 脏项＝@@DIRTYSRC@@）。
新增件按**角色**分六桶，桶和＝@@BUCKETSUM@@＝新增总数（一件落进 0 个或 ≥2 个桶都拒绝生成）：
分片 26 回执链 @@NCH26@@ 件／分片 26 表头控件面 @@NCTL26@@ 件／§26.@@SECN@@ 的 A3 预检面 @@NA3@@ 件／
本轮表头溯源审计仪器 @@NAUD@@ 件／本轮归档写手 @@NDOC@@ 件／本信息生成器与干跑控件 @@NMSG@@ 件。
修改 @@NM@@ 件＝五张归档面：开发日志 @@DL0@@→@@DL1@@（＋@@DLD@@，更正腿 @@DC@@）、计划文档 @@PL0@@→@@PL1@@（＋@@PLD@@，
@@PC@@）、总览 @@OV0@@→@@OV1@@（＋@@OVD@@，@@OC@@）、当日记忆 @@DY0@@→@@DY1@@（＋@@DYD@@，@@YC@@）、
MEMORY @@MEM0@@→@@MEM1@@（＋@@MEMD@@，@@MC@@）。这些数**两个出处**：写手 stdout（@@WRITERREL@@ 那次运行，
归档 _raw_s27_docs.out md5 @@WRAWMD5@@）与 `git show :<面>` 的暂存 blob 行数现算逐面相等，且每面满足
`numstat 净增 ＝ 暂存行数 − HEAD 行数`；另有 `git diff --name-only` 为空＝无未暂存漂移。开发日志本轮节＝§26.@@SECN@@，
子条 @@BULLETS@@ 共 @@NBULLET@@ 条（字母连续性由**暂存 blob** 现取现验，不读工作树）。
**一处必须说出口的意外**（首版在这里断言"写手旧行数＝HEAD 实得"，把**正确**的仪器判红、被干跑控件抓出）：
写手的"改前"是**工作树**而不是 HEAD——上一轮 §26.@@SECN@@ (a)–(k) 的归档当时没被任何提交带走
（命令可证：`### 26.@@SECN@@` 在 HEAD 里 @@SECH@@ 次、在暂存 blob 里 @@SECSTAGED@@ 次 ⇒ 本节整体随本片落地）。
⇒ 改成可加的三段分解并断言：五面对 HEAD 净增合计 @@SINCEHEAD@@ 行＝本轮写手 @@WRITERTOT@@ 行＋写手动笔前
工作树里已有的 @@PRETOT@@ 行（逐面同式，见生成器 writer_check）。

本轮无代码腿 ⇒ **没重跑全量回归**（重跑只会再量同一棵树），引用分片 26 那次的凭据是**指纹四路闭合**：
现算 src+tests 指纹＝@@FINGER@@＝回归件 SRCFINGER_START＝SRCFINGER_END＝上一片回执表头 SRC指纹
（四路任一路断裂即拒绝生成信息）。被引用的那次：**@@TALLY@@**（＝@@NF@@ failed＋@@NP@@ passed＋@@NS@@ skipped），
墙钟 @@WALL@@，rc=@@RC@@（rc=1＝两条既有红灯，非本轮引入），@@D1@@ → @@D2@@，
当时的树＝@@REGHEAD@@ dirty=@@REGDIRTY@@（内容级＝现树；提交级不同：那次跑在当时的未提交工作树上）。
两条登记红的 E 行（逐字节从 @@REG@@ md5 @@REGMD5@@ 现取）：①@@E1@@；②@@E2@@。
`grep -ic 'xfail|xpass'`＝@@XFAIL@@、`--collect-only`＝@@COLLECT@@ ⇒ A0 断言未放宽、#27 未换指标、无新增 skip/xfail
（这三条本轮只**引用**那次运行的读数，因为本轮没有新测试可与之比）。

分片 26 回执入库（@@CURL@@，md5 @@CURLMD5@@；读数一律从回执正文＋表头正则现取）：@@VERDICT@@；
落地树 远端 main＝本地 HEAD＝`@@HEAD@@`（父 `@@PARENT@@`＝@@PARENT40@@）；
tree 条目 @@ENTRIES@@＝blob @@BLOB@@＋tree @@TREEENT@@（truncated=False）；
闭合式 远端 @@REMOTE@@ == 父提交 @@PARENT@@ ＋ 新增(A) @@A26@@ ＝ @@EXPECT@@ -> PASS；
双向路径差集 @@DL@@/@@DLDEN@@ 与 @@DR@@/@@DRDEN@@；全库 blob SHA 不符 @@SHA@@/@@SHADEN@@；
本片单列复核 @@MATCH@@/@@NFILE@@ MATCH（A @@A26@@／M @@M26@@）。四个正对照同一条运行里都能红：
A 远端一条 sha 首位改 f ⇒ 不符 @@CA0@@/@@CA1@@；B 远端删一条路径 ⇒ 本地独有 @@CB1@@/@@CB1D@@、远端独有 @@CB2@@/@@CB2D@@；
C 闭合式右边 +1 ⇒ @@CC_L@@ == @@CC_R@@ 判 @@CC_V@@；D 远端删**本片**一条路径 ⇒ 本片缺失 @@CD1@@/@@CD1D@@、SHA 不符 @@CD2@@/@@CD2D@@。
回执自身的两条**本轮复跑**（不是引用它自己的声称）：`cmp -i @@BOFF@@:0` 三件套 rc＝@@CMP@@（同偏移须 0、任一侧 +1 须非 0）
⇒ 正文区是那次运行 stdout 的逐字节后缀；表头顶格字段行数由 `am_s26_hdr.py` 的 `FIELDS`（ast 现取、不执行）反查＝@@NDEF@@/@@NFIELDS@@。
尺子 md5 前 12＝@@RULERMD5@@ 与回执表头声称相同；**全树** glob＋md5 现算＝同一把尺子已有 @@COPIES@@ 份逐字节副本
（含本轮入库的这份）；本片下一把副本要到推送之后才现抄 ⇒ 那时才是 @@NEXTCOPIES@@ 份，此处不预先声称。

审计腿（本轮核心交付，@@AUDITREL@@ md5 @@AUDITMD5@@）：表头里每个读数必须有五类出处之一
（C 命令读数／D 器件常量／B 本面正文／X 跨面正文且引用句自带出处词／S 结构式重算），查不到即 rc=3。
本轮**复跑**该件（rc=0、stderr 0 字节）：主面表头数字 @@AUDITN1@@ 个、控件面 @@AUDITN2@@ 个，
两面**无出处 0／跨面无归属 0／无出处 hex 0** ⇒ @@AUDITVERDICT@@；复跑与档案 _raw_a3_hdr_audit.out（md5 @@ARAWMD5@@）
在"总数＋三类红旗"这四个数上逐位相同。**复现口径本轮收窄一次**：普查行的**类别分布**（B/C/D/S/X 各几个）随
证据目录内容移动（出处池＝整个目录，本轮新入库 @@NA@@ 件就可能让同一个数从 B 改判 C），把分布当"复现"来断言
会造出一条对不齐的闸门 ⇒ 只钉总数＋三类红旗，类别分布只进写手 stdout、不入档（§26.@@SECN@@ (l)）。
正对照（@@CTLREL@@ md5 @@CTLMD5@@）本轮**也复跑**：@@CTLN@@ 条**输入**扰动（主面 @@CTLMAIN@@／控件面 @@CTLCTL@@），
每条判据＝rc≠0 **且**命中指定闸门句，未扰动基线先 rc=0 且判读行全过 ⇒ 十条各自命中
（`CONTROL_VERDICT = ALL TEN PERTURBATIONS GO RED ON THEIR OWN GATE`，复跑与归档 _raw_a3_hdr_audit_controls.out
md5 @@CRAWMD5@@ 的判读行逐字节相同）。

归档腿（@@WRITERREL@@ md5 @@WRITERMD5@@）：字节闭合＝主面 @@BC1@@＋@@BC2@@=@@BC3@@／控件面 @@BC4@@＋@@BC5@@=@@BC6@@
（表头＋原始正文＝落盘回执，`cat | cmp` rc=0）；更正锚点 @@BC7@@ 处（＝五面更正腿之和 @@DC@@+@@PC@@+@@OC@@+@@YC@@+@@MC@@=@@CORRTOT@@，
两路同数）；旧值按**词边界**查＝历史面 @@BC8@@／改前 @@BC9@@／诱饵 @@BC10@@／子串 @@BC11@@（诱饵是闭式解里的 0.944860，
不是过期值，故不计进"未自证"）。过期值检测器三面：历史面（`git show HEAD:`）@@DET1@@／改前工作树 @@DET2@@／
**改后全文面**未自证 @@DET3@@｜标本（该被抓住的）@@DET4@@／抹掉真值后红 @@DET5@@｜自纠累计 @@DET6@@（本轮 +@@DET7@@）。
全文面是**自指**的（新条目里就引着那个旧数）⇒ 写手对最终落盘文本做不动点迭代，写盘后再对磁盘上的五面复测一次。
写手那轮的树＝HEAD @@WT1@@（父 @@WT2@@）｜dirty total @@WT3@@／evidence @@WT4@@／src+tests @@WT5@@｜GPU @@WT8@@ 个、我的 @@WT9@@/@@WT10@@。

诚实登记（本轮没做什么）：① 无代码腿、无新测试、无回归重跑；② A3 的形态数字本轮为零（预检面只验接线，
`PREFLIGHT_VERDICT` 那句原文＝"本件未解，零条形态数字"）；③ **性能数字本轮为零**——GPU 仍停放，吞吐/加速/预算类
数字只在 A6000 上实测，CPU 侧只报算法与正确性；④ §26.29 那两张 A3 面的**手工表头没有被重写**（重写会让 (k) 里
已归档的 md5／字节数声称再次过期）⇒ 债的折法是"事后审计"而不是"改史"；⑤ **本片（分片 @@SLICE@@）自己的落地证明
要到下一轮才存在**：本信息只声称分片 @@PREVSLICE@@ 那一份已落地（回执正文 ①–⑤ 段），不预先声称本片被复核过。
GPU：全轮 CPU 钉住（每条计算命令内联 `CUDA_VISIBLE_DEVICES=` ＋ `JAX_PLATFORMS=cpu`，含一行命令）；
生成信息时现取卡上 compute 进程 @@NV@@ 个、我的解释器（/proc/PID/cmdline 含 /tmp/amvenv）＝@@NVME@@（非 0 拒绝写盘）；
钉住口径出处＝开发日志 §26.@@SECN@@ 那句「不动 GPU」。
排期：#27 腿②（缺省 tuned→hk 切换）＋#33 的 15 位点系数重标定**同批**，且只对 A3 的 18 道外部趋势靶打分
（单道散度 width p50 15.7%／area 19.8%，n=6；过线判据 14%／11%）⇒ 随 #10/A3；A3 下一步 `--stage smoke`
（dx=r_b，按 J6 只验接线、不出形态数字）→ `--stage pilot --tracks 1,2,3 --arms both` →
`--stage full --measured-per-track`（ΣN=1..18 对 BUDGET_S=14400 s）；#40 → #39；#41 待自己一轮；#29／#30 仍等用户裁决。
"""

if __name__ == "__main__":
    m, c = build()
    open("/tmp/am_s27_msg.txt", "w", encoding="utf-8").write(m)
    print(m)
    print("---- 现取核对 ----")
    for k, v in c.items():
        print("%s=%s" % (k, v))
