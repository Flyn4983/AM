#!/usr/bin/env python3
"""S7 分片 28 提交信息生成器：信息里的每个数字都从命令／档案现取，不许手敲。

本轮**没有代码腿**（`git diff --cached --name-only -- src tests` 现数 0 件）⇒ 不写"改了哪个求解器"，
改为把"在网回归仍是分片 26 那一份、且它量的就是现树"用**指纹四路闭合**证出来：
现算 src+tests 指纹 ＝＝ 回归件 SRCFINGER_START ＝＝ SRCFINGER_END ＝＝ 上一片回执表头 SRC指纹，
四路同值才允许在信息里引用那次回归；任一路断裂即拒绝生成（不是降级成叙述）。
这条闸门**能过**是有事实基础的：本轮只动 docs/evidence 与归档面。

口径沿用（不是本轮新增）：
① 正文一律 **PLACEHOLDER＋str.replace**，不用 f-string（§26.28 (q)：f-string 会把字面量 `{32}` 当替换
   字段**静默吃掉**）⇒ 生成后**双向**查：模板里没取值的字段拒绝，取值表里模板没用到的读数也拒绝。
② 复现声称只钉**结构上钉得住的读数**（总数＋红旗类），类别分布随目录移动。
③ 引命令**原文**并把三个数闭合到 `--numstat` 逐行现算（git 对 1 用单数「1 deletion(-)」，按复数形状取数
   会把它静默折成 0）。
④ 每个 0 与它的分母同印；每条判据都要在控件面（`am_s28_msg_controls.py`）里被**坏输入**打红过。

本轮新增的第 ⑤ 条口径（写在这里，因为它改变了取数方式）：**基准读数的"可评分面"必须由闸门决定，
不能由我挑**。上一轮 G9 之前的版本把触边行数只印在 SOLVE_VERDICT 里、不入 all_pass ⇒ 若照旧引用，
信息里就会把 17/18 个被边界截断的读数当测量值写进来。所以本件对 A3 面的取数**只允许**来自
`[PASS] G9` 的那一条道次，其余道次的宽／积只以"触边 N／M 行"的形态出现（下界，不是读数）。
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

REG = E + "/_raw_t27_full.log"
FACTS = E + "/am_t27_facts.txt"
CURL = E + "/am_s7_verify27.py".replace("verify27.py", "verify27.log")   # 上一片回执（本轮入库）
RULER = E + "/am_s7_verify27.py"
HDR27 = E + "/am_s27_hdr.py"
RAW27 = E + "/_raw_verify27.out"
RAW27ERR = RAW27[:-4] + ".err"
PILOT = E + "/_raw_a3_pilot_amb_g9_norm_t123.out"
PILOT_PRE = E + "/_raw_a3_pilot_amb_norm_t123.out"
RAWN1 = E + "/_raw_a3_pilot_amb_raw_N1.out"
SMOKE = E + "/_raw_a3_smoke_g8fix_card300.out"
SMOKE_AMB = E + "/_raw_a3_smoke_g8fix_ambk29315.out"
PRE_REG = E + "/_raw_a3_preflight_regress.out"
PRE_REG_N = E + "/_raw_a3_preflight_regress_norm.out"
PRE_ORIG = E + "/_raw_a3_preflight.out"
ICLEG = {L: E + "/_raw_a3ic_%s.out" % L for L in "ABCDEF"}
SELF = "am_s28_msg.py"
DEVLOG = "docs/开发日志.md"
# 归档面：本轮**实际**改了哪几张由 git 现取，下面只做"该有的必须在场"的断言（不硬数）
DOC_FACES = [DEVLOG, "docs/项目评估与下一步计划_2026-10-06.md", "docs/项目开发总览.md",
             ".workbuddy/memory/2026-10-10.md"]
INSTRUMENT_FACES = [E + "/am_a3_driver.py", E + "/am_a3_run.sh"]
LEDGER = ".workbuddy/memory/MEMORY.md"
DRIVER = E + "/am_a3_driver.py"
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
# 上一片回执：表头来历读数＋正文读数（两出处互证）
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
# A3 面：道次头／solidus 行／G9 行／SOLVE_VERDICT／代价行／趋势行
RE_TRK = re.compile(r"^\[\w+/\w+\] \w N=(\d+) dx=([\d.]+)µm .*?nvox=(\d+) n_steps=(\d+) "
                    r"voxel_steps=([\d.e+]+)", re.M)
RE_SOLIDUS = re.compile(r"^\s+(?:\[PROBE\] )?solidus: 宽=([\d.]+)µm .*?积=(\d+)µm².*?"
                        r"行数=(\d+) 有熔=(\d+) 触边=(\d+)(.*)$", re.M)
RE_DEVW = re.compile(r"靶 width_um: 中位 ([\d.]+)（n=(\d+)，极差 ([\d.]+)%，([\d–]+)）⇒ 偏差 ([\d.]+)%")
RE_DEVA = re.compile(r"靶 area_um2: 中位 ([\d.]+)（n=(\d+)，极差 ([\d.]+)%，([\d–]+)）⇒ 偏差 ([\d.]+)%")
# 达标线／未达线也从面里现取（J7 的机器口径写在 BARS，面把同一对数印出来）⇒ 信息不许自带 14/11/20
RE_BARW = re.compile(r"靶 width_um:.*?偏差 [\d.]+%（≤([\d.]+)% 达标／>([\d.]+)% 未达）")
RE_BARA = re.compile(r"靶 area_um2:.*?偏差 [\d.]+%（≤([\d.]+)% 达标／>([\d.]+)% 未达）")
RE_G9 = re.compile(r"\[(PASS|FAIL)\] G9 .*?max touched=(\d+)／窗口行数=(\d+)", re.M)
RE_VERD = re.compile(r"SOLVE_VERDICT = (\w+) wall=([\d.]+)s arm=(\w+)")
RE_COST = re.compile(r"代价实测[^：]*：最大 N=(\d+) 墙钟 ([\d.]+)s ⇒ 每道均摊 ([\d.]+)s；"
                     r"(\d+) 道逐 N 之和 ≈ (\d+)s（预算 (\d+)s）⇒ (\S+)")
RE_TREND = re.compile(r"趋势（norm 臂，宽度 µm）：(\[[^\]]*\]) 非降=(\w+) 末/首=([\d.]+)")
RE_G8 = re.compile(r"\[(PASS|FAIL)\] G8a .*?缺口 ([+\-][\d.]+)K")
RE_G8B = re.compile(r"\[(PASS|FAIL)\] G8b ")
# 面内的"跑前定死"行＋夹具行：把 dx=r_b/2、amb、tracks、arms、α 全部**从面里现取**并互证
RE_SEL = re.compile(r"本轮选用（跑前定死）：stage=(\w+) strategy=(\w+) tracks=\[([\d,\s]+)\] α=([\d.]+)"
                    r".*?dx=([\d.]+)µm arms=\['(\w+)'\]")
RE_AMB = re.compile(r"^材料卡现取：T_solidus=[\d.]+K T_liquidus=[\d.]+K T_ambient=([\d.]+)K", re.M)
RE_RB = re.compile(r"r_b=([\d.]+)µm")
RE_CTL4 = re.compile(r"^\s+\[(PASS|FAIL)\] (C[1-4]) ", re.M)
RE_SOFTLAB = re.compile(r"^\s+\[SOFT\]\s+([A-Z]+\d+)", re.M)
RE_CTLN = re.compile(r"⑤ 正对照 ([A-Z])", re.M)
# §26.11(c) 合并口径散差表（组数／p50／p90／max）——排期那句引用的出处
RE_P50W = re.compile(r"^\| 熔池宽度 \| (\d+) \| \*\*([\d.]+)%\*\* \| \*\*([\d.]+)%\*\* \| ([\d.]+)% \|", re.M)
RE_P50A = re.compile(r"^\| 顶面面积 \| (\d+) \| \*\*([\d.]+)%\*\* \| \*\*([\d.]+)%\*\* \| ([\d.]+)% \|", re.M)
RE_SHORTSTAT = re.compile(r"(\d+) files? changed(?:, (\d+) insertions?\(\+\))?(?:, (\d+) deletions?\(-\))?")
RE_SEC = re.compile(r"^### 26\.(\d+)[^\n]*$", re.M)
RE_BULLET = re.compile(r"^- \*\*(.+?)\*\*", re.M)
# 开发日志里那句"未入库面"的登记（本轮的入库面数由它推到暂存实数，差额只许是本件自己的四张面）
RE_UNTRACKED = re.compile(r"未入库＝未跟踪 \*\*(\d+)\*\*＋修改 \*\*(\d+)\*\*＝\*\*(\d+)\*\*")

# 新增件按**角色**分桶（桶和必须＝新增总数；落不进桶或落进两个桶都拒绝生成）
BUCKETS = [
    ("分片 27 回执链", r"^am_s7_verify27\.(py|log)$|^_raw_verify27\.(out|err)$|^am_s27_hdr\.py$"),
    ("分片 27 表头控件面＋归档自查面", r"^am_s27_hdr_controls\.py$|^am_s27_archive_check\.py$"
                                        r"|^_raw_s27_hdr_(controls|write)\.(out|err)$"
                                        r"|^_raw_s27_archive_check\.(out|err)$"),
    ("A3 启动记录", r"^am_a3_ic_clamp_probe\.sh$"),
    ("A3 #51 六腿探针面", r"^_raw_a3ic_"),
    ("A3 pilot 读数面", r"^_raw_a3_pilot_amb"),
    ("A3 smoke 与预检回归面", r"^_raw_a3_smoke_|^_raw_a3_preflight_regress"),
    ("本信息生成器与干跑控件", r"^am_s28_msg(_controls)?\.py$|^_raw_s28_msg_controls\.(out|err)$"),
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
    assert sum(len(v) for v in hits.values()) == len(added), "桶和 %d ≠ 新增总数 %d" % (
        sum(len(v) for v in hits.values()), len(added))
    return hits


def numstat_closure():
    """shortstat 原文 ↔ numstat 逐行现算，两出处必须给同一组三个数。"""
    sraw = sh("git", "-c", "core.quotePath=false", "diff", "--cached", "--shortstat").strip()
    m = RE_SHORTSTAT.search(sraw)
    assert m, "shortstat 形状取不出三个数：" + sraw
    ss = tuple(int(x) if x else 0 for x in m.groups())
    rows = [l.split("\t") for l in sh("git", "-c", "core.quotePath=false", "diff", "--cached",
                                      "--numstat").splitlines() if l.strip()]
    assert all(r[0] != "-" for r in rows), "numstat 里有二进制行（`-`）⇒ 逐行求和口径不成立"
    ns = (len(rows), sum(int(r[0]) for r in rows), sum(int(r[1]) for r in rows))
    assert ss == ns, "shortstat %s ≠ numstat 现算 %s ⇒ 有一侧取数形状错（首版栽在单数『1 deletion(-)』）" % (
        ss, ns)
    return sraw, ss


def parse_reg(raw):
    t = RE_TALLY.search(raw)
    assert t, "回归件里取不到汇总行"
    tm = RE_TREE.search(raw)
    assert tm, "回归件里取不到 TREE 行"
    heads = dict(RE_FINGER.findall(raw))
    assert set(heads) == {"START", "END"}, "SRCFINGER 只取到 %s" % sorted(heads)
    assert heads["START"] == heads["END"], "SRCFINGER 起≠止 ⇒「回归期间源码一字未动」不成立"
    dates = dict(RE_DATE.findall(raw))
    elines = RE_ELINES.findall(raw)
    assert len(elines) == 2, "两条红的 E 行实取 %d 条 ⇒ 判据形状已变，不许静默通过" % len(elines)
    return dict(tally=t.group(0), nf=int(t.group(1)), npass=int(t.group(2)), nskip=int(t.group(3)),
                wall=t.group(4) + "s", head=tm.group(1), dirty=tm.group(2), fs=heads["START"],
                fe=heads["END"], d1=dates["START"], d2=dates["END"], elines=elines,
                rc=RE_RC.search(raw).group(1))


def gate_prev_receipt(text):
    """上一片回执的闸门：读数全部现取，任一条不绿就拒绝引用它。"""
    close = RE_C_CLOSE.search(text)
    assert close, "回执里取不到闭合式"
    remote, parent, na, expect, verdict_close = close.groups()
    assert int(remote) == int(parent) + int(na) == int(expect), \
        "闭合式在本件里不成立（%s/%s/%s/%s）⇒ 不许引用" % (remote, parent, na, expect)
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


def fingerprint_closure(live, reg, curl):
    """四路同值：现算＝回归起＝回归止＝上一片回执表头。返回被比较的第四个出处（回执名）。"""
    assert reg["fs"] == reg["fe"], "回归件起≠止"
    assert live == reg["fs"] == reg["fe"] == curl["finger"], (
        "指纹四路不闭合：现算 %s／回归起 %s／止 %s／回执表头 %s ⇒ 不许引用那次回归" % (
            live, reg["fs"], reg["fe"], curl["finger"]))
    return True


def solidus_lines(text):
    """RE_SOLIDUS 命中行的**整行原文**（逐字节比较用；只取到该行行尾，不含下一行）。"""
    outs = []
    for m in RE_SOLIDUS.finditer(text):
        nl = text.find("\n", m.start())
        outs.append(text[m.start():(len(text) if nl == -1 else nl)])
    return outs


def parse_pilot(text):
    """把 pilot 面解析成 per-track 记录；闸门要求 solidus 行与 SOLVE_VERDICT／G9 行**同轨**配对。"""
    trk = RE_TRK.findall(text)
    sol = [m.groups() for m in RE_SOLIDUS.finditer(text)]
    g9 = RE_G9.findall(text)
    ver = RE_VERD.findall(text)
    assert len(trk) == len(sol) == len(g9) == len(ver), (
        "pilot 面四类行数不等（道次 %d／solidus %d／G9 %d／判决 %d）⇒ 取数口径变了，拒绝配对" % (
            len(trk), len(sol), len(g9), len(ver)))
    out = []
    for (n, dx, nvox, nsteps, vsteps), s, (ok9, touched, nrows), (vword, wall, arm) in zip(
            trk, sol, g9, ver):
        r = dict(n=int(n), dx=float(dx), nvox=int(nvox), nsteps=int(nsteps),
                 vsteps=vsteps, width=float(s[0]), area=int(s[1]), nrows=int(s[2]),
                 nmelt=int(s[3]), touched=int(s[4]), g9=ok9, g9_touched=int(touched),
                 g9_rows=int(nrows), verdict=vword, wall=float(wall), arm=arm)
        # 靶偏差必须**同轨**取：整面 search 会静默拿到第一条（本轮恰好是 N=1＝唯一可评分那条），
        # 而三条的偏差数互不相同（13.8／10.1／29.6）⇒ 一旦道次顺序变了就会引错数。
        mw, ma = RE_DEVW.search(s[5]), RE_DEVA.search(s[5])
        assert mw and ma, "N=%s 的 solidus 行里取不到靶 width／area 段 ⇒ 面格式变了" % n
        r.update(devw=float(mw.group(5)), deva=float(ma.group(5)),
                 tw=mw.group(1), tn=int(mw.group(2)), trng=float(mw.group(3)), tspan=mw.group(4),
                 tamed=float(ma.group(1)), tarng=float(ma.group(3)), wstr=s[0])
        # 达标线现取＋偏差**独立重算**：面里的 devw 是它自己印的，本件按 (宽−中位)/中位 再算一遍，
        # 两者只许差在打印精度内 ⇒ 「13.8%」这类数不是把面里的字符串搬过来就当证据。
        bw, ba = RE_BARW.search(s[5]), RE_BARA.search(s[5])
        assert bw and ba, "N=%s 的 solidus 行里取不到达标线 ⇒ 面格式变了" % n
        calcw = abs(r["width"] - float(r["tw"])) / float(r["tw"]) * 100.0
        calca = abs(r["area"] - r["tamed"]) / r["tamed"] * 100.0
        assert abs(calcw - r["devw"]) <= 0.05 + 1e-9, (
            "N=%s 宽偏差：面印 %s%% vs 本件重算 %.3f%% ⇒ 超出 0.1 位打印精度，不同源" % (
                n, r["devw"], calcw))
        assert calca - r["deva"] <= 0.05 + 1e-9, (
            "N=%s 积偏差：面印 %s%% vs 本件重算 %.3f%%" % (n, r["deva"], calca))
        r.update(barw=float(bw.group(1)), barw_fail=float(bw.group(2)),
                 bara=float(ba.group(1)), bara_fail=float(ba.group(2)))
        out.append(r)
    # 轨道内一致性：solidus 行的触边数必须等于 G9 行取到的 max（不等＝读数与闸门不同源）
    for r in out:
        assert r["g9_touched"] >= r["touched"], "G9 的 max touched %d < solidus 触边 %d ⇒ 不同轨配对" % (
            r["g9_touched"], r["touched"])
        assert (r["g9"] == "PASS") == (r["g9_touched"] == 0), "G9 判读与触边数不自洽"
        assert (r["verdict"] == "OK") == (r["g9"] == "PASS"), "SOLVE_VERDICT 与 G9 不自洽"
    return out


def scorable(rows):
    """只有 G9 PASS 的道次可进趋势评分；其余只作下界引用（本轮口径 ⑤）。"""
    return [r for r in rows if r["g9"] == "PASS"]


def leg_solidus(text):
    m = RE_SOLIDUS.search(text)
    assert m, "探针腿里取不到 solidus 行"
    probe = "[PROBE] solidus:" in text
    g8a = RE_G8.search(text)
    g8b = RE_G8B.search(text)
    ver = RE_VERD.search(text)
    return dict(width=float(m.group(1)), wstr=m.group(1), area=int(m.group(2)), nrows=int(m.group(3)),
                touched=int(m.group(5)), probe=probe, g8a=g8a.group(1), gap=float(g8a.group(2)),
                g8b=g8b.group(1), wall=float(ver.group(2)),
                lines=[l for l in text.splitlines() if "[PROBE]" in l])


def probe_relations(legs):
    """六腿之间的**结构性**断言（全部由解析结果现算，不引用任何手打数）。"""
    A, B, C, D, E_, F = (legs[x] for x in "ABCDEF")
    assert A["probe"] and C["probe"] and D["probe"], "A/C/D 应带 [PROBE] 标记 ⇒ 标记口径变了"
    assert not (B["probe"] or E_["probe"] or F["probe"]), "B/E/F 不该带 [PROBE] 标记"
    # G8a 的判读方向必须与"这条腿本来该不该可表示"一致（不一致＝探针的自证面塌了）
    assert [legs[k]["g8a"] for k in "ABCDEF"] == ["FAIL", "PASS", "FAIL", "FAIL", "PASS", "PASS"], (
        "六腿 G8a 判读＝%s ⇒ 与夹具覆写设计不符" % [legs[k]["g8a"] for k in "ABCDEF"])
    assert A["gap"] > 0 and D["gap"] > 0 and E_["gap"] < 0 and F["gap"] < 0, \
        "缺口符号与 G8a 判读不自洽"
    # 打印分辨率现取（数小数位），因为「Δ＝0.00」这句话的真值只到这一位：
    # 面里印的是两位小数 ⇒ 观测到相等只能界定 |A−B| ≤ 半个末位，不能断言"严格为零"。
    dec = len(A["wstr"].split(".")[1])
    assert A["wstr"] == B["wstr"], "钳位腿 A 与忠实腿 B 的**打印值**不同 ⇒ 本轮的『钳位不改形态』结论不成立"
    assert A["width"] == B["width"] and A["area"] == B["area"], (
        "钳位腿 A 与忠实腿 B 不同 ⇒ 本轮的『≤0.004%』结论不成立，信息必须改写")
    assert A["lines"] == C["lines"], "重复腿 C 的 [PROBE] 行与 A 不同 ⇒ 确定性断了，Δ=0 不可归因于钳位无效应"
    assert D["width"] != A["width"], "极端腿 D 与 A 相同 ⇒ 探针是瞎表，Δ=0 没有可失败对照"
    assert E_["width"] != A["width"], "可表示初温腿 E 与钳位腿 A 相同 ⇒ 『同报 300K 即同态』的等价假设复活了"
    dE = E_["width"] - B["width"]
    dF = F["width"] - B["width"]
    assert dE > 0 and dF > dE, "两条斜率腿不单调（dE=%s dF=%s）⇒ 斜率外推不能用" % (dE, dF)
    slopeE = dE / abs(E_["gap"])
    slopeF = dF / abs(F["gap"])
    assert abs(slopeE - slopeF) / slopeF < 0.05, "两腿斜率差 >5%%（%s vs %s）⇒ 非线性，外推作废" % (
        slopeE, slopeF)
    bound_um = 10 ** -dec / 2.0
    return dict(dec=dec, bound_um=bound_um,
                dA=round(A["width"] - B["width"], 2), dC=round(C["width"] - B["width"], 2),
                dD=round(D["width"] - B["width"], 2), dE=round(dE, 2), dF=round(dF, 2),
                slopeE=round(slopeE, 4), slopeF=round(slopeF, 4),
                extrap=round(slopeF * abs(E_["gap"]), 3),
                clamp_pct=round(bound_um / B["width"] * 100, 4),
                gapA=A["gap"], gapD=D["gap"], gapE=E_["gap"])


def check_placeholder_coverage(tmpl, vals):
    used = set(re.findall(r"@@([A-Za-z0-9_]+)@@", tmpl))
    missing = sorted(used - set(vals))
    assert not missing, "模板有未取值的字段：" + str(missing)
    unused = sorted(set(vals) - used)
    assert not unused, "取值表里有模板没用到的读数（＝下一条等着的谎）：" + str(unused)
    return used


def doc_face_rows(modified, devlog_staged):
    """四张归档面：HEAD 行数 vs 暂存 blob 行数，与 numstat 净增闭合。"""
    out = []
    for face in DOC_FACES:
        assert face in modified, "归档面 %s 没在暂存清单里 ⇒ 本轮没归档到位" % face
        head_n = len(sh("git", "show", "HEAD:" + face).splitlines())
        st = sh("git", "show", ":" + face)
        st_n = len(st.splitlines())
        num = [r for r in sh("git", "-c", "core.quotePath=false", "diff", "--cached",
                             "--numstat", "--", face).splitlines() if r.strip()]
        assert len(num) == 1, "%s 的 numstat 不是 1 行" % face
        ins, dels = int(num[0].split("\t")[0]), int(num[0].split("\t")[1])
        assert st_n - head_n == ins - dels, "%s：暂存−HEAD 行数 %d ≠ numstat 净增 %d−%d" % (
            face, st_n - head_n, ins, dels)
        out.append(dict(face=face, old=head_n, new=st_n, ins=ins, dels=dels))
    for face in INSTRUMENT_FACES:
        assert face in modified, "仪器面 %s 没在暂存清单里 ⇒ 本轮改的仪器没入库" % face
    # 本轮新节必须**整体**随本片落地：HEAD 里没有 §26.30，暂存 blob 里恰 1 次。
    # 注意**不能**要求"节号全不重复"：本日志的既有规矩是同一节的续写用 `### 26.27 (t)/(u)` 这种带字母的
    # 标题（现数 §26.27 有 5 条标题）。也**不能**要求"节号单调不减"：既有的 §26.10 就排在 §26.6 之前
    # （HEAD 里数得 1 处降序），那是历史状态、不是本轮的错。能用的闸门只能对着 HEAD 比：
    # 暂存的标题序列必须＝HEAD 的全部标题＋末尾恰好 1 条，且这条是全文件最大节号。
    nums = [int(x) for x in RE_SEC.findall(devlog_staged)]
    head_log = sh("git", "show", "HEAD:" + DEVLOG)
    nums_head = [int(x) for x in RE_SEC.findall(head_log)]
    assert nums[:-1] == nums_head, (
        "暂存日志的节标题序列 ≠ HEAD 的全部标题＋末尾新 1 条（HEAD %d 条／暂存去掉末位 %d 条）"
        "⇒ 本轮改写了历史节或在中间插了节，「最后一节＝本轮」就不成立" % (len(nums_head), len(nums) - 1))
    secn = RE_SEC.findall(devlog_staged)[-1]
    assert int(secn) == max(nums), "最后一节 §26.%s 不是全文件最大节号（最大 §26.%d）⇒「最新＝末位」不成立" % (
        secn, max(nums))
    n_break_head = sum(1 for i in range(1, len(nums_head)) if nums_head[i] < nums_head[i - 1])
    n_break_staged = sum(1 for i in range(1, len(nums)) if nums[i] < nums[i - 1])
    assert n_break_staged == n_break_head, (
        "节号降序处数 HEAD %d → 暂存 %d ⇒ 本轮新增了一处节号乱序" % (n_break_head, n_break_staged))
    n_head_total = len(nums_head)
    descent = "；".join("§26.%d 排在 §26.%d 之前" % (nums_head[i - 1], nums_head[i])
                        for i in range(1, len(nums_head)) if nums_head[i] < nums_head[i - 1]) or "无"
    n_sec_head = len(re.findall(r"^### 26\.%s(?!\d)" % secn, head_log, re.M))
    n_sec_staged = len(re.findall(r"^### 26\.%s(?!\d)" % secn, devlog_staged, re.M))
    assert n_sec_staged == 1, "暂存 blob 里 §26.%s 出现 %d 次" % (secn, n_sec_staged)
    assert len(nums) == n_head_total + 1, (
        "暂存日志比 HEAD 多了 %d 条节标题（应恰 1 条）⇒ 本轮要么没写新节，要么动了历史节" % (
            len(nums) - n_head_total))
    start = devlog_staged.rindex("### 26." + secn)
    nxt = devlog_staged.find("\n### ", start)
    seg = devlog_staged[start:] if nxt == -1 else devlog_staged[start:nxt]
    bullets = RE_BULLET.findall(seg)
    assert bullets, "§26.%s 里取不到子条 ⇒ 日志结构又变了（本件的锚句要跟着改）" % secn
    return out, secn, bullets, n_head_total, n_sec_head, len(nums), n_break_head, descent


def build():
    rows, added, modified = staged()
    assert not sh("git", "-c", "core.quotePath=false", "diff", "--name-only").strip(), \
        "有未暂存漂移 ⇒ 本件取的行数不能代表待提交内容"
    buckets = bucketize(added)
    sraw, ss = numstat_closure()
    n_code = len([p for p in rows if p[-1].startswith("src/") or p[-1].startswith("tests/")])
    dirty_src = len([l for l in sh("git", "-c", "core.quotePath=false", "status", "--porcelain",
                                   "--", "src", "tests").splitlines() if l.strip()])
    assert n_code == 0 and dirty_src == 0, "本轮声称无代码腿，但暂存 src/tests %s 条／工作树 %s 条" % (
        n_code, dirty_src)
    finger_now = subprocess.run(["bash", "-c", FINGER_CMD], cwd=REPO, env=ENV,
                                capture_output=True, text=True).stdout.strip()
    reg = parse_reg(rd(REG))
    curl = gate_prev_receipt(rd(CURL))
    head40_local = sh("git", "rev-parse", "HEAD").strip()
    assert head40_local[:7] == curl["head40"][:7], (
        "本地 HEAD＝%s 而上一片回执声称的远端 main＝%s ⇒ 两者必须同一（本片的父提交就是那棵树），"
        "不等说明中间又有一次提交，本件里『父提交／落地树』的叙述全部要重算" % (
            head40_local[:7], curl["head40"][:7]))
    assert fingerprint_closure(finger_now, reg, curl)
    facts = dict(l.split("=", 1) for l in rd(FACTS).splitlines() if "=" in l)

    # ---- A3 面 ----
    ptext = rd(PILOT)
    prows = parse_pilot(ptext)
    sc = scorable(prows)
    assert len(prows) == 3, "pilot 面道次数＝%s（应为 1,2,3 三条）" % len(prows)
    assert [r["n"] for r in prows] == [1, 2, 3], "pilot 面道次顺序变了"
    assert [r["n"] for r in sc] == [1], "本轮可评分道次应只有 N=1，实得 %s" % [r["n"] for r in sc]
    n1 = sc[0]
    # 靶数**只从可评分那条道次的 solidus 行里取**（口径 ⑤）；整面 search 也能拿到 N=1 的数，
    # 但那是运气不是闸门 ⇒ 这里反过来证明"同轨配对"不是空转：三条道次的偏差数互不相同。
    assert len({r["devw"] for r in prows}) == len(prows), \
        "三条道次的靶偏差有重复（%s）⇒ 同轨配对看不见差别，取数口径要换" % sorted(
            {r["devw"] for r in prows})
    assert len(scorable(prows)) == 1, "可评分道次不止一条 ⇒ 信息里『只有 N=1』那句要改写"
    # 达标线**两源互证**：面里每条道次印的 (≤达标/>未达) 三轨必须一致，且等于**待提交驱动**里的 BARS。
    # 读 `git show :driver`（暂存 blob）而不是工作树 ⇒ 信息描述的就是本片要落地的那份代码。
    barset = {(r["barw"], r["barw_fail"], r["bara"], r["bara_fail"]) for r in prows}
    assert len(barset) == 1, "三条道次印出的达标线不一致 %s ⇒ 判据在面内就变了" % sorted(barset)
    bwv, bwfv, bav, bafv = (prows[0]["barw"], prows[0]["barw_fail"],
                            prows[0]["bara"], prows[0]["bara_fail"])
    dblob = sh("git", "show", ":" + DRIVER)
    mbw = re.search(r"width_um=\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)", dblob)
    mba = re.search(r"area_um2=\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)", dblob)
    assert mbw and mba, ("待提交驱动 %s 里取不到 BARS 的宽／积判据（J7 的机器口径）⇒ 面／代码两源互证断了"
                         % DRIVER)
    mb = tuple(mbw.groups()) + tuple(mba.groups())
    assert [float(x) for x in mb] == [bwv, bwfv, bav, bafv], (
        "面里印的达标线 (%s,%s,%s,%s) ≠ 驱动 BARS (%s) ⇒ 运行用的判据和落地的判据不是一个" % (
            bwv, bwfv, bav, bafv, mb))
    tw = dict(median=float(n1["tw"]), n=n1["tn"], rng=n1["trng"], span=n1["tspan"], dev=n1["devw"])
    ta = dict(median=n1["tamed"], rng=n1["tarng"], dev=n1["deva"])
    assert n1["g9"] == "PASS" and n1["touched"] == 0, "N=1 自己就不干净 ⇒ 无从评分"
    # 反序证据（能量更多而形态更小）＝#30 的直接证据，必须现算而不是引用叙述
    assert prows[2]["area"] < prows[1]["area"], "N=3 面积不再小于 N=2 ⇒ 边界截断的论据变了，信息要改写"
    assert prows[2]["touched"] > prows[1]["touched"] > n1["touched"], "触边行数不随 N 单调 ⇒ 取数不同轨"
    cost = RE_COST.search(ptext)
    assert cost, "取不到代价登记行"
    assert len(RE_COST.findall(ptext)) == 1, "代价登记行在面里不止 1 条 ⇒ search 引的是哪一条没有依据"
    # 夹具与"跑前定死"行现取互证：本件不许替面里那行话做总结
    sel = RE_SEL.search(ptext)
    assert sel, "取不到「本轮选用（跑前定死）」行 ⇒ dx／tracks／arms 没有出处"
    stage_s, strat_s, tracks_s, alpha_s, dx_s, arm_s = sel.groups()
    assert (stage_s, strat_s, arm_s) == ("pilot", "C", "norm"), (
        "面内选用行＝%s/%s/%s ⇒ 与本件「pilot 档、C 策略、norm 臂」那句不符" % (stage_s, strat_s, arm_s))
    tl = [int(x) for x in re.findall(r"\d+", tracks_s)]
    assert tl == [r["n"] for r in prows], "跑前定死的 tracks %s ≠ 面里实得道次 %s" % (tl, [r["n"] for r in prows])
    assert all(r["arm"] == arm_s for r in prows), "面里有别的臂的读数混进同一条趋势 ⇒ 两臂对比无效"
    amb = float(RE_AMB.search(ptext).group(1))
    rb = float(RE_RB.search(ptext).group(1))
    assert abs(n1["dx"] - rb / 2) < 1e-3, "面里 dx=%sµm ≠ r_b/2=%sµm ⇒「dx=r_b/2」那句是假的" % (n1["dx"], rb / 2)
    ctl = RE_CTL4.findall(ptext)
    ctl_pass = [k for k, _ in ctl if k == "PASS"]
    n_ctl = len(ctl_pass)
    ctl_labels = sorted({lab for _, lab in ctl})
    assert ctl_labels == ["C1", "C2", "C3", "C4"], (
        "读数核合成控件的标签集＝%s ⇒ 面里的控件不再是 C1–C4 这一组，「／4」那个分母要跟着改" % ctl_labels)
    assert n_ctl == len(ctl_labels), (
        "控件共 %d 条标签但 [PASS] 只有 %d 条 ⇒ 有控件是红的，本轮不能引它当正对照" % (
            len(ctl_labels), n_ctl))
    per_track = round(prows[-1]["wall"] / prows[-1]["n"], 1)
    assert abs(float(cost.group(3)) - per_track) < 0.15, "面里印的每道均摊 %s 与现算 %s 不符" % (
        cost.group(3), per_track)
    # 均摊的分母用**该腿自己的道次数**（现取），趋势上限的 18 也**从面里读**而不是本件写死
    assert int(cost.group(1)) == prows[-1]["n"], "面里的「最大 N=%s」≠ 实得最后一腿 N=%s" % (
        cost.group(1), prows[-1]["n"])
    assert abs(float(cost.group(2)) - prows[-1]["wall"]) < 0.06, (
        "面里的墙钟 %ss ≠ SOLVE_VERDICT 行的 %ss ⇒ 代价行与判决行不同源" % (
            cost.group(2), prows[-1]["wall"]))
    ntrend = int(cost.group(4))
    sigma = sum(range(1, ntrend + 1)) * per_track
    assert abs(float(cost.group(5)) - sigma) < 60, "面里印的 Σ≈%s 与现算 %s 偏离过大" % (
        cost.group(5), round(sigma))
    trend = RE_TREND.search(ptext)
    assert trend, "取不到趋势行"
    # 旧版（无 G9）面与新版面互为确定性件：solidus **整行原文**逐字节相同（不只是宽／积两个数）
    old_lines = solidus_lines(rd(PILOT_PRE))
    new_lines = solidus_lines(ptext)
    assert len(old_lines) == len(new_lines) == len(prows), "两面的 solidus 行数不等（%d/%d/%d）" % (
        len(old_lines), len(new_lines), len(prows))
    assert old_lines == new_lines, "G9 改动**前后**形态整行不同 ⇒ 仪器改动扰动了数值，本轮的确定性论据不成立"
    assert solidus_lines(old_lines[0]) == [old_lines[0]], "整行取数自己就不闭合（首行再取一次变了）"
    rraw = parse_pilot(rd(RAWN1))
    assert len(rraw) == 1 and rraw[0]["arm"] == "raw", "raw 臂面不是单条 raw ⇒ 两臂对比无效"
    assert rraw[0]["width"] != n1["width"], "两臂宽度相同 ⇒ 选臂不影响形态，与档案矛盾"
    assert rraw[0]["g9"] == "PASS" and rraw[0]["touched"] == 0, "raw 臂 N=1 自己就触边 ⇒ 两臂对比不可用"
    raw_dev = rraw[0]["devw"]
    # smoke 面：结构性禁掉形态数字 ⇒ 两条面各 0 条 `solidus: 宽=`；布尔出口按**两条腿分别**现取
    # （card300 腿 G8a FAIL ⇒ 连布尔行都不印；ambk29315 腿两条 G8 全绿 ⇒ 印 4 条、每条 6 个布尔／计数）
    stxt, s2txt = rd(SMOKE), rd(SMOKE_AMB)
    nsol = (len(RE_SOLIDUS.findall(stxt)), len(RE_SOLIDUS.findall(s2txt)))
    assert nsol == (0, 0), "smoke 两条腿里出现了 solidus 数字 %s ⇒ 结构性关闭失效，形态出口没被禁掉" % (nsol,)
    bool_lines = [l for l in s2txt.splitlines() if "接线=True" in l]
    assert len(bool_lines) == 4, "smoke 布尔行＝%d 条（四个等值面各一条）⇒ 出口形状变了" % len(bool_lines)
    bool_fields = ("接线", "宽有限", "积有限", "行数", "有熔", "触边")
    bfmiss = ["%s：缺 %s" % (l.strip()[:40], f) for l in bool_lines for f in bool_fields if f + "=" not in l]
    assert not bfmiss, "布尔行的字段表与「每条 N 个字段」那句不符 ⇒ " + "；".join(bfmiss[:3])
    n_boolum = sum(l.count("µm") for l in bool_lines)
    assert n_boolum == 0, "布尔行里出现 %d 处 `µm` ⇒ 「不产出形态数字」那句是假的" % n_boolum
    # `--amb-k` 那个数从**这条腿自己的面**取，并和 pilot 面的环境温度对表（两腿必须同一环境温度才谈得上对照）
    amb_s = float(RE_AMB.search(s2txt).group(1))
    assert amb_s == amb, "smoke ambk 腿面内 T_ambient=%s ≠ pilot 腿的 %s ⇒ 信息里「--amb-k」引的是另一条腿的数" % (
        amb_s, amb)
    n_boolfail = len(re.findall(r"^\s+\[SOFT\] ", stxt, re.M))
    n_soft = len(re.findall(r"^\s+\[SOFT\] ", s2txt, re.M))
    n_soft_end = sum(len(re.findall(r"STAGE_DONE[^\n]*软闸", t)) for t in (stxt, s2txt))
    n_done = sum(len(re.findall(r"^STAGE_DONE = smoke ", t, re.M)) for t in (stxt, s2txt))
    n_rofail = len(re.findall(r"READOUT_FAIL", stxt))
    assert n_soft == 2, "smoke 面 [SOFT] 闸门行＝%s（本件口径＝J6＋G6 恰好 2 条）" % n_soft
    assert n_boolfail == 2, "缺省卡腿的 [SOFT] 行＝%s ⇒ 软闸数量在两腿之间不一致" % n_boolfail
    assert n_rofail == 1 and "READOUT_FAIL" not in s2txt, (
        "READOUT_FAIL 行数＝%s（缺省卡腿应 1 条、amb 腿应 0 条）⇒ G8 与形态出口的联动变了" % n_rofail)
    # 软闸的**名字**也从面里现取（原来信息里写死「＝J6＋G6」，那是我对面的形状的假设）
    softlab = [sorted(set(RE_SOFTLAB.findall(t))) for t in (stxt, s2txt)]
    assert softlab[0] == softlab[1], "两条 smoke 腿的软闸标签集不同 %s vs %s ⇒「各报」那句把两腿混成了一条" % (
        softlab[0], softlab[1])
    assert len(softlab[0]) == n_soft == n_boolfail, (
        "软闸标签 %d 个 ≠ [SOFT] 行 %s／%s 条 ⇒ 有行同名或有行没标签，「各报 N 条」没有分母" % (
            len(softlab[0]), n_soft, n_boolfail))
    soft_join = "／".join(softlab[0])
    assert "J6" in softlab[0], (
        "面里的软闸标签集 %s 没有 J6 ⇒「按 J6 结构性禁掉形态数字」那句没有出处" % softlab[0])
    assert n_soft_end == n_done, "STAGE_DONE 里『软闸』字样 %s 次 vs 判读行 %s 次 ⇒ 不是每条判读都自带软闸标签" % (
        n_soft_end, n_done)
    # 预检回归：sed 归一后必须与原预检面逐字节相同
    assert md5(PRE_REG_N) == md5(PRE_ORIG), "预检复跑（标签归一后）与已落地面不同 ⇒ 回归论据不成立"
    diff_lines = len([l for l in subprocess.run(
        ["diff", os.path.join(REPO, PRE_ORIG), os.path.join(REPO, PRE_REG)],
        capture_output=True, text=True).stdout.splitlines() if l.startswith(("<", ">"))])
    n_soft_reg = len(re.findall(r"^\s+\[SOFT\] ", rd(PRE_REG), re.M))
    assert n_soft_reg == 0, "预检复跑里出现 %d 条 [SOFT] ⇒ 软闸适用面漏到了 preflight" % n_soft_reg
    legs = {k: leg_solidus(rd(v)) for k, v in ICLEG.items()}
    pr = probe_relations(legs)
    # 六腿的 wall 全在同一量级 ⇒ 支撑"Δ 不是计时噪声"（跨进程确定性另有 C 腿逐字节那条）
    walls = sorted(legs[k]["wall"] for k in "ABCDEF")
    assert walls[-1] / walls[0] < 1.25, "六腿墙钟跨度过大 %s–%s ⇒ 不能把 Δ 说成同量级运行" % (
        walls[0], walls[-1])

    # ---- 归档面 ----
    devlog_staged = sh("git", "show", ":" + DEVLOG)
    docs, secn, bullets, n_head_total, n_sec_head, n_head_staged, n_break, descent = doc_face_rows(
        modified, devlog_staged)
    assert secn == "30", "本轮节号现取＝%s（本件按 §26.30 写）⇒ 节号变了要改本件的措辞" % secn
    # 排期那句要引的"单道散度 p50"逐字取自**暂存的**开发日志 §26.11(c) 合并口径表（不是对话记忆）
    p50w, p50a = RE_P50W.search(devlog_staged), RE_P50A.search(devlog_staged)
    assert p50w and p50a, "暂存日志里取不到 §26.11(c) 的散差表两行 ⇒ 排期那句没有出处，本件不许自带数字"
    # search 只会落在第一条 ⇒ 表在整个日志里只许出现 1 次，否则"引的是哪张表"就是运气
    assert len(RE_P50W.findall(devlog_staged)) == 1, "宽度散差行在暂存日志里不止 1 条 ⇒ 引用哪一条没有依据"
    assert len(RE_P50A.findall(devlog_staged)) == 1, "面积散差行在暂存日志里不止 1 条 ⇒ 同上"
    # 引用的**节号／子标记**也现取：把「§26.11(c) 那张表」从我对手工引用的信任变成一条可失败的读数的。
    head_spans = [(mm.group(1), mm.start()) for mm in RE_SEC.finditer(devlog_staged)]

    def sec_of(pos):
        cur = None
        for num, st in head_spans:
            if st <= pos:
                cur = num
            else:
                break
        return cur

    p50_sec = sec_of(p50w.start())
    p50_sub = re.findall(r"\*\*\((\w)\)",
                         devlog_staged[max(st for _, st in head_spans if st < p50w.start()):p50w.start()])
    assert p50_sec == "11" and p50_sub and p50_sub[-1] == "c", (
        "散差表两行现落 §26.%s、其前子标记 %s ⇒ 信息里那句「§26.11(c)」的出处变了" % (
            p50_sec, (p50_sub[-1] if p50_sub else "无")))
    p50_cite = "§26.%s(%s)" % (p50_sec, p50_sub[-1])
    # 钉住 GPU 那句口径的出处＝**最后**一次出现在哪一节（最近一次登记才算现行约束）
    gpuq = "继续，还是不动GPU"
    n_gpuq = devlog_staged.count(gpuq)
    gpu_sec = sec_of(devlog_staged.rfind(gpuq))
    assert n_gpuq >= 1 and gpu_sec == secn, (
        "「%s」在暂存日志出现 %d 次，最后一次落在 §26.%s 而不是本轮 §26.%s ⇒ 信息不能把它当本轮登记的约束来引" % (
            gpuq, n_gpuq, gpu_sec, secn))
    assert p50w.group(1) == p50a.group(1), (
        "宽度／面积的样本数不等（%s vs %s）⇒ 合并口径表自己不自洽" % (p50w.group(1), p50a.group(1)))
    assert float(p50w.group(2)) < float(p50w.group(3)) < float(p50w.group(4)), (
        "宽度 p50/p90/max 不单调（%s/%s/%s）⇒ 表被改坏了" % p50w.group(2, 3, 4))
    assert float(p50a.group(2)) < float(p50a.group(3)) < float(p50a.group(4)), "面积 p50/p90/max 不单调"
    byface = {d["face"]: d for d in docs}
    # 插入行的两个出处必须同一：逐面 numstat（本件按面取）与全局 numstat／shortstat（上面已闭合）
    ins_map = {}
    for l in sh("git", "-c", "core.quotePath=false", "diff", "--cached", "--numstat").splitlines():
        if l.strip():
            r = l.split("\t")
            ins_map[r[-1]] = int(r[0])
    doc_add = sum(d["ins"] for d in docs)
    instr_add = sum(ins_map[p] for p in INSTRUMENT_FACES)
    assert doc_add + instr_add == sum(ins_map[p] for p in modified), (
        "四张归档面＋两张仪器面的插入行 %d≠六张修改面实得 %d ⇒ 角色表漏了面" % (
            doc_add + instr_add, sum(ins_map[p] for p in modified)))
    assert sum(ins_map[p] for p in added) == ss[1] - sum(ins_map[p] for p in modified), \
        "新增件插入＋修改件插入 ≠ shortstat 总插入 ⇒ 上面那句『三个数闭合』是假的"
    # 开发日志里登记过的"未入库面"与本片实际入库面**必须能对上，且差额只能是本件自己的四张面**
    logs = list(RE_UNTRACKED.finditer(devlog_staged))
    assert logs, "开发日志 §26.%s 里取不到「未入库＝未跟踪…」那句登记 ⇒ 这条闭合没有出处" % secn
    # findall 会丢掉位置；这里必须知道取的是**哪一条**（该句式在日志里出现多次，取最新一条）
    assert sec_of(logs[-1].start()) == secn, (
        "「未入库…」登记的最新一条落在 §26.%s 而不是本轮 §26.%s ⇒ 引的是历史面数" % (
            sec_of(logs[-1].start()), secn))
    lun, lmod, ltot = (int(x) for x in logs[-1].groups())
    assert lun + lmod == ltot, "日志那句自身不算术闭合（%d＋%d≠%d）" % (lun, lmod, ltot)
    msg_bucket, msg_pat = BUCKETS[-1]
    assert re.search(msg_pat, SELF), (
        "本件自身（%s）不落进最后一桶「%s」⇒「差额恰好是本件那几张」这句是假的" % (SELF, msg_bucket))
    gap = len(added) - lun
    assert gap == len(buckets[msg_bucket]), (
        "暂存新增 %d 与日志登记的未跟踪 %d 差 %d 件，而本信息生成器桶只有 %d 件 ⇒ 中间还动过别的文件，"
        "那句『全部交分片 28』不再成立" % (len(added), lun, gap, len(buckets[msg_bucket])))
    assert lmod == len(modified), "日志登记的修改面数 %d ≠ 暂存实得 %d" % (lmod, len(modified))
    assert sorted(modified) == sorted(DOC_FACES + INSTRUMENT_FACES), (
        "暂存的修改面与本件的角色表不符：%s" % sorted(modified))
    # 「台账本轮未改」要能红：只说"它不在修改面里"是**恒真**的（它本就不在角色表里），
    # 真正的口径是暂存 blob 逐字节等于 HEAD 版本。
    assert sh("git", "show", ":" + LEDGER) == sh("git", "show", "HEAD:" + LEDGER), (
        "repo 台账 %s 的暂存 blob ≠ HEAD 版本 ⇒ 信息那句「本轮未改」是假的" % LEDGER)

    # ---- 尺子与回执链 ----
    ruler_md5 = md5(os.path.join(REPO, RULER))
    copies = sorted(p for p in (os.path.relpath(x, REPO) for x in
                                glob.glob(os.path.join(REPO, "docs/evidence/*/am_s7_verify*.py")))
                    if re.fullmatch(os.path.join("docs", "evidence", "[^/]+", r"am_s7_verify\d+\.py"), p)
                    and md5(os.path.join(REPO, p)) == ruler_md5)
    assert RULER in copies, "本片的尺子副本没被自己的 glob 数到"
    assert ruler_md5[:12] == curl["ruler_md5"], (
        "本轮尺子 md5 前 12＝%s ≠ 上一片回执声称的 %s ⇒ 尺子被改过，跨片比对失去意义" % (
            ruler_md5[:12], curl["ruler_md5"]))
    assert int(curl["copies"]) == len(copies), (
        "上一片回执声称 %s 份逐字节副本，本轮全树现算 %d 份 ⇒ 两者本应相同（本片的下一把要到推送后才抄），"
        "不等说明中间有人动过尺子家族" % (curl["copies"], len(copies)))
    # 回执链件数**用分类表的同一桶**（原来这里重打了一遍该桶的正则＝第二个口径，会漂）
    chain = len(buckets[BUCKETS[0][0]])
    assert chain == 5, "分片 27 的回执链本轮入库 %d/5 件（尺子／回执／out／err／表头）" % chain
    # 正对照的**个数与字母**从回执正文现取（curl 的 ctl_a…ctl_d 是本件自己构造的，数它等于数自己）
    ctrl_letters = RE_CTLN.findall(rd(CURL))
    assert sorted(ctrl_letters) == sorted(k[4:].upper() for k in curl if k.startswith("ctl_")), (
        "回执正文里的正对照 %s ≠ 本件取数的 %s ⇒ 有一侧的对照没被引到" % (
            sorted(ctrl_letters), sorted(k[4:].upper() for k in curl if k.startswith("ctl_"))))

    nv = [l.strip() for l in subprocess.run(["nvidia-smi", "--query-compute-apps=pid",
                                             "--format=csv,noheader"], capture_output=True,
                                            text=True).stdout.splitlines() if l.strip()]
    nv_mine, nv_read = [], 0
    for p in nv:
        try:
            cmd = open("/proc/%s/cmdline" % p, "rb").read().decode("utf-8", "replace")
        except OSError:
            continue
        nv_read += 1
        if "/tmp/amvenv" in cmd:
            nv_mine.append(p)
    # 匹配式正对照：同一段代码喂一条**已知该命中**的合成行，必须数出 1（否则那个 0 是假零）
    assert nv_read == len(nv), "卡上 %d 个 compute 进程只读回 %d 条 cmdline ⇒ 那个『我的 0』没数完" % (
        len(nv), nv_read)
    ctl = [p for p in ["/proc/self/cmdline"] if "/tmp/amvenv" in rd(p)]
    assert len(ctl) == 1, "GPU 匹配式的正对照没命中 ⇒ 下面的『我的 0』不可信"
    assert nv_mine == [], "我的解释器在 GPU 上留有进程 %s ⇒ 与「不动 GPU」冲突" % nv_mine

    V = {
        "SLICE": "28", "PREVSLICE": "27", "SECN": secn, "NBULLET": str(len(bullets)),
        "BULL0": bullets[0], "BULLN": bullets[-1], "SOFTLAB": soft_join,
        "NA": str(len(added)), "NM": str(len(modified)), "NTOT": str(len(rows)),
        "SHORTSTAT": sraw, "SSF": str(ss[0]), "SSI": str(ss[1]), "SSD": str(ss[2]),
        "NCODE": str(n_code), "DIRTYSRC": str(dirty_src),
        "NBUCKET": str(len(buckets)),
        "NDOC": str(len(DOC_FACES)), "NINSTR": str(len(INSTRUMENT_FACES)),
        "INSTRNAMES": "／".join("`%s`" % os.path.basename(p) for p in INSTRUMENT_FACES),
        "BUCKETLIST": "\n".join("- %s＝%d 件" % (k, len(v)) for k, v in buckets.items()),
        "MSGBUCKET": msg_bucket,
        "NMSG": str(len(buckets[msg_bucket])),
        "BUCKETSUM": str(sum(len(v) for v in buckets.values())),
        "DOCADD": str(doc_add),
        "DL0": str(byface[DEVLOG]["old"]), "DL1": str(byface[DEVLOG]["new"]),
        "DLD": str(byface[DEVLOG]["ins"]), "DLD2": str(byface[DEVLOG]["dels"]),
        "PL0": str(byface[DOC_FACES[1]]["old"]), "PL1": str(byface[DOC_FACES[1]]["new"]),
        "PLD": str(byface[DOC_FACES[1]]["ins"]), "PLD2": str(byface[DOC_FACES[1]]["dels"]),
        "OV0": str(byface[DOC_FACES[2]]["old"]), "OV1": str(byface[DOC_FACES[2]]["new"]),
        "OVD": str(byface[DOC_FACES[2]]["ins"]), "OVD2": str(byface[DOC_FACES[2]]["dels"]),
        "DY0": str(byface[DOC_FACES[3]]["old"]), "DY1": str(byface[DOC_FACES[3]]["new"]),
        "DYD": str(byface[DOC_FACES[3]]["ins"]), "DYD2": str(byface[DOC_FACES[3]]["dels"]),
        "SECH": str(n_sec_head), "SECSTAGED": "1",
        "HEADTOT": str(n_head_total), "HEADSTAGED": str(n_head_staged), "NBREAK": str(n_break),
        "P50W": p50w.group(2), "P90W": p50w.group(3), "MAXW": p50w.group(4),
        "P50A": p50a.group(2), "P90A": p50a.group(3), "MAXA": p50a.group(4), "P50N": p50w.group(1),
        "P50CITE": p50_cite, "NQUOTE": str(n_gpuq), "GPUSEC": gpu_sec,
        "DESCENT": descent,
        "HEAD7": head40_local[:7], "CHEAD7": curl["head40"][:7], "CHEAD40": curl["head40"],
        "PBLOB": curl["parent"],
        "FINGER": finger_now, "REG": REG, "REGMD5": md5(REG, 12),
        "REGHEAD": reg["head"], "REGDIRTY": reg["dirty"], "TALLY": reg["tally"],
        "RC": reg["rc"], "NF": str(reg["nf"]), "NP": str(reg["npass"]), "NS": str(reg["nskip"]),
        "WALL": reg["wall"], "D1": reg["d1"], "D2": reg["d2"],
        "E1": reg["elines"][0].strip(), "E2": reg["elines"][1].strip(),
        "XFAIL": facts["XFAIL"], "COLLECT": facts["COLLECT"],
        "CURL": CURL, "CURLMD5": md5(CURL, 12), "SELF": SELF, "SELFMD5": md5(os.path.join(HERE, SELF), 12),
        "NOW": sh("date", "+%F %T %z").strip(),
        "VERDICT": curl["verdict"], "REMOTE": curl["remote"], "EXPECT": curl["expect"],
        "PARENT40": curl["parent40"], "PARENT7": curl["parent40"][:7],
        "ENTRIES": curl["entries"], "BLOB": curl["blob"], "TREEENT": curl["tree"],
        "NFILE": str(curl["nfile"]), "MATCH": str(curl["match"]), "A27": str(curl["a"]),
        "NCHAIN": str(chain),
        "NCTRL": str(len(ctrl_letters)), "CTRLEN": "／".join(ctrl_letters),
        "M27": str(curl["m"]), "DL0C": curl["dl"], "DLDEN": curl["dlden"], "DR": curl["dr"],
        "DRDEN": curl["drden"], "SHA": curl["sha"], "SHADEN": curl["shaden"], "BOFF": curl["boff"],
        "CA0": curl["ctl_a"][0], "CA1": curl["ctl_a"][1], "CB1": curl["ctl_b"][0],
        "CB1D": curl["ctl_b"][1], "CB2": curl["ctl_b"][2], "CB2D": curl["ctl_b"][3],
        "CC_L": curl["ctl_c"][0], "CC_R": curl["ctl_c"][1], "CC_V": curl["ctl_c"][2],
        "CD1": curl["ctl_d"][0], "CD1D": curl["ctl_d"][1], "CD2": curl["ctl_d"][2],
        "CD2D": curl["ctl_d"][3], "RULERMD5": ruler_md5[:12], "COPIES": str(len(copies)),
        "COPIESNEXT": str(len(copies) + 1),
        "PREVCOP": curl["copies"], "HDR27": HDR27, "RAW27": RAW27, "RAW27ERR": RAW27ERR,
        "RAW27MD5": md5(RAW27, 12),
        # A3 读数
        "PILOT": PILOT, "PILOTMD5": md5(PILOT, 12),
        "N1W": "%.2f" % n1["width"], "N1A": str(n1["area"]), "N1R": str(n1["nrows"]),
        "N2W": "%.2f" % prows[1]["width"], "N2A": str(prows[1]["area"]), "N2T": str(prows[1]["touched"]),
        "N2R": str(prows[1]["nrows"]), "N3W": "%.2f" % prows[2]["width"], "N3A": str(prows[2]["area"]),
        "N3T": str(prows[2]["touched"]), "N3R": str(prows[2]["nrows"]),
        "TW": "%.1f" % tw["median"], "TN": str(tw["n"]), "TRNG": "%.1f" % tw["rng"], "TSPAN": tw["span"],
        "DEVW": "%.1f" % tw["dev"],
        "TAMED": "%.1f" % ta["median"], "TADEV": "%.1f" % ta["dev"], "TARNG": "%.1f" % ta["rng"],
        "RAWN1W": "%.2f" % rraw[0]["width"], "RAWDEV": "%.1f" % raw_dev,
        "W1": "%.2f" % prows[0]["wall"], "W2": "%.2f" % prows[1]["wall"], "W3": "%.2f" % prows[2]["wall"],
        "PERTRACK": "%.1f" % per_track, "SIGMA": "%.0f" % sigma, "BUDGET": cost.group(6),
        "NTREND": str(ntrend), "SUMN": str(sum(range(1, ntrend + 1))),
        "FACEPER": cost.group(3), "FACETOT": cost.group(5),
        "COSTVERD": cost.group(7), "TREND": trend.group(1), "TRENDFLAT": trend.group(2),
        "TRENRATIO": trend.group(3), "NVOX": str(n1["nvox"]), "NSTEPS": str(n1["nsteps"]),
        "N1T": str(n1["touched"]), "AMB": "%.2f" % amb, "RB": "%.3f" % rb,
        "DX1": "%.3f" % n1["dx"], "TRACKS": str(tl), "NTRACKS": str(len(prows)),
        "NUNSC": str(len(prows) - len(sc)), "ALPHA": alpha_s, "NCTL4": str(n_ctl),
        "NCTLDEN": str(len(ctl_labels)),
        "C14": "C%s–C%s" % (ctl_labels[0][1:], ctl_labels[-1][1:]),
        "NMAXC": cost.group(1),
        "BARW": "%g" % bwv, "BARFAIL": "%g" % bwfv, "BARA": "%g" % bav,
        "SLACKW": "%.1f" % (bwv - n1["devw"]), "OVERA": "%.1f" % (n1["deva"] - bav),
        "SOFTSMOKE": str(n_soft), "SOFTREG": str(n_soft_reg),
        "PREDIFF": str(diff_lines), "SMOKE": SMOKE, "SMOKEMD5": md5(SMOKE, 12),
        "LEGTAB": "｜".join("%s %s/%s" % (k, legs[k]["g8a"], "P" if legs[k]["probe"] else "-")
                           for k in "ABCDEF"),
        "DA": "%.2f" % pr["dA"], "DC": "%.2f" % pr["dC"], "DD": "%.2f" % pr["dD"],
        "DE": "%.2f" % pr["dE"], "DF": "%.2f" % pr["dF"],
        "SLOPEE": "%.4f" % pr["slopeE"], "SLOPEF": "%.4f" % pr["slopeF"],
        "EXTRAP": "%.3f" % pr["extrap"], "CLAMPPCT": "%.4f" % pr["clamp_pct"],
        "GAPEABS": "%.4f" % abs(pr["gapE"]),
        "MAGW": "%.0f" % (bwv / pr["clamp_pct"]), "MAGA": "%.0f" % (bav / pr["clamp_pct"]),
        "ERATIO": "%.3f" % (pr["extrap"] / pr["dE"]),
        "BOUNDUM": "%.3f" % pr["bound_um"], "PDEC": str(pr["dec"]),
        "DEVWUNI": str(len({r["devw"] for r in prows})),
        "GAPA": "%.4f" % pr["gapA"], "GAPD": "%.4f" % pr["gapD"], "GAPE": "%.4f" % pr["gapE"],
        "WALLMIN": "%.2f" % walls[0], "WALLMAX": "%.2f" % walls[-1],
        "ICMD5": "｜".join("%s=%s" % (k, md5(ICLEG[k], 12)) for k in "ABCDEF"),
        "NV": str(len(nv)), "NVME": str(len(nv_mine)), "NVREAD": str(nv_read),
        # 面的**名字与 md5**（信息里凡是"哪张面"的指称一律现取，不靠读者去猜）
        "PILOT_PRE": PILOT_PRE, "PILOT_PREMD5": md5(PILOT_PRE, 12),
        "SMOKEAMB": SMOKE_AMB, "SMOKEAMBMD5": md5(SMOKE_AMB, 12),
        "PREREG": PRE_REG, "PREREGMD5": md5(PRE_REG, 12),
        "PREREGN": PRE_REG_N, "PREREGNMD5": md5(PRE_REG_N, 12),
        "PREORIG": PRE_ORIG, "PREORIGMD5": md5(PRE_ORIG, 12),
        "LOGUN": str(lun), "LOGMOD": str(lmod), "LOGTOT": str(ltot), "GAP": str(gap),
        "NLOG": str(len(logs)), "LEDGER": LEDGER,
        "INSTRADD": str(instr_add),
        "SOFTEND": str(n_soft_end), "STAGEDONE": str(n_done),
        "NBOOL": str(len(bool_lines)), "NBOOLFIELD": str(len(bool_fields)),
        "NSOL": str(nsol[0]), "NSOL2": str(nsol[1]), "NBOOLUM": str(n_boolum), "AMBS": "%.2f" % amb_s,
        "BOOLFIELDS": "／".join(bool_fields), "NROFAIL": str(n_rofail), "SOFTFAIL": str(n_boolfail),
    }
    # 门：V 的字面键**不得重复**。Python 的字典字面量对重复键静默 last-wins，本件曾让一组
    # P50* 连出三份而 MAXW／MAXA 被前两行留下 ⇒ 键表用 **ast** 现取（文本形状不可靠：值里
    # 的 markdown 粗体 `**词**` 与真正的 `**解包` 在正则里长得一样），并要求字面键集合
    # 等于实得 V 的键集合（不许有键从别处塞进来或从字面里漏掉）。
    # 读**正在执行的这份源码**（`__file__`）而不是按文件名再去找：直接跑时两者同一个文件，
    # 而控件面跑的是同目录副本 ⇒ 只有读 `__file__` 才让这三条键表闸门真的能红（否则副本审的是原件）。
    own = open(os.path.abspath(__file__), encoding="utf-8").read()
    vdict = [n.value for n in ast.walk(ast.parse(own))
             if isinstance(n, ast.Assign) and isinstance(n.value, ast.Dict)
             and any(isinstance(t, ast.Name) and t.id == "V" for t in n.targets)]
    assert len(vdict) == 1, "本件源码里名为 V 的 dict 字面量不是恰好 1 处（实得 %d）⇒ 键表门无法定位" % len(vdict)
    lit = []
    for k in vdict[0].keys:
        # 解包 `**{…}` 在 ast.Dict.keys 里是 **None** ⇒ 消息式子不许对它调 ast.dump（一调就 TypeError，
        # 看客拿到的是崩栈而不是这条闸门句；控件面 C37 实测到，判据也按「抛出的那一行」取数）。
        assert isinstance(k, ast.Constant) and isinstance(k.value, str), (
            "V 的字面里有非字符串常量键（%s）⇒ 键表门只认字面字符串键"
            % ("解包 **" if k is None else ast.dump(k)))
        lit.append(k.value)
    dup = sorted({k for k in lit if lit.count(k) > 1})
    assert not dup, "V 里有重复字面键（dict 字面量静默 last-wins）：" + "、".join(dup)
    assert len(lit) == len(V), "V 的字面键 %d 个 ≠ 实得键 %d 个 ⇒ 有键在字面量之外被改写／补上" % (len(lit), len(V))
    assert set(lit) == set(V), ("V 的字面键表与实得键集合不符：源码独有 %s ／ 实得独有 %s"
                                % ("、".join(sorted(set(lit) - set(V))) or "无",
                                   "、".join(sorted(set(V) - set(lit))) or "无"))
    check_placeholder_coverage(TMPL, V)
    msg = TMPL
    for k, v in V.items():
        msg = msg.replace("@@" + k + "@@", str(v))
    assert "@@" not in msg and "＠" not in msg, "信息里残留占位符 ⇒ 替换没走完"
    counts = dict(total=len(rows), added=len(added), modified=len(modified), code_changes=n_code,
                  bucket_sum=sum(len(v) for v in buckets.values()), ruler_copies=len(copies),
                  scorable=[r["n"] for r in sc], touched=[r["touched"] for r in prows],
                  gpu_total=len(nv), gpu_mine=len(nv_mine), gpu_read=nv_read,
                  shortstat=ss, numstat=ss, section=secn, bullets=len(bullets),
                  doc_insertions=doc_add, finger=finger_now[:12],
                  clamp_pct=pr["clamp_pct"], soft_smoke=n_soft, soft_regress=n_soft_reg)
    return msg, counts


TMPL = """S7 分片 @@SLICE@@：A3 腿②＝smoke 接线跑＋pilot 三面（N=@@TRACKS@@）入库 ⇒ **只有 N=1 可评分**（新 G9 触边闸 @@N1T@@/@@N2T@@/@@N3T@@ 行，N=3 面积反小于 N=2）；#51 焓模型钳位量化到 ≤@@CLAMPPCT@@% 出 A3 关键路径（降级不关单）；分片 @@PREVSLICE@@ 回执链入库；开发日志 §26.@@SECN@@

本片＝分片 @@SLICE@@，上一片＝分片 @@PREVSLICE@@；信息本身由 `@@SELF@@`（md5 @@SELFMD5@@，生成于 @@NOW@@）从命令现取，
不是手打。入库面：新增 @@NA@@ 件＋修改 @@NM@@ 件＝@@NTOT@@ 件（`git diff --cached --shortstat` 原文
「@@SHORTSTAT@@」，三个数与 `--numstat` 逐行现算闭合到同一组＝@@SSF@@ files／@@SSI@@ insertions／@@SSD@@ deletions）。
这 @@NA@@ 件对得上开发日志自己的登记：§26.@@SECN@@ 里「未入库＝未跟踪…」这句有 @@NLOG@@ 条，取**最新**一条＝「未跟踪 **@@LOGUN@@**＋修改 **@@LOGMOD@@**＝**@@LOGTOT@@**」；它的
@@LOGUN@@ 未跟踪＋@@GAP@@ 件＝@@NA@@，而差额 @@GAP@@ 件**恰好**是桶表里的「@@MSGBUCKET@@」那几张（@@NMSG@@ 件，本件自身按该桶的正则命中），
修改面数 @@LOGMOD@@＝暂存实得 @@NM@@ ⇒ 落笔之后到暂存之间没动过别的文件。
**src／tests 本轮 @@NCODE@@ 件改动**（`git diff --cached --name-only -- src tests` 现数 @@NCODE@@ 条；工作树 src+tests 脏项＝@@DIRTYSRC@@）。
新增件按**角色**分 @@NBUCKET@@ 桶，桶和＝@@BUCKETSUM@@＝新增总数（一件落进 0 个或 ≥2 个桶都拒绝生成）；
下面每行的**桶名与件数**都取自本件分类用的同一张表（不是先写名字再配数）：
@@BUCKETLIST@@
修改 @@NM@@ 件＝@@NDOC@@ 张归档面＋@@NINSTR@@ 张仪器面（@@INSTRNAMES@@）：开发日志 @@DL0@@→@@DL1@@ 行
（＋@@DLD@@／−@@DLD2@@）、计划文档 @@PL0@@→@@PL1@@（＋@@PLD@@／−@@PLD2@@）、总览 @@OV0@@→@@OV1@@（＋@@OVD@@／−@@OVD2@@）、
当日记忆 @@DY0@@→@@DY1@@（＋@@DYD@@／−@@DYD2@@）；@@NDOC@@ 张归档面合计新增 @@DOCADD@@ 行、@@NINSTR@@ 张仪器面合计 ＋@@INSTRADD@@ 行，
两者之和＝`--numstat` 里 @@NM@@ 张修改面的插入行总数，逐面还满足 `暂存 blob 行数 − HEAD 行数 ＝ numstat 净增`，
且 `git diff --name-only` 为空＝无未暂存漂移。
本轮新节 §26.@@SECN@@ 在 HEAD 里 @@SECH@@ 次、在暂存 blob 里 @@SECSTAGED@@ 次 ⇒ 本节整体随本片落地；
整份日志的 `### 26.NN` 标题数 HEAD＝@@HEADTOT@@ → 暂存＝@@HEADSTAGED@@，且暂存序列＝HEAD 序列＋末位 1 条、
末位节号＝全文件最大。闸门不假设节号单调：HEAD 里本就有 @@NBREAK@@ 处既有降序（@@DESCENT@@），
那是历史状态、本片原样带走，暂存日志的降序处数与 HEAD 相同。
子条 @@NBULLET@@ 条，首＝「@@BULL0@@」、末＝「@@BULLN@@」（由**暂存 blob** 现取现验，不读工作树）。
⚠ 在 repo MEMORY 台账（`@@LEDGER@@`）本轮**未改**（现验＝它的暂存 blob 与 HEAD 版本逐字节相同）⇒ 它不在上面的修改面里，这是事实不是遗漏：
本轮没有新增需要进台账的长期约定（新的口径都写进了 §26.@@SECN@@ 与计划文档）。

A3 腿②的读数面（@@PILOT@@，md5 前 12 @@PILOTMD5@@）——夹具与档位**一律从面里现取**，本件不替它做总结：
「本轮选用（跑前定死）」行＝stage=pilot、strategy=C、tracks=@@TRACKS@@、α=@@ALPHA@@、dx=@@DX1@@µm、arms=['norm']；
dx 与 r_b 的关系由本件断言（现取 r_b=@@RB@@µm ⇒ r_b/2＝dx，误差 <1e-3）；材料卡现取 T_ambient=@@AMB@@K；
N=1 的 nvox=@@NVOX@@、n_steps=@@NSTEPS@@；读数核合成控件 @@C14@@ 在面里 [PASS] @@NCTL4@@ 条／@@NCTLDEN@@（＝"读数本身能失败"那条控件）。
· **可评分面只有 N=1**：solidus 宽 @@N1W@@µm／积 @@N1A@@µm²（窗口 @@N1R@@ 行、触边 @@N1T@@）。
  宽靶中位 @@TW@@µm（n=@@TN@@，极差 @@TRNG@@%，@@TSPAN@@）⇒ 偏差 **@@DEVW@@%**（达标线 ≤@@BARW@@%、未达线 >@@BARFAIL@@%，
  两线逐字取自面里那条 solidus 行并与**待提交驱动**的 `BARS` 互证；距达标线只剩 @@SLACKW@@ 个点＝**刀口**）；
  积靶中位 @@TAMED@@µm²（极差 @@TARNG@@%）⇒ 偏差 **@@TADEV@@%**（达标线 ≤@@BARA@@%、未达线 >@@BARFAIL@@% ⇒ 未达，超出达标线 @@OVERA@@ 个点）。
  靶数**从该道次自己的 solidus 行里取**（整面 search 也会落在第一条＝N=1，但那是运气）：@@NTRACKS@@ 条道次的宽偏差互不相同
  （实得 @@DEVWUNI@@ 个不同值）⇒ "同轨配对"这条闸门看得见差别，不是空转。
· **N≥2 是下界不是读数**：N=2 宽 @@N2W@@／积 @@N2A@@ 而触边 @@N2T@@/@@N2R@@ 行；N=3 宽 @@N3W@@／积 @@N3A@@ 而触边 @@N3T@@/@@N3R@@ 行。
  **道次更多、输入能量更多而积反而更小（@@N3A@@ < @@N2A@@）＝物理不可能** ⇒ 唯一解释是窗口被试片边界切走。
  这条反序是 **#30「加大试片」裁决的直接证据**，也是 `--stage full`（@@NTREND@@ 道趋势靶）本轮**不跑**的判据来源。
· 趋势行（面内原样）：@@TREND@@ 非降=@@TRENDFLAT@@ 末/首=@@TRENRATIO@@；两臂在唯一可评分的 N=1 上：norm @@N1W@@µm vs raw @@RAWN1W@@µm（raw 偏差 @@RAWDEV@@%）⇒ 选臂实打实决定结论。
· **G9 前后的两面互为确定性件**：G9 版与无 G9 版（@@PILOT_PRE@@，md5 前 12 @@PILOT_PREMD5@@）的 @@NTRACKS@@ 条 solidus
  **整行原文**逐字节相同（宽／积／布尔／触边／靶偏差全部一致，不是只比两个数）⇒ 仪器改动没动数值，
  且 @@NTRACKS@@ 次独立进程（墙钟 @@W1@@／@@W2@@／@@W3@@s）在 @@NTRACKS@@ 个 N 上复现同一形态数字。
· 代价登记（可行性口径，**不是性能声明**）：最大 N=@@NMAXC@@ 墙钟 @@W3@@s ⇒ 每道均摊现算 @@PERTRACK@@s（分母＝该腿自己的道次数）、ΣN=1..@@NTREND@@＝@@SUMN@@×均摊 ≈@@SIGMA@@s 对预算 @@BUDGET@@s ⇒ 面内判读「@@COSTVERD@@」；
  本件重算与面里那句由驱动打印的读数（@@FACEPER@@s／≈@@FACETOT@@s）在 ±0.15s／±60s 内一致（断言在生成器里）。CPU 跑得起 full，但 G9 已把 N≥2 判死。

#51 探针（六腿 @@LEGTAB@@，P＝带 [PROBE] 标记；G8a 缺口 A=@@GAPA@@K／D=@@GAPD@@K／E=@@GAPE@@K）：
钳位腿 A 相对忠实腿 B 的形态差 Δ宽=@@DA@@µm；面里宽只印 @@PDEC@@ 位小数 ⇒ 这句话的真值只到半个末位，
即 **|Δ| ≤ @@BOUNDUM@@µm ＝ B 的 @@CLAMPPCT@@%**（不是"严格为零"）。重复腿 C 与 A 的 [PROBE] 行**逐字节相同**（Δ=@@DC@@）
⇒ 那个零不是运行间噪声；极端腿 D Δ=@@DD@@µm ⇒ 探针**不瞎**（有可失败的敏感度对照）；
可表示初温腿 E Δ=@@DE@@µm、F Δ=@@DF@@µm ⇒ 斜率 @@SLOPEE@@／@@SLOPEF@@ µm/K（两腿差 <5%＝线性，分母取各腿自己的
G8a 缺口绝对值），用 F 外推 |E 缺口|=@@GAPEABS@@K＝@@EXTRAP@@µm 对 E 实测 @@DE@@µm（比 @@ERATIO@@，
本件现算＝外推值／实测值，与「两腿斜率差 <5%」那条闸门同一批数）。
**结论**＝钳位对 A3 形态 ≤@@CLAMPPCT@@%，比 @@BARW@@%／@@BARA@@% 的达标线分别小 @@MAGW@@／@@MAGA@@ 倍（现算＝达标线÷本上界）
⇒ A3 关键路径**不**经过生产侧修复；但**降级不关单**：H≤0 在 T(H) 里是一整段平台，冷区绝对温度仍错 @@GAPEABS@@K，
凡按绝对 T／ΔT 取样的下游（α·ΔT 热应变、残余应力、冷却速率）都会带走这个错，且 A≠E 已量化"同报 300K 并不同态"。
六腿归档件 md5：@@ICMD5@@；墙钟 @@WALLMIN@@–@@WALLMAX@@s（同量级，Δ 不是计时噪声）。

smoke 档两条腿（缺省卡＝@@SMOKE@@ md5 前 12 @@SMOKEMD5@@；--amb-k @@AMBS@@＝@@SMOKEAMB@@ md5 前 12 @@SMOKEAMBMD5@@，该值与 pilot 腿面内的 T_ambient 同数）按 J6
**结构性**禁掉形态数字 ⇒ 本件现验：两条面的 `solidus: 宽=` 行各 **@@NSOL@@**／**@@NSOL2@@** 条；amb 腿取而代之印 @@NBOOL@@ 条布尔行
（@@NBOOL@@ 个等值面各一条，每条 @@NBOOLFIELD@@ 个字段：@@BOOLFIELDS@@），且这些行里 `µm` 出现 **@@NBOOLUM@@** 次
⇒ 粗网格数字没有可被误引的出口；缺省卡腿因 G8a FAIL **连布尔行都不印**，只有 @@NROFAIL@@ 条 READOUT_FAIL。
两条腿各报 @@SOFTSMOKE@@／@@SOFTFAIL@@ 条 [SOFT] 闸（标签现取自面＝@@SOFTLAB@@），每条 STAGE_DONE 判读行（共 @@STAGEDONE@@ 条）
自带"软闸"标签 @@SOFTEND@@ 次。
预检回归：复跑 --stage preflight（@@PREREG@@ md5 前 12 @@PREREGMD5@@）与已落地面 diff 恰 @@PREDIFF@@ 行（只剩计时与标签），
把标签 sed 归一后与已落地面 md5 相同（归一件 `@@PREREGN@@`：@@PREREGNMD5@@ ＝ 已落地面 `@@PREORIG@@`：@@PREORIGMD5@@）；复跑里 [SOFT] 出现 @@SOFTREG@@ 次，
而同一条正则在这两条 smoke 腿各命中 @@SOFTSMOKE@@ 次 ⇒ 那个 @@SOFTREG@@ 有分母（软闸的适用面没有漏到 preflight）。

分片 27 回执入库（@@CURL@@，md5 前 12 @@CURLMD5@@；读数一律从回执正文＋表头正则现取，正文窗口锚点
`BODY_OFFSET=@@BOFF@@` 也取自该回执而非本件另猜；表头闭合式与正文 blob 数两处互证）：@@VERDICT@@；
落地树 上一片回执声称的远端 main＝@@CHEAD7@@（@@CHEAD40@@），本件现取的本地 HEAD＝@@HEAD7@@
⇒ 两者同一，本片的父提交就是那棵树（父的父＝@@PARENT7@@，即 @@PARENT40@@）；
tree 条目 @@ENTRIES@@＝blob @@BLOB@@＋tree @@TREEENT@@（truncated=False）；
闭合式 远端 blob @@REMOTE@@ ＝ 父提交 blob @@PBLOB@@ ＋ 该片新增(A) @@A27@@ ＝ @@EXPECT@@ -> PASS；
双向路径差集 @@DL0C@@/@@DLDEN@@ 与 @@DR@@/@@DRDEN@@；全库 blob SHA 不符 @@SHA@@/@@SHADEN@@；
本片单列复核 @@MATCH@@/@@NFILE@@ MATCH（A @@A27@@／M @@M27@@）。@@NCTRL@@ 个正对照（@@CTRLEN@@）同一条运行里都能红：
A 远端一条 sha 首位改 f ⇒ 不符 @@CA0@@/@@CA1@@；B 远端删一条路径 ⇒ 本地独有 @@CB1@@/@@CB1D@@、远端独有 @@CB2@@/@@CB2D@@；
C 闭合式右边 +1 ⇒ @@CC_L@@ == @@CC_R@@ 判 @@CC_V@@；D 远端删**本片**一条路径 ⇒ 本片缺失 @@CD1@@/@@CD1D@@、SHA 不符 @@CD2@@/@@CD2D@@。
回执链 @@NCHAIN@@ 件本轮入库（尺子／回执／`@@RAW27@@`＋`@@RAW27ERR@@`／表头生成器 `@@HDR27@@`）；
`@@RAW27@@` md5 前 12＝@@RAW27MD5@@（`cp -p` 的原始 stdout，表头生成时复跑过并逐字节比对）。
尺子 md5 前 12＝@@RULERMD5@@；**全树** glob＋md5 现算＝同一把尺子已有 @@COPIES@@ 份逐字节副本（含本轮入库这份；
上一片回执声称 @@PREVCOP@@ 份，本片下一把副本要到推送之后才现抄 ⇒ 那时才是 @@COPIESNEXT@@ 份（现算 @@COPIES@@＋1，本件代算），此处不预先声称）。

本轮无代码腿 ⇒ **没重跑全量回归**（重跑只会再量同一棵树），引用分片 26 那次的凭据是**指纹四路闭合**：
现算 src+tests 指纹＝@@FINGER@@＝回归件 SRCFINGER_START＝SRCFINGER_END＝上一片回执表头 SRC指纹（四路任一路断裂即拒绝生成）。
被引用的那次：**@@TALLY@@**（＝@@NF@@ failed＋@@NP@@ passed＋@@NS@@ skipped），墙钟 @@WALL@@，rc=@@RC@@
（rc＝@@RC@@＝@@NF@@ 条既有红灯，非本轮引入），@@D1@@ → @@D2@@，当时的树＝@@REGHEAD@@ dirty=@@REGDIRTY@@（内容级＝现树）。
两条登记红的 E 行（逐字节从 @@REG@@ md5 前 12 @@REGMD5@@ 现取）：①@@E1@@；②@@E2@@。
`grep -ic 'xfail|xpass'`＝@@XFAIL@@、`--collect-only`＝@@COLLECT@@ ⇒ A0 断言未放宽、#27 未换指标、无新增 skip/xfail
（这三条本轮只**引用**那次运行的读数，因为本轮没有新测试可与之比）。

诚实登记（本轮没做什么）：① 无代码腿、无新测试、无回归重跑；② **`--stage full` 没跑**，且这不是排期推出来的而是
判据推出来的（已跑的 @@NTRACKS@@ 面里 @@NUNSC@@ 面被 G9 判废，触边行数 @@N1T@@→@@N2T@@→@@N3T@@ 随道次递增 ⇒
@@NTREND@@ 道只会更触边；**这是从单调趋势外推的推断，不是跑出来的读数**）；③ N≥2 的宽／积**只作下界引用**，不进任何达标句；
④ **性能数字本轮为零**——GPU 仍停放，吞吐/加速/预算类数字只在 A6000 上实测，CPU 侧只报算法与正确性，
本段的墙钟数一律标"可行性代价"；⑤ 生产缺陷 #51 **未修**（只量化）；⑥ **本片（分片 @@SLICE@@）自己的落地证明
要到下一轮才存在**：本信息只声称分片 @@PREVSLICE@@ 那份已落地（回执正文 ①–⑤ 段），不预先声称本片被复核过。
GPU：全轮 CPU 钉住（每条计算命令内联 `CUDA_VISIBLE_DEVICES=""` ＋ `JAX_PLATFORMS=cpu`，含一行命令）；
生成信息时现取卡上 compute 进程 @@NV@@ 个、可读回 @@NVREAD@@/@@NV@@、我的解释器（/proc/PID/cmdline 含 /tmp/amvenv）＝@@NVME@@/@@NV@@（非 0 拒绝生成），
匹配式在**同一条运行**里对合成样例行命中 1（否则那个 0 是假零）；钉住口径出处＝开发日志 §26.@@GPUSEC@@ 那句「继续，还是不动GPU」
（该句在**暂存的**日志里共出现 @@NQUOTE@@ 次，而**最后**一次落在本轮 §26.@@SECN@@ 节内 ⇒ 引的是现行约束，不是历史登记）。
排期：**#30「加大试片」的用户裁决** → pilot 的 raw 臂 N≥2 与 `--stage full --measured-per-track` →
**#27 腿②（缺省 tuned→hk）＋#33 的 15 位点系数重标定同批**，且只对 A3 的 1→@@NTREND@@ 外部趋势靶打分
（单道散差 p50＝宽 @@P50W@@%／积 @@P50A@@%，p90＝@@P90W@@%／@@P90A@@%，max＝@@MAXW@@%／@@MAXA@@%，表内样本数 @@P50N@@，逐字取自**暂存的**开发日志
@@P50CITE@@ 合并口径表；过线判据＝面里那对 @@BARW@@%／@@BARA@@%）；#40 → #39；#41 待自己一轮；#37／#38 随后；
#29 仍等用户裁决。红线不变：A0 断言不放宽、#27 不换指标、无新增 skip/xfail、性能数字只在 A6000 上取。
"""

if __name__ == "__main__":
    m, c = build()
    open("/tmp/am_s28_msg.txt", "w", encoding="utf-8").write(m)
    print(m)
    print("---- 现取核对 ----")
    for k, v in c.items():
        print("%s=%s" % (k, v))
