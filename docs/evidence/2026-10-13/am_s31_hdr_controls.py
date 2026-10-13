#!/usr/bin/env python3
"""表头生成器 `am_s31_hdr.py` 的**控件面**：每条坏输入都要让它拒绝一次，否则那条闸门是空转。

与上一件控件面（`am_s31_hdr_after_controls.py`，验的是**只读仪器**）的根本区别：本面扰动的对象是
**生产者**（写回执表头那件），判据是「它会不会把一份坏表头写进档案」。⇒ 扰动一律落在生成器源码的
取数行与锚句上（把某个 needle 指歪、把一个口径拿错流、把两个出处互换……），而子运行**一律不带
`--write`**：跑的是预检分支，只打印、不落盘。真写盘那条分支的四条闸门在这里够不着，理由与替代证据
登记在 UNCOVERED 第一条。

本面自己的四条口径：
① 扰动副本必须落在**同一个日期目录**（`HERE`）下、用 `zctl_` 前缀命名。原因是生成器把 `REPO` 由
   `__file__` 上溯三层算出来，又用目录名拼脚本路径（`SCRIPT_REL`）——副本放别处就红在路径算错上，
   而不是红在我要验的那条闸门上。前缀 `zctl_` 同时保证副本不被它自己的两处全树 glob 认成尺子
   （`am_s7_verify*.py`）或生成器（`am_s*_hdr.py`）⇒ 副本不改变 NRULE／标本份数这些读数。
② **行号窗口按副本算**：每条扰动都会增删字节、可能挪动行号，所以「该句在第几行」必须从**那一份副本**
   现取，而不是从原始文件取（原始文件的行号与子件 traceback 的行号本来就不同源）。判据不变：rc≠0 且
   **抛出的那一行**含指定闸门句 且 raise 行号落在该句在副本里的 ±3 行窗口。唯一的例外是那条
   `raise SystemExit`（不动点不收敛）：SystemExit 不打 traceback ⇒ 该条改判「末行以 SystemExit: 开头
   且含该句，且全文没有 Traceback」。
③ **没有一次子运行许可能写盘**：每条子运行前后比对档案面 md5，收面之后再比对一次起点快照；任何一张
   变了就从内存备份还原并以 ENV FAILURE 退出。本面跑在真目录里，这条是"探针不咬档案"的凭据。
④ 联网配额是这条腿的**真实约束**（开发日志最后一节里登记的那笔账：表头面每条扰动都要复跑一次联网尺子；
   节号由 `sect_body()` 现取，本面不写死）。
   每条子运行＝1 次 GitHub 未认证 API 调用；起面之前与每次子运行之前读 `/rate_limit`（该端点不计数），
   余量不足就等到窗口重置再继续。等待次数与累计等待秒数印在收面行，余量本身是易失读数、不作分母。
   本片的红路＋绿路比单窗口上限多 ⇒ **整批必然跨窗口**：上一片那句"余量必须一次装下全部条数"在数学上
   不可满足（单窗口上限比条数小），照抄它就等于把整面判成环境故障；`want_runs()` 因此分段——装得下就
   一次装下（防跑到一半被限流、某条红在环境而不是红在指定闸门），装不下就只保证眼下这一次有额度。
④′ **去向分类**（命中／环境／空转）是第二次真跑教出来的：R8 的副本当次死在生成器 `am_s31_hdr.py:69`
   那条复现闸门（"复跑 stdout 与归档正文不同"＝尺子这一轮拿不到与归档一致的回文），却被记成 `×`＝
   "这条闸门空转"。那是把天灾算成凶手的证据，也是把没跑成的判据当成判据失败。`classify()` 因此只在
   红路且抛出的那一行含复现句时才判"环境"，判了环境按 `ENV_RETRY` 有界重跑一次，再犯就计入 envfail、
   整面收 INCOMPLETE，前缀用「！」而不用「×」（`× ` 在本族档案口径里专指"声明的红路没命中"，没跑成
   的那条既没红也没绿、判据本身没跑，没资格占这个记号）。绿路没有这条后门：绿路要的就是读数一致，
   复现闸门对它本来就是判据。N5 那张形状表把两头各验一次，含"rc=0 但输出里带那句字面"与"A1 的期望句
   正是复现闸门本身"这两个最容易写反的形状。

继承上一片的两条常量：子运行上限 300 s、环境故障累计 3 条即停，超时**既不判红也不过**、单独计数。
纸面预检先跑（不起子进程、不花一次 API）：每条期望句在生成器源码里恰好命中 1 处且是某条
`assert`／`raise` 消息的子串、每个锚点恰好命中 1 处、扰动后的副本里该句仍恰好一行、期望句里不许有
格式符；外加**前提普查**——本面的坏输入全靠"某个 needle 命中几条"成立，档案形状一变它就不是坏输入
而是环境变了：偏松的 needle 必须命中 ≥2、偏歪的必须命中 0、HEAD 的 A／M 件数必须不等。
前提普查里最贵的一课来自 F1：备用解释器最初取 `/usr/bin/python3`，而本机有两个长命进程（系统无人值守
升级、终端模拟器）的 argv[0] 正是它 ⇒ F1 的副本先在 `am_s31_hdr.py:249`"整词匹配数到我的解释器进程"
抢红，把本该命中的 `:647` 那条挡在后面空转。凡是**扰动目标在文件后段、而它前面有读机器状态的闸门**，
就要在纸上先普查"这台机器现在会不会让前面那条先响"；备用解释器因此换成 `/usr/bin/python3.12`
（argv[0] 命中数实测 0），并把它连同"它 import 得到生成器全部标准库""被弃用的那个仍数得到进程"一起
写进预检的六条前提。

本片比上一片多出三族坏输入，因为生成器本轮多出三族**新读的档案**：
① GPU 那族（P1–P6）——nvidia-smi 本轮起不来，「我没占卡」改由 `/proc/*/cmdline` 的 argv[0] 整词匹配承担，
   判据是**按构造**的合成对照，所以控件也按构造配：把「仪器坏了」那条断言反向、把 NVML 的 needle 指歪、
   去掉自指排除、把两条合成对照的口径互换。前提普查因此查「仪器确实还坏着＋我自己的 argv[0] 确实等于
   sys.executable＋那条合成命令行确实松散命中而整词不命中」，而不是上一片那句「GPU 上必须有别人的进程」
   （本轮拿它当前提就会把 P 系列的空转误判成环境故障）。
② ⑤ 读数块那族（Q1–Q13）——表头里的"第一次跑拒绝／12 条空转／全过面"三个故事全部改成从**已归档的
   控件面原始件**现取，于是每条出处都要配一枚坏输入（glob 丢 tag、锚句末字改错、needle 指歪、取列口径
   变严、闭合式反向、不剥占位符）。
③ 形制与"上一片"出处那族（R1–R9）——「两片同用新形制」不再是叙述：本节与上一片那一节各自按行首普查
   旧形制条数，上一片的节号从上一片的回执表头现取 ⇒ 过滤放宽、正则收紧、期望值写死、键取错各配一条红。
"""
import ast
import glob
import os
import re
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
GEN = os.path.join(HERE, "am_s31_hdr.py")
SELF = os.path.abspath(__file__)
TEXT = open(GEN, encoding="utf-8").read()
SELFTEXT = open(SELF, encoding="utf-8").read()
DEVLOG = os.path.join(REPO, "docs/开发日志.md")
PREFIX = "zctl_s31_hdr_"
SUB_TIMEOUT = 300
ENV_FAIL_STOP = 3
GREEN_MARK = "（预检模式：未写盘。加 --write。）"
# F1 的"换解释器"要换到一条**本机没有别人在用**的路径上。第一次真跑换的是 /usr/bin/python3，
# 结果副本先在生成器「argv[0] 整词匹配数到我的解释器进程」那条红（本机有两个长驻进程的 argv[0]
# 恰好就是它），抢走了本该落在「CMD 行描述失真」上的红＝抛出顺序吃掉了靶。候选路径的三条前提
# （存在／不在 /tmp/amvenv 下／外来同 argv[0] 进程为 0）由纸面预检现取，见下面的解释器前提普查。
REJECTED_INTERP = "/usr/bin/python3"
ALT_INTERP = "/usr/bin/python3.12"
# 生成器 :69「复跑 stdout 与归档正文不同」是**仪器复现闸门**：副本里的尺子当次没拿到与归档一致的回文
# （限流、网络、别人推送都会走这句）。它不是"这条控件的靶"，所以红路死在这儿要判**环境故障**而不是空转；
# 但对 A1 而言这句就是它自己的靶（它的期望句「仪器不可复现」正是这条断言的后半），所以分类必须先看 ok。
REPRO_MSG = "复跑 stdout 与归档正文不同"
ENV_RETRY = 1
MARGIN = 2
DECOY = "/tmp/_raw_verify31.out"
# ③ 的名单：本面跑在真目录里，凡是子运行**可能**写到的都钉进来（生成器只对 OUT 开过 w）
WRITE_GUARD = [os.path.join(HERE, "am_s7_verify31.log"), os.path.join(HERE, "_raw_verify31.out"),
               os.path.join(HERE, "_raw_verify31.err")]
GUARD = WRITE_GUARD + [os.path.join(HERE, "am_s31_hdr.py"), os.path.join(HERE, "am_s31_hdr_after.py"),
                       os.path.join(HERE, "am_s31_hdr_after_controls.py"), DEVLOG]
ENV = dict(os.environ)
ENV["CUDA_VISIBLE_DEVICES"] = ""
NOOP_EXPECT = "扰动是空转"
SPECLET = "形制约束的落实＝本轮另起 §26.33，不在 §26.32 内追加"

# --------------------------------------------------------------------------- needle 现取
# 前提普查要用的两个 needle 一律从**生成器源码**现取（不在本面重敲一遍，否则生成器改了锚句而本面
# 还按旧锚句查前提＝两条档案各说各话）。本片生成器把锚句存成 `LESSON_NEEDLE = "…"`，上一片是把它
# 直接写在 `bullet_for(r"…", …)` 的调用行里 ⇒ 两种形状都试，但**只许命中一种**（都命中就是本面的
# 取法本身歧义，拒绝）。
def gen_needle(text, tag):
    cands = []
    m = re.search(r'^LESSON_NEEDLE = "([^"]+)"', text, re.M)
    if m:
        cands.append(m.group(1))
    m = re.search(r'LESSONLET, _LSEG = bullet_for\(r"([^"]+)"', text)
    if m:
        cands.append(m.group(1))
    assert len(cands) == 1, "%s的 LESSON 锚句在这份源码里取到 %d 种形状（须 1）：%s ⇒ 取法本身歧义，拒绝" % (
        tag, len(cands), cands)
    return cands[0]


SLICE = int(re.search(r"^am_s(\d+)_hdr\.py$", os.path.basename(GEN)).group(1))
_prev_gen = [p for p in glob.glob(os.path.join(HERE, "../*/am_s%d_hdr.py" % (SLICE - 1)))
             if os.path.basename(p) == "am_s%d_hdr.py" % (SLICE - 1)]
assert len(_prev_gen) == 1, "上一片（%s 号）的表头生成器在兄弟目录里命中 %d 份（须 1）⇒ 锚句没有唯一出处" % (
    SLICE - 1, len(_prev_gen))
LESSON_NEEDLE = gen_needle(TEXT, "本片")
PREV_NEEDLE = gen_needle(open(_prev_gen[0], encoding="utf-8").read(), "上一片")
assert LESSON_NEEDLE != PREV_NEEDLE, "两片 LESSON 锚句相同（都是「%s」）⇒ 本面 C5／C6 那两条对照没有坏输入" % LESSON_NEEDLE
LOOSE_NEEDLE = "回执"


def sect_body():
    dev = open(DEVLOG, encoding="utf-8").read()
    secs = list(re.finditer(r"^### 26\.(\d+)[^\n]*$", dev, re.M))
    assert secs, "开发日志里取不到 `### 26.NN` 标题 ⇒ 前提普查无从做起"
    tail = dev[secs[-1].end():]
    nxt = tail.find("\n### ")
    return dev, (tail if nxt == -1 else tail[:nxt])


def bullet_labels(sec):
    return [m.group(1) for m in re.finditer(r"^- \*\*(.+?)\*\*", sec, re.M)]


def gen_tmpl():
    """从生成器源码里 ast 取出 TMPL 那条赋值的**值**（编号普查要与生成器同口径）。

    为什么不能直接在整份源码文本上切那两句锚：同一短语在源码里还会出现一次——就是生成器自己那两行
    `TMPL.index(…)`，拿源码计次就把基准读成歧义。
    """
    for node in ast.walk(ast.parse(TEXT)):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "TMPL" for t in node.targets):
            try:
                v = ast.literal_eval(node.value)
            except Exception:
                return None
            return v if isinstance(v, str) else None
    return None


def form_census(dev):
    """按节普查两种子条形制的**条数**（与生成器同口径：节号 → (新形制条数, 旧形制条数)）。

    本面要拿它当 R5／R6／R7 的前提：如果"换计数键"或"丢最后一段"根本不改变那几个 max，
    那两条扰动就是空转，本面就没资格给它们判红。
    """
    secs = list(re.finditer(r"^### 26\.(\d+)[^\n]*$", dev, re.M))
    out = {}
    for i, m in enumerate(secs):
        end = secs[i + 1].start() if i + 1 < len(secs) else len(dev)
        seg = dev[m.end():end]
        out[m.group(1)] = (len(re.findall(r"^- \*\*(.+?)\*\*", seg, re.M)),
                           len(re.findall(r"^\*\*\((\w)\)", seg, re.M)))
    return out


def bullet_hits(sec, pat):
    ls = bullet_labels(sec)
    n = 0
    for i, let in enumerate(ls):
        st = sec.index("- **%s**" % let)
        en = sec.index("- **%s**" % ls[i + 1]) if i + 1 < len(ls) else len(sec)
        if re.search(pat, sec[st:en]):
            n += 1
    return n


def assert_msgs():
    """生成器源码里 assert／raise AssertionError／raise SystemExit 的消息文本（含 % 拼接的原始片段）。

    那条「BODY_OFFSET 未收敛」用 `raise SystemExit(...)`，不在 `assert` 里；上一面栽过这个盲区。
    """
    out = []
    for node in ast.walk(ast.parse(TEXT)):
        msg = None
        if isinstance(node, ast.Assert):
            msg = node.msg
        elif isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call) \
                and getattr(node.exc.func, "id", "") in ("AssertionError", "SystemExit"):
            msg = node.exc.args[0] if node.exc.args else None
        if msg is None:
            continue
        try:
            v = ast.literal_eval(msg)
            # literal_eval 会把「两条相邻字符串常量」并成一条，也可能返回**非字符串**（某些消息的
            # 左边是一个数字字面量的拼接形状）⇒ 非字符串就退回源码片段，别把消息集里塞进 int
            out.append(v if isinstance(v, str) else (ast.get_source_segment(TEXT, msg) or ""))
        except Exception:
            out.append(ast.get_source_segment(TEXT, msg) or "")
    return out


# --------------------------------------------------------------------------- 扰动表
# (名字, [(锚, 换), …], 期望闸门句, 旗标)；锚在源码里必须恰好命中 1 处，旗标见 run_copy
CASES = [
    # ---- A／B：复现面与入库清单面（沿用上一片的坏输入形状，锚句全部重新指认过） ----
    ("A1 归档正文少一个字节（复现对照应红）",
     (('body = open(RAW, encoding="utf-8").read()',
       'body = open(RAW, encoding="utf-8").read()[:-1]'),), "仪器不可复现", {}),
    ("A2 stderr 的字节数拿成 stdout 那条流",
     (('err_bytes = len(rp.stderr.encode())', 'err_bytes = len(rp.stdout.encode())'),),
     "复跑核验件 rc/stderr 不齐", {}),
    ("A3 全过判读行的常量多写一个 X",
     (('"LANDING_VERDICT = ALL CHECKS PASS" in body',
       '"LANDING_VERDICT = ALL CHECKS PASSX" in body'),), "正文里没有全过的判读行", {}),
    ("A4 /tmp 里放一枚同名而不同内容的原始件（cp -p 凭据应失效）",
     (), "归档件不是那次运行的原始 stdout", {"decoy": True}),
    ("B1 尺子全树 glob 的 needle 指歪（数不到自己）",
     (('"docs/evidence/*/am_s7_verify*.py"', '"docs/evidence/*/am_s7_verify9[0-9].py"'),),
     "计数口径失真", {}),
    ("B2 取 A 的口径放宽成 A＋M（状态字母闭合应红）",
     (('added = [r[-1] for r in rows if r[0] == "A"]',
       'added = [r[-1] for r in rows if r[0] in ("A", "M")]'),), "之外的状态字母", {}),
    ("B3 三分类里「/tmp 已无同名件」那一类静默丢件",
     (('        md_absent.append(p)', '        md_absent.extend(())'),),
     "三分类没覆盖全部新增件", {}),
    ("B4 入库清单两个出处互证时左右互换",
     (('assert (NA_BODY, NM_BODY) == (len(added), len(mods)), (',
       'assert (NA_BODY, NM_BODY) == (len(mods), len(added)), ('),), "有一侧的清单口径变了", {}),
    # ---- C：凡是"从档案现取"的锚句，各配一枚指歪／放宽的坏输入 ----
    ("C1 片号互证的文件名一侧回退一片",
     (('"am_s7_verify%s.py" % SLICE', '"am_s7_verify%s.py" % (int(SLICE) - 1)'),),
     "安在本片头上，拒绝归档", {}),
    ("C2 节号锚句指歪成 `### 99.NN`（现取应取空）",
     ((r'"^### 26\.(\d+)[^\n]*$"', r'"^### 99\.(\d+)[^\n]*$"'),), "节号是凭记忆的", {}),
    ("C3 子条形制锚句指回上一轮的 `**(x)**`（本轮该形制为 0 条）",
     ((r'BULLETS = [m.group(1) for m in re.finditer(r"^- \*\*(.+?)\*\*", SEC_BODY, re.M)]',
       r'BULLETS = [m.group(1) for m in re.finditer(r"^\*\*\((\w)\)", SEC_BODY, re.M)]'),),
     "本轮日志形制又变了", {}),
    ("C4 日志里出现两条同名子条（切段会串行）",
     (('DEV = open(DEVLOG, encoding="utf-8").read()',
       'DEV = open(DEVLOG, encoding="utf-8").read() + "\\n- **%s**\\n"' % SPECLET),),
     "由标签切段会串行", {}),
    ("C5 LESSON 锚句换成上一片那条（本节查无此句⇒命中 0 条）",
     (('LESSON_NEEDLE = "%s"' % LESSON_NEEDLE, 'LESSON_NEEDLE = "%s"' % PREV_NEEDLE),),
     "引用出处不唯一", {}),
    ("C6 LESSON 锚句放宽成「回执」二字（同一句命中多条）",
     (('LESSON_NEEDLE = "%s"' % LESSON_NEEDLE, 'LESSON_NEEDLE = "%s"' % LOOSE_NEEDLE),),
     "引用出处不唯一", {}),
    ("C7 钉住口径的原文锚句指歪（「当前「…」」取不到）",
     (('re.search(r"当前「([^」]+)」", SEC_BODY)', 're.search(r"从未「([^」]+)」", SEC_BODY)'),),
     "这种钉住口径原文", {}),
    ("C8 f-string 成因句的定位锚句放宽成裸词（出处不再唯一）",
     ((r'r"f-string 把 `\{32\}` 当成\*\*替换字段"', r'r"f-string"'),),
     "f-string 事故句在开发日志里出现", {}),
    ("C9 标本生成器的定位锚句放宽成裸词 PLACEHOLDER（多份命中）",
     ((r'if re.search(r"本片的实物证据＝\*\*改之前\*\*那句", open(',
       'if re.search("PLACEHOLDER", open('),), "标本原文所在生成器有", {}),
    ("C10 标本那句的两个片号取自同一个捕获组",
     (('SPECOUT, SPECIN = _g.group(1), _g.group(2)',
       'SPECOUT, SPECIN = _g.group(1), _g.group(1)'),), "事故不存在", {}),
    ("C11 标本取句的引号锚点变松（只截到四个字，两数取不到）",
     (('那句「(.+?)」",', '那句(.{4})",'),), "标本不是那次事故", {}),
    # ---- D：回执链节奏（一片一环）的三个读数各配一枚坏输入 ----
    ("D1 回执链计数丢掉 int／str 归一（分组是字符串，int 永不相等）",
     (('    n = str(n)  #', '    n = int(n)  #'),), "回执链在 HEAD 里只", {}),
    ("D2 再上一片的链去 HEAD 里查（本该查父提交）",
     (('"--format=", "HEAD^")', '"--format=", "HEAD")'),), "节奏不是一片一环", {}),
    ("D3 本片的链拿上一片的片号去查（那句「留给分片」应成假话）",
     (('OWNHIT = len(chain_ids(SLICE, prev_rows))',
       'OWNHIT = len(chain_ids(PREVSLICE, prev_rows))'),), "本片回执链里有", {}),
    # ---- E／F：渲染面与运行环境 ----
    ("E1 vals 里删掉一个键（模板有槽位没取值）",
     (('"SLICE": SLICE, "NEXTSLICE": NEXTSLICE,', '"NEXTSLICE": NEXTSLICE,'),),
     "模板有未取值的字段", {}),
    ("E2 一枚已过闸的读数里带进半角占位符（渲染残留应红）",
     (('    "MDNOTE": MDNOTE,', '    "MDNOTE": MDNOTE + "@@Z@@",'),),
     "渲染出的表头残留 PLACEHOLDER", {}),
    ("E3 表头里粘连一条缩进的「标签＋冒号」行",
     (('TMPL = """生成时刻: @@NOW@@',
       'TMPL = """  复核    : 这条是假的粘连行\\n生成时刻: @@NOW@@'),), "漏了续行缩进", {}),
    ("E4 表头里塞一条顶格而非字段的行",
     (('TMPL = """生成时刻: @@NOW@@',
       'TMPL = """这一行顶格但不是字段\\n生成时刻: @@NOW@@'),), "顶格却不像字段行", {}),
    ("E5 FIELDS 声明比表头多一条（计数器应红）",
     (('FIELDS = ["生成时刻", "CMD",', 'FIELDS = ["生成时刻", "额外字段", "CMD",'),),
     "有字段漏写或改名", {}),
    ("E6 不动点迭代上限砍到 1（表头含这一行自身，必不收敛）",
     (('for _ in range(12):', 'for _ in range(1):'),), "BODY_OFFSET 未收敛", {"notb": True}),
    ("F1 用系统解释器跑副本（CMD 行的描述就该失真）", (), "CMD 行描述失真", {"interp": True}),
    ("F2 不带空串 CUDA_VISIBLE_DEVICES 跑副本（ENV 行是假话）", (),
     "生成器自身未带空串 CVD", {"no_cvd": True}),
    # ---- P：GPU 面（本轮换了仪器：nvidia-smi 起不来 ⇒ 断言的是「它仍然起不来」＋ /proc 整词匹配） ----
    ("P1 「nvidia-smi 本轮又能用了」那条断言反向（仪器坏了才是本轮的事实）",
     (('assert nv.returncode != 0,', 'assert nv.returncode == 0,'),), "本轮又能用了", {}),
    ("P2 失败原因的 needle 指歪（NVML 换成查无此串）",
     (('assert "NVML" in NVWHY', 'assert "NVMLZZZ" in NVWHY'),), "的失败原因换了", {}),
    ("P3 整词普查不排除自己与父进程（那条 0 的正对照）",
     (('    if int(d) in (me, myparent):', '    if int(d) in ():'),),
     "整词匹配数到我的解释器进程", {}),
    ("P4 负对照改用松散口径（含路径就算命中⇒构造对照失效）",
     (('CTLNEG = 1 if argv0(SYN_BLOB) == INTERP else 0',
       'CTLNEG = 1 if INTERP in SYN_BLOB.decode("utf-8", "replace") else 0'),),
     "两条对照失效", {}),
    ("P5 松散口径改用整词取值（误指形状取不到）",
     (('CTLLOOSE = 1 if INTERP in SYN_BLOB.decode("utf-8", "replace").replace("\\x00", " ") else 0',
       'CTLLOOSE = 1 if argv0(SYN_BLOB) == INTERP else 0'),), "没有误指", {}),
    ("P6 未跟踪行的对照 needle 换成不相干前缀（两条写法不再是同一件事，那条 0 应红）",
     (('and not l.startswith("?? ")])', 'and not l.startswith(" M ")])'),),
     "行以 `??` 开头却不以", {}),
    # ---- Q：⑤ 读数块（数字全部从已归档的控件面现取，每条闸门各配一个坏输入） ----
    ("Q1 控件面 glob 丢掉 tag 段（三种角色命中同一批文件）",
     (('"" if not tag else "." + tag + "-*"', '"" if not tag else "*"'),),
     "引用哪一份都不唯一", {}),
    ("Q2 num 的一条锚句末字改错（保持绿的条数取不到）",
     ((r'r"保持绿＝(\d+) 条"', r'r"保持绿＝(\d+) 个"'),), "本件不许凭记忆写数", {}),
    ("Q3 标本面的顶格 × 行 needle 指歪（红面被当成不是红面）",
     ((r'X2 = re.findall(r"^× ", SP2, re.M)', r'X2 = re.findall(r"^× Q ", SP2, re.M)'),),
     "里没有顶格", {}),
    ("Q4 × 行的编号取列口径变严（单位数编号全丢）",
     ((r'r"^× ([CG]\d+) "', r'r"^× ([CG]\d\d+) "'),), "但带编号的只有", {}),
    ("Q5 全过面与标本面的判读行比对反向（相同也算过）",
     (('assert VERD2 != VERD3', 'assert VERD2 == VERD3'),), "判读行相同", {}),
    ("Q6 全过面的 × 行 needle 放宽成任意两字节（绿面被当成红面）",
     (('assert not re.findall(r"^× ", SP3, re.M)', 'assert not re.findall(r"^..", SP3, re.M)'),),
     "里有顶格", {}),
    ("Q7 编号普查不先剥占位符（槽位名 @@KC2@@ 里的 C2 被当成点名的控件号）",
     ((r'_c5 = re.sub(r"@@[A-Z0-9_]+@@", "", _c5)', '_c5 = _c5'),), "那段点名的控件号", {}),
    ("Q8 标本行里方括号锚点换成书名号（C1 那条拒绝取不到）",
     ((r'r"在生成器源码里 (\d+) 处（\[(\d+), (\d+)\]）"',
       r'r"在生成器源码里 (\d+) 处《\[(\d+), (\d+)\]》"'),), "那条拒绝没有出处", {}),
    ("Q9 预检处数与行号的闭合式期望值写错（2 处说成 3 处）",
     (('assert int(KTWICE) == 2 == len({KLINE1, KLINE2})',
       'assert int(KTWICE) == 3 == len({KLINE1, KLINE2})'),), "预检印的处数", {}),
    ("Q10 松散现数锚句的量词改错（个→行⇒日志里取不到）",
     ((r'r"松散子串匹配现数 (\d+) 个"', r'r"松散子串匹配现数 (\d+) 行"'),),
     "那段少了第二出处", {}),
    ("Q11 锚点普查那条的 needle 指歪（「超时必须为零」换成查无此句）",
     ((r'''_mrc = re.search(r"^\s*(C\d+) 超时必须为零的那条链改成容忍一条", SP3, re.M)''',
       r'''_mrc = re.search(r"^\s*(C\d+) 查无此句的那条链改成容忍一条", SP3, re.M)'''),),
     "那条锚点普查", {}),
    ("Q12 口径原文的锚句指歪（「一条对照必须」换成「一条对照无须」）",
     ((r'r"一条对照必须[^\n]*?。"', r'r"一条对照无须[^\n]*?。"'),), "那句口径原文", {}),
    ("Q13 顶格式 C1 预检行的行首锚点变松（不再是那一条的形状）",
     ((r'r"^  · C1 "', r'r"^·C1 "'),), "顶格式那条 C1 预检行", {}),
    # ---- R：形制普查与「上一片」的两处出处 ----
    ("R1 上一片回执的过滤放宽成全收 .log（那一节的节号出处不再唯一）",
     (('if os.path.basename(p) == PREVLOG]', 'if p.endswith(".log")]'),),
     "那一节的节号取不到唯一出处", {}),
    ("R2 上一片生成器的 glob 与文件名口径同时放宽成任意片号（出处不再唯一）",
     (('"docs/evidence/*/am_s%d_hdr.py" % PREVSLICE', '"docs/evidence/*/am_s*_hdr.py"'),
      (r'"am_s%d_hdr\.py" % PREVSLICE', r'"am_s\d+_hdr\.py"')),
     "那句「换过锚句」没了出处", {}),
    ("R3 从上一片源码取锚句的那条正则收紧（形状变了就取不到）",
     ((r'''LESSONLET, _LSEG = bullet_for\(r"([^"]+)"''',
       r'''LESSONLET, _LSEG = bullet_for\(r"([^"]+)Z"'''),),
     "取不到 LESSON 锚句那行", {}),
    ("R4 「两片锚句必须不同」那条断言反向（相同也算过⇒那句「换过」成假话）",
     (('assert LESSONPREV != LESSON_NEEDLE', 'assert LESSONPREV == LESSON_NEEDLE'),),
     "那句「从 X 换成本节的 Y」是假话", {}),
    ("R5 按节普查把本节那一段丢掉（本节的节号不在表里）",
     (('for _i, _m in enumerate(_secs):', 'for _i, _m in enumerate(_secs[:-1]):'),),
     "不在按节普查的表里", {}),
    ("R6 「旧形制最后出现的节」改用新形制计数（顺序断言应红）",
     (('for n, (_new, old) in _FMC.items() if old > 0)',
       'for n, (_new, old) in _FMC.items() if _new > 0)'),), "不成序", {}),
    ("R7 两片同形制那条的期望值写死成非零（现数是 0 条也应红）",
     (('assert (FSECOLD, PSECOLD) == (0, 0)', 'assert (FSECOLD, PSECOLD) == (7, 8)'),),
     "还数得出旧形制子条", {}),
    ("R8 上一片那一节的节号写死成查无此节的数字（普查里没有它）",
     (('PREVSECN = _psecs[0]', 'PREVSECN = "99"'),), "在开发日志里普查不到", {}),
    ("R9 那一节节号的唯一性期望写死成 0（现取到 1 个也说成出处不唯一）",
     (('assert len(_psecs) == 1', 'assert len(_psecs) == 0'),), "个不同节号", {}),
]

# 良性对照（须保持绿）：每条排除一种"把无关形状当成错误"的共因。
# 与只读仪器那面不同，这里**存在**"改表头还能绿"的方向：本件跑预检分支，BODY_OFFSET 由不动点现算，
# 盘上那份回执不参与比对 ⇒ 表头长一行不叫错位。跨出处闭合（写盘面字节数 ↔ 盘上回执）属于只读仪器，
# 已由 am_s31_hdr_after_controls 的 D1 在同一份档案上造过红。
GREENS = [
    ("G1 未扰动副本（同目录、同解释器、不带 --write）", (), None, {"green": True}),
    ("G2 良性：源码末尾追加一行注释（不动任何读数）",
     ((TEXT.rstrip("\n"), TEXT.rstrip("\n") + "\n# 这条注释不动任何读数，只验本件不因无关形状而拒绝"),),
     None, {"green": True}),
    ("G3 良性：表头里加一条缩进续行（不含标签冒号，偏移由不动点重算）",
     (('TMPL = """生成时刻: @@NOW@@',
       'TMPL = """生成时刻: @@NOW@@\\n          （这条是良性续行：缩进、不含标签冒号）'),),
     None, {"green": True}),
    ("G4 良性：⑤ 那段改一处措辞（不在两个切片锚句上、也不动任何读数）",
     (("重新指认后重跑的那张 @@SPEC3@@", "重新指认之后重跑的那一张 @@SPEC3@@"),),
     None, {"green": True}),
    ("G5 良性：vals 里多一枚模板没用到的键（`missing` 那条只查单向，不该反过来拒绝）",
     (('    "MDNOTE": MDNOTE,', '    "MDNOTE": MDNOTE, "ZUNUSED": "模板里没有这个槽位",'),),
     None, {"green": True}),
]

UNCOVERED = [
    ("写盘分支的四条（正文区逐字节复核失败／回读出的表头残留 PLACEHOLDER／表头含全角 at 号／"
     "BODY_OFFSET ≠ 实际表头字节数／cmp 三联 rc 0／1／1）",
     "本面一次都不传 `--write`：副本与真档案同目录，让它写盘就是把本片将要生成的回执覆盖掉。这几条判据的"
     "**盘上那侧**归同目录的只读仪器 `am_s31_hdr_after.py` 与其控件面 `am_s31_hdr_after_controls.py`——"
     "**本面运行时那两张还不存在**（落地节奏：控件面全过 → 表头写盘 → 只读仪器 → 只读仪器的控件面），"
     "所以这里不声称它们验过什么、也不引用它们的条数；本件这几行**代码路径**本身没被造红 ⇒ 如实登记。"),
    ("「渲染出的表头残留 PLACEHOLDER」只能从值一侧造红，模板一侧造不出",
     "模板里任何一枚槽位没取值都会先撞「模板有未取值的字段」（那条 `missing` 检查在渲染之前），"
     "所以 E2 走的是「一枚已过闸的值里带进新占位符」这一侧。要造模板侧的红就得同时删掉 `missing` 那条"
     "闸门——那不是坏输入而是拆闸门，红了也不代表闸门有效。"),
    ("`missing` 只查单向：模板有槽位、vals 没取值会红，反过来（vals 里有键而模板里没槽位）不会",
     "G5 就是按这个方向做的**良性**对照——它证明那条闸门没被写成双向集合相等（写死就拒绝了）。本轮没有"
     "「定义了却没用的槽位」这类闸门，所以也没有它的控件；下一轮若要加，得先加闸门再配控件，不是先配控件。"),
    ("「节号写死成一个数字」这类坏输入没有闸门读它",
     "C2 造红的是「现取的锚句指歪 ⇒ 取空」那一侧。把 `SECN = _secs[-1]` 换成写死的数字**不会红**："
     "生成器里没有任何一条断言把 SECN 与子条所在节比对，SECN 只被印出来。⇒ 这一侧的保护是"
     "「现取」这个做法本身而不是闸门；下一轮若要把写死也拦住，得先加一条 SECN ↔ bullet 所属节的比对，"
     "再给它配控件。这条是本轮实测到的**保护缺口**，不是漏测。（对比：`PREVSECN` 的写死**有**闸门读它，"
     "就是 R8 造红的那条 `PREVSECN in _FMC` ⇒ 同一个「写死」能不能拦，取决于有没有第二条断言与它比对。）"),
    ("GPU 那两条（P1／P2）的**前提**是本轮环境，不是生成器的判据",
     "本轮 nvidia-smi 起不来（NVML 与内核模块版本不匹配），所以 P1 造的红是「把『它仍然起不来』这条断言反向」"
     "——它证明那条断言真在读 rc，不证明「真能用时表头会怎么写」。如果下一轮仪器修好了，本条与生成器那句"
     "注文要**一起**改写（前提普查里那条「nvidia-smi 本轮又能用了」会先拦住本面，不让它拿旧前提判红）。"),
    ("「真有人占 GPU 时那条红线会不会拒绝」造不出",
     "让 `nv_mine` 真非空的唯一办法是我的解释器在 GPU 上留进程，那是红线（本面全程 CPU 钉住，且预检里现取"
     "本面自己的 argv[0] 必须等于 sys.executable 才肯起子进程）。P3 因此按「去掉自指排除 ⇒ 那条 0 必须响」"
     "造红，它证明的是**整词口径读得到我自己的进程**（那条 0 有分母、不是仪器坏了换来的假零），"
     "不是「别人占卡时它会拒绝」。"),
    ("现场松散口径的读数不进判据（所以它随轮变化也不会红）",
     "生成器那句「松散子串把 shell 也认进来」的判据是**按构造**的那一对（CTLLOOSE／CTLNEG，P4／P5 造红的就是"
     "它），现场数出几个只作对照印出。⇒ 现场读数为 0 的那一轮（就是本轮）不会红，也不该红；把它写成判据"
     "就复现了上一片那条「拿易失读数当分母」的错。"),
    ("⑤ 那段的编号互证只对**本轮档案形状**成立",
     "Q7／Q4 的红证明的是「点名集合 ↔ 标本面 × 行集合」这条闭式在读数（不是在读我的记忆）。它是**同轮**闭合："
     "下一轮的 ⑤ 段必然点名下一轮自己的控件号，届时若集合不等，红的是注文该重写，而不是闸门失效。"
     "本轮没有、也无法有一条跨轮的编号闸门。"),
    ("③ 那条 md5 写保护是本面的自有闸门，不进 CASES",
     "它由两处自查兜着：每条子运行前后的逐张比对，与收面时与起点快照的再一次全量比对（N2 验的就是"
     "`diff_snap` 两个方向都灵：不一致抓得出、一致不报警——上一片那一版把 N2 写反了，正着跑必然"
     "走进 except 之外，于是把自己的绿判成红，一整轮跑完却收在收面段）。它验的是"
     "「本面没咬档案」，不是生成器的某条断言。"),
]

Waits = []
Retries = []
EnvKinds = []
Made = []


def apply_subs(text, subs, name):
    """施加扰动；每条锚必须恰好命中 1 处，且扰动后字节必须真的变了（空转普查）。"""
    out = text
    for old, new in subs:
        n = out.count(old)
        assert n == 1, "锚点在源码里命中 %d 处（须 1）：%s｜锚=%s" % (n, name, old[:60])
        out = out.replace(old, new, 1)
    assert out != text, NOOP_EXPECT + "：副本与生成器逐字节相同（%s）" % name
    return out


def copy_text(subs, name):
    """空扰动（A4／F1／F2／G1 那几条靠**环境或外挂文件**做坏输入）＝原样副本，不走空转普查。"""
    return apply_subs(TEXT, subs, name) if subs else TEXT


def build_copy(text_out, tag):
    path = os.path.join(HERE, "%s%s.py" % (PREFIX, tag))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text_out)
    return path


def md5(path):
    return subprocess.run(["md5sum", path], capture_output=True, text=True).stdout.split()[0]


def snapshot():
    return {p: md5(p) for p in GUARD if os.path.exists(p)}


def backup_writable():
    return {p: open(p, "rb").read() for p in WRITE_GUARD if os.path.exists(p)}


def diff_snap(a, b):
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def run_copy(path, flags):
    env = dict(ENV)
    if flags.get("no_cvd"):
        env.pop("CUDA_VISIBLE_DEVICES", None)
    interp = ALT_INTERP if flags.get("interp") else sys.executable
    p = subprocess.Popen([interp, path], cwd=REPO, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        out, err = p.communicate(timeout=SUB_TIMEOUT)
        return p.returncode, (err or "") + (out or "")
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        out, err = p.communicate()
        return "TIMEOUT", (err or "") + (out or "")


def raised(msg):
    ls = [l for l in msg.splitlines() if l.strip()]
    for l in reversed(ls):
        if re.search(r"(AssertionError|UnicodeDecodeError|SystemExit|SyntaxError|Error)\b", l) \
                and not l.strip().startswith("File "):
            return l
    return ls[-1] if ls else ""


def trace_line(msg):
    hit = re.findall(r'File "[^"]*/%s[^"]*", line (\d+)' % re.escape(PREFIX), msg)
    return int(hit[-1]) if hit else None


def gate_lines(txt, expect):
    return [n + 1 for n, l in enumerate(txt.splitlines()) if expect in l]


def notb_hit(msg, rc, expect, gl):
    """SystemExit 那类闸门的判据：闸门句要落在**末条非空行**、且这一下退出没有 traceback。

    为什么不能写成 `err.startswith("SystemExit:")`（本面第二次真跑就是栽在这上面）：
    CPython 对未捕获的 `SystemExit(str)` 只把消息原样打到 stderr，既不加 `SystemExit:` 前缀、
    也不打 traceback ⇒ 那个合取对**任何**正确实现恒假，E6 的闸门明明响了（rc=1、末行就是那句）
    却被判成「未命中」。这里读的是真实形状：rc≠0 ＋ 期望句在末条非空行 ＋ 全文无 Traceback／无
    AssertionError（assert 类失败必带 traceback 且必写 AssertionError）＋ 该闸门句在副本里恰好 1 行。
    """
    ls = [l for l in msg.splitlines() if l.strip()]
    return (rc != 0 and bool(ls) and expect in ls[-1] and "Traceback" not in msg
            and "AssertionError" not in msg and len(gl) == 1)


def classify(ok, rc, err, green):
    """一条子运行的三个去向，先看"命没命中自己的闸门"，再看"是不是被仪器复现闸门抢了红"。

    为什么顺序不能反：A1 的期望句「仪器不可复现」正是生成器那条复现断言的后半，如果先查
    `REPRO_MSG in err`，A1 那条**本来就该红**的命中会被判成环境故障而整条不算（＝把判据自己
    的空转说成天灾）。反过来，一条死在复现闸门上的别的控件也不能记 `×`：那句红说明副本压根
    没跑到自己的靶（限流、网络、别人推送都能造成），它是"没跑成"而不是"跑成了但没红"。
    环境那条只许重跑一次（ENV_RETRY），再犯就计入 envfail⇒整面收 INCOMPLETE，绝不悄悄放过。
    """
    if ok:
        return "命中"
    if not green and rc != 0 and REPRO_MSG in err:
        return "环境"
    return "空转"


def argv0_hits(target, exclude):
    """整词扫 /proc/*/cmdline 的 argv[0]，返回等于 target 的 PID（排除 exclude 里的 pid）。

    与生成器同一条口径（NUL 分隔取第一个 token 整词比对，不做子串匹配），只是这里的目标是
    "候选备用解释器有没有别人在跑"，不是"我自己的进程"。
    """
    me = {str(p) for p in exclude}
    hit = []
    for d in os.listdir("/proc"):
        if not d.isdigit() or d in me:
            continue
        try:
            with open(os.path.join("/proc", d, "cmdline"), "rb") as fh:
                blob = fh.read()
        except OSError:
            continue
        if not blob:
            continue
        if blob.split(b"\0", 1)[0].decode("utf-8", "replace") == target:
            hit.append(d)
    return sorted(hit)


# F1 的备用解释器读数（导入时现取一次，纸面预检据此拒绝起面，主面把这些计数印成对照）：
# 探针先跑完再扫 /proc，免得把自己的子进程数进去。
ALT_PROBE = subprocess.run(
    [ALT_INTERP, "-c", "import sys, json, re, subprocess, urllib.request; print(sys.executable)"],
    cwd=REPO, capture_output=True, text=True)
ALT_RC = ALT_PROBE.returncode
ALT_OUT = (ALT_PROBE.stdout.strip().splitlines() or [""])[-1]
ALT_ERRTAIL = " ".join(ALT_PROBE.stderr.strip().splitlines()[-2:])
ALT_FOREIGN = argv0_hits(ALT_INTERP, (os.getpid(), os.getppid()))
REJ_FOREIGN = argv0_hits(REJECTED_INTERP, (os.getpid(), os.getppid()))


def quota(tries=5, pause=15):
    """读 GitHub 的配额响应头（该端点不计数）。**带重试**：本面会在等完一个窗口之后再读一次，
    那一次离上次网络活动已隔了近一小时，一次 curl 的空回文（DNS／TLS／瞬时抖动）不代表环境不给跑。
    第一次真跑实测到：等完 3347 秒之后 `curl -s -i` 回了空串 ⇒ 三个头一个都没读到 ⇒ 整面在配额闸门里
    崩掉，跑到第 4 条为止的逐条读数留下了、后面的整批没了。⇒ 判"环境不给跑"之前必须先证明自己读过。
    """
    last = ""
    for i in range(tries):
        o = subprocess.run(["curl", "-s", "-i", "--max-time", "60",
                            "https://api.github.com/rate_limit"],
                           capture_output=True, text=True).stdout
        m = re.search(r"x-ratelimit-remaining: (\d+)", o, re.I)
        r = re.search(r"x-ratelimit-reset: (\d+)", o, re.I)
        lm = re.search(r"x-ratelimit-limit: (\d+)", o, re.I)
        if m and r and lm:
            return int(m.group(1)), int(r.group(1)), int(lm.group(1))
        last = "｜".join(o.strip().splitlines()[:2])
        if i + 1 < tries:
            print("  ！配额读数第 %d 次没读到响应头（回文 %r）⇒ 停 %d 秒后重试（读不到不判红也不判过）"
                  % (i + 1, last[:60], pause))
            time.sleep(pause)
    raise AssertionError("读不到 GitHub 配额响应头（试了 %d 次，最后一次回文 %r）⇒ 配额闸门没法工作，"
                         "本件一条子运行都不该起" % (tries, last[:80]))


def want_runs(need, limit, batch):
    """本轮该要求多少余量：整批装得进一个窗口就一次装下，装不进就只保证眼下这一次有额度。

    上一片的判据是 `rem ≥ 剩余条数`。本片的红路＋绿路比未认证上限多，那个式子**按构造不可能成立**
    （同族错误＝上一片 E6 要求 SystemExit 带 `SystemExit:` 前缀：判据不可满足，红就不是闸门而是判据）。
    "不许跑到一半被限流"这条保护在这里仍然保留，但只在它可满足时才用：整批 ≤ 单窗口上限时，
    起面之前就必须有覆盖整批的余量；整批 > 上限时任何窗口都装不下，就退化成"每条之前至少还剩
    MARGIN 次"——跑空了等窗口重置再继续，跨几个窗口由 Waits 与收面那行印出来当账。
    MARGIN 取 2 而不是 1 是本面第一次跨窗口的实测教训：只保证"还剩 1 次"时，那 1 次会被**别人**
    （这台机器上还有别的未认证消费者）吃掉，于是尺子拿 403 回来、副本死在 :69 的复现闸门上。
    最后一条不足 MARGIN 时按 need 走，不许反过来把收尾那条卡死。
    """
    return need if batch <= limit else min(need, MARGIN)


def ensure_quota(need, batch):
    rem, reset, limit = quota()
    want = want_runs(need, limit, batch)
    if rem >= want:
        return
    wait = max(0, reset - int(time.time()) + 5)
    if wait > 3660:
        print("CONTROL_VERDICT = ENV FAILURE（配额余量 %d＜本轮需要 %d，窗口重置还要 %d 秒——超出一个小时，"
              "那不是等待能解决的环境）" % (rem, want, wait))
        sys.exit(4)
    Waits.append(wait)
    print("  ！配额闸门：余量 %d＜本轮需要 %d（剩余条数 %d｜整批 %d｜单窗口上限 %d ⇒ %s）⇒ "
          "等待窗口重置 %d 秒后继续（等待既不判红也不判过）"
          % (rem, want, need, batch, limit,
             "整批装得进一个窗口" if batch <= limit else "整批装不进一个窗口 ⇒ 每条之前只保证 %d 次额度" % MARGIN, wait))
    time.sleep(wait)
    rem2, _, _ = quota()
    assert rem2 >= want, "等待之后余量仍不足（%d＜%d）⇒ 环境确实不给跑，本件没资格判闸门" % (rem2, want)


def precheck():
    errs = []
    dev, sec = sect_body()
    msgs = assert_msgs()
    for e in sorted(set([c[2] for c in CASES if c[2]])):
        if TEXT.count(e) != 1:
            errs.append("期望句「%s」在生成器源码里命中 %d 次（须 1）" % (e, TEXT.count(e)))
        if not any(e in m for m in msgs):
            errs.append("期望句「%s」不是生成器里任何一条 assert／raise 消息的子串" % e)
        if "%d" in e or "%s" in e:
            errs.append("期望句里带格式符（抛出那一行早已代入数字）：" + e)
    for name, subs, expect, flags in CASES + GREENS:
        for old, _new in subs:
            n = TEXT.count(old)
            if n != 1:
                errs.append("锚点命中 %d 处（须 1）：%s｜%s" % (n, old[:64], name))
        # 每条扰动先在**本进程**里 compile 一次：上一面的 C3 就是只加左半边括号、副本连语法都过不了，
        # 于是红在 SyntaxError 上而不是红在指定闸门上（＝红错了地方）。compile 是免费的，一次 API 不花。
        try:
            cp = copy_text(subs, name)
        except AssertionError as e:
            errs.append("扰动连构造都失败：%s｜%s" % (name, str(e)[:80]))
            continue
        try:
            compile(cp, name, "exec")
        except SyntaxError as e:
            errs.append("扰动后的副本语法不过（会红在 SyntaxError 而不是指定闸门上）：%s｜%s"
                        % (name, e.msg))
        if expect:
            gl = gate_lines(cp, expect)
            if len(gl) != 1:
                errs.append("扰动后的副本里该句命中 %d 行（须 1，±3 窗口才有意义）：%s" % (len(gl), name))
    # 前提普查
    if LESSON_NEEDLE != "自审的方向性":
        errs.append("前提不成立：从生成器现取的 LESSON 锚句是「%s」，与本面登记的形状不同" % LESSON_NEEDLE)
    if bullet_hits(sec, LESSON_NEEDLE) != 1:
        errs.append("前提不成立：基准 needle 在本节命中 %d 条（须 1），C5／C6 的对照面失去意义"
                    % bullet_hits(sec, LESSON_NEEDLE))
    if bullet_hits(sec, LOOSE_NEEDLE) < 2:
        errs.append("前提不成立：needle「%s」在本节命中 <2 条，C6 造不出歧义" % LOOSE_NEEDLE)
    if bullet_hits(sec, PREV_NEEDLE) != 0:
        errs.append("前提不成立：上一片那条 needle 在本节仍命中，C5 不是查无此句")
    if len(re.findall(r"^\*\*\((\w)\)", sec, re.M)) != 0:
        errs.append("前提不成立：本轮日志正文里仍有行首 `**(x)**`，C3 造不出取空")
    if sec.count("- **%s**" % SPECLET) != 1:
        errs.append("前提不成立：C4 要复制的那条顶级子条标签在本节不是恰好 1 处")
    if dev.count("f-string") < 2:
        errs.append("前提不成立：全日志「f-string」出现 %d 次，C8 造不出出处不唯一" % dev.count("f-string"))
    gens = [os.path.relpath(x, REPO) for x in glob.glob(os.path.join(REPO, "docs/evidence/*/am_s*_hdr.py"))]
    with_ph = [p for p in gens if "PLACEHOLDER" in open(os.path.join(REPO, p), encoding="utf-8").read()]
    if len(with_ph) < 2:
        errs.append("前提不成立：含 PLACEHOLDER 的生成器只有 %d 份，C9 造不出多出处" % len(with_ph))
    spec = [p for p in gens if re.search(r"本片的实物证据＝\*\*改之前\*\*那句",
                                         open(os.path.join(REPO, p), encoding="utf-8").read())]
    if len(spec) != 1:
        errs.append("前提不成立：标本句所在生成器不是恰好 1 份（%d），C10／C11 的基准会变形" % len(spec))
    rows = [r.split("\t") for r in subprocess.run(
        ["git", "-c", "core.quotePath=false", "show", "--name-status", "--format=", "HEAD"],
        cwd=REPO, capture_output=True, text=True).stdout.splitlines() if r.strip()]
    na, nm = len([r for r in rows if r[0] == "A"]), len([r for r in rows if r[0] == "M"])
    if na == nm:
        errs.append("前提不成立：HEAD 的 A／M 件数相等（%d＝%d），B4 的互换造不出红" % (na, nm))
    # ---- GPU 面（P1–P6）的前提：本轮事实＝仪器坏着＋我的进程按整词口径本可数到＋合成对照按构造成立 ----
    # 上一片这里查的是「nvidia-smi 现取的 compute 进程里必须有别人的」——本轮那把尺子起不来，
    # 拿它当前提就会把 P1／P2 的空转当成环境故障（这正是本片 §26.33 ⑤ 登记的那类"前提随轮而变"）。
    nvrun = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                           capture_output=True, text=True)
    if nvrun.returncode == 0:
        errs.append("前提不成立：nvidia-smi 本轮又能用了（rc=0）⇒ P1／P2 要造的「普查仪器坏了」不是本轮事实，"
                    "生成器那句注文与本面的前提得一起改写")
    elif "NVML" not in (nvrun.stdout + nvrun.stderr):
        errs.append("前提不成立：nvidia-smi 的失败原因不再是 NVML（首行 %r）⇒ P2 那条 needle 指歪了本轮的真因"
                    % ((nvrun.stdout + nvrun.stderr).strip().splitlines() or [""])[0][:60])
    own0 = open("/proc/%s/cmdline" % os.getpid(), "rb").read().split(b"\0")[0].decode("utf-8", "replace")
    if own0 != sys.executable:
        errs.append("前提不成立：本面自己的 argv[0] 不等于 sys.executable（%r≠%r）⇒ P3 去掉自指排除也不会数到我，"
                    "那条 0 没有正对照" % (own0, sys.executable))
    _syn = b"/bin/bash\0-c\0grep " + sys.executable.encode() + b"\0"
    _syn0 = _syn.split(b"\0")[0].decode("utf-8", "replace")
    _synflat = _syn.decode("utf-8", "replace").replace("\x00", " ")
    if not (_syn0 != sys.executable and sys.executable in _synflat):
        errs.append("前提不成立：那条合成命令行的「松散命中／整词不命中」不成立（%r／%r）⇒ P4／P5 的坏输入与基准同形"
                    % (_syn0, _synflat[:60]))
    _qq = len([l for l in subprocess.run(
        ["git", "-c", "core.quotePath=false", "status", "--porcelain", "-uall"],
        cwd=REPO, capture_output=True, text=True).stdout.splitlines() if l.startswith("??")])
    if _qq < 1:
        errs.append("前提不成立：porcelain 现取 0 行未跟踪件 ⇒ P6 换掉 needle 也数不出差，那是环境不是闸门")
    # ---- ⑤ 读数块（Q1–Q13）的前提：三张控件面各恰好 1 份、标本面是红面且编号齐全、全过面是绿面 ----
    faces = sorted(os.path.relpath(x, REPO) for x in glob.glob(os.path.join(
        REPO, "docs/evidence/*/_raw_s%d_msg_controls*.out" % SLICE)))
    if len(faces) < 2:
        errs.append("前提不成立：%d 号片的控件面全树只命中 %d 份 ⇒ Q1 丢掉 tag 段也是 1 份，那是空转" % (SLICE, len(faces)))
    for tag in ("attempt1", "attempt2"):
        n = len([p for p in faces if ".%s-" % tag in os.path.basename(p)])
        if n != 1:
            errs.append("前提不成立：%d 号片的 %s 标本面命中 %d 份（须 1）⇒ 生成器 ⑤ 那段没有出处，Q 系列判不了" % (SLICE, tag, n))
    bare = [p for p in faces if not re.search(r"\.attempt\d+-", os.path.basename(p))]
    if len(bare) == 1:
        t3 = open(os.path.join(REPO, bare[0]), encoding="utf-8").read()
        if re.findall(r"^× ", t3, re.M):
            errs.append("前提不成立：全过面（无 attempt 段）里仍有顶格「× 」行 ⇒ 它不是全过面，Q6 的放宽造不出与基准之差")
    sp2 = [p for p in faces if ".attempt2-" in os.path.basename(p)]
    if len(sp2) == 1:
        t2 = open(os.path.join(REPO, sp2[0]), encoding="utf-8").read()
        xr = re.findall(r"^× ", t2, re.M)
        ids = sorted(set(re.findall(r"^× ([CG]\d+) ", t2, re.M)))
        if len(xr) < 2 or len(ids) != len(xr):
            errs.append("前提不成立：标本面 × 行 %d 条／带编号 %d 条 ⇒ Q3／Q4 的坏输入没有可丢的对象" % (len(xr), len(ids)))
        if not any(re.fullmatch(r"[CG]\d", i) for i in ids):
            errs.append("前提不成立：标本面的编号全是两位数 ⇒ Q4 把取列口径收成 `\\d\\d+` 也不丢东西，那是空转")
        for ph in ("⑴ **扰动没有真的移动匹配集**", "（表头写盘后仍要"):
            if TEXT.count(ph) < 1:
                errs.append("前提不成立：⑤ 那段的切片锚句「%s」在生成器源码里根本没有 ⇒ 基准段无从切出" % ph)
        tm = gen_tmpl()
        if tm is None:
            errs.append("前提不成立：取不到生成器里 TMPL 那条赋值（ast）⇒ 编号普查没法与生成器同口径")
        elif all(tm.count(ph) == 1 for ph in ("⑴ **扰动没有真的移动匹配集**", "（表头写盘后仍要")):
            # 与生成器**同一口径**：它切的是 TMPL 这个字符串，不是整份源码（同一短语在源码里还会出现一次
            # ——就是它自己那两处 `TMPL.index(...)` 调用 ⇒ 拿源码计次会把基准误判成歧义）
            seg = re.sub(r"@@[A-Z0-9_]+@@", "", tm[tm.index("⑴ **扰动没有真的移动匹配集**"):
                                                   tm.index("（表头写盘后仍要")])
            named = sorted(set(re.findall(r"[CG]\d+", seg)))
            if named != ids:
                errs.append("前提不成立：生成器 ⑤ 点名的编号 %s ≠ 标本面 × 行编号 %s ⇒ 基准自己已烂，Q4／Q7 的红无从判读" % (
                    named, ids))
        else:
            errs.append("前提不成立：⑤ 那段的两个切片锚句在 TMPL 里不是各恰好 1 处（%s／%s）⇒ Q7 的编号普查没有唯一基准段" % (
                tm.count("⑴ **扰动没有真的移动匹配集**"), tm.count("（表头写盘后仍要")))
    # ---- 形制普查与「上一片」出处（R1–R9）的前提 ----
    if len(gens) < 2:
        errs.append("前提不成立：全树 hdr 生成器只有 %d 份 ⇒ R2 放宽成任意片号也是同一批，那是空转" % len(gens))
    alllog = [os.path.relpath(x, REPO) for x in glob.glob(os.path.join(REPO, "docs/evidence/*/am_s7_verify*.log"))]
    if len(alllog) < 2:
        errs.append("前提不成立：全树回执只有 %d 份 ⇒ R1 放宽成 `.log` 结尾也是 1 份，那是空转" % len(alllog))
    fmc = form_census(dev)
    if len(fmc) < 2:
        errs.append("前提不成立：按节普查只得到 %d 节 ⇒ R5 丢掉最后一段等于丢掉全部，顺序断言失去意义" % len(fmc))
    _newsec = sorted(int(n) for n, (new, _old) in fmc.items() if new > 0)
    _oldsec = sorted(int(n) for n, (_new, old) in fmc.items() if old > 0)
    if len(_newsec) < 2 or max(_newsec) == max(_oldsec):
        errs.append("前提不成立：新形制所在节 %s 与旧形制所在节 %s ⇒ R6 换计数键也不会改变那个 max，那是空转" % (
            _newsec, _oldsec))
    if os.path.exists(DECOY):
        errs.append("前提不成立：%s 已存在，A4 的诱饵不是新放的那一枚" % DECOY)
    # ---- F1 的备用解释器前提：换到的那条路径必须"确实不是本面这条"且"本机没有别人在用它当 argv[0]" ----
    # 第一次真跑用的 /usr/bin/python3 正是被后面这一条放过的：本机两个长驻进程的 argv[0] 就是它，
    # 副本于是先在生成器「argv[0] 整词匹配数到我的解释器进程」红掉，抢走了 F1 自己的靶。
    if not os.path.exists(ALT_INTERP):
        errs.append("前提不成立：备用解释器 %s 不存在，F1 没法跑" % ALT_INTERP)
    if ALT_INTERP.startswith("/tmp/amvenv"):
        errs.append("前提不成立：备用解释器 %s 还挂在 /tmp/amvenv 下 ⇒ 生成器那条「CMD 行描述失真」"
                    "按构造红不了，F1 是空转" % ALT_INTERP)
    if ALT_INTERP == sys.executable:
        errs.append("前提不成立：备用解释器与本面解释器同一条（%s）⇒ F1 根本没换解释器" % ALT_INTERP)
    if ALT_RC != 0:
        errs.append("前提不成立：备用解释器 %s 连 stdlib 探针都跑不起来（rc=%s｜%s）⇒ F1 的副本会在"
                    "任何闸门之前就崩，红不在闸门上" % (ALT_INTERP, ALT_RC, ALT_ERRTAIL[:90]))
    elif ALT_OUT != ALT_INTERP:
        errs.append("前提不成立：用 %s 起跑得到的是 sys.executable＝%r（不是它本身）⇒ 生成器读到的解释器"
                    "路径与我以为的不是同一条，F1 的判读没有出处" % (ALT_INTERP, ALT_OUT))
    if ALT_FOREIGN:
        errs.append("前提不成立：本机有 %d 个进程的 argv[0] 就是备用解释器 %s（PID %s）⇒ F1 的副本会先在"
                    "「整词匹配数到我的解释器进程」那条抢红，本面没资格起" % (
                        len(ALT_FOREIGN), ALT_INTERP, "、".join(ALT_FOREIGN[:4])))
    return errs


def judge_run(cp, name, expect, flags, idx, bak, attempt):
    """造一张副本、跑它、判它，返回 (rc, err, ok, shown, why, cls)。

    `cls` 由 classify() 给（命中／环境／空转／超时）；"环境"那条把重跑与否交给调用方，所以这里
    只如实分类，不藏重试。副本一律用完即删，删除前若发现它改动过档案面就整面退出（先还原）。
    """
    path = build_copy(cp, "c%d" % idx if not attempt else "c%d-r%d" % (idx, attempt))
    Made.append(path)
    decoy_on = bool(flags.get("decoy"))
    if decoy_on:
        with open(DECOY, "w", encoding="utf-8") as fh:
            fh.write("诱饵：这一枚不是那次运行的原始 stdout\n")
    snap_before = snapshot()
    rc, msg = run_copy(path, flags)
    snap_after = snapshot()
    if decoy_on and os.path.exists(DECOY):
        os.remove(DECOY)
    moved = [k for k in diff_snap(snap_before, snap_after) if k != path]
    if moved:
        for k in moved:
            if k in bak:
                with open(k, "wb") as fh:
                    fh.write(bak[k])
        print("CONTROL_VERDICT = ENV FAILURE（子运行改动了档案面：%s ⇒ 已从内存备份还原并拒绝继续）"
              % "、".join(os.path.basename(k) for k in moved))
        sys.exit(4)
    os.remove(path)
    Made.remove(path)
    if rc == "TIMEOUT":
        return rc, "", False, "（超时，单独计数）", "环境故障", "超时"
    err = raised(msg)
    tl = trace_line(msg)
    if flags.get("green"):
        ok = rc == 0 and GREEN_MARK in msg
        shown, why = "（须保持绿）", ""
    else:
        gl = gate_lines(cp, expect)
        if flags.get("notb"):
            ok = notb_hit(msg, rc, expect, gl)
            why = "无 traceback 的那类退出⇒改判末条非空行；该句在副本第 %s 行" % gl
        else:
            ok = rc != 0 and expect in err and len(gl) == 1 and tl is not None \
                and abs(gl[0] - tl) <= 3
            why = "句在副本 %s 行／raise 在 %s 行" % (gl, tl)
        shown = expect
    return rc, err, ok, shown, why, classify(ok, rc, err, bool(flags.get("green")))


def main():
    # 逐行刷盘：本面一条子运行＝一次联网尺子，跑到第 30 条时若后面的自查件崩掉，块缓冲里的
    # 「每张扰动各自命中哪条闸门」就会整批消失（上一片那一版实测到：整轮跑完、原始件只剩三行）。
    # 判据是"每条都能红"，那每条的读数就必须每条都落得下盘。
    sys.stdout.reconfigure(line_buffering=True)
    assert sys.executable.startswith("/tmp/amvenv"), "本件解释器不是 /tmp/amvenv 而是 %s" % sys.executable
    assert "LC_ALL" not in os.environ, "本件环境里有 LC_ALL ⇒ 与生成器口径不符，那不是闸门红"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "本件没带空串 CVD"
    bad = precheck()
    if bad:
        print("纸面预检不合格（%d 条），一次 API 都不花：" % len(bad))
        for b in bad:
            print("  · " + b)
        sys.exit(3)
    print("纸面预检通过＝期望句唯一且都是某条 assert／raise 消息的子串＋锚点各命中 1 处＋扰动后副本里"
          "该句仍恰好一行＋前提普查全成立（红路 %d 条＋绿路 %d 条，其中 1 条红路走 SystemExit 例外判据）"
          % (len(CASES), len(GREENS)))
    print("F1 的备用解释器＝%s（本面解释器＝%s）｜它自己的 stdlib 探针 rc＝%s 且 sys.executable 现取＝%s｜"
          "本机以它为 argv[0] 的进程＝%d 个（须 0，否则副本会被「整词匹配数到我的解释器进程」抢红）｜"
          "被弃用的 %s 现数同口径＝%d 个（第一次真跑的两条空转之一就是栽在那上面，这里只作对照印出，不进判据）"
          % (ALT_INTERP, sys.executable, ALT_RC, ALT_OUT, len(ALT_FOREIGN),
             REJECTED_INTERP, len(REJ_FOREIGN)))

    rows, allok, envfail = [], True, 0
    base_snap = snapshot()
    bak = backup_writable()
    try:
        rem0, _, lim0 = quota()
        print("配额闸门起点＝/rate_limit 现取余量（易失读数，不作下一轮分母）＝%d 次｜单窗口上限＝%d 次｜本批子运行＝%d 条（%s）"
          % (rem0, lim0, len(CASES) + len(GREENS),
             "整批装得进一个窗口" if len(CASES) + len(GREENS) <= lim0 else "整批装不进一个窗口 ⇒ 必然跨窗口，等待次数见收面"))
        path = build_copy(copy_text(GREENS[0][1], "baseline"), "base")
        Made.append(path)
        rc, msg = run_copy(path, GREENS[0][3])
        assert rc == 0 and GREEN_MARK in msg, (
            "未扰动基线就没走到预检分支（rc=%s）⇒ 本件的正对照全部无效。尾巴：\n%s"
            % (rc, "\n".join(msg.splitlines()[-8:])))
        os.remove(path)
        Made.remove(path)
        print("基线（未扰动副本，同目录同解释器、不带 --write）rc=%s 且打印出预检标记＝True（所有红的对照面）"
              % rc)

        for idx, (name, subs, expect, flags) in enumerate(CASES + GREENS[1:]):
            cp = copy_text(subs, name)
            attempt = 0
            while True:
                ensure_quota(len(CASES) + len(GREENS) - idx, len(CASES) + len(GREENS))
                rc, err, ok, shown, why, cls = judge_run(cp, name, expect, flags, idx, bak, attempt)
                if cls == "环境" and attempt < ENV_RETRY:
                    attempt += 1
                    Retries.append(name)
                    print("  ！%s 的副本死在生成器那条复现闸门（尺子当次没拿到与归档一致的回文＝环境故障，"
                          "不是这条闸门的靶）⇒ 有界重跑第 %d 次（重跑次数进收面账）" % (name, attempt))
                    continue
                break
            if cls in ("超时", "环境"):
                envfail += 1
                EnvKinds.append((name, cls))
                if envfail >= ENV_FAIL_STOP:
                    print("CONTROL_VERDICT = ENV FAILURE（累计 %d 条子运行没跑成，本件没资格判闸门）" % envfail)
                    sys.exit(4)
                rows.append((name, rc, False, shown, "环境故障"))
                # 前缀用「！」而不是「×」：`× ` 在本族的档案口径里专指"声明的红路没命中"，
                # 没跑成的那条没资格占这个记号（它既没红也没绿，判据本身没跑）。
                print("！%s｜rc=%s｜期望「%s」**未判**｜环境故障（%s）" % (name, rc, shown, cls))
                continue
            allok = allok and ok
            if not ok:
                why += "｜实得抛出行：" + err[:170]
            rows.append((name, rc, ok, shown, why))
            print(("  " if ok else "× ") + name + "｜rc=" + str(rc) + "｜期望「" + shown + "」"
                   + ("命中" if ok else "**未命中**") + ("｜" + why if why else ""))
    finally:
        for p in Made:
            if os.path.exists(p):
                os.remove(p)
        if os.path.exists(DECOY) and "诱饵" in open(DECOY, encoding="utf-8").read()[:30]:
            os.remove(DECOY)

    # 残留与档案完整性普查放在**清理之后**（跑的时候这些文件本来就在）
    leak = [f for f in os.listdir(HERE) if f.startswith(PREFIX)]
    assert not leak, "跑完后 HERE 里仍有扰动副本残留：" + "、".join(leak)
    moved_final = diff_snap(base_snap, snapshot())
    assert not moved_final, "本面跑完之后档案面与起点快照不一致：" + \
        "、".join(os.path.basename(k) for k in moved_final)
    assert os.path.exists(DECOY) is False, "收面时 /tmp 还留着诱饵件：" + DECOY
    try:
        apply_subs(TEXT, (("没有这段文字的锚点", "换了也没用"),), "n1")
        n1 = False
    except AssertionError as e:
        n1 = "命中 0 处" in str(e)
    assert n1, "锚点命中数普查没拦住一条注定落空的锚 ⇒ 扰动表的自查是空转"
    try:
        assert diff_snap({"x": "a"}, {"x": "b"}) == ["x"], "现算的 diff_snap 抓不出不一致"
        assert diff_snap({"x": "a"}, {"x": "a"}) == [], "现算的 diff_snap 把相同也说成不一致"
        n2 = True
    except AssertionError as e:
        n2 = False
        n2why = str(e)
    assert n2, ("md5 快照的两个方向各验一次（不一致要抓得出、一致要不报警）都没通过 ⇒ 口径 ③ 那条"
                "写保护是空转%s" % ("：" + n2why if not n2 else ""))
    # N3＝`notb_hit` 谓词自身的两个方向，纯内存、零配额。为什么必须单独验它：E6 这一条子运行
    #     花一次联网配额，而上一版的判据把「收」那一侧写成了恒假（要求 `SystemExit:` 前缀，
    #     CPython 对未捕获的 SystemExit(str) 根本不加前缀），于是闸门响了、对照却记成未命中。
    #     判据的正对照要用**实测到过的那一行**，不是我以为它会打出来的那一行。
    _E6 = "BODY_OFFSET 未收敛 ⇒ 拒绝归档"
    NOTB_TABLE = [
        ("真实现场形状：裸消息、无 traceback（须收）", _E6 + "\n", 1, [440], True),
        ("带 SystemExit: 前缀的形状也在（前缀不是判据要读的东西，须收）",
         "SystemExit: " + _E6 + "\n", 1, [440], True),
        ("assert 类失败：带 traceback＋AssertionError（须拒）",
         'Traceback (most recent call last):\n  File "zctl_s31_hdr_x.py", line 440, in <module>\n'
         "AssertionError: " + _E6 + "\n", 1, [440], False),
        ("闸门句不在末条非空行（更早的别的错误先崩）（须拒）", _E6 + "\n后面还有一条别的错误\n",
         1, [440], False),
        ("rc=0：根本没红，谈不上命中（须拒）", _E6 + "\n", 0, [440], False),
        ("闸门句在副本里 0 行：锚点落空（须拒）", _E6 + "\n", 1, [], False),
        ("闸门句在副本里 2 行：±3 行窗口没有唯一靶（须拒）", _E6 + "\n", 1, [440, 470], False),
    ]
    _wrong = [t[0] for t in NOTB_TABLE if notb_hit(t[1], t[2], _E6, t[3]) != t[4]]
    assert not _wrong, "notb 谓词在 %d／%d 个形状上判错：%s" % (
        len(NOTB_TABLE) - len(_wrong), len(NOTB_TABLE), "、".join(_wrong))
    assert any(t[4] for t in NOTB_TABLE) and any(not t[4] for t in NOTB_TABLE), \
        "notb 判定表 7 个形状全要求同一侧 ⇒ 这张表两头都验不到"
    # N4＝配额闸门那条"本轮需要多少余量"的分段，纯内存、零配额。第一次真跑就栽在它上面：
    #     上一版写成 `min(need, limit)`，跑到第 4 条时剩余条数正好等于上限 60 ⇒ 要 60 次余量，
    #     等完一个窗口之后又因为 `curl` 空回文（quota() 没有重试）整面崩掉。判据不可满足＋读数不重试，
    #     两条都是"判红的方式自己没被造一次红"的形状 ⇒ 这里两个方向各验一次。
    BATCH = len(CASES) + len(GREENS)
    _LIM = 60
    WANT_TABLE = [
        # (名字, need, limit, batch, 期望 want)
        ("整批装得进窗口：起面时要覆盖整批（须 12）", 12, _LIM, 12, 12),
        ("整批装得进窗口：跑到一半只需覆盖剩余（须 3）", 3, _LIM, 12, 3),
        ("整批装不进窗口：起面那一问不许要整批，只要 MARGIN（须 2）", BATCH, _LIM, BATCH, MARGIN),
        ("整批装不进窗口：跑到中途也一样（须 2）", 4, _LIM, BATCH, MARGIN),
        ("整批装不进窗口：收尾那条不足 MARGIN 时按 need 走，不许反过来卡死（须 1）",
         1, _LIM, BATCH, 1),
        ("上限读数变小也照 batch 分段，不写死 60（须 2）", 4, 3, 4, MARGIN),
    ]
    _wbad = [t[0] for t in WANT_TABLE if want_runs(t[1], t[2], t[3]) != t[4]]
    assert not _wbad, "配额分段在 %d／%d 个形状上判错：%s" % (
        len(WANT_TABLE) - len(_wbad), len(WANT_TABLE), "、".join(_wbad))
    assert any(t[4] > 1 for t in WANT_TABLE) and any(t[4] == 1 for t in WANT_TABLE), \
        "配额分段表全要求同一侧 ⇒ 两头都验不到（要么保护丢了，要么又写成不可满足）"
    assert all(want_runs(n, l, b) <= l for n, l, b, _ in [(t[1], t[2], t[3], t[4]) for t in WANT_TABLE]), \
        "分段之后仍然问出超过单窗口上限的余量 ⇒ 那条判据还是按构造不可满足"
    # N5＝`classify()` 的去向分类，纯内存、零配额。第二次真跑就是栽在分类上：R8 的副本死在生成器
    #     那条复现闸门（尺子当次拿不到与归档一致的回文），却被记成 `×`＝"这条闸门空转"。
    #     两个方向都必须验：A1 的期望句「仪器不可复现」正是那条闸门自己，把它的**命中**判成环境
    #     就等于把判据的空转推给天灾；而 rc=0 却输出里带这句字面的形状也不许算环境。
    CLS_TABLE = [
        ("A1 的形状：抛出的就是自己的复现闸门句（ok=True）⇒ 命中，不许判环境",
         True, 1, "AssertionError: " + REPRO_MSG + " ⇒ 仪器不可复现，本件的判读行不能代表那次运行",
         False, "命中"),
        ("别的控件死在复现闸门（ok=False、rc=1、含该句）⇒ 环境，它有资格被有界重跑一次",
         False, 1, 'Traceback (most recent call last):\nAssertionError: ' + REPRO_MSG, False, "环境"),
        ("别的控件死在另一条闸门（F1 第一次实测到的「整词匹配数到我的解释器进程」）⇒ 空转",
         False, 1, "AssertionError: argv[0] 整词匹配数到我的解释器进程", False, "空转"),
        ("副本跑成功了却没红（rc=0，哪怕输出里带复现句的字面）⇒ 空转，不许算环境",
         False, 0, "生成时刻: x\n" + REPRO_MSG + "\n", False, "空转"),
        ("绿路没保持绿（green=True、ok=False、err 含复现句）⇒ 空转：绿路没有「重跑」这条后门",
         False, 1, "AssertionError: " + REPRO_MSG, True, "空转"),
    ]
    _cbad = [t[0] for t in CLS_TABLE if classify(t[1], t[2], t[3], t[4]) != t[5]]
    assert not _cbad, "去向分类在 %d／%d 个形状上判错：%s" % (
        len(CLS_TABLE) - len(_cbad), len(CLS_TABLE), "、".join(_cbad))
    assert {t[5] for t in CLS_TABLE} == {"命中", "环境", "空转"}, \
        "分类表没同时覆盖三个去向 ⇒ 分类器至少有一侧从未被验过"
    SELFCHK = ["N1 锚点落空普查", "N2 md5 快照双向", "N3 notb 谓词七形状双向",
               "N4 配额分段六形状双向", "N5 去向分类五形状双向"]

    rem1, _, lim1 = quota()
    print("配额闸门收面＝现取余量（易失，不作分母）＝%d 次｜单窗口上限（现取）＝%d 次｜等待触发 %d 次｜等待累计 %d 秒"
      % (rem1, lim1, len(Waits), sum(Waits)))
    print("环境故障重跑＝%d 次（有界上限 %d 次／条）：%s"
          % (len(Retries), ENV_RETRY, "、".join(Retries) if Retries else "无"))
    print("红路声明＝%d 条｜保持绿＝%d 条（判据 rc=0 且打印出预检标记那行）｜基线＝1 条（不进表，"
          "红在开头即整面作废）｜环境故障（不判红也不判过）＝%d 条（按去向细分：%s）｜本面自查＝%d 条（%s；"
          "每条都两个方向各验一次：错的那侧要抓得出、对的那侧要说不报警）"
          % (len(CASES), len(GREENS), envfail,
             "、".join("%s（%s）" % (n, k) for n, k in EnvKinds) if EnvKinds else "无",
             len(SELFCHK), "＋".join(SELFCHK)))
    print("每条子运行＝一次联网尺子复跑（开发日志最后一节里登记的那笔配额账，节号由 sect_body() 现取）；红路判据＝rc≠0 且**抛出的那一行**"
          "含指定闸门句 且 raise 行号落在该句在**副本**里的 ±3 行窗口；逐条读数已在上面按跑到的顺序"
          "即时落盘（本面第一版把它们攒到最后统一印，结果被收面自查件崩掉时整批丢掉）")
    print("本轮造不出红路的闸门＝%d 条（逐条附原因，不算被验）：" % len(UNCOVERED))
    for gate, why in UNCOVERED:
        print("  · " + gate + "：" + why)
    # 表长闭合：循环体对 `CASES + GREENS[1:]` 逐条各写一行（超时那条也写、判据栏写「环境故障」），
    # 所以条数必须等于声明数减基线那一条。不等就是漏跑，不许悄悄少报。
    assert len(rows) == len(CASES) + len(GREENS) - 1, "实跑 %d 行 ≠ 声明 %d＋%d−1（基线不进表）" % (
        len(rows), len(CASES), len(GREENS))
    if envfail:
        print("CONTROL_VERDICT = INCOMPLETE（%d 条子运行**没跑成**＝环境故障（按去向：%s），既没判红也没判过"
              "⇒ 本面没有资格宣称「每条都能红」。已判读的 %d 条逐条见上表；没跑成那几条不等于「闸门空转」，"
              "也不等于「闸门有效」，它只说明这一轮的额度或网络不给读数）"
              % (envfail,
                 "、".join("%s（%s）" % (n, k) for n, k in EnvKinds) if EnvKinds else "无",
                 len(rows) - envfail))
        sys.exit(4)
    print("CONTROL_VERDICT =", "ALL CONTROLS CAN GO RED" if allok else "PROBLEM（有闸门空转）")
    sys.exit(0 if allok else 3)


if __name__ == "__main__":
    main()
