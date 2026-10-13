#!/usr/bin/env python3
"""表头生成器 `am_s30_hdr.py` 的**控件面**：每条坏输入都要让它拒绝一次，否则那条闸门是空转。

与上一件控件面（`am_s30_hdr_after_controls.py`，验的是**只读仪器**）的根本区别：本面扰动的对象是
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
④ 联网配额是这条腿的**真实约束**（§26.32 第 ⑤ 条登记：表头面每条扰动都要复跑一次联网尺子）。
   每条子运行＝1 次 GitHub 未认证 API 调用；起面之前与每次子运行之前读 `/rate_limit`（该端点不计数），
   余量不足以跑完剩余条数就等到窗口重置再继续。等待次数印出来，余量本身是易失读数、不作分母。

继承上一片的两条常量：子运行上限 300 s、环境故障累计 3 条即停，超时**既不判红也不判过**、单独计数。
纸面预检先跑（不起子进程、不花一次 API）：每条期望句在生成器源码里恰好命中 1 处且是某条
`assert`／`raise` 消息的子串、每个锚点恰好命中 1 处、扰动后的副本里该句仍恰好一行、期望句里不许有
格式符；外加**前提普查**——本面的坏输入全靠"某个 needle 命中几条"成立，档案形状一变它就不是坏输入
而是环境变了：偏松的 needle 必须命中 ≥2、偏歪的必须命中 0、HEAD 的 A／M 件数必须不等、GPU 上必须有
别人的进程可做分母、我的进程必须为 0、/tmp 里必须没有同名诱饵。
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
GEN = os.path.join(HERE, "am_s30_hdr.py")
SELF = os.path.abspath(__file__)
TEXT = open(GEN, encoding="utf-8").read()
SELFTEXT = open(SELF, encoding="utf-8").read()
DEVLOG = os.path.join(REPO, "docs/开发日志.md")
PREFIX = "zctl_s30_hdr_"
SUB_TIMEOUT = 300
ENV_FAIL_STOP = 3
GREEN_MARK = "（预检模式：未写盘。加 --write。）"
ALT_INTERP = "/usr/bin/python3"
DECOY = "/tmp/_raw_verify30.out"
# ③ 的名单：本面跑在真目录里，凡是子运行**可能**写到的都钉进来（生成器只对 OUT 开过 w）
WRITE_GUARD = [os.path.join(HERE, "am_s7_verify30.log"), os.path.join(HERE, "_raw_verify30.out"),
               os.path.join(HERE, "_raw_verify30.err")]
GUARD = WRITE_GUARD + [os.path.join(HERE, "am_s30_hdr.py"), os.path.join(HERE, "am_s30_hdr_after.py"),
                       os.path.join(HERE, "am_s30_hdr_after_controls.py"), DEVLOG]
ENV = dict(os.environ)
ENV["CUDA_VISIBLE_DEVICES"] = ""
NOOP_EXPECT = "扰动是空转"
SPECLET = "形制约束的落实＝本轮另起 §26.32，不在 §26.31 内追加"

# --------------------------------------------------------------------------- needle 现取
# 前提普查要用的三个 needle 一律从**生成器源码**的调用行里 ast 现取（不在本面重敲一遍，
# 否则生成器改了锚句而本面还按旧锚句查前提＝两条档案各说各话）
def needle_from_gen(func, which):
    """从 `bullet_for(<pat>, <what>)`／`re.search(<pat>, SEC_BODY)` 这类调用行里取第一个字符串实参。"""
    for node in ast.walk(ast.parse(TEXT)):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == func:
            a0 = node.args[0] if node.args else None
            if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                if which is None or which == a0.value:
                    return a0.value
    return None


LESSON_NEEDLE = needle_from_gen("bullet_for", "对照的可满足性")
PREV_NEEDLE = "判据的可通过性"  # 上一片 LESSON 用的那条，本片日志里应查无
LOOSE_NEEDLE = "回执"
assert LESSON_NEEDLE, "本面从生成器源码里取不到 LESSON 锚句 ⇒ 前提普查只能重敲，拒绝"


def sect_body():
    dev = open(DEVLOG, encoding="utf-8").read()
    secs = list(re.finditer(r"^### 26\.(\d+)[^\n]*$", dev, re.M))
    assert secs, "开发日志里取不到 `### 26.NN` 标题 ⇒ 前提普查无从做起"
    tail = dev[secs[-1].end():]
    nxt = tail.find("\n### ")
    return dev, (tail if nxt == -1 else tail[:nxt])


def bullet_labels(sec):
    return [m.group(1) for m in re.finditer(r"^- \*\*(.+?)\*\*", sec, re.M)]


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
    ("A5 把 GPU 匹配口径放宽成「任何进程都算我的」（那条 0 的正对照）",
     (('if "/tmp/amvenv" in open(', 'if "/" in open('),), "不动 GPU」冲突", {}),
    ("B1 尺子全树 glob 的 needle 指歪（数不到自己）",
     (('"docs/evidence/*/am_s7_verify*.py"', '"docs/evidence/*/am_s7_verify9[0-9].py"'),),
     "计数口径失真", {}),
    ("B2 取 A 的口径放宽成 A＋M（状态字母闭合应红）",
     (('added = [r[-1] for r in rows if r[0] == "A"]',
       'added = [r[-1] for r in rows if r[0] in ("A", "M")]'),), "取列口径破了", {}),
    ("B3 三分类里「/tmp 已无同名件」那一类静默丢件",
     (('        md_absent.append(p)', '        md_absent.extend(())'),),
     "三分类没覆盖全部新增件", {}),
    ("B4 入库清单两个出处互证时左右互换",
     (('assert (NA_BODY, NM_BODY) == (len(added), len(mods)), (',
       'assert (NA_BODY, NM_BODY) == (len(mods), len(added)), ('),), "有一侧的清单口径变了", {}),
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
    ("C5 LESSON 锚句指回上一片那条（本片日志里查无此句）",
     (('bullet_for(r"%s", "对照的可满足性这条从上一片带进的口径")' % LESSON_NEEDLE,
       'bullet_for(r"%s", "判据的可通过性这条从上一片带进的口径")' % PREV_NEEDLE),),
     "引用出处不唯一", {}),
    ("C6 LESSON 锚句放宽成「回执」二字（同一句命中多条）",
     (('bullet_for(r"%s", "对照的可满足性这条从上一片带进的口径")' % LESSON_NEEDLE,
       'bullet_for(r"%s", "回执这两个字在多条子条里都出现")' % LOOSE_NEEDLE),),
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
    ("D1 回执链计数丢掉 int／str 归一（分组是字符串，int 永不相等）",
     (('    n = str(n)  #', '    n = int(n)  #'),), "回执链在 HEAD 里只", {}),
    ("D2 再上一片的链去 HEAD 里查（本该查父提交）",
     (('"--format=", "HEAD^")', '"--format=", "HEAD")'),), "节奏不是一片一环", {}),
    ("D3 本片的链拿上一片的片号去查（那句「留给分片」应成假话）",
     (('OWNHIT = len(chain_ids(SLICE, prev_rows))',
       'OWNHIT = len(chain_ids(PREVSLICE, prev_rows))'),), "本片回执链里有", {}),
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
]

# 良性对照（须保持绿）：每条排除一种"把无关形状当成错误"的共因。
# 与只读仪器那面不同，这里**存在**"改表头还能绿"的方向：本件跑预检分支，BODY_OFFSET 由不动点现算，
# 盘上那份回执不参与比对 ⇒ 表头长一行不叫错位。跨出处闭合（写盘面字节数 ↔ 盘上回执）属于只读仪器，
# 已由 am_s30_hdr_after_controls 的 D1 在同一份档案上造过红。
GREENS = [
    ("G1 未扰动副本（同目录、同解释器、不带 --write）", (), None, {"green": True}),
    ("G2 良性：源码末尾追加一行注释（不动任何读数）",
     ((TEXT.rstrip("\n"), TEXT.rstrip("\n") + "\n# 这条注释不动任何读数，只验本件不因无关形状而拒绝"),),
     None, {"green": True}),
    ("G3 良性：表头里加一条缩进续行（不含标签冒号，偏移由不动点重算）",
     (('TMPL = """生成时刻: @@NOW@@',
       'TMPL = """生成时刻: @@NOW@@\\n          （这条是良性续行：缩进、不含标签冒号）'),),
     None, {"green": True}),
]

UNCOVERED = [
    ("写盘分支的四条（正文区逐字节复核失败／回读出的表头残留 PLACEHOLDER／表头含全角 at 号／"
     "BODY_OFFSET ≠ 实际表头字节数／cmp 三联 rc 0／1／1）",
     "本面一次都不传 `--write`：副本与真档案同目录，让它写盘就是把本片刚生成的回执覆盖掉。这几条判据的"
     "**盘上那侧**由同目录的 `am_s30_hdr_after_controls.py` 在同一份档案上逐条造红（那张面 28 条全过，"
     "其中 A1／A2 对 BODY_OFFSET 行、A9 对半角占位符、A10 对全角 at 号、B1 对 cmp 三联、A4 对两把尺子"
     "对不上）；本件这几行**代码路径**本身没被造红 ⇒ 如实登记，不声称验过。"),
    ("「渲染出的表头残留 PLACEHOLDER」只能从值一侧造红，模板一侧造不出",
     "模板里任何一枚槽位没取值都会先撞「模板有未取值的字段」（那条 `missing` 检查在渲染之前），"
     "所以 E2 走的是「一枚已过闸的值里带进新占位符」这一侧。要造模板侧的红就得同时删掉 `missing` 那条"
     "闸门——那不是坏输入而是拆闸门，红了也不代表闸门有效。"),
    ("「节号写死成一个数字」这类坏输入没有闸门读它",
     "C2 造红的是「现取的锚句指歪 ⇒ 取空」那一侧。把 `SECN = _secs[-1]` 换成写死的数字**不会红**："
     "生成器里没有任何一条断言把 SECN 与子条所在节比对，SECN 只被印出来。⇒ 这一侧的保护是"
     "「现取」这个做法本身而不是闸门；下一轮若要把写死也拦住，得先加一条 SECN ↔ bullet 所属节的比对，"
     "再给它配控件。这条是本轮实测到的**保护缺口**，不是漏测。"),
    ("GPU 那条只有「放宽匹配口径」一个方向",
     "让 `nv_mine` 真非空的唯一办法是我的解释器在 GPU 上留进程，那是红线（本面全程 CPU 钉住，"
     "预检里现取我的进程数必须为 0 才肯起子进程）。A5 因此按「把合法集合放宽 ⇒ 闸门必须响」造红，"
     "它证明的是那条 0 有分母、口径确实读到了别人的进程，不是「真有人占 GPU 时它会拒绝」。"),
    ("③ 那条 md5 写保护是本面的自有闸门，不进 CASES",
     "它由两处自查兜着：每条子运行前后的逐张比对，与收面时与起点快照的再一次全量比对（N2 验的就是"
     "`diff_snap` 两个方向都灵：不一致抓得出、一致不报警——本面第一版把 N2 写反了，正着跑必然"
     "走进 except 之外，于是把自己的绿判成红，31 条跑完却收在收面段）。它验的是"
     "「本面没咬档案」，不是生成器的某条断言。"),
]

Waits = []


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


def quota():
    o = subprocess.run(["curl", "-s", "-i", "https://api.github.com/rate_limit"],
                       capture_output=True, text=True).stdout
    m = re.search(r"x-ratelimit-remaining: (\d+)", o, re.I)
    r = re.search(r"x-ratelimit-reset: (\d+)", o, re.I)
    assert m and r, "读不到 GitHub 配额响应头 ⇒ 配额闸门没法工作，本件一条子运行都不该起"
    return int(m.group(1)), int(r.group(1))


def ensure_quota(need):
    rem, reset = quota()
    if rem >= need:
        return
    wait = max(0, reset - int(time.time()) + 5)
    if wait > 3600:
        print("CONTROL_VERDICT = ENV FAILURE（配额余量 %d＜需要 %d，窗口重置还要 %d 秒）"
              % (rem, need, wait))
        sys.exit(4)
    Waits.append(wait)
    print("  ！配额闸门：余量 %d＜剩余条数 %d ⇒ 等待窗口重置 %d 秒后继续（等待既不判红也不判过）"
          % (rem, need, wait))
    time.sleep(wait)
    rem2, _ = quota()
    assert rem2 >= need, "等待之后余量仍不足 ⇒ 环境确实不给跑，本件没资格判闸门"


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
    if LESSON_NEEDLE != "对照的可满足性":
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
    nv = [l.strip() for l in subprocess.run(["nvidia-smi", "--query-compute-apps=pid",
                                             "--format=csv,noheader"], capture_output=True,
                                             text=True).stdout.splitlines() if l.strip()]
    mine = [p for p in nv if "/tmp/amvenv" in open("/proc/%s/cmdline" % p, "rb").read()
            .decode("utf-8", "replace")]
    if len(nv) < 1:
        errs.append("前提不成立：nvidia-smi 现取 0 个 compute 进程 ⇒ A5 的放宽没有分母，那是环境不是闸门")
    if mine:
        errs.append("前提不成立：我的解释器在 GPU 上有 %d 个进程 ⇒ 与红线冲突，本面拒绝起子进程" % len(mine))
    if os.path.exists(DECOY):
        errs.append("前提不成立：%s 已存在，A4 的诱饵不是新放的那一枚" % DECOY)
    if not os.path.exists(ALT_INTERP):
        errs.append("前提不成立：备用解释器 %s 不存在，F1 没法跑" % ALT_INTERP)
    return errs


def main():
    # 逐行刷盘：本面一条子运行＝一次联网尺子，跑到第 30 条时若后面的自查件崩掉，块缓冲里的
    # 「每张扰动各自命中哪条闸门」就会整批消失（第一版实测到：31 条跑完、原始件只剩三行）。
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

    rows, allok, envfail = [], True, 0
    base_snap = snapshot()
    bak = backup_writable()
    made = []
    try:
        rem0, _ = quota()
        print("配额闸门起点＝/rate_limit 现取余量（易失读数，不作下一轮分母）＝%d 次" % rem0)
        path = build_copy(copy_text(GREENS[0][1], "baseline"), "base")
        rc, msg = run_copy(path, GREENS[0][3])
        assert rc == 0 and GREEN_MARK in msg, (
            "未扰动基线就没走到预检分支（rc=%s）⇒ 本件的正对照全部无效。尾巴：\n%s"
            % (rc, "\n".join(msg.splitlines()[-8:])))
        os.remove(path)
        print("基线（未扰动副本，同目录同解释器、不带 --write）rc=%s 且打印出预检标记＝True（所有红的对照面）"
              % rc)

        for idx, (name, subs, expect, flags) in enumerate(CASES + GREENS[1:]):
            ensure_quota(len(CASES) + len(GREENS) - idx)
            cp = copy_text(subs, name)
            path = build_copy(cp, "c%d" % idx)
            made.append(path)
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
            made.remove(path)
            if rc == "TIMEOUT":
                envfail += 1
                print("  ！%s 子运行超过 %ds ⇒ 判环境故障，不判红也不判过" % (name, SUB_TIMEOUT))
                if envfail >= ENV_FAIL_STOP:
                    print("CONTROL_VERDICT = ENV FAILURE（累计 %d 条超时，本件没资格判闸门）" % envfail)
                    sys.exit(4)
                rows.append((name, rc, False, "（超时，单独计数）", "环境故障"))
                # 前缀用「！」而不是「×」：`× ` 在本族的档案口径里专指"声明的红路没命中"，
                # 超时那条没资格占这个记号（它既没红也没绿，判据本身没跑）。
                print("！%s｜rc=TIMEOUT｜期望「（超时，单独计数）」**未判**｜环境故障" % name)
                continue
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
            allok = allok and ok
            if not ok:
                why += "｜实得抛出行：" + err[:170]
            rows.append((name, rc, ok, shown, why))
            print(("  " if ok else "× ") + name + "｜rc=" + str(rc) + "｜期望「" + shown + "」"
                   + ("命中" if ok else "**未命中**") + ("｜" + why if why else ""))
    finally:
        for p in made:
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
         'Traceback (most recent call last):\n  File "zctl_s30_hdr_x.py", line 440, in <module>\n'
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
    SELFCHK = ["N1 锚点落空普查", "N2 md5 快照双向", "N3 notb 谓词七形状双向"]

    rem1, _ = quota()
    print("配额闸门收面＝现取余量（易失，不作分母）＝%d 次｜等待触发 %d 次" % (rem1, len(Waits)))
    print("红路声明＝%d 条｜保持绿＝%d 条（判据 rc=0 且打印出预检标记那行）｜基线＝1 条（不进表，"
          "红在开头即整面作废）｜超时（环境故障，不判红也不判过）＝%d 条｜本面自查＝%d 条（%s；"
          "每条都两个方向各验一次：错的那侧要抓得出、对的那侧要说不报警）"
          % (len(CASES), len(GREENS), envfail, len(SELFCHK), "＋".join(SELFCHK)))
    print("每条子运行＝一次联网尺子复跑（§26.32 第 ⑤ 条登记的那笔账）；红路判据＝rc≠0 且**抛出的那一行**"
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
        print("CONTROL_VERDICT = INCOMPLETE（%d 条子运行超时＝环境故障：既没判红也没判过 ⇒ 本面没有资格"
              "宣称「每条都能红」。已跑到的 %d 条判据逐条见上表）"
              % (envfail, len(rows) - envfail))
        sys.exit(4)
    print("CONTROL_VERDICT =", "ALL CONTROLS CAN GO RED" if allok else "PROBLEM（有闸门空转）")
    sys.exit(0 if allok else 3)


if __name__ == "__main__":
    main()
