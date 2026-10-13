#!/usr/bin/env python3
"""只读仪器 `am_s30_hdr_after.py` 的**控件面**：每条坏输入都必须让它红，每条良性输入都必须让它绿，
否则那条闸门是空转。§26.32「跑前登记的闸门」第 ③ 条只承诺了「写盘后另跑一件只读仪器做 cmp 三联」，
本件是那一条的对照面——没有它，「只读仪器全绿」和「只读仪器什么都看不见」读数相同。

与提交信息／表头生成器那两面控件面的**根本区别**：本面**不扰动仪器的源码**，扰动的是**喂给它的四张面**
（回执／原始 stdout／生成器源码／生成器写盘面原始 stdout）。原因＝被验对象是一个**校验器**，
「把输入换成坏输入它是否拒绝」才是它的正对照；改它自己的源码只能证明"代码变了它会跟着变"，
那不等于证明它拦得住坏档案。⇒ 子运行跑的是**落地那把仪器本身**（原文件、原路径），
红行的行号与仪器源码逐行对得上，本面因此**一条都不联网**（配额 60 次/小时那笔账留给表头生成器的控件面）。

本面的四条自有口径：
① 四张面必须**整组**换：扰动副本一律落进 `HERE` 下的临时子目录，目录里的文件名与真档案**逐字相同**。
   用子目录而不是随机名文件，是因为仪器有一条「四张输入面片号不一致」闸门（片号从各自文件名现取）——
   名字一花就红在那条上，而不是红在我要验的那条上。
② **判绿与判红同权重**：三条良性对照（G1 裸跑默认四张面、G2 只给生成器面加一行注释、G3 只把原始件
   的 mtime 改到过去而字节不变）必须 rc=0 且走到判读行。它们若红了，说明仪器把"任何改动"都当成
   "正文区错位"＝假拒绝，那它的绿也没价值。**没有**"改表头一个字还能绿"这种对照：理由见 CASES 之后那段。
③ 改表头内容之后必须**重新落 stamp**（`restamp`），否则任何一条表头扰动都先红在算术闭合上，
   我要验的那条闸门根本没被执行。重落 stamp 用不动点迭代（改那一行的位数会再改长度），与生成器同制式；
   故意把 stamp 落错的三条（A2 删行／A3 右移一字节／A4 指到正文区下一行）各自关掉 restamp。
④ 期望句里**不许有 `%d`**：判据读的是**抛出的那一行**，那里面的数字早已代入。
   本面第一版把 A5 写成「顶格字段 %d 行 ≠」，纸面预检查的是源码、真跑判的是 stderr ⇒ 两条都过不了，
   改成取该句代入之后的尾巴「有字段漏写或改名」。同理 `off` 一律按**字节**切（标本见仪器 docstring ④）。

继承上一片的两条常量（不因本面离线而放松）：子运行上限 300 s、环境故障累计 3 条即停；
超时**既不判红也不判过**，单独计数并退出非 0。纸面预检先跑：每条期望句在仪器源码里**恰好命中 1 处**
且必须是某条 `assert` 消息的子串、每条扰动的锚点在目标面里**恰好命中 1 处**。
"""
import ast
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
SRC = os.path.join(HERE, "am_s30_hdr_after.py")
SELF = os.path.abspath(__file__)
PREFIX = "zctl_s30_after_"
SUB_TIMEOUT = 300
ENV_FAIL_STOP = 3
VERDICT_LINE = "HDR_AFTER_VERDICT = ALL READ-ONLY CHECKS PASS"
TEXT = open(SRC, encoding="utf-8").read()
SELFTEXT = open(SELF, encoding="utf-8").read()
ENV = dict(os.environ)
ENV["CUDA_VISIBLE_DEVICES"] = ""


def faces_from_instr():
    """四张面的文件名**从仪器源码的 FACE_NAMES 字面量现取**（ast＋literal_eval，不执行它）。

    为什么取 FACE_NAMES 而不是 DEFAULTS：DEFAULTS 那行带 `os.path.join`，literal_eval 只吃字面量
    （本面第一版就是去 literal_eval DEFAULTS，模块级当场 ValueError）。名字取出来在本面自己 join HERE，
    两边同目录 ⇒ 路径与仪器默认值一致，且一致与否由仪器那条「四张面片号一致」闸门兜着。
    """
    tree = ast.parse(TEXT)
    names = None
    has_defaults = False
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "FACE_NAMES" for t in node.targets):
            names = ast.literal_eval(node.value)
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "DEFAULTS" for t in node.targets):
            has_defaults = True
    assert names is not None, "仪器源码里没有 FACE_NAMES 字面量 ⇒ 面的出处只能重敲，拒绝"
    assert has_defaults, "仪器源码里没有 DEFAULTS ⇒ 本面与它的默认值不再同源"
    assert set(names) == {"log", "raw", "gen", "writeface"}, "FACE_NAMES 的键集合变了：" + str(sorted(names))
    return {k: os.path.join(HERE, v) for k, v in names.items()}


FACES = faces_from_instr()


def assert_msgs():
    """仪器源码里所有 assert 与 raise AssertionError 的消息文本（预检：期望句必须是某条闸门消息的子串）。

    `fields_from_gen` 那条用的是 `raise AssertionError(...)` 而不是 `assert`，本面第一版只收 `assert`，
    于是那条真闸门被判成"不是任何一条闸门的消息"＝纸面预检自己有个盲区（它查的是消息集合的完整性，
    集合取窄了就等于没查）。
    """
    out = []
    for node in ast.walk(ast.parse(TEXT)):
        if isinstance(node, ast.Assert):
            msg = node.msg
        elif isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call) \
                and getattr(node.exc.func, "id", "") == "AssertionError":
            msg = node.exc.args[0] if node.exc.args else None
        else:
            continue
        if msg is None:
            continue
        try:
            out.append(ast.literal_eval(msg))
        except Exception:
            out.append(ast.get_source_segment(TEXT, msg) or "")
    return out


def stamp_m(head):
    m = re.search(r"^BODY_OFFSET=(\d+)$", head, re.M)
    assert m, "表头里取不到 BODY_OFFSET 行"
    return m


def restamp(head):
    s = len(head.encode())
    for _ in range(8):
        # 注意：`re.Match` 没有 `.sub`（那是 `re.Pattern` 的方法），这里必须用模块级 re.sub
        new = re.sub(r"^BODY_OFFSET=(\d+)$", "BODY_OFFSET=%d" % s, head, count=1, flags=re.M)
        n2 = len(new.encode())
        if n2 == s:
            return new
        head, s = new, n2
    raise AssertionError("重新落 stamp 不收敛 ⇒ 这条扰动没法只测一条闸门")


def insert_head(head, text):
    m = stamp_m(head)
    return head[:m.start()] + text + head[m.start():]


def split_log(path):
    """按**字节**偏移切表头／正文（cmp -i 的口径就是字节；先解码再按下标切会切进正文区）。"""
    b = open(path, "rb").read()
    m = re.search(rb"^BODY_OFFSET=(\d+)$", b, re.M)
    off = int(m.group(1))
    return b[:off].decode("utf-8"), b[off:].decode("utf-8")


def _drop(text, pat):
    m = re.search(pat, text, re.M)
    assert m, "锚点没命中：" + pat
    return text[:m.start()] + text[m.end():]


PAD = '没有"光秃秃的 0"。'  # 分母口径字段的续行尾巴：缩进、不是顶格 ⇒ 动它不碰字段计数


def mut_log(kind):
    def f(head, body):
        if kind == "dup_stamp":
            return insert_head(head, stamp_m(head).group(0) + "\n"), body, True
        if kind == "drop_stamp":
            return _drop(head, r"^BODY_OFFSET=\d+$\n"), body, False
        if kind == "stamp_plus1":
            m = stamp_m(head)
            return head[:m.start()] + "BODY_OFFSET=%d" % (int(m.group(1)) + 1) + head[m.end():], body, False
        if kind == "stamp_nextline":
            m = stamp_m(head)
            bl = body.splitlines(True)[0]
            return head[:m.start()] + "BODY_OFFSET=%d" % (len(head.encode()) + len(bl.encode())) \
                + head[m.end():], body, False
        if kind == "drop_field_line":
            return re.sub(r"^生成时刻: .*$\n", "", head, count=1, flags=re.M), body, True
        if kind == "swap_two_fields":
            ls = head.splitlines(True)
            i = [n for n, l in enumerate(ls) if l.startswith("TREE    :")][0]
            j = [n for n, l in enumerate(ls) if l.startswith("SRC指纹 :")][0]
            ls[i], ls[j] = ls[j], ls[i]
            return "".join(ls), body, True
        if kind == "indent_glue":
            return insert_head(head, "  复核    : 这条是假的粘连行\n"), body, True
        if kind == "loose_top_line":
            return insert_head(head, "这一行顶格但不是字段\n"), body, True
        if kind == "half_placeholder":
            return head.replace(PAD, PAD + "@@BOGUS@@", 1), body, True
        if kind == "fullwidth_at":
            return head.replace(PAD, PAD + "＠", 1), body, True
        raise AssertionError("未知 log 扰动：" + kind)
    return f


def trunc_last(text):
    return "".join(text.rstrip("\n").splitlines(True)[:-1])


def mut_raw(kind):
    def f(text):
        if kind == "flip_one_byte":
            # 锚点必须选**唯一**的那处：raw 里「MATCH」出现 43 次（④ 段每个文件一行），
            # 纸面预检因此把这条判成"锚点命中 43 处"——命中数不唯一的锚点根本不能说清改了哪一处。
            return text.replace(" -> MATCH", " -> MATch", 1)
        raise AssertionError("未知 raw 扰动：" + kind)
    return f


def mut_gen(kind):
    def f(text):
        if kind == "one_label_more":
            # 声明里**多**一条（档案里没有那个字段）⇒ 只有这条能让"字段数"闸门从声明一侧红。
            # 反向的"声明里少一条"红不到它：那条标签从 lab 交替式里消失之后，档案里那一行不再被认成
            # 字段，于是先红在"顶格却不像字段的行"上（＝A8 那条闸门），本面把这一侧登记进 UNCOVERED。
            return text.replace('"生成时刻", ', '"生成时刻", "额外字段", ', 1)
        if kind == "renamed":
            return re.sub(r"^FIELDS = \[", "FIELDSZ = [", text, count=1, flags=re.M)
        if kind == "not_literal":
            # 推导式必须**配平**：只加"[x for x in ["而不补右括号 ⇒ 副本连 compile 都过不了，
            # 红在 SyntaxError 上而不是红在闸门上（本面第一版就是这样，实测到的是"红错了地方"）。
            text = re.sub(r"^FIELDS = \[", "FIELDS = [x for x in [", text, count=1, flags=re.M)
            return text.replace('"用途", "教训"]', '"用途", "教训"]]', 1)
        if kind == "not_str":
            return text.replace('FIELDS = ["生成时刻"', "FIELDS = [1", 1)
        if kind == "benign_comment":
            # 良性：加一行注释。FIELDS 的 ast 取数不受影响 ⇒ 必须**保持绿**（G2）
            return text + "\n# 这条注释不动任何读数，只验本件不会因为无关形状而拒绝\n"
        raise AssertionError("未知 gen 扰动：" + kind)
    return f


def mut_wf(kind):
    def f(text):
        if kind == "bytes_off_by_one":
            return text.replace(" 18068 bytes", " 18067 bytes", 1)
        if kind == "cmp_left_zero":
            return text.replace("左 +1 须非 0＝ 1", "左 +1 须非 0＝ 0", 1)
        if kind == "fields_15":
            return text.replace("顶格字段行数＝ 16 / 16", "顶格字段行数＝ 15 / 16", 1)
        if kind == "drop_line":
            return _drop(text, r"^cmp 复核 rc.*$")
        raise AssertionError("未知 writeface 扰动：" + kind)
    return f


MUTS = {"log": mut_log, "raw": mut_raw, "gen": mut_gen, "writeface": mut_wf}

# (名字, 目标面, 扰动, 期望闸门句)——下面 20 条**全是红路**；保持绿的三条住在 CASES 之后的 GREENS 段
CASES = [
    ("A1 表头里出现第二条 BODY_OFFSET 行", "log", "dup_stamp", "BODY_OFFSET 行不是恰好 1 条"),
    ("A2 表头里没有 BODY_OFFSET 行", "log", "drop_stamp", "BODY_OFFSET 行不是恰好 1 条"),
    ("A3 stamp 右移一字节（指到行中间）", "log", "stamp_plus1", "BODY_OFFSET 不落在行首"),
    ("A4 stamp 指到正文区第二行的行首（是行首但错位）", "log", "stamp_nextline", "两把尺子对不上"),
    ("A5 删掉一条字段的整行", "log", "drop_field_line", "有字段漏写或改名"),
    ("A6 两条字段整行互换（数对、序错）", "log", "swap_two_fields", "字段顺序与 FIELDS 声明不同序"),
    ("A7 缩进处出现「标签＋冒号」", "log", "indent_glue", "缩进处出现"),
    ("A8 顶格却不像字段的行", "log", "loose_top_line", "有顶格却不像字段的行"),
    ("A9 表头残留半角占位符", "log", "half_placeholder", "盘上表头残留半角占位符"),
    ("A10 表头含全角 at 号", "log", "fullwidth_at", "含全角 at 号"),
    ("A11 正文区与原始件同时截掉末行（判读行没了）", "log+raw", "trunc_last",
     "原始件末行与判读行常量不符"),
    ("B1 原始件里改一个字节", "raw", "flip_one_byte", "cmp 三联不是"),
    ("C1 生成器 FIELDS 多一条声明（档案里没这个字段）", "gen", "one_label_more", "有字段漏写或改名"),
    ("C2 生成器里没有 FIELDS 赋值", "gen", "renamed", "没有 FIELDS 赋值"),
    ("C3 FIELDS 是推导式不是字面量", "gen", "not_literal", "不是列表／元组字面量"),
    ("C4 FIELDS 里有非字符串元素", "gen", "not_str", "非字符串元素"),
    ("D1 写盘面报的总字节数差一", "writeface", "bytes_off_by_one", "写盘面报出"),
    ("D2 写盘面报的 cmp 左 +1 是 0", "writeface", "cmp_left_zero", "写盘面 cmp 三联"),
    ("D3 写盘面报的字段数 15/16", "writeface", "fields_15", "写盘面字段数"),
    ("D4 写盘面里没有 cmp 那一行", "writeface", "drop_line", "在文件里命中"),
]

# 「必须保持绿」的对照在防什么：仪器因为**无关形状**而拒绝（＝假拒绝）。假拒绝的绿没有价值，
# 所以判绿与判红同权重，每条绿都在排除一种共因。
# 为什么没有"改表头一个字还能绿"这种绿：本件有一条把**回执总字节数**钉到生成器写盘面读数的闸门
# （两出处闭合），动表头就必然动字节数 ⇒ 那种"良性"扰动在这一件上按构造不可能绿。第一版写了这样
# 一条"良性"对照，它如实红在「写盘面报出」上——那不是仪器过敏感，是它本来就在做的事（本面因此
# 把良性对照换成三个真正与读数正交的方向，见 GREENS 段）。
GREENS = ("G1 裸跑（默认四张面＝档案本身，不经临时目录）",
          "G2 良性：生成器面多一行注释（不动 FIELDS 的取数）",
          "G3 良性：原始件 mtime 改到过去（字节不变）")

# 每条扰动的锚点（纸面预检用：命中数必须恰好 1）
ANCHORS = {
    ("log", "dup_stamp"): r"^BODY_OFFSET=\d+$",
    ("log", "drop_stamp"): r"^BODY_OFFSET=\d+$",
    ("log", "stamp_plus1"): r"^BODY_OFFSET=\d+$",
    ("log", "stamp_nextline"): r"^BODY_OFFSET=\d+$",
    ("log", "drop_field_line"): r"^生成时刻: .*$",
    ("log", "swap_two_fields"): r"^TREE    :",
    ("log", "indent_glue"): r"^BODY_OFFSET=\d+$",
    ("log", "loose_top_line"): r"^BODY_OFFSET=\d+$",
    ("log", "half_placeholder"): re.escape(PAD),
    ("log", "fullwidth_at"): re.escape(PAD),
    ("log+raw", "trunc_last"): r"^LANDING_VERDICT = ALL CHECKS PASS$",
    ("raw", "flip_one_byte"): " -> MATCH",
    ("gen", "one_label_more"): '"生成时刻", ',
    ("gen", "renamed"): r"^FIELDS = \[",
    # 两条锚点：推导式必须配平，只开不闭会让副本连语法都过不了（红在 SyntaxError 上＝红错地方）
    ("gen", "not_literal"): [r"^FIELDS = \[", '"用途", "教训"]'],
    ("gen", "not_str"): r'FIELDS = \["生成时刻"',
    ("writeface", "bytes_off_by_one"): " 18068 bytes",
    ("writeface", "cmp_left_zero"): r"左 \+1 须非 0＝ 1",
    ("writeface", "fields_15"): "顶格字段行数＝ 16 / 16",
    ("writeface", "drop_line"): r"^cmp 复核 rc.*$",
}
NOOP_EXPECT = "扰动是空转"
OTHER_EXpects = ["本件不是只读件", "张冠李戴", "不符合形制", NOOP_EXPECT]


def assert_changed(new, data, face, kind):
    assert not (new == data), NOOP_EXPECT + "：副本与原件逐字节相同（%s／%s）" % (face, kind)


def build(dest, face=None, kind=None):
    """把四张面整组抄进 dest（二进制原样），只对目标面施加扰动。"""
    for k, srcpath in FACES.items():
        data = open(srcpath, "rb").read()
        new = data
        if face == "log+raw" and k in ("log", "raw"):
            # A11 这一条必须同时截两张：只截一张会先红在 cmp／算术闭合上，验不到末行那两条闸门
            if k == "log":
                head, body = split_log(srcpath)
                new = (head + trunc_last(body)).encode("utf-8")
            else:
                new = trunc_last(data.decode("utf-8")).encode("utf-8")
        elif k == face:
            if k == "log":
                head, body = split_log(srcpath)
                h2, b2, rs = mut_log(kind)(head, body)
                new = ((restamp(h2) if rs else h2) + b2).encode("utf-8")
            else:
                new = MUTS[k](kind)(data.decode("utf-8")).encode("utf-8")
        if new is not data:
            assert_changed(new, data, k, kind)
            data = new
        with open(os.path.join(dest, os.path.basename(srcpath)), "wb") as fh:
            fh.write(data)


def run_instr(dest=None, extra=None):
    cmd = [sys.executable, SRC]
    if dest:
        for k in ("log", "raw", "gen", "writeface"):
            cmd += ["--" + k, os.path.join(dest, os.path.basename(FACES[k]))]
    cmd += extra or []
    p = subprocess.Popen(cmd, cwd=REPO, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, start_new_session=True)
    try:
        out, err = p.communicate(timeout=SUB_TIMEOUT)
        return p.returncode, (err or "") + (out or ""), None
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        out, err = p.communicate()
        return "TIMEOUT", (err or "") + (out or ""), dest


def precheck():
    errs = []
    # 锚点表与扰动表必须**双向**闭合：少一个键＝那条扰动从没被查过（纸面预检查的是锚点命中数），
    # 多一个键＝档案里留着一个 CASES 里没有的形状，读起来像一条对照而实际没人跑。
    need = {(c[1], c[2]) for c in CASES}
    for pair in sorted(need - set(ANCHORS)):
        errs.append("扰动 %s 没有锚点声明 ⇒ 纸面预检查不到它" % (pair,))
    for pair in sorted(set(ANCHORS) - need):
        errs.append("ANCHORS 里有 CASES 之外的死键：%s" % (pair,))
    msgs = assert_msgs()
    for e in set([c[3] for c in CASES if c[3] is not None] + OTHER_EXpects):
        base = SELFTEXT if e == NOOP_EXPECT else TEXT
        who = "本件" if e == NOOP_EXPECT else "仪器"
        n = base.count(e)
        if n != 1:
            errs.append("期望句「%s」在%s源码里命中 %d 次（须 1）" % (e, who, n))
        if e != NOOP_EXPECT and not any(e in m for m in msgs):
            errs.append("期望句「%s」不是仪器里任何一条 assert 的消息子串" % e)
    for name, face, kind, expect in CASES:
        pats = ANCHORS.get((face, kind))
        if pats is None:
            continue  # 上面已经记了一条"没有锚点声明"，这里再 KeyError 会让预检本身崩掉
        if isinstance(pats, str):
            pats = [pats]
        for pat in pats:
            for f in face.split("+"):
                t = open(FACES[f], encoding="utf-8").read()
                n = len(re.findall(pat, t, re.M))
                if n != 1:
                    errs.append("锚点 %s 在 %s 面里命中 %d 处（须 1）：%s" % (pat, f, n, name))
        if expect and ("%d" in expect or "%s" in expect):
            errs.append("期望句里带格式符（抛出那一行已代入数字，永远匹配不上）：" + name)
    return errs


def gate_line(expect):
    src = SELFTEXT if expect == NOOP_EXPECT else TEXT
    return [n + 1 for n, l in enumerate(src.splitlines()) if expect in l]


def trace_line(msg):
    hit = re.findall(r'File "[^"]*/am_s30_hdr_after\.py", line (\d+)', msg)
    return int(hit[-1]) if hit else None


def raised(msg):
    ls = [l for l in msg.splitlines() if l.strip()]
    for l in reversed(ls):
        if re.search(r"(AssertionError|UnicodeDecodeError|SystemExit|Error)\b", l) \
                and not l.strip().startswith("File "):
            return l
    return ls[-1] if ls else ""


def residue():
    return [f for f in os.listdir(HERE) if f.startswith(PREFIX)]


UNCOVERED = [
    ("「输入面 md5 跑前后逐一相同」那条只读兜底",
     "红路需要让**仪器自己**去写盘，而本面的扰动一律落在输入面上（docstring 第一段说了为什么不扰动它的"
     "源码）。同一声称的另一侧已由 E1／E2 造红：ast 普查确实认得出 open 的可写模式与 os.unlink 这类构造，"
     "所以 md5 那条是第二层保险，不是唯一保险。"),
    ("「表头按字节切完再编码回去不等于 off」那条",
     "任何把 off 挪到多字节字符中间的 stamp 扰动都会先撞「BODY_OFFSET 不落在行首」（该字符前一字节不是"
     "换行），而让 decode 直接失败的坏字节序列则先撞 UnicodeDecodeError ⇒ 这条 assert 是前两条都过了"
     "之后的兜底，本面造不出只命中它的输入。A3 就是那条先行的闸门。"),
    ("「正文区末行不是判读行」那条（与「原始件末行」成对）",
     "本件对正文区与原始件的末行用的是同一把尺子：只动正文区必然先红在 cmp／算术闭合上，只动原始件也"
     "必然先红在同两条上。A11 同时截两张，已经把这对闸门红了一次（命中的是原始件那半边）。"),
    ("「四张输入面片号不一致」的更复杂形状",
     "现档只有 29／30 两个片号可用，构造不出「三张同片、另两张各片」的形状；F1 用的是 log／gen／"
     "writeface 同片而 raw 取自上一片那一种，F2 用的是同一闸门的名字侧（不合形制）。"),
    ("「顶格字段行数 ≠ 声明行数」这条闸门的**声明偏少**一侧",
     "本面只从「声明偏多」那一侧造红（C1：声明里多加一条档案里没有的标签 ⇒ 16≠17 红在计数字段上）。"
     "反向（把声明里某个标签删掉）红不到这条：那个标签从 `lab` 交替式里消失之后，档案里那一行不再被"
     "认成字段，于是**先**撞「有顶格却不像字段的行」（＝A8 那条闸门，已被 A8 独立造红一次）。⇒ 计数字段"
     "这一条只有单侧证据，另一侧由 A8 兜住；本面第一版把 C1 写成「声明偏少」，实测到的正是这种"
     "「红错了闸门」的形状。"),
    ("良性扰动三条各只挡一种共因",
     "G1（裸跑默认）排除「副本机制本身跑不起来」，G2（生成器面加注释）排除「ast 取数被无关形状带崩」，"
     "G3（只动 mtime）排除「仪器把时间戳当成内容」。三条都是「内容没变而形状变了」的方向；"
     "**改表头内容还能绿**那种良性按构造不存在，原因写在 CASES 之后那段（回执字节数被钉在生成器写盘面）。"
     "共因的另一半——「整组文件被仪器一概拒绝」——由 20 条红路各自报出自己的闸门句排除，不靠绿路。"),
]


def main():
    assert sys.executable.startswith("/tmp/amvenv"), "解释器不是 /tmp/amvenv 而是 %s" % sys.executable
    assert "LC_ALL" not in os.environ, "本件环境里有 LC_ALL ⇒ 与仪器的口径不符，那不是闸门红"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "本件没带空串 CVD"
    bad = precheck()
    if bad:
        print("纸面预检不合格（%d 条），不起子进程：" % len(bad))
        for b in bad:
            print("  · " + b)
        sys.exit(3)
    print("纸面预检通过＝期望句唯一且都是某条 assert 的消息子串＋锚点各命中 1 处＋期望句里没有格式符"
          "（%d 条扰动）" % len(CASES))

    root = tempfile.mkdtemp(prefix=PREFIX, dir=HERE)
    rows, allok, envfail = [], True, 0
    try:
        dest = os.path.join(root, "base")
        os.mkdir(dest)
        build(dest)
        rc0, msg0, _ = run_instr(dest)
        assert rc0 == 0, ("未扰动基线就没通过（rc=%s）⇒ 本件的正对照全部无效。尾巴：\n%s"
                          % (rc0, "\n".join(msg0.splitlines()[-8:])))
        assert VERDICT_LINE in msg0, "未扰动基线没走到判读行 ⇒ 副本跑的不是同一条路径"

        for name, face, kind, expect in CASES:
            d = os.path.join(root, "c%d" % len(rows))
            os.mkdir(d)
            build(d, face, kind)
            rc, msg, _ = run_instr(d)
            if rc == "TIMEOUT":
                envfail += 1
                print("  ！%s 子运行超过 %ds ⇒ 判环境故障，不判红也不判过" % (name, SUB_TIMEOUT))
                if envfail >= ENV_FAIL_STOP:
                    print("CONTROL_VERDICT = ENV FAILURE（累计 %d 条超时，本件没资格判闸门）" % envfail)
                    sys.exit(4)
                continue
            err, gl, tl = raised(msg), gate_line(expect), trace_line(msg)
            ok = rc != 0 and expect in err and len(gl) == 1 and tl is not None \
                and abs(gl[0] - tl) <= 3
            allok = allok and ok
            why = "句在源码 %s 行／raise 在 %s 行" % (gl, tl)
            if not ok:
                why += "｜实得抛出行：" + err[:170]
            rows.append((name, rc, ok, expect, why))

        for label, target in (("E1 只读普查指向表头生成器（含 open mode=w）", os.path.join(HERE, "am_s30_hdr.py")),
                              ("E2 只读普查指向信息控件面（含 os.unlink）",
                               os.path.join(HERE, "am_s30_msg_controls.py"))):
            d = os.path.join(root, "e%d" % len(rows))
            os.mkdir(d)
            build(d)
            rc, msg, _ = run_instr(d, ["--auditsrc", target])
            expect = "本件不是只读件"
            err, gl, tl = raised(msg), gate_line(expect), trace_line(msg)
            ok = rc != 0 and expect in err and len(gl) == 1 and tl is not None and abs(gl[0] - tl) <= 3
            allok = allok and ok
            why = "句在源码 %s 行／raise 在 %s 行" % (gl, tl)
            if not ok:
                why += "｜实得抛出行：" + err[:170]
            rows.append((label, rc, ok, expect, why))

        # 本片新增的一条闸门：四张面必须同片 ⇒ 把 raw 换成上一片那一张（同目录、名字合法、片号不同）
        d = os.path.join(root, "f1")
        os.mkdir(d)
        build(d)
        prev_raw = os.path.join(REPO, "docs/evidence/2026-10-10/_raw_verify29.out")
        shutil.copyfile(prev_raw, os.path.join(d, os.path.basename(prev_raw)))
        rc, msg, _ = run_instr(d, ["--raw", os.path.join(d, os.path.basename(prev_raw))])
        expect = "张冠李戴"
        err, gl, tl = raised(msg), gate_line(expect), trace_line(msg)
        ok = rc != 0 and expect in err and len(gl) == 1 and tl is not None and abs(gl[0] - tl) <= 3
        allok = allok and ok
        rows.append(("F1 混入上一片的原始件（名字合法但片号不同）", rc, ok, expect,
                     "句在源码 %s 行／raise 在 %s 行" % (gl, tl)))

        # 同一条闸门的另一侧：文件名压根不符合形制 ⇒ 取不到片号
        d = os.path.join(root, "f2")
        os.mkdir(d)
        build(d)
        bad_name = os.path.join(d, "wf_copy.txt")
        os.rename(os.path.join(d, os.path.basename(FACES["writeface"])), bad_name)
        rc, msg, _ = run_instr(d, ["--writeface", bad_name])
        expect = "不符合形制"
        err, gl, tl = raised(msg), gate_line(expect), trace_line(msg)
        ok = rc != 0 and expect in err and len(gl) == 1 and tl is not None and abs(gl[0] - tl) <= 3
        allok = allok and ok
        rows.append(("F2 把一张面改成不合形制的名字", rc, ok, expect,
                     "句在源码 %s 行／raise 在 %s 行" % (gl, tl)))

        # 「必须保持绿」的三条：每条都在排除一种**共因**（红了不代表闸门有效，只代表仪器跑不起来）
        # 标签一律取自模块级 GREENS 声明，不在这里重敲 ⇒ 声明与实际跑的条数／名字若分家，下面那条
        # 逐条对照会当场红，而不是留下一份"名字对不上读数"的档案。
        greens = []
        rc, msg, _ = run_instr(None)          # G1 裸跑默认四张面＝档案本身
        greens.append((GREENS[0], rc, msg))
        d = os.path.join(root, "g2")
        os.mkdir(d)
        build(d, "gen", "benign_comment")     # G2 生成器面加一行注释：与读数无关
        rc, msg, _ = run_instr(d)
        greens.append((GREENS[1], rc, msg))
        d = os.path.join(root, "g3")
        os.mkdir(d)
        build(d)                              # G3 只动原始件的 mtime，内容逐字节不变
        rp = os.path.join(d, os.path.basename(FACES["raw"]))
        os.utime(rp, (1_700_000_000, 1_700_000_000))
        rc, msg, _ = run_instr(d)
        greens.append((GREENS[2], rc, msg))
        assert len(greens) == len(GREENS), "绿路实跑 %d 条 ≠ 声明 %d 条" % (len(greens), len(GREENS))
        for label, rc, msg in greens:
            ok = rc == 0 and VERDICT_LINE in msg
            allok = allok and ok
            rows.append((label, rc, ok, "（须保持绿）",
                         "" if ok else "实得尾巴：" + " ⏎ ".join(msg.splitlines()[-3:])[:200]))

        # 对**本面自己**的闸门：空转扰动必须被拒（否则"改了源码没改行为"会冒充一条红路）
        expect = NOOP_EXPECT
        try:
            assert_changed(b"same", b"same", "raw", "noop")
            rows.append(("N1 空转扰动普查（本件自己的闸门）", 0, False, expect, "副本居然与原件相同却没被拦"))
            allok = False
        except AssertionError as e:
            okk = expect in str(e)
            allok = allok and okk
            rows.append(("N1 空转扰动普查（本件自己的闸门）", 3 if okk else 0, okk, expect,
                         "本件自己的异常，句在本件源码 %s 行" % gate_line(expect)))
    finally:
        shutil.rmtree(root, ignore_errors=True)

    # 残留普查放在**清理之后**：本面的每条扰动都住在 root 这个临时目录里，跑的时候 root 必然存在
    # （第一版把这条 assert 放在 try 内，于是它查到的永远是自己的临时目录＝一条永远红的闸门）。
    leak = residue()
    assert not leak, "跑完后 HERE 里仍有扰动目录残留：" + "、".join(leak)

    print("基线（四张面原样、显式传参）rc=%s 且走到判读行＝True（所有红的对照面）" % rc0)
    print("解释器＝%s｜被验的仪器＝%s｜每条子运行上限＝%ds｜环境故障累计上限＝%d 条" % (
        sys.executable, os.path.relpath(SRC, REPO), SUB_TIMEOUT, ENV_FAIL_STOP))
    print("CASES 声明＝%d 条（全部红路）｜保持绿＝%d 条（声明与实跑逐条对照，见 greens 段的长度 assert）｜"
          "只读普查红路＝2 条｜片号侧红路＝2 条｜空转普查＝1 条｜超时（环境故障，不判红也不判过）＝%d 条"
          % (len(CASES), len(GREENS), envfail))
    print("对照条数＝%d（其中红路以 rc≠0 且**抛出的那一行**含指定闸门句、且 raise 行号落在该句源码位置的 "
          "±3 行窗口为判据；绿路以 rc=0 且含判读行为判据）" % (len(rows) + envfail))
    for name, rc, ok, expect, why in rows:
        print(("  " if ok else "× ") + name + "｜rc=" + str(rc) + "｜期望「" + expect + "」"
               + ("命中" if ok else "**未命中**") + ("｜" + why if why else ""))
    print("本轮造不出红路的闸门＝%d 条（逐条附原因，不算被验）：" % len(UNCOVERED))
    for gate, why in UNCOVERED:
        print("  · " + gate + "：" + why)
    print("CONTROL_VERDICT =", "ALL CONTROLS CAN GO RED" if allok else "PROBLEM（有闸门空转）")
    sys.exit(0 if allok else 3)


if __name__ == "__main__":
    main()
