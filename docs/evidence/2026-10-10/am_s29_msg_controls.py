#!/usr/bin/env python3
"""提交信息生成器 am_s29_msg.py 的**正对照**：每条扰动都要让它拒绝在**自己那条闸门**上，否则那条闸门是空转。

扰动改的是**生成器的源码副本**（正则级改写），跑的是同一件（同一条 `python <file>`）⇒ 被验的就是
落地那把仪器本身，不是另写的平行实现。判据＝rc≠0 **且**抛出的那一行含指定的闸门句 **且** raise 的行号
落在该句在源码里的位置（±3 行）——只看 rc 会把"环境错"记成"闸门对"。

本轮（分片 29）新增／改动的闸门族，逐条都有红路：
① 片号三处互证（回执 glob 的最大号角本、`SLICE not in RCPT`、日志节号＝上一节＋1）；
② 桶名与桶形由**现取片号插值**（C44：把插值写死成上一轮的号 ⇒ 所有件「落进 0 个桶」而红，
   不是静默沿用上一片的故事——这是 §26.27 (u)／§26.30 记过的机械克隆病形的结构性修法）；
③ 取数一律 `one()`／`grab()` 的**恰好一次**（C22 把形状放宽到 naturally 两处命中、C29 把读数面指错）；
④ 键表自查读 `__file__`（C34–C36：重复字面键／非串键／字面外塞键），且顺序在 `missing`／`unused` 之前；
⑤ 局部仪器（归档自查件）的两条：现跑必须红在锚点句（C30），且不许把 stdout 断言改成读另一条流（C31）；
⑥ GPU 的两个假零形状：正对照指到不存在的解释器（C32）、把"我的"读成自己的 cmdline（C33）。

七条**仪器自己的**坑（继承＋本轮新增）：
① 判据不能只看 rc≠0；副本也不能写在 /tmp——生成器用自身位置反推 REPO ⇒ 副本必须落在**同一目录**，
   并加一条**未扰动基线必须 rc=0 且走到打印分支**（C0），否则所有红都不作数。
② **期望闸门句里不许写死节号／片号／读数**：那些是档案的读数，写死就随片腐烂（红是红了、红的却不是
   指定闸门）。本件所有期望句都取自闸门自己的固定短语。
③ 按**抛出的那一行**判红，不按整段 stderr：3.11+ 的 traceback 把源码行原样回显，而闸门句本来就写在
   源码里 ⇒ 只要栈落在附近任何一行，整段输出都含那句＝假命中。
④ **必须用 /tmp/amvenv/bin/python 跑本件**：生成器的解释器口径与 GPU 正对照读 `/proc/self/cmdline`，
   换解释器 ⇒ C0 就没通过（那正是闸门该做的事，但不是本件的判据）。C0 失败时本件把 stderr 尾巴印出来。
⑤ **改了源码 ≠ 改了行为**：本轮第一条 C16 把表头区换成整份回执，副本 rc=0——正文区没有顶格的 `FIELDS`
   标签行，两种切段同数。这种扰动只会让"红"变成不存在，所以它要么反过来切（C16 现在只取正文区），
   要么就如实进 `UNCOVERED`；判据必须允许「这条造不出红」被登记，否则下一轮会把它当成已验过的闸门。
⑥ **扰动落在 dict 的字面键上时，取值处要同批改**：把 needle `判据的可通过性` 改成别的串，`n[...]` 先撞
   `KeyError` 而不是走到断言 ⇒ 抛出行里没有闸门句。C27 因此用两条 `subs`（键＋取值处一起换），并把
   needle 选成**在同一节里出现两次**的串，让它红成"计数 2 被拦"而不是红成"查不到这个键"。
⑦ 跑本件之前先做**纸面预检**（锚点在源码里恰好命中 1 处、期望句在源码里 1 处、扰动后仍能 `compile()`）：
   这三条在 `perturb()` 里只验第一条，且要真跑 45 个子进程才暴露；纸面预检把「锚点 0 命中」这种空转
   在花钱之前挑出来（本轮 C8／C11／C33 三条就是这么发现的）。
"""
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
SRC = os.path.join(HERE, "am_s29_msg.py")
text = open(SRC, encoding="utf-8").read()

# (标签, [(源码正则, 替换文本)], 期望闸门句)
CASES = [
    ("C1 未暂存漂移检查把已暂存当成未暂存",
     [(r'^    assert not sh\("git", "-c", "core\.quotePath=false", "diff", "--name-only"\)\.strip\(\), \\$',
       '    assert not sh("git", "-c", "core.quotePath=false", "diff", "--cached", "--name-only").strip(), \\')],
     "还有未暂存的已跟踪改动"),
    ("C2 无代码腿普查的作用域指到 docs（本轮真正动过的面）",
     [(r'"--", "src", "tests"\)', '"--", "docs")')],
     "『纯归档腿』成了假话"),
    ("C3 一件落进两个桶（回执链桶吞掉表头控件面）",
     [(r'\^am_s%d_hdr\\\.py\$"', '^am_s%d_hdr.*\\.py$"')],
     "角色口径不唯一，拒绝生成"),
    ("C4 桶普查静默截断（每桶最多记三件）",
     [(r'^        hits\[m\[0\]\]\.append\(b\)$',
       '        if len(hits[m[0]]) < 3:\n            hits[m[0]].append(b)')],
     "≠ 新增总数"),
    ("C5 shortstat 与 numstat 两出处（把插入／删除两列读反）",
     [(r'^    ns = \(len\(rows\), sum\(int\(r\[0\]\) for r in rows\), sum\(int\(r\[1\]\) for r in rows\)\)$',
       '    ns = (len(rows), sum(int(r[1]) for r in rows), sum(int(r[0]) for r in rows))')],
     "≠ numstat 现算"),
    ("C6 shortstat 的删除列整组没吃到（S7 首版栽过的单复数形状）",
     [(r'\(\?:, \(\\d\+\) deletions\?\\\(-\\\)\)\?', '')],
     "≠ numstat 现算"),
    ("C7 现算指纹取错列（md5sum 输出的第二列是文件名）",
     [(r"cut -d' ' -f1", "cut -d' ' -f2")],
     "指纹四路不齐"),
    ("C8 生成器的闭合式模式与上一片回执的实际形状不一致 ⇒ 拒绝引用",
     [(r'\\\+ 本片新增\\\(A\\\)', r'\+ 本片新增\(B\)')],
     "模式取到 0 处"),
    ("C9 闭合式右边的算术闭合失效（把声称值再 +1 也当成立）",
     [(r'== int\(expect\), "闭合式不成立', '== int(expect) + 1, "闭合式不成立')],
     "闭合式不成立"),
    ("C10 落地判读行的期望写错（不再要求 ALL CHECKS PASS）",
     [(r'verdict\.strip\(\)\.endswith\("ALL CHECKS PASS"\)', 'verdict.strip().endswith("ALL CHECKS RED")')],
     "回执判读行不是"),
    ("C11 回执正文的 truncated 声称读反",
     [(r'^    assert tr == "False", "truncated=%s ⇒ 回执正文被截断，不许引用" % tr$',
       '    assert tr == "True", "truncated=%s ⇒ 回执正文被截断，不许引用" % tr')],
     "回执正文被截断"),
    ("C12 tree 条目＝blob＋tree 改成差（等式形状失效）",
     [(r'int\(entries\) == int\(blob\) \+ int\(treeent\)', 'int(entries) == int(blob) - int(treeent)')],
     "tree 条目 ≠ blob＋tree"),
    ("C13 回执表头的 dirty 等式换成别的值",
     [(r'    assert curl\["dirty"\] == "0", "回执表头 dirty', '    assert curl["dirty"] == "1", "回执表头 dirty')],
     "dirty(src+tests)"),
    ("C14 cmp 三件套的期望值写错（第三腿本该非 0）",
     [(r'    assert rcs == \[0, 1, 1\]', '    assert rcs == [0, 1, 0]')],
     "cmp 三件套复跑＝"),
    ("C15 正文区逐字节比对少取一字节（偏移腿失效）",
     [(r'    assert rb\(CURL\)\[off:\] == rb\(RAWR\)', '    assert rb(CURL)[off + 1:] == rb(RAWR)')],
     "回执正文区与原始 stdout 不逐字节相同"),
    ("C16 顶格字段数改从正文区取（切段方向反了＝表头区一件没数）",
     [(r'^    head_lines = rb\(CURL\)\[:int\(curl\["boff"\]\)\]\.decode\("utf-8"\)\.splitlines\(\)$',
       '    head_lines = rb(CURL)[int(curl["boff"]):].decode("utf-8").splitlines()')],
     "≠ FIELDS 声明"),
    ("C17 FIELDS 的分母取不到（在生成器里找不到那条赋值）",
     [(r'== "FIELDS":', '== "FIELDS_ZZ":')],
     "里取不到 FIELDS 赋值"),
    ("C18 尺子 md5 被改 ⇒ 与回执表头两出处互证失效",
     [(r'^    fam = md5\(RULER, 12\)$', '    fam = md5(RULER, 12) + "0"')],
     "≠ 回执表头声称"),
    ("C19 尺子普查 glob 指到不存在的目录（本件没被自己数到）",
     [(r'"docs/evidence/\*/am_s7_verify\*\.py"', '"docs/evidence/2099-*/am_s7_verify*.py"')],
     "计数漏了本体"),
    ("C20 SRCFINGER 的键名集换了 ⇒ 起／止两路取不到",
     [(r'\^SRCFINGER_\(START\|END\):', r'^SRCFINGER_(START|STOP):')],
     "SRCFINGER 起≠止"),
    ("C21 两条登记红的 E 行条数期望改成 1",
     [(r'    assert len\(elines\) == 2', '    assert len(elines) == 1')],
     "条红的 E 行实取"),
    ("C22 HEAD 片号取数的形状放宽到「分片 (N)」⇒ 同一行两处命中",
     [(r'one\(r"S7 分片 \(\\d\+\)", SUBJ', 'one(r"分片 (\\d+)", SUBJ')],
     "模式取到 2 处"),
    ("C23 最新回执号角本取成最小号（片号口径失效）",
     [(r'assert max\(RCPT\) == PREVSLICE', 'assert min(RCPT) == PREVSLICE')],
     "最新回执号角本"),
    ("C24 本轮回执缺席的检查被反写",
     [(r'assert SLICE not in RCPT', 'assert SLICE in RCPT')],
     "已经存在"),
    ("C25 节号推进的等式被写成「同一节」（形制约束＝另起一节这件事失去检查）",
     [(r'assert int\(secn\) == PREVSEC \+ 1', 'assert int(secn) == PREVSEC')],
     "形制约束没落实"),
    ("C26 切段边界一律取到节尾（每条子条吞掉后面的子条＝歧义）",
     [(r'^        end = text\.index\("- \*\*%s\*\*" % labs\[i \+ 1\]\) if i \+ 1 < len\(labs\) else len\(text\)$',
       '        end = len(text)')],
     "表头生成器会当场红"),
    ("C27 唯一性 needle 换成节内出现两次的串（计数 2 必须被拦，且键要同批改否则先撞 KeyError）",
     [(r'\(prevlog, "当前「", "判据的可通过性"\)\}', '(prevlog, "当前「", "分片 28")}'),
      (r'n\["判据的可通过性"\] == 1', 'n["分片 28"] == 1')],
     "节内唯一性破了"),
    ("C28 上一节引用数取成「全部顶级子条」（与自查件两出处失效）",
     [(r'^    prev_cite = len\(\[s for _l, s in top_bullets\(prev_body\)\[1\] if prevlog in s\]\)$',
       '    prev_cite = len(top_bullets(prev_body)[1])')],
     "≠ 归档自查件"),
    ("C29 自查件的档案读数指到表头控件面（取数面错了）",
     [(r'ARCHK_RAW = "%s/_raw_s%d_archive_check\.out"', 'ARCHK_RAW = "%s/_raw_s%d_hdr_controls.out"')],
     "模式取到 0 处"),
    ("C30 局部仪器的锚点句期望写错（把它的红当成没红）",
     [(r'    assert "锚点被本轮碰坏" in last', '    assert "ZZ_不是锚点句" in last')],
     "抛出的那一行不是锚点闸门"),
    ("C31 局部仪器的 stdout 断言改成读另一条流（红件的 stderr 被当成越界输出）",
     [(r'^    assert not r\.stdout, "局部仪器复跑竟有 stdout（%d B）⇒ 归档件里那份不是同一条运行" % len\(r\.stdout\.encode\(\)\)$',
       '    assert not r.stderr, "局部仪器复跑竟有 stdout（%d B）⇒ 归档件里那份不是同一条运行" % len(r.stdout.encode())')],
     "局部仪器复跑竟有 stdout"),
    ("C32 GPU 匹配式的正对照指到不存在的解释器 ⇒ 那个 0 是假零",
     [(r'    ctl = mine\(\["/tmp/amvenv/bin/python fake\.py"\]\)',
       '    ctl = mine(["/tmp/no_such_env/bin/python fake.py"])')],
     "GPU 匹配式的正对照没命中"),
    ("C33 GPU 的「我的进程」读了自己的 cmdline（分母与分子错位）",
     [(r'^    real = mine\(\[open\("/proc/%s/cmdline" % p, "rb"\)\.read\(\)\.decode\("utf-8", "replace"\) for p in pids\]\)$',
       '    real = mine([open("/proc/self/cmdline", "rb").read().decode("utf-8", "replace")])')],
     "我的解释器在卡上留有进程"),
    ("C34 V 的字典字面量出现重复键（Python 静默 last-wins）",
     [(r'^        "SLICE": str\(SLICE\), "PREVSLICE": str\(PREVSLICE\),',
       '        "SLICE": str(SLICE), "SLICE": str(SLICE), "PREVSLICE": str(PREVSLICE),')],
     "V 里有重复字面键"),
    ("C35 V 的字面里出现非字符串常量键（** 解包）",
     [(r'^        "SLICE": str\(SLICE\), "PREVSLICE": str\(PREVSLICE\),',
       '        **dict(ZZ="1"), "SLICE": str(SLICE), "PREVSLICE": str(PREVSLICE),')],
     "非字符串常量键"),
    ("C36 键从字面量之外塞进来（字面键表 ≠ 实得键集合）",
     [(r'^    # 键表自查：读 \*\*`__file__`\*\*',
       '    V["ZZ_NOT_LITERAL"] = "1"\n    # 键表自查：读 **`__file__`**')],
     "字面键表 ≠ 实得键集合"),
    ("C37 模板里有没取值的字段（占位符替换没走完）",
     [(r'^TMPL = """S7 分片 @@SLICE@@：', 'TMPL = """S7 分片 @@ZZ_NO_FIELD@@：')],
     "模板有未取值的字段"),
    ("C38 取值表里有模板没用到的读数（＝下一条等着的谎）",
     [(r'^        "SLICE": str\(SLICE\), "PREVSLICE": str\(PREVSLICE\),',
       '        "ZZ_UNUSED": "1", "SLICE": str(SLICE), "PREVSLICE": str(PREVSLICE),')],
     "取值表里有模板没用到的读数"),
    ("C39 现取 HEAD 的形状降级成短 sha（40 位口径失效）",
     [(r'^    head_now = sh\("git", "rev-parse", "HEAD"\)\.strip\(\)$',
       '    head_now = sh("git", "log", "-1", "--format=%h").strip()')],
     "现取 HEAD 不是 40 位十六进制"),
    ("C40 本地 HEAD 与回执所验的树等式被反写",
     [(r'assert sh\("git", "rev-parse", "HEAD"\)\.strip\(\) == curl\["head40"\]',
       'assert sh("git", "rev-parse", "HEAD").strip() != curl["head40"]')],
     "本地 HEAD ≠ 回执所验的树"),
    ("C41 五张归档面的清单少一张（修改面数与清单不再互证）",
     [(r'^        "\.workbuddy/memory/2026-10-10\.md", "\.workbuddy/memory/MEMORY\.md"\]$',
       '        ".workbuddy/memory/2026-10-10.md"]')],
     "修改面 ≠ 五张归档面"),
    ("C42 numstat 增行与行数差的等式 +1（两出处互证失效）",
     [(r'        assert add == n_staged - n_head', '        assert add == n_staged - n_head + 1')],
     "≠ 行数差"),
    ("C43 「只允许净增」那条改成必须恰好删 1 行",
     [(r'        assert dele == 0', '        assert dele == 1')],
     "相对 HEAD 有"),
    ("C44 桶形的片号插值被写死成上一轮的号（机械克隆把上一片的故事安在本片头上）",
     [(r'\^am_s%d_hdr\\\.py\$" % \(PREVSLICE, PREVSLICE, PREVSLICE\)',
       '^am_s%d_hdr\\.py$" % (27, 27, 27)')],
     "角色口径不唯一，拒绝生成"),
]

ENV = dict(os.environ)
ENV["CUDA_VISIBLE_DEVICES"] = ""
ENV["JAX_PLATFORMS"] = "cpu"
PREFIX = "zctl_s29_msg_"
SRCNAME = os.path.basename(SRC)


def perturb(name, subs, src=None):
    """把一条扰动落到源码上；**锚点必须在源码里唯一命中**，否则这条扰动不可信（C45 验的就是这句）。"""
    bad = text if src is None else src
    for pat, rep in subs:
        hits = re.findall(pat, bad, re.M)
        assert len(hits) == 1, "%s：扰动锚点在源码里命中 %d 处 ⇒ 扰动本身不可信" % (name, len(hits))
        bad = re.sub(pat, lambda m: rep, bad, count=1, flags=re.M)
    assert bad != (text if src is None else src), name + "：扰动没改变源码 ⇒ 空转"
    return bad


def trace_line(msg):
    """从 stderr 里取**最后**一帧 `File "am_s29_msg.py", line N`——副本的文件名是临时名，所以按行号取。
    副本可能比原件多几行（有的扰动就是插一行），所以行号判定只按 ±3 的窗口给结论。"""
    cand = re.findall(r'File "[^"]*/(am_s29_msg\.py|zctl_s29_[\w.-]+)", line (\d+)', msg)
    return int(cand[-1][1]) if cand else None


def raised(msg):
    """把**实际抛出的那一行**取回来（红得对不对，先看它红成了什么）。
    判据必须用它，而不是"整段 stderr 里出现过那句话"：3.11+ 的 traceback 会把**源码行原样回显**，
    而闸门句就写在源码里 ⇒ 只要栈落到附近任何一行，整段输出都含那句＝假命中。"""
    hits = re.findall(r"^(?:\w+Error|AssertionError)[^\n]*", msg, re.M)
    return hits[-1] if hits else "（没取到异常行）"


def gate_line(expect):
    """期望句在本件源码里的行号；出现 0 次或多次都不给结论（句子里含 %s 时按前缀取）。"""
    key = expect.split("%")[0].strip() or expect
    hits = [i + 1 for i, l in enumerate(text.splitlines()) if key in l]
    return hits


def run_one(bad):
    """在同一目录落一份扰动副本、跑它、拿 rc＋合并输出，然后删掉副本。"""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8",
                                     dir=HERE, prefix=PREFIX) as fh:
        fh.write(bad)
        tmp = fh.name
    try:
        r = subprocess.run([sys.executable, tmp], cwd=REPO, env=ENV,
                           capture_output=True, text=True)
        return r.returncode, (r.stderr or "") + (r.stdout or "")
    finally:
        os.unlink(tmp)


def residue():
    return [f for f in os.listdir(HERE) if f.startswith(PREFIX)]


# 基线＝逐字节相同的副本必须**通过**并真的走到打印分支。没有这一步，上面所有红都可能只是环境错（坑①④）。
BASE_RC, BASE_MSG = run_one(text)
assert BASE_RC == 0, ("未扰动副本就没通过（rc=%s）⇒ 本件的正对照全部无效，先修仪器。stderr 尾巴：\n%s"
                      % (BASE_RC, "\n".join(BASE_MSG.splitlines()[-8:])))
assert "---- 现取核对 ----" in BASE_MSG, "未扰动副本没走到打印分支 ⇒ 副本跑的不是同一条路径"
assert not residue(), "扰动副本没被清掉：" + "、".join(residue())

rows, allred = [], True
for name, subs, expect in CASES:
    rc, msg = run_one(perturb(name, subs))
    gl = gate_line(expect)
    tl = trace_line(msg)
    err = raised(msg)
    text_hit = expect in err
    if len(gl) == 1 and tl is not None:
        near = gl[0] - 3 <= tl <= gl[0] + 3
        why = "句在源码 %d 行／raise 在 %d 行%s" % (gl[0], tl, "（同一条闸门）" if near else "（不是同一条！）")
    else:
        near = True
        why = "句在源码 %s 处／raise 在 %s 行（位置判定不适用，只按抛出行判红）" % (len(gl), tl)
    ok = rc != 0 and text_hit and near
    allred = allred and ok
    rows.append((name, rc, ok, expect, why + "｜实抛＝" + err[:88]))

# C45＝对**本件自己**的闸门：锚点不存在时 perturb 必须拒绝。不验这条，下一轮克隆后"锚点已移位"的
# 扰动会静默地没改动任何源码（＝跑了一遍真件），红就变成假红。
try:
    perturb("C45", [(r'^    nvzz = 1$', "pass")])
    rows.append(("C45 锚点普查（本件自己的闸门）", 0, False, "扰动锚点在源码里命中 0 处", ""))
    allred = False
except AssertionError as e:
    hit = "扰动锚点在源码里命中 0 处" in str(e)
    rows.append(("C45 锚点普查（本件自己的闸门）", 3 if hit else 0, hit,
                 "扰动锚点在源码里命中 0 处", "本件自己的异常"))
    allred = allred and hit
    assert not residue(), "C45 没落副本，但目录里有残留"

# 本轮**造不出红路**的闸门，逐条登记原因（不假装它们被验过）。
UNCOVERED = [
    ("角色桶非空（`角色桶 %s 里一件都没有`）", "把任一桶形改窄会先命中「落进 0 个桶」，改成万能形会先命中「落进 2 个桶」⇒ "
     "这条只能被上游两条拦住；它的价值是给下一轮「某角色整族改名」兜底（本轮 C44 造的就是那一条的红路）。"),
    ("numstat 里有 `-` 行（二进制／rename）", "红路需要往暂存集里放一张二进制面，与本轮「只归档文本面」直接冲突，"
     "不能为造红去改暂存集；两列求和的机制已由 C5／C42 覆盖。"),
    ("我的解释器在卡上留有进程（真实读数）", "C33 是把**别的 cmdline** 冒充成我的来验闸门本身能红，不等于本轮真没占卡；"
     "真实读数是打印行「GPU 卡上进程数／我的进程数」两个数（跑时现取，口径与全量回归件相同）。"),
    ("回执表头 SRC 指纹与现算闭合到「打印精度」", "把指纹改成 31 位会先命中 `SRCFINGER` 的形状断言（C20），"
     "单独造「只差一位」的红路需要改档案面，不在本轮动。"),
    ("顶格字段计数＝**只**依赖表头区这一段", "本轮先试过把 `head_lines` 换成整份回执（表头＋正文），副本 rc=0 而不是红 ⇒ "
     "那条扰动已从 CASES 删掉，只在这里留名：正文区里没有顶格的 `FIELDS` 标签行，两种切段同数，所以「切段口径」这一维"
     "只有**反过来切**（只取正文区，C16）才造得出红。教训＝一条扰动改了源码却没改行为＝空转，判据必须允许「红不了」这一结果被登记。"),
]

print("基线（未扰动副本）rc=%s 且走到打印分支＝True（所有红的对照面）" % BASE_RC)
print("解释器＝%s（GPU 正对照读的是自己的 cmdline）｜被扰动的件＝%s" % (sys.executable, SRCNAME))
print("扰动条数＝%d｜每条判据＝rc≠0 且**抛出的那一行**含指定闸门句 且 raise 行号落在该句源码位置（±3 行）" % len(rows))
for name, rc, ok, expect, why in rows:
    print(("  " if ok else "× ") + name + "｜rc=" + str(rc) + "｜期望句「" + expect + "」"
          + ("命中" if ok else "**未命中**") + ("｜" + why if why else ""))
print("本轮造不出红路的闸门＝%d 条（逐条附原因，不算被验）：" % len(UNCOVERED))
for gate, why in UNCOVERED:
    print("  · " + gate + "：" + why)
print("CONTROL_VERDICT =", "ALL CONTROLS CAN GO RED" if allred else "PROBLEM（有闸门空转）")
sys.exit(0 if allred else 3)
