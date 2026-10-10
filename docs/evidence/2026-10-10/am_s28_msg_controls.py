#!/usr/bin/env python3
"""提交信息生成器 am_s28_msg.py 的**正对照**：每条扰动都必须让它拒绝，否则那条闸门是空转。

扰动改的是**生成器的源码副本**（正则级改写），跑的是同一件（同一条 `python <file>`）⇒ 被验的就是
落地那把仪器本身，不是另写的平行实现。判据＝rc≠0 **且** 抛出的那一行含指定的闸门句 **且** raise 的行号
落在该句在源码里的位置附近（±3 行）——只看 rc 会把"环境错"记成"闸门对"。

本轮（分片 28）新增的闸门族全部有红路：V 键表 ast 普查（重复字面键／非串键／字面外塞键）、
达标线两源（面↔待提交驱动 BARS）、靶偏差独立重算、代价行／散差表行的**唯一性**、引用节号现取
（sec_of）、软闸标签集三兄弟（两腿一致／集合大小＝行数／J6 在场）、布尔出口字段表与 µm 普查、
台账"未改"＝暂存 blob 逐字节等于 HEAD、桶普查（0 桶／两桶／截断）、GPU 匹配式正对照与 cmdline 分母。

四条**仪器自己的**坑（前两条是实测继承，后两条本轮新增）：
① 判据不能只看 rc≠0，副本也不能写在 /tmp。生成器用自身位置反推 REPO ⇒ 副本必须落在**同一目录**，
   并加一条**未扰动基线必须 rc=0**（C0），否则所有红都不作数。
② **期望闸门句里不许写死节号**：节号是档案的读数，写死就随片腐烂（红是红了、红的却不是指定闸门）。
   本件的节号期望由 `git show :docs/开发日志.md` 现取最后一个 `### 26.NN` 拼出来，不手打。
③ 键表闸门原本 `open(os.path.join(HERE, SELF))` 读的是**按文件名找到的原件** ⇒ 副本扰动 V 的字面键时，
   ast 数到的是原件的键表，三条键表闸门永远不红（＝空转）。本轮把它改成读 `__file__`：直接跑时同一个
   文件、语义不变，副本才审得到自己。C36–C38 验的就是这条改动确实把红路打通了。
④ **必须用 /tmp/amvenv/bin/python 跑本件**：生成器的 GPU 正对照读的是 `/proc/self/cmdline`，
   换解释器 ⇒ C0 就没通过（那正是闸门该做的事，但不是本件的判据）。C0 失败时本件把 stderr 尾巴印出来。
⑤ 判据里的"含指定闸门句"必须按**抛出的那一行**判，不能按整段 stderr 判：3.11+ 的 traceback 把源码行
   原样回显，而闸门句本来就写在源码里 ⇒ 只要栈落在附近任何一行，整段输出都含那句＝**假命中**。
   实测到：C37 那轮真抛的是 `TypeError: expected AST, got 'NoneType'`（生成器的消息式子对 `**解包`
   的 None 键调了 `ast.dump`）——按整段判会记成"闸门命中"，按抛出行判才记成"闸门自己会崩"（已修）。
"""
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
SRC = os.path.join(HERE, "am_s28_msg.py")
text = open(SRC, encoding="utf-8").read()
DEV_STAGED = subprocess.run(["git", "-c", "core.quotePath=false", "show", ":docs/开发日志.md"],
                            cwd=REPO, capture_output=True, text=True, check=True).stdout
SECN = re.findall(r"^### 26\.(\d+)[^\n]*$", DEV_STAGED, re.M)[-1]

# (标签, [(源码正则, 替换文本)], 期望闸门句)
CASES = [
    ("C1 未暂存漂移检查把已暂存当成未暂存",
     [(r'^    assert not sh\("git", "-c", "core\.quotePath=false", "diff", "--name-only"\)\.strip\(\), \\$',
       '    assert not sh("git", "-c", "core.quotePath=false", "diff", "--cached", "--name-only").strip(), \\')],
     "有未暂存漂移"),
    ("C2 无代码腿普查的作用域指到 docs（本轮真正动过的面）",
     [(r'"--", "src", "tests"\)', '"--", "docs")')],
     "本轮声称无代码腿"),
    ("C3 一件落进两个桶（回执链桶吞掉表头控件面）",
     [(r'\^am_s27_hdr\\\.py\$"\)', '^am_s27_hdr.*\\.py$")')],
     "落进 2 个桶"),
    ("C4 桶普查静默截断（每桶最多记三件）",
     [(r'^        hits\[m\[0\]\]\.append\(b\)$',
       '        if len(hits[m[0]]) < 3:\n            hits[m[0]].append(b)')],
     "≠ 新增总数"),
    ("C5 shortstat 与 numstat 两出处（把插入／删除两列读反）",
     [(r'^    ns = \(len\(rows\), sum\(int\(r\[0\]\) for r in rows\), sum\(int\(r\[1\]\) for r in rows\)\)$',
       '    ns = (len(rows), sum(int(r[1]) for r in rows), sum(int(r[0]) for r in rows))')],
     "≠ numstat 现算"),
    ("C6 现算指纹取错列（md5sum 输出的第二列是文件名）",
     [(r"cut -d' ' -f1", "cut -d' ' -f2")],
     "指纹四路不闭合"),
    ("C7 上一片回执的形状变了 ⇒ 拒绝引用",
     [(r'^RE_C_CLOSE = re\.compile.*$',
       'RE_C_CLOSE = re.compile(r"闭合式 远端 (\\d+) == 父提交 (\\d+) \\+ 本片新增\\(B\\) (\\d+) ＝ (\\d+) -> (\\w+)")')],
     "回执里取不到闭合式"),
    ("C8 pilot 面四类行数不同轨（G9 行只取第一条）",
     [(r'^    g9 = RE_G9\.findall\(text\)$', '    g9 = RE_G9.findall(text)[:1]')],
     "pilot 面四类行数不等"),
    ("C9 靶偏差独立重算拿错量（用面积配宽度靶）",
     [(r'^        calcw = abs\(r\["width"\] - float\(r\["tw"\]\)\) / float\(r\["tw"\]\) \* 100\.0$',
       '        calcw = abs(r["area"] - float(r["tw"])) / float(r["tw"]) * 100.0')],
     "宽偏差：面印"),
    ("C10 待提交驱动里取不到 BARS（读错了面）",
     [(r'^    dblob = sh\("git", "show", ":" \+ DRIVER\)$', '    dblob = sh("git", "show", ":" + LEDGER)')],
     "里取不到 BARS 的宽／积判据"),
    ("C11 达标线两源互证失效（把驱动的宽／积 pass 线读成 fail 线）",
     [(r'^    mb = tuple\(mbw\.groups\(\)\) \+ tuple\(mba\.groups\(\)\)$',
       '    mb = tuple(mbw.groups())[::-1] + tuple(mba.groups())[::-1]')],
     "面里印的达标线"),
    ("C12 控件有一条红还被当正对照（只数 FAIL）",
     [(r'^    ctl_pass = \[k for k, _ in ctl if k == "PASS"\]$', '    ctl_pass = [k for k, _ in ctl if k == "FAIL"]')],
     "有控件是红的"),
    ("C13 代价登记行唯一性（面里出现第二条同形行 ⇒ 引哪条没有依据）",
     [(r'^    cost = RE_COST\.search\(ptext\)$',
       '    cost = RE_COST.search(ptext)\n    ptext = ptext + "\\n" + cost.group(0)')],
     "代价登记行在面里不止 1 条"),
    ("C14 读数核控件标签集变了（把 G1 也算控件 ⇒ 「／4」那个分母不成立）",
     [(r'^RE_CTL4 = re\.compile.*$', r'RE_CTL4 = re.compile(r"^\s+\[(PASS|FAIL)\] (C[1-4]|G1) ", re.M)')],
     "读数核合成控件的标签集"),
    ("C15 G9 前后互为确定性件（对照面其实来自另一条臂）",
     [(r'^    old_lines = solidus_lines\(rd\(PILOT_PRE\)\)$',
       '    old_lines = solidus_lines(rd(RAWN1)) * 3')],
     "形态整行不同"),
    ("C16 两腿分工（缺省卡腿被读成 amb 腿 ⇒ READOUT_FAIL 没了）",
     [(r'^    stxt, s2txt = rd\(SMOKE\), rd\(SMOKE_AMB\)$', '    stxt, s2txt = rd(SMOKE_AMB), rd(SMOKE_AMB)')],
     "READOUT_FAIL 行数＝"),
    ("C17 布尔出口的字段表漏一项（与「每条 N 个字段」那句不符）",
     [(r'^    bool_fields = \("接线", "宽有限", "积有限", "行数", "有熔", "触边"\)$',
       '    bool_fields = ("接线", "宽有限", "积有限", "行数", "有熔", "触")')],
     "布尔行的字段表与「每条 N 个字段」那句不符"),
    ("C18 µm 普查算错了行集（整张面而不是那四条布尔行）",
     [(r'^    n_boolum = sum\(l\.count\("µm"\) for l in bool_lines\)$',
       '    n_boolum = sum(l.count("µm") for l in s2txt.splitlines())')],
     "处 `µm`"),
    ("C19 --amb-k 引的是另一条腿的数（缺省卡 300K）",
     [(r'^    amb_s = float\(RE_AMB\.search\(s2txt\)\.group\(1\)\)$',
       '    amb_s = float(RE_AMB.search(stxt).group(1))')],
     "smoke ambk 腿面内 T_ambient"),
    ("C20 两条 smoke 腿的软闸标签集其实取自不同面",
     [(r'^    softlab = \[sorted\(set\(RE_SOFTLAB\.findall\(t\)\)\) for t in \(stxt, s2txt\)\]$',
       '    softlab = [sorted(set(RE_SOFTLAB.findall(t))) for t in (stxt, rd(PRE_REG_N))]')],
     "两条 smoke 腿的软闸标签集不同"),
    ("C21 软闸标签集大小 ≠ 面里 [SOFT] 行数（正则只认 G 族）",
     [(r'^RE_SOFTLAB = re\.compile.*$', 'RE_SOFTLAB = re.compile(r"^\\s+\\[SOFT\\]\\s+G\\d+", re.M)')],
     "≠ [SOFT] 行"),
    ("C22 形态档标签不叫 J6（⇒「按 J6 结构性禁掉形态数字」失去出处）",
     [(r'^    stxt, s2txt = rd\(SMOKE\), rd\(SMOKE_AMB\)$',
       '    stxt, s2txt = rd(SMOKE).replace("J6", "G7"), rd(SMOKE_AMB).replace("J6", "G7")')],
     "没有 J6"),
    ("C23 散差表行唯一性（宽度式子也匹配面积行 ⇒ 引用哪一条是运气）",
     [(r'^RE_P50W = re\.compile.*$',
       'RE_P50W = re.compile(r"^\\| (?:熔池宽度|顶面面积) \\| (\\d+) \\| \\*\\*([\\d.]+)%\\*\\* \\| \\*\\*([\\d.]+)%\\*\\* \\| ([\\d.]+)% \\|", re.M)')],
     "宽度散差行在暂存日志里不止 1 条"),
    ("C24 引用落到最后一节而不是表所在节（sec_of 的位置取错）",
     [(r'^    p50_sec = sec_of\(p50w\.start\(\)\)$', '    p50_sec = sec_of(devlog_staged.rfind("### 26."))')],
     "散差表两行现落"),
    ("C25 GPU 那句取第一次出现而不是最后一次（引成历史约束）",
     [(r'^    gpu_sec = sec_of\(devlog_staged\.rfind\(gpuq\)\)$', '    gpu_sec = sec_of(devlog_staged.find(gpuq))')],
     "而不是本轮 §26." + SECN),
    ("C26 台账「本轮未改」是假的（把它换成一张真的改过并暂存的面）",
     [(r'^LEDGER = "\.workbuddy/memory/MEMORY\.md"$', 'LEDGER = ".workbuddy/memory/2026-10-10.md"')],
     "的暂存 blob ≠ HEAD 版本"),
    ("C27 日志那句自身算术（三个数各 +1）",
     [(r'^    lun, lmod, ltot = \(int\(x\) for x in logs\[-1\]\.groups\(\)\)$',
       '    lun, lmod, ltot = tuple(int(x) + 1 for x in logs[-1].groups())')],
     "那句自身不算术闭合"),
    ("C28 差额取错项（拿修改面数当未跟踪数）",
     [(r'^    gap = len\(added\) - lun$', '    gap = len(added) - lmod')],
     "而本信息生成器桶只有"),
    ("C29 本件自身不落进最后一桶 ⇒「差额恰好是本件那几张」是假的",
     [(r'^    msg_bucket, msg_pat = BUCKETS\[-1\]$',
       r'    msg_bucket, msg_pat = BUCKETS[-1][0], r"^am_s99_nosuch\.py$"')],
     "不落进最后一桶"),
    ("C30 尺子 md5 被改 ⇒ 全树现算出一份都不认（家族普查失效）",
     [(r'^    ruler_md5 = md5\(os\.path\.join\(REPO, RULER\)\)$',
       '    ruler_md5 = md5(os.path.join(REPO, RULER)) + "0"')],
     "没被自己的 glob 数到"),
    ("C31 尺子 md5 两出处互证失效（回执声称的那串被改写）",
     [(r'ruler_md5=RE_C_RULERMD5\.search\(text\)\.group\(1\)',
       'ruler_md5=RE_C_RULERMD5.search(text).group(1)[:11] + "0"')],
     "≠ 上一片回执声称的"),
    ("C32 副本数两出处互证失效（回执多声称一份）",
     [(r'^                ruler_md5=RE_C_RULERMD5\.search\(text\)\.group\(1\), copies=RE_C_COPIES\.search\(text\)\.group\(1\),$',
       '                ruler_md5=RE_C_RULERMD5.search(text).group(1), copies=str(int(RE_C_COPIES.search(text).group(1)) + 1),')],
     "本轮全树现算"),
    ("C33 正对照个数与字母两出处（回执正文少认一条）",
     [(r'^    ctrl_letters = RE_CTLN\.findall\(rd\(CURL\)\)$', '    ctrl_letters = RE_CTLN.findall(rd(CURL))[:3]')],
     "有一侧的对照没被引到"),
    ("C34 GPU 匹配式的正对照指到不存在的解释器 ⇒ 那个 0 是假零",
     [(r'^    ctl = \[p for p in \["/proc/self/cmdline"\] if "/tmp/amvenv" in rd\(p\)\]$',
       '    ctl = [p for p in ["/proc/self/cmdline"] if "/tmp/no_such_env" in rd(p)]')],
     "GPU 匹配式的正对照没命中"),
    ("C35 cmdline 分母起错（只数命中项就会等于 0）",
     [(r'^    nv_mine, nv_read = \[\], 0$', '    nv_mine, nv_read = [], -1')],
     "只读回"),
    ("C36 V 的字典字面量出现重复键（Python 静默 last-wins）",
     [(r'^        "SLICE": "28", "PREVSLICE": "27", "SECN": secn, "NBULLET": str\(len\(bullets\)\),$',
       '        "SLICE": "28", "SLICE": "28", "PREVSLICE": "27", "SECN": secn, "NBULLET": str(len(bullets)),')],
     "V 里有重复字面键"),
    ("C37 V 的字面里出现非字符串常量键（解包）",
     [(r'^        "SLICE": "28", "PREVSLICE": "27", "SECN": secn, "NBULLET": str\(len\(bullets\)\),$',
       '        **dict(ZZ="1"), "SLICE": "28", "PREVSLICE": "27", "SECN": secn, "NBULLET": str(len(bullets)),')],
     "V 的字面里有非字符串常量键"),
    ("C38 键从字面量之外塞进来（字面键表 ≠ 实得键集合）",
     [(r'^    \}\n    # 门：V 的字面键', '    }\n    V["ZZ_NOT_LITERAL"] = "1"\n    # 门：V 的字面键')],
     "≠ 实得键"),
    ("C39 模板里有没取值的字段（占位符替换没走完）",
     [(r'^TMPL = """S7 分片 @@SLICE@@：', 'TMPL = """S7 分片 @@ZZ_NO_FIELD@@：')],
     "模板有未取值的字段"),
    ("C40 取值表里有模板没用到的读数（＝下一条等着的谎）",
     [(r'^        "SLICE": "28", "PREVSLICE": "27", "SECN": secn, "NBULLET": str\(len\(bullets\)\),$',
       '        "ZZ_UNUSED": "1", "SLICE": "28", "PREVSLICE": "27", "SECN": secn, "NBULLET": str(len(bullets)),')],
     "取值表里有模板没用到的读数"),
]

ENV = dict(os.environ)
ENV["CUDA_VISIBLE_DEVICES"] = ""
ENV["JAX_PLATFORMS"] = "cpu"
PREFIX = "zctl_s28_msg_"
SRCNAME = os.path.basename(SRC)


def perturb(name, subs, src=None):
    """把一条扰动落到源码上；**锚点必须在源码里唯一命中**，否则这条扰动不可信（C41 验的就是这句）。"""
    bad = text if src is None else src
    for pat, rep in subs:
        hits = re.findall(pat, bad, re.M)
        assert len(hits) == 1, "%s：扰动锚点在源码里命中 %d 处 ⇒ 扰动本身不可信" % (name, len(hits))
        bad = re.sub(pat, lambda m: rep, bad, count=1, flags=re.M)
    assert bad != (text if src is None else src), name + "：扰动没改变源码 ⇒ 空转"
    return bad


def trace_line(msg):
    """从 stderr 里取**最后**一帧 `File "am_s28_msg.py", line N`——副本的文件名是临时名，所以按行号取。
    注意：副本可能比原件多几行（有的扰动就是插一行），所以行号判定只按 ±3 的窗口给结论。"""
    cand = re.findall(r'File "[^"]*/(am_s28_msg\.py|zctl_[\w.-]+)", line (\d+)', msg)
    return int(cand[-1][1]) if cand else None


def raised(msg):
    """把**实际抛出的那一行**取回来（红得对不对，先看它红成了什么）。
    判据必须用它，而不是"整段 stderr 里出现过那句话"：3.11+ 的 traceback 会把**源码行原样回显**，
    而闸门句就写在源码里 ⇒ 只要栈落到附近任何一行，整段输出都含那句＝假命中。本件的 C37 实测到这一点
    （那轮真抛的是 `TypeError: expected AST, got 'NoneType'`，因为消息式子对 `**解包` 的 None 键调了
    ast.dump；按整段判红会把它记成"闸门命中"，按抛出行判红才把它记成"闸门自己会崩"＝已修）。"""
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


# 基线＝逐字节相同的副本必须**通过**并真的走到打印分支。没有这一步，上面所有红都可能是环境错（坑①④）。
BASE_RC, BASE_MSG = run_one(text)
assert BASE_RC == 0, ("未扰动副本就没通过（rc=%s）⇒ 本件的正对照全部无效，先修仪器。stderr 尾巴：\n%s"
                      % (BASE_RC, "\n".join(BASE_MSG.splitlines()[-6:])))
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

# C41＝对**本件自己**的闸门：锚点不存在时 perturb 必须拒绝。不验这条，下一轮克隆后"锚点已移位"的
# 扰动会静默地没改动任何源码（＝跑了一遍真件），红就变成假红。
try:
    perturb("C41", [(r'^    nv_mine, nv_read = \[\], 7$', "pass")])
    rows.append(("C41 锚点普查（本件自己的闸门）", 0, False, "扰动锚点在源码里命中 0 处", ""))
    allred = False
except AssertionError as e:
    hit = "扰动锚点在源码里命中 0 处" in str(e)
    rows.append(("C41 锚点普查（本件自己的闸门）", 3 if hit else 0, hit,
                 "扰动锚点在源码里命中 0 处", "本件自己的异常"))
    allred = allred and hit
    assert not residue(), "C41 没落副本，但目录里有残留"

# 本轮**造不出红路**的闸门，逐条登记原因（不假装它们被验过）。
UNCOVERED = [
    ("「未入库…」登记的最新一条落在哪一节", "暂存日志里两条同句式登记（51＋6＝57、57＋6＝63）都在 §26.%s 内，"
     "把最新一条换成最早一条不会跨节 ⇒ 这条闸门的红路要等有同类登记落在别的节才造得出；"
     "同一取数机制由 C24／C25 覆盖。" % SECN),
    ("我的解释器在 GPU 上留有进程（nv_mine 必须为空）", "红路需要**我自己的进程占着卡**，与本轮「不动 GPU」直接冲突，"
     "不能为造红去起进程；它配套的正对照由 C34／C35 覆盖（假零检测）。"),
    ("日志登记的修改面数＝暂存实得", "把日志里的数改到别处会先命中上一条算术闭合闸门（C27），单独的红路需要重写档案面。"),
]

print("基线（未扰动副本）rc=%s 且走到打印分支＝True（所有红的对照面）" % BASE_RC)
print("解释器＝%s（GPU 正对照读的是自己的 cmdline）｜期望句节号 §26.%s 现取自暂存日志最后一个标题"
      % (sys.executable, SECN))
print("扰动条数＝%d｜每条判据＝rc≠0 且**抛出的那一行**含指定闸门句 且 raise 行号落在该句源码位置（±3 行）" % len(rows))
for name, rc, ok, expect, why in rows:
    print(("  " if ok else "× ") + name + "｜rc=" + str(rc) + "｜期望句「" + expect + "」"
          + ("命中" if ok else "**未命中**") + ("｜" + why if why else ""))
print("本轮造不出红路的闸门＝%d 条（逐条附原因，不算被验）：" % len(UNCOVERED))
for gate, why in UNCOVERED:
    print("  · " + gate + "：" + why)
print("CONTROL_VERDICT =", "ALL CONTROLS CAN GO RED" if allred else "PROBLEM（有闸门空转）")
sys.exit(0 if allred else 3)
