#!/usr/bin/env python3
r"""提交信息生成器 am_s30_msg.py 的**正对照**（全离线）：每条扰动都要让它拒绝在**自己那条闸门**上。

扰动改的是**生成器的源码副本**（正则级改写），跑的是同一件（同一条 `python <file>`）⇒ 被验的就是
要落地的那把仪器本身，不是另写的平行实现。判据＝rc≠0 **且**抛出的那一行含指定的闸门句 **且** raise 的
行号落在该句在源码里的位置（±3 行）——只看 rc 会把"环境错"记成"闸门对"。

本轮（分片 30）新增闸门的红路，逐条都要有：
① 口径 ⑥（角色桶 ↔ §26.32 落笔分组）：C4 把桶普查静默截断、C5 把自件桶的片号插值写成上一轮的号、
   C7／C8 把配对错位、C9 把落笔闭合式两边写反；
② 口径 ⑦（指纹的 locale 耦合）：C16 取消钉环境（两条同值 ⇒ 那条"必须不同"该红）、C17 交换两条读数
   （参与闭合的那条变成钉了 locale 的那条）；
③ 口径 ⑧（换算出来的读数也要有第二出处）：C26 把时长换算退回 `divmod`（7717 s 会渲染成 `2h517m`）、
   C27 把承重面从 max 改成 min、C28／C29 拆 PLACEHOLDER 的双向自查；
④ 继承 families：按位取行（C45）、逐字节比较错位（C12）、顶格切段（C13／C14）、普查含本体（C15）、
   局部仪器复跑的两条流与标本末行（C18／C19／C20）、GPU 的两个假零形状（C34／C35）、键表自查（C30–C33）。

九条**仪器自己的**坑（继承＋本轮新增）：
① 判据不能只看 rc≠0；副本也不能写在 /tmp——生成器用自身位置反推 REPO ⇒ 副本必须落在**同一目录**，
   并加一条**未扰动基线必须 rc=0 且走到打印分支**（C0），否则所有红都不作数。
② **期望闸门句里不许写死节号／片号／读数**：那些是档案的读数，写死就随片腐烂。本件所有期望句都取自
   闸门自己的固定短语。
③ 按**抛出的那一行**判红，不按整段 stderr：3.11+ 的 traceback 把源码行原样回显，闸门句本来就写在源码里。
   raise 的行号取**最后一个**匹配本副本名的帧——第一版取第一条，量到的是调用链而不是抛出点。
④ **必须用 /tmp/amvenv/bin/python 跑本件**，且环境里不许有 `LC_ALL`（生成器第一条就断言它缺席，并把
   不钉 locale 的那条读数用于四路闭合）；不满足就跑第一步会全红，那不是闸门红。
⑤ **改了源码 ≠ 改了行为**：一条扰动若让副本 rc=0，它要么反过来切，要么如实进 `UNCOVERED`（本轮的
   七条见文件末尾）。判据必须允许「这条造不出红」被登记，否则下一轮会把它当成已验过的闸门。
⑥ **扰动落在 dict 的字面键上时，取值处要同批改**；needle 选成在源码里**多处**出现的串时，红可能落在
   别的那条闸门上 ⇒ 本件对每条都验「抛出行 ±3」。
⑦ 跑本件之前先做**纸面预检**（锚点在源码里恰好命中 1 处、期望句在源码里可定位、扰动后仍能 `compile()`）：
   这三条要靠真跑几十个子进程才暴露，纸面预检把「锚点 0 命中」这种空转在花钱之前挑出来（`--precheck`）。
⑧ **本件的基线依赖一个时序**：生成器有一条「本件的干跑控件面必须在入库清单里」的断言 ⇒ 跑本件**之前**
   要先 `git add` 本件自己（它的原始件此刻不存在，生成器不读它们，只按文件名计数）。这条不是循环：
   循环只发生在"生成器读自己控件的 stdout"那种引用上，所以本件的读数由**分片 31** 引用，本片不引用。
⑨ 每条子运行自带 `SUB_TIMEOUT`，超时按**进程组** SIGKILL；超时判成**环境故障**
   （`CONTROL_VERDICT = ENV FAILURE…`＋exit 4），既不判红也不判过——本件全离线，超时是机器不是闸门。
"""
import os
import re
import signal
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
SRCNAME = "am_s30_msg.py"
SRC = os.path.join(HERE, SRCNAME)
PREFIX = "zctl_s30_msg_"
SUB_TIMEOUT = 300
ENV_FAIL_STOP = 3
text = open(SRC, encoding="utf-8").read()

ENV = dict(os.environ)
ENV["CUDA_VISIBLE_DEVICES"] = ""
ENV["JAX_PLATFORMS"] = "cpu"

# (标签, [(源码正则, 替换文本)], 期望闸门句)
CASES = [
    ("C1 未暂存漂移检查把已暂存当成未暂存",
     [(r'^    assert not sh\("git", "-c", "core\.quotePath=false", "diff", "--name-only"\)\.strip\(\), \\$',
       '    assert not sh("git", "-c", "core.quotePath=false", "diff", "--cached", "--name-only").strip(), \\')],
     "还有未暂存的已跟踪改动"),
    ("C2 无代码腿普查的作用域指到 docs（本轮真正动过的面）",
     [(r'"--", "src", "tests"\)', '"--", "docs")')],
     "纯归档腿』成了假话"),
    ("C3 一件落进两个桶（回执链桶吞掉表头控件面）",
     [(r'\^am_s%d_hdr\\\.py\$"', '^am_s%d_hdr.*\\.py$"')],
     "角色口径不唯一，拒绝生成"),
    ("C4 桶普查静默截断（每桶最多记三件）",
     [(r'^        hits\[m\[0\]\]\.append\(b\)$', "        hits[m[0]] = (hits[m[0]] + [b])[:3]")],
     "≠ 新增总数"),
    ("C5 本件自件桶的片号插值写死成上一轮的号（换轮后件落不进任何桶）",
     [(r'% \(SLICE, SLICE, PREVSLICE, SLICE\)\)', '% (PREVSLICE, PREVSLICE, PREVSLICE, PREVSLICE))')],
     "角色口径不唯一，拒绝生成"),
    ("C6 跨日期目录硬写成本件目录",
     [(r'^PREVDIR = os\.path\.dirname\(CURL\)$', "PREVDIR = E")],
     "上一片回执与本件同目录"),
    ("C7 角色桶与落笔分组的配对错位（zip 到自件桶名）",
     [(r'for \(glabel, gclaim\), key in zip\(groups, BKEY_PIN\):',
       "for (glabel, gclaim), key in zip(groups, BKEY_SELF):")],
     "本件按文件名现数"),
    ("C8 分组标签前缀循环的顺序倒置",
     [(r'    for t, key in zip\(toks, BKEY_PIN\):', "    for t, key in zip(toks, list(reversed(BKEY_PIN))):")],
     "顺序与本件桶序不一致"),
    ("C9 落笔闭合式的左边写成「未跟踪＝总计」",
     [(r'assert nu == sum\(c for _l, c in rows\) and nu \+ nm == tot', "assert nu == tot and nu + nm == tot")],
     "落笔闭合式不齐"),
    ("C10 shortstat 把删除列要求成必需（git 的单数／缺列陷阱）",
     [(r'deletions\?\\\(-\\\)\)\?', "deletions?\\(-\\))")],
     "shortstat 形状不认识"),
    ("C11 numstat 的增／删两列互换",
     [(r'ns = \(len\(rows\), sum\(int\(r\[0\]\) for r in rows\), sum\(int\(r\[1\]\) for r in rows\)\)',
       "ns = (len(rows), sum(int(r[1]) for r in rows), sum(int(r[0]) for r in rows))")],
     "≠ numstat 现算"),
    ("C12 回执正文逐字节比较错位一格（cmp 却仍过）",
     [(r'assert rb\(CURL\)\[off:\] == rb\(RAWR\)', "assert rb(CURL)[off + 1:] == rb(RAWR)")],
     "回执正文区与原始 stdout 不逐字节相同"),
    ("C13 顶格字段只取正文区（切段方向反了）",
     [(r'head_lines = rb\(CURL\)\[:int\(curl\["boff"\]\)\]\.decode',
       'head_lines = rb(CURL)[int(curl["boff"]):].decode')],
     "FIELDS 声明"),
    ("C14 缩进口径与顶格同形（缩进那条尺子丢了 \\s\\+ 前缀）",
     [(r'ind = \[l for l in head_lines if re\.match\(r"\^\\s\+\(\?:%s\)\\s\*:" % lab, l\)\]',
       'ind = [l for l in head_lines if re.match(r"^(%s)\\s*:" % lab, l)]')],
     "回执表头有缩进的标签行"),
    ("C15 尺子普查把自己排除在外（计数漏了本体）",
     [(r'group = sorted\(os\.path\.relpath\(p, REPO\) for p in alls if md5\(p, 12\) == fam\)',
       'group = sorted(os.path.relpath(p, REPO) for p in alls if md5(p, 12) == fam\n                and os.path.basename(p) != os.path.basename(RULER))')],
     "计数漏了本体"),
    ("C16 钉 locale 的那条不再钉环境（耦合检测空转）",
     [(r'text=True,\n                       env=dict\(os\.environ, LC_ALL="C"\)\)\.stdout\.strip\(\)',
       'text=True).stdout.strip()')],
     "两条 locale 现算同值"),
    ("C17 交换两条 locale 读数（参与闭合的那条变成钉了 locale 的那条）",
     [(r'^    return d, c$', "    return c, d")],
     "指纹四路不齐"),
    ("C18 局部仪器复跑把 stdout 断言改成读另一条流",
     [(r'assert not r\.stdout, "局部仪器复跑竟有 stdout', 'assert not r.stderr, "局部仪器复跑竟有 stdout')],
     "局部仪器复跑竟有 stdout"),
    ("C19 标本末行取成第一行",
     [(r'spec = \[l for l in rd\(ARCHK_PIN \+ "\.err"\)\.splitlines\(\) if l\.strip\(\)\]\[-1\]',
       'spec = [l for l in rd(ARCHK_PIN + ".err").splitlines() if l.strip()][0]')],
     "本轮实跑末行 ≠ 已归档标本末行"),
    ("C20 标本的 stdout 检查指向 .err 那一张面",
     [(r'assert not rb\(ARCHK_PIN \+ "\.out"\), "标本的 stdout 非 0',
       'assert not rb(ARCHK_PIN + ".err"), "标本的 stdout 非 0')],
     "标本的 stdout 非 0（"),
    ("C21 写入件标本句的四个读数取错序",
     [(r'nu, nm, tot, pin = grab\(r"落笔现数', 'nu, tot, nm, pin = grab(r"落笔现数')],
     "标本句里的闭合式自己不齐"),
    ("C22 控件面的源码常量指到 ENV_FAIL_STOP（两出处互证被拆开）",
     [(r'sub = int\(one\(r"\^SUB_TIMEOUT = \(\\d\+\)\$"', 'sub = int(one(r"^ENV_FAIL_STOP = (\\d+)$"')],
     "控件面 stdout 的上限"),
    ("C23 基线那一条从「CASES＋1＝子运行条数」的等式里掉出来",
     [(r'assert int\(decl\) \+ 1 == rows', "assert int(decl) == rows")],
     "那条等式的形状变了"),
    ("C24 卡死副本的身份 needle 指歪（按 cmdline 找到 0 条）",
     [(r'if "zctl_s%d_hdr" % PREVSLICE in c\]', 'if "zctl_s%d_zzz" % PREVSLICE in c]')],
     "现场件里副本不唯一"),
    ("C25 现场件的 ps 段按两列取（字段数变了）",
     [(r'ps = re\.findall\(r"\^\(\\d\+\)\\s\+\(\\d\+\)\\s\+\(\\d\+\)\\s\+S\\s\*\$"',
       'ps = re.findall(r"^(\\d+)\\s+(\\d+)\\s+S\\s*$"')],
     "取法形状变了"),
    ("C26 时长换算退回 divmod（口径 ⑧：余数是秒不是分）",
     [(r'h, m = hangs // 3600, \(hangs % 3600\) // 60', "h, m = divmod(hangs, 3600)")],
     "换算与落笔那份不是同一个数"),
    ("C27 承重面的对照面指错（开发日志换成登记表）",
     [(r'assert top\[0\] == DEVLOG, "承重最多的一面不是开发日志',
       'assert top[0] == REG, "承重最多的一面不是开发日志')],
     "承重最多的一面不是开发日志"),
    ("C27b 承重面唯一所指的门槛写错（并列容忍写成 2）",
     [(r'assert n_at_max == 1, "承重最多的一面有', 'assert n_at_max == 2, "承重最多的一面有')],
     "承重最多的一面有"),
    ("C28 模板字段被硬编码（取值表里剩一条没人用的读数）",
     [(r'== 父提交 blob @@PCOUNT@@', "== 父提交 blob 764")],
     "取值表里有模板没用到的读数"),
    ("C29 模板字段名多打一个 P（模板有未取值的字段）",
     [(r'其中 cmp 三联＝@@WRCMP@@', "其中 cmp 三联＝@@WRCMPP@@")],
     "模板有未取值的字段"),
    ("C30 V 的字面里出现重复键（后值静默覆盖前值）",
     [(r'^        "CMP": "/"\.join\(str\(x\) for x in cmp_rcs\), "NDEF": str\(n_defs\),$',
       '        "CMP": "0", "CMP": "/".join(str(x) for x in cmp_rcs), "NDEF": str(n_defs),')],
     "V 里有重复字面键"),
    ("C31 V 的字面里从字面量之外解包塞进一个键",
     [(r'^        "CMP": "/"\.join\(str\(x\) for x in cmp_rcs\), "NDEF": str\(n_defs\),$',
       '        **{"ZZ": "1"}, "CMP": "/".join(str(x) for x in cmp_rcs), "NDEF": str(n_defs),')],
     "非字符串常量键"),
    ("C32 键表自查的锚点键名写错（那条自查就成了空转）",
     [(r'getattr\(node\.targets\[0\], "id", None\) == "V"', 'getattr(node.targets[0], "id", None) == "VV"')],
     "取不到 V 的字面字典"),
    ("C33 键表自查改读上一片的生成器（审的是别人的源码）",
     [(r'selftree = ast\.parse\(open\(__file__, encoding="utf-8"\)\.read\(\)\)',
       'selftree = ast.parse(open(os.path.join(HERE, os.pardir, "2026-10-10", "am_s29_msg.py"),\n'
       '                     encoding="utf-8").read())')],
     "字面键表 ≠ 实得键集合"),
    ("C34 GPU 匹配式的正对照指到不含该解释器的 cmdline（假零）",
     [(r'ctl = mine\(\["/tmp/amvenv/bin/python fake\.py"\]\)', 'ctl = mine(["/usr/bin/python3 fake.py"])')],
     "GPU 匹配式的正对照没命中"),
    ("C35 把「卡上的我的进程」读成自己的 cmdline",
     [(r'real = mine\(\[open\("/proc/%s/cmdline" % p, "rb"\)\.read\(\)\.decode\("utf-8", "replace"\) for p in pids\]\)',
       'real = mine([open("/proc/self/cmdline", "rb").read().decode("utf-8", "replace")])')],
     "我的解释器在卡上留有进程"),
    ("C36 形制约束退回「本节就是上一节」",
     [(r'assert int\(secn\) == PREVSEC \+ 1', "assert int(secn) == PREVSEC")],
     "形制约束没落实"),
    ("C37 节内唯一性的 LESSON needle 写错（两处一起改，红仍落在那条断言）",
     [(r'"当前「", "对照的可满足性"', '"当前「", "对照的不通过性"'),
      (r'n\["当前「"\] == 1 and n\["对照的可满足性"\]', 'n["当前「"] == 1 and n["对照的不通过性"]')],
     "节内唯一性破了"),
    ("C38 上一节的切段取成本节（两出处互证指错面）",
     [(r'assert prev_bul == arch_bul', "assert len(top_bullets(body)[0]) == arch_bul")],
     "切段口径变了"),
    ("C39 「本节在 HEAD 里 0 次」写成 1 次",
     [(r'assert \(nh, ns\) == \(0, 1\)', "assert (nh, ns) == (1, 1)")],
     "次／暂存 blob 里"),
    ("C40 本件干跑控件面的文件名少一个 s（入库清单里查无此件）",
     [(r'os\.path\.basename\(p\) == "am_s%d_msg_controls\.py" % SLICE',
       'os.path.basename(p) == "am_s%d_msg_control.py" % SLICE')],
     "算进自件桶」成了假话"),
    ("C41 正对照逐行的前缀少两个空格（逐行数到 0 行）",
     [(r'l\.startswith\("  正对照："\)', 'l.startswith("正对照：")')],
     "该行形状变了"),
    ("C42 承载面闭合式里少了当日纪实",
     [(r'== sorted\(\[f\[0\] for f in faces\] \+ \[DAILY\]\)', "== sorted([f[0] for f in faces])")],
     "自查件承载行的五张面"),
    ("C43 修改面比对喂进去的是新增清单（调用处口径错位）",
     [(r'faces = faces_check\(modified\)', "faces = faces_check(added)")],
     "登记的那批"),
    ("C44 面行数闭合式两边相减写反",
     [(r'assert add == n_staged - n_head', "assert add == n_head - n_staged")],
     "行数差"),
    ("C45 暂存清单按位取行把状态字母当成路径（§26.30 的病形）",
     [(r'return rows, \[r\[-1\] for r in rows if r\[0\] == "A"\]', 'return rows, [r[0] for r in rows if r[0] == "A"]')],
     "角色口径不唯一，拒绝生成"),
    ("C46 探针上限对照读错列（真件上限与探针上限混用）",
     [(r'assert int\(pcap2\) == sub', "assert int(pcap) == sub")],
     "探针 stdout 的真件上限"),
    ("C47 回执表头的 dirty 旗标期望写成 2",
     [(r'assert curl\["dirty"\] == "0"', 'assert curl["dirty"] == "2"')],
     "dirty(src+tests)"),
]


def perturb(name, subs):
    """按源码正则改写；**每条锚点必须恰好命中 1 处**，否则这条扰动根本没改动源码＝空转。"""
    bad = text
    for pat, rep in subs:
        try:
            outs = re.findall(pat, bad, re.M)
        except re.error as e:
            raise AssertionError("%s：扰动锚点正则自己就非法（%s）：%s" % (name, e, pat))
        if len(outs) != 1:
            raise AssertionError("扰动锚点在源码里命中 %d 处（须恰好 1）：%s ⇒ 这条控制根本没改动源码" % (
                len(outs), pat))
        try:
            bad = re.sub(pat, lambda m: rep, bad, count=1, flags=re.M)
        except re.error as e:
            raise AssertionError("%s：扰动**替换文本**非法（%s）：%s" % (name, e, rep))
    if bad == text:
        raise AssertionError("%s：扰动前后源码逐字节相同 ⇒ 空转控制" % name)
    compile(bad, SRCNAME, "exec")
    return bad


def precheck():
    """纸面预检（坑⑦）：锚点恰好命中 1 处＋扰动后可编译＋期望句在源码里**唯一**；一条子运行都不起。"""
    bad_anchor, dup_expect, runtime_only = [], [], []
    for name, subs, expect in CASES:
        try:
            perturb(name, subs)
        except AssertionError as e:
            bad_anchor.append("%s：%s" % (name, e))
            continue
        key = expect.split("%")[0].strip() or expect
        hits = [i + 1 for i, l in enumerate(text.splitlines()) if key in l]
        if len(hits) == 0:
            runtime_only.append("%s：期望句「%s」在生成器源码里 0 处 ⇒ 位置判定不适用（只按抛出行判红）" % (name, expect))
        elif len(hits) > 1:
            dup_expect.append("%s：期望句「%s」在生成器源码里 %d 处（%s）⇒ 不唯一，±3 判定会指错闸门" % (
                name, expect, len(hits), hits))
    print("纸面预检：CASES 声明＝%d 条｜锚点／编译不合格＝%d 条｜期望句不唯一＝%d 条｜期望句只在运行时（位置判定不适用）＝%d 条" % (
        len(CASES), len(bad_anchor), len(dup_expect), len(runtime_only)))
    for r in bad_anchor + dup_expect + runtime_only:
        print("  · " + r)
    return not bad_anchor and not dup_expect


def raised(msg):
    """只取**最后一条**异常行：traceback 会把源码行原样回显，而闸门句本来就写在源码里。"""
    hits = re.findall(r"^(?:\w+Error|AssertionError)[^\n]*", msg, re.M)
    return hits[-1] if hits else "（没取到异常行）"


def trace_line(msg):
    """traceback 里**最后一个** `File ".../zctl_s30_msg_*.py", line N`＝真正抛出的那一行。"""
    hits = re.findall(r'File "[^"]*%s[^"]*\.py", line (\d+)' % PREFIX, msg)
    return int(hits[-1]) if hits else None


def gate_line(expect):
    key = expect.split("%")[0].strip() or expect
    return [i + 1 for i, l in enumerate(text.splitlines()) if key in l]


def run_one(bad):
    """在同一目录落一份扰动副本、跑它、拿 rc＋合并输出；超时＝环境故障（坑⑨），按**进程组**杀。"""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8",
                                     dir=HERE, prefix=PREFIX) as fh:
        fh.write(bad)
        tmp = fh.name
    p = subprocess.Popen([sys.executable, tmp], cwd=REPO, env=ENV,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         start_new_session=True)
    try:
        out, err = p.communicate(timeout=SUB_TIMEOUT)
        return p.returncode, (err or "") + (out or ""), None
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        out, err = p.communicate()
        return "TIMEOUT", (err or "") + (out or ""), tmp
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def residue():
    return [f for f in os.listdir(HERE) if f.startswith(PREFIX)]


def main():
    assert sys.executable.startswith("/tmp/amvenv"), "解释器不是 /tmp/amvenv 而是 %s" % sys.executable
    assert "LC_ALL" not in os.environ, "本件环境里有 LC_ALL ⇒ 生成器第一条就红（坑④），那不是闸门红"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "本件没带空串 CVD ⇒ 副本也拿不到（钉住口径失真）"
    if "--precheck" in sys.argv:
        sys.exit(0 if precheck() else 3)
    if not precheck():
        print("纸面预检不合格，不起子进程（先把锚点改对再花钱）")
        sys.exit(3)

    # 基线＝逐字节相同的副本必须**通过**并真的走到打印分支。没有这一步，上面所有红都可能只是环境错（坑①④）。
    base_rc, base_msg, _ = run_one(text)
    assert base_rc == 0, ("未扰动副本就没通过（rc=%s）⇒ 本件的正对照全部无效，先修仪器。stderr 尾巴：\n%s"
                          % (base_rc, "\n".join(base_msg.splitlines()[-8:])))
    assert "---- 现取核对 ----" in base_msg, "未扰动副本没走到打印分支 ⇒ 副本跑的不是同一条路径"
    assert not residue(), "扰动副本没被清掉：" + "、".join(residue())

    rows, allred, envfail = [], True, 0
    for name, subs, expect in CASES:
        rc, msg, _ = run_one(perturb(name, subs))
        if rc == "TIMEOUT":
            envfail += 1
            print("  ！%s 子运行超过 %ds ⇒ 判环境故障，不判红也不判过" % (name, SUB_TIMEOUT))
            if envfail >= ENV_FAIL_STOP:
                print("CONTROL_VERDICT = ENV FAILURE（累计 %d 条超时，本件没资格判闸门）" % envfail)
                sys.exit(4)
            continue
        gl = gate_line(expect)
        tl = trace_line(msg)
        err = raised(msg)
        hit = expect in err
        if len(gl) == 1 and tl is not None:
            near = gl[0] - 3 <= tl <= gl[0] + 3
            why = "句在源码 %d 行／raise 在 %d 行%s" % (gl[0], tl, "（同一条闸门）" if near else "（不是同一条！）")
        else:
            near = True
            why = "句在源码 %s 处／raise 在 %s 行（位置判定不适用，只按抛出行判红）" % (len(gl), tl)
        ok = rc != 0 and hit and near
        allred = allred and ok
        if not ok:
            why += "｜实得抛出行：" + err[:120]
        rows.append((name, rc, ok, expect, why))

    # 对**本件自己**的闸门：锚点不存在时 perturb 必须拒绝。不验这条，下一轮克隆后"锚点已移位"的扰动
    # 会静默地没改动任何源码（＝又跑了一遍真件），红就变成假红。
    try:
        perturb("C48", [(r"^    nvzz = 1$", "pass")])
        rows.append(("C48 锚点普查（本件自己的闸门）", 0, False, "扰动锚点在源码里命中 0 处", ""))
        allred = False
    except AssertionError as e:
        okk = "扰动锚点在源码里命中 0 处" in str(e)
        rows.append(("C48 锚点普查（本件自己的闸门）", 3 if okk else 0, okk,
                     "扰动锚点在源码里命中 0 处", "本件自己的异常"))
        allred = allred and okk
    assert not residue(), "跑完后目录里有扰动副本残留：" + "、".join(residue())

    # 本轮**造不出红路**的闸门，逐条登记原因（不假装它们被验过）。
    UNCOVERED = [
        ("片号三处互证里的「本轮回执已经存在」（`SLICE not in LOGS`）",
         "红路需要档案目录里真有一张 am_s7_verify30.log＝本片的回执；按形制它只能在本片落地之后存在，"
         "为造红预写一张假回执正是这条闸门要拦的事。它会在分片 31 变成活闸（那时 30 号已入库）。"),
        ("回执表头里日志节号的**集合相等**（`set(rsecs) == {PREVSEC}`）",
         "现档那两处引用同值（都是 §26.31），把 findall 截断成第一条仍 rc=0＝改了源码没改行为（坑⑤）；"
         "红路需要一张「表头里引用了两个不同节号」的回执，那要改已入库的档案面。"),
        ("把 V 的某个槽位填成**另一个已经过闸的读数**（例如 PCOUNT 填 parent40 前缀）",
         "本件的单位断言审的是**回执里那个字段本身**（\\d+），不是「哪一格填了哪个字段」⇒ 填错槽位时"
         "两条数都合法，副本 rc=0。这条是本轮纠出 PCOUNT 之后仍然留下的形状债：模板里每一格的"
         "**语义**没有机器可读的第二出处，只能靠人读＋分片 31 的自查比对原文。"),
        ("`numstat 里有 - 行（二进制／rename）` 与 `kinds == [A, M]`",
         "两条都需要往暂存集里放一张二进制面或一次删除，与本轮「只归档文本面、只净增」直接冲突；"
         "两列求和的机制已由 C11 覆盖，状态字母按位取行已由 C45 覆盖。"),
        ("面行数的「只允许净增」（`dele == 0`）",
         "红路要往四张档案面里真删一行；同上一条冲突。它数的是 `git show` 两个 blob 的行数差，"
         "该口径本身已由 C44 的相减反向打过红。"),
        ("副本 etimes 与父控件面的长短关系（`gap > 0`）",
         "改 needle 会先撞「按身份找到 0／N 条」（C24）⇒ 走不到这句；把它改写成 `hangs - min(...)` 是"
         "**放宽**而不是坏输入，按规矩不放进 CASES。"),
        ("「当日纪实必须在新增清单」（`len(daily_in_a) == 1`）",
         "C40 用的是同一条闸门的姊妹句（干跑控件面）；要这条单独红需要造一张**同名**的第二当日纪实，"
         "那是改档案而不是改仪器。"),
        ("角色桶空转（`assert not empty`，「那个角色的命名口径已变」）",
         "桶形改窄时，那些件必然先落进 0 个桶而撞上逐件断言（C3／C5 落的那条），轮不到「空桶」这一句；"
         "要它单独红得让一整角色的件还在清单里却不被任何桶认领＝改档案面命名。C5 已把同一处插值的"
         "错位打到红，只是红在前一条闸门。"),
    ]

    print("基线（未扰动副本）rc=%s 且走到打印分支＝True（所有红的对照面）" % base_rc)
    print("解释器＝%s｜被扰动的件＝%s｜每条子运行上限＝%ds｜环境故障累计上限＝%d 条" % (
        sys.executable, SRC, SUB_TIMEOUT, ENV_FAIL_STOP))
    print("CASES 声明＝%d 条｜本件自己的锚点普查＝1 条｜超时（环境故障，不判红也不判过）＝%d 条｜因超时没跑到的＝%d 条" % (
        len(CASES), envfail, envfail))
    print("扰动条数＝%d｜每条判据＝rc≠0 且**抛出的那一行**含指定闸门句 且 raise 行号落在该句源码位置的 ±3 行窗口" % (
        len(rows) + envfail))
    for name, rc, ok, expect, why in rows:
        print(("  " if ok else "× ") + name + "｜rc=" + str(rc) + "｜期望句「" + expect + "」"
               + ("命中" if ok else "**未命中**") + ("｜" + why if why else ""))
    print("本轮造不出红路的闸门＝%d 条（逐条附原因，不算被验）：" % len(UNCOVERED))
    for gate, why in UNCOVERED:
        print("  · " + gate + "：" + why)
    print("CONTROL_VERDICT =", "ALL CONTROLS CAN GO RED" if allred else "PROBLEM（有闸门空转）")
    sys.exit(0 if allred else 3)


if __name__ == "__main__":
    main()
