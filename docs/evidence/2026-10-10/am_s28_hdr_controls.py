#!/usr/bin/env python3
"""表头生成器 am_s28_hdr.py 的**正对照**：每条扰动都必须让它拒绝，否则那条闸门是空转。

扰动改的是**生成器的源码副本**（正则级改写，个别条改的是**喂给它的环境**），跑的是同一件（同一条
`python <file>`）⇒ 被验的就是落地那把仪器本身，不是另写的平行实现。
判据＝rc≠0 **且**抛出的那一行含指定闸门句 **且** raise 行号落在该句源码位置的 ±3 行窗口内。

本轮（分片 28）**动过的锚句**只有两类，它们必须各自有红路，而不只是"当前能跑通"：
⑴ 取子条的**形制**：上一轮 §26.29 的子条写在行首 `**(x)**`，本轮 §26.30 写在行首 `- **标签**`
   ⇒ 正则与切段口径随形制改过。C4＝把正则退回字母形制（在 §26.30 上数出 **0** 条子条＝取空），
   C5＝切段边界一律取到节尾（每条子条的段都覆盖后面所有子条＝歧义），C11＝档案里出现同名标签
   （新加的"标签必须互异"闸门）。
⑵ LESSONLET 的**锚句**（换成 §26.30 里本轮入档的那句「判据的可通过性」）：C8＝查无、C9＝歧义。
其余 C1–C3、C12–C26 复验从上一片克隆过来的闸门（回执链五件、尺子全树 glob、复跑逐字节、
入库数两源、片号／连号／节奏互证、GPU 归还、字段粘连、占位符残留、BODY_OFFSET 不动点）。

五条**仪器自己的**坑（①②④是继承，③⑤本轮新增，都实测到、不是推想）：
① 判据不能只看 rc≠0。扰动副本必须落在**同一目录**（生成器按自身位置反推 REPO），并加一条**未扰动
   基线 rc=0 且进预检分支**（C0），否则所有红都可能是环境错。
② **期望闸门句里不许写死节号**：bullet_for 那句的节号本来就是 `%s` 现取的 ⇒ 期望句一律取源码里的
   **字面片段**（如「里匹配到」），既不写死节号，又保证该片段在本件源码里**只出现一次**（否则
   行号定位不适用，见下面打印的"位置判定"列）。
③ **形制变化这一轮真的发生了**：上一片的生成器拿 `**(x)**` 取子条，落在 §26.30 上会**整条取空**。
   ⇒ 改口径不等于改对了：新口径的三种坏输入（取空／歧义／同名标签）必须逐条造红。
④ 判"红"必须按**抛出的那一行**，不能按整段 stderr：Python 3.11+ 的 traceback 把**源码行原样回显**，
   而闸门句就写在源码里 ⇒ 崩在别处也能让"整段含那句"成立＝假命中（本片的提交信息控件面实测到，
   口径继承到这里）。
⑤ `raise SystemExit("…")` 打到 stderr 的消息**不带异常名前缀**（实测：C24 那轮 stderr 最后一行就是
   「BODY_OFFSET 未收敛 ⇒ 拒绝归档」，没有任何 `Error:` 行）⇒ raised() 在取不到 Error 行时回落到
   stderr 最后一个非空行，并把这种回落如实打印出来。
⑥ **C24 的第一版是个假红候选，实测到的是"绿"**：原本把 `cand = len(render().encode())` 改成 `+1`
   想让不动点迭代打红，结果它在第 3 轮就停在一个**错位但自洽**的偏移上（BOFF＝真实表头长度＋1）
   并 rc=0——因为预检模式下**没有任何闸门读 BODY_OFFSET**，它只在写盘分支被 `back[off:] == body`
   与 `int(vals["BOFF"]) == off == len(head_txt)` 两条查住。⇒ 这条改判成"把迭代轮数从 12 砍到 1"，
   让那条 `raise SystemExit` 真的发生；而"预检模式对偏移量盲目"这件事本身登记进 UNCOVERED，
   不假装 C24 验过写盘段的三件套。
"""
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
SRC = os.path.join(HERE, "am_s28_hdr.py")
PREFIX = "zctl_s28_hdr_"
text = open(SRC, encoding="utf-8").read()
DEV_STAGED = subprocess.run(["git", "-c", "core.quotePath=false", "show", ":docs/开发日志.md"],
                            cwd=REPO, capture_output=True, text=True, check=True).stdout
SECN = re.findall(r"^### 26\.(\d+)[^\n]*$", DEV_STAGED, re.M)[-1]

ENV = dict(os.environ)
ENV["CUDA_VISIBLE_DEVICES"] = ""

LESSON_ANCHOR = r'^LESSONLET, _LSEG = bullet_for\(r"判据的可通过性", "判据的可通过性这条本轮口径"\)$'
SEC_ANCHOR = r'^SECLET, _SECSEG = bullet_for\(re\.escape\(PREVLOG\), "上一片回执名"\)$'

# 每条＝(名字, [(源码锚点正则, 替换后的整行)], 期望闸门句的字面片段[, 覆盖环境])
CASES = [
    ("C1 回执链口径里删掉表头生成器（五件变四件）",
     [r'^CHAIN_RE = re\.compile.*$',
      'CHAIN_RE = re.compile(r"^am_s7_verify(\\d+)\\.(py|log)$|^_raw_verify(\\d+)\\.(out|err)$")'],
     "回执链在 HEAD 里只"),
    ("C2 链计数归一化失效（int 与 regex 分组的字符串混比）",
     [r"^    n = str\(n\).*$", "    pass"],
     "回执链在 HEAD 里只"),
    ("C3 尺子 glob 指到不存在的目录（本件没被自己数到）",
     [r'glob\.glob\(os\.path\.join\(REPO, "docs/evidence/\*/am_s7_verify\*\.py"\)\)',
      'glob.glob(os.path.join(REPO, "docs/evidence/NO_SUCH_DIR/am_s7_verify*.py"))'],
     "本件的尺子副本没被自己的"),
    # ---- 本轮改动①：子条形制（`**(x)**` → `- **标签**`）的三条红路：取空／歧义／同名标签 ----
    ("C4 取子条的正则退回上一轮的字母形制（落在 §26.%s 上数出 0 条＝整条取空）" % SECN,
     [r'^BULLETS = \[m\.group\(1\) for m in re\.finditer.*$',
      'BULLETS = [m.group(1) for m in re.finditer(r"^\\*\\*\\((\\w)\\)[^\\n]*$", SEC_BODY, re.M)]'],
     "正文里取不到行首"),
    ("C5 切段边界一律取到节尾（每条子条的段都吞掉后面的子条＝歧义）",
     [r"^        end = \(SEC_BODY\.index.*$", "        end = len(SEC_BODY)"],
     "里匹配到"),
    ("C6 上一片回执的引用查无（正则口径又变了）",
     [SEC_ANCHOR, 'SECLET, _SECSEG = bullet_for("am_s7_verify99.log", "上一片回执名")'],
     "里匹配到"),
    ("C7 上一片回执的引用歧义（空模式：每条子条必命中）",
     [SEC_ANCHOR, 'SECLET, _SECSEG = bullet_for("", "上一片回执名")'],
     "里匹配到"),
    # ---- 本轮改动②：LESSONLET 锚句 ----
    ("C8 本轮新锚句在日志里查无（取空）",
     [LESSON_ANCHOR,
      'LESSONLET, _LSEG = bullet_for(r"这句本轮日志里没有", "判据的可通过性这条本轮口径")'],
     "里匹配到"),
    ("C9 本轮新锚句歧义（空模式：每条子条必命中）",
     [LESSON_ANCHOR, 'LESSONLET, _LSEG = bullet_for("", "判据的可通过性这条本轮口径")'],
     "里匹配到"),
    ("C10 钉住口径的取句指到不存在的原文（空串 CVD 那句没了出处）",
     [r'^GPUQUOT_M = re\.search.*$',
      'GPUQUOT_M = re.search(r"当前「这句日志里没有」", SEC_BODY)'],
     "这种钉住口径原文"),
    ("C11 档案里出现同名子条标签（新加的「标签互异」闸门）",
     [r'^DEV = open\(DEVLOG, encoding="utf-8"\)\.read\(\)$',
      'DEV = open(DEVLOG, encoding="utf-8").read() + "\\n- **树与钉住**\\n"'],
     "有同名子条标签"),
    # ---- 克隆过来的闸门逐条复验 ----
    ("C12 复跑核验件的 rc 被读成 1（复跑与正文不同一次运行）",
     [r'^rc_txt = str\(rp\.returncode\)$', 'rc_txt = "1"'],
     "复跑核验件 rc"),
    ("C13 复跑 stdout 与归档正文差一个字节（仪器不可复现）",
     [r"^REPRO = rp\.stdout == body$", "REPRO = rp.stdout == body + 'x'"],
     "与归档正文不同"),
    ("C14 正文判读行被改坏（回执不该被当作全过归档）",
     [r'^assert "LANDING_VERDICT = ALL CHECKS PASS" in body',
      'assert "LANDING_VERDICT = PROBLEM" in body'],
     "不该被当作回执归档"),
    ("C15 入库清单两个出处互证失效（body 侧 +1）",
     [r"^assert \(NA_BODY, NM_BODY\) == \(len\(added\), len\(mods\)\), \($",
      "assert (NA_BODY, NM_BODY) == (len(added) + 1, len(mods)), ("],
     "核验件印出的入库数"),
    ("C16 片号写死成上一片（与脚本文件名的互证失效＝机械克隆张冠李戴）",
     [r'^SLICE = re\.search\(r"S7 分片 \(\\d\+\)", SUBJ\)\.group\(1\)$', 'SLICE = "27"'],
     "本件所验尺子"),
    ("C17 上一片回执取成隔片（连号闸门：有一片没落就该拒绝）",
     [r"^PREVSLICE = max\(n for n in RCPTS if n != int\(SLICE\)\)$",
      "PREVSLICE = max(n for n in RCPTS if n != int(SLICE)) - 1"],
     "不连号"),
    ("C18 本片的链其实已在 HEAD（CMD 行那句「留给分片 29」成了假话）",
     [r"^OWNHIT = len\(chain_ids\(SLICE, prev_rows\)\)$",
      "OWNHIT = len(chain_ids(SLICE, prev_rows)) + 1"],
     "已在 HEAD"),
    ("C19 GPU 检测器把别的 PID 算成我的（只验这条闸门本身能红，不代表现读）",
     [r"^nv_mine = \[p for p in nv$", 'nv_mine = [p for p in (nv or ["1"])',
      r'^           if "/tmp/amvenv" in open\(f"/proc/\{p\}/cmdline", "rb"\)\.read\(\)\.decode\("utf-8", "replace"\)\]$',
      '           if True]'],
     "冲突，拒绝归档"),
    ("C20 解释器口径（CMD 行描述失真的守卫；正对照读的是自己的 cmdline）",
     [r'^assert sys\.executable\.startswith\("/tmp/amvenv"\), f"解释器不是',
      'assert sys.executable.startswith("/tmp/no_such_env"), f"解释器不是'],
     "解释器不是"),
    ("C21 跑生成器时没带空串 CVD（环境侧坏输入，与 ENV 行的声称不符）",
     [], "生成器自身未带空串", {"CUDA_VISIBLE_DEVICES": "1"}),
    ("C22 模板里有没取值的字段（占位符替换没走完）",
     [r"^ENV     : CUDA_VISIBLE_DEVICES=.*$", "ENV     : CUDA_VISIBLE_DEVICES=@@ZZ_NO_FIELD@@"],
     "模板有未取值的字段"),
    ("C23 有一个占位符的替换被跳过（表头残留 PLACEHOLDER）",
     [r'^        h = h\.replace\("@@" \+ k \+ "@@", v\)$',
      '        h = h if k == "NOW" else h.replace("@@" + k + "@@", v)'],
     "表头残留 PLACEHOLDER"),
    ("C24 BODY_OFFSET 取不到不动点（迭代轮数砍到 1 轮，走那条 SystemExit）",
     [r"^for _ in range\(12\):$", "for _ in range(1):"],
     "未收敛 ⇒ 拒绝归档"),
    ("C25 字段行粘连：TREE 字段被缩进两格（它就不再是字段行）",
     [r"^TREE    : head=.*$",
      "  TREE    : head=@@HEAD@@ parent=@@PARENT@@ dirty(src+tests)=@@DIRTY@@"],
     "缩进处出现"),
]


def perturb(name, subs, src=None):
    """把扰动落到源码上；**锚点必须在源码里唯一命中**，否则这条扰动不可信（C26 验的就是这句）。"""
    bad = text if src is None else src
    for i in range(0, len(subs), 2):
        pat, rep = subs[i], subs[i + 1]
        hits = re.findall(pat, bad, re.M)
        assert len(hits) == 1, "%s：扰动锚点在源码里命中 %d 处 ⇒ 扰动本身不可信" % (name, len(hits))
        bad = re.sub(pat, lambda m: rep, bad, count=1, flags=re.M)
    if subs:
        assert bad != (text if src is None else src), name + "：扰动没改变源码 ⇒ 空转"
    return bad


def trace_line(msg):
    """取**最后一帧** `File "...", line N`（副本是临时名，所以只按行号取；插行的扰动会让行号漂移，
    故位置判定给 ±3 的窗口而不是严格相等）。"""
    cand = re.findall(r'File "[^"]*/(am_s28_hdr\.py|zctl_[\w.-]+)", line (\d+)', msg)
    return int(cand[-1][1]) if cand else None


def raised(msg):
    """把**实际抛出的那一行**取回来。优先取 `XxxError:`／`AssertionError:` 行；取不到时回落到 stderr
    最后一个非空行——`raise SystemExit("…")` 的消息不带异常名前缀（C24 实测到），回落时在名字里标注。"""
    hits = re.findall(r"^(?:\w+Error|AssertionError)[^\n]*", msg, re.M)
    if hits:
        return hits[-1], False
    lines = [l for l in msg.splitlines() if l.strip()]
    return (lines[-1] if lines else "（stderr 空）"), True


def gate_line(expect):
    """期望句在本件源码里的行号；出现 0 次或多次都不做位置判定。"""
    key = expect.split("%")[0].strip() or expect
    return [i + 1 for i, l in enumerate(text.splitlines()) if key in l]


def run_one(bad, env=None):
    """在同一目录落一份扰动副本、跑它、拿 rc＋stderr／stdout，然后删掉副本。"""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8",
                                     dir=HERE, prefix=PREFIX) as fh:
        fh.write(bad)
        tmp = fh.name
    run_env = dict(ENV) if env is None else dict(ENV, **env)
    try:
        r = subprocess.run([sys.executable, tmp], cwd=REPO, env=run_env,
                           capture_output=True, text=True)
        return r.returncode, (r.stderr or ""), (r.stdout or "")
    finally:
        os.unlink(tmp)


def residue():
    return [f for f in os.listdir(HERE) if f.startswith(PREFIX)]


# 基线＝逐字节相同的副本必须**通过**并真的走到预检分支。没有这一步，上面所有红都可能是环境错（坑①）。
BASE_RC, BASE_ERR, BASE_OUT = run_one(text)
assert BASE_RC == 0, ("未扰动副本就没通过（rc=%s）⇒ 本件的正对照全部无效，先修仪器。stderr 尾巴：\n%s"
                      % (BASE_RC, "\n".join(BASE_ERR.splitlines()[-6:])))
assert "（预检模式：未写盘" in BASE_OUT, "未扰动副本没进预检分支 ⇒ 副本跑的不是同一条路径"
assert not residue(), "扰动副本没被清掉：" + "、".join(residue())

rows, allred = [], True
for item in CASES:
    name, subs, expect = item[0], item[1], item[2]
    env = item[3] if len(item) > 3 else None
    rc, err, out = run_one(perturb(name, subs), env)
    gl, tl = gate_line(expect), trace_line(err)
    line, fell_back = raised(err)
    text_hit = expect in line
    if len(gl) == 1 and tl is not None:
        near = gl[0] - 3 <= tl <= gl[0] + 3
        why = "句在源码 %d 行／raise 在 %d 行%s" % (gl[0], tl, "（同一条闸门）" if near else "（不是同一条！）")
    else:
        near = True
        why = "句在源码 %s 处／raise 在 %s 行（位置判定不适用，只按抛出行判红）" % (len(gl), tl)
    if fell_back:
        why += "｜按 stderr 末行判（SystemExit 不带异常名前缀）"
    ok = rc != 0 and text_hit and near
    allred = allred and ok
    rows.append((name, rc, ok, expect, why + "｜实抛＝" + line[:88]))

# C26＝对**本件自己**的闸门：锚点不存在时 perturb 必须拒绝。不验这条，下一轮克隆后"锚点已移位"的
# 扰动会静默地没改动任何源码（＝跑了一遍真件），红就变成假红。
try:
    perturb("C26", [r'^LESSONLET = bullet_for\("这条锚点在本件源码里不存在"\)$', "pass"])
    rows.append(("C26 锚点普查（本件自己的闸门）", 0, False, "扰动锚点在源码里命中 0 处", ""))
    allred = False
except AssertionError as e:
    hit = "扰动锚点在源码里命中 0 处" in str(e)
    rows.append(("C26 锚点普查（本件自己的闸门）", 3 if hit else 0, hit,
                 "扰动锚点在源码里命中 0 处", "本件自己的异常"))
    allred = allred and hit
    assert not residue(), "C26 没落副本，但目录里有残留"

# 本轮**造不出红路**的闸门，逐条登记原因（不假装它们被验过）。
UNCOVERED = [
    ("全树 glob 口径（尺子份数 9 份 vs 本目录 3 份）",
     "把 glob 的根从 REPO 换成 D（本目录口径）**不会**让任何断言红：SCRIPT_REL 就在本目录，"
     "自数闸门照样过，只是把「一把尺子量多片」印成 3 份＝**读数错而不是判读红**。"
     "所以这条口径靠的是表头**同印两个读数**（@@NRULE@@／@@NDIR@@）让人能查，"
     "它配套的断言红路由 C3 覆盖（glob 指错目录时本件被自己数不到）。"),
    ("写盘段的三件套（正文区逐字节复核／全角 at 号／cmp 两侧 +1 须红）＋预检模式下的 BODY_OFFSET 数值",
     "本件的副本一律跑**预检分支**（不写盘 ⇒ 不会污染档案），所以那三条够不到；"
     "其中「两侧偏移各 +1 必须变红」已由真件写盘那一次实跑证过（_raw_s28_hdr_write.out 印 cmp rc＝0/1/1），"
     "另两条只有绿路。**顺带登记一个盲区**（坑⑥）：预检分支里 BODY_OFFSET 的数值没人查，"
     "所以偏移量错了在本件跑不到的那条路上是**静默**的——它只在写盘分支被两条断言查住。"),
    ("「我的解释器在 GPU 上留有进程」这条的真实现读数",
     "C19 是把**别的 PID 冒充成我的**来验闸门本身能红，不等于本轮真的没占卡；真实读数是表头 GPU 行的 "
     "0/3（生成回执时现取，且 /proc/PID/cmdline 口径与全量回归件相同）。"),
]

print("基线（未扰动副本）rc=%s 且进预检分支＝True（所有红的对照面）" % BASE_RC)
print("解释器＝%s｜期望句里的节号 §26.%s 现取自暂存日志最后一个标题（句子里不含节号，见坑②）"
      % (sys.executable, SECN))
print("扰动条数＝%d｜每条判据＝rc≠0 且**抛出的那一行**含指定闸门句 且 raise 行号落在该句源码位置（±3 行）" % len(rows))
for name, rc, ok, expect, why in rows:
    print(("  " if ok else "× ") + name
          + "｜rc=" + str(rc) + "｜期望句「" + expect + "」"
          + ("命中" if ok else "**未命中**") + ("｜" + why if why else ""))
print("本轮造不出红路的闸门＝%d 条（逐条附原因，不算被验）：" % len(UNCOVERED))
for gate, why in UNCOVERED:
    print("  · " + gate + "：" + why)
print("CONTROL_VERDICT =", "ALL CONTROLS CAN GO RED" if allred else "PROBLEM（有闸门空转）")
sys.exit(0 if allred else 3)
