#!/usr/bin/env python3
r"""提交信息生成器 `am_s31_msg.py` 的**正对照**（全离线）：每条扰动都要让它拒绝在**自己那条闸门**上。

扰动改的是**生成器的源码副本**（正则级改写），跑的是同一件（同一条 `python <file>`）⇒ 被验的就是要落地的
那把仪器本身，不是另写的平行实现。判据＝rc≠0 **且**抛出的那一行含指定的闸门句 **且** raise 的行号落在该句
在本副本里的位置（±3 行）——只看 rc 会把「环境错」记成「闸门对」。

本轮（分片 31）新增闸门的红路，逐条都要有：
① 「单值取数」族的**调用面**：片号／回执 glob／尺子 glob／上一片回执唯一性（C1–C6）；
② 角色桶与 §26.33 落笔分组的配对：桶序、zip 对象、分组项形状、标签前缀、闭合式左边、清单切分（C7–C14）；
③ 本轮新加的 `ncode == 0` 闸（模板里那句「现数 0 条」原先只是读数、不是断言＝作用域一挪就成假话）（C15）；
④ 本轮新加的「顶格字段」全日志普查（把 16 改回 99＝消息里承诺的那条坏输入；尺子失效）（C27–C28）；
⑤ NVML 坏掉之后新写的三条：dpkg 状态放宽到残留包（两个版本）、架构过滤写成表里没有的值（零个版本）、
   版本号不剥 release 后缀、realpath→abspath、指到残留文件、glob 出处为空、内核模块读数被换成用户态值
   （C31–C38）；
⑥ 替代红线本身：整词匹配退回子串、合成正／负对照、不排除自身与父进程（C39–C42）；
⑦ 干跑控件面读数的**两侧闭合**（表内 G 行／标签数、基线行／声明、绿路＝表内＋基线、三面判读行一致）（C43–C46）。

十条**仪器自己的**坑（继承＋本轮新增）：
① 判据不能只看 rc≠0；副本也不能写在 /tmp——生成器用自身位置反推 REPO ⇒ 副本必须落在**同一目录**，并加一条
   **未扰动基线必须 rc=0 且逐字节命中基线标记句**（见 SELF_BASE_MARK），否则所有红都不作数。
② **期望闸门句里不许写死节号／片号／读数**：那些是档案的读数，写死就随片腐烂。本件所有期望句都取自闸门自己
   的固定短语。
③ 按**抛出的那一行**判红，不按整段 stderr：3.11+ 的 traceback 把源码行原样回显，闸门句本来就写在源码里。
   raise 的行号取**最后一个**匹配本副本名的帧——第一版取第一条，量到的是调用链而不是抛出点。
③′ 本轮把「该句在源码里的位置」从**单点**改成**多点取任一**：`one()`／`grab()` 的报错句是通用文本（在源码里
   两处），按第一版「须恰好 1 处才做位置判定」会把这类真红判成不合格，于是这条要么放弃位置判定（退回只看 rc＝
   坑①的形状）要么放宽判据。**放宽方向选在判定器而不是数据上**：允许 raise 落在该句的任一出现处 ±3，仍是
   「红在别的闸门」的反例（别的闸门不在这些行上）。
④ **必须用 /tmp/amvenv/bin/python 跑本件**，且环境里不许有 `LC_ALL`（生成器第一条就断言它缺席，并把不钉 locale
   的那条读数用于四路闭合）；不满足就跑第一步会全红，那不是闸门红。
⑤ **改了源码 ≠ 改了行为**：一条扰动若让副本 rc=0，它要么反过来切，要么如实进 `UNCOVERED`（本轮见文件末尾）。
   判据必须允许「这条造不出红」被登记，否则下一轮会把它当成已验过的闸门。
⑥ **扰动落在 dict 的字面键／正则上时，取值处要同批改**；needle 选成在源码里**多处**出现的串时，红可能落在别的
   那条闸门上 ⇒ 本件对每条都验「抛出行 ±3」（同 ③′）。
⑦ 跑本件之前先做**纸面预检**（锚点在源码里恰好命中 1 处、扰动后仍能 `compile()`、期望句在源码里可定位），再加
   两条**本件自己**的对照：本文件声明的两个自述常量必须各在生成器源码里命中 1 行（正）、写歪的锚点句必须 0 命中
   （负）。这三条要靠真跑几十个子进程才暴露，纸面预检把「锚点 0 命中」这种空转在花钱之前挑出来（`--precheck`）。
⑧ **本件的基线依赖一个时序**：生成器有一条「本件的干跑控件面必须在入库清单里」的断言 ⇒ 跑本件**之前**要先
   `git add` 本件自己（它的原始件此刻不存在，生成器不读它们，只按文件名计数）。这条不是循环：循环只发生在
   「生成器读自己控件的 stdout」那种引用上，所以本件的读数由**分片 32** 引用，本片不引用。
⑨ 每条子运行自带 `SUB_TIMEOUT`，超时按**进程组** SIGKILL；超时判成**环境故障**（`CONTROL_VERDICT = ENV
   FAILURE…`＋exit 4），既不判红也不判过——本件全离线，超时是机器不是闸门。
⑩ **绿路必须真绿、不许是 Python 恒真式**：`assert X is not False or True` 这种形状永远通过（第一版就写了这么
   一条，靠「跑完之后现场必须无残留」把它变成实判）。本件的绿路全部**跑命令**（cmp／dpkg／git／/proc／复跑
   真仪器），每条末尾再断言本件自己的现场闸（副本文件已删、档案面未被改）。
⑪ **有一条扰动会去碰另一件仪器的写盘腿**（C110：克隆本归档的写入件并带 `--write` 跑它）。这条是本轮唯一会执行
   **别的仪器**的扰动，所以它前后都比对「工作树对暂存面零漂移」（`git diff --name-only` 为空＝那五张档案面逐字节
   未动）——比对不为空就当场红，不把「没写」当假设。**实跑得知的次序**：克隆副本先撞上写入件自己的落笔闭合式
   （现场 3＋75＝78 vs §26.33 写死 67）而拒绝，**走不到**那句「有别的仪器正在跑」的守卫 ⇒ 本条登记的期望句改成它
   实际打红的那条闸门，守卫那句进 UNCOVERED（第一版把它写成守卫＝声称了一件没发生过的事）。
⑫ **基线的「命中行数」判据要按生成器的**打印形制**定，不是按「标记句理应唯一」的直觉定**：第一版写「标记句在副本
   stdout 里须恰好命中 1 行」，实跑就红在 2 行——生成器的信息正文里有一条新学 bullet 就是那一句（逐字节相同），
   「诚实登记 ⑤」又把同一句作为引用嵌在长句里（带前后文），**按构造出现两次**。判据改成「至少一行逐字节相同」，
   歧义由另两条尺子承担（本文件里那句须 1 行＋生成器报的字节数与本件同值）。这是「判据按构造红不了」的第四次，
   形状不同：前几次是等式两边不可同时成立，这次是**把出现次数写死成 1**。
⑬ **桶形必须容得下红灯现场的命名形制**：本件的桶第一版只列 `_controls`／`_controls_red\d+`，于是第一次实跑的红灯面
   （`…_controls.attempt1-mark-line-hits-twice.out`）落进 0 个桶。红得对（逐件断言拦住了），但那批面本来就该被登记；
   修法＝`.attemptN-<原因>` 后缀写成可选，并按**真实现场清单**预演逐组等值。
⑭ **模板槽位的取值须是一个原子字符串**：四个正对照的取数正则原先各捕多字段，`grab` 返回元组、直接落到槽上 ⇒
   渲染腿的类型闸（`isinstance(val, str)`）拒印＝**真件本轮红过一次**（不是造出来的红）。修法＝每条 K_CTL* 改成
   捕一个整段（每格＝回执里逐字节摘出的一段），并把模板里重复的标签删掉；这条闸门的正对照＝C97（列表落进槽位）。
⑮ **扰动必须真的移动匹配集，否则它只是纸面扰动**：本轮四条按形状抄来的扰动实跑全 rc＝0＝空转——① glob 的片号
   插值 `%d`→`%s`（`%` 格式化里两者产出同一串）；② needle `?? `→`??`（porcelain 的未跟踪行永远带那个空格）；③ 把
   「切到 `；` 为止」改成不切（解释里不含清单分隔符 `／` ⇒ 条数不变，红落在**另一条**闸门上）；④ 钉 locale 的那条
   改钉 en_US.UTF-8（本机默认不是它 ⇒ 两条读数照旧不同）。出路只有两条：把扰动改成**真的移动**匹配集（①去掉插值、
   ③换错分隔符、④两条都钉成同值），或如实进 UNCOVERED 并附实测（②）。预检能验「锚点命中 1 处」，验不出「命中了
   但行为没变」——这一族只有真跑才知道。
⑯ **扰动只能落在有这个变量的作用域里**：把 `faces_check` 里的 `modified` 换成 `added` 得到的是 `NameError: name
   'added' is not defined`（`added` 是 `build()` 的局部名），那不是红在闸门上。要么改扰**调用处**（C22），要么换一个
   当场真有、且真会判错的值（C23 用 `pinned_faces`＝把 relpath 混进 basename 清单）。
⑰ **删掉比较的一项不是坏输入，是放宽**：C63／C96 第一版各删掉闭合式的一段，副本照旧绿（放宽后的判据更容易满足）。
   正对照要证的是「这条比较真的在算」⇒ 改成**反向**（现算与另一出处本就同值／标本里两个数本就不同值），实跑必红。
"""
import glob
import os
import re
import signal
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
SRCNAME = "am_s31_msg.py"
SRC = os.path.join(HERE, SRCNAME)
PREFIX = "zctl_s31_msg_"
SUB_TIMEOUT = 300
ENV_FAIL_STOP = 3
WINDOW = 3
text = open(SRC, encoding="utf-8").read()

# 本件对**自己**声明的两个自述常量：生成器只读这两个字面（不读本件的 stdout），并要求各自在本源码里恰好命中 1 行。
# ⇒ 这两行是**唯一**出现处：下面任何标签／注释都不许再抄一遍，否则生成器 selfctl_check 当场红。
SELF_UNUSED_NEEDLE = "取值表里有模板没用到的读数"
SELF_BASE_MARK = "本轮新学（环境级＋口径级）＝**普查仪器坏了 ⇒ 红线换承担者，而不是换口径，也不去修共享状态**："

ENV = dict(os.environ)
ENV["CUDA_VISIBLE_DEVICES"] = ""
ENV["JAX_PLATFORMS"] = "cpu"


def sh(*args):
    return subprocess.run(list(args), cwd=REPO, capture_output=True, text=True).stdout


def rd(path):
    return open(path, "rb").read().decode("utf-8", "replace")


# (标签, [(源码正则, 替换文本)], 期望闸门句)
CASES = [
    # —— ① 单值取数族（one／grab 的「恰好 1」）——
    ("C1 片号出处正则改成主题里没有的形状（单值取数 0 命中）",
     [(r'"S7 分片 \(\\d\+\)", SUBJ', r'"S7 分片号X (\d+)", SUBJ')],
     "（须恰好 1）⇒ 出处不唯一或已腐烂，拒绝生成"),
    ("C2 回执档案 glob 的件名形状改成 .out（号角本普查空转）",
     [(r'if re\.fullmatch\(r"am_s7_verify\(\\d\+\)\\\.log", os\.path\.basename\(p\)\)\)',
       'if re.fullmatch(r"am_s7_verify(\\d+)\\.out", os.path.basename(p)))')],
     "全树 glob 没取到任何回执档案"),
    ("C3 最新回执与本件片号的关系从「＝HEAD 片号」改成「＝本片号」",
     [(r'^assert max\(LOGS\) == PREVSLICE', 'assert max(LOGS) == SLICE')],
     "有一片没落或片号口径变了"),
    ("C4 上一片回执 glob 的片号插值写成本片号（回执 0 份）",
     [(r'"docs/evidence/\*/am_s7_verify%d\.log" % PREVSLICE', '"docs/evidence/*/am_s7_verify%d.log" % SLICE')],
     "上一片回执在全树命中 "),
    ("C5 上一片回执的 glob 丢掉片号插值（同一条 glob 命中全树 13 份回执）",
     [(r'"docs/evidence/\*/am_s7_verify%d\.log" % PREVSLICE', '"docs/evidence/*/am_s7_verify*.log"')],
     "上一片回执在全树命中 "),
    ("C6 跨日期目录硬写成本件目录",
     [(r'^PREVDIR = os\.path\.dirname\(CURL\)$', "PREVDIR = E")],
     "本轮的跨日期登记要改"),
    ("C7 回归件 glob 改成不存在的后缀（第四个出处没了）",
     [(r'"docs/evidence/\*/_raw_t\*_full\.log"', '"docs/evidence/*/_raw_t*_full.LOG"')],
     "全树回归件命中 "),

    # —— ② 角色桶 ↔ §26.33 落笔分组 ——
    ("C8 桶名清单与桶序的配对改成自件桶在前",
     [(r'^assert \[k for k, _ in BUCKETS\] == BKEY_PIN \+ BKEY_SELF',
       'assert [k for k, _ in BUCKETS] == BKEY_SELF + BKEY_PIN')],
     "桶名与桶序不一致"),
    ("C9 表头控件面桶的形制放宽到吞掉回执链里的表头生成器（一件落进两桶）",
     [(r'\^am_s%d_hdr_controls\\\.py\$', '^am_s%d_hdr.*\\.py$')],
     "角色口径不唯一，拒绝生成"),
    ("C10 桶普查静默截断（每桶最多记三件）",
     [(r'^        hits\[m\[0\]\]\.append\(bn\)$', "        hits[m[0]] = (hits[m[0]] + [bn])[:3]")],
     "≠ 新增总数"),
    ("C11 自件桶的片号插值写成上一轮的号（本件的面落不进任何桶）",
     [(r'^         SLICE, SLICE\)\),$', "         PREVSLICE, PREVSLICE)),")],
     "角色口径不唯一，拒绝生成"),
    ("C12 分组项与桶名的 zip 对象换成自件桶（配对错位）",
     [(r'for i, \(t, key\) in enumerate\(zip\(toks, BKEY_PIN\)\)',
       'for i, (t, key) in enumerate(zip(toks, BKEY_SELF))')],
     "顺序与本件桶序不一致"),
    ("C13 分组顺序倒置（zip 同一批但反着配）",
     [(r'for i, \(t, key\) in enumerate\(zip\(toks, BKEY_PIN\)\)',
       'for i, (t, key) in enumerate(zip(toks, list(reversed(BKEY_PIN))))')],
     "顺序与本件桶序不一致"),
    ("C14 分组项的形状自查反向（标签里的数与加粗声称本应同值）",
     [(r'assert int\(want\) == int\(claim\)', "assert int(want) != int(claim)")],
     "标签里的数 "),
    ("C15 分组等式左边写成「未跟踪＝总计」",
     [(r'assert nu == sum\(c for _l, c in rows\) and nu \+ nm == tot', "assert nu == tot and nu + nm == tot")],
     "落笔闭合式不齐"),
    ("C16 分组项数与生成器前桶数的关系写成多一项",
     [(r'assert len\(toks\) == len\(BKEY_PIN\)', "assert len(toks) == len(BKEY_PIN) + 1")],
     "项 ≠ 生成器前桶"),
    ("C17 修改面清单的分隔符从 `／` 换成 `；`（整串当一条 ⇒ 条数闸红；把 `；` 那半边删掉只改内容不改条数＝空转）",
     [(r'    lst = sorted\(faces\.split\("／"\)\)\s*$', '    lst = sorted(faces.split("；"))')],
     "修改面清单现数 "),

    # —— ③ 入库面计数／差异口径 ——
    ("C18 无代码腿普查的作用域指到 docs（本轮真正动过的面）",
     [(r'NCODE_ARGV = \("git", "diff", "--cached", "--name-only", "--", "src", "tests"\)',
       'NCODE_ARGV = ("git", "diff", "--cached", "--name-only", "--", "docs", "tests")')],
     "两句都成了假话"),
    ("C19 暂存清单的状态字母集合要求成只有 A（M 面被当成非法）",
     [(r'assert kinds == \["A", "M"\]', 'assert kinds == ["A"]')],
     "暂存清单里有 A/M 之外的状态字母"),
    ("C20 未暂存漂移检查把已暂存当成未暂存",
     [(r'"diff", "--name-only"\)\.strip\(\)', '"diff", "--cached", "--name-only").strip()')],
     "还有未暂存的已跟踪改动"),
    ("C21 面行数闭合式两边相减写反",
     [(r'assert add == n_staged - n_head', "assert add == n_head - n_staged")],
     "≠ 行数差"),
    ("C22 修改面比对喂进去的是新增清单（调用处口径错位）",
     [(r'faces = faces_check\(modified, pinfaces\)', "faces = faces_check(added, pinfaces)")],
     "登记的那批"),
    ("C23 上一片纪实的归属按 basename 清单判定（relpath 与 basename 口径混用；`added` 是 build() 的局部名，在这个函数里根本不存在）",
     [(r'assert PREVDAILY in modified', "assert PREVDAILY in pinned_faces")],
     "修改面里没有上一片的纪实"),
    ("C24 落笔现数三处闭合里把桶和挪掉一件",
     [(r'^    assert int\(wv\["wnu"\]\) == pnu == u\["pin"\]', '    assert int(wv["wnu"]) == pnu == u["pin"] - 1')],
     "落笔未跟踪数三处不闭合"),
    ("C25 落笔闭合式两边写反（总计当成未跟踪）",
     [(r'assert int\(wv\["wnt"\]\) == ptot', 'assert int(wv["wnt"]) == ptot + 1')],
     "落笔闭合式两边不闭合"),
    ("C26 前 12 桶和与落笔未跟踪数的关系加一",
     [(r'assert sum\(len\(hits\[k\]\) for k in BKEY_PIN\) == pnu', "assert sum(len(hits[k]) for k in BKEY_PIN) == pnu + 1")],
     "前 12 桶和 "),
    ("C27 修改面数的三个出处少配一个（声称与暂存 M 差一）",
     [(r'assert pnface == len\(pinfaces\) == nm', "assert pnface == len(pinfaces) == nm + 1")],
     "登记的修改面 "),
    ("C28 落笔登记的修改件数与暂存 M 的关系加一",
     [(r'assert pnm == nm', "assert pnm == nm + 1")],
     "落笔登记的修改件数 "),

    # —— ④ 顶格字段槽位普查（本轮新增）——
    ("C29 把 FIELDS 声明数改回 99（消息里承诺的那条坏输入）",
     [(r'^    ndef = len\(fields_of\(os\.path\.join\(REPO, HDRC\)\)\)$', "    ndef = 99")],
     "有读数被填错了槽位"),
    ("C30 顶格字段普查的正则失去等号字符类（尺子取到 0 处）",
     [(r'\[＝= \]\+\\\*\{0,2\}', '[x]+\\*{0,2}')],
     "这条普查尺子失效"),
    ("C31 FIELDS 的 ast 赋值名写成 FIELD（顶格字段的分母不是现取的）",
     [(r'getattr\(node\.targets\[0\], "id", None\) == "FIELDS"', 'getattr(node.targets[0], "id", None) == "FIELD"')],
     "取不到 FIELDS 赋值"),

    # —— ⑤ NVML 坏掉之后新写的用户态版本双出处 ——
    ("C32 dpkg 状态过滤放宽到含 rc（把已卸载包的残留当成在用版本）",
     [(r'cols\[1\] != "amd64" or cols\[2\] != "ii "', 'cols[1] != "amd64" or cols[2] not in ("ii ", "rc ")')],
     "的 libnvidia-compute 有 "),
    ("C33 dpkg 架构过滤写成本机表里没有的值（版本出处为空）",
     [(r'cols\[1\] != "amd64" or', 'cols[1] != "arm64" or')],
     "的 libnvidia-compute 有 "),
    ("C34 dpkg 的版本列不剥 release 后缀（形状不认识）",
     [(r'v = cols\[3\]\.split\("-"\)\[0\]', 'v = cols[3]')],
     "的 ii 行版本号形状不认识"),
    ("C35 加载器解析退回 abspath（拿到的是符号链接名而不是三段版本）",
     [(r'used = os\.path\.basename\(os\.path\.realpath\(', 'used = os.path.basename(os.path.abspath(')],
     "加载器解析到的 NVML 文件名不是三段版本号"),
    ("C36 把在用库名硬写残留版本号（两个出处不同值＝本轮第二版踩过的形状）",
     [(r'used = os\.path\.basename\(os\.path\.realpath\("/usr/lib/x86_64-linux-gnu/libnvidia-ml\.so\.1"\)\)',
       'used = os.path.basename(os.path.realpath("/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.535.288.01"))')],
     "两个出处不同值"),
    ("C37 盘上旁证 glob 的目录写错（glob 出处为空）",
     [(r'glob\.glob\("/usr/lib/\*/libnvidia-ml\.so\.\*"\)', 'glob.glob("/usr/lib64/*/libnvidia-ml.so.*")')],
     "⇒ glob 出处为空"),
    ("C38 内核模块读数换成用户态的值（根因句的前提被造假）",
     [(r'drv = rd\("/sys/module/nvidia/version"\)\.strip\(\)', 'drv = "580.178.04"')],
     "这句根因本轮不成立，要改写"),

    # —— ⑥ 替代红线：/proc 的 argv[0] 整词匹配 ——
    ("C39 整词匹配退回子串（两条对照失效＝假零的形状）",
     [(r'^    return blob\.split\(b"\\0"\)\[0\]\.decode\("utf-8", "replace"\) == INTERP$',
       '    return INTERP.encode() in blob')],
     "的两条对照失效"),
    ("C40 合成的负对照命令行里不带解释器路径（负样本形状失效）",
     [(r'neg_blob = b"/bin/bash\\0-c\\0grep " \+ INTERP\.encode\(\) \+ b"\\0"',
       'neg_blob = b"/bin/bash\\0-c\\0grep nope\\0"')],
     "松散口径的负对照形状失效"),
    ("C41 合成的正对照换成别的解释器（那条 1 数不出了）",
     [(r'pos = match_argv0\(\(INTERP \+ "\\0x\.py"\)\.encode\(\)\)',
       'pos = match_argv0(("/usr/bin/env" + "\\0x.py").encode())')],
     "的两条对照失效"),
    ("C42 红线不再排除自身与父进程（自己的解释器被抓进分子＝当场拒绝生成）",
     [(r'        if int\(d\) in \(me, myparent\):', "        if int(d) in (me,) and False:")],
     "那条红线不成立"),

    # —— ⑦ 干跑控件面读数的两侧闭合 ——
    ("C43 表内 G 行的行首前缀少两个空格（逐行数到 0 行）",
     [(r'l\.startswith\("  G"\)', 'l.startswith("G")')],
     "表头控件面里表内 G 行 "),
    ("C44 表内 G 行的行首字母改成 E（标签与行数配不上）",
     [(r'l\.startswith\("  G"\)', 'l.startswith("  E")')],
     "表头控件面里表内 G 行 "),
    ("C45 基线行的前缀写歪（「基线不进表」那条口径没有第二出处）",
     [(r'l\.startswith\("基线（未扰动副本"\)', 'l.startswith("基线（扰动副本")')],
     "表头控件面基线行现数 "),
    ("C46 绿路闭合式里丢掉基线那一条",
     [(r'assert int\(hgreen\) == grow \+ hbase', "assert int(hgreen) == grow")],
     "绿路账没有第二出处"),
    ("C47 三面判读行的一致性门槛写成 3",
     [(r'assert len\(set\(verd\)\) == 1', "assert len(set(verd)) == 3")],
     "三面判读行不一致"),
    ("C48 超时必须为零的那条链改成容忍一条（环境故障被当全过）",
     [(r'assert int\(nto\) == hto == ato == 0', "assert int(nto) == hto == ato == 1")],
     "不能当全过面引用"),
    ("C49 正对照逐行定位器的合成标本丢掉两个前导空格（这条对照本身失明）",
     [(r'\["  正对照：合成的一条"\]', '["正对照：合成的一条"]')],
     "正对照逐行定位器对合成样本失明"),
    ("C50 正对照逐行定位器的负标本带上前导空格（非顶格样本也被算进来）",
     [(r'\["正对照：换了形状"\]', '["  正对照：换了形状"]')],
     "正对照定位器把非顶格样本也算进来了"),
    ("C51 承重面唯一所指的门槛从 1 改成 2（该断言的反例方向本轮造不出，用门槛位移证明它是活闸）",
     [(r'assert len\(\[x for x in carry if int\(x\[1\]\) == int\(top\[1\]\)\]\) == 1',
       'assert len([x for x in carry if int(x[1]) == int(top[1])]) == 2')],
     "⇒ 那句「最多的一面」没有所指"),
    ("C51b 承载行分母从五张面改成四张（那句「承载面读数」不再逐面现取）",
     [(r'assert len\(carry\) == 5', "assert len(carry) == 4")],
     "自查件的承载行实取 "),

    # —— ⑧ 上一片回执／局部仪器的复跑 ——
    ("C52 回执正文逐字节比较错位一格（cmp 却仍过）",
     [(r'assert rb\(CURL\)\[off:\] == rb\(RAWR\)', "assert rb(CURL)[off + 1:] == rb(RAWR)")],
     "回执正文区与原始 stdout 不逐字节相同"),
    ("C53 cmp 三联的期望写成两条零（偏移 +1 的那两条本该非零）",
     [(r'assert rcs == \[0, 1, 1\]', "assert rcs == [0, 0, 1]")],
     "三件套复跑＝"),
    ("C54 只读仪器那句三联的期望改成 0/1/2（全绿声称的形状变了）",
     [(r'assert aftriple == "0/1/1"', 'assert aftriple == "0/1/2"')],
     "那张回执不能当全绿引用"),
    ("C55 回执总长与正文区边界的闭合式加一",
     [(r'assert b\(CURL\) == off \+ b\(RAWR\)', "assert b(CURL) == off + b(RAWR) + 1")],
     "正文区边界不闭合"),
    ("C56 局部仪器复跑的 rc 期望从 1 改成 0（它本轮仍全绿＝它已不是局部仪器）",
     [(r'assert r\.returncode == 1, "复跑', 'assert r.returncode == 0, "复跑')],
     "它本轮仍全绿"),
    ("C57 归档标本比对读错流（stderr 去等 stdout）",
     [(r'assert rb\(ARCHK_PIN_ERR\) == r\.stderr\.encode\(\)', "assert rb(ARCHK_PIN_ERR) == r.stdout.encode()")],
     "与已归档复跑现场不逐字节相同"),
    ("C58 标本 stdout 的零字节门槛写成 1",
     [(r'assert b\(ARCHK_PIN_OUT\) == 0', "assert b(ARCHK_PIN_OUT) == 1")],
     "归档的复跑现场 stdout 非 0 B"),
    ("C59 复跑红点的归属件名写成归档自查件",
     [(r'assert os\.path\.basename\(raised\[0\]\) == os\.path\.basename\(PREVMSG\)',
       'assert os.path.basename(raised[0]) == os.path.basename(ARCHPY)')],
     "而非上一片生成器"),
    ("C60 上一片信息生成器复跑的红点归属也指到尺子（同族的第二处错位）",
     [(r'assert os\.path\.basename\(raised\[0\]\) == os\.path\.basename\(PREVMSG\)',
       'assert os.path.basename(raised[0]) == os.path.basename(RULER)')],
     "而非上一片生成器"),
    ("C61 局部仪器红点 needle 写歪（归因句拦不住别的断言）",
     [(r'rerun_local\(ARCHPY, \("LESSON 锚句"', 'rerun_local(ARCHPY, ("LESSON 锚句X"')],
     "复跑红在了别的断言上"),
    ("C62 写入件标本 attempt1 的末行形状 needle 写歪",
     [(r'"落笔现数" in t1 and "本节的闭合式写的是" in t1', '"落笔现数" in t1 and "闭合式写死" in t1')],
     "attempt1 的末行不像那条闭合式闸门"),
    ("C63 attempt1 句里「现数≠写死分母」反向（标本里 78 与 67 本就不同值 ⇒ 反向必红；只删这一项是**放宽**，副本照旧绿）",
     [(r'and int\(nums\[2\]\) != int\(nums\[3\]\)', 'and int(nums[2]) == int(nums[3])')],
     "attempt1 那句不闭合"),
    ("C64 attempt2 的守卫标本按本件（信息生成器）的名字找",
     [(r'"有仪器正在跑" in t2 and os\.path\.basename\(WRITER\) in t2',
       '"有仪器正在跑" in t2 and os.path.basename(SELF) in t2')],
     "attempt2 的末行不像那条守卫"),
    ("C65 追加面张数的分母从 4 改成 5",
     [(r'assert len\(app\) == 4', "assert len(app) == 5")],
     "写入件登记的追加面实取 "),
    ("C66 写入件落笔现数那句的闭合式改成「未跟踪＝总计」",
     [(r'assert int\(wnu\) \+ int\(wnm\) == int\(wnt\)', "assert int(wnu) == int(wnt)")],
     "写入件那句落笔现数不闭合"),

    # —— ⑨ 计数口径 `-uall` 的本轮真实形态 ——
    ("C67 `git status` 索引列的前缀 needle 改成合并态（索引列现数 0）",
     [(r'if l\.startswith\("M "\)\)', 'if l.startswith("MM "))')],
     "与暂存 M 清单不齐"),
    ("C68 落笔清单的归属自查从差集改成并集（入库清单永不自洽）",
     [(r'gone = sorted\(set\(pin_basenames\) - set\(os\.path\.basename\(p\) for p in added\)\)',
       'gone = sorted(set(pin_basenames) | set(os.path.basename(p) for p in added))')],
     "入库清单已腐烂"),
    ("C70 控件面锚点句的唯一性门槛从 1 改成 2（两处出现会被静默拿第一条）",
     [(r'^    assert nun == 1', "    assert nun == 2")],
     "那条扰动会被判空转或 ±3 窗口指错闸门"),
    ("C71 基线标记句的唯一性门槛从 1 改成 0（标记句出现两次也算过）",
     [(r'^    assert nmark == 1', "    assert nmark == 0")],
     "副本的「走到打印分支」判定不可用"),

    # —— ⑩ 尺子族（md5 分组 vs glob 形状）——
    ("C72 逐字节副本族的组数门槛从 1 改成 2",
     [(r'assert len\(big\) == 1', "assert len(big) == 2")],
     "逐字节副本族有 "),
    ("C73 尺子族现数与回执登记份数的关系加一",
     [(r'assert len\(members\) == want, "尺子族现数', 'assert len(members) + 1 == want, "尺子族现数')],
     "有人另起过一把尺子或抄丢了"),
    ("C74 尺子 md5 的前缀从 12 位截成 8 位（两个出处比的是不同长度）",
     [(r'return len\(members\), len\(shape\), md5\(RULER, 12\), stray', "return len(members), len(shape), md5(RULER, 8), stray")],
     "尺子 md5：现取 "),
    ("C75 形状普查剔掉那条带名字的例外（「差额有名字」这句失去所指）",
     [(r'    for p in glob\.glob\(os\.path\.join\(REPO, "docs/evidence/\*/am_s7_verify\*\.py"\)\):',
       '    for p in [x for x in glob.glob(os.path.join(REPO, "docs/evidence/*/am_s7_verify*.py"))\n'
       '               if "dryrun" not in x]:')],
     "那句「差额有名字」本轮不成立"),
    ("C76 上一片那把尺子是否属于本族不再验（族里换进别人的件）",
     [(r'assert os\.path\.basename\(RULER\) in \[os\.path\.basename\(x\) for x in members\]',
       'assert os.path.basename(HDRC) in [os.path.basename(x) for x in members]')],
     "它不是逐字节副本"),

    # —— ⑪ 指纹的 locale 耦合 ——
    ("C77 默认 locale 那条也改钉 LC_ALL＝C（两条同值＝耦合检测空转；本机默认不是 en_US.UTF-8，把它改钉成那个值仍是两个不同读数＝空转）",
     [(r'    d = run_bash\(FINGER_CMD\)\.strip\(\)',
       '    d = subprocess.run(["bash", "-c", FINGER_CMD], cwd=REPO, capture_output=True,\n                       text=True, env=dict(os.environ, LC_ALL="C")).stdout.strip()')],
     "两条 locale 现算同值"),
    ("C78 取消钉环境（同上一条方向，验同一句的另一侧）",
     [(r'text=True,\n                       env=dict\(os\.environ, LC_ALL="C"\)\)\.stdout\.strip\(\)',
       'text=True).stdout.strip()')],
     "两条 locale 现算同值"),
    ("C79 交换两条 locale 读数（参与闭合的那条变成钉了 locale 的那条）",
     [(r'^    return d, c$', "    return c, d")],
     "指纹四路不闭合"),
    ("C80 闭合式里把回执表头那条指纹换成回执的父提交（槽位接错）",
     [(r'assert fd == reg\["fs"\] == g\["finger"\]', 'assert fd == reg["fs"] == g["parent"]')],
     "指纹四路不闭合"),
    ("C81 与 HEAD 登记互证的那条按 16 位取十六进制串（两个 32 位读数都不在集合里）",
     [(r'hexes = set\(re\.findall\(r"\\b\[0-9a-f\]\{32\}\\b", head_dev\)\)',
       'hexes = set(re.findall(r"\\b[0-9a-f]{16}\\b", head_dev))')],
     "未登记的值"),

    # —— ⑫ 开发日志的形制与节内唯一性 ——
    ("C82 形制约束退回「本节就是上一节」",
     [(r'assert int\(secn\) == PREVSEC \+ 1', "assert int(secn) == PREVSEC")],
     "形制约束没落实"),
    ("C83 顶级子条标签的互异性检查反向",
     [(r'assert len\(labs\) >= 5 and len\(labs\) == len\(set\(labs\)\)',
       "assert len(labs) >= 5 and len(labs) != len(set(labs))")],
     "且有重名标签"),
    ("C84 LESSON 锚句写歪（节内唯一性里那条承接断言失明）",
     [(r'^LESSON_NEEDLE = "自审的方向性"', 'LESSON_NEEDLE = "自审的方向性X"')],
     "节内唯一性破了"),
    ("C85 本轮新学锚句写歪（同族的第二条）",
     [(r'^NEWLESSON_NEEDLE = "普查仪器没有第二出处"', 'NEWLESSON_NEEDLE = "普查仪器没有第二出处X"')],
     "节内唯一性破了"),
    ("C86 钉住口径的引号 needle 换成单书名号（四条唯一性里的第三条）",
     [(r'"钉住口径": len\(\[s for _l, s in segs if "当前「" in s\]\)',
       '"钉住口径": len([s for _l, s in segs if "当前『" in s])')],
     "节内唯一性破了"),
    ("C87 上一节引用数的门槛从 1 改成 0（该节形制与回执表头的声称不再比对）",
     [(r'assert prev_cite == 1', "assert prev_cite == 0")],
     "该节形制与回执表头的声称不符"),
    ("C88 上一节的切段取成本节（两出处互证指错面）",
     [(r'assert prev_bul == arch_bul', "assert len(top_bullets(body)[0]) == arch_bul")],
     "切段口径变了"),
    ("C89 「本节在 HEAD 里 0 次」写成 1 次",
     [(r'assert \(nh, ns\) == \(0, 1\)', "assert (nh, ns) == (1, 1)")],
     "次／暂存 blob 里"),

    # —— ⑬ 回执表头自身的读数控件 ——
    ("C90 回执正文被截断的旗标期望反过来（截断件也敢引用）",
     [(r'assert tr == "False"', 'assert tr == "True"')],
     "回执正文被截断，不许引用"),
    ("C91 tree 条目等式丢掉 tree 那一项",
     [(r'assert int\(entries\) == int\(blob\) \+ int\(treeent\)', "assert int(entries) == int(blob)")],
     "tree 条目 ≠ blob＋tree"),
    ("C92 回执表头「上一片」的连号关系写成同一片",
     [(r'assert prev2 == PREVSLICE - 1', "assert prev2 == PREVSLICE")],
     "连号断了"),
    ("C93 单列复核丢掉「A＋M＝文件数」那半边",
     [(r'assert int\(match\) == int\(nfile\) and int\(a\) \+ int\(m\) == int\(nfile\)',
       "assert int(match) == int(nfile) and int(a) == int(nfile)")],
     "单列复核不全绿"),
    ("C94 回执判读行的必需尾串写成短串（放宽到 PASS 也算全绿）",
     [(r'verdict\.endswith\("ALL CHECKS PASS"\)', 'verdict.endswith("ALL CHECKS")')],
     "回执判读行不是 ALL CHECKS PASS"),
    ("C95 回执里日志节号的集合相等改成偏移一节（两个出处不再互证）",
     [(r'assert rsecs == \{str\(PREVSEC\)\}', "assert rsecs == {str(PREVSEC + 1)}")],
     "回执里的日志节号＝"),
    ("C96 顶格字段的三出处闭合反向（现算与仪器那句本就同值 ⇒ 反向必红；丢掉第三个比较只是放宽，副本照旧绿）",
     [(r'assert str\(len\(fields\)\) == afld\.split\("/"\)\[0\] == afld\.split\("/"\)\[1\]',
       'assert str(len(fields)) != afld.split("/")[0] == afld.split("/")[1]')],
     "顶格字段：FIELDS 声明 "),
    # 这条不是纸面设想：本轮**真件自己**就红在这条闸门上一次（四个正对照的 K_CTLB／K_CTLC／K_CTLD 各捕多字段，
    # 落到槽位上是元组 ⇒ 渲染腿拒印）。修成「每格一个原子读数」之后，这里用「列表落进槽位」继续验这条不空转。
    ("C97 槽位落进一个非字符串取值（渲染腿的类型闸）",
     [(r'CTLA=g\["ctla"\]', 'CTLA=rows')],
     "的取值不是字符串"),
    ("C96c 正对照读数的分段取错侧（取 `、` 的**后段**＝同一个对照里那个预期为 0 的第二把尺子）",
     [(r'n = int\(re\.search\(r"\\d\+", val\.split\("、"\)\[0\]\)\.group\(\)\)',
       'n = int(re.search(r"\\d+", val.split("、")[-1]).group())')],
     "的首位不匹配数＝"),

    # —— ⑭ 模板形制闸（渲染前后）——
    ("C98 模板槽位漏一个尾 @（坏形制静默不替换）",
     [(r'表头顶格字段 @@NDEF@@ 行', "表头顶格字段 @@NDEF@ 行")],
     "个不属于成对槽"),
    ("C99 模板里加一个取值表没有的槽（那条数没现取）",
     [(r'三件套 rc＝@@TRIPLE@@', "三件套 rc＝@@ZZNOSLOT@@")],
     "模板里有未取值的槽"),
    ("C100 模板字段名多打一个字母（同族的第二处坏形制）",
     [(r'正文区与 `@@RAWRN@@`', "正文区与 `@@RAWRNN@@`")],
     "模板里有未取值的槽"),
    ("C101 取值表里留一条模板没用到的读数",
     [(r'^    V = dict\($', '    V = dict(ZZUNUSED="1",\n')],
     SELF_UNUSED_NEEDLE),
    ("C102 取值表里把一条读数填成非字符串（int 会被静默 str 化就没人知道）",
     [(r'WNU=str\(wv\["wnu"\]\), WNM=str\(wv\["wnm"\]\), WNT=str\(wv\["wnt"\]\),',
       'WNU=wv["wnu"], WNM=str(wv["wnm"]), WNT=str(wv["wnt"]),')],
     "的取值不是字符串"),
    ("C103 渲染后的正文里混进一个孤立 @（槽位与值没分家）",
     [(r'SS_RAW=raw_ss,', 'SS_RAW=raw_ss + "@X",')],
     "渲染后的文本里仍有 "),
    ("C104 主题第一行被加料（登记的那句与打印的那句不再是同一行）",
     [(r'    v\["SUBJECT"\] = SUBJECT$', '    v["SUBJECT"] = SUBJECT + " 追加"')],
     "主题第一行必须就是登记的那句"),
    ("C105 主题长度上限退成一米（超长主题没人拦）",
     [(r'assert len\(first\) <= 300', "assert len(first) <= 1")],
     "字符过长"),
    ("C106 全角括号按行配对被拆掉一处（模板里落笔就不成对）",
     [(r'（全树回执 glob 里 @@SLICE@@ 号角本缺席，本件现验），本片不预支。',
       '（全树回执 glob 里 @@SLICE@@ 号角本缺席，本件现验，本片不预支。')],
     "全角括号按行不成对"),
    ("C107 行内代码的反引号少一个",
     [(r'信息本身由 `@@SELFNAME@@`（md5 前 12', "信息本身由 @@SELFNAME@@`（md5 前 12")],
     "行内代码的反引号不成对"),
    ("C108 粗体标记少一半（剔除行内代码后仍该拦住）",
     [(r'\*\*src／tests 本轮 @@NCODE@@ 件改动\*\*', "*src／tests 本轮 @@NCODE@@ 件改动**")],
     "粗体标记不成对"),
    ("C109 渲染结果没有尾换行",
     [(r'^    txt = TMPL$', "    txt = TMPL.rstrip(\"\\n\")")],
     "尾换行"),

    # —— ⑮ 会执行**别的仪器**的一条扰动（见坑⑪：前后都验「工作树对暂存面零漂移」）——
    # 本轮实测：克隆副本**走不到**写入件的守卫——它先撞上写入件自己的落笔闭合式（现场里未跟踪 3＋已登记 75＝78，
    # 而 §26.33 写的是 67）。那条红仍然是**真件的活闸**（档案面没落盘就拒绝写），只是不是本条原先声称的那句。
    ("C110 克隆本归档的写入件并带 --write 跑它（不许真动档案：本轮它红在落笔闭合式）",
     "CLONE_WRITER", "本节的闭合式写的是"),
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


def unstaged():
    """工作树对**暂存面**的漂移清单：为空＝那五张档案面自上次 `git add` 起逐字节未动。"""
    return [l for l in sh("git", "-c", "core.quotePath=false", "diff", "--name-only").splitlines() if l.strip()]


def run_clone_writer():
    """C110：把本归档的写入件按字节克隆到本目录，带 `--write` 跑它。

    安全性论证（不是「我相信它不会写」）：写入件的守卫按 `/proc/PID/cmdline` 的**整词**比对 TOKENS，
    而本件的件名就在它那份 TOKENS 里 ⇒ 克隆副本一定先撞上「有别的仪器正在跑」并在任何追加**之前**退出。
    这条论证本身由前后两次 `unstaged()` 比对兜底：真写了就会非空，本件当场红而不是留下一份被改过的档案。
    """
    assert not unstaged(), "跑克隆副本之前工作树对暂存面已有漂移：%s ⇒ 这条控制的「没写」判据失效，先补齐 git add" % (
        unstaged(),)
    src = rd(os.path.join(HERE, "am_s31_docs.py"))
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8",
                                     dir=HERE, prefix="zz_s31_writer_clone_") as fh:
        fh.write(src)
        tmp = fh.name
    try:
        p = subprocess.Popen([sys.executable, tmp, "--write"], cwd=REPO, env=ENV,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             start_new_session=True)
        try:
            out, err = p.communicate(timeout=SUB_TIMEOUT)
            msg = (err or "") + (out or "")
            rc = p.returncode
            if rc == 0:
                raise AssertionError("克隆副本竟然写盘成功（rc=0）⇒「别的仪器仍按整词命中拦停」这句是假话")
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            p.communicate()
            return "TIMEOUT", ""
        after = unstaged()
        assert not after, "克隆副本退出后工作树出现漂移 %s ⇒ 守卫没拦住，档案面被真写过（这条红必须修仪器）" % after
        return rc, msg
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def precheck():
    """纸面预检（坑⑦）：锚点恰好命中 1 处＋扰动后可编译＋期望句在源码里可定位；再加本件自己的两条对照。"""
    bad_anchor, dup_expect, runtime_only, special = [], [], [], []
    for name, subs, expect in CASES:
        if subs == "CLONE_WRITER":
            special.append(name)
            continue
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
            dup_expect.append("%s：期望句「%s」在生成器源码里 %d 处（%s）⇒ 位置判定按「任一出现处 ±%d」" % (
                name, expect, len(hits), hits, WINDOW))
    # 本件自己的两条对照（正＝两个自述常量各在生成器源码里命中 1 行；负＝写歪的锚点句必须 0 命中）
    nun = len([l for l in text.splitlines() if SELF_UNUSED_NEEDLE in l])
    nmark = len([l for l in text.splitlines() if SELF_BASE_MARK in l])
    if nun != 1:
        bad_anchor.append("自件正对照：判据锚点句在生成器源码里命中 %d 行（须 1）" % nun)
    if nmark != 1:
        bad_anchor.append("自件正对照：基线标记句在生成器源码里命中 %d 行（须 1）" % nmark)
    nneg = len([l for l in text.splitlines() if SELF_UNUSED_NEEDLE + "X" in l])
    if nneg != 0:
        bad_anchor.append("自件负对照：写歪的锚点句竟然命中 %d 行 ⇒ 这条 needle 不判别（假绿形状）" % nneg)
    own = [l for l in open(SRC.replace(SRCNAME, os.path.basename(__file__)), encoding="utf-8").read().splitlines()
           if re.match(r'^SELF_(UNUSED_NEEDLE|BASE_MARK) = "', l)]
    assert len(own) == 2, "本文件里两个自述常量的声明行实取 %d 行（须 2）⇒ 生成器 selfctl_check 会当场红" % len(own)
    print("纸面预检：CASES 声明＝%d 条（含克隆腿 %d 条）｜锚点／编译不合格＝%d 条｜期望句多处出现（按任一±%d 判）＝%d 条｜"
          "期望句只在运行时（位置判定不适用）＝%d 条" % (
              len(CASES), len(special), len(bad_anchor), WINDOW, len(dup_expect), len(runtime_only)))
    for r in bad_anchor + dup_expect + runtime_only:
        print("  · " + r)
    return not bad_anchor


def raised(msg):
    """只取**最后一条**异常行：traceback 会把源码行原样回显，而闸门句本来就写在源码里。"""
    hits = re.findall(r"^(?:\w+Error|AssertionError)[^\n]*", msg, re.M)
    return hits[-1] if hits else "（没取到异常行）"


def trace_line(msg):
    """traceback 里**最后一个** `File ".../zctl_s31_msg_*.py", line N`＝真正抛出的那一行。"""
    hits = re.findall(r'File "[^"]*%s[^"]*\.py", line (\d+)' % PREFIX, msg)
    return int(hits[-1]) if hits else None


def gate_lines(expect):
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
    return [f for f in os.listdir(HERE) if f.startswith(PREFIX) or f.startswith("zz_s31_writer_clone_")]


# ---------- 绿路：每条都**跑命令**（坑⑩：Python 恒真式不算绿路），末尾由本件自己的现场闸把关 ----------
def g_zero_offset():
    """G1 正文区**零偏移**必须逐字节相同（C52／C53 的对照面：那条比对不是永红）。读数全现取。"""
    subj = sh("git", "log", "-1", "--format=%s").strip()
    prev = int(re.search(r"S7 分片 (\d+)", subj).group(1))
    curls = glob.glob(os.path.join(REPO, "docs/evidence/*/am_s7_verify%d.log" % prev))
    assert len(curls) == 1, "回执在全树命中 %d 份（须 1）⇒ G1 没有唯一所指" % len(curls)
    rawrs = glob.glob(os.path.join(os.path.dirname(curls[0]), "_raw_verify%d.out" % prev))
    assert len(rawrs) == 1, "原始 stdout 命中 %d 份（须 1）" % len(rawrs)
    off = int(re.search(r"BODY_OFFSET=(\d+)", rd(curls[0])).group(1))
    rc = subprocess.run(["cmp", "-i", "%d:0" % off, curls[0], rawrs[0]], capture_output=True).returncode
    assert rc == 0, "零偏移的正文区比对 rc=%s（须 0）⇒ 基线形状已变，那条比对是永红的" % rc
    return "cmp -i %d:0 -> 0" % off


def g_default_locale_fingerprint():
    """G2 不钉 locale 的那条命令必须**等于**回归件登记的 SRCFINGER（C77–C79 的对照面：这条闭合是可达的）。
    命令串从生成器源码现取（`FINGER_CMD`），不在本件里重敲配方。"""
    cmd = re.search(r'^FINGER_CMD = \(((?:\s*"[^"]*")+)\)$', text, re.M)
    assert cmd, "从生成器源码里取不到 FINGER_CMD 的字面拼接 ⇒ 本条绿路没有配方出处"
    frags = re.findall(r'"([^"]*)"', cmd.group(1))
    assert len(frags) >= 1, "FINGER_CMD 的相邻字面段数不认识：%r" % frags
    full = "".join(frags)  # 相邻字面量在 Python 里**无分隔**拼接，所以这里不能用 " ".join
    got = subprocess.run(["bash", "-c", full], cwd=REPO, capture_output=True, text=True,
                         env=dict(os.environ, CUDA_VISIBLE_DEVICES="")).stdout.strip()
    regpat = re.search(r'^REGS = sorted\(os\.path\.relpath\(p, REPO\) for p in glob\.glob\(os\.path\.join\(REPO, "([^"]+)"\)\)\)', text, re.M)
    assert regpat, "从生成器源码里取不到 REGS 的 glob 字面 ⇒ 本条绿路没有出处"
    logs = sorted(glob.glob(os.path.join(REPO, regpat.group(1))))
    assert len(logs) == 1, "回归件在全树命中 %d 份（须 1）：%s ⇒ 与生成器那条断言不同口径" % (len(logs), logs)
    want = re.search(r"^SRCFINGER_START: (\w{32})$", rd(logs[0]), re.M).group(1)
    assert re.fullmatch(r"[0-9a-f]{32}", got), "默认 locale 的指纹读数形状不认识：%r" % got
    assert got == want, "默认 locale 现算 %s ≠ 回归件登记 %s ⇒ 那条四路闭合按构造红不了（G2 失效）" % (got, want)
    return "默认 locale＝SRCFINGER_START＝%s" % got


def g_argv0_matches_itself():
    """G3 整词匹配对**本件自己**这条命令行必须为真（C39 的反面：这条尺子不是永零）。"""
    blob = open("/proc/%s/cmdline" % os.getpid(), "rb").read()
    assert blob.split(b"\0")[0].decode() == sys.executable, "本件 argv[0] 不是解释器 ⇒ 这条对照失去前提"
    tokens = [a.decode("utf-8", "replace") for a in blob.split(b"\0") if a]
    assert os.path.basename(__file__) in [os.path.basename(t) for t in tokens[1:]], "本件命令行里搜不到自己的件名"
    return "整词命中本件 argv[0]＝1（分母＝%d 个 argv 段）" % len(tokens)


def g_dpkg_single_ii():
    """G4 dpkg 的 ii／amd64 过滤必须留下**恰好一个**版本号（C32／C33 的对照面：那条门槛是可达的）。"""
    q = subprocess.run(["dpkg-query", "-W",
                        "-f=${Package}|${Architecture}|${db:Status-Abbrev}|${Version}\n",
                        "libnvidia-compute*"], capture_output=True, text=True)
    assert q.returncode == 0, "dpkg-query rc=%s ⇒ 本条绿路没有出处" % q.returncode
    pkgs = sorted({l.split("|")[3].split("-")[0] for l in q.stdout.splitlines()
                   if len(l.split("|")) == 4 and l.split("|")[1] == "amd64"
                   and l.split("|")[2] == "ii " and re.fullmatch(r"libnvidia-compute-\d+", l.split("|")[0])})
    assert len(pkgs) == 1, "ii／amd64 的版本有 %d 个（须 1）：%s" % (len(pkgs), pkgs)
    return "ii／amd64 唯一版本＝%s" % pkgs[0]


def g_this_face_is_ingested():
    """G5 本件必须已在**暂存**清单里（坑⑧的时序凭据：生成器那条「干跑控件面在入库清单里」的断言可达）。"""
    staged = [l for l in sh("git", "-c", "core.quotePath=false", "diff", "--cached", "--name-only").splitlines()
              if os.path.basename(l.strip()) == os.path.basename(__file__)]
    assert len(staged) == 1, "本件的原始件不在暂存清单（命中 %d 行）⇒ 先 git add 再跑本件" % len(staged)
    return "暂存清单含本件 1 行"


GREENS = [
    ("G1 正文区零偏移比对必须判 0（比对不是永红）", g_zero_offset),
    ("G2 默认 locale 的指纹必须等于回归件登记值（闭合可达）", g_default_locale_fingerprint),
    ("G3 整词匹配必须数出本件自己（红线不是永零）", g_argv0_matches_itself),
    ("G4 dpkg 的 ii／amd64 过滤必须留下恰好一个版本（门槛可达）", g_dpkg_single_ii),
    ("G5 本件必须已在暂存清单里（生成器那条入库断言可达）", g_this_face_is_ingested),
]

UNCOVERED = [
    ("片号四处互证里的「本轮回执已经存在」（`SLICE not in LOGS`）",
     "红路要在档案目录里真有一张 am_s7_verify31.log＝本片的回执；按形制它只能在本片落地之后存在，为造红预写"
     "一张假回执正是这条闸门要拦的事。它会在分片 32 变成活闸（那时 31 号已入库）。"),
    ("角色桶空转（`assert not empty`，「那个角色的命名口径已变」）",
     "桶形改窄时那些件必然先落进 0 个桶而撞上逐件断言（C9／C11 落的那条），轮不到「空桶」这一句；要它单独红"
     "得让一整角色的件还在清单里却不被任何桶认领＝改档案面命名而不是改仪器。"),
    ("`numstat 里有 - 行（二进制／rename）`、`kinds == [A, M]` 的 D 方向、面行数的「只允许净增」（`dele == 0`）",
     "三条都要往暂存集里放一张二进制面／一次删除，与本轮「只归档文本面、只净增」直接冲突；两列求和的机制已由 "
     "C21 覆盖，状态字母按位取行已由 C19 覆盖，`-` 行的检测本身没有本轮可造的坏输入。"),
    ("`-uall` 口径的三条（折叠目录行、未跟踪与落笔清单的交集、落笔清单重名）＋原 C69 的 needle 放宽",
     "本轮实测：`git status --porcelain -uall` 里以 `??` 开头但**不**以 `?? ` 开头的行＝0（未跟踪行永远是 `?? `"
     "＋路径），所以把 needle 从 `?? ` 放宽到 `??` 是**字面 no-op**、副本 rc＝0（第一版把它当红路是抄了上一轮的"
     "形状，那时未跟踪侧非空且含别的行形状）⇒ C69 已从 CASES 移除，不再算被验。另三条要工作树真的**未跟踪**且"
     "含整目录：到本件生成这一刻那批面已全部暂存（未跟踪侧只剩本件自己那两张 raw，现数 2）⇒ 空集上无法造非空"
     "交集。重名那条也按构造不成立（桶是逐件互斥的，C9 就是它的反例）。"),
    ("`assert not mo`（工作树有未暂存的修改面）",
     "红路要真留下一处未暂存漂移；同上一条同形——而且本件的 G5／C110 前后都验「工作树对暂存面零漂移」，那条"
     "比对在本轮是**实跑两次**的活判据（漂移一旦出现，本件与生成器都红）。"),
    ("`assert \"LC_ALL\" not in os.environ`（生成器 finger_both 的第一条）",
     "那是**环境**前提，不是源码形状：红路要往子进程环境里注入 LC_ALL，而注入之后本件自己那条「环境里不许有 "
     "LC_ALL」（坑④）会先停。本件 main() 的三条前置断言就是这条的替身。"),
    ("`nvidia-smi 本轮又能用了`（普查仪器恢复时的拒绝）",
     "红路要真的把内核模块与用户态库对齐（重载模块或装包），那是共享状态且会打断用户的 GPU 进程 ⇒ 本轮不做。"
     "替代证据＝G 族之外的一条实跑：C32／C33／C34／C36 全都在**真包表**上取数（含 rc 残留行与 i386 行），"
     "所以那条根因句的两个出处已被两个方向各验一次。"),
    ("模板里某一格的**语义**（两个读数都合法，只是填错了槽）",
     "本件的单位断言审的是「每条读数取自唯一出处」，不是「哪一格填了哪个字段」⇒ 填错槽位时副本可以 rc=0。"
     "本轮**实测过两次**：① 把 V 里 RAWRB（原始 stdout 的字节数）与 RAWRM（同件的 md5 前 12）互换——两个都是"
     "字符串、都过类型闸，副本 rc=0 且照旧打印整条信息＝这条红不了；② 把非字符串（staged 的 rows 列表）落进"
     "槽位——副本 rc≠0，红在「槽 CTLA 的取值不是字符串」，所以**形状**这一侧有闸（＝上面 C97），语义那一侧没有。"
     "①因此不放进 CASES（放进去就是一条永红不了的控制＝又一遍真件）。本轮新增的两处补强正是这一族里能被机器"
     "拦下的形状：`ncode == 0`（C18，那句旁边写着死的 0 条）与四个正对照读数的非零断言（C96c，模板那句"
     "「四个正对照都能红」现在数的是读数而不是修辞）；其余仍要靠分片 32 的归档自查腿逐条比对原文。"),
    ("写入件的两条拒绝：守卫那句「有别的仪器正在跑」与「wrapper 不参与判定」那一侧",
     "本轮实测**两句都造不出红**：C110 的克隆副本在守卫之前就先撞上写入件自己的落笔闭合式（78 vs 67），走不到"
     "那句守卫；要红守卫得让副本**通过**闭合式（＝档案面真落盘）——那需要把本片整条链先写进档案，而写入件一旦"
     "通过闭合式就会真改五张档案面。所以本轮 C110 改成登记它**实际**打红的那条闸门（现场不符时拒绝写盘），"
     "守卫那句留作分片 32 之后另立一轮的活（需要一份不会落盘的写入件替身）。"),
    ("`selfctl_check` 的两条「命中 1 行」门槛（本件自己的锚点句与标记句重复）",
     "本件不能改自己的源码来造重复（改了就不是这一版仪器）。这条由**基线**承担：一旦任一常量重复出现，生成器"
     "副本 rc≠0 ⇒ 本件在基线那一步就停（坑①），整轮正对照作废——预检里那两条自件对照（正 1 处／负 0 处）是它"
     "的花前份。"),
]


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
    hitl = [l for l in base_msg.splitlines() if SELF_BASE_MARK in l]
    exact = [l for l in hitl if l == SELF_BASE_MARK]
    # 那句在副本 stdout 里**按构造出现两次**（信息正文的那条新学 bullet＝逐字节相同的那一行；
    # 「诚实登记 ⑤」把它当作引用嵌在句子里＝带前后文的那一行）。所以判据是「至少一行逐字节相同」，
    # 不是「恰好一行命中」——本轮实跑第一次就撞在这条上（命中 2 行被判红），那是**本件的判据写错**而不是副本错。
    # 歧义风险由另外两条尺子承担：本文件里那句须恰好 1 行（下面的 `own`）＋生成器报的字节数与本件同值。
    assert len(exact) >= 1, "基线标记句在副本 stdout 里命中 %d 行、逐字节相同 %d 行（各须 ≥1）⇒ 副本没走到打印分支，或那句被包进了别的句子" % (
        len(hitl), len(exact))
    # 生成器 selfctl_check 把**本件源码里那一行的字节数**写进信息；本件自己数同一行，两个出处必须同值。
    own = [l for l in open(os.path.join(HERE, os.path.basename(__file__)), encoding="utf-8").read().splitlines()
           if SELF_BASE_MARK in l]
    assert len(own) == 1, "本文件里基线标记句命中 %d 行（须 1）⇒ 生成器 selfctl_check 会红" % len(own)
    ownb = len(own[0].encode())
    nb = int(re.search(r"标记句那行 (\d+) B", base_msg).group(1))
    assert nb == ownb, "生成器从本件源码取的标记行字节数 %d ≠ 本件自己数的 %d ⇒ 两边的「标记行」不是同一行" % (nb, ownb)
    assert not residue(), "扰动副本没被清掉：" + "、".join(residue())

    rows, allred, envfail = [], True, 0
    for name, subs, expect in CASES:
        if subs == "CLONE_WRITER":
            rc, msg = run_clone_writer()
        else:
            rc, msg, _ = run_one(perturb(name, subs))
        if rc == "TIMEOUT":
            envfail += 1
            print("  ！%s 子运行超过 %ds ⇒ 判环境故障，不判红也不判过" % (name, SUB_TIMEOUT))
            if envfail >= ENV_FAIL_STOP:
                print("CONTROL_VERDICT = ENV FAILURE（累计 %d 条超时，本件没资格判闸门）" % envfail)
                sys.exit(4)
            continue
        gl = gate_lines(expect)
        tl = trace_line(msg)
        err = raised(msg)
        hit = expect in err
        if gl and tl is not None:
            near = any(g - WINDOW <= tl <= g + WINDOW for g in gl)
            why = "句在源码 %s 行／raise 在 %d 行%s" % ("、".join(str(x) for x in gl[:3]), tl,
                                                    "（同一条闸门）" if near else "（不是同一条！）")
        else:
            near = True
            why = "句在源码 %s 处／raise 在 %s 行（位置判定不适用，只按抛出行判红）" % (len(gl), tl)
        ok = rc != 0 and hit and near
        allred = allred and ok
        if not ok:
            why += "｜实得抛出行：" + err[:200]
        rows.append((name, rc, ok, expect, why))

    # 对**本件自己**的闸门：锚点不存在时 perturb 必须拒绝。不验这条，下一轮克隆后「锚点已移位」的扰动
    # 会静默地没改动任何源码（＝又跑了一遍真件），红就变成假红。
    try:
        perturb("C-self-1 锚点普查（本件自己的闸门）", [(r"^    nvzz = 1$", "pass")])
        rows.append(("C-self-1 锚点普查（本件自己的闸门）", 0, False, "扰动锚点在源码里命中 0 处", ""))
        allred = False
    except AssertionError as e:
        okk = "扰动锚点在源码里命中 0 处" in str(e)
        rows.append(("C-self-1 锚点普查（本件自己的闸门）", 3 if okk else 0, okk,
                     "扰动锚点在源码里命中 0 处", "本件自己的异常"))
        allred = allred and okk

    grows = []
    for name, fn in GREENS:
        try:
            reading = fn()
        except (AssertionError, subprocess.SubprocessError) as e:
            grows.append((name, False, "绿路跑挂了：%s" % str(e)[:200]))
            allred = False
            continue
        grows.append((name, True, reading))
    res = residue()
    assert not res, "跑完后目录里有副本残留：" + "、".join(res)
    drift = unstaged()
    assert not drift, "整轮跑完工作树对暂存面出现漂移 %s ⇒ 本轮有控制真改了档案面" % drift
    allred = allred and not res and not drift

    print("基线（未扰动副本）rc=%s 且基线标记句逐字节命中＝True（所有红的对照面）" % base_rc)
    print("解释器＝%s｜被扰动的件＝%s｜每条子运行上限＝%ds｜环境故障累计上限＝%d 条" % (
        sys.executable, SRC, SUB_TIMEOUT, ENV_FAIL_STOP))
    print("CASES 声明＝%d 条（全部红路，其中克隆别的仪器 %d 条）｜保持绿＝%d 条｜本件自己的锚点普查＝1 条｜"
          "超时（环境故障，不判红也不判过）＝%d 条｜因超时没跑到的＝%d 条" % (
              len(CASES), sum(1 for _n, s, _e in CASES if s == "CLONE_WRITER"), len(GREENS), envfail, envfail))
    print("判红窗口＝raise 行号落在该句在本副本源码里**任一**出现处的 ±%d 行（坑③′：通用报错句在源码里两处，"
          "单点判定会把真红判成不合格）" % WINDOW)
    print("扰动条数＝%d｜每条判据＝rc≠0 且**抛出的那一行**含指定闸门句 且 行号落在窗口内" % (len(rows) + envfail))
    print("工作树对暂存面的漂移：跑前跑后各数一次，两次都为空＝那五张档案面本轮逐字节未动（G 族与 C110 的现场闸）")
    for name, rc, ok, expect, why in rows:
        print(("  " if ok else "× ") + name + "｜rc=" + str(rc) + "｜期望句「" + expect + "」"
               + ("命中" if ok else "**未命中**") + ("｜" + why if why else ""))
    print("绿路条数＝%d（每条都跑命令，末尾由现场闸把关；Python 恒真式不在此列）：" % len(grows))
    for name, ok, reading in grows:
        print(("  " if ok else "× ") + name + "：" + reading)
    print("本轮造不出红路的闸门＝%d 条（逐条附原因，不算被验）：" % len(UNCOVERED))
    for gate, why in UNCOVERED:
        print("  · " + gate + "：" + why)
    print("CONTROL_VERDICT =", "ALL CONTROLS CAN GO RED" if allred else "PROBLEM（有闸门空转）")
    sys.exit(0 if allred else 3)


if __name__ == "__main__":
    main()
