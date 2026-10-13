#!/usr/bin/env python3
"""本片落地核验件的**表头生成器**：表头每字段由命令现取，正文区逐字节等于核验件的原始 stdout。

片号、上一片回执名、本轮日志节号与**子条标签**、尺子份数**都从命令现取**（HEAD 提交说明／全树 glob／
开发日志正则），因为档案里记过一次实物事故（标本不在这里转述，由"教训"字段从最早那把生成器的原文
现取）：机械克隆只换得动"含旧片号的那一处"，其余注文原样留下 ⇒ 上一片的故事被安在本片头上。
文件名里的片号与 HEAD 的片号互相印证，不符就拒绝归档。

两条**已经固化进代码**的口径（不是本轮新增，本片沿用）：
① 尺子份数与"上一片回执"一律**全树 glob**；只 glob 自己那个日期目录会在跨日期抄放时数漏 ⇒
   旧口径的读数一并印出来（@@NDIR@@）作对照，而不是把它当结论。
② 节号先定位**最后一个** `### 26\\.\\d+` 标题，再在该节正文里按**子条形制**的正则现取；早先的版本用过
   "三级标题带字母"的正则，日志结构一变就**整条取空**（静默拿到上一轮的标签或直接抛）⇒ 下一轮若日志结构
   再变，这两处锚句必须跟着改，改完还要用控件面证明它们各自能红。**两种形制各自在哪些节里出现，本片不再
   凭叙述**：下面按全日志普查 `^- \\*\\*` 与 `^\\*\\*\\(x\\)` 两种行首形制，"旧形制最后出现的节"与
   形制的历史改问"旧形制最后出现在哪一节＋本节与上一片那一节各数出几条旧形制"（现取读数，见下面的按节普查）。
   "沿用"不等于"没验"：本件的控件面（同目录、同片号的 `*_hdr_controls.py`）仍要逐条造红，因为**锚句的语法
   会随档案的形状腐烂**，而腐烂的第一征兆是"取空"或"取到上一轮的东西"，不是报错。

本片（分片号见文件名，与 HEAD 提交说明互证）相对上一片的绑定改动只有三类，都由本文件的 diff 承担：
**四件路径常量的片号**、**"教训"字段 ⑤ 那段本轮口径**（换成本轮控件面实际撞到的那几族：一次真跑有 12 条扰动
没红在指定闸门上，标本留在档案里）、以及**LESSON 锚句本身**——上一片按 `对照的可满足性` 取它那节的口径子条，
本片那条是 `自审的方向性`（两个短语各自只命中一条顶级子条，与本片提交信息生成器用的那条 needle 同名同口径；
上一片那个短语**由本件从上一片的生成器源码里现取**，不是凭对话记忆，见 @@LESSONPREV@@）。锚句换了 ⇒ 它必须
重新被控件面逐条红一遍，"沿用"与"改过"都要证明不是空转。子条形制锚句本片**没动**。
其余注文若还写着上一片的节号／标签／份数，就是假读数——所以下面凡是数字一律现取，现取不到的只能引用
**能被命令查到的档案原文**。

为什么不用 f-string 承载正文（§26.27 (q) 那一片就是为此记了一条自纠）：f-string 会把字面量里的 `{32}`
当替换字段**静默吃掉**。⇒ 本件一律用 **PLACEHOLDER ＋ str.replace**，值只从命令读数来；且**占位符只替换
纯数值／原子读数**（§26.28 (i) 的自纠：整串 tally 塞进需要单值的中文句法位会语义错乱，而 cmp 全绿）。

自证五条：① 字段标签只允许出现在**行首**（粘连闸门，沿用 am_s18_hdr.py 的规则）；
② BODY_OFFSET 用**不动点迭代**求（写这一行本身会改变偏移量），不收敛即拒绝；
③ 写盘后实跑 `cmp -i <正文起点>:0`，rc≠0 就抛；
④ 表头不得残留 PLACEHOLDER；
⑤ 入库节奏（上一片回执链在本片 HEAD 里、本片回执链不在）由 `git show --name-status` 现取并断言。
"""
import glob
import os
import re
import subprocess
import sys

D = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(D)))
DAY = os.path.basename(D)
RAW = os.path.join(D, "_raw_verify31.out")
ERR = os.path.join(D, "_raw_verify31.err")
OUT = os.path.join(D, "am_s7_verify31.log")
SCRIPT_REL = "docs/evidence/" + DAY + "/am_s7_verify31.py"
FINGER_CMD = ("find src tests -name '*.py' -not -path '*__pycache__*' -print0 "
              "| sort -z | xargs -0 md5sum | md5sum | cut -d' ' -f1")


def sh(*args):
    return subprocess.run(args, cwd=REPO, capture_output=True, text=True, check=True).stdout


body = open(RAW, encoding="utf-8").read()
# 复跑核验件：既取 rc/stderr，又要求 stdout 与归档正文**逐字节相同**（＝仪器可复现的正对照）
ENV = dict(os.environ)
ENV["CUDA_VISIBLE_DEVICES"] = ""
rp = subprocess.run([sys.executable, os.path.join(REPO, SCRIPT_REL)], cwd=REPO, env=ENV,
                    capture_output=True, text=True)
rc_txt = str(rp.returncode)
err_bytes = len(rp.stderr.encode())
REPRO = rp.stdout == body
assert REPRO, "复跑 stdout 与归档正文不同 ⇒ 仪器不可复现，本件的判读行不能代表那次运行"

FIELDS = ["生成时刻", "CMD", "ENV", "脚本", "TREE", "SRC指纹", "GPU", "RUN", "判读",
          "四个正对照", "分母口径", "正文区", "复核", "同片对照", "用途", "教训"]

TMPL = """生成时刻: @@NOW@@
CMD     : 正文那条运行＝CUDA_VISIBLE_DEVICES= /tmp/amvenv/bin/python @@SCRIPT@@
          （stdout→_raw_verify@@SLICE@@.out，stderr→_raw_verify@@SLICE@@.err，两条都在本件里被复跑并逐字节比对；
          这两件由那次运行**直接写进本目录**（shell 重定向，无 /tmp 中转、无 cp -p）⇒ md5＝@@RAWM@@／@@ERRM@@
          就是那次运行的原始流，下面"首次归档运行"的时刻取的是这两件的 mtime，不是复制时刻）
          本表头由同解释器跑 am_s@@SLICE@@_hdr.py 生成，它内部又用同一环境复跑了上面那条命令一次。
          入库节奏**由 git 现取**（不是叙述）：上一片（@@PREVSLICE@@ 号）回执链在 HEAD 清单
          @@PREVCHAINHIT@@/@@CHAINN@@ 件，再上一片（@@PREV2@@ 号）链在其父提交 @@PARENTCHAINHIT@@/@@CHAINN@@ 件
          ＝一片一环的节奏在两片上都成立；本片的链（尺子／回执／两件原始件／本生成器）在 HEAD 清单
          @@OWNHIT@@/@@CHAINN@@ 件 ⇒ 本件留给分片 @@NEXTSLICE@@ 入库。
ENV     : CUDA_VISIBLE_DEVICES=（空串＝CPU 钉住；钉住口径**从档案现取**＝开发日志 §26.@@SECN@@ (@@GPULET@@)
          那句「@@GPUQUOT@@」，不是凭对话记忆转述）；
          本件**不 import jax**（只跑 git／GitHub API），带空串是习惯性钉住而不是必需，照实说明。
脚本    : @@SCRIPT@@（＝分片 @@RULER_FIRST@@ 那把尺子的**逐字节副本**：与 @@RULE18REL@@ 现比 md5＝@@MDEQ@@，
          本件 md5 @@MD5PY@@；**全树** docs/evidence/*/ 下 glob＋md5 现算＝同一把尺子已有 @@NRULE@@ 份逐字节副本
          （含本件），多片共用同一仪器，不是"为这一片重写一把尺子"。这条全树口径**不是本轮改的**（本轮沿用），
          但两个读数都现取并同印：只 glob 本目录的旧口径此处数出 @@NDIR@@ 份，全树数出 @@NRULE@@ 份 ⇒
          跨日期抄放时旧口径会数漏。这些数**现算**，写死它就随片腐烂）
TREE    : head=@@HEAD@@ parent=@@PARENT@@ dirty(src+tests)=@@DIRTY@@
SRC指纹 : @@FINGER@@（口径与全量回归件一致 ⇒ 核验时被测树就是本片落地的那棵树）
GPU     : 本表头生成时 nvidia-smi **仍起不来**（rc=@@NVRC@@，首行「@@NVWHY@@」）⇒ 「我没占卡」这条红线本轮没有它原来那把尺子，
          改由 /proc 的 argv[0] **整词**匹配承担：分母＝@@NVTOTAL@@ 个 PID，数到我的解释器（@@INTERP@@，排除本件与父进程）
          ＝@@NVME@@/@@NVTOTAL@@（须 0，非 0 本件当场拒绝归档）；两条按构造的对照＝合成 argv[0] 命中 @@CTLPOS@@／命令行
          **含**该路径而 argv[0] 是 /bin/bash 的命中 @@CTLNEG@@（须 1／0，否则整词匹配本身不可信）；同一条合成命令行
          再按**松散子串**口径量一次＝@@CTLLOOSE@@（须 1，与整词的 @@CTLNEG@@ 相对）⇒ "为什么不能用子串匹配"由这一对
          按构造的读数承担。现场松散口径本轮数出 @@NVLOOSE@@ 个（argv[0] 依次为 @@NVLOOSE0@@），而开发日志 §26.@@SECN@@
          那次记的是 @@NVLOOSELOG@@ 个 ⇒ 现场形状随轮而变，本件不把它当判据、只把上面那一对当判据。
          根因的两个版本读数（内核模块／用户态库）由本片提交信息生成器现取并写进提交说明，本表头不重复取、不另起口径。
RUN     : 表头生成时**复跑**核验件：rc=@@RC@@、stderr @@ERRB@@ 字节（0＝没有一条警告被吞），且复跑 stdout 与
          下面的正文区**逐字节相同**（本件写入前有断言）⇒ 判读行与正文是同一次运行的口径，非两次拼接。
          首次归档运行 @@DATE_START@@（其 stderr 留档案 @@ERRF@@ 字节），复跑于 @@DATE_END@@。
判读    : 远端 main == 本地 HEAD；tree 条目 @@ENTRIES@@ = blob @@NB@@ + tree @@NTREE@@；
          闭合式 远端 @@NB@@ == 父提交 @@NP@@ + 本片新增(A) @@NA@@ ＝ @@EXPECT@@ -> PASS；
          双向路径差集 @@DL@@/@@NB@@ 与 @@DR@@/@@NB@@；全库 blob SHA 不符 @@MS@@/@@NSHARED@@；
          本片单列复核 @@SLICE_MATCH@@/@@NFILES@@ MATCH（A @@NA@@／M @@NM@@ 个文件）。
四个正对照: A 远端一条 sha 首位改 f ⇒ 不符 @@CA@@/@@CBA@@；B 远端删一条路径 ⇒ 本地独有 @@CB1@@/@@CDB1@@、远端独有 @@CB2@@/@@CDB2@@；
          C 闭合式右边 +1 ⇒ @@NB@@ == @@NBP@@ 判 FAIL；D 远端删**本片**一条路径 ⇒ 本片缺失 @@CD1@@/@@NFILES@@、SHA 不符 @@CD2@@/@@NFILES@@。
          四条读数一律从正文 ⑤ 段正则现取（本表头不许自己造数），逐条都能红、所以不是空转。
分母口径: 每个 0 都与其分母同印（@@DL@@/@@NB@@、@@MS@@/@@NSHARED@@、@@SLICE_MATCH@@/@@NFILES@@），
          没有"光秃秃的 0"。
正文区  : 下面 BODY_OFFSET 之后＝核验件原始 stdout，逐字节未改（含 ①〜⑤ 与 LANDING_VERDICT 行）。
复核    : 在仓库根跑 cmp -i @@BOFF@@:0 @@OUTREL@@ @@RAWREL@@   （rc=0 才算正文区逐字节相同；两侧偏移各 +1 必须变红）
同片对照: 本片新增 @@NA@@ 件，与 /tmp **同名件现比 md5**（本表头自己数，不抄任何一份档案里的件数）＝
          相同 @@MDSAME@@ 件／不同 @@MDIF@@ 件（@@MDIFES@@）／/tmp 已无同名件 @@MDAB@@ 件（临时件被清理或当初就没落在 /tmp）。
          @@MDNOTE@@⇒ 这条只作辅助溯源，落地证明以正文区为准。
用途    : S7 分片 @@SLICE@@ 的落地回执（片号由 HEAD 提交说明现取，并与本件文件名 am_s7_verify@@SLICE@@.py
          互证，两边不符即拒绝归档 ⇒ 机械克隆不再能把上一片的故事安在本片头上）。入库清单由
          `git show --name-status HEAD` 现取（本表头自己数、自己列名）＝
          新增 @@NAA@@ 件（@@ADDBAS@@）＋修改 @@NMOD@@ 件（@@MODBAS@@）；逐条 SHA 复核在正文区 ④ 段。
          本轮开发日志 §26.@@SECN@@ (@@SECLET@@) 引用的回执是**上一片**的 @@PREVLOG@@（@@PREVSLICE@@ 号）；
          这条"哪一子条引用了哪份回执"不是手打的：先现取最后一个 `### 26\\.\\d+` 标题（＝@@SECN@@），
          再在该节正文里找**唯一**含 @@PREVLOG@@ 的子条标签（本轮日志的子条形制是行首 `- **标签**`，
          不是上一轮 §26.29 的行首 `**(x)**` ⇒ 取子条的正则随形制改过，见"教训"(b)）；对本件（am_s7_verify@@SLICE@@.log）的日志引用
          要到下一轮才出现，本表头不预先声称它已被引用。
教训    : ① 仪器复用＝同一把尺子量多片，比对才有效；② 闭合式的右边含本片新增数，所以它随片而变、不是恒等式：
          分片 @@RULER_FIRST@@ 的档案印「@@S18@@」、本片印「远端 @@NB@@ == 父提交 @@NP@@ ＋ 新增 @@NA@@ ＝ @@EXPECT@@」，
          两条各自被控制 C（右边 +1）证过"能红"；但闭合式只数**个数**，看不见"少一个＋多一个"的形状，
          所以必须与双向路径差集、全库 SHA 不符两条**同读**，三条一起才构成落地证明；③ 表头不用 f-string
          承载（@@FSSEC@@ 的成因），改用 PLACEHOLDER＋replace，写盘后再查残留；④ 本件由上一片的生成器
          **机械克隆**（sed 只换得出文件名里的片号）而来 ⇒ 凡"来历／片号／节号与子条标签／份数"写死在注文里的
          都会随片腐烂，diff 不会自动改它。事故标本**从档案里现取**（不是凭记忆转述）：生成器
          `@@HDR_SPEC@@` 的"教训"行原文印着「@@SPEC@@」——同一句里外层片号 @@SPECOUT@@、括号里
          @@SPECIN@@，两个数不同 ⇒ 那句故事说的是 @@SPECIN@@ 号那一片，而写它的生成器是 @@SPECOUT@@ 号自己的。
          本片的防法不是手改文字，而是四条**现取＋互证**闸门：(a) 片号从 HEAD 提交说明现取，并与本件文件名
          互证（两边都＝@@SLICE@@）；(b) 日志节号从开发日志现取最后一个 `### 26\\.\\d+`（＝@@SECN@@），
          子条**标签**再由"该节里唯一引用上一片回执的那一条"现取（＝@@SECLET@@）——早先版本用过"三级标题带
          字母"的正则，日志结构一变它就**整条取空**（静默拿到上一轮的标签或直接抛），所以口径要先于读数复查；
          两种形制各自落在哪些节**不由本件叙述**：按节普查行首形制给出"旧形制 `**(x)**` 最后出现的节"＝§26.@@FORMOLDLAST@@，
          上一片那一节（节号从上一片回执表头那句「本轮开发日志 §26.NN」现取）＝§26.@@PREVSECN@@，本节＝§26.@@SECN@@；
          两节的形制条数现数＝上一片 新@@PSECNEW@@／旧@@PSECOLD@@ 条，本片 新@@FSECNEW@@／旧@@FSECOLD@@ 条 ⇒ 两片都只用
          新形制（旧形制 0 条），子条形制锚句**没改**；"没改"不等于"没验"，本件的控件面仍逐条造红：形制指错（把锚句指回
          `**(x)**` ⇒ 取空）、标签引用不唯一（歧义）、引用查无、节号写死，四类坏输入各命中一条指定闸门句——
          **沿用一条口径的第一风险，是它悄悄变成空转**。
          **本轮确实改了一处锚句**：LESSON 那条从上一片的 `@@LESSONPREV@@` 换成本节的 `@@LESSONNEEDLE@@`（上一片那个短语
          由本件从上一片那把生成器的源码里现取，不是凭对话记忆）⇒ 换过的锚句同样要证明它抓得住：把 needle 指歪
          （换成日志里查无此句的短语）必须让本件当场拒绝，needle 命中两条时必须按"出处不唯一"拒绝，而不是静默拿到第一条。
          (c) 入库清单从 `git show --name-status HEAD` 现取，并与正文区 ④ 段的 A／M 计数互证（@@NAA@@/@@NMOD@@）；
          (d) 尺子份数与上一片回执**按全树 glob**（本轮沿用该口径，两个读数都现取）：旧"本目录"口径
          在本片只数出 @@NDIR@@ 份 ⇒ 会把"一把尺子
          量多片"印成"本片自带一把新尺子"，比对当场失去意义。教训一句话：**数字一律现取，文字也要；不能现取的文字就只能引用一份能被命令查到的档案原文。**
          ⑤ 本轮口径（引句从 §26.@@SECN@@ (@@LESSONLET@@) 现取，本轮落地子条随后并入同一节）：
          **自审的方向性**——从上一片的落地半腿带进本节（§26.@@SECN@@ 里那条子条就是它的出处，标签由本件现取
          ＝@@LESSONLET@@；那句口径的原文也从同一条子条现取＝「@@LESSONQUOT@@」）。本片它命中六处，
          六处都在**控件面自己**身上：
          ⓪ 判"红"的那段代码**先红了一次**：第一次跑纸面预检就拒绝（标本 @@SPEC1@@，那一版的 CASES 声明＝@@KC1@@ 条），
          它"期望句在源码里必须只出现一次"的假设被 C1 打破——同一个短语在生成器里现数 @@KTWICE@@ 处（第
          @@KLINE1@@ 与 @@KLINE2@@ 行，两个行号取自标本那行自己的方括号读数）⇒ 位置口径改成"任一出现处 ±3"之后才允许真跑；
          ⑴ **扰动没有真的移动匹配集**（四条）：C5 把回执 glob 的片号 `%d` 松成 `%s`，而那条 glob 本来就带 `*`
          ⇒ 两种写法命中同一批文件、副本 rc=0；C69 把未跟踪行 needle 从 `?? ` 收成 `??`，本表头生成时同一份
          porcelain（`-uall`）现数＝@@KQQ@@ 行未跟踪，其中"以 `??` 开头但不以 `?? ` 开头"的＝@@KQQDIFF@@ 行 ⇒ 两种写法
          本轮数同一批（这条经实测确认本轮**造不出红**，与其余各条一并进 UNCOVERED，共 @@KUNCOV@@ 条）；
          C63／C96 的第一版是"删掉闭合式里的一项比较"⇒ 那是**放宽**判据，坏输入照旧
          通过，换成反向比较才让合法集合真的变小；C77 把钉 locale 改钉 en_US.UTF-8，而它不是本机默认 ⇒ 两条读数
          照样不同、耦合检测照旧绿，改成"把两条都钉成 LC_ALL=C"才红；C18 把无代码腿普查的作用域指到 docs ⇒ 那条
          闸门本轮对它无话（rc=0），因为 docs 确实被动过，坏输入不构成坏输入；
          ⑵ **扰动落点里没有那个变量**：C22／C23 都引用 `added`，而它是 `build()` 的局部、在 `faces_check`
          作用域里根本不存在 ⇒ 抛的是 `NameError` 而不是闸门句；C22 改到调用处（把实参换成新增清单），
          C23 改用作用域里真有的 `pinned_faces`（relpath 与 basename 两种形状必不相等）；
          ⑶ **期望句不是原文／落点不是那条**：C57 的期望句我多打了一个"的"字（闸门句必须逐字节回抄源码原句）；
          C17 第一版"不切到分号"实际先撞在前面的「修改面 ≠ §26.33 登记的那批」而不是本轮想指的那条 ⇒ 落点按
          "这个坏输入真的先撞哪条"重新指认，坏输入本身换成"把清单分隔符 `／` 改成 `；`"；
          ⑷ **克隆腿的抛序**：C110 克隆本归档的写入件并带 `--write` 跑它，本意是让"有别的仪器正在跑"那条守卫拦停，
          实测它先撞自己那句闭合式 ⇒ 期望句改登记**实际打出的那一条**，守卫那一侧进 UNCOVERED（不许凭本意声称验过）；
          ⑸ **绿路也要有配方出处**：G2 那条"默认 locale 的指纹＝回归件登记值"起初命中 2 份日志 ⇒ 断言的分母自己
          就错位；改成从生成器源码现取 `REGS` 那一行的 pattern，glob 与仪器共用同一条口径。
          第一次真跑共 @@NX@@ 条没红在指定闸门上（标本 @@SPEC2@@，判读行「@@VERD2@@」，顶格「× 」行现数＝@@NX@@ 条；
          上面点名的控件号集合与该标本的 × 行集合**在本件里由一条断言互证**，不等即拒绝归档 ⇒ 编号不可能是我编的）；
          那一版 CASES 声明＝@@KC2@@ 条、扰动条数＝@@KD2@@ 条。重新指认后重跑的那张 @@SPEC3@@（CASES 声明＝@@KC3@@ 条｜
          保持绿＝@@KGREEN@@ 条｜超时＝@@KTO@@ 条｜扰动条数＝@@KD3@@ 条）判读行「@@VERD3@@」才是本件引用的全过面，
          它对自身锚点的那条普查（@@KRECASE@@ 那条"超时必须为零的链改成容忍一条 ⇒ 环境故障被当全过"必须红）就是
          "判红的方式自己也造一次红"的落地形状。⇒ 一条"从没红过的闸门"和"坏闸门"读数相同；控件面必须逐条造红，
          而"这条造不出红"必须是一个允许登记、且带原因的结果，不是假装验过。
          另有一条从上一片沿用到本片的**流程**侧实测：编辑一张**已暂存**的仪器 ⇒ 未暂存漂移闸门先打掉未扰动基线、
          整轮控件作废 ⇒ "跑控件面之前先把控件面自己 `git add`"是那条闸门的时序前提（本片每改一次就重新 `git add`
          一次，本件引用的那张控件面与它当时读到的源码是同一份暂存内容）。
          （表头写盘后仍要**回读**：字段行数、残留占位符、cmp 三件套——语义错乱在 cmp 里是全绿的。）
BODY_OFFSET=@@BOFF@@
"""


def md5(path):
    return subprocess.run(["md5sum", path], capture_output=True, text=True).stdout.split()[0]


# ---------- GPU 红线：本轮 NVML 那把尺子起不来（§26.@@SECN@@ 那条环境级新学），改由 /proc 的 argv[0] 整词匹配 ----------
# 本件不重算根因（内核模块与用户库那两个版本读数由本片提交信息生成器现取并写进提交说明），只做三件事：
# ① 断言 nvidia-smi **仍然坏着**——它若恢复，本表头那句「普查仪器坏了」就成了假话；
# ② 全 /proc 数分母，argv[0] **整词**等于本解释器路径的 PID 必须为 0（排除自己与父进程），非 0 就拒绝归档；
# ③ 一正一负对照：合成的 argv[0] 必须数出 1；命令行**含**该路径而 argv[0] 不是它的必须数出 0。
INTERP = sys.executable


def argv0(blob):
    return blob.split(b"\0")[0].decode("utf-8", "replace")


def cmdline(pid):
    try:
        blob = open("/proc/%s/cmdline" % pid, "rb").read()
    except OSError:
        return None
    return blob or None


nv = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                    capture_output=True, text=True)
assert nv.returncode != 0, "nvidia-smi 本轮又能用了（rc=%d）⇒ 本表头那句「普查仪器坏了」要改写" % nv.returncode
NVWHY = (nv.stdout + nv.stderr).strip().splitlines()[0]
assert "NVML" in NVWHY, "nvidia-smi 的失败原因换了：%r ⇒ 那句新学要重取" % NVWHY
nvpids = [d for d in os.listdir("/proc") if d.isdigit()]
me, myparent = os.getpid(), os.getppid()
nv_mine, nv_loose, nv_loose0 = [], [], []
for d in nvpids:
    if int(d) in (me, myparent):
        continue
    blob = cmdline(d)
    if blob is None:
        continue
    if argv0(blob) == INTERP:
        nv_mine.append(d)
    flat = blob.decode("utf-8", "replace").replace("\x00", " ")
    if INTERP in flat:
        nv_loose.append(d)
        nv_loose0.append(argv0(blob))
# 两条对照都按**构造**判，不靠现场运气：正＝合成 argv[0] 命中；负＝含该路径但 argv[0] 是 /bin/bash。
# 同一条合成命令行再拿**松散子串**口径量一次（松散 1／整词 0）：这才是"为什么不能用子串匹配"的证据。
# 现场松散口径数出几个**随轮而变**，所以它只作对照印出（NVLOOSE／NVLOOSELOG 两个读数），不进判据。
SYN_BLOB = b"/bin/bash\0-c\0grep " + INTERP.encode() + b"\0"
CTLPOS = 1 if argv0((INTERP + "\0x.py").encode()) == INTERP else 0
CTLNEG = 1 if argv0(SYN_BLOB) == INTERP else 0
CTLLOOSE = 1 if INTERP in SYN_BLOB.decode("utf-8", "replace").replace("\x00", " ") else 0
assert (CTLPOS, CTLNEG) == (1, 0), "argv[0] 整词匹配的两条对照失效（正 %d／负 %d）⇒ 这条红线不可信" % (CTLPOS, CTLNEG)
assert (CTLLOOSE, CTLNEG) == (1, 0), "松散口径在那条合成命令行上没有误指（%d/%d）⇒ 「松散会把 shell 也认进来」要改写" % (
    CTLLOOSE, CTLNEG)
# 原 nvidia-smi 口径下「我的进程」这件事本轮**量不到**：印成 0 个是把仪器坏了当成没有进程。
assert not nv_mine, "argv[0] 整词匹配数到我的解释器进程 %s（分母＝%d 个 PID）⇒ 「GPU 全程未碰」这条红线不成立，拒绝归档" % (
    nv_mine, len(nvpids))
# 未跟踪行 needle 的两条写法在本轮是不是同一件事（⑤ 那条「扰动没移动匹配集」的实测由这两个读数承担）
_por = sh("git", "-c", "core.quotePath=false", "status", "--porcelain", "-uall")
QQ = len([l for l in _por.splitlines() if l.startswith("??")])
QQDIFF = len([l for l in _por.splitlines() if l.startswith("??") and not l.startswith("?? ")])
assert QQDIFF == 0, "porcelain 里有 %d 行以 `??` 开头却不以 `?? ` 开头 ⇒ 「needle 放宽是 no-op」那句要改写" % QQDIFF

# 同一把尺子在全树已有几份（glob＋md5 现算）：写死"两份/三份"会随片腐烂，本目录口径会数漏别的日期目录
ruler_md5 = md5(os.path.join(REPO, SCRIPT_REL))
ruler_copies = sorted(p for p in (os.path.relpath(x, REPO) for x in
                                  glob.glob(os.path.join(REPO, "docs/evidence/*/am_s7_verify*.py")))
                      if re.fullmatch(os.path.join("docs", "evidence", "[^/]+", r"am_s7_verify\d+\.py"), p)
                      and md5(os.path.join(REPO, p)) == ruler_md5)
assert SCRIPT_REL in ruler_copies, "本件的尺子副本没被自己的 glob 数到 ⇒ 计数口径失真"
ruler_nums = sorted(int(re.fullmatch(r"am_s7_verify(\d+)\.py", os.path.basename(p)).group(1))
                    for p in ruler_copies)
RULER_FIRST = str(ruler_nums[0])
RULE18REL = [p for p in ruler_copies if os.path.basename(p) == "am_s7_verify%s.py" % RULER_FIRST][0]
# 旧口径（只数本目录）的读数一并印出来，作为"为什么必须改全树"的实证而不是主张
ruler_dir_copies = [p for p in ruler_copies if os.path.dirname(p) == os.path.relpath(D, REPO)]
NDIR = str(len(ruler_dir_copies))

# 本片新增件里有多少是 /tmp 原件的逐字节副本（现比现取，不抄开发日志的数）
rows = [r.split("\t") for r in sh("git", "-c", "core.quotePath=false", "show",
                                  "--name-status", "--format=", "HEAD").splitlines() if r.strip()]
added = [r[-1] for r in rows if r[0] == "A"]
mods = [r[-1] for r in rows if r[0] == "M"]
assert len(added) + len(mods) == len(rows), "入库清单里有 A/M 之外的状态字母 ⇒ 上面的取列口径破了"
md_same, md_diff, md_absent = [], [], []
for p in added:
    t = os.path.join("/tmp", os.path.basename(p))
    if not os.path.exists(t):
        md_absent.append(p)
    elif md5(os.path.join(REPO, p)) == md5(t):
        md_same.append(p)
    else:
        md_diff.append(p)
assert len(md_same) + len(md_diff) + len(md_absent) == len(added), "三分类没覆盖全部新增件 ⇒ 有一条既不算同也不算异也不算缺"
# "不同"的那件**不写死来历**：上一片把它写成一句具体故事，这一片的差集若换了名字，那句故事就成了假话
# ⇒ 读数（件名、件数）一律现取，注文只说"两版逐字节不同"这一件能被 md5 证明的事实。
if md_diff:
    MDNOTE = ("其中逐字节**不同**的是 " + "、".join(os.path.basename(p) for p in md_diff)
              + "：/tmp 那份与档案那份是两个版本（差异由 md5 现比判出，来历不在本表头里编）。")
else:
    MDNOTE = "没有一件不同 ⇒ /tmp 现存的同名件全部逐字节相同。"

# 入库清单两个出处互证：正文区 ④ 段（核验件从远端树＋git 现算）与本表头（本地 git show）
NA_BODY = int(re.search(r"新增 (\d+) \+ 修改", body).group(1))
NM_BODY = int(re.search(r"新增 \d+ \+ 修改 (\d+)", body).group(1))
assert (NA_BODY, NM_BODY) == (len(added), len(mods)), (
    "核验件印出的入库数 %s/%s ≠ 本地 git show 实得 %s/%s ⇒ 有一侧的清单口径变了，拒绝归档"
    % (NA_BODY, NM_BODY, len(added), len(mods)))

# ---------- 片号／节号／上一片回执／事故标本／入库节奏：闸门全靠现取，防机械克隆张冠李戴 ----------
SUBJ = sh("git", "log", "-1", "--format=%s").strip()
SLICE = re.search(r"S7 分片 (\d+)", SUBJ).group(1)
assert os.path.basename(SCRIPT_REL) == "am_s7_verify%s.py" % SLICE, \
    "HEAD 提交说明的片号 %s ≠ 本件所验尺子 %s ⇒ 上一片的故事安在本片头上，拒绝归档" % (
        SLICE, SCRIPT_REL)
NEXTSLICE = str(int(SLICE) + 1)

# 回执档案也改全树 glob（本目录口径在跨日期的片上取不到上一片回执，会直接抛）
RCPT_REL = sorted(p for p in (os.path.relpath(x, REPO) for x in
                              glob.glob(os.path.join(REPO, "docs/evidence/*/am_s7_verify*.log")))
                  if re.fullmatch(os.path.join("docs", "evidence", "[^/]+", r"am_s7_verify(\d+)\.log"), p))
RCPTS = sorted(int(re.fullmatch(r"am_s7_verify(\d+)\.log", os.path.basename(p)).group(1)) for p in RCPT_REL)
assert len(RCPTS) == len(set(RCPTS)), "回执档案里有重号角本"
PREVSLICE = max(n for n in RCPTS if n != int(SLICE))
PREVLOG = "am_s7_verify%d.log" % PREVSLICE
assert int(SLICE) == PREVSLICE + 1, \
    "本片 %s 与上一片回执 %s 不连号 ⇒ 有一片没落，或片号口径变了" % (SLICE, PREVSLICE)
assert PREVLOG in [os.path.basename(p) for p in RCPT_REL], "取到的上一片回执 %s 不在全树档案里" % PREVLOG

# 开发日志：先现取最后一个 ### 26.NN 标题，再在该节正文里按**本轮的子条形制**现取标签
# （§26.29 的行首形制是 `**(x)**`，§26.30 换成 `- **标签**` ⇒ 两处口径随形制改，改完由控件面证它能红）
DEVLOG = os.path.join(REPO, "docs/开发日志.md")
DEV = open(DEVLOG, encoding="utf-8").read()
_secs = list(re.finditer(r"^### 26\.(\d+)[^\n]*$", DEV, re.M))
assert _secs, "开发日志里取不到 `### 26.NN` 标题 ⇒ 节号是凭记忆的"
SECN = _secs[-1].group(1)
SEC_BODY = DEV[_secs[-1].end():(_secs[-1].end() + (DEV.find("\n### ", _secs[-1].end())
                if DEV.find("\n### ", _secs[-1].end()) != -1 else len(DEV)))]
BULLETS = [m.group(1) for m in re.finditer(r"^- \*\*(.+?)\*\*", SEC_BODY, re.M)]
assert BULLETS, "§26.%s 正文里取不到行首 `- **标签**` 子条 ⇒ 本轮日志形制又变了（口径要先于读数复查）" % SECN
assert len(BULLETS) == len(set(BULLETS)), "§26.%s 里有同名子条标签 ⇒ 由标签切段会串行" % SECN


def bullet_for(pattern, what):
    """返回**唯一**其正文匹配该模式的子条标签与正文；不唯一或没有就拒绝（引用必须有唯一出处）。"""
    outs = []
    for i, let in enumerate(BULLETS):
        start = SEC_BODY.index("- **%s**" % let)
        end = (SEC_BODY.index("- **%s**" % BULLETS[i + 1]) if i + 1 < len(BULLETS) else len(SEC_BODY))
        seg = SEC_BODY[start:end]
        if re.search(pattern, seg):
            outs.append((let, seg))
    assert len(outs) == 1, "%s 在 §26.%s 里匹配到 %d 条子条 ⇒ 引用出处不唯一，拒绝写死" % (
        what, SECN, len(outs))
    return outs[0]


SECLET, _SECSEG = bullet_for(re.escape(PREVLOG), "上一片回执名")
GPUQUOT_M = re.search(r"当前「([^」]+)」", SEC_BODY)
assert GPUQUOT_M, "§26.%s 里取不到「当前「…」」这种钉住口径原文" % SECN
GPULET, _GPUSEG = bullet_for(r"当前「" + re.escape(GPUQUOT_M.group(1)) + r"」", "钉住口径")
GPUQUOT = GPUQUOT_M.group(1)
# 现场松散口径的"上一轮读数"也从档案现取（本件不转述对话记忆）：它和本轮的实测**可以不同**，
# 所以这条只作对照印出，不进判据（判据是上面那一对按构造的读数）。
_mloose = re.search(r"松散子串匹配现数 (\d+) 个", SEC_BODY)
assert _mloose, "§26.%s 里取不到「松散子串匹配现数 N 个」这个读数 ⇒ GPU 那段少了第二出处" % SECN
NVLOOSELOG = _mloose.group(1)
LESSON_NEEDLE = "自审的方向性"
LESSONLET, _LSEG = bullet_for(LESSON_NEEDLE, "自审的方向性这条从上一片带进的口径")
# 上一片那条 LESSON 锚句**从上一片的生成器源码现取**（不凭对话记忆）：本片的注文要说"换过"，就得有两个出处。
_prev_hdrs = [p for p in (os.path.relpath(x, REPO) for x in
                          glob.glob(os.path.join(REPO, "docs/evidence/*/am_s%d_hdr.py" % PREVSLICE)))
              if re.fullmatch(os.path.join("docs", "evidence", "[^/]+", r"am_s%d_hdr\.py" % PREVSLICE), p)]
assert len(_prev_hdrs) == 1, "上一片（%s 号）的表头生成器在全树命中 %d 份（须 1）：%s ⇒ 那句「换过锚句」没了出处" % (
    PREVSLICE, len(_prev_hdrs), _prev_hdrs)
_pm = re.search(r'LESSONLET, _LSEG = bullet_for\(r"([^"]+)"',
                open(os.path.join(REPO, _prev_hdrs[0]), encoding="utf-8").read())
assert _pm, "上一片生成器里取不到 LESSON 锚句那行 ⇒ 拒绝声称「换过」"
LESSONPREV = _pm.group(1)
assert LESSONPREV != LESSON_NEEDLE, "上一片的锚句与本片的相同（都是 %s）⇒ 那句「从 X 换成本节的 Y」是假话" % LESSONPREV

# 子条**形制**的历史也不凭叙述：按节普查两种行首形制的**条数**（早期节里偶尔冒出一条 `- **…**` 的的项目符号，
# 所以"新形制最早出现在哪一节"这个问法会把 §26.5 那样 incidental 的一条读成"起用的节"⇒ 本件改问"本节与上一片
# 那一节各自数出几条旧形制"，两片都是 0 条才叫同形制。）
_FMC = {}
for _i, _m in enumerate(_secs):
    _end = _secs[_i + 1].start() if _i + 1 < len(_secs) else len(DEV)
    _FMC[_m.group(1)] = (len(re.findall(r"^- \*\*(.+?)\*\*", DEV[_m.end():_end], re.M)),
                         len(re.findall(r"^\*\*\((\w)\)", DEV[_m.end():_end], re.M)))
FORMOLDLAST = str(max(int(n) for n, (_new, old) in _FMC.items() if old > 0))
assert SECN in _FMC, "本节 §26.%s 不在按节普查的表里 ⇒ 形制条数没有出处" % SECN
# 上一片那一节的**节号**从上一片的回执表头现取（它自己在"用途"字段里印着「本轮开发日志 §26.NN」）
_prev_rcpt = [p for p in RCPT_REL if os.path.basename(p) == PREVLOG]
assert len(_prev_rcpt) == 1, "上一片回执 %s 在全树命中 %d 份（须 1）⇒ 那一节的节号取不到唯一出处" % (PREVLOG, len(_prev_rcpt))
_psecs = sorted(set(re.findall(r"本轮开发日志 §26\.(\d+)",
                               open(os.path.join(REPO, _prev_rcpt[0]), encoding="utf-8").read())))
assert len(_psecs) == 1, "上一片回执表头里「本轮开发日志 §26.NN」现取到 %d 个不同节号（%s）⇒ 出处不唯一" % (
    len(_psecs), _psecs)
PREVSECN = _psecs[0]
assert PREVSECN in _FMC, "上一片那一节 §26.%s 在开发日志里普查不到 ⇒ 形制对比没有分母" % PREVSECN
FSECNEW, FSECOLD = _FMC[SECN]
PSECNEW, PSECOLD = _FMC[PREVSECN]
assert int(SECN) > int(PREVSECN) > int(FORMOLDLAST), "本节 %s／上一片 %s／旧形制最后出现的节 %s 不成序 ⇒ 那句「两片同用新形制」没有依据" % (
    SECN, PREVSECN, FORMOLDLAST)
assert (FSECOLD, PSECOLD) == (0, 0), "本片这节与上一片那节里还数得出旧形制子条（%d／%d 条）⇒ 「同形制、锚句没改」是假话" % (
    FSECOLD, PSECOLD)


# f-string 那一条成因的出处：整份日志里定位该句，再回推它所属的标题＋子条字母（不写死 §26.27 (q)）
_fs = list(re.finditer(r"f-string 把 `\{32\}` 当成\*\*替换字段", DEV))
assert len(_fs) == 1, "f-string 事故句在开发日志里出现 %d 次 ⇒ 出处不唯一" % len(_fs)
_fsidx = _fs[0].start()
_fshead = [m for m in re.finditer(r"^### (26\.\d+)[^\n]*$", DEV, re.M) if m.start() < _fsidx][-1]
_fslet = [m for m in re.finditer(r"^\*\*\((\w)\)", DEV, re.M) if m.start() < _fsidx]
FSSEC = "§" + _fshead.group(1) + (" (" + _fslet[-1].group(1) + ")" if _fslet else "")

# 事故标本**从档案里现取**（不凭记忆转述）：最早那份记着"改之前"原话的生成器
_spec_candidates = [p for p in (os.path.relpath(x, REPO) for x in
                                 glob.glob(os.path.join(REPO, "docs/evidence/*/am_s*_hdr.py")))
                    if re.search(r"本片的实物证据＝\*\*改之前\*\*那句", open(os.path.join(REPO, p),
                                                                            encoding="utf-8").read())]
assert len(_spec_candidates) == 1, "标本原文所在生成器有 %d 份 ⇒ 出处不唯一，拒绝" % len(_spec_candidates)
HDR_SPEC = _spec_candidates[0]
_cap = re.search(r"本片的实物证据＝\*\*改之前\*\*那句「(.+?)」",
                 open(os.path.join(REPO, HDR_SPEC), encoding="utf-8").read())
SPEC = _cap.group(1)
_g = re.search(r"分片 (\d+)（分片 \*\*(\d+)\*\*", SPEC)
assert _g, "标本那句里没有「外层片号＋括号里旧片号」两个数 ⇒ 标本不是那次事故：" + SPEC
SPECOUT, SPECIN = _g.group(1), _g.group(2)
assert SPECOUT != SPECIN, "标本那句两个数相同 ⇒ 事故不存在，这条引用是空转"

# 入库节奏：回执链＝尺子／回执／两件原始件／表头生成器（5 件），全部按片号现拼、现查 git
CHAIN_RE = re.compile(r"^am_s7_verify(\d+)\.(py|log)$|^_raw_verify(\d+)\.(out|err)$|^am_s(\d+)_hdr\.py$")


def chain_ids(n, name_status_rows):
    n = str(n)  # PREVSLICE 是现取的 int：int 与 regex 分组（字符串）永不相等 ⇒ 不归一就静默数出 0
    hit = []
    for c, p in name_status_rows:
        b = os.path.basename(p)
        m = CHAIN_RE.match(b)
        if m and n in m.groups():
            hit.append(b)
    return sorted(set(hit))


prev_rows = [r.split("\t") for r in sh("git", "-c", "core.quotePath=false", "show",
                                       "--name-status", "--format=", "HEAD").splitlines() if r.strip()]
parent_rows = [r.split("\t") for r in sh("git", "-c", "core.quotePath=false", "show",
                                         "--name-status", "--format=", "HEAD^").splitlines() if r.strip()]
PREVCHAINHIT = len(chain_ids(PREVSLICE, prev_rows))
PREV2 = str(int(PREVSLICE) - 1)
PARENTCHAINHIT = len(chain_ids(PREV2, parent_rows))
OWNHIT = len(chain_ids(SLICE, prev_rows))
CHAINN = 5
assert PREVCHAINHIT == CHAINN, "上一片回执链在 HEAD 里只 %d/%d 件 ⇒ 本片没按惯例替上一片入库" % (
    PREVCHAINHIT, CHAINN)
assert PARENTCHAINHIT == CHAINN, "再上一片（%s 号）回执链在本片父提交里只 %d/%d 件 ⇒ 节奏不是一片一环" % (
    PREV2, PARENTCHAINHIT, CHAINN)
assert OWNHIT == 0, "本片回执链里有 %d 件已在 HEAD ⇒ CMD 行那句「留给分片 %s」成了假话" % (
    OWNHIT, NEXTSLICE)

# ---------- ⑤ 那段的所有数字：从**已归档的控件面原始件**现取（面名按片号全树 glob，须唯一） ----------
# 两条互证防的是同一族腐烂：① 点名的控件号必须真在标本面的 × 行里（编号不许是我编的）；
# ② 全过面与标本面的判读行必须不同（否则"红面"这个说法没有读数支撑）。
_CTL_FULL = re.compile(os.path.join("docs", "evidence", "[^/]+",
                                    r"_raw_s%s_msg_controls(\.attempt\d+-[^/]+)?\.out" % SLICE))


def ctl_face(tag):
    pat = "docs/evidence/*/_raw_s%s_msg_controls%s.out" % (SLICE, "" if not tag else "." + tag + "-*")
    hits = sorted(p for p in (os.path.relpath(x, REPO) for x in glob.glob(os.path.join(REPO, pat)))
                  if _CTL_FULL.fullmatch(p))
    assert len(hits) == 1, "%s 号片的控件面（tag＝%s）在全树命中 %d 份（须 1）：%s ⇒ 引用哪一份都不唯一" % (
        SLICE, tag or "全过面", len(hits), hits)
    return os.path.basename(hits[0]), open(os.path.join(REPO, hits[0]), encoding="utf-8").read()


def num(text, pat, what, face):
    m = re.search(pat, text)
    assert m, "%s：%s 里取不到「%s」这个读数 ⇒ 本件不许凭记忆写数" % (what, face, pat)
    return m.group(1)


SPEC1, SP1 = ctl_face("attempt1")
SPEC2, SP2 = ctl_face("attempt2")
SPEC3, SP3 = ctl_face("")
KC1 = num(SP1, r"CASES 声明＝(\d+) 条", "预检拒绝那一次的 CASES 数", SPEC1)
_m1 = re.search(r"在生成器源码里 (\d+) 处（\[(\d+), (\d+)\]）", SP1)
assert _m1, "%s 里取不到「在生成器源码里 N 处（[a, b]）」那行 ⇒ C1 那条拒绝没有出处" % SPEC1
KTWICE, KLINE1, KLINE2 = _m1.groups()
assert int(KTWICE) == 2 == len({KLINE1, KLINE2}), "预检印的处数 %s 与行号 %s/%s 不闭合 ⇒ 那句判读自己读歪了" % (
    KTWICE, KLINE1, KLINE2)
assert re.search(r"^  · C1 ", SP1, re.M), "%s 里找不到顶格式那条 C1 预检行 ⇒ 「被 C1 打破」是假话" % SPEC1
X2 = re.findall(r"^× ", SP2, re.M)
assert X2, "%s 里没有顶格「× 」行 ⇒ 它不是红面，不能当标本引用" % SPEC2
IDS2 = sorted(set(re.findall(r"^× ([CG]\d+) ", SP2, re.M)))
assert len(IDS2) == len(X2), "× 行有 %d 条但带编号的只有 %d 条 ⇒ 有一条没有编号，取列口径破了" % (
    len(X2), len(IDS2))
NX = str(len(X2))
KC2 = num(SP2, r"CASES 声明＝(\d+) 条（全部红路", "标本面的 CASES 数", SPEC2)
KD2 = num(SP2, r"扰动条数＝(\d+)", "标本面的扰动条数", SPEC2)
VERD2 = num(SP2, r"CONTROL_VERDICT = (.+)", "标本面的判读行", SPEC2)
KC3 = num(SP3, r"CASES 声明＝(\d+) 条（全部红路", "全过面的 CASES 数", SPEC3)
KD3 = num(SP3, r"扰动条数＝(\d+)", "全过面的扰动条数", SPEC3)
KGREEN = num(SP3, r"保持绿＝(\d+) 条", "全过面保持绿的条数", SPEC3)
KTO = num(SP3, r"超时（环境故障，不判红也不判过）＝(\d+) 条", "全过面的超时数", SPEC3)
KUNCOV = num(SP3, r"本轮造不出红路的闸门＝(\d+) 条", "全过面 UNCOVERED 的条数", SPEC3)
VERD3 = num(SP3, r"CONTROL_VERDICT = (.+)", "全过面的判读行", SPEC3)
_mrc = re.search(r"^\s*(C\d+) 超时必须为零的那条链改成容忍一条", SP3, re.M)
assert _mrc, "%s 里找不到「超时必须为零的链改成容忍一条」那条锚点普查 ⇒ 「判红的方式也被造一次红」没有出处" % SPEC3
KRECASE = _mrc.group(1)
assert VERD2 != VERD3, "标本面与全过面的判读行相同（都是 %s）⇒ 有一侧不是我说的那个面" % VERD3
assert not re.findall(r"^× ", SP3, re.M), "%s 里有顶格「× 」行 ⇒ 把它当「全过面」引用是假话" % SPEC3
_c5 = TMPL[TMPL.index("⑴ **扰动没有真的移动匹配集**"):TMPL.index("（表头写盘后仍要")]
# 槽位名里就带着 C＋数字（@@KC2@@／@@KC3@@）：不先剥掉占位符，编号普查会把槽位当成点名的控件号
_c5 = re.sub(r"@@[A-Z0-9_]+@@", "", _c5)
NAMED = sorted(set(re.findall(r"[CG]\d+", _c5)))
assert NAMED == IDS2, "⑤ 那段点名的控件号 %s ≠ 标本 %s 里 × 行的编号集合 %s ⇒ 有编号是编的或有 × 行没被讲过" % (
    NAMED, SPEC2, IDS2)
_mq = re.search(r"一条对照必须[^\n]*?。", _LSEG)
assert _mq, "§26.%s (%s) 里取不到「一条对照必须…」那句口径原文 ⇒ ⑤ 那段的引句没了出处" % (SECN, LESSONLET)
LESSONQUOT = _mq.group(0).replace("**", "")

vals = {
    "SLICE": SLICE, "NEXTSLICE": NEXTSLICE, "SECN": SECN, "SECLET": SECLET,
    "SPEC1": SPEC1, "SPEC2": SPEC2, "SPEC3": SPEC3, "KC1": KC1, "KTWICE": KTWICE,
    "KLINE1": KLINE1, "KLINE2": KLINE2, "NX": NX, "KC2": KC2, "KD2": KD2, "VERD2": VERD2,
    "KC3": KC3, "KD3": KD3, "KGREEN": KGREEN, "KTO": KTO, "KUNCOV": KUNCOV, "VERD3": VERD3,
    "KRECASE": KRECASE, "LESSONQUOT": LESSONQUOT, "LESSONNEEDLE": LESSON_NEEDLE,
    "GPULET": GPULET, "GPUQUOT": GPUQUOT, "LESSONLET": LESSONLET, "FSSEC": FSSEC,
    "PREVLOG": PREVLOG, "PREVSLICE": str(PREVSLICE), "PREV2": PREV2,
    "HDR_SPEC": HDR_SPEC, "SPEC": SPEC, "SPECOUT": SPECOUT, "SPECIN": SPECIN,
    "RULER_FIRST": RULER_FIRST, "RULE18REL": RULE18REL, "NDIR": NDIR,
    "PREVCHAINHIT": str(PREVCHAINHIT), "PARENTCHAINHIT": str(PARENTCHAINHIT),
    "OWNHIT": str(OWNHIT), "CHAINN": str(CHAINN),
    "NOW": sh("date", "+%F %T %z").strip(),
    "SCRIPT": SCRIPT_REL,
    "MD5PY": md5(os.path.join(REPO, SCRIPT_REL))[:12],
    "NRULE": str(len(ruler_copies)),
    "MDEQ": ("相同" if md5(os.path.join(REPO, SCRIPT_REL)) == md5(os.path.join(REPO, RULE18REL))
             else "不同 ⇒ 尺子被改过"),
    "HEAD": sh("git", "rev-parse", "HEAD").strip(),
    "PARENT": sh("git", "rev-parse", "HEAD^").strip(),
    "DIRTY": str(len([l for l in sh("git", "-c", "core.quotePath=false", "status", "--porcelain",
                                    "--", "src", "tests").splitlines() if l.strip()])),
    "FINGER": subprocess.run(["bash", "-c", FINGER_CMD], cwd=REPO, capture_output=True,
                             text=True).stdout.strip(),
    "NVRC": str(nv.returncode), "NVWHY": NVWHY, "NVTOTAL": str(len(nvpids)),
    "NVME": str(len(nv_mine)), "INTERP": INTERP,
    "CTLPOS": str(CTLPOS), "CTLNEG": str(CTLNEG),
    "NVLOOSE": str(len(nv_loose)),
    "NVLOOSE0": "、".join(sorted({os.path.basename(x) or "空" for x in nv_loose0})) or "无",
    "LESSONPREV": LESSONPREV, "FORMOLDLAST": FORMOLDLAST,
    "PREVSECN": PREVSECN, "FSECNEW": str(FSECNEW), "FSECOLD": str(FSECOLD),
    "PSECNEW": str(PSECNEW), "PSECOLD": str(PSECOLD),
    "CTLLOOSE": str(CTLLOOSE), "NVLOOSELOG": NVLOOSELOG,
    "KQQ": str(QQ), "KQQDIFF": str(QQDIFF),
    "RC": rc_txt,
    "ERRB": str(err_bytes),
    "ERRF": str(os.path.getsize(ERR)),
    "DATE_START": sh("date", "-r", RAW, "+%F %T %z").strip(),
    "DATE_END": sh("date", "+%F %T %z").strip(),
    "ENTRIES": re.search(r"tree 条目 (\d+) = ", body).group(1),
    "NB": re.search(r"blob (\d+) \+ tree", body).group(1),
    "NTREE": re.search(r"blob \d+ \+ tree (\d+)", body).group(1),
    "NP": re.search(r"父提交 = (\d+)", body).group(1),
    "NA": re.search(r"新增 (\d+) \+ 修改", body).group(1),
    "NM": re.search(r"新增 \d+ \+ 修改 (\d+)", body).group(1),
    "EXPECT": re.search(r"\+ 新增 (\d+) = (\d+)", body).group(2),
    "NBP": re.search(r"闭合式右边 \+1）⇒ (\d+) == (\d+) 判", body).group(2),
    "CA": re.search(r"正对照 A（远端 \S+ 的 sha 首位改 f）⇒ 不符 (\d+)/(\d+)", body).group(1),
    "CBA": re.search(r"正对照 A（远端 \S+ 的 sha 首位改 f）⇒ 不符 (\d+)/(\d+)", body).group(2),
    "CB1": re.search(r"正对照 B（远端删一条路径）⇒ 本地独有 (\d+)/(\d+)，远端独有 (\d+)/(\d+)", body).group(1),
    "CDB1": re.search(r"正对照 B（远端删一条路径）⇒ 本地独有 (\d+)/(\d+)，远端独有 (\d+)/(\d+)", body).group(2),
    "CB2": re.search(r"正对照 B（远端删一条路径）⇒ 本地独有 (\d+)/(\d+)，远端独有 (\d+)/(\d+)", body).group(3),
    "CDB2": re.search(r"正对照 B（远端删一条路径）⇒ 本地独有 (\d+)/(\d+)，远端独有 (\d+)/(\d+)", body).group(4),
    "CD1": re.search(r"正对照 D（远端删本片一条路径）⇒ 本片缺失 (\d+)/(\d+)，SHA 不符 (\d+)/", body).group(1),
    "CD2": re.search(r"正对照 D（远端删本片一条路径）⇒ 本片缺失 (\d+)/(\d+)，SHA 不符 (\d+)/", body).group(3),
    "DL": re.search(r"本地独有 (\d+)/", body).group(1),
    "DR": re.search(r"远端独有 (\d+)/", body).group(1),
    "MS": re.search(r"blob SHA 不符 (\d+)/", body).group(1),
    "NSHARED": re.search(r"blob SHA 不符 \d+/(\d+)", body).group(1),
    "SLICE_MATCH": re.search(r"MATCH (\d+)/", body).group(1),
    "NFILES": re.search(r"本片文件 (\d+) 个", body).group(1),
    "MDSAME": str(len(md_same)),
    "MDIF": str(len(md_diff)),
    "MDIFES": "、".join(os.path.basename(p) for p in md_diff) or "无",
    "MDAB": str(len(md_absent)),
    "S18": re.search(r"② 闭合式 (.+?) -> PASS",
                     open(os.path.join(REPO, [p for p in RCPT_REL
                                              if os.path.basename(p) == "am_s7_verify%s.log" % RULER_FIRST][0]),
                          encoding="utf-8").read()).group(1),
    "OUTREL": "docs/evidence/" + DAY + "/am_s7_verify%s.log" % SLICE,
    "RAWREL": "docs/evidence/" + DAY + "/_raw_verify%s.out" % SLICE,
    "RAWM": md5(RAW)[:12],
    "ERRM": md5(ERR)[:12],
    "MDNOTE": MDNOTE,
    "NAA": str(len(added)),
    "NMOD": str(len(mods)),
    "ADDBAS": "、".join(os.path.basename(p) for p in added),
    "MODBAS": "、".join(os.path.basename(p) for p in mods),
    "BOFF": "0",
}
# 原始件的来历：/tmp 里那份若还在，必须与归档件逐字节相同（cp -p 的凭据），否则 CMD 行的"逐字节归档"是假话
for _n in (os.path.basename(RAW), os.path.basename(ERR)):
    _t = os.path.join("/tmp", _n)
    if os.path.exists(_t):
        assert md5(_t) == md5(os.path.join(D, _n)), "/tmp 与归档的 " + _n + " 不同 ⇒ 归档件不是那次运行的原始 stdout"
assert "LANDING_VERDICT = ALL CHECKS PASS" in body, "正文里没有全过的判读行 ⇒ 本件不该被当作回执归档"
assert rc_txt == "0" and err_bytes == 0, "复跑核验件 rc/stderr 不齐 ⇒ 与正文不同一次运行，拒绝归档"
missing = sorted(set(re.findall(r"@@([A-Za-z0-9_]+)@@", TMPL)) - set(vals))
assert not missing, "模板有未取值的字段：" + str(missing)


def render():
    h = TMPL
    for k, v in vals.items():
        h = h.replace("@@" + k + "@@", v)
    return h


# BODY_OFFSET 必须等于"表头字节数"，而表头含这一行自身 ⇒ 迭代到不动点
for _ in range(12):
    cand = len(render().encode())
    if cand == int(vals["BOFF"]):
        break
    vals["BOFF"] = str(cand)
else:
    raise SystemExit("BODY_OFFSET 未收敛 ⇒ 拒绝归档")
hdr = render()
assert "@@" not in hdr, "渲染出的表头残留 PLACEHOLDER"


def gate(h):
    """字段标签必须**顶行首且紧跟冒号**；缩进行只能是续行；顶格非字段行只允许 BODY_OFFSET。

    口径收窄成「行首标签＋冒号」而不是「行内出现标签」：正文里有「本片单列复核 25/25」这类词组，
    用"包含"判会把自己的正确表头判红（写分片 18 那件时实测到，故在此注明口径）。
    """
    lines = [l for l in h.splitlines() if l.strip()]
    lab = "|".join(map(re.escape, FIELDS))
    defs = [l for l in lines if re.match(r"^(%s)\s*:" % lab, l)]
    mis = [l for l in lines if re.match(r"^\s+(?:%s)\s*:" % lab, l)]
    loose = [l for l in lines if not l[0].isspace() and l not in defs
             and not l.startswith("BODY_OFFSET=")]
    assert not mis, "缩进处出现「标签＋冒号」（漏了续行缩进？）：" + str(mis[:2])
    assert not loose, "顶格却不像字段行：" + str(loose[:2])
    assert len(defs) == len(FIELDS), "顶格字段行数 %d ≠ FIELDS 声明 %d ⇒ 有字段漏写或改名" % (
        len(defs), len(FIELDS))
    return len(defs)


assert sys.executable.startswith("/tmp/amvenv"), f"解释器不是 /tmp/amvenv 而是 {sys.executable} ⇒ CMD 行描述失真"
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "生成器自身未带空串 CVD ⇒ 与 ENV 行的描述不符"
gate(hdr)

if "--write" not in sys.argv:
    print(hdr)
    print("（预检模式：未写盘。加 --write。）")
    print("尺子全树份数＝", len(ruler_copies), "（本目录口径＝", NDIR, "）｜子条＝§26.%s (%s)｜钉住出处＝(%s)「%s」"
          % (SECN, SECLET, GPULET, GPUQUOT),
          "｜f-string 成因出处＝", FSSEC, "｜标本生成器＝", HDR_SPEC)
    print("回执链现取：上一片在 HEAD＝", PREVCHAINHIT, "/", CHAINN, "｜再上一片在父提交＝", PARENTCHAINHIT,
          "/", CHAINN, "｜本片在 HEAD＝", OWNHIT, "/", CHAINN)
    print("md5 现比 /tmp 同名件＝同", len(md_same), "／异", len(md_diff), "／缺", len(md_absent),
          "，共", len(added), "｜GPU 我的解释器进程＝", len(nv_mine), "/", len(nvpids))
    raise SystemExit(0)

open(OUT, "w", encoding="utf-8").write(hdr + body)
back = open(OUT, "rb").read()
off = int(re.search(rb"BODY_OFFSET=(\d+)", back).group(1))
assert back[off:] == body.encode(), "正文区逐字节复核失败"
head_txt = back[:off].decode()
# 两条残留检查的**消息必须字面不同**：控件面按"抛出的那一行含指定闸门句 **且**行号落在该句源码 ±3 行"
# 判红，同名的两条闸门会让窗口指到够不到的那一条（写盘分支在控件面的预检副本里永不执行）。
assert "@@" not in head_txt, "回读出的表头残留 PLACEHOLDER"
assert "＠" not in head_txt, "表头含全角 at 号 ⇒ 那种占位符会静默不替换"
n_defs = gate(head_txt)
assert int(vals["BOFF"]) == off == len(head_txt.encode()), "BODY_OFFSET ≠ 实际表头字节数"
c1 = subprocess.run(["cmp", "-i", str(off) + ":0", OUT, RAW], capture_output=True, text=True)
c2 = subprocess.run(["cmp", "-i", str(off + 1) + ":0", OUT, RAW], capture_output=True, text=True)
c3 = subprocess.run(["cmp", "-i", str(off) + ":1", OUT, RAW], capture_output=True, text=True)
print("写盘完成：", OUT, len(back), "bytes")
print("cmp 复核 rc（同偏移须 0）＝", c1.returncode, "｜左 +1 须非 0＝", c2.returncode,
      "｜右 +1 须非 0＝", c3.returncode, "｜顶格字段行数＝", n_defs, "/", len(FIELDS))
print("md5 现比 /tmp 同名件＝同", len(md_same), "／异", len(md_diff), "／缺", len(md_absent),
      "，共", len(added), "｜GPU 我的解释器进程＝", len(nv_mine), "/", len(nvpids),
      "｜复跑 stdout 与正文逐字节相同＝", REPRO)
assert c1.returncode == 0 and c2.returncode != 0 and c3.returncode != 0
