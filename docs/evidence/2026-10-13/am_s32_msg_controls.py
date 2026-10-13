#!/usr/bin/env python3
r"""提交信息生成器 `am_s32_msg.py` 的**正对照**（全离线）：每条扰动都要让它拒绝在**自己那条闸门**上。

扰动改的是**生成器的源码副本**（正则级改写），跑的是同一件（同一条 `python <file>`）⇒ 被验的就是要落地的
那把仪器本身，不是另写的平行实现。判据＝rc≠0 **且**抛出的那一行含指定的闸门句 **且** raise 的行号落在该句
在本副本里的位置（±3 行）——只看 rc 会把「环境错」记成「闸门对」。后一条有两支（坑③／③′／③″）：直投＝落在
抛出点 ±3；参数句＝该句在**更早一帧**的 ±3 内（`one()`／`grab()` 那类把闸门身份当实参传的助手）。两支由
`C-self-2` 与直投／更早帧／链外三个形状双向把关。

本轮（分片 32）新增闸门的红路，逐条都要有：
① **上一节节号的两个出处**：回执表头 ENV 行那枚与 HEAD 那版开发日志末节那枚必须相等，不符就拒绝生成（C0a）；
   那枚「从 HEAD 主题取节号」的旧针本轮须命中 **0**，而这个 0 配一条**合成正对照**（同一枚针在合成样本上必须
   数出 99）＋一条「把针打歪到合成样本取不到」的红（C0b／C0c／C0d）——没有对照的 0 分不清「主题里没有」与
   「针坏了」，这正是 §26.34 登记的那条 LESSON 的现场；
② **同日轮的两条等值**：`PREVDIR == E` 与 `PREVDAILY == DAILY` 都被钉成断言，红路＝把上一片回执／纪实指到别的
   日期目录（C6／C6b）；链归属的唯一性来自片号而不是目录，所以「带片号的 glob 命中 1 份」与「同一条 glob 去掉
   片号命中多份」两侧各有扰动（C4／C5／C0e）；
③ **落笔三去向闭合式的每条腿**：未跟踪侧须 0（C0f）、已暂存新增三出处（C24／C26）、修改数（C27／C28）、
   总计（C25）——上一片那条「未跟踪＋修改＝总计」的两去向形状在本轮命中 0，取数正则整条换过 ⇒ 分组配对族的锚点
   全部重对齐（C12–C16）；
④ **面名复用之后唯一有现场的仪器侧证据**＝本片写入件的**复跑**：末行针、读数闭合、计数行缺席、stdout 是成功
   预检面的前缀，四条各自可红（C62／C63／C64b／C64c）；上一片那族 `.attemptN-` 标本针在本轮源码里根本不存在，
   照抄就是空转（见坑⑲）；
⑤ **「写入件那一族没有 attempt 现场」那条 0 的正对照**：同一把尺子（总体＝`git ls-files`＝入库清单）在上一片
   同形状上必须数出 >0 件，两个方向各扰一次（C0g／C0h）；那句的**范围**另有一条闸（本片其它族须 >0 件，
   否则「只圈写入件那一族」要改成整片口径），两侧各扰一次（C0k／C0k2）；
⑥ 两条**原先只是读数、本轮升成闸**的声称：三面 stderr 全 0（C0i）与排期段那批专名必须在开发日志里有出处
   （C0j）；
⑦ 继承族（单值取数／桶形与分组配对／入库面计数／顶格字段普查／NVML 坏掉之后的用户态版本双出处与 /proc 整词
   红线／三面读数两侧闭合／回执复跑与局部仪器／`-uall` 口径／尺子 md5 族／locale 指纹／开发日志形制与节内唯一性／
   回执表头读数／模板形制闸）逐条保留，锚点与期望句按本轮源码重新对齐。

十七条**仪器自己的**坑（继承＋本轮新增）：
① 判据不能只看 rc≠0；副本也不能写在 /tmp——生成器用自身位置反推 REPO ⇒ 副本必须落在**同一目录**，并加一条
   **未扰动基线必须 rc=0 且逐字节命中基线标记句**（见 SELF_BASE_MARK），否则所有红都不作数。
①′ 同目录那份副本本身**就是一张未跟踪面**：本轮生成器新加了「每件落盘即 `git add` ⇒ 未跟踪侧须 0」那条闭合闸，
   于是每一条控制都先红在那条与判据无关的现场闸上（实测 rc=1／未跟踪现数 1＝副本自己）。修法不是把闸放宽到
   「按名字放过 `zctl_*`」，而是**按身份**：环境变量逐字等于正在跑的这一份的件名、且未跟踪侧只有它一件，才算豁免；
   真实运行没有那枚变量⇒ 门槛与之前逐字相同。三个方向各有闸（C0l 撤掉豁免／C0m 身份比对换件名／C0n 一致性关系反切）。
② **期望闸门句里不许写死节号／片号／读数**：那些是档案的读数，写死就随片腐烂。本件所有期望句都取自闸门自己
   的固定短语。
③ 按**抛出的那一行**判红，不按整段 stderr：3.11+ 的 traceback 把源码行原样回显，闸门句本来就写在源码里。
   raise 的行号取**最后一个**匹配本副本名的帧——第一版取第一条，量到的是调用链而不是抛出点。
③′ 本轮把「该句在源码里的位置」从**单点**改成**多点取任一**：`one()`／`grab()` 的报错句是通用文本（在源码里
   两处），按第一版「须恰好 1 处才做位置判定」会把这类真红判成不合格，于是这条要么放弃位置判定（退回只看 rc＝
   坑①的形状）要么放宽判据。**放宽方向选在判定器而不是数据上**：允许 raise 落在该句的任一出现处 ±3，仍是
   「红在别的闸门」的反例（别的闸门不在这些行上）。
③″ 参数句形状（本轮实测的第二种）：`one(pat, text, "那枚针的合成正对照")` 的**闸门身份是第三个实参**，
   raise 落在助手段（80 行）、期望句只写在调用点（165 行）⇒ 按 ③／③′ 那种「抛出行 ±3」判会把真红读成空转
   （本轮 C0d 现场：rc=1、抛出行逐字节含该句，但 raise 在 80／句在 165）。判据因此**不是**放宽成「整段 stderr 命中」，
   而是要求两件事同时成立：该句在本副本源码里的那一处**出现在 trace 的更早一帧**（＝它确实在抛出链上，不是别处的文本）
   ＋抛出行本身含该句（＝报错的是这一条闸门而不是别的）。判定器抽成 `same_gate()`，真跑与自查共用（不另写平行实现），
   并由 `C-self-2` 给三个形状：直投（同一条）、参数句（放行）、句在链外（**必须仍然红**）——最后那条是这条放宽的反例，
   少了它这条放宽就不可红。
④ **本件自己不许用钉住的那个解释器跑**（本轮实测出的形状反转：上一版这里写的是「必须用 `/tmp/amvenv/bin/python` 跑本件」）：
   副本会**重跑上一片的归档自查件**，而那件按 argv[0] 整词数「机上还有几个钉住的解释器」，排除表只有**自身与父进程**两级
   ⇒ 「控件面（钉住）→ 扰动副本（钉住）→ 归档自查件」这条三级链必定把**控件面自己**数进去，副本于是红在一条与判据无关的
   现场闸上（本轮实测：rc=1／「机上还留着钉住的那个解释器在跑 ['3592278']」，期望句「顶级子条＝7 条」根本没走到）。
   修法＝本件用**别的**解释器跑（`/usr/bin/python3`，仍逐条内联空串 `CUDA_VISIBLE_DEVICES=""`），**副本一律仍用钉住的那个**
   （常量 `PY3`＝生成器一族钉的那个解释器）；`main()` 第一条就拒（钉住解释器在场时整轮正对照作废，别去花钱跑几十个子进程）。
   同一坑的另一半不变：环境里不许有 `LC_ALL`（生成器第一条就断言它缺席，并把不钉 locale 的那条读数用于四路闭合）；
   不满足就跑第一步会全红，那不是闸门红。
⑤ **改了源码 ≠ 改了行为**：一条扰动若让副本 rc=0，它要么反过来切，要么如实进 `UNCOVERED`（本轮见文件末尾）。
   判据必须允许「这条造不出红」被登记，否则下一轮会把它当成已验过的闸门。
⑥ **扰动落在 dict 的字面键／正则上时，取值处要同批改**；needle 选成在源码里**多处**出现的串时，红可能落在别的
   那条闸门上 ⇒ 本件对每条都验「抛出行 ±3」（同 ③′）。
⑦ 跑本件之前先做**纸面预检**（锚点在源码里恰好命中 1 处、扰动后仍能 `compile()`、期望句在源码里可定位），再加
   两条**本件自己**的对照：本文件声明的两个自述常量必须各在生成器源码里命中 1 行（正）、写歪的锚点句必须 0 命中
   （负）。这三条要靠真跑几十个子进程才暴露，纸面预检把「锚点 0 命中」这种空转在花钱之前挑出来（`--precheck`）。
⑧ **本件的基线依赖一个时序**：生成器有一条「本件的干跑控件面必须在入库清单里」的断言（＋那条「当前未跟踪侧
   必须为 0」的流程前提）⇒ 跑本件**之前**要先 `git add` 本件自己与它的三对原始面（内容此刻不存在，生成器不读
   它们，只按文件名计数）。这条不是循环：循环只发生在「生成器读自己控件的 stdout」那种引用上，所以本件的读数由
   **分片 33** 的归档自查腿引用，本片不引用。
⑨ 每条子运行自带 `SUB_TIMEOUT`，超时按**进程组** SIGKILL；超时判成**环境故障**（`CONTROL_VERDICT = ENV
   FAILURE…`＋exit 4），既不判红也不判过——本件全离线，超时是机器不是闸门。
⑩ **绿路必须真绿、不许是 Python 恒真式**：`assert X is not False or True` 这种形状永远通过（第一版就写了这么
   一条，靠「跑完之后现场必须无残留」把它变成实判）。本件的绿路全部**跑命令**（cmp／dpkg／git／/proc／复跑
   真仪器），每条末尾再断言本件自己的现场闸（副本文件已删、档案面未被改）。
⑪ **有一条扰动会去碰另一件仪器的写盘腿**（C110：克隆本归档的写入件并带 `--write` 跑它）。这条是本轮唯一会执行
   **别的仪器**的扰动，所以它前后都比对「工作树对暂存面零漂移」（`git diff --name-only` 为空＝那五张档案面逐字节
   未动）——比对不为空就当场红，不把「没写」当假设。**实跑得知的次序**（本轮由一次真探针取得，不是抄上一片）：
   克隆副本**连上一片那条落笔闭合式都没走到**——它先撞上写入件更前面那条「未跟踪侧必须为空」的现场闸（副本自己就是
   那张未跟踪面，所以这条红**按构造必然发生**，比上一轮的「现数与写死分母不等」更强）。它同样**走不到**那句
   「有仪器正在跑」的守卫 ⇒ 本条的期望句按实际打红的那条闸门写，守卫那句进 UNCOVERED。
   上一轮的现场（写入件在**本件之外**另跑一次时红在落笔闭合式）仍归档在 `_raw_s32_docs_rerun_after_landing.err`，
   它是生成器那条复跑归因闸的靶，不是本条克隆腿的靶——两个现场分属两条闸门，混为一谈就是声称没发生过的事。
⑫ **基线的「命中行数」判据要按生成器的**打印形制**定，不是按「标记句理应唯一」的直觉定**：上一版写「标记句在副本
   stdout 里须恰好命中 1 行」，实跑就红在 2 行——那一轮的标记句既是一条 bullet（逐字节相同）又被「诚实登记」嵌进
   长句（带前后文），**按构造出现两次**。本件沿用放宽后的判据「至少一行逐字节相同」，但标记句换了本轮的一条：
   本轮那句只以 bullet 形式出现（现数 1 行），所以这里不是靠放宽蒙过去——歧义由另外两条尺子承担（本文件里那句须
   1 行＋生成器报的字节数与本件同值）。这是「判据按构造红不了」的一族，形状＝**把出现次数写死成 1**。
⑬ **桶形必须容得下红灯现场的命名形制**：本件的桶第一版只列 `_controls`／`_controls_red\d+`，于是第一次实跑的红灯面
   （`…_controls.attempt1-mark-line-hits-twice.out`）落进 0 个桶。红得对（逐件断言拦住了），但那批面本来就该被登记；
   修法＝`.attemptN-<原因>` 后缀写成可选，并按**真实现场清单**预演逐组等值。
⑭ **模板槽位的取值须是一个原子字符串**：四个正对照的取数正则原先各捕多字段，`grab` 返回元组、直接落到槽上 ⇒
   渲染腿的类型闸（`isinstance(val, str)`）拒印＝**真件本轮红过一次**（不是造出来的红）。修法＝每条 K_CTL* 改成
   捕一个整段（每格＝回执里逐字节摘出的一段），并把模板里重复的标签删掉；这条闸门的正对照＝C97（列表落进槽位）。
⑮ **扰动必须真的移动匹配集，否则它只是纸面扰动**（上一轮＝分片 31 实测四条，本轮逐条重验而不是照抄结论）：① glob 的片号
   插值 `%d`→`%s`（`%` 格式化里两者产出同一串）；② needle `?? `→`??`（porcelain 的未跟踪行永远带那个空格）；③ 把
   「切到 `；` 为止」改成不切（解释里不含清单分隔符 `／` ⇒ 条数不变，红落在**另一条**闸门上）；④ 钉 locale 的那条
   改钉 en_US.UTF-8（本机默认不是它 ⇒ 两条读数照旧不同）。出路只有两条：把扰动改成**真的移动**匹配集（①去掉插值、
   ③换错分隔符、④两条都钉成同值），或如实进 UNCOVERED 并附实测（②）。预检能验「锚点命中 1 处」，验不出「命中了
   但行为没变」——这一族只有真跑才知道。
⑯ **扰动只能落在有这个变量的作用域里**：把 `faces_check` 里的 `modified` 换成 `added` 得到的是 `NameError: name
   'added' is not defined`（`added` 是 `build()` 的局部名），那不是红在闸门上。要么改扰**调用处**（C22），要么换一个
   当场真有、且真会判错的值（C23 用 `pinned_faces`＝把 relpath 混进 basename 清单）。
⑰ **删掉比较的一项不是坏输入，是放宽**：上一轮的 C63／C96 第一版各删掉闭合式的一段，副本照旧绿（放宽后的判据更
   容易满足）。正对照要证的是「这条比较真的在算」⇒ 改成**反向**（现算与另一出处本就同值／标本里两个数本就不同值），
   实跑必红。本轮凡是「两个读数本应不等」的闸（复跑那句的现数 vs 写死那条）都按这一族写。
⑱ **本轮新增：锚点腐烂是机械克隆的默认结局，不是例外**。第一次纸面预检（现场＝`_raw_s32_msg_precheck.out`）
   就从上一版抄来的 111 条里挑出 19 条锚点在本轮源码里命中 0 处。根因不是字面写错，而是**那些闸门换了形状**：
   分组配对从 `zip(toks, BKEY_PIN)` 换成「标签整串逐字命中」、`attemptN-` 标本换成「未扰动源码复跑现场」、
   `ndef` 从函数内现取换成调用方传入。所以每条换锚点都必须连**期望句**一起换——只换锚点会造出「锚点命中 1 处、
   红却落在别处」的空转（＝坑⑥与坑③′的合流）。
⑲ **本轮新增：一条 0 读数当证据时，控件面得能把它打红**。生成器里有三条「按构造须为 0」的闸（未跟踪侧 0、
   `ncode == 0`、`stderr3 == {0}`）。这类闸的正对照不能是「把它删掉」（删掉＝放宽，见坑⑰），只能是把**比较的
   另一侧**挪成非零值或读错流（C0f／C18／C0i）——证的是「这条 0 真的在算」。
⑳ **本轮新增：期望句里的百分号会削弱位置判定**。判定器取 `expect.split("%")[0]` 作为源码定位针，所以期望句若
   写成「前 %d 桶和 %d …」，针就退化成「前 」这种满屏命中的短串 ⇒ 位置判定名义上有、实际空转。本件的做法：
   期望句只取闸门消息里**不含 % 的那一段**（且该段在源码里出现处数打印出来，多于 1 处时按「任一出现处 ±3」判，
   同坑③′），绝不在期望句里写死随轮变化的数（同坑②）。
㉑ **本件的归档面自己也是被普查的跟踪面**（本轮实测的形状，与前两例同源＝仪器与现场相互作用）：扰动副本跑的是
   生成器，而生成器第一条现场闸就是「工作树对暂存面零漂移」。于是**任何一张刚写过、还没重新 `git add` 的跟踪面**
   都会让副本红在那条闸上——本轮第一次就是这样（刷新过的 `_raw_s32_msg_precheck.out` 比索引里那一版多 26 B ⇒
   基线副本 rc=1、期望句根本没走到）。第二条更隐蔽：本件把 stdout 重定向进 `_raw_s%d_msg_controls.out`，那张面
   在整轮里必然偏离索引，只是末尾一次性打印让 8 KB 缓冲到退出才 flush，所以侥幸没漂；一旦中途有输出穿过缓冲
   （超时那条就当场 `print`），从那一刻起**后面每条控制**都会红在同一条闸上。
   修法不是把漂移闸关掉，而是分两处：**本件自己的**前置闸按身份放过本进程那两个写入 fd（`/proc/self/fd/1`、`/2`
   解析到仓库内的路径），其余一律算漂移；而**副本那边没有这枚豁免**（它们的 stdout 是管道），所以跑法本身改成
   先把两条流落到仓库外的临时面、跑完再 `cp -p` 进归档面＝整轮里索引与磁盘逐字节同值。并在基线之前先数一次漂移
   （花钱之前就停，而不是留下一整轮假红）。窄化由 `C-self-3` 双向把关：放过本件归档面的同时，
   **真档案面（开发日志）混在里面时必须仍然报出来**。
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
SRCNAME = "am_s32_msg.py"
SRC = os.path.join(HERE, SRCNAME)
PREFIX = "zctl_s32_msg_"
# 副本（和它再跑的归档自查件）用**钉住的**那个解释器；本件自己必须用别的跑，见坑④那条实测。
PY3 = "/tmp/amvenv/bin/python"
SUB_TIMEOUT = 300
ENV_FAIL_STOP = 3
WINDOW = 3
text = open(SRC, encoding="utf-8").read()

# 本件对**自己**声明的两个自述常量：生成器只读这两个字面（不读本件的 stdout），并要求各自在本源码里恰好命中 1 行。
# ⇒ 这两行是**唯一**出现处：下面任何标签／注释都不许再抄一遍，否则生成器 selfctl_check 当场红。
SELF_UNUSED_NEEDLE = "取值表里有模板没用到的读数"
SELF_BASE_MARK = "入库那两面因此只描述**最后两次成功运行**，信息里按「造不出红路」如实登记，不假称有红灯现场。"

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
    ("C6 上一片回执的目录被硬写成另一个日期（同日轮那条「等值钉成断言」的闸红）",
     [(r'^PREVDIR = os\.path\.dirname\(CURL\)$', 'PREVDIR = "docs/evidence/2026-01-01"')],
     "本轮是跨日期轮"),
    ("C6b 上一片纪实的文件名不再由 PREVDIR 派生（当日纪实＝同一张面那条等值闸红）",
     [(r'^PREVDAILY = "\.workbuddy/memory/%s\.md" % os\.path\.basename\(PREVDIR\)$',
       'PREVDAILY = ".workbuddy/memory/2026-01-01.md"')],
     "本轮是跨日期轮"),
    ("C0a 上一节节号的两个出处被强制差一（回执 ENV 行与 HEAD 开发日志末节互证那条）",
     [(r'^assert _prev_env_sec == _head_sec', "assert _prev_env_sec == _head_sec + 1")],
     "上一节的节号两个出处不符"),
    ("C0b 把「HEAD 主题里那枚节号针」的现数硬写成 1（那句「主题不可靠」的现场就没靶了）",
     [(r'^PREVSEC_SUBJ_N = len\(re\.findall\(r"开发日志 §26\\\.\(\\d\+\)", SUBJ\)\)$', "PREVSEC_SUBJ_N = 1")],
     "那句「主题不可靠」要改写"),
    ("C0c 合成正对照的期望值从 99 挪成 98（那条「针在合成样本上必须数出数」的对照失明）",
     [(r'^assert int\(_synth\) == 99', "assert int(_synth) == 98")],
     "连合成样本都取不到"),
    ("C0d 合成样本里去掉「§」使那枚针取不到 ⇒ 正对照真的在算而不是恒真",
     [(r'"钉住口径从档案现取＝开发日志 §26\.99（合成样本）', '"钉住口径从档案现取＝开发日志 26.99（合成样本）')],
     "那枚针的合成正对照"),
    ("C0e 不带片号的宽 glob 插上片号（两条读数同值 ⇒「唯一性来自片号而不是目录」那句失去靶）",
     [(r'glob\.glob\(os\.path\.join\(REPO, "docs/evidence/\*/am_s7_verify\*\.log"\)\)\)',
       'glob.glob(os.path.join(REPO, "docs/evidence/*/am_s7_verify%d.log" % PREVSLICE)))')],
     "唯一性来自片号而不是目录"),
    ("C0f 落笔那句的未跟踪侧门槛从 0 挪成 1（「每件落盘即 git add」那条流程前提不再在算）",
     [(r'^    assert nu == 0', "    assert nu == 1")],
     "本轮口径要求它是 0"),
    ("C0g 上一片同形状 attempt 面的正对照门槛抬到 99（本片的 0 就没了配对）",
     [(r'^    assert att_prev > 0', "    assert att_prev > 99")],
     "那把尺子对本片的「0 件」读数就是失明的"),
    ("C0h 正对照的片号换成一个不存在的轮号（同一把尺子两边都数出 0＝假零形状）",
     [(r'att_here, att_prev = _attn\("_raw_s%d_docs" % SLICE\), _attn\("_raw_s%d_docs" % PREVSLICE\)',
       'att_here, att_prev = _attn("_raw_s%d_docs" % SLICE), _attn("_raw_s%d_docs" % (SLICE + 9))')],
     "那把尺子对本片的「0 件」读数就是失明的"),
    ("C0k 范围登记那条门槛抬到 99（「只圈写入件那一族」不再在算）",
     [(r'^    assert att_other > 0', "    assert att_other > 99")],
     "那句的范围要改成整片口径"),
    ("C0k2 整片普查的族前缀收成写入件那一族（att_other 由 2 掉到 0＝那条范围没了靶）",
     [(r'att_other = _attn\("_raw_s%d_" % SLICE\) - att_here',
       'att_other = _attn("_raw_s%d_docs" % SLICE) - att_here')],
     "那句的范围要改成整片口径"),
    ("C0l 撤掉那条按身份的未跟踪豁免（副本自己被数进去＝每条控制红在同一现场闸）",
     [(r'eff = \[p for p in un if not \(CONTROLS and p == COPYREL\)\]', "eff = list(un)")],
     "落笔未跟踪侧不闭合"),
    ("C0m 豁免的身份比对换成生成器自己的件名（正在跑的副本对不上＝豁免不能用）",
     [(r'assert un == \[COPYREL\]', "assert un == [SELF]")],
     "那条按身份的豁免不能用"),
    ("C0n 「豁免只在控件环境里改变读数」那条一致性关系反过来切（真实运行也允许读数错位）",
     [(r'assert \(u\["unraw"\] == u\["un"\]\) == \(not CONTROLS\)', 'assert (u["unraw"] == u["un"]) == CONTROLS')],
     "那条豁免用错了地方"),
    ("C0i 「三面 stderr 全为 0」那条声称读错了流（拿 stdout 的字节数当 stderr）",
     [(r'stderr3 = \{os\.path\.basename\(x\): b\(x\[:-4\] \+ "\.err"\) for x in \(MSGCTL, HDRCTL, AFCTL\)\}',
       'stderr3 = {os.path.basename(x): b(x) for x in (MSGCTL, HDRCTL, AFCTL)}')],
     "那批面不是全过面"),
    ("C0j 排期段里那批专名换成一条档案里没有的说法（手打的排期句就该拒绝生成）",
     [(r'"≤0.004%", "--measured-per-track", "唯一可评分"', '"≤0.9999%", "--measured-per-track", "唯一可评分"')],
     "那是手打的排期句"),
    ("C7 回归件 glob 改成不存在的后缀（第四个出处没了）",
     [(r'"docs/evidence/\*/_raw_t\*_full\.log"', '"docs/evidence/*/_raw_t*_full.LOG"')],
     "全树回归件命中 "),

    # —— ② 角色桶 ↔ §26.34 落笔分组 ——
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
    ("C12 分组配对整串倒着配（同一批标签反序 ⇒ 第一项就对不上）",
     [(r'for i, key in enumerate\(BKEY_PIN\):', "for i, key in enumerate(list(reversed(BKEY_PIN))):")],
     "分组第"),
    ("C13 分组配对只配前若干项（落笔的最后一组没人认领 ⇒ 句尾还剩尾巴）",
     [(r'for i, key in enumerate\(BKEY_PIN\):', "for i, key in enumerate(BKEY_PIN[:-1]):")],
     "个标签之后还剩"),
    ("C14 标签后面那枚项间分隔符换成一枚档案里不存在的字符（针的尾段真的在算，不是装饰）",
     [(r'\(\?:／\|\$\)"', '(?:｜|$)"')],
     "分组第"),
    ("C15 分组等式左边的比较反向（本件现数与各组声称本就同值 ⇒ 反向必红）",
     [(r'assert lhs == \[c for _l, c in rows\]', "assert lhs != [c for _l, c in rows]")],
     "分组等式左边"),
    ("C16 分组项数与生成器前桶数的关系写成多一项",
     [(r'assert int\(ngrp\) == len\(BKEY_PIN\)', "assert int(ngrp) == len(BKEY_PIN) + 1")],
     "生成器前桶"),
    ("C17 修改面清单的分隔符从 `／` 换成 `；`（整串当一条 ⇒ 条数闸红；把 `；` 那半边删掉只改内容不改条数＝空转）",
     [(r'    lst = sorted\(faces\.split\("／"\)\)\s*$', '    lst = sorted(faces.split("；"))')],
     "修改面清单现数 "),
    ("C15b 落笔三去向闭合式整体反向（现算与落笔本就同值 ⇒ 反向必红；只删一段是放宽，副本照旧绿）",
     [(r'assert na == sum\(c for _l, c in rows\) and nu \+ na \+ nm == tot',
       "assert na == sum(c for _l, c in rows) and nu + na + nm != tot")],
     "落笔闭合式不齐"),

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
    ("C24 落笔未跟踪侧的三出处闭合把最后的 0 挪成 1（那条「每件落盘即 git add」不再在算）",
     [(r'assert int\(wv\["wnu"\]\) == pnu == u\["un"\] == 0', 'assert int(wv["wnu"]) == pnu == u["un"] == 1')],
     "落笔未跟踪侧不闭合"),
    ("C24b 落笔已暂存新增数的三出处闭合里把桶现数挪掉一件",
     [(r'assert int\(wv\["wna"\]\) == pna == u\["pin"\]', 'assert int(wv["wna"]) == pna == u["pin"] - 1')],
     "落笔已暂存新增数三处不闭合"),
    ("C24c 落笔修改数的三出处闭合与暂存 M 的关系加一",
     [(r'assert int\(wv\["wnm"\]\) == pnm == nm', 'assert int(wv["wnm"]) == pnm == nm + 1')],
     "落笔修改数三处不闭合"),
    ("C25 落笔闭合式两边不相等（总计加一）",
     [(r'assert int\(wv\["wnt"\]\) == ptot', 'assert int(wv["wnt"]) == ptot + 1')],
     "落笔闭合式两边不闭合"),
    ("C26 前 11 桶和与落笔已暂存新增数的关系加一",
     [(r'assert sum\(len\(hits\[k\]\) for k in BKEY_PIN\) == pna', "assert sum(len(hits[k]) for k in BKEY_PIN) == pna + 1")],
     "落笔已暂存新增数"),
    ("C27 修改面数的三个出处少配一个（声称与暂存 M 差一）",
     [(r'assert pnface == len\(pinfaces\) == nm', "assert pnface == len(pinfaces) == nm + 1")],
     "登记的修改面 "),
    ("C28 落笔登记的修改件数与暂存 M 的关系加一",
     [(r'assert pnm == nm', "assert pnm == nm + 1")],
     "落笔登记的修改件数 "),

    # —— ④ 顶格字段槽位普查 ——
    ("C29 把传给普查的 FIELDS 声明数换成 99（消息里承诺的那条坏输入，本轮由调用处传入）",
     [(r'devlog_check\(len\(fields\)\)', "devlog_check(99)")],
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
    ("C43 表内 G 行的行首前缀少两个空格（表头面逐行数到 0 行）",
     [(r'grow = len\(\[l for l in hdr\.splitlines\(\) if l\.startswith\("  G"\)\]\)',
       'grow = len([l for l in hdr.splitlines() if l.startswith("G")])')],
     "表头控件面里表内 G 行"),
    ("C44 表内 G 行的行首字母改成 E（标签与行数配不上）",
     [(r'grow = len\(\[l for l in hdr\.splitlines\(\) if l\.startswith\("  G"\)\]\)',
       'grow = len([l for l in hdr.splitlines() if l.startswith("  E")])')],
     "表头控件面里表内 G 行"),
    ("C43b 另两面的绿路比较反向（提交说明面的声明与表内 G 行本就同值 ⇒ 反向必红）",
     [(r'assert mglow == int\(mgreen\) and afgrow == int\(afgreen\)',
       "assert mglow != int(mgreen) and afgrow == int(afgreen)")],
     "另两面的绿路声明与表内 G 行不齐"),
    ("C43c 只读仪器面的 G 行前缀多一个空格（那条「基线在表内」的第二把尺子失明）",
     [(r'afgrow = len\(\[l for l in af\.splitlines\(\) if l\.startswith\("  G"\)\]\)',
       'afgrow = len([l for l in af.splitlines() if l.startswith("   G")])')],
     "另两面的绿路声明与表内 G 行不齐"),
    ("C45 基线行的前缀写歪（「基线不进表」那条口径没有第二出处）",
     [(r'l\.startswith\("基线（未扰动副本"\)', 'l.startswith("基线（扰动副本")')],
     "表头控件面基线行现数"),
    ("C46 绿路闭合式里把基线那半边挪掉一件",
     [(r'assert int\(hgreen\) == grow \+ hbase', "assert int(hgreen) == grow + hbase + 1")],
     "绿路账没有第二出处"),
    ("C47 三面判读行的一致性门槛写成 3",
     [(r'assert len\(set\(verd\)\) == 1', "assert len(set(verd)) == 3")],
     "三面判读行不一致"),
    ("C48 三面超时／环境故障必须为零的那条链改成容忍一条（环境故障被当全过）",
     [(r'assert int\(mto\) == hto == ato == 0', "assert int(mto) == hto == ato == 1")],
     "不能当全过面引用"),
    ("C48b 提交说明控件面「因超时没跑到」的门槛写成 1（分母不完整性感）",
     [(r'assert int\(mskipped\) == 0', "assert int(mskipped) == 1")],
     "那份读数的分母不完整"),
    ("C48c 环境故障重跑／装配失败那两条的门槛反过来（非零也敢当全过面引用）",
     [(r'assert henvrerun == 0 and afbuild == 0', "assert henvrerun != 0 and afbuild != 0")],
     "两面不是全过面"),
    ("C48d 只读仪器控件面的第三本账加一（实跑行数≠红路＋绿路时该拒绝）",
     [(r'assert afrun == int\(afcase\) \+ int\(afgreen\)', "assert afrun == int(afcase) + int(afgreen) + 1")],
     "只读仪器控件面：实跑"),
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
    ("C60b trace 帧普查的正则写歪（取不到任何帧⇒ 那句「抛出点」失去出处；本轮实测该 trace 有 2 帧，上一版用 one() 要求 1 帧就是红在这条的形状上）",
     [(r're\.findall\(r.File "\(\[\^"\]\+\)", line \(\\d\+\), in., r\.stderr\)',
       're.findall(r"File X", r.stderr)')],
     "取不到 trace 帧"),
    ("C60c 逐帧归属的比对对象换成归档自查件（链路里全是上一片生成器却要求别的件名）",
     [(r'\{os\.path\.basename\(x\[0\]\) for x in frames\} - \{os\.path\.basename\(PREVMSG\)\}',
       '{os.path.basename(x[0]) for x in frames} - {os.path.basename(ARCHPY)}')],
     "复跑红点的链路里有别的件名"),
    ("C61 归档自查件复跑的红点 needle 写歪（归因句拦不住别的断言）",
     [(r'rerun_local\(ARCHPY, \("顶级子条＝7 条"', 'rerun_local(ARCHPY, ("顶级子条＝99 条"')],
     "复跑红在了别的断言上"),
    ("C61b 上一片生成器复跑的红点 needle 写歪（同族的第二条归因闸）",
     [(r'rerun_local\(PREVMSG, \("HEAD 提交说明里的日志节号"', 'rerun_local(PREVMSG, ("HEAD 提交说明里的日志节号X"')],
     "复跑红在了别的断言上"),
    ("C62 写入件复跑现场的末行针写歪（那条计数闸的归因拦不住别的红）",
     [(r'"本节的闭合式写的是" in last', '"本节的闭合式写死" in last')],
     "复跑末行不像那条闭合式闸门"),
    ("C63 复跑那句里「现数≠写死那条」反向（现场里两个数本就不同值 ⇒ 反向必红；只删这一项是**放宽**，副本照旧绿）",
     [(r'and int\(nums\[2\]\) != int\(nums\[3\]\)', 'and int(nums[2]) == int(nums[3])')],
     "复跑那句不闭合"),
    ("C64b 复跑 stdout 不再要求计数行缺席（它跑完再判红也敢归因）",
     [(r'assert "待入库"\.encode\(\) not in out', 'assert "待入库".encode() in out')],
     "复跑 stdout 里出现计数行"),
    ("C64c 复跑 stdout 与成功预检面的前缀关系要求逐字节相同（两次跑的根本不是同一条路径）",
     [(r'assert pre\.startswith\(out\) and out != pre', 'assert pre.startswith(out) and out == pre')],
     "不是成功预检面的前缀"),
    ("C65 追加面张数的分母从 4 改成 5",
     [(r'assert len\(app\) == 4', "assert len(app) == 5")],
     "写入件登记的追加面实取 "),
    ("C66 写入件落笔现数那句的闭合式反向（三个现数与总计本就同值 ⇒ 反向必红）",
     [(r'assert int\(wnu\) \+ int\(wna\) \+ int\(wnm\) == int\(wnt\)',
       "assert int(wnu) + int(wna) + int(wnm) != int(wnt)")],
     "写入件那句落笔现数不闭合"),
    ("C66b 写入件落笔时的未跟踪侧门槛从 0 挪成 1（那条流程前提的读数不再在算）",
     [(r'assert int\(wnu\) == 0', "assert int(wnu) == 1")],
     "那条流程前提的读数要改写"),
    ("C66c 写入件判读行的两条固定尾串之一放宽（那句「已落盘并重数唯一性」没有第二出处）",
     [(r'assert v1\.endswith\("LANDED, UNIQUENESS RECOUNTED ON DISK"\)', 'assert v1.endswith("LANDED")')],
     "写入件判读行不以那句固定尾串结束"),
    ("C66d 写入件填值判读行的那条固定尾串换成档案里不存在的拼写（同一族的第二条，各配一条红）",
     [(r'assert v2\.endswith\("DERIVED, STRUCTURE GATES PASS"\)', 'assert v2.endswith("DERIVED, STRUCTURE GATES PASSED")')],
     "写入件的填值判读行不以那句固定尾串结束"),

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
     [(r'^LESSON_NEEDLE = "普查仪器没有第二出处"', 'LESSON_NEEDLE = "普查仪器没有第二出处X"')],
     "节内唯一性破了"),
    ("C85 本轮新学锚句写歪（同族的第二条）",
     [(r'^NEWLESSON_NEEDLE = "同日轮没有目录级证据"', 'NEWLESSON_NEEDLE = "同日轮没有目录级证据X"')],
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
    ("C95 前缀 census 退回裸节号普查（本轮实测裸族里有 §26.27／§26.29 那两枚旧号 ⇒「本节 ∪ 上一节」那条越界闸当场红）",
     [(r'set\(re\.findall\(r"开发日志 §26\\\.\(\\d\+\)", text\)\)', 'set(re.findall(r"§26\\.(\\d+)", text))')],
     "回执里出现的节号"),
    ("C95b 回执 ENV 行的节号与本件现取的上一节强行差一（两个出处不再互证）",
     [(r'assert int\(envsec\) == PREVSEC', "assert int(envsec) == PREVSEC + 1")],
     "回执 ENV 行的节号"),
    ("C95c 那句「承接上一节的引用必须有出处」的门槛抬到 99（现数 1 处也敢判成没出处 ⇒ 反向必红）",
     [(r'assert prev_secref >= 1', "assert prev_secref >= 99")],
     "那句「承接上一节」"),
    ("C95d 上一节那枚的裸节号搜索换成回执里根本没有的号（那句「承接上一节」真的失去靶；只放宽门槛是另一侧）",
     [(r're\.findall\(r"§26\\\.%d\(\?!\\d\)" % \(PREVSEC - 1\)', 're.findall(r"§26\\.%d(?!\\d)" % (PREVSEC - 3)')],
     "那句「承接上一节」"),
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
     [(r'WNU=str\(wv\["wnu"\]\), WNA=str\(wv\["wna"\]\), WNM=str\(wv\["wnm"\]\), WNT=str\(wv\["wnt"\]\),',
       'WNU=wv["wnu"], WNA=str(wv["wna"]), WNM=str(wv["wnm"]), WNT=str(wv["wnt"]),')],
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
    # 本轮实测（探针跑过一次，现场末行见坑⑪末尾那条登记）：克隆副本**走不到**写入件的守卫，也走不到那条写死分母
    # 的落笔闭合式——它先撞上写入件「未跟踪侧必须为空」那条闸（克隆件自己就是工作树里的一张未跟踪面，按形制必然命中）。
    # 那条红仍然是**真件的活闸**（档案面没落盘就拒绝写），只是不是本条原先声称的那句守卫。期望句取那句里的固定短语、
    # 不含读数（坑②）；该短语只在写入件源码里 ⇒ 本条的位置判定不适用，由「抛出的那一行」＋前后两次漂移比对把关。
    ("C110 克隆本归档的写入件并带 --write 跑它（不许真动档案：本轮它红在「未跟踪侧必须为空」那条闸）",
     "CLONE_WRITER", "与本轮登记的流程前提不符"),
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


def own_write_targets():
    """本进程 stdout／stderr 指向的**仓库内**面（＝本件自己的归档面，见坑㉑）。
    只有落在仓库里的那两个 fd 才放过；终端／管道／别处的路径一个都不算，真实运行里这张集合就是那两张 `.out`／`.err`。"""
    out = set()
    for fd in ("1", "2"):
        try:
            t = os.path.realpath("/proc/self/fd/" + fd)
        except OSError:
            continue
        rel = os.path.relpath(t, REPO)
        if os.path.exists(t) and not rel.startswith(".."):
            out.add(rel)
    return out


def drop_own(paths, own):
    """把「本件自己正在写的那两张面」从漂移清单里挑出去，其余（含空白行处理）逐字保留原口径。"""
    return [p.strip() for p in paths if p.strip() and p.strip() not in own]


def unstaged():
    """工作树对**暂存面**的漂移清单：为空＝那五张档案面自上次 `git add` 起逐字节未动。
    本件的归档面（stdout／stderr 那两张）本次运行必然偏离索引，按身份放过（坑㉑）。"""
    return drop_own(sh("git", "-c", "core.quotePath=false", "diff", "--name-only").splitlines(),
                    own_write_targets())


def run_clone_writer():
    """C110：把本归档的写入件按字节克隆到本目录，带 `--write` 跑它。

    安全性论证**不靠**「我相信它不会写」：本轮实测的红点是写入件自己的第一条现场闸——它的未跟踪侧计数把克隆件
    自己算进去（副本就落在那个目录里，按形制必然未跟踪），于是在任何追加之前 `AssertionError`。上一版这里写的是
    「守卫按整词命中拦停」，那条守卫在源码里排在闭合式之后，本轮**没走到**（＝坑⑪的同一个形状）。
    论证之外还有两道现场闸：跑之前与退出之后各数一次 `unstaged()`，非空即本件当场红，而不是留下一份被改过的档案。
    """
    assert not unstaged(), "跑克隆副本之前工作树对暂存面已有漂移：%s ⇒ 这条控制的「没写」判据失效，先补齐 git add" % (
        unstaged(),)
    src = rd(os.path.join(HERE, "am_s32_docs.py"))
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8",
                                     dir=HERE, prefix="zz_s32_writer_clone_") as fh:
        fh.write(src)
        tmp = fh.name
    try:
        p = subprocess.Popen([PY3, tmp, "--write"], cwd=REPO, env=ENV,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             start_new_session=True)
        try:
            out, err = p.communicate(timeout=SUB_TIMEOUT)
            msg = (err or "") + (out or "")
            rc = p.returncode
            if rc == 0:
                raise AssertionError("克隆副本竟然写盘成功（rc=0）⇒「未跟踪侧必须为空那条闸会拦住副本」这句是假话")
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            p.communicate()
            return "TIMEOUT", ""
        after = unstaged()
        assert not after, "克隆副本退出后工作树出现漂移 %s ⇒ 它的前置闸没拦住，档案面被真写过（这条红必须修仪器）" % after
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


def trace_lines(msg):
    """traceback 里本副本的**全部**帧号（入口→抛出点）。最后一个＝真正抛出的那一行（坑③）；更早的帧只在
    坑③″那条「参数句形状」里当作调用链证据，不参与「整段 stderr 命中」那种放宽。"""
    return [int(x) for x in re.findall(r'File "[^"]*%s[^"]*\.py", line (\d+)' % PREFIX, msg)]


def same_gate(gl, frames, tl):
    """位置判定（坑③／③′／③″），返回 (是否同一条闸门, 是否走参数句那一支)。

    直投＝raise 落在该句在本副本源码里**任一**出现处的 ±WINDOW；
    参数句＝该句的那一处落在 trace 的**更早一帧**（±WINDOW）⇒ 它确实在抛出链上，只是身份由实参传进助手。
    两支都不看「整段 stderr」：报错文本是否含该句由调用处那条 `hit` 单独把关。
    """
    if tl is None or not gl:
        return True, False
    if any(g - WINDOW <= tl <= g + WINDOW for g in gl):
        return True, False
    up = frames[:-1]
    routed = [g for g in gl if any(f - WINDOW <= g <= f + WINDOW for f in up)]
    return bool(routed), bool(routed)


def gate_lines(expect):
    key = expect.split("%")[0].strip() or expect
    return [i + 1 for i, l in enumerate(text.splitlines()) if key in l]


def run_one(bad):
    """在同一目录落一份扰动副本、跑它、拿 rc＋合并输出；超时＝环境故障（坑⑨），按**进程组**杀。

    副本按生成器那条按身份的豁免放行（`AM_S32_CTL_COPY`＝本副本的件名）：不加这枚环境变量，副本会先红在
    「落笔未跟踪侧不闭合」那条现场闸上（副本自己就是那张未跟踪面），每条控制都红在同一处＝整轮作废（本轮实测）。
    """
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8",
                                     dir=HERE, prefix=PREFIX) as fh:
        fh.write(bad)
        tmp = fh.name
    p = subprocess.Popen([PY3, tmp], cwd=REPO, env=dict(ENV, AM_S32_CTL_COPY=os.path.basename(tmp)),
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
    return [f for f in os.listdir(HERE) if f.startswith(PREFIX) or f.startswith("zz_s32_writer_clone_")]


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
     "红路要在档案目录里真有一张 am_s7_verify32.log＝本片的回执；按形制它只能由本片的落地腿在本片提交之后写出来，"
     "为造红预写一张假回执正是这条闸门要拦的事。本片实测：树里现有回执的最大片号＝31（C3 就是打这条的），所以"
     "它在本片是「不可能红」的一侧，到**分片 33** 变成活闸（那时 32 号回执已在树里，同一把 glob 会数到它）。"),
    ("角色桶空转（`assert not empty`，「那个角色的命名口径已变」）",
     "桶形改窄时那些件必然先落进 0 个桶而撞上逐件断言（C9／C11 落的那条），轮不到「空桶」这一句；要它单独红"
     "得让一整角色的件还在暂存清单里却不被任何桶认领＝改档案面的命名而不是改仪器。本轮另有两条把桶配错的红路"
     "（C12 整串反着配、C13 少配一组）＝同一族闸门的另一侧，都被真件拦住。"),
    ("`numstat 里有 - 行（二进制／rename）`、`kinds == [A, M]` 的 D 方向、面行数的「只允许净增」（`dele == 0`）",
     "三条都要往暂存集里放一张二进制面／一次删除，与本轮「只归档文本面、只净增」直接冲突。本轮实测："
     "`git diff --cached --numstat` 里以 `-` 开头的行＝0，状态字母只有 A 与 M 两类（无 D）；两列求和的机制已由 "
     "C21 覆盖，状态字母按位取行已由 C19 覆盖，`-` 行的检测本身没有本轮可造的坏输入。"),
    ("`-uall` 口径的三条（折叠目录行、未跟踪与落笔清单的交集、落笔清单重名）＋原 C69 的 needle 放宽",
     "本轮实测：`git status --porcelain -uall` 里以 `??` 开头但**不**以 `?? ` 开头的行＝0（未跟踪行永远是 `?? `＋"
     "路径），把 needle 从 `?? ` 放宽到 `??` 是**字面 no-op**、副本 rc＝0 ⇒ C69 不进 CASES（上一轮把它当红路是抄了"
     "更上一轮的形状）。另两条要工作树真的未跟踪且含整目录：本片的口径是「每件落盘即 `git add`」，生成器那侧把落笔"
     "的未跟踪侧钉成 0（＝C0f 验的那条活闸），所以到本件跑这一刻未跟踪侧为空集＝交集与折叠目录行都没有靶。重名那条"
     "按构造不成立（桶逐件互斥，C9 就是它的反例；本轮实测未跟踪那三行的基名两两不同）。"),
    ("`assert not mo`（工作树有未暂存的修改面）",
     "红路要真留下一处未暂存漂移，而漂移改不了副本源码（扰动改的是**副本**，这条读的是**现场**）。同族的"
     "needle／方向问题由 C20 覆盖（把已暂存当成未暂存⇒ 现数变成 0，那条等式当场红）。本轮这条**真的红过**：本件的"
     "生成器在 `git add` 之后又被改写过，`git status` 一度显示 `AM`＝索引里有、工作树里另有，修法＝改完立刻重新 "
     "`git add`；本件的 G5／C110 前后也各数一次同一把尺子，所以它是实跑的活判据，只是红路不能进 CASES。"),
    ("`assert \"LC_ALL\" not in os.environ`（生成器 finger_both 的第一条）",
     "那是**环境**前提，不是源码形状：红路要往子进程环境里注入 LC_ALL，而注入之后本件自己那条「环境里不许有 "
     "LC_ALL」（坑④）会先停。本件 main() 的三条前置断言就是这条的替身。"),
    ("`nvidia-smi 本轮又能用了`（普查仪器恢复时的拒绝）",
     "红路要真的把内核模块与用户态库对齐（重载模块或装包），那是共享状态且会打断用户的 GPU 进程 ⇒ 本轮不做。"
     "替代证据＝四条在**真包表**上取数的红路（C32／C33／C34／C36，含 rc 残留行与 i386 行）＋绿路 G4（ii／amd64 "
     "恰好留一个版本＝那条门槛可达），所以根因句的两个出处已被两个方向各验一次。"),
    ("模板里某一格的**语义**（两个读数都合法，只是填错了槽）",
     "本件的单位断言审的是「每条读数取自唯一出处」＋「槽的取值是一个字符串」，不是「哪一格填了哪个字段」⇒ 填错"
     "槽位时副本可以 rc＝0。本轮实测同一对槽（生成器源码 980 行的 RAWRB＝原始 stdout 字节数与 RAWRM＝同件 md5 前 "
     "12，两个都是字符串）互换后副本 **rc＝0、stdout 25045 B**，打印出来的那句成了「`_raw_verify31.out`（1a8eb7d1e867 B，"
     "md5 前 12 6933，stderr …）」＝形状全对、语义全错，那条红不了，所以它不进 CASES（放进去就是一条永红不了的控制＝又跑一遍"
     "真件）。形状那一侧有闸（C97：非字符串落进槽位⇒ 红在「取值不是字符串」）；本轮能被机器拦下的两处补强＝"
     "`ncode == 0`（C18，那句旁边写着死的 0 条）与正对照读数的取段方向（C96c，取错段就是取到同一个对照里那把预期"
     "为 0 的第二尺子）。其余语义仍要靠本片落地腿的归档自查件逐条比对原文，并由分片 33 复跑那张现场。"),
    ("写入件的两条拒绝：守卫那句「有仪器正在跑」与「wrapper 不参与判定」那一侧",
     "本轮实测（真探针一次）：C110 的克隆副本**连落笔闭合式都没走到**——它先撞上写入件更前面那条「未跟踪侧必须为"
     "空」的现场闸，而命中它的正是克隆件自己（副本按形制落在那个目录里＝一张未跟踪面）。所以 C110 登记的期望句是"
     "那句实际打红的闸门，期望句里只有固定短语、没有读数（坑②）；该短语只存在于**写入件**源码 ⇒ 本条的 raise 行号"
     "取不到本件副本的帧，位置判定不适用，改由「抛出的那一行」＋跑前跑后两次漂移比对把关。要红那句守卫得让副本"
     "**通过**它前面的现场闸＝档案面真落盘，而写入件一旦通过就会真改五张档案面 ⇒ 本轮不做（需要一份不会落盘的写入"
     "件替身，另立一轮）。「wrapper 不参与判定」那条写在写入件的注释与 TOKENS 里：造一条真在跑的 wrapper 现场就会"
     "撞上本件自己的红线（坑：本件全离线），所以本轮同样没有可造的坏输入。"),
    ("`selfctl_check` 的两条「命中 1 行」门槛（本件自己的锚点句与标记句重复）",
     "本件不能改自己的源码来造重复（改了就不是这一版仪器）。这条由**基线**承担：一旦任一常量重复出现，生成器副本 "
     "rc≠0 ⇒ 本件在基线那一步就停（坑①），整轮正对照作废——预检里那两条自件对照（正 1 处／负 0 处）是它的花前份。"),
    ("角色分组匹配那句里的 `re.escape(key)`（标签当字面量而不是正则那一层）",
     "本轮实测（第一次实跑的现场）：把它丢掉副本 **rc＝0**，期望句「分组第」没走到。两个原因叠在一起：① 只替换"
     "`re.escape(key)` 前半段会把整行拆成 `re.match(key) + r\"…\", seg[pos:]`，`m` 变成非空元组⇒`assert m` 恒真；"
     "② 更根本的是**标签里根本没有 ASCII 正则元字符**——按 ast 从生成器现取 `BKEY_PIN` 的 11 枚标签，含 "
     "`. * + ? ( ) [ ] { } | ^ \\` 的枚数＝**0**（本轮现算），所以 `re.escape` 在当下这一轮是**防御性**的一层，"
     "换一枚标签才可能有用。同一条闸门的**红路**由 C12（整串反着配）、C13（少配一组）与本轮改写的 C14"
     "（针尾那枚项间分隔符换成档案里不存在的字符）三条承担；这一层若要变成可判定的闸门，得先把「标签含元字符时"
     "仍按字面匹配」写成一条断言，那是口径改动而不是控件改动，本轮不做。"),
]


def main():
    assert os.path.exists(PY3), "钉住的解释器 %s 不在盘上 ⇒ 副本没有可钉的运行时，整轮正对照作废" % PY3
    # 坑④那条实测的替身：本件若在钉住解释器下跑，副本重跑归档自查件时会把它自己数成「占卡的解释器」，
    # 于是**每一条**控制都红在同一现场闸上（那不是判据）。这条门槛在花钱之前停，而不是留下一整轮假红。
    assert sys.executable != PY3, ("本件正用钉住的那个解释器跑（%s）⇒ 副本里那条 argv[0] 整词普查只会排除自身与父进程，"
                                   "本件必然被数进去，每条控制都红在同一条现场闸（见坑④）。请换解释器跑本件，副本仍走 %s"
                                   % (sys.executable, PY3))
    assert "LC_ALL" not in os.environ, "本件环境里有 LC_ALL ⇒ 生成器第一条就红（坑④），那不是闸门红"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "本件没带空串 CVD ⇒ 副本也拿不到（钉住口径失真）"
    if "--precheck" in sys.argv:
        sys.exit(0 if precheck() else 3)
    if not precheck():
        print("纸面预检不合格，不起子进程（先把锚点改对再花钱）")
        sys.exit(3)
    # 坑㉑：副本跑的是生成器，而生成器第一条现场闸是「工作树对暂存面零漂移」。刚写过、还没重新 `git add` 的
    # 任何一张跟踪面都会让**每一条**控制红在那条闸上（本轮实测一次），所以这一步在花钱之前停。
    pre_drift = unstaged()
    assert not pre_drift, ("跑基线之前工作树就有漂移 %s ⇒ 副本会红在生成器那条漂移闸上（那不是闸门红，正对照整轮作废）。"
                           "本件自己的归档面按 fd 放过，其余面先补齐 git add（见坑㉑）" % (pre_drift,))

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
        fs = trace_lines(msg)
        tl = fs[-1] if fs else None
        err = raised(msg)
        hit = expect in err
        near, routed = same_gate(gl, fs, tl)
        if gl and tl is not None:
            why = "句在源码 %s 行／raise 在 %d 行%s" % ("、".join(str(x) for x in gl[:3]), tl,
                                                    "（同一条闸门）" if near and not routed else
                                                    ("（同一条闸门＝参数句：句在调用链帧 %s 上）" % (
                                                        "、".join(str(f) for f in fs[:-1]))) if routed
                                                    else "（不是同一条！）")
        else:
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

    # 对**判定器自己**的对照（坑③″）：直投／参数句两支放行、句在链外与远句两支必须仍然红。
    # 走的就是真跑用的那个 `same_gate()`（不另写平行实现）；(near, routed) 两值一起验，
    # 否则「放宽成整段 stderr」那种写法会把 routed 也报成直投、把这条自查读成绿。
    _d = same_gate([80], [165, 80], 80)
    _p = same_gate([165], [165, 80], 80)
    _o = same_gate([970], [700, 966], 966)
    _f = same_gate([400], [165, 80], 80)
    _n = same_gate([], [165, 80], 80)
    ok2 = _d == (True, False) and _p == (True, True) and _o[0] is False and _f[0] is False and _n == (True, False)
    rows.append(("C-self-2 判定器的四支（直投／参数句／句在更早帧／句在链外，加位置判定不适用一支）",
                 3 if ok2 else 0, ok2, "四支各按其形状判对",
                 "本件自己的异常" if ok2 else "实得＝直投%s／参数句%s／更早帧%s／链外%s／不适用%s" % (
                     _d, _p, _o, _f, _n)))
    allred = allred and ok2

    # 漂移豁免的窄化对照（坑㉑）：放过本件归档面**且只**放过它。三个形状各判一次，全对才算这条在算。
    _face = os.path.relpath(HERE, REPO) + "/_raw_s32_msg_controls.out"
    _a = drop_own([_face, "docs/开发日志.md", ""], {_face})
    _b = drop_own([_face], {_face})
    _c = drop_own(["docs/开发日志.md"], set())
    ok3 = _a == ["docs/开发日志.md"] and _b == [] and _c == ["docs/开发日志.md"]
    rows.append(("C-self-3 漂移豁免只放过本件归档面（真档案面混进来必须仍然报）",
                 3 if ok3 else 0, ok3, "三个形状各按其口径判对",
                 "本件自己的异常" if ok3 else "实得＝混合%s／只本件%s／无豁免%s" % (_a, _b, _c)))
    allred = allred and ok3

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

    nself = sum(1 for r in rows if r[0].startswith("C-self-"))
    print("基线（未扰动副本）rc=%s 且基线标记句逐字节命中＝True（所有红的对照面）" % base_rc)
    print("解释器（本件／副本）＝%s／%s｜被扰动的件＝%s｜每条子运行上限＝%ds｜环境故障累计上限＝%d 条" % (
        sys.executable, PY3, SRC, SUB_TIMEOUT, ENV_FAIL_STOP))
    print("CASES 声明＝%d 条（全部红路，其中克隆别的仪器 %d 条）｜保持绿＝%d 条｜本件自己的锚点与判定器自查＝%d 条｜"
          "超时（环境故障，不判红也不判过）＝%d 条｜因超时没跑到的＝%d 条" % (
              len(CASES), sum(1 for _n, s, _e in CASES if s == "CLONE_WRITER"), len(GREENS), nself, envfail, envfail))
    print("判红窗口＝raise 行号落在该句在本副本源码里**任一**出现处的 ±%d 行（坑③′：通用报错句在源码里两处，"
          "单点判定会把真红判成不合格）；不落在抛出点时只再走一支＝该句出现在 trace 的更早一帧（坑③″ 参数句形状），"
          "两支都由 C-self-2 双向把关" % WINDOW)
    print("扰动条数＝%d｜每条判据＝rc≠0 且**抛出的那一行**含指定闸门句 且 行号落在窗口内" % (len(rows) + envfail))
    print("工作树对暂存面的漂移：基线之前、C110 前后、整轮末尾各数一次，全为空＝那五张档案面本轮逐字节未动"
          "（G 族与 C110 的现场闸）；本件自己的归档面按写入 fd 放过＝坑㉑，窄化由 C-self-3 双向把关")
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
