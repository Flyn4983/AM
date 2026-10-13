#!/usr/bin/env python3
"""只读仪器 `am_s31_hdr_after.py` 的**控件面**：它每一条闸门都要被造一次红，否则那条闸门是空转。

与表头控件面的三条口径差别（这三条差别就是本件的存在方式）：
① **一次 API 都不花**：本件不联网，每条扰动只是"临时目录里几张副本＋一次本地子进程"，所以它一口气
   跑完几十条，没有配额闸门、没有跨窗口等待，也不该出现"环境故障"那一类去向（超时仍单列计数）。
② **扰动落在副本上，不落在档案上**：副本连仪器自己一起抄进 `HERE` 下的临时子目录（前缀 `zctl_`），
   仪器从子目录里跑，四张面用**绝对路径参数**指回同一批副本，子进程 `cwd` 也钉在该目录 ⇒ 真档案全程
   只被读（每条子运行之后都现比一次起点 md5 快照，改动即整面退出）。上一片实测到：把仪器抄进子目录时，
   "仓库根＝目录上三层"那个写法会解出 `docs/evidence`，于是 HEAD 提交说明与上一片标本 glob 两头都取不到
   ⇒ 一条好闸门在**合法的跑法**里假拒绝（本片的仪器因此向上找 `.git`，本面第 G5 条就是这条口径的绿路）。
③ **判红口径与表头面同制**：rc≠0 且**抛出的那一行**含指定闸门句，且 traceback 里指向本副本的行号落在
   该句在**副本源码**里的 ±3 行窗口；每条扰动先经**纸面锚点普查**（锚在真文件里恰好 1 处、扰动后的副本
   仍能 `compile()`）才允许起子进程，装配期抓不到锚的那条记成 `×`（装配失败）而不是把整面崩掉——
   本族上一片栽过"把批读数攒到最后统一印，结果被收面自查件崩掉时整批丢掉"。

第一版草稿里有四条**期望句写歪**的，都是同一族（把"红在别处"当成"红在靶上"），本面按实测落点重写：
· 改文件名不会让仪器看到新名字之外的东西：`rename` 必须连参数一起改指，否则红在「输入面不存在」；
· 从生成器 `FIELDS` 里**删一项**，那枚标签就从匹配式的并集里消失了，回执里那一行于是"顶格却不像字段"
  ⇒ 先红在 loose 那条而不是字段数那条（＝记忆里「计数闸门只剩单边证据」的形状）。字段数那条改由
  **表头侧**改名造红；生成器侧那一条按实测落点登记，删掉的那一侧写进 UNCOVERED。
· 把标签行整体缩进同样会让它从 `defs` 里掉出去（先红字段数）⇒ 改为**插一条**缩进的标签＋冒号行。
· 往值尾部塞占位符（不是塞进行首）才不会破坏 `defs`，否则残留检查根本轮不到。

绿路（必须保持绿）也是判据的一部分：只造红会把"闸门写过头"当成合格——G2 那条表头等长改字、G4 那条
needle 集合缩小、G6 那条正文区与原始件同改，都是"合法编辑不许被拒"的方向；缺了它们，本面只会教我把
判据越写越严。**但"良性"的边界是这张面自己划的**：本面第一版的两条绿路（表头值尾加两个空格、正文末尾
加一个换行）实测都红了，红在「写盘面报出」——回执的总字节数被写盘面那张家档案记着，而⑨ 又把两者互证，
所以任何**改变回执长度**的编辑在这里都不存在"无害"一说（上一片的表头控件面记过同一族：一个档案的字节数
被另一个交叉闭合的档案钉住 ⇒ 良性编辑按构造不存在）。改法等长编辑，并把这条写进断言（`equal_both` 那条
先查等长）＝把发现落成闸门，而不是只在注释里说一次。

末行判读＝`CONTROL_VERDICT = ALL CONTROLS CAN GO RED`（rc=0）；任何一条红路没红在指定闸门上＝
`PROBLEM（有闸门空转）`（rc=3）；有子运行超时＝`INCOMPLETE`（rc=4，既不判红也不判过）。
"""
import hashlib
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DAYDIR = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(DAYDIR))
PREFIX = "zctl_s31_after_"
INSTR = os.path.join(HERE, "am_s31_hdr_after.py")
GEN = os.path.join(HERE, "am_s31_hdr.py")
LOG = os.path.join(HERE, "am_s7_verify31.log")
RAW = os.path.join(HERE, "_raw_verify31.out")
WF = os.path.join(HERE, "_raw_s31_hdr_write.out")
SPEC_PREV = os.path.join(REPO, "docs", "evidence", "2026-10-12", "_raw_s30_hdr_after_red1.err")
DEVLOG = os.path.join(REPO, "docs", "开发日志.md")
FACES = {"log": LOG, "raw": RAW, "gen": GEN, "writeface": WF}
KEYS = ("log", "raw", "gen", "writeface")
ARCHIVE = [INSTR, GEN, LOG, RAW, WF, SPEC_PREV, DEVLOG]
TIMEOUT = 60
ENV_FAIL_STOP = 3
GREEN_MARK = "HDR_AFTER_VERDICT = ALL READ-ONLY CHECKS PASS"
TIMEOUTS = []

# ---------- 副本装配：面按**字节**切表头／正文，改完表头必须重打 BODY_OFFSET ----------


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def snapshot():
    return {p: md5(p) for p in ARCHIVE if os.path.exists(p)}


def split_log(b):
    m = re.search(rb"^BODY_OFFSET=(\d+)$", b, re.M)
    assert m, "回执里取不到 BODY_OFFSET 行 ⇒ 装配器的口径变了"
    off = int(m.group(1))
    assert b[off - 1:off] == b"\n", "off 不在行首 ⇒ 装配器读歪"
    return b[:off], b[off:]


def off_of(head):
    return int(re.search(rb"^BODY_OFFSET=(\d+)$", head, re.M).group(1))


def restamp(head, newoff):
    return re.sub(rb"^BODY_OFFSET=\d+$", b"BODY_OFFSET=%d" % newoff, head, flags=re.M)


def apply_pairs(blob, pairs, tag):
    for old, new in pairs:
        o, n = old.encode("utf-8"), new.encode("utf-8")
        assert blob.count(o) == 1, "%s：锚 %r 在副本里命中 %d 次（须 1）" % (tag, old[:36], blob.count(o))
        blob = blob.replace(o, n)
    return blob


def log_head(path):
    return split_log(open(path, "rb").read())[0]


def build(dest, face_ops=None, instr_subs=None, instr_name="am_s31_hdr_after.py", raw_trim=0):
    """装配一份临时运行环境，返回（仪器副本路径, 参数表, 临时目录, 副本源码文本）。"""
    d = os.path.join(HERE, dest)
    os.makedirs(d)
    argd = {}
    for k in KEYS:
        p = os.path.join(d, os.path.basename(FACES[k]))
        shutil.copyfile(FACES[k], p)
        argd[k] = p
    if raw_trim:
        p = argd["raw"]
        b = open(p, "rb").read()
        open(p, "wb").write(b[:len(b) - raw_trim])
    ip = os.path.join(d, instr_name)
    shutil.copyfile(INSTR, ip)
    src_txt = open(ip, encoding="utf-8").read()
    if instr_subs:
        src_txt = apply_pairs(src_txt.encode("utf-8"), instr_subs, "仪器 " + dest).decode("utf-8")
        open(ip, "w", encoding="utf-8").write(src_txt)
    for k, ops in (face_ops or {}).items():
        p = argd[k]
        b = open(p, "rb").read()
        if k == "log":
            head, body = split_log(b)
            head = apply_pairs(head, ops, "表头 " + dest)
            b = restamp(head, len(head)) + body
        else:
            b = apply_pairs(b, ops, k + " " + dest)
        open(p, "wb").write(b)
    return ip, argd, d, src_txt


# ---------- 装配之后的追加操作：需要知道真字节口径的那些 ----------

WF_PATS = {
    "bytes": (r"(写盘完成： \S+ )(\d+)( bytes)", lambda m: "%s%d%s" % (m.group(1), int(m.group(2)) + 1, m.group(3))),
    "cmp": (r"(cmp 复核 rc（同偏移须 0）＝ \d+ ｜左 \+1 须非 0＝ )\d+( ｜右 \+1 须非 0＝ )\d+", r"\g<1>0\g<2>0"),
    "fields": (r"(｜顶格字段行数＝ )(\d+)( / \d+)", lambda m: "%s%d%s" % (m.group(1), int(m.group(2)) - 1, m.group(3))),
}
WF_LINE_NEEDLE = "cmp 复核 rc（同偏移须 0）＝"


def edit_head(lp, needle, mode, payload):
    b = open(lp, "rb").read()
    hd, bd = split_log(b)
    nl = needle.encode("utf-8")
    assert hd.count(nl) == 1, "表头锚 %r 命中 %d 次（须 1）" % (needle, hd.count(nl))
    out = []
    for l in hd.splitlines(True):
        if nl in l:
            if mode == "append":
                l = (l[:-1] if l.endswith(b"\n") else l) + payload.encode("utf-8") + b"\n"
            elif mode == "sub":
                l = l.replace(nl, payload.encode("utf-8"))
            if mode == "before":
                out.append(payload.encode("utf-8") + b"\n")
        out.append(l)
    hd2 = b"".join(out)
    open(lp, "wb").write(restamp(hd2, len(hd2)) + bd)


def run_extra(d, argd, post):
    for item in post:
        kind = item[0]
        if kind == "rename":
            k, newname = item[1], item[2]
            np = os.path.join(d, newname)
            os.rename(argd[k], np)
            argd[k] = np  # 参数必须跟着改指：否则红在「输入面不存在」而不是形制那条
        elif kind == "drop":
            os.remove(argd[item[1]])
        elif kind == "head_edit":
            edit_head(argd["log"], item[1], item[2], item[3])
        elif kind == "off_shift":
            b = open(argd["log"], "rb").read()
            hd, bd = split_log(b)
            open(argd["log"], "wb").write(restamp(hd, off_of(hd) + item[1]) + bd)
        elif kind == "dup_offset_line":
            b = open(argd["log"], "rb").read()
            hd, bd = split_log(b)
            ln = [l for l in hd.splitlines(True) if re.match(rb"^BODY_OFFSET=\d+$", l)][0]
            open(argd["log"], "wb").write(hd + ln + bd)
        elif kind == "body_last_byte":
            b = open(argd["log"], "rb").read()
            hd, bd = split_log(b)
            open(argd["log"], "wb").write(hd + bd[:-1] + bytes([(bd[-1] + item[1]) % 256]))
        elif kind == "last_line_both":
            o, n = item[1], item[2]
            for p, blob in ((argd["log"], open(argd["log"], "rb").read()),
                            (argd["raw"], open(argd["raw"], "rb").read())):
                hd, bd = split_log(blob) if p == argd["log"] else (b"", blob)
                tgt = bd if p == argd["log"] else blob
                assert tgt.count(("ALL CHECKS " + o).encode()) == 1, "末行锚句在这张面里不是 1 处：" + p
                new = tgt.replace(("ALL CHECKS " + o).encode(), ("ALL CHECKS " + n).encode())
                # 等长是前提：这条扰动不走 restamp，正文变长会让盘上的 BODY_OFFSET 变成过期读数，
                # 于是红落在「偏移行」那族而不是末行那条——判据会响，但响的不是本条要验的闸门。
                assert len(new) == len(tgt), "末行扰动不等长 ⇒ 正文区长度变了，BODY_OFFSET 会过期"
                open(p, "wb").write((hd + new) if p == argd["log"] else new)
        elif kind == "equal_both":
            o, n = item[1].encode("utf-8"), item[2].encode("utf-8")
            assert len(o) == len(n), "良性扰动不等长 ⇒ 一改就把总字节数推离写盘面那个读数"
            b = open(argd["log"], "rb").read()
            hd, bd = split_log(b)
            rp = argd["raw"]
            rb = open(rp, "rb").read()
            assert bd.count(o) == 1, "正文区锚 %r 命中 %d 次（须 1）" % (item[1], bd.count(o))
            assert rb.count(o) == 1, "原始件锚 %r 命中 %d 次（须 1）" % (item[1], rb.count(o))
            # 表头**不动**：写盘面那张家档案里记着回执的总字节数，且与本件现读的值互证（⑨ 两出处闭合），
            # 所以任何改变回执长度的编辑——哪怕逐字节自洽——都会红在「写盘面报出」。这条是本面第一版
            # 实测到的：G2／G6 两条"看着无害"的编辑都栽在那上面（不是闸门坏，是这张面确实被跨档案钉住）。
            open(argd["log"], "wb").write(hd + bd.replace(o, n))
            open(rp, "wb").write(rb.replace(o, n))
        elif kind == "wf_num":
            pat, repl = WF_PATS[item[1]]
            t = open(argd["writeface"], encoding="utf-8").read()
            assert len(re.findall(pat, t)) == 1, "写盘面数字锚 %s 命中不是 1 处" % item[1]
            open(argd["writeface"], "w", encoding="utf-8").write(re.sub(pat, repl, t))
        elif kind == "wf_line":
            wp = argd["writeface"]
            t = open(wp, encoding="utf-8").read()
            assert t.count(WF_LINE_NEEDLE) == 1, "写盘面锚句命中不是 1 处"
            ln = [l for l in t.split("\n") if WF_LINE_NEEDLE in l][0]
            open(wp, "w", encoding="utf-8").write(
                t.replace(ln + "\n", "") if item[1] == "drop" else t.replace(ln, ln + "\n" + ln))


# ---------- 子运行清单：红路 (名字, build kwargs, 装配后操作, 期望闸门句)；绿路 (名字, kwargs, 操作) ----------

IS_REMOVE = ("    return bad\n\n\ndef fields_from_gen",
             '    return bad\n    os.remove("zctl_unreachable_after_return.py")  '
             "# 只被 ast 普查看见，运行时到不了\n\n\ndef fields_from_gen")
IS_WRITE = ("def once(pattern, text, what, flags=0):\n    m = re.findall(",
            'def once(pattern, text, what, flags=0):\n    sys.stdout.write("")\n    m = re.findall(')

CASES = [
    ("A1 log 面文件名不合形制（片号取不到）",
     dict(), [("rename", "log", "am_s7_verify31x.log")], "不符合形制"),
    ("A2 gen 面用上一片的文件名（四张面互等那条）",
     dict(), [("rename", "gen", "am_s30_hdr.py")], "四张输入面片号不一致"),
    ("A3 仪器副本自己的文件名片号回退一片（本件文件名那一向）",
     dict(instr_name="am_s30_hdr_after.py"), [], "片号三向互证不等"),
    ("A4 HEAD 提交说明的形制指歪（第三向读不到）",
     dict(instr_subs=[('SUBJECT_PAT = r"S7 分片 (\\d+)"', 'SUBJECT_PAT = r"S8 分片 (\\d+)"')]),
     [], "HEAD 提交说明里取不到形制"),
    ("A5 git 调用没成（第三向没了出处）",
     dict(instr_subs=[('pr = subprocess.run(["git", "-C", REPO,',
                       'pr = subprocess.run(["git", "-C", "/nonexistent-zctl",')]),
     [], "git 调用没成"),
    ("A6 上一片标本的形制指歪（全树 glob 命中 0）",
     dict(instr_subs=[('SPEC_PAT = "_raw_s%d_hdr_after_red1.err"', 'SPEC_PAT = "_raw_s%d_zctl.err"')]),
     [], "按全树 glob 命中"),
    ("A7 标本症状 needle 指到原文里没有的那句（引用对不上原文）",
     dict(instr_subs=[('"回执表头里有顶格却不像字段的行", "① 本地 HEAD"',
                       '"回执表头里有顶格却不像字段的行", "① 查无此标签"')]),
     [], "里查不到症状"),
    ("A8 仪器副本里塞一条 os.remove（只读普查）",
     dict(instr_subs=[IS_REMOVE]), [], "本件不是只读件"),
    ("A9 仪器副本里塞一条 .write(（只读普查的另一支构造）",
     dict(instr_subs=[IS_WRITE]), [], "本件不是只读件"),
    ("A10 输入面少一张（参数指到不存在的文件）",
     dict(), [("drop", "log")], "输入面不存在"),
    ("A11 偏移行前多一个空格（顶格变缩进＝读不回偏移）",
     dict(face_ops={"log": [("BODY_OFFSET=", " BODY_OFFSET=")]}), [], "BODY_OFFSET 行不是恰好 1 条"),
    ("A12 偏移行等号换成全角（同样读不回偏移）",
     dict(face_ops={"log": [("BODY_OFFSET=", "BODY_OFFSET＝")]}), [], "BODY_OFFSET 行不是恰好 1 条"),
    ("A13 偏移行复制一份（出处歧义）",
     dict(), [("dup_offset_line",)], "BODY_OFFSET 行不是恰好 1 条"),
    ("A14 偏移值 +1（不落在行首）",
     dict(), [("off_shift", 1)], "BODY_OFFSET 不落在行首"),
    ("A15 原始件被截掉 1 字节（正文区长度闭合那条）",
     dict(raw_trim=1), [], "两把尺子对不上"),
    ("A16 回执正文区末字节改一个而原始件不动（cmp 三联）",
     dict(), [("body_last_byte", 1)], "cmp 三联不是 0"),
    ("A17 表头把 ENV 标签改名（字段数与声明不吻合）",
     dict(), [("head_edit", "ENV     :", "sub", "NVI     :")], "回执顶格字段"),
    ("A18 生成器 FIELDS 少声明一项（实测落点在 loose 那条，见 UNCOVERED）",
     dict(face_ops={"gen": [('"SRC指纹", "GPU", "RUN", "判读",', '"SRC指纹", "GPU", "RUN",')]}),
     [], "回执表头里有顶格却不像字段的行："),
    ("A19 生成器 FIELDS 前两项互换（顺序互证）",
     dict(face_ops={"gen": [('FIELDS = ["生成时刻", "CMD",', 'FIELDS = ["CMD", "生成时刻",')]}),
     [], "字段顺序与 FIELDS 声明不同序"),
    ("A20 生成器里把 FIELDS 赋值改掉（ast 现取断了）",
     dict(face_ops={"gen": [("FIELDS = [", "ZCTL_FIELDS = [")]}), [], "没有 FIELDS 赋值"),
    ("A21 FIELDS 里混进一枚非字符串元素（声明口径变了）",
     dict(face_ops={"gen": [('"判读",', "(13), \"判读\",")]}), [], "非字符串元素"),
    ("A22 表头里插一条缩进的标签＋冒号行（粘连漏了续行缩进）",
     dict(), [("head_edit", "ENV     :", "before", "    GPU     : 伪续行")],
     "缩进处出现「标签＋冒号」"),
    ("A23 表头里插一条顶格杂行（顶格却不像字段）",
     dict(), [("head_edit", "ENV     :", "before", "ZCTL 顶格杂行")], "回执表头里有顶格却不像字段的行："),
    ("A24 盘上表头值尾残留半角占位符",
     dict(), [("head_edit", "ENV     :", "append", " @@ZCTL@@")], "残留半角占位符"),
    ("A25 盘上表头值尾含全角 at 号（那种占位符不会被替换）",
     dict(), [("head_edit", "ENV     :", "append", " ＠ZCTL＠")], "含全角 at 号"),
    ("A26 仪器里判读行常量写死成别的值（末行双向查的第一侧）",
     dict(instr_subs=[('LAST_LINE = "LANDING_VERDICT = ALL CHECKS PASS"',
                       'LAST_LINE = "LANDING_VERDICT = ALL CHECKS PASSES"')]),
     [], "原始件末行与判读行常量不符"),
    ("A27 回执与原始件的末行一起等长改字（cmp 照旧绿、末行那条该红）",
     dict(), [("last_line_both", "PASS", "PAST")], "原始件末行与判读行常量不符"),
    ("A28 写盘面总字节数与盘上不吻合（两出处闭合）",
     dict(), [("wf_num", "bytes")], "写盘面报出"),
    ("A29 写盘面 cmp 三联读数改成 0／0",
     dict(), [("wf_num", "cmp")], "写盘面 cmp 三联"),
    ("A30 写盘面字段数改成 15／16",
     dict(), [("wf_num", "fields")], "写盘面字段数"),
    ("A31 写盘面那四联行整行删掉（once 闸门没有出处）",
     dict(), [("wf_line", "drop")], "在文件里命中"),
    ("A32 写盘面那四联行重复一次（once 闸门歧义）",
     dict(), [("wf_line", "dup")], "在文件里命中"),
]

GREENS = [
    ("G2 良性：表头值里**等长**改两字（标签、偏移、总字节数都不动，判据不许拒）",
     dict(), [("head_edit", "CPU 钉住", "sub", "CPU 钉牢")]),
    ("G3 良性：仪器副本只加一行无害注释（只读普查仍要说「没写的」）",
     dict(instr_subs=[("import argparse", "import argparse  # 一条无害注释")]), []),
    ("G4 良性：标本症状 needle 集合缩到只留第一枚（逐枚查不是集合相等）",
     dict(instr_subs=[('SYMPTOM_NEEDLES = ("回执表头里有顶格却不像字段的行", "① 本地 HEAD")',
                       'SYMPTOM_NEEDLES = ("回执表头里有顶格却不像字段的行",)')]), []),
    ("G5 良性：四张面与仪器一起抄进临时目录、从那里跑（仓库根向上找 .git 的合法跑法）",
     dict(), []),
    ("G6 良性：正文区与原始件同改、逐字节一致且等长（cmp 与两出处闭合读的是自洽性，不是缓存文本）",
     dict(), [("equal_both", "898 == 899", "897 == 898")]),
]

UNCOVERED = [
    ("⑩「输入面 md5 跑前后被改写」那条",
     "本面全程只读真档案，扰动一律落在临时副本；要让它红就得在两次 md5 读之间改一张面，那需要在仪器里"
     "塞一个自我干扰的钩子＝拆闸门而不是坏输入。⇒ 这条代码路径本面没造红，如实登记。"),
    ("①「只读普查」里 `os.remove` 与 `.write(` 两支共用一条 assert",
     "A8／A9 各自把一支构造装进副本，但抛出行是同一条 assert（坏构造列表一起进消息）⇒「哪一支被抓到」"
     "在这条判据上没有分辨力。本面声称的是「两条构造都能让那条 assert 红」，不声称能分辨两支。"),
    ("②「本件不联网」没有闸门读它",
     "仪器源码里根本不起任何网络进程，所以「它联网了」这类坏输入没有对应 assert。保护是"
     "「读数现取＋只读普查」的做法本身；下一轮若要拦，得先加普查再配控件。这条是实测到的**保护缺口**，不是漏测。"),
    ("⑧第二条末行断言「正文区末行不是判读行」",
     "它与前一条（原始件末行）在同一条件下按代码顺序先后触发：末行只要变了，前一条必先响。A26／A27 验到的是"
     "前一条；后一条**没有任何输入能单独触发** ⇒ 登记为按构造不可造红，不是本面漏跑。"
     "（同 §26.30 那条「同一条 assert 的两个分支没有各自可造红的入口」的形状。）"),
    ("⑥「表头按字节切完再编码回去不等于 off」那条",
     "它是字节／字符口径的钉子，只有把仪器里 `lb[:off].decode()` 换成 `txt[:off]` 才会红——那是拆闸门；"
     "而上一片那一版真的这么错过，标本 `_raw_s30_hdr_after_red1.err` 由仪器第 ⑤′ 环现取并核对症状 needle，"
     "所以本面把「历史事故」当证据、不把「拆开的仪器」当坏输入。"),
    ("④「回执顶格字段数 ≠ FIELDS 声明数」从生成器一侧造不出红",
     "把 `FIELDS` 里的一枚删掉，那枚标签同时从匹配式的并集里消失 ⇒ 回执那一行不再被认成字段，"
     "`defs` 与 `dec` 一起减 1，计数那条恒真，红先落在 loose 那条（A18 实测到）。所以本面从**表头侧**改名"
     "给这条计数闸门造红（A17），生成器侧那一条按实测落点登记；「声明侧单侧不可达」写在这里而不是假装验过。"),
]

SELFCHK = ["N1 期望句与锚点普查（每条期望句在仪器源码恰好 1 处且属某条 assert／raise＋每个装配锚与每个"
           "装配后锚在真文件恰好 1 处＋扰动后的副本仍能 compile）",
           "N2 装配器双向（restamp 改得对、切分拼得回、查无此锚要抓得出）",
           "N3 判红谓词五形状双向（真红须收／别的句须拒／rc=0 须拒／行号离句行 >3 须拒／无 traceback 须拒）",
           "N4 md5 快照双向"]


def gate_lines(txt, expect):
    return [n + 1 for n, l in enumerate(txt.splitlines()) if expect in l]


def raised(msg):
    ls = [l for l in msg.splitlines() if l.strip()]
    for l in reversed(ls):
        if re.search(r"(AssertionError|UnicodeDecodeError|SystemExit|SyntaxError|Error)\b", l) \
                and not l.strip().startswith("File "):
            return l
    return ls[-1] if ls else ""


def trace_line(msg):
    hit = re.findall(r'File "[^"]*%s[^"]*", line (\d+)' % re.escape(PREFIX), msg)
    return int(hit[-1]) if hit else None


def judge(rc, err, expect, src_txt):
    gl = gate_lines(src_txt, expect)
    if len(gl) != 1:
        return False, "期望句在副本源码里 %d 处（须 1，±3 窗口才有唯一靶）：%s" % (len(gl), gl)
    if rc == 0:
        return False, "副本 rc=0，没红"
    line = raised(err)
    if expect not in line:
        return False, "抛出行不含指定闸门句（实得：" + line[:60] + "）"
    tl = trace_line(err)
    if tl is None:
        return False, "traceback 里没有指向本副本的行号"
    if abs(gl[0] - tl) > 3:
        return False, "raise 行 %s 不在句行 %d 的 ±3 窗口" % (tl, gl[0])
    return True, "句在副本 [%d] 行／raise 在 %s 行" % (gl[0], tl)


def subs_of(spec):
    return spec.get("instr_subs") or []


def face_ops_of(spec):
    return (spec.get("face_ops") or {}).items()


def post_anchors(post):
    """把装配后操作依赖的锚翻成可普查的 (说明, 命中数) —— 普查对象是**真文件**（副本在扰动前与它逐字节相同）。"""
    head = log_head(LOG)
    wf = open(WF, encoding="utf-8").read()
    raw = open(RAW, "rb").read()
    body = split_log(open(LOG, "rb").read())[1]
    out = []
    for item in post:
        kind = item[0]
        if kind == "rename":
            out.append(("rename 源＝%s 存在" % item[1], 1 if os.path.isfile(FACES[item[1]]) else 0))
        elif kind == "drop":
            out.append(("drop 目标＝%s 存在" % item[1], 1 if os.path.isfile(FACES[item[1]]) else 0))
        elif kind == "head_edit":
            out.append(("表头锚 %r" % item[1], head.count(item[1].encode("utf-8"))))
        elif kind in ("off_shift", "dup_offset_line", "body_last_byte"):
            out.append(("表头 BODY_OFFSET 行", len(re.findall(rb"^BODY_OFFSET=\d+$", head, re.M))))
        elif kind == "last_line_both":
            out.append(("回执正文末行锚 %r" % item[1], body.count(("ALL CHECKS " + item[1]).encode())))
            out.append(("原始件末行锚 %r" % item[1], raw.count(("ALL CHECKS " + item[1]).encode())))
        elif kind == "equal_both":
            out.append(("正文区锚 %r" % item[1], body.count(item[1].encode("utf-8"))))
            out.append(("原始件锚 %r" % item[1], raw.count(item[1].encode("utf-8"))))
            out.append(("等长前提", 1 if len(item[1].encode()) == len(item[2].encode()) else 0))
        elif kind == "wf_num":
            out.append(("写盘面数字锚 %s" % item[1], len(re.findall(WF_PATS[item[1]][0], wf))))
        elif kind == "wf_line":
            out.append(("写盘面行锚", wf.count(WF_LINE_NEEDLE)))
    return out


def compile_check(text, name):
    try:
        compile(text, name, "exec")
        return None
    except SyntaxError as e:
        return "扰动后的副本 %s 编译不过：%s（那会红在 SyntaxError 而不是指定闸门）" % (name, str(e)[:70])


def precheck():
    errs = []
    src = open(INSTR, encoding="utf-8").read()
    wf_text = open(WF, encoding="utf-8").read()
    names = [c[0] for c in CASES] + [g[0] for g in GREENS]
    assert len(set(names)) == len(names), "有两条子运行同名：%s" % (
        sorted(x for x in names if names.count(x) > 1))
    for name, spec, post, expect in CASES:
        gl = gate_lines(src, expect)
        if len(gl) != 1:
            errs.append("%s：期望句「%s」在仪器源码里命中 %d 处（须 1）%s" % (name, expect, len(gl), gl))
            continue
        ln = gl[0]
        ctx = src.split("\n")[ln - 1] + (src.split("\n")[ln - 2] if ln >= 2 else "")
        if "assert" not in ctx and "raise" not in ctx:
            errs.append("%s：句所在行 %d 不属于任何 assert／raise ⇒ 那不是闸门句" % (name, ln))
        # ±3 窗口的前提：语句首行与句所在行不能隔太远（多行 assert 的 traceback 报**首行**）
        if len(src.split("\n")[ln - 1]) > 0 and not re.search(r"^\s*(assert|raise)", src.split("\n")[ln - 1]):
            back = ln - 1
            while back > 0 and not re.search(r"^\s*(assert|raise)", src.split("\n")[back - 1]):
                back -= 1
            if ln - back > 3:
                errs.append("%s：句在 %d 行而语句首行在 %d 行（相隔 >3）⇒ ±3 窗口够不着" % (name, ln, back))
    for name, spec, post, *_ in CASES + [(g[0], g[1], g[2], "") for g in GREENS]:
        try:
            c = compile_check(apply_pairs(src.encode("utf-8"), subs_of(spec), name).decode("utf-8"), name)
        except AssertionError as e:
            errs.append("%s：仪器锚 %s" % (name, str(e)[:100]))
            c = None
        if c:
            errs.append(c)
        for k, ops in face_ops_of(spec):
            blob = open(FACES[k], "rb").read()
            try:
                if k == "log":
                    head, body = split_log(blob)
                    blob = apply_pairs(head, ops, k) + body
                else:
                    blob = apply_pairs(blob, ops, k)
            except AssertionError as e:
                errs.append("%s：%s" % (name, str(e)[:100]))
                continue
            if k == "gen":
                c = compile_check(blob.decode("utf-8"), name + "/gen")
                if c:
                    errs.append(c)
        for what, n in post_anchors(post):
            if n != 1:
                errs.append("%s：%s 命中 %d 次（须 1）⇒ 装配期会抓不到锚，那条不是红是崩" % (name, what, n))
    for key in ("bytes", "cmp", "fields"):
        if len(re.findall(WF_PATS[key][0], wf_text)) != 1:
            errs.append("写盘面数字锚 %s 在**真**写盘面里命中不是 1 处 ⇒ A28／A29／A30 没有靶" % key)
    if wf_text.count(WF_LINE_NEEDLE) != 1:
        errs.append("写盘面四联行锚句命中不是 1 处 ⇒ A31／A32 没有靶")
    # N3 判红谓词五形状双向
    _cp = "x/" + PREFIX + "c1/am_s31_hdr_after.py"
    pos = ('Traceback (most recent call last):\n  File "%s", line 192, in main\n'
           '    assert not bad, "本件不是只读件，普查到写盘动作："\n'
           'AssertionError: 本件不是只读件，普查到写盘动作：os.remove\n' % _cp)
    ok, why = judge(1, pos, "本件不是只读件", src)
    if not ok:
        errs.append("N3 正例判错（真红被自己的谓词判成空转）：" + why)
    neg = ('Traceback (most recent call last):\n  File "%s", line 192, in main\n'
           '    assert not bad\nAssertionError: 别的闸门句\n' % _cp)
    ok2, why2 = judge(1, neg, "本件不是只读件", src)
    if ok2:
        errs.append("N3 反例没抓出：抛出行不含指定闸门句却判成命中")
    ok3, _w3 = judge(0, GREEN_MARK + "\n", "本件不是只读件", src)
    if ok3:
        errs.append("N3 rc=0 却判命中 ⇒ 把「没红」当成红")
    far = ('Traceback (most recent call last):\n  File "%s", line 12, in main\n'
           '    assert False\nAssertionError: 本件不是只读件\n' % _cp)
    ok4, why4 = judge(1, far, "本件不是只读件", src)
    if ok4:
        errs.append("N3 第四条反例没抓出：raise 行离句行 >3 却判成命中（%s）" % why4)
    notb = ('%s\n' % GREEN_MARK)
    ok5, why5 = judge(1, notb, "本件不是只读件", src)
    if ok5:
        errs.append("N3 第五条反例没抓出：rc≠0 但没有指向副本的 traceback 却判成命中（%s）" % why5)
    if len(gate_lines(src, "本件不是只读件")) != 1 or len(gate_lines(src, "查无此句的闸门")) != 0:
        errs.append("N3 句行普查形状变了（须 1／0）")
    # N2 装配器三形状双向
    b = open(LOG, "rb").read()
    head, body = split_log(b)
    if off_of(restamp(head, off_of(head) + 5)) != off_of(head) + 5:
        errs.append("N2 正例：restamp 没把偏移按新值改写")
    if split_log(restamp(head, len(head)) + body)[1] != body:
        errs.append("N2 反例：装配器切完再拼回去，正文区变了")
    try:
        apply_pairs(head, [("ZCTL 顶格杂行-不存在", "换了也没用")], "n2")
        errs.append("N2 反例没抓出：查无此锚的扰动在表头里静默通过")
    except AssertionError as e:
        if "命中 0 次" not in str(e):
            errs.append("N2 反例抓错了形状：%s" % str(e)[:60])
    return errs


def main():
    sys.stdout.reconfigure(line_buffering=True)
    assert sys.executable.startswith("/tmp/amvenv"), "本件解释器不是 /tmp/amvenv 而是 %s" % sys.executable
    assert "LC_ALL" not in os.environ, "本件环境里有 LC_ALL ⇒ 与仪器口径不符，那不是闸门红"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "本件没带空串 CVD"
    for p in ARCHIVE:
        assert os.path.isfile(p), "输入面不存在：" + p
    for k, v in snapshot().items():
        assert v, "输入面 %s 是空文件 ⇒ 本面的锚点无从普查" % os.path.basename(k)
    bad = precheck()
    if bad:
        print("纸面预检不合格（%d 条），一次子进程都不起：" % len(bad))
        for b in bad:
            print("  · " + b)
        sys.exit(3)
    print("纸面预检通过＝每条期望句在仪器源码里恰好 1 处且属于某条 assert／raise（且与语句首行相隔 ≤3）＋"
          "每个装配锚与装配后锚在真文件里恰好 1 处＋扰动后的副本与生成器副本都能 compile＋子运行名唯一＋"
          "判红谓词五形状双向＋装配器 restamp／切分／空锚三形状双向"
          "（红路 %d 条＋绿路 %d 条，本面全程不联网、零配额）" % (len(CASES), len(GREENS)))
    print("只读前提现取＝真档案四张面 md5 前 8：%s｜上一片标本存在＝True（%s）｜仪器副本跑在临时子目录时"
          "仓库根由向上找 .git 得到＝%s"
          % (" ".join("%s=%s" % (os.path.basename(FACES[k]), md5(FACES[k])[:8]) for k in KEYS),
             os.path.basename(SPEC_PREV), os.path.basename(REPO)))

    base_snap = snapshot()
    rows = []
    allok, envfail, assemble_fail = True, 0, []

    r = subprocess.run([sys.executable, INSTR], capture_output=True, text=True, timeout=TIMEOUT)
    assert r.returncode == 0 and GREEN_MARK in r.stdout, (
        "未扰动真档案就没走全（rc=%s）⇒ 本件的正对照全部无效。尾巴：\n%s"
        % (r.returncode, r.stderr[-400:]))
    rows.append(("G1 真仪器裸跑真档案（基线／所有红的对照面）", 0, True, "保持绿"))
    print("  " + rows[-1][0] + "｜rc=0｜保持绿")

    def one(name, spec, post, expect, is_green, tag):
        dest = PREFIX + tag
        d, argd = None, None
        try:
            ip, argd, d, src_txt = build(dest, **spec)
            run_extra(d, argd, post)
            cmd = [sys.executable, ip]
            for k in KEYS:
                cmd += ["--" + k, argd[k]]
            try:
                rr = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT, cwd=d)
            except subprocess.TimeoutExpired:
                return "超时", None, False
            if is_green:
                okg = rr.returncode == 0 and GREEN_MARK in rr.stdout
                return ("绿判读", rr.returncode,
                        (okg, "" if okg else "实得抛出行：" + raised(rr.stderr)[:120]))
            ok, why = judge(rr.returncode, rr.stderr, expect, src_txt)
            if not ok:
                why += "｜实得尾：" + raised(rr.stderr)[:110]
            return ("判读", rr.returncode, (ok, why))
        except (AssertionError, OSError, UnicodeDecodeError, re.error, SyntaxError) as e:
            return ("装配失败", None, "%s：%s" % (type(e).__name__, str(e)[:140]))
        finally:
            if d and os.path.isdir(d):
                shutil.rmtree(d)

    def drift():
        return [k for k, v in base_snap.items() if not os.path.exists(k) or md5(k) != v]

    for i, (name, spec, post) in enumerate(GREENS):
        kind, rc, g = one(name, spec, post, None, True, "g%d" % (i + 2))
        if kind == "超时":
            envfail += 1
            TIMEOUTS.append(name)
            rows.append((name, rc, False, "超时"))
            print("！%s｜超时（环境故障，不判红也不判过）" % name)
            continue
        if kind == "装配失败":
            allok = False
            assemble_fail.append(name)
            rows.append((name, None, False, g))
            print("× %s｜**装配失败**（锚点问题，不是红路）：%s" % (name, g))
            continue
        ok, why = g
        allok = allok and ok
        rows.append((name, rc, ok, "保持绿" if ok else "**绿路被自己判红**｜" + why))
        print(("  " if ok else "× ") + name + "｜rc=" + str(rc)
              + ("｜保持绿" if ok else "｜**绿路被自己判红**｜" + why))
        moved = drift()
        assert not moved, "绿路段子运行改动了档案面：" + "、".join(os.path.basename(k) for k in moved)

    for idx, (name, spec, post, expect) in enumerate(CASES):
        kind, rc, res = one(name, spec, post, expect, False, "c%d" % (idx + 1))
        if kind == "超时":
            envfail += 1
            TIMEOUTS.append(name)
            rows.append((name, rc, False, "超时"))
            print("！%s｜超时（环境故障，不判红也不判过）" % name)
            if envfail >= ENV_FAIL_STOP:
                print("CONTROL_VERDICT = ENV FAILURE（累计 %d 条超时，本件没资格判闸门）" % envfail)
                sys.exit(4)
            continue
        if kind == "装配失败":
            allok = False
            assemble_fail.append(name)
            rows.append((name, None, False, res))
            print("× %s｜**装配失败**（锚点问题，不是红路）：%s" % (name, res))
            continue
        ok, why = res
        allok = allok and ok
        rows.append((name, rc, ok, why))
        print(("  " if ok else "× ") + name + "｜rc=" + str(rc) + "｜期望「" + expect + "」"
               + ("命中" if ok else "**未命中**") + "｜" + why)
        moved = drift()
        if moved:
            print("CONTROL_VERDICT = ENV FAILURE（子运行改动了档案面：%s）"
                  % "、".join(os.path.basename(k) for k in moved))
            sys.exit(4)

    leak = [f for f in os.listdir(HERE) if f.startswith(PREFIX)]
    assert not leak, "跑完后 HERE 里仍有临时副本残留：" + "、".join(leak)
    fin = drift()
    assert not fin, "本面跑完之后档案面与起点快照不一致：" + "、".join(os.path.basename(k) for k in fin)

    # N4 双向：上面那条 assert 走的是"没变⇒不报警"那侧；这侧再证"变了⇒抓得出"。
    _probe = os.path.join(HERE, PREFIX + "n4probe")
    try:
        shutil.copyfile(SPEC_PREV, _probe)
        open(_probe, "ab").write(b"x")
        assert os.path.getsize(_probe) == os.path.getsize(SPEC_PREV) + 1, "N4 探针没写进字节"
        os.remove(_probe)
        assert not os.path.exists(_probe), "N4 探针没删掉"
    finally:
        if os.path.exists(_probe):
            os.remove(_probe)
    assert _probe not in base_snap, "N4：探针文件竟然在起点快照里 ⇒ 「档案面清单」这个集合口径变了"
    print("N4 双向＝起点快照在一致档案上零报警（上面那条 assert 已证）＋探针文件在快照外＝True"
          "（写一个字节再删掉；若快照对它报警就说明口径变了）｜本面自查＝%d 条（%s）"
          % (len(SELFCHK), "＋".join(SELFCHK)))
    green_rows = len([x for x in rows if x[0].startswith("G")])
    print("红路声明＝%d 条｜保持绿＝%d 条（逐条 rc=0 且打印出判读行；含裸跑基线 1 条）｜实跑行数＝%d 条｜"
          "超时（环境故障，不判红也不判过）＝%d 条%s｜装配失败（锚点问题，不是红路）＝%d 条%s｜"
          "造不出红路的闸门＝%d 条"
          % (len(CASES), green_rows, len(rows), len(TIMEOUTS),
             ("（" + "、".join(TIMEOUTS) + "）") if TIMEOUTS else "（无）",
             len(assemble_fail), ("（" + "、".join(assemble_fail) + "）") if assemble_fail else "（无）",
             len(UNCOVERED)))
    print("每条子运行＝一次本地子进程（临时目录里的副本），零 API、零配额；判据＝rc≠0 且**抛出的那一行**"
          "含指定闸门句 且 traceback 里指向该副本的行号落在该句在**副本源码**里的 ±3 行窗口；"
          "读数逐条即时落盘。绿路 %d 条没有重试后门：它们只要没保持绿就当条报出。" % green_rows)
    print("本轮造不出红路的闸门＝%d 条（逐条附原因，不算被验）：" % len(UNCOVERED))
    for gate, why in UNCOVERED:
        print("  · " + gate + "：" + why)
    assert len(rows) == len(CASES) + len(GREENS) + 1, "实跑 %d 行 ≠ 声明 %d＋%d＋1（基线也在表里）" % (
        len(rows), len(CASES), len(GREENS))
    if TIMEOUTS:
        print("CONTROL_VERDICT = INCOMPLETE（%d 条子运行超时＝环境故障，既没判红也没判过）" % len(TIMEOUTS))
        sys.exit(4)
    print("CONTROL_VERDICT =", "ALL CONTROLS CAN GO RED" if allok else "PROBLEM（有闸门空转）")
    sys.exit(0 if allok else 3)


if __name__ == "__main__":
    main()
