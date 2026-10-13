#!/usr/bin/env python3
"""写盘后**另跑的只读仪器**＝§26.32「跑前登记的闸门」第 ③ 条承诺的那一件：
「回执表头只由生成器产出（预检面与写盘面各留原始件，`BODY_OFFSET` 由不动点迭代，写盘后**另跑一件只读仪器**
做 `cmp -i 偏移:0` 三联＝0/1/1，顶格字段数与生成器 `FIELDS` 声明互证）」。

为什么必须另跑一件，而不是引用生成器自己那条写盘分支打印的读数：生成器的写盘分支用**同一段代码**
先求出 `BODY_OFFSET`、再拿它去 `cmp`，所以它打印的「0/1/1」只能证明"这个数在它自己的口径下自洽"，
不能证明"这个数与磁盘上那张档案真正对得上"。本件把生成器当成**被验对象**：偏移量从写好的回执里
**读**回来（`^BODY_OFFSET=…` 那一行），不是自己迭代出来的；字段表从生成器源码里用 **ast 现取**（不执行它）。
⇒ 两边同值才叫互证。

本件三条硬约束（每条都由下面的一条断言兜着，不是宣言）：
① **只读**：不写任何文件、不起任何写进程。自检办法＝对自己 `__file__` 的源码做 ast 普查（见到
   `open(..., 可写模式)`／`os.remove`／`.write(` 就拒绝运行）＋跑前跑后现比四张输入面的 md5。
   「只读普查」的审计对象可以由 `--auditsrc` 指到别的件——那是给控件面造红用的入口，本件默认只审自己。
② **不联网**：本件只读本地四张面（回执／原始 stdout／生成器源码／生成器写盘面原始 stdout）。
   表头控件面每条扰动都要复跑一次联网尺子（配额 60 次/小时），本件不需要，所以它可以排在任何时候跑。
③ **读数一律现取**：偏移量、正文区字节数、字段数、写盘面那四条数都从文件现解；
   本件里不出现任何"上一片是几个字"的写死数（那种数随片腐烂）；唯一保留的字面量是判读行本身，
   而它被**双向**查了一遍（原始件末行＋正文区末行都得等于它）。
④ `off` 是**字节**偏移（`cmp -i` 的口径），不许拿去切 `str`：中文一条三个字节，`txt[:off]` 会切进正文区
   ——本件第一版就栽在这里，红在"顶格却不像字段的行"那条闸门上（标本 `_raw_s30_hdr_after_red1.err`）。

判读行末行＝`HDR_AFTER_VERDICT`。任何一条不合格即抛 `AssertionError`，退出非 0。
"""
import argparse
import ast
import hashlib
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
DAILY = os.path.relpath(HERE, REPO)

# 四张输入面的文件名＝本片的四件路径常量（与生成器同一制式：片号是这一片的绑定改动，随片而变）。
# 本件**不靠自己的文件名**反推片号：片号一律从传进来的四张面各自的文件名现取，再要求四者相等
# （见 main 里的「四张输入面片号不一致」那条）。FACE_NAMES 必须是**纯字面量**，因为控件面要用 ast
# 现取它（literal_eval 吃不下带 os.path.join／条件式的赋值——本件第一版就是那样写的，控件面当场崩）。
FACE_NAMES = {
    "log": "am_s7_verify30.log",
    "raw": "_raw_verify30.out",
    "gen": "am_s30_hdr.py",
    "writeface": "_raw_s30_hdr_write.out",
}
DEFAULTS = {k: os.path.join(HERE, FACE_NAMES[k]) for k in ("log", "raw", "gen", "writeface")}
SLICE_PAT = {
    "log": r"^am_s7_verify(\d+)\.log$",
    "raw": r"^_raw_verify(\d+)\.out$",
    "gen": r"^am_s(\d+)_hdr\.py$",
    "writeface": r"^_raw_s(\d+)_hdr_write\.out$",
}

LAST_LINE = "LANDING_VERDICT = ALL CHECKS PASS"


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def self_audit(path):
    """只读普查：本件源码里不许出现写盘动作。返回命中的坏构造列表。"""
    tree = ast.parse(open(path, encoding="utf-8").read())
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        fname = node.func.attr
        if fname in ("remove", "unlink", "rename", "rmdir", "truncate", "mkdir", "makedirs"):
            bad.append("%s 行 %s：os.%s" % (os.path.basename(path), node.lineno, fname))
        # 只拦**面向字节流的写**：`bad.append(...)` 这类列表追加不是写盘，早先把它一起列进去，
        # 本件普查到自己、当场拒绝运行（＝一条把仪器判成坏仪器的口径错）。
        if fname in ("write", "writelines", "write_text"):
            bad.append("%s 行 %s：.%s" % (os.path.basename(path), node.lineno, fname))
        if fname == "open":
            mode = ""
            if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                mode = str(node.args[1].value)
            for kw in node.keywords:
                if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                    mode = str(kw.value.value)
            if any(c in mode for c in "wax+"):
                bad.append("%s 行 %s：open(mode=%r)" % (os.path.basename(path), node.lineno, mode))
    return bad


def fields_from_gen(path):
    """从生成器源码里用 ast 现取 FIELDS（不执行生成器）。"""
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "FIELDS" for t in node.targets):
            assert isinstance(node.value, (ast.List, ast.Tuple)), "FIELDS 不是列表／元组字面量 ⇒ 取不到声明"
            out = []
            for e in node.value.elts:
                assert isinstance(e, ast.Constant) and isinstance(e.value, str), \
                    "FIELDS 里有非字符串元素 ⇒ 声明口径变了"
                out.append(e.value)
            return out
    raise AssertionError("生成器源码里没有 FIELDS 赋值 ⇒ 字段表不是声明出来的")


def once(pattern, text, what, flags=0):
    m = re.findall(pattern, text, flags)
    assert len(m) == 1, "%s 在文件里命中 %d 次（须恰好 1）⇒ 出处不唯一，拒绝判读" % (what, len(m))
    return m[0]


def cmp_rc(a, b, left_off, right_off):
    return subprocess.run(["cmp", "-i", "%d:%d" % (left_off, right_off), a, b],
                          capture_output=True, text=True).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=DEFAULTS["log"])
    ap.add_argument("--raw", default=DEFAULTS["raw"])
    ap.add_argument("--gen", default=DEFAULTS["gen"])
    ap.add_argument("--writeface", default=DEFAULTS["writeface"])
    ap.add_argument("--auditsrc", default=os.path.abspath(__file__))
    a = ap.parse_args()

    ins = [a.log, a.raw, a.gen, a.writeface]
    for k, p in (("log", a.log), ("raw", a.raw), ("gen", a.gen), ("writeface", a.writeface)):
        assert p, "%s 面没给默认值也没给参数 ⇒ 本件在随机文件名下被裸跑" % k
        assert os.path.isfile(p), "输入面不存在：" + p
    # 四张面必须来自**同一分片**（片号从各自文件名现取）：把上一片的原始件配到本片回执上，
    # cmp 照样能红成"正文一致"，而那根本不是这片的故事。
    nums = {}
    for k, p in zip(("log", "raw", "gen", "writeface"), ins):
        m = re.fullmatch(SLICE_PAT[k], os.path.basename(p))
        assert m, "%s 面文件名 %s 不符合形制 %s ⇒ 取不到片号" % (k, os.path.basename(p), SLICE_PAT[k])
        nums[k] = m.group(1)
    assert len(set(nums.values())) == 1, "四张输入面片号不一致（%s）⇒ 张冠李戴，拒绝判读" % nums
    print("四张输入面片号现取自文件名并互证＝", nums["log"], "｜", "、".join(os.path.basename(p) for p in ins))
    md_before = {p: md5(p) for p in ins}

    # ① 只读自检（审计对象是本件自己，由 --auditsrc 暴露给控件面）
    bad = self_audit(a.auditsrc)
    assert not bad, "本件不是只读件，普查到写盘动作：" + "；".join(bad)

    lb = open(a.log, "rb").read()
    rb = open(a.raw, "rb").read()

    # ② 偏移量**从写好的回执读回来**（这是与生成器的本质区别：它自己迭代，本件只认盘上那个数）
    txt = lb.decode()
    lines = txt.splitlines(True)
    offs = [ln for ln in lines if re.match(r"^BODY_OFFSET=\d+$", ln)]
    assert len(offs) == 1, "回执里 BODY_OFFSET 行不是恰好 1 条（实得 %d 条）⇒ 偏移量没有唯一出处" % len(offs)
    off = int(re.search(r"BODY_OFFSET=(\d+)", offs[0]).group(1))
    assert lb[off - 1:off] == b"\n", "BODY_OFFSET 不落在行首 ⇒ 偏移量指到行中间，正文区不可能逐字节对齐"
    print("BODY_OFFSET（从盘上现读）＝", off, "｜该行原文＝", offs[0].strip())

    # ③ 算术闭合：回执总长 − 偏移＝原始件字节数（与 cmp 是两条独立的查法）
    assert len(lb) - off == len(rb), "正文区长度 %d ≠ 原始件字节数 %d ⇒ 两把尺子对不上" % (
        len(lb) - off, len(rb))
    print("回执总长＝", len(lb), "｜回执总长 − 偏移＝", len(lb) - off, "｜原始件字节数＝", len(rb), "→ 闭合")

    # ④ cmp 三联：同偏移须 0，任一侧各 +1 须非 0（偏移不对称就拦不住整体平移）
    c1 = cmp_rc(a.log, a.raw, off, 0)
    c2 = cmp_rc(a.log, a.raw, off + 1, 0)
    c3 = cmp_rc(a.log, a.raw, off, 1)
    assert (c1, c2 != 0, c3 != 0) == (0, True, True), "cmp 三联不是 0／非0／非0（实得 %s／%s／%s）" % (
        c1, c2, c3)
    print("cmp -i 三联（本件自己起进程跑的，不是抄生成器的）＝", c1, "/", c2, "/", c3)

    # ⑤ 顶格字段数与生成器 FIELDS 声明互证（ast 现取，不执行生成器）
    # ⚠ `off` 是**字节**偏移（`cmp -i` 的口径），不是字符下标：表头里满是一条三个字节的中文字，
    # 用 `txt[:off]` 切会一把切进正文区。本件第一版就是这么写的，实测把正文的「① 本地 HEAD = …」
    # 当成"顶格却不像字段的行"判红（标本 `_raw_s30_hdr_after_red1.err`，抛出行末列那两行）⇒
    # 一律先按**字节**切、再解码，并加一条 `len(head.encode()) == off` 把这个口径钉住。
    dec = fields_from_gen(a.gen)
    head = lb[:off].decode()
    assert len(head.encode()) == off, "表头按字节切完再编码回去不等于 off ⇒ 上面那句字节／字符口径又松了"
    lab = "|".join(map(re.escape, dec))
    defs = [l for l in head.splitlines() if l.strip() and re.match(r"^(%s)\s*:" % lab, l)]
    names = [re.match(r"^(%s)" % lab, l).group(1) for l in defs]
    assert len(defs) == len(dec), "回执顶格字段 %d 行 ≠ 生成器 FIELDS 声明 %d 行 ⇒ 有字段漏写或改名" % (
        len(defs), len(dec))
    seq_ok = names == dec
    assert seq_ok, "回执字段顺序与 FIELDS 声明不同序：" + str(list(zip(names, dec)))
    mis = [l for l in head.splitlines() if l[:1].isspace() and re.match(r"^\s+(?:%s)\s*:" % lab, l)]
    assert not mis, "回执表头缩进处出现「标签＋冒号」（粘连漏了续行缩进）：" + str(mis[:2])
    loose = [l for l in head.splitlines() if l.strip() and not l[0].isspace()
             and not re.match(r"^(%s)\s*:" % lab, l) and not re.match(r"^BODY_OFFSET=\d+$", l)]
    assert not loose, "回执表头里有顶格却不像字段的行：" + str(loose[:2])
    print("顶格字段行数＝", len(defs), "/", len(dec), "（声明取自", os.path.relpath(a.gen, REPO),
          "，ast 现取不执行）｜同序＝", seq_ok)

    # ⑥ 残留检查：半角占位符与全角 at 号（后者会静默不替换）
    assert "@@" not in head, "盘上表头残留半角占位符 ⇒ 有字段没取到值"
    assert "＠" not in head, "盘上表头含全角 at 号 ⇒ 那种占位符不会被替换"

    # ⑦ 正文区没被截断：末行必须是判读行
    body = lb[off:].decode()
    assert rb.rstrip(b"\n").splitlines()[-1].decode() == LAST_LINE, \
        "原始件末行与判读行常量不符 ⇒ 本件里那句判读行是写死的且已腐烂"
    assert body.rstrip("\n").splitlines()[-1] == LAST_LINE, "正文区末行不是判读行 ⇒ 正文被截断或判读行变了"

    # ⑧ 两出处闭合：本件（盘上现解）与生成器写盘面（当时打印）必须同值
    wf = open(a.writeface, encoding="utf-8").read()
    wbytes = int(once(r"写盘完成： \S+ (\d+) bytes", wf, "写盘面总字节数"))
    m = once(r"cmp 复核 rc（同偏移须 0）＝ (\d+) ｜左 \+1 须非 0＝ (\d+) ｜右 \+1 须非 0＝ (\d+)"
             r" ｜顶格字段行数＝ (\d+) / (\d+)", wf, "写盘面那四联读数")
    w_c1, w_c2, w_c3, w_defs, w_dec = m
    assert wbytes == len(lb), "写盘面报出 %d 字节 ≠ 盘上现有 %d 字节 ⇒ 写过之后这张家档案被人动过" % (
        wbytes, len(lb))
    assert (int(w_c1), int(w_c2), int(w_c3)) == (c1, c2, c3), "写盘面 cmp 三联 %s/%s/%s ≠ 本件实得 %s/%s/%s" % (
        w_c1, w_c2, w_c3, c1, c2, c3)
    assert (int(w_defs), int(w_dec)) == (len(defs), len(dec)), "写盘面字段数 %s/%s ≠ 本件实得 %d/%d" % (
        w_defs, w_dec, len(defs), len(dec))
    print("两出处闭合＝写盘面(%s／%s／%s，字段 %s/%s，%d bytes) 与本件(盘上现解) 同值" % (
        w_c1, w_c2, w_c3, w_defs, w_dec, wbytes))

    # ⑨ 只读的最终证据：跑前跑后 md5 逐一相同
    drift = [p for p in ins if md5(p) != md_before[p]]
    assert not drift, "本件运行期间输入面被改写：" + "、".join(os.path.basename(p) for p in drift)
    print("输入面 md5 跑前后逐一相同＝True（%d 张面）" % len(ins), "｜只读普查对象＝",
          os.path.relpath(a.auditsrc, REPO) if a.auditsrc.startswith(REPO) else a.auditsrc,
          "命中坏构造＝", len(bad))
    print("HDR_AFTER_VERDICT = ALL READ-ONLY CHECKS PASS")


if __name__ == "__main__":
    sys.exit(main())
