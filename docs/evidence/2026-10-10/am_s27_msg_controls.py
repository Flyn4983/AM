#!/usr/bin/env python3
"""分片 27 提交信息仪器 `am_s27_msg.py` 的**干跑正对照**：每条探测器都要能红。

做法＝import 真仪器（模块级只有常量与函数，`build()` 不在 import 时执行 ⇒ 无副作用），
对**同一把**正则／闸门喂合成坏件 ⇒ 要求拒绝；再喂真档案 ⇒ 要求通过。只判"能不能红"，
不写盘、不改档案、不动 git 状态。

为什么必须有这一件：本片的探测器里有四条是**本轮新写**的（角色分桶、写手行数两出处互证、
cmp 三件套复跑、暂存 blob 的节号／子条连续性），一条"从没红过的闸门"与"坏闸门"读数相同 ⇒
必须逐条造红。三条坑沿用档案里的教训：
① 判据不能只看"抛异常"，要**指定闸门句**（否则红的是别的原因，等于没验这条）；
② 扰动不许"只翻转极性"——翻一下断言方向不算扰动，必须**造坏输入**（§26.29 (l)）；
③ 合成副本一律落**本件自己的目录**（逃到 /tmp 会让 `__file__` 反推的 REPO 变成 `/` ⇒ 环境错假红），
   跑完逐条删除，末尾再普查一次"没有 zctl_ 残留"。
"""
import importlib.util
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MSG_PY = os.path.join(HERE, "am_s27_msg.py")
spec = importlib.util.spec_from_file_location("am_s27_msg", MSG_PY)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

bad, notes = [], []


def must_reject(name, fn, expect=None):
    try:
        r = fn()
    except Exception as e:  # noqa: BLE001 — 对照只关心"红不红、红在哪条闸门"
        msg = "%s: %s" % (type(e).__name__, str(e))
        if expect and expect not in msg:
            bad.append(name)
            notes.append("× %s 红了但**没命中指定闸门句**（期望「%s」，实得「%s」）" % (
                name, expect, msg[:110]))
            return
        notes.append("  %s 坏件 ⇒ 拒绝（命中「%s」）：%s" % (name, expect or "任意异常", msg[:110]))
        return
    bad.append(name)
    notes.append("× %s 坏件 ⇒ **探测器没红**（返回 %s）⇒ 该判据是空转" % (name, str(r)[:70]))


def must_accept(name, fn):
    try:
        r = fn()
    except Exception as e:
        bad.append(name)
        notes.append("× %s 真件 ⇒ **被拒**（%s: %s）⇒ 判据过严或口径变了" % (
            name, type(e).__name__, str(e)[:110]))
        return None
    notes.append("  %s 真件 ⇒ 通过：%s" % (name, str(r)[:110]))
    return r


def staged_added_modified():
    rows = [l.split("\t") for l in M.sh("git", "-c", "core.quotePath=false", "diff", "--cached",
                                         "--name-status").splitlines() if l.strip()]
    return [r[-1] for r in rows if r[0] == "A"], [r[-1] for r in rows if r[0] == "M"]


curl_txt = M.rd(M.CURL)
reg_txt = M.rd(M.REG)
wrep_txt = M.rd(M.WRITER_RAW)
added, modified = staged_added_modified()
curl = must_accept("A0-回执闸门", lambda: M.gate_curl(curl_txt))
reg = must_accept("A1-回归件解析", lambda: M.parse_reg(reg_txt))
must_accept("A2-角色分桶（真暂存集）", lambda: M.bucketize(added))
wrows = must_accept("A3-写手两出处互证", lambda: M.writer_check(modified)[0])
must_accept("A4-cmp 三件套＋表头字段反查", lambda: M.cmp_check(curl, M.fields_of(M.HDR26)))
secn_letters = must_accept("A5-暂存日志节号", lambda: M.devlog_check())
secn, letters = (secn_letters[0], secn_letters[1]) if secn_letters else ("", [])


def a6():
    # 本件首版按**复数**形状取数，git 在 1 条时写单数 ⇒ 第三分组静默不参与。这条基线只验一件事：
    # 现在的形状能吃到单数（三个分组都取到值、且不是 None）。
    m = M.RE_SHORTSTAT.search("1 file changed, 1 insertion(+), 1 deletion(-)")
    assert m.groups() == ("1", "1", "1"), "单数形状取数＝%s ⇒ 有分组没吃到" % (m.groups(),)
    return m.groups()


must_accept("A6-shortstat 单数形状取数", a6)
real_ss = must_accept("A7-shortstat 与 numstat 闭合（真暂存集）", lambda: M.shortstat_check())
dev_txt = M.sh("git", "show", ":" + M.DEVLOG)
if bad:
    print("分片 27 信息仪器的干跑正对照｜**基线就没过** ⇒ 后面所有红都不作数，先修仪器")
    for n in notes:
        print(n)
    print("CONTROL_VERDICT = PROBLEM（基线 %s）" % bad)
    sys.exit(3)

# C1–C7：回执闸门逐条造红（每条只动一个读数，期望句＝闸门自己的措辞）
must_reject("C1-闭合式右边 +1", lambda: M.gate_curl(curl_txt.replace(
    "闭合式 远端 668 == 父提交 648", "闭合式 远端 669 == 父提交 648")), "闭合式在信息生成器里不成立")
must_reject("C2-路径差集非零", lambda: M.gate_curl(curl_txt.replace(
    "双向路径差集 0/668", "双向路径差集 1/668")), "非零差集")
must_reject("C3-单列复核缺一", lambda: M.gate_curl(curl_txt.replace(
    "本片单列复核 25/25 MATCH（A 20／M 5 个文件）", "本片单列复核 24/25 MATCH（A 20／M 5 个文件）")),
    "单列复核不是全绿")
must_reject("C4-判读行降级", lambda: M.gate_curl(curl_txt.replace(
    "LANDING_VERDICT = ALL CHECKS PASS", "LANDING_VERDICT = PROBLEM（见上面非零项）")), "不是 ALL CHECKS PASS")
must_reject("C5-truncated 翻真", lambda: M.gate_curl(curl_txt.replace(
    "② truncated=False", "② truncated=True")), "truncated=")
must_reject("C6-tree 条目少一条", lambda: M.gate_curl(curl_txt.replace(
    "tree 条目 736 = blob 668 + tree 68", "tree 条目 735 = blob 668 + tree 68")), "tree 条目 ≠ blob＋tree")
# C7 只动**正文**的 blob 数、不动表头闭合式 ⇒ 两出处分裂必须红（证明"互证"不是装饰）
must_reject("C7-正文与表头 blob 数分裂", lambda: M.gate_curl(curl_txt.replace(
    "② truncated=False，tree 条目 736 = blob 668 + tree 68",
    "② truncated=False，tree 条目 736 = blob 667 + tree 68")), "正文的 blob 数与表头闭合式左值不等")

# C8–C9：角色分桶的两条失败路径（落进 0 个桶／落进 2 个桶）
must_reject("C8-陌生件落进 0 个桶", lambda: M.bucketize(added + ["zctl_unexpected.bin"]), "落进 0 个桶")
_orig_buckets = list(M.BUCKETS)
M.BUCKETS = _orig_buckets + [("重复桶", r"^am_s27_docs\.py$")]
must_reject("C9-一件落进两个桶", lambda: M.bucketize(["am_s27_docs.py"]), "落进 2 个桶")
M.BUCKETS = _orig_buckets
assert len(M.bucketize(["am_s27_docs.py"])["归档写手"]) == 1, "还原 BUCKETS 后正常件反而被拒"


# C10–C13：写手互证的四条坏形状，用**绝对路径**的合成副本喂同一把闸门（不改档案）
def with_writer_copy(name, mutate, expect):
    p = os.path.join(HERE, "zctl_s27_msg_wraw.txt")
    open(p, "w", encoding="utf-8").write(mutate(wrep_txt))
    old = M.WRITER_RAW
    M.WRITER_RAW = p
    try:
        must_reject(name, lambda: M.writer_check([r["rel"] for r in wrows]), expect)
    finally:
        M.WRITER_RAW = old
        os.unlink(p)


with_writer_copy("C10-增量与箭头两侧不等",
                 lambda t: t.replace("4266 → 4287 行（＋21）", "4266 → 4287 行（＋22）"), "写手声称＋")
with_writer_copy("C11-新行数被改成不是暂存 blob",
                 lambda t: t.replace("4266 → 4287 行（＋21）", "4266 → 4288 行（＋22）"), "≠ 暂存 blob 实得")
with_writer_copy("C12-增量夸大到超过 HEAD 净增",
                 lambda t: t.replace("4266 → 4287 行（＋21）", "4100 → 4287 行（＋187）"), "对 HEAD 净增")
with_writer_copy("C13-少报一张面",
                 lambda t: "\n".join(l for l in t.splitlines() if not l.startswith("WRITE docs/项目开发总览")),
                 "五张面")

# C14–C15：指纹四路闭合的两条断腿（回归件 SRCFINGER 缺一路／起≠止）
must_reject("C14-SRCFINGER 缺 END", lambda: M.parse_reg(
    re.sub(r"^SRCFINGER_END: \w+$", "", reg_txt, flags=re.M)), "SRCFINGER 只取到")
must_reject("C15-SRCFINGER 起≠止", lambda: M.parse_reg(reg_txt.replace(
    "SRCFINGER_END: 2a49fedf28e08e8d5a68ae23124e391d",
    "SRCFINGER_END: 2a49fedf28e08e8d5a68ae23124e391e")), "起≠止")

# C16：FIELDS 反查——没有 FIELDS 赋值的合成"表头生成器"必须拒绝（分母不许是手打的 16）
_fp = os.path.join(HERE, "zctl_s27_msg_nofields.py")
open(_fp, "w", encoding="utf-8").write("X = 1\n")
try:
    must_reject("C16-FIELDS 取不到", lambda: M.fields_of(_fp), "取不到 FIELDS 赋值")
finally:
    os.unlink(_fp)

# C17：cmp 三件套的偏移 +2 ⇒ 必须红（证明那三个 rc 不是恒 [0,1,1] 的常数）
_log = os.path.join(HERE, "zctl_s27_msg_curl.log")
shutil.copyfile(os.path.join(M.REPO, M.CURL), _log)
_old = M.CURL
M.CURL = _log
try:
    must_reject("C17-偏移 +2 的 cmp", lambda: M.cmp_check(dict(curl, boff=str(int(curl["boff"]) + 2)),
                                                          M.fields_of(M.HDR26)), "须 [0,1,1]")
finally:
    M.CURL = _old
    os.unlink(_log)


# C18–C20：暂存日志的三条形状（子条字母断号／入库句被删／钉住口径原文被删）
# 做法＝只给 `git show :<面>` 这一条命令换返回值，其余命令照原样跑 ⇒ 不改档案、不动索引
_real_sh = M.sh


def with_fake_log(mutate, fn):
    def _f(*args, **kw):
        if len(args) >= 3 and args[0] == "git" and args[1] == "show" and str(args[2]).startswith(":"):
            return mutate(dev_txt)
        return _real_sh(*args, **kw)
    M.sh = _f
    try:
        fn()
    finally:
        M.sh = _real_sh


must_reject("C18-子条字母断号", lambda: with_fake_log(
    lambda t: t.replace("**(i) ", "**(q) "), M.devlog_check), "不连续")
must_reject("C19-入库句被删", lambda: with_fake_log(
    lambda t: t.replace("am_s7_verify26.log", "am_s7_verifyXX.log"), M.devlog_check), "没有引用上一片回执名")
must_reject("C20-钉住口径原文被删", lambda: with_fake_log(
    lambda t: t.replace("当前「不动 GPU」", "当前「随便写」"), M.devlog_check), "取不到钉住口径原文")
assert M.devlog_check() == secn_letters, "还原真 sh 后节号读数变了 ⇒ 上面的替身漏了别的命令"


def with_fake_shortstat(text, fn):
    def _f(*args, **kw):
        if args[:1] == ("git",) and args[-1] == "--shortstat":
            return text + "\n"
        return _real_sh(*args, **kw)
    M.sh = _f
    try:
        fn()
    finally:
        M.sh = _real_sh


# C21–C23：shortstat 的三个数逐条改坏 1，numstat 仍现算 ⇒ 闭合闸门必须逐条红。
# 扰动都是"造坏输入"（改数字），不是翻转断言极性；三条各自命中自己的那句闸门，
# 证明它们不是同一条泛化断言的三种叫法。
ss_files, ss_ins, ss_dels = real_ss[1], real_ss[2], real_ss[3]
must_reject("C21-删除数虚高 1", lambda: with_fake_shortstat(
    "%d files changed, %d insertions(+), %d deletions(-)" % (ss_files, ss_ins, ss_dels + 1),
    M.shortstat_check), "删除数")
must_reject("C22-增加数虚高 1", lambda: with_fake_shortstat(
    "%d files changed, %d insertions(+), %d deletions(-)" % (ss_files, ss_ins + 1, ss_dels),
    M.shortstat_check), "增加数")
must_reject("C23-文件数虚高 1", lambda: with_fake_shortstat(
    "%d files changed, %d insertions(+), %d deletions(-)" % (ss_files + 1, ss_ins, ss_dels),
    M.shortstat_check), "文件数")
# 首版那条静默漏项的形状：删掉 deletions 子句 ⇒ 分组取到 None，`or 0` 折成 0。
# 本轮 numstat 现算删除数＝%d（非 0），所以这条有靶；若某轮真为 0，它会自动红（探测器没红＝空转）。
must_reject("C24-deletions 子句被摘掉", lambda: with_fake_shortstat(
    "%d files changed, %d insertions(+)" % (ss_files, ss_ins), M.shortstat_check), "删除数")
assert M.shortstat_check() == real_ss, "还原真 sh 后 shortstat 读数变了 ⇒ 替身漏了别的命令"
assert ss_dels > 0, "本轮 numstat 现算删除数＝0 ⇒ C21/C24 打不到靶，需要另造坏输入而不是留着假绿"

assert not [p for p in os.listdir(HERE) if p.startswith("zctl_s27_msg")], "合成副本没被清掉"
n_acc = len([n for n in notes if re.match(r"^\s+A\d", n)])
n_rej = len(notes) - n_acc
print("分片 27 信息仪器干跑正对照｜被验仪器＝am_s27_msg.py（md5 %s）｜跑在同一目录、用同一把探测器" % M.md5(MSG_PY, 12))
print("真件基线 %d 条全过｜坏件 %d 条｜判据＝坏件必拒绝**且命中指定闸门句**、真件必通过" % (n_acc, n_rej))
print("真暂存集＝新增 %d 件／修改 %d 件｜§26.%s 子条 %s（%d 条）" % (
    len(added), len(modified), secn, "-".join([letters[0], letters[-1]]), len(letters)))
for n in notes:
    print(n)
print("CONTROL_VERDICT =", "ALL DRY-RUN GATES CAN GO RED" if not bad else
      "PROBLEM（%d 条空转：%s）" % (len(bad), bad))
sys.exit(0 if not bad else 3)
