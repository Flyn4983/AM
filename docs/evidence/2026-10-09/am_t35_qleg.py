#!/usr/bin/env python3
"""#35 档案腿的 (q) 追加：把"自己写坏了自己的正文"登记进三份档案，计数当场数出来。

纪律同前：锚点取自文件末行且 `count==1`；正文里的计数由脚本现数，且**必须 >0** 才允许登记"修了三处"
（否则这条修复本身是空转）；正文一律用**普通三引号字符串**，不用 f-string —— 本轮登记的缺陷正是
f-string 把字面量 `{32}` 当替换字段吃掉了。
"""
import os
import subprocess
import sys

REPO = "/home/shy/桌面/2026年开发-显式冲击有限元-物质点法-sph-fvm+am/AM-Qoder"
DEV = os.path.join(REPO, "docs/开发日志.md")
DAY = os.path.join(REPO, ".workbuddy/memory/2026-10-09.md")
WBK = os.path.join(REPO, ".workbuddy/memory/MEMORY.md")
GEN = "/tmp/am_t35_docs.py"
FIXER = "/tmp/am_t35_fixbrace.py"
BAD = "`[0-9a-f]32`"
GOOD = "`[0-9a-f]{32}`"
NOW = subprocess.run(["date", "+%H:%M"], capture_output=True, text=True).stdout.strip()
md5_gen = subprocess.run(["md5sum", GEN], capture_output=True, text=True).stdout.split()[0]
md5_fix = subprocess.run(["md5sum", FIXER], capture_output=True, text=True).stdout.split()[0]

files = {"开发日志": DEV, "daily": DAY, "workbuddy MEMORY": WBK}


def bad_outside_q(text):
    """BAD 检测器分段：**(q)** 段里那句是**刻意引文**（记录当初印错的样子），段外才是真损坏。"""
    i = text.find("**(q)")
    return (text if i < 0 else text[:i]).count(BAD)


txts = {k: open(v, encoding="utf-8").read() for k, v in files.items()}
cur_bad = {k: bad_outside_q(t) for k, t in txts.items()}
cur_good = {k: t.count(GOOD) for k, t in txts.items()}
gen_txt = open(GEN, encoding="utf-8").read()
g_bad, g_ok = gen_txt.count("`[0-9a-f]{32}`"), gen_txt.count("`[0-9a-f]{{32}}`")
print("现数：(q) 段之外仍损坏＝" + str(sum(cur_bad.values())) + " 处（须为 0）｜各处已修好标志 GOOD＝"
      + "、".join(k + "=" + str(v) for k, v in cur_good.items()) + "（各须 ≥1）｜生成器未转义＝"
      + str(g_bad) + "／已转义＝" + str(g_ok))
assert sum(cur_bad.values()) == 0, "(q) 段之外还有未修的损坏字面量 ⇒ 不许登记『已修复』"
assert all(v >= 1 for v in cur_good.values()), "修复标志不齐 ⇒ 正文里的处数要重取"
assert (g_bad, g_ok) == (0, 3), "生成器里还有未转义的正文字面量 ⇒ 复跑还会坏"

DEV_TXT = """
**(q) 档案腿自己把正文写坏了（落盘后被回读抓住 ⇒ 自纠 ＋1 ⇒ §26.27 由 19 条增至 20 条）**
- **现象**：本轮五腿正文由 `/tmp/am_t35_docs.py` 的 **f-string** 装配，而正文里要引用一条正则字面量
  `[0-9a-f]{32}`。f-string 把 `{32}` 当成**替换字段**求值（`32` 是合法表达式 ⇒ 不报错、静默渲染成 `32`），
  于是落盘的三处全成了 `[0-9a-f]32`＝**一个语法残缺的字面量进了档案正文**。实测面＝开发日志 1 ＋
  daily 1 ＋ workbuddy MEMORY 1 ＝ **3 处**；计划文档与总览两条腿本来没写这句 ⇒ 不在损坏面内（不是漏数）。
- **为什么闸门没拦住**：五道预检只看**锚点** `count==1`，装配后的正文没有再查一次"字面量有没有被
  求值器改动"；而 `SyntaxWarning` 只对**未转义反斜杠**响（本轮那两处 `(\\S+)` 就是这么暴露的），
  花括号求值**完全无声**。抓它的是**写盘后的回读 grep**（按"回读实测"惯例跑的那一遍）⇒ 回读不是仪式。
- **修复**＝`/tmp/am_t35_fixbrace.py`（md5 """ + md5_fix[:12] + """，先预检后 `--write`）：
  ① 生成器正文串里把 `[0-9a-f]{32}` 转义成 `{{32}}`（**3 处**；代码里 raw 串的那两处正则不受影响，
  它们不是 f-串），② 三份档案各把 BAD 换回 GOOD 并回读断言 BAD＝0 且 GOOD＝旧 GOOD＋旧 BAD。
  仪器自带正对照：修复前 BAD＝0 就直接**拒绝登记**，防止把"修了三处"写在一处都没坏的时候。
- **计数口径（不许两头便宜）**：这条**计 ＋1**，因为它**已落进档案正文**、要靠第二次写盘才干净；
  本轮那五条"写盘前被闸门拦下"的假红／假拒（G0/G1/G3/G4/G8）仍计 ＋0，分界就在这里。
- **检测器必须跟着分段**（同轮第二处方法学收获）：本节要**引用**当初印错的样子，于是"全文 BAD 计数＝0"
  这条判据被自己的引文打翻（第一次跑 `--write` 时开发日志 BAD＝1 就是这句话）。⇒ 判据改成
  **"(q) 段之外 BAD 恒 0，(q) 段之内 BAD＝1（刻意引文）"**，本脚本的 `verify()` 就按这个口径断言。
  这是记忆里"陈旧值检测器要分当前／史录／用户可见三段"那条的又一次命中：**登记一处坏文本的档案，
  自己会变成同一检测器的命中项**，不分段就会在下一次复跑时假报警（或更糟：被人为放宽成"整文件允许"）。
- **同族归并**：`echo "…"` 吃反引号（本轮第 6 次，见 (p) 段）与本次 f-string 吃 `{}` 是同一条坑的两个宿主
  ⇒ **凡是让正文穿过一个求值器（shell 引号／f-string／模板），都要先问"这段文本里哪个字符会被吃掉"**；
  正确做法＝正文走**普通字符串／三引号字面量**或**带引号的 heredoc**，要插值就只插值明确字段。
  本件（(q) 腿）自己就用三引号普通串写正文，所以文中出现的 `[0-9a-f]{32}` 与 `{32}` 都按原样落盘。
- **排期不变**：下一项仍是 **#18**（路径程序驱动热源，CSV 时间轴 opt-in）→ #27＋#33 → #10/A3；
  **#40 → #39**；#29／#30 等用户裁决。生成器当前 md5＝""" + md5_gen[:12] + """，落笔 """ + NOW + """（`date` 现取）。
"""

SHORT = """
**(q) 摘要（全文见开发日志 §26.27 (q)）**：档案腿用 f-string 承载正文 ⇒ 字面量 `[0-9a-f]{32}` 里的
`{32}` 被当替换字段**静默吃掉**，三处印成 `[0-9a-f]32`（开发日志／daily／workbuddy MEMORY 各 1），
写盘后的**回读 grep** 抓住 ⇒ `/tmp/am_t35_fixbrace.py` 修回正文并给生成器补上 `{{32}}` 转义（3 处）。
**自纠 (q) ＋1 ⇒ §26.27 由 19 条增至 20 条**（它落进了档案正文；与"写盘前被闸门拦下"的 5 条 ＋0 分界在此）。
教训＝正文不许穿过求值器：`echo` 会吃反引号（第 6 次），f-string 会吃花括号（第 1 次，本轮）。
"""

ANCHORS = {DEV: "  随后 #27＋#33（同批，只用外部 18-track 靶）→ #10/A3；**#40 → #39**；#29／#30 等用户裁决。",
           DAY: "  NIST CSV 在盘、`test_gradient_through_enthalpy_is_finite` 决定 CSV 只能是 opt-in）。",
           WBK: '  "支持 LSF/DED 正向模拟/数字孪生"；性能数字一律留待 GPU 窗口。'}
print("\n=== 预检：锚点 count（必须＝1）且＝文件末行；已含 (q) 者转为**只校验**（幂等，不重复追加）===")


def verify(path, tag=""):
    """把 BAD 检测器**分段**：(q) 段里那句是刻意引文，(q) 段之外必须恒 0。"""
    s = open(path, encoding="utf-8").read()
    i = s.find("**(q)")
    before, after = s[:i], s[i:]
    b_bad, a_bad = before.count(BAD), after.count(BAD)
    q_cnt = s.count("**(q)")
    print("  " + tag + os.path.basename(path) + "：**(q)** " + str(q_cnt) + " 次｜(q) 段之前 BAD＝"
          + str(b_bad) + "（须 0）｜(q) 段之内 BAD＝" + str(a_bad) + "（刻意引文，须 1）｜GOOD＝"
          + str(s.count(GOOD)))
    assert q_cnt == 1 and b_bad == 0 and a_bad == 1, path
    assert s.count(GOOD) >= 1, path
    return q_cnt == 1


ok = True
todo = []
sync = []
for path, anchor in ANCHORS.items():
    s = open(path, encoding="utf-8").read()
    c, at_end, has_q = s.count(anchor), s.rstrip().endswith(anchor), s.count("**(q)")
    print("  " + os.path.basename(path) + " 锚点 count=" + str(c) + " 末行＝" + str(at_end)
          + " 已含(q)＝" + str(has_q))
    if has_q == 1:
        verify(path, tag="〔已写过，校验〕")
        sync.append(path)
        continue
    ok = ok and c == 1 and at_end and has_q == 0
    todo.append(path)
if not ok:
    raise SystemExit("锚点/幂等预检未过 ⇒ 整批不写盘")
if not todo and not sync:
    raise SystemExit("没有任何待写／待同步的件 ⇒ 本脚本是空转，先查为什么")
print("  待追加＝" + str(len(todo)) + "，待同步（(q) 段已存在但正文已修订）＝" + str(len(sync)))
if "--write" not in sys.argv:
    print("\n（预检模式：未写盘。加 --write。）")
    raise SystemExit(0)
for path in todo:
    s = open(path, encoding="utf-8").read()
    open(path, "w", encoding="utf-8").write(s + (DEV_TXT if path == DEV else SHORT))
    print("  追加 " + os.path.basename(path))
    verify(path, tag="〔新写〕")
for path in sync:
    # (q) 段是文件末段 ⇒ 从标记处截断后按当前全文重贴（第一次写盘时那段还缺"检测器分段"一条）
    s = open(path, encoding="utf-8").read()
    i = s.find("**(q)")
    head = s[:i - 1]
    assert head.rstrip().endswith(ANCHORS[path]), path
    new = head + DEV_TXT
    assert new.count("**(q)") == 1 and bad_outside_q(new) == 0, path
    open(path, "w", encoding="utf-8").write(new)
    print("  同步 " + os.path.basename(path) + "：行数 " + str(len(s.splitlines())) + " -> "
          + str(len(open(path, encoding="utf-8").read().splitlines()))
          + "（补上「检测器必须跟着分段」一条）")
    verify(path, tag="〔同步后〕")
print("完成：BAD 只在 (q) 段的引文里出现，(q) 段之外恒 0 ⇒ 检测器分段后仍然可用。")
