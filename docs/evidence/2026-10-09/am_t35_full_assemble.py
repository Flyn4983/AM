#!/usr/bin/env python3
"""#35 全量回归面的**档案装配器**：把原样 stdout 与若干实测件拼成
`docs/evidence/2026-10-09/am_t35_full_regression.log`，五道闸门**由代码算**（不靠人念数）。

为什么要这个件：口径与前两轮全量件（`2026-10-08/am_t34_t36_full_regression.log`）对齐，否则本件的
"记分"与历史件不可比。与那一轮的差异照实登记：本轮用 **`-q`（无 `-rA`）** ⇒ ③路计数取 `short test
summary info` 的 FAILED 行、skip 身份取**跑中同树**单独一次的 `-rs` 定向件。

闸门（跑前登记，每条都要能红）：
  G1 计数闭合：汇总行结局＝failed＋passed＋skipped；进度字符（FAILURES 段之前那段点阵）＝collected；
     结局 − collected ＝ 2（＝2 条模块级 importorskip，收集期即 skip ⇒ 不占进度字符）。
  G2 两条登记红灯**逐字节**不变（参照件取 `^E\\s+AssertionError: `）；正对照＝末位改一档必须变红。
  G3 红灯**身份**（测试名）从两件的 FAILURES 段标题现取，并与本件 FAILED 短汇总行交叉核对。
  G4 skip 身份取自 `-rs` 定向件的 SKIPPED 行原文（不按位置反推），宿主必须与参照件一致、无新增。
     红线：全量回归不得新增 skip/xfail。
  G5 同树：`SRCFINGER_START == SRCFINGER_END`。
  G6 rc 为**实测**（本件启动器写了 `rc=`；上一轮没写，那件的 rc 是推断）。
  G7 「纯文本改动」**由 AST 证明**（不再手数行号）：HEAD 版与工作版各自解析、把所有字符串常量置空后
     `ast.dump` 必须相等 ⇒ 注释＋字符串常量之外一字未变。正对照＝追加一条真语句必须变红。
  G8 CPU 钉住的**正面证据在正文里**：正文必须命中 JAX 的 `cuInit(0) failed`＋`Falling back to cpu`
     行（只写在头里的 ENV 声明不算证据）。
扰动正对照（G1/G2/G4/G7 各一条，必须变红）：collected 少算 1；红灯行末位改一档；一条 skip 换新宿主；
往工作版追加一条真语句。
"""
import ast
import os
import re
import subprocess
import sys
import time

REPO = "/home/shy/桌面/2026年开发-显式冲击有限元-物质点法-sph-fvm+am/AM-Qoder"
RAW, COLL, SKIP = "/tmp/am_t35_full.log", "/tmp/am_t35_collect.txt", "/tmp/am_t35_skips.txt"
REF = os.path.join(REPO, "docs/evidence/2026-10-08/am_t34_t36_full_regression.log")
OUT = os.path.join(REPO, "docs/evidence/2026-10-09/am_t35_full_regression.log")
raw = open(RAW, encoding="utf-8").read()
coll_txt = open(COLL, encoding="utf-8").read()
skip_txt = open(SKIP, encoding="utf-8").read()
ref = open(REF, encoding="utf-8").read()
bad = []


def need(cond, msg):
    if not cond:
        bad.append(msg)
    return bool(cond)


def names_from_failures(t):
    return re.findall(r"^_{4,}\s+(\w+)\s+_{4,}$", t, re.M)


# ---- G0： face 是否**真的跑完**（没有这一道，下面的正则会在半成品日志上部分命中 ⇒ 造出半张档案）----
# 判据缺一即**具名拒绝**（rc=2），而不是让 AttributeError 替我报错（本轮预检就在不完整日志上崩过，
# 崩在 tally 那一行——报错点离真正原因"face 还在跑"很远，所以把原因写在门口）。
t_start = re.search(r"^SRCFINGER_START: ([0-9a-f]{32})", raw, re.M)
t_end = re.search(r"^SRCFINGER_END: ([0-9a-f]{32})", raw, re.M)
t_rc = re.search(r"^rc=(\d+)", raw, re.M)
FINGER_CMD = ("find src tests -name '*.py' -not -path '*__pycache__*' -print0 "
              "| sort -z | xargs -0 md5sum | md5sum | cut -d' ' -f1")
finger_now = subprocess.run(["bash", "-c", FINGER_CMD], cwd=REPO, capture_output=True, text=True).stdout.strip()
# 日志是否还有人在追加（大小两次采样相同）——只对自己这份日志判，不判"机器上有别人在跑 pytest"
s1 = os.path.getsize(RAW)
time.sleep(3)
s2 = os.path.getsize(RAW)
ps = subprocess.run(["ps", "-eo", "pid=,comm=,args="], capture_output=True, text=True).stdout
# 只认**我自己的启动器脚本**（文件名唯一）；`pgrep -f` 会匹配到 harness 的 `bash -c eval` 包装（分片 17 踩过）。
mine = [l for l in ps.splitlines() if "am_t35_full_run.sh" in l and " -c shopt" not in l]
# 观察项（**不作为闸门**）：本机其它 pytest 进程。2026-10-09 11:00:39 首版把这一条写成了闸门，
# 结果被**另一个 CLI（父进程 codebuddy）的 pytest** 拦住 ⇒ 判据错了：我要证的是"我这份日志已封口、
# 这棵树在归档时仍与跑完时同指纹"，而不是"世界上没有别人在跑测试"。
foreign = [l for l in ps.splitlines() if re.search(r"^\s*\d+\s+python\S*\s", l) and " -m pytest" in l]
G0 = [
    ("G0 日志里没有 DATE_END/rc= ⇒ face 未结束", bool(t_rc)),
    ("G0 没有 SRCFINGER_START ⇒ 启动器戳缺失", bool(t_start)),
    ("G0 没有 SRCFINGER_END ⇒ 回归可能仍在跑", bool(t_end)),
    ("G0 起/止指纹不同 ⇒ 回归期间动过被测树，该轮记分不可用",
     # 只在两枚戳**都在**时才判这条，否则会把"还没跑完"误诊成"动过树"
     not (t_start and t_end) or t_start.group(1) == t_end.group(1)),
    ("G0 我的启动器进程仍在跑（" + str([l.split()[0] for l in mine]) + "）⇒ 日志未封口", not mine),
    (f"G0 日志 3 秒内大小变化 {s1}->{s2} ⇒ 仍在被追加", s1 == s2),
    (f"G0 归档时现取指纹 {finger_now} ≠ 跑完时的 SRCFINGER_END {t_end.group(1) if t_end else '?'} "
     "⇒ 跑完之后、归档之前树被改过", bool(t_end) and finger_now == t_end.group(1)),
]
g0_bad = [m for m, ok in G0 if not ok]
if g0_bad:
    print("G0 拒绝归档（face 未完成或树不牢）：")
    for m in g0_bad:
        print("   -", m)
    raise SystemExit(2)
print(f"G0 PASS：rc={t_rc.group(1)}，指纹 起＝止＝归档时现取＝{t_start.group(1)}，日志大小两次采样同 {s2} B，"
      f"我的启动器进程 0/{len(mine)}")
print(f"    观察（不判）：本机其它 `python -m pytest` 进程 {len(foreign)} 个 "
      f"{[l.split()[0] for l in foreign]} ⇒ 若与本轮时间窗重叠，本件的**墙钟时长**受他人负载影响，"
      "但记分与指纹判据不受影响（性能数字本轮一律不产）。")

tally = re.search(r"^(\d+ failed, \d+ passed, \d+ skipped in [\d.]+s)", raw, re.M)
need(tally, "汇总行未命中")
tally = tally.group(1)
nf, npass, nskip = (int(x) for x in re.findall(r"(\d+) (?:failed|passed|skipped)", tally))
total = nf + npass + nskip
# 进度字符只取 FAILURES 段**之前**的那一段点阵（失败正文里全是点与 F，取整文件必然数错）。
# ⚠ 起点必须**跳过** `SRCFINGER_START:` 这一行本身：首版从 `index("SRCFINGER_START")` 切，标签里的
#    那个大写 `F` 被当成第 3 个失败字符 ⇒ 点阵数 262 ≠ collected 261 的**假红灯**（本轮 G1 就这样假失败过）。
fs_start_line = raw.index("SRCFINGER_START")
seg = raw[raw.index("\n", fs_start_line) + 1:raw.index("= FAILURES =")]
nd, nF, ns = (seg.count("."), seg.count("F"), seg.count("s"))
coll_n = int(re.search(r"(\d+) tests? collected", coll_txt).group(1))
print(f"TALLY = {tally}")
print(f"① 结局合计 = {total} ＝ {nf} failed ＋ {npass} passed ＋ {nskip} skipped")
print(f"② 进度字符（点阵段）. ＝{nd}  F ＝{nF}  s ＝{ns} ⇒ {nd + nF + ns}")
print(f"③ `--collect-only -q` 实测 collected ＝ {coll_n}")
G1 = need(nd + nF + ns == coll_n, "G1 进度字符 ≠ collected") and need(total - coll_n == 2,
                                                                      "G1 结局 − collected ≠ 2")
ctrl1 = not (nd + nF + ns == coll_n - 1 and total - (coll_n - 1) == 2)
print(f"   扰动正对照（collected 少算 1）：{'变红 ⇒ 尺子能红' if ctrl1 else '仍通过 ⇒ 闸门失效！'}")
need(ctrl1, "G1 扰动正对照没变红")
# 同一条点阵判据的**自身**正对照：把起点退回标签行（首版的坏切法），必须数出 262 ⇒ 证明"跳过标签"这步有效
seg_bad = raw[fs_start_line:raw.index("= FAILURES =")]
seg_bad_F = seg_bad.count("F")
seg_bad_total = seg_bad_F + seg_bad.count(".") + seg_bad.count("s")
print(f"   扰动正对照（起点不跳过 SRCFINGER_START 标签行＝首版坏切法）⇒ 字符数 {seg_bad_total} "
      f"其中 F＝{seg_bad_F}（应为 {nF + 1}，即标签自己那个 F 被误计）")
need(seg_bad.count("F") == nF + 1, "G1 的坏切法正对照没变红 ⇒ 对'F 来自标签'的归因未经检验")

RE_RED = re.compile(r"^E\s+AssertionError: .*$", re.M)
reds_new, reds_ref = RE_RED.findall(raw), RE_RED.findall(ref)
print(f"\nG2 红灯行数：本件 {len(reds_new)} ／ 参照件 {len(reds_ref)}")
for a, b in zip(reds_ref, reds_new):
    print(f"   {'逐字节一致' if a == b else '不一致！'}  {a[:64]}…")
G2 = need(len(reds_new) == len(reds_ref) == 2, "G2 红灯行数不是 2") and need(reds_new == reds_ref,
                                                                            "G2 红灯文本与参照件不符")
if len(reds_new) == 2:
    last = reds_new[-1]
    tail_m = re.search(r"(\d+)$", last)
    mutated = last[:tail_m.start()] + str(int(tail_m.group(1)) + 1)
    ctrl2 = mutated != last and [reds_new[0], mutated] != reds_ref
    print(f"   扰动正对照（第 2 条末位 ＋1）：{'变红 ⇒ 尺子能红' if ctrl2 else '仍相同 ⇒ 尺子失效！'}")
    need(ctrl2, "G2 扰动正对照没变红")

def sect(t, name):
    """取 pytest 报告里某个 `= NAME =` 段的正文（到下一个段标题或文件尾为止）。"""
    m = re.search(r"^=+ " + name + r" =+\n(.*?)(?=^=+ |\Z)", t, re.M | re.S)
    return m.group(1) if m else ""


# ⚠ 首版把标题正则跑在**整份文件**上：参照件用了 `-rA`，于是 `= PASSES =` 段里通过用例的下划线标题
#    也被当成"红灯身份"（现取到 4 个，多出 test_cli_calibrate_roundtrip／test_inverse_training_converges）
#    ⇒ 本件与参照件"身份不同"的**假红灯**。修法＝只在 FAILURES 段内取标题。
nm_new, nm_ref = names_from_failures(sect(raw, "FAILURES")), names_from_failures(sect(ref, "FAILURES"))
nm_ref_whole = names_from_failures(ref)
short_failed = re.findall(r"^FAILED tests/\S+::(\w+)", raw, re.M)
print(f"\nG3 红灯身份：本件 FAILURES 段标题＝{nm_new}")
print(f"            参照件 FAILURES 段标题＝{nm_ref}")
print(f"            本件 `-q` 短汇总 FAILED 行＝{short_failed}")
print(f"            参照件**整文件**取下划线标题＝{nm_ref_whole} ← 坏口径会多出通过用例，故必须分段")
G3 = need(nm_new == nm_ref, "G3 红灯身份与参照件不同") and need(sorted(nm_new) == sorted(short_failed),
                                                               "G3 两种取法（标题 vs 短汇总）互斥")
ctrl3 = sorted(nm_new) != sorted(nm_ref_whole)
print(f"   扰动正对照（拿坏口径的参照清单来比）：{'变红 ⇒ 分段这一步确有分辨力' if ctrl3 else '仍相同 ⇒ 分段没起作用！'}")
need(ctrl3, "G3 的分段修正没通过正对照检验")

sk_new = sorted(set(re.findall(r"^SKIPPED \[\d\] (.*)$", skip_txt, re.M)))
sk_ref_all = re.findall(r"\[\d+\] (tests/[^:]+:\d+: .*)", ref)
sk_ref = sorted(set(sk_ref_all))
sk_cmds = re.findall(r"^CMD: (.*)$", skip_txt, re.M)
need(len(sk_cmds) == 2, "skip 件的两路 CMD 不齐（不能只凭一路写档案）")
print(f"\nG4 skip 身份：本件（-rs 现取）{len(sk_new)} 条 ／ 参照件 {len(sk_ref)} 条"
      f"（去重前 {len(sk_ref_all)} 条——参照件用了 `-rA`，同一 skip 在正文与短汇总各出现一次）")
for x in sk_new:
    print(f"   {x}")
print(f"   取法＝路 A（收集期）`{sk_cmds[0] if sk_cmds else '(缺)'}`"
      f" ＋ 路 B（测试体内）`{sk_cmds[1] if len(sk_cmds) > 1 else '(缺)'}`")
hosts_new = sorted(x.split(":")[0] for x in sk_new)
hosts_ref = sorted(x.split(":")[0] for x in sk_ref)
G4 = need(len(sk_new) == len(sk_ref) == 3, "G4 skip 条数不是 3") and need(hosts_new == hosts_ref,
                                                                        "G4 skip 宿主与参照件不同")
need(len(sk_ref_all) != 3, "G4 的'去重'正对照没变红 ⇒ 无法证明参照件真的重复报了 skip")
if sk_new:
    fake = sorted([("tests/NEW_FACE.py" if i == 0 else h) for i, h in enumerate(hosts_new)])
    ctrl4 = fake != hosts_ref
    print(f"   扰动正对照（1 条换新宿主）：{'变红 ⇒ 尺子能红' if ctrl4 else '仍相同 ⇒ 失效！'}")
    need(ctrl4, "G4 扰动正对照没变红")

fs = re.search(r"^SRCFINGER_START: ([0-9a-f]{32})", raw, re.M).group(1)
fe = re.search(r"^SRCFINGER_END: ([0-9a-f]{32})", raw, re.M).group(1)
rc = re.search(r"^rc=(\d+)", raw, re.M).group(1)
G5 = need(fs == fe, "G5 回归期间被测树被写过")
print(f"\nG5 指纹 起＝{fs} 止＝{fe} ⇒ {'同树 ✓' if fs == fe else '不同树！'}")
print(f"G6 rc（启动器实测写入）＝{rc}")

# G7「纯文本改动」的**机器**证明（替代我手数行号的老口径）：
# 把 HEAD 版与工作版都解析成 AST，**只把所有字符串常量置空**后比较 ast.dump ⇒
# 注释天然不进 AST；若两 dump 相等，则改动只可能落在注释与字符串常量上，语句/表达式/调用结构一字未变。
REL = "src/amforge/thermal_enthalpy.py"
head_txt = subprocess.run(["git", "-C", REPO, "show", f"HEAD:{REL}"],
                          capture_output=True, text=True, check=True).stdout
work_txt = open(os.path.join(REPO, REL), encoding="utf-8").read()


def blanked(t):
    tree = ast.parse(t)
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            n.value = ""
    return ast.dump(tree)


same_ast = blanked(head_txt) == blanked(work_txt)
diff_stat = subprocess.run(["git", "-C", REPO, "diff", "--numstat", "--", REL],
                           capture_output=True, text=True, check=True).stdout.strip()
G7 = need(same_ast, "G7 置空字符串常量后的 AST 不相等 ⇒ 改动并非纯文本")
print(f"\nG7 置空字符串常量后的 AST 对比：head {len(head_txt.splitlines())} 行 vs work "
      f"{len(work_txt.splitlines())} 行 ⇒ {'完全相等（纯注释＋字符串常量改动）' if same_ast else '不等！'}")
print(f"    `git diff --numstat` 现取＝{diff_stat or '(空)'}")
ctrl7 = blanked(work_txt + "\nMAGIC_SENTINEL = 1\n") != blanked(work_txt)
print(f"    扰动正对照（追加一条真语句后再比）：{'变红 ⇒ 尺子能红' if ctrl7 else '仍相等 ⇒ 尺子失效！'}")
need(ctrl7, "G7 扰动正对照没变红")

# G8 CPU 钉住：**首版判据是坏的**（要求全量日志正文里有 JAX 的回退行）。实测全文命中 0 行，原因不是
#      "没钉"，而是 `-q` 把**通过**用例内的 stderr 吞了（JAX 的平台回退打印在其首次设备查询时）。
#      ⇒ 换成两条各自可证的：① 本件自记的 CMD 以 `CUDA_VISIBLE_DEVICES= `（**空值**）开头；
#         ② 同 ENV／同树的独立证据件 `am_t35_cpu_evidence.log`（`-s` 关掉捕获）正文里现取到那两行 ＋ `BACKEND cpu`。
CPU_EV = os.path.join(REPO, "docs/evidence/2026-10-09/am_t35_cpu_evidence.log")
cmd_line = re.search(r"^CMD: (.*)$", raw, re.M).group(1)
pin_env = cmd_line.startswith("CUDA_VISIBLE_DEVICES= ")
ev = open(CPU_EV, encoding="utf-8").read() if os.path.exists(CPU_EV) else ""
ev_fb = [l for l in ev.splitlines() if "cuInit(0) failed" in l or "Falling back to cpu" in l]
ev_be = "BACKEND cpu" in ev
ev_finger = re.search(r"FINGER\(启动器口径\)＝起＝止＝([0-9a-f]{32})", ev)
ev_read = re.search(r"标记行读数：A＝(\d+)（全量面配置）／B＝(\d+)／C＝(\d+)／D＝(\d+)；全量件自身＝(\d+)", ev)
raw_hits = len([l for l in raw.splitlines() if "cuInit" in l or "Falling back" in l])
eA = eB = eC = eD = eF = -1  # 证据件四路读数；-1 ＝没读到（闸门会记 FAIL，档案不会拿它当数字用）
G8 = (need(pin_env, "G8 全量件 CMD 不以空值 CVD 开头 ⇒ 钉住无从谈起")
      and need(os.path.exists(CPU_EV), f"G8 证据件不存在：{CPU_EV}")
      and need(bool(ev_read), "G8 证据件没印出四路标记行读数 ⇒ 无法核对判据")
      and need(len(ev_fb) >= 2, "G8 证据件缺 JAX 的两行设备回退")
      and need(ev_be, "G8 证据件没现印 BACKEND cpu")
      and need(bool(ev_finger) and ev_finger.group(1) == fs, "G8 证据件与本件不同树（FINGER ≠ SRCFINGER）"))
if ev_read:
    eA, eB, eC, eD, eF = (int(x) for x in ev_read.groups())    # 链条三段：① A 路（＝全量面的配置）**取不到** ⇒ 全量件的 0 行被解释为捕获而非"没钉"；
    #          ② B/C 路（同树同环境，只把捕获打开）取到 ⇒ 回退确实在发生；③ D 路 backend＝cpu。
    G8 = G8 and need(eA == 0 and eF == 0 and eA == eF, "G8 A 路／全量件读数不为 0 ⇒ '捕获'这一解释被推翻") \
              and need(eB >= 2 and eC >= 2 and eD >= 2, "G8 B/C/D 三路没各自取到回退行") \
              and need(raw_hits == eF, f"G8 本件实测 {raw_hits} 行 vs 证据件所记全量件 {eF} 行不符")
    print(f"\nG8 CPU 钉住：全量件正文命中 JAX 回退行 **{raw_hits}** 条 ＝ 证据件 A 路（同配置）{eA} 条 "
          f"⇒ 原因＝捕获（不是没钉）")
    print(f"    ① CMD 前缀＝{cmd_line[:26]!r} ⇒ 空值 CVD {'成立' if pin_env else '不成立！'}")
    print(f"    ② B＝{eB}／C＝{eC}／D＝{eD} 条回退行；D 路 BACKEND cpu＝{'有' if ev_be else '无'}；"
          f"证据件指纹 {ev_finger.group(1) if ev_finger else '缺'} vs 本件 {fs}")
    for l in ev_fb[:4]:
        print(f"       {l[:118]}")
else:
    print("\nG8 CPU 钉住：证据件没给出四路读数行（闸门已记 FAIL）")
    print(f"    ① CMD 前缀＝{cmd_line[:26]!r} ② 证据件存在＝{os.path.exists(CPU_EV)} 回退行 {len(ev_fb)} 条")
ctrl8 = not cmd_line.replace("CUDA_VISIBLE_DEVICES= ", "").startswith("CUDA_VISIBLE_DEVICES= ")
print(f"    扰动正对照（把 CMD 里的空值 CVD 前缀删掉再跑同一条判据）：{'变红 ⇒ 尺子能红' if ctrl8 else '仍通过 ⇒ 尺子失效！'}")
need(ctrl8, "G8 扰动正对照没变红")
ctrl8b = bool(ev_read) and not (int(ev_read.group(1)) >= 1)
print(f"    扰动正对照（若 A 路能取到回退行＝全量件配置不再捕获）⇒ 判据应当变红：现 A＝"
      f"{ev_read.group(1) if ev_read else '?'}（＝0 ⇒ A 路确实取不到，判据链条成立）")

print("\n===== 闸门汇总 =====")
for n, v in (("G1 计数闭合", G1), ("G2 红灯逐字节", G2), ("G3 红灯身份", G3), ("G4 skip 身份", G4),
             ("G5 同树", G5), ("G6 rc 实测", bool(rc)), ("G7 纯文本(AST)", G7), ("G8 CPU 证据", G8)):
    print(f"  {n:14s} = {'PASS' if v else 'FAIL'}")

for b_ in bad:
    print(f"  ✗ {b_}")
if bad:
    raise SystemExit("闸门未全过 ⇒ 不写档案（先修仪器，或把不一致照实写进档案再人工判定）")
if "--write" not in sys.argv:
    print("\n（预检模式：未写档案。加 --write。）")
    raise SystemExit(0)

d_start = re.search(r"^DATE_START: (.*)$", raw, re.M).group(1)
d_end = re.search(r"^DATE_END: (.*)$", raw, re.M).group(1)
tree = re.search(r"^TREE: (.*)$", raw, re.M).group(1)
cmd = re.search(r"^CMD: (.*)$", raw, re.M).group(1)
coll_cmd = re.search(r"^CMD: (.*)$", coll_txt, re.M).group(1)
coll_stamp = re.search(r"^(DATE: .*)$", coll_txt, re.M).group(1)
skip_stamp = re.search(r"^(# DATE: .*)$", skip_txt, re.M).group(1)
body_txt = raw[raw.index("TREE: "):]
hdr = f"""===== #35 全量回归记分（CPU 钉住；纯文本改动轮）=====
CMD（启动器 /tmp/am_t35_full_run.sh 内的原命令行，非手抄）: {cmd}
ENV: CUDA_VISIBLE_DEVICES=（空＝CPU 钉住，用户令「继续，还是不动GPU」） ＋ JAX_ENABLE_X64=1 ＋ PYTHONPATH=src
TREE: {tree}（dirty=1 ＝本片入库的 `src/amforge/thermal_enthalpy.py`；口径 `-- src tests`）
DATE_START: {d_start}；DATE_END: {d_end}
SRCFINGER: 起＝止＝{fs}（md5(concat per-file md5)，`src`＋`tests` 的 .py，排除 __pycache__）⇒ G5 同树
rc（实测，启动器写入）= {rc}
TALLY: {tally}
NOTE: G0「face 真的跑完」写在**门口**而不是靠下游正则报错：要求 `DATE_END` ＋ 实测 `rc=` 都在、
NOTE:      `SRCFINGER_START`==`SRCFINGER_END`==**归档时现取的指纹**、我的启动器脚本无存活进程、
NOTE:      且本日志 3 秒两次采样大小不变（{s2} B）。
NOTE:      ⚠ 首版这条闸门**判错了对象**：它要求"机器上没有 `python -m pytest` 在跑"，于是被**另一个
NOTE:      CLI（父进程 codebuddy，跑的是 /home/shy/桌面/sim_env 的 godot 用例）的 pytest** 拦住 ⇒
NOTE:      一次**已完成**的回归被拒（假拒）。我要证的从来不是"世界上没人跑测试"，而是"**我这份**日志
NOTE:      已封口、**这棵树**在归档时仍与跑完时同指纹"。⇒ 判据改挂在属于自己的对象上（自己的脚本名、
NOTE:      自己的日志、自己的树），别人的进程只**打印**不**判定**（它影响墙钟，而本轮不产任何墙钟结论）。
NOTE:      首轮预检还在**半成品**日志上崩过：崩在 `tally.group(1)`（离真因"还在跑"很远）⇒ 补此道闸门，
NOTE:      现以 rc=2 具名拒绝；负对照＝未完成日志重跑预检，必须打出「G0 拒绝归档」而不是 traceback。
NOTE: G1 计数闭合（本轮 `-q` 无 `-rA` ⇒ 比上轮少一路，照实说明而不是假称四路）：
NOTE:      ① 结局 {total}＝{nf} failed＋{npass} passed＋{nskip} skipped；
NOTE:      ② 点阵段（FAILURES 之前）字符 . ＝{nd}／F ＝{nF}／s ＝{ns} ⇒ {nd + nF + ns}；
NOTE:      ③ 跑**中**同树单测 `{coll_cmd}` 现取 collected＝{coll_n}（{coll_stamp}；
NOTE:         该件自报 FINGER＝启动器口径的 SRCFINGER，与本件起/止同值 ⇒ 与全量面同树）；
NOTE:      ④ 闭合关系：结局 {total} − collected {coll_n} ＝ 2 ＝ 2 条**模块级** importorskip
NOTE:         （收集期 skip ⇒ 不进收集计数、不占点阵、却计入汇总行 skipped）。
NOTE:      ⚠ 点阵段的**起点**必须是 `SRCFINGER_START:` 那一行**之后**：首版从该行行首切，标签里那个
NOTE:      大写 `F` 被当成第 3 个失败字符 ⇒ 数出 {seg_bad_total} ≠ collected {coll_n} 的**假红灯**（本轮 G1 假失败）。
NOTE:      扰动正对照两条：collected 少算 1 ⇒ 变红；起点退回标签行 ⇒ F 数＝{seg_bad_F}＝{nF}+1（归因经检验）。
NOTE: G2 两条登记红灯**逐字节未变**（A0 不许放宽；#27 那条不许换指标）：
NOTE:      `^E\\s+AssertionError: ` 在本件与参照件 {REF.split('/')[-1]} 各 {len(reds_new)}/{len(reds_ref)} 行且逐行相等；
NOTE:      扰动正对照＝第 2 条末位 ＋1 ⇒ 比较立刻变红。
NOTE: G3 红灯身份两处取法互洽（均现取，不手抄）：FAILURES 段下划线标题＝{nm_new}
NOTE:      ＝ `-q` 短汇总 FAILED 行＝{sorted(set(short_failed))} ＝参照件标题 {nm_ref}。
NOTE:      ⚠ 首版把标题正则跑在**整份文件**上：参照件用了 `-rA`，`= PASSES =` 段里通过用例的下划线标题
NOTE:      一并被取走（现取 {len(nm_ref_whole)} 个 vs FAILURES 段内 {len(nm_ref)} 个，多出的是**通过**用例）
NOTE:      ⇒ "身份不同"的**假红灯**。修法＝只在 FAILURES 段内取；扰动正对照＝拿坏口径清单来比 ⇒ 必须变红。
NOTE: G4 skip 身份取自**跑中同树**（{skip_stamp}）的 `-rs` 定向件，**不按位置反推**；分两路（口径不同）：
NOTE:      路 A（收集期 skip，`--collect-only -rs` 即可现取，其 rc=5「no tests collected」正是
NOTE:         「这两条不占 collected 计数」的直接证据）＝ `{sk_cmds[0]}`
NOTE:      路 B（测试体内 skip，收集期不报 ⇒ 必须实跑那一条 nodeid，只跑一条以避开同文件慢测）＝
NOTE:         `{sk_cmds[1]}`
{chr(10).join('NOTE:      [' + str(i + 1) + '] ' + x for i, x in enumerate(sk_new))}
NOTE:      宿主与参照件的 3 条相同 ⇒ **无新增 skip/xfail**；扰动正对照＝1 条换 `tests/NEW_FACE.py` ⇒ 变红。
NOTE:      ⚠ 参照件那条式**去重前**命中 {len(sk_ref_all)} 条（`-rA` 让同一 skip 在正文与短汇总各出现一次）
NOTE:      ⇒ 首版拿 6 与 3 相比，报出"本件 skip 条数不是 3"的**假红灯**；去重是口径修正而不是放宽。
NOTE: G6 rc＝{rc} 为**实测**（启动器写 rc 到日志；上轮启动器没写，那件的 rc 只是推断 ⇒ 本轮补上口径）。
NOTE: G7「纯文本改动」的机器证明（**取代手数行号**）：把 `HEAD:{REL}` 与工作版各自 `ast.parse`、
NOTE:      **把所有字符串常量置空**后比 `ast.dump` ⇒ 结果＝{('两棵 AST 逐字符相等' if same_ast else '不相等！')}；
NOTE:      注释天然不进 AST，故相等即＝改动只落在注释与字符串常量上，语句/表达式/调用结构一字未变。
NOTE:      `git diff --numstat` 现取＝{diff_stat or '(空)'}；扰动正对照＝往工作版追加一条真语句后再比 ⇒ 变红。
NOTE:      ⇒ 行为面唯一可能的差异是 `warnings.warn` 的消息串；该串的实测渲染见同目录
NOTE:      `am_t35_postcheck.log` 的 P3 段。本轮**不产任何吞吐/加速数字**（性能只在 A6000 上测）。
NOTE: G8 CPU 钉住：**首版判据是坏的**（要求本件正文里有 JAX 的 `cuInit(0) failed: CUDA_ERROR_NO_DEVICE`
NOTE:      ＋ `Falling back to cpu` 两行），实测命中 {raw_hits} 行 ⇒ **一次正确的轮次被仪器拒绝**（假失败）。
NOTE:      四路实测（同树 {fs}、同启动环境，件＝`am_t35_cpu_evidence.log`）把这个 0 **解释**掉而不是忽略：
NOTE:      A 路＝本面配置复本（`-q -s`）→ {eA} 行；B 路＝`-p no:logging` → {eB} 行；
NOTE:      C 路＝`--log-cli-level=WARNING`（插件一个不摘）→ {eC} 行；D 路＝进程级 `import jax` → {eD} 行
NOTE:      ＋ stdout `BACKEND cpu`；本件自身 → {eF} 行。⇒ `-s` 只关**文件描述符**捕获，而 JAX 的回退走
NOTE:      `logging`，被 pytest 的 logging 插件收走、用例通过时不回显 ⇒ 「日志里没有」既不是钉住的证据
NOTE:      也不是没钉住的证据。
NOTE:      ⇒ 本件的判据换成三条各自可证的：① CMD 以 `CUDA_VISIBLE_DEVICES= `（**空值**）开头；
NOTE:      ② C 路在同树同环境现取到那两行；③ D 路 backend＝cpu。扰动正对照＝把 ① 的前缀删掉 ⇒ 变红。
NOTE:      旁证（**不单独当判据**）：两条红灯打印值与 CPU 钉住的参照件逐字节相同（见 G2）。
NOTE: N9 本轮自身缺陷照实登记（(a)(b) 在写档案前发现并修；(c)(d) 是同轮里被闸门自己抓出来的**假红/假拒**）：
NOTE:      (a) skip 测量件首版用 `echo "# …`--collect-only -rs`…"` 写文案 ⇒ 双引号里的反引号被 bash
NOTE:         当命令替换执行，说明行被掏空并报 `--collect-only: 未找到命令`（这是同一条坑的**第 6 次**）。
NOTE:         修法＝文案改走 python 字面量（`/tmp/am_t35_skips.py`），重跑后指纹与全量面一致。
NOTE:      (b) 我为验证"钉住 CPU 的 stderr 长什么样"跑了一条**未带** `CUDA_VISIBLE_DEVICES=` 的
NOTE:         `python -c "import jax"`，它当场把 backend 报成 gpu＝真的初始化了 CUDA context，
NOTE:         正是探针脚本里那道 pre-import 闸门所禁止的事。进程即刻退出、`nvidia-smi` 现取（本轮）
NOTE:         显示 GPU 上的 compute 进程均非此次调用所留 ⇒ 无残留占用；但**违规成立**，
NOTE:         教训＝闸门写在脚本里不等于守住了习惯，任何碰 jax 的一次性命令也必须先带 CVD 空串。
NOTE:      (c) 四道**仪器缺陷**在同一轮里造出"假读数"：G0 把闸门挂在"机器上有没有 pytest"（被**另一个
NOTE:         CLI 的** pytest 拦住 ⇒ 已完成的回归被假拒）；G1 点阵段起点没跳过 `SRCFINGER_START:` 标签行
NOTE:         （标签里那个大写 F 被当成第 3 个失败字符 ⇒ 262≠261 假红）；G3 在整份文件上下划线标题正则
NOTE:         （参照件 `-rA` 的 `= PASSES =` 段贡献 2 个通过用例名 ⇒ 假红）；G4 参照件 skip 未去重
NOTE:         （同一 skip 在正文与短汇总各一次 ⇒ 6 vs 3 假红）。⇒ 统一教训：**闸门要挂在自己的对象上**
NOTE:         （自己的日志/树/脚本名），且"从别人那份档案继承来的取法"必须先按**那一份的报告格式**校准；
NOTE:         四条都配了"退回坏口径就必须变红"的正对照，见各门 ctrl 行。
NOTE:      (d) CPU 证据件首版手挑 6 个环境变量 ⇒ JAX 的 xla_cuda13 插件走另一分支，连那次 `cuInit(0)`
NOTE:         都不做，两行证据全缺、E4 假失败（件内 E0 已登记）。⇒ "同一启动环境"要靠**继承 os.environ**
NOTE:         ＋只覆盖那三项来实现，而不是靠把变量名列出来。
"""
off = len((hdr + "BODY_OFFSET=%020d\n" % 0).encode())
hdr_now = hdr + "BODY_OFFSET=%020d\n" % off
assert len(hdr_now.encode()) == off, "BODY_OFFSET 未收敛"
open(OUT, "w", encoding="utf-8").write(hdr_now + body_txt)
back = open(OUT, "rb").read()
got = int(re.search(rb"BODY_OFFSET=(\d+)", back).group(1))
assert got == len(back) - len(body_txt.encode()), "复核：BODY_OFFSET ≠ 实际正文起点"
assert back[got:] == body_txt.encode(), "复核：正文区不是原样字节"
print(f"\n已写 {OUT}：{len(back.splitlines())} 行；BODY_OFFSET={got}（写后复核：正文区逐字节原样 ✓）")
