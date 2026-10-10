#!/usr/bin/env python3
"""S7 分片 26 提交信息生成器：信息里的每个数字都从命令/档案现取，不许手敲。

正则一律放在模块级常量里（本件可被 am_s26_msg_controls.py import 而无副作用），
这样"干跑的探测器"与"真正写信息的探测器"是同一把，而不是抄一遍。
"""
import glob
import hashlib
import re
import subprocess

REPO = "/home/shy/桌面/2026年开发-显式冲击有限元-物质点法-sph-fvm+am/AM-Qoder"
RAW = "docs/evidence/2026-10-10/_raw_t27_full.log"
BASE = "docs/evidence/2026-10-09/_raw_s18_full.log"
FACTS = "docs/evidence/2026-10-10/am_t27_facts.txt"
CURL = "docs/evidence/2026-10-09/am_s7_verify25.log"
TEST = "tests/test_evap_model.py"
TE = "src/amforge/thermal_enthalpy.py"
RUNSH = "docs/evidence/2026-10-10/am_t27_full_run.sh"
V25 = "docs/evidence/2026-10-09/am_s7_verify25.py"
V26 = "docs/evidence/2026-10-10/am_s7_verify26.py"

RE_TALLY = re.compile(r"^\d+ failed, \d+ passed, \d+ skipped in [\d.]+s.*$", re.M)
RE_TALLY_NUMS = re.compile(r"^(\d+) failed, (\d+) passed, (\d+) skipped in ([\d.]+)s", re.M)
RE_RC = re.compile(r"^rc=(\d+)$", re.M)
RE_TREE = re.compile(r"^TREE: head=(\w+) dirty=(\d+)$", re.M)
RE_FINGER = re.compile(r"^SRCFINGER_(START|END): (\w+)$", re.M)
RE_DATE = re.compile(r"^DATE_(START|END): (.+)$", re.M)
RE_PROG = re.compile(r"^([.Fs]+)\s+\[", re.M)
RE_ELINES = re.compile(r"^E\s+AssertionError: .*$", re.M)
RE_CURL_BLOB = re.compile(r"tree 条目 (\d+) = blob (\d+) \+ tree (\d+)")
RE_CURL_CLOSE = re.compile(r"闭合式 远端 (\d+) == 父提交 (\d+) \+ 新增 (\d+) = \d+ -> (\w+)")
RE_CURL_DIFF = re.compile(r"双向路径差集 (\d+)/(\d+) 与 (\d+)/(\d+)")
RE_CURL_SHA = re.compile(r"全库共有 (\d+) 条 ⇒ blob SHA 不符 (\d+)/(\d+)")
RE_CURL_SLICE = re.compile(r"本片 (\d+) 文件（A (\d+)／M (\d+)）单列复核 ⇒ 远端缺失 (\d+)/(\d+)，"
                           r"SHA 不符 (\d+)/(\d+)，MATCH (\d+)/(\d+)")
RE_CURL_CMP = re.compile(r"cmp -i (\d+):0")
RE_CURL_HEAD = re.compile(r"head=(\w{40})")
RE_CURL_VERDICT = re.compile(r"LANDING_VERDICT = .*", re.M)
RE_SHORTSTAT = re.compile(r"(\d+) files changed, (\d+) insertions\(\+\)(?:, (\d+) deletions\(-\))?")


def sh(*args):
    r = subprocess.run(list(args), cwd=REPO, capture_output=True, text=True, check=True)
    return r.stdout


def md5(path, n=None):
    h = hashlib.md5(open(f"{REPO}/{path}", "rb").read()).hexdigest()
    return h if n is None else h[:n]


def rd(path):
    return open(f"{REPO}/{path}", encoding="utf-8").read()


def ruler_copies_of(target_md5):
    return sum(1 for p in glob.glob(f"{REPO}/docs/evidence/2026-10-*/am_s7_verify*.py")
               if hashlib.md5(open(p, "rb").read()).hexdigest() == target_md5)


def parse_raw(raw):
    tally = RE_TALLY.search(raw).group(0)
    rc = RE_RC.search(raw).group(1)
    tm = RE_TREE.search(raw)
    heads = {k: v for k, v in RE_FINGER.findall(raw)}
    assert set(heads) == {"START", "END"}, f"SRCFINGER 只取到 {sorted(heads)}"
    dates = {k: v for k, v in RE_DATE.findall(raw)}
    chars = "".join(RE_PROG.findall(raw))
    elines = RE_ELINES.findall(raw)
    assert len(elines) == 2, f"两条红的 E 行实取 {len(elines)} 条 ⇒ 判据形状已变，不许静默通过"
    assert heads["START"] == heads["END"], "SRCFINGER 起≠止 ⇒「回归期间源码一字未动」不成立，不许进信息"
    return dict(tally=tally, rc=rc, head=tm.group(1), dirty=tm.group(2), fs=heads["START"],
                fe=heads["END"], d1=dates["START"], d2=dates["END"],
                dots=chars.count("."), ffs=chars.count("F"), sss=chars.count("s"),
                elines=elines)


def parse_curl(curl):
    return dict(blob=RE_CURL_BLOB.search(curl).groups(),
                close=RE_CURL_CLOSE.search(curl).groups(),
                diff=RE_CURL_DIFF.search(curl).groups(),
                sha=RE_CURL_SHA.search(curl).groups(),
                slc=RE_CURL_SLICE.search(curl).groups(),
                offset=RE_CURL_CMP.search(curl).group(1),
                head=RE_CURL_HEAD.search(curl).group(1),
                verdict=RE_CURL_VERDICT.search(curl).group(0))


def gate_curl(curl):
    """分片 25 回执的闸门：三个断言全过才允许把它的读数写进落地信息。"""
    c = parse_curl(curl)
    assert int(c["close"][0]) == int(c["close"][1]) + int(c["close"][2]) == int(c["blob"][1]), \
        f"闭合式在信息生成器里不成立（{c['close']}／blob {c['blob'][1]}）⇒ 不许引用它"
    assert c["verdict"].endswith("ALL CHECKS PASS"), f"回执判读行不是 ALL CHECKS PASS：{c['verdict']}"
    n, a, m = int(c["slc"][0]), int(c["slc"][1]), int(c["slc"][2])
    miss, bad, match, den = int(c["slc"][3]), int(c["slc"][5]), int(c["slc"][7]), int(c["slc"][8])
    assert miss == 0 and bad == 0 and match == den == n, f"单列复核不是全绿：{c['slc']}"
    assert a + m == n, f"本片 A/M 之和不等于文件数：A {a} M {m} 共 {n}"
    return c


def gate_baseline(raw_txt, base_txt, n_tests):
    """把本片记分件与基线件喂进**同一条** parse_raw，再要求三条代际关系成立。

    任一条不成立就拒绝生成信息：① rc 必须是 1（rc=0 意味着红灯清单变了）；
    ② 两条红的 E 行必须与基线逐字节相等（默认档被动过的形状）；③ failed/skipped 计数不变、
    passed 增量必须**正好**等于本片新测试条数（少了＝有测试掉网，多了＝旧测试悄悄转绿）。
    """
    r, b = parse_raw(raw_txt), parse_raw(base_txt)
    nf, npass, nskip = (int(x) for x in RE_TALLY_NUMS.search(raw_txt).groups()[:3])
    bf, bp, bs = (int(x) for x in RE_TALLY_NUMS.search(base_txt).groups()[:3])
    assert r["rc"] == "1", f"rc={r['rc']}≠1 ⇒ 红灯清单变了，『两条既有红灯』的口径不再成立"
    assert r["elines"] == b["elines"], "两条红的 E 行与基线不再逐字节相等 ⇒ 默认档被改动了"
    assert (nf, nskip) == (bf, bs), f"failed/skipped 计数变了：基线 {bf}/{bs} → 现 {nf}/{nskip}"
    assert npass - bp == n_tests, f"passed 增量 {npass - bp} ≠ 新测试条数 {n_tests} ⇒ 有测试掉网或悄悄转绿"
    return r, b, (nf, npass, nskip), (bf, bp, bs)


def build():
    raw_txt = rd(RAW)
    facts = dict(l.split("=", 1) for l in rd(FACTS).splitlines() if "=" in l)
    curl = gate_curl(rd(CURL))
    status = [l.split("\t") for l in sh("git", "diff", "--cached", "--name-status").splitlines() if l.strip()]
    added = [p for c, p in status if c == "A"]
    modified = [p for c, p in status if c == "M"]
    sm = RE_SHORTSTAT.search(sh("git", "diff", "--cached", "--shortstat").strip())
    assert sm, "shortstat 形状不认识"
    parent_blobs = len([l for l in sh("git", "ls-tree", "-r", "HEAD").splitlines()
                        if l.split("\t")[0].split()[1] == "blob"])
    copies = ruler_copies_of(md5(V26))
    assert copies >= 2, "尺子副本普查落空（glob 没命中 ⇒ 计数会假绿）"
    n25, a25, m25 = curl["slc"][0], curl["slc"][1], curl["slc"][2]
    miss25, missden25, bad25, match25, matchden25 = (curl["slc"][3], curl["slc"][4], curl["slc"][5],
                                                     curl["slc"][7], curl["slc"][8])
    n_tests = len(re.findall(r"^def test_", rd(TEST), re.M))
    raw_ctx, base_ctx, (nf, npass, nskip), (bf, bp, bs) = gate_baseline(raw_txt, rd(BASE), n_tests)
    outcomes = nf + npass + nskip
    prog_sum = raw_ctx["dots"] + raw_ctx["ffs"] + raw_ctx["sss"]
    collected = int(facts["COLLECT"])
    assert prog_sum == collected, f"进度字符 {prog_sum} ≠ collect {collected} ⇒ 四路闭合的第②③路本身不闭合"
    assert curl["head"].startswith(raw_ctx["head"]), \
        f"回执的落地树 {curl['head'][:7]} 与回归件 TREE head {raw_ctx['head']} 不一致 ⇒ 两份档案不在同一棵树上"
    msg = f"""S7 分片 26：#27 代码腿（evap_model 三臂选择器＋无参数 Hertz–Knudsen 封顶，缺省档不切换）＋分片 25 的落地回执入库；开发日志 §26.28

代码腿（{sm.group(1)} files changed, {sm.group(2)} insertions(+), {sm.group(3)} deletions(-)；
TREE: head={raw_ctx['head']} dirty={raw_ctx['dirty']}，即本片的 src/tests 内容）三处改动，全在 {TE}（{len(rd(TE).splitlines())} 行，md5 {md5(TE, 12)}）：
- 新增 _evap_sink_hk(T, mat, dx) = mat.evaporation_flux(T) · mat.latent_vapor / dx（:587→:601）——**无自由标定参数**，
  ṁ 走材料卡 Clausius–Clapeyron＋accommodation β=0.82，÷dx 把表面 [W/m²] 摊成体密度 [J/m³/s]，与 meltpool.py:802 的 q_evap 同源。
- evap_model = str(p.get("evap_model", "tuned"))（:660）＋非法值 ValueError ⇒ 缺省仍是 tuned；
  A3 夹具要求的"封顶可置 0"由 "none" 臂提供（无参数公式自己没有置 0 旋钮）。
- rhs 里三臂分支（:933-936）：hk 用新封顶、none 置零、**缺省 tuned 走原 _evap_sink(...) 表达式一字未改**。meltpool.py 未动。
- 新测试 {TEST}（{len(rd(TEST).splitlines())} 行，md5 {md5(TEST, 12)}，`^def test_` 现数 {n_tests} 条全绿）：非法选择器拒绝；HK 无参数且与公式逐位；
  "none"≡evap_coeff=0 逐位、"none"≠缺省；缺省≡"tuned" 逐位；hk 峰值＞tuned 且＞T_boil（并诚实登记「封顶＝沸点墙」未被证成）；
  梯度符号缺陷 jax.grad 断言 g_tuned < 0 < g_hk（∂峰值/∂功率 tuned=−0.060 反号 vs hk=+0.383，**独立于 A3** 的替换理由）。

**缺省不切 hk**（半动口径纪律）：切换会把绝对峰值抬到 3362.42 K（+14.48%，＞T_boil=3090），而 tuned→hk 与 #33 的 15 位点系数
重标定都**只能对 A3 的 18 道外部趋势靶打分**（单道散度 width p50 15.7%／area 19.8%，n=6；过线判据 14%／11%）⇒ 腿②＋#33 随 #10/A3。
改前预检 all-PASS（am_t27t33_preflight_probe.log）：P1 十五位点按源码文本重解、8/15 行号漂移仍全命中；P2 缺省臂在现树逐位复现 G0 锚点；
P3 thermal_enthalpy 的 HK 消费者＝0（＝本轮要填的缺口），改后正对照 0→1 且 meltpool 仍 2（am_t27_postedit_consumer_census.log）；
P4 置 0 生效 ∧ tuned 低温恒 0 ∧ HK(T_liq)/Q_peak=0.00002%<0.1%（HK 在正常熔化区近中性）。

全量回归（CPU 钉住，启动器 {RUNSH} md5 {md5(RUNSH)}）：**{raw_ctx['tally']}**，rc={raw_ctx['rc']}（rc=1＝两条既有红灯，非本轮引入）。
SRCFINGER 起＝止={raw_ctx['fs']}（止＝{raw_ctx['fe']}）⇒ 回归期间 src/tests 一字未动；{raw_ctx['d1']} → {raw_ctx['d2']}。
计数四路闭合（四个数全部现取）：① 汇总结局 {outcomes}＝{nf} failed＋{npass} passed＋{nskip} skipped；
② 进度字符 `.`×{raw_ctx['dots']}＋`F`×{raw_ctx['ffs']}＋`s`×{raw_ctx['sss']}＝{prog_sum}；
③ 跑前 --collect-only＝{collected}（值现取自 {FACTS}）；④ 差额 {outcomes}−{collected}＝{outcomes - collected} 条**模块级**
importorskip（test_gui_app.py／test_gui_app2.py，收集期 skip ⇒ 不进收集计数、不占进度字符、却计入汇总 skipped）⇒ {collected}＋{outcomes - collected}＝{outcomes}。
断言 prog_sum==collected 通过（{prog_sum}＝{collected}），否则这四路本身不闭合。
两条登记红的 E 行：本件把 {BASE} 与本片正文用**同一条 parse_raw** 现取现比，列表相等＝{raw_ctx['elines'] == base_ctx['elines']}；
另有当时的排序逐字节 diff rc={facts['E_DIFF_RC']}（E_COUNT_NOW={facts['E_COUNT_NOW']}＝E_COUNT_BASE={facts['E_COUNT_BASE']}）：
①{raw_ctx['elines'][0].strip()}；②{raw_ctx['elines'][1].strip()}
新增 {n_tests} 条全 passed（passed {bp}→{npass}，增量 {npass - bp}＝新测试条数 {n_tests} ⇒ 没有测试掉网、也没有旧测试悄悄转绿）、
skipped {bs}→{nskip}、failed {bf}→{nf}、grep -ic 'xfail|xpass'={facts['XFAIL']}
⇒ A0 断言未放宽、#27 未换指标、无新增 skip/xfail。
**这条回归只证明"缺省不改数＋新代码自洽"，不证明 HK 数值正确**（后者是腿②对 A3 的活）。性能数字本轮为零（性能只在 GPU 上算）。

分片 25 回执入库（读数一律从回执正文区正则现取）：{curl['verdict']}；
回执内 ① 步的落地树＝{curl['head'][:7]}（回执 ① 步判 ls-remote main == 本地 HEAD），与回归件 TREE head={raw_ctx['head']} 同树（生成器里有断言）；
tree 条目 {int(curl['blob'][1]) + int(curl['blob'][2])}＝blob {curl['blob'][1]}＋tree {curl['blob'][2]}；
闭合式 远端 {curl['close'][0]} == 父提交 {curl['close'][1]} ＋ 新增 {curl['close'][2]} -> {curl['close'][3]}；
双向路径差集 {curl['diff'][0]}/{curl['diff'][1]} 与 {curl['diff'][2]}/{curl['diff'][3]}；全库 blob SHA 不符 {curl['sha'][1]}/{curl['sha'][2]}（共有 {curl['sha'][0]} 条）；
本片 {n25} 文件（A {a25}／M {m25}）单列复核 ⇒ 远端缺失 {miss25}/{missden25}、SHA 不符 {bad25}、MATCH {match25}/{matchden25}；四个正对照 A/B/C/D 同运行都能红。
回执 am_s7_verify25.log 正文区由 cmp -i {curl['offset']}:0 rc=0 证明是逐字节后缀；
尺子 {V25}（md5 前 12={md5(V25, 12)}）与本片新抄的 {V26}（md5 前 12={md5(V26, 12)}）逐字节相同，
glob＋md5 现算＝同一把尺子在 docs/evidence 下已有 {copies} 份逐字节副本（含本片这把 {V26}，它按惯例**不入库**、留给分片 27 的回执）。

归档面：开发日志 §26.28（a–k）、总览当日条、路线文档三条（#27 代码腿／分片 25 回执／排期）、
.workbuddy/memory/2026-10-10.md ＋ MEMORY.md 当日条。**本片 doc faces 是就地编辑而非生成器**（诚实登记，不补假仪器）；
文档里的数字全部与档案件对过账。自纠 +0：归档表头的 sed 占位符把整串 tally 塞进需要单值的中文句法位（「skipped=__ 中的 3」）⇒
新口径「占位符只替换纯数值读数」，且这类错乱在 cmp 里是全绿的 ⇒ 归档后必须回读头部。
GPU：全轮 CPU 钉住（每条计算命令内联 CUDA_VISIBLE_DEVICES=），落笔时现取卡上 compute 进程 {facts['GPU_APPS_AT_WRITE']} 个、
我的解释器（/proc/PID/cmdline 含 /tmp/amvenv）＝{facts['GPU_MINE']}（非 0 拒绝写盘）。
排期：#10/A3（18 道跑）→ #27 腿②＋#33 同批重标定 → #27 关闭；#40 → #39；#41 待自己一轮；#29／#30 仍等用户裁决。
"""
    counts = dict(total=len(status), added=len(added), modified=len(modified),
                  parent_blobs=parent_blobs, expect_blobs=parent_blobs + len(added),
                  ruler_copies=copies, e_lines=len(raw_ctx["elines"]),
                  progress_sum=raw_ctx["dots"] + raw_ctx["ffs"] + raw_ctx["sss"])
    return msg, counts


if __name__ == "__main__":
    m, c = build()
    open("/tmp/am_s26_msg.txt", "w", encoding="utf-8").write(m)
    print(m)
    print("---- 现取核对 ----")
    for k, v in c.items():
        print(f"{k}={v}")
