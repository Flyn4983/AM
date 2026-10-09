#!/usr/bin/env python3
"""分片 25 的提交说明生成器：**一个读数都不手敲**，全部从已归档的仪器件里正则现取。

为什么要有这个文件：提交说明也是档案面（GitHub 上第一条会被读的读数），而既有教训是
「数字一律现取，文字也要」——手写 body 里的 md5/行数/记分，就是下一片的假读数来源。
只有当下决定（排期、本轮做了什么）由人写；带数字的句子一律由命令填。
闸门：正文区逐字节＝归档 stdout、cmp 三联、闭合式、两红逐行等于基线、记分闭合式
`基线 passed ＋ 新收集 ＝ 现树 passed`、指纹起止相等且＝现算、skip 标记行不变、
尺子名单现比、自纠计数从总览现取、假哈希传染面**分两面**现数（历史面须 0／工作树面每处须与真值同段，
并把真值抹掉做正对照）、+0 名单由列表数出来、GPU 我的进程须 0。
"""
import json
import os
import re
import subprocess

REPO = "/home/shy/桌面/2026年开发-显式冲击有限元-物质点法-sph-fvm+am/AM-Qoder"
D = os.path.dirname(os.path.abspath(__file__))
FALSEH = "3b4aabb"          # 台账里那个手敲的假短哈希＝缺陷标本（引文，不是读数）
FIVE = ["docs/开发日志.md", "docs/项目开发总览.md", "docs/项目评估与下一步计划_2026-10-06.md",
        ".workbuddy/memory/2026-10-09.md", ".workbuddy/memory/MEMORY.md"]
SCORE = r"(\d+) failed, (\d+) passed, (\d+) skipped in ([\d.]+)s"
SCORE_T = r"(\d+) failed, (\d+) passed in ([\d.]+)s"


def sh(*a):
    return subprocess.run(a, cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def md5abs(p):
    return subprocess.run(["md5sum", p], capture_output=True, text=True).stdout.split()[0]


def md5rel(rel):
    return md5abs(os.path.join(REPO, rel))[:12]


def txtabs(p):
    return open(p, encoding="utf-8").read()


def nl(rel):
    return str(len(txtabs(os.path.join(REPO, rel)).splitlines()))


def rd(rel):
    return txtabs(os.path.join(D, rel))


# ---------------- 分片 24 回执：正文区按 BODY_OFFSET 的**字节**位置切 ----------------
raw_b = open(os.path.join(D, "am_s7_verify24.log"), "rb").read()
boff = int(re.search(rb"BODY_OFFSET=(\d+)\n", raw_b).group(1))
head_txt = raw_b[:boff].decode("utf-8")
body_txt = raw_b[boff:].decode("utf-8")
assert raw_b[boff:] == open(os.path.join(D, "_raw_verify24.out"), "rb").read(), "正文区不是那次 stdout 的逐字节后缀"
CMP = [subprocess.run(["cmp", "-i", "%d:%d" % (a, b), os.path.join(D, "am_s7_verify24.log"),
                       os.path.join(D, "_raw_verify24.out")], capture_output=True).returncode
       for a, b in ((boff, 0), (boff + 1, 0), (boff, 1))]
assert CMP == [0, 1, 1], "cmp 三联不齐：%s" % CMP


def B(pat, grp=1):
    m = re.search(pat, body_txt)
    assert m, "回执正文取不到 " + pat
    return m.group(grp)


def HP(pat, grp=1):
    m = re.search(pat, head_txt)
    assert m, "回执表头取不到 " + pat
    return m.group(grp)


VERDICT = B(r"(LANDING_VERDICT = .*)").strip()
assert VERDICT == "LANDING_VERDICT = ALL CHECKS PASS", VERDICT
NB, NP, NA = int(B(r"blob (\d+) \+ tree")), int(B(r"父提交 = (\d+)")), int(B(r"新增 (\d+) \+ 修改"))
EXPECT = int(B(r"\+ 新增 \d+ = (\d+)"))
assert NB == NP + NA == EXPECT, "闭合式在正文里就不成立：%d vs %d+%d vs %d" % (NB, NP, NA, EXPECT)

# ---------------- 本轮仪器件 ----------------
rec, side, full, base = rd("_raw_s18_receipt.out"), rd("_raw_s18_side.log"), rd("_raw_s18_full.log"), \
    rd("am_t35_full_regression.log")
fs, bs = re.search(SCORE, full), re.search(SCORE, base)
tg = re.search(SCORE_T, side.split("===== TARGETED 段 =====")[1])
assert fs and bs and tg, "记分/定向读数取不齐"
RC_FULL = re.search(r"^rc=(\d+)$", full, re.M).group(1)
FP_START = re.search(r"^SRCFINGER_START: ([0-9a-f]{32})$", full, re.M).group(1)
FP_END = re.search(r"^SRCFINGER_END: ([0-9a-f]{32})$", full, re.M).group(1)
assert RC_FULL == "1", "全量 rc 须 1（登记红还在），实得 %s" % RC_FULL
assert FP_START == FP_END, "回归期间 src/tests 被改动 ⇒ 测的不是要提交的那棵树"
live_finger = subprocess.run(
    ["bash", "-c", "find src tests -name '*.py' -not -path '*__pycache__*' -print0 "
                   "| sort -z | xargs -0 md5sum | md5sum | cut -d' ' -f1"],
    cwd=REPO, capture_output=True, text=True).stdout.strip()
assert live_finger == FP_END, "现算树指纹 %s ≠ 回归件止指纹 %s" % (live_finger, FP_END)
ared = re.findall(r"^E\s+AssertionError: .*$", full, re.M)
assert len(ared) == 2 and ared == re.findall(r"^E\s+AssertionError: .*$", base, re.M), \
    "两条登记红不是 2 行，或与基线不逐行相等"
fpair = re.search(r"^E\s+AssertionError: (熔体积不收敛：\S+ vs \S+)$", full, re.M).group(1)
PAT = r"pytest\.mark\.(skip|xfail)"
SKIP_HEAD = int(subprocess.run(["bash", "-c", "git grep -hE '%s' HEAD -- 'tests/*.py' | wc -l" % PAT],
                               cwd=REPO, capture_output=True, text=True).stdout)
SKIP_WORK = int(subprocess.run(["bash", "-c", "grep -rE --include=*.py '%s' tests | wc -l" % PAT],
                               cwd=REPO, capture_output=True, text=True).stdout)
assert SKIP_HEAD == SKIP_WORK, "skip/xfail 标记行数变了（HEAD %d → 现树 %d）⇒ 违反「不得新增 skip」" % (
    SKIP_HEAD, SKIP_WORK)

n_decl = int(re.search(r"共 (\d+) 条读数全部通过", rec).group(1))
assert (len(re.findall(r"^G\d+ PASS", rec, re.M)), n_decl) == (19, 19), "接线回执不是 19/19"
assert "FAIL" not in rec, "回执里有 FAIL 行"
g02 = re.search(r"相同=(\d+)/(\d+) 不同=(\[[^\]]*\])", rec)
assert g02.group(3) == "[]", "缺省档不再逐位相同 ⇒ 这段说明不能照写"
g06 = re.search(r"peak=([\d.]+) n_steps=(\d+) nvox=(\d+)", rec)
g08 = re.search(r"pre=([0-9a-f]{32}) post=([0-9a-f]{32})", rec)
g09 = re.search(r"G09 PASS\s*\n\s*pre=(\d+) post=(\d+)", rec)
g11 = re.search(r"机上 (\d+) 个进程，我的 (\d+) 个", rec)
g18 = re.search(r"G18 PASS\s*\n\s*注释掉后代码行命中 (\d+)", rec)
g19 = re.search(r"G19 PASS\s*\n\s*整份文本=(\d+) 代码行=(\d+)", rec)
for nm, gg in (("G02", g02), ("G06", g06), ("G08", g08), ("G09", g09), ("G11", g11),
               ("G18", g18), ("G19", g19)):
    assert gg, "回执取不到 " + nm
assert int(g11.group(2)) == 0, "回执 G11：卡上有我的进程 ⇒ 与「不动 GPU」冲突"
assert int(g18.group(1)) == 0, "G18 正对照没把命中数打到 0 ⇒ 那条闸门是空转"
NFIELD_FP = str(len(json.load(open(os.path.join(D, "_raw_s18_post_fingerprint.json"),
                                   encoding="utf-8"))["fingerprint"]))

n_collect = int(re.search(r"(\d+) tests? collected", side).group(1))
n_def = len(re.findall(r"^def test_", txtabs(os.path.join(REPO, "tests/test_scan_program.py")), re.M))
assert int(fs.group(2)) == int(bs.group(2)) + n_collect, \
    "passed %s ≠ 基线 %s ＋ 新收集 %s ⇒ 有测试掉出收集网" % (fs.group(2), bs.group(2), n_collect)

# 尺子：与目录里其它 verify 件**逐字节**现比（不写「与 18/21/22/23 相同」这种手敲名单）
rule = os.path.join(D, "am_s7_verify24.py")
rule_md5 = md5abs(rule)
same = sorted(os.path.basename(f).replace("am_s7_verify", "").replace(".py", "")
              for f in os.listdir(D)
              if re.fullmatch(r"am_s7_verify\d+\.py", f) and f != "am_s7_verify24.py"
              and md5abs(os.path.join(D, f)) == rule_md5)
RULER_CMP = "与 " + "／".join(same) + " 现比逐字节相同"
NRULE = HP(r"已有 (\d+) 份档案")
assert int(NRULE) == len(same) + 1, "表头印的份数 %s ≠ 现算 %d ⇒ 两处口径分家" % (NRULE, len(same) + 1)

HEADS = sh("git", "rev-parse", "--short", "HEAD")
assert HEADS != FALSEH, "真假短哈希相同 ⇒ 这条自纠是空转"


def head_of(rel):
    return subprocess.run(["git", "show", "HEAD:" + rel], cwd=REPO, capture_output=True,
                          text=True, check=True).stdout


def specimen_ok(text):
    """假哈希的每处出现必须与真值同段（段＝空行切分）——判据取读数本身，不取措辞。

    与写手 `am_s25_docs.py` 同一口径：本轮首版用「假」字当标记，而计划文档那句写的是"短哈希写错"
    ⇒ 合法标本被读成传染；措辞一改就误报的门不是门。
    """
    ok = bad = 0
    for para in text.split("\n\n"):
        n = para.count(FALSEH)
        if n:
            ok, bad = (ok + n, bad) if HEADS in para else (ok, bad + n)
    return ok, bad


# 过期值检测器**分两面**：历史面（HEAD 版）＝「没传染」的证据；工作树面允许 >0 但每处须自证是标本
HASH_HEAD = sum(head_of(rel).count(FALSEH) for rel in FIVE)
assert HASH_HEAD == 0, "五份文档 HEAD 版已有假短哈希 %d 处 ⇒ 真传染" % HASH_HEAD
_pair = [specimen_ok(open(os.path.join(REPO, rel), encoding="utf-8").read()) for rel in FIVE]
HASH_SPEC = sum(o for o, _b in _pair)
HASH_UNPROVEN = sum(b for _o, b in _pair)
assert HASH_UNPROVEN == 0, "工作树有 %d 处引用假哈希却没与真值同段 ⇒ 传染，拒发这条说明" % HASH_UNPROVEN
# 正对照：抹掉真值 ⇒ 同一个检测器必须逐处变红（否则标本面是空转）
HASH_CTRL = sum(specimen_ok(re.sub(re.escape(HEADS), "X",
                                   open(os.path.join(REPO, rel), encoding="utf-8").read()))[1]
                for rel in FIVE)
assert HASH_CTRL == HASH_SPEC, "正对照不齐：抹掉真值后红 %d 处、标本 %d 处" % (HASH_CTRL, HASH_SPEC)
pairs = re.findall(r"\*\*(\d+)(?:⇒(\d+))?\*\* 条", txtabs(os.path.join(REPO, "docs/项目开发总览.md")))
assert pairs, "总览里取不到自纠计数 ⇒ 拒绝手填"
PREVCNT, CNT = pairs[-1][0], (pairs[-1][1] or str(int(pairs[-1][0]) + 1))
assert int(CNT) == int(PREVCNT) + 1, "自纠计数不是 ＋1：%s→%s" % (PREVCNT, CNT)

# +0 名单**由列表数出来**（"另有四处"这种写法就是下一片的假读数）
CIRCLED = "①②③④⑤⑥⑦⑧⑨"
DEF_PREWRITE = [
    "位点普查没剥注释（整份文本把成对的 %s 处读成 %s 处）" % (g19.group(2), g19.group(1)),
    "正对照在剥注释之后才动手术＝空转",
    "探针 SRCFINGER 未 export，印出 SRC=?（标本已入库 _raw_s18_probe_diff.txt）",
    "sed 克隆把上一片的故事安在本片头上（改成从 git show --name-status HEAD 现取清单）",
    "标本标记取自措辞（拿「假」字当判据）⇒ 计划文档那句写的是\"短哈希写错\"，合法标本被读成传染；"
    "改用\"与真值同段\"的读数判据",
]
DEF_PRECOMMIT = [
    "假哈希检测器未分两面：(u) 段落笔后同一条命令把自己判红 ⇒ 写手拒绝自己的修复"
    "（现分成历史面／工作树面各自判），开发日志据此多落一条 (k) 补记腿",
]
DEF_PREWRITE_S = "、".join("%s %s" % (CIRCLED[i], d) for i, d in enumerate(DEF_PREWRITE))
DEF_PRECOMMIT_S = "、".join("%s %s" % (CIRCLED[len(DEF_PREWRITE) + i], d)
                            for i, d in enumerate(DEF_PRECOMMIT))

DIFFSTAT = sh("git", "diff", "--stat", "--", "src", "tests").splitlines()[-1].strip()
dirty_rows = [l for l in sh("git", "-c", "core.quotePath=false", "status", "--porcelain",
                            "--", "src", "tests").splitlines() if l.strip()]
n_m = sum(1 for l in dirty_rows if l[:2].strip() == "M")
n_a = sum(1 for l in dirty_rows if l[:2].strip() == "??")
assert n_m + n_a == len(dirty_rows), "dirty 两分类没覆盖全部行"
nv = [l.strip() for l in subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                                        capture_output=True, text=True).stdout.splitlines() if l.strip()]
nv_mine = [p for p in nv if "/tmp/amvenv" in
           open("/proc/%s/cmdline" % p, "rb").read().decode("utf-8", "replace")]
assert nv_mine == [], "落笔时我的解释器在卡上留有进程 %s ⇒ 与「不动 GPU」冲突" % nv_mine

MSG = """S7 分片 25：#18 代码腿（路径程序 opt-in 驱动热源）＋分片 24 的落地回执入库；开发日志 §26.27 (u)

代码腿（{DIFFSTAT}；dirty(src+tests)={DIRTY}＝{NMOD} 改／{NADD} 新）：
- 新模块 src/amforge/scan_program.py（{NLSP} 行，md5 {MDSP}）：PathProgram（折点＋逐折点功率）、
  纯 jnp 弧长采样、**比值功率门控** gate=power(t)/laser_power（常功率列逐位＝1.0）、require_program 域校验、
  read_track_table/program_from_track_table（全部列都留、unit 缺省 1.0 不猜单位）、implied_scan_speed 只作诊断量。
  零长段的 sqrt 用双层 where 兜住 ⇒ 前值精确 0、梯度精确 0（该位点已进 #20 的普查名单）。
- thermal_enthalpy.py（{NLTE} 行，md5 {MDE}）接线 5 处（先剥注释再数，逐处 1）＋ chain_schedule(program=None)
  按折线总长定价曝光，并公开警告「hatch_spacing/layer_thickness 对程序档的温度场梯度恒零」。
- 新测试 tests/test_scan_program.py（{NLTS} 行，md5 {MDTS}；{NDEF} 个函数、--collect-only 实收 {NCOLLECT} 条）⇒ 进收集网。

「opt-in 不改缺省」是**两棵源码树之间量出来的**（不是注释声称）：接线前探针 SRC={SPRE}（dirty={DPRE}）
↔ 接线后 SRC={SPOST}（dirty={DPOST}），读数 **{SAME}/{SAMEDEN} 逐字节相同**、ThermalHistory {NFIELD_FP} 个字段指纹相同
（peak={PEAK} K、n_steps={NSTEPS}、熔格 {NVOX}）；机制＝power_gate is None 是 trace 期的 Python 静态判断
⇒ 缺省分支的 jaxpr 里没有那次乘法。接线回执 am_s18_wiring_receipt.py {NREC}/{NREC} 全过
（G18 正对照＝注释掉位点后代码行命中 {G18HIT}；G19 坏口径闸门＝整份 {PAIRALL}／代码行 {PAIRCODE}）。

全量回归（CPU 钉住，启动器自印 CMD/TREE/起止日期/起止指纹）：**{NF} failed, {NPA} passed, {NSI} skipped in {FSEC}s**
＝基线 {BPA} ＋ 新收集 {NCOLLECT}（闭合式成立 ⇒ 没有测试掉出收集网），rc={FRC}，SRCFINGER 起＝止={FFP}（＝现算的提交树指纹），
两条登记红的打印值与基线逐行相等（①{ARED1}；②{FPAIR}）⇒ A0 断言未放宽；skip/xfail 标记行 {SKIPHEAD}→{SKIPWORK}（无新增）。
定向面（自带 CMD）＝{TGFAIL} failed, {TGPASS} passed in {TGSEC}s。性能数字本轮为零（性能只在 GPU 上算）。

分片 24 回执入库（读数引自回执正文区）：远端 main={HEADS}＝本地 HEAD；tree 条目 {ENTRIES}＝blob {NB}＋tree {NTREE}；
闭合式 远端 {NB} == 父提交 {NP} ＋ 新增 {NA} ＝ {EXPECT} -> PASS；双向路径差集 {DL}/{NB} 与 {DR}/{NB}；
全库 SHA 不符 {MS}/{NSHARED}；本片 {SLICE_MATCH}/{NFILES} MATCH；{VERDICT}；四个正对照 A/B/C/D 同运行都能红。
回执 am_s7_verify24.log {R24SIZE} 字节、BODY_OFFSET={BOFF}、cmp -i {BOFF}:0 三联 {CMP0}/{CMP1}/{CMP2}；
尺子 am_s7_verify24.py（md5 前 12={MDRULE}）{RULER_CMP}，表头印的份数与 glob＋md5 现算一致＝{NRULE} 份。

自纠 +1（§26.27 {PREVCNT}→{CNT}）：任务台账 #14 的短哈希是手敲的假值 {FALSEH}，实为 {HEADS}。
传染面**分两面现数**（这是本轮的仪器教训，见下）：历史面＝五份文档的 `git show HEAD:` 版命中 {HASHHEAD} 处
⇒ 没传染到文档；工作树面命中 {HASHSPEC} 处，全部是本轮自纠条目里"与真值同段"的标本引文，
抹掉真值后正对照 {HASHCTRL}/{HASHSPEC} 逐处变红 ⇒ 检测器不是空转。⇒ 新口径「短哈希一律由命令现取，不手敲」。
另有 {NDEFPRE} 处仪器缺陷**拦在写盘前 ⇒ +0**：{DEFPRE}；{NDEFPOST} 处**拦在提交前 ⇒ +0**：{DEFPOST}。
本段说明由同目录的 am_s25_msg.py 从归档件正则现取，一个读数都不手敲（+0 名单也由列表数出来，不写"四处"）。
GPU：回执时卡上 compute 进程 {NVREC} 个、我的解释器所留 {NVME}/{NVREC}；落笔时现取 {NVNOW} 个、我的 {NVMENOW}/{NVNOW}；全程未占卡。
物理侧登记：默认档驻留律在层/道边界瞬移、解析总长只对道间跳距计费 ⇒ 连续折线总长＝解析＋跨层＋对位，
故默认档与程序档的「速度↔时长」换算不可混用（已写进 #10/A3 注意事项）。另立 #41（两处既有缺陷，实测发现、当场不修）。
排期：#27＋#33 → #10/A3；#40 → #39；#29／#30 仍等用户裁决。
""".format(
    DIFFSTAT=DIFFSTAT, DIRTY=str(len(dirty_rows)), NMOD=str(n_m), NADD=str(n_a),
    NLSP=nl("src/amforge/scan_program.py"), MDSP=md5rel("src/amforge/scan_program.py"),
    NLTE=nl("src/amforge/thermal_enthalpy.py"), MDE=md5rel("src/amforge/thermal_enthalpy.py"),
    NLTS=nl("tests/test_scan_program.py"), MDTS=md5rel("tests/test_scan_program.py"),
    NDEF=str(n_def), NCOLLECT=str(n_collect),
    SPRE=g08.group(1), SPOST=g08.group(2), DPRE=g09.group(1), DPOST=g09.group(2),
    SAME=g02.group(1), SAMEDEN=g02.group(2), NFIELD_FP=NFIELD_FP,
    PEAK=g06.group(1), NSTEPS=g06.group(2), NVOX=g06.group(3),
    NREC=str(n_decl), G18HIT=g18.group(1), PAIRALL=g19.group(1), PAIRCODE=g19.group(2),
    NF=fs.group(1), NPA=fs.group(2), NSI=fs.group(3), FSEC=fs.group(4), BPA=bs.group(2),
    FRC=RC_FULL, FFP=FP_END, ARED1=ared[0][2:].strip(), FPAIR=fpair,
    SKIPHEAD=str(SKIP_HEAD), SKIPWORK=str(SKIP_WORK),
    TGFAIL=tg.group(1), TGPASS=tg.group(2), TGSEC=tg.group(3),
    HEADS=HEADS, ENTRIES=B(r"tree 条目 (\d+) = "), NB=str(NB), NTREE=B(r"blob \d+ \+ tree (\d+)"),
    NP=str(NP), NA=str(NA), EXPECT=str(EXPECT),
    DL=B(r"本地独有 (\d+)/"), DR=B(r"远端独有 (\d+)/"), MS=B(r"blob SHA 不符 (\d+)/"),
    NSHARED=B(r"blob SHA 不符 \d+/(\d+)"), SLICE_MATCH=B(r"MATCH (\d+)/"), NFILES=B(r"本片文件 (\d+) 个"),
    VERDICT=VERDICT, R24SIZE=str(os.path.getsize(os.path.join(D, "am_s7_verify24.log"))),
    BOFF=str(boff), CMP0=str(CMP[0]), CMP1=str(CMP[1]), CMP2=str(CMP[2]),
    MDRULE=rule_md5[:12], RULER_CMP=RULER_CMP, NRULE=NRULE,
    NVREC=g11.group(1), NVME=g11.group(2), NVNOW=str(len(nv)), NVMENOW=str(len(nv_mine)),
    HASHHEAD=str(HASH_HEAD), HASHSPEC=str(HASH_SPEC), HASHCTRL=str(HASH_CTRL),
    NDEFPRE=str(len(DEF_PREWRITE)), NDEFPOST=str(len(DEF_PRECOMMIT)),
    DEFPRE=DEF_PREWRITE_S, DEFPOST=DEF_PRECOMMIT_S,
    FALSEH=FALSEH, PREVCNT=PREVCNT, CNT=CNT,
)

assert not re.search(r"[{][A-Z0-9_]+[}]", MSG), "说明里有未替换的花括号字段"
assert "@@" not in MSG and "＠" not in MSG, "说明里有占位符残留"
open("/tmp/am_s25_msg_out.txt", "w", encoding="utf-8").write(MSG)
print(MSG)
print("---- 自检：尺子名单＝%s｜自纠 %s→%s｜假哈希分两面：历史 %s／标本 %s（未自证 %s、抹真值红 %s）｜+0 名单 %s＋%s｜GPU 我的＝%s/%s｜记分闭合 %s＋%s＝%s"
      % (RULER_CMP, PREVCNT, CNT, HASH_HEAD, HASH_SPEC, HASH_UNPROVEN, HASH_CTRL,
         len(DEF_PREWRITE), len(DEF_PRECOMMIT), len(nv_mine), len(nv),
         bs.group(2), n_collect, fs.group(2)))
