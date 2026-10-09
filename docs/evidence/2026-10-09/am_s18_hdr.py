#!/usr/bin/env python3
"""分片 18 三件落地证据的**表头生成器**：表头每个字段都由命令现取，正文逐字节等于原始输出。

用法：/tmp/amvenv/bin/python docs/evidence/2026-10-09/am_s18_hdr.py
产出：am_s7_verify18.log、am_s7_verify18_dryrun.log、am_s7_verify18_first_attempt.log
自证：①写完后 `got.endswith(body)`；②表头里写死一条可复跑的 `cmp -i 0:<表头字节数>` 复核命令，
      偏移量用**不动点迭代**求（加这一行本身会改变偏移量）；③生成器再实跑一次该 cmp，rc≠0 就抛；
      ④表头不得残留占位。
"""
import hashlib
import os
import re
import subprocess

D = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(D)))


def sh(*args):
    return subprocess.run(args, cwd=REPO, capture_output=True, text=True, check=True).stdout


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


FIELDS = ["生成时刻", "CMD", "ENV", "脚本", "TREE", "GPU", "生成器", "RUN", "判读",
          "四个正对照", "缺陷标本", "正文区", "复核", "用途", "读法警告", "教训"]


def check_no_merged_field(hdr, who):
    """字段粘连闸门：干跑时发现 `cmdnote` 不带换行会把 "…未换体。" 与 "ENV    : …" 拼成一行，
    字段在渲染结果里**消失**（既不报错也不可 grep）。规则：字段标签只允许出现在行首。
    长度上限对这类缺陷无效（拼出来的行可以很短，注文本身可以很长）。"""
    for f in FIELDS:
        for m in re.finditer(rf"{re.escape(f)}\s*:\s", hdr):
            if m.start() != 0 and hdr[m.start() - 1] != "\n":
                ctx = hdr[max(0, m.start() - 40):m.start() + 12].replace("\n", "⏎")
                raise SystemExit(f"{who}: 字段「{f}」出现在行中（疑粘连）⇒ 拒绝归档。上下文：…{ctx}…")
    n = sum(1 for l in hdr.splitlines() if any(re.match(rf"^{re.escape(f)}\s*: ", l) for f in FIELDS))
    print(f"  {who}: 行首字段 {n} 条（模板应含 {len(FIELDS)} 类），粘连检查通过")


if os.environ.get("HDR_SELFTEST") == "1":  # 合成件干跑：一正一负，负例必须被拒
    good = "CMD    : x\n⚠ 注：一句话。\nENV    : y\n复核    : z\n"
    bad = "CMD    : x\n⚠ 注：一句话。ENV    : y\n复核    : z\n"  # 旧缺陷：note 无换行 ⇒ 粘连
    check_no_merged_field(good, "正例(须通过)")
    try:
        check_no_merged_field(bad, "负例(须被拒)")
        raise SystemExit("SELFTEST FAIL：负例没变红 ⇒ 闸门是空转")
    except SystemExit as e:
        if "拒绝归档" not in str(e):
            raise
        print(f"SELFTEST OK：正例通过、负例被拒 ⇒ {str(e)[:24]}…")
    raise SystemExit(0)


head_full = sh("git", "rev-parse", "HEAD").strip()
parent_full = sh("git", "rev-parse", "HEAD^").strip()
dirty_src = sh("git", "status", "--porcelain", "--", "src", "tests").strip()
dirty_doc = sh("git", "status", "--porcelain", "--", "docs", ".workbuddy").strip()
n_src = len(dirty_src.splitlines()) if dirty_src else 0
n_doc = len(dirty_doc.splitlines()) if dirty_doc else 0
when = sh("date", "+%F %T %Z").strip()

COMMON = f"""TREE   : head={head_full[:7]}（父={parent_full[:7]}）；本次是**只读核验＋归档**，
         不跑 face、不改 `src`/`tests`。`git status --porcelain -- src tests`＝**{n_src} 行**；
         `-- docs .workbuddy`＝{n_doc} 行（＝正在归档的本轮文档改动，属预期，非 face 干扰）。
GPU    : 全程未发起任何占用 GPU 的进程（用户指令「继续，还是不动GPU」）；本件**不含**任何
         性能/吞吐论断（性能只在 GPU 上测）。
生成器 : docs/evidence/2026-10-09/am_s18_hdr.py（表头字段由 `git rev-parse`／
         `git status --porcelain`／`date`／`md5sum` 现取，无手写数字）。
"""

SPECS = [
    {
        "out": "am_s7_verify18.log",
        "body_src": "_raw_verify18.out",
        "script": "am_s7_verify18.py",
        "cmd": (f"cd {REPO} && /tmp/amvenv/bin/python /tmp/am_s7_verify18.py "
                "> /tmp/am_verify18b.out 2>/tmp/am_verify18b.err; echo rc=$?"),
        "cmdnote": (f"⚠ 实际执行的是 **/tmp** 下的同名脚本；入库副本 docs/evidence/2026-10-09/"
                    f"am_s7_verify18.py 与 /tmp/am_s7_verify18.py **逐字节相同**"
                    f"（`cmp /tmp/am_s7_verify18.py docs/evidence/2026-10-09/am_s7_verify18.py` "
                    f"rc={subprocess.run(['cmp', '/tmp/am_s7_verify18.py', os.path.join(D, 'am_s7_verify18.py')], cwd=REPO, capture_output=True).returncode}"
                    f"，md5 见下一行脚本行）⇒ 表头 CMD 与正文出自同一次运行，未换体。"),
        "note": """RUN    : 原始 stdout 存为 _raw_verify18.out 后随本件归档；当轮回显 **rc=0**
         （四步核验＋四个正对照全过 ⇒ LANDING_VERDICT=ALL CHECKS PASS）。
         ⚠ 这条 rc 敢写，是因为启动命令**当时就回显了** `echo rc=$?`；全量记分件那种没回显的，
         一律写明"不写 reg_exit="。
判读    : 远端 main == 本地 HEAD；`truncated=False`、**575 blobs == 本地 575**；闭合式
         远端 575 == 父提交 574 ＋ 本片新增 1；双向路径差集 **0/575**；全库共有 **575** 条
         逐文件 blob SHA 不符 **0/575**；本片 **2/2 MATCH**（含 CJK 文档名 `docs/开发日志.md`）。
四个正对照 : A 远端某条 sha 首位改 `f` ⇒ 不符 **1/575**；B 远端删一条路径 ⇒ 本地独有 **1/575**、
         远端独有 **0/574**；C 闭合式右边 +1 ⇒ 判 **FAIL**；D 远端删**本片**一条路径 ⇒
         本片缺失 **1/2**。四个都能变红，那三个 0 才成立。
缺陷标本 : 首版把本片清单写成 `for p, _ in status` ⇒ `p` 取到**状态字母**而非路径，于是
         「远端缺失」＝文件数（假事故）、「SHA 不符」＝**0/2（假零，条件永假）**，而同一次运行的
         逐条 MATCH 行与之矛盾 ⇒ 缺陷是被**两处读数互相冲突**抓出来的，不是被脚本自己。原始错误
         输出见 am_s7_verify18_first_attempt.log；合成件三态干跑见 am_s7_verify18_dryrun.log。
         修法：路径一律取 `row[-1]` ＋ `assert len(paths)==len(set(paths))==len(status)`
         ＋ 三类判定之和必须覆盖清单 ＋ 新增正对照 D。
""",
    },
    {
        "out": "am_s7_verify18_dryrun.log",
        "body_src": "_raw_dryrun.out",
        "script": "am_s7_verify18_dryrun.py",
        "cmd": (f"cd {REPO} && /tmp/amvenv/bin/python docs/evidence/2026-10-09/"
                f"am_s7_verify18_dryrun.py > docs/evidence/2026-10-09/_raw_dryrun.out 2>&1; "
                f"echo rc=$?"),
        "cmdnote": "",
        "note": """用途    : **归档仪器自己的合成件干跑**（规则：给归档件写仪器时先拿合成件跑一遍，
         假绿与假拒一起抓）。三种真实状态（全对／缺一条／SHA 不符一条）必须给出三种不同读数。
判读    : 修好版＝3 种不同读数（对真相敏感）；首版缺陷写法＝**1 种**（恒读 (2,0,2)）⇒
         当场复现"形状完好的假零"是怎么产生的，也证明修好版不会重犯。当轮回显 rc=0。
""",
    },
    {
        "out": "am_s7_verify18_first_attempt.log",
        "body_src": "_raw_verify18_first.out",
        "script": "am_s7_verify18.py",
        "cmd": (f"cd {REPO} && /tmp/amvenv/bin/python /tmp/am_s7_verify18.py "
                "> /tmp/am_verify18.out 2>&1; echo rc=$?   # 修好前的那一次"),
        "cmdnote": "标本件：正文是**缺陷版脚本**对真实仓库那次运行的输出（rc=3、LANDING_VERDICT=PROBLEM）。"
                   "缺陷版字节未入库（脚本随后被修好并入库，故本件表头「脚本 md5」指向**修好后**的字节）；"
                   "缺陷那 3 行原样保留在 am_s7_verify18_dryrun.py 的 `review_buggy()` 里，可复跑。",
        "note": """读法警告：正文「远端缺失 2/2」与「SHA 不符 0/2」**都不可采信**——前者是状态字母
         被当成路径（假事故），后者是条件永假的**假零**。同一次运行里逐条 MATCH 的两行与它矛盾，
         这才是缺陷现形的方式。现行结论见 am_s7_verify18.log。
教训    : 该件的 PROBLEM 结论本身也是**误判**：落地其实早已成功（①②③④四项全真），报错来自仪器
         而非仓库 ⇒ 「仪器拒绝」≠「事情没做成」，先复核仪器再下结论。
""",
    },
]

for sp in SPECS:
    body = open(os.path.join(D, sp["body_src"]), encoding="utf-8").read()
    base = f"""# {sp['out']} ｜ S7 分片 18（补记轮）落地核验与仪器自检
生成时刻: {when}（`date '+%F %T %Z'` 现取）
CMD    : {sp['cmd']}
{(sp['cmdnote'].strip() + chr(10)) if sp['cmdnote'].strip() else ''}ENV    : 无 JAX/pytest 参与；仅 `git`＋GitHub Trees API（HTTPS，只读）＋本地文件读取。
         本件不需要 CPU 钉记（不跑数值 face），故不写 `CUDA_VISIBLE_DEVICES`——
         ⚠ 与"全量记分"表头不同：那里必须实测 `/proc/<pid>/environ`，本件没有长命令。
脚本   : docs/evidence/2026-10-09/{sp['script']}（md5={md5(os.path.join(D, sp['script']))}）
{COMMON}{sp['note']}正文区  : 逐字节等于 {sp['body_src']}；正文区 md5=
         {hashlib.md5(body.encode()).hexdigest()}，行数 {len(body.splitlines())}。归档**只加表头**，
         读数一字未改。
"""
    # 复核行要写死偏移量，而"写这一行"本身改变表头字节数 ⇒ 对长度做不动点迭代
    num, hdr = 0, base
    for _ in range(10):
        line = (f"复核    : 在本目录跑 `cmp -i 0:{num} {sp['body_src']} {sp['out']}`，"
                f"rc 必须＝0（diffutils 3.10 的 `-i` 用**冒号**分隔 SKIP1:SKIP2，写成逗号会被判"
                f"「无效值」而 rc=2——那是**仪器方言错**，不是正文有差）。\n")
        hdr = base + line
        if len(hdr.encode()) == num:
            break
        num = len(hdr.encode())
    else:
        raise SystemExit("复核行长度未收敛，拒绝据此归档")
    assert not re.search(r"@[A-Z0-9]+@", hdr), "表头有未替换占位"
    check_no_merged_field(hdr, sp["out"])
    out = os.path.join(D, sp["out"])
    with open(out, "w", encoding="utf-8") as f:
        f.write(hdr + body)
    got = open(out, "rb").read()
    assert got.endswith(body.encode()), sp["out"] + " 正文区不等于原始件"
    off = len(hdr.encode())
    r = subprocess.run(["cmp", "-i", f"0:{off}", sp["body_src"], sp["out"]],
                       cwd=D, capture_output=True, text=True)
    print(f"{sp['out']}: 表头 {len(hdr.splitlines())} 行／{off} 字节，正文 {len(body.splitlines())} 行，"
          f"整件 {len(got.splitlines())} 行，md5={md5(out)}，"
          f"| 实跑 cmp -i 0:{off} rc={r.returncode}（须＝0）{r.stdout.strip()}{r.stderr.strip()}")
    assert r.returncode == 0, sp["out"] + " cmp 复核失败"
