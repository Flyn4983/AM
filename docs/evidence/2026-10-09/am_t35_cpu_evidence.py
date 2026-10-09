#!/usr/bin/env python3
"""#35 全量回归的「CPU 真钉住了」证据件（四路实测；写 docs/evidence/2026-10-09/am_t35_cpu_evidence.log）。

要证的事：`am_t35_full.log` 那条 261 项的回归**没有碰 GPU**。
障碍（本轮实测出来的，不是猜的）：全量件全文 grep `cuInit`/`Falling back` 命中 **0** 行。首版 G8 把
"日志正文里有这两行"当判据，于是**假失败**——一次正确的轮次被仪器拒绝。为什么没有？`-s` 只关掉
**文件描述符**的捕获，而 JAX 的平台回退是走 `logging` 出来的（`jax._src.xla_bridge`），仍被 pytest 的
logging 插件收走、且在用例通过时不回显。⇒ 本件把四种启动配置各跑一次，用**同一棵树**、**同一份环境**
（继承自本 shell ＋ 只覆盖 CVD/`JAX_ENABLE_X64`/`PYTHONPATH`，与启动器 `/tmp/am_t35_full_run.sh` 同形）：

  A ＝全量面的**配置复本**（`-q -s`）        → 期望 0 行：这就是全量件里没有那两行的原因（捕获，不是没发生）
  B ＝`-q -s -p no:logging`（摘掉捕获插件）  → 期望 ≥2 行
  C ＝`-q -s --log-cli-level=WARNING`（保留全部插件，只把 log 显示门打开）→ 期望 ≥2 行，且**最接近真实面**
  D ＝进程级 `python -c "import jax"`        → 期望 stderr 里同样两行 ＋ stdout `BACKEND cpu`

自门（缺一条即 rc=2 拒绝写档案）：
  E0 ENV 必须是**继承来**的（首版手挑 6 个变量 ⇒ JAX 走另一分支，两行证据全缺、假失败）。
  E1 CVD ＝空串（不是"未设"——未设会去抢卡，本轮已犯过一次并登记 N9(b)）。
  E2 四路前后指纹一致，且与全量件 SRCFINGER_END 同值 ⇒ 与本轮回归同树、可比。
  E3 A 路 rc＝0（选绿用例，免得设备行混在失败噪声里）。
  E4 A 路标记行 ＝ 0 且 B/C/D ≥ 2 ⇒ "全量件没有那两行"被**解释**而不是被**忽略**。
  E5 逐字节核对：B 与 C 命中的 `cuInit(0) failed: CUDA_ERROR_NO_DEVICE` 行**同一串** ⇒ 不是我的正则造出来的。
"""
import os
import re
import subprocess
import sys

REPO = "/home/shy/桌面/2026年开发-显式冲击有限元-物质点法-sph-fvm+am/AM-Qoder"
OUT = os.path.join(REPO, "docs/evidence/2026-10-09/am_t35_cpu_evidence.log")
FULL = "/tmp/am_t35_full.log"
NODEID = "tests/test_enthalpy_thermal.py::test_scan_recipe_is_grid_independent"
PY = "/tmp/amvenv/bin/python"
FINGER_CMD = ("find src tests -name '*.py' -not -path '*__pycache__*' -print0 "
              "| sort -z | xargs -0 md5sum | md5sum | cut -d' ' -f1")
MARK = ("cuInit", "Falling back")
ENV = dict(os.environ)
ENV.update({"CUDA_VISIBLE_DEVICES": "", "JAX_ENABLE_X64": "1", "PYTHONPATH": "src"})


def sh(cmd):
    return subprocess.run(["bash", "-c", cmd], cwd=REPO, capture_output=True, text=True).stdout.strip()


def run(args, env=None):
    p = subprocess.run([*args], cwd=REPO, env=env or ENV, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def marks(txt):
    return [l for l in txt.splitlines() if any(m in l for m in MARK)]


assert ENV["CUDA_VISIBLE_DEVICES"] == "", "E1 CVD 不是空串 ⇒ 本件本身就会抢卡"
assert len(ENV) >= 20, f"E0 环境只有 {len(ENV)} 个键 ⇒ 不是继承来的，与全量面不可比"

base = [PY, "-m", "pytest", "-q", "-s", "-p", "no:cacheprovider"]
variants = {
    "A 全量面配置复本（-q -s）": base + [NODEID],
    "B 摘掉 logging 插件（-p no:logging）": base + ["-p", "no:logging", NODEID],
    "C 打开 log 实时显示（--log-cli-level=WARNING）": base + ["--log-cli-level=WARNING", NODEID],
}
finger0 = sh(FINGER_CMD)
head = sh("git rev-parse --short HEAD")
dirty = sh("git -c core.quotePath=false status --porcelain -- src tests | wc -l")
d0 = sh("date '+%F %T %z'")
res = {}
for k, args in variants.items():
    rc, body = run(args)
    res[k] = (rc, body, marks(body))
rc_d, body_d = run([PY, "-c", "import jax; print('BACKEND', jax.default_backend())"])
marks_d = marks(body_d)
backend_line = [l for l in body_d.splitlines() if l.startswith("BACKEND")]
d1 = sh("date '+%F %T %z'")
finger1 = sh(FINGER_CMD)
full = open(FULL, encoding="utf-8", errors="replace").read()
fe = re.search(r"^SRCFINGER_END: (\S+)", full, re.M)
full_hits = marks(full)

va, vb, vc = (res[k][0] for k in variants)
ma, mb, mc = (res[k][2] for k in variants)
cu = "cuInit(0) failed: CUDA_ERROR_NO_DEVICE"
fails = []
if not (fe and finger0 == fe.group(1) == finger1):
    fails.append(f"E2 指纹不齐：本件起={finger0} 止={finger1} 全量件 END={fe.group(1) if fe else '缺'}")
if va != 0:
    fails.append(f"E3 A 路 rc={va}（应为 0）")
if len(ma) != 0:
    fails.append(f"E4 A 路标记行 {len(ma)} 条（应为 0）⇒ 全量件那 0 行的解释不成立，需重查")
if not (len(mb) >= 2 and len(mc) >= 2 and len(marks_d) >= 2):
    fails.append(f"E4 B/C/D 标记行不齐：B={len(mb)} C={len(mc)} D={len(marks_d)}")
if not (any(cu in l for l in mb) and any(cu in l for l in mc)):
    fails.append("E5 B/C 两条路没命中同一串 cuInit 标记")
if "BACKEND cpu" not in " ".join(backend_line):
    fails.append(f"E4b D 路没印出 BACKEND cpu（现取＝{backend_line}）")
if fails:
    print("本件自检未过 ⇒ 不写档案：")
    for m in fails:
        print("   -", m)
    for k in variants:
        print(f"----- {k} 正文前 800 字 -----")
        print(res[k][1][:800])
    print("----- D 路正文前 800 字 -----")
    print(body_d[:800])
    raise SystemExit(2)

blocks = "".join(
    f"{'=' * 12} {k} ｜rc＝{res[k][0]} ｜标记行 {len(res[k][2])} 条\n"
    + "".join("M  | " + l + "\n" for l in res[k][2])
    + "--- 该路正文（原样，含 rc=0 的成功行）---\n"
    + res[k][1] + "\n"
    for k in variants)
hdr = f"""===== #35 全量回归的 CPU 钉住证据件（四路实测，同树同环境）=====
CMD A: {' '.join(variants['A 全量面配置复本（-q -s）'])}
CMD B: {' '.join(variants['B 摘掉 logging 插件（-p no:logging）'])}
CMD C: {' '.join(variants['C 打开 log 实时显示（--log-cli-level=WARNING）'])}
CMD D: {PY} -c "import jax; print('BACKEND', jax.default_backend())"
ENV: 继承本 shell 的 {len(ENV)} 个变量 ＋ 只覆盖 CUDA_VISIBLE_DEVICES=（**空串**）／JAX_ENABLE_X64=1／PYTHONPATH=src
     ＝启动器 /tmp/am_t35_full_run.sh 里那条命令的同形环境
TREE: head={head} dirty(src+tests 改动行数)={dirty}
DATE_START: {d0}；DATE_END: {d1}
FINGER(启动器口径)＝起＝止＝{finger0}；与全量件 `am_t35_full.log` 的 SRCFINGER_END **同值** ⇒ 同树可比
标记行读数：A＝{len(ma)}（全量面配置）／B＝{len(mb)}／C＝{len(mc)}／D＝{len(marks_d)}；全量件自身＝{len(full_hits)}
E5 逐字节：B 与 C 都命中同一串 {cu!r} ⇒ 不是正则的产物
D 路 stdout：{' '.join(backend_line)}
NOTE: 本件为什么必须存在（而不是"我在头里写了 ENV"就算）：全量件的 CPU 判据**不能**是"正文里有那两行"，
NOTE:   因为 A 路证明**同样的配置就是取不到**（标记 0 行）——`-s` 只关 fd 捕获，JAX 的回退走 `logging`，
NOTE:   被 pytest 的 logging 插件收走且用例通过时不回显。⇒ 全量件的钉住判据＝① 它自记的 CMD 以
NOTE:   `CUDA_VISIBLE_DEVICES= `（空值）开头 ＋ ② 本件 C 路（保留全部插件、只把显示门打开，最接近真实面）
NOTE:   在同树同环境下现取到那两行 ＋ D 路 `BACKEND cpu`。
NOTE: 支持性一致（**不单独当判据**）：本轮全量件两条红灯的打印值与 CPU 钉住的参照件
NOTE:   `docs/evidence/2026-10-08/am_t34_t36_full_regression.log` 逐字节相同（见全量件 G2）；换后端不太可能
NOTE:   保持 64 位断言值逐字节一致，但这条只作旁证。
NOTE: E0 本件首版的**自身缺陷**：ENV 手挑 6 个变量 ⇒ JAX 的 xla_cuda13 插件在该子环境里走另一分支
NOTE:   （"a CUDA-enabled jaxlib is not installed"），连做那次 `cuInit(0)` 都不做，两行证据全缺、E4 假失败。
NOTE:   ⇒ "同一个启动环境"不是口号：少传一个变量，要取的那两行就会消失。修法＝继承 os.environ ＋ E0 断言。
NOTE: ⚠ 与 N9(b) 的关系：我为对照跑过一条**未带** CVD 的 `python -c "import jax"`，当场报 gpu＝真的建了
NOTE:   CUDA context（进程即刻退出、`nvidia-smi` 现取无我的残留）。⇒ 脚本里有闸门不等于命令带了闸门；
NOTE:   本件的 D 路因此**只跑带空串的那一条**，去掉 CVD 的正对照**不跑**（与「不动 GPU」冲突，如实登记）。
"""
off = len((hdr + "BODY_OFFSET=%020d\n" % 0).encode())
hdr_now = hdr + "BODY_OFFSET=%020d\n" % off
assert len(hdr_now.encode()) == off, "BODY_OFFSET 未收敛"
body_bytes = blocks.encode()
if "--write" in sys.argv:
    open(OUT, "w", encoding="utf-8").write(hdr_now + blocks)
    back = open(OUT, "rb").read()
    got = int(re.search(rb"BODY_OFFSET=(\d+)", back).group(1))
    assert got == len(back) - len(body_bytes), "复核：BODY_OFFSET ≠ 实际正文起点"
    assert back[got:] == body_bytes, "复核：正文区不是原样字节"
    print("写入", OUT, len(back), "bytes，正文区逐字节复核通过")
else:
    print(hdr_now)
    print("（预检模式：自检全过，未写档案。加 --write。）")
