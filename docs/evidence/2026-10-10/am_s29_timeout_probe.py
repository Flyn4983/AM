#!/usr/bin/env python3
"""给 `am_s29_hdr_controls.py` 新加的 SUB_TIMEOUT 守卫做**正对照**（坑⑨ 的凭据）。

不重写一份平行实现：用 `ast` 逐条执行控件面模块顶层语句，**一直执行到那行 `BASE_RC, ... = run_one(text)`
之前为止**（那一行才开始付子进程的钱），于是 `run_one`／`residue`／`ENV`／`SUB_TIMEOUT` 全是控件面
落地那一份的代码文本。然后把 SUB_TIMEOUT 压到 5s，喂给它一份**故意卡死并自己起一个孙进程**的脚本。

四条必须同时成立，缺一条就是这条守卫没被验过：
① rc 必须等于哨兵 "TIMEOUT"（守卫真的会触发＝红路可达，不是死代码）；
② 孙进程必须**真的消失**（证明按进程组 SIGKILL 有效；只杀直接子进程的写法在这里会留下孤儿）；
③ 副本临时文件必须被清掉（`finally` 在超时路径上也走得到）；
④ "孙进程消失"这个检测器自己要有正对照：先让检测器确认一个**我亲手起的活进程**为"在"，
   再杀掉它确认检测器读回"不在"——否则 ② 的"不在"可能只是检测器瞎了。

运行＝`CUDA_VISIBLE_DEVICES= /tmp/amvenv/bin/python /tmp/ctl_timeout_probe.py`（不 import jax，不碰 GPU）。
"""
import ast
import os
import subprocess
import sys
import time

CTL = sys.argv[1]
src = open(CTL, encoding="utf-8").read()
tree = ast.parse(src)
ns = {"__file__": CTL, "__name__": "__probe__"}

# 逐条执行模块顶层语句，到"第一处调用 run_one 的赋值"为止（含 CASES 之后的 run_one／residue 定义）。
stopped = False
for node in tree.body:
    calls_run_one = (isinstance(node, ast.Assign)
                     and any(isinstance(c, ast.Call) and getattr(c.func, "id", "") == "run_one"
                             for c in ast.walk(node.value)))
    if calls_run_one:
        stopped = True
        break
    exec(compile(ast.Module(body=[node], type_ignores=[]), CTL, "exec"), ns)
assert stopped, "控件面顶层找不到 `… = run_one(...)` 那行 ⇒ 本探针的截断口径不适用"
assert "run_one" in ns and "residue" in ns, "没取到控件面自己的 run_one／residue ⇒ 探针不适用"

HERE, PREFIX = ns["HERE"], ns["PREFIX"]
before = sorted(os.listdir(HERE))

HANG = ('import subprocess, time, sys\n'
        'p = subprocess.Popen(["sleep", "600"])\n'
        'print("PROBE_CHILD=%d" % p.pid, flush=True)\n'
        'time.sleep(600)\n')

REAL_TIMEOUT = ns["SUB_TIMEOUT"]      # 先取走真件里的上限，再把它压小（否则最后那行会打印压小后的值）
DEADLINE = 5
ns["SUB_TIMEOUT"] = DEADLINE          # 函数读的是模块命名空间里的全局 ⇒ 这一行就改得了它
ns["ENV_FAIL_STOP"] = 99


def alive(pid, wait=0.0):
    """检测器口径＝`/proc/PID/cmdline` 里含 `sleep`。

    必须能重试：第一版没有，`subprocess.Popen(["sleep","600"])` 之后**立刻**读 cmdline 拿到的是空串
    （fork 已完成、exec 还没落地），于是把一个活着的进程判成"不在"＝检测器自己瞎（实测到，探针 rc=1）。
    `wait>0` 时在这个上限内重试；`wait=0` 是"现在就报我看到的状态"，用于已确认消失的场合。
    """
    t0 = time.time()
    while True:
        try:
            if b"sleep" in open("/proc/%d/cmdline" % pid, "rb").read():
                return True
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            return False
        if time.time() - t0 >= wait:
            return False
        time.sleep(0.05)


def wait_gone(pid, limit=5.0):
    """在 limit 秒内等检测器判"不在"；等不到就返回 False（＝真有孤儿）。"""
    t0 = time.time()
    while time.time() - t0 < limit:
        if not alive(pid):
            return True
        time.sleep(0.05)
    return not alive(pid)


# ④ 检测器自己的正对照：先证明它能把一个活着的 sleep 判成"在"，再证明杀掉后判成"不在"。
decoy = subprocess.Popen(["sleep", "600"])
assert alive(decoy.pid, wait=2.0), "检测器把一个活着的 sleep 600 判成不在 ⇒ ② 的『消失』没有意义"
decoy.kill()
decoy.wait()
assert not alive(decoy.pid), "检测器把一个已杀掉的进程仍判成在 ⇒ ② 会永远变红（假红）"

rc, err, out = ns["run_one"](HANG)
assert rc == "TIMEOUT", "守卫没触发（rc=%s）⇒ SUB_TIMEOUT 这条是死代码，坑⑨ 的登记不成立" % rc
kid = int([l for l in out.splitlines() if l.startswith("PROBE_CHILD=")][0].split("=")[1])
assert wait_gone(kid), ("孙进程 %d 在 5s 内没消失 ⇒ killpg 没打到进程组，孤儿仍占着资源" % kid)
left = sorted(f for f in os.listdir(HERE) if f.startswith(PREFIX))
assert not left, "超时路径上副本没被删掉：" + "、".join(left)
assert sorted(os.listdir(HERE)) == before, "探针在本目录留下了别的东西"

print("① 守卫触发＝rc==TIMEOUT（探针里压到 %ds，真件里的上限是 %ds）" % (DEADLINE, REAL_TIMEOUT))
print("② 孙进程已随进程组消失（PID 是每次变的 volatile 读数⇒只断言不入 stdout；"
      "检测器正对照＝活的判在、杀的判不在，两次都实测）")
print("③ 副本已删（本目录文件数前后相同这条只断言不入 stdout：它会随归档件自己移动 ⇒ volatile）")
print("TIMEOUT_GUARD_VERDICT = GUARD FIRES AND KILLS THE GROUP")
