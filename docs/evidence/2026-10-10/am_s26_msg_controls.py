#!/usr/bin/env python3
"""分片 26 提交信息仪器的**干跑正对照**：每条探测器都要能红。

做法＝import 真仪器（模块级只有常量与函数定义，`build()` 不在 import 时执行 ⇒ 无副作用），
对**同一把**正则/闸门喂合成坏件 ⇒ 要求拒绝；再喂真档案 ⇒ 要求通过。只判"能不能红"，不写盘、不改档案。
"""
import importlib.util
import re
import sys

MSG_PY = ("/home/shy/桌面/2026年开发-显式冲击有限元-物质点法-sph-fvm+am/AM-Qoder"
          "/docs/evidence/2026-10-10/am_s26_msg.py")
spec = importlib.util.spec_from_file_location("am_s26_msg", MSG_PY)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

raw_txt = M.rd(M.RAW)
base_txt = M.rd(M.BASE)
curl_txt = M.rd(M.CURL)
n_tests = len(re.findall(r"^def test_", M.rd(M.TEST), re.M))
bad = []


def must_reject(name, fn):
    try:
        r = fn()
    except Exception as e:  # noqa: BLE001 — 对照只关心"红不红"
        print(f"[{name}] 坏件 ⇒ 拒绝：{type(e).__name__}: {str(e)[:100]}")
        return
    bad.append(name)
    print(f"[{name}] 坏件 ⇒ **探测器没红**（返回 {str(r)[:60]}）⇒ 该判据是空转")


def must_accept(name, fn):
    r = fn()
    print(f"[{name}] 真件 ⇒ 通过：{str(r)[:100]}")
    return r


# C1 汇总行形状：pytest 换措辞就必须取不到（否则 TALLY 会静默取到旧形状）
must_reject("C1-tally-shape", lambda: M.RE_TALLY.search(raw_txt.replace("286 passed", "286 pass")).group(0))
must_accept("C1-tally-real", lambda: M.RE_TALLY.search(raw_txt).group(0))
# C2 rc 行缺失＝旧片"不写 reg_exit="的缺陷形状
must_reject("C2-rc-missing", lambda: M.RE_RC.search(re.sub(r"^rc=\d+$", "", raw_txt, flags=re.M)).group(1))
must_accept("C2-rc-real", lambda: M.RE_RC.search(raw_txt).group(1))
# C3 红的 E 行少一条 ⇒ parse_raw 必须拒绝（"两红逐字节"的分母不能假）
must_reject("C3-eline-count", lambda: M.parse_raw(
    raw_txt.replace("E       AssertionError: 熔体积不收敛", "E       Note: 熔体积不收敛")))
must_accept("C3-eline-real", lambda: len(M.parse_raw(raw_txt)["elines"]))
# C4 SRCFINGER 起≠止 ⇒「回归期间源码一字未动」不成立
must_reject("C4-finger-diff", lambda: M.parse_raw(
    raw_txt.replace("SRCFINGER_END: 2a49fedf28e08e8d5a68ae23124e391d",
                    "SRCFINGER_END: 2a49fedf28e08e8d5a68ae23124e391e")))
must_accept("C4-finger-real", lambda: M.parse_raw(raw_txt)["fs"] == M.parse_raw(raw_txt)["fe"])
# C5 进度字符敏感性：`.`→`F` 一处 ⇒ 两个计数必须同时变（否则第②路是常数）
p0, p1 = M.parse_raw(raw_txt), M.parse_raw(raw_txt.replace("............F....", "............FF...", 1))
print(f"[C5-progress] 真件 `.`×{p0['dots']} F×{p0['ffs']} s×{p0['sss']} → 扰动 `.`×{p1['dots']} F×{p1['ffs']}"
      f" Σ变化 {(p0['dots'] - p1['dots'], p1['ffs'] - p0['ffs'])}")
if not (p0["dots"] - p1["dots"] == 1 and p1["ffs"] - p0["ffs"] == 1):
    bad.append("C5-progress")
# C6 回执闭合式：父提交 +1 ⇒ gate_curl 必须红
must_reject("C6-closure", lambda: M.gate_curl(curl_txt.replace("闭合式 远端 648 == 父提交 623",
                                                               "闭合式 远端 648 == 父提交 624")))
must_accept("C6-closure-real", lambda: M.gate_curl(curl_txt)["close"])
# C7 单列复核：MATCH 31/31 → 30/31 ⇒ 必须红
must_reject("C7-match", lambda: M.gate_curl(curl_txt.replace("MATCH 31/31", "MATCH 30/31")))
# C8 判读行：ALL CHECKS PASS → PROBLEM ⇒ 必须红
must_reject("C8-verdict", lambda: M.gate_curl(curl_txt.replace("ALL CHECKS PASS", "PROBLEM（见上面非零项）")))
# C9 尺子副本普查：坏 md5 ⇒ 0 份（证明 glob 真在数，不是恒返回常数）
c_real, c_null = M.ruler_copies_of(M.md5(M.V26)), M.ruler_copies_of("0" * 32)
print(f"[C9-ruler] 真 md5 ⇒ {c_real} 份；坏 md5 ⇒ {c_null} 份（须 0，真值须 ≥2）")
if not (c_real >= 2 and c_null == 0):
    bad.append("C9-ruler")
# C10 shortstat：旧格式必须取不到；新格式（合成样本，不读实时暂存区 ⇒ 对照与暂存集无关）必须取到
must_reject("C10-shortstat-old", lambda: M.RE_SHORTSTAT.search("21 files changed, 1685(+)").group(1))
must_accept("C10-shortstat-new", lambda: M.RE_SHORTSTAT.search(
    "22 files changed, 1900 insertions(+), 3 deletions(-)").groups())
# C11 基线代际：把基线 passed 改成 280 ⇒ passed 增量闸门必须红；把基线 E 行改 1 位 ⇒ 相等闸门必须红
must_reject("C11-base-passed", lambda: M.gate_baseline(
    raw_txt, base_txt.replace("2 failed, 281 passed", "2 failed, 280 passed"), n_tests))
must_reject("C11-base-eline", lambda: M.gate_baseline(
    raw_txt, base_txt.replace("2908.771450618092", "2908.771450618093"), n_tests))
must_reject("C11-rc0", lambda: M.gate_baseline(raw_txt.replace("rc=1", "rc=0"), base_txt, n_tests))
must_accept("C11-real", lambda: M.gate_baseline(raw_txt, base_txt, n_tests)[2:])

print("---- 套判 ----")
print(f"n_tests={n_tests}  真件全过＝{not bad}")
print("CONTROL_VERDICT =", "ALL DETECTORS CAN GO RED" if not bad else f"PROBLEM：{bad}")
sys.exit(0 if not bad else 3)
