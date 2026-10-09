"""#18 接线验收件：把"接入求解器之后缺省链逐位不变"这句主张换成可失败的读数。

三条纪律：
* 每个 0 都配分母（48 行里差 0 行 ⇒ 先证明有 48 行可比）；
* 每个"相符"都配一条**能变红**的扰动正对照（改一个数字、翻一个 md5），
  并且正对照与真读数在**同一次运行**里打印；
* 环境读数（GPU 归属、CPU 钉住证据）由命令现取，不转述。
"""
import json
import os
import re
import subprocess

EV = "/home/shy/桌面/2026年开发-显式冲击有限元-物质点法-sph-fvm+am/AM-Qoder" \
     "/docs/evidence/2026-10-09"
SRC = "/home/shy/桌面/2026年开发-显式冲击有限元-物质点法-sph-fvm+am/AM-Qoder" \
      "/src/amforge/thermal_enthalpy.py"

pre_out = open(os.path.join(EV, "_raw_s18_probe.out"), encoding="utf-8").read()
post_out = open(os.path.join(EV, "_raw_s18_probe_post.out"), encoding="utf-8").read()
post_err = open(os.path.join(EV, "_raw_s18_probe_post.err"), encoding="utf-8").read()
pre_fp = json.load(open(os.path.join(EV, "_raw_s18_pre_fingerprint.json"), encoding="utf-8"))
post_fp = json.load(open(os.path.join(EV, "_raw_s18_post_fingerprint.json"), encoding="utf-8"))
src_txt = open(SRC, encoding="utf-8").read()

NR = 0


def gate(label, ok, detail=""):
    global NR
    NR += 1
    print(("G%02d PASS  " % NR) if ok else ("G%02d FAIL  " % NR) + label)
    if detail:
        print("        " + detail)
    assert ok, label


def rows(text):
    return [ln for ln in text.splitlines() if " :: " in ln]


# --- 1. 读数行数（分母先行）------------------------------------------------
a, b = rows(pre_out), rows(post_out)
gate("G1 分母：改前后读数行数都必须是 49 行（少了就没有可比性）",
     len(a) == 49 and len(b) == 49, f"pre={len(a)} post={len(b)}")
ka = {ln.split(" :: ")[0]: ln for ln in a}
kb = {ln.split(" :: ")[0]: ln for ln in b}
same = [k for k in ka if k != "K0 环境" and k in kb and ka[k] == kb[k]]
diff = [k for k in ka if k != "K0 环境" and k in kb and ka[k] != kb[k]]
gate("G2 除 K0（环境行必然不同）之外，48 条读数**逐字节**相同",
     len(same) == 48 and not diff, f"相同={len(same)}/48 不同={diff}")

# --- 2. 扰动正对照：把 post 里一个数字改掉，比较器必须变红 ------------------
tampered = post_out.replace("3015.6814325277965", "3015.681432527796")
ta = {ln.split(" :: ")[0]: ln for ln in rows(pre_out)}
tb = {ln.split(" :: ")[0]: ln for ln in rows(tampered)}
caught = [k for k in ta if k != "K0 环境" and k in tb and ta[k] != tb[k]]
gate("G3 正对照：篡改峰值最后一位 ⇒ 比较器必须至少抓到 1 行（抓到 0 行说明读数瞎了）",
     len(caught) >= 1, f"抓到 {len(caught)} 行：{caught}")

# --- 3. 缺省链指纹：9 个字段全等 -------------------------------------------
fa, fb = pre_fp["fingerprint"], post_fp["fingerprint"]
keys = sorted(fa)
gate("G4 分母：指纹字段数＝9（两侧都要有）",
     len(keys) == 9 and sorted(fb) == keys, f"pre={len(keys)} post={len(sorted(fb))}")
bad_md5 = [k for k in keys
          if (fa[k].get("md5") if isinstance(fa[k], dict) else fa[k])
          != (fb[k].get("md5") if isinstance(fb[k], dict) else fb[k])]
gate("G5 9/9 字段的 md5（或标量 repr）在接线前后逐位相同",
     not bad_md5, f"不符={bad_md5}")
peak = fa["peak_temperature"]["max"]
gate("G6 关键字段现值：peak max 与 n_steps/体素数就是登记串",
     peak == 3015.6814325277965 and pre_fp["n_steps"] == post_fp["n_steps"] == 137
     and pre_fp["nvox"] == post_fp["nvox"] == 343,
     f"peak={peak!r} n_steps={pre_fp['n_steps']} nvox={pre_fp['nvox']}")
flip = json.loads(json.dumps(fb))
flip["peak_temperature"]["md5"] = flip["peak_temperature"]["md5"][:-1] + "0"
bad_flip = [k for k in keys
            if (fa[k].get("md5") if isinstance(fa[k], dict) else fa[k])
            != (flip[k].get("md5") if isinstance(flip[k], dict) else flip[k])]
gate("G7 正对照：翻掉 post 里一个 md5 的末位 ⇒ 字段比对必须变红",
     bad_flip == ["peak_temperature"], f"抓到={bad_flip}")

# --- 4. 两次运行确实是不同的树（否则"逐位相同"可能只是同一次运行的复制）----
s_pre = re.search(r"^K0 环境 :: SRC=([0-9a-f]{32})", pre_out, re.M)
s_post = re.search(r"^K0 环境 :: SRC=([0-9a-f]{32})", post_out, re.M)
gate("G8 K0 的 SRC 指纹两侧都是 32 位十六进制且**互不相同**（改前/改后确为两棵树）",
     bool(s_pre) and bool(s_post) and s_pre.group(1) != s_post.group(1),
     f"pre={s_pre.group(1) if s_pre else '?'} post={s_post.group(1) if s_post else '?'}")
dirty_pre = re.search(r"dirty\(src\+tests\)=(\d+)", pre_out)
dirty_post = re.search(r"dirty\(src\+tests\)=(\d+)", post_out)
gate("G9 正对照的第二半：脏文件计数从 1 涨到 3（新增 1 个模块＋1 个测试，且求解器被改）",
     dirty_pre.group(1) == "1" and dirty_post.group(1) == "3",
     f"pre={dirty_pre.group(1)} post={dirty_post.group(1)}")

# --- 5. CPU 钉住的现场证据（不解释 GPU 数字，只证明我没占它）--------------
gate("G10 post 的 stderr 里有 CPU 回退证据（把 CUDA_VISIBLE_DEVICES 钉空的直接后果）",
     "Falling back to cpu" in post_err, f"stderr 前 80 字={post_err[:80]!r}")
apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                       "--format=csv,noheader"], capture_output=True, text=True)
pids = [ln.split(",")[0].strip() for ln in apps.stdout.splitlines() if ln.strip()]
mine = [pid for pid in pids
        if os.path.exists(f"/proc/{pid}/cmdline")
        and b"/tmp/amvenv" in open(f"/proc/{pid}/cmdline", "rb").read()]
gate("G11 此刻 GPU 上的计算进程里属于我的＝0（分母＝机上进程数；本轮全程未占 GPU）",
     len(mine) == 0, f"机上 {len(pids)} 个进程，我的 {len(mine)} 个：{mine}")

# --- 6. 接线位点普查：三处覆盖 + 缺省分支不乘门控 --------------------------
# 只数**代码行**：注释里也写着这些表达式（那是解释，不是位点），按整份文本匹配会
# 把 3 当成"命中 3 处位点"——正是这类读数最容易自己骗自己的地方。
def code_of(text):
    """先滤掉注释行再返回文本。扰动正对照**也必须走同一条管线**：首版在已滤好的
    ``code_txt`` 上就地注入 ``# …``，注释行没被重新过滤 ⇒ 正则照样命中 ⇒ 控制恒绿。"""
    return "\n".join(ln for ln in text.splitlines() if not ln.strip().startswith("#"))


code_txt = code_of(src_txt)
sites = {
    "opt-in 取值": r'program = p\.get\("scan_program"\)',
    "eager 守卫": r"require_program\(program, reference_power=P\)",
    "总长覆盖": r"path_length = program\.total_length\(\)",
    "位置覆盖": r"positions, _pw_col = program\.sample_steps\(n_steps\)",
    "门控构造": r"power_gate = gate_from_power\(_pw_col, P\)",
    "缺省分支不乘（静态判断成对出现）": r"power_gate is None",
}
for label, pat in sites.items():
    want = 2 if "成对" in label else 1
    n = len(re.findall(pat, code_txt))
    n_all = len(re.findall(pat, src_txt))
    gate(f"接线位点『{label}』在代码行里的命中数＝{want}",
         n == want, f"代码行 {n}/{want}，含注释 {n_all} 处（正则={pat}）")
doctored = code_of(src_txt.replace(
    "        path_length = program.total_length()",
    "        # path_length = program.total_length()"))
gate("正对照：把求解器里的总长覆盖行改成注释 ⇒ 同一管线普查必须变红",
     len(re.findall(r"path_length = program\.total_length\(\)", doctored)) == 0,
     f"注释掉后代码行命中 {len(re.findall(r'path_length = program\\.total_length\\(\\)', doctored))} "
     "⇒ 该读数真的在读代码，不是硬编码 True")
gate("正对照的第二半：连注释一起数就会把 2 处位点读成 3 处（首版的坏口径）",
     len(re.findall(r"power_gate is None", src_txt)) == 3
     and len(re.findall(r"power_gate is None", code_txt)) == 2,
     f"整份文本={len(re.findall('power_gate is None', src_txt))} "
     f"代码行={len(re.findall('power_gate is None', code_txt))}")

print(f"\n共 {NR} 条读数全部通过。")
