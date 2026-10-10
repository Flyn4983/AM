#!/usr/bin/env python3
"""审计件 `am_a3_hdr_audit.py` 的**正对照面**：每条扰动都必须让它变红，并命中**指定的闸门句**。

扰动改的是**输入面**（表头的 zctl_ 副本；回执按「表头＋原正文」重建，好让 `cat` 复现式保持绿），
所以红的来源只能是判据本身，而不是"改坏了别的东西"。运行的是**同一件审计器**的源码副本，
落在**它自己的目录**（本件用 `__file__` 反推仓库根；副本逃到 /tmp ⇒ git 在 `/` 上跑＝环境错，
上一片的生成器正对照就是这么把八条"假红"当成通过，见 §26.28 记过的那条）。

两条判读纪律（都是本轮实测到的坑，不是推想）：
① 只看 rc≠0 不作数 ⇒ 必须**同时**命中指定的闸门句；并且先跑一条**未扰动基线必须 rc=0**（C0），
   否则所有红都可能是环境错。
② 每条扰动的锚点在源文本里必须**恰好命中 1 处**，且扰动前后字节必须不同（否则那条对照是空转）。
   扰动一律**保长**（数字换数字）⇒ 表头字节数、`／9400 字节`、mtime 之类相邻声称不跟着变，
   红项就只剩被指的那一条闸门（C6 是唯一的例外：它**添加**一句，用来单测跨面归属门）。

十条扰动清单（每条对应审计器里一类出处／一项声称）：
  C1 十进制普查（无出处数字）      C2 原始件 md5 声称      C3 主面 PASS 行数声称
  C4 记分分解式的道次数            C5 记分分解式的闸门枚举  C6 跨面读数无出处词
  C7 十六进制普查（无出处 hex）    C8 正文 mtime 声称      C9 控件面"不锚 [FAIL]"行数
  C10 控件面 PASS 行数声称
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
INSTR = os.path.join(HERE, "am_a3_hdr_audit.py")
FACES = {
    "主面": dict(hdr="am_a3_preflight_hdr.txt", log="am_a3_preflight.log",
                 raw="_raw_a3_preflight.out"),
    "控件面": dict(hdr="am_a3_preflight_controls_hdr.txt", log="am_a3_preflight_controls.log",
                   raw="_raw_a3_preflight_controls.out"),
}
PRE = "zctl_a3_"

CASES = [
    # (名称, 面, [(锚点正则, 替换)], 期望命中的闸门句)
    ("C1 表头里造一个无出处的十进制读数（1.331e-02→1.337e-02，337 哪面都没有）",
     "主面", [(r"1\.331e-02", "1.337e-02")], "无出处: 句"),
    ("C2 正文原始件的 md5 前 12 过期（末位 4→5）",
     "主面", [(r"md5 前 12＝8b272dceabd4", "md5 前 12＝8b272dceabd5")],
     "原始件 md5 或字节数与实测不符"),
    ("C3 主面 PASS 行数声称与正文对不上（37→36）",
     "主面", [(r"PASS=37", "PASS=36")], "行首 [PASS] 行数 表头声称 36 而实测 37"),
    ("C4 记分分解式的道次数被改（正文 tracks=[1, 2, 3] 仍是 3）",
     "主面", [(r"3 个道次", "4 个道次")], "道次数不符"),
    ("C5 记分分解式声称 11 条闸门、括号里只枚举 10 条（删 J6）",
     "主面", [(r" G6 G7 J6 J11", " G6 G7 J11")], "闸门枚举现数 10 条"),
    ("C6 句末添一条**只有另一面正文**才有的读数，且不写出处词",
     "主面", [(r"\Z", "\n补记：另一面里的那个网格和 0.195760 也顺手记在这里。\n")],
     "跨面无归属"),
    ("C7 表头里的空 stderr md5 改一位（不进任何 md5 池）",
     "主面", [(r"md5 d41d8cd98f00", "md5 d41d8cd98f01")], "无出处 hex"),
    ("C8 正文运行 mtime 声称改一秒（cp -p 保下来的实测时刻不动）",
     "主面", [(r"mtime 11:12:21 由 cp -p 保留", "mtime 11:12:22 由 cp -p 保留")],
     "表头声称 mtime 11:12:22"),
    ("C9 控件面「不锚标签位」的 [FAIL] 行数声称（4→5）",
     "控件面", [(r"数出 \*\*4\*\*", "数出 **5**")],
     "不锚标签位的 [FAIL] 行数 表头声称 5 而实测 4"),
    ("C10 控件面 PASS 行数声称（10→11，正文仍是 10 行）",
     "控件面", [(r"PASS=10", "PASS=11")], "行首 [PASS] 行数 表头声称 11 而实测 10"),
]

text = open(INSTR, encoding="utf-8").read()
ENV = dict(os.environ)
ENV["CUDA_VISIBLE_DEVICES"] = ""
ENV["JAX_PLATFORMS"] = "cpu"


def cleanup():
    for f in os.listdir(HERE):
        if f.startswith(PRE):
            os.unlink(os.path.join(HERE, f))


def run(src_text, tag):
    p = os.path.join(HERE, PRE + tag + ".py")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(src_text)
    try:
        r = subprocess.run([sys.executable, p], cwd=REPO, env=ENV,
                           capture_output=True, text=True)
        return r.returncode, (r.stderr or "") + (r.stdout or "")
    finally:
        os.unlink(p)


def relocate(face, tag):
    """把审计器 faces() 里那一面的表头／回执两个文件名换成 zctl_ 副本（正文原始件保持真件）。
    两个锚点各自必须**恰好命中 1 处**——命中 2 处意味着两面都被换，扰动就不再只作用于目标面。"""
    bad = text
    for key in ("hdr", "log"):
        old = '"%s"' % FACES[face][key]
        new = '"%s%s_%s.txt"' % (PRE, re.sub(r"\W", "", face), key)
        assert bad.count(old) == 1, "%s 的 %s 锚点在审计器源码里命中 %d 处 ⇒ 扰动面不唯一" % (
            face, key, bad.count(old))
        bad = bad.replace(old, new)
    return bad, FACES[face]["hdr"], FACES[face]["log"]


# ---------- C0 未扰动基线：不先证明"好的一面会过"，上面所有红都不作数 ----------
cleanup()
rc0, out0 = run(text, "baseline")
assert rc0 == 0, "未扰动基线 rc=%s ⇒ 审计器本身就没过，本件的十条红全部无效（先修仪器）" % rc0
assert "AUDIT_VERDICT = ALL HEADER READINGS PROVENANCE-PASS" in out0, "基线没印出全过判读行"
assert not [f for f in os.listdir(HERE) if f.startswith(PRE)], "基线副本没被清掉"

rows, allred = [], True
for name, face, subs, expect in CASES:
    src = open(os.path.join(HERE, FACES[face]["hdr"]), encoding="utf-8").read()
    bad_hdr = src
    for pat, rep in subs:
        hits = re.findall(pat, bad_hdr, re.M)
        assert len(hits) == 1, "%s：锚点在表头里命中 %d 处 ⇒ 扰动本身不可信" % (name, len(hits))
        bad_hdr = re.sub(pat, lambda m: rep, bad_hdr, count=1, flags=re.M)
    assert bad_hdr != src, name + "：扰动没改字节 ⇒ 空转"
    instr, _, _ = relocate(face, name)
    hdr_p = os.path.join(HERE, PRE + re.sub(r"\W", "", face) + "_hdr.txt")
    log_p = os.path.join(HERE, PRE + re.sub(r"\W", "", face) + "_log.txt")
    # 回执＝**新表头＋原正文**重建 ⇒ `cat` 复现式不跟着一起红，红项只剩被指的那条闸门
    with open(hdr_p, "w", encoding="utf-8") as fh:
        fh.write(bad_hdr)
    with open(log_p, "wb") as fh:
        fh.write(bad_hdr.encode("utf-8") + open(os.path.join(HERE, FACES[face]["raw"]), "rb").read())
    rc, msg = run(instr, re.sub(r"\W", "", name[:3]))
    # 只数**真红行**（普查汇总行里也含"无出处 0"字样，拿它当红项数＝自造读数）
    reds = [l.strip() for l in msg.splitlines()
            if l.strip().startswith("红:") or "无出处:" in l or l.strip().startswith("跨面无归属:")
            or l.strip().endswith("⇒ 红")]
    ok = rc != 0 and expect in msg
    allred = allred and ok
    rows.append((name, face, rc, ok, expect, len(reds)))
    os.unlink(hdr_p)
    os.unlink(log_p)

assert not [f for f in os.listdir(HERE) if f.startswith(PRE)], "扰动副本／数据没被清掉"

print("正对照面｜被验仪器＝am_a3_hdr_audit.py（同一件源码的副本，跑在同一目录）")
print("C0 未扰动基线 rc=%s 且判读行全过＝True（十条红的对照面；缺这一步每条红都可能是环境错）" % rc0)
print("扰动条数＝%d｜每条判据＝rc≠0 **且**输出含指定闸门句｜锚点唯一性＋字节变化＝件件断言" % len(rows))
for name, face, rc, ok, expect, nred in rows:
    print(("  " if ok else "× ") + "%s｜面＝%s｜rc=%s｜真红行 %d 条（普查汇总行不计）｜期望句「%s」%s" % (
        name, face, rc, nred, expect, "命中" if ok else "**未命中**"))
print("CONTROL_VERDICT =", "ALL TEN PERTURBATIONS GO RED ON THEIR OWN GATE" if allred
      else "PROBLEM（有闸门空转或有红没命中指定句）")
sys.exit(0 if allred else 3)
