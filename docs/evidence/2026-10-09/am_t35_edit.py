#!/usr/bin/env python3
"""#35 换数：把 `dx>r` 欠分辨文案里的实测百分比**由证据件解析得到**（不手抄），整段替换。

纪律与自检（预检模式即全部执行；`--write` 才落盘）：
1. 待替换的每个数字都 `re` 自 `docs/evidence/2026-10-09/am_t35_dxr_rerun.log`，字段命中数逐个断言＝1；
2. 旧锚点在 src 中出现次数断言＝1（锚点漂移就拒绝写盘）；
3. **换数之后**再用探针那套 needles 反向解析新块 ⇒ 逐字等于日志里的 token（往返自证，不是我抄一遍）；
4. 用户可见的 `warnings.warn` 串里**不得再有任何配方实测数**：断言该串不含 `%` 字符；
5. 正对照（闸门必须能被"错的实现"变红）：同一套检测器跑在**旧块**上必须抓到那 5 个过期数
   ＋旧 warn 串里确有 `%` ⇒ 检测器不是空转。

用法：AM_REPO=… /tmp/amvenv/bin/python /tmp/am_t35_edit.py            # 预检（只打印，不写）
      AM_REPO=… /tmp/amvenv/bin/python /tmp/am_t35_edit.py --write    # 落盘
"""
import os
import re
import sys

REPO = os.environ["AM_REPO"]
LOG = os.path.join(REPO, "docs/evidence/2026-10-09/am_t35_dxr_rerun.log")
SRC = os.path.join(REPO, "src", "amforge", "thermal_enthalpy.py")
log = open(LOG, encoding="utf-8").read()

# ---- 1) 从日志取数（每条断言命中＝1） ----
STALE = ("15.6", "46.7", "52.8", "35.1", "56.9")


def one(pat, label="", flags=re.M):
    m = re.findall(pat, log, flags)
    assert len(m) == 1, f"{label} 命中 {len(m)} 次（应＝1）：{pat}"
    assert isinstance(m[0], str), f"{label} 捕获组不唯一（应恰好 1 组）"
    return m[0]


def nums(pat, n, label):
    m = re.search(pat, log)
    assert m, f"{label} 未命中：{pat}"
    return m.groups()


# ⚠ 该式要用 `%` 运算符填模型名 ⇒ 字面百分号必须写成 `%%`（否则 `%\s` 是不支持的格式字符）。
RE_C2 = (r"\[%-10s\] peak 偏低\s+([\d.]+)%%\s+Ly 偏低\s+([\d.]+)%%\s+计数 vol 偏低\s+([\d.]+)%%"
         r"\s+fv 加权 vol 偏低\s+([\d.]+)%%")
RE_ABS = (r"绝对值 peak ([\d.]+) -> ([\d.]+)K\s+Ly ([\d.]+) -> ([\d.]+)mm\s+vol ([\d.]+) -> ([\d.]+)mm³"
          r"\s+vol_fv ([\d.]+) -> ([\d.]+)mm")
ip, iw, iv, ifv = nums(RE_C2 % "integrated", 4, "C2 integrated")
pp, pw, pv, pfv = nums(RE_C2 % "point", 4, "C2 point")
arows = re.findall(RE_ABS, log)
assert len(arows) == 2, f"绝对值行应 2 条，实得 {len(arows)}"
(ic_pk, if_pk, ic_ly, if_ly, ic_v, if_v, ic_fv, if_fv), (pc_pk, pf_pk, pc_ly, pf_ly, pc_v, pf_v,
                                                         pc_fv, pf_fv) = arows
tree_m = re.search(r"^TREE: head=(\S+) dirty=(\d+)", log, re.M)
assert tree_m, "日志缺 TREE 戳"
date = one(r"^DATE: (\S+ \S+)", label="DATE")
ns = one(r"\[integrated\s*\] dx=\s*12\.5 ns=\s*(\d+)", label="最细档 ns")
V = dict(ip=ip, iw=iw, iv=iv, ifv=ifv, pp=pp, pw=pw, pv=pv, pfv=pfv,
         ic_pk=ic_pk, if_pk=if_pk, ic_ly=ic_ly, if_ly=if_ly, ic_v=ic_v, if_v=if_v,
         pc_pk=pc_pk, pf_pk=pf_pk, pc_ly=pc_ly, pf_ly=pf_ly, pc_v=pc_v, pf_v=pf_v,
         head=tree_m.group(1), dirty=tree_m.group(2), date=date, ns=ns)
print("从证据件解析出的替换值（逐字来自日志，未手抄）：")
for k, v in V.items():
    print(f"    {k:7s} = {v}")

OLD = '''        # 实测（2026-10-07 在 #19 口径下重跑四档，同夹具 1.2×0.6×0.4mm、
        # 600W/0.8m·s⁻¹/r=100µm、integrated 源，
        # docs/evidence/2026-10-07/am_t2_a0_conv_rerun.log）：dx=r 档横向仅 2 个体素跨过
        # 1/e² 光斑，相对最细档 dx=r/8=12.5µm 峰值低 15.6%（2125.6 vs 2518.0K）、
        # 熔宽低 46.7%（Ly 0.300 vs 0.563mm）、熔体积小 52.8%（0.0320 vs 0.0678mm³）；
        # point 源的同对比是 35.1% / 56.9%。⇒ 峰值与形态都不可用于标定。
        # ⚠ 2026-10-06 旧表在此处写的是"峰值只低 1.1%、熔宽低 45%、熔体积小 36%"，那是
        #   #19 之前 `sdf<0` 严格掩膜下的数（该口径连四档峰值本身都带 O(dx) 的域热容误差，
        #   见 §26.16 与任务 #23），已按新口径重测替换。
        # 能量守恒不受影响（Σfrac=1 与 dx 无关），故只警告不报错；但取熔池形态
        # 或峰值做标定/判据时必须 dx≤r/2。
        warnings.warn(
            f"体素 dx={dx_c*1e6:.1f}µm 大于光束半径 r={r_c*1e6:.1f}µm：横向仅约 "
            f"{2*r_c/dx_c:.1f} 个体素跨过光斑，能量守恒但**熔池形态与峰值不可信**"
            f"（实测 dx=r 相对 dx=r/8 档：熔宽偏低 46.7%、熔体积偏低 52.8%）。"
            f"定量熔池形态请取 dx≤r/2。",
            stacklevel=2)'''

NEW = '''        # 实测（{date} 在**提交树** head={head} dirty={dirty} 上重跑同一夹具四档
        # dx=100/50/25/12.5µm × 两个源模型，1.2×0.6×0.4mm、600W/0.8m·s⁻¹、r=100µm、
        # `suggest_n_steps` 定步数（最细档 ns={ns}），
        # docs/evidence/2026-10-09/am_t35_dxr_rerun.log）：dx=r 档横向仅 2 个体素跨过 1/e² 光斑，
        # 相对最细档 dx=r/8=12.5µm 的偏低幅度——
        #   integrated：峰值低 {ip}%（{ic_pk} vs {if_pk}K）、熔宽低 {iw}%（Ly {ic_ly} vs {if_ly}mm）、
        #               熔体积小 {iv}%（{ic_v} vs {if_v}mm³；fv 加权口径 {ifv}%）。
        #   point    ：峰值低 {pp}%（{pc_pk} vs {pf_pk}K）、熔宽低 {pw}%（Ly {pc_ly} vs {pf_ly}mm）、
        #               熔体积小 {pv}%（{pc_v} vs {pf_v}mm³；fv 加权口径 {pfv}%）。
        # ⇒ 两个源的峰值与形态都不可用于标定；point 源对欠分辨**更不敏感**（熔宽 {pw}% vs {iw}%），
        #   但其绝对量级随 #22 的光斑口径整体上移，两源不可混用同一张旧表。
        # ⚠ 本段文案已**过期两次**：2026-10-06 写"峰值只低 1.1%、熔宽低 45%、熔体积小 36%"（那是
        #   #19 之前 `sdf<0` 严格掩膜口径，该口径连四档峰值本身都带 O(dx) 的域热容误差，见 §26.16
        #   与任务 #23）；2026-10-07 换成 15.6%/46.7%/52.8%（point 35.1%/56.9%），出自
        #   `docs/evidence/2026-10-07/am_t2_a0_conv_rerun.log`——该件 TREE 自报 head=af9fc89
        #   **dirty=30**，即那组数**不对应任何提交**、无法在提交树上复现（本轮只登记"基线不可复现"，
        #   不指认差值来自 af9fc89..HEAD 里哪一笔触及 src 的提交；候选集与逐档对照表都在上述
        #   10-09 件的 C6 段）。⇒ 自本行起**用户可见的运行时文本不再嵌配方实测数**：会腐烂的
        #   百分比只留在注释与证据件，警告串只给几何事实＋指向该件。
        # 能量守恒不受影响（Σfrac=1 与 dx 无关），故只警告不报错；但取熔池形态
        # 或峰值做标定/判据时必须 dx≤r/2。
        warnings.warn(
            f"体素 dx={{dx_c*1e6:.1f}}µm 大于光束半径 r={{r_c*1e6:.1f}}µm：横向仅约 "
            f"{{2*r_c/dx_c:.1f}} 个体素跨过 1/e² 光斑。能量守恒（Σfrac=1 与 dx 无关）但"
            f"**熔池形态与峰值不可信**，不得用于标定或形态判据；定量熔池形态请取 dx≤r/2。"
            f"跨 dx 的实测偏低幅度见 docs/evidence/2026-10-09/am_t35_dxr_rerun.log。",
            stacklevel=2)'''.format(**V)

# ---- 分段：CUR=当前实测段 / HIST=⚠ 历史段 / RUN=用户可见 warn 串 ----
def segs(block):
    i = block.index("# ⚠")
    j = block.index("warnings.warn(")
    return block[:i], block[i:j], block[j:]


cur_o, hist_o, run_o = segs(OLD)
cur_n, hist_n, run_n = segs(NEW)

# ---- 2) 锚点唯一 ----
src = open(SRC, encoding="utf-8").read()
print(f"\n旧锚点在 src 中出现次数 = {src.count(OLD)}（必须＝1）")
assert src.count(OLD) == 1, "锚点漂移 ⇒ 拒绝写盘"

# ---- 5) 正对照先跑：检测器在**旧块**上必须变红 ----
hit_old = [s for s in STALE if s in cur_o]
print(f"正对照（检测器 vs 旧块）：旧当前段含过期数 {len(hit_old)}/5 -> {hit_old}")
assert hit_old == list(STALE), "检测器抓不到旧块的过期数 ⇒ 下面的自检是空转"
assert "%" in run_o, "旧 warn 串本应含配方百分比"

# ---- 3) 往返自证：正则反向解析新块，逐字对日志的**全部** token（含绝对值与 fv 口径） ----
# 先把注释块的空白与注释符 `#` 压掉 ⇒ 解析不依赖换行位置、对齐空格或续行的 `#`（三者都可能被后续
# 排版改动；实测：只压空白时续行的 `#` 卡在「）、」与「熔体积小」之间，反向解析命中 0 次）。
flat = re.sub(r"[\s#]+", "", cur_n)
RE_SRC = (r"{name}：峰值低([\d.]+)%（([\d.]+)vs([\d.]+)K）、熔宽低([\d.]+)%（Ly([\d.]+)vs"
          r"([\d.]+)mm）、熔体积小([\d.]+)%（([\d.]+)vs([\d.]+)mm³；fv加权口径([\d.]+)%）")
FIELDS = ("ip", "ic_pk", "if_pk", "iw", "ic_ly", "if_ly", "iv", "ic_v", "if_v", "ifv")
POINT = ("pp", "pc_pk", "pf_pk", "pw", "pc_ly", "pf_ly", "pv", "pc_v", "pf_v", "pfv")
print("\n往返自证（新块反向解析 vs 日志 token，10+10 个字段）：")
for name, keys in (("integrated", FIELDS), ("point", POINT)):
    m = re.findall(RE_SRC.format(name=name), flat)
    assert len(m) == 1, f"{name} 行反向解析命中 {len(m)} 次（应＝1 ⇒ 新块结构不唯一或写坏）"
    for k, v in zip(keys, m[0]):
        print(f"    {name:10s} {k:6s} 新块={v:>8s}  日志={V[k]:>8s}  {'✓' if v == V[k] else '✗'}")
        assert v == V[k], f"{name}/{k} 往返不等：{v} vs {V[k]}"

# ---- 4) 新当前段不得残留旧数；运行时串不得含配方百分比 ----
resid = [s for s in STALE + ("2125.6", "2518.0", "0.563", "0.0320", "0.0678") if s in cur_n]
print(f"\n新当前实测段残留旧数 = {resid}（必须＝空）")
assert not resid
print(f"新 warn 串含 '%' 字符数 = {run_n.count('%')}（必须＝0 ⇒ 用户可见文本无配方实测数）")
assert run_n.count("%") == 0
assert "偏低 " not in run_n
print(f"历史段仍如实留档（含旧数条数）= {sum(1 for s in STALE if s in hist_n)}（应≥2：15.6/46.7/52.8/35.1/56.9 的⚠记录）")
assert sum(1 for s in STALE if s in hist_n) >= 2

print("\n自检全部通过：旧数出清＋往返逐字相等＋运行时无配方数＋检测器有分辨力。")
if "--write" in sys.argv:
    before = len(src.splitlines())
    open(SRC, "w", encoding="utf-8").write(src.replace(OLD, NEW, 1))
    after = len(open(SRC, encoding="utf-8").read().splitlines())
    print(f"已写入 {SRC}（{before} -> {after} 行）")
else:
    print("（预检模式：未写盘。加 --write 落盘。）")
