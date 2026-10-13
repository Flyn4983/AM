#!/usr/bin/env python3
"""S7 分片 18 落地四步核验（每步印分母＋带扰动正对照＋本片新增数闭合式）。"""
import json, subprocess, sys, urllib.request

REPO = "/home/shy/桌面/2026年开发-显式冲击有限元-物质点法-sph-fvm+am/AM-Qoder"
API = "https://api.github.com/repos/Flyn4983/AM/git/trees/{sha}?recursive=1"


def git(*args, quote=False):
    pre = ["-c", "core.quotePath=false"] if quote else []
    r = subprocess.run(["git", *pre, *args], cwd=REPO, capture_output=True, text=True, check=True)
    return r.stdout


def blobs_of(rev):
    d = {}
    for line in git("ls-tree", "-r", rev, quote=True).splitlines():
        meta, path = line.split("\t", 1)
        mode, typ, sha = meta.split()
        if typ == "blob":
            d[path] = sha
    return d


head = git("rev-parse", "HEAD").strip()
print(f"① 本地 HEAD = {head}")

lsr = subprocess.run(["git", "ls-remote", "--heads", "origin"], cwd=REPO,
                     capture_output=True, text=True, check=True).stdout
remote_main = [l.split("\t")[0] for l in lsr.splitlines() if l.split("\t")[1] == "refs/heads/main"][0]
print(f"① ls-remote refs/heads/main = {remote_main}  -> {'MATCH' if remote_main == head else 'DIFF'}")

with urllib.request.urlopen(API.format(sha=remote_main), timeout=90) as f:
    tree = json.load(f)
assert tree.get("truncated") is False, "API 树被截断 ⇒ 本次核验无效"
remote, n_tree = {}, 0
for e in tree["tree"]:
    if e["type"] == "blob":
        remote[e["path"]] = e["sha"]
    else:
        n_tree += 1
print(f"② truncated=False，tree 条目 {len(tree['tree'])} = blob {len(remote)} + tree {n_tree}")

local = blobs_of("HEAD")
parent = blobs_of("HEAD^")
print(f"② 本地 blob = {len(local)}，父提交 = {len(parent)}")

# 本片新增/改动清单（name-status，逐条判 A/M）
status = [l.split("\t") for l in git("show", "--name-status", "--format=", quote=True).splitlines() if l.strip()]
added = [p for c, p in status if c == "A"]
modified = [p for c, p in status if c == "M"]
print(f"② 本片文件 {len(status)} 个 = 新增 {len(added)} + 修改 {len(modified)}")
exp = len(parent) + len(added)
print(f"② 闭合式 远端 {len(remote)} == 父提交 {len(parent)} + 新增 {len(added)} = {exp} -> "
      f"{'PASS' if len(remote) == exp else 'FAIL'}")

only_local = sorted(set(local) - set(remote))
only_remote = sorted(set(remote) - set(local))
print(f"③ 路径差集：本地独有 {len(only_local)}/{len(local)}  远端独有 {len(only_remote)}/{len(remote)}")
for p in (only_local + only_remote)[:5]:
    print("   PATH-DIFF:", p)

shared = set(local) & set(remote)
mismatch = [p for p in shared if local[p] != remote[p]]
print(f"④ 全库共有 {len(shared)} 条 ⇒ blob SHA 不符 {len(mismatch)}/{len(shared)}")
for p in mismatch[:5]:
    print("   SHA-DIFF:", p)

# 本片文件单列复核：路径一律取 row[-1]（重命名/复制时前缀字段不止一个）
paths = [row[-1] for row in status]
assert len(paths) == len(set(paths)) == len(status), "本片清单字段数异常"
verdicts = {}
for (c, _), p in zip(status, paths):
    if p not in remote:
        verdicts[p] = "MISSING"
    elif remote[p] != local.get(p):
        verdicts[p] = "SHA-DIFF"
    else:
        verdicts[p] = "MATCH"
missing = [p for p, v in verdicts.items() if v == "MISSING"]
bad = [p for p, v in verdicts.items() if v == "SHA-DIFF"]
print(f"④ 本片 {len(status)} 文件（A {len(added)}／M {len(modified)}）单列复核 ⇒ 远端缺失 {len(missing)}/{len(status)}，"
      f"SHA 不符 {len(bad)}/{len(status)}，MATCH {sum(1 for v in verdicts.values() if v=='MATCH')}/{len(status)}")
assert sum(1 for v in verdicts.values()) == len(status), "三类判定之和未覆盖本片清单"
for c, p in status:
    print(f"   {verdicts[p]}  {c}  {p!r}")

k = sorted(shared)[3]
pert = dict(remote); pert[k] = "f" + pert[k][1:]
mm = len([p for p in shared if local[p] != pert[p]])
print(f"⑤ 正对照 A（远端 {k} 的 sha 首位改 f）⇒ 不符 {mm}/{len(shared)}（必须 ≥1）")
pert2 = dict(remote); pert2.pop(k)
print(f"⑤ 正对照 B（远端删一条路径）⇒ 本地独有 {len(set(local)-set(pert2))}/{len(local)}，"
      f"远端独有 {len(set(pert2)-set(local))}/{len(pert2)}")
pert3 = len(parent) + len(added) + 1
print(f"⑤ 正对照 C（闭合式右边 +1）⇒ {len(remote)} == {pert3} 判 {'PASS' if len(remote) == pert3 else 'FAIL'}（须 FAIL）")
# 正对照 D：本片专用——从远端清单删掉本片的一个路径，单列复核必须报 1/2 缺失
pert4 = dict(remote); pert4.pop(paths[0])
m4 = [p for p in paths if p not in pert4]
b4 = [p for p in paths if p in pert4 and pert4[p] != local.get(p)]
print(f"⑤ 正对照 D（远端删本片一条路径）⇒ 本片缺失 {len(m4)}/{len(paths)}，SHA 不符 {len(b4)}/{len(paths)}"
      f"（缺失数必须＝1，否则单列复核是空转）")

ok = (remote_main == head and len(remote) == exp and not only_local and not only_remote
      and not mismatch and not missing and not bad and mm >= 1
      and len(set(local) - set(pert2)) == 1 and len(set(pert2) - set(local)) == 0
      and len(m4) == 1 and len(b4) == 0)
print("LANDING_VERDICT =", "ALL CHECKS PASS" if ok else "PROBLEM（见上面非零项）")
sys.exit(0 if ok else 3)
