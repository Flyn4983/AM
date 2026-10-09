#!/usr/bin/env python3
"""分片 18 落地仪器的**合成件干跑**：证明"本片单列复核"这段逻辑对真实状态是敏感的。

背景：首版把清单行写成 `for p, _ in status` ⇒ `p` 取到的是**状态字母**（A/M）而不是路径，
于是「远端缺失」恒等于文件数、「SHA 不符」**恒为 0**（条件 `p in remote` 永假），
这是一个形状完好、但与真相无关的假零。修好后 `p` 取 `row[-1]`。
三种情形必须给出三种不同读数，否则仪器无效。"""


def review(status, remote, local):
    """修好版：路径一律取 row[-1]。"""
    paths = [r[-1] for r in status]
    v = {}
    for (c, _), p in zip(status, paths):
        v[p] = "MISSING" if p not in remote else ("SHA-DIFF" if remote[p] != local.get(p) else "MATCH")
    return (sum(x == "MISSING" for x in v.values()), sum(x == "SHA-DIFF" for x in v.values()),
            sum(x == "MATCH" for x in v.values()), len(paths))


def review_buggy(status, remote, local):
    """首版缺陷写法（原样保留作标本）：`for p, _ in status` ⇒ p 是状态字母。"""
    missing = [p for p, _ in status if p not in remote]
    bad = [p for p, _ in status if p in remote and remote[p] != local.get(p)]
    return (len(missing), len(bad), len(status))


st = [["A", "a.md"], ["M", "docs/b.md"]]
cases = {
    "①全都存在且 SHA 相同（正确实现该得 0/0/2/2）":
        ({"a.md": "1", "docs/b.md": "2"}, {"a.md": "1", "docs/b.md": "2"}),
    "②远端缺一条（该 1/0/1/2）":
        ({"docs/b.md": "2"}, {"a.md": "1", "docs/b.md": "2"}),
    "③一条 SHA 不符（该 0/1/1/2）":
        ({"a.md": "f", "docs/b.md": "2"}, {"a.md": "1", "docs/b.md": "2"}),
}
out = []
for k, (remote, local) in cases.items():
    out.append((k, review(st, remote, local), review_buggy(st, remote, local)))
for k, good, buggy in out:
    print(f"{k:42s} 修好版={good}  首版缺陷版={buggy}")
uniq_buggy = len({b for _, _, b in out})
uniq_good = len({g for _, g, _ in out})
print(f"判读：修好版在三种情形下给出 {uniq_good} 种不同读数（须＝3 ⇒ 仪器对真相敏感）；"
      f"首版缺陷版给出 {uniq_buggy} 种（＝1 ⇒ 恒读 (2,0,2)，即假零的成因）")
print("DRYRUN_VERDICT =", "仪器有效（可分三态，且缺陷标本复现假零）"
      if (uniq_good == 3 and uniq_buggy == 1) else "仪器无效，拒绝据此归档")
